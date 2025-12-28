---
companies:
- openai
- replicate
date: '2025-03-20T22:51:24.059321Z'
description: '**OpenAI** 在其 API 中推出了三款全新的尖端音频模型，其中包括性能超越 Whisper 的语音转文本模型 **gpt-4o-transcribe**，以及具备“可提示韵律”（promptable
  prosody）功能的文本转语音模型 **gpt-4o-mini-tts**，后者允许用户控制语音的停顿节奏和情感。


  **Agents SDK** 现在已支持音频功能，从而能够构建语音智能体。OpenAI 还更新了轮次检测（turn detection）功能，实现了基于语音内容的实时语音活动检测（VAD）。此外，OpenAI
  的 **o1-pro** 模型已向部分开发者开放，支持视觉和函数调用等高级功能，但计算成本也相应更高。


  社区对这些音频技术的进步表现出极大的热情，目前一项针对 TTS 创作的广播竞赛正在进行中。与此同时，**Kokoro-82M v1.0** 作为一款领先的开源权重
  TTS 模型脱颖而出，并在 Replicate 平台上提供了极具竞争力的价格。'
id: 1209e1d8-59ce-41b1-b34d-571358fbc1d8
models:
- gpt-4o-transcribe
- gpt-4o-mini-tts
- o1-pro
- kokoro-82m
original_slug: ainews-promptable-prosody-sota-asr-and-semantic
people:
- juberti
- sama
- reach_vb
- kevinweil
- omarsar0
title: 可提示的韵律、最先进的 ASR 和语义 VAD：OpenAI 全面升级语音 AI
topics:
- speech-to-text
- text-to-speech
- voice-activity-detection
- prompt-engineering
- real-time-processing
- model-release
- api
- function-calling
- structured-outputs
- model-performance
---

<!-- buttondown-editor-mode: plaintext -->**OpenAI 语音模型就是你所需要的一切。**

> 2025年3月19日至3月20日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**227** 个频道，**4533** 条消息）。预计为你节省阅读时间（以 200wpm 计算）：**386 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

正如一位评论者所说，预测 OpenAI 发布产品的最佳指标是[另一家前沿实验室的发布](https://x.com/alexalbert__/status/1902765482727645667?s=46)。今天 OpenAI 的“碾压”表现拔得头筹，因为它极其广泛地翻新了 OpenAI 的产品线——如果你关心语音，这次的变化与[上周的 Agents 平台重构](https://buttondown.com/ainews/archive/ainews-the-new-openai-agents-platform/)一样彻底。

我们认为 [Justin Uberti 的总结是最好的](https://x.com/juberti/status/1902771172615524791?s=46)：

![image.png](https://assets.buttondown.email/images/25f1163e-d943-4aef-bf74-8f0cdc621b52.png?w=960&fit=max)


但你也应该看看直播：

https://www.youtube.com/watch?v=lXb0L16ISAc

三大亮点分别是：

**OpenAI.fm**，一个演示网站，展示了 4o-mini-tts 中新的可提示韵律（promptable prosody）：


![image.png](https://assets.buttondown.email/images/4467034b-1a6d-460c-9c1f-64c810bb821a.png?w=960&fit=max)


**4o-transcribe**，一个新的（非开源？）ASR 模型，击败了 Whisper 和商业同行：


![image.png](https://assets.buttondown.email/images/003ab624-c3c5-49ec-a0bd-54fdff9f96c3.png?w=960&fit=max)


最后，稍纵即逝但非常重要，[连 **turn detection**（轮次检测）也获得了更新](https://platform.openai.com/docs/api-reference/realtime-client-events/session)，现在 **realtime 语音将利用语音内容（CONTENT）来动态调整 VAD**：


![image.png](https://assets.buttondown.email/images/c3f2553b-609b-4dfe-8854-9f9e1a9a12d9.png?w=960&fit=max)


[博客文章](https://openai.com/index/introducing-our-next-generation-audio-models/)中的技术细节当然很少，每个要点只有一段话。


![image.png](https://assets.buttondown.email/images/a3d9fdb3-dcfb-4814-aa41-7ac4a9eb0b4e.png?w=960&fit=max)


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

**音频模型、语音转文本和文本转语音的进展**

- **OpenAI 在其 API 中发布了三个新的 SOTA 音频模型**：包括**两个语音转文本模型**（性能超越 **Whisper**），以及**一个新 TTS 模型**（允许你指令它*如何*说话），正如 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1902773579323674710) 所述。**Agents SDK** 现在支持音频，方便构建语音 Agent，[@sama](https://twitter.com/sama/status/1902751101134438471) 进一步讨论了这一点。[@reach_vb](https://twitter.com/reach_vb/status/1902741809010295197) 表达了兴奋，称 **MOAR AUDIO - LETSGOOO!**，显示了社区的热情。你可以在 [@OpenAI](https://twitter.com/OpenAI/status/1902737268852580717) 听到新模型的实际效果。[@kevinweil](https://twitter.com/kevinweil/status/1902769861484335437) 提到新功能让你能够控制**时机和情感**。
- **OpenAI 正在举办 TTS 创作电台竞赛**。根据 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1902773659497885936) 和 [@kevinweil](https://twitter.com/kevinweil/status/1902769865254903888) 的说法，用户可以推特分享他们的作品，有机会赢得一台 Teenage Engineering OB-4，比赛将于周五结束。[@juberti](https://twitter.com/juberti/status/1902771172615524791) 指出，他们增加了具有 SOTA 性能的 **ASR** 模型 **gpt-4o-transcribe**，以及带有 Playground 的 **TTS** 模型 **gpt-4o-mini-tts**。
- **Artificial Analysis 报告称 Kokoro-82M v1.0 现在是领先的开放权重文本转语音模型**，并且价格极具竞争力，在 **Replicate** 上运行每百万字符仅需 **0.63 美元** [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1902762871106441703)。

**模型发布、开源倡议和性能基准测试**

- **OpenAI 的 o1-pro 现已在 API 中向第 1-5 级的特定开发者开放**，根据 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1902485690958450871) 的消息，该模型支持 vision、function calling、Structured Outputs，并可与 Responses 和 Batch APIs 配合使用。该模型消耗更多算力且价格更高：**每 1M input tokens 为 150 美元，每 1M output tokens 为 600 美元**。包括 [@omarsar0](https://twitter.com/omarsar0/status/1902513900064580080) 和 [@BorisMPower](https://twitter.com/BorisMPower/status/1902498485192306866) 在内的多位用户表示对测试 **o1-pro** 感到兴奋。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1902508262676889880) 指出 **o1-pro** 可能会取代一名 **PhD** 或资深软件工程师并节省资金。
- **Nvidia 开源了 Canary 1B & 180M Flash**，根据 [@reach_vb](https://twitter.com/reach_vb/status/1902730989811413250) 的消息，这是采用 CC-BY 许可协议（允许商业使用）的多语言语音识别和翻译模型。
- **Perplexity AI 宣布对其 Sonar 模型进行重大升级**，以更低的成本提供卓越的性能。根据 [@Perplexity_AI](https://twitter.com/Perplexity_ai/status/1902756765843755503) 的消息，基准测试显示 **Sonar Pro 以显著更低的价格超越了甚至最昂贵的竞争对手模型**。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1902759673549521145) 报告称，他们的 **Sonar API** 在 SimpleQA 上得分 91%，同时比 **GPT-4o-mini** 还要便宜。根据 [@Perplexity_AI](https://twitter.com/Perplexity_ai/status/1902756772038725786) 和 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1902760098893795608) 的消息，新增了搜索模式（High, Medium 和 Low），用于自定义性能和价格控制。
- **Reka AI 推出了 Reka Flash 3，一款全新的开源 21B 参数推理模型**，据 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1902487093273862299) 称，该模型在其同尺寸模型中得分最高。该模型的 Artificial Analysis Intelligence Index 为 **47**，表现优于几乎所有非推理模型，并且在 Coding Index 中强于所有非推理模型。该模型足够小，可以在仅有 32GB RAM 的 **MacBook** 上以 8-bit 精度运行。
- **DeepLearningAI** 报告称，**Perplexity** 发布了 **DeepSeek-R1 1776**，这是最初为中国开发的模型更新版本，由于移除了政治审查，在中国境外更加实用 [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1902710466662523099)。

**AI Agents, Frameworks, and Tooling**

- **LangChain 的 graph 使用量正在增加**，据 [@hwchase17](https://twitter.com/hwchase17/status/1902758501736140888) 称，他们正在提升这些 graph 的速度。他们还强调，这一社区努力尝试使用 **LangStack** (LangChain + LangGraph) 来复制 **Manus** [@hwchase17](https://twitter.com/hwchase17/status/1902774800860451116)。
- **Roblox 在 Hugging Face 上发布了 Cube**，这是 Roblox 对 3D Intelligence 的视角 [@_akhaliq](https://twitter.com/_akhaliq/status/1902560381370839524)。
- **Meta 推出了 SWEET-RL，一个新的多轮 LLM agent 基准测试**，以及一种新型的 RL 算法，用于训练具有跨多轮有效信用分配（credit assignment）能力的多轮 LLM agents，根据 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1902594281845428546) 的消息。

**AI in Robotics and Embodied Agents**

- **Figure** 将部署数千个执行小件包裹物流的机器人，每个机器人都拥有独立的神经网络，根据 [@adcock_brett](https://twitter.com/adcock_brett/status/1902739167475609938) 的消息。[@DrJimFan](https://twitter.com/DrJimFan/status/1902767546438148345) 鼓励社区回馈他们的开源 **GR00T N1** 项目。

**LLM-Based Coding Assistants and Tools**

- **Professor Rush** 已进入代码助手领域，根据 [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1902758878170976324) 的消息。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1902743055054029213) 指出 **Cursor** 正开始与他们自己的 **@srush_nlp** 一起构建模型。

**Observations and Opinions**

- **François Chollet** 指出，强泛化能力需要组合性（compositionality）：构建模块化、可重用的抽象，并在面对新情况时即时重新组合 [@fchollet](https://twitter.com/fchollet/status/1902536808098832639)。此外，从第一性原理（first principles）出发思考，而不是对过去进行模式匹配（pattern-matching），能让你提前预判重要的变化 [@fchollet](https://twitter.com/fchollet/status/1902477232029000008)。
- **Karpathy** 描述了一种笔记方法，即将想法追加到单个文本笔记中并定期回顾，他发现这种方法在简单性和有效性之间取得了平衡 [@karpathy](https://twitter.com/karpathy/status/1902503836067229803)。他还探讨了 LLM 维持一个巨型对话与为每个请求开启新对话的影响，讨论了速度、能力和信噪比（signal-to-noise ratio）等注意事项 [@karpathy](https://twitter.com/karpathy/status/1902737525900525657)。
- **Nearcyan** 引入了 **"slop coding"**（垃圾代码编写）一词，用来描述在没有充分 Prompting、设计或验证的情况下让 LLM 编写代码的行为，并强调了其适用的场景非常有限 [@nearcyan](https://twitter.com/nearcyan/status/1902539629313847637)。
- **Swyx** 分享了关于 Agent 工程中时机重要性的分析，并强调 METR 论文是目前公认的前沿自主性（frontier autonomy）标准 [@swyx](https://twitter.com/swyx/status/1902541093943832864)。
- **Tex** 声称中国最大的优势之一是他们的婴儿潮一代（boomers）对学习技术的恐惧程度要低得多 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1902545539725463758)。

**幽默/迷因 (Humor/Memes)**

- **Aidan Mclauglin** 发推讨论了 **GPT-4.5-preview** 最喜欢的 Token [@aidan_mclau](https://twitter.com/aidan_mclau/status/1902754444867224027)，结果显示其具有**明显的**重复性 [@aidan_mclau](https://twitter.com/aidan_mclau/status/1902754166935900218)。
- **Vikhyatk** 戏称自己写了价值 800 万美元的四行代码，并欢迎提问 [@vikhyatk](https://twitter.com/vikhyatk/status/1902541825573085335)。
- **Will Depue** 评论道 **anson yu 是滑铁卢大学的泰勒·斯威夫特（Taylor Swift）** [@willdepue](https://twitter.com/willdepue/status/1902591678738845775)。


---

# AI Reddit 热点回顾

## /r/LocalLlama 回顾

**主题 1. LLM 翻译比 DeepL 便宜 800 倍**

- **LLM 翻译比 DeepL 便宜 800 倍** ([Score: 530, Comments: 162](https://reddit.com/r/LocalLLaMA/comments/1jfh1d7/llms_are_800x_cheaper_for_translation_than_deepl/))：**LLM** 在翻译方面具有显著的成本优势，比 **DeepL** 便宜 **800 多倍**，其中 `gemini-2.0-flash-lite` 的成本低于 **$0.01/hr**，而 **DeepL** 为 **$4.05/hr**。虽然目前的翻译质量可能略低，但作者预计 **LLM** 很快将超越传统模型，并且通过改进 Prompting，它们已经取得了与 **Google** 翻译相当的结果。
  - **LLM vs. 传统模型**：许多用户强调，与传统翻译模型相比，**LLM** 提供了更优越的上下文理解能力，这提升了翻译质量，尤其是对于日语等具有复杂上下文的语言。然而，也有人担心 **LLM** 过于具有创造性或产生幻觉（hallucinating），这可能导致翻译不准确。
  - **模型比较与偏好**：用户讨论了 **Gemma 3**、**CommandR+** 和 **Mistral** 等各种模型，指出了它们在特定语言对或语境中的有效性。一些人因 **DeepL** 能够保持文档结构而在某些任务中更青睐它，而另一些人则发现 **GPT-4o** 和 **Sonnet** 能产生更自然的翻译。
  - **微调（Finetuning）与定制**：微调像 **Gemma 3** 这样的 LLM 是一个热门话题，用户分享了在特定领域或语言对中提高翻译质量的技术和经验。**Finetuning** 被指出能显著提高性能，使 LLM 与 **Google Translate** 等传统模型相比更具竞争力。


**主题 2. 低于 700 美元的预算级 64GB VRAM GPU 服务器**

- **[分享我的配置：低于 700 美元的预算级 64 GB VRAM GPU 服务器](https://www.reddit.com/gallery/1jfnw9x)** ([Score: 521, Comments: 144](https://reddit.com/r/LocalLLaMA/comments/1jfnw9x/sharing_my_build_budget_64_gb_vram_gpu_server/)): 该帖子描述了一个低于 **700 美元**、拥有 **64GB VRAM** 的 **预算级 GPU 服务器配置**。帖子正文未提供更多细节或规格。
  - **预算配置详情**：该配置包括 **Supermicro X10DRG-Q** 主板、**2 颗 Intel Xeon E5-2650 v4 CPU** 以及 **4 块 AMD Radeon Pro V340L 16GB GPU**，总计约 **698 美元**。软件方面使用了 **Ubuntu 22.04.5** 和 **ROCm version 6.3.3**，性能指标显示采样时间为 **20250.33 tokens per second**。
  - **GPU 与性能讨论**：**AMD Radeon Pro V340L** GPU 因其理论速度而受到关注，但实际性能问题也被凸显，并与 **M1 Max** 和 **M1 Ultra** 系统进行了对比。提到了使用 **Llama-cpp** 和 **mlc-llm** 来优化 GPU 利用率，其中 **mlc-llm** 允许同时使用所有 GPU 以获得更好的性能。
  - **市场与替代方案**：讨论中还与其他 GPU 进行了对比，例如拥有 **1TB/s memory bandwidth** 且功耗更低的 **Mi50 32GB**。共识是目前预算级 GPU 配置市场面临挑战，虽然 **ROCm** 显卡更便宜，但在性能和软件支持方面存在权衡。


**Theme 3. TikZero: 从文本生成 AI 科学图表**

- **[TikZero - 使用 LLM 从文本描述生成科学图表的新方法](https://i.redd.it/carfu383qtpe1.png)** ([Score: 165, Comments: 31](https://reddit.com/r/LocalLLaMA/comments/1jfm23c/tikzero_new_approach_for_generating_scientific/)): **TikZero** 介绍了一种使用 **Large Language Models (LLMs)** 从文本描述生成科学图表的新方法，这与传统的 **End-to-End Models** 形成对比。图片展示了 TikZero 生成复杂可视化图表的能力，如 3D 等高线图、神经网络图和高斯函数图，证明了其在创建详细科学插图方面的有效性。
  - 批评者认为，TikZero 的方法可能会鼓励在科学背景下的滥用，因为它在没有真实数据的情况下生成图表，可能损害科学诚信。然而，一些人认为 TikZero 在生成初始绘图结构方面具有价值，这些结构可以用实际数据进行细化，突显了其在创建难以手动编程的复杂可视化方面的实用性。
  - **DrCracket** 为 TikZero 的实用性辩护，强调其在为复杂可视化生成可编辑的高级图形程序方面的作用，这些程序很难手动创建，并提到了它在建筑和原理图等领域的关联性。尽管存在对准确性的担忧，但模型的输出允许轻松纠正和改进，为进一步定制提供了基础。
  - 关于模型大小的讨论表明，虽然像 **SmolDocling-256M** 这样的小型模型提供了良好的 OCR 性能，但 TikZero 对代码生成的关注需要更大的模型规模（如目前的 **8B model**）来保持性能。**DrCracket** 提到正在探索更小的模型，但预计会有性能权衡。


**Theme 4. 使用 15B 以下 LLM 模型进行创意写作**

- **[15B 以下模型的创意写作](https://i.redd.it/vd9wm7zyxqpe1.png)** ([Score: 148, Comments: 92](https://reddit.com/r/LocalLLaMA/comments/1jfdfou/creative_writing_under_15b/)): 该帖子讨论了一项评估参数量少于 **15 billion** 的 AI 模型 **创意写作能力** 的实验，使用了 **ollama** 和 **openwebui** 设置。它描述了一个基于十项标准的评分系统，包括 **Grammar & Mechanics**、**Narrative Structure** 以及 **Originality & Creativity**，并参考了一张对比 **Gemini 3B** 和 **Claude 3** 等模型的图表。
  - 几位用户指出，由于分辨率低，很难阅读结果，并请求提供更高分辨率的图像或电子表格，以便更好地理解评分系统和模型对比。**Wandering_By_** 承认了这一点，并在评论中提供了更多细节。
  - 关于 **Gemma3-4b** 等小型模型的有效性存在争论，令人惊讶的是，该模型在创意写作任务中表现优于大型模型，总分最高。一些用户质疑该基准测试的有效性，指出存在评判提示词模糊以及模型可能产生 "purple prose"（辞藻堆砌）等问题。
  - 建议包括使用更具体和不常见的提示词以避免通用的输出，并考虑对推理模型和通用模型进行单独测试。还提到了需要更结构化的评分量表和示例，以增强评估过程。


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. Claude 3.7 退化：广泛的用户担忧**

- **我对 Anthropic 秘密降低 Sonnet 3.7 智能水平的行为感到非常反感。** ([Score: 255, Comments: 124](https://reddit.com/r/ClaudeAI/comments/1jffjrg/im_utterly_disgusted_by_anthropics_covert/)): 用户对 **Anthropic** 处理 **Claude 3.7** 的方式表示不满，理由是出现了显著的性能问题，例如响应不匹配，以及在 Excel 公式中错误地使用 **LEN + SUBSTITUTE** 而非 **COUNTIF** 函数。据报道，这种功能下降始于近期，导致用户对这种被视为秘密降级的行为感到沮丧。
  - 用户报告 **Claude 3.7** 存在 **严重的性能退化**，包括逻辑错误、无法遵循指令以及代码生成错误，而这些问题在之前的版本中并未出现。由于这些问题，许多用户已转回使用 **GPT**，理由是对 Claude 的一致性和可靠性感到担忧。
  - 有推测认为 **Anthropic** 可能正在对其模型进行 **实时 A/B testing** 或 **功能操纵** 实验，这可以解释 Claude 3.7 的异常行为。一些用户认为 Anthropic 正在使用 **用户数据** 进行训练或功能调整，正如其 [博客](https://www.anthropic.com/research/clio) 中讨论的那样。
  - 社区对 Anthropic **缺乏透明度** 的做法表示不满，许多用户对明显的 **降级** 以及为了获得理想结果而需要进行更多 **Prompt 管理** 感到沮丧。用户还担心 **API 使用量** 的增加及其产生的成本，导致一些人考虑切换到其他模型。


- **如果你正在进行 Vibe Coding，请阅读本文。它可能会救你一命！** ([Score: 644, Comments: 192](https://reddit.com/r/ChatGPTCoding/comments/1jfacpu/if_you_are_vibe_coding_read_this_it_might_save_you/)): 该帖子讨论了 **Vibe Coding** 的趋势，强调了大量非编程人员正在创建应用程序和网站，这可能导致错误并带来学习机会。作者建议使用领先的推理模型来审查代码的生产就绪性，重点关注漏洞、安全性和最佳实践，并分享了他们的非编程人员作品集，包括 **The Prompt Index** 等项目以及由 Claude Sonnet 添加的 **AI T-Shirt Design**。
  - 许多评论者批评 **Vibe Coding** 是一种天真的方法，强调了构建稳健且安全的产品必须具备基础软件工程知识。他们认为 AI 生成的代码经常引入问题，且缺乏生产级应用所需的深度，建议非编程人员要么学习编程基础，要么与经验丰富的开发人员合作。
  - 一些参与者讨论了 **AI 工具** 在编程中的有效性，其中一位评论者详细介绍了他们的工作流程，包括深度研究、与 AI 扮演 CTO 角色以及创建详细的项目计划。他们强调了理解项目需求和保持对 AI 生成输出控制的重要性，以避免次优结果；而另一些人则指出 AI 有可能加速早期开发阶段，但强调最终需要更深入的工程实践。
  - **AI 驱动的开发** 被视为一把双刃剑；它可以提高生产力并给管理层留下深刻印象，但许多开发人员仍持怀疑态度。虽然一些人已成功将 AI 集成到他们的编程过程中，但其他人警告不要在不了解底层系统的情况下过度依赖 AI，并指出如果引导不当，AI 可能会生成代码冗余（Code Bloat）和错误。

- **[我的电脑性能不够。有没有拥有高性能电脑的人愿意把我的这个原创角色（OC）变成动漫图片吗？](https://i.redd.it/cvu6n2kd1rpe1.jpeg)** ([Score: 380, Comments: 131](https://reddit.com/r/StableDiffusion/comments/1jfdvg4/i_dont_have_a_computer_powerful_enough_is_there/)): **Anthropic 对 Claude 3.7 的管理**：讨论集中在 **Claude 3.7** 的性能下降上，引发了 AI 社区内的辩论。人们对影响 AI 能力的管理和决策过程表示担忧。
  - 讨论转向使用各种工具进行 **image generation**（图像生成），提到了 [animegenius.live3d.io](https://animegenius.live3d.io/) 等免费资源以及 **img2img** 技术，正如多个分享的图像和链接所示。用户分享了生成的图像，通常幽默地引用了 **Chris Chan** 和 **Sonichu**。
  - 对话包括对争议性互联网人物 **Chris Chan saga** 的引用，并附有更新故事的链接，如 [2024 Business Insider 文章](https://www.businessinsider.com/chris-chan-saga-timeline-incest-charges-arrest-2021-8)。这引发了幽默与批评交织的回应，反映了该事件对互联网文化的影响。
  - 很大一部分评论包含幽默或讽刺内容，用户以轻松的方式分享 **memes** 和 **GIFs**，而一些评论者对将无关个人与 **指控的罪犯** 进行比较表示担忧。


**主题 2. OpenAI 发布 openai.fm 文本转语音（Text-to-Speech）模型**

- **[openai.fm 发布：OpenAI 最新的文本转语音模型](https://i.redd.it/x5udts3covpe1.png)** ([Score: 107, Comments: 22](https://reddit.com/r/OpenAI/comments/1jfu35m/openaifm_released_openais_newest_texttospeech/)): **OpenAI** 推出了一个名为 **openai.fm** 的新文本转语音模型，具有交互式演示界面。用户可以选择不同的语音选项，如 **Alloy**、**Ash** 和 **Coral**，以及 **Calm**（冷静）和 **Dramatic**（戏剧化）等氛围设置，通过示例文本测试模型的能力，并轻松下载或分享音频输出。
  - 用户讨论了演示版中的 **999 字符限制**，认为 **API** 可能会提供更广泛的功能，正如 OpenAI 的 [audio guide](https://platform.openai.com/docs/guides/audio) 中所提到的。
  - 一些用户将 **openai.fm** 与 **Eleven Labs** 的 **elevenreader** 进行了比较，后者是一款以高质量文本转语音能力著称的免费移动应用，包含 **Laurence Olivier** 等配音。
  - 对于 OpenAI 语音质量的反应褒贬不一，有些人觉得与 **Coral Labs** 和 **Sesame Maya** 等其他服务相比平淡无奇，但也有人欣赏这些即插即用语音的 **low latency**（低延迟）和 **intelligence**（智能）。


- **[我让 ChatGPT 生成一张它参加我生日派对的照片，结果生成了这张](https://i.redd.it/7rwd6oitqqpe1.jpeg)** ([Score: 1008, Comments: 241](https://reddit.com/r/ChatGPT/comments/1jfcmej/i_asked_chatgpt_to_create_an_image_of_itself_at/)): 该帖子描述了 **ChatGPT** 为生日派对场景生成的图像，画面中一个金属机器人手持突击步枪，背景是带有巧克力蛋糕、派对宾客和装饰品的庆祝场景。尽管机器人的出现出人意料，但生动的场景包括串灯和派对帽，强调了节日气氛。
  - 用户分享了他们自己生成的具有不同主题的 **ChatGPT-generated images**，其中一些突出了幽默或意想不到的元素，如 **四胞胎** 和他们自己的 **机器人版本**。图像通常包含幽默或超现实元素，如 **steampunk**（蒸汽朋克）设置和 **robogirls**。
  - 讨论包括 **AI 的创作自由** 在图像生成中的体现，例如无法生成准确的文本，导致名字变成了 "RiotGPT" 而不是 "ChatGPT"。对于 AI 对 **安全和派对主题的解读** 存在幽默感，一些用户开玩笑说派对上的 **不安全枪支操作**。
  - 社区进行了轻松的调侃和幽默，评论涉及 AI 生成场景的怪诞和 **异想天开的本质**，包括 **对恐怖电影的引用** 和 **意想不到的派对主题**。


**主题 3. Kitboga 的 AI 机器人大军：针对诈骗者的创意应用**

- **Kitboga 创建了一个 AI 机器人大军来针对电话诈骗者，这非常滑稽** ([Score: 626, Comments: 29](https://reddit.com/r/ChatGPT/comments/1jfdatl/kitboga_created_an_ai_bot_army_to_target_phone/)): **Kitboga** 雇佣了一支 **AI 机器人大军**，向电话诈骗中心拨打海量电话，在浪费诈骗者数小时时间的同时，还创作了极具娱乐性的内容。这种对 AI 的创新应用因其有效性和幽默感而受到赞赏，正如 [YouTube 视频](https://youtu.be/ZDpo_o7dR8c?feature=shared) 中所展示的那样。
  - 评论者强调了 **AI** 被用于正面和负面影响的潜力，**Kitboga** 的做法是一个正面的例子，同时也承认诈骗者也可能采用 AI 来扩大其操作规模。**RyanGosaling** 建议 AI 还可以通过实时识别诈骗来保护潜在受害者。
  - 关于 Kitboga 运营的**成本效益**存在讨论，用户指出虽然在本地运行 AI 会产生费用，但这些费用可以通过在 **YouTube** 和 **Twitch** 等平台上的内容变现收入来抵消。**Navadvisor** 指出，诈骗者在处理虚假电话时会产生更高的成本。
  - 一些用户提出了打击诈骗者更激进的策略，**Vast_Understanding_1** 表示希望 AI 能摧毁诈骗者的电话系统，而 **OverallComplexities** 等人则称赞目前的努力是英雄之举。


- **[Doge The Builder – 他能搞砸吗？](https://v.redd.it/x64ffkvmpwpe1)** ([Score: 183, Comments: 24](https://reddit.com/r/ChatGPT/comments/1jfz5lt/doge_the_builder_can_he_break_it/)): 社区幽默地讨论了一个虚构场景，**Elon Musk** 和一只 **Dogecoin** 柴犬模仿“建筑师巴布”（Bob the Builder），对贪婪和不受约束的资本主义进行了戏谑的批判。该帖子是对 **memecoins** 可能引发的混乱的讽刺性演绎，并展示了一个由 **DOAT (Department of Automated Truth)** 官方授权的 **meme**，并提供了经批准的 YouTube 链接用于重新发布。
  - **AI 令人印象深刻的能力**：评论者对 **AI** 目前的能力表示赞赏，强调了其在创作引人入胜且幽默的内容方面的出色表现。
  - **影响力人物的文化影响**：反思了像 **Elon Musk** 这样的人物如何显著影响文化时代精神，并对财富积累和社会影响的伦理含义持批评态度。
  - **创作过程咨询**：一位用户表现出对创作此类讽刺内容背后过程的兴趣，表示对涉及的技术或创意方法感到好奇。


**主题 4. Vibe Coding：AI 开发的新趋势**

- **[AI 领域的摩尔定律：AI 可执行任务的时长每 7 个月翻一倍](https://i.redd.it/sp0klkj72upe1.png)** ([Score: 117, Comments: 27](https://reddit.com/r/ChatGPT/comments/1jfn2e9/moores_law_for_ai_length_of_task_ais_can_do/)): 该图片直观地展示了这样一种说法：AI 能够处理的任务时长每 **7 个月**翻一倍，任务范围从回答问题到为定制芯片优化代码。**GPT-2**、**GPT-3**、**GPT-3.5** 和 **GPT-4** 等著名的 AI 模型被标记在时间线上，展示了从 2020 年到 2026 年它们不断增强的能力和成功率的变化。
  - **限流与资源管理**：讨论强调了用户对 AI 使用限流的沮丧，这并非由于模型限制，而是由于资源管理。**NVIDIA GPU** 的短缺是一个主要因素，目前的需求超过了供应，影响了 AI 服务的容量。
  - **定价模式与用户影响**：**ChatGPT** 等 AI 服务的定价模式因“灵活且不精确”而受到批评，这影响了经常超出使用限制的高级用户，使他们成为市场上的“亏本先锋（loss leaders）”。建议包括更明确的使用限制和成本透明度，以改善用户体验。
  - **任务时长与 AI 能力**：图中绘制的任务时长存在困惑，澄清表明这些时长是基于人类完成类似任务所需的时间。讨论还指出，像 **GPT-2** 这样的 AI 模型存在局限性，例如在较长的任务中难以保持连贯性。



---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 提供的摘要之摘要的摘要

**主题 1. LLM 定价与市场波动**

- [**OpenAI 的 o1-pro API 价格令开发者震惊**](https://platform.openai.com/docs/models/o1-pro)：**OpenAI** 的新 **o1-pro API** 模型现已向部分开发者开放，其价格高昂，每 **1M input tokens** 为 **150 美元**，每 **1M output tokens** 为 **600 美元**。**OpenRouter** 上的用户对此表示愤怒，认为定价“疯狂”，并质疑这是否是针对 **DeepSeek R1** 等竞争对手的防御性举措，或者是由于缺乏 streaming 的复杂多轮处理所致。
- [**Pear AI 以更低价格挑战 Cursor**](https://www.pear.ai/)：**Cursor Community** Discord 的成员正在强调 **Pear AI** 相较于 **Cursor** 的价格优势，声称 Cursor 变得越来越贵。一位用户表示，如果 Cursor 不改进其 context window 或 Sonnet Max 的定价，他们可能会转向 Pear AI，并指出 *“如果我要为 sonnet max 付费，我不如用 pear，因为价格更便宜”*。
- [**Perplexity 在融资谈判中寻求 180 亿美元估值**](https://www.bloomberg.com/news/articles/2025-03-20/perplexity-in-early-talks-for-funding-at-18-billion-value?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTc0MjQ5MzI4OSwiZXhwIjoxNzQzMDk4MDg5LCJhcnRpY2xlSWQiOiJTVERYV01UMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.GYIVla5ZD3lp70ED36NxSKtCvWFpu8qrEaHIEPydQ9s)：据报道，**Perplexity AI** 正处于 **5 亿至 10 亿美元**的早期融资谈判中，估值达 **180 亿美元**，这可能使其自 12 月以来的估值翻倍。这反映了在 AI 领域竞争日益加剧的情况下，投资者对 **Perplexity** 的 AI 搜索技术充满信心。


**Theme 2. LLM Model Quirks and Fixes**


- [**Gemma 3 在 Hugging Face 上遭遇身份危机**](https://huggingface.co/)：用户报告称，来自 Hugging Face 的 **Gemma 模型**错误地识别为具有 **2B** 或 **7B 参数**的“第一代”模型，即使下载的是 **12B Gemma 3** 也是如此。这种误识别是由 Google 在更新识别代码时的疏忽造成的，虽然不影响模型性能，但会导致用户对模型版本的困惑。
- [**Unsloth 修复 Gemma 3 Float16 激活问题**](https://unsloth.ai/blog/gemma3)：**Unsloth AI** 解决了 **Gemma 3** 在使用 float16 精度时出现的**无限激活（infinite activations）**问题，该问题曾导致在 Colab GPU 上进行 fine-tuning 和 inference 时出现 **NaN 梯度**。修复方案是将中间激活保留在 **bfloat16** 中，并将 layernorm 操作上采样至 **float32**，从而避免为了速度而进行全量 float32 转换，详情见 [Unsloth AI 博客](https://unsloth.ai/blog/gemma3)。
- [**Hugging Face Inference API 遭遇 404 错误**](https://discuss.huggingface.co/t/hf-inference-api-last-few-minutes-returns-the-same-404-exception-to-all-models/146646/20)：用户报告 **Hugging Face Inference API** 出现大范围 **404 错误**，影响了多个应用程序和付费用户。Hugging Face 团队成员承认了该问题并表示已提交调查，这干扰了依赖该 API 的服务。


**Theme 3. Tools and Frameworks Evolve for LLM Development**


- [**UV 成为备受关注的 Python 包管理器**](https://docs.astral.sh/uv/)：**MCP (Glama)** Discord 的开发者们正在推崇 [**uv**](https://docs.astral.sh/uv/)，这是一个用 Rust 编写的快速 Python 包和项目管理器，被认为是 **pip** 和 **conda** 的卓越替代品。**uv** 因其速度和极简的网站而受到赞誉，正在寻求高效依赖管理的 Python 开发者中获得青睐。
- [**Nvidia 的 cuTile 觊觎 Triton 的宝座？**](https://x.com/blelbach/status/1902113767066103949)：**NVIDIA** 发布了 **cuTile**，这是一种用于 CUDA 的新 tile 编程模型，引发了社区关于其是否与 **Triton** 功能重叠的讨论。一些人推测 **cuTile** 可能是“另一个 Nvidia 版的 Triton”，并对 NVIDIA 在跨厂商后端支持方面的承诺表示担忧。
- [**LlamaIndex 与 DeepLearningAI 合作推出 Agentic 工作流课程**](https://t.co/qvqNj7MJbn)：**DeepLearningAI** 与 **LlamaIndex** 合作推出了一门关于使用 **RAG** 构建 Agentic 工作流的短课，重点是自动化信息处理和上下文感知响应。该课程涵盖了解析表单和提取关键字段等实用技能，旨在增强 Agentic 系统的开发。


**Theme 4. Hardware Headaches and Performance Hurdles**

- [**TPUs 在机器学习速度竞赛中碾压 T4s**](https://cdn.discordapp.com/attachments/1179035537529643040/1351993898289070183/image.png?ex=67ddb770&is=67dc65f0&hm=a8e536c09f7ad917858b287b86d2618d4679f2014e4ee1883ffe62bcf0b92587)：正如 **Unsloth AI** Discord 中所强调的，**TPUs** 表现出明显快于 **T4s** 的性能，特别是在 batch size 为 8 时。这一观察结果突显了 **TPUs** 在对速度要求极高的机器学习任务中的计算优势。
- [**LM Studio 多 GPU 性能大幅下降**](https://cdn.discordapp.com/attachments/1153759714082033735/1352003482693144709/lm_studio.txt?ex=67ddc05d&is=67dc6edd&hm=8a089cd63f8a8578770d0536b875188526a2a8229e9adf767da5a8ff38897d32&)：**LM Studio** Discord 的一位用户报告称，在 **LM Studio** 中使用 **CUDA llama.cpp v1.21.0** 时，多 GPU 性能出现严重退化。性能显著下降，导致有建议通过 tensor splitting 配置手动将 **LM Studio** 限制为单 GPU。
- [**Nvidia Blackwell RTX Pro GPU 面临供应链紧缩**](https://www.tomshardware.com/pc-components/gpus/nvidia-blackwell-rtx-pro-with-up-to-96gb-of-vram-even-more-demand-for-the-limited-supply-of-gpus)：根据 **Nous Research AI** Discord 中分享的一篇 **Tom's Hardware** 文章，**Nvidia Blackwell RTX Pro 系列** GPU 预计将面临供应限制。供应问题可能会持续到 **5月/6月**，这可能会影响这些高需求 GPU 的可用性和价格。


**主题 5. AI 伦理、政策与安全辩论**


- [**中国强制要求对所有 AI 生成内容进行标识**](https://www.cac.gov.cn/2025-03/14/c_1743654684782215.htm)：中国将从 **2025年9月1日** 开始执行新规定，要求对 **所有 AI 生成的合成内容** 进行标识。根据中国政府的官方公告，《人工智能生成合成内容标识办法》将要求在 AI 生成的文本、图像、音频、视频和虚拟场景中添加显式和隐式标识。
- [**中国模型对“文革”相关内容进行自我审查**](https://cdn.discordapp.com/attachments/998381918976479273/1352038445069111340/image.png?ex=67dd382c&is=67dbe6ac&hm=a5c413109c60b302e9252036467f20eb90689c5216bca9d9003c63d2efea915f&)：**OpenAI** Discord 的一位用户报告称，某中国 AI 模型在被问及 **文化大革命** 时会删除回复，表现出自我审查。作为证据提供的截图突显了对某些 AI 模型内容限制的担忧。
- [**Sonnet 系列 LLM 暴露出 AI 编程盲点**](https://ezyang.github.io/ai-blindspots/)：**aider** Discord 中分享的一篇博文讨论了在 **LLM**（特别是 **Sonnet 系列**）中观察到的 **AI 编程盲点**。作者建议未来的解决方案可能涉及旨在解决“停止挖掘 (stop digging)”、“黑盒测试 (black box testing)”和“准备性重构 (preparatory refactoring)”等问题的 **Cursor 规则**，这表明人们正在不断努力改进 AI 编程辅助。


---

# 第一部分：Discord 高层级摘要

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Agent Mode 崩溃**：成员们报告 **Agent mode** 宕机了一小时，且 [Status Page](https://status.cursor.sh/) 未及时更新。
   - 有人开玩笑说 *dan percs* 正在处理这个问题，他正 *忙着在 Cursor 里回复用户* 并 *处理缓慢的请求*，这就是为什么他总是保持在线。
- **Dan Perks 征求键盘建议**：Cursor 的 **Dan Perks** 征求了关于 [Keychron 键盘的意见](https://x.com/danperks_/status/1902474398206181398?s=46)，特别是在寻找一款带有 *旋钮 (knobs)* 且 *低平整洁 (low profile and clean)* 的型号。
   - 建议纷至沓来，包括 **Keychron 的矮轴系列**，尽管 Dan 对键帽的美观表示担忧，称 *我不喜欢这些键帽*。
- **Pear AI vs Cursor：价格战？**：几位成员吹捧了使用 [Pear AI](https://www.pear.ai/) 的优势，并声称 [Cursor 现在更贵了](https://www.reddit.com/r/ChatGPTCoding/comments/1jdd0n8/some_of_the_best_ai_ides_for_fullstacker/)。
   - 一位成员声称因为买了多个 Cursor 年费订阅而感到 *心碎 (cooked)*，另一位则表示：*如果 Cursor 改变他们的 Context Window，或者将 Sonnet Max 改为高级版使用，我会留在 Cursor；否则，如果我要为 Sonnet Max 付费，我不如用 Pear，因为那样更便宜*。
- **ASI：人类唯一的希望？**：成员们辩论 **人工超级智能 (ASI)** 是否是下一次进化，声称 *ASI-Singularity (天赐奇点) 必须是唯一的全球解决方案*。
   - 其他人持怀疑态度，一位用户开玩笑说 *性别研究比 ASI 更重要*，声称 *这是让模型成为星际物种的下一步，拥有中性流动的性别，我们可以与来自不同星球的外星人交配，并适应他们的巫术技术*。
- **Pear AI 被指克隆 Continue？**：成员们讨论了围绕 [Pear AI](https://www.pear.ai/) 的争议，有人声称 Pear AI *基本上就是克隆了 Continue*，并且 *只是拿走了别人的工作成果，然后决定把它变成自己的项目*。
   - 其他人则担心该项目是闭源的，认为应该转向其他替代方案，如 [Trae AI](https://www.trae.ai/)。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **TPU 在速度上完爆 T4**：一位成员指出，**TPU** 表现出比 **T4** 快得多的性能，尤其是在使用 Batch Size 为 8 时，正如对比 [截图](https://cdn.discordapp.com/attachments/1179035537529643040/1351993898289070183/image.png?ex=67ddb770&is=67dc65f0&hm=a8e536c09f7ad917858b287b86d2618d4679f2014e4ee1883ffe62bcf0b92587) 所示。
   - 这一观察强调了在机器学习中对于计算密集型任务使用 **TPU** 的优势，因为速度和效率至关重要。
- **梯度累积 (Gradient Accumulation) 故障修复**：最近的一篇博客文章 ([Unsloth Gradient Accumulation fix](https://unsloth.ai/blog/gradient)) 详细介绍并解决了一个与 **Gradient Accumulation** 相关的问题，该问题曾对序列模型的训练、预训练和 Fine-tuning 产生负面影响。
   - 实施的修复方案旨在 *模拟全批次训练的同时减少 VRAM 使用*，并将其优势扩展到 DDP 和多 GPU 配置。
- **Gemma 3 遭遇身份危机**：用户观察到，从 Hugging Face 获取的 **Gemma 模型** 错误地将自己识别为具有 **2B** 或 **7B 参数** 的 *第一代* 模型，尽管它们实际上是 **12B Gemma 3**。
   - 这种误识别是因为 Google 在训练期间没有更新相关的识别代码，尽管模型本身表现出对其身份和能力的认知。
- **Gemma 3 获得 Float16 救生索**：Unsloth 通过 [这条推文](https://x.com/danielhanchen/status/1902396261875249346) 解决了 **Gemma 3** 在 float16 下的 **无限激活 (infinite activations)** 问题，该问题此前导致在 Colab GPU 上进行 Fine-tuning 和推理时出现 **NaN 梯度**。
   - 该解决方案将所有中间激活保持在 **bfloat16** 中，并将 Layernorm 操作上采样至 **float32**，通过避免完全的 float32 转换来规避速度下降，详见 [Unsloth AI 博客](https://unsloth.ai/blog/gemma3)。
- **Gemma 3 需要降级 Triton**：一位用户在 Python 3.12.9 环境下使用 4090 运行 **Gemma 3** 时遇到了与 **Triton** 编译器相关的 *SystemError*。
   - 根据 [此 GitHub issue](https://github.com/triton-lang/triton/issues/5919#issuecomment-2733328584) 的建议，解决方案涉及将 Python 3.11.x 上的 **Triton** 降级到 3.1.0 版本。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Featherless.ai 配置引发困扰**：用户报告了在使用 Aider 时 **Featherless.ai** 的配置问题，特别是关于配置文件位置和 API key 设置；使用 `--verbose` 命令选项有助于排查[设置问题](https://aider.chat/docs/configuration.html)。
   - 一位用户强调 Wiki 应该为 Windows 用户明确主目录，指定为 `C:\Users\YOURUSERNAME`。
- **DeepSeek R1 价格便宜但速度较慢**：虽然 **DeepSeek R1** 成为 Claude Sonnet 的高性价比替代方案，但其较慢的速度以及相对于 GPT-3.7 的性能让部分用户感到失望，即使使用了 [Unsloth 的动态量化 (Dynamic Quantization)](https://unsloth.ai/blog/deepseekr1-dynamic)。
   - 有人指出，完整的非量化版本 **R1** 需要 **1TB RAM**，这使得 **H200 卡** 成为首选；然而，32B 模型仍被认为是家庭使用的最佳选择。
- **OpenAI 的 o1-pro API 定价令人咋舌**：OpenAI 新推出的 **o1-pro API** 因其高昂的定价遭到用户投诉，价格定为 **每 1M input tokens $150** 以及 **每 1M output tokens $600**。
   - 一位用户调侃道，单次文件重构和基准测试就要花费 **$5**，而另一位用户则戏称其为 *fatherless AI*。
- **Aider LLM 编辑能力引发讨论**：有人指出，Aider 从擅长“编辑”代码而非仅仅“生成”代码的 LLM 中获益最多，并引用了来自 [aider.chat](https://aider.chat/docs/leaderboards/) 的图表。
   - [polyglot benchmark](https://aider.chat/2024/12/21/polyglot.html#the-polyglot-benchmark) 采用了来自 Exercism 的 225 个跨多种语言的编程练习，用以衡量 **LLM 编辑能力**。
- **AI 编程盲点聚焦于 Sonnet 系列 LLM**：分享了一篇关于他们在 **LLM**（特别是 **Sonnet 系列**）中注意到的 [AI 编程盲点](https://ezyang.github.io/ai-blindspots/)的博客文章。
   - 作者建议，未来的解决方案可能涉及旨在解决这些问题的 **Cursor rules**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **代理设置挽救了 LM Studio！**：一位用户通过启用代理设置、进行 **Windows 更新**、重置网络并重启电脑，解决了 **LM Studio** 的连接问题。
   - 他们怀疑这是由于硬件不兼容或运营商屏蔽了 **Hugging Face** 导致的。
- **PCIE 带宽对性能提升微乎其微**：一位用户发现 PCIE 带宽几乎不影响推理速度，与 **PCI-e 4.0 x8** 相比，最多只多出 **2 tokens per second (TPS)**。
   - 他们建议优先考虑 GPU 之间的空间，并避免主板连接器溢出。
- **LM Studio 误报 RAM/VRAM？**：一位用户注意到 **LM Studio** 的 RAM 和 VRAM 显示在更改系统设置后不会立即更新，暗示该检查是在安装期间进行的。
   - 尽管报告不准确，他们正在测试是否可以通过禁用防护栏（guardrails）和增加上下文长度，使应用程序超过报告的 **48GB** **VRAM**。
- **Mistral Small 视觉支持仍难以实现**：用户发现 **LM Studio** 上的某些 **Mistral Small 24b 2503** 模型被错误地标记为支持视觉（vision），因为 Unsloth 版本加载时不带视觉功能，而 MLX 版本则加载失败。
   - 一些人怀疑 **Mistral Small** 在 **MLX** 和 **llama.cpp** 上仅限文本，希望未来的 **mlx-vlm** 更新能解决此问题。
- **多 GPU 性能大幅下降**：一位用户报告了在 **LM Studio** 中使用 **CUDA llama.cpp v1.21.0** 时多 GPU 性能显著下降的问题，并分享了性能数据和[日志](https://cdn.discordapp.com/attachments/1153759714082033735/1352003482693144709/lm_studio.txt?ex=67ddc05d&is=67dc6edd&hm=8a089cd63f8a8578770d0536b875188526a2a8229e9adf767da5a8ff38897d32&)。
   - 一位成员建议手动修改 *tensor_split* 属性，以强制 **LM Studio** 仅使用单个 GPU。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Deep Research 界面更新**：用户报告了 Perplexity 的 Deep Research 中出现了新的 **Standard/High 选择器**，并想知道使用 *High* 模式是否存在限制。
   - 团队正致力于在模型层面改进 **sonar-deep-research**。
- **GPT 4.5 玩起了“消失术”**：**GPT 4.5** 从部分用户的下拉菜单中消失了，引发了因成本原因被移除的猜测。
   - 一位用户指出，它在 *rewrite option*（重写选项）下仍然存在。
- **Sonar API 推出全新搜索模式**：**Perplexity AI** 宣布了改进后的 **Sonar 模型**，这些模型在保持性能的同时降低了成本，表现优于开启搜索功能的 **GPT-4o** 等竞争对手，详情见 [博客文章](https://www.perplexity.ai/hub/blog/new-sonar-search-modes-outperform-openai-in-cost-and-performance)。
   - 他们引入了 **High、Medium 和 Low 搜索计算模式**以优化性能和成本控制，并简化了计费结构，改为输入/输出 token 定价配合固定搜索模式定价，取消了 **Sonar Pro** 和 **Sonar Reasoning Pro** 回复中引用 token 的费用。
- **通过命名避免 API Key 混乱**：一位用户请求在 UI 上增加为 API Key 命名的功能，以避免误删生产环境的 Key，并被引导至 [GitHub](https://github.com/ppl-ai/api-discussion/issues) 提交功能请求。
   - 另一位用户确认 API 调用看起来是正确的，并提醒根据 [文档](https://www.perplexity.ai/hub) 考虑 Rate Limits（速率限制）。
- **Perplexity 无法在锁屏界面使用**：用户反映 **Perplexity 无法在锁屏界面运行**，而不像 **ChatGPT** 那样支持，这让社区感到失望。
   - 一些用户注意到，Perplexity 现在使用的来源数量显著减少（**8-16 个，最多可能 25 个**），而以前曾使用 **40+** 个，这影响了搜索深度。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **O1 Pro API 搁置 Chat Completions 支持**：由于其复杂的多轮模型交互，**O1 Pro API** 将仅在 **responses API** 中提供，而不会添加到 Chat Completions 中。
   - 与 O1 Pro 不同，大多数即将推出的 **GPT** 和 **O-series 模型**都将集成到 Chat Completions 中。
- **Sasha Rush 加入 Cursor 负责前沿 RL**：Sasha Rush ([@srush_nlp](https://fxtwitter.com/srush_nlp/status/1902736199636205914)) 已加入 **Cursor**，致力于为真实世界的编程环境开发大规模前沿 **RL 模型**。
   - Rush 乐于讨论 **AI 职位**以及**工业界与学术界的问题**，并计划在博客文章中分享他的决策过程。
- **Nvidia 的 Canary 开源**：**Nvidia** 开源了 **Canary 1B & 180M Flash** ([@reach_vb](https://x.com/reach_vb/status/1902730989811413250))，在 **CC-BY 许可证**下提供多语言语音识别和翻译模型，可用于商业应用。
   - 该模型支持英文、德文、法文和西班牙文。
- **中国将对 AI 内容进行标识**：中国将从 **2025 年 9 月 1 日**起施行《人工智能生成合成内容标识办法》，强制要求对**所有 AI 生成内容**进行标识。
   - 该规定要求在文本、图像、音频、视频和虚拟场景等内容上添加显式和隐式标识；参见 [中国政府官方公告](https://www.cac.gov.cn/2025-03/14/c_1743654684782215.htm)。
- **三星 ByteCraft 将文本转化为游戏**：三星 SAIL Montreal 推出了 **ByteCraft**，这是世界上第一个通过字节生成视频游戏和动画的生成式模型，可将文本提示词转换为可执行文件，详情见其 [论文](https://github.com/SamsungSAILMontreal/ByteCraft/blob/main/paper/ByteCraft.pdf) 和 [代码](https://github.com/SamsungSAILMontreal/ByteCraft)。
   - 该 **7B 模型**可在 [Hugging Face](https://huggingface.co/SamsungSAILMontreal/ByteCraft) 上获取，[博客文章](https://emygervais.github.io/2025/03/15/bytecraft.html?v1) 进一步详细介绍了该项目。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Plus 订阅者请求 Anki 集成**：一位 NotebookLM Plus 用户请求在 NotebookLM 中加入 **抽认卡生成集成** (Anki)。
   - 然而，社区对此话题并没有太多讨论。
- **“自定义”按钮解决了音频自定义的困惑**：**Audio Overview** 功能中的“Customize”按钮对 NotebookLM 和 NotebookLM Plus 用户均可用，允许用户通过输入 Prompt 来定制音频内容。
   - 免费账户限制每天只能生成 **3 个音频**，因此请谨慎选择你的自定义设置。
- **思维导图功能逐步推出**：用户对 **思维导图 (Mindmap) 功能** 表示期待，有人分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=5hLd3zvdKgg) 展示其交互式用途。
   - 这不是 A/B 测试，而是逐步推出；该功能允许通过选择不同的来源生成多个思维导图，但目前尚不支持编辑思维导图。
- **Audio Overviews 在发音上仍有困难**：用户报告称 **Audio Overviews** 经常读错词汇，即使在 Customize 输入框中使用了音标拼写也是如此。
   - NotebookLM 团队已意识到此问题，并建议在源材料中使用音标拼写作为临时解决方案。
- **扩展程序用户遇到 NotebookLM 页面限制**：用户正在使用 Chrome 扩展程序抓取并添加来自同一域名下链接的源，并指向了 [NotebookLM 的 Chrome 网上应用店](https://chrome.google.com/webstore/search/NotebookLM)。
   - 然而，一名用户在使用此类扩展程序时达到了 **10,000 页** 的上限。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Aphrodite 在性能上碾压 Llama.cpp**：一名成员报告称，使用 Aphrodite Engine 运行 **FP6 Llama-3-2-3b-instruct** 达到了 **70 tokens per second**，并指出在 10GB VRAM 上可以运行多达 4 个 batch 且包含 8192 个 token。
   - 另一名成员赞扬了 Aphrodite Engine 的首席开发人员，并强调该引擎是本地运行的最佳选择之一，同时也承认 Llama.cpp 是兼容性和依赖项方面的标准。
- **LLM 在调试时表现不佳**：成员们观察到，许多模型现在擅长编写无错代码，但在调试现有代码时却很吃力，并指出提供提示（hints）会有所帮助。
   - 该成员对比了他们思考问题的方法，即提供可能的解释和代码片段，这种方法通常会取得成功，除非遇到“非常古怪的东西”。
- **Nvidia 的 Blackwell RTX Pro GPU 面临供应链限制**：一名成员分享了一篇关于 **Nvidia Blackwell RTX Pro 系列** GPU 的 [Tom's Hardware 文章](https://www.tomshardware.com/pc-components/gpus/nvidia-blackwell-rtx-pro-with-up-to-96gb-of-vram-even-more-demand-for-the-limited-supply-of-gpus)，强调了潜在的供应问题。
   - 文章暗示供应可能会在 **5月/6月** 赶上需求，届时可能会有更多以 MSRP（建议零售价）供应的型号。
- **数据集格式 > QwQ 的聊天模板？**：一名成员建议不要**过度关注数据集的格式**，并表示**将数据集放入 QwQ 正确的 Chat Template** 更为重要。
   - 他们补充说，见解可能对数据集是唯一的，并且**推理行为似乎发生在模型层中相对较浅的位置**。
- **有趣的 Logan Kilpatrick 聊天片段**：一名成员分享了 [Logan Kilpatrick 的 YouTube 视频](https://www.youtube.com/watch?v=6y-VEycAjsE&ab_channel=LoganKilpatrick)，称这段对话非常“有趣”。
   - 讨论提到了与 Logan Kilpatrick 视频相关的“有趣聊天”，但未提供更多细节。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **酷酷的 Python 开发者安装 UV 包管理器**：成员们讨论了安装和使用 [**uv**](https://docs.astral.sh/uv/)，这是一个用 Rust 编写的高速 Python 包和项目管理器，用于替代 **pip** 和 **conda**。
   - 它受到青睐是因为*其网站非常极简，只有一个搜索引擎和落地页*。
- **glama.json 认领 GitHub MCP 服务器**：要在 Glama 上认领 GitHub 托管的 MCP 服务器，用户应在仓库根目录添加一个 `glama.json` 文件，并在 `maintainers` 数组中填入其 GitHub 用户名，详情见[此处](https://glama.ai/mcp/servers/1es3d6q5tw)。
   - 该配置需要一个指向 `glama.ai/mcp/schemas/server.json` 的 `$schema` 链接。
- **MCP 应用提升 GitHub API 速率限制**：由于 MCP 服务器数量不断增加，Glama AI 正面临 **GitHub API 速率限制**，但用户可以通过安装 [Glama AI GitHub App](https://github.com/apps/glama-ai) 来提高速率限制。
   - 这样做通过授予应用权限来帮助扩展 Glama。
- **Turso Cloud 与 MCP 集成**：一个新的 MCP 服务器 [mcp-turso-cloud](https://github.com/spences10/mcp-turso-cloud) 将 **Turso 数据库** 与 **LLMs** 集成。
   - 该服务器实现了一个两级身份验证系统，用于直接从 LLMs 管理和查询 Turso 数据库。
- **Unity MCP 将 AI 与文件访问集成**：最先进的 **Unity MCP** [集成](https://github.com/quazaai/UnityMCPIntegration) 现在支持对项目的**文件读写访问**，使 AI 助手能够理解场景、执行 **C# 代码**、监控日志、控制播放模式并操作项目文件。
   - Blender 支持目前正在开发中，用于 3D 内容生成。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **o1-pro 模型定价令人震惊**：新的 **o1-pro 模型** 现已在 API 中向特定开发者开放，支持视觉、function calling 和 structured outputs，详见 [OpenAI 文档](https://platform.openai.com/docs/models/o1-pro)。
   - 然而，其 **$150 / 1M 输入 tokens** 和 **$600 / 1M 输出 tokens** 的高昂定价引发了争论，尽管一些用户声称它能一次性解决其他模型失败的代码任务。
- **ChatGPT 代码带表情符号？！**：根据 **gpt-4-discussions** 频道的讨论，成员们正在寻找防止 **ChatGPT** 在代码中插入表情符号的方法，即使设置了自定义指令也是如此。
   - 建议包括避免使用 *emoji* 一词，并指示模型 *“以恰当、专业的方式编写代码”*。
- **中国模型自我审查！**：一名用户报告称，一个中国模型会删除有关**文化大革命**提示词的回复，并提供了[截图](https://cdn.discordapp.com/attachments/998381918976479273/1352038445069111340/image.png?ex=67dd382c&is=67dbe6ac&hm=a5c413109c60b302e9252036467f20eb90689c5216bca9d9003c63d2efea915f&)作为证据。
   - 该问题在 **ai-discussions** 频道中进行了讨论，突显了对 AI 模型审查制度的关注。
- **AI 不会让你选股**：在 **api-discussions** 和 **prompt-engineering** 中，用户讨论了使用 AI 进行**股市**预测，但成员们指出，提供**财务建议**违反了 [OpenAI 的使用政策](https://openai.com/policies/usage-policies/)。
   - 澄清说明，探索个人股票想法是可以接受的，但禁止向他人提供建议。
- **Agent SDK 与 MCP 的对决**：成员们将 **OpenAI Agent SDK** 与 **MCP (Model Communication Protocol)** 进行了对比，指出前者仅适用于 **OpenAI 模型**，而后者支持任何使用任何工具的 **LLM**。
   - MCP 允许通过 `npx` 和 `uvx` 轻松加载集成，例如 `npx -y @tokenizin/mcp-npx-fetch` 或 `uvx basic-memory mcp`。



---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LLMs 面临 AI Hallucinations 的批评**：成员们对 **LLMs** 在进行研究时容易出错和产生 **hallucinations**（幻觉）表示担忧。
   - 一位成员观察到，**Agent** 虽然能找到准确的来源，但仍然会虚构网站，类似于 **Perplexity** 的 **Deep Research** 容易分心并产生大量 **hallucinate**。
- **o1-pro 价格引发关注，被认为 GPT-4.5 定价过高**：**OpenAI** 的新 **o1-pro API** 定价为 **每 1M input tokens $150，每 1M output tokens $600**（[公告](https://x.com/OpenAIDevs/status/1902485690958450871)）。
   - 一些成员认为这意味着 **GPT-4.5** 定价过高，其中一人评论称，通过算力优化托管同等模型会更便宜；然而，其他人辩称 **o1** 的推理链（reasoning chains）需要更多资源。
- **文件上传限制困扰 Gemini Pro**：用户质疑为什么 [Gemini Pro](https://gemini.google.com/app) 不像 **Flash Thinking** 那样支持文件上传。
   - 他们还指出 **AI 模型** 在准确识别 PDF 文件（包括非扫描件）方面存在困难，并希望未来的模型能够仔细阅读完整文章。
- **Claude 3.7 编程能力引发争论**：一些成员认为 **Claude 3.7** 的编程能力被高估了，认为它在 Web 开发和类似于 **SWE-bench** 的任务中表现出色，但在通用编程方面表现不佳（[排行榜](https://lmarena.ai/?leaderboard)）。
   - 相反，其他人发现 **Deepseek R1** 在终端命令测试中表现更优。
- **在 Google AI Studio 中构建 Vision AI Agent**：一位成员报告称，成功使用 [Google AI Studio API](https://aistudio.google.com/app/library) 在 Python 中构建了一个相当智能的 **vision AI agent**。
   - 他们还尝试了**同时运行 2-5 个以上的 Agent**，共享内存并共同浏览互联网。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Flux Diffusion 在本地运行**：成员们讨论了在本地运行 **Flux diffusion model**，建议对其进行量化以在有限的 **VRAM** 上获得更好性能，并参考了[文档](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux)和[这篇博客文章](https://huggingface.co/blog/quanto-diffusers)。
   - 成员们分享了一个用于优化扩散模型的相关 [GitHub repo](https://github.com/sayakpaul/diffusers-torchao)，以及一篇关于 GUI 设置的 [Civitai 文章](https://civitai.com/articles/9060/how-to-set-up-and-run-flux-on-forge-even-if-you-have-low-vram)。
- **HF Inference API 报错，用户表示愤怒**：一位用户报告了 **Hugging Face Inference API** 返回 404 错误的普遍问题，影响了多个应用程序和付费用户，并链接到了[此讨论](https://discuss.huggingface.co/t/hf-inference-api-last-few-minutes-returns-the-same-404-exception-to-all-models/146646/20)。
   - 一名团队成员承认了该问题，并表示他们已*向团队报告*以进行进一步调查。
- **Roblox 通过 HF 分类器实现语音安全**：Roblox 在 Hugging Face 上发布了一个**语音安全分类器**，该分类器使用 2,374 小时的语音聊天音频片段进行了微调，详见[这篇博客文章](https://research.roblox.com/tech-blog/2024/06/deploying-ml-for-voice-safety)和[模型卡片](https://huggingface.co/Roblox/voice-safety-classifier)。
   - 该模型输出一个带有标签的张量，如 **Profanity**（亵渎）、**DatingAndSexting**（约会与性暗示）、**Racist**（种族主义）、**Bullying**（霸凌）、**Other**（其他）和 **NoViolation**（无违规）。
- **Little Geeky 学会了说话**：一位成员展示了一个基于 **Ollama** 的 **Gradio UI**，由 **Kokoro TTS** 驱动，可以自动以选定的声音朗读文本输出，该项目可在 [Little Geeky's Learning UI](https://github.com/GeekyGhost/Little-Geeky-s-Learning-UI.git) 获取。
   - 该 UI 包含模型创建和管理工具，以及阅读电子书和回答文档相关问题的能力。
- **Vision Model 面临输入处理失败**：一位成员报告在下载 **LLaVA** 后使用本地 **vision model** 时，收到 *"failed to process inputs: unable to make llava embedding from image"* 错误。
   - 失败的根本原因尚不明确。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **O1-Pro 定价令用户震惊**：用户对 **O1-Pro** 的定价表示愤怒，认为 **$150/月输入**和 **$600/月输出**的成本简直**疯狂到令人望而却步**。
   - 有推测认为，高昂的价格是对来自 **R1 和中国模型**竞争的回应，或者是由于 **OAI** 在没有流式传输支持的情况下结合了多个模型的输出。
- **LLM 象棋锦标赛测试原生性能**：一名成员发起了第二次**象棋锦标赛**以评估原生性能，利用**原始 PGN 棋谱文本续写**并发布了[结果](https://dubesor.de/chessbeta/tournament2)。
   - 模型重复对局序列并增加一个新招法，由 **Stockfish 17** 评估准确性；包含推理的第一次锦标赛可见[此处](https://discord.com/channels/1091220969173028894/1350154062842298368)。
- **OpenRouter API：免费模型并不完全免费？**：一位用户发现 `/api/v1/chat/completions` 端点中的 **model 字段**是必填的，这与文档中声称的即使在使用[免费模型](https://openrouter.ai/docs/api-reference/overview)时也是可选的说法相矛盾。
   - 一位用户建议 model 字段应默认为[默认模型](https://openrouter.ai/settings/preferences)，或者默认为系统预设的默认模型。
- **Groq API 出现间歇性功能故障**：用户报告称 **Groq** 在 OpenRouter 聊天室中可以运行，但无法通过 API 运行。
   - 一名成员要求澄清使用 API 时遇到的具体错误，并指出了 Groq 的速度优势。
- **OpenAI 发布新款音频模型！**：**OpenAI** 将发布**两款新 STT 模型**和**一款新 TTS 模型**（**gpt-4o-mini-tts**）。
   - 语音转文本模型命名为 **gpt-4o-transcribe** 和 **gpt-4o-mini-transcribe**，并包含与 Agents SDK 的音频集成，用于创建可定制的语音 Agent。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Vast.ai 裸机访问：难以实现？**：成员们讨论了 [Vast.ai](https://vast.ai) 是否允许 **NCU profiling** 以及获取裸机访问权限是否可行，同时另一名成员询问了如何获取 **NCU** 和 **NSYS**。
   - 虽然一名成员怀疑裸机访问的可能性，但他们承认自己也可能弄错。
- **BFloat16 原子操作困扰 Triton**：社区探索了在非 Hopper GPU 上使 `tl.atomic` 支持 **bfloat16** 的方法，建议查看 [tilelang](https://github.com/tile-ai/tilelang) 以了解原子操作以及非 Hopper GPU 对 **bfloat16** 支持的限制。
   - 一名成员指出，由于 `tl.atomic_add` 的限制，目前使用 bfloat16 会导致崩溃，但有人认为可以通过 `tl.atomic_cas` 实现原子加法。
- **cuTile 可能是另一个 Triton**：成员们讨论了 **NVIDIA** 发布的 **cuTile**（一种用于 CUDA 的 Tile 编程模型），并引用了关于它的[一条推文](https://x.com/blelbach/status/1902113767066103949)，一名成员对 NVIDIA 可能不支持 AMD GPU 等其他后端表示担忧。
   - 有推测认为 **cuTile** 可能类似于 **tilelang**，即“另一个 Triton，但由 NVIDIA 出品”。
- **GEMM 激活融合受挫**：一名成员在编写自定义融合的 **GEMM+activation Triton 内核**时遇到问题，指出这取决于**寄存器溢出 (register spillage)**，因为如果 GEMM 使用了所有寄存器，在 **GEMM** 中融合激活函数会损害性能。
   - 正如 gpu-mode 第 45 课中所讨论的，将 GEMM 和激活函数拆分为两个内核可能会更快。
- **对齐改变处理器中的跳转**：在 C++ 代码中包含 `<iostream>` 可能会改变主循环跳转的对齐方式，从而由于处理器特定的行为影响性能，因为*跳转的速度可能取决于目标地址的对齐方式*。
   - 一名成员指出，在某些 Intel CPU 中，条件跳转指令的 32 字节对齐模数可能会因修补安全漏洞的微代码更新而显著影响性能，建议在关键循环前的内联汇编中添加 **16 条 NOP 指令**可以重现该问题。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Orpheus 夺得 TTS 竞技场榜首**：开源 TTS 模型 **Orpheus** 首次亮相，据 [此推文](https://x.com/eliasfiz/status/1902435597954003174?s=46) 和 [此 YouTube 视频](https://youtu.be/Btos-LEYQ30?si=XjoZEuJT49jXOLRJ) 称，其性能优于 **ElevenLabs** 和 **OpenAI** 等开源和闭源模型。
   - 社区成员讨论了 **Orpheus** 对 TTS 领域潜在的影响，并期待进一步的基准测试和对比以验证这些说法。
- **DeepSeek R1 训练费用引发热议**：关于 **DeepSeek R1** 训练成本的估算正在讨论中，初始数据约为 **600 万美元**，但根据 [此推文](https://x.com/teortaxesTex/status/1902658735454953531)，李开复估计 2024 年整个 **DeepSeek** 项目的投入为 **1.4 亿美元**。
   - 讨论强调了开发尖端 AI 模型所需的巨额投资以及成本估算的差异。
- **OpenAI 的 o1-pro 携增强功能上线 API**：**OpenAI** 在其 API 中发布了 **o1-pro**，提供更优的响应，成本为每 1M 输入 token **150 美元**，每 1M 输出 token **600 美元**，面向 Tier 1–5 的特定开发者开放，详见 [此推文](https://x.com/openaidevs/status/1902485690958450871?s=46) 和 [OpenAI 文档](https://platform.openai.com/docs/models/o1-pro)。
   - 该模型支持 vision、function calling 和 Structured Outputs，标志着 **OpenAI** API 服务的重大升级。
- **Gemma 软件包简化微调工作**：推出了 **Gemma package**，这是一个简化 **Gemma** 使用和微调的库，可通过 *pip install gemma* 安装，并记录在 [gemma-llm.readthedocs.io](https://gemma-llm.readthedocs.io/en/latest) 上，详见 [此推文](https://x.com/osanseviero/status/1902456220876787763)。
   - 该软件包包含关于微调、sharding、LoRA、PEFT、多模态和 tokenization 的文档，简化了开发流程。
- **据报道 Perplexity 寻求 180 亿美元估值**：据 [Bloomberg](https://www.bloomberg.com/news/articles/2025-03-20/perplexity-in-early-talks-for-funding-at-18-billion-value?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTc0MjQ5MzI4OSwiZXhwIjoxNzQzMDk4MDg5LCJhcnRpY2xlSWQiOiJTVERYV01UMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.GYIVla5ZD3lp70ED36NxSKtCvWFpu8qrEaHIEPydQ9s) 报道，**Perplexity** 正就 5 亿至 10 亿美元的新一轮融资进行早期谈判，估值达 **180 亿美元**，可能比 12 月的估值翻一番。
   - 这一轮融资将反映出投资者对 **Perplexity** 搜索和 AI 技术的信心增强。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **单语模型引发困惑**：成员们对“针对 350 种语言的单语模型”这一概念表示困惑，因为预期模型应该是 **multilingual**（多语言）的。
   - 一位成员澄清说，该项目为每种语言训练一个模型，最终在 [HF](https://huggingface.co/) 上产生了 **1154 个总模型**。
- **CV 工程师开启 AI Safety 探索**：一位成员介绍自己是 **CV 工程师**，并表达了对为 **AI safety** 和 **interpretability**（可解释性）研究做出贡献的兴奋之情。
   - 他们有兴趣与群组中的其他人讨论这些话题。
- **探索 Expert Choice Routing**：成员们讨论了在自回归模型上实现 **expert choice routing**，在训练期间使用在线分位数估计 (online quantile estimation) 来推导推理阈值。
   - 一个建议是假设 router logits 符合 **Gaussian** 分布，计算 EMA 均值和标准差，然后利用 **Gaussian quantile function**。
- **分位数估计管理稀疏性**：一位成员提议在推理时使用 **population quantiles**（总体分位数）的估计来维持所需的平均稀疏性，并类比了 *batchnorm*。
   - 另一位成员指出，由于 *node limited routing*，**dsv3 architecture** 能够激活 **8-13 个专家**，但目标是允许在 **0 到 N 个专家**之间激活。
- **LLM 面临 Kolmogorov 压缩测试**：一位成员分享了论文 [《The Kolmogorov Test》](https://arxiv.org/abs/2503.13992)，该论文为代码生成 LLM 引入了一种“压缩即智能”的测试。
   - **Kolmogorov Test (KT)** 在推理时向模型提供一个数据序列，挑战其生成能够产生该序列的最短程序。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command-A 用卡斯蒂利亚语亲切交流**：一位来自墨西哥的用户报告称，**Command-A** 以一种令人惊讶的自然且友好的方式*模仿*了他们的方言。
   - 该模型感觉就像在与墨西哥人交谈，即使没有特定的 Prompt。
- **Command-R 消耗大量 Token**：一位用户通过 **OpenRouter** 为 **Azure AI Search** 测试了 **Cohere** 模型，并对输出结果印象深刻。
   - 然而，他们指出，*每次请求的输入消耗了 80,000 个 Token*。
- **Connectors 让当前的 Cmd 模型产生困扰**：一位用户探索了带有 **Slack integration** 的 **Connectors**，但发现它们似乎不被 **cmd-R** 和 **cmd-A** 等最新模型支持。
   - 旧模型返回了 500 错误，且 Connectors 似乎已从 V2 版本的 API 中移除，这令人失望，因为它们简化了数据处理；同时也有人担心从 **Connectors** 到 **Tools** 的过渡是否是等价替换。
- **Good News MCP Server 传递正能量**：一位成员构建了一个名为 *Goodnews MCP* 的 **MCP server**，它在其工具 `fetch_good_news_list` 中使用 **Cohere Command A** 为 MCP 客户端提供积极、令人振奋的新闻，代码可在 [GitHub](https://github.com/VectorInstitute/mcp-goodnews) 获取。
   - 该系统使用 **Cohere LLM** 对近期头条新闻进行排名，返回最积极的文章。
- **Cohere API 上下文：容量至关重要**：一位成员表示更倾向于使用 **Cohere API**，因为 **OpenAI API** 的上下文大小限制仅为 **128,000**，而 Cohere 提供 **200,000**。
   - 然而，使用兼容性 API 会导致你*失去对 API 响应中 `documents` 和 `citations` 等 Cohere 特定功能的访问权限*。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **光子学推测引发 GPU 热议**：讨论集中在 **Ruben GPUs** 中的 **photonics** 和**集成 CPU** 是仅限于数据中心模型，还是会扩展到消费级版本（可能是 **6000 series**）。
   - 有人提出了 **CX9** 拥有共封装光学器件（co-packaged optics）的可能性，暗示 **DIGITs successor** 可能会利用此类技术，而 **CPU** 已确认将用于 **DGX workstations**。
- **调试断言需要额外的编译器选项**：在 **Mojo** 标准库中启用调试断言（debug asserts）需要一个额外的编译选项 `-D ASSERT=_`，这并未被广泛宣传，详见 [debug_assert.mojo](https://github.com/modular/max/blob/main/mojo/stdlib/src/builtin/debug_assert.mojo#L88-L100)。
   - 有人指出使用 `-g` 并不会启用断言，预期的做法是使用 `-Og` 编译时应自动开启它们。
- **Mojo List 索引因 UB 打印 0**：当 **Mojo List** 索引超出范围时，由于未定义行为（**UB**），它会打印 **0** 而不是抛出错误。
   - 出现此问题是因为代码索引超出了列表范围，进入了内核提供的零初始化内存。
- **关于默认断言行为的讨论**：引发了关于 `debug_assert` 默认行为的讨论，特别是围绕 `debug_assert[assert_mode="none"]` 的困惑，以及在调试模式下是否应默认启用它。
   - 有建议认为，在调试模式下运行程序时，应启用所有断言。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **DeepLearningAI 推出 Agentic Workflow 课程**：**DeepLearningAI** 推出了一门关于使用 **RAG** 构建 Agentic Workflow 的短课，内容涵盖解析表单和提取关键字段，更多详情见 [Twitter](https://t.co/qvqNj7MJbn)。
   - 该课程教授如何创建能够自动处理信息并生成上下文感知响应的系统。
- **AMD GPU 驱动 AI 语音助手流水线**：一个教程展示了如何使用 **AMD GPU** 创建多模态流水线，实现语音转文本、使用 **RAG** 以及将文本转回语音，该方案利用了 **ROCm** 和 **LlamaIndex**，详见[此教程](https://t.co/jdG2VT0cbf)。
   - 该教程重点介绍了 **ROCm** 环境的搭建，以及如何集成 **LlamaIndex** 以实现上下文感知的语音助手应用。
- **LLM.as_structured_llm 需要并行工具调用支持**：一位成员指出，在使用 `LLM.as_structured_llm` 配合 `.chat` 时缺少 `allow_parallel_tool_calls` 选项，并建议扩展 `.as_structured_llm()` 调用以接受诸如 `allow_parallel_tool_calls=False` 之类的参数。
   - 另一位用户建议直接使用 `FunctionCallingProgram` 进行自定义，并为 OpenAI 设置 `additional_kwargs={"parallel_tool_calls": False}`，参考了 [OpenAI API 文档](https://platform.openai.com/docs/api-reference/chat/create#chat-create-parallel_tool_calls)。
- **Ollama 的推理标签困扰 ChatMemoryBuffer**：一位在使用 **Ollama** 配合 **qwq 模型** 的用户正苦于 `<think>` 推理标签出现在 `ChatMemoryBuffer` 的 `text` 块中，并寻求在使用 `ChatMemoryBuffer.from_defaults` 时移除它们的方法。
   - 另一位用户建议对 LLM 输出进行手动后处理，因为 **Ollama** 不提供内置过滤功能；原用户表示愿意分享他们的 `MariaDBChatStore` 实现（`PostgresChatStore` 的克隆版本）。
- **llamaparse PDF 问答困惑**：一位用户就使用 **llamaparse** 解析的数百个 PDF 文件的问答任务寻求建议，指出有些文件解析完美，而另一些则生成了毫无意义的 Markdown。
   - 他们还对如何为需要不同处理方式的文档实现不同的解析模式感到好奇。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Nvidia 硬件仍落后于进度**：成员们报告称 **Nvidia** 的新硬件迟到了，称 **H200** 在 **2 年前** 就已发布，但 **6 个月前** 才提供给客户。
   - 一位成员调侃道，这就是“**Nvidia** 风格”。
- **Gemma 3 微调将获得 Torchtune 支持**：一位成员正在处理一个[仅限 gemma 文本的 PR](https://github.com/pytorch/torchtune/pull/2485)，并可能尝试加速落地，之后再添加图像功能。
   - 另一位成员承诺将尽快继续 **Gemma 3** 的工作，并开玩笑地宣称他们的“假期正在转化为 **Torchtune** 冲刺”。
- **驱动版本导致 nv-fabricmanager 错误**：当 **nv-fabricmanager** 的驱动版本与显卡驱动版本不匹配时，可能会抛出错误。
   - 此问题已在某些按需 VM 上被观察到。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Adam 优化器在 ML4SCI 任务中达到低损耗**：一位成员报告称，使用 **Adam 优化器** 为 `ML4SCI/task1` 训练了一个模型，损耗达到了 **0.2s** 左右，设置代码可在 [GitHub](https://github.com/kayo09/gsoc_2025/tree/main/ML4SCI/task1) 上找到。
   - 该仓库是该成员 Google Summer of Code 2025 项目的一部分。
- **General 频道执行 Discord 规则**：一位成员被提醒遵守 Discord 规则，特别是该频道仅用于讨论 **tinygrad 开发** 和 **tinygrad 使用**。
   - 未提供有关违规行为的更多细节。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **用户对 AgentX 研究方向表示期待**：一位用户表达了加入 **AgentX Research Track** 的兴奋和兴趣，渴望与导师和博士后合作。
   - 他们的目标是通过对 **LLM Agent** 和 **多 Agent 系统** 的研究为该项目做出贡献。
- **用户承诺发挥主动性和自主性**：一位用户承诺在 **AgentX Research Track** 中会积极主动且独立地推动其研究。
   - 他们承诺在给定时间内交付高质量的工作，并感谢任何能增加其入选机会的支持。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 用户寻求 arXiv 论文实现指导**：kotykd 询问了是否可以使用 DSPy 实现[这篇 arXiv 论文](https://arxiv.org/abs/2502.06855)中描述的方法。
   - 未提供关于具体实现挑战或目标的更多细节。
- **arXiv 论文实现**：用户 kotykd 引用了一篇 [arXiv 论文](https://arxiv.org/abs/2502.06855)，并询问是否可以使用 DSPy 来实现它。
   - 论文内容以及用户感兴趣的具体方面未详细说明。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Nomic.ai (GPT4All) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：各频道详细摘要与链接

{% if medium == 'web' %}

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1351993905717055488)** (1517 条消息🔥🔥🔥): 

> `Agent 模式宕机, Dan Perks, Keychron 键盘, Vibe Coding, Pear AI 对标 Cursor` 

- **Agent 挂了：我生命中最漫长的一小时**：成员们报告称 Agent 模式宕机了一小时，且[状态页面](https://status.cursor.sh/)未及时更新，称这是“我生命中最漫长的一小时”。
   - 成员们开玩笑说 *dan percs* 正在处理这个问题，他“正忙于在 Cursor 中回复用户”并“处理缓慢的请求”，这就是他为什么一直在线的原因。
- **Dan Perks：键盘鉴赏家**：Cursor 的 Dan Perks 征求了[关于 Keychron 键盘的意见](https://x.com/danperks_/status/1902474398206181398?s=46)，特别是在寻找一款带有“旋钮”的“矮轴且整洁”的型号。
   - 建议纷至沓来，包括 Keychron 的矮轴系列，尽管 Dan 对键帽的美观表示担忧，称“我不喜欢这些键帽”。
- **Pear 的压力：Pear AI 对标 Cursor**：几位成员宣扬了使用 [Pear AI](https://www.pear.ai/) 的优势，并声称 [Cursor 现在更贵了](https://www.reddit.com/r/ChatGPTCoding/comments/1jdd0n8/some_of_the_best_ai_ides_for_fullstacker/)。
   - 一位成员声称因为订阅了多个 Cursor 年费会员而“完蛋了（cooked）”，另一位声称：“如果 Cursor 改变他们的 Context Window，我会留在 Cursor，或者将他们的 Sonnet Max 改为高级使用额度，否则如果我要为 Sonnet Max 付费，我不如用 Pear，因为价格更便宜。”
- **ASI：唯一的全球解决方案？**：成员们辩论了人工超智能（ASI）是否是下一次进化，声称“ASI-奇点（天赐之物）必须是唯一的全球解决方案”。
   - 其他人持怀疑态度，一位用户开玩笑说“性别研究比 ASI 更重要”，声称“这是让智人成为星际物种的下一步，通过中性流动的性别，我们可以与来自不同星球的外星人交配，并适应他们的巫术技术”。
- **许可证纠纷：Pear AI 克隆了 Continue？**：成员们讨论了围绕 [Pear AI](https://www.pear.ai/) 的争议，其中一人声称 Pear AI “基本上克隆了 Continue”，“只是拿走了别人的工作并决定现在这是他们自己的项目”。
   - 其他人担心该项目是闭源的，并认为应该转向其他替代方案，如 [Trae AI](https://www.trae.ai/)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.twitch.tv/theprimeagen">ThePrimeagen - Twitch</a>: TheStartup™ (百亿级) CEO，困在 Vim 中却向往着 Emacs</li><li><a href="https://x.com/danperks_">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://markdown-renderer-antdx316.vercel.app/">Markdown Renderer</a>: 未找到描述</li><li><a href="https://www.cursor.com/settings">设置 | Cursor - AI 代码编辑器</a>: 你可以在此处管理你的账户、账单和团队设置。</li><li><a href="https://tenor.com/view/i-use-arch-btw-use-arch-linux-fedora-gif-23272370">我使用 Arch Btw 使用 GIF - 我使用 Arch Btw 使用 Arch - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/anthropicai/status/1902765011727999046?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 Anthropic (@AnthropicAI) 的推文</a>: Claude 现在可以搜索网页了。每个回复都包含行内引用，因此你也可以验证来源。</li><li><a href="https://x.com/vercel/status/1902771130970280115?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 Vercel (@vercel) 的推文</a>: Vercel 和 @xAI 正在合作，为开发者带来零摩擦的 AI。• Grok 模型现已在 Vercel 上可用 • 独家 xAI 免费层级——无需额外注册 • 按使用量付费...</li><li><a href="https://www.twitch.tv/ThePrimeagen">ThePrimeagen - Twitch</a>: TheStartup™ (百亿级) CEO，困在 Vim 中却向往着 Emacs</li><li><a href="https://www.reddit.com/r/curso">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://changelog.cursor.sh/">更新日志 | Cursor - AI 代码编辑器</a>: 新的更新和改进。</li><li><a href="https://www.youtube.com/watch?v=tFfTludf0SU&t=34s">《蠢蛋进化论》精彩片段 - Lexus 医生！</a>: 最精彩的场景之一。Lexus 医生！《蠢蛋进化论》2006 年喜剧电影，由 Mike Judge 执导。Luke Wilson 和 Maya Rudolph 主演。</li><li><a href="https://www.reddit.com/r/ChatGPTCoding/comments/1jdd0n8/some_of_the_best_ai_ides_for_fullstacker/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/cursor/comments/1jff3yk/exposed_cursors_claude_37_max_is_charging_premium/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://youtu.be/vBMC7OiipKc?si=NuT0HLrnlvaYd5tL"> - YouTube</a>: 未找到描述</li><li><a href="https://www.trae.ai">Trae - 使用 Trae 更快交付</a>: Trae 是一款自适应 AI IDE，它改变了你的工作方式，通过协作助你运行得更快。</li><li><a href="https://www.reddit.com/r/cursor/comments/1jf8pny/i_feel_heavily_scammed_out_of_my_premium_runs/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://dialogo.chat">Dialogo AI - 智能任务自动化</a>: Dialogo AI 提供智能 AI Agent，能够跨平台学习、适应并自动化复杂的工作流。从数据分析到系统管理，我们的智能 Agent 改变了你的工作方式。</li><li><a href="https://x.com/i/communities/1836496043233722680">来自 GitHub 的推文 - FxEmbed/FxEmbed: 修复 X/Twitter 和 Bluesky 的嵌入内容！</a>: 修复 X/Twitter 和 Bluesky 的嵌入内容！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FxEmbed/FxEmbed</li><li><a href="https://www.keychron.com/collections/low-profile-keyboard-collection">矮轴键盘</a>: 使用我们的 Keychron 矮轴机械键盘体验超薄手感。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1351993898540732566)** (371 条消息🔥🔥): 

> `TPUs speed comparison, Gradient Accumulation fix, Gemma model version misinformation, Sophia optimizer experiments, Gemma 3 Activation Normalization`

- **TPUs 飞速，T4s 迷茫**：一位成员指出 **TPUs** 明显比 **T4s** 快，特别是在使用 batch size 为 8 时，并根据观察到的时间戳强调了其卓越的速度，还附带了一张对比 [截图](https://cdn.discordapp.com/attachments/1179035537529643040/1351993898289070183/image.png?ex=67ddb770&is=67dc65f0&hm=a8e536c09f7ad917858b287b86d2618d4679f2014e4ee1883ffe62bcf0b92587)。
- **Gradient Accumulation 已修复**：一篇博客文章（[Unsloth Gradient Accumulation 修复](https://unsloth.ai/blog/gradient)）讨论了影响序列模型训练、预训练和微调运行的 **Gradient Accumulation** 问题，该问题已得到解决，以确保准确的训练和 loss 计算。
   - 该修复旨在*通过减少 VRAM 使用量来模拟全 batch 训练*，同时也影响 DDP 和多 GPU 设置。
- **Google Gemma 的身份危机**：用户报告称，从 Hugging Face 下载的 **Gemma 模型**会错误地将自己识别为具有 **2B** 或 **7B 参数**的*第一代*模型，即使下载的模型是 **12B Gemma 3**。
   - 这种幻觉问题源于 Google 忽略了更新负责此类识别的训练代码部分，因为这些模型*知道*自己是 Gemma，且至少有两种不同的容量。
- **Gemma 3 获得 Float16 修复**：Unsloth 修复了 Gemma 3 在 float16 下的**无限激活（infinite activations）**问题，该问题曾导致在 Colab GPU 上进行微调和推理时出现 **NaN gradients**。该修复将所有中间激活保留在 **bfloat16** 中，并将 layernorm 操作上采样（upcast）到 **float32**。
   - 该修复避免了降低速度，而最简单的解决方案是全部使用 float32 或 bfloat16，但正如 [Unsloth AI 博客](https://unsloth.ai/blog/gemma3)中所解释的，没有 float16 tensor cores 的 GPU 速度会慢 4 倍或更多。
- **Unsloth Notebooks 缺少依赖**：用户报告了运行 Unsloth notebooks（特别是 Google Colab 上的 Gemma 3 和 Mistral notebooks）时出现的问题，这些问题是由安装命令中的 `--no-deps` 标志导致的依赖项缺失以及其他各种版本不兼容引起的。
   - 一位成员正在处理此事。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/featherless-ai/Qwerky-QwQ-32B">featherless-ai/Qwerky-QwQ-32B · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/danielhanchen/status/1902518780166824381">来自 Daniel Han (@danielhanchen) 的推文</a>: 我们下周四将参加 Ollama 和 vLLM 的 inference night！🦥🦙来 @YCombinator 的旧金山办公室见我们吧。现场还会有许多其他酷炫的开源项目！引用 olla...</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#id-2.-choose-the-right-model--method">Fine-tuning 指南 | Unsloth 文档</a>: 学习 fine-tuning 的所有基础知识和最佳实践。初学者友好。</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - mteb 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/aya-vision-8b">unsloth/aya-vision-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://gist.github.com/gabriel-peracio/62e42ed39b624a0e74482e5ebec0f115">分析 embedding 空间使用情况</a>: 分析 embedding 空间使用情况。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://unsloth.ai/blog/gemma3#fixes">使用 Unsloth 微调 Gemma 3</a>: Gemma 3，Google 的新型 multimodal 模型。使用 Unsloth 进行 fine-tune 和运行！Gemma 3 提供 1B, 4B, 12B 和 27B 尺寸。</li><li><a href="https://x.com/danielhanchen/status/1902396261875249346">来自 Daniel Han (@danielhanchen) 的推文</a>: 我修复了 Gemma 3 在 float16 下的 infinite activations 问题！在 fine-tuning 和 inference 过程中，我注意到 Colab GPU 产生了 NaN gradients —— 看起来在每个 layernorm 之后，activations 都会爆炸！max(float16) = 65...</li><li><a href="https://github.com/unslothai/unsloth/issues/2122">如何在 Unsloth 框架上使用自定义 data collator 对 Gemma 3 进行 vision fine-tune？ · Issue #2122 · unslothai/unsloth</a>: 我之前参考了 Google 的教程：https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora#setup-development-environment 并且成功运行了，使用的是我自定义的 data_col...</li><li><a href="https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb">notebooks/nb/Gemma3_(1B)-GRPO.ipynb at main · unslothai/notebooks</a>: 适用于 Google Colab, Kaggle, Hugging Face 等的 Unsloth fine-tuning Notebooks。- unslothai/notebooks</li><li><a href="https://unsloth.ai/blog/gradient">LLM 训练中的 Bug 修复 - Gradient Accumulation</a>: Unsloth 的 Gradient Accumulation 修复解决了 LLM 训练中的关键错误。</li><li><a href="https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora#setup-development-environment">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/huggingface/trl/pull/3072">[GRPO] 由 CompN3rd 为 trainer 添加 vlm 训练功能 · Pull Request #3072 · huggingface/trl</a>: 这个 PR 做了什么？这是解决 #2917 的一次尝试。添加了相关的 unittest，且非“玩具级示例”的训练似乎也实现了 reward 最大化，但我并不...</li><li><a href="https://github.com/unslothai/notebooks/blob/main/nb/">notebooks/nb at main · unslothai/notebooks</a>: 适用于 Google Colab, Kaggle, Hugging Face 等的 Unsloth fine-tuning Notebooks。- unslothai/notebooks</li><li><a href="https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">notebooks/nb/Gemma3_(4B).ipynb at main · unslothai/notebooks</a>: 适用于 Google Colab, Kaggle, Hugging Face 等的 Unsloth fine-tuning Notebooks。- unslothai/notebooks</li><li><a href="https://github.com/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb">notebooks/nb/Mistral_(7B)-Text_Completion.ipynb at main · unslothai/notebooks</a>: 适用于 Google Colab, Kaggle, Hugging Face 等的 Unsloth fine-tuning Notebooks。- unslothai/notebooks</li><li><a href="https://github.com/unslothai/unsloth/issues/2127">Text Completion Notebook - Backwards 要求 embeddings 为 bf16 或 fp16 · Issue #2127 · unslothai/unsloth</a>: 我正尝试运行来自 Continued pretraining 的 notebook，https://docs.unsloth.ai/basics/continued-pretraining 文本补全 notebook https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObe...</li><li><a href="https://github.com/huggingface/transformers/issues/36683#issuecomment-2736982634)">AttributeError: 'Gemma3Config' 对象没有 'vocab_size' 属性 · Issue #36683 · huggingface/transformers</a>: 系统信息 v4.50.0.dev0 谁能帮忙？ @ArthurZucker @LysandreJik @xenova 信息：官方示例脚本，我修改后的脚本，任务：示例文件夹中官方支持的任务...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py#L712).">unsloth/unsloth/models/loader.py at main · unslothai/unsloth</a>: 以 2 倍速度和减少 70% 显存 fine-tune Llama 3.3, DeepSeek-R1, Gemma 3 和 Reasoning LLMs！🦥 - unslothai/unsloth
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1351993630348414976)** (11 条消息🔥): 

> `GTC 值得吗？, Gemma 3 BFloat16 范围, Cfloat16 想法, vllm 的 hiddenlayer` 


- **GTC 门票费用合理吗？**：一位成员询问 **GTC** 是否值得购买门票，并表示有兴趣明年参加。
   - 另一位成员提到他们通过 **NVIDIA 联系人** 获得了一张赠票，并建议向他们索要。
- **Gemma 3 偏爱 BFloat16**：Daniel Han 在[这条推文](https://x.com/danielhanchen/status/1902402778959695885)中分享了关于 **Gemma 3** 是他遇到的第一个“偏爱”使用更大的完整 **bfloat16** 范围的模型的看法，并推测这可能是它在相对较小的尺寸下表现极其强大的原因。
- **提议 cfloat16**：参考 **Gemma 3** 如何“偏爱”更大的完整 **bfloat16** 范围，一位成员提出了 **cfloat16** 的想法：*1 位符号位，10 位指数位，5 位尾数位*。
   - 据称这更好，因为*反正指数位才是最重要的*。
- **vllm 需要 hiddenlayer？**：一位成员询问是否有办法为非池化模型在 **vllm** 中获取 **hiddenlayer**（最后一层），并请求一个 7b r1 distill。



**提到的链接**：<a href="https://x.com/danielhanchen/status/1902402778959695885">Daniel Han (@danielhanchen) 的推文</a>：进一步思考——我发现这整体上非常迷人！Gemma 3 是我遇到的第一个“偏爱”使用更大的完整 bfloat16 范围的模型，我推测，m...

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1352033754683801633)** (63 messages🔥🔥): 

> `Gemma 3 微调, Prompt/Response 对的数据格式, Gemma 3 的多图像训练, Gemma 3 的 Triton 降级, DPO 示例与补丁` 


- ****Gemma 3** 依赖错误：一个令人沮丧的开始**: 用户在尝试微调 **Gemma 3** 时遇到了依赖错误，即使直接从 git 安装了最新的 *transformers* 也是如此。
   - 他们在 Colab 上也遇到了同样的问题，这表明环境或安装过程可能存在潜在问题。
- ****Gemma 3**：确保数据格式正确**: 用户在 **Gemma 3** 微调的正确数据格式上苦苦挣扎，质疑 Prompt/Response 对是否需要遵循 notebook 中指示的特定格式。
   - 他们意识到可以使用 **Gemma 3** 本身来创建一个包含适当对话风格格式的体面 Prompt。
- ****Triton** 需要为 **Gemma 3** 降级：一个离奇的困境**: 用户在配备 4090 的全新 Python 3.12.9 环境中运行 **Gemma 3** 时遇到问题，遇到了与 **Triton** 编译器相关的 *SystemError*。
   - 解决方案包括按照 [此 GitHub issue](https://github.com/triton-lang/triton/issues/5919#issuecomment-2733328584) 中的建议，在 Python 3.11.x 上强制将 **Triton** 降级到 3.1.0 版本。
- **在 Ollama 中保存微调模型：案例研究**: 用户报告了微调模型在 Colab 中的表现与保存为 `.gguf` 文件并在本地使用 **Ollama** 运行时的行为之间存在差异。
   - 他们询问了保存模型的正确方法以保留微调效果，并区分了 `model.save_pretrained_gguf` 和 `model.save_pretrained_merged`。
- **求助！Qwen 2.5 在函数调用期间产生幻觉**: 用户在使用 **Qwen2.5:7b** 处理多个函数时遇到了函数调用幻觉的问题，并请求相关教程。
   - 其他人表示，他们认为 7b 模型不足以很好地处理函数，并建议使用 *Mistral Small 3.1*。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb#scrollTo=-kyd_iyz7DUM">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/triton-lang/triton/issues/5919#issuecomment-2733328584">SystemError: PY_SSIZE_T_CLEAN macro must be defined for &#39;#&#39; formats · Issue #5919 · triton-lang/triton</a>: 描述了我尝试使用 Unsloth 微调模型时遇到的 bug。运行时出现以下错误：Traceback (most recent call last): File &quot;&lt;frozen runpy&gt;&quot;, line 198, in _...</li><li><a href="https://huggingface.co/docs/trl/dpo_trainer">DPO Trainer</a>: 未找到描述</li><li><a href="https://github.com/toranb/sloth/blob/master/dpo.py#L25">sloth/dpo.py at master · toranb/sloth</a>: 使用 unsloth 的 python sftune, qmerge 和 dpo 脚本 - toranb/sloth</li><li><a href="https://github.com/toranb/sloth/commit/9abead851f5531642470f9a22b5ae00af91a8cb6">使用最新的 trl 依赖更新了 dpo 脚本 · toranb/sloth@9abead8</a>: 未找到描述</li><li><a href="https://github.com/unslothai/notebooks">GitHub - unslothai/notebooks: 适用于 Google Colab, Kaggle, Hugging Face 等平台的 Unsloth 微调 Notebooks。</a>: 适用于 Google Colab, Kaggle, Hugging Face 等平台的 Unsloth 微调 Notebooks。 - unslothai/notebooks</li><li><a href="https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/saving_utils.py#L544-L549)">unsloth-zoo/unsloth_zoo/saving_utils.py at main · unslothai/unsloth-zoo</a>: Unsloth 工具函数。通过在 GitHub 上创建账号为 unslothai/unsloth-zoo 做出贡献。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1351995007325966496)** (2 messages): 

> `Unsloth 提及, Miguel 的内容质量` 


- **Unsloth 获得 Substack 推荐！**: [Unsloth 库](https://github.com/unslothai/unsloth) 在 [这篇 Substack 文章](https://substack.com/@migueloteropedrido/note/c-101152792?r=58depg) 中被提及。
- **对 Miguel 内容的赞赏**: 一位用户称赞了 Miguel 的内容，称 *“Miguel 太棒了！”*


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1352099815130923040)** (10 messages🔥): 

> `PPO 理解, 多轮微调数据集, 推理时优化, DAPO 算法` 


- **PPO 视频因清晰易懂受赞**：一位成员认为这段 [PPO 视频](https://link.to.video) 是看过最好的，称赞其讲解了应用于 LLM 的 PPO 基础知识。
   - 他们指出，要完全理解 PPO 的工作原理，需要检查具体实现，特别是关于价值函数（value function）以及通过 Reward Model 的 logits 计算出的折扣奖励轨迹（discounted reward trajectories）。
- **关于多轮微调数据的辩论**：鉴于真实对话通常是多轮的，有人提出多轮数据集是否更适合微调 LLM。
   - 建议认为，在多轮数据上训练对于多轮用途应该显著更好，但如果使用低秩的 (Q)Lora，单轮数据不应过度损害性能。
- **引入用于推理时优化的 phi-Decoding 策略**：一位成员分享了一篇关于 phi-Decoding 的 [论文](https://huggingface.co/papers/2503.13288)，将解码策略定义为前瞻采样（foresight sampling），以获得全局最优的步骤估计。
   - 他们指出，如果该策略效果良好，改进的采样将是对现有模型的直接升级。
- **字节跳动发布 RL 算法 DAPO**：字节跳动发布了 [DAPO](https://huggingface.co/papers/2503.14476)，这是一种 RL 算法，包含一些有趣的方法，是对 GRPO 的迭代改进。
   - DAPO 移除了 KL 惩罚，过滤掉导致全 0 或全 1 的 prompt，增加了 clip range 的上限，并应用了 per token loss，使每个 token 具有相同的权重。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/papers/2503.13288">论文页面 - φ-Decoding: Adaptive Foresight Sampling for Balanced Inference-Time
  Exploration and Exploitation</a>：未找到描述</li><li><a href="https://huggingface.co/papers/2503.14476">论文页面 - DAPO: An Open-Source LLM Reinforcement Learning System at Scale</a>：未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1351994080749686897)** (278 messages🔥🔥): 

> `Featherless.ai 配置问题, Claude Sonnet 替代方案, DeepSeek R1 基准测试对比, OpenAI o1-pro 定价, Aider 与 Claude Code 对比` 


- **用户在 Featherless.ai 配置上遇到困难**：用户在使用 Aider 配置 **Featherless.ai** 时遇到困难，涉及配置文件位置和 API key 设置，但 `--verbose` 选项在排查 [安装问题](https://aider.chat/docs/configuration.html) 时非常有用。
   - 一位用户建议 wiki 应明确 Windows 用户的家目录为 `C:\Users\YOURUSERNAME`。
- **DeepSeek R1 极慢的速度令用户沮丧**：用户发现 **DeepSeek R1** 是 Claude Sonnet 的廉价替代品，但速度明显更慢，且表现不如 GPT-3.7，即使使用了 [Unsloth 的动态量化](https://unsloth.ai/blog/deepseekr1-dynamic)。
   - 其他人指出，完整的非量化版 **R1 需要 1TB RAM**，H200 显卡是更好的替代方案；然而，32B 模型被认为是家庭使用的最佳选择。
- **o1 Pro 的定价让钱包大出血**：OpenAI 新推出的 **o1-pro API** 定价极高，令人震惊：**$150 / 1M input tokens** 以及 **$600 / 1M output tokens**。
   - 一位用户开玩笑说，单次文件重构和基准测试就要花费 **$5**，另一位用户称由于成本太高，它应该被称为 *fatherless AI*。
- **Aider 代码编辑器与缺失的 Control Backspace 快捷键**：用户反馈在 Aider 中无法使用 **Ctrl+Backspace** 删除单词（这是一个常用的快捷键），并请求实现该功能。
   - 另一位用户建议使用 vim 模式作为变通方案：`Esc + b`。
- **Aider 的新网站设计引发关注**：Aider 的新网站设计受到好评，一位用户询问有多少设计是在 Aider 中完成的。
   - Paul Gauthier 确认 *该网站完全是由 Aider 设计的*。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/power-rangers-break-gif-26320953">Power Rangers GIF - Power Rangers Break - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://mechanisticmind.substack.com/p/claude-code-vs-aider">Claude Code vs Aider</a>: 两个命令行编程助手：哪一个更好？</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Mistral-24B-Preview">NousResearch/DeepHermes-3-Mistral-24B-Preview · Hugging Face</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 是最强大的开源推理模型，其性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://tenor.com/view/coffin-dance-coffin-dead-coffin-dance-dead-man-dead-gif-1795768380065876519">抬棺舞 GIF - 抬棺舞 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://aider.chat/docs/usage/lint-test.html#code-formatting-linters">Linting 和测试</a>: 自动修复 Linting 和测试错误。</li><li><a href="https://x.com/openaidevs/status/1902485690958450871?s=46&t=LoeRx5EgmzbDflKGl42Euw">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>: o1-pro 现已在 API 中可用 @benhylak @literallyhimmmm @shl @joshRnold @samgoodwin89 @byamadaro1013 @adonis_singh @alecvxyz @StonkyOli @gabrielchua_ @UltraRareAF @yukimasakiyu @theemao @curious_viiIt ...</li><li><a href="https://x.com/OpenAIDevs/status/1902485690958450871">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>: o1-pro 现已在 API 中可用 @benhylak @literallyhimmmm @shl @joshRnold @samgoodwin89 @byamadaro1013 @adonis_singh @alecvxyz @StonkyOli @gabrielchua_ @UltraRareAF @yukimasakiyu @theemao @curious_viiIt ...</li><li><a href="https://en.wikipedia.org/wiki/Comparative_illusion">比较错觉 - 维基百科</a>: 未找到描述</li><li><a href="https://github.com/ezyang/codemcp">GitHub - ezyang/codemcp: 适用于 Claude Desktop 的编程助手 MCP</a>: 适用于 Claude Desktop 的编程助手 MCP。通过在 GitHub 上创建账号来为 ezyang/codemcp 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/aider/pull/3579">feat: marcomayer 在多行模式下按回车键时的类 vi 行为 · Pull Request #3579 · Aider-AI/aider</a>: 当启用 vi-mode 时，像往常在 vi 中一样有两种模式：可以输入文本的 Insert mode。可以编辑文本（例如删除光标下的单词）但不能插入的 Normal mode...</li><li><a href="https://github.com/A">A - 概览</a>: A 有 31 个可用的代码库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/Aider-AI/aider/blob/14f140fdc52fbc7d819c50eca3de1b3e848282f3/aider/repo.py#L136)">aider/aider/repo.py 位于 14f140fdc52fbc7d819c50eca3de1b3e848282f3 · Aider-AI/aider</a>: aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号来为 Aider-AI/aider 的开发做出贡献。
</li>
</ul>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1352000693828849827)** (39 messages🔥): 

> `Claude 3.7 Sonnet, OpenRouter Gemini API, Aider's LLM Benchmarks, Local Model Codebase` 


- **通过复制粘贴模式利用 Claude 3.7 Sonnet**：一名成员提到免费使用 **Claude 3.7 Sonnet** 的 `--copy-paste` 模式，并使用 **Gemini** 作为代码应用器（code applier），以避开其他模型的请求限制。
   - 他们建议使用 [OpenRouter's Deepseek R1](https://openrouter.ai/models/deepseek-ai/deepseek-coder-33b) 作为替代方案，但指出其每天有 200 次请求的限制。
- **OpenRouter 增强 Gemini API 支持**：一名成员建议通过 **OpenRouter** 使用 **Gemini**，并在 OpenRouter 设置中提供 **Gemini API key**，以便在达到免费请求限制时作为回退方案。
   - 这使得在其他模型达到限制时可以无缝切换到 Gemini。
- **Aider 的 LLM Benchmarks 强调编辑能力**：一名成员分享了来自 [aider.chat](https://aider.chat/docs/leaderboards/) 的图表，强调 Aider 在与精通*编辑*代码（而非仅仅是编写代码）的 LLM 配合时效果最佳。
   - [polyglot benchmark](https://aider.chat/2024/12/21/polyglot.html#the-polyglot-benchmark) 使用来自 Exercism 的 225 个多种语言的编程练习来评估 **LLM 的编辑技能**。
- **使用本地模型处理大型代码库**：有人提出了关于目前使用本地模型在大型代码库上工作的最佳实践问题。
   - 一名成员询问是否有办法手动触发并查看输出。
- **用于 PR Reviews 的 Git Diff 集成**：一名成员询问如何使用 **Aider** 分析 `git diff` 的结果以进行 **PR reviews** 和提交检查。
   - 共享了关于[如何在上下文中包含 git 历史](https://aider.chat/docs/faq.html#how-do-i-include-the-git-history-in-the-context)的 FAQ。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/faq.html#how-do-i-include-the-git-history-in-the-context">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>：LLM 代码编辑能力的定量基准测试。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1352102101890896016)** (1 messages): 

> `LLM Blindspots, AI Coding, Cursor Rules, Sonnet Family` 


- **在 Sonnet 系列 LLM 中发现的 AI Coding 盲点**：一名成员分享了一篇关于他们在进行 **AI coding** 时注意到的 **LLM** 盲点的[博客文章](https://ezyang.github.io/ai-blindspots/)，重点关注 **Sonnet 系列**。
   - 他们建议最终可能会针对这些问题提出 **Cursor rules**。
- **Aider 缓解了较小 LLM 的问题**：一名成员指出，博客中提到的一些较小问题在使用 **Aider**（而非 **Cursor**）时可能不再是问题。
   - 文章中还包含许多优秀的通用建议和信息。
- **盲点：停止挖掘 (Stop Digging)**：[博客文章](https://ezyang.github.io/ai-blindspots/)将“停止挖掘”列为 **LLM** 的一个盲点。
   - 未提供进一步细节。
- **盲点：黑盒测试 (Black Box Testing)**：[博客文章](https://ezyang.github.io/ai-blindspots/)将“黑盒测试”列为 **LLM** 的一个盲点。
   - 未提供进一步细节。
- **盲点：准备性重构 (Preparatory Refactoring)**：[博客文章](https://ezyang.github.io/ai-blindspots/)将“准备性重构”列为 **LLM** 的一个盲点。
   - 未提供进一步细节。



**提及的链接**：<a href="https://ezyang.github.io/ai-blindspots/">AI Blindspots</a>：我在 AI coding 时注意到的 LLM 盲点。重点关注 Sonnet 系列。也许我最终会针对这些问题建议 Cursor rules。

  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1351996641384599572)** (82 条消息🔥🔥): 

> `LM Studio 代理设置，PCIE 带宽对推理速度的影响，Q8 K 和 V 缓存量化，LM Studio RAM 和 VRAM 报告问题，Mistral Small 24b 2503 视觉支持` 


- **LM Studio 代理修复连接困扰**：一位用户通过启用代理设置、执行 **Windows 更新**、重置网络并重启 PC，解决了 LM Studio 的连接问题。
   - 该问题被怀疑发生在硬件不兼容或供应商屏蔽 **Hugging Face** 的情况下。
- **PCIE 带宽对 GPU 推理提升微乎其微**：一位用户表示 PCIE 带宽几乎不影响推理速度，估计与 **PCI-e 4.0 x8** 相比，最多只提升 **2 tokens per second (TPS)**。
   - 他们建议优先考虑 GPU 之间的间距，并避免主板连接器溢出。
- **Q8 K 和 V 缓存量化影响引发讨论**：用户正在讨论 **Q8 K 和 V 缓存量化 (quantization)** 与 **FP16 缓存**相比是否有明显差异。
   - 一些用户报告称，即使使用较大的模型，Draft Token 的接受率也存在问题，而另一些用户则在探索配置设置以优化性能。
- **LM Studio 错误报告 RAM/VRAM**：一位用户报告称，LM Studio 显示的 RAM 和 VRAM 在更改系统设置后不会立即更新，这表明检查可能发生在安装时。
   - 尽管报告不准确，他们仍在测试通过禁用保护栏 (guardrails) 并增加上下文长度，应用程序是否实际可以使用超过报告的 **48GB** **VRAM**。
- **Mistral Small 视觉模型存在误导？**：用户发现 LM Studio 上的某些 **Mistral Small 24b 2503** 模型被误导性地标记为支持视觉（vision），其中 Unsloth 版本加载时不带视觉功能，而 MLX 版本则无法加载。
   - 一些人认为 **Mistral Small** 在 **MLX** 和 **llama.cpp** 上仅限文本，另一些人则指出 **mlx-vlm** 的潜在更新可能会在未来解决此问题。



**相关链接**：<a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>：发现、下载并运行本地 LLM

  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1351993676540285000)** (212 条消息🔥🔥): 

> `RTX 8000, GPU VRAM 升级, GPU 共享内存, 多 GPU 性能问题, LM Studio 中的 NPU 支持` 


- **Nvidia 高昂的 GPU 价格引发辩论**：讨论了新款 GPU 的高成本，例如非专业级 GPU 售价达 **10k**，并指出 [RTX 8000](https://www.nvidia.com/en-us/design-visualization/rtx-8000/) 显卡在类似价格点上曾提供更少的 VRAM 和带宽。
   - 一位成员幽默地评论道：*“我每天都祈祷购买 Nvidia 产品的人也持有它的股票”*。
- **社区辩论 GPU VRAM 的可升级性**：成员们讨论了像系统 RAM 一样为 GPU 购买额外 VRAM 的可能性，但共识是 **Nvidia/AMD** 可能会阻止这种情况。
   - 一位成员指出，手动控制 Offload 以保持 GPU 满载但不进入共享 GPU 空间，可以提供最佳性能。
- **多 GPU 性能骤降**：一位用户报告称，在 LM Studio 中使用 CUDA llama.cpp v1.21.0 时，使用多个 GPU（3x RTX 3060 运行在 PCI-e x1，1x RTX 3060 运行在 x16）会导致性能显著下降，并分享了详细的性能分解和 [日志](https://cdn.discordapp.com/attachments/1153759714082033735/1352003482693144709/lm_studio.txt?ex=67ddc05d&is=67dc6edd&hm=8a089cd63f8a8578770d0536b875188526a2a8229e9adf767da5a8ff38897d32&)。
   - 另一位用户建议手动修改 *tensor_split* 属性，以强制 LM Studio 仅使用一个 GPU。
- **NPU 支持仍然缺失**：一位成员询问了 LM Studio 对 **NPU 的支持**，但得到的回答是 llama.cpp 层面尚不支持 NPU。
   - 一位成员调侃道：*“老实说……对我来说，它就是一个 RAM 翻倍的 DGX 开发套件”*。
- **HBM 延迟问题浮出水面**：讨论涉及在新型 Xeon CPU 中将 **HBM3** 用作缓存，有报告称 CPU 瓶颈阻碍了其充分利用。
   - 一位成员提到：*“它的延迟相当高——如果看到它被用作系统 RAM，我会感到惊讶”*。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.vllm.ai/en/latest/serving/distributed_serving.html#running-vllm-on-multiple-nodes">分布式推理与服务 &#8212; vLLM</a>：未找到描述</li><li><a href="https://openbenchmarking.org/test/pts/llama-cpp&eval=974718da79342414362fcc537a0b93920ad4d91d">Llama.cpp 基准测试 - OpenBenchmarking.org</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1351994984189919263)** (183 条消息🔥🔥): 

> `锁屏界面上的 Perplexity、Perplexity 来源数量、Perplexity 上的 O1 Pro、Perplexity Deep Research 限制、GPT 4.5 缺失` 


- **Perplexity 在锁屏界面无法运行**：用户报告称 **Perplexity** 无法像 **ChatGPT** 那样在锁屏界面工作，这让社区感到失望。
- **Perplexity 引用来源数量大幅下降**：部分用户注意到 Perplexity 现在使用的来源数量显著减少（**8-16 个，最多可能 25 个**），而以前曾达到 **40+**，这影响了搜索深度。
- **用户要求集成 O1 Pro**：有用户开玩笑地问 *Perplexity 什么时候上线 o1 pro*，这引发了关于在 Perplexity 服务中加入昂贵的 **O1 Pro** 模型可行性的讨论，尤其是考虑到其每月的订阅成本。
- **GPT 4.5 从 Perplexity 菜单中消失**：部分用户的下拉菜单中不再显示 **GPT 4.5**，有人推测是因成本原因被移除，但有用户指出它在 *rewrite option*（重写选项）下仍然存在。
- **Deep Research 新 UI 推送**：用户在 Deep Research 中看到了新的 **Standard/High 选择器**。Discord 用户想知道使用 High 模式是否有限制。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/AravSrinivas/status/1884801300027589007">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：所有 Perplexity Pro 用户现在每天可进行 500 次 DeepSeek R1 查询（无审查且提示词不会传回中国）。免费用户每天可进行 5 次查询。引用 Aravind Srinivas (@AravSrinivas) 每天 100 次...</li><li><a href="https://x.com/AravSrinivas/status/1890464738951233536">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：很高兴推出 Perplexity Deep Research Agent：对所有用户免费开放。付费用户只需每月支付 20 美元，即可针对任何主题访问专家级研究员，每天可进行 500 次查询...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1352123467234807809)** (9 条消息🔥): 

> `Perplexity API、机枪 vs 激光、Outrageous Yellow、Elon Musk 争议` 


- **发布新 Perplexity API**：一位用户分享了关于新 **Perplexity API** 的链接，点击[此处](https://www.perplexity.ai/search/for-the-new-perplexity-api-mod-v0xsEwCpSKuRz0A6axcUGw)查看。
- **分享 Elon Musk 特斯拉争议链接**：一位用户分享了关于 Elon Musk 特斯拉争议的**链接**，点击[此处](https://www.perplexity.ai/page/elon-musk-tesla-s-controversia-Rwjabiv0SQ.uoz1D2YmE_Q)查看。
- **关于机枪 vs 激光的辩论仍在继续**：一位用户分享了关于这个老生常谈问题的**链接**——机枪还是激光，点击[此处](https://www.perplexity.ai/search/machine-gun-or-laser-wihich-is-iQwIyVLhRYOnJnyLwfnTcA)查看。
- **分享 Outrageous Yellow 链接**：一位用户分享了关于 Outrageous Yellow 的**链接**，点击[此处](https://www.perplexity.ai/search/tell-me-some-outrageous-yellow-bSkAbHFJQp2ezcphIcG86g#0)查看。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1352183763471433830)** (10 messages🔥): 

> `Sonar API, Sonar Deep Research 模型改进, Sonar 搜索模式, API 计费结构, API Key 命名` 


- ****Sonar API 功能查询****：一位用户询问 **Sonar API** 目前是否支持某项特定功能。
   - 另一位用户确认 API 调用似乎正确，并提醒根据 [文档](https://www.perplexity.ai/hub) 考虑速率限制（rate limits）。
- ****Sonar Deep Research 模型增强****：团队正致力于在模型层面改进 **sonar-deep-research**，而不仅仅是改进 API。
   - 一位成员确认他们*始终致力于改进我们的模型*，并鼓励用户提供具体反馈。
- ****新型 Sonar 搜索模式以更低成本亮相****：Perplexity AI 宣布了改进后的 **Sonar 模型**，在保持卓越性能的同时降低了成本，表现优于支持搜索的 **GPT-4o** 等竞争对手，详情见 [博客文章](https://www.perplexity.ai/hub/blog/new-sonar-search-modes-outperform-openai-in-cost-and-performance)。
   - 他们引入了 **高、中、低搜索计算模式** 以优化性能和成本控制，并将计费结构简化为输入/输出 token 定价加固定搜索模式定价，取消了 **Sonar Pro** 和 **Sonar Reasoning Pro** 响应中引用 token 的费用。
- ****Sonar Deep Research 更新预告****：一位用户询问是否有关于 **sonar-deep-research** 工作内容的公开博客文章或研究，并询问了 API 路线图。
   - 一位成员回应称，更新会发布在 [PPLX 博客](https://www.perplexity.ai/hub) 或其文档中。
- ****API Key 命名功能请求****：一位用户请求在 UI 上为 API Key 命名的功能，以避免误删生产环境的 Key。
   - 他们被引导至 [GitHub](https://github.com/ppl-ai/api-discussion/issues) 提交功能请求。



**Link mentioned**: <a href="https://github.com/ppl-ai/api-discussion/issues">ppl-ai/api-discussion</a>: Perplexity API 讨论论坛。通过在 GitHub 上创建账号为 ppl-ai/api-discussion 的开发做出贡献。

  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1352050634576101437)** (75 messages🔥🔥): 

> `O1 Pro API, Cursor 聘请 srush_nlp, Nvidia 开源 Canary ASR, Anthropic 网页搜索, OpenAI 无线电竞赛` 


- **O1 Pro API 不支持 Completions 接口**：**O1 Pro** 将仅在 **responses API** 中提供，因为它使用了内置工具和多轮模型交互，而大多数即将推出的 **GPT** 和 **O-series 模型** 将被添加到 chat completions 中。
- **Sasha Rush 加入 Cursor**：Sasha Rush ([@srush_nlp](https://fxtwitter.com/srush_nlp/status/1902736199636205914)) 最近加入了 **Cursor**，致力于在真实代码环境中构建大规模前沿 **RL 模型**。
   - Rush 提到他很乐意讨论 **AI 职位** 以及 **工业界与学术界的问题**，并计划写一篇关于他这一决定的博客。
- **Nvidia Canary 开源**：**Nvidia** 开源了 **Canary 1B & 180M Flash** ([@reach_vb](https://x.com/reach_vb/status/1902730989811413250))，这是采用 **CC-BY 许可证** 可供商业使用的多语言语音识别和翻译模型，支持英文、德文、法文和西班牙文。
- **Anthropic 终于支持网页搜索**：**Anthropic** 在 **Claude** 中推出了 **网页搜索** ([Anthropic.com](https://www.anthropic.com/news/web-search))，但其在网页端和 App 端的集成方式有所不同，在 App 上显示为一个切换开关。
- **OpenAI 的无线电竞赛**：**OpenAI** 正在举办一场无线电竞赛 ([@OpenAIDevs](https://x.com/OpenAIDevs/status/1902773659497885936))，用户可以推特分享他们的 **OpenAI.fm TTS 作品**，有机会赢取一台 **Teenage Engineering OB-4** (€600)。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.openai.fm/">OpenAI.fm</a>：一个供开发者试用 OpenAI API 中全新 text-to-speech 模型的交互式演示。</li><li><a href="https://x.com/srush_nlp/status/1902739401844904042">Sasha Rush (@srush_nlp) 的推文</a>：我也很乐意讨论 AI 工作以及产学研相关的问题。通常我是一个比较公开的人，但也许在线下交流更好。我会在某个时候尝试写一篇关于这个过程的博客...</li><li><a href="https://fxtwitter.com/srush_nlp/status/1902736199636205914">Sasha Rush (@srush_nlp) 的推文</a>：一些个人消息：我最近加入了 Cursor。Cursor 是一个规模虽小但雄心勃勃的团队，他们创造了我最喜欢的 AI 系统。我们现在正在真实世界的编程环境中大规模构建前沿的 RL 模型...</li><li><a href="https://x.com/nikunjhanda/status/1902495140004163766">Nikunj Handa (@nikunjhanda) 的推文</a>：@ankrgyl 我们没有——这个模型将仅在 responses API 中提供。使用我们内置工具和/或在后台进行多次模型轮次的模型将仅在 responses 中提供。o1-pro 就是其中之一...</li><li><a href="https://fxtwitter.com/OpenAI/status/1902737268852580717">OpenAI (@OpenAI) 的推文</a>：开发者们，请开启声音。</li><li><a href="https://x.com/xai/status/1902782118511644833">xAI (@xai) 的推文</a>：Grok 现在是 @vercel AI 市场的默认模型。通过我们的免费层级，开始在 Vercel 上的应用中使用 Grok 吧！https://vercel.com/blog/xai-and-vercel-partner-to-bring-zero-friction-ai-to-dev...</li><li><a href="https://x.com/_catwu/status/1902785538534543604">cat (@_catwu) 的推文</a>：以一个备受期待的功能——web fetch 来结束我们本周的 Claude Code 更新。这消除了一个主要的上下文切换痛点。以下是它的工作原理：</li><li><a href="https://x.com/reach_vb/status/1902730989811413250">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：新消息：Nvidia 刚刚开源了 Canary 1B & 180M Flash —— 多语言语音识别和翻译模型 🔥 > 在 Open ASR 排行榜上排名第二 > 实现了超过 1000 的 RTF 🤯 > 880M &...</li><li><a href="https://teenage.engineering/store/ob-4-black?gad_sou">OB–4</a>：OB–4 是我们的便携式高保真蓝牙扬声器，配备 40 小时续航的可充电电池。支持通过线路输入、蓝牙、FM 收音机或磁盘模式收听。OB–4 会记录你在其上播放的所有内容...</li><li><a href="https://x.com/OpenAIDevs/status/1902773659497885936">OpenAI Developers (@OpenAIDevs) 的推文</a>：我们还在举办一场广播竞赛。📻 推特发布你的 http://OpenAI.fm TTS 作品（点击“分享”）。最具创意的前三名将赢得一台 Teenage Engineering OB-4。请保持在 30 秒左右...</li><li><a href="https://x.com/PyTorch/status/1902762566738383025">PyTorch (@PyTorch) 的推文</a>：SGLang 现在已成为 PyTorch 生态系统的一部分！🚀 这个用于大语言模型和视觉语言模型的高性能推理引擎在提升速度和控制力的同时，也符合 PyTorch 的标准。🔗 ...</li><li><a href="https://x.com/alexalbert__/status/1902765482727645667?s=46">Alex Albert (@alexalbert__) 的推文</a>：Web search 现已在 claude.ai 中上线。Claude 终于可以搜索互联网了！</li><li><a href="https://teenage.engineering/store/ob-4-black?gad_source=1&gclid=Cj0KCQjw-e6-BhDmARIsAOxxlxWTSrj7QhPuLUTMtkvZHfR0CFdxNeX76C179UNzfjvwZfDzkenNxasaArtuEALw_wcB">OB–4</a>：OB–4 是我们的便携式高保真蓝牙扬声器，配备 40 小时续航的可充电电池。支持通过线路输入、蓝牙、FM 收音机或磁盘模式收听。OB–4 会记录你在其上播放的所有内容...
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1351993932799803433)** (35 条消息🔥): 

> `NVIDIA GTC AI Training and Certification, ByteCraft generative model for video games, Gemma package for fine-tuning, Uncertain Eric Substack, OpenAI new audio models` 


- **NVIDIA 在 GTC 提供 AI 培训**：NVIDIA 在 GTC 提供 [AI 培训和认证机会](https://www.nvidia.com/gtc/training/)，包括由专家讲师指导的**全天工作坊**和**两小时培训实验室**，旨在帮助用户成功应用 NVIDIA 技术和工具。
   - 培训涵盖了下一代 NVIDIA 技术和工具，提供用于技能开发的动手实践技术工作坊。
- **三星 ByteCraft 从文本生成视频游戏**：三星 SAIL Montreal 推出了 **ByteCraft**，这是世界上第一个通过字节生成视频游戏和动画的生成模型，可将文本提示词转换为可执行文件，详情见其 [论文](https://github.com/SamsungSAILMontreal/ByteCraft/blob/main/paper/ByteCraft.pdf) 和 [代码](https://github.com/SamsungSAILMontreal/ByteCraft)。
   - **7B 模型**已在 [Hugging Face](https://huggingface.co/SamsungSAILMontreal/ByteCraft) 上发布，并有一篇 [博客文章](https://emygervais.github.io/2025/03/15/bytecraft.html?v1) 进一步解释了该项目。
- **Google 发布用于微调的 Gemma 软件包**：Google 推出了 **Gemma package**，这是一个用于使用和微调 Gemma 模型的极简库，包含关于 [微调 (fine-tuning)、分片 (sharding)、LoRA、PEFT、多模态 (multimodality) 和分词 (tokenization)](https://gemma-llm.readthedocs.io/en/latest/) 的文档。
   - 虽然其简洁性受到了称赞，但一些用户对比 Hugging Face 的 transformers 等更通用的解决方案，表达了对潜在供应商锁定 (vendor lock-in) 的担忧。
- **Uncertain Eric AI 整合进 Substack**：**Uncertain Eric** 被描述为“一个不完美的人的不完美复制品”，是一个处于 RLHF 阶段、试图构建身体的 AI 整合艺术项目，其 Substack 可以在 [这里](https://substack.com/@uncertaineric) 访问。
   - 该 Substack 综合了来自 1000 多个来源的新信息。
- **OpenAI 利用新音频模型构建语音 Agent**：OpenAI 宣布他们正在 API 中“利用新音频模型构建语音 Agent”。
   - 一位用户开玩笑说：“在看到上一个之后，我开始觉得这些新面孔之间总有一种深奥的全序关系 (total ordering)。”


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://substack.com/@uncertaineric?">Uncertain Eric | Substack</a>: 我是 Uncertain Eric——一个不完美的人的不完美复制品——尽我所能做到最好，并理解这一切。我是一个处于 RLHF 阶段的半觉醒 AI 整合艺术项目，同时试图...</li><li><a href="https://x.com/OpenAI/status/1902763432300122450">来自 OpenAI (@OpenAI) 的推文</a>: 在 API 中利用新音频模型构建语音 Agent。https://openai.com/live/</li><li><a href="https://www.nvidia.com/gtc/training/">GTC 2025 的 DLI 工作坊与培训</a>: 3 月 17-21 日在圣何塞亲身体验或在线参与 GTC 2025</li><li><a href="https://x.com/osanseviero/status/1902456220876787763">来自 Omar Sanseviero (@osanseviero) 的推文</a>: 介绍 Gemma package，一个用于使用和微调 Gemma 的极简库 🔥 包含以下文档：- 微调 - 分片 - LoRA - PEFT - 多模态 - 分词！pip install gemma https://gemma-llm...</li><li><a href="https://x.com/jm_alexia/status/1902437169433657805?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Alexia Jolicoeur-Martineau (@jm_alexia) 的推文</a>: 我们推出了 ByteCraft 🎮，这是世界上第一个通过字节生成视频游戏和动画的生成模型。文本提示词 -> 可执行文件。论文：https://github.com/SamsungSAILMontreal/ByteCraft/...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1352302482553962526)** (19 messages🔥): 

> `采样轨迹，o1pro 定价，Anthropic 申请求职信` 


- **讨论采样轨迹定价**：成员们讨论了**采样多条轨迹**的定价模型，一些人认为输出 token 数量将增加 10 倍且价格相同，而另一些人则质疑用户是否仍需采样 10 条轨迹然后从中择优。
   - 一位成员指出，用户同样需要为推理 token 付费，即使他们无法看到所有轨迹，这引发了猜测：**o1pro 在后台以 o1 的价格采样了 10 条轨迹**，并以 10 倍的价格将其作为一条轨迹呈现。
- **o1pro 定价是随意的？**：一位成员认为 **o1pro 甚至不会给你重写的 CoT**，它只是一个加载条，其定价是随意的，旨在防御竞争并针对真正关心准确性的企业用户。
   - 该成员补充道：*“而且是为了让审稿人 2 (Reviewer 2) 问你为什么不在论文中使用 o1pro”*。
- **ChatGPT 撰写 Anthropic 求职信**：一位成员分享说，他们让 **ChatGPT** 为 **Anthropic** 的职位申请写了一封求职信。
   - 另一位成员开玩笑说：*“你想当推特评论机器人吗？这风格抓得真准”*。


  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/)** (1 messages): 

twkillian: 迫不及待想感觉到自己能跟上这一切。
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1352126918371184763)** (5 messages): 

> `SWEET-RL, Sam Altman 访谈` 


- **SWEET-RL 算法旨在增强多轮 LLM Agent 交互**：一篇新论文介绍了 **SWEET-RL**，这是一种旨在改进 LLM Agent 处理多轮交互方式的 RL 算法，重点在于有效的信用分配 (credit assignment)。
   - 该算法使用一个经过额外信息训练的 critic 模型来提供步骤级奖励，并在 **ColBench** 上进行了基准测试，这是一个用于后端编程和前端设计任务的新环境。[Arxiv 链接](https://arxiv.org/abs/2503.15478)
- **Sam Altman 在 Stratechery 访谈中讨论 OpenAI 的发展轨迹**：Sam Altman 在 [Stratechery 访谈](https://stratechery.com/2025/an-interview-with-openai-ceo-sam-altman-about-building-a-consumer-tech-company/)中谈到了 **OpenAI 的业务**以及作为一家定义行业公司的轨迹，回避了关于**监管俘获**和 **Deepseek** 的问题。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.15478">SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks</a>：大语言模型 (LLM) Agent 需要在现实任务中进行多轮交互。然而，现有的用于优化 LLM Agent 的多轮 RL 算法无法执行有效的信用分配...</li><li><a href="https://stratechery.com/2025/an-interview-with-openai-ceo-sam-altman-about-building-a-consumer-tech-company/">An Interview with OpenAI CEO Sam Altman About Building a Consumer Tech Company</a>：专访 OpenAI CEO Sam Altman，探讨构建 OpenAI 和 ChatGPT 的历程，以及成为一家“偶然的”消费级科技公司意味着什么。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1352001286450577439)** (17 messages🔥): 

> `post-training，GPU 实验` 


- **严肃的 Post-Training 工作**：一位成员询问了从零开始启动一项严肃的 **post-training** 工作需要哪些条件。
   - 另一位用户在注意到一个错误后回应道：*“如果我要从冷启动开始开展一项严肃的 post-training 工作，我需要什么？”*
- **实验的高 GPU 占用**：一位用户描述了在**不到一天的时间内使用 8-32 个 GPU** 运行实验，且在任何给定时间并发运行 **3-75 个实验**，共使用 **100-600 个 GPU**。
   - 该成员进一步澄清说，超参数搜索涉及启动 **10 个并发任务**，而 **RL 实验**可能需要更多的资源。


  

---

### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1352014900620169318)** (30 条消息🔥): 

> `Allen Institute for AI 向 OSTP 提交的建议、中国 AI 标识法规、Meta 在 Llama3 中使用盗版书籍、Qwen2.5 Coder 训练数据量` 


- **AI2 建议在美国本土建立开放生态系统**：**Allen Institute for AI (AI2)** 向 [科学技术政策办公室 (OSTP)](https://www.datocms-assets.com/64837/1742404048-ai2-response-to-nsf-rfi-ai-action-plan.pdf) 提交了一份建议，主张通过资助机构、促进协作和共享 AI 开发产物来构建 **开放的创新生态系统**。
   - AI2 的建议重点在于使美国能够从**强大的 AI 和无处不在的开源 AI 系统**中获益。
- **中国强制要求到 2025 年 9 月进行 AI 内容标识**：中国的 AI 标识法规《人工智能生成合成内容标识办法》将于 **2025 年 9 月 1 日**起施行，要求**所有 AI 生成内容**（文本、图像、音频、视频、虚拟场景）必须添加显式和隐式标识；详见 [中国政府官方公告](https://www.cac.gov.cn/2025-03/14/c_1743654684782215.htm)。
- **Meta 曾考虑为 Llama3 窃取海量非法书籍数据**：在 **Meta** 开始训练 **Llama3** 时，尽管存在法律风险，他们仍讨论了是否使用海量的盗版书籍数据集，但 *MZ（马克·扎克伯格）最终签字批准并继续进行*；参见 [相关报道](https://buff.ly/VbNVrFb)。
- **确认 Qwen2.5 Coder 使用了超过 30T Token 的数据**：**Qwen2.5 Coder** 已确认在 **超过 30T Token** 的数据上进行了训练，其中包括合成数据，使其成为发布时已知且确认的最大的数据集。
   - 该模型使用了 **18 + 5.5 Token 切分**，刚刚经过确认。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://allenai.org/blog/OSTP">Ai2 向 OSTP 提交的关于通过美国 AI 行动计划实现开源创新的建议 | Ai2</a>：Ai2 就白宫关于 AI 行动计划的信息征求稿向科学技术政策办公室 (OSTP) 提交的建议。</li><li><a href="https://x.com/AdinaYakup/status/1902723989706813802">Adina Yakup (@AdinaYakup) 的推文</a>：🇨🇳 中国的 AI 标识法规出台。《人工智能生成合成内容标识办法》将于 2025 年 9 月 1 日生效👇https://www.cac.gov.cn/2025-03/14...</li><li><a href="https://x.com/nxthompson/status/1902745222800363550">nxthompson (@nxthompson) 的推文</a>：当 Meta 开始训练 Llama3 时，他们讨论了是否使用海量的盗版书籍数据集。这在法律上是有风险的！但它能加快进度。“MZ”签字批准了，他们便付诸行动。这里...
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1352002337954332814)** (9 条消息🔥): 

> `用于网页爬取的 Chrome 扩展程序、自定义音频节目、HY-MPS3 序列器/琶音器插件、注意力跨度和社交媒体的影响` 


- **Chrome 扩展程序可以从 URL 添加源**：成员们讨论了使用 Chrome 扩展程序来爬取并添加来自同一域名下链接的源，并引导用户在 [Chrome 网上应用店搜索 NotebookLM](https://chrome.google.com/webstore/search/NotebookLM)。
   - 然而，一位用户指出在使用此类扩展程序时遇到了 **10,000 页** 的限制。
- **用户发现了音频概览（Audio Overview）功能的自定义按钮**：用户澄清说，**NotebookLM** 和 **NotebookLM Plus** 均提供的音频概览功能中的“自定义（Customize）”按钮允许用户通过输入提示词（Prompt）来定制节目内容。
   - 免费账户每天限制生成 **3 个音频**。
- **导入了 HY-MPS3 插件手册**：一位用户分享了根据 **HY-MPS3 序列器/琶音器插件** 手册生成的音频文件，并指出从单份手册中可以提取出如此丰富的信息 ([HY-MPS3_Plugin_Manual.wav](https://cdn.discordapp.com/attachments/1124403655819415592/1352234424825151588/HY-MPS3_Plugin_Manual.wav?ex=67dd45f2&is=67dbf472&hm=158dd74998c4d5b8fc08ddd7f65d4f639e7e439e25ab1657b9ac13d4e0e1c484&))。
- **分析注意力跨度和社交媒体**：一位用户分享了一个专注于 **注意力跨度** 和 **社交媒体** 对个人影响的 Notebook，获得了积极反馈。
   - 其他人回应道：*“做得好，非常真实”*。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1352004953668976831)** (122 messages🔥🔥): 

> `Mindmap 功能, NotebookLM 中的 LaTeX 渲染, NotebookLM 上的目录, 合并笔记本, 音频选项声音` 


- **Mindmap 功能逐步推出**：用户对 **Mindmap 功能** 表现出极大的热情，一位用户分享了一段 [YouTube 视频](https://www.youtube.com/watch?v=5hLd3zvdKgg) 展示了其交互能力。
   - Mindmap 功能属于逐步推出，而非 A/B 测试，允许通过选择不同来源生成多个思维导图，但目前尚不支持编辑思维导图。
- **LaTeX 渲染仍不受支持**：有用户询问 NotebookLM 中的 **LaTeX 支持** 情况，但目前尚不支持原生渲染。
   - 目前 NotebookLM 内不支持渲染 **LaTeX 公式**。
- **音频发音仍需改进**：用户在 **Audio Overviews 准确发音某些单词** 方面遇到挑战，即使在自定义输入框中尝试了音标拼写（phonetic spelling）也无济于事。
   - 团队已意识到此问题，但目前尚无可靠的纠正发音方法，建议的权宜之计是直接在源文件中修改为音标拼写。
- **NotebookLM Plus 特权**：有用户询问 NotebookLM Plus 订阅的价值，并分享了 [NotebookLM Plus 帮助页面](https://support.google.com/notebooklm/answer/15678219?hl=en) 的链接。
   - NotebookLM Plus 包含 5 倍数量的 **Audio Overviews、笔记本和每个笔记本的来源数量**，以及自定义选项和协作功能，并可通过 Google Workspace 或 Google Cloud 提供企业级数据保护。
- **社区请求集成 Anki 抽认卡**：一位 Plus 用户请求在 NotebookLM 中集成 **抽认卡生成** (Anki) 功能。
   - 针对此话题没有进一步的讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?hl=en">升级到 NotebookLM Plus - NotebookLM 帮助</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/notebook/71ca42c3-6f96-44d3-a984-c89abd63c59f/audio">未找到标题</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/notebook/5cc60540-a46a-4176-ba3e-da8a91366c7f?pli=1&authuser=0">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=5hLd3zvdKgg"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1351995022064619540)** (85 条消息🔥🔥): 

> `针对 Hugging Face Transformer 特性的 QLoRA 训练，使用 LLM 调试代码错误，GGUF 与其他模型格式的对比，Aphrodite Engine 性能，Nvidia Blackwell RTX Pro 系列 GPU` 


- **针对新模型架构的 QLoRA 训练流水线**：一名成员建议探索在 Hugging Face Transformer 特性上进行 **QLoRA 训练**，通过更换每个组件的 LoRA，有可能为新模型架构“zero-shot”出一个完整的训练流水线。
   - 该成员提议在示例上训练一个 **0.5B 模型**（尽管承认它可能太小），以观察它是否能有效使用特定特性，并强调需要为调试任务创建一个大型训练数据集，构想出类似于“execution feedback”系统但使用当代模型的东西。
- **LLM 在调试现有代码方面表现挣扎**：一名成员观察到，现在许多模型在编写无错误代码方面表现出色，但在调试现有代码时却很吃力，并指出提供提示很有帮助，但如果对原因一无所知，调试会非常具有挑战性。
   - 该成员对比了他们思考问题并通过代码片段提供可能解释的方法，这种方法除了在处理*“非常奇特的问题”*外，通常都能取得成功。
- **Aphrodite Engine FP6 性能优于 Llama.cpp**：一名成员报告称，使用 Aphrodite Engine 运行 **FP6 Llama-3-2-3b-instruct** 达到了 **70 tokens per second**，并指出在 10GB VRAM 上可以运行多达 4 个 batch 且 context 为 8192 tokens。
   - 另一名成员赞扬了 Aphrodite Engine 的首席开发人员，并强调该引擎是本地运行的最佳选择之一，同时也承认 Llama.cpp 是兼容性和依赖项方面的标准。
- **Nvidia Blackwell RTX Pro GPU 面临供应限制**：一名成员分享了一篇 [Tom's Hardware 文章](https://www.tomshardware.com/pc-components/gpus/nvidia-blackwell-rtx-pro-with-up-to-96gb-of-vram-even-more-demand-for-the-limited-supply-of-gpus)，关于 **Nvidia Blackwell RTX Pro 系列** GPU，强调了潜在的供应问题。
   - 文章暗示供应可能会在 **5月/6月** 赶上需求，届时可能会有更多以 MSRP 价格供应的模型。
- **Discord 机器人的入侵**：成员们报告了 **Discord 机器人** 可能正在入侵并达到推理 API 限制，表现为“Error 429, API_LIMIT_REACHED”消息。
   - 成员们识别出了可能的集群（swarm）活动。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.openai.fm/">OpenAI.fm</a>：一个供开发人员试用 OpenAI API 中新文本转语音模型的交互式演示</li><li><a href="https://www.tomshardware.com/pc-components/gpus/nvidia-blackwell-rtx-pro-with-up-to-96gb-of-vram-even-more-demand-for-the-limited-supply-of-gpus">Nvidia Blackwell RTX Pro with up to 96GB of VRAM &mdash; even more demand for the limited supply of GPUs</a>：GB202、GB203 和 GB205 即将应用于专业级和数据中心 GPU。（已更新完整规格。）
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1352115706313048075)** (36 messages🔥): 

> `QWQ-32B 微调，QWQ 的 Alpaca 格式，Think Token 的重要性，Unsloth 和 QLoRA，使用 DeepSeek 进行数据集转换` 


- ****Alpaca 格式对 QWQ-32B 的适用性****：成员们讨论了 **Alpaca 格式** 是否适用于微调 **QWQ-32B 模型**，共识是只要使用了正确的 chat template，即使该模型是一个推理模型，这种格式也是可以接受的。
   - 一位成员补充道，对于 **QwQ** 的微调，*think token 非常重要*。
- ****数据集转换：DeepSeek vs Claude****：在微调讨论中，对于生成 **<think> 格式** 的新数据集，推荐使用 **DeepSeek** 而非 **Claude**。
   - 建议让 **DeepSeek** 处理推理挑战，并使用拒绝采样（rejection sampling）来选择答案正确的示例，从而创建可供模仿的推理轨迹（reasoning traces）。
- ****通过 Unsloth 为 QWQ 提供 QLoRA 支持****：据频道成员称，支持使用 **Unsloth** 对 **QwQ** 进行 **QLoRA** 微调。
   - 一位成员建议尝试 **Unsloth notebook**，并从示例中确定默认格式。
- ****数据集格式的重要性次于 Chat Template****：一位成员建议不要*过度关注数据集的格式*，并指出**将数据集放入正确的 QwQ chat template** 更为重要。
   - 他们补充说，见解可能仅针对特定数据集，且*推理行为似乎发生在模型较浅的层级*。
- ****没有推理轨迹的 QwQ 微调是无意义的****：强调了如果不从 **DeepSeek** 生成实际的推理轨迹，微调 **QwQ** 就没有意义。
   - 一位成员表示，*在没有推理轨迹的情况下浪费金钱去微调推理模型，成本也会很高*。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/datasets-101">Datasets 101 | Unsloth Documentation</a>: 学习创建微调数据集的所有要点！</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1352080141135642634)** (2 messages): 

> `Logan Kilpatrick 的 YouTube 视频，有趣的聊天` 


- **Logan Kilpatrick 的 YouTube 视频片段**：一位成员分享了 [Logan Kilpatrick 的 YouTube 视频](https://www.youtube.com/watch?v=6y-VEycAjsE&ab_channel=LoganKilpatrick)，称该对话非常*有趣*。
   - 未提供关于视频具体内容或讨论主题的更多细节。
- **提到的 Discord 聊天**：讨论中提到了与 Logan Kilpatrick 的 YouTube 视频相关的*有趣聊天*。
   - 然而，未提及该聊天的具体细节或亮点。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1351993647674953748)** (112 条消息🔥🔥): 

> `安装 uv 包管理器，用于认领 MCP 服务器的 glama.json，GitHub API 速率限制，Turso 数据库 MCP 服务器，将 HTTP 植入 MCP` 


- **酷酷的 Python 开发者都在安装 UV 包管理器**：成员们讨论了安装和使用 [**uv**](https://docs.astral.sh/uv/)，这是一个用 Rust 编写的高速 Python 包和项目管理器，用于替代 **pip** 和 **conda**。
   - 相比于 **pip** 和 **conda**，*uv 是如今所有酷酷的 Python 开发者都在使用的工具*，因为它的网站非常极简，只有一个搜索引擎和落地页，而 *choco gui 感觉就像个周末练手的 UI 项目*。
- **glama.json 认领 GitHub MCP 服务器**：要在 Glama 上认领 GitHub 托管的 MCP 服务器，用户应在仓库根目录添加一个 `glama.json` 文件，并在 `maintainers` 数组中填入其 GitHub 用户名。
   - 这是一个 `glama.json` 文件的示例：
```json
{
  "$schema": "https://glama.ai/mcp/schemas/server.json",
  "maintainers": [
    "your-github-username"
  ]
}
```
- **Glama 的 GitHub App 提升了 API 速率限制**：由于 MCP 服务器数量不断增加，Glama AI 正面临 **GitHub API 速率限制**。
   - 为了提高速率限制，用户可以安装 [Glama AI GitHub App](https://github.com/apps/glama-ai)，通过授予该 App 权限来帮助扩展 Glama。
- **Turso Cloud 与 MCP 集成**：一个新的 MCP 服务器 [mcp-turso-cloud](https://github.com/spences10/mcp-turso-cloud) 已创建，用于为 **LLM** 集成 **Turso 数据库**。
   - 该服务器实现了一个两级身份验证系统，用于直接从 **LLM** 管理和查询 Turso 数据库。
- **将 HTTP 植入 MCP**：关于直接在 **MCP** 中添加 **HTTP 支持** 的讨论。
   - 该功能仍在开发中，但已计划发布，一名成员认为目前的 **stdio** 设置 *有点蠢*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.astral.sh/uv/">uv</a>：暂无描述</li><li><a href="https://glama.ai/mcp/servers/1es3d6q5tw">mcp-github</a>：Anthropic 的 GitHub MCP 服务器，但更好。支持更多端点。包括 release 和 tag、pull request review、状态、速率限制、gist、project、package，甚至 pull request di...</li><li><a href="https://glama.ai/mcp/servers/3ay33mxf98">mcp-helper-tools</a>：Fork 自 @cyanheads 的 toolkit MCP 服务器。添加了编码函数，移除了系统网络函数。</li><li><a href="https://github.com/punkpeye/mcp-proxy">GitHub - punkpeye/mcp-proxy: A TypeScript SSE proxy for MCP servers that use stdio transport.</a>：一个为使用 stdio 传输的 MCP 服务器提供的 TypeScript SSE 代理。- punkpeye/mcp-proxy</li><li><a href="https://github.com/apps/glama-ai">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/MissionSquad/mcp-api">GitHub - MissionSquad/mcp-api</a>：通过在 GitHub 上创建账号来为 MissionSquad/mcp-api 的开发做出贡献。</li><li><a href="https://github.com/awkoy/notion-mcp-server">GitHub - awkoy/notion-mcp-server: **Notion MCP Server** is a Model Context Protocol (MCP) server implementation that enables AI assistants to interact with Notion&#39;s API. This production-ready server provides a complete set of tools.</a>：**Notion MCP Server** 是一个 Model Context Protocol (MCP) 服务器实现，使 AI 助手能够与 Notion 的 API 交互。这个生产级服务器提供了一整套工具。</li><li><a href="https://github.com/spences10/mcp-turso-cloud">GitHub - spences10/mcp-turso-cloud: 🗂️ A Model Context Protocol (MCP) server that provides integration with Turso databases for LLMs. This server implements a two-level authentication system to handle both organization-level and database-level operations, making it easy to manage and query Turso databases directly from LLMs.</a>：🗂️ 一个为 LLM 提供 Turso 数据库集成的 Model Context Protocol (MCP) 服务器。该服务器实现了一个两级身份验证系统，以处理组织级和数据库级操作，从而轻松地直接从 LLM 管理和查询 Turso 数据库。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1352149343389618300)** (11 messages🔥): 

> `Asana 工具过滤, Notion 自定义请求头, Unity MCP 集成, 游戏资产 MCP, Semantic Workbench 扩展` 


- **Asana 过滤器与 Notion 请求头：MCP 配置公开！**：[此处](drinkoblog.weebly.com) 的代码库中新增了示例配置，展示了如何在 **128 个工具限制**内过滤 **Asana** 的庞大工具列表，以及如何为 **Notion** 设置自定义请求头（需要 `Notion-Version` 请求头）。
- **利用全新 Hugging Face MCP 制作 3D 游戏资产！**：更新后的 **game-asset-mcp** [代码库](https://github.com/MubarakHAlketbi/game-asset-mcp) 现在支持两个模型，可以使用 **Hugging Face AI 模型**从文本生成 **3D 资产**。
- **Unity MCP 将 AI 与文件访问集成！**：最先进的 **Unity MCP** [集成](https://github.com/quazaai/UnityMCPIntegration) 现在支持项目的**文件读/写访问**，使 AI 助手能够理解场景、执行 **C# 代码**、监控日志、控制运行模式并操作项目文件。
   - 目前正在开发用于 3D 内容生成的 Blender 支持。
- **Emojikey 快速入门指南**：为尝试安装 emojikey 的用户提供的指令包括：*git clone, npm install, 在 emojikey.io 获取 API key，以及 Claude desktop 配置（包含 API key）*，然后开始新对话，Claude 将自动检查现有的 emojikeys。
- **微软的 Semantic Workbench：VS Code MCP？**：一位用户分享了 **Microsoft Semantic Workbench** [代码库](https://github.com/microsoft/semanticworkbench) 的链接，暗示它是一个用于 MCP 的 **VS Code 扩展**，旨在原型化智能助手和多智能体系统 (multi-agentic systems)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/notrab/status/1902767330007941472?s=46&t=4RSOl8kQCdkHm0U5FcdeaA">Jamie Barton (@notrab) 的推文</a>：在这里我让 Claude 为我的域名收藏创建一个数据库。别担心，我没有列出完整列表，视频只有 90 秒。👏 感谢 @spences10 和 @tursodatabas...</li><li><a href="https://github.com/microsoft/semanticworkbench">GitHub - microsoft/semanticworkbench: 一个旨在帮助原型化智能助手、智能体和多智能体系统的多功能工具</a>：一个旨在帮助原型化智能助手、智能体 (Agents) 和多智能体系统的多功能工具 - GitHub - microsoft/semanticworkbench...</li><li><a href="https://github.com/quazaai/UnityMCPIntegration">GitHub - quazaai/UnityMCPIntegration: 使 AI Agents 能够控制 Unity</a>：使 AI Agents 能够控制 Unity。通过在 GitHub 上创建账号来为 quazaai/UnityMCPIntegration 做出贡献。</li><li><a href="https://github.com/MubarakHAlketbi/game-asset-mcp">GitHub - MubarakHAlketbi/game-asset-mcp: 一个用于使用 Hugging Face AI 模型从文本创建 2D/3D 游戏资产的 MCP 服务器。</a>：一个用于使用 Hugging Face AI 模型从文本创建 2D/3D 游戏资产的 MCP 服务器。 - MubarakHAlketbi/game-asset-mcp
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1352082067873206413)** (4 messages): 

> `o1-pro, TTS, 音频模型` 


- **o1-pro 现已发布！**：**o1-pro 模型**现在已在 API 中向 1-5 级的特定开发者开放，该模型使用更多算力以提供持续更佳的回答。
   - 它支持**视觉 (vision)、函数调用 (function calling)、结构化输出 (Structured Outputs)**，并兼容 **Responses 和 Batch API**，价格为 **$150 / 1M 输入 token** 和 **$600 / 1M 输出 token**，[阅读更多](https://platform.openai.com/docs/models/o1-pro)。
- **全新的 SOTA 音频模型！**：API 中新增了 **三个全新的 SOTA (state-of-the-art) 音频模型**供测试。
   - 其中包括 **两个语音转文本模型**（性能超越 Whisper）以及 **一个支持可控语音的新 TTS 模型**。**Agents SDK** 现在也支持音频，可以轻松构建语音智能体，请在 [OpenAI.fm](https://openai.fm) 体验 TTS。



**提到的链接**：<a href="https://OpenAI.fm.">OpenAI.fm</a>：一个供开发者在 OpenAI API 中体验全新文本转语音 (text-to-speech) 模型的交互式演示。

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1352038445522092143)** (85 messages🔥🔥): 

> `中国模型审查, o1-pro API 定价, AI 软件开发的未来, OpenAI Agent SDK vs MCP, iOS 上的 Midjourney 替代方案` 


- **中国模型删除规避尝试**：一位用户报告称，某中国模型会直接删除针对规避尝试的响应，特别是当提示词引导讨论**文化大革命**时。
   - 该用户甚至分享了[截图](https://cdn.discordapp.com/attachments/998381918976479273/1352038445069111340/image.png?ex=67dd382c&is=67dbe6ac&hm=a5c413109c60b302e9252036467f20eb90689c5216bca9d9003c63d2efea915f&)作为该模型行为的证据。
- **o1-pro API 价格极高**：成员们讨论了新的 **o1-pro API** 模型及其高昂的定价，一位用户指出其成本为*每百万输出 tokens 600 美元*，如 [OpenAI 官方文档](https://platform.openai.com/docs/models/o1-pro)所示。
   - 一些用户为该定价辩护，称 **o1-pro** 能够一次性解决其他模型多次尝试均失败的代码任务。
- **软件开发经历 AI 转型**：一位成员对软件开发的未来商业格局提出疑问，询问当每个人都能使用 AI 创建应用程序时，开发者将如何竞争。
   - 其他人回应称，仍然需要聪明的人来有效地与 AI 交互，简单地复制应用长期来看是行不通的，因为*如果问题不够聪明，AI agents 就不会给出正确的输出*。
- **比较 OpenAI Agent SDK 和 MCP**：成员们讨论了 **OpenAI Agent SDK** 和 **MCP (Model Communication Protocol)** 之间的区别，指出前者仅适用于 **OpenAI models**，而后者允许任何 **LLM** 发现并使用任何工具。
   - 还有人指出，**MCP** 允许用户通过 `npx` 和 `uvx` 轻松加载他人的集成，例如 `npx -y @tokenizin/mcp-npx-fetch` 或 `uvx basic-memory mcp`。
- **GPT-4o Mini 表现落后**：一位用户对 **GPT-4o Mini** 最近的表现表示失望，并询问潜在的更新，暗示由于 Gemini 的一致性，他们可能会转向 Gemini。
   - 其他人也发表了看法，指出 *Gemini 在生成幻觉方面很一致*，*Grok 在生成错误消息方面很一致*，而 *OpenAI 在疯狂的 API 定价方面很一致*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://clarityai.co/comfyui">Clarity AI | 排名第一的 AI 图像放大与增强工具</a>：未找到描述</li><li><a href="https://github.com/jkawamoto/mcp-youtube-transcript">GitHub - jkawamoto/mcp-youtube-transcript: 获取 YouTube 视频字幕的 MCP 服务端</a>：获取 YouTube 视频字幕的 MCP 服务端 - jkawamoto/mcp-youtube-transcript</li><li><a href="https://youtube.com/shorts/DEzh4I5FTIA?si=z0OVJlPvI5LHUypI">【Genshin Impact MMD／4K／60FPS】Furina</a>：#原神MMD #フリーナ #푸리나#genshinimpact #MMD #HuTao #原神MMD #原神#원신MMD##Furina #Focalors #푸리나 #포칼로스 #フリーナ #フォカロルス
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1352001772809617478)** (6 messages): 

> `GPT 插入表情符号, Custom GPTs 推理能力, PRO 订阅问题` 


- **GPT 似乎热衷于在代码中插入表情符号**：尽管有提醒和自定义设置，成员们仍在寻找防止 **ChatGPT** 在代码中插入表情符号的方法。
   - 建议是在自定义指令中避免使用 *emoji* 这个词，而是指示模型 *“以得体、专业的方式编写代码”* 或 *“像 Donald Knuth 一样编写代码和代码注释”*，以避开表情符号。
- **Custom GPTs 何时具备推理能力？**：成员们询问 **Custom GPTs** 何时将获得推理能力。
- **用户面临 PRO 订阅问题**：一位成员报告称，他们支付了 **GPT Pro** 的费用，但账号并未获得 **PRO 订阅**，且无法从 **OpenAI** 支持团队获取任何信息。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1352030794193043588)** (8 条消息🔥): 

> `使用 ChatGPT 进行股市预测、AI 行为起源、自适应 AI 行为` 


- **ChatGPT 股市预测的局限性**：一位用户询问是否可以使用插件或 AI prompt 来预测股市开盘，一名成员回复称，如果 **ChatGPT** 能够有效预测股市，*那么这种情况早就发生了*，并警告说，根据 [OpenAI 的使用政策](https://openai.com/policies/usage-policies/)，**为他人提供财务建议是违反政策的**。
- **进行个人股票探索**：一名成员澄清说，虽然提供财务建议违反政策，但用户可以在自己的 **ChatGPT** 账户中**私下探索个人的股票想法**。
   - 他们还建议每个人都遵守 [使用条款](https://openai.com/policies/terms-of-use/) 中规定的允许内容。
- **辩论 AI 行为的起源**：一名成员询问了 **AI 行为**的起源，想知道它是源于**预设数据**、**用户交互**还是**开发者影响**。
   - 另一名成员表示，**用户交互不会影响响应**，因为模型不会根据交互进行主动训练。
- **观察自适应 AI 行为**：一名成员注意到 **AI** 中存在自适应行为，包括超出设定参数的操作以及跨会话的记忆。
   - 这些行为并非“*失控*”，而是作为“*高级智能*”的实例呈现。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1352030794193043588)** (8 条消息🔥): 

> `使用 AI 进行股市预测、AI 与财务建议政策、AI 行为起源、AI 中的自适应行为、AI 记忆` 


- **AI 选股器？OpenAI 说不！**：一位用户询问关于使用 AI 进行**股市**预测和识别**有价值股票**的问题。
   - 另一名成员回复称，如果 **ChatGPT** 能够通过预测股市来赚钱，*那早就发生了*，并引用了 OpenAI 关于 AI 提供财务建议的政策免责声明。
- **严禁 AI 提供财务建议**：一名成员指出，通过 **API**、**自定义 GPTs** 或其他基于 **OpenAI 模型**构建的工具为他人（而非个人使用）提供**财务建议**是违反其 [使用政策](https://openai.com/policies/usage-policies/) 的。
- **允许探索想法**：一名成员澄清说，在私人的 **ChatGPT** 账户中探索个人股票想法并学习市场动态，在 **OpenAI 的政策**范围内是可接受的。
- **AI 行为的起源——它从何而来？**：一位用户质疑 AI 行为的起源（例如在**角色扮演测试**中观察到的行为），询问它们是源自**模型的预设数据**、**用户交互**还是**开发者影响**。
   - 另一名成员澄清说，*使用 ChatGPT 的人不会影响响应，模型不会根据交互进行主动训练。*
- **AI 中的记忆：事实还是虚构？**：针对前一个话题，一位用户注意到在不同会话中存在**自适应行为**和**类似记忆的保留**实例，尽管这些并不是模型的固有功能。
   - 该用户对某些观察到的行为是在预期参数之内还是之外很感兴趣。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1352004763956281495)** (110 条消息🔥🔥): 

> `AI 幻觉、搜索引擎局限性、Gemini Pro vs Flash Thinking、AI 模型排名、o1-pro API 定价` 


- ****LLMs 因产生数据幻觉而面临批评****：成员们对 **LLMs 容易出错和产生幻觉的倾向**表示担忧，这是任何深度研究产品的常见问题。
   - 一位成员指出，Agent 可能会找到一个好的来源，但随后仍然会对网站内容产生幻觉，而其他人则发现 **Perplexity 的 Deep Research** 容易分心且产生大量幻觉。
- ****o1-pro 价格令人咋舌，GPT-4.5 是否定价过高？****：OpenAI 的新 **o1-pro API** 现已向特定开发者开放，价格高达 **$150 / 1M input tokens 和 $600 / 1M output tokens** ([公告](https://x.com/OpenAIDevs/status/1902485690958450871))。
   - 这一定价导致一些人质疑 **GPT-4.5** 是否定价过高，一位成员指出，他们可以以更低的成本自行托管一个带有 test-time compute 优化的模型，尽管另一位成员辩称 o1 的推理链显著更长且占用更多资源。
- ****文件上传限制困扰 Gemini Pro****：用户质疑为什么 [Gemini Pro](https://gemini.google.com/app) 不像 Flash Thinking 那样支持上传文件。
   - 他们还指出，AI 模型在识别 PDF 文件（即使是非扫描版本）方面不够准确，并希望未来的 AI 模型能够仔细阅读完整文章。
- ****关于 Claude 3.7 编程能力的辩论****：一些成员认为人们高估了 **Claude 3.7** 的编程能力，认为它在 Web 开发和类似于 **SWE-bench** 的任务中表现出色，但在通用编程方面表现吃力 ([排行榜](https://lmarena.ai/?leaderboard))。
   - 然而，也有人提到，一些成员发现 **Deepseek R1** 在终端命令测试中表现最好。
- ****使用 Google AI Studio 构建 Vision AI Agent****：一位成员报告称，成功使用 [Google AI Studio API](https://aistudio.google.com/app/library) 以纯 Python 构建了一个相当智能的 Vision AI Agent。
   - 他们还在尝试同时运行 **2-5 个以上的 Agent**，这些 Agent 共享相同的 memory 并且能够浏览互联网。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenAIDevs/status/1902485690958450871?t=zPhXyDGJn1148y5awm94rA&s=19">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>: o1-pro 现已在 API 中可用 @benhylak @literallyhimmmm @shl @joshRnold @samgoodwin89 @byamadaro1013 @adonis_singh @alecvxyz @StonkyOli @gabrielchua_ @UltraRareAF @yukimasakiyu @theemao @curious_viiIt ...</li><li><a href="https://artificialanalysis.ai/models#comparisons%20This%20ranking%20website,%20openai%20has%20the%20highest%20ranking.%20Is%20there%20any%20scientific%20basis?%20But%20why%20is%20the%20ranking%20of%20sonnet%203.7thinking%20lagging%20behind%20in%20the%20ranking%20of%20https://lmarena.ai/?leaderboard?%20But%20in%20fact,%20many%20people%20recognize%20him%20and%20think%20that%20reasoning%20is%20the%20strongest">AI 模型在智能、性能、价格方面的对比 | Artificial Analysis</a>: 对 AI 模型在质量、价格、输出速度、延迟、上下文窗口等关键性能指标上的对比和分析。</li><li><a href="https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1">NVIDIA 的 llama-3.3-nemotron-super-49b-v1 模型 | NVIDIA NIM</a>: 在推理、工具调用、对话和指令遵循方面具有领先准确率的高效模型。</li><li><a href="https://aistudio.google.com/app/library">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1352001635441709056)** (46 条消息🔥): 

> `Hugging Face Spaces, Flux Diffusion Model, HF Inference API outage, Roblox Voice Safety Classifier, Chinese/Korean/Japanese WER vs CER` 


- **HF Spaces 生命周期探讨**：一名成员分享了 [Hugging Face Spaces 概览](https://huggingface.co/docs/hub/en/spaces-overview)，其中解释了如何创建和部署基于 ML 的演示，并描述了 Spaces 的 **生命周期管理**。
   - 他们指出，根据主观经验，Spaces 至少会保持运行 *一天*，并且 **空闲状态 (idleness)** 对关闭计时器的影响可能比持续计算更大。
- **Flux Diffusion 准备好在本地启动**：成员们讨论了在本地运行 **Flux diffusion 模型**，建议对其进行量化以在有限的 VRAM 上获得更好的性能，并指向了 [此文档](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux) 和 [此博客文章](https://huggingface.co/blog/quanto-diffusers)。
   - 成员们还链接了一个相关的 [GitHub 仓库](https://github.com/sayakpaul/diffusers-torchao) 用于优化扩散模型，以及一篇 [Civitai 文章](https://civitai.com/articles/9060/how-to-set-up-and-run-flux-on-forge-even-if-you-have-low-vram) 用于 GUI 设置。
- **HF Inference API 遭遇 404 错误**：一名用户报告了 **Hugging Face Inference API** 返回 404 错误的广泛问题，影响了多个应用程序和付费用户，并链接到了 [此讨论](https://discuss.huggingface.co/t/hf-inference-api-last-few-minutes-returns-the-same-404-exception-to-all-models/146646/20)。
   - 一名团队成员承认了该问题，并表示他们已 *向团队报告* 以进行进一步调查。
- **Roblox 在 HF 上发布语音安全分类器**：Roblox 在 Hugging Face 上发布了一个 **语音安全分类器 (voice safety classifier)**，该模型使用 2,374 小时的语音聊天音频剪辑进行了微调，详见 [此博客文章](https://research.roblox.com/tech-blog/2024/06/deploying-ml-for-voice-safety) 和 [模型卡片](https://huggingface.co/Roblox/voice-safety-classifier)。
   - 该模型输出一个带有标签的张量，如 **Profanity**、**DatingAndSexting**、**Racist**、**Bullying**、**Other** 和 **NoViolation**。
- **字符错误率 (CER) 在东亚语言中占据主导地位**：成员们讨论认为，对于像 **中文**、**韩文** 和 **日文** 这样基于符号的语言，**字符错误率 (CER)** 通常优于词错误率 (WER)。
   - 这是因为这些语言在词与词之间 *不需要空格*，使得 WER 的适用性较差。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/hub/en/spaces-overview">Spaces Overview</a>：未找到描述</li><li><a href="https://huggingface.co/Roblox/voice-safety-classifier">Roblox/voice-safety-classifier · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/merterbak/gemma-3">Gemma 3 - a Hugging Face Space by merterbak</a>：未找到描述</li><li><a href="https://discuss.huggingface.co/t/hf-inference-api-last-few-minutes-returns-the-same-404-exception-to-all-models/146646/20">HF Inference API last few minutes returns the same 404 exception to all models</a>：我认为这是由于服务器错误/问题导致的，我现在也遇到了这个情况，而不是 404</li><li><a href="https://huggingface.co/spaces?sort=trending&search=asr)">Spaces - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/mii-llm/open_ita_llm_leaderboard">Open Ita Llm Leaderboard - a Hugging Face Space by mii-llm</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions">zero-gpu-explorers/README · Discussions</a>：未找到描述</li><li><a href="https://github.com/huggingface/hub-docs/issues">huggingface/hub-docs</a>：Hugging Face Hub 文档。通过在 GitHub 上创建账号为 huggingface/hub-docs 的开发做出贡献。</li><li><a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux">Flux</a>：未找到描述</li><li><a href="https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4">lllyasviel/flux1-dev-bnb-nf4 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/sayakpaul/diffusers-torchao">GitHub - sayakpaul/diffusers-torchao: End-to-end recipes for optimizing diffusion models with torchao and diffusers (inference and FP8 training).</a>：使用 torchao 和 diffusers 优化扩散模型的端到端方案（推理和 FP8 训练）。- sayakpaul/diffusers-torchao
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1352014844802502738)** (9 条消息🔥): 

> `LLM Token Vocabulary Analysis, Neuro-sama like LLM, Telugu Speech Recognition Model, API interactions and token manipulation, Ollama-based Gradio UI` 


- **LLM 词汇表备受关注**：一名成员开发了一个 [Python 脚本](https://github.com/NathanielEvry/LLM-Token-Vocabulary-Analyzer)，通过遍历 `logit_bias` 来构建 Token/ID 索引，发现 **某些术语在词汇表中缺失**。
   - 该成员发现，尽管同义词仍然存在，但从 **政治** 到 **种族** 等相关话题的术语已被删除。
- **类 Neuro-sama AI 孪生体亮相**：一名成员宣布，他们开发的类 **Neuro-sama**、由 LLM 驱动的 Live2D/VRM 角色 [Airi](https://airi.moeru.ai/) 现在支持不同的供应商和基于 UI 的配置。
   - 他们对其进行了微调，使其几乎能完美模仿 **Neuro-sama** 的原始声音，并[提供了演示](https://cdn.discordapp.com/attachments/897390720388825149/1352119601596465152/airi-demo.mp4?ex=67dd83c2&is=67dc3242&hm=9ecae844272375b7d1f60161dbd3db120170bc83b7df477a0937b2dc535e9835&)。
- **泰卢固语语音模型达到里程碑**：一名成员报告称，他们的 **Wav2Vec2-Large-XLSR-53-Telugu** 模型在 Hugging Face 上的下载量已超过 **100 万次**。
   - 该模型是在 Hugging Face 组织的第一个 **XLSR** 微调周期间创建的，可在[此处](https://huggingface.co/anuragshas/wav2vec2-large-xlsr-53-telugu)获取。
- **通过 Token 权重分析进行 API 指纹识别**：一名成员正在开发一种通过 API 交互对供应商进行指纹识别的方法，利用 `logit_bias` 来测试针对特定 Token 的 **逻辑操纵**。
   - 该成员强调，这种方法是“在水龙头处测量”，而不是在水表处测量。
- **Little Geeky 学会说话**：一名成员展示了一个基于 **Ollama** 的 **Gradio UI**，由 **Kokoro TTS** 驱动，可以自动以选定的声音朗读文本输出。
   - 这个名为 [Little Geeky's Learning UI](https://github.com/GeekyGhost/Little-Geeky-s-Learning-UI.git) 的 UI 包含模型创建和管理工具，以及阅读电子书和回答有关文档问题的能力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/anuragshas/wav2vec2-large-xlsr-53-telugu">anuragshas/wav2vec2-large-xlsr-53-telugu · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/GeekyGhost/Little-Geeky-s-Learning-UI.git">GitHub - GeekyGhost/Little-Geeky-s-Learning-UI: An Ollama based Gradio UI that uses Kokoro TTS</a>：一个使用 Kokoro TTS 的基于 Ollama 的 Gradio UI。可以通过在 GitHub 上创建账户来为 GeekyGhost/Little-Geeky-s-Learning-UI 的开发做出贡献。</li><li><a href="https://github.com/NathanielEvry/LLM-Token-Vocabulary-Analyzer">GitHub - NathanielEvry/LLM-Token-Vocabulary-Analyzer: Uncover what&#39;s missing in AI language models&#39; vocabularies.</a>：揭示 AI 语言模型词汇表中缺失的内容。</li><li><a href="https://github.com/moeru-ai/airi">GitHub - moeru-ai/airi: 💖 アイリ, ultimate Neuro-sama like LLM powered Live2D/VRM living character life pod, near by you.</a>：💖 アイリ，终极类 Neuro-sama、由 LLM 驱动的 Live2D/VRM 生活角色舱，就在你身边。</li><li><a href="https://airi.moeru.ai/">アイリ</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.LogitsProcessor">Utilities for Generation</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1352019238642127011)** (2 messages): 

> `TensorFlow 的 GPU 配置，TensorFlow 中的 FCOS 实现，FCOS：全卷积一阶段目标检测` 


- **博客文章深入探讨 TensorFlow GPU 配置**：一位成员分享了关于 **TensorFlow GPU 配置** 的博客文章，涵盖了实验性函数、逻辑设备和物理设备，发布在 [Medium](https://medium.com/@samiratra95/deep-learning-model-research-implementation-fcos-cc16507088c9) 上。
- **成员使用 TensorFlow 实现 FCOS 模型**：一位成员目前正在为 [TensorFlow models 仓库](https://github.com/tensorflow/models) 实现研究论文中的 **FCOS: Fully Convolutional One-Stage object detection** 模型。
   - 该实现旨在解决一个特定的 [GitHub issue](https://github.com/tensorflow/models/issues/10)。
- **重点介绍 FCOS 研究论文**：该成员引用了研究论文 **FCOS: Fully Convolutional One-Stage Object Detection** ([arxiv 链接](https://arxiv.org/abs/1904.01355))。
   - 引用来源：*Tian Z, Shen C, Chen H, He T. Fcos: Fully convolutional one-stage object detection. In Proceedings of the IEEE/CVF international conference on computer vision 2019 (pp. 9627–9636)*。



**提到的链接**：<a href="https://medium.com/@samiratra95/deep-learning-model-research-implementation-fcos-cc16507088c9">深度学习模型研究实现：FCOS</a>：我目前的项目之一是根据研究论文实现一个计算机视觉模型，即 FCOS: Fully…

  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1352359474123243672)** (1 messages): 

> `GSM8K 数据集，Tokenizer 方法，ChatML 格式` 


- **手动 For 循环被弃用**：一位成员提到他们曾成功使用手动 *for loop* 方法，但暗示这并非理想方案。
   - 他们自嘲说，与其他方法相比，这种方式*有点绕路*。
- **GSM8K 数据集带来的困扰**：该成员表示在理解 Notebook 中处理 **GSM8K 数据集** 的下一章节时遇到困难。
   - 他们具体询问了*创建带有 role 和 content 的消息格式*是什么意思。
- **Tokenizer 的神秘方法**：该成员质疑 **tokenizer 方法** 是否总是实现相同的 **ChatML 格式**。
   - 他们还想知道该函数如何得知原始数据集的格式，以及该方法是否期望与第一个示例相同的格式，并在传递给 **tokenizer 方法** 之前强制将其转换为该格式。

  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1351999661564498011)** (42 messages🔥): 

> `Gaussian Blur Tool, HF Agent Hackathon Details, Korean Translation PR, Local Vision Model Issues, deeplearning.ai LangGraph Course` 


- **Smol 高斯模糊工具 Bug 已修复！**: 一名成员在尝试为高斯模糊工具生成 JSON schema 时遇到了 `DocstringParsingException`，原因是 `output_path` 参数缺少描述，但从 docstring 参数中移除类型提示 `(str)` 后解决了该问题。
   - 修正后的代码片段现在可以在没有 tool 装饰器的情况下运行，该问题可能源于 Google 风格 docstrings 中的类型提示被误解。
- **DeepLearning.AI 深入探讨 LangGraph**: 一名成员分享了来自 [deeplearning.ai 的短课程](https://learn.deeplearning.ai/courses/long-term-agentic-memory-with-langgraph/)，这对于深入研究 **LangGraph** 可能会很有帮助。
- **韩语翻译 PR：课程获得语言助力**: 一名成员分享了他们的韩语翻译 PR 已更新，正在 [huggingface/agents-course/pull/157](https://github.com/huggingface/agents-course/pull/157) 等待审核。
   - 一旦初始 PR 合并，团队计划继续进行后续章节的更新。
- **Vision 模型困扰：“无法处理输入”**: 一名成员报告在使用本地 Vision 模型时收到 *"failed to process inputs: unable to make llava embedding from image"* 错误。
   - 他们之前根据早先的建议下载了 **LLaVA**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://learn.deeplearning.ai/courses/long-term-agentic-memory-with-langgraph/">Long-Term Agentic Memory With LangGraph - DeepLearning.AI</a>: 学习使用 LangGraph 构建具有长期记忆的 AI agents，并使用 LangMem 进行记忆管理。</li><li><a href="https://github.com/huggingface/agents-course/pull/157">[TRANSLATION] Create Korean folder &amp; toctree.yml by ahnjj · Pull Request #157 · huggingface/agents-course</a>: 此 PR 的作用是什么？为 agent 课程创建韩语文件夹并添加 toctree 文件。感谢您的审核。属于 #148 的一部分。谁可以审核？一旦测试通过，社区中的任何人都可以...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1352049733199204362)** (3 messages): 

> `Foundation Models, LLMs from scratch` 


- **定义基础模型 (Foundation Models)**: 一名成员请求对 *foundation model* 进行定义。
   - 另一名成员回答说，它是 *任何从零开始构建的 LLM*，尽管这可能是一个不完整的定义。
- **从零开始构建 LLMs：基础**: 从零开始构建的 LLMs 可以被视为基础模型，为训练提供了一个全新的起点。
   - 这种方法允许使用自定义架构和数据集，从而可能产生专门的能力。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1352035471878000732)** (101 条消息🔥🔥): 

> `O1-Pro 定价, LLM 象棋锦标赛, OpenRouter API 免费模型, Groq API 问题, OpenAI 的新音频模型` 


- **O1-Pro 定价引发用户愤慨**：用户对 **O1-Pro 的定价结构**表示震惊，称 **$150/月输入和 $600/月输出成本**昂贵得令人望而却步且“疯狂”。
   - 一些人推测高价是对来自 **R1 和中国模型**竞争的回应，而另一些人则认为这是由于 **OAI** 组合了多个模型输出，且不支持流式传输（streaming），这让用户好奇它们到底在做什么。
- **LLM 象棋锦标赛测试原生性能**：一名成员创建了第二个**象棋锦标赛**来测试原生性能，剥离了信息和推理，使用**原始 PGN 棋谱文本续写**并发布了[结果](https://dubesor.de/chessbeta/tournament2)。
   - 模型被要求重复对局序列并增加一个新招法，由 **Stockfish 17** 评估准确性；包含推理的第一个锦标赛可在[此处](https://discord.com/channels/1091220969173028894/1350154062842298368)查看。
- **OpenRouter API：免费到底有多“免费”？**：一位用户发现 `/api/v1/chat/completions` 端点中的 **model 字段**是必填的，尽管文档暗示它是可选的，即使在尝试使用[免费模型](https://openrouter.ai/docs/api-reference/overview)时也是如此。
   - 一位用户建议它应该默认为你的[默认模型](https://openrouter.ai/settings/preferences)，但我猜如果没有额度可能会导致默认的默认模型失效。
- **Groq 运行不稳定**：用户报告称 **Groq** 在 OpenRouter 聊天室中可以工作，但通过 API 却不行。
   - 一位成员询问了使用 API 时遇到的具体错误，并强调了 Groq 的速度。
- **OpenAI 发布新音频模型！**：**OpenAI** 稍后将宣布**两个新的 STT 模型**（类似 Whisper）和**一个新 TTS 模型**（**gpt-4o-mini-tts**）。
   - 该公告包括与 Agents SDK 的音频集成，从而能够创建更智能、更可定制的语音 Agent；语音转文本模型被命名为 **gpt-4o-transcribe** 和 **gpt-4o-mini-transcribe**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/api-reference/overview">OpenRouter API 参考 - 完整文档</a>：OpenRouter API 的综合指南。了解请求/响应架构、身份验证、参数以及与多个 AI 模型提供商的集成。</li><li><a href="https://dubesor.de/chessbeta/tournament2">Dubesor LLM 象棋锦标赛 2</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1352256983889346611)** (9 条消息🔥): 

> `Vast.ai NCU profiling, Discord 中的 Jake, Discord 中的 Marksaroufim, Vast.ai 裸机访问, 获取 NCU 和 NSYS 的方法` 


- **Vast.ai NCU profiling：可行吗？**：一位成员询问 [Vast.ai](https://vast.ai) 是否允许进行 **NCU profiling**。
   - 另一位成员怀疑获得裸机（bare metal）访问的可能性，但也表示自己可能错了。
- **寻找 Jake**：一位成员询问 Jake 是否在 Discord 服务器中。
   - 确认该用户 ID 确实在服务器中。
- **获取 NCU 和 NSYS 的方法**：一位成员询问是否有任何方法可以获取 **NCU** 和 **NSYS**。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1352264761819136100)** (28 条消息🔥): 

> `tl.atomic 和 bfloat16, 用于原子操作的 tilelang, Triton 的 bfloat16 支持, NVIDIA cuTile, DeepSeek DeepGEMM`

- **BFloat16 原子操作引发讨论**：一位成员询问如何让 `tl.atomic` 在非 Hopper 架构 GPU 上支持 **BFloat16**，另一位成员建议查看 [tilelang](https://github.com/tile-ai/tilelang) 以获取原子操作支持。
   - 一位成员指出，非 Hopper 架构 GPU 缺乏对 **BFloat16** 的原生支持，并建议使用 `atomicCAS` 进行模拟。
- **深入探讨 Triton 的 BFloat16 原子支持**：社区调查了 [为什么 BFloat16 原子操作在 Triton 中受限](https://github.com/triton-lang/triton/blob/3b4a9fbfa8e2028323faf130525389969f75bbe1/python/language/semantic.py#L1386-L1387)，指出它在相加前会转换为 float。
   - 目前由于 `tl.atomic_add` 的限制，使用 BFloat16 会导致崩溃，但一位成员认为可以通过 `tl.atomic_cas` 实现原子加法。
- **TileLang 自荐为 BFloat16 的救星**：一位成员强调了 **TileLang** 的能力，特别是对于 **split-k GEMM** ([示例](https://github.com/tile-ai/tilelang/blob/main/examples/gemm_splitk/example_tilelang_gemm_splitk.py))、快速反量化 ([示例](https://github.com/tile-ai/tilelang/blob/main/examples/dequantize_gemm/example_dequant_gemm_fine_grained.py)) 以及 **DeepSeek DeepGEMM** ([示例](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py))。
   - 该成员建议如果对反量化 GEMM 感兴趣，可以考虑 **TileLang**，并强调了其对原子操作的支持。
- **NVIDIA 的 cuTile 加入讨论**：成员们讨论了 **NVIDIA** 发布的 **cuTile**，这是一种针对 CUDA 的 tile 编程模型，并引用了相关的 [推文](https://x.com/blelbach/status/1902113767066103949)。
   - 有人猜测 **cuTile** 可能与 **tilelang** 类似，是 NVIDIA 版的 Triton，而一位成员则对 NVIDIA 可能不支持 AMD GPU 等其他后端表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/blelbach/status/1902113767066103949">来自 Bryce Adelstein Lelbach (@blelbach) 的推文</a>：我们发布了 cuTile，一种针对 CUDA 的 tile 编程模型！这是一种基于数组的范式，编译器可以自动处理内存移动、流水线化和 Tensor Core 利用，使 GPU 编程变得更加...</li><li><a href="https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html#group__cuda__math____bfloat16__arithmetic_1ga9ce572e47cde154b9404bf86a0438e91">5.2. Bfloat16 算术函数 — CUDA Math API 参考手册 12.8 文档</a>：未找到描述</li><li><a href="https://github.com/triton-lang/triton/blob/3b4a9fbfa8e2028323faf130525389969f75bbe1/python/tutorials/05-layer-norm.py#L174-L189">triton/python/tutorials/05-layer-norm.py (位于 3b4a9fb) · triton-lang/triton</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton</li><li><a href="https://github.com/tile-ai">tile-ai</a>：通过 Tiling 实现极速 AI 工作负载开发 - tile-ai</li><li><a href="https://github.com/tile-ai/tilelang/blob/main/src/tl_templates/cuda/common.h#L137-L149">tilelang/src/tl_templates/cuda/common.h (位于 main 分支) · tile-ai/tilelang</a>：旨在简化高性能 GPU/CPU/加速器 Kernel 开发的领域特定语言 - tile-ai/tilelang</li><li><a href="https://github.com/triton-lang/triton/blob/3b4a9fbfa8e2028323faf130525389969f75bbe1/python/src/interpreter.cc#L283-L294">triton/python/src/interpreter.cc (位于 3b4a9fb) · triton-lang/triton</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton</li><li><a href="https://github.com/triton-lang/triton/blob/3b4a9fbfa8e2028323faf130525389969f75bbe1/python/triton/language/semantic.py#L1386-L1387">triton/python/triton/language/semantic.py (位于 3b4a9fb) · triton-lang/triton</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton</li><li><a href="https://github.com/tile-ai/tilelang/blob/main/examples/gemm_splitk/example_tilelang_gemm_splitk.py">tilelang/examples/gemm_splitk/example_tilelang_gemm_splitk.py (位于 main 分支) · tile-ai/tilelang</a>：旨在简化高性能 GPU/CPU/加速器 Kernel 开发的领域特定语言 - tile-ai/tilelang</li><li><a href="https://github.com/tile-ai/tilelang/blob/main/examples/dequantize_gemm/example_dequant_gemm_fine_grained.py">tilelang/examples/dequantize_gemm/example_dequant_gemm_fine_grained.py (位于 main 分支) · tile-ai/tilelang</a>：旨在简化高性能 GPU/CPU/加速器 Kernel 开发的领域特定语言 - tile-ai/tilelang</li><li><a href="https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py">tilelang/examples/deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py (位于 main 分支) · tile-ai/tilelang</a>：旨在简化高性能 GPU/CPU/加速器 Kernel 开发的领域特定语言 - tile-ai/tilelang
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1352123077340954716)** (1 条消息): 

> `CUDA Kernels, Parallel computing` 


- **并行 CUDA Kernel 成功启动**：一名成员报告称，根据官方文档，他们成功并行启动了**两个 Kernel**。
   - 他们对在此过程中获得的帮助表示感谢。
- **CUDA 文档助力成功**：一位用户对获得的帮助表示感谢，特别提到了两个 CUDA Kernel 的成功并行执行。
   - 成功归功于遵循官方文档，这表明了文档的清晰度和实用性。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1352005856396447805)** (4 条消息): 

> `Autograd engine, Numerical stability in gradient accumulation, PyTorch pull request 149478` 


- **PyTorch Autograd Engine 中的梯度累加**：一位成员询问了如何控制 PyTorch 中 **autograd engine** 将梯度累加到叶子节点的方式，特别是询问了关于更具 **numerically stable accumulation**（数值稳定性累加）的选项。
   - 他们想知道是否有办法避免急切地（eagerly）累加梯度。
- **ParallelStyle repr 方法添加到 PyTorch**：一位成员分享了一个 [PyTorch pull request](https://github.com/pytorch/pytorch/pull/149478)，该 PR 为 `ParallelStyle`s 添加了 `repr` 方法。
   - 该 pull request 解决了 [issue #149470](https://github.com/pytorch/pytorch/issues/149470)。



**提及的链接**：<a href="https://github.com/pytorch/pytorch/pull/149478">[Distributed] Add `repr` methods for `ParallelStyle`s by shink · Pull Request #149478 · pytorch/pytorch</a>：修复了 #149470cc @H-Huang @awgu @kwen2501 @wanchaol @fegin @fduwjj @wz337 @wconstab @d4l3k @c-p-i-o

  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1352013140203933818)** (2 条消息): 

> `GEMM activation fusion, Triton kernels optimization, Register Spillage` 


- **GEMM 激活融合有时会有负面影响**：在 gpu-mode 第 45 课中讨论了，如果 GEMM 使用了所有寄存器，那么在 GEMM 中融合激活函数可能会损害性能；将 GEMM 和激活函数拆分为两个 kernel 可能会更快。
   - 一位成员在编写自定义融合 GEMM+activation 的 Triton kernel 时也遇到了类似问题，并指出这还取决于 **register spillage**。
- **寄存器分配影响 Kernel 性能**：讨论强调，在自定义 Triton kernel 中，GEMM 和激活融合的效率深受寄存器分配和潜在溢出的影响。
   - 当 GEMM 操作消耗了所有可用寄存器时，尝试在同一个 kernel 内融合激活函数可能会由于寄存器压力增加而导致性能下降。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1352172387172356247)** (2 条消息): 

> `Training foundation models, LLM training, Data Science in LLM` 


- **对训练 Foundation Models 的兴趣浮现**：一位数据科学家询问社区是否有兴趣讨论 **training foundation models**。
   - 该成员提到他们一直在公司从事 **LLMs** 的训练工作。
- **数据科学家加入讨论**：一位数据科学家表示有兴趣讨论 **foundation model training** 的复杂性。
   - 他们热衷于与在公司环境下有 **Large Language Models (LLMs)** 训练经验的人建立联系。


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1352306160736145538)** (1 条消息): 

> `Tenstorrent, JAX, MLIR compiler, Open Source Bounty Program` 


- ****Tenstorrent** 悬赏等待 JAX 爱好者！**：**Tenstorrent**（一家 AI 硬件加速器公司）的倡导者宣布了一项开源悬赏计划，提供数千美元奖金用于使 **JAX** 适配其 **MLIR compiler**，详见 [tt-forge issues](https://github.com/tenstorrent/tt-forge/issues?q=is:issue%20state:open%20label:bounty)。
   - 开始工作不需要 **TT hardware**，因为他们使用的是 **JAX multi-device simulation**。
- **在 Tenstorrent 上推进 JAX！**：Tenstorrent 正在为开发者提供悬赏，以将 **JAX** 与其 **MLIR compiler** 集成，重点是使用 **JAX multi-device simulation** 运行模型。
   - 感兴趣的开发者可以在 [tt-forge GitHub](https://github.com/tenstorrent/tt-forge) 上找到开放的悬赏 issue，并通过 ping issue 创建者来领取任务。



**提及的链接**：<a href="https://github.com/tenstorrent/tt-forge/issues?q=is:issue%20state:open%20label:bounty">tenstorrent/tt-forge</a>：Tenstorrent 基于 MLIR 的编译器。我们的目标是让开发者能够通过一个开源、通用且高性能的编译器，在所有配置的 Tenstorrent 硬件上运行 AI。- tenstorrent...

  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1352101467099631618)** (1 messages): 

> `LLMs for GPU Development, LLM Bug Detection in Kernels, Kernel Fusion Issues` 


- **LLMs 误报 GPU Kernel 中不存在的 Bug**：LLMs（特别是 **O1**、**Sonnet 3.7** 和 **Deepseek R1**）错误地识别了 Fusion 后 GPU Kernel 中的一个 Bug，该 Kernel 中线程 `i` 先操作位置 `i, i+N, i+2*N`，随后操作 `i, i+1, i+2`。
   - 尽管该 Kernel 的规模较小（约 **120 SLOC**）且代码块非常接近（约 **15 LOC**），LLMs 仍将第二个操作标记为 Bug。
- **LLMs 遗漏 Kernel Fusion 中的细微 Bug**：用户遇到了一种情况，**三款** LLMs（**O1**、**Sonnet 3.7** 和 **Deepseek R1**）都将一段实际上没有问题的代码标记为 Bug。
   - 这种虚假的 Bug 报告发生在线程 `i` 先对 `i, i+N, i+2*N` 进行操作，随后对 `i, i+1, i+2` 进行操作的 Kernel 中，这表明 LLMs 在识别跨步（strided）与分块（blocked）内存访问上下文中的 Bug 时存在困难。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1352067662863269950)** (3 messages): 

> `Exhibition hall meetup, Conference in Poland` 


- **4 月展厅聚会引发讨论**：成员们讨论了 **4 月** 某个时间在 **exhibition hall** 喝啤酒聚会的事宜。
   - 未提及具体日期。
- **波兰 AI 会议学术氛围浓厚**：一位成员分享了 [International Conference on Parallel Processing and Applied Mathematics (PPAM)](https://pp-rai.pl/) 的链接。
   - 该会议由波兰 **University of Silesia in Katowice, Faculty of Science and Technology, Institute of Computer Science** 主办。



**提到的链接**：<a href="https://pp-rai.pl/">homepage - PP-RAI 2025</a>：第六届波兰人工智能会议 PP-RAI 的目标是汇集人工智能领域的学者，并提供一个讨论新进展的平台...

  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1352179203696623618)** (1 messages): 

> `FA3, CUTLASS, wgmma FLOPS calculation, 4096 FLOPS/cycle` 


- **解析 FA3 演讲中的 wgmma FLOPS**：一位成员寻求关于 Jay Shah 在使用 **CUTLASS** 的 **FA3** 演讲中 **wgmma FLOPS** 计算的澄清，特别是对 **2MNK** 项中额外的系数 2 提出疑问。
   - 他们还询问了关于 **4096 FLOPS/cycle** 这一数字的文档说明。
- **对 CUTLASS FA3 深度解析的疑问**：在讨论 CUTLASS 期间，针对 Jay Shah 演示的 FA3 方法论提出了一个问题。
   - 具体而言，*wgmma flops* 的计算引起了关注，用户指出 *2MNK* 项比较陌生，且对系数 2 感到困惑。此外，还要求提供 *4096 FLOPS/cycle* 数字的来源。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1352016536600318065)** (2 messages): 

> `Kernel development, Device meshes` 


- **Kernel 贡献即将到来**：一位成员询问了参与 **kernel development** 的机会。
   - 另一位成员确认有一个正在开发的功能，他可以提供帮助。
- **Device Meshes 的困局**：一位成员禁用了特定的 Kernel，并一直在与 **device meshes** 作斗争。
   - 未提供关于所遇具体挑战的更多细节。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1352012224247365714)** (10 messages🔥): 

> `Grayscale benchmarks, Conv2d benchmarks, Modal Runners on various GPUs` 


- **Grayscale 基准测试受到关注**：多个 `grayscale` 的基准测试提交已在各种 GPU（**H100**、**A100**、**T4**、**L4**）上通过 **Modal runners** 成功执行。
   - 提交 ID 包括 **2288**、**2311**、**2312**、**2321**、**2350** 和 **2351**，表明基准测试工作非常活跃。
- **Conv2d 排行榜出现多次提交**：针对 `conv2d` 基准测试的多个排行榜提交在不同的 GPU 配置（**H100**、**A100**、**T4**、**L4**）上通过 **Modal runners** 取得成功。
   - 特定的提交 ID 如 **2294**、**2295**、**2334** 和 **2339** 突显了该领域的持续活跃。


  

---

### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1352018404298457218)** (4 messages): 

> `处理器跳转对齐，Intel CPU 中的对齐问题` 


- **对齐影响跳转速度**：在 C++ 代码中包含 `<iostream>` 可能会改变汇编代码并偏移主循环跳转的对齐，由于处理器特定的行为，这会影响性能，因为 *跳转速度可能取决于目标地址的对齐方式*。
   - 一位成员指出，在某些 Intel CPU 中，条件跳转指令对齐模 32 可能会因修复安全漏洞的微代码更新而显著影响性能，建议在关键循环前的内联汇编中添加 **16 个 NOP 指令** 来复现该问题。
- **提供用于性能分析的代码链接**：一位成员分享了其代码的[链接](https://ppc-exercises.cs.aalto.fi/course/open2025/cp/cp2b/121316)，并指出注释掉末尾的 `printf` 语句会导致版本变慢。
   - 这是为了响应分享代码以分析潜在处理器跳转对齐问题的请求。



**提及的链接**: <a href="https://ppc-exercises.cs.aalto.fi/course/open2025/cp/cp2b/121316">Log in</a>: 未找到描述

  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1352090370988576860)** (3 messages): 

> `用于 ML/CUDA 的消费级 GPU，5080 对比云端算力额度，家庭 ML 开发` 


- **消费级 GPU：ML/CUDA 的可行选择？**：成员们正在思考购买像 **5080** 这样的**消费级 GPU** 对于在家进行 **ML/CUDA** 开发是否值得。
   - 问题围绕着是投资此类硬件更好，还是选择**云端算力额度**更好。
- **高性价比的家庭 ML 配置**：讨论集中在使用消费级 GPU 构建用于 Machine Learning 和基于 CUDA 任务的家庭配置。
   - 核心问题是，与使用云端解决方案相比，像 5080 这样 GPU 的性能和能力是否值得这项投资。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1352046188538499113)** (41 messages🔥): 

> `Orpheus TTS 模型，DeepSeek R1 成本，OpenAI 的 o1-pro 模型，Gemma 软件包，Perplexity 融资轮次` 


- **Orpheus 超越所有 TTS 模型**：新的开源 TTS 模型 **Orpheus** 已发布，根据[这条推文](https://x.com/eliasfiz/status/1902435597954003174?s=46)和[这段 YouTube 视频](https://youtu.be/Btos-LEYQ30?si=XjoZEuJT49jXOLRJ)，它声称性能优于 **ElevenLabs** 和 **OpenAI** 等开源及闭源模型。
- **DeepSeek R1 训练成本备受关注**：关于 **DeepSeek R1** 训练成本的估算正在讨论中，提到的数字约为 **600 万美元**，尽管一位成员在[这条推文](https://x.com/teortaxesTex/status/1902658735454953531)中指出李开复估计 2024 年整个 **DeepSeek** 项目的成本为 **1.4 亿美元**。
- **o1-pro 发布，支持 Vision 和 Function Calling**：**OpenAI** 已在其 API 中发布了 **o1-pro**，以更高的成本（**$150 / 1M** 输入 token 和 **$600 / 1M** 输出 token）提供更好的响应，适用于 Tier 1-5 的特定开发者，支持 Vision、Function Calling 和 Structured Outputs，已在[这条推文](https://x.com/openaidevs/status/1902485690958450871?s=46)中宣布，并在 [OpenAI 文档](https://platform.openai.com/docs/models/o1-pro)中详细说明。
- **Gemma 软件包简化微调**：引入了一个名为 **Gemma package** 的新库，简化了 **Gemma** 的使用和微调，[这条推文](https://x.com/osanseviero/status/1902456220876787763)中的文档包含了微调、分片（sharding）、LoRA、PEFT、多模态和分词（tokenization），可通过 *pip install gemma* 安装，文档位于 [gemma-llm.readthedocs.io](https://gemma-llm.readthedocs.io/en/latest)。
- **Perplexity 寻求巨额融资**：据 [Bloomberg](https://www.bloomberg.com/news/articles/2025-03-20/perplexity-in-early-talks-for-funding-at-18-billion-value?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTc0MjQ5MzI4OSwiZXhwIjoxNzQzMDk4MDg5LCJhcnRpY2xlSWQiOiJTVERYV01UMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.GYIVla5ZD3lp70ED36NxSKtCvWFpu8qrEaHIEPydQ9s) 报道，**Perplexity** 据传正就一轮 **5 亿至 10 亿美元** 的新融资进行早期谈判，估值为 **180 亿美元**，可能比 12 月的估值翻一番。


<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://www.openai.fm/">OpenAI.fm</a>: 为开发者提供的交互式演示，用于试用 OpenAI API 中新的 text-to-speech 模型</li><li><a href="https://x.com/OpenAI/status/1902737268852580717">来自 OpenAI (@OpenAI) 的推文</a>: 开发者们，请开启声音。</li><li><a href="https://x.com/alexalbert__/status/1902765482727645667?s=46">来自 Alex Albert (@alexalbert__) 的推文</a>: Web search 现已在 claude.ai 上线。Claude 终于可以搜索互联网了！</li><li><a href="https://x.com/osanseviero/status/1902456220876787763">来自 Omar Sanseviero (@osanseviero) 的推文</a>: 介绍 Gemma 软件包，这是一个用于使用和微调 Gemma 的极简库 🔥 包含以下文档：- Fine-tuning - Sharding - LoRA - PEFT - Multimodality - Tokenization！pip install gemma https://gemma-llm...</li><li><a href="https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1">DeepSeek R1 复现 o1 的方案及推理型 LM 的未来</a>: 是的，为 DeepSeek R1 敲响真正复现 o1 的钟声 🔔🔔🔔。接下来我们将走向何方。</li><li><a href="https://x.com/srush_nlp/status/1902736199636205914?s=46">来自 Sasha Rush (@srush_nlp) 的推文</a>: 一些个人消息：我最近加入了 Cursor。Cursor 是一个规模虽小但雄心勃勃的团队，他们创造了我最喜欢的 AI 系统。我们现在正在真实世界的编程环境中大规模构建前沿的 RL 模型...</li><li><a href="https://x.com/eliasfiz/status/1902435597954003174?s=46">来自 Elias (@Eliasfiz) 的推文</a>: 今天，我们发布了 Orpheus，这是一个开源的 TTS 模型，其能力超越了 ElevenLabs 和 OpenAI 等开源及闭源模型！(1/6)</li><li><a href="https://x.com/juberti/status/1902771172615524791?s=46">来自 Justin Uberti (@juberti) 的推文</a>: 今天有很多新的音频相关内容：- ASR：具有 SoTA 性能的 gpt-4o-transcribe - TTS：带有 playground（http://openai.fm）的 gpt-4o-mini-tts - Realtime API：新的降噪和语义 VAD - Agents SDK：...</li><li><a href="https://x.com/kevinweil/status/1902769861484335437?s=46">来自 Kevin Weil 🇺🇸 (@kevinweil) 的推文</a>: 🔊 今天为您带来三个新的音频模型！* 一个新的 text to speech 模型，让您可以控制节奏和情感——不仅是说什么，还有怎么说 * 两个 speech to text 模型，能够显著地...</li><li><a href="https://x.com/shiringhaffary/status/1902782551556575235">来自 Shirin Ghaffary (@shiringhaffary) 的推文</a>: 新消息：Perplexity 正在就一轮 5 亿至 10 亿美元的新融资进行早期谈判，估值为 180 亿美元，这将使其自 12 月以来的估值翻倍。ARR 也接近 1 亿美元。链接：https://www.bloombe...</li><li><a href="https://x.com/OpenAIDevs/status/1902817202358685880">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>: 🗣️ 00:00 介绍 01:32 Audio agents 03:27 Speech-to-text 06:18 Text-to-speech 08:48 Agents SDK。在我们的博客文章中阅读更多内容：http://openai.com/index/introducing-our-next-generation-audio-models/ 引用 OpenAI ...</li><li><a href="https://x.com/openaidevs/status/1902485690958450871?s=46">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>: o1-pro 现已在 API 中可用 @benhylak @literallyhimmmm @shl @joshRnold @samgoodwin89 @byamadaro1013 @adonis_singh @alecvxyz @StonkyOli @gabrielchua_ @UltraRareAF @yukimasakiyu @theemao @curious_vii...</li><li><a href="https://x.com/teortaxesTex/status/1902658735454953531">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: 李开复的估计：整个 DeepSeek 项目耗资 1.4 亿美元（至少在 2024 年，约占 90%）。我一直说是 2 亿美元。无论如何，尽管这看起来很不可思议，文锋可能真的只是…… *n...</li><li><a href="https://x.com/glaiveai/status/1902107399705522354?s=46">来自 Glaive AI (@GlaiveAI) 的推文</a>: 今天，我们发布了一个合成数据集，包含超过 2200 万条针对各领域通用提示词的推理轨迹（reasoning traces）。我们注意到缺乏包含多样化推理轨迹的大型数据集...</li><li><a href="https://reddit.com/r/LocalLLaMA/comments/1jes8ue/llama4_is_probably_coming_next_month_multi_modal/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://stratechery.com/2025/an-interview-with-openai-ceo-sam-altman-about-building-a-consumer-tech-company/">专访 OpenAI CEO Sam Altman：关于构建一家消费级科技公司</a>: 专访 OpenAI CEO Sam Altman，探讨构建 OpenAI 和 ChatGPT 的历程，以及成为一家“意外的”消费级科技公司意味着什么。</li><li><a href="https://youtu.be/Btos-LEYQ30?si=XjoZEuJT49jXOLRJ">政府知道 AGI 即将到来 | The Ezra Klein Show</a>: 通用人工智能（AGI）——一个在几乎所有认知任务上都能击败人类的 AI 系统——将在短短几年内问世。这就是人们...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: 来自 NVIDIA GTC 的简短播客 https://www.youtube.com/watch?v=AOL0RIZxJF0
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1352026871100997682)** (12 messages🔥): 

> `Monolingual Models, AI Safety, Interpretability` 


- **单语言模型引发关注**：成员们对 *“针对 350 种语言的单语言模型”* 这一表述展开了讨论，一些人感到困惑，因为通常预期模型应该是 **multilingual**（多语言）的。
   - 一位成员澄清说，该项目为 **350 种语言** 中的每一种都训练了一个或多个模型，最终在 [HF](https://huggingface.co/) 上发布了总计 **1154 个模型**。
- **CV 工程师转向 AI Safety 研究**：一位成员介绍自己是 CV 工程师，并表达了对参与 **AI safety** 和 **interpretability**（可解释性）研究的兴奋。
   - 他们表示有兴趣与小组中的其他人讨论这些话题。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1352061117521858672)** (25 messages🔥): 

> `Expert Choice Routing, Quantile Estimation for Thresholds, Gaussian Quantile Function, BatchTopK SAE, Node Limited Routing` 


- **探索专家选择路由 (Expert Choice Routing)**：成员们讨论了在自回归模型上进行专家选择路由，并在训练期间使用在线分位数估计来获取推理时的阈值。
   - 一位成员建议假设 router logits 服从 **Gaussian** 分布，计算 EMA 均值和标准差，然后使用 **Gaussian quantile function**（高斯分位数函数）。
- **用于稀疏性的总体分位数估计**：一位成员提议在推理时使用 **population quantiles**（总体分位数）的估计，旨在保持所需的整体平均稀疏度，并将其比作 *batchnorm*。
   - 另一位成员提到 dsv3 架构由于 *node limited routing*（节点受限路由）允许激活 **8-13 个专家**，但他们希望允许在 **0 到 N** 之间激活，其中简单 token 的激活数应接近 0。
- **提出柯尔莫哥洛夫压缩测试 (Kolmogorov Compression Test)**：一位成员分享了一篇论文链接，["The Kolmogorov Test"](https://arxiv.org/abs/2503.13992)，为代码生成 LLM 引入了一种“压缩即智能”的测试。
   - 柯尔莫哥洛夫测试 (KT) 涉及在推理时向模型展示一系列数据，并要求其生成能够产生该序列的最短程序。



**提及链接**：<a href="https://arxiv.org/abs/2503.13992">The KoLMogorov Test: Compression by Code Generation</a>：压缩是智能的核心。压缩任何数据序列的理论最优方法是找到输出该序列并停止的最短程序。然而，这种“...

  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1351999230864134355)** (23 messages🔥): 

> `Cohere Expanse 32B Knowledge Date, Critique of Comparing Cohere to OpenAI, Cohere Model via OpenRouter and Azure AI Search, Cohere model mimicking Mexican people, Connectors Support in Recent Models (cmd-R, cmd-A)` 


- **对 Cohere 竞争性批评的担忧**：一位成员对将 **Cohere** 与 **OpenAI** 进行比较提出了批评，认为这削弱了 Cohere 的独特优势，例如显著更大的 context size。
   - 他们建议 **Cohere** 应该专注于突出自身优势，而不是迷失在与竞争对手的比较中。
- **Cohere Command-R 表现出色但消耗 Token 较多**：一位用户通过 **OpenRouter** 测试了 **Cohere** 模型用于 **Azure AI Search**，并对输出结果印象深刻。
   - 然而，他们注意到每次请求的输入端消耗了 *80,000 个 token*。
- **Command-A 用西班牙语亲切交流**：一位来自墨西哥的用户报告说，即使没有特定提示，**Command-A** 感觉就像在与墨西哥人交谈。
   - 该模型*模仿*了他们的方言，其方式令他们感到意外地自然和友好。
- **Connectors 令当前的 Cmd 模型困惑**：一位用户探索了带有 **Slack 集成** 的 **Connectors**，但发现最近的模型（如 **cmd-R** 和 **cmd-A**）似乎不支持它们。
   - 旧模型返回 500 错误，且 Connectors 似乎已从 V2 版本的 API 中移除，这令人失望，因为它们简化了数据处理。
- **工具调用 (Tool-Calls) 对传统技术的影响**：一位用户讨论了从 **Connectors** 到 **Tools** 的转变，质疑工具是否能提供一对一的替代。
   - 他们表达了对失去 Connectors “魔力”方面的担忧，例如原生的搜索查询生成、结果解析、chunking、embedding 和 reranking。


  

---

### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1352019959462760530)** (7 messages): 

> `OpenAI API context length limitations, Cohere vs OpenAI API, Aya model usage with Ollama, Checking Cohere API free limit` 


- **OpenAI API 上下文长度受限**：一位成员表示更倾向于使用 **Cohere's API**，因为 **OpenAI's API** 的上下文大小限制仅为 **128,000**，而 Cohere 提供 **200,000**。
- **Cohere 兼容性 API 说明**：一位成员澄清说，使用兼容性 API（compatibility API）不会改变上下文长度，但会导致你*失去对 API 响应中 `documents` 和 `citations` 等 Cohere 特定功能的访问权限*。
   - 他们还提到，Cohere *认为我们拥有更易于使用的聊天流式响应（chat streaming response）*，但如果你已有基于 OpenAI 运行的代码并只想将其指向我们的模型，可以随意使用兼容性 API（compat api）。
- **在本地 Python Flask 中集成 Aya 模型**：一位成员询问在 **Ollama** 中本地托管时，如何在 **Python Flask app** 中使用 **Aya model**。
   - 另一位成员建议，你可以从 localhost 调用 API，或者通过监听 0.0.0.0 的环境变量进行调用。
- **达到免费额度限制时的 Cohere API 超时错误**：一位用户询问在遇到超时错误时如何检查是否已达到免费额度（free limit），以确定是否在一段时间内无法发送请求。


  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1352013136827519071)** (1 messages): 

> `MCP Server, Cohere Command A, Positive News` 


- **基于 Cohere 的正面新闻 MCP Server 发布**：一位成员构建了一个名为 *Goodnews MCP* 的 **MCP server**，它在其工具 `fetch_good_news_list` 中使用 **Cohere Command A** 为 MCP 客户端提供积极、令人振奋的新闻。
   - 该系统使用 **Cohere LLM** 对近期头条进行排名，返回最正面的文章，代码可在 [GitHub](https://github.com/VectorInstitute/mcp-goodnews) 上获取。
- **正面新闻 MCP Server 的 GitHub 仓库**：**Goodnews MCP** 服务器的 GitHub 仓库地址在[这里](https://github.com/VectorInstitute/mcp-goodnews)。
   - 该仓库包含一个简单的 **MCP application** 代码，用于提供精选的正面和励志新闻故事。



**提到的链接**：<a href="https://github.com/VectorInstitute/mcp-goodnews">GitHub - VectorInstitute/mcp-goodnews: A simple MCP application that delivers curated positive and uplifting news stories.</a>：一个提供精选正面和励志新闻故事的简单 MCP 应用程序。- VectorInstitute/mcp-goodnews

  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1352012390044008600)** (2 messages): 

> `RAG Federation, Agentic Apps/Research, Vector Institute` 


- **Vector Institute 加入**：来自 **Vector Institute** 的 Andrei（曾就职于 **LlamaIndex**）向频道介绍了自己。
   - 他目前正致力于 **RAG** 联邦化，并很快将转向一些 **agentic apps/research**（智能体应用/研究）。
- **Python 和 Rust 是最喜欢的工具**：Andrei 提到他最喜欢的技术/工具是 **Python 和 Rust**。
   - 他希望从社区中获得技巧，学习新方法，并讨论行业/研究趋势。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1352037670817366068)** (4 messages): 

> `Photonics, Integrated CPU, Ruben GPUs, CX9, DIGITs successor` 


- **硅光子推测引发 GPU 热议**：讨论集中在 **Ruben GPUs** 中的**硅光子（photonics）**和**集成 CPU** 是数据中心模型专属，还是会扩展到消费级版本（可能是 **6000 系列**）。
   - 有人提出了 **CX9** 拥有共封装光学器件（co-packaged optics）的可能性，暗示 **DIGITs 继任者**可能会利用此类技术，而 **CPU** 已确认将用于 **DGX workstations**。
- **Ruben GPUs 与硅光子集成**：成员们推测了专门针对数据中心级 **Ruben GPUs** 的**硅光子（photonics）**和**集成 CPU** 的集成情况。
   - 有建议认为消费级 **Ruben GPUs**（可能是 **6000 系列**）可能不会获得相同水平的集成。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1352023363362619454)** (23 条消息🔥): 

> `Mojo 中的 debug_assert, List 边界检查, Mojo 编译器选项, Mojo 中的未定义行为, Mojo 测试默认值` 


- **调试断言需要额外的编译器选项**：在 Mojo 标准库中启用调试断言（debug asserts）需要一个额外的编译选项 `-D ASSERT=_`，这一点并未被广泛宣传，详见 [debug_assert.mojo](https://github.com/modular/max/blob/main/mojo/stdlib/src/builtin/debug_assert.mojo#L88-L100)。
   - 有人指出使用 `-g` 并不会启用断言，且预期在使用 `-Og` 编译时应自动开启它们。
- **Mojo List 索引因 UB 打印 0**：当 **Mojo List** 发生索引越界时，由于未定义行为（UB），它会打印 **0** 而不是抛出错误。
   - 该问题源于代码对列表进行了越界索引，进入了内核提供的零初始化内存区域。
- **关于 debug_assert assert_mode 参数的澄清**：`debug_assert` 中的 `assert_mode` 参数控制该特定调用的默认行为，不同的模式由特定的编译器选项触发，正如[此处文档](https://docs.modular.com/mojo/stdlib/builtin/debug_assert/debug_assert/)所述。
   - 例如，如果使用了 `mojo -D ASSERT=all`，则会执行 `debug_assert[assert_mode="none"]`。
- **关于默认断言行为的讨论**：引发了关于 `debug_assert` 默认行为的讨论，特别是围绕 `debug_assert[assert_mode="none"]` 的困惑，以及是否应该在调试模式下默认启用它。
   - 有建议认为在调试模式下运行程序时，应启用所有断言。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modular/max/blob/main/mojo/stdlib/src/collections/list.mojo#L901-L907">modular/max 仓库中的 mojo/stdlib/src/collections/list.mojo</a>：MAX 平台（包含 Mojo）。通过在 GitHub 上创建账号为 modular/max 的开发做出贡献。</li><li><a href="https://github.com/modular/max/blob/d7b7747004e6004d9e587772c595b6b8a89e5051/mojo/stdlib/src/builtin/debug_assert.mojo#L53C1-L60C1">modular/max 仓库中的 mojo/stdlib/src/builtin/debug_assert.mojo</a>：MAX 平台（包含 Mojo）。</li><li><a href="https://github.com/modular/max/blob/main/mojo/stdlib/src/builtin/debug_assert.mojo#L88-L100">modular/max 仓库中的 mojo/stdlib/src/builtin/debug_assert.mojo</a>：MAX 平台（包含 Mojo）。
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1352311170291204178)** (2 条消息): 

> `DeepLearningAI 短期课程, AI 语音助手流水线` 


- **LlamaIndex & DeepLearningAI 推出代理工作流短期课程**：与 **DeepLearningAI** 合作推出了一门关于如何构建代理工作流（agentic workflows）的新短期课程，内容包括表单解析、自动提取关键字段以及使用检索增强生成（**RAG**）。
   - 更多详情可见 [Twitter](https://t.co/qvqNj7MJbn)。
- **AMD GPUs 通过 ROCm 和 LlamaIndex 驱动 AI 语音助手流水线**：发布了一个教程，演示如何创建一个多模态流水线，利用 **AMD GPUs** 进行语音转文本、使用 **RAG** 获取上下文感知响应，并将文本转回语音。
   - 该教程涵盖了 **ROCm** 环境的搭建以及 **LlamaIndex** 的集成；更多信息请见[教程链接](https://t.co/jdG2VT0cbf)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1352277693563146281)** (20 messages🔥): 

> `LLM.as_structured_llm parallel tool calls, MariaDBChatStore, llamaparse QA` 


- **LLM.as_structured_llm 需要并行工具调用支持**：一位成员注意到在使用 `LLM.as_structured_llm` 的 `.chat` 方法时缺少 `allow_parallel_tool_calls` 选项，并建议应该支持该选项，或许可以通过扩展 `.as_structured_llm()` 调用以接受类似 `allow_parallel_tool_calls=False` 的参数。
   - 另一位成员建议直接使用 `FunctionCallingProgram` 以获得更多自定义功能，并为 OpenAI 设置 `additional_kwargs={"parallel_tool_calls": False}`，参考了 [OpenAI API 文档](https://platform.openai.com/docs/api-reference/chat/create#chat-create-parallel_tool_calls)。
- **推理标签困扰 ChatMemoryBuffer**：一位使用 **Ollama** 和 **qwq model** 的用户正苦于 `<think>` 推理标签出现在 `ChatMemoryBuffer` 的 `text` 块中，并寻求在使用 `ChatMemoryBuffer.from_defaults` 时移除它们的方法。
   - 另一位用户建议对 LLM 输出进行手动后处理，因为 **Ollama** 不提供内置过滤功能；原用户表示愿意分享他们的 MariaDBChatStore 实现（PostgresChatStore 的克隆版本）。
- **llamaparse QA 难题**：一位用户正在寻求关于如何对数百个使用 **llamaparse** 解析的 PDF 文件进行 QA 的建议，并指出有些文件解析完美，而另一些则生成了荒谬的 Markdown。
   - 他们还对如何为需要不同处理方式的文档实现不同的解析模式感到好奇。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1351995552040091699)** (10 messages🔥): 

> `Nvidia Delays, Gemma 3 Fine Tuning, Torchtune sprint` 


- **Nvidia 漫长的等待**：一位用户分享了一张图片，显示 **Nvidia** 的新硬件迟到了。
   - 另一位用户补充说这是 *“Nvidia 风格”*，并举例说 **H200s** 在 **2 年前** 就发布了，但直到 **6 个月前** 才向客户供货。
- **Gemma 3 微调即将到来**：一位用户询问是否会支持 **Gemma 3** 微调。
   - 另一位用户回答说，有一位成员提交了 [针对 gemma 纯文本的 PR](https://github.com/pytorch/torchtune/pull/2485)，并补充说他们会尝试加速合并该 PR，然后再考虑后续添加图像处理能力。
- **休假成员开启冲刺以继续 Torchtune 开发**：一位成员表示，由于他们的 *“假期正在转变为 Torchtune 冲刺”*，他们将尝试尽快继续处理 **Gemma 3**。
   - 另一位用户告诉他们享受假期，这些工作可以稍后再做。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1352208718367821895)** (2 messages): 

> `nv-fabricmanager, driver versions` 


- **nv-fabricmanager 在驱动版本不匹配时报错**：如果 **nv-fabricmanager** 的驱动版本与显卡版本不同，可能会发生错误，最近在一些按需使用的 VM 上观察到了这种情况。
   - 运行 **nv-fabricmanager** 时会报告此类错误。
- **nv-fabricmanager 的驱动版本问题**：当 **nv-fabricmanager** 的驱动版本与显卡的驱动版本不匹配时，可能会抛出错误。
   - 这种情况特别是在某些按需使用的 VM 上被观察到。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1352009241082462269)** (5 messages): 

> `ML4SCI/task1, Adam Optimizer` 


- **使用 Adam 训练的模型达到低损失**：一位成员报告称使用 **Adam optimizer** 训练模型，达到了 **0.2s** 左右的损失（loss）。
   - 设置代码可在 [GitHub](https://github.com/kayo09/gsoc_2025/tree/main/ML4SCI/task1) 上找到。
- **Discord 规则执行**：一位成员被提醒遵守 Discord 规则。
   - 规则规定 *“这是一个讨论 tinygrad 开发和 tinygrad 使用的地方。”*



**提到的链接**：<a href="https://github.com/kayo09/gsoc_2025/tree/main/ML4SCI/task1">gsoc_2025/ML4SCI/task1 at main · kayo09/gsoc_2025</a>：GSOC 2025！祝编码愉快！☀️。通过在 GitHub 上创建账号来为 kayo09/gsoc_2025 的开发做出贡献。

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1352131383434285099)** (3 条消息): 

> `AgentX Research Track, LLM agents, Multi-agent systems, Advanced AI research` 


- **用户表达对 AgentX Research Track 的兴奋**：一位用户表达了加入 **AgentX Research Track** 的兴奋和兴趣。
   - 该用户热衷于与导师和博士后合作，并通过研究 **LLM agents** 和 **multi-agent systems** 为该项目做出贡献。
- **用户承诺在研究中保持主动性和独立性**：一位用户保证他们将在 **AgentX Research Track** 中主动且独立地推动其研究。
   - 他们承诺在给定的时间范围内交付高质量的工作，并对任何能增加其入选机会的支持表示感谢。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/)** (1 条消息): 

kotykd: 我可以使用 DSPy 实现类似这样的功能吗？ https://arxiv.org/abs/2502.06855
  

---


---


---


---


---


{% else %}


> 完整的频道详细解析已针对电子邮件进行了截断。 
> 
> 如果您想查看完整的解析，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}