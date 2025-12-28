---
companies:
- openai
- deep-learning-ai
- meta-ai-fair
- google-deepmind
- saama
- langchain
- nvidia
date: '2025-01-18T02:33:34.160647Z'
description: '以下是为您翻译的中文内容：


  **DeepSeek-V3** 是一款拥有 **6710 亿参数的混合专家模型 (MoE)**，在编程和数学基准测试中超越了 **Llama 3.1 405B**
  和 **GPT-4o**。**OpenAI** 宣布将于 **2023 年 4 月 27 日**发布 **GPT-5**（注：原文日期如此）。**ai-gradio**
  中的 **MiniMax-01 Coder 模式** 能够一次性构建出国际象棋游戏。**Meta** 的研究强调了缩放视觉分词器（visual tokenizers）时的权衡。**Google
  DeepMind** 通过推理时扩展（inference-time scaling）提升了扩散模型的质量。**RA-DIT** 方法通过微调大语言模型（LLM）和检索器来优化
  RAG（检索增强生成）的响应效果。美国提议对 AI 芯片和模型实施三级出口限制体系，将 **中国** 和 **俄罗斯** 等国排除在外。披露了 AI 聊天机器人中涉及
  CSRF（跨站请求伪造）和提示词注入的安全漏洞。人们对超级智能和武器级 AI 模型表示了担忧。**ai-gradio** 的更新包括对 NVIDIA NIM 的兼容以及
  **cosmos-nemotron-34b** 等新模型。**LangChain** 与 **Claude-3-haiku** 集成，用于构建具有持久化记忆的
  AI 智能体。**Triton Warp 特化（specialization）** 优化了用于矩阵乘法的 GPU 利用率。**Meta** 微调的 Llama
  模型 **OpenBioLLM-8B** 和 **OpenBioLLM-70B** 专注于个性化医疗和临床试验。'
id: 16216ffd-69b2-4dc6-a394-725a31ef929a
models:
- deepseek-v3
- llama-3-1-405b
- gpt-4o
- gpt-5
- minimax-01
- claude-3-haiku
- cosmos-nemotron-34b
original_slug: ainews-not-much-happened-today-9518
people:
- akhaliq
title: 今天没发生什么。
topics:
- mixture-of-experts
- coding
- math
- scaling
- visual-tokenizers
- diffusion-models
- inference-time-scaling
- retrieval-augmented-generation
- ai-export-restrictions
- security-vulnerabilities
- prompt-injection
- gpu-optimization
- fine-tuning
- personalized-medicine
- clinical-trials
- ai-agents
- persistent-memory
---

<!-- buttondown-editor-mode: plaintext -->**一个安静的长周末正是我们所需要的。**

> 2025年1月16日至1月17日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **34** 个 Discord 社区（**225** 个频道，**2327** 条消息）。预计节省阅读时间（以 200wpm 计算）：**298 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

[o3-mini 即将到来](https://x.com/sama/status/1880356297985638649)。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型发布与评估**

- **DeepSeek-V3 的进展**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1880087643964199260) 宣布 **DeepSeek-V3** 采用了拥有 **6710 亿参数的 Mixture-of-Experts 架构**，在关键基准测试中超越了 **Llama 3.1 405B** 和 **GPT-4o**，特别是在 **Coding 和数学任务**方面。
  
- **GPT-5 发布公告**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1880111090824278189) 分享了 **OpenAI** 将于 **2023 年 4 月 27 日**发布 **GPT-5** 的消息，在社区内引起了巨大期待。
  
- **MiniMax-01 Coder 可用性**：[@_akhaliq](https://twitter.com/_akhaliq/status/1880059318785176043) 在 **ai-gradio** 中引入了 **MiniMax-01 Coder 模式**，重点展示了其在单次尝试（single shot）中构建**可运行象棋游戏**的应用。

**研究论文与技术见解**

- **扩展 Visual Tokenizers**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1880164031987589413) 展示了 **Meta 关于扩展 Visual Tokenizers 的新论文**研究结果，强调 **小型 Encoder 是最优的**，且**增加 Bottleneck 大小**可以**提高重建质量**，但会**降低生成性能**。

- **Diffusion Models 的 Inference-Time Scaling**：[@sainingxie](https://twitter.com/sainingxie/status/1880106419573387528) 讨论了 **Google DeepMind 关于 Inference-Time Scaling 的最新工作**，该工作通过增强**搜索算法和验证器（verifiers）**来提高 **Diffusion Model 的样本质量**。

- **用于 RAG 设置的 RA-DIT 方法**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1880021894054985922) 详细介绍了 **Retrieval-Augmented Dual Instruction Tuning (RA-DIT)** 方法，该方法通过**同时微调 LLM 和检索器（retrievers）**来**增强 RAG 设置中的响应质量**。

**AI 政策、监管与安全**

- **美国 AI 出口限制**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1880341893873033511) 概述了**美国拟议的先进 AI 技术出口限制**，为获取 **AI 芯片和模型**建立了**三级体系**，其中**中国**和**俄罗斯**等**第三类国家**将被**完全排除在外**。

- **AI Chatbot 漏洞**：[@rez0__](https://twitter.com/rez0__/status/1880016611568197663) 揭示了 **AI Chatbot 中的 CSRF 和 Prompt Injection 漏洞**，强调了与**前端集成**相关的**安全风险**。

- **AGI 与超人工智能担忧**：[@danintheory](https://twitter.com/polynoamial/status/1880344112521781719) 强调**超人工智能（Superintelligence）尚未实现**，而 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1880252602396348467) 则对 **R1 被认定为武器级模型**表示担忧，这引发了**监管和国家安全问题**。

**工具、框架与开发**

- **AI-Gradio 增强**：[@_akhaliq](https://twitter.com/_akhaliq/status/1880314518753956261) 介绍了 **ai-gradio** 的更新，包括 **NVIDIA NIM 兼容性**和 **cosmos-nemotron-34b** 模型，便于 **AI 应用的快速部署**。

- **LangChain 集成**：[@LangChainAI](https://twitter.com/LangChainAI/status/1880299047178715244) 展示了如何使用 **LangChain**、**PostgreSQL** 和 **Claude-3-haiku LLM** 构建**具有持久记忆的 AI Agent**，支持 **Python** 和 **Node.js** 实现。

- **Triton Warp Specialization**：[@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1880098533350732203) 解释了 **Triton 的 Warp Specialization**，它可以**自动调度 Warp 组**并发运行，从而**优化矩阵乘法等任务的 GPU 资源利用率**。

**AI 行业与用例**

- **基于 Llama 模型的个性化医疗**：[@AIatMeta](https://twitter.com/AIatMeta/status/1880338816491499737) 介绍了 **OpenBioLLM-8B 和 OpenBioLLM-70B**，这是由 **Saama** 微调的 **Llama 模型**，旨在**加速临床试验**和**个性化医疗**。

- **AI 对冲基金开发**：[@virattt](https://twitter.com/virattt/status/1880031667873583556) 描述了他们的 **AI hedge fund**，该基金通过一个包含 **valuation**（估值）、**technical**（技术面）、**sentiment**（情绪）和 **fundamentals analysts**（基本面分析师）以及 **risk agents** 和 **portfolio managers** 的系统来 **交易多只股票**。

- **AI 在认知行为疗法中的应用**：[@omarsar0](https://twitter.com/omarsar0/status/1880283025595867631) 分享了关于 **AutoCBT** 的见解，这是一个用于 **Cognitive Behavioral Therapy** 的 **multi-agent framework**，通过 **dynamic routing** 和 **memory mechanisms** 提升了 **对话质量**。

**梗/幽默**

- **对模糊 AI 炒作的批评**：[@polynoamial](https://twitter.com/polynoamial/status/1880334203214291231) 表达了对 **模糊 AI 炒作** 的沮丧，呼吁社区内进行更多 **具体且透明的讨论**。

- **AI Agents 尚未准备好投入大规模应用**：[@HamelHusain](https://twitter.com/HamelHusain/status/1880157373119201612) 幽默地承认 **Devin (AI SWE)** “**尚未完全准备好投入大规模应用**”，同时推荐 **Aider** 作为免费替代方案。


---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. ElevenLabs 的 TTS：卓越质量背后的因素**

- **ElevenLabs 在做什么？为什么它这么出色？** ([Score: 320, Comments: 130](https://reddit.com/r/LocalLLaMA/comments/1i31ji5/what_is_elevenlabs_doing_how_is_it_so_good/))：**ElevenLabs** 的文本转语音 (TTS) 技术明显优于本地模型，这引发了关于它使用的是 **full Transformer model** 还是 **Diffuser** 的讨论。帖子推测该公司是否对人体解剖结构进行了建模以提高模型的准确性。
  - 评论者的共识是，**高质量数据**对于实现卓越的文本转语音 (TTS) 性能至关重要，**ElevenLabs** 利用实际的有声读物数据超越了竞争对手。**Kokoro TTS** 被提及作为一个开源替代方案，但被指出在情感表达方面逊于 ElevenLabs。
  - 几条评论强调，**ElevenLabs** 的成功归功于使用相对较小的计算设置 (32x3090 GPUs) 并专注于高质量数据集而非合成数据。一些人推测 ElevenLabs 可能是基于 **Tortoise** 并进行了专有优化，强调了使用优质语音样本进行 **finetuning** 的重要性。
  - 讨论还涉及由于成本和法律问题，获取高质量、经授权的有声读物数据集所面临的挑战，并建议 **Mozilla** 可以在委托专业配音演员制作训练数据集方面发挥作用。**公共领域**资源 **LibriVox** 被认为是此类数据的潜在来源。


**主题 2. OpenWebUI 的 Canvas：增强的多语言支持**

- **OpenWebUI Canvas 实现 —— 即将推出！（更好的 Artifacts）** ([Score: 176, Comments: 34](https://reddit.com/r/LocalLLaMA/comments/1i3as1m/openwebui_canvas_implementation_coming_soon/))：**OpenWebUI** 正在增强其 **Canvas** 功能，将语言支持从 HTML、CSS、JavaScript 和 SVG 扩展到包括 **C#, Python, Java, PHP, Ruby, Bash, Shell, AppleScript, SQL, JSON, XML, YAML, Markdown, 和 HTML**。此外，新功能将允许用户在 Web 设计的 **Design view**（设计视图）和 **Code view**（代码视图）之间切换，预计在未来几周内提交 pull request。
  - 用户建议通过插件/扩展模型来扩展 **OpenWebUI**，以允许更多自定义，类似于浏览器。人们对在未来版本中支持 **Latex**、**dot**、**gnuplot**、**R**、**VHDL** 和 **Powershell** 等其他技术表现出兴趣。
  - 几位用户对集成 **mermaid.js** 和 **chart.js** 等图表库表示热烈欢迎，其中 **mermaid** 已经得到支持。一些用户指出 **mermaid** 对绘图的影响是变革性的。
  - 用户希望将 **OpenWebUI** 与 **GitHub Copilot Edit** 等工具进行比较，并询问其编辑功能的工作原理，特别是关于大文件处理。一些用户有兴趣在 OpenWebUI 之上构建更复杂的操作，如 **OS integration** 和 **CoT solutions**。


**主题 3. DeepSeek V3 vs Claude 3.5 Sonnet：分析实际优势**

- **DeepSeek V3 是否被过度炒作？** ([Score: 116, Comments: 93](https://reddit.com/r/LocalLLaMA/comments/1i2y810/is_deepseek_v3_overhyped/))：作者将 **DeepSeek V3** 与 **3.5 Sonnet** 进行了对比，指出虽然基准测试结果相当，但 DeepSeek V3 缺乏 Sonnet 那种令人印象深刻的感觉和细腻的输出。他们将 DeepSeek V3 描述为一个具有极少人类强化学习的大规模基础模型，这与 **OAI** 和 **LLaMa** 等模型形成对比。
  - **成本与性能**：**DeepSeek V3** 因以**极低的成本提供约 75% 的 Sonnet 性能**而受到称赞，用户注意到在使用过程中节省了大量成本。**Recoil42** 强调 **DeepSeek** 的成本效益极高，足以在大多数任务中不限量使用，使其成为日常编码和简单任务的首选，而 **Sonnet** 则保留用于更复杂的问题。
  - **模型比较与用例**：**DeepSeek V3** 以其经济性和多功能性著称，特别是在 **Java** 和 **C** 等编码任务中，它在某些领域优于 **Sonnet**。然而，**Sonnet** 被认为在 **UI generation** 以及针对 **React Python** 等特定语言的后期训练方面更胜一筹，**Charuru** 强调 **Sonnet 独特的 prompt engineering** 增强了其类人交互。
  - **开源与可访问性**：**DeepSeek V3** 因其开源和可访问性而受到欢迎，允许用户不受限制或不受道德说教地利用其功能，这与其他一些模型不同。**Odd-Environment-7193** 欣赏其详尽的回答和适应性，使其成为全栈工程师和寻求现代、灵活 AI 模型的人士的宝贵工具。

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. OpenAI 任务管理的不完善：用户挫败感显现**

- **[请，我求你了。让它停下来……](https://i.redd.it/nn1h36xx1kde1.jpeg)** ([评分: 353, 评论: 74](https://reddit.com/r/OpenAI/comments/1i3g7hy/please_i_beg_you_make_it_stop/)): 该帖子作者表达了对 AI 任务自动化的挫败感，特别是关于设置 **Arsenal**（阿森纳）足球比赛提醒和每日世界新闻摘要。尽管尝试通过 **ChatGPT** 取消这些任务，提醒依然存在，导致过多的通知和电子邮件。
  - **AI Misalignment**（AI 对齐失调）被强调为一个现实世界的问题，用户对尽管尝试取消但仍持续不断的通知表示沮丧。**Levoniust** 评论称这是 AI 对齐失调的一个显著案例。
  - **任务自动化挑战**：用户分享了相关经历，**Ziscz** 提到即使可以在设置中关闭通知，停止自动化任务依然很困难。
  - **幽默轶事**：关于 **Arsenal** 的评论增加了帖子的共鸣感，几位用户分享了关于足球比赛和通知的个人故事或笑话。

---

# AI Discord 综述

> 由 o1-preview-2024-09-12 提供的总结之总结

**主题 1. 重大融资轮次与公司里程碑**

- [**Cursor IDE 融资 1.05 亿美元以变革编程**](https://x.com/cursor_ai/status/1880003590493991072): **Cursor IDE** 宣布从 Thrive Capital、Andreessen Horowitz 和 Benchmark 获得了 **1.05 亿美元** 融资，激发了对未来更新的乐观情绪。社区期待这笔资金能带来代码生成功能的显著增强、更快的修复速度以及更广泛的模型支持。
- [**Anysphere 获得 1.05 亿美元融资以实现代码自动化**](https://www.cursor.com/blog/series-b): **Anysphere** 完成了 **1.05 亿美元** 的 B 轮融资，旨在为开发者推进 AI 驱动的编程工具。该投资旨在服务数百万程序员，反映了对 AI 驱动的开发者工具的强劲信心，并预示着代码自动化领域的激动人心发展。
- **Aider 庆祝 GitHub Star 数突破 2.5 万**: **Aider** AI 编程助手在 **GitHub** 上的 Star 数超过了 **25,000**，标志着一个重要的里程碑。社区成员赞扬其作为协作编程中的杰出工具所取得的成功，认可了其对开发者生产力的影响。

**主题 2. AI 模型开发与性能进展**

- [**NanoGPT Speedrun 在 3 分钟内完成模型训练**](https://x.com/leloykun/status/1880301753213809016): 一项新的 **NanoGPT speedrun** 在 8xH100 集群上实现了 **3 分钟** 以内的训练完成，每次尝试成本约为 **0.40 美元**。这展示了通过 **modded-nanogpt** 代码实现的训练效率的巨大提升，突显了 AI 模型优化方面的进展。
- [**Google 发布 TITANS 以增强记忆能力**](https://arxiv.org/abs/2501.00663v1): **Google Research** 介绍了 **TITANS**，这是一种使用动态子模型来模拟类记忆功能的模型架构。虽然它改善了 **Transformer** 处理长序列的能力，但持续学习（continuous learning）仍处于研发阶段，引发了关于未来进展的讨论。
- [**MiniMax-01 统一了注意力机制**](https://arxiv.org/abs/2501.08313): **MiniMax-01** 论文提出了一种统一 **MHA** 和 **GQA** 的模型，以高效处理更长的上下文。社区成员称赞了其易于理解的数学推导和开源代码发布，指出其在处理 AI 模型长序列方面的潜在影响。

**主题 3. 增强开发者工作流的 AI 工具与集成**

- [**TraycerAI 在 Cursor AI 中自动化代码库任务**](https://x.com/sanketdongre369/status/1880159755337101316): **TraycerAI** 扩展程序通过在 **Cursor AI** 中跟踪整个代码库、自动化任务并生成实现计划，给用户留下了深刻印象。开发者赞赏其增强的工作流和效率，强调了该工具简化复杂编程项目的能力。
- [**Windsurf Wave 2 携网页搜索与记忆功能登场**](https://codeium.com/blog/windsurf-wave-2): **Codeium** 发布了 **Windsurf Wave 2**，为 **Cascade** 引入了网页搜索功能和自动生成的记忆。此更新允许用户将实时网络上下文纳入对话，并在不同会话间保持连续性，显著提升了用户体验。
- **MCP Marketplace 简化了 Servlet 安装**: **Sage** 推出了 **MCP Marketplace**，支持在 iPad、iPhone 和 Mac 上一键安装 **MCP** servlet。社区成员赞扬了这种无摩擦的部署方式，认为这是跨平台可访问性和开发者便利性方面的一个充满希望的飞跃。

**主题 4. AI 模型使用与实现中的挑战与问题**

- **Bolt 和 Cursor IDE 用户反馈挫败感**：用户对 **Bolt** 表达了明显的挫败感，指出了诸如错误的代码删除和 token 使用量虚高等问题，这导致用户需要更好的 prompt 实践。同样，**Cursor IDE** 用户在集成 **Claude** 时面临较长的等待时间，损害了实时可用性，促使一些人考虑替代方案。
- **Perplexity Pro 模型设置引发困惑**：**Perplexity Pro** 用户遇到了某些模型无法识别的问题，即使在排查故障后依然存在。社区对响应质量下降和模型性能不一致表示担忧，寻求更可靠体验的改进。
- **OpenRouter 活动页面引发困惑**：用户对 **OpenRouter** 的 **activity page** 提出质疑，报告称不同 key 的使用图表显示完全相同。他们怀疑这是一个 bug，并强调需要更好的按 key 统计的使用指标，引发了关于数据可能存在误导的讨论。

**主题 5：AI 社区倡议与活动**

- [**Women in AI 助力 RAG Hackathon**](https://t.co/2Bzg80dh29)：组织者邀请女性技术人员参加在帕洛阿尔托举行的 **Women in AI RAG Hackathon**，重点关注基于开源向量数据库 **Zilliz** 的 **Retrieval-Augmented Generation**。该活动旨在促进 AI 领域女性之间的社交和导师指导，强调该领域的协作增长。
- [**Agent Recipes 为 AI Agents 提供代码模板**](https://x.com/nutlope/status/1879587920744788172)：一个名为 **Agent Recipes** 的新网站为 Agent 工作流提供代码模板，开发者可以轻松地将其集成到自己的 AI 应用程序中。早期用户称赞了使用提供的代码片段实现基于 Agent 的解决方案的便利性和速度。
- [**关于 Large Language Models 基础的新书发布**](https://arxiv.org/abs/2501.09223)：一本涵盖 Large Language Models 基础知识的综合性书籍发布，重点关注预训练、生成式架构、prompt 方法和对齐方法。该书面向学生和从业者，为现代语言模型开发提供了全面的基础。

---

# PART 1: 高层级 Discord 摘要

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Lucide 图书管理员拯救 Bolt**：StackBlitz 解决了 **Lucide 图标未找到错误**，让 Bolt 的 Agent 可以调用整个图标库，正如 [StackBlitz 推文](https://x.com/stackblitz/status/1880291208574235134)中所记录的那样。
   - 他们引入了**确定性图标**以减少猜测，社区对无需额外 token 或调试的实时修复表示赞赏。
- **Prompt 技巧：React 与 NPM**：成员们发现，在 React 代码中指示 AI 添加 **NPM 软件包**可以防止部分代码编辑或“删减”，从而改善功能。
   - 他们还建议明确 AI 何时应扩展现有部分，以保持关注点而不是重写元素。
- **TraycerAI 与文件文档的协同效应**：社区反馈赞扬了 **TraycerAI** 扩展，该扩展可以在 **Cursor AI** 中跟踪整个代码库，自动执行任务并生成实施计划。
   - 一些人还建议为一个 PDF 注释 Web 应用建立一个包含详尽文件结构文档的**指令文件夹**，但他们偶尔会发现 AI 生成虚构的细节。
- **Bolt 的 Bug 与 Git 带来的收益**：由于用户报告 Bolt 中存在**错误的代码删除**、token 使用量虚高以及需要更好的 prompt 实践，挫败感很高。
   - 计划中的 **Git 集成**将允许用户直接将仓库克隆到 Bolt 中，这可能会减少这些问题并简化项目管理。
- **Supabase 障碍与域名梦想**：**Supabase** 的连接器导致无效的 UUID 错误，引发了记录输入以定位不匹配点的建议。
   - 一位用户同时在开发一个域名爬虫来识别即将过期的域名，为那些有兴趣抢注有价值 URL 的人构思潜在利润。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RWKV 测试进展迅速**：在对 [BlinkDL's RWKV Gradio](https://huggingface.co/spaces/BlinkDL/RWKV-Gradio-1) 的检查中，**RWKV** 0.4B 模型表现出强劲的结果，但在 **box puzzle** 困惑度（perplexities）方面表现挣扎。
   - 社区讨论建议，更多的训练调整（如 **CoT methods**）可能会解决这些棘手的任务，并进一步提升 RWKV 的性能。
- **NanoGPT 竞速（Speedrun）降低训练成本**：一项新的 **NanoGPT speedrun** 创下了在 8xH100 集群上不到 **3 分钟**完成的纪录，每次尝试的成本约为 **$0.40**。
   - [leloykun 的推文](https://x.com/leloykun/status/1880301753213809016) 展示了 **modded-nanogpt** 中进一步的代码优化，大幅缩减了计算时间，令旁观者印象深刻。
- **QRWKV 项目旨在实现线性预填充（Linear Prefill）**：**QRWKV** 致力于转换 Transformer 模型以实现更高效的 prefix 处理，详见 [Q-RWKV-6 32B Instruct Preview](https://substack.recursal.ai/p/q-rwkv-6-32b-instruct-preview)。
   - 爱好者们提到了即将推出的 **QRWKV7** 方案，希望能看到在多个基准测试中的持续提升。
- **压缩梯度（Gradient Gusto with Compression）**：工程师们讨论了 **Deep Gradient Compression** 技术，以减少分布式 **SGD** 中的带宽占用，参考了[这篇论文](https://arxiv.org/abs/1712.01887)。
   - 随着这些压缩理念的整合，爱好者们看到了大规模训练的潜力，尽管在主流设置中的采用仍然有限。
- **上下文预热（Context Warmup）促进增长**：一种灵活的 **sliding window** 方法将上下文长度扩展到 **~1856** tokens，让训练者在不丢失数据顺序的情况下提升容量。
   - 支持者表示，这种方法减少了训练难题并确保了更好的文本连续性，从而产生更稳健的模型输出。



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 获得 1.05 亿美元融资**：Cursor 宣布从 Thrive、Andreessen Horowitz 和 Benchmark 筹集了 **1.05 亿美元**，突显了其在开发者工具领域日益增长的影响力。[这条推文](https://x.com/cursor_ai/status/1880003590493991072) 确认了这笔资金，引发了对未来更新的巨大乐观情绪。
   - 社区将此支持视为对代码生成功能的强心针，并有早期迹象表明将扩大模型支持。他们期待在未来的版本中看到更快的修复和更强大的功能。
- **Claude 减慢了代码流**：开发者在使用 **Cursor IDE** 的 Claude 集成时遇到了长达 10 分钟的等待时间，破坏了实时可用性。一些人考虑使用本地解决方案或替代集成来避免延迟。
   - 讨论集中在如何减少开销以及 [Anthropic's status](https://status.anthropic.com/) 是否是一个因素。其他人则争论通过本地缓存抵消开销是否能对工作流有所帮助。
- **O1 模型在复杂任务中表现出色**：**O1** 模型提升了编码工作流并简化了高级问题解决，引发了对使用个人 API key 的兴趣。多位测试者报告称，在处理大型代码库时误解更少。
   - 社区成员询问了那些偏好通过 Cursor 直接访问 O1 的用户的成本结构。他们主张透明的集成途径，并指出与基于 Agent 的任务可能存在的协同效应。
- **UI 小故障引发权宜之计**：重叠的代码建议和粘贴问题阻碍了一些用户的可用性，**Ctrl+Shift+V** 作为一个部分修复方案。他们抱怨在聊天（chat）和编辑器（composer）模式之间切换的不便。
   - 几个人建议在生成补全时添加警报系统以减少困惑。其他人建议为代码建议设立专门的面板，以防止文本遮挡。
- **Agent 模式 vs 普通模式增强终端访问**：一篇 [论坛帖子](https://forum.cursor.com/t/what-is-the-difference-between-agent-and-normal-modes/31981) 强调了模式之间的差异，Agent 模式支持终端命令。一些人质疑潜在的安全影响，但称赞了扩展的控制能力。
   - 反馈表明，该功能为更动态的编码会话奠定了基础。尽管存在一些疑虑，用户仍对增加的灵活性表示欢迎，并指出基于 Agent 的流程可用于高级自动化。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 2.5 快速思考策略**：新的 [Qwen 2.5 模型](https://huggingface.co/Ba2han/Qwen-2.5-7B-Woonderer-0.1) 采用两阶段过程——先思考，后生成——在产出答案前优化上下文。
   - 它有时会产生非预期或过长的输出，引发了通过进一步调优来控制失控回复的呼声。
- **Llama-3.2 表现出色**：[Codelion 的 Llama-3.2](https://huggingface.co/codelion/Llama-3.2-3B-o1) 拥有 3.21B 参数，使用 Unsloth 进行微调，实现了更快的训练速度和显著的性能提升。
   - 它在一个月内获得了 139 次下载，但一些用户期待扩展到更大的模型（如 70B）以获得更细腻的结果。
- **LoRa 速度竞赛引发讨论**：用户对比了使用 Unsloth 和 Hugging Face 训练的 LoRa 适配器，强调 Unsloth 的训练速度快了 **2倍**，但推理速度相似。
   - 他们分享了减少依赖冲突和缩短训练周期的经验，激发了对性能优化的好奇。
- **Prompt 追踪器投入使用**：社区请求开发用于在多个开源 **LLM** 之间*追踪和比较 Prompt* 的包或工具，加强了对一致性测试的推动。
   - 他们希望有简化的框架来帮助维持模型输出的一致性，同时衡量不同任务下的性能。
- **知识蒸馏 (KD) 全量微调与 LORA 结合**：简短的交流探讨了*知识蒸馏 (KD)* 是否可以像 **LORA** 方法一样结合选择性权重。
   - 成员们权衡了方法设计中潜在的重叠，激发了对提升模型性能新技巧的兴趣。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Sage 的 MCP 市场亮相**：Sage 最近赢得了 MCP Run 黑客松，展示了一个新的 **MCP Marketplace**，允许在 iPad、iPhone 和 Mac 上一键安装 MCP servlet。
   - 他们将其定位为一种无摩擦的部署方式，促使成员们称其为跨平台可访问性方面的一个充满希望的飞跃。
- **MCP-Bridge 让初学者困惑**：一位用户尝试将 **MCP-Bridge** 与 **AnythingLLM** 配合使用但遇到了困难，并从 [MCP-Bridge 文档](https://github.com/SecretiveShell/MCP-Bridge/blob/master/docs/usecases.md) 中寻求示例和最佳实践。
   - 其他人建议加入 **MCP-Bridge Discord** 以获取更深层的支持，并分享说它扩展了标准的 OpenAI 端点以编排多个 servlet。
- **集成与测试 MCP SDK 受到关注**：成员们寻求针对实际 MCP 服务器的官方 **Python SDK** 单元测试，参考了 [子进程测试方法](https://github.com/modelcontextprotocol/python-sdk/blob/main/tests/server/test_session.py)。
   - 他们辩论了带有外部依赖的集成测试的可靠性，但一致认为强大的覆盖范围能确保 **MCP** 工作流中更少的回归。
- **用户模拟技巧引起开发者兴趣**：一位成员透露了一种巧妙的模拟 Discord 交互的方法，强调了一个能近乎完美模仿用户消息的专用系统提示词。
   - 在他们解释了这些模拟尝试中带有讽刺意味的人为性质后，他们就脚本化用户输入得出了“证明了我的观点”的结论。
- **frgmt0 的 Alpha 代码发布**：开发者公开了处于 Alpha 阶段的 [新 GitHub 项目](https://github.com/frgmt0/blnk.git)，邀请同行对架构和性能提供反馈。
   - 他们欢迎 Bug 报告和建议以塑造代码库，寻求通过协作过程最终达到生产就绪状态。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **SWE-bench & WeirdML 的惊艳表现**：SWE-bench Multimodal 代码横空出世，专注于 JavaScript 的小故障（如地图渲染和按钮文本），详见[此更新](https://x.com/jyangballin/status/1879990781030854897)。
   - 与此同时，[WeirdML](https://x.com/htihle/status/1879872398666965236) 发布了一个全新的 PyTorch 另类任务基准测试，引发了关于 LLM 灵活性不断增强的讨论。
- **OpenAI 神秘的预告遭到批评**：社区成员对 [OpenAI 模糊的公告](https://x.com/polynoamial/status/1880333390525919722)表示不满，敦促其在时间表和功能方面提高透明度。
   - 他们强调，直接且具体的更新对于建立对 AI 进展的信任至关重要。
- **Deepseek R1 传闻与竞争**：有关 [Deepseek R1](https://x.com/StringChaos/status/1880317308515897761) 在代码推理方面可能与 o1-Medium 旗鼓相当的传闻四起，引发了对这一新竞争对手的热议。
   - 观察人士预计，如果传闻中的发布能达到这些性能声明，排行榜将会重新洗牌。
- **NeurIPS PC 风波与透明度之争**：根据 [Andreas Kirsch 的批评](https://x.com/BlackHC/status/1880211847422308618)，批评者称 NeurIPS 委员会是一场**“小丑表演”**，因为他们优先考虑热度而非严格审查。
   - 抗议者认为，沟通不畅和监督不力损害了研究标准，这反映了公众对 AI 领域保密行为的广泛抗议。
- **Devin AI 为自主编程融资 2100 万美元**：Devin 在 2024 年 3 月获得了 **2100 万美元** 的 A 轮融资，由 Founders Fund 和其他主要投资者支持，声称其可以在极少人工干预的情况下处理编程任务。
   - [Answer.AI](https://www.answer.ai/posts/2025-01-08-devin.html) 报道的早期演示显示，Devin 处理 PyTorch 问题的成功率为 13.86%，引发了关于未来“AI 自由职业者”可能性的讨论。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Wave 2 势头强劲**：**Windsurf Wave 2** 的正式发布带来了重大升级，如性能提升和 Dev Container 修复，详见 [Codeium 博客](https://codeium.com/blog/windsurf-wave-2)。
   - 从系统可靠性到用户工作流都得到了改进，实时更新发布在 [Codeium 状态页面](https://status.codeium.com)。
- **Cascade 能够联网搜索并生成记忆**：在新版本中，**Cascade** 现在可以自动或通过 URL 输入进行**网页搜索**，并由保持连续上下文的**自动生成记忆**提供支持。
   - 用户称赞了这种实时引用链接的简化方法，称其为极大的体验提升。
- **学生面临折扣和退款纠纷**：一些 .edu 邮箱持有者被意外收取了 10 美元而非 6.90 美元的费用，而一名沮丧的用户要求 **297 美元退款**，但几乎未得到解决。
   - Codeium 承认了折扣方面的困惑，并承诺将业务扩展到美国以外，但较旧的 .edu 域名仍会引发问题。
- **工具集成想法引发关注**：社区成员建议接入外部爬虫（如 **crawl AI**）和用户提供的 API，以扩展 **Windsurf** 的功能。
   - 他们还提议将这些命令加入系统提示词（system prompts）中，希望能有更灵活的使用场景。
- **Bug、登录和 IDE 反馈**：报告强调了 **autocomplete（自动补全）失效**、死循环以及 Linux 上的登录障碍，并建议提交日志以便快速修复。
   - 其他人提到了 [Open VSX Registry](https://open-vsx.org/extension/Codeium/codeium) 等参考资料，并呼吁建立官方支持工单系统。

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **活动页面混乱：是 Bug 还是特性？**：用户对 [OpenRouter](https://openrouter.ai/docs/provider-routing) 的 **activity page** 表示困惑，抱怨不同 Key 的使用情况图表看起来完全相同，引发了对 **bug** 的担忧。
   - 他们坚持要求提供更精确的单 Key 使用指标，这引发了关于设计可能误导数据的猜测。
- **Gemini 2.0 Flash 干扰 Endpoint**：**Gemini 2.0 flash** 模型引入了新的 Endpoint，导致 [OpenRouter integrations](https://openrouter.ai/docs/integrations) 中出现请求错误。
   - 成员们证实 **website documentation** 需要更新以匹配这些变化，这些变化曾短暂导致现有配置失效。
- **香港请求受阻**：多名用户报告 **OpenRouter** 在香港的请求失败，但通过新加坡路由时正常，暗示了新的中继需求。
   - 他们回想起 **OpenAI** 和 **Anthropic** 历史上曾限制某些地区，这可能解释了间歇性的封锁。
- **DeepSeek V3 引发褒贬不一的评价**：社区讨论集中在来自 [DeepSeek team](https://openrouter.ai/deepseek/deepseek-chat) 的 **DeepSeek V3**，强调其在不同任务和使用场景下的性能表现不一。
   - 一些人建议通过调整配置来改善输出，引发了关于在复杂场景下保持一致可靠性的辩论。
- **BYOK 设置需要更清晰的信号**：用户称赞了 **Bring Your Own Key** 功能，但请求在 Key 集成到 [OpenRouter](https://openrouter.ai/docs/integrations) 时提供更明确的确认。
   - 他们还建议在请求中添加额外的 Metadata，以确认正确的 Key 是否处于激活状态，从而减少高级用例中的猜测。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek 3 在 Context 和 Quantization 方面遇到麻烦**：一名用户在使用来自 [OpenRouter](https://openrouter.ai/docs/provider-routing) 的 **DeepSeek3 模型** 配合 **16k context** 时反复报错，建议通过忽略该 Provider 来解决。
   - 其他人讨论了 Q4 或 Q5 Quantization 之间的性能差异，对过度降低 **DeepSeek3** 的精度表示怀疑。
- **Aider 庆祝获得 25k GitHub stars**：**Aider** 社区庆祝在 GitHub 上突破 **25k stars**，标志着这款 AI 编程助手的一个重要里程碑。
   - 成员们赞扬了它的成功，并认可其作为协作编程中出色工具的地位。
- **CodeGate 保护本地开发隐私**：开发者展示了用于保护 AI 辅助代码中私有数据的 **CodeGate**，并引用了 [CodeGate's repo](https://github.com/stacklok/codegate) 以及 [YouTube demos](https://www.youtube.com/watch?v=WimBevc_Ji0) 和 (https://www.youtube.com/watch?v=lH0o7korRPg)。
   - 他们强调了 **CodeGate** 的加密层可以防止意外泄露，增强了对 AI 驱动编程的信任。
- **Agentic 工具助力代码探索**：参与者探讨了 **Aide.dev**、**Cursor** 以及自定义 CLI 方案用于探索代码库，参考了 [Cursor's forum thread](https://forum.cursor.com/t/cursor-not-able-to-access-full-code-base/36021/11)。
   - 他们将改进的 RAG 策略与处理高 Context 任务的策略相结合，强调通过本地 Prompt 管理来提高结果。
- **Helicone 监控 LLM 使用情况和成本**：[Helicone repository](https://github.com/Helicone/helicone) 展示了一个 **开源 LLM 可观测性** 套件，通过 Docker 或云端提供成本分析、安全层和速率限制。
   - 一些人注意到它与 [Activepieces](https://www.activepieces.com/) 的协同作用，可以实现强大的多 LLM 使用指标监控，展示了多样化的集成方法。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous 获得 4 亿美元巨额融资**：成员们确认 **Nous Research** 获得了高达 **4 亿美元** 的融资，引发了关于其潜在增长以及如何挑战其他 AI 实验室的讨论。
   - 有人提到在 [OpenRouter](https://openrouter.ai/nousresearch/hermes-3-llama-3.1-405b/providers) 上托管他们的模型，而其他人则注意到对高级 GPU 服务的广泛兴趣。
- **OpenAI 独特的薪酬路径**：讨论集中在 OpenAI 的 **利润参与单位** (PPUs) 上，参考了不同于标准股票期权的复杂股权方案，详见[此概述](https://www.levels.fyi/blog/openai-compensation.html)。
   - 几位成员引用了随后的要约收购（tender offers），允许员工套现，突显了这些股份结构如何影响现实世界的支出。
- **GPT-2 RAG 机器人失效**：一位用户抱怨 **GPT-2** 无法处理基于 PDF 的检索，经常返回平淡或重复的响应。
   - 贡献者建议切换到更新的小型模型，如 **smollm** 和 **Qwen**，并评论说在处理大型源文档时，结构化输出（structured output）仍然很棘手。
- **Titans 与内存改造**：开发者们称赞了 [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663v1)，因其在不牺牲并行训练速度的情况下引用历史上下文的方法。
   - **lucidrains** 开发的 [PyTorch 版本](https://github.com/lucidrains/titans-pytorch) 因其降低 Transformer 模型内存开销的潜力而受到关注。
- **LLM 入门书籍走红**：一本关于大语言模型的新书（见[此处](https://arxiv.org/abs/2501.09223)）涵盖了四个主要支柱——预训练（pre-training）、生成式架构（generative architectures）、提示方法（prompting approaches）和对齐方法（alignment methods）。
   - 该书针对希望在现代语言模型开发基础方面获得深入了解的学生和从业者。



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **虚拟旅游 Agent 机器人起飞**：一位用户成功举办了一个关于赞比亚旅游的**虚拟旅游 Agent**研讨会，并指向了[这份官方 NotebookLM 大纲](https://notebooklm.google.com/notebook/51fb6a47-1703-4c03-ac83-12ef3b1b0caf/audio)。
   - 与会者注意到该机器人有效地推荐了住宿和旅游路线，尽管一些人认为 **NotebookLM** 可以在提高结果速度方面进行增强。
- **AI Studio 胜过 NotebookLM**：一位参与者认为 **AI Studio** 比 **NotebookLM** 更可靠，称赞其在各种任务中具有更高的准确性。
   - 他们对 **NotebookLM** 形成深度连接的能力表示怀疑，主张在复杂场景中使用 **AI Studio**。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonar 在实验室中亮相**：工程师在实验室中发现了 **Sonar** 和 **Sonar-Pro** 模型，引发了关于 **Perplexity** API 即将发生变化的猜测。官方 [model cards](https://docs.perplexity.ai/guides/model-cards) 概述了在文本生成和自定义停止参数（custom stop parameters）方面的潜在增强。
   - 用户询问这些进展是否预示着未来会有更多模型变体，并引用了 **CrewAI** 关于多个模型试验中持续出现自定义停止错误的报告。
- **OpenAI 的经济蓝图**：一个共享链接揭示了 **OpenAI** 的经济蓝图，描述了可持续收入和行业定位的新策略。观察者强调了成本管理方法，这可能会引发整个行业的广泛更新。
   - 成员们对这一路线图的连锁反应表示关注，有人称其为减少对既有平台依赖的大胆举措。
- **Starship 7 意外失利**：几位用户讨论了 **Starship 7** 失去飞行稳定性的问题，引用了[此处](https://www.perplexity.ai/search/starship-7-lost-in-flight-2oHRnlZlR5mGDqkus5TtHA)的早期分析。调查人员正在探索结构或推进系统故障作为主要原因。
   - 社区成员考虑了大气因素和发射时机，说明了多变的飞行条件如何影响大规模航天项目。
- **中国的轨道太阳能雄心**：一段发布的视频展示了**中国**建造巨型轨道太阳能电池阵列的计划，可在该 [YouTube 概览](https://www.youtube.com/embed/necQU3gNx2g)中查看。观察者期待新的能源试验，这可能会扩大全球电力能力。
   - 爱好者将这种方法与标准的卫星网格进行了对比，认为国家级项目可以更快地推进空间能源解决方案。
- **Apple 首款美国制造的 iPhone 芯片**：**Apple** 确认打算首次在美国生产 iPhone 芯片，标志着国内制造努力的转变。观察者指出，此举可以重塑供应链并促使成本重新评估。
   - 社区成员将其视为 **Apple** 的战略转型，受全球制造趋势和公司长期硬件计划的影响。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Lynch 的小屋引发笑声**：成员们开玩笑说 **David Lynch** 带着“黑色幽默”出现在小屋（Lodge）中，引用了他艺术中不可预测的道德维度。
   - 这些古怪的言论展示了社区幽默的一面，一条评论称其为受 Lynch 风格启发的“恐惧与着迷的结合”。
- **Stable Diffusion 获得商业动力**：多次讨论涉及 **Stable Diffusion** 的**商业用途**场景，强调了需要放大（upscaling）的按需打印图像。
   - 参与者辩论了**许可细节**，但确认用户输出通常是被允许的，除非受到模型本身的限制。
- **ControlNet 困惑难倒创作者**：用户在将 **ControlNet** 与参考图像集成时遇到困难，发现对于 image-to-image 任务，提示词（prompt）仍然是必不可少的。
   - 建议包括采用 lineart 或其他替代方法，强调了提取数据以获得更一致输出的各种方式。
- **个人照片训练 LoRA 的教训**：一位用户在用孩子的照片训练 **LoRA** 模型时遇到问题，询问如何最好地裁剪图像以及处理分辨率限制。
   - 成员们建议仔细准备数据集，并可能进行架构调整以改进训练结果。
- **切换 WebUI 引发卡通式混乱**：一位用户从 **SD Forge** 迁移到 **Automatic1111**，并处理了由于 Hugging Face 模型不匹配导致的滑稽输出。
   - 他们提到了[这个 GitHub 仓库](https://github.com/lllyasviel/stable-diffusion-webui-forge)用于管理 **styles.csv** 中的提示词，强调了保持设置一致如何防止意外结果。



---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic 采用 Apache 2.0 协议开源**：Nomic Embed Vision 现已采用 [Apache 2.0 License](https://x.com/nomic_ai/status/1880313093097693212)，据报道在多个基准测试中超越了 **OpenAI CLIP** 和 **text-embedding-3-small**。
   - 他们还发布了**开源权重和代码**，为开发者提供了灵活的图像、文本和多模态集成方案。
- **有限 VRAM 上的模型竞赛**：成员们对比了 **LocalLlama** 和 **DavidAU** 的版本，以求在 8GB 配置上获得更好性能，并探索了 [quantization](https://huggingface.co/docs/transformers/main/main_classes/quantization) 技巧。
   - 他们注意到不同设备上的结果各异，从更流畅的吞吐量到随机的卡顿不等，引发了对进一步加速方案的兴趣。
- **自定义 URL Scheme 优化工作流**：一位用户测试了使用自定义的 **hyperscope://** 协议链接到 Emacs 以实现直接文件访问，并讨论了嵌入 .md 或 .html 文件。
   - 其他成员也加入讨论，强调自动启动程序可以简化专业知识检索并减少开销。
- **Qwen2.5-1.5B 的模板困扰**：在使用 ChatML 风格模板时，解析错误困扰着某些 **Qwen2.5-1.5B** 的提示词，迫使开发者对 [LocalDocs 说明](https://github.com/nomic-ai/gpt4all/issues/3362#issuecomment-2595330752)进行了调整。
   - 一位用户在改用 **Quadro NVS300** 等旧款 GPU 时感到非常沮丧，因为极小的 VRAM 对运行高级模型限制太大。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **LeetGPU 提供免费 CUDA 游乐场**：全新的 [LeetGPU](https://leetgpu.com) 提供了一个免费、无需注册的网页端 **CUDA** 实验环境，并推荐配合 《**CUDA by Example**》一书快速上手。
   - 社区成员指出，虽然这本书较老，但它深入浅出地涵盖了 **GPU** 基础知识，并辅以[官方文档](https://developer.nvidia.com/cuda-example)中的参考资料。
- **带有 Warp Specialization 的 Triton 策略**：开发者通过调整缓冲区大小提升了 **stage1_v2** 的性能，实现了更快的 **DRAM** 访问，并展示了 [Automatic Warp Specialization Optimization](https://github.com/triton-lang/triton/pull/5622)。
   - 他们讨论了基于数据流的 kernel fusion 的 **barriers**（屏障），并庆祝 warp specialization 合并到了 **Triton** 主仓库。
- **Torch Double Backward 的曲折**：一位用户在 Torch profiler 中遇到了 **libkineto** 的**内存损坏（memory corruption）** bug，而另一位用户则在探索用于 addbmm 和带有 double backward 的 **Softplus** 激活函数的**自定义 autograd.Function**。
   - 他们注意到 **torch.compile()** 目前缺乏 double backward 支持，这引发了关于管理中间 **tensors** 和减少冗余反向传播的讨论。
- **Arm64 Runner 与 Copilot 的错误解释功能**：团队宣布在公共仓库中免费提供 **Linux arm64 托管 runner**，正如 [GitHub changelog](https://github.blog/changelog/2025-01-16-linux-arm64-hosted-runners-now-available-for-free-in-public-repositories-public-preview) 所述。
   - 他们还引入了 **Copilot** 的“解释错误（Explain Error）”功能，为 **Actions** 任务失败提供即时见解，从而简化实时调试。
- **Thunderkittens 针对 Ampere GPU**：成员们强调了开发中 **tensor cores** 的重要性，建议使用基于 Ampere 架构的显卡（如 **A100**、**H100** 或 **4090**）以获得最大效能。
   - 他们为没有专用硬件的用户推荐了 [LeetGPU](http://leetgpu.com)，并提到了一个基于 **Apple** 的移植版本以实现 M 系列芯片的兼容性。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **棘手的 Flash Attention 惨败**：在 **Tinygrad** 中嵌入 **Flash Attention** 的尝试耗时八小时，尽管尝试将嵌套循环映射到张量维度，最终仍遭遇 **GPU OOM** 和内存问题。一个小小的胜利是 **stable diffusion** 的一个部分步骤在 **25GB** 的 GPU RAM 上成功运行，带来了一线希望。
   - 参与者对 **Flash Attention** 所需的 **explicit loops**（显式循环）表示沮丧，质疑 **Tinygrad** 是否能在不重新考虑其内存控制的情况下进行有效适配。
- **算子（反）融合的自由**：一份关于 **operator (un)fusion**（算子反/融合）的 [GitHub 教程](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20250117_fusion.md)分享了在 **Tinygrad** 中合并算子以减少开销的见解。该资源重点介绍了维度处理的复杂性，并概述了优化调度的方法。
   - 成员们讨论了在平衡性能与内存限制时采用单算子内核（single-kernel）方法的权衡，坚持认为**合理的切片（chunking）**可以避免运行时减速。
- **抖动的 JIT 调整**：贡献者们探索了在保持 **JIT** 吞吐量的同时处理**可变 batch sizes** 的方法，建议使用 `.realize()` 调用来控制计算图。一些人考虑使用 **padding** 技术来保持输入的一致性。
   - 他们辩论了将用于训练和测试的 **JIT** 机制分开的可能性，强调切换优化可能会带来性能不一致的风险。
- **Tinygrad 中的 FP8 尝试**：在增加功能标志的呼声下，**FP8** 支持应运而生，确保对现有测试的影响降至最低。开发者计划隔离脆弱的代码路径，并逐步集成这一新的精度选项。
   - 他们的目标是在进行高级数值实验的同时保持向后兼容性，强调采用谨慎的逐行处理方法以避免破坏现有功能。
- **Windows 的苦恼与转机**：在有参考资料暗示将停止支持后，社区成员对 **Windows support** 提出了疑问，但开发者表示除了 **mmap** 常量外，大部分功能仍然可用。他们分享了一些修复程序使测试得以运行，表明该平台并未被完全放弃。
   - 爱好者们利用这些见解来维持 Windows 的可行性，同时也意识到特定平台的特性仍需要针对性的补丁。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **FORTRAN 重燃，CUDA 遭批，Triton 崭露头角**：令人意外的是，**FORTRAN** 引发了关于在现代 HPC 背景下维护旧语言的讨论。
   - 成员们对 **CUDA** 的复杂性表示不满，并称赞 **Triton** 的 Python 基础，尽管有人指出“ChatGPT 并不擅长它”。
- **复杂损失函数与 V JEPA 争议**：参与者探索了用于高级 AI 指标的**复杂损失函数**，分享了遇到的最苛刻的设计。
   - 他们还重新审视了 **V JEPA 论文**，讨论了其注意力层和 **softmax** 可能如何影响下游任务中的 embeddings。
- **MiniMax-01 论文与 3090 训练捷报**：与会者剖析了 [MiniMax-01 论文](https://arxiv.org/abs/2501.08313)，该论文统一了 **MHA** 和 **GQA** 以处理更长的上下文。
   - 一位用户在 **3090 TI** 上训练了一个 1 亿参数的 flow matching 模型，称赞其数学原理易于理解且代码发布简洁。
- **主动推理与非语言暗示**：一段由 **Karl Friston** 出镜的 [YouTube 视频](https://www.youtube.com/watch?v=N5H5I6cvcrQ)激起了关于 **active inference**（主动推理）的讨论，涵盖了**自由能（free energy）**和**时间**维度。
   - 成员们强调了**非语言交流**可能占总交互的 **60%**，并重点讨论了面部表情和手势。
- **显存改装与 CaPa 的 4K 网格方法**：爱好者们讨论了 **3090 memory mods**（显存改装），思考 GPU 升级的前景。
   - 他们还关注了用于快速生成 4K 网格输出的 [**CaPa** 方法](https://ncsoft.github.io/CaPa/)，并引发了与 **Trellis** 的对比。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **TITANS 与双模型记忆的博弈**：Google Research 推出了一款名为 [“TITANS” 的新模型](https://www.youtube.com/watch?v=x8jFFhCLDJY)，该模型使用两个较小的动态子模型来模拟类记忆功能，旨在增强长序列处理能力。
   - 成员们指出，该模型仍然缺乏持续学习能力，这表明它尚不是自适应召回（adaptive recall）的完整解决方案。
- **RunwayML 的“内衣抽屉”困境**：一个关于**内衣抽屉（underwear drawer）**的奇特引用触发了 RunwayML 的内容审核，引发了对过滤器过度敏感的质疑。
   - 其他人注意到这些规则具有讽刺性的细节，因为看似无害的短语也可能让工具进入意外的警报模式。
- **主 AI Agent 瞄准 LLM 日志**：一位用户提议构建一个**主 AI Agent**，用于检查来自多个 LLM 的大型对话存档，并生成针对性的子 Agent。
   - 他们征求了相关经验分享，并提到了整合来自不同语言模型的海量数据流所面临的挑战。
- **Mind Journal 故障与日期缺陷**：重新勾选 GPT Editor 中的 **DALL·E** 选项框解决了 Mind Journal 的问题，此前该问题导致了对正常功能的困惑。
   - 用户还报告了版本历史记录中出现 **INVALID DATE** 占位符的问题，这使得可靠的变更跟踪变得复杂。
- **Prompt Engineering 计划与 Jailbreak 担忧**：一位成员计划在 **30 天内**编写一本关于 *Prompt Engineering* 的书，并参考了官方 [OpenAI 文档](https://chatgpt.com/share/67897fc3-6580-8000-af35-d693f933acfb) 进行结构化学习。
   - 与此同时，社区对显式的 *Jailbreak* 讨论表示警惕，强调了严格的审核标准以及触碰边缘话题的风险。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Molmo 视觉模型在 trust_remote_code 上受阻**：在使用 **Molmo** 视觉模型时遇到错误，迫使用户启用 `trust_remote_code=True`，但 **LM Studio** 不允许这种操作方式。
   - 一位成员确认，需要此设置的 **MLX 模型**将无法在 **LM Studio** 上运行，导致视觉支持方面存在空白。
- **Llama 3.2 Vision 运行受限**：用户在运行 **Llama 3.2** 视觉模型时遇到了未知的架构错误，确认其仅能在 **Mac MLX** 构建版本上运行。
   - Windows/Linux 版 **LM Studio** 的不兼容性引发了困惑，因为该模型目前仍锁定在 Mac 上使用。
- **Mac 在 Phi-4 的低 Token 速率下表现挣扎**：拥有 16GB RAM 的 Mac 用户发现，在 **LM Studio** 中使用 **Phi-4** 生成文本时，速率低至 **0.05 tokens/sec**。
   - 他们注意到起步非常缓慢，但在生成几个 Token 后速度有所提升，这表明资源限制阻碍了初始性能。
- **MiniMax-01 表现平平**：与 **WizardLM-2** 的对比显示，**MiniMax-01** 的结果并不理想，尤其是在格式化和**中文输出**任务中。
   - 一位用户认为它是一个平庸的选择，称其相对于成熟的竞争模型改进微乎其微。
- **视觉模型卡在第一张图片上**：一位用户注意到，除非重置对话，否则视觉模型中的新图片仍会引用第一张图片。
   - 他们建议清除或重新加载会话，并评论说这是多个视觉模型中反复出现的故障。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **吸引人的自我介绍与学生 AI 项目**：一位用户敦促新成员进行更充实的**介绍**（introductions），鼓励他们分享简单的问候以外的内容，以促进活跃的交流。另一位用户讨论了一个关于 **Generative AI** 的毕业设计，提到了更深层次社区参与和头脑风暴的潜力。
   - 他们建议，尽早分享目标或问题可以激发**技术协作**，社区随时准备提供针对性的见解和建设性的反馈。
- **重排序聊天历史与相关性提升**：一位成员询问如何在 **rerank** 提示词中按正确的顺序结构化对话日志，并提供足够的上下文。另一位成员强调，**更多细节**能改善语义对齐，特别是在为了实现更好的检索而进行精确索引时。
   - 他们还讨论了捕获旧消息以加强引用的方法，并将 *“模型看到的数据越多，其推荐就越精准”* 作为使用 reranker 的指导原则。
- **Command R 模型成本与 8-2024 版本的困惑**：成员们质疑 **8-2024** 版本的 **command-r** 是否与之前的版本定价相同，对任何成本变化表示不确定。其他人观察到默认的 **command-r** 仍指向旧的时间戳，这为版本命名和潜在新功能的猜测留下了空间。
   - 用户提到了 **8-2024** 部署中的一些异常情况，并建议密切监控性能，因为实际反馈可能会揭示意想不到的怪癖。
- **Cohere 的免费深度学习路径**：Cohere 重点展示了 [LLM University](https://docs.cohere.com/v1/docs/the-cohere-platform) 和 [Cookbooks](https://docs.cohere.com/v1/page/cookbooks)，它们提供了手把手的 “Hello World!” 教程，并在前三个月提供 **$75** 的额度。这些资源让新手能够快速实验用于各种任务的 **Language AI**。
   - 他们还强调了 [AWS Cloud](https://docs.cohere.com/v1/docs/cohere-on-aws) 集成，该集成支持托管环境，在支持高级部署的同时消除了繁重的基础设施需求。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 的神奇迁移**：所有公共 GitHub 仓库已从 [ModularML](https://github.com/modularml) 迁移到 [Modular](https://github.com/modular)，并设置了**自动重定向**（auto redirects），实现了轻松导航。
   - 成员们还提议将 **Mojo** 和 **MAX** 项目添加到 [awesome-for-beginners](https://github.com/MunGell/awesome-for-beginners)，以扩大在初学者中的曝光度。
- **Mojo 的并行困境**：一位用户反馈了在 **Mojo** 中对 Python 代码使用 **parallelize** 时的问题，当 `num_work_items` 和 `num_workers` 同时超过 1 时会失败，而纯 Mojo 代码则运行正常。
   - 他们指出，这专门发生在连接到 **Foo** 类的结构体的 `start` 函数中，表明可能需要进一步的调试。
- **Variant 作为和类型的优势**：工程师们考虑将 Mojo 中的 **Variant** 作为和类型（sum type）支持的替代方案，但由于语言的持续变化，目前仍保持谨慎。
   - 他们还讨论了可能的库重构，建议在标准库稳定之前采用增量方法。
- **MAX 与 .NET：关于可组合性的思考**：成员们推测 **MAX 的最终形态** 可能会像 **.NET** 一样，成为一套可组合的组件，可能使用 **Mojo** 或 **C#** 作为核心语言。
   - 他们的对话强调了可组合性的重要性，并参考了框架之间在跨平台扩展方面的协同作用。
- **JSON 与量子致谢**：一位用户称赞 **yyjson** 在高效处理大型 JSON 数据方面的表现，重点介绍了 [yyjson 文档](https://ibireme.github.io/yyjson/doc/doxygen/html/md_doc__data_structure.html) 中的不可变和可变结构。
   - 他们还感谢社区向其推荐了 [quantum.country](https://quantum.country)，称其为量子概念的绝佳训练场。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **SWEBench 表现随 o1 Agent 激增**：我们的 CTO 宣布，他们**基于 o1 的 AI 编程 Agent** 在 SWEBench 上获得了 **64.6%** 的分数，标志着一个性能里程碑，详见[这条推文](https://x.com/shawnup/status/1880004026957500434)。他们正在准备正式提交以供验证，并重点介绍了在 o1 驱动开发中获得的关键见解。
   - 据称这是已知首个完全由 o1 驱动的 Agent，引发了新的 Benchmark 尝试计划。一些社区成员期待通过扩展测试场景来验证这些令人印象深刻的分数。
- **Anysphere 获得 1.05 亿美元融资以实现代码自动化**：Anysphere 锁定了 **1.05 亿美元** 的 B 轮融资，以推进 AI 驱动的编程，详见 [Cursor 的博客](https://www.cursor.com/blog/series-b)。其支持者包括 Thrive Capital 和 Andreessen Horowitz，重点关注为数百万程序员服务的编辑器。
   - 社区对代码自动化可能的升级和更深层次的 R&D 突破感到兴奋。一些与会者提到了类似的针对法律领域的 AI 融资，但官方数据仍然有限。
- **Agent Recipes 推出**：一个名为 **Agent Recipes** 的网站上线，提供了 Agent 工作流的代码模板，详见[这条推文](https://x.com/nutlope/status/1879587920744788172)。它承诺通过复制粘贴示例，轻松集成到 AI 应用程序中。
   - 早期用户称赞了使用提供的代码片段快速构建基于 Agent 的解决方案的速度。社区将其视为整合 Agent 行为的便捷途径。
- **拜登发布网络安全行政令**：总统乔·拜登颁布了一项重大的网络安全行政令，如[这篇 Wired 文章](https://www.wired.com/story/biden-executive-order-cybersecurity-ai-and-more)所述，旨在加强 AI 安全和身份识别措施。该计划应对外国网络威胁，并为美国机构设定了指南。
   - 一些工程师预计这些规则将重塑政府对 AI 供应商的采购决策。其他人则预见到将这些指令与大规模工作流同步的挑战。
- **对 OpenAI webRTC API 的担忧**：开发者对实现 **OpenAI 的 webRTC 实时 API** 表示沮丧，因为除了内部演示外几乎没有示例。许多人请求提供开源参考或针对实时流媒体设置的知识库。
   - 他们指出了平衡数据吞吐量和开销的复杂性。讨论以推动收集社区驱动的解决方案和文档告终。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Women in AI 呼吁关注 RAG**：组织者邀请女性技术人员参加在帕罗奥图举行的 [Women in AI RAG Hackathon](https://t.co/2Bzg80dh29)，重点展示使用开源向量数据库 **Zilliz** 的 **Retrieval-Augmented Generation**。
   - 与会者将在为期一整天的活动中与同行和导师交流，该活动重点关注强大的 **RAG** 方法。
- **GraphRAG 成为焦点**：最近的一次网络研讨会强调了 **Memgraph** 和 **LlamaIndex** 如何联手创建基于图的 Agent 应用，重点关注 **GraphRAG** 以实现更好的上下文检索 [点击观看](https://t.co/a4SMTY5pC3)。
   - 演讲者强调了 Agent 策略和改进 **RAG pipeline** 的技巧，扩展了开发者整合上下文数据的方式 [更多信息](https://t.co/PaK8dt1m9y)。
- **CAG 概念激发创新**：成员们讨论了使用 Gemini 和 LlamaIndex 的 **Cached Augmented Generation (CAG)**，透露这通常需要直接的模型访问，例如 PyTorch。
   - 他们分享了一个 [CAG 实现](https://github.com/hhhuang/CAG/blob/main/kvcache.py)，展示了一种用于加速生成的强大缓存技术。
- **Azure 集成引发困惑**：一名用户在处理 **Azure AI** 将调用路由到 OpenAI 时遇到困难，指出服务配置不完整。
   - 建议包括设置专用的 **embedding model**，同时呼吁提供更好的示例页面以澄清模型选择。
- **元数据与 Prompt 追踪受到关注**：参与者澄清说，可以通过 `excluded_llm_metadata_keys` 和 `excluded_embed_metadata_keys` 为 chunking 和 embedding 任务切换 **node metadata**。
   - 他们还在寻找一个可以跨开源 **LLM** 追踪和比较 Prompt 的软件包，尽管目前尚未出现具体的解决方案。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy V3 错过第一季度目标**：开发团队确认 **DSPy v3** 由于重大的内部变动，将不会在第一季度发布，具体的发布日期目前仍悬而未决。
   - 他们提到目前正在讨论就绪情况，并暗示在发布这个大版本之前，可能会先推出一些较小的更新。
- **Stable Diffusion 结合 Chain-of-Thought 势头强劲**：一项新尝试旨在通过“Chain-of-Thought”方法优化 **Stable Diffusion** 的提示词，如 [Thorondor LLC 的推文](https://x.com/thorondorllc/status/1880048546382221313)所示。
   - 社区成员对于利用 **DSPy** 进行迭代式提示词构建表现出极大兴趣，重点在于逐步增强文本嵌入（text embeddings）。
- **ReAct 加法工具引发骚动**：一位用户在 **dspy ReAct** 中遇到了错误，其 *addition* 工具无法对两个数字求和，理由是存在未知的必需参数。
   - 他们在 LM-Studio 下运行 **LLama**，怀疑是重定义冲突，并被要求提供完整的错误日志以定位原因。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **ChatML 与 Llama3 的较量**：成员们讨论了 **ChatML** 相对于 **Llama3** 的优势，暗示了一场模型霸权的竞争。
   - 一位参与者给出了随意的回答“*duh*”（意为“显而易见”），强调了他们对 **ChatML** 的信心。
- **ShareGPT 数据集获得认可**：有人询问使用 **ShareGPT** 是否可能存在复杂情况，但参与者确认不存在任何问题。
   - 他们指出已经有一个现成的键映射（key mapping）配置，标志着可以直接使用而无障碍。
- **从 ShareGPT 迁移的工作持续推进**：一段对话强调了从 **ShareGPT** 迁移出来的文档化路径，确保了平稳过渡。
   - 用户提到该参考资料涵盖了每一个步骤，解决了频繁出现的数据集疑虑。
- **Torchtune 的微调需求增长**：一位参与者指出，**Torchtune** 目前需要进行重大的修改。
   - 这一要求暗示任何依赖该工具功能的人都需要进行更深层的代码调整。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **沉默的截图引发困惑**：一位用户分享了一张与 **OpenInterpreter** 相关的截图，但没有提供任何背景或评论，让其他人不确定该如何回应。
   - 没有人跟进或提问，表明大家对该截图内容的兴趣极低或对其意图不明。
- **错失视觉洞察的机会**：成员们没有对分享的图片进行分析，这表明关于 **OpenInterpreter** 潜在功能或问题的对话尚未被挖掘。
   - 该提示未得到回应，显示出该小组在进一步贡献之前，希望看到更多实质内容或细节。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **功能 FOMO 与好奇心探索**：一位用户询问某个 **feature** 是如何被发现的，想知道他们之前是否已经了解该功能，还是最近才探索到的。
   - 这激发了人们对 **engagement**（参与）模式如何揭示未尝试的功能和被忽视的潜力的兴趣。
- **测试纠葛与错失的机会**：另一位用户强调了在尝试极少使用的工具时遇到的 **roadblocks**（障碍），认为缺乏熟悉度阻碍了更广泛的实验。
   - 参与者指出，深入的探索需要一个支持无风险尝试的环境，以及对潜在陷阱的开放对话。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Torchtune Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**LAION Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# PART 2: 详细频道摘要与链接

{% if medium == 'web' %}

### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1329852789794537614)** (1 条消息): 

> `Lucide Icons, Bolt 更新, 错误修复` 


- **Lucide Icons 错误已解决**：Bolt 的 Agent 现在可以访问整个图标库，从而有效地消除了 **Lucide icon not found 错误**。
   - *无需额外的 Token 或调试！* 自最新更新起，此改进已在所有项目中生效。
- **Bolt 更新：确定性图标（Deterministic Icons）**：Stackblitz 的最新更新引入了 **确定性图标**，确保 LLM 在选择图标时不会产生幻觉（hallucination）。
   - 现在，在所有项目中都能 **每次** 准确地选取图标，简化了用户体验。



**提到的链接**：<a href="https://x.com/stackblitz/status/1880291208574235134">来自 StackBlitz (@stackblitz) 的推文</a>：Bolt 🧠 更新：确定性图标。LLM 往往会幻觉图标名称从而导致错误，但 Bolt 的 Agent 现在可以访问整个图标库，并每次都选出完美的图标。（无需额外的...

  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1329605396528234527)** (6 条消息): 

> `React 中的 NPM 使用, 针对特定新增内容的 Prompting, PDF 注释 Web App 指令, 文件结构文档, TraycerAI 扩展评测` 


- **在 React 中使用 NPM 提升功能**：成员们讨论了在 **React** 开发时指示 AI 使用特定的 **NPM 软件包**，从而提升功能。
   - 有人指出，强调在开发过程中不进行删减可以保持流程的完整性。
- **清晰地 Prompt AI 进行新增**：一位用户提到，他们在与 AI 沟通时，通过明确表示只需要 **新增** 而不需要任何 **删减**，取得了成功。
   - 他们强调，提醒 AI 不要“改进”现有元素有助于保持专注。
- **为 PDF 注释构建详细指令**：一位成员建议为一个 PDF 注释 Web App 创建一个 **指令文件夹**，其中包含详细的 Markdown 文件，概述 **应用流程**、后端结构和其他必要的项目需求。
   - 然而，当被要求添加细节时，AI 开始编造实际 Web App 中不存在的信息。
- **文件结构文档以提高清晰度**：为了增强项目指令，提议创建一个文件结构文档，列出所有 **网站文件** 及其用途。
   - 此举旨在提高对项目组件的整体理解和清晰度。
- **对 TraycerAI 扩展的正面反馈**：成员们讨论了 **TraycerAI** 扩展极具潜力的功能，它在 **Cursor AI** 中表现良好，并能跟踪整个代码库。
   - 使用它可以创建特定任务并生成实施计划，显著增强了工作流。



**提到的链接**：<a href="https://x.com/sanketdongre369/status/1880159755337101316">来自 Sanket Dongre (@sanketdongre369) 的推文</a>：刚刚试用了 @TraycerAI 扩展，它在 @cursor_ai 中运行良好！它能跟踪整个代码库。你可以创建特定的任务来实施、改进或修复功能。它会生成一个计划，你可以...

  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1329549962547171430)** (283 条消息🔥🔥): 

> `Bolt 功能面临的挑战、Supabase 用户体验、Bolt 的 Git 集成、域名检查工具开发、高效协作策略` 


- **对 Bolt Bug 和错误的挫败感**：用户对 Bolt 的功能表示了极大的挫败感，指出代码中存在错误删除和占位符等问题，导致了过度的 Token 消耗。
   - 一位用户报告称需要开发人员协助排查故障，而其他用户则分享了利用 Git 的技巧以及构建有效 Prompt 对缓解问题的重要性。
- **将 Supabase 集成到应用程序的经验**：几位用户讨论了将应用程序连接到 Supabase 时遇到的挑战，包括导致请求失败的无效 UUID 错误。
   - 一位用户建议，验证并记录 UUID 输入有助于排查 Supabase 集成中的问题。
- **即将推出的 Bolt Git 集成**：一位用户宣布计划为 Bolt 发布一个 Git 集成工具，旨在简化项目管理和导入流程。
   - 该集成将允许用户更有效地克隆仓库和管理项目，从而最大限度地减少导入过程中的错误。
- **开发中的酷炫域名爬虫工具**：一位用户正在创建一个域名爬虫工具，用于检查每天过期的域名，并计划根据特定标准过滤结果。
   - 讨论强调了开发此类工具的变现潜力，即为感兴趣的用户识别有价值的域名注册。
- **与开发人员协作的策略**：用户分享了邀请开发人员协助处理代码的各种策略，包括共享 Zip 文件或使用 Git。
   - 社区强调了在与外部开发人员合作时提供清晰结构和指令的重要性，以避免产生混淆。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://getdiscount.io/">getdiscount.io</a>: 未找到描述</li><li><a href="https://prnt.sc/CVZgu1OObu9G">Screenshot</a>: 使用 Lightshot 捕获
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1329550718725656588)** (227 messages🔥🔥): 

> `RWKV Model Performance, Box Puzzle AI Challenge, Model Size vs. Inference Speed, QRWKV Project, CoT Tuning` 


- **RWKV 模型展现潜力**：RWKV 模型，特别是 0.4B 版本，在各项测试中表现良好，但在处理如 Box Puzzle 等复杂任务时仍显吃力，表明仍有改进空间。
   - 与 Qwen 等其他模型的对比突显了 RWKV 模型的独特能力，并讨论了是否需要引入 CoT tuning 等额外训练技术。
- **Box Puzzle 作为基准测试**：引入了一个复杂的 Box Puzzle 作为测试 LLM 能力的基准，结果显示许多模型无法提供高效的解决方案。
   - 模型的回复往往会导致不必要的循环或异常行为，强调了在 AI 中实现常识推理的挑战。
- **模型大小与推理兼容性**：讨论探讨了模型大小（例如 72B vs. 32B）与推理速度之间的平衡，指出更大的模型通常会导致更高的运营成本。
   - 对于许多用户而言，模型效率至关重要，因为大型模型的计算需求可能高得令人望而却步。
- **QRWKV 项目进展**：QRWKV 项目旨在将 Transformer 模型转换为 QRWKV 格式，从而受益于线性时间 prefill 和全面的历史上下文。
   - 分享了关于未来迭代的见解，包括 QRWKV7 的开发工作以及解决各种基准测试中的性能指标。
- **CoT Tuning 的考量**：强调了 Chain of Thought (CoT) tuning 对于提高模型解决拼图能力的重要性，将其作为增强性能的潜在途径。
   - 讨论围绕调整模型训练方法以包含专注于推理的方法，从而在 AI 任务中获得更好的结果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ">Neural Networks: Zero to Hero</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/BlinkDL/RWKV-Gradio-1">RWKV-Gradio-1 - BlinkDL 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://substack.recursal.ai/p/q-rwkv-6-32b-instruct-preview">Q-RWKV-6 32B Instruct Preview</a>: 迄今为止最强、最大的 RWKV 模型变体：QRWKV6 32B Instruct Preview</li><li><a href="https://github.com/SmerkyG/RWKV_Explained/tree/main">GitHub - SmerkyG/RWKV_Explained: RWKV, in easy to read code</a>: 易于阅读的代码实现 RWKV。欢迎在 GitHub 上为 SmerkyG/RWKV_Explained 的开发做出贡献。</li><li><a href="https://huggingface.co/mollysama/QRWKV6-32B-Instruct-Preview-GGUF/tree/main">mollysama/QRWKV6-32B-Instruct-Preview-GGUF at main</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1329544383359619213)** (39 条消息🔥): 

> `BERT and GPT Attention Mechanisms, Gradient Sparsity and Compression, NanoGPT Speedrun Achievements, SIRENs and Derivatives, Context Length Warmup Strategies` 


- **BERT 的 CLS Token 方向性**：BERT 中的 **CLS token** 可以关注所有其他 token，将其放在左侧仅仅是为了方便，并不影响性能，因为信息是双向流动的。
   - 相比之下，**GPT** token 仅关注之前的 token，这可能会限制其上下文理解能力，引发了关于性能影响的讨论。
- **梯度压缩与效率**：讨论强调了 **Deep Gradient Compression** (DGC) 方法，该方法通过消除冗余的梯度交换来减少分布式 SGD 中的带宽，从而可能增强可扩展性。
   - 一些人认为梯度稀疏在实践中仍未得到充分利用，主要是因为它侧重于联邦学习（Federated Learning），而后者尚未广泛普及。
- **3 分钟 NanoGPT Speedrun 纪录**：**NanoGPT speedrun** 达成了一项新纪录，在 8xH100 配置下，单次运行成本仅为 **$0.40**，训练完成时间缩短至 **3 分钟** 以内。
   - 这展示了相比之前努力的巨大改进，证明了 **modded-nanogpt** 仓库迭代在训练效率方面取得的进展。
- **探索用于导数学习的 SIRENs**：一位成员指出，**SIRENs** 有助于高效计算高阶导数，适用于需要从坐标到输出的隐式映射的任务。
   - 有人担心输出需要是无旋（curl-free）的，以便进行一致的标量势导数推导。
- **上下文长度预热实现**：一位参与者描述了一种在训练期间预热 **sliding window attention size** 的方法，旨在保持数据顺序的同时增强有效上下文长度。
   - 他们目标通过该策略实现约 **1856** 的最大上下文，强调了其在训练过程中的有效性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1712.01887">Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training</a>: 大规模分布式训练需要大量的通信带宽来进行梯度交换，这限制了多节点训练的可扩展性，并且需要昂贵的高带宽网络...</li><li><a href="https://arxiv.org/abs/2411.04434">Scaling Laws for Pre-training Agents and World Models</a>: 具身智能体（Embodied Agents）的性能已被证明可以通过增加模型参数、数据集大小和计算量来提高。这在从机器人到视频游戏的各个领域都得到了证明，当生成...</li><li><a href="https://arxiv.org/abs/2411.19870">DeMo: Decoupled Momentum Optimization</a>: 训练大型神经网络通常需要通过专门的高速互连在加速器之间共享梯度。借鉴频率分解的信号处理原理...</li><li><a href="https://x.com/leloykun/status/1880301753213809016">leloy! (@leloykun) 的推文</a>: 3 分钟内完成 NanoGPT Speedrun 纪录。我们很自豪地宣布我们刚刚突破了 3 分钟大关！这意味着使用每小时 8 美元的临时 8xH100 节点，训练一个 GPT-2 级别的...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1329550731132276846)** (19 messages🔥): 

> `Low Resolution Data Effects, Model Approximation Phenomena, Precision vs Accuracy in Model Training, Ground Truth Data Challenges, Finite Element Method (FEM) Convergence` 


- **低分辨率数据可能有助于学习**：@hawk1399 指出，使用 **low resolution data** 可能会引导模型更好地逼近 **ground truth**，尽管论文中的数据生成方法尚不明确。
   - @paganpegasus 对此表示支持，提到除非发生 **overfitting**，否则模型很可能会逼近训练数据的 **low-pass** 版本。
- **精度（Precision）与准确度（Accuracy）详解**：成员们讨论了模型训练中 **precision** 和 **accuracy** 的区别，暗示模型的误差可能低于训练数据，但仍无法反映真实的 **ground truth**。
   - @uwu1468548483828484 指出，虽然 **FEM** 提供了向真实解收敛的能力，但了解相关的 **PDE** 有助于确定精确误差。
- **对 Ground Truth 数据的担忧**：@hawk1399 对 **ground truth data** 的存在表示怀疑，认为模型由于仅仅是在逼近模拟数据而非事实真相，其理解力可能会下降。
   - @paganpegasus 表示赞同，强调如果训练数据不是 **real** 而是模拟的，模型可能难以掌握相关概念。
- **去卷积（Deconvolution）辩论**：讨论强调了在 **deconvolution** 上的分歧，@uwu1468548483828484 反驳了 **low-pass** 逼近的说法，称其可能导致 **fake detail**。
   - 尽管存在分歧，@paganpegasus 仍寻求澄清，要求重新表述这些与模型训练相关的概念。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1329541646043447449)** (3 messages): 

> `HF Fair Use Claim, Git Repository Availability` 


- **对 HF 合理使用主张的质疑**：有人对 **HF** 主张合理使用（fair use）的能力表示怀疑，强调他们主要是分发者。
   - 这种观点认为，单纯的分发并不足以获得合理使用保护。
- **Git 仓库和 Tar 文件状态**：据指出，**git repository** 仍然可以访问，同时还提供了 **tar file** 的链接。
   - 这表明相关资源对于感兴趣的人员仍然可用。


  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1329544538544541838)** (236 条消息🔥🔥): 

> `Cursor IDE 性能、Claude 集成、融资确认、用户体验问题、O1 模型反馈` 


- **Cursor IDE 请求缓慢**：用户报告在 **Cursor IDE** 中遇到了延迟，尤其是在使用 **Claude** 进行补全任务时，部分用户等待响应的时间超过了 10 分钟。
   - 几位成员表示，**Claude** 的集成和缓慢的请求功能降低了 IDE 的可用性，引发了关于替代方案的讨论。
- **Cursor 确认 1.05 亿美元融资**：Cursor 宣布从 **Thrive**、**Andreessen Horowitz** 和 **Benchmark** 等投资者处筹集了 **1.05 亿美元**，进一步证实了其在开发者工具市场的影响力。
   - 社区成员表示乐观，认为这笔资金将带来重大的功能更新和改进，特别是随着底层模型的持续开发。
- **Cursor 的用户界面挑战**：用户遇到了 AI 建议重叠遮挡文本以及无法将文本粘贴到聊天框的问题，一些用户找到了如 **Control + Shift + V** 之类的变通方法。
   - 用户对在聊天和 composer 面板之间进行导航表示担忧，并建议提供更无缝的切换体验。
- **关于 O1 模型使用的反馈**：用户讨论了 **O1** 模型的效果，分享了它如何显著改善了他们在编码任务中的工作流，特别是针对复杂问题。
   - 有关于通过个人 API 密钥使用 O1 与通过 Cursor 额外付费使用的咨询，表明用户希望集成选项能更加清晰。
- **对 Cursor 功能的建议**：社区成员提出了对 Cursor 的增强建议，例如更好的文档访问以及在聊天和 composer 模式之间更直观的切换。
   - 建议包括让 IDE 在任务完成时提醒用户，以提升整体用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://onecompiler.com/bootstrap/436b3tzs4">未找到标题</a>：未找到描述</li><li><a href="https://x.com/openaidevs/status/1880306077738365211?s=46">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：我们整理了一个使用 Realtime API 构建和编排 Agent 模式的参考实现。你可以使用这个仓库在不到...的时间内原型化一个使用多 Agent 流的语音应用。</li><li><a href="https://x.com/alexalbert__/status/1879917906294870196?s=46">来自 Alex Albert (@alexalbert__) 的推文</a>：@AnthropicAI 开发者的体验升级：我们调整了 prompt caching，现在你只需要在提示词中指定缓存写入点——我们会自动检查缓存命中...</li><li><a href="https://bsky.app/profile/emollick.bsky.social/post/3lfsnssanxs2e">Ethan Mollick (@emollick.bsky.social)</a>：世界银行对尼日利亚学生使用 GPT-4 作为导师进行的一项新的随机对照试验。六周的课后 AI 辅导 = 2 年的典型学习收益，表现优于 80% 的其他...</li><li><a href="https://x.com/cursor_ai/status/1880003590493991072?s=46&t=kUuVqsG2GMX14zvB592G5w">来自 Cursor (@cursor_ai) 的推文</a>：我们从 Thrive、Andreessen Horowitz、Benchmark 以及现有投资者那里筹集了 1.05 亿美元的 B 轮融资。我们很高兴地报告，Cursor 现在被数百万工程师用作他们的...</li><li><a href="https://forum.cursor.com/t/what-is-the-difference-between-agent-and-normal-modes/31981">“agent” 模式和 “normal” 模式有什么区别？</a>：目前我能注意到的唯一区别是运行终端命令的能力。</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1329559665217900666)** (136 条消息🔥🔥): 

> `模型保存问题、Hugging Face API 链接、Colab GPU 实例、Qwen2 与 Multi-gpu 支持、Kaggle 和 Unsloth 的更新` 


- **模型保存的挑战**：用户报告了由于“显存不足”（out of memory）错误导致模型保存失败的问题，即使是对于非 CPT 的普通训练模型也是如此。
   - 讨论了故障排除方法以及在模型保存期间有效管理资源分配的命令。
- **Hugging Face API 通信错误**：一位用户遇到了 Hugging Face API 指向错误模型链接的问题，导致在尝试加载特定配置时产生困惑。
   - 尽管使用了社区提供的正确模型链接，API 仍尝试访问错误的仓库版本。
- **Colab 作为低成本 GPU 解决方案**：成员们讨论了最低的 GPU 实例价格，有人提到了 Vast.ai 的使用经验以及 Colab 的免费实例。
   - 虽然一些用户表示不喜欢 Colab，但也有人对其基础使用的零成本表示赞赏。
- **Multi-GPU 支持时间表**：针对 Multi-GPU 设置的咨询，目前似乎已初步定于今年年初实现。
   - 讨论暗示了在即将到来的更新中可能会有更好的 Multi-GPU 配置支持。
- **Kaggle 上的 Unsloth 更新**：用户强调了允许在 Kaggle 上通过 pip 安装 Unsloth 的新更新，声称模型训练的设置时间更快。
   - 社区鼓励尝试分享的资源，如 Kaggle notebooks，以进行高效的学习和执行。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/unsloth/llama-32-vision-673b04868f51fde3c5786e72">Llama 3.2 Vision - Unsloth 集合</a>：暂无描述</li><li><a href="https://huggingface.co/burgasdotpro/bgGPT-Phi-4">burgasdotpro/bgGPT-Phi-4 · Hugging Face</a>：暂无描述</li><li><a href="https://x.com/UnslothAI/status/1879942441538609583">Unsloth AI (@UnslothAI) 的推文</a>：你现在可以在 @Kaggle 上免费微调 Phi-4 了！你将学习如何：• 准备数据集 • 通过 Kaggle 的免费 GPU 训练 Phi-4 • 运行、评估并保存你的模型。Unsloth 微调 LLM 的速度快 2 倍...</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit">unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit · Hugging Face</a>：暂无描述</li><li><a href="https://github.com/unslothai/unsloth/pull/1503#issuecomment-2575640930">CoffeeVampir3 对 granite 模型的微小修复 · Pull Request #1503 · unslothai/unsloth</a>：对 llama 类中 4.47.1 transformers 的微小修复，且 granite 模型的配置似乎与之前的版本略有不同，residual mult 现在直接位于层上</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama：改进 BPE 预处理 + LLaMA 3 和 Deepseek 支持，由 ggerganov 提交 · Pull Request #6920 · ggerganov/llama.cpp</a>：延续 @dragnil1 在 #6252 中的工作。此 PR 为 llama.cpp 添加了对 BPE 预分词的支持。总结：到目前为止，对于所有基于 BPE 的模型，llama.cpp 都应用了默认的预处理...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1329914437225807922)** (1 条消息): 

> `LLM 的 Prompt 追踪工具，开源 LLM 对比` 


- **关于跨 LLM 追踪 Prompt 工具的咨询**：一位成员询问了是否有可用的包或工具，可以在多个开源 **LLM** 之间*追踪和比较 Prompt*。
   - 该请求表明社区对于在不同模型之间保持一致的性能指标有着日益增长的兴趣。
- **社区寻求 LLM 对比解决方案**：讨论强调了社区内部对能够促进在各种**开源 LLM** 之间*轻松比较* Prompt 的资源的需求。
   - 成员们正在寻找评估模型在相似任务上表现的有效方法，强调了 Prompt 一致性的重要性。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1329675404327256116)** (42 messages🔥): 

> `Unsloth AI 的 Docker 镜像，Unsloth 与 Hugging Face 的推理速度对比，Molmo 的微调支持，Docker 镜像设置期间的错误消息，LoRa 适配器训练差异` 


- **Unsloth Docker 镜像讨论**：一位用户找到了一个 [Unsloth AI 的 Docker 镜像](https://hub.docker.com/layers/foundationmodels/unsloth/latest/images/sha256-c63319ae5b72c59efb666bea916263c139768f01366996acbc449fd0e4397b12)，但经确认该镜像并非官方提供。
   - 一名成员建议改用带有 CUDA 支持的通用 PyTorch 镜像，并强调了所发现镜像的兼容性问题。
- **推理速度：Unsloth vs Hugging Face**：一位用户报告称，Pixtral-12b LoRa 适配器的推理速度在 Unsloth 和 Hugging Face 上是相同的，并对预期性能提出了疑问。
   - 另一名成员提到，从长远来看，Unsloth 平均可以快 **2倍**，并提供了相关资源以获取更多信息。
- **对 Molmo 微调的支持**：一位用户询问 Unsloth 是否支持 Molmo 的微调（finetuning），初步回复表示目前尚不支持。
   - 然而，随后的一条消息暗示可能很快就会提供支持，并请另一位用户进行测试。
- **Docker 设置期间的错误消息**：一位用户在尝试为 Unsloth 设置带有 PyTorch 2.4.1 的 Docker 镜像时遇到了 pip 依赖冲突错误。
   - 建议的解决方案是使用与 Unsloth 兼容的旧版本基础镜像，以避免此类冲突。
- **LoRa 适配器之间的训练差异**：一位用户对比了使用 Unsloth 和 Hugging Face 训练 LoRa 适配器的训练时间，指出 Unsloth 的训练速度快了 **2倍**。
   - 讨论表明，虽然 Unsloth 提高了训练速度，但推理速度似乎相当，这引发了对性能的进一步探究。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/gemma"> 使用 Unsloth 微调 Gemma</a>：通过 Unsloth 微调 Google 的新 Gemma 模型，速度提升 2.4 倍，显存（VRAM）占用减少 58%！</li><li><a href="https://hub.docker.com/r/pytorch/pytorch/tags">未找到标题</a>：未找到描述</li><li><a href="https://hub.docker.com/layers/foundationmodels/unsloth/latest/images/sha256-c63319ae5b72c59efb666bea916263c139768f01366996acbc449fd0e4397b12">未找到标题</a>：未找到描述</li><li><a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html">PyTorch Release 24.01 - NVIDIA Docs</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1329667159021977631)** (5 messages): 

> `思考模型，Llama 模型微调，AI 模型基准测试` 


- **Qwen 2.5：一种新的思考模型**：[Qwen 2.5 模型](https://huggingface.co/Ba2han/Qwen-2.5-7B-Woonderer-0.1)强调两步处理方法：思考阶段后接答案生成，旨在通过更好的上下文增强输出质量。
   - 然而，它仍面临诸如生成非预期答案以及有时产生过长输出的问题。
- **Llama Thinker 模型详情**：[Codelion 的 Llama-3.2 模型](https://huggingface.co/codelion/Llama-3.2-3B-o1)使用 Unsloth 框架进行了微调，实现了更快的训练速度并旨在提升性能。
   - 该模型拥有 3.21B 参数，目前已公开使用，上个月获得了 139 次下载。
- **小型 Llama 模型基准测试**：Codelion 指出 **3B 模型太小**，无法在 LLM 中有效诱导“思考”，因此计划对 **Llama-3.3-70B** 模型进行微调。
   - 这反映了 AI 模型开发中专注于增加模型规模以在复杂任务中获得更好性能的广泛趋势。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Ba2han/Qwen-2.5-7B-Woonderer-0.1">Ba2han/Qwen-2.5-7B-Woonderer-0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/codelion/Llama-3.2-3B-o1">codelion/Llama-3.2-3B-o1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/codeli">codeli (haohao)</a>：未找到描述</li><li><a href="https://huggingface.co/codelion/Llama-3.2-3B-o1-lora">codelion/Llama-3.2-3B-o1-lora · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/codelion/Sky-T1_data_17k">codelion/Sky-T1_data_17k · Datasets at Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1329781559607431231)** (1 messages): 

> `KD Full Fine-tuning, Selective Weights, LORA` 


- **探索 KD Fine-tuning 方法**：*KD 是学生模型的一种全量 Fine-tuning 方法*，引发了关于其与 Selective Weights 结合适用性的讨论。
   - 一位成员指出这与 **LORA** 存在潜在的相似之处，暗示方法论上可能存在重叠。
- **辩论 Selective Weights 的使用**：讨论集中在 KD 是否能像 LORA 中的方法那样，通过 *Selective Weights* 有效运作。
   - 这些方法对模型性能的影响被视为一个关键关注点。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1329576970203304028)** (100 messages🔥🔥): 

> `MCP terminology confusion, MCP tools and clients, Sage and its marketplace feature, Timeout limitations in MCP, Integration and testing of MCP SDK` 


- **MCP 术语混淆**：几位成员对 MCP 术语表示困惑，有人称多次阅读 MCP Bridge 的 README 后仍未完全理解。
   - 这引发了关于文档清晰度和社区内资源共享的讨论。
- **Sage 的 MCP Marketplace 功能**：Sage 最近赢得了 MCP Run 黑客松，展示了其新的 Marketplace 功能，允许一键安装 MCP servlets。
   - 用户对其在 iPad、iPhone 和 Mac 等设备上的易用性感到兴奋。
- **MCP 中的超时限制**：成员们讨论了 MCP 服务器响应的 60 秒超时问题，有人指出这是 Claude Desktop 的限制，而非协议本身的问题。
   - 探索了使用会话标识符（session identifier）和通知来管理此限制的替代方案。
- **MCP SDK 的集成与测试**：一位用户询问了关于 Python SDK 使用实际服务器进行单元测试的问题，引发了关于 subprocess 测试有效性的讨论。
   - 虽然集成测试可能因依赖关系而不稳定，但成员们考虑了各种测试方法以确保功能的稳健性。
- **MCP 工具与客户端**：多项讨论围绕 Sage 和 Claude 等各种 MCP 客户端展开，一些成员对缺失的功能或 Bug 表示遗憾。
   - 还提到了需要一份支持 MCP 的聊天应用的完整列表，以及有效管理多个工具的挑战。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.npmjs.com/package/json-schema-to-zod">json-schema-to-zod</a>: 将 JSON schema 对象或文件转换为 Zod schemas。最新版本：2.6.0，最后发布于 20 天前。通过运行 `npm i json-schema-to-zod` 在你的项目中使用。</li><li><a href="https://www.pulsemcp.com/clients">39 MCP Clients: AI-powered apps compatible with MCP servers | PulseMCP</a>: 一系列能够作为 Model Context Protocol (MCP) 客户端与日益增多的 MCP 服务器交互的 AI 应用和工具。</li><li><a href="https://docs.mcp.run/tasks/using-tasks">Working with Tasks | 🤖</a>: 任务允许你在一套已安装的 servlets 中注册提示词并触发。</li><li><a href="https://github.com/SecretiveShell/MCP-Bridge/blob/master/docs%2Fusecases.md">MCP-Bridge/docs/usecases.md at master · SecretiveShell/MCP-Bridge</a>: 一个提供兼容 OpenAI 端点以调用 MCP 工具的中间件。</li><li><a href="https://github.com/appcypher/awesome-mcp-servers">GitHub - appcypher/awesome-mcp-servers: Awesome MCP Servers - A curated list of Model Context Protocol servers</a>: Awesome MCP Servers - Model Context Protocol 服务器的精选列表。</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/blob/main/tests/server/test_session.py">python-sdk/tests/server/test_session.py at main · modelcontextprotocol/python-sdk</a>: Model Context Protocol 服务器和客户端的官方 Python SDK。</li><li><a href="https://github.com/modelcontextprotocol/inspector/pull/100">Allow setting the timeout with the &quot;timeout&quot; URL parameter by evalstate · Pull Request #100 · modelcontextprotocol/inspector</a>: 允许通过 &quot;timeout&quot; URL 参数设置以毫秒为单位的请求超时。动机与背景：支持测试响应时间超过 10 秒的工具。</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1329560668818505758)** (25 条消息🔥): 

> `MCP-Bridge 使用方法, Discord Bot 开发, David Shapiro 的 ACE Framework, 关于用户模拟的 Drinks 博客, frgmt0 的 GitHub 项目` 


- **MCP-Bridge 用户寻求指导**：一名用户尝试将 **MCP-Bridge** 与 **AnythingLLM** 结合使用，但遇到了挑战，并请求提供有效使用的示例。
   - 另一名用户建议加入 **MCP-Bridge Discord server** 以获取进一步帮助。
- **关于 Discord Bot 的争议**：成员们就 Discord Bot 的有效性展开了辩论，一些人认为现代 Discord 命令已经涵盖了大部分所需功能。
   - 一位成员指出：*'反正现代 Discord 已经内置了你想要的大部分功能的命令。'*
- **对 David Shapiro 的质疑**：讨论集中在 David Shapiro 的 **ACE framework** 上，一名成员对其缺乏社区项目表示失望。
   - 另一位补充说，Shapiro 就像是一个资历较浅的 **Ray Kurzweil**，并强调了 **Ray** 著作的深度。
- **揭秘用户模拟技巧**：一名用户分享了模拟 Discord 交互的技巧，强调了使用特定 system prompt 的好处。
   - 他们在贬低典型的交互方法后，对用户模拟的讽刺性评论道：*'我的观点得到了证实'*。
- **Frgmt0 发布早期项目**：一名用户在 GitHub 上展示了他们的早期 Alpha 项目，并邀请他人提供反馈。
   - 他们提供了[项目链接](https://github.com/frgmt0/blnk.git)，并鼓励用户报告任何问题。



**提到的链接**：<a href="https://github.com/frgmt0/blnk.git">GitHub - frgmt0/blnk</a>：通过在 GitHub 上创建账户来为 frgmt0/blnk 的开发做出贡献。

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1329550440513405029)** (102 messages🔥🔥): 

> `SWE-bench 发布, WeirdML 基准测试, OpenAI 模糊发布, Deepseek R1 发布传闻, 对 AI 透明度的担忧` 


- **SWE-bench 多模态评估代码发布**：新的 [SWE-bench MM](https://x.com/jyangballin/status/1879990781030854897) 引入了 JavaScript 问题，重点关注地图渲染和按钮可见性等视觉组件。
   - 此次发布预计将增强 AI 社区内的多模态评估。
- **WeirdML 达成新基准**：来自 [WeirdML](https://x.com/htihle/status/1879872398666965236) 基准测试的结果表明，在利用 PyTorch 评估 LLM 应对异常机器学习挑战方面取得了显著进展。
   - 该基准测试强调了 LLM 通过反馈进行的迭代学习能力，引发了关于这些模型角色演变的讨论。
- **OpenAI 的模糊发布（Vagueposting）策略受到质疑**：成员们对 OpenAI 目前的模糊发布趋势表示沮丧，认为这削弱了透明度并助长了对其项目的猜测。
   - 他们强调了 AI 开发中清晰沟通的重要性，特别是在安全性和模型能力方面。
- **Deepseek R1 发布备受期待**：有关即将发布的 [Deepseek R1](https://x.com/StringChaos/status/1880317308515897761) 的传闻正在流传，早期结果显示其性能与 o1-Medium 相当。
   - 此次发布预计将与重大技术进步同步，并可能改变 AI 社区内的动态。
- **对 AI 研究透明度的质疑**：人们对 AI 研究透明度的现状表示担忧，讨论围绕技术报告中省略数据策展（data curation）细节的问题展开。
   - 成员们对 AI 领域从开放讨论和知识共享转向更加保守的做法感到惋惜。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/StringChaos/status/1880317308515897761">Naman Jain (@StringChaos) 的推文</a>：DeepSeek-R1 (Preview) 结果 🔥 我们与 @deepseek_ai 团队合作，在 LiveCodeBench 上评估了 R1 Preview 模型。该模型的表现接近 o1-Medium，提供了 SOTA 级别的推理性能...</li><li><a href="https://x.com/stalkermustang/status/1880246516599910641">Igor Kotenkov (@stalkermustang) 的推文</a>：顺便说一下，o3-mini 的访问权限已对（至少部分）测试人员开放。原以为有两个模型，看来 OpenAI 在元旦假期期间一点也没闲着。</li><li><a href="https://x.com/jyangballin/status/1879990781030854897">John Yang (@jyangballin) 的推文</a>：SWE-bench 多模态评估代码现已发布！SWE-bench MM 是一组带有视觉组件的新 JavaScript 问题（如“地图未正确渲染”、“按钮文本未出现”）。</li><li><a href="https://x.com/sama/status/1880356297985638649">Sam Altman (@sama) 的推文</a>：感谢测试 o3-mini 的外部安全研究人员。我们现在已经敲定了一个版本并开始发布流程；计划在大约两周内发货。此外，我们听到了反馈...</li><li><a href="https://x.com/htihle/status/1879872398666965236">Håvard Ihle (@htihle) 的推文</a>：很高兴分享 WeirdML 的结果——这是一个测试 LLM 通过编写可运行的 PyTorch 代码并从反馈中迭代学习来解决奇怪且异常的机器学习任务能力的基准测试。</li><li><a href="https://x.com/sandersted/status/1879719653632770461,">Ted Sanders (@sandersted) 的推文</a>：@jeremyphoward 哈哈，谁说 OpenAI 已经构建了 ASI？我怀疑你可能误解了（确实有人认为 AGI 可能在未来几年内实现，虽然我不同意他们的观点，但我认为...）</li><li><a href="https://x.com/polynoamial/status/1880333390525919722">Noam Brown (@polynoamial) 的推文</a>：最近社交媒体上有很多模糊的 AI 炒作。有充分的理由对进一步的进展感到乐观，但仍有大量未解决的研究问题。</li><li><a href="https://x.co">出售域名 | 购买域名 | 停放域名</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1329789248601198663)** (4 条消息): 

> `NeurIPS PC 批评, ML 会议中的人格特质, ML 职业转型` 


- **NeurIPS PC 面临严重批评**：一名成员对 NeurIPS 的组织过程表示失望，称其为 **“小丑表演” (clown show)**，并强调了社区中缺乏学术诚信的问题。
   - 他们感叹现状允许炒作驱动且缺乏问责制的论文发表，并指出 *“ML 社区既没有牙齿，也没有学术诚信的胃口。”*
- **会议需要警钟**：另一位成员表示赞同，强调会议在处理严肃问题时需要一个**警钟 (wake-up call)**。
   - 他们建议 *“坚持传递这一信息”*，认为社区值得更严肃的对待。
- **DbrxMosaicAI 的职业转型**：一位成员分享了在 **DbrxMosaicAI** 完成最后一周工作后的感激之情，回顾了在初创公司生态系统中三年的经验。
   - 他们提到自己在领导一支优秀的 LLM 数据研究团队期间，为公司的**成功退出 (successful exit)** 以及发布三个开源模型做出了贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/BlackHC/status/1880211847422308618">Andreas Kirsch 🇺🇦 (@BlackHC) 的推文</a>：我很抱歉因为称 NeurIPS PC 为小丑并指出明显的领域利益冲突而伤害了某些人的感情。我并不是针对任何 PC 个人，而是针对这个组织……</li><li><a href="https://x.com/code_star/status/1880355601546674203">Cody Blakeney (@code_star) 的推文</a>：上周是我在 @DbrxMosaicAI 的最后一周。非常感激能成为这样一个了不起的团队和旅程的一员。在这三年里，我学到了很多关于初创公司生态系统的知识，参与了一次成功的……
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1329619259919302656)** (6 条消息): 

> `Olmo 2 Pod 体验, 互联网业务, 资源分配, H100 与 CS-2 对比` 


- **对 Olmo 2 Pod 的评价褒贬不一**：办公室内对 **Olmo 2 pod** 的体验被评价为“一般”，但总体而言，这个 pod 带来了一段*有趣且充实的时光*。
   - 在此期间发现了一个新的音频工具，为体验增添了令人兴奋的维度。
- **互联网创业的乐趣**：目前流行着**许多有趣的互联网业务**，引发了人们的兴趣和好奇。
   - 这反映了人们对创新在线初创公司和创业尝试日益增长的热情。
- **资源分配的秘密**：有人暗示在讨论结束时留下了一个关于资源分配的“**彩蛋 (Easter egg)**”，暗示了某种潜在的见解。
   - 这表明在团队内部共享重要信息时采用了一种细致入微的方法。
- **LLM 预训练中的 H100s vs CS-2**：有人询问在 LLM 预训练中，多少个 **H100** 相当于一台 Cerebras 的 **CS-2**，特别强调了资源数量。
   - 一名成员表示 **1000 个 H100** 可能足够，但也承认对比的具体细节存在不确定性。



**提到的链接**：<a href="https://auphonic.com/">
      
  Auphonic

    </a>：未找到描述

  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1329674579785810000)** (3 条消息): 

> `推理模型综述论文, 对综述论文的批评` 


- **强化推理模型综述**：一篇题为《[迈向大推理模型：大语言模型强化推理综述](https://arxiv.org/pdf/2501.09686)》的论文被分享以供审阅，引发了对其深度和质量的质疑。
   - 一名成员评论说，综述论文往往包含很多**水分 (fluff)**，对它们的整体价值表示怀疑。
- **对综述的普遍悲观**：一名成员表示，综述论文有时会让人**大失所望**，表达了对这类研究结果的普遍不满。
   - 这种情绪凸显了社区内对综述类研究实用性的共同担忧。


  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1329605388412260475)** (4 条消息): 

> `Debt: The First 5000 Years, Devin AI, Answer AI 的影响` 


- **《债：第一个5000年》（Debt: The First 5000 Years）获得认可**：一位用户称赞 *Debt: The First 5000 Years* 是一本伟大的著作，并对其作者表示赞赏，强调了其价值。
   - 该书也有有声书版本，提供了更广泛的获取途径。
- **Devin AI 凭借 A 轮融资引起轰动**：2024 年 3 月，一家名为 Devin 的新 AI 公司脱颖而出，完成了由 Founders Fund 领投的 **2100 万美元** A 轮融资，并获得了 Collison 兄弟和 Elad Gil 等行业知名人士的支持。
   - Devin 被设计为一个全自动软件工程师，能够*像人类同事一样聊天*，学习新技术，甚至独立完成 Upwork 任务。
- **早期演示展示了 Devin 极具前景的能力**：Devin 的早期演示展示了其自主完成 PyTorch 项目的能力，在 GitHub issues 的解决率达到了 **13.86%**，展现了令人印象深刻的技术实力。
   - 一段视频演示了 Devin 在无需人工干预的情况下完成 Upwork 悬赏任务的能力。
- **对 Answer AI 贡献的赞赏**：一位成员对 Answer AI 表示感谢，赞赏其对 AI 领域的独特贡献。
   - *Answer AI* 因其在 AI 领域的独立运营和创新方法而受到认可。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.answer.ai/posts/2025-01-08-devin.html">Thoughts On A Month With Devin – Answer.AI</a>: 在给 Devin 分配了 20 多个任务后，我们对其的印象。</li><li><a href="https://acrobat.adobe.com/id/urn:aaid:sc:VA6C2:c8d84e7d-19bb-42c7-83bc-d24ca664e02c">Adobe Acrobat</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1329815538079236157)** (1 条消息): 

> `H2O.ai 产品, H2O LLM Datastudio, LLM 数据准备` 


- **寻求关于 H2O.ai 使用经验的见解**：一位成员正在寻求有关 **H2O.ai 产品**的使用经验，特别是 **H2O LLM Datastudio**，以辅助数据准备工作。
   - 感谢社区分享关于该工具的任何经验或见解。
- **H2O LLM Datastudio 概览**：**H2O LLM Datastudio** 是一款无代码应用程序，旨在简化与 Large Language Models (LLMs) 相关的数据策展、准备和增强任务。
   - 详细信息可在 [文档](https://docs.h2o.ai/h2o-llm-data-studio/) 中找到。



**提及的链接**: <a href="https://docs.h2o.ai/h2o-llm-data-studio/">H2O LLM DataStudio | 文档 | H2O LLM DataStudio | 文档</a>: &lt;H2OHome title=&quot;H2O LLM DataStudio&quot; description=&quot;一款无代码应用程序和工具包，旨在简化与 Large Language Models 相关的资料策展、准备和增强任务 (...

  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1329913370354385019)** (1 条消息): 

> `Windsurf Wave 2 发布，Cascade 网页搜索，Cascade 自动生成记忆，性能改进，系统状态` 


- **Windsurf Wave 2 浪潮来袭**：备受期待的 **Windsurf Wave 2** 已发布，带来了重大新功能，包括网页搜索能力和 Cascade 的自动生成记忆。
   - 在 [Codeium 博客](https://codeium.com/blog/windsurf-wave-2) 查看公告的完整详情。
- **Cascade 现在可以搜索网页**：Cascade 引入了**搜索网页**的能力，支持自动搜索、通过 **URL 输入**或使用 `@web` 和 `@docs` 命令。
   - 此功能允许用户粘贴特定上下文的 URL 或根据查询触发搜索，非常易于使用。
- **Cascade 自动生成记忆让对话更智能**：通过此次更新，Cascade 现在可以**自动生成记忆**，以在对话中保持上下文。
   - 这一增强旨在通过确保连续性来丰富用户体验，使交互更流畅、更相关。
- **性能提升与大量修复**：在**修复了多个 Dev Container 问题**的同时，还进行了显著的性能改进，提升了整体效率。
   - 得益于这些健康检查和技术增强，用户可能会体验到更完善的性能表现。
- **Codeium 系统状态更新**：用户现在可以在 https://status.codeium.com 查看 **Windsurf/Codeium 的状态**，目前报告显示一切运行正常。
   - 他们通过在其状态页面上跟踪近期发生的事件、诊断问题和解决方案来保持透明度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://status.codeium.com">Codeium Status</a>: 未找到描述</li><li><a href="https://codeium.com/blog/windsurf-wave-2">Windsurf Wave 2</a>: 介绍 Wave 2，我们对 Windsurf 编辑器的第二批更新。</li><li><a href="https://x.com/windsurf_ai/status/1880354013922857384">Windsurf (@windsurf_ai) 的推文</a>: Wave 2 来了。本次更新包含：🌐网页搜索🧠自动生成记忆💼企业级就绪... 以及更多！</li><li><a href="https://www.codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器和 Codeium 扩展</a>: Windsurf 编辑器的最新更新和变化。
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1329557142159495299)** (28 条消息🔥): 

> `Codeium 客户服务问题，学生折扣计划扩展，IDE 功能与 Bug 报告，Codeium 扩展登录问题，Bug 修复与新功能反馈` 


- **对 Codeium 客户服务的不满**：一位成员对尝试获取 **297 美元退款**的过程表示非常沮丧，称客服没有提供帮助，而是要求提供更多请求细节。
- **学生折扣计划的扩展**：关于学生折扣公告的讨论，一位成员询问该计划是否仅适用于当地学生。
   - 另一位成员确认，他们正在努力**扩展该计划**，以包括美国以外的其他大学。
- **IDE Bug 与反馈**：多位用户报告了在尝试更新 IDE 时遇到的 **autocomplete 失败**和仓库问题，并寻求帮助。
   - 一位用户称赞了修复和新功能，指出每次更新都是一次巨大的进步浪潮。
- **Codeium 扩展的登录问题**：一位用户在 **Linux** 上登录 Codeium 扩展时遇到挑战，报告称自动和手动登录均失败，而 Mac 上则正常。
   - 建议他们提交日志和支持工单以寻求协助。
- **应用开发资源咨询**：一位用户询问是否有用于应用开发的模板或样板项目，提到了像 **Lovable 或 Bolt** 这样的起点。
   - 他们向处理过类似开发任务的其他用户寻求建议或链接。



**提到的链接**: <a href="https://open-vsx.org/extension/Codeium/codeium)">Open VSX Registry</a>: 未找到描述

  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1329553076784205946)** (90 条消息🔥🔥): 

> `Windsurf 账户问题、Windsurf 工具集成、学生定价差异、Windsurf 功能用户体验、Bug 报告及支持查询` 


- **学生面临定价问题**：使用 .edu 邮箱的用户反映被收取了 10 美元的“早期采用者”价格，而非预期的 6.90 美元学生价，特别是当他们的邮箱后缀不完全是“.edu”时。
   - 用户对旧的 .edu 账户无法享受折扣表示担忧，这表明系统可能对某些邮箱地址设置了标识。
- **工具集成建议**：一名成员建议封装类似 crawl AI 的工具以增强 Windsurf 的功能，允许用户通过集成在系统提示词（system prompt）中的命令执行网页抓取。
   - 另一位用户对实现允许用户为外部工具提供 API keys 的功能表示兴趣，认为这类功能将促进 Windsurf 的更广泛使用。
- **对 Windsurf 功能的挫败感**：多条消息指出 Windsurf 的当前功能（如自动代码编辑或命令执行）未能按预期运行，经常导致错误。
   - 用户注意到了死循环、API 问题以及与工具内操作相关的模糊错误，导致了困惑和挫败感。
- **升级方案困难**：尝试升级 Windsurf 方案的用户报告称收到了成功提示，但面临账户状态保持不变的问题。
   - 一名用户发布了截图，显示尽管有升级成功的消息，但没有任何实际调整，这引发了联系支持团队的建议。
- **资源共享与社区帮助**：成员们讨论了某些 Windsurf 功能缺乏文档的问题，特别是关于扩展开发以及在平台上使用特定编程语言的说明。
   - 社区成员分享了资源和指南，强调更好的用户支持和更详细的教程将提升整体体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/cascade#memories">Windsurf - Cascade</a>: 未找到描述</li><li><a href="https://x.com/windsurf_ai/status/1879332883489591404">来自 Windsurf (@windsurf_ai) 的推文</a>: 如何使用 Command</li><li><a href="https://codeium.com/blog/windsurf-wave-2">Windsurf Wave 2</a>: 介绍 Wave 2，我们对 Windsurf 编辑器的第二批更新。</li><li><a href="https://codeium.canny.io/feature-requests/p/1-upload-image-2-web-crawling-for-information-3-indexing-docs">1. 上传图片，2. 网页抓取信息，3. 索引文档 | 功能请求 | Codeium</a>: 上传图片或截图到聊天中，以便在进行 UI 更改时作为参考。使用 @web 让 Agent 抓取网页并获取相关资源。</li><li><a href="https://cursor.directory/">Cursor Directory</a>: 为你的框架和语言寻找最佳的 Cursor 规则。
</li>
</ul>

</div>

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1329541236884766720)** (116 条消息🔥🔥): 

> `活动页面困惑, Gemini 模型端点变更, OpenRouter API 与地区限制, DeepSeek 性能, BYOK API Key 集成` 


- **对活动页面的困惑**：用户对**活动页面**表示不满，图表在多个 Key 之间显示相同的信息，导致用户困惑这究竟是一个 **bug** 还是预期功能。
   - 几位用户指出，按 Key 追踪使用情况对于更好的管理至关重要。
- **Gemini 模型的端点变更**：一位成员指出 **Gemini 2.0 flash** 模型有了新的请求端点，这导致 OpenRouter 设置中出现了错误。
   - 其他用户确认**网站文档**必须更新以反映这些变化。
- **地区限制问题**：多位用户报告称，来自**香港**的 OpenRouter 请求面临限制，而使用新加坡 IP 则可以解决该问题，这表明可能存在新的中继节点。
   - 其他人回想起 OpenAI 和 Anthropic 长期以来在该地区受到限制，这可能解释了当前的挑战。
- **DeepSeek 性能与配置**：关于 **DeepSeek V3** 性能的用户体验讨论，包括关于获得最高质量输出的最佳设置的咨询。
   - 一些用户注意到结果质量参差不齐，引发了关于在不同用例下有效性的讨论。
- **BYOK 集成反馈**：用户建议 **BYOK (Bring Your Own Key)** 功能可以受益于更清晰的确认 Key 集成成功的消息。
   - 用户分享了关于请求的额外元数据反馈，以指示 BYOK 是否成功激活。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/docs/provider-routing',">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://meowapps.com/ai-engine/">AI Engine</a>：为 WordPress 添加 AI 功能。聊天机器人、表单、Copilot、内容生成等等！</li><li><a href="https://openrouter.ai/docs/integrations">Integrations | OpenRouter</a>：通过 OpenRouter 使用您自己的提供商 Key</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat">DeepSeek V3 - API, Providers, Stats</a>：DeepSeek-V3 是 DeepSeek 团队的最新模型，建立在先前版本的指令遵循和编码能力之上。在近 15 万亿 token 上进行预训练，报告的评估结果...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1329560289318015008)** (45 条消息🔥): 

> `DeepSeek 3 性能与配置，Aider 与 GitHub 里程碑，OpenRouter 提供商设置，量化对 DeepSeek 的影响，CodeGate 安全特性` 


- **DeepSeek 3 面临上下文问题**：一位成员报告称 OpenRouter 仅将他们路由到具有 **16k context** 能力的 **DeepSeek3 模型**，导致频繁出现上下文大小错误。
   - 另一位成员建议在设置中忽略该提供商，这一方案得到了认可。
- **Aider 庆祝 GitHub 成功**：Aider 社区庆祝在 GitHub 上突破 **25k stars**，标志着该项目的一个重要里程碑。
   - 许多人对其被认可为顶尖 AI 编程助手表示赞赏。
- **关于 DeepSeek 性能对比的讨论**：一位成员表示需要对 **DeepSeek3 的全精度版本与 Q4 或 Q5 等量化版本**进行性能对比，以评估影响。
   - 他们表达了对全精度的满意，并对使用重度量化版本持怀疑态度。
- **CodeGate 专注于隐私的功能**：一位开发者分享了关于 **CodeGate** 的见解，强调其通过本地运行来增强安全性，防止在 AI 编程过程中泄露敏感数据。
   - 他们链接了两个 YouTube 视频，演示了 CodeGate 如何保护编程免受安全风险。
- **对 Python 应用失败成本的担忧**：一位用户指出，**Python app** 中失败尝试的循环可能会迅速升级并产生巨额成本。
   - 这突显了在运行时维护高效代码所面临的运维挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/provider-routing)">OpenRouter</a>: LLM 的统一接口。为您的 prompt 寻找最佳模型和价格。</li><li><a href="https://aider.chat/docs/usage/commands.html">In-chat commands</a>: 使用 /add、/model 等聊天内命令控制 aider。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ho0w52/deepseek_does_not_need_5_hours_to_generate_1/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF">unsloth/DeepSeek-V3-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=WimBevc_Ji0">Avoid risky dependencies in AI-generated code with CodeGate</a>: AI 编程助手是惊人的生产力助推器，但由于知识过时，它们可能会给您的项目引入安全漏洞。CodeGate ...</li><li><a href="https://www.youtube.com/watch?v=lH0o7korRPg">Stop AI coding assistants from leaking your secrets with CodeGate</a>: 您的 AI 编程助手是否正在泄露您的秘密？答案很可能是“是的”。了解 CodeGate 如何通过加密您的...来保护您的隐私和安全。</li><li><a href="https://github.com/stacklok/codegate">GitHub - stacklok/codegate: CodeGate: CodeGen Privacy and Security</a>: CodeGate: CodeGen 隐私与安全。通过在 GitHub 上创建账号为 stacklok/codegate 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1329542416138506263)** (60 条消息🔥🔥): 

> `Agentic Tools for Code Exploration, User Customization of Aider, Scraping Limitations with Aider, Context Limits Issue, Sparse Priming Representation` 


- **讨论了多种 Agentic 工具选项**：重点介绍了用于探索代码库的各种 Agentic 工具，包括 **Aide.dev**、**Cursor** 以及使用 **PydanticAI** 开发的自定义代码探索工具。
   - 一位用户分享了构建 **代码探索 CLI** 的经验，该工具利用了终端、ctags 和文件读取功能。
- **Aider Prompt 的用户自定义**：一位用户分享了针对特定任务自动创建“编码功能 Prompt”的方法，强调了有效查询管理的必要性。
   - 提出了改进本地 RAG 流程的建议，旨在降低成本并提高用户 Prompt 的效率。
- **遇到的抓取限制**：用户在使用 **Azure GPT-4o** 时遇到了问题，过大的抓取内容因超出上下文限制而导致 `BadRequestError`。
   - 建议手动复制抓取内容的相关部分，而不是提交整个页面，以避免此错误。
- **上下文限制引发的挫败感**：一位用户表达了对触发上下文限制的沮丧，尽管在使用 Aider 时感觉仍在限制范围内。
   - 讨论集中在 Aider 是否将完整的抓取数据发送给模型，这被指出是一个问题。
- **对 Sparse Priming Representation 的兴趣**：介绍了 **Sparse Priming Representation** 的概念，并提供了深入理解的相关资源引用。
   - 用户讨论了它的影响，以及它如何潜在地增强 Aider 的功能和效率。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.ag2.ai/notebooks/agentchat_swarm_enhanced">Enhanced Swarm Orchestration with AG2 - AG2</a>: 未找到描述</li><li><a href="https://forum.cursor.com/t/cursor-not-able-to-access-full-code-base/36021/11">Cursor not able to access full code base</a>: 你们不该那样打广告</li><li><a href="https://bw2.github.io/ConfigArgParse/configargparse.ArgumentParser.html#__init__):">configargparse.ArgumentParser</a>: 未找到描述</li><li><a href="https://x.com/VictorTaelin/status/1873948475299111244">Taelin (@VictorTaelin) 的推文</a>: 成功了！演示时间 🥳（下一个 10 倍生产力飞跃来了？）假设你必须重构一个大型代码库，例如：&gt; “使用 I32 代替 U32 作为原生数字类型” 这个任务本身很简...</li><li><a href="https://github.com/TheFoundation-Global/TheRitualistsPrimer.git">GitHub - TheFoundation-Global/TheRitualistsPrimer: A Sparse Priming Representation reference library</a>: 一个 Sparse Priming Representation 参考库。通过在 GitHub 上创建一个账号来为 TheFoundation-Global/TheRitualistsPrimer 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1329581141849669642)** (2 条消息): 

> `Helicone LLM 可观测性, Activepieces 工具概览, LLM 文档资源` 


- **Helicone 提供 LLM 可观测性**：[Helicone GitHub 仓库](https://github.com/Helicone/helicone) 展示了一个**开源 LLM 可观测性平台**，旨在高效地监控、评估和实验 AI 模型。
   - 核心功能包括**追踪请求和成本**、LLM 安全层、缓存以及可自定义的速率限制，并建议使用 docker-compose 或其云版本运行。
- **Activepieces 连接各种 LLM**：[Activepieces 网站](https://activepieces.com/) 提供了有用的集成，以便与多个 LLM 服务协作，并通过 [llms.txt](https://www.activepieces.com/docs/llms.txt) 提供其指标数据的文档。
   - 指标包括 Activepieces 服务中如 **3747** 等各种用途，展示了其作为资源丰富的集成工具的潜力。
- **LLM 文档资源**：有许多 LLM 资源可用，包括位于 [squared.ai](https://squared.ai/) 的 **AI Squared**，在其[文档](https://docs.squared.ai/llms.txt)中可以找到特定的使用指标。
   - 其他选项包括 Anthropic 和 Aporia，可通过其各自的链接访问完整文档，其中显示了 Anthropic 的使用指标为 **90337**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llmstxt.site">llms.txt 目录</a>: 查找并探索来自各种产品和服务的 llms.txt 文件。</li><li><a href="https://github.com/Helicone/helicone">GitHub - Helicone/helicone: 🧊 开源 LLM 可观测性平台。一行代码即可监控、评估和实验。YC W23 🍓</a>: 🧊 开源 LLM 可观测性平台。一行代码即可监控、评估和实验。YC W23 🍓 - Helicone/helicone
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1329611693441028147)** (58 条消息🔥🔥): 

> `Nous 融资, OpenAI 薪酬结构, 员工所有权与股份, 模型托管选项, 训练框架` 


- **Nous Research 获得 4 亿美元融资**：一位成员指出 **Nous Research** 已成功获得 **4 亿美元** 融资，这表明外界对其未来发展充满浓厚兴趣。
   - *许多人通过 HF 进行托管或购买专业订阅*，这表明对其服务的需求正在增长。
- **关于 OpenAI 薪酬策略的见解**：讨论集中在 OpenAI 复杂的薪酬结构上，部分员工获得的是 **利润参与单位 (PPUs)** 而非传统的股票。
   - 据指出，虽然资深工程师可以获得股份，但普通研究人员可能只有 **非股份权益薪酬**。
- **预期中的员工股票期权**：成员们讨论了 OpenAI 和 Anthropic 的员工可能如何获得股份，特别是在允许 2022 年在职员工进行出售的二级市场轮次之后。
   - 考虑到在公司现有结构下这些股份可能并不代表真正的所有权，人们对其股份的实际价值表示担忧。
- **关于模型托管能力的激烈辩论**：一位成员指出了通过 Hugging Face 托管模型或利用专业订阅的选项，展示了多样化的使用场景。
   - 另一位成员指出 **咨询** 和 **GPU 租赁** 服务也很受欢迎，突显了获取 **AI 资源** 的各种方法。
- **各种训练框架受到关注**：讨论透露 Nous 团队主要使用 **Axolotl** 进行微调，并提到了 **LlamaFactory** 和 **Unsloth** 等替代方案。
   - 对所使用的更广泛框架的见解包括 **Lingua**、**Olmo** 和 **TorchLightning**，标志着模型训练的多样化方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.levels.fyi/blog/openai-compensation.html">OpenAI PPUs: OpenAI 独特的股权补偿运作方式</a>：深入了解当今最热门且最神秘的 AI 公司之一。</li><li><a href="https://www.levels.fyi/companies/openai/salaries/software-engineer">OpenAI 软件工程师薪资 | $362K-$1.34M+ | Levels.fyi</a>：OpenAI 在美国的软件工程师薪资从 L3 的每年 36.2 万美元到 L6 的 134 万美元以上不等。美国年薪中位数为 23.8 万美元。查看...</li><li><a href="https://fortune.com/2024/12/17/hundreds-openai-employees-10-million-payday-softbank-stock-tender-offer-details/">数百名 OpenAI 现任和前任员工即将通过在私人股票出售中套现高达 1000 万美元而获得巨额回报</a>：据消息人士告诉 Fortune，作为公司向 SoftBank 提供的 16 亿美元要约收购的一部分，一组 OpenAI 现任和前任员工有资格套现价值高达 1000 万美元的股份。</li><li><a href="https://openrouter.ai/nousresearch/hermes-3-llama-3.1-405b/providers">Nous: Hermes 3 405B Instruct – 供应商状态</a>：查看供应商状态并向 Nous: Hermes 3 405B Instruct 发起负载均衡请求 - Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力...</li><li><a href="https://www.businessinsider.com/microsoft-openai-put-price-tag-achieving-agi-2024-12">报道称 OpenAI 和 Microsoft 已为实现 AGI 标价</a>：据报道，两家公司去年签署了一项协议，将 AGI 定义为能够产生 1000 亿美元利润的系统。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1329561924236607559)** (17 条消息🔥): 

> `RAG 聊天机器人架构，小型模型替代方案，LLM 修改建议，结构化输出挑战` 


- **RAG 聊天机器人面临 GPT-2 限制**：一位用户分享了他们在基于 **PDF 数据**构建 RAG 聊天机器人时使用 **GPT-2** 的困扰，指出 Token 大小限制导致输出无意义且重复。
   - 参与者建议切换到更新的小型模型，因为 **GPT-2** 可能不适合该任务。
- **探索替代的小型模型**：另一位成员推荐了 **smollm** 和 **Qwen** 等较新的模型，作为能够超越 **GPT-2** 的可行替代方案。
   - 该用户表示有兴趣在他们的项目中探索这些建议的模型。
- **LLM 文档修改的挑战**：一位成员询问了让 LLM 对长文档提供建议，并根据定义的一组规则返回 **JSON 输出** 的最佳方法。
   - 另一位用户分享了在结构化输出方面的糟糕体验，理由是他们的模型无法准确评估修改。
- **改进文档修改输出**：出现了关于 LLM 理解力的问题，导致即使没有违反规则，也会产生大量的修改建议。
   - 随后讨论了在其修改 Schema 中包含原因字段（reason fields）和开始/停止标记（start/stop markers）的有效性。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1329546499566342227)** (6 条消息): 

> `新型神经网络架构，Titans 的 PyTorch 实现，大语言模型书籍` 


- **令人兴奋的新型神经网络架构揭晓**：一位成员分享了一种[新型神经网络架构](https://arxiv.org/pdf/2501.00663v1)，旨在改进 Attention 机制中的长期记忆，并详细介绍了其在增强依赖建模方面的潜力。
   - 该研究讨论了在保持快速可并行化训练和推理的同时，利用历史上下文。
- **Lucidrains 的 PyTorch Titans 实现已上线**：一位成员强调 @lucidrains 已开始在 PyTorch 中实现 [Titans 架构](https://github.com/lucidrains/titans-pytorch)，该架构以其针对 Transformer 的内存效率而闻名。
   - 此实现旨在提供易于使用的工具，以便在 Transformer 模型中利用最先进的记忆策略。
- **大语言模型的基础见解**：一位成员提到了一本关于[大语言模型的书籍](https://arxiv.org/abs/2501.09223)，该书侧重于基础概念，包括预训练、生成模型和对齐方法。
   - 该书专为自然语言处理领域的学生和专业人士量身定制，是理解大语言模型关键方面的有用参考。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.09223">Foundations of Large Language Models</a>: 这是一本关于大语言模型的书。正如书名所示，它主要侧重于基础概念，而非涵盖所有尖端技术。该书...</li><li><a href="https://arxiv.org/abs/2501.00663v1">Titans: Learning to Memorize at Test Time</a>: 十多年来，关于如何有效利用循环模型和 Attention 进行了广泛的研究。虽然循环模型旨在将数据压缩到固定大小的内存中...</li><li><a href="https://github.com/lucidrains/titans-pytorch">GitHub - lucidrains/titans-pytorch: Unofficial implementation of Titans, SOTA memory for transformers, in Pytorch</a>: Titans 的非官方 PyTorch 实现，Transformer 的 SOTA 内存方案。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1329546499566342227)** (6 messages): 

> `新型神经长期记忆模块，PyTorch 中的 Titans 实现，大语言模型概览` 


- **神经长期记忆的新架构**：引入了一种新的 [神经长期记忆模块](https://arxiv.org/abs/2501.00663v1)，通过允许 Attention 在处理当前输入时访问历史上下文，增强了依赖建模能力。
   - 摘要强调，该方法保留了快速并行训练和推理能力，平衡了上下文长度和依赖准确性。
- **PyTorch 中的 Titans 记忆实现**：开发者 [lucidrains](https://github.com/lucidrains/titans-pytorch) 已开始实现 Titans 架构，该架构专注于为 PyTorch 中的 Transformer 提供 SOTA 记忆能力。
   - 该 GitHub 仓库提供了一个非官方实现，为广大 AI 社区探索高级 Transformer 功能做出了贡献。
- **大语言模型的基础概念**：讨论了一本新书，该书围绕大语言模型的基础概念展开，重点关注四个关键领域：预训练 (pre-training)、生成模型 (generative models)、提示技术 (prompting techniques) 和对齐方法 (alignment methods)。
   - 正如 [摘要](https://arxiv.org/abs/2501.09223) 所述，该书面向大学生和专业人士，为对自然语言处理领域感兴趣的人士提供了有用的参考。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.09223">Foundations of Large Language Models</a>：这是一本关于大语言模型的书。正如书名所示，它主要关注基础概念，而非涵盖所有前沿技术。这本书是...</li><li><a href="https://arxiv.org/abs/2501.00663v1">Titans: Learning to Memorize at Test Time</a>：十多年来，关于如何有效利用循环模型和 Attention 进行了广泛的研究。虽然循环模型旨在将数据压缩到固定大小的内存中...</li><li><a href="https://github.com/lucidrains/titans-pytorch">GitHub - lucidrains/titans-pytorch: Unofficial implementation of Titans, SOTA memory for transformers, in Pytorch</a>：Titans 的非官方实现，为 Pytorch 中的 Transformer 提供 SOTA 记忆 - lucidrains/titans-pytorch
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1329573547378610289)** (11 messages🔥): 

> `NotebookLM 提示词，虚拟旅行代理机器人，用于科学的 NotebookLM，AI Studio 的可靠性，用于计算机主题的 NotebookLM` 


- **分享提示词促进学习**：一位成员强调了分享提示词的重要性，认为这能增强关于使用 **NotebookLM** 的集体知识。
   - “这就是为什么我们也需要分享提示词，哈哈”是支持这一想法的幽默评论。
- **虚拟旅行代理研讨会取得成功**：另一位成员分享了他们最近关于为赞比亚度假创建 **虚拟旅行代理机器人 (virtual travel agent bot)** 的研讨会经验，并展示了会议中一份详细的大纲文档。
   - 您可以在 [此处](https://notebooklm.google.com/notebook/51fb6a47-1703-4c03-ac83-12ef3b1b0caf/audio) 找到完整的大纲。
- **NotebookLM 在科学主题上面临挑战**：有人对 **NotebookLM** 在科学应用中的局限性表示担忧，一位成员将其描述为“在建立联系方面极其愚蠢”。
   - 他们指出，它往往只注意到单词的邻近性，而不理解它们之间的相互作用。
- **AI Studio 作为可靠的替代方案**：一位用户推荐 **AI Studio** 作为处理各种任务时比 **NotebookLM** 更可靠的工具，强调了其卓越的性能。
   - 他们对 **NotebookLM** 的能力表示怀疑，特别是在建立准确联系方面。
- **NotebookLM 在计算机主题方面表现出色**：一位成员报告了使用 **NotebookLM** 理解 HTTP 等计算机相关主题的积极体验，称其表现优于其他领域。
   - 他们发现 **AI** 在理解和解释全新信息方面非常有效，并指出了该工具的长处。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com/notebook/51fb6a47-1703-4c03-ac83-12ef3b1b0caf/audio">未找到标题</a>：未找到描述</li><li><a href="https://youtu.be/Ce3HjJ9hTaA">Your FREE AI Tutor is Here! Learn 10x Faster with NotebookLM 🚀</a>：订阅以获取最新动态：https://bit.ly/3Q98G7p 探索来自 Google 的强大 AI 工具 NotebookLM 如何彻底改变您的学习体验。在...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1329554586846892245)** (76 条消息🔥🔥): 

> `Podcast Generation Limitations, Notebooks Usage Questions, YouTube Video Link Handling, Interactive Mode and Bugs, Audio Overview Customization` 


- **Podcast 生成限制**：用户讨论了有效生成 Podcast 的困难，特别是在处理多个来源或上传时，有人提到需要更简洁的链接格式。
   - 一位用户指出，他们已经等待了一个多月 Podcast 功能才开始运作，这引发了对潜在 Bug 或服务器问题的担忧。
- **Notebooks 使用问题**：关于用户可以创建的 Notebook 数量限制存在担忧，一位成员提到**免费用户限制约为 100 个**。
   - 此外，用户讨论了共享 Notebook 的影响，以及控制协作者添加或删除来源的权限。
- **YouTube 视频链接处理**：一位用户在将直播视频上传到 NotebookLM 时遇到问题，原因是链接格式，后来发现将其转换为直接的 YouTube 链接解决了问题。
   - 社区确认，NotebookLM 需要直接链接才能正确识别视频，应避免包含 'live' 的链接。
- **交互模式与 Bug**：多位用户对交互模式表示沮丧，有些人无法访问其功能或遇到持续加载的问题。
   - 故障排除建议包括管理 Cookie 以及探索不同浏览器的兼容性作为潜在解决方案。
- **音频概览自定义**：围绕自定义音频摘要的可能性展开了讨论，一些成员询问绕过某些自动回复的可行性。
   - 用户对音频概览中重复使用对话填充词及其对用户体验的影响表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtube.com/live/j7rItPwWDaY.">litdb + Jupyter lab</a>: litdb 现在与 Jupyter lab 兼容。此视频将展示一些新功能。现在您可以将文献搜索和分析保存在笔记中...</li><li><a href="https://www.mdpi.com/2075-4698/15/1/6">AI Tools in Society: Impacts on Cognitive Offloading and the Future of Critical Thinking</a>: 社会中的 AI 工具：对认知卸载的影响和批判性思维的未来。人工智能 (AI) 工具的普及改变了日常生活的许多方面，但其对批判性思维的影响仍未得到充分探索。本研究调查了...</li><li><a href="https://www.youtube.com/watch?v=j7rItPwWDaY">litdb + Jupyter lab</a>: litdb 现在与 Jupyter lab 兼容。此视频将展示一些新功能。现在您可以将文献搜索和分析保存在笔记中...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1329547536603680840)** (68 条消息🔥🔥): 

> `Perplexity Pro Issues, Model Performance, Image Generation Problems, Subscription Activation Challenges, Student Discounts Inquiry` 


- **用户遇到 Perplexity Pro 问题**：多位成员报告了 Perplexity Pro 服务的问题，特别是即使在尝试了故障排除建议后，**o1** 等模型设置仍无法识别。
   - 一位用户提到他们的 VPN 可能会导致问题，而另一位用户确认 **SONNET** 和 **4o** 等模型运行正常。
- **图像生成审查投诉**：讨论了图像生成审核的不一致性，一位用户指出特定请求（如生成粉色马头）的审核失败。
   - 他们强调了一个令人困扰的案例，即生成了不当内容，而预期的输出却被拒绝。
- **激活促销代码的挑战**：一位成员表示在激活从 T-Mobile 获得的免费 Perplexity Pro 访问促销代码时遇到困难，正在寻求社区指导。
   - 另一位用户建议联系 **support@perplexity.ai** 以寻求激活问题的帮助。
- **用户咨询学生折扣**：一位用户询问使用 Perplexity 是否有针对学生的优惠费率，特别是在欧盟地区。
   - 他们表示希望收到关于学生折扣的支持性消息。
- **对回答质量的担忧**：用户注意到最近更新后，模型的回答长度和详细程度显著减少，使得交互显得缺乏信息量。
   - 许多人表示失望，特别是对于他们之前非常欣赏的回答深度有所下降。


  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1329602382824476783)** (5 条消息): 

> `Starship 7 incident, China's Space Solar Array, Apple's USA-Made iPhone Chips, OpenAI's Economic Blueprint, Time Travel Discussion` 


- **Starship 7 在飞行中失联**：多位用户分享了讨论 **Starship 7** 事故的链接，特别是关于其在飞行过程中失联的情况。详细的调查和分析可以在[这里](https://www.perplexity.ai/search/starship-7-lost-in-flight-2oHRnlZlR5mGDqkus5TtHA)找到。
- **中国雄心勃勃的空间太阳能阵列计划**：发布了一段讨论**中国**空间太阳能阵列计划的视频，强调了其对能源资源的潜在影响。更多内容可以在标题为“[YouTube](https://www.youtube.com/embed/necQU3gNx2g)”的视频中查看。
- **Apple 首款美国制造的 iPhone 芯片**：提到了 **Apple** 正在生产其首款美国制造的 iPhone 芯片，强调了制造业的转移。关于这一进展的细节在经济影响的背景下进行了进一步讨论。
- **OpenAI 经济蓝图**：分享了一个讨论 **OpenAI 经济蓝图**的链接，该蓝图概述了未来增长和可持续发展的战略。这些战略的影响对整个科技行业都具有重要意义。
- **时间旅行——可能吗？**：一个讨论**时间旅行**概念及其可行性的链接引发了用户的好奇。关于这一概念背后科学可能性的见解可以在[这里](https://www.perplexity.ai/page/zeitreisen-moglich-und-wie-JXlUhH1GTcSYQrkAeUNIIw)探索。



**提及的链接**：<a href="https://www.youtube.com/embed/necQU3gNx2g">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1329550935243886704)** (5 条消息): 

> `Sonar and Sonar-Pro models, Custom stop parameters error, crewAI model troubleshooting` 


- **Sonar 模型的潜在变化**：一位成员注意到 **Sonar** 和 **Sonar-Pro** 模型出现在 labs 中，引发了关于 API 模型即将发生变化的猜测。
   - *“API 模型是否又要改变了？”* 暗示了模型领域的持续发展。
- **crewAI 用户遇到自定义停止参数错误**：一位用户报告了在尝试让 **pplx** 与基础 crew 配合工作时遇到的**自定义停止参数**错误，引发了一些故障排除讨论。
   - 另一位用户建议查看模型文档（链接在[这里](https://docs.perplexity.ai/guides/model-cards)）作为解决方案的一部分。
- **尝试了所有模型但问题仍然存在**：用户在尝试了所有三个可用模型但仍遇到相同错误后表达了挫败感，并表示 *“谢谢，但我已经尝试了所有三个模型”*。
   - 他们表示该问题可能并非 Perplexity 特有，并考虑在 **crewAI** 社区继续讨论。
- **寻找停止参数的快速修复方法**：他们提到发现了一个 *monkey fix*（临时补丁），可以在进行 API 调用之前绕过 **litellm** 中的停止参数，提供了一个临时解决方案。
   - 这一解决方案突显了用户在等待正式修复时解决技术挑战的聪明才智。



**提及的链接**：<a href="https://docs.perplexity.ai/guides/model-cards">未找到标题</a>：未找到描述

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1329546130731696235)** (76 条消息🔥🔥): 

> `David Lynch 在 Lodge 中, 将 Stable Diffusion 用于商业, 图像生成与 ControlNet 问题, 使用个人照片训练 LoRA, 在 Stable Diffusion webUI 之间切换` 


- **David Lynch 在 Lodge 的出现引发了黑色幽默**：成员们对 David Lynch 在 Lodge 的现身做出了反应，讨论反映了对 *艺术中出人意料的道德人物* 的反思。
   - 评论中包含了关于 Lynch 艺术声誉的 *黑色幽默*，突显了对话的奇特本质。
- **探索 Stable Diffusion 的商业用途**：讨论围绕使用 Stable Diffusion 为按需打印（print-on-demand）商店生成图像展开，强调了为打印分辨率进行图像放大（upscaling）的需求。
   - 成员们分享了关于商业用途许可问题的见解，并澄清除非在特定应用程序中使用该模型，否则输出内容通常是可以自由使用的。
- **图像生成和 ControlNet 的常见困扰**：一位用户提出了关于使用 Stable Diffusion 生成图像以及如何有效使用图像输入的问题，确认了即使在图生图（image-to-image）任务中也需要提示词（prompts）。
   - 建议了不同的方法，如使用 lineart 或 ControlNet，重点是通过各种技术提取可用数据。
- **训练 LoRA 模型的挑战**：一位用户在为孩子的照片创建 LoRA 模型时遇到了问题，询问是否需要裁剪图像以及如何处理分辨率。
   - 针对图像处理和潜在的模型更改给出了建议，强调了数据集质量在训练中的重要性。
- **在 SD Forge 和 Automatic1111 之间切换**：一位考虑切换到 Automatic1111 的用户遇到了与特定 Huggingface 模型相关的卡通化输出问题，这表明存在严重的功能差异。
   - 提到提示词样式存储在 *styles.csv* 中，方便在不同 webUI 之间移动时传输和管理保存的提示词。



**提及的链接**：<a href="https://github.com/lllyasviel/stable-diffusion-webui-forge?tab=readme-ov-file#stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>：通过在 GitHub 上创建账号，为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。

  

---


### **Nomic.ai (GPT4All) ▷ #[announcements](https://discord.com/channels/1076964370942267462/1090471714888102009/1329873241363451944)** (1 条消息): 

> `Nomic Embed Vision, Apache 2.0 许可证, 多模态任务, 开放权重与代码` 


- **Nomic Embed Vision 采用 Apache 2.0 许可证**：Nomic Embed Vision 现在根据 [Apache 2.0 License](https://x.com/nomic_ai/status/1880313093097693212) 发布，为开发者提供了灵活性和访问权限。
   - *这一转变允许为图像、文本和多模态任务提供高质量、统一的嵌入空间（embedding space），* 显著增强了可用性。
- **Nomic Embed Vision 超越竞争对手**：据报道，该平台 *性能优于 OpenAI CLIP 和 text-embedding-3-small*，使其成为嵌入能力的竞争性选择。
   - **高性能基准测试** 突显了在各个领域进行高级应用的潜力。
- **获取开放权重和代码**：Nomic Embed Vision 还提供 **开放权重和代码**，使社区能够在现有架构的基础上进行构建。
   - *此举代表了对透明度* 和协作开发的承诺，促进了该领域的创新。



**提及的链接**：<a href="https://x.com/nomic_ai/status/1880313093097693212">来自 Nomic AI (@nomic_ai) 的推文</a>：Nomic Embed Vision 现在采用 Apache 2.0 许可证。- 高质量、统一的图像、文本和多模态任务嵌入空间。- 性能优于 OpenAI CLIP 和 text-embedding-3-small - 开放权重...

  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1329576263676989653)** (66 条消息🔥🔥): 

> `AI 伦理、模型推荐、模型性能、自定义 URL 方案、Qwen2.5-1.5B 模板` 


- **关于伦理护栏（Ethical Guardrails）的辩论**：讨论强调了关于谁的伦理标准在主导 AI “伦理护栏”的担忧，并对西方和中国的标准持有不同意见。
   - 一位用户强调，如果护栏过度限制了对良性话题之外的探索，就会削弱 AI 的存在意义。
- **模型建议与性能**：建议包括使用 LocalLlama 进行通用模型查询，以及使用 DavidAU 的特定模型作为写作辅助，并强调了 8GB VRAM 的性能权衡。
   - 用户讨论了使用量化（Quantization）技术来优化模型的速度和效率，并报告了在不同配置下的不同结果。
- **为应用程序创建自定义链接**：一位用户寻求关于创建自定义 URL 方案（URL scheme）的建议，以便根据链接类型（例如 `hyperscope://`）打开不同的程序（如 Emacs）。
   - 有建议认为嵌入 .md 或 .html 文件可以促进直接调用特定应用程序，从而简化对某些知识片段的访问。
- **模型模板的挑战**：一位用户在 Qwen2.5-1.5B 中遇到了解析模板的问题，在使用某些格式时收到错误，而其他用户则分享了使用 ChatML 模板的成功经验。
   - 用户对模板失败表示沮丧，特别是在配合 LocalDocs 使用时，并强调了进行适当设置调整的必要性。
- **旧硬件的性能权衡**：一位用户幽默地指出了使用旧款 NVIDIA Quadro NVS300 GPU 运行 AI 的局限性，并强调即使是移动设备的性能也会更好。
   - 参与者承认，在 VRAM 非常有限的情况下，运行更复杂的 AI 模型是不切实际的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/nomic-ai/gpt4all/wiki#feature-matrix">Home</a>：GPT4All：在任何设备上运行本地 LLM。开源且可用于商业用途。 - nomic-ai/gpt4all</li><li><a href="https://huggingface.co/mradermacher/Hel-v2.5-8b-DARK-FICTION-i1-GGUF/tree/main">mradermacher/Hel-v2.5-8b-DARK-FICTION-i1-GGUF at main</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/main_classes/quantization">Quantization</a>：未找到描述</li><li><a href="https://github.com/nomic-ai/gpt4all/issues/3362#issuecomment-2595330752">EM German Mistral 的默认提示词模板解析失败 · Issue #3362 · nomic-ai/gpt4all</a>：错误报告 “一旦你提问任何问题，就会出现上述错误信息，之后无法进行任何操作。当我切换到像 GPT-4All Falcon 这样的语言模型时，查询可以工作，但速度很慢...”</li><li><a href="https://huggingface.co/DavidAU">DavidAU (David Belton)</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1329548302571540572)** (8 条消息🔥): 

> `LeetGPU 平台、CUDA 学习资源` 


- **LeetGPU：你的免费 CUDA 游乐场**：一个新平台 [LeetGPU](https://leetgpu.com/) 已经发布，它是一个免费的在线游乐场，用于编写和执行 CUDA 代码，无需注册或 GPU 访问权限。
   - 开发者希望改进用户体验，鼓励用户提供反馈。
- **CUDA 入门指南**：一位用户询问 CUDA 的入门指南，得到了书籍 **《CUDA by Example: An Introduction to General-Purpose GPU Programming》** 的推荐。
   - 虽然该资源有些过时，但用户指出它有效地涵盖了基础知识，并辅以各种可下载的材料和勘误表。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://leetgpu.com/,">LeetGPU</a>：未找到描述</li><li><a href="https://developer.nvidia.com/cuda-example">CUDA By Example</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1329638289975672863)** (6 条消息): 

> `Triton kernel 优化，Warp specialization，Triton kernel 融合` 


- **Triton Stage1_v2 Buffer Size 优化**：**stage1_v2** 的改进表明，使用新 Buffer Size（aligned_num_tokens, aligned_num_experts）的向量化比原始实现访问 DRAM 更快。
   - 讨论暗示将新旧方法结合，以获得预期更快的 kernel。
- **Triton 中的 Warp Specialization**：[Automatic Warp Specialization Optimization](https://github.com/triton-lang/triton/pull/5622) 被强调为通过采用异步执行模型来管理不同的硬件单元，从而增强 kernel 性能。
   - 这种优化允许更好的数据通信，并提升 kernel 执行的性能。
- **针对 Triton Kernel 融合建议的 Barrier**：计划实现**基于数据流的 barrier**，这将有助于解决由于缺乏 blockwise 同步而导致的 Triton kernel 融合困难。
   - 该建议强调了在 Triton 开发中改进同步机制的需求。
- **仓库变更引发关注**：一位成员对 **warp specialization** 仓库（曾作为镜像公开）现在成为 Triton 主仓库的一部分感到兴奋。
   - 在遇到分支构建问题后，鉴于更新后的可访问性，他们认为这次变更是一个“幸运日”。



**提到的链接**：<a href="https://github.com/triton-lang/triton/pull/5622">Automatic Warp Specialization Optimization by htyu · Pull Request #5622 · triton-lang/triton</a>：Warp specialization 通过利用异步执行模型来增强 kernel 性能，其中 kernel 的不同部分由独立的硬件单元处理。数据通信...

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1329608399092056064)** (29 条消息🔥): 

> `Torch Profiler 问题，自定义 Autograd Function 实现，PyTorch 中的 Double Backward，优化资源共享，Backward Pass 中的中间张量管理` 


- **Torch Profiler 因内存 Bug 崩溃**：一位成员报告在 **libkineto** 中遇到了**内存损坏 bug**，导致在使用 torch 的 profiler 导出内存时间线时崩溃。
   - 尽管某些版本存在问题，但他们指出，这可能取决于所使用的具体 **Python** 和 **Torch** 版本。
- **实现带有 Double Backward 的自定义 Autograd Function**：一位成员正尝试创建一个可以执行 **addbmm** 和 **Softplus** 激活的**自定义 torch.autograd.Function**，并在实现 double backward 时遇到了问题。
   - 另一位用户建议为 backward pass 使用独立的 autograd function，并演示了在 `.backward()` 中使用**链式** backward 的方法。
- **自定义算子中 Double Backward 逻辑的挑战**：在讨论使用 **triton** 进行优化所需的自定义 double backward 时，一位成员指出，由于每个操作都是独立的 CUDA kernel 启动，PyTorch 的 autograd 引擎目前较慢。
   - 大家承认 **torch.compile()** 目前不支持 double backward，这影响了优化工作。
- **学习优化的资源共享**：有人询问学习 PyTorch 优化的资源，引发了关于 **Python runtime** 修改相关的有用文档和 YouTube 视频的讨论。
   - 一位用户强调在 **PyTorch 文档**中查找信息，但没有一份用于其写作的整合资源列表。
- **在 Backward Pass 期间管理中间张量**：一位成员提出了关于执行**两步 backward** pass 的问题，同时通过手动从计算图中删除节点来管理中间张量以减少内存使用。
   - 讨论引发了对如何在不重新调用完整 backward 计算的情况下高效执行此操作的方法的好奇。



**提到的链接**：<a href="https://pytorch.org/docs/2.5/generated/torch.func.grad.html#torch.func.grad">torch.func.grad &mdash; PyTorch 2.5 documentation</a>：未找到描述内容。

  

---

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1329805330393075784)** (1 messages): 

> `Custom torch.autograd.Function, addbmm with Softplus activation, Double backward implementation, PyTorch ops and Triton ops, Mathematical derivation` 


- **实现自定义 autograd 函数**：一名成员正尝试实现一个结合了 **addbmm** 和 **Softplus** 激活的自定义 `torch.autograd.Function`，旨在初步利用 PyTorch ops，随后过渡到 **Triton ops**。
   - 他们目前面临挑战，特别是在 **double backward** 的实现方面，并提供了一个[代码文件](https://cdn.discordapp.com/attachments/1189861061151690822/1329805329743085568/doble_bckwd_addbm_softplus.py?ex=678bad39&is=678a5bb9&hm=642b9eccbc52a388576c515689a4814880aedc2f5156c5519daad37366a37616&)供审查。
- **寻求 Double Backward 数学推导的帮助**：该成员请求关于如何在自定义函数中正确编写 **double backward** 功能的指导，并对所需的数学推导表示不确定。
   - 他们提到发现的相关资源有限，因此在社区内进一步询问是否有共享经验或有用的参考资料。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1329600388323606529)** (6 messages): 

> `LLM Agents, Business Logic vs LLM, Automation in Programming` 


- **对 LLM Agents 主导逻辑的担忧**：一名成员对使用 LLM 替代业务逻辑表示怀疑，认为这在**航空**等敏感应用中可能会导致灾难。
   - *“到处都在抛出 ‘Agent’ 这个流行语，”* 并补充说**逻辑约束**是必要的。
- **承认 LLM 的缺陷**：另一名成员同意，虽然 LLM 替代工程师是一个有趣的想法，但 LLM 应用中固有的缺陷是巨大的。
   - 令人担忧的是，当前的逻辑问题可能无法仅通过自动化来解决。
- **LLM 与工程工作流的未来**：讨论暗示了未来 LLM 可能会使用 “chain-of-thought” 方法来生成模型和代码，并与 SMT solvers 和全证明器（full provers）集成以增强工程工作流。
   - 然而，有人指出这将是**计算密集型（computationally expensive）**的，且仍可能无法解决根本的逻辑错误。
- **自动化减少人类工程师的需求**：一名成员推测 LLM 有潜力减少编程任务中对人类工程师的需求，尤其是在像 **Python** 这样结构化的语言中。
   - 这一过程可能会通过减少显而易见的问题来提高质量，但并不能消除逻辑错误。


  

---


### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1329567823948812300)** (1 messages): 

> `Linux arm64 runners, Copilot chat enhancements` 


- **Linux arm64 runners 发布**：团队宣布发布 **Linux arm64 托管 runners**，目前已在公共仓库中提供免费的公开预览。此更新可以增强使用 arm64 架构项目的 CI/CD 流水线。
   - 有关此版本的更多详细信息可以在 [GitHub changelog](https://github.blog/changelog/2025-01-16-linux-arm64-hosted-runners-now-available-for-free-in-public-repositories-public-preview/) 中找到。
- **Copilot 现在可以解释失败的 Actions 任务**：用户现在可以利用 PR 合并框或 Actions 任务页面中的 **“Explain Error”** 功能，向 Copilot 询问任务失败的原因。这实现了一个更具交互性的调试过程。
   - 展示此功能的截图可以在[发布文档](https://github.com/user-attachments/assets/04ffd085-cede-4342-b75c-7a80dbff7be9)中找到，显示了两种语境下的界面。



**提到的链接**：<a href="https://github.blog/changelog/2025-01-16-linux-arm64-hosted-runners-now-available-for-free-in-public-repositories-public-preview/">Linux arm64 hosted runners now available for free in public repositories (Public Preview) · GitHub Changelog</a>：Linux arm64 托管 runners 现在在公共仓库中免费提供（公开预览）

  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 messages): 

0x000ff4: 有没有什么我可以贡献的活跃话题？
  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1329601523822366832)** (7 条消息): 

> `Hardware requirements for development, CUDA coding without GPU, Ampere architecture necessity, Ideal GPUs for tensor cores, Apple M chip compatibility` 


- **讨论了 GPU 开发的硬件需求**：一位用户询问了开发所需的硬件，特别是是否需要 **Ampere 架构** 的 GPU。
   - 回复指出，虽然拥有硬件很有帮助，但即使没有物理 GPU，也可以在 [leetgpu.com](http://leetgpu.com) 针对 CUDA runtime API 进行编程。
- **披露了对 Tensor Cores 的关注**：会议澄清了 **thunderkittens** 主要专注于 **tensor cores**，这表明其更倾向于使用 **Ampere 系列** 及以上的 GPU。
   - 推荐的理想开发选项包括 **A100**、**4090** 或 **H100** 等 GPU。
- **Apple M 芯片兼容性**：一位成员提到，他们为使用 **Apple 芯片** 的用户提供了一个仓库，其中包含对 **M 芯片** 的适配。
   - 这突显了人们对跨多样化硬件生态系统兼容性日益增长的兴趣。



**提到的链接**：<a href="http://leetgpu.com">LeetGPU</a>：未找到描述

  

---


### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1329685741046857761)** (4 条消息): 

> `Familiarity with arc-agi-2 codebase, Self-improvement with grounding signal, Tree-sampling experiments with MCTS, Llama-3.2 performance on math tasks, AI feedback for model training` 


- **熟悉 arc-agi-2**：一位用户计划下周花时间熟悉 `arc-agi-2` 代码库，并寻求关于符合路线图的原子任务的指导。
   - 他们表示愿意投入 GPU 资源进行探索和贡献。
- **用于自我提升的 Grounding Signal**：一位成员讨论了他们专注于使用 grounding signal 进行自我提升，从 Llama-3.2 3b 的简单数学任务开始。
   - 他们的目标是产生可信的思维链（CoTs），通过潜在的 token 级强化学习来增强模型能力。
- **Llama-3.2 的数学任务表现**：另一位用户报告了 Llama-3.2 3b 在各种数学求和任务中的解决率，并指出最简单的提示词效果最好。
   - 他们强调了在成功的完成结果上进行训练以提高模型性能的重要性，并提到了非零温度采样的问题。
- **计划树采样实验**：一位成员分享说，他们正计划进行涉及简单树采样的实验，灵感来自蒙特卡洛树搜索（MCTS）技术。
   - 这种方法旨在增强他们在 AI 领域解决问题的论证方法。
- **利用 AI 反馈进行输出修正**：对话强调了利用 AI 反馈来有效评估和修正模型输出的策略。
   - 该方法旨在减少垃圾输出的发生，同时提高模型响应的可靠性。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1329675803210027028)** (10 条消息🔥): 

> `Flash Attention in Tinygrad, Memory Control Challenges, GPU Out-Of-Memory Errors, Nested-Loop Representation` 


- **Flash Attention 的尝试未达预期**：在花了 8 小时尝试将 **Flash Attention** 整合进 **Tinygrad** 后，由于进展渺茫，挫败感倍增。
   - *“我还没成功运行过……所以谁知道呢，”* 这句话总结了这种挣扎。
- **显式循环和内存控制的限制**：尝试中发现的核心问题是 **Tinygrad** 在处理 *显式循环* 和 *内存控制* 方面存在困难，而 **Flash Attention** 严重依赖这些特性。
   - 这让人怀疑该实现是否能在该框架内有效执行。
- **对张量维度表示的拼死尝试**：在一次极具创意的尝试中，有人努力将 Flash Attention 的 **嵌套循环形式** 表示为张量维度，从而可能减少 kernel 的创建。
   - 然而，尽管付出了这些努力，*“我成功搞出了……一个 GPU OOM”* 反映了所面临的挑战。
- **挣扎中的小小胜利**：尽管挑战不断，但通过使用高达 **25GB** 的 GPU 显存计算出类似 **stable diffusion** 的至少一个步骤，取得了一个小小的胜利。
   - *“这是一个小小的胜利，但我们接受它，”* 这句话概括了充满希望的坚持。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1329547329912569906)** (39 条消息🔥): 

> `算子（反）融合 (Operator (Un)Fusion)、Flash Attention 挑战、Tinygrad JIT 优化、添加 FP8 支持、Tinygrad 的 Windows 支持` 


- **算子（反）融合见解**：一位成员分享了关于算子（反）融合的简短笔记，并为感兴趣的人提供了 [GitHub 教程](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20250117_fusion.md)链接。
   - 这可以作为理解 tinygrad 中实现细节和细微差别的资源。
- **实现 Flash Attention 的困难**：成员们讨论了让 **Flash Attention** 运行的挑战，特别提到了对**单内核 (single kernel) softmax** 的需求以及调度器（scheduler）的更改。
   - 一位成员决定先对 Stable Diffusion 进行性能分析 (profile)，而不是在不理解现有代码性能的情况下继续当前的尝试。
- **针对可变 Batch Size 优化 Tinygrad JIT**：一位用户询问了如何在保持速度的同时处理可变 Batch Size 的 JIT，以及是否应该为训练和测试阶段分开 JIT。
   - 建议包括有效地使用 `.realize()` 来管理计算图，并尝试使用填充 (padding) 来保持输入的一致性。
- **FP8 支持的增量开发**：讨论集中在如何在不破坏现有测试的情况下在 tinygrad 中添加 **FP8 (8位浮点数)** 支持，并提出了使用特性标志 (feature flags) 的建议。
   - 建议将识别代码中的破坏性行作为逐步集成这一新特性的策略。
- **Tinygrad 中关于 Windows 支持的困惑**：有人寻求关于 tinygrad 在 Windows 支持方面的澄清，一位成员质疑如果不支持 Windows，测试该如何运行。
   - 创始人确认虽然存在一些小问题，但他们在 Windows 上成功运行过，并建议解决与 **mmap 常量**相关的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20250117_fusion.md">tinygrad-notes/20250117_fusion.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账号为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/extra/models/rnnt.py">tinygrad/extra/models/rnnt.py at master · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/discussions/1697">tinygrad 0.7.0 · tinygrad/tinygrad · Discussion #1697</a>：代码量再次增加，达到 4311 行 :( 但是，这次有很多新功能！自 0.6.0 以来有超过 500 次提交。发布亮点：为了专注于 Linux 和 Mac OS，已停止对 Windows 的支持。一些功能...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1329553550358876201)** (17 条消息🔥): 

> `FORTRAN 复兴、CUDA 替代方案、Triton vs CUDA、复杂损失函数、V JEPA 论文问题` 


- **FORTRAN 意外回归**：*“天哪，我们又绕回了 FORTRAN”* 展示了一种古老编程语言的惊人回归。
   - 这反映了在不断发展的技术格局中调整经典语言的更广泛趋势。
- **寻找用户友好的 CUDA 替代方案**：一段对话强调了对 **CUDA** 复杂性的沮丧，并指出很快 **LLM** 可能会完全自动化编码。
   - 有推测认为，熟练的编译器开发人员将在创建下一代 **LLM** 中发挥关键作用。
- **比较 Triton 和 CUDA 的功能**：Triton 因其 **Python 兼容性**而脱颖而出，使其比基于 C++ 的 **CUDA** 更容易优化，有人认为后者并没有带来真正的益处。
   - 尽管 Triton 有优势，但 *ChatGPT 对它的掌握并不好*——这表明其效果取决于用户案例和提示词 (prompting) 策略。
- **关于复杂损失函数的询问**：一位用户询问了他们遇到过的最**复杂的损失函数**，揭示了对高级 AI 指标的共同好奇心。
   - 这一话题鼓励了对 AI 社区内创新损失函数设计的探索。
- **V JEPA 论文需要澄清**：有人对 **V JEPA 论文**提出了疑虑，特别是关于下游任务中注意力层 (attentive layer) 的功能。
   - 讨论集中在理解与嵌入 (embeddings) 相关的张量解释和 softmax 操作。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1329600729463394334)** (22 messages🔥): 

> `MiniMax 论文讨论, 模型训练结果, Active Inference 见解, 非语言交流线索, Machine Learning 资源` 


- **探索 MiniMax 论文**：成员们讨论了即将对 [MiniMax-01 论文](https://arxiv.org/abs/2501.08313) 进行的审查，该论文统一了 MHA 和 GQA 等各种 Attention 机制。
   - 一位成员认为其中的数学集成部分尚可处理，且作者发布的代码易于理解。
- **模型训练的积极结果**：一位成员分享了他们在 **3090 TI** 上训练一个 100M 参数的 flow matching 模型的心得，并发现其结果在不使用 RoPE 的情况下与 MHA Attention 难分伯仲。
   - 另一位成员对他们在 3090 上持续进行模型训练的高产出表示赞赏。
- **非语言交流见解**：一位用户引用了一篇关于**非语言交流**在人类互动中重要作用的论文，指出它贡献了高达 **60%** 的总交流量。
   - 讨论强调了理解面部表情和手势对于有效沟通的重要性。
- **深入研究 Active Inference**：一位成员推荐了一个 [YouTube 视频](https://www.youtube.com/watch?v=N5H5I6cvcrQ&ab_channel=ActiveInferenceInstitute)，由讨论者 **Karl Friston** 主讲，重点关注 Active Inference 原理。
   - 该视频提出了关于**自由能 (free energy)、时间**和**意识**的见解，有助于对 Active Inference 的基础理解。
- **Machine Learning 资源**：一位成员分享了一个 [GitHub 指南](https://github.com/stas00/ml-engineering/blob/master/debug/make-tiny-models-tokenizers-datasets.md)，用于训练微型模型，这对于硬件能力有限的人非常有用。
   - 该指南为在性能较低的设备上优化模型训练提供了实用策略。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.08313">MiniMax-01: Scaling Foundation Models with Lightning Attention</a>：我们推出了 MiniMax-01 系列，包括 MiniMax-Text-01 和 MiniMax-VL-01，它们可与顶级模型媲美，同时在处理长上下文方面提供卓越的能力。其核心在于...</li><li><a href="https://www.challenge.gov/">Challenge.Gov</a>：Challenge.Gov 是官方的 GSA 政府网站，支持由美国联邦政府赞助的奖金挑战和竞赛。在这里，联邦机构向...提供奖金。</li><li><a href="https://github.com/stas00/ml-engineering/blob/master/debug/make-tiny-models-tokenizers-datasets.md">ml-engineering/debug/make-tiny-models-tokenizers-datasets.md at master · stas00/ml-engineering</a>：Machine Learning 工程公开书。通过在 GitHub 上创建账号为 stas00/ml-engineering 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=N5H5I6cvcrQ&ab_channel=ActiveInferenceInstitute">Karl Friston ~ Active Inference Insights 001 ~ Free Energy, Time, Consciousness</a>：在 Active Inference Insights 的第一集中，Darius Parvizi-Wayne 与 Active Inference 的首席架构师 Karl Friston 坐下来对谈。Fris 教授...</li><li><a href="https://www.nature.com/articles/s41598-023-34932-z">Study on emotion recognition bias in different regional groups - Scientific Reports</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1329751400212729858)** (7 messages): 

> `3090 显存改装, CaPa 论文发布, 网格生成技术` 


- **关于 3090 显存改装 (Memory Mods) 的问题**：一位成员对改装显卡以升级显存的可能性表示好奇，特别是这是否仅适用于 **3090s**。
   - 另一位成员表达了类似的兴趣，提到他们拥有四块 **3090s**，并希望了解更多关于显存改装的信息。
- **CaPa 的高效 4K 纹理网格生成**：关于 **CaPa** 的一篇新论文详细介绍了一种在 **30 秒**内生成 **4K 纹理网格 (textured meshes)** 的方法，旨在应用于游戏和 VR/AR。
   - 虽然被认为是一项显著的进步，但一位成员提到它不是开源的，因此不符合论文推荐的标准。
- **与 Trellis 的比较**：一位成员认为 **CaPa** 在网格生成能力方面似乎优于 **Trellis**。
   - 大家对网格生成 (mesh generation) 技术领域取得的进展普遍感到兴奋。



**提到的链接**：<a href="https://ncsoft.github.io/CaPa/">CaPa: Carve-n-Paint Synthesis for Efficient 4K Textured Mesh Generation</a>：未找到描述

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1329542286199099403)** (20 条消息🔥): 

> `Google Research TITANS, RunwayML 内容审核, 用于 LLM 日志分析的 Master AI agent, API 访问限制, OpenAI token 的隐私担忧` 


- **Google Research 发布 TITANS**：一位成员分享了名为 ["Google Research Unveils 'Transformers 2.0' aka TITANS"](https://www.youtube.com/watch?v=x8jFFhCLDJY) 的 YouTube 视频链接，讨论了一种通过使用两个较小的动态子模型来模仿人类记忆的新模型。
   - 另一位成员指出，虽然这种架构有助于处理更长的序列，但它还不是一个完全的持续学习系统，暗示仍有改进空间。
- **RunwayML 标记审核问题**：一位用户分享了一个幽默的事件，RunwayML 标记了一个包含“**underwear drawer**”（内衣抽屉）短语的视频输入，引发了关于内容审核敏感性的讨论。
   - 评论中提到了 AI 审核政策的讽刺之处，强调了术语在意外语境下可能变得有问题的情况。
- **用于分析 LLM 日志的 Master AI agent**：一位成员提议创建一个“master AI agent”，用于分析各种 LLM 之间的大量对话存档，以识别模式并生成定制的子 Agent。
   - 他们寻求关于其他人类似项目经验的反馈，以深入了解此类实施可能面临的挑战。
- **对 API 访问层级的担忧**：讨论涉及访问高级 API 模型的限制，特别是 **O1** 由于容量限制仅限于高级别用户。
   - 成员们对访问权限的排他性表示沮丧，并建议采用候补名单可能是管理需求的更公平方式。
- **OpenAI token 的隐私问题**：一位用户感叹他们为了换取 **每天 100 万个免费 O1 token** 而放弃了隐私，结果却发现自己不是 Tier 5，因此无法获得这些 token。
   - 另一位成员对这种情况进行了幽默的评论，表示庆幸自己不必放弃隐私，同时也承认了这一担忧的严重性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=x8jFFhCLDJY">Google Research Unveils &quot;Transformers 2.0&quot; aka TITANS</a>: 我们是否终于破解了赋予模型“类人”记忆的密码？观看视频一探究竟！加入我的 Newsletter 以获取定期 AI 更新 👇🏼https://forwardfu...</li><li><a href="https://www.youtube.com/watch?v=pU5Zmv4aq2U">Google Reveals SURPRISING New AI Feature &quot;TITANS&quot;</a>: 最新的 AI 新闻。了解 LLM、生成式 AI，并为 AGI 的推出做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 领域的最新动态。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1329576892004827166)** (7 条消息): 

> `Mind Journal 功能, DALL·E 集成问题, 版本历史记录不准确` 


- **Mind Journal 恢复正常运行**：经过排查，**lazypeon.zzz** 发现 GPT Editor 中的 DALL·E 复选框未勾选，导致了该问题。
   - 勾选后，一切恢复正常，这凸显了一个类似于“你重启电脑了吗”的简单修复方法。
- **疑似导致复选框重置的故障**：成员 **solbus** 提到一位用户分享了他们的 GPT 中 DALL·E 复选框被取消勾选的情况，促使大家检查设置。
   - 确认重置引发了关于系统中什么原因可能导致此故障的疑问。
- **版本历史记录包含 'INVALID DATE'**：**lazypeon.zzz** 注意到版本历史记录中的许多旧版本被标记为 'INVALID DATE'，表明存在潜在错误。
   - 这种不准确性可能会困扰那些希望跟踪版本历史更改的用户。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1329541557254230099)** (6 messages): 

> `Prompt Engineering Book, ChatGPT Jailbreaks, Community Moderation` 


- **30 天掌握 Prompt Engineering**：一位成员询问是否可以在 **30 天**内学会 **prompt engineering** 并写一本书，并参考 OpenAI 文档作为指导。
   - 另一位成员确认这是可能的，建议使用**自我探索技术**来引导 AI，并分享了一个[资源链接](https://chatgpt.com/share/67897fc3-6580-8000-af35-d693f933acfb)以扩展知识。
- **关于 ChatGPT Jailbreaks 的担忧**：一位成员表达了对 **ChatGPT jailbreaks** 的渴望，引发了关于此类讨论是否恰当的回应。
   - 一则回复指出，虽然可以在不触发过滤器的情况下探索敏感话题，但由于该社区**管理级别较高**，建议在此保持谨慎。
- **遵守社区准则**：成员们强调关于 jailbreaks 的讨论不适合在这个平台进行，并建议保持谨慎。
   - 他们鼓励在讨论敏感话题时保持**真实性**，并提到意料之外的询问可能会导致管理干预。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1329541557254230099)** (6 messages): 

> `Prompt Engineering, ChatGPT Jailbreaks, AI Moderation` 


- **我能在 30 天内学会 Prompt Engineering 吗？**：一位成员询问是否可能在 **30 天**内学会 **prompt engineering** 并写一本书，建议使用 [OpenAI documentation](https://openai.com/docs/)。
   - 另一位成员确认这是可行的，并推荐了有效提示的自我探索技术。
- **社区对 ChatGPT Jailbreaks 的立场**：一位成员表达了对 **ChatGPT jailbreaks** 的向往，但其他人指出这里不是请求或分享此类尝试的合适场所。
   - 随后展开了关于话题敏感性的讨论，强调这里的审核明显比 ChatGPT 更加严格。
- **探讨敏感话题**：一位用户提到，在讨论敏感话题时保持**谨慎和真实**通常有助于避免触发审核过滤器。
   - 然而，考虑到社区的高标准审核，他们警告不要公开讨论潜在的有问题的主题。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1329611939047018578)** (39 messages🔥): 

> `Molmo Vision Model Errors, Llama Model Loading Issues, Model Performance on Mac, MiniMax-01 Benchmarks, Image Handling Bugs` 


- **Molmo Vision 模型的 `trust_remote_code` 问题**：用户在运行 **Molmo** vision 模型时遇到错误，该模型需要设置 `trust_remote_code=True` 选项，而 LM Studio 并不支持此选项。
   - 另一位成员确认 LM Studio 不支持需要此设置的 **MLX models**。
- **Llama 3.2 Vision 问题**：用户报告了加载 **Llama 3.2** vision 模型时的问题，错误提示为未知架构，且与现有的 LM Studio 版本不兼容。
   - 澄清指出 **3.2 Vision models** 与 Windows/Linux 版的 LM Studio 不兼容，仅能在 Mac 的 MLX 上运行。
- **Mac 系统上的性能瓶颈**：用户反映在 16GB RAM 的 Mac 上运行 **Phi-4** 性能缓慢，仅为 **0.05 tokens/sec**，这可能归因于资源限制。
   - 另一位用户注意到初始性能异常，但表示在处理前几个 token 后速度大幅提升。
- **MiniMax-01 模型表现平平**：一位用户评估了 **MiniMax-01** 模型，将其与 **WizardLM-2** 进行对比，并指出其在各项任务中的表现并不理想。
   - 他们报告了不遵守格式的小问题，特别是在 **中文输出** 方面，认为该模型整体表现平庸。
- **Vision 模型的图像加载问题**：一位遇到 vision 模型问题的用户指出，当加载新图像时，响应仍会引用第一张图片。
   - 他们建议通过重新加载或清除聊天作为临时解决方案，并表示这可能是不同模型间的通用问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/NKnRPykTIJs?si=G8CLk2pf3ID7vH43">MiniMax-01: This OPENSOURCE Model HAS LONGEST 4M CONTEXT &amp; BEATS OTHERS!</a>：访问 OnDemand 并免费领取 $150 额度：https://app.on-demand.io/auth/signup?refCode=AICODEKING。在本视频中，我将向你介绍 MiniMax...</li><li><a href="https://lmstudio.ai/docs/advanced/sideload">Sideload models - Advanced | LM Studio Docs</a>：使用在 LM Studio 之外下载的模型文件
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1329545958782013441)** (10 条消息🔥): 

> `用户介绍、学生项目、账单支持` 


- **用户介绍应更具影响力**：一位成员建议新用户不仅要说简单的 'hi'，还应分享他们的动机和潜在的 AI 项目，以进行更具吸引力的自我介绍。
   - 另一位成员强化了这一观点，表示这种方法会带来更有意义的对话。
- **关于 Gen AI 的大学毕业设计**：一位成员分享了他们对大学毕业设计规划的想法，表示有兴趣从事与 **Generative AI** 相关的研究。
   - 这为社区正在进行的讨论带来了学术视角，特别是在探索 AI 应用的学生群体中。
- **分享账单支持指南**：一位成员提供了有关账单查询的信息，引导用户联系支持团队：[support@cohere.com](mailto:support@cohere.com)。
   - 这有助于简化账单问题的沟通，并确保用户获得适当的协助。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1329589845072547963)** (7 条消息): 

> `用于重排序的 Prompt engineering、聊天历史相关性、用例探索、Vercel 托管咨询` 


- **用于重排序的聊天历史提示词**：一位用户询问了将用户与助手之间的聊天历史消息传递到 rerank 提示词中的最佳方式，并质疑了这样做的正确时间顺序。
   - 另一位用户指出，虽然不需要显式的 Prompt engineering，但**更多细节**会带来**更好的语义相关性**，并强调了结果索引的重要性。
- **探索重排序的用例**：一位成员表示有兴趣了解用户的 reranker 使用场景，询问他们是否在从历史记录中寻找相关的聊天消息。
   - 这种对学习的开放态度凸显了在小组内提高理解和功能的协作方法。
- **Vercel 托管查询**：一位成员回应了关于 Vercel 的无关查询，澄清 Vercel 是一项托管服务，并建议用户直接与他们沟通。
   - 该声明指出该查询可能与当前的 Cohere 讨论无关，从而将焦点保持在相关主题上。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1329901741185044491)** (1 条消息): 

> `Command R 模型成本对比、默认模型时间戳混淆、8-2024 版本的问题` 


- **Command R 模型的成本对比**：成员们质疑带有 **8-2024** 时间戳的 Command R 模型在 API 上的成本是否与早期版本相同。
   - 这引发了关于价格透明度和新模型更新**价值**的讨论。
- **默认模型未指向较新的时间戳**：有成员注意到默认模型 **command-r** 并没有引用较新的 **8-2024** 时间戳，这引起了用户的好奇。
   - 成员们推测这可能是一种营销策略，或者是 API 配置中的潜在疏忽。
- **对 8-2024 模型问题的担忧**：有人提问是否有人注意到 Command R 模型的 **8-2024** 版本发布后存在问题。
   - 参与者表示，随着用户开始采用该模型，应密切监控性能反馈。


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1329546941838921739)** (11 条消息🔥): 

> `深度学习免费平台、深度学习初学者资源、Cohere LLM University、Cohere Cookbooks、Cohere 的 AWS Cloud 平台` 


- **Cohere 推荐的深度学习免费平台**：Cohere 提供了多种学习深度学习的免费资源，包括由专家领导课程的 [LLM University](https://docs.cohere.com/v1/docs/the-cohere-platform) 和提供实践教程的 [Cookbooks](https://docs.cohere.com/v1/page/cookbooks)。
   - 新用户在前三个月可以享受 **$75 的免费额度**，且仅需按需付费。
- **Cohere 提供的初学者友好资源**：对于初学者，Cohere 的 [LLM University](https://docs.cohere.com/v1/docs/the-cohere-platform) 提供专家课程，以及包含“Hello World! Meet Language AI”等快速入门指南的 [Cookbooks](https://docs.cohere.com/v1/page/cookbooks)。
   - 此外，用户可以通过 [AWS Cloud](https://docs.cohere.com/v1/docs/cohere-on-aws) 访问 Cohere 的语言模型，以获得全托管体验。


  

---

### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1329805004525142122)** (1 条消息): 

> `Community Guidelines, Mod Role` 


- **社区在 Mod 的指导下蓬勃发展**：一名成员对另一名成员的笔记表示感谢，并强调 **Mod 有效地管理着这里**。
   - 遵循 **Mod 指南** 对于成员成功融入社区至关重要。
- **提醒 Mod 的重要性**：对话强调了 Mod 在维持秩序方面的作用，重申他们对社区运作至关重要。
   - 鼓励成员遵守既定规则，以获得更好的体验。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1329589830547542098)** (4 条消息): 

> `Modular GitHub Repository Migration, Community Package Showcase, Adding Mojo/MAX Projects to Awesome for Beginners, Concerns about Mojo Language Stability, Request for Mojo-Specific Projects List` 


- **Modular GitHub 仓库已迁移**：Modular 的公共 GitHub 仓库已从 [ModularML](https://github.com/modularml) 迁移到新的 [Modular](https://github.com/modular) 组织，并已设置自动重定向。
   - 鼓励用户报告与此次迁移相关的任何意外问题。
- **展示你的 MAX 和 Mojo 项目！**：Modular 网站即将推出一个新页面，通过 Magic 展示社区贡献的包，并邀请向 Magic 社区频道提交项目。
   - 感兴趣的贡献者可以在这里找到 [提交说明](https://www.modular.com/community/package-submission)，并且必须提交一个 Pull Request 来添加 rattler-build 配方。
- **建议在 GitHub 上包含 Mojo/MAX 项目**：一名成员提议将 Mojo 和 MAX 项目添加到拥有 7 万星的仓库 [awesome-for-beginners](https://github.com/MunGell/awesome-for-beginners)，以吸引更多新贡献者加入社区。
   - 这将有助于在 Modular 生态系统中推广初学者友好的项目。
- **与 Mojo 快速变化相关的风险**：有人对 Mojo 语言变化的快速节奏表示担忧，认为由于不稳定，这可能会阻碍潜在的新用户。
   - 该成员强调了在进行更广泛的广告宣传和社区参与之前，稳定语言特性的重要性。
- **呼吁建立 Mojo 专用项目列表**：提到了需要一份专门的初学者友好 Mojo 项目列表，重点是标准库中新手可以参与的任务。
   - 给出的例子包括哈希表、BTree 映射和 CSV 解析，这些可能对寻求贡献的人有吸引力。



**提及的链接**：<a href="https://github.com/MunGell/awesome-for-beginners">GitHub - MunGell/awesome-for-beginners: A list of awesome beginners-friendly projects.</a>：一个优秀的初学者友好项目列表。通过在 GitHub 上创建账户为 MunGell/awesome-for-beginners 的开发做出贡献。

  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1329614332744564787)** (18 条消息🔥): 

> `Mojo Parallelization Constraints, yyjson Data Structures, Type System and Language Improvements, Using Variant for Sum Type Support, Quantum Country Resource Feedback` 


- **Mojo 与 Python 交互时的 `parallelize` 限制**：一位用户报告称，在与 Python 交互时，如果 `num_work_items` 和 `num_workers` 同时超过 1，Mojo 中的 `parallelize` 会在运行时失败，但在纯 Mojo 代码中运行正常。
   - 提供的示例表明，这种行为特别发生在利用 `Foo` 类的结构体的 `start` 函数中。
- **用于高效 JSON 处理的 yyjson**：一位用户讨论了对 [yyjson](https://ibireme.github.io/yyjson/) 库的研究，强调了其在 JSON 文档中对不可变和可变数据结构的使用。
   - 用户注意到 yyjson 处理大型文档的方法，并考虑了 zero-copy 设计在处理 JSON 数据时的效率。
- **规划未来的语言改进**：关于 Mojo 类型系统和可能影响 API 设计的语言特性的潜在改进展开了讨论，强调了周密计划的重要性。
   - 有人担心由于语言及其标准库即将发生的变更，需要避免对大型代码库进行重构。
- **使用 Variant 作为 Sum Type 的替代方案**：一位用户分享了他们在 Mojo 中使用 `Variant` 的看法，认为它是未来 Sum Type 支持的占位符。
   - 这反映了在语言仍在演进过程中，人们对于深入进行优化存在普遍的犹豫。
- **对 Quantum Country 资源的反馈**：一位用户对被推荐 [quantum.country](https://quantum.country) 表示感谢，称其为现象级的资源。
   - 他们还分享了最初因缺乏电子阅读器友好版本而感到的沮丧，但后来理解了这是有意为之。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ibireme.github.io/yyjson/doc/doxygen/html/md_doc__data_structure.html">yyjson: Data Structures</a>：未找到描述</li><li><a href="https://docs.rs/yoke/latest/yoke/">yoke - Rust</a>：未找到描述</li><li><a href="https://ibireme.github.io/yyjson">Introduction</a>：C 语言中最快的 JSON 库
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1329872186672152628)** (1 条消息): 

> `MAX's final form, Composable components, .NET platform comparison` 


- **关于 MAX 最终形态的思考**：一位成员在思考 **MAX 的最终形态** 是否类似于微软的 .NET 平台，即作为一组**可组合组件的套件**。
   - 他们推测 **C#** 或 **Mojo** 可能处于这种潜在架构的核心。
- **关于架构相似性的讨论**：该对比引发了关于 **微软 .NET 平台** 结构与 **MAX 架构** 之间关系的疑问，并邀请进行技术分析。
   - 讨论强调了可组合性在现代技术栈中的价值。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1329583852317835357)** (22 条消息🔥): 

> `SWEBench 验证，AI 融资轮次，Agent Recipes，网络安全行政命令，OpenAI 的 webRTC API` 


- **SWEBench 验证达成**：我们的 CTO 宣布，他们基于 o1 的 AI 编程 Agent 在 SWEBench 上达到了 **64.6%** 的 **SOTA (state-of-the-art)** 性能，标志着这是已知首个完全由 o1 驱动的 Agent。
   - 目前正在准备正式的验证提交，这表明 AI 编程工具的性能指标取得了长足进步。
- **Anysphere 获得 1.05 亿美元 B 轮融资**：Anysphere 已筹集 **1.05 亿美元** B 轮融资，以推进其代码自动化的使命，Thrive Capital 和 Andreessen Horowitz 提供了重要支持。
   - 该资金旨在加强他们在代码自动化方面的研究，作为数百万程序员首选的编辑器提供服务。
- **Agent Recipes 发布**：一个名为 **Agent Recipes** 的新网站已上线，为开发者提供 Agent 工作流的代码示例，可以轻松集成到他们的 AI 应用程序中。
   - 该计划旨在成为学习编程中 Agent 实现策略的首选资源。
- **拜登的网络安全指令**：总统乔·拜登发布了一项全面的网络安全行政命令，旨在加强政府对 AI 的使用，并防御外国网络威胁。
   - 该指令概述了加强数字基础设施和为美国公民实施新身份识别措施的策略。
- **关于 OpenAI webRTC API 的讨论**：一位用户对实现 OpenAI 的 **webRTC 实时 API** 的挑战表示担忧，指出该 API 目前主要由 OpenAI 员工展示。
   - 社区呼吁提供支持，包括请求共享代码库或解决方案，以缓解实现过程中的困难。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.cursor.com/blog/series-b">B 轮融资与代码自动化 | Cursor - AI 代码编辑器</a>：我们筹集了 1.05 亿美元以进一步实现代码自动化的使命。</li><li><a href="https://www.answer.ai/posts/2025-01-08-devin.html">使用 Devin 一个月后的感想 – Answer.AI</a>：我们在给 Devin 分配了 20 多个任务后的印象。</li><li><a href="https://www.forbes.com/sites/robtoews/2024/12/22/10-ai-predictions-for-2025/">2025 年 10 大 AI 预测</a>：2025 年将是人工智能领域的重大年份。</li><li><a href="https://www.wired.com/story/biden-executive-order-cybersecurity-ai-and-more/">拜登发布内容丰富的新行政命令，应对网络安全、AI 等问题</a>：美国总统乔·拜登刚刚发布了一份长达 40 页的行政命令，旨在加强联邦网络安全保护，指导政府使用 AI，并对微软的主导地位进行了抨击。</li><li><a href="https://x.com/shawnup/status/1880004026957500434">Shawn Lewis (@shawnup) 的推文</a>：我基于 o1 的 AI 编程 Agent 现在在 SWE-Bench Verified 上达到了 SOTA！它解决了 64.6% 的问题。这是我们知道的首个完全由 o1 驱动的 Agent。我们在构建它的过程中学到了很多。</li><li><a href="https://x.com/pitdesi/status/1879982274831347890?s=46">Sheel Mohnot (@pitdesi) 的推文</a>：为律师事务所提供 AI 服务的 Harvey 正在从 Sequoia 筹集另一轮资金（30 亿美元估值，筹集 3 亿美元）。上一轮是 7 月的 C 轮，由 GV 领投，15 亿美元估值，筹集 1 亿美元。当时估计他们的收入为 3000 万美元，不知道现在...</li><li><a href="https://x.com/nutlope/status/1879587920744788172?s=46">Hassan (@nutlope) 的推文</a>：宣布 Agent Recipes！一个学习 Agent/工作流配方的网站，包含你可以轻松复制并粘贴到自己 AI 应用中的代码示例。我将把它打造成开发者学习的首选资源...</li><li><a href="https://share.snipd.com/episode/4361fc13-7775-4afd-acc0-65560f27ea1e">2025 年你必须了解的 AI 工程知识 | Chip Huyen，《AI Engineering》作者</a>：2025 年你必须了解的 AI 工程知识 | Chip Huyen，《AI Engineering》作者
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1329562335081271386)** (2 messages): 

> `Women in AI RAG Hackathon，结合 LlamaIndex 和 Memgraph 的 GraphRAG` 


- **Women in AI RAG Hackathon 邀请**：邀请技术领域的女性参加在 Palo Alto 举办的 [Women in AI RAG Hackathon](https://t.co/2Bzg80dh29)，重点关注使用开源向量数据库 @zilliz_universe 的 **检索增强生成 (RAG)**。
   - 参与者将有机会在这次全天活动中与其他女性技术专家和导师建立联系。
- **GraphRAG 网络研讨会见解**：最近的研讨会介绍了 @memgraphdb 和 LlamaIndex 如何协作构建代理式图应用，重点是 **GraphRAG**，以增强生成式 AI 工作流中的上下文检索 [点击观看](https://t.co/a4SMTY5pC3)。
   - 讨论了通过代理式方法改进 **RAG 流水线** 的关键策略，丰富了 AI 应用的工具包 [更多信息](https://t.co/PaK8dt1m9y)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1329696243118506055)** (18 messages🔥): 

> `缓存增强生成 (CAG)，结合 OpenAI 的 Azure AI，Embedding 模型配置，RAG 领域影响研究，跨 LLM 追踪 Prompt` 


- **关于缓存增强生成 (CAG) 的讨论**：成员们讨论了在 Gemini 和 LlamaIndex 中实现 **缓存增强生成 (CAG)**，强调这可能需要直接的模型访问权限，例如 PyTorch。
   - 分享了一个展示实现的示例：[GitHub - CAG](https://github.com/hhhuang/CAG/blob/main/kvcache.py)。
- **Azure AI 集成挑战**：一位成员在将 Azure AI 与 OpenAI 结合使用时遇到问题，指出其集成尝试仍将请求导向 OpenAI 而非 Azure 服务。
   - 另一位成员建议可能需要随 LLM 一起配置 **Embedding 模型** 来解决此问题。
- **分块过程中的元数据处理**：关于分块 (Chunking) 在 Embedding 过程中如何处理节点元数据的咨询，确认 **nodegetcontent.metadata.embed** 确实是相关部分。
   - 澄清了用户可以修改 `excluded_llm_metadata_keys` 和 `excluded_embed_metadata_keys` 来控制在此过程中包含哪些元数据。
- **文档改进需求**：一位成员对缺乏关于更改 Azure AI 所用模型的清晰文档表示沮丧，指出了潜在的改进领域。
   - 他们希望在示例页面中提供更明确的指导，以更好地促进模型选择过程。
- **关于跨 LLM 追踪 Prompt 工具的咨询**：一位成员寻求可以跨多个开源 LLM 追踪和比较 Prompt 的包或工具推荐。
   - 讨论中未提供具体的解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/hhhuang/CAG/blob/main/kvcache.py">CAG/kvcache.py at main · hhhuang/CAG</a>: 缓存增强生成：一种简单、高效的 RAG 替代方案 - hhhuang/CAG</li><li><a href="https://chat.whatsapp.com/JcXJDtmi0gL9kWP4K1cIiU">Ai - ML - qb</a>: WhatsApp 群组邀请</li><li><a href="https://chat.whatsapp.com/JN9pUV3uMydHzxT52HryF8">Quantum-qb</a>: WhatsApp 群组邀请
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1329590014018850951)** (5 messages): 

> `DSPy v3 发布，Stable Diffusion 优化，思维链 (Chain-of-thought) 风格迭代` 


- **DSPy v3 发布时间线**：第一季度不会发布 **DSPy v3**，因为它代表了较大规模的变更，在此之前计划先发布早期版本。
   - 随着关于后续发布准备工作的讨论继续，v3 的具体时间仍不确定。
- **Stable Diffusion 新项目**：一个新项目旨在通过“思维链”迭代风格来 **优化 Stable Diffusion Prompt**，展示了一种基于 **DSPy** 构建的新颖方法。
   - [Thorondor LLC 的推文](https://x.com/thorondorllc/status/1880048546382221313?s=46) 重点介绍了这一举措，分享了对该创新策略的兴奋之情。



**提到的链接**: <a href="https://x.com/thorondorllc/status/1880048546382221313?s=46">Thorondor LLC (@ThorondorLLC) 的推文</a>: 新项目！通过“思维链”风格的迭代优化你的 Stable Diffusion Prompt - 基于 DSPy 构建

  

---

### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1329551564049879100)** (3 messages): 

> `dspy ReAct usage, addition function error, LLama model issues` 


- **dspy ReAct 在加法函数上的错误**：一位用户遇到了一个错误，提示 *tool addition is not designed to calculate sum of two numbers*（工具 addition 并非设计用于计算两个数字之和），并需要一些未知的额外参数。
   - 他们提到使用了通过 LM-Studio 托管的 LLama 模型，并寻求社区帮助以解决此问题。
- **错误信息的澄清**：另一位成员询问了完整的错误信息以便诊断问题，怀疑在上下文中重新定义 `addition` 可能会覆盖原始函数。
   - 这表明代码中可能存在冲突，从而阻止了程序的正常执行。


  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1329575261573877882)** (5 messages): 

> `ChatML, Llama3, Torchtune, ShareGPT Dataset, Migration from ShareGPT` 


- **ChatML 与 Llama3 的讨论**：一位成员询问是使用 **ChatML** 还是 **Llama3**，这暗示了关于最佳模型使用的持续争论。
   - 另一位成员以随意的“duh”回应，表示对该讨论非常熟悉。
- **ShareGPT 数据集澄清**：一位成员询问使用 **ShareGPT** 数据集是否存在任何问题，暗示了潜在的担忧。
   - 另一位成员澄清说没有问题，并强调存在可以轻松映射键（keys）的配置。
- **从 ShareGPT 迁移的说明**：讨论强调了目前已有关于从 **ShareGPT** 迁移的解释，可以在文档中找到。
   - 这表明官方正在努力确保用户在不同数据集之间切换时的平滑过渡。
- **Torchtune 的调整**：一位成员提到 **Torchtune** 目前需要进行重大修改。
   - 这一说法暗示了该工具功能需求的变化可能会影响用户的实现。


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1329901315546943488)** (1 messages): 

> `Screenshot Analysis, Image Insights` 


- **关于截图分析的讨论**：一位成员分享了一张截图进行分析，但未提供有关其内容的任何上下文或细节。
   - 随后没有关于所展示图像的进一步评论或见解。
- **缺乏对图像见解的参与**：尽管分享了截图，但成员们并未参与讨论或提出澄清性问题。
   - 图像发布后的沉默表明可能错失了社区分析的机会。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1329768939713990688)** (1 messages): 

> `` 


- **用户对发现时机的好奇**：一位用户对某人如何发现特定功能表示好奇，想知道他们是之前就知晓并刚开始测试，还是根本没有尝试过。
   - *对发现过程的好奇可以引发关于用户参与和探索的有趣讨论。*
- **探索中的测试障碍**：一位用户发起了关于在测试之前未尝试过的功能时可能遇到的障碍的讨论。
   - *了解这些延迟或犹豫有助于澄清用户体验并提高功能的可访问性。*


  

---


---


---


---


---


---


---


{% else %}


> 由于邮件篇幅限制，各频道的详细分析已截断。
> 
> 如果您想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}