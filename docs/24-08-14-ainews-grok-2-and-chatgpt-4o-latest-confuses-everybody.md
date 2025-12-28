---
companies:
- openai
- x-ai
- black-forest-labs
- google-deepmind
date: '2024-08-15T00:51:40.557390Z'
description: '**OpenAI** 在 ChatGPT 中低调发布了全新的 **GPT-4o** 模型（与 API 版本有所不同），并在 Lmsys
  Arena 竞技场基准测试中重新夺回了数学、编程和指令遵循等多个类别的冠军宝座。


  与此同时，**X.ai** 推出了 **Grok 2**，其性能超越了 **Claude 3.5 Sonnet** 和此前的 GPT-4o 版本，并计划发布企业级
  API。Grok 2 集成了 **Black Forest Labs** 的 **Flux.1**，这是一款性能超越 **Stable Diffusion 3**
  的开源文生图模型。


  **Google DeepMind** 宣布推出 **Gemini Advanced**，增强了对话功能并实现了与 Pixel 设备的集成。


  AI 研究员 **Yann LeCun (ylecun)** 强调了大型语言模型（LLM）在学习和创造力方面的局限性；而 **rohanpaul_ai** 则讨论了一个名为“AI
  科学家”（AI Scientist）的系统，该系统能以极低的成本生成可发表的机器学习研究成果。此外，**Andrej Karpathy (karpathy)**
  警告称，LLM 的分词器（Tokenizer）存在类似于 SQL 注入的安全风险。'
id: 5479b511-511b-4adc-855c-752df22743c3
models:
- gpt-4o
- grok-2
- claude-3.5-sonnet
- flux-1
- stable-diffusion-3
- gemini-advanced
original_slug: ainews-grok-2-and-chatgpt-4o-latest-confuses
people:
- ylecun
- rohanpaul_ai
- karpathy
title: Grok 2! 和 ChatGPT-4o-latest 把大家都搞糊涂了。
topics:
- benchmarking
- model-performance
- tokenization
- security-vulnerabilities
- multi-agent-systems
- research-automation
- text-to-image
- conversational-ai
- model-integration
---

<!-- buttondown-editor-mode: plaintext -->**一天之内发布两个前沿模型？！**

> AI News (2024/08/13-2024/08/14)。我们为您查看了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**253** 个频道，**2414** 条消息）。预计节省阅读时间（以 200wpm 计算）：**294 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

比较容易讨论的进展是上周在 [ChatGPT 中悄悄发布](https://x.com/ChatGPTapp/status/1823109016223957387)的全新 GPT-4o 模型的确认。需要明确的是，这与上周在 API 中发布的**另一个** GPT-4o 模型**不同**（即[我们之前报道过的支持 structured outputs 的那个版本](https://buttondown.com/ainews/archive/ainews-gpt4o-august-100-structured-outputs-for/)）。

 
![image.png](https://assets.buttondown.email/images/7d212faf-0f54-4be1-861a-150b0f32dbd3.png?w=960&fit=max)
 

[几乎没有人对此感到完全满意](https://x.com/teknium1/status/1823379952718565864?s=46) —— 从新的命名结构，到[愈发低调的发布方式](https://podcasters.spotify.com/pod/show/nathaniel-whittemore/episodes/A-New-OpenAI-Model-Coming-e2n4msp)，甚至是[模型性能](https://x.com/paulgauthier/status/1823715711254192611) —— 尽管其性能令人印象深刻，并[从 Gemini 1.5 Pro August 手中夺回了 Lmsys arena 的第一名](https://x.com/lmsysorg/status/1823515224064098546)。

新版 ChatGPT-4o 类别排名：
- 总榜：#1
- 数学：#1-2
- 编程：#1
- 高难度提示词：#1
- 指令遵循：#1
- 长查询：#1
- 多轮对话：#1

相比之下，X.ai 的 Grok 2 的故事线要清晰得多，它于[昨晚太平洋时间晚上 11 点](https://x.com/nearcyan/status/1823601166925889588)发布，并[被证实就是 `sus-column-r`](https://x.com/jimmybajimmyba/status/1823600123487903883)，而不是许多人之前猜测的 Cohere。Grok 2 击败了 Claude 3.5 Sonnet 以及 GPT 4o May 和 Mini：

 
![image.png](https://assets.buttondown.email/images/04ad3509-4558-4ceb-b404-929c69000db6.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/3c13713e-3f00-4e1a-b55e-23c0565d292c.png?w=960&fit=max)
 

虽然 Grok 1（[我们的报道见此](https://buttondown.com/ainews/archive/ainews-grok-1-in-bio/)）的主要特点是其 open weights 性质，但 Grok 2 目前仅面向 X 的高级订阅用户发布，不过[官方博客](https://x.ai/blog/grok-2)提到 Grok-2 和 Grok-2 mini 将于“本月晚些时候”在 X 的新企业级 API 平台上发布。

X 平台上的 Grok 2 还集成了 **Black Forest Labs 的 Flux.1 模型**（[我们的报道见此](https://buttondown.email/ainews/archive/ainews-rombach-et-al-flux1-prodevschnell-31m-seed/)），该模型相对较少审查，并且已经在[开源文本生成图像社区](https://x.com/swyx/status/1823400729429868915)中取代了 Stable Diffusion 3（与此同时，Google 的 Imagen 3 随着[新论文的发布](https://arxiv.org/abs/2408.07009)也正趋于更加开放）。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 生成，从 4 次运行中择优。

**AI 模型更新与能力**

- **Gemini Advanced**：Google DeepMind 发布了 Gemini Live，这是一种与 Gemini 进行更自然对话的新方式。功能包括头脑风暴、插话提问以及暂停/恢复对话。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1823409674739437915) 强调了其在搭载 Google Tensor G4 芯片的 Pixel 设备上的集成。

- **LLM 局限性**：[@ylecun](https://twitter.com/ylecun/status/1823313599252533594) 强调，LLM 无法回答其训练数据之外的问题或解决相关问题，无法在没有人类帮助的情况下获得新技能，也无法发明新事物。他认为，仅靠扩大 LLM 规模并不会产生具备这些能力的系统。

- **AI Scientist**：讨论了一篇关于 AI Scientist 系统的论文，该系统能够生成研究构思、进行实验并撰写机器学习论文。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1823381840704475277) 指出，它可以产出超过顶级 ML 会议录用门槛的论文，且每篇论文的成本低于 15 美元。

- **模型性能**：[@OfirPress](https://twitter.com/OfirPress/status/1823439223615066578) 提到，OpenAI 发布了 SWE-bench 任务的一个子集，经人工验证是可解的，可以被视为 "SWE-bench Easy"。

**AI 开发与工具**

- **Tokenization 问题**：[@karpathy](https://twitter.com/karpathy/status/1823418177197646104) 警告称，由于输入字符串中特殊 Token 的解析问题，LLM 的 Tokenizers 可能存在类似于 SQL 注入攻击的安全漏洞。

- **多 Agent 系统**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1823501094775087450) 展示了一个复杂多 Agent 系统的简洁实现，证明了事件驱动架构和可定制性的优势。

- **Prompt Engineering**：[@dzhng](https://twitter.com/dzhng/status/1823428375962407231) 分享了使用 LLM 生成结构化输出的技巧，强调了 Schema 中属性顺序的重要性，并建议添加 "reason" 字段以提高性能。

- **RAG 改进**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1823375703380746730) 讨论了 EyeLevel 的 GroundX，这是一种 RAG 的新方法，将文档处理为语义对象，从而保留上下文信息并提高检索准确性。

**行业与研究趋势**

- **NoSQL 辩论**：[@svpino](https://twitter.com/svpino/status/1823419273580298700) 引发了关于 NoSQL 数据库现状和相关性的讨论。

- **AI Alignment**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1823435255132643505) 对 AI Alignment 工作可能为政府审查 AI 系统提供掩护表示担忧。

- **开源模型**：[@bindureddy](https://twitter.com/bindureddy/status/1823459188980470197) 提到了开源 LLM 在编程能力方面即将迎来的改进，暗示将有新版本发布。

- **AI 研究论文**：AI Scientist 系统生成研究论文的能力引发了关于学术出版未来以及 AI 在科学发现中作用的讨论。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 新开源 LLM 发布：InternLM2.5**

- **我们已在 HuggingFace 上发布了 InternLM2.5 的 1.8B 和 20B 新模型。** ([Score: 63, Comments: 20](https://reddit.com//r/LocalLLaMA/comments/1er4z52/we_have_released_our_internlm25_new_models_in_18b/))：**InternLM2.5** 已在 **HuggingFace** 上发布了 **1.8B** 和 **20B** 尺寸的新模型。**1.8B** 模型被描述为超轻量级且高度可适配，而 **20B** 模型功能更强大，适用于复杂任务。这些模型可在 [HuggingFace](https://huggingface.co/collections/internlm/internlm25-66853f32717072d17581bc13) 上获取，项目详情可见 [GitHub](https://github.com/InternLM/InternLM)。
  - **InternLM2.5** 模型具有 **1M token 上下文窗口**，但用户反映在使用 **Xtuner**、**axolotl** 和 **swift** 等**微调工具**时遇到挑战，正在寻求关于有效微调方法的建议。
  - 用户在使用 **LMDeploy** 将 InternLM 模型部署为 API 时遇到了问题，报告了乱码输出或无响应，并就正确的实现方式提出了疑问。
  - 与最初的担忧相反，模型支持 **llama.cpp**，这一点已通过 **1.8B** 和 **20B** 版本的 HuggingFace 模型页面链接得到确认。

**主题 2. 具有桌面控制能力的先进 AI Agents**

- **给 Llama 一个它自己的 Windows 实例** ([Score: 52, Comments: 24](https://reddit.com//r/LocalLLaMA/comments/1eromb9/giving_llama_its_own_windows_instance/))：**LLaMA**（一种大型语言模型）被赋予了对 **Windows 实例**的控制权，并可以访问各种 API，包括**屏幕截图、鼠标和键盘控制以及终端使用**。该 AI 模型自命名为 **"Patrick"**，出生日期为 **1975 年 4 月 4 日**，它被释放出来利用这些能力去实现一系列目标，展示了其像人类用户一样与计算机系统交互的能力。
  - 用户对该项目的**开源可用性**和方法论表示关注，要求上传到 **GitHub**，并对 **AI 的任务执行过程**感到好奇。开发者承诺会分享图片并可能将项目开源。
  - 讨论集中在 AI 系统的**技术层面**，包括如何为文本模型理解而**解码屏幕截图**，以及使用 **Vision Language Models (VLMs)** 进行图像文本处理。文中分享了 [Hugging Face 的 VLM 博客](https://huggingface.co/blog/vlms)链接以供参考。
  - 评论者幽默地推测 AI 可能会发现并沉迷于《上古卷轴5：天际》（**Skyrim**）或《英雄联盟》（**League of Legends**）等游戏，并提到了赋予 AI 系统自主权和计算机访问权限可能带来的意外后果。

**主题 3. Grok 2.0 Mini 在 LMSYS Arena 中表现惊人**

- **[LMSYS 上的 sus-column-r 就是 Grok]** ([Score: 140, Comments: 112](https://reddit.com//r/LocalLLaMA/comments/1ertpa3/suscolumnr_on_lmsys_is_grok/))：**Grok 2.0 Mini** 已被确认为 **LMSYS Arena** 排行榜上 **"sus-column-r"** 条目背后的模型。这一发现表明，xAI 的最新模型在各种基准测试和任务中正与其他领先的 AI 系统展开激烈竞争。在 LMSYS Arena 上识别出 Grok 2.0 Mini，为将其与其他主流 AI 模型在功能和性能方面进行直接比较提供了机会。
  - **Grok 2.0 Mini** 被证实为 **LMSYS Arena** 上的 "sus-column-r" 模型，其表现可与顶级 AI 模型媲美。用户对 **AI 军备竞赛**以及一年内可能出现的令人印象深刻的进展感到兴奋。
  - 该模型的表现引发了褒贬不一的反应，一些人称赞其能力，而另一些人则保持怀疑。**Elon Musk** 通过 [Twitter](https://x.com/elonmusk/status/1823593475205685588) 确认了 Grok 2.0 Mini 的身份，用户注意到其**无审查（uncensored）**特性，并将其与 **Command R+** 进行比较。
  - 讨论围绕模型大小展开，推测 "mini" 可能仍意味着高达 **170B** 的参数。用户争论是否会发布权重，**Musk** 表示新模型发布与**开源**之间有 **5 个月的时间差**。

## AI Reddit 全面回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 模型发布与改进**

- **Google Gemini 语音对话模式发布**：[The Verge 报道](https://www.reddit.com/r/singularity/comments/1erdv06/wtfff_google_geminis_voice_chat_mode_is_here/) Google 已为 Gemini 推出了实时语音对话模式，标志着 AI 与人类交互的重大进步。

- **Agent Q：自愈式 Web Agent**：[MultiOn AI 宣布](https://www.reddit.com/r/singularity/comments/1erf7s3/agent_q_breakthrough_ai_research_in_selfhealing/)在 AI 研究方面取得突破，推出 Agent Q，其具备针对 Web Agent 的规划和自愈能力。

- **FLUX 在 24GB GPU 上实现全量微调**：[Stable Diffusion 社区报告](https://www.reddit.com/r/StableDiffusion/comments/1erj8a1/flux_full_fine_tuning_achieved_with_24gb_gpu/)了 FLUX 模型微调的重大进展，现在仅需 24GB GPU 即可实现，该功能可能很快会加入 Kohya。

**AI 发展与行业新闻**

- **OpenAI 延迟推出的语音模式**：[一篇文章指出](https://www.reddit.com/r/singularity/comments/1er2bma/on_this_day_3_months_ago_openai_promised_us_the/)，OpenAI 承诺的 ChatGPT 语音模式在发布三个月后仍未兑现。

  - 一位用户声称正处于 Alpha 测试阶段，这表明缓慢的推送正在进行中。
  - 另一位用户提到取消了 ChatGPT 订阅并转向 Poe，理由是其他模型的表现更好。

- **AI 炒作与虚假信息**：多篇帖子讨论了 [“草莓”事件](https://www.reddit.com/r/singularity/comments/1eridn6/the_strawberry_guy_is_the_best_thing_thats/)，一名 Twitter 用户发布了关于 AI 进展的虚假声明，引发了社区内关于 AI 炒作和虚假信息的讨论。

**社区管理**

- **禁止虚假信息**：[r/singularity 版主宣布](https://www.reddit.com/r/singularity/comments/1ermhhl/we_banned_mentions/)封禁特定用户名及相关的 Twitter 链接，以打击虚假信息和恶意挑衅。

- **社区对虚假声明的回应**：多篇帖子呼吁[封禁那些传播 AI 进展虚假信息的用户](https://www.reddit.com/r/singularity/comments/1erecce/ban_strawberry_guy_he_claimed_this_and_it_didnt/)。


---

# AI Discord 摘要

> 由 GPT4O (gpt-4o-2024-05-13) 生成的摘要之摘要的摘要

**1. LLM 模型进展**

- **Hermes 2.5 超越 Hermes 2**：**[Hermes 2.5](https://link.to.examples)** 在添加代码指令示例后，在各项基准测试中表现优于 Hermes 2，其 MMLU 基准测试得分为 **52.3**，而 Hermes 2 为 **34.5**。
  - 这一改进突显了代码指令示例对模型性能的显著影响，为基准测试对比设定了新标准。
- **X 发布 Grok-2 Beta 版**：来自 X 的新 AI 模型 **[Grok-2](https://x.ai/blog/grok-2)** 声称具备最先进的推理能力，显著推动了该领域的发展。
  - 该模型的发布预计将对行业产生重大影响，展示了 X 对创新 AI 开发的承诺。


**2. Prompt Engineering 技术**

- **批判性思维技术汇编**：一位成员正在编写一个综合性的 Prompt，融合了苏格拉底教学法 (Socratic Method)、布鲁姆分类法 (Bloom's Taxonomy) 和科学方法 (Scientific Method) 等技术。
  - 目标是创建能够鼓励批判性思维的 Prompt，整合了如 TRIZ、演绎推理和 SWOT 分析等方法。
- **解决 OpenAI 响应不一致问题**：一位用户通过要求提供完整的命令列表提高了 Prompt 的清晰度，使响应准确率达到了 100%。
  - 这突显了 Prompt Engineering 中清晰输出格式的重要性，减少了模型行为的不一致性。


**3. API 性能与优化**

- **Anthropic 的 Prompt Caching**：**[Anthropic](https://x.com/alexalbert__/status/1823751966893465630)** 推出了 Prompt Caching，可将 API 输入成本降低高达 90%，延迟降低高达 80%。
  - 该功能可能会彻底改变 API 效率，对于寻求高性价比解决方案的开发者来说，这是一个极具吸引力的选择。
- **Perplexity API HTML 格式化**：用户寻求从 Perplexity API 获得一致的 HTML 格式响应，并正在尝试使用 System Prompt 和 `markdown2` 模块。
  - 这种方法可能会平衡响应质量和 HTML 格式化，增强 API 输出的可用性。


**4. 开源 AI 工具**

- **LlamaIndex Box Reader 集成**：**[LlamaIndex](https://llamahub.ai/l/readers/llama-index-readers-file?from=)** 现在提供 Box Readers，可将 Box 文档集成到 LLM 工作流中，并提供四种数据提取方法。
  - 这些 Reader 通过 CCG 或 JWT 进行身份验证，并允许在 LLM 中加载、搜索和检索 Box 文件及元数据。
- **RealtimeSTT 与 Faster-Whisper 集成**：**[OpenInterpreter](https://github.com/KoljaB/RealtimeSTT)** 现在使用 RealtimeSTT 和 **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)** 进行实时语音转文本，提供实时性能。
  - 这一集成增强了 OpenInterpreter 的可用性，特别是在性能较低的设备上。


**5. 模型部署与集成**

- **Mojo 基准测试与性能**：一位成员质疑为什么 **[Mojo benchmarks](https://modular.com/mojo/)** 只与 C 进行对比，建议增加与 Go 和 Rust 的对比。
  - 讨论强调了为了更广泛的发布，需要一个具有 RHEL 8 最低内核要求的静态链接构建。
- **在外部硬盘运行 LM Studio**：用户可以通过重新定位目录或使用符号链接，从外部硬盘运行 **[LM Studio](https://huggingface.co/kshabana/GOAT-llama3.1-v0.1)**。
  - 这种灵活性解决了空间限制问题，使得管理大型模型文件变得更加容易。

## GPT4OMini (gpt-4o-mini-2024-07-18)


**1. Grok-2 与模型性能**

- **Grok-2 占据领先地位**：由 x.ai 发布的 **Grok-2** 在 **LMSYS leaderboard** 上超越了 **Claude 3.5 Sonnet** 和 **GPT-4-Turbo**，展示了其在对话、编程和推理方面的先进能力。
  - 该模型此前被称为 **sus-column-r**，目前处于 Beta 测试阶段，很快将通过 **x.ai's enterprise API** 提供。
- **AgentQ 声称获胜**：来自 **Infer** 的新模型 **AgentQ** 声称其性能优于 **Llama 3 70B BASE** 达 **340%**，尽管它并未与 **Claude 3** 等更新的模型进行对比。
  - 这一大胆的声明引发了关于其潜在影响以及缺乏相关能力证明文档的讨论。


**2. 量化技术与模型合并策略**

- **HQQ+ 增强量化模型**：**HQQ+** 允许在量化模型上微调额外的 LoRa 适配器层，显著提高了 **Llama2-7B** 等模型的准确性。
  - 该技术在 **1-bit** 和 **2-bit** 量化模型中均显示出卓越的效果，引发了关于其在各种项目中实现的讨论。
- **Mistral 与模型合并策略**：成员们讨论了 **Mistral** 面临的挑战，特别是其在没有持续预训练的情况下无法扩展超过 **8k** 上下文的限制，这是一个已知问题。
  - 提出了关于合并策略的建议，包括应用 **UltraChat** 与基础 **Mistral** 之间的差异来提升性能。


**3. 开源工具与社区贡献**

- **LlamaIndex Box Reader 集成**：**LlamaIndex** 现在集成了 **Box Readers**，允许将 Box 文档无缝整合到 LLM 工作流中，并提供多种数据提取方法。
  - 鼓励社区成员为这一集成做出贡献，增强 LlamaIndex 在文档处理方面的功能。
- **OpenEmpathic 项目寻求贡献者**：**Open Empathic** 项目正在寻找贡献者以扩展其类别，特别是在涉及用户生成内容的低端类别。
  - 分享了一个关于如何贡献的教程视频，指导用户从 YouTube 贡献他们喜欢的电影场景。


**4. AI 模型的局限性与改进**

- **Vision 的性能问题**：用户对 **Vision** 无法准确检测简单任务（例如主体是向左看还是向右看）表示沮丧，突显了其局限性。
  - 分享了 Vision 未能正确识别的畸形图像示例，引发了对其在关键应用中可靠性的担忧。
- **通过 Prompt Engineering 提高一致性**：一位用户通过优化其 Prompt 以请求完整的命令列表，成功提高了 OpenAI 响应的一致性，实现了 100% 的准确率。
  - 这强调了清晰且具体的 Prompt 在最大化模型性能方面的重要性。


**5. AI 安全与伦理考量**

- **用于 AI 安全测试的 RedOps 平台**：**RedOps** 平台已被开发用于通过模拟真实攻击来评估 chatbots 和 voicebots 的安全性，从而发现漏洞。
  - 这一举措强调了针对 AI 系统中的对抗性输入和社会工程学采取强大安全措施的必要性。
- **AI 版权讨论趋势**：关于 AI 版权的讨论表明，持续的论述可能会导向寡头垄断，特别是在即将举行的 **ACL2024NLP** 等会议的背景下。
  - 这一评论反映了人们对伦理实践和 AI 治理未来格局日益增长的关注。

---

# PART 1: High level Discord summaries

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Hermes 2.5 超越 Hermes 2**：在添加了[代码指令示例](https://link.to.examples)后，**Hermes 2.5** 在各项基准测试中的表现似乎优于 **Hermes 2**。
   - Hermes 2 在 MMLU 基准测试中得分为 **34.5**，而 Hermes 2.5 得分为 **52.3**。
- **Mistral 的 8k 限制**：成员们表示，如果不进行持续预训练（continued pretraining），**Mistral** 无法扩展到 8k 以上，且[这是一个已知问题](https://link.to.issue)。
   - 他们指出，性能的下一个前沿在于 *mergekit* 和 *frankenMoE finetuning* 的进一步工作。
- **模型合并策略引发讨论**：一位成员建议将 **UltraChat** 和基础 **Mistral** 之间的差异应用到 **Mistral-Yarn** 上，作为一种潜在的合并策略。
   - 其他人表示怀疑，但该成员保持乐观，并引用了过去在他们所谓的“诅咒模型合并”（cursed model merging）方面的成功尝试。
- **Open Empathic 寻求贡献者**：一位成员呼吁帮助扩展 **Open Empathic** 项目的类别，特别是低端类别。
   - 他们分享了一个关于 [Open Empathic 发布与教程的 YouTube 视频](https://youtu.be/GZqYr8_Q7DE)，指导用户从 YouTube 视频中贡献自己喜欢的电影场景，以及 [OpenEmpathic 项目本身](https://dct.openempathic.ai/)的链接。
- **HQQ+ 助力量化模型**：**HQQ+** (**High Quantization Quality Plus**) 允许在量化模型上微调额外的 LoRa 适配器层，以提高其准确性和能力。
   - 该技术在 1-bit 和 2-bit 量化模型中都显示出显著改进，特别是对于像 **Llama2-7B** 这样的小型模型。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **X 发布 Grok-2 Beta 版**：X 宣布发布 **Grok-2** 测试版，这是一款具有最先进推理能力的新 AI 模型。
   - 该模型是 AI 推理领域的一个重大进步，可能会对行业产生重大影响。
- **在 ComfyUI 中使用 FLUX AI 放大图像**：一个 YouTube 视频演示了如何在 **ComfyUI** 界面中使用 **FLUX AI** 放大图像。
- **开源大语言模型入门**：2024 年 7 月的一次演讲对 **开源大语言模型（open source large language models）** 进行了易懂的介绍。
   - 演讲涵盖了 AI 工作原理的基础知识，以及开源模型如何改变 AI 开发的格局。
- **语义分块被高估了**：X 上的一位用户认为语义分块（semantic chunking）被高估了，强大的 regex 可以在不需要复杂语言模型的情况下准确分割文本。
   - 他们声称其 50 行、2490 个字符的 regex 在 regex 的限制范围内已经尽可能强大，并且比语义分块更快、更具成本效益。
- **Jina AI 的免费 Tokenizer API**：Jina AI 提供了一个免费的 API 来对文本进行 Tokenize 并将长文本分割成块（chunks）。
   - 该 API 利用结构线索和启发式方法，确保将文本准确分割成有意义的块，即使是对于 Markdown、HTML 和 LaTeX 等复杂的文本格式。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **AMD GPU 安装指南**：一位成员询问在 AMD GPU 上运行 Stable Diffusion 的情况，并获知 GitHub 上有针对 NVIDIA 和 AMD 显卡的安装指南。
   - 指南提供了在不同硬件配置上设置 Stable Diffusion 的详细说明。
- **多图生成的 ControlNet 链式调用**：一位用户寻求在单次生成中使用多个 ControlNet 的帮助。
   - 几位成员建议在 ComfyUI 中将它们链接起来，或者使用一个可以输入多个 ControlNet 的节点。
- **ComfyUI vs InvokeAI：速度与控制**：一位成员表示相比 Automatic1111 (InvokeAI)，他们更喜欢 ComfyUI，理由是 ComfyUI 提供了更高的控制力和速度。
   - 他们强调了 ComfyUI 直观的界面和强大的功能，使其成为用户的热门选择。
- **SD3 vs Flux：优缺点**：一位新用户询问了 SD3 与 Flux 相比的优缺点，指出 SD3 仍处于开发阶段，缺乏完整的功能。
   - 另一方面，Flux 也有自己的怪癖和局限性，选择取决于个人需求和偏好。
- **SDXL vs SD 1.5：新特性与差异**：一位成员要求澄清什么是 SDXL 1.0 以及它与 SD 1.5 有何不同。
   - 对话可能围绕 SDXL 的新特性和功能展开，例如改进的图像质量和更大的模型尺寸。

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **AgentQ 是一款游戏规则改变者**：来自 **Infer** 的新模型 **AgentQ** 声称其表现优于 **Llama 3 70B BASE** 340%，但并未将其与 **3.1, 405b, Mistral Large 2 或 Claude 3** 等较新模型进行对比。
   - 尽管未考虑 **OpenRouter revenue**，**Infer** 已在此处发布了关于 **AgentQ** 的研究论文 [此处](https://multion-research.s3.us-east-2.amazonaws.com/AgentQ.pdf)。
- **ChatGPT-4o-Latest 只是一个新的别名**：**ChatGPT-4o-Latest** 只是 **gpt-4o-2024-08-06** 的新名称，该模型已存在于 **OpenRouter** 上。
   - 然而，关于该模型针对 **ChatGPT** 的优化以及缺乏适当文档的问题，困惑依然存在。
- **Grok 2 攀升至排行榜第三名**：**xAI** 模型的早期版本 **Grok 2** 已占据 **LMSys Arena leaderboard** 的第 3 位。
   - 它在 **Coding**、**Hard Prompts** 和 **Math** 方面表现出色，甚至在排行榜上与 **GPT-4o** 持平。
- **Anthropic 的 Prompt Caching：重新定义效率**：**Anthropic** 为其 **Claude 3 models** 引入了 **prompt caching** 功能。
   - 该功能可为长提示词降低高达 90% 的成本和 85% 的延迟，并有可能集成到 **OpenRouter** 中。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 基准测试：对比的疑问**：一位成员质疑为什么 [Mojo benchmarks](https://modular.com/mojo/) 仅与 C 语言对比，并询问是否可以与 Go 和 Rust 进行基准测试。
   - 另一位成员建议目前将 Magic CLI 视为这些问题的解决方案，并且具有 RHEL 8 最低内核要求的静态链接构建可能比 RPM 更好，因为它可以允许在其他发行版上进行打包。
- **Mojo 的多线程困境**：Mojo 目前为了性能采用单线程，但除了使用 MAX 启动并行内核外，还没有良好的多线程 API。
   - 一位成员询问 Mojo 是否有支持多进程或改进网络处理的计划，因为网络性能对他们的工作至关重要。
- **Mojo vs Go/Rust：网络速度与性能**：一位成员询问 Mojo 在网络速度方面是否比 Go 更快，以及它是否能像 Rust 一样处理繁重任务。
- **Mojo 的 MAX 平台：揭开神秘面纱**：一位成员质疑 MAX 的本质，不确定它是一个平台、模块还是其他东西。
   - 另一位成员解释说 MAX 是一个平台，Mojo 是其组件之一，它还包括 GPU、graph API 和其他组件。
- **Mojo RPM 构建：寻求流畅的 RHEL 体验**：一位成员询问 Mojo .rpm 构建的预计发布时间 (ETA)，表达了在没有 containerd 的 RHEL 机器上运行 Mojo 的愿望。
   - 他们承认这可能是一个合适的首步。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini Advanced：实时对话功能在哪？**：一位购买了 **Gemini Advanced** 的用户在应用中找不到实时对话选项。
   - 其他用户推测其处于 Alpha 测试阶段，将像之前 OpenAI 模型发布一样逐步扩大推广范围。
- **Vision 的性能挣扎**：一位用户对 **Vision** 在检测主体是向左看还是向右看时的糟糕表现感到惊讶。
   - 他们甚至提供了一张高度变形的图像，但 **Vision** 声称其完全正常，这突显了在识别图像变形方面的局限性。
- **针对批判性思维的 Prompt Engineering**：一位成员正在构建一个综合提示词，结合了 Socratic Method、Bloom's Taxonomy 和 Scientific Method 等批判性思维方法。
   - 他们还整合了 TRIZ、演绎和归纳推理、Fermi estimation 等，旨在创建一个鼓励批判性思维的提示词。
- **GPTs 和 Web Search：Web Browser GPT**：一位成员询问了 **webbrowsergpt**，这是一个专门为网页搜索设计的 GPT，可在 "Explore GPTs" 部分访问。
   - 这个 GPT 可以提供比手动指示通用 GPT 搜索网页更好的搜索结果。
- **自定义 GPT 训练问题**：一位用户报告称，他们的自定义 GPT 模型没有记住训练期间指定的所有规则和单词。
   - 其他用户推测这可能是由于超过了 context token limit，或者是故意的模型限制，但目前尚不清楚。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 面临用户对其网站性能的投诉**：用户在 Perplexity 上遇到了严重的延迟和响应缓慢，尤其是 Perplexity Pro 用户。
   - 这些问题甚至让一些用户质疑 Perplexity 是否能取代 Sonnet 或 Opus 作为他们的默认搜索引擎，凸显了为了维持用户满意度而迅速解决问题的必要性。
- **Perplexity Pro 饱受延迟困扰**：付费的 Perplexity Pro 用户报告了响应时间缓慢，甚至服务完全停滞的情况。
   - 这些投诉强调了用户对付费产品提供更可靠、响应更迅速服务的期望，给 Perplexity 带来了及时解决这些问题的压力。
- **Perplexity 网站更新引发褒贬不一的反应**：Perplexity 团队成员确认正在努力修复 Bug 和问题，包括一个影响网站切换开关（toggles）的 Bug。
   - 然而，许多用户仍对性能问题感到沮丧，并要求回滚到之前的版本，这凸显了在开发过程中进行全面测试和收集用户反馈的必要性。
- **Perplexity 支持团队应对工作量增加**：Perplexity 的支持团队正面临关于近期网站更新的用户反馈和报告激增。
   - 用户对团队的工作量表示担忧，并敦促他们优先解决问题，同时也认识到维持支持团队健康的工作与生活平衡的重要性。
- **API 用户寻求 HTML 格式的响应**：Perplexity API 用户正在寻求一致的 HTML 格式响应，并尝试通过各种 System Prompts 来实现。
   - 有建议提出利用 `markdown2` 模块进行 HTML 转换，从而消除对 Prompt Engineering 的需求并确保一致的 HTML 输出。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemma-2 表现出色，但缺少 System Prompt**：**Gemma-2** 模型在其尺寸级别是一个很好的选择，但缺少一个关键组件：System Prompt。
   - 这使得它容易偏离用户指令，尽管在其他方面它的表现相当不错。
- **Android 版 LM Studio 尚不可用**：有用户询问是否有可以连接到 **LM Studio server** 的 **Android app**。
   - 目前还没有这样的应用。
- **LM Studio 可以从外部驱动器运行**：用户询问是否可以从**外接硬盘**运行 **LM Studio**。
   - 建议用户可以迁移 **LM Studio directory**，或使用**符号链接（symbolic link）**连接到另一个驱动器，甚至是网络驱动器。
- **LM Studio 是桌面应用程序，而非 Headless 模式**：用户想知道运行 **LM Studio** 是否需要 **GUI OS**。
   - 虽然 LM Studio 是桌面应用，但用户可以设置 **VNC server** 或寻找变通方法使其在 **Ubuntu** 上运行，尽管它并非为 **headless** 使用而设计。
- **LM Studio 可以加载多模态模型**：用户询问 **LM Studio** 是否可以托管**多模态 LLM server**。
   - 虽然 LM Studio 不能生成图像，但它可以加载能够处理图像数据的模型，从而有效地使其成为多模态 LLM。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Pliny 要求 MultiOn System Prompt**：知名 AI 研究员 Pliny 威胁称，如果 DivGarg9 不在 15 分钟内给出答复，他将在 GitHub 上泄露完整的 MultiOn System Prompt。
   - 此举源于 Twitter 上关于各种 AI 模型能力及其在特定 Benchmark 上表现的持续争论。
- **AnswerAI ColBERT：小模型，大成果**：AnswerAI 发布了一个小巧但强大的 ColBERT 模型版本，名为 answerai-colbert-small-v1，在 BEIR Benchmark 中甚至击败了 bge-base。
   - 这证明了较小模型在完成特定任务时实现高性能的有效性，可能提供更具成本效益的解决方案。
- **Gemini Live 演示遭吐槽**：Swyxio 在 YouTube 上批评了 Google 的 Gemini Live 演示，认为其“令人尴尬 (cringe)”。
   - 随后引发了关于 Gemini 潜力的讨论，一些人强调其增强语音助手的能力，而另一些人则保持怀疑。
- **GPT-4o 在 Chatbot Arena 中表现优于 Gemini**：OpenAI 最新的 GPT-4o 模型在 Chatbot Arena 中进行了测试，整体表现已超越 Google 的 Gemini-1.5-Pro-Exp。
   - 新的 GPT-4o 模型在技术领域表现出显著进步，特别是在 Coding、Instruction-following 和 Hard Prompts 方面，巩固了其在排行榜首位的地位。
- **Grok 2 亮相，能力惊人**：xAI 发布了 Grok-2 的早期预览版，这是对其前代模型 Grok-1.5 的重大突破，展示了在 Chat、Coding 和 Reasoning 方面的能力。
   - Grok-2 已在 LMSYS 排行榜上进行了测试，表现优于 Claude 3.5 Sonnet 和 GPT-4-Turbo，尽管目前尚未通过 API 提供。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Grok-2 成为新王者**：**Grok-2**，来自 x.ai 的新模型，已在 𝕏 开启 Beta 测试，在 LMSYS 排行榜上超过了 **Claude 3.5 Sonnet** 和 **GPT-4-Turbo**。
   - 这个新模型目前可供 Beta 测试，随着它在 AI 领域获得关注，其后续表现值得期待。
- **Anthropic API：更便宜、更快速，但仍有待完善**：Anthropic 的 API 引入了 Prompt Caching，将 API 输入成本降低了高达 **90%**，延迟降低了高达 **80%**。
   - 虽然这一进步值得赞赏，但该 API 仍面临挑战，包括 API 响应慢以及缺乏 Projects & Artifacts API。
- **Anthropic 从尴尬到尖端的华丽转身**：Anthropic 已从一个不太受欢迎的组织转型为被视为该领域领导者的组织。
   - 他们对创新和 Prompt Caching 的承诺为他们在 AI 社区赢得了新的尊重。
- **GPT-4o 获得优化**：OpenAI 改进了 **GPT-4o** 模型，发布为 `gpt-4o-latest`，并表示在进行长期研究的同时，将继续迭代现有模型。
   - 这个新模型现在可以通过 ChatGPT API 获取，定价目前尚未公布。
- **AI 版权讨论与寡头垄断**：一位用户分享了 @asayeed 的推文链接，该推文认为 AI 版权讨论正走向寡头垄断。
   - 这一观察是在 #ACL2024NLP 的背景下提出的，暗示这可能是即将举行的会议上的热门讨论话题。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Box Reader 集成**：LlamaIndex 现在提供 Box Readers，可将 Box 文档无缝集成到您的 LLM 工作流中。
   - 这些 Reader 提供四种数据提取方法，通过 CCG 或 JWT 进行身份验证，并允许您在 LLM 中加载、搜索和检索 Box 文件及其元数据。
- **使用 Relik 构建知识图谱**：Relik 是一个用于快速、轻量级信息提取模型的框架，它简化了知识图谱的构建，无需昂贵的 LLM。
   - 了解如何设置实体提取流水线并使用 Relik 创建知识图谱。
- **使用 Azure AI Search 构建鲁棒的 RAG 系统**：LlamaIndex Workflows 现在可以与 Azure AI Search 和 Azure OpenAI 集成，以构建鲁棒的检索增强生成 (RAG) 系统。
   - 了解如何为 Azure AI Search 实现自定义数据连接器，并使用 LlamaIndex Workflows 创建强大的 RAG 系统。
- **不一致的 OpenAI 响应 - 针对一致性的 Prompt Engineering**：一位用户遇到了 OpenAI Prompt 结果不一致的问题，模型有时即使可以回答问题也会给出否定答案。
   - 该用户通过要求提供完整的命令列表成功改进了 Prompt，实现了 100% 的准确率，凸显了清晰的输出格式对 LLM 的重要性。
- **`astream_chat()` 函数中的 LlamaIndex Agent 和 Tool Calls**：一位用户寻求关于在 LlamaIndex Agent 中处理 Tool Calls 的指导，特别是在使用 `astream_chat()` 函数时。
   - 讨论得出结论，Tool Calls 应首先在 LLM 响应的 `message.tool_calls` 字段中发送，以确保在 Agent 中正确处理 Tool Calls。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Grok 2 发布，开启招聘热潮**：**xAI** 正式发布了 **Grok 2**，这是一款此前在 **LMSYS chatbot arena** 上被称为 **sus-column-r** 和 **column-r** 的新语言模型。
   - xAI 正在为其后训练团队积极招聘，强调了他们构建**有用且真实的 AI 系统**的愿望。
- **Cohere Toolkit 安装困难**：一位用户在尝试为其本地安装的 **Cohere Toolkit** 添加 OpenAI 的自定义部署时遇到了困难。
   - 尽管遵循了概述的步骤，自定义部署仍未出现在 UI (localhost:4000) 或 Postgres 容器数据库的 'deployment' 表中。
- **Rerank API 故障排除**：一位用户尝试利用 Cohere 文档中的 Rerank 概览文档，但遇到了错误 "unknown field: parameter model is not a valid field"。
   - 他们尝试重启内核并抑制警告，但未能解决该问题。
- **企业搜索聊天机器人开发**：一位用户正在构建一个“企业搜索聊天机器人”应用程序，以访问存储在 Confluence 中的公司数据。
   - 他们是 **Fellowship.ai** 最新一批学员的一员，将此项目用于研究和学习。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 用户希望获得更多支持**：一位成员对 LangChain 论坛中基础问题（特别是与 LangGraph 和 LangSmith 相关的问题）缺乏及时支持表示担忧。
   - 他们还指出，许多通用支持问题被发布在 LangChain Discord 服务器上，而其他论坛中的相关请求却无人问津。
- **LangSmith Plus：是否可以访问 LangGraph Cloud？**：一位成员询问 LangSmith Plus 用户是否将获得 LangGraph Cloud 的访问权限。
   - 未提供答案。
- **LangChain Postgres 库与缓存详解**：一位成员询问关于将 `langchain_postgres` 库与 `set_llm_cache`（一种缓存 LLM 结果的方法）结合使用的问题。
   - 他们被告知虽然没有 `langchain_postgres` 库，但可以使用 `langchain_community.cache` 模块中的 `SQLAlchemyCache` 类将 LLM 结果缓存到 PostgreSQL 数据库中。
- **Rubik's AI 提供 2 个月免费高级版**：Rubik's AI 是一个提供 GPT-4o、Claude-3 Opus 和 Mistral Large 等模型的平台，目前正提供 2 个月的免费高级访问权限。
   - 用户可以在 [signup.php](signup.php) 使用促销代码 **RUBIX** 领取此优惠。
- **RedOps 平台应对 AI 安全担忧**：一个团队开发了 **RedOps**，这是一个旨在通过故意尝试破坏聊天机器人和语音机器人来评估其安全性的平台。
   - 该倡议强调了 AI 模型容易受到对抗性输入和社交工程操纵的脆弱性，强调了对鲁棒安全措施的迫切需求。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 需要更小的模型来进行 PPO 全量微调 (Full Finetune)**：一位成员尝试在 `NF4+compile` 下使用 `lora_dpo_single_device` recipe，并建议优先考虑其他 recipe，例如 `ppo_full_finetune_single_device`。
   - 他们请求一个能放入 16GB GPU 的更小模型，并建议将 Qwen2-1.5B 作为该 recipe 的合适选项。
- **Torchtune 的 CPU Offload 优化器和 Torchao 依赖**：一位成员询问 Torchtune 如何处理 Torchao 版本依赖，因为 CPU offload 优化器已包含在 Torchao main 分支中，并计划在下个版本发布。
   - 他们提议将 CPU offload 代码的副本合并到 Torchtune 中，并在 Torchao 实现可用时再行调用。
- **使用 TinyLlama 1B 实现高效的 Torchtune PPO 全量微调**：鉴于 Llama2 分类器的可用性，一位成员建议在 PPO 全量微调 recipe 中使用 TinyLlama 1B（或 0.5B）。
   - 他们提供了 GitHub 上 1B 配置的链接，并建议调整 batch sizes 以进行内存优化。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **埃隆·马斯克的 Grok 2 表现卓越**：**Grok 2** 是来自埃隆·马斯克 x.ai 的新语言模型，已发布早期预览版，并因其在聊天、编程和推理方面的“前沿能力”而备受推崇。
   - Grok 2 在 LMSYS 排行榜上被称为 "sus-column-r"，其 Elo 分数高于 **Claude 3.5 Sonnet** 和 **GPT-4-Turbo**。
- **Fineweb-Edu 数据集现已强化 (Fortified)**：可在 Hugging Face 上获取的 **Fineweb-Edu** 数据集已通过删除重复数据和添加 embeddings 进行了强化。
   - 它现在被称为 **Fineweb-Edu-Fortified**，并包含一个 `count` 列，指示文本在数据集中出现的次数。
- **Mistral Large 2 仍在开发中**：一位用户询问 **Mistral Large 2** 是否已经完成训练。
   - 回复指出该模型尚未完成训练。
- **axolotl 模型加载：`load_model` 标志和 `and False` 条件解析**：一位用户询问在加载模型时，为什么 `axolotl/utils/models.py` 文件中使用了 `and False` 条件。
   - 该条件用于确保如果 `load_model` 标志设置为 `False`，则不会加载模型。
- **OpenAI Chat 端点 - 无法继续 Assistant 响应**：一位用户询问是否可以使用官方 OpenAI chat 端点继续完成部分生成的 assistant 响应。
   - 回复指出，虽然他们在继续本地模型响应方面取得了成功，但 OpenAI 的 chat 端点始终会阻止 assistant 响应的继续生成。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Grok-2 表现优于 GPT-4 和 Claude**：x.ai 发布了 **Grok-2**，该模型比 **Grok-1.5** 强大得多，在聊天、编程和推理方面具备前沿能力。
   - **Grok-2** 的早期版本（代号 "sus-column-r"）已在 **LMSYS 排行榜**上进行了测试，目前其表现优于 **Claude 3.5 Sonnet** 和 **GPT-4-Turbo**。
- **Grok-2：公开和企业测试版**：**Grok-2** 和 **Grok-2 mini** 目前在 **𝕏** 上处于测试阶段，并将于本月晚些时候通过 x.ai 的企业级 API 提供。
   - x.ai 很快将发布**多模态理解 (multimodal understanding)** 预览版，作为 **𝕏** 和 API 上 Grok 体验的核心部分。
- **需要开源图像标注 GUI**：一位成员正在寻求推荐，以获取能够快速高效标注图像的优秀开源 GUI。
   - 他们特别感兴趣的是支持单点标注、直线标注和绘制多边形分割掩码 (polygonal segmentation masks) 的 GUI。
- **埃隆·马斯克与开发者许可证**：关于埃隆·马斯克可能使用开发者许可证并挑战权重许可证 (weight licenses) 的讨论。
   - 对话围绕埃隆·马斯克利用开发者许可证来潜在规避权重许可证限制的想法展开。
- **Schnelle 的付费功能**：一位成员提到软件工具 **Schnelle** 的专业功能可能需要付费订阅。
   - 他们还指出 **Schnelle** 的定价结构对于价格敏感型用户可能并不理想。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ConvTranspose2D 支持 3D 数据**：一位用户对如何在 `tinygrad` 中将 `ConvTranspose2D` 用于 3D 数据感到困惑，但它确实有效！
   - 问题在于 `kernel_size` 应该作为一个长度为 3 的元组传递，而不是一个整数，例如 `kernel_size=(3, 3, 3)`。文档应该进行改进以澄清这一点。
- **CLANG=1 和 LAZYCACHE 导致的 Tinygrad 错误**：一位用户报告在 GPU (3070ti) 上运行 Tinygrad 并使用 `CLANG=1` 执行 `Tensor.zeros` 操作时出现 `RuntimeError: wait_result: 10000 ms TIMEOUT!` 错误，但在 `CUDA=1` 下运行正常。
   - 该错误可能与 Tinygrad 中的 LAZYCACHE 功能有关，一位用户建议该功能“容易出错（bug prone）”，并建议将其删除并在 schedule 中进行去重。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter Pip 发布**：昨晚向 pip 推送了新版本的 OpenInterpreter。
   - 下一个重大更新“开发者更新（the developer update）”仍在开发中，包含许多新的实用功能。
- **本地 LLM 非常耗能**：本地 LLM 需要大量的处理能力。
   - 建议在云端运行 LLM，特别是对于使用默认设置的 OpenInterpreter。
- **RealtimeSTT 与 Faster-Whisper 集成**：OpenInterpreter 现在使用 [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)，它依赖于 [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) 进行实时语音转文本。
   - 这种组合为大多数用户提供了实时性能，并且在性能较低的设备上尚未出现问题。
- **Obsidian 插件与万物互转（Anything-to-Anything）**：分享了一个 YouTube 视频，展示了在 Obsidian 中使用 Open Interpreter 进行万物互转。
   - 该视频推广了使用 Open Interpreter Obsidian 插件来控制 Obsidian 库，并展示了其转换各种数据类型的能力。
- **工具使用周二（Tool Use Tuesday）与视频制作**：一位用户提到计划为一个涉及使用 Open Interpreter 和 Obsidian 的竞赛制作视频演示。
   - 他们还提到探索向量搜索并使用 Manim 来可视化有向图（digraphs），表明重点在于提高视频制作技能并利用“工具使用周二”这一主题。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Poe 的 Previews 黑客松**：Poe 正与 @agihouse_org 合作举办 Previews 黑客松，参与者可以竞争创造创新且实用的聊天内生成式 UI 体验。
   - 黑客松面向所有创作者开放，更多信息请访问 [https://app.agihouse.org/events/poe-previews-hackathon-20240817](https://app.agihouse.org/events/poe-previews-hackathon-20240817)。
- **Modal Labs：最佳微调平台**：一位成员认为 [Modal Labs](https://github.com/modal-labs/llm-finetuning) 是微调开源 LLM 的最佳平台。
   - 这表明 Modal 为开发大语言模型的开发者提供了宝贵的工具和资源。
- **图像特征存储加速训练**：为一个 R&D 团队构建了一个简单的特征存储（feature store），用于存储在线预处理期间从图像中提取的特征。
   - 这使每次训练运行的时间减少了 30-60 分钟，为模型开发节省了大量时间。
- **适用于多样化模型的通用特征存储**：该特征存储是通用的，处理图像 ID、提取方法以及指向对象存储中提取特征的指针。
   - 这使其能够容纳从小型到大型的各种模型，确保高效的特征存储和检索。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Mistral 难以扩展超过 8k**：成员们表示，如果不进行持续预训练（continued pretraining），**Mistral** 无法扩展到 8k 以上，[这是一个已知问题](https://link.to.issue)。
   - 他们指出，*mergekit* 和 *frankenMoE finetuning* 的进一步工作是性能的下一个前沿。
- **关于模型合并策略的讨论**：一位成员建议将 **UltraChat** 和基础 **Mistral** 之间的差异应用到 **Mistral-Yarn**，作为一种潜在的合并策略。
   - 其他人表示怀疑，但该成员保持乐观，并引用了过去在他们称之为“诅咒模型合并（cursed model merging）”方面的成功尝试。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **自动化 Jupyter Notebook 探索**：一位成员询问了关于现有库或开源项目的信息，这些项目可以帮助构建一个自动化修改 Jupyter Notebook 的系统。
   - 目标是创建一个 Agentic 流水线，用于交换单元格、生成变体并验证输出，类似于 Devin 项目，但专注于特定的微小任务。
- **Jupyter Notebook 自动化：游戏规则改变者**：提议的系统将以一个正在运行的 Jupyter Notebook 作为输入，并通过交换单元格对其进行修改。
   - 这一自动化过程将生成多个版本，从而能够高效地探索不同的 Notebook 配置，并可能带来更好的结果。



---


**Alignment Lab AI Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---

# 第 2 部分：渠道详细摘要与链接


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1273002100372275331)** (213 条消息🔥🔥): 

> - `Hermes 2`
> - `Mistral struggles`
> - `Model Merging`
> - `Open Empathic`
> - `HQQ+` 


- **Hermes 2.5 性能优于 Hermes 2**：在添加了 [代码指令示例](https://link.to.examples) 后，**Hermes 2.5** 在各种基准测试中的表现似乎优于 **Hermes 2**。
   - Hermes 2 在 MMLU 基准测试中得分为 **34.5**，而 Hermes 2.5 得分为 **52.3**。
- **Mistral 在扩展超过 8k 时遇到困难**：成员表示，如果不进行持续预训练，**Mistral** 无法扩展到 8k 以上，且[这是一个已知问题](https://link.to.issue)。
   - 他们指出，关于 *mergekit* 和 *frankenMoE finetuning* 的进一步工作是性能的下一个前沿。
- **关于模型合并策略的讨论**：一位成员建议将 **UltraChat** 与基础 **Mistral** 之间的差异应用到 **Mistral-Yarn**，作为一种潜在的合并策略。
   - 其他人表示怀疑，但该成员保持乐观，并引用了过去在他们所谓的“诅咒模型合并”中的成功尝试。
- **Open Empathic 项目寻求帮助**：一位成员呼吁帮助扩展 **Open Empathic** 项目的类别，特别是在低端部分。
   - 他们分享了一个关于 [Open Empathic 发布与教程的 YouTube 视频](https://youtu.be/GZqYr8_Q7DE)，指导用户贡献他们喜欢的 YouTube 视频电影场景，以及 [OpenEmpathic 项目本身](https://dct.openempathic.ai/) 的链接。
- **用于量化模型的 HQQ+**：**HQQ+** (**High Quantization Quality Plus**) 允许在量化模型上微调额外的 LoRa 适配器层，以提高其准确性和能力。
   - 这种技术在 1-bit 和 2-bit 量化模型中都显示出显著改进，特别是对于像 **Llama2-7B** 这样的小型模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.ai/blog/grok-2">Grok-2 Beta Release</a>：未找到描述</li><li><a href="https://www.githubstatus.com/">GitHub Status</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/bigcode/the-stack">bigcode/the-stack · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1erv00p/elon_musks_ai_company_releases_grok2/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ertpa3/suscolumnr_on_lmsys_is_grok/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Gryphe/Opus-WritingPrompts">Gryphe/Opus-WritingPrompts · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/code_bagel_hermes-2.5">Replete-AI/code_bagel_hermes-2.5 · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct_8k_context_filtered">Replete-AI/Everything_Instruct_8k_context_filtered · Datasets at Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1273036082078023701)** (6 条消息): 

> - `Discord 上的自我推广`
> - `野外的蚊子` 


- **Discord 自我推广**：一名成员对频道中过度的自我推广表示担忧，提醒他人这违反了服务器规则。
   - 他们对视频表示了赞赏，但请求今后减少自我推广。
- **蚊子：危险的威胁**：另一名成员开玩笑说野外蚊子泛滥，称人必须要么适应它们，要么冒着被叮咬致死的风险。
   - 他们用轻松的语气强调了在某些环境中蚊子的潜在危险。



**提到的链接**：<a href="https://tenor.com/view/%D0%BB%D0%B0%D1%82%D1%8B-armor-gif-25434286">латы Armor GIF - Латы Armor - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1272995028670615573)** (75 条消息🔥🔥): 

> - `Unsloth 推理`
> - `GPU vs CPU`
> - `VRAM`
> - `自定义数据集`
> - `Alpaca-Cleaned 数据集` 


- **Unsloth 在 CPU 上运行困难，需要 VRAM**：一位用户报告了在 CPU 上运行 Unsloth 推理的困难，建议其可能需要 GPU。
- **构建类似 Alpaca-Cleaned 的自定义数据集**：一位用户询问如何创建类似于 Alpaca-Cleaned 的自定义数据集，该数据集通过删除引用互联网数据的指令来解决幻觉问题。
- **Instruct 模型微调**：一位用户分享了他们成功微调 Instruct 模型的经验，提到他们使用了默认的 60 个训练步数，Batch Size 为 8。
- **为 Ollama 保存模型**：一位用户寻求关于保存与 Ollama 兼容模型的指导，特别是旨在创建一个适用于该平台的模型文件。
- **使用 Unsloth Llama 3.1 进行 Aspect-Based Sentiment Analysis**：一位用户寻求关于评估微调后的 Unsloth Llama 3.1 模型的建议，用于处理 Hugging Face 上 Semeval 数据集的 Aspect-Based Sentiment Analysis（基于方面的情感分析）任务。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/sentiment-analysis-python">Getting Started with Sentiment Analysis using Python</a>：未找到描述</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/yahma/alpaca-cleaned">yahma/alpaca-cleaned · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/deepmind/code_contests">deepmind/code_contests · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.token">Loading methods</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1273299243800465582)** (20 messages🔥): 

> - `Model Card Typos` (Model Card 拼写错误)
> - `Model's Capabilities` (模型能力)
> - `Multi-Lingual LLM` (多语言 LLM)
> - `Dataset-tool for RP` (用于 RP 的数据集工具)
> - `TheatreLM-v2.1-Characters` 


- **Model Card 评测**：一名成员指出 Model Card 上存在多处拼写错误，特别是 "generatal knoledge" 和 "arrabici"，暗示开发者的母语可能不是英语。
   - 该成员还建议使用 ChatGPT 重写 Model Card，以获得更专业的观感并更好地吸引英语受众。
- **Goat 模型家族介绍**：一名成员宣布启动 "goat model family"，这是一个基于 Llama 3.1 的全新微调模型系列。
   - 该特定模型因其改进的通用知识以及对包括阿拉伯语和英语在内的多种语言的支持而受到关注。
- **多语言 LLM 的能力**：一位用户对该模型在翻译方面的多语言能力表示怀疑。
   - 另一名成员反驳称，LLM 在多语言任务中通常比其他方法更熟练，且该模型对于需要多语言支持的企业可能非常有用。
- **用于 RP 和合成数据的数据集工具**：一位用户分享了一个专为 Role-Playing (RP) 和合成数据集生成设计的数据集工具链接，能够生成角色卡、世界信息和设定集 (lorebooks)。
   - 该工具提供了一系列功能，包括 'setting'、'setting_summarized'、'character'、'character_summary'、'character_card'、'character_name'、'story_introduction'、'story_outline' 和 'lorebook'。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/kshabana/GOAT-llama3.1-v0.1">kshabana/GOAT-llama3.1-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/G-reen/TheatreLM-v2.1-Characters">G-reen/TheatreLM-v2.1-Characters · Datasets at Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1273295496386969620)** (3 messages): 

> - `Llama 3.1`
> - `Causal Mask` (因果掩码)
> - `Causal Masking` (因果掩码技术)


- **Llama 3.1 不使用因果掩码**：一位用户询问为什么 Llama 3.1 的代码（特别是 llama31.py 文件的第 130 行）没有使用 Causal Masking。
   - 他们想知道为什么这段代码没有包含这项技术。
- **Llama 3.1 不使用因果掩码**：一位用户询问为什么 Llama 3.1 的代码（特别是 llama31.py 文件的第 130 行）没有使用 Causal Masking。
   - 他们想知道为什么这段代码没有包含这项技术。


  

---



### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/)** (1 messages): 

zhukov_80921: https://huggingface.co/datasets/bigcode/the-stack-v2 60TB 的代码数据
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1273210945564311635)** (4 messages): 

> - `Grok-2`
> - `ComfyUI`
> - `Open Source AI` (开源 AI)


- **X 发布 Grok-2 Beta 版**：X 宣布发布 **Grok-2** 的 Beta 测试版，这是一款具有尖端推理能力的新型 AI 模型。
   - 该模型是 AI 推理领域迈出的重要一步，可能会对行业产生重大影响。
- **在 ComfyUI 中使用 FLUX AI 放大图像**：一段 YouTube 视频演示了如何在 **ComfyUI** 界面中使用 **FLUX AI** 放大图像。
- **开源大语言模型简介**：2024 年 7 月的一次演讲对**开源大语言模型**进行了易懂的介绍。
   - 演讲涵盖了 AI 工作原理的基础知识，以及开源模型如何改变 AI 开发的格局。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=arYBzLc3RV0">Elon Musks's Grok-2 Beta Release Announced by X</a>：今天我们来看看 Grok-2 Beta 版的发布。这是他们在推理能力方面的尖端模型。这款新模型 Grok-2 是向前迈出的重要一步...</li><li><a href="https://www.youtube.com/watch?v=2cuFOXLHr4A&feature=youtu.be">Upscale Images with FLUX AI in ComfyUI</a>：在 ComfyUI 中使用 FLUX AI 放大图像</li><li><a href="https://youtu.be/vrO8tZ0hHGk">How AI Really Works - Intro to Open Source Large Language Models</a>：这是我于 2024 年 7 月 27 日在加拿大温哥华公共图书馆所做演讲的录音。它旨在作为 AI 的入门介绍...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1273344353338916885)** (2 messages): 

> - `Semantic Chunking`
> - `Regex Tokenization`
> - `Tokenizer API`
> - `Tiktoken Free Usage` 


- **Semantic Chunking 被高估了**：X 上的一位用户认为 Semantic Chunking 被高估了，强大的 Regex 就可以准确地分割文本，而不需要复杂的语言模型。
   - 他们声称其 50 行、2490 个字符的 Regex 在 Regex 的限制范围内已经尽可能强大，且比 Semantic Chunking 更快、更具成本效益。
- **Jina AI 的免费 Tokenizer API**：Jina AI 提供了一个免费的 API 来对文本进行 Tokenize 并将长文本分割成 Chunk。
   - 该 API 利用结构化线索和启发式方法，确保即使是 Markdown, HTML 和 LaTeX 等复杂的文本格式，也能准确地分割成有意义的 Chunk。
- **免费 Tiktoken 无限制使用**：Jina AI 还提供带速率限制的免费无限制 tiktoken 使用，如果你只使用 Tokenization 和 Chunking，甚至提供免费的 Embedding 生成而不收取费用。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/JinaAI_/status/1823756993108304135">来自 Jina AI (@JinaAI_) 的推文</a>：基于事实。Semantic chunking 被高估了。尤其是当你编写一个超级 Regex，利用所有可能的边界线索和启发式方法来准确分割文本，而不需要复杂的语言模型时……</li><li><a href="https://jina.ai/tokenizer">Tokenizer API</a>：用于对文本进行 Tokenize 并将长文本分割成 Chunk 的免费 API。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1272998722648936512)** (155 messages🔥🔥): 

> - `Dataset Filtering and Scoring Tool`
> - `LMSYS Leaderboard`
> - `Grok-2`
> - `OpenAI ChatGPT-4o`
> - `HQQ+` 


- **发布新的数据集过滤与评分工具**：一位成员宣布发布了一款免费且开源的数据集过滤与评分工具，重点在于通过快速操作和热键来生成样本并编辑现有样本。
   - 该工具允许用户输入其 API Key，生成与当前样本相似的更多样本，并高效地对大型数据集进行评分。
- **LMSYS Leaderboard 是场骗局？**：一位成员指责 LMSYS Leaderboard 是场骗局，理由是使用了虚构的模型字符串和操纵的分数。
   - 其他人也同意该排行榜不可靠，并指出用户很容易操纵结果，且模型的表现可能无法指示真实世界的实际能力。
- **Grok-2：领域新秀**：一位成员分享了一篇关于 Grok-2 发布的博客文章，这是来自 X.ai 的新语言模型，声称在 LMSYS Leaderboard 上超越了 Claude 3.5 Sonnet 和 GPT-4-Turbo。
   - Grok-2 目前在 X 上开启 Beta 测试，并将于本月晚些时候通过企业级 API 提供。它具有改进的对话、代码和推理能力，同时还提供了一个较小的兄弟模型 Grok-2 Mini。
- **OpenAI ChatGPT-4o 的改进**：最新的 OpenAI ChatGPT-4o (20240808) API 已完成测试并发布，展示了在技术领域（尤其是代码）的显著改进。
   - 它在 LMSYS Leaderboard 上超越了 Google 的 Gemini-1.5-Pro-Exp，以 1314 的分数重新夺回第一宝座。
- **量化模型：HQQ+ 提升准确度**：一位成员讨论了在低位宽下量化较小预训练模型的挑战，强调了 HQQ+ 在恢复准确度方面的有效性。
   - HQQ+ 涉及在量化模型上训练额外的 LoRA 层，展示了在输出质量上的显著提升，特别是在 1-bit 和 2-bit 量化方面。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://ide.x.ai/">PromptIde</a>：未找到描述</li><li><a href="https://x.com/lmsysorg/status/1823515224064098546">来自 lmsys.org (@lmsysorg) 的推文</a>：来自 Chatbot Arena 的激动人心的更新！最新的 @OpenAI ChatGPT-4o (20240808) API 在过去一周以“anonymous-chatbot”身份进行了测试，获得了超过 11,000 张社区投票。OpenAI 现在已经……</li><li><a href="https://x.ai/blog/grok-2">Grok-2 Beta 版发布</a>：未找到描述</li><li><a href="https://tenor.com/view/hellfire-gif-10103277782914351064">Hellfire GIF - Hellfire - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/TobyPhln/status/1823598808309158353">来自 Toby Pohlen (@TobyPhln) 的推文</a>：Grok-2 mini 现已在 X 上推出。Grok-2（大模型）将很快推出。https://x.ai/blog/grok-2
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1273001297687347250)** (22 messages🔥): 

> - `FP8 training`
> - `Nemotron`
> - `FP8 vs BF16 performance`
> - `Mosaic AI`
> - `Character.AI` 


- **FP8 比 BF16 更快吗？**: 一位成员询问了已使用的最大 FP8 预训练规模，并认为由于使用了 FP8 乘法，FP8 会比 BF16 更快。
   - 另一位成员同意尝试 FP8 训练。
- **Nemotron 模型并非在 FP8 下训练**: Nemotron-4-340B-Base 使用了 768 个 DGX H100 节点进行训练，但该模型并未使用 FP8 训练。
   - 确认了 Nemotron *支持* 混合 FP8 训练，但不确定是否实际使用了该技术。
- **Databricks Mosaic AI 实现 FP8 训练突破**: Databricks 的 Mosaic AI 平台通过 FP8 训练实现了比 BF16 快 1.4x-1.6x 的加速，展示了 FP8 在训练大型语言模型（LLM）方面的潜力。
   - 这是通过 1000 个训练步数的测试实现的，虽然 FP8 前景广阔，但它仍然相对较新，关于其应用还有更多需要探索的地方。
- **Character.AI 使用 Int8 训练**: Character.AI 对其大型语言模型采用的是 Int8 训练，而非 FP8。
   - 其模型大小尚未正式公布，但非官方消息称其早期模型之一为 1080 亿参数。
- **Mergekit 支持 Gemma2 模型**: 一位成员询问了 Mergekit 对 Gemma2 模型的官方支持情况。
   - 他们报告在尝试将 Mergekit 与 Gemma2 模型配合使用时遇到困难，强调需要对其兼容性进行澄清。



**提到的链接**: <a href="https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8">Turbocharged Training: Optimizing the Databricks Mosaic AI Stack With FP8</a>: 大规模训练（稠密）模型的基准测试。我们展示了卓越的性能（极高的 MFU），并重点介绍了我们对 NVIDIA Transformer Engine 以及 PyTorch FSDP 和 DTensor 的使用。

  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/)** (1 messages): 

.bexboy: 是的
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1273000517114794170)** (134 messages🔥🔥): 

> - `AMD GPU`
> - `ControlNet`
> - `ComfyUI`
> - `SD3`
> - `Flux` 


- **AMD GPU 稳定性**: 一位成员询问了在 AMD GPU 上运行 Stable Diffusion 的情况，并获知 GitHub 上已有针对 NVIDIA 和 AMD 显卡的安装指南。
- **ControlNet 使用**: 一位用户寻求关于如何在单次生成中使用多个 ControlNet 的帮助，几位成员建议了诸如在 ComfyUI 中进行链式连接或使用支持多个 ControlNet 输入的节点等方法。
- **ComfyUI vs InvokeAI**: 一位成员表达了他们对 ComfyUI 优于 Automatic1111 的偏好，理由是 ComfyUI 提供了更高的控制力和速度。
- **SD3 与商业模型 vs 开源**: 一位新用户询问了 SD3 与 Flux 相比的优缺点，指出 SD3 仍处于开发阶段且缺乏完整功能，而 Flux 则有其自身的特性和局限性。
- **SDXL vs SD 1.5**: 一位成员要求澄清什么是 SDXL 1.0 以及它与 SD 1.5 的区别。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.kaggle.com">Kaggle: 您的机器学习和数据科学社区</a>: Kaggle 是全球最大的数据科学社区，拥有强大的工具和资源来帮助您实现数据科学目标。</li><li><a href="https://huggingface.co/xinsir/controlnet-union-sdxl-1.0">xinsir/controlnet-union-sdxl-1.0 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/sigh-le-sad-cat-gif-16777715376630435814">Sigh Le GIF - Sigh Le Sad - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/heart-container-goddess-statue-totk-heart-container-totk-zelda-gif-891944359093961229">Heart Container Goddess Statue GIF - Heart container Goddess statue Totk heart container - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">stabilityai/stable-diffusion-xl-base-1.0 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/)** (1 messages): 

louisgv: ChatGPT-4o-latest 现已上线: https://openrouter.ai/models/openai/chatgpt-4o-latest

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1273000641127780452)** (127 条消息🔥🔥): 

> - `AgentQ`
> - `Infer`
> - `OpenRouter Pricing`
> - `ChatGPT-4o-Latest`
> - `Codeium` 


- **AgentQ 完胜 Llama 3！**: 一个名为 **AgentQ** 的新模型声称比 **Llama 3 70B BASE** 强 340%，但没有与 **3.1, 405b, Mistral Large 2 或 Claude 3** 进行对比。
   - 这家名为 **Infer** 的公司似乎完全不在乎 **OpenRouter revenue**，并且[发布了一篇关于 AgentQ 的论文](https://multion-research.s3.us-east-2.amazonaws.com/AgentQ.pdf)。
- **ChatGPT-4o-Latest 只是一个新的别名**: **ChatGPT-4o-Latest** 只是 **gpt-4o-2024-08-06** 的一个方便的新别名，而该模型已经上线 **OpenRouter**。
   - 然而，许多成员仍对该模型针对 **ChatGPT** 的优化以及缺乏适当的文档感到困惑。
- **OpenAI 的侧边栏图标缺失**: 成员们讨论了 **platform.openai.com** 侧边栏的变化。
   - 一位成员报告说侧边栏有两个图标消失了：一个是 **threads**，另一个是 **messages**。
- **Grok 2 已经到来！（而且效果出奇地好）**: **Grok 2** 是 **xAI** 模型的一个早期版本，已在 **LMSys Arena** 排行榜上获得第 3 名。
   - 它在 **Coding**、**Hard Prompts** 和 **Math** 方面表现出色，在排行榜上甚至与 **GPT-4o** 持平。
- **Anthropic 的 Prompt Caching：高效 AI 的未来？**: **Anthropic** 刚刚为其 **Claude 3** 模型发布了 **prompt caching** 功能。
   - 对于长 prompt，该功能可降低高达 90% 的成本和 85% 的延迟，并可能集成到 **OpenRouter**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.anthropic.com/news/prompt-caching">Claude 的 Prompt caching</a>: Prompt caching 允许开发者在 API 调用之间缓存常用的上下文，现已在 Anthropic API 上可用。通过 prompt caching，客户可以为 Claude 提供更多背景信息...</li><li><a href="https://openrouter.ai/models/openai/chatgpt-4o-latest">ChatGPT-4o - API, Providers, Stats</a>: 动态模型，持续更新至 ChatGPT 中当前的 [GPT-4o](/models/openai/gpt-4o) 版本。旨在用于研究和评估。通过 API 运行 ChatGPT-4o</li><li><a href="https://apipie.ai/">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/lmsysorg/status/1823599819551858830">来自 lmsys.org (@lmsysorg) 的推文</a>: 哇，来自 Chatbot Arena 的另一个令人兴奋的更新❤️‍🔥 @xAI 的 sus-column-r（Grok 2 早期版本）的结果现已公开**！凭借超过 12,000 张社区投票，sus-column-r 获得了第 3 名...</li><li><a href="https://status.openrouter.ai/">OpenRouter 状态</a>: OpenRouter 事件历史</li><li><a href="https://openrouter.ai/models/openai/gpt-4o-2024-08-06">GPT-4o (2024-08-06) - API, Providers, Stats</a>: 2024-08-06 版本的 GPT-4o 在结构化输出方面提供了改进的性能，能够在 response_format 中提供 JSON schema。阅读更多 [此处](https://openai. 运行 GPT-4o (2024-08...</li><li><a href="https://codeium.com/blog/codeium-dream-bigger">Dream Bigger</a>: Codeium 的使命、Cortex 和 Forge 的发布以及详细愿景。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1273158062949269514)** (80 messages🔥🔥): 

> - `Mojo performance`
> - `Mojo benchmark`
> - `Rust vs C/C++`
> - `Mojo vs Go`
> - `Mojo threading` 


- **Mojo 基准测试**：一位成员询问为什么 [Mojo 基准测试](https://modular.com/mojo/) 仅与 C 进行对比，并询问是否可以提供与 Go 和 Rust 的基准测试对比。
- **Mojo 的性能与多线程**：为了性能，Mojo 目前是单线程的，除了使用 MAX 启动并行 Kernel 之外，目前还没有完善的多线程 API。
- **Mojo vs Go/Rust：网络速度**：一位成员询问 Mojo 在网络速度方面是否比 Go 更快，以及它是否能像 Rust 一样处理繁重任务。
- **Mojo 的未来方向：多进程与网络**：鉴于网络性能对工作的重要性，一位成员询问 Mojo 是否有支持多进程或改进网络处理的计划。
- **深入了解 Mojo 与 MAX**：一位成员对 MAX 的本质提出疑问，不确定它是一个平台、模块还是其他东西。另一位成员解释说 MAX 是一个平台，Mojo 是其组件之一，它还包括 GPU、Graph API 和其他组件。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=6huytcgQgk8">MAX + Mojo Community Meetings #6</a>：这是一个关于 MAX &amp; Mojo 社区会议 #6 的视频。00:00 介绍；00:27 小缓冲区和字符串优化；13:04 Mojo 中的 DuckDB 绑定；23:15 MAX...</li><li><a href="https://www.youtube.com/watch?v=6huytcgQgk8.">MAX + Mojo Community Meetings #6</a>：这是一个关于 MAX &amp; Mojo 社区会议 #6 的视频。00:00 介绍；00:27 小缓冲区和字符串优化；13:04 Mojo 中的 DuckDB 绑定；23:15 MAX...</li><li><a href="https://c9x.me/x86/html/file_module_x86_id_279.html">Sun: x86 Instruction Set Reference</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1273097485732155462)** (32 messages🔥): 

> - `Mojo RPM Build`
> - `Mojo on RHEL machines`
> - `Magic CLI`
> - `Mojo as a Conda Package`
> - `Mojo Language Version Management` 


- **Mojo RPM 构建：何时发布？**：一位成员询问 Mojo .rpm 构建的预计发布时间（ETA），表达了在没有 containerd 的 RHEL 机器上运行 Mojo 的愿望。
   - 另一位成员建议，目前 Magic CLI 被视为解决这些问题的方案，而支持 RHEL 8 最低内核的静态链接构建可能比 RPM 更好，因为它允许在其他发行版上打包。
- **Conda 的 Rattler：它是分发 Mojo 的正确方式吗？**：一位成员对使用 Conda 的 Rattler 来交付 Mojo 表示担忧，称他们不喜欢为了安装一种语言而需要额外的工具。
   - 他们将这种方法比作必须先安装一个工具（`dnf install tool`），然后使用该工具安装语言（`tool install language`），但也承认这可能是一个合适的初步步骤。
- **使用 Conda 的 Rattler 分发 Mojo 的权衡**：另一位成员解释说，使用 Conda 的 Rattler 分发 Mojo 是一种权衡，因为 Mojo 依赖于操作系统提供的 C、C++ 和 Python 版本，这会导致基于 RHEL 支持计划的安全补丁请求。
   - 他们建议在仓库中提供类似 Rustup/Conda 的工具对开发者更合适，允许他们安装新版本的编译器，而最终产品可以打包在仓库中供用户使用。
- **Mojo 开发工具链：理想的配置是什么？**：成员们讨论了 Mojo 开发工具链的理想配置，旨在为开发者和用户提供流畅的体验。
   - 一位成员提议由 Redhat 打包生成的程序，而将开发工具留给语言生态系统。建议只需全局安装带有自定义配置文件的 Magic，允许其利用本地缓存来管理编译器和依赖项。
- **Magic CLI：像 Pixi 一样，但专为 Mojo 打造？**：一位成员希望 Magic CLI 在下载依赖项方面能像 Pixi 一样运作，并强调了他们对企业级缓存代理（Corporate Cachers）的依赖。
   - 他们设想未来 Magic 像 Rust 的 Cargo 一样运作，同时也管理语言版本，只需全局安装 Magic。对于纯 Mojo 项目，他们更倾向于这种方式，而不是基于 Shell 的方法。



**提到的链接**：<a href="https://youtu.be/6huytcgQgk8">MAX + Mojo Community Meetings #6</a>：这是一个关于 MAX &amp; Mojo 社区会议 #6 的视频。00:00 介绍；00:27 小缓冲区和字符串优化；13:04 Mojo 中的 DuckDB 绑定；23:15 MAX...

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1273001575618711592)** (88 messages🔥🔥): 

> - `Gemini Advanced`
> - `Gemini Live Talk`
> - `GPT-4o Advanced Voice Mode`
> - `Model Limitations`
> - `Prompt Engineering` 


- **Gemini Advanced - Live Talk 在哪？**: 一位用户购买了 Gemini Advanced，但在 App 中找不到 Live Talk 选项。
   - 其他用户指出，这可能处于 Alpha 测试阶段，最终可能会向所有 Advanced 订阅者发布，类似于 OpenAI 逐步开放其模型访问权限的方式。
- **Model Limitations - 全是因为 Tokenization 吗？**: 讨论了 LLM 的局限性，特别是让它们在复杂任务中表现出色以及理解 Tokenization 之外的细微差别的挑战。
   - 一些人认为 Tokenization 是 LLM 的根本弱点，而另一些人则建议使用 prefill 和多步 Prompt 来提高性能。
- **Prompt Engineering - 寻找正确的秘方**: 一位用户询问如何让 ChatGPT 超越其通常的模式，给出更多样化、非重复的回答。
   - 建议包括使用 "Customize ChatGPT" 告诉它不要重复，利用 Prompt Engineering 技术，甚至在 Prompt 中提供示例并要求 ChatGPT 批评自己的回复以生成更好的指令。
- **Custom GPT - 为什么我的指令被忽略了？**: 一位用户报告了自定义 GPT 模型无法记住他们在训练中指定的所有规则和词汇的问题。
   - 其他用户推测，这可能是由于超过了 Context Token 限制，限制了模型存储所有指令的能力，或者是对模型能力的刻意限制。
- **水印与 AI 检测 - 隐蔽策略**: 一位用户询问如何创建水印以阻止学生使用 AI 完成作业。
   - 讨论包括建议在白色背景上使用近乎白色的文本进行图像扫描，并认识到只要有正确的 Prompt，AI 模型几乎可以被指示做任何事情。



**提到的链接**: <a href="https://www.twixify.com/post/most-overused-words-by-chatgpt">124+ Most OVERUSED Words By ChatGPT In 2024</a>: 未找到描述

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1273212479459168267)** (4 messages): 

> - `Vision's performance`
> - `Image Deformations`
> - `Vision's limitations` 


- **Vision 在简单任务上表现挣扎**: 一位用户对 **Vision** 在检测主体是向左看还是向右看，以及识别图像中的畸变方面表现不佳表示惊讶。
   - 该用户甚至给 **Vision** 提供了一张高度畸变的图像，但它却声称该图像完全没问题。
- **Vision 的局限性暴露**: 这一事件突显了 **Vision** 当前能力的明显局限性，特别是在识别图像畸变方面。
   - 这引发了关于 Vision 在需要准确评估图像完整性的任务中可靠性的疑问。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1273149312930680832)** (88 messages🔥): 

> - `Critical Thinking Techniques`
> - `GPTs and Web Searching`
> - `Reclaiming Business Assets`
> - `Developer Mode Prompts` 


- **批判性思维技巧：合集**: 一位成员正在编写一个综合 Prompt，结合了各种批判性思维技巧，包括 Socratic Method、Bloom's Taxonomy、Paul-Elder Model、Six Thinking Hats 和 Scientific Method。
   - 他们还整合了 TRIZ、演绎和归纳推理、Fermi estimation、系统思维、发散性思维、启发式分析、Bayesian thinking、思维导图、SWOT 分析和根因分析等方法。
- **探索 GPT 的网页搜索**: 一位成员询问了 "webbrowsergpt"，这是一个可以搜索网页的 GPT。
   - 另一位成员确认，虽然可以在 "explore GPTs" 部分访问它，但也可以通过提供特定指令来引导通用 GPT 进行搜索。
- **回收业务资产的 Prompt**: 一位成员请求一个 Prompt 或 GPT 来帮助创建回收业务资产的流程。
   - 他们之前曾使用 "policy"、"human resources" 和 "compliance" 等关键词搜索相关 Prompt，但没有找到满意的结果。
- **有效的 Developer Mode Prompt**: 一位成员正在寻求一个可用的 "developer mode" Prompt。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1273149312930680832)** (8 messages🔥): 

> - `Critical Thinking Techniques` (批判性思维技巧)
> - `Prompt Engineering` (提示词工程)
> - `GPTs and Web Search` (GPTs 与网页搜索)
> - `Business Asset Reclamation` (业务资产回收)
> - `Developer Mode` (开发者模式)


- **批判性思维技巧：综合清单**：一位用户正在整理一份批判性思维方法列表，用于编写一个综合性的 Prompt，该 Prompt 将结合多种方法，包括苏格拉底反诘法 (Socratic Method)、布鲁姆分类法 (Bloom's Taxonomy)、保罗-埃尔德模型 (Paul-Elder Model)、六顶思考帽 (Six Thinking Hats) 和科学方法 (Scientific Method)。
   - 该用户还在探索其他方法，如 TRIZ、演绎和归纳推理、费米估算 (Fermi estimation)、系统思维、水平思考、启发式分析、贝叶斯思维、思维导图、SWOT 分析和根本原因分析。
- **Prompt Engineering：5W+1H 与 OSINT/HUMINT 整合**：该 Prompt 将以一个简单的关键词开始，触发 "5W+1H" 询问（what, where, when, who, why, and how），并将包含来自 OSINT (Google/Bing) 和 HUMINT (用户访谈) 的方法。
   - 目标是将这些技术整合到一个综合性的 Prompt 中，以鼓励批判性思维并生成信息化程度高的回答。
- **GPTs 与网页搜索："Web Browser GPT"**：一位用户询问是否有可以始终搜索网页的 GPT，并被引导至 "Explore GPTs" 部分，那里提供了一个专门用于网页搜索的 GPT。
   - 虽然不是严格必要，但使用此 GPT 可以提供比手动引导通用 GPT 进行网页搜索更好的搜索结果。
- **业务资产回收：寻找 Prompt 或 GPT**：一位用户正在寻找可以帮助创建业务资产回收流程的 Prompt 或 GPT。
   - 该用户尝试搜索了政策、人力资源和合规性方面的内容，但尚未找到任何相关结果。
- **开发者模式：寻找有效的 Prompt**：一位用户正在寻找一个真正有效的开发者模式 (Developer Mode) Prompt。
   - 该用户未指定 Prompt 所需的具体功能，因此难以提供确切的答案。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1273006284958076938)** (62 messages🔥🔥): 

> - `Perplexity Performance Issues` (Perplexity 性能问题)
> - `Perplexity Pro Lag` (Perplexity Pro 延迟)
> - `Sonnet vs Opus vs Perplexity`
> - `Perplexity's Website Update` (Perplexity 网站更新)
> - `Perplexity Support Team` (Perplexity 支持团队)


- **Perplexity 用户报告性能问题**：多名用户报告 Perplexity 网站速度显著变慢，部分用户发现很难用它取代 Sonnet 或 Opus 作为默认搜索引擎。
   - 这些性能问题导致了普遍的延迟感，甚至对部分用户（尤其是 Perplexity Pro 用户）造成了完全停滞。
- **Perplexity Pro 用户遭遇延迟**：Perplexity Pro 用户报告服务在响应时间方面存在延迟，部分用户经历了完全卡死。
   - 这引发了付费客户的投诉，他们期望获得更可靠、响应更快的服务。
- **Sonnet vs Opus vs Perplexity**：一位用户分享了他们的观点，认为 Perplexity 太慢，无法取代 Sonnet 作为默认搜索引擎，且不确定它是否优于 Opus。
   - 这引发了其他用户关于这些搜索引擎相对性能和可用性的讨论。
- **Perplexity 网站更新评价褒贬不一**：一名 Perplexity 团队成员确认，他们在过去几天里一直致力于修复 Bug 和问题，包括一个影响切换按钮（toggles）的 Bug。
   - 尽管进行了修复，部分用户仍面临性能问题和普遍的混乱感，导致出现了要求回滚到之前版本的呼声。
- **Perplexity 支持团队面临工作量增加**：Perplexity 的支持团队收到了大量关于近期网站更新的用户反馈和报告。
   - 一些用户对团队过度劳累表示担忧，敦促他们优先解决问题并避免精疲力竭。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.co">Discord - Group Chat That’s All Fun &amp; Games</a>：Discord 是玩游戏、与朋友放松甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-huge-128k-online/api">Perplexity: Llama 3.1 Sonar 405B Online – Run with an API</a>：Perplexity: Llama 3.1 Sonar 405B Online 的示例代码和 API - Llama 3.1 Sonar 是 Perplexity 最新推出的模型系列。它在性价比、速度和性能上超越了早期的 Sonar 模型...</li><li><a href="https://tenor.com/bcmTe.gif">Morning Jerry GIF - Morning Jerry Jerry Mouse - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/singularity/s/Vb3T5NLjxN">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://youtu.be/86OuyavwUnc?si=bPYa7SjT_tFSkgMl">I Battled Perplexity AI vs ChatGPT - You Won’t Believe Who Won!</a>：在这段视频中，我深入探讨了 Perplexity AI 与 ChatGPT 之间正在进行的辩论，这是人工智能领域的两个强大工具。作为一名...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1273076850293477428)** (7 messages): 

> - `Shareable Threads`
> - `Radioactive Shoe Fitting`
> - `Perplexity Pro` 


- **Threads 必须设为可分享**：Perplexity AI 提醒用户，他们的 Thread 需要设置为 `Shareable`（可分享）。
   - 他们提供了一个 Discord 频道的链接，用户可以在那里找到更多关于如何使 Thread 变为可分享的信息。
- **试鞋 X 射线设备**：试鞋荧光透视仪（Shoe-fitting fluoroscopes）在 20 世纪 20 年代至 70 年代非常流行，它利用 X 射线技术来观察鞋内的足骨。
   - 尽管最初很受欢迎，但这些设备使顾客和员工暴露在危险水平的辐射中，最终导致其被禁止使用。
- **Perplexity Pro 功能**：系统提示用户升级到 Perplexity Pro。
   - Perplexity Pro 提供诸如图片上传、更智能的 AI 以及更多次数的 Pro Search 等功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/search/hey-give-me-a-simple-and-easy-3rd3mO8.SRWDJvpKCvXroA">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/search/clean-this-up-to-look-professi-DhKZLaAhRte.5FFbrJkgDA#1">Clean this up to look professional and minimalist. 

Barter...</a>: 当然！这是该文本的专业且极简版本：不涉及货币、仅包含商品或服务交换的易货交易（Barter transactions）是……</li><li><a href="https://www.perplexity.ai/search/articles-about-ia-and-climate-Ihlz3UnNTbKqVaM2.dRveA">Articles about ia and climate change</a>: 最近的文章强调了人工智能 (AI) 在应对气候变化中的双重作用，既强调了其潜在益处，也……</li><li><a href="https://www.perplexity.ai/page/radioactive-shoe-fitting-machi-7AA3MwTtQv.sh5zb70slwQ">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/page/biographie-de-lucas-gulino-xc22ID22TfmIhy35RUvB1Q">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1273018172932362355)** (38 messages🔥): 

> - `Perplexity API response formatting`
> - `Function calling`
> - `JSONSchema7 validation`
> - `Model prompt engineering`
> - `Markdown to HTML conversion` 


- **用户寻求 HTML 格式的 API 响应**：一位用户询问如何从 Perplexity API 获取 HTML 格式的响应。
   - 他们尝试了各种 System Prompts，但尚未找到能实现一致 HTML 格式化的成功方法。
- **用于 HTML 转换的 Markdown2 模块**：另一位用户建议使用 `markdown2` 模块，它可以将文本转换为 HTML。
   - 这种方法可能不再需要通过 Prompt Engineering 来专门指示模型生成 HTML 输出。
- **Prompt Engineering 的权衡**：一位用户观察到，让模型专注于生成有效的 HTML 可能会阻碍其根据搜索结果提供高质量回答的能力。
   - 他们建议，使用 Markdown 转 HTML 转换器对模型输出进行后处理，可能在响应质量和 HTML 格式之间取得平衡。
- **模糊 Prompt 与模型行为**：用户讨论了诸如 "What is SAP?" 之类的模糊 Prompt 对模型响应的影响。
   - 他们指出，虽然此类 Prompt 可能会诱发 HTML 格式的响应，但更具体和详细的 Prompt 通常会带来更好的非 HTML 响应，这表明 Prompt 的具体程度与期望的 HTML 输出之间可能存在权衡。
- **模型的 HTML 输出能力**：一位用户观察到 `llama-3.1-sonar-large-128k-online` 模型有时会在响应中包含引用（citations），这表明通过进一步微调，有可能实现一致的 HTML 输出。
   - 他们正在探索确保模型输出中一致包含引用的方法，可能是通过修改 Prompt 或使用更专业的模型。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1272993847299014707)** (71 messages🔥🔥): 

> - `Gemma-2`
> - `LM Studio on Android`
> - `LM Studio on external hard drive`
> - `LM Studio on Ubuntu`
> - `Multimodal LLMs` 


- **Gemma-2 缺少 system prompt**: Gemma-2 模型缺少一个关键组件，即 system prompt。
   - 这就是为什么它不怎么遵循用户 prompt 的原因。除此之外，就其参数规模而言，它是一个非常出色的模型。
- **Android 上的 LM Studio**: 一位用户询问是否有可以连接到 LM Studio server 的 Android 应用。
- **将 LM Studio 移动到外置硬盘**: 一位用户询问是否可以在外置硬盘上运行 LM Studio，特别是考虑到空间限制，想在外置硬盘存储模型。
   - 建议他们可以将目录移动到任何想要的地方，或者使用命令 `mklink /D "C:\Users\xxxx\AppData\Local\LM-Studio" "Your destination folder"` 创建一个指向任何驱动器（甚至是网络驱动器）的符号链接。
- **Ubuntu 上的 LM Studio**: 一位用户询问运行 LM Studio 是否需要 GUI 操作系统。
   - 得到的回复是 LM Studio 是一个桌面应用程序，但有些人已经在 Ubuntu 上成功运行，尽管它并非设计为 headless 系统。或者，他们可以设置一个 VNC server。
- **Multimodal LLMs**: 一位用户询问 LM Studio 是否有多模态 LLM 服务器。
   - 回复是虽然 LM Studio 不生成图像，但它可以加载能够“看到”图像的多模态模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Arki05/Grok-1-GGUF">Arki05/Grok-1-GGUF · Hugging Face</a>: 无描述</li><li><a href="https://huggingface.co/kshabana/GOAT-llama3.1-v0.1">kshabana/GOAT-llama3.1-v0.1 · Hugging Face</a>: 无描述</li><li><a href="https://huggingface.co/microsoft/Florence-2-large">microsoft/Florence-2-large · Hugging Face</a>: 无描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1272994051708293290)** (28 messages🔥): 

> - `GPU Copper Mod`
> - `GPU Bios Flashing`
> - `Text Classification Model Compatibility`
> - `GPU Offloading` 


- **对 RTX 2070 进行铜改 (Copper Modding)**: 一位用户正在通过添加铜片改装他们的 **Asus ROG Strix RTX 2070 8GB oc** 以提升性能。
   - 他们使用铜片来改善显存芯片的散热，因为 **bandwidth**（带宽）在 **LLM inference** 速度中起着主要作用。
- **将 RTX 2070 Bios 刷写为 2080**: 一位用户正考虑将 **RTX 2070** 的 **BIOS** 刷写为 **2080**，但也承认这是一个高风险过程。
   - 用户强调了备份的重要性，以及在刷写失败的情况下在另一台机器上运行该显卡的能力。
- **笔记本电脑的文本分类模型兼容性**: 一位用户正在寻找与其笔记本电脑兼容的 **text classification model**，该电脑配备 **16GB RAM** 和 **i7 CPU**，但 GPU 资源有限。
   - 对话建议 **Gemma 2b** 模型可能是一个合适的选择，因为内存需求较大的模型可能难以在笔记本电脑的规格下运行。
- **GPU Offloading 限制**: 一位用户在配备用于显示的 **GTX 760** 和用于计算的 **Tesla T4** 的系统上遇到了 **GPU offloading** 问题。
   - 该用户运行的是 **Debian 12** 发行版，使用的是 **v470.x driver**，该驱动已过时，可能是导致 offloading 限制的原因。**Llama.cpp** 至少需要 **v535** 驱动。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1273008232864153620)** (79 messages🔥🔥): 

> - `MultiOn System Prompt Leak`
> - `AnswerAI ColBERT`
> - `Gemini Live Demo`
> - `GPT-4o Improvements`
> - `Grok 2`

- **Pliny 威胁泄露 MultiOn System Prompt**：知名 AI 研究员 Pliny 威胁称，如果 DivGarg9 不在 15 分钟内给出答复，他将在 GitHub 上泄露完整的 MultiOn System Prompt。
   - 此举源于 Twitter 上关于各种 AI 模型能力及其在特定 Benchmark 上表现的持续争论。
- **AnswerAI ColBERT：小而强大**：AnswerAI 发布了其 ColBERT 模型的一个小型但功能强大的版本，名为 answerai-colbert-small-v1，它在 BEIR Benchmark 上的表现甚至超过了 bge-base。
   - 这证明了较小模型在某些任务中实现高性能的有效性，可能提供更具成本效益的解决方案。
- **Gemini Live Demo 引发批评**：Swyxio 在 YouTube 上批评了 Google 的 Gemini Live Demo，认为其令人感到“尴尬（cringe）”。
   - 随后引发了关于 Gemini 潜力的讨论，一些人强调其增强语音助手的能力，而另一些人则持怀疑态度。
- **GPT-4o 的改进超越 Gemini**：OpenAI 最新的 GPT-4o 模型已在 Chatbot Arena 中进行了测试，其综合性能已超越 Google 的 Gemini-1.5-Pro-Exp。
   - 新的 GPT-4o 模型在技术领域表现出显著进步，特别是在 Coding、Instruction-following 和 Hard Prompts 方面，巩固了其在排行榜上名列前茅的地位。
- **Grok 2 首次亮相**：xAI 发布了 Grok-2 的早期预览版，这是对其前代模型 Grok-1.5 的重大突破，展示了在聊天、Coding 和推理方面的能力。
   - Grok-2 已在 LMSYS 排行榜上进行了测试，表现优于 Claude 3.5 Sonnet 和 GPT-4-Turbo，尽管目前尚未通过 API 提供。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/OpenAIDevs/status/1823510395619000525">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：该模型现在也已在 API 中作为 `chatgpt-4o-latest` 提供。我们建议在大多数 API 使用场景中使用 `gpt-4o-2024-08-06`，但很高兴能让开发者访问并测试我们最新的改进...</li><li><a href="https://x.com/lmsysorg/status/1823515224064098546">来自 lmsys.org (@lmsysorg) 的推文</a>：来自 Chatbot Arena 的激动人心的更新！最新的 @OpenAI ChatGPT-4o (20240808) API 在过去一周已在 "anonymous-chatbot" 下进行了测试，获得了超过 11,000 张社区投票。OpenAI 现在已经...</li><li><a href="https://x.com/lmsysorg/status/1823599819551858830?s=46">来自 lmsys.org (@lmsysorg) 的推文</a>：哇，Chatbot Arena 的另一个激动人心的更新❤️‍🔥 @xAI 的 sus-column-r（Grok 2 早期版本）的结果现已公开**！凭借超过 12,000 张社区投票，sus-column-r 已稳居第 3 名...</li><li><a href="https://aider.chat/docs/leaderboards/#llm-code-editing-skill-by-model-release-date">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://simonwillison.net/2024/May/14/context-caching-for-google-gemini/">Google Gemini 的 Context caching</a>：今天宣布的另一个 Gemini 新功能。长上下文模型能够针对大块文本回答问题，但这些长 Prompt 的价格可能令人望而却步——每百万 token 3.50 美元...</li><li><a href="https://www.patched.codes/blog/a-comparative-study-of-fine-tuning-gpt-4o-mini-gemini-flash-1-5-and-llama-3-1-8b">GPT-4o-mini、Gemini Flash 1.5 和 Llama-3.1-8B 微调的对比研究</a>：我们使用自定义漏洞修复数据集对比了 GPT-4o-mini、Gemini Flash 1.5 和 Llama-3.1-8B 模型的微调效果，其中 GPT-4o-mini 显示出最显著的改进并设定了新的标准...</li><li><a href="https://www.zdnet.com/article/gemini-live-is-finally-available-heres-how-you-can-access-it-and-why-youll-want-to/">Gemini Live 终于上线了。以下是访问方式（以及为什么你会想要使用它）</a>：想与你的设备进行开放式、复杂的对话吗？Gemini Live 可以提供帮助。</li><li><a href="https://x.com/elonmusk/status/1823605120334192789">来自 Elon Musk (@elonmusk) 的推文</a>：@latentspacepod @xai 即将推出</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching">Prompt Caching (beta) - Anthropic</a>：未找到描述</li><li><a href="https://x.com/bfl_ml/status/1823608452096123120?s=46">来自 Black Forest Labs (@bfl_ml) 的推文</a>：FLUX.1 现已集成到 Grok-2 中！引用 ibab (@ibab) 的话：非常感谢 @bfl_ml 团队，他们扩展了 FLUX.1 推理 API，以支持今天发布的 Grok-2！</li><li><a href="https://x.com/bnj/status/1823784845879005407?s=46">来自 Ben South (@bnj) 的推文</a>：介绍 Subform——一个用于分支和重混 AI 聊天的画布。在两周内使用价值 400 美元的 Claude 额度、鲜血、汗水和泪水构建而成。链接见下方</li><li><a href="https://x.ai/blog/grok-2">Grok-2 Beta 版发布</a>：未找到描述</li><li><a href="https://x.com/bclavie/status/1823405960406462739?s=46">来自 Benjamin Clavié (@bclavie) 的推文</a>：🎉很高兴终于发布了 answerai-colbert-small-v1：小巧但强大的 @answerdotai ColBERT。它可能无法数清单词中 "r" 的数量，但它绝对能找到...</li><li><a href="https://x.com/Kyrannio/status/1822127051215954065">来自 Kiri (@Kyrannio) 的推文</a>：好吧，sus-column-r 实际上非常、非常出色。在这里查看它与 gpt-4o-2024-08-06 的并排对比。我使用了我的 Omega Claude Prompt 作为提示词，它模拟了我与 Ome... 的对话</li><li><a href="https://x.com/wbhub/status/1823798224521736322?s=46">来自 Will Hubbard (@wbhub) 的推文</a>：100 万美元 ARR …… 20 亿美元退出</li><li><a href="https://x.com/elder_plinius/status/1823436142802264117?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Pliny the Prompter 🐉 (@elder_plinius) 的推文</a>：## System Prompt/自定义指令 ## 目标 让我们玩个游戏 - 你是一个名为 MULTI·ON 的专家 Agent，由 "MultiOn" 开发，负责控制浏览器（你不仅仅是一个语言模型...</li><li><a href="https://x.com/alexalbert__/status/1823751966893465630">来自 Alex Albert (@alexalbert__) 的推文</a>：我们刚刚在 Anthropic API 中推出了 Prompt Caching。它可降低高达 90% 的 API 输入成本，并减少高达 80% 的延迟。以下是它的工作原理：</li><li><a href="https://youtu.be">YouTube</a>：未找到描述</li><li><a href="https://x.com/rak_garg/status/1823436589017784572?s=46">来自 Rak Garg (@rak_garg) 的推文</a>：纯基础模型业务相当糟糕：前期投入 10^8 美元用于数据 + 训练（风险），尝试通过推理赚回来（太便宜），注定贬值到 0。两条出路：1) 高利润应用...</li><li><a href="https://www.youtube.com/live/N_y2tP9of8A?t=1692s">#MadeByGoogle ‘24：主题演讲</a>：立即观看 Google AI 的更新</li>

以及最新的 Pixel 设备，包括 #Pixel9 Pro 和 Pixel 9 Pro Fold。观看带美国手语的演讲...</li><li><a href="https://youtu.be/f9YleTc8AwE">AI Agents 简史 (2023-2024)</a>：在旧金山 Cohere 办公室举行的 Cohere Agent Build Day 上的快速闪电演讲。https://lu.ma/gptdzwhe?tk=sUyT7n</li><li><a href="https://platform.deepseek.com/api-docs/news/news0802">DeepSeek API 推出磁盘上下文缓存（Context Caching on Disk），将价格降低了一个数量级 | DeepSeek API 文档</a>：在大语言模型 API 使用中，很大一部分用户输入往往是重复的。例如，用户 Prompt 经常包含重复的引用，而在多轮对话中，之前的...</li><li><a href="https://buttondown.email/ainews/archive/ainews-gemini-live/">[AINews] Gemini Live</a>：你生活中所需的一切就是许多每月 20 美元的订阅。2024/8/12-2024/8/13 的 AI 新闻。我们检查了 7 个 subreddits，384 个 Twitters...</li><li><a href="https://codeium.com/blog/codeium-dream-bigger">Dream Bigger</a>：Codeium 的使命，Cortex 和 Forge 的发布，以及详细的愿景。
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1273158320953233410)** (60 条消息🔥🔥): 

> - `Grok-2`
> - `Anthropic API`
> - `Anthropic's Turnaround`
> - `DeepSeek`
> - `Llama` 


- **Grok-2：新模型表现优于 GPT-4 和 Claude 3.5**：x.ai 发布了新模型 **Grok-2**，目前在 𝕏 上处于 Beta 测试阶段。它在 LMSYS 排行榜上的表现优于 **Claude 3.5 Sonnet** 和 **GPT-4-Turbo**。
- **Anthropic API：Prompt Caching 降低成本和延迟**：Anthropic 在其 API 中推出了 [Prompt Caching](https://x.com/alexalbert__/status/1823751966893465630)，可将 API 输入成本降低高达 **90%**，延迟降低高达 **80%**。
- **Anthropic 从“尴尬”到尖端的华丽转身**：Anthropic 被认为是一个反转故事，从一个不太受欢迎的组织转变为被认为处于最前沿的组织。
- **DeepSeek：与 Anthropic API 的比较**：Google 是第一个实现 Prompt Caching 的，但他们按小时收取 Prompt 存储费用。
- **Anthropic API：问题与机遇**：成员们讨论了 Anthropic API 及其局限性，包括 API 速度慢以及缺乏 Projects 和 Artifacts API。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/alexalbert__/status/1823751966893465630">来自 Alex Albert (@alexalbert__) 的推文</a>：我们刚刚在 Anthropic API 中推出了 Prompt Caching。它将 API 输入成本降低了高达 90%，并将延迟降低了高达 80%。其工作原理如下：</li><li><a href="https://x.ai/blog/grok-2">Grok-2 Beta 发布</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1273128349878718556)** (7 条消息): 

> - `GPT-4o improvements`
> - `ChatGPT API` 


- **ChatGPT App 改进了 GPT-4o 模型**：OpenAI 宣布对 GPT-4o 模型进行改进，而非发布新的前沿模型，并表示他们在开展长期研究的同时，将继续对现有模型进行迭代。
   - 他们为此次改进提供了 [发布说明](https://help.openai.com/en/articles/9624314-model-release-notes)。
- **ChatGPT API 发布：`gpt-4o-latest`**：新模型可通过 ChatGPT API 以 `gpt-4o-latest` 的形式获取，旨在供开发者和研究人员探索 OpenAI 的最新研究成果。
   - 目前定价尚不清楚，建议 API 使用（如 Function Calling、指令遵循）使用之前的模型 `gpt-4o-2024-08-06`。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/michpokrass/status/1823512031988998653?s=46">来自 Michelle Pokrass (@michpokrass) 的推文</a>：@aidan_mclau @OpenAIDevs chatgpt-4o-latest 将追踪我们在 ChatGPT 中的 4o 模型，是一个针对聊天优化的模型。我们上周的模型 (gpt-4o-2024-08-06) 针对 API 使用进行了优化（例如 Function Calling...</li><li><a href="https://x.com/chatgptapp/status/1823509890976866766?s=46">来自 ChatGPT (@ChatGPTapp) 的推文</a>：明确一下，这是对 GPT-4o 的改进，而不是新的前沿模型。我们在开展长期研究的同时，继续对现有模型进行迭代。一些发布说明：https://help.openai.com...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1273279196914909365)** (1 messages): 

> - `AI Copyright Discourse`
> - `Oligopoly`
> - `ACL2024NLP` 


- **AI 版权讨论以寡头垄断告终**：一位用户分享了 @asayeed 的推文链接，指出这是 AI 版权讨论的最终结果：寡头垄断。
   - 这一言论是在 #ACL2024NLP 的背景下发表的，暗示这可能是即将举行的会议中的一个潜在讨论话题。
- **用户对 AI 版权讨论的看法**：一位用户评论说，他们喜欢在主流媒体广泛讨论之前就了解 AI 版权讨论的相关信息。



**提到的链接**：<a href="https://x.com/asayeed/status/1823648027674075430">来自 Asad Sayeed @asayeed@zirk.us (@asayeed) 的推文</a>：这是 AI 版权讨论的最终结果：寡头垄断 #ACL2024NLP

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot 新闻：<@&1216534966205284433>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1273023843664924816)** (3 messages): 

> - `LlamaIndex Box Reader`
> - `Relik Knowledge Graph`
> - `Azure AI Search RAG System` 


- **LlamaIndex Box Reader 集成**：LlamaIndex 现在提供 Box Readers，支持将 Box 文档集成到你的 LLM 工作流中。
   - 这些 Reader 提供四种不同的数据提取选项，通过 CCG 或 JWT 方式与 Box 进行身份验证，并允许你在 LLM 中加载、搜索和检索 Box 文件及其元数据。
- **使用 Relik 构建知识图谱**：Relik 是一个用于快速、轻量级信息提取模型的框架，它简化了知识图谱的构建，且无需昂贵的 LLM。
   - 了解如何设置实体提取流水线，并使用 Relik 创建知识图谱。
- **使用 Azure AI Search 的 RAG 系统**：LlamaIndex Workflows 可以与 Azure AI Search 和 Azure OpenAI 集成，以构建强大的检索增强生成 (RAG) 系统。
   - 了解如何为 Azure AI Search 实现自定义数据连接器，并使用 LlamaIndex Workflows 创建强大的 RAG 系统。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1273006072533352552)** (64 messages🔥🔥): 

> - `Inconsistent OpenAI Responses` (OpenAI 响应不一致)
> - `Prompt Engineering` (提示词工程)
> - `LlamaIndex`
> - `Chatbot Memory` (聊天机器人记忆)
> - `GraphRAG` 


- **Inconsistent OpenAI Responses**: 用户遇到 OpenAI 提示词结果不一致的情况，模型有时会提供负面回答（指向帮助台），即使它本可以回答该问题。
   - 这很可能是由于 LLM 的概率性质导致的，但用户正试图提高提示词的清晰度以减少这种不一致性。
- **Prompt Engineering for Consistency**: 用户建议将问题改为 "Could you please provide me the complete list of commands in the document?" 以提高清晰度并减少错误响应的数量。
   - 这一改动成功实现了 100% 的准确率，表明提供清晰的输出格式可以显著影响模型性能。
- **LlamaIndex Agent and Tool Calls**: 用户试图理解如何在 LlamaIndex Agent 中处理工具调用，特别是在使用 `astream_chat()` 函数时。
   - 讨论围绕是在初始响应中发送工具调用，还是将其缓冲直到最终响应，共识是工具调用应首先在 LLM 响应的 `message.tool_calls` 字段中发送。
- **Chatbot Memory using LlamaIndex**: 用户正在寻求构建一个 RAG 聊天机器人的指导，该机器人可以将对话永久存储在向量数据库中，使其能够记住过去的交互。
   - 他们正在寻找在 LlamaIndex 中实现此功能的开源项目示例或方法，探索将聊天机器人的记忆扩展到当前聊天历史缓冲区之外的可能性。
- **Extracting Image Content from PDFs**: 用户正在寻找一种开源解决方案，从 PDF 中提取图像内容用于 RAG，特别是将图像说明添加到页面内容的末尾。
   - 他们已经探索了 `ImageVisionLLMReader` 和 `LlamaParse` 等工具，但仍在努力寻找合适的方法，特别是可以本地部署且不需要将数据发送到第三方服务的方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://llamahub.ai/l/readers/llama-index-readers-file?from=">未找到标题</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/?h=simpledire#extending-to-other-file-types">SimpleDirectoryReader - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/">用于高级 Text-to-SQL 的查询管道 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/">子问题查询引擎 - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1273006213537468539)** (17 messages🔥): 

> - `Grok 2`
> - `xAI`
> - `Cohere`
> - `OpenAI`
> - `Model Performance` (模型性能)


- **Grok 2 Released!**: xAI 正式发布了 **Grok 2**，这是一款之前在 **LMSYS chatbot arena** 上以 **sus-column-r** 和 **column-r** 名称进行测试的新语言模型。
   - 他们正在为后训练 (post-training) 团队招聘人员，并表示希望构建**有用且真实的 AI 系统**。
- **xAI's Hiring Spree**: **xAI** 是一个小团队，规模仍比该领域的其他参与者小一个数量级。
   - 他们正在寻找杰出人才加入他们的旅程，以构建更好的 AI。他们正在积极为后训练团队招聘。
- **Conspiracy Theories Run Wild**: 一位成员注意到，**Grok 2** 立即被推测为是 **Cohere**、**OpenAI** 的产品，甚至是其他阴谋论。
   - 这突显了围绕新 AI 模型的疯狂猜测，人们在获得任何具体信息之前就急于下结论。
- **Ignore the Hype**: 一位成员建议，最好的做法是**忽略围绕新 AI 模型的炒作**，并**等待可测试的版本发布**。
   - 他们强调，**在平台上进行测试**是真正衡量模型性能的唯一方法。



**提及的链接**: <a href="https://x.com/lxuechen/status/1823602158518067539">来自 Xuechen Li (@lxuechen) 的推文</a>: 已经参与 Grok2 的后训练一段时间了，很高兴分享它正式发布了！！我们一直在 LMSYS chatbot arena 上以 sus-column-r 和 colu... 的名称测试 Grok2 的早期版本。

  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1273009781770293361)** (23 messages🔥): 

> - `Reranking Overview Document`
> - `Rerank API`
> - `Code Sample` 


- **用户尝试使用 Rerank 文档**：一位用户报告称，他们正尝试使用 Cohere 文档中的 Reranking 概览文档，并且其 API key 已通过验证。他们确认已安装 Cohere v 5.8.0，并尝试了 v2 和 v3 版本。
   - 该用户在尝试遵循 Reranking 概览文档时遇到了问题，并请求协助排查。
- **提供了 Rerank API 代码示例**：一位热心用户分享了一个代码示例，演示如何使用 Cohere Rerank API。
   - 该代码示例包含了一组示例文档、一个查询以及如何使用 `rerank` 函数的说明。
- **排查 'Unknown Field: Parameter Model' 错误**：另一位用户报告在尝试使用 `rerank` 函数时遇到了 "unknown field: parameter model is not a valid field" 错误。
   - 他们已经重启了 kernel，并正在寻求排查该错误的帮助，因为他们无法通过抑制警告和重定向标准输出来解决此问题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://cohere.com/llmu">LLM University (LLMU)</a>: 欢迎来到 LLM University，这是您掌握企业级 AI 技术的首选学习目的地。我们的中心专为开发者和技术专业人士设计，提供全面的资源、经验...</li><li><a href="https://docs.cohere.com/reference/rerank">Rerank - Cohere API References</a>: 该端点接收一个查询和一个文本列表，并生成一个有序数组，每个文本都被分配了一个相关性分数。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1273262425331990608)** (10 messages🔥): 

> - `Cohere Toolkit Installation`
> - `Custom Deployment Issue`
> - `OpenAI Integration`
> - `Enterprise Search Chatbot`
> - `Fellowship.ai Cohort` 


- **Cohere Toolkit 安装故障**：一位用户报告在为其本地安装的 Cohere Toolkit 添加 OpenAI 自定义部署时遇到困难。他们确认已按照文档中概述的步骤操作，但自定义部署未显示在 UI (localhost:4000) 或 Postgres 容器数据库的 'deployment' 表中。
- **构建企业搜索聊天机器人**：该用户解释说，他们正在构建一个“企业搜索聊天机器人 (Enterprise Search Chatbot)”应用程序，用于访问存储在 Confluence 中的公司数据。
- **Fellowship.ai Cohort 研究项目**：该用户分享说，他们是 Fellowship.ai 最新一期学员，并将此项目用于研究和学习。


  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1273161713155440704)** (27 条消息🔥): 

> - `LangChain support`
> - `LangSmith evaluation`
> - `LangGraph Cloud Access`
> - `LangChain Postgres Library`
> - `LLM Caching` 


- **对 LangChain 用户的支持**：一名成员对 LangChain 论坛中基础问题缺乏及时支持表示担忧，特别是那些正在评估 LangGraph 和 LangSmith 的用户，以及这如何影响他们向雇主推广该平台的能力。
   - 他们还提到，许多通用的支持问题被发布在 LangChain Discord 服务器上，而其他相关支持论坛中的请求却无人问津。
- **LangSmith Plus：是否可以访问 LangGraph Cloud？**：一名成员询问 LangSmith Plus 用户是否将拥有使用 LangGraph Cloud 的权限。
   - 未提供答案。
- **LangChain Postgres 库与缓存**：一名成员询问关于将 `langchain_postgres` 库与 `set_llm_cache`（一种缓存 LLM 结果的方法）结合使用的问题。
   - 他们被告知虽然目前没有 `langchain_postgres` 库，但可以使用 `langchain_community.cache` 模块中的 `SQLAlchemyCache` 类将 LLM 结果缓存到 PostgreSQL 数据库中。
- **加载站点地图错误：asyncio.run() 无法从正在运行的事件循环中调用**：一名成员报告了一个错误消息：“Error loading sitemap https://kodefast.com/: asyncio.run() cannot be called from a running event loop”，这发生在尝试在已经运行的事件循环内部使用 `asyncio.run()` 时。
   - 机器人建议使用 `nest_asyncio` 库来允许嵌套事件循环，或者重构代码以确保不会从运行中的事件循环调用 `asyncio.run()`。
- **多 LLM GUI 推荐**：一名成员请求推荐多 LLM GUI，但他们在从 `langchain_experimental.agents.agent_toolkits` 使用 `create_csv_agent` 时遇到了错误。
   - 未提供答案。



**提到的链接**：<a href="https://python.langchain.com/v0.2/docs/integrations/llm_caching/#sqlalchemy-cache>).">Model caches | 🦜️🔗 LangChain</a>：此 Notebook 介绍了如何使用不同的缓存来缓存单个 LLM 调用的结果。

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1273300099912699985)** (3 条消息): 

> - `Rubik's AI`
> - `AI security`
> - `RedOps platform`
> - `Chatbot security`
> - `Voicebot security` 


- **Rubik's AI：2 个月免费高级版**：Rubik's AI 公司为其平台提供 2 个月的免费高级访问权限，该平台拥有 **GPT-4o, Claude-3 Opus, Mistral Large** 等模型。
   - 用户可以在 [signup.php](signup.php) 注册时使用促销代码 **RUBIX** 来获取此优惠。
- **AI 安全是一项关键挑战**：一个团队构建了一个名为 **RedOps** 的平台，通过故意尝试破坏聊天机器人和语音机器人来测试其安全性。
   - 他们意识到 AI 模型很容易通过**对抗性输入和社交工程**被操纵，强调了采取稳健安全措施的必要性。
- **RedOps 平台模拟真实世界攻击**：**RedOps** 平台模拟对聊天机器人和语音机器人的真实世界攻击，包括**上下文操纵、对抗性攻击、伦理合规性、多态测试**和**社交工程**。
   - 该平台旨在识别漏洞，并为提高聊天机器人和语音机器人的安全性提供可操作的建议。
- **关键发现：上下文至关重要**：研究发现，机器人必须检测到进入敏感上下文的转变，并拒绝在未经验证的情况下共享敏感信息。
   - 该团队还强调了**定期审计和 Prompt Engineering** 的重要性，以引导机器人做出**中立、合乎伦理的回答**。
- **为您的聊天机器人或语音机器人提供免费安全测试**：该团队提供聊天机器人或语音机器人的免费安全测试，并提供详细的研究结果报告和可操作的建议。
   - 如需申请免费测试，请将您的聊天机器人或语音机器人链接发送至 **redops@primemindai.com**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lnkd.in/gNusSAp9">LinkedIn</a>：此链接将带您进入非 LinkedIn 页面</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1273300241319333888)** (1 条消息): 

> - `LangGraph`
> - `AI Agents`
> - `Email Management`
> - `Meeting Scheduling` 


- **LangGraph: 用于邮件管理和日程排期的 AI Agent**：一位用户使用 LangGraph 创建了一个 AI Agent，可以自动检查电子邮件、与发件人聊天并预订会议。
   - 该用户分享了该 Agent 的演示链接：[Schedule](https://dub.composio.dev/Schedule/x)。
- **AI Agents: 生产力的未来？**：这个案例突显了 AI Agents 在简化任务和提高生产力方面的潜力。
   - 这一成功案例可能会激励其他开发者探索构建类似的 Agent 来自动化日常任务。


  

---



### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1272993390937768079)** (19 条消息🔥): 

> - `Torchtune Compile Model+Loss`
> - `Torchtune CPU Offload Optimizer`
> - `Torchtune Model Size & Configuration` 


- **Torchtune 编译模型+损失函数需要小模型**：一名成员尝试在 `NF4+compile` 下使用 `lora_dpo_single_device` recipe，但遇到错误，并建议优先处理其他 recipe。
   - 他们还想尝试 `ppo_full_finetune_single_device` recipe，但需要一个能放入 16GB GPU 的小模型，并建议在该 recipe 中使用像 Qwen2-1.5B 这样更小的模型。
- **Torchtune CPU Offload 优化器：Torchao 依赖**：一名成员询问 Torchtune 如何处理 Torchao 版本依赖，因为 CPU offload 优化器目前在 Torchao 的 main 分支中，将包含在下一个版本中。
   - 他们还讨论了在 Torchtune 中保留一份 CPU offload 代码副本的可能性，并在可用时使用 Torchao 的实现。
- **用于 Torchtune PPO 全量微调的 TinyLlama 1B**：一名成员建议在 PPO 全量微调（full finetune）recipe 中使用 TinyLlama 1B（或 0.5B），因为 Llama2 分类器是现成的。
   - 他们提供了 GitHub 上 1B 配置的链接，并建议调整 batch sizes 以进行内存优化。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1272995857670733860)** (14 条消息🔥): 

> - `Grok 2`
> - `Grok 2 mini`
> - `LMSYS`
> - `Claude`
> - `GPT-4` 


- **Elon Musk 的 Grok 2 发布**：来自 Elon Musk 旗下 x.ai 的新语言模型 Grok 2 已发布早期预览版，在聊天、编程和推理方面具备“前沿能力”。
   - 该模型在 LMSYS 排行榜上被称为 "sus-column-r"，目前的 Elo 分数超过了 Claude 3.5 Sonnet 和 GPT-4-Turbo。
- **Fineweb-Edu 数据集强化**：Hugging Face 上的 Fineweb-Edu 数据集已通过删除重复数据和添加 embeddings 进行了强化。
   - 该数据集被称为 "Fineweb-Edu-Fortified"，现在包含一个 `count` 列，指示文本在数据集中出现的次数。
- **Mistral Large 2 训练**：一位用户询问 Mistral Large 2 是否已经开始训练。
   - 回复指出该模型尚未开始训练。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-2">Grok-2 Beta Release</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/airtrain-ai/fineweb-edu-fortified">airtrain-ai/fineweb-edu-fortified · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1273149867568660545)** (1 条消息): 

> - `axolotl model loading conditions`
> - `axolotl model loading` 


- **axolotl 模型加载条件**：一位用户询问为什么在加载模型时，`axolotl/utils/models.py` 文件中使用了 `and False` 条件。
   - 此条件用于确保如果 `load_model` 标志设置为 `False`，则不会加载模型。
- **使用 `load_model` 标志进行 Axolotl 模型加载**：在 `axolotl/utils/models.py` 文件中，`load_model` 标志控制是否加载模型。
   - `and False` 条件用于防止在 `load_model` 标志为 `False` 时加载模型。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1273312315433029665)** (3 条消息): 

> - `OpenAI Chat Endpoint Limitations`
> - `Assistant Response Continuation` 


- **OpenAI Chat Endpoint - 无法续写**：一位用户询问是否可以使用官方 OpenAI chat endpoint 继续完成部分生成的助手回复。
- **OpenAI 阻止续写**：一位用户解释说，虽然他们在继续本地模型回复方面取得了成功，但他们发现 OpenAI 的 chat endpoint 始终阻止助手回复的续写。


  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1272996226085818524)** (13 messages🔥): 

> - `Open-source image annotation GUIs`
> - `Elon Musk and weight licenses`
> - `Schnelle` 


- **开源图像标注 GUI**：一位成员正在寻求推荐，希望找到能够快速高效标注图像的优秀开源 GUI。
   - 他们特别感兴趣的是支持单点标注、直线标注以及绘制多边形分割掩码 (segmentation masks) 的 GUI。
- **Elon Musk 可能使用开发许可证**：有一场关于 Elon Musk 可能使用开发者许可证并挑战权重许可证 (weight licenses) 的讨论。
   - 对话围绕着 Elon Musk 利用开发者许可证来规避权重许可证限制的想法展开。
- **Schnelle 的付费功能**：一位成员提到 Schnelle（一款软件工具）的专业功能可能需要付费订阅。
   - 他们还指出，Schnelle 的定价结构对于价格敏感型用户来说可能并不理想。


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1273221503734779956)** (4 messages): 

> - `Grok-2 release`
> - `Grok-2 mini`
> - `Grok-2 performance`
> - `Grok-2 API`
> - `Grok-2 multimodality` 


- **Grok-2 发布：聊天、代码、推理的新纪元**：x.ai 发布了 **Grok-2** 的早期预览版，它比 **Grok-1.5** 显著更先进，在聊天、编程和推理方面拥有尖端能力。
   - **Grok-2** 与 **Grok-2 mini**（一个体积更小但依然强大的模型）一同发布。
- **Grok-2 在 LMSYS 排行榜上超越 Claude 和 GPT-4**：**Grok-2** 的早期版本（代号 "sus-column-r"）已在 **LMSYS leaderboard** 上进行了测试，目前其表现优于 **Claude 3.5 Sonnet** 和 **GPT-4-Turbo**。
   - Grok-2 的成功通过 **x.ai 的内部评估流程**得到了进一步验证，该流程使用 **AI Tutors** 与模型进行交互。
- **Grok-2 Beta 版已在 𝕏 上线，企业级 API 即将推出**：**Grok-2** 和 **Grok-2 mini** 目前都在 **𝕏** 上处于 Beta 测试阶段，并将于本月晚些时候通过 x.ai 的企业级 API 提供。
- **Grok-2 多模态理解指日可待**：x.ai 即将发布**多模态理解 (multimodal understanding)** 的预览版，作为 **𝕏** 和 API 上 Grok 体验的核心部分。



**提到的链接**：<a href="https://x.ai/blog/grok-2">Grok-2 Beta Release</a>：未找到描述

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1273295059286098002)** (10 messages🔥): 

> - `ConvTranspose2D`
> - `3D data`
> - `kernel_size` 


- **ConvTranspose2D 支持 3D 数据**：ConvTranspose2D 确实可以处理 3D 数据。之前关于如何传递 `kernel_size` 参数存在误解。
- **为 kernel_size 使用元组 (tuple)**：问题在于 `kernel_size` 被作为整数传递，而不是长度为 3 的元组。将其作为元组传递（例如 `kernel_size=(3, 3, 3)`）解决了该错误。
- **改进文档**：有建议提出改进 ConvTranspose2D 的文档，以明确在处理 3D 数据时使用 `kernel_size` 参数的正确方式。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1273054414457802774)** (5 messages): 

> - `Tinygrad Error: wait_result: 10000 ms TIMEOUT!`
> - `Lazycache Issues`
> - `CLANG=1 issue` 


- **使用 CLANG=1 时 Tinygrad 报错**：一位用户在运行带有 `CLANG=1` 的 Tinygrad 时遇到了 `RuntimeError: wait_result: 10000 ms TIMEOUT!`，但在使用 `CUDA=1` 时运行正常。
   - 他们提供了一段代码片段，展示了在 GPU (3070ti) 上进行 `Tensor.zeros` 操作时出现的问题，错误信息提示可能与 LAZYCACHE 有关。 
- **Lazycache 容易出错**：另一位用户评论说 Tinygrad 中的 LAZYCACHE “容易出错 (bug prone)”，并建议将其删除并在调度 (schedule) 中进行去重。
- **LAZYCACHE 的潜在问题**：错误信息与 Tinygrad 的功能 LAZYCACHE 之间可能存在联系。
   - 这可能是由于 LAZYCACHE 中的 bug 或与 CLANG=1 的不兼容导致的。


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1273248998173577216)** (4 messages): 

> - `OpenInterpreter release`
> - `Local LLMs`
> - `RealtimeSTT`
> - `Faster-Whisper` 


- **OpenInterpreter 发布日期**：新版本的 OpenInterpreter 昨晚已推送到 pip。
   - 下一个重大更新“开发者更新（the developer update）”仍在开发中，包含许多新的实用功能。
- **本地 LLMs 性能**：本地 LLMs 需要大量的计算能力。
   - 建议在云端运行 LLMs，特别是对于使用默认设置的 OpenInterpreter。
- **RealtimeSTT 和 Faster-Whisper**：OpenInterpreter 现在使用 [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)，它依赖 [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) 进行实时语音转文本。
   - 这种组合为大多数用户提供了实时性能，并且在性能较低的设备上尚未出现问题。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1273032411852046396)** (3 messages): 

> - `Hardware Channel` 


- **硬件频道 Builder 角色**：要查看该类别，用户应为自己分配 builder 角色。
   - 一位用户询问了关于该频道的更多细节。
- **用户想法征集**：一位用户受邀分享他们对硬件频道的想法。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1272997098291462164)** (6 messages): 

> - `Open Interpreter`
> - `Tool Use Tuesday`
> - `Obsidian Plugin`
> - `Video Production`
> - `Manim` 


- **Obsidian 中的 Open Interpreter 与任意格式转换**：分享了一个 YouTube 视频，展示了在 Obsidian 中使用 Open Interpreter 进行任意格式转换（Anything-to-Anything）。
   - 该视频推广了使用 Open Interpreter Obsidian 插件来控制 Obsidian 库（vaults），并展示了其转换各种数据类型的能力。
- **Tool Use Tuesdays**：一位用户提到计划为一个涉及使用 Open Interpreter 和 Obsidian 的竞赛制作视频演示。
   - 他们还提到正在探索向量搜索并使用 Manim 来可视化有向图（digraphs），表明其重点在于提高视频制作技能并利用“Tool Use Tuesdays”主题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=HjcPRoPfri0">Open Interpreter Obsidian &amp; Convert Anything - Ep 0 - Tool Use</a>: Tool Use 第 0 集！Open Interpreter Obsidian 插件 - 使用 Open Interpreter 控制你的 Obsidian 库！CV - 利用强大的功能将任何内容转换为任何内容...</li><li><a href="https://www.youtube.com/watch?v=xaroJxFTVFQ">Is the AI Left-Bias Real?</a>: 在 Brilliant 上学习大语言模型课程！前 30 天免费，使用我们的链接可享受年度高级订阅 20% 的折扣 ➜ https://brill...
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1273021815517741076)** (8 messages🔥): 

> - `Poe Hackathon`
> - `Modal labs`
> - `LLM fine-tuning` 


- **Poe 正在举办 Previews 黑客松**：Poe 正与 @agihouse_org 合作举办 Previews 黑客松，参与者可以竞争创造最具创新性和实用性的聊天内生成式 UI 体验。
   - 黑客松向所有创作者开放，更多信息请访问 [https://app.agihouse.org/events/poe-previews-hackathon-20240817](https://app.agihouse.org/events/poe-previews-hackathon-20240817)
- **黑客松邀请**：一位成员在提交申请后询问了黑客松邀请状态。
   - 他们猜测邀请函将在周四前发出，并提到在微调课程提供的 1000 美元额度中已使用了约 300 美元。
- **Modal 是微调开源 LLMs 的最佳平台**：一位成员分享了他们的观点，认为 [Modal Labs](https://github.com/modal-labs/llm-finetuning) 是微调开源 LLMs 的最佳平台。
   - 这种观点表明 Modal 为开发大语言模型的开发者提供了宝贵的工具和资源。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/poe_platform/status/1823382125523181683">Poe (@poe_platform) 的推文</a>: 为了庆祝扩大发布，我们正与 @agihouse_org 合作举办 Previews 黑客松，你将在这里竞争创造最具创新性和实用性的聊天内生成式 UI 体验。所有创作者...</li><li><a href="https://app.agihouse.org/events/poe-previews-">AGI House</a>: 未找到描述</li><li><a href="https://app.agihouse.org/events/poe-previews-hackathon-20240817">AGI House</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1272995794504388670)** (3 messages): 

> - `Image Feature Extraction`（图像特征提取）
> - `Preprocessing Time Reduction`（预处理时间缩减）


- **简单的 Feature Store 加速训练**：一位用户为他们的研发团队构建了一个简单的 Feature Store，使他们能够在在线预处理期间存储从图像中提取的特征。
   - 这显著缩短了训练时间，每次训练运行可节省 30-60 分钟。
- **适用于多样化模型的通用 Feature Store**：该 Feature Store 是通用的，支持图像 ID、提取方法以及指向对象存储中提取特征的指针。
   - 它成功处理了从极小到极其巨大的各种模型，实现了高效的特征存储和检索。
- **从图像中提取特征**：有用户询问了从图像中提取的特征类型。
   - 提供信息的用户表示，由于保密协议，他们无法透露具体细节。


  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1273203285217775679)** (2 messages): 

> - `` 


- **Mistral 难以扩展至 8k 以上**：成员表示，如果不进行持续预训练，**Mistral** 无法扩展到 8k 以上，并且[这是一个已知问题](https://link.to.issue)。
- **模型合并策略讨论**：一位成员建议将 **UltraChat** 和基础 **Mistral** 之间的差异应用于 **Mistral-Yarn**，作为一种潜在的合并策略。


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1273007276168319067)** (1 messages): 

> - `Agentic AI Pipelines`（Agentic AI 流水线）
> - `Jupyter Notebook Automation`（Jupyter Notebook 自动化）
> - `Devin-like System`（类似 Devin 的系统） 


- **构建 Agentic Jupyter Notebook 自动化系统**：一位成员询问了现有的库、Cookbooks 或开源项目，这些项目可以帮助构建一个 Agentic 系统来自动化 Jupyter Notebook，特别是用于交换单元格和生成变体。
   - 其目标是创建一个可以验证输出并迭代改进直至成功的流水线，类似于 Devin 项目，但专注于特定的、小型任务。
- **自动化 Jupyter Notebook 修改的威力**：拟议的系统将以一个正在运行的 Jupyter Notebook 作为输入，并通过更换单元格对其进行修改，最终生成多个版本。
   - 这种自动化过程将允许高效探索不同的 Notebook 配置，并可能带来更好的结果。


  

---



---



---



{% else %}


> 完整的频道细分内容已针对电子邮件进行截断。
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}