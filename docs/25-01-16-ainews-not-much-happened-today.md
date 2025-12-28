---
companies:
- harvey
- meta-ai-fair
- stability-ai
- alibaba
- deepseek
- hugging-face
date: '2025-01-17T06:04:28.978541Z'
description: '以下是为您翻译的内容：


  **Harvey** 获得了 **3 亿美元的新一轮融资**。**OuteTTS 0.3 1B & 500M** 文本转语音模型发布，其特点是支持 **零样本语音克隆**、**多语言支持**（英、日、韩、中、法、德）以及
  **情感控制**，由 **OLMo-1B** 和 **Qwen 2.5 0.5B** 提供算力支持。**HOVER** 模型问世，这是一个用于 **敏捷运动控制**
  的 **150 万参数神经网络**，利用了 **人体动作捕捉数据集** 和 **大规模并行强化学习** 技术。**kokoro.js** 实现了在浏览器中以极低依赖本地运行
  AI 模型。**Meta AI** 为 **区域语言理解**、**复杂推理** 和 **交互式编程环境** 等项目颁发了 **20 万美元的大模型（LLM）评估资助**。**Stability
  AI 的 Twitter 账号被盗**，引发了安全警示。**阿里巴巴 Qwen** 通过 **共识过滤机制** 改进了 **过程奖励模型 (PRMs)**，以提升
  **数学推理** 能力。**DeepSeek V3** 采用 **流水线并行** 技术，增强了 **分布式推理** 和 **长文本生成效率**。会议还重点讨论了
  **法律框架中的 AI 政策** 以及 **AI 在推动教育民主化中的作用**。此外，现场还分享了一些轻松的 AI 相关幽默。'
id: 396334e6-b342-4817-8596-176a39a5bfc0
models:
- oute-tts-0.3-1b
- oute-tts-0.3-500m
- olm-1b
- qwen-2.5-0.5b
- hover
- gpt-4o
- deepseek-v3
original_slug: ainews-not-much-happened-today-9711
people:
- reach_vb
- drjimfan
- vikhyatk
- mervenoyann
- aiatmeta
- iscienceluvr
- alibaba_qwen
- awnihannun
- ajeya_cotra
- emollick
- qtnx_
- designerx
title: 今天没发生什么。
topics:
- text-to-speech
- zero-shot-learning
- multilinguality
- emotion-control
- motor-control
- reinforcement-learning
- local-ai
- distributed-inference
- pipeline-parallelism
- mathematical-reasoning
- process-reward-models
- legal-ai
- education-ai
- ai-security
- humor
---

<!-- buttondown-editor-mode: plaintext -->**一个长周末就够了。**

> 2025/1/15-2025/1/16 的 AI News。我们为你检查了 7 个 subreddits、[**433** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **34** 个 Discord（**225** 个频道，**2732** 条消息）。预计节省阅读时间（以 200wpm 计算）：**327 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

恭喜 Harvey 完成 [新一轮 3 亿美元融资](https://x.com/pitdesi/status/1879982274831347890?s=46)。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型进展**

- **高级文本转语音 (TTS) 模型**：[@reach_vb](https://twitter.com/reach_vb/status/1879647151145590905) 宣布发布 **OuteTTS 0.3 1B & 500M** 模型，具有 **零样本语音克隆 (zero-shot voice cloning)**、**多语言能力**（英、日、韩、中、法、德）以及 **情感控制** 功能。这些模型由 **OLMo-1B & Qwen 2.5 0.5B** 驱动，是 **开放文本转语音 (Open Text-to-Speech) 革命** 的重要一步。
  
- **用于运动控制的 HOVER 基础模型**：[@DrJimFan](https://twitter.com/DrJimFan/status/1879922307923411081) 介绍了 **HOVER** 模型，这是一个专为 **敏捷运动控制** 设计的 **150 万参数神经网络**。该模型利用了 **稳健的硬件设计**、**人体动作捕捉数据集** 以及 **大规模并行 RL 训练**，展示了 **机器人运动协调** 方面的进步。

**AI 工具与产品发布**

- **用于本地 AI 运行的 kokoro.js**：[@reach_vb](https://twitter.com/reach_vb/status/1879913142873944282) 推出了 **kokoro.js**，允许开发者以 **极简依赖** 在 **浏览器中直接运行 AI 模型**。该工具可通过 `npm -i kokoro-js` 获取，促进了无需依赖服务器的 **本地 AI 实验**。
  
- **Moondream 集成与工具**：[@vikhyatk](https://twitter.com/vikhyatk/status/1879741343347724341) 预告了在沃尔玛发售的 **独家 Moondream 贴纸**，而 [@mervenoyann](https://twitter.com/mervenoyann/status/1879947783442202666) 展示了 **smolagents 的视觉支持**，支持使用 **gpt-4o** 等 API 以及各种 **HuggingFace transformers 视觉 LM**。

**公司与行业新闻**

- **Meta 的 LLM 评估资助**：[@AIatMeta](https://twitter.com/AIatMeta/status/1879990701234221190) 宣布了其 **20 万美元 LLM 评估研究资助** 的获得者，支持专注于 **区域语言理解**、**LLM 中的复杂推理** 以及 **交互式编程环境** 的项目。
  
- **Stability AI Twitter 账号被盗**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1879743492450037889) 报告称 **Stability AI 的 Twitter 账号被黑**，建议用户在恢复访问权限前 **避免点击可疑链接**。

**技术洞察与研究**

- **过程奖励模型 (PRMs) 增强**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1879966399499759661) 详细介绍了他们在 **过程奖励模型 (PRMs)** 方面的研究，强调了在 **数据标注** 和 **评估** 方面的改进，以提升 LLM 的 **数学推理** 能力。引入的 **共识过滤机制** 将 **MC 估计** 与 **LLM-as-a-judge** 方法相结合。
  
- **DeepSeek V3 的分布式推理**：[@awnihannun](https://twitter.com/awnihannun/status/1879679524167995901) 解释了 **DeepSeek V3** 中 **流水线并行 (pipeline parallelism)** 的实现，该技术通过在机器间 **按层对模型进行分片** 来 **降低通信延迟**，从而提高 **长上下文生成** 的 **推理效率**。

**政策与社会影响**

- **AI 政策与法律信任**：[@ajeya_cotra](https://twitter.com/ajeya_cotra/status/1879962661225730376) 讨论了 **AI 在法律框架中的集成**，重点是通过 **实时验证** 和 **颜色编码反馈** 系统来 **确保 AI 生成的法律信息的准确性**。
  
- **AI 在教育与无障碍领域的应用**：[@emollick](https://twitter.com/emollick/status/1879688323419275430) 强调了 **AI 在教育民主化** 中的作用，重点介绍了一些让以前没有 **电脑使用权** 的 **学生** 从 **AI 驱动的学习工具** 中受益的项目，展示了 **AI 开启机遇的潜力**。

**梗 / 幽默**

- **关于 AI 与技术的幽默观点**：
  - [@qtnx_](https://twitter.com/qtnx_/status/1879868377059213585) 幽默地表示要避免使用某些词汇，称：“不再使用 retard 这个词了，因为 Elon 在用，这看起来很尴尬 (cringe)。”
  - [@DesignerX](https://twitter.com/Teknium1/status/1879952088131891460) 对 **数学评估** 及其复杂性发表了调侃性评论，表现出对 **技术挑战** 的轻松态度。
  - [@AravSrinivas](https://twitter.com/AravSrinivas/status/1879930212336701872) 发送了一个大笑的表情符号回应 [@elonmusk](https://twitter.com/elonmusk)，将 **技术讨论** 与 **日常幽默** 融合在一起。
  

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Google 的神经记忆架构革命**

- **[Google 刚刚发布了一种新架构](https://arxiv.org/abs/2501.00663)** ([Score: 891, Comments: 283](https://reddit.com/r/LocalLLaMA/comments/1i29wz5/google_just_released_a_new_architecture/)): **Google** 发布了一种专注于 **neural memory**（神经记忆）的新架构，旨在解决模型中的长期依赖问题。主创作者在 [Twitter 线程](https://x.com/behrouz_ali/status/1878859086227255347)中详细讨论了这一公告，暗示其在提升 AI 能力方面的重大意义。
  - **Neural Memory Module**: 讨论强调了 **Neural Memory Module** 是 Google 新架构的核心组件，它利用语义键（semantic keys）和动态内存管理来处理长期依赖。它将 **Titans** 与 **RAG (Retrieval Augmented Generation)** 进行了对比，指出 **Titans** 在推理过程中提供持续学习能力，而 **RAG** 则是静态方法。[来源](https://arxiv.org/pdf/2302.00487)。
  - **性能与内存管理**: 评论对新架构的性能表示担忧，一些人对其是否优于 **Llama 3.1** 等现有模型持怀疑态度。该架构动态管理内存和处理更大知识库的能力被视为显著优势，尽管 **catastrophic forgetting**（灾难性遗忘）的挑战仍未解决。
  - **上下文与推理**: 人们对 **Titans** 实现高准确度 **200k context window** 的潜力很感兴趣，但对推理速度以及超过特定上下文长度后的准确度下降仍存疑虑。讨论涉及该架构如何在不取代传统 Transformer 的情况下将记忆集成到模型中，一些人将其视为一种潜在的演进而非革命。


- **ATTENTION IS ALL YOU NEED PT. 2 - TITANS: Learning to Memorize at Test Time** ([Score: 311, Comments: 34](https://reddit.com/r/LocalLLaMA/comments/1i26nk4/attention_is_all_you_need_pt_2_titans_learning_to/)): **Google Research** 推出了 **Titans**，这是一种新的 AI 模型，在测试时加入了专门的“长期记忆”，使其能够动态调整和更新记忆。与传统 Transformer 的平方时间复杂度相比，该模型在长输入序列下具有更高效的线性时间复杂度，理论上可以实现无限的 **context windows**。
  - 在 **Titans** 等 AI 模型中集成 **长期和短期记忆** 被视为一项重大进展，可能突破 AI 能力的边界。然而，人们对计算开销和内存需求表示担忧，用户质疑在较慢的存储介质中存储长期记忆的可行性，以及是否需要重新训练像 **llama-4** 这样的模型。
  - **Titans** 的 **线性时间复杂度** 引发了热议，用户正热切期待基准测试来验证这些说法。一些用户对在现有模型中立即采用此类进展持怀疑态度，认为大规模实施的时间表会更久。
  - **Titans** 的架构，特别是用于记忆更新的“惊喜（surprise）”机制引起了关注，并被拿来与 **SMiRL** 等其他研究进行参考。用户讨论了可能需要进行的架构调整，以有效平衡记忆与 **token** 预测。


**主题 2. UMbreLLa 增强了 LLM 在消费级 GPU 上的性能**

- **UMbreLLa: Llama3.3-70B  INT4 在 RTX 4070Ti 上达到最高 9.6 Tokens/s! 🚀** ([Score: 132, Comments: 75](https://reddit.com/r/LocalLLaMA/comments/1i28pfq/umbrella_llama3370b_int4_on_rtx_4070ti_achieving/)): **UMbreLLa** 使得在 **RTX 4070 Ti** 和 **RTX 4090** 等消费级 GPU 上运行 **Llama3.3-70B** 模型成为可能，速度分别达到令人印象深刻的 **9.7 tokens/sec** 和 **11.4 tokens/sec**。它通过**参数卸载 (parameter offloading)**、**投机采样 (speculative decoding)** 和**量化 (AWQ Q4)** 实现了这一目标，让高性能 LLM 推理在平价硬件上变得触手可及，尤其适用于编程任务。[GitHub 链接](https://github.com/Infini-AI-Lab/UMbreLLa)。
  - **推理速度与硬件**：用户报告的 Token 生成速度因硬件和 PCIE 设置而异，例如由于 **PCIE 带宽**差异，某些设置的速度要**慢 3 倍**。一位用户提到在 **16GB 显存的 4080** 上达到了 **10 tokens/sec**，而另一位用户指出在 **3090 Ti** 上仅为 **1-3 tokens/sec**。
  - **投机采样与性能**：**投机采样 (Speculative decoding)** 是核心功能，通过预测多达 **256 个 tokens**，在每次**前向传播 (forward pass)** 中实现 **13-15 个 tokens** 的产出，在编程任务中甚至可能超过 **20 个 tokens**。然而，在编程任务之外，性能可能达不到预期，甚至可能比 CPU 卸载效果更差。
  - **兼容性与未来计划**：目前该项目不支持 **AMD GPU**，但有计划扩展兼容性。用户还对支持 **Nemotron 51B** 等模型以及与 OpenAI 兼容 API 的潜在集成感兴趣。


**主题 3. Wayfarer 模型重新定义 AI Dungeon 体验**

- **介绍 Wayfarer：一个极具挑战性的角色扮演模型，旨在让你失败和死亡。** ([Score: 160, Comments: 26](https://reddit.com/r/LocalLLaMA/comments/1i2t82i/introducing_wayfarer_a_brutally_challenging/)): **Wayfarer** 是一款新推出的 **AI 角色扮演模型**，旨在解决玩家对 **AI Dungeon** 中过于宽容的 AI 的不满。该模型目前已在 [Hugging Face](https://huggingface.co/LatitudeGames/Wayfarer-12B) 上开源，提供频繁发生失败和死亡的挑战性冒险，并获得了玩家的积极反馈。
  - 用户对 **Wayfarer** 的体验评价褒贬不一，一位用户注意到在交互过程中存在角色混淆。**Nick_AIDungeon** 确认了用户反馈，并表示愿意接受更多建议。
  - 用户对扩大模型规模充满热情，**Nick_AIDungeon** 证实目前正在训练更大的模型以增强体验。
  - 该模型因其独特的方法而受到赞赏，被类比为“类魂 (souls-like)”体验，用户对开源可用性以及挑战性 AI 交互的机会表示感谢。


**主题 4. 提升 LLM 任务管理的元提示策略**

- **元提示 (Meta Prompts) —— 因为你的 LLM 可以做得比 Hello World 更好** ([Score: 133, Comments: 19](https://reddit.com/r/LocalLLaMA/comments/1i2b2eo/meta_prompts_because_your_llm_can_do_better_than/)): **元提示 (Meta-prompts)** 通过结构化提示将复杂项目分解为可管理的任务，显著增强了**大语言模型 (LLMs)** 的能力。该概念源于一篇[研究论文](https://arxiv.org/pdf/2401.12954)，涉及使用提示来定义角色、规则和交付成果，使 LLM 能够充当软件架构师、项目经理和开发人员。通过提供上下文、结构和清晰的输出，元提示将 LLM 转化为高效的团队成员，能够处理企业级复杂度，正如各种[示例](https://gist.github.com/pyros-projects/c77402249b5b45f0a501998870766ae9)和[指南](https://gist.github.com/pyros-projects/e2c96b57ac7883076cca7bc3dc7ff527)所展示的那样。
  - **提示工程 (Prompt Engineering)** 类似于向人类提出发人深省的问题；它通过使用 LLM 关联高质量回答的问题来利用 LLM 的训练成果，从而激发其最佳且最具洞察力的输出。
  - **闭源担忧**：有观点认为，为了利润而闭源可能不符合该子版块的精神，表明了对开源或社区驱动方法的偏好。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Titans：具有类人记忆的 Transformer 继任者**

- **著名 Transformer 的继任者：Titans** ([Score: 264, Comments: 63](https://reddit.com/r/OpenAI/comments/1i2bc5d/successor_to_the_famous_transformer_titans/)): Google Research 发布了一篇关于 **Titans** 的论文，这是一种仅凭 **3 亿参数** 就能超越更大模型的新型模型。这一进展表明其具备类似于人类认知的实时学习和思考能力，对 **2025** 年的 AI 发展具有重大意义。[阅读更多](https://arxiv.org/pdf/2501.00663)。
  - **Titans 模型特征**：**Titans 模型**以其新颖的神经记忆模块而著称，该模块通过记住“令人惊讶”的事件来模拟人类记忆，并拥有高达 **200 万 token** 的超大上下文窗口。然而，它并不像人类认知那样通过更新模型权重来进行传统意义上的实时学习，这是与人类认知的一个关键区别。
  - **与 Transformers 的对比**：讨论强调 **Titans** 可能是超越 Transformer 的潜在一步，它结合了 RNNs 和 Transformer 的元素，但人们对其革命性影响仍持怀疑态度。该模型的记忆机制直接集成到架构中，类似于 Attention 机制，使其能够更有效地处理超长上下文，但在实际应用中仍需考虑经济成本。
  - **类人记忆**：几位评论者强调，**Titans** 倾向于记住惊讶事件以及记忆随时间逐渐衰减的特性，让人联想到人类的记忆过程。虽然这被认为很有前景，但也有人指出 **Titans** 并没有解决持续学习（continual learning）的根本问题，因为其记忆是有限的且基于上下文的，而非基于权重的学习。


- **[OpenAI 研究员暗示他们拥有一个在“不可破解”盒子中进行递归自我改进的 AI](https://i.redd.it/s7ozn53ml8de1.png)** ([Score: 189, Comments: 79](https://reddit.com/r/OpenAI/comments/1i2a9cl/openai_researcher_indicates_they_have_an_ai/)): 据报道，**OpenAI** 正在开发一种能够在**“不可破解”的环境**中进行**递归自我改进**的 AI。这一说法基于 **Jason Wei** (@_jasonwei) 的一条推文，该推文提到了一个在安全 RL 环境中运行的 **RL 优化算法**。
  - **“不可破解”**一词被批评为具有误导性，因为它可能指的是 AI 无法利用奖励函数（reward function）漏洞的 RL 环境，而不是完全不受外部黑客攻击。**Jason Wei** 的推文被视为 **OpenAI** 员工惯用的模糊炒作模式的一部分，会导致误导和不必要的兴奋。
  - 讨论中对 **OpenAI** 的说法以及**递归自我改进**的可能性表示怀疑。一些人认为这个概念并不新鲜，并将其与 **AlphaGo** 的自我博弈（self-play）方法进行了比较，后者通过针对自身进行训练来提高性能。
  - 讨论还提出了在没有伦理保障的情况下开发 AGI 的潜在风险，并提到即使在所谓的安全系统中，社会工程学也是一个可能的漏洞，强调了采取强大安全措施的必要性。


**主题 2. AI 订阅与使用的财务分析**

- **[我每月支付 200 美元的专业订阅费用，这就是我的用途](https://i.redd.it/c1u1n894fcde1.png)** ([Score: 2051, Comments: 187](https://reddit.com/r/OpenAI/comments/1i2n2ib/i_pay_200month_for_pro_subscription_and_this_is/)): 该帖子讨论了一项**每月 200 美元的专业订阅**服务（很可能是 **ChatGPT**），用于开发 **React 网站**。互动过程突出了该服务的处理能力，并承认了潜在的错误，如“ChatGPT 可能会犯错”的提示所示。
  - **ChatGPT 的效率与实用性**：许多用户对每月 200 美元的订阅价值表示怀疑，一些人期望它能让收入翻倍或具备读心术。然而，也有人赞赏 AI 能够有效地引导非编程人员开发 React 应用，并强调了提供具体指令以获得理想结果的重要性。
  - **React 与开发挑战**：用户讨论了使用 React 的挑战，一些人对该框架表示不屑，而另一些人则强调了使用 AI 生成样板代码（boilerplate code）所节省的时间。少数人分享了个人经历，即 ChatGPT 在处理实现图论算法等复杂任务时表现吃力，导致他们不得不手动完成任务。
  - **AI 的直接性与可信度**：几条评论强调 AI 的直接回答是一个优点，这与早期版本过于详细的回答形成对比。这种直接性被比作真实开发者对模糊项目需求的回应，增强了对 AI 能力的信任感。


---

# AI Discord 摘要

> 由 o1-preview-2024-09-12 生成的摘要之摘要的摘要

**主题 1. AI 工具获得资金，但陷入“末日循环”**

- **Cursor 斩获 1.05 亿美元融资，但用户陷入“末日循环”**：根据[官方声明](https://x.com/cursor_ai/status/1880003590493991072)，Cursor 宣布从 Thrive、a16z 和 Benchmark 筹集了 **1.05 亿美元** 的 B 轮融资。然而，用户持续反馈请求缓慢和反复停滞，称之为“末日循环”（loop of doom）。尽管存在挫败感，许多人仍因 Cursor 强大的自动补全和集成环境而保持忠诚，认为其生产力提升超过了传统配置。
- **Codeium 的 Windsurf 在学生折扣期间遭遇停机**：Codeium 在[官网](https://www.codeium.com)推出了全新的 **Windsurf Editor** 并为 .edu 邮箱提供**学生折扣**，但用户经历了服务中断、功能改进延迟，甚至因一笔 **297 美元退款** 争议导致账号被注销。Codeium 在其[对比页面](https://www.codeium.com/compare)上宣传其相对于 **GitHub Copilot** 的性能优势，引发了用户间的激烈辩论。
- **Phi-4 微调热潮遭遇瓶颈**：Unsloth AI 用户成功在免费的 Colab GPU 上利用小数据集微调了 **Phi-4** 模型，但在保存合并模型时遇到了显存溢出（out-of-memory）错误。讨论集中在动态量化与 **GGUF** 格式的挑战，以及 **Phi-4** 在 `llama.cpp` 下出现的无限生成问题。

**主题 2. 新型 AI 架构有望超越巨头**

- **Google 的 Titans 瞄准 GPT-4 的宝座**：Google Research 发布了 **Titans** 架构，引入了一个神经长期记忆模块，能够处理超过 **2M** 的上下文窗口，详见其[论文](https://arxiv.org/abs/2501.00663v1)。成员们推测这是否能为 LLM 破解“类人”记忆，从而可能超越 GPT-4。
- **修改版 NanoGPT 打破训练速度记录**：如[此推文](https://x.com/hi_tysam/status/1879687807678959729)所述，一个修改版的 **NanoGPT** 在 **3.17 分钟** 内完成了训练，打破了此前 3.33 分钟的记录。开发者将速度提升归功于 **Long-Short Sliding Window Attention** 等优化。
- **Tensor Product Attention 削减 KV Cache 膨胀**：一篇新论文提出了 **Tensor Product Attention (TPA)**，旨在以更小的 KV Cache 扩展语言模型，参考了 [T6 实现](https://github.com/tensorgi/T6/blob/d4f6168852397a7b0b0d9fd65326bb91976c7067/model/T6.py#L64)。作者计划推出 Flash 版本的 TPA，目标是在大规模部署中进一步提升速度。

**主题 3. AI 伦理大地震：数据政策与 DMCA 移除**

- **OpenAI 停止窥探——默认不进行数据训练**：OpenAI 更改了其 API 数据使用政策，表示除非用户主动加入（opt-in），否则不会使用客户数据进行训练，解决了对数据隐私的担忧。详情分享在 [TechCrunch 文章](https://techcrunch.com/2023/03/01/addressing-criticism-openai-will-no-longer-use-customer-data-to-train-its-models-by-default/)中，标志着 AI 公司处理用户数据方式的转变。
- **DMCA 移除令导致 MATH 数据集下架**：据[此推文](https://x.com/tmkadamcz/status/1879584048429105238)报道，广受欢迎的 **Hendrycks MATH** 数据集收到了 DMCA 通知，涉及来自 **aops** 的内容。社区成员对这一损失表示哀悼，称其为“比 The Pile 或 Books 3 更大的损失”，强调了该数据集对开源数学资源的重要性。
- **Bora's Law 挑战以算力为中心的 AI 发展**：成员们辩论了 **Bora's Law**，即[这篇文章](https://chrisbora.substack.com/p/boras-law-intelligence-scales-with)中提出的原则：“智能随约束而扩展，而非算力”。批评者认为，过度的规模扩张忽视了智能的基本层面，建议应关注约束驱动的模型。

**主题 4. 程序员在 AI 编程助手上产生分歧**

- **Codeium vs. Copilot：代码生成之战**：用户对比了 **Codeium** 和 **GitHub Copilot**，Codeium 在其[对比页面](https://www.codeium.com/compare)上宣传其性能优势。尽管其先进的自动补全受到称赞，但用户批评了功能推出延迟和客户服务问题，包括一起 **297 美元退款** 纠纷。
- **Cursor 的编程能力与故障并存**：用户称赞了 **Cursor** 先进的自动补全和集成环境，报告称尽管面临响应缓慢和“末日循环”停滞，但工作流仍有重大改进。许多人认为 Cursor 优于 **Windsurf** 等替代方案，理由是 Cursor 拥有更深厚的工具集和更好的性价比。
- **ChatGPT 不会写代码？用户辩论 AI 的开发技能**：关于 **ChatGPT** 无法胜任真正的软件工程师角色的讨论浮出水面，用户指出虽然它可以辅助编程，但缺乏独立开发复杂应用的能力。人们表达了对未来增强功能以弥补这一差距的期望。

**主题 5. 多智能体系统（Multi-Agent Systems）与工具链成为焦点**

- **MCP 的动态工具发现（Dynamic Tool Discovery）令开发者惊叹**：MCP 引入了动态工具发现功能，允许客户端列出可用工具并在工具更改时接收实时更新，减少了重启的需求。这种方法帮助开发者跟上工具签名的频繁调整，并保持稳定的使用。
- **Open-Swarm 实现智能 Agent 调度**：Open-Swarm 框架提供了 [OpenAI 原始 swarm 框架](https://github.com/openai/swarm/blob/main/swarm/core.py)的直接替代方案，专注于 Agent 角色的清晰度和内置工具的使用。它以极低的开销简化了数据库查询和网页交互等任务。
- **OpenAI 的 Realtime Agents 探索高级模式**：OpenAI 在其 [openai-realtime-agents](https://github.com/openai/openai-realtime-agents) GitHub 仓库中发布了基于 Realtime API 构建的高级 Agent 模式演示。这展示了用于增强交互的多 Agent 编排，指向了更符合人体工程学且轻量级的多 Agent 系统。


---

# 第一部分：Discord 高层级摘要


## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的速度缓慢问题**：用户报告了请求缓慢和反复停滞的问题，称其为“末日循环（loop of doom）”，并尝试通过局部修复来提高稳定性。
   - 他们还考虑了 Windsurf 等替代编辑器，尽管许多人仍忠于 Cursor 更深层次的工具集。
- **Cursor 的巨额融资**：Cursor 宣布在 B 轮融资中从 Thrive、Andreessen Horowitz 和 Benchmark 筹集了 **1.05 亿美元**，正如其[官方声明](https://x.com/cursor_ai/status/1880003590493991072)中所确认的那样。
   - 社区成员希望这笔资金能强化功能并减少性能故障。
- **Cursor 作为生产力中心**：多位用户称赞了 Cursor 先进的自动补全和集成环境，报告称与旧设置相比，工作流程有了重大改进。
   - 他们指出，这些优势掩盖了响应缓慢的缺点，使 Cursor 成为当前工具中的首选。
- **Cursor 与 Windsurf 之争**：参与者对比了 Cursor 和 Windsurf，引用了 Cursor 更强大的功能和更好的性价比。
   - 尽管存在一些减速情况，大多数人仍偏好 Cursor 强大的功能而非其他编辑选项。
- **Python 路径难题**：一位用户发现 Cursor 意外地将项目的 Python 环境应用到了全局，导致其配置混乱。
   - 社区成员讨论了环境选择，强调需要与本地工具进行更清晰的集成。



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 获得实时工具更新**：动态工具发现确保了可用能力的实时列表，减少了功能更改时的重启次数。
   - 这种方法帮助开发者跟上工具签名的频繁调整，并保持稳定的使用。
- **Open-Swarm 实现智能多 Agent 调度**：Open-Swarm 提供了[原始 swarm 框架](https://github.com/openai/swarm/blob/main/swarm/core.py)的直接替代方案，专注于 Agent 角色的清晰度和内置工具的使用。
   - 它以极低的开销简化了数据库查询和网页交互等任务。
- **来自 OSP 的营销工具重塑产品定位**：Open Strategy Partners 引入了 [osp_marketing_tools](https://github.com/open-strategy-partners/osp_marketing_tools)，使 LLM 能够处理产品营销任务。
   - 它专注于价值映射和写作风格检查，为推广内容增加了清晰度。
- **SSE 在 Sage 和 Smithery 中势头强劲**：Sage 客户端正在开发 SSE 支持，并讨论了为获得更好控制而定制请求体的问题。
   - Smithery 推出了使用 SSE 的 STDIO 服务器云托管选项，由基于 JSON 的配置驱动。
- **Discord 机器人引起不满**：成员们批评了现有的机器人，表示他们宁愿编写一个更高效的替代品。
   - 他们还提到了现代 Discord 内置功能（如 **/ban**），指向了更强大的用户选项。



---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Editor 与学生定价优惠**：Codeium 推出了全新的 **Windsurf Editor**，配备了大量以开发者为中心的功能，同时在其[官网](https://www.codeium.com)为拥有 .edu 邮箱地址的学生提供**学生折扣**。
   - 使用 **.ac.uk** 和 **.unina.it** 域名的国际学生表达了对资格限制的担忧，促使他们联系 [support](https://codeium.com/support)，直到该优惠范围进一步扩大。
- **DeepSeek 让用户陷入循环**：尽管 Codeium 宣传其 Benchmark 表现出色，但 DeepSeek 在与 Cline 配合使用时因导致无限循环而收到负面反馈。
   - 社区成员称其*不适合日常使用*，敦促工程师修复这些可靠性问题。
- **Cascade 提示词技巧与功能抱怨**：成员们分享了 **Cascade** 的策略，如内联命令和提示词复用，以最大限度地提高额度使用率和输出质量。
   - 他们还批评了改进延迟（如缺失拖放功能），指出了数月未处理的请求，并敦促加快功能交付。
- **退款风波与 Codeium vs Copilot 对决**：一名用户的 **297 美元退款**纠纷导致账号被注销而非解决，引发了对 Codeium 支持方式的抵制。
   - 与此同时，Codeium 在[对比页面](https://www.codeium.com/compare)中宣传其相对于 **GitHub Copilot** 的性能优势，尽管目前仍有关于服务中断的投诉。
- **企业版计划与无 GPL 训练**：Codeium 宣传了具有自托管能力的**企业版计划**，并强调他们不在 **GPL** 代码上进行训练，参考了[这篇博客文章](https://www.codeium.com/blog/copilot-trains-on-gpl-codeium-does-not)。
   - 他们认为这一立场对于保护组织免受法律陷阱的影响至关重要，同时仍能提供先进的 AI 驱动开发工作流。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Phi-4 微调热潮**：一位用户利用[免费的 Colab GPU](https://x.com/UnslothAI/status/1879942441538609583) 成功在小数据集上微调了 **Phi-4**，并强调了在保存合并模型时遇到的显存溢出（OOM）挑战。他们还比较了**动态量化**与 **GGUF** 格式的推理效率。
   - 讨论涉及了 **Phi-4** 在 `llama.cpp` 下出现的错误无限生成问题，以及关于 **Ollama** 正确聊天模板的不确定性，参考了 [Unsloth 文档](https://docs.unsloth.ai/get-started/installing-+-updating)。
- **Onnx 与 TensorRT 之争**：一位用户发现通过 **Onnx** 与 **TensorRT** 运行同一模型时存在**显著的输出差异**。他们质疑是框架优化还是转换步骤导致了这种不匹配。
   - 目前尚未提供具体的修复方案，但这种差异引发了对不同推理引擎间**部署一致性**的担忧，尤其是对于关键任务。
- **Flash Attention 2 故障**：有人报告了用于性能测试的 **Flash Attention 2** 安装失败。另一位成员提供了 Colab 环境的直接帮助来进行排查。
   - 他们建议验证依赖项和一致的 GPU 驱动程序，确保 **Flash Attention 2** 不会破坏高级微调的关键速度测试。
- **Grokking 收益与 LORA 蒸馏**：关于 **grokking** 和模型突然泛化的讨论引用了[一段 YouTube 视频](https://youtu.be/H3OofROzlA0?si=e9b_lK1592TxgJcQ)，探讨了过拟合如何转化为意想不到的洞察力。对话暗示，关于记忆与真正学习的见解可能会影响 **Unsloth** 的训练技术。
   - 成员们还辩论了应用 **LORA** 进行知识蒸馏的可行性，质疑其是否等同于高级训练策略中基于响应的蒸馏。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLM 批处理势头强劲**：成员们探讨了*批量文本续写 (batch text continuations)*，指出 **llama.cpp** 仅支持单个 prompt，并推举 **vllm** 作为解决方案。
   - 他们认为基于批处理的 API 对于简化**逐 token 训练 (token-by-token training)** 至关重要，并称其为*下一波可扩展 LLM 服务的浪潮。*
- **DMCA 下架通知导致 MATH 下架**：一份 **DMCA 通知** 导致 Hugging Face 上的 **Hendrycks MATH** 被停止访问，引用了来自 **aops** 的内容，详情见[此推文](https://x.com/tmkadamcz/status/1879584048429105238)。
   - 社区成员称其为*比 The Pile 或 Books 3 更大的损失*，强调了该数据集对开源数学资源的重要性。
- **修改版 NanoGPT 打破速度记录**：一个修改版的 **NanoGPT** 在 **3.17 分钟**内完成了训练，打破了[此推文](https://x.com/hi_tysam/status/1879687807678959729)中分享的 3.33 分钟的前纪录。
   - 开发者将上下文增益归功于 **Long-Short Sliding Window Attention**，并指向一个 [GitHub pull request](https://github.com/KellerJordan/modded-nanogpt/pull/71) 以获取进一步改进。
- **TruthfulQA 技巧浮现**：成员们通过简单的启发式方法将 **TruthfulQA** 的准确率提升至 **79%**，详见[此帖](https://turntrout.com/original-truthfulqa-weaknesses)。
   - 他们认为有缺陷的人类标注削弱了 *Halueval*，呼吁设计更强大的 benchmark 以保护测试的完整性。
- **Deepspeed Zero 阶段引发开发者分歧**：一位用户发现 **Deepspeed zero stage 2** 与模型并行不兼容，如[此代码片段](https://github.com/EleutherAI/gpt-neox/blob/f7a5a6f9da47de4d4d7cdf776c0832b257f329ef/megatron/training.py#L958-L973)所示。
   - 他们报告在 **512 个 AMD MI250x GPU** 上每单位仅有 **28 TFLOPs**，描述了*与 AMD 官方规格之间的差距。*

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt 中的标题修改功能**：**Bolt** 的新更新允许直接编辑项目标题，正如 [Stackblitz Twitter](https://x.com/stackblitz/status/1879625416706785365) 所宣布的那样，这使得在列表中跟踪项目变得更加简单。
   - 这一改进通过将标题与实际项目目标同步，帮助用户保持工作区整洁。
- **聊天快照在重新加载后保留**：来自 **thecodacus** 的名为 `feat: restoring project from snapshot on reload` 的 pull request 引入了聊天历史的快照系统（如[此处](https://github.com/stackblitz-labs/bolt.diy/pull/444)所示），允许用户在重新加载时恢复项目状态。
   - 它确保了用户交互的连续性，并跨会话保留相关的代码文件系统数据。
- **Git 支持即将到来**：Office hours 确认 **Git 支持** 可能会在约 **2-4 周**内上线，这增加了人们对 **Bolt** 中强大版本控制功能的期待。
   - 社区成员期待该功能发布后能实现更顺畅的协作和代码跟踪。
- **Token 海啸触发警告**：日志显示单个命令消耗了 **400 万个 token**，在频道中引发了警报。
   - 参与者呼吁进行更深入的调查，以将使用量保持在实际限制范围内，并防止进一步的 token 激增。
- **部署困境与 Stripe 故障**：用户在部署大型 **Bolt** 项目时面临难题，促使了诸如将资产移动到 **Amazon S3** 之类的建议。
   - 与此同时，**Stripe** 集成咨询依然存在，因为一些用户在结账流程中遇到了配置障碍。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Swarm 席卷 A1111**：由于持续的更新和详尽的文档，SWARM 在用户采用率上盖过了 A1111，许多人称赞其在专门任务中的表现。
   - 爱好者们认为开发者的积极参与是这个新兴界面的核心优势。
- **可疑诈骗惊扰 Stability**：**@StabilityAI** 一个被盗的 Twitter 账号发布了虚假的代币公告，引发了即时警报。
   - 成员们分享了来自 [Dango233 的推文](https://x.com/dango233max/status/1879734940264481006)作为证据，并回顾了以往针对毫无防备的追随者的诈骗案例。
- **衡量灵感**：用户权衡了 **Stable Diffusion** 的每秒迭代次数（it/s）指标，并参考 [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 作为基准性能。
   - 他们注意到各种 UI 中内置的计时器和元数据日志是评估图像生成速度的有效方法。
- **许可知识减轻负担**：参与者澄清说，**Stability AI 的社区许可**通常不需要非商业用途的正式署名。
   - 他们承认建议署名可以建立良好的信誉，而商业场景可能需要更深入的许可考量。
- **打印潜力势头渐起**：一位按需打印企业家探索了放大 **Stable Diffusion** 输出结果以用于大规模项目的方法。
   - 建议通过私信提供，重点介绍了适用于业务应用的高分辨率预设和自定义工作流。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek 的低迷与 Sonnet 的崛起**：成员们观察到 **DeepSeek3** 的延迟以及传闻中 **500GB VRAM** 的需求，并引用了 [Reddit 讨论](https://www.reddit.com/r/LocalLLaMA/comments/1ho0w52/deepseek_does_not_need_5_hours_to_generate_1/)中相互矛盾的细节。
   - 他们转向使用 **Sonnet** 以获得更好的性能，并考虑使用价格为 **$0.25/mtok** 的 **Hyperbolic**，这暗示了用户对高性价比解决方案的广泛追求。
- **MOE 减少 GPU 损耗**：一些用户强调了 **MOE (Mixture of Experts)** 的部分权重加载功能，该功能通过仅激活所需的专家模型来减少大型系列运行时的资源占用。
   - 他们推测精确的批处理（batching）可能会进一步降低整体成本，引发了对更高效工作负载的期待。
- **Aider 中的 CEDARScript 对话**：一位用户展示了一个 [GitHub PR](https://github.com/CEDARScript/cedarscript-integration-aider)，旨在让 **Aider** 采用 **CEDARScript** 作为编辑格式，且开销极小。
   - 讨论内容包括合并是否会带来实质性的优势，但这些提案尚未达成明确结果。
- **Helicone 的单行代码可观测性**：**Helicone** 推出了一款 [开源 LLM 可观测性工具](https://github.com/Helicone/helicone)，承诺通过单行代码集成实现成本追踪、**LLM security** 和请求指标监控。
   - 他们推荐云端托管，但也支持通过 **docker-compose** 进行本地运行，并提供缓存和自定义速率限制以优化性能。
- **提升 AI 安全的安全层**：一些参与者讨论了实施 **security filter**（安全过滤器）以在发送请求前拦截敏感数据，强调了潜在的风险规避。
   - 他们指出先前的资源泄露是前车之鉴，结论是专门的安全防护模块对于企业环境可能至关重要。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 推出周边基金**：成员们澄清 **Nous Research** 是一个私人组织，部分资金通过周边销售和私募股权筹集，与政府或学术界的联系极少。
   - 少数人对**贴纸**表现出兴趣，暗示这是一种适度但充满活力的增加收入的方式。
- **LLAMA 1B QLoRA 面临压力**：成员们审查了 **LLAMA 1B QLoRA** 的训练图表，对较小的数据集规模和有限的训练步数表示担忧。
   - 他们辩论了在评估模型输出时，计算适应度分数（fitness scores）与更简单的**性能指标（performance metrics）**各自的优劣。
- **优化器对决：GrokAdamW、Ortho Grad 和 GrokFast**：参与者对比了 **GrokAdamW** 和 **Ortho Grad**，注意到 GrokAdamW 改进了损失指标（loss metrics）并有 GitHub 引用，但 Ortho Grad 可能存在冲突点。
   - GrokFast 在**稳定性**方面表现不佳，促使人们对 **Orthograd** 产生兴趣，将其视为 torch 优化器的潜在替代方案。
- **PRMs 和记忆化引起关注**：成员们深入探讨了用于中间步骤彻底监督的**过程奖励模型（PRMs）**，并引用了 Qwen 团队的文档。
   - 他们还涉及了 **LLM 记忆化**方法，引用了 [Anthropic 的研究](https://www.anthropic.com/research/mapping-mind-language-model)进行更深入的探索。
- **神经长期记忆旨在寻求平衡**：一篇新论文介绍了一种用于捕获历史上下文的**神经长期记忆**模块，链接至 [arXiv](https://arxiv.org/abs/2501.00663v1)。
   - 它将**循环模型（recurrent models）**与 attention 结合，承诺在处理长程依赖关系时实现快速训练和推理，且无需高昂成本。



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **数字病理学与 Groovy 脚本收益**：一位用户通过使用 NotebookLM 处理数字病理学中的**图像标注**，克服了寻找 Groovy 脚本的困难，节省了大量项目时间。他们称赞 NotebookLM 能够迅速解析需求并为棘手的用例生成**功能性脚本**。
   - 其他人也表达了热情，称其为*显著的生产力提升*，并建议使用 NotebookLM 为专门的工作流创建类似的领域特定脚本。
- **交互模式引发课堂热议**：成员们称赞了 NotebookLM 中的**交互模式（Interactive Mode）**，认为它能快速加载模块资源并促进对学术内容的实时探索。分享的截图显示了对课程材料进行提示（prompting）如何激发新的教学策略。
   - 他们还提到对即将到来的学期充满*期待*，建议更多教育工作者可以采用这种方法来简化教学。
- **播客生成难题**：几位成员在从多个来源提取内容时遇到了**播客生成**问题，最终通过将来源分开放入不同的 notebook 找到了解决方法。他们注意到取消勾选无关来源可以提高准确性，但对于这是否是 **NotebookLM Plus** 的功能仍存在困惑。
   - 社区反馈强调了主持人互动不佳和音频质量平平的问题，并讨论了可能用于生成更连贯最终文件的指令。
- **Workspace 困扰与 NotebookLM 许可澄清**：关于各种 Google Workspace 计划中 **NotebookLM Plus** 的困惑接踵而至，根据 [Workspace 官方博客](https://workspace.google.com/blog/product-announcements/empowering-businesses-with-AI)，澄清了 Gemini 和 NotebookLM Plus 等 AI 功能将继续包含在内，无需额外费用。
   - 社区成员引用了 [Bora's Law](https://chrisbora.substack.com/p/boras-law-intelligence-scales-with-constraints) 来断言更广泛的扩展策略，而其他人则确认旧版许可不会失去现有功能。
- **来源上传困难影响效率**：NotebookLM 目前没有批量上传选项，这让想要快速导入大量 URL 的用户感到困惑。目前他们必须手动添加每个来源或依赖单文件上传。
   - 一些人抱怨缺失该功能对**多源**工作流的影响，指出更集成的方案可以大幅优化大规模数据摄取。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Minimax 强大的 4M 上下文**：新推出的 **Minimax-01** 在 **4M** 上下文长度下通过了 **Needle-In-A-Haystack** 测试，表现惊人，详见 [OpenRouter 页面](https://openrouter.ai/minimax/minimax-01)。
   - 爱好者们对公告中附带的图片表示赞赏，认为这暗示了 **Minimax-01** 潜在的多模态能力。
- **DeepSeek 延迟令人失望**：关于 **DeepSeek** 的问题包括在繁忙时段服务不可靠的报告，许多用户遇到了 API 减速。
   - 一些社区成员分享了故障排除技巧，如调整 API 设置和关注供应商错误，以保持任务正常运行。
- **OpenRouter 的区域锁定引发争议**：据确认，**OpenRouter** 遵循 **OpenAI** 和 **Anthropic** 的政策执行区域限制，这让用户感到意外。
   - 社区讨论集中在如何应对这些限制，并分享了在被封锁区域的使用经验。
- **Gemini 出现异常**：**Gemini flash 2.0** 模型意外更改了端点，给活跃用户带来了困惑和错误。
   - 受影响的用户交流了隐私设置的变通方法，并坚持认为迫切需要官方修复或文档。
- **活动页面谜团**：用户注意到**活动页面**为不同的 API keys 显示相同的图表，导致对使用数据的困惑。
   - 针对该页面的设计引发了辩论，一些用户要求更清晰地分离交易，以帮助准确跟踪部署。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R+ 获得多语言优势**：#discussions 频道的参与者报告称，**Command R+** 覆盖了多种编程语言，如 **Python** 和 **JavaScript**，并可以通过 API 进行测试。
   - 一位用户建议进行类似于 08-2024 版本的持续更新，并提醒每个新迭代本质上都构成了一个不同的模型。
- **Stripe 介入并提供支付便利**：参与者澄清说，**Stripe** 处理 Cohere 平台内的支付流程，提供了一条简单的升级路径。
   - 他们解释说，**OpenRouter** 将查询路由到所有 **Cohere models**，为需要统一访问的开发者简化了采用过程。
- **Rerank 3.5 助力代码**：成员们称赞 **Rerank 3.5** 在涵盖 **Python**、**JavaScript** 和 **C++** 的代码任务中表现强劲，尽管一些利基用例仍不受支持。
   - 他们注意到当加载更多文档时，模型倾向于语义匹配，建议进行额外的校准以获得更高的准确度。
- **Embeddings 遇到瓶颈**：开发者对更新 **embedding models** 需要重新对海量数据进行 Embedding 表示沮丧，因为没有从旧版本的迁移路径。
   - 他们强调，由于重新处理的开销，这种负担往往导致用户长期依赖现有的 Embeddings。
- **用于深度学习的 LLMU 与 Cookbooks**：人们强调 **LLM University (LLMU)** 是一个免费资源，同时还有 Cookbooks 和为新账户提供的 **$75** 积分，链接见 [LLM University](https://cohere.com/llmu#text-representation)。
   - 他们推荐这些课程来启动生成式 AI 实验，称其为初学者的有益入门途径。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 通过 JSPI 进军浏览器端**：Tinygrad 现在可以通过启用 **JSPI flag** 在浏览器中运行，并且已在 Mac、Ubuntu 和 Windows 上成功运行，详见[此测试页面](https://mesozoic-egg.github.io/tinygrad/)。
   - 用户确认“在启用 JSPI flag 后，在我的 M1 Pro 上可以运行”，并强调这种新方法极大地提升了广泛的兼容性。
- **George Hotz 奇特的云端 GPU 愿景**：George Hotz 提出了一个设想：所有联网的机器都可以像单个 GPU 一样运行，正如[这条推文](https://x.com/__tinygrad__/status/1879930546652156027)所述。
   - 他强调“在当前的 NVIDIA 技术栈之上，存在着一个充满可能性的全新世界”，暗示了并行计算的未来方向。
- **Conda 安装故障**：一位用户在 conda 环境中安装 Tinygrad 时遇到了 `libgcc_s.so` 不是 ELF 文件的错误，参考了[此 GitLab 链接](https://git.informatik.uni-hamburg.de/4kirsano/master-thesis/-/blob/main/.conda/lib/libgcc_s.so)。
   - 切换到不带 venv 的标准 Python 解决了该问题，这暗示 conda 可能会覆盖关键的系统库。
- **TinyJit 与 Metal 的博弈**：TinyJit 在配备 Metal 后端的 2019 款 MacBook Pro 上运行较慢，经追溯发现是 GPU 同步瓶颈所致。
   - 通过调试日志的支持，对 JIT 设置进行微调并在旧款 Intel MacBook Pro 上禁用 Metal graph 后，性能得到了一些提升。
- **导出模型与算子融合 (Operator Fusion)**：Tinygrad 允许用户对 jitted 模型进行 pickle 处理以便快速重新加载，这与 openpilot 复用编译产物的方法如出一辙。
   - 在 [tinygrad-notes/20250117_fusion.md](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20250117_fusion.md) 分享了关于算子融合的链接后，社区兴趣大增，该文档展示了通过融合 (fusion) 和反融合 (un-fusion) 策略进行的性能优化。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **TITANS 攻克“类人”记忆**：分享了 [Google Research 的 Transformers 2.0（又名 **TITANS**）](https://www.youtube.com/watch?v=x8jFFhCLDJY)的链接，并询问它们是否已经为 LLM 破解了**类人**记忆的难题。
   - 成员们想知道这个框架是否能促进更多上下文丰富的输出，称其为“记忆扩展的一次重大飞跃”。
- **全模态过载：延迟与质疑**：**OpenAI** 和 **Gemini** 因推迟图像生成功能的上线而面临质疑，在社区中引发了不确定性。
   - 一些用户推测可能会出现更精细的开源音频模型，但情感输出的处理仍然是一个“棘手的环节”。
- **PrivateGPT 与 Obsidian：知识组合拳**：成员们探索了将 **PrivateGPT** 与 **Obsidian** 笔记结合，旨在将个人数据输入到本地 AI 工作流中。
   - 他们讨论了让用户自有文档与模型输出之间实现更平滑协同的方法，强调了“强大的个人知识检索”能力。
- **30 天快速掌握 Prompt 技巧**：一位用户提议利用[共享资源](https://chatgpt.com/share/67897fc3-6580-8000-af35-d693f933acfb)，在短短 30 天内学习 **Prompt Engineering** 并撰写一本书。
   - 其他人则敦促使用“自我发现技术”和额外的网页搜索，坚持认为“熟练的提示词”可以加速写作。
- **GPT-4o 获得 Canvas 与任务魔法**：新的 **GPT-4o** 任务允许用户安排提醒，例如“下午 3 点练习西班牙语”，ChatGPT 会准时提醒。
   - 与此同时，**Canvas** 仍然存在于工具箱图标后面，尽管有些人在版本历史记录中遇到了界面异常。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Bora 定律挑战大型 AI**：一位成员引用了工作论文 [Bora’s Law: Intelligence Scales With Constraints, Not Compute](https://chrisbora.substack.com/p/boras-law-intelligence-scales-with)，认为**既有方法**可能存在缺陷。
   - 他们提出**智能**随着定义良好的约束而增长，从而引发了对替代性 *AI development* 路径的关注。
- **新的 'Sonar' 和 'Sonar-Pro' 引发猜测**：一位用户在 labs 中发现了对 **sonar** 和 **sonar-pro** 的引用，引发了关于即将推出的模型扩展的疑问。
   - 他们分享了一张[引用这些模型的图片](https://cdn.discordapp.com/attachments/1161802929053909012/1329550935038623877/image.png)，助长了关于另一个潜在 **API** 变动的传闻。
- **Claude Sonnet 在代码任务上受挫**：几位成员报告 **Claude Sonnet** 在 CSV 文件处理请求上表现不佳，质疑其在编程方面的可靠性。
   - 他们讲述了因错误建议而产生的*持续冲突*，对该 AI 的一致性表示怀疑。
- **图像生成大比拼**：社区对来自 ChatGPT, Flux, Grok 和 Perplexity 的**图像输出**进行了辩论，强调了主要的质量差异。
   - 一位用户在比较日出视觉效果时宣称“差距巨大”，强调了 **Perplexity** 的相对弱点。
- **AI 工具辅助 3D 打印势头强劲**：成员们探索了 **AI 驱动的 3D 物体设计**，展示了对创建机械零件和爱好者玩具的新方法的兴趣。
   - 他们在一个[讨论链接](https://www.perplexity.ai/search/can-perplexity-ai-be-used-to-g-tp.qKJ1JQJaAAmpqQUb7Og#2)中提供了技巧，暗示了 **3D printing** 与 AI 之间更深层次的协同作用。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **挤满 Token：Context Window 难题**：一位用户对“context 已满 90.5%”的警告提出疑问，引发了关于 **Context Window** 以及 Token 如何随着对话增长而累积的解释。
   - 社区成员指出，有时建议调整模型的容量以避免部分截断，并建议在未来提供更大的 context 设置。
- **系统 RAM vs VRAM：大辩论**：一场讨论澄清了 CPU 推理使用系统内存，而基于 GPU 的设置依赖 VRAM，如果 GPU 资源耗尽则回退到 RAM。
   - 成员们建议查看 [LM Studio 网站](https://lmstudio.ai/download)了解硬件详情，特别是对于遇到缓存问题的 M2 Mac 用户。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 苦战电影剧本**：一位用户尝试使用 GPT4All 分析一份 **45 页的剧本**，但发现它只能处理单个场景，尽管该模型声称拥有 **128KB** 的容量。
   - 他们测试了分块处理（chunk-by-chunk）的方法来进行更广泛的分析，在调整工作流并重新加载应用后获得了更好的结果。
- **伦理边界：ChatGPT 4.0 vs 其他模型**：**ChatGPT 4.0** 与其替代版本在处理显式内容方面出现了差异，突显了不同的审查政策。
   - 参与者质疑这些**伦理门控**是否限制了用户获取平衡数据的权利，一些人呼吁制定统一的指南。
- **用于暗黑场景的 DavidAU 和 Magnum 模型**：社区建议倾向于使用 **DavidAU** 的模型进行前卫或非暗黑风格的写作，并指向 [huggingface.co/DavidAU](https://huggingface.co/DavidAU) 作为参考。
   - 其他人提到了 **Magnum** 模型，并推荐了特定的 VRAM 设置，以优化各种写作任务的性能。
- **Quantization 与模型管理技巧**：一位用户调整了在 [Hugging Face 文档](https://huggingface.co/docs/transformers/main/main_classes/quantization)中找到的 Quantization 设置，以提升 **Gemma** 模型在 GPU 上的速度。
   - 他们发现将新模型添加到 GPT4All 的指定文件夹并重启应用是必不可少的，并参考了 [Llama 比较图表](https://www.prompthackers.co/compare/llama-3.2-3b/llama-3-8b)获取指导。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **LeetGPU 的发布吸引了 CUDA 开发者**：新的 **LeetGPU** [在线 CUDA 实验场](https://leetgpu.com/) 提供免费的 GPU 代码执行且无需注册，让开发者可以在任何环境下快速测试 **CUDA** 例程。
   - 创建者鼓励社区分享反馈，激发了那些为 GPU 相关项目寻找 **collaborators** 的人们的兴趣。
- **Torchinductor 策略与编译心得**：社区成员重点介绍了一篇关于 **Torchinductor** 的博客，这是一个使用 **define-by-run IR** 和 **symbolic shapes** 的 *PyTorch 原生编译器*，并参考了 [TorchDynamo](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747) 以及它如何加速动态 Python 代码。
   - 他们还分享了来自 [这个 GitHub 仓库](https://github.com/DWarez/torch_compile_blogpost) 的 *Dissecting Torch Compile*，强调了从 **Caffe** 向更用户友好的 ML 框架的转变。
- **MI300X 内存魔力与 MLPerf 之谜**：讨论涉及了将 **MI300X** 节点划分为多个共享部分如何通过减轻 **infinity cache** 的负载来增强内存性能。
   - 另一位用户想知道 **MLPerf** 供应商如何在 GPT-3 未完全开源的情况下运行 **GPT-3** 基准测试，暗示了存在封闭合作或部分访问权限。
- **用 CUDA 实现 Flash Attention**：一个名为 [damienjose/cuda-flashattention](https://github.com/damienjose/cuda-flashattention) 的 **Flash Attention with CUDA** GitHub 仓库引起了小组的注意，为加速注意力机制提供了参考。
   - 建议的用法包括针对大规模序列任务的 *blockwise matmul* 方法，为在 GPU 上高效处理 token 开启了门路。
- **Arm64 Runner 与修复故障的聊天功能**：**GitHub** 为公共仓库推出了免费的 **Linux arm64 托管 runner**，为在 ARM 硬件上构建的开发者扩展了部署选项，详见其 [Changelog 条目](https://github.blog/changelog/2025-01-16-linux-arm64-hosted-runners-now-available-for-free-in-public-repositories-public-preview/)。
   - 他们还引入了一项新的 **Copilot chat** 功能，可以实时解释 *Actions job failures*，让开发者直接从 PR 合并框或任务页面进行排错。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **教师模型蒸馏势头强劲**：成员们测试了一个引导小型学生模型的教师模型，重点关注专业化数据而非广泛覆盖。
   - 他们辩论了当学生模型在较窄的输出上进行训练时，在实际使用中是否仍能保持良好的稳定性。
- **Google 的新蓝图超越 Transformers**：Google Research 发布了一种声称在某些任务中超越标准 Transformers 的方法，引用了[这篇新论文](https://arxiv.org/abs/2501.00663)。
   - 聊天中还探讨了与 **Gemini 1.5** 的潜在联系，暗示它可能集成了新设计的功能。
- **OpenAI 调整数据使用并面临成本超载**：OpenAI 现在仅在用户选择加入时才使用 API 数据进行训练，以回应有关强制数据使用的担忧。
   - 报告显示他们可能会在 Azure 服务器上花费 **40 亿美元**，在训练上花费 **30 亿美元**，引发了对财务可行性的质疑。
- **张量积注意力 (Tensor Product Attention) 削减 KV Cache 膨胀**：一篇新论文提出用 TPA 来扩展具有更小 KV Cache 的语言模型，参考了 [T6 实现](https://github.com/tensorgi/T6/blob/d4f6168852397a7b0b0d9fd65326bb91976c7067/model/T6.py#L64)。
   - 作者计划为 TPA 开发 Flash 方法，旨在进一步提升大规模部署中的速度。
- **更薄的 4090 显卡避免损坏**：沉重的 4090 GPU 可能会导致 PCB 断裂，引发了中国境内将其重新封装为双槽变体的努力。
   - 一个针对双槽位 48GB RTX 4090 的 [eBay 列表](https://www.ebay.com/itm/126885374543) 在一天内获得了 **23** 次浏览，说明了人们对这些改良板卡的兴趣。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Chollet & Knoop 启动 Ndea**：Francois Chollet 与 Mike Knoop 合作推出了 [Ndea](https://x.com/fchollet/status/1879583863368032432)，强调通过**深度学习引导的程序合成（deep learning-guided program synthesis）**来扩展 AI 的能力。他们的方法将适应与**发明（invention）**视为先进 AI 进步的基石。
   - 观察者指出，这一方向可能会重塑模型处理代码生成和创意的方式，人们对动态学习领域的潜在突破充满期待。
- **Curator 合成数据激增**：开源库 [Curator](https://x.com/madiator/status/1879579213554147665) 承诺将**高质量合成数据**的创建速度提升 **10倍**，这对于后训练（post-training）数据集至关重要。社区成员强调了它在为 LLM 和专用 Agent 生成稳健数据集方面的实用价值。
   - 他们还提到，高效的合成数据流水线可能会减少耗时的手动标注，从而能够更快地对新模型变体进行实验。
- **Titans 应对超长上下文**：**Titans** 架构提供了一种可以在测试时调整的元上下文内存（meta in-context memory），其上下文限制可能超过 **2M**，表现有望超越 **GPT-4**。这种方法挑战了标准的 Attention 机制，为处理海量序列提供了不同的路径。
   - 与会者引用了 [Ali Behrouz](https://x.com/behrouz_ali/status/1878859086227255347) 的观点，对内存限制以及该设计是否能在实际任务中超越现有解决方案提出了疑问。
- **HAL 登上 Agent 评分榜**：[HAL](https://x.com/sayashk/status/1879932823668498576) 项目在 **11 个基准测试**中评估了超过 **90 个 AI Agent**，将推理型模型与标准语言模型进行了对比。爱好者们强调了成本权衡和可靠性，指出巨大的性能提升可能伴随着高昂的价格。
   - 他们还讨论了 Agent 评估的可信度，以及推理驱动的方法在日常场景中是否真的优于更简单的语言模型。
- **Harvey 获得 3 亿美元巨额融资**：据报道，法律初创公司 **Harvey** 正以 **30 亿美元**的估值筹集 **3 亿美元**资金，此前该公司在 7 月份以 **15 亿美元**的估值筹集了 **1 亿美元**。讨论集中在他们 **3000 万美元**的收入如何通过这笔资金增长，并推动 AI 在律师事务所的更快部署。
   - 推测集中在 AI 法律服务市场的竞争，以及 Harvey 激进的融资策略是否为其他行业参与者树立了先例。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 的 Subreddit 社区上线**：现在有了**官方的 Modular subreddit** [r/ModularAI](https://www.reddit.com/r/ModularAI/)，邀请社区成员加入。
   - 一位成员惊叹道 *“这就是正确之道！”*，其他成员也对在这个新平台聚集表现出兴奋。
- **Modular 仓库的 GitHub 组织架构调整**：**Modular** 已将其公开的 GitHub 仓库从 [ModularML](https://github.com/modularml) 迁移到 [Modular](https://github.com/modular)，并保留了所有历史记录。
   - 他们预计会自动重定向，但鼓励社区报告遇到的任何**意外问题**。
- **Mojo 的复杂递归类型**：一位用户报告了在 **Mojo** 中实现递归类型（recursive types）的挑战，指出了 `UnsafePointer` 的陷阱以及官方支持的不完善。
   - 他们建议在 `List` 上使用**拷贝构造函数（copy constructor）**以避免崩溃，并参考了 [Issue #3917](https://github.com/modularml/mojo/issues/3917) 中相关的调试级问题。
- **SIMD 的表现引发讨论**：开发者讨论了 **SIMD** 并不总是能带来速度提升，并引用了 [Ice Lake AVX-512 Downclocking](https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html)。
   - 他们警告说，SIMD 的收益因 CPU 而异，如果盲目期待性能提升，可能会变成一个**陷阱（footgun）**。
- **Mojo 中可选参数的异常**：Mojo 中的一个**可选参数（optional argument）**在求值为 None 时导致了段错误（segmentation faults），记录在 [Issue #3950](https://github.com/modularml/mojo/issues/3950) 中。
   - 贡献者建议查看 GitHub 上的示例修复方案，同时承认该 Bug 仍在调查中。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **身份黑客松：$5k Xeno 资助**：Plastic Labs 和 Betaworks 启动了 **Agent Identity Hackathon**，奖金总额为 5,000 美元，邀请团队在 [Luma](https://lu.ma/5rlcrlpb) 报名。
   - 申请将于 **1 月 26 日**截止，敦促参与者分享 GitHub 链接，以便 **资助委员会（grants committee）** 进行审核。
- **模型基准测试势头**：LiveCodeBench 新增了 **167 个新问题**（总计 880 个），以展示 **Gemini-Flash** 和 R1 等模型改进后的推理能力，详见[此推文](https://x.com/StringChaos/status/1879619028651745287)。
   - SWE-bench 还推出了多模态 JavaScript Bug 评估，同时 **TGI** 采用了支持 AMD 和 TPU 的多后端支持，详见 [Hugging Face 博客](https://huggingface.co/blog/tgi-multi-backend)。
- **Cerebras 芯片挑战传统观念**：Cerebras 声称其 **晶圆级芯片（wafer-scale chip）** 保持了与较小设计相当的良率，详见[其博客](https://cerebras.ai/blog/100x-defect-tolerance-how-cerebras-solved-the-yield-problem)。
   - 他们将故障与 **H100** 大小的芯片进行了比较，声称强大的容错能力抵消了巨大的 50 倍芯片面积。
- **AMD 的 Ai2 梦想与 Intel 的对比策略**：有人提议 **AMD** 应该给 **Ai2** 每人 1 万美元，并利用 MI300X 加速器，正如 [Tensorwave](https://tensorwave.com/) 所宣传的那样，以实现*更快、更简单*的 AI 解决方案。
   - 与此同时，**Intel** 赞助了 Stability AI，引发了对 GPU 厂商寻求明智联盟的对比。
- **人类、LLM 与 Meta 的 Project Aria**：**下一步最佳行动系统（next best action system）** 可以赋予人类操作员优势，目前存在关于针对 AI 的虚构社会运动的讨论，以及对技术突然转变的怀疑。
   - 同时，**Meta** 扩大了 [Project Aria](https://www.projectaria.com/) 的注册范围并澄清了数据使用情况，允许用户随时退订促销邮件。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 与 llmlingua2 联动**：一位用户将 **llmlingua2** 集成到了 LlamaIndex 中，引用了 [GitHub 上的 PR](https://github.com/run-llama/llama_index/pull/17531)，但在使用 `make` 时遇到了 linting 问题。
   - 另一位用户建议安装 **pre-commit** 或运行 `make lint` 来快速处理脚本，强调了 LlamaIndex 与 **llmlingua2** 之间的协同作用。
- **ChromaDB 中的过滤热潮**：一位成员探索了在 **ChromaDB** 中使用 **ExactMatchFilters** 来处理数千份法律文档，但不确定子索引路由（sub-index routing）是否是最佳方法。
   - 他们对性能开销表示怀疑，并询问现有的元数据过滤方法是否能更有效地处理大规模数据。
- **Neomagus 在 LLM x Law 黑客松中获胜**：**Neomagus** 背后的团队在以法律为主题的黑客松中凭借实时验证功能获胜，该功能可以当场标记错误的引用（[更多详情](https://t.co/jEqZrnn11H)）。
   - 参与者指出，提高 **AI 生成的法律信息的准确性**是增强对基于 LLM 解决方案信任的关键。
- **Women in AI RAG 黑客松升温**：在帕洛阿尔托宣布举办 **Women in AI RAG Hackathon**，重点关注与 [@zilliz_universe](https://t.co/2Bzg80dh29) 合作的 **检索增强生成（Retrieval-Augmented Generation）**。
   - 组织者鼓励女性技术人员参加这一全天活动，分享了[更多信息](https://t.co/ey8ebq9fbx)并提供强大的导师指导机会。
- **标签提取之争**：一位用户询问 **标签提取（tag extraction）** 是否应该与产品描述任务分开还是合并，强调了成本和性能方面的担忧。
   - 他们强调了 **延迟（latency）** 挑战以及重复调用可能导致的 **标签质量** 差异。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **极速 Text-to-SQL 设置**：一位用户在短短 **20 分钟**内构建了一个 **Text-to-SQL 流水线**，并评论说设置过程非常快速且简单。
   - 他们强调了其 **用户友好性**，并指出这是未来基于 AI 的数据查询的一个宝贵经验。
- **关于 DSPy V3 发布时间的推测**：有人提出了关于 **DSPy v3** 何时发布的问题，反映了对潜在新功能的关注。
   - 目前尚未引用正式公告，社区仍在等待更多信息。
- **dspy ReAct 工具与加法函数问题**：一位用户在 **dspy ReAct** 中遇到了错误，该错误标记加法工具由于缺少参数而无法计算两个数字。
   - 进一步的问题包括一个语法错误，即 'retur' 替换了 'return'，导致在使用 **LM-Studio** 配合 **加法函数（addition function）** 时输出错误。

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **Chat Template Tangle**: **聊天模板纠葛**：该小组讨论了如何构建**理想的聊天模板**，探讨了将 ChatML 或 **Llama3** 作为可选方案。
   - 他们追求最小的开销，但要求格式一致，这促使建立更清晰指南的压力增加。
- **Torchtune Tussle**: **Torchtune 之争**：一位成员透露，集成 **Torchtune** 需要*剥离大量内容*，暗示了重大的代码调整。
   - **caseus_** 调侃了停滞不前的进展，指出在顺利对接方面的精力（bandwidth）有所不足。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Cooperative AI Summer School Kicks Off**: **协作式 AI 暑期学校启动**：**协作式 AI 暑期学校**的申请截止日期为 **2025 年 3 月 7 日**，活动将于 **2025 年 7 月 9 日至 13 日**在伦敦附近的 Marlow 举行。
   - 已确认的演讲者包括 **Michael Wellman**、**Zarinah Agnew** 和 **Ariel Procaccia**，涵盖协作式 AI 的前沿研究，并提供了[财务援助详情](https://www.cooperativeai.com/summer-school/summer-school-2025)。
- **Cost Controls Steer Technology Choices**: **成本控制引导技术选择**：参与者强调，**成本**驱动着维持 MLOps 工作流中经受过考验的解决方案的决策。
   - 预算强烈影响团队选择或坚持使用稳定的技术，以确保实用性。
- **Churn Prevention Approaches Spark Interest**: **流失预防方法引起关注**：一位阔别两年的用户询问了关于**流失规避**的新策略，以及如何开始学习当前工具。
   - 其他人指出了现代框架和真实案例在减少不断变化的市场中用户流失的重要性。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Bora's Law Reframes AGI Growth**: **Bora 定律重构 AGI 增长**：一位成员批评了 OpenAI 实现 **AGI** 的方法，强调了 **Bora 定律**，即*智能随约束而扩展，而非算力*，并引用了 [Chris Bora 的这篇文章](https://chrisbora.substack.com/p/boras-law-intelligence-scales-with)。
   - 他们声称暴力扩展忽略了约束的核心作用，建议专注于**约束驱动的数学**是实现真正智能的关键。
- **Open Interpreter's Code Execution Tweak**: **Open Interpreter 代码执行调整**：爱好者们注意到 **Open Interpreter 1.0** 将其直接代码执行功能限制在**命令行**操作中，引发了对效率降低的担忧。
   - 其他人呼吁恢复该功能并添加 **Python 便捷函数**以帮助 LLM 有效学习，认为这些限制是重大的降级。



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba Jolt vs OpenAI**: **Jamba 冲击 vs OpenAI**：一位用户将 **Jamba API** 集成到多个后端服务中，推测其表现可能超越 **OpenAI** 的响应。
   - 他们指出这引发了关于 **OpenAI** 地位的质疑，激发了在实际应用中速度和有效性的对比。
- **Community Cheers for Jamba**: **社区为 Jamba 欢呼**：其他用户对关于 **Jamba API** 的正面评论表示赞赏，肯定了其支持者群体。
   - 这些反馈突显了人们对 **Jamba** 作为日常使用中 **OpenAI** 有力替代方案的兴趣日益增长。



---


**LLM Agents (Berkeley MOOC) Discord** 频道没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**Torchtune Discord** 频道没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**LAION Discord** 频道没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 频道没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**HuggingFace Discord** 频道没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 频道没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1329178802613653637)** (450 条消息🔥🔥🔥): 

> `Cursor 性能问题, 新融资公告, 用户体验与生产力, 与其他工具的对比, Python 环境问题` 


- **Cursor 性能问题持续存在**：许多用户报告了对 Cursor 请求缓慢以及无法执行简单任务的持续挫败感，将他们的体验比作陷入了“末日循环”。用户建议了各种故障排除步骤，但在效率方面仍然面临困难。
   - 性能问题导致一些用户开始探索 Windsurf 等替代方案，尽管用户表示 Cursor 的能力使其仍然是首选。
- **Cursor 获得 1.05 亿美元 B 轮融资**：Cursor 宣布从知名投资者处筹集了 1.05 亿美元的 B 轮融资，这意味着其在功能增长和改进方面具有潜力。
   - 用户希望这笔资金能够增强 Cursor 的功能，同时不损害服务质量。
- **用户体验凸显生产力提升**：几位用户报告称，与之前的编程体验相比，使用 Cursor 显著提高了生产力，并指出了其先进的 autocomplete 和高效率。
   - 尽管有一些关于请求缓慢的抱怨，用户仍认为 Cursor 的预测能力使其优于市场上的其他工具。
- **Cursor 与 Windsurf 的对比**：用户对 Cursor 的评价优于 Windsurf，强调了 Cursor 的功能、性价比和整体性能。
   - 随着 Cursor 被视为明显的赢家，讨论强调了即使在当前存在一些技术挑战的情况下，它的功能也优于 Windsurf。
- **Python 环境困惑**：一位用户提出了关于 Cursor 在全局范围内使用特定项目的 Python 环境而非默认环境的问题，这影响了他们的工作流。
   - 这引发了围绕项目设置的讨论，强调了在 Cursor 中更好地管理 Python 环境的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.radix-ui.com/colors/custom">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/cursor_ai/status/1880003590493991072?s=46&t=kUuVqsG2GMX14zvB592G5w">来自 Cursor (@cursor_ai) 的推文</a>: 我们从 Thrive, Andreessen Horowitz, Benchmark 以及现有投资者处筹集了 1.05 亿美元的 B 轮融资。我们很高兴地报告，Cursor 现在被数百万工程师用作他们的...</li><li><a href="https://huggingface.co/omkarthawakar/LlamaV-o1">omkarthawakar/LlamaV-o1 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/facepalm-really-stressed-mad-angry-gif-16109475">Facepalm Really GIF - Facepalm Really Stressed - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.refactoringui.com/">Refactoring UI</a>: 未找到描述</li><li><a href="https://v0.dev">Vercel 出品的 v0</a>: 与 v0 聊天。通过简单的文本提示生成 UI。复制、粘贴、交付。</li><li><a href="https://vercel.com/templates">Dashboard</a>: 未找到描述</li><li><a href="https://x.com/alexalbert__/status/1879917906294870196?s=46">来自 Alex Albert (@alexalbert__) 的推文</a>: 为 @AnthropicAI 开发者提供的生活质量升级：我们调整了 prompt caching，现在你只需要在提示词中指定缓存写入点——我们将自动检查缓存命中...</li><li><a href="https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/">设置你的第一个隧道 · Cloudflare Zero Trust 文档</a>: 要创建和管理隧道，你需要在源服务器上安装并验证 cloudflared。cloudflared 是将你的服务器连接到 Cloudflare 的全球网络的桥梁。</li><li><a href="https://www.youtube.com/watch?v=FvLYLQKn2GM">Jessica Sachs | Vite 在移动端上的魔力 | ViteConf 2023</a>: 想要在构建下一个移动应用时利用 Vite 极快的开发服务器和丰富的生态系统吗？你可以结合 Vite 和 Ionic Capacit... 的力量。</li><li><a href="https://cursor.directory">Cursor Directory</a>: 为你的框架和语言寻找最佳的 Cursor 规则。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1329182234560823438)** (241 条消息🔥🔥): 

> `MCP Tool Discovery, Open-Swarm Framework, Semantic Tool Selection, Dynamic Tool Updates, Home Assistant Integration` 


- **MCP 动态工具发现**：MCP 中的动态工具发现允许客户端列出可用工具，并在工具发生变化时接收通知，这有助于在无需重启的情况下保持功能最新。
   - *这对于频繁更新工具签名的 API 特别有用，能最大限度地减少对用户的干扰。*
- **Open-Swarm 框架增强**：Open-Swarm 框架旨在提供 OpenAI 原始 Swarm 框架的直接替代方案，并增强了用户友好交互和原生工具支持。
   - 利用角色定义明确的 Agent 可以更高效地处理任务，例如数据库查询和网页交互。
- **实现语义工具选择**：语义工具选择涉及使用 Embedding 在向量空间中表示工具，从而能够根据任务相关性和用户上下文进行更智能的工具选择。
   - 这种方法可能会提高工具利用效率，并降低与 API 访问相关的成本。
- **与 Home Assistant 集成**：将 MCP 与 Home Assistant 集成可以增强自动化能力，例如根据用户位置触发动作的提醒。
   - 这种集成展示了智能家居技术如何通过自动化促进个性化任务管理。
- **澄清 MCP 术语**：用户对 MCP 术语表示困惑，特别是关于 MCP Bridge 及其功能，建议改进文档。
   - 理解工具、客户端能力和底层协议之间的关系对于有效实施至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.mcp.run/tasks/using-tasks">Working with Tasks | 🤖</a>: 任务允许您在安装的一系列 Servlet 中注册 Prompt 并触发</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/tool-use#tool-use-system-prompt)">Tool use (function calling) - Anthropic</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd/tokenizer_config.json#L198>">tokenizer_config.json · Qwen/Qwen2.5-32B-Instruct at 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd</a>: 未找到描述</li><li><a href="https://github.com/SecretiveShell/MCP-Bridge/blob/master/docs%2Fusecases.md">MCP-Bridge/docs/usecases.md at master · SecretiveShell/MCP-Bridge</a>: 一个提供兼容 OpenAI 端点以调用 MCP 工具的中间件 - SecretiveShell/MCP-Bridge</li><li><a href="https://github.com/openai/swarm/blob/main/swarm/core.py">swarm/swarm/core.py at main · openai/swarm</a>: 探索人体工程学、轻量级多 Agent 编排的教学框架。由 OpenAI 解决方案团队管理。 - openai/swarm</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/blob/4c71c6168fb70c70cd1c7e358e78b664a794210c/src/mcp/types.py#L759>">python-sdk/src/mcp/types.py at 4c71c6168fb70c70cd1c7e358e78b664a794210c · modelcontextprotocol/python-sdk</a>: 用于 Model Context Protocol 服务端和客户端的官方 Python SDK - modelcontextprotocol/python-sdk
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1329219357204811847)** (24 条消息🔥): 

> `MCP-Bridge, SSE 支持, 开源客户端改进, Open Strategy Partners 工具, Discord 功能` 


- **MCP-Bridge 增加采样支持**：[MCP-Bridge](https://github.com/SecretiveShell/MCP-Bridge) 现在支持采样（sampling），允许 **OpenAI chat completions** 与 MCP 服务器无缝集成。
   - 这一增强功能让开发者可以在之前不支持采样的客户端中使用采样功能。
- **Open Strategy Partners 发布营销工具**：Open Strategy Partners 发布了 [osp_marketing_tools](https://github.com/open-strategy-partners/osp_marketing_tools)，这是一个基于 Python 的 MCP 服务器，旨在增强 LLM 在产品营销方面的能力。
   - 该工具辅助完成价值映射和写作风格检查等任务，以简化营销工作流程。
- **Sage 客户端讨论 SSE 支持**：成员们对 Sage 客户端即将支持 **SSE** 表示兴奋，这可能会增强其功能。
   - 讨论集中在自定义请求体和有效集成功能上。
- **Smithery 增加云托管选项**：针对 STDIO 服务器，Smithery 引入了利用 SSE 的云托管选项，并采用 JSON 格式的配置数据。
   - 成员们对探索这一功能表现出兴趣，并指出了其在服务器管理方面的潜在优势。
- **对 Discord 机器人功能的抱怨**：有成员抱怨目前的 Discord 机器人功能不足，建议大家可以协作创建自己的机器人。
   - 此外，有人指出现代 Discord 包含了如 **/ban** 等内置命令，以增强用户控制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/open-strategy-partners/osp_marketing_tools">GitHub - open-strategy-partners/osp_marketing_tools: 一个 Model Context Protocol (MCP) 服务器，赋能 LLM 使用 Open Strategy Partners 的核心写作和产品营销技术。</a>：一个 Model Context Protocol (MCP) 服务器，赋能 LLM 使用 Open Strategy Partners 的核心写作和产品营销技术。 - open-strategy-partners/osp_marketing_tools</li><li><a href="https://github.com/SecretiveShell/MCP-Bridge">GitHub - SecretiveShell/MCP-Bridge: 一个提供兼容 OpenAI 接口以调用 MCP 工具的中间件</a>：一个提供兼容 OpenAI 接口以调用 MCP 工具的中间件 - SecretiveShell/MCP-Bridge
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1329220375556395088)** (1 条消息): 

> `学生折扣定价, Windsurf 编辑器发布, Codeium 对比 GitHub Copilot, 企业版方案, 训练数据` 


- **学生通过折扣定价享受大幅优惠**：Codeium 为持有有效 **.edu** 邮箱的用户推出了**学生折扣定价**，在 Pro 级 Windsurf 上提供大幅折扣。
   - 学生可以[在此注册](https://www.codeium.com)以享受这一限时优惠。
- **推出 Windsurf 编辑器**：Codeium 发布了 **Windsurf 编辑器**，这是一款专为无缝编码体验打造的全新 IDE。
   - 此次发布强调了专门针对开发者需求设计的先进功能。
- **Codeium 表现优于 GitHub Copilot**：Codeium 自信地宣称其是最智能的 AI 代码生成工具，并提供了数据来支持其与 **GitHub Copilot** 对比的各项主张。
   - 用户可以[阅读更多关于性能质量对比的内容](https://www.codeium.com/compare)来查看 Codeium 的表现。
- **通过企业版方案释放潜力**：Codeium 推广其**企业版方案**，旨在提供高质量且安全的 AI 工具，以实现更快的工程交付。
   - 该方案提供灵活的部署和自托管选项，确保为企业提供量身定制的解决方案。
- **保护用户免受法律风险**：Codeium 强调其不在非许可代码（如 **GPL**）上进行训练，从而保护用户免受潜在的法律问题。
   - 更多细节可以在其讨论这一关键区别的 [博客文章](https://www.codeium.com/blog/copilot-trains-on-gpl-codeium-does-not) 中找到。



**提到的链接**：<a href="https://www.codeium.com">Windsurf 编辑器和 Codeium 扩展</a>：Codeium 是开发者喜爱、企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。

  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1329182230995537990)** (102 messages🔥🔥): 

> `学生折扣, 客户服务问题, Windsurf 更新, VSCode 和 IDE 偏好, 服务中断` 


- **关于学生折扣的澄清**：成员们询问非 **.edu** 结尾的学生邮箱（如 **.ac.uk** 或 **.unina.it**）是否可以用于折扣，回复建议目前的优惠主要针对 **.edu** 域名。
   - 有建议提出联系支持团队进行澄清，因为他们正在努力扩大除 **.edu** 限制之外的资格范围。
- **对客户服务的投诉**：一位用户对 Codeium 关于 **$297 退款**请求的客户服务表示不满，称其账户被错误注销，而退款问题未得到解决。
   - 该用户对支持团队表示不满，而另一位成员指出，尽管有服务方面的投诉，但该 IDE 非常出色。
- **Windsurf 和 IDE 偏好**：一些用户讨论了他们相比 **VSCode** 等其他 IDE 更倾向于使用 **Windsurf**，而另一些用户则对服务交付和支持表示担忧。
   - 一条评论强调，即使 IDE 表现良好，支持和客户服务体验也会极大地影响整体满意度。
- **服务中断和问题**：几位成员报告了 **autocomplete** 功能的问题，一些人表示需要一个**状态页面 (status page)** 来提供服务中断的更新。
   - 讨论串表明，关于服务中断的查询很常见，用户希望平台的运行状态能有更高的透明度。
- **许可证问题和账户管理**：讨论了从普通方案切换到**学生折扣方案**的过程，一些成员不确定毕业后当前的订阅将如何管理。
   - 用户寻求关于现有订阅者是否可以在学生身份变更后保留其价格的澄清。



**提到的链接**：<a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>：需要帮助？联系我们的支持团队以获得个性化协助。

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1329198537245851670)** (157 条消息🔥🔥): 

> `Windsurf 功能问题、学生折扣疑虑、DeepSeek 性能、功能请求反馈、Cascade 提示词技巧` 


- **Windsurf 在功能性方面表现不佳**：用户报告了 Windsurf 的各种**功能问题**，包括在使用 Cline 配合 DeepSeek 时出现死循环，以及在代码编辑过程中出现内部错误。
   - 一些用户指出，虽然 Windsurf 在某些任务中表现良好，但无法提供持续稳定的表现，导致用户感到沮丧。
- **学生折扣资格困惑**：多位用户对**学生折扣的资格**表示困惑，特别是针对非美国机构的 .edu 电子邮件地址。
   - 目前正在讨论将该计划扩展到更多国家，但许多国际学生目前仍无法获得该折扣。
- **DeepSeek 支持不足**：用户对 DeepSeek 的能力表示不满，评论称其经常导致循环，且在即时使用中并不实用。
   - 其他人建议，虽然 DeepSeek 的基准测试令人印象深刻，但其与特定应用程序的集成目前尚显不足。
- **基础功能请求被忽视**：一位用户指出，一些简单的功能请求（如对图像和文件的拖放支持）已被忽视数月。
   - 用户普遍感到沮丧，认为尽管通过正规渠道提交了请求，但往往被忽视。
- **高效使用 Cascade 的专业技巧**：用户正在寻求 **Cascade 的专业技巧**，特别是关于如何编写提示词以获得更好的输出以及如何利用其功能。
   - 建议包括利用内联命令以避免额度耗尽、在生成提示词时发挥创意，以及在社区成员之间分享成功的提示词。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/text-phone-waiting-hurry-messenger-gif-4073783462256955308">Text Phone GIF - Text Phone Waiting - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/curse-aukerman-comedy-fist-in-air-shaking-gif-5620082">Curse Aukerman GIF - Curse Aukerman Comedy - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://codeium.com/contact">Contact | Windsurf Editor and Codeium extensions</a>：联系 Codeium 团队以获取支持并了解更多关于我们企业级产品的信息。</li><li><a href="https://www.codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>：需要帮助？联系我们的支持团队以获取个性化协助。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1329187352442179605)** (171 条消息🔥🔥): 

> `Phi-4 模型微调、动态量化对比、Ollama 聊天模板问题、保存合并模型、CPT 训练结果` 


- **Phi-4 的微调与推理**：一位用户在拥有 16GB VRAM 的免费 Colab 实例上成功在小型数据集上微调了 Phi-4 模型。然而，在将其保存为合并模型（merged model）时遇到了问题，特别是与显存不足（OOM）错误相关的问题。
- **Ollama 聊天模板的挑战**：有用户报告在使用 Ollama 服务器时出现无限生成的现象，这可能是由于聊天模板不正确导致的。建议确保使用来自 Unsloth 训练的正确模板。
- **动态量化 vs GGUFs**：针对动态 4-bit 压缩精度与 4-k-m 等 GGUFs 的对比提出了疑问，这在最近的讨论中未被提及。强调了动态量化更适合微调和 Serving（推理服务）用途，而非本地运行。
- **训练会话状态**：用户分享了 Phi-4 正在进行的训练经验，其中一位开始使用调整后的 LoRA 参数进行新训练。如何正确保存合并模型是一个反复出现的关注点。
- **Wikipedia 数据集协作**：一位用户在 Hugging Face 上创建了一个 Wikipedia 数据集仓库，邀请他人请求特定语言并为持续学习（continuous learning）做出贡献。团队对分享资源以进行进一步训练表示了热忱。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/vision">Llama 3.2 Vision Fine-tuning with Unsloth</a>：通过 Unsloth 以 2 倍速开源微调 Meta 的 Llama 3.2 Vision、Llava、Qwen 2.5 Vision 模型！对初学者友好。</li><li><a href="https://huggingface.co/OuteAI/OuteTTS-0.3-500M">OuteAI/OuteTTS-0.3-500M · Hugging Face</a>：暂无描述</li><li><a href="https://www.youtube.com/watch?v=LPZh9BOjkQs">Large Language Models explained briefly</a>：在此深入探索：https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi 技术细节讲座：https://youtu.be/KJtZARuO3JY 这是...</li><li><a href="https://docs.unsloth.ai/basics/datasets-101">Datasets 101 | Unsloth Documentation</a>：学习创建微调数据集的所有核心要素！</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-3B-Instruct">unsloth/Llama-3.2-3B-Instruct · Hugging Face</a>：暂无描述</li><li><a href="https://huggingface.co/burgasdotpro/bgGPT-Phi-4">burgasdotpro/bgGPT-Phi-4 · Hugging Face</a>：暂无描述</li><li><a href="https://x.com/UnslothAI/status/1879942441538609583">Unsloth AI (@UnslothAI) 的推文</a>：你现在可以在 @Kaggle 上免费微调 Phi-4 了！你将学习如何：• 准备数据集 • 通过 Kaggle 的免费 GPU 训练 Phi-4 • 运行、评估并保存模型。Unsloth 微调 LLM 的速度快 2 倍...</li><li><a href="https://github.com/KellerJordan/modded-nanogpt">GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 3.14 minutes</a>：3.14 分钟内完成 NanoGPT (124M)。通过在 GitHub 上创建账号为 KellerJordan/modded-nanogpt 做出贡献。</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>：以下是我们所有 Notebook 的列表：</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/is-fine-tuning-right-for-me">Is Fine-tuning Right For Me? | Unsloth Documentation</a>：如果你在纠结微调是否适合你，请看这里！</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct">meta-llama/Llama-3.2-3B-Instruct · Hugging Face</a>：暂无描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1329410421072990249)** (1 条消息): 

> `Onnx, TensorRT 输出差异` 


- **Onnx vs TensorRT：显著的输出差异**：一位成员提出了关于其模型在 **Onnx** 上的输出与 **TensorRT** 存在显著差异的问题。
   - 他们询问其他人是否在两个框架之间的输出一致性方面遇到过类似问题。
- **关于框架差异的可能见解**：虽然没有分享更多的进一步见解，但该询问反映了在 **Onnx** 和 **TensorRT** 之间进行模型部署时的常见挑战。
   - 社区可能会探索导致观察到的不一致现象的潜在原因，如模型优化或转换问题。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1329184310422409257)** (63 messages🔥🔥): 

> `Flash Attention 2 安装问题, 在 Colab 上训练 Phi-4, 使用 VLLM 进行模型推理服务, Phi-4 的无限生成问题, 数据打包 (Data packing) 与性能顾虑` 


- **出现 Flash Attention 2 安装问题**：一位用户报告了 **Flash Attention 2** 安装损坏的问题，他们需要解决此问题以便在测试期间进行速度对比。
   - 作为回应，另一位成员建议如果需要 Colab notebooks 的测试帮助，可以直接联系。
- **在免费版 Colab 上训练 Phi-4 是可行的**：一位用户询问在谜题数据集上训练 **Phi-4** 等模型是否需要 Colab 高级账号，另一位成员确认在免费账号上已成功完成训练。
   - 然而，他们指出在保存完整模型及其配置时，免费空间有限是一个挑战。
- **在 VLLM 中部署模型的挑战**：一位用户分享了保存 **QwenVL2-7B** 模型的 **LoRA** 参数的经验，并表达了在使用 **VLLM** 提供模型服务时遇到的困难。
   - 他们发现文档不够详细，导致排查问题时面临挑战。
- **Phi-4 无休止地生成回复**：一位用户遇到了 **Phi-4** 持续生成响应而不产生序列结束符（end-of-sequence token）的问题，这可能是一个 Bug。
   - 他们正在使用 llama.cpp 进行推理，并观察到生成过程一直持续没有停止。
- **训练中数据打包（Data packing）的顾虑**：一位用户指出，在自定义数据集上使用 packing 会降低性能，并质疑使用更高的 batch size 配合更少的训练步数是否效果较差。
   - 回复强调，虽然更高的 batch size 可能会使 loss 趋于稳定，但效果仍取决于多种因素，包括 packing 中潜在的数据污染。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/unsloth/qwen-25-coder-6732bc833ed65dd1964994d4">Qwen 2.5 Coder - Unsloth 集合</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating">安装与更新 | Unsloth 文档</a>：学习如何在本地或在线安装 Unsloth。</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-upd">Unsloth 文档</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1329186825071493150)** (7 messages): 

> `从记忆到 Grokking 的转变, Grokking 现象, Unsloth 训练技术, 使用 LoRA 进行知识蒸馏` 


- **研究从记忆（Memorizing）到 Grokking 的转变**：一位成员表示希望研究人员正在检查**从记忆到 Grokking 的转变**，并认为这可能会揭示类似于生物神经元的训练方法见解。
   - *这种技术可能隐藏着* AI 模型训练新方法的秘密。
- **关于 LLM 中 Grokking 现象的 YouTube 视频**：一位成员分享了一个名为 [Activate GROKKING NOW - Performance Phase of LLMs (II)](https://youtu.be/H3OofROzlA0?si=e9b_lK1592TxgJcQ) 的视频，讨论了 LLM 中的 **Grokking** 现象。
   - 视频详细阐述了这种突然的泛化是如何在模型长时间过拟合后发生的。
- **Unsloth 的 train_on_completions 方法解析**：一位成员询问 **Unsloth** 的 **train_on_completions** 方法（仅对助手输出进行训练）是否等同于使用 hard targets 的 **response-based distillation**（基于响应的蒸馏）。
   - 这突显了对理解训练技术对模型性能影响的兴趣。
- **使用 LoRA 进行知识蒸馏**：另一位成员询问在全量微调过程中是否可以利用 **LoRA** 进行知识蒸馏，引起了参与者的好奇。
   - 一位社区成员回应表示不确定，认为这是一种可能性，但需要进一步澄清。



**提到的链接**：<a href="https://youtu.be/H3OofROzlA0?si=e9b_lK1592TxgJcQ">Activate GROKKING NOW - Performance Phase of LLMs (II)</a>：Grokking，即 AI 模型对新知识的突然泛化——发生在 LLM 长时间过拟合之后，是一个令人惊讶的现象...

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1329443821574033439)** (125 条消息🔥🔥): 

> `用于批量文本续写的 LLM API、训练 LLM 与 Token 预测、生成模型与 VAEs、Attention 机制 vs. MLP、现代 AI 中 LSTMs 的探索` 


- **寻求支持批量输出的 LLM 后端**：一位成员询问是否有提供支持批量文本续写 API 端点的 LLM 后端，并指出 **llama.cpp** 仅支持单个续写。
   - 另一位成员建议使用 **vllm** 作为潜在解决方案。
- **逐 Token 训练讨论**：对话深入探讨了 LLM 训练的复杂性，特别是模型如何在基于概率而非固定输出进行训练时预测下一个 Token。
   - 成员们强调，通过这种方法，模型学会了在多样化的回答中进行泛化，而不是死记硬背特定的序列。
- **关于生成模型中使用 VAEs 的不同意见**：一位成员主张将变分自编码器（VAEs）作为进入生成模型领域的一个易于上手的切入点，强调其简单性和轻量化特性。
   - 然而，他们建议在研究生成架构时，具备概率图模型的基础知识可能会更有裨益。
- **Transformers vs. MLP Mixers**：讨论涉及了 Transformer 架构的简洁性，将其比作将复杂问题分解为更简单的组件，类似于“三明治”。
   - 一位成员幽默地提到，尽管承认 Attention 机制具有强大的优势，但他们仍坚持使用 **MLP mixers**。
- **重新审视当代 AI 中的 LSTMs**：一位成员对 LSTMs 的现状表示好奇，认为与训练速度更快的 Transformers 相比，它们显得有些过时。
   - 其他人则考虑了 LSTMs 的潜在应用，特别是在有状态视频世界模型等场景中，表明它们在特定情况下可能仍具有相关性。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1329262512524365824)** (65 条消息🔥🔥): 

> `Modded NanoGPT Speedrun 纪录、TruthfulQA 数据集利用、BERT vs. GPT 用于分类、混合 Attention 模型、神经网络中的 Euler-Lagrange 方程`

- **Modded NanoGPT Speedrun 记录设定为 3.17 分钟**：@fern.bear 宣布了 Modded NanoGPT Speedrun 的新记录 **3.17 分钟**，打破了之前 **3.33 分钟** 的记录。关于各种技巧和改进的详细讨论可以在[此推文线程](https://x.com/hi_tysam/status/1879687807678959729)中找到。
   - 成员们还讨论了即将到来的改进，包括 **Long-Short Sliding Window Attention** 等技术，暗示在上下文处理方面会有进一步增强。
- **TruthfulQA 数据集的缺陷被揭露**：据报道，成员们成功利用了 **TruthfulQA** 数据集中的漏洞，通过几个简单的策略实现了高达 **79% 的准确率**。这一事件强调了对基准测试进行批判性审查的必要性，详见[此处](https://turntrout.com/original-truthfulqa-weaknesses)链接的详细文章。
   - 讨论还转向了 **Halueval** 的问题，指出常用数据集中的人工标注（human annotations）经常存在不准确的情况。
- **BERT 的双向注意力与 GPT 的因果注意力对比**：@kaltcit 提出了关于 BERT 中的双向注意力在 Masked Language Modeling 之外的任务中的影响问题。会议指出，虽然 **BERT** 的注意力允许更强大的表示，但它不能用于文本生成，限制了其适用性。
   - 成员们讨论了 **GPT** 的单向注意力如何导致不同的性能结果，特别是对于需要更广泛上下文的任务。
- **混合注意力模型可能占据主导地位**：在关于注意力机制的讨论中，成员们反思了结合滑动窗口和 Full Attention 等策略的**混合模型**，指出它们在超过 1M 的上下文中具有优越性。共识认为，与简单的架构相比，混合模型能有效平衡速度和上下文保留。
   - 参与者对仅仅依赖滑动窗口机制表示怀疑，认为适当的混合在长上下文任务中能提供更好的性能。
- **确保神经网络遵循 Euler-Lagrange 方程**：对话涉及了在不使用高阶 Autodiff 方法的情况下，确保神经网络遵循 **Euler-Lagrange 方程** 的挑战，并提出了实现输出积分形式架构的想法。建议包括利用模型输出来确保无旋（curl-free）条件，以正确表示标量势。
   - 提出了一种创新的方法，让模型输出其导数，从而建立一个从分析上确保符合物理定律约束的框架。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sakana.ai/transformer-squared/">未找到标题</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2411.04434">Scaling Laws for Pre-training Agents and World Models</a>: 具身 Agent 的性能已证明可以通过增加模型参数、数据集大小和计算量来提升。这在从机器人到视频游戏的各个领域都得到了验证，当生成...</li><li><a href="https://x.com/Turn_Trout/status/1879710659904254081">来自 Alex Turner (@Turn_Trout) 的推文</a>: Mark Kurzeja 和我在隐藏问题的情况下利用了多选题 TruthfulQA 数据集的弱点！几条简单的经验法则就达到了 79% 的准确率。即使是备受推崇的基准测试也可能存在缺陷。...</li><li><a href="https://x.com/hi_tysam/status/1879687807678959729">来自 Fern (@hi_tysam) 的推文</a>: 新的 NanoGPT 训练速度记录：在 8xH100 上，3.17 分钟内达到 3.28 FineWeb 验证损失。之前的记录（重现）：3.32 分钟。有很多变化！- 新的 Token 相关 lm_head 偏置 - 融合了多个操作 - Multi...</li><li><a href="https://github.com/KellerJordan/modded-nanogpt/pull/71">Long-Short Sliding Window Attention (提升 3.2 秒或 0.053 分钟) 由 leloykun 提交 · Pull Request #71 · KellerJordan/modded-nanogpt</a>: 目前，我们在所有层中以相同的速率预热滑动窗口注意力的上下文长度。这次尝试在某些层中以不同的方式预热上下文长度。这导致了...</li><li><a href="https://github.com/facebookresearch/coconut">GitHub - facebookresearch/coconut: 训练大语言模型在连续潜空间中进行推理</a>: 训练大语言模型在连续潜空间中进行推理 - facebookresearch/coconut</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/6d62a69cb5db963f998c486af6efee43fca63dd3/docs/task_guide.md?plain=1#L57>),">lm-evaluation-harness/docs/task_guide.md at 6d62a69cb5db963f998c486af6efee43fca63dd3 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 Few-shot 评估的框架。- EleutherAI/lm-evaluation-harness</li><li><a href="https://www.nature.com/articles/s41467-024-55628-6">大语言模型缺乏可靠医学推理所需的必要元认知 - Nature Communications</a>: 大语言模型在医学考试中展示了专家级的准确性，支持了将其纳入医疗场景的潜力。在这里，作者揭示了它们的元认知能力处于...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1329502969158434826)** (16 条消息🔥): 

> `PDE 基础模型, 模型中的 Scaling Laws, 隐式 vs 显式求解器, 模型输出 vs 训练数据` 


- **PDE 模型与 Scaling Laws 解析**: 成员们讨论了为什么 **PDE 基础模型** 具有与 **LLM** 类似的 Scaling Laws，将其与记忆基准联系起来，但质疑其如何应用于 **PDE** 模型。
   - *一位成员建议 PDE 模型应该直接“悟透”（grokking）系统动力学，而不是仅仅依赖于训练数据的幂律概率。*
- **隐式求解器可能并非至关重要**: 对话转向模型学习**隐式求解器**是否必要，一些成员对其重要性表示怀疑。
   - *有人担心模型可能没有彻底掌握系统动力学，从而对学习机制产生怀疑。*
- **时间步稳定性在求解器中的作用**: 强调了对于**显式求解器**，稳定性和能量守恒取决于解的“声速”以及约束时间步的长度尺度。
   - *相比之下，隐式求解器没有这些限制，从而导致对模型学习的不同影响。*
- **模型输出可能超过训练数据的准确性**: 一位成员指出，模型的输出可能比用于生成它的训练数据更接近 *Ground Truth*，这引发了对其发生机制的好奇。
   - *使用低分辨率数据可能允许模型估计一个更接近真实值的平均值，尽管论文在数据生成方法上缺乏清晰度。*


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1329253449040134174)** (8 messages🔥): 

> `MATH DMCA 下架, MATH 数据集影响, YAML Quickstarter 更新` 


- **MATH 数据集遭遇 DMCA 下架**：据 [推文](https://x.com/tmkadamcz/status/1879584048429105238) 报道，**Hendrycks MATH** 已收到 DMCA 下架通知，导致该数据集被禁用。讨论集中在这一行动的影响以及数据集的来源（归功于 **aops**）。
   - *“他们一直公开承认题目是从那里获取的。”*
- **对 MATH 数据集丢失的担忧**：成员们表示，**MATH** 数据集的丢失可能会对社区产生重大影响，甚至可能超过 **The Pile** 或 **Books 3** 等其他著名数据集。
   - *“我认为其影响甚至比 Pile、Books 3 或 Book Corpus 还要大。”*
- **建议更新 YAML Quickstarter**：一名成员建议根据文档将空白 YAML 更新为 **quickstarter YAML**，并收到了关于其实用性的积极反馈。
   - *“那实际上会非常有帮助！”*
- **关于合理使用（Fair Use）和 DMCA 的讨论**：一些成员讨论了团队是否会尝试根据 **fair use** 挑战 DMCA 通知。
   - 一位成员对 **Hugging Face** 能否主张合理使用表示怀疑，强调了他们作为分发者的角色。
- **Git 仓库状态**：尽管有 DMCA 通知，**MATH** 的 Git 仓库和 **tar 文件** 链接仍然可以访问。
   - 这引发了关于在法律挑战中资源持续可用性的疑问。



**提到的链接**：<a href="https://x.com/tmkadamcz/status/1879584048429105238">Tom Adamczewski (@tmkadamcz) 的推文</a>：Hendrycks MATH 刚刚收到了 DMCA 下架通知。该数据集目前已被禁用。https://huggingface.co/datasets/hendrycks/competition_math/discussions/5

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1329202599164907570)** (2 messages): 

> `Deepspeed 中的 Zero stages, 模型并行挑战, 30b 模型性能优化, AMD MI250x GPU 性能` 


- **Zero stages 不兼容问题**：一名成员询问为什么 **zero stages 2 和 3** 与 **Deepspeed** 中的模型并行和流水线并行都不兼容，并建议根据 [training.py 代码](https://github.com/EleutherAI/gpt-neox/blob/f7a5a6f9da47de4d4d7cdf776c0832b257f329ef/megatron/training.py#L958-L973) 仅关闭流水线并行就足够了。
   - *他们担心无法使用模型并行会导致 Deepspeed 在大型模型训练中失效。*
- **30b 模型训练性能困境**：同一位成员分享了在 **512 个 AMD MI250x GPU** 上最大化 **30b 模型训练** 性能的困难，目标是结合使用 **Deepspeed stage 2** 和模型并行。
   - *目前，他们每个逻辑单元仅达到 **28 TFLOPs**，远低于 AMD 声明的预期性能。*



**提到的链接**：<a href="https://github.com/EleutherAI/gpt-neox/blob/f7a5a6f9da47de4d4d7cdf776c0832b257f329ef/megatron/training.py#L958-L973)">gpt-neox/megatron/training.py (位于 f7a5a6f) · EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自回归 Transformer - EleutherAI/gpt-neox

  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1329184978311057479)** (1 messages): 

> `Bolt 项目标题编辑` 


- **Bolt 现已支持编辑标题**：**Bolt** 的最新更新允许用户直接编辑项目标题，增强了在列表中的项目发现能力。
   - *根据 [Stackblitz Twitter](https://x.com/stackblitz/status/1879625416706785365) 的公告，这一功能使查找项目变得更加容易。*
- **增强的项目组织功能**：修改项目标题的新功能有助于简化 **Bolt** 内的项目管理，促进工作区整洁。
   - 用户现在可以轻松地使项目标题与其内容保持一致，以便更好地进行组织。



**提到的链接**：<a href="https://x.com/stackblitz/status/1879625416706785365">StackBlitz (@stackblitz) 的推文</a>：📢 Bolt 最新更新：你现在可以更改项目标题了 —— 让你在项目列表中更容易找到它！

  

---

### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1329277705711128587)** (5 条消息): 

> `聊天历史快照系统、日期输入问题、新用户介绍、GitHub 仓库交互` 


- **聊天历史快照系统已实现**：thecodacus 提交了一个名为 *'feat: restoring project from snapshot on reload'* 的 Pull Request，引入了聊天历史快照系统，允许用户在重新加载时恢复之前的状态。PR 详情可以在[这里](https://github.com/stackblitz-labs/bolt.diy/pull/444)查看。
   - 此实现旨在保持用户交互的连续性，并确保相关的文件系统状态得以保留。
- **日历日期输入问题**：一位用户报告称，部分用户在从日历弹出窗口选择日期后，提交时日期会发生变化。用户请求提供排查建议，并希望构建一个能准确获取所选日期的 Prompt。
   - 另一位成员建议检查当前日期输入实现中的 Bug 或逻辑问题，以防止这些差异。
- **欢迎新用户**：一位新成员在频道中介绍了自己，表达了加入社区的兴奋之情。虽然没有提供更多信息，但其他成员随后表达了欢迎。
   - 鼓励与新用户互动，以营造支持性的环境。
- **针对 GitHub 仓库使用的 Bolt Prompt 优化**：有人询问 Bolt 是否有可能利用特定的 GitHub 仓库来实现功能，以及是否有计划增强对仓库 README 的检索增强生成 (RAG) 使用。这将旨在确保正确且高效地使用这些仓库。
   - 关于该话题的进一步讨论可能会明确未来与仓库交互相关的增强功能。



**提到的链接**：<a href="https://github.com/stackblitz-labs/bolt.diy/pull/444">feat: restoring project from snapshot on reload by thecodacus · Pull Request #444 · stackblitz-labs/bolt.diy</a>：添加聊天历史快照系统。概述：此 PR 为聊天历史引入了快照系统，允许恢复之前的聊天状态及其关联的文件系统状态。这...

  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1329183087241859102)** (178 条消息 🔥🔥): 

> `Git 支持讨论、Bolt 中的 Shadcn 默认设置、会话与 Token 问题、部署挑战、Stripe 集成问题` 


- **即将举行的 Git 支持会议**：在今天的 office hours 中提到，稍后将举行会议讨论 **Git 支持**，可能在未来 **2-4 周**内发布。
   - 成员们对该功能的潜力表示期待。
- **Shadcn 成为 Bolt 的默认选项**：一位用户询问 Bolt 是否已将使用 **Shadcn** 作为默认设置，正如最近的聊天中所提到的。
   - 这一变化紧随早期项目中对 **Headless UI** 的使用。
- **Token 与会话挑战**：参与者报告了 **会话 Token 管理** 的问题，据称 Prompt 消耗了过多的 Token。
   - 一位成员指出单个命令使用了 **400 万个 Token**，引发了质疑并呼吁支持团队进行调查。
- **遇到的部署问题**：几位用户在部署项目时面临持续的挑战，由于项目体积过大导致对项目稳定性的担忧。
   - 有人建议将资产迁移到 **Amazon S3**，以缓解一些与 **部署体积** 相关的问题。
- **Stripe 集成咨询**：用户在集成 **Stripe** 时遇到了持续存在的问题，在实现过程中感到困难。
   - 反馈指向需要更清晰的开发环境，并可能需要解决更新中提到的访问问题。



**提到的链接**：<a href="https://tenor.com/view/apes-together-strong-0p1sf-gif-20906166">Apes Together Strong 0p1sf GIF - Apes Together Strong 0p1sf - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1329195107764928642)** (133 条消息🔥🔥): 

> `Swarm vs A1111，Twitter 诈骗警报，图像生成指标，社区许可证归属，使用 Stable Diffusion 进行按需打印` 


- **Swarm UI 的受欢迎程度超过 A1111**：成员们讨论了他们偏好的 AI 图像生成界面，**Swarm** 因其活跃的开发和文档而受到关注，而 **A1111** 自 7 月以来一直没有更新。
   - 一位用户指出 Swarm 的后端对专门任务很有帮助，并称赞了其开发者活跃的社区。
- **Twitter 账号被盗并发布诈骗信息**：关于与 @StabilityAI 相关的 Twitter 账号被盗的担忧蔓延，该账号发布了有关代币发行的虚假帖子，提示用户不要点击链接。
   - 社区成员迅速向他人发出诈骗警报，并引用了过去人们被类似欺诈活动利用的经验。
- **关于图像生成指标的问题**：一位用户询问了使用 **Stable Diffusion** 模型生成图像的指标，特别是关于每秒迭代次数（iterations per second）等时间指标。
   - 其他人分享说，一些 UI 会显示总时间和步数，而具体细节可以在元数据或其他 UI 元素中找到。
- **社区许可证下的归属要求**：有人寻求关于在使用 Stability AI 社区许可证生成的图像时是否需要注明出处的澄清。
   - 一位成员指出，虽然注明出处是有益的，但输出内容通常可以免费使用，除非涉及需要授权的商业应用。
- **将 SD 用于按需打印业务**：一位按需打印商店的所有者表示有兴趣利用 **Stable Diffusion** 创建大尺寸图像，并询问了如何调整打印分辨率。
   - 通过私信提供了指导，以协助将 AI 用于商业目的，强调了 Stable Diffusion 在此背景下的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/dango233max/status/1879734940264481006">来自 Dango233 (@dango233max) 的推文</a>: 刚刚联系了我的 SAI 朋友。这是个诈骗！！！！@StabilityAI 的 X 账号被盗了。不要相信它！</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">stabilityai/stable-diffusion-xl-base-1.0 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1329198246916263967)** (62 messages🔥🔥): 

> `更新后的 Lint 错误、DeepSeek 的使用与替代方案、模型性能担忧、MOE 与 GPU 效率、AI 安全层` 


- **更新后 Lint 错误的困惑**：多位成员报告称，自最近一次更新以来遇到了大量的 **lint 错误**，其中一人询问这是否是一个普遍问题。
   - 另一位成员澄清说，新版本只是显示了 linter 的输出，并没有导致更多错误，这表明可能存在误解。
- **探索 DeepSeek 的替代方案**：一位用户提到由于性能问题，已从 **DeepSeek** 切换到 **Sonnet**，并表示后者更优。
   - 其他人询问了硬件要求，普遍认为 DeepSeek 需要巨大的资源（如 **500GB VRAM**）才能发挥有效性能。
- **对模型性能的评价褒贬不一**：有人对 **DeepSeek3** 最近性能下降表示担忧，促使用户寻找其他 AI 模型提供商。
   - 讨论了一些具有性价比的替代方案，例如 **Hyperbolic**，其价格显著低于 DeepSeek，仅为 **$0.25/mtok**。
- **讨论 MOE 模型的效率**：讨论集中在 **MOE** (Mixture of Experts) 模型如何通过激活权重子集来优化性能并节省资源。
   - 据信，如果管理得当，批处理（batching）可以降低成本并提高模型运行效率。
- **对 AI 安全层的需求**：有人建议 **aider** 加入一个安全层模型，在将数据发送给提供商之前过滤敏感数据。
   - 成员们承认了确保与 AI 模型交互安全的重要性，以防止无意的数据泄露。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1ho0w52/deepseek_does_not_need_5_hours_to_generate_1/">Reddit - 深入了解一切</a>：未找到描述</li><li><a href="https://youtu.be/rx0wP9k4wGM?si=BAt3ZUfjGRB6lX3Q">SakanaAI 发布 "Transformer Squared" - 测试时学习 (Test Time LEARNING)</a>：加入我的新闻通讯以获取定期 AI 更新 👇🏼https://forwardfuture.ai 我的链接 🔗👉🏻 订阅：https://www.youtube.com/@matthew_berman 👉🏻 Twitter：https:/...</li><li><a href="https://github.com/Aider-AI/aider/actions/runs/12814806715/">feat: 使用 /subtree 命令更改当前子树 · Aider-AI/aider@29a4a67</a>：aider 是你终端里的 AI 配对编程助手。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/aider/actions/runs/12814806715/job/35732104905?pr=2881):">feat: 使用 /subtree 命令更改当前子树 · Aider-AI/aider@29a4a67</a>：aider 是你终端里的 AI 配对编程助手。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/aider/actions/runs/12801873727/job/35692059901#step:5:1">feat: 使用 /subtree 命令更改当前子树 · Aider-AI/aider@7ab00bb</a>：aider 是你终端里的 AI 配对编程助手。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/aider/issues/2777">生成用于微调模型的数据 · Issue #2777 · Aider-AI/aider</a>：Issue：我经常会拒绝编辑并告诉模型修改某些内容。理想的情况是有一个 /fine-tune 命令，它可以：指示模型考虑修正，但生成...
</li>
</ul>

</div>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1329194044634566689)** (64 messages🔥🔥): 

> `Aider API 日志记录, Git Commit 问题, CEDARScript 集成, 聊天会话管理, 用于代码探索的 Agentic 工具` 


- **关于 Aider API 调用日志记录的请求**：用户询问了 Aider 记录 LLM API 调用的方法，因为聊天历史记录无法捕获完整的 API 调用。
   - 目前似乎还没有现成的解决方案，引发了关于在本地运行 Aider 以记录此类信息的讨论。
- **Aider Commit 功能的问题**：有用户报告了 Aider 在完成更改后无法向其 git 仓库提交的问题。
   - 建议使用 `--auto-commit` 选项，而另一位用户分享说 architect 模式有助于提交更改。
- **CEDARScript 与 Aider 的集成**：一名成员分享了一个关于 CEDARScript 的 GitHub 链接，该工具允许 Aider 使用 CEDARScript 作为编辑格式。
   - 随后讨论了是否可以将此功能合并到 Aider 中，但对其定量收益尚未达成共识。
- **在 Aider 中保存聊天会话**：关于 Aider 如何保存聊天会话的担忧，因为现有方法仅保存加载了哪些文件，而不包含完整的聊天历史。
   - 机器人澄清说，虽然目前无法保存不同的聊天历史，但使用 `/read-only` 命令可以帮助管理混乱。
- **用于代码探索的多样化工具**：一位用户列出了各种用于探索大型代码库的 agentic 工具，强调了它们的灵活性和集成能力。
   - 他们还分享了自己的代码探索工具，并讨论了使用 AG2 的 swarm 编排方法来增强该工具的计划。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/2025/01/15/uv.html">使用 uv 作为安装程序</a>：可靠地打包和分发 Python CLI 工具很困难。Aider 以新颖的方式使用 uv，以便轻松安装 aider CLI 及其依赖项和 Python 3.12。全部都在隔离的环境中。</li><li><a href="https://aider.chat/docs/git.html">Git 集成</a>：Aider 与 git 紧密集成。</li><li><a href="https://bw2.github.io/ConfigArgParse/configargparse.ArgumentParser.html#__init__):">configargparse.ArgumentParser</a>：未找到描述</li><li><a href="https://github.com/CEDARScript/cedarscript-integration-aider?tab=readme-ov-file#installation">GitHub - CEDARScript/cedarscript-integration-aider: 允许 Aider 使用 CEDARScript 作为编辑格式</a>：允许 Aider 使用 CEDARScript 作为编辑格式。通过在 GitHub 上创建账号为 CEDARScript/cedarscript-integration-aider 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/aider/pull/2877">docs: 在编辑错误故障排除指南中添加 architect 模式章节，由 golergka 提交 · Pull Request #2877 · Aider-AI/aider</a>：未找到描述</li><li><a href="https://docs.ag2.ai/notebooks/agentchat_swarm_enhanced">使用 AG2 增强 Swarm 编排 - AG2</a>：未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1329581141849669642)** (1 messages): 

> `Helicone LLM 可观测性, LLM 安全特性, 使用 Docker 的本地部署` 


- **Helicone 发布开源 LLM 可观测性工具**：Helicone 推出了一个[开源可观测性平台](https://github.com/Helicone/helicone)，用于 LLM，使用户只需一行代码即可进行监控、评估和实验。
   - 该平台突出了请求和成本的 **track&trace**（追踪与溯源）、**LLM 安全层**以及额外的指标跟踪等功能。
- **提供本地和云端部署选项**：Helicone 可以通过 **docker-compose** 在本地运行，尽管他们建议使用云端版本。
   - 这种设置支持 **caching**（缓存）和**自定义速率限制**等功能，以增强性能。



**提到的链接**：<a href="https://github.com/Helicone/helicone">GitHub - Helicone/helicone: 🧊 开源 LLM 可观测性平台。一行代码即可监控、评估和实验。YC W23 🍓</a>：🧊 开源 LLM 可观测性平台。一行代码即可监控、评估和实验。YC W23 🍓 - Helicone/helicone

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1329180078713602076)** (52 messages🔥): 

> `Nous Research 隶属关系、周边资金、微调技术、LLAMA 1B QLoRA 训练、社区参与` 


- **Nous Research 是一个私有实体**：成员们澄清说 **Nous Research** 与任何政府或学术机构没有隶属关系，作为一个纯粹的私有组织运营。
   - 讨论重点在于开放性，并探讨了其他 AI 公司可能存在的政府背景。
- **通过周边和私募股权融资**：据了解，**Nous Research** 的资金主要来自周边销售和私募股权，尽管周边收入相对较小。
   - 成员们还讨论了在周边订单中包含贴纸的话题，并对此表示出兴趣。
- **微调技术与建议**：在关于微调的讨论中，一位成员分享了关于 RL 等训练技术重要性的见解，以及输出准确性对多样化设置的依赖。
   - 其他人提到了有效 Prompt 设计的重要性，以及在训练过程中与模型的交互如何增强准确性。
- **关于 LLAMA 1B QLoRA 训练图表的反馈**：成员们审查了 **LLAMA 1B QLoRA** 的训练图表，指出了对小数据集和训练步数不足的担忧。
   - 讨论围绕适应度分数（fitness scores）的计算以及在评估过程中简化性能指标的偏好展开。
- **社区互动与闲聊**：多位成员进行了轻松的交流，包括问候和对当天情绪的幽默评论。
   - 成员们互相鼓励进行讨论或咨询，展示了良好的社区参与度。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1329191150220873778)** (35 messages🔥): 

> `GrokAdamW 与 Ortho Grad、LLM 记忆、过程奖励模型 (PRMs)、基于 PDF 数据的聊天机器人架构` 


- **GrokAdamW 与 Ortho Grad 合并**：成员们讨论了将 **GrokAdamW** 与 **Ortho Grad** 结合的概念可行性，指出 GrokAdamW 提供了更好的 Loss 指标，尽管 Ortho Grad 可能存在缺点，但这种结合仍具有优势。
   - 重点介绍了 *Eric Hartford 开发的 GrokAdamW*，并分享了 GitHub 链接以供参考。
- **关于 LLMs 记忆文本的研究**：发起了一场关于 **LLMs** 记忆文本能力的讨论，建议探索 Anthropic 关于**可解释性 AI**（explainable AI）的研究以获得更深入的见解。
   - 一位参与者提到正在进行与该主题相关的字典学习（dictionary learning）实验，反映了对该材料的亲身实践。
- **过程奖励模型（PRMs）的挑战**：分享了关于开发 **Process Reward Models (PRMs)** 复杂性的见解，特别是关于数据标注和评估方法及其对性能的影响。
   - 提到 **Qwen 团队**在 PRMs 方面的工作，指出了有用的文档和对开发过程的见解。
- **针对 PDF 数据的聊天机器人架构**：一位成员评估了使用 **GPT-2** 构建基于 PDF 数据的 RAG 聊天机器人的情况，表示面临上下文窗口（context window）过小和输出无意义的挑战。
   - 其他人的反馈建议，尝试用 GPT-2 完成此任务可能不可行，建议考虑使用更大的模型以提高性能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.07301">The Lessons of Developing Process Reward Models in Mathematical Reasoning</a>：过程奖励模型 (PRMs) 正在成为大语言模型 (LLMs) 数学推理中过程监督的一种有前景的方法，旨在识别和减轻中间错误...</li><li><a href="https://www.anthropic.com/research/mapping-mind-language-model">Mapping the Mind of a Large Language Model</a>：我们已经确定了数百万个概念在 Claude Sonnet（我们部署的大语言模型之一）内部是如何表示的。这是有史以来第一次对现代生产级大模型内部的详细观察...</li><li><a href="https://buttondown.com/ainews/archive/">AI News</a>：我们总结了顶级的 AI Discord + AI Reddit + AI X/Twitter，并每天为您发送汇总！查看存档以获取示例。“这是我每天花费的最具杠杆作用的 45 分钟” - Soumith...</li><li><a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>：通过在 GitHub 上创建账户来为 cognitivecomputations/grokadamw 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1329546499566342227)** (3 messages): 

> `新架构设计, Recurrent models 与 Attention, Neural long-term memory` 


- **新架构出现**：一名成员宣布引入了一种**新架构**，称其非常有趣，并提供了论文的 [PDF 链接](https://arxiv.org/pdf/2501.00663v1)。
   - 另一名成员指出，将 URL 中的 'pdf' 更改为 'abs' 即可生成论文摘要，并分享了 [摘要](https://arxiv.org/abs/2501.00663v1) 的链接。
- **探索 Recurrent models 与 Attention**：该论文的摘要讨论了关于如何有效利用 **Recurrent models** 和 **Attention** 的广泛研究工作，强调了它们在建模依赖关系方面的优势和局限性。
   - 论文指出，虽然 Attention 可以捕获所有 token 之间的依赖关系，但它会产生二次方成本（quadratic cost），这限制了保证准确性下的上下文长度。
- **创新的 Neural Long-Term Memory 模块**：该研究引入了一个 **Neural long-term memory** 模块，通过学习记忆历史上下文，增强了 Attention 处理当前上下文的能力。
   - 这种方法支持快速并行训练并保持快速推理（inference），旨在平衡内存效率与准确的依赖关系建模。



**提到的链接**：<a href="https://arxiv.org/abs/2501.00663v1">Titans: Learning to Memorize at Test Time</a>：十多年来，关于如何有效利用 Recurrent models 和 Attention 进行了广泛的研究。虽然 Recurrent models 旨在将数据压缩到固定大小的备忘录中...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1329241405167636500)** (2 messages): 

> `GrokFast 优化器, Orthograd 优化器, Coconut 仓库` 


- **GrokFast 优化器的稳定性问题**：许多用户在 LLM 训练期间难以通过 **GrokFast 优化器** 获得稳定性，表明它可能不够可靠。
   - 这引发了社区的共同情绪，强调需要更好的替代方案。
- **Orthograd 优化器作为替代品**：**Orthograd 优化器** 似乎是 **torch SGD** 或 **AdamW** 的一个封装（wrapper），为用户提供了一个潜在的即插即用替代方案。
   - 随着这一进展，人们希望用户能分享更多尝试这种新优化器的经验和结果。
- **Coconut GitHub 仓库发布**：来自 Facebook Research 的 [Coconut 仓库](https://github.com/facebookresearch/coconut) 专注于训练 LLM 在连续潜空间（continuous latent space）中进行推理。
   - 这种创新方法旨在增强模型的推理能力，标志着 AI 研究迈出了新的一步。



**提到的链接**：<a href="https://github.com/facebookresearch/coconut">GitHub - facebookresearch/coconut: Training Large Language Model to Reason in a Continuous Latent Space</a>：在连续潜空间中训练大语言模型进行推理 - facebookresearch/coconut

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1329546499566342227)** (3 messages): 

> `新神经架构, Recurrent Models 与 Attention, Neural Long-Term Memory 模块` 


- **令人兴奋的新神经架构发布**：一名成员分享了一个[新架构](https://arxiv.org/pdf/2501.00663v1)论文的链接，并指出它看起来很有趣。
   - 另一名成员提到，将 URL 中的 'pdf' 更改为 'abs' 即可访问论文摘要。
- **Recurrent Models 与 Attention 的影响**：论文讨论了 **Recurrent models** 和 **Attention** 的演变，强调了它们在建模依赖关系方面的优势和局限性。
   - 论文断言，虽然 Attention 允许准确的依赖关系建模，但它会产生限制上下文长度的二次方成本。
- **引入 Neural Long-Term Memory 模块**：该研究提出了一种 **Neural long-term memory 模块**，旨在通过记忆历史上下文来增强 Attention 机制。
   - 这种方法承诺了快速的可并行化训练和高效推理，同时平衡了短期和长期信息。



**提到的链接**：<a href="https://arxiv.org/abs/2501.00663v1">Titans: Learning to Memorize at Test Time</a>：十多年来，关于如何有效利用 Recurrent models 和 Attention 进行了广泛的研究。虽然 Recurrent models 旨在将数据压缩到固定大小的备忘录中...

  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1329194292224327861)** (13 条消息🔥): 

> `新手作家工作坊, 数字病理学 Groovy 脚本, NotebookLM 中的交互性, 播客分享, Notebook 可用性反馈` 


- **新手作家工作坊探索 Mists**：一场针对新手作家 **Roseline K Marie** 的讨论深入探讨了她处女作小说的早期草稿以及 **Mists** 在情节中的作用。
   - 参与者分享了*疯狂的理论*，并试图评估他们的微妙伏笔是否有效。
- **数字病理学 Groovy 脚本成功案例**：一位成员表示惊讶，在尝试通过论坛帖子寻找无果后，NotebookLM 提供了一个用于处理**图像标注**的**功能性脚本**。
   - 他们指出，在将需求输入 Notebook 后，项目节省了大量时间。
- **NotebookLM 交互模式引发关注**：随着新学期的开始，一位成员兴奋地开始将他们的模块资源加载到 NotebookLM 中，并尝试使用 **Interactive Mode**（交互模式）。
   - 他们通过截图分享了进度，并表示已为即将到来的课程做好准备。
- **希望建立播客广告频道**：一位成员请求创建一个专门用于**播客推广**的新频道，以保持对**使用案例**讨论的专注。
   - 这一建议突显了社区内内容分享需要更好的组织管理。
- **Notebook 可用性与 Prompt 分享**：成员分享了关于 NotebookLM 在文档格式方面的局限性反馈，特别是它不直接接受 **docx** 或 Google 文档。
   - 成员们建议也分享 Prompt，并认可了探索 Notebook 功能的协作本质。



**相关链接**: <a href="https://player.captivate.fm/show/406b155f-62af-46d5-9a27-17766ca55b91">轻松收听 NotebookLM ➡ Token Wisdom ✨</a>：快速免费地收听 NotebookLM ➡ Token Wisdom ✨！

  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1329180466514886687)** (73 条消息🔥🔥): 

> `NotebookLM Plus 访问权限, 播客功能增强, 来源管理, Google Workspace 许可, 新工具集成` 


- **澄清 NotebookLM Plus 访问权限**：几位成员对 Google Workspace 计划中 NotebookLM Plus 及其相关功能的访问权限表示困惑，特别是关于从旧许可转换的问题。
   - 会议澄清了现有功能不会被设为付费墙，并将包含在 Workspace 产品中，无需额外费用。
- **播客生成挑战**：用户报告了使用多个来源生成播客时的困难，并建议为每个来源创建单独的 Notebook 以方便输出。
   - 该问题的一个解决方法包括取消勾选不需要的来源，尽管关于这是否属于 Plus 功能的确认仍不明确。
- **对播客质量的担忧**：关于播客功能的反馈强调了反馈音（back channeling）、主持人互动不一致以及影响专业输出的音频质量问题。
   - 成员们讨论了可能有助于简化播客生成并提高连贯性的潜在指令。
- **来源上传的限制**：目前还没有将来源批量上传到 Notebook 的功能，这让希望一次导入多个 URL 的用户感到沮丧。
   - 建议成员手动添加来源或作为单个条目上传，同时期待未来有更高效的选择。
- **自定义播客输出**：用户寻求自定义播客的音频输出，但注意到目前大多数功能都围绕基础摘要展开，缺乏详细的指令选项。
   - 通过专注于特定来源来定制播客的能力可以增强其在教育用途上的可用性。


<div class="linksMentioned">

<strong>相关链接</strong>:

<ul>
<li>
<a href="https://workspace.google.com/blog/product-announcements/empowering-businesses-with-AI">为每家企业赋能 AI 驱动的工作未来 | Google Workspace 博客</a>：Google AI 的精华现已包含在 Workspace Business 和 Enterprise 计划中，为您提供 Gemini 和 NotebookLM Plus 等 AI 功能，无需插件。</li><li><a href="https://chrisbora.substack.com/p/boras-law-intelligence-scales-with?r=aszci">Bora 定律：智能随约束而非算力扩展</a>：这是一篇探讨人工智能发展中新兴原则的工作论文。</li><li><a href="https://support.google.com/a/answer/181865#zippy=%2Cturn-services-on-or-off-for-users">为用户开启或关闭额外的 Google 服务 - Google Workspace 管理员帮助</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1329178982276661248)** (1 条消息): 

> `Minimax-01, Needle-In-A-Haystack 测试, 在 Discord 上申请模型` 


- **Minimax-01 以创纪录的上下文长度发布**：新模型 **Minimax-01** 现已上线，它是首个在惊人的 **4M** 上下文长度下通过 **Needle-In-A-Haystack 测试** 的开源 LLM。更多详情请访问 [OpenRouter 页面](https://openrouter.ai/minimax/minimax-01)。
   - *如需申请该模型的访问权限*，请访问我们的 [Discord](https://discord.gg/fVyRaUDgxW)。
- **图像分析更新**：关于 **Minimax-01** 的公告中附带了一张图片，为该模型提供了视觉参考。该图片包含了与发布详情相关的分析内容。
   - 有关该图片的进一步见解可在相关的 [Discord 附件](https://cdn.discordapp.com/attachments/1092729520181739581/1329178982020812851/image.png?ex=678ab764&is=678965e4&hm=321979ba1263cb2992785532f4e5fdb3b644d91db5662473db7c82d3bfd8462a&) 中找到。



**提到的链接**：<a href="https://openrouter.ai/minimax/minimax-01>">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格。

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1329188916334563459)** (85 条消息🔥🔥): 

> `Minimax 模型性能, DeepSeek 问题, OpenRouter 区域限制, Gemini flash 模型错误, 活动页面功能` 


- **Minimax 模型评估引发关注**：用户对新模型 **Minimax** 在开发者任务中的表现感到好奇，尤其是与 **DeepSeek** 等现有选项的对比。
   - 讨论指出，虽然有些人预期其表现仅为尚可，但 **humaneval** 等已公布的评分可能值得参考。
- **DeepSeek 遭遇延迟**：成员们报告了 **DeepSeek** 持续存在的问题，包括延迟和提供商可靠性，特别是在使用高峰时段。
   - 用户讨论了故障排除策略，包括检查提供商错误以及通过调整 API 设置进行潜在修复。
- **OpenRouter 的区域限制被披露**：据确认，**OpenRouter** 一段时间以来一直在执行符合 **OpenAI** 和 **Anthropic** 政策的区域限制。
   - 这一披露引发了关于这些限制的影响以及用户应对经验的讨论。
- **Gemini 模型端点更改引发混乱**：关于 **Gemini flash 2.0** 模型更新的消息表明其端点发生了变化，导致用户在尝试访问服务时出现意外错误。
   - 受影响的用户分享了解决方案，包括调整隐私设置以解决端点访问问题。
- **活动页面功能受到质疑**：一位用户对 **activity page**（活动页面）提出了质疑，该页面似乎为不同的 API key 显示了相同的图表，从而导致了混乱。
   - 澄清显示，该页面目前汇总了所有交易而不作区分，这引发了关于其设计和实用性的辩论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1879908197420675322">来自 OpenRouter (@OpenRouterAI) 的推文</a>：由 @MiniMax__AI 开发的 Minimax-01 现已上线：这是一款低成本、456B 参数的多模态开源 LLM。它是首个在高达 4M 的上下文下通过原生 Needle-In-A-Haystack 测试的模型：</li><li><a href="https://meowapps.com/ai-engine/">AI Engine</a>：为 WordPress 添加 AI 功能。包括聊天机器人、表单、Copilot、内容生成等等！</li><li><a href="https://openrouter.ai/docs/provider-routing">提供商路由 | OpenRouter</a>：在多个提供商之间路由请求
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1329214489933123696)** (19 条消息🔥): 

> `Command R+ 编程语言, 使用 Stripe 进行支付处理, Cohere 模型代理` 


- **用户对 Command R+ 的语言训练感到好奇**：一位用户询问了 **Command R+** 主要针对哪些编程语言进行训练，以及是否有资源可以查阅此信息。
   - 另一位成员建议用户可以通过 API 访问该模型，以测试其对特定用例的适用性。
- **支付处理说明**：一位成员指出，支付是通过 **Stripe** 处理的，并可以选择使用 **OpenRouter** 以获得额外功能。
   - 此外，他们提到 **OpenRouter** 代理了所有可用的 **Cohere models**。


  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1329263875568435291)** (13 messages🔥): 

> `持续更新 Command R 模型、Rerank 3.5 编程能力、Embedding 模型限制、Rerank 的 Prompt 构建` 


- **持续更新 Command R 模型**：一位成员建议，现有的 **Command R** 和 **R+** 模型应使用最新数据和微调技术持续更新，类似于 08-2024 更新版本。
   - 另一位成员指出，以这种方式更新模型最终会导致开发出一个新模型。
- **Rerank 3.5 在编程任务中表现出色**：一位成员强调 **Rerank 3.5** 特别擅长编程，并在 **Python**、**JavaScript** 和 **C++** 等常用语言上进行了训练。
   - 他们指出，虽然该模型非常有效，但某些特定的用例并未包含在其训练中。
- **Embedding 模型的挑战**：一位成员对 **Embedding** 模型的局限性表示担忧，指出更新的唯一选择是重新对所有数据进行 **Embedding**，这是一项巨大的任务。
   - 他们澄清说，目前没有有效的方法将 **Embedding** 从一个模型版本迁移到另一个版本，这导致特定 **Embedding** 模型的使用周期较长。
- **理解 Rerank 模型的偏差**：一位成员提出了关于 **Rerank 3.5** 的问题，特别注意到当提供更多文档时准确性会下降，并且模型更偏向于语义文档而非词法文档。
   - 他们询问这些现象是否是 **Rerank** 模型的固有属性，或者是否可以通过与他们的数据集成来进行优化。
- **为 Rerank 构建有效的 Prompt**：一位成员就如何为 **Rerank** 构建有效的 **Prompt** 寻求建议，特别是关于如何组织用户与助手之间的聊天历史。
   - 他们询问在 **Prompt** 中传递这些信息时，是否应保持时间顺序。


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1329195020707696661)** (53 messages🔥): 

> `随机数生成、Cohere Rerank 工具、Deep Learning 学习资源` 


- **Cmd R Bot 生成随机数**：**Cmd R Bot** 根据用户请求成功生成了随机数，提供了诸如 **84**、**12** 和 **37** 等数值。
   - 当用户询问其最高随机数时，机器人的回答是 *Gulp*，强调了其局限性。
- **Cohere Rerank 工具详解**：**Cohere** 的 **Rerank** 是一种语义搜索工具，利用 **Rerank** 模型根据查询对文档相关性进行排序。
   - 最新模型 **Rerank-v3.5** 具备跨多个领域的先进多语言检索能力。
- **学习 Deep Learning 的资源**：**Cohere** 提供了多种免费学习 **Deep Learning** 的资源，包括 **LLM University**、Cookbooks 以及为新账户提供的 75 美元额度。
   - 对于初学者，推荐材料包括生成式 AI（Generative AI）的指导课程和实用的 Cookbook，如“Hello World! Meet Language AI”。



**Link mentioned**: <a href="https://cohere.com/llmu#text-representation)">LLM University (LLMU)</a>: 欢迎来到 LLM University，这是您掌握企业级 AI 技术的首选学习目的地。专为开发者和技术专业人士设计，我们的中心提供全面的资源、专家指导...

  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1329457126854230078)** (12 条消息🔥): 

> `Tinygrad 浏览器版, JSPI Flag 集成, 跨平台测试, 云计算目标, 非均匀内存管理` 


- **Tinygrad 在浏览器中运行**：在启用 [JSPI flag](https://v8.dev/blog/jspi) 的帮助下，Tinygrad 现在可以在浏览器中运行，并已在包括 Mac 和 Ubuntu 在内的多个平台上测试成功。
   - 项目可以通过[此链接](https://mesozoic-egg.github.io/tinygrad/)访问，供他人测试和探索。
- **JSPI 的跨平台成功**：用户确认通过在 Chrome 中启用 **JSPI flag**，可以在 Windows 10、Windows 11 和 Mac M1 上成功运行 Tinygrad，展示了其广泛的兼容性。
   - 一位用户提到，“启用 jspi flag 后在我的 M1 pro 上可以运行”，进一步验证了该集成。
- **George Hotz 对云计算的愿景**：George Hotz 分享了宏大的云计算目标，建议联网机器可以像一个 GPU 一样协同运行，挑战现有架构。
   - 在讨论计算能力的潜在未来时，他强调，“在当前的 NVIDIA 栈之上，存在着一个充满可能性的全新世界”。
- **解决非均匀内存问题**：Hotz 表示，在某个规模上解决 **non uniform memory**（非均匀内存）问题可能会带来更广泛的解决方案，从而有助于降低芯片设计成本。
   - 他指出，“如果我们能在一个规模上解决非均匀内存问题，我们就能在所有规模上解决它”，强调了这一挑战的重要性。
- **浏览器实现的 Draft PR**：已创建一个 Draft Pull Request ##8645 以实现 Tinygrad 的浏览器功能，并邀请社区进行进一步测试。
   - 特别征求了在 Windows 上进行测试的反馈，以确保跨系统的兼容性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mesozoic-egg.github.io/tinygrad/">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/__tinygrad__/status/1879930546652156027">来自 tiny corp (@__tinygrad__) 的推文</a>: @ID_AA_Carmack @nisargypandya 疯狂的是，在当前的 NVIDIA 栈之上，存在着一个充满可能性的全新世界。为什么所有联网的机器不能像一个 GPU 那样运作呢？
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1329344184540467312)** (56 messages🔥🔥): 

> `Tinygrad 安装问题、TinyJit 性能、前 M3 设备上的 Metal 后端、Tinygrad 中的模型导出/导入、算子融合（Operator fusion）笔记` 


- **在 conda 中安装 Tinygrad 的问题**：一位用户报告了在 conda 环境中使用 `pip install -e .` 安装 Tinygrad 时出现错误，提示 `libgcc_s.so` 不是一个 ELF 文件。
   - 讨论表明，使用不带 venv 的标准 Python 可以正常工作，并指出这可能是由于 conda 覆盖系统库导致的 bug。
- **TinyJit 在 Metal 后端表现异常**：一位用户在搭载 Metal 后端的 2019 款 MacBook Pro 上使用 TinyJit 时遇到了优化步骤变慢的问题，引发了关于 GPU 同步的讨论。
   - 该问题与未同步 GPU 有关，解决方案包括调整 JIT 设置和检查调试日志。
- **为 Intel MacBook Pro 禁用 Metal graph**：建议为 M3 芯片之前的 Intel MacBook Pro 用户默认禁用 Metal graph，因为 Metal 驱动存在性能问题。
   - 大家一致认为，这一改动可以提升 Intel 机型用户的体验。
- **导出和导入 jitted 模型**：讨论了在 Tinygrad 中导出 jitted 模型的可行性，以便在不重新编译的情况下实现更快的重新加载和推理。
   - 有人指出 jitted 函数可以被 pickle 序列化，从而实现高效的模型使用，这与 openpilot 中的流程类似。
- **算子融合（Operator fusion）见解**：一位用户分享了一个详细介绍 Tinygrad 中算子（反）融合的链接，提供了关于优化技术的见解。
   - 提供的文档是理解算子融合的资源，展示了社区对提升 Tinygrad 性能的关注。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tellusim.com/metal-mdi/">MultiDrawIndirect and Metal - Tellusim Technologies Inc.</a>：未找到描述</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20250117_fusion.md">tinygrad-notes/20250117_fusion.md at main · mesozoic-egg/tinygrad-notes</a>：Tinygrad 教程。欢迎在 GitHub 上为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/blob/f91ca508cf88b09c616473561f68d2d46fbfcef9/tinygrad/renderer/ptx.py#L118">tinygrad/tinygrad/renderer/ptx.py at f91ca508cf88b09c616473561f68d2d46fbfcef9 · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你爱 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://git.informatik.uni-hamburg.de/4kirsano/master-thesis/-/blob/main/.conda/lib/libgcc_s.so)">Files · main · 4kirsano / Master-Thesis · GitLab</a>：UHH Informatics GitLab EE</li><li><a href="https://github.com/uuuvn/tinygrad.git">GitHub - uuuvn/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>：你喜欢 pytorch？你喜欢 micrograd？你爱 tinygrad！❤️ - GitHub - uuuvn/tinygrad: You like pytorch? You like micrograd? You love tinygrad！❤️
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1329180373074186280)** (36 条消息🔥): 

> `AI 项目文件问题，PrivateGPT 与 Obsidian，ChatGPT 软件工程局限性，生成式 AI 开发障碍，全模态图像生成延迟` 


- **AI 在理解项目文件方面存在困难**：一名成员对 AI 无法正确处理项目文件表示担忧，引发了其他人的共鸣。
   - 其他人推测了潜在的解决方案或规避方法，以避免开启新会话。
- **探索 PrivateGPT 与 Obsidian 的集成**：一名成员询问是否有人成功使用 PrivateGPT 自动学习 Obsidian 笔记本的内容，表现出对工具集成的兴趣。
   - 对话暗示了增强 AI 与个人知识系统交互的可能工作流。
- **ChatGPT 缺乏真正的软件工程能力**：成员们讨论了 ChatGPT 虽然可以辅助编码，但缺乏作为软件工程师职能的能力，特别是在创建复杂应用程序方面。
   - 一位用户对未来可能弥补这一差距的增强功能表示期待，并将理想场景与现有的 AI 助手进行了对比。
- **对生成式 AI 局限性的担忧**：讨论强调了一种观点，即生成式 AI 的能力目前局限于媒体创作，在功能性方面没有重大突破。
   - 成员们对 Midjourney 等工具生成高质量内容的成本表示沮丧。
- **对全模态图像生成发布时间表的质疑**：成员们思考了 OpenAI 和 Gemini 等公司在全模态图像生成方面进展缓慢的原因，反映了产品发布的延迟。
   - 此外，还有评论提到开源音频模型在有效管理情感输出方面的困难。



**提到的链接**：<a href="https://www.youtube.com/watch?v=x8jFFhCLDJY">Google Research 发布 "Transformers 2.0" 又名 TITANS</a>：我们是否终于破解了赋予模型“类人”记忆的密码？观看视频一探究竟！订阅我的 Newsletter 获取定期 AI 更新 👇🏼https://forwardfu...

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1329259374459355197)** (15 条消息🔥): 

> `Web 端 Canvas，GPT-4o 任务，图像渲染问题，自定义 GPTs，版本历史故障` 


- **Web 端 Canvas 现在仅显示任务**：一名成员注意到他们在 Web 版上只能看到任务，但另一名成员澄清说，点击文本输入框左下角的工具箱图标仍可以找到 **Canvas**。
   - 似乎其他人也对界面变化感到暂时困惑。
- **GPT-4o 任务的功能**：**GPT-4o** 中的任务充当提醒功能，例如“提醒我每天下午 3 点练习西班牙语”，ChatGPT 会在预定时间通知用户。
   - 该功能支持及时采取行动，增强了用户与模型的交互。
- **Journal Prompt GPT 中的图像渲染问题**：一名成员报告称其 GPT 停止根据日志提示渲染图像，引发了关于潜在配置问题的讨论。
   - 经发现，可能是功能设置中的 DALL·E 选项被取消勾选导致了故障。
- **确认自定义 GPTs 使用 GPT-4o**：明确了**自定义 GPTs** 使用的是 **GPT-4o** 版本，确保用户了解他们正在使用的模型。
   - 这有助于用户了解其自定义实现的各项能力。
- **版本历史显示故障**：一名成员观察到版本历史中的一些旧版本显示为“INVALID DATE”而非实际日期。
   - 这一故障引起了担忧，尽管目前原因尚不明确。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1329541557254230099)** (3 条消息): 

> `提示工程，撰写 AI 书籍，自我发现技术` 


- **30 天学会提示工程并写一本书？**：一名用户询问是否有可能利用 OpenAI 文档在 **30 天**内学会 **prompt engineering** 并写出一本书。
   - 另一名成员肯定了其可行性，并强调*只要能写出 prompt，就能立刻开始写书*。
- **提示工程推荐资源**：一名成员建议使用提供的[链接](https://chatgpt.com/share/67897fc3-6580-8000-af35-d693f933acfb)并结合额外的网络搜索来增强提示工程方面的知识。
   - 他们鼓励在学习过程中采用**自我发现技术 (self-discovery techniques)** 来对 AI 进行提示。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1329541557254230099)** (3 条消息): 

> `Prompt Engineering, Writing a Book on Prompting Techniques, OpenAI Documentation Utilization` 


- **30 天掌握 Prompt Engineering？**：一位用户询问是否可以在短短 **30 天**内学习 **prompt engineering** 并写一本书，并参考 OpenAI 的文档进行指导。
   - 另一位成员确认这是可能的，并建议一旦建立了有效的 prompt，书就可以准备好了。
- **Prompting 中的自我发现技术**：有人建议采用**自我发现技术 (self-discovery techniques)** 来增强 prompt 技能，强调个人探索而非仅仅依赖现有资源。
   - 一位成员分享了一个[特定资源](https://chatgpt.com/share/67897fc3-6580-8000-af35-d693f933acfb)的链接，可以帮助这一学习过程。
- **通过网页搜索扩展知识**：鼓励用户在利用所提供资源的同时，结合**网页搜索**来拓宽对 prompting 技术的理解。
   - 这种方法旨在建立一个结合多种信息来源的全面基础。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1329209654886269122)** (46 条消息🔥): 

> `Perplexity AI Issues, Image Generation Quality, Claude Sonnet Performance, Bora's Law, Market Strategies` 


- **Perplexity AI 面临功能问题**：用户报告称 Perplexity 一直出现错误并重复回答，即使启用了 PRO 搜索也是如此，这导致一些成员感到沮丧。
   - 一位用户指出，使用 **GROK** 时，性能比 Perplexity 有显著提升。
- **图像生成质量引发辩论**：关于图像生成质量的讨论浮出水面，用户争论 ChatGPT、Flux 和 Grok 等平台与 Perplexity 相比的有效性。
   - *这促使一位成员对比了 prompt，并表示*在生成简单的日出图像时，“差距非常大”。
- **Claude Sonnet 的用户体验**：几位用户分享了他们在 **Claude Sonnet** 上遇到的困难，特别提到了代码建议的不一致性以及 AI 的实用性问题。
   - 一位用户描述了他们与 AI 在 CSV 文件处理任务上的持续冲突，反映了对其可靠性的更广泛担忧。
- **Bora's Law 的潜力**：一位成员认为现有的 AGI 方法（包括 OpenAI 的方法）存在缺陷，并提出了 **Bora's Law**，指出智能随约束而非算力 (compute) 扩展。
   - 他们在一篇文章中引用了自己的发现，以支持这一 AI 发展中的新兴原则。
- **市场策略和成功声明**：一位用户推销了一项计划，声称能帮助个人在数字市场每周赚取超过 10 万美元，但需要支付一定比例的利润作为报酬。
   - 这一提议遭到了其他人的怀疑，他们幽默地将其比作“卖蛇油”。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://chrisbora.substack.com/p/boras-law-intelligence-scales-with?r=aszci">Bora&#x27;s Law: Intelligence Scales With Constraints, Not Compute</a>：这是一篇探讨人工智能发展中新兴原则的工作论文。</li><li><a href="https://www.youtube.com/watch?v=itpcsQQvgAQ&feature=youtu.be">Nintendo Switch 2 – 预告片首秀</a>：介绍 Nintendo Switch 的继任者 Nintendo Switch 2，将于 2025 年发布。了解更多：https://ninten.do/6003ohKAB
</li>
</ul>

</div>

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1329334988918751285)** (2 条消息): 

> `Enron Prank Revival, SEC Sues Elon Musk, TikTok Sale Consideration, 3D Printing AI Tools, 3D Printing Toys` 


- **Enron 恶作剧复兴登上头条**：最近的一段 [YouTube 视频](https://www.youtube.com/embed/NZ9FtiBAPUc) 讨论了与 **Enron** 相关的恶作剧有趣回归。
   - 该视频捕捉到了围绕这一历史上著名的企业丑闻话题的怀旧感和幽默感。
- **SEC 对 Elon Musk 采取行动**：在一项重大的法律行动中，**SEC** 对 **Elon Musk** 提起了诉讼，引发了关于科技领域监管影响的讨论。
   - 这一进展引发了人们对 Musk 未来创业项目和公开言论影响的疑问。
- **中国官员关注 TikTok 出售**：有推测称，受国际市场审查力度加大的推动，**中国官员**正在考虑潜在的 **TikTok** 出售事宜。
   - 这可能会对该应用的全球运营及其用户群影响力产生重大影响。
- **探索用于 3D 打印的 AI 工具**：一位成员表示有兴趣了解用于创建 **3D 对象文件**的 AI 工具，展示了对该技术日益增长的好奇心。
   - 你可以查看[此处的讨论链接](https://www.perplexity.ai/search/can-perplexity-ai-be-used-to-g-tp.qKJ1JQJaAAmpqQUb7Og#2)以获取见解和建议。
- **对 3D 打印玩具的热情**：一位成员分享了他们对 **3D 打印**的兴奋之情，提到他们喜欢制作小玩具，但尚未尝试功能性打印。
   - 这反映了爱好者探索现代制造技术创意应用的日益增长的趋势。



**提到的链接**：<a href="https://www.youtube.com/embed/NZ9FtiBAPUc">YouTube</a>: 未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1329382917226364960)** (4 条消息): 

> `search_domain_filter, API model changes, CrewAI custom stop parameters` 


- **关于 search_domain_filter 的求助请求**：一位用户请求协助开启使用 **search_domain_filter** 的权限。
   - 这表明用户对在平台内利用域名过滤功能的持续兴趣。
- **对 API 模型变更的推测**：一位成员观察到实验室中引入了 **sonar** 和 **sonar-pro**，并询问 **API** 模型是否会再次发生变化。
   - 他们分享了一张[图片](https://cdn.discordapp.com/attachments/1161802929053909012/1329550935038623877/image.png?ex=678ac04c&is=67896ecc&hm=411e3da2e4bdb49d1732bd458d3fdbf26e6147a332d51ec2687feb39fe5581f0&)，其中可能包含有关这些模型的相关信息。
- **search_domain_filter 的重复问题**：另一位用户报告了关于 **search_domain_filter** 功能的相同问题，表明多位用户正面临这一挑战。
   - 这突显了社区内对于设置某些功能使用权限的普遍关注。
- **CrewAI 自定义停止参数错误**：一位用户在尝试使用 **CrewAI** 时遇到了 **'custom stop parameters'** 错误，并对此表示沮丧。
   - 他们询问是否有开发者在场，能够提供关于如何让 **pplx** 正确运行的见解。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1329204325343297628)** (51 条消息🔥): 

> `模型搜索问题, LM Studio 日志记录, Context Window 解释, VRAM 与系统 RAM, M2 Mac 模型加载问题` 


- **用户遇到模型加载搜索问题**：一位用户对**模型搜索**无法工作表示沮丧，其他人建议这可能与 **Hugging Face** 的问题有关。
   - 进一步的讨论表明，性能问题可能源于特定的系统要求，特别是关于 CPU AVX2 指令的要求。
- **用于故障排除的日志机制**：用户讨论了缺乏清晰的**日志窗口**来进行故障排除的问题，并建议使用开发者选项卡和终端命令来获取深入信息。
   - 终端命令 `lms log stream` 被强调为一种更彻底检查日志的有用方法。
- **了解 Context Window 管理**：一位用户询问“上下文已满 90.5%”的含义，从而引发了关于 **Context Window** 及其与模型 Token 容量关系的解释。
   - 有人指出，虽然大多数模型能很好地管理上下文，但可能需要用户进行调整以增加 Context Size。
- **RAM 和 VRAM 使用讨论**：一位用户询问模型是存储在 **VRAM 还是系统 RAM** 中，得到的澄清是使用情况取决于硬件配置。
   - 对于 **CPU 推理**，模型使用系统 RAM，而 **GPU 推理**利用 VRAM，必要时会溢出到 RAM。
- **Mac 上 LM Studio 0.3.6 的问题**：多位用户报告在更新到 **LM Studio 0.3.6** 后加载模型出现困难，暗示这可能与缓存问题有关。
   - 清理 `.lm-studio` 缓存解决了加载问题，证实了在更新期间进行细致缓存管理的必要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>: 发现、下载并运行本地 LLM</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/115#issuecomment-2576418933">LM Studio 在一段时间后无法再获取模型或扩展，只有重启系统才有帮助 (macOS) · Issue #115 · lmstudio-ai/lmstudio-bug-tracker</a>: 不确定该发布到哪里，请随意将其移动到正确的位置。一段时间后（我不确定需要多久，系统休眠可能会导致此问题），这种情况发生了几次……
</li>
</ul>

</div>
  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1329271987167690824)** (50 条消息🔥): 

> `剧本反馈、模型性能与对比、AI 伦理护栏、模型推荐、GPT4All 中的模型管理` 


- **剧本分析的挑战**：一位用户对 GPT4All 在分析其 **45 页剧本**时的字符限制表示沮丧，指出尽管模型拥有 **128 KB** 的容量，但似乎一次只能分析一个场景。
   - 获取详细分析已被证明非常繁琐，这促使人们寻找变通方法和效率提升技术。
- **AI 模型响应的差异**：讨论围绕为什么 **ChatGPT 4.0** 比其替代版本能更好地处理显式内容展开，这表明不同的模型是在不同的审查标准下训练的。
   - 人们对这些伦理护栏（ethical guardrails）对模型性能以及用户获取平衡信息的影响表示担忧。
- **写作模型推荐**：用户推荐使用 **DavidAU 的模型**进行写作辅助，强调了其在有效生成黑暗和非黑暗内容方面的能力。
   - 建议还包括 **Magnum** 模型，以及根据 VRAM 和量化（quant）设置优化性能的使用技巧。
- **GPT4All 中的模型管理**：一位用户询问如何将下载的模型导入 GPT4All，得到的澄清是：模型需要放置在指定文件夹中，并且必须重启应用程序。
   - 强调了在更改后关闭并重新打开 App 以确保模型列表更新的重要性。
- **技术性能与设置**：一位使用 **Gemma 模型**的用户遇到了性能问题，建议通过调整量化设置来提高响应速度。
   - 讨论还包括 GPU 设置的影响，以及为了实现最佳运行而微调配置的必要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.prompthackers.co/compare/llama-3.2-3b/llama-3-8b">Compare Llama 3.2 3B vs Llama 3 8B Instruct - Pricing, Benchmarks, and More</a>：对比 Llama 3.2 3B 与 Llama 3 8B Instruct 的价格、基准测试、模型概览等。深入对比 Llama 3.2 3B 与 Llama 3 8B Instruct。</li><li><a href="https://huggingface.co/docs/transformers/main/main_classes/quantization">Quantization</a>：未找到描述</li><li><a href="https://huggingface.co/DavidAU">DavidAU (David Belton)</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1329468720678179000)** (5 条消息): 

> `LeetGPU 在线游乐场、讨论频道重定向、征集贡献者` 


- **LeetGPU 提供免费 CUDA 游乐场**：一位成员宣布推出 [LeetGPU](https://leetgpu.com/)，这是一个无需注册或 GPU 权限即可编写和执行 **CUDA** 代码的在线游乐场（playground）。
   - 他们鼓励成员进行尝试并分享反馈。
- **请求更好的讨论频道**：一位成员建议重要的讨论应在特定频道中进行，并指出这将有助于更好地组织沟通。
   - 他们指出了频道 ID 以供未来参考，从而保持讨论的专注度。
- **为项目征集贡献者**：另一位成员询问是否有项目正在寻求贡献者加入，表示对协作努力感兴趣。
   - 这反映了社区内参与和扩展项目合作的意愿。



**提到的链接**：<a href="https://leetgpu.com/,">LeetGPU</a>：未找到描述

  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1329252890178748466)** (4 条消息): 

> `Triton 初始化错误、tl.gather 限制、优化 Moe Kernel、向量化性能问题` 


- **Triton 指针初始化错误**：一名成员指出，在 Triton 中使用 **int** 指针是不正确的，建议应使用 **float**，因为指针代表内存地址，且应仅为标量。
   - *这一修正可能会解决与数据类型相关的各种初始化问题*。
- **tl.gather 对常量的限制**：有用户报告 Triton 中的 **tl.gather** 不接受 **tl.constexpr**，仅接受 **tl.tensor**，这限制了他们在优化 Moe Kernel 时的使用场景。
   - 他们强调，使用 **tl.gather** 访问片上（on-chip）数值对于利用专家出现次数（expert occurrences）实现 **cuRadix** 等算法至关重要。
- **向量化导致的性能下降**：在寻求避免对常量使用 **tl.gather** 的解决方案后，一名成员发现其 Kernel 虽然可以正确运行，但在向量化后经历了**显著的性能下降**。
   - 他们正在寻找进一步的优化策略来解决这一性能衰退问题。
- **tl.store 和内存访问的不当使用**：有人指出，在循环中使用 **tl.store** 会因为内存传输时的线程执行等待时间而严重阻碍性能。
   - 给出的建议是强调使用带有块指针（block pointers）的 **tl.load**，以提高数据访问效率。



**提到的链接**：<a href="https://github.com/sgl-project/sglang/pull/2913">[Triton] try to optimzie triton moe kernel implmenet with vectorization and tl.gather triton-3… by yiakwy-xpu-ml-framework-team · Pull Request #2913 · sgl-project/sglang</a>：动机：在调试有问题的 Moe CUDA Kernel（非法内存访问）时，我尝试优化 Triton Moe Kernel。该 Triton Moe Kernel 基本上是利用出现次数实现基数排序（fi...

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1329421211318358147)** (2 条消息): 

> `ONNX 到 TensorRT 的转换、Myelin CUDA 错误` 


- **ONNX 转换过程中的 Myelin CUDA 错误**：一名成员在尝试将模型从 **ONNX** 转换为 **TensorRT** 时遇到了 **CUDA error 400**，具体发生在 `__myl_Res` Kernel 处。
   - 他们正在寻求解决此转换问题的潜在**解决方案**。
- **寻求转换问题的解决方案**：一名用户请求协助处理在 ONNX 到 TensorRT 转换过程中遇到的 **Myelin CUDA 错误**。
   - 对话反映出希望社区协助排查这一特定错误的意愿。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1329444313821610065)** (4 条消息): 

> `Torchinductor、Torch Compile、机器学习框架、Caffe 的历史背景` 


- **关于 Torchinductor 的深入讨论**：分享了一篇讨论 **Torchinductor** 的博客文章，它被描述为一个具有 **define-by-run IR** 和**符号形状（symbolic shapes）**的 PyTorch 原生编译器。讨论强调了内容的知识性，并包含许多有价值的链接。
   - 尽管内容有些过时，但仍鼓励读者查看，并指出其获得了积极的反响。
- **关于 Torch Compile 的有用见解**：一名成员分享了他们的博客文章《剖析 Torch Compile》（*Dissecting Torch Compile*），该文章探讨了 **Torch Compile** 等现代机器学习工具背后的复杂性，以及机器学习多年来的演变。文章包含了博客链接及其对应的 [GitHub 仓库](https://github.com/DWarez/torch_compile_blogpost)。
   - 作者反思了从 **Caffe** 开始的转型，并强调了现代框架变得多么易于使用，同时也承认了其底层的复杂性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://themlsurgeon.substack.com/p/dissecting-torchcompile-surgical">Dissecting torch.compile: Surgical Precision in PyTorch Optimization</a>：你可以通过此链接查看该博客文章的 GitHub 仓库</li><li><a href="https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747">TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes</a>：PyTorch 团队一直在构建 TorchDynamo，它通过动态 Python 字节码转换帮助解决 PyTorch 的图捕获问题。为了真正让 PyTorch 变快，TorchDynamo 必须...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1329186842150699120)** (6 条消息): 

> `GPU 知识内化，MLPerf 参与，MI300X 节点性能，MI300A 中的 XCD 划分` 


- **GPU 用户需要内化知识**：一位成员引用了 @vrushankdes 制作的 **A100 版本**动画，强调了内化 GPU 知识的重要性。
   - 这表明对于任何认真对待图形处理单元（GPU）工作的人来说，深入理解是核心重点。
- **MLPerf 厂商与 GPT-3 架构**：有人提出了一个问题：鉴于 **GPT-3** 的架构和权重并未开源，**MLPerf** 中的厂商是如何获取它们的。
   - 这一询问反映了人们对参与者实际使用的资源持续感到好奇。
- **划分 MI300X 可提升性能**：讨论了如何将 **MI300X 节点**划分为 8、4 或 2 份，以及这对内存性能的影响。
   - 潜在的效率提升源于减轻了 **infinity cache** 的工作负载。
- **MI300 变体中的 XCD 分离**：讨论中提到了将 **MI300A** 划分为多个份额，这与早期型号中已知的 **XCDs** 数量一致。
   - 这种方法与 MI250 的早期技术类似，旨在优化设备性能。



**提到的链接**：<a href="https://fixupx.com/fleetwood___/status/1879511438538281350">来自 Fleetwood (@fleetwood___) 的推文</a>：每个从事 GPU 工作的人都需要内化这一点。@vrushankdes 动画的 A100 版本。

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1329258402429538384)** (9 条消息🔥): 

> `GPU 请求批处理，Kernel 实现反馈，使用 CUDA 的 Flash Attention，使用 Triton 进行分块矩阵乘法，评估硬件需求` 


- **理解 GPU 请求批处理**：由于**请求是批处理的**，单个 GPU 可以为多个用户提供服务，但容量取决于用于 **KV cache** 的可用 **VRAM**。
   - 要评估服务 N 个用户的硬件需求，需要考虑 token 数量并根据模型参数进行计算。
- **寻求 Triton kernels 的反馈**：一位用户请求对其用于线性层的分块矩阵乘法（blockwise matmul）kernel 实现提供反馈，并提到他们希望优化其 GPU 性能。
   - *“是的，它主要基于分块矩阵乘法教程……”*，这表明需要关于使用 **tl.fma** 进行操作组合的正确指导。
- **CUDA 版 Flash Attention 资源**：一位成员向对 **CUDA 版 Flash Attention** 感兴趣的用户推荐了一个 [GitHub 仓库](https://github.com/damienjose/cuda-flashattention)，用于实现和测试。
   - 对于想要开始接触这项技术的 **CUDA** 初学者来说，这个资源特别有帮助。
- **关于性能的 YouTube 教程**：分享了一个 YouTube 视频链接，作为学习 CUDA 中优化**矩阵乘法**的额外资源。
   - 这补充了关于优化 Triton 实现以获得更好线性层性能的讨论。
- **分块矩阵乘法教程参考**：一位成员指出，正在审查的 kernel 与 **Triton 教程**中的内容相似，特别是关于块级矩阵乘法的部分。
   - 他们提供了一个[教程链接](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py)，详细阐述了高效的 FP16 矩阵乘法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py">矩阵乘法 &mdash; Triton 文档</a>：未找到描述</li><li><a href="https://github.com/damienjose/cuda-flashattention">GitHub - damienjose/cuda-flashattention</a>：通过在 GitHub 上创建账号来为 damienjose/cuda-flashattention 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1329410107288588348)** (1 条消息): 

> `Torch TensorRT 安装，PyTorch 2.1.1 文档` 


- **Torch TensorRT 入门**：*一位用户*表达了在为 **PyTorch 2.1.1** 安装 **Torch TensorRT** 时遇到的挑战，认为文档没有帮助。
   - 他们正在寻求关于如何进行安装过程的更清晰指导。
- **对文档感到失望**：该用户批评了 **Torch TensorRT** 的**文档**，称其未能为安装问题提供足够的支持。
   - 他们表示感到沮丧，并正在寻找更有效的资源或社区帮助。

### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1329567823948812300)** (1 条消息): 

> `Linux arm64 Runners, Copilot Chat for Actions Job Failures` 


- **面向公共仓库推出 Linux arm64 Runners**：团队今天宣布发布 **Linux arm64 hosted runners**，并在 [公共仓库](https://github.blog/changelog/2025-01-16-linux-arm64-hosted-runners-now-available-for-free-in-public-repositories-public-preview/) 中免费提供。这一补充支持在 ARM 架构上运行工作流，增强了开发者的灵活性。
- **Copilot chat 现在可解决 Actions 任务失败问题**：通过 **“Explain Error”** 功能向 **Copilot chat** 咨询 Actions 任务失败原因的功能现已正式发布。用户可以直接从 **PR mergebox** 或 **Actions Job Page** 激活此功能，讨论任务失败的原因及解决方案。
   - 这一新功能通过提供与 Copilot 的直接交互，针对错误日志提供见解，从而改善了排错体验，具体示例如[此处](https://github.com/user-attachments/assets/04ffd085-cede-4342-b75c-7a80dbff7be9)和[此处](https://github.com/user-attachments/assets/57c0eb6b-d567-4a95-becc-4edca865c351)的截图所示。



**提到的链接**：<a href="https://github.blog/changelog/2025-01-16-linux-arm64-hosted-runners-now-available-for-free-in-public-repositories-public-preview/">Linux arm64 hosted runners now available for free in public repositories (Public Preview) · GitHub Changelog</a>：Linux arm64 hosted runners 现在公共仓库中免费提供（公开预览版）

  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 条消息): 

0x000ff4: 有什么我可以参与贡献的活跃话题吗？
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1329551300626743328)** (1 条消息): 

> `Happy New Year 2025, ArXiv Submission, Researcher Endorsement` 


- **2025 新年祝福**：一位成员分享了他们的热情并致以友好的问候，祝大家 **2025 新年快乐**！
   - 这一节日问候为接下来的讨论奠定了积极的基调。
- **寻求 ArXiv 投稿背书**：一位成员正在为 ArXiv 上 **cs.LG** 类别的论文投稿寻求背书，强调了他们的支持请求。
   - 他们对这个以工程师为主导的社区中是否存在研究人员表示不确定。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1329223651194835016)** (12 条消息🔥): 

> `Modal Registry for Popcorn Bot, GPU Type Management, nvidia-smi and deviceQuery, Discord Leaderboard Integration, Function Versioning in Modal` 


- **系统化 Popcorn 机器人的 Modal Registry**：成员们讨论了系统化 Popcorn 机器人的 **modal registry** 的策略，包括在设置 GPU 类型时**部分应用 (partially applying)** 函数所面临的挑战。
   - 他们建议为每种 GPU 类型创建多个 Modal Functions，而不是尝试通用地应用单个函数。
- **GPU 类型内省计划**：分享了未来允许在 Modal Functions 内部使用 **modal.container.gpu** 更轻松地内省 GPU 类型的计划。
   - 目前，变通方法可能涉及直接从 Discord 机器人设置 GPU 架构，从而可能允许对计算能力进行实验。
- **使用 nvidia-smi 获取 GPU 能力**：建议了一种在脚本中直接使用 `nvidia-smi` 或 **deviceQuery** 工具来确定 GPU 计算能力的方法。
   - 这允许开发者动态优化 GPU 使用，从不同架构之间的定制性能中获益。
- **探索函数版本控制**：提出了维护一个函数的多个版本的想法，每个版本都配置有不同的基础设施参数。
   - 这一功能被强调为促进 **Discord leaderboard** 集成的理想特性。
- **考虑小众功能**：一位成员指出，允许调用具有错误编译架构的端点可以用于一些小众目的，例如比较不同架构之间的速度差异。
   - 这种灵活性可能会为对性能比较感兴趣的高级用户带来意想不到的好处。



**提到的链接**：<a href="https://stackoverflow.com/questions/40695455/what-utility-binary-can-i-call-to-determine-an-nvidia-gpus-compute-capability))">What utility/binary can I call to determine an nVIDIA GPU's Compute Capability?</a>：假设我有一个安装了单个 GPU 的系统，并且假设我也安装了最新版本的 CUDA。我想确定我的 GPU 的计算能力。如果我可以...

  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1329306348592038020)** (3 条消息): 

> `Kernel 选项文档，入职协助` 


- **文档协作**：一位成员确认当前文档目前可行，并表示愿意在未来提供更详尽的文档。
   - 他们表示，如果有更多信息，会主动联系。
- **列出的 Kernel 选项**：另一位成员分享说，他们已在文档中列出了几个 Kernel 选项，并准备好在需要时协助入职。
   - 他们对帮助他人尝试建议的 Kernel 表现出极大的热情。
- **成员将审阅文档**：一位成员确认了 Kernel 选项，并表示将先审阅文档。
   - 这表明了持续的协作以及参与提供资源的意愿。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1329192774813094063)** (29 条消息🔥): 

> `LLM 的惊喜与非惊喜，Discord LLM 推荐，模型蒸馏方法，ChatGPT 的动人时刻，文档处理技术` 


- **LLM 在惊喜与非惊喜之间挣扎**：一位成员对当前的 LLM 动态（特别是 softmax）在平衡输出中的**惊喜 (surprise) 与非惊喜 (unsurprise)** 方面的弱点表示担忧，主张应关注惊喜方面。
   - *一个可行的模型应该比非惊喜部分更关注惊喜部分*。
- **寻找 Discord LLM**：有人询问是否有适合对话的 **Discord LLM** 推荐，因为成员们看到的机器人无法满足特定需求。
   - 另一位成员澄清了他们对通过 jailbreaking 模型来增强其能力的兴趣。
- **讨论模型蒸馏技术**：一位成员讨论了**模型蒸馏 (model distillation) 方法**，包括从教师模型生成输出以教导学生模型，重点关注特定细节而无需过量数据。
   - 有人担心，即使采用有针对性的方法，学生模型是否仍能植根于自然的数据分布。
- **ChatGPT 的温馨互动**：一个感人的故事讲述了一个孩子如何与 **ChatGPT** 互动，询问桌上的各种物品，而 ChatGPT 负责任地将电子烟 (vape) 描述为“成年人的东西”。
   - *它甚至能认出那是电子烟，这让我大为震惊*。
- **处理大文档的技术**：成员们辩论了高效处理**大文档**的方法，强调向量数据库 (vector databases) 和 embeddings 是常见的解决方案。
   - 一位成员推荐使用 **Langchain** 处理文档，并为初学者提供了教程建议。



**提到的链接**：<a href="https://python.langchain.com/docs/tutorials/">Tutorials | 🦜️🔗 LangChain</a>：刚接触 LangChain 或 LLM 应用开发？阅读这些材料以快速上手构建你的第一个应用程序。

  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1329255997893644378)** (11 messages🔥): 

> `Google Research 新架构, OpenAI API 政策变更, OpenAI 计算成本, Tensor Product Attention (TPA), 每日论文讨论录音` 


- **Google Research 提出新架构**：今天，小组将探讨来自 Google Research 的一种新架构，他们声称该架构在某些领域**优于 Transformers**，重点关注其方法和结果。讨论将基于[此处](https://arxiv.org/abs/2501.00663)的论文。
   - 成员们对该架构与 **Gemini 1.5** 之间潜在的联系表示了兴趣。
- **OpenAI 修改 API 数据使用政策**：OpenAI 将不再默认使用其 API 的数据进行模型训练，除非组织**主动选择加入 (opt in)**。这些变更旨在回应开发者和用户的批评。
   - 针对此前在未经明确同意的情况下进行的**默认数据训练**，人们提出了担忧。
- **OpenAI 飙升的计算费用**：报告指出，OpenAI 今年可能在 Microsoft 的服务器上花费 **40 亿美元**用于推理工作负载，需要额外资金来弥补巨额亏损。训练成本（特别是 ChatGPT）预计也将飙升至 **30 亿美元**左右。
   - 分享了关于 OpenAI 从 Microsoft Azure 获得的 **A100 服务器费率特定折扣**的细节。
- **引入 Tensor Product Attention (TPA)**：一篇新论文提出了 Tensor Product Attention (TPA)，这是一种创新的机制，可以最小化语言模型推理过程中的 **KV cache 大小**。该架构命名为 T6，在与标准 Transformer 模型的实证评估中显示出极具前景的结果。
   - 作者提到，未来的开发将集成 **Flash 版本的 TPA** 以增强性能。
- **每日论文讨论录音已发布**：昨天名为“Solving the ARC-AGI AI Benchmark with ICOM”的每日论文讨论录音可在[此处](https://youtu.be/WeJuLJrVhf4)观看。该环节遇到了一些显著的技术挑战，例如由 OBS 引起的重音问题。
   - 鼓励参与者回顾讨论及其关于该基准测试的见解。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.04620">Learning to (Learn at Test Time): RNNs with Expressive Hidden States</a>: Self-attention 在长上下文中表现良好，但具有平方复杂度。现有的 RNN 层具有线性复杂度，但它们在长上下文中的性能受限于其表达能力...</li><li><a href="https://arxiv.org/abs/2501.06425">Tensor Product Attention Is All You Need</a>: 扩展语言模型以处理更长的输入序列通常需要巨大的 Key-Value (KV) caches，导致推理过程中产生巨大的内存开销。在本文中，我们提出了 Tensor...</li><li><a href="https://youtu.be/WeJuLJrVhf4">Daily Paper Discussion: Solving the ARC-AGI AI Benchmark with ICOM</a>: 我很抱歉没有预料到 OBS 与 Discord 的兼容性较差，导致 Discord 上其他人的发言产生回声，因为软件...</li><li><a href="https://github.com/tensorgi/T6/blob/d4f6168852397a7b0b0d9fd65326bb91976c7067/model/T6.py#L64">T6/model/T6.py at d4f6168852397a7b0b0d9fd65326bb91976c7067 · tensorgi/T6</a>: Tensor ProducT ATTenTion Transformer (T6) 的官方实现 - tensorgi/T6</li><li><a href="https://www.datacenterdynamics.com/en/news/openai-training-and-inference-costs-could-reach-7bn-for-2024-ai-startup-set-to-lose-5bn-report/?ref=ai-recon.ghost.io">OpenAI training and inference costs could reach $7bn for 2024, AI startup set to lose $5bn - report</a>: 关于其 Microsoft Azure 计算集群的细节泄露</li><li><a href="https://techcrunch.com/2023/03/01/addressing-criticism-openai-will-no-longer-use-customer-data-to-train-its-models-by-default/">Addressing criticism, OpenAI will no longer use customer data to train its models by default | TechCrunch</a>: OpenAI 更改了其开发者政策，增加了数据保留选项并澄清了其对客户数据的使用。</li><li><a href="https://community.openai.com/t/does-the-openai-api-get-access-to-the-data-i-send-it-or-store-the-data/599538/2">Does the openai API get access to the data I send it or store the data</a>: 欢迎来到社区！通过 OpenAI 在 Discord 上的 kapa.ai 实现进行回答…… OpenAI 高度重视数据安全和隐私。根据摘录中提供的信息：...
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1329184507776991295)** (5 条消息): 

> `4090 GPU 故障，双槽位 4090 的供应情况` 


- **沉重的 4090 导致高故障率**：由于重量原因，**4090 GPU** 的故障率较高，会导致 **PCB 裂纹**和 **BGA 故障**。为了解决这个问题，中国市场正在将部分显卡重新封装为**双槽位 (2-slot) 4090**。
- **eBay 上的双槽位 4090 供应情况**：一名成员建议在 **eBay** 上查看**双槽位 4090**，并指出目前有很多选项。
   - 其中一个特定的商品列表是一款 [OEM 48GB RTX 4090 Founders Edition](https://www.ebay.com/itm/126885374543)，支持从大中华区加急发货，在**过去 24 小时内获得了 23 次浏览**。



**提到的链接**：<a href="https://www.ebay.com/itm/126885374543">OEM 48GB RTX 4090 Founders Edition Dual width GPU Graphics card Ganming/ Server  | eBay</a>：未找到描述

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1329193055957549066)** (38 条消息🔥): 

> `Francois Chollet 的 AI 实验室 Ndea，用于合成数据生成的 Curator，Titans 内存架构，HAL 全方位 Agent 排行榜，Harvey AI 融资` 


- **Francois Chollet 启动新 AI 实验室 Ndea**：Francois Chollet 宣布与 Mike Knoop 合作创办 [Ndea](https://x.com/fchollet/status/1879583863368032432)，专注于**深度学习引导的程序合成 (program synthesis)**，旨在推动 AI 创新。
   - 他们正在寻求一条独特的路径，以增强 AI 在**适应 (adaptation)** 和**发明 (invention)** 方面的能力。
- **推出 Curator，一款新的合成数据工具**：[Curator](https://x.com/madiator/status/1879579213554147665) 是一个开源库，旨在简化**高质量合成数据生成**，这对于训练 LLM 和 Agent 至关重要。
   - 据报道，该工具将创建后训练数据集的生产力提高了 **10 倍**。
- **Titans：一种革命性的内存架构**：新的 **Titans 架构**引入了一种元上下文内存 (meta in-context memory)，可以在测试时进行记忆，其表现潜力可能超越包括 **GPT-4** 在内的现有模型。
   - 这一进展可以有效地将上下文窗口扩展到 **2M** 以上，重新定义了 AI 模型中的内存使用方式。
- **HAL：AI Agent 评估排行榜**：推出了一项名为 [HAL](https://x.com/sayashk/status/1879932823668498576) 的新计划，旨在跨 **11 个基准测试**对超过 **90 个 Agent** 进行评估。
   - 它提出了关于推理模型 (reasoning models) 与标准语言模型相比的成本和有效性的重要问题。
- **Harvey AI 获得巨额投资**：据报道，法律科技初创公司 **Harvey** 正在进行由 Sequoia 领投的 **3 亿美元**融资，估值达到 **30 亿美元**。
   - 此前该公司曾以 **15 亿美元**的估值融资 **1 亿美元**，而其营收预估为 **3000 万美元**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://blogs.worldbank.org/en/education/From-chalkboards-to-chatbots-Transforming-learning-in-Nigeria">从黑板到聊天机器人：在尼日利亚通过 Prompt 逐步改变学习方式</a>：“AI 帮助我们学习，它可以充当导师，它可以成为你想要的任何角色，这取决于你编写的 Prompt，”学生 Omorogbe Uyiosa（朋友们称他为“Uyi”）说道……</li><li><a href="https://evalplus.github.io/leaderboard.html">EvalPlus 排行榜</a>：未找到描述</li><li><a href="https://x.com/synthesiaio/status/1879475235390660833?s=46">Synthesia 🎥 (@synthesiaIO) 的推文</a>：🎉 重大新闻：我们已完成 1.8 亿美元的 D 轮融资 🎉 尽管前方仍有大量工作，但前进的道路从未如此清晰。当然，如果没有我们了不起的客户，这一切都不可能实现……</li><li><a href="https://www.microsoft.com/en-us/research/blog/mattergen-a-new-paradigm-of-materials-design-with-generative-ai/">利用 AI 重新思考材料创新</a>：微软研究人员推出了 MatterGen，这是一个可以根据特定需求（如高效太阳能电池或二氧化碳回收）发现新材料的模型，推动了超越试错实验的进展……</li><li><a href="https://x.com/shawnup/status/1880004026957500434">Shawn Lewis (@shawnup) 的推文</a>：我基于 o1 的 AI 编程 Agent 现在在 SWE-Bench Verified 上达到了业界领先水平（SOTA）！它解决了 64.6% 的问题。这是我们已知的第一个完全由 o1 驱动的 Agent。在构建它的过程中，我们学到了很多。</li><li><a href="https://x.com/sayashk/status/1879932823668498576">Sayash Kapoor (@sayashk) 的推文</a>：最顶尖的 SWE-Bench Agent 有多贵？推理模型是否优于语言模型？我们能信任 Agent 的评估吗？📢 宣布推出 HAL，一个用于评估 AI Agent 的全面排行榜（Holistic Agent Leaderboard）……</li><li><a href="https://x.com/samuel_colvin/status/1879627376990224417">Samuel Colvin (@samuel_colvin) 的推文</a>：我们刚刚发布了 @Pydantic AI v0.0.19。这是自我们发布 PydanticAI 以来最大的新功能——Graph 支持！我最初对 Graph 持怀疑态度，但现在我感到非常兴奋……</li><li><a href="https://x.com/fchollet/status/1879583863368032432">François Chollet (@fchollet) 的推文</a>：我正与 @mikeknoop 联手创办 Ndea (@ndeainc)，一家新的 AI 实验室。我们的重点是：深度学习引导的程序合成。我们押注于一条不同的道路，以构建具有真正发明能力的 AI……</li><li><a href="https://x.com/behrouz_ali/status/1878859086227255347">Ali Behrouz (@behrouz_ali) 的推文</a>：Attention 一直是 LLM 大多数进展的关键组件，但它无法扩展到长上下文。这是否意味着我们需要寻找替代方案？介绍 Titans：一种结合了 Attention 的新架构……</li><li><a href="https://x.com/madiator/status/1879579213554147665?s=46">Mahesh Sathiamoorthy (@madiator) 的推文</a>：我们很高兴地宣布 Curator，一个旨在简化合成数据生成的开源库！高质量的合成数据生成对于训练和评估 LLM/Agent/RAG 至关重要……</li><li><a href="https://x.com/pitdesi/status/1879982274831347890?s=46">Sheel Mohnot (@pitdesi) 的推文</a>：面向律师事务所的 AI 公司 Harvey 正在从红杉资本进行新一轮融资（以 30 亿美元估值融资 3 亿美元）。上一轮是 7 月份的 C 轮，由 GV 领投，以 15 亿美元估值融资 1 亿美元。当时估计他们的收入为 3000 万美元，想知道现在的……</li><li><a href="https://x.com/emollick/status/1879633485004165375?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Ethan Mollick (@emollick) 的推文</a>：关于尼日利亚学生使用 GPT-4 作为导师的新随机对照试验。6 周的课后 AI 辅导 = 2 年的典型学习收益，表现优于 80% 的其他教育干预措施……</li><li><a href="https://x.com/emollick/status/1879633485004165375?s=46&t=6FDPaNxZcbSsELal6Sv7">Ethan Mollick (@emollick) 的推文</a>：关于尼日利亚学生使用 GPT-4 作为导师的新随机对照试验。6 周的课后 AI 辅导 = 2 年的典型学习收益，表现优于 80% 的其他教育干预措施……</li><li><a href="https://github.com/openai/openai-realtime-agents">GitHub - openai/openai-realtime-agents：这是一个基于 Realtime API 构建的更高级 Agent 模式的简单演示。</a>：这是一个基于 Realtime API 构建的更高级 Agent 模式的简单演示。 - openai/openai-realtime-agents</li><li><a href="https://www.youtube.com/watch?v=x8jFFhCLDJY">Google Research 发布“Transformers 2.0”即 TITANS</a>：我们终于破解了如何赋予模型“类人”记忆的密码了吗？观看视频一探究竟！订阅我的时事通讯以获取定期 AI 更新 👇🏼 https://forwardfu...</li><li><a href="https://www.forbes.com/sites/philkirschner/2025/01/15/did-ai-cause-those-layoffs-ny-employers-may-have-to-disclose/?utm_source=chatgpt.com">AI 导致了那些裁员吗？纽约雇主可能必须披露。</a>：纽约州……</li>

宣布了一项重要举措，通过要求企业披露明确与 AI 采用相关的裁员情况，来应对 AI 对劳动力的潜在影响</li><li><a href="https://buttondown.com/ainews/archive/ainews-titans-learning-to-memorize-at-test-time/">[AINews] Titans: Learning to Memorize at Test Time</a>：Neural Memory 就是你所需要的一切。2025/1/14-2025/1/15 的 AI 新闻。我们检查了 7 个 subreddits、433 个 Twitters 和 32 个 Discords（219 个频道和 2812 条消息）...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1329234711913959515)** (4 messages): 

> `Modular subreddit, GitHub organization move` 


- **Modular 发布官方 Subreddit**：现在有了**官方 Modular subreddit**！欢迎加入社区 [r/ModularAI](https://www.reddit.com/r/ModularAI/) 🥳。
   - *“这就是我们要走的路！”* 一位成员在回应公告时感叹道。
- **GitHub 仓库迁移**：Modular 的公共 GitHub 仓库已从 [**ModularML**](https://github.com/modularml) 组织迁移到 [**Modular**](https://github.com/modular) 组织。
   - 所有之前的链接都应自动重定向，但团队鼓励报告在过渡期间出现的任何**意外问题**。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1329253563108425738)** (28 messages🔥): 

> `Recursive Types in Mojo, SIMD Performance Concerns, Variadic Lists to Dictionary Implementation Issues` 


- **Mojo 中的递归类型处理**：一位成员指出了在 Mojo 中实现递归类型的挑战，建议使用 `UnsafePointer` 可能会导致问题，并建议在 List 上使用拷贝构造函数（copy constructor）。
   - 他们还提到了在调试运行（debug running）时遇到的一些问题，并暗示目前对递归类型的支持并不完善。
- **SIMD 的性能陷阱**：成员们讨论了 SIMD 可能存在的*性能陷阱*，指出根据架构的不同，SIMD 并不总是能保证比标量（scalar）版本有更好的性能。
   - 一位成员指出，SIMD 是否能提供性能提升很大程度上取决于具体的 CPU 及其优化。
- **Variadic List 到 Dictionary 的转换问题**：一位成员分享了在尝试将 VariadicList 拆分为字典时遇到的困难，在字符串捕获赋值时遇到了意外行为。
   - 另一位成员提供了一种解决方法，即在构建字典之前将参数复制到单独的列表中，从而避免了这种异常行为。
- **Mojo 中的可选参数处理**：关于 Mojo 类中可选参数问题的讨论也随之出现，因为一位成员在求值为 None 时遇到了段错误（segmentation faults）。
   - 建议包括查看特定的 GitHub issues 以获取有关可选参数的修复和示例。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html">Ice Lake AVX-512 Downclocking</a>：研究 Intel Ice Lake CPU 上与 AVX 相关的降频程度</li><li><a href="https://github.com/modularml/mojo/issues/3950">[Help wanted] Evaluating optional argument to not None gives segmentation fault · Issue #3950 · modular/mojo</a>：问题描述：我有一个需要可选参数的类。当求值为 None 时，它会报错。如果求值为 None 报错，我该如何求值？而且，我也可能...</li><li><a href="https://github.com/modularml/mojo/issues/3917">[BUG] --debug-level full crashes when importing · Issue #3917 · modular/mojo</a>：Bug 描述：使用调试器运行 mojo 脚本会发生段错误，而运行常规 mojo 时则能运行完成（尽管我也注意到常规脚本中存在奇怪的行为...）
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1329188117319647262)** (2 条消息): 

> `Agent Identity Hackathon, Xeno Grant Applications` 


- **加入 Agent Identity Hackathon！**：快来加入 [Plastic Labs](https://plasticlabs.ai) 和 [Betaworks](https://betaworks.com) 举办的 Agent 身份黑客松，以此拉开 [Xeno Grant](https://xenogrant.org) 的序幕。专注于 Agent 身份的杰出项目将获得总计 **5,000 美元** 的奖金。
   - 参与者可以个人或团队形式参赛，现场提供餐饮，并在前一晚举行启动交流会。
- **Xeno Grant 申请即将截止！**：Xeno Grant 申请将于 **1 月 26 日（星期日）** 截止，鼓励参与者在注册时分享其 GitHub、作品集或个人网站，以便获得审核批准和候补名单优先级。
   - 此次机会邀请所有 Agent 开发者参与，申请者将有机会与 **Plastic/Betaworks 资助委员会** 见面。



**提到的链接**：<a href="https://lu.ma/5rlcrlpb">Xeno Grant: Agent Identity Hackathon · Luma</a>：加入 Plastic Labs 和 Betaworks 举办的 Agent 身份黑客松，开启 Xeno Grant（由 $YOUSIM 提供支持）。最引人注目的项目将获得 5,000 美元奖金……

  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1329188540789293228)** (9 条消息🔥): 

> `LiveCodeBench Update, Cerebras Yield Problem, Contextual AI Platform Launch, TGI Backend Expansion, SWE-bench Multimodal Evaluation` 


- **LiveCodeBench 达成里程碑，新增 167 个问题**：**LiveCodeBench** 的最新更新增加了 **167 个新问题**，使总数达到 **880 个**——相比版本 1 的 **400 个** 有了显著增长。
   - 此次更新展示了 o1、**Gemini-Flash** 以及即将推出的 R1 等推理模型的改进，引发了社区的热烈讨论。
- **Cerebras 重新思考芯片良率**：Cerebras 宣传其 **晶圆级芯片（wafer-scale chip）** 比传统芯片大 **50 倍**，同时实现了相当的良率，挑战了传统的半导体常识。
   - 他们的详细分析对比了 **Cerebras Wafer Scale Engine** 与 H100 尺寸芯片的良率，断言尽管尺寸巨大，但其具备更好的容错能力。
- **Contextual AI 平台庆祝发布**：**Contextual AI 平台** 正式发布，该平台由 **Meta 的 Llama 3.3** 提供支持，运行在 **Google Cloud** 上并利用 **NVIDIA GPUs**，标志着一个重要的里程碑。
   - 团队对包括 **NVIDIA**、**Meta** 以及其他为此成就做出贡献的风险投资合伙人和投资者表示了感谢。
- **TGI 成为推理框架中的 Keras**：Hugging Face 的 **Text-Generation-Inference (TGI)** 持续进化，现在支持包括 **AMD** 和 **Google TPU** 在内的多个后端。
   - TGI 因其易于部署而受到赞誉，提供了一种无代码解决方案，帮助开发者在各种平台上高效运行 LLM。
- **SWE-bench 引入多模态评估代码**：新的 **SWE-bench MM** 包含了专注于视觉组件的 JavaScript 问题，增强了多模态评估能力。
   - 新问题的示例包括“地图渲染不正确”和 UI Bug 等渲染问题，扩展了其评估框架的范围。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://cerebras.ai/blog/100x-defect-tolerance-how-cerebras-solved-the-yield-problem">100x Defect Tolerance: How Cerebras Solved the Yield Problem - Cerebras</a>：未找到描述</li><li><a href="https://x.com/jyangballin/status/1879990781030854897">John Yang (@jyangballin) 的推文</a>：SWE-bench 多模态评估代码现已发布！SWE-bench MM 是一组具有视觉组件的新 JavaScript 问题（如“地图渲染不正确”、“按钮文本未出现”）。</li><li><a href="https://x.com/ContextualAI/status/1879563309080547376">Contextual AI (@ContextualAI) 的推文</a>：Contextual AI 平台自豪地基于 Meta 的 Llama 3.3 构建，运行在 Google Cloud 上，并在 NVIDIA GPUs 上进行训练。我们对这一里程碑感到非常自豪，并感谢所有的客户、合作伙伴……</li><li><a href="https://huggingface.co/blog/tgi-multi-backend">Introducing multi-backends (TRT-LLM, vLLM) support for Text Generation Inference</a>：未找到描述</li><li><a href="https://x.com/StringChaos/status/1879619028651745287">Naman Jain (@StringChaos) 的推文</a>：📢 很高兴分享 LiveCodeBench 的第 5 次更新。这次我们增加了 167 个新问题，总计收集了 880 个问题，比 v1 的 400 个问题增加了一倍多。排行榜 ⬇️- 🥇 open a...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1329252427832229980)** (8 条消息🔥): 

> `AMD GPU Support, Intel Sponsorship, Ai2 Funding` 


- **AMD 应该投资 Ai2**：一名成员建议 **AMD** 应该资助整个 **Ai2** 并提供大量的 GPU 资源，提议给每位成员分配 **$10k**。
   - *“为什么不利用 AMD 的 GPU？”* 是社区成员提出的潜在问题，强调了对更好模型可访问性的需求。
- **靠近 AMD 总部的地理优势激发灵感**：另一名成员提到距离位于 **Santa Clara** 的 AMD 总部仅 **200 码**，正考虑直接走过去找 **Lisa Su** 谈谈。
   - 他们开玩笑地建议给她发私信，呼吁分发更多 GPU，并表示：*“除了我们，还有谁会要它们呢？”*
- **与 Tensorwave 的竞争亮点**：讨论中包含了一个指向 **Tensorwave** 的链接，强调了他们专注于由 **AMD MI300X** 加速器驱动的企业级 AI 计算解决方案。
   - Tensorwave 标榜其系统具有“更快、可扩展且更易于使用”等优势，并强调作为 MI300X 的发布合作伙伴，可立即提供服务。
- **Intel 的战略赞助**：一位成员提到了 **Intel** 赞助 **Stability AI** 项目的明智举动，并将其与 AMD 目前的参与方式进行了对比。
   - 这引起了人们对竞争格局以及公司在 AI 领域确保合作伙伴关系所采取策略的关注。



**提到的链接**：<a href="https://tensorwave.com/">立即访问 MI300X GPU | TensorWave | MI300X 云</a>：立即在 TensorWave Cloud 上访问 AMD MI300X GPU。今天就联系我们开始使用。

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1329272230076481709)** (2 条消息): 

> `Terminology for non-reasoning models, GPT-4o, Autoregressive models` 


- **定义非推理模型**：关于非推理模型的正确术语展开了讨论，特别是将其与像 **o1** 这样的推理模型进行对比。
   - “Vanilla ass basic autoregressive model”（普通基础自回归模型）被建议作为 **GPT-4o** 的非正式术语。
- **模型类型的澄清**：参与者寻求关于 **GPT-4o** 在模型类型光谱中位置的澄清，特别是相对于基于推理的模型。
   - *Vanilla* 一词表示与其它更复杂的架构相比，这是一种更基础的自回归方法。


  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/)** (1 条消息): 

natolambert: 哇，怀旧。这本应该更……
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1329196458624290928)** (4 条消息): 

> `Human Decision-Making vs AI, Technology Cycle Expectations, Social Movements against AI, Public Understanding of LLMs` 


- **配备“下一步最佳行动”系统的人类表现优于 AI**：一位参与者指出，配备了 **next best action system** 的人类在决策复杂性方面很难被超越。
   - 这反映了关于人类直觉与机器学习实际能力的持续讨论。
- **被误估的技术周期**：一位成员对技术创新从根本上改变经济的速度表示怀疑，认为对“无限盒子 (infinity box)”重新设计基础设施的预期是不切实际的。
   - 他们预计，如果发生如此迅速的转型，将会出现重大的**社会动荡**。
- **缺乏反对 AI 的社会运动**：评论者指出，反对 AI 的严肃社会运动尚未出现，这可能会显著影响技术的未来格局。
   - 这引发了关于公众对新兴技术的准备程度和反应能力的问题。
- **公众在使用 LLMs 时的挣扎**：一位参与者对人们利用大语言模型 (LLMs) 的糟糕程度表示沮丧，将这项技术比作用户无法掌握的“魔法”。
   - 尽管提供了清晰的指令，用户经常生成**陈词滥调的结果**，揭示了在理解 LLM 功能方面的差距。


  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1329236127244353586)** (1 messages): 

> `Project Aria, Meta 通讯` 


- **Project Aria 更新订阅服务上线**：Meta 宣布了 [Project Aria](https://www.projectaria.com/) 的更开放版本，用户可以订阅以获取最新更新。
   - *通过提供您的电子邮件，即表示您同意接收来自 Meta 的营销相关电子通讯*，包括有关 Project Aria 的新闻和活动。
- **Meta 的数据政策透明度**：Meta 提供了关于他们如何处理用户数据的见解，并敦促用户阅读其 [Data Policy](https://www.projectaria.com/privacy-policy/)。
   - 他们强调，用户可以随时通过电子邮件中的退订链接取消订阅通讯。



**提到的链接**：<a href="https://www.projectaria.com/">Introducing Project Aria, from Meta</a>：Project Aria 是来自 Meta 的一项研究计划，旨在帮助负责任地构建未来。Project Aria 开启了我们与世界连接和体验世界的新可能性。

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1329212731655585812)** (3 messages): 

> `Vellum AI 合作伙伴关系, Neomagus 赢得 LLM x Law 黑客松, Women in AI RAG 黑客松` 


- **Vellum AI 调查合作伙伴关系**：我们很高兴在这次调查中与 **@vellum_ai** 合作；快来查看 [用例数据！](https://t.co/0zvs7ZkKJQ)
- **Neomagus 在 LLM x Law 黑客松中获胜**：了解 **Neomagus** 如何凭借一个旨在确保 **AI 生成的法律信息准确性** 的项目赢得 LLM x Law 黑客松。
   - 他们的解决方案具有法律引用的 **实时验证** 和不准确信息的即时标记功能（[更多详情](https://t.co/jEqZrnn11H)）。
- **Women in AI RAG 黑客松邀请**：诚邀技术领域的女性参加在帕罗奥图举行的 **Women in AI RAG 黑客松**，重点关注使用开源 [向量数据库 @zilliz_universe](https://t.co/2Bzg80dh29) 的 **Retrieval-Augmented Generation** (RAG)。
   - 这场全天活动提供了与女性技术同行和导师交流的机会（[更多信息](https://t.co/ey8ebq9fbx)）。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1329274593835352084)** (16 messages🔥): 

> `注册问题已解决, ChromaDB 中的元数据过滤, LLM 与 llmlingua 的集成, 标签提取方法` 


- **注册问题已解决**：用户之前遇到了注册问题，但已确认问题源于 **auth 升级**，目前已修复。
   - 一位用户报告在修复后成功登录，并澄清这只是一个 **临时错误**。
- **ChromaDB 中的元数据过滤**：一位用户询问关于在从数千个文档创建的大型向量库中使用 **ExactMatchFilters** 配合元数据来过滤法律案例的问题。
   - 他们对创建用于路由的子索引表示担忧，质疑其与现有过滤方法相比的有效性和性能。
- **LLM 与 llmlingua 的集成**：一位成员讨论了增强 LlamaIndex 与 **llmlingua2** 的集成，但在过程中遇到了 MakeFile 的代码检查（linting）问题。
   - 另一位成员提供了帮助，建议安装 `pre-commit` 进行自动代码检查，或者手动运行 `make lint` 进行修复。
- **标签提取方法**：一位用户询问，是使用单独的 LLM 调用来优化产品描述和提取标签更有效，还是将它们合并到单个调用中更好。
   - 他们指出了在每天调用量很大的情况下，**标签质量**与**延迟/成本**之间的权衡。



**提到的链接**：<a href="https://github.com/run-llama/llama_index/pull/17531">Add longllmlingua2 integration by tituslhy · Pull Request #17531 · run-llama/llama_index</a>：描述：添加了 LLMLingua 2 集成！LLMLingua2 是对 LLMLingua1 的改进，使用了一种较小尺寸的提示词压缩方法，通过 GPT-4 的数据蒸馏进行训练，用于 Token 分类...

  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1329464136060502018)** (1 messages): 

> `text-to-SQL 流水线` 


- **20 分钟完成 Text-to-SQL 流水线**：一位成员分享了他们在短短 **20 分钟** 内创建 **text-to-SQL 流水线** 的经验，并对过程的简易性表示惊讶。
- **Text-to-SQL 的初步成功**：这是他们第一次尝试此类任务，强调了所用工具的 **用户友好性**。
   - 他们不敢相信设置竟然如此简单，并强调这是一个重要的学习机会。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/)** (1 messages): 

jimmc414_00230：有关于什么时候可以期待 DSPy v3 的消息吗？
  

---

### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1329551564049879100)** (2 条消息): 

> `dspy ReAct usage, Addition function error, LLama model issues` 


- **dspy ReAct 函数用法错误**：一位用户遇到了错误，提示工具 addition 并非设计用于计算两个数字之和，且缺少必要的参数。
   - 他们提到使用了通过 LM-Studio 托管的 LLama 模型，并向社区寻求帮助。
- **加法函数返回错误输出的问题**：加法函数定义正确，但存在语法错误，因为它返回的是 'retur' 而不是 'return'。
   - 这可能导致了功能崩溃，使得用户在执行该函数时收到错误消息。


  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1329472534407544884)** (4 条消息): 

> `Ideal chat template, Torchtune integration` 


- **寻求理想的聊天模板**：@duh_kola 询问了关于**理想聊天模板**的问题，引发了关于使用 ChatML 或 Llama3（如果采用该路线）的讨论。
   - 该查询突显了关于优化聊天界面的持续讨论。
- **Torchtune 集成挑战**：一名成员指出，集成 **Torchtune** 目前涉及“剥离大量内容”，暗示该过程具有相当高的复杂性。
   - 对话表明对该集成的关注有所滞后，**caseus_** 幽默地评论说距离上次处理这个问题已经有一段时间了。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1329489319039074444)** (1 条消息): 

> `Cooperative AI Summer School, Confirmed Speakers, Application Details, Sortition in Democracy, Financial Assistance` 


- **Cooperative AI 暑期学校申请开放**：**Cooperative AI Summer School** 的申请截止日期为 **2025 年 3 月 7 日**，活动将于 **2025 年 7 月 9 日至 13 日**在伦敦附近的 Marlow 举行。
   - 该项目面向 AI、计算机科学及相关领域的学生和早期职业专业人士，重点关注社交网络和职业发展。
- **公布知名讲师名单**：已确认的讲师包括来自密歇根大学的 **Michael Wellman** 和来自 The Collective Intelligence Project 的 **Zarinah Agnew**，他们为学校带来了多元化的专业知识。
   - 来自哈佛大学的 **Ariel Procaccia** 也将出席，讨论公民议会中的随机参与者选择算法。
- **全面的项目见解**：暑期学校将探索 **cooperative AI** 的基础概念和前沿研究，旨在吸引具有影响力驱动力的候选人。
   - 讲座将涵盖 cooperative AI 研究最前沿的目标和方法。
- **提供财务援助**：为确保参与者能够负担得起，暑期学校提供财务援助。
   - 该倡议旨在支持对 cooperative AI 表现出浓厚兴趣的候选人，无论其财务状况如何。
- **在 Cooperative AI 网站了解更多信息**：欲了解更多信息并申请，请访问 **Cooperative AI 官方网站**：[cooperativeai.com](https://www.cooperativeai.com/summer-school/summer-school-2025)。
   - 网站还包含有关申请评估流程的常见问题解答（FAQ）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://strategicreasoning.org/michael-p-wellman/">Michael P. Wellman &#8211; Strategic Reasoning Group</a>: 未找到描述</li><li><a href="https://www.zarinahagnew.com/">z a r i n a h  :   a g n e w</a>: 个人简介 : 愿景声明</li><li><a href="https://procaccia.info/">Home - Ariel Procaccia</a>: 未找到描述</li><li><a href="https://www.cooperativeai.com/summer-school/summer-school-2025">Cooperative AI</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1329235959107162213)** (2 条消息): 

> `Cost of solutions, Churn prevention strategies` 


- **成本驱动方案决策**：*成本*是选择成熟方案的重要因素，正如一位成员指出，这是坚持使用现有有效方案的主要原因。
   - 这突显了预算在工具和策略决策中的重要性。
- **咨询流失预防趋势**：一名成员对*流失预防/规避（churn prevention/aversion）*的最新发展表示感兴趣，他已经离开该领域两年多了。
   - 他们正在寻求指导，了解从哪里开始可以快速跟上当前的行业现状。


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1329206651978387526)** (2 条消息): 

> `Bora's Law, Open Interpreter 功能, 自主系统开发` 


- **Bora's Law 挑战传统的 AGI 扩展方式**：一位成员批评了 OpenAI 的 AGI 实现方法，认为**智能随约束而扩展**，而非算力，并引用了 [Bora's Law: Intelligence Scales With Constraints, Not Compute](https://chrisbora.substack.com/p/boras-law-intelligence-scales-with?r=aszci) 中的见解。该原则认为，过度关注扩展算力忽略了对真正智能发展至关重要的数学关系。
   - *文章建议，对智能的追求应从暴力破解转向理解约束如何发挥关键作用。*
- **对 Open Interpreter 代码执行功能的担忧**：一位成员表示担心 Open Interpreter 1.0 可能取消了直接代码执行功能，将其限制为仅命令行操作。这一功能对于提高效率和提示 LLM 学习至关重要。
   - 社区对改进代码执行方面表现出兴趣，提议通过添加 Python 便捷函数来帮助 LLM 更有效地学习新技能。



**提到的链接**：<a href="https://chrisbora.substack.com/p/boras-law-intelligence-scales-with?r=aszci">Bora&#x27;s Law: Intelligence Scales With Constraints, Not Compute</a>：这是一篇探讨人工智能发展中新兴原则的工作论文。

  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1329437460802965588)** (2 条消息): 

> `Jamba API 性能, 与 OpenAI 的对比分析` 


- **Jamba API 的性能给用户留下深刻印象**：@bjorn02796 分享称他们已在多个应用程序中运行 **Jamba API**，并对其性能非常满意。
   - *如果它的表现优于 OpenAI 的响应*，这将引发对 OpenAI 当前标准的质疑。
- **用户反馈得到认可**：针对该反馈，用户 **keepitirie** 对这些正面评价表示感谢。
   - 这突显了社区的参与度以及对 **Jamba API** 有效性的认可。


  

---


---


---


---


---


---


{% else %}


> 完整的频道逐条分析已在邮件中截断。 
> 
> 如果你想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}