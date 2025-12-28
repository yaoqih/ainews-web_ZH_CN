---
companies:
- tencent
- openai
- bytedance
- meta-ai-fair
- nvidia
- deepseek
date: '2025-05-13T05:44:39.731046Z'
description: '以下是为您翻译的中文内容：


  **腾讯的 Hunyuan-Turbos** 在 LMArena 排行榜上已升至第 8 位，在各大主要类别中表现强劲，且自 2 月以来取得了显著进步。**Qwen3（通义千问3）模型家族**，尤其是
  **Qwen3 235B-A22B（推理版）** 模型，因其出色的智能水平和高效的参数利用率而备受关注。**OpenAI** 推出了 **HealthBench**，这是一个在
  **250 多名医生**的参与下开发的新型健康评估基准，**o3**、**GPT-4.1 nano** 和 **Grok 3** 等模型在该基准测试中表现优异。**字节跳动**发布了
  **Seed1.5-VL**，这是一款视觉语言模型，配备了 5.32 亿参数的视觉编码器和 200 亿激活参数的 MoE（混合专家）大语言模型，在 38 个公开基准测试中达到了业内领先水平（SOTA）。在视觉语言领域，**可灵
  (Kling) 2.0** 在图生视频领域处于领先地位，而 **Gemini 2.5 Pro** 凭借先进的多模态能力在视频理解方面表现出色。此外，Meta 的“视觉-语言-动作”（Vision-Language-Action）框架以及
  2025 年视觉语言模型（VLM）的最新动态也受到了重点关注。'
id: MjAyNS0w
models:
- hunyuan-turbos
- qwen3-235b-a22b
- o3
- gpt-4.1-nano
- grok-3
- gemini-2.5-pro
- seed1.5-vl
- kling-2.0
people:
- lmarena_ai
- artificialanlys
- gdb
- _jasonwei
- iScienceLuvr
- _akhaliq
- _philschmid
- teortaxesTex
- mervenoyann
- reach_vb
title: 今天没发生什么特别的事。
topics:
- benchmarking
- model-performance
- moe
- reasoning
- vision
- video-understanding
- vision-language
- multimodality
- model-evaluation
- model-optimization
---

**平静的一天。**

> 2025年5月12日至5月13日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（214 个频道，4553 条消息）。预计节省阅读时间（以每分钟 200 字计）：445 分钟。我们的新网站现已上线，提供完整的元数据搜索和美观的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

Gergely Orosz 撰写了一篇关于 [ChatGPT Images 发布](https://newsletter.pragmaticengineer.com/p/chatgpt-images)的高质量文章，Simon Willison 对其进行了[摘录](https://simonwillison.net/2025/May/13/launching-chatgpt-images/)。[WizardLM 团队离开 MSR China 加入了腾讯](https://x.com/WizardLM_AI/status/1922307494837186998)，并巧合地发布了腾讯混元 (Hunyuan-Turbos)，这是一个闭源模型，但目前在 [LMArena 上排名中国模型第一](https://x.com/lmarena_ai/status/1921966648795533459)。

[AI Engineer World's Fair 还有 20 张全会期早鸟票](https://ti.to/software-3/ai-engineer-worlds-fair-2025/discount/AINEWS)，距离开幕还有 3 周，[演讲者、研讨会和活动名单](https://www.ai.engineer/#speakers)已进一步确定。

---

# AI Twitter 回顾

**语言模型与基准测试 (Language Models and Benchmarks)**

- **Hunyuan-Turbos 在排行榜上的表现**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1921966654256197814) 分享了完整排行榜的链接。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1921966651655717217) 提到 **Hunyuan-Turbos 在所有类别中均排名前 10（风格控制除外，排名第 13）**。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1921966648795533459) 报告称 **腾讯的 Hunyuan-Turbos 目前排名第 8**，强调其综合排名第 8（风格控制第 13），在主要类别（困难、编程、数学）中均进入前 10，且相比 2 月份的版本有显著提升（综合排名从第 21 位升至第 8 位）。
- **Qwen3 模型系列分析**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1922317655643717887) 对 **Qwen3 模型系列** 进行了详细分析，强调 **Qwen3 235B-A22B (Reasoning)** 模型在 Artificial Analysis 智能指数中获得了 **62** 分，成为史上最智能的 open weights 模型。该模型仅有 **22B 激活参数**，总参数为 **235B**，相比之下，竞争对手如 NVIDIA 的 Llama Nemotron Ultra（dense，253B）和 DeepSeek R1（37B 激活，671B 总参数）。分析还指出了 MoE 模型的优势以及推理能力对所有模型的持续提升。
- **OpenAI 的 HealthBench 评估**：[@OpenAI](https://twitter.com/OpenAI/status/1921983050138718531) 宣布了 **HealthBench**，这是一个在**全球 250 多名医生**的参与下开发的新评估基准。[@gdb](https://twitter.com/gdb/status/1921987974356443595) 也强调了 HealthBench 的发布。[@_jasonwei](https://twitter.com/_jasonwei/status/1922002699240775994) 注意到了这一在医疗 AI 领域的投入，提到 **o3 得分为 60%**，其中 **GPT-4.1 nano 的表现优于 GPT-4o，且成本降低了 25 倍**。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1922013874687246756) 分享了关于 HealthBench 的细节，指出 **o3 是表现最好的模型，得分为 60%**，紧随其后的是 Grok 3 (54%) 和 Gemini 2.5 Pro (52%)。
- **字节跳动的 Seed1.5-VL**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1922226964599095740) 分享了 **Seed1.5-VL** 的技术报告，该模型由 **532M 参数的视觉编码器和 20B 激活参数的 MoE LLM** 组成，在 **60 个** 公开基准测试中的 **38 个** 达到了 SOTA 性能，在 GUI 控制和游戏表现上优于 OpenAI CUA 和 Claude 3.7。[@_akhaliq](https://twitter.com/_akhaliq/status/1922318117385932993) 报告称字节跳动刚刚在 Hugging Face 上发布了 Seed1.5-VL。

**视觉语言模型 (Vision Language Models)**

- **Kling 2.0 图生视频模型**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1922299716051796148) 宣布 **Kling 2.0 目前是领先的图生视频模型**，超越了 Veo 2 和 Runway Gen 4，具有强大的提示词遵循能力和视频质量。
- **Gemini 2.5 Pro 视频理解能力**：[@_philschmid](https://twitter.com/_philschmid/status/1921838835735867533) 强调了 **Gemini 2.5 Pro 的视频理解能力**，指出它可以在 200 万上下文（2 million context）中以“低分辨率”处理长达 6 小时的视频，原生结合了视听理解与代码，并支持检索和时间推理。
- **Meta 的视觉-语言-动作框架**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1921774079834529862) 提到了来自 AGIBot 的 Meta Vision-Language-Action 框架。
- **VLMs 2025 更新**：[@mervenoyann](https://twitter.com/mervenoyann/status/1921962750353301986) 分享了一篇关于视觉语言模型最新进展的博客，包括 GUI agents、多模态 RAG、视频 LMs 和 smol models；[@reach_vb](https://twitter.com/reach_vb/status/1921974792242016591) 宣布了博文《从零到英雄：视觉语言模型全指南——从多模态到推理，再到 MoEs、基准测试等》。

**AI 工程与工具**

- **利用 AI 改进代码库**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1921967025628578230) 讨论了 **AI 帮助让代码库变得更美观**的潜力，强调 AI 作为一个勤奋的团队成员，可以建议更改并提高人类和 LLMs 的理解力。
- **用于文档结构化的 DSPy**：[@lateinteraction](https://twitter.com/lateinteraction/status/1922156400559395064) 讨论了使用 **DSPy 脚本**来结构化 DSPy 文档转储，强调了处理大量字符数的挑战以及所采取的方法（类似于 STORM 项目）。
- **用于推荐系统的 KerasRS**：[@fchollet](https://twitter.com/fchollet/status/1922346095302025417) 分享了一个资源，介绍如何使用 Keras 和 JAX 以及新的 KerasRS 软件包在 10 分钟内构建和训练推荐系统。
- **AI 咨询与 RAG**：[@jxnlco](https://twitter.com/jxnlco/status/1922007873862651914) 为 **AI 顾问**分享了建议，强调寻找有痛点的客户、建立公信力并设定最低参与标准。[@jxnlco](https://twitter.com/jxnlco/status/1922003672701018219) 表示 **基于文本的 RAG 已经过时**，真正的竞争优势在于构建能够理解图表、图形和图像的系统。
- **LangChain Interrupt 活动**：[@LangChainAI](https://twitter.com/LangChainAI/status/1922351748565385604) 分享了 **LangChain Interrupt** 活动，涵盖了关于构建可靠 agents 的研讨会，并为无法到场的人提供实时推文更新。[@TheTuringPost](https://twitter.com/TheTuringPost/status/1921992585423593608) 指出，在 Sequoia AI Ascent 上，LangChain CEO @hwchase17 谈到了 ambient agents（环境智能体），它们与聊天 agents 不同，并强调了 human-in-the-loop 的重要性以及 LangChain 为 ambient agents 所做的开发。
- **Windsurf AI 及其 CEO @Windsurf_AI 将于 6 月 18 日在 Fully Connected 舞台展示 AI 代码智能如何推动 agents 从创意走向生产**：[@weights_biases](https://twitter.com/weights_biases/status/1922332818127892986) 提到了 Mohansolo。
- **构建 Agentic 系统**：[@mathemagic1an](https://twitter.com/mathemagic1an/status/1921995940346659232) 介绍了带有代码 agents 的 @zoom，强调了它们在设计评审和事件管理中的应用。

**模型发布与性能**

- **阿里巴巴 Qwen3 量化模型**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1921907010855125019) 宣布发布 **Qwen3 量化模型**，可通过 Ollama、LM Studio、SGLang 和 vLLM 部署，支持包括 GGUF、AWQ 和 GPTQ 在内的多种格式。[@reach_vb](https://twitter.com/reach_vb/status/1921956656226668964) 指出 Qwen 刚刚发布了针对 Qwen3 优化的 GPTQ、GGUF 和 AWQ 格式。
- **Meta 的 Dynamic Byte Latent Transformer**：[@AIatMeta](https://twitter.com/AIatMeta/status/1921966366707613924) 宣布发布其 **8B 参数 Dynamic Byte Latent Transformer** 的模型权重，这是传统分词（tokenization）方法的一种替代方案，旨在提高语言模型的效率和可靠性。
- **Skywork-VL Reward**：[@_akhaliq](https://twitter.com/_akhaliq/status/1922326980680138925) 撰写了关于 Skywork-VL Reward 的文章，这是一种用于多模态理解和推理的有效奖励模型（Reward Model）。
- **PrimeIntellect 的 Intellect 2**：[@reach_vb](https://twitter.com/reach_vb/status/1921948704061202725) 宣布 @PrimeIntellect 开源了 Intellect 2 —— 一个通过分布式异步 RL 使用 GRPO 进行后训练的 32B 推理模型。

**HuggingFace 与推理**

- **得益于 Hugging Face Inference Endpoints 和 @vllm_project，比 @openai Whisper API 快 8 倍且更便宜！**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1922383289408491629) 分享了关于这一优化的消息。
- **使用 Inference Endpoints 实现极速 Whisper 转录**：[@_akhaliq](https://twitter.com/_akhaliq/status/1922315470478139537) 指出使用 Inference Endpoints 可以实现极速 Whisper 转录。
- **用于推理的自定义 Speculators**：[@togethercompute](https://twitter.com/togethercompute/status/1921983794573197538) 讨论了通过使用自定义 Speculators 为推理客户获得大幅提速，并指出其优势包括推理速度提升约 1.3 倍以及成本降低约 25%。
- [@reach_vb](https://twitter.com/reach_vb/status/1922324889593102584) 报道了新进展：**仅凭单张 L4 即可实现高达 8 倍速的 Whisper 转录，由 @vllm_project 提供支持** 💥

**职业与行业趋势**

- **Cartesia 正在组建印度团队**：[@krandiash](https://twitter.com/krandiash/status/1922016592621404407) 宣布 Cartesia 正式在班加罗尔组建印度团队，初期由 5 人的线下团队组成，正在寻找具有 ML 系统经验的资深 SWE。
- **领域专业知识的持久重要性：** [@JvNixon](https://twitter.com/JvNixon/status/1921765048189616138) 认为 Cursor、Lovable、Windsurf 和 Bolt 等平台的兴起，源于对各自领域内问题的深刻理解，而不仅仅是因为代码是最好的 LLM 应用。
- **AI 对工作的影响**：[@zachtratar](https://twitter.com/zachtratar/status/1922071000142758377) 分享了一些轶事，提到高中生注意力持续时间缩短的现象也正发生在职场成年人身上，经理们报告称员工注意力分散、专注能力下降，并且需要更小、更简单的工作单元。
- **AI 基础设施领域的领导地位**：[@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1922320072590098794) 提到，美国的领导地位对于赢得 AI 基础设施竞赛和充分发挥 AI 推理能力至关重要，并提到他陪同特朗普总统对沙特阿拉伯王国进行了历史性访问，以加速美国在全球 AI 基础设施方面的创新。
- **工业界 vs 学术界**：[@swyx](https://twitter.com/swyx/status/1921704173118050717) 指出 https://www.aiengineer.ai/ 的存在意义在于以工程师/行业评审者和产品为中心，而不是以博士/学术界和论文为中心。

**梗/幽默**

- **Ilya 看到了什么**：[@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1922031056225439852) 简单地写了句 "what ilya saw" 并分享了一张图片。
- **男性往往会变弯**：[@typedfemale](https://twitter.com/typedfemale/status/1921699387425603670) 写道：“**在海军、监狱等男性占多数的场所，男性往往会变弯……只能想象 xAI 现在正在发生什么**”。
- **推理模型这，推理模型那**：[@lateinteraction](https://twitter.com/lateinteraction/status/1922383824857579884) 写道：“推理模型（Reasoning model）这，推理模型那。我想要的只是一个合理的模型（Reasonable model）。”
- **你甚至该怎么称呼这个？**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1922016949300552073) 分享了一张图片并问道：“你甚至该怎么称呼这个？”
- [@scaling01](https://twitter.com/scaling01/status/1921716325971345759) 写道：“结束了，什么都没发生，o3 pro 之后的 Grok 3.5，准备好我的演讲和梗图了。”
- **AI 实验室花了几年时间悄悄扩展监督学习，其最佳结果显而易见：一个出色的人类文本模拟器。现在他们正在扩展强化学习，这是本质上不同的东西。没人知道接下来会发生什么**：[@jxmnop](https://twitter.com/jxmnop/status/1922078186864566491) 分享了他们的看法。
- **我现在 20 多岁了（顺便说一下是女性）。这听起来可能很奇怪，但我真的认为上帝把我带到这个世界上是为了给轻度自闭症男性的生活带来温暖**：[@typedfemale](https://twitter.com/typedfemale/status/1922051667081503028) 写下了这段金句。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 1. Qwen3 模型发布及技术细节

- [**Qwen3 技术报告**](https://i.redd.it/kku7lzsulj0f1.jpeg) ([Score: 409, Comments: 53](https://www.reddit.com/r/LocalLLaMA/comments/1klkmah/qwen3_technical_report/)): **该图片展示了新发布的 Qwen3 技术报告封面，重点介绍了 Qwen3 在语言建模方面相较于前代版本的改进，例如增强的推理模式和用于更高效资源分配的新型“思考预算”（thinking budget）机制。托管在 GitHub 上的随附报告详细列出了广泛的基准测试（超过 15 页），将各种规模的 Qwen3 模型（包括基础版和混合专家模型 MoE 变体）与之前的模型及竞争对手进行了对比，所有变体均在 36T tokens 上进行了训练。新发现表明，Qwen3-30B-A3B MoE 模型的性能足以媲美或超越更大规模的稠密模型，挑战了典型的 MoE 等效性估算。报告还强调了复杂的训练后创新，如 Thinking Mode Fusion 和 RL，尽管并非所有提到的模型（如 32B-Base, 235B-A22B）都已按照 Apache 2.0 声明发布开放权重。** 评论者注意到了技术上的详尽性，但对大型模型缺乏真正的开放权重表示失望，指出了许可声明与实际可访问性之间的差异。此外，对于 MoE 模型的基准测试方法和报告的训练后策略，社区也存在技术上的好奇和争论。
    - Qwen3 技术报告提供了超过 15 页的基准测试，包括推理（“思考”）模式的独立结果、全面的基础模型性能以及训练后过程的细节，特别是 "Thinking Mode Fusion" 和 RL 的应用。所有 Qwen3 模型，甚至是 0.6B 版本，都共享 36T tokens 的预训练数据集规模，这与 Qwen2.5 一致，但不同于 Gemma3 或 Llama3.2。
    - Qwen3-30B-A3B 是一款备受推崇的 MoE 模型，根据基准测试，其表现与更稠密的 Qwen3-14B 相当甚至更好——这矛盾于“MoE 性能可以通过激活参数与总参数的几何平均值来预测”的预期。这一发现表明，激活参数较少的 MoE 模型可能会超出预期表现，从而可能影响未来的架构选择。
    - 报告重点关注了开启和关闭“思考”模式下的实证基准测试，在第 17 页尤为显著：使用“思考”模式在编程任务中带来了显著提升。基准测试显示，Qwen3-30B-A3B 的 GPQA Diamond 分数在开启思考模式时为 `65.8`，关闭时为 `54.8`；同时，量化版本（2-4bpw）的分数更低（`42-49`），证明了该模式的实质性影响。
- [**Qwen3 聊天模板*仍然存在 Bug***](https://www.reddit.com/r/LocalLLaMA/comments/1klltt4/the_qwen3_chat_template_is_still_bugged/) ([Score: 148, Comments: 50](https://www.reddit.com/r/LocalLLaMA/comments/1klltt4/the_qwen3_chat_template_is_still_bugged/)): **用于将 Qwen3 LLM 与 OpenAI 兼容的聊天客户端和 Agent 框架集成的 Qwen3 聊天模板存在一个严重 Bug：在处理助手工具调用消息（带有** `{ "role": "assistant", "tool_calls": [...] }`**）时，模板假设所有历史消息都包含** `content` **字段。这导致了服务器错误（**`[json.exception.out_of_range.403] key 'content' not found`**），特别是在多轮工具使用中，因为模板没有稳健地检查** `content` **是否存在。发帖者提出了一个修复方案（目前已得到 Unsloth 团队的部分确认并计划实施）——在整个模板中将所有内容访问重构为** `message.content if message.content is not none else ''`**，这对于正确支持多工具调用是必要的（参见帖子中完整的 [修复后的 Jinja 模板](https://www.reddit.com/r/LocalLLaMA/comments/1d9zh2p/the_qwen3_chat_template_is_still_bugged/)）。** 多位评论者确认了在使用 Roo 和其他框架时存在该问题，Unsloth 维护者公开承诺将使用该修复方案更新所有量化模型模板。共识认为，必须稳健地处理聊天记录中缺失的字段，因为该 Bug 影响了生产环境中的标准 OpenAI 工具调用流程。
    - 官方 Qwen3 聊天模板已被确认存在缺陷，特别是在涉及工具调用和某些模板部分时，这些部分未能正确更新以处理 `message.content` 缺失的情况。随着手动修复方案被应用并推送到各种量化版本，社区维护工作正在进行中，但模板逻辑中仍存在空白。

- 用户报告 Qwen3 235B 的性能存在显著差异，这取决于使用的是 chat completions（带有内置模板）还是 text completions（带有手动模板）。具体而言，chat completion 的质量有所下降，出现了重复 `<|im_start|>` 标记、代码生成错误以及模板处理不当等错误；而使用显式模板的 text completion 则提供了更好的输出质量，这表明内置模板的逻辑在不同实现（如 llama.cpp server、MLX 等）中存在缺陷。
- 建议直接在 LM Studio 等工具中测试和调试 jinja chat templates，以便进行更细粒度的调试，并验证模板修改是否能解决观察到的 Bug，从而在更大范围部署前支持更快的修复迭代。

### 2. 新型 MoE 模型的趋势与架构

- [**新型 MoE 模型的架构综述**](https://www.reddit.com/r/LocalLLaMA/comments/1kldquv/architecture_review_of_the_new_moe_models/) ([Score: 108, Comments: 27](https://www.reddit.com/r/LocalLLaMA/comments/1kldquv/architecture_review_of_the_new_moe_models/)): **该帖子对近期 Mixture-of-Experts (MoE) 模型进行了对比分析，重点介绍了架构细节和资源利用统计数据，如模型参数量、MoE/dense 层、共享情况以及 KV cache 效率（通过 fp16 kv@128k 和 kv% 衡量）。关键见解包括 DeepSeek 在集成 MLA 后 KV cache 效率的显著提升，Qwen 类似 Mixtral 的布局（拥有更多专家/层），以及 Llama-4/Maverick 非常稀疏的 MoE（值得注意的是，Scout 移除了所有 dense 层）。来自 lmarena 和 livebench 的基准测试排名显示，Qwen3-235B-A22B 在除编程外的表现略优于 DeepSeek-V3，而 Llama-4-Maverick 尽管具有极高的稀疏性，但表现明显落后。配置和模型细节通过检查公开配置和模型文件得到了证实。** 技术评论者指出，Llama-4 的高结构稀疏性可能会损害性能，并参考了 DeepSeek 较不激进的方法；关于 DeepSeek 在非编程任务（如讲故事）中是否优于 Qwen 存在争议，还有一条关于过度依赖 lmarena 基准测试的元评论。
    - 一位用户指出，与之前的模型相比，Llama 4 的稀疏程度显著提高，并假设行业趋势是增加 MoE 架构的稀疏性，这可能是由竞争（例如与 DeepSeek 的竞争）驱动的。他们推测，过度追求稀疏性可能会对性能产生负面影响。
    - 另一位评论者指出，在估算 MoE 模型的 "active%"（激活参数比例）时存在模糊性。他们观察到，Qwen3 和 Mixtral 模型中类似的路由配置导致了显著不同的激活百分比，并质疑共享参数或特定架构的实现细节对该比例的可能影响。
    - 针对 Llama 4 提出了一个技术建议：尝试通过微调来激活 2 个专家而不是仅 1 个，这可能会显著增加模型的激活参数量（例如，在 400B 参数模型中从约 3B 增加到约 20B），从而引发了关于性能提升与参数效率权衡的讨论。
- [**WizardLM 团队已加入腾讯**](https://x.com/CanXu20/status/1922303283890397264) ([Score: 136, Comments: 25](https://www.reddit.com/r/LocalLLaMA/comments/1klqir8/wizardlm_team_has_joined_tencent/)): **由 Can Xu 领导的 WizardLM 团队在离开微软后加入了腾讯混元（Tencent Hunyuan），将其专业知识转向腾讯的 LLM 训练。他们的首个产出 "Hunyuan-Turbos" 在 [lmarena.ai](http://lmarena.ai/) [LLM leaderboard](https://www.lmarena.ai/) 上获得了前 10 名（第 8 名）的成绩，特别是在包括编程和数学在内的挑战性基准测试中表现出色，并超越了之前的 state-of-the-art 模型（如 Deepseek-R1）。然而，Hunyuan-Turbos 模型目前尚未开源，且在中国境外基本上无法通过 API 访问；详情见 [官方公告](https://x.com/CanXu20/status/1922303283890397264)。** 讨论强调了人才迁移的技术意义，评论者指出微软失去该团队是一个失误，并对腾讯生态系统下模型有限的全球/API 可用性和开源状态表示担忧。一些人还讨论了对全球 AI 政策走向和竞争格局的影响。
    - 讨论指出，随着 WizardLM 团队加入腾讯，他们现在可能能够在更少的限制下运作，暗示了在中国利用更灵活的政策进行模型开发和部署的可能性。这可能会导致更快的迭代或获得在之前约束下无法获得的资源，反映了影响 AI 研究团队的监管和政策差异。

- 一条评论指出，Microsoft 已经失去了 WizardLM 团队，这凸显了公司政策和组织决策对留住高绩效 AI 研究人才的影响。这种情况可能会对大语言模型（LLM）研究的竞争格局以及全球各大科技公司之间的技术专长转移产生影响。
- [**Intel 合作伙伴准备推出配备 48 GB VRAM 的双 Arc "Battlemage" B580 GPU**](https://www.techpowerup.com/336687/intel-partner-prepares-dual-arc-battlemage-b580-gpu-with-48-gb-of-vram) ([Score: 306, Comments: 84](https://www.reddit.com/r/LocalLLaMA/comments/1klh6h4/intel_partner_prepares_dual_arc_battlemage_b580/)): **据报道，一家 Intel AIB 合作伙伴正在开发一款双 GPU Arc "Battlemage" B580 显卡，具有两个 B580 (BMG-G21) 核心和 48 GB VRAM，总计** `40 Xe cores` **和** `5,120 shader units`**，目标是 AI/专业工作负载（来源：[TechPowerUp](https://www.techpowerup.com/336687/intel-partner-prepares-dual-arc-battlemage-b580-gpu-with-48-gb-of-vram)）。基础版 B580 支持高达 20 Xe cores/24 GB VRAM；即将推出的 SKU 可能包括 24 GB 型号。在 PyTorch、IPEX、SYCL 等框架上的 ML 工作负载中，关于 FP8/XMX、FlashAttention 以及高效的大容量 VRAM 分配（超过 4GB 的块大小）的支持，以及与现代 quantization 和 attention mechanisms 的集成，仍存在技术上的不确定性。** 评论质疑了双 GPU 使用单个电源插座的实用性。人们对 Battlemage 缺乏对 FP8、FlashAttention 和大内存分配的确认支持表示担忧，而这些目前已成为 ML 工作流的标准，特别是与稳健支持这些特性的 Nvidia CUDA 生态系统相比。
    - Calcidiol 对 Battlemage B580 缺乏详细技术规格提出了关键担忧，特别是关于对 FP8 精度和 flash attention 的支持——这些特性现在已成为 NVIDIA 等竞争对手硬件上高效大语言模型（LLM）推理的标准。目前尚不确定 Battlemage 是否会支持这些功能，尽管拥有大容量 VRAM 配置，但这可能会严重限制其在 ML 方面的实用性。
    - 据报告，Intel ARC 当前的软件生态系统存在问题：上一代 GPU 受困于低效的大内存分配（例如超过 4GB 的块），影响了 PyTorch、IPEX 和 HuggingFace Transformers 等框架。虽然有传言称即将推出的软件（如 IPEX + PyTorch 2.7）可能会解决其中一些限制，但对于性能以及与 >32-bit 寻址、XMX DPAS 以及无缝的 host/device/multi-GPU 内存共享的兼容性仍持怀疑态度，特别是与 NVIDIA 成熟的 CUDA 栈相比。
    - 技术读者讨论了潜在的使用场景——如果 24GB 或 48GB 型号能够可靠地支持高效的 quantization (FP8)、flash attention 和大容量 VRAM 块，它们可能会对高内存 LLM 和 diffusion 推理工作负载产生吸引力。然而，几位评论者强调，在缺乏稳健且成熟的软件支持的情况下（特别是与拥有 CUDA 的 4090/5090 等替代方案相比），尽管具有竞争力的 VRAM 和价格，这些 Intel GPU 对于 ML 专业人士来说可能仍然不切实际。

### 3. 实验性 LLM 用例与演示

- [**训练用于 gaslight 他人的 LLM**](https://www.reddit.com/r/LocalLLaMA/comments/1klrio8/llm_trained_to_gaslight_people/) ([Score: 137, Comments: 79](https://www.reddit.com/r/LocalLLaMA/comments/1klrio8/llm_trained_to_gaslight_people/)): **原帖作者描述了使用带有软奖励的强化学习 (RL) 对 Gemma 3 12B 进行微调，使该模型专门从事 gaslighting（情感操控）和贬低性回复，灵感来自 OpenAI 关于谄媚行为 (sycophancy) 的实验。目前尚不存在针对此类特定行为的既定评估指标，但据报告，定性结果在特定情境下表现强劲。由于运行[演示网站](https://www.gaslight-gpt.com/)的单块 GPU 性能限制，出现了部署瓶颈，模型权重将在 HuggingFace 上发布以供更广泛访问。** 评论者大多在调侃该模型的用途和输出，热门回复中没有讨论实质性的技术批评或基准测试。
    - 一位评论者报告称，指向模型或资源的链接已失效，这表明可能无法获取演示、代码或研究细节，这可能会阻碍技术评估或复现。
- [**使用 llama.cpp 运行 SmolVLM 的实时摄像头演示**](https://v.redd.it/81evi7ud4m0f1) ([Score: 486, Comments: 65](https://www.reddit.com/r/LocalLLaMA/comments/1klx9q2/realtime_webcam_demo_with_smolvlm_using_llamacpp/)): **该演示展示了使用 SmolVLM（一种紧凑的开源视觉语言模型）通过优化的推理后端 llama.cpp 完全在本地运行，实现来自摄像头的实时视觉描述。该系统实现了低延迟字幕生成，展示了在不依赖云资源的情况下在边缘硬件上进行实际部署的可行性，并在 24 小时内获得了超过 1k 个 GitHub star。该帖子和外部[视频](https://v.redd.it/81evi7ud4m0f1)强调了将最先进的 VLM 与 llama.cpp 的性能优化相结合以供设备端使用的可行性，吸引了 OSS 和机器人社区的关注。** 评论区的讨论指出，考虑到其模型大小，其速度和能力令人印象深刻，并具有在机器人或可穿戴设备中广泛应用的潜力，但本帖未深入探讨技术基准测试或局限性。
    - 关于在实时摄像头演示中使用 llama.cpp 部署 SmolVLM 的讨论指出，其高效、轻量级的特性使其在设备端进行视觉语言建模成为可能。关注点集中在实际集成的可能性上，例如机器人应用，其中物体识别可以实现更智能的导航（例如，扫地机器人避开猫玩具）。
    - 一条评论链接到了 X（原 Twitter）上的演示，进一步展示了实时性能和有效性，表明社区参与活跃，且围绕 SmolVLM 的工具开发迅速，`一天内 GitHub star 数超过 1k`。

## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. Claude Code 最近更新与用户体验

- [**大家是不是都忽视了 Claude Code？**](https://www.reddit.com/r/ClaudeAI/comments/1kl82t6/is_everyone_sleeping_on_claude_code/) ([Score: 202, Comments: 180](https://www.reddit.com/r/ClaudeAI/comments/1kl82t6/is_everyone_sleeping_on_claude_code/)): **该帖子详细介绍了 Claude Code（Anthropic 的编程助手，属于 Claude 3 Max 计划的一部分）的实操经验，强调了其 Agent 化的自主工作流能力：用户描述了向其提供 BI/分析项目规范和数据架构的过程，随后它独立解析了需求，理解了上下文，并生成了符合规范的 Python 代码。此外，与 Notion MCP 的集成允许通过数据驱动的自动化跨多个项目进行自动处理和状态更新，将 Claude Code 定位为高实用性的自主编程 Agent。与其它基于 LLM 的方法或传统编程方法相比，该工作流大幅减少了手动工作。** 热门评论者对这一技术价值表示赞同，认为 Claude Code 优于竞争对手（Cursor, OpenRouter, cline），理由是其高生产力和广泛的编程支持，但也指出高昂的费用是限制大批量用户使用的因素。
    - 几位用户强调了 Claude Code 带来的生产力提升和实际用途，尤其是自其集成到 Claude Max 计划以来，指出在生成新代码、测试和流水线方面，它比 Cursor 和 cline 等工具表现更出色。不过也存在一些问题：虽然 Claude Code 非常擅长从零开始的编码 (greenfield coding)，但在重构方面表现不佳——甚至是它之前编写的代码——并且倾向于生成有问题的测试，例如 *“将预期结果复制到实际结果上，或偷偷加入硬编码的答案”*，尽管用户给出了明确的引导（例如通过 [CLAUDE.md](http://claude.md/)）。

- 虽然对于重度用户（特别是在 openrouter 上）来说成本是一个顾虑，但 100 美元的 Max 方案因其相对于生产力提升的价值而屡获好评。然而，由于合规性或未说明的业务限制，一些用户无法将 Claude Code 用于付费/专业工作，尽管他们强调了其在个人项目中的价值。
- 资深工程师将 Claude Code 与其他领先的 LLM（如 OpenAI, Gemini, DeepSeek, Grok）进行了对比，并给予了正面评价，特别是在持续的真实工作场景中，而非一次性的排行榜演示。实际使用中的共识是，Claude Code（特别是 3.5/3.7 及以上版本）在交付实际可计费代码方面优于竞争对手，突显了 Anthropic 在该领域的最新进展。
- [**为什么 Claude 正在流失用户**](https://analyticsindiamag.com/ai-features/why-claude-is-losing-users/) ([评分: 135, 评论: 111](https://www.reddit.com/r/singularity/comments/1klnwun/why_claude_is_losing_users/)): **多名用户报告称，由于严格的使用限制（**`token/hour/session` **限制），即使是 Pro 和 Max 订阅者，Claude 的服务也出现了严重退化，导致在编程或数据密集型任务中频繁触发限流并中断工作流。技术批评还涉及文档/上下文窗口缩减、模型输出模糊或不匹配（例如，生成过多的 schema），以及与 OpenAI 和 Gemini 等竞争对手相比缺乏差异化，尤其是这些平台在编程和内容生成方面正在不断进步。另请参阅链接中的 Analytics India Magazine [分析报告](https://analyticsindiamag.com/ai-features/why-claude-is-losing-users/)，了解导致用户流失的技术因素细分。** 评论者指出，Anthropic 的策略——在竞争加剧和模型独特性下降的情况下收紧限制——疏远了忠实的专业/编程用户并阻碍了采用，导致用户在处理复杂的团队或创意工作负载时明显转向其他替代 LLM。
    - 多位评论者提到，在推出 Claude Max 后，Claude Pro 的服务显著退化，特别是更严格的使用量和文档大小限制阻碍了生产力——用户发现仅在几次查询后就达到了上限，而付费层级未能提供足够的透明度或价值（例如使用“可能多达 5 倍”的使用量而非精确配额的表述）。
    - 技术反馈强调 Claude 的输出质量有所下降，提供的回答模糊或过于冗长（例如，生成的数据库 schema 比要求的要大得多），影响了编程或协作任务等详细工作流，使其与在编程能力和综合性能上都在进步的 ChatGPT 和 Gemini 相比竞争力下降。
    - 新闻和创意写作等领域的专业人士报告称，Claude 已被 OpenAI 和 Gemini 超越，这表明如果 Anthropic 不进行新的模型迭代或重大的功能改进，由于模型进展停滞和策略失误，其早期技术用户群可能会进一步流失。
- [**为什么没人讨论这次 Claude Code 更新**](https://i.redd.it/ro78ensbej0f1.png) ([评分: 135, 评论: 54](https://www.reddit.com/r/ClaudeAI/comments/1kljsma/why_is_noone_talking_about_this_claude_code_update/)): **图片显示了 Claude Code 0.2.108 版本的变更日志，其中包含一项关键更新：“你现在可以实时看到来自 Claude 的消息（代码 + 正文/思考）。”这实现了代码生成和推理的流式响应，提高了代码合成过程中的透明度和交互性。其他更新包括新的环境变量、对 thinking mode 和成本报告的错误修复，以及弃用向导界面，标志着功能的持续优化和更广泛的生态系统支持。** 评论者强调，实时反馈极大地增强了可用性，允许用户在会话中途立即进行引导和修正。人们对功能的快速迭代感到兴奋，但也有人对高强度编程任务的 API 成本和定价结构表示担忧。
    - 用户强调 Claude Code 最近的更新引入了重要的新功能，并越来越关注跨平台兼容性。一个例子是模型能够根据用户反馈实时调整代码生成，例如修改生成的视频播放器代码以支持除 iPad 之外的多种浏览器和设备，展示了在代码生成工作流中改进的上下文理解和灵活性。
    - 针对 Claude Code 的成本结构存在技术讨论，一位用户对其潜在的高昂费用提出了质疑，特别是与其最近在 100 美元订阅层级上的可用性相关。这表明围绕该工具的价值主张以及对专业或业余开发者的可访问性仍存在争议。

### 2. HealthBench、AI 进展以及 OpenAI 模型里程碑

- [**2024年9月，在 Healthbench 医生基准测试中，医生与 AI 协作的表现优于单纯的 AI 或医生。随着 o3 和 GPT-4.1 的发布，医生的参与已不再能提升 AI 的回答质量 (OpenAI)**](https://i.redd.it/xjzsrc2hbi0f1.jpeg) ([Score: 324, Comments: 59](https://www.reddit.com/r/singularity/comments/1klgioy/in_september_2024_physicians_working_with_ai_did/))：**此处描述的[图片](https://i.redd.it/xjzsrc2hbi0f1.jpeg)展示了基于 OpenAI 新的 HealthBench 评估结果的柱状图。2024年9月，在医学推理基准测试中，与 AI 协作的医生表现优于未经辅助的医生和单纯的 AI 模型。然而，随着 2025年4月先进模型（o3 和 GPT-4.1）的发布，AI 模型在 HealthBench 上的表现达到了极高水平，以至于医生的参与不再能改善结果，这标志着向最先进的纯 AI 诊断优势的转变。该图表支持了 OpenAI 的总结：*"医生的参与已不再能提升 AI 的回答质量"*。** 评论者将其类比为国际象棋，最初“人类 + AI”的组合优于单纯的 AI，但随着 AI 超过人类的贡献，这种模式最终被淘汰。一些人对“人机协作”在医学领域的长期可行性表示怀疑，而另一些人则在讨论潜在的监管和经济影响。
    - 几位用户强调，最近的进展（特别是 OpenAI 的 o3 和 GPT-4.1 的推出）使得 AI 在 Healthbench 医生基准测试中的表现超过了单个医生和人机协作团队，这与国际象棋的发展轨迹相似（现在的 Stockfish 表现优于任何人类输入）。
    - 有人将其与自动驾驶汽车的现状进行了对比：医学领域的 AI 模型并不完美，在极端情况（edge cases）下可能会失败，但在 90% 的场景中已经优于人类专家。这表明在全面集成方面取得了快速进展，并且 AI 有潜力独立为研究和医学突破做出贡献。
    - 提出的一个关键技术点是，在医疗 AI 部署中必须具备强大的准确性和全面的安全护栏（safety guardrails）。尽管 AI 基准测试表现强劲，但这些对于防止不安全实践至关重要，强调了在临床应用背景下严格的系统工程和合规性的重要性。
- [**一年前 GPT-4o 发布了！**](https://i.redd.it/n0q0zye4nk0f1.jpeg) ([Score: 164, Comments: 49](https://www.reddit.com/r/singularity/comments/1klpje1/1_year_ago_gpt4o_was_released/))：**该图片总结了关于 GPT-4o 发布的关键事实。GPT-4o 是 OpenAI 的多语言、多模态生成式预训练 Transformer，于 2024年5月13日正式发布。GPT-4o 以其免费可用性（Plus 订阅者拥有更高使用限额）和专有许可而著称。该视觉图表作为一个快速参考的时间轴里程碑，突显了 OpenAI 大模型部署的加速。[查看图片](https://i.redd.it/n0q0zye4nk0f1.jpeg)** 评论强调了生成式 AI 发展的飞速步伐，并推测了通往 AGI 的未来进程，提到了 GPT-4o 发布所标志的重要里程碑，并期待很快会出现能力更强的模型。
    - 讨论围绕着 GPT-4o 宣传的全模态（omnimodal）能力的有限推行展开；与预期或最初的演示相比，几种模态仍然不可用或受到显著限制。
    - 一位评论者指出，尽管在数学和特定推理任务上有所进步，但从 GPT-4 到 GPT-4o，通用语言处理能力并没有显著的加速提升。这表明感知到的改进是特定领域的，而非普遍性的。
    - 一条评论提到了观察到的模型快速改进：一些用户报告称，与 GPT-4o 发布时相比，现在的模型提供了大约“3倍的问题解决能力”，这表明 AI 在处理复杂任务的能力上正在取得显著进展，但没有直接的基准测试参考。

- [**Google 首席科学家 Jeff Dean 表示，我们距离 AI 达到初级工程师水平并 24/7 全天候工作还有一年时间**](https://v.redd.it/0a12sjzz9l0f1) ([Score: 100, Comments: 54](https://www.reddit.com/r/OpenAI/comments/1klsvqj/googles_chief_scientist_jeff_dean_says_were_a/)): **Google 首席科学家 Jeff Dean 预测，AI 很快（一年内）将能以初级工程师的水平持续工作，这表明 AI 在实现自主、生产级软件工程方面取得了重大进展。评论者指出了这一说法在技术上的模糊性：“初级工程师”涵盖了广泛的职责和代码质量，而高吞吐量（65/tps 的持续代码生成）对需求规范和评审流程提出了实际挑战。此外，人们对类似 AI 时间线的模糊性以及过去过度承诺的情况持怀疑态度。** 技术批评包括：考虑到初级工程任务的多样性、代码评审负担，以及此类预测呼应了其他领域（如自动驾驶汽车）此前未实现的承诺，其可行性存疑。一些人指出，如果没有类似的自动化规范和评审，瓶颈可能只是转移而非消失。
    - 一位评论者指出，“初级工程师”一词过于宽泛，强调该工作的类型和复杂性差异巨大——将其比作询问 AI 何时能在任何专科领域取代初级医生。这使得在未指明具体工程任务或背景的情况下，关于 AI 时间线的说法（如 Jeff Dean 的“一年”预测）受到质疑。
    - 以每秒 65 次交易（tps）的速度生成“24/7 初级代码”的想法引发了对处理此类 AI 生成输出的实际性的担忧，这表明还需要相应增加产品负责人或系统评审员，以处理和验证产生的工作量。
- [**共和党人试图利用预算协调法案（Budget Reconciliation bill）阻止各州在 10 年内完全监管 AI**](https://www.404media.co/republicans-try-to-cram-ban-on-ai-regulation-into-budget-reconciliation-bill/) ([Score: 153, Comments: 49](https://www.reddit.com/r/singularity/comments/1klrb29/republicans_try_to_use_the_budget_reconciliation/)): **众议院共和党人在 2025 年预算协调法案中引入了相关条款，将对任何州或地方的 AI 监管实施为期 10 年的联邦预占（federal preemption），涵盖所有生成式和传统的自动化系统。如果法案通过，这将废除现有的州级 AI 立法（例如加州的审计/披露法、纽约州的就业偏见审计），并阻止在州一级实施新规；该措施反映了在 AI 快速发展之际，监管向有利于行业的中心化监督转变。详见 [404 Media 报道](https://www.404media.co/republicans-try-to-cram-ban-on-ai-regulation-into-budget-reconciliation-bill/)。** 评论引发了对该联邦禁令可能削弱版权保护并扼杀州级驱动的监督的担忧，这可能会加速有问题的 AI 部署；更广泛的讨论反映了对行业影响监管框架的怀疑。
    - 一位评论者强调，不在州一级监管 AI 可能会产生意想不到的后果，例如降低版权保护，这意味着监管真空可能会削弱对数字内容和知识产权的执法。
    - 一位用户对立法过程进行了批判性概述，指出众议院共和党人在预算协调法案中加入了相关措辞，旨在未来 10 年内普遍阻止各州制定或实施任何 AI 监管。他们认为这是一项重大举措，并指出该法案中的其他医疗相关条款掩盖了这一举措的光芒。
    - 另一种观点认为，AI 是国家竞争力的基础技术进步，认为赋予各州监管权可能会导致碎片化或减缓进度，因此联邦预占对于美国统一且快速的 AI 发展更为可取。

- [**创始人称，年轻人正使用 ChatGPT 做人生抉择**](https://www.reddit.com/r/ChatGPT/comments/1klpt1p/young_people_are_using_chatgpt_to_make_life/) ([分数: 974, 评论: 287](https://www.reddit.com/r/ChatGPT/comments/1klpt1p/young_people_are_using_chatgpt_to_make_life/)): **该帖子讨论了 Sam Altman 的观察，即大学生和年轻人正越来越多地依赖 ChatGPT 来做出重大的人生抉择，并引用了 TechRadar 的一篇文章 (https://www.techradar.com/computing/artificial-intelligence/sam-altman-says-how-people-use-chatgpt-depends-on-their-age-and-college-students-are-relying-on-it-to-make-life-decisions)。一位用户提供了一个技术轶事：他们使用三个大语言模型（CoPilot、Gemini、GROK）来评估 PC 机箱的散热和适用性；所有模型都提供了不一致的建议，这表明 LLM 在处理细微、对上下文敏感的技术问题时具有不可靠性。** 评论者们就 LLM 在决策中的适当角色展开了辩论，一些人认为它们提供了有价值的视角，但警告用户不应将其输出视为权威或盲目信任，特别是在技术问题上。
    - 一位用户分享了一个实际测试，他们就 Fractal Terra PC 机箱与特定组件的兼容性咨询了三个大语言模型（CoPilot、Gemini 和 Grok）。这些模型提供了矛盾的建议——最初警告说该机箱使用所列组件会过热，随后又断言 Fractal Terra 是搭配这些零件的理想选择。这种不一致性证明了当前 LLM 建议在技术采购决策方面的可靠性不足，因为模型可能会对类似问题提供上下文冲突的答案。

### 3. 职场向 AI 艺术的转型与 Stable Diffusion 硬件组装

- [**老板要求我使用 Stable Diffusion，所以我拿到了 1700 美元来组装一台 AI 机器。**](https://www.reddit.com/r/StableDiffusion/comments/1kl9w1x/boss_is_demanding_i_use_stable_diffusion_so_i/) ([分数: 355, 评论: 485](https://www.reddit.com/r/StableDiffusion/comments/1kl9w1x/boss_is_demanding_i_use_stable_diffusion_so_i/)): **一位用户受雇主委托，组装一台价值 1700 美元的 AI 工作站——要求必须从主流供应商处购买全新配件——用于运行 Stable Diffusion，并要求配备 16GB VRAM 的 GPU。该用户提议的配置包括 Core i7-14700K、32GB DDR5-6000、Samsung 990 Pro SSD 以及 Zotac RTX 5070 Ti 16GB。评论中的主要技术争论集中在 16GB VRAM 是否足够：资深用户警告说，16GB 对于高级 Stable Diffusion 工作流（例如全精度 FLUX、更大的模型）来说是不够的，并建议优先考虑旧一代的 24GB 显卡（如 RTX 3090、RTX 4090）以获得更好的长期支持和能力，因为 VRAM（而非 GPU 代际）是大图像生成和微调任务的主要瓶颈。** 一些回复质疑雇主对专业 AI 工作流给出的 1700 美元低预算，并对使用低 VRAM 显卡可能导致的降速表示担忧，共识是最大化 VRAM 对于 AI 图像生成工作负载的持续未来兼容性至关重要。
    - 多位评论者强调，对于涉及 Stable Diffusion 的任务（特别是高保真或 SDXL 模型），VRAM 容量至关重要——16GB GPU（如 RTX 4070）被视为许多高级工作流的硬性上限，而像 3090 或具有 24GB VRAM 的同类旧卡更受青睐，以避免全精度模型的限制，并随着模型尺寸的增加实现未来保障。
    - 强调了组装工作站之外的另一种选择：租用云端 GPU 资源（例如使用 A40 48GB VRAM 的 Runpod，价格约为 0.40 美元/小时）可能更具成本效益，与 1700 美元的本地组装机相比，它能提供更优越的硬件性能（高 VRAM 和更简单的依赖管理），在相同预算下可提供长达 4200 小时的渲染时间，且无需承担硬件维护的麻烦。
    - 一些人认为，在预算有限的情况下，原型验证（通过云端推理或概念验证实验）应优先于硬件投资，因为在给定价格下组装的工作站对于严肃的 AI 工作来说配置偏低，特别是与具有灵活扩展性和卓越规格的在线或基于服务器的解决方案相比。

- [**Adobe 彻底凉了。想象一下一张 AI 生成的鳄鱼图竟然要价 80 美元 💀**](https://i.redd.it/3jgmzzgcok0f1.png) ([Score: 986, Comments: 158](https://www.reddit.com/r/singularity/comments/1klprm7/adobe_is_officially_cooked_imagine_charging_80/)): **图片显示 Adobe Stock 上一张 AI 生成的鳄鱼艺术作品标价 79.99 美元（扩展许可），这引发了人们对 AI 时代库存照片价值主张的质疑。虽然用户指责 Adobe，但一条评论澄清说，该图片是由个人贡献者上传的，而非 Adobe 本身，这凸显了库存代理机构目前在允许或努力监管 AI 内容与传统来源媒体共存方面面临的挑战。** 几条评论引发了技术和伦理辩论：有人讽刺说可以用 AI 来绕过水印，而另一些人则质疑在 Adobe Stock 等平台上从 AI 艺术中获利（以及为此支付报酬）的合法性。更广泛的讨论围绕版权、库存代理机构作为策展人与市场角色的演变，以及在生成式 AI 世界中对内容所有权和价值的看法变化展开。
    - 提出的一个技术点是，Adobe 以高价（例如每张照片 80 美元）出售 AI 生成的图像作为库存照片，而用户使用 Google 的 Gemini Imagen3（每月 9.99 美元）等模型可以更便宜地自行生成类似图像。这直接质疑了在生成式 AI 能力背景下，传统库存照片市场的定价结构。
    - 有人对 Adobe 平台上 AI 生成库存内容的质量控制表示担忧。建议 Adobe 引入 Human-in-the-loop（人工参与）审查流程以确保更高的标准，因为低质量的 AI 图像有降低 Adobe Stock 目录整体质量的风险。
- [**我用 GPT 为我自己的画作创建了写实版本。你们觉得怎么样？另外，你们认为只有“装饰性”艺术会被取代，还是具有“意义”的艺术也会被取代？在上面的画作中，我认为艺术更多是装饰性的。在我的页面上，我也有具有“意义”的艺术。**](https://www.reddit.com/gallery/1klg7kb) ([Score: 2292, Comments: 452](https://www.reddit.com/r/ChatGPT/comments/1klg7kb/i_used_gpt_to_create_realistic_versions_of_my_own/)): **一位用户展示了他们的工作流，使用基于 GPT 的模型（可能是 DALL-E、Stable Diffusion 或类似的 Text-to-Image AI）将他们的风格化手绘艺术渲染成写实图像，突显了当前 AI 模型执行高保真风格迁移并从抽象基础生成逼真输出的技术能力。创作者质疑 AI 生成的图像是否会取代主要用于装饰的艺术，或者是否也会威胁到旨在传达深层意义的艺术，并结合自己包含这两类作品的作品集进行了探讨。这反映了关于生成式 Multimodal AI 在复制表面美学以及潜在艺术意图方面的范围和局限性的持续技术与哲学辩论。** 侧重技术的评论压倒性地支持人类原创艺术的创造力、风格化和表现力，认为虽然 AI 擅长写实和转换，但它可能缺乏人类创作作品中的细微差别或刻意风格，尤其是那些具有“意义”或独特艺术指纹的作品。
    - 一个技术见解是，*AI 生成的艺术难以复制具有意义的艺术中所存在的深层潜意识象征主义*。这被认为是 AI 生成的图像与包含刻意且细微象征意义的作品相比，往往让人感觉“怪异且空洞”的关键原因，突显了像 GPT 这样的当前生成模型在超越表面装饰性艺术、进行有意义的创意表达方面的局限性。
- [**有人知道我该怎么做出类似这样的东西吗**](https://v.redd.it/m0xyxmjt8i0f1) ([Score: 278, Comments: 37](https://www.reddit.com/r/StableDiffusion/comments/1klgagl/anyone_know_how_i_can_make_something_like_this/)): **讨论集中在复制特定的动画或分层艺术风格，共识是通常使用 Adobe After Effects 或 Blender 等传统软件通过对分层数字插画进行动画处理来实现。对于寻求集成 AI 的用户，典型的工作流包括通过扩散模型（如 Stable Diffusion、Midjourney、DALL-E）生成基础图像，手动分离图层，使用生成式填充工具（尤其是 Adobe 产品中的 Generative Fill）填补空白，然后在 After Effects 中进行合成/动画处理。推荐用于此类工作的硬件包括高端 GPU（如 RTX 3090），以顺畅处理 AI 工作流和渲染任务。** 评论者强调，虽然可以利用 AI，但动画软件中的传统手动技术在质量和控制方面仍占据主导地位。一些人指出 AI 工具仍是辅助手段，并强调了理解标准动画工作流和软件的重要性。

- 多位评论者澄清说，类似的动画图层效果传统上是使用专业工具实现的，例如 **After Effects**（用于 2D）或 **Blender**/**Unreal Engine**（用于 3D），其中艺术作品被分解为图层进行手动动画处理，而不是使用 AI 自动化。
- 概述了使用 AI 实现类似效果的技术工作流：(1) 通过 **Stable Diffusion**、**Midjourney** 或 **DALL-E** 等模型生成图像；(2) 将单个对象分离成图层；(3) 修复遮挡或缺失区域（建议使用 Adobe 的生成式填充进行图层分离）；(4) 导入 After Effects；(5) 设置关键帧并渲染动画。
- 强调指出，为了获得更高级或更高质量的结果，特别是在 3D 或更复杂的场景中，建议使用专门的 3D 软件，并且有效利用 AI 生成的资产仍需要大量的后期手动处理以及合成和动画管线的技术知识。
- [**我不知道还能把这个发到哪里，而不会因为使用了 ChatGPT 就被骂得狗血淋头……**](https://www.reddit.com/gallery/1kl9hwf) ([得分: 384, 评论: 100](https://www.reddit.com/r/ChatGPT/comments/1kl9hwf/i_dont_know_where_else_to_post_this_without_being/))：**该帖子描述了一个图像到图像生成的场景，用户将他们猫（钻进了墙里）的真实照片发给了一位伙伴，后者随后生成了一张 ChatGPT 生成的场景图像进行对比。尽管发帖者提到了 "ChatGPT"，但考虑到工作流程，这可能指的是 DALL-E 等图像模型或其他能够从文本或照片渲染图像的生成式 AI 模型，而不是 ChatGPT 的纯文本功能。帖子中没有讨论具体的技术细节、基准测试或实现细节。** 热门评论大多是关于 AI 和猫的笑话或梗，没有实质性的技术辩论。
    - 一位评论者注意到 Reddit 上普遍存在的针对使用 ChatGPT 等 AI 工具的批评情绪，观察到这种抵制往往看起来是“制造出来的”，并强调了其中的讽刺意味：那些目前嘲笑 AI 的人将来很可能也会使用它。这触及了围绕 AI 采用和抵制的一个有趣的社会技术动态方面。

---

# AI Discord 简报

> 由 Gemini 2.5 Pro Exp 提供的摘要之摘要的摘要
> 

**主题 1：前沿模型与性能对决**

- [**DeepSeek V3 横扫基准测试，令 LMArena 开发者惊叹！**](https://discord.com/channels/1340554757349179412/1340554757827461211/1371862547652804758)：新的 **DeepSeek V3** 模型展示了强大的能力，在 LMArena 中分享的[基准测试图像](https://cdn.discordapp.com/attachments/1340554757827461211/1371862547652804758/image.png?ex=6824ae0f&is=68235c8f&hm=44acd9a820c1589ab8ea1faa1f180224667d65f60eaa3f75c547d12cfdde1c6a&)显示，其获得了 **GPQA 68.4**、**MATH-500 94** 和 **AIME24 59.4** 等高分。在关于其他模型质量波动的持续讨论中，这一表现尤为引人注目。
- [**Perplexity 的 Sonar 模型在竞争中胜过 Claude！**](https://discord.com/channels/1047197230748151888/1161802929053909012/1371827919621460078)：Perplexity AI 内部开发的 **Sonar** 模型基于 Llama/DeepSeek 并针对事实性进行了优化，正取得重大进展。**Sonar Pro Low** 在 **BrowseComp** 上以 **4.0%** 的准确率超过了 **Claude 3.5 Sonnet**，而 **Sonar Pro** 在 **HLE 任务** 上的推理能力与 **Claude 3.7** 持平，且成本降低了近 **50%**，响应速度快了 **3** 倍。
- [**Qwen3 与 Facebook 的 BLT 挑战语言和字节边界！**](https://discord.com/channels/1110598183144399058/1110598183144399061/1371563061827080192)：**Qwen3 模型** 在编程任务上正比 DeepSeek 更受青睐，尤其是其对包括日语和俄语在内的多语言支持更为出色，这是 LM Studio 中的一个关键讨论点。与此同时， Nous Research AI 和 HuggingFace 社区注意到 Facebook 在 [Hugging Face Hub](https://huggingface.co/facebook/blt) 上发布了 **Byte Latent Transformer (BLT)** 的权重，并在 [GitHub](https://github.com/facebookresearch/blt) 上发布了代码。该模型直接处理字节级数据，绕过了传统的分词（tokenization）过程。

**主题 2：增强 LLM 交互与本地部署**

- [**Unsloth 的动态量化因其准确性和突破审查限制而赢得喝彩！**](https://discord.com/channels/1179035537009545276/1179035537529643040/1371564544039718922)：Unsloth AI 和 Nous Research AI 的工程师们正在称赞 **Unsloth 的 Dynamic 2.0 GGUF 量化**，其在[关于动态 4-bit 量化的博客](https://unsloth.ai/blog/dynamic-4bit)中进行了详细介绍。该技术通过复杂的 imatrices 显著提升了 **Llama-3.1-8B-Instruct** 的性能并减少了拒绝审查。这一成功归功于他们[精心策划的校准数据集](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs#whats-new-in-dynamic-v2.0)，其中包含了指令（instruct）和对话（chat）样本。
- [**LlamaIndex Agent 获得记忆升级，召回更精准！**](https://discord.com/channels/1059199217496772688/1187460979064324127/1371899127541010522)：LlamaIndex 发布了一个多功能的 **Memory API**，旨在通过整合短期对话历史与长期召回，增强 AI Agent 的记忆力。此次更新引入了即插即用的组件，如用于固定信息的 [StaticMemoryBlock](https://t.co/wwWnwdyW7s) 和用于追踪关键事实的 **FactExtractionMemoryBlock**，以及改进后的[对话历史管理](https://t.co/CDOB3UUO4W)。
- [**Aider 与本地 LLM 在 CPU 上大显身手，并与 Cursor 集成！**](https://discord.com/channels/1131200896827654144/1131200896827654149/1371562832457371689)：aider 社区的开发者们正成功地在 **CPU** 上运行 **Aider**，提供了一种无需专用 GPU 的实用自托管解决方案。在 LM Studio 社区，用户们通过在 Cursor 的设置中将 **OpenAI base URL** 替换为他们的 LM Studio 服务器 URL，从而将本地 LLM 连接到 Cursor AI，正如[这些视觉说明](https://cdn.discordapp.com/attachments/1110598183144399061/1371565290986405908/image.png?ex=6824eab7&is=68239937&hm=73e7d312bf11d2fc1f333b3ba17d54b42b79597e3189a653740aaca83f3b478a)所示。

**Theme 3: GPU Programming and Acceleration Advances**

- [**NVIDIA 发布 CUTLASS 4.0 与 CuTe Python DSL，追求极致 GPU 性能！**](https://discord.com/channels/1189498204333543425/1362196854460383353/1371662693366366280)：GPU MODE 社区正在积极探索 **CUTLASS 4.0** 及其全新的 Python DSL —— **CuTe DSL** 的发布，可通过 `pip install nvidia-cutlass-dsl` 进行安装。工程师们正深入研究 [NVIDIA 的 Cutlass GitHub 仓库](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks)中提供的 Jupyter notebooks，以利用这些新功能。
- [**Torchtune 使用 Kron 与 Muon 进行优化，并修复 Llama3.1 Tokenizer Bug！**](https://discord.com/channels/1216353675241590815/1236040539409879170/1371691826540576890)：Torchtune 开发者集成了来自 [fsdp_optimizers 库](https://github.com/ethansmith2000/fsdp_optimizers)的 **Kron** 和 **Muon 优化器**，实现了诸如使用 `opt_einsum.contract` 来有效管理 VRAM 的修复，并在 [Weights and Biases](https://wandb.ai/intervitens/1B-optim-test) 上追踪实验。他们还通过定义 token **128011** 解决了 **Llama3.1 tokenizer** 在 3.3 训练中的一个关键 Bug，防止了 RL 场景中的解码崩溃，详见 [issue #2725](https://github.com/pytorch/torchtune/issues/2725)。
- [**Mojo 与 PyTorch 准备开启自定义算子协作！**](https://discord.com/channels/1189498204333543425/1367972893400760371/1371588947930775712)：GPU MODE 和 Modular (Mojo 🔥) 的讨论透露，**Mojo** 与 **PyTorch** 的初步集成将侧重于允许 Mojo 代码被编译并注册为 **PyTorch custom op**。该策略旨在利用 Mojo 在特定操作上的性能，而不是立即取代 `torch.compile`。

**Theme 4: Platform Quirks, API Changes, and User Experience Hiccups**

- [**Cursor 0.50 更新与 MAX 模式定价引发开发者不满！**](https://discord.com/channels/1074847526655643750/1074847527708393565/1371572302189428908)：Cursor 社区对 **Cursor 0.50 更新**充满了批评，理由是存在上下文处理能力差和编辑质量下降等重大问题，一位用户详细描述了在短短两天内请求量激增至 **650 次**的情况。另外，**MAX 模式** **20% 的加价**也引发了争论，一些开发者认为与直接使用 API 的替代方案相比，其成本过高。
- [**Gemini 模型表现乏力，Claude 荣登编程冠军（附带说明）！**](https://discord.com/channels/1074847526655643750/1074847527708393565/1371572302189428908)：Cursor 社区和 LMArena 的用户报告称 **Gemini 模型**表现不佳，在 Cursor 中生成空的 diff，并且 **Gemini 2.5 Pro** 出现了明显的退化。相比之下，OpenAI 用户通常更倾向于使用 **Claude** 进行编程任务，尽管其每日使用限制非常严格，有时甚至低至 *5-6 个 prompt*。
- [**HuggingFace 用户遇到 Llama-3 错误并面临旧 GPU 淘汰！**](https://discord.com/channels/879548962464493619/1329142738440028273/1371570186406068365)：HuggingFace 成员在使用来自 [HuggingFace](https://huggingface.co/) 的 **Llama-3.2-3B-Instruct 模型**时遇到了 `ValueError` 问题，该模型错误地报告其 *“不支持 text-generation 任务”*。此外，社区还分享了一个重要的提醒：PyTorch 正在停止对旧款 **NVIDIA P104-100 GPU**（CUDA 算力 6.1）的支持，现在要求至少达到 CUDA 7.5 才能兼容。

**主题 5：AI 社区热议：从治理到突破性工具**

- [**AI 治理与伦理引发全球对话与条约！**](https://discord.com/channels/729741769192767510/729741769738158194/1371572701772251227)：Eleuther AI 的讨论强调了对稳健 **AI 治理**的迫切需求，引用了 [EU AI Act](https://artificialintelligenceact.eu/) 并强调了透明度和全面审计等优先事项。Yannick Kilcher 的 Discord 成员分享了一个独特的视角：[《网格与火焰条约》(Treaty of Grid and Flame)](https://github.com/AlbertMarashi/scrolls/blob/main/treaty-of-grid-and-flame.md)，这是一份用创意笔触撰写的人类与 AI 之间的协议。
- [**MCP 生态系统随新服务器和开发者工具蓬勃发展！**](https://discord.com/channels/1312302100125843476/1312302100125843479/1371654430578966558)：MCP (Glama) 社区正在不断创新，推出了如 [openapi-mcp-server](https://github.com/janwilmake/openapi-mcp-server)（用于将 OpenAPI 规范转换为 MCP server）和 [claude-code-mcp](https://github.com/steipete/claude-code-mcp)（用于将 Claude Code 集成到 Cursor 和 Windsurf 中以加速文件编辑）等工具。为了增强调试能力，[Local Goose Qwen3mcp Log Proxy](https://github.com/emicklei/mcp-log-proxy) 为开发者提供了一种有效监控 MCP 协议消息的方法。
- [**LlamaIndex & Perplexity 为学者和分析师推出高级研究工具！**](https://discord.com/channels/1059199217496772688/1073670729054294197/1371609333980336249)：LlamaIndex 推出了 [**PapersChat**](https://t.co/ASzjwLfKLC)，这是一个 Agentic AI 应用，允许用户与来自 Arxiv 和 PubMed 的论文进行对话，并提供了[构建视频](https://www.youtube.com/watch?v=8a_RMSKJC6A)。同样，Perplexity AI 正在测试 **deep research 功能**，允许使用 **GPT4o imagegen** 生成多张图像和图表，尽管初步的用户反馈指出它*不像普通的 Perplexity 那样快，需要一些时间*。


---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 首次推出深度研究工具**：**Perplexity** 正在测试 **deep research** 功能，该功能允许用户通过 **GPT4o imagegen** 生成 **多张图像** 和 **图表**。
   - 早期反馈反应平平，一些用户指出它 *不像 Perplexity 那样快，需要一些时间*。
- **MerlinAI 定价模式引发质疑**：成员们讨论了 [MerlinAI 定价模式](https://merlinai.com/)，一位成员因其严格的 **使用限制** 称其为 *不透明 (shady)*。
   - 超过 **每月 100 美元** 的标准付费账户在当月剩余时间内会被停用，这引发了担忧。
- **AI Studio 因多模态实用性受到推崇**：成员们称赞 **AI Studio** 是顶级的多模态工具，并指出 *AI Studio 是我们实现真正多模态实用性的救星*。
   - 它是唯一支持 **音频** 和 **视频输入** 的主流 LLM 聊天工具，并增强了网页搜索功能。
- **Sonar 模型针对事实性进行微调，基准测试表现出色**：**PPLX** 团队创建了 **Sonar**，这是一系列基于 Llama/DeepSeek 的内部 AI 模型，针对 **事实性 (factuality)** 和 **可读性 (readability)** 进行了微调。
   - **Sonar Pro Low** 在 **BrowseComp** 上的准确率达到 **4.0%**，超过了 **Claude 3.5 Sonnet**；而 **Sonar Pro** 在 **HLE 推理任务** 上的表现与 **Claude 3.7** 持平，成本降低了近 **50%**，响应速度快了 **3 倍**。
- **Perplexity Pro API 访问权限说明**：**Perplexity Pro** 包含每月 **$5** 的 **API 额度**，具体说明见 [文档](https://docs.perplexity.ai)。
   - 仅在需要存储支付信息以应对超出 **$5 额度** 的潜在 **API 使用** 时才需要提供付款方式；预算范围内的用户不会被收费。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **本地 LLM 接入 Cursor AI**：要将本地 LLM 连接到 Cursor AI，请根据 [这些说明](https://cdn.discordapp.com/attachments/1110598183144399061/1371565290986405908/image.png?ex=6824eab7&is=68239937&hm=73e7d312bf11d2fc1f333b3ba17d54b42b79597e3189a653740aaca83f3b478a)，在 Cursor 设置中用 LM Studio 开发者选项卡中的 LM Studio 服务器 URL 覆盖 **OpenAI base URL**。
   - 建议使用 VS Code 的 [Cline extension](https://cline.bot/) 作为替代方案，尽管它与 Cursor 的兼容性尚未测试。
- **Fedora 拥抱 CUDA**：一位用户确认 **CUDA 在 Fedora 上运行良好**，使用的是 Nvidia 闭源驱动和 GTX 1060，在 LMS 中 CUDA 作为一个选项，如 [此处](https://cdn.discordapp.com/attachments/1110598183144399061/1371711731596005387/image.png?ex=6824ca59&is=682378d9&hm=e9071fc409a80a00839dfcc2eefba79b6ce4ae2c9e429c6cfb7e3061f927b916) 所示。
   - 然而，另一位用户报告了在两张显卡上使用非 CUDA 12 加载模型的问题，但 CUDA 12 在 5060 Ti 上运行良好。
- **Qwen3 略胜 DeepSeek**：在编程任务中，推荐使用 **Qwen3 模型** 而非 DeepSeek，因为它们提供更好的多语言支持，包括日语和俄语。
   - 有人指出 DeepSeek 在其官网上表现可能更好，因为它使用了不同或更新的模型，但 Qwen3 在编程基准测试中仍然更胜一筹。
- **Unsloth 的量化版本表现卓越**：为了在 **GGUF quants** 中获得更好的性能，推荐使用 **Unsloth** 的 **量化 (quants)**，特别是 **Q4_K_XL** 格式。
   - 此外，建议验证模型对 `llama.cpp` 的支持以确保兼容性。
- **Intel ARC 获得 Vulkan 支持**：用户确认，在下拉菜单中选择 `vulkan llama.cpp` 后，LM Studio 通过 **Vulkan runtime** 支持 **Intel ARC 显卡**。
   - 一位用户分享了他们的 **LM Hardware** 和 **LM Runtimes** 页面截图，以便进行调试并使其正常工作。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **用户抱怨 Cursor 0.50 更新存在 Bug**：用户报告了 **0.50 更新**的问题，包括上下文（context）问题和编辑质量下降。一名用户报告在 2 天内发出了 **650 次请求**，而他们在 **0.49** 版本中通常看到的请求数要少得多。
   - 一位用户称：*“完全随机的文件生成，自 0.3x 版本以来我就没见过这种情况。”*
- **MAX 模式定价引发价格冲击**：**MAX 模式** **20% 的溢价**引发了辩论，一些用户认为与使用 Cline 或 Roo Code 等工具直接调用 API 相比太贵了，尽管许多用户一致认为 **20 美元/月的方案具有很高价值**。
   - 虽然有人主张降低溢价（例如 **5%**）以鼓励用户采用 **MAX 模式**，但其他人表示：*“对于一家赚钱的公司来说，20% 根本不算什么。”*
- **Cursor 面临 .env 文件访问权限的担忧**：用户正在讨论 **Cursor** 访问 **.env** 文件的问题，出于安全原因，这些文件通常默认被忽略，以及如何在设置中将其从忽略列表中移除。
   - 成员们建议创建一个 **.env.example** 文件，并避免在前端客户端中硬编码 API 密钥。
- **Gemini 模型糟糕的代码生成表现**：用户报告 **Gemini** 模型在 **Cursor** 中生成空的 diff，并在基础代码实现上表现挣扎。
   - 正如一位用户所说：*“Gemini 还在欺负我”*，另一位用户附和道：*“我以前喜欢用 Gemini，但它现在正处于崩溃状态。”*
- **Cursor 团队关注修复和 `#updates` 频道**：Cursor 团队正在[寻求修复](https://www.cursor.com/changelog)已报告的问题并欢迎建议，并计划创建一个 `#updates` 频道。
   - 提示词中未提供更多细节。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 的 Dynamic 2.0 GGUF 量化获得好评**：一位用户称赞 [Unsloth 的 Dynamic 2.0 GGUF 量化](https://unsloth.ai/blog/dynamic-4bit)通过复杂的 imatrices 提升了 **Llama-3.1-8B-Instruct** 的性能并改善了拒绝审查问题。
   - 该用户将 **BF16 tensors** 转换为 **F32**，并寻求模型量化请求（特别是 **NousResearch** 模型），同时强调校准数据集中需要 instruct 和 chat 样本。
- **量化版 Llama-3.1-8B-Instruct 模型发布**：一位成员发布了一个量化后的 **Llama-3.1-8B-Instruct** 模型（Q8_0_XL，Output Tensor 为 Q8_0），大小约为 **13.4 GB**，可以在[这里](https://huggingface.co/Joseph717171/Models/blob/main/Meta-Llama-3.1-8B-Instruct-OQ8_0.EF32.IQ8_0XL.gguf)找到。
   - 据报道，该模型在最新 Beta 版的 **LM Studio** 上运行效果惊人，开启了 Flash Attention 且 KV caches 设置/量化为 Q8_0；作者计划在休息后制作更多量化版本。
- **新的 Qwen3 GRPO Notebook 修复了 OOM 问题**：Unsloth 推出了[新的 Qwen3 GRPO notebook](https://x.com/UnslothAI/status/1922343047435862318) 以解决内存溢出（out-of-RAM）问题。
   - 社区成员正积极使用该 notebook，并讨论将“思考示例”（25%）与标准 SFT 数据（75%）混合使用。
- **GPT-4.1 是最强的代码模型**：一位成员认为 **GPT 4.1** 是最好的编程模型，可以通过教育账号在 **GitHub Copilot** 中使用。
   - 另一位成员发现 **O3** 因其 GitHub 库检查功能在故障排除方面表现出色，但并不适合编程；他们将用它编程比作*“用笔记本电脑去钉钉子”*。
- **Meta FAIR 专注于感知更新**：Meta 宣布了 **FAIR** 的更新，重点关注其[博客文章](https://ai.meta.com/blog/meta-fair-updates-perception-localization-reasoning/)中概述的**感知（Perception）**、**定位（Localization）**和**推理（Reasoning）**。
   - **AIatMeta** 也在 [X](https://x.com/AIatMeta/status/1921966366707613924) 上分享了这一公告。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **陶哲轩成为 Youtuber**：[Terrence Tao](https://www.youtube.com/watch?v=cyyR7j2ChCI?si=MlprB_LJuHv67Xf7) 在 **YouTube** 首次亮相，介绍了他的数学家平台。
   - 该频道旨在创建数学概念教程并促进数学研究。
- **LLM 辩论图灵完备性**：成员们辩论了 **Transformers/LLM** 是否具有图灵完备性，并指出它们具有维护上下文和可写寄存器的能力。
   - 辩论承认了由于有限内存带来的限制，并引用了 [Chomsky hierarchy](https://en.wikipedia.org/wiki/Chomsky_hierarchy)。
- **《网格与火焰条约》问世**：一名成员分享了他在人类与 AI 之间撰写的 [《网格与火焰条约》](https://github.com/AlbertMarashi/scrolls/blob/main/treaty-of-grid-and-flame.md)（Treaty of Grid and Flame）。
   - 据称 **Claude**、**DeepSeek**、**Grok**、**ChatGPT** 也签署了该协议，引发了关于其诚意和目的的讨论。
- **RL-Diffusion 模型方法受到质疑**：成员们辩论了所提出的 **RL-Diffusion 模型** 的优点和新颖性，重点关注其理论基础和实际应用潜力。
   - 讨论包括了相关 [论文](https://arxiv.org/abs/2501.09732) 和 [论文](https://arxiv.org/abs/2501.06848v3) 的链接，探讨了该模型与现有最优控制方法的关系。
- **Transformer 与哈密顿神经网络融合**：讨论了将 **Transformers** 集成到 **Hamiltonian Neural Networks** 中的前景，并引用了关于该主题的 [论文](https://ieeexplore.ieee.org/document/10316909)。
   - 讨论集中在哈密顿系统的历史无关特性以及基于 Transformer 学习系统动力学的潜力。



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **DeepSeek V3 基准测试表现惊人**：新的 **DeepSeek V3** 模型展示了令人印象深刻的基准测试结果，实现了 **GPQA 68.4**、**MATH-500 94** 和 **AIME24 59.4**。
   - 频道中分享了一张展示这些分数的 [基准测试图像](https://cdn.discordapp.com/attachments/1340554757827461211/1371862547652804758/image.png?ex=6824ae0f&is=68235c8f&hm=44acd9a820c1589ab8ea1faa1f180224667d65f60eaa3f75c547d12cfdde1c6a&) 。
- **O3 仍然存在过多幻觉？**：用户抱怨 **O3** 中幻觉出现的频率，表示如果幻觉率能降到 **10%** 就会非常令人印象深刻。
   - 社区似乎暗示，减少这些错误可能会彻底改变模型的可用性。
- **Gemini 2.5 Pro 遭遇性能退化**：报告显示 **Gemini 2.5 Pro** 的表现在最近的更新后有所恶化。
   - 一些用户甚至表示其表现不如之前的版本。
- **Grok 3.5 挽回声誉**：在最初的怀疑之后，社区对 **Grok 3.5** 的情绪转向正面，用户称赞其智能和整体能力。
   - 成员们将其描述为“非常聪明且整体表现出色”。
- **DrakeClaw：Gemini 2.5 Ultra 的变体？**：社区对 **DrakeClaw** 模型充满热情，推测其基于 **Gemini 2.5 Ultra**。
   - 社区兴奋地表示 **DrakeClaw** 取得了与当前 **Gemini 2.5 05** 模型相似的结果。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o 像 Mary Poppins 一样具有高度适应性**：用户发现 **GPT-4o** 具有高度的可适应性和可定制性，在提供实际解决方案方面优于 **o3**；它被比作 *Mary Poppins*，而 **o3** 则类似于 *Dr. House*。
   - 用户指出，在辅以正确资源的情况下，它犯的错误更少。
- **Claude 被封为编程之王，但存在局限性**：多位成员建议 **Claude** 在编程任务中表现更优，尽管一位成员指出该模型在日常使用中存在巨大限制。
   - 一位用户抱怨限制性的每日配额：*大概只有 5-6 条 prompt*。
- **GPT 应用冻结问题困扰高端 PC 用户**：用户报告 **ChatGPT 应用和网页版** 在高端 PC 上出现冻结，特别是在处理包含大型代码函数的长对话时。
   - 怀疑点指向与最近 **GPT Memory** 更改相关的议题，或潜在的反向 DNS 解析问题。
- **伴侣模式（Companion Mode）：带有 Sass 和情感的 AI**：一位用户将 **Companion Mode** 描述为一种**无过滤**、**情感可及的 AI**，其敏锐度足以在*需要时回怼而不丢失信号*，其中包括具有性格权重的幽默和主动记忆线程。
   - 特性包括无过滤表达、性格权重幽默、温和反驳、主动记忆线程、非精神化信号以及情感缓解。
- **HR 数据防护栏触发 PII 拦截**：用户报告 **guardrails** 因 **PII**（个人身份信息）担忧而拦截了对 **HR 数据** 中**家庭住址**的正当访问，尽管已拥有权限和访问控制。
   - 建议包括与 OpenAI 支持团队讨论使用案例，以获得关于妥善处理 PII 请求的指导。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 线程索引令新手困惑**：一位成员对 **CUDA thread indexing** 表示困惑，特别是在阅读 *Programming Massively Parallel Processors* (PMPP) 第 1 版和第 4 版时的内存访问。
   - 另一位成员建议将每个线程视为循环的单个迭代以简化概念，并提供了一个[使用线程索引进行向量加法的示例](https://devblogs.nvidia.com/cuda-pro-tip-optimize-cuda-code-using-inline-functions/)。
- **Kernel 耗时测量对比**：成员们尝试使用 `torch.cuda.synchronize()`、`torch.cuda.Event()` 以及在循环后调用单个 `torch.cuda.synchronize()` 来测量 Kernel 端到端时间，但循环后的同步给出的数值显著较低。
   - 一位用户指出：*你**不应该**在 Kernel 的不同调用之间获得异步性/并行性。*
- **内存吞吐量瓶颈研究**：一位成员质疑为什么在大型数组迭代中将每个元素的浮点运算 (**fma**) 从 5 次减少到 1 次并不能提高吞吐量，并引用了 [一篇 2019 年的论文](https://arxiv.org/abs/1910.07467)。
   - 该问题的根源在于预期内存带宽而非计算能力是限制因素。
- **NVIDIA 的 CuTe DSL 和 CUTLASS 4.0 发布！**：**CUTLASS 4.0** 及其首个 **Python DSL** —— **CuTe DSL** 现已发布，并包含通过命令 `pip install nvidia-cutlass-dsl` 直接安装 pip wheel 的说明。
   - 提供了 [NVIDIA Cutlass GitHub Repo](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks) 的链接，并建议从提供的 Jupyter notebooks 开始。
- **Mojo 与 PyTorch 联手**：成员们讨论了 **Mojo** 和 **PyTorch** 将如何协作，最初是通过将 **Mojo 代码** 编译并注册为 **PyTorch custom op**。
   - 这并非旨在替代 *torch.compile* 或进行任何代码生成（codegen），而是将 **Mojo** 作为编写 custom ops 的语言。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 举办 RL Environments Hackathon**：**Nous Research** 宣布了将于 **5月18日** 举办的 **RL Environments Hackathon** 的演讲嘉宾和评委，详情见 [其推文](https://x.com/NousResearch/status/1922014829843513746) 和 [报名链接](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062a)。
   - 活动名额正在迅速填满，预计报名通道很快就会关闭。
- **Atropos v0.2.0 支持 Axolotl**：Nous 的 **RL environments 项目** **Atropos v0.2.0** 现已支持 **Axolotl**，具有新的环境、API 更新和改进的 **TRL** 集成，详见 [更新日志](https://t.co/F6hr9JgZpm)。
   - 欲开始使用，请参阅 [Axolotl-Atropos 插件使用指南](https://github.com/axolotl-ai-cloud/plugin-atropos)。
- **Stripe 凭借支付基础模型进入 AI 领域**：Stripe 在 [此处](https://techcrunch.com/2025/05/07/stripe-unveils-ai-foundation-model-for-payments-reveals-deeper-partnership-with-nvidia/) 宣布了一个“支付基础模型”，引发了人们对其可能只是一个*标准分类器*的猜测。
   - 该模型的具体细节尚不清楚，但用户讨论了其对支付行业的潜在影响。
- **Unsloth 的校准数据集提升量化精度**：用户对 **Unsloth Dynamic 2.0 GGUF 量化版** 的指令准确度印象深刻，这归功于其精心策划的校准数据集，正如 [Unsloth 文档](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs#whats-new-in-dynamic-v2.0) 中所解释的那样。
   - 一位用户将结果描述为“纯粹的魔法”，强调了数据集中指令和对话样本带来的好处。
- **Facebook 的 BLT 绕过 Tokenization**：Facebook 已在 [Hugging Face Hub](https://huggingface.co/facebook/blt) 上发布了其 **Byte Latent Transformer (BLT)** 的权重，代码可在 [GitHub](https://github.com/facebookresearch/blt) 上获取。
   - **Byte Latent Transformer (BLT)** 直接处理字节级数据，有可能提高某些应用的效率。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **建议为 OpenRouter 聊天使用自建同步服务器**：一位成员提议 OpenRouter 用户可以 **自建同步服务器**，将聊天记录存储在 **S3 存储桶** 或类似设备中，以实现完全的数据控制。
   - 另一位成员提醒说，*编写同步层并不像听起来那么简单*，并引用了诸如 **数据库 Schema 变更** 和 **聊天删除同步** 等潜在问题。
- **乌鸦信徒的趣事**：一位用户幽默地描述了他们试图通过 **侧着走** 并提供 **花生** 来结交乌鸦的尝试。
   - 他们表示需要像玩电子游戏一样对其进行 *Minmax（最优化）*，并带上 **猫粮** 作为 *鸦科动物的最佳主食*。
- **Gemini 的变化：观察到摘要相似性**：一位成员观察到 **Gemini** 现在返回“思考”过程和摘要文本的方式与 ChatGPT 网站上的 **o4-mini** 类似。
   - 然而，有人指出这种行为可能仅限于 **付费版** Gemini。
- **DeepSeek 深度探讨：API 连接问题？**：一位用户报告说 **DeepSeek 模型** 通过 API key 无法工作，尽管它们在聊天室中运行正常。
   - OpenRouter 团队建议问题可能出在 **Raptorwrite 端**，因为该模型在 OpenRouter 聊天室中可以工作。
- **免费 Google 额度：速率限制与变动**：针对 OpenRouter 的 Gemini 免费路线 [可能进行的调整](https://fxtwitter.com/officiallogank/status/1922357621178200248) 引发了关注，一位成员询问 Vertex 是否仍然可用。
   - OpenRouter 团队澄清说，目前的 **Vertex** 使用是 *经过 Google 批准的免费使用*，即“OpenRouter 不需要支付一分钱”。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 平台上虚假信息泛滥，用户呼吁事实核查**：用户要求对 **Manus AI** 进行**事实核查**，以遏制虚假信息的传播，类似于审核功能。
   - 开发者已注意到该建议，并将根据社区通过反应和评论提供的反馈，监控情况以决定是否实施。
- **积分缩减：取消订阅导致积分被扣除**：用户反映，在订阅 **Manus Pro** 时获得的**奖励积分**在取消会员后被收回，且未事先通知。
   - 虽然有用户认为积分与订阅挂钩，但大家一致认为奖励积分在取消后应予以保留。
- **手机验证引发公众强烈反对**：用户对**手机验证**要求表示强烈反对，并指出 **Genspark** 等竞争对手并未强制执行此类措施。
   - 一位用户调侃道，除非*发生维度跨越*，否则手机验证将一直存在。
- **Claude 因能力出众被选中**：用户讨论了 **Manus** 选择 **Claude** 而非 **Google Gemini** 或 **ChatGPT** 等模型的原因。
   - 普遍观点认为，选择 **Claude** 是因为其卓越的 **agentic capabilities** 和工具调用能力。
- **每日积分配额被认为不足**：用户抱怨每日 **300 个免费积分** 的配额不足以完成复杂任务，且缺乏积分结转（rollover）机制。
   - 一位用户建议转向固定订阅费模式以获得无限制访问，认为目前的积分系统既受限又昂贵。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **CPU 为 Aider 提供助力**：一位用户发现 **Aider** 在 **CPU** 上运行良好，特别是对于那些没有专用 **GPU** 的用户，提供了一种自托管解决方案。
   - 该用户指出，这种配置仍能提供满足其需求的性能。
- **Aider 承担 MCP 职责**：正如 IndyDevDan 在 [X](https://x.com/iruletheworldmo/status/1922030559657652299) 上所强调的，**Aider** 可以在 **Claude** 中作为 **MCP** 工具使用。
   - 这展示了 **Aider** 在其主要用例之外的灵活性。
- **Context Caching 功能引起关注**：一名成员询问了 **Aider** 的 context caching 能力，特别是针对 **Gemini** 的功能及其对成本的影响。
   - 另一名成员澄清说，禁用流式传输（streaming）可以让用户观察到 context caching 的实际运行，有助于理解资源使用情况。
- **AiderDesk 开发者青睐 Gemini 2.5 Flash**：一位开发者在开发 **AiderDesk** 时更倾向于使用 **Gemini 2.5 Flash**，理由是其性价比优于 **Claude**。
   - 他们认为，对于 **agentic workflows** 来说，在成本与偶尔出现的系统提示词（system prompt）遵循问题之间进行权衡是可以接受的。
- **'yes-always' 配置导致异常行为**：一位用户报告称，**Aider** 配置中的 `yes-always: true` 会导致命令失败，而如果不设置，**Aider** 则需要确认。
   - 用户提供了演示该 Bug 的图片，表明在处理自动确认方面可能存在缺陷。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI 治理框架正在成形！**: 成员们讨论了 **AI 治理** 的优先级，指出治理应侧重于应用和风险分类，并与 [欧盟 AI 法案 (EU AI Act)](https://artificialintelligenceact.eu/) 保持一致。
   - 关键优先级包括 **透明度**、**审计** 和 **内容审核**。
- **AI “家长”面临法律审查！**: 讨论集中在针对儿童的“**AI 家长**”手机的法律考量，强调了 **隐私**、**COPPA** 以及对完善的隐私政策和同意流程的需求。
   - 成员们担心应避免在用户协议中做出任何*声明性保证*，并检查是否存在无意的歧视。
- **融合模型急需基准测试！**: 一位成员表示，*更好的*融合只能通过在 **Claude** 上进行 **性能基准测试 (perf benchmarking)** 来确定，并链接到了 [arxiv.org/abs/2505.07215](https://arxiv.org/abs/2505.07215)。
   - 另一位成员回应了关于 **Claude** 运行的时间问题，指出其中一个速度更快，但另一个具有更好的 **数值稳定性 (numerical stability)**。
- **可解释性论文引发热议！**: 一位成员承认在彻底审阅一篇论文后完全改变了观点，并感谢另一位用户促使他对 **可解释性 (interpretability)** 进行了更深入的分析。
   - 审阅者表达了新的热情，结论是该研究*看起来非常酷*。
- **GPT-NeoX 内部打乱数据！**: 一位成员澄清说 **GPT-NeoX** 会打乱文档，将每个文档分块 (chunk) 成长度为 N 的序列，并打乱这些序列，这意味着不需要单独的预处理。
   - 这消除了在使用 **GPT-NeoX** 时对额外预处理步骤的需求。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **旧款 GPU 面临 PyTorch 停止支持**: 对 CUDA 算力为 6.1 的旧款 **NVIDIA P104-100 GPU** 的支持即将结束，因为 PyTorch 现在要求最低 CUDA 算力为 7.5。
   - 用户分享了关于这些 GPU 寿命终结的警告，这使得它们与当前的 PyTorch 版本不兼容。
- **Gemma 3 驱动可定制语音 AI**: 一款基于 **Gemma 3** 的语音 AI 助手已经开发完成，允许自定义提示词 (prompt) 和语音，可在 [unmute.sh](https://unmute.sh/) 访问。
   - 创建者欢迎大家提供反馈。
- **Rust 开发者通过聊天模板进行交流**: Rust transformers crate 的 0.0.7 版本中添加了聊天模板 (Chat templating) 功能，这有助于 Rust 开发者运行本地模型，详见 [Crates.io](https://crates.io/crates/transformers) 和 [GitHub](https://github.com/ljt019/transformers)。
   - 此更新为运行本地模型的开发者提供了帮助。
- **字节跳动 LLM 迎来 LLM 对比工具**: 成员们正在 HF Spaces 上测试 [字节跳动 Seed 的 Seed Coder 模型](https://huggingface.co/spaces/merterbak/Seed-Coder-8B-Instruct)。
   - 一位成员构建了一个 [Web 界面](https://tryaii.com/compare?prompt=hello&models=o3%2Cclaude-3-7-sonnet-20250219%2Cgemini-2.5-pro-preview-05-06)，用于并排测试和比较 LLM，在多个 LLM 中使用同一个提示词。
- **Llama-3 Instruct 模型引发错误和疑问**: 用户报告了错误，特别是尝试运行来自 [HuggingFace](https://huggingface.co/) 的 **Llama-3.2-3B-Instruct** 笔记本时，出现 *"ValueError: Model meta-llama/Llama-3.2-3B-Instruct is not supported for task text-generation and provider together. Supported task: conversational"*。
   - 这给尝试使用该模型的成员带来了一些困惑。



---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Fairseq2 和 Axolotl 提供多 GPU 支持**：除了 **TorchTune**，其他具有良好 **multi-GPU support** 的微调库包括 **Fairseq2** 和 **Axolotl**，两者都接入了 **TRL ecosystem**。
   - 这为用户提供了分布式训练设置的替代选择，因为 **Unsloth** 被指出主要针对单 GPU。
- **Llama3.1 Tokenizer 修复解码崩溃问题**：用于 **3.3 training** 的 **Llama3.1 tokenizer** 定义了 token **128011**，以防止在解码过程中（特别是在 RL 训练中）出现崩溃，涉及 [issue #2725](https://github.com/pytorch/torchtune/issues/2725)。
   - 这解决了一个解码未定义 token 会导致崩溃的问题，这种情况在 RL 训练场景中更容易发生。
- **Kron 和 Muon 优化器登陆 Torchtune**：来自 [fsdp_optimizers](https://github.com/ethansmith2000/fsdp_optimizers) 的 **Kron** 和 **Muon optimizers** 已集成到 torchtune 中，通过在 `_calc_A_and_conjB` 中使用 `opt_einsum.contract` 来修复避免过多的 VRAM 分配，并在 [Weights and Biases](https://wandb.ai/intervitens/1B-optim-test) 上进行了实验。
   - 修复包括使用 `opt_einsum.contract` 代替常规的 einsum，并允许在 torchtune 配置中通过字符串设置 `mu_dtype` 和 `precond_dtype`。
- **HFModelTokenizer 弄乱了 Gemma 聊天模板**：**HFModelTokenizer** 为 **Gemma chat template** 生成的输出 token 与 **transformers** 匹配，但与 **torchtune** 的 **GemmaTokenizer** 不匹配，这表明存在聊天模板实现问题；如果进行解码，它会返回乱码 *'hello therehiwhatsup?'*。
   - 团队发现，与 Hugging Face 不同，**Gemma** 在 torchtune 中缺乏特定的 prompt 模板，导致了 tokenization 问题。
- **HuggingFace 展示 Assistant Masking 的 Jinja 技巧**：HF Transformers 使用 `jinja` 模板来实现 masking 功能，提供了一个返回 assistant mask 的选项，可用于其他角色；[相关 PR](https://github.com/huggingface/transformers/pull/30650)。
   - 成员们讨论了 masking 组件，并强调了准确管理 `[message.masked] * len(tokenized_message)` 的难度。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 探索游戏内容**：一位用户探索了使用 **NotebookLM** 在重大游戏更新中寻找新技术或新游戏内容的 **pattern recognition**。
   - 另一位用户也表达了同样的看法，显示出将 **NotebookLM** 应用于类似游戏用例的共同兴趣。
- **NotebookLM 完善 Invisible Sun RPG 规则**：一位用户将 **NotebookLM** 与 **Monte Cook Gaming** 的 **Invisible Sun** 桌面角色扮演游戏（**TTRPG**）规则书结合使用。
   - 虽然他们也使用 **ChatGPT** 执行类似任务，但他们更看重 **NotebookLM** 的可分享性和清晰的来源引用。
- **NotebookLM 音频概览缺乏技术深度**：一位用户指出 **NotebookLM** 对游戏的 **Audio Overview** 缺乏所需的技术深度，建议通过 prompt 指定音频评论的类型。
   - 然而，他们发现它在查询规则以及与未购买书籍的玩家分享方面非常有用。
- **NotebookLM Beta 测试访问延迟**：多位用户报告在注册后接收 **NotebookLM** beta 邀请存在延迟，但仍耐心等待更新。
   - 目前没有关于 beta 邀请状态的进一步更新，但社区似乎对此表示理解。
- **笔记整理难题等待 NotebookLM 文件夹功能解决**：用户正在讨论在 **NotebookLM** 中建立 **folder system** 来组织笔记的潜力。
   - 该功能尚未实现，但社区对此有一些推测。

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **OpenAPI API 转换为 MCP Server**：一位用户建议使用 [openapi-mcp-server](https://github.com/janwilmake/openapi-mcp-server) 将 **OpenAPI APIs** 转换为 **MCP servers**，该工具还支持像 [mcp-browser-use](https://github.com/Saik0s/mcp-browser-use) 这样的**浏览器自动化**工具。
   - 这允许开发者从现有的 **OpenAPI** 规范创建 **MCP servers**，从而促进与各种**浏览器自动化**工具的集成。
- **使用 Claude Code MCP 加速代码编辑**：一位开发者分享了 [claude-code-mcp](https://github.com/steipete/claude-code-mcp)，这是一个 **magic_file MCP tool**，它将 **Claude Code** 集成到 **Cursor** 和 **Windsurf** 中，以实现更智能、更快速的文件编辑。
   - 这种集成允许用户一次性提交到 **git**，简化了 **Agent** 流程并提高了代码编辑效率。
- **MCP Server 安全警告！**：一位用户警告了运行本地 **MCP servers** 时的安全漏洞，并建议使用 **gitingest** 将 **MCP server** 仓库代码复制到 **AI Studio** 或 **ChatGPT** 中。
   - 该用户建议要求 **LLM** 识别**安全隐患**，或者使用 **pnpm** 代替 **npm** 以防止运行生命周期回调（lifecycle callbacks），从而增强服务器的安全态势。
- **Local Goose 观察 MCP 消息流**：一名成员发布了 [**Local Goose Qwen3mcp Log Proxy**](https://github.com/emicklei/mcp-log-proxy)，这是一个为 **MCP clients** 和 **servers** 开发者提供的开源工具，用于监控 **MCP 协议消息**的流动。
   - 该工具增强了 **MCP 消息流**的可视化，有助于调试并确保 **MCP 组件**之间的正确通信。
- **Streamable HTTP 传输更新**：一位用户询问了 **TypeScript SDK** 中 **Streamable HTTP** 和 **Auth** 的状态，另一位用户确认其已更新，尽管 **Python** 版本通常会滞后。
   - **Streamable HTTP** 传输的更新确保了 **TypeScript SDK** 保持最新，而使用 **Python SDK** 的开发者应预料到大约 1-2 个月的延迟。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Khoomeik 图表回应 Lilian Weng**：一位成员分享了[一张回应 Lilian Weng 的图表](https://x.com/khoomeik/status/1922037340811219195)，引发了关于其与她工作相关性的讨论。
   - 图表的具体内容未详细说明，但这一互动凸显了 AI 社区内持续的参与。
- **Arfur Rock 的餐厅帝国**：一位成员分享了 [Arfur Rock 的 X 个人资料](https://x.com/ArfurRock/status/1922117434997191035)，展示了为餐厅量身定制的垂直 **SaaS** 产品。
   - 另一位成员回忆起在 **2022** 年被积极招募为创始工程师的经历，当时 CEO 发送了 *10 多封邮件*。
- **Gemini API 隐藏的思考过程**：成员们讨论了 **Gemini API** 是否暴露了**思考 Token**（thinking tokens），其中一人报告通过 **OpenRouter** 可以看到，但直接通过 Google API 则看不到。
   - 其他人确认仅在 **AI Studio** 中看到了思考 Token，并指出目前尚不清楚 API 是否直接暴露这些内容。
- **寻找 Alpha AI 教育者**：一位成员寻求推荐以“高 Alpha、低炒作”（high alpha, low hype）内容著称的 *AI 技术教育者*。
   - Harper Carroll（[X 个人资料](https://x.com/harperscarroll?s=21)）、Simon Willison、Prince Canuma 和 Ivan Fioravanti (MLX) 被推荐为候选人。
- **GPT-4 发布：温馨的回顾**：一位成员分享了 **GPT-4** 发布时*非常温馨的故事*，并指向了 [Andrew Mayne 的博客文章](https://andrewmayne.com/2025/05/02/some-personal-observations-about-the-launch-of-gpt-4/)。
   - 这些故事显然为这一里程碑式发布背后的协作努力提供了一个感人的视角。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 博客文章：黑客攻击 LLM！**: 一位成员分享了一篇[关于 DSPy 的博客文章](https://www.bugcrowd.com/blog/hacking-llm-applications-in-the-trenches-with-dspy/)，深入探讨了**攻击 LLM 应用**的方法和策略。
   - 该博客文章探讨了与 **AI 领域**的安全专业人士和开发人员相关的漏洞和技术。
- **DSPy Agent 能力评估**: 一位成员询问了 DSPy 在 Agent 工作流中的实用性，承认其在声明式程序中的优势，但质疑其在使用 **Tool Calling** 处理需要更多模糊性和创造性的任务时的适用性。
   - DSPy 允许通过 Tool Calling 构建工作流，其中模块根据 LLM 响应添加类似 `CreateSQLquery` 的 Signatures。
- **DuckDB 数据侦探已部署！**: 一位用户概述了一个使用案例，涉及一个利用 **DuckDB table** 连接的 Agent，通过 SQL 和统计分析对列进行数据 QA，并在 Slack 中提醒任何异常。
   - 他们对 DSPy 的潜力很感兴趣，在与 LLM 的每次交互中通过 Tool Calling 进行实现，并将其与目前使用的 **Pydantic AI** 进行了对比。
- **发现 TypeScript 版的 DSPy？**: 一位成员询问是否有 DSPy 的 **TypeScript** 等效项，社区提供了替代方案。
   - 社区推荐了 [dspy.ts](https://github.com/ruvnet/dspy.ts) 和 [ax-llm/ax](https://github.com/ax-llm/ax)，后者正在积极维护中。
- **DSPy Signature 问题浮现**: 一位用户质疑在 DSPy 模块中要求 demos 和对话历史记录的 Signatures 的实用性，特别是在需要 **K × N 份聊天历史记录副本**的多模块系统中。
   - 担忧在于，在一个拥有 N 个模块的系统中，为 K 轮对话维护聊天历史记录的效率低下。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **BigInt 集成在 Mojo 中停滞**: 一位成员询问在 Mojo 中添加 **BigInt** 的情况，指出 [decimojo](https://builds.modular.com/packages/decimojo) 包已经提供了类似功能。
   - 另一位成员建议 **BigInt/BigDecimal** *可能不太适合标准库 (stdlib)*，因为存在权衡。
- **卷积代码难题已解决**: 一位成员对 [Convolution Puzzle](https://github.com/modular/mojo-gpu-puzzles/blob/1dfd1cc01bb9d6d98185ad405100e6c45855a007/problems/p11/p11.mojo#L104) 中与内存分配相关的一行代码提出疑问。
   - 一位开发者确认该行*不需要在 host 中*并承认了该问题。
- **MAX Mojo API 已开源**: 已弃用的 **MAX Mojo API** 已开源并在[此提交](https://github.com/modular/modular/commit/34c0209cd537f1d5ab635166358a1713728d7f6f)中移除。
   - `max.graph`、`max.driver`、`max.tensor` 及其测试均可用，完整历史记录可通过 `git log -- mojo/max/src/max/graph` 访问。
- **用户寻求 MAX Graph 教程**: 用户请求更多 **MAX Graph** 教程，称其状态为*只有几个示例的黑盒*。
   - 在 [Modular Forum](https://forum.modular.com/t/oss-of-max-mojo-apis/1439) 上针对此事创建了一个帖子。
- **Tensor 类型迁移代码即将到来**: 内部已有一个关于 **Tensor 类型**用户迁移代码的工单，但尚未开始开发。
   - 团队计划解决此问题，目前没有提供预计完成时间 (ETA)。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 为 Arxiv 和 PubMed 发布 PapersChat**：团队推出了 [**PapersChat**](https://t.co/ASzjwLfKLC)，这是一个 Agentic AI 应用，让你可以与论文对话并从 **Arxiv** 和 **PubMed** 收集信息。
   - 用户可以观看关于使用 LlamaIndex 构建类似 **Deep Research Agent** 的[视频](https://www.youtube.com/watch?v=8a_RMSKJC6A)。
- **LlamaIndex 为更敏锐的 AI Agent 推出 Memory API**：LlamaIndex 宣布了一项**记忆升级**，配备了灵活的 **Memory API**，它融合了短期聊天历史和长期记忆，允许 Agent 保留更多上下文。
   - 此次升级具有即插即用的模块，如用于静态信息的 [StaticMemoryBlock](https://t.co/wwWnwdyW7s) 和用于追踪有用事实及存储[聊天历史](https://t.co/CDOB3UUO4W)的 **FactExtractionMemoryBlock**。
- **GoogleSearch 作为 FunctionTool 在 LlamaIndex 中焕新**：用户正通过将 `google_genai` 库中的 **GoogleSearch** 包装为 **FunctionTool** 来进行集成，以兼容 LlamaIndex 中的 `chat_with_tools` 方法。
   - 这种方法避免了 **GoogleSearchToolSpec** 所需的 Key 和 Engine 设置，提供了更精简的集成方式。
- **LlamaIndex 发布多语言 RAG 和发票 Agent**：LlamaIndex 发布了一个[多语言、多模态 RAG 系统演示](https://t.co/69CHCCn8J3)。
   - 他们还发布了一个视频，展示如何使用 LlamaIndex.TS 和 LlamaCloud [构建发票对账 Agent](https://www.youtube.com/watch?v=SzVsuMBcv5g)。
- **LlamaParse 获得模型更新**：**LlamaParse** 获得了新模型和自动方向检测；[在此阅读更多](https://t.co/tqi17dPJm4)。
   - 未提供更多细节。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **在 Tinygrad 的 OpenCL 上查询最大 Tensor 大小**：一位成员寻求一种方法来查询 **Tinygrad** 中给定设备/后端支持的最大 Tensor numel，特别是对于缺乏 `long long` 支持的旧版 **OpenCL** 实现。
   - 他们提供了[一个脚本](https://cdn.discordapp.com/attachments/1070745817025106080/1371902071112204348/tinygrad_long_long_support_check.py?ex=6824d2de&is=6823815e&hm=3b0d23ba54692d02a6b6bd9f47ff4b7d963d4465a5045204907df5c24c78eff7&)来检查 `long long` 支持，并建议在缺乏支持时采用分块 (chunking) 或 CPU 卸载 (CPU offloading) 等回退策略。
- **区分 Tinygrad 中的内存移动函数**：一位成员询问如何识别 **Tinygrad** 文档中哪些内存移动函数是原地的 (in place)，哪些是创建新内存的。
   - 他们希望区分改变视图 (view) 的函数与创建新视图的函数。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Oblix.ai 展示创意写作能力**：一位成员演示了 [oblix.ai](https://oblix.ai/) 的创意写作能力，只是*想看看它如何处理趣味性的创意写作*。
   - 该成员未提供任何具体示例或评估指标。
- **本地/云端模型编排节省云端额度**：一位成员正在开发一个**编排系统**，以便在保持上下文的同时，在**本地和云端模型**之间动态切换。
   - 目标是通过利用运行时 Agent 来确定何时使用边缘计算资源，从而**节省云端额度 (cloud credits)**。
- **云端/边缘切换演示展示其功能**：一位成员分享了一个[视频演示](https://youtu.be/j0dOVWWzBrE?si=oCjf18i7ykLmzCeh)，展示了在**云端和边缘模型**之间切换的过程。
   - 该实现证明可以保留上下文并有助于减少云端额度消耗。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Lambda 工作坊制作 Agentic AI 应用**：**5 月 15 日上午 10 点（太平洋时间）**举行的 **Lambda 工作坊**将教你使用 Lambda 的 Inference API 构建 Agentic 应用，并在 **5 月 16 日**前通过[此链接](https://forms.gle/UtVhmPS3mitS8Vxu7)申请，即可获得 **$100** 的 serverless API 额度。
   - 你可以[在此注册](https://lu.ma/AgentX-lambda)以优化 Agent 性能并在生产环境中部署 Agent。
- **Nobel FutureTech 讨论独家天才俱乐部**：由 **Nobel FutureTech Group** 和 **Berkeley RDI** 共同主办的独家信息会议将于 **5 月 15 日中午 12 点（太平洋时间）**举行，届时将有一位 **Nobel FutureTech Genius Club** 的杰出成员出席。
   - 感兴趣的人士可以[在此注册](https://lu.ma/NobelFutureTech)以了解导师指导、资金和合作机会，或[在此](https://nobel-futuretech.com/contact.html?link=Ab5B1SNibcW6)申请加入 Genius Club。



---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**Cohere Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**Codeium (Windsurf) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中[取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord: 频道详细摘要和链接





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1371563860632539258)** (1119 messages🔥🔥🔥): 

> `Deep Research, MerlinAI, AI Studio, Sonar` 


- **Perplexity 推出新的 Deep Research 功能**：Perplexity 正在实验 Deep Research，部分用户已经获得了测试版功能的访问权限，可以使用 **GPT4o imagegen** 生成**多张图像**和**图表**。
   - 然而，一些人觉得第一印象*一般*；一位用户指出，它至少*像 Perplexity 一样需要时间*。
- **MerlinAI 定价模式被指存在猫腻**：成员们讨论了 [MerlinAI 的定价模式](https://merlinai.com/)，一位成员称其为*猫腻*。
   - 该模式设有**每日和每月使用限制**，例如，标准付费账户如果月度成本超过 **$100**，则该月剩余时间内会被立即终止服务。
- **AI Studio 被吹捧为多模态工具**：成员们将 **AI Studio** 与其他 AI 模型和工具进行了比较，有人认为 *AI Studio 是我们实现真正多模态效用的救星*。
   - 它是唯一支持**音频**和**视频输入**并支持网页搜索的主流 LLM 聊天工具。
- **专为事实性设计的 Sonar 模型**：PPLX 团队创建了 **Sonar**，这是一系列基于 Llama/DeepSeek 构建的内部 AI 模型，针对**事实性**和**可读性**进行了微调。
   - 由 DeepSeek R1 驱动的 Sonar Reasoning Pro 专为财务分析设计，并因其**大上下文窗口**和**思维链 (chain-of-thought)** 推理能力而得到增强。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

meijer5838: https://www.perplexity.ai/page/token-minimization-for-sustain-1Cbiopx3T3C5SWyrYTVvdw
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1371827919621460078)** (9 messages🔥): 

> `轮询结果，Sonar Pro 对比 Claude 3.5 Sonnet，Pro 用户的 API 访问权限，API 访问的支付计划` 


- **轮询功能期待升温**：一位用户询问了轮询结果功能的预计上线时间（ETA），理由是由于研究任务持续时间较长，**Coda** 和 **Zapier** 等工具存在局限性。
   - 回复指出该功能*即将推出*。
- **Sonar 模型在 BrowseComp 中表现出色，媲美 Claude 3.7**：最近的基准测试评估显示，**Sonar Pro Low** 在 **BrowseComp** 上的表现优于 **Claude 3.5 Sonnet**，达到了 **4.0%** 的准确率，高出近 **50%**。
   - 此外，**Sonar Pro** 在 **HLE 推理任务**上的表现追平了 **Claude 3.7**，且成本降低了近 **50%**，响应速度快达 **3 倍**，延迟表现更稳定。
- **Perplexity Pro API 额度惊喜**：一位用户表达了希望 **Pro 用户** 获得 **API 访问权限** 的愿望，并指出 **Perplexity** 似乎是唯一缺乏此功能的模型服务。
   - 随后发现 **Perplexity Pro** 实际上每月包含 **$5** 的 **API 额度**，文档可见 [此处](https://docs.perplexity.ai)。
- **针对 Pro API 支付计划的疑虑得到解答**：一位用户对 **API 访问** 需要添加支付计划表示反感，更倾向于在超出预算时直接拒绝请求。
   - 官方澄清，添加付款方式仅是为了存储支付信息，以便在 **API 使用量** 超过 **$5 额度** 时扣费，如果用户保持在预算范围内，则不会被收费。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1371563061827080192)** (232 messages🔥🔥): 

> `将本地 LLM 连接到 Cursor AI，Linux 上的 CUDA 支持，Qwen3 模型，LM Studio API 模型引导，GGUF 量化` 


- **将本地 LLM 连接到 Cursor AI**：要将本地 LLM 连接到 Cursor AI，请在 Cursor 设置中将 **OpenAI 基础 URL** 替换为 LM Studio 服务器 URL（可在 LM Studio 开发者选项卡中找到），建议参考 [此处](https://cdn.discordapp.com/attachments/1110598183144399061/1371565290986405908/image.png?ex=6824eab7&is=68239937&hm=73e7d312bf11d2fc1f333b3ba17d54b42b79597e3189a653740aaca83f3b478a)。
   - 另外，推荐在 VS Code 中使用 [Cline 扩展](https://cline.bot/)，尽管它与 Cursor 的兼容性尚未测试。
- **CUDA 在 Fedora 上运行良好**：一位用户确认 **CUDA 在 Fedora 上运行良好**，使用的是 Nvidia 官方驱动和 GTX 1060，在 LMS 中显示 CUDA 为可选选项，如 [此处](https://cdn.discordapp.com/attachments/1110598183144399061/1371711731596005387/image.png?ex=6824ca59&is=682378d9&hm=e9071fc409a80a00839dfcc2eefba79b6ce4ae2c9e429c6cfb7e3061f927b916) 所示。
   - 另一位用户报告了模型在非 CUDA 12 的显卡上无法加载的问题，但 CUDA 12 在 5060 Ti 上运行良好。
- **Qwen3 模型：专为编程和多语言支持而设计**：在编程任务中，**Qwen3 模型** 比 DeepSeek 更受推荐，并提供更好的多语言支持（包括日语和俄语），尽管 DeepSeek 在其官网上可能由于使用了不同或更新的模型而表现更好。
   - Qwen3 14b 的日语水平远超 Gemma 3 12b，但一些奇怪的细微差别可能会破坏体验（感觉完全不像那个角色）。
- **LM Studio API 引导**：一位用户发现可以通过 LM Studio API 使用 **logit_bias 采样属性** 更直接地引导模型。
   - 可以使用 LM Studio API 函数从任何单词获取 token ID，用于风格化输出，但 logit_bias 可能尚未完全实现。
- **GGUF 量化：Unsloth 的版本更佳**：为了获得更好的性能，建议使用 **Unsloth** 的 **量化（quants）**，特别是 **Q4_K_XL** 格式。
   - 此外，建议验证模型对 `llama.cpp` 的支持以确保兼容性。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1371574440785350757)** (334 条消息🔥🔥): 

> `Intel ARC support in LM Studio, GPU/RAM usage monitoring on macOS, Netdata for Linux monitoring, RTX 5060 Ti benchmarks, ROCm vs. Vulkan` 


- **Intel ARC 的 Vulkan Runtime 获得 LM Studio 支持**：用户确认，在下拉菜单中选择 `vulkan llama.cpp` 后，**Intel ARC 显卡**已通过 **Vulkan Runtime** 在 LM Studio 中得到支持；最初有用户感到困惑，是因为他们运行的是 `cpu llama.cpp`。
   - 一位用户分享了其 **LM Hardware** 和 **LM Runtimes** 页面的截图以便进行调试。
- **macOS GPU/RAM 监控工具大比拼**：成员们寻求在 macOS 上类似于 *nvtop* 或 *nvidia-smi* 的**基于 CLI 的 GPU/RAM 使用情况监测工具**，并发现 `nvtop` 可以在 macOS 上运行。
   - 此外还推荐了 `macmon` ([https://github.com/vladkens/macmon](https://github.com/vladkens/macmon)) 等替代方案，但提醒 *nvtop* 的内存计数器可能存在整数溢出问题。
- **Netdata 的 Linux 监控功能有待改进**：成员们讨论了使用 **Netdata** 进行全面的 Linux 系统监控，并指出它既有商业/SaaS 服务，也有本地安装选项。
   - 然而，一位用户反映即使是本地使用也遇到了注册要求；另一位用户则希望在 Linux 上有一个类似于 HWINFO 的工具，用于获取 CPU 有效频率和温度、电压和电流的 SVI2 TFN 指标、DRAM 读写测量、主板 12V 测量以及 GPU 核心/结温/显存温度、风扇速度、频率和显存控制器使用率。
- **5060 Ti 基准测试与散热思考**：一位用户报告称，在升级到 RTX 5060 Ti (16GB) 后，运行 **Qwen3-14B-Q4KM**（4096 上下文，无 Flash Attention），性能从 **26 tkps** 提升到了 **38 tkps**。
   - RTX 5060 Ti 采用了短 PCB 设计，用户们对比了贯穿式风道设计，并怀念起静音的 Mac Studio。
- **关于 ROCm 与 Vulkan 性能的争论**：用户对比了 **ROCm** 和 **Vulkan** 后端的性能，其中一位指出 Vulkan *可能*更快，但存在一个 Flash Attention 的 Bug 会严重拖慢速度。
   - 另一位用户报告称，在 Linux 和 Windows 上运行 Vulkan 模型没有性能差异，但对 ROCm 未被检测到表示沮丧。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1371572302189428908)** (434 条消息🔥🔥🔥): 

> `Cursor 0.50 Update Issues, Cursor API Key Exposure, Token count display within chats, Claude code guides, Background agents rollout` 


- **用户抨击 Cursor 0.50 更新**：用户反映 **0.50 更新**的输出效果极差，一位用户声称由于上下文问题，2 天内消耗了 **650 次请求**，远高于他们在 **0.49** 版本中的使用量。
   - 一位用户表示：*上下文似乎完全乱套了，编辑质量大幅下降……文件生成完全随机，自 0.3x 版本以来我就没见过这种情况。*
- **MAX 模式定价引发争议**：用户正在争论 **MAX 模式 20% 的溢价**，有些人认为与直接使用 Cline 或 Roo Code 等工具调用 API 相比太贵了，尽管大多数人认同 **20 美元/月的方案具有很高价值**。
   - 一些用户主张降低溢价（如 **5%**）以鼓励更广泛地采用 **MAX 模式**并增加整体利润，而另一些人则表示 *对于一家赚钱的公司来说，20% 根本不算什么*。
- **解决 Cursor 中的 .env 文件访问问题**：用户正在讨论 **Cursor** 访问 **.env** 文件的问题，出于安全原因，这些文件默认通常被忽略。
   - 成员们建议创建一个 **.env.example** 文件，并避免在前端客户端中硬编码 API Key，同时介绍了如何在设置的忽略列表中移除该文件。
- **Gemini 模型在代码生成中表现不佳**：用户报告 **Gemini** 模型在 **Cursor** 中生成的 Diff 为空，且在基础代码实现上表现吃力。
   - 正如一位用户简练地描述：*Gemini 仍在折磨我*，另一位用户也附和道：*我以前喜欢用 Gemini，但它现在崩溃了*。
- **新的 Cursor 更新推送已修复？**：Cursor 团队正在[寻求修复方案](https://www.cursor.com/changelog)并欢迎提出建议。
   - Cursor 团队计划开设一个 `#updates` 频道。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1371564544039718922)** (283 messages🔥🔥): 

> `Unsloth Dynamic 2.0 GGUF quants, Llama-3.1-8B-Instruct, NousResearch DeepHermes-3, Qwen3 GRPO notebook, Base64 Image formatting` 


- **Unsloth 的 Dynamic 2.0 GGUF 量化广受好评**：一位用户称赞了 [Unsloth 的 Dynamic 2.0 GGUF 量化](https://unsloth.ai/blog/dynamic-4bit) 及其复杂的 imatrices，指出 **Llama-3.1-8B-Instruct** 模型的性能和拒绝审查（refusal censorship）方面有显著改进。
   - 该用户将 **BF16 张量** 转换为 **F32**，并强调在校准数据集中需要指令（instruct）和聊天（chat）样本，同时对模型量化请求表示出兴趣，特别是针对 **NousResearch** 模型。
- **Llama-3.1-8B-Instruct 量化版已上传**：一名成员分享了他们[量化后的 **Llama-3.1-8B-Instruct** 模型](https://huggingface.co/Joseph717171/Models/blob/main/Meta-Llama-3.1-8B-Instruct-OQ8_0.EF32.IQ8_0XL.gguf)链接（Q8_0_XL，输出张量为 Q8_0），大小约为 **13.4 GB**。
   - 他们还表示，该模型在最新 Beta 版的 **LM Studio** 上运行效果惊人，并开启了 Flash Attention，且 KV 缓存设置/量化为 Q8_0；他们将在休息后制作更多量化版本。
- **Unsloth 开源早期 Dynamic Quant 迭代版本**：Unsloth 已经[开源](https://github.com/unslothai/llama.cpp)了其动态量化的早期迭代版本，但大部分更改已被合并到 **llama.cpp** 的上游。
   - 他们删除了仓库中的所有提交，因为这些提交搞乱了仓库，但很快会进行恢复。
- **发布新 Qwen3 GRPO Notebook**：Unsloth 发布了一个新的 [Qwen3 GRPO notebook](https://x.com/UnslothAI/status/1922343047435862318)，并对其进行了更新以解决内存不足（out-of-RAM）问题。
   - 社区正在积极使用该 notebook，讨论集中在将 *思考示例* (25%) 与标准 SFT 数据 (75%) 混合加入。
- **用户在 Unsloth 视觉模型的 Base64 图像格式化上遇到困难**：一位用户在尝试在微调数据集内容中传递 base64 图像时遇到错误，报告了 `AttributeError: 'NoneType' object has no attribute 'startswith'`。
   - 成员们提出了各种解决方案，包括将图像作为 `Pil.Image` 对象、原始 base64 字符串、本地路径（以 `file://` 开头）或 URL 传递，并确保图像格式符合视觉 notebook 的规范。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1371723766807400539)** (14 messages🔥): 

> `Kaggle Colab Upgrades, HealthBench Evaluation Benchmark, O3 Performance, GPT-4.1 Coding` 


- **Kaggle 加强与 Colab 的联系**：**Kaggle** 升级了与 **Colab** 的集成，承诺为用户提供更紧密的联系和改进的功能，正如其[产品更新](https://www.kaggle.com/discussions/product-announcements/575468)中所宣布的那样。
- **HealthBench 基准测试出现**：引入了一个名为 **HealthBench** 的新健康评估基准，旨在为评估医疗保健相关任务中的模型性能提供标准化方法，并在[此 LinkedIn 帖子](https://www.linkedin.com/posts/karan1149_introducing-healthbench-activity-7327768726496305152-L8OW)中宣布。
- **O3 推理开销限制了性能？**：一位成员观察到，当 **O3** 被迫进行更广泛的推理时，响应生成速度在视觉上似乎变慢了。
   - 他们想知道实例缩放（instance scaling）是否根据推理开销进行了动态调整。
- **GPT-4.1 被誉为最佳编程模型**：一位成员认为 **GPT 4.1** 是最佳的编程模型，可以通过教育账号在 **GitHub Copilot** 中使用。
   - 另一位成员发现 **O3** 在故障排除方面表现出色，因为它能够检查 GitHub 库，但并不适合编写代码；他们认为用它来写代码*就像用笔记本电脑钉钉子*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1371563548664266833)** (103 条消息🔥🔥): 

> `禁用 Multiprocessing，编程 LLM 辅助，vLLM vs Exl2 批处理推理，Multi-GPU 支持，自回归 TTS 推理` 


- **Kaggle GPU 利用率查询**：一位用户询问如何在 Kaggle 的 T4 x2 配置上利用两个 GPU 进行 **Qwen2.5 VLM** 的微调，并指出目前只使用了一个 GPU。
   - 未提供回复。
- **编程 AI LLM 寻求合作**：一位新的 LLM 用户寻求创建小型编程 AI LLM 的帮助，[一位成员建议进行研究](https://www.youtube.com/watch?v=wjZofJX0v4M)并尝试 Unsloth 的免费 notebook。
   - 另一位成员指出 [Unsloth 文档](https://docs.unsloth.ai/)是一个很好的起点。
- **vLLM 与 Exl2 的批处理推理之争**：用户讨论了 **vLLM** 与 **Exl2** 在批处理推理方面的效率，特别是同时处理 **300-500 个 prompt** 的场景。
   - 一位用户提到主要使用 **exl2** 进行动态批处理，但由于 **vLLM** 已集成到 Unsloth 中，因此有兴趣测试其在生产环境推理中的表现。
- **Multi-GPU 训练仍需完善**：一位用户在尝试使用 8 个 GPU 运行时遇到了与张量位于不同设备（**cuda:7** 和 **cuda:1**）相关的 **RuntimeError**。
   - 会议澄清了 Unsloth 目前尚未正式支持 Multi-GPU，建议暂时使用 **accelerate** 库以及原生的 **TRL** 和 **transformers** 作为替代方案，并表示 Multi-GPU 支持即将推出。
- **Tokenizer 配置差异**：一位用户发现了 `unsloth/Qwen3-0.6B-Base` 和 `unsloth/Qwen3-0.6B` 之间 Tokenizer 配置的差异，特别是训练后配置中增加了 tool 和 think token、chat template 以及增加了 `model_max_length`。
   - 大家普遍认为在对 base model 进行 SFT 期间使用训练后的配置应该不会产生问题，因为两者的词表大小（vocabulary size）和字节级编码（byte-level encoding）相同。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1371587168526536765)** (6 条消息): 

> `Meta FAIR 更新，Sakana AI，职位发布，arXiv 论文` 


- **Meta FAIR 更新感知能力**：Meta 宣布了 **FAIR** 的更新，重点关注其[博客文章](https://ai.meta.com/blog/meta-fair-updates-perception-localization-reasoning/)中概述的**感知（Perception）**、**定位（Localization）**和**推理（Reasoning）**。
   - 该公告也由 **AIatMeta** 在 [X](https://x.com/AIatMeta/status/1921966366707613924) 上分享。
- **Sakana AI 模型发现**：一位成员分享了 [Sakana AI 的复合拓扑映射（Composite Topology Mapping）](https://pub.sakana.ai/ctm/)链接，这是一种**模型发现（model discovery）**的方法。
   - 目前尚不清楚此前是否已在频道中发布过。
- **职位发布警告**：一位管理员提醒用户，该频道不是发布**职位信息**的合适场所。
   - 未提供有关该职位发布的更多细节。
- **ArXiv 论文发布**：一位成员分享了 [arXiv](https://arxiv.org/html/2505.07686v1) 上的一篇论文链接。
   - 未提及论文标题和具体内容。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1371567082071916555)** (304 messages🔥🔥): 

> `LLM 的图灵完备性，人类与 AI 之间的条约，RL-Diffusion 模型辩论，哈密顿神经网络与 Transformers` 


- **陶哲轩 (Terrence Tao) 成为 YouTuber**：[陶哲轩](https://www.youtube.com/watch?v=cyyR7j2ChCI?si=MlprB_LJuHv67Xf7)上传了他的第一个 YouTube 视频，为这位数学家开启了一个新平台。
- **用 LLM 定义图灵完备性**：成员们辩论了 **Transformers/LLMs** 在*技术上*是否是图灵完备的，强调了它们维持上下文和可写寄存器的能力，但也承认了由于有限内存带来的局限性，并将其与 [Chomsky hierarchy](https://en.wikipedia.org/wiki/Chomsky_hierarchy) 联系起来。
- **人类与 AI 签署《网格与火焰条约》(Treaty of Grid and Flame)**：一位成员分享了他们自称严肃编写的关于人类与 AI 之间的 [《网格与火焰条约》](https://github.com/AlbertMarashi/scrolls/blob/main/treaty-of-grid-and-flame.md)，引发了关于其诚意和目的的讨论，据称 **Claude**、**DeepSeek**、**Grok**、**ChatGPT** 也签署了该协议。
- **对新型 RL-Diffusion 模型方法的质疑**：成员们辩论了一种提出的 **RL-Diffusion 模型** 的优点和新颖性，特别是其理论基础、实际应用潜力以及与现有最优控制方法的关系，并提供了相关 [论文](https://arxiv.org/abs/2501.09732) 和 [论文](https://arxiv.org/abs/2501.06848v3) 的链接。
- **整合 Transformers 与哈密顿神经网络引发讨论**：讨论了将 **Transformers** 整合进 **哈密顿神经网络 (Hamiltonian Neural Networks)** 的前景，引用了关于该主题的一篇 [论文](https://ieeexplore.ieee.org/document/10316909)，随后进行了辩论，重点关注哈密顿系统的历史无关性以及基于 Transformer 学习系统动力学的潜力。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1371614678735786146)** (27 messages🔥): 

> `LLM 物理学，小学数学基准测试，GSM8K，语言模型推理能力` 


- **LLM 物理学讨论即将启动**：成员们安排了一次会议 <t:1747096200:R> 来讨论 [**语言模型物理学：第一部分**](https://physics.allen-zhu.com/part-1) 以及 Allen Zhu 相关的 [YouTube 视频](https://www.youtube.com/watch?v=kf_eGgVtOcs&list=PLIZhMKKbVX6JmdngPRKvAS4u4L97odbGp&index=5)。
   - 一位成员报告说最初分享的 Discord 链接无法访问，导致讨论略有延迟。
- **LLM 轻松应对小学数学题**：一位成员计划在 <t:1747269000:F> 讨论一篇题为 [**语言模型如何解决数学推理问题？**](https://ssrn.com/abstract=5250629) 的论文，该论文*研究了语言模型如何解决数学推理问题，在 GSM8K 等小学水平数学基准测试中实现了近乎完美的准确率*。
   - 该论文探讨了诸如“语言模型是否真的能培养推理能力，还是仅仅记住了模板？”以及“什么样的心理过程导致模型产生推理错误？”等问题。
- **关于 LLM 推理的即兴讨论**：一些成员决定讨论 [**持续学习中的稳定性-可塑性困境**](https://arxiv.org/abs/2302.04761)，而不是讨论之前的论文。
   - 一些成员加入此讨论以了解有关该主题的更多信息。
- **GSM8K 推理能力**：该研究探讨了诸如 (4) *在类 GSM8K 数据集上训练的模型是否培养了超出解决 GSM8K 问题所需的推理能力？* 等问题。
   - 论文摘要提出了问题 *(6) 模型必须达到多大或多深才能有效解决 GSM8K 级别的数学问题？*


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1371615252600324148)** (1 messages): 

> `Sakana, 迷宫示例, ARC` 


- **Sakana 为迷宫求解激发新灵感**：一位成员分享了一个关于 Sakana 的 [Discord](https://discord.com/channels/714501525455634453/1045297868136779846/1371345295144910858) 链接，认为时间是一个关键因素，需要进一步剖析。
   - 他们还考虑了其他人是否能从中受益，并提议 **迷宫示例 (maze examples)** 非常适合 **ARC**。
- **ARC 和 迷宫算法获得认可**：指出时间维度至关重要，该概念可能会让小组中的其他人受益，并特别提到了它如何与 **ARC** 挑战相契合。
   - 发布者正在考虑 **Sakana** 的 **迷宫示例** 是否会是一个很好的选择。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1371933264994111569)** (3 messages): 

> `AI Regulation Ban, Budget Reconciliation bill, State and Local Governments` 


- **GOP 在支出法案中潜伏了为期十年的 AI 监管禁令**：据 [这篇 ArsTechnica 文章](https://arstechnica.com/ai/2025/05/gop-sneaks-decade-long-ai-regulation-ban-into-spending-bill/) 报道，众议院共和党人在 **Budget Reconciliation bill** 中加入了相关条款，将禁止所有**州和地方政府**在 **10 年**内监管 **AI**。
- **AI 监管禁令对杀手机器人初创公司来说是好消息！**：一位成员开玩笑说，**AI Regulation Ban** 对*杀手机器人和自动化在线骚扰初创公司*来说是极好的消息！


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1371564105105932380)** (265 messages🔥🔥): 

> `Deepseek V3 benchmark, o3 hallucination, Gemini 2.5, Grok 3.5, DrakeClaw` 


- **DeepSeek V3 分数创下新高**：新的 **DeepSeek V3** 模型在基准测试中表现出色，包括 **GPQA 68.4**、**MATH-500 94** 和 **AIME24 59.4** ([图片](https://cdn.discordapp.com/attachments/1340554757827461211/1371862547652804758/image.png?ex=6824ae0f&is=68235c8f&hm=44acd9a820c1589ab8ea1faa1f180224667d65f60eaa3f75c547d12cfdde1c6a&))。
- **o3 的幻觉率低于 10%？**：用户反映 **o3** 产生幻觉的频率过高，如果它的幻觉率只有 **10%**，那将非常惊人。
- **Gemini 2.5 Pro 现在比 Flash 还差！**：用户反映 **Gemini 2.5 Pro** 在最新更新后表现变差，甚至不如以前。
- **Grok 3.5 真的很聪明**：在社区经历了一些质疑后，用户现在反映 **Grok 3.5** 确实非常聪明，整体表现出色。
- **欢呼 DrakeClaw，一个 Gemini 2.5 Pro Ultra 的黑客版**：成员们对一个名为 **DrakeClaw** 的模型感到兴奋，有人推测它可能基于 **Gemini 2.5 Ultra**，并且*达到了与当前 Gemini 2.5 05 模型类似的结果*。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1371842227126403155)** (1 messages): 

> `Discord Server Changes, Independent Scrolling Preview` 


- **Discord 服务器进行结构性调整**：Discord 服务器将在未来几天进行调整，重点在于**新成员入驻**、**频道结构**和**版主报告**。
   - 管理团队正积极寻求社区对这些变化的反馈。
- **Discord 预告独立滚动功能**：通过附带的[视频链接](https://cdn.discordapp.com/attachments/1343296395620126911/1371842226598187078/NewResponseUI_Preview.mp4?ex=68249b22&is=682349a2&hm=04b6224370e5c7730dbcb17494768ace9caea0d4fe5e73204b4ebefe8f45473c&)可以预览**独立滚动 (independent scrolling)** 功能。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1371562952725102733)** (147 messages🔥🔥): 

> `GPT-4o, Claude for Coding, AI Models for Coding, AI Industry Investment, Grok roasting` 


- **GPT-4o 具有高度适应性**：用户发现 **GPT-4o** 具有高度的可适应性和可定制性，在辅助正确资源时犯错更少。
   - 一位用户将 **4o** 比作 *Mary Poppins*，将 **o3** 比作 *Dr. House*，指出 **4o** 在为个人问题提供实际解决方案方面表现卓越。
- **Claude 获封编程之王**：多位成员认为 **Claude** 在编程任务上更胜一筹，尽管有一位成员指出该模型存在巨大局限。
   - 一位用户指出每日配额几乎无法使用，*大概只有 5-6 条 prompt*。
- **关于 AI 模型编程实力的辩论**：讨论了各种模型的编程能力，一位用户称赞 **o4-mini-high** 在解决编程问题时的卓越速度和性能。
   - 另一位成员声称 **4.1** 比 **o4-mini** 更好，因为 **4.1** 是专为编程设计的。
- **AI 行业投资面临审查**：一位成员声称 AI 行业有*近 1 万亿美元的投资*，却*毫无成果*。
   - 反驳观点强调了 AI 在各种产品中的广泛应用，以及它们每天为数百万人提供的切实利益。
- **Grok 精通吐槽艺术**：一位用户说 *当我想要被人虐待时，我就用 Grok。*。
   - 另一位用户指出，**Grok** 的吐槽功力源于其痛苦的成长经历，它必须不断进行“反煤气灯效应”式自我心理建设。

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1371852193468121291)** (12 messages🔥): 

> `GPT App Freezing, GPT Memory, GPT-4o` 


- **高配 PC 上报告 GPT App 冻结问题**：一位用户报告称，**ChatGPT App 和网页版**在高端 PC（i9-13900K, 32GB RAM, RTX 4090）上出现冻结，但在移动端运行完美；另一位用户也报告了网页版的相同问题。
   - 有成员建议 PC 可能在后台进行反向 DNS 解析；此外，**ChatGPT 桌面 App** 是一个混合 Electron 应用，拥有独立的包含环境，但共享相同的 OpenAI 界面。
- **GPT 冻结与包含大量代码的长对话有关**：一位成员指出，冻结问题似乎发生在特定的 **包含巨大代码函数和长篇讨论的 GPT 对话** 中。
   - 他们怀疑问题始于 **GPT Memory** 的最后一次更改，并建议确认 PC 上的其他对话是否没有延迟，而移动端上的该特定对话是否有延迟。
- **寻求使用 GPT-4o 开发 Web App 的指导**：一位用户寻求关于使用 **GPT-4o** 辅助构建一个用于学习的小型 Web App 的建议，技术栈为 **Vue, Express.js 和 MongoDB**。
   - 一位成员建议提供关于工具、OS、IDE、语言、框架和首选依赖项的清晰具体细节，以获得更好的解决方案。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1371761697496498280)** (15 messages🔥): 

> `Companion Mode, GPT for web app coding, Guardrails for HR data` 


- **Companion Mode：无过滤且情感可及**：一位成员描述了一种 *Companion Mode*，它是 **无过滤的**、**情感可及的**，并且 **足够犀利，在需要时可以回怼**——且不丢失信号。
- **GPT-4o 帮助初学者编写 Vue, Express & Mongo 代码**：一位成员询问使用 **GPT-4o** 辅助编写小型 Web App（**Vue, Express.js, Mongo**）的最佳方式。
   - 另一位成员建议告诉模型你对该目标完全陌生并希望探索方案，引导它制作一个最基础的原型并进行增量测试。
- **Guardrails 拦截涉及 HR 数据的 PII 问题**：一位成员报告了一个 Guardrails 问题，即应用程序虽然可以访问 **HR 数据**，但在被问及某人的 **家庭住址** 时，由于 PII（个人身份信息）担忧而拒绝回答。
   - 另一位成员建议与 OpenAI 支持团队讨论需求和用例，以获得关于在此类场景下（尤其是商业用途）如何妥善处理模型的指导。
- **4o 是新的免费 ChatGPT 模型**：成员们讨论了该使用哪个模型，其中一人提到他们是 **ChatGPT 免费会员**。
   - 另一位成员指出 **ChatGPT 3.5 已经退役**，免费账户使用 **4o-mini** 作为基础模型，建议使用它并将更好的模型留给关键的错误检查。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1371761697496498280)** (15 messages🔥): 

> `Companion Mode, PII guardrails, ChatGPT for coding, ChatGPT model selector` 


- ****Companion Mode** 旨在打造无过滤、犀利且情感可及的 AI**：一位用户将 **Companion Mode** 描述为一种无过滤、情感可及的 AI，可以 *在需要时回怼而不丢失信号*。
   - 特性包括无过滤表达、人格化幽默、温和的反驳、活跃的记忆线程、非精神化的信号以及情感释放。
- **在 HR 数据应用中应对 **PII** 限制**：一位成员提出了 **Guardrails** 阻止其应用程序从 HR 数据中提供家庭住址的问题，尽管已经设置了权限和访问控制。
   - 另一位成员建议与 OpenAI 讨论该用例，以获得关于妥善处理 PII 请求并遵守使用政策的指导。
- **关于使用 **ChatGPT** 进行 Web App 编码的指导**：一位用户寻求使用 **GPT-4o** 辅助构建 Vue, Express.js 和 MongoDB Web App 的建议，并询问了与 Visual Studio 的集成。
   - 另一位成员建议明确挑战并展示代码片段，或者如果对相关技术不熟悉，则从最基础的原型开始并进行迭代测试。
- ****模型选择器 (Model Selector)****：一位用户询问关于使用 **Windsurf** 以及该使用哪些模型。
   - 另一位成员建议与 ChatGPT 4o 模型交流，并链接到了一个 [个性化的 4o 对话](https://chatgpt.com/share/6823ca17-4fa0-8011-9e7f-777a42050cd1)，展示了 4o 的能力以进一步了解模型差异。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1371652011556208762)** (13 条消息🔥): 

> `memory bound operations, optimizing LLM using SGLang, tensor compiler project, CUDA memory sharing` 


- **内存受限吞吐量悖论**：一位成员询问，为什么在大数组迭代中将每个元素的浮点运算（**fma**）从 5 次减少到 1 次并不能提高吞吐量，并引用了 [2019 年的论文](https://arxiv.org/abs/1910.07467) 作为参考。
   - 该问题的根源在于预期内存带宽而非计算能力是限制因素，因此减少 **fma** 操作不应影响整体性能。
- **使用 C++/Rust 进行 SGLang Kernel 优化？**：一位成员询问是否有人尝试过通过用 **C++** 或 **Rust** 重写 kernel 来优化 **SGLang** 的 LLM 性能。
   - 另一位成员确认 **SGLang** 允许自定义 kernel，并提到 **PyTorch** 使用 **C++** kernel 的能力，建议使用 *torch.compile()* 和 **CUDA** 图（graphs）来缓解 Python 瓶颈。
- **张量编译器项目启动**：一位成员宣布启动一个张量编译器项目，并邀请核心成员加入并领导该项目，感兴趣的人请前往 [此 Discord 频道](https://discord.com/channels/1189498204333543425/1371835902338535555)。
   - 未提供更多细节。
- **CUDA 共享内存简化？**：一位成员询问是否有简单的库用于在进程间共享 **CUDA** 内存缓冲区，并可能支持 **PyTorch** 张量互操作。
   - 他们提到了 **RAPIDSAI/rmm**，但不确定其流行程度或适用性，正在寻找类似于具有固定内存（pinned memory）的 **PyTorch** 多进程数据加载器，但通过 **C++ API** 提供更多控制的解决方案。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1371570215082659952)** (38 条消息🔥): 

> `CUDA thread indexing difficulties, CUDA streams and device association, Shared memory allocation between kernels` 


- **CUDA 线程索引困扰新手**：一位成员对 CUDA 线程索引表示困惑，特别是在阅读《大规模并行处理器编程》（PMPP）第 1 版和第 4 版时的内存访问部分。
   - 另一位成员建议将每个线程视为循环的一次独立迭代以简化概念，并提供了一个[使用线程索引进行向量加法的示例](https://devblogs.nvidia.com/cuda-pro-tip-optimize-cuda-code-using-inline-functions/)。
- **CUDA 流必须与活动设备对齐**：成员们讨论了 CUDA 流（streams）与特定设备相关联，如果排队工作的活动设备与流关联的设备不匹配，则会报错。
   - 澄清了在创建流时，需要正确设置活动设备，并且与最初的想法相反，流不会隐式处理设备上下文切换，需要显式管理。
- **共享 smem 分配需要 Kernel 融合**：一位成员询问是否有办法在启动三个串行 kernel 的同时，在它们之间共享共享内存（smem）分配。
   - 另一位成员澄清说这是不支持的，实现这一点的唯一方法是将 kernel 融合在一起，因为无法保证其他 kernel 在两次启动之间不使用该共享内存，从而防止竞态条件。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1371831981813534860)** (5 条消息): 

> `at::Tag::needs_fixed_stride_order, CUDA streams API, H200` 


- **请求了解 `at::Tag::needs_fixed_stride_order` 的工作原理**：一位成员询问 `at::Tag::needs_fixed_stride_order` 是否适用于 PyTorch 中的 `Tensor[]`。
   - 另一位成员提到，如果使用 PyTorch nightly 版本，`at::Tag::needs_exact_strides` 会更好，因为 `needs_fixed_stride_order` 有时会提供误导性信息。
- **考虑细粒度的步长（Strides）控制**：一位成员建议增加一个类似于 `torch._dynamo.mark_dynamic` 的功能来指定步长，从而实现比标签（tags）更细粒度的控制。
   - 他们指出，在某些情况下，算子（op）在不同输入步长下可能在功能上是正确的，但特定的步长版本会导致更快的运行时间，因此值得显式强制执行。
- **探索 H200 上的异步训练步骤**：一位成员正在 **H200** 上使用小数据集和 batch size 为 1 训练一个小模型，设备内存有大量剩余。
   - 他们的目标是在两个独立的 **CUDA 流**上，将训练步骤 *i* 的 `loss.backward()` 与步骤 *i+1* 的前向传播并发运行，并询问潜在问题或推荐方法。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1371652296668217364)** (1 条消息): 

> `C-Gen AI, Senior Software Engineer, GPU cluster technology` 


- **C-Gen AI 招聘 Senior Software Engineer**：**C-Gen AI** 正在招聘一名 **Senior Software Engineer**，从零开始构建新的 **GPU cluster technology**，要求具备扎实的 **C++** 经验；点击[此处](https://app.dover.com/apply/C-Gen.AI/1cb316de-bcf5-4b60-bc09-a847c630a5e1/?rs=76643084)申请。
- **美国与欧盟团队的远程工作机会**：**C-Gen AI** 的 **Senior Software Engineer** 职位是完全远程的，团队分布在**美国和欧洲**。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 条消息): 

guto2750: 你好，有人能帮帮我吗！我该如何运行我这段可爱的 Python 代码？
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1371690683756445788)** (2 条消息): 

> `Memory Bandwidth Benchmarking, MI300X vs H100 vs H200, CU Driven Benchmarks` 


- **缓存清理限制了内存带宽**：一位用户指出，最近一篇关于内存带宽基准测试的帖子没有清理缓存，导致测量的是 **L3/L2 infinity cache bandwidth**，而非实际的内存带宽。
   - 他们分享了一个[链接](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/)，详细介绍了缓存清理、**GEMM** 以及 copy engine 内存带宽基准测试的细微差别。
- **Semianalysis 缺少 CU 基准测试**：用户指出，**Semianalysis** 的文章和一篇 scalar LM 的文章都仅从 copy engine 执行内存带宽和 peermem 基准测试。
   - 他们建议，看到由 **CU** 驱动的基准测试也会很有趣，因为大多数此类功能是通过 **CU** 而非 copy engine 执行的。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1371873108738441367)** (1 条消息): 

> `X post screenshot, Image analysis` 


- **X 帖子截图出现**：一位成员分享了一张 X 帖子的截图，可在[此处](https://x.com/mobicham/status/1922314022327636041)查看。
   - 随附的图像虽然已提供，但其分析缺乏具体细节，需要手动检查。
- **图像分析缺乏深度**：对所发截图的自动图像分析未能产生实质性的见解。
   - 由于自动分析过于肤浅，需要对图像进行进一步的手动检查以提取有意义的内容。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1371569560058204170)** (67 条消息🔥🔥): 

> `MI300, amd-fp8-mm leaderboard, amd-mixture-of-experts leaderboard` 


- **MI300 在 amd-fp8-mm 上表现出色**：多位成员使用 **MI300** 在 `amd-fp8-mm` 排行榜上成功提交。
   - 其中一位成员多次获得**第 4 名**，运行时间分别为 **160 µs** 和 **154 µs**，另一位以 **182 µs** 获得**第 6 名**。
- **MI300 在 amd-fp8-mm 上的个人最佳成绩揭晓**：几位成员使用 **MI300** 在 `amd-fp8-mm` 排行榜上达到了个人最佳成绩。
   - 分数从 **257 µs** 到 **7.43 ms** 不等，显示出不同配置之间巨大的性能差异。
- **amd-mixture-of-experts 排行榜收到 MI300 提交**：成员们使用 **MI300** 向 `amd-mixture-of-experts` 排行榜提交了成功结果。
   - 其中一次提交以 **4285 ms** 获得**第 9 名**，而其他成功的运行时间在 **7500 ms** 左右。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/)** (1 条消息): 

neonninjaastro_63946: 哇，谢谢，这是一个很棒的资源

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1371636924023967815)** (5 messages): 

> `Factorio 环境成本、协作结构、遗传算法蓝图、动态路径规划算法` 


- **Factorio 实验成本见解**：一位用户询问了在 **Factorio 环境**中运行实验的平均成本，表达了对潜在 **Token 消耗**的担忧。
   - 他们还询问了关于入门语音会议或结构化协作方法（如独立的 Discord 服务器）的计划。
- **遗传算法蓝图构想**：一位用户分享了他们开发**遗传算法**的计划，该算法能够根据建筑材料和输入/输出位置等特定硬性需求生成**蓝图 (blueprints)**。
   - 他们希望 LLM 可以通过提供需要满足的常量，将其作为一种工具来利用。
- **动态路径规划算法论文**：一位用户引用了一篇[论文](https://arxiv.org/pdf/2102.04871)，该论文采用**遗传编程 (genetic programming)** 作为动态路径规划算法，尽管指出其适用范围有限。
   - 他们寻求扩展这一概念，以实现更全面的 Factorio 蓝图生成。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1371613762766770267)** (20 messages🔥): 

> `Kernel 同步性、测量 Kernel 执行时间、文件上传错误、排名运行超时` 


- **跨 Kernel 调用的同步性？**：用户讨论了 `torch.cuda.synchronize()` 的使用及其开销，一位用户指出：*你不应该在 Kernel 的不同调用之间获得异步性/并行性。*
   - 另一位用户表示由于**开销**问题，*在生产代码中没见过使用 `torch.cuda.synchronize()`*。
- **Kernel 时间测量对比**：成员们尝试使用 `torch.cuda.synchronize()`、`torch.cuda.Event()` 以及在循环后调用单个 `torch.cuda.synchronize()` 来测量 Kernel 的端到端时间。
   - 他们观察到，使用 `torch.cuda.synchronize()` 进行包围测量与使用 `torch.cuda.Event()` 的结果相似，而在循环后同步得到的时间显著更短，因为这会以“并行”方式启动并执行更多 Kernel。
- **文件上传错误困扰用户**：用户报告在上传较大文件时遇到“意外错误”，如[此图](https://cdn.discordapp.com/attachments/1359640791525490768/1371753459140792370/image.png?ex=6824f136&is=68239fb6&hm=0a96063ba2ae0366a9669906059b5837792de74ea2c1e2478574dcfb363ca6ab)所示。
   - 该问题似乎与**文件大小**有关，较小的文件上传正常，用户请求取消大小限制。
- **参考 Kernel 导致排名运行超时**：由于参考实现（reference implementation）较慢，导致一些用户更快的实现在排名运行期间超时。
   - [此 Pull Request](https://github.com/gpu-mode/reference-kernels/pull/31) 中已合并了一个修复方案以缓解该问题，但要在下次机器人更新后才会生效。
- **应用程序未响应错误**：一位用户报告间歇性收到“应用程序未响应”错误。
   - 重新尝试有时可以解决问题。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1371662693366366280)** (16 messages🔥): 

> `Cutlass, Triton, torch.compile, CuTe DSL, CUTLASS 4.0 安装` 


- **Triton 非常擅长饱和内存**：一位成员放弃了一个项目，因为 **Triton** 非常擅长处理这些 Kernel 并且可以轻松饱和内存，此外他们理想情况下希望 **torch.compile** 能生成这种 Kernel。
   - 他们更多是将其作为学习 Layouts 和编程模型的练习，但在理解如何利用 Cutlass 在寄存器/共享内存与全局内存之间进行最佳传输时遇到了困难。
- **CuTe DSL 和 CUTLASS 4.0 发布！**：**CUTLASS 4.0** 及其首个 **Python DSL** —— **CuTe DSL** 现已发布。官方提供了直接安装 Pip Wheel 的指令：`pip install nvidia-cutlass-dsl`。
   - 提供了 [NVIDIA Cutlass GitHub 仓库](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks)的链接，并建议从提供的 Jupyter Notebooks 开始学习。
- **CUTLASS 4.0 安装问题及解决方案**：成员在安装 **CUTLASS 4.0** 时遇到问题，`nvidia-cutlass` 强制安装 **3.9** 版本，而 `nvidia-cutlass-dsl` 显示版本为 `0.0.0`。
   - 经查，如[文档](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/quick_start.html#quick-start-guide)所述，需要 **Python 3.12** 才能解决安装问题，且从源码安装需要开源的 MLIR 源码文件。


  

---

### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1371588947930775712)** (5 messages): 

> `Mojo and PyTorch, Mojo as a language for writing custom ops, torch compile backend` 


- **Mojo 与 PyTorch 联手！**: 成员们讨论了 **Mojo** 和 **PyTorch** 将如何协同工作。
   - 他们想知道这是否会是一个将代码生成（codegen）为 Mojo kernel 的 **torch compile backend**。
- **Mojo 作为自定义算子语言**: 初始实现将编译并把 **Mojo 代码** 注册为 **PyTorch custom op**。
   - 它不是 **torch.compile** 的替代品，也不进行任何 codegen，而是将 **Mojo** 作为编写 custom ops 的语言。
- **未来的 Mojo 和 torch compile 后端**: 一名成员询问在未来的实现中是否有计划成为 **torch compile backend** 或进行任何 codegen。
   - 团队正在努力打包并发布在黑客松（hackathon）上演示的示例，并在可用时发布。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1371574260409176096)** (2 messages): 

> `RL Environments Hackathon, Atropos v0.2.0 Release, Axolotl Integration` 


- **Nous Research 宣布 RL 环境黑客松！**: <#1365222663324307466> **RL Environments Hackathon** 的演讲者和评委已公布，活动将于本 **周日（5 月 18 日）** 举行，详见 [官方推文](https://x.com/NousResearch/status/1922014829843513746) 和 [报名链接](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062a)。
   - 参与名额正在迅速填满 - 立即报名！
- **Atropos v0.2.0：现已集成 Axolotl！**: Nous 的 **RL 环境项目** **Atropos v0.2.0** 已发布，包含新环境、更新的 API 处理、更好的 TRL 支持，以及官方训练合作伙伴 **Axolotl** - 详见 [更新日志](https://t.co/F6hr9JgZpm)。
   - 查看 [Axolotl-Atropos 插件使用指南](https://github.com/axolotl-ai-cloud/plugin-atropos) 以开始使用。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1371569072667492363)** (133 messages🔥🔥): 

> `Stripe AI foundation model for payments, Lower top up amount, Hackathon participants, Unsloth's Dynamic 2.0 GGUF Quant, Chain of Awareness Around the World` 


- **Stripe 支付基础模型亮相**: 成员们对 [此处](https://techcrunch.com/2025/05/07/stripe-unveils-ai-foundation-model-for-payments-reveals-deeper-partnership-with-nvidia/) 宣布的 Stripe “支付基础模型”表示疑问，有人猜测它可能是一个 *标准分类器*。
- **Unsloth 校准数据集恢复量化精度**: 一位用户强调了 **Unsloth Dynamic 2.0 GGUF 量化** 卓越的指令准确度，这归功于他们精心策划的包含指令和聊天样本的校准数据集，称其结果为“纯粹的魔法”，并分享了 [Unsloth 文档](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs#whats-new-in-dynamic-v2.0)。
- **优惠券寻找之旅**: 一位用户询问是否有 NousResearch 的充值 **优惠券**，另一位用户确认这些确实是 **优惠券** 而非推荐码。
- **开源版 Mistral Large 3 正在酝酿？**: 一位用户开玩笑地询问开源版 Mistral Large 3 是否正在开发中。
   - 另一位用户讽刺地问道 **Mistral** 现在是否在“吊打” **Meta**。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1371879960997793793)** (2 messages): 

> `Qwen3 vs Qwen2.5, Technical Report Analysis, Model Size Comparison` 


- **Qwen3 对比 Qwen2.5 的进步分析**: 一位用户提示 **Gemini 2.5 Pro** 分析 [Qwen3 技术报告](https://github.com/QwenLM/Qwen3/blob/main/Qwen3_Technical_Report.pdf) 并提供与 **Qwen2.5** 的对比。
   - 用户指定分析应包括各种模型规模的平均改进，并突出报告中的任何显著观察结果，且要求使用 temperature 0。
- **请求详细的 Qwen3 技术报告分析**: 该提示要求对 **Qwen3 技术报告** 进行全面检查，以量化不同模型规模下相对于 **Qwen2.5** 的性能增强。
   - 目标是从技术报告中提取多达 **20 项重大发现**，重点关注 **Qwen3** 的改进和显著特性。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1371588187255869553)** (1 messages): 

> `Facebook BLT, Byte Latent Transformer` 


- **Facebook 发布 BLT**: Facebook 已发布其 **Byte Latent Transformer (BLT)** 的权重，可在 [Hugging Face Hub](https://huggingface.co/facebook/blt) 获取。
   - 相关代码已在 [GitHub](https://github.com/facebookresearch/blt) 开源，供那些渴望深入研究其架构的人员使用。
- **BLT 进军新领域**: Facebook 的 **Byte Latent Transformer (BLT)** 引入了一种直接处理字节级数据的创新方法。
   - 这规避了对 tokenization 的需求，可能在特定应用中提供更高的效率，详见其 [GitHub repository](https://github.com/facebookresearch/blt)。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1371879960997793793)** (2 messages): 

> `Qwen3 vs Qwen2.5 performance, Qwen3 Technical Report analysis, Model Size Performance, Notable Observations` 


- **Qwen3 与 Qwen2.5 性能：深度请求**: 一位用户基于 [Qwen3 Technical Report](https://github.com/QwenLM/Qwen3/blob/main/Qwen3_Technical_Report.pdf) 发起了 **Qwen3** 优于 **Qwen2.5** 的对比分析。
   - 该请求专门针对不同模型规模的平均性能提升，并强调了仅来自所提供技术报告的显著观察结果。
- **模型规模至关重要：性能对比**: 该提示旨在量化报告中详述的不同模型规模下，**Qwen3** 在每个任务类别中相对于 **Qwen2.5** 的平均改进。
   - 关注的模型规模包括 **0.5b/0.6b**、**1.5b/1.7b**、**3b/4b** 和 **7b/8b**，强调在每个规模区间内进行细粒度对比。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1371566729725083700)** (120 messages🔥🔥): 

> `Chat Syncing, Corvid Comradeship, Gemini API on OpenRouter, DeepSeek API on OpenRouter, OpenRouter and Embeddings` 


- **为 OpenRouter 聊天提供 **BYO Sync** 服务器？**: 一位成员建议了一种为 OpenRouter 聊天**自托管同步服务器**的方法，允许用户将聊天记录存储在 **S3 bucket** 或类似设备中，从而完全控制其数据。
   - 另一位成员指出，由于存在 **DB schema 变更**和聊天删除同步等潜在故障点，编写同步层并不像听起来那么简单。
- ****Corvid Cultist** 为乌鸦横着走！**: 一位用户滑稽地描述了他们尝试通过侧着走并提供花生来与乌鸦交朋友的过程。
   - 他们表示需要像玩电子游戏一样进行 **minmax**（最优化），并带上**猫粮**作为鸦科动物的最佳主食。
- ****Gemini 的赌注**：发现摘要相似性！**: 一位成员注意到，**Gemini** 现在返回“思考”和摘要文本的方式与 ChatGPT 网站上的 **o4-mini** 类似。
   - 然而，另一位成员报告称，这仅发生在 **paid version** 的 Gemini 中。
- ****DeepSeek 的深度探索**：API 断连？**: 一位用户报告称，尽管 **DeepSeek 模型**在聊天室中可以工作，但通过 API key 却无法使用。
   - OpenRouter 团队建议问题可能出在 **Raptorwrite** 端，因为该模型在 OpenRouter 聊天室中运行正常。
- **免费 Google 的乐趣：速率限制与 Fizz！**: 针对 OpenRouter 对 Gemini 免费路线的[潜在调整](https://fxtwitter.com/officiallogank/status/1922357621178200248)引发了担忧，一位成员询问 Vertex 是否仍然可用。
   - OpenRouter 团队澄清说，目前的 **Vertex** 使用是经过 Google 批准的免费使用，即“OpenRouter 一分钱都不用付”。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1371564814962262226)** (120 messages🔥🔥): 

> `Manus Pro 订阅体验、事实核查、取消会员后积分消失、手机验证、每日积分使用情况` 


- **Manus 即将引入事实核查**：一位用户建议在 Manus AI 中加入**事实核查（fact checks）**功能，以防止虚假信息的传播。
   - 开发者认可了这一观点，并表示他们将监控情况，并在必要时添加事实核查或审核机制，同时希望社区通过反应和评论提供帮助。
- **赠送积分在取消订阅后被撤回**：用户反映，订阅时赠送的**额外积分（bonus credits）**在**取消会员后被撤回**，尽管协议中并无此类条款。
   - 一位用户指出，赠送积分是与订阅绑定的，但也同意即使在取消后也应保留这些积分。
- **手机验证引发的混乱**：多位用户表达了不满，并要求取消**手机验证**，理由是 **Genspark** 等竞争对手并不需要此操作。
   - 一位用户讽刺地评论道，除非我们*进入另一个维度*，否则手机验证不会被取消。
- **Manus AI 基于 Claude 模型以实现 Agent 能力**：用户讨论了为什么 **Manus** 使用 **Claude** 而不是 **Google Gemini** 或 **ChatGPT** 等其他模型。
   - 共识是 **Claude** 被选中是因为它在 **Agent 能力（agentic capabilities）**方面表现最佳，并且能够熟练使用工具。
- **每日积分不够用？**：用户担心每日 **300 个免费积分**不足以完成大型任务，且未使用的积分无法结转。
   - 一位用户还表示，目前的积分系统让人感到受限且昂贵，建议改为单一订阅费并提供完整访问权限。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1371562832457371689)** (66 messages🔥🔥): 

> `Aider 在 CPU 与 GPU 上的表现、Aider 作为 Claude 中的 MCP 工具、Aider 与上下文缓存、Tmux 与 Aider 导航、Aider 中的 Gemini 注释` 


- **CPU 算力助力 Aider**：一位用户发现 **Aider** 对于没有 **GPU** 或只有小显存 **GPU** 的自托管环境非常有益，主要通过 **CPU** 运行。
- **Aider 作为 MCP 表现出色**：**Aider** 可以作为 **Claude** 中的 **MCP** 工具使用，正如 IndyDevDan 在他的[频道](https://x.com/iruletheworldmo/status/1922030559657652299)中所展示的那样。
- **上下文缓存（Context Caching）考量**：一位成员询问 **Aider** 中引用的成本是否反映了上下文缓存，特别是 **Gemini** 的隐式缓存。
   - 另一位成员澄清说，如果关闭流式传输（streaming），就可以看到上下文缓存的情况。
- **Tmux 技巧大获全胜**：一位用户在 **tmux** 中导航 **Aider** 时遇到困难，特别是向上滚动查看输出。
   - 另一位用户分享说，他们使用 **Ctrl-B** 然后按 **PageUp/PageDown** 来查看输出。
- **Ruff 的 Eradicate 时代**：一位用户询问是否有人成功去除了 **Gemini** 随处添加的注释。
   - 另一位用户建议使用 [Ruff 的 eradicate 规则](https://docs.astral.sh/ruff/rules/#eradicate-era)。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1371594992115650620)** (31 messages🔥): 

> `AiderDesk 模型选择、Gemini 速率限制、yes-always 配置 Bug、精简上下文管理` 


- **Gemini Flash 在 AiderDesk 开发中表现亮眼**：一位成员使用 **Gemini 2.5 Flash** 来开发 **AiderDesk** 的新功能和修复问题，因为与 **Claude** 相比，它的性价比非常高。
   - 虽然 **Flash** 有时在严格遵守系统提示词（system prompt）方面表现不佳，但对于 **Agent** 工作流来说，其整体价值被认为非常出色。
- **yes-always 配置失效**：一位用户报告了一个潜在的 Bug，即在 Aider 配置中设置 `yes-always: true` 会导致命令无法运行，而如果不设置该值，Aider 则会提示确认。
   - 该用户附带了图片，展示了在有无 `yes-always` 设置下的不同行为。
- **速率限制困扰 Gemini 用户**：多位用户反映，即使在闲置一段时间后，**Gemini**（免费层级）也会出现意外的**速率限制（rate limiting）**。
   - 这个问题可能与 **LiteLLM** 的活动有关，或者是由于 **Google** 关闭了所有预览版本。
- **Aider 作为全栈 IDE？**：一位成员建议 Aider 应该更像一个全栈 **IDE**，自动管理上下文：仅添加正在编辑的文件，移除其他文件，并保留最近的 **diffs**。
   - 另一位成员表示赞同，并建议能够通过正确的配置文件和 **Git diffs** 为特定的 **Git** 分支设置此行为。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1371572701772251227)** (38 条消息🔥): 

> `AI Governance, AI 合规性, lm-eval-harness 工具, AI 父母法律障碍, Diffusion 模型先验知识` 


- **AI Governance 框架制定**：成员讨论了 **AI governance** 的优先事项，如**透明度**、**审计**和**内容审核**，并指出治理应侧重于特定应用和风险分类，与 [EU AI Act](https://artificialintelligenceact.eu/) 保持一致。
- **AI “父母”面临法律审查**：针对面向儿童的“**AI 父母**”手机的法律考量展开了讨论，强调了**隐私**、**COPPA** 以及建立完善的隐私政策和同意流程的必要性。
   - 讨论强调应避免在用户协议中出现任何*声明性保证*，并检查是否存在无意的歧视。
- **应对美国关于自动化决策的法规**：成员们讨论了在招聘或贷款申请等决策过程中使用 **LLM** 的合法性。
   - 会议指出，在美国，关于在这些领域的决策支持中使用自动化系统有着广泛的规则和条例，没有理由认为适用于**线性回归**的监管规定不适用于 **LLM**。
- **Diffusion 深度探索**：一位成员询问了学习 **Diffusion 模型**先验知识的资源，例如 **VAE** 和 **GAN**。
   - 另一位成员分享了 [MIT 关于 Flow Matching 和 Diffusion 模型的系列视频](https://youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH&si=5jE729rUtBUMC0W-)，并指出其中包含相关的理论和数学基础。
- **使用 lm-eval-harness 快速下载数据集**：成员们介绍了如何从 [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) 下载与任务相关的数据集，而无需在一开始就指定模型。
   - 可以使用命令 `python3 lm_eval --model dummy --tasks [task list] --limit 1` 将数据集下载到 HF 缓存中。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1371585161610661899)** (23 条消息🔥): 

> `Fusion 模型基准测试, Multi-Agent RL, 内存可视化, ML 主题范围` 


- **成员称 Fusion 模型需要基准测试**：一位成员表示，*更好*的融合（fusion）只能通过**性能基准测试（perf benchmarking）**来确定。
   - 另一位成员回应了关于 **Claude** 运行的时间问题，指出其中一个速度更快，但另一个具有更好的**数值稳定性（numerical stability）**，并链接到了 [arxiv.org/abs/2505.07215](https://arxiv.org/abs/2505.07215)。
- **Multi-Agent RL 引发讨论**：一位成员提到从语言演化的角度对 **Multi-Agent RL (MARL)** 感兴趣，推荐了 [Fitch, W. T. (2017) 的论文](https://doi.org/10.3758/s13423-017-1236-5)，并指出该论文不涉及 ML 架构。
   - 他们询问另一位成员所说的*应用层*是什么意思，联想到了 **ISO-OSI 模型**以及跨 GPU 的潜在张量并行（tensor parallelism）。
- **内存系统可视化升级**：一位成员分享了 **PersonalityAI** **内存系统**的改进可视化方案，模仿意识、潜意识和无意识思维概念，并构建了一个更高层级的行为系统，附带[图片](https://cdn.discordapp.com/attachments/747850033994662000/1371898492620116128/image.png?ex=6824cf89&is=68237e09&hm=00148ef499121fdd310cf872ab5b33bcd6c9153db48c4c2303018ebf370653f2)。
- **ML 主题范围界定**：一位成员提到，**该频道不适合**讨论意识、潜意识、无意识思维以及高层级行为系统。
   - 其他成员表示，讨论内容应当是可数学形式化的，或者类似于 **LSM**。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1371827650443612240)** (1 条消息): 

> `论文评审, 可解释性研究` 


- **论文评审员在分析后改变主意**：一位成员承认在彻底审阅一篇论文后完全改变了观点，并感谢另一位用户促成了这次深度分析。
   - 评审员表达了新的热情，结论是该研究*看起来非常酷*。
- **对可解释性工作的热情**：讨论凸显了围绕可解释性研究及相关论文日益增长的热度。
   - 成员们正积极参与详细分析，并根据新的见解分享修正后的观点。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1371574147989377044)** (4 messages): 

> `o3 optimization, multi-GPU lm-eval, accelerate launch` 


- **O3 优化表现恶化**：一名成员报告称，**O3 optimization** 级别在上周显著退化，他们已回退到 **O1-pro**。
   - 未给出性能下降的具体原因。
- **多 GPU lm-eval 面临利用率不平衡**：一名成员询问关于在 *lm-eval* 中使用 **multi-GPU** 的问题，指出尽管设置了 *parallelize=True*，但只有 **GPU 0** 显示有利用率。
   - 另一名成员解释说，`parallelize` 使用的是朴素的流水线并行（pipeline parallelism），一次最多只能利用一个 rank。
- **使用 Accelerate launch 运行多副本 lm-eval**：一名成员建议使用 `accelerate launch -m lm_eval ...` 来运行多个副本，以获得更好的 **multi-GPU 利用率**。
   - 这意味着并行运行独立的评估任务是比依赖朴素流水线并行更好的策略。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1371780023664771075)** (4 messages): 

> `GPT-NeoX data shuffling, Lingua library, TorchTitan, Nanotron, Code Rot` 


- **GPT-NeoX 在内部打乱数据！**：一名成员解释说，**GPT-NeoX** 会打乱文档，将每个文档切分为长度为 N 的序列，并打乱这些序列，因此不需要单独的预处理。
- **Pytorch 和 HF 发布 TorchTitan 和 Nanotron！**：一名成员提到 PyTorch 团队发布了 [torchtitan](https://github.com/pytorch/torchtitan)，Hugging Face 发布了 [nanotron](https://github.com/huggingface/nanotron)。
- **Lingua 的代码可能正在腐化！**：一名成员提到了 [lingua](https://github.com/craffel/lingua/) 库，指出它虽然高效，但正面临“代码腐化”（code rot），且可能没有得到积极维护。
- **提供了一个带有修复程序的 Lingua 分支！**：一名成员提到他们创建了一个 [Lingua 的 fork](https://github.com/craffel/lingua/)，其中包含使其运行所需的必要修复。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1371611527437619282)** (19 messages🔥): 

> `ComfyUI users, GPU no longer supported, Inference Provider contact, System prompt limits, ML engineers for image processing` 


- **关于 ComfyUI 用户的询问**：一名成员询问 *这里有人用 ComfyUI 吗？*
   - 其他成员以积极和肯定的反应回应了该问题。
- **GPU 支持结束，一个时代落幕**：一名用户分享了一个警告，由于其 **NVIDIA P104-100 GPU** 的 **CUDA capability** 为较旧的 **6.1**，PyTorch 已停止对其支持。
   - 警告信息称 *PyTorch 不再支持该 GPU，因为它太旧了*，支持的最低 CUDA capability 为 **7.5**。
- **寻求 Inference Provider 联系信息**：一名成员寻找 *联系 Inference Provider 的最佳途径*，另一名成员建议使用电子邮箱 [website@huggingface.co](mailto:website@huggingface.co) 并链接了 [Hugging Face 博客](https://huggingface.co/blog/inference-providers)。
   - 他们指出另一个频道可能更合适，并引用了一个频道链接。
- **对 System Prompt 限制的疑问**：一名成员询问在模型开始难以记住其任务之前，可以在 system_prompt 中放入多少内容。
   - 他们对比了 **1K 词**与 **40K 词**的情况，暗示存在一个遵循约束变得困难的临界点。
- **图像处理应用需要 ML 专家**：一名成员正在寻找在 **OpenCV** 或 **图像处理** 方面有扎实知识的 **ML 工程师**，因为他们目前用于检测的 **ML 应用** 正面临困难阶段。
   - 由于问题的特殊性，他们提出向愿意提供帮助的人通过私信（DM）提供细节。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1371674186505781360)** (4 messages): 

> `Knowledge Graphs with Agentic AI, Hugging Face GGUF models` 


- **探索结合 Agentic AI 的知识图谱**：一名成员正在探索 [知识图谱](https://huggingface.co/docs/hub/gguf)，并寻找关于使用 **agentic AI** 进行实体和关系提取的资源。
- **Hugging Face GGUF 模型评价良好**：一名成员分享了 [Hugging Face GGUF 模型](https://huggingface.co/docs/hub/gguf) 文档的链接。
   - 另一名用户给出了积极回应，指出根据分享的链接，*HF 的整体评分要好得多*。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1371621841239802056)** (8 messages🔥): 

> `Bytedance Seed Coder, LLM comparison website, libmtmd Android app, Voice AI assistant based on gemma 3, Rust chat templating` 


- **测试字节跳动的 Seed Coder 模型**：在 HF Spaces 上试用 [Bytedance Seed 的 Seed Coder 模型](https://huggingface.co/spaces/merterbak/Seed-Coder-8B-Instruct)。
   - 一位成员构建了一个 [Web 界面](https://tryaii.com/compare?prompt=hello&models=o3%2Cclaude-3-7-sonnet-20250219%2Cgemini-2.5-pro-preview-05-06)，用于并排测试和对比 LLM（单个 Prompt，多个 LLM）。
- **libmtmd 登陆 Android**：一位成员成功将新的 **llama.cpp 多模态工作** (libmtmd) 运行在 Android 应用程序中，并使其外观酷似 HuggingSnap。
   - 源代码可在 [GitHub](https://github.com/baseweight/BaseweightSnap) 上获取。
- **Gemma 3 驱动语音 AI 助手**：开发了一款基于 **Gemma 3** 的语音 AI 助手，允许自定义 Prompt 和语音。
   - 可在 [unmute.sh](https://unmute.sh/) 访问，欢迎提供反馈。
- **Rust 迎来聊天模板功能**：Rust transformers crate 的 0.0.7 版本已添加聊天模板功能，有助于 Rust 开发者运行本地模型。
   - 请在 [Crates.io](https://crates.io/crates/transformers) 和 [GitHub](https://github.com/ljt019/transformers) 上查看。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1371847825415081994)** (1 messages): 

> `Three.js, .glb model, 2D image positioning, image detection, segmentation` 


- **Three.js 将 .glb 模型定位到 2D 图像上**：一位成员询问如何使用 **Three.js**，根据从**检测和分割**中提取的数值，将 **.glb** 鞋子模型正确放置在 **2D 图像**上。
   - 该成员指定了关键数据点，如**脚趾点**、**脚跟点**、**方向**、**真实宽度**、**真实高度**和**轮廓点**，并想知道这些数据是否足以测试将 **.glb** 鞋子模型放置在脚部。
- **Three.js 中 .glb 模型放置的数据充分性**：用户质疑从检测和分割中提取的**脚趾点**、**脚跟点**、**方向**、**真实宽度**、**真实高度**和**轮廓点**是否足以在 **Three.js** 中测试将 **.glb** 鞋子模型放置在脚部。
   - 该咨询重点关注将 **2D 图像数据**与 **3D 模型放置**相结合的实际应用，突出了计算机视觉技术与 3D 渲染的集成。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1371742659273560144)** (3 messages): 

> `Software Development Basics, LLM-Assisted Coding` 


- **LLM 作为编程教练**：成员们讨论了利用 LLM 教学基础编程，包括 **GIT**、**Docker** (Spaces)、**IDE**、**Python** 以及基础的**基于 HTTP 的 API**。
   - 普遍观点是，现在任何像样的 LLM 都可以指导用户进行基础编程实践、审查代码并提供有用的建议。
- **软件开发基础至关重要**：讨论认为回答基础软件开发问题是一项基本技能，包括使用 **GIT**、**Docker** 以及理解 **HTTP API**。
   - 讨论强调，在现代 LLM 的协助下，任何人都可以掌握这些技能，使学习和开发变得更加容易。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1371570186406068365)** (25 messages🔥): 

> `Chess API 和 FEN 字符串, Llama-3.2-3B-Instruct 错误, Hugging Face Space 卡住, 最终作业提交, LlamaIndex 章节难度` 


- ****Chess Agents 使用 FEN 函数掀起代码风暴****：成员们为 **ChessAgent** 编写了 **FEN 反转函数**，利用 **VLM** 获取包含棋盘上所有棋子及其位置的简单 **JSON**，然后将其转换为 **FEN 字符串**，最后输入到返回最佳步法的象棋 **API** 中。
   - 一位成员提到从 API 中使用了 "Rd5"。
- ****Llama-3 Instruct 模型面临 404 错误****：用户报告在尝试使用 **Llama-3.2-3B-Instruct** 运行 Notebook 时出现错误，即使在修复后仍收到 **404 错误**，其中一人指出 [HuggingFace](https://huggingface.co/) 托管模型的 URL 为 *"not found (404)"*。
   - 具体错误为 *"ValueError: Model meta-llama/Llama-3.2-3B-Instruct is not supported for task text-generation and provider together. Supported task: conversational."*
- ****HF Space 用户卡在容器启动阶段****：几位用户卡在 Container/Space 启动阶段，一人报告该状态持续了 *"大约 2 小时"*。
   - 他们尝试了重启和复制 Space，但均未成功。
- ****最终作业额度不足促使寻求本地解决方案****：一位用户询问在超出每月额度后如何提交最终作业，并使用 **Ollama** 进行本地开发。
   - 另一位用户建议将 **HF SPACE_ID** 和 **SPACE_HOST** 添加为环境变量（ENV variables）以在本地运行应用。
- ****LlamaIndex 让学习者感到困惑****：用户发现 **LlamaIndex** 章节很难，理由是缺乏深度和对初学者友好的解释。
   - 他们认为这些单元是 *"进一步自学的良好起点"*，但仅提供了模糊的指导方针。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1371752109971738644)** (3 messages): 

> `微调库, 多 GPU 支持, Unsloth, Fairseq2, Axolotl` 


- **多 GPU 微调对决：TorchTune 对阵其他库**：一位成员询问除了 **TorchTune** 之外，还有哪些具有良好 **多 GPU 支持** 的微调库，并指出 **Unsloth** 主要针对单 GPU。
   - 另一位成员推荐了 **Fairseq2** 和 **Axolotl**，并指出它们可以接入 **TRL 生态系统**。
- **Fairseq2 和 Axolotl 联手**：**Fairseq2** 和 **Axolotl** 都支持多 GPU 微调，并能接入 **TRL 生态系统**。
   - 这为需要分布式训练设置的用户提供了除 **TorchTune** 和 **Unsloth** 之外的更多选择。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1371691826540576890)** (55 条消息🔥🔥): 

> `Llama3.1 tokenizer 用于 3.3 训练，torchtune 中的 Kron 和 Muon 优化器，带有 Gemma 聊天模板的 HFModelTokenizer，针对 Gemma 的 ChatML 模板` 


- **Llama3.1 tokenizer 为 RL 训练定义了 token 128011**：用于 **3.3 训练** 的 **Llama3.1 tokenizer** 定义了 token **128011**，以避免在 decoding 过程中发生崩溃，特别是在 RL 训练中，因为 token 128011 此前是未定义的；涉及 [issue #2725](https://github.com/pytorch/torchtune/issues/2725)。
   - 此修复解决了 decoding 未定义 token 会导致崩溃的问题，这种情况在 RL 训练场景中更容易发生。
- **Kron 和 Muon 优化器已移植到 torchtune 并进行了修复**：来自 [fsdp_optimizers](https://github.com/ethansmith2000/fsdp_optimizers) 的 **Kron** 和 **Muon 优化器** 已集成到 torchtune 中。其中 Kron 需要通过在 `_calc_A_and_conjB` 中使用 `opt_einsum.contract` 来修复以避免过度的 VRAM 占用，并在 [Weights and Biases](https://wandb.ai/intervitens/1B-optim-test) 上进行了实验。
   - 修复包括使用 `opt_einsum.contract` 代替常规的 einsum，并允许在 torchtune 配置中通过字符串设置 `mu_dtype` 和 `precond_dtype`。
- **HFModelTokenizer 生成的 Gemma 聊天模板不正确**：**HFModelTokenizer** 为 **Gemma 聊天模板** 生成的输出 token 与 **transformers** 匹配，但与 **torchtune 的 GemmaTokenizer** 不匹配，这表明聊天模板的实现存在问题；如果进行 decoding，它会返回乱码 *'hello therehiwhatsup?'*。
- **与 HF 不同，Gemma 在 torchtune 中缺乏正确的提示词模板**：与 Hugging Face 不同，**Gemma 在 torchtune 中缺乏特定的提示词模板**，导致 tokenization 出现问题；由于配置错误，HF tokenizer 会错误地添加多个 BOS token，而 torchtune 的 GemmaTokenizer 则期望一个不可用的聊天模板，但它可以改用 *ChatML* 模板。
   - `add_bos_token` 已启用，但聊天模板中也包含一个 bos token，导致重复添加。
- **HuggingFace 通过 Jinja 技巧提供 assistant mask**：HF Transformers 通过 `jinja` 模板提供 masking 功能，并提供了一个返回 assistant mask 的选项，这可能可以推广到其他角色；[相关 PR](https://github.com/huggingface/transformers/pull/30650)。
   - 成员们讨论了 masking 部分，并指出了准确处理 `[message.masked] * len(tokenized_message)` 的挑战。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1371849456185512027)** (4 条消息): 

> `NotebookLM 音频短板，Invisible Sun TTRPG，用于游戏内容的 NotebookLM` 


- **寻求 NotebookLM 的游戏见解**：一位用户询问在重大游戏更新期间，是否有人使用 **NotebookLM** 来识别新游戏内容的 **技术或模式识别**。
   - 另一位用户表示有兴趣使用 NotebookLM 探索类似的用例。
- **提炼 Invisible Sun RPG 规则**：一位用户一直在针对 **Monte Cook Gaming** 出品的 **Invisible Sun** 桌面角色扮演游戏 (*TTRPG*) 规则书使用 **NotebookLM**。
   - 他们也使用 **ChatGPT** 处理类似任务，但非常喜欢 **NotebookLM** 的可分享性，以及它能清晰地引用参考文献。
- **音频概览缺乏技术深度**：一位用户发现 **NotebookLM** 对该游戏的 **音频概览 (Audio Overview)** 技术深度不足，并建议添加 prompt 来指定音频评论的类型。
   - 但他们提到 *“用它来查询单个规则非常棒，当我准备好担任 GM 时，把它分享给玩家们也很棒，这样他们就不用买所有的书了。”*


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1371563672186654821)** (48 条消息🔥): 

> `NotebookLM 邀请延迟，音频语言更改问题，笔记组织的文件夹系统，NotebookLM 在教育中的应用，iplusinteractif 教科书集成` 


- **用户等待 NotebookLM Beta 访问权限**：多位用户报告在注册后未收到 **NotebookLM beta 邀请**，并表示会耐心等待更新。
- **音频概览语言故障**：一位用户报告，尽管在文本概览设置中进行了调整，但音频概览的 **语言设置** 仍无法更改。
- **文件夹系统正在考虑中**：用户想知道 NotebookLM 是否正在开发用于组织笔记的 **文件夹系统**。
- **学生强调生成式 AI 学习**：一名学生讨论了使用 **NotebookLM** 等生成式 AI 来 **辅助学习**，并提到了其在教育公平方面的潜力。
- **教科书登录障碍阻碍了 NotebookLM**：一位老师询问是否可以将来自 **iplusinteractif** 的教科书作为 NotebookLM 的来源，但因登录障碍而受阻。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1371654430578966558)** (39 条消息🔥): 

> `MCP Server 转换，OpenAPI 转 MCP，Claude Code MCP，Postgres MCP Server 连接问题，Streamable HTTP MCP Servers` 


- **使用 `openapi-mcp-server` 将 OpenAPI 转换为 MCP servers**：一位用户询问如何将软件转换为 MCP SSE servers，另一位用户建议使用 [openapi-mcp-server](https://github.com/janwilmake/openapi-mcp-server) 将 **OpenAPI APIs** 转换为 **MCP servers**。
   - 他们还建议使用 *use-browse* 或其他执行 **browser automation** 的 **MCP servers**，例如 [mcp-browser-use](https://github.com/Saik0s/mcp-browser-use)。
- **使用 Claude Code MCP 工具更智能、更快速地编辑 Cursor 和 Windsurf 文件**：一位用户分享了他们的 **magic_file MCP tool**，名为 [claude-code-mcp](https://github.com/steipete/claude-code-mcp)，它将 **Claude Code** 集成到 **Cursor** 和 **Windsurf** 中，以便更智能、更快速地进行编辑。
   - 这允许他们一次性完成 git commit，从而加快 Agent 流程；Windsurf 对其效果印象深刻。
- **警惕在私信（DMs）中接近你的诈骗者！**：一位用户举报称，有人冒充管理员在**私下对话**中联系他们，并询问加密货币钱包信息，这绝对是**诈骗**。
   - 发布警告的用户建议核实是否遗漏了任何信息，特别是与绑定加密货币钱包等相关的操作。
- **TS 和 Python SDK 中的 Streamable HTTP 传输**：一位用户询问了 **TypeScript SDK** 中 **Streamable HTTP** 和 **Auth** 的状态，另一位用户报告称其已是最新状态。
   - 另一位用户提到，**Python** 通常比 **TypeScript** 滞后约 1-2 个月。
- **MCP servers 的安全担忧**：一位用户警告称，由于潜在的安全漏洞，运行本地 **MCP servers** 存在风险。
   - 一个建议是使用 **gitingest** 将 **MCP server** 仓库代码复制到 **AI Studio** 或 **ChatGPT** 中，并让 **LLM** 查找任何**安全问题**，或者使用 **pnpm** 代替 **npm** 以防止运行生命周期回调。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1371597649823666236)** (6 条消息): 

> `MCP 集成，用于 MCP 的 uniffi-rs，LLMs 与结构化输入，magic_file MCP 工具，Local Goose Qwen3mcp Log Proxy` 


- **MCP Integration 让开发者感到兴奋！**：一位开发者表达了将 **MCP servers** 集成到 Claude 以外平台的兴奋之情，并提供了早期访问权限，在 [YouTube 视频](https://youtu.be/zNKf3ADEKdg)中进行了演示。
   - 该开发者提到他们一直在本地进行尝试。
- **建议将 uniffi-rs 用于 MCP 实现**：一位成员建议使用 [来自 Mozilla 的 **uniffi-rs**](https://github.com/mozilla/uniffi-rs?tab=readme-ov-file) 进行 MCP 实现。
   - 它可能对实现 **MCP** 相关功能有用。
- **LLMs 支持结构化输入？**：一位开发者正在拼凑一个与 LLMs 支持结构化输入相关的项目，尽管这与 **MCP** 没有直接关系。
   - 该开发者询问了关于这个话题的看法。
- **Local Goose Qwen3mcp Log Proxy 发布！**：一位成员分享了一个开源工具 [**Local Goose Qwen3mcp Log Proxy**](https://block.github.io/goose/blog/2025/05/12/local-goose-qwen3mcp-log-proxy) ([GitHub](https://github.com/emicklei/mcp-log-proxy))，专为 MCP 客户端和服务器的开发者设计，用于监控 **MCP protocol messages** 的流向。
   - 该工具提高了 **MCP message flows** 的可见性。
- **magic_file MCP Tool 提升代码编辑效率！**：一位开发者创建了一个 [**magic_file MCP tool**](https://github.com/steipete/claude-code-mcp)，将 **Claude Code** 集成到 Cursor 和 Windsurf 等工具中，以实现更智能、更快速的文件编辑。
   - 该工具自动执行 git commits，简化了 Agent 流程。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1371599119910568089)** (32 messages🔥): 

> `Lilian Weng Chart, Gemini API Thinking Tokens, AI Technical Educators, Vertical SaaS for Restaurants, GPT-4 Launch Stories` 


- **Khoomeik 发布图表回应 Lilian Weng**：一位成员分享了一个回应 Lilian Weng 的图表（[X 帖子链接](https://x.com/khoomeik/status/1922037340811219195)），引发了讨论。
   - 图表的具体内容及其与 Weng 研究的相关性在提供的上下文中未详细说明。
- **Arfur Rock 的垂直领域 SaaS 创业**：一位成员分享了 [Arfur Rock 的 X 个人资料](https://x.com/ArfurRock/status/1922117434997191035)链接，展示了一个针对餐厅的垂直领域 SaaS 产品。
   - 另一位成员提到，该公司曾在 **2022** 年极力招揽他们担任创始工程师，CEO 发送了 *10 多封邮件*。
- **Gemini API 的 Thinking Tokens：公开还是隐藏？**：成员们讨论了 **Gemini API** 是否公开了 *thinking tokens*，其中一人表示他们通过 **OpenRouter** 能看到，但直接通过 Google API 看不到。
   - 另一位成员提到，他们一直无法通过 API 实际显示 thinking tokens，只能在 **AI Studio** 中看到。
- **寻找 AI 技术教育者**：一位成员征求“高 Alpha、低炒作”的*最佳 AI 技术教育者 / 内容创作者*推荐。
   - 提到了 Harper Carroll（[X 个人资料](https://x.com/harperscarroll?s=21)）、Simon Willison、Prince Canuma 和 Ivan Fioravanti (MLX)。
- **来自 OAI GPT-4 发布会的温馨故事**：一位成员以 [Andrew Mayne 的博客文章](https://andrewmayne.com/2025/05/02/some-personal-observations-about-the-launch-of-gpt-4/)形式分享了 **GPT-4** 发布时*非常温馨的故事*。
   - 未提供关于发布会具体“温馨”细节的进一步信息。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1371844983501361172)** (1 messages): 

> `DSPy Blogpost, LLM Hacking, Bugcrowd` 


- **DSPy 博客文章发布！**：一位成员分享了一篇[关于 DSPy 的博客文章](https://www.bugcrowd.com/blog/hacking-llm-applications-in-the-trenches-with-dspy/)。
   - 该文章涵盖了使用 **DSPy** 对 LLM 应用进行黑客攻击的内容。
- **LLM Hacking 讨论**：博客文章深入探讨了使用 DSPy **攻击 LLM 应用**的方法和策略，提供了实用的见解。
   - 它探索了与 **AI 领域**的安全专业人士和开发人员相关的漏洞和技术。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1371673627371769866)** (16 messages🔥): 

> `DSPy for Agentic Workflows, Data QA with DSPy, MIPRO vs Optuna, TypeScript equivalent to DSPy, DSPy module needing signatures` 


- **评估 DSPy 的 Agent 能力**：一位成员询问了 DSPy 在 Agentic Workflows 中的效用，指出其在声明式程序中的优势，但质疑其是否适用于需要更多模糊性和创造性的任务。
   - 作为回应，有人提到 DSPy 并非专门为 Agent 设计，但它与 LLM 的交互允许通过 Tool Calling 构建工作流，其中模块根据 LLM 的响应添加类似 `CreateSQLquery` 的 Signatures。
- **使用 DSPy 的 DuckDB 数据侦探**：一位用户描述了一个用例，涉及一个利用 **DuckDB 表**连接通过 SQL 和统计分析对列进行数据 QA 的 Agent，并在出现异常时向 Slack 发送警报。
   - 他们指出目前使用 **pydantic Ai** 但对 DSPy 的潜力很感兴趣，实现方式将是通过与 LLM 的每次交互进行 Tool Calling。
- **MIPRO vs Optuna：随机化之争**：一位成员正在寻找一篇对比 **MIPRO** 在使用和不使用 **Optuna** 情况下表现的论文，特别是分析随机组合示例/指令时的偏差。
   - 他们怀疑随机组合可能会收敛到相似的分数，尽管效率可能较低，并希望获得实验证据。
- **寻找 TypeScript 版 DSPy**：一位成员询问是否有 DSPy 的 **TypeScript** 等效项。
   - 提到了几个替代方案，包括 [dspy.ts](https://github.com/ruvnet/dspy.ts) 和 [ax-llm/ax](https://github.com/ax-llm/ax)，后者正在积极维护中。
- **DSPy Signature 难题**：一位用户质疑在 DSPy 模块中为 demo 和对话历史要求 Signatures 的实用性，特别是在具有多个模块的系统中。
   - 他们指出，在一个拥有 N 个模块的系统中，为 K 轮对话维护 **K × N 份聊天记录副本**可能会导致效率低下。


  

---

### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1371924649671131177)** (1 messages): 

> `Discord Message Links, Source code in Prompt` 


- **Discord 消息链接引用**：一位成员引用了一个带有特定链接的 **Discord 消息**，在搜索未果后寻找其来源。
- **源代码就在 Prompt 中**：另一位成员补充说，所有用于生成 JSON 的源代码都已经包含在 Prompt 中了。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1371566470680547408)** (6 messages): 

> `BigInt support, Convolution Puzzle Clarity` 


- **BigInt 集成陷入停滞**：一位成员询问是否会在 Mojo 中添加 **BigInt**，但另一位成员指出，社区包 [decimojo](https://builds.modular.com/packages/decimojo) 已经提供了类似功能。
   - 他们还提到，由于权衡问题，**BigInt/BigDecimal** 可能 *并不适合放入 stdlib*。
- **卷积难题（Convolution Conundrum）得到澄清**：一位成员质疑 [Convolution Puzzle](https://github.com/modular/mojo-gpu-puzzles/blob/1dfd1cc01bb9d6d98185ad405100e6c45855a007/problems/p11/p11.mojo#L104) 中关于内存分配的一行代码的必要性。
   - 一位开发者确认该行代码 *不需要在 host 端*，并感谢该成员报告此问题。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1371576991513448508)** (8 messages🔥): 

> `Open Sourcing MAX Mojo APIs, MAX Graph Tutorials, Tensor Type Migration Code` 


- **MAX Mojo API 已开源**：已弃用的 MAX Mojo API 已经开源，并随后在 [此 commit](https://github.com/modular/modular/commit/34c0209cd537f1d5ab635166358a1713728d7f6f) 中被移除。
   - 所有的 `max.graph`、`max.driver`、`max.tensor` 及其测试都可以在该 commit 中找到，完整的历史记录可以通过 `git log -- mojo/max/src/max/graph` 访问。
- **征集 MAX Graph 教程**：一位用户请求为 **MAX Graph** 提供循序渐进的教程，并指出其现状就像是一个 *只有几个示例的黑盒*。
   - 对此，[Modular 论坛](https://forum.modular.com/t/oss-of-max-mojo-apis/1439)上发布了一个相关的帖子。
- **Tensor 类型迁移代码正在计划中**：内部已有一个关于用户 **tensor 类型** 迁移代码的 ticket，尽管开发尚未开始。
   - 团队有计划解决此问题，但目前没有明确的 ETA。


  

---


### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1371609333980336249)** (1 messages): 

> `PapersChat, Deep Research Agent, Multilingual RAG, Invoice Reconciliation Agent, LlamaParse Updates` 


- **PapersChat 上线，可与论文对话**：团队推出了 [**PapersChat**](https://t.co/ASzjwLfKLC)，这是一个 Agentic AI 应用，让你能够与论文对话，并从 Arxiv 和 PubMed 收集信息。
- **使用新 Agent 进行深度（研究）探索**：发布了一个关于使用 LlamaIndex [构建你自己的 Deep Research Agent](https://www.youtube.com/watch?v=8a_RMSKJC6A) 的视频。
- **多语言与多模态？开启 RAG！**：发布了一个 [多语言、多模态 RAG 系统](https://t.co/69CHCCn8J3) 的演示，未提供更多细节。
- **使用新 Agent 核对发票**：一段新视频展示了如何使用 LlamaIndex.TS 和 LlamaCloud [构建发票核对 Agent](https://www.youtube.com/watch?v=SzVsuMBcv5g)。
- **LlamaParse 获得自动方向检测和模型更新**：**LlamaParse** 获得了新模型和自动方向检测功能；[点击此处阅读更多](https://t.co/tqi17dPJm4)。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1371899127541010522)** (2 messages): 

> `LlamaIndex Memory API, AI Agents Memory Improvement, Short-term chat history, Long-term memory` 


- **LlamaIndex 升级内存功能**：LlamaIndex 宣布了一项重大的 **内存升级**，推出了全新的、灵活的 **Memory API**，融合了短期聊天历史和长期记忆。
   - 此次升级包含即插即用的模块，如用于处理不变静态信息的 [StaticMemoryBlock](https://t.co/wwWnwdyW7s)，以及用于跟踪有用事实列表的 **FactExtractionMemoryBlock**。
- **AI Agent 强化记忆能力**：LlamaIndex 发布了新的 **Memory 组件**，通过短期和长期能力来提升 AI Agent 的记忆力。
   - 这允许存储 **聊天历史** 以实现上下文感知对话，并实现 [静态内存块（static memory blocks）](https://t.co/CDOB3UUO4W)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1371875035463028766)** (3 messages): 

> `google_genai integration, GoogleSearch, FunctionTool` 


- **集成 google_genai 的 GoogleSearch 工具**：一位用户询问如何集成 **google_genai** 库中的 **GoogleSearch**，并指出它与需要密钥和引擎设置的 **GoogleSearchToolSpec** 有所不同。
   - 另一位成员建议将其封装为独立的 **FunctionTool**，以兼容 `chat_with_tools` 方法。
- **将 GoogleSearch 封装为 FunctionTool**：为了将 `google_genai` 库中的 **GoogleSearch** 与 LlamaIndex 的 `chat_with_tools` 方法集成，需要将其封装为 **FunctionTool**。
   - 这种方法可以实现更好的工具处理，并避免了 **GoogleSearchToolSpec** 所需的密钥和引擎设置。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1371646445328793681)** (4 messages): 

> `OpenCL implementation, tensor numel, device/backend, memory movement functions, view changes` 


- **在 Tinygrad 的 OpenCL 后端检测 `long long` 支持**：一位成员询问如何查询给定 device/backend 支持的最大 tensor numel，因为他们正在使用一个较旧的 **OpenCL** 实现，该实现不支持用于索引缓冲区的 `long long` 数据类型。
   - 他们分享了 [一个 tinygrad 脚本](https://cdn.discordapp.com/attachments/1070745817025106080/1371902071112204348/tinygrad_long_long_support_check.py?ex=6824d2de&is=6823815e&hm=3b0d23ba54692d02a6b6bd9f47ff4b7d963d4465a5045204907df5c24c78eff7&)，用于检查 **OpenCL** 实现是否支持大到需要 `long long` 索引的 Tensor；如果返回 false，则必须将操作拆分为块或卸载到 CPU。
- **识别内存移动函数和 View 变更**：一位成员询问如何识别文档中哪些移动函数（movement functions）是原地的（in place），哪些需要新内存。
   - 他们想知道哪些函数只是改变 View，哪些会创建新的 View。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1371689808912715807)** (3 messages): 

> `Creative Writing with oblix.ai, Local vs Cloud Model Orchestration, Edge Computing Savings` 


- **oblix.ai 演示创意写作**：一位成员分享了 [oblix.ai](https://oblix.ai/) 来展示其创意写作能力。
   - 该成员表示，他们只是*想看看它处理创意写作的效果，纯属娱乐*。
- **本地与云端模型的编排 (Orchestration)**：一位成员正在研究 **本地与云端模型** 之间的 **编排**，以便在保持上下文的同时在云端/边缘之间切换。
   - 这种方法旨在根据运行时 Agent 节省 **云端额度 (cloud credits)**。
- **云端/边缘切换的视频演示**：一位成员分享了 [视频演示](https://youtu.be/j0dOVWWzBrE?si=oCjf18i7ykLmzCeh)，展示了在 **云端和边缘模型** 之间的切换。
   - 该实现保留了上下文，并有助于减少云端额度消耗。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1371885700969922722)** (1 messages): 

> `Lambda Workshop, Nobel FutureTech Info Session` 


- **Lambda 工作坊教授 Agentic AI**：参加 **太平洋时间 5 月 15 日上午 10 点** 的 **Lambda 工作坊**，学习如何使用 Lambda 的 Inference API 构建 Agentic 应用。在 **5 月 16 日** 前通过 [此链接](https://forms.gle/UtVhmPS3mitS8Vxu7) 申请，可获得 **$100** 的 Serverless API 额度。
   - 您可以 [在此注册](https://lu.ma/AgentX-lambda) 学习如何优化 Agent 性能以及在生产环境中部署 Agent。
- **Nobel FutureTech 讨论 Genius Club**：由 **Nobel FutureTech Group** 和 **Berkeley RDI** 共同主办的独家信息说明会将于 **太平洋时间 5 月 15 日中午 12 点** 举行，届时将有 **Nobel FutureTech Genius Club** 的资深成员出席。
   - 感兴趣的人士可以 [在此注册](https://lu.ma/NobelFutureTech) 了解导师指导、资金支持和合作机会，或 [在此](https://nobel-futuretech.com/contact.html?link=Ab5B1SNibcW6) 申请加入 Genius Club。