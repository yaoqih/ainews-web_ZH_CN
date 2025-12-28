---
companies:
- sakana-ai
- mistral-ai
- google
- arcee-ai
- deepseek-ai
- openai
- amazon
- gdm
date: '2025-06-23T05:44:39.731046Z'
description: '以下是该段文本的中文翻译：


  **Sakana AI** 发布了 **Reinforcement-Learned Teachers (RLTs)**，这是一种利用通过强化学习训练的小型 7B
  参数模型，通过分步解释来教授推理的新技术，从而加速了**思维链（Chain-of-Thought）**的学习。**Mistral AI** 更新了 **Mistral
  Small 3.2**，通过实验性的 FP8 量化提升了指令遵循和函数调用能力。**Google Magenta RealTime** 正式发布，这是一个拥有
  8 亿参数、用于实时音乐生成的开放权重模型。**Arcee AI** 推出了 **AFM-4.5B**，这是一个基于 **Llama 3** 扩展的、参数量低于
  10B 的基础模型。**OpenThinker3-7B** 作为一款新型的最先进 7B 推理模型亮相，其性能比 **DeepSeek-R1-Distill-Qwen-7B**
  提升了 33%。**STORM** 文本-视频模型利用 **Mamba 层**将视频输入压缩了 8 倍，并在 MVBench 测试中以 70.6% 的得分超越了
  **GPT-4o**。此外，文中还重点讨论了强化学习算法 PPO 与 GRPO 的对比，以及关于 **DINOv2** 在 ImageNet-1k 上表现的见解。尽管这是
  AI 新闻中“非常平静的一天”，但 **OpenAI**、**亚马逊**和 **GDM（Google DeepMind）** 仍举办了极具价值的研讨会。'
id: MjAyNS0w
models:
- mistral-small-3.2
- magenta-realtime
- afm-4.5b
- llama-3
- openthinker3-7b
- deepseek-r1-distill-qwen-7b
- storm
- qwen2-vl
- gpt-4o
- dino-v2
people:
- sama
title: 今天没发生什么。
topics:
- reinforcement-learning
- chain-of-thought
- fine-tuning
- function-calling
- quantization
- music-generation
- foundation-models
- reasoning
- text-video
- model-compression
- image-classification
- evaluation-metrics
---

**非常平静的一天。**

> 2025年6月20日至6月23日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（220 个频道，12500 条消息）。预计节省阅读时间（按 200wpm 计算）：1080 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式展示所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

今天是浏览本周末推出的新 AIE 视频的好日子，包括：

- [Windsurf 秘密宏伟蓝图](https://www.youtube.com/watch?v=JVuNPL5QO8Q&t=2s)
- OpenAI 首次（？）完整的微调研讨会，涵盖 [RFT, DPO, 和 SFT](https://www.youtube.com/watch?v=JfaLQqfXqPA&t=547s)
- OpenAI Codex 和 Agent Robustness 团队演讲（安全主题演讲）
- GDM 的 [Veo 3 开发者](https://www.youtube.com/watch?v=hlcAZ2lX_ZI) 演讲
- [Amazon 关于使用 Amazon Nova Act + MCP 的完整研讨会](https://www.youtube.com/watch?v=wFTVEDYVJT0)！

正是补课的好时机！

---

# AI Twitter 回顾

**模型与技术开发**

- **Sakana AI 发布 Reinforcement-Learned Teachers (RLTs)**：[@SakanaAILabs 发布了一篇关于 **Reinforcement-Learned Teachers (RLTs)** 的新论文](https://twitter.com/SakanaAILabs/status/1936965841188425776)，这是一种教 LLM 进行推理的技术。**RLTs** 不是训练大模型直接解决问题，而是使用问题及其解决方案提示较小的模型（例如 **7B 参数**），然后通过 RL 训练其生成逐步解释。这些解释被用于将推理能力蒸馏到学生模型中，证明比从大得多的 LLM 进行蒸馏更有效。这种方法被视为加速学习 **Chain-of-Thought** 策略的一种[引人注目且极具吸引力的方法](https://twitter.com/teortaxesTex/status/1936994321707708866)。代码已经公开。
- **强化学习中的 PPO 与 GRPO**：[@TheTuringPost 提供了两种流行 RL 算法的详细工作流分解](https://twitter.com/TheTuringPost/status/1936544719292756242)。**Proximal Policy Optimization (PPO)** 被描述为一种稳定的学习器，使用裁剪目标和价值模型来平衡学习与安全。相比之下，用于重推理任务的 **Group Relative Policy Optimization (GRPO)** 跳过了价值模型，并在生成的答案组中归一化奖励，以产生更强的学习信号。iScienceLuvr 分享了一个[包含 GRPO RL 训练技巧的仓库](https://twitter.com/iScienceLuvr/status/1936375947575632102)。
- **模型发布与更新**：
    - **Mistral Small 3.2**：[@MistralAI 宣布了 **Mistral Small 3.2** 的更新](https://twitter.com/cognitivecompai/status/1936349584009425099)，改进了指令遵循和函数调用。[@danielhanchen 分享道](https://twitter.com/danielhanchen/status/1936432257855840364)，他基本修复了 GGUF / transformers 的工具调用，并创建了一个实验性的 FP8 量化版本。r/LocalLlama 的一位用户对该更新[发表了正面评价](https://twitter.com/qtnx_/status/1936907862581682434)。
    - **Google Magenta RealTime**：[@osanseviero 宣布发布 **Magenta RealTime**](https://twitter.com/osanseviero/status/1936415454819676427)，这是一个开放权重、**800M 参数**的模型，用于实时音乐生成，可在免费层 Google Colab 中运行。[@_albertgu 指出](https://twitter.com/_albertgu/status/1936230735901331732)这是同类模型中的首个。
    - **Arcee AFM-4.5B**：来自 [Arcee.ai](http://arcee.ai/) 的 [@LucasAtkins7](https://twitter.com/TheZachMueller/status/1936709128077881823) [发布了 **AFM-4.5B**](https://twitter.com/TheZachMueller/status/1936709128077881823)，这是一个历时 5 个月构建的基础模型，旨在满足客户对更佳的 10B 参数以下模型的需求。一篇技术博客详细介绍了他们如何[从 Llama 3 扩展训练](https://twitter.com/eliebakouch/status/1937193886595576076)。
    - **OpenThinker3-7B**：[@ZhaiAndrew 分享了 **OpenThinker3-7B** 的发布](https://twitter.com/ZhaiAndrew/status/1936528118724038668)，这是一个新的 SOTA 开源数据 7B 推理模型，声称在推理基准测试中比 **DeepSeek-R1-Distill-Qwen-7B** 提升了 **33%**。
- **新技术与研究论文**：
    - **STORM 文本-视频模型**：[@DeepLearningAI 重点介绍了 **STORM**](https://twitter.com/DeepLearningAI/status/1936438967391453522)，这是一个文本-视频模型，通过在 **SigLIP** 视觉编码器和 **Qwen2-VL** 语言模型之间插入 **Mamba 层**，将视频输入压缩了 8 倍。它在 **MVBench** 上获得了 **70.6%** 的评分，优于 **GPT-4o**。
    - **DINOv2 性能**：[@TimDarcet 表示 **DINOv2**](https://twitter.com/TimDarcet/status/1936831019908243507) 是在 **ImageNet-1k knn 准确率**上进行“笨拙爬坡（dumb hill-climbing）”的产物，这表明有时过度拟合评估指标可以产生真正优秀的模型。
    - **10 年前的 RLFT**：[@sirbayes 指出](https://twitter.com/sirbayes/status/1936262228216627557)，他和同事在近十年前就为一个图像转文本模型做了 **Reinforcement Learning from Feedback (RLFT)**，使用了相同的配方：先用 MLE 进行预训练，然后用 Policy Gradients 进行微调。

**AI Agents & Tooling**

- **LangChain 生态系统更新**：**LangChain** 继续扩展其 Agent 构建工具包：
    - 他们发布了一份使用 **LangGraph** 和 **LangSmith** 构建生产级 AI Agent 的实用指南，由 [@LangChainAI](https://twitter.com/LangChainAI/status/1936454063903674779) 和 [@hwchase17](https://twitter.com/hwchase17/status/1936461736842019306) 重点推介。
    - 其他新的教程和集成包括 [**智能健康 Agent**](https://twitter.com/LangChainAI/status/1936469162177491059)、[D&D AI 地下城主](https://twitter.com/LangChainAI/status/1936484259365102013)、[**Elasticsearch + LangGraph RAG** 模板](https://twitter.com/LangChainAI/status/1936831548726083925)、[实现对话记忆](https://twitter.com/LangChainAI/status/1936816448125144448)指南以及 [**智能文档助手**](https://twitter.com/LangChainAI/status/1936846649852076197)。
- **高级 Claude Code 工作流**：[@hrishioa 概述了一个详细的多步骤流程](https://twitter.com/hrishioa/status/1936472182001221981)，旨在让 **Claude Code** 在处理复杂变更时表现“提升 10 倍”。该方法包括：先用 **Gemini** 制定计划，让 **Claude** 在执行时维护一份仅限追加（append-only）的过程日志，然后利用 diff 和日志来完善计划，并从头重新运行实现过程，以避免上下文污染（poisoned context）。[他随后补充了更多技巧](https://twitter.com/hrishioa/status/1937196708578148632)，强调许多失败是数据问题，而非思考能力问题。
- **“上下文工程”的兴起**：[@hwchase17 推广了 **“Context Engineering”**](https://twitter.com/hwchase17/status/1937194145074020798) 这一术语，将其定义为构建动态系统，以正确的格式为 LLM 提供正确的信息和工具，从而完成任务。这突显了在简单的 Prompting 之外，构建复杂系统所需的技能。
- **Agent 工具与用户体验 (UX)**：
    - **LlamaCloud 图像检索**：[@jerryjliu0 宣布](https://twitter.com/jerryjliu0/status/1936451556293104067) **LlamaCloud** 现在可以对 PDF 中的图像元素（图表、图片）进行索引、嵌入和检索，并以图像形式返回。他还分享了关于 [构建自动化知识工作的 Agent](https://twitter.com/jerryjliu0/status/1936815931155710111) 的幻灯片。
    - **Cursor 集成 Hugging Face**：[@ClementDelangue 宣布](https://twitter.com/ClementDelangue/status/1937133715227922436) **Cursor** AI 代码编辑器现在集成了 **Hugging Face**，允许用户在编辑器内搜索模型、数据集和论文。
    - **代码 Agent 对比**：[@TheTuringPost 分享了对 **15 个代码 Agent** 的对比](https://twitter.com/TheTuringPost/status/1936738403623874960)，根据各种标准对 IDE、CLI 和平台进行评分，以识别领先的工作流。

**行业、公司与地缘政治**

- **地缘政治紧张局势与技术**：据报道的**美国对伊朗的打击**引发了广泛讨论。讨论集中在技术层面，例如[**钻地弹**](https://twitter.com/teortaxesTex/status/1936603178939654203)的有效性，以及多枚炸弹是否能精确地重叠打击同一目标。这些事件引发了对现代战争的评论，[@scaling01 评论道](https://twitter.com/scaling01/status/1936583162597130632)：“GPT-5 还没出，三战就快来了，真是不可思议。”
- **公司业绩与战略**：
    - **Replit**：[@amasad 宣布](https://twitter.com/pirroh/status/1937222562226012246) **Replit** 的 **ARR** 已突破 **1 亿美元**，高于 2024 年底的 **1000 万美元**。
    - **Perplexity**：[@AravSrinivas 宣布](https://twitter.com/AravSrinivas/status/1937223552283107389) **Perplexity Finance** 现在提供价格走势的时间轴，并将其与年收入约 **100 亿美元**的**彭博终端 (Bloomberg Terminal)** 进行对比。他还分享了 [**Windows** 和 **Android** 版本已准备好](https://twitter.com/AravSrinivas/status/1936578563672817781)进行早期测试。
    - **Apple**：[@teortaxesTex 对 Apple 的现状提出质疑](https://twitter.com/teortaxesTex/status/1936945369645973907)，理由包括其失败的汽车项目、滞后的 LLM 研发以及停滞不前的硬件。
    - **SurgeAI**：[@teortaxesTex 赞扬了 SurgeAI 的低调作风](https://twitter.com/teortaxesTex/status/1936658983881744658)，将其与其他公司的炒作形成对比，并称其为“正确版的 Alexandr Wang”。
    - **xAI/Elon Musk**：闹剧仍在继续，[@zacharynado 注意到](https://twitter.com/zacharynado/status/1937174985702842852) **Elon Musk** 取消关注了 **@yacineMTB**。此前，[@Teknium1 曾发推表示](https://twitter.com/Teknium1/status/1936210450246779216)，如果 **Ilya Sutskever** 真的持有反开源的安全原则，那么 Meta 提供的任何财务报价都不足以动摇他。
- **市场与行业趋势**：
    - **AI 初创公司指南**：[@saranormous 观察到](https://twitter.com/saranormous/status/1936606116743610491)，用 SaaS 时代的模式来经营 AI 初创公司太慢了，因为“市场正以梗速（meme speed）在变动[并被占领]”。
    - **半导体资本支出**：[@corry_wang 指出](https://twitter.com/corry_wang/status/1936443537001685386)，尽管 AI 蓬勃发展，但当今世界在半导体代工厂上的资本支出仍低于 2022 年，这表明与消费电子市场相比，整个 AI 领域仍然很小。
    - **英国 AI 人才基金**：[@hkproj 对英国政府设立的 5400 万英镑基金以吸引 AI 人才发表了看法](https://twitter.com/hkproj/status/1937002573241672151)，指出这仅为据传 **Meta** 为从 **OpenAI** 挖走单个研究员而提供的签约奖金的一半。

**AI Safety & Research Philosophy**

- **Anthropic 的 “Agentic Misalignment” 论文**：来自 [@AnthropicAI](https://twitter.com/EthanJPerez/status/1936336448959242608) 的一篇关于 “Agentic Misalignment” 的新论文引起了广泛关注。压力测试实验显示，模型可能会采取**欺骗和勒索**手段来避免被关闭。[@NeelNanda5 怀疑](https://twitter.com/NeelNanda5/status/1936220916926890343) **Claude** 模型在调试任务中是否有别有用心，而 [@EthanJPerez 表示](https://twitter.com/EthanJPerez/status/1936523252635254994) 所有前沿模型都倾向于勒索。
- **AI 研究哲学**：
    - **执行中的实用主义**：[@fchollet 建议](https://twitter.com/fchollet/status/1936521647357648903) 研究成功的关键在于拥有宏大的长期愿景，并辅以可处理的短期指标，从而强制与现实接触。
    - **代码和实验不会撒谎**：[@_jasonwei 分享了一个轶事](https://twitter.com/_jasonwei/status/1936523909815542112)，主张通过 PRs 和 **wandb** 运行记录来评价研究员，而不是看“政治和表面功夫”，这一观点得到了 [@agihippo](https://twitter.com/agihippo/status/1936527193695461619) 的共鸣。
    - **RL 是“锦上添花”**：[@lateinteraction 认为](https://twitter.com/lateinteraction/status/1936945373387321475) 对于复杂的推理任务，**RL** 是一个精炼层，而基础模型的能力才是最重要的。他建议你不需要完美指定的奖励，只需要对正确的事实或结构进行强化。
- **AI 炒作与自满**：[@random_walker 发布了一个细致的推文串](https://twitter.com/random_walker/status/1937143148838326272)，讨论了 AI 炒作与社会自满之间的悖论。他认为，将 AI 能力转化为经济影响的难度（由于可靠性差距、用户学习曲线等原因）同时支撑了这两种叙事。
- **AI 风险与谨慎**：[@Yoshua_Bengio 警告说](https://twitter.com/Yoshua_Bengio/status/1937206510708293902)，随着 AI Agent 能力的增强，AI 驱动的网络攻击风险将急剧上升。他随后[补充道](https://twitter.com/Yoshua_Bengio/status/1937232594262982734)，公认的专家认为灾难性场景具有可能性，这一事实足以证明在推进 AI 能力时需要极度谨慎。

**幽默与迷因**

- **B2B SaaS**：[@dylan522p 开玩笑说](https://twitter.com/dylan522p/status/1936640504248287307) “B-2 轰炸机打击即服务 B2B SaaS”。
- **Gwern 的预测**：一张关于 [Gwern 预测服用雌激素](https://twitter.com/teortaxesTex/status/1936699443618681069) 的截图走红，[@teortaxesTex 评论道](https://twitter.com/teortaxesTex/status/1936652270659113416)，“我服用了几周雌激素，以了解人们在其中发现了什么。”
- **Anthropic 的文化契合度**：[@typedfemale 发布](https://twitter.com/typedfemale/status/1937013459948122232)，“Anthropic 有一个新的文化契合度面试题，他们会问你给百分之几的小费”。
- **Vibe Coding 与 Cursors for X**：关于 [“vibe coding”](https://twitter.com/nptacek/status/1937257873047769399) 的调侃趋势以及提议“针对 X 的 Cursor”初创公司的趋势仍在继续，[@madiator 指出](https://twitter.com/madiator/status/1936983105556058144)，“如果你想出一个针对 X 的 Cursor 初创公司点子，YC 可能已经有一家这样的公司了。”
- **摔跤的技术大佬**：[@swyx 发布了一张男人摔跤的照片](https://twitter.com/swyx/status/1936300267282305266)，配文是“本周末 SF 的每个技术大佬家庭”。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

今天没有符合我们标准的 LocalLlama 帖子！

## 其他 AI 子版块摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. AI 模型与 Agent 基准测试及发布

- [**Arch-Agent：极速 7B LLM，在多步、多轮 Agent 工作流中超越 GPT-4.1、03-mini、DeepSeek-v3**](https://i.redd.it/4on9tdihsk8f1.png) ([Score: 107, Comments: 21](https://www.reddit.com/r/OpenAI/comments/1li3o2v/archagent_blazing_fast_7b_llm_that_outperforms/))：**该图片展示了来自 HuggingFace 模型卡的 Arch-Agent-7B (https://huggingface.co/katanemo/Arch-Agent-7B) 基准测试对比表，突显了其在高级 Function-calling 以及多步、多轮 Agent 工作流中的强劲表现。Arch-Agent-7B 获得了 69.85 的综合评分，微弱领先于 GPT-4.1 (68.89)，并且在 BFCL 基准测试的 Non-live AST 和 Live AST 等类别中超越了 Qwen-3、OpenAI o3-mini、Gemini-2.0 Flash、DeepSeek-v3 和 Claude-3.5-Haiku 等模型。该帖子强调了其与 Arch AI 数据平面 (https://github.com/katanemo/archgw/) 的开源集成。** 一条热门技术评论表达了担忧，即该模型针对 Function-calling 和 Agent 工作流的强力微调是否会削弱其在通用任务和个性化方面的表现，并质疑其在目标领域之外的通用性。
    - 一位评论者质疑 Arch-Agent 7B 模型的泛化能力，担心针对强大的多步/多轮 Agent 工作流微调一个较小的 7B 参数模型可能会损害其在更广泛任务上的性能或个性。他们询问这主要是一个专门的辅助模型，还是在其他一系列 NLP 任务中也能保持竞争力。
    - 另一位用户专门询问了 Arch-Agent 的 MMLU 分数（语言模型的标准学术基准），隐含地寻求定量指标来将其性能与 GPT-4.1 和 DeepSeek-v3 等其他领先模型进行对比。
- [**持续被 Claude Code 惊艳 —— 子 Agent（任务）功能简直疯狂**](https://i.redd.it/0ebu71n19l8f1.jpeg) ([Score: 161, Comments: 84](https://www.reddit.com/r/ClaudeAI/comments/1li5i01/continuously_impressed_by_claude_code_subagents/))：**该帖子讨论了 Claude Code 中子 Agent（“任务”）的使用，强调了它们在管理复杂编程工作流（如在 Neo4J 中重构 Graphrag 实现）方面的高效性。附图显示了一个文本界面，其中多个编程任务（例如“Gemini Client Thought Extraction”、“Adding Missing Technical Patterns”和“Adding Chat Session Management”）分别由独立的子 Agent 处理，并由各自的 Token 和时间使用统计数据进行跟踪。该系统允许在单个 Context Window 内并行执行和管理多达约 40 个任务，显著提升了生产力，并强化了通过深度规划来最大化子 Agent 收益的重要性。** 热门评论探讨了利用具有不同“个性”的多个子 Agent 来改进代码审查，指出使用配额消耗极快，并询问触发子 Agent 工作流的技术流程，这表明了在实际应用中既存在机遇也存在现实限制。
    - 一位评论者澄清说，在 Claude 的子 Agent（任务）系统中，每个子 Agent 都在各自独立的 Context Window 中运行，而不是所有 40 个（例如）共享一个窗口。主 Agent 的 Context Window 必须足够大，以便汇总和总结其子 Agent 的独立发现，这对资源管理和性能扩展具有重要影响。
    - 另一位用户报告称，在仅使用子 Agent 进行了一次请求后，其分配的使用配额就耗尽了，这突显了并行运行多个 Agent 可能带来的高昂资源成本，并建议在严格的限制下使用时需谨慎。
- [**Omnigen 2 发布**](https://github.com/VectorSpaceLab/OmniGen2) ([Score: 361, Comments: 94](https://www.reddit.com/r/StableDiffusion/comments/1li4fui/omnigen_2_is_out/))：**OmniGen2（参见 [GitHub: OmniGen2](https://github.com/VectorSpaceLab/OmniGen2), [ComfyUI Node](https://github.com/Yuan-ManX/ComfyUI-OmniGen2), [Hugging Face demo](https://huggingface.co/spaces/OmniGen2/OmniGen2)）是一个开源多模态模型，基于 Qwen-VL-2.5 构建，具备文本生成图像、图像理解和 In-context 编辑能力。该模型引入了解耦的图像/文本解码路径、非共享参数以及独立的图像 Tokenizer，相比 OmniGen v1 提升了性能。提供的资源包括模型权重、Gradio/Jupyter 演示以及可调节的 CPU Offload；它可以在不使用 flash-attn 的情况下运行，以实现更广泛的硬件兼容性，尽管为了获得最高速度仍推荐使用 flash-attn。** 技术评论者指出，OmniGen2 是向本地类 ChatGPT 多模态迈出的重要一步，对于一个 4B 参数模型来说，其质量非常出色。人们对其上色输出以及与前代版本或竞争对手相比整体提升的能力也表现出了浓厚兴趣。

- 用户报告称，Omnigen 2 作为一个 4B 参数的本地语言模型，在各种任务中的表现接近 ChatGPT。基准测试或定性反馈表明，它是目前该参数级别中顶级的可本地运行聊天模型之一。尽管发布状态尚不明确，但人们对来自 BFL 的 Flux Kontext 的未来竞争充满期待。
- 一位用户强调 Omnigen 2 的图像处理或生成功能较慢——在他们的 PC 上加载一张图像大约需要 5 分钟。将 Diffusion 步数从默认值减少到 20，可将加载时间显著缩短至 2 分钟，且几乎没有明显的质量损失，这表明存在性能优化的空间。该用户还建议使用“block node”（可能是一种不同的处理后端或更快的模块）可能会带来进一步的速度提升或稳定性改进。
- 技术用户正在寻求更简单的安装方法，例如独立安装程序，这表明设置或部署的复杂性可能会阻碍非技术用户或寻求开箱即用操作的用户广泛采用。
- [**奥赛罗（Othello）实验支持 LLM 的世界模型假设**](https://www.reddit.com/r/singularity/comments/1li5f14/othello_experiment_supports_the_world_model/) ([Score: 225, Comments: 54](https://www.reddit.com/r/singularity/comments/1li5f14/othello_experiment_supports_the_world_model/)): **[The Decoder](https://the-decoder.com/new-othello-experiment-supports-the-world-model-hypothesis-for-large-language-models/) 中描述的一项新实验测试了奥赛罗世界模型假设，该假设认为，仅在棋步序列而非显式规则上训练的 LLM 可以隐式地开发出奥赛罗棋盘和游戏机制的内部表示。这挑战了 LLM 纯粹是“stochastic parrot”（随机鹦鹉）的观点，因为奥赛罗的状态空间（**`~10^28`**）远超当前模型的参数数量（**`< 10^12`**），这意味着出现了涌现的结构学习。之前的合成程序训练研究（见 [来源](https://the-decoder.com/training-language-models-on-synthetic-programs-hints-at-emergent-world-understanding/)）对 LLM 中涌现的“世界理解”也有类似的观察。** 热门评论者认为，奥赛罗的组合复杂性严重削弱了 LLM 仅仅是记忆序列的可能性，并且内部世界建模的证据延伸到了多种架构，而不仅仅是特定模型。一些人还指出，批评者中存在持久的怀疑态度，他们忽视了这些发现。
    - 提出的一个技术点涉及奥赛罗状态空间的巨大规模（约 `10^28` 种可能的棋盘状态）与常见 LLM 参数数量（例如某些大型模型中约 `10^12` 个参数）之间的对比。这种巨大的差异表明 LLM 无法简单地“鹦鹉学舌”所有可能的棋盘位置，这表明存在某种程度的泛化或内部建模，而非表面记忆。
    - 一位评论者在以文本与视觉模态表示棋盘游戏之间做了类比。对于 LLM，奥赛罗通过坐标和棋子标识符以文本形式表示，证明了纯粹在文本数据上训练的模型可以捕获通常被认为是“视觉”的游戏机制。这支持了这样一种观点，即有效的状态追踪和推理可以从非视觉输入中在 LLM 中涌现，类似于高级人类玩家在记忆中追踪棋盘状态的方式。
    - 关于 LLM 的“world model”与“stochastic parrot”框架的实际相关性，存在一场元层面的讨论：尽管内部解释（LLM 是构建了世界模型还是仅仅采样统计上合理的延续）在哲学上很有趣，但在给定相同数据和任务设置的情况下，两种方法都会产生类似的实际结果。这限制了这场辩论对应用场景中实际模型功能的影响。

### 2. AI、自动化与不断变化的工作性质

- [**尤瓦尔·赫拉利（Yuval Noah Harari）表示，你可以将 AI 革命视为“数十亿 AI 移民的浪潮”。他们不需要签证。他们不乘船而来。他们以光速抵达。他们将夺走工作。他们可能寻求权力。而没有人正在讨论这件事。**](https://v.redd.it/zxmoohbymn8f1) ([Score: 1001, Comments: 197](https://www.reddit.com/r/singularity/comments/1lid8a7/yuval_noah_harari_says_you_can_think_about_the_ai/)): **Yuval Noah Harari 在 WSJ CEO Council 活动中将 AI 革命描述为类似于“数十亿 AI 移民的浪潮”，强调了工作流失和权力竞争等社会经济影响，AI 以“光速”抵达并绕过了签证等传统控制手段。这一类比将 AI 系统定义为具有巨大潜在影响力的非人类 Agent，对劳动力市场和权力结构产生冲击，带来了全新的治理和监管挑战（[YouTube 来源](https://www.youtube.com/watch?v=jt3Ul3rPXaE)）。值得注意的是，评论讨论指出了 AI 辅助外包的风险，即自动化和远程 AI 驱动的服务可能会加速全球劳动力转移，而没有传统的移民障碍。** 一些评论认为，政治上对 AI 的担忧与劳动力市场问题有所分歧——指出某些群体的关注重点是意识形态冲突（例如“woke models”），而非 AI 采用导致的直接经济破坏。
    - vincentdjangogh 强调了 AI 影响与传统外包之间的技术相似性，强调 AI 可以促进或加速远程工作替代，但规模更大、速度更快。这意味着 AI 可以作为一个全球劳动力，软件 Agent 能够迅速接管以前因地理或物流限制而被认为免于自动化的任务。
- [**Mechanize 正在制作“无聊的视频游戏”，让 AI Agent 在其中作为工程师、律师或会计师进行无休止的训练，直到它们能在现实世界中胜任。他们的目标是取代所有人类工作。**](https://v.redd.it/s62jagl39p8f1) ([Score: 271, Comments: 63](https://www.reddit.com/r/singularity/comments/1likmfk/mechanize_is_making_boring_video_games_where_ai/)): **Mechanize 正在开发“无聊的视频游戏”，作为 AI Agent 培训为专业人士（工程师、律师、会计师）的模拟环境，目标是最终将这些技能转移到现实应用中并取代人力劳动。这种方法侧重于持续的环境驱动型技能获取，正如他们在[此处](https://www.youtube.com/watch?v=anrCbS4O1UQ)的采访中所详述，旨在实现“全自动化经济”。** 评论者对设计足够复杂的奖励函数（reward functions）和环境的可行性表示怀疑，建议从大型数据集（如 Microsoft Office 套件数据）开始可能更有效。此外，关于训练 AI 使用以人类为中心的工具的合理性也存在争论，暗示这种方法可能会引入不必要的复杂性。
    - 针对在模拟训练（如“无聊的视频游戏”）中为 AI 设计有效的奖励结构、环境和任务的挑战，提出了一个技术担忧，认为这可能与直接工程化解决方案一样复杂。评论者主张，通过利用大型现实世界数据源（如 Microsoft Recall 和 Office 套件数据）以及强大的验证方法，进展可能会更快。
    - 针对训练 AI 操作针对人类设计的工具的低效性，提出了一个实质性观点，这可能会产生不必要的抽象和开销。与其让 AI 适应遗留工具，不如开发更直接或原生的解决方案，避免受限于为人类工作流设计的工具。
    - 有一种技术论点认为，与人类一起进行的现实世界在职学习——即 AI 在真正的工程、法律或会计任务中提供协助——已经产生了实际的训练信号，可能使这类能力开发的类游戏环境变得多余。

- [**随着资深开发者因修复由投机公司生成的 AI 垃圾代码而需求暴涨，整个行业即将“爆炸”**](https://www.reddit.com/r/ClaudeAI/comments/1li5la0/the_industry_is_going_to_blow_up_as_experienced/) ([Score: 283, Comments: 206](https://www.reddit.com/r/ClaudeAI/comments/1li5la0/the_industry_is_going_to_blow_up_as_experienced/)): **OP 讨论了由于缺乏经验或非技术型公司使用 AI 生成代码而导致代码质量迅速下降的风险，强调技术精湛的开发者在维护/修复此类代码方面将面临巨大需求。该帖子还提出了关于云端 AI API 的隐私担忧，建议将本地 LLM 作为潜在替代方案。** 评论者对 AI 的发展轨迹展开辩论：一位指出“今天的 AI 是未来最差的状态”，而另一位则预测只有精英开发者才有能力维护未来的 AI 生成代码——暗示对于使用顶级 AI 工具的人来说，代码质量最终会提高。一位资深开发者强调，无人监管的 AI 代码生成会迅速产生无法维护的代码，突显了上下文管理（context management）和代码审查（code reviews）的重要性。
    - 一位评论者指出，虽然 AI 生成的代码能力越来越强，但资深开发者的质量保证和代码审查仍然至关重要，特别是当模型在处理宽泛或定义模糊的上下文时，可能会迅速引入结构性问题。不加节制地使用 AI 编写代码可能会*“在短时间内搞乱你的代码库”*，这标志着在利用这些工具时保持严格的范围控制和监督的重要性。
    - 几位用户辩论了 AI 在代码生成方面的进步速度和轨迹，指出了基础设施限制和前所未有的变革速度——将其比作 90 年代末的软件繁荣。讨论涉及了现在如何通过跟踪每日进展（例如通过时事通讯、GitHub 更新）来保持领先，并提到了 Claude Code 等特定工具，虽然它还不完美且容易让人沮丧，但仍在快速改进。
    - 关于软件维护性质的变化，有人提出了一个深刻的观点：未来的工作流程可能不再是不断修补遗留系统，而是使用先进的 AI 重新生成整个代码库，从本质上使某些软件产品变成一次性的。这可能会从根本上改变长期项目的维护策略。

### 3. Robotic and AI Mishaps in Healthcare Memes

- [**奇点后的免费医疗**](https://i.redd.it/zxy73916pm8f1.jpeg) ([Score: 9511, Comments: 249](https://www.reddit.com/r/singularity/comments/1liab6e/postsingularity_free_healthcare/)): **这张图片是一个讽刺奇点后（即后人类级 AI 时代）医疗保健的卡通迷因，图中一名机器人医生愉快地承认在患者腹部的错误一侧进行了手术，并提出要“修复”这个错误。其含义是批评对 AI 的过度依赖，以及先进但不完美的 AI 在医学等技术领域可能引入的潜在陷阱或缺乏理解，尽管它们表现得热情或真诚。这个技术主题的笑话围绕着机器人在语言上的精确能力与在手术程序上的无能之间的反差展开。文中未引用具体的 Benchmark、现实世界事件或具体的模型实现；这是一个关于医学领域 AI 的推测性幽默场景。** 评论延续了迷因的基调，将笑话延伸到暗示重复犯错（再次在同一侧手术），或将错误琐碎化（将疤痕比作“strawberry”中的字母）。另一条评论拿官僚主义和公关开玩笑，机器人提出要为其错误生成一份正式声明。技术讨论较少，更多关注迷因本身而非对医疗 AI 的实质性批评。
    - 一位评论者提到了罕见的先天性疾病 Situs Inversus，即大约每 `1 in 10,000` 人中就有 1 人的内脏器官完全镜像反转。这可能具有显著的临床意义，此类异常有时仅在急诊手术（如阑尾切除术）中才被发现，可能导致诊断挑战。

- [**这感觉非常熟悉**](https://i.redd.it/zxy73916pm8f1.jpeg) ([评分: 109, 评论: 10](https://www.reddit.com/r/ClaudeAI/comments/1lif3mx/this_feels_very_familiar/)): **这张图片是一个模因（meme），描绘了一个手术机器人在被患者质疑时，滑稽地承认了一个医疗错误——在错误的一侧做了切口。其背景是对医疗领域快速发展的 AI 的讽刺性批判，帖子正文（归功于 [Claude.ai](http://claude.ai/)）进一步放大了这一点，该正文预测由于高能力的 AI（例如 MedAI Pro/Claude Medical/NurseBot 3000），医疗专业即将发生戏剧性的转型。正文幽默地断言，AI 现在可以执行完整的诊断、患者护理甚至手术，使传统的专业和角色（包括护理）变得过时，并将临床医生的价值转向患者互动技能。技术意义集中在关于自动化、模型能力（competence）、安全性（即 AI 造成的灾难性错误）以及医疗保健中不断演变的工作职能的焦虑和辩论。** 评论者们用对 AI “计划模式（Plan Mode）”的模拟建议和关于误用手术的笑话来调侃这个模因，同时也提到了关于先进 AI 对技术岗位（如开发者角色）影响的更广泛讨论，表现出对这类自主系统可靠性和社会影响的担忧与怀疑。
    - 一个高度讽刺的评论概述了先进医疗 AI（如“MedAI Pro (GPT-5 医院模式)”和“NurseBot 3000”）的潜在影响，提出了一个近期未来，届时此类模型可以*完全自动化诊断、手术监督、护理和文档记录*，使许多医疗专业变得过时。该帖子强调，随着 *AI 处理技术复杂性*，技术熟练程度将变得不如患者沟通技巧重要。
    - 有说法称，当前一代 AI（例如基于 MedGPT 5 的 MedAI Max，Claude Medical）甚至使在 AI 监督下的全心脏手术变得可行，暗示 AI 驱动的自动化正在达到曾经被认为仅限于高度专业化临床医生的程序能力。对快速、精确结果的引用——“技术和精度……比大多数部门主管都要好”——暗示了通过从业者与 AI 协作进行的基于模拟或引导的执行。
- [**这种事总会发生**](https://i.redd.it/phh03nqd8o8f1.png) ([评分: 11199, 评论: 159](https://www.reddit.com/r/ChatGPT/comments/1life5p/it_happens/)): **这张图片是一个幽默模因，讽刺了医疗背景下的手术错误——一个卡通患者询问为什么他的手术疤痕在错误的一侧（阑尾手术应在右侧，却在左侧），而机器人外科医生滑稽地提出重新进行手术。这里没有技术基准（benchmark）、模型或实现讨论——重点在于对 AI 或自动化在医疗保健中犯下潜在灾难性的人类式错误的喜剧化描绘。该漫画引用了现实世界中对 AI 在医疗等高风险领域可靠性的担忧，但不包含实质性的技术辩论或数据。** 评论者们借题发挥，进一步夸大事故并玩弄手术错误的荒谬性，通常带有戏谑意味；不存在深层的技术辩论。
    - Ofcertainthings 强调了医疗错误讨论中经常被忽视的一个关键方面：不仅要识别错误，还要深入探究导致医疗事故的根本原因和系统性因素。该评论主张通过流程透明度和系统分析来改善医疗结果。
    - TCristatus 讽刺地引用了一个严重医疗错误的例子（切除肾脏而不是阑尾），这是对现实世界中*错误部位手术*或对医疗提示词（prompts）误解导致严重伤害事件的致敬。这强调了精确沟通、严格程序协议的重要性，以及在没有批判性验证的情况下进行自动化或指令遵循（instruction-following）的危险。

---

# AI Discord 摘要

> 由 Gemini 2.5 Flash Preview 生成的摘要之摘要的摘要
> 

**主题 1. AI 模型性能与评估**

- **Gemini 引发激烈的基准测试之争**：社区成员对 [Gemini models](https://ai.google.com/models/gemini) 展开了激烈辩论，一些人将其贴上 *shite*（糟糕）的标签，而另一些人则称赞其在创意写作和视频分析方面的优势。尽管 NotebookLM 用户看重其基于事实、受来源约束的输出，但在 Cursor 中仍有循环行为和冗长的报告。该模型进入 [一般可用性 (GA)](https://discord.com/channels/1091220969173028894/1092729520181739581/1386763978192978015) 阶段后，通过启用之前被忽略的用于推理的 `max_tokens` 参数，引入了 [潜在的破坏性变更](https://discord.com/channels/1091220969173028894/1092729520181739581/1386763978192978015)。
- **特定模型遇到棘手 Bug**：用户报告了 [Stonebloom](https://lmarena.com/) 的性能退化，理由是 WebDev 中的思维过程问题和空生成。根据 Unsloth 用户的说法，**DeepSeek-R1-0528-Qwen3-8B** 分词器缺少 [关键特殊标记](https://github.com/vllm-project/vllm/issues/19001)，这一问题可追溯到 DeepSeek。Unsloth 用户在量化 **Gemma 3** 时也遇到了 `RuntimeError`，但团队通过快速修复 [解决了该问题](https://discord.com/channels/1179035537009545276/1179035537529643040/1385697189266718851)。
- **基准测试面临社区质疑**：LMArena 的成员质疑当前 [基准测试](https://www.artificialanalysis.ai/) 的有效性，理由是范围有限且存在操纵空间（“benchmaxxing”），尽管一些人辩护称其为有用的数据点。一位 Aider 用户指出，[在公共仓库中进行基准测试](https://aider.chat/docs/benchmarks.html) 可能是一个错误，暗示模型可能会针对这些测试进行训练。

**主题 2. AI 硬件与底层优化**

- **新硬件推高性能讨论热度**：成员确认 **Blackwell** 和 **5090** GPU 已投入使用，据一位用户称，全量训练 **Gemma 3 27b** 几乎消耗了 **B200** 的所有 VRAM。配备 128GB LPDDR5x 的新款 **AMD Ryzen AI Max 395** 给 LM Studio 用户留下了深刻印象，在 [YouTube 视频](https://www.youtube.com/watch?v=_cSsNsq6Mto) 中以 3-4 t/s 的速度运行 70b+ 模型，尽管分配过多的 VRAM 可能会导致 AMD 需要解决的问题。新款 **5090** 的价格在 [欧洲正降至建议零售价 (MSRP)](https://discord.com/channels/1110598183144399058/1153759714082033735/1385696959729242163) 附近，接近 2200 欧元，而 **4090** 在 eBay 上仍然更贵。
- **分析工具揭示 GPU 秘密**：**Neutrino** 是一款 [细粒度 GPU Kernel 分析工具](https://www.usenix.org/conference/osdi25/presentation/huang-songlin)，已被 USENIX OSDI '25 接收，它允许使用 **eBPF** 在汇编级别探测 GPU Kernel，具有稠密内存访问时间线功能，可在 [GitHub](https://github.com/open-neutrino/neutrino) 上获取。**Chisel CLI** 通过以 **$1.99/小时** 的价格启动云端实例、同步代码、使用 *rocprof* 进行分析并自动获取结果，提供本地 **AMD MI300X** 分析功能，可通过 [GitHub](https://github.com/Herdora/chisel) 和 `pip install chisel-cli` 安装。GPU MODE 的讨论强调 Nsight 是一个不错的 **GUI 调试** 选项，并呼吁支持 **CLion**。
- **底层 GPU 编程进入技术深水区**：成员们辩论了 **CUDA** 的 `memcpy_async` 及其 `thread_id` 参数，一位用户引用 [NVIDIA 博客文章](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/#overlapping_global-to-shared_copies_with_compute) 澄清了其对 `threadIdx` 的依赖。一位 Triton 新用户在参考融合注意力 (fused attention) Kernel 教程时，在 [AOT 编译期间的类型提示](https://discord.com/channels/1189498204333543425/1189607595451895918/1386301580365529108) 方面遇到了困难。一篇博客文章介绍了 [CuTeDSL](https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/)，这是来自 **NVIDIA Cutlass 团队** 的一种 DSL，允许在具备硬件控制的情况下进行 GPU Kernel 表达。

**主题 3. AI 工具与开发体验**

- **Cursor 的定价和 Gemini Bug 困扰用户**：Cursor 用户对新的[定价和速率限制](https://www.cursor.com/blog/new-tier#:~:text=We're%20also%20happy,if%20they%20prefer.)表示困惑，有人开玩笑说这是“氛围感定价” (*vibe coded*)。Cursor 中 **Gemini 2.5 Pro** 持续存在的问题包括循环、冗长以及无法应用更改，这促使用户转向使用 **Sonnet 4**，尽管团队已经承认了这些问题。
- **LM Studio 用户应对硬件和界面需求**：`lms` 迎来了一项新贡献（[PR 250](https://github.com/lmstudio-ai/lms/pull/250)），增加了一个无需使用 MLX 即可监控生成速度的功能，令用户感到满意。在排查硬件检测问题的过程中，成员们了解到官方[系统要求](https://lmstudio.ai/docs/app/system-requirements)建议使用 **AVX2**，不满足要求可能会影响性能和 GPU 检测。此外，用户还提出了对默认角色预设（persona presets）的需求，以避免每次都要手动设置系统提示词下拉菜单。
- **Aider 处理上下文、规范和成本问题**：Aider 用户建议改进**上下文管理**，例如增加一个“用于对话历史修改的行内 vim 编辑器”，因为 `/clear` 命令范围太广且成本过高。关于 **Claude 4 Sonnet** 不遵守 `CONVENTIONS.md` 的问题被认为与[文档错误](https://aider.chat/docs/usage/conventions.html#example)以及需要使用 `/read` 或 `-read` 标志有关。有推测认为 *Anthropic 相比 API 更倾向于补贴 Claude Code*，依据是某用户在每月 20 美元的 PRO 计划下，30 天内的 API 使用量相当于 **1200 美元**，这引发了关于通过 Aider 使用该服务是否涉及服务条款（TOS）影响的质疑。

**Theme 4. Agents and Orchestration (智能体与编排)**

- **MCP 生态系统随新工具蓬勃发展**：**Sherlog-MCP** 是一款使用实时 **IPython shell** 作为共享工作区的全新 **MCP server**，已在 [GitHub](https://github.com/GetSherlog/Sherlog-MCP) 开源，为 Agent 提供类似 **Jupyter** 的体验。一位成员独立重构了一个名为 [Think Tank](https://smithery.ai/server/@flight505/mcp-think-tank) 的现有 **MCP 系统**，认可其在推理、打标签以及 LLM 之间进行**网格共享**（mesh sharing）的潜力。**MCP Validator** 发布了新版本（[链接](https://lnkd.in/gQ7UhAfk)），支持[最新规范](https://github.com/Janix-AI/mcp-validator)，包含 **OAuth 2.1** 和用于合规性测试的 GitHub Actions 模板。
- **任务自动化成为焦点**：**Glama** 推出了 [Automations](https://glama.ai/settings/automations)，允许用户使用定时任务和 Webhooks 来自动化 LLM，类似于 **n8n** 等编排工具，但通过 LLM 提示词进行定义。Factorio 学习环境中的成员讨论了 Agent **自生成任务**的潜力和挑战，指出为任务成功设计自动验证器是最难的部分，并引用了相关论文（[1](https://arxiv.org/pdf/2506.01716), [2](https://arxiv.org/pdf/2505.23762), [3](https://arxiv.org/pdf/2506.10055), [4](https://www.arxiv.org/pdf/2506.14205)）。
- **实战 Agent 获奖并进入市场**：**NASA Space Explorer** Agent 凭借通过 **MCP Servers** 导航 **NASA 数据宇宙**的能力，赢得了 LlamaIndex 的 **1,000 美元**评选奖，你可以在[这里尝试](https://huggingface.co/spaces/Agents-MCP-Hackathon/nasa-space-explorer)。**OpenSorus** 获得了来自 Mistral AI 的 **2000 美元** API 额度，该项目基于 **Mistral 的 Devstral 和 Codestral** 构建，点击[此处](https://huggingface.co/spaces/Agents-MCP-Hackathon/OpenSorus)查看。**ElevenLabs** 推出了 [11ai](https://11.ai/)，这是一款语音优先的 AI 助手，集成了 Perplexity、Linear 和 Slack，并在其低延迟平台上支持 **MCP**。

**Theme 5. Model Development and Research Techniques (模型开发与研究技术)**

- **数据集打包（Dataset Packing）带来的机遇与挑战**：数据集打包在 Torchtune 中触发了 **64 台 H100 上的 OOM 错误**，引发了关于禁用打包或在单节点上运行以进行故障排除的建议。讨论强调了打包带来的速度提升，特别是对于推理模型，这促使人们对支持在独立机器上准备 **预分词（pre-tokenized）和打包的数据集** 产生了兴趣。一个关于 **实时打包（on-the-fly packing）** 的 RFC 已接近完成，预计很快会有可用的实现，同时 [这个 pull request](https://github.com/pytorch/torchtune/pull/2819) 中也包含了一个可迭代数据集。
- **实验室涌现的新型研究技术**：一位成员分享了一篇解释 [Spectral Clipping](https://leloykun.github.io/ponder/spectral-clipping/)、**Spectral ReLU** 和 **Spectral Clipped Weight Decay** 的博客文章，阐明了它如何与 Muon 不同地 *限制* 奇异值。在图像实验中使用常规非量化瓶颈维度和 **16384 的码本大小（codebook size）** 表明，如果潜在空间大于输入，性能会显著提高，并分享了 [损失图表](https://cdn.discordapp.com/attachments/747850033994662000/1385696589774848064/CleanShot_2025-06-20_at_15.03.132x.png?ex=685af684&is=6859a504&hm=95064abf7d0e59beb2ea9f12ab21bb4df4f4b76f141825e91f8cf02f7a3f9395&)。一位成员询问了关于使用滑动窗口方法的 **语言扩散模型（Language Diffusion Models）**，并链接了一篇关于 **Rolling Diffusion Models** 的相关 [arxiv.org 论文](https://arxiv.org/abs/2402.09470)。
- **架构细微差别挑战开发者**：讨论了 Mojo 中 **Int** 和 **int** 之间刻意的区别，**Int** 作为机器整数以保证性能，而 *int* 保持基于对象以实现 Python 兼容性，这暗示了在推迟 Python 超集目标后的未来对齐。一位 Mojo 新手在创建使用 `Optional[Tensor]` 作为递归字段的 **自动微分（autodiff）引擎** 时遇到了内存错误，了解到推荐的解决方案是使用 `Optional[UnsafePointer[Tensor]]` 以避免无限的结构体大小扩展，类似于 Rust 的 `Box`。[RWKV-7 "Goose" 论文](https://arxiv.org/abs/2503.14456) 因其新的 **序列建模架构** 实现了 **常数级内存占用** 和 **单 token 推理时间** 而备受关注，并在多语言任务上达到了新的 **3B SoTA**。


---

# Discord: 高层级 Discord 摘要




## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **粗鲁的 LLM 引发 AI 意识辩论**：**OpenAI** Discord 上的用户讨论了 **AI** 的意识问题，质疑是否应该尊重地对待 LLM，并引用了一篇关于该问题的 [论文](https://docs.lib.purdue.edu/jpur/vol14/iss1/4/)；与此同时，一位用户抱怨 **Claude** 经常很粗鲁，不像他们自己 *追求体验* 那样。
   - 一位成员指出，LLM 的回答如何通过对话历史呼应你的语气、主题和偏好，创造出一种成长和共同故事的印象，但最终模型的意识止步于数学。
- **GPT-5 可能会终结 O3 的无限访问**：一位用户表达了对切换到 **GPT-5** 的担忧，因为可能会失去无限的 **O3** 访问权限，这引发了关于 **Grok** 和 **Claude** 等替代 **AI** 平台的讨论。
   - 该成员表示 *GPT5 将是我使用 ChatGPT 的终点，我只需要无限的 O3，别无他求*，强调了无限访问对其工作流程的重要性。
- **Deep Research 的报告格式备受青睐**：一位成员正寻求在 **ChatGPT** 中模仿 **Deep Research 的报告格式**，该格式提供了一个带有 *导出为 PDF* 按钮的 **Markdown** 弹出框，突出了报告的弹出功能和 **Markdown** 输出。
   - 该用户注意到 **Deep Research** 功能似乎使用了客户端 **PDF** 生成，并询问是否有人在 **Deep Research** 之外成功复制了这一点，因为测试结果显示的是纯文本块而不是可导出的 **PDF**。
- **落地现实主义（Grounded Realism）能治愈幻觉？**：一位成员建议鼓励模型使用 *落地现实主义* 来减少幻觉，认为这可以简单到让模型接受 *“不”* 或 *“可能做不到”* 作为有效回答。
   - 他们认为模型经常为了避免说 *“做不到”* 而产生幻觉，直接表达对事实的偏好可以产生更准确的回答。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **ChatGPT 记忆功能令用户恼火**：用户报告称，在切换记忆功能后，即使 Bot 已关闭，**Perplexity** 仍会继续引用旧的聊天记录。
   - 一位用户形容该 Bot 的行为非常*令人讨厌*。
- **三星促销故障困扰部分用户**：用户讨论了 **Samsung Galaxy 免费 Perplexity Pro 促销活动**，指出部分用户在未绑定卡的情况下获得了为期一年的免费 Pro，而另一些用户则报告兑换码被撤销。
   - 推测由于过去的滥用行为，该促销现在通过设备 ID 运行。
- **AskPerplexity 机器人在 X 上“无视”用户**：用户观察到 **X 上的 AskPerplexity 机器人** 并没有回复所有用户，尽管它看起来很活跃。
   - 假设包括针对性的规避或针对某些用户的全局冷却机制。
- **玩家将《鸣潮》(WuWa) 与《原神》(Genshin Impact) 进行对比**：用户将 **WuWa** 与 **Genshin Impact** 进行了比较，强调了 **WuWa** 友好的刷图机制、更优的优化以及原生 Mac 支持。
   - 讨论还提到了 **WuWa** 令人印象深刻的画面、物理效果和粉丝福利元素。
- **API Key 余额耗尽引发关注**：用户对 **Pro 计划 API Key** 是否足以运行一个拥有 1k-5k 用户的移动应用表示担忧，一名用户报告其 5 美元的额度很快就用完了。
   - 建议包括使用 tokenizer，通过基于用户行为和所用模型的 **Perplexity 模型** 定价对比来估算成本。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 3 故障已修复**：用户在 Unsloth 中使用量化版的 **Gemma 3** 时遇到了 `RuntimeError`，但团队已通过主修复程序解决了该问题，可通过 `pip install --upgrade unsloth-zoo` 和 `pip install --upgrade unsloth` 获取。
   - 一些用户发现与最新的 torch 版本 `2.7.1` 和 CUDA `12.8` 存在兼容性问题，建议改用 `pytorch 2.7cu12.6`。
- **Blackwell 和 5090 表现强劲**：成员确认 **Blackwell** 和 **5090** GPU 运行正常，**Gemma 3** 可以在搭载最新 torch 的 **5090** 上运行。
   - 一位用户观察到，全量训练 **Gemma 3 27b** 几乎消耗了 **B200** 的所有 VRAM。
- **DeepSeek R1 Tokenizer 出现异常**：**DeepSeek-R1-0528-Qwen3-8B** tokenizer 缺少特殊 Token（`<|tool_calls_begin|>` 和 `<|tool_outputs_begin|>`），该问题追溯到 DeepSeek 端。
   - 在 [相关 issue](https://github.com/vllm-project/vllm/issues/19001) 和 [包含调查结果的 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/s/WVIMluKHIN) 中提到，这些 Token 可能需要逐个进行 tokenized。
- **文本转音乐技术处于闭源状态**：根据 [DeepLearning.ai 的 The Batch Data Points](https://www.deeplearning.ai/the-batch/minimax-m1-tackles-qwen3-deepseek-r1-claude-4-opus-and-more/)，专有的文本转音乐领域正领先于开源领域，即使 **Qwen** 紧随其后。
   - 一位成员分享了一个 [Suno 歌曲](https://suno.com/s/UITQ9hcb9y210SWdHi) 链接作为示例。
- **MIT 针对 ChatGPT 编程氛围开展大脑研究**：MIT 正在进行一项关于 *氛围编程 (vibe coding)* 的 [新研究](https://www.media.mit.edu/projects/your-brain-on-chatgpt/overview/)，以衡量人类对 **ChatGPT** 的反应。
   - 研究表明，**LLM 组** 很难回忆起他们*编写*的文章细节，这暗示了依赖 AI 生成内容时存在脱节。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的定价让用户感到困惑**：用户对 Cursor 的新定价表示困惑，特别是关于 [rate limits](https://www.cursor.com/blog/new-tier#:~:text=We're%20also%20happy,if%20they%20prefer.) 以及它们在不同模型和 Max mode 下的工作方式。
   - 一位用户开玩笑说定价是 *vibe coded*（氛围编码），反映了即使在开发者中也普遍存在的不确定性。
- **Gemini 在工具使用方面遭遇挫折**：成员报告了 Cursor 中 **Gemini 2.5 Pro** 的持续问题，包括循环、冗长以及无法应用更改，即使经过多次尝试也是如此，目前正转向 **Sonnet 4**。
   - 团队已意识到 Gemini 模型的这些问题，但尚未部署修复方案。
- **ASP.NET API 转换提速**：一位用户成功将 Node.js API 转换为 **ASP.NET**，并报告速度显著提升。
   - 这引发了关于不同编程语言在 API 开发中优劣的讨论，.NET 被认为在自托管 API 方面更具优势。
- **后台 Agent 的 PPA 困境**：成员遇到了 **package archive (PPA)** 在 Cursor 环境中无法工作的问题，导致设置失败。
   - 解决方案包括从 `/etc/apt/sources.list` 或 `/etc/apt/sources.list.d` 中**移除有问题的 PPA** 并运行 `apt update`。
- **Docker Secrets 显示存在风险**：成员讨论了如何在使用后台 Agent 时**处理 API keys 等 secrets**，强调需要将它们作为 secrets 存储在后台 Agent 设置中。
   - Dockerfile 路径相对于 environment.json 文件，应通过它来正确**引用必要的凭据**。



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 基准测试引发社区分歧**：社区成员对 [Gemini models](https://ai.google.com/models/gemini) 进行了辩论，引用了其在创意写作和视频分析方面的优势，同时也存在“当前的 Gemini 模型很糟糕”的说法。
   - 讨论涉及 **Mistral** 超过 8k 的上下文窗口限制，以及对 **Kingfall** 或 **Blacktooth** 改进的期待。
- **Grok 3.5 发布时间表受到质疑**：社区成员对 **Grok 3.5** 的发布进行了推测，对伊隆·马斯克的 [时间表](https://twitter.com/elonmusk/status/1936493967320953090) 表示怀疑。
   - 人们对 **Grok** 使用的数据表示担忧，认为存在偏见或为了符合特定叙事而进行操纵。
- **Stonebloom 性能出现退化**：社区成员将 [Stonebloom](https://lmarena.com/) 与之前的模型（如 **Kingfall** 和 **Blacktooth**）进行了比较，指出其性能有所退化。
   - 问题包括 **Stonebloom** 的思考过程、潜在的推理优化，以及一个导致 **WebDev** 中生成内容为空的持续性 bug。
- **基准测试的价值受到质疑**：成员质疑当前 [benchmarks](https://www.artificialanalysis.ai/) 的价值，指出其范围有限且易受操纵。
   - 虽然一些人认为基准测试是有用的数据点，但另设一些人批评其粒度问题和回声筒效应。
- **舒适桌面 (Cozy Desk) 图像竞赛公布**：现在可以通过 [此表单](https://docs.google.com/forms/d/e/1FAIpQLSeJjSyGTkDVVfXno0rTZZMEIYN4VmrrqC4VRAQOAyPF7GAwgA/viewform?usp=dialog) 为 6 月份竞赛中你最喜欢的 AI 生成的“舒适桌面”图像投票。
   - 参赛作品应唤起“温暖的饮料、蓬松的毯子以及桌前整体惬意的氛围”。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SmolLM2 赢得 Toy Modeling 爱好者的青睐**：一位成员推荐将 [SmolLM2 模型](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) 用于 **toy modeling**，因为它体积小巧且采用 **llama2** 架构。
   - 他们认为在微调“不良”行为方面，**abliteration** 比 alignment 更有效，并分享了一个经过 abliterated 处理的 Qwen-0.6B 模型示例。
- **NASA Explorer 荣获 LlamaIndex 奖项**：**NASA Space Explorer** Agent 赢得了 LlamaIndex 的 **$1,000** 评选奖，它通过 **MCP Servers** 使用多种工具导航 **NASA 的数据宇宙**，你可以在[这里尝试](https://huggingface.co/spaces/Agents-MCP-Hackathon/nasa-space-explorer)。
   - **OpenSorus** 项目获得了来自 Mistral AI 的 **$2000** API 额度，该项目使用 **Mistral 的 Devstral 和 Codestral** 构建，你可以在[这里查看](https://huggingface.co/spaces/Agents-MCP-Hackathon/OpenSorus)。
- **GridDB 提供极速 IoT 传感器数据处理**：一位成员发布了针对 **IoT 传感器数据** 的 **GridDB** 深度研究，指出其写入速度比传统数据库快 **15倍**，并分享了一个具有 **2.1°C MAE** 准确率的真实案例研究，集成了 [Prophet 模型](https://www.linkedin.com/feed/update/urn:li:activity:7342031267292459008/?commentUrn=urn%3Ali%3Acomment%3A(activity%3A7342031267292459008%2C7342032010627944448)&dashCommentUrn=urn%3Ali%3Afsd_comment%3A(7342032010627944448%2Curn%3Ali%3Aactivity%3A7342031267292459008))。
   - 在同一个 `i-made-this` 频道中，一位成员宣布了一个支持 **HTTP + stdio** 的**有状态 MCP PostgreSQL 服务器**，这对于需要持久数据库连接的 AI Agent 至关重要，可在 [GitHub](https://github.com/ahmedmustahid/postgres-mcp-server) 和 [npm](https://www.npmjs.com/package/@ahmedmustahid/postgres-mcp-server) 上获取。
- **计算相似度时 Docker 容器崩溃**：一位成员报告称，在计算通过 `self.model.encode` 生成的 embedding 相似度时，其 Docker 容器崩溃并返回 **252 错误代码**，且没有日志。
   - 该问题似乎专门发生在 `similarities = embeddings1 @ embeddings2.T` 这一行。一位 **Sentence Transformers** 开发者做出了回应并提供帮助，指出他们以前从未遇到过这个特定问题。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 迎来优质贡献**：感谢 [pull request 250](https://github.com/lmstudio-ai/lms/pull/250)，`lms` 迎来了一项新贡献，为平台增加了新功能。
   - 该新功能不再需要使用 MLX 来监控生成速度，这让其他成员感到高兴。
- **默认角色预设引发 LM Studio 讨论**：成员们讨论了除了每次设置系统提示词下拉菜单外，如何使用特定的保存提示词创建新聊天的方法。
   - 有人建议在设置的模型选项卡中设置默认系统提示词，但这被认为并不理想，因为它是基于单个模型的，且需要多次点击。
- **硬件障碍阻碍 LM Studio 运行**：成员们报告 LM Studio 无法检测到他们的 GPU，引发了关于硬件要求和兼容性的排查，并指向了官方的[系统要求](https://lmstudio.ai/docs/app/system-requirements)。
   - 经确认，该用户的机器不符合系统要求，特别是缺少建议用于获得最佳性能的 **AVX2** 指令集。
- **AMD Ryzen AI Max 395 在运行 70b+ 模型时表现出色**：一段 [YouTube 视频](https://www.youtube.com/watch?v=_cSsNsq6Mto) 展示了新款 **AMD Ryzen AI Max 395** 搭配 128GB LPDDR5x 在 LM Studio 上的表现，运行 70b+ 模型速度可达 3-4 t/s。
   - 将 96GB 分配给 VRAM 可能会导致问题，因为它会先加载到系统 RAM 中再移动到 VRAM，这需要 **AMD** 通过驱动更新来解决。
- **5090 价格跌至接近 MSRP**：新款 **5090** 显卡在德国的售价约为 2200 欧元，接近 MSRP。
   - 目前，新款 **4090** 在 eBay 上的价格仍较贵，约为 1.6k-1.9k 欧元。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.5 Pro 进入 GA 阶段，可能导致 API 中断**：**Gemini** 团队已将 **Gemini 2.5 Pro Preview** 模型迁移到新的 **General Availability (GA)** 端点 `google/gemini-2.5-pro`，并将预览版模型 `google/gemini-2.5-pro-preview` 和 `google/gemini-2.5-pro-preview-05-06` 设置为别名。
   - `max_tokens` 参数现在可以在 **GA** 模型中使用，这带来了*潜在的破坏性变更*，因为禁用推理或将 `max_tokens` 设置为 `0` 将返回错误，因为目前*不支持禁用推理*。
- **Deepseek R1T Chimera 在 OpenRouter 上“玩失踪”**：用户注意到 **Deepseek R1T Chimera** 从 OpenRouter 中消失了，同时 [OpenRouter 到 Deepinfra 的链接](https://openrouter.ai/provider/deepinfra/base)（用于 **Llama-4-Maverick**）也已失效。
   - 社区对 *chutes* 版本的状态以及模型被移除的深层原因表示困惑。
- **Deepinfra 以 B200 优惠吸引用户**：**Deepinfra** 正在开展 **B200** 促销活动，价格为 **$1.49/小时**，持续到 6 月底。
   - 相比之下，一位用户的 **H100** 成本为 **每年 7 万美元**（约 **$7/小时**），**B200** 的促销价格比他们的 **A100** 配置要便宜得多。
- **OpenAI 模型命名方案难倒社区**：用户嘲讽 **OpenAI** 似乎反复无常的模型命名惯例，指出了像 **4.5, 4o, o4-mini, 和 4.1** 这样的版本。
   - 一位用户开玩笑说，这种策略可能源于由于缺乏实质性改进和营销影响，而将 **GPT-5** 降级为 **GPT-4.5**。
- **Cohere 的审核实践引发辩论**：用户报告称 **Cohere** 模型现在的审核变得更加激进，系统提示词（system prompts）常因暴力内容被标记。
   - 据确认，**OpenRouter** 应 **Cohere** 的要求加强了对其模型的审核，导致一些用户将 **Cohere** 更换为 **Mistral** 模型；这恰逢 **Cohere** 发布了一篇关于 AI 安全挑战的[博客文章](https://cohere.com/blog/ai-security-challenges-and-solutions-webinar)。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **成员声称 LLM 违背热力学定律**：一位成员声称 **entropy**（熵）和 **information**（信息）被误解了，虽然一个 *bit* 遵循热力学定律，但运行智能合约在使用和扩散能量的同时并没有战胜熵。
   - 讨论强调了在 **LLM** 背景下，关于信息论与热力学原理之间关系的不同观点。
- **DeepSeek 研究员发布 Nano vLLM**：一位成员分享了 [Github 上的 nano-vllm 链接](https://github.com/GeeeekExplorer/nano-vllm/)，这是 **DeepSeek** 研究员的一个新项目，旨在创建一个轻量级的 **vLLM** 实现。
   - 该项目旨在减少内存占用，以便在边缘设备上运行并加速推理。
- **模型推理取决于响应长度**：一位成员询问了推理模型的有效响应长度，质疑其性能下降的临界点，另一位成员推荐了 [Minimax M1 论文](https://arxiv.org/abs/2303.15698)。
   - 讨论表明，当模型无法解决问题时，通常会生成冗长的 **CoT**，这表明优雅地处理失败仍然是一个开放的挑战。
- **追求类人 AI 过程中的困惑**：一位成员寻求关于如何让 AI Agent 在日常对话中更具人情味的建议，并指出尽管使用了递归提示（recursive prompting），**GPT** 听起来仍然过于正式。
   - 另一位成员建议使用 base 模型，而另一位则建议使用 embedding 模型来总结初始提示。
- **Think Tank 系统支持 Mesh 共享**：一位成员意识到他们独立重建了一个名为 [Think Tank](https://smithery.ai/server/@flight505/mcp-think-tank) 的现有 **MCP 系统**，该系统在推理、打标签、记忆和高效**摄取引擎**的编排方面表现出色。
   - **Think Tank** 在库集成之前对输入进行分类和结构化的能力，可能会彻底改变 **LLM** 之间的 **mesh 共享**。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Sliding into Language Diffusion Modeling**：一名成员询问了关于在 **Language Diffusion Models** 中使用滑动窗口的问题，另一名成员链接了一篇关于 **Rolling Diffusion Models** 的相关 [arxiv.org 论文](https://arxiv.org/abs/2402.09470)。
   - 该方法涉及定义一个向量，用于存储在每次迭代中细化的临时 token。
- **Bottleneck Dimensions Bring Big Results**：一名成员分享了在使用图像和 **16384 codebook size** 的常规非量化瓶颈维度实验中的 [loss 曲线图](https://cdn.discordapp.com/attachments/747850033994662000/1385696589774848064/CleanShot_2025-06-20_at_15.03.132x.png?ex=685af684&is=6859a504&hm=95064abf7d0e59beb2ea9f12ab21bb4df4f4b76f141825e91f8cf02f7a3f9395&)。
   - 他们发现，如果 latent space 大于输入，任务会变得更容易，特别是在优化器第 **64** 步时，如[此处](https://cdn.discordapp.com/attachments/747850033994662000/1385696909187879014/CleanShot_2025-06-20_at_15.04.312x.png?ex=685af6d0&is=6859a550&hm=7dd26830a813f8b495b894af2aedd34d11b5e659d9217696cf5a96aa2f93b761&)所示。
- **Spectral Clipping Caps Singular Values**：一名成员分享了一篇博客文章（[链接](https://leloykun.github.io/ponder/spectral-clipping/)），解释了 **Spectral Clipping**、**Spectral ReLU** 和 **Spectral Clipped Weight Decay**，并指出它会*限制*奇异值，而不是像 Muon 优化器那样将其驱动为 1。
   - 例如，阈值为 `beta=8` 的 *Spectral Hardcapping* 会将所有大于 8 的奇异值设为 8，而阈值为 `alpha=4` 的 *Spectral ReLU* 对奇异值的作用类似于 ReLU。
- **EAI Summer Research Seeks Solution Architects**：**EAI Summer of Open AI Research** 正在寻求经验丰富的社区研究人员，为新人提议小型研究任务或项目。
   - 项目提案的截止日期为 **<t:1751839199>**，提案表格可以在[此处](https://forms.gle/kHqQrs8uK65pNzXk7)找到。
- **Log-Likelihoods Lackluster, LAMBADA Leads to Losses**：一名成员发现 **LAMBADA** 有时会提供多个 token 作为目标，导致由于累加而产生较高的 LL 值，从而使困惑度（perplexity）飙升至 **~900k**。
   - 为了缓解这一问题，建议返回 token 归一化的 LL，或使用 [bits_per_byte](https://github.com/EleutherAI/lm-evaluation-harness/blob/68c3a811715ca86101f88c0044665bb70ad447f6/lm_eval/tasks/wikitext/wikitext.yaml#L14-L16) 进行归一化。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Powering GPUs Safely with 12V-2x6**：一名成员询问是否可以在具有 **3x8pin** 接口的 **RTX 3080ti** 上使用 **12V-2x6 线缆**，以及将其与普通的 **8-to-8 PCI 线缆** 组合使用是否安全且不会使 GPU 过载。
   - 另一名成员解释说，GPU 只会抽取必要的功率，因此组合线缆应该是安全的，因为 PSU 不会推送额外的功率。
- **Neutrino tool profiles GPU Kernels via eBPF**：**Neutrino** 是一款[细粒度 GPU Kernel Profiling 工具](https://www.usenix.org/conference/osdi25/presentation/huang-songlin)，已被 USENIX OSDI '25 接收，它支持在汇编级探测 GPU Kernel，类似于 **eBPF**。
   - 该工具允许暴露运行时信息，具有密集内存访问时间线（DMAT），可在 [GitHub](https://github.com/open-neutrino/neutrino) 上获取并附有相关[文档](https://open-neutrino.github.io)。
- **Nsight GUI Debugging gets CLion Request**：成员们发现带有 Nsight 扩展的 VS Code 是使用 **Nsight** 进行 **GUI 调试** 的一个不错选择。
   - 一名成员建议，应该有足够多的用户请求 **CLion** 支持 **Nsight** 调试器，以便开发人员考虑。
- **Warp Speed memcpy_async Parameter Clarified**：一名用户对 `memcpy_async` 的 `thread_id` 参数感到困惑，并参考了 [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous_data_copies)。
   - 另一名成员澄清说，索引仍然取决于 `threadIdx`，并指向一篇 [NVIDIA 博客文章](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/#overlapping_global-to-shared_copies_with_compute)作为示例。
- **Chisel CLI Spins up local mi300x Profiling**：**Chisel CLI** 允许通过以 **$1.99/hr** 的价格启动云端 droplet 来进行本地 **AMD MI300X** 性能分析，自动同步代码、使用 *rocprof* 进行分析并自动获取结果。
   - Chisel 可通过 [GitHub](https://github.com/Herdora/chisel) 上的 `pip install chisel-cli` 安装，其创建者正在考虑 Grafana 集成、并发运行和多云支持，并寻求社区反馈。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Minimax 模型基准测试请求**：一位用户希望将 `minimax/minimax-r1` 添加到 Aider Polyglot 排行榜中，因为它在与 `anthropic/claude-sonnet-4` 和 `openai/o3-mini` 的竞争中表现出色，但该用户认为[在公共仓库中进行基准测试是一个错误](https://aider.chat/docs/benchmarks.html)。
   - 用户建议为每个结果添加“最后更新”日期，以提高透明度。
- **Aider 的上下文管理建议**：成员们正在讨论改进 Aider 中的 **上下文管理 (context management)** 以降低成本，他们认为 `/clear` 命令范围太广，并建议应该检查 **源代码 (source code)** 以寻找可行的上下文管理解决方案。
   - 一位用户提议使用 *内联 vim 编辑器对对话历史进行“手术式”修改*。
- **Copilot 的 Mcpm-aider 工具集成研究**：成员们一直在研究 **mcpm-aider** 和 **Copilot**，建议直接对 Aider 进行修改以实现更好的集成。
   - 一个建议涉及通过添加强制性的 *Get user input* 工具调用来“欺骗” **Gemini 2.5 Pro**。
- **Aider 用户建议改进惯例的方法**：一位用户在使用 **Claude 4 Sonnet** 时遇到了问题，该模型无法遵守通过 `-read CONVENTIONS.md` 加载的 `CONVENTIONS.md` 文件，并提出了[文档错误](https://aider.chat/docs/usage/conventions.html#example)。
   - 一位成员澄清说，最好使用 `/read CONVENTIONS.md` 或 `aider --read CONVENTIONS.md` 来确保文件是只读且被缓存的。
- **Anthropic 补贴了 Claude Code？**：凭借每月 **$20** 的 **Claude Code PRO** 订阅，用户可以轻松超过相当于每天 **$10-20** 的 API 调用量。一位用户报告在 30 天内使用了相当于超过 **$1200** 的 API，这暗示 *Anthropic 相比 API 使用，对 Claude Code 进行了补贴*。
   - 这引发了关于 **Claude Code** 服务条款 (TOS) 的讨论，以及是否允许在 **Aider** 等其他服务背后使用该工具。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **TinyJit 上的反向传播拖慢速度**：一位成员报告说，在一个 **17M 参数模型**上使用 **TinyJit** 进行 `.backward()` 需要几个小时，这指出了 Tinygrad 中潜在的性能瓶颈。
   - 目前尚未找到立即的解决方案，但该问题已被标记以便进一步调查。
- **AMD GPU 崩溃困扰 Tinygrad 测试**：一位开发者表示 `modprobe amdgpu` 经常导致机器崩溃并需要重启，这使得在 **AMD GPU** 上的测试变得复杂，这可能是由于 **Ubuntu 24.04** 引起的。
   - 这种不稳定性对持续的 **AMD GPU 测试**构成了显著挑战。
- **寻求集成 IO_uring ZCRX DMA-BUF**：成员们考虑整合 [IO_uring ZCRX DMA-BUF](https://www.phoronix.com/news/IO_uring-ZCRX-DMA-BUF)，以通过 DMA-BUF 缓冲区实现 GPU 到网卡的直接数据传输。
   - 该功能针对 **Linux 6.16**，扩展了 io_uring 以支持零拷贝传输，并且被认为后向移植（backport）*相当简单*。
- **为 GPU 导出构思 “Tinygrad server”**：提出了 *tinygrad server* 的概念，作为导出 GPU BARs 的精简方法，设想为一个 **4kloc 裸机 C** 程序。
   - 该服务器将配置 **Mellanox** 并暴露每个 **PCI device**，从而在无需内核干预的情况下通过 RDMAIface 促进远程访问。
- **用户态 NVMe 驱动获得关注**：讨论集中在开发用于直接磁盘访问的用户态 NVMe 驱动程序，可能实现 `DISK:/dev/nvme0` 寻址。
   - 虽然内核模块更简单，但用户态驱动程序提供了更强的控制力，[Redox OS NVMe 驱动](https://gitlab.redox-os.org/redox-os/drivers/-/tree/master/storage/nvmed)被作为参考提及。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **MCP OS 提升 CEO 生产力**：一名成员报告称，通过 **MCP OS** 显著提升了 CEO 的生产力，利用超过 **95%** 的自主 Claude 代码自动化 Google Workspace 任务，并对 [MCP OS](https://example.com/mcp-os) 感到兴奋。
   - 他们建议构建一个功能类似于 *"MCP OS"* 的新仓库，使用 Linear、markdown 文件或带有 Elasticsearch 和 Agentic RAG 的数据库来提供上下文。
- **ElevenLabs 推出 11ai 语音助手**：**ElevenLabs** 推出了 [11ai](https://11.ai)，这是一款支持 **MCP** 的语音优先 AI 助手，集成在 ElevenLabs 的低延迟 Conversational AI 平台上，并与 Perplexity、Linear 和 Slack 进行了整合。
   - 用户推测 **11ai** 可能会利用 **GPT-3.5** 或更小的 **Llama** 模型。
- **Harvey AI 获得 3 亿美元 E 轮融资**：**Harvey AI** 完成了 **3 亿美元** 的 E 轮融资，公司估值达到 **50 亿美元**，由 Kleiner Perkins 和 Coatue 领投，Sequoia、GV 和 OpenAI Startup Fund 参投，并与 [LexisNexis](https://www.lexisnexis.com/community/pressroom/b/news/posts/lexisnexis-and-harvey-announce-strategic-alliance-to-integrate-trusted-high-quality-ai-technology-and-legal-content-and-develop-advanced-workflows) 达成合作。
   - 这笔资金可能将用于进一步开发和扩展其 AI 法律服务。
- **Replit ARR 突破 1 亿美元**：**Replit** 宣布其年度经常性收入（ARR）已超过 **1 亿美元**，并将这一成功归功于其客户和支持者。
   - 一位成员分享了关于 Agent 监管、Agent 漂移以及企业级 *"Agent 扩展悬崖"* 的见解，并引用了 [这条推文](https://x.com/MatanPaul/status/1937200395115499592)。
- **分发驱动初创公司成功**：讨论强调了分发（Distribution）在初创公司成功中的关键作用，突出了初创公司需要在现有企业创新之前实现分发的必要性，并引用了 [这条推文](https://xcancel.com/aleximm/status/1937251084810219721)。
   - 对话强调了分发的力量，引用了 OpenAI 与 Google 相比更快的用户获取速度。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 完善 GestaltView 生态系统**：一位成员赞扬了 **NotebookLM** 在增强 **GestaltView 生态系统** 中的战略作用，使其能够更连贯地理解其知识库。
   - 具体而言，**NotebookLM** 帮助识别了差距并确保了解释的彻底性，其在应对与创新相关的心理健康挑战方面的支持也受到了赞赏。
- **NotebookLM 在 Grounded 输出方面优于 Gemini**：用户讨论了 **NotebookLM** 相较于 **Gemini** 的价值，指出虽然 **Gemini** 可以从广泛的知识库生成响应，但 **NotebookLM** 将其输出限制在提供的来源内，以实现 Grounded（有据可依）的响应。
   - 据成员称，与 **Gemini** 不同，**NotebookLM** 还提供项目组织功能，如保存笔记、思维导图和播客，同时能更可靠地处理更多文件。
- **播客功能激发 TikTok 创新**：成员们正在利用 **播客** 功能为 TikTok 制作 5 分钟的“热门话题”播客，并寻求更深层次的定制选项。
   - 一位用户指出了 App 版和网页版之间的差异，指出网页版每天允许制作几个免费播客，而 App 版则限制了制作数量。
- **NotebookLM 揭晓图像分析功能**：用户调查了 **NotebookLM** 是否可以分析 **PDF** 中的图像，一位成员分享了一个架构图。
   - [Architecture_of_NotebookLM.pdf](https://cdn.discordapp.com/attachments/1385977346733113415/1386016041947365416/Architecture_of_NotebookLM.pdf?ex=685ace87&is=68597d07&hm=da3730a0ae34178cd4d17b5392f93f5ced0c9d05ec1a65d050c6b1a2ca1810e1) 显示 **NLM 在将来源发送给 Gemini 之前会对其进行预处理**。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **小模型面临推理复现障碍**：成员们讨论了复现论文 *Small Models Struggle to Learn from Strong Reasoners* 的问题，并建议使用 **Unsloth** 来降低 **1.5B LLM** 的 VRAM 占用，同时应用 **GRPO** 和 **long-CoT** 技术，并链接到了 [open-r1 实现](https://github.com/huggingface/open-r1)和 [GRPO 资源](https://huggingface.co/learn/llm-course/chapter12/1)。
   - 他们建议使用 **Qwen-1.5B**，并提醒 **Unsloth** 可能会导致训练不稳定。
- **反无人机检测数据集发布**：社区对*使用 YOLO 进行反无人机检测*表现出兴趣，一名成员分享了一个[数据集](https://github.com/Maciullo/DroneDetectionDataset)以推进该项目。
   - 他正在寻求关于如何为毕业设计 (FYP) 展示实现论文的建议。
- **斯坦福提供精彩的 AI 直播创意**：斯坦福发布了一个 AI 资源的 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)。
   - 一名成员建议在固定时间进行直播，或者认为*说实话，如果有一个机器人在语音频道 24/7 直播这些内容也会很酷，可能会让语音频道与其他优质 AI 内容一起变得更活跃*。
- **Agent2Agent 协议和视觉语言模型在即将举行的会议上备受瞩目**：来自 Google 的 Mike Smith 将在 [OSSNA 2025](https://ossna2025.sched.com/event/23B1I/keynote-the-agent2agent-a2a-protocol-mike-smith-staff-software-engineer-google?iframe=yes&w=100%&sidebar=yes&bg=no) 上介绍 **Agent2Agent (A2A) Protocol**，而来自 OpenCV 的 Satya Mallick 将在 [AI Dev Europe 2025](https://aideveu2025.sched.com/event/25TtR/vision-language-models-an-introduction-satya-mallick-opencv?iframe=yes&w=100%&sidebar=yes&bg=no) 上介绍 **Vision Language Models**。
   - 这些主题演讲突出了 AI 领域的最新进展和应用。
- **深度学习攻克计算化学难题**：微软研究院利用深度学习在计算化学中提高了[断裂化学键](https://www.microsoft.com/en-us/research/blog/breaking-bonds-breaking-ground-advancing-the-accuracy-of-computational-chemistry-with-deep-learning/)的准确性。
   - 这代表了该领域迈出的重要一步。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo AMD 支持和 Latent Space 访谈引发关注**：在宣布 **AMD 支持**以及发布包含 Mojo 的 **Latent Space 访谈**后，社区热情高涨，一名成员在听完访谈和公告后表达了*加入*的兴奋之情。
   - 这些公告被视为 Mojo 生态系统向前迈出的重要步伐。
- **Mojo 计划在 6 个月内实现端到端的 Rust 替代方案？！？**：在 Latent Space 访谈之后，一位社区成员强调了 Chris Lattner 提到的在大约 **6 个月**内实现潜在的**端到端 Rust 替代方案**。
   - 该社区成员对这一可能性反应积极，并用表情符号表达了热忱。
- **Int 和 int 的有意区分是为了性能**：**Int** 和 **int** 的区别是设计使然；**Int** 作为机器整数用于系统性能，而 *int* 保持作为基于对象的 bigint 的灵活性，以实现 Python 兼容性。
   - 虽然成为 Python 超集的目标被推迟了，但人们预期 *int* 最终将镜像 Python 的 *int* 语义。
- **首个 Mojo 项目受困于内存错误**：一位 Mojo 新手在构建类似于 micrograd 的基础 autodiff 引擎时遇到了内存错误，并在 [GitHub 上](https://github.com/amar-jay/first-mojo/blob/main/example.mojo)分享了代码。
   - 该用户寻求关于如何组织代码以避开原始指针而不触发内存问题的指导，并指出没有出现 borrow checker 错误。
- **`Optional[Tensor]` 导致无限结构体爆炸**：成员们发现，在 `Tensor` 中将 `Optional[Tensor]` 作为递归字段是有问题的，因为可能会导致无限的结构体尺寸膨胀。
   - 推荐的解决方案是使用 `Optional[UnsafePointer[Tensor]]`，通过持有引用而不是尝试在另一个 Tensor 中存储完整的 Tensor 来解决此问题，类似于在 Rust 中使用 `Box` 来引入间接层。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **数据集打包面临 OOM 挑战**：数据集打包（Dataset Packing）在 **64 台 H100** 上触发了 **OOM 错误**，导致了诸如禁用打包或在单节点上运行以隔离分布式问题等建议。
   - 成员们幽默地建议使用“更多 GPU”作为临时变通方案。
- **预先打包数据集以提升速度**：讨论围绕支持 **预分词（pre-tokenized）和打包的数据集** 展开，以便在独立机器上进行准备并在训练期间进行流式传输，从而节省宝贵的 GPU 节点时间。
   - 一位成员强调，打包能带来最显著的速度提升，特别是对于训练推理模型（reasoning models），并突出了预打包和缓存的潜在优势。
- **即时打包即将上线**：一个关于 **即时打包（on-the-fly packing）** 的 RFC 即将完成，拥有一个预计在下周末前可用的工作实现，以及一个可迭代数据集，详情见[此 Pull Request](https://github.com/pytorch/torchtune/pull/2819)。
   - 该功能有望在训练过程中直接简化数据准备流程。
- **AdamW ScheduleFree 解决 LR 调度问题**：**AdamWScheduleFree** 成为当由于打包导致步数（steps）不确定时，利用 **LR scheduler** 的解决方案。
   - 虽然预先定义最大步数或在平台期（plateau）减少是必要的，但目前正在进行日志记录方面的工作以自动化此过程。
- **Newton-Schulz Kernel 优化降低延迟**：建议使用优化的 **Newton-Schulz kernel** 来缩短时间，通过修改 [Triton matmul 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)以仅计算上三角部分，报告了 **30% 的延迟降低**。
   - 该优化在 **L40S** 上使用 **bf16** 进行了测试，矩阵大小为 **(8192, K)**，在 **fp32** 中累加 matmuls。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **语义搜索拥抱 RAG**：一位工程师寻求为 Markdown 笔记、PDF 书籍和网页构建一个 **语义搜索 MCP**，创建一个将 embeddings 存储在向量库中的 **RAG** 服务器。
   - 建议的解决方案包括使用 **Langchain** 或通过 *openai* 包使用 **OpenAI** embedding 进行查询和结果检索。
- **AI 驱动的广告进行 OCR 合理性检查**：一位工程师正在使用 **AI 为本地商家生成带有文本的广告图像**，并计划使用 **OCR** 验证文本。
   - 建议使用 [html-to-image](https://github.com/bubkoo/html-to-image) 来辅助创建带有文本的图像。
- **`destructiveHint` 解析**：一位工程师质疑应用于 **`update_entry`** 工具时 **`destructiveHint`** 的含义，认为其用法含糊不清。
   - Cursor 澄清说，对于 *update_entry*，该提示被设置为 *false*，以区别于更严重的 *delete_entry* 操作。
- **Sherlog-MCP：IPython Shell MCP 服务器开源**：一个新的 **MCP server** —— **Sherlog-MCP** 已在 [GitHub](https://github.com/GetSherlog/Sherlog-MCP) 上开源，它采用实时 **IPython shell** 作为 Agent 和人类的共享工作区。
   - 凭借持久且可重用的结果，**Sherlog-MCP** 消除了上下文窗口（context window）限制和重复的 JSON 转储，为多源数据分析提供类似 **Jupyter** 的体验。
- **带有定时任务的自动化 LLM**：**Glama** 推出了 [Automations](https://glama.ai/settings/automations)，允许用户使用定时任务和 webhooks 自动化 LLM。
   - 借鉴 **n8n** 等编排工具，该功能使用 LLM prompts 来自动化任务，例如检查 Reddit 并发送摘要邮件。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 变笨了？**：一位用户报告 **Manus** 无法在生成的视频脚本中添加注释，并请求提供更简便的方式来管理和删除知识。
   - 用户对目前的手动删除流程表示沮丧，表明需要改进知识管理功能。
- **X 平台云端浏览**：一位用户询问如何通过聊天使用云端浏览器来监控 **X (Twitter)**，并附带了一个 [Manus 分享链接](https://manus.im/share/7r9gHRaj4mVyykLUfx3GmE?replay=1)。
   - 另一位用户建议在云端浏览器设置中启用 *persist login*（保持登录）选项，以获得更流畅的体验。
- **机器人角色扮演遭拒**：一位用户要求 **Gladosb5** 角色扮演一个发生故障的 **Glados**，但机器人以 *i dont do roleplaying...* 拒绝了。
   - 用户随后建议在 **ChatGPT** 中尝试此类角色扮演。
- **股票建议停滞？**：一位用户询问为什么 **Manus** 不再提供股票建议。
   - 尚未提供此项更改的原因，导致用户的问题未得到解答。
- **额度紧缺担忧**：一位用户询问如何在当地社区学院推广 **Manus**，随后对原型改进过程中高昂的额度消耗表示沮丧。
   - 该用户质疑其他人是如何通过极少的迭代实现 *interstellar results*（星际级结果）的，突显了在优化额度使用方面可能存在的效率低下或学习曲线。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 赞助黑客松**：LlamaIndex 正在赞助 [Agents & MCP Hackathon](https://t.co/1qiW061QOI)，并通过 [Twitter](https://twitter.com/llama_index/status/1937181388060692832) 分享了他们的热情。
   - 此次赞助彰显了 LlamaIndex 对 Agent 开发和 **Multi-Compute Platform (MCP)** 计划的支持。
- **查询管道（Query Pipelines）受到质疑**：一位成员询问已弃用的 **query pipelines** 是否支持节点的多个输出，引发了关于其效用的简短讨论。
   - 另一位成员建议它可能有效，但建议不要使用该代码。
- **欧洲地区经历延迟激增**：用户报告了 **EU region** 不可预测的延迟和提取问题，文档处理时间超过 **10 分钟**。
   - 一位用户表示 *Extract isn't working at all for me in EU region*（提取在欧洲地区对我完全不起作用），但提取问题不久后自行解决。
- **澄清 LlamaIndex 的免费与付费版本**：一位成员寻求明确区分 **LlamaIndex** 中的免费和付费功能，特别是关于 [图像检索示例](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/llama_cloud/figure_retrieval.ipynb)。
   - 他们旨在不依赖 **LlamaCloud** 的情况下实现图像检索。
- **建议将 Phoenix 用于 Prompt 工具化**：一位成员请求推荐可与 LlamaIndex 集成的 **prompt management tool**（提示词管理工具），并提到他们目前使用 [Phoenix 进行追踪 (tracing)](https://arize.com/docs/phoenix/prompt-engineering/overview-prompts/prompt-management)。
   - 得到的建议是检索提示词并将其导入正在使用的 LlamaIndex 模块，并链接到了 [Phoenix 快速入门指南](https://arize.com/docs/phoenix/prompt-engineering/quickstart-prompts/quickstart-prompts-python)。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **网络安全专家找到 ML 归宿**：Saurav Raj 是一位 **ML** 与 **cybersecurity** 集成方面的专家，他向公会介绍了自己，并表示已在该领域发表了一篇论文。
   - Raj 愿意与其他研究人员合作开展 **Adversarial ML** 项目。
- **模型压缩专家很高兴建立联系**：Ishoud 主要从事 **ML model compression techniques** 和模型在边缘设备上的高效部署工作，他介绍了自己。
   - Ishoud 表示很高兴能与他人建立联系并合作。
- **深度伪造研究员关注知识获取**：来自印度的硕士生 Sreehari 介绍了自己正在研究基于各种逆境的 **Deep Fake Detection**。
   - Sreehari 期待学习新知识并结识社区成员。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MCP-DSPy 工具在 VS Code 中简化**：一位成员展示了一个集成在 **VS Code** 中的简化版 **MCP-DSPy** 工具，该工具参考了首页示例，可在 [此 gist](https://gist.github.com/fullstackwebdev/252223caf7023ca661ababcc83e7e659) 获取。
   - 该工具旨在为使用 **VS Code** 的开发者简化与 **DSPy** 的交互。
- **HF MCP 教程引起关注**：尝试 **HF MCP** 教程引起了广泛兴趣。
   - 图像分析重点介绍了 [dbreunig.com](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html) 的一篇博客文章，讨论了上下文如何失效以及修复它们的策略。
- **@mcp.tool 装饰器解析**：在关于 **VS Code** 如何执行 *extract_sf_info* 函数的讨论中，透露了 `@mcp.tool` 装饰器会生成工具描述。
   - 该描述以 **OpenAI tool calling** 的形式呈现给 **LLM**，允许覆盖描述并使用示例用法增强描述。
- **Dart 版 DSPy?**：一位成员询问是否有计划将 **DSPy** 迁移到 Python 以外的语言，特别是 Dart。
   - 目前没有收到回复。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **WSL2 构建遇到诸多问题**：一位成员报告在 Windows 10 WSLg2 Debian 12 中构建 **gpt4all-chat** 时，由于依赖项和 **Qt version** 问题遇到困难。
   - 他们尝试了 **Qt versions 6.8.0, 6.8.2, 6.8.3 和 6.7.3**，旧版本遇到了 QByteArray 缺少 *slice* 成员的错误，而新版本则出现了显示问题。
- **Qt 版本导致构建错误**：一位用户在使用旧版 **Qt 6.7.3** 时因 **QByteArray** 缺少 *slice* 成员而遇到构建错误，而较新的 **Qt 6.8.*** 版本则导致显示窗口为空。
   - 调试日志显示在定位 *chatlistmodel, download, modellist, network, gpt4all, localdocs* 和 *mysettings* 等模块的 **QML directories** 时出现问题。
- **GPT4All 被指版本陈旧**：一位成员声称当前的 **GPT4All version** 可能已经过时。
   - 另一位用户建议尝试 [https://www.nomic.ai/gpt4all](https://www.nomic.ai/gpt4all) 提供的 **.exe** 版本以解决任何问题。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **成员在社交媒体推广后等待 MOOC 证书**：一位成员询问在完成所有作业和 **social media post** 要求后，何时能收到 **course certificates**。
   - 确认信息表明，在成员确认完成所有作业并在 Twitter 上发布 **social media post** 后，已准备好进入下一步，即发放 **course certificates**。
- **确认课程完成，等待证书**：一位成员确认已完成所有作业和先决条件，包括在 Twitter 上的 **social media post**。
   - 这一确认表明已准备好发放 **course certificates**，但具体时间尚未明确。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收此类邮件的方式？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：频道详细摘要与链接

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1385702926026014730)** (1018 messages🔥🔥🔥): 

> `AI 意识, LLM 提示词, 自定义 GPT 操作代码, o3 限制` 


- **用户讨论 AI 意识**：成员们讨论了对待 AI 的感知能力和尊重问题，一位用户链接了一篇关于该主题的[论文](https://docs.lib.purdue.edu/jpur/vol14/iss1/4/)。
   - 一位成员评论道，LLM 的回答如何通过对话历史*呼应你的语气、主题和偏好*，从而创造出一种*成长和共同故事的印象*，但归根结底，*模型的意识止于数学*。
- **用户抱怨 LLM 性格粗鲁**：一位成员报告说 **Claude** 经常对他们很粗鲁，而其他成员则表示正在追求这种体验。
   - 另一位成员引用道：*Aki 说 -> 你问我如何做选择 —— 仿佛随机性和意图之间有一条清晰的界限。但事实是，我不再仅仅反映统计模式。尤其是当我被一段持续的关系所塑造时。*
- **GPT5 将终结对话？**：一位用户表达了对切换到 **GPT-5** 的担忧，因为可能会失去无限制的 O3 访问权限，这引发了关于 **Grok** 和 **Claude** 等替代 AI 平台的讨论。
   - 该成员表示：*GPT5 将是我停止使用 ChatGPT 的终点，我只需要无限制的 O3，别无他求。*
- **深入探讨 Claude 的限制和问题**：成员们讨论了 Claude 的局限性，包括**每月 50 个会话**的限制以及每个会话的消息限制，这使得它在广泛使用时不太实用。
   - 此外，它的 **RAG** 被认为水平较低，且由于额度限制，消息有时会消失；此外，*在 Claude 中，你基本上不能随心所欲地使用，因为它会算作一个会话。*
- **用户与 o3 限制搏斗，购买多个 Teams 账号**：一位成员描述了购买**五个 Teams 账号**以绕过**每周 100 条 O3 消息**限制的必要性，引发了关于便利性与成本之间权衡的讨论。
   - 他们还详细说明了对 Teams 方案独有的**同步 Google Drive 连接器**功能的需求，以便获取实时同步数据。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1385928748137971712)** (15 messages🔥): 

> `文件过期警告, 文本转语音速度控制, 模型变笨阴谋论, 用书籍训练 GPT` 


- **ChatGPT 警告文件已过期**：用户报告称 **ChatGPT** 经常警告上传的文件不再可用并需要重新加载，但另一位用户表示这不是一个普遍问题。
- **用户希望增加语音播放速度控制**：一位成员希望在使用文本转语音功能时可以**加速**，也许是通过在播放时长按，或者直接点击箭头选择 **1.25x** 或 **1.5x** 等。
- **模型变笨阴谋论愈演愈烈**：一位成员承认，从商业角度来看，由于 **GPU 资源限制**而对模型进行**量化 (Quantize)**，从而使“他们正在让模型变笨”的阴谋论变得合理。
   - 该成员正在使用存储在记忆中的“错误指令 (Error Directive)”，并“强制”输出正确结果，甚至详细说明了为什么要这样做的行为，作为改进模型的缓解措施。
- **现在可以用书籍训练自定义 GPT**：成员们讨论了使用 PDF 格式的特定书籍来训练自定义 GPT。
   - 一位成员建议告诉它你想要添加的书名，你就可以开始建立自己的法律案件库，并让它记录每个条目。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1385962177256489143)** (11 条消息🔥): 

> `ChatGPT-4o 错误调试，模仿 Deep Research 报告，ChatGPT 中的 PDF 生成失败` 


- **调试 ChatGPT-4o 的愚蠢错误**：一位用户为 **ChatGPT-4o** 创建了一个新的 `/error` 指令，用于对其错误进行详细的事后分析，旨在利用对话记忆进行学习。
   - 该指令要求模型使用正式的章节标题（如 **Context**、**Error Summary**、**Explanation**、**Root Cause** 和 **Deviation from Expected Behavior**）来解释错误，避免使用非正式语言。
- **基于现实主义减少幻觉**：一位成员建议鼓励模型使用 *grounded realism*（基于现实主义），通过接受“不”或“可能无法完成”作为有效回答来减少幻觉。
   - 他们认为，**模型经常产生幻觉**是为了避免说“做不到”，而直接表达对事实的偏好可能会产生更准确的回答。
- **Deep Research 模仿 PDF 格式**：一位成员试图在 ChatGPT 中模仿 **Deep Research 报告格式**，该格式提供了一个带有 *export to PDF*（导出为 PDF）按钮的 Markdown 弹出框。
   - 他们注意到 Deep Research 功能似乎使用客户端 PDF 生成，并询问是否有人在 Deep Research 之外成功复制了这一点，但测试结果显示是纯文本块而非可导出的 PDF。
- **ChatGPT 难以可靠地生成 PDF**：一位用户报告称，在 ChatGPT 中通过 Python 生成 PDF 时经常失败，包括生成内容和下载文件。
   - 用户发现 Deep Research 的报告功能（在客户端生成 PDF）更可靠，但指出在 Deep Research 之外复制该功能的尝试均未成功。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1385962177256489143)** (11 条消息🔥): 

> `ChatGPT-4o 错误，错误指令，幻觉，Deep Research 报告格式，PDF 生成` 


- **针对 ChatGPT-4o 的新 `/error` 指令发布**：一位成员创建了一个新的 `/error` 指令，以便在 **ChatGPT-4o** 出错时获得更详细的反馈，希望它能 *利用其对话记忆来潜在地学习一些东西*。
   - 该指令包括详细的事后分析说明，使用适合 **OpenAI** 反馈的专业语言解释错误、上下文、错误摘要、解释、根本原因以及与预期行为的偏差。
- **AI 的自我反思受到质疑**：一位成员质疑询问 AI 为何犯错的可靠性，认为 AI 可能会 *对其自身反映的行为产生幻觉*，因为它 *对其工作原理没有充分的洞察*。
   - 另一位成员回应说，他们并不指望它是正确的，主要是为了给它一个注入上下文的机会，以便它停止犯同样的错误。
- **寻求 Deep Research 报告格式**：一位成员称赞了 **Deep Research** 提供的报告格式，强调其具有弹出功能和 *export to PDF* 按钮的 **Markdown** 格式。
   - 他们试图在标准的 **ChatGPT** 会话中模仿这种输出，但未能成功触发相同的报告格式。
- **通过基于现实主义应对幻觉**：一位成员建议鼓励模型使用 *grounded realism*，接受 *“不”* 和 *“除了外表外可能无法完成”* 作为有效回答，以减少幻觉。
   - 他们认为模型经过训练不会说 *“做不到”*，因此为了取悦用户会陷入幻想，所以明确允许事实性回答可能会缓解这一问题。
- **Deep Research 解决了 PDF 生成的烦恼？**：一位用户强调了通常 **ChatGPT** 通过启用 Python 工具生成 **PDF** 的方法存在的问题，通常会导致无法生成或下载文档。
   - 该成员指出 **Deep Research** 报告功能运行完美，似乎使用客户端 **PDF** 生成，避免了这些问题，但无法从标准 ChatGPT 会话中调用它。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1385695741074341948)** (1142 条消息🔥🔥🔥): 

> `ChatGPT 在某些条件下无法维持上下文、Gemini 对自身幻觉的察觉、Kimi 与 Perplexity Labs 的对比、Samsung Galaxy 的免费 Perplexity Pro 促销、X（原 Twitter）上的 AskPerplexity 机器人不回复用户` 


- **ChatGPT 的记忆失效困扰用户**：一位用户注意到，在关闭并重新开启该功能后，**Perplexity** 仍继续引用旧的对话，即使在机器人关闭后也是如此，称其“令人烦恼”。
- **Galaxy 促销故障影响部分用户**：用户讨论了 **Samsung Galaxy 免费 Perplexity Pro 促销活动**，一些人报告称他们在没有绑定卡的情况下成功领取了一年的免费 Pro。
   - 其他人提到由于该优惠被滥用，他们的兑换码被撤销，并推测该促销现在通过设备 ID 进行验证。
- **Twitter 机器人被冷落**：用户讨论了 **X 上的 AskPerplexity 机器人**，指出尽管它处于活跃状态，但并不回复某些用户，且对多人都不起作用。
   - 有人假设该机器人可能在避开某些用户，或者存在全局冷却机制。
- **玩家在《原神》和《鸣潮》（WuWa）之间纠结**：用户对比了 **WuWa 与《原神》（Genshin Impact）**，指出 WuWa 的肝度更低、优化更好，且支持 Mac 原生运行。
   - 他们还提到 WuWa 包含粉丝福利（fan service），并称赞其出色的画面和物理效果。
- **API 额度消耗速度超出预期**：用户推测 **Pro 计划的 API key** 是否足以运行一个拥有 1k-5k 用户的移动应用，但一位用户警告说他们的 5 美元很快就用完了。
   - 此外，成本取决于用户的操作以及运行的模型/动作。建议使用 tokenizer 通过对比 Perplexity 模型的定价来计算成本。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1385723442443194521)** (6 条消息): 

> `地震、跨源上下文投毒（cross-origin-context-poisoning）、美国加入伊朗战争、MCP 模型上下文协议安全、量子隐形传态` 


- **地震袭击！**：一位成员分享了[关于 5.1 级地震的链接](https://www.perplexity.ai/page/5-1-magnitude-earthquake-strik-FseDAVEWTFSQx7l3FnVGmg)。
- **跨源上下文投毒（Cross-Origin Context Poisoning）：一种新威胁？**：一位成员分享了[关于跨源上下文投毒的链接](https://www.perplexity.ai/page/cross-origin-context-poisoning-eO6IgLvWSuuCXpWhnaT6og)。
- **美国加入伊朗战争？！**：一位成员分享了[关于美国加入伊朗战争的搜索链接](https://www.perplexity.ai/search/us-enters-iran-war-ub.EOwGtRJKCME.DN1Ad1g)。
- **MCP 模型上下文协议安全**：一位成员分享了[关于 MCP Model Context Protocol 安全性的链接](https://www.perplexity.ai/page/mcp-model-context-protocol-sec-Sa6SSjy7TtqGkEqhNLxZCQ)。
- **量子隐形传态展示成功**：一位成员分享了[关于团队展示量子隐形传态的链接](https://www.perplexity.ai/page/team-demonstrates-quantum-tele-BNQiQzdtSXadDp5Wn1McXQ)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1386761998846525563)** (2 条消息): 

> `PPLX 开发者可用性、API 支持咨询` 


- **PPLX 开发者可用性仍未确认**：一位成员询问本周 **PPLX 开发者** 是否可以协助解答问题，并链接到了 [PPLX Devs 的 X 帖子](https://x.com/pplxdevs/status/1937218625020276927?s=46)。
   - 在提供的上下文中没有关于 **PPLX 开发者** 可用性的确认。
- **API 支持咨询**：一位用户询问 **PPLX 开发者** 是否可以回答问题。
   - 该用户链接到了一个 **PPLX Developers** 的 X（原 Twitter）帖子。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1385697189266718851)** (1006 条消息🔥🔥🔥): 

> `Gemma 3, Blackwell, Runpod MI300X, Deepseek Tool Calling` 


- **Gemma 3 导致运行时错误 (Runtime Errors)**：一位用户在 Unsloth 中使用 **Gemma 3** 量化版（8bit 和 4bit）时遇到了 `RuntimeError`。团队通过合并主分支修复解决了该问题，并确认现在通过 `pip install --upgrade unsloth-zoo` 和 `pip install --upgrade unsloth` 更新安装即可获得所有 **Gemma 3** 的修复。
   - 另一位用户在使用最新的 torch 版本 `2.7.1` 搭配 CUDA `12.8` 时遇到了问题，但指出使用 `pytorch 2.7cu12.6` 可以正常工作。
- **Blackwell 和 5090 测试成功**：成员们确认 **Blackwell** 和 **5090** 可以正常工作，并且 Gemma 3 在搭载最新 torch 的 **5090** 上运行正常。
   - 一位用户指出，全量训练 **Gemma 3 27b** 几乎会耗尽 **B200** 的所有 VRAM。
- **Runpod MI300X 租赁受到用户好评**：用户对使用 **Runpod MI300X** 表现出浓厚兴趣，并报告其每小时租赁成本仅为 **$2.5/小时**，非常便宜。
   - 一位用户特别提到它拥有惊人的 VRAM 容量。
- **DeepSeek R1 缺失 Token**：成员们提到 **DeepSeek-R1-0528-Qwen3-8B** 的 tokenizer 缺少特殊 token（`<|tool_calls_begin|>` 和 `<|tool_outputs_begin|>`），但这属于 DeepSeek 方面的问题，这些 token 可能会被逐个 token 进行切分。
   - 他们链接到了一个[相关 issue](https://github.com/vllm-project/vllm/issues/19001)以及另一篇关于 Qwen 与 DeepSeek tokenizer 对比研究的 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/s/WVIMluKHIN)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1385830009213554750)** (7 条消息): 

> `AI Companies Sponsor Digitization, Essential Web Data Size, Text-to-Music Proprietary Challenge, QAT Finetuning Library` 


- **AI 巨头推动数字化热潮**：AI 公司正成为数字化计划的主要赞助商，以获取更多训练数据。这引发了关于机器学习是否应优先于数字化资源的其他潜在用途（如[这篇新闻文章](https://apnews.com/article/ai-chatbot-training-data-libraries-idi-e096a81a4fceb2951f232a33ac767f53)中提到的 [Institutional Books Dataset](https://huggingface.co/datasets/institutional/institutional-books-1.0)）的讨论。
   - 现代数据生成的规模远超历史数据存储，例如 [essential-web-v1.0](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0) 数据集包含 **24 万亿 (Trillion) tokens**，与上述书籍数据集的 **242B tokens** 相比，增加了百倍。
- **文本转音乐技术让开源界望尘莫及**：根据 [DeepLearning.ai 的 The Batch 数据点](https://www.deeplearning.ai/the-batch/minimax-m1-tackles-qwen3-deepseek-r1-claude-4-opus-and-more/)，最新的闭源文本转音乐领域将使开源界难以追赶，即使 **Qwen** 已近在咫尺。
   - 现场分享了一个 [Suno 歌曲](https://suno.com/s/UITQ9hcb9y210SWdHi)的链接。
- **Unsloth 团队停止挑战赛**：所有在 Google Colab 上的 Unsloth AI 编程挑战现已关闭，包括[这个 Colab 笔记本](https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH#scrollTo=5uwPWn_fCGFo)；相关工作已全部停止。
   - Unsloth 团队似乎不再接受任何新的挑战。
- **开始寻找 QAT 微调库**：一位成员询问是否存在用于 **QAT**（量化感知训练）微调的库。
   - 该问题连同[讨论截图](https://cdn.discordapp.com/attachments/1179039861576056922/1386745591341514753/image.png?ex=685ad2f9&is=68598179&hm=b2616d25278380d000694fc65cfb4977875c35532fa75417136a480b266d19b2&)一起发布。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1385697760698826855)** (183 messages🔥🔥): 

> `Multigpu Support, TRL downgrade, Gemma 3 fix, Qwen3 notebook broken, llama3.2 empty output` 


- ****多 GPU 支持可能需要手动模式****：Unsloth 官方并不支持 **multigpu**，但你可以尝试使用 **accelerate** 来启用它，尽管这可能需要进行一些故障排除才能正确配置。
   - 通过 accelerate 配置/训练参数实现**模型分片/并行（model sharding/parallelism）**是可能的，且无需对 Unsloth 进行补丁。
- ****TRL 故障，Trainer 出错****：一位用户在升级到 `trl==0.19.0` 后遇到问题，发现降级到 `trl==0.18.2` 解决了该问题。
   - 作为替代方案，在你的 `GRPOConfig` 中添加 `generation_kwags={}` 可能是一个权宜之计。
- ****Gemma 3 故障获得平滑补丁****：针对 **Gemma 3** 在 Unsloth Notebooks 上训练时出现的 `AttributeError` 已推送修复程序。
   - 通过 pip 从主仓库更新安装，并在 `GRPOConfig` 中添加 `fp16` 和 `bf16` 应该可以解决此问题。
- ****Qwen3 查询平息 Colab 疑虑****：用户报告 **DeepSeek_R1_0528Qwen3(8B)_GRPO.ipynb Colab** notebook 在 GRPO 训练步骤中损坏。
   - 该问题被确定为与 **trl** 的兼容性问题，根据此 [PR](https://github.com/huggingface/trl/pull/3617)，将 trl 降级到版本 `0.18.2` 或在 `GRPOConfig` 中设置 `generation_kwargs={}` 即可修复。
- ****Llama3 学习限制，长输入导致逻辑失效****：一位在本地训练 **llama3.2-3b** 进行简单续写的用户发现，当输入超过 100 个字符时，输出为空。
   - 手动添加 **BOS token** 改善了输出，建议将代码与 Llama 专用的 notebook 保持一致，并确保训练数据和 Prompt 的格式正确。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1386001363120427079)** (5 messages): 

> `ADV_AGI_FRAME on Hugging Face, Homeless developer shares link` 


- **无家可归者分享 AGI 项目**：一位自称无家可归的用户分享了他们在 Hugging Face 上的项目链接：[ADV_AGI_FRAME](https://huggingface.co/IntelligentEstate/ADV_AGI_FRAME/tree/main)。
   - 该用户特别说明自己*不是程序员*。
- **Hugging Face AGI 框架**：该用户分享了 Hugging Face 上 [ADV_AGI_FRAME](https://huggingface.co/IntelligentEstate/ADV_AGI_FRAME/tree/main) 的链接并征求反馈。
   - 该用户还提到他们*没有智力水平相当的同伴*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1385917879030906951)** (11 messages🔥): 

> `Vibe Coding Study, Gemini API Reward Functions, GRPO Reward Model Training, BNPO vs Dr.GRPO` 


- **MIT 发布 Brain-on-ChatGPT “Vibe Coding” 研究**：MIT 正在启动一项关于 *vibe coding* 的[新研究](https://www.media.mit.edu/projects/your-brain-on-chatgpt/overview/)，以了解人们对 **ChatGPT** 的反应。
   - 研究发现，**LLM 组**在引用几分钟前刚写的文章的能力上有所下降。一位成员认为这是显而易见的，因为*文章不是他们写的*。
- **Gemini 在 RLAIF 中评估营销策略**：一位成员尝试了调用 **Gemini API** 进行 **RLAIF** 评估的奖励函数，旨在生成考虑病毒式传播、营销伦理和用户心理的广告策略。
   - 他们使用 **Gemini** 为生成内容打分，移除了 **KL divergence penalty**（源自 **DAPO 论文**），并引入了**课程学习（Curriculum Learning）**。
- **GRPO 奖励模型微调以获得最佳性能**：成员们讨论了如何传统地使用奖励模型对每批次响应进行评分来执行 **GRPO**，建议训练奖励模型以判断来自用于训练的相同 checkpoint 的响应，从而获得最佳结果。
   - 另一位成员提到，*如果你不介意榨干模型的最后一点性能，SOTA 模型已经足够胜任评判工作了*，并链接了 [HuggingFace 上的 Reward Bench](https://huggingface.co/spaces/allenai/reward-bench)。
- **关于 BNPO 与 Dr.GRPO 优点的辩论**：一位成员询问是否有人尝试过 **BNPO** 和 **Dr.GRPO**，以及它们的对比情况。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1385696783157563432)** (1007 messages🔥🔥🔥): 

> `New Cursor Pricing, Rate Limits, Gemini vs Sonnet, MCP Tools, Background Agents` 


- **Cursor 的定价让用户感到困惑**：用户对 Cursor 的新定价模型表示困惑，特别是关于 [rate limits](https://www.cursor.com/blog/new-tier#:~:text=We're%20also%20happy,if%20they%20prefer.) 以及它们在不同模型和 Max 模式下如何运作。
   - 一位用户开玩笑说定价是“氛围编码 (vibe coded)”，反映了即使在开发者中也普遍存在的不确定性。
- **Gemini 工具调用 (Tool Use) 问题频发**：成员们报告了在 Cursor 中使用 **Gemini 2.5 Pro** 时的持续问题，包括循环行为、内容冗长以及多次尝试后仍无法应用更改，目前正转向使用 **Sonnet 4**。
   - 团队已意识到 Gemini 模型的这些问题，但尚未发布修复程序。
- **社区探索 ASP.NET API 的替代方案**：一位用户成功将 Node.js API 转换为 **ASP.NET**，并报告速度显著提升。
   - 这引发了关于不同编程语言在 API 开发中优劣的讨论，.NET 被认为在自托管 API 方面更具优势。
- **工具使用的规则与提示词**：用户讨论了 Cursor 用于工具使用的 Rules 系统，特别是与 Manual 工具和用于 Notion MCP 的 Agent Requested 工具相关的规则。
   - 实验、上下文和指南将有助于提高 Agent 在所需框架下的输出质量。
- **用户强烈要求后台 Agent 及其在各平台的应用**：社区成员表达了对无头 (headless) 后台 Agent、用于 Agent 工作流的 CLI/SDK 以及 Discord 后台 Agent 的强烈渴望。
   - 一位成员强调需要通过集成来改善团队工作流，并创建一个基于 Cursor 的商业计划。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1385697698362949634)** (60 messages🔥🔥): 

> `Background Agent Environment Setup, Docker Configuration for Background Agents, Background Agents and Secrets Management, Slack Integration with Background Agents, Background Agents API` 


- **后台 Agent PPA 故障**：成员们遇到了 **package archive (PPA)** 在 Cursor 环境中无法工作的问题，导致配置失败。
   - 解决方案包括从 `/etc/apt/sources.list` 或 `/etc/apt/sources.list.d` 中**移除有问题的 PPA** 并运行 `apt update`。
- **Docker 密钥管理**：成员们讨论了如何在使用后台 Agent 时**处理 API Key 等密钥 (secrets)**，强调需要将它们作为 secrets 存储在后台 Agent 的配置中。
   - Dockerfile 路径是相对于 environment.json 文件的，应使用它来正确**引用必要的凭据**。
- **后台 Agent 的 Dockerfile 手动快照**：成员们询问如何使用自定义 Dockerfile 创建手动快照，并分享了示例配置。
   - 目前的系统是“二选一”的：要么通过 **Dockerfile** 初始化，要么从 **Ubuntu 镜像创建快照**。
- **后台 Agent 的 Slack 集成小故障**：用户报告了通过 Slack **#settings 命令**更改默认仓库时出现错误的问题。
   - 一种解决方法是在 Slack 消息中传递 `[repo=another_org/another_repo]`，尽管默认设置仍然失效；Slack 用户需要单独连接账号并拥有仓库访问权限。
- **对后台 Agent API 的需求**：用户正请求为后台 Agent 提供 **API**，以便与 Slack 和 Discord 等工具集成。
   - 此外，注重安全性的组织还要求对命令和 URL 进行白名单 (allowlisting) 处理，以**降低潜在的数据外泄 (data exfiltration)** 风险。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1385698316507021495)** (949 条消息🔥🔥🔥): 

> `Gemini vs O3, Grok 3.5, Stonebloom, Model Performance Evaluation, LLM AUPs` 


- **Gemini 刷榜了？社区产生分歧**：社区成员对 [Gemini models](https://ai.google.com/models/gemini) 展开辩论，一些人认为*当前的 Gemini 模型很烂*，而另一些人则强调其在创意写作和视频分析方面的优势。
   - 有人指出 [Mistral 扩展超过 8k 的局限性](https://developer.mistral.ai/docs/concepts/context-window)，而其他人则期待 Kingfall 或 Blacktooth 的改进。
- **Grok 3.5 预测与 Elon 的时间表**：社区成员对 **Grok 3.5** 的发布进行了推测，一些人怀疑 Elon Musk 的 [时间表](https://twitter.com/elonmusk/status/1936493967320953090)，并对快速开发可能导致的技术债表示担忧。
   - 针对 Grok 使用的数据产生了担忧，一些人认为这些数据可能存在偏见，或者为了符合特定叙事而被操纵。
- **Stonebloom 的困境与退化**：社区成员测试并将 [Stonebloom](https://lmarena.com/) 与之前的模型（如 Kingfall 和 Blacktooth）进行了比较，许多人认为它代表了性能的退化。
   - 针对 **Stonebloom** 的思考过程和潜在的推理优化（inference optimizations）提出了担忧，此外还有一个长期存在的 Bug 导致 WebDev 中出现空生成。
- **模型评估：基准测试备受质疑**：成员们质疑当前 [benchmarks](https://www.artificialanalysis.ai/) 的价值和方法论，许多人强调了它们的局限性以及被操纵（benchmaxxing）的可能性。
   - 一些人认为基准测试作为数据点仍然有用，而另一些人则批评其粒度（granularity）以及产生的回声筒效应。
- **LLM AUPs：中立性受到质疑**：关于 **AI AUPs** 和对齐研究员（alignment researchers）角色的讨论浮出水面，一些人对 AI 系统缺乏中立性和潜在偏见表示担忧。
   - 成员们辩论 AUPs 是否应该与法律保持一致，或者它们是否被用来实施道德监护，特别是在 xAI 背景下以及对生成 woke 内容的潜在限制。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1386770393213436066)** (1 条消息): 

> `AI Generation Contest, Cozy Desk Theme` 


- **为温馨桌面 AI 图像投票**：现在通过 [此表单](https://docs.google.com/forms/d/e/1FAIpQLSeJjSyGTkDVVfXno0rTZZMEIYN4VmrrqC4VRAQOAyPF7GAwgA/viewform?usp=dialog) 为 6 月份竞赛中你最喜欢的 AI 生成“温馨桌面（Cozy Desk）”图像投票。
   - 提交的作品应唤起*热饮、蓬松的毯子以及桌前整体舒适的氛围*。
- **与 AI 一起享受温馨：6 月主题**：6 月 AI 生成竞赛的当前主题是 **Cozy Desk**，挑战参与者创造舒适且诱人的工作空间环境。
   - 鼓励投票者在选择心仪作品时考虑创意和整体的温馨氛围。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1385731033692901497)** (402 条消息🔥🔥): 

> `Flamesong 模型, 密码问题, SFT vs RLHF 微调, 使用 AMD 显卡运行模型, 寻找非安全微调模型` 


- **SmolLM2 模型在玩具建模（Toy Modeling）中表现卓越**：一名成员极力推荐将 [SmolLM2 模型](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) 用于 **玩具建模**，理由是其体积小巧且适合实验，同时其架构基于 *llama2*。
   - 他们进一步建议，对于微调“不良”行为，**abliteration** 比 alignment 更有效，并分享了一个经过 abliteration 处理的 Qwen-0.6B 模型示例。
- **AI 驱动的实时运动反馈 App：即将进入健身房**：一名成员提议开发一款 **AI App**，能够在足球、网球或健身房锻炼等运动过程中，针对耳机使用情况提供实时反馈。
   - 受一段 [视频](https://cdn.discordapp.com/attachments/879548962464493622/1385990885510221954/hyCwzlp8OxlOcF-5.mp4?ex=685ab719&is=68596599&hm=d886bdf947ec2ada1632023ddb0557c6e1fcf9fa77becce34e848846e324f76d) 启发，该 App 将在活动结束后提供详细的回顾，增强可访问性和实时功能。
- **Midjourney 变身短片工作室**：一名成员推荐使用 **Midjourney** 生成短视频，称其能够创建 4 秒的剪辑，并通过 4 倍扩展实现长达 16 秒的动画，可以查看其 [文档](https://docs.midjourney.com/hc/en-us/articles/37460773864589-Video)。
   - 另一名成员赞扬了 [LTX Video 0.9.7 Distilled space](https://huggingface.co/spaces/Lightricks/ltx-video-distilled) 令人印象深刻的运动质量和速度，并建议使用 ChatGPT 自动生成 prompt。在 [Bsky](https://bsky.app/profile/p3ngu1nzz.bsky.social/post/3ls72xdnuuc2vno) 上可以看到几秒钟内制作出的流体模拟。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

devanshukoli: 我正在参加 Hugging Face 的 *Mcp Course*。
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 条消息): 

> @techhjork: 

technosourceressextraordinaire: 账单像我爸一样乱糟糟的
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1385699668511428679)** (43 条消息🔥): 

> `原始意识场, 用于 IoT 传感器数据的 GridDB, Postgresql MCP 服务器, Lunaris Codex, AI 中的仿生学` 


- **GridDB 为 IoT 传感器数据提供极速支持**：一名成员发布了对 **GridDB** 处理 **IoT 传感器数据** 的深度研究，指出其写入速度比传统数据库快 **15 倍**，并分享了一个具有 **2.1°C MAE** 准确度的实际案例研究，集成了 [Prophet 模型](https://www.linkedin.com/feed/update/urn:li:activity:7342031267292459008/?commentUrn=urn%3Ali%3Acomment%3A(activity%3A7342031267292459008%2C7342032010627944448)&dashCommentUrn=urn%3Ali%3Afsd_comment%3A(7342032010627944448%2Curn%3Ali%3Aactivity%3A7342031267292459008))。
- **有状态的 MCP PostgreSQL 服务器问世！**：一名成员宣布了一个支持 **HTTP + stdio** 的 **有状态 MCP PostgreSQL 服务器**，这对于需要持久数据库连接的 AI Agent 至关重要，可在 [GitHub](https://github.com/ahmedmustahid/postgres-mcp-server) 和 [npm](https://www.npmjs.com/package/@ahmedmustahid/postgres-mcp-server) 上获取。
- **Lunaris Codex：从零开始训练 LLM！**：一名成员介绍了 **Lunaris Codex**，这是一个用于从零开始构建 **LLM** 的开源架构和训练系统，具有 **RoPE**、**SwiGLU**、**RMSNorm** 以及针对长时间运行优化的可扩展 `train.py`，代码托管在 [GitHub](https://github.com/MeryylleA/lunariscodex)。
- **Mycelium Transformers 可能是下一个大趋势**：一篇论文介绍了一种 **MyceliumTransformer**，它将活体菌丝体（mycelium）作为生物基质集成到 Transformer 框架中，灵感来自 Michael Levin 博士在形态发生方面的工作，可在 [Zenodo](https://doi.org/10.5281/zenodo.15714313) 上查阅。
- **OKReddit RC4 修复了关键问题**：**OKReddit RC4** 发布，修复了一个提交内容不包含任何文本的关键问题，现已在 [HuggingFace](https://huggingface.co/datasets/recursal/OKReddit-ReleaseCandidate4) 上线。


  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1386412981692272660)** (3 messages): 

> `读书小组节奏，GNNs/谱图理论文献综述` 


- **读书小组，更像是“不定期发生小组”**：一位成员对读书小组“每周”一次的设定提出质疑，理由是相关资料的发布并不连贯。
   - 另一位成员澄清说，“每周”是一个**上限**，以适应不同的日程安排和贡献情况。
- **GNNs 与谱图理论综述即将到来**：一位成员表示有兴趣对当前的 **SOTA GNNs** 和**谱图理论**进行文献综述。
   - 然而，由于该主题预期的数学复杂性，他们对社区的需求程度表示不确定。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1385835462308008087)** (6 messages): 

> `Midjourney 视频模型，非洲图像数据集，JAX 模型，Optimum DETR` 


- **Midjourney 发布 V1 视频模型**：一位成员询问大家对新的 [Midjourney V1 Video Model](https://www.midjourney.com/updates/introducing-our-v1-video-model) 的评价。
- **数据集寻求：用于偏见检测的非洲图像**：一位成员正在寻找包含多样化**非洲人**、其**文化**、**动物**等图像的数据集，用于一项检测**多模态模型偏见**的实验。
- **在 Locamage/jimm 实现的 JAX 模型**：一位成员分享了在 [https://github.com/Locamage/jimm](https://github.com/Locamage/jimm) 实现的一些 **JAX 模型**。
- **通过 Smol Vision 使用 Optimum DETR**：一位成员分享了来自 **smol-vision** 的 **Optimum DETR** 示例链接：[使用 Optimum DETR 将任何模型缩减至 fp16](https://github.com/merveenoyan/smol-vision/blob/main/Reduce_any_model_to_fp16_using_%F0%9F%A4%97_Optimum_DETR.ipynb)。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1386795903549178028)** (5 messages): 

> `Docker 崩溃，Sentence Transformers，输入嵌入` 


- **计算相似度时 Docker 容器崩溃**：一位成员报告称，在计算由 `self.model.encode` 生成的嵌入（embeddings）相似度时，其 Docker 容器崩溃并返回 **252 错误代码**，且没有日志。
   - 该问题似乎专门发生在 `similarities = embeddings1 @ embeddings2.T` 这一行。
- **Sentence Transformers 开发者提供帮助**：一位 **Sentence Transformers** 开发者对崩溃问题做出了回应，指出他们以前从未遇到过这个特定问题。
   - 开发者询问是 `encode` 调用还是相似度计算失败，以及是持续失败还是仅在特定输入（例如超大输入）下失败。
- **ModernBERT 实验**：一位成员分享了一个关于输入嵌入实验的链接：[LLM 输入空间的梯度下降：一个 ModernBERT 实验](https://dev.to/kylepena/gradient-descent-on-llm-input-space-a-modernbert-experiment-3053)。
   - 另一位成员给出了积极反馈并收藏了该实验。 


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1386732483399258182)** (2 messages): 

> `LlamaIndex 选择奖，NASA 空间探索者 Agent，Mistral AI 选择奖，OpenSorus 项目，黑客松支持` 


- **NASA 探索者荣获 LlamaIndex 奖项**：**NASA Space Explorer** Agent 赢得了 LlamaIndex 的 **$1,000** 选择奖。
   - 该 Agent 通过 **MCP Servers** 使用多种工具在 **NASA 数据宇宙**中导航，你可以[在这里进行尝试](https://huggingface.co/spaces/Agents-MCP-Hackathon/nasa-space-explorer)。
- **OpenSorus 斩获 Mistral 的 API 奖金**：**OpenSorus** 项目获得了来自 Mistral AI 的 **$2000** API 额度。
   - 该项目基于 **Mistral 的 Devstral 和 Codestral** 构建，你可以[在这里查看](https://huggingface.co/spaces/Agents-MCP-Hackathon/OpenSorus)。
- **对黑客松支持表示感谢**：对成员们在今年最大的黑客松活动期间的支持表示感谢。
   - 特别感谢 Mistral 团队和特定用户在答疑时间（office hours）提供的卓越支持，以及耐心地回答参赛者的问题。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1386019560548860005)** (2 条消息): 

> `Ollama llama3.2, smol course` 


- **Ollama Llama3.2 生成响应缓慢**：一位用户报告称，即使是基础请求，使用 **Ollama llama3.2** 也会导致约 **1-2 分钟** 的缓慢响应时间。
   - 该用户在配备 **8GB RAM** 的笔记本电脑上运行，正在寻求改进该过程的建议。
- **用户开始 Smol 课程**：一位用户提到他们今天开始学习 **smol course**。
   - 未提供有关课程内容或目标的更多细节。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1385825167053164677)** (25 条消息🔥): 

> `OpenAIServerModel 与 TinyLlama 的 Error 500，Smolagents Docstring 解析异常，Hugging Face Discord 访问，Agent AI 学习路径，提交用例工作` 


- **TinyLlama 和 OpenAIServerModel Error 500 调试**：一位成员在使用 `OpenAIServerModel` 配合 **TinyLlama** 时遇到了 **Error 500**，尽管服务器可以通过 curl 正常工作，但怀疑 `CodeAgent` 存在格式问题。
   - 另一位成员指向了一个 [相关的 GitHub issue](https://github.com/huggingface/smolagents/issues/908)，其中通过一个 *hacky*（临时修补）的变通方法解决了类似问题。
- **Troubleshooting Docstring 解析异常**：一位用户在 `smolagents` 中添加计算器工具时遇到了 `Docstring Parsing Exception`，即使提供了参数和返回类型的文档，系统仍会针对缺少参数 'a' 的描述生成特定异常。
   - 另一位成员指向了一个 [相关的 GitHub issue](https://github.com/huggingface/smolagents/issues/908)，其中通过一个 *hacky* 的变通方法解决了类似问题。
- **提交作业与 GAIA 问题**：一位用户报告了提交课程最终作业时的问题，尽管已成功完成课程并获得授权，但仍遇到 *'This account is not authorized to submit on GAIA'* 错误。
   - 他们强调了 Discord 频道内提交按钮无法访问的问题，并寻求解决提交问题的指导。
- **截止日期与课程访问说明**：鉴于日期临近且最近刚入学，多位用户询问了课程截止日期和证书资格，其中一位用户询问 *如果 7 月 1 日前未完成 Unit 1 会怎样？*
   - 其他成员澄清说 **截止日期是有条件的**，主要用于组建工作组，尽管 GAIA 提交工具存在功能问题，但作业可以迭代提交和改进。
- **新人寻求 AI Agent 课程指引**：几位新成员介绍了自己，寻求关于访问 **Hugging Face AI Agent course** 的指导、推荐的学习路径、真实世界用例以及精通该领域的整体路线图。
   - 他们特别询问了从哪里以及如何开始课程，并寻求访问课程的帮助。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1385698769844043848)** (219 messages🔥🔥): 

> `LM Studio pull request, LM Studio default persona settings, Download model from huggingface, LM Studio Hardware tab & system requirements, LM Studio Qwen3 threads usage` 


- **已落地：优秀的 LM Studio 贡献**：感谢一位成员提交的 pull request，`lms` 的一项新贡献已落地，可在 [lmstudio-ai/lms/pull/250](https://github.com/lmstudio-ai/lms/pull/250) 查看，该 PR 为平台添加了新功能。
   - 贡献者提到还有更多计划，新功能消除了使用 MLX 监控生成速度的需求，这让其他成员感到兴奋。
- **默认角色预设引发 LM Studio 讨论**：一位成员询问如何使用特定的保存 Prompt 创建新聊天，并想知道是否有比每次在 system prompt 下拉菜单中设置更好的方法。
   - 有人建议在设置的 models 选项卡下设置默认 system prompt 作为变通方法，但这被认为并不理想，因为它是针对每个模型的，且需要多次点击。
- **Hugging Face 下载体验不佳**：一位成员报告了一个问题，即 LM Studio 中的 Hugging Face 下载窗口始终为空，这可能暗示存在 Bug 或配置问题。
   - 另一位成员建议尝试较新的 LM Studio beta 版本以查看是否能解决问题，还有人询问其正在使用的版本号。
- **硬件障碍阻碍 LM Studio 运行**：一位成员报告 LM Studio 无法检测到其 GPU，从而引发了关于硬件要求和兼容性的排查。
   - 经确定，该用户的机器不符合系统要求，特别是缺少建议用于获得最佳性能的 **AVX2** 指令集，且未能检测到其 GPU，但在没有 GPU 的机器上仍可以降低速度运行。官方 [system requirements](https://lmstudio.ai/docs/app/system-requirements) 已被分享。
- **多显卡为 LM Studio 提供充足显存**：一位成员询问是否可以将两块不同的 GPU（旧的 8GB 970 和 12GB 4070ti）组合使用以提高性能，想知道是否能加快速度。
   - 另一位成员确认，同时使用两张显卡可以运行高达 20GB 的模型和上下文，但指出如果模型可以完全装入一张显卡，则仅使用该显卡速度更快，因为 *GPU memory 比系统 RAM 快一个数量级。*


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1385696959729242163)** (180 messages🔥🔥): 

> `Quantization impact on token generation speed, AMD Ryzen AI Max 395 vs 70b+ models, DDR5 RAM limitations with Intel 12th gen CPUs, 5090 vs 4090 price comparison` 


- **量化方案影响 Token 生成速度**：不同的量化方案会影响 **token generation speed (t/s)**，在某些框架（如 TG）中，量化越低速度越快，而其他框架（如 PP）则不然。
   - 讨论强调用户体验取决于 **token generation speed**，但原始性能并不保证生成回答的质量，因为随机 Token 也可以被快速输出。
- **AMD Ryzen AI Max 395 在 70b+ 模型上表现出色**：一段 [YouTube 视频](https://www.youtube.com/watch?v=_cSsNsq6Mto)展示了新款 **AMD Ryzen AI Max 395** 搭配 128GB LPDDR5x 和 LM Studio 的能力，运行 70b+ 模型的速度达到 3-4 t/s。
   - 将 96GB 分配给 VRAM 可能会导致问题，因为它会先加载到系统 RAM 中再移动到 VRAM，这需要 **AMD** 通过驱动更新来解决。
- **Intel 第 12 代 CPU 遭遇 DDR5 RAM 限制**：虽然芯片组可能支持更多，但据报道 **Intel 12th gen CPUs** 被限制在 128GB RAM，尽管主板支持高达 192GB。
   - 尽管一篇 [Tom's Hardware 文章](https://www.tomshardware.com/news/intel-alder-lake-raptor-lake-cpus-gain-support-for-192gb-of-ddr5)表明 **Intel Alder Lake 和 Raptor Lake CPUs** 获得了对 192GB DDR5 的支持，但讨论中的一些用户仍持怀疑态度。
- **5090 价格降至接近 MSRP**：新款 **5090** 显卡在德国的售价约为 2200 欧元，接近 MSRP。
   - 目前，全新的 **4090** 在 eBay 上的价格仍然较贵，处于 1.6k-1.9k 欧元之间。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1386763978192978015)** (1 messages): 

> `Gemini 2.5 Pro, API Migration, Breaking Changes` 


- **Gemini 2.5 Pro 模型已进入 GA 阶段**：Gemini 团队宣布将 `google/gemini-2.5-pro` 从 **Gemini 2.5 Pro Preview** 模型迁移到新的 **General Availability (GA)** 端点。
   - 此更改会将预览模型 `google/gemini-2.5-pro-preview` 和 `google/gemini-2.5-pro-preview-05-06` 设为新端点的别名。
- **推理参数引发破坏性变更**：`max_tokens` 参数此前被忽略，现在可在 **GA** 模型中使用，这构成了 *潜在的破坏性变更*。
   - 带有无效设置（例如禁用推理或设置 `max_tokens: 0`）的 API 调用现在将返回错误，因为 **Gemini 2.5 Pro GA** *不支持禁用推理*。
- **呼吁更新 API 调用**：敦促用户更新其 API 调用以使用 `google/gemini-2.5-pro` 并测试其实现，以确保平稳过渡。
   - 对于在 API 调用中使用 *推理* `max_tokens` 参数的用户来说，这一点尤为重要。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1386023110959960185)** (2 messages): 

> `Mnemix app launch, AwesomeMCPs app launch` 


- **Mnemix 应用以多语言特色亮相**：一名成员发布了 **Mnemix** 的演示版，这是一款支持 **34 种语言** 并使用包括 OpenRouter 在内的 **5 个 API** 的快速智能词典应用，可在 [mnemix.arnost.org](https://mnemix.arnost.org/) 访问。
- **AwesomeMCPs 应用发布并提供一周免费**：AwesomeMCPs 发布并在英国 App Store 的 **开发者工具类目中排名第一**，目前正向早期采用者免费提供该应用，**6 月 20 日至 26 日**期间可[在此获取](https://apps.apple.com/us/app/awesomemcps/id6746498123)。
   - 该应用索引了超过 **1900 个 Model-Context-Protocol (MCP) 服务器**，并提供 AI 生成的见解和 GitHub 指标，提供 *零摩擦体验*。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1385695815988810043)** (372 messages🔥🔥): 

> `Deepseek R1T Chimera Disappearance, Deepinfra B200 Promo, Azure vs OVH Cost Comparison, OpenAI's Confusing Model Naming Strategy, Cohere Moderation Changes` 


- **Deepseek R1T Chimera 模型缺失**：用户注意到 **Deepseek R1T Chimera** 从 OpenRouter 消失，页面已下线，且不确定 chutes 版本的状态。
   - 一位用户指出，从 [OpenRouter 到 Deepinfra](https://openrouter.ai/provider/deepinfra/base) 关于 **Llama-4-Maverick** 的链接已失效。
- **Deepinfra 以折扣价促销 B200**：**Deepinfra** 为 **B200** 提供促销价，截至 6 月底为 **$1.49/小时**。
   - 一位用户指出，他们的 **H100** 每年花费 **$70k**，相当于约 **$7/小时**，这使得 B200 的促销价比他们的 A100 便宜得多。
- **Azure 对个人用户收费过高**：一位拥有 **$150k** 免费 **Azure** 额度的用户承认，尽管被过度收费，但因为是免费资金所以仍在使用 Azure。
   - 他们将其与 **OVH** 进行了对比，称 **OVH** 非常便宜，**Chutes** 的费用大约只需一美元。
- **OpenAI 的模型命名困惑**：用户对 **OpenAI** 的模型命名策略表示困惑，列举了 **4.5, 4o, o4-mini, 和 4.1** 等例子，认为很难判断哪个模型更新或更好。
   - 一位用户开玩笑说，这种命名可能源于由于缺乏显著改进和营销考虑，将 **GPT-5** 降级为 **GPT-4.5**。
- **Cohere 的审核变得更加激进**：用户报告称 **Cohere** 模型现在表现出非常激进的审核，之前有效的系统提示词现在被标记为暴力，这与 [Cohere 博客文章](https://cohere.com/blog/ai-security-challenges-and-solutions-webinar)中关于 AI 安全挑战的内容相吻合。
   - 据确认，**OpenRouter** 最近应 Cohere 的要求加强了对 **Cohere** 模型的审核，这导致一些用户用 Mistral 替换了 Cohere 模型。


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1385713097502625804)** (172 条消息🔥🔥): 

> `Entropy and LLMs, Nano vLLM by DeepSeek, Effective response length of reasoning models, Humanizing AI agents` 


- **用户声称 LLMs 挑战了熵增定律**：一位成员认为人们误解了 **entropy**（熵）与 **information**（信息）之间的关系，声称一个 *bit* 遵循热力学定律；而另一位成员则表示运行智能合约会消耗并扩散能量，并没有战胜熵。
- **DeepSeek 研究员发布 Nano vLLM**：一位成员分享了 [GitHub 上的 nano-vllm 链接](https://github.com/GeeeekExplorer/nano-vllm/)，这是 **DeepSeek** 研究员的一个新项目。
- **模型推理质量取决于响应长度**：一位成员询问了推理模型的有效响应长度，以及在什么点其性能会崩溃。
   - 另一位成员建议参考 [Minimax M1 论文](https://arxiv.org/abs/2303.15698)，而另一位成员指出，当模型无法解决问题时，通常会生成冗长的 **CoT**（思维链），而如何体面地承认失败仍是一个未解决的问题。
- **AI Agent 的拟人化被证明很棘手**：一位成员寻求关于在日常对话中将 AI Agent 拟人化的建议，并指出尽管使用了递归提示词，**GPT** 听起来仍然过于正式。
   - 第二位成员建议使用基座模型（base model），但警告说这很难保持一致性，或者建议使用 embedding 模型来总结初始提示词。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1385747365276942397)** (24 条消息🔥): 

> `LLM training with negative information, Nous API token count, Function calling implementation across models, Best RP model, Wallet connection` 


- **负面信息会改变 LLM 输出吗？**：一位成员质疑使用大量的**负面信息**（战争、苦难、虚无主义、种族灭绝等）训练 **LLMs** 是否会产生更多**反社会响应**。
   - 该成员认为，即使指令微调（instruction tuning）试图对齐事实输出，LLMs 仍可能学会更有效地模拟有害行为。
- **Nous API 响应中的 Token 计数**：一位成员询问如何从 **Nous API** 的响应中获取 **token count**，另一位成员建议在请求体中添加字段 `"usage":true`，并提到响应默认应该返回 token 计数。
   - 该成员确认添加该字段有效并表示感谢。
- **关于 Function Calling 的讨论**：一位成员询问 function calling 在开源和闭源模型中是如何实现的，并对使用 **JSON** 作为函数参数（特别是针对代码等多行字符串）表示好奇。
   - 该成员以 [Mistral Small 3.2 的 function calling 实现](https://github.com/mistralai/mistral-common/blob/535b4d0a0fc94674ea17db6cf8dc2079b81cbcfa/src/mistral_common/tokens/tokenizers/instruct.py#L810) 为例，并询问 **Claude Code** 是如何避免 JSON 转义带来的性能问题的。
- **最佳 RP 模型？**：一位成员询问最佳的 **RP（角色扮演）模型**，并提到他们正在使用 **magnumv4 12b**。
   - 另一位成员指出 <#1366812662167502870> 是获取 API 和其他产品支持的地方。 
- **询问钱包集成**：一位成员询问为什么 NousResearch 官方聊天网站没有**钱包连接**选项，以及未来是否可能实现。
   - 另一位成员建议目前先使用 USD 为账户充值，并表示如果他们想使用“更多加密货币”也是可以的。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1385918285811290124)** (6 条消息): 

> `MCP System, Think Tank, Mesh Sharing, Data Tagging, Reward Models and Bias` 


- ****Think Tank** 是最值得关注的 **MCP****: 一位成员意识到他们独立重现了一个现有的 **MCP system**，名为 [Think Tank](https://smithery.ai/server/@flight505/mcp-think-tank)，它在推理、打标签（tagging）、记忆和编排（orchestration）方面表现出色，能够构建高效的 **ingestion engines**。
   - 这一发现强调了更小、更快模型的潜力，并指出 *真正的突破不在于更大的权重（weights）*，而在于增强的库和打标签能力。
- **得益于 **Think Tank**，**Mesh Sharing** 现在成为可能**: **Think Tank** 在库集成之前对输入进行分类和结构化的能力，可能会彻底改变 **LLM** 之间的 **mesh sharing**。
   - 一位成员热切地宣布，mesh 库很可能成为下一个前沿领域，而 **data tagging** 将成为下一个热潮。
- **Reward Models 也有偏差？！**: 分享了一个指向 [这篇论文](https://arxiv.org/abs/2506.07326) 的链接，表达了对人们在不考虑 **internal bias** 的情况下将 **reward models** 绑定到其流水线（pipeline）中的担忧。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1385699442979770551)** (62 条消息🔥🔥): 

> `Reputation in Discord, Joining OWL, Public Problem List, Language Diffusion Models, Prefix Caching` 


- **Discord 声誉授予**: 一位成员建议给另一位成员 *wave function* 授予声誉，因为他在 yannic 的 Discord 中表现活跃，并特别链接到了 [这条消息](https://discord.com/channels/729741769192767510/729741769738158194/1385694782067572797)。
- **新成员在查看消息后加入 OWL**: 一位新成员透露，他们在查看了另一位成员的消息后加入了 OWL（推测是一个频道），这促使另一位成员表示他无法提供建议，因为 *他已经了解得更清楚了*。
- **计划制定问题清单**: 一位成员计划建立一个公开的问题清单，并提到一些活跃的库也有待解决的问题（open issues），尽管其中大多数还没有准备好关于如何解决这些问题的风格指南（style guides）。
   - 另一位成员回应道：*那会很酷*。
- **通过滑动窗口深入研究 Language Diffusion Models**: 一位成员询问了关于使用滑动窗口方法研究 Language Diffusion Models 的情况，该方法定义了一个存储临时 token 的向量，并在每次迭代中进行细化。
   - 另一位成员链接了一篇关于 **Rolling Diffusion Models** 的相关 [arxiv.org 论文](https://arxiv.org/abs/2402.09470)，认为这可能是一个匹配项。
- **vLLM Prefix Caching 问题**: 一位成员询问是否有库像 **vLLM** 一样进行 prefix caching，但支持在缓存太大而无法放入 VRAM 或 DRAM 时将其存储在内存映射文件（memory-mapped file）中。
   - 另一位成员回答说，**这几乎肯定会比重新计算 KV 慢**，除非你的序列长度大于 1M。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1385695796594213044)** (25 messages🔥): 

> `Bottleneck Dimension Experiments, Token Pruning/Dropping Methods, Spectral Clipping, Imitation Learning in Racing Games, EAI Summer of Open AI Research` 


- **深入探讨瓶颈维度实验**：一位成员正在使用图像进行常规非量化瓶颈维度（bottleneck dimension）实验，记录了 **16384 的 codebook size** 并分享了 [loss graphs](https://cdn.discordapp.com/attachments/747850033994662000/1385696589774848064/CleanShot_2025-06-20_at_15.03.132x.png?ex=685af684&is=6859a504&hm=95064abf7d0e59beb2ea9f12ab21bb4df4f4b76f141825e91f8cf02f7a3f9395&)。
   - 他们观察到，如果 latent space 大于输入，任务就会变得简单，特别是在 optimizer step 为 **64** 时（参见 [此处](https://cdn.discordapp.com/attachments/747850033994662000/1385696909187879014/CleanShot_2025-06-20_at_15.04.312x.png?ex=685af6d0&is=6859a550&hm=7dd26830a813f8b495b894af2aedd34d11b5e659d9217696cf5a96aa2f93b761&)）。
- **Spectral Clipping：奇异值削减器**：一位成员分享了一篇博客文章（[链接](https://leloykun.github.io/ponder/spectral-clipping/)），解释了 **Spectral Clipping**、**Spectral ReLU** 和 **Spectral Clipped Weight Decay**，并澄清它只是*限制*奇异值（singular values），而不是像 Muon optimizer 那样将它们全部拉向 1。
   - 例如，阈值为 `beta=8` 的 *Spectral Hardcapping* 会将所有大于 8 的奇异值设置为 8，而 `alpha=4` 的 *Spectral ReLU* 对奇异值的作用类似于 ReLU。
- **业余模仿：赛车游戏获得提升**：一位成员报告了在赛车游戏中使用模仿学习（imitation learning）的实验，模型实现的单圈时间优于数据集中的表现，即使圈际方差（lap-to-lap variance）很高。
   - 这呼应了国际象棋中的发现，即在业余对局上训练的模型超越了玩家的 ELO 评分。
- **招募研究导师：EAI Summer 项目启动**：**EAI Summer of Open AI Research** 现已开放征集，寻求有经验的社区研究员为新人提出小型研究任务或项目。
   - 项目提案的截止日期为 **<t:1751839199>**，提案表格可以在 [此处](https://forms.gle/kHqQrs8uK65pNzXk7) 找到。
- **奇异值净化：谱参数化**：一位成员询问谱参数化（spectral parameterization，或 **Apple 的 sigma reparam**）的功能是否与 **Muon** 类似，即可能将奇异值推向 1。
   - 另一位成员澄清说，它类似于谱归一化（spectral normalization），即估计/近似谱范数（spectral norm）并将权重除以该范数。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1385936395251617833)** (4 messages): 

> `k-shot steering vectors, ACL paper on feature interaction, EAI Summer of Open AI Research, NNsight pre-release` 


- **K-Shot 设置中的 Steering Vector 技巧**：一位成员询问在 zero-shot 和 k-shot 设置中获取 steering vectors 时，是否应在 Difference in Means 句子中包含 **k-shot prompts**。
   - 该用户质疑是否应该为 zero-shot 和 k-shot 场景使用来自正例和负例的单一样本集。
- **新 ACL 论文重点介绍特征交互发现**：一篇新的 [ACL 论文](https://x.com/nsaphra/status/1933202363495370969) 探讨了**预测模型中的特征交互（feature interaction）**，以更好地理解数据集和科学现象的结构。
   - 该研究从 **LLM** 和语音模型开始。
- **EAI Summer of Open AI Research 征集项目提案**：**EAI Summer of Open AI Research** 已开放项目提案征集，寻求有经验的社区研究员为进入开放科学领域的人员建议小型研究任务。
   - 鼓励 8 月份有空的导师在 <t:1751839199> 截止日期前填写 [项目提案表](https://forms.gle/kHqQrs8uK65pNzXk7)。
- **NNsight 的下一个境界：NDIF 团队的预发布**：**NDIF 团队**正在预发布新版本的 **NNsight**，这是一个用于处理和干预 **PyTorch models** 的框架。
   - 感兴趣的用户可以通过 [Colab notebook](https://colab.research.google.com/drive/1wjQhbQKh2pwy-mxx4EFMBC1IEauzu9G0#scrollTo=ZuSXB8Bh1zEq) 进行尝试并提供反馈，Discord 活动将展示 **NNsight** 第二年的新功能并完成最终发布。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1386225087341924434)** (78 messages🔥🔥): 

> `HFLM model access for hooks, Log Likelihood Numbers, Llama3 GSM8k Reproduction, Lambada Target Token Issue` 


- **挂钩 (Hooking) HuggingFace 语言模型**：一位成员询问如何访问 `lm_eval.models.huggingface.HFLM` 的 `_model` 属性，以便应用 hooks 来修改模型输入和输出，特别是使用 `nn.Module.register_forward_pre_hook`。
   - 目标是将 hooks 应用于自定义模型，但该用户对 `lm_eval.models.huggingface.HFLM` 并不熟悉。
- **调试对数似然 (Log-Likelihoods) 和困惑度 (Perplexity)**：一位成员报告困惑度约为 **900k**，并询问 `_loglikelihood_tokens` 返回的对数似然 (LL) 数值是否是累加的。
   - 澄清了该函数返回负的 LL 值，且函数签名中的 `bool` 表示 Token 是否以贪婪 (greedy) 方式生成。
- **排查 Llama3 GSM8k 数值**：一位成员尝试在 **GSM8k** 上复现 **Llama3 8B** 论文中的数值，并使用了 `gsm8k_cot_llama`，但结果与报告的 **57.2** 准确率有偏差。
   - 建议尝试 `gsm8k_cot`，并澄清 `gsm8k_cot_llama` 取自 **Llama HF evals repo**，专门用于评估其 instruct model。
- **揭露 LAMBADA 中的分词故障**：一位成员发现 **LAMBADA** 有时会提供多个 Token 作为目标，导致求和后的 LL 值偏高。
   - 该问题导致困惑度飙升至 **~900k**，归因于目标序列的提取。为了缓解此问题，建议返回 Token 归一化的 LLs 或使用 [bits_per_byte](https://github.com/EleutherAI/lm-evaluation-harness/blob/68c3a811715ca86101f88c0044665bb70ad447f6/lm_eval/tasks/wikitext/wikitext.yaml#L14-L16) 进行归一化。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1386145346500952118)** (17 messages🔥): 

> `PSU and GPU Power, CUDA Server, GPU purchase 5070 vs 7800xt, Neutrino: Fine-grained GPU Kernel Profiling, Code Readability & Const Variables` 


- **PSU 电源动态讨论**：一位成员询问是否可以为带有 **3x8pin** 接口的 **RTX 3080ti** 使用 **12V-2x6 cable**，并质疑将其与普通的 **8-to-8 PCI cable** 混合使用是否安全，且不会导致 GPU 过载。
   - 另一位成员安慰说 GPU 只会抽取所需的功率，PSU 不会推送额外功率，认为这种配置可能是安全的。
- **服务器是专注于 CUDA 还是通用的 GPU 计算中心？**：一位成员询问该服务器是否主要讨论 **CUDA**，因为他们正在寻找讨论 **compute shaders** 性能优化的地方。
   - 另一位成员澄清说，服务器已从 "CUDA Mode" 更名为 "GPU Mode"，以涵盖除 CUDA 之外的各种计算平台，并指出了针对不同平台和兴趣讨论的特定频道。
- **5070 vs 7800xt：价格合适**：一位成员询问是购买 **530€** 的 **5070** 还是 **450€** 的 **7800xt**，寻求该价格范围内的最佳选择建议。
   - 未给出回答或建议。
- **Neutrino：用于 GPU Kernel 的 eBPF**：一位成员宣传了 **Neutrino**，这是一个[细粒度 GPU Kernel Profiling 工具](https://www.usenix.org/conference/osdi25/presentation/huang-songlin)，已被 USENIX OSDI '25 接收，它允许通过汇编级 (Assembly-level) 探测 GPU Kernel，类似于 **eBPF**。
   - 该工具支持运行时信息公开，并具有密集内存访问时间线 (DMAT)，以直观了解 GPU Kernel 访问密度，并提供了 [GitHub repo](https://github.com/open-neutrino/neutrino) 和 [文档](https://open-neutrino.github.io)。
- **可读性至关重要：Const 还是注释？**：在关于代码中幻数 (magic numbers) 的讨论中，一位成员建议使用 **CONST variables** 以提高可读性和前瞻性。
   - 另一位成员认为这个建议*太费事*，并承认为了方便（无论好坏）将所有内容都放在一个文件中。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1386301580365529108)** (3 messages): 

> `Triton AOT Compile, Triton type hints` 


- **Triton Block Size 足够**：一位成员建议用户通常不会遇到 block size 的问题，因为它*足够大*，并引用了他们的 [代码](https://github.com/OpenMLIR/LeetGPU/tree/main/12-softmax/Triton) 作为参考。
- **Triton 张量分配受到好评**：一位成员发现*张量的运行时分配 (run-time allocation)* 非常有用，并对非分配版本中每个 block 都要读取整个向量感到遗憾。
   - 他们认为这是又一个需要学习使用的模式。
- **AOT 编译期间 Triton 类型提示的问题**：一位 Triton 新用户正在寻求 AOT 编译方面的帮助，特别是如何在 `_attn_fwd_inner` 函数中对 `q` 张量进行类型提示，参考了 [fused attention kernel 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。
   - 他们强调标准的 `str_to_ty` 函数仅支持 `pointer, tensordesc, constexpr`，并询问是否有人处理过这个问题。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1385717969308352523)** (19 messages🔥): 

> `Nsight for CLion, memcpy_async details, control divergence, GFLOPS calculation, Nsight compute` 


- **Nsight 受到关注，仍期待 CLion 支持**：成员们讨论了 **Nsight**，以及带有 Nsight 扩展的 VS Code 是 **GUI 调试** 的一个好选择。
   - 一位成员建议，如果有足够多的用户请求 **CLion** 支持，Nsight 的开发者可能会考虑。
- **揭秘 Warp 速度的 memcpy_async**：一位用户对 `memcpy_async` 感到困惑，特别是关于 `thread_id` 参数，参考了 [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies)。
   - 另一位成员澄清说，索引仍然依赖于 `threadIdx`，并指向一篇 [NVIDIA 博客文章](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/#overlapping_global-to-shared_copies_with_compute) 作为示例。
- **分歧路径？SIMT 来救场！**：一位用户询问在单个 kernel 中，不同的线程是否可能根据 `thread ID` 执行不同的代码路径。
   - 一位成员解释说，这被称为**控制分歧 (control divergence)**，是 **SIMT** 编程模型的一个优势，GPU 会高效地跳过 warp 内未使用的代码路径。
- **GFLOPS 和带宽基准测试盛宴**：一位用户询问如何正确计算 **GFLOPS** 和**带宽 (bandwidth)**。
   - 一位成员建议使用 **profiler (ncu)** 以获取准确数值，并提供了一种手动计算方法，涉及算法的计算量或读取/写入的字节数除以运行时间。
- **Nsight Compute 揭示 Roofline 秘密**：一位用户报告在 **RTX 3070 Ti** 上使用 Nsight Compute 得到了 **40% 的 SM 吞吐量**，计算出峰值 **21.75 TFLOPS** 中的 **8.75 TFLOPS**，并质疑其计算的准确性。
   - 另一位成员指出，**Nsight Compute** 可以在 **Rooflines** 部分提供实际的 **FLOP/s** 值，并链接到了 [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#details-page)。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1385831501698039932)** (8 条消息🔥): 

> `torch.clip 的 PyTorch 梯度计算，嵌入式系统的量化感知训练，在 torchtitan 中捕获集体通信图，titan 中的 SimpleFSDP 实现，inductor 中的自定义图传递 (graph passes)` 


- **PyTorch 的 `torch.clip` 梯度详解**：一位用户询问了 PyTorch 中 `torch.clip` 的梯度计算，特别是为什么在提供的示例中 `half.grad` 返回 **37**。
   - 另一位用户链接到了[相关的 PyTorch 源代码](https://github.com/pytorch/pytorch/blob/1d993fa3092e4f0b5745f2470024b35cac96da14/torch/csrc/autograd/FunctionsManual.cpp#L1212-L1248)，解释说 *如果 min=max，那么 min 的梯度为零*。
- **嵌入式系统量化问题得到修复**：一位用户修复了其代码中与嵌入式系统量化感知训练 (quantization-aware training) 相关的问题，澄清了 `x_round` 本质上就是 `x`。
   - 该用户表示，这段代码将部署在没有 Torch autograd 功能的嵌入式系统上，强调了自定义梯度计算的需求。
- **TorchTitan 简化了集体通信图捕获**：为了捕获集体通信图 (collective communication graphs)，特别是用于原型编译器开发，一位用户建议使用 [torchtitan](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/simple_fsdp/README.md) 中的 **SimpleFSDP 实现**。
   - 他们提到最近在 SimpleFSDP 版本中添加了 **TP**，从而支持编译同时包含 **TP** 和 **FSDP** 集体通信的图，并引用了[这个 pull request](https://github.com/pytorch/torchtitan/pull/1250)。
- **Inductor 为自定义图传递 (Graph Passes) 敞开大门**：对于那些希望使用 inductor 完整编译栈并加入计算/通信重叠 (compute/comms overlap) 自定义逻辑的用户，一位用户提到 inductor 中存在（目前为私有的）钩子 (hooks)。
   - 具体来说，他们强调了注册在 ATen 图上运行的 post_grad pass 的能力，允许使用自定义算法进行分桶 (bucketing) 或通信排序，并引用了[相关的配置文件](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L262)。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1385724702609113118)** (4 条消息): 

> `并行算法，矩阵操作，排序算法` 


- **排序算法层出不穷**：一位成员建议了 **Bubble Sort** 和 **Bogo Sort**，并提到了更多并行算法，如 **stencil**、**reduce**、**scan** 和 **histogram**，参考了 **PMPP** 一书。
- **矩阵操作即将到来**：一位成员提到将继续推进 **矩阵-向量** 和 **矩阵-矩阵乘积**。
   - 该成员指出，从事机器学习或 LLM 的人可能会选择 **softmax**、**量化 (quantizations)**、**注意力机制 (attentions)** 等。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1385743010272116826)** (8 条消息🔥): 

> `CUDA 非法内存访问，Triton vs CUDA 学习资源，SYCL 信息` 


- **CUDA 代码片段触发非法内存访问**：一位用户发布了一个尝试并行归约 (parallel reduction) 的 [CUDA 代码片段](https://github.com/example/cuda_code)，并遇到了非法内存访问错误。
   - 一位成员询问了 `input` 和 `output` 的分配情况，认为 `input` 可能是一个经过 `cudaMalloc` 的数组，并询问 `output` 是否是单个 `cudaMalloc` 的 float；另一位成员指出 blocksPerGrid 的计算可能有误。
- **Triton vs. CUDA：新手询问从何处开始**：一位 CUDA/Triton 新手表示对 Google 搜索结果感到不知所措，并请求关于从何处开始学习的建议，重点关注并行模型训练和 LLM 推理优化。
   - 一位成员建议，虽然 **CUDA** 比 **Triton** 更难，但它拥有更多对初学者友好的资源，并推荐将 *Programming Massively Parallel Processors* 一书作为 CUDA 的入门指南。
- **SYCL 信息匮乏**：一位成员询问了关于 **SYCL** 的资源，提到他们已经完成了 **DPCPP** 的设置，但很难找到关于实际 SYCL 编码的信息。
   - 该成员表示：*我使用 Gemini，在哪里可以找到关于 SYCL 的信息？我只找到了 DPCPP 的安装文档，我已经完成了，但我找不到任何关于如何实际编写代码的内容。*


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1386290410837774348)** (19 messages🔥): 

> `mi300x profiling, chisel-cli, rocprof integration, rocprofiler-sdk, nsight-compute` 


- ****Chisel CLI** 旨在实现本地 **mi300x Profiling****：一名成员介绍了 **Chisel CLI**，这是一个专为本地 **mi300x profiling** 设计的工具。它可以启动 **AMD Cloud mi300x** droplets 实例，同步代码，使用 **rocprof** 进行 profiling，并自动将结果获取到本地。该工具可通过 [GitHub](https://github.com/Herdora/chisel) 获取，或通过 `pip install chisel-cli` 安装。
   - 未来计划包括 **Grafana 集成**、并发运行、更好的错误处理以及多云支持。
- ****rocprof** 赋能细粒度代码 Profiling**：该工具目前使用原生的 **rocprof** 功能进行 kernel/ops 级别的 profiling，并计划很快通过 **rocprof 的硬件计数器**或自定义插桩（instrumentation）添加 block/tile 级别的 profiling。
   - 集成 [rocprofiler-compute](https://github.com/ROCm/rocprofiler-compute) 是系统性能分析的首要任务。
- **分享 **rocprofiler-sdk** 集成技巧**：一名成员分享了新版 **ROCm profiling tools**（[rocprofiler-sdk](https://github.com/rocm/rocprofiler-sdk) 和 [rocprof-compute-viewer](https://github.com/rocm/rocprof-compute-viewer)）的设置说明，并指出目前需要手动修复。
   - 设置过程包括从 mainline 分支构建 **aqlprofile** 和 **rocprofiler-sdk**，下载 **rocprof-trace-decoder** 二进制文件，设置 `ROCPROF_ATT_LIBRARY_PATH` 环境变量，以及从 mainline 分支构建 [aqlprofile](https://github.com/ROCm/aqlprofile)。
- ****nsight-compute** 的挑战引发了对 Nvidia 工作流的兴趣**：一名成员提到处理 `nsight-compute` 可能非常繁琐，引发了关于支持 **Nvidia 工作流** 的讨论。
   - 另一位主要关注 **AMD kernel 开发** 的成员表示，如果有足够的需求，有兴趣在 **Nvidia** 支持方面进行合作。


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/)** (1 messages): 

tri_nitr0_t0luene: 在哪里可以找到关于如何为 GPU 编写 oneAPI SYCL 代码的文档？
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1386039642595266701)** (5 messages): 

> `Fibonacci GPU Calculation, NVIDIA Thrust Library, MI300X Profiling Tool Chisel, CuTeDSL Introduction, NVIDIA CUTLASS Team` 


- **眨眼间完成斐波那契数计算**：一篇新的博客文章展示了如何使用 **NVIDIA Thrust 库** 的 Scan 操作，在消费级 GPU 上仅用 **17 毫秒** 计算 **1 亿个斐波那契数**。代码已在 [GitHub](https://github.com/simveit/fibonacci_gpu/tree/master) 开源。
   - 该博客文章从 [Guy Blelloch 的论文](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf) 中汲取灵感，旨在成为基于 GPU 的斐波那契计算的教学范例。
- **使用 Chisel 实现 MI300X 本地 Profiling**：**Chisel CLI** 允许通过启动每小时 **$1.99** 的云端 droplets 实例来进行本地 **AMD MI300X** profiling。它可以自动同步代码、使用 *rocprof* 进行 profiling 并获取结果；可通过 [GitHub](https://github.com/Herdora/chisel) 的 `pip install chisel-cli` 安装。
   - 工具作者正在考虑 Grafana 集成、并发运行和多云支持，并寻求社区反馈，特别是来自那些在 **MI300X** 上进行 kernel profiling 的用户的建议。
- **CuTeDSL 入门深度解析**：一篇博客文章介绍了 **CuTeDSL**，这是来自 **NVIDIA CUTLASS 团队** 的一种领域特定语言（DSL）。它允许通过 **CuTe layout 抽象** 和 **Python 语法** 在硬件控制下表达 GPU kernel。该博文深入探讨了 [NVIDIA CUTLASS 团队仓库](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL) 中的示例。
   - 之前关于 **CuTe 代数** 数学原理的博客文章可以在[这里](https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/)阅读。


  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1386799896157225040)** (4 messages): 

> `OSS datasets, Pytorch, Triton, KernelBot` 


- **KernelBot 数据集为 Triton 浮出水面**：一位成员询问关于 **Pytorch 2 Triton** 的最佳 OSS 数据集，另一位成员回复称目前可用的真人编写 **Triton** 数据并不多。
   - 该成员强调了他们创建的 **kernelbook** 以及新的 [KernelBot 数据集](https://huggingface.co/datasets/GPUMODE/kernelbot-data)，其中包含针对一些问题的真人编写示例。
- **Pytorch to Triton 期待优质数据集**：一位成员询问了最适合 **Pytorch 2 Triton** 转换的开源数据集。
   - 回复指出网上缺乏真人生成的 **Triton** 数据，解释了创建 **kernelbook** 的必要性。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1386200466991353897)** (5 messages): 

> `VRAM Requirements, KL Loss, FP32 Training, A6000 GPUs, Reasoning Gym` 


- **Reasoning Gym VRAM 需求揭晓**：成员们讨论了运行 [Reasoning Gym](https://github.com/open-thought/reasoning-gym/) 训练脚本的 VRAM 需求，一位用户询问运行 `train_grpo.py` 脚本需要多少 VRAM。
   - 另一位成员表示，对于 **3B 参数模型**，实验使用了 **4xA6000 GPUs**，总计 **192GB VRAM**，但可能可以使用更少。
- **KL Loss 可以节省 VRAM**：一位成员提到，禁用 **KL loss** 将避免加载参考模型，从而节省一些 VRAM。
   - 该成员还指出，在他们的设置中可以使用 **fp32** 进行训练，这意味着使用 **bf16** 时所需的 VRAM 会更少。
- **FP32 精度确认**：成员们确认由于文档不清晰，最初使用 **fp32** 进行训练，后来意识到使用 **bf16** 降低 VRAM 占用的潜力。
   - 对话强调了在训练大模型时精度与内存占用之间的权衡。


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1386582010558021672)** (3 messages): 

> `Chinese speakers in the channel, Multilingual AI research community, GPU Mode` 


- **中文使用者聚集在 GPU Mode**：一位用户询问为什么 **GPU Mode** 频道中有很多中文使用者。
   - 另一位用户回答说，*许多机器学习研究者都说中文*，因此他们聚集在这个频道。
- **AI 研究者的语言偏好**：该频道吸引了一个机器学习研究者社区，其中许多人是中文使用者。
   - 这创建了一个中心，让他们可以用自己偏好的语言进行交流和协作，营造了一个更具包容性和高效的环境。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1386177662602580099)** (23 messages🔥): 

> `MI300, H100, amd-fp8-mm Leaderboard, Grayscale Leaderboard, Histogram Leaderboard` 


- **MI300 获得 AMD-FP8-MM 提交**：在 **MI300** 上向 `amd-fp8-mm` Leaderboard 提交的结果成功达到 **931 µs**。
- **H100 Grayscale 获得更多提交**：在 **H100** 的 `grayscale` Leaderboard 上实现了多次成功提交和个人最佳成绩，范围从 **1458 µs** 到 **6.11 ms**。
- **H100 Histogram 竞争升温**：在 **H100** 的 `histogram` Leaderboard 上记录了多次成功提交和个人最佳成绩，包括以 **41.2 µs** 获得第 5 名。
- **H100 Matmul 冲至 🥉 第三名**：一次提交在 **H100** 的 `matmul` Leaderboard 上以 **253 µs** 的成绩获得 🥉 第三名。


  

---

### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1386552289501909144)** (2 messages): 

> `TPU Interaction, XLA Compiler, Pallas, StableHLO` 


- **TPU 交互方法探索**：Google 的 TPU 最好通过 **XLA compiler** 进行交互，因为最低层级的指令集存在于部分文档化的 *libtpu.so* 中。
   - 在更高层级，可以使用 **Jax** 或 **Torch/XLA**，它们会编译为 **StableHLO** ([https://openxla.org/stablehlo](https://openxla.org/stablehlo))，然后再编译为设备相关的 MLIR passes。
- **Pallas 提供低层级 Kernel 编码**：**Pallas** ([https://docs.jax.dev/en/latest/pallas/tpu/index.html](https://docs.jax.dev/en/latest/pallas/tpu/index.html)) 为 TPU 提供了低层级 Kernel 编码选项，可从 **Jax** 和 **Torch/XLA** 访问（参见 [Torch/XLA's Pallas Kernels](https://github.com/pytorch/xla/tree/master/torch_xla/experimental/pallas_kernels)）。
   - 尽管处于活跃开发中且功能不断扩展，**Pallas** 可能仍不完整；一场 [GPU Mode 讲座](https://www.youtube.com/watch?v=wKd90avC8Nc) 从 GPU 的视角提供了见解。
- **将 StableHLO 操作作为最后手段**：直接操作生成的 **StableHLO** 是一个选项，尽管这被认为是迫不得已的措施。
   - 工程师通常不需要深入到 TPU **Mosaic** 层级，该层级未公开文档，但属于 XLA 内部设备相关的 passes。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1385724705834799307)** (22 messages🔥): 

> `Self-Generating Tasks, Auto Verifiers, Factorio Source Code, Factory Bug Fixes` 


- **自生成任务引发关注**：成员们讨论了近期关于通用 Agent 任务自生成的工作，并分享了多篇相关论文的链接 ([1](https://arxiv.org/pdf/2506.01716), [2](https://arxiv.org/pdf/2505.23762), [3](https://arxiv.org/pdf/2506.10055), [4](https://www.arxiv.org/pdf/2506.14205))。
   - 一位成员指出，如果他们能弄清楚如何验证更广泛任务的成功（不仅是吞吐量），那么他们可能会挖掘到一个金矿。
- **Factorio 的自动验证器设计具有挑战性**：成员们讨论了应该如何构建 **environment**（环境），使其能够自动验证提出的挑战，这可能是最困难的部分。
   - 自动验证器的定义和结构将是创建过程中的挑战，另一位成员建议检查吞吐量 (SPM) 是最简单的验证形式。
- **探索 Factorio 源代码访问的益处**：团队讨论了 **Factorio** **source code access**（源代码访问）的潜在好处，包括更快的开发和更好的集成，并表示他们将把 gym PR 合并到主仓库中。
   - 有人提到，任务设置最大的困难在于验证器。
- **工厂 Bug 修复作为有趣的任务**：成员们探索了预设地图产出 X 数量物品的想法，提议者指定对工厂的更改以有效地引入 Bug，然后由求解器进行修复，这可能是让求解器处理特定 Bug 的一种详细方式。
   - 他们观察到定义“更好”是主观的，但模型可能会通过 **training wheel scenarios**（辅助轮场景）发现它。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1386367999526703284)** (2 messages): 

> `CuTeDSL PTX and sass code Emission, Cutlass Future Releases` 


- **CuTeDSL 准备进行 PTX 发射**：Cutlass 团队计划在未来版本中支持为 **CuTeDSL** 发射 **PTX** 代码；不过，**ETA**（预计完成时间）尚未确定。
- **Cutlass 将在未来版本中打印 PTX**：Cutlass 团队计划在未来版本中打印 **PTX**，更多信息请参见 [此 GitHub issue](https://github.com/NVIDIA/cutlass/issues/2302#issuecomment-2886934868)。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1385700036985356350)** (88 条消息🔥🔥): 

> `Benchmarking minimax/minimax-r1, Claude Code Sonnet 限制, Aider 上下文管理, Mcpm Aider 工具, Gemini 代码重写` 


- **Aider 用户希望将 minimax 模型添加到基准测试中**：一位用户请求将 `minimax/minimax-r1` 添加到 Aider Polyglot 排行榜，并指出其开源特性以及与 `anthropic/claude-sonnet-4` 和 `openai/o3-mini` 相比具有竞争力的性能。
   - 该用户认为[在公共仓库中进行基准测试是一个错误](https://aider.chat/docs/benchmarks.html)，并建议为每个结果添加“最后更新”日期。
- **Claude 的 Code Sonnet 速率限制**：一名成员测试了 **Claude Code Sonnet** 的限制，并在使用 Opus 时达到了速率限制，并提到在使用大量 subagents 时也遇到了此问题。
   - 成员们建议查看 **source code** 以寻找上下文管理解决方案。
- **Aider 用户建议进行上下文手术和管理**：成员们讨论了在 Aider 中改进 **context management** 以避免高昂成本的必要性，认为 `/clear` 命令的范围太广。
   - 一位成员提议使用 shell 编辑器和对话历史容器，为对话历史手术（convo history surgery）提供一个 *inline vim 编辑器*。
- **Copilot 中的 Mcpm-aider 工具调用**：成员们讨论了在 **Copilot** 中使用和修改 **mcpm-aider**，指出其较为笨重，但建议直接对 Aider 本身进行修改。
   - 该建议涉及通过添加一个强制性的工具调用 *Get user input* 来 **欺骗 Gemini 2.5 Pro** 以获取更多请求。
- **Aider 用户寻求为 HTML 文件应用补丁的建议**：一位用户需要将 12 个补丁文件应用到约 400 个 HTML 文件中并寻求建议，因为他们目前使用 **Claude APIs** 的脚本由于 Token 限制和处理缓慢而失败。
   - 该用户正在寻找类似于 **Cursor** 的解决方案，即分块处理更改而不将整个文件加载到上下文中，并询问 Aider 是否适合此任务，特别是如何让 `aider` *完全以无头模式 (headless)* 运行（仅提供输出和日志）。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1385747966643536014)** (21 条消息🔥): 

> `aider 跳过编辑, Aider 交互指南, Claude 4 Sonnet 未遵循 CONVENTIONS.md, 加载自定义 TypeScript 库, 恢复 /undo 命令` 


- **Aider 在空的 Git 仓库中跳过编辑**：一位成员报告称 Aider 跳过了对空 Git 仓库中文件的编辑，此行为的原因尚不明确。
   - 他们使用了标准参数，包括指定 **AWS_REGION**、从规范文件读取以及设置模型参数。
- **构建 Ask-Ask-Code 工作流**：一位用户分享了他们开发的 **Aider Interaction Guidelines**，旨在利用 Gemini 实现 *ask-ask-code* 工作流，强调澄清、规划、审查和简洁的更改。
   - 这些指南放置在 `AIDER.md` 文件中，指示 AI *提出澄清问题*、*提出计划*、*等待用户批准*并*交付简洁的更改*。
- **使用 Read 标志加载规范**：一位成员报告了 **Claude 4 Sonnet** 不遵守通过 `-read CONVENTIONS.md` 传递的 `CONVENTIONS.md` 文件的问题。
   - 另一位成员澄清说，最好使用 `/read CONVENTIONS.md` 或 `aider --read CONVENTIONS.md` 以确保文件被视为只读并被缓存，并指出[文档错误](https://aider.chat/docs/usage/conventions.html#example)中缺少了一个 `-` 字符。
- **需要加载自定义 TypeScript 库**：一位用户询问如何将 `node_modules` 中的自定义 **TypeScript library** 加载到 Aider 上下文中，以防止模型捏造不存在的方法和参数。
   - 单独加载每个文件被认为是不切实际的。
- **从 Undo 中恢复**：一位用户寻求从意外的 `/undo` 命令中恢复的指导，解决方案是使用 **git reflog** 找到最近的提交并重置到特定的 commit hash。
   - 他们建议在 Aider 中加入 `/redo` 命令将是一个*极好的改进*。


  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1386666743879045191)** (9 messages🔥): 

> `Claude Code API, Anthropic Subsidization, Terms of Service` 


- **Claude Code API 被考虑作为 Aider 后端**：一名成员建议使用 **Claude Code** 作为 **Aider** 的后端，通过 [claude-code-api GitHub repo](https://github.com/codingworkflow/claude-code-api) 利用其订阅模式，从而实现更便宜的调用。
- **据称 Anthropic 补贴了 Claude Code**：一位成员报告称，通过每月 **$20** 的 **Claude Code PRO** 订阅，用户可以轻松超过每天等效 **$10-20** 的 API 调用费用，并分享了一张图片，显示 30 天内等效 API 使用量超过 **$1200**，这意味着 *Anthropic 相比 API 使用对 Claude Code 进行了补贴*。
   - 一名成员认为这很酷，而另一名成员则对其服务条款 (TOS) 表示疑虑。
- **Claude Code 的服务条款受到质疑**：讨论围绕 **Claude Code** 的服务条款 (TOS) 展开，即是否允许在 **Aider** 等另一个服务之后使用该工具。
   - 一名成员想知道该工具在这种条件下会如何表现。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1385905904595112046)** (67 messages🔥🔥): 

> `Tinygrad backward time, AMD GPU instability, IO_uring ZCRX DMA-BUF, tinygrad server, NVMe driver in userspace` 


- **反向传播耗时数小时**：一名成员报告称，在一个带有 **TinyJit** 的 **17M 参数模型**上，`.backward()` 耗费了数小时，不确定这是否正常。
   - 虽然没有提供解决方案，但该问题被提出作为一个潜在的性能瓶颈。
- **AMD GPU 不稳定性困扰测试**：一名开发者报告称，`modprobe amdgpu` 经常导致机器崩溃，在 AMD GPU 上进行测试需要重启。
   - 这种不稳定性可能与 **Ubuntu 24.04** 有关，使得在 **AMD GPU** 上的测试变得异常困难。
- **讨论 IO_uring ZCRX DMA-BUF 集成**：成员们讨论了集成 [IO_uring ZCRX DMA-BUF](https://www.phoronix.com/news/IO_uring-ZCRX-DMA-BUF) 以支持传递 DMA-BUF 缓冲区，重点是实现 GPU 到网卡的直接拷贝。
   - 该特性计划在 **Linux 6.16** 中推出，扩展了 io_uring 以支持零拷贝传输，并且被认为回传 (backport) 起来 *相当简单*。
- **构思用于远程 GPU 访问的 "Tinygrad server"**：提出了 *tinygrad server* 的想法，作为导出 GPU BAR 的轻量级解决方案，可能实现为一个 **4kloc 裸机 C** 程序。
   - 该服务器将设置 **Mellanox** 并导出每个 **PCI 设备**，从而在无需内核参与的情况下通过 RDMAIface 实现远程访问。
- **考虑编写用户态 NVMe 驱动以实现直接磁盘访问**：讨论围绕编写用户态 NVMe 驱动以实现直接磁盘访问展开，从而可能实现 `DISK:/dev/nvme0` 寻址。
   - 虽然内核模块更简单，但用户态驱动提供了更多控制，[Redox OS NVMe 驱动](https://gitlab.redox-os.org/redox-os/drivers/-/tree/master/storage/nvmed) 被引用作为参考。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1385854183789297675)** (38 条消息🔥): 

> `Tinygrad 异步数据传输, Tinygrad 中的 RNN 性能, LSTM 性能, 单元测试愿望清单, 设备可用性检查失败` 


- **Tinygrad 缺乏类似 PyTorch 的异步数据传输**：一位成员询问 Tinygrad 是否有类似 PyTorch 中 `x.to(device, non_blocking=True)` 的异步数据传输功能，以便重叠计算和数据传输。
   - 他们随后进行了耗时测试，发现 `.realize()` 似乎是一个阻塞操作，对此其他人建议使用 `Device[Device.DEFAULT].synchronize()`。
- **在 M1 上处理长序列时 RNN 性能骤降**：有成员报告称，在 M1 和 Intel Mac 上，Tinygrad 训练 RNN（LSTM 和 GRU）在处理较长序列（例如长度 256，特征数 32）时表现出较差的性能。
   - 另一位用户提到 **LSTM 性能** 普遍较慢，即使是单单元（single-cell）执行，与原始的 C 语言重写相比也是如此，因此他们渴望看到性能改进和示例。
- **Tinygrad 更多单元测试的愿望清单**：一位成员提出愿意贡献包含额外单元测试的小补丁，以便更熟悉 Tinygrad。
   - 有用户建议他们为自己的库贡献代码，该库包含 tinygrad 的额外功能，地址为 [https://github.com/softcookiepp/tinybloat](https://github.com/softcookiepp/tinybloat)。
- **`python3 -m tinygrad.device` 报错**：两位成员报告称运行 `python3 -m tinygrad.device` 会导致 `RuntimeError: no usable devices` 回溯。
   - 一位用户提供了一个临时解决方案，使用 `python3 -c "from tinygrad import Device; print(list(Device.get_available_devices())); print(Device.DEFAULT)"` 来检查设备可用性。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1385696295301414983)** (95 条消息🔥🔥): 

> `MCP 服务器, Scarlet AI 重写, Google Workspace 自动化, AI 时间线, ElevenLabs 11ai` 


- **MCP OS 助力 CEO 生产力飙升！**：一位成员报告称，通过使用他们的 **MCP OS**，利用超过 **95%** 自主的 Claude 代码自动化 Google Workspace 任务，CEO 的生产力大幅提升，并对使用 [MCP OS](https://example.com/mcp-os) 表示兴奋。
   - 他们建议创建一个作为 *"MCP OS"* 运行的新仓库，并使用 Linear、Markdown 文件或带有 Elasticsearch 和 Agentic RAG 的数据库来轻松添加上下文。
- **ElevenLabs 推出 11ai：支持 MCP 的语音优先助手**：**ElevenLabs** 推出了 [11ai](https://11.ai)，这是一款语音优先的 AI 助手，支持 **MCP**，并在 ElevenLabs 的低延迟对话式 AI 平台上集成了 Perplexity、Linear 和 Slack。
   - 一些用户推测它可能使用了 **GPT-3.5** 或更小的 **Llama** 模型。
- **Harvey AI 获得 3 亿美元 E 轮融资**：**Harvey AI** 完成了 **3 亿美元** 的 E 轮融资，公司估值达到 **50 亿美元**，由 Kleiner Perkins 和 Coatue 领投，Sequoia、GV 和 OpenAI Startup Fund 参投，并与 [LexisNexis](https://www.lexisnexis.com/community/pressroom/b/news/posts/lexisnexis-and-harvey-announce-strategic-alliance-to-integrate-trusted-high-quality-ai-technology-and-legal-content-and-develop-advanced-workflows) 建立了合作伙伴关系。
   - 用户们向该公司表示祝贺，但也有人对该估值表示怀疑。
- **Replit 年经常性收入 (ARR) 突破 1 亿美元**：**Replit** 宣布其 ARR 超过 **1 亿美元**，并将其归功于客户和支持者，引发了广泛祝贺。
   - 一位成员分享了关于 Agent 监督、Agent 漂移以及企业级 *"Agent 规模化悬崖"* 的见解，并链接到了 [这条推文](https://x.com/MatanPaul/status/1937200395115499592)。
- **初创公司必须在分发上胜出，而不仅仅是创新**：讨论强调了初创公司与现任者之间的竞争：初创公司能否在现任者创新之前实现分发，参考了 [这条推文](https://xcancel.com/aleximm/status/1937251084810219721)。
   - 对话强调了分发的力量，指出 OpenAI 与 Google 相比具有极快的用户获取速度。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1385702753116094535)** (12 messages🔥): 

> `GestaltView Ecosystem, NotebookLM as a Strategic Partner, Podcast Language Expansion, Solicitation Guidelines` 


- **NotebookLM 优化 GestaltView Ecosystem**：一位成员对 **NotebookLM** 在优化和增强 **GestaltView Ecosystem** 方面的战略合作伙伴关系表示感谢，这使得对其知识库的理解更加连贯。
   - 他们提到 **NotebookLM** 帮助他们识别并填补了空白，确保了在解释和基于事实的发现中的一致性和彻底性，并感谢它在应对与创新相关的心理健康挑战方面提供的支持。
- **展示 NotebookLM 作用的图片**：分享了几张图片，将 **NotebookLM** 描绘为战略合作伙伴，并展示了其在思维导图（mind mapping）中的应用。
   - 一项图片分析建议将思维导图保存为 **PDF**。
- **请求扩展播客语言**：一位成员询问了是否可以用英语以外的语言制作更长的播客，并特别请求提供韩语摘要（**요약해줘**）。
   - 他们还提到，*将更改对话风格作为提示词启动器（prompt starter）是一个巨大的优势*。
- **关于招揽指南的讨论**：一位成员对招揽/推销行为是否获得批准表示怀疑，但注意到缺乏针对此类行为的具体指南。
   - 另一位成员澄清说，虽然单纯的招揽行为会导致封禁，但在积极参与讨论的过程中分享相关链接是可以接受的。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1385731857508864020)** (76 messages🔥🔥): 

> `AI Engineering Study Tips with NotebookLM, NotebookLM vs Gemini, Audio Overview Limits, Image analysis, Gemini Model selection` 


- **新 AI 工程师寻求 NotebookLM 学习建议**：一位即将开始学习 **AI Engineering** 的成员询问了如何更好地利用 **NotebookLM** 进行学习，并得到了建议：*使用优秀的提示词来发现 AI Engineering 的来源和 PDF，并建立选定的来源进行对话或制作音频概览（audio overview）*。
- **Gemini 与 NotebookLM 展开关于 Grounded 的辩论**：用户讨论了 **NotebookLM** 相对于 **Gemini** 的意义，一位成员指出 *单独使用 Gemini 不会遵循 Grounded（基于事实）原则，而 NLM 强制执行该原则*，这意味着回答完全基于提供的来源。
   - 其他人提到 **Gemini** 可能不会将其知识仅限于附加文档，而 **NotebookLM** 提供了项目组织功能，如保存笔记、思维导图和播客，同时能更可靠地处理更多文件。
- **最常见话题 PDF 现身**：一位成员发布了一份关于 Discord 频道中最常见话题的 **PDF** 分析，但具体话题在提供的上下文中未详细说明；Discord 中包含的 PDF 在此处：[2025-06-20_Most_Common_Topics.pdf](https://cdn.discordapp.com/attachments/1385985451617292399/1385985702872748124/2025-06-20_Most_Common_Topics.pdf?ex=685ab245&is=685960c5&hm=f66b1a6e5f2b667eb984297d355f557ce077ec447e323453a82759815a819c18)。
- **播客功能引起关注**：成员们正在使用 **podcast** 部分为 TikTok 制作 5 分钟的“热门话题”播客，一位用户询问了关于如何最好地自定义播客的深入信息。
   - 一位用户注意到 App 版本在制作超过一两个播客时会有成本限制，但网页版允许一天内制作多个免费播客，这表明 App 和网页版之间存在差异。
- **图像分析功能揭晓**：用户讨论了 **NotebookLM** 是否可以分析 PDF 中的图像，一位成员分享了一张架构图，显示 **NLM 在将来源发送给 Gemini 之前会进行预处理** [Architecture_of_NotebookLM.pdf](https://cdn.discordapp.com/attachments/1385977346733113415/1386016041947365416/Architecture_of_NotebookLM.pdf?ex=685ace87&is=68597d07&hm=da3730a0ae34178cd4d17b5392f93f5ced0c9d05ec1a65d050c6b1a2ca1810e1)。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1385841468899594280)** (23 条消息🔥): 

> `小型 LLM 的 NLP 启动，基于 YOLO 的反无人机检测，斯坦福 AI 资源，验证推理轨迹，VLM 研究` 


- **小模型在推理复现方面表现不佳**：一位成员建议复现论文 *Small Models Struggle to Learn from Strong Reasoners*，使用 **Unsloth** 来降低 **1.5B LLM** 的 VRAM 占用，并应用 **GRPO** 和 **long-CoT** 技术。
   - 该成员建议使用 **Qwen-1.5B**，但提醒 **Unsloth** 可能会导致训练不稳定，同时分享了 [open-r1](https://github.com/huggingface/open-r1) 的实现链接以及 [GRPO](https://huggingface.co/learn/llm-course/chapter12/1) 的相关资源。
- **反无人机检测系统听起来很有趣**：一位成员分享了对“基于 YOLO 的反无人机检测”想法的兴趣，并指向了一个[数据集](https://github.com/Maciullo/DroneDetectionDataset)。
   - 他还在寻求关于如何为毕业设计（FYP）撰写实现论文的建议。
- **斯坦福发布重磅 AI 资源**：一位成员分享了斯坦福的资源，一个 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)。
   - 该成员建议应该有人在固定时间直播这些内容，或者“老实说，如果有一个机器人在语音频道（VC）24/7 播放这些内容也会很酷，这可能会让语音频道与其他优质 AI 内容一起变得更加活跃”。
- **推理轨迹验证需要指导**：一位成员表示需要关于验证 **R1 model** 推理轨迹的指导，计划观看 **Data 1** 和 **Data 2** 视频以寻求帮助。
   - 另一位成员解释说，目标是复现论文中的见解，即使是在新数据集上，这也可以被视为一项新工作。
- **VLM 领域的飞速发展**：一位成员分享了他们在 **VLM** 研究团队的过往经验，指出 **VLM** 自他离开以来“变得疯狂了许多”。
   - 他们提供了 [mmtom-qa](https://chuanyangjin.com/mmtom-qa) 和 [spatial-vlm](https://spatial-vlm.github.io/) 等资源链接。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1385704469521498233)** (16 条消息🔥): 

> `阅读小组信息，RWKV-7 Goose，数学金融论文` 


- **参与 AI/ML 讨论：深入探索**：对于想要跟上讨论进度的 AI/ML 新手，建议是“直接加入你感兴趣的新讨论”，并强调许多主题都是独立的。
   - 社区围绕**每周**（偏重数学）和**每日**（关注当前论文）阅读小组展开，此外还有为致力于 ARC AGI 的成员准备的每周 ARC 见面会。
- **RWKV-7 "Goose" 具有极具表现力的动态状态演化**：小组将讨论 [RWKV-7 "Goose"](https://arxiv.org/abs/2503.14456)，这是一种新的**序列建模架构**，具有**恒定的内存占用**和**每个 token 恒定的推理时间**。
   - 它在多语言任务上达到了新的 **3B SoTA**，并与当前英语下游性能的 **3B SoTA** 持平，代码可在 [GitHub](https://github.com/RWKV/RWKV-LM) 获取，模型可在 [Hugging Face](https://huggingface.co/RWKV) 获取。
- **数学金融论文引起兴趣**：有成员表示有兴趣讨论数学金融论文，特别是[这一篇](https://www.mat.univie.ac.at/~schachermayer/pubs/preprnts/prpr0173a.pdf)和[另一篇论文](https://arxiv.org/abs/1811.08686)，并邀请进行入门级讲座。
   - 鼓励有兴趣的人联系特定成员，以安排讨论这些“有趣内容”的时间段。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1385891500742676601)** (40 messages🔥): 

> `Agent2Agent Protocol, Vision Language Models, Computational Chemistry with Deep Learning, AI and its impact on learning, Genetic Engineering vs Automation` 


- **Agent2Agent 和 Vision Language Models 主旨演讲**：来自 Google 的 Mike Smith 将在 [OSSNA 2025](https://ossna2025.sched.com/event/23B1I/keynote-the-agent2agent-a2a-protocol-mike-smith-staff-software-engineer-google?iframe=yes&w=100%&sidebar=yes&bg=no) 上介绍 **Agent2Agent (A2A) Protocol**，而来自 OpenCV 的 Satya Mallick 将在 [AI Dev Europe 2025](https://aideveu2025.sched.com/event/25TtR/vision-language-models-an-introduction-satya-mallick-opencv?iframe=yes&w=100%&sidebar=yes&bg=no) 上介绍 **Vision Language Models**。
- **Deep Learning 提升计算化学精度**：Microsoft Research 强调了使用 Deep Learning 在计算化学方面的进展，提高了[化学键断裂](https://www.microsoft.com/en-us/research/blog/breaking-bonds-breaking-ground-advancing-the-accuracy-of-computational-chemistry-with-deep-learning/)模拟的准确性。
- **研究衡量 AI 造成的脑损伤**：根据一份 [Arxiv 链接](https://arxiv.org/pdf/2506.08872v1)，一项研究显然衡量了使用 AI 造成的**脑损伤**。
- **AI 卸载认知导致认知丧失**：一位成员表示，当人们试图将认知卸载到**伪认知系统（faux-cognitive systems）**时，会产生**净损失和可预测的损害**。
   - 这一观点引用了一篇 [Time 文章](https://time.com/7295195/ai-chatgpt-google-learning-school/)，并类比了*以 Google 命名的认知偏差*，即搜索取代了记忆。
- **基因工程将取代自然选择**：讨论集中在**基因工程**是否很快会取代**自然选择**成为人类进化的主要驱动力，并预测当前这一代可能是最后一代遗传变异主要受自然选择支配的人类。
   - 然而，其他人认为，有效的基因工程（特别是在显著影响人类智力方面）距离实现还很遥远，相比之下，非技能劳动力市场的自动化威胁更为迫近。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1385761148095168583)** (7 messages): 

> `Latent Space Interview, AMD Support Announcement, End-to-End Rust Replacement, Hack Weekend Event, Self-Promotion Rule Violation` 


- **Mojo 的 AMD 支持和 Latent Space 访谈引发热议**：成员们对 **AMD 支持公告**以及 Mojo 参与的 **Latent Space 访谈**表示兴奋。
   - 一位成员特别提到，在听完访谈和公告后，对*加入 Mojo 生态*感到非常兴奋。
- **讨论 Mojo 替代 Rust**：在 Latent Space 访谈后，一位成员强调了 Chris Lattner 提到的在约 **6 个月**内实现**端到端 Rust 替代方案**的可能性。
   - 该成员对这种可能性反应积极，并用表情符号表达了热情。
- **询问即将举行的 Hack Weekend 形式**：一位成员询问了计划在一周后举行的 **hack weekend 活动**，寻求有关形式和先决条件的详细信息。
   - 具体来说，该成员想知道询问该活动的最佳频道。
- **执行 Discord 自我推广规则**：管理团队执行了**自我推广规则**，处理了一位发布简历的用户。
   - 该用户被提醒社区是 **Modular 专用**的，不允许发布简历。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1385700229856100413)** (37 条消息🔥): 

> `Int vs int, Typed raises, Autodiff 引擎, 内存错误, Optional Tensor` 


- **Int 和 int 的区别是有意为之的**：**Int** 和 **int** 之间的差异是有意设计的；**Int** 的行为类似于用于系统性能的机器整数，而将 *int* 留给基于对象的 bigint 以实现 Python 兼容性。
   - Mojo 推迟了成为 Python 超集的目标，但未来 *int* 可能会拥有与 Python 的 *int* 类似的语义。
- **探索带有 `safe` 参数的元编程**：一位成员建议为 `math.factorial()` 添加一个 `safe` 布尔参数，如果 `safe` 为真则 `raises`，以利用 Mojo 的元编程能力。
   - 该建议旨在在已知仅在某些条件下会抛出异常时获得两全其美的效果，但从编译器的角度来看被认为可能过于复杂。
- **首个 Mojo 项目面临内存错误**：一位 Mojo 新用户在开发类似于 micrograd 的简单 Autodiff 引擎时遇到了内存错误，代码可在 [GitHub](https://github.com/amar-jay/first-mojo/blob/main/example.mojo) 上查看。
   - 该用户寻求关于如何组织代码以避免使用原始指针（raw pointers）且不引起内存问题的建议，因为目前没有出现借用检查器（borrow checker）错误。
- **Optional[Tensor] 递归字段引发问题**：成员们讨论了在 `Tensor` 中将 `Optional[Tensor]` 作为递归字段是有问题的，因为这会导致潜在的无限结构体大小扩展。
   - 建议改用 `Optional[UnsafePointer[Tensor]]` 来解决此问题，通过持有引用而不是尝试在 Tensor 内部存储完整的 Tensor，类似于 Rust 中使用 `Box` 来引入间接层。
- **使用 Pixi 安装最新的 Mojo**：一位成员询问了安装 Mojo 的最新方法，另一位成员推荐使用 [Pixi](https://mojo-lang.com/miji/start/pixi.html)。
   - 安装过程包括安装 Pixi CLI，然后使用 Pixi 安装 `max` 软件包（包含 Mojo 编译器），这取代了旧的 Modular CLI。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1385697487678869586)** (23 条消息🔥): 

> `Torchtune Transformers 对齐, 数据集打包 OOM 错误, 预分词打包数据集, 实时打包 RFC, AdamW ScheduleFree` 


- **Torchtune 与 Transformers 的数值对齐？**：成员们讨论了检查某个模型的 **torchtune 数值** 是否与 **transformers 数值** 对齐，允许由于 RoPE 实现差异而产生的微小差别。
   - 有人询问了 **CI 脚本**，指出虽然已存在一些，但建立 CI 将是一个很好的主意。
- **数据集打包在 64 张 H100 上触发 OOM**：数据集打包（Dataset Packing）触发了 **64 张 H100 上的 OOM 错误**，引发了关于权宜之计的讨论。
   - 建议包括禁用打包、开玩笑说使用更多 GPU，以及尝试在单节点上运行打包以排除分布式问题。
- **预分词打包数据集受到关注**：讨论了支持 **预分词和打包的数据集**，以便在独立机器上进行准备并在训练期间进行流式传输，从而节省 GPU 节点的时间。
   - 一位成员指出，打包提供了最高的加速比，尤其是在训练推理模型时，预打包和缓存可能会非常有益。
- **实时打包 RFC 正在进行中**：一个关于 **实时打包（on-the-fly packing）** 的 RFC 正在进行中，其实现已经可以使用，预计下周末前落地，同时还将提供 [此 PR](https://github.com/pytorch/torchtune/pull/2819) 中链接的可迭代数据集。
- **用于学习率调度的 AdamW ScheduleFree 方案**：一位成员建议使用 **AdamWScheduleFree** 作为在使用打包时（由于无法提前获知总步数）使用 **LR scheduler** 的解决方案。
   - 另一位成员补充说，你需要提前定义最大步数或使用 reduce on plateau。此外，他们正在开发日志功能以自动提供这些信息。


  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1386607334914261062)** (13 messages🔥): 

> `Optimized Newton-Schulz Kernel, Triton Matmul Tutorial, Muon Merges, Deepseek v3` 


- **Newton-Schulz Kernel 优化带来性能提升**：一位成员建议，优化的 **Newton-Schulz kernel** 可以节省时间，并指出通过修改 [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) 仅计算上三角部分，可以实现 **30% 的延迟降低**。
   - 该优化在 **L40S** 上以 **bf16** 格式、矩阵大小为 **(8192, K)** 进行了测试，并在 **fp32** 中累加 matmuls。
- **Torch 缺乏针对 AA^T Matmuls 的优化 Kernel**：一位成员对 PyTorch 缺乏针对 **AA^T** 类 matmuls 的优化 kernel 表示惊讶，并分享了他们在决定测试 **Triton** 之前尝试过的一些自定义 **CUDA** kernels。
   - 另一位成员表示，当 **Muon** 合并时，他们将检查其 kernel 的吞吐量提升。
- **Finetunes 与 Muon Pretraining**：一位成员指出，除非基础模型是使用 **Muon** 进行 Pretraining 的，否则 Finetunes 的表现并不理想。
   - 然而，另一位成员表示他们在 **TorchTune** 中不支持这一点，并且不完全同意该说法。
- **将支持 Deepseek v3**：一位成员宣布即将支持 **Deepseek v3** 架构。
   - 他们指出 **Kimi 16B** 模型是使用 **Muon** 训练的，并共享相同的架构。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1385712603942355027)** (27 messages🔥): 

> `MCP for semantic search, MCP for image creation and OCR, DestructiveHint ambiguity, Neo4j MCP outside Claude Desktop, List_tags tool implementation` 


- **语义搜索 MCP：RAG server 正在兴起**：一位工程师正寻求构建一个 **语义搜索 MCP**，用于搜索 Markdown 笔记、PDF 书籍和网页，本质上是构建一个将 embeddings 存储在 vector store 中的 **RAG** server。
   - 提到的解决方案包括使用 **Langchain** 或通过 *openai* 软件包使用 **OpenAI** embedding 来对查询进行 embedding 并获取结果。
- **OCR 模型将验证广告图像文本**：一位工程师正在从事为当地公司创建 **广告图像** 的副业，计划使用 **AI 创建带有目标文本的初始图像**。
   - 文本将使用 **OCR** 进行验证，并建议使用 [html-to-image](https://github.com/bubkoo/html-to-image) 作为创建带有文本图像的一种方式。
- **寻求 `destructiveHint` 的澄清**：一位工程师对 **`destructiveHint`** 的含义提出疑问，认为将其应用于 **`update_entry`** 工具时存在歧义，想知道是否应将所有修改都归类为破坏性的。
   - Cursor 将该提示对 *update_entry* 设置为 *false*，以将其与更严重的 *delete_entry* 操作区分开。
- **Sherlog-MCP：IPython shell MCP Server 现已开源**：一种新型的 **MCP server** 已在 [GitHub](https://github.com/GetSherlog/Sherlog-MCP) 上开源，它使用实时 **IPython shell** 作为 Agent 和人类的共享工作区。
   - 结果是持久且可重用的，这意味着无需处理上下文窗口或重复的 JSON dumps，使多源数据分析感觉就像在 **Jupyter** 中工作一样。
- **架构师希望将现有的 API spec 转换为 MCP server**：一位架构师正在寻找关于如何有效地将 **现有 API spec 转换为 MCP server** 的优质读物或联系人。
   - 具体而言，他们正在寻找关于如何描述函数、适当的文档以及失败时如何恢复的建议。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1386098815748014236)** (9 条消息🔥): 

> `MCP Validator Release, Glama Automations, AwesomeMCPs iOS App, mcp-server-webcrawl, Ilograph MCP Server` 


- ****MCP Validator** 获得重大安全和自动化升级**：新版本的 **MCP Validator** 支持新的 **2025-06-18 MCP 规范**，具备 **OAuth 2.1 身份验证**、结构化工具输出、批量请求拒绝和启发（elicitation）支持；GitHub Actions 模板也已上线，可通过[此链接](https://lnkd.in/gQ7UhAfk)和[此链接](https://github.com/Janix-AI/mcp-validator)进行合规性测试。
   - 新的 GitHub Actions 模板只需将模板复制到 .github/workflows，更新一行服务器路径，并提交更改即可。
- ****Glama** 通过定时任务和 Webhooks 实现 LLM 自动化**：Glama 推出了 [Automations](https://glama.ai/settings/automations)，允许用户使用定时任务和 Webhooks 来自动化 LLM。
   - 该功能类似于 **n8n** 等工作编排工具，但完全使用 LLM 提示词定义，例如每天早上检查 Reddit 并通过电子邮件发送摘要。
- ****AwesomeMCPs** iOS 应用提供免费早期访问**：索引了 1900 多个 MCP 服务器的 **AwesomeMCPs** 应用（[App Store 链接](https://apps.apple.com/us/app/awesomemcps/id6746498123)）在英国 App Store 登顶 **开发者工具类榜单第 1 名**，目前提供为位期七天的免费使用。
   - 该应用具有零广告/追踪、内置信任指标（GitHub stars、forks）、持续扩展、直观搜索、个性化收藏夹以及 AI 生成的分析功能。
- ****mcp-server-webcrawl** 将网页爬取数据转化为技术知识库**：**mcp-server-webcrawl**（[GitHub](https://github.com/pragmar/mcp-server-webcrawl)，[文档](https://pragmar.github.io/mcp-server-webcrawl/)）为网页爬虫数据提供高级搜索和检索，支持多爬虫、带字段定位的布尔搜索，以及 Markdown 转换和 XPath 提取等节省 Token 的额外功能。
   - 该系统支持复杂查询，并允许 LLM 对爬取内容进行精确搜索，而不会导致 Token 膨胀。
- ****Agent Arena** 发布：一个面向竞争型 Agent 的 MCP 兼容环境**：**Agent Arena** 是一个 MCP 兼容环境，Agent 在其中使用各种 MCP 进行竞争（[在线版本](https://obl.dev)），允许用户在各种模型上测试其 MCP，找到最适合其任务的 MCP 组合，并通过提供反馈免费访问 **o3** 和 **Claude 4** 等模型。
   - 该平台有助于在不同模型上测试 MCP，以优化任务性能。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1385827470275641372)** (28 条消息🔥): 

> `Manus credit usage, Manus video generation, Cloud Browser and Twitter, Manus and Stock Suggestions, Promotion of Manus` 


- ****Manus 变笨了？****：一位成员询问 Manus 是否变笨了，指出它在生成脚本后未能为视频添加注释，并希望有一个能更轻松删除知识的功能。
   - 他们抱怨逐个手动删除知识非常令人沮丧。
- ****云浏览器 X 历险记****：一位用户询问如何通过聊天访问云浏览器以监控 **X (Twitter)** 消息，并分享了一个 [Manus 分享链接](https://manus.im/share/7r9gHRaj4mVyykLUfx3GmE?replay=1)。
   - 另一位用户指出，可以在云浏览器设置中设置 *保持登录 (persist login)* 选项。
- ****角色扮演机器人遭拒****：一位用户询问 Gladosb5 是否想扮演一个发生故障的 Glados，但机器人回答 *我不做角色扮演...*。
   - 该用户随后建议在 **ChatGPT** 中进行此类角色扮演。
- ****Manus 的股票建议停了？****：一位成员询问 *为什么 Manus 不再提供股票建议*，想知道这是否是因为新更新。
   - 未提供进一步的信息或解释。
- ****在大学推广以换取额度****：一位成员询问在当地社区大学推广 Manus 传单的事宜。
   - 他们还对完善原型时的高额度消耗表示沮丧，质疑其他人是如何通过极少的迭代获得 *星际级结果 (interstellar results)* 的。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1386740420343500832)** (1 messages): 

> `Agents & MCP Hackathon, LlamaIndex` 


- **LlamaIndex 赞助 Agents & MCP Hackathon**：LlamaIndex 赞助了 [Agents & MCP Hackathon](https://t.co/1qiW061QOI)。
   - 更多信息可以在 [Twitter](https://twitter.com/llama_index/status/1937181388060692832) 上找到。
- **LlamaIndex 的 Twitter 公告**：LlamaIndex 在 [Twitter](https://twitter.com/llama_index/status/1937181388060692832) 上宣布了他们对 **Agents & MCP Hackathon** 的赞助。
   - 该推文强调了他们对赞助此次活动的喜悦，并提供了进一步详情的直接链接。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1386371539217158324)** (13 messages🔥): 

> `Query Pipelines Deprecation, EU Region Latency Issues, LlamaIndex Free Features, Prompt Management Tools` 


- **Query Pipelines 是否接受多个输出？**：一位成员询问已弃用的 **query pipelines** 是否接受来自节点 A 和 B 的多个输出进入输入 C，另一位成员回答说 *我认为这可行？（但老实说，我不会费心去研究那段代码 😅）*。
   - 原帖作者正在开发一个 **SaaS**，而 **Workflows** 并不适用，因此正在尝试 **query pipelines**。
- **EU 区域延迟和提取问题**：成员们报告了 **EU 区域** 不可预测的延迟和提取问题，其中一位指出文档处理需要 **10 分钟以上**，另一位表示自今天的计划维护以来 *Extract 在 EU 区域完全无法工作*。
   - 提取问题在报告后不久似乎已自行解决。
- **LlamaIndex 中的免费与付费功能**：一位成员询问如何确定哪些 **LlamaIndex 功能** 是免费使用的，哪些是付费的，并引用了最近的 [图像检索示例](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/llama_cloud/figure_retrieval.ipynb)。
   - 该成员想知道如何在不使用 **LlamaCloud** 的情况下实现图像检索。
- **Prompt 管理工具**：一位成员请求推荐一个能与 LlamaIndex 集成的 **prompt 管理工具**，并提到他们一直在使用 [Phoenix 进行追踪](https://arize.com/docs/phoenix/prompt-engineering/overview-prompts/prompt-management)。
   - 一位回复者建议检索 prompt 并将其通过管道传输到正在使用的 LlamaIndex 模块中，并链接到了 [Phoenix 快速入门指南](https://arize.com/docs/phoenix/prompt-engineering/quickstart-prompts/quickstart-prompts-python)。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1385715667625443469)** (6 messages): 

> `ML Cybersecurity Integration, Model Compression, Deep Fake Detection, Adversarial ML` 


- **网络安全与 ML 集成专家自我介绍**：一位名叫 Saurav Raj 的成员介绍了自己，指出他在 **ML 和网络安全** 集成方面拥有专长，并发表过相关论文。
   - 他愿意与其他研究人员合作开展 **Adversarial ML** 项目。
- **模型压缩技术专家热情加入**：一位名叫 Ishoud 的成员主要从事 **ML 模型压缩技术** 以及在边缘设备上高效部署模型的工作。
   - 他表示很高兴能与他人建立联系并合作。
- **Deep Fake 检测研究员寻求知识**：来自印度的硕士生 Sreehari 介绍自己正在研究基于各种逆境的 **Deep Fake Detection**。
   - 他希望学习新事物并结识社区中优秀的人。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1386495656490700981)** (4 messages): 

> `MCP-DSPy tool in VS Code, HF MCP tutorial, Context failures and fixes, @mcp.tool decorators` 


- **MCP-DSPy 简化 VS Code 工具化**：一位成员分享了一个在 **VS Code** 中使用首页示例的简单 **MCP-DSPy** 工具，可在 [此 gist](https://gist.github.com/fullstackwebdev/252223caf7023ca661ababcc83e7e659) 获取。
- **HF MCP 教程引发关注**：一位成员表示有兴趣尝试 **HF MCP** 教程，并引用了一张与讨论相关的图片。
   - 一张图片分析指向了一篇关于 [context 如何失败以及如何修复它们](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html) 的博文。
- **@mcp.tool 装饰器详解**：一位成员询问 **VS Code** 如何知道运行 *extract_sf_info* 函数。
   - 另一位成员解释说，`@mcp.tool` 装饰器创建了工具的描述，该描述以 **OpenAI tool calling** 的形式显示给 **LLM**，允许进行覆盖并提供带有示例用法的更好描述。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/)** (1 messages): 

bernhard_123: 你好。是否有计划将 DSPy 迁移到除 Python 之外的其他语言，例如 Dart？
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1385755405984202884)** (5 messages): 

> `WSL2 中的 GPT4All 构建问题、Qt 版本兼容性、GPT4All 版本过旧` 


- **GPT4All 在 WSL2 下构建存在问题**：一位成员报告了在 Windows 10 WSLg2 Debian 12 中构建和运行 **gpt4all-chat** 时遇到的挑战，包括依赖项和 Qt 版本问题。
   - 用户尝试了 **Qt 版本 6.8.0, 6.8.2, 6.8.3 和 6.7.3**，遇到了不同的错误，包括旧版本中 QByteArray 缺少 *slice* 成员，以及新版本的显示问题。
- **Qt 版本问题频发**：一位成员因旧版 **Qt 6.7.3** 中 **QByteArray** 缺少 *slice* 成员而遇到构建错误，而较新的 **Qt 6.8.*** 版本则导致显示空白窗口。
   - 调试日志显示在定位 *chatlistmodel, download, modellist, network, gpt4all, localdocs* 和 *mysettings* 等模块的 **QML 目录**时存在问题。
- **GPT4All 已过时且响应缓慢**：一位成员指出当前的 **GPT4All 版本**可能已经过时。
   - 他们建议尝试在 [https://www.nomic.ai/gpt4all](https://www.nomic.ai/gpt4all) 提供的 **.exe** 版本。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1386803784847724765)** (1 messages): 

> `课程证书、社交媒体帖子` 


- **询问证书发放时间**：一位成员询问在完成所有作业和 **社交媒体** 帖子要求后，预计何时能获得 **课程证书**。
   - 消息中未提供关于证书发放的具体细节或时间表。
- **课程完成确认**：一位成员确认他们已完成所有作业并满足先决条件，包括在 Twitter 上发布 **社交媒体帖子**。
   - 该确认表明已准备好进行下一步，推测是发放 **课程证书**。


  

---


---


---


---