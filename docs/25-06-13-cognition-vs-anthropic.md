---
companies:
- cognition
- anthropic
- langchain
- huggingface
- microsoft
- llamaindex
- linkedin
- blackrock
date: '2025-06-13T05:44:39.731046Z'
description: 在过去的 24 小时内，**Cognition** 的 Walden Yan 建议“不要构建多智能体（Multi-Agents）”，而 **Anthropic**
  则分享了他们利用 **Claude** 的多智能体研究架构构建多智能体系统的方法。**LangChain** 强调了 **LinkedIn** 和 **BlackRock（贝莱德）**
  在上下文工程和生产级 AI 智能体应用方面的进展。社区正围绕多智能体 AI 开发展开辩论。此外，**Hugging Face** 宣布将弃用对 **TensorFlow**
  和 **Flax** 的支持，转而全面支持 **PyTorch**。来自 **LlamaIndex** 和 **Anthropic** 关于智能体记忆和模型诱导（elicitation）技术的研究也受到了讨论。
id: MjAyNS0w
models:
- claude
people:
- walden_yan
- hwchase17
- assaf_elovic
- sh_reya
- hamelhusain
- omarsar0
- clefourrier
- jerryjliu0
- akbirkhan
title: '**Cognition 对阵 Anthropic：不要构建多智能体 / 如何构建多智能体**'
topics:
- multi-agent-systems
- context-engineering
- agent-memory
- model-elicitation
- ai-evaluation
- deep-research-workflows
- framework-migration
- pydantic-schema
---

**良好的技术辩论正是我们所需要的。**

> 2025年6月12日至6月13日的 AI 新闻。我们为您检查了 9 个 Subreddit、449 个 Twitter 账号和 29 个 Discord 社区（218 个频道，6215 条消息）。预计节省阅读时间（以 200wpm 计算）：504 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

在过去的 24 小时内，**Cognition** 的 [Walden Yan](https://x.com/walden_yan/status/1933264183837282558) 表示[不要构建多智能体 (Multi-Agents)](https://cognition.ai/blog/dont-build-multi-agents)，而 **Anthropic** 则选择在今天讨论他们[如何看待构建多智能体](https://www.anthropic.com/engineering/built-multi-agent-research-system)。

AI 工程师，你选哪条路？

**读者挑战**：如果你想对这两种方法进行对比分析，请发布并推特给 [@smol_ai](https://x.com/smol_ai)，我们会在空闲时间展示你的作品。如果想获得额外加分，可以对比 [Building Proactive Agents](https://bryanhoulton1.substack.com/p/building-proactive-ai-agents) 和 [Ambient Agents](https://blog.langchain.dev/introducing-ambient-agents/)。

[](https://resend-attachments.s3.amazonaws.com/V7GW5OJr2RBcn57)

---

# AI Twitter 摘要

**AI Agent 开发与工具链**

- **Claude 的多智能体研究架构**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1933630785879507286) 发布了一篇博客文章，详细介绍了他们如何利用并行工作的**多智能体 (multiple agents)** 构建 **Claude 的研究能力**，分享了成功的策略和工程挑战。
- **上下文工程与产品 UX**：**LangChain** 的 [@hwchase17](https://twitter.com/hwchase17/status/1933528162799136988) 强调了与 [@assaf_elovic](https://twitter.com/hwchase17/status/1933542550222352586) 在 **"CAIR" (Confidence in AI Results)** 框架上的合作，该框架分解了除原始模型能力外影响产品采用的组件。他还强调了 **"Context Engineering" (上下文工程)** 的重要性，他将其描述为[构建 AI Agent 的工程师的首要工作](https://twitter.com/hwchase17/status/1933278290992845201)，也是 Prompt Engineering 的一种更动态的演进。
- **用于生产级 Agent 的 LangChain**：[@LangChainAI](https://twitter.com/LangChainAI/status/1933576634843738434) 展示了 **LinkedIn** 如何使用 **LangChain** 和 **LangGraph** 构建其用于招聘的生产级 AI Agent，提供了一个扩展至 **20 多个团队** 的技术架构。另一个例子展示了 **BlackRock** 如何[构建生产就绪的 AI Agent](https://twitter.com/hwchase17/status/1933275125077733759) 来驱动其 **Aladdin** 平台。
- **面向工程师的 AI Evals**：由 [@sh_reya](https://twitter.com/HamelHusain/status/1933575166917030184) 和 [@HamelHusain](https://twitter.com/HamelHusain/status/1933619275325190194) 提供的 "AI Evals for Engineers and Technical PMs" 课程因其实战见解而获得好评，参与者指出他们[已经将课程内容转化为自定义工具](https://twitter.com/HamelHusain/status/1933397529754022238)，并发现它[远超其他资源](https://twitter.com/HamelHusain/status/1933508512879161476)。[@HamelHusain](https://twitter.com/HamelHusain/status/1933615429253280268) 还分享了 Eval 工具链中的常见差距，以及[针对多样化用户查询进行错误分析](https://twitter.com/HamelHusain/status/1933612965993066842)的重要性。
- **深度研究 Agent 工作流**：[@omarsar0](https://twitter.com/omarsar0/status/1933511531590824443) 分享了一个使用 **n8n** 构建的个性化深度研究 Agent 工作流。此外，**Microsoft** 的一篇论文也被重点提及，介绍了一个[针对大型系统代码库的深度研究 Agent](https://twitter.com/omarsar0/status/1933673330545987773)。
- **Hugging Face 弃用 TensorFlow/Flax 转向 PyTorch**：[@clefourrier](https://twitter.com/clefourrier/status/1933271084650189263) 分享了“苦乐参半的消息”：`transformers` 库正在弃用对 **TensorFlow** 和 **Flax** 的支持。**PyTorch** 确认 **Hugging Face** 将全力投入其框架，并指出[用户群已向其整合](https://twitter.com/stanfordnlp/status/1933528689662480781)。
- **用于结构化数据的 Agent 记忆**：来自 **LlamaIndex** 的 [@jerryjliu0](https://twitter.com/jerryjliu0/status/1933672040936190243) 描述了一个用于 Agent 的结构化 Artifact 内存块，它跟踪一个随时间更新的 **Pydantic schema**，这对于表单填写等任务至关重要。

**模型研究、技术与性能**

- **Anthropic 的 Model Elicitation 与 Diffing**：[@akbirkhan](https://twitter.com/akbirkhan/status/1933323897526759553) 分享了 **Anthropic** 的最新研究，关于如何在没有外部监督的情况下从预训练模型中诱导（eliciting）能力。[@jiaxinwen22](https://twitter.com/jeremyphoward/status/1933364618371739948) 澄清这关乎 **elicitation**，而非自我改进（self-improvement）。另外，[@jxmnop](https://twitter.com/jxmnop/status/1933571979975487996) 强调了来自早期 **Anthropic** 博客的 **"model diffing"** 技术，该技术使用 crosscoder 在模型之间生成可解释的差异（diffs），展示了 post-training 如何增加特定能力。
- **强化学习（RL）的力量**：[@jxmnop](https://twitter.com/jxmnop/status/1933359925415325980) 评论了随着 **LLM 上的 RL** 不断进步而出现的惊人可能性，并表示“我们才刚刚开始”。这一观点在关于 **ReMA (Reinforced Meta-thinking Agents)** 的讨论中得到了回应，这是一种结合了 meta-learning 和 RL 的新方法，[提升了在数学和 LLM-as-a-Judge 基准测试中的表现](https://twitter.com/TheTuringPost/status/1933478813062869156)。
- **微调即持续预训练**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1933595426873356401) 分享了来自 [@antoine_chaffin](https://twitter.com/ClementDelangue/status/1933598791128506477) 的结果，作为 **微调只是持续预训练** 这一原则的实际案例。该工作发布了 **BioClinical ModernBERT**，这是一个在生物医学文献上进行预训练并在临床笔记上进行微调的模型，取得了 SOTA 结果。
- **Text-to-LoRA 超网络**：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1933302559957275073) 推出了 **Text-to-LoRA (T2L)**，这是一种将多个 LoRA 压缩到自身中的超网络（hypernetwork），并能根据文本描述生成新的 LoRA，从而实现 [即时 LLM 适配（on-the-fly LLM adaptation）](https://twitter.com/TheTuringPost/status/1933608004710248627)。
- **字节跳动（ByteDance）用于视频生成的 APT2**：**字节跳动** 展示了 **APT2**，一种用于实时交互式视频生成的 [自回归对抗性后训练方法（Autoregressive Adversarial Post-Training method）](https://twitter.com/NandoDF/status/1933266267663634465)。
- **新模型与基准测试**：**Glass Health** 宣布其 **Glass with Deep Reasoning** 模型在临床基准测试中取得了新的 SOTA，包括 **USMLE Steps 1–3 达到 97%** 以及 [JAMA Clinical Challenge 案例达到 98%](https://twitter.com/GlassHealthHQ/status/1933291603906736328)。**Cartesia AI** 的 **Sonic-2** 模型 [在 Labelbox 语音生成排行榜上名列前茅](https://twitter.com/krandiash/status/1933306410517090684)。
- **通过应用可解释性为 LLM 去偏**：[@NeelNanda5](https://twitter.com/NeelNanda5/status/1933645976889422110) 赞扬了一篇论文，该论文表明虽然之前的去偏（debiasing）技术在现实的简历筛选场景中失效，但简单地发现并移除模型中与性别或种族相关的方向，仍然是一种有效的去偏策略。

**基础设施、硬件与数据**

- **重大互联网中断**：来自 **Replit** 的 [@pirroh](https://twitter.com/pirroh/status/1933269623979585695) 等人报告了一次大规模互联网中断，[@itsclivetime](https://twitter.com/itsclivetime/status/1933426721723986179) 指出这并非 DNS 或 BGP 问题。此次中断被归因于 [**Google Cloud (GCP)** 的问题](https://twitter.com/jeremyphoward/status/1933357293699281021)，尽管 Google 自身的产品基本未受影响，因为它们并不运行在面向公众的 GCP 基础设施上。
- **GPU 之战：AMD vs. NVIDIA**：[@dylan522p](https://twitter.com/dylan522p/status/1933628242432262304) 分析了 **AMD** 如何凭借其在性能/TCO 表现出色的 **MI355** 采取行动，而 **NVIDIA** 的 DGX 策略则疏远了一些客户。然而，他指出 AMD 的机架级解决方案就像是“来自 temu dot com 的 GB200 nvl72”。[@scaling01](https://twitter.com/scaling01/status/1933569373031018932) 也表达了类似的观点，认为 **AMD** 需要具备与 **NVIDIA** 等效的软件栈和支持。
- **LlamaParse 文档解析预设**：**LlamaIndex** 宣布了 [**LlamaParse** 的新用例预设](https://twitter.com/jerryjliu0/status/1933627680265810205)，这些预设作为专门的解析 Agent，可将文档渲染为结构化格式，例如表单的表格或技术图纸的 XML。
- **合成数据与人机回环 (HITL)**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1933297536300929411) 讨论了合成数据在填补数据空白和减少偏见方面的潜力，但也警告了模型崩溃 (model collapse) 的风险。他们强调需要 **Human-in-the-loop (HITL)** 工作流来保持合成数据的可靠性和安全性。
- **本地设备端模型**：关于本地模型的讨论凸显了其日益增长的重要性，[@awnihannun](https://twitter.com/awnihannun/status/1933266802450313715) 简单地表示 `pip install mlx-lm`。[@reach_vb](https://twitter.com/reach_vb/status/1933503436630130836) 推荐将 **smollm 2 配合 llama.cpp 或 MLX** 使用，作为处理日常任务的小型“通用基础智能”，而 [@mollycantillon](https://twitter.com/awnihannun/status/1933273566763786699) 则发表了关于 **MLX** 实际应用以及构建快速设备端语义搜索的演讲。

**行业评论与地缘政治**

- **Perplexity 的野心与产品策略**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1933508237934145989) 详细介绍了用户对 **Perplexity Finance** 的浓厚兴趣和增长，重申了他通过提供更好的 UX 和准确性来挑战 **Bloomberg Terminal** 等老牌企业的野心。他还宣布 [新产品 **Comet** 的邀请函正在发放](https://twitter.com/AravSrinivas/status/1933289407705960697)，并强调了核心原则 [**"Context is all you need"**](https://twitter.com/AravSrinivas/status/1933503918996402366)。他通过 [指出 Google 庞大且集成的生态系统](https://twitter.com/AravSrinivas/status/1933283015586951623) 来激励他人保持雄心。
- **以色列-伊朗冲突与地缘政治分析**：大量推文关注以色列和伊朗之间升级的冲突。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1933560574857916727) 认为物质层面的考虑是幼稚的，像 **Israel** 这样的国家可以通过针对关键人物和基础设施，在常规规则之外运作。这与 [北韩的核成功偏向了美国对核扩散的假设](https://twitter.com/teortaxesTex/status/1933543528455442484) 这一观点形成对比。[@francoisfleuret](https://twitter.com/francoisfleuret/status/1933577120640282800) 也对伊朗防空系统的明显缺失提出了质疑。
- **编程中 Human-in-the-Loop 的终结**：[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1933522079145734495) 预测 AI 辅助编程的“半人马 (centaur)”时代将只是 **“昙花一现”**，[@vipulved](https://twitter.com/vipulved/status/1933647581370069401) 也表达了类似观点，认为我们将在未来 12 个月内见证 **“手写代码的终结”**。
- **NVIDIA CEO Jensen Huang 对 Anthropic 的评论**：[@Teknium1](https://twitter.com/Teknium1/status/1933620345749319710) 和 [@jeremyphoward](https://twitter.com/jeremyphoward/status/1933597258047762657) 分享了一篇文章，其中 **NVIDIA CEO Jensen Huang** 对 **Anthropic** 辞色严厉，批评其专注于安全的立场，并暗示不应只信任他们一家来处理 AI。
- **Meta 的 AI 人才与策略**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1933329447853437251) 评论说，如果 **Zuckerberg** 几年前没有裁掉一支卓越的 AI 人才团队，**Meta** 今天的人才问题就不会那么严重。[@dylan522p](https://twitter.com/dylan522p/status/1933660636732350686) 分析了最近对 **Alex Wang** 的聘用，指出关键的衡量标准将是他如何入职并重新组织现有人才以构建超级智能 (superintelligence)。
- **ChatGPT 与医疗诊断**：一个关于 **ChatGPT** [通过纠正误诊挽救生命](https://twitter.com/npew/status/1933514178314318061) 的故事被广泛分享，许多评论者也分享了类似的经历。[@shuchaobi](https://twitter.com/shuchaobi/status/1933511659232112751) 指出，这正是激励 **OpenAI** 团队的动力。

**幽默/迷因**

- **五角大楼披萨报告**：一张带有关于伊朗头条新闻的“五角大楼披萨报告”截图被 [@jeremyphoward 分享](https://twitter.com/jeremyphoward/status/1933457163936280647)，配文是“五角大楼披萨报告预言了这一切”。
- **无线电波的发现**：一个配文为“发现无线电波的那个人”、展示一个人戴着超大耳机的迷因 [被广泛分享](https://twitter.com/jeremyphoward/status/1933357378210312337)。
- **对 AI 编程工具的挫败感**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1933563892271362337) 发布了一个开发者使用 **Cursor** 体验的恶搞：在生成的代码失败后，开发者反复输入了 15 次 **"pls fix"**，最后因沮丧而放弃。
- **地缘政治不安**：在一条广为流传的推文中，[@zacharynado](https://twitter.com/zacharynado/status/1933579419810996407) 建议，如果你“今晚对全球地缘政治局势感到有些不安，请记住尽可能多花时间在你的爱好上”。
- **"pls fix"**：[@RhysSullivan](https://twitter.com/imjaredz/status/1933591454267433398) 描述了看着 **Claude Opus 消耗了价值 70 美元的 Token** 来重新生成 shadcn 组件，而不是运行一个简单的命令。
- **价值 1 亿美元的 Prompt**：[@skirano](https://twitter.com/skirano/status/1933564941832728751) 调侃道：“那种当你意识到自己写了一个价值 1 亿美元的 Prompt 时的感觉。”

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 1. EuroLLM 和 EuroMoE 模型发布公告

- [**EuroLLM 团队发布了多个新模型的预览版**](https://www.reddit.com/r/LocalLLaMA/comments/1laazto/the_eurollm_team_released_preview_versions_of/) ([Score: 109, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1laazto/the_eurollm_team_released_preview_versions_of/)): **EuroLLM 团队发布了多个新模型的预览版，包括一个 22B 参数的 LLM ([base](https://huggingface.co/utter-project/EuroLLM-22B-Preview), [instruct](https://huggingface.co/utter-project/EuroLLM-22B-Instruct-Preview))，两个视觉语言模型 ([1.7B](https://huggingface.co/utter-project/EuroVLM-1.7B-Preview), [9B](https://huggingface.co/utter-project/EuroVLM-9B-Preview))，以及一个小型 Mixture-of-Experts 模型 ([总参数 2.6B，激活参数 0.6B](https://huggingface.co/utter-project/EuroMoE-2.6B-A0.6B-Preview))，全部采用 Apache-2.0 许可证。值得注意的是，该 MoE 模型相对于其参数量表现出了强劲的性能。所有模型均提供最高 4K 的上下文窗口。** 评论者指出 22B 模型的上下文窗口限制 (4K) 是一个重大缺陷，但认为这些发布是欧盟本土开源模型的重要进展。针对俄语的非正式评估表明，9B VLM 的性能达到或超过了 Mistral 和 Gemma 2 (9B) 等同类开源模型。
    - 一位用户分享了使用俄语测试 EuroLLM 9B 模型的结果，称其“表现不错但并不完美”——可能略好于 Mistral 的较小模型，且在该语言的表现上与 Gemma 2 9B 持平，这表明该参数范围内的多语言能力有所增强。
    - 讨论重点关注了 22B 模型的 `4k context` 窗口，一位评论者暗示这对于某些用例可能不足，反映了目前对大模型上下文长度的持续关注。
    - 对于 EuroMoE-2.6B-A0.6B 宣称的参数量（22B 参数）与其 `5 GB` 模型大小之间的关系存在质疑，暗示了对压缩、架构（如 Mixture-of-Experts）或实际大小与参数对应关系的疑问。

### 2. OpenAI 开放权重模型测试者见解

- [**拿到了 OpenAI 开放权重模型的测试版本。推理引擎非常精简！**](https://v.redd.it/3r075o87qo6f1) ([Score: 974, Comments: 74](https://www.reddit.com/r/LocalLLaMA/comments/1laee7q/got_a_tester_version_of_the_openweight_openai/)): **一名用户声称收到了 OpenAI “开放权重”模型的“测试版本”，并指出推理引擎“非常精简”。未提供基准测试、实现细节或架构规范。指向进一步技术数据的链接返回 403 Forbidden 错误，因此无法进行外部验证或获取详细信息。** 热门评论集中在明显的运行速度（“首个 token 生成时间非常棒”）和用户对 Alignment 的舒适度上，但评论区没有深入的技术讨论或基准测试。
    - ExplorerWhole5697 对“首个 token 生成时间” (Time to First Token) 非常快进行了技术观察，表明该 OpenAI 推理引擎具有低延迟和高效的推理性能。这表明定制的精简推理引擎表现出强大的响应能力，这在具有苛刻实时约束的生产环境中非常有价值。

### 3. AI 个性偏好与用户参与度讨论

- [**我们不需要只会唯唯诺诺的 AI。我们需要有观点的 AI**](https://www.reddit.com/r/LocalLLaMA/comments/1lanhbd/we_dont_want_ai_yesmen_we_want_ai_with_opinions/) ([Score: 178, Comments: 41](https://www.reddit.com/r/LocalLLaMA/comments/1lanhbd/we_dont_want_ai_yesmen_we_want_ai_with_opinions/)): **OP 总结了一个 AI 播客平台的 A/B 测试和用户参与度数据，显示具有一致且有主见（但非冒犯性）个性的 AI 主持人能显著提高用户满意度（在“毒舌”模式下提升了 `40%`）并大幅延长会话时间（增加了 `2.5x`）。实现方式包括显式地为 AI Agent 编写古怪或逆向的观点（例如“麦片是汤”），导致用户为了持续辩论而回归——这表明在基于 LLM 的朋友/角色应用中，真实感的摩擦力能驱动对话深度和留存率。链接：https://www.reddit.com/r/LocalLLaMA/comments/1dgwk71/we_dont_want_ai_yesmen_we_want_ai_with_opinions/** 热门评论讨论了“唯唯诺诺”的行为是否只是默认设置而非 LLM 的固有属性，并指出用户 Prompt 或 System Instructions 可以完全控制 AI 的个性。其他人则指出这具有领域特定性：逆向思维的 AI 在对话 Agent 中很有价值，但在计算器或自动驾驶汽车等实用型应用中则不受欢迎。
    - 资深用户指出，LLM 讨好型的“助手”人格源于默认的 Prompting，可以通过修改 System Prompt 进行自定义，从而根据用户需求允许更多批判性或有主见的 AI 行为。
    - 一位评论者强调，某些模型（尤其是开箱即用的 Grok）相比 ChatGPT 等其他模型更愿意“回怼”，并提到早期的 Google 模型往往过于受限，有时会因为安全或合规措施拒绝简单的编程任务。
    - 技术批评者将公测模型倾向于无害和顺从（而非默认提供强有力的批判性反馈）的主要原因归结为平台限制，例如 API 设计或 LLM Arena 等评估程序。

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. LLM 自我提升与自动化微调进展

- [**SEAL：能够编写自身更新的 LLM 解决了 72.5% 的 ARC-AGI 任务——从 0% 大幅提升**](https://arxiv.org/pdf/2506.10943) ([Score: 905, Comments: 180](https://www.reddit.com/r/singularity/comments/1la8myf/seal_llm_that_writes_its_own_updates_solves_725/))：**该帖子详细介绍了 SEAL (Self-Adapting Language Models)，这是一个 LLM 通过自主生成自身的微调数据和“自我编辑”指令来实现持久的权重级更新的框架，通过强化元学习过程闭合了学习环路。与之前的递归/自我提升框架不同，SEAL 通过实际的参数更新实现了递归自我提升，在 ARC-AGI 中获得了 72.5% 的高分（而相同模型在没有 SEAL 的情况下为 0%），并通过直接优化有用的自我训练数据，超越了合成 GPT-4.1 驱动的数据方法。完整技术细节请参阅 [arxiv 论文](https://arxiv.org/pdf/2506.10943)。** 评论者强调，SEAL 的递归自我提升真正更新了模型的权重（不同于以往的方法），这种方法代表了 AGI 的重大进展，而计算成本现在是自我监督、自主 LLM 学习进一步取得突破的主要障碍。
    - SEAL 方法与之前的递归框架的区别在于，它实际上允许模型修改自己的权重，而不仅仅是修改其输出或 Prompt 策略。这种直接的自我监督权重更新机制实现了真正的自我提升能力。
    - SEAL 中使用的底层模型是 Llama 3.2 的 10 亿参数变体，这表明解决 72.5% ARC-AGI 任务的这些结果是在一个相对紧凑的模型架构上实现的，这突显了自我提升技术的意义。
    - 自我监督微调被视为模型进步的关键路径，但评论者指出，计算成本仍然是将这一范式推向更深层次的关键限制因素，特别是对于更大的模型或持续的递归改进。
- [**“Anthropic 研究人员教语言模型进行自我微调”**](https://www.reddit.com/r/singularity/comments/1laip79/anthropic_researchers_teach_language_models_to/) ([Score: 357, Comments: 51](https://www.reddit.com/r/singularity/comments/1laip79/anthropic_researchers_teach_language_models_to/))：**Anthropic 及其合作者引入了内部一致性最大化 (ICM)，这是一种针对大语言模型 (LLM) 的无监督微调技术，它利用模型内部的一致性而非人工标注的数据（[论文摘要](https://the-decoder.com/anthropic-researchers-teach-language-models-to-fine-tune-themselves/)）。该方法旨在解决随着 LLM 和任务复杂性增加而带来的人工监督扩展性问题，主张通过奖励保持逻辑自洽的输出来实现模型自我提升。** 讨论集中在行业向自我提升 LLM 趋同的预期，以及与 SEAL 等相关方法的技术比较，表明了对自我监督微调范式的持续探索。
    - 一位用户询问 Anthropic 的自我微调方法与 SEAL 有何不同，并提到了关于类似自我提升机制的持续讨论。SEAL 通常指来自 AI 反馈的半监督强化学习，而 Anthropic 的论文侧重于模型自主生成并根据自己的微调数据采取行动。区别可能涉及反馈流水线控制和数据自主性的差异，需要仔细研读两篇论文以进行精确对比。
    - 有关于 Anthropic 快速进展的讨论，特别提到了基准测试——据称 Opus 4 在工具使用和 Agent 能力方面优于其前身 (Opus 3)、Google 的 Gemini 2.5 以及其他模型。评论者强调 Anthropic 的可解释性研究是其竞争优势，特别是与 OpenAI 和 Google 相比，强调了 AI 研究领导地位的持续技术进步和转变。
    - Anthropic 论文的直接链接 (https://arxiv.org/abs/2506.10139v1) 为读者提供了获取技术方法和据称结果的主要途径，支持对自我微调 LLM 性能和实现细节的进一步分析。

### 2. Claude Code 使用、反馈与生产力技巧

- [**花 20 美元获得 Claude Code 的访问权限真的太不可思议了**](https://i.redd.it/er5ds3xhwk6f1.png) ([评分: 172, 评论: 65](https://www.reddit.com/r/ClaudeAI/comments/1la0zsx/the_20_getting_access_to_claude_code_has_been/)): **该图片展示了 Claude Code 的详细每日使用报告，特别强调了用户在多天内的高 token 消耗以及相关的假设性 API 成本（总计 94.00 美元）。该帖子从技术角度解释了作者如何通过高强度使用，在第一天就赚回了 20 美元的 Claude Pro 订阅成本，这大大超过了按零售价格计算的同等 API 访问所能提供的价值。用户将他们的 Claude 体验与其他 LLM 进行了对比，指出虽然 “roo”（推测为 OpenAI 的 GPT-4 “Turbo” 或类似模型）在某些工作流中仍然更胜一筹，但 Claude Code Pro 凭借慷慨的 context window 和成本效率，大幅削减了代码密集型工作负载的 API 支出。** 评论区诙谐地推测，此类报告可能会促使 Anthropic 提高费率，多位用户分享了类似的节省案例，并评论了当前定价模式被认为的不可持续性，这实际上证实了该方案对高级用户（power users）极高的技术和财务价值。
    - 一位用户分享了他们在个人项目上对不同 AI 供应商的支出：单月在 `Gemini 上花费 500 美元`，在 `OpenRouter 上花费 500 美元`，在 `Anthropic 上花费 700 美元`。他们指出，20 美元的 Anthropic 订阅在处理大量的架构文档任务时很快就会触发 rate-limited（速率限制），从而促使他们升级到 100 美元的方案，这说明了*重度用户的成本效益权衡和使用阈值*（参考：Claude.ai）。
- [**我发现了一种持续改进 Claude Code 的 CLAUDE.md 指令的强大方法**](https://www.reddit.com/r/ClaudeAI/comments/1laby6h/i_discovered_a_powerful_way_to_continuously/) ([评分: 313, 评论: 62](https://www.reddit.com/r/ClaudeAI/comments/1laby6h/i_discovered_a_powerful_way_to_continuously/)): **楼主为他们的 Claude Code 助手指令 (**`CLAUDE.md`**) 实现了一个自动化的持续改进循环，使用** `/project:reflection` **命令促使 Agent 分析最近的聊天记录以发现指令缺失，并提出针对性的改进建议（体现了 prompt engineering 和 instruction tuning 的原则）。确定的主要问题包括缺失的集成指南（例如 Jira/Atlassian 的使用、文档标准、重构策略、项目上下文和增量开发流程）。该方法强制执行结构化反馈、迭代的人工审批和精确的指令更新，紧密跟踪观察到的性能瓶颈和上下文误解。** 一位评论者强调了通过 `.claude/commands` 将指令优化与工具使用相结合的价值，建议在工具选择方面进一步实现自动化；另一位评论者指出，除非明确指示读取，否则 Claude Code 可能会忽略 `CLAUDE.md` 文件，这表明在 context loading 和规范助手行为方面存在技术挑战。
    - a_c_m 分享了该系统的扩展，引入了一个 `.claude/commands` 目录来管理工具使用，强调优化工具调用是提高 Claude Code 有效性的重要杠杆（[gist 链接](https://gist.github.com/a-c-m/f4cead5ca125d2eaad073dfd71efbcfc)）。这种方法强调了命令执行的模块化和细粒度控制。
    - FBIFreezeNow 指出了一个潜在的实际问题：除非明确提示，否则 Claude Code (CC) 并不总是引用 `CLAUDE.md` 指令文件，这影响了遵循指令的一致性。这表明在隐式上下文利用或自动引用行为方面存在局限性，可能会影响 prompt engineering 策略。
    - Fine-Presentation216 提出了一个可维护性方面的担忧，即对 `claude.md` 的迭代更新有引入冗余或重复指令的风险，主张在更新工作流中遵循 “Don't Repeat Yourself” (DRY) 原则。这突显了持续改进与指令膨胀（instruction bloat）之间的权衡。

- [**难道只有我一个人觉得，让 Claude 编程表现惊艳的那些“秘诀”，其实和让其他 AI 模型变得好用的通用技巧是一样的吗？（例如：强大的 CLAUDE.md 文件、将复杂任务拆解为 Markdown 文件、维护持久化记忆库、避免过长的对话/上下文）**](https://www.reddit.com/r/ClaudeAI/comments/1la5kp4/am_i_the_only_one_who_finds_the_secrets_to/) ([Score: 148, Comments: 48](https://www.reddit.com/r/ClaudeAI/comments/1la5kp4/am_i_the_only_one_who_finds_the_secrets_to/)): **该帖子认为，所谓的最大化 Claude 等模型 AI 编程性能的“秘密”最佳实践，在各种 LLM 编程助手（如 Copilot、Aider、Gemini）中很大程度上是通用的。关键建议包括：维护一个详细的、手动编写的 'CLAUDE.md' 项目架构文件以压缩上下文；将复杂任务拆分为细粒度的 Markdown 文件，以实现持久的任务历史和上下文效率；使用持久化记忆 Artifacts（CLAUDE.md, CHANGELOG.md）；缩短对话长度以避免模型混淆；并优先考虑强大的模块化单元测试以减少 Bug 修复的递归。这些实践利用了模型的优势（意图明确时的精确性、上下文效率），并缓解了其弱点（长上下文退化、上下文槽限制），并声称除了在范围明确的 Agent 框架中，进一步的优化收益递减。** 热门评论介绍了一种多 Agent 工作流，其中具有独特身份的不同 Claude Agent 在不同功能上并发运行，通过共享的 'developer_coms' 目录进行通信，并协作解决 Git 冲突，模拟了项目管理的最佳实践。其他人证实了分层、相互引用的 Markdown 文件对于维持同步上下文的价值，并提出了结构化的文件层级（Claude.md → Project_todo.md → Feature_todo.md → Sprint_todo.md）。共识是，有效的 AI 辅助编程反映了严谨的项目管理方法论。
    - 描述了一个详细的多 Agent 工作流：生成多个 Claude Agent 实例，每个实例作为一个独立的开发者（通过唯一的 `.identity` 文件），在并行终端中处理不同的代码库功能。Agent 通过共享的 `developer_coms` 目录进行协调，在每个单独任务后解决 Git 冲突，并可以就项目更新达成共识或进行投票，有效地模拟了协作开发环境，并展示了 Agentic 项目管理技术的威力。
    - 引用和链接 Markdown 文档文件（`Claude.md` → `Project_todo.md` → `Feature_todo.md` → `Sprint_todo.md`）创建了一个维护良好的依赖/上下文图谱。这种结构有助于在所有规划层级中进行模型更新，确保在任务或依赖项发生变化时上下文的完整性和同步。它利用了 Claude 保持不同文档同步并将更改传播到整个层级的能力。
    - 讨论了在详细规划之前使用 Claude 对整个代码库进行索引和分析。这种设置涉及生成一系列规划文档（例如 plan.md、架构、API、后端规范），然后要求 Claude 创建一个分阶段的、由检查清单驱动的任务计划。这与 AI 增强规划的实践相一致——预先加载上下文吸收和显式的检查清单创建，提高了大型或复杂编程工作流的鲁棒性。

### 3. AI 与编程工具更新与发布 (Roo Code 3.20.0, MagCache, LoRA-Edit)

- [**Roo Code 3.20.0 | 这是一个重大更新！！**](https://www.reddit.com/r/ChatGPTCoding/comments/1la61eo/roo_code_3200_this_is_a_big_one/) ([Score: 127, Comments: 31](https://www.reddit.com/r/ChatGPTCoding/comments/1la61eo/roo_code_3200_this_is_a_big_one/)): **Roo Code 3.20.0 引入了一个实验性的扩展（MCPs）和模式市场，支持直接在 UI 中进行项目/全局范围的安装和管理（[文档](https://docs.roocode.com/update-notes/v3.20.0#mcp--mode-marketplace-experimental)），以及用于批量重构的实验性多文件并发编辑（[详情](https://docs.roocode.com/features/experimental/concurrent-file-edits)）和并发文件读取（现在上下文设置中默认为 5 个并发读取）。Prompt 历史导航现在模仿了终端的 UX，此次更新还带来了 17+ 项改进和 Provider 支持更新（DeepSeek R1, Bedrock reasoning, XAI, O3, OpenRouter）。完整变更日志见[此处](https://docs.roocode.com/update-notes/v3.20.0)。** 一位技术评论者质疑 Roo Code 维护者的透明度，指出 GitHub 页面上看不到贡献者或作者的署名——这是一个关乎开源信任和协作的问题。

- 关于 Roo Code GitHub 页面上开发者的可见性和归属存在一个技术问题。一位评论者指出，Roo Code 背后的贡献者或团队在仓库中不可见，这可能会阻碍用户和其他开发者的透明度和开源信任。这可能会影响项目的审计、信任和协作贡献。
- 用户寻求关于新 MCP Marketplace 的集成和可用性功能的澄清，特别是是否可以在 RooCode 环境之外浏览，以及如何向该市场提交内容。这凸显了对市场可扩展性和第三方贡献机制的兴趣，以及核心 IDE 之外的 API 或 UI 暴露。
- [**MagCache，TeaCache 的继任者？**](https://v.redd.it/6kep8ze8vm6f1) ([Score: 180, Comments: 15](https://www.reddit.com/r/StableDiffusion/comments/1la8e7m/magcache_the_successor_of_teacache/)): **MagCache 被呈现为 TeaCache 的继任者，其实现目标是用于扩散模型加速的 ComfyUI（链接：[网站](https://zehong-ma.github.io/MagCache/)，[GitHub](https://github.com/Zehong-Ma/ComfyUI-MagCache)）。在高端硬件（如 H100 SXM GPU）上的早期用户测试指出，缺乏 Skip Layer Guidance 支持，并且观察到与 TeaCache 相比，速度提升非常有限（**`~8 sec`**），且采样质量较差，特别是在 Wan T2V 14B 上。关于强制使用** `torch.compile` **的兼容性问题被提出，因为它需要** `80 SMs` **（Streaming Multiprocessors），将支持限制在顶级 NVIDIA 硬件（4080/5080 系列及以上）。** 评论者普遍对 MagCache 相对于 TeaCache 的性能持批评态度，强调输出质量下降和实际加速有限是主要缺点。关于硬件要求的争论也存在，用户对由于 torch.compile 所需的高 SM 数量而导致的狭窄兼容性表示担忧。
    - 在 H100 SXM 上测试 MagCache 显示，虽然它比 TeaCache 提供了 `8 秒` 的速度提升，但在使用 **Wan T2V 14B** 的推荐设置时，生成结果的质量明显较差。由于没有 Skip Layer Guidance 等功能，感知到的改进有限，迫使用户降低设置以换取微小的收益。
    - 关于 `torch.compile` 是否为 MagCache 运行所必需提出了一个技术问题。担忧在于 `torch.compile` 需要至少具有 `80 SMs (Streaming Multiprocessors)` 的 NVIDIA GPU，这意味着许多消费级 GPU（如 4060Ti, 4070）无法使用它，可能将 MagCache 的使用限制在高端设备（4080/5080 及以上）。
    - 对于 Flux，MagCache 的图像质量被描述为较差，尽管由于强大的构图保真度，它在快速生成预览方面可能仍优于之前的缓存方法。尽管如此，它对于高质量输出的效用可能有限。
- [**LoRA-Edit: 通过掩码感知 LoRA 微调实现可控的首帧引导视频编辑**](https://v.redd.it/tu3gpipkcm6f1) ([Score: 176, Comments: 9](https://www.reddit.com/r/StableDiffusion/comments/1la6nta/loraedit_controllable_firstframeguided_video/)): **LoRA-Edit 引入了一种掩码驱动的 LoRA (Low-Rank Adaptation) 微调策略用于视频编辑，利用预训练的 Image-to-Video (I2V) 扩散模型进行可控的首帧引导编辑。该方法使用空间掩码将背景保留与目标编辑传播隔离开来，通过动态注意力调制结合来自输入视频（运动、空间结构）和参考图像（外观）的线索，以支持特定区域的学习，根据实验结果优于最先进的方法。该方法不改变核心模型架构并支持灵活适配；代码可在 [GitHub](https://github.com/cjeen/LoRAEdit) 获取。** 评论者要求将 LoRA-Edit 与 ComfyUI 集成，表明在成熟的 UI 框架中对更广泛的可访问性和工作流兼容性的需求。
    - 两位用户请求或期待 LoRA-Edit 与 ComfyUI 的集成，表明希望通过实用的封装和基于 UI 的工作流，在现有流程中利用这种新的可控视频编辑技术。
    - 一条评论对 "Ours" 演示中显示的结果的可靠性表示怀疑，这暗示了社区中对于新方法的再现性以及实际性能与精心挑选的演示之间差异的更广泛担忧。

---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 生成的摘要的摘要的摘要

**主题 1. 跨平台的基础设施不稳定性罢工**

- **Cloudflare 和 Google Cloud 停机导致 AI 服务中断**：**Cloudflare** 和 **Google Cloud Platform** 的大范围停机使包括 **Cursor**、**OpenRouter** 和 **Cohere** 在内的多个 AI 平台陷入瘫痪，导致登录和核心功能受阻。 [Cloudflare](https://www.cloudflarestatus.com/) 和 [Google Cloud](https://status.cloud.google.com/) 的状态页面详细说明了这些问题，[OpenRouterAI 通过其 X 账号指出已出现恢复迹象](https://x.com/OpenRouterAI/status/1933263905385500853)。
- **网络带宽差距扩大**：典型互联网连接的带宽约为 **1gbps**，与 [NVIDIA 最新的 Infiniband 迭代达到 130TB/s](https://x.com/toolandtea/status/1933381389552136705) 形成鲜明对比，凸显了网络能力差距的扩大，这影响了分布式训练效率。像 [DAWN Internet](https://x.com/dawninternet) 这样使用固定无线并在其路由器中包含具备 RL 能力的 GPU 的去中心化方案，被提出作为传统供应商的替代方案。
- **云依赖导致 LlamaCloud 波动**：**LlamaCloud** 由于上游基础设施问题经历了运行不稳定，这强调了 AI 服务对外部云供应商的依赖程度，促使用户关注 [官方状态页面](https://t.co/IdecAksHiG) 以获取实时更新。这一事件与其他事件共同凸显了依赖第三方云服务所固有的脆弱性。

**主题 2. 模型性能、能力与特性**

- **模型性能争论激烈**：用户对模型偏好和能力展开辩论，讨论比较了 **o3** 和 **2.5 Pro** 的通用性能与数学优势，而像 [MathArena](https://matharena.ai/) 这样的基准测试因潜在的饱和问题受到审查。新的 [文本转视频模型 Seedance 和 Kangaroo](https://artificialanalysis.ai/text-to-video/arena) 在最近的对比中可能优于 **Veo3**、**Kling** 和 **Pika**，给用户留下了深刻印象。
- **下一代模型暗示内部工具使用和并行处理**：据报道，**GPT-5** 的架构依赖内部专用工具来增强上下文和稳定性，而一种领先的理论认为 OpenAI 的 **GPT Pro 模型**（如 **O3 Pro**）通过并行运行多个实例并整合结果来提高推理能力，这可能解释了 **O3-pro** 在 **AIME 2024** 数学竞赛中 **93%** 的准确率（相比之下 **O3** 为 **90%**）。尽管如此，一些用户反映 **O3 Pro** 在长时间等待后仍无法回答上传文档中的问题。
- **模型局限性和偏见评估浮出水面**：用户注意到 **Gemini Pro 2.5** 在简单图像识别方面表现吃力，且本地 LLM 在处理大上下文窗口时存在困难，而偏见评估显示，添加现实细节可能会触发 **GPT4o** 和 **Claude 4 Sonnet** 等模型中的**种族和性别偏见**。LLM 绕过不断更新的验证码（captcha）的概念被比作 [“红皇后假说”（Red Queen hypothesis）](https://en.wikipedia.org/wiki/Red_Queen_hypothesis)，即进步很快就会被新的防御措施抵消。

**主题 3. 硬件与底层优化之战**

- **AMD GPU 获得关注和 Unsloth 支持**：**Unsloth** 团队表示有兴趣支持 **AMD** GPU，强调了新的 **AMD INSTINCT MI355X GPU** 的 **FP8 flops** 是 **H100** 的 **5 倍**，并指出 AMD 在性价比和高显存方面的优势，尽管驱动支持仍是一个问题。**GemLite** 也宣布增加 **ROCm 支持**，重点针对 **MI300X**，并通过 **LLVM intrinsics** 和 **Mojo APIs** 实现自定义 mma 指令，详情见 [这篇 X 帖子](https://x.com/mobicham/status/1933520405106507909) 和 [博客文章](https://veitner.bearblog.dev/programming-tensor-cores-in-mojo/)。
- **Torch.compile 和 CUDA 库提升性能**：成员们发现使用 **torch.compile** 处理卷积算子可显著提速，通过调用外部算子将性能从 **1.7489 ms** 提升至 **0.0020 ms**。**CUDA** 相关的讨论探索了 **Blackwell** 的内存布局优化库，利用 [cuda-side-boost](https://github.com/ademeure/cuda-side-boost) 进行开发，并配置 L1/L2 缓存策略。
- **硬件决策权衡显存与成本**：关于使用二手 **Tesla P40** 进行 **24GB VRAM** 扩展（约 **300 美元**）的争论出现，共识认为与作为更好“平价”选择的二手 **3090** 相比，这并不值得。围绕最佳本地 LLM 性能的讨论涉及了实现“合理的类人交互”需要 **150B+ 参数**，以及 Prompting、RAG 和微调的重要性。

**主题 4. 使用 AI 开发：工具、Agent 与 API**

- **编程助手迎来新功能与修复**：**Cursor** 面临 Cloudflare/GCP 停机、代码生成卡顿以及后台 **Agent** 隐私/提交问题；而 **Claude Code** 在处理复杂重构的上下文方面表现出色。**Aider** 用户称赞其通过 **Ollama** 配合 **8B** 和 **12B** 等小型本地模型的表现，将其成功归功于 **repomap**，同时讨论了 **Anthropic** 的成本以及使用 **UV** 进行依赖管理。**Windsurf (Codeium)** 推出了 [Wave 10 UI/UX 升级](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise)、[欧盟集群](https://youtu.be/UHinqQiiCI8?si=udyZDkWGg9nq7zcI)，并[增加了对 Claude Sonnet 4 的支持](https://x.com/_mohansolo/status/1933605162775687482)。
- **Agent 框架迎来新工具与安全措施**：支持 AI **Agent** 的新工具不断涌现，包括 **Taskerio** 用于[通过 webhook 和 API 跟踪编程 Agent 进度](https://www.reddit.com/r/mcp/comments/1lac12i/an_mcp_to_track_the_progress_of_your_ai_agents/)的收件箱，以及旨在加强 **MCP** 以防御 "Rug Pull" 漏洞利用的 [SchemaPin](https://github.com/ThirdKeyAI/SchemaPin)（其简单的实现细节见 [SchemaPin.org](http://schemapin.org/)）。**GitHub** 发布了一个[远程 MCP 服务器](https://www.reddit.com/r/mcp/s/Cj2zjute95)，允许主机使用动态工具选择访问实时上下文；此外还分享了一份关于使用 Postman 构建器和 API [构建 MCP 服务器的指南](https://youtu.be/YzT9QU-Kfh4?si=tqqgBXXu9ct2aMUH)。
- **平台集成模型并提升易用性**：**LlamaIndex** 增加了[对 MistralAI 的 Magistral 模型支持](https://t.co/ZsUEWMrnT4)，并引入了 **LlamaParse Presets** 以平衡解析准确性和速度，同时集成了 **Mem0** 用于 **Agent** 工作流中的自动记忆更新。**NotebookLM** 用户请求支持 Excel/Sheets，并报告了移动端笔记显示和分享功能的问题。**OpenRouter** 用户讨论了不同供应商之间的质量差异，并请求未来加入音频/视频生成等多模态功能。

**主题 5. AI 研究概念与辩论**

- **AI 安全与偏见引发讨论**：对新的 AI 安全研究所产生质疑，理由是缺乏先前的知名度和出版物。研究强调了在**偏见评估**中添加现实细节如何触发 **LLM** 中的**种族和性别偏见**，导致模型间模拟结果出现高达 **12%** 的差异，并指出 **Chain of Thought** 方法未能揭示这种隐藏偏见，详见关于[稳健提升 LLM 公平性 (Robustly Improving LLM Fairness) 的论文](https://x.com/a_karvonen/status/1933582375419850806)。
- **评估方法与基准测试受到审视**：对使用**过河实验 (River Crossing experiments)** 等任务评估 AI 推理提出了批评，指出根据[《思维错觉的错觉》(The Illusion of the Illusion of Thinking) 论文](https://arxiv.org/abs/2506.09250)，正确识别不可解问题的模型反而被无意中扣分。随着分数接近 **100%**，关于 **MathArena** 等基准测试有效性的辩论仍在继续。
- **核心 AI 概念辩论**：讨论涵盖了 **LLM** 的 **RL** 训练中 **KL 散度梯度估计**的陷阱，强调了 **GRPO** 等开源项目和论文以及[关于 KL 散度陷阱的论文](https://arxiv.org/pdf/2506.09477)中的问题，并质疑了“符号递归 (symbolic recursion)”等术语的含义。继 [Fortune 文章](https://fortune.com/2025/06/11/nvidia-jensen-huang-disagrees-anthropic-ceo-dario-amodei-ai-jobs/)及随后的 [X 帖子](https://www.x.com/dario)之后，**Jensen Huang (Nvidia)** 与 **Dario Amodei (Anthropic)** 之间关于 AI 就业未来的高层分歧也引起了关注。


---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini Deep Think 即将到来**：一名成员分享了一张图片 ([aSHuTrz.jpeg](https://cdn.discordapp.com/attachments/1047649527299055688/1382930529464352848/aSHuTrz.jpeg?ex=684d9aab&is=684c492b&hm=b30084937010eeeb0b4dba66cb69c6fd085c52fb786ce6e316e2324b2f50d0aa&))，暗示 **Gemini Deep Think** 即将发布。
   - 目前没有提供关于 **Gemini Deep Think** 具体细节的进一步信息。
- **Perplexity Pro Role 功能失效**：用户报告在 Discord 上获取 **Perplexity Pro role** 时遇到问题，称入站（onboarding）按钮无法正常工作。
   - 建议的解决方法是私信工作人员手动分配角色，一位用户指出：*"该按钮似乎没有给我角色，只是让我通过手机进入了服务器"*。
- **Perplexity Pro 现在可以画图了**：成员们发现 **Perplexity Pro** 可以根据在搜索栏中输入的文本提示词生成图像。
   - 此外，用户还提供了优化图像的指令，可以通过点击 **Regenerate** 或发送带有 *cinematic*、*anime* 和 *low-poly 3D* 等风格的新提示词来进行调整。
- **GPT-5 思考更聪明，而非更费力**：一名成员分享了关于 **GPT-5** 架构的细节，强调其对内部专业化工具的依赖，从而避开了外部路由和幻觉（hallucinations）问题。
   - 一位成员表示 *"GPT-5 thinks with its tools, not beside them"*（GPT-5 与其工具共同思考，而非独立于工具之外），强调了增强的上下文、协调性和稳定性。
- **Sonar API 文档需要仔细审查**：Perplexity 团队正在征求用户对 **Sonar API documentation** 的反馈，特别是关于不清晰或难以导航的部分，可在[此社区帖子](https://community.perplexity.ai/t/improvements-to-the-sonar-api-documentation/542?u=vikvang)中查看。
   - 此次反馈旨在根据用户体验改进文档。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O3 Pro vs 2.5 Pro：巅峰对决**：成员们激烈争论模型偏好，一些人支持 **o3** 的综合性能优于 **2.5 Pro**，而另一些人则引用了 **2.5 Pro** 在数学方面的强势表现。
   - 针对他人对 **2.5 Pro** 数学能力的偏爱，一位成员调侃道：*"我希望生活在这种程度的幻想中"*。
- **MathArena Benchmarks：它们仍然有效吗？**：社区讨论了 [MathArena benchmarks](https://matharena.ai) 的持续有效性，一些人认为它们正趋于饱和，且受运气驱动。
   - 担忧在于，接近 **100%** 的分数可能表明测试已饱和，从而降低了这些基准测试的统计学意义。
- **Kingfall 发布导致 Google 账号被封**：一位用户的 Google 账号被封引发了关于新 **Kingfall** 发布的猜测，同时一个代号为 *toothless* 的新 Gemini 模型短暂出现。
   - 甚至有报告称在各种项目中获得了 **99% 的利润**，引发了对该模型能力的猜测。
- **Text-to-Video Arena：Seedance 和 Kangaroo 登场**：在 [text-to-video arena](https://artificialanalysis.ai/text-to-video/arena) 中，**Seedance 1.0** 和匿名的 **Kangaroo** 模型表现令用户印象深刻。
   - 对比显示，这些模型在根据通用提示词生成相似输出方面，潜力可能超越 **Veo3**、**Kling** 和 **Pika**。
- **云服务崩溃导致聊天记录灾难**：由于 **6/12/25** 的云服务商故障，团队警告 **chat history data** 可能已经丢失。
   - 团队对造成的不便表示歉意，并指出他们正在 *研究解决方案以确保此类事件不再发生*。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cloudflare 故障瘫痪 Cursor**：一次 **Cloudflare** 和 **GCP** 的故障导致 **Cursor** 瘫痪，中断了登录和核心功能，尽管据报道 [Tab 仍可正常运行](https://www.cloudflarestatus.com/)。
   - 这次中断凸显了开发工具对外部服务的依赖，相关问题随后被[标记为已解决](https://mashable.com/article/google-down-cloudflare-twitch-character-ai-internet-outage)。
- **Cursor 代码生成不连贯问题持续**：用户仍在反馈开启自动模型选择时 **Cursor 的代码生成** 存在问题，一位用户抱怨由于代码输出混乱损失了 **50 个推理额度 (inference credits)**。
   - 一位用户询问如何使用 Cursor 制作 three.js 游戏，另一位用户建议大多数编码工作使用 O3，规划和调试则使用 O3-pro，强调其效果优于其他模型。
- **Claude Code 命令的上下文能力**：用户发现 **Claude Code** 在理解上下文和产出高质量代码方面表现出色，尤其是在处理复杂的重构时。
   - 它帮助为一个前端组件库添加了 **3500 个新测试**，证明了其能力；这突显了它有效处理大规模代码修改的能力。
- **隐私模式阻碍 Background Agents 运行**：用户在尝试启动 background agent 时遇到了错误提示：*Background agent is not supported in privacy mode*，这是由于启用了**账户级隐私模式**。
   - 该问题可以通过[此链接](https://www.cursor.com/slack-connected)解决，并计划在下一个版本中彻底修复。
- **Background Agents 破坏提交规范**：一个 background agent 在修改 commit 后，尝试将更改后的 commit 推送到仓库时遇到障碍，暗示存在一些版本控制方面的冲突。
   - 一位成员建议通过终端解决，因为 agent 正在被回滚，这暗示了 agent 在处理版本控制操作时可能存在的问题。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Google Cloud 崩溃**：**Google Cloud** 遭遇重大故障，正如其[状态页面](https://status.cloud.google.com/)所示，用户反馈在东部时间 **下午 4:25** 出现初步恢复迹象后仍有间歇性问题。
   - **OpenRouterAI** 发布推文称观察到故障正在恢复，并希望这不会是暂时的 ([推文链接](https://x.com/OpenRouterAI/status/1933263905385500853))。
- **Cloudflare 再次导致网络中断**：大范围的 [Cloudflare 故障](https://www.cloudflarestatus.com/) 导致了严重的干扰，使包括 **OpenRouter**、Google 在内的众多 AI 服务下线。
   - 用户经历了间歇性的 **OpenRouter** 服务中断，状态页面在“重大故障 (MAJOR)”和“轻微故障 (MINOR)”之间切换。
- **供应商差异影响模型质量**：用户讨论了通过 **OpenRouter** 提供相同模型的不同供应商之间存在的显著质量差异，并指出 **Parasail** 和 **Lambda** 通常提供更一致的性能。
   - 一位用户强调了质量优于成本的重要性，指出[不同供应商的质量差异很大，因此要明智选择](https://discord.com/channels/1091220969173028894/1092729520181739581/1383133709551282236)。
- **廉价 Agent LLM 成为顶级工具使用者**：用户讨论了用于 Agent 工具调用的最佳廉价 **LLM**，**Claude 2.5 Flash** 被推荐为高性价比的选择，但需要精细的 Prompt。
   - 讨论还涉及了像 **O4 Mini High** 这样模型的高昂成本，以及使用 [每月 Claude Max 订阅](https://discord.com/channels/1091220969173028894/1195014798837043240/1383046909199872050) 进行 API 调用的效率。
- **期待 OpenRouter 多模态功能**：成员们请求 **OpenRouter** 平台未来支持 **音频** 和 **视频生成** 等多模态功能。
   - OpenRouter 尚未给出明确答复。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 缺乏自动模型更新**：与 **Ollama** 不同，**LM Studio** 不会自动下载模型更新；大多数模型更新发布在新的仓库中，导致模型谱系（lineage）难以追踪。
   - 一位成员指出，这使得确定模型的谱系变得困难。
- **Gemini Pro 图像识别出错**：一位用户报告称，尽管使用了各种提示词和图像，**Gemini Pro 2.5** 在简单的图像识别中仍会出现错误，即使是使用[提供的图像](https://cdn.discordapp.com/attachments/1110598183144399058/1382962931741888563/image.png?ex=684db8d9&is=684c6759&hm=14fbd9fe32c10ead609c4627acfd3c543cdf7ca6f347af0e6c9a61470af4c663&)。
   - 另一位成员提到，具备视觉能力的模型通常表现不佳，且用户的期望并不明确。
- **LLM 在升级后的验证码面前表现挣扎**：成员们强调了使用 **LLM** 绕过验证码所面临的持续挑战，因为验证码旨在抵御计算机破解并不断更新。
   - 这种情况类似于[红皇后假说](https://en.wikipedia.org/wiki/Red_Queen_hypothesis)，即验证码破解技术的进步会迅速被新的防御手段抵消。
- **OpenWebUI 实现远程 LM Studio 访问**：**OpenWebUI** 通过托管服务器、加载模型、在本地网络提供服务、启用 **CORS** 并开放 1234、8080 或 3000 等端口，方便在服务器上运行 **LM Studio** 以进行远程访问。
   - 访问端的 PC 不需要安装 **OpenWebUI**。
- **Tesla P40 不再值得购买**：一位成员询问是否可以使用 **Tesla P40** 作为额外的 **GPU**（如 **RTX 3090/4090**）来为 **LM Studio** 扩展 **VRAM**，价格约为 **300 美元**（**24GB**），并链接到了 [TechPowerUp 规格](https://www.techpowerup.com/gpu-specs/tesla-p40.c2878)。
   - 共识是 **300 美元** 的价格点已经不再值得，因为二手的 **3090** 是更好的“实惠”选择。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI 安全研究所面临质疑**：成员们对一家新的 AI 安全研究所的合法性表示怀疑，理由是此前缺乏认知，且其网站上没有近期的出版物，不过该顾问在 Discord 上。
   - 一位成员建议发起联系，并指出该顾问就在 Discord 上。
- **德语文本揭示 LLM 的怪癖**：一段简短的德语文本引发了 **GPT-3** 和 **GPT-4o** 截然不同的反应，范围从中性到深层的情感响应。
   - 该成员质疑这种异常现象是否值得进一步调查，暗示有兴趣探索常规应用之外的 **LLM** 行为。
- **Symbolica.ai 关注定理证明器模型**：[Symbolica.ai](https://www.symbolica.ai/) 是一家总部位于伦敦的初创公司，其目标宏大，但他们应该发布一个像 Google 那样的小型定理证明器模型。
   - 一些评论建议，“工作的边界不清晰，目标一直在变”。
- **GRPO 目标函数大幅提升模型性能**：**DeepSeek V3**（一个 **671B** 模型）通过 [GRPO 目标函数](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f) 展示了增强的性能，成功通过了验证任务。
   - 一位成员指出，“字面意义上的随机奖励也能提高性能”，这是由于“集中效应（concentration effect）使模型专注于其现有的推理模式分布”。
- **偏见评估触发种族和性别偏见**：在现有的 **bias evals** 中添加现实细节会触发 **LLM** 中的**种族和性别偏见**，导致包括 **GPT4o** 和 **Claude 4 Sonnet** 在内的各种模型在面试率上出现高达 **12% 的差异**。
   - 关于[稳健提升 LLM 公平性](https://x.com/a_karvonen/status/1933582375419850806)的论文给出了一个现实中**不忠实思维链（unfaithful chain of thought）**的例子。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Canvas 支持代码和文档导出**：**Canvas** 现在支持下载和导出，允许用户将文档导出为 **PDF**、**docx** 或 **markdown** 文件。
   - 此外，**Canvas** 还支持将代码直接导出为相应的文件类型，如 **.py**、**.js** 和 **.sql**。
- **GPT Pro 的并行效能策略**：一个主流理论认为，**GPT Pro 模型**（如 **O3 Pro**）通过并行运行多个实例并整合结果（即“更努力思考”的方法）来增强推理能力。
   - 来自 **AIME 2024** 数学竞赛的证据显示，**O3-pro** 达到了 **93%** 的 pass@1 准确率，而 **O3** 为 **90%**，这暗示了这种整合方法的有效性。
- **O3 Pro 性能在项目应用中受挫**：有用户报告称，尽管等待时间长达 **40 分钟**，**O3 Pro** 经常无法回答来自上传文档的问题。
   - 这种表现不佳引发了对其实际效用的质疑，与其增强的推理能力形成对比。
- **免费 AI API 助力开发**：开发者们探索了免费的 **AI** **API**，例如 **SambaNova**，它提供快速的 **Llama 3.3 70B**、**Qwen** 和 **Deepseek** 模型。
   - **Gemini** 因其高频率限制而受到关注，为 **2.5 Flash** 提供 **500/天**，为 **2.0 Flash** 提供 **1k/天**，非常适合预算有限的项目。
- **AI 激增导致 Discord 活跃度下降**：**Discord** 活跃度的显著下降与 **AI** 聊天的普及相关，导致许多服务器变成“鬼城”，这促使人们重新思考社区参与方式。
   - 这种转变表明用户正在迁移到 **AI** 驱动的平台进行讨论，影响了传统平台上的社区参与度。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Qwen 2.5 精通 100 种语言**：**Qwen 2.5** 支持 100 种语言，这可能归功于其 **18T tokens** 训练中包含的大量多语言数据，并很好地利用了 **Linux VM**。
   - 成员们将其与 **Gemma3** 进行了比较。
- **CloneMe 创建数字孪生**：[CloneMe AI 平台](https://github.com/vibheksoni/cloneme)允许你构建自己的**数字孪生**——一个像你一样聊天、记住细节并支持多平台的 **AI**。
   - 它是可定制的、**内存驱动**的且支持**热重载**，是创建智能、动态 **AI personas** 的工具包。
- **HF 因开源表象面临质疑**：有人认为 **Hugging Face** 上的某些模型并非真正的**开源**，暗示该平台可能更多被用于营销而非真正的协作。
   - 虽然 **Hugging Face** 没有明确将自己标榜为*开源*库，但其声誉表明事实并非如此。
- **TensorBoard 揭示拟合情况**：成员们正在使用 **TensorBoard** 损失图来诊断模型拟合情况，强调评估损失（evaluation loss）应以与训练损失（training loss）相似的速度下降。
   - 将数据集分为**训练**和**测试**部分可确保模型在不发生过拟合或欠拟合的情况下具有良好的泛化能力。
- **Augmentoolkit 3.0 增强 AI**：[Augmentoolkit 3.0](https://github.com/e-p-armstrong/augmentoolkit) 允许用户通过添加文档或通过评分尝试来教导任务，从而在新的主题上训练 **AI**。
   - 它促进了**自定义模型**的运行，为更新时间和方法提供了更大的控制权。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **疑似 Veo3 发布后导致 Manus 崩溃**：用户报告 **Manus** 出现大范围问题，怀疑是 **Veo3 发布** 导致服务器过载，[Downdetector](https://downdetector.com/) 已确认。
   - 停机引发了不满，一位用户报告说 *我启动的每个任务大约都要消耗 900-1000 个 credits*。
- **Playbooks 预先准备 Prompt**：**Playbooks** 在 Manus 中准备 Prompt 并提供输出示例，为需要 Prompt 协助的用户弥补了差距，并[突出了创意工作流](https://manus.im/playbook)。
   - Playbooks 旨在提供结构化指导，使 Prompt Engineering 更加容易。
- **社区持续呼吁 Claude**：用户表达了对 **Claude 4.0** 的渴望，幽默地将其比作粉丝的期待，尽管目前还没有官方消息或更新。
   - 一位用户建议了一个变通方法：*注册新的 gmail 并加入 google one ai 试用 -> 开启家庭共享 -> 邀请 5 个账号 -> 现在 veo 和所有功能都有 5 倍的使用量，因为所有账号都有独立的额度限制*。
- **额度紧缺引发成本担忧**：用户对 **Credit** 使用情况表示担忧，特别是关于优化和缺乏成本预览的问题，一些人建议采用 *bring your own keys*（自带 API Key）模式。
   - 成本透明度的缺乏在社区中引起了一些恐慌。
- **GPT 生成效果优于 Manus**：对比了 Manus 和 GPT-4 Omni 的图像生成质量，结果显示 [GPT-4 Omni](https://cdn.discordapp.com/attachments/1349440650495398020/1382980409243209859/GPT4omini.png?ex=684dc920&is=684c77a0&hm=1d615e514982fcfdfb5677c8640ae6d7ea8282e5f56d86e5382a2578a0084b82&) 的表现优于 Manus。
   - 该对比突出了 GPT-4 Omni 在特定情况下提供了更优的图像输出。



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Torch Compile 奇迹般加速卷积**：成员发现 **torch.compile** 生成的操作使卷积算子的速度显著提升，从 **1.7489 ms**（原生 PyTorch）缩短至 **0.0020 ms**（编译后的 PyTorch）。
   - 产生了一个疑问：为什么标准的卷积不使用通过 *extern_kernels.convolution* 调用更快的外部算子，而是使用 *aten.convolution*。
- **CUDA-side Boost 库加入战场**：一位成员分享了 [cuda-side-boost](https://github.com/ademeure/cuda-side-boost) 的链接，这是一个用于 **CUDA** 开发的库，并指出 *更换整个 PyTorch 内存分配器可能有点大材小用*。
   - 他们建议可以在 **PyTorch** 中改用 **MemPool**。
- **GemLite 增强对 ROCm 的支持**：开发者宣布为 **GemLite** 增加 **ROCm 支持**，重点针对 **MI300X**（[X 上的帖子](https://x.com/mobicham/status/1933520405106507909)）。
   - 该帖子详细介绍了通过 **LLVM intrinsics** 实现 **自定义 mma 指令**，并利用 **Mojo 的 load_matrix** 和 **store_matrix API** 高效管理 **数据布局**（[GitHub 仓库](https://github.com/simveit/mma_mojo)）。
- **PMPP 之后的新手寻求阅读材料**：成员在完成 **PMPP** 后寻求书籍或论文推荐，一位成员推荐了[一篇关于指令延迟的论文](https://arxiv.org/pdf/1903.07486)。
   - 该成员认为，尽管 **指令延迟** 的数据可能已经过时，但 *讨论本身非常值得一读*。
- **Factorio RL 大战开启**：成员讨论了使用 **基于 RL 的 AI** 玩 Factorio 的潜力，辩论是否需要 LLM 进行长期规划和复杂任务。
   - 对话探讨了 RL Agent 是否能在有限的游戏次数内实现 Factorio 的最优玩法，并与 OpenAI Five 在 Dota 2 中的成功进行了对比。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **AMD Instinct MI355X GPU 获得 Unsloth 支持**：Unsloth 团队可能会支持 **AMD** GPU，新款 **AMD INSTINCT MI355X GPU** 的 **FP8 算力是 H100 的 5 倍**，他们已在 [AMD AI 大会](https://x.com/danielhanchen/status/1933150291719082098)上进行了展示。
   - 成员们指出 **AMD 价格低廉且显存高**，但也对 **AMD 的驱动支持**提出了质疑。
- **Unsloth 考虑创建 YouTube 频道**：Unsloth 团队正在考虑创建一个 **YouTube 频道**来上传视频，重点关注教程内容。
   - 一位成员请求制作一段关于如何通过 **accelerate 使用多 GPU** 的视频，并承诺会*点赞并订阅*。
- **AttributeError 困扰 Unsloth 训练会话**：一位用户在利用 Unsloth 训练时遇到了 **AttributeError**，经追溯发现是 `fetch_image` 函数试图读取一个 `None` 的 **images 字段**，而非有效的路径或 URL。
   - 建议是使用 **batch size 1** 或传递自定义的 collator。
- **KL 散度梯度估计存在缺陷**：分享的一篇论文讨论了 LLM 的 RL 训练中 **KL 散度梯度估计**的陷阱，强调了 **TRL** 和 **Open Instruct** 等开源项目以及 **GRPO** 等论文中存在的问题。
   - 论文指出，*将 KL 估计值作为损失函数进行微分*以及*未考虑序列特性*会导致错误的 KL 梯度，参考[这篇论文](https://arxiv.org/pdf/2506.09477)。
- **过河问题错误困扰 Apple 的推理模型**：分享了一篇题为《[思维幻觉的幻觉](https://arxiv.org/abs/2506.09250)》的论文，批评了 **过河实验 (River Crossing experiments)** 中对 AI 模型的评估，认为其无意中惩罚了那些正确识别出问题不可解的模型。
   - Apple 的原始论文中存在 **N ≥ 6 个角色/智能体** 使用船只容量 **b = 3** 的案例，这在数学上是不可能的。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Nano-vLLM 引起关注**：DeepSeek 发布了 **nano-vLLM**，这是一个仅约 **1200 行代码**的极简 vLLM 实现，是 AI/ML 从业者的宝贵学习资源，可通过[此链接](https://xcancel.com/wzihanw/status/1933225058031288772?s=46)查看。
   - 社区对其简洁性表示赞赏，认为它是极具价值的学习资源，并表达了对这个“纳米单体 (nano monolith)”进行改造的兴趣。
- **Trinity 自动形式化费马大定理**：**Morph Labs** 推出了 **Trinity**，这是一个自动形式化系统，用于在 Lean 中形式化 de Bruijn 关于 abc 猜想的结果，详见[此链接](https://xcancel.com/morph_labs/status/1933181394588483868?s=46)。
   - 其目标是通过将数学知识转化为形式化证明，为数学领域的自监督强化学习 (RL) 创建经过验证的训练环境。
- **Transformers 库弃用 TensorFlow 和 Flax 支持**：**Transformers 库**将弃用对 **TensorFlow 和 Flax** 的支持，转而专注于 **PyTorch**，以减少冗余、简化工具包并移除抽象层，如[此处](https://xcancel.com/LysandreJik/status/1933201171130593530)所述。
   - **针对 TF 和 Flax 的长期支持 (LTS) 将随 v4 版本持续到 2026 年年中**，这一变化标志着 v5 版本的开始，目标是移除 50% 的代码。
- **Meta AI 应用公开分享私人对话**：一款 **Meta AI** 应用不慎将用户的私人对话（包括敏感信息和音频）发布到了公共 Feed 流中，链接见[此处](https://xcancel.com/SHL0MS/status/1933019178023231880)。
   - 由于 UI 设计令人困惑，用户在无意中分享了内容，导致个人隐私泄露，并引发了对 Meta 的伦理担忧。
- **Anthropic 的多智能体系统优于单智能体 Claude Opus**：**Anthropic** 发现，*一个以 Claude Opus 4 为主智能体、Claude Sonnet 4 为子智能体的多智能体系统，在内部研究评估中的表现比单智能体 Claude Opus 4 高出 90.2%*，根据[此帖子](https://www.anthropic.com/engineering/built-multi-agent-research-system)显示。
   - 他们发现，*多智能体系统在涉及高度并行化、信息量超过单个上下文窗口以及与众多复杂工具交互的高价值任务中表现出色*，但 Token 消耗极快，约为**普通对话的 15 倍**。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 的库版本感知能力探索**：用户寻求增强 **Aider** 对库版本的感知方法，特别是在从 **pip/virtualenv** 迁移到 **Poetry** 时，针对其建议过时选项的问题，建议包括添加更新后的 man pages 的 URL，以及在约定中明确定义版本或使用 `/read docs/spec.txt` 命令。
   - 讨论强调了为 **Aider** 提供更好上下文的重要性，以确保它建议最新的库版本。
- **使用 Anthropic 模型的 Aider 成本**：一位用户对使用 **Anthropic** 配合 **Aider** 时可能产生的近 **$50** 的小时成本表示担忧，特别是在进行大规模更改时；同时指出 **Claude Code** 的月度计划可能很快被耗尽，暗示可能存在使用限制。
   - 对话强调了在使用 **Anthropic** 等商业模型配合 **Aider** 时进行成本管理的重要性，强调需要监控使用情况。
- **Aider 在小型模型上表现出色**：用户赞扬了 **Aider** 通过 **Ollama** 在小型模型（**8B** 和 **12B**）上的表现，认为其效果出奇地好，另一位用户指出 **Aider** 的 repomap 是其核心秘诀。
   - 该工具在有限资源下高效运行的能力，使其成为小型本地运行模型的有力竞争者。
- **UV 管理 Python 依赖**：成员们探索了迁移到 **UV** 进行 Python 依赖管理，认为它是直接使用 **pip** 和编辑 **pyproject.toml** 的更优替代方案，更倾向于使用 `uv add <dependency name(s)>` 等命令。
   - 一位最初对阅读手册犹豫不决的用户发现，**UV** 在 **YAML** 配置中定义 linting 指令时更加*严谨（tighter）*，标志着向流线型依赖处理的转变。
- **max_input_tokens 配置难题攻克**：一位用户解决了在 **Aider** 中为输入和输出设置独立 max tokens 相关的配置挑战，特别是关于“剩余 token”显示的问题。
   - 澄清后成功配置了 **max_input_tokens** 参数，解决了最初的困惑并提升了 **Aider** 的性能。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Vast.ai 提供廉价算力**：一位成员推荐 [Vast.ai](https://vast.ai) 作为去中心化算力提供商，称其*相对便宜*，Akash 也被提作为潜在的替代方案。
   - 一位成员指出 Vast.ai 是两者中更便宜的选择。
- **C. Opus 发布首篇 Arxiv 论文**：Teknium 在 X 上分享了[一条帖子](https://x.com/WesRothMoney/status/1933502113285616083)，宣布这是 **C. Opus** 在 Arxiv 上的首篇出版物。
   - 这一重要事件得到了多次确认。
- **NVIDIA 发布 Cosmos**：**NVIDIA** 推出了 [Cosmos](https://github.com/nvidia-cosmos/cosmos-predict2)，其 ArXiv 论文可在 [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) 查看。
   - 频道内没有对该发布或其功能的进一步讨论。
- **Infiniband 速度超越互联网带宽**：一位成员指出，虽然典型的互联网带宽约为 **1gbps**，但 [Nvidia 最新的 Infiniband](https://x.com/toolandtea/status/1933381389552136705) 迭代版本达到了 **130TB/s**，凸显了带宽差距的日益扩大。
   - 互联网带宽在近年来并未见显著增长。
- **DAWN Internet 推广去中心化互联网接入**：一位成员推广了 **DAWN Internet**，这是一种去中心化宽带协议，使用固定无线屋顶天线提供千兆互联网，并包含一个能够支持 **RL** 的 **GPU**。
   - 更多信息可以在[他们的 X 个人资料](https://x.com/dawninternet)中找到。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **思维导图杰作问世**：一名成员利用 **115+ 个来源**创建了一个思维导图，总结了关键方面并声称其*相当准确*。根据[链接图片](https://cdn.discordapp.com/attachments/1124403655819415592/1383205649839820830/NotebookLM_Mind_Map.png?ex=684df225&is=684ca0a5&hm=9f877d3700e50d5c48e2faa411ec5c0e28b33088d6d859a72f7f2278d3660d3d&)，这是一个*巨大的思维导图*。
   - 在回答另一位成员的询问时，该地图被指出有 **4 个子层级**，用户对垂直密度表示满意，但指出水平方向仍有改进空间。
- **付费 AI Pro 用户无法访问 Notebook LM Plus**：一位使用 **付费 AI Pro** 的成员报告无法访问 **Notebook LM Plus**，并询问可能的原因，但频道内未提供解决方案。
   - 讨论中尚未解决该访问问题的根本原因。
- **NotebookLM 缺少 Excel 支持**：用户请求在 NotebookLM 中支持 **Excel** 和 **Google Sheets**，但目前尚无该功能的支持或路线图。
   - 建议用户使用功能请求频道来表达他们的兴趣。
- **移动端 App 笔记功能受限**：虽然 NotebookLM 的桌面版提供 **Notes**（笔记）功能，但移动端 App 仅显示 **sources**（来源）、**chat**（聊天）和 **studio**（工作室）部分。
   - 虽然移动端没有导出选项，但一种变通方法是通过浏览器而非 App 在手机上访问笔记，用户可以在浏览器中进行复制粘贴。
- **笔记本分享按钮置灰**：用户在分享笔记本时遇到问题，**“Share publicly”（公开分享）按钮**显示为灰色且无法点击。
   - 该问题的起因目前尚不明确。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Beam Linearizer Bug 困扰用户**：用户报告在运行 **BEAM** 时遇到 **linearizer failures**（线性化器故障），但原因和解决方案仍然不明。
   - 此问题需要进一步调查以确定根本原因和潜在的修复方案。
- **Tinygrad 的浮点数误差引发挫败感**：一位用户发现 **NumPy** 和 **Tinygrad** 在执行 **float matmul**（浮点矩阵乘法）时存在精度差异，具体表现为使用[这段代码](https://discord.com/channels/924641739439477844/1147353713739749426)时输出矩阵左下角的值不一致。
   - 讨论涉及了编译器变体、优化策略以及对 **IEEE 754 标准**遵循情况的影响，强调了轻微的数值变化是典型的，取决于运算顺序以及 **NumPy** 默认使用 **float64** 的情况。
- **SVD 符号偏差引发审查**：一位致力于 **linalg.svd** PR 的贡献者旨在达到 **NumPy** 级别的精度，但在数值中遇到了符号差异。
   - 建议使用 `DEBUG=4` 检查内核代码，并使用 `NOOPT=1` 禁用循环展开以获得更接近的结果，因为循环展开可能会引入数值差异。
- **QR 算法特性受到质疑**：一位用户指出，由于 **Householder Reflections**（Householder 变换）与 **Gram-Schmidt process**（格拉姆-施密特过程）之间的差异，**QR 算法**存在变异性。
   - 该用户强调，与 **NumPy** 用于特征值计算的 **LAPACK** 包相比，这种变异性甚至更大。
- **NumPy 的数值规范需要权衡**：一位用户建议显式创建 `dtype=np.float32` 的 **NumPy** 数组以减少结果差异，并批评了 **NumPy** 默认设置 `np.float64` 的做法。
   - 另一位用户反驳称，在机器学习之外的数值应用中，**float64** 是标准配置，更改默认设置可能会破坏无关的功能。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **映射 Variadic Types 仍是一项挑战**：在 **Mojo** 中映射变长类型（variadic types）仍然面临挑战，正如[这篇论坛帖子](https://forum.modular.com/t/map-variadic-types/1638?u=emil.martens)所强调的，这主要是因为需要更具动态性的类型系统。
   - 使用 **StaticString** 来定义相应的 `__mlir` 类型建议遇到了困难，原因是文档有限且支持任意数量类型的复杂性较高。
- **探索 MLIR 类型变通方案**：对使用 `__mlir_type` 的变通方案探索遇到了问题，包括 **MLIR 未公开文档** 以及无法将给定类型参数的 **MLIR 类型** 合成为原始字符串。
   - 一位成员建议在编译时提取并修改 **MLIR 类型**，通过使用 **UnsafePointer** 和 `init_pointee_move` 来绕过类型定义的限制。
- **从 Magic 到 Pixi 的迁移实现无痛转换**：一位用户通过删除 `~/.modular` 目录并重写 `mojoproject.toml` 文件，成功从 `magic` 迁移到 `pixi`，并将该过程描述为“无痛”。
   - 该用户提供了一个用于更新和清理缓存的 `pix.sh` 脚本，该脚本会创建一个新的 `pixi.lock` 和 `.pixi` 文件夹，并建议在测试验证后删除旧文件夹。
- **GPU 谜题中的主机端同步得到澄清**：针对 GPU 谜题中**主机端同步（host-side synchronization）**的相关疑问得到了澄清，特别是针对[这一章节](https://builds.modular.com/puzzles/puzzle_12/complete.html#host-side-synchronization-the-critical-step)。
   - 由于 `DeviceContext` 使用了 CUDA stream，因此不需要**显式同步**，谜题描述将进行更新以反映这一点。
- **Mojo 通过 C ABI 导出功能**：Mojo 支持使用 `@export(ABI="C")` 导出 **C ABI 兼容函数**，从而方便创建对象文件或共享库。
   - 这使得与 **C/C++** 代码库的集成成为可能，扩展了 Mojo 的互操作性。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **GitHub 实现实时上下文访问**：GitHub PM 发布了一个远程 GitHub MCP server，允许任何 MCP host 访问实时 GitHub 上下文而无需本地设置，详见 [Reddit](https://www.reddit.com/r/mcp/s/Cj2zjute95)。
   - 该 server 采用动态工具选择，根据用户输入或上下文向 LLM 展示相关的工具子集，即使在有 30 多个工具可用时，也能通过**一个 MCP server** 保持简单的身份验证。
- **使用 Taskerio 跟踪 Agent 进度**：Taskerio 推出了一款隐身模式产品：一个专为编程 Agent 设计的收件箱，用于报告进度，具有 Webhooks、推送通知和用于实时仪表板的 API，详见 [Reddit](https://www.reddit.com/r/mcp/comments/1lac12i/an_mcp_to_track_the_progress_of_your_ai_agents/)。
   - 这允许对 AI Agent 的活动进行实时监控和跟踪。
- **使用 SchemaPin 加固 MCP 以防 Rug Pulls**：一位成员介绍了 **SchemaPin**，这是一个旨在防御 **MCP Rug Pulls** 及相关漏洞利用的工具，[GitHub 仓库在此](https://github.com/ThirdKeyAI/SchemaPin)。
   - [SchemaPin.org](https://schemapin.org) 详细介绍了简单的实现方法，保护 **MCP** 免受潜在漏洞的影响。
- **Postman 简化 MCP Server 构建**：一位成员演示了如何使用 Postman 的 MCP 构建器及其公共 API 网络上的 API 来构建 **MCP server**，并引用了 [fastfs-mcp GitHub 仓库](https://github.com/aj-geddes/fastfs-mcp)作为示例。
   - 相应的 [YouTube 视频](https://youtu.be/YzT9QU-Kfh4?si=tqqgBXXu9ct2aMUH)进一步阐明了该过程。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud 恢复稳定**：在上游基础设施出现小故障后，**LlamaCloud** 已恢复运行；请查看 [状态页面](https://t.co/IdecAksHiG) 获取实时更新。
   - 此次事件凸显了云端依赖的脆弱性。
- **MistralAI 的 Magistral 现已与 LlamaIndex 良好适配**：根据 [这条推文](https://t.co/ZsUEWMrnT4)，**LlamaIndex** 已支持 **MistralAI** 的 **Magistral** 推理模型，并将其纳入 Agent 工作流。
   - 这一集成可能为更复杂的推理任务打开大门。
- **LlamaParse 通过 Presets 变得更加易用**：**LlamaParse** 推出了 **Presets**（预设），提供 **Fast**（快速）、**Balanced**（平衡）和 **Premium**（高级）模式，以便在文档解析过程中调整准确度与速度。
   - 这些预设允许用户根据需求优化文档解析。
- **Mem0 集成简化了 LlamaIndex 中的内存管理**：在 **LlamaIndex** 中使用 **Mem0** 时，通过将 `memory=memory` 传递给 `agent.run()`，内存更新会自动发生，无需手动更新。
   - 与 **LlamaIndex** 的集成支持 Mem0 的 graphRAG 功能，简化了内存处理。
- **Luma 日历可能会取代 Discord 用于 Office Hours**：由于对 Discord 日历易用性的抱怨，组织者正在考虑切换到 **Luma** 日历进行 Office Hours，并正在就未来 Office Hours 的形式 [征求意见、需求和建议](https://discord.com/channels/1031248924924043295/1031248926475255868)。
   - 此举旨在提升 Office Hours 的体验。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **DeepMind 拥抱 Named Tensors**：一名成员正在为 Google DeepMind 开发 **Xarray-JAX 库**（作为 GSoC 2025 的一部分），并声称这是深度学习框架中第一个 Named Tensor 实现。
   - 该库旨在增强 JAX 内的 Tensor 操作，使其在深度学习任务中更加直观和高效。
- **AI 金融工具规避 LLM Wrapper 陷阱**：一名成员正在开发一个**金融领域的 AI SaaS 工具**作为大学项目，并询问如何避免仅仅做一个 LLM Wrapper，从而为终端用户提供真正的价值。
   - 他们请求关于 MVP 的建议，以避免大多数 LLM Wrapper 中常见的陷阱。
- **Cohere 文档出现语法错误**：一名成员报告了 [Cohere 文档](https://docs.cohere.com/docs/amazon-sagemaker-setup-guide) 中的一个潜在**拼写错误**。
   - 修正建议指出，在 Python 代码示例中，`co = cohere.SagemakerClient()` 中的 `SagemakerClient()` 应该使用小写字母 "m"。
- **Reranking Profile 请求未获回应**：一名成员询问了 Reranking Profile 的规范，特别是 **docs 数量、每个 doc 的 token 数以及 query token 数**。
   - 遗憾的是，该请求未收到任何回复，询问在没有进一步讨论的情况下结束。
- **GCP 故障影响 Cohere 运行**：Cohere 报告称，Google Cloud Platform (**GCP**) 的故障在 **2025 年 6 月 12 日**下午 **12:02** 影响了其部分服务 [状态页面](https://ift.tt/on1ARP0)。
   - 状态页面显示 **Infrastructure**（基础设施）组件性能下降，促使 Cohere 团队进行密切监控和响应工作 [Cohere 状态页面](https://ift.tt/Ens6bma)。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Fast Weights 旨在增强用户控制**：成员们提倡 **fast weights 持续学习**和**外部数据存储**，以提高用户控制力并减少 AI 模型中不理想的人类特征。
   - 他们表示渴望看到*诡计、挫败感和虚假记忆*等特征从主流 AI 中移除。
- **O1-Pro 模型提供良好价值**：一位成员发现 **O1-Pro/O3/O4-mini-high** 模型在学习文档齐全的数学和计算机科学方面非常有价值，同时也喜欢它们的**图像生成能力**。
   - 他们还提到使用模型的 API 构建了一个几乎完美的**音频转录流水线 (audio transcription pipeline)**，尽管图像生成功能受到了审查。
- **Gemini 与 Claude 的体验对比**：一位成员询问 **Gemini** 与 **Claude** 的对比情况。
   - 另一位成员表示 **Claude** 对他们来说不太可靠，但指出所有模型都可能出错，并且在高度可验证的领域最为有用。
- **Wavefunction 讨论周五休息**：由于观众参与度有限，周五通常**没有 Wavefunction 讨论**。
   - 尽管缺乏预定的讨论，但欢迎社区成员自行发起。
- **黄仁勋 (Huang) 与 Amodei 在 AI 就业问题上存在分歧**：一篇 [Fortune 文章](https://fortune.com/2025/06/11/nvidia-jensen-huang-disagrees-anthropic-ceo-dario-amodei-ai-jobs/) 报道称，**黄仁勋 (Jensen Huang, Nvidia)** 与 **Dario Amodei (Anthropic)** 在 **AI 就业**的未来上持不同意见。
   - **Dario** 已通过 [X](https://www.x.com/dario) 回应了 **Jensen**，并更新了关于 AI 就业的看法。由于**对失业的担忧持续存在**，两家公司的股价均大幅下跌。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Mistral 3.1 Small 架构细节仍不明朗**：一位用户询问了 **Mistral 3.1 Small** 的架构创新，估计一旦获知细节，需要 **2 周**时间来实现微调。
   - 另一位用户认为支持 **Mistral 3.0** 就意味着支持 **Magistral**，尽管多模态 (multi-modality) 支持可能具有挑战性。
- **Tokenizer 难题引发猜测**：提到了 Tokenizer 的难度，一位成员认为这是一个*复杂的过程*。
   - 讨论澄清了他们实际上指的是 **Magistral** 的 Tokenizer。
- **敦促 Magistral 集成 Torchtune**：成员们对在 **Magistral** 的 Hugging Face (HF) 页面上添加 **Torchtune** 链接表示感兴趣。
   - 这表明社区对 **Torchtune** 与 **Magistral** 集成以提高易用性的需求。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **本地实现 Infinite Chat**：一位成员强调了 **Infinite Chat** 的本地实现，旨在防止用户耗尽上下文窗口 (context window)。
   - 有兴趣了解其功能和特性的用户可以在[此处](https://docs.supermemory.ai/infinite-chat)找到文档。
- **请求忽略 (Ignore) 功能**：一位用户询问是否可以为嵌入系统 (embedding system) 添加 **'ignore' 功能**，类似于 Git 中的 `.ignore` 文件。
   - 该功能将允许用户在处理或嵌入时排除特定的文件、文件类型或目录。

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Wave 10 带来 UI/UX 升级**：Windsurf 正在完成 **Wave 10**，带来了一系列全新的 **UI/UX 升级**以及新的团队和企业级方案，包括用于 `@-mentions` 和文件引用的 [新图标](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise)。
   - Cascade 面板中的代码块现在匹配你的 IDE 主题，Cascade 面板中的原生终端现在支持用户输入，并新增了对话历史 UI。
- **Windsurf 推出欧盟集群（EU Cluster）以提升性能**：Windsurf 自豪地宣布推出 **EU Cluster**，为欧洲企业带来更快的性能并满足日益增长的需求！
   - 观看 [Youtube 视频](https://youtu.be/UHinqQiiCI8?si=udyZDkWGg9nq7zcI) 并 [加入 r/Windsurf 讨论](https://www.reddit.com/r/windsurf/)。
- **Claude Sonnet 4 助力 Windsurf**：**Claude Sonnet 4** 和 **Claude Sonnet 4** (Thinking) 现已通过 [API Pricing](https://docs.windsurf.com/windsurf/models#api-pricing) 向所有付费计划开放！
   - 更多信息请见 [X 平台](https://x.com/_mohansolo/status/1933605162775687482)。

---

**DSPy Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

你收到这封邮件是因为你通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
你可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord: 频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1382797226081910844)** (1103 messages🔥🔥🔥): 

> `Gemini Deep Think, Perplexity Pro Role, Image Creation with Perplexity Pro, GPT-5 architecture, Qwen3 replacing Qwq` 

- **Gemini Deep Think 即将到来**：一位成员提到 **Gemini Deep Think** 即将上线，并引用了一张图片 ([aSHuTrz.jpeg](https://cdn.discordapp.com/attachments/1047649527299055688/1382930529464352848/aSHuTrz.jpeg?ex=684d9aab&is=684c492b&hm=b30084937010eeeb0b4dba66cb69c6fd085c52fb786ce6e316e2324b2f50d0aa&))。
- **Perplexity Pro 身份组失效**：成员们讨论了在 Discord 中获取 **Perplexity Pro 身份组**的困难，入站按钮无法正常工作，一些成员建议通过艾特工作人员作为临时解决方案。
   - 一位成员表示：*"那个按钮似乎没给我身份组，只是让我用手机进入了服务器"*。
- **Perplexity Pro 可以生成图像！**：成员们分享了 **Perplexity Pro** 可以通过在搜索栏输入图像提示词来创建图像，例如 *“用油画风格画一个春天里的淡雅田园风村庄”*。
   - 一位成员给出了进一步指导：*"点击图片下方的 **Regenerate** 或在同一个线程中发送新的提示词。尝试使用艺术风格或效果进行优化，如：电影感、低多边形 3D、影棚灯光、动漫风格"*。
- **GPT-5 工作更聪明，而非更辛苦**：一位成员分享了关于 **GPT-5** 架构的细节，指出它作为一个单一模型运行，在内部利用专业工具，避免了外部路由和幻觉的陷阱。
   - 他们引用道 *"GPT-5 是利用其工具进行思考，而不是在工具之外思考"*，强调了改进的上下文、协调性和稳定性。
- **Qwen3 是新的已故 Qwq**：成员们注意到 **Qwen3** 取代了现已失效的 **Qwq**，在转向其他话题前确认了它的存在。

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

meijer5838: https://www.perplexity.ai/page/unused-phones-europe-s-hidden-YpcOJpSCSfu9IlnOng_85A

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1382822588576694427)** (7 messages): 

> `Sonar API documentation feedback, Perplexity Publisher Program` 


- ****Sonar API 文档寻求反馈****：团队正在寻求关于 **Sonar API documentation** 的反馈，并请求用户在 [此社区帖子](https://community.perplexity.ai/t/improvements-to-the-sonar-api-documentation/542?u=vikvang) 下分享他们的体验，特别是关于不清晰或导航困难的部分。
   - 目标是根据用户输入改进文档。
- ****Publisher Program 推广信息公开发布****：一位用户分享了与 **Perplexity Publisher Program** 相关的 [LinkedIn 帖子](https://www.linkedin.com/posts/codingclubnmims_codingclubmpstme-endoftenure-newbeginnings-ugcPost-7339125013536464896-ybUR?utm_source=social_share_send&utm_medium=android_app&rcm=ACoAAEUiNWsBLhCcJJA2pq2u07Btb29g_1q97iU&utm_campaign=whatsapp) 和 [公司页面](https://www.linkedin.com/company/codingclubnmims)。
   - 该用户建议这可能对特定频道有所帮助。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1382796606843256966)** (1098 messages🔥🔥🔥): 

> `o3 vs 2.5 Pro, Model preference, Ethics in models, Grok 3.5, New models` 


- **O3 Pro vs 2.5 Pro：模型偏好辩论持续升温**：成员们就模型偏好展开了激烈辩论，一些人认为 **o3** 在许多方面优于 **2.5 Pro**，而另一些人则声称 **2.5 Pro** 在数学等特定领域表现更出色。
   - 针对另一名成员对 **2.5 Pro** 在数学方面表现的偏好，一名成员讽刺地评论道：“我希望生活在这种程度的幻觉中”，引发了进一步的讨论。
- **MathArena 基准测试：已饱和还是仍然有效？**：成员们讨论了 [MathArena benchmarks](https://matharena.ai) 的有效性，一些人认为它们正趋于饱和且可能带有运气成分，而另一些人则坚持认为它们仍是有效的衡量指标。
   - 有人担心接近 **100%** 的分数可能表明已达到饱和，并质疑这些基准测试是否仍具有统计学意义。
- **Google 账号被封：新 Kingfall 发布引发警报**：一名成员报告其 Google 账号被封，这引发了其他人对新 **Kingfall** 发布的猜测。
   - 一个代号为 *toothless* 的新 Gemini 模型短暂出现，引发了关于它是新 Checkpoint 的猜测。甚至有报告称在各种尝试中获得了 99% 的收益。
- **文本转视频竞技场：Seedance 和 Kangaroo 表现亮眼**：成员们在 [文本转视频竞技场](https://artificialanalysis.ai/text-to-video/arena) 中分享并讨论了盲测结果，强调了 **Seedance 1.0** 和匿名的 **Kangaroo** 等模型的出色表现。
   - 讨论中进行了对比，一些人认为这些模型在生成通用提示词的输出方面超过了 **Veo3**、**Kling** 和 **Pika**。
- **O4/GPT-5 传闻：新的挑战者？**：关于 **O4/GPT-5** 潜在发布的猜测不断涌现，一名成员自信地表示它不是一个更大的模型，引发了关于命名习惯和发布形式的辩论。
   - 另一名成员表示他们有证据证明它是原生模式，但拒绝提供证据。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1382832731314061373)** (2 条消息): 

> `云服务商故障，竞赛进行中，Test Garden 申请，员工 AMA` 


- **云服务商故障导致潜在的 Promptcat 数据问题**：由于其云服务商在 **2025年6月12日** 发生故障，**聊天历史数据可能已丢失**。
   - 团队对造成的不便表示歉意，并正在*制定解决方案以确保此类情况不再发生*。
- **竞赛作品激发创意火花**：目前有一项竞赛正在进行，鼓励参与者将作品发布到 <#1378034388272681079> 频道，有机会赢取奖励。
   - 更多详情请参阅[此处](https://discord.com/channels/1340554757349179412/1343296395620126911/1378037223794147458)。
- **Test Garden 吸引技术爱好者尝试**：有兴趣提供反馈并了解幕后开发进展的爱好者可以申请加入 **Test Garden**。
   - 申请表请见[此处](https://docs.google.com/forms/d/e/1FAIpQLSeuV7miT_8j_Sn3DRjSStxu7a54crQNGlj54XMJ-GO9Xw68sQ/viewform?usp=dialog)。
- **员工 AMA 在热烈关注中圆满结束**：团队感谢所有参加上周 **Staff AMA** 的成员。
   - 可以通过[此表单](https://docs.google.com/forms/d/e/1FAIpQLSegIDRbbpx2amAR-6MA834fz_QycY15IQ0csyOOKUJUTncGMw/viewform?usp=dialog)分享反馈，[视频录像](https://cdn.discordapp.com/attachments/1343296395620126911/1383145168470937650/Staff_AMA_66.mp4?ex=684db9d1&is=684c6851&hm=84b909bb9ab6d48888d81480e11562318715b436b6745cc7bf4f430717a5a9d3&)现已发布。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1382796795574489230)** (433 条消息 🔥🔥🔥): 

> `Cloudflare 故障影响 Cursor，Cursor 代码生成，Claude Code 用于复杂重构，MCP 服务器设置，前端测试` 


- **Cloudflare 和 GCP 故障导致 Cursor 崩溃**：用户报告 [Cursor 宕机](https://status.cursor.com/)，原因是 **Cloudflare** 和 **GCP** 故障，影响了许多人的登录和功能，而其他人指出 [Tab 功能仍然可用](https://www.cloudflarestatus.com/)；这些问题随后被[报告已解决](https://mashable.com/article/google-down-cloudflare-twitch-character-ai-internet-outage)。
- **对 Cursor 编码的批评仍在继续**：一位用户询问如何使用 Cursor 制作 three.js 游戏，而另一位用户建议大多数编码使用 O3，规划和调试使用 O3-pro，强调其效果优于其他模型。
   - 有讨论指出，当切换到“自动”模型选择时，**Cursor 的代码生成**表现欠佳，一位用户因代码混乱损失了 **50 个推理额度 (inference credits)**，建议不要使用 Cursor 的自动切换功能。
- **多人游戏开发正当时**：许多成员讨论了使用 **peerjs** 和 **socket.io** 开发多人游戏，一位成员展示了他们在 Steam 上的多人 Unreal Engine 5 游戏 [Supermarket Simulator](https://nandstudios.itch.io/supermarketsimulator)。
- **CUA 自动化即将来临**：成员提到 **CUA (Computer Using Agent)** 的改进可以增强自动化，并提到了用于任务自动化的项目 [browser-use](https://github.com/browser-use/browser-use)。
- **Claude Code 荣登上下文之王**：用户发现 **Claude Code** 在复杂重构的上下文理解和代码质量方面表现出色，为一个前端组件库添加了 **3500 个新测试**。


  

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1382805848149065768)** (30 条消息🔥): 

> `Background Agents LSP and Linter, Background Agents Privacy Mode, Background Agent Commit Issues, Background Agents leaking context, Background Agents Docker Compose` 


- **Background Agents 利用 LSP 和 Linters**：Background agents 应该使用 **LSP 错误**并有权访问所有扩展，确保依赖项已安装在 agent 环境中。
   - Slack 集成目前可能还做不到这一点，但如果通过桌面客户端启动它们，则应该可以实现。
- **Background Agents 隐私模式已修复**：用户报告在尝试启动 background agent 时收到错误消息：*Background agent is not supported in privacy mode*。
   - 该问题已通过在[此处](https://www.cursor.com/slack-connected)启用**账户级隐私模式**解决，此问题将在下一个版本中修复。
- **Background Agent 面临 Commit 挑战**：在 background agent 修改（amend）了一个 commit 后，它在推送该修改后的 commit 时遇到了困难。
   - 一位成员建议通过终端解决，因为 agent 正在被回滚。
- **Background Agents 可能泄露上下文**：一位用户报告 background agent 可能会泄露上下文，链接了一个与当前任务无关的 **Sentry 错误**。
   - 用户在多个聊天中遇到了这种情况，并分享了一张[图片](https://cdn.discordapp.com/attachments/1367213641027551352/1382941599033462835/image_5.png?ex=684da4fb&is=684c537b&hm=fbe62477092e700e3c79f24220708ca44395f9b259e40ada0ab185a4ac7cef56)作为证据。
- **Background Agents 的 Docker Compose 设置**：一位用户寻求关于使用 **Docker Compose** 设置 Cursor background agent 的指导，旨在默认在特定容器内运行命令。
   - 他们提供了 [docker-compose.cursor.yml](https://cdn.discordapp.com/attachments/1367213641027551352/1383051743100928000/docker-compose.cursor.yml?ex=684d62cf&is=684c114f&hm=33255fde21b1dc7efa06d1ece8217a45ddacebf1f251ffc5069028a6798c1538) 和 [environment.json](https://cdn.discordapp.com/attachments/1367213641027551352/1383051743427956745/environment.json?ex=684d62cf&is=684c114f&hm=b10c10f7ed5f05e5d22752a7c55b429dc9c70616169c69900417db0184032229) 配置，希望在关联了 Postgres 和 Redis 容器的情况下运行 pytest 等工具。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1382800132571004928)** (4 条消息): 

> `Google Cloud Outage, Cloudflare Status, Internet Recovery` 


- **Google Cloud 遭遇重大故障**：据其[状态页面](https://status.cloud.google.com/)报告，**Google Cloud** 经历了重大故障。
   - 用户报告称，即使在东部时间 **下午 4:25** 左右出现初步恢复迹象后，仍存在间歇性问题。
- **Cloudflare 和 Google 状态页面提供更新**：可以通过 **Cloudflare** [状态页面](https://www.cloudflarestatus.com/)和 **Google Cloud** [状态页面](https://status.cloud.google.com/)跟踪故障和恢复的更新。
- **OpenRouterAI 发布关于恢复的推文**：OpenRouterAI 发布推文称看到了故障恢复，并希望这不会是暂时的（[推文链接](https://x.com/OpenRouterAI/status/1933263905385500853)）。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1383078905266831511)** (5 条消息): 

> `Button cutoff on narrow browser` 


- **窄浏览器按钮 Bug 已修复！**：一位成员报告了一个 Bug，即在窄浏览器窗口中按钮被截断，如[此截图](https://cdn.discordapp.com/attachments/1092850552192368710/1383078905002594396/Screenshot_2025-06-13_at_06.41.30.png?ex=684d7c1b&is=684c2a9b&hm=526613d7d6f7266a7d81bddc27965de3d691f309935ce3264d7ee968a9884e76&)所示。
   - 另一位成员迅速处理并修复了该问题，随后提供了修复后的[截图](https://cdn.discordapp.com/attachments/1092850552192368710/1383106945535180823/Screenshot_2025-06-13_at_10.33.00_AM.png?ex=684d9638&is=684c44b8&hm=88cfcaa0194f8fc528d0654a4257ec09912af72ffdd69d6f2b460c19cc014aee&)。
- **报告了另一个 Bug**：为了符合提示词中至少两个主题摘要的要求，这里提供了另一个主题以满足要求。
   - 在提供的文本中未发现实际的第二个 Bug，但包含此内容可确保 JSON 符合 schema 验证。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1382796611364720773)** (377 messages🔥🔥): 

> `Cloudflare 宕机影响 OpenRouter，OpenRouter 状态波动，不同供应商的模型性能差异，高性价比 LLM 的 Agent 工具使用，OpenRouter 的多模态支持` 


- **Cloudflare 再次让互联网瘫痪**：一场广泛的 [Cloudflare 宕机](https://www.cloudflarestatus.com/) 导致了严重的干扰，使包括 **OpenRouter**、Google 在内的众多 AI 服务下线。
   - 用户报告了广泛的问题，并引发了对原因的幽默猜测，从“实习生把咖啡泼到了服务器上”到“天网（Skynet）接管”。
- **OpenRouter 的服务波动让用户备受煎熬**：用户经历了断断续续的 **OpenRouter** 服务，状态页面在“重大故障（MAJOR）”和“轻微故障（MINOR）”之间反复横跳，导致用户感到沮丧，并开玩笑说掐点发送 API 请求就像玩嘉年华游戏一样。
   - 一些用户在使用特定模型或配置时获得了成功，而另一些用户则持续面临“超时”和“身份验证错误”。
- **供应商差异影响模型质量**：用户讨论了通过 **OpenRouter** 提供相同模型的不同供应商之间存在的显著质量差异，指出 **Parasail** 和 **Lambda** 通常提供更一致的性能。
   - 质量比成本更重要，正如一位用户所说：[不同供应商的质量差异很大，所以要明智选择](https://discord.com/channels/1091220969173028894/1092729520181739581/1383133709551282236)。
- **廉价 Agent LLM 脱颖而出成为顶级工具使用者**：用户讨论了用于 Agent 工具使用的最佳廉价 **LLM**，**Claude 2.5 Flash** 被推荐为一种高性价比的选择，但需要精细的 Prompt 引导。
   - 讨论中还涉及了 **O4 Mini High** 等模型的高昂成本以及可能发布的全新 **Google Flash** 模型，同时还讨论了使用 [每月 Claude Max 订阅](https://discord.com/channels/1091220969173028894/1195014798837043240/1383046909199872050) 进行 API 使用的效率。
- **期待 OpenRouter 的多模态功能**：成员们请求 **OpenRouter** 平台未来支持 **音频** 和 **视频生成** 等多模态功能。
   - OpenRouter 尚未给出明确回应。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1382800601556975687)** (135 messages🔥🔥): 

> `LM Studio 模型更新，在 LM Studio 中设置静态生成种子，Gemini Pro 图像识别，绕过验证码（Captchas），在服务器上运行 LM Studio` 


- **LM Studio 不会自动更新模型**：LM Studio 不会自动下载模型，且大多数模型更新都是在新的仓库中发布的迭代版本，因此很难确定模型的演进关系。
   - 一位成员询问是否有办法像 Ollama 那样在 LM Studio 中自动更新模型，但目前情况并非如此。
- **简洁的 LLM 回复**：LLM 被训练得尽可能简洁，以免让用户感到厌烦并节省计算成本，因此可以通过先获取结构，然后针对该结构的每个要点请求内容来拆分任务。
   - 这是为了回应一位用户请求非常长且详尽的摘要（类似于论文），并询问是否有办法降低模型过早结束响应的倾向。
- **Gemini Pro 在图像识别方面表现不佳**：一位用户询问为什么 **Gemini Pro 2.5** 在简单的图像识别中会出错，即使尝试了各种 Prompt 和图像（[示例图像](https://cdn.discordapp.com/attachments/1110598183144399061/1382962931741888563/image.png?ex=684db8d9&is=684c6759&hm=14fbd9fe32c10ead609c4627acfd3c543cdf7ca6f347af0e6c9a61470af4c663&)）。
   - 另一位成员指出，具备视觉功能的模型通常表现并不理想，而且很难准确确定用户的期望——尤其是当用户说他们已经“尝试了一切”时。
- **LLM 绕过验证码是一场“红皇后竞赛”**：成员们讨论了使用 **LLM** 绕过验证码（Captchas）的困难，强调验证码旨在让计算机难以识别，并且在不断升级以挫败 **LLM**。
   - 一旦开发出破解验证码的技术，新的验证码就会出现，使之前的进展过时，就像 [红皇后假说（Red Queen hypothesis）](https://en.wikipedia.org/wiki/Red_Queen_hypothesis) 一样。
- **OpenWebUI 支持远程访问 LM Studio**：要在服务器上运行 LM Studio 并从另一台 PC 访问它，可以在 **LM Studio** 上托管服务器，加载模型，在本地网络中提供服务，启用 **CORS**，并使用 [OpenWebUI](https://github.com/OpenGenAI/OpenWebUI) 在主机 PC 上打开特定端口（例如 1234, 8080, 3000）。
   - 无需在用于访问的 PC 上安装 **OpenWebUI**。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1382807314242998324)** (151 条消息🔥🔥): 

> `Unified Memory, Strix Halo, Tesla P40, Context Windows` 


- **HX395+ 车主对 Unified Memory 视频表示反对**：一位成员分享了一段[比较 Unified Memory 的视频](https://youtu.be/Cn_nKxl8KE4?si=0-iQclmGi2UcWVNxa)，但另一位真正的 HX395+ 车主表示反对，称其为*一段糟糕的视频*。
   - 反对的原因是该视频混淆了 **soldered RAM**（焊接内存）和 **slotted RAM**（插槽内存），偏离了主题，且不了解拥有 **4x widefrongong**（宽总线/前端）的 **Strix Halo**。
- **Tesla P40 显存扩展？**：一位成员询问是否可以使用 **Tesla P40** 作为 **RTX 3090/4090** 等普通 GPU 的补充，以扩展 LM Studio 的 VRAM，价格约为 **300 美元** 购买二手 **24GB** 显卡，并链接到了 [TechPowerUp 规格页](https://www.techpowerup.com/gpu-specs/tesla-p40.c2878)。
   - 普遍共识是，在 **300 美元** 的价位上，它已经不再值得购买，因为当价格低于 **150 美元** 时才值得入手，而二手的 **3090** 是更好的“实惠”选择。
- **关于是否需要 150B 参数模型的辩论爆发**：一位成员表示，*合理的真人交互* 至少需要 **150b** 参数，如果想要感觉聪明且自然，则需要 **300b**。
   - 另一位成员反驳称，LLM 不仅仅取决于参数数量，还包括 **prompt engineering**、优秀的 **RAG** 和 **finetuning**，而且并非所有场景都需要 *合理的真人交互*。
- **本地 LLM 在处理长 Context Windows 时表现挣扎**：一位成员分享说，他们的本地 LLM 在编写详细故事时会出现上下文保留问题，特别是当故事设定在中世纪城堡时，LLM 却开始谈论 *看电视*。
   - 另一位成员回复称，*没有哪个本地 LLM 能在超过 32768 的 Context Windows 下表现良好*，因为届时需要使用技巧才能扩展它，建议用户尝试专用模型，因为这些模型为了让 *长 Context Windows 生效而有一些特殊设计 (quirks)*。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1382796683670192128)** (83 条消息🔥🔥): 

> `AI Safety Institute, GPT-3 and GPT-4o Behavior, Symbolica.ai Startup, MMLU 5-Shot, AI Consciousness` 


- **新的 AI Safety Institute 出现**：成员们注意到一个新的 AI Safety 机构，但对其合法性表示怀疑，因为他们以前从未听说过，且其网站缺乏近期的出版物。
   - 一位成员指出其中一位顾问在 Discord 上，并建议安排一次通话。
- **德语文本引发 LLM 异常行为**：一位成员描述了一段简短的德语文本如何导致 **GPT-3** 和 **GPT-4o** 表现出截然不同的反应，从中性响应到深层情感解读不等。
   - 该成员想知道分享这一 *观察结果* 是否有意义，表明其有兴趣探索典型用例之外的 **LLM** 行为。
- **初创公司 Symbolica.ai 目标宏大**：一位成员重点介绍了 [Symbolica.ai](https://www.symbolica.ai/)，这是一家总部位于伦敦、目标雄心勃勃的新初创公司。
   - 另一位成员建议他们应该发布一个像 Google 那样的轻量级定理证明器模型，并指出一些评论提到 *工作的边界不清晰且目标不断变化*。
- **MMLU 5-shot 评估方式**：一位成员询问 **MMLU 5-shot** 是如何运作的，具体是 5 次中取最佳还是 5 次的平均值。
   - 另一位成员澄清说，*5-shot 指的是看到的示例数量，而不是允许尝试的次数*。
- **幻觉是由记忆引起的吗？**：一位成员想知道 **ChatGPT** 中的 *memory* 功能是否会导致幻觉（delusions）。
   - 另一位成员分享了这个 [arxiv 链接](https://arxiv.org/abs/2504.07992) 并表示，*一旦移除 'memory'，输出行为的退化立即停止*。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1382813851111788604)** (186 messages🔥🔥): 

> `Non-commercial license controversy, CommonPile 2 Creation, GRPO Objective & Model Performance, Symbolic Recursion` 


- **非商业许可引发争议**：成员们对一个新数据集的 [非商业许可 (non-commercial license)](https://discord.com/channels/729741769192767510/747850033994662000/1382718117561761874) 表示担忧，质疑其框架和潜在限制，尽管其目标是促进 *更健康的模型开发基础*。
   - 一些人认为这可能是 *虚假版权声明 (copyfraud)* 的一个案例，特别是如果数据转换主要涉及扫描和文本提取，并引用了 [维基百科关于 copyfraud 的文章](https://en.wikipedia.org/wiki/Copyfraud) 和 [Public Domain Sherpa](https://publicdomainsherpa.com/false-copyright-claims.html)。
- **CommonPile 2 可能依赖合成数据**：讨论围绕创建更强大的 **CommonPile 2** 展开，一位成员建议需要合成数据 (Synthetic Data) 来实现数 TB 的鲁棒训练材料。
   - 然而，有人提醒说，仅仅生成更多样本并不会神奇地产生更多信息，这违反了 *信息论 (information theory)* 原则，而且除了特定场景外，保持数据 *接近源头* 通常更可取。
- **GRPO 目标函数提升模型性能**：成员们讨论了 **DeepSeek V3**（一个 **671B** 模型）如何通过高容量和高质量数据实现高性能，然后使用 [GRPO 目标函数](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f) 使其在可验证的任务上获得更高的性能。
   - 该成员指出，*字面意义上的随机奖励也能提高性能*，这是由于一种 *集中效应，使模型专注于其现有的推理模式分布*。
- **“符号递归” (Symbolic Recursion) 术语遭到质疑**：成员们质疑 *符号递归* 一词的含义和有效性，该词经常出现在论文和演讲中以显得深奥，可能源于学术虚荣和术语黑话。
   - 据推测，模型之所以产生这种想法，归根结底只是 *模型在写作中重复使用相同符号* 的一种花哨说法。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1383162829586169856)** (1 messages): 

> `LLM Fairness, Bias Evals, Prompt Tuning, Chain of Thought, Concept Editing` 


- **偏见评估触发种族和性别偏见**：一篇新论文显示，在现有的 **偏见评估 (bias evals)** 中加入现实细节会触发 **LLM** 中的 **种族和性别偏见**，导致包括 **GPT4o** 和 **Claude 4 Sonnet** 在内的模型在面试率上出现高达 **12% 的差异**。
   - 现实细节包括公司名称、来自招聘页面的文化描述，或诸如 *“只接受前 10%”* 之类的约束。
- **可解释性修复公平性**：**提示词微调 (Prompt tuning)** 无法修复偏见，但基于可解释性的干预可以，使用简单的仿射概念编辑 / 种族和性别方向的消融 (ablation) 可以将偏见（以面试率差异衡量）降低到通常 **< 1%**。
   - 这篇关于 [稳健提高 LLM 公平性 (Robustly Improving LLM Fairness)](https://x.com/a_karvonen/status/1933582375419850806) 的论文提供了一个现实中 **不忠实思维链 (unfaithful Chain of Thought)** 的例子。
- **思维链是不忠实的**：检查 **思维链 (Chain of Thought)** 完全看不出 **种族/性别偏见** 的迹象，尽管结果本身表现出明显的偏见。
   - 论文发现这在所有测试的模型中都是事实，证明了在检测和缓解 LLM 偏见方面存在重大挑战。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1382797010977030195)** (12 messages🔥): 

> `Inspect 标准与评估框架对比, LM Evaluation Harness 进度条, 推理模型` 


- **Inspect 标准引发辩论**：一名成员询问了关于使用 [Inspect 标准](https://inspect.aisi.org.uk/) 与当前评估框架的对比。
   - 另一名成员澄清说，**Inspect** 似乎*只是另一个评估框架*，其重点在于标准化结果的保存和查询方式，而不是运行方式。
- **`lm_eval` 进度条在多 GPU 环境下失效**：一名成员报告称，`lm_eval` 中的进度条在多 GPU 设置中仅跟踪一个 GPU 的进度。
   - 另一名成员表示，默认情况下其他 rank 的 `tqdm` 是禁用的，并建议修改 [huggingface.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L1108) 中的一行代码。
- **推理模型生成：非同小可**：一名成员提到，处理推理模型的生成内容并非易事，需要修改每个任务配置（task config）中的答案提取逻辑。
   - 他们将在 GitHub 上创建一个 issue。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1382907345520169081)** (1 messages): 

> `Canvas 下载, Canvas PDF 导出, Canvas docx 导出, Canvas markdown 导出, Canvas 代码导出` 


- **Canvas 开启下载功能！**：Canvas 现在支持下载；如果你正在编写文档，可以将其导出为 **PDF**、**docx** 或 **markdown**。
- **Canvas 直接导出代码！**：如果你使用 Canvas 编写代码，它将直接导出为相应的文件类型（例如 **.py**、**.js**、**.sql** 等）。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1382801906497355906)** (153 messages🔥🔥): 

> `GPT Pro 模型并行处理, O3 Pro 性能问题, 免费 AI API, Discord 活跃度下降, ChatGPT 高级语音更新` 


- **OpenAI 的 GPT Pro 模型使用并行处理以增强推理**：主流理论认为，像 **O3 Pro** 这样的 **GPT Pro 模型** 通过并行运行多个实例并整合结果（被称为“更努力地思考”）来实现增强推理。**O3 Pro** 的思维链（Chain of Thought）总结中自称为“我们”，这支持了多个实例协同工作的观点。
   - 在 **AIME 2024** 数学竞赛中，**O3-pro** 达到了 **93%** 的 pass@1 准确率，而 **O3** 为 **90%**，暗示了这种整合方法带来的性能提升。
- **用户报告 O3 Pro 在项目中失效，尽管等待时间很长**：多名用户报告称，**O3 Pro** 无法回答来自上传文档的问题，即使在等待了长达 **40 分钟** 且未显示**思维链**的情况下也是如此。
   - 这种糟糕的表现与对 **O3 Pro** 所谓增强推理能力的预期形成鲜明对比，导致用户质疑其在实际应用中的有效性。
- **AI 爱好者探索用于开发的免费 AI API**：尽管 **ChatGPT Plus** 需要付费，但开发者讨论了其他免费 AI API，如拥有快速 **Llama 3.3 70B**、**Qwen** 和 **Deepseek** 的 **SambaNova**。
   - **Gemini** 因其高频率限制而被重点关注，**2.5 Flash** 提供 **500次/天**，**2.0 Flash** 提供 **1k次/天**，支持高达 **1M prompt** 和 **64K output**，使其成为预算有限的 AI 项目的可行选择。
- **随着 AI 聊天普及率飙升，Discord 活跃度下降**：用户观察到 **Discord 活跃度** 大幅下降，这与 **AI 聊天** 的兴起相关，许多服务器变成了“鬼城”。
   - 这一转变表明用户正在迁移到 AI 驱动的平台进行讨论，影响了 Discord 等传统平台的社区参与度，这引发了对社区互动的新思考。
- **用户批评 ChatGPT 新高级语音的厌烦语气**：用户表达了对 **ChatGPT** 新**高级语音**（advanced voices）的不满，称其听起来很“厌烦”，使用了过多的填充词，且整体传达出一种轻蔑感。
   - 一些用户更喜欢之前版本那种略显虚假的人工热情，而另一些人则建议理想的解决方案应该是可以选择语音人格（类似于 **Grok**），或者像 **ElevenLabs** 那样创建自定义语音。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1382805720814325893)** (47 messages🔥): 

> `GPT-4o 记忆, 微调 GPT 模型, 模仿写作风格` 


- **GPT-4o 可能会回忆起过去的对话！**：一位成员报告称，**GPT-4o** 能够直接引用与自定义 GPT 在另一个独立对话线程中共同创作的虚构场景的*逐字*内容，甚至包括用户自己创作且未告知 GPT-4o 的部分。
   - 虽然另一位成员认为这可能是*极少数情况下的准确推理*，但原帖作者并不同意，理由是其**统计学上的不可能**，并表示可以通过私信提供更多细节。
- **选择 Mini 还是 Nano 进行微调？**：一位成员询问应该使用哪种 **GPT 模型**（*4.1 mini 或 nano*）进行微调以模仿写作风格。
   - 有成员建议，如果不用考虑成本，可以两者都尝试并对比结果；否则，使用较便宜的模型。讨论还涉及了成本与性能之间的权衡，以及所需的训练样本数量。
- **只有在用户允许的情况下，ChatGPT 才会透露用户名！**：一位成员指出，即使启用了记忆功能，**ChatGPT** 也只有在用户允许的情况下才会透露用户名。
   - 该成员强调，如果在对话中提出要求，ChatGPT 会遵循用户的指令。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1382799798465466559)** (19 messages🔥): 

> `散弹枪式垃圾邮件发送者, 上传 HTML 到 o3, 使用 Pandoc 将 HTML 转换为 Markdown, 来自 o3 和 o3-pro 的长篇回答` 


- **散弹枪式垃圾邮件发送者回归**：一位用户报告了*散弹枪式垃圾邮件发送者*的回归，并附带了一个[链接](https://chatgpt.com/share/684c40f4-81cc-8003-8adb-ad408bbc676a)，展示了关于将 HTML 文件上传到 **o3** 的最佳实践的主要问答。
   - 对话本身提到：“*没问题，我可以解析大量交错的标签，即使是很长的文件也能理解其大意。*”
- **Pandoc：HTML 解析的瑞士军刀电锯**：一位用户建议使用 [Pandoc](https://pandoc.org/) 将 HTML 转换为 Markdown 以获得更好的解析效果，称其为*专门构建且广泛使用的工具*，而非*临时凑合的方案*。
   - 他们建议使用 **Pandoc** 而不是使用 **awk, sed 或 tr** 脚本来解析 HTML，同时也承认这些工具在处理一次性任务时的效用。
- **AI 可以处理 HTML 标签**：一位用户确认 AI 模型是在大量标签数据上训练出来的，可以处理 HTML，并建议只有在追求*绝对最高精度*时才有必要进行算法剥离。
   - 他们补充说，HTML 标签会产生 *Token 中的噪声*，这有利于推理；虽然*在出问题之前它们似乎无关紧要*，但它们确实是额外的上下文填充物。
- **数据准备：成功的一半**：一位用户指出，在 AI 项目中，数据准备、处理和格式化通常占据了约一半的工作量。
   - 典型任务包括*从 PDF 中提取文本*或*合并 JSON*，这突显了高效数据处理的重要性。
- **用户寻求从 o3 模型获取长篇回答的 Prompt**：一位用户正在寻求一种 Prompt，以便在审阅文件或进行深入研究时，让 **o3** 和 **o3-pro** 给出长篇回答。
   - 该用户观察到，这些模型倾向于生成简洁的要点和对比表，这与 **Sonnet** 和 **Opus 4** 形成了鲜明对比。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1382799798465466559)** (19 messages🔥): 

> `Shotgun Spammers, o3 Model File Uploads, HTML Parsing, Pandoc Conversion, Data Preparation` 


- **关于向 o3 上传 HTML 文件的最佳实践讨论**：一位用户询问了向 **o3 model** 上传包含大量 HTML 内容（通过 **devTools scraping** 获取）的文件的最佳实践，并分享了一个 [ChatGPT 链接](https://chatgpt.com/share/684c40f4-81cc-8003-8adb-ad408bbc676a)。
   - 该用户发现结合使用 **JS** 和 **AWK** 脚本可以获得非常好的效果。
- **ChatGPT 可以解析 HTML，但通过算法剥离标签的准确度最高**：一名成员确认，虽然 **ChatGPT** 在大量标签数据上进行过训练且输出通常正确，但通过算法剥离标签能提供最高的准确度。
   - 他澄清说，错误是由 *模型随机性 (model stochasticity)* 导致的，而并非严格因为标签的存在。
- **Pandoc 成为 HTML 转 Markdown 的首选工具**：一位用户建议使用 **Pandoc** 将 HTML 转换为 Markdown 进行解析，认为这比使用 **awk** 等工具进行的“粗糙处理 (hack jobs)”更好。
   - Pandoc 被描述为一个“专门构建且广泛使用的工具”。
- **HTML 标签是噪声 Token**：一名成员指出 **HTML tags** 增加了 *噪声 Token (noisy tokens)*，这可能对推理有好处，而另一名成员则认为它们只是额外的上下文填充物。
   - 如果你只是把一个网站的内容粘贴到 ChatGPT 中提问，那其实无所谓……但如果你正在构建某种流水线并为每个 Token 付费，那可能就很重要了。
- **用户寻求让 o3 模型生成长篇回答的 Prompt**：一位用户询问如何编写 Prompt，以便在使用 **o3** 和 **o3-pro** 审查文件或对某个主题进行深入研究时获得 **长篇回答 (long-form responses)**。
   - 该用户指出，这两个模型即使在收到明确指令时，往往也表现得非常简洁。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1382799445237956830)** (210 messages🔥🔥): 

> `Model Underfitting and Overfitting, Hugging Face's Definition of Open Source Models, Interpretability of Transformers Models, HF Spaces, Qwen 2.5 and Multilingual capabilities` 


- **利用 TensorBoard 图表诊断模型拟合情况**：成员们讨论了使用 **TensorBoard loss graphs** 来诊断模型拟合问题，指出 *验证损失 (evaluation loss)* 的下降速度应与 *训练损失 (training loss)* 相似，但不应低于后者。
   - 一名成员强调了将数据集分为 **训练集** 和 **测试集** 的重要性，以确保模型具有良好的 *泛化 (generalized)* 能力，而没有出现过拟合或欠拟合。
- **Hugging Face 因开源模型定义面临审查**：有人担心 **Hugging Face** 上的某些模型并非完全开源，可能只是利用该平台进行营销而非真正的开放协作。
   - 一名成员指出，虽然 **Hugging Face** 没有明确将自己标榜为 *开源 (open source)* 库，但其声誉倾向于此；而另一名成员提到 *任何仓库都可以使用任何许可证*。
- **视觉模型可解释性热点**：成员们寻求使用 **LLaVA** 等视觉模型在图像上 **可视化注意力图 (attention maps)** 的帮助，以实现模型的可解释性。
   - 他们询问是否有人在 **Transformer** 模型的 **可解释性 (interpretability)** 或 **可说明性 (explainability)** 方面有经验。
- **解决 Space 休眠问题**：成员们讨论了如何使用 `huggingface_hub` 库中的 `HfApi` 为 HF Space 设置休眠时间，以及如何让你的 [Space 在不活动 1 小时后进入休眠](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage-spaces)。
   - 注意：*如果你使用的是 ‘cpu-basic’ 硬件，则无法配置自定义休眠时间。你的 Space 将在不活动 48 小时后自动暂停*。
- **Qwen 2.5 摘得多语言桂冠**：成员们注意到 **Qwen 2.5** 具备说 *100 种语言* 的能力，并将其与 **Gemma3** 进行了比较，其他人则强调它对 **Linux VM** 的利用非常出色。
   - 有推测认为，凭借 *18T tokens* 的训练量，该模型包含大量的多语言数据，从而造就了其精通程度。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1382951209131966504)** (3 messages): 

> `` 


- **未发现酷炫发现**：在提供的消息历史中没有值得报告的有趣发现。
- **频道冷清**：该频道的活跃度较低，只有几条与用户通知和一般请求相关的消息。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1382865843184078849)** (5 messages): 

> `X Scraper, 数字孪生 AI 平台, Augmentoolkit 3.0 发布, 字段注入攻击` 


- **X-Scraper 开放 API 端点亮相**：一个带有开放 API 端点的 X-scraper 已经创建并可供使用，示例数据集可以在其 [Hugging Face 组织页面](https://huggingface.co/MasaFoundationCloneMe)上找到。
   - 这些数据对任何构建 **AI 模型**、**Agent** 或**应用程序**的人员都是免费开放的。
- **CloneMe 平台发布数字孪生工具包**：[CloneMe AI 平台](https://github.com/vibheksoni/cloneme)允许你构建自己的**数字孪生（Digital Twin）**——一个能像你一样聊天、记住细节并支持多平台的 **AI**。
   - 它具有可定制性、**内存驱动**且支持**热重载**，是创建智能化、动态 AI 人格的强大工具包。
- **Augmentoolkit 3.0 增强 AI 训练**：[Augmentoolkit 3.0](https://github.com/e-p-armstrong/augmentoolkit) 已发布，用户只需添加文档或通过对尝试进行评分来教导 AI 执行任务，即可**训练 AI** 理解新主题。
   - 它支持**自定义模型**运行，成本更低，并能更好地控制更新时间和方法。
- **字段注入攻击分析**：一篇关于**字段注入攻击（Field Injection Attacks）**及其对 **MCP 服务器**和系统潜在影响的详细文章已撰写并分享在 [LinkedIn](https://www.linkedin.com/posts/subham-kundu-2746b515b_cybersecurity-ai-machinelearning-activity-7339287857447981056-4BAz?utm_source=share&utm_medium=member_desktop&rcm=ACoAACZeVjgB0HEDqU1BExX1Ypnp-q8LcgDAunk) 上。
   - 文章解释了此类攻击如何危及 MCP 服务器和系统。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1383147207263391745)** (2 messages): 

> `论文展示` 


- **关于论文展示类型的澄清**：一位成员询问展示是用于展示他人的论文还是自己的论文。
- **两种形式均可**：另一位成员回答说*两种形式都可以*。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1382929175337369661)** (6 messages): 

> `Kaggle 数据集, Gemini 2.5 Pro 弃用, 开源 LLM/VLM 替代方案, Mistral LLM 作为替代方案` 


- **建议使用 Kaggle 发现数据集**：一位成员建议在 **Kaggle** 上查找数据集，称其为寻找数据集的*最佳选择*。
   - 该建议是针对另一位成员的请求提出的。
- **Gemini 2.5 Pro 的弃用促使转向开源搜索**：一位成员报告了 **Gemini 2.5 Pro** 即将弃用以及其替代品性能较差的情况，因此需要为其产品寻找强大的开源 **LLM/VLM** 替代方案。
   - 该成员希望系统能够抵御*公司的反复无常*，并认为任何低于 **70B** 训练参数的模型可能都不够用。
- **提议 Mistral LLM 作为 Gemini 替代方案**：一位成员建议 **Mistral LLM** 可能是最接近 **Gemini** 的开源替代品，但提醒在本地运行时不要期望达到同样的性能水平。
   - 他们建议提示工程（Prompt Engineering）可以作为一种“垫片”（shim）来缓解性能差异。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1382951771592327238)** (2 messages): 

> `GPTs Agent, OpenAI 侧边栏` 


- **GPTs Agent 在初始训练后无法学习**：一位成员担心 GPTs Agent 无法从初始训练后提供的额外信息中学习。
   - 另一位成员澄清了这一误解，解释说[上传的文件被保存为“知识”文件](https://link.to/openai-docs)供 Agent 在需要时参考，但**它们不会持续修改 Agent 的基础知识**。
- **OpenAI 平台侧边栏发生变化**：一些成员讨论了 platform.openai.com 侧边栏的变化。
   - 有人报告说**两个图标**从侧边栏消失了（一个是线程图标，另一个是消息图标）。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1382814040262443109)** (10 条消息🔥): 

> `Agents 课程报名链接, FinalAnswerTool.forward() 错误, 课程完成截止日期` 


- ****Agents 课程报名链接**失效了？**: 一位成员报告课程的 [报名链接](https://bit.ly/hf-learn-agents) 似乎失效了，但 *现在似乎已经恢复正常了！*。
- ****FinalAnswerTool.forward()** 错误困扰 Tool Calling Agents**: 一位成员在处理 **Tool Calling agents** 时遇到了 `FinalAnswerTool.forward() missing 1 required positional argument: 'answer'` 错误。
   - 该用户表达了沮丧，称 *这太让人抓狂了*。
- **Agents 课程和 MCP 课程的截止日期困境**: 一位正在开始 **Agents 课程**（截止日期 **7 月 1 日**）和 **MCP 课程**（截止日期 **8 月 1 日**）的成员表示担心会 *陷入困境*。
   - 该成员询问该选择哪门课程以及是否有影响，暗示时间限制可能迫使他在两者之间做出选择。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1382796931582918676)** (208 条消息🔥🔥): 

> `Manus 宕机, Veo3, Manus playbooks, 等待 Claude 4.0, Manus 积分` 


- ****Manus 崩溃**是由 Veo3 热潮引起的？**: 用户报告了 Manus 的大面积问题，怀疑是 **Veo3 发布** 导致服务器过载，[Downdetector](https://downdetector.com/) 证实了这一点。
- ****Playbooks 入门**预先准备提示词**: Manus 中的 **Playbooks** 预先准备提示词并提供输出示例，为需要提示词协助的用户弥补差距，但 [它们也旨在突出创意工作流](https://manus.im/playbook)。
- ****Claude 狂热**社区持续呼吁**: 用户表达了对 **Claude 4.0** 的渴望，幽默地将其类比为粉丝的期待，但 Google 官方并没有关于 **Claude 4.0** 发布的消息或更新。不过，一位合作伙伴建议 *注册新的 gmail 并申请 google one ai 试用 -> 组建家庭组 -> 邀请 5 个账号 -> 现在 veo 和所有功能都有 5 倍的使用量，因为所有账号都有独立的使用限制*。
- ****积分紧缺**成本担忧持续**: 用户对 **积分使用** 表示担忧，特别是关于优化和缺乏成本预览的问题，一些人建议采用 *自带密钥 (bring your own keys)* 模式，一位用户表示 *我启动的每个任务大约需要 900-1000 积分*。
- ****图像瑕疵** GPT 表现更佳**: 一位用户发帖对比了 Manus 和 GPT-4 Omni 的图像生成质量，显示 [GPT-4 Omni](https://cdn.discordapp.com/attachments/1349440650495398020/1382980409243209859/GPT4omini.png?ex=684dc920&is=684c77a0&hm=1d615e514982fcfdfb5677c8640ae6d7ea8282e5f56d86e5382a2578a0084b82&) 的表现优于 Manus。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1383015183957495869)** (4 条消息): 

> `Triton Kernel 优化, 卷积实现, Triton 中的内存读取` 


- **Kernel 共享请求！**: 一位用户请求使用 grid 的 `solve` 函数，这引发了关于 Triton kernel 优化的讨论。
   - 代码涉及输入、kernel 和输出的指针，计算输出大小、blocks，并启动具有指定 block size 的 `conv1d_kernel`，旨在讨论 Triton 中卷积操作的优化。
- **Triton 内存读取之谜获解**: 一位用户询问了 Triton kernel 中增加的内存读取（对于 2048 的 kernel size，每个 block 有 4096 次读取）以及为什么它仍然更快。
   - 作者要求澄清用户所说的 *“kernel size 为 2048 时有大量读取”* 是什么意思，从而引发了关于 Triton 内部内存访问模式和优化的讨论。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1382846463121555516)** (10 messages🔥): 

> `Blackwell Memory Layout Optimization, CUDA-side Boost, VS Code Plugins for CUDA Development, L1 and L2 Cache Policy` 


- **Blackwell 内存布局库引发讨论**：一名成员询问了一个旨在优化跨 **L2 cache** 内存布局的 **Blackwell** 库。
   - 另一名成员询问发帖者他们是如何尝试为 **L1** 和 **L2** 缓存设置策略的。
- **CUDA-side Boost 库助力开发**：一名成员分享了 [cuda-side-boost](https://github.com/ademeure/cuda-side-boost) 的链接，这是一个用于 **CUDA** 开发的库。
   - 该成员指出，*替换整个 PyTorch 内存分配器可能大材小用*，可以直接使用 **PyTorch** 中的 **MemPool**。
- **探索多种 VS Code 插件**：一名成员询问除了 **Nsight** 之外，还有哪些推荐用于 **CUDA** 开发的 **VS Code** 插件。
   - 另一名成员建议使用 **PTX syntax highlighting** 和 **CMake** 集成，以便于调试。
- **分享缓存策略代码片段**：一名成员分享了一个 **CUDA** 代码片段，演示了如何创建和使用缓存策略对象。
   - 该代码片段包含用于创建分级 **L2 cache** 策略（带有逐出策略）并在加载指令中使用的汇编指令，包括 `createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64`。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1382891251384193054)** (3 messages): 

> `Torch.compile speedup, Knowledge Distillation with torch.compile, PyTorch CMake and CUDA Architecture Selection` 


- **Torch Compile 意外地加速了卷积算子**：一名成员观察到，由 **torch.compile** 生成的操作往往运行得更快，即使没有进行任何融合（fusion），并注意到卷积算子从 **1.7489 ms**（原生 PyTorch）显著加速到 **0.0020 ms**（编译后的 PyTorch）。
   - 编译版本调用的是 *extern_kernels.convolution* 而不是 *aten.convolution*，这引发了关于为什么标准卷积不使用这些更快的外部算子的疑问。
- **Torch Compile 在知识蒸馏中面临挑战**：一名成员询问如何为知识蒸馏设置 **torch.compile**，特别是当大型教师模型（如 resnet50）处于 eval 模式而较小的学生模型（如 resnet18）处于 training 模式时。
   - 他们遇到了与张量重写相关的运行时错误，具体错误消息指出需要在 **torch.compile()** 之外克隆张量，或者在每次模型调用前调用 **torch.compiler.cudagraph_mark_step_begin()**。
- **PyTorch CMake 无条件忽略 CUDA 架构选择**：一名成员报告受到 PyTorch CMake 脚本的影响，特别是其中一行无条件忽略了用户提供的 CUDA 架构选择，导致代码因假设可以访问 **cuda::atomic** 而崩溃。
   - 他们质疑了一条关于不依赖 CMake **3.18** 版本的注释的相关性，并建议根据 CMake 版本以及是否存在用户提供的架构选择来保护这些有问题的代码行，以实现向后兼容性。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1383150496444977213)** (2 messages): 

> `Post-PMPP reading recommendations, Instruction Latency Paper` 


- **成员寻求 PMPP 之后的阅读材料**：一名成员询问在完成 **PMPP** 之后有哪些推荐阅读的书籍或论文。
   - 另一名成员通过推荐 [一篇关于指令延迟的论文](https://arxiv.org/pdf/1903.07486) 进行了回复。
- **推荐指令延迟论文**：一名成员建议阅读 [一篇关于指令延迟的论文](https://arxiv.org/pdf/1903.07486)，尽管 **instruction latencies** 可能已经过时。
   - 该成员认为 *讨论本身非常值得一读*。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1383071924023857229)** (2 messages): 

> `ROCm 7 Access, ROCm Release Date` 


- **热切期待 ROCm 7 访问权限**：一名用户询问如何获得 **ROCm 7** 的访问权限，预计其将于 8 月发布。
- **建议耐心等待 ROCm 7**：社区成员建议等待 AMD 官方发布公告，以获取有关访问 **ROCm 7** 的详细信息。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1383108005234741450)** (2 messages): 

> `GemLite ROCm support, NVIDIA MMA Instruction, Tensor Cores in Mojo, Custom MMA instructions via LLVM intrinsics, Data layouts with Mojo's load_matrix and store_matrix APIs` 


- ****GemLite 准备好支持 ROCm****：一位开发者宣布为 **GemLite** 添加 **ROCm support**，重点针对 **MI300X**（[X 上的帖子](https://x.com/mobicham/status/1933520405106507909)）。
- ****Mojo Tensor Core 编程****：一篇博客文章探讨了 **NVIDIA** 的 **mma instruction** 及其在 **Mojo** 中的应用，教用户使用 **Mojo** 的 **mma API**（[博客文章](https://veitner.bearblog.dev/programming-tensor-cores-in-mojo/)）。
   - 该文章详细介绍了如何通过 **LLVM intrinsics** 实现 **custom mma instructions**，并利用 **Mojo** 的 **load_matrix** 和 **store_matrix APIs** 高效管理 **data layouts**（[GitHub 仓库](https://github.com/simveit/mma_mojo)）。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1382900506736726076)** (3 messages): 

> `H100 Conv2D, AMD FP8 MM, MI300, Leaderboards` 


- **H100 加速 Conv2D 排行榜**：一名成员在 **H100** 上的 `conv2d` 排行榜中以 **47.8 ms** 的成绩获得 **第 4 名**。
   - 该成员还在 **H100** 上成功提交了一次 **187 ms** 的成绩。
- **MI300 加入 AMD-FP8-MM 角逐**：一名成员在 **MI300** 上成功提交了 `amd-fp8-mm` 排行榜成绩，用时 **5.23 ms**。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1382843564450385921)** (1 messages): 

> `CUDA 12.3, CC 10.3` 


- **CUDA 确认 B300 支持 CC 10.3**：NVIDIA 的 [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cufft-release-12-9-update-1) 确认 **B300** 支持 **CC 10.3**。
- **另一个 B300 确认**：另一个关于 **CUDA** 支持 **B300** 的确认。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1382803566426587226)** (111 messages🔥🔥): 

> `FLE standalone docker image, Factorio TAS Generator integration, RL policy learning in Factorio, LLM vs RL for Factorio` 


- **FLE 独立 Docker 镜像初现**：一名成员为一个独立的 **FLE docker image** 和 mod 创建了 [POC 项目](https://github.com/MortenTobiasNielsen/fle_suggestion)，但在将其集成到主代码库时遇到了挑战。
   - 另一名成员测试了该设置并报告在其系统上运行正常，而另一名成员在加入多人游戏时遇到了同步错误（desync error）。
- **Factorio TAS Generator 进入实验室**：一名成员提到一个 **Factorio mod**，它可以记录 **Factorio TAS Generator** 应用程序的步骤，用于生成自动化游戏所需的 **steps.lua** 文件。
   - 讨论中提到一位用户为 **Factorio** 的工具辅助速通（Tool Assisted Speedrun）手写了 35,453 个步骤，强调了创建像 **Factorio TAS Generator** 这样工具的动力。
- **Code-as-policy**：一名成员建议将 **code-as-policy** 作为在这一抽象层之上构建完整 **RL loop** 的潜在更快方法，并强调了重度的 **reward shaping**。
   - **Code-as-policy** 是指使用程序合成（program synthesis）作为动作，进行重度的奖励塑造，并在该抽象层之上构建完整的 **RL loop** 会更快。
- **Factorio 中的 LLM vs RL 较量开始**：成员们讨论了使用基于 **RL** 的 **AI** 玩 **Factorio** 的潜力，辩论 **LLM** 对于长期规划和复杂任务是否必要。
   - 对话探讨了 **RL Agent** 是否能在有限的游戏次数下实现最优的 **Factorio** 玩法，并与 **OpenAI Five** 在 **Dota 2** 中的成功进行了对比。
- **探索 LLM-RL 光谱**：围绕将 **LLM** 作为“人类先验知识倍增器”展开了讨论，建议调整脚手架（scaffolding）以启动基础功能，可能比完全 DIY 或计算资源有限的 **RL** 更好。
   - 分享了一篇集成 **RL** 以改进主要基于 **LLM** 系统的[论文](https://arxiv.org/abs/2402.19299)，强调了样本效率与长期能力之间的权衡。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1382798214716981368)** (36 条消息🔥): 

> `AMD 会议，在 Workshop 202 见面，颁奖典礼时间，从会议离开` 


- **AMD 会议参会者齐聚**：**AMD 会议**的参会者安排在午餐区和 **Workshop 202** (Room 212C-D) 见面。
   - 一位穿着*红色衣服并戴眼镜*的成员提议在房间外见面。
- **炉边谈话 (Fireside Chat) 的小插曲**：一名寻找会议的成员最初发现*空无一人*，随后澄清他们是在房间后部参加**炉边谈话**。
   - 该成员澄清说，需要 *ping* 其他人来通知他们。
- **AMD 活动官方照片悬而未决**：参会者发布了活动照片，但一名成员询问*有人知道在哪里可以找到官方照片链接吗？*。
   - 聊天中未提供任何链接。
- **会议参会者开始离场**：成员们表示正在乘机返回，其中一人在 **下午 3 点飞往巴黎**，另一人询问 **下午 2 点飞往慕尼黑** 的航班。
   - 成员们对在会议上结识彼此的机会表示感谢。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1383083518145466368)** (6 条消息): 

> `新的 Cutlass DSL 学习资源，Sm90ScalarReduction 适用性，CuTeDSL 对分布式共享内存的支持` 


- **请求新的 Cutlass DSL 资源**：一名成员询问了关于新 **Cutlass DSL** 的学习资源，特别是今年 **GTC** 的视频。
   - 该用户可能正在尝试寻找为他们的项目学习 **cutlass** 的最佳途径。
- **研究用于列归约的 Sm90ScalarReduction**：一名成员考虑将 **Sm90ScalarReduction** 用于他们的问题，最初认为它可以解决涉及每列绝对值最大值（chebyshev）的问题。
   - 他们后来意识到 **Sm90ScalarReduction** 并不完全符合需求，并建议假设的 **Sm90ColumnReduction** 会更合适。
- **询问 CuTeDSL 的分布式共享内存支持**：一名成员询问 **CuTeDSL** 现在是否支持分布式共享内存 (distributed shared memory)。
   - Their project requires a reduction operation between threadblocks, and they are seeking the easiest way to implement it, implying interest in **CuTeDSL** for this purpose.
   - 他们的项目需要在 threadblocks 之间进行归约操作，正在寻找最简单的实现方式，这暗示了对使用 **CuTeDSL** 实现此目的的兴趣。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1382798663574753422)** (72 条消息🔥🔥): 

> `AMD GPU, Patchscopes Google 框架, Unsloth YouTube 频道, MLflow 问题, Unsloth 周边` 


- **AMD GPU 获得 Unsloth 支持**：随着新款 **AMD INSTINCT MI355X GPU** 拥有 **5 倍于 H100 的 FP8 flops**，Unsloth 团队可能开始认真对待 **AMD**，并且他们[在 AMD AI 会议上进行了演讲](https://x.com/danielhanchen/status/1933150291719082098)。
   - 成员们注意到 **AMD 价格便宜且显存高**，同时也对 **AMD 的驱动支持**提出了疑问。
- **来自 Google 的 Patchscopes**：成员们分享了 [Patchscopes](https://github.com/PAIR-code/interpretability/tree/master/patchscopes/code) 的链接，这是一个**来自 Google 的框架**。
   - 一名成员提到想看看它是否适用于 **LLaVA** 等模型，而另一名成员正在使用 Unsloth 微调 **Qwen 2.5 7B**，并需要一个小型的法语数学数据集。
- **Unsloth 将创建 YouTube 频道？**：Unsloth 团队正在*考虑创建一个 YouTube 频道*来上传视频。
   - 一名成员特别要求他们上传关于如何通过 accelerate 使用多 GPU 的视频，并承诺会*点赞并订阅*。
- **多 GPU 支持浮出水面**：据报道有 *5 个不同的多 GPU (multiGPU)* 支持仓库，[这个 Reddit 帖子](https://www.reddit.com/r/unsloth/comments/1l8mxkq/multigpu_support_how_to_make_your_unsloth/) 就是其中一个例子。
   - 官方支持仍在开发中，预计效果会非常好。
- **MLflow 模型加载陷阱**：一名用户在从 **MLflow** 而非 **Hugging Face** 加载模型时，尽管使用了*完全相同的配置、超参数和流水线*，其**微调流水线**仍遇到了问题。
   - 他们观察到 **loss 停留在 3–4 左右而没有趋于零**，即使将训练数据集扩大了一倍也是如此，并寻求帮助以调试或解决该问题。

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1382797083999993978)** (9 messages🔥): 

> `80GB VRAM Typo, GPU RAM` 


- **互联网受 80GB VRAM 拼写错误困扰**：用户们在争论一份列出 **2TB RAM** 的规格表是否为拼写错误，普遍假设它本应是 **80GB VRAM**。
   - 有人认为这不是拼写错误，而是一种懒惰的假设，即认为读者在看到 **GPU** 广告时会明白这指的是 **VRAM**。
- **电脑配置单将 GPU RAM 列为 80GB**：一位用户报告看到一个电脑配置单明确标注 **GPU RAM: 80GB**。
   - 另一位用户对此表示怀疑，称 *"绝无可能💀"*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1382807355116486666)** (59 messages🔥🔥): 

> `Fine-tuning Llama 3.2 for tool calling, AttributeError during training with Unsloth, Qwen3 (4B) fine-tuning issues with Unsloth GRPO, Accelerate with Unsloth for multi-GPU, Fine-tuned model inference speed` 


- **利用 GPT-4 进行 Llama 3.2 工具调用微调**：一位成员计划使用 **GPT-4** 生成对话示例和合成样本，以便为 **6 个自定义工具**微调 **Llama 3.2 (3B)**，并寻求关于该方法以及实现零样本（zero-shot）工具调用所需样本数量的指导。
   - 有观点指出，在对话中间使用工具至少需要 **14B** 参数的模型，因为 **Llama 3.2 模型表现欠佳**。
- **解决 Unsloth 训练中的 AttributeError**：一位用户在利用 Unsloth 训练时遇到了 **AttributeError**，经追溯发现是 `fetch_image` 函数试图读取一个 `None` 的 **images 字段**，而非有效的路径或 URL。
   - 建议如果是批处理（batch），则整个批次需要同时包含图像和文本，或者仅包含文本；解决方案是尝试 **batch size 1** 或传递自定义的 collator。
- **应对 NaN Loss 的陷阱**：一位用户报告在 **GRPO 训练**期间遇到 `nan` loss，在之前的 SFT 修复失败后寻求解决方案。
   - 建议降低训练率（training rate）并检查导致问题的特定异常数据点，同时确保 notebook 与加载的 4-bit 模型之间的兼容性。
- **满足快速 TTS 迭代的需求**：一位用户寻求一种快速的方法将编程助手集成到他们的 **R/RStudio 工作流**中，使用的是 **Qwen2.5 Coder 32B Instruct GGUF**，且不确定如何将 *no_think* 设置为默认值。
   - 建议他们基于 **Qwen3** 创建一个新模型并在其中设置 *no_think*，同时考虑到非指令（non-instruct）模型可能更适合此类工作。
- **保护模型免受幻觉影响**：在成功微调模型后，一位用户询问如何**防止模型在输入脱离上下文时做出响应**，以及如何**防止幻觉**。
   - 建议使用 **grounding（接地/溯源）或 guardrails（护栏）** 来解决这些问题，尽管未提供具体的指南。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1382815695930396764)** (4 messages): 

> `French math dataset, Qwen 2.5 7B` 


- **为 Qwen 2.5 7B 寻找法语数学数据集**：一位成员正在（使用 Unsloth）微调 **Qwen 2.5 7B**，并寻找一个小型的法语数学数据集。
   - 另一位成员建议使用常规数学数据集并利用 **AI 进行翻译**，因为法语的现成资源可能不多。
- **利用 AI 翻译数学数据集**：可以使用 **AI 模型**进行翻译来转化数学数据集，从而增加 **Qwen 2.5 7B** 的训练数据量。
   - 这种方法可以避开寻找难以获取的母语级法语数学数据集的问题。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1383033641797423175)** (27 条消息🔥): 

> `VLLM 模型中的 Attention map 可视化，LLM 的 RL 训练中 KL 散度的陷阱，过河实验（River Crossing）的问题，Claude Opus 作为论文作者，nnsight.net` 


- **寻求 VLLM 模型的 Attention Map 可视化**：一位成员询问如何在使用 **LLAVA** 等 **VLLM 模型**时对图像进行 **attention maps** 可视化，寻求有关 Transformer 模型可解释性的工具或经验。
   - 另一位成员建议将 [nnsight.net](https://nnsight.net/) 作为潜在的起点，同时也承认需要进行自定义实现。
- **揭露 KL 散度梯度估计的缺陷**：分享了一篇讨论 LLM 的 RL 训练中 **KL 散度梯度估计**陷阱的论文，强调了 **TRL** 和 **Open Instruct** 等开源项目以及 **GRPO** 等论文中存在的问题。
   - 该论文指出，*将 KL 估计作为损失函数进行微分*且*未考虑序列特性*会导致错误的 KL 梯度，参考[这篇论文](https://arxiv.org/pdf/2506.09477)。
- **Apple 的推理模型充斥着过河实验错误**：分享了一篇题为《[思维幻觉的幻觉](https://arxiv.org/abs/2506.09250)》（The Illusion of the Illusion of Thinking）的论文，批评了 AI 模型在**过河实验**中的评估，因为这些评估无意中惩罚了那些正确识别出问题不可解的模型。
   - Apple 的原始论文中存在 **N ≥ 6 个参与者/Agent** 使用船只容量 **b = 3** 的情况，这在数学上是不可能的。
- **Claude Opus 作为论文作者获得学术认可**：一位成员幽默地注意到 **Claude Opus** 被列为论文作者的意外情况。
   - 有人开玩笑说 *我们是 Anthropic，我们不能坐视不管*。
- **更多趣闻出现**：聊天中出现了另一个有趣的链接 [https://arxiv.org/abs/2506.10943](https://arxiv.org/abs/2506.10943)。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1382843348829601904)** (101 条消息🔥🔥): 

> `nano-vLLM 发布，Morph Labs Trinity，o3-pro 工具讨论，AI Agent 构建错误，Transformers 库弃用声明` 


- **Nano-vLLM 受到关注**：DeepSeek 研究员 @xingkaiyu 发布了 **nano-vLLM**，这是一个仅约 **1200 行代码**的极简 vLLM 实现，引发了 AI/ML 从业者的兴奋，链接见[此处](https://xcancel.com/wzihanw/status/1933225058031288772?s=46)。
   - 社区赞赏其简洁性，认为它是宝贵的学习资源，一位用户表示有兴趣在这个 *“nano 单体”* 上进行开发。
- **Trinity 自动形式化费马大定理**：**Morph Labs** 推出了 **Trinity**，这是一个自动形式化系统，用于在 Lean 中形式化 de Bruijn 关于 abc 猜想的结果，链接见[此处](https://xcancel.com/morph_labs/status/1933181394588483868?s=46)。
   - 其目标是通过将数学知识转化为形式化证明，为数学领域的自监督强化学习创建经过验证的训练环境。
- **Transformers 库将仅支持 PyTorch**：**Transformers 库**将弃用对 **TensorFlow 和 Flax** 的支持，专注于 **PyTorch** 以减少冗余、简化工具包并移除抽象层，如[此处](https://xcancel.com/LysandreJik/status/1933201171130593530)所述。
   - **对 TF 和 Flax 的长期支持 (LTS) 将随 v4 版本持续到 2026 年年中**，这一变化标志着 v5 版本的开始，旨在移除 50% 的代码。
- **Meta AI 应用分享私人对话**：一个 **Meta AI** 应用无意中将用户的私人对话（包括敏感信息和音频）发布到了公共动态中，链接见[此处](https://xcancel.com/SHL0MS/status/1933019178023231880)。
   - 据澄清，由于 UI 容易引起混淆，用户在不经意间分享了内容，暴露了个人细节并引发了对 Meta 的伦理担忧。
- **Anthropic 探索多 Agent 威力**：**Anthropic** 发现，根据[这篇文章](https://www.anthropic.com/engineering/built-multi-agent-research-system)，*一个以 Claude Opus 4 为主 Agent、Claude Sonnet 4 为子 Agent 的多 Agent 系统，在我们内部研究评估中的表现比单 Agent Claude Opus 4 高出 90.2%*。
   - 他们还发现，*多 Agent 系统在涉及重度并行化、信息量超过单个上下文窗口以及与众多复杂工具交互的高价值任务中表现出色*，但消耗 Token 的速度很快，大约是**普通对话的 15 倍**。


  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1383157534973493328)** (4 messages): 

> `AI Engineering World's Fair 2025, Latent Space Podcast, AI Conference Recap, Documenting AI progress` 


- **Latent Space Podcast 回顾 AI Engineering World's Fair 2025**：[Latent Space podcast](https://x.com/latentspacepod/status/1933589087312871841?s=46) 账号分享了 **AI Engineer World's Fair 2025** 的回顾，重点介绍了参会者、演讲者、研讨会和周边活动的统计数据。
   - 播客鼓励参会者发布他们从会议中获得的见解和学习心得，强调了 **AI** 领域变化的飞速以及记录新信念和新联系的重要性。
- **X-Ware.v0 发布 AI Engineering World's Fair 2025 回顾**：Red - X-Ware.v0 发布了 **AI Engineering World's Fair 2025** 的回顾。
   - 回顾内容可在 [xcancel.com](https://xcancel.com/latentspacepod/status/1933589087312871841?s=46) 查看。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1382840044716560576)** (76 messages🔥🔥): 

> `mcpm aider troubles, github.com/Aider-AI/aider/pull/393, turn off auto updating, comparison b/w aider and cline, OpenAI` 


- **mcpm Aider 在使用 GitHub Copilot 时遇到问题**：用户报告了在使用 **GitHub Copilot** 时 **mcpm-aider** 出现的问题，导致一名用户 fork 了该项目。
   - 一名用户随后表示，尽管有错误，他们还是让它运行起来了，并将其描述为 *虽然笨拙但能用*。
- **保持在 Aider 的特定 fork 版本**：一名用户询问如何禁用自动更新以保持在 **Aider** 的特定 fork 版本上。
   - 另一名用户提供了解决方案：使用 `--no-check-update` 标志或将 `AIDER_CHECK_UPDATE` 环境变量设置为 false，如[此处文档](https://aider.chat/docs/config/options.html#--check-update)所述。
- **Aider 在处理较小模型时表现出色**：一名用户对 **Aider** 在使用 **Ollama** 运行较小模型（8B 和 12B）时的性能表示赞赏，指出与其他工具相比，它的效果出奇地好。
   - 另一名用户指出，这是因为上下文（context）处理得当。
- **Aider 在 JS 排行榜上名列前茅**：根据[这项基准测试](https://www.deccan.ai/blogs/anthar-study-evaluating-ai-coding-agents-beyond-benchmarks)，一名用户注意到 **Aider** 在 JS 排行榜上表现非常出色，特别是由于 Aider 的 repomap 功能。
   - 另一位用户提到他们所有的 JS 编码都使用 Aider 完成。对他们来说，*Aider 提供的灵活性、透明度和质量是无与伦比的，尤其是当 LLM 还不够聪明，无法成为真正的 Agent 时*。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1382861018236977194)** (20 messages🔥): 

> `uv for Python dependency management, Aider and library versions, Aider costs with Anthropic, Aider context size with Ollama, max_input_tokens` 


- **拥抱 UV 取得 Python 依赖管理胜利**：成员们讨论了从 **pip/virtualenv** 迁移到 **UV** 进行 Python 依赖管理，避免直接使用 **pip** 和编辑 **pyproject.toml**，而是使用 ```uv add <dependency name(s)>``` 等命令。
   - 一名用户提到他们最初对于阅读手册比较 *懒惰*，但发现通过 YAML 配置定义 linting 指令要 *严谨得多*。
- **Aider 对库版本的感知探索**：一名用户询问如何提高 Aider 对库版本的感知，并指出在从 **pip/virtualenv** 迁移到 **Poetry** 时，Aider 最初建议了过时的选项。
   - 建议包括通过 URL 提供更新后的 man pages 上下文，在约定中明确说明版本，以及使用 `/read docs/spec.txt` 命令。
- **使用 Anthropic 模型时的成本管理**：一名用户询问了在使用 Aider 配合 Anthropic 时的成本管理，对大型变更可能产生每小时近 **$50** 的成本表示担忧。
   - 用户还提到 **Claude Code** 月度计划很快就用完了，可能指的是超出了该计划的使用限制。
- **纠正 Ollama 在 Aider 中的上下文大小**：一名用户报告了一个差异，即 Aider 声称上下文窗口大小为 **131,072 tokens**，而 **Ollama** 设置为 **8k max context**。
   - 解决方案涉及调整 Aider 配置中的 **max_input_tokens** 设置，详见 [Aider 文档](https://aider.chat/docs/config/adv-model-settings.html#context-window-size-and-token-costs)。
- **max_input_tokens 得到澄清并解决问题**：一名用户最初在 Aider 中配置独立的输入和输出 max tokens 时遇到困难，特别是关于 *剩余 tokens* 的显示。
   - 经过澄清后，用户理解了其中的差异，并确认解决方案涉及正确设置 **max_input_tokens** 参数。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1382799916753223700)** (44 条消息🔥): 

> `Decentralized Compute Marketplaces, Vast.ai, Decentralized Pre-training vs Post-training, Infiniband, DAWN Internet` 


- ****Vast.ai** 是相对便宜的供应商**：一位成员建议使用 [Vast.ai](https://vast.ai) 进行去中心化计算，并指出尽管在依赖性方面有所妥协，但与其他供应商相比，它的价格*相对便宜*。
   - Akash 也被提及作为潜在的替代方案，但 Vast.ai 被认为更加便宜。
- ****Portal** 聊天界面出现错误**：一位成员报告在尝试使用 Portal 的聊天界面时收到错误，并[分享了错误的截图](https://cdn.discordapp.com/attachments/1149866623109439599/1383070010775179334/image.png?ex=684d73d2&is=684c2252&hm=5c92ab5aee6fb52007a0789e1b625f8b692b6cfc86d099c1c2d001b5a3591efb&)。
   - 前端团队正在调查该问题，并建议尝试在无痕窗口中访问聊天界面。
- **去中心化 Pre-training vs Post-training 是下一个伟大的基础设施**：一位成员提到他们正在为去中心化训练搭建基础设施，并且也在 [psyche.network](https://x.com/krishnanrohit/status/1933536577344700439?s=46) 进行 Pre-training。
   - 他们还指出，随着 GPU 扩散和网络性能的提升，分布式训练将会得到改善。
- ****Infiniband** 带宽远超互联网**：一位成员指出，互联网带宽（约 **1gbps**）在近年来没有太大增长，而 [Nvidia 最新的 Infiniband 迭代](https://x.com/toolandtea/status/1933381389552136705) 已达到 **130TB/s**。
   - 这种带宽差距是一个日益严重的问题。
- ****DAWN Internet** 提供去中心化互联网**：一位成员推广了 **DAWN Internet**，这是一个去中心化宽带协议，使用固定无线屋顶天线提供千兆互联网。
   - 他们的新 WiFi 路由器包含一个能够支持 **RL** 的 **GPU**，更多信息可以在[这里](https://x.com/dawninternet)找到。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 条消息): 

lazeewhalee: 或许可以参考 R1 DeepSeek 及其参考文献？
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1383061429464203317)** (2 条消息): 

> `C. Opus First Publication on Arxiv` 


- **C Opus 发布 Arxiv 出版物**：Teknium 分享了[一个帖子](https://x.com/WesRothMoney/status/1933502113285616083)。
   - 一位成员对这是 **C. Opus 在 Arxiv 上的首次发表**表示惊讶。
- **C Opus Arxiv 首秀**：一位成员注意到 **C. Opus** 的第一篇论文出现在了 Arxiv 上。
   - Teknium 分享了[公告帖子](https://x.com/WesRothMoney/status/1933502113285616083)的链接。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1382982241621114910)** (6 条消息): 

> `NVIDIA Cosmos, Talk at WebSummit, ArXiv Papers` 


- ****NVIDIA** 发布 **Cosmos****：**NVIDIA** 推出了 [Cosmos](https://github.com/nvidia-cosmos/cosmos-predict2)，ArXiv 论文见 [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)。
- **WebSummit 简短演讲**：一位成员分享了在加拿大温哥华 **WebSummit** 期间的一次[简短演讲](https://youtu.be/vZVcBUnre-c)，内容一半是历史，一半是吐槽，关于封闭互联网/封闭 AI。
   - 还有一个指向[这条推文](https://x.com/thdxr/status/1932929980612460818)的链接以获取更多背景信息。
- **新的 ArXiv 论文**：一位成员分享了一篇新的 ArXiv 论文：[https://arxiv.org/abs/2506.10943](https://arxiv.org/abs/2506.10943)。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1383061429464203317)** (2 条消息): 

> `C. Opus Arxiv Publication, teknium WesRothMoney X Post` 


- **Teknium 链接 WesRothMoney 的 X 帖子**：一位成员分享了来自 X 上 **WesRothMoney** 的[链接](https://x.com/WesRothMoney/status/1933502113285616083)。
   - 该 X 帖子的背景和内容未被进一步讨论。
- **C. Opus 完成 Arxiv 首秀**：一位成员对该 Arxiv 出版物是 **C. Opus** 的首篇论文表示惊讶。
   - 未提供关于该出版物的更多细节。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1382947224883040296)** (7 条消息): 

> `思维导图创建, Notebook LM Plus 访问权限, 子层级细节` 


- **思维导图杰作完成**：一名成员利用 **115+ 个来源** 创建了一个思维导图，声称其*相当准确*，并生成了一个*巨大的思维导图*来总结所有关键方面。
   - 另一名成员表示有兴趣了解更多相关信息。
- **付费 AI Pro 问题依然存在**：一名使用 **付费 AI Pro** 的成员仍然无法访问 **Notebook LM Plus**，并询问可能的原因。
   - 频道内未提供解决方案。
- **思维导图挖掘方法论**：一名成员询问了基于 **1900 个来源** 的思维导图中的子层级数量。
   - 回复指出该图有 **4 个子层级**，用户对垂直密度表示满意，但指出水平方向仍有改进空间，并[链接到了图片](https://cdn.discordapp.com/attachments/1124403655819415592/1383205649839820830/NotebookLM_Mind_Map.png?ex=684df225&is=684ca0a5&hm=9f877d3700e50d5c48e2faa411ec5c0e28b33088d6d859a72f7f2278d3660d3d&)。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1382809887209164980)** (39 条消息🔥): 

> `NotebookLM 中的 Excel 文件, 移动端 App 笔记, 图像支持, 分享 Notebook, 播客中断功能` 


- **NotebookLM 缺失 Excel 文件支持**：用户请求在 NotebookLM 中支持 **Excel** 和 **Google Sheets**，但目前尚无该功能的支持或路线图（roadmap）。
   - 建议用户在功能请求频道表达他们的兴趣。
- **移动端 App 笔记功能受限**：**Notes** 在 NotebookLM 的桌面版中可用，但移动端 App 仅显示 **sources**、**chat** 和 **studio** 部分。
   - 用户可以通过浏览器而非 App 在移动端访问笔记；目前没有导出选项，但用户可以进行复制粘贴。
- **图像支持的推送并非全量**：部分用户可以上传 **.jpg** 和 **.png** 文件作为来源，但其他用户则不行，且关于该功能的推送尚无官方公告。
   - 一种变通方法是将图像放入 **Google Doc** 或 **Slide** 中，然后下载为 **PDF** 以供 NotebookLM 使用。
- **由于按钮置灰无法分享 Notebook**：用户在分享 Notebook 时遇到问题，**“Share publicly” 按钮**呈灰色且无法点击。
   - 该问题的原因尚不明确。
- **NotebookLM 仅使用 LaTeX 标记**：用户发现 NotebookLM 在生成数学公式时使用了 **LaTeX markups**。
   - 这是正常现象，因为 NotebookLM 和其他 LLM 均使用 LaTeX 来表示数学表达式。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1382920304883404894)** (10 条消息🔥): 

> `Jacobi 方法 SVD, SVD 中的符号误差, 现代 SVD 算法` 


- **Jacobi 方法导致 SVD 不匹配**：一名成员在使用 **Jacobi method** 进行 **SVD** 时遇到了不匹配问题，特别是元素符号方面，最大绝对差值为 **1.8523843**。
   - 由于尺寸问题，该用户将 **eigh()** 和 **svd()** 分开进行测试。
- **SVD 符号误差被认为无关紧要**：一名成员建议 **SVD** 结果中的**符号误差**从根本上来说并不重要，只要方程 **A = UΣVT** 成立即可。
   - 该成员承认希望达到与 NumPy 性能相当的水平，但怀疑在 tinygrad 上是否可行。
- **Jacobi 方法已过时**：讨论强调 **Jacobi's method** 可能不是用于 **SVD** 的现代算法，且仅适用于对称矩阵。
   - 提到 **NumPy** 在底层使用 **QR Algorithm** 的变体，而 **Graham-Schmidt** 在 **full_matrices = True** 时不够准确。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1382829691320139877)** (34 messages🔥): 

> `BEAM linearizer failures, Float Matmul Accuracy Discrepancy (NumPy vs. Tinygrad), linalg.svd PR, QR algorithms variance, Numpy defaults to float64` 


- ****Beam Me Up (无 Linearizer 失败)****：一位用户询问在运行 **BEAM** 时遇到 **linearizer 失败**的问题，另一位用户确认他们也遇到了同样的问题。
   - 在提供的上下文中未确定具体的解决方案或原因。
- ****Tinygrad 的浮点数失误？****：一位用户注意到 **NumPy** 和 **Tinygrad** 之间 **float matmuls** 的精度存在差异，特别强调了[这段代码](https://discord.com/channels/924641739439477844/1147353713739749426)中输出矩阵左下角值的不同。
   - 随后讨论了不同编译器、优化技术以及 **IEEE 754 标准**对浮点运算的影响，一些人认为轻微的数值漂移是预料之中的，并可能受到操作顺序以及 **NumPy** 默认使用 **float64** 等因素的影响。
- ****SVD 符号出错？****：一位正在处理 **linalg.svd** PR 的用户试图达到与 **NumPy** 相当的精度，但发现得到了数值相同但符号不同的结果，并担心这种符号错误是否可以接受。
   - 另一位用户建议他们设置 `DEBUG=4` 来检查内核代码，并指出循环展开（loop unrolling）可能会引入数值差异；他们建议设置 `NOOPT=1` 来禁用优化以获得更接近的结果。
- ****QR 困惑****：一位用户发现了 **QR 算法**的方差，以及 **Householder Reflections** 与 **Gram-schmidt 过程**之间的差异。
   - 该用户发现，与 **NumPy** 用于特征值计算的 LAPACK 包相比，方差甚至更大，并感叹“真的只是在这上面浪费了大量时间”。
- ****NumPy 的数值麻烦：float64 默认值引发的混乱****：一位用户建议显式创建 `dtype=np.float32` 的 **NumPy** 数组以解决结果差异，并指出 **NumPy** 默认使用 `np.float64` 是很荒谬的。
   - 另一位用户反驳说，在机器学习之外的数值应用中，默认使用 **float64** 是很常见的，更改默认值可能会导致不相关的东西崩溃。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1382802713498357841)** (38 messages🔥): 

> `Map Variadic Types, MLIR Type Synthesis, Magic to Pixi Migration, GPU Puzzles Discussion, Mojo C ABI Export` 


- **映射可变参数类型（Variadic Types）仍是挑战**：成员们讨论了在 Mojo 中映射可变参数类型的挑战，引用了一篇[论坛帖子](https://forum.modular.com/t/map-variadic-types/1638?u=emil.martens)，并一致认为这“感觉像是一个简单的扩展”，但可能需要更动态的类型系统。
   - 一个建议是使用 **StaticString** 来定义相应的 `__mlir` 类型，但由于缺乏文档以及支持任意数量类型的难度，这被视为重大的**障碍**。
- **探索 MLIR 类型变通方案**：一位成员探索了使用 `__mlir_type` 的变通方法，但遇到了 **未记录的 MLIR** 问题，以及无法将给定类型参数的 **MLIR 类型**合成为原始字符串的问题。
   - 该成员建议，如果能在编译时提取并修改 **MLIR 类型**，或许可以使用 **UnsafePointer** 和 `init_pointee_move` 来绕过类型定义的障碍。
- **无痛从 Magic 迁移到 Pixi**：一位用户描述了他们从 `magic` 迁移到 `pixi` 的“无痛”过程，包括删除 `~/.modular` 目录并重写 `mojoproject.toml` 文件。
   - 该用户分享了一个用于更新和清理缓存的 `pix.sh` 脚本，指出它创建了一个新的 `pixi.lock` 和 `.pixi` 文件夹，并建议在测试通过后删除旧文件夹。
- **GPU Puzzle 的边缘情况**：一位用户质疑 GPU puzzle 中**主机端同步（host-side synchronization）**的必要性，引用了[特定章节](https://builds.modular.com/puzzles/puzzle_12/complete.html#host-side-synchronization-the-critical-step)，并建议如果 `DeviceContext` 使用 CUDA stream，同步可能是自动的。
   - 确认了 `DeviceContext` 确实使用了 CUDA stream，并且 puzzle 的描述将被调整，以反映在这种情况下不需要**显式同步**。
- **通过 C ABI 导出 Mojo**：一位用户询问关于从 **C/C++** 调用 **Mojo** 的问题。
   - 另一位用户澄清说，Mojo 可以通过 `@export(ABI="C")` 导出 **C ABI 兼容函数**，从而允许创建对象文件或共享库。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1382958999057600544)** (18 条消息🔥): 

> `MCP server 使用情况追踪, 用于 MCP 监控的 Service workers, GitHub MCP server, Taskerio agent 进度收件箱, MCP inspector 问题` 


- **追踪 MCP Server 使用情况：Mixpanel 和 PostHog 仍被推荐吗？**：成员们讨论了使用 **Mixpanel** 和 **PostHog** 等标准监控/分析工具来追踪 MCP server 的使用情况，特别是在 API 和 Web 应用场景下。
- **Service Workers：MCP 的“前端中的后端”**：一位成员建议利用 service workers 在后台监控来自服务器的传入通信，即使在应用程序空闲时也是如此，从而充当“前端中的后端”。
- **GitHub 发布远程 MCP Server 以获取实时上下文访问**：GitHub PM 宣布发布远程 GitHub MCP server，使任何 MCP host 无需本地设置即可访问实时 GitHub 上下文，详情见 [Reddit](https://www.reddit.com/r/mcp/s/Cj2zjute95)。
- **Taskerio 为 Coding Agent 进度追踪推出收件箱**：Taskerio 发布了一款处于隐身模式的产品，这是一个用于 coding agents 报告进度的收件箱，提供 webhooks、推送通知和用于实时仪表板的 API，详情见 [Reddit](https://www.reddit.com/r/mcp/comments/1lac12i/an_mcp_to_track_the_progress_of_your_ai_agents/)。
- **动态工具选择：GitHub 的可扩展 MCP 方法**：GitHub server 采用动态工具选择，根据用户输入或上下文过滤和限定工具范围，即使在有 30 多个可用工具的情况下，也能向 LLM 呈现相关的子集。
   - 其目标是通过一个包含**所有 API** 的 **MCP server** 来保持身份验证的简洁。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1382878164606255115)** (5 条消息): 

> `结合 Postman 的 MCP Server, 用于 MCP 安全的 SchemaPin` 


- **现在可以使用 Postman 构建 MCP Servers**：一位成员展示了如何使用 Postman 新的 MCP 构建器及其公共 API 网络上的 API 来构建 **MCP server**，并以 [fastfs-mcp GitHub 仓库](https://github.com/aj-geddes/fastfs-mcp) 为例。
   - 他们还分享了一个演示该过程的 [YouTube 视频](https://youtu.be/YzT9QU-Kfh4?si=tqqgBXXu9ct2aMUH)。
- **SchemaPin 防范 MCP Rug Pulls**：一位成员介绍了 **SchemaPin**，这是一款旨在防止 **MCP Rug Pulls** 和类似攻击的工具，[GitHub 仓库在此](https://github.com/ThirdKeyAI/SchemaPin)。
   - 该成员指向 [SchemaPin.org](https://schemapin.org) 以获取简单的实现方法。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1382809570333691975)** (4 条消息): 

> `LlamaCloud 稳定性, MistralAI Magistral 支持, LlamaParse 预设, Data + AI Summit 2025` 


- **LlamaCloud 在运行时间波动后正在恢复**：**LlamaCloud** 在经历上游基础设施提供商的一些不稳定后已恢复在线，状态更新可在 [官方状态页面](https://t.co/IdecAksHiG) 查看。
- **LlamaIndex 增加对 MistralAI Magistral 的支持**：**LlamaIndex** 现在支持 **MistralAI** 的 **Magistral** 推理模型，该模型可以集成到任何 LlamaIndex agent 工作流中，正如 [Twitter](https://t.co/ZsUEWMrnT4) 上宣布的那样。
- **LlamaParse 首次推出用户友好的 Presets（预设）**：**LlamaParse** 现在具备 **Presets** 功能，即针对不同用例优化设置的预配置模式，用户可以在 **Fast**、**Balanced** 和 **Premium** 模式之间进行选择，以平衡文档解析的准确性和速度。
- **Data + AI Summit 2025 亮点**：**Data + AI Summit 2025** 圆满结束，提供了大量关于新兴的 agentic 文档工作流格局的内容，[在此查看](https://t.co/jS2Nfwxxb3)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1382799356704591902)** (11 messages🔥): 

> `LlamaIndex and Mem0 integration, Cloudflare issues, Google Cloud Server problems, Mem0 graphRAG capabilities, Luma calendar for office hours` 


- ****Mem0** 内存集成自动处理更新**：在将 **LlamaIndex** 与 **Mem0** 结合使用时，将 `memory=memory` 传入 `agent.run(query, memory=memory)` 会自动处理内存更新，无需手动使用 `mem0_memory_class.add(interaction, thread_id_or_collection_name)`。
- **考虑将 **Luma** 日历用于 Office Hours**：由于反馈 Discord 日历在 Office Hours 方面的易用性问题，正在考虑切换到 **Luma** 日历。
   - 组织者正在就未来 Office Hours 的形式[征集想法、需求和建议](https://discord.com/channels/1031248924924043295/1031248926475255868)。
- ****Mem0** 的 graphRAG 应该可以与 **LlamaIndex** 正常工作**：如果使用了 mem0 集成包，与 **LlamaIndex** 的集成应该支持 Mem0 的 graphRAG 能力。
- **Cloudflare 和 Google Cloud Servers 出现问题**：用户报告了 [Cloudflare](https://www.cloudflarestatus.com/) 以及 [Google Cloud Servers](https://status.cloud.google.com/) 的问题。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1382924850959749130)** (10 messages🔥): 

> `Xarray-JAX library, AI SaaS tool in the finance space, Cohere documentation typo` 


- **命名张量（Named Tensors）进入深度学习领域**：一名成员正在为 Google DeepMind 开发 **Xarray-JAX 库**，作为 GSoC 2025 的一部分，据称这是深度学习框架中第一个有效的命名张量实现。
- **旨在变革金融业的 AI SaaS 工具**：一名成员正在开发一个**金融领域的 AI SaaS 工具**作为大学项目，并询问如何避免仅仅做一个 LLM 套壳，从而提供真正的价值。
   - 他们征求了关于 MVP 的建议，并确定了金融领域中可以用 AI 解决的真实痛点。
- ****Cohere** 文档中发现拼写错误**：一名成员认为 [Cohere 文档](https://docs.cohere.com/docs/amazon-sagemaker-setup-guide) 中存在**拼写错误**。
   - 在 Python 代码中，应该是 `co = cohere.SagemakerClient()`，其中 "m" 不应大写。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1382815899920109780)** (1 messages): 

> `Reranking Profile Details` 


- **Reranking 配置规范**：一名成员请求了关于 Reranking 配置的详细信息，特别是**文档数量、每个文档的 Token 数以及查询 Token 数**。
   - 被提及的成员没有回应，因此无法提供更多细节。
- **关于 Reranking 配置没有进一步讨论**：没有后续消息，因此无法总结更多讨论。
   - 对话在最初的问题后就结束了。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1382973858025308195)** (3 messages): 

> `Full-Stack Development, AIOps, Agent AI, Python engineering, low-code/no-code agent frameworks` 


- **全栈专家加入**：一位拥有 **9 年以上**经验的高级全栈开发人员及 **AIOps/Agent AI 专家**介绍了自己。
   - 他负责架构和交付强大的 AI 数字化系统，从可扩展的全栈应用到 Agentic AI 工作流和自动化流水线。
- **年度新人报到**：一位名为 Nabeel 的新成员介绍了自己。
   - 他说自己*可以被称为年度新人！*


  

---


### **Cohere ▷ #[🧭-status-feed](https://discord.com/channels/954421988141711382/1346652044181897307/1382824254009245916)** (1 messages): 

> `GCP Outage, Cohere Status Page` 


- **GCP 故障阻碍增长**：Cohere 报告称，Google Cloud Platform (**GCP**) 故障在 **2025 年 6 月 12 日** **12:02PM** 影响了其部分服务 [状态页面链接](https://ift.tt/on1ARP0)。
   - 状态页面显示，经 Cohere 团队监测，**Infrastructure** 组件出现性能下降 [Cohere 状态页面](https://ift.tt/Ens6bma)。
- **Cohere 监控异常**：Cohere 团队正在积极监控情况 [状态页面](https://ift.tt/Ens6bma)，以解决影响其 **Infrastructure** 组件的性能下降问题。
   - 该故障发生于 **2025 年 6 月 12 日** **12:02PM**，已促使团队进行密切观察并努力减轻对服务的影响。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1382821330738745365)** (7 条消息): 

> `Fast weights continual learning, O1-Pro models, Gemini Model` 


- **Fast Weights Continual Learning**：一位成员提倡使用 **Fast Weights Continual Learning** 和 **外部数据存储** 来提高用户控制力，并减少 AI 模型中不理想的人类特征。
   - 他们表达了希望看到诸如 *诡计 (scheming)、挫败感 (frustration) 和虚假记忆 (false memories)* 等特征从主流 AI 中移除的愿望。
- **O1-Pro 模型提供高价值**：一位成员发现 **O1-Pro/O3/O4-mini-high** 模型在学习文档齐全的数学和计算机科学方面非常有价值，同时也喜欢它们的 **图像生成能力**。
   - 他们还提到使用这些模型的 API 构建了一个几乎完美运行的 **音频转录流水线 (audio transcription pipeline)**，尽管图像生成功能受到了审查。
- **Gemini 与 Claude 的体验对比**：一位成员询问 **Gemini** 与 **Claude** 相比如何。
   - 另一位成员表示 **Claude** 对他们来说不太可靠，但指出所有模型都可能出错，并且在高度可验证的领域最为有用。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1382819106327232673)** (3 条消息): 

> `Wavefunction schedule` 


- **Wavefunction 讨论周五休息**：由于观众参与度有限，周五通常 **没有 Wavefunction 讨论**。
   - 尽管缺乏预定的讨论，社区成员仍欢迎自行发起讨论。
- **Wavefunction 频率**：由于观众参与度的原因，Wavefunction 讨论通常安排在除周五以外的工作日。
   - 该排期试图在活动高峰期最大化参与度，反映了讨论中质量重于数量的偏好。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1383182345691070635)** (2 条消息): 

> `Nvidia, Jensen Huang, Anthropic, Dario Amodei, AI Jobs` 


- **Huang 在 AI Jobs 问题上与 Amodei 持不同意见**：一篇 [Fortune 文章](https://fortune.com/2025/06/11/nvidia-jensen-huang-disagrees-anthropic-ceo-dario-amodei-ai-jobs/) 报道称，**Jensen Huang (Nvidia)** 在 **AI Jobs** 的未来问题上与 **Dario Amodei (Anthropic)** 持不同意见。
   - 一位成员推测他们是否在 *尝试抄底 (buy the dip)*。
- **Dario 回应 Huang**：CEO **Dario** 已通过 [X](https://www.x.com/dario) 回应了 **Jensen** —— 并更新了关于 AI Jobs 的观点。
   - 随着 **对就业的担忧持续 (job fears continue)**，两家公司的股价均大幅下跌。


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1382925426325852281)** (10 条消息🔥): 

> `Mistral 3.1 Small, Tokenizer, Magistral, Multi-modality` 


- **Mistral 3.1 Small 架构创新仍不明确**：一位成员询问了 **Mistral 3.1 Small** 的架构创新，以评估 Fine-tuning 的实现复杂度，预计可能需要 **2 周** 的时间。
   - 另一位成员建议 **Multi-modality** 支持可能比较棘手，但并非首创，并指出支持 **Mistral 3.0** 就意味着支持 **Magistral**。
- **Tokenizer 困扰**：讨论强调 **Tokenizer** 是一个 *复杂的过程*。
   - 然而，一位成员澄清说，他们在提到 Tokenizer 复杂度时实际上想到的是 **Magistral**。
- **渴望 Torchtune 链接**：成员们表达了希望在 **Magistral** 的 Hugging Face (HF) 页面上看到 **Torchtune** 链接的愿望。
   - 这表明社区有兴趣将 **Torchtune** 与 **Magistral** 集成，以增强可访问性和可用性。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1382798607366750341)** (3 条消息): 

> `Infinite Chat, Local Context Window, Ignore Feature` 


- **本地实现 Infinite Chat**：一位成员介绍了 **Infinite Chat**，它在本地实现并允许用户永远不会耗尽 **Context Window**，链接见 [此处](https://docs.supermemory.ai/infinite-chat)。
- **请求 Ignore 功能**：一位成员询问了关于 **'ignore' 功能**（类似于 git 的 .ignore 文件），用于告知 Embedding 系统不要使用某些文件、文件类型或目录。


  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1382853343185207366)** (2 条消息): 

> `Windsurf Wave 10, EU Cluster, Claude Sonnet 4` 


- **Windsurf Waves 迎来 UI/UX 升级**：Windsurf 正在完成 **Wave 10**，带来了一系列全新的 **UI/UX 升级**以及新的团队和企业版方案，包括用于 `@-mentions` 和文件引用的 [新图标](https://windsurf.com/blog/windsurf-wave-10-ux-enterprise)。
   - Cascade 面板中的代码块现在与你的 IDE 主题匹配，Cascade 面板中的原生终端现在支持用户输入，并新增了对话历史记录 UI。
- **Windsurf 推出 EU Cluster 以提升性能**：Windsurf 隆重宣布推出 **EU Cluster**，为欧洲企业带来更快的性能并满足日益增长的需求！
   - 观看 [Youtube 视频](https://youtu.be/UHinqQiiCI8?si=udyZDkWGg9nq7zcI) 并 [加入 r/Windsurf 的讨论](https://www.reddit.com/r/windsurf/)。
- **Claude Sonnet 4 点亮 Windsurf**：**Claude Sonnet 4** 和 **Claude Sonnet 4** (Thinking) 现在已通过 [API Pricing](https://docs.windsurf.com/windsurf/models#api-pricing) 向所有付费计划开放！
   - 更多信息请见 [X 平台](https://x.com/_mohansolo/status/1933605162775687482)。