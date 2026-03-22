---
companies:
- cursor
- kimi
- fireworks
- anthropic
- langchain
date: '2026-03-20T05:44:39.731046Z'
description: '基于 **Kimi K2.5** 构建的 **Cursor Composer 2** 引发了关于模型归属和许可协议的讨论，凸显了行业正转向基于开源模型、通过领域特定微调（fine-tuning）和强化学习开发的后训练衍生模型。


  与此同时，**Claude Code** 正在向 **T3 Code** 等第三方工具以及 Telegram 和 Discord 等通讯渠道扩展。**LangChain**
  也在从流程编排演进为多智能体产品，推出了 **Deep Agents/Open SWE** 和 **LangSmith Fleet** 等方案。目前的行业讨论强调了明确基础模型归属、遵守许可合规，以及通过微调和用户体验（UX）实现产品差异化的重要性。'
id: MjAyNS0x
models:
- kimi-k2.5
- claude-code
people:
- clementdelangue
- leerob
- amanrsanger
- yuchenj_uw
- kimmonismus
title: 今天没什么事。
topics:
- model-attribution
- fine-tuning
- reinforcement-learning
- open-source
- agent-products
- model-licensing
- software-integration
- product-differentiation
---

**平静的一天。**

> 2026年3月19日至3月20日的 AI 新闻。我们检查了 12 个 Subreddit、[544 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 且没有新增 Discord 信息。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率！

---

# AI Twitter 回顾

**Coding Agents、模型归属以及 Cursor/Kimi Composer 2 的争议**

- **Cursor 的 Composer 2 基于 Kimi K2.5 构建，归属声明的缺失成为了热门话题**：当天最重要的工程/产品讨论集中在 Cursor 的新编程模型上。最初的猜测通过 tokenizer/模型 URL 信号将 Composer 2 与 Kimi 联系起来，批评者质疑为什么没有预先披露基础模型，以及是否遵守了许可条款 ([@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2035012260008272007), [@eliebakouch](https://x.com/eliebakouch/status/2035041428535939535), [@ClementDelangue](https://x.com/ClementDelangue/status/2035042945884463538))。Cursor 随后澄清说 Composer 2 始于 **Kimi K2.5**，最终模型中只有约 **1/4 的计算量**来自基础模型，其余则来自持续预训练（continued pretraining）和高算力 RL，且使用情况已通过 **Fireworks 托管的商业合作伙伴条款**覆盖 ([@leerob](https://x.com/leerob/status/2035035355364081694), [@leerob](https://x.com/leerob/status/2035050444347600936), [@amanrsanger](https://x.com/amanrsanger/status/2035079293257359663))。Kimi 随后公开确认了这一合作伙伴关系，并将 Cursor 的工作视为开放模型生态系统的一个案例：K2.5 提供了基础，Cursor 增加了持续预训练和 RL，而 Fireworks 提供了托管的 RL/推理基础设施 ([Kimi Moonshot](https://x.com/Kimi_Moonshot/status/2035074972943831491))。
- **为什么这在技术和战略上很重要**：这一事件凸显了行业的一条分界线：高性能产品正日益成为**强大 OSS 基础模型的后训练衍生品**（尤其是中国开源权重模型），而不是从零开始的预训练成果。一些从业者认为，这正是“基础模型（foundation models）”的用途——前提是清晰地处理好归属和许可义务 ([code_star](https://x.com/code_star/status/2035108747119665535), [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2035040667492393230), [@Teknium](https://x.com/Teknium/status/2035099247528615993))。另一些人则呼吁针对命名基础模型、在 Evals 中与基础模型进行对比以及给予 OSS 实验室明确致谢建立更强的规范 ([@kimmonismus](https://x.com/kimmonismus/status/2035066695564328975), [@cloneofsimo](https://x.com/cloneofsimo/status/2035058598725001553))。最终结果与其说是丑闻，不如说是一个信号：**产品差异化正在向特定领域的 CPT/RL、Evals 和 UX 转移**，而基础模型的来源在战略上虽然敏感，但披露已变得日益重要。

**开源编程工具：Claude Code、T3 Code、Deep Agents、Fleet 和 Hermes**

- **Claude Code 的生态系统正在向第三方产品和渠道扩展**：Theo 在 **T3 Code 中发布了 Claude 集成**，实际上允许安装了本地 Claude Code CLI 的用户在 T3 Code 内部使用它，他随即开玩笑说可能存在法律风险和社区附注（[Theo](https://x.com/theo/status/2034831968463200359), [Theo](https://x.com/theo/status/2034879231633871185), [Theo](https://x.com/theo/status/2034869281914310912)）。另外，Anthropic 似乎正在将 Claude Code 从终端扩展到 Telegram 和 Discord 等 **Channels**（[kimmonismus](https://x.com/kimmonismus/status/2034934937388806528)），同时开源维护者描述了 Claude 支持计划在 Diffusers 集成、性能分析和硬件感知流水线优化等任务中带来的显著 OSS 生产力提升（[RisingSayak](https://x.com/RisingSayak/status/2034873534774927501)）。
- **LangChain 正在从编排向 Agent 产品转型**：多条推文强调了 **Deep Agents/Open SWE** 作为 Claude Code 的开源替代方案，以及 **LangSmith Fleet** 作为多 Agent/劳动力风格的产品层（[KSimback](https://x.com/KSimback/status/2034793268886601986), [BraceSproul](https://x.com/BraceSproul/status/2034804646313959488), [EvanRimer](https://x.com/EvanRimer/status/2035008854845514192), [hwchase17](https://x.com/hwchase17/status/2035035096806510872)）。LangChain 还发布了面向生产的内容：**Building Reliable Agents** 课程、LangSmith Prompt Hub 中的 **仅限所有者 Prompt 推广**、`@langchain/react` 中的 React Suspense 支持，以及更多关于生产环境中非确定性 Agent 可观测性的消息（[LangChain](https://x.com/LangChain/status/2035039198575476799), [LangChain](https://x.com/LangChain/status/2035058068460106213), [LangChain](https://x.com/LangChain/status/2035068122000969980), [LangChain_JS](https://x.com/LangChain_JS/status/2035022974588526620)）。
- **Hermes/OpenClaw/local-agent 工作流继续快速成熟**：HermesWorkspace v0.2.0 增加了单命令启动、基于 UI 的 Provider/Model 配置、实时模型目录以及新的 Config/Model 端点（[outsource_](https://x.com/outsource_/status/2034788187944431914)）。Hermes 还获得了 **并行网页搜索/页面提取**、**工作流记录/回放系统**，以及通过 **Camel Guard v0.4** 实现的更强 Prompt 注入防护（[p0](https://x.com/p0/status/2034948093980926167), [0xbyt4](https://x.com/0xbyt4/status/2035112284482343059), [WeXBT](https://x.com/WeXBT/status/2034982164794576927)）。社区对 Hermes 和 OpenClaw 的对比强调了 Hermes 紧凑且重检索的内存设计，而 OpenClaw 则拥有更大的回放历史记录，这对于交互式使用的延迟具有具体影响（[witcheer](https://x.com/witcheer/status/2035024543526359134)）。这些帖子中反复出现的一个主题是：**Agent UX 变得不再仅仅关乎单次模型 IQ，而更多关乎内存架构、工具可靠性和循环延迟（loop latency）**。

**模型发布与基准测试：Nemotron-Cascade 2, Mistral Small 4, V-JEPA 2.1, MiMo 以及设计/编程排名**

- **NVIDIA 的 Nemotron-Cascade 2 是最引人注目的开源模型发布**：NVIDIA 发布了 **Nemotron-Cascade 2**，这是一个拥有 **30B MoE（3B 激活参数）** 的开源模型，定位为高密度推理/Agentic 模型。其目标雄心勃勃：在 **IMO 2025、IOI 2025 和 ICPC World Finals 2025 中达到金牌水平**，拥有同类最佳的数学/代码/对齐/指令遵循性能，并优于近期的 **Qwen3.5-35B-A3B** 和 **Qwen3.5-122B-A10B** 变体。该模型由 **Cascade RL 加上多领域 on-policy 蒸馏**提供支持（[\_weiping](https://x.com/_weiping/status/2034877099908243746), [HuggingPapers](https://x.com/HuggingPapers/status/2034876841475838329), [ollama](https://x.com/ollama/status/2035088633225781389)）。这次发布不仅因为权重而备受关注，更因为它使紧凑且具有高激活效率的推理模型成为了一流的 OSS 选择。
- **Mistral Small 4 增加了混合推理 + 多模态能力，但在智能水平上落后于同行**：Artificial Analysis 将 **Mistral Small 4** 总结为具有 **119B MoE（6.5B 激活参数）** 的模型，采用 Apache 2.0 许可，支持推理和非推理模式以及图像输入。它在 AA 智能指数的推理模式下得分为 **27**，高于之前的 Mistral small 模型并与 Magistral Medium 1.2 持平，但仍落后于 **gpt-oss-120B (33)**、**Nemotron 3 Super 120B A12B (36)** 和 **Qwen3.5 122B A10B (42)** 等同行。不过，与其中一些同行相比，它在 Token 效率上相对较高，且幻觉更少（[Artificial Analysis](https://x.com/ArtificialAnlys/status/2034960206736892365)）。
- **V-JEPA 2.1 是一个重要的视觉 SSL 更新，特别是对于密集理解**：FAIR 的新 **V-JEPA 2.1** 从仅掩码监督转向同时学习 **被掩码和可见的 Token**，增加了 **跨中间层的深度自监督**，并在共享编码器下使用 **特定模态的分词器（tokenizers）**（[TheTuringPost](https://x.com/TheTuringPost/status/2034795966931640533), [murloren](https://x.com/murloren/status/2034920039065714742), [massiviola01](https://x.com/massiviola01/status/2034885267580997982)）。报告的提升包括在零样本现实世界操作中，机器人抓取成功率比 V-JEPA 2 提高了 **20%**，并在 Ego4D 和 EPIC-KITCHENS 密集预测任务中刷新了 SOTA 记录（[TheTuringPost](https://x.com/TheTuringPost/status/2034796099110936741)）。
- **其他基准测试动态**：DesignArena 报告称 **Anthropic Opus 4.6** 现在在广泛的设计导向型编码任务中领先——包括 Web、移动端、3D 设计、游戏开发和数据可视化——而 **Gemini 3.1** 夺得了 SVG 设计冠军（[Designarena](https://x.com/Designarena/status/2034788729068691787)）。小米的 **MiMo V2 Pro/Omni** 出现在 Arena 排名和独立评论中，被视为一个严肃但表现不均衡的竞争者——在指令遵循和某些长任务处理方面表现良好，但在编码一致性和抗幻觉能力方面较弱（[arena](https://x.com/arena/status/2035068569063690289), [ZhihuFrontier](https://x.com/ZhihuFrontier/status/2034951742526521660)）。阿里巴巴也透露了 **Qwen 3.5 Max Preview** 的排名：**数学第 3**，Arena 专家榜前 10，总榜前 15（[AlibabaGroup](https://x.com/AlibabaGroup/status/2034824822052856136)）。

**训练、RL、检索与系统效率**

- **针对性数据的 Pretraining 持续展现出优于 Finetuning 的优势**：来自 Stanford/Marin 生态系统的一项显著研究指出，通过**合成“megadocs”实现高效数据 Pretraining**，报告了约 **1.8 倍的数据效率**提升，并强调在 Pretraining 期间混合小型领域数据集比重复 Finetuning 或 Replay 能更好地抵抗过拟合 ([konwookim](https://x.com/konwookim/status/2035029597491011984), [percyliang](https://x.com/percyliang/status/2035112178580398341), [leavittron](https://x.com/leavittron/status/2035049864862867691))。这与更广泛的讨论相一致，即“Midtraining = RL Prior”，且模型适配正成为一项关键的应用能力 ([cooperleong22](https://x.com/cooperleong22/status/2035031300810449289), [code_star](https://x.com/code_star/status/2035055796468555895))。
- **RL 正在从数学/聊天领域向检索和预测领域多样化发展**：CMU/Meta 推出了一种**代码搜索模型的 RL 方案**，仅使用 bash 终端作为探索接口，在避免使用特殊工具的同时仍取得了强劲的效果 ([gneubig](https://x.com/gneubig/status/2035037624105410926))。Tinker 和 Mantic 报告称，在 **gpt-oss-120b** 上进行判断性预测（judgmental forecasting）的 RL 表现优于前沿模型在事件预测上的表现，推动了“自动化超级预测（automated superforecasting）”的发展 ([tinkerapi](https://x.com/tinkerapi/status/2035038766067499496), [johnschulman2](https://x.com/johnschulman2/status/2035063683416813820))。
- **Infra 和 Kernel 仍然是瓶颈——也是护城河**：关于 Kernel 工程的高参与度帖子强调，编写定制化 Kernel 现在可能是系统方向工程师投资回报率（ROI）最高的技能之一 ([jxmnop](https://x.com/jxmnop/status/2034809761800364292), [clattner_llvm](https://x.com/clattner_llvm/status/2034838928818397523))。ThunderKittens 被引用为从研究到生产推理迁移的典范，在 Coding Agents 的 50 多个工具调用中，每次生成节省的几百毫秒会产生实质性的累积效应 ([boyuan_chen](https://x.com/boyuan_chen/status/2034818086147121246))。在推理服务方面，**vLLM** 被称为事实上的标准，RunPod 生产数据集中约**一半的纯文本端点**都在运行 vLLM 变体 ([vllm_project](https://x.com/vllm_project/status/2034828731983044971))。

**应用工具、产品发布与 Agent 基础设施**

- **文档解析作为 Agent 原语正趋于商品化**：LlamaIndex 推出了 **LiteParse**，这是一个免费的本地解析器，可以通过单行命令 `npx skills add ... --skill liteparse` 接入 **40–46 个以上的 Agent**；它被定位为既是任务解决工具，也是为 Coding Agents 提供文档上下文的一种方式 ([jerryjliu0](https://x.com/jerryjliu0/status/2034790590572060848), [llama_index](https://x.com/llama_index/status/2035024635738431986), [Saboo_Shubham_](https://x.com/Saboo_Shubham_/status/2035051817080643726))。LlamaParse 还发布了一个官方 Agent Skill，用于处理跨格式、表格、图表和图像的更复杂文档理解 ([llama_index](https://x.com/llama_index/status/2035069934372626812))。
- **本地/离线深度研究和本地 Agent 栈正变得日益可靠**：多篇帖子重点介绍了 **Local Deep Researcher**，这是一个基于 MIT 协议的本地研究闭环，它能够编写自己的搜索查询、爬取数据、识别空白点，并使用兼容 Ollama 的模型迭代生成带有引用的 Markdown 报告 ([ihtesham2005](https://x.com/ihtesham2005/status/2035009684386771306), [RoundtableSpace](https://x.com/RoundtableSpace/status/2035072633172074697))。社区演示还展示了在 Apple Silicon 和旧款 GPU 上使用 Hermes/OpenClaw、Qwen、Nemotron、Ollama 和混合运行器的组合构建的全本地 Agent 栈 ([agenticmate](https://x.com/agenticmate/status/2034851442666926382), [elldeeone](https://x.com/elldeeone/status/2034885539045024124))。
- **Perplexity、Devin 和企业级 Agent 控制面持续扩展**：Perplexity Computer 增加了对 **Pitchbook、Statista 和 CB Insights** 数据访问权限，进一步深入分析师/VC 工作流 ([AravSrinivas](https://x.com/AravSrinivas/status/2035043246376984663))。Devin 增加了**自调度定期任务**功能，将一次性会话转变为周期性工作流 ([cognition](https://x.com/cognition/status/2035041799245279498))。Okta 的“AI Agents 作为受管非人类身份”蓝图以及 Factory 的企业设置层级结构，都指向了生产环境中 Agent 治理的清晰模式 ([dl_weekly](https://x.com/dl_weekly/status/2035053972403220846), [FactoryAI](https://x.com/FactoryAI/status/2035118566064976373))。

**热门推文（按参与度排序）**

- **模型溯源与产品定位**：互动量最高的技术新闻是 Kimi 确认 **Cursor Composer 2 通过授权的 Fireworks 合作伙伴关系使用 Kimi-k2.5**，这有效地为之前的归属权之争画上了句号 ([Kimi Moonshot](https://x.com/Kimi_Moonshot/status/2035074972943831491))。
- **开放编程产品发布**：Theo 发布的 **T3 Code 现已支持 Claude** 引起了广泛关注，并成为围绕编程 Agent 集成的平台/TOS（服务条款）问题的焦点 ([Theo](https://x.com/theo/status/2034831968463200359))。
- **面向学生和开发者的 Agent 工具**：OpenAI 推出了 **Codex for Students**，为美国和加拿大的大学生提供 **100 美元的额度** ([OpenAIDevs](https://x.com/OpenAIDevs/status/2035033703274201109))。
- **具有广泛系统影响的研究发布**：NVIDIA 的 **Nemotron-Cascade 2** 发布在众多模型中脱颖而出，它结合了强大的推理能力声明和异常精简的激活参数量 ([\_weiping](https://x.com/_weiping/status/2034877099908243746))。
- **视觉自监督**：FAIR 的 **V-JEPA 2.1** 引起了强烈关注，它被认为是一种更密集、更具扩展性的基于视频的视觉理解路径 ([TheTuringPost](https://x.com/TheTuringPost/status/2034795966931640533))。


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. 经典硬件上的本地 AI

  - **[在 2002 年的 PowerBook G4 上本地运行 TinyLlama 1.1B。Mac OS 9 系统，无网络连接，通过 CD 安装。](https://www.reddit.com/r/LocalLLaMA/comments/1ryu7rr/running_tinyllama_11b_locally_on_a_powerbook_g4/)** (热度: 282)：**图片展示了一台运行 Mac OS 9 的 2002 年 PowerBook G4，使用了“MacinAI Local”软件界面，标志着复古计算领域的一项重大成就。该项目与以往的复古 AI 尝试不同，它引入了一个专门为经典 Macintosh 硬件设计的定制 C89 推理引擎，支持 GPT-2 和 TinyLlama 等多种模型。它包含一个在 Macintosh 特定文本上训练的 100M 参数 Transformer，并通过 AltiVec SIMD 优化实现了 7.3 倍的加速，同时包含磁盘分页功能，以便在有限的 RAM 上处理大型模型。该设置完全离线，通过 CD-R 光盘安装，展示了复古硬件与现代 AI 能力的独特结合。** 评论者对该项目的创新性和技术成就表示钦佩，其中一位提到了 TinyLlama 模型推理时间的幽默之处，另一位则对将其与 Hypercard stack XCMD 集成感到兴奋。

    - 在 PowerBook G4 上实现 TinyLlama 的显著特点是使用了 AltiVec 优化，实现了 `7.3x 的加速`。考虑到 2002 年代机器的硬件限制，这种优化至关重要。此外，使用磁盘分页系统来管理超过可用 RAM 的层是一个聪明的权变方案，使得模型在资源受限的情况下仍能有效运行。
    - 该项目通过利用 Agent 式的 AppleScript 控制来增强其实用性，展示了在老旧硬件上运行语言模型的潜力。这种方法不仅证明了在旧系统上运行 LLMs 的可行性，还通过为复古机器提供 AI 的实际应用，为复古计算社区做出了贡献。
    - 讨论强调了在 G4 上运行任何 LLM 的惊人本质，强调了所克服的技术挑战，如针对 AltiVec 的优化以及通过磁盘分页管理内存限制。这些解决方案使该项目成为复古计算和 AI 集成领域的一项重大成就。

### 2. Qwen 模型性能与优化

  - **[Qwen3.5 是一只实干派。](https://www.reddit.com/r/LocalLLaMA/comments/1ryljps/qwen35_is_a_working_dog/)** (热度: 623): 该帖子讨论了 Qwen3.5 模型，强调其需要大量的 context 才能有效运作，特别指出 `27B` 模型至少需要 `3K` tokens 才有实用价值。作者认为这些模型被设计为“Agent 优先”，意味着在给出明确目标和环境上下文（而非极简 Prompt）时表现更好。帖子还批评 `35B MoE` 模型表现不佳。Qwen 模型被描述为“实干派（working dogs）”，在执行特定任务时表现出色，这符合 **Alibaba** 对开源权重模型的设计初衷。评论者普遍同意 Qwen 模型需要明确的指令，其中一位指出 `27B` 模型需要清晰的指令以避免不必要的动作。另一位评论者分享了使用 `122B` 模型的积极经验，指出其在 `600tk` 的 system prompt 限制下表现出色，这表明它受益于高级的开放世界工具环境。

    - 用户 'abnormal_human' 讨论了他们使用 122B 模型的经验，强调了对 system prompt 进行严格的 600 token 限制的有效性。他们指出，这种方法通过专注于提示行为而非模式匹配（pattern matching）来增强模型性能，将其类比为 Claude 代码环境。据称这种方法可以防止过度思考并提高任务执行效率。
    - 用户 'zasad84' 分享了实验各种模型（包括 35b-a3b, 27b 和 9b）的见解。他们强调了在提供大量、直接的 system prompt 时，9b 模型在特定任务中出人意料的效果。利用 Unsloth 量化（unsloth quant）使他们能够在 24GB 显存的显卡上利用完整的 context window，这对于上下文容量是限制因素的任务至关重要。他们还提到使用像 Gemini 3.1 pro 这样的 SOTA 模型来编写有效的 system prompt。
    - 'ggonavyy' 指出，在使用 27B 模型时，必须提供显式指令，以防止模型尝试所有可能的解决方案（即使是在规划模式下）。这表明如果没有明确的指令，模型可能会过度扩展其解决问题的努力，突显了通过精确的 Prompt Engineering 来有效引导模型行为的重要性。

### 3. 新模型与硬件考量

  - **[128gb M5 Max for local agentic ai?](https://www.reddit.com/r/LocalLLM/comments/1rz58qn/128gb_m5_max_for_local_agentic_ai/)** (热度: 112): **该帖子讨论了 128GB M5 Max MacBook 是否适合运行本地大语言模型 (LLMs) 和个人 AI Agents，重点在于隐私保护和避免使用云端解决方案。用户目前使用 RTX 4070 配 16GB RAM，但发现其在运行本地模型时存在限制。他们正在考虑 M5 Max，因其能够通过 Q4/Q5 量化（Quantization）处理像 `gpt-oss-120b` 和 `nemotron-3-super-120b-a12b` 这样的大型模型，从而实现高效的本地处理，无需依赖 Claude 或 ChatGPT 等外部 API。MacBook 的灵活性和性能（即使在运行大型模型时）得到了强调，尽管在重负载下可能会产生明显的风扇噪音。** 评论者普遍认为 **128GB M5 Max MacBook** 能够高效运行大型模型，其中一位指出经过一些调整后它可以处理 `qwen 3.5 390b`，达到约 `12 tokens/second`。与专用 GPU 设置相比，MacBook 的灵活性受到了称赞，尽管有些人因其高昂的价格而犹豫不决。

    - JuliaMakesIt 强调了 M5 Max 128GB MacBook Pro 能够轻松处理大型 AI 模型（如 `gpt-oss-120b`、`nemotron-3-super-120b-a12b` 和 `qwen3.5-122b-a10b`），尤其是在使用 Q4/Q5 量化时。这种配置提供了强大的本地处理能力，减少了对 Claude 或 ChatGPT 等外部 API 的依赖，并提供了超越专用 GPU 阵列的灵活性。
    - Consistent-Cold4505 提到 M5 Max 在经过一些优化后可以高效运行 `qwen 3.5 390b`，达到大约 `12 tokens per second`。这种性能归功于 Apple 的设计，它特别适合此类任务。文中还将其与 48GB Mac Mini M3 Max 进行了比较，后者在 4-bit 精度下达到近 `5 tokens per second`，彰显了 M5 Max 更卓越的能力。
    - TimLikesAI 将 M4 Max 128GB 与 M5 Max 进行了对比，指出虽然 M4 的 Prefill 速度较慢，但它仍然支持有效运行大型模型。这表明即使是上一代硬件也具有相当的能力，尽管 M5 Max 为苛刻的 AI 工作负载提供了更好的性能。

  - **[Ooh, new drama just dropped 👀](https://www.reddit.com/r/LocalLLaMA/comments/1ryv7rg/ooh_new_drama_just_dropped/)** (热度: 1482): **这张图片是一个幽默的梗图，描绘了关于 Cursor 的新模型 Composer 2 的争议，据称该模型是在没有妥善署名（Attribution）的情况下基于 Kimi K2.5 构建的。这引发了关于许可和署名的讨论，特别是针对 Kimi K2.5 使用的修改版 MIT License，该协议要求如果软件被用于具有显著用户量或收入指标的商业产品，则必须进行署名。梗图暗示了这一潜在问题的“揭露”，呼应了标题中提到的新争议。[查看图片](https://i.redd.it/o5jfb7y747qg1.jpeg)。** 一些评论者认为，Cursor 的做法是那些迅速利用市场空白的公司的典型行为，但由于依赖现有产品和 API 成本，他们面临着局限性。其他人则批评了创建“套壳（Wrapper）”产品的趋势，这些产品被认为缺乏实质内容且由炒作驱动。

    - _wOvAN_ 讨论了月之暗面（Moonshot AI）使用的修改版 MIT License，其中包括一个独特的条款，要求月活跃用户超过 1 亿或月收入超过 2000 万美元的公司在其用户界面上展示 “Kimi K2”。此修改旨在确保软件在大规模商业应用中的可见性和署名。
    - Everlier 深入分析了 Cursor 的商业模式，强调他们通过快速提供解决方案来利用市场空白，但缺乏强大的基础根基。他们依赖现有产品，这限制了他们的创新潜力。Everlier 还指出，Cursor 对 Kimi 2.5 的使用符合许可条款，这些条款允许公司进行此类改编。
    - Technical-Earth-3254 质疑 Cursor 定价策略的局限性，特别是该模型在他们的计划中是否是不限量的。这反映了对于 Cursor 如何构建其服务以及定价层级中模型使用的潜在约束的广泛不确定性。

## 非技术性 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 模型与基准测试发布

- **[Cursor 的 ‘Composer 2’ 模型据称只是经过 RL 微调的 Kimi K2.5。Moonshot AI 表示从未收到付款或给出许可](https://www.reddit.com/r/singularity/comments/1ryrs2w/cursors_composer_2_model_is_apparently_just_kimi/)** (热度: 739): **图像显示，Cursor 的 ‘Composer 2’ 模型本质上是经过额外强化学习（RL）微调的 Kimi K2.5 模型。这一发现引发了对 Cursor AI 做法的担忧，据称他们在未获得许可或支付必要费用的情况下使用了 Kimi 的 tokenizer。图像中的终端截图显示了一个调试会话，将该模型识别为 ‘kimi-k2p5-rl-0317-s515-fast’，这表明该模型的核心源自 Kimi K2.5。** 评论者对 Cursor 的未来表示怀疑，指出他们更倾向于 open models，并批评 Cursor 在使用 Kimi 模型方面缺乏透明度。此外，Cursor 在没有适当致谢或授权的情况下使用他人开发的模型的做法也遭到了批评。

    - 讨论强调了对 Cursor ‘Composer 2’ 模型透明度和原创性的担忧，据报道该模型是 open-source 模型 Kimi K2.5 的微调版本。这引发了关于专有模型真实性的质疑，以及在没有对原始创作者（如本例中的 Moonshot AI）进行适当致谢或补偿的情况下，对 open-source 模型进行品牌重塑的伦理问题。
    - 舆论对 Cursor 的商业模式提出了批评，特别是决定为本质上是现有 open-source 模型的更名版本收取每月 20 美元的费用。这种情况突显了 open models 的重要性，以及公司可能在不回馈社区或提供明确归属的情况下利用这些模型的潜力，这可能会在所提供技术进步的真实性质上误导用户。
    - 评论还涉及了许可方面，指出虽然 Cursor 在技术上可能遵守了修改后的 MIT license 条款，但在其商业产品中没有显著标明原始 Kimi K2.5 模型的伦理影响受到了质疑。这种情况体现了在商业环境中如何解释和执行 open-source 许可的更广泛问题。

  - **[Cursor 的新 Composer 2 在编程方面刚刚击败了 Claude Opus，且价格便宜 10 倍](https://www.reddit.com/r/DeepSeek/comments/1ry895w/cursors_new_composer_2_just_beat_claude_opus_at/)** (热度: 195): **Cursor** 发布了 **Composer 2**，这是一款编程模型，在 **Terminal-Bench 2.0** 上达到了 `61.7%` 的得分，超越了 **Claude Opus 4.6** 的 `58.0%`。其价格为 `每百万 token 0.50 美元`，明显低于 Opus 的 `$5.00`。虽然它落后于 **GPT-5.4** 的 `75.1%`，但它提供了 `1/5` 价格的成本优势。该模型专门针对代码进行训练，并具有 “self-summarization” 功能，可以在不丢失上下文的情况下压缩长会话。与此同时，**OpenAI** 已收购 **Astral** 以增强 Codex，表明 AI 编程模型领域的竞争正在加剧。一些评论者认为 Composer 2 可能基于 **GLM 5**，并批评 Cursor 没有从零开始开发，暗示他们是在对 **Kimi-K2.5** 或 **GLM 4.7/5** 等现有的 open-source 模型进行微调。

    - 有用户推测 Cursor Composer 2 的底层模型是 GLM 5，并指出开发者并非从头开始进行 pretraining，而是对现有的 open-source 模型进行微调。这种方法可能涉及 Kimi-K2.5 或 GLM 4.7/5 等模型，表明其依赖于成熟的架构而非原创性开发。
    - 一位用户将 Cursor 的 Composer 2 与 Claude Opus 进行了比较，指出虽然 Composer 2 宣传能像 GitHub Copilot 一样读取整个代码库，但据报道其性能较差。该用户在代码可读性和集成方面遇到了问题，需要频繁的人工干预，这与他们使用 Claude Opus 时极少需要干预的体验形成了鲜明对比。

- **[MacBook M5 Pro + Qwen3.5 = 全本地 AI 安全系统 — 93.8% 准确率，25 tok/s，无需云端 (96-Test Benchmark vs GPT-5.4)](https://www.reddit.com/r/Qwen_AI/comments/1ryoaub/macbook_m5_pro_qwen35_fully_local_ai_security/)** (Activity: 235): **该帖子讨论了 **Qwen3.5** 模型在 **Apple M5 Pro** 上作为家庭安全系统基准测试 **HomeSec-Bench** 的一部分本地运行时的性能表现。**Qwen3.5-9B** 模型实现了 `93.8%` 的准确率，仅以 `4.1 分` 之差落后于 **GPT-5.4**，同时运行速度为 `25 tokens per second`，占用 `13.8 GB` 内存。**Qwen3.5-35B MoE** 变体展示了比任何测试过的 OpenAI 云端端点（cloud endpoint）更快的首字时间（`435ms TTFT`）和更高的吞吐量（`42 tok/s`）。该基准测试在上下文预处理、事件去重和安全分类等任务上评估模型，强调了在确保数据隐私且无需 API 成本的情况下，在本地运行先进 AI 模型的可行性。完整结果和方法论可在 [GitHub](https://github.com/SharpAI/DeepCamera/tree/master/skills/analysis/home-security-benchmark) 获取。** 一位评论者建议探索 **jang_q models**，据称这些模型在 MMLU 等基准测试中优于 **mlx 4bit minimax** 等其他模型，凭借 `60GB` 的模型即可达到接近 `80%` 的准确率。另一位用户对选择使用量化版本的 **Qwen3.5-9B** 而非完整模型提出了疑问。

    - **HealthyCommunicat** 强调了 JANG_Q 模型的性能，指出 2-bit JANG_Q 等效模型（60GB）在 MMLU 上达到了接近 80% 的分数，优于得分低于 30% 的 4-bit MLX minimax（120GB）。他们强调了 JANG_Q 模型的效率，特别是 Qwen 3.5 397b 的 180GB 版本，其 MMLU 得分为 93%，并建议将其作为本地 AI 系统的更优替代方案。
    - **just one Question** 询问了为何选择量化版本 Qwen3.5-9B (Q4_K_M) 而非完整版。这个问题暗示了对模型大小与性能之间权衡的考量，即 Q4_K_M 等量化模型通常用于在保持合理准确率的同时，降低计算负载和内存占用，使其适用于资源有限的设备。
    - **pascon** 询问了 QWEN 模型与配备 24MB 统一内存（unified memory）的 Mac Mini 的兼容性。这个问题对于理解本地部署 AI 模型时的硬件要求和局限性至关重要，因为它直接影响在消费级硬件上运行大型模型的可行性。

### 2. Claude Code 与开发工具

  - **[我构建了一个 Claude 技能，可以为任何 AI 工具编写准确的 Prompt。为了停止在糟糕的 Prompt 上浪费额度。我们在 GitHub 上刚刚突破了 600 stars‼️](https://www.reddit.com/r/ClaudeAI/comments/1rxyarx/i_built_a_claude_skill_that_writes_accurate/)** (活跃度: 1339): **`prompt-master` 是一个专为优化各种 AI 工具的 Prompt 生成而设计的 **Claude skill**，目前在 GitHub 上已获得超过 `600 stars`。它能智能检测目标 AI 工具并应用特定策略，例如从用户输入中提取 `9 个维度` 并识别 `35 个常见的 Prompt 问题`，从而提高 Prompt 的准确性和效率。该工具支持包括 **Claude, ChatGPT, Midjourney, 和 Eleven Labs** 在内的广泛平台，并具备针对不同任务定制的 `12 个自动选择的 Prompt 模板`。该项目是开源的，并根据社区反馈持续改进，最近发布了最新的 `v1.4` 版本。** 评论者强调，该工具能够针对不同的 AI 模型（如 Midjourney 和 Claude Code）进行专门的 Prompt 路由，这是它区别于通用 Prompt 工具的关键点。也有用户对其与开源模型的兼容性表示关注，例如一位用户在 `5090` GPU 上通过 `comfyui` 本地运行它。

    - **dovyp** 强调了该 Claude 技能中针对特定工具路由的重要性，指出它区分了 Midjourney 和 Claude Code 等不同 AI 工具所需的结构。这种针对性至关重要，因为大多数通用 Prompt 工具无法解决每个工具的独特需求，这使得该技能特别有价值。
    - **JMdesigner** 询问了该 Claude 技能与开源模型的兼容性，并提到他们在 5090 GPU 上使用 ComfyUI 的配置。这表明用户有兴趣在专有模型之外利用该技能的能力，从而可能在多样化的 AI 环境中扩展其用途。
    - **dogazine4570** 反思了他们使用类似 Claude Code 工具的经验，指出虽然有一定帮助，但仍需要对 Prompt 进行手动调整。他们对该技能处理工具特定特性（如 Cursor 和 Claude Code 之间的差异）的能力表示感兴趣，这可能会增强其核心实用性。

  - **[我很确定我没有发挥出 Claude 的全部潜力 - 有哪些插件/连接器值得一试？](https://www.reddit.com/r/ClaudeAI/comments/1rxswkv/pretty_sure_im_not_using_claude_to_its_full/)** (活跃度: 863): **这篇 Reddit 帖子讨论了如何通过集成各种插件和连接器来优化 **Claude** 的使用。一个值得注意的建议是使用 `/insights` 命令来生成使用报告和改进建议。另一个高级配置涉及创建一个单一的 MCP 服务器，通过 Chrome 浏览器扩展路由工具调用，利用现有的 Web 应用会话（例如 Slack, Linear, Datadog, Google Sheets）来简化工作流，而无需管理独立的 API keys。这种设置促进了跨 100 多个 Web 应用的无缝任务自动化，开源实现可在 [GitHub](https://github.com/opentabs-dev/opentabs) 上找到。此外，**Superpowers 插件** 被推荐给开发者，它可以增强 Claude 的能力，并可通过官方插件列表获取。** 评论强调了将 Claude 与现有 Web 应用集成以简化工作流的倾向，强调了单一 MCP 服务器设置比管理多个 API keys 更高效。Superpowers 插件因其对生产力的提升而特别受到开发者的重视。

    - **opentabs-dev** 描述了一种将 Claude 与 Slack、Linear 和 Datadog 等各种 Web 应用集成的精简方法。他们不使用多个独立的 MCP 服务器和 API keys，而是使用一个通过 Chrome 扩展路由工具调用的单一 MCP 服务器，从而利用活跃的浏览器会话。这种设置允许 Claude 执行诸如 `slack_send_message` 或 `linear_create_issue` 之类的任务而无需管理凭据，通过自动化跨 100 多个 Web 应用的“衔接工作”显著提高了工作流效率。该开源项目已发布在 [GitHub](https://github.com/opentabs-dev/opentabs) 上。
    - **Judecale** 为使用 Claude 的开发者推荐了 “superpowers” 插件，强调了它对专业开发工作流带来的变革性影响。该插件是官方插件列表的一部分，可以通过 CLI 轻松添加，表明它为编程任务提供了显著的增强。
    - **eo37** 提到使用 “context7” 和 “superpowers” 作为主要的 MCP，强调了它们在管理任务中的实用性。他们还构建了一个自定义 MCP 来访问最新的 LLM 模型并跟踪 API 成本，这有助于确定长期运行和生产环境的开销。这种方法强调了成本管理和紧跟最新模型进展的重要性。

### 3. AI in Creative and Personal Projects

  - **[一位澳大利亚 ML 研究员使用 ChatGPT+AlphaFold，在花费 2,000 美元对其爱犬的 DNA 进行测序后，仅用两个月时间就开发出一种个性化 mRNA 疫苗，使其患有生命威胁的 MCT 癌性肿瘤的爱犬肿瘤缩小了 75%](https://www.reddit.com/r/singularity/comments/1ry961j/an_australian_ml_researcher_used_chatgptalphafold/)** (Activity: 768): **澳大利亚机器学习研究员 **Paul Conyngham** 利用 **ChatGPT** 和 **AlphaFold** 为他的爱犬 Rosie 开发了一种个性化 mRNA 疫苗，Rosie 患有危及生命的肥大细胞瘤 (MCT)。通过花费约 `$2,000` 进行肿瘤 DNA 测序，Conyngham 使用 ChatGPT 识别了新抗原 (neoantigens)，并利用 AlphaFold 预测了蛋白质结构。在与新南威尔士大学 (UNSW) 的 **Martin Smith** 进行基因组测序合作，以及与 **Pall Thordarson** 进行 mRNA 合成合作后，尽管 Conyngham 没有正式的生物学或医学背景，但他成功地在两个月内使肿瘤缩小了 `75%`。这一案例凸显了 AI 在个性化医疗和疫苗快速开发方面的潜力 ([来源](https://www.the-scientist.com/chatgpt-and-alphafold-help-design-personalized-vaccine-for-dog-with-cancer-74227))。** 评论者们正在辩论此案例的影响，质疑这究竟代表了医疗民主化 (healthcare democratization) 的重大转变，还是仅仅是炒作。一些人认为，正如这一不受监管的场景所展示的快速进展那样，监管障碍正在阻碍医学进步。


  - **[ChatGPT 真的在帮我做出极其美味的料理](https://www.reddit.com/r/ChatGPT/comments/1ryk7m4/chatgtp_is_legit_helping_me_cook_insane/)** (Activity: 721): **该帖子讨论了如何利用 **ChatGPT** 通过生成针对现有食材和理想呈现效果定制的分步食谱来提升烹饪技巧。用户报告称，ChatGPT 可以根据食材替换动态调整食谱，并能根据超市收据提供饮食计划。此外，它还能计算热量和 macros 等营养信息。文中提到了一个名为 [Chef Genius Generator](https://bbum.net/pages/chef-genius-generator/) 的工具，它可以根据现有的厨房设备、调料和饮食偏好为 ChatGPT 生成 Prompt。** 评论者强调了 ChatGPT 在实时调整食谱方面的灵活性，以及它在赢得烹饪比赛中的效用。生成个性化饮食计划和营养信息的能力也受到了赞赏。

    - ewbankpj 强调了 ChatGPT 在烹饪中的动态适应性，指出它在替换食材时具有实时调整食谱比例的能力。这一功能对于那些需要根据现有食材或饮食需求做出快速改变的人特别有用。
    - DueCommunication9248 提到使用 ChatGPT 进行营养分析，强调了其根据食材或图片计算热量和 macros 的能力。这一功能对于追踪饮食摄入或参加烹饪比赛的人特别有益。
    - bbum 分享了一个名为 “chef-genius-generator” 的工具，该工具可根据现有的厨房设备、调料和饮食偏好为 ChatGPT 创建 Prompt。该工具增强了食谱生成的个性化程度，允许用户根据特定的厨房配置和饮食偏好优化食谱。




# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会再以这种形式恢复它，但我们将很快发布全新的 AINews。感谢阅读到这里，这是一段美好的历程。