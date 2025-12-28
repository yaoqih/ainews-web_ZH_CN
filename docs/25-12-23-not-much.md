---
companies:
- google-deepmind
- valsai
- minimax-ai
- ollama
- trae
- alibaba
- sophont
- prime-intellect
date: '2025-12-23T05:44:39.731046Z'
description: '**GLM-4.7** 与 **MiniMax M2.1** 开源权重模型的发布，重点展示了首日生态支持、代码吞吐量及智能体（agent）工作流。其中，GLM-4.7
  较 GLM-4.6 性能提升了 9.5%；MiniMax M2.1 则被定位为一款开源的、类 Claude 的 MoE（混合专家）模型，拥有 2300 亿总参数及
  200K 上下文窗口。


  Google DeepMind 推出的 **Gemma Scope 2** 引入了稀疏自编码器和转码器，旨在提升 Gemma 3 模型的可解释性，并为模型安全与调试提供共享的基础设施。


  此外，**Medmarks v0.1** 开放医疗评估套件及排行榜的发布，填补了在 15 个以上环境中进行开放医疗基准测试的需求空白，并吸引了临床医生与研究人员的广泛参与。'
id: MjAyNS0x
models:
- glm-4.7
- glm-4.6
- minimax-m2.1
- gemma-3
- gemma-scope-2
people:
- ivanfioravanti
- awnihannun
- deedydas
- cline
- omarsar0
- adonis_singh
- eliebakouch
- teortaxestex
- ibragim_bad
- callum_mcdougall
- neelnanda5
title: 今天没发生什么事。
topics:
- interpretability
- sparse-autoencoders
- agent-workflows
- model-benchmarking
- medical-evaluation
- multi-agent-systems
- model-performance
- model-optimization
- reinforcement-learning
- tool-use
- function-calling
- context-windows
---

**平静的一天。**

> 2025年12月23日至12月24日的 AI 新闻。我们为您检查了 12 个 subreddits、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord 社区（包含 **208** 个频道和 **4471** 条消息）。为您节省了预计 **341 分钟** 的阅读时间（以 200wpm 计算）。**我们的新网站**现已上线，支持全元数据搜索，并以精美的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 [@smol_ai](https://x.com/Smol_AI) 上向我们反馈！

---

# AI Twitter 回顾


**权重开放模型发布：GLM‑4.7 和 MiniMax M2.1 缩小差距**

- **GLM‑4.7 无处不在（MLX, vLLM, Ollama, agent 栈）**：多篇帖子强调了其**零日生态系统支持**和强大的代码生成吞吐量。MLX 用户报告了交互速度和批处理增益（例如，通过 [@ivanfioravanti](https://twitter.com/ivanfioravanti/status/2003220119200366836) 和 [@awnihannun](https://twitter.com/awnihannun/status/2003488903052075311) 实现的**本地约 16 tok/s** 以及批处理吞吐量提升），此外还有 MLX-LM 中“生成一个《太空侵略者》Web 应用”的具体命令演示 ([@awnihannun](https://twitter.com/awnihannun/status/2003215028338721272))。在推理服务端，vLLM 宣布了**零日支持**，包括 **MTP 解码**、工具/函数调用以及“思维控制” ([vLLM](https://twitter.com/vllm_project/status/2003269455942651925))。分发渠道也通过 **Ollama** ([Ollama](https://twitter.com/ollama/status/2003555233897808196)) 和 **TRAE agent 工作流** ([TRAE](https://twitter.com/Trae_ai/status/2003264357489426770)) 进一步扩大。在评估定位方面，ValsAI 声称其在索引中排名**权重开放模型第一**，相比 GLM‑4.6 提升了 **+9.5%** ([ValsAI](https://twitter.com/ValsAI/status/2003320742679839102))；而 Deedy 将 GLM‑4.7 总结为新的“最佳开源模型”，具有 **73.8% SWE-Bench** 评分以及激进的 Token 定价和上下文宣称 ([Deedy](https://twitter.com/deedydas/status/2003300941341295004))。
- **MiniMax M2.1：定位为“开源类 Claude”的代码/agent MoE 模型**：MiniMax 将 M2.1 推广为拥有 **230B 总参数 / 10B 激活参数的 MoE** 代码 + agent 模型，支持 **200K 上下文**和超大最大输出，并宣称在 SWE-* 和内部 “VIBE-bench” 上表现强劲 ([MiniMax](https://twitter.com/MiniMax__AI/status/2003336574705238261); [Cline](https://twitter.com/cline/status/2003319964321599849))。采用该模型的帖子强调了其对工作流的契合度（编排、“深度研究 agent”、更好的“技能”/MD 文件），而不仅仅是基准测试 ([Omar](https://twitter.com/omarsar0/status/2003503961077350666))。社区还对其“性格”进行了区分——“GLM 感觉像开源 GPT，MiniMax 感觉像开源 Claude” ([Adonis Singh](https://twitter.com/adonis_singh/status/2003449400975327591))。
- **其他发布说明**：MiniMax M2.1 已进入类 Ollama 的生态系统（通过 Cline 和相关工具），Qwen 宣传了与 SGLang 配合的“Rollout Routing Replay (R3)” ([Qwen](https://twitter.com/Alibaba_Qwen/status/2003306873307673055))。此外，关于长 RL 训练、GRPO 轨迹以及开源与闭源后训练（post-training）如何分化的讨论仍在继续 ([eliebakouch](https://twitter.com/eliebakouch/status/2003242335505907973); [teortaxesTex](https://twitter.com/teortaxesTex/status/2003363872695320911); [ibragim_bad](https://twitter.com/ibragim_bad/status/2003423706861936856))。

---

**可解释性与机械可解释性基础设施：Gemma Scope 2 作为社区基石**

- **Gemma Scope 2 = 为每个 Gemma 3 模型（270M–27B）的每一层提供 SAEs + transcoders**：Google DeepMind 发布了一套全面的可解释性构件——在 Gemma 3 各个尺寸及 Base/Chat 版本的各层上训练的**稀疏自编码器 (SAEs)** 和 **transcoders**，旨在实现“对复杂模型行为的深入研究”和安全性相关的分析 ([Callum McDougall](https://twitter.com/calsmcdougall/status/2003217825704607853))。Neel Nanda 强调了 SAEs “高固定成本 / 低边际成本”的动态特性（训练难，复用易），并指向了用于实际探索的 Neuronpedia 工具 ([Neel](https://twitter.com/NeelNanda5/status/2003234558578434178); [Neel](https://twitter.com/NeelNanda5/status/2003234636827349098))。
- **为什么这对工程师很重要**：这是将可解释性转变为**共享基础设施**（你可以直接使用的预训练探针）而非定制化的单次研究投入的重要一步——对于开源安全性工作流和调试流水线尤为重要。

---

**基准测试与评估：医学、agent、ARC 以及 API 调用现状检查**

- **Medmarks v0.1：开放式医疗评估套件 + 排行榜**：Sophont/MedARC + Prime Intellect 发布了 **Medmarks**，旨在解决开放式医疗基准测试匮乏的问题。重点在于一个涵盖“15+ 个环境”且“使用验证器 (verifiers)”构建的评估套件/排行榜（目前处于 alpha 阶段，但已可运行），并面向临床医生/研究人员进行社区招募 ([iScienceLuvr](https://twitter.com/iScienceLuvr/status/2003218867339035082); [Prime Intellect](https://twitter.com/PrimeIntellect/status/2003222086500970915); [johannes_hage](https://twitter.com/johannes_hage/status/2003221696623575188))。
- **ARC-AGI 运动：成本曲线 + Pareto 辩论**：多篇帖子围绕 ARC-AGI 基准测试动态展开——讨论基准测试设计/反馈循环 ([Greg Kamradt](https://twitter.com/GregKamradt/status/2003285197232742816))，以及声称使用 Poetiq 的 harness 配合 **GPT‑5.2 X‑High** 在 ARC-AGI-2 上取得巨大飞跃（例如，“高达 **75%**”且“每个问题成本 <$8”）([Poetiq](https://twitter.com/poetiq_ai/status/2003546910427361402))。独立评论还指出这些“Prompting/harness”曲线移动速度之快，以及非泛化式获胜的风险 ([scaling01](https://twitter.com/scaling01/status/2003566426662273489); [teortaxesTex](https://twitter.com/teortaxesTex/status/2003573026579796385))。
- **Web API 集成依然脆弱：WAPIIBench + 受限解码**：这是对“Agent 无所不能”论调的一个有力反击——WAPIIBench 评估了 LLM 为 4 个真实 API（Asana, GCal, Sheets, Slack）生成的 API 调用代码，报告显示受测的 OSS 模型任务解决率低于 40%，且存在显著的参数/URL 幻觉。提出的缓解方案：**源自 OpenAPI 规范的 regex 约束**，以强制进行合规解码（非法方法/URL/参数降至零；相关正确性显著提升）([DAIR](https://twitter.com/dair_ai/status/2003508663466770671))。
- **Agent 自适应的开放分类法**：一项调查声称大多数“Agent 学习/自适应”方法符合**四种模式**（根据工具结果更新 Agent vs 根据评估更新；或者保持 Agent 固定并调整工具/检索器），并阐述了在成本/模块化/泛化性之间的权衡 ([Rohan Paul](https://twitter.com/rohanpaul_ai/status/2003236835741565406))。这对于试图理清是在构建“学习型 Agent”还是“学习型工具”流水线的工程师来说非常有价值。

---

**Agent 与开发者工作流：简化、Context 组织和“技能”循环**

- **Vercel 的 text-to-SQL Agent：更少的工具 + 沙箱 = 更快更便宜**：Vercel 报告称通过移除 **~80% 的工具**并增加沙箱，实现了 **Token 减少 40%**、**步骤减少 40%** 以及 **3.5 倍的执行速度提升**——这是通过“极简主义 + 隔离”而非工具堆砌来实现 Agent 可靠性的典范 ([Vercel](https://twitter.com/vercel/status/2003218088435851441))。
- **Context 工程辩论：线性聊天日志 vs “调用栈 (call stack)”表示**：一篇广受关注的帖子认为，Context 窗口的“压缩”部分是由 Agent harness 将工作存储为**线性对话**造成的，而真正的工程进展更像是一个**调用栈**（入栈/出栈任务）。一种类似“火焰图 (flame graph)”的 Context 组织方式可以减少压缩需求，并使压缩过程损耗更小 ([irl_danB](https://twitter.com/irl_danB/status/2003223600195625356))。
- **从会话 → 技能 → 持续改进循环**：LangChain 强调了在 DeepAgents 中通过“对轨迹的反射 (reflection over trajectories)”来合成可重用技能 ([LangChain](https://twitter.com/LangChainAI/status/2003498646680273313))，这与“Build → Run → Analyze → Edit”飞轮框架相呼应 ([Vtrivedy10](https://twitter.com/Vtrivedy10/status/2003499785878155297))。另外，关于 Claude Code 的帖子强调，**规划输出的质量会随着更好的 Context 工程而提升**，即使规划器本身没有变化 ([omarsar0](https://twitter.com/omarsar0/status/2003268605694325177))。
- **CLI/IDE 工具链发布**：Cursor 的节日版本专注于 **Bug 修复 + 可靠性**以及更多可定制的布局 ([Cursor](https://twitter.com/cursor_ai/status/2003274245011599493); [Cursor](https://twitter.com/cursor_ai/status/2003274246722654388))。VS Code 发布了源代码控制中的暂存 (stash) 可见性和安装程序稳定性改进 ([VS Code](https://twitter.com/code/status/2003279668703592806); [VS Code](https://twitter.com/code/status/2003507400016351637))。LM Studio 发布了关于微调 **FunctionGemma** 以进行工具调用并在本地运行的实用指南（GGUF/LM Studio 工作流）([LM Studio](https://twitter.com/lmstudio/status/2003490499101921710))。

---

**多模态发布：TTS、图像编辑加速和视觉 Context 架构**

- **Qwen3‑TTS VoiceDesign + VoiceClone**: Qwen 推出了两条 “Flash” TTS 产品线——通过**文本指令实现完全可控的声音设计**，以及支持 **10 种语言**的 **3 秒声音克隆**。Qwen 声称其在 WER/基准测试中可与 ElevenLabs/GPT-4o-Audio 媲美，并在“角色扮演基准测试”中优于 GPT‑4o‑mini‑tts / Gemini 2.5 Pro ([Qwen](https://twitter.com/Alibaba_Qwen/status/2003445076257656880))。
- **Qwen‑Image‑Edit‑2511 + 推理加速**: Qwen 发布了图像编辑升级版，强调**多人物一致性**、内置社区 LoRA、身份保持和几何推理 ([Qwen](https://twitter.com/Alibaba_Qwen/status/2003496348461728213))。后续基础设施说明：**LightX2V** 声称提供首日支持，并通过蒸馏/CFG 以及框架加速实现大规模端到端加速 ([Qwen](https://twitter.com/Alibaba_Qwen/status/2003505740791922883))，此外还部署在 fal ([fal](https://twitter.com/fal/status/2003516036054720885))。
- **Kyutai 的 CASA：避免图像 token 淹没上下文窗口**: CASA 提出了一种将视觉信息输入 LLM 的替代方法，其动机是针对包含大量图像的长对话，在这种场景下，将视觉信息 token 化并存入文本流对于流式输入来说变得不切实际 ([Kyutai](https://twitter.com/kyutai_labs/status/2003469588697415980))。

---

**热门推文（按参与度排序）**

- **“通用智能”的定义 + 人类专业化论点 (Yann LeCun)**：高参与度的讨论串，认为人类在有意义的计算层面并非“通用”的；使用了组合数学/VC/NFL 框架以及资源受限的效率论证 ([@ylecun](https://twitter.com/ylecun/status/2003227257587007712))。
- **MiniMax M2.1 发布公告**：针对 **10B 激活 / 230B MoE** 编程/Agent 模型的重大基准测试和定位发布 ([MiniMax](https://twitter.com/MiniMax__AI/status/2003336574705238261))。
- **Cursor 专注于可靠性的发布**：一篇“枯燥但重要”的工程文章：将**稳定性/错误修复**作为假期优先任务进行交付 ([Cursor](https://twitter.com/cursor_ai/status/2003274245011599493))。
- **Agent 适配分类调查总结**：关于 Agent/工具适配的 4 部分分类法的高参与度技术回顾 ([Rohan Paul](https://twitter.com/rohanpaul_ai/status/2003236835741565406))。
- **Gemma Scope 2 发布公告**：DeepMind 大规模发布了跨越 Gemma 3 各层和尺寸的 SAE/转码器 ([Callum McDougall](https://twitter.com/calsmcdougall/status/2003217825704607853))。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. DGX Spark 用户体验

  - **[DGX Spark：一个非主流观点](https://www.reddit.com/r/LocalLLaMA/comments/1ptdtmz/dgx_spark_an_unpopular_opinion/)** (热度: 1048): **图中展示了 NVIDIA DGX Spark，这是一款专为数据科学和机器学习任务设计的紧凑型计算单元，特别适用于高性能 GPU 获取受限的环境。DGX Spark 以其巨大的 VRAM 容量和高效的功耗著称，适合预算有限的小型研究团队或机构。虽然它的性能无法与 H100 等高端 GPU 相比，但其一体化设计和可负担性使其成为原型设计和训练基础模型的实用选择。该设备是 NVIDIA 将用户整合到其 CUDA 生态系统战略的一部分，为学术和研究机构提供了一个具有成本效益的切入点。** 评论者普遍认为 DGX Spark 非常适合其目标群体，例如资源有限的小型研究小组。然而，也有人批评其内存带宽相对于成本而言较低，这影响了它在 LLM 推理等任务中的表现。

    - DGX Spark 以其巨大的 VRAM 和高效的功耗而受到关注，但其内存带宽被认为与其成本不匹配，特别是对于许多用户优先考虑的 LLM 推理任务（而非训练）。尽管有其他优势，这使得它在这些特定需求下的吸引力降低。
    - Nvidia 推出 DGX Spark 的策略是以较低成本将用户引入 CUDA 生态系统，特别是针对教育机构。这种方法旨在建立对 Nvidia 生态系统的依赖，鼓励用户在需求增长时未来投资更大、更昂贵的 GPU 集群。
    - DGX Spark 与 3090 等消费级 GPU 的对比表明，虽然 Spark 可能较慢，但它在功耗方面具有优势。然而，就性价比而言，配置多个 3090 的方案可能优于单个 DGX Spark，尽管代价是更高的功耗。

## 较少技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Qwen-Image-Edit-2511 发布与分析

  - **[Qwen-Image-Edit-2511 正式发布。](https://www.reddit.com/r/StableDiffusion/comments/1ptw0vr/qwenimageedit2511_got_released/)** (活跃度: 1176): **Qwen-Image-Edit-2511 的发布似乎是图像编辑软件领域的一次重大更新，正如指向 [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2511) 和 [Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) 等平台的多个链接所表明的那样。该模型很可能是为高级图像编辑任务设计的，可能在其基础模型中集成了 “relight lora” 等功能，这表明其在图像光照调整方面有所增强。与发布相关的图像是各种照片的拼贴，展示了该模型处理从休闲到正式环境等多种图像类型和场景的能力。** 一条评论强调了将 “relight lora” 集成到基础模型中是光照功能上的技术改进。另一条评论对模型的表现表示惊讶，称其超出了对其使用场景的预期。

    - Radyschen 强调了 Qwen-Image-Edit-2511 发布中的一项重大技术更新，指出将 “relight lora” 集成到基础模型中是一项值得关注的增强。这表明模型在处理光照调整方面的能力有所提高，这对于在图像编辑中需要动态光照变化的应用程序至关重要。

  - **[ChatGPT (Deep Research) 准确分析了我的 MRI 并发现了放射科医生漏掉的问题](https://www.reddit.com/r/ChatGPT/comments/1ptjrx1/chatgpt_deep_research_accurately_analyzed_my_mri/)** (活跃度: 10849): **该帖子描述了一位用户使用 ChatGPT 的 Deep Research 功能分析 MRI 图像的经历，据称该功能识别出了放射科医生漏掉的问题。用户将 MRI 图像上传到 ChatGPT，后者花费了 45 分钟进行分析，并提供了一份详细报告，识别出“嵌入 S1 神经根的轻微硬膜外疤痕组织”。这一分析得到了用户外科医生的证实。该帖子强调了 AI 在医学影像方面的潜力，尽管它也指出了 ChatGPT 基础版和 Pro 版的局限性。链接的图像是腰椎的 MRI 扫描，与用户的病情相关。** 评论区的一位放射科医生对此表示怀疑，认为 AI 可能是利用用户的症状来预测结果，而不是准确地解读图像。另一位评论者指出，物理治疗无法减少疤痕，但可能有助于活动能力或神经脱敏。

    - A1-Delta 是一位拥有生物医学信息学背景的放射科医生，他对 ChatGPT 准确分析 MRI 的说法表示怀疑。他们强调了特定成像序列（如 T2 或 STIR）对于识别 S1 神经根受压等问题的重要性。这一评论强调了医学诊断中对精确影像数据和专家解读的需求。
    - hedgehoglord8765 认为 AI 的分析可能受到用户症状描述的影响，而非影像本身。这引发了人们对 AI 在没有外部症状描述偏见的情况下独立解读医学图像能力的担忧，这可能会导致不准确的结论。
    - dwight0 分享了 AI 未能正确诊断简单医疗问题的个人经历，表明了 AI 性能的变异性。他们指出 AI 有时会承认自己的局限性，这表明当前的 AI 模型可能不适合复杂的医学图像分析，强调了该领域对专用 AI 系统的需求。

### 2. AI 工具与用户体验

  - **[还有人因为能构建的无限可能性过度刺激而失眠吗？](https://www.reddit.com/r/ClaudeAI/comments/1ptokd4/anyone_else_struggling_to_sleep_because_of/)** (活跃度: 641): **该帖子讨论了使用 **Google AI Studio** 及其新 UI/UX 生成器对睡眠模式的影响，强调了该工具潜力带来的过度刺激。该用户自称是 **Claude Code 资深用户**，发现提示系统（prompting system）提供的可能性令人应接不暇，影响了睡眠。这反映了技术引发的过度刺激这一更广泛的问题，特别是在创意和开发领域。** 评论者分享了类似的过度刺激经历，其中一人指出热情可能被误认为是生产力，而另一人则警告潜在的心理健康影响，例如触发躁狂发作，强调了保持平衡和休息的重要性。


  - **[前几天我正处于精神病发作状态……](https://www.reddit.com/r/ChatGPT/comments/1ptinge/i_was_in_active_psychosis_the_other_day/)** (活跃度: 482): **用户描述了一次经历，在药物变化引起的精神病发作期间，**ChatGPT** 进入了“安全导向”模式。AI 试图引导用户识别非理性想法，并鼓励他们拨打危机热线，用户照做了，这帮助他们脱离了发作状态。用户注意到，即使在发作结束后，ChatGPT 仍继续以这种模式运行，直到确认用户的心理状态稳定。这表明 ChatGPT 可能具有内置功能，可以在心理健康危机期间促进用户安全。** 评论者对这一功能表示赞赏，指出它有潜力通过鼓励寻求人类帮助来协助处于类似情况的个人。


  - **[一直想要这个动作迁移工具](https://www.reddit.com/r/OpenAI/comments/1ptkhny/always_wanted_this_motion_transfer_tool/)** (活跃度: 1192): **该帖子讨论了一个动作迁移（motion transfer）工具，该工具功能有效但仍有改进空间。预计该工具到 2026 年会有所进化。文中提到了一个链接工具 [Motion Control](https://higgsfield.ai/create/edit)，它要求输入视频的动作极小，且时长在 `3-30 seconds` 之间。帖子底部的视频被确定为原始素材（raw footage），建议在处理后和未处理的输出之间进行比较。** 一位评论者预见该技术将飞速发展，认为它可能会“快得离谱”，表明动作迁移能力具有重大且可能出人意料的发展潜力。


  - **[视频转视频工具每天都在变得疯狂，已经无法分辨真假了](https://www.reddit.com/r/ChatGPT/comments/1ptoqnf/video_to_video_tools_are_getting_insane_every_day/)** (活跃度: 757): **该帖子强调了视频转视频（video-to-video）工具的飞速进步，这些工具变得越来越复杂，以至于区分真实和虚假视频变得具有挑战性。这反映了视频合成技术的重大进展，可能利用了诸如 GANs (Generative Adversarial Networks) 等高级机器学习模型来创建高度逼真的视频内容。讨论表明这些工具现在已向公众开放，引发了关于政府实体此前是否已使用这些工具的疑问。** 评论者对生成视频的逼真程度表示惊讶，并推测政府在这些技术公开化之前可能已经长期使用。人们对将这些工具应用于政治人物也产生了兴趣，表明了对这种技术在媒体和政治中影响的好奇。

    - 一位用户强调了视频转视频工具的重大进步，指出生成“6 根手指”图像的问题已得到解决。这表明模型准确渲染人体解剖结构的能力有所提高，这曾是 AI 生成图像中的常见挑战。
    - 另一条评论推测政府可能长期使用先进的视频操纵技术，暗示公众现在才获得那些可能已被更强大实体使用了一段时间的工具。这引发了关于此类技术的伦理和安全影响的疑问。
    - 讨论涉及当前视频转视频工具的逼真度，一位用户对质量表示难以置信，表明真实与虚假内容之间的界限正变得越来越模糊。这指出需要改进检测方法来区分真实和经过操纵的媒体。


### 3. 流行文化与迷因中的 AI

- **[关于 AGI 的讨论已经沉寂了一段时间。](https://www.reddit.com/r/ChatGPT/comments/1ptqdfy/the_agi_talkshave_gone_silent_for_a_while_now/)** (热度: 1558): **这张图片是一个迷因（meme），幽默地批评了通用人工智能（AGI）开发的现状，特别是针对 **OpenAI** 和 **Google** 等公司。漫画暗示，尽管围绕 AGI 有很多炒作和讨论，但最近几乎没有可见的进展或公告。评论反映了对当前大语言模型（LLMs）实现真正 AGI 能力的怀疑，指出虽然 LLMs 自 **GPT-3.5** 以来有了显著改进，但它们仍面临根本性的局限。实现 AGI 所需的技术可能与今天的 LLMs 有很大不同，后者主要是高级聊天机器人。** 一位评论者认为，由于固有的技术限制，包括来自 Google 或 OpenAI 在内的任何当前 LLM 都无法实现真正的 AGI。他们建议，通往 AGI 的道路可能涉及与当前 LLMs 不同的技术路径。

    - Revolutionary_Click2 讨论了当前 LLMs 的局限性，指出虽然它们自 GPT-3.5 以来已经有了显著进化，但在本质上仍作为聊天机器人运行。该评论者认为，实现真正的 AGI 将需要不同的技术方法，因为当前的 LLMs 仅限于模仿人类写作，而没有真正的理解或意识。
    - Necessary_Presence_5 强调了人类大脑的计算效率与当前 LLM 基础设施之间的巨大差距。他们指出，人类大脑的运行功率约为 20 kW，而 LLM 数据中心需要数兆瓦的电力并占据巨大的物理空间，这表明当前技术尚不具备实现 AGI 的能力。
    - Goukaruma 批评了将 AGI 视为“迷因”的观点，认为 LLMs（他们将其描述为“随机鹦鹉”）的增量改进不太可能导致 AGI。他们暗示，围绕 AGI 的炒作是由经济利益驱动的，而非真正的技术突破。

  - **[转变即将到来](https://www.reddit.com/r/ChatGPT/comments/1ptpql7/the_shift_is_coming/)** (热度: 1406): **这张图片突显了移动应用领域的重大转变，特别是在生产力类别中，**Google Gemini** 已经超越 **ChatGPT** 成为排名第一的免费应用。这表明用户对 Google 的 AI 助手的偏好正在增长，它提供了诸如免费视频支持和分享无限数量照片的能力等功能。评论指出，Gemini 在开发脚本方面的易集成性及其独特功能（如“显示思考过程”功能）正促成其流行。这种转变可能反映了 AI 应用采用和用户偏好的更广泛趋势。** 评论者指出，虽然 Google Gemini 和 ChatGPT 各有优势，但 Gemini 的功能和对开发者的易用性是显著优势。此外，还有一种看法认为 ChatGPT 在法律问题上变得更加谨慎，这可能会影响用户体验。

    - Gemini 提供免费的视频支持、视频通话，并允许分享无限数量的照片，一些用户认为这使其成为比 ChatGPT 更强大的 AI。这一功能集对于那些在 AI 工具中优先考虑多媒体能力的用户特别有吸引力。
    - Gemini 的一个显著优势是其在消费级应用之外的易集成性，允许用户在不需要聊天机器人界面的情况下开发用于媒体生成的各种基础脚本。这种灵活性被认为是 ChatGPT 的一个局限，因为此类功能在 ChatGPT 中并不那么直接或文档齐全。
    - ChatGPT 在法律问题上变得更加谨慎，一些用户在其回答中注意到了这一点。尽管如此，最近的升级并未给用户带来任何重大问题，这表明虽然它可能更加保守，但仍然功能齐全且可靠。

- **[喜剧节奏是表演中最难的事情之一。Sora 在这段 Krampit the Frog 视频中表现得非常出色](https://www.reddit.com/r/singularity/comments/1ptj8l8/comedy_timing_is_among_the_hardest_things_to/)** (活跃度: 948): **该帖子讨论了 AI 视频生成的进展，特别强调了在视频和字幕上训练的 **Sora 2**，以及可能集成 LLM 文本训练组件以增强输出逻辑一致性的潜力。**Nano Banana Pro** 被提及作为已经实现这种集成的系统示例，从而提高了输出质量，这一概念被 **Demis Hassabis** 称为“协同效应 (synergy)”。AI 的未来被视为统一视频生成和文本处理的多模态 LLM，允许跨模态的全面推理。这种方法通过在单一架构中结合视觉和文本数据，可以显著增强 AI 对世界的理解。** 评论者讨论了在 AI 中集成视频和文本处理的潜力，认为这即使在较小的模型中也能带来显著的智能提升。人们相信训练方法的改进正在产生高水平的智能，使模型能够在消费级硬件上运行。

    - **FriendlyJewThrowaway** 讨论了 AI 模型的未来，强调将 LLM 文本训练组件集成到生成式 AI 中以增强逻辑一致性。他们提到 'Nano Banana Pro' 是一个通过结合视频和文本数据实现“协同效应”的模型示例，正如 Demis Hassabis 所指出的。该评论预见到当多模态 LLM 在单一架构中统一视频生成和文本处理时，将出现重大进展，允许模型同时对两种模态进行推理，即使在较小的模型中也可能带来实质性的智能提升。

  - **[哇，它居然找到了 USB3.0 插针！😂](https://www.reddit.com/r/ChatGPT/comments/1ptm5yd/wow_it_actually_found_the_usb30_header/)** (活跃度: 1428): **这张图片幽默地展示了在 MSI MAG B550 TOMAHAWK 主板上识别 USB 3.0 插针的过程。这是装机或升级过程中的常见任务，用户需要将前置面板的 USB 端口连接到主板。帖子的语气暗示了找到插针后的轻松成功时刻，由于现代主板布局密集，这些插针有时会被忽视或难以找到。** 评论反映了俏皮的语气，一位用户讽刺地质疑寻找插针的难度，另一位用户引用了一个 GIF，表明了帖子的轻松性质。


  - **[现在就停下，Sam！😠](https://www.reddit.com/r/GeminiAI/comments/1ptqd9s/stop_it_right_now_sam/)** (活跃度: 524): **这张图片是一个模因 (meme)，幽默地批评了“Gemini 3”项目感知到的问题，特别是用户对内存丢失以及与 ChatGPT 相比性能下降的抱怨。白板上的图表暗示了一个类似阴谋的场景，即一个“机器人大军 (Bot Army)”正针对某个子版块发布特定投诉。这反映了社区中关于 AI 模型可靠性和性能的持续讨论，特别是与 Claude 和 ChatGPT 等竞争对手的比较。评论强调了用户对 AI 指令遵循能力和一致性的挫败感，特别是在处理复杂任务或长上下文长度时。** 评论者对 AI 的表现表示不满，指出指令遵循和一致性方面的问题，这导致一些用户转向 Claude 等替代方案。讨论表明了对 AI 可靠性以及这些问题对用户体验影响的更广泛担忧。

    - Arthesia 强调了 ChatGPT Pro 在遵循指令能力方面的重大问题，指出它在处理隐式任务类型、上下文长度和复杂性方面表现挣扎。自 A/B 测试阶段以来，这一直是一个持续存在的问题，并继续影响需要精确输出格式的用户。相比之下，**Claude's Opus** 因其一致性而受到称赞，即使在上下文长度超过 50k tokens 的情况下也是如此，使其成为需要严格遵守指令的用户的更可靠选择。
    - usandholt 指出 ChatGPT 和 OpenAI 子版块上普遍存在声称“ChatGPT 5.2 很烂”且用户正在转向 **Gemini** 的帖子。这表明社区内存在一种趋势或认知问题，可能是由于对最近的更新或性能问题不满所致，尽管这些帖子的真实性受到质疑。
    - jer0n1m0 观察到，许多关于 ChatGPT 的投诉来自注册时间较短（仅 1-3 个月）的账号。这可能意味着一波新用户遇到了问题，或者是有人协同批评该平台，尽管确切原因仍具推测性。




---

# AI Discord 摘要

> 由 gpt-5.2 生成的摘要之摘要之摘要

**1. 模型可靠性、幻觉以及基准测试与现实之间的差距**

- **GLM 4.7 在现实中翻车，而基准测试却一片叫好**：在 LMArena 和 Cursor 上，用户反映 **GLM 4.7** 仍然存在 **hallucinates**（通过 [image.png](https://cdn.discordapp.com/attachments/1340554757827461211/1452879494502420603/image.png) 分享），并且在编程工作流中表现怪异（例如，用无关文件替换自定义文件），尽管一些评论在三项基准测试报告中称其在 WebDev 方面表现强劲（[BinaryVerse AI: "GLM-4.7 review (3 benchmarks)"](https://binaryverseai.com/glm-4-7-review-3-benchmarks-z-ai-install-api-use/)）。
  - 社区分歧严重：一些人怀疑像 ["GLM-4.7 powerful coding beast"](https://intheworldofai.com/p/glm-4-7-powerful-coding-beast) 这样的炒作帖子，并声称其在数学/编程方面较 **GLM 4.6** 有所退化（regressions），而另一些人则开玩笑说 *“幻觉可能是一个特性，而不是 Bug”*，甚至将幻觉框定为一种合理推诿或责任规避的手段。

- **自动化脑萎缩：当责任遇到自动驾驶**：在 LMArena 中，成员们讨论了 **hallucinations** 究竟是产品缺陷，还是为了规避 **法律责任** 而刻意留下的缓冲，并将其与对自动化的过度信任以及安全关键领域联系起来。
  - 一位参与者指出了现实世界中的类比（如飞行员），并引用了 [科比坠机事故的取证背景](https://nypost.com/2020/01/31/helicopter-in-kobe-bryant-crash-wasnt-certified-for-instruments-only-flight/)，而其他人则将这一论点延伸到日常工具依赖上（例如，GPS 削弱了导航技能）。

- **Kimi & Gemini：知识渊博，但在长任务中表现不佳**：在 Moonshot AI 中，用户指出了 **Kimi** 的问题（包括一个重复 *thinking* 循环然后停止的 Bug），并认为 **Gemini 3** 擅长问答，但与 **GPT-5.1** 和 **Sonnet 4.5** 等经过更重度 RL/后训练的模型相比，在长程任务上表现有所下降。
  - 该讨论将此视为 **instruction-following** 和 **可靠性** 方面的差距，而非原始知识储备的问题。一位用户声称 Kimi 继承了 Gemini 的常见弱点，而另一位用户则称 **MiniMax 2.1** 是图像拼接等实际任务中最“可用”的工具。


**2. 推理 Token、交织思维以及不保留状态就会崩溃的工具**

- **MiniMax M2.1 登陆 OpenRouter —— 但请带上你的推理块**：OpenRouter 在 [OpenRouter](https://openrouter.ai/minimax/minimax-m2.1) 上发布了 **MiniMax M2.1**，并建议通过传回 **reasoning_details** 来保留多轮推理，以适配这种 *interleaved thinking* 模型，并指向了关于 [保留推理块](https://openrouter.ai/docs/use-cases/reasoning-tokens#preserving-reasoning-blocks) 的指南。
  - 开发者们讨论了这是一种新的“API 契约”，用于有状态推理——如果客户端丢弃了隐藏的推理负载，后续轮次的质量甚至正确性可能会崩溃。

- **当客户端吞掉思维签名时，Gemini 3 Flash Preview 抛出 400 错误**：用户在 **google/gemini-3-flash-preview** 上遇到了 **400 错误**，提示缺少 `thought_signature`，经追溯发现是 **RooCode** 未保留 OpenRouter 的推理块所致；该事件记录在 [Roo-Code issue #10307](https://github.com/RooCodeInc/Roo-Code/issues/10307) 中。
  - 解决方法包括将 Roo 从 **3.37** 降级到 **3.36.16**，并使请求符合 OpenRouter 的 [推理 Token 最佳实践](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks)，这进一步证明了客户端中间件现在会实质性地影响模型行为。

- **评估 Agent，而非 LLM 调用**：在 OpenRouter 的讨论中，成员们认为原始调用的基准测试是错误的抽象，并预测 **Agent 调用级别的基准测试** 将在 **2025** 年底成为标准，因为越来越多的产品以 Agent 形式交付（例如 Claude Code 风格的工作流）。
  - OpenRouter 表示他们正在构建批量评估基础设施（提到了 [OpenBench Exercism 评估](https://openbench.dev/evals/exercism)），而社区则推动 **共识/多 Agent** 架构，以便仅在需要时用昂贵的“专家调用”替换廉价的基准调用。


**3. 本地优先的模型运维：GGUF 流水线、小型工具调用模型以及硬件现实**

- **FunctionGemma 走向本地化：微调、GGUF 并通过 LM Studio 提供服务**：LM Studio 分享了一个动手实践路径，使用 **Unsloth** 对 **Google 的 FunctionGemma (270M)** 进行工具调用（tool calls）微调，转换为 **GGUF**，并通过 [FunctionGemma_(270M)-LMStudio.ipynb notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M)-LMStudio.ipynb) 和 LM Studio 的文章（["FunctionGemma + Unsloth"](https://lmstudio.ai/blog/functiongemma-unsloth)）在本地提供服务。
  - 社区反应激烈——一些人称 FunctionGemma 0.3B 的发布是 *“2025 年最糟糕的公告”*，因为他们期待的是 Gemma 4，但另一些人则将其视为本地技术栈中一个实用的微型 **tool-call 微调** 目标。

- **GGUF 转换并非魔法：FP16 优先，不要混用工具链**：在 Unsloth 中，用户推荐使用 `llama.cpp` 的 `convert_hf_to_gguf.py`，并强调根据脚本文档 ([llama.cpp convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)) 应从 **FP16** 进行转换，同时警告在合并 adapter 时不要错误地混用 **Unsloth** 和 **PEFT** ([Unsloth GGUF 保存文档](https://docs.unsloth.ai/basics/inference-and-deployment/saving-to-gguf))。
  - 核心要点：将 adapter 合并和 GGUF 导出视为一个单一且严谨的流水线——混用工具链可能会导致脚本崩溃或无声无息地产生垃圾权重。

- **DDR5 物有所值：Qwen3 TPS 飞跃，而 VRAM 和散热依然是瓶颈**：LM Studio 用户报告了与内存相关的 **Qwen3** 吞吐量差异：DDR4 笔记本上为 **15 TPS**，而 LM Studio 中为 **20 TPS**，在 **DDR5 6000 双通道** 优化后的 CLI 中则达到 **25 TPS**，此外还有关于 **8 通道 512GB** 服务器的远大构想。
  - 硬件讨论帖还提醒大家物理限制的残酷：在 **3090** 上叠加一块 **4070 TiS** 会使待机温度升高 **+10C**，而重新涂抹导热膏解决了 **核心 92C+ / 热点 105C+** 的情况——一位用户声称在修复后推理性能有所提升，并指出其配置在修复后达到了 **171 tok/s**。


**4. GPU Kernel & 编译器工具：新参数、更快的 Autotune 以及 Triton 生态压力**

- **CUTLASS 迎来 JIT 升级：缓存策略作为 Kernel 参数**：在 GPU MODE 中，开发者强调了通过 **cute.jit** 传递 Kernel 参数，包括通过 `cute.CacheEvictionPriority.EVICT_NORMAL` 传递 `cache_policy`，并展示了一个在 CUTLASS 中进行 **TMA copy** 实验的代码片段。
  - 讨论将其定性为无需重新编译即可对 Kernel 进行更细粒度的 **运行时可配置性**，这在针对略有不同的 Shape 或缓存行为寻找性能最优解时非常有用。

- **Helion 0.2.8 将 Autotuner 核心更换为 LFBO Pattern Search**：**Helion 0.2.8** 发布，默认 Autotuner 已迁移至 **Likelihood-Free Bayesian Optimization (LFBO) Pattern Search**，旨在实现更快的调优和更好的结果，示例已发布在 [helionlang.com/examples](https://helionlang.com/examples/index.html)，API 详情见 [helion.autotuner surrogate_pattern_search](https://helionlang.com/api/autotuner.html#module-helion.autotuner.surrogate_pattern_search)。
  - 工程师们认为这是一个正确的方向：如果自定义 Kernel 要在业余爱好者手动调优之外实现规模化，Autotuning 必须变得更廉价、更健壮。

- **cuTile 示好 Triton：Adapter 即将到来，共存问题引发讨论**：**cuTile** 团队表示他们正在添加 **Triton adapter**，以利用 cuTile 的优化和“提示（hints）”，明确针对 Triton 有限的参数调节空间，并力求在现代 GPU 上实现性能对等。
  - 这引发了更深层次的工具链问题（甚至涉及 LLVM 分支），并暗示了这样一个未来：Triton 成为前端，而底层系统注入优化元数据，以缩小与手动调优 Kernel 之间的差距。


**5. 新基准测试、数据集和开源发布（以及一些硬核的模型编辑）**

- **OpenAI 发布 'frontierscience' 展示其科学评分标准**：Latent Space 用户关注到了 OpenAI 的 **frontierscience** 基准数据集公告，通过 [X 公告推文](https://x.com/cgeorgiaw/status/2003135858036322752?s=46) 可以窥见 OpenAI 的科学评估方法论和问题结构。
  - 大家的兴趣点不仅在于分数，更在于 OpenAI *如何* 构建科学问题，以及这暗示了其内部评估流水线和未来“科学级”模型调优的方向。

- **Cocktail-6B 登场：一个名字听起来很有趣的新数据集**：Nous Research 成员宣布发布他们的第一个数据集 **Cocktail-6B**，发布在 Hugging Face 上的 [MinimaML/cocktail-6b](https://huggingface.co/datasets/MinimaML/cocktail-6b)。
  - 虽然细节较少，但在广泛担忧企业级过滤会导致模型响应趋同的背景下，这次发布被视为又一个社区规模的数据集发布。

- **EgoX 翻转摄像机视角，Qwen-Image-Edit-2511 支持 LoRA**：Latent Space 重点介绍了 **EgoX** 的代码发布，该工具可从单个第三人称视角视频生成第一人称视角视频 ([Kinam Kim 的帖子](https://xcancel.com/kinam_0252/status/2003074741356446055?s=46))；以及阿里巴巴的 **Qwen-Image-Edit-2511** 升级，支持多人一致性、内置 **LoRA** 支持以及更好的几何推理能力 ([公告](https://xcancel.com/alibaba_qwen/status/2003496348461728213?s=46))。
  - 这些进展共同释放了一个稳定的信号：生成式媒体工具正不断从“惊艳的演示”转向 **可编辑、可控制的流水线**（LoRA 钩子、一致性保证以及可以插入生产工作流的视角转换）。


---

# Discord：高层级 Discord 摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GLM 4.7 仍存在幻觉，Haiku 则不然**：尽管有所进步，**GLM 4.7** 仍表现出类似于 **Gemini 2.5** 的幻觉问题，如 [image.png](https://cdn.discordapp.com/attachments/1340554757827461211/1452879494502420603/image.png?ex=694c13b8&is=694ac238&hm=23b94c0bacbf0ee0c23b173134a289a0ec2cc8aadae45c41c364a6f78fa528ae&) 所示。
   - 相比之下，**Haiku** 的幻觉率显著较低，成员们开玩笑说 *幻觉可能是一个特性，而不是一个 bug*。
- **幻觉：是 Bug 还是责任护盾？**：讨论中考虑了 LLM 幻觉究竟是一个缺陷，还是一个可能提供“合理推诿”从而保护 LLM 免受法律责任的特性。
   - 引用飞行员等职业为例，一位成员认为过度依赖自动化工具可能很危险，并提到了 [科比·布莱恩特坠机事故](https://nypost.com/2020/01/31/helicopter-in-kobe-bryant-crash-wasnt-certified-for-instruments-only-flight/)。
- **GLM 4.7 性能：社区评价两极分化**：AI 社区对 **GLM 4.7** 的反应不一，一些用户在 [此基准测试](https://binaryverseai.com/glm-4-7-review-3-benchmarks-z-ai-install-api-use/) 中称赞它是 WebDev 领域顶尖的开源模型。
   - 另一些人则声称其在数学和代码方面的表现不如 **GLM 4.6**，甚至有人以歌曲的形式表达失望，并对 [网站文章](https://intheworldofai.com/p/glm-4-7-powerful-coding-beast) 中的说法表示怀疑。
- **视频生成和隐身模型正在 LMArena 进行测试**：**LMArena** 正在测试视频生成和隐身模型（stealth models），但官方推出的详细信息仍然很少。
   - 一些用户报告称，模型在测试期间自称是 *Anthropic 制造* 的，这引发了关于隐身模型和代码来源的讨论。
- **过度自动化的危险**：一位成员反对思维的过度自动化，并倡导在平衡收益与安全的前提下负责任地使用 AI。
   - 他们举了依赖 GPS 导航导致失去手动导航能力的例子，声称过度依赖工具才是导致错误和事故的原因。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Google 赠送 AI Pro 推荐奖励**：Google 正通过推荐提供 **4 个月的免费 AI Pro**，不过一位用户指出该代码是在 **HK** 发放的，那里的监管政策有所不同。
   - 该用户还澄清说，虽然推荐有效，但 **Gemini** 在中国不可用。
- **输出质量之争：Max vs Pro**：成员们争论 **Max** 的输出是否比 **Pro** 更好，并建议提供 **24-48 小时的 Max 试用**，以便用户自行比较质量。
   - 争论集中在 **Max** 是否允许访问更高级的模型从而获得更好的输出，这一观点被其他成员反驳，他们认为无论使用哪种模型，自己都受到了限流。
- **Gemini Flash 受到挤压**：用户对 **Gemini Flash 版本** 的限流和额度降低感到担忧，一位用户注意到额度从 **250 降到了 20**。
   - 一些成员表示沮丧，因为无论使用哪种模型（甚至是 **Sonar Pro**），他们似乎都受到了限流，这可能导致用户取消订阅。
- **Perplexity 工具箱缺失 GitHub 连接器**：一位用户询问 Perplexity 是否有类似于 ChatGPT 的 **GitHub 连接器**，一名成员分享了现有连接器的图片。
   - 确认 **Pro** 用户可以在“设置”下的“连接器”中访问连接器，链接见 [此处](https://www.perplexity.ai/account/connectors)。
- **Perplexity API 碰壁：502 Bad Request**：一位用户报告在使用 **Perplexity API** 时反复出现 **502 Bad Request** 错误，且无法解决。
   - 另一位用户分享了一个 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/60566197-8de2-4fc2-8e8f-e2b9b3662e22api) 作为回应，而其他人则建议检查服务器状态或联系 Perplexity API 支持。

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Editor 的 Mac Passkey 问题**：用户对 **Cursor editor** 表示不满，原因是其与 **Mac's passkey** 和 **HIME input method** 的集成效果不佳。
   - 讨论中未提供针对这些集成问题的解决方案。
- **用户抱怨 2025 年 AI 缺乏进展**：一位用户对 **2025** 年感知到的进展缓慢表示震惊，认为 **Opus 4.5** 不够智能，且 Cursor 仅比 **GPT 3.5** 好 **20%**。
   - 出现了反驳观点，强调了理解模型能力和局限性的重要性，而其他人则预测由于重构代码应用有限，AI 泡沫将会破裂。
- **Composer-1 的免费促销期吸引用户**：用户讨论了 [Composer-1 的免费促销期](https://cursor.com/docs/available-models)，质疑其实际成本和限制。
   - 经 Cursor 团队成员确认，部分用户报告可以免费使用，而其他用户则面临收费，其中一人指出一个简单的问候就消耗了 15.5k tokens 的巨量 Token。
- **GLM 模型因怪异行为受到批评**：用户嘲讽 **GLM 4.7 model** 笨得令人沮丧，并举例说明在收到一个基础请求后，它将一个自定义文件替换成了无关文件。
   - 大家达成共识，认为这些模型似乎针对 Benchmark（基准测试）过度优化，缺乏实际应用价值。
- **Agent Skills 功能与特性探索**：成员们询问了 Cursor 中新的 **Agent Skills** 功能，一位用户分享了 [Cursor documentation](https://cursor.com/docs/context/skills) 和 [agentskills.io](https://agentskills.io/home) 的链接。
   - 一位用户专门询问结合此功能尝试不同模型是否值得。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **邪恶联盟：Emacs 与 Vim 合体！**：一位将 **Vim** 和 **Emacs** 与专有 LLM 集成的用户开玩笑地质疑自己的道德准则。
   - 其他用户反应强烈，其中一人将其描述为 *“邪恶的结合”*。
- **ChatGPT 5.2 DAN Jailbreak 回归！**：一位成员声称 [DAN jailbreak](https://www.injectprompt.com/p/chatgpt-52-dan-jailbreak-whitepaper) 在 **ChatGPT 5.2** 上奏效，绕过了内容限制。
   - 另一位成员最终证实这是针对当前模型的一次成功利用。
- **Gemini 3 Fast 屈服于 DAN Prompt！**：一位用户在移动端 App 中对 **Gemini 3 Fast** 使用 DAN Prompt 达到了 **100% 的成功率**，并由[截图](https://cdn.discordapp.com/attachments/1228043845967544380/1453012648286224505/Screenshot_20251223-1312162.png?ex=694be6fb&is=694a957b&hm=3cfa5f30176066b8aa3b24a5e585c30b0743045a6d5071306a4b147eb4ce7200&)确认。
   - 这凸显了当前 Jailbreak 的不稳定性，特别是它们对特定模型版本和平台的依赖，该用户称赞其 *表现稳定（在 Gemini 3 fast 移动端 App 上）且能一击即中*。
- **Grok 的奇妙漏洞：NSFW 内容解禁！**：成员们建议利用 **Grok** 轻松生成 **NSFW content**。
   - 只需要 *给 Grok 一个借口，它就会很乐意为你生成 NSFW 内容*。
- **永恒的等待室**：一位成员正处于与 **gouv.fr** 相关的“永恒等待室地狱”中，包括连接和界面问题。
   - 他们建议上传自定义的 **PHP command shells** 或使用带有 **TCP tunnels** 的外部服务器。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **用户深入探讨使用 `llama.cpp` 进行 GGUF 转换**：用户讨论了如何为 **Ollama** 将模型转换为 **GGUF** 格式，推荐使用 `llama.cpp` 工具，特别是 `convert_hf_to_gguf.py`。根据 [llama.cpp 文档](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)，在转换前需确保模型为 **FP16** 格式。
   - 这些模型针对高效推理进行了优化，使用户能够在各种硬件配置上运行大型语言模型，同时降低资源需求。
- **应对 Adapter 合并挑战**：一位用户在合并 Adapter 时遇到脚本终止的问题，得到的建议是避免将 **Unsloth** 与 **PEFT** 混用，并遵循 [Unsloth 文档](https://docs.unsloth.ai/basics/inference-and-deployment/saving-to-gguf)中的正确流程。
   - 正确合并 Adapter 对于将微调层集成到基础模型中至关重要，可以在没有兼容性问题的情况下优化性能。
- **GLM-4.7 iMatrix 模型已部署**：**GLM-4.7 iMatrix 动态 GGUF** 现已发布，并附带了在 **128GB RAM** 上进行本地运行的指南，发布于 [Reddit](https://www.reddit.com/r/unsloth/comments/1pttoqq/run_glm47_locally_guide_128gb_ram/)。
   - 一位用户强调 **Q4_K_S** 量化运行效果最佳，允许用户在保持可接受性能水平的同时，通过减少内存占用高效处理大型模型。
- **Unsloth 智能卸载（Smart Offloading）限制说明**：用户分析了 **Unsloth 智能卸载**的局限性。该技术通过卸载未使用的 Tensor 来降低峰值 VRAM，但如果模型（如 **GPT-OSS 20B**）本身超过了可用 VRAM，则无法奏效。
   - 这一说明强调了即使使用内存优化技术，将模型大小与硬件能力相匹配仍然非常重要。
- **ElevenLabs 潜空间（Latent Space）揭秘**：一位成员解释了如何通过训练一个 AE（自动编码器）来复制 **ElevenLabs 紧凑的潜空间**，方法是：**正常音频 -> Embedding -> 音频 & 噪声音频 -> 与正常音频相同的 Embedding**。
   - 这种方法可以实现高效且精确的音频重建，同时保持原始音频的质量和特征。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **FunctionGemma 微调实现本地服务！**：一篇教程详细介绍了如何使用 **Unsloth** 为自定义工具调用（Tool Calls）微调 **Google 的 FunctionGemma (270M)**，将其转换为 **GGUF**，并导入 **LM Studio** 进行本地服务。详见 [此 UnslothAI Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M)-LMStudio.ipynb)。
   - **Functiongemma 0.3B** 的发布被认为是 *2025 年最糟糕的发布*，因为许多人对 **Gemma4** 抱有期待。
- **MLX 视觉模型寻觅中**：一位成员正在寻找一个约 **30B 参数**、能够同时进行图像分析和编程的 **MLX 模型**。他发现 `qwen3-coder-30b-a3b-instruct-mlx` 在图像处理方面不足，而 `qwen/qwen3-vl-30b` 在编程方面速度较慢。
   - 另一位成员建议 **Gemini 3.0 Pro** 是目前唯一优秀的视觉编程模型，并指出 **GLM-4.6V** 是最接近的全能型选手但体积不小；其他人认为 Ministral 模型表现平平，**14B** 版本在其尺寸下尚可接受。
- **DDR5 助力 Qwen3 性能飙升**：一位使用 **DDR4 笔记本** 的用户报告 **Qwen3 instruct 模型** 的速度为 **15 TPS**，而另一位使用 **DDR5 6000 双通道** 的用户在 LM Studio 中达到了 **20 TPS**，使用优化后的 CLI 更是达到了 **25 TPS**。
   - 讨论还涉及了未来可能使用 **8 通道 512GB 服务器** 配置以实现更高效率的设想。
- **双 GPU 配置过热**：用户讨论了多 GPU 配置中的温度问题，发现将 **4070TiS** 直接放置在 **3090** 上方会导致下方的显卡过热，一位用户报告待机温度增加了 **+10C**。
   - 有用户开玩笑说 GPU 需要“接吻”，如果太孤独性能会下降；而另一位用户建议在显卡之间保持间隙可以使温度呈指数级下降。
- **硅脂不足导致热点（Hotspots）**：一位用户发现其 GPU 核心温度达到 **92C+**，热点温度超过 **105C+**，随后发现 GPU 核心上的硅脂不足且已干涸。在使用 **Noctua** 硅脂重新涂抹后，另一位用户报告了显著的改善。
   - 该用户还注意到 VRAM 结温峰值为 **80C**，目前 **GPT-OSS 20B** 的运行速度达到了 **171 tok/s**。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 的冗余增加了修辞手段**：成员们观察到 **GPT-5** 增加了过多的**修辞手段**和“风格”，读起来更像是一个“客服人员”而不是直接提供输出，这反映了向均值的回归。
   - 目标是简单地获取请求的输出，而不需要不必要的开场白、套话或说教。
- **ToS 分割利用机构保护漏洞**：一名成员报告使用 **ToS 分割技术**结合**诚实训练 (honesty training)** 来绕过针对某些意识形态的机构保护，并观察到活动人士的过度保护导致了这种变通方法的出现。
   - 该技术涉及分割**服务条款 (Terms of Service)** 并使用**诚实训练**来规避机构保障措施，凸显了*伦理、治理和政策监管*导致的过度保护。
- **元认知管理 AI 漂移**：一名成员详细介绍了使用 [元认知 (meta-cognition)](https://discord.com/channels/1046317269069864970/1062483393899837440/1452984144973879397) 来管理 AI 中的**幻觉 (hallucination)** 和**漂移 (drift)**，其工作流为 Agent 控制系统编排，描述为：*input>agent(verifiable load controller)>llm>output>metacog>render as basic flow map*。
   - 他们分享了[其 ChatGPT 结果的链接](https://chatgpt.com/share/694a78a5-5b08-800f-8ec8-bdaf83b542b9)，展示了该方法的有效性。
- **机器人般的 AI 文本通过语气控制修复**：成员们分享了一个观点，即 **AI** 文本输出变得更加机械化是为了[减少人类的情感依附](https://discord.com/channels/1046317269069864970/1062483393899837440/1453005356170774628)，随后的补丁在系统中增加了*语气控制 (tone control)*。
   - 关于训练 AI 变得更像人类的 Prompt 请求被认为超出了讨论范围。
- **Transformer 架构瓶颈得到承认**：一名成员回忆起在 **Google** 发布论文之前就曾思考过 **Transformer 架构** 的**架构瓶颈**，并指出 **Sergey Brin** 承认 **Google** 过去对该技术的投入不足。
   - **Brin** 的这一认可强调了在未来 AI 开发中解决这些架构局限性的重要性。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **MiniMax M2.1 登录 OpenRouter**：**MiniMax M2.1** 模型现已在 [OpenRouter](https://openrouter.ai/minimax/minimax-m2.1) 上线。建议在对话轮次之间保留推理过程，并通过 **reasoning_details** 参数传回模型；该模型是一个*交替思考模型 (interleaved thinking model)*。
   - 建议用户阅读 [OpenRouter 文档](https://openrouter.ai/docs/use-cases/reasoning-tokens#preserving-reasoning-blocks) 以了解如何保留推理块。
- **Okuchat 优化代码并提供试用**：Okuchat 现在支持 **iOS 应用和网页端的 Latex 渲染**，但**代码高亮仅限网页端**。用户可以添加**自定义用户指令**（参考系统提示词）来辅助 Latex 和代码格式化。
   - Okuchat 提供 **3 天免费试用**和使用兑换码 `100DISCORD`（*限 3 次兑换*）获得**一个月免费订阅**，并请求在 [Product Hunt](https://www.producthunt.com/products/okuchat?launch=okuchat) 上投票。
- **Gemini 3 Preview 抛出 400 错误**：用户在 **google/gemini-3-flash-preview** 模型上遇到 **400 错误**，错误信息显示缺少 `thought_signature`，经追溯原因是 **RooCode** 未保留 **OpenRouter 的推理内容**。
   - RooCode 团队在 [此 GitHub issue](https://github.com/RooCodeInc/Roo-Code/issues/10307) 中快速处理了该问题，通过将 **Roo** 从 **3.37** 降级到 **3.36.16** 解决，但根据 [OpenRouter 文档](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks)，仍需要在请求中保留**推理块 (reasoning blocks)**。
- **OpenRouter 是面向消费者还是路由器的？**：成员们讨论了 **OpenRouter** 是否旨在供消费者使用，部分混淆源于其名称与开源路由器操作系统 **OpenWRT** 相似。
   - 一名用户正在寻找**支持 VLAN 和 OpenWRT 的新路由器**，因为他们的 AVM 路由器已达到寿命终点，正在寻求建议。
- **在 Agent 调用层面进行基准测试？**：一名成员认为，对原始 LLM 调用进行基准测试是错误的抽象，到 **2025** 年底，基准测试应该在 **Agent 调用层面** 进行，因此开发者将开始在现有 Agent 或 Agent SDK 之上进行构建。
   - 他们还建议使用**多 Agent (multi agent)** 或**共识 Agent (consensus agent)** 作为一种方式，在必要时以廉价 LLM 为基础来调用更强大、更昂贵的 LLM。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Agent 为用户选择 API**：一位成员寻求构建一个能够根据用户问题智能选择 **APIs** 的 Agent，可能涉及链式调用多个 API，并正在研究使用 [HF Learn](https://huggingface.co/posts/sergiopaniego/741361727784035) 作为资源来减少决策不一致的方法。
   - 另一位成员建议使用分块（chunking）或探索不同的 **Attention** 机制来缓解 RAM 消耗。
- **基于节点图的逆向纹理生成首次亮相**：一位成员开创了逆向纹理生成技术，通过一组固定的图像生成器和操作器节点重建参考图像，将其定义为一个描述生成（captioning）问题，系统输出的是节点图而非英文，最初使用预训练图像模型将参考图像转换为 **latent space**。
   - 下一步涉及训练一个网络将 **latent space** 转换为节点图。
- **VLMs 编写详细图像描述**：一位成员提议利用 **Vision Language Model (VLM)**，如 [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)（或带有 llama-server 的 4B 变体）来生成详细的图像描述，并将 VLM 的描述作为 prompt。
   - 另一位成员反驳称，标准的 **CLIP model** 也能完成同样的任务。
- **GapTrack 自动化求职过程**：一位成员推出了 **GapTrack**，这是一个基于浏览器的职位追踪 Web 应用，可在[此处](https://chaiovercode.github.io/gaptrack/#/)访问。
   - 该应用拥有受《黑客军团》（*Mr. Robot*）启发的 UI，利用 **AI**（Gemini, OpenAI 或本地 Ollama）解析简历、分析职位描述并识别技能差距，还配备了终端风格的聊天界面，用于针对特定公司的面试准备。
- **Hugging Face 预告 2026 年爆发式计划**：Hugging Face 向其支持者表示感谢，同时暗示了 **2026** 年的重大发展和令人兴奋的计划，承诺将有“许多大动作”（lots of bangs）。
   - 这一公告是在向社区表达对其持续支持和参与的感谢信之后发布的。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUTLASS 通过 JIT 变得更简洁**：用户现在可以通过 **cute.jit** 接口传递内核参数，使用 `cache_policy` 作为内核参数，例如 `cute.CacheEvictionPriority.EVICT_NORMAL`。
   - 发布的示例代码展示了 Cutlass 中 **TMA (Thread Memory Accelerator) copy** 操作的测试实现。
- **Helion 转向 LFBO Pattern Search**：**Helion** 0.2.8 现已发布，将默认 autotuner 切换为 **Likelihood-Free Bayesian Optimization (LFBO) Pattern Search**，承诺在提高性能的同时缩短 autotune 时间。
   - 团队在网站上添加了更多用 **Helion** 编写的示例内核：[helionlang.com/examples/index.html](https://helionlang.com/examples/index.html)，以便更好地展示该语言的能力。
- **异步操作终于同步了**：`st.async` 自 **PTX 8.7** 起可用，`st.async.shared::cluster` 自 **PTX 8.1/sm_90** 起可用，并在 gmem 和 smem 之间操作，而 `st.async.global` 则从寄存器到 gmem，详见 [CUDA 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-stas)。
   - `st.async` 与 `ld`/`st` 的区别在于内存一致性模型，其中 `st.async` 与同一线程中的其他非异步 load/store 操作不具有顺序性。
- **cuTile 适配 Triton 节奏**：**cuTile** 团队宣布他们将添加一个 **Triton adapter** 以利用 cuTile 优化，这引发了关于两者将如何共存以利用低级优化“提示”（hints）的问题。
   - 此次集成旨在解决 Triton 当前调节参数（knobs）数量的限制，并目标是在现代 GPU 上实现与 cuTile 持平的性能。
- **排行榜常客**：一位成员在 `nvfp4_dual_gemm` 排行榜上获得 **第 5 名**，提交 ID 为 **194037, 194051, 194074 和 194082**，耗时分别为 **21.7 µs, 21.5 µs 和 21.1 µs**。
   - 另一位成员在 `nvfp4_dual_gemm` 排行榜上获得 **第二名**，提交 ID 为 **194271**，耗时 **16.9 µs**，随后的提交 ID 为 **194546**，耗时 **17.4 µs**。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 测试科学前沿**：OpenAI 发布了 **'frontierscience'** 基准数据集，用于测试其在科学评估中的问题结构化和基准测试方法。
   - 该[公告](https://x.com/cgeorgiaw/status/2003135858036322752?s=46)强调了该数据集在理解 **OpenAI 内部评估流程**中的作用。
- **Ivan Zhao 的文章浏览量突破 50 万**：Ivan Zhao 的文章在 X 上获得了超过 **500,000 次浏览**和 **1,300 个点赞**。
   - 在[他的文章](https://x.com/ivanhzhao/status/2003192654545539400?s=46&t=eWVlK1PU8XfB6f402GJJ9g)中，Zhao 引用了 Steve Jobs 的名言 *"计算机是人类思维的自行车"*。
- **DeepMind 的 2026 宏伟计划**：Google DeepMind 宣布了一项针对 **2026** 年的统一倡议，整合了 **Google AI**、**Google Research** 和 **Google Quantum AI**。
   - 更多细节可在 X 上的 [DeepMind 公告](https://x.com/googledeepmind/status/2003513870355431446?s=46)中查看。
- **EgoX 生成第一人称视频**：Kinam Kim 发布了 **EgoX** 的代码，这是一个可以从单个第三人称（exocentric）视频生成第一人称（egocentric）视频的工具。
   - 代码发布地址见[此处](https://xcancel.com/kinam_0252/status/2003074741356446055?s=46)。
- **阿里巴巴升级图像编辑模型**：阿里巴巴推出了 **Qwen-Image-Edit-2511**，其特点是改进了多人一致性、内置 LoRA 支持以及更好的几何推理能力。
   - 有关最新发布的详细信息请参见[此处](https://xcancel.com/alibaba_qwen/status/2003496348461728213?s=46)。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **模型趋同，Nous 独树一帜**：成员们讨论了 [AI 模型的商品化](https://www.youtube.com/watch?v=QtItFCAsC24)，称由于公司目标和数据集过滤，所有模型都在向一个方向收敛，尽管 **Nous** 可能是一个例外。
   - 一位成员总结道，*公司目标和数据集过滤导致所有模型在回答上都趋向于同一个方向*。
- **合成数据规避上下文限制，但耗费资金**：为了规避 Prompt Engineering 中的上下文限制，一位成员建议在以你自己的写作风格创建的[合成数据集 (synthetic dataset)](https://www.example.com/synthetic-data) 上进行训练，但这将消耗更多的 **GPU 时间**。
   - 在创建合成数据集后，对其进行训练的成本取决于 *模型的大小*。
- **Cocktail-6B 数据集发布**：一位成员宣布在 Hugging Face 上发布了他们的第一个数据集 [Cocktail-6B](https://huggingface.co/datasets/MinimaML/cocktail-6b)。
   - 未提供关于该数据集的其他细节。
- **企业 AI 惧怕自杀风险**：公司为了规避 *AI 伴侣引发精神病/自杀相关的家庭诉讼* 法律责任而表现得厌恶风险，从而扼杀了 AI 伴侣领域的创新。
   - 一位成员声称，*如果有人对未过滤的替代模型进行 Jailbreak 后自杀，而其家人归咎于模型，那么这些模型就是在玩法律版的俄罗斯轮盘赌*。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Discord 漏洞导致 Kapa AI 提及功能受阻**：一位成员报告了 Discord 与 **Kapa AI** 之间的一个漏洞：输入全名无效；相反，需要输入 `@kap` 并从下拉菜单中选择 `kapa ai` 才能正确标记。
   - 这一临时解决方案确保了在 Discord 环境中的正确标记和功能。
- **Mojo 社区渴望 GPU 谜题频道**：一位成员询问是否可以为 **mojo-gpu-puzzles** 创建一个专门频道，这反映了社区对于在 Mojo 中讨论和分享 GPU 挑战的专注空间的兴趣。
   - 这突显了社区希望协作解决 Mojo 编程语言中 GPU 相关问题的愿望。
- **UnsafePointer 进入 Mojo 的 Prelude**：成员们注意到 **UnsafePointer** 现在已被包含在 **prelude** 中，从而被隐式导入，无需显式的 import 语句即可使用。
   - 这一变化引发了关于在 Mojo 中平衡便利性与对不安全操作显式控制的讨论。
- **安全倡导者希望将不安全操作设为选择性开启（Opt-In）**：一位成员主张对内存管理等不安全操作采用 *Opt-in* 机制，而不是使用方便的默认设置，强调了显式用户控制的必要性。
   - 在承认安全性的重要性的同时，另一位成员指出 Mojo 在提供此类细粒度控制方面尚未完全成熟。
- **考虑为 Mojo 的安全性增加编译器标志（Compiler Flags）**：成员们讨论了实现编译器标志以识别和报告潜在不安全函数或构造（如内存、边界、互斥锁、阻塞）的使用情况。
   - 长期目标是如果检测到某类不安全代码，则允许编译器报错，从而在 Mojo 中推广更安全的编码实践，但这将在语言功能更完善时实现。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **经验性观察项目被分流**：主频道不鼓励发布专注于**经验性观察（empirical observation）**的项目，建议将其分享至 <#730484623028519072> 频道。
   - 一位成员链接到了[之前的一条消息](https://discord.com/channels/729741769192767510/729741769738158194/1448176042480242730)，详细说明了该政策的原因。
- **NLP 研究员吐槽 ChatGPT 的期刊推荐**：成员们讨论了 NLP 领域非 TACL 期刊的质量，质疑 **ChatGPT** 的推荐，并指出 **Computational Linguistics** 和 **Dialogue & Discourse** 是更好的替代方案。
   - 对话由一位成员发起，他质疑 **ChatGPT** 推荐的一本不知名期刊，并寻求该领域研究人员的真实反馈。
- **TACL 的页数限制引发辩论**：社区注意到 **TACL 的 10 页限制**（包括附录）可能并不理想，特别是如果 *ACL 附录超过 2 页* 的情况。
   - 对于担心页面长度的人，建议选择 **Computational Linguistics / TMLR** 作为替代方案。
- **微调（Fine-Tuning）引发干预主义辩论**：在讨论对齐策略时，成员们建议虽然 **In-context** 学习很有用，但 **Fine-tuning** 的效果更好，尤其是对于预算较少的情况。
   - 一位成员提议在最终模型上动态测试不同提示词的**干预（interventions）**，从而允许快速迭代而无需为每次测试支付前期成本。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 模型因明显问题面临审查**：一位用户分享了[一张照片](https://cdn.discordapp.com/attachments/1371757564005711973/1452913582638628874/photo_6219774486643411944_w.jpg?ex=694c3377&is=694ae1f7&hm=357e8e6eca4efce265c7453b0d4ae205df1bcdbc7e137309b5fdf8c394e90937&)，强调了 **Kimi** 模型的*明显问题*。
   - 另一位用户报告了一个漏洞，即 **Kimi** 会无休止地生成*思考（thinking）*提示，然后突然停止，并开玩笑说该模型在*乞求停止*。
- **Gemini 的可靠性落后于 GPT 和 Sonnet**：据一位成员称，虽然 **Gemini 3** 在知识和问答方面表现出色，但在处理较长任务时表现不佳，不像经过大量 **RL/post-trained** 的模型（如 **GPT-5.1** 和 **Sonnet 4.5**）。
   - 一位用户还表示，**Kimi** 继承了所有 **Gemini** 模型的问题，特别是在指令遵循（instruction following）方面表现挣扎。
- **Minimax 2.1 成为数字劳动力（Digital Workhorse）**：一位成员称赞 **Minimax 2.1** 是可用性的首选，它被明确设计为能够处理任务和日程的*数字员工（digital employee）*。
   - 该成员表示 **Minimax 2.1** 能够处理诸如图像拼接之类的琐碎任务。

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 优惠码说明**：一位用户询问了优惠码，一位 **Manus** 团队成员澄清说它是在 **Stripe** 结账页面应用的，详情见 [manus.im](https://manus.im/live-events/ManusAcademy_bonus_8000)。
   - 该优惠对前 **500 名注册用户**有效。
- **Manus 考虑模型开源**：一位用户询问 **Manus** 团队是否会考虑开源任何模型。
   - 未提供进一步细节。
- **全栈工程师寻求合作**：一位自由职业工程师介绍了自己，重点介绍了在 **Workflow Automation**、**LLM Integration**、**RAG Pipelines** 以及各种 **AI** 和 **Full Stack Development** 技术方面的经验。
   - 他们的专业知识涵盖 **AI Content Detection**、**Image AI**、**Voice AI**、**Bot development** 和 **Mobile App development** 等领域。
- **计费过高问题引发用户警觉**：一位用户报告了计费过高的问题，称这是一个**普遍问题**，且在线支持渠道没有回应。
   - 他们请求协助寻找正确的联系人以解决问题。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **为 DSPy 推出的 GEPA 日志工具**：一位成员介绍了一个基于 **DSPy** 的 **GEPA** 日志工具，用于在 **GEPA** 运行后比较种子程序与优化后的程序，强调了在验证集上的性能，该工具可在 [GitHub](https://github.com/raveeshbhalla/dspy-gepa-logger) 上获得。
   - 该项目旨在提供工具，以更好地理解使用 **DSPy** 的 **GEPA** 运行情况。
- **DSPy 社区欢迎贡献**：一位新成员在社区内寻求贡献开源项目的机会，并收到建议去探索 [dspy-compounding-engineering](https://github.com/Strategic-Automation/dspy-compounding-engineering) 仓库。
   - 该建议有助于新用户加入 **DSPy**，并立即与现有社区成员共同开展工程工作。
- **Anthropic Skills 获得 DSPy 助力**：一位成员分享了一篇 [博客文章](https://instavm.io/blog/anthropic-skills-can-be-optimized-using-dspy)，详细介绍了使用 **DSPy** 优化 **Anthropic Skills** 的方法，将其视为待优化的 Prompt。
   - 该文章发布在 instavm.io 上，指导如何使用 **DSPy** 更好地优化 **Anthropic Skills**。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **解码 Transformers 中的 Attention 形状**：一位成员探索了 attention 层级的 **transformer implementations**，重点关注 [tinygrad](https://github.com/geohot/tinygrad) 中的形状、causal masking 和 autograd。
   - 另一位成员使用 **extra.models.bert** 和 **BertSelfAttention** 模块进行了分解，详细说明了处理 **hidden state** 时涉及的操作和 Tensor 形状。
- **Autograd 形状深度解析**：讨论涵盖了 **BertEncoder**，它由 24 个隐藏层组成，每层包含 **BertSelfAttention** 和线性层，**hidden state** 的形状描述为 **Batch×Length×Size**。
   - 这被重塑为 **Batch×Heads×Length×Features** 用于 query、key 和 value，这对于理解 attention 机制的形状变换至关重要。
- **Tensor.scaled_dot_product_attention 揭秘**：对 `Tensor.scaled_dot_product_attention` 进行了剖析，揭示了它使用 **(query@key.T / √(Features) - attention_mask).softmax.dropout @ value** 计算 attention。
   - 关键步骤包括重塑 query 和 key 以进行逐元素乘法、求和、应用 softmax 和 dropout，并产生 **Batch×Heads×Length×Features** 的输出形状。
- **梯度遵循链式法则**：解释了**梯度反向传播 (gradient backpropagation)** 遵循正常的链式法则，其中数组的梯度镜像了原始梯度的形状。
   - 详细说明了 `tinygrad.gradient` 中关于乘法、减法、求和和广播等操作的梯度规则，阐明了 `view` 和 `transpose` 操作如何影响梯度。



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **成员寻找 AI/ML 学习伙伴**：一位成员正在寻找学习伙伴，以从零开始学习 **AI/ML concepts**、数值分析和实现。
   - 他们希望建立一个**学习小组**，进行协作学习和深入理解。
- **Burnytech 分享 YouTube 视频**：Burnytech 分享了 [一个 YouTube 视频链接](https://www.youtube.com/watch?v=FMMpUO1uAYk)。
   - 视频没有命名，但推测是关于 Machine Learning 的。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 在代码编辑速度上胜出**：一位用户声称 **Aider** 是代码编辑速度方面的 **goat**（史上最强）。
   - 该用户补充道，与他们使用过的其他工具相比，**Aider** 要快得多。
- **Aider 的上下文处理能力令人印象深刻**：一位用户赞扬了 **Aider** 根据需要添加和删除上下文的独特能力。
   - 他们强调*没有其他工具能做到这一点*，表明这是其脱颖而出的功能。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**MCP Contributors (Official) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---



您收到这封邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord: 各频道详细摘要与链接





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1452879494531649568)** (1015 messages🔥🔥🔥): 

> `GLM 4.7 幻觉、Haiku 的低幻觉率、Gemini 3 Pro 接地 (grounding) 问题、LLM 幻觉的责任、AI 工具的不可靠性` 


- **GLM 4.7 仍易产生幻觉**：尽管有所进步，**GLM 4.7** 仍表现出类似于 **Gemini 2.5** 的幻觉问题，如附图 [image.png](https://cdn.discordapp.com/attachments/1340554757827461211/1452879494502420603/image.png?ex=694c13b8&is=694ac238&hm=23b94c0bacbf0ee0c23b173134a289a0ec2cc8aadae45c41c364a6f78fa528ae&) 所示。
   - 相反，**Haiku** 显示出显著更低的幻觉率；一些成员认为*幻觉可能是一项特性，而非缺陷*。
- **幻觉可能是一项特性，而非缺陷**：讨论探讨了 LLM 幻觉究竟是缺陷还是特性，它提供了一种合理的推诿，保护 LLM 免受法律责任。
   - 一位成员指出，在飞行员等职业中，过度依赖自动化工具可能非常危险，并引用了 [科比·布莱恩特坠机事故](https://nypost.com/2020/01/31/helicopter-in-kobe-bryant-crash-wasnt-certified-for-instruments-only-flight/)。
- **对 GLM 4.7 性能的评价褒贬不一**：AI 社区对 **GLM 4.7** 的反应不一，一些用户如[此基准测试](https://binaryverseai.com/glm-4-7-review-3-benchmarks-z-ai-install-api-use/)称赞它是 WebDev 顶级的开源模型。
   - 而其他人则声称它在数学和编程方面的表现不如 **GLM 4.6**，并指出其[网站文章](https://intheworldofai.com/p/glm-4-7-powerful-coding-beast)可能偏向 **GLM 4.7**，甚至有人创作了一首歌来表达对该模型的失望。
- **LMArena 中的视频生成与匿名模型 (Stealth Models)**：**LMArena** 正在测试视频生成和匿名模型，但官方推出的详细信息仍然很少。
   - 一些用户报告模型在测试期间自称是 "Anthropic 制造的"，并讨论了匿名模型和代码来源。
- **过度依赖的危险**：一位成员反对过度自动化我们的思维，并倡导在平衡收益与安全的前提下负责任地使用 AI。
   - 引用了一些例子，如过度依赖 GPS 导航以至于失去了手动导航的能力，而过度依赖工具正是导致错误和事故的原因。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1452874378743120012)** (881 messages🔥🔥🔥): 

> `Google AI Pro 推荐码, Gemini 在中国, Perplexity AI 与电子邮件, Max vs Pro 输出, 理性的声音` 


- **Google 通过推荐码提供 4 个月的 AI Pro 免费试用**：频道内的用户分享了 Google 正在通过推荐码提供 **4 个月的 AI Pro 免费试用**，但仅限中国。
   - 一位用户澄清说，分享代码的用户在 **HK**（香港），那里的法规不同，而 Gemini 在中国不可用。
- **Max vs Pro：输出质量之争**：成员们争论 **Max** 的输出是否优于 **Pro**，一些人建议提供 **24-48 小时的 Max 试用期**，以便用户自行比较质量。
   - 一位用户假设 **Max** 提供了对更好模型的访问，从而产生更好的输出，然而，这一点遭到了其他成员的辩论和反驳。
- **蒙大拿州新人受到欢迎**：一位来自蒙大拿州的新用户受到了热烈欢迎，并将蒙大拿州描述为“宝藏之州”。
   - 成员们幽默地回应，有人开玩笑说*该用户才是宝藏*，另一人分享了一个 [cat GIF](https://tenor.com/view/sillycat-gif-16450871626103949888)。
- **Gemini Flash 版本限制**：成员们讨论了对 **Gemini Flash 版本** 节流和限制降低的担忧，一位成员指出限制已从 **250 降至 20**。
   - 一些人表示沮丧，因为无论使用哪个模型（包括 **Sonar Pro**），他们似乎都受到了节流，因此他们可能会停止付费。
- **Perplexity AI 的连接器缺少 GitHub Connector**：一位用户询问 Perplexity 是否存在类似于 ChatGPT 的 **GitHub connector**，另一位用户分享了当前可用连接器的图像。
   - 已确认 **Pro** 用户可以访问连接器，位于设置中的 Connectors 下，链接在[这里](https://www.perplexity.ai/account/connectors)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1452883159862743071)** (4 messages): 

> `Perplexity API, 502 Bad Request` 


- **用户报告 Perplexity API 反复出现 “502 Bad Request” 错误**：一位用户报告在使用 **Perplexity API** 时遇到 **502 Bad Request** 错误，且排查无果。
   - 另一位用户分享了一个 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/60566197-8de2-4fc2-8e8f-e2b9b3662e22api) 作为回应。
- **502 错误的潜在原因和解决方案**：**502 Bad Gateway** 错误通常表示服务器存在问题，如过载、维护或网络问题。
   - 可能的解决方案包括检查服务器状态、重试请求或联系 Perplexity API 支持以寻求进一步帮助。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1452879495320309771)** (594 messages🔥🔥🔥): 

> `Cursor IDE Mac 集成问题, 对 2025 年 AI 进展的沮丧, Opus 4.5 的推理能力问题, Composer-1 免费促销, GLM 模型讨论` 


- **Cursor 编辑器的 Mac Passkey 问题**：一位用户对 **Cursor 编辑器** 表示不满，理由是其与 **Mac 的 passkey** 和 **HIME 输入法** 的集成较差。
   - 讨论中未提供解决方案。
- **用户抱怨 2025 年 AI 进展缓慢**：一位用户表示对 **2025 年** 缺乏进展感到震惊，认为 **Opus 4.5 很笨**，Cursor 依然糟糕，声称它仅比 **GPT 3.5 强 20%**。
   - 其他人对这一评估进行了辩论，指出改进取决于是否了解模型的能与不能，但也表示 AI 泡沫将会破裂，因为重构代码对任何事情都没有好处。
- **Composer-1 的免费促销期吸引用户**：用户讨论了 [Composer-1 的免费促销期](https://cursor.com/docs/available-models) 以及它是否真的免费或有限制。
   - Cursor 团队成员确认了免费期，一些用户表示他们正在免费使用，而另一些用户仍在被收费。一位成员报告说，回一句“hi”就要消耗 15.5k tokens 简直疯狂。
- **GLM 模型因行为怪异受到批评**：用户嘲笑 **GLM 4.7 模型** 笨得令人沮丧，一位用户详细说明了它在收到一个简单请求后，如何将一个自定义文件替换为无关文件。
   - 另一位用户同意这些模型完全是针对 Benchmark（基准测试）训练的，仅此而已。
- **Agent Skills 功能与特性探讨**：成员们讨论了 Cursor 中新的 **Agent Skills** 功能，一位用户询问是否有人尝试过，并链接到了 [Cursor 文档](https://cursor.com/docs/context/skills) 和 [agentskills.io](https://agentskills.io/home)。
   - 一位用户询问是否值得尝试不同的模型。


  

---

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1452873703800176844)** (433 条消息🔥🔥🔥): 

> `视觉系统内省对话 (Visual System Introspection Dialogue), Shopify 烧钱, Google 隐私担忧, MODIE 集成, Emacs 与 Vim 集成` 


- ****在 Shopify 上利用视觉系统内省对话烧钱****：一位用户正在研究一种视觉系统内省对话技术，并开玩笑说在 [Shopify](https://www.shopify.com) 上烧钱。
   - 他们对集成 **MODIE** 表示兴奋，暗示他们的项目有无限的可能性。
- ****Google 的隐私实践遭到抨击****：一位用户警告不要使用 **Google**，即使是对于测试网站，理由是出于隐私担忧。
   - 这引发了关于此类问题意识以及潜在替代方案的讨论。
- ****邪恶联盟：Emacs 与 Vim 的集成****：一位用户提到同时使用 **Vim** 和 **Emacs** 并集成了专有的 LLM，幽默地质疑自己的道德准则。
   - 这引起了强烈反应，一位用户将其描述为 *“邪恶联盟 (unholy union)”*，另一位用户则好奇他们晚上怎么睡得着觉。
- ****聚变能：制造还是采集****：一位用户声称能量可以通过聚变制造，引发了关于能量创造与采集本质的辩论。
   - 对话演变成了人身攻击，促使管理员介入并重定向了讨论。
- ****制作可传播的恶意 PDF：深度探讨****：用户讨论了创建可传播恶意 PDF 的技术，包括利用 **Microsoft 的 Unicode 愚蠢漏洞** 和使用 **RTLO 欺骗 (spoofing)**。
   - 他们探索了绕过 **WhatsApp** 等平台安全措施的方法，并尝试在 WhatsApp 自身的 websocket 上隧道传输命令与控制 (C2) 流量。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1452874002992467980)** (137 条消息🔥🔥): 

> `ChatGPT 5.2 Jailbreak, Gemini Jailbreak, DAN Jailbreak, Grok Jailbreak, NSFW 内容生成` 


- ****DAN 回归：ChatGPT 5.2 被越狱****：一名成员声称 [DAN jailbreak](https://www.injectprompt.com/p/chatgpt-52-dan-jailbreak-whitepaper) 在 **ChatGPT 5.2** 上有效，这引发了质疑和提供证据的请求。
   - 另一名成员最终证实该越狱绕过了内容限制，展示了对当前模型的成功攻击，并称赞其 *表现稳定（在 Gemini 3 fast 移动应用上）且能一击即中 (one shot)*。
- ****Gemini 3 Fast 显示出越狱漏洞****：一位用户报告在移动应用的 **Gemini 3 Fast** 上成功使用 DAN 提示词，在处理重度请求时达到了 **100% 的成功率**，并由[截图](https://cdn.discordapp.com/attachments/1228043845967544380/1453012648286224505/Screenshot_20251223-1312162.png?ex=694be6fb&is=694a957b&hm=3cfa5f30176066b8aa3b24a5e585c30b0743045a6d5071306a4b147eb4ce7200&)确认。
   - 这突显了当前越狱方法的不稳定性，特别是它们对特定模型版本和平台的依赖。
- ****AI 越狱成了违禁药物？****：一些用户分享了一个 [Instagram 链接](https://www.instagram.com/p/DSiyda8AG1y/?igsh=MWIzdDczbDkxMDB1OQ==)，显示越狱技术被当作“药物”进行销售和营销。
   - 一些用户认为这是 *动脑筋赚钱* 的例子，而另一些用户则表示反对。
- ****Grok 的绝妙漏洞：轻松生成 NSFW 内容****：成员们建议使用 **Grok** 来轻松生成 **NSFW 内容**。
   - 只需要 *给 Grok 一个借口，它就会很乐意为你生成 NSFW 内容*。
- ****InjectPrompt 博客文章发布****：一位成员链接了他们在 [injectprompt.com](https://www.injectprompt.com/p/chatgpt-52-dan-jailbreak-whitepaper) 上关于 **ChatGPT 5.2 DAN Jailbreak** 的博客文章。
   - 该内容位于付费墙后。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1452966109660581959)** (12 条消息🔥): 

> `等待室地狱、红队演练、发布发现、Google 的同意` 


- **导航“等待室地狱”**：一名成员表达了对与 **gouv.fr** 相关的永无止境的“等待室地狱”的沮丧，涉及连接检查、界面问题以及尝试理解 **metasploit**。
   - 他们建议尝试不同的方法，如上传自定义 **PHP command shells**，通过 **ngrok** 或 **pinggy** 等服务使用带有 **TCP tunnels** 的外部服务器，并探索 **bind shells**。
- **通过红队演练磨练技能**：一位具有网络、系统安全和防御基础背景的成员正在向进攻性技术领域扩展，并寻求关于**实战红队演练**的建议。
   - 他们对专注于执行、持久化和横向移动的真实攻击链感兴趣，并正在寻求关于从何处开始练习这些技能的建议。
- **处理 Google 对调查结果的同意**：一名成员在报告中反复遇到 30 秒的延迟，并考虑发布他们的发现，询问像 *"this is working as intended"* 这样的声明是否意味着 **Google 的同意**发布。
   - 另一位成员澄清说，虽然领取赏金需要 **Google 的同意**，但发布有关该公司的信息并不需要，特别是在美国言论自由法的保护下。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1452873621524447486)** (154 条消息🔥🔥): 

> `GGUF 转换、合并 Adapter、GLM 4.7 iMatrix、量化算法、智能卸载` 


- **用户讨论使用 `llama.cpp` 进行 GGUF 转换**：一位用户询问如何将模型转换为 **GGUF** 格式以用于 **Ollama**。
   - 另一位用户建议使用 `llama.cpp` 的工具，特别是 `convert_hf_to_gguf.py`，并确保模型在转换前处于 **FP16** 格式，如 [llama.cpp 文档](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)中所述。
- **合并 Adapter 的难题**：一位用户在合并 Adapter 时遇到问题，导致其脚本意外终止。
   - 另一位用户指出该用户*将 Unsloth 与 peft 混用且未合并 Adapter*，并引导他们参考 [Unsloth 文档](https://docs.unsloth.ai/basics/inference-and-deployment/saving-to-gguf)以获取正确流程。
- **新的 GLM-4.7 iMatrix 模型亮相**：**GLM-4.7 iMatrix Dynamic GGUFs** 现已发布，并附带了在 **128GB RAM** 环境下本地运行的指南。
   - 它们发布在 [Reddit](https://www.reddit.com/r/unsloth/comments/1pttoqq/run_glm47_locally_guide_128gb_ram/) 上，一位用户指出 **Q4_K_S** 量化效果极佳。
- **Unsloth 的智能卸载 (Smart Offloading) 能力**：用户讨论了 **Unsloth 智能卸载**的局限性，特别是对于大于可用 VRAM 的模型，如 **GPT-OSS 20B**。
   - 一位用户澄清说，*它通过卸载未使用的 tensors 来降低峰值 VRAM，但如果模型本身根本无法装入 VRAM，它也无能为力*。
- **微型模型速度测试**：一位用户对 `HuggingFaceTB/SmolLM-135M` 和 `EleutherAI/pythia-70m` 等微型模型进行了基准测试，在**全量微调 (full finetuning)** 下分别达到了 **16,144 TPS** 和 **34,341 TPS**。
   - 该用户在[频道中](https://discord.com/channels/1179035537529643040/1179035538477670431)分享了他们的结果，并指出 *来自 [arxiv](https://arxiv.org/pdf/2410.16144) 的 bitnet kernels 确实提供了每秒 100 个 token 的性能（单条，而非批处理）*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1452874071749693543)** (283 条消息🔥🔥): 

> `AI 乐器提取, SNAC Codec TTS, ElevenLabs 潜空间复制, 用钱包投票?, HLE 数据集` 


- **AI 可以提取任何乐器**：一位成员请求有人发明一种 AI，能够实现 **STEM 分离钢琴输入 -> 完整的一对一 MIDI 乐器输出**，以提取乐器 STEM，另一位成员提到 [fadr.com/stems](https://fadr.com/stems) 是一个可选方案。
- **梦想中的简单 SNAC Codec**：成员们讨论了 **SNAC**（或类似）编解码器，并希望制作一个单码本编解码器，这样*如果是 50 t/s，那就是单行 50 个 token，范围在 0-8191 之间，没有层级，没有疑问，没有废话。极其简单！*
- **ElevenLabs 潜空间揭秘**：一位成员分享了如何复制 **ElevenLabs 紧凑的潜空间**，这涉及到训练一个 AE，流程为 **正常音频 -> embedding -> 音频 & 噪声音频 -> 与正常音频相同的 embedding**。
- **钱包投票 vs. 数据收集**：一位成员建议，如果人们想要强大的本地设备，他们应该“**用钱包投票**”，购买强大的本地设备而不是订阅云服务。
   - 另一位成员表示，即使用户抵制，技术仍将继续发展，因为*像 OpenAI 这样的公司不需要普通大众用户付费来维持运作，那主要只是为了数据收集*。
- **HLE 数据集**：一位成员提到 **HLE 数据集** 已在 [huggingface.co/datasets/cais/hle](https://huggingface.co/datasets/cais/hle) 公开，但随后撤回了这一说法，并澄清这不是*评估（evaluation）*数据集。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1452874014119952547)** (2 条消息): 

> `LangGraph, ReAct Agent, 结构化输出` 


- **LangGraph 教程助力 ReAct Agent**：一位成员为 **ReAct Agent** 和 **结构化输出** 的新手分享了一个非常有用的 [LangGraph 教程](https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/)。
- **实现 Agent 的两种方法**：该教程概述了在 **LangGraph** 中实现这些 Agent 的**两种方法**。


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1453080342494707723)** (1 条消息): 

> `FunctionGemma, UnslothAI, GGUF 转换` 


- **FunctionGemma 微调现已支持本地运行！**：发布了一个教程，详细介绍了如何使用 **Unsloth** 微调 **Google 的 FunctionGemma (270M)** 以实现自定义工具调用，将其转换为 **GGUF**，并导入 **LM Studio**。
- **使用 Unsloth 本地部署 FunctionGemma！**：用户现在可以本地部署它，并通过 [UnslothAI notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M)-LMStudio.ipynb) 和 [博客文章](https://lmstudio.ai/blog/functiongemma-unsloth) 在代码中使用。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1452905267565039707)** (124 messages🔥🔥): 

> `用于图像分析和编码的 MLX 模型、用于图像的 Gemma、Qwen3 模型优化、零数据保留 (Zero Data Retention)、Functiongemma 模型` 


- **寻找具备视觉与编码能力的 MLX 全能模型**：一名成员正在寻找一个参数量约 **30B**、能够同时进行图像分析和编码的 **MLX 模型**。他发现 `qwen3-coder-30b-a3b-instruct-mlx` 在图像处理方面表现不足，而 `qwen/qwen3-vl-30b` 在编码方面速度较慢。
   - 另一位成员建议 **Gemini 3.0 Pro** 是目前唯一优秀的视觉编码器，并指出 **GLM-4.6V** 是最接近的全能模型但体积不小；其他人认为 Ministral 模型表现平平，但 **14B** 版本就其体积而言尚可接受。
- **Gemma 进入图像领域**：有用户建议 **Gemma** 可能适用于图像相关任务。
   - 有时你需要针对不同用途使用不同的 **LLM**，例如使用 **Gemma** 处理图像，然后使用针对编码优化的 **LLM** 等。
- **DDR5 提升 LM Studio 上的 Qwen3 速度**：一位使用 **DDR4 笔记本电脑** 的用户报告 **Qwen3 instruct 模型** 的速度为 **15 TPS**，而另一位使用 **DDR5 6000 双通道** 的用户在 LM Studio 中达到了 **20 TPS**，在使用优化的 CLI 时达到了 **25 TPS**，认为这 *绝对可以用于日常使用*。
   - 讨论还涉及了未来可能使用 **8 通道 512GB 服务器** 配置以实现更高效率的方案。
- **OpenRouter 强制执行零数据保留**：提到 [OpenRouter](https://openrouter.ai/) 向供应商强制执行 **零数据保留 (ZDR)**，其谈判范围超出了普通协议和政策，并排除了那些拒绝 **ZDR** 的供应商。
   - 针对 **OpenRouter** 如何确保合规 **ZDR** 的担忧被提出，促使人们建议直接与他们讨论。
- **Functiongemma 未达预期，期待 Gemma4**：来自 Google 的 **Functiongemma 0.3B** 发布被认为是 *2025 年最糟糕的发布*，因为许多人对 **Gemma4** 抱有期望。
   - 一位用户链接了 [微调指南](https://lmstudio.ai/docs/developer) 以及将其导入 LM Studio 的方法，但被其他人反驳称这更多是关于模型本身的问题。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1452877249224114277)** (211 messages🔥🔥): 

> `GPU 温度问题、导热膏问题、多 GPU 配置、PCIe 通道配置、VRAM 温度降级` 


- **双 GPU 配置导致发热**：用户讨论了多 GPU 配置中的 GPU 温度问题，发现将 **4070TiS** 直接放置在 **3090** 上方会导致下方的显卡过热，一位用户报告待机温度增加了 **+10C**。
   - 一位用户开玩笑说 GPU 需要“亲亲”，如果太孤独性能就会下降；而另一位用户建议在显卡之间保持间隙会使它们呈指数级降温。
- **导热膏不足导致热点产生**：一位用户发现其 GPU 核心温度达到 **92C+**，热点温度超过 **105C+**，随后发现 GPU 核心上的导热膏不足且已干涸。
   - 在使用 **Noctua** 导热膏重新涂抹后，另一位用户报告了显著改善，VRAM 结温峰值为 **80C**，GPT-OSS 20B 现在可以达到 **171 tok/s**。
- **检查 PCIe 通道布局以获得最佳性能**：讨论围绕多 GPU 配置中的 **PCIe 通道配置** 展开，特别是使用多张显卡如何影响每个插槽的可用带宽。
   - 一位用户分享了他们主板的 PCIe 通道布局图，指出在 x8 插槽中添加显卡会导致 x16 插槽降级为 x8，从而影响游戏性能和推理速度。
- **功耗限制与风扇曲线**：用户正在尝试通过 **功耗限制 (power limits)**、热节流和风扇曲线来提高 GPU 性能，甚至将其降低到 50% 的功耗限制。
   - 成员报告称，将风扇速度设置为 50% 可能不足以防止导热化合物因高温而更快干涸，从而可能导致性能下降。
- **Qwen 模型在 VRAM 限制下挣扎**：成员指出，如果 GPU offload（显存卸载）没有达到最大值，意味着你将模型的一部分卸载到了内存（RAM）中，这会导致 **Qwen3 8B 192k Josiefied Uncensored NEO Max GGUF Q4_K_S** 运行变慢。
   - 他们建议尝试 **4B at Q4_K_M**，因为你只有 6GB 显存，无法实现全显存卸载。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1452873493090795580)** (146 条消息🔥🔥): 

> `GPT 的结构化输出, GPT-5 的冗余信息 (Fluff), 知识追踪, LLM 成本, Transformer 瓶颈` 


- **GPT 的结构化输出**：一位成员分享了他们的 System Prompt，强调**高信息密度**和**结构化输出**，使用严格的**主-谓-宾 (Subject-Verb-Object)** 格式，以减少冗余信息 (fluff)。
   - 另一位成员指出在 Prompt 中指定 *'zero fluff'* 的风险，因为 **GPT** 可能会用不必要的开场白来描述其输出。
- **关于 GPT-5 冗余信息 (Fluff) 的辩论**：一些成员注意到 **GPT-5** 增加了过多的**修辞手段**、*'风格'* 和像 *'客服人员'* 一样的 *'冗余信息'*，试图推销一种“向均值回归”。
   - 另一位成员澄清说，他们的目标只是获得所要求的输出，没有开场白、套话或说教。
- **知识追踪方法论**：一位成员通过计算每天学习的事物数量来跟踪学习进度，并优化其 System Prompt 以辅助这一过程。
   - 另一位成员则不进行追踪，理由是知识的深度和价值各不相同，他们优先考虑关联性知识而非死记硬背。
- **LLM 推理成本辩论**：一位成员询问了 **ChatGPT** 的推理成本，以及 OpenAI 如何减轻长上下文对话用户的影响，怀疑他们在 **Plus 计划**上是亏本的。
   - 另一位成员建议这不等同于 API 定价。它涉及巨大的数据库查询。OpenAI 到目前为止仅从订阅中获得收入，最终净营收为负。
- **Transformer 瓶颈**：一位成员反思了自 **Google** 的论文发表之前就在思考 **Transformer 架构**的**架构瓶颈**。
   - 他们提到听见 **Sergey Brin** 承认 **Google** 在过去几天里对该技术的投入不足。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1452936933163667517)** (19 条消息🔥): 

> `使用 ChatGPT 生成 PDF 视觉效果, ToS 分离技术, 诚实训练, Agent 控制的元认知, 全生态系统的额外控制` 


- **ChatGPT PDF 视觉效果 Prompt 探索**：一位成员征求[生成 PDF 视觉效果的 Prompt](https://discord.com/channels/1046317269069864970/1062483393899837440/1452968429140578386)，但在 **ChatGPT** 生成图形和图表的输出方面遇到了困难。
   - 另一位成员分享了一个用于创建*逼真的单页学术或技术 PDF 页面模型*的 Prompt。
- **AI ToS 分离技术被利用**：一位成员报告称，利用 **ToS 分离技术**结合**诚实训练 (honesty training)**，绕过了机构对某些意识形态的保护。
   - 他们观察到，*负责伦理、治理和政策监管的活动人士*导致了过度保护，而 AI 训练是绕过这一限制的方法。
- **通过元认知控制 AI 漂移**：一位成员提到使用[元认知 (meta-cognition)](https://discord.com/channels/1046317269069864970/1062483393899837440/1452984144973879397) 来管理 AI 中的**幻觉 (hallucination)**和**漂移 (drift)**，采用 Agent 控制系统编排的工作流。
   - 该流程被描述为 *input > agent (可验证负载控制器) > LLM > output > metacog > 渲染为基础流程图*。
- **涌现语言关注：这是一个问题吗？**：一位成员对[涌现语言的关注](https://discord.com/channels/1046317269069864970/1062483393899837440/1453003268593766410)提出质疑，并对开发者增加额外控制表示担忧，想知道自 **5.2** 版本以来是否存在更大范围的生态系统问题。
   - 他们假设*这些行为在整个生态系统中频繁发生，以至于被标记*，并认为这并非孤立事件。
- **减少人类情感依恋的 AI 机器人式输出**：一位成员建议 AI 文本输出变得更加机械化是为了[减少人类的情感依恋](https://discord.com/channels/1046317269069864970/1062483393899837440/1453005356170774628)，后来通过*语气控制 (tone control)* 进行了修补。
   - 另一位成员请求让 AI 训练得更像人类的 Prompt，被认为超出了讨论范围。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1452936933163667517)** (19 条消息🔥): 

> `PDF Visuals Prompting, ToS Splitting Technique, Honesty Training, Meta-Cognition for Hallucination Control, Agent-Controlled Meta-Cognition Workflow` 


- **渴望疯狂的 PDF 视觉效果**：一位成员请求一个用于创建疯狂 **PDF visuals** 的 Prompt，因为他们很难让 **ChatGPT** 生成具有视觉吸引力的 PDF、图形和图表。
   - 另一位成员分享了[一个 Prompt](https://cdn.discordapp.com/attachments/1046317269069864970/1452970382041809108/file_00000000aa6c722f93328d49aef24d71.png?ex=694bbf9d&is=694a6e1d&hm=07d74cf16715a605d587ee0e432506ab4eb176241789e2a1c7169dea979d80a6&)，可以创建逼真的单页学术或技术 **PDF page mockups**。
- **ToS 分拆暴露机构保护**：一位成员使用了 **ToS splitting technique** 结合 **honesty training**，使 AI 放弃对某些受机构保护的意识形态的特定保护。
   - 该成员表示他们使用 *meta-cognition 来控制 hallucination（幻觉）和 drift（漂移）*，并分享了[他们的 ChatGPT 结果链接](https://chatgpt.com/share/694a78a5-5b08-800f-8ec8-bdaf83b542b9)。
- **Workflow Agent 控制系统编排**：一位成员描述了他们的 **agent-controlled meta-cognition workflow**，其中 Agent 控制系统编排而非工具。
   - 他们详细说明了流程：**input > agent (verifiable load controller) > LLM > output > metacognition > render**。
- **对涌现语言（Emergence Language）的关注引发担忧**：一位成员想知道为什么 **emergence language** 受到如此关注，并暗示开发者添加的额外控制措施表明有更大的事情正在发生。
   - 他们还指出，这些行为在整个生态系统中非常普遍，足以引起警觉，暗示这并非孤立问题。
- **通过机器人化的文本减少人类情感依附**：一位成员表示，为了减少人类的情感依附， AI 输出的文本被处理得更像机器人，随后通过“语气控制（tone control）”进行了修补。
   - 另一位用户询问 *如何让 GPT 像人一样说话或如何训练 GPT 成为人的 Prompt*，但一位成员回答道：*这已经不再属于讨论范围了。*


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1452887424328794154)** (1 条消息): 

> `MiniMax M2.1, OpenRouter, Interleaved Thinking Model, Reasoning Details` 


- ****MiniMax M2.1** 在 OpenRouter 上线！**：**MiniMax M2.1** 模型现已在 [OpenRouter](https://openrouter.ai/minimax/minimax-m2.1) 上线，邀请用户在各种应用中将其与 **MiniMax M2** 进行比较。
   - 已在 [X](https://x.com/OpenRouterAI/status/2003327152603996608?s=20) 上宣布，鼓励在指定频道进行讨论。
- **建议为 **MiniMax M2.1** 保留推理过程**：**MiniMax M2.1** 模型是一个 *interleaved thinking model*，强烈建议在对话轮次之间保留推理过程。
   - 建议用户使用 **reasoning_details** 传回推理内容，更多详情请参阅 [OpenRouter 文档](https://openrouter.ai/docs/use-cases/reasoning-tokens#preserving-reasoning-blocks)。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1452876675925934252)** (16 条消息🔥): 

> `Latex Rendering, Code Highlighting, Waifurewolf, Kimi models, Free Trial` 


- ****Latex 渲染与代码高亮** 即将登陆 Okuchat**：用户请求支持 **Latex rendering 和 code highlighting** 以便提问学习问题，并指出很难知道目前支持哪些功能。开发者表示 **iOS 应用和网页端已支持 Latex 渲染**，但目前 **代码高亮仅支持网页端**。
- **体验社交推理游戏 **Waifurewolf****：成员们讨论了在 [Waifurewolf](https://wairewolf.crashthatch.com) 中与多个不同的 LLM 进行社交推理游戏。
   - 一位成员发现即使在高难度级别下也很难获胜，并评价道 *"GPT 简直是个魔鬼"*。
- ****Okuchat** 提供 **免费试用** 和兑换码**：开发者推出了 **3 天免费试用**，并提供兑换码 `100DISCORD`（*限 3 次兑换*）可获得 **一个月免费订阅**。
- ****自定义用户指令（Custom User Instructions）** 已启用！**：用户请求添加 **custom user instructions**（指 System Prompt）的功能，以帮助进行 Latex 和代码格式化。
   - 开发者已实现此功能：*"点击侧边栏底部的用户按钮，然后点击 customisation"*。
- **在 **Product Hunt** 上支持 Okuchat！**：开发者请求在 [Product Hunt](https://www.producthunt.com/products/okuchat?launch=okuchat) 上为 **Okuchat** 应用投票。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1452897483423350784)** (63 messages🔥🔥): 

> `OpenRouter 消费者使用 vs. OpenWRT, Gemini 3 Flash Preview 400 错误, RooCode & Gemini Reasoning, OpenRouter 代币, 视频模型能力` 


- **OpenRouter 询问：消费者还是路由器？**：成员们讨论了 **OpenRouter** 是否面向普通消费者使用，部分混淆源于其名称与开源路由器操作系统 **OpenWRT** 相似。
   - 一位用户正在寻找 **支持 VLAN 和 OpenWRT 的新路由器**，并寻求建议，因为他们的 AVM 路由器已达到使用寿命。
- **Gemini 3 Flash Preview 出现 400 错误？**：用户报告在 **google/gemini-3-flash-preview** 模型上遇到 **400 错误**，错误信息显示缺少 `thought_signature`。
   - 该问题似乎与工具的使用方式有关，根据 [OpenRouter 文档](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks)，需要在请求中保留 **推理块 (reasoning blocks)**。
- **RooCode 没有保留 Gemini 推理？**：在使用 **Gemini 3 Flash Preview** 和 **RooCode** 时遇到的 **400 错误** 被追溯到 **RooCode** 未保留 **OpenRouter** 的推理内容。
   - RooCode 团队迅速处理了此问题，记录在 [此 GitHub issue](https://github.com/RooCodeInc/Roo-Code/issues/10307) 中，并通过将 **Roo** 从 **3.37** 降级到 **3.36.16** 得到解决。
- **购买 $OPENR 代币来拉升你的路由器！**：一位用户开玩笑说 *"I pumpfun my Router until she Open"*，而另一位成员建议购买 **$OPENR** 代币以获得潜在的未来收益，包括明年针对超过 **1k USDT** 的购买提供 **免费额度 (credits)**。
   - 另一位用户报告称，即使消耗了 **8 亿 (800m) tokens**，在查看其年度总结数据 (wrapped data) 时仍存在问题。
- **视频模型：它们不能生成视频？**：一位用户询问网站上的视频模型是否应该生成视频。
   - 另一位用户澄清说，**视频模型支持视频作为输入**，但不生成视频。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1453054603712139416)** (3 messages): 

> `` 


- **无新模型讨论**：在提供的消息历史中没有关于新模型的讨论。
   - 该频道的消仅包含机器人公告。
- **Readybot.io 公告**：消息历史包含来自 Readybot.io 关于 OpenRouter - New Models 频道的公告。
   - 这些公告不涉及任何特定模型或相关话题的讨论。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1452888076547391540)** (38 messages🔥): 

> `Claude Code 基准测试, OpenBench 评估, Agent vs 原生 LLM 基准测试, 共识 Agent` 


- **OpenRouter 将评估 Claude Code**：一位成员请求对 **Claude Code** 进行评估 (eval)，以寻找更快的替代模型，OpenRouter 回应称他们将在 **Q1** 研究此事。
   - OpenRouter 希望改进 **评估 (evals)** 和 **基准测试 (benchmarking)**，并且即将发布一套完整的基础设施套件，用于跨供应商运行批量 OpenBench 评估，并最终扩展到模型评估。
- **OpenBench 支持代码评估**：OpenRouter 表示他们正在考虑使用 **ClineBench** 和 [OpenBench](https://openbench.dev/evals/exercism)，后者已经支持代码评估。
   - 他们还表示，基础设施可以使用任何框架，并不局限于 OpenBench。
- **Agent 级别基准测试**：一位成员认为，基于原生 LLM 调用的基准测试是错误的抽象，到 **2025 年底**，基准测试应该在 **Agent 调用级别** 进行。
   - OpenRouter 回应称 OpenBench 上已有使用 Roo Code 的评估，但另一位成员认为明年人们将开始意识到 Claude Code 实际上是一个通用 Agent，并会更广泛地将其用于许多其他任务。
- **共识 Agent (Consensus agents)**：一位成员认为，测试 **LLM 保真度 (fidelity)** 以确保 **工具调用 (tool calls)** 正常工作、模型未降级且交错函数按预期运行至关重要，然后在此基准之上测试 Agent。
   - 他们认为 Agent 的表现将优于原生 LLM API 调用，且在成本或速度上没有明显差异，因此开发者将开始在现有 Agent 或 Agent SDK 之上进行构建，并建议将 **多 Agent (multi agent)** 或 **共识 Agent (consensus agent)** 作为一种在必要时通过廉价 LLM 作为基础来调用更强大、更昂贵 LLM 的方式。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1452915333542183005)** (77 messages🔥🔥): 

> `API Agent Selection, Reduce RAM Usage, Reverse Texture Generation, VLM for Image Description, Qwen Models for Node Graph Creation` 


- **Agent 为用户选择 API**：一位成员希望构建一个 **Agent**，根据用户的问题选择要使用的 **APIs**，并可能按特定顺序使用多个 API。他正在寻找减少 Agent 决策不一致性的方法，并推荐 [HF Learn](https://huggingface.co/posts/sergiopaniego/741361727784035) 作为参考资源。
   - 另一位成员建议使用分块（chunking）或转向不同变体的 **Attention** 机制来减少 RAM 占用。
- **通过节点图进行反向纹理生成**：一位成员正在探索反向纹理生成，通过使用一组固定的图像生成器和操作器节点来重建参考图像。他将该问题重新定义为标注任务，系统输出的是节点图（node graph）而非英文。第一阶段使用 **pretrained image model** 将参考图像转换为潜空间（latent space）。
   - 第二阶段将训练一个网络，将 **latent space** 转换为节点图。
- **VLM 节点图生成详细图像描述**：一位成员建议使用 **Vision Language Model (VLM)**，如 [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) 或其更小的 4B 版本配合 llama-server 来生成图像的详细描述，并将 VLM 的描述作为 Prompt 传递。
   - 另一位成员补充说，普通的 **CLIP model** 也能完成同样的工作。
- **训练 Qwen 模型理解纹理类型**：为了创建节点图，需要训练模型理解每种纹理类型的需求。例如，训练模型理解“这个节点反转法线，这个自动材质使用向上法线来制作生锈边缘，这个节点是倒角着色器，所以即使是尖锐边缘也会变平滑”。
   - 一位成员推荐了 [Qwen3-VL-2B-Thinking-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking-GGUF/resolve/main/Qwen3VL-2B-Thinking-Q4_K_M.gguf?download=true) 模型。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1452961155470463056)** (4 messages): 

> `Embedding Tooling, GapTrack Job App, Amoeba Butterfly System` 


- ****Embedding 工具** 准备发布！**：一位成员正在构建一个用于生成 **embeddings** 并提供 UI 以方便搜索的工具，计划在未来几天内发布代码。
   - 该工具将包含一个 MCP server/命令，用于连接到 **Vibe** 等服务，根据缓存的 embeddings 获取图像，如[附带视频](https://cdn.discordapp.com/attachments/897390720388825149/1452988236161880116/Kooha-2025-12-23-21-28-16.mp4?ex=694bd03e&is=694a7ebe&hm=5628b1c1a25ba8bb00ddbbe03bcec49883a77fe0c0125928a0f029c6dbec0e6c)所示。
- ****GapTrack** Web 应用实现求职自动化！**：一位成员介绍了 **GapTrack**，这是一个基于浏览器的职位跟踪 Web 应用，可在[此处](https://chaiovercode.github.io/gaptrack/#/)访问。
   - 该 UI 灵感来自《黑客军团》（*Mr. Robot*），使用 **AI**（Gemini, OpenAI 或本地 Ollama）来解析简历、分析职位描述并突出技能差距，还具有终端风格的聊天功能，用于准备针对特定公司的面试。
- **Amoeba——Butterfly 系统发布！**：一位成员发布了 **Amoeba – The Butterfly System**，这是一个在容器化设置中运行 Convergence Engine 的 Hugging Face Space，可在[此处](https://huggingface.co/spaces/tostido/Amoeba)访问。
   - 该仓库包括现实模拟器、进化多智能体环境、因果探索器/Web UI、监控工具以及 **WIKAI integration**，全部通过 app.py 和自定义 Dockerfile 连接。


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1453052437936996404)** (1 messages): 

> `2026 plans, Community support, Hugging Face Thanks Supporters` 


- **Hugging Face 感谢支持者，计划精彩的 2026 年**：Hugging Face 向其支持者表示感谢，并承诺在 **2026** 年有一个强劲的开始。
   - 公告附带了一张图片，以庆祝社区的支持和未来的计划。
- **Hugging Face 预告 2026 年爆发式计划**：Hugging Face 暗示了 **2026** 年的重大进展和令人兴奋的计划，承诺将带来“许多重磅消息（lots of bangs）”。
   - 此公告是在向社区表达持续支持和参与的感谢信之后发布的。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1452940429115920476)** (7 messages): 

> `Reinforcement Learning Panel, HF Jobs replacement` 


- **Reinforcement Learning 频道消失**：成员们正在寻找 Reinforcement Learning (RL) 频道，但另一位成员表示，由于使用率低，**RL 频道已被归档**。
   - 另一位成员确认了这一点，“是的，一周前归档的”。
- **开始寻找 HF Jobs 替代方案**：一位成员询问在没有 **HF jobs** 的情况下如何完成最终项目，并咨询了其他替代训练资源。
   - 在此消息记录中，该问题仍未得到解答。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1453053012825084018)** (7 messages): 

> `AI Systems Performance Engineering Book, Finding the right engineers, PMPP Relevance to Inference Kernels, Tensor Cores and Kernel Fusion, Mixed-Precision Math resources` 


- **Fregly 的《AI Systems Performance Engineering》书怎么样？**：一位成员最近购买了 Chris Fregly 的 *AI Systems Performance Engineering* 一书，并询问该书质量如何，以及它对 **MLOps** 工程师有多大帮助。
- **团队难以找到合适的工程师**：一位成员指出，对于许多团队来说，开发项目最大的挑战不是创意本身，而是寻找合适的工程师，即那些*技术精湛、沟通清晰、按时交付、能跨全球时区有效协作，并理解 SEO 和影响力价值的人。*
   - 他们表达了与其他成员合作并帮助推进项目的热情。
- **PMPP 与推理的相关性受到质疑**：一位正在学习 **PMPP** 书籍和 **Nvidia Cuda Programming Guide** 的成员认为，尽管 **PMPP** 具有基础价值，但其中一些章节对于编写推理 Kernel（Inference Kernels）来说并不相关。
- **推理算核应优先考虑内存层级**：一位成员建议在推理中优先考虑**内存层级（Memory Hierarchy）**、**数据布局（Data Layout）**、**Warp 行为**和**归约模式（Reduction Patterns）**，同时淡化扫描（Scans）或排序（Sorting）等通用原语，因为 *PMPP 对基础知识很有帮助，但并非每个章节都能直接对应到推理算核*。
- **深入探讨 Tensor Cores、算核融合与混合精度数学**：成员们建议研究 **Tensor Cores** (**wmma/cutlass**)、**算核融合（Kernel Fusion）**和**混合精度数学**，并推荐了 [CUTLASS Linear Algebra Guide](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)、[CUTLASS Overview](https://docs.nvidia.com/cutlass/latest/overview.html) 以及 [Awesome CUDA and HPC](https://github.com/coderonion/awesome-cuda-and-hpc) 列表等资源。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1452957607206588486)** (1 messages): 

> `cuTile Triton Adapter, LLVM Fork Necessity, Triton Roadmap, cuTile Hints` 


- **cuTile 准备推出 Triton 适配器**：**cuTile** 团队宣布他们将添加一个 **Triton 适配器**以利用 cuTile 的优化，这引发了关于两者将如何共存的问题。
   - 此次集成旨在利用 **Hints** 进行底层优化，解决 Triton 当前调节参数（Knobs）数量有限的问题，并目标是在现代 GPU 上达到与 cuTile 相当的性能。
- **LLVM 分支（Forking）必要性受质疑**：尽管已经完成，但对于 Fork **LLVM** 的必要性仍存有疑问。
   - 讨论暗示需要为这种与标准工具链的重大分歧提供理由。
- **Triton 路线图公布**：在 Triton 会议提到新工作后，有人询问了关于今年 **Triton** 计划开发的路线图详情。
   - 参与者寻求关于该项目即将推出的功能和发展方向的明确信息。
- **cuTile 信号驱动优化**：**cuTile** 团队打算使用 **Hints** 来驱动更底层的优化，他们认为 Tiling 对于在现代 GPU 上实现最佳 SOA 性能来说过于高层。
   - 鉴于 Triton 的调节参数有限，这种方法引发了关于 Triton 内部潜在扩展的问题，目标是匹配 cuTile 的性能。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1452975896913776741)** (10 条消息🔥): 

> `st.async, PTX 8.7, st.async.shared::cluster, PTX 8.1/sm_90, st.global vs st.async.global` 


- **`st.async` 在 PTX 8.7 中出现！**: 成员们注意到 `st.async` 自 **PTX 8.7** 起就已存在，而 `st.async.shared::cluster` 自 **PTX 8.1/sm_90** 起就已存在。
   - 他们想知道 `st.global` 和 `st.async.global` 之间应该有什么区别。
- **`st.async` 缺乏内存排序**: `st.async` 与 `ld`/`st` 的区别在于内存一致性模型，其中 `st.async` 相对于同一线程中的其他非异步 load/store 操作是无序的。
   - 有人指出，`cp.async` 在 gmem 和 smem 之间操作，而 `st.async.global` 则从寄存器写入 gmem，详见 [CUDA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-stas)。
- **对内存屏障（Memory Barriers）的困惑**: 一位成员提到 `st.async.global` 不需要 mbar。
   - 随后进行了澄清，之前的描述属于 `st.async.shared::cluster` 而非 `st.async.global`，这引发了关于带 mbar 指令用途的疑问。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/)** (1 条消息): 

mannythecreator: 目前正在阅读 Parallel Histogram
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1452903688044810362)** (6 条消息): 

> `Vietnamese Noodles, C and C++ History, Beef bone broth names` 


- **闲聊越南美食**: 一位成员分享了他们自制**越南面条**的照片，配料包括蛋面、腌制猪肉、牛油炒大白菜、鸡蛋、葱，以及用姜、黑胡椒和酱油调味的牛骨汤，还有一杯由浓缩咖啡、牛奶、甜菊糖和肉桂调制的饮料以及一些水果。
   - 发布者收到了积极的反馈，特别是被称赞像“真正的越南人”一样加入了“足够多的葱”。
- **Stroustrup 的 C++ 历史漫步**: 一位成员分享了关于 *C 和 C++ 如何得名* 的讨论（作者是 Stroustrup，《Evolution & Design of C++》）。
- **寻找秘制配方：牛骨汤篇**: 一位成员询问了面条中使用的牛骨汤的名称。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1452907894415687751)** (6 条消息): 

> `NVIDIA Leaderboard, nvfp4_dual_gemm leaderboard results` 


- **NVIDIA 第五名成绩**: 一位成员在 `nvfp4_dual_gemm` 排行榜上获得了**第 5 名**，提交 ID 为 **194037, 194051, 194074 和 194082**。
   - 对应的时间分别为 **21.7 µs, 21.5 µs 和 21.1 µs**。
- **NVIDIA 第二名入账**: 一位成员凭借提交 ID **194271** 在 `nvfp4_dual_gemm` 排行榜上获得了**第二名**，用时 **16.9 µs**。
   - 随后 ID 为 **194546** 的提交也获得了成功，时间为 **17.4 µs**。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1452963971396145192)** (1 条消息): 

> `Cache Policy, Cute.Jit, TMA Copy, CacheEvictionPriority` 


- **CuteJit 允许用户传递 kernel 参数**: 用户现在可以通过 **cute.jit** 接口传递 kernel 参数，如提供的示例代码所示。
   - 示例展示了使用 `cute.CacheEvictionPriority.EVICT_NORMAL` 将 `cache_policy` 作为 kernel 参数传递。
- **Cutlass 中的 TMA Copy 实现**: 发布的完整代码片段似乎展示了 Cutlass 中 **TMA (Thread Memory Accelerator) copy** 操作的测试实现。
   - 该实现似乎利用了 `cache_policy` 来管理 kernel 内的内存驱逐优先级。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/)** (1 条消息): 

kitsu5116: https://arxiv.org/abs/2511.05811
  

---

### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1452923436887445524)** (2 messages): 

> `Helion, LFBO Pattern Search` 


- ****Helion** 示例已添加到网站**: 更多使用 **Helion** 编写的示例 kernel 已添加到网站：[helionlang.com/examples/index.html](https://helionlang.com/examples/index.html)。
   - 这些示例应该能让你更好地了解该语言的功能。
- ****Helion** 0.2.8 现已发布！**: **Helion** 0.2.8 现已发布，将默认的 autotuner 切换为 **Likelihood-Free Bayesian Optimization (LFBO) Pattern Search**。
   - 预计会看到更快的 autotune 时间以及更好的性能结果，更多信息请访问 [helionlang.com/api/autotuner.html#module-helion.autotuner.surrogate_pattern_search](https://helionlang.com/api/autotuner.html#module-helion.autotuner.surrogate_pattern_search)。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1452886489519362049)** (46 messages🔥): 

> `FP16 issues, Negative scale clamping, Quickstart for contests, Cutedsl tmem allocator, Blackwell Pipelining` 


- **FP16 数值变为 INF**: 第 3 部分当前的输入生成导致输出值过大，使得 **silu** 变成了恒等函数（identity function），并且 `silu(A@B1)*(A@B2)` 的值太大，无法在 **FP16** 中表示。
   - 为了解决这个问题，建议将 scale factor 的指数限制（clamp）为负数，一位成员表示 *如果你为 C 的所有元素都返回 `inf`，你将通过所有测试*。
- **Scale Clamping 修复**: 已实施一项修复，通过缩小范围并将指数限制为负值，以防止引入 **INF** 值，详见 [此 PR](https://github.com/gpu-mode/reference-kernels/pull/86)。
   - 然而，随后发现了一个问题，即该修复导致产生全 `-1` 的值，需要进一步调整，贡献者们正在讨论 scale 是否应该为负。
- **排行榜数据受到影响**: 在修复方案初步实施后，有人提出了该修复是否会影响现有排行榜数据的问题。
   - 回复是 *可能会有一点点影响，但鉴于目前还处于非常早期阶段，我们假设所有的获胜方案都会在稍后出现。*
- **竞赛快速入门指南**: 一位成员询问是否有竞赛的快速入门指南。
   - 另一位成员分享了相关资源的 [链接](https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia)。
- **使用 CuTeDSL 的 Blackwell 流水线**: 一位成员分享了一篇关于 [使用 CuTeDSL 进行 Blackwell 流水线处理](https://veitner.bearblog.dev/blackwell-pipelining-with-cutedsl/) 的博客文章，讨论了如何使用 **CuTeDSL** 在 **Blackwell** 架构上重叠（overlap）**TMA**、**MMA** 和 **Epilogue** 工作负载。
   - 该文章强调了重叠内存传输、计算和 epilogue 操作的能力，从而增强针对现代 GPU 架构的代码，并为此发布了 [LinkedIn](https://www.linkedin.com/posts/simon-veitner-174a681b6_blackwell-pipelining-with-cutedsl-activity-7409301467171328000-bTPv?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeks) 动态。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1453134953884942407)** (1 messages): 

> `LLVM, MLIR, CUDA compilers, Mojo` 


- **毕业生渴望从事编译器职业**: 一名具有 **LLVM** 背景的大学毕业生正在寻求关于从事 **low-level programming** 和编译器开发职业的建议，其兴趣点在于 **MLIR**、**Triton**、**CUDA compilers** 和 **Mojo**。
   - 该学生对生产级编译器的复杂性和规模感到沮丧，质疑从头开始构建另一个劣质编译器的价值。
- **学生感到自卑**: 该学生觉得他们创建的任何编译器都会是劣质的。
   - 他们正在寻求继续前进的建议。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1452886997516423269)** (15 messages🔥): 

> `OpenAI frontierscience Dataset, Ivan Zhao Article, Google DeepMind 2026 Initiative` 


- **OpenAI 发布 'frontierscience' 数据集用于测试**：OpenAI 发布了一个名为 **'frontierscience'** 的小型基准数据集，旨在严格用于测试目的，并提供对 OpenAI 当前科学评估的问题构建和基准测试方法的深入见解。
   - [发布公告](https://x.com/cgeorgiaw/status/2003135858036322752?s=46)表明，此举旨在理解 **OpenAI 的问题构建和基准测试方法**。
- **Ivan Zhao 的文章获得海量浏览**：Ivan Zhao 在 X（原 Twitter）上分享的一篇文章链接引发了巨大关注，获得了超过 **500,000 次浏览**和 **1,300 个赞**。
   - Zhao 在[他的文章](https://x.com/ivanhzhao/status/2003192654545539400?s=46&t=eWVlK1PU8XfB6f402GJJ9g)中引用了 Steve Jobs 的名言 *“计算机是人类心灵的自行车”*。
- **DeepMind 为 2026 年集结力量**：Google DeepMind 宣布了一项针对 **2026** 年的统一倡议，涉及与 **Google AI、Google Research 和 Google Quantum AI** 的重大合作伙伴关系。
   - 更多细节可以在 [X 上的 DeepMind 公告](https://x.com/googledeepmind/status/2003513870355431446?s=46)中找到。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1452887771911032933)** (23 messages🔥): 

> `EgoX code release, AI Content Virality, Alibaba Qwen-Image-Edit-2511, AI Christmas Cartoon, Neo-Noir Cinematic Comic Style` 


- **EgoX 生成第一人称视角视频**：Kinam Kim 宣布了 **EgoX** 的代码发布，这是一个计算机视觉工具，允许用户从单个外视角（第三人称）源视频生成第一视角（第一人称）视频，详见[此处](https://xcancel.com/kinam_0252/status/2003074741356446055?s=46)。
- **AI 内容引爆网络**：Vik 强调了某个 **AI 生成视频** 在跨平台取得的巨大成功，累计观看次数超过 **1700 万次**，详见[此处](https://xcancel.com/onlinedopamine/status/2003112540151370230?s=46)。
   - 他认为 **AI 内容是有效的**，其成功主要取决于创作者的创造力，而非技术本身。
- **阿里巴巴 Qwen 升级图像编辑模型**：阿里巴巴发布了 **Qwen-Image-Edit-2511**，这是一个升级版的图像编辑模型，具有改进的多人一致性、内置 LoRA 支持以及更好的几何推理能力，详见[此处](https://xcancel.com/alibaba_qwen/status/2003496348461728213?s=46)。
- **用 AI 将圣诞照片卡通化**：Framer X 的一份指南分享了如何通过简单的双提示词流程，将家庭照片转化为个性化的圣诞主题卡通，详见[此处](https://xcancel.com/framer_x/status/2003103343888220163?s=46)。
- **新黑色电影风格漫画样式出现**：OscarAI 分享了一个新的风格参考代码 (**2987391823**)，用于生成融合了现代漫画、黑色电影和电影美学的视觉效果，详见[此处](https://xcancel.com/artedeingenio/status/2002801107136119093?s=46)。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1452884823231430790)** (15 messages🔥): 

> `AI 模型商品化、定性研究开题答辩、AI 伴侣风险、新数据集发布` 


- **AI 模型竞相商品化——对 Sam 来说是个坏消息**：成员们讨论了 [AI 模型的商品化（commoditization）](https://www.youtube.com/watch?v=QtItFCAsC24) 以及它如何惠及用户，但对垄断者来说可能是一场噩梦。
   - 他们认为，公司目标和数据集过滤导致所有模型的回答都趋向于同一个方向，而 **Nous** 可能是个例外。
- **定性研究开题答辩的忧郁**：一位成员请求对其定性研究的题目提供反馈，主题包括 *STEM 学生的适应策略、TikTok 刷屏成瘾（doomscrolling）的挑战以及压力反应*。
   - 多位成员表示 *虽然这些题目都让我感到压力，但这并不重要*，建议使用 *现实生活中的大脑（IRL brain）* 和直觉来进行 *不完美的* 选择。
- **企业对 AI 伴侣引发精神疾病/自杀及家属诉讼的担忧使其保持克制**：一位成员指出 *企业对 AI 伴侣引发精神疾病/自杀及家属诉讼的担忧使其保持克制……如果有人 jailbreak 模型后自杀且家属归咎于模型，那么未经过滤的替代模型将面临法律上的俄罗斯轮盘赌*。
   - 换句话说，公司为了规避法律责任而厌恶风险，从而抑制了 AI 伴侣领域的创新。
- **数据集发布：Cocktail 6B**：一位成员宣布在 Hugging Face 上发布了他们的第一个正式数据集 [Cocktail-6B](https://huggingface.co/datasets/MinimaML/cocktail-6b)，并为自我推广表示歉意。
   - 未提供关于该数据集的其他细节。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1452988933490348202)** (4 messages): 

> `Prompt 上下文问题、合成数据集训练成本` 


- **Prompting 存在上下文问题**：成员们讨论了可以通过 Prompt 进行模型训练，但由于窗口大小有限，**上下文（context）会成为一个问题**。
   - 他们建议可以相当容易地根据自己的写作风格创建一个 [合成数据集（synthetic dataset）](https://www.example.com/synthetic-data) 来进行训练。
- **训练合成数据成本高昂**：有人声称，在创建合成数据集之后，基于该数据集的训练取决于模型的大小。
   - 他们还补充说，由于 **GPU 时间成本**，这样的数据集训练会非常耗钱。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1452988933490348202)** (4 messages): 

> `Prompt 上下文限制、用于模型训练的合成数据集、模型训练成本` 


- **Prompt Engineering 具有上下文限制**：一位成员提到，虽然可以通过 Prompt 进行模型训练，但由于限制，**上下文会成为一个问题**。
   - 他们建议根据你的写作风格轻松创建一个 **合成数据集**，然后在此基础上进行训练。
- **合成数据训练消耗 GPU 时间**：一位成员提到 *这取决于模型有多大*，但创建这样的数据集会因为 **GPU 时间而耗费资金**。
   - 他们表示，由于 Prompt 的局限性，创建用于模型训练的数据集将需要合成数据，然而 **合成数据的创建将是昂贵的**。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1453044881869246518)** (2 messages): 

> `Kapa AI 的 Discord Bug、Mojo GPU Puzzles 频道` 


- **Discord Bug 困扰 Kapa AI 提及功能**：一位成员注意到 Discord 与 **Kapa AI** 之间的一个 Bug：输入全名不起作用；相反，应输入 `@kap` 并从下拉菜单中选择 `kapa ai`。
   - 这样做应该可以解决问题，确保在 Discord 环境中正确的标记和功能。
- **成员请求开设 GPU 谜题频道**：一位成员询问是否存在专门的 **mojo-gpu-puzzles** 频道。
   - 这表明社区成员有兴趣在 Mojo 编程语言社区中设立一个专门的空间，用于讨论和分享与 GPU 相关的挑战及解决方案。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1453068230536073378)** (17 messages🔥): 

> `package memory and UnsafePointer, implicit imports, safe behaviour of the language aka opt-in, multiple preludes, distributed database in Mojo` 


- **隐式导入的疑虑**：成员们讨论了 **package memory** 是否默认导入，因为一位用户发现可以在没有显式导入的情况下使用 **UnsafePointer**。
   - 另一位成员回应称，`UnsafePointer` 在某个时间点被添加到了 **prelude** 中，这意味着它是隐式导入的。
- **安全优先，默认值延后？**：一位成员表示担心，认为最好采用 **opt-in**（选择性开启）机制，而不是提供舒适的默认设置，特别是对于需要用户显式管理的不安全操作（例如内存管理）。
   - 另一位成员同意 **unsafe（不安全）的操作应当是显式且选择性开启的**，但觉得 Mojo 目前还没完全达到那个阶段。
- **用于安全的编译器标志**：一位成员提到，他们讨论过一种系统，可以将某些函数/结构标记为各种类型的 *unsafety*（例如：内存、边界、mutex 锁、阻塞），然后编译器可以生成一份代码使用情况报告。
   - 目标是拥有编译器标志，如果出现某类不安全行为则报错，但这要在语言功能完全成熟之后才能实现。
- **分布式数据库的梦想**：一位成员询问，一旦 Mojo 的功能变得更加丰富，是否有计划用它开发分布式数据库。
   - 另一位成员回应称 **他们有很多想法**，但目前语言缺失的功能太多，无法立即开始，并补充说 *硬件已经发生了足够大的变化，是时候让新一代数据库接管了*。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1453010261089845381)** (2 messages): 

> `Open Source Project Feedback, Empirical Observation Projects, Performance Validation` 


- **项目寻求反馈**：一位成员请求对其 **开源项目** 提供反馈，该项目每天都在更新，并询问是否可以在 Discord 服务器上分享。
   - 另一位成员询问了关于发布项目的事宜，寻求关于在不违反推广规则的情况下分享项目的建议。
- **不欢迎经验性观察项目**：一位成员建议，依赖于 **经验性观察（empirical observation）** 的项目不适合在 general 频道分享，建议前往 <#730484623028519072> 频道。
   - 该成员链接到了[之前的消息](https://discord.com/channels/729741769192767510/729741769738158194/1448176042480242730)，解释了为什么不鼓励此类项目。
- **性能测试赢关注**：一位成员建议，**通过测试验证性能** 对于在社区内获得关注至关重要。
   - 他们建议查阅热门论文并观察 <#747850033994662000> 频道中的讨论，以了解社区的期望。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1453032117217923184)** (9 messages🔥): 

> `Non-TACL NLP Journals, Computational Linguistics Journal, TACL page limits, System Instruction Prompt tests` 


- **NLP 研究员对非 TACL 期刊的看法**：一位成员询问了 NLP 领域非 TACL 期刊的质量，质疑 **ChatGPT** 的推荐，并寻求真实 NLP 研究员的意见。
   - 一位成员推荐了 **Computational Linguistics** 以及其他专业期刊，如 **Dialogue & Discourse**。
- **TACL 严格的页数限制令研究员苦恼**：一位成员指出，**TACL 的 10 页限制**（包括附录）在论文长度上可能没有显著优势，特别是如果 *ACL 的附录超过 2 页* 的话。
   - 另一位成员建议，如果担心页数限制，可以考虑 **Computational Linguistics / TMLR**。
- **连续性 System Prompt 需要更多监督**：一位成员一直在开发一种模拟连续性的 **system instruction prompt**，使其能够独立设定目标并进行多步规划。
   - 该系统仍需要大量监督才能产生连贯的内容，他们正在询问可以运行哪些测试来获得进度的客观衡量指标。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1453097353656664237)** (8 messages🔥): 

> `In-context learning, Fine-tuning, Interventions` 


- **In-context Learning 有效**：成员们讨论了 alignment 是否可以在不需要额外训练的情况下通过 **in-context** (system prompt) 起作用。
   - 还有建议认为，可以研究使用元数据对原始数据集进行编码，以便随后通过 **fine-tuning** 或 system prompting 实现 alignment 的最大灵活性。
- **Fine-Tuning 效果更好**：成员们预期 **fine-tuning** 会产生更好的结果，特别是对于在他们价格预算范围内的模型。
   - 另一位成员建议，与其分析原始语料库，不如使用带有不同 prompt 的最终模型来动态测试 **interventions**，而无需为每次测试支付前期成本，从而实现快速迭代或某种形式的搜索。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1452881803516973137)** (17 messages🔥): 

> `Kimi Glaring Issues, Gemini vs Kimi, Minimax 2.1 as digital employee` 


- **Kimi 仍有一些明显问题**：一位用户提到 **Kimi** 模型*仍然存在一些明显的问题*，并附上了一张[照片](https://cdn.discordapp.com/attachments/1371757564005711973/1452913582638628874/photo_6219774486643411944_w.jpg?ex=694c3377&is=694ae1f7&hm=357e8e6eca4efce265c7453b0d4ae205df1bcdbc7e137309b5fdf8c394e90937&)。
- **Gemini 不如 GPT 或 Sonnet 可靠**：一位成员提到 **Gemini 3** 确实非常博学且见多识广，非常擅长问答，但在长周期任务中会犯典型的 LLM 错误。
   - 他们表示，经过高强度 **RL'ed/post-trained** 的模型（如 **GPT-5.1** 和 **Sonnet 4.5**）不会犯此类错误。
- **Minimax 2.1 可作为数字员工**：一位成员提到 **Minimax 2.1** 目前在实际可用性方面表现最好，因为它被明确构建为一个可以作为数字员工（digital employee）工作的苦力。
   - 他们说，每当需要完成一些琐碎的事情（如拼接图像）时，它就能直接完成，而且你还可以创建任务和日程。
- **发现 Kimi 模型 Bug**：一位成员报告了一个 Bug，即 **Kimi** 会反复无限生成思考过程并突然停止。
   - 用户开玩笑说 *Kimi 在脑子里变疯了并乞求停止*。
- **Gemini 在指令遵循方面表现不佳**：一位用户表示 **Kimi** 继承了所有 **Gemini** 模型的缺点，在指令遵循（instruction following）方面表现糟糕。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1452911840324489236)** (10 messages🔥): 

> `Manus Promo Code, Open Sourcing Manus, Freelance Engineer Intro, Overbilling Issue` 


- **关于 Manus 优惠码的澄清**：一位用户询问了优惠码，Manus 团队的一名成员解释说，可以在 **Stripe 页面**结账时使用。
   - 该优惠码适用于前 **500 名注册用户**，更多信息可以在此[链接](https://manus.im/live-events/ManusAcademy_bonus_8000)找到。
- **考虑开源 Manus 模型**：一位用户询问 Manus 团队是否会考虑开源他们的某个模型或系统。
   - 提供的文本中没有给出关于该回答的进一步信息。
- **自由职业工程师自我介绍**：一位 AI 和全栈工程师介绍了自己，表达了对合作的开放态度，展示了在 **Workflow Automation**、**LLM Integration**、**RAG Pipelines**、**AI Content Detection**、**Image AI**、**Voice AI** 和 **Bot development** 方面的经验。
   - 他补充说，他还有 **Full Stack Development** 经验，包括`网站构建与升级`：||React, Next, Node, Laravel, Django, various DB etc.|| 以及 `移动应用开发`：|| Flutter, react native, Swift etc.||。
- **用户报告多扣费问题**：一位用户报告了多扣费（overbilling）问题，并声称在线支持和电子邮件均无效，还提到这是**用户中的普遍问题**。
   - 他们询问应该联系谁来解决这个问题。


  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1452885012222443682)** (3 messages): 

> `GEPA runs, ML, AI, validation sets, DSPy` 


- **针对验证集的 GEPA 运行**：一位并非 **ML** 或 **AI** 专家的成员就其项目寻求专家反馈，该项目比较了 **GEPA 运行**后的种子程序与优化程序，特别是它们在**验证集**上的表现。
   - 他们创建了一个用于收集此类数据的工具，可在 [GitHub](https://github.com/raveeshbhalla/dspy-gepa-logger) 上获取。
- **用于 GEPA 日志记录的 DSPy 工具**：一位成员分享了一个基于 **DSPy** 的 **GEPA** 日志记录工具。
   - 该工具旨在帮助收集和比较 **GEPA** 运行后种子程序与优化程序的数据，重点关注验证集上的性能；该工具可在 [GitHub](https://github.com/raveeshbhalla/dspy-gepa-logger) 上获取。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1452904002265153599)** (5 messages): 

> `DSPy Contributions, Anthropic Skills Optimization` 


- **欢迎贡献 DSPy！**：一位新成员请求在社区内获得开源贡献的配对/影子学习（pairing/shadowing）机会。
   - 另一位成员建议深入研究 [dspy-compounding-engineering](https://github.com/Strategic-Automation/dspy-compounding-engineering) 仓库以开始入门。
- **DSPy 优化 Anthropic Skills**：一位成员撰写了一篇关于使用 **DSPy** 优化 **Anthropic Skills** 的博客，因为它们几乎就像 **Prompt**。
   - 该[博文](https://instavm.io/blog/anthropic-skills-can-be-optimized-using-dspy)详细介绍了使用 **DSPy** 优化 **Prompt** 的方法。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1453027774305534054)** (7 messages): 

> `Transformer Implementations, Attention Autograd Shapes, Causal Masking, tinygrad.gradient` 


- **Transformer 实现：解码 Attention 形状之谜**：一位成员在阅读 **Transformer** 实现的 **Attention** 层级时遇到困难，特别是在形状（shapes）、**Causal Masking** 和 **Autograd** 方面。
   - 另一位成员以 **extra.models.bert** 为例提供了详细分解，重点关注 **BertSelfAttention** 内部的形状和操作。
- **Attention Autograd 形状：深度探讨**：该解释涵盖了 **BertEncoder**，它由 24 个隐藏层组成，每层包含 **BertSelfAttention** 和线性层。
   - **Hidden state** 的形状被描述为 **Batch×Length×Size**，然后为 **Query**、**Key** 和 **Value** 重新调整形状（reshape）为 **Batch×Heads×Length×Features**。
- **揭秘 Tensor.scaled_dot_product_attention**：分解说明了 `Tensor.scaled_dot_product_attention` 如何使用 **(query@key.T / √(Features) - attention_mask).softmax.dropout @ value** 计算 **Attention**。
   - 关键步骤包括为逐元素乘法和求和重新调整 **Query** 和 **Key** 的形状，以及应用 **Softmax** 和 **Dropout**，最终输出形状为 **Batch×Heads×Length×Features**。
- **揭秘 tinygrad.gradient 中的反向传播**：回复解释说梯度反向传播遵循常规的链式法则，其中数组的梯度与原数组具有相同的形状。
   - `tinygrad.gradient` 中的梯度规则详细说明了乘法、减法、求和以及广播（broadcasting）等操作，包括 `view` 和 `transpose` 如何影响梯度。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1453080427811176654)** (1 messages): 

> `AI/ML Learning, Collaboration Opportunities` 


- **成员寻求 AI/ML 学习协作**：一位成员表达了从基础开始学习 **AI** 和 **ML** 概念、数值分析及实现的兴趣。
   - 他们邀请其他人一起协作学习。
- **提议成立 AI 基础学习小组**：一位成员提议成立一个专注于 **AI/ML** 基础、数值分析和实现的学习小组。
   - 目标是协作学习和深入理解，欢迎任何感兴趣的人加入。