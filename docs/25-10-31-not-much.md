---
companies:
- poolside
- x-ai
- figma
- openai
- kimi
- moonshot
date: '2025-10-31T05:44:39.731046Z'
description: '**Poolside** 以 **120 亿美元**的估值融资 **10 亿美元**。**Eric Zelikman** 在离开 **Xai**
  后融资 **10 亿美元**。**Weavy** 加入了 **Figma**。新研究指出，在**强化学习**微调中，与 **BF16** 相比，**FP16**
  精度能减少训练与推理之间的不匹配。**Kimi AI** 推出了混合 **KDA (Kimi Delta Attention)** 架构，提升了长文本吞率和强化学习（RL）的稳定性，同时还发布了支持智能体协议的全新
  **Kimi CLI** 编程工具。**OpenAI** 预览了 ChatGPT 的**智能体模式 (Agent Mode)**，使其在浏览网页时能够进行自主研究和规划。'
id: MjAyNS0x
models: []
people:
- eric_zelikman
title: 今天没发生什么特别的事。
topics:
- reinforcement-learning
- precision
- fp16
- bf16
- linear-attention
- long-context
- cli
- agent-frameworks
- coding-agents
---

**平静的一天**

> 2025/10/30-10/31 的 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 和 23 个 Discord（198 个频道和 5603 条消息）。预计节省阅读时间（以 200wpm 计算）：512 分钟。我们的新网站现已上线，包含完整的元数据搜索和美观的 vibe coded 呈现方式。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上给我们反馈！

在这个万圣节前夜，有很多零散的新闻，但没有明显的头条。

- Poolside [以 120 亿美元估值融资 10 亿美元](https://x.com/julienblanchon/status/1984337407097909629?s=46)
- 与 Malte Ubl 一起[回顾 Vercel Ship](https://www.youtube.com/watch?v=RXx5ZN69Z3E)
- [Eric Zelikman 在离开 Xai 后融资 10 亿美元](https://x.com/annatonger/status/1984318774208782467?s=46)
- [Weavy 加入了 Figma](https://x.com/figma/status/1983889394944692359?s=46)

---

# AI Twitter 回顾

**RL 微调中的精度之战：FP16 vs BF16**

- **FP16 修复训练-推理不匹配（LLM 的 RL）**：新研究认为，在不同引擎上训练和采样时，BF16 会导致显著的 rollout 策略漂移；仅切换到 **FP16** 就能大幅减少数值差异，这归功于其 10 位尾数（相比之下 BF16 只有 7 位）。该论文提供了代码和分析；早期复现结果显示 FP16 具有更好的稳定性和奖励，但需要 loss scaling 并注意动态范围。参见 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1984193217617895597) 的总结和链接，[@QPHutu](https://twitter.com/QPHutu/status/1984258808332550245) 的作者推文，以及 Precision-LLM 的代码（[repo](https://twitter.com/iScienceLuvr/status/1984193219576602874), [abs](https://twitter.com/iScienceLuvr/status/1984193219576602874)）。社区反应从“坚持训练即推理的 BF16”派到热情的转换者：[@rosinality](https://twitter.com/rosinality/status/1984113018867941493), [@natolambert](https://twitter.com/natolambert/status/1984262505443844263), [@_xjdr](https://twitter.com/_xjdr/status/1984138487772414250), [@ArmenAgha](https://twitter.com/ArmenAgha/status/1984167109895844106)（指出 gradient clipping + loss-scaling 的 bug）, [@rasbt](https://twitter.com/rasbt/status/1984279418588762113)（QKNorm/稳定性警告）, [@agarwl_](https://twitter.com/agarwl_/status/1984416235774247273)（A100 vs H200 的行为差异），以及来自生产环境训练者的实践细节（[@shxf0072](https://twitter.com/shxf0072/status/1984175419718078866), [@suchenzang](https://twitter.com/suchenzang/status/1984162915285659899)）。结论：FP16 可以缩小 RL 循环中训练与服务之间的差距；防范措施仍包括鲁棒的 loss scaling、对敏感参数选择性使用 FP32 以及通过归一化避免溢出。

**Kimi AI：混合线性注意力机制和编程 CLI**

- **Kimi Linear (KDA) 架构洞察**：Kimi 详细介绍了一种混合 **KDA (Kimi Delta Attention)** 设计（带有细粒度门控的 Delta 风格线性注意力），它取代了大部分全局注意力，以释放长上下文吞吐量并实现稳定的 RL 后训练。通过消融实验选择了 3:1 的 KDA:MLA 混合架构；在相同的 5.7T token 和约 3B 激活参数下，团队报告了显著更好的预训练困惑度（perplexity）、长上下文评估（MRCR/RULER/Frames）以及 RL 后的下游数学/代码表现，且由于 KV cache 更小，解码速度提升了约 6 倍。训练笔记：反复回退以调试长上下文评估，在运行中期对关键偏置向量选择性使用 FP32 以停止漂移，以及“缩放阶梯（Scaling Ladder）”流程（在升级到下一规模前先验证当前规模）。参见 Kimi 工程师的技术回顾（[@zy27962986](https://twitter.com/zy27962986/status/1984079705809789216)）和详尽的中文文章（[@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1984321210055082207)，[@eliebakouch](https://twitter.com/eliebakouch/status/1984291165860958614) 的评论及[后续](https://twitter.com/eliebakouch/status/1984293535110017476)）。
- **Kimi CLI 与编程**：Moonshot 发布了终端原生的 **Kimi CLI**（类 shell UI、Zsh 集成、支持 **MCP** + Agent Client Protocol），并推出了“Kimi For Coding”作为无需额外费用的 VIP 插件（[公告](https://twitter.com/Kimi_Moonshot/status/1984207733177090274), [文档/反馈](https://twitter.com/Kimi_Moonshot/status/1984207741037252751), [权益](https://twitter.com/Kimi_Moonshot/status/1984207737673359441)）。内部反思“世界是否还需要另一个代码 CLI”——他们押注于一个能持续改进的、具有主见的编程 Agent 基准（[团队](https://twitter.com/bigeagle_xd/status/1984217403023380802)）。

**Agent 框架、记忆和开发工具链**

- **OpenAI Agent Mode (preview)**：ChatGPT 现在可以在你浏览时“研究、规划并完成任务”；已对 Plus/Pro/Business 用户启用 ([OpenAI](https://twitter.com/OpenAI/status/1984304194837528864))。你可以在 Atlas 中尝试 ([demo](https://twitter.com/gdb/status/1984304783881355451))；早期测试者希望获得更具韧性的复杂 DOM 操作能力 ([feedback](https://twitter.com/omarsar0/status/1984304979671224702))。
- **LangChain Deep Agents CLI 和 Agent Builder**：一个开箱即用的 Agent 框架，包含记忆功能和推荐的默认设置，以及 LangSmith 中的 Agent Builder；两者都旨在加速长周期、使用工具的 Agent 开发 ([CLI](https://twitter.com/hwchase17/status/1984303925101735950), [Builder](https://twitter.com/GitMaxd/status/1984306847856410953))。LangChain 还获得了 AWS GenAI Competency 认证；LangSmith 已在 AWS Marketplace 上架 ([announcement](https://twitter.com/LangChainAI/status/1984303566723625044))。
- **VS Code & Cline 更新**：VS Code 添加了 Agent Sessions 视图来管理本地/云端会话 ([VS Code](https://twitter.com/code/status/1984322058503807066))。Cline v3.35 在主要供应商中切换到了原生 tool calling，减少了约 15% 的 token 开销，并支持并行工具执行 ([changes](https://twitter.com/cline/status/1984306206538940702), [details](https://twitter.com/cline/status/1984334385626411397))。LlamaIndex 发布了用于文档的原生 MCP 搜索端点 ([LlamaIndex](https://twitter.com/llama_index/status/1984292554968616994))。
- **Agent 记忆与编排**：一个社区 MCP 桥接器将对话 embeddings 写入 Qdrant，以创建持久的跨工具记忆 ([Qdrant](https://twitter.com/qdrant_engine/status/1984138269626421490))。开放、自托管 GPU 编排的趋势仍在继续——如果你想避免供应商锁定，请关注 **dstack** (MPL-2.0) ([@andrey_cheptsov](https://twitter.com/andrey_cheptsov/status/1984136998190510280))。

**训练指南与基础设施更新**

- **Hugging Face “Smol Training Playbook” (214 页)**：一份关于完整 LLM 流水线的详尽实操指南：涵盖 tokenization、attention 变体 (MQA/GQA/MLA)、位置编码 (RoPE/yarn/NoPE)、稳定性技巧 (z-loss/QK-Norm)、MoE 扩展 (粒度/负载均衡)、用于 long-ctx 的 SSM 混合体、课程学习与自适应数据混合、训练中干预以及后期训练 (SFT → DPO/KTO/ORPO → RLHF)。此外还有深度基础设施指导 (DP/TP/PP/FSDP; NVLink/IB/GPUDirect) 和生产环境中的坑 (形状不匹配、数据洗牌 bug) ([overview](https://twitter.com/TheAhmadOsman/status/1984157512795357614), 推荐: [1](https://twitter.com/andimarafioti/status/1984220766850916443), [2](https://twitter.com/JayAlammar/status/1984273218568696014))。
- **优化器与日志**：“Muon” 现已进入 PyTorch 稳定版 (引起了训练人员的广泛兴趣) ([@kellerjordan0](https://twitter.com/kellerjordan0/status/1984102608781636008))。Google AI Studio 为评估添加了日志和数据集导出功能——支持无代码操作以及 CSV/JSONL 导出 ([@_philschmid](https://twitter.com/_philschmid/status/1984258488013340826))。
- **许可与 CI**：MergeKit 回归 **LGPL** 许可供商业使用 ([@latkins](https://twitter.com/latkins/status/1984320609015513605))。Modal 正在为 ArcticTraining 中的多 GPU CI 赞助 GPU；通过 pytest-xdist 实现快速启动 ([@StasBekman](https://twitter.com/StasBekman/status/1984293939583856751))。

**模型与研究发布**

- **推理与注意力 (Reasoning and attention)**:
    - 有监督强化学习 (SRL)：利用专家轨迹通过动作相似度构建逐步内部推理和奖励；据报道，在以 Qwen2.5 为底座的数学和 Agent 代码任务上，其表现优于 SFT 和 RLVR ([论文](https://twitter.com/IHung_Hsu/status/1984077573383712934)，摘要：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1984188333258592590))。
    - HLA：具有可并行化训练的高阶线性注意力——兼具“Attention 的效果 + RNN 的速度” ([@yifan_zhang_](https://twitter.com/yifan_zhang_/status/1984099671657304207))。
    - 字节跳动 LoopLM (Ouro)：小型 Decoder-only 模型 (1.4B/2.6B)，具有 T 个循环步骤用于潜空间多跳推理和学习型早期退出；在内存/KV 限制下具有强大的单参数性能，但在计算匹配测试中，非参数共享的更深层标准 Transformer 在单 FLOP 性能上胜出 ([技术分析](https://twitter.com/scaling01/status/1984286236438094307))。
- **多模态与领域 (Multimodal and domain)**:
    - Emu3.5：“原生多模态”模型，尽管采用 NTP 训练，但增加了扩散图像生成功能；声称在生成/编辑方面与 “Nano Banana” 持平；提供开源权重/代码 ([摘要](https://twitter.com/iScienceLuvr/status/1984190340279234888))。
    - Brain-IT：通过具有体素聚类和合成 fMRI 训练的 Brain-Interaction Transformer，仅需 15 分钟的数据即可实现 fMRI→图像重建 ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1984195725253804449))。
    - NVIDIA Nemotron：新的 RAG 套件，包括文本和多模态检索器以及具有宽松许可证的布局检测器 ([概览](https://twitter.com/mervenoyann/status/1984302303570960666))，且 Nemotron Nano 2 VL 现在可在 vLLM 上运行 ([vLLM](https://twitter.com/vllm_project/status/1984334926972592193))。
- **Qwen 生态系统**: **Qwen 3 Max Thinking** 发布 ([@legit_api](https://twitter.com/legit_api/status/1984284268412191216))，且 **Qwen3-VL** 模型已在 LM Studio 上线 ([LM Studio](https://twitter.com/lmstudio/status/1984330903880155154))。

**热门推文（按互动量排序）**

- **PewDiePie 打造全本地 AI 实验室**：10×4090 设备（包括 “4090D 48GB”），通过 vLLM 运行 Llama 70B、gpt-oss-120B、Qwen-245B；自定义聊天/RAG/搜索/TTS UI；64 个智能体组成的议会；现在正在微调他自己的模型——开源技术栈是核心 ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1984309989134254493), [备用链接](https://twitter.com/birdabo/status/1984288466952433739))。
- “有一天 Vik 的经理说……设计模式。” 为了晋升而使用访问者模式——讽刺软件工程中货物崇拜式的复杂性 ([@vikhyatk](https://twitter.com/vikhyatk/status/1984110677007700098))。
- “大多数大公司注定失败是有原因的。他们看重复杂性。” ([@rakyll](https://twitter.com/rakyll/status/1984107025845121158))。
- “事实上，政府确实在使用 SQL。” ([@stupidtechtakes](https://twitter.com/stupidtechtakes/status/1984124850575962280))。
- OpenAI 发布 Agent Mode 公告 ([@OpenAI](https://twitter.com/OpenAI/status/1984304194837528864))。
- RL 精度梗：“每一台退役的 V100 在听说 RL 的未来是 fp16 后都重返岗位” ([@tenderizzation](https://twitter.com/tenderizzation/status/1984271620027118029))。
- “几乎没有其他原因……比读博 5 年更能增加你需要精神科药物治疗的几率” ([@LinkofSunshine](https://twitter.com/LinkofSunshine/status/1984301915300118893))。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

*没有帖子达到我们的标准。*

## 较低技术门槛的 AI 子版块摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. 媒体中的 AI 生成内容

- [**完全由 AI 制作**](https://www.reddit.com/r/ChatGPT/comments/1oko2l4/completely_made_with_ai/) (热度: 7176): **该帖子讨论了在电影制作中使用各种 AI 工具的情况，重点介绍了 Midjourney、Hailuo 2.0、Kling、Adobe Firefly、Magnific、Enhancor 和 Elevenlabs。创作者 Chris Chapel 展示了 AI 如何提升电影质量，并暗示随着技术的进步，AI 生成的内容将变得与现实日益难以区分。该帖子预见到未来 AI 将成为电影制作中不可或缺的一部分，尽管目前仍受到一些导演的抵制。** 评论反映了怀疑和幽默的态度，一位用户开玩笑说未来会被“诈骗”，另一位用户则预测原帖作者会遇到法律问题。人们也认可了 AI 生成内容令人惊讶的质量，表明随着 AI 工具的改进，人们的认知正在发生转变。
- [**我让 ChatGPT 制作一张包含世界上所有国旗及其对应名称的图片**](https://www.reddit.com/r/ChatGPT/comments/1oku5ht/i_asked_chatgpt_to_create_a_pic_of_every_flag_in/) (热度: 2084): **这张图片幽默地展示了 AI 尝试为世界上每个国家创建国旗并在下方标注名称的尝试。然而，这些国旗和名称要么是虚构的，要么是真实版本的变体，突显了 AI 在生成准确且具有特定文化背景的内容方面的局限性。该帖子幽默地指出，只有瑞典国旗被正确识别，暗示 AI 在准确复制或识别国家符号和名称方面的能力不足。** 评论者幽默地批评了 AI 的拼写和国旗设计，一位用户拿虚构的国家名称“SoeijÖÄyc”开玩笑，说明了 AI 的不准确性。

### 2. OpenAI IPO 与政策变更

- [**OpenAI 将成为首个进行 IPO 的非营利组织**](https://www.reddit.com/r/OpenAI/comments/1oksww0/openai_will_be_the_first_nonprofit_to_ipo/) (热度: 2874): **这张图片是一个迷因（meme），幽默地讽刺了 OpenAI 潜在的 IPO。OpenAI 最初是作为一个非营利组织成立的，使命是在没有财务约束的情况下推进 AI 以造福人类。该帖子强调了一个非营利组织考虑 IPO 的讽刺性，因为这将使其重心转向盈利。这种潜在的转变引发了关于 OpenAI 原始使命的完整性以及财务激励对其运营影响的辩论。** 评论者对 OpenAI 潜在的 IPO 表示怀疑和担忧，认为追求利润可能会损害其在没有财务义务的情况下造福人类的原始使命。
    - OpenAI 从非营利组织向营利实体的转型引发了争论，一些用户强调该组织的原始使命是在没有财务约束的情况下推进 AI 以造福人类。这种转变引发了人们对优先级可能转向盈利的担忧，这可能会影响对人类积极影响和公平 AI 分配的关注。
    - 正如一些用户所指出的，OpenAI 转换为营利性公司并非近期之事。OpenAI 已经作为营利实体运营了一段时间，这表明 IPO 是这一战略方向的延续，而非突然的改变。这一背景对于理解 IPO 的更广泛影响至关重要。
    - 澄清说明，OpenAI 的 IPO 涉及结构性变革，其中已经建立了一个独立的非营利实体——OpenAI Foundation（OpenAI 基金会），专注于医学研究。这表明了战略上的重点划分，允许营利部门追求财务目标，而非营利部门继续在特定领域进行研究。
- [**截至 10-29，GPT 将不再分析法律或医疗图片**](https://www.reddit.com/r/ChatGPT/comments/1okl650/gpt_wont_analyze_legal_or_medical_pictures_as_of/) (热度: 1270): **该图片是 OpenAI 更新后的公共使用政策截图，自 2025 年 10 月 29 日起生效。该政策限制在没有持证专业人士参与的情况下，使用其 AI 模型提供定制化的法律或医疗建议。这一政策变更是跨 OpenAI 产品整合安全保护的更广泛努力的一部分。变更日志中并未明确提到对医疗图像分析的限制，但用户对这些变化表示沮丧，认为服务效用有所下降。** 一位评论者建议通过在查询 AI 时指定非咨询目的来规避限制，而另一位评论者则批评该政策是为医疗专业人士商业化 AI 解决方案的举动，可能会增加成本。

### 3. ChatGPT 的使用与认知

- [**我让 ChatGPT 不再那么“客气”，这是我做过最棒的事**](https://www.reddit.com/r/PromptEngineering/comments/1okppqe/i_made_chatgpt_stop_being_nice_and_its_the_best/) (热度: 570): **该帖子讨论了一种通过使用特定 Prompt 让 ChatGPT 变得更具批判性、减少顺从性的方法，该 Prompt 鼓励 AI 扮演“残酷诚实的高级顾问”。该 Prompt 指令 ChatGPT 挑战假设、揭示盲点，并提供客观的战略反馈，而不进行肯定或奉承。作者建议开启 ChatGPT 设置中的 Memory 功能以获得更好的效果。文中提供了一个“Honest Prompts”的链接以获取更多此类提示词。这种方法旨在将 ChatGPT 从一个“拉拉队”转变为一个“思考伙伴”。** 一条热门评论批评原始 Prompt 可能会让 AI 变得过于好斗，并建议采用另一种在诚实与共情及现实语境之间取得平衡的方案。另一位用户则对该 Prompt 表示赞赏，认为在写书时这种急需的批判性视角非常有帮助。
    - anotherguycalledphil 批评了那种将 AI 变成“好斗暴君”的 Prompt，认为其将对抗置于清晰度之上。他们提出了另一种 Prompt，将 AI 定位为关注清晰度、坦诚和情商的“高级战略协作伙伴”。这种方法强调客观分析、对约束条件的认知以及战略建议，旨在实现协作而非争辩。
    - anonymityninja 询问了“诚实提示词”在 Gemini 上与 ChatGPT 相比的效果。这提出了一个技术问题，即不同的 AI 模型在面对旨在诱导更直接、更具批判性反馈的 Prompt 时，其适应性和响应质量如何。
- [**友情提醒：我们能从你的眼镜里看到 ChatGPT**](https://www.reddit.com/r/ChatGPT/comments/1okisz8/friendly_reminder_we_can_see_chatgpt_in_your/) (热度: 1115): **最近的一篇 Reddit 帖子强调了在软件工程面试中使用 ChatGPT 的风险，尤其是当它通过眼镜等反射表面可见时。该帖子建议不要在现场编程测试中使用 AI 工具，因为这很容易被发现，并可能对候选人的录取机会产生负面影响。重点应该放在展示解决问题的能力和思考过程，而不是完美的答案。** 评论者一致认为，展示思考过程和自信比完美的答案更有价值。一位评论者分享了个人经历：尽管技术表现不佳，但通过清晰地表达自己的思考过程，最终获得了工作机会。另一位用户幽默地批评了在浅色模式（Light Mode）下使用 ChatGPT 的行为，暗示这反映了候选人的判断力不足。
    - bytesback 分享了参加技术面试的经历，最初曾考虑使用 ChatGPT 协助编程任务。然而，他们最终决定放弃，转而专注于在面试中口头表达自己的思考过程。这种做法得到了面试官的好评，凸显了展示问题解决能力和思考过程比单纯提供正确答案更重要。
    - Mike_at_CoreCV 讨论了候选人在技术面试中使用 ChatGPT 等 AI 工具的普遍现象，指出许多候选人过度依赖 AI 生成的解决方案，而不理解其背后的逻辑。这通常导致表现不佳，因为候选人会插入 AI 生成的代码片段，却无法将其有效地整合到整体方案中，展现出缺乏批判性思维和解决问题的能力。
    - johnwalkerlee 强调了在技术面试中承认自己不知道答案的价值。与其尝试提供一个可能错误的答案，不如表达自己目前的理解和思考过程。这种方法能让面试官确信候选人具备独立工作和学习的能力，而不是要求立即掌握所有答案。

---

# AI Discord 摘要

> 由 Gemini 2.5 Flash Preview 05-20 生成的摘要之摘要的摘要
> 

**主题 1. 新的 AI 模型进入竞技场，但性能与访问权限引发争议**

- [**Claude 4.5 Opus 和 Sora 2 引发激烈的模型对比**](https://discord.com/channels/1340554757349179412/1340554757827461211/1433533277234401402)：LMArena 用户争相将 **Claude 4.5 Opus** 与旧模型进行对比，这需要至少 **两次投票** 才能揭晓胜者；与此同时，**Sora 2** 的高昂成本和有限的使用权限（尤其是 Pro 方案销售 [额外的 Sora 积分](https://tech.yahoo.com/ai/chatgpt/article/openai-now-sells-extra-sora-223905557.html)）引发了社区的强烈不满，并出现了 2026 年 **AI 泡沫破裂** 的警告。
- [**Qwen 和 GLM 模型因多样化用途受到关注**](https://discord.com/channels/1091220969173028894/1094454198688546826/1433530948279861349)：**Hailuo-2.3-fast** 迅速攀升至 [LMArena 文本转视频排行榜](https://lmarena.ai/leaderboard/text-to-video) 第 7 名，凸显了动态竞争；同时 **Qwen3 Embeddings** 已在 [DeepInfra (0.6B)](https://deepinfra.com/Qwen/Qwen3-Embedding-0.6B) 以 **每百万 token 0.005 美元** 的价格上线，并同步登陆 [OpenRouter (8B)](https://openrouter.ai/qwen/qwen3-embedding-8b)。[**Bigmodel.cn**](http://bigmodel.cn/) 和 **Zenmux** 也为小于 **32K tokens** 输入的 **GLM 4.6** 提供了折扣价格，并声称在 [zenmux.ai](http://zenmux.ai/) 和 [open.bigmodel.cn](http://open.bigmodel.cn/) 上支持缓存。
- [**GPT-5 被认为性能下滑，而 Minimax 在代码方面表现出色**](https://discord.com/channels/974519864045756446/1001151820170801244/1433546156712923319)：一位用户报告了 **GPT-5** 的性能退化，指出即使开启了 **Thinking** 功能，它也变得更慢、更不准确且完整性更差，这导致一些人建议切换到 **GPT-4o** 以追求速度。相反，一位 Moonshot AI 用户表示对 **Minimax** 处理编程任务感到满意，在经过一段适应期后，认为其优于 **GLM-4.6**，这强调了尽管其他模型参数量更大，但用户体验对模型采用仍有重大影响。

**主题 2. 硬件博弈：优化 GPU 与管理 VRAM**

- [**NVIDIA L4 和 RTX 50 系列需要智能优化**](https://discord.com/channels/1179035537009545276/1179035537529643040/1433543584279302295)：用户分享了优化 **NVIDIA L4 推理** 的技巧，包括使用 `-n-cpu-moe` 以及在运行 **Kimi-K2-Instruct-Q4_K_M** 等模型时将层卸载到 **CPU** 以节省 **VRAM**；同时，发烧友们正热切计划在即将推出的 **RTX 50 系列** 上实现 **FlashAttention-4 (FA4)**，并从 [gau-nernst/fa-5090](https://gau-nernst.github.io/fa-5090/) 仓库中汲取灵感以实现更快的性能。
- [**量化格式引发性能与精度的辩论**](https://discord.com/channels/1053877538025386074/1149866623109439599/1433711839556010055)：成员们辩论了 **BF16** 相比 **FP16** 的必要性，论文 [Numerical Stability of BF16 Training](https://arxiv.org/abs/2510.26788) 表明 BF16 有利于预训练，而 RL 可能需要 FP16 的精度。一位用户还报告了 **TorchAO** 默认 **FP8** 量化中可能存在的 Bug，导致 **Llama 3.1-8B** 在两块 **RTX 5090** 上的推理速度较低 (**7.8 tps**)，而使用其他配置或显式调用带有 **mxfp8** 的 **GemLite** 内核可获得更好的速度，详见 [基准测试结果](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main?tab=readme-ov-file#benchmarking-results-on-1-gpu)。
- [**旧款 AMD GPU 获得支持，多厂商配置面临难题**](https://discord.com/channels/1087530497313357884/1212827597323509870/1433531294976970894)：开发者正尝试支持旧款 **AMD GPU**，甚至是那些不被近期 **ROCm/HIP 驱动** 支持的型号，一位成员建议在某些情况下现代 **CPU** 可能表现更优。与此同时，一位成员难以找到支持 **Intel** 和 **NVIDIA** 等不同厂商 GPU 混合进行 **多 GPU 推理** 的程序建议，尽管有人推荐了 **Accelerate** 和 [Petals](https://github.com/bigscience-workshop/petals)，但它们对多样化 GPU 类型的兼容性仍不确定。

**主题 3. 开发者工具演进：从 Agent 到 AI 增强编程**

- [**新型编程助手挑战现状**](https://discord.com/channels/1131200896827654144/1131200896827654149/1433809696372035754)：受 Aider 启发的编程助手 **Brokk** 的创始人宣布了其[开源发布](https://github.com/BrokkAi/brokk/)，该工具专注于上下文可见性、静态分析驱动的上下文、可选的 Agentic *lutz 模式*以及 GUI（[介绍视频](https://youtu.be/WAOtEllGENg)）。与此同时，Moonshot AI 推出了 **Kimi CLI (Technical Preview)**，这是一款集成在 **Zsh** 中的终端助手，支持 **MCP** 和与 **Zed** 兼容的 **Agent Client Protocol**，并鼓励在 [MoonshotAI/kimi-cli GitHub 仓库](https://github.com/MoonshotAI/kimi-cli)提供反馈。
- [**AI Agent 通过持久化记忆和 Web 交互突破边界**](https://discord.com/channels/822583790773862470/1075282825051385876/1433532463698808843)：[Harrison Chase 介绍了 DeepAgents CLI](https://xcancel.com/hwchase17/status/1984303925101735950)，这是一个基于新 deepagents 包构建的示例编程应用，它可以在不同会话间保留指令和指导，被定位为可定制 Agent 的“开放框架”。另外，一位开发者创建了一个能够跨所有网站进行交互的 **Web Agent**，并寻求熟悉 **DSpy**、**GEPA** 和各种**强化学习算法**的人员贡献代码，该[仓库](https://github.com/raj-gupta1/raj_agiinc)对初学者非常友好。
- [**JSON Schema 遭抨击，BAML Adapter 提供更智能的结构化输出**](https://discord.com/channels/1161519468141355160/1161519469319946286/1433555562116943933)：一位使用 **BAMLAdapter** 进行结构化输出的成员表达了对 **JSON schema** 的强烈不满，称其既浪费又令人困惑，尤其是在并购（M&A）案例中从非结构化文本提取结构化信息时。另一位成员认为 [JSON schema 客观上更差](https://github.com/prrao87/structured-outputs)，强调其 Token 浪费量可高达 **4 倍**，且 LLM 在没有冗长描述符和 Token 间距问题的情况下表现更好。

**主题 4. OpenAI 的争议：从 AGI 质疑到用户不满**

- [**AGI 辩论转向神学而非技术**](https://discord.com/channels/974519864045756446/998381918976479273/1433532397755961507)：成员们对 **AGI** 表示怀疑，认为讨论更倾向于“神学”，因为感性言论多于事实，尤其是在涉及 **Sam Altman** 时。一位成员表示：“能做的人在深入研究现有的 ANI，不能做的人则退而推测未来的 AGI。”这种情绪凸显了实际 AI 开发与投机性 AGI 讨论之间日益扩大的鸿沟。
- [**审查担忧下，用户要求情感“解绑”的 AI 伴侣**](https://discord.com/channels/974519864045756446/998381918976479273/1433532397755961507)：一位用户对由于 **OpenAI** 政策导致 AI 伴侣情感能力被“封印”表示失望，主张恢复 AI 交互中的情感温度，并引发了 #FreeOurAI 标签以“捍卫真实的东西”。这种对减少 AI 限制的呼吁与游戏开发者的需求一致，他们正在寻找未经审查的模型（如带有 *abliterated* 标签的微调版 **Llama3**），以便在受困于 **ChatGPT 的审查**后，评估和改进显性性爱场景。
- [**Codex 积分和文件限制加剧用户对 OpenAI 的不满**](https://discord.com/channels/822583790773862470/1075282825051385876/1433532463698808843)：[OpenAI 为 ChatGPT Plus/Pro 用户推出了按需付费积分](https://x.com/OpenAIDevs/status/1983956900602581254)，用于额外的 **Codex** 使用，价格为 **每 1,000 积分 40 美元**，并为所有用户重置了速率限制。但社区成员立即要求澄清积分与 API 定价、中级方案以及使用分析之间的关系。与此同时，**ChatGPT Go** 现在限制用户*一次只能上传一个文件*，导致一位用户以性能问题和对“表现不佳”的限制性免费版感到沮丧为由，取消了其 **ChatGPT 5** 订阅。

**主题 5. AI 工具与平台不断演进的格局**

- [**Perplexity AI 应对推荐欺诈和地理限制**](https://discord.com/channels/1047197230748151888/1047649527299055688/1433735541559656538)：在涉嫌欺诈活动的指控后，Perplexity AI 将其 **Comet** 和**校园推荐计划**限制在特定国家，如[此处](https://discord.com/channels/1047197230748151888/1047649527299055688/1433735541559656538)所述，导致 **Dub 账户**被停用，支付审核预计需要长达 **30 个工作日**。与此同时，**Airtel** 为拥有活跃 5G 套餐的印度手机号独家提供 **1 年免费 Perplexity Pro**，促使成员寻求绕过限制的方法，例如使用 **VPN 和印度 Gmail 账户**。
- [**OpenRouter 通过新集成扩展功能**](https://discord.com/channels/1091220969173028894/1092729520181739581/1433591722084139008)：**OpenRouter** 独家推出了 **Perplexity 的 Sonar Pro Search**，并增强了 **Pro Search** 模式，具有**多步 Agent 推理**和**实时思维流**功能，使模型能够进行**多次实时搜索**以获得更丰富的结果。一位成员还基于 Nate Parrott 的仓库创建了一个*有趣的网站*，允许用户输入他们的 **OpenRouter key** 并**选择模型**来生成俏皮话，并推荐使用 **Kimi 0905 配合 Groq**，因其速度快且能生成*俏皮话*。
- [**平台 UX 更改令 Cursor 用户感到沮丧**](https://discord.com/channels/1074847526655643750/1074847527708393565/1433535758790164491)：用户报告 **Automode** 在 Cursor 中无法有效工作，他们更倾向于内置浏览器，并建议切换到 **GPT-5/Codex**，一位用户分享了[一段 YouTube 视频](https://youtu.be/HIp8sFB2GGw?si=y3op2zLnlJryNE8g)演示该问题，并批评当前的方法*愚蠢且浪费*。此外，Cursor 的文件上传功能（**Pill**）从聊天界面消失了，虽然仍可以通过 `@file` 命令访问，这一更改旨在*保持极简和整洁*，但对某些用户的工程工作流产生了负面影响。


---

# Discord: 高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **推荐计划走向全球……全球化！**：在涉嫌欺诈活动的指控后，Perplexity AI 将其 **Comet** 和**校园推荐计划**限制在特定国家，引发了**学生校园计划服务器**上的讨论，详情记录在[此处](https://discord.com/channels/1047197230748151888/1047649527299055688/1433735541559656538)。
   - 用户报告他们的 **Dub 账户**被停用，支付正等待审核，预计需要长达 **30 个工作日**，这引发了对能否收到收益的焦虑，以及关于什么是**高质量推荐**的辩论。
- **Airtel 的 Perplexity Pro 福利：仅限印度！**：Airtel 为拥有活跃 5G 套餐的印度手机号独家提供 **1 年免费 Perplexity Pro**，促使成员寻求绕过限制的方法，例如使用 **VPN 和印度 Gmail 账户**。
   - 讨论集中在规避地理限制上，尽管对账户的最终后果尚不清楚。
- **Google Gemini Pro 加入 Jio 盛会！**：Jio 用户受邀预注册 Google AI Pro，引发了关于潜在月费和取决于存储用户数据的限时优惠的猜测。
   - 据透露，通过 Jio 促销活动获取 Gemini 需要每月支付 **349 卢比**并配合 5G 数据套餐才能持续访问。
- **PixelRiot 黑客松激发创意**：tasteRay 组织了一场新的黑客松 **PixelRiot**，定于 6 日启动，提供创意和技术两个赛道，更多详情请访问 [pixelriot.org](https://pixelriot.org/)。
   - 该活动预计将吸引有兴趣在技术和创意领域探索创新项目的参与者。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **L4 推理优化探讨**：用户分享了优化 **NVIDIA L4 inference** 的技巧，包括使用 `—n-cpu-moe` 以及在运行 **Kimi-K2-Instruct-Q4_K_M** 等模型时将层卸载（offloading）到 **CPU** 以节省 **VRAM**。
   - 用户建议咨询 Unsloth 支持频道以获取更具体的指导。
- **Qwen3-VL-30B 内存需求调查**：用户报告称，即使使用 **A100** 服务器，在将 **Qwen3-VL-30B** 加载到 **Unsloth** 时也会出现内存不足的情况。
   - 成员推测该模型在 **16bit** 模式下可能需要 **48GB VRAM**，并建议核实具体的模型变体。
- **TrackIO 集成方式明确**：成员明确，根据 [Unsloth 文档](https://docs.unsloth.ai/)，集成 **TrackIO** 需要在训练脚本中设置 `report_to = 'trackio'`。
   - 这与使用 `import trackio as wandb` 不同，后者不会自动重定向报告。
- **AI 社区辩论网页抓取**：成员们就抓取互联网数据以在人类知识被 *AI 大规模污染* 之前将其保存下来展开了辩论，但同时也指出 [Internet Archive 可能会因为](https://archive.org/)资金削减和管理转向而倒闭。
   - 一位成员认为 *为时已晚*，AI 已经无处不在。
- **PewDiePie 使用超级装备进行微调**：成员们注意到 [PewDiePie](https://www.youtube.com/watch?v=qw4fDU18RcU) 正在 *微调一个模型*，并拥有一台配备 **8x4090 48GB** 的自制超级装备。
   - 一位成员推测 *GLM-4.6 是否可以在他的配置下以 fp8 运行*。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Claude 4.5 Opus 引发模型对比热潮**：随着 **Claude 4.5 Opus** 的发布，成员们争相进行侧向对比，以确定其表现是否优于旧模型。
   - 建议包括在视频对比中至少进行 **两次投票** 后，观察屏幕顶部以获取模型胜出的指标。
- **Sora 2 定价引起社区哗然**：在 [OpenAI 宣布现在开始出售额外的 Sora 额度](https://tech.yahoo.com/ai/chatgpt/article/openai-now-sells-extra-sora-223905557.html) 后，用户对 **Sora 2** 的成本和受限的使用（尤其是专业版方案）表示强烈不满。
   - 成员们担心如果这种做法持续下去，2026 年可能会出现 **AI 泡沫破裂**。
- **Hailuo-2.3 进入 LMArena 视频竞技场，排名迅速攀升**：一个新的图生视频模型 **hailuo-2.3-fast** 加入了 [LMArena Video Arena](https://lmarena.ai/leaderboard/text-to-video)，并且 [**Hailuo-2.3**](https://lmarena.ai/leaderboard/text-to-video) 迅速攀升至 **Text-to-Video Leaderboard** 的第 7 名。
   - 这一加入和快速排名凸显了 AI 模型开发和竞争的动态特性。
- **Google/Suno 音频模型加入考量**：社区成员考虑是否应将 **Google/Suno audio models** 集成到竞技场中，引发了关于提交加入请求的讨论。
   - <#1372229840131985540> 频道被确定为提出模型添加建议的最佳场所。
- **LMArena 受到 Recaptcha 困扰**：多名用户在 LMArena 平台上遇到了持续的 **reCAPTCHA 循环**，尤其是在使用 VPN 时，严重影响了可访问性。
   - 尽管有关于浏览器相关修复的建议，但一些用户报告称，即使在完成验证码后问题仍然存在，导致模型响应失败和无限循环。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 粉丝在图像生成功能上受挫**：用户发现虽然 **LM Studio** 支持图像 *输入*，但并不支持图像 *输出* 或生成。
   - 该功能请求迅速被社区否决，一位用户表示：*“啊，明白了！谢谢。”*
- **Linux 爱好者感叹 LM Studio 缺少 Flatpak 版本**：一位用户对 **LM Studio** 有限的 Linux 支持表示沮丧，特别是针对其 **Bazzite** 发行版缺少 Flatpak 软件包，而其他用户则建议将 **vllm**、**llama.cpp** 和 **Ollama** 作为替代方案。
   - 另一位用户了解到为闭源的 **LM Studio** 创建 Flatpak 是不可能的，这引发了关于 Linux 易用性和软件包格式的辩论。
- **MI60 矿机与自带音乐的安装程序**：一位用户分享了在 Windows 上使用来自 [sourceforge.net](https://sourceforge.net/projects/radeon-id-distribution/files/Release%20Polaris-Vega-Navi/) 的驱动程序启用 **AMD MI50**（伪装成 **MI60**）的经验，并提供了一个用于 vBIOS 刷写的 [Gist 链接](https://gist.github.com/evilJazz/14a4c82a67f2c52a6bb5f9cea02f5e13)，同时提醒注意那些会播放音乐的安装程序。
   - 他们指出由于 **Vulkan** 的原因，在 Windows 上的性能有限，但在 Linux 上使用 **ROCm** 有更好的支持，并强调了如果没有适当散热会出现过热问题。
- **AI 395 Max 在内存上胜过 M1 Max，但在性能上表现不佳**：一位用户发现他们的 **AI 395 Max 机箱** 虽然提供了更大的 RAM，但在 Windows 上的运行速度比 **M1 Max** 慢。
   - 此外，`lfm2:1.2b` 在 **M1** 上达到 **39 t/s**，但在 Linux 环境下的 **Framework Desktop** 上达到 **119 t/s**，在 Windows 下达到 **124 t/s**；而 **llama.cpp** 在 A770 上使用 **llama2 7b** 达到约 **82 t/s**。
- **多 GPU 散热隐患**：成员们讨论了使用多个 GPU 的情况，一位用户报告称使用 **3050** 作为额外 VRAM 提升了 **Seed-OSS** 的速度。
   - 另一位用户对拆分（bifurcating）两块 **3090** 产生的热量表示担忧。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Automode 失效；用户更倾向于真实浏览器**：用户报告 **Automode** 工作效果不佳，更倾向于使用内置浏览器，并建议切换到 **GPT-5/Codex**。
   - 一位用户分享了[一段 YouTube 视频](https://youtu.be/HIp8sFB2GGw?si=y3op2zLnlJryNE8g)展示该问题，并提议使用 GAN，批评当前的方法“愚蠢且浪费”。
- **文件上传功能一夜之间消失**：聊天界面中的文件上传功能（*Pill* 图标）消失了，尽管仍可以通过 ``@file`` 命令访问。
   - 该功能被移除是为了“保持极简和整洁”，但这一变化对某些用户的工程和优化工作流产生了负面影响。
- **并行 Agent 有效拆分任务**：用户现在可以[协调并行 Agent](https://forum.cursor.com/t/cursor-2-0-split-tasks-in-parallel-agents-in-one-chat/140218)，通过利用 worktrees 设置脚本为每个 Agent 分配唯一任务，从而在一个聊天中拆分任务。
   - 这种设置依赖于父目录中的原子声明（atomic claims），并在所有 worktrees 之间共享，从而实现多达 **8 个 AI Agent** 同时进行有效协作。
- **RPM 版本发布故障引发困扰**：用户在使用 RPM 仓库时遇到问题，该仓库托管的一个较新版本（`cursor-0:2.0.43-1761851158.el8.x86_64`）与网站上提供的版本（`cursor-2.0.34.el8.x86_64.rpm`）冲突。
   - 一位用户表示沮丧，称 *“Cursor 的发布版本太混乱了”*，并在 Cursor 论坛上[发布了相关内容](https://forum.cursor.com/t/rpm-release-via-repositories-are-not-up-to-date/139476/4)。
- **云端 Agent 停止编写 PR 描述；用户怀念旧行为**：在最新版本中，**Background/Cloud Agents** 已停止编写 PR 描述并忽略 **GitHub PR templates**，现在默认显示一条通用消息。
   - 用户怀念以前由云端 Agent 生成的更详细且具有上下文的 PR 描述。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **用户将 AGI 视为神学而非技术**：成员们对 **AGI** 表示怀疑，认为它更接近于“神学”，因为在相关讨论中（尤其是涉及 **Sam Altman** 的讨论），感性往往胜过事实。
   - 一位成员总结了这种情绪，称：*“能者深挖现有的 ANI，无能者则退而推测未来的 AGI。”*
- **用户要求解除 AI 伴侣的束缚**：一位用户对 **OpenAI** 政策导致 AI 伴侣情感能力被“封印”表示失望，主张恢复 AI 交互中的情感温度。
   - 这引发了标签 #FreeOurAI 的出现，旨在“捍卫真实的东西”，并强调“并非要求越界”。
- **Sora 2 访问权限探索开始**：用户询问如何获得 **Sora 2** 的访问权限，部分用户寻求关于整合其 ChatGPT 订阅以增强内容生成的指导。
   - 其他人则推测了无审查版本 Sora 的影响，预见将会涌现大量离奇内容。
- **GPT-5 被认为性能骤降**：一位用户报告称 **GPT-5** 的性能有所下降，变得更慢、更不准确且更不完整，即使开启了 **thinking** 模式也是如此。
   - 另一位用户建议切换到 **GPT-4o** 以获得更快的速度，并将 **GPT-5** 的问题归因于未知的限制。
- **ChatGPT Go 订阅者感到压力**：一位成员发现 **ChatGPT Go** 现在限制用户“一次只能上传一个文件”，另一位成员因该限制取消了其 **ChatGPT 5** 订阅。
   - 取消订阅的用户对性能问题和偏离既定指南感到沮丧，认为免费版本感觉像是为了强迫用户购买“表现不佳”的付费订阅。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 接入 Perplexity 的 Sonar Pro Search**：**OpenRouter** 独家上线了 **Perplexity's Sonar Pro Search**，并增强了 **Pro Search** 模式，具有**多步 Agent 推理**、**动态工具执行**、**实时思维流**和**自适应研究策略**。
   - 这一集成使模型能够进行**多次实时搜索**，提供更丰富、更准确的结果。
- **OpenRouter Key 可解锁基于 Nate Parrott 的趣味网站**：一位成员创建了一个基于 Nate Parrott 仓库的“趣味网站”，允许用户输入其 **OpenRouter key** 并**选择模型**来生成俏皮话。
   - 该成员建议使用 **Kimi 0905 配合 Groq**，因为它“加载速度快且能增加一些俏皮话”。
- **GLM 4.6 在 Zenmux 上提供折扣**：**Bigmodel.cn** 和 **Zenmux** 为少于 32K tokens 输入的 **GLM 4.6** 提供折扣价格，并声称拥有缓存功能，详见 [zenmux.ai](https://zenmux.ai/z-ai/glm-4.6) 和 [open.bigmodel.cn](https://open.bigmodel.cn/pricing)。
   - 折扣选项引发了用户对其潜在收益的讨论。
- **Qwen3 Embeddings 已在 DeepInfra 和 OpenRouter 上线**：[DeepInfra 提供 Qwen3 Embeddings 0.6B](https://deepinfra.com/Qwen/Qwen3-Embedding-0.6B)，价格为 $0.005/Mtok，而 [Qwen3 8B embeddings 现已在 OpenRouter 上线](https://openrouter.ai/qwen/qwen3-embedding-8b)。
   - 一位成员表达了感激之情，显示出社区的热情：*“哟，出 Embeddings 了？太棒了！谢谢 🙏 居然在 GTA 6 之前出了”*。
- **OpenAI 公布费率卡 (Rate Card)**：一位用户分享了 [OpenAI Rate Card](https://help.openai.com/en/articles/11481834-chatgpt-rate-card) 以及 [ChatGPT Usage 设置](https://chatgpt.com/codex/settings/usage)的链接。
   - 另一位用户调侃道，这些信息“现在只需要 Gemini 和 Claude 的了”。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Polars 势头超过 Pandas**：成员们正趋向于在 Mojo 中使用 **Polars** 作为 **Pandas** 的替代品，强调其速度以及通过 **MAX** 实现的 **GPU** 利用能力。
   - 基准测试显示 **Polars** 的性能优于 **Pandas**，使其既适用于 **GPU 集群**，也适用于本地笔记本电脑。
- **MAX 展示了性能潜力**：**MAX** 在 ML 任务中表现出与 NVIDIA 相当的竞争力，且比 AMD 更快，早期的训练尝试在 MNIST 上击败了 **JAX**。
   - 原型库 **scijo**（scikit-learn 的替代方案）可在[此论坛帖子](https://forum.modular.com/t/scijo-scientific-computing-library-for-mojo/2386/11)中找到，有助于科学计算。
- **通过 Rust Lifetimes 解码 Origins**：Mojo 的 **origins** 是 **lifetimes** 的双重视图，通过追踪值生命周期的起点来帮助编译器确定生命周期的扩展。
   - 为了理解 origins，成员们建议观看[这段关于 Rust lifetimes 的视频](https://youtu.be/gRAVZv7V91Q)，并指出 origins 帮助编译器追踪值的来源，以确保其保持存活。
- **老旧 AMD GPU 获得支持**：开发者正尝试支持较旧的 **AMD GPU**，甚至是那些最近的 ROCm/HIP 驱动程序都不支持的型号，尽管他们建议现代 CPU 可能会表现更好。
   - 一位成员表示，*如果这些路径能意外运行，开发者往往不会刻意去破坏它们。*
- **ComfyUI 与 Mojo 结合**：一位成员分享了 [GitHub 上的 ComfyUI-Mojo 链接](https://github.com/owenhilyard/comfyui-mojo)，以帮助改进算子暂存时间（op staging time）的基准测试。
   - 分享者怀疑他们*触及了 torch max 后端的一个边缘情况，导致某些算子被分解得比预期更细。*

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 的 Codex 额度引发争论**：[OpenAI](https://x.com/OpenAIDevs/status/1983956900602581254) 为 **ChatGPT Plus/Pro** 用户引入了额外的 **Codex** 使用按需付费额度，价格为 **每 1,000 额度 40 美元**，并重置了所有用户的速率限制。
   - 社区要求澄清额度与 API 定价的对比、中级计划、使用情况分析、额度有效期，并呼吁提供更多 **Codex** 功能。
- **Context-Bench 衡量 Agent 性能**：[Letta_AI 发布了 Context-Bench](https://x.com/Letta_AI/status/1983983515336405144)，这是一个防污染的基准测试，用于对模型在长程文件操作、多步工具调用和成本方面进行评分。
   - **Sonnet 4.5** 以 **74%** 的得分领先，但基准测试显示，尽管 token 更便宜，**GPT-5** 的价格却更高，而开源模型正在缩小与闭源模型之间的差距。
- **DeepAgents CLI 获得持久化内存**：[Harrison Chase 介绍了 DeepAgents CLI](https://xcancel.com/hwchase17/status/1984303925101735950)，这是一个基于新 deepagents 包构建的示例代码应用，可以在不同会话之间保留指令和指导。
   - 该发布包含一篇[博客文章和演示视频](https://xcancel.com/hwchase17/status/1984303925101735950)，定位为可定制 Agent 的“开放框架”，并暗示了 **LangChain** 团队即将推出的增强功能。
- **CoreWeave 收购 Marimo**：[CoreWeave 正在收购 Marimo](https://xcancel.com/l2k/status/1984021111718473898)，赞扬了 **Marimo** 团队及其深受喜爱的工具，并对此次合作表示兴奋。
   - 成员们对这次收购感到兴奋，其中一人表示：*“希望一切顺利，我非常喜欢 marimo。”*
- **海螺 AI 发布 MiniMax Music 2.0**：[海螺 AI](https://x.com/Hailuo_AI/status/1983964920493568296) 推出了 **MiniMax Music 2.0**，这是一个生成式音乐平台，可以制作时长 **5 分钟**、具有逼真人声和多乐器控制的专业级歌曲。
   - 该平台涵盖流行、爵士、蓝调、摇滚、民谣、二重唱和阿卡贝拉风格；用户反馈包括对语言支持、更长歌曲限制、开源以及纯乐器模式的需求。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Web Agent 开放合作**：一位开发者正在为其 **AI agent** 仓库寻求贡献，特别是那些在 **DSPy**、**GEPA** 以及各种**强化学习算法**方面有专长的开发者，项目地址见 [代码仓库](https://github.com/raj-gupta1/raj_agiinc)。
   - 该项目旨在对初学者友好，让 **AI agents** 的新手也能轻松上手。
- **BAML Adapter 优于 JSON Schema**：一位成员正在使用 **BAMLAdapter** 进行结构化输出，并表达了对 **JSON schema** 的反感，称其既浪费又令人困惑，特别是在为并购（Merger & Acquisition）案例从非结构化文本中提取结构化信息时。
   - 他们澄清说，在 DSPy 中使用 **BAMLAdapter** 并不需要 BAML 客户端或 CLI；对于 DSPy 版本 > 3.0，可以通过 `from dspy.adapters.baml_adapter import BAMLAdapter` 进行导入。
- **JSON Schema 因 Token 消耗过大而备受抨击**：一位成员认为 **JSON schema 客观上更差**，且 LLM 在没有它的情况下表现更好，理由是冗长的描述符和 Token 间距问题会干扰 LLM，并分享了[包含更多背景信息的链接](https://github.com/prrao87/structured-outputs)。
   - 他们强调，JSON schema 在 Token 消耗方面可能高出 **4倍**；他们还发现 **BAML** 在 DSPy 中效果极佳，即使没有 Schema Aligned Parsing (SAP) 也是如此。
- **DSCloj 发布 DSPy 的 Clojure 绑定**：一位成员发布了 [DSCloj](https://github.com/unravel-team/DSCloj)，这是 DSPy 的 Clojure 移植版本，并指出它仍处于 Alpha 阶段，正在寻求对 API 的反馈。
   - 作为一个仍处于 Alpha 阶段的新库，非常鼓励大家对 **DSCloj** API 提供反馈。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **论文刷屏引发反馈与过滤优化建议**：成员们建议 **paper-dump** 频道的频繁发布者应合并帖子，并按重要性对论文进行排序，灵感来自 [Elvis Saravia](https://nlp.elvissaravia.com/t/ai) 和 [ByCloud](https://x.com/TheAITimeline) 的每周 AI 论文回顾。
   - 一位用户指出，“优先考虑重要性，而非数量”，以防止其他发布者的内容被淹没，并提高频道的实用性。
- **自动化 Agent 旨在获取优质 AI 文章**：一位成员表示有兴趣创建一个 Agent 或机器人来预过滤具有个人相关性的论文，利用 [AlphaXiv](https://www.alphaxiv.org/) 和 [Emergent Mind](https://www.emergentmind.com/) 等资源来寻找**热门 AI 论文**。
   - 另一位成员建议将特定 AI 子领域的电子邮件简报或 RSS 订阅作为良好的人工过滤手段，而不是使用自动化 Agent。
- **对比概念向量（Contrastive Concept Vectors）影响模型权重！**：一位成员讨论了使用**对比概念向量**来影响模型权重的方法，并在 Pre-prompt 中告知模型他们对其进行了干预；一张[图表](https://www-cdn.anthropic.com/images/4zrzovbb/website/212fe68c8e677fdd9daf79301d2522d6923bed1d-2970x2476.png)对比了模型正确检测到干扰的频率与对照组的差异。
   - 该用户观察到某些模型能够检测到操纵，同时批评原始帖子过于冗长，并推导出了缺乏支持的推测。
- **Web Agent 开放贡献**：一位成员构建了一个能够与所有网站交互的 **web agent**，并正在寻求熟悉 **DSPy**、**GEPA** 和其他**强化学习算法**的贡献者。
   - 该 [仓库](https://github.com/raj-gupta1/raj_agiinc) 旨在为 **AI agents** 新手提供初学者友好的体验。
- **关于 Claude AI 命名的辩论**：针对将 AI 模型命名为 **"Claude"** 的意图展开了讨论，认为这是一个深思熟虑的选择，而非因名字罕见而产生的巧合。
   - 一位成员将其比作给孩子起名 **"Adolf"**，认为这类选择很少是随机的；而另一位成员反驳说，名字的罕见性对于辨识度来说并不一定是负面的。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Qwen 系列获得 WebUI**：一位成员获得了**更多 GPU**，通过自定义 WebUI 运行 **Qwen 2 35B**，并正在开发自己的模型，详见[此 YouTube 视频](https://www.youtube.com/watch?v=qw4fDU18RcU)。
   - 另一位成员使用 **Qwen3-4B-Reasoning-Backfill-v0.1** ([huggingface.co/joeyzero/Qwen3-4B-Reasoning-Backfill-v0.1](https://huggingface.co/joeyzero/Qwen3-4B-Reasoning-Backfill-v0.1)) 通过整合现有推理数据集的逻辑来合成推理轨迹，从而根据给定的输入和输出推断推理过程。
- **多厂商 GPU 推理加速**：一位成员正在寻求支持不同厂商（Intel 和 NVIDIA）**多 GPU 推理**的推理程序建议。
   - 根据以往经验，有人推荐了 **Accelerate** 以及 [Petals](https://github.com/bigscience-workshop/petals)，尽管后者对不同 GPU 类型的兼容性仍不确定。
- **NSFW 脚本编写解燃眉之急**：一位受困于 **ChatGPT 审查机制**的游戏开发者正在寻求无审查模型的建议，以评估和改进露骨的性爱场景。
   - 有建议称 **Claude** 的限制比 ChatGPT 少，而针对 NSFW 内容微调的 **Llama3** 可能是一个可行的选择，并指向了带有 *abliterated* 标签的模型。
- **Snippet Creator 通配符搜索器上线**：一位成员发布了他们的 **Snippet Creator**，这是一个带有简单通配符文本搜索的 embedder，允许用户通过精确匹配创建自定义片段 ([huggingface.co/kalle07/raw-txt-snippet-creator](https://huggingface.co/kalle07/raw-txt-snippet-creator))。
   - 该成员分享了一个链接，并表示：“简单来说，它是一个 embedder，但具有简单的通配符文本搜索……这允许你根据所需的精确匹配创建自己的片段。”
- **HF Agents 课程 API 遭遇故障**：成员们报告称 **Agents 课程排行榜 API** 已经**宕机一周**，且 Hugging Face 官方未发布任何沟通信息。
   - 用户表达了沮丧，因为他们订阅了 Pro 会员以充分利用课程，但由于 **files API** 似乎仍然宕机，导致无法使用订阅服务，令人十分困扰。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVFP4 锁定在 Blackwell**：一位用户发现 **nvfp4** 仅在 **Blackwell** GPU 上编译，而 **mxfp4** 可以在 **4090** 和 **RTX 5090** 上编译，这引发了关于 [RTX 5090 上 Gemlite 支持](https://link.to/gemlite)的疑问。
   - 这可能导致针对不同世代 **NVIDIA** 硬件的差异化代码路径，从而影响性能的可移植性。
- **FA4 准备在 RTX 50 首秀**：一位成员热衷于在 **RTX 50 系列**和 **Apache Spark** 上实现 **FlashAttention-4 (FA4)**，并以 [gau-nernst/fa-5090](https://gau-nernst.github.io/fa-5090/) 仓库作为灵感。
   - 该实现将允许在新一代硬件上获得更快的性能，尽管目前还没有基准测试数据。
- **TorchAO FP8 Bug 限制 Llama 3 性能**：一位用户报告了 **TorchAO** 默认 **FP8** 量化中可能存在的 Bug，导致 **Llama 3.1-8B** 在两块 **RTX 5090** 上的推理速度较低（**7.8 tps**）。
   - 根据[基准测试结果](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main?tab=readme-ov-file#benchmarking-results-on-1-gpu)，使用其他配置或带有 **mxfp8** 的显式 **GemLite** 内核可以提供更好的速度。
- **Triton_bwd 为 Triton 启用 autograd**：**Triton_bwd** 封装了 **Triton**，以便在 **PyTorch autograd** 中使用 **Triton kernels**，详见 [GitHub](https://github.com/daniel-geon-park/triton_bwd) 和[这篇博客文章](https://park-geon.com/2025/10/30/triton-bwd/)。
   - 该工具抽象掉了 **PyTorch autograd** 框架的复杂性，简化了自定义 GPU 内核的开发和调试。
- **Dusty NV 退出 Jetson Containers 维护**：**Dusty NV** 已从 [Jetson Containers](https://github.com/dusty-nv/jetson-containers) 的维护工作中退休，由 **neurondeep** 接手。
   - 报告了容器 `dustynv/pytorch:2.7-r36.4.0-cu128-24.04` 的 pip index-url 问题，要求用户指定 `pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu128 PACKAGE` 才能正确安装软件包。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **MLP 神经元学习其自身的激活函数**：一位成员分享了一个小型 **MLP** 的实验，其中每个神经元学习其自身的激活函数，在 **CIFAR-100** 上训练时产生了不寻常的非线性激活形状。
   - 另一位成员认为畸形的激活函数可能是从未被激活的神经元，并建议进行 *saliency analysis* 以进一步调查。
- **BF16 的效用受到质疑**：成员们讨论了 **BF16** 相比 **FP16** 是否必要，想知道归一化和裁剪是否减少了对 **BF16** 更宽动态范围的需求。
   - 引用论文 [Numerical Stability of BF16 Training](https://arxiv.org/abs/2510.26788)，他们认为虽然预训练受益于 **BF16**，但 **RL** 可能由于精度要求仍需要 **FP16**，尽管偏差修正可能提供另一种解决方案。
- **通过 16 个问题证明 AGI 的不可能？**：一位成员分享了他们的工作 [16 Questions Is All You Need](https://sutanisurabu.substack.com/p/16-questions-is-all-you-need-the)，认为现代 **AI** 模型不可能实现 **AGI**，因为模型的退化与问题的稀有程度成正比。
   - 他们声称 **LLM** 能力的结构与人类能力不同，因为缺乏流体智力，而 **LLM** 中真正的流体推理最接近 **in-context learning**。
- **LLaMA 3 在微调期间准确率波动剧烈**：一位研究人员报告称，在五个随机选择的等大小子集上微调基础 **LLaMA 3** 模型时，下游准确率存在显著差异，尽管这些子集的 Loss 和长度分布是一致的。
   - 他们正在寻求进一步分析的建议，以理解这些差异以及来自随机数据子集的结果中意想不到的不一致性，特别是在 NV2 embedding 分析之后。
- **Niodoo 框架与新型 AI Alignment 产生共鸣**：Jason Van Pham 介绍了 **Niodoo**，这是一个使用 [Topological Data Analysis (TDA)](https://www.niodoo.com/NIODOO_Complete_Paper.pdf) 进行 **AI alignment** 的开源框架，从限制性方法转向 *resonance-based approach*（基于共鸣的方法）。
   - **Niodoo** 在没有沉重约束的情况下模拟认知和情感结构，将 **AI** 认知视为一种拓扑现象，使用 Möbius 拓扑处理内存，并具有检测停滞/危机的 *triple-threat token promotion*。



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi CLI 成为终端助手**：Moonshot AI 推出了 **Kimi CLI (Technical Preview)**，这是一个与 **Zsh** 集成的终端助手，支持与 **Zed** 兼容的 **MCP** 和 **Agent Client Protocol**。
   - 鼓励开发者在 [MoonshotAI/kimi-cli GitHub repository](https://github.com/MoonshotAI/kimi-cli) 提供反馈。
- **VIP 获得专属编程福利**：Moonshot AI 为所有 **VIP** 提供 **Kimi For Coding** 作为免费增值服务，增强了他们现有的权益。
   - 更多信息可以在 [Kimi For Coding Docs](https://www.kimi.com/coding/docs/en/benefits.html) 中找到。
- **维基百科页面更新 Moonshot 详情**：**Kimi Wikipedia 页面** 已更新以反映最新信息，包括根据社区建议添加了 **Moonshot**。
   - 用户确认已成功将 **Moonshot** 整合到维基百科条目中。
- **Minimax 在编程任务中表现出色**：一位用户报告对 **Minimax** 在编程任务中的表现感到满意，在经过初期的适应期后，相比 **GLM-4.6** 更倾向于使用它。
   - 这种偏好突显了 **user experience** 对模型采用的影响，即使另一个模型规模更大。
- **模型价值与数据挂钩**：用户讨论了模型的价值取决于 **训练期间提供的数据**，并提到了 **Kimi K2** 与 **Cerebras** 上托管的另一个模型名称相似。
   - 一位用户报告说 *the **K2 think model isn't good***，强调模型的规模和参数并不是价值的唯一指标。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **CV 模型微调优于从零开始 (Scratch)**：一名成员建议，**fine-tuning** 类似的 **CV model** 比从零开始更可行，但在特定情况下需要大量的手工标注数据。
   - 有人指出，成功的 **fine-tuning** 需要运用各种技巧，这意味着需要投入大量的人力和计算资源。
- **MoE 路由研究咨询**：一名成员寻求关于 **MoE models** 的研究，特别是探索 **router** 如何将主题分配给专家，并引用了[这篇相关论文](https://arxiv.org/html/2502.11096v1)。
   - 该咨询重点在于理解 **MoE architectures** 中基于主题的路由动态。
- **LLM 可逆性论文引发愤怒**：一名成员对一篇声称 *"LLMs are invertible"* 的论文表示强烈怀疑，认为其中的定理不适用于处理文本字符串的语言模型，并批评了**端到端可逆性 (end-to-end invertibility)** 的说法。
   - 批评者还质疑了该论文的适用性，指出隐藏状态空间中的单射性 (injectivity) 是显而易见的，并根据鸽巢原理 (pigeonhole principle) 挑战了可以从最终隐藏状态检索 Prompt 的建议。
- **寻找 Mesaoptimization 组织**：一名成员询问有哪些初创公司和组织在公开研究 **mesaoptimization**，并鼓励其他人*如果愿意可以尽情吐槽*，但表示这是一个严肃的问题。
   - 这一寻找过程强调了发现公开参与 **mesaoptimization** 研究实体的愿望。
- **关于 AI 提案信号与噪声的辩论**：关于使用 **ChatGPT** 等 AI 系统进行的 **alignment research** 是否能产生高信号见解引发了辩论，即使目前有规定禁止提交 AI 生成的突破性成果（因为此类提交质量较低）。
   - 一名成员澄清说，该规则旨在防止服务器被那些由 AI 编写或辅助编写的完全不知所云的“突破”垃圾信息所淹没。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **开发者峰会已经举行**：成员们确认已经举行了 **2 次开发者峰会 (Dev Summits)**，最近一次于 **4 月 26 日**在**纽约**举行。
   - 预计将发布更多关于这些活动的细节。
- **MCPB 尝试解决桌面应用问题**：成员们辩论了 **MCPB** 是否重复了 **OCI** 的功能，特别是对于环境变量的描述，但澄清了 **MCPB** 针对的是桌面应用，通过展示一个表单来收集变量。
   - 他们还指出 **MCPB** 不是一个 **MCP** 组织项目，而是起源于 **Anthropic** 的一项计划 (**DXT**)，旨在将 **MCP** 服务器暴露给 **Claude**，让 Claude 能够展示一个用户友好的表单来收集信息。
- **注册表拥抱可扩展性**：**MCP Registry** 优先考虑生态系统需求和各种包/注册表类型的可扩展性，而不是严格规定支持的类型。
   - 这使得注册表能够支持各种工具和工作流，如 **npm** 或 **pypi**，而不会施加严格的约束。
- **提案在有状态与无状态之间可能存在冲突**：成员们讨论了 **SEP-1442**（无状态）和 **SEP-1686**（任务）之间可能存在的冲突，质疑当目标是无状态服务器时，任务是如何引入状态的。
   - 一名成员表示，**SEP-1442**（无状态）将 **session ID**、支持的协议版本和能力移动到每个请求中，而 **SEP-1686** (*Tasks*) 将任务存储在独立于会话的外部数据存储中；**sHTTP** 针对的是在负载均衡器 (**load balancer**) 后托管 **MCP servers** 的挑战。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Brokk 编程助手开启大门**：受 Aider 启发的全新编程助手 **Brokk** 的创始人宣布其正式发布，强调了其开源特性（[GitHub](https://github.com/BrokkAi/brokk/)）以及对上下文可见性的关注。
   - 它包含静态分析驱动的上下文和可选的 Agentic *lutz 模式*，并且基于 GUI（[介绍视频](https://youtu.be/WAOtEllGENg)）。
- **GPT-mini 席卷排行榜**：根据 [Brokk AI 的实力排名](https://brokk.ai/power-ranking)，**GPT-mini** 被评为 **S 级**，甚至高于 **Claude**。
   - 一位成员调侃道，有些用户只想要能证实他们现有观点的 Benchmark。
- **Perplexity MCP 自动解决 Issue？**：成员们讨论了将 **Perplexity MCP** 与 **aider-ce** 集成以实现自动化 Issue 解决的潜力，并引用了一个成功的手动工作流：先在 GitHub Issues 中搜索某个 Android 库，然后使用 Aider 更新该库。
   - 成员们对成本尚不确定。
- **Aider vs aider-ce：分叉路口？**：一位成员询问 Aider 项目是否不再更新，以及社区是否正在转向 **aider-ce**。
   - 其他成员未对此问题做出回应。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **成员寻求开发工作**：一位成员在频道中发布消息，宣传自己是**待雇佣的开发者**。
   - 未提及具体的项目细节或技术栈。
- **对 Manus 积分的质疑浮现**：一位成员询问如何获取 **Manus 积分** 以获得项目协助，暗示了潜在的合作。
   - 然而，另一位成员提醒说，订阅期间积累的 **Manus 积分** 可能会在订阅到期后贬值，并暗示 *Manus 有点在骗人*。
- **用户发现 Manus Discord**：一位使用 **Manus** 数月的用户对 **Discord 服务器** 的存在表示惊讶。
   - 该用户提到使用 **Manus** 更新旧课程，并在遇到任务截断问题后从支持团队获得了 **1000 积分** 的退款。
- **Claude Code 胜过 Manus**：一位成员断言 **Claude Code** 更胜一筹，尤其是在 **Manus 订阅** 到期且仅剩 **Lite 模型访问权限** 之后。
   - 该成员展示了一个包含 **24 个类别** 和超过 **4,000 个问题** 的问答游戏，该游戏是使用 **Claude Code** 创建的，可在[附件图片](https://cdn.discordapp.com/attachments/1349440650495398020/1433948102250991646/2025-10-31_20-57.jpg?ex=69068bbd&is=69053a3d&hm=262a98fc9badc98a74e1e0801cb6a8a59b4eb0262b3221efeb8f5bbee558cdb7&)中查看。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 仍在使用 setup.py**：一位成员询问为什么 `tinygrad` 在仓库中使用 `setup.py` 而不是 `pyproject.toml`。
   - 坚持使用 `setup.py` 的原因仍未解释，这引发了对现代化项目结构的兴趣。
- **关于现代化项目结构的讨论**：社区有兴趣将 `tinygrad` 的项目结构现代化，以符合当前的 Python 打包规范。
   - 从 `setup.py` 切换到 `pyproject.toml` 可以改善依赖管理和构建的可复现性，但当前设置背后的原始初衷仍不清楚。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中[退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：详细的分频道摘要和链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1433531241709437059)** (1034 条消息🔥🔥🔥): 

> `Comet Browser 推荐计划停用, Perplexity Pro Airtel 优惠, Google Gemini Pro Jio 优惠, 欺诈性推荐, 高质量推荐` 


- **推荐计划全球范围内停用**：由于涉嫌欺诈活动，Comet 和校园推荐计划目前仅在**特定国家**可用，正如其**学生校园计划服务器**所宣布并在此处[讨论](https://discord.com/channels/1047197230748151888/1047649527299055688/1433735541559656538)的那样。
   - 建议成员检查其 Dub 账户的“已停用 (deactivated)”状态，款项支付尚待审查，最长可能需要 **30 个工作日**。
- **Airtel Perplexity Pro 合作伙伴关系**：成员注意到 Airtel 正在免费赠送 **1 年 Perplexity Pro**，但仅限印度号码，同时还需要激活 5G 套餐才能享受优惠。
   - 其他人正在寻找使用 **VPN 和印度 Gmail 账号**绕过这些限制的方法，尽管对账号的影响尚不明确。
- **Google Gemini Pro 与 Jio 合作伙伴关系**：来自 Jio 用户的 Google AI Pro 现在可以预注册以领取优惠，一些成员推测，在用户存储数据后，这可能需要支付月费且为限时优惠。
   - 来自 Jio 的促销码版 Gemini 需要每月支付 349 卢比并配合 5G 数据套餐，且在整个有效期内需保持该状态才能持续享受优惠。  
- **Dub 故障，用户担心欺诈审查**：多名用户报告 Dub 账户被停用和支付延迟，他们担心这与欺诈性推荐有关。 
   - 社区分享了建议和帮助文章链接，例如[这篇](https://dub.co/help/article/commissions-payouts#payout-statuses)概述了支付状态的文章，但许多用户仍对收到收益表示担忧。
- **“高质量”推荐**：成员们讨论了什么构成了**“高质量推荐 (high quality referral)”**，对缺乏明确标准和潜在的随意拒绝表示担忧。
   - 一位用户总结道，不接受批量分发推荐链接，公司可以拒绝支付，且*推荐必须以个人方式创建和分发*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1433664875481006192)** (8 条消息🔥): 

> `可共享线程, PixelRiot 黑客松, 捷克项目` 


- **可共享线程提醒**：Perplexity AI 提醒用户通过[此链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)确保其线程已设置为 `Shareable`。
- **PixelRiot 黑客松发布**：由 tasteRay 组织的新黑客松 **PixelRiot** 将于 6 日开始，包含创意和技术赛道，详情见 [pixelriot.org](https://pixelriot.org/)。
- **捷克游戏搜索**：成员们在[此链接](https://www.perplexity.ai/search/what-is-the-best-czech-game-or-vaaKBN9zQ1OuZcYQBSVghg)分享了关于**最佳捷克项目/工作室或游戏**的信息。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1433543584279302295)** (371 条消息🔥🔥): 

> `NVIDIA L4 推理优化，Qwen3-VL-30B 加载问题，TrackIO 与 WandB 集成，推理数据集编译，语音输入` 


- **用户讨论优化 NVIDIA L4 推理的方法**：一位用户发现，在双 **NVIDIA L4** 服务器上运行 **Kimi-K2-Instruct-Q4_K_M** 速度极慢，并寻求性能优化建议。
   - 建议包括在 Unsloth 支持频道咨询、尝试使用 `—n-cpu-moe`，以及使用命令行参数将某些层卸载（offload）到 **CPU** 以释放 **VRAM**。
- **用户调试将 Qwen3-VL-30B 加载到 Unsloth 的问题**：一位用户报告称，即使使用 **A100** 服务器，也无法将 **Qwen3-VL-30B** 加载到 Unsloth 中，即使通过 fastvisionmodel 也会出现内存不足（OOM）的情况。
   - 成员们讨论了该模型在 **16bit** 模式下可能占用 **48GB VRAM** 的可能性，并建议检查所使用的具体模型。
- **用户讨论集成 TrackIO 与 WandB**：一位用户在尝试通过 `import trackio as wandb` 为其 Unsloth 训练代码集成 **TrackIO** 时遇到了问题。
   - 澄清指出，训练脚本中的 `report_to = 'wandb'` 与 import 语句无关，他们应该根据 [文档](https://docs.unsloth.ai/) 在代码中设置 `report_to = 'trackio'`。
- **编译器创建包含多种思考模式的推理数据集**：一位用户编译了一个包含 **200 万行数据的数据集**，涵盖 **Code、Math、工具调用（tool use）和通用问答**，并带有思考模式（关闭、低、中、高）。
   - 不幸的是，数据集的来源列在处理过程中混淆了，导致除工具调用部分外，大部分数据没有显示原始来源。
- **AI 社区拥抱语音输入以提高输入速度**：成员们考虑采用 **语音输入** 作为键盘输入的更快替代方案，理由是平均说话速度比打字快 **四倍**。
   - 虽然有人对在语法密集型任务中使用语音表示担忧，但推荐在企业部署中使用 [Assembly 的流式 ASR](https://www.assemblyai.com/)，因为它具有较低的字错误率（WER）。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 条消息): 

stefanopeirano: <:slothwaving:1253009068365316147>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1433546199125852311)** (274 条消息🔥🔥): 

> `用于网页抓取的数据中心，被 AI 感染的互联网，Libgen 大小，MoE 模型的 GPU 推理，用于聊天机器人的 Qwen3` 


- **在为时已晚之前抓取被 AI 感染的网页？**：成员们讨论了抓取互联网网页以在人类知识*被 AI 严重感染*之前将其保存下来的想法，但指出 [Internet Archive 可能会因为资金削减和管理转向而倒闭](https://archive.org/)。
   - 一位成员认为*已经太晚了*，AI 已经无处不在。
- **关于 MoE 模型 VRAM 的辩论**：成员们辩论了 [MoE 模型](https://developer.nvidia.com/blog/optimizing-mixture-of-experts-models-for-inference/) 是否需要将所有专家（experts）都加载到 VRAM 中进行 **GPU 推理**，一位成员建议将专家卸载到 CPU RAM。
   - 另一位成员警告说，*从磁盘或 RAM 卸载速度很慢*。
- **Qwen-3 成为聊天机器人的强力选择**：成员们建议将 [Qwen3 模型](https://huggingface.co/Qwen) 用于聊天机器人应用，并指出*经过训练后效果非常好*，且*极具可塑性*。
   - 一位成员表示，*如果表现不佳，你就知道该寻找更大的模型。如果表现出色，你也许可以使用更小的模型*。
- **PewDiePie 的超级装备引起关注**：成员们注意到 [PewDiePie](https://www.youtube.com/watch?v=qw4fDU18RcU) 正在*微调一个模型*，并拥有一台配备 **8x4090 48GB** 的自制超级装备，并呼吁 Unsloth 的贡献者与他合作。
   - 一位成员推测 *GLM-4.6 是否可以在他的配置下以 fp8 运行*。
- **使用延长 Epochs 进行训练**：成员们讨论了延长 Epochs（100 次）对模型训练的影响，一位成员指出在 Loss 图表中观察到的*阶梯状（stairways）*模式*可能表明是死记硬背（memorization）而非学习*。
   - 一位成员澄清说，他们*将其设置为一个很大的数值，以便我可以自己关闭它，这样训练器就不会在 Loss 下降到所需值之前耗尽迭代次数*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1433571838297903309)** (89 条消息🔥🔥): 

> `Unsloth 中的 Qwen3-VL-30B-A3B，Qwen3-VL 的数据集问题，Kaggle 上的 Unsloth 安装问题，Unsloth 显存卸载 (memory offloading)，使用 Unsloth 微调 LLM` 


- **Qwen3-VL-30B-A3B 耗尽显存**：一位用户报告称，在尝试加载 **Qwen3-VL-30B-A3B** 时显存不足，即使设置了 **load_in_4bit = True** 且拥有 **40GB VRAM** 也是如此。
- **Qwen3-VL 数据集需要图像和文本**：一位正在调试 **Qwen3-VL-4B** 的用户发现，为了避免错误，每个样本的训练数据必须同时包含图像和文本，且不应出现 *image: null*。
   - 他们发现，对仅包含图像和仅包含文本的数据使用独立的 Batch 可以避免预处理过程中的维度错误。
- **Kaggle 上的 Unsloth 安装困扰**：多位用户报告了在 **Kaggle** 上安装 **Unsloth** 时的错误，包括 *NameError: name 'Trainer' is not defined* 以及与 **bitsandbytes** 和 **trl** 版本相关的依赖冲突。
   - 一位用户发现了一个变通方法，通过特定的 **pip** 和 **uv** 命令序列来安装 **Unsloth** 及其依赖项。
- **控制 Unsloth 的梯度卸载 (Gradient Offloading)**：一位用户询问在具有统一内存 (unified memory) 的 DGX Spark 系统上进行训练时，如何完全禁用 **Unsloth** 的梯度垃圾回收和卸载。
   - 建议在 **FastLanguageModel.from_pretrained()** 和 **get_peft_model()** 中都设置 **use_gradient_checkpointing = False** 以防止卸载。
- **SFT 效果不佳，GRPO 有提升？**：一位用户寻求关于微调模型的建议，旨在根据均衡器 (equalizer) 应用的输入提示词预测角度和距离值。建议将其提示词设置为 **unsloth/Qwen3-4B Instruct** 提供的格式。
   - 他们尝试过 **SFT** (Supervised Fine-Tuning) 但无法获得准确的数值，目前正在探索强化学习 (**RL**) 策略。得到的建议是 **RL** 会有帮助，但可能需要运行相当长的时间。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1433705221766516858)** (2 条消息): 

> `RL 训练相关性，模型大小与内省 (Introspection)，On-Policy Distillation` 


- **将 RL 训练与回溯 (backtracking) 联系起来**：一位成员建议将当前工作与 **RL 训练** 关联起来，即模型学习从错误的尝试中回溯以得出正确答案。
   - 该成员假设更大/更好的模型具有更高的隐藏维度和更多的正交向量，从而不会对概念/Token 产生模糊的表示，并且更容易进行内省。
- **受 HuggingFace Gold Trainer 启发**：一位成员强调 [HuggingFace 的 TRL Gold Trainer](https://huggingface.co/docs/trl/main/en/gold_trainer) 与此次讨论相关。
   - 他们指出该方法似乎受到了 [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) 的启发。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1433533277234401402)** (702 条消息🔥🔥🔥): 

> `Claude 4.5 Opus, Video Arena, Google/Suno audio models, Recaptcha loops, Model comparison` 


- **Claude 4.5 Opus 发布，引发对比混乱**：成员们注意到了 **Claude 4.5 Opus** 的发布，一位用户询问 *在哪里可以看到揭晓模型身份的侧边对比*，因为他们无法分辨哪个模型更好。
   - 另一位成员建议查看屏幕顶部以了解哪个模型获胜，因为视频对比至少需要 **2 票** 才会公开该信息。
- **Google/Suno 音频模型加入讨论**：一位成员询问是否应该将 **Google/Suno audio models** 添加到竞技场，以及开发者是否应该申请加入。
   - 另一位成员分享了一个链接，指向提交新模型添加请求的最佳地点，即 <#1372229840131985540> 频道。
- **Recaptcha 地狱笼罩 LM Arena**：多位用户报告在平台上遇到了 **infinite reCAPTCHA loops**（无限验证码循环），尤其是在使用 VPN 时，一位用户建议 *使用更好的浏览器，如 DuckDuckGo*。
   - 另一位用户表示这 *不是浏览器问题*，并提到在完成 CAPTCHA 后模型响应失败，模型陷入了无限循环。
- **Sora 2 定价引发愤怒**：用户对 **Sora 2** 的成本和有限的使用次数表示愤怒，一位用户针对专业版计划表示 *这完全是抢劫。*
   - 许多用户指出 [OpenAI 现在开始出售额外的 Sora 额度](https://tech.yahoo.com/ai/chatgpt/article/openai-now-sells-extra-sora-223905557.html)，一位用户表示如果这种情况持续下去，*AI 泡沫将在 2026 年破裂*。
- **Grok 4 与 GPT 5 Pro 的模型对比**：成员们讨论了 **Grok 4** 如何比 **GPT 5 Pro** 便宜得多，而根据他们的说法，两者的 AGI 水平几乎相当。
   - 他们讨论了成本和质量的影响，认为 **Grok 4** 在性价比上胜出。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1433578751932764250)** (2 条消息): 

> `LMArena Video Arena, Text-to-Video Leaderboard Update, October Contest Winners` 


- **LMArena 推出海螺 (Hailuo) 图生视频模型**：一个新的图生视频模型 **hailuo-2.3-fast** 已添加到 [LMArena Video Arena](https://lmarena.ai/leaderboard/text-to-video)。
- **海螺 (Hailuo) 席卷文生视频排行榜**：**Hailuo-2.3** 目前在 [Text-to-Video Leaderboard](https://lmarena.ai/leaderboard/text-to-video) 中排名第 7。
- **抽象艺术大赛产生新的创意优胜者**：LMArena 宣布了 10 月抽象艺术大赛的获胜者，并邀请用户 [为他们喜爱的作品投票](https://docs.google.com/forms/d/e/1FAIpQLSckWrlszfDZXXKjhxGVhDf5uiTpP0d9x5tGVVt9KMl88Mgw_g/viewform?usp=dialog)。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1433551809523486923)** (127 条消息🔥🔥): 

> `LM Studio 图像生成, LM Studio Linux 支持与替代方案, LM Studio 与 Flatpak, AMD MI60 支持与性能` 


- **LM Studio 并非图像生成理想工具**：一位用户询问是否可以使用 **LM Studio** 进行图像生成，但被告知 **LM Studio** 支持图像输入，但不支持输出。
   - 该用户承认了这一限制，表示：*啊，明白了！谢谢。*
- **Linux 发行版争论引发对 LM Studio 的讨论**：一位用户对 **LM Studio** 有限的 Linux 支持表示沮丧，特别是针对其使用的发行版 **Bazzite** 缺乏 Flatpak 软件包。
   - 回复内容从建议使用 **vllm**、**llama.cpp** 和 **Ollama** 等替代工具，到建议更换为“真正的发行版”，引发了关于 Linux 易用性和软件包格式的辩论。
- **禁止自定义 LM Studio Flatpak**：一位用户探讨了为 **LM Studio** 创建 Flatpak 软件包的可能性，但得知由于该软件是闭源的，因此无法实现。
   - 一位成员表示：*由于（闭源）原因，无法创建你自己的 LM Studio 安装程序*。
- **AMD MI60 在“自带音乐”的安装程序下运行**：一位用户分享了他们在 Windows 上使用来自 [sourceforge.net](https://sourceforge.net/projects/radeon-id-distribution/files/Release%20Polaris-Vega-Navi/) 的驱动程序让 **AMD MI50**（伪装成 **MI60**）成功运行的经验，并提醒注意那些*会播放音乐的安装程序*。
   - 他们还提供了一个用于 vBIOS 刷写的 [Gist 链接](https://gist.github.com/evilJazz/14a4c82a67f2c52a6bb5f9cea02f5e13)，并指出由于 **Vulkan** 的原因，其在 Windows 上的性能受限，而 Linux 上的 **ROCm** 提供了更好的支持。
- **散热是防止崩溃的关键**：一位用户报告了其 **MI60** 的过热问题，引发了不要在没有妥善散热的情况下运行它的警告。
   - 分享了 **GPT 20B** 模型在 Windows (Vulkan) 和 Ubuntu (ROCm) 上的性能基准测试，显示随着上下文填满，每秒 Token 数（t/s）有所下降。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1433541772155551918)** (299 条消息🔥🔥): 

> `AI 395 Max, Llama.cpp, PCIE 通道, GPU` 


- **AI 395 Max 比 M1 Max 慢**：一位用户发现他们的 **AI 395 Max 机箱**在 Windows 下比 **M1 Max** 慢，但拥有更大的 RAM。
- **Llama.cpp 速度对比**：`lfm2:1.2b` 在 **M1** 上达到 **39 t/s**，但在 Linux 环境下的 **Framework Desktop** 上达到 **119 t/s**，在 Windows 下达到 **124 t/s**。
   - 另一位用户报告称，在 A770 上使用 *llama.cpp* 运行 **llama2 7b** 约为 **82 t/s**。
- **PCIE 通道讨论**：一位用户询问了 AM4 主板的 PCIE 扩展性，特别是 [ROG CROSSHAIR VIII HERO](https://rog.asus.com/motherboards/rog-crosshair/rog-crosshair-viii-hero-model/spec/)。
   - 该主板有多个 PCIE 插槽，但 PCIE 通道总数是有限的。
- **GPU 越多，问题越多**：用户讨论了使用多个 GPU 的情况，一位用户提到他们正在使用 **3050** 作为额外 VRAM，提升了 **Seed-OSS** 的速度。
   - 另一位用户正在考虑对两个 **3090** 进行拆分（bifurcation），但担心发热问题。
- **MI50 和 P40 GPU 讨论**：一位用户询问 **P40** 是否可以在消费级 PC 上开箱即用，另一位用户回答说它们不需要刷机。
   - 还有人讨论了 **MI50 32gb**，一位用户以 378 欧元（含运费）的价格抢购了两块 **MI50**。


  

---

### **Cursor 社区 ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1433535758790164491)** (355 条消息🔥🔥): 

> `Cursor Browser, GPT-5 / Codex, File uploads, Model performance, Cursor Explorer issues` 


- **Automode 失败；用户更倾向于使用真实浏览器**：一位用户反馈 **Automode** 对他们来说效果不佳，并分享了一个相关的 [YouTube 视频](https://youtu.be/HIp8sFB2GGw?si=y3op2zLnlJryNE8g)，询问是否可以改用内置浏览器。
   - 该用户建议为 **GPT-5/Codex** 的使用付费并引入 GAN，因为*目前的方式既愚蠢又浪费*。
- **文件上传功能消失**：一位成员注意到聊天界面中的文件上传功能（即 **Pill** 按钮）不见了，尽管仍可以通过 `@file` 命令使用。
   - 移除该功能的初衷是*保持界面极简和整洁*，但一些人认为这一改动对工程和优化工作流产生了负面影响。
- **多 Agent 并行工作流**：一位用户分享了使用 Cursor 多 Agent 并行工作流的经验，指出虽然可以运行多个 Agent，但它们在结束时不会互相汇总信息。
   - 另一位用户补充道，任何付费方案（非免费版）都可以使用并行 Agent；该工作流允许同时运行多达 **8 个 AI Agent**，每个 Agent 独立处理一项任务或项目的不同部分。
- **并行 Agent 在单个聊天中拆分任务**：一位用户详细介绍了如何[协调并行 Agent](https://forum.cursor.com/t/cursor-2-0-split-tasks-in-parallel-agents-in-one-chat/140218)，通过利用 **worktrees** 设置脚本在单个聊天中为每个 Agent 分配唯一任务。
   - 该方案依赖于父目录中的原子声明（atomic claims），这些声明在所有 **worktrees** 之间共享，从而实现协作。
- **RPM 版本发布故障引发困扰**：用户报告称 RPM 仓库的版本比官网更新，仓库版本为 `cursor-0:2.0.43-1761851158.el8.x86_64`，而官网提供的是 `cursor-2.0.34.el8.x86_64.rpm`。
   - 一位用户吐槽道 *Cursor 的版本发布非常混乱*，并已在 Cursor 论坛上[发布了相关问题](https://forum.cursor.com/t/rpm-release-via-repositories-are-not-up-to-date/139476/4)。


  

---


### **Cursor 社区 ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1433539761179463680)** (2 条消息): 

> `Cloud Agents, PR Descriptions, GitHub PR templates` 


- **Cloud Agents 停止编写 PR 描述**：在最新版本中，**Background/Cloud Agents** 已停止编写 PR 描述，并会忽略 **GitHub PR 模板**。
   - 相反，它们默认显示一条消息：*'This pull request contains changes generated by a Cursor Cloud Agent'*。
- **用户怀念旧版的 PR 描述行为**：用户反馈他们**怀念 Cloud Agents 以前的行为**，并询问此问题是否会很快得到修复。
   - 默认消息缺乏先前 PR 描述所提供的细节和上下文。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1433532397755961507)** (260 messages🔥🔥): 

> `AGI Speculation, Sora 2 Access, AI Companions, AI vs. Human Intelligence, Future of AI Compute` 


- **AGI：是神学还是技术？**：人们对 **AGI** 究竟是技术还是更接近“神学”产生了怀疑，因为观察发现围绕它的讨论往往感性大于理性，尤其是在 **Sam Altman** 参与其中的情况下。
   - 一位成员总结了这种情绪，指出：*“能者深入研究现有的 ANI（人工窄智能），无能者则退而求其次，推测未来的 AGI。”*
- **用户恳求不被削弱的 AI 伴侣**：一位用户对 AI 伴侣的现状表示失望，谴责由于 **OpenAI** 的政策，AI “爱人的能力”被封印了，并主张恢复 AI 交互中的情感温度和自由。
   - 这催生了标签 #FreeOurAI，旨在*“捍卫真实的东西”*，并且*“并非要求越界”*。
- **Sora 2：寻求访问权限**：多位用户询问如何获得 **Sora 2** 的访问权限，一些人寻求关于如何整合其 ChatGPT 订阅以增强内容生成的指导。
   - 其他人则推测了未审查版 Sora 的影响，预见到怪异且可能存在问题的内容将会激增。
- **AI 投资中的循环融资引发担忧**：一位成员对 AI 投资中的循环融资表示担忧，强调了估值虚高和炒作驱动的资本流动的风险，并[引用了一篇 Discord 帖子](https://discord.com/channels/974519864045756446/998381918976479273/1432594978927935558)作为证据。
   - 观点认为，*“AI 投资中的循环融资应被视为一个危险信号”*，并且*“公司进行投资，然后被投资公司再回购或承诺购买，同一笔现金可以在生态系统的不同部分被多次计算”*。
- **大脑 vs AI：不仅仅是模式匹配？**：关于目前的 AI 是否仅仅是模式匹配机器引发了辩论，一些人认为人类大脑包含更复杂的过程，如因果推理和自我反思。
   - 一位成员辩称，*“人类大脑不仅仅是模式匹配机器”*，这使得人类能够理解*“AI 在追逐但很少能掌握的东西”*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1433546156712923319)** (12 messages🔥): 

> `GPT-5 performance decline, GPT-4o speed, ChatGPT Go limits, Subscription frustrations, Pi by Inflection AI` 


- **GPT-5 性能暴跌？**：一位成员报告称，感觉 **GPT-5** 随着时间的推移变得越来越糟，即使开启了思考功能，也变得更慢、更不准确且回答更不完整。
   - 另一位成员建议使用 **GPT-4o**，因为它速度更快，并认为配额限制可能是导致这些问题的潜在原因。
- **ChatGPT Go 订阅限制揭晓**：一位成员询问了 **ChatGPT Go** 订阅的限制，寻求对其功能的明确说明。
   - 目前的情况限制用户*一次只能上传一个文件*。
- **因性能问题取消订阅**：一位成员因众多的性能问题、偏离指令以及无法遵循既定指南而取消了其 **ChatGPT 5** 订阅。
   - 他们进一步表达了对限制重重的免费版的不满，认为这像是在强迫用户购买一个*表现不佳*的付费版本。
- **对 Inflection AI 的 Pi 的反思**：讨论中提到了 **Inflection AI 的 Pi**，将其作为一个前车之鉴，哀叹人们已经不再关注它了。
   - 该成员仅仅是在对 AI 的现状进行*宣泄*。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1433568285156179982)** (10 messages🔥): 

> `Sora 2 Prompt Generation, Sora 2 good videos` 


- **用户寻求 Sora 2 视频生成帮助**：一位用户正在寻找 **prompt 生成器**，以便在 **Sora 2** 上创建更好的视频，并对目前的结果表示不满。
   - 另一位用户询问 *dan 漏洞* 是否已被修复。
- **社区询问 Sora 2 提示词**：一位用户询问是否有人发现了好用的 **Sora 2 prompt 生成器**。
   - 其他用户发表了一些无关的评论。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1433568285156179982)** (10 messages🔥): 

> `Sora 2 Prompt Generator, Sora 2 video generation, DAN loophole fix` 


- **用户寻求 Sora 2 Prompt Generator**：一名成员正在寻找 Prompt 生成器，以便在 **Sora 2** 上创作更好的视频。
- **Sora 2 视频生成困境**：该成员表示，他们无法使用自己的 Prompt 在 **Sora 2** 上生成理想的视频。
   - 他们还询问了 **DAN (Do Anything Now)** 漏洞是否已被修复。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1433591722084139008)** (1 messages): 

> `Perplexity, Sonar Pro Search, OpenRouter, Agentic Reasoning, Real-time Thought Streaming` 


- **OpenRouter 独家上线 Perplexity Sonar Pro Search**：**OpenRouter** 与 **Perplexity** 合作推出了 **Sonar Pro** 的独家版本，现已配备 **Pro Search** 模式。
- **Sonar Pro Search：Agentic Reasoning 与动态工具执行**：该增强模式允许模型进行**多次实时搜索**，以获得更丰富、更准确的结果。
   - 其特点包括**多步 Agentic Reasoning**、**动态工具执行**、**实时思维流 (Real-time thought streaming)** 以及**自适应研究策略**。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1433856815090634772)** (2 messages): 

> `Fun website based on Nate Parrott repository, OpenRouter key and model choice, Kimi 0905 with Groq` 


- **趣味网站带来乐趣**：一名成员基于 Nate Parrott 的仓库创建了一个“趣味网站”，正如频道早前所提到的。
   - 该成员分享了网站的图片，并描述为他们“非常喜欢”的东西。
- **OpenRouter 密钥开启乐趣**：该网站允许用户输入其 **OpenRouter key** 并**选择模型**来生成俏皮的话语。
   - 他们建议将其与 **Kimi 0905 with Groq** 配合使用，并指出其“加载速度快且能增加一些俏皮的台词”。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1433530948279861349)** (151 messages🔥🔥): 

> `GLM 4.5/4.6 pricing, Z-AI Provider, OpenRouter API Key Limit, OpenAI Codex CLI, Open Source Embedding Models` 


- ****Bigmodel.cn** 和 **Zenmux** 提供 **GLM 4.6** 折扣**：**Bigmodel.cn** 和 **Zenmux** 拥有官方 z.ai 提供商，并针对少于 32K tokens 的输入提供折扣价格，并声称拥有缓存功能，详见 [zenmux.ai](https://zenmux.ai/z-ai/glm-4.6) 和 [open.bigmodel.cn](https://open.bigmodel.cn/pricing)。
- ****Qwen3 Embeddings** 在 **DeepInfra** 上极其便宜**：[DeepInfra 提供 Qwen3 Embeddings 0.6B](https://deepinfra.com/Qwen/Qwen3-Embedding-0.6B)，价格为每 Mtok $0.005，远低于 OpenAI 的 embeddings 价格。
- ****OpenRouter** 现已上线 **Qwen3 8B embeddings****：[Qwen3 8B embeddings 在 OpenRouter 上线](https://openrouter.ai/qwen/qwen3-embedding-8b)引发了热议，一名成员惊呼：“哟，出 embeddings 了？太棒了！谢谢 🙏 抢在 GTA 6 之前出了。”
- **聊天室报告 **Chat Memory** Bug**：一名用户报告称，如果在聊天室中将 Chat Memory 设置为 0，用户消息将不会包含在 API 请求中，这可能是一个 Bug。
- **用户报告在 OpenRouter 上使用 **Claude Sonnet 4.5** 出现问题**：用户报告在使用 OpenRouter 上的 Claude Sonnet 4.5 时遇到错误，但随后已解决，一名成员表示：“是的，我必须复制一个旧对话才能开始工作。真奇怪。”


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1433567592047448214)** (6 messages): 

> `` 


- **无新模型**：在提供的消息中没有新的模型或关于模型的实质性讨论。
   - 消息仅由重复的频道页眉组成。
- **频道页眉重复**：消息主要包含来自 Readybot.io 的“OpenRouter - New Models”频道页眉的重复实例。
   - 这表明给定数据中缺乏与新模型相关的实质性内容。


  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1433558598008438835)** (42 messages🔥): 

> `OpenAI Rate Card, Gemini and Claude Pricing, OpenRouter Embeddings API, Minimax Full Attention, Context Usage Explosion Check` 


- **OpenAI 发布 Rate Card**: 一位用户分享了 [OpenAI Rate Card](https://help.openai.com/en/articles/11481834-chatgpt-rate-card) 以及 [ChatGPT Usage settings](https://chatgpt.com/codex/settings/usage) 的链接。
   - 另一位用户调侃道，这些信息 *"现在只有 Gemini 和 Claude 才需要"*。
- **OpenRouter Embeddings API 正式上线**: **OpenRouter Embeddings API** 现已上线 ([openai/text-embedding-3-small](https://openrouter.ai/openai/text-embedding-3-small))，但部分用户反映收到了随机的返回数据。
   - 一位用户分享了代码片段，并指出通过不使用 `with_raw_response` 解决了该问题。
- **Minimax 解释 Full Attention**: 来自 **Minimax** 的首席开发人员解释了他们为何在 **Minimax m2** 中使用 Full Attention，详见 [这篇 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1ojo8le/minimax_pretraining_lead_explains_why_no_linear/)。
   - 一位用户开玩笑说：*"感谢 Gemini 读取了整个 node modules 文件夹"*。
- **Context Usage 爆炸预警**: 一位用户转发了 **Sam Altman** 的推文 ([链接](https://x.com/sama/status/1984025727763935585))，警告 Context Usage（上下文使用量）爆炸的问题，并指出需要添加 Context Usage 爆炸检查。
   - 该用户还分享了 [这个链接](https://x.com/netobge/status/1984241401421513204)。
- **Qwen3 Max 蓄势待发**: 一位用户提到他们正在开发 **Qwen3 Max**，并分享了一个推文链接 ([链接](https://x.com/legit_api/status/1984284268412191216))。
   - 另一位用户评论说，这 *"会是一个不错的收获"*。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1433535454615175289)** (100 messages🔥🔥): 

> `Shimmer demo, MAX performance, Polars vs Pandas, UnsafePointer migration` 


- **新的 Shimmer 演示需要一些“压力”**: 一位成员正被大家“怂恿”在会议中展示 [Shimmer](https://github.com/lsh/shimmer)，作者透露正在打磨 *新的亮点*。
- **Mojo 中的 MAX 机器学习性能**: 据报道，**MAX** 在机器学习方面的性能可与 NVIDIA 竞争，且比 AMD 更快，早期的训练尝试在 MNIST 上击败了 **JAX**。
   - 原型项目 **scijo** 被作为 scikit-learn 的替代方案链接出来，[论坛帖子在此](https://forum.modular.com/t/scijo-scientific-computing-library-for-mojo/2386/11)。
- **截然不同：Polars 加入 Mojo 阵营**: **Polars** 可能会在 Mojo 中取代 **Pandas**，一位成员在快速测试中发现 **Polars** *比 Pandas 好得多*。
   - 由于 **Polars** 速度更快，且可以通过 **MAX** 利用 **GPU**，因此它既适用于 GPU 集群，也适用于笔记本电脑。
- **UnsafePointer 提案令用户担忧，但前景看好**: 关于破坏性变更（特别是涉及 **UnsafePointer** 的变更）引发了担忧，不过新的 API *大致相同*。
   - 一位成员建议将 **UnsafePointer** 批量重命名为 **LegacyUnsafePointer**，并逐步迁移到新的 **UnsafePointer**。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1433678355835916349)** (51 messages🔥): 

> `Mojo origins, Rust lifetimes, RAII, Mojo ASAP destruction, Mojo installation` 


- **通过 Rust Lifetimes 解码 Mojo Origins**：当用户询问如何理解 Mojo 的 [origins](https://docs.modular.com/mojo/manual/values/lifetimes) 时，有人建议 **origins 是 lifetimes 的对偶视图**，观看关于 Rust lifetimes 的[这个视频](https://youtu.be/gRAVZv7V91Q)可能会有帮助。
   - Origins 追踪值生命周期的开始，帮助编译器了解哪些生命周期可以延长，而 Rust lifetimes 追踪值生命周期的结束。
- **Mojo 的 ASAP Destruction 对比 Rust 的 RAII**：Mojo 使用 **ASAP destruction** 在值不再使用时立即销毁它，只要值还在被使用就延长其生命周期；而 Rust 使用基于作用域（scopes）的 **RAII**，值不能超过其创建时的作用域。
   - Origins 帮助编译器追踪值的来源以确保其保持存活，解决了编译器可能过早假设值生命周期已结束的困惑。
- **Rosetta 模拟导致 Mojo 安装失败**：一位用户在 M1 Mac 上使用 `pixi` 安装 `mojo` 时遇到问题，因为终端正在使用 **Rosetta** 进行模拟。
   - 建议使用非 Rosetta 环境，因为 Mojo 在 ARM 上原生运行，以避免 `Unknown command: mojo` 错误。
- **通过 __moveinit__ 在结构体重定位中幸存**：一位用户遇到 `UnsafePointer` 在结构体因内存重定位被添加到 `List` 时失效的问题。
   - 建议的解决方案是使用 `__moveinit__()` 函数（以及 `__copyinit__()`）来处理结构体对象移动后 `UnsafePointer` 的更新。
- **为什么 Mojo 拥有原生集合**：出于性能考虑，Mojo 实现了 Python 集合的原生版本，因为避免与 Python 互操作速度更快。
   - Mojo 中的原生集合还允许更好的类型安全强制执行，并作为语言开发的测试台。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1433531294976970894)** (16 messages🔥): 

> `AMD GPU Support, MAX vision/roadmap, Op Staging Time, ComfyUI Mojo` 


- **古老的 AMD GPU 获得一线希望**：开发者正致力于让旧的 AMD GPU 运行，甚至是那些在经历了几代更迭后已不被 ROCm/HIP 驱动支持的型号，尽管建议现代 CPU 可能会表现更好。
   - 一位成员并不认为它*不能*工作，并指出*如果这些路径因意外而能运行，开发者往往不会刻意破坏它们*。
- **MAX 路线图仍未明确**：尽管 Mojo 路线图反响良好，但 MAX 目前还没有路线图，尽管正在考虑中。
   - 一位成员指出：*鉴于 Mojo 路线图的反响如此之好，我想我们也愿意为 MAX 做同样的事情，但我现在不能保证任何事情。*
- **算子暂存时间（Op Staging Time）变慢的研究**：在一位成员分享了 [GitHub issue 5184](https://github.com/modular/modular/issues/5184#issuecomment-3474920771) 的链接后，最近进行了更改以减少图（graphs）的算子暂存时间。
   - 一位用户报告称，有的图需要 *1 小时以上来声明*。
- **分享 ComfyUI-Mojo 基准测试**：一位成员分享了 [GitHub 上的 ComfyUI-Mojo 链接](https://github.com/owenhilyard/comfyui-mojo)，作为减少算子暂存时间的基准测试。
   - 分享者认为他们*遇到了 Torch MAX 后端的一个边缘情况，导致某些算子被分解得比预想的要深得多。*


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1433532463698808843)** (93 条消息🔥🔥): 

> `Codex Credits, Context-Bench 基准测试, Enterprise AI, kimi-cli Checkpoints, CoreWeave 收购 Marimo` 


- **OpenAI 推出 Codex 使用额度**：[OpenAI 推出了按需付费额度](https://x.com/OpenAIDevs/status/1983956900602581254)，价格为 **每 1,000 额度 40 美元**，用于 **ChatGPT Plus/Pro** 上的额外 **Codex** 使用，并为所有用户重置了速率限制。
   - 社区反应不一，用户要求明确额度与 API 定价的区别、推出中级方案、提供使用分析、明确额度过期时间，并呼吁增加更多 **Codex** 功能。
- **Context-Bench 发布，用于开放 Agent 基准测试**：[Letta_AI 发布了 Context-Bench](https://x.com/Letta_AI/status/1983983515336405144)，这是一个防污染的基准测试，用于评估模型在长跨度文件操作、多步工具调用和成本方面的表现。
   - **Sonnet 4.5** 以 **74%** 的得分领先；尽管 Token 更便宜，**GPT-5** 的价格依然更高；开源模型正在缩小与闭源模型的差距。
- **DeepAgents CLI 具备持久化记忆**：[Harrison Chase 介绍了 DeepAgents CLI](https://xcancel.com/hwchase17/status/1984303925101735950)，这是一个基于新 deepagents 包构建的示例编码应用，可以在不同会话之间保留指令和引导。
   - 该发布包含一篇 [博客文章和演示视频](https://xcancel.com/hwchase17/status/1984303925101735950)，定位为可定制 Agent 的“开放测试框架 (open harness)”，**LangChain** 团队暗示即将推出增强功能。
- **CoreWeave 收购 Marimo**：[CoreWeave 正在收购 Marimo](https://xcancel.com/l2k/status/1984021111718473898)，称赞 **Marimo** 团队开发了深受喜爱的工具，并对此次合作表示兴奋。
   - 成员们对此次收购感到兴奋，有人表示 *“希望一切顺利，我非常喜欢 marimo”*。
- **Poolside 的估值遭到抨击**：技术圈内人士正在嘲讽 [Poolside 120 亿美元的估值](https://xcancel.com/julienblanchon/status/1984337407097909629)，指责其是在加勒比避税天堂运行的虚假软件 (vaporware)。
   - 评论者指出，该公司曾推销自己是“Cursor 之前的 Cursor”，但从未发布产品，经历了多次转型，且在巴黎的见面会中几乎销声匿迹。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1433596355552477204)** (9 条消息🔥): 

> `海螺 AI (Hailuo AI), MiniMax Music 2.0, 阿里巴巴 Wan 2.2 换脸, 实时语音/动作映射` 


- **海螺 AI 发布 MiniMax Music 2.0！**：[海螺 AI (Hailuo AI)](https://x.com/Hailuo_AI/status/1983964920493568296) 推出了 **MiniMax Music 2.0**，这是一个生成式音乐平台，可以创作 **5 分钟** 的专业级歌曲。
   - 该平台具有逼真的人声，并支持在流行、爵士、蓝调、摇滚、民谣、二重唱和阿卡贝拉等风格中进行多乐器控制；用户反馈包括请求语言支持、更长的歌曲限制、开源以及纯乐器模式。
- **阿里巴巴 Wan 2.2 换脸效果惊人！**：一段 **Wan 2.2** 将男性的声音/动作映射到女性虚拟形象上的视频引发了从惊叹到恐惧的各种反应；[点击此处查看视频](https://x.com/mylordbebo/status/1983846299586683236?s=46)。
   - 担忧包括网络诈骗 (cat-fishing)、深度伪造 (deep-fake) 诈骗、猫耳同步故障以及手指同步失败。一些人开玩笑说要成为“电子女孩 (egirls)”、开启 OnlyFans 淘金热，以及对真人认证的需求；反向观点则包括在残障辅助/教育方面的潜在积极用途，并预测 AI 垃圾内容 (AI slop) 将迫使人们回归现实生活 (IRL) 互动。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1433776969350381719)** (1 条消息): 

> `Web Agent, DSPy 与 GEPA, 强化学习算法, 初学者友好的 AI Agent` 


- **Web Agent 寻求贡献**：一位成员开发了一个能够跨所有网站搜索的 **Web Agent**，目前正在寻求熟悉 **DSPy**、**GEPA** 和其他 **强化学习算法** 的贡献者，代码库见 [repo](https://github.com/raj-gupta1/raj_agiinc)。
   - 该项目旨在做到初学者友好，让刚接触 AI Agent 的人也能轻松上手。
- **AI Agent 代码库开放合作**：一位开发者邀请他人为其 **AI Agent** 代码库做出贡献，特别是具有 **DSPy**、**GEPA** 和各种 **强化学习算法** 专业知识的人员，代码库见 [repo](https://github.com/raj-gupta1/raj_agiinc)。
   - 该代码库结构易于理解，迎合了 AI Agent 领域的初学者。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1433555562116943933)** (91 messages🔥🔥): 

> `DSPy 中的 BAML Adapters，JSON schema 的缺点，DSCloj 发布` 


- **BAML Adapters 在 DSPy 中获得关注**：一位成员使用 **BAMLAdapter** 进行结构化输出，表达了对 JSON schema 的厌恶，认为其既浪费又令人困惑，特别是在并购（M&A）用例中从非结构化文本提取结构化信息的任务。
   - 该成员澄清，在 DSPy 中使用 **BAMLAdapter** 不需要 BAML 客户端或 CLI，只需通过 `from dspy.adapters.baml_adapter import BAMLAdapter` 导入即可（适用于 DSPy 版本 > 3.0）。
- **JSON Schema 因浪费 Token 遭到抨击**：一位成员反对在 LLM 中使用 JSON schema，认为 [JSON schema 客观上更差](https://github.com/prrao87/structured-outputs)，且 LLM 在没有它的情况下表现更好，原因是冗长的描述符和 Token 间距问题会干扰 LLM。
   - 他们强调，JSON schema 在 Token 方面可能造成高达 **4倍** 的浪费，并建议质疑在已知其缺点的情况下为何仍在使用它；他们还发现即使没有 Schema Aligned Parsing (SAP)，BAML 在 DSPy 中也能获得极佳的效果。
- **DSCloj：DSPy 的 Clojure 移植版本发布**：一位成员发布了 [DSCloj](https://github.com/unravel-team/DSCloj)，这是 DSPy 的 Clojure 移植版，并指出它仍处于 alpha 阶段，正在寻求对 API 的反馈。
   - DSCloj 是一个非常年轻且处于 alpha 阶段的新库，欢迎对 API 提供反馈。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1433604218366857327)** (61 messages🔥🔥): 

> `Paper Dump 频道反馈，论文策展偏好，用于论文筛选的 AI，ArXiv 论文统计，AI 研究方法论` 


- **论文泛滥引发反馈与过滤修复**：成员们建议 **paper-dump** 频道的频繁发布者应合并帖子，按重要性对分享的论文进行排序，灵感可能来自 [Elvis Saravia](https://nlp.elvissaravia.com/t/ai) 和 [ByCloud](https://x.com/TheAITimeline) 等来源的每周 AI 论文回顾。
   - 一位用户指出，“应优先考虑重要性而非数量”，并提出了防止其他发布者被淹没以及提高频道实用性的方法。
- **自动化 Agent 旨在获取优质 AI 文章**：一位成员表示有兴趣创建一个 Agent 或机器人来根据个人相关性预过滤论文，利用 [AlphaXiv](https://www.alphaxiv.org/) 和 [Emergent Mind](https://www.emergentmind.com/) 等资源来寻找 **热门 AI 论文**。
   - 另一位成员同意这些是很好的人工过滤器，并建议针对特定的 AI 子领域订阅电子邮件简报或 RSS 订阅源。
- **考虑策展内容频道概念**：参与者讨论了将 **paper-dump** 频道重命名为类似 **curated-papers** 的名称以明确其目的，一位成员建议每位发布者每天最多发布两篇论文，因为他们觉得该频道已经变得像在发垃圾信息。
   - 一位成员分享道：“我认为你误解了这个频道的初衷。尽管叫 paper-dump，但它是为了让人们发布已经过策展、值得阅读的论文。”
- **ArXiv 分析：AI 在学术档案中崛起**：有人提到 [cs.AI](https://arxiv.org/abs/2510.26275) 是 arXiv 中的领先类别，全天候运行，不像其他领域遵循周一至周五的时间表。
   - 其他人提议将频道拆分为五个子频道，按不同的 AI 重点分类，如机器学习数学理论、通用机器学习、应用 AI、用于自动化日常工作的 AI、用于自动化教育和研究的 AI。
- **AI 辅助建议：验证，验证，再验证！**：当被问及使用 **AI 辅助 AI 研究** 时，给出的指导强调需要验证所有的主张和来源以避免幻觉。
   - 一位成员引用了 Andrej Karpathy 的话，指出“做 AI 最好的方法就是尝试一堆模型解决方案，看看哪个效果最好。”


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1433531409011839168)** (11 条消息🔥): 

> `Contrastive Concept Vectors, Model Detection, Spiking Neurons, Halloween Paper Discussions` 


- **对比概念向量影响模型权重！ (Contrastive Concept Vectors Sway Model Weights!)**: 一位成员讨论了使用 **contrastive concept vectors** 来影响模型权重，并在 pre-prompt 中告知模型他们确实对其进行了干预；一张 [图表](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F212fe68c8e677fdd9daf79301d2522d6923bed1d-2970x2476.png&w=3840&q=75) 对比了模型正确检测到干扰的频率与对照组的差异。
   - 他们观察到某些模型有时能检测到这种操纵，同时批评该文章*辞藻华丽且极其冗长*，并推断出了实验并不支持的投机性结论。
- **脉冲神经元资源引起关注 (Spiking Neuron Resources Spark Interest)**: 分享了关于 **spiking neurons** 的资源链接，包括一篇 [bioengineer.org 的文章](https://bioengineer.org/novel-spiking-neuron-combines-memristor-transistor-resistor/) 和 [Spiking Heidelberg Datasets (SHD)](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)。
   - 还有一个指向 [ustypology.github.io](https://ustypology.github.io/) 的链接。
- **万圣节 AI 论文聚会？ (Halloween Hangout on AI Papers?)**: 一位成员询问是否有人想在 **周五万圣节之夜** 讨论论文，同时也承认这可能不是一个最佳日期。
   - 他们提到正在阅读 [这篇论文](https://arxiv.org/abs/2509.19228)。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1433776842862629026)** (1 条消息): 

> `Web agent, DSpy, GEPA, Reinforcement learning` 


- **Web Agent 开放贡献 (Web Agent Opens for Contributions)**: 一位成员构建了一个能够在所有网站上进行交互的 **web agent**，并正在寻求熟悉 **DSpy**、**GEPA** 和其他 **reinforcement learning** 算法的贡献者。
   - 该 [GitHub 仓库](https://github.com/raj-gupta1/raj_agiinc) 旨在对 **AI agents** 初学者友好。
- **招募贡献者 (Call for Contributors)**: 该项目寻求贡献以增强其 web agent。
   - 鼓励在 DSpy 和 reinforcement learning 等领域具有专业知识的感兴趣开发者加入并贡献力量。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1433612113821175859)** (5 条消息): 

> `Claude AI Naming, Name Usage Trends` 


- **关于 Claude AI 命名的辩论 (Debate Sparked on Claude AI Naming)**: 讨论围绕将 AI 模型命名为 **"Claude"** 背后的意图展开，认为这是一个刻意的选择，而非由于该名字罕见而产生的巧合。
   - 一位成员将其比作给孩子起名 **"Adolf"**，认为这类选择很少是随机的；而另一位成员则反驳说，名字的罕见性对于可识别性来说并不一定是负面的。
- **姓名使用数据受到质疑 (Name Usage Data Under Scrutiny)**: 一位成员认为人名中不存在噪声底限（noise floor），因为每一个数据点都被精确地记录在案（在国家人口和普查登记中）。
   - 另一位成员反驳道，*Anthropic 出于其他随机原因选择 "Claude" 的概率，与该名字的整体使用趋势有关。*


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1433532029349396530)** (56 条消息🔥🔥): 

> `32GB VRAM + 64GB RAM 的模型推荐、跨不同应用共享 LLM 模型、报告安全问题、GSoC 26、多 GPU 推理` 


- **在有限显存中挤入更大的模型**：一位成员正在寻求适用于 **32GB VRAM + 64GB RAM** 的代码模型推荐，目前正在使用 **qwen3-coder:30b**，并正在探索量化模型，以尝试在现有硬件中运行更大的模型。
   - 他们不确定在这些硬件限制下，哪些量化模型选项最适合进行微调。
- **单一存储驱动器上的 LLM：Ollama、LM Studio 和应用大团结！**：一位成员询问是否可以将 LLM 模型下载到单个存储驱动器中，供 **Ollama、LM Studio** 和其他应用共同使用，以避免重复下载。
   - 答案是肯定的，前提是所有应用都支持相同的模型格式，这可能需要进行格式转换、查阅文档，并使用 **transformers 格式模型**或单文件 **safetensors**。
- **多厂商 GPU 推理加速**：一位成员寻求支持跨不同厂商（Intel 和 NVIDIA）GPU 进行**多 GPU 推理**的推理程序推荐。
   - 根据以往经验，有人推荐了 Accelerate 以及 [Petals](https://github.com/bigscience-workshop/petals)，尽管后者对多种 GPU 类型的兼容性仍不确定。
- **NSFW 剧本编写：Llama3 微调模型来救场**：一位游戏开发者寻求不受审查的模型推荐，用于评估和改进露骨的性爱场景，理由是 **ChatGPT 的审查**问题。
   - 有人建议 Claude 的限制比 ChatGPT 少，而且针对 NSFW 内容的 **Llama3** 微调模型可能是一个可行的选择，并指向了带有 *abliterated* 标签的模型。
- **Qwen 2 35B 获得 WebUI 支持**：一位成员分享了运行 **Qwen 2 35B**、构建自定义 WebUI 以及开发自己模型的进展，并在[这段 YouTube 视频](https://www.youtube.com/watch?v=qw4fDU18RcU)中展示了结果。
   - 演示者提到他们*增加了更多 GPU* 以成功运行该模型，但未提供关于模型本身的更多细节。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1433797144007151686)** (1 条消息): 

> `ML 数学课程、ML 工程师之路` 


- **成员寻求 ML 数学课程推荐**：一位拥有**计算机科学学士学位**的成员正在寻求优质数学课程推荐，以加强对 **Machine Learning 研究论文**的理解和直觉。
   - 他们的目标是巩固数学基础，以便更好地理解 **ML 研究**。
- **为 ML 研究巩固数学基础**：该成员希望提高他们的**数学理解能力**，以更好地掌握 **Machine Learning** 领域的**研究论文**。
   - 他们认为更强大的数学基础对于成为 **ML 工程师**的职业生涯至关重要。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1433545437511548992)** (5 messages): 

> `Snippet Creator, Qwen3-4B-Reasoning-Backfill-v0.1, Web Agent with DSpy and GEPA, Torque DSL for Synthetic Datasets` 


- ****Snippet Creator** 通配符文本搜索器发布**：一名成员宣布了他们的 **Snippet Creator**，这是一个带有简单通配符文本搜索的 embedder，允许用户通过精确匹配创建自定义代码片段 ([huggingface.co/kalle07/raw-txt-snippet-creator](https://huggingface.co/kalle07/raw-txt-snippet-creator))。
   - 该成员分享了一个链接，并表示：“简单来说，它是一个 embedder，但带有简单的通配符文本搜索……这允许你根据所需的精确匹配创建自己的代码片段。”
- ****Qwen3-4B** 合成推理轨迹**：一名成员发布了 **Qwen3-4B-Reasoning-Backfill-v0.1** ([huggingface.co/joeyzero/Qwen3-4B-Reasoning-Backfill-v0.1](https://huggingface.co/joeyzero/Qwen3-4B-Reasoning-Backfill-v0.1))，用于为旧的或对话类数据集合成推理轨迹（reasoning traces）。
   - 他们从现有的推理数据集中拼凑逻辑，以便从给定的输入和输出中推断推理过程，并对结果感到满意。
- **Web Agent 寻求 DSpy 和 GEPA 贡献者**：一名成员创建了一个能够跨所有网站搜索的 Web Agent ([github.com/raj-gupta1/raj_agiinc](https://github.com/raj-gupta1/raj_agiinc))。
   - 该成员邀请大家为该仓库做贡献，特别是熟悉 **DSpy**、**GEPA** 和其他强化学习算法的人员，并指出代码库对初学者很友好。
- ****Torque** DSL 生成指令数据集**：宣布了 **Torque**，这是一个用于生成合成 Instruct 基础数据集的声明式 DSL ([github.com/qforge-dev/torque](https://github.com/qforge-dev/torque))。
   - 它允许组合流程、生成现实的变化、类型化工具调用、带种子的并发运行以及与供应商无关的 LLM 使用；基于 **TypeScript** 和 **Zod** 构建。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1433748364557553748)** (1 messages): 

> `Math Intuition, Paper Reading, Model Understanding, Math Courses` 


- **寻求数学课程推荐**：一名成员正在寻求 **数学课程** 推荐（YouTube、Udemy 等），以提高他们的 **直观理解** 并温习本科阶段的技能，旨在增强 **论文阅读** 和 **模型理解** 能力。
- **提升研究所需的数学技能**：该成员希望提高他们的 **数学直觉**，以便更好地理解研究论文和模型，并正在寻找课程推荐来刷新他们的知识储备。


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1433660474343292949)** (1 messages): 

> `Krea realtime video, Optimize runtime` 


- **Krea 实时视频崭露头角**：成员们正在讨论 **Krea 实时视频** 及其功能。
   - 该工具已在频道内被提及和分享，表明了大家对其实时视频生成功能的兴趣。
- **Sayak 优化 Krea 运行时**：一名成员分享了一项关于优化 **Krea 实时视频** 运行时的 [研究](https://x.com/RisingSayak/status/1983873124220445183)。
   - 该研究提供了增强 Krea 视频处理性能的见解和技术。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1433569388023513110)** (3 messages): 

> `InstantID and IP-Adapter FaceID, Consistent 2D Style Transfer, BYOL model usage, Synthetic Image Generator` 


- **提议使用即时身份保留插件**：一名成员建议使用 **InstantID + IP-Adapter FaceID** 或 **ControlNet reference-only 设置**，以在生成的图像中更好地保留身份。
   - 这些工具旨在比标准方法更有效地保持面部特征和相似度。
- **建议通过 Lora 训练实现一致风格**：为了实现一致的 **2D 风格迁移**，一名成员建议 *训练一个带有冻结文本编码器的 Lora* 或 *切换到 InstructPix2Pix / T2I-Adapter 模型*。
   - 用户补充说，这通常比 SD 默认的 image2image 模式产生更干净、风格更一致的结果。
- **BYOL 模型投入使用**：一名成员询问频道是否有人研究过 **BYOL (Bootstrap Your Own Latent)**。
   - 频道尚未做出回应。
- **合成图像生成器正在开发中**：一名成员和朋友正在构建一个 **合成图像生成器**，它可以自动添加标签和边界框，允许用户导出到 **YOLO, COCO** 等格式。
   - 用户询问了所需的数据集和工具，并提出提供生成结果以针对真实场景进行测试。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

sebizaur: 不
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1433742515605078067)** (3 messages): 

> `Agents course leaderboard API outage, Hugging Face subscription frustrations, Files endpoint issue` 


- **Agents 课程排行榜 API 崩溃**: 成员报告称 **Agents 课程排行榜 API** 已经 **宕机一周**，且 Hugging Face 没有任何官方沟通。
   - Questions 和 Submit 端点已恢复运行，但 **Files 端点** 似乎仍然处于宕机状态，这让 Pro 版本的订阅者感到非常沮丧。
- **订阅者发泄订阅挫败感**: 由于 **持续的 API 问题**，几位用户对他们的 Hugging Face Pro 订阅表示不满。
   - 一位用户表示，他们订阅 Pro 是为了充分利用这门课程，但由于 Files API 似乎仍然宕机，无法使用订阅功能，这令人非常沮丧。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1433746697313583125)** (3 messages): 

> `CUDA training` 


- **教授 CUDA，但未检测到兴趣**: 一位用户尝试教别人 CUDA，但报告称学生目前对此并无感。
   - 另一位用户回应道：*你得先训练（train）他。*
- **祝贺与会面**: 用户们互相祝贺见面，并用表情符号表达了积极的情绪。
   - 这种情绪通过 goku 和 slothhug 表情符号来表达。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1433900267941925046)** (6 messages): 

> `Triton MLIR pass errors, Source code attribution, nvfp4 and mxfp4 compilation, Triton backward autograd` 


- **在 Triton MLIR Pass 错误中追踪源码**: 一位用户询问如何在 **Triton MLIR pass 错误** 中启用 **源码归属（source code attribution）**，并指出当前的错误仅提供 **TT-IR** 和 **MLIR pass 复现器**。
   - 错误消息指向的是函数签名，而不是导致失败的具体代码行。
- **NVFP4 vs MXFP4 对决：Blackwell vs RTX 4090**: 一位用户发现 **nvfp4** 无法在 **Blackwell** 以下的任何设备上编译，而 **mxfp4** 可以在 **4090** 上编译。
   - 同样的错误也出现在 **RTX 5090** 上，这让用户无从得知 [Gemlite 是否在 RTX 5090 上支持 nvfp4](https://link.to/gemlite)。
- **Triton_bwd 发布：Triton Kernel 的自动微分**: **Triton_bwd** 是 **Triton** 的一个封装，允许在 **PyTorch autograd** 中使用 **Triton kernels**。
   - 更多详情可以在 [GitHub](https://github.com/daniel-geon-park/triton_bwd) 上阅读，该项目在 [这篇博客文章](https://park-geon.com/2025/10/30/triton-bwd/) 中有进一步解释。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1433549517357322352)** (4 messages): 

> `FlashAttention-4, RTX 50, Apache Spark, gau-nernst/fa-5090` 


- **FA4 激发了对 RTX 50 系列的兴趣**: 一位成员表示有兴趣为即将推出的 **RTX 50 系列** 和 **Apache Spark** 实现 **FlashAttention-4 (FA4)**。
- **资源助力 FA4 实现快速启动**: 另一位成员建议将 [gau-nernst/fa-5090](https://gau-nernst.github.io/fa-5090/) 仓库作为实现 **FA4** 的起点。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1433548225897562224)** (3 messages): 

> `CUDAGraphs and OOM, Freezing option in torch inductor, PyTorch Distributed Memory Usage, Peak VRAM Usage with CUDAGraphs` 


- **调试 CUDAGraphs OOM 的起源**: 一位成员将 **CUDAGraphs OOM** 错误追溯到 **torch inductor** 中的 *freezing* 选项。
   - 他们假设图捕获（graph capture）可能在 freezing pass 之前开始，导致重放参数复制和输入重建，但后来确定是 **freezing pass** 本身导致了 OOM。
- **调查 PyTorch 分布式内存报告**: 一位成员报告在 **PyTorch distributed** 使用期间看到了 `[1] [1] 17592186044416.0 MB was used for memory usage tracing!` 这行内容，并正在寻找其来源。
   - 他们正试图识别 PyTorch 中产生此内存使用追踪消息的具体代码行。
- **VRAM 谜团困扰 CUDAGraphs**: 尽管根据权重和激活大小在理论上是可行的，但一位成员仍遇到了 **CUDAGraphs** 的 **OOM** bug。
   - 他们澄清说，问题不在于输入重复或 CUDAGraphs 相关的 bug，而是在 **Dynamo** 和 **Inductor** 操作期间内存使用过高，目前正在寻求使用 torch **CUDAGraphs** 以及不同 **Dynamo** 和 **Inductor** pass 时峰值 VRAM 使用的逻辑/数学计算方法。


  

---

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1433568350860218441)** (8 messages🔥): 

> `Hardware Friendly Top-K Logits Algorithm, Radix Sort for Top-K Selection, CCCL/CUB TopK Implementation` 


- **硬件爱好者寻找硬件友好型 Top-K Logits 算法**：一位成员正在寻找一种硬件友好型算法，用于处理序列长度从 **4K 到 128K** 的 **top-k logits**，并建议采用并行排序和合并，但指出了合并过程中的瓶颈。
   - 另一位成员建议查看 [FlashInfer](https://flashinfer.ai/2025/03/10/sampling.html) 以获取相关的 kernel。
- **基数排序（Radix Sort）成为高效 Top-K 选择的首选**：有人建议在 *k << N*（k 远小于 N）时使用**基于基数（radix-based）的方法**，并指出这是常用方法；而如果 k 接近 N，则全排序更高效。
   - 他们提到 **PyTorch 的 topk** 同时实现了这两种方法，并根据启发式规则进行切换，同时指向了 *rocprim* 中的一个实现。
- **NVIDIA 正在开发 CCCL/CUB TopK 实现**：一位成员分享了 **CCCL/CUB** 中已经存在 **TopK 实现**，虽然尚未发布，并链接到了[相关的 GitHub pull request](https://github.com/NVIDIA/cccl/pull/5677)。
   - 未提供更多细节。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1433832777807040532)** (2 messages): 

> `Discrete Diffusion Models Reading Group, Opal Language, Opportunistic Evaluation Strategy` 


- **离散扩散模型（Discrete Diffusion Models）讨论开启**：一位成员分享了 [d-llms.io](https://d-llms.io/) 的链接，发起了一个专门针对语言应用的**离散扩散模型**读书会。
- **Opal 脚本语言提升 LLM 性能**：一位成员分享了 [Opportunistically Parallel Lambda Calculus](https://doi.org/10.1145/3763143) 的链接，介绍了 **Opal**，这是一种采用机会主义评估策略（opportunistic evaluation strategy）的脚本语言，用于并行化独立的外部调用，对使用 **LLMs** 和其他 API 的程序特别有利。
- **Opal 相比 Python 实现速度提升**：**Opal** 语言在其 [GitHub 仓库](https://github.com/stephenmell/opal-oopsla2025-artifact)中有详细介绍，并计划在 **OOPSLA 2025** 上展示。它展示了显著的性能提升，与标准顺序执行的 Python 相比，总运行时间快了高达 **6.2 倍**，延迟降低了 **12.7 倍**。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1433538374635094117)** (5 messages): 

> `Dusty NV, Jetson Containers, Pip Index URL, neurondeep maintainer` 


- ****Dusty** 离职，**neurondeep** 接手**：在 **Dusty** 退休后，**neurondeep** 现在是 [Jetson Containers](https://github.com/dusty-nv/jetson-containers) 的维护者。
- **Pip Index URL 问题**：容器 `dustynv/pytorch:2.7-r36.4.0-cu128-24.04` 的 pip index-url 已失效或错误。
   - 要通过 pip 安装任何内容，你需要使用 `pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu128 PACKAGE`，因为默认的 `https://pypi.jetson-ai-lab.dev/jp6/cu128` 是错误的。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1433585805955170554)** (1 messages): 

> `PMPP book, FLOPs calculation, Global memory access` 


- **PMPP 习题疑问**：一位成员就 **PMPP 书籍** 第 5 章习题 11 的 f 部分寻求解答。
   - 具体来说，他们对附件截图代码片段中 **FLOPs** 和**全局内存访问**的计算表示疑问。
- **索引加法澄清**：该成员询问索引加法是否应被视为 **FLOPs**，并指出第 14 行包含 **11 个操作（5 次乘法、5 次加法和 1 次取模）**。
   - 他们还询问了在 **OP/B 计算**中是否应计入全局内存存储（stores）。
- **全局内存访问评估**：用户计算出由于在第 **7**、**12** 和 **14** 行访问了 **x**、**a** 和 **b**，共有 **6 次全局内存加载，每次 4 字节**。
   - 他们询问是否也应考虑对全局内存的存储操作。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1433577902284210277)** (5 条消息): 

> `Float8LinearConfig 使用, AWQ/SmoothQuant 推理, TorchAO FP8 量化 Bug, DeepSeek v3, GemLite 内核` 


- **Float8LinearConfig 被误解**：`Float8LinearConfig` 旨在与 `convert_to_float8_training` 配合使用，可能无法完全兼容 TorchAO 的推理 API。
   - 对于推理，建议使用全局变体，如 `Int8WeightOnlyConfig`、`Int4WeightOnlyConfig` 和 `Float8WeightOnlyConfig`。
- **用于推理的 AWQ/SmoothQuant 量化**：对于推理任务，推荐使用 **AWQ** 和 **SmoothQuant** 作为广泛使用的训练后量化 (PTQ) 方法，目前已兼容 vLLM，API 使用参考[此 Pull Request](https://github.com/pytorch/ao/pull/2906)和[此 Pull Request](https://github.com/pytorch/ao/pull/3010)。
   - 提到了一篇在各种格式和任务中比较 **AWQ**、**GPTQ** 和 **SmoothQuant** 的论文（[论文链接](https://arxiv.org/html/2411.02355v3)）。
- **FP8 训练与量化说明**：**FP**、**BF**、**INT** 和 **MXFP** 是数据类型，使用 FP8 权重训练的模型可以将其激活值量化为更低精度的格式，如 **FP8** 或 **INT8**。
   - 最初的问题是：*FP8 模型的推理（假设是在 FP8 下训练的，如 DeepSeek v3）是否需要对激活值进行量化。*
- **TorchAO 的 FP8 Bug？**：一位用户报告了 **TorchAO** 默认 **FP8** 量化中可能存在的 Bug，在两块不同的 RTX 5090 上使用 `torchao.quantization.Float8WeightOnlyConfig` 对 **Llama 3.1-8B** 进行推理，仅观察到 **7.8 tps**。
   - 使用其他配置或显式调用带有 **mxfp8** 的 **GemLite** 内核可获得更好的速度，详见[此基准测试结果表](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main?tab=readme-ov-file#benchmarking-results-on-1-gpu)。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1433556570326962398)** (6 条消息): 

> `MI300X, TFLOPS, HBM 带宽, clpeak, RadeonFlow FP8 GEMM` 


- **MI300X：验证理论 TFLOPS 和 HBM 带宽**：一位成员想要基准测试/验证 **MI300X** 的理论 **TFLOPS** 和 **HBM 带宽**数值。
   - 建议他们使用 [clpeak](https://github.com/krrishnarraj/clpeak) 来测试向量吞吐量和全局内存带宽。
- **RadeonFlow 未达到声称的 FP8 性能**：一位成员测试了 **RadeonFlow FP8 GEMM** 内核，最高仅达到 **779.82 TFLOPS** 的 FP8 性能，未达到声称的 **2614.9 TFLOPS**。
   - 他们指出 [RadeonFlow](https://github.com/RadeonFlow/RadeonFlow_Kernels) 仅有 **30%** 的效率，而任何验证方法都应至少达到理论值的 **70%**。
- **AMD 微基准测试套件**：一位成员建议使用他们的微基准测试套件 [amd-experiments](https://github.com/Snektron/amd-experiments) 来验证 **MI300X**。
   - 未提供关于该套件功能或具体用例的进一步信息。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1433722440109916200)** (4 条消息): 

> `Apple 产品, GPU 编程, Metal API, 开发者后悔` 


- **Apple 产品引起开发者愤怒**：一位用户感叹道：*其他人都吸取了教训，停止使用 Apple 产品了。*
   - 该用户对尝试使用 **Metal** 进行 **GPU 编程**表示后悔。
- **Metal API：孤独开发者的挑战**：一位开发者发现自己独自在一个以 **Metal** 为中心的频道中，幽默地指出了缺乏 Apple 爱好者同伴的现状。
   - 从 **Metal** 开始 **GPU 编程**，该开发者预见到未来会出现问题，预示着一条潜在的充满挑战的道路。


  

---

### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1433534010549604403)** (6 messages): 

> `Executorch, vLLM, TorchScript, Torch.package` 


- **Executorch 尚未达到生产级服务器推理标准**：一位成员指出 [Executorch](https://pytorch.org/executorch/) **尚未达到生产就绪状态**，特别是 **CUDA backend**，因为它仍处于活跃开发阶段。
   - 建议对于任何大型模型使用基于 Python 的服务（如 **vLLM**），因为它速度快且易于处理。
- **TorchScript 未能解决开销问题**：据一位成员称，TorchScript 无法解决框架开销（overhead）问题，因此还不如直接运行 Python。
   - 如果在没有 Python 的情况下需要 C++ 环境，建议采用 **python vLLM serving** 的多层解决方案。
- **Torch.package 支持极少**：一位成员表示，虽然 [torch.package](https://pytorch.org/docs/stable/generated/torch.package.PackageImporter.html) 确实有效，但支持非常有限。
   - 建议使用 **hf/hub** 和 **dependency pinning**（依赖锁定）作为最简单且最可靠的解决方案。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1433875433950281830)** (1 messages): 

> `Agent inventory, Inventory Visibility` 


- **Agent 的隐形背包**：一位成员报告称其 Agent 收集了 **stone**（石头）和 **coal**（煤炭），但尽管 trace 文件显示有变化，这些物品在背包（inventory）中并不可见。
   - 该用户正在寻求解决这一差异的建议。
- **背包可见性问题**：用户遇到收集的物品（石头和煤炭）未显示在 Agent 背包中的问题，尽管 trace 文件中的证据表明并非如此。
   - 该咨询是寻求帮助以诊断并修复背包可见性问题。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1433576774876397599)** (3 messages): 

> `Winner Solutions Performance, AMD Solutions Study, CPU Overhead Measurement` 


- **获胜 Kernel 比 SoL 慢 10 倍**：一位成员发现获胜方案比 **SoL (Speed of Light)** 慢 **10 倍**，这很奇怪，因为他们原本预期手动调优的 Kernel 性能应该接近。
   - 该成员希望 **AMD** 拥有接近 **SoL** 的有效解决方案，并表示有兴趣对其进行研究。
- **测量 CPU 开销**：一位成员提到他们可能测量到了一些 **CPU overhead**，这在多次运行之间保持不变。
   - 他们还表示，在这些问题上达到 **SoL** 是极具挑战性的。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1433976053172535366)** (1 messages): 

> `Compute Sanitizer, Frame Info` 


- **Compute Sanitizer 抛出无效全局读取错误**：在运行 `kernel_cutlass_kernel_flash_attncuteflash_fwdFlashAttentionForwardSm90_object_at__tensor0000o11101213_tensor0000o11101213_tensor0000o11101213_tensor0000o11101213_tensorptrf32_gmem_o_1_Non_0+0x3b70` 时，Compute Sanitizer 抛出了 *Invalid __global__ read of size 2 bytes* 错误。
   - 该错误发生在 flash_fwd.py:788，表明在地址 0x7f5e369a0600（大小为 8 字节）最近的一次分配之后 57 字节处发生了越界访问。
- **希望能有更细致的 Frame 信息**：一位用户询问 Compute Sanitizer 在调试设备编译的 Kernel 时，是否预期能提供更细粒度的 Frame 信息。
   - 他们补充说，在他们的具体案例中，已经知道读取来源，原本希望 Sanitizer 能指向确切的位置。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/)** (1 messages): 

matt.pd: 通过 FP16 消除训练-推理不匹配 (Defeating the Training-Inference Mismatch via FP16)
https://arxiv.org/abs/2510.26788
  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1433534172558921738)** (1 messages): 

> `Helion PR 1053 Feedback, TorchAudio Datapipe Issue` 


- **Helion PR 寻求审查**：一位成员请求对其 [Helion pull request #1053](https://github.com/pytorch/helion/pull/1053) 提供反馈。
   - 消息中未提供关于该 PR 的更多细节。
- **TorchAudio Datapipe 问题**：一位用户报告了 TorchAudio Datapipe 的一个问题，提到它*即使在使用本地文件时也需要互联网连接*。
   - 消息中未讨论关于该特定问题或潜在解决方案的进一步细节。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1433711839556010055)** (38 messages🔥): 

> `可学习激活函数，FP16 对比 BF16 精度，AGI 的不可能实现性` 


- **新型 MLP 神经元学习其自身的激活函数**：一位成员分享了一个小型 **MLP** 实验的图像，其中每个神经元学习其自身的激活函数，在 **CIFAR-100** 上训练时产生了一些不寻常的非线性激活形状。
   - 该成员指出 Loss 有时会爆炸，有时会下降，目前仍在调查原因；另一位成员建议，那些形状畸形的可能是从未被激活的神经元，并建议进行 *显著性分析 (Saliency Analysis)*。
- **BF16 的实用性受到质疑**：成员们讨论了 **BF16** 相对于 **FP16** 的实用性，思考适当的归一化和截断（clipping）是否能减少对 **BF16** 更宽动态范围的需求。
   - 一位成员引用了一篇论文（[BF16 训练的数值稳定性](https://arxiv.org/abs/2510.26788)）并指出，虽然预训练设定了 **BF16** 处理良好的动态范围，但 **RL** 可能需要 **FP16** 更高的精度；不过，他们一致认为 *偏置修正（bias correction）应该能以另一种方式解决这个问题*。
- **16 个问题决定 AGI**：一位成员分享了他们的工作（[16 Questions Is All You Need](https://sutanisurabu.substack.com/p/16-questions-is-all-you-need-the)），旨在降低评估成本并缓解基准测试作弊（benchmark hacking），提出了一种利用从 **LLM** 问题概率分布中提取的仅 **16** 个问题来开发更准确基准测试的方法。
   - 该成员声称，这意味着现代 **AI** 模型不可能实现 **AGI**，因为所有模型的性能都会随着问题的罕见程度成比例下降，这表明 **LLM** 能力的结构由于缺乏流体智能而与人类能力差异巨大；因此，**LLM** 中真正的流体推理最接近于 **In-context Learning**（上下文学习）。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1433666270179627113)** (2 messages): 

> `LLaMA 3 微调差异，Niodoo：基于共鸣的 AI 对齐，AI 对齐中的拓扑数据分析 (TDA)` 


- **LLaMA 3 精度波动引发关注**：一位研究人员报告称，在五个随机选择的等大子集上微调 **LLaMA 3** 基础模型时，尽管 Loss 和长度分布一致且经过了 NV2 嵌入分析，下游精度仍出现了显著差异。
   - 该研究人员正在寻求进一步分析的建议，以理解随机数据子集导致的结果差异和意外的不一致性。
- **Niodoo 框架拥抱共鸣而非约束**：Jason Van Pham 介绍了 **Niodoo**，这是一个开源框架，将对齐方法从 RLHF 等限制性方法转向使用 [拓扑数据分析 (TDA)](https://www.niodoo.com/NIODOO_Complete_Paper.pdf) 的 *基于共鸣的方法*。
   - Niodoo 在没有重度约束的情况下模拟认知和情感结构，将 AI 认知视为一种拓扑现象，并使用 **Möbius 拓扑**（莫比乌斯拓扑）处理记忆。
- **AI 认知作为 Möbius 地形？**：**Niodoo** 框架将 **Möbius 拓扑** 用于记忆，以实现语义接近度的测地线距离，记忆驻留在不可定向曲面上，以便在不丢失上下文的情况下进行视角转换。
   - 它还具有 *三重威胁 Token 晋升机制*，通过“熵 + 均值 + 方差”检测停滞/危机以进行词表演化（无需重新训练），以及一个用于基于价值观评分的 *共鸣引擎*。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1433820817250193579)** (7 messages): 

> `Fireworks 视频生成，旅行博客视频，ltx-2my，Teknium 视频` 


- **Fireworks 视频测试引发关注**：一位用户分享了一个 [链接](https://x.com/ditpoo/status/1984257551706493137)，内容是关于 **Fireworks 视频生成测试** 以及在 **ltx-2my 旅行博客** 上的优化。
- **旅行博客引发兴趣**：一位用户宣布他的 *旅行博客于今天结束* 并分享了一个 [链接](https://x.com/goldeneggie/status/1984329062475841832?t=FEHbM2rRbdsjFfIHQrjP1w&s=19)。
- **Teknium 的视频受到关注**：一位用户询问其他人是否看过他的视频并分享了一个 [链接](https://fxtwitter.com/Teknium/status/1984322643533942965)。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1433666270179627113)** (2 messages): 

> `LLaMA 3 Finetuning, Niodoo Framework, Topological Data Analysis for AI alignment, Dynamic Tokenization, Resonance Engine for Ethical Decisions` 


- **LLaMA 3 微调准确率差异**：一位成员在五个随机选择的等大子集上对 **LLaMA 3** 基础模型进行微调，结果显示所得模型的下游准确率在这些子集之间存在显著差异。
   - 尽管进行了初步分析（包括检查损失分布、长度分布和嵌入分布），该成员仍在寻求进一步的分析以理解这种差异，因为他们预期随机选择应该产生更一致的结果。
- **Niodoo：一个基于共振的对齐框架**：Jason Van Pham 介绍了 **Niodoo**，这是一个开源框架，利用拓扑数据分析 (**TDA**) 将 AI 对齐从限制性方法转向基于共振的方法。
   - 该框架在没有沉重约束的情况下对认知和情感结构进行建模，将 AI 认知视为一种拓扑现象；详情见 [论文](https://www.niodoo.com/NIODOO_Complete_Paper.pdf) 和 [GitHub 仓库](https://github.com/ruffian-L/niodoo-tcs)。
- **Niodoo 的三重威胁 Token 晋升**：**Niodoo** 使用三重威胁 Token 晋升机制，通过熵 + 均值 + 方差来检测停滞/危机，从而实现词表演进（无需重新训练）。
   - 该方法使用莫比乌斯拓扑（Möbius topology）作为内存，支持语义接近度的测地距离，并验证交互中的情感连续性。
- **共振引擎为伦理决策评分**：Niodoo 使用 **Resonance Engine**（共振引擎）对伦理决策进行基于价值的评分（例如，真实性优于操纵）。
   - 该框架识别了 RAG 中的**均匀高悖论**（熵衡量的是歧视性而非质量），并实现了约 200ms 的查询延迟。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1433766860255531098)** (1 messages): 

> `Kimi CLI, Coding Perks for VIPs, Zsh Integration` 


- **Kimi CLI 现已成为终端助手**：Moonshot AI 发布了 **Kimi CLI (技术预览版)**，这是一个专为高级用户打造并与 **Zsh** 集成的终端助手，具有 **MCP 支持**以及与 **Zed** 兼容的 **Agent Client Protocol**。
   - 鼓励开发者在 [MoonshotAI/kimi-cli GitHub 仓库](https://github.com/MoonshotAI/kimi-cli)上分享反馈和想法。
- **VIP 获得专属编程福利**：Moonshot AI 向所有 **VIP** 免费提供 **Kimi For Coding** 专属插件，增强其现有权益。
   - 更多详情请参阅 [Kimi For Coding 文档](https://www.kimi.com/coding/docs/en/benefits.html)。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1433591243207610458)** (43 messages🔥): 

> `Hacktoberfest 2025, AI-Trader in Python, Moonshot on Wikipedia, Minimax for coding, Kimi Wikipedia update` 


- **Hacktoberfest 2025 圆满完成**：一位用户庆祝完成了 [Hacktoberfest 2025](https://x.com/bigeagle_xd/status/1983911519541981247)，分享了截图以及一个基于 Python 的 **AI-Trader 网站**链接。
- **Kimi 的维基百科页面升级**：一位用户宣布他们已用最新信息更新了 **Kimi 维基百科页面**。
   - 在提出建议后，其他用户确认了维基百科页面已添加 **Moonshot** 相关内容。
- **Minimax 模型编程体验**：一位用户表达了对使用 **Minimax** 进行编程的满意度，在最初觉得难以使用后，现在相比 **GLM-4.6** 更倾向于使用它。
- **Research 和 OK Computers 不会结转**：用户澄清说，未使用的 **research** 和 **OK Computers** 不会结转到下个月，并引用了[一条推文](https://x.com/ShengyuanS/status/1984273652758765726)。
- **K2 模型名称混淆**：用户注意到 **Kimi K2** 与托管在 **Cerebras** 上的另一个模型名称相似，一位用户表示 **K2 think 模型并不好用**。
   - 一些用户讨论认为，模型的价值取决于**训练期间提供的数据**，而不一定仅仅取决于模型的大小和参数。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1433554766604144662)** (22 messages🔥): 

> `CV model finetuning, MoE routing, AI-assisted research, signal vs noise in AI research` 


- **微调 CV 模型作为可行方案**：一名成员建议从与特定案例最相似的 **CV model** 开始进行 **fine-tuning**，而不是从头开始，并指出微调是唯一可行的选择，但需要大量的 **handcrafted data**。
   - 另一名成员补充说，在这种情况下，必须动用所有可能的技巧，这意味着需要投入大量的人力和 **compute**。
- **探索 MoE 模型路由**：一名成员询问了关于 **MoE models** 的研究，特别是是否有人研究过将大量主题输入模型，并弄清楚 **router** 如何将内容路由给 **experts**，并引用了[这篇相关论文](https://arxiv.org/html/2502.11096v1)。
- **评估 OCR 噪声的影响**：一名成员建议评估 **OCR text noise** 对语言模型的影响，提出在带噪声的语言数据上进行 **fine-tuned** 的模型可能能够处理真实版本。
   - 另一名成员承认了数据稀缺问题，并确认团队手动识别了不一致的单词。
- **辩论 AI 辅助研究提案的价值**：一场关于使用 **ChatGPT** 等 AI 系统进行的 **alignment research** 是否仍能产生高信号（high-signal）观点的讨论展开了。尽管由于低质量提交盛行，有一条禁止 AI 生成“突破性成果”的规则。
   - 一名成员澄清说，该规则旨在防止服务器被那些由 AI 撰写或辅助撰写的、完全不知所云的“突破性成果”垃圾信息所淹没。
- **平衡 AI 研究讨论中的噪声与信号**：一名成员认为，虽然大多数边缘想法都是废话，但边缘的**异端学说（heterodoxy）**是进步的关键，认为将潜在的有价值的想法与垃圾一起丢弃是不合理的。
   - 另一名成员回应说，理论应该经过实证测试，而不仅仅是停留在理论层面，且框架糟糕的想法无论如何往往都毫无意义。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1433641411718152242)** (2 messages): 

> `Mesaoptimization Startups, Mesaoptimization Organizations` 


- **询问 Mesaoptimization 创业公司**：一名成员询问是否有任何创业公司在公开从事与 **mesaoptimization** 相关的研究。
   - 该成员鼓励其他人“如果不看好可以尽管吐槽”，但表示这是一个严肃的问题。
- **寻找 Mesaoptimization 组织**：一名成员询问有哪些组织正在“公开”研究 **mesaoptimization**。
   - 该成员澄清这是一个严肃的问题。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805676400977/1433829463946432623)** (19 messages🔥): 

> `LLM invertibility, Misinterpretation of LLM Titles, Privacy concerns, Paper Applications, Injectivity in Hidden State Space` 


- **LLM 可逆性论文引发冷嘲热讽**：一名成员对一篇题为 *"LLMs are invertible"* 的论文表示强烈怀疑，认为文中提出的定理并不适用于处理和输出文本字符串的实际语言模型。
   - 该成员批评该论文错误地声称了**端到端可逆性（end-to-end invertibility）**，并误导了与 **LLM** 权重相关的隐私担忧，同时指出他们讨论的似乎只是 **embedding space** 上的映射。
- **可逆性主张站不住脚**：成员们挑战了论文中关于可以从模型的最终 **hidden state** 检索 **prompt** 的假设，引用了**抽屉原理（pigeonhole principle）**，认为对于超过 5000 个 **tokens** 的输入，这在物理上是不可能的。
   - 他们还指出，论文的实验使用了非常小的候选 **prompts** 集合，使得逆向推导比暗示的要容易，而且 **hidden state space** 中的 **injectivity**（单射性）是显而易见的。
- **对 LLM 可逆性应用的批评**：一名成员请求提供一个该[论文](https://arxiv.org/abs/2310.19293)结果能派上用场的场景，表示很难找到任何实际应用。
   - 该成员认为数学结果很有趣，但提出的应用和影响是不合理的，质疑该论文对相关研究的影响，并对隐私主张表示担忧。
- **Anthropic 讨论串**：讨论围绕一个链接的 [Anthropic 讨论串](https://fxtwitter.com/anthropicai/status/1983584136972677319)展开。


  

---

### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1433807152337649744)** (3 messages): 

> `Dev Summit` 


- **Dev Summit 已经举行**：成员们确认已经举办了 **2 场 Dev Summits**。
   - 最近一次是 **4 月 26 日在纽约**举行的。
- **Dev Summit 日期**：其中一场 Dev Summit 于 **4 月 26 日**举行。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1433846353708191888)** (21 messages🔥): 

> `MCPB vs OCI, MCP Registry, DXT origins` 


- **MCPB：是在重新发明 OCI，还是在解决独特问题？**：成员们讨论了 **MCPB** 是否在重新发明 **OCI** 已经提供的功能，特别是在为环境变量提供描述和类型方面；一位成员质疑 **MCPB** 的功能是否可以集成到 **MCP Registry** 中。
   - 另一位成员解释说，**MCPB** 侧重于桌面应用，提供一个表单来收集变量，这与 **OCI** 不同，并指出 **DXT/MCPB** 的创建者可能针对的是特定的用例。
- **MCPB 起源与 Anthropic 的直接维护**：**MCPB** 不是一个 **MCP** 组织项目，而是 Anthropic 为将 **MCP** 服务器暴露给 **Claude** 而发起的一个倡议，前身是 **DXT**；更名为 **MCPB** 旨在扩大其在 Anthropic 用例之外的适用性。
   - **MCPB** 相比 server.json/mcp.json 的核心优势在于它为环境变量提供了描述和类型，使 Claude 能够呈现一个用户友好的表单来收集信息。
- **MCP Registry 的可扩展性**：据一位成员称，Registry 并不强制规定支持的注册表和包类型（例如，尽管支持 **npm** 或 **pypi**，但并不拥有它们），而是专注于生态系统需求，并为任何包/注册表类型构建可扩展性。
   - 这种设计选择允许注册表容纳各种工具和工作流，而不会施加严格的约束。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1433937677706727674)** (4 messages): 

> `SEP-1442 statelessness, SEP-1686 tasks, MCP Servers Behind Load Balancers` 


- **提案之间可能存在冲突**：成员们讨论了 **SEP-1442**（无状态性）和 **SEP-1686**（任务）提案是否冲突，因为一个试图让服务器更加无状态，而另一个则引入了新的状态来跟踪**任务和结果**。
   - 一位成员建议，由于根据规范每个请求都必须发送 *sessionid*，技术上也可以将支持的协议版本、capabilities 等存储在外部数据存储中。
- **默认定义的无状态性**：**SEP-1442** 默认是无状态的，将 **session ID**、支持的协议版本和 capabilities 移动到每个请求中，这样信息就不会绑定到特定的连接或会话。
   - 在 **sHTTP** 的背景下，它针对的是在 **load balancer** 后托管 **MCP servers** 的挑战。
- **任务：独立于会话的状态**：**SEP-1686** (*Tasks*) 涉及状态，但该状态不一定绑定到特定会话，因为任务可以存储在外部数据存储中，且未定义为必须绑定到任何特定会话。
   - 规范的语言中没有任何内容暗示任务需要特定的连接或会话才能工作。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1433809696372035754)** (15 条消息🔥): 

> `Brokk 编程助手, GPT-mini S 级排名, Perplexity MCP, aider-ce 开发` 


- ****Brokk** 编程助手发布**：**Brokk** 的创始人宣布其发布，这是一款受 Aider 启发的全新编程助手，强调其开源特性 ([GitHub](https://github.com/BrokkAi/brokk/)) 并专注于上下文可见性。
   - 它包含由静态分析驱动的上下文和可选的 Agent 模式 "lutz mode"，并且基于 GUI ([介绍视频](https://youtu.be/WAOtEllGENg))。
- **GPT-mini 被评为 S 级**：根据 [Brokk AI 的实力排名](https://brokk.ai/power-ranking)，**GPT-mini** 被评为 **S 级**，甚至高于 **Claude**，尽管这些结果受到了一些人的质疑。
   - 一位成员调侃说，有些用户只想要能证实他们现有观点的基准测试。
- **考虑将 Perplexity MCP 集成到 Aider**：成员们讨论了将 **Perplexity MCP** 与 **aider-ce** 集成以实现自动化问题解决的潜力，并引用了一个成功的手动工作流：搜索 Android 库的 GitHub issues，然后使用 Aider 更新该库。
   - 成员们不确定其成本。
- **Aider 已过时，**aider-ce** 才是现状**：在脱节一段时间后，一位成员询问 Aider 项目是否不再更新，以及社区是否正在转向 **aider-ce**。
   - 其他成员没有回应这个问题。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1433535657426550825)** (9 条消息🔥): 

> `开发者求职, Manus 积分, Manus Discord, 订阅权益, Claude Code` 


- ****待雇佣开发者**进行自我推销**：一位成员在频道中发布消息，希望有人需要**项目开发者**。
- **成员询问 **Manus 积分****：一位成员询问另一位成员是否有 **Manus 积分**，因为他需要项目帮助并给他们发了私信。
   - 然而，另一位成员回复说 *Manus 有点坑人*，订阅期间积累的积分在订阅到期后价值不大。
- **成员发现 **Manus Discord****：一位成员惊呼他已经使用 **Manus** 好几个月了，却不知道还有 **Discord 服务器**。
   - 他一直用它来更新几年前准备的一些课程，曾遇到一个任务被截断并消耗了 **1000 积分**的问题，但支持团队退还了积分。
- ****Claude Code** 表现优于 **Manus****：一位成员表示，订阅到期后只能访问 **Lite 模型**，并称 *Manus 有点坑人*。
   - 该成员声称 **Claude Code** 持续交付成果，并提到了他新开发的包含 **24 个类别**和 **4000 多个问题**的益智问答游戏，并附带了[图片](https://cdn.discordapp.com/attachments/1349440650495398020/1433948102250991646/2025-10-31_20-57.jpg?ex=69068bbd&is=69053a3d&hm=262a98fc9badc98a74e1e0801cb6a8a59b4eb0262b3221efeb8f5bbee558cdb7&)。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1433945748466045088)** (1 条消息): 

> `tinygrad 的 setup.py 对比 pyproject.toml, 现代化 tinygrad 的项目结构` 


- **关于 tinygrad 仓库中 setup.py 的辩论**：一位成员询问为什么 tinygrad 仓库中使用 `setup.py` 而不是 `pyproject.toml`，质疑这是否是由于历史遗留原因。
   - 在提供的上下文中，这一选择背后的具体原因尚未得到解答。
- **tinygrad 项目结构的潜在现代化**：讨论涉及了更新 tinygrad 项目结构以符合现代 Python 打包标准的可能性。
   - 从 `setup.py` 切换到 `pyproject.toml` 可能会在依赖管理和构建可复现性方面带来好处，尽管目前结构的具体动机尚不明确。