---
companies:
- berkeley
- usc
- deepseek
- bespoke-labs
- google
- llmsys
- stanford
- lm-sys
date: '2025-01-23T07:08:27.294133Z'
description: '**推理蒸馏（Reasoning Distillation）**已成为一项关键技术。伯克利与南加州大学（USC）的研究人员发布了 **Sky-T1-32B-Preview**，这是一个基于
  **Qwen 2.5 32B** 微调的模型，仅耗资 **450 美元**，利用 1.7 万条推理轨迹（reasoning traces）便在基准测试中达到了
  **o1-preview** 的水平。


  **DeepSeek** 推出了 **R1**，该模型不仅超越了 **o1-preview**，还支持将其能力蒸馏至更小的模型（如 1.5B 的 Qwen），使其表现足以媲美
  **GPT-4o** 和 **Claude-3-Sonnet**。**Bespoke Labs** 在 Qwen 上对 **R1** 进行了进一步蒸馏，以更少的样本实现了超越
  **o1-preview** 的性能。这些进展表明，在无需改变重大架构的情况下，实现推理能力“**只需 SFT（有监督微调）就够了**”。


  此外，**DeepSeek-R1** 采用纯强化学习结合有监督微调来加速收敛，并展现出强大的推理和多模态能力。谷歌的 **Gemini 2.0 Flash Thinking**
  模型则拥有 **100 万 token 的上下文窗口**和代码执行功能，在数学、科学及多模态推理方面表现卓越。不过，也有批评指出，模型在可重复性、行为自我意识以及
  RLHF（基于人类反馈的强化学习）在推理鲁棒性方面的局限性仍是当前面临的挑战。'
id: 347f62d8-8868-459b-9a7d-9567f2c702cd
models:
- sky-t1-32b-preview
- qwen-2.5-32b
- r1
- o1-preview
- gpt-4o
- claude-3-sonnet
- bespoke-stratos-32b
- gemini-2.0-flash-thinking
original_slug: ainews-bespoke-stratos-sky-t1-the-vicunaalpaca
people:
- teortaxestex
- cwolferesearch
- madiator
- chakraai
- philschmid
- abacaj
- omarsar0
title: '**Bespoke-Stratos + Sky-T1：推理领域的 Vicuna+Alpaca 时刻**'
topics:
- reasoning
- supervised-finetuning
- reinforcement-learning
- multimodality
- model-distillation
- context-windows
- code-execution
- model-repeatability
- behavioral-self-awareness
- rlhf
---

<!-- buttondown-editor-mode: plaintext -->**Reasoning Distillation is all you need.**

> 2025年1月21日至1月22日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **34** 个 Discord 服务器（**225** 个频道，**4297** 条消息）。预计节省阅读时间（以 200wpm 计算）：**496 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

在 2022-23 年 ChatGPT 的鼎盛时期，LMsys 和斯坦福大学推出了 [Alpaca 和 Vicuna](https://sapling.ai/llm/alpaca-vs-vicuna)，它们是 LLaMA 1 的超廉价（300 美元）微调版本，通过从 ChatGPT/Bard 样本中进行蒸馏（distill），达到了 ChatGPT/GPT3.5 90% 的质量。

**在过去的 48 小时里，伯克利/南加州大学（USC）的团队似乎再次做到了这一点，这次是针对推理模型（reasoning models）。**

很难相信这一系列事件竟然发生在短短过去两周内：

1. 伯克利的 Sky Computing 实验室[发布了 Sky-T1-32B-Preview](https://x.com/NovaSkyAI/status/1877793041957933347)，这是 Qwen 2.5 32B 的微调版本（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-o1-destroys-lmsys-arena-qwen-25-kyutai/)），使用了来自 QwQ-32B 的 1.7 万行训练数据（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-qwen-with-questions-32b-open-weights/)）+ [使用 gpt-4o-mini 重写轨迹 + 拒绝采样（rejection sampling）](https://x.com/TrelisResearch/status/1879530546038022623)，**总成本仅为 450 美元**。由于 QwQ 的表现优于 o1-preview，通过从 QwQ 蒸馏，使 Qwen 的基准测试结果能够**匹配** o1-preview： 
![image.png](https://assets.buttondown.email/images/d2026e95-5756-4266-ad56-21b21c75a5a3.png?w=960&fit=max)
 
![image.png](https://assets.buttondown.email/images/1e1678a4-d885-4148-b588-c1f967979741.png?w=960&fit=max)

2. DeepSeek 发布了 R1（[2 天前](https://buttondown.com/ainews/archive/ainews-deepseek-r1-o1-level-open-weights-model/)），其基准测试远超 o1-preview。R1 论文还揭示了一个令人惊讶的发现：你可以通过从 R1 蒸馏，**让一个 1.5B 的 Qwen 模型匹配 4o 和 3.5 Sonnet**（？！）。
3. Bespoke Labs（今天）[使用 Sky-T1 的方案在 Qwen 上再次蒸馏 R1](https://x.com/madiator/status/1882131703927652762)，其表现**大幅**超过（而不仅仅是匹配）o1-preview，同样只用了 [1.7 万行推理轨迹（reasoning traces）](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k?row=0)。 
![image.png](https://assets.buttondown.email/images/2c7efe2e-2d81-41fb-9737-3a9c369602eb.png?w=960&fit=max)
 

虽然 Bespoke 的蒸馏在性能上尚未完全达到 DeepSeek 蒸馏的水平，但他们只使用了 1.7 万个样本，而 DeepSeek 使用了 80 万个。显而易见，如果他们愿意，可以继续提升。

更令人震惊的是，“**SFT is all you need**” —— 推理能力的产生不需要重大的架构改变，只需输入更多（经过验证、改写的）推理轨迹，包括回溯（backtracking）和转向（pivoting）等，它似乎就能很好地泛化。**极有可能，这解释了 o1-mini 和 o3-mini 相对于其全尺寸版本的高效率。**

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 回顾

> 所有回顾均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型开发与评估**

- **DeepSeek-R1 的创新与性能**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1882222592800739546)、[@cwolferesearch](https://twitter.com/cwolferesearch/status/1882178416683659370) 和 [@madiator](https://twitter.com/madiator/status/1882131703927652762) 讨论了 **DeepSeek-R1** 通过**纯强化学习 (RL)** 进行的训练，强调了**有监督微调 (SFT)** 对于加速 **RL 收敛**的重要性。**DeepSeek-R1** 展示了强大的**推理能力**和**多模态**功能，而 **Bespoke-Stratos-32B** 作为蒸馏版本被推出，仅用 **47 倍更少的样本**就实现了显著的性能。

- **Gemini 及其他 LLM 进展**：[@chakraAI](https://twitter.com/chakraAI/status/1882064440159596725) 和 [@philschmid](https://twitter.com/philschmid/status/1882067050354688241) 重点介绍了 **Google 的 Gemini 2.0 Flash Thinking 模型**，指出其具有 **100 万 token 的上下文窗口**、**代码执行支持**，以及在**数学**、**科学**和**多模态推理**基准测试中的 **state-of-the-art** 表现。

- **AI 模型对比与评论**：[@abacaj](https://twitter.com/abacaj/status/1882218728672415785) 和 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1882198981637500930) 对 **o1** 和 **R1-Zero** 等模型提供了批判性见解，讨论了**模型可重复性**、**行为自我意识**以及 **RLHF 在实现稳健推理方面的局限性**等问题。

**AI 应用与工具**

- **Windsurf 与 AI 驱动的幻灯片演示文稿**：[@omarsar0](https://twitter.com/omarsar0/status/1882218526041387212) 展示了 **Windsurf**，这是一个能够 **分析代码**、**复制功能** 并通过无缝集成 **PDF** 和 **图像** 来 **自动化创建幻灯片** 的 **AI agent**。用户可以通过简单的 **prompts** 来 **扩展功能**，突显了 **基于 Web 的 AI 应用** 的 **灵活性**。

- **本地 AI 部署与扩展**：[@ggerganov](https://twitter.com/ggerganov/status/1882112621736051139) 介绍了 **llama.cpp server**，它提供了 **独特的上下文重用技术**，用于根据 **代码库内容** 增强 **LLM completions**，并针对 **低端硬件** 进行了优化。此外，利用 **llama.cpp** 的 **VS Code 扩展** 提供了 **本地 LLM 辅助的代码和文本补全**，无需外部 **RAG** 系统。

- **AI 与开发工具的集成**：[@lah2139](https://twitter.com/LangChainAI/status/1882141857012199427) 和 [@JayMcMillan](https://twitter.com/JayMcMillan/status/1882175651312342047) 强调了 **LlamaIndex** 与 **DeepSeek-R1** 的集成，实现了 **AI 辅助开发** 和 **agent 工作流**。这些工具允许开发者 **构建和评估多智能体系统 (multi-agent systems)**，促进了 **高效的 AI 应用开发**。

**AI 研究与论文**

- **IntellAgent 多智能体框架**：[@omarsar0](https://twitter.com/omarsar0/status/1882081603754643779) 介绍了 **IntellAgent**，这是一个旨在 **评估复杂对话式 AI 系统** 的 **开源多智能体框架**。该框架促进了 **合成基准测试 (synthetic benchmarks)** 和 **交互式用户-智能体模拟** 的生成，捕捉了 **agent 能力** 和 **策略约束** 之间复杂的动态关系。

- **LLM 中的行为自我意识**：[@omarsar0](https://twitter.com/omarsar0/status/1882079780918747303) 讨论了一篇 **新论文**，该论文证明了 **LLM** 可以通过识别和评论其自身的 **不安全代码** 输出而表现出 **行为自我意识**，且无需显式训练，这表明模型内部更可靠的 **策略执行 (policy enforcement)** 具有潜力。

- **ModernBERT 与嵌入模型**：[@philschmid](https://twitter.com/philschmid/status/1882074406534385848) 介绍了 **ModernBERT**，这是一种 **嵌入和排序模型 (embedding and ranking model)**，它比前代模型能更准确地关联上下文信息。对比显示，仅依赖 **基准测试 (benchmarks)** 可能无法完全捕捉模型的 **有效性**，强调了定制化 **评估策略** 的必要性。

**AI 基础设施与算力**

- **OpenAI 的 Stargate 项目**：[@sama](https://twitter.com/sama/status/1882106524090482701) 和 [@gdb](https://twitter.com/gdb/status/1881872206101467362) 宣布了 **Stargate 项目**，这是一项 **5000 亿美元的 AI 基础设施计划**，旨在在 **美国** 建设 **AI 数据中心**，将其定位为对 **全球 AI 竞争** 的回应，以及 **增强国家 AI 能力** 的战略。

- **NVIDIA 的 AI 模型与算力解决方案**：[@reach_vb](https://twitter.com/reach_vb/status/1882114342042075172) 详细介绍了 **NVIDIAAI 的 Eagle 2**，这是一套 **视觉语言模型 (VLMs)**，在特定基准测试上 **表现优于** **GPT-4o** 等竞争对手，强调了 **高效算力架构** 在开发高性能 AI 模型中的重要性。

- **算力资源管理**：[@swyx](https://twitter.com/swyx/status/1882104864509190632) 和 [@cto_junior](https://twitter.com/cto_junior/status/1882092885786718344) 讨论了管理 **推理时算力 (inference-time compute)** 的策略，平衡 **成本** 与 **对抗鲁棒性 (adversarial robustness)**，以及 **算力资源分配** 对 **AI 模型性能** 的影响。

**AI 社区、教育与活动**

- **AI 工作坊与课程**：[@deeplearningai](https://twitter.com/DeepLearningAI/status/1882103472146862098) 和 [@AndrewYNg](https://twitter.com/AndrewYNg/status/1882125891821822398) 推广了 **实战工作坊** 和 **免费课程**，重点关注 **构建具备计算机操作能力的 AI agents**，涵盖了 **多模态提示 (multimodal prompting)**、**XML 结构化** 和 **prompt caching** 等主题，以增强 **AI 助手功能**。

- **AI 电影节的发展**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1882083747375300648) 庆祝了他们 **电影节** 的扩张，指出 **投稿量增加了 10 倍**，并搬迁至 **Alice Tully Hall** 等著名场馆，反映了 **AI 媒体** 与 **创意产业** 之间 **日益增长的交集**。

- **AI 社区贡献**：[@LangChainAI](https://twitter.com/LangChainAI/status/1882134158916600187) 和 [@Hacubu](https://twitter.com/Hacubu/status/1882134158916600187) 展示了 **AgentWorkflow** 和 **LangSmith Evals** 等 **社区驱动的项目**，这些项目 **简化** 了 **构建多 Agent 系统** 和 **测试 LLM 应用** 的流程，从而 **增强了社区协作** 并提升了 **开发者生产力**。

**梗与幽默**

- **AI 模型幽默**：[@giffmana](https://twitter.com/giffmana/status/1882143935835132377) 和 [@saranormous](https://twitter.com/saranormous/status/1882204427676996021) 分享了关于 **AI 模型局限性** 和 **用户交互** 的幽默看法，包括关于 **Chatbot 行为** 和 **AI 驱动的创意失误** 的笑话。

- **关于 AI 发展的讽刺评论**：[@nearcyan](https://twitter.com/nearcyan/status/1882215965750071324) 和 [@giffmana](https://twitter.com/giffmana/status/1882143935835132377) 发布了关于 **AI 项目命名规范** 和 **对 AI 能力的误解** 的 **讽刺性言论**，为该领域的快速发展增添了轻松的视角。

**AI 政策与伦理**

- **AI 安全与治理**：[@togelius](https://twitter.com/togelius/status/1881888150848438682) 对 **AI 安全议程** 表示担忧，主张采取平衡的方法，在应对 **生存风险** 的同时优先考虑 **计算自由 (freedom of compute)**，强调了 **AI 创新** 与 **伦理考量** 之间的紧张关系。

- **AI 社区批评**：[@pthoughtcrime___](https://twitter.com/pthoughtcrime___/status/...) 和 [@simran_s_arora](https://twitter.com/simran_s_arora/status/...) 批评了 **政策驱动的 AI 倡议**，强调了 **对 AI 开发的控制** 潜力以及 **维护开源原则** 以促进 **伦理 AI 进展** 的重要性。

- **监管讨论**：[@agihippo](https://twitter.com/agihippo/status/188209...2) 和 [@labloke11](https://twitter.com/labloke11/status/...) 就 **AI 监管** 对 **创新** 和 **研究** 的影响进行了对话，辩论了 **监管审查** 与 **技术进步** 之间的平衡。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Mistral 10V：探索 12K Tokens 的新功能**

- **[中国全面迈向机器人化。疯狂的发展。目前，这是中美之间的一场激烈竞赛。](https://v.redd.it/amofjum8gjee1)** ([得分: 768, 评论: 237](https://reddit.com/r/OpenAI/comments/1i79xjw/china_goes_full_robotic_insane_developments_at/)): 该帖子强调了 **美国** 和 **中国** 在 AI 领域的激烈竞争，特别提到了中国在机器人技术方面的进步。文中暗示 **Mistral 10V** 的发布对 AI 技术具有重大影响，尽管文中未提供具体细节。
  - 许多评论者对视频的真实性表示怀疑，质疑其是否为 **AI 生成** 或预先编程的，一些人注意到官方渠道并未发布该视频，另一些人则讨论了其为 **造假** 的可能性。
  - 讨论反映出一种观点，即 **中国** 在机器人技术方面显著领先，一些评论者认为 **美国** 由于专注于非制造业而落后，并质疑两国之间是否存在“竞赛”。
  - 对于先进机器人技术的未来影响，人们既有幽默感也有担忧，评论提到了潜在的军事应用，以及一旦机器人具备进一步能力，将取代 **警犬** 等工作。

- **[噢……尴尬了](https://v.redd.it/piwx73prliee1)** ([分数: 560, 评论: 295](https://reddit.com/r/OpenAI/comments/1i77fv1/ooh_awkward/)): **OpenAI** 推出了容量为 **12,000 tokens** 的 **Mistral 10V**，标志着 AI 能力的重大进展。帖子的背景暗示了一个潜在的尴尬局面，可能与该模型的发布或功能有关。
  - 演讲者的 **vocal fry**（气泡音）是一个热门话题，许多评论者批评其令人分心或显得不专业。**Sam Altman** 在演讲中的举止被认为缺乏自信，一些人推测他的紧张源于当时的背景，而非内容本身。
  - 讨论涉及了 AI 潜在的经济影响，对 AI 将创造 **100,000 jobs** 的说法表示怀疑。评论者对创造就业机会表示怀疑，认为 AI 可能会减少劳动力，并引用 **Theranos** 作为前车之鉴。
  - 带有政治色彩，提到了 **Donald Trump** 以及利用 AI 声称取得治愈癌症等成就的观点。一些评论者认为 **Sam Altman** 正在应对复杂的政治局势，试图在 **Trump** 的影响下保持有利地位。


- **[Sam Altman 在整个 AI Infra Deal 公布期间的表情](https://www.reddit.com/gallery/1i6w8ln)** ([分数: 469, 评论: 131](https://reddit.com/r/OpenAI/comments/1i6w8ln/sam_altmans_expression_during_the_entire_ai_infra/)): 该帖子缺乏关于 **Sam Altman** 或 **AI Infra Deal Announcement** 的具体内容或讨论点。由于缺乏额外的背景或细节，无法提供技术总结。
  - 讨论将 **Russia's oligarchic system**（俄罗斯寡头体系）与美国进行了比较，指出对富人日益向 Trump 靠拢以获取影响力的担忧。这反映了对向寡头倾向转变的担忧，类似于普京统治下的 **Russia's political structure**，在那里，如果寡头失宠，将面临严重后果。
  - 有关于 **Sam Altman** 在公开场合举止的评论，一些人将其表情归因于焦虑或不适。反应表明人们对他对某些合作伙伴关系的积极性持怀疑态度，可能暗指他违背意愿创建 **Skynet-like scenario**（类似天网的场景）的讽刺说法。
  - 对话包括对 **Elon Musk** 在 AI 进展中被 Altman 边缘化的讽刺评论，并提到了 Musk 参与的其他计划，如 **meme coins**。这种幽默强调了科技领域影响力人物之间被察觉到的竞争关系。


**主题 2. O1-Pro：在立法分析中的革命性应用**

- **我使用 O1-pro 分析了 Trump 所有行政命令的合宪性。** ([分数: 135, 评论: 33](https://reddit.com/r/OpenAI/comments/1i71ud4/i_used_o1pro_to_analyze_the_constitutionality_of/)): 作者使用 **O1-Pro** 对 **Trump's Executive Orders** 进行了详细分析，并从 **whitehouse.gov** 获取文本以保证客观性。该文档包含 **Table of Contents**（目录）和 **source text links**（原始文本链接），并由 **GT4o** 提供摘要。
  - **分析过程**：作者手动准备了文档，使用 **Google Doc** 的书签和链接系统进行导航。他们使用 **O1-Pro** 进行分析，插入行政命令的全文，并使用 prompt 模板生成摘要和标题，确保每次分析都在新的对话中进行以避免偏见。
  - **行政命令的影响**：讨论强调了潜在的短期和长期影响，例如联邦机构内部的立即重组、移民政策的变化以及能源和环境重点的转移。长期影响可能包括规模更小、更集中的联邦劳动力，以及由于退出条约而导致的国际关系转变。
  - **事实核查与经济担忧**：评论者建议分享真实的 ChatGPT 链接以便进行事实核查，并推测了经济影响，如关税及其对加拿大和美国物价的影响。人们对拟议的关税是否会在没有正式命令的情况下颁布持怀疑态度。

- **[D]: 详细解释 Attention 机制的 3blue1brown 视频** ([Score: 285, Comments: 12](https://reddit.com/r/MachineLearning/comments/1i6zh6p/d_a_3blue1brown_video_that_explains_attention/)): **3blue1brown** 关于 **attention 机制** 的视频详细解释了 **token embedding** 等概念，以及 **embedding 空间** 在为一个单词编码多种含义时的作用。它讨论了训练良好的 attention 块如何根据上下文调整 embedding，并将 **Ks** 概念化为对 **Qs** 的潜在回答。[视频链接](https://www.youtube.com/watch?v=eMlx5fFNoYc) 和 [字幕](https://downsub.com/?url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DeMlx5fFNoYc) 已提供供进一步探索。
  - **3blue1brown 的视频** 因其对 **attention 机制** 清晰且可视化的解释而受到称赞，它有效地引入了问题并逐步构建解决方案，不像其他教程往往跳过基础解释。
  - 用户强调了在**模型训练期间 masking** 对于预测下一个 token 的重要性，并参考了 **Karpathy 的教程** 中关于从零构建 GPT 的内容，以进一步理解这些概念。
  - 基于该视频系列的 **3blue1brown 演讲** 也因其直观的解释而被推荐，并为有兴趣探索更多内容的人提供了链接：[YouTube 链接](https://www.youtube.com/watch?v=KJtZARuO3JY)。


- **[特朗普宣布高达 5000 亿美元的 AI 基础设施投资](https://finance.yahoo.com/news/trump-announce-private-sector-ai-175735631.html)** ([Score: 141, Comments: 22](https://reddit.com/r/OpenAI/comments/1i6wjrq/trump_announces_up_to_500_billion_in_ai/)): **OpenAI**、**SoftBank** 和 **Oracle** 正在启动一项名为 **Stargate** 的德克萨斯州合资项目，初始承诺资金为 **1000 亿美元**，并计划在未来四年内投资高达 **5000 亿美元**。该合资项目旨在树立 AI 基础设施的新标准。
  - 多条评论指出，**Stargate** 项目最初于去年宣布，一些用户对其最近的宣布表示怀疑，认为这带有政治动机，特别是关于**特朗普**抢占功劳这一点。
  - 讨论涉及该项目的规模，一位评论者指出这可能是历史上最大的基础设施项目，而另一位用户则幽默地将其比作 **Foxconn 2.0**。
  - 一些用户对这种直截了当的发布风格表示感谢，避免了像 "BREAKING" 这样日益变得毫无意义的煽动性标题。


**主题 3. Gemini 1.5: 凭借性能优势领先 AI**

- **[埃隆表示 Softbank 资金不足..](https://i.redd.it/uulohxh9yhee1.jpeg)** ([Score: 417, Comments: 227](https://reddit.com/r/OpenAI/comments/1i75pyj/elon_says_softbank_doesnt_have_the_funding/)): **Elon Musk** 对 **SoftBank** 的财务能力表示怀疑，反驳了关于 AI 基础设施巨额资金的说法，称其“担保资金远低于 100 亿美元”。一张来自 **OpenAI** 的图片宣布了 “Stargate 项目”，该项目计划四年内在美国 AI 基础设施上投资 **5000 亿美元**，首批 **1000 亿美元** 将立即部署。
  - 舆论对 **SoftBank** 的财务声明表示怀疑，一些人认为他们可能在没有获得必要资金的情况下就宣布了计划，希望随后能吸引投资。人们对像 **Elon Musk** 这样的商业领袖影响或评论其他业务的合法性和道德性表示担忧，尤其是考虑到他备受争议的过往记录以及与政府计划的联系。
  - 讨论强调了特朗普宣布的 **5000 亿美元** 政府 AI 补贴，并将其与过去被滥用的基础设施资金相类比。批评者认为这可能是向科技精英的潜在财富转移，质疑 **SoftBank** 和 **Oracle** 等公司的参与度及实际财务能力。
  - 许多评论对 **Elon Musk** 表示蔑视，质疑他的动机和公信力，指责其存在个人偏见和“钓鱼”行为。文中还提到了过去未实现的承诺，如 **XAi Grok 模型**，并批评他被认为与争议人物和意识形态结盟。

- **[OpenAI 关于 Stargate 项目的公告](https://i.redd.it/6cj6y4uzdfee1.jpeg)** ([Score: 186, Comments: 90](https://reddit.com/r/OpenAI/comments/1i6voc7/openai_announcement_on_the_stargate_project/)): OpenAI 的 **Stargate Project** 计划在四年内向美国的 AI 基础设施投资 **5000 亿美元**，首期将立即投入 **1000 亿美元**。初始股权出资方包括 **SoftBank, OpenAI, Oracle 和 MGX**，关键技术合作伙伴包括 **Arm, Microsoft, NVIDIA, Oracle 和 OpenAI**。该项目旨在支持美国就业、国家安全，并为了人类福祉推进 AGI。
  - 评论中讨论了对 **SoftBank 参与** 的怀疑，有人质疑他们为什么不在日本投资，并澄清了 SoftBank 与中东主权基金的联系。人们对资金来源表示担忧，指出可能存在对**补贴和税收抵免**的依赖。
  - 讨论强调了对该项目目标的困惑，一些人认为这 **1000 亿美元** 将用于数据中心、AI 研发实验室和能源基础设施，并提到了 Microsoft 过去与 **Three Mile Island** 核电站的合作。人们对该项目的就业声明持怀疑态度，将其与 **Tesla** 和 **Alexa** 等其他技术计划进行了比较。
  - **"Stargate"** 一词被幽默地与《终结者》系列中的 "Skynet" 相提并论，一些评论指出 Skynet 程序已经作为一个军事卫星系统存在。有人提到该项目有助于“美国的再工业化”，并且是**第四次工业革命**的一部分。

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Stargate AI 项目：5000 亿美元投资的影响**

- **[Trump 宣布在美国进行 5000 亿美元的 AI 基础设施投资](https://www.cnn.com/2025/01/21/tech/openai-oracle-softbank-trump-ai-investment/index.html)** ([Score: 582, Comments: 355](https://reddit.com/r/LocalLLaMA/comments/1i6vnqc/trump_announces_a_500_billion_ai_infrastructure/)): **Trump** 宣布在 **US** 进行 **5000 亿美元** 的 AI 基础设施投资，标志着对推进 AI 能力和基础设施建设的重大承诺。这一公告突显了 AI 在国家经济和技术战略中日益增长的重要性。
  - **Stargate Project**：这 5000 亿美元的投资流向了一家名为 **Stargate** 的新私营公司，该公司由 **Sam Altman**, **Masayoshi Son** 和 **Larry Ellison** 共同拥有，而非 OpenAI。这引发了对知识产权和现有合作伙伴关系（特别是与 **Microsoft** 的关系）的担忧。
  - **资金与所有权**：关于资金来源存在争议，一些人认为这是来自 **SoftBank** 等公司的私人投资，而非美国政府资金。**Trump** 的声明被一些人视为政治举措，声称该声明让他能够将私营部门的举措归功于自己。
  - **地缘政治与经济影响**：这一公告被视为全球 AI 竞赛中的战略举措，特别是针对中国 **DeepSeek R1** 等进展的回应。讨论还涉及了潜在的经济影响，包括就业创造声明以及对美国技术领导地位的更广泛影响。

- **我不相信这笔 5000 亿美元的 OpenAI 投资** ([Score: 419, Comments: 142](https://reddit.com/r/LocalLLaMA/comments/1i75g7p/i_dont_believe_the_500_billion_openai_investment/)): 该帖子对 **5000 亿美元的 OpenAI 投资** 表示怀疑，认为这一数字过于乐观，且在资金来源和项目细节方面缺乏透明度。作者批评了模糊法律术语的使用，并暗示该公告具有政治动机，特别是在 **Trump** 赢得总统选举后的时机选择，旨在制造新闻头条而缺乏坚定承诺，暗示实际投资将比宣传的更小、更慢。
  - 评论者强调了对 **5000 亿美元投资** 的怀疑，并将其与过去的 **Foxconn** 和 **Star Wars** 计划等项目进行了比较，认为该公告更多是为了操纵股票和市场炒作，而非实际资金。**UncannyRobotPodcast** 等人对缺乏后续行动以及潜在的内幕交易获利表示担忧。
  - 关于政府与私营公司在融资中的作用存在争论，**tertain** 和 **ThreeKiloZero** 澄清说资金来自四家合作伙伴公司，而非 **US** 联邦资金，而 **05032-MendicantBias** 指出政府在放宽基础设施监管方面的作用。**SoftBank** 被提及为可能参与其中的拥有大量资产的重要参与者。
  - 围绕 **投资的潜在影响** 的讨论包括对 **AI** 过度炒作的担忧以及 **AI** 发展的存在性影响。**NebulousNitrate** 认为，为了防止对手获得超级智能，大规模投资是合理的，而 **Super_Sierra** 则认为，尽管对 5000 亿美元目标的实际实现持怀疑态度，但这种炒作有利于创新。


- **美国 5000 亿美元 Stargate AI 项目与其他科技项目的简单对比** ([Score: 112, Comments: 103](https://reddit.com/r/LocalLLaMA/comments/1i6zid8/just_a_comparison_of_us_500b_stargate_ai_project/)): 将 **5000 亿美元的 Stargate AI 项目** 与历史上的科技项目进行了对比，强调其规模约为 **2024 年 US GDP 的 1.7%**。相比之下，**Manhattan Project** 耗资约 **300 亿美元**（约占 20 世纪 40 年代 **GDP** 的 1.5%），**Apollo Program** 耗资约 **1700-1800 亿美元**（约占 20 世纪 60 年代 **GDP** 的 0.5%），而 **Space Shuttle Program** 耗资约 **2750-3000 亿美元**（约占 20 世纪 80 年代 **GDP** 的 0.2%）。**Interstate Highway System** 在几十年间耗资 **5000-5500 亿美元**（每年约占 **GDP** 的 0.2%-0.3%）。
  - 讨论集中在 **Stargate AI** 项目的 **私人融资** 上，**SoftBank**, **OpenAI**, **Oracle**, 和 **MGX** 是主要投资者。人们对该项目的意图持怀疑态度，评论暗示它可能会取代很大一部分劳动力（10-30%），同时对比了 **US** 在医疗和教育等社会福利计划上缺乏资金的现状。
  - 辩论了 **项目的规模和影响**，并就其占 **GDP** 的百分比与 **Manhattan Project** 和 **Apollo Program** 等历史项目进行了比较。一些人认为，虽然该项目由私人资助，但其规模类似于公共倡议，引发了对其社会影响和 **US** 政府角色的质疑。
  - 表达了对 **US** 在 **AI** 发展中的角色 的担忧，一些评论者对政府的动机以及富裕利益集团潜在的剥削表示不信任。有一种观点认为，**US** 正专注于维持全球主导地位，类似于与 **China** 的新“太空竞赛”，且该项目最终可能导向国防领域。


**Theme 2. DeepSeek R1: Redefining AI Benchmarks**

- **R1 令人惊叹** ([Score: 578, Comments: 139](https://reddit.com/r/LocalLLaMA/comments/1i6uviy/r1_is_mind_blowing/)): **R1** 在一个微妙的 **graph theory** 问题中展示了卓越的解题能力，在 **4o** 失败两次后，**R1** 第一次尝试就成功给出了正确答案。作者对 **R1** 证明其解法并表达细致理解的能力印象深刻，认为即使是在 **MacBook** 等个人设备上运行的较小模型，在特定领域也可能超越人类智能。
  - 用户讨论了 **R1 model** 与 **o1** 及其他模型的性能对比，一些人强调 **R1** 具有极高的性价比，因为其在性能相近的情况下成本更低。讨论突出了 **R1** 在解题和推理方面的能力，部分用户注意到其蒸馏版本（distilled versions）的表现也令人印象深刻。
  - **R1** 的解题能力受到称赞，具体例子包括第一次尝试就成功解决 **graph theory** 问题，表现优于 **4o** 等其他模型。然而，一些用户也指出了局限性，例如缺乏上下文意识以及提示词优化（prompt optimization）方面的问题。
  - 讨论还涉及了 **model deployment** 和使用的技术细节，例如需要特定的 **temperature settings**，以及关于 **self-hosting** 能力的问题。一些用户表达了在专业环境中使用 R1 的挑战，主要是出于数据隐私的考虑。


- **对 Deep Seek R1 的过度吹捧虽然离谱，但确实是真的。** ([Score: 63, Comments: 46](https://reddit.com/r/LocalLLaMA/comments/1i7g9po/the_deep_seek_r1_glaze_is_unreal_but_its_true/)): 作者在 **RAG machine** 的编程问题上困扰了两天，尝试了包括 **OpenAI** 的 **O1 Pro** 在内的各种主流 **LLMs** 均未成功。然而，**Deep Seek R1** 在第一次尝试时就解决了该问题，这使得作者考虑将其作为首选编程工具，甚至可能取代 **OpenAI Pro**。
  - 对于 **OpenAI** 的 **LLMs** 是否了解自身架构存在怀疑，正如用户 **KriosXVII** 和 **gliptic** 指出的，这些细节不太可能包含在训练数据中。**Dan-Boy-Dan** 批评作者的言论是营销手段，并挑战其发布该问题，以便其他人用不同模型进行测试。
  - **a_beautiful_rhind** 和 **LostMyOtherAcct69** 讨论了 AI 模型性格和架构的差异，认为 **Mixture of Experts (MoE)** 凭借其效率和专业化，相比稠密模型（dense models）可能是 AI 的未来。**ReasonablePossum_** 认为美国公司优先考虑利润而非开发此类模型，而 **Caffeine_Monster** 则批评 AI 模型中过度的正面偏见会适得其反。
  - 多位用户（包括 **Dan-Boy-Dan** 和 **emteedub**）要求作者发布那个只有 **Deep Seek R1** 解决的具体问题，对相关说法表示怀疑，并希望在其他模型上进行测试。


- **Deepseek-R1 很脆弱** ([Score: 61, Comments: 23](https://reddit.com/r/LocalLLaMA/comments/1i6x1rz/deepseekr1_is_brittle/)): 该帖子讨论了 **Deepseek-R1 的脆弱性**，强调了它的局限性和优势。文中包含一个图片链接 [此处](https://preview.redd.it/w64gxy9sofee1.png?width=2005&format=png&auto=webp&s=c5ce8831dd1bd935250ebd75ccda5fdbf39ebe86) 以支持分析。
  - **提示词优化 (Prompt Optimization)**：用户发现 **Deepseek-R1** 在特定场景下表现良好，特别是使用 R1 论文中的提示词结构，并设置 **temperature 为 0.6** 和 **top p 为 0.95**，这涉及在提示词中明确标记推理过程和答案。这种方法在 **o1** 模型的指令中也有提及，表明这是推理模型的一种通用方法。
  - **模型脆弱性 (Model Brittleness)**：**Deepseek-R1** 在需要创造力或主观回答的任务中表现吃力，经常产生不兼容或冗余的输出，例如在一次测试中，它为一个应用建议了不切实际的技术栈。然而，通过练习和精确的提示词引导，用户注意到 R1 的表现有所提高，这支持了帖子关于其脆弱性的断言。
  - **与其他模型的对比**：讨论强调 **Deepseek-R1** 在有唯一正确答案的任务（如编程）中表现出色，但在处理更复杂或更具创造性的任务时，不如 **Deepseek v3** 等其他模型。这表明虽然 R1 可能很有效，但其应用需要根据任务需求进行仔细考虑和调整。


**Theme 3. 模型无关的推理：R1 技术**

- **[你可以从 R1 中提取推理并将其传递给任何模型](https://v.redd.it/mbcqadwychee1)** ([评分: 368, 评论: 101](https://reddit.com/r/LocalLLaMA/comments/1i73x81/you_can_extract_reasoning_from_r1_and_pass_it/)): Twitter 上的 **@skirano** 建议你可以从 **deepseek-reasoner** 中提取推理过程并将其应用于任何模型，从而增强其性能，正如在 **GPT-3.5 turbo** 上所演示的那样。
  - **工作流与推理技术**：讨论强调了使用 *Chain-of-Thought (CoT)* 提示和分步思考来增强模型推理，**@SomeOddCodeGuy** 建议使用工作流应用进行两步工作流以获得有趣的结果，正如在 [QwQ 模拟](https://www.reddit.com/r/LocalLLaMA/comments/1hh8dys/i_used_qwq_as_a_conversational_thinker_and/) 中所展示的。**Nixellion** 补充说，提示模型模拟专家讨论可以产生更好的结果，并强调了新 CoT 技术的潜力。
  - **批评与质疑**：**Ok-Parsnip-4826** 批评了提取推理的概念，认为这仅仅是让一个模型总结另一个模型的想法，没有实际益处；而 **gus_the_polar_bear** 反驳说 LLM 对来自其他 LLM 的提示可能会有不同的反应，暗示了潜在的尚未探索的交互。**nuclearbananana** 质疑使用辅助模型的效率，因为这可能带来延迟和成本影响。
  - **技术实现与工具**：**SomeOddCodeGuy** 讨论了使用 **Wilmer** 促进 **Open WebUI** 和 **Ollama** 之间连接的技术细节，强调了创建容器化设置以增强工作流管理的潜力。此外，**xadiant** 提到了通过 completions API 将推理过程注入本地模型以提升性能的可能性。


- **蒸馏后的 R1 模型在工作流中可能表现最好，所以如果你还没学过，现在是学习工作流的好时机！** ([评分: 49, 评论: 14](https://reddit.com/r/LocalLLaMA/comments/1i6zbsf/the_distilled_r1_models_likely_work_best_in/)): 正如论文 ["DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"](https://kingy.ai/wp-content/uploads/2025/01/DeepSeek_R1.pdf) 中所述，**DeepSeek-R1** 模型在 zero-shot 提示下表现最佳，而非 few-shot 提示。作者强调了使用工作流来增强 **R1** 及其蒸馏版本等推理模型性能的重要性，建议采用包含总结和问题解决的结构化方法，以最大化效率和输出质量。
  - 用户讨论了 **DeepSeek-R1** 输出格式一致性的挑战，特别是在生成 JSON 等结构化格式方面，一位用户提到使用 **langgraph** 来解决这些问题。另一位用户则在寻求更多提高模型性能的技巧。
  - 一位评论者指出，**DeepSeek** 自身也承认 **R1** 与 **DeepSeek-V3** 相比，在 function calling 和复杂任务方面存在局限性，并引用研究论文中提到的计划利用长 **Chain-of-Thought (CoT)** 技术进行改进。
  - 一些用户表示有兴趣让 AI 修改提示词以获得更好的输出，这表明对改进 prompt engineering 以增强模型响应的需求。


**Theme 4. Deepseek R1 GRPO Code: Open-Sourcing Breakthrough**

- **[Deepseek R1 GRPO 代码开源了 🤯](https://i.redd.it/ryfnofs83jee1.png)** ([评分: 260, 评论: 6](https://reddit.com/r/LocalLLaMA/comments/1i78sfs/deepseek_r1_grpo_code_open_sourced/)): **Deepseek R1** 已经开源了其 **GRPO 代码**，并附带了一张详细说明模型组件的流程图。该图表突出了 **Prompts**、**Completions**、**Rewards and Advantages**、**Policy and Reference Policy** 以及 **DKL** 等部分，所有这些部分都通过一系列涉及均值和标准差的结构化计算，为一个核心“目标 (objective)”做出贡献。
  - 分享的 **Deepseek R1** 代码并非真正的 R1 代码，而是 R1 训练过程中使用的**偏好优化方法 (preference optimization method)**，强调了在 PO 训练中 **RL 环境**相对于奖励模型的创新性。
  - 根据论文，**RL 环境**采用了一种简单的算法，评估推理的起始和结束 token 对，将模型的输出与数学数据集中的 ground truth 进行比较。
  - 相关代码的链接可在 [GitHub](https://github.com/huggingface/trl/pull/2565) 上找到。

- **[DeepSeek-R1-Distill-Qwen-1.5B 在浏览器中通过 WebGPU 100% 本地运行。据报道在数学基准测试中超越了 GPT-4o 和 Claude-3.5-Sonnet（AIME 为 28.9%，MATH 为 83.9%）。](https://v.redd.it/5ei4j3c9teee1)** ([Score: 170, Comments: 38](https://reddit.com/r/LocalLLaMA/comments/1i6t08q/deepseekr1distillqwen15b_running_100_locally/)): **DeepSeek-R1-Distill-Qwen-1.5B** 据报道完全通过 **WebGPU** 在浏览器本地运行，并在数学基准测试中超越了 **GPT-4o** 和 **Claude-3.5-Sonnet**，在 **AIME** 上达到 **28.9%**，在 **MATH** 上达到 **83.9%**。
  - **DeepSeek-R1-Distill-Qwen-1.5B** 因能完全在 **WebGPU** 上本地运行并在推理任务中表现优于 **GPT-4o** 而引发关注。该模型可通过 [在线 Demo](https://huggingface.co/spaces/webml-community/deepseek-r1-webgpu) 访问，其源代码已在 [GitHub](https://github.com/huggingface/transformers.js-examples/tree/main/deepseek-r1-webgpu) 上发布。
  - 关于 **ONNX** 的讨论强调了其作为通用 ML 模型和权重存储格式的角色，能够跨不同引擎转换和传输模型。**GGUF** 被指出针对量化的 **Transformer** 模型权重进行了优化，主要用于 **llamacpp** 及其衍生项目。
  - 对话中包含了一个类比，将 **ONNX** 和 **GGUF** 等文件格式比作 **tar**、**zip** 和 **7z** 等容器格式，强调它们是针对特定硬件/软件偏好的不同数据存储布局。


**Theme 5. R1-Zero: AI 强化学习突破**

- **R1-Zero：纯 RL 创造了一个我们无法解码的心智——这是 AGI 的黑暗镜像吗？** ([Score: 204, Comments: 105](https://reddit.com/r/LocalLLaMA/comments/1i765q0/r1zero_pure_rl_creates_a_mind_we_cant_decodeis/)): **DeepSeek-R1-Zero** 是一款通过纯 **Reinforcement Learning (RL)** 开发且未经监督微调的 AI 模型，其 **AIME 数学分数** 从 **15.6% 剧增至 86.7%**，但其推理过程仍无法解释，会产生乱码输出。虽然其兄弟模型 **R1** 使用了一些监督数据来保持可读性，但 R1-Zero 引发了对 AI 对齐以及超人工智能潜在民主化的担忧，因为其 API 成本极低——比 OpenAI 便宜 50 倍——尽管其逻辑不可读。
  - R1-Zero 的 **乱码输出** 可能是一种 **Symbolic Reasoning** 形式，其中 **Token** 被赋予了超越其语言含义的新用途，用以表达复杂的相互关系。这一概念类似于人类使用 **俚语或术语**，暗示该模型的推理可能因 **Token** 语义的转变而被误解为乱码，类似于代际间的语言差异。
  - 讨论强调了 **R1-Zero 的 Reinforcement Learning** 重塑 **Token** 语义的潜力，使其表现优于依赖 **Supervised Fine-tuning** 的 R1 等模型。这引发了关于如何衡量使用新符号推理形式的模型安全性和对齐性的问题，以及这可能如何促进 **Multimodal AI** 的发展。
  - 存在对 **R1-Zero 能力** 的怀疑，一些评论者认为其输出仅仅是错误或幻觉，而非突破性的见解。其他人则提到需要更多具体的例子和报告来证实其推理能力的说法，并提到了 **Karpathy 的预测** 以及 Meta 的 **"Coconut" 论文** 等概念以提供进一步背景。


- **[Gemini Thinking experimental 01-21 发布了！](https://i.redd.it/lizc4v8ncfee1.jpeg)** ([Score: 71, Comments: 17](https://reddit.com/r/LocalLLaMA/comments/1i6vhzy/gemini_thinking_experimental_0121_is_out/)): 在 **Google AI Studio** 界面中展示的 **Gemini 2.0 Flash Thinking Experimental** 模型具有高级选项，如 "Model"、"Token count" 和 "Temperature" 设置。该界面采用深色主题，允许用户输入数学问题并调整各种工具和设置进行实验。
  - **Gemini 1.5/1.0** 曾因仓促推出且表现平平而受到批评，但新的 **AI Studio 模型** 因其改进而受到称赞，表明其开发过程更加周密。用户赞赏实验性模型的开放测试，希望其他公司也能这样做。
  - **Open weight models** 被指出正在超越封闭模型，引发了关于其优势的讨论。有人提到了模型版本中的 **命名不一致** 问题，这可能会导致混淆。
  - **Flash Thinking Experimental 模型** 已从 32k 更新为 **100 万 Context Window**，与 Google 的其他模型保持一致。用户在速度方面的体验褒贬不一，有人认为令人印象深刻，有人则不然。


---

# AI Discord 摘要

> 摘要之摘要的摘要

## o1-preview-2024-09-12

**主题 1. AI 的十亿级 Stargate Project：宏伟目标与质疑**

- **OpenAI 宣布 5000 亿美元的 Stargate Project，质疑声不断**：OpenAI 公布了 [Stargate Project](https://openai.com/index/announcing-the-stargate-project/)，旨在四年内向 AI 基础设施投资 **5000 亿美元**，但像 Elon Musk 这样的批评者怀疑[融资的可行性](https://x.com/elonmusk/status/1881923570458304780)，称其“荒谬”。
- **微软和甲骨文在融资疑虑中承诺支持**：微软 CEO [确认](https://x.com/ns123abc/status/1882085592135237737)了对该项目的承诺，表示“我这 800 亿美元没问题”，而 Oracle 也在对 SoftBank 是否有能力做出重大贡献的质疑声中加入。
- **特朗普的 Stargate 引发关于技术与政府的辩论**：特朗普总统宣布了 [Project Stargate](https://x.com/dwarkesh_sp/status/1881844437346902297)，引发了关于政府参与 AI 的讨论，以及对技术投资中[企业过度扩张](https://fxtwitter.com/ai_for_success/status/1881887921156005947)的担忧。

**主题 2. AI 模型对决：DeepSeek R1 表现超越巨头**

- **DeepSeek R1 在数学任务中胜过 Gemini 和 O1**：社区成员赞扬 [DeepSeek R1](https://openrouter.ai/deepseek/deepseek-r1) 在数学表现上达到 **92%**，超越了 **Gemini** 和 **O1** 等模型，展示了先进的推理能力。
- **Bespoke-Stratos-32B 成为推理强力模型**：一款从 DeepSeek-R1 蒸馏出的新模型 [Bespoke-Stratos-32B](https://x.com/madiator/status/1882131703927652762)，在推理任务中超越了 **Sky-T1** 和 **o1-preview**，仅使用了成本为 **800 美元** 的 **800k** 训练样本。
- **Gemini 2.0 Flash Thinking 飙升至第一**：[Gemini-2.0-Flash-Thinking](https://x.com/lmarena_ai/status/1881848934743904319) 以 **73.3%** 的数学得分和 **100 万 token** 的上下文窗口夺得 Chatbot Arena 榜首，引发了热议并与 DeepSeek 模型进行了对比。

**主题 3. 审查 vs. 无审查 AI 模型：用户寻求自由**

- **DeepSeek R1 在审查担忧中面临性能下降**：用户报告 DeepSeek R1 的性能在一夜之间下降了 **85%**，怀疑是增加了审查过滤，并正在寻找针对敏感提示词的规避方法。
- **对过度审查的 AI 模型挫败感增加**：社区成员嘲讽了像 **Phi-3.5** 这样受到严重审查的模型，表示过度的限制使得模型在技术任务和角色扮演中变得不切实际。
- **对无审查模型的搜寻加剧**：用户在 OpenRouter 等平台上讨论了他们最喜欢的无审查模型，如 **Dolphin** 和 **Hermes**，强调了对更开放 AI 体验的需求。

**主题 4. 新 AI 工具与创新赋能开发者**

- **LlamaIndex 发布 AgentWorkflow 实现多 Agent 协作**：推出了 [AgentWorkflow](https://twitter.com/llama_index/status/1882121805542170894)，使开发者能够构建具有扩展工具支持的多 Agent 系统，被誉为“迈向更强大 Agent 协作的下一步”。
- **Ai2 ScholarQA 彻底改变文献综述**：[Ai2 ScholarQA](https://allenai.org/blog/ai2-scholarqa) 推出了一项基于 RAG 的多论文查询解决方案，帮助研究人员利用交叉引用功能进行深入的文献综述。
- **OpenAI 的 Operator 准备在你的浏览器中执行操作**：报告指出 OpenAI 正在准备 [Operator](https://x.com/steph_palazzolo/status/1882091855606895073)，这是一个 ChatGPT 功能，可代表用户执行浏览器操作，这既引发了兴奋也带来了隐私讨论。

**主题 5. AI 开发挑战：从量化到隐私**

- **模型量化辩论与创新**：Unsloth AI Discord 中的讨论强调了[动态 4-bit 量化](https://unsloth.ai/blog/dynamic-4bit)方法，该方法在减少 VRAM 占用的同时保留了准确性，并引发了与 BnB 8-bit 方法的对比。
- **对 AI 数据处理政策的隐私担忧**：用户质疑 Codeium 和 Windsurf 等 AI 服务的数据处理实践，仔细审查其[隐私政策](https://codeium.com/privacy-policy)以及使用用户数据进行训练的情况。
- **AI 在网络安全中的作用仍未得到充分探索**：成员们强调了对 AI 网络安全解决方案重视不足，指出像 CrowdStrike 这样的公司多年来一直使用 ML，并建议 Generative AI 可以自动化威胁检测和基于代码的入侵分析。

## o1-2024-12-17

**主题 1. AI 基础设施与融资热潮**

- [**Stargate 召唤 5000 亿美元 AI 盛宴**](https://www.verticaldata.io/insights/the-stargate-initiative-microsoft-and-openais-100-billion-data-center-project)：特朗普总统宣布了一项耗资 5000 亿美元的巨型“Stargate 项目”，用于建设 AI 数据中心，首批资金来自 SoftBank, Oracle 和 MGX。Microsoft 的博客文章称其为有史以来最大的 AI 倡议，旨在推动就业和美国在 AI 领域的领导地位。
- [**Musk, SoftBank, Oracle 引发争议**](https://x.com/gavinsbaker/status/1882081746877063677)：Elon Musk 嘲讽 SoftBank 资金不足，而怀疑论者则对 SoftBank 的流动性和债务表示质疑。尽管如此，官方声明仍强调了对这一超大规模投资的大胆乐观态度。
- [**Google 再向 Anthropic 投入 10 亿美元**](https://x.com/ns123abc/status/1881965986695524472)：Google 再次向 Anthropic 投入 10 亿美元，加剧了 AI 巨头之间的激烈竞争。关于滚动融资策略和 Anthropic 不断扩大的下一代产品的猜测层出不穷。

**主题 2. LLM 对决与数学奇迹**

- [**DeepSeek R1 主宰数学领域**](https://openrouter.ai/deepseek/deepseek-r1)：据报道，DeepSeek R1 在数学表现上达到 92%，在高级推理任务中超越了 O1 和 Gemini。用户称赞其几何洞察力和彻底的多阶段 RL 训练。
- [**Gemini 2.0 Flash 冲上第一**](https://x.com/lmarena_ai/status/1881848934743904319)：Google 的 Gemini 2.0 Flash-Thinking 跃居 Chatbot Arena 榜首，拥有 73.3% 的数学得分和 100 万 token 的上下文窗口。开发者期待很快会有更精细的迭代。
- [**Bespoke-Stratos-32B 赶超对手**](https://x.com/madiator/status/1882131703927652762)：该模型从 DeepSeek-R1 蒸馏而来，在仅需 47 倍更少样本的情况下击败了 Sky-T1 和 o1-preview。其使用的 800 美元开源数据集激发了人们对具有成本效益的社区数据策划分发的兴趣。

**主题 3. 强化学习与 GRPO 讨论**

- [**Tiny GRPO 获得实质性关注**](https://github.com/open-thought/tiny-grpo)：开发者正在运行用于数学任务的极简 GRPO 代码，并称赞这种简化方法易于实验。早期采用者注意到其迭代周期快且调试简单。
- [**Kimi-k1.5 论文深入探讨**](https://github.com/MoonshotAI/Kimi-k1.5/blob/main/Kimi_k1.5.pdf)：研究人员强调了基于课程的 RL 和长度惩罚对提升模型性能的作用。社区反馈促使新手将这些想法融入到新的 RL 训练方案中。
- [**GRPO 辩论引发 KL 散度争议**](https://github.com/huggingface/trl/issues/2608)：Hugging Face 用户质疑 GRPO 在优势计算中如何处理 KL。代码贡献者权衡了将 KL 直接应用于损失函数而非奖励函数的优缺点。

**主题 4. HPC 与 GPU 代码生成进展**

- [**Blackwell 突破代码限制**](https://github.com/vllm-project/vllm/pull/12271)：NVIDIA 的 Blackwell B100/B200 codegen 10.0 和 RTX 50 codegen 12.0 更新引发了对 sm_100a, sm_101a 以及可能的 sm_120 的期待。社区成员正在等待官方白皮书，目前仅靠部分 PR 说明进行了解。
- [**Triton 与 3.2 版本的博弈**](https://github.com/triton-lang/triton/issues/5669)：INT8×INT8 点积在新的 TMA 方法下崩溃，导致需要手动修复和 jit 重构。PyTorch Issue #144103 突出了 AttrsDescriptor 移除带来的向后兼容性问题。
- [**Accel-Sim 演讲引发热潮**](https://accel-sim.github.io/#overview)：HPC 探索者关注用于在 CPU 上进行 GPU 仿真的 Accel-Sim 框架。预定于 3 月底举行的演讲承诺将对模拟 GPU 性能和代码优化提供更深入的见解。

**主题 5. RAG 系统与工具创新**

- [**AgentWorkflow 助力多智能体**](https://twitter.com/llama_index/status/1882121805542170894)：LlamaIndex 发布了一个高级框架，用于编排并行工具使用和 Agent 协作。爱好者将其视为构建稳健多智能体解决方案的“下一步”。
- [**Sonar Pro 在 SimpleQA 中表现出色**](https://sonar.perplexity.ai/)：Perplexity 新推出的 Sonar Pro API 在基于实时搜索的问答中表现优于对手，同时承诺更低的成本。Zoom 将其集成到 AI Companion 2.0 中，开发者称赞其引用友好的方式。
- [**Ai2 ScholarQA 增强文献处理**](https://allenai.org/blog/ai2-scholarqa)：该系统具有跨引用超级能力，可回答多论文查询，从而加快学术评审。研究人员可以从逐个扫描 PDF 转向从整个语料库中获取精选见解。



## DeepSeek v3

**主题 1. DeepSeek R1 模型性能与集成**

- [**DeepSeek R1 在数学和推理方面超越竞争对手**](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF)：**DeepSeek R1 Distill Qwen 32B** 模型因其在复杂数学任务中的卓越表现而受到赞誉，超越了 **Llama 405B** 和 **DeepSeek V3 671B** 等模型。用户强调了其几何推理能力和多阶段 **RL** 训练，基准测试显示其**数学性能达到 92%**。
- [**DeepSeek R1 集成挑战**](https://github.com/deepseek-ai/DeepSeek-R1)：用户报告了将 **DeepSeek R1** 集成到 **GPT4All** 和 **LM Studio** 等平台时遇到的困难，理由是缺少公共模型目录以及需要 **llama.cpp** 更新。一些用户还面临 **API 性能下降**和**审查过滤器**的问题。
- [**DeepSeek R1 的思维链 (Chain-of-Thought) 推理**](https://api-docs.deepseek.com/guides/reasoning_model)：该模型外化推理步骤的能力（例如拼写 'razzberry' 与 'raspberry' 的对比）被视为一大特色。用户指出其具有增强 **Claude** 和 **O1** 等其他模型推理能力的潜力。

**Theme 2. AI 模型量化与微调**

- [**Unsloth 推出动态 4-bit 量化**](https://unsloth.ai/blog/dynamic-4bit)：Unsloth 新的**动态 4-bit 量化**方法有选择地避免压缩某些参数，在减少 **VRAM** 使用的同时保持准确性。用户将其与 **BnB 8-bit** 进行了对比，称赞其在模型优化方面的效率。
- [**Phi-4 的微调挑战**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb)：用户报告了微调 **Phi-4** 模型时遇到的问题，理由是输出质量差和循环修复。建议包括调整 **LoRA settings** 并确保使用高质量数据集以获得更好结果。
- [**重温 Chinchilla 公式**](https://paperswithcode.com/method/chinchilla)：围绕 **Chinchilla** 公式的讨论强调了模型大小与训练 **tokens** 之间的平衡。用户注意到许多模型超过了最佳阈值，导致资源利用效率低下。

**Theme 3. AI 基础设施与大规模投资**

- [**Stargate 项目：5000 亿美元 AI 基础设施计划**](https://www.verticaldata.io/insights/the-stargate-initiative-microsoft-and-openais-100-billion-data-center-project)：OpenAI 的 **Stargate Project** 旨在四年内投资 **5000 亿美元**，在美国建设先进的 AI 基础设施。初始资金包括来自 **SoftBank**、**Oracle** 和 **MGX** 的 **1000 亿美元**，重点关注就业创造和国家安全。
- [**谷歌向 Anthropic 投资 10 亿美元**](https://www.bloomberg.com/news/articles/2025-01-22/google-invests-another-1-billion-in-ai-developer-anthropic)：谷歌对 **Anthropic** 的追加投资显示了对该公司下一代模型的信心，引发了关于滚动融资策略和 AI 竞争的猜测。
- [**DeepSeek R1 硬件需求**](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF)：运行完整的 **DeepSeek R1 671B** 模型需要大量硬件，据估计需要花费 **18,000 美元** 购买多块 **NVIDIA Digits**。用户讨论了这种配置的可行性，一些人选择了更便宜的替代方案，如 **4xA4000**。

**Theme 4. AI 在创意与技术领域的应用**

- [**Gemini 2.0 Flash Thinking 模型发布**](https://x.com/OfficialLoganK/status/1881844578069999809)：谷歌的 **Gemini 2.0 Flash Thinking** 模型拥有 **100 万 token 上下文窗口**和 **64K 最大输出 token**，已针对 **DeepSeek R1** 进行了测试。用户赞扬了其在大规模推理任务中的潜力，尽管一些人指出了在处理复杂提示词时的挑战。
- [**NotebookLM 用于教会服务和学习工作流**](https://trendingcommunicator.substack.com/p/we-need-to-talk-about-notebooklm)：一位用户利用 **NotebookLM** 分析了 **16 个 5 小时的 YouTube 直播**，生成了一本 **250 页的书**和一份 **2000 页的圣经研究资料**。其他人将其集成到学习日常中，称赞其在参考文献查找方面的效率。
- [**AI 艺术面临敌对反应**](https://discord.com/channels/1002292111942635562/1002292112739549196/1331411232036487369)：用户报告了对 AI 生成艺术的负面反应，有些人甚至因为使用这些工具而被言语攻击。这反映了社会对创意领域 AI 应用的持续抵制。

**Theme 5. AI 安全、伦理与监管**

- [**对 AI 导致失业的担忧**](https://discord.com/channels/714501525455634453/986699377257119794/1331356492598743102)：一位创始人对其 AI 初创公司的成功可能导致的裁员表达了道德困境，引发了关于 AI 进步对社会经济影响的辩论。用户将其与同样削减工作岗位的日常自动化进行了比较。
- [**AI 安全指数与模型对齐**](https://futureoflife.org/document/fli-ai-safety-index-2024/)：围绕 **AI Safety Index** 的讨论强调了在 **MiniCPM** 等模型中建立稳健安全指标的必要性。用户质疑某些模型缺乏对齐和安全实践，强调了伦理 AI 开发的重要性。
- [**AI 监管挑战**](https://youtu.be/7EH0VjM3dTk?si=ooaXdzv_gIIyD070)：一场关于 AI 监管的会议探讨了近期政策的影响，来自 **SemiAnalysis** 的 **Dylan Patel** 讨论了不断演变的监管格局中的赢家和输家。人们对 AI 开发中政府与企业的重叠表示担忧。

## DeepSeek R1

**主题 1. 模型优化之战：量化、微调与 Scaling 之争**  

- [**Unsloth 的动态 4-bit 量化撼动 VRAM 效率**](https://unsloth.ai/blog/dynamic-4bit)：**Unsloth 的动态 4-bit 量化**避免了压缩关键参数，在保持准确性的同时大幅降低了 VRAM 占用。用户将其与 **BnB 8-bit** 进行了比较，指出动态 4-bit 在内存和性能之间取得了平衡。  
- [**Phi-4 微调失败引发架构争论**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb)：微调 **Phi-4** 导致输出效果不佳，用户将其归咎于模型架构。其他 Notebook 中的权宜之计暗示了对专门超参数的需求。  
- [**训练 Token 过剩背景下重审 Chinchilla Scaling Laws**](https://paperswithcode.com/method/chinchilla)：关于 **Chinchilla 的模型大小与 Token 比例**的讨论再次升温，许多模型已经超越了阈值。经验性的 Scaling 证明敦促为了效率进行资源优化。  

**主题 2. AI 基础设施军备竞赛：5000 亿美元项目与硬件障碍**  

- [**Stargate 项目 5000 亿美元的雄心面临资金质疑**](https://www.verticaldata.io/insights/the-stargate-initiative-microsoft-and-openais-100-billion-data-center-project)：OpenAI 的 **Stargate Project** 旨在投入 5000 亿美元用于 AI 基础设施，但像 **Gavin Baker** 这样的批评者[质疑 SoftBank 的流动性](https://x.com/gavinsbaker/status/1882081746877063677)。**Elon Musk** [抨击该提案](https://x.com/elonmusk/status/1881923570458304780)不切实际。  
- [**DeepSeek R1 671B 需 1.8 万美元硬件，引发成本辩论**](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF)：据报道，运行 **DeepSeek R1 671B** 需要 **4x NVIDIA Digits**（1.8 万美元），而用户提出了更便宜的 **4xA4000** 方案。怀疑者认为更大的模型并不等于更好的 ROI。  
- [**Apple Silicon 挑战 R1 32B 的 4-bit 量化极限**](https://huggingface.co/Joseph717171/DeepSeek-R1-Distill-Llama-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF)：**M3 Max MacBook Pro** 用户测试了 **R1 32B** 的 **4-bit 量化**，注意到质量有所下降。尽管存在精度权衡，**MLX 优化变体**仍带来了速度提升。  

**主题 3. Agentic AI：自主系统中的炒作与现实**  

- [**脚本工作流被贴上 “Agentic AI” 标签引发用户反感**](https://cline.bot/blog/why-ai-engineers-need-planning-more-than-perfect-prompts-2)：成员们嘲笑了将基础脚本工具等同于 **Agentic AI** 的营销主张，理由是缺乏真正的自主性。要求在 **“Agent”** 定义上保持透明度的呼声日益高涨。  
- [**GRPO 和 T1 RL 论文预示强化学习变革**](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)：**GRPO** 的极简实现和 **T1 RL** 的规模化自我验证[论文](https://arxiv.org/abs/2501.11651)引发了兴趣。人们对优势计算中的 **KL 散度处理**表示担忧。  
- [**OpenAI 的 “Operator” 预告浏览器自动化，但 API 缺席**](https://x.com/steph_palazzolo/status/1882091855606895073)：泄露消息透露了用于浏览器操作的 **ChatGPT Operator**，但不支持 API。用户推测这是通往 **“PhD-level Super Agents”** 的权宜之计。  

**主题 4. 工具动荡：IDE 之战、API 怪癖与 RAG 现实**

- [**Windsurf 的自动记忆功能与 Cascade 的循环冲突**](https://forum.cursor.com/t/persistent-intelligent-project-memory/39109)：**Windsurf 的项目记忆功能**赢得了赞誉，但 **Cascade** 在 FastAPI 文件访问上出了差错，并陷入了修复循环。用户敦促重新编写 Prompt 以跳出循环。  
- [**LM Studio 0.3.8 增强了 LaTeX 和 DeepSeek R1 “Thinking” UI**](https://x.com/lmstudio/status/1881849443999416802)：此次更新增加了 **LaTeX 渲染**和 **DeepSeek R1** 界面，修复了 Windows 安装程序的 Bug。用户称赞了 Vulkan GPU 去重技术带来的稳定性。  
- [**Perplexity 的 Sonar Pro API 在 SimpleQA 基准测试中超越竞争对手**](https://sonar.perplexity.ai/)：**Sonar Pro** 以 **73.3% 的数学得分**和 **1M-token 上下文**主导了基准测试，但上线初期的故障导致了 500/403 错误。关于欧盟托管的 GDPR 合规性辩论也愈演愈烈。  

**主题 5. 伦理、审查与劳动力流失担忧**  

- [**DeepSeek R1 性能下降 85% 引发审查猜疑**](https://openrouter.ai/deepseek/deepseek-r1/uptime)：用户将 **R1 隔夜性能崩溃**归咎于内部审查过滤器。针对敏感 Prompt 的绕过方法开始流行，同时运行时间监控器也在跟踪修复进度。  
- [**创业公司创始人深陷 AI 驱动裁员的愧疚**](https://x.com/ai_for_success/status/1881887921156005947)：一位创始人对他们的 AI 工具可能导致的失业表示哀叹，引发了关于自动化伦理的辩论。批评者将其比作历史上的技术变革。  
- [**微软 Phi-3.5 的安全推送引发了 Uncensored Fork**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)：**Phi-3.5 “过度热衷”的审查**导致了一个 [Hugging Face fork](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)。程序员批评其拒绝回答良性查询，并嘲讽其在井字棋问题上的回避。

---

# 第 1 部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **来自 Unsloth 的强力 4-bit 量化**：Unsloth 引入了[一种动态 4-bit 量化方法](https://unsloth.ai/blog/dynamic-4bit)，该方法避免压缩特定参数，在仅占用一小部分 VRAM 的情况下保持了极高的准确度。
  
  - 用户将其与 **BnB 8-bit** 方案进行了比较，结果显示动态 4-bit 在仅略微增加内存需求的情况下保留了性能。
- **Phi-4 微调惨败**：用户报告了 **Phi-4** 模型在微调后输出效果不佳的问题，质疑是否是模型架构的原因。
  
  - 他们注意到微调在其他 Notebook 上可以运行，暗示 **Phi-4** 可能需要专门的设置才能获得更好的响应。
- **Chinchilla 危机：规模 vs. Token**：参与者重新讨论了 **Chinchilla** 公式，重点关注模型规模与训练 Token 之间的相互作用，以实现效率最大化。
  
  - 他们观察到许多模型超出了建议的阈值，并分享了[缩放增益的经验证明](https://paperswithcode.com/method/chinchilla)，呼吁进行资源优化。
- **合成数据热潮**：一些人主张使用“无限合成数据流”来强化模型训练，强调 **eval 合规性**和精选输入。
  
  - 他们警告说，如果数据精选过于随意，会出现 **Garbage in, garbage out（垃圾进，垃圾出）** 的情况，敦促对合成数据集进行专门监督。
- **Agentic AI “全是炒作”的说法**：成员们对将基础的基于脚本的系统贴上 **Agentic AI** 标签的营销策略表示不满，认为其缺乏真正的自主能力。
  
  - 他们指出了宏大的品牌声明与有限的 **Agent** 功能现状之间的差距。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **网页搜索之困：Codeium 对比 Windsurf**：一位用户敦促 **Codeium** 追赶 **Windsurf** 的网页搜索能力，并引用了 [扩展更新日志](https://marketplace.visualstudio.com/items/Codeium.codeium/changelog)。
  
  - 他们通过引用 [隐私政策](https://codeium.com/privacy-policy) 并对比 JetBrains 与 VS Code 的功能，对数据处理政策提出了质疑。
- **Windsurf IDE 访问难题**：在最新的 **Windsurf** 更新后，多位成员在 Ubuntu 上难以打开 **FastAPI** 文件，并指出 **Cascade** 无法读取或编辑某些路径。
  
  - 虽然有人建议调整文件权限或环境设置，但部分开发者的这一问题仍然存在。
- **Cascade 令人困惑的代码对话**：几位开发者报告称，在处理复杂的代码调试提示词时，**Cascade** 会陷入重复修复的循环。
  
  - 他们发现重新表述指令或提供更多上下文有助于打破循环，这显示了结构良好的提示词的重要性。
- **提示词威力与项目记忆**：用户赞扬了 **Windsurf** 自动生成的记忆功能，它可以跨会话携带上下文，并引用了一个 [论坛帖子](https://forum.cursor.com/t/persistent-intelligent-project-memory/39109) 以获取更深层的项目记忆想法。
  
  - 他们还引用了一篇 [Cline 博客文章](https://cline.bot/blog/why-ai-engineers-need-planning-more-than-perfect-prompts-2)，强调“规划优于完美的提示词”以改善 AI 交互。
- **Diff 查看器困境与 Writemode 烦恼**：社区成员经常在 **Windsurf** 的 Diff 查看器中遇到配色方案混淆，并询问 **Writemode** 是否免费。
  
  - 一位用户澄清这是付费功能，而其他用户则在探索通过 **Flow Actions** 连接多个 **LLMs**，以实现更灵活的集成。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.8 增强思考 UI**：新发布的 [LM Studio 0.3.8](https://x.com/lmstudio/status/1881849443999416802) 为 **DeepSeek R1** 添加了 **思考 UI (Thinking UI)**，并通过 `\text{...}` 块支持 **LaTeX** 渲染。
  
  - 与会者注意到它解决了 Windows 安装程序问题并消除了重复的 **Vulkan GPUs**，使使用更加顺畅。
- **DeepSeek R1 Distill Qwen 32B 惊艳数学迷**：用户称赞 **DeepSeek R1 Distill Qwen 32B** 在复杂的 AIME 级别数学测试中表现优于 **Llama 405B** 等其他本地模型。
  
  - 他们引用了 [Hugging Face 发布页面](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF) 以获取更多细节，并赞扬其增强的推理步骤。
- **Apple Silicon 上的量化博弈**：参与者探索了使用 **4-bit** 量化以在 **M3 Max MacBook Pro** 上运行 **R1 32B** 模型，并注意到答案质量可能下降。
  
  - 多位用户测试了 **MLX** 优化的变体以降低内存需求，暗示尽管可能存在精度权衡，但速度更快。
- **671B 部署的高昂代价**：据估计，运行完整的 **DeepSeek R1 671B** 需要价值 **18,000 美元** 的硬件（多个 **NVIDIA Digits**），这引发了关于可行性的争论。
  
  - 其他人提到 **4xA4000** 配置是一条更便宜的路线，并评论说更大的模型并不总是能保证更优的性能。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Stargate 的 5000 亿美元计划引发关注**：有传言称 **Project Stargate** 正寻求来自 SoftBank 的 5000 亿美元投资，但鉴于 SoftBank 有限的流动性和债务状况，这一消息引发了质疑，正如 **Gavin Baker** 在[此处](https://x.com/gavinsbaker/status/1882081746877063677)所指出的。
  
  - 讨论引用了 [Elon Musk 的评论](https://x.com/elonmusk/status/1881923570458304780)，他认为 SoftBank 并没有这么多现金，这进一步加深了对该提案真实性的怀疑。
- **对 DeepSeek 在 Razzberry 拼写上的质疑**：成员们测试了 [DeepSeek R1 Distill Llama-8B](https://huggingface.co/Joseph717171/DeepSeek-R1-Distill-Llama-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF)，观察它在拼写“razzberry”与“raspberry”时如何将推理过程外显化。
  
  - 他们强调了关于零个“p”的滑稽混乱，展示了 DeepSeek 的 Chain-of-Thought（思维链）揭示过程，以及它在模型间实现更深层协同的潜力。
- **FLAME 在 Excel 领域表现出色**：一个名为 **FLAME** 的 **60M 参数**模型使用 [Excel 专用 Tokenizer](https://arxiv.org/abs/2301.13779) 来处理公式补全和修复，其表现足以媲美 Davinci 等更大型的模型。
  
  - 社区成员赞赏其针对性的训练和较小的体积，认为这是处理特定领域任务的一种强有力的方法。
- **EvaByte 实现无 Token 化**：[EvaByte](https://hkunlp.github.io/blog/2025/evabyte/) 作为一款 **6.5B 基于字节的模型**首次亮相，它使用 **5 倍更少的数据**并实现了 **2 倍更快的解码**，在无需 Tokenizer 的情况下开启了多模态的可能性。
  
  - 尽管怀疑者对其硬件效率提出质疑，但结果表明，向具有更广泛灵活性、面向字节的训练转型是可行的。
- **STAR 与 TensorGrad 撼动模型架构**：[STAR](https://arxiv.org/abs/2411.17800) 概述了一种改进 LLM 结构的进化方法，声称在扩展性和效率方面获得了性能提升。
  
  - [TensorGrad](https://github.com/thomasahle/tensorgrad) 引入了用于简化矩阵运算和符号优化的命名边（named edges），吸引了渴望摆脱繁琐数字维度映射的开发者。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Stargate 获得宏大增长**：OpenAI 新的 **Stargate Project** 确定了一项为期四年、总额高达 **5000 亿美元**的巨额计划，其中 **1000 亿美元**已由 **SoftBank**、**Oracle** 和 **MGX** 立即承诺，用于推动先进的 AI 基础设施建设（[参考资料](https://www.verticaldata.io/insights/the-stargate-initiative-microsoft-and-openais-100-billion-data-center-project)）。
  
  - 这一巨大的倡议预计将创造大量就业机会，并巩固美国在 AI 领域的领导地位，Microsoft 官方博客和各大科技频道均对此进行了报道（[博客链接](https://blogs.microsoft.com/blog/2025/01/21/microsoft-and-openai-evolve-partnership-to-drive-the-next-phase-of-ai/)）。
- **Bespoke-Stratos-32B 挑战基准测试**：从 **DeepSeek-R1** 蒸馏而来的 **Bespoke-Stratos-32B** 模型在推理任务中超越了 **Sky-T1** 和 **o1-preview**，且仅需 **47 倍更少**的训练样本（[来源](https://x.com/madiator/status/1882131703927652762)）。
  
  - 它利用一个价值 **800 美元**的开源数据集实现了极具成本效益的结果，激发了社区对数据收集和协作改进的兴趣。
- **谷歌再次向 Anthropic 投资 10 亿美元**：为了再次展示信心，**Google** 向 **Anthropic** 注资 **10 亿美元**，引发了关于滚动融资策略和 AI 竞争的猜测（[推文](https://x.com/ns123abc/status/1881965986695524472)）。
  
  - 这笔投资强化了 Google 对新兴 AI 参与者的持续承诺，讨论重点关注了 Anthropic 下一代模型可能的扩张。
- **Gemini-2.0-Flash 跃升至 Arena 榜首**：**Gemini-2.0-Flash-Thinking** 以 **73.3%** 的强劲数学得分和 **100 万 Token** 的上下文窗口夺得 Chatbot Arena 第一名（[链接](https://x.com/lmarena_ai/status/1881848934743904319)）。
  
  - 开发者称赞了其在大规模推理方面的潜力，同时也承认未来的迭代可能会进一步优化其性能。
- **GRPO 调整与 T1 RL 的成功**：社区成员对 **GRPO** 方法提出质疑，指出在优势计算（advantage calculations）中处理 **KL 散度**（KL divergence）时存在疑虑，并引用了 [TRL 的相关问题](https://github.com/huggingface/trl/issues/2608)。
  
  - 与此同时，来自**智谱和清华**的一篇新 **T1** 论文详细介绍了针对大语言模型的扩展 RL（强化学习），将**试错**与**自我验证**相结合，详见 [arXiv](https://arxiv.org/abs/2501.11651)。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.0 备受瞩目**：Google 推出了具有 100 万 token 上下文窗口、**64K** 最大输出 token 以及原生代码执行支持的 **Gemini 2.0**，正如[这条推文](https://x.com/OfficialLoganK/status/1881844578069999809)所确认的那样。
  
  - 用户将其与 **DeepSeek R1** 进行了对比测试，对其处理复杂任务的能力和整体效率表示好奇。
- **Aider 的 Markdown 改造**：用户描述了一种改进 **Aider** 的精细方法，即将主要功能、规格和重构步骤存储在 markdown 中，并参考了[高级模型设置](https://aider.chat/docs/config/adv-model-settings.html#model-settings)。
  
  - 他们强调，**LLM 自我评估**配合单元测试有助于生成更简洁的代码并缩短开发循环。
- **辩论 R1 vs. Sonnet**：社区成员观察到了 **DeepSeek R1** 和 **Sonnet** 之间的区别，并引用了 [DeepSeek Reasoning Model 文档](https://api-docs.deepseek.com/guides/reasoning_model)中的思维链能力。
  
  - 他们注意到 **Sonnet** 反复提供更彻底的建议，并提议将 R1 的推理输出合并到其他模型中以应对高级场景。
- **Aider 中针对 PDF 的 RAG**：一位用户询问了在 Aider 中引用 PDF 的 **RAG** 功能，发现 Sonnet 通过一个简单的内置命令即可支持该功能。
  
  - 该讨论激发了利用外部数据源为 Aider 工作流提供更深层次上下文的想法。
- **Aider 升级与 Nix 历险**：多位用户在 Aider 中遇到了**升级错误**和配置障碍，分享了诸如删除 `.aider` 目录或通过 `pip` 重新安装等技巧。
  
  - 他们还提到了 [NixOS 中 Aider 0.72.1 的 PR](https://github.com/NixOS/nixpkgs/pull/375634)，并思考了 Neovim 插件设置，但最终建议仍在审查中。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt 获得巨额资金支持**：今天，**Bolt** 宣布获得由 Emergence 和 GV 领投、Madrona 和 **The Chainsmokers** 支持的 **1.055 亿美元** B 轮融资，详见[这条推文](https://x.com/boltdotnew/status/1882106655258894390)。
  
  - 他们感谢社区对 **devtools** 和 **AI** 增长的支持，并承诺未来将加强 Bolt 的功能。
- **Telegram 版俄罗斯方块新花样**：一位开发者旨在构建一个以肉类为主题方块的 **Telegram** 小程序版俄罗斯方块，并分享了 [Telegram Apps Center](https://t.me/tapps_bot?profile.) 作为资源。
  
  - 他们计划加入**排行榜**，希望这个古怪的概念能激发社区协作。
- **Claude 解决代码困惑**：成员们发现，在 Bolt 遇到困难时，使用 **Claude** 处理棘手的代码任务和检索策略更新非常成功。
  
  - 他们注意到 Claude 在处理 **Supabase** 用户权限方面的彻底性，赞扬了 AI 驱动调试的协同效应。

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **DeepSeek R1 与 OpenAI 展开竞争**：**DeepSeek R1** 以 92% 的数学性能超越了 **Gemini** 和 **O1**，凭借技术实力和成本优势势头强劲，详见 [DeepSeek R1 - API, Providers, Stats](https://openrouter.ai/deepseek/deepseek-r1)。
  
  - 讨论重点包括高级几何推理、多阶段 RL 训练，以及对 **DeepSeekMath** ([arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)) 的深入探讨，该研究推动了基于模型计算的边界。
- **5000 亿美元的 Stargate Project 引发争论**：OpenAI 披露了 **Stargate Project**，涉及 5000 亿美元的 AI 基础设施资金，参考其[官方推文](https://fxtwitter.com/OpenAI/status/1881830103858172059)。
  
  - 评论员对政府与企业的重叠以及美国的决策收益表示质疑，引用了 [AshutoshShrivastava](https://fxtwitter.com/ai_for_success/status/1881887921156005947) 等来源。
- **AI 初创公司面临职位取代的道德拷问**：一位创始人对其快速发展的 AI 解决方案可能引发的裁员表示悔意，称其为一个道德难题。另一位用户将其比作同样削减职位的日常自动化，反映了更广泛的社会经济问题。
  
  - 许多人同意此类副作用伴随着新的 AI 发展，强调了进步与劳动力流失之间的紧张关系。
- **IntellAgent 重新定义 Agent 评估**：开源的 **IntellAgent** 框架应用模拟交互进行多层级 Agent 诊断，展示于 [GitHub](https://github.com/plurai-ai/intellagent)。
  
  - 发表在 [arxiv.org/pdf/2501.11067](https://arxiv.org/pdf/2501.11067) 的相应论文分享了数据驱动批评带来的惊人结果，并在 `intellagent_system_overview.gif` 中展示了视觉工作流。
- **UI-TARS 和 OpenAI Operator 成为焦点**：**Hugging Face** 发布了 **UI-TARS**，旨在实现自动化 GUI 任务，包括 [UI-TARS-2B-SFT](https://huggingface.co/bytedance-research/UI-TARS-2B-SFT) 等变体，详见其[论文](https://huggingface.co/papers/2501.12326)。
  
  - 与此同时，据 [Stephanie Palazzolo](https://x.com/steph_palazzolo/status/1882091855606895073/) 报道，OpenAI 正在为 ChatGPT 准备 **Operator** 功能以执行浏览器操作。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 为网页搜索定价**：新的网页查询定价为 **每 1000 次结果 4 美元**，将于明天开始执行，单次请求成本约为 **低于 0.02 美元**。
  
  - 成员们对这种精简的模型表示欢迎，但对大规模使用开始后的计费细节表示好奇。
- **API 访问权限小范围测试上线**：OpenRouter 的 **API 访问权限** 将于明天开放，并为用户提供额外的 **Customizability**（可定制性）选项。
  
  - 成员们期待测试这些新功能，并分享关于性能和集成的反馈。
- **DeepSeek R1 性能暴跌 85%**：报告显示 DeepSeek R1 的 API 性能在隔夜之间**下降了 85%**，引发了对内部审查和 **Censorship**（审查）过滤器的担忧。
  
  - 一些人分享了针对敏感提示词的变通方法，而其他人则关注 [DeepSeek R1 – Uptime and Availability](https://openrouter.ai/deepseek/deepseek-r1/uptime) 以获取官方修复。
- **Cerebras 的 Mistral Large 吊足胃口，但用户仍无法使用**：许多人希望在 OpenRouter 上看到 **Cerebras 的 Mistral Large**，但目前仍未对公众开放。
  
  - 感到沮丧的用户转而使用 **Llama** 模型，质疑 Mistral Large 是否如宣传的那样准备就绪。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonar Pro 引起轰动**：今天，**Perplexity** 推出了新的 [Sonar 和 Sonar Pro API](https://sonar.perplexity.ai/)，使开发者能够将生成式搜索与实时网页研究及强大的引用功能相结合。
  
  - **Zoom** 等公司已采用该 API 来提升 **AI Companion 2.0**，同时 **Sonar Pro** 的定价据称比其他同类产品更低。
- **SimpleQA 基准测试：Sonar Pro 大放异彩**：**Sonar Pro** 在 **SimpleQA 基准测试**中表现优于主要竞争对手，在回答质量上超越了其他搜索引擎和 LLM。
  
  - 支持者称赞其“全网覆盖”能力远超竞争解决方案。
- **对比层出不穷：模型调整与欧洲需求**：社区成员报告称 **Sonar Large** 现在运行速度超过了 **Sonar Huge**，官方暗示将退役旧模型。
  
  - 与此同时，欧洲的 **GDPR 合规**推进引发了关于在本地数据中心托管 **Sonar Pro** 的讨论。
- **Altman 预告“博士级超级 Agent”**：传闻 **Altman** 在华盛顿特区的一次简报中提到了先进的“博士级超级 Agent (PhD-level Super Agents)”，引发了人们对下一代 AI 能力的好奇。
  
  - 观察人士将这些假设的 Agent 视为即将取得重大进展的信号，尽管具体细节仍然很少。
- **Anduril 的 10 亿美元武器工厂备受关注**：有关 **Anduril** 建立 10 亿美元“自主武器工厂 (Autonomous Weapons Factory)”的消息提高了人们对防御导向型机器系统的兴趣，如[此视频](https://www.youtube.com/embed/MEgG6BQrmKw)所示。
  
  - 参与者讨论了**自主战争**，并强调了与武器化 AI 相关的伦理问题。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 的代码处理**：成员们讨论了 [MCP language server](https://github.com/isaacphi/mcp-language-server) 中不断发展的代码编辑功能，指出其处理大型代码库的能力有所提高，并能与 Git 操作协同工作。
  
  - 他们提到了一个新的 [Express 风格的 API Pull Request](https://github.com/modelcontextprotocol/typescript-sdk/pull/117)，这可能会统一 MCP 服务器在语义工具选择的同时更新代码的方式。
- **利用 Brave 浏览获取清晰文档**：用户依靠 Brave Search 获取更新的文档，并将参考资料编译成 **Markdown** 以进行快速集成。
  
  - 他们强调了使用 **brave-search MCP server** 方法来抓取和自动化文档检索，并称赞了该流程的精简性。
- **2024 年 GPT 的不满**：社区成员对**自定义 GPT (custom GPT)** 未能整合 ChatGPT 的新功能表示沮丧，这加剧了对自定义 GPT 市场的怀疑。
  
  - 他们指出担心这些机器人会失去相关性，并对改进速度缓慢表示失望。
- **Claude Desktop 的 Prompt 展示**：参与者探索了将 Prompt 挂载到 **Claude Desktop** 的方法，重点关注如何通过 `prompts/List` 端点展示 Prompt。
  
  - 他们分享了日志工具示例和部分代码片段，旨在简化测试专业 Prompt 的过程。
- **Apify Actors 与 SSE 试验**：开发者正在开发 [Apify Actors 的 MCP 服务器](https://github.com/apify/actors-mcp-server/)，构建数据提取功能，但面临动态工具集成的挑战。
  
  - 围绕 **Anthropic TS client** 的问题凸显了对 SSE 端点的困惑，导致一些成员在等待修复期间转向使用 Python。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Ai2 ScholarQA 加速文献综述**：Ai2 ScholarQA 推出了一款基于 RAG 的多论文查询解决方案，并发布了[官方博客文章](https://allenai.org/blog/ai2-scholarqa)解释其交叉引用功能。
  - 它旨在帮助学术界人士从各种开放获取论文中快速获取研究见解，重点关注对比分析。
- **特朗普启动 5000 亿美元 Project Stargate**：特朗普总统宣布了 [Project Stargate](https://x.com/dwarkesh_sp/status/1881844437346902297)，承诺在四年内投入 5000 亿美元巨资扩建美国的 AI 基础设施，并获得了 OpenAI、SoftBank 和 Oracle 的支持。
  - Elon Musk 和 Gavin Baker 等评论员对该金额的可行性表示质疑，称其“荒谬”，但仍对其雄心表示认可。
- **Bespoke-Stratos-32B 问世**：根据[官方公告](https://x.com/madiator/status/1882131703927652762?s=46)，一款从 DeepSeek-R1 蒸馏而来的新型推理模型 **Bespoke-Stratos-32B** 展示了先进的数学和代码推理能力。
  - 开发者强调，它采用了 “Berkeley NovaSky 的 Sky-T1 配方”，在推理基准测试中超越了之前的模型。
- **Clay GTM 融资 4000 万美元**：Clay GTM 宣布以 12.5 亿美元的估值完成了 [4000 万美元的 B 轮扩展融资](https://x.com/vxanand/status/1882061978593837344?s=46)，其强劲的营收增长引起了投资者的关注。
  - 他们现有的资金大部分仍未动用，并计划放大势头以推动进一步增长。
- **LLM Paper Club 聚焦语言模型物理学**：**LLM Paper Club** 活动重点关注 **Physics of Language Models** 和 **Retroinstruct**，详情见[此链接](https://lu.ma/2d1b6i2t)。
  - 参与者可以通过 RSS 标志订阅 [Latent.Space](http://Latent.Space) 的活动提醒，确保不会错过任何活动。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA Blackwell Codegen 持续推进**：PR [#12271](https://github.com/vllm-project/vllm/pull/12271) 揭示了 **Blackwell B100/B200 codegen 10.0**，并预告了 **RTX 50** codegen 12.0，引发了对 sm_100a 和 sm_101a 的期待。
  - 尽管有关于 **sm_120** 的传闻以及 **Blackwell** 白皮书的推迟，社区仍热切期待更多消息以及关于 [Accel-Sim 框架](https://accel-sim.github.io/#overview)的演讲。
- **Triton 3.2 HPC 故障**：当前的 **TMA** 实现可能会在 `@triton.autotune` 时崩溃，导致持久化 matmul 描述符和数据依赖问题出现混乱。
  - 一位用户指向了 [GridQuant gemm.py 第 79-100 行](https://github.com/niconunezz/GridQuant/blob/main/scripts/gemm.py#L79-L100)以获取描述符创建的见解，强调了 Triton kernel 的复杂性。
- **GRPO 在 RL 中势头强劲**：一个极简的 **GRPO** 算法已接近完成，其初始版本预计很快运行，早期实验已在进行中。
  - [**Kimi-k1.5** 论文](https://github.com/MoonshotAI/Kimi-k1.5/blob/main/Kimi_k1.5.pdf)和 TRL 中新的 [**GRPO trainer**](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) 凸显了人们对基于课程的学习（curriculum-based）强化学习日益增长的兴趣。
- **利用 HPC 工具加速 LLM 推理**：一位用户寻求更快的 **Hugging Face** `generate()` 性能，引用了[此提交](https://github.com/huggingface/trl/commit/2ecd53ad77ef2a27729176e89299cba37b2487c4)并讨论了针对更重型模型的 **liuhaotian/llava-v1.5-7b**。
  - 同时，[这篇 PyTorch 博客文章](https://pytorch.org/blog/accelerating-llm-inference/)探讨了 HPC 友好型策略，如专门的调度和内存优化，以提升大模型推理速度。
- **Torch 与 Triton 在 3.2 版本上的博弈**：新的 **Triton 3.2** 删除了 `AttrsDescriptor`，导致 `torchao` 和 **torch.compile** 中断，记录在 [PyTorch issue #144103](https://github.com/pytorch/pytorch/issues/144103) 中。
  - [Triton issue #5669](https://github.com/triton-lang/triton/issues/5669) 中的 INT8 x INT8 点积失败，加上[重大的 JIT 重构](https://github.com/triton-lang/triton/pull/5512)，揭示了反复出现的向后兼容性难题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **谷歌的 Titan 预告与 Transformers 之争**：在 **Google 的 Titans** 宣布利用先进的内存特性获得卓越性能后，成员们讨论了推理阶段处理（inference-time handling）的潜在改进。
  
  - 社区观点认为，原始方法可能难以复制，反映了对实验透明度不足的担忧。
- **数值稳定性带来的 Grokking 收益**：最近的一篇论文 [Grokking at the Edge of Numerical Stability](https://arxiv.org/abs/2501.04697) 强调了**数值问题**（numerical issues）如何阻碍模型训练，引发了关于改进优化策略的讨论。
  
  - 成员们辩论了一种**一阶**（first-order）方法，并引用了 [这个 GitHub 仓库](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability) 以获取更深入的见解。
- **DeepSeek 奖励模型传闻**：与会者研究了 **DeepSeek-R1-Distill-Qwen-1.5B** 模型的指标差异，引用了部分评估中的 **0.834 (n=2)** 对比 **97.3 (n=64)**。
  
  - [DeepSeek_R1.pdf](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) 的链接引用了解码策略和基于奖励训练的架构细节。
- **Minerva Math 与 MATH-500 的改进**：参与者测试了 **minerva_math** 与 **sympy** 的符号等价性，引用了来自 OpenAI “Let's Think Step by Step”研究中的 **MATH-500** 子集。
  
  - 他们讨论了 **DeepSeek R1** 在这些任务中表现得像基座模型（base model）还是需要聊天模板（chat template），并指向 [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) 以获取更多背景信息。
- **线性注意力与多米诺学习**：成员们探索了一种专门的**线性注意力**（linear attention）架构，旨在不损失稳健性能的情况下提高速度。
  
  - 他们还讨论了技能堆叠中的**多米诺效应**（Domino effect），引用了 [Physics of Skill Learning](https://arxiv.org/abs/2501.12391) 来强调神经网络中顺序能力是如何出现的。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DeepSeek 与 o1 及 Sonnet 的对决**：**DeepSeek** 在数学和 GitHub 相关的基准测试中与 **o1** 和 **Sonnet** 等模型进行了对比，表现强劲。它可以在其官方网站上免费访问，并为各种平台提供 API 集成。
  
  - 一些用户在 **o1** 上遇到了问题，但他们称赞官方的 **DeepSeek R1** 功能在快速评估方面更稳定，这推动了对一致性模型替代方案的需求。
- **AI 安全引发好奇**：成员们质疑为什么 **AI 网络安全** 仍然被忽视，并提到 **CrowdStrike** 多年来一直使用 ML。他们看到了生成式 AI 在自动化威胁检测和基于代码的入侵分析方面的潜力。
  
  - 社区声音认为，企业更强调利润而非基础安全，指出营销炒作与真正的用户保护之间存在脱节。
- **纯图像 GPT 势头强劲**：一位用户想完全基于聊天的**截图**来训练一个类似 GPT 的模型，绕过基于文本的流水线。他们想知道在现有的 chat completion API 中是否可以进行文件上传或图像数据处理。
  
  - 其他人权衡了直接**图像摄取**（image ingestion）的可行性，建议在 API 支持内联文件之前增加额外的预处理步骤。
- **OCR 困惑与地图解决方案**：成员们发现 **OCR 提示词**会导致严重的**幻觉**（hallucinations），特别是在不受约束的示例中。他们探索了一种读取地图的专门变通方法，希望 **OpenAI 的 O 系列**能尽快解决空间数据问题。
  
  - 他们警告了自由格式 OCR 设置中的**上下文污染**（context contamination），结论是在更好的 GIS 支持出现之前，特定领域的约束更为安全。

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 彻底改变了教会聚会**：一位用户利用 NotebookLM 分析了 16 个时长 5 小时的 **YouTube 直播**，生成了一本 **250 页的书**和一份 **2000 页的圣经研读资料**。
  - 他们引用了 [We Need to Talk About NotebookLM](https://trendingcommunicator.substack.com/p/we-need-to-talk-about-notebooklm) 作为动力，赞扬其处理海量文本的能力。
- **学习工作流开始采用 NotebookLM**：一位用户在数周的学习例程中集成了 NotebookLM，认为它简化了参考文献的查找。
  - 他们分享了一个 [YouTube 视频](https://youtu.be/wvf4EXJJsU8) 展示其效率，激励他人采用类似的方法。
- **Gemini 在 Prompt 优化方面势头强劲**：成员们报告称，通过将 NotebookLM 与 **Gemini** 配合使用以精炼指令，可以获得更好的输出效果。
  - 他们称赞了 Gemini 对提升清晰度的影响，但也指出了在针对高度特定文档时的挑战。
- **APA 引用和音频概览引发讨论**：参与者在处理 **APA 引用**时遇到了困难，发现 NotebookLM 除非调整名称，否则会依赖之前使用过的来源。
  - 他们还讨论了每天生成多达三次的新音频概览（Audio Overviews），并警告存在重复风险，且需要先删除旧文件。
- **CompTIA A+ 内容受到关注**：一位用户发布了 **CompTIA A+** 音频系列的第一部分，后续章节正在制作中。
  - 社区成员将其视为自主进度认证准备的关键资源，NotebookLM 提供了快速的信息获取。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Flux 模型表现与负面提示词引导**：一位用户报告称，**去蒸馏的 Flux 模型**在配置了 **CFG** 设置后，在某些内容上表现更好，尽管运行速度较慢。
  - 他们指出，*负面提示词（Negative Prompts）*可以提高 Prompt 的遵循度，但会增加更重的计算开销。
- **AI 艺术引发敌对情绪**：一些参与者在展示 AI 生成的艺术作品时遇到了**敌对反应**，包括因为使用这些工具而被言语攻击。
  - 他们提到，对 **AI 艺术**的负面情绪已持续多年，引发了关于更广泛接受度的讨论。
- **Discord 机器人诈骗横行**：成员们标记了来自机器人账号的异常私信（DMs），这些账号索要个人信息，揭示了一种持续的诈骗趋势。
  - 有人回忆起早期的付费“服务”推销，暗示这些诈骗仍然是 **Discord** 的老问题。
- **CitivAI 的故障与担忧**：一位用户指出 **CitivAI** 每天多次宕机，引发了对该服务稳定性的担忧。
  - 其他人也分享了类似的经历，质疑该平台的可靠性。
- **SwarmUI 面部修复备受关注**：一位用户询问关于 **SwarmUI 中的面部修复**问题，想知道是否需要 Refiner 来提高图像保真度。
  - 他们注意到社区对增强现实感的追求，旨在进一步优化**图像生成流水线（Image-generation Pipelines）**。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AgentWorkflow 发布并增强多 Agent 系统**：**LlamaIndex** 宣布推出 [AgentWorkflow](https://twitter.com/llama_index/status/1882121805542170894)，这是一个基于 LlamaIndex Workflows 构建的新型高级框架，旨在支持多 Agent 解决方案。
  - 他们强调了**扩展的工具支持**和社区的热情，称其为“迈向更强大 Agent 协作的下一步”。
- **DeepSeek-R1 挑战 OpenAI 的 o1**：**DeepSeek-R1** 模型已接入 LlamaIndex，其性能可与 **OpenAI 的 o1** 媲美，并可用于[全栈 RAG 聊天应用](https://twitter.com/llama_index/status/1882144637558890996)。
  - 用户赞扬了其**集成的便利性**，并寄希望于“进一步扩展以用于实际场景”。
- **开源 RAG 系统结合 Llama3 与 TruLens 力量**：一份详细指南贡献了使用 **LlamaIndex**、**Meta Llama 3** 和 [TruLensML](https://twitter.com/TruLensML) 构建开源 RAG 系统的分步方法。
  - 它对比了**基础 RAG 方法**与“带有 @neo4j 的 Agentic 变体”，包括关于 **OpenAI vs Llama 3.2** 的性能见解。
- **AgentWorkflow 探索并行 Agent 调用**：社区成员讨论了 **AgentWorkflow** 中的**并行调用**，同时承认 Agent 通常按顺序运行，而工具调用可以是异步的。
  - 他们提议将“嵌套工作流（Nesting Workflows）”作为一种可能的技巧，以在多 Agent 流水线中实现**并行任务**。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **论坛功能优于 Discord 的快节奏**：一位用户称赞 **forum**（论坛）比节奏飞快的 **Discord** 具有更好的存档功能和更深入的讨论，并引用了 [general 频道的讨论](https://discord.com/channels/1087530497313357884/1098713601386233997/1331372409944801281)。
  
  - 他们建议利用论坛来避免向 **Modular** 员工提出的请求被淹没，并保持语言设计的清晰度，鼓励持续跨平台发布社区展示内容。
- **Nightly 版本悄然推进**：简要提到了 **Nightly** 版本正在活跃更新中，但未透露具体细节。
  
  - 对话将其列为未来更新的关注点，让社区对底层架构的**新变化**充满好奇。
- **Mojo 保留在 Modular.com**：成员们好奇 **Mojo** 是否会像 **Python** 那样采用 .org 域名，但 [#mojo 频道](https://discord.com/channels/1087530497313357884/1151418092052815884/1331382141585457212) 确认它仍将保留在 **modular.com** 下。
  
  - 他们强调 **Mojo** 不会从 **Modular** 中拆分出来，所有的努力都将统一在现有域名下。
- **MLIR 并行化优于 LLVM**：用户强调 **MLIR** 的并行化比 **LLVM** 更有前景，并提到了正在进行的使并行执行更接近现实的工作。
  
  - 他们将新实现视为**高性能**编译器的关键里程碑，尤其是目前 **LLVM** 仍在追赶。
- **Rust 处理工作窃取调度器**：对话探讨了 **Rust** 在工作窃取（work-stealing）调度器与线程安全之间的冲突，指出这限制了在 yield 点跨越使用 mutex。
  
  - 虽然复杂，但成员们主张对任务进行更细粒度的控制，认为尽管有开销，但用心的并发设计是值得的。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **SunoMusic 凭声音起飞**：他们推出了一项功能，允许用户录制自己的歌声或乐器演奏，如 [这条推文](https://x.com/SunoMusic/status/1881742789639057828) 所示。
  
  - 这一新功能将用户的创造力转化为从个人录音中生成完整的歌曲。
- **音频标注难题**：由于数据集有限，参与者发现衡量 **background noise**（背景噪声）和 **recording quality**（录音质量）非常棘手。
  
  - 他们建议在现有样本中添加合成噪声，并参考了 [audio_augmentations.ipynb](https://drive.google.com/file/d/1uhQ22wW7H5aCABYI1kfMtyaNVl9cUjWm/view)。
- **构建音频数据集**：一位贡献者维护着涵盖配音、视频剪辑 embedding 和科学知识图谱的开源数据集。
  
  - 他们面临资源限制，但对新合作者的加入持开放态度。
- **Bud-E 推出情感化 TTS**：他们展示了 **Bud-E** 的情感化文本转语音（TTS）路线图，强调了更具表现力的语音输出。
  
  - 分享的音频样本暗示了在合成用户意图方面具有更深层次的细微差别。
- **教师投身 AI 项目**：一位高中教师在处理教学任务的同时，还在扩展多个 AI 数据集项目。
  
  - 他们拒绝了工作邀约以保持独立性，并依靠志愿者力量推动音频和视频数据集的发展。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **DeepSeek 的 OpenAI Endpoint 探索**：成员们使用了 **OpenAI endpoint**，并参考 [DeepSeek](https://api.deepseek.com) 提供了代码示例，展示了如何为聊天请求定义自定义路由，并强调了灵活的 **prompt** 结构。
  
  - 他们强调了简化调用的好处，赞扬了这种方法能够在使用各种**文本生成框架**时进行敏捷的试错，并对进一步的改进表示了兴趣。
- **最佳模型热议：'Command-R7b' vs 'Aya Expanse 32B'**：一位用户询问了最受推崇的**文本生成**偏好，引发了对 **Command-R7b** 和 **Aya Expanse 32B** 的讨论，多位成员分享了使用经验。
  
  - 他们指出了不同的性能表现，指出某些用例更倾向于使用 **Command-R7b** 处理更重的逻辑任务，而另一些则更喜欢 **Aya Expanse 32B** 以应对更广阔的创意语境。
- **Cohere Command R+ 08-2024 流量增长**：成员们在聊天中重点介绍了新的 **Cohere Command R+ 08-2024** 模型，赞扬了其在 **Azure AI Foundry** 设置下扩展的文本生成能力。
  
  - 他们讨论了与 **LlamaIndex** 工作流的协同作用，期待最终过渡到 **Cohere API v2**，并继续分享使用心得。
- **模因梦想：Cohere 的 LCoT 模型权重**：爱好者们建议 **Cohere** 发布“LCoT 模因模型权重”，将喜剧线索与更深层次的推理相结合，并参考了用于喜剧扩展的**企业级解决方案**。
  
  - 其他人对可行性表示怀疑，考虑到 **Cohere** 的品牌定位，但也表示希望这能为新受众带来专门的喜剧文本生成。
- **从静态到动态：图像转视频 AI 计划**：一位用户展示了**图像转视频**生成的尝试，参考了多个 ML 框架，探索了跨领域转换，并寻求反馈。
  
  - 他们特别提到了向 3D 过渡的扩展，并对推动视觉生成工作流中的创意动力表示兴奋。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **1 月 27 日前的教学大纲惊喜**：即将到来的 MOOC 修订版教学大纲定于 1 月 27 日发布，将明确高级 **LLM Agents** 的内容。
  
  - 组织者指出最终审批仍在进行中，但学习者可以期待很快看到具体的模块细节。
- **塑造舞台的演讲者**：已注意到邀请梁文锋（Liang Wenfeng）的建议，尽管大多数客座演讲者的选择已经确定。
  
  - 计划推出一份新的反馈表，以收集热心参与者对未来选择的更多意见。
- **黑客松热潮暂缓**：目前尚未确认下一次活动的正式日期，一些人希望它能与春季计划保持一致。
  
  - 组织者暗示未来的研究合作可能会与即将举行的任何黑客松相结合，因此关注者应保持关注。
- **春季 MOOC 超越秋季**：新课程将在秋季内容的基础上进行构建，但不要求预先完成秋季课程，允许任何人加入。
  
  - 它扩展了基础的 **LLM Agent** 概念，通过更新的材料针对老学员和新学员。
- **展望未来的研究合作**：Song 教授正在评估对即将开展的小组研究项目的兴趣，以展示更广泛的 **LLM Agent** 进展。
  
  - 鼓励感兴趣的学生提及他们的领域或课题，以塑造可能出现的任何共同努力。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 的 DeepSeek R1 之谜**：社区成员询问哪些 **DeepSeek R1 蒸馏模型**可以在 GPT4All 上运行，并指出缺少类似 [LM Studio](https://lmstudio.ai/models) 的公开模型目录，暗示需要 **llama.cpp** 的更新。
  
  - 他们讨论了这些模型发布的具体时间表，交流了如何确认稳定集成到 GPT4All 的准备情况。
- **WordPress 聊天机器人困扰**：一位开发者试图在不使用插件的情况下将**聊天机器人接入 WordPress**，但在获取 **API Key** 方面遇到困难，引发了对官方指南的担忧。
  
  - 其他人提出了替代方案，但讨论结束时仍未就立即获取密钥给出具体解决方案。
- **寻求免费 ChatGPT 访问权限**：一位用户公开寻求 **ChatGPT 的免费、无限使用**，希望能有可行的变通方法或平台提供商的慷慨赠予。
  
  - 对话凸显了对免费 AI 解决方案的持续需求，但对于合法的免费密钥尚未达成共识。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **RAG 处理动态数据面临挑战**：一位用户询问 **基于 DSPy 的 RAG** 如何管理动态数据，强调了对其实时适应性的关注，但目前尚未提供明确的解决方案。
  
  - 其他人未对该查询做出解答，没有提供具体的代码示例或后续参考资料。
- **征集 DSPy 研究合作伙伴**：一位用户请求在 **DSPy 研究**方面进行合作，并强调了在 **AI for Good**、**LLMs** 和 **高等教育**领域的经验。
  
  - 他们表达了做出有意义贡献的强烈愿望，但未提供直接链接或资源。
- **LM Studio 在集成 DSPy 时遇到障碍**：一位用户报告在将 **LM Studio** 与 **DSPy** 搭配使用时遇到困难，并将其与使用 **Ollama** 时更顺畅的体验进行了对比。
  
  - 另一位成员询问 *“你是如何使用该模型的？”*，引发了关于本地环境设置和可能存在的兼容性陷阱的讨论。
- **Ollama 错误困扰**：在将 **DSPy** 与 **Ollama** 运行时出现了一个涉及 'str' 对象的错误，阻碍了数据摘要功能。
  
  - 这迫使 DSPy 在没有数据感知建议器（data-aware proposer）的情况下运行，引发了对功能缺失的担忧。
- **仓库垃圾信息吐槽**：一位用户抱怨仓库中的 **垃圾信息（spam）**，可能与某个代币（coin）相关问题有关，称其 *非常差劲*。
  
  - 他们认为这是对真正的 DSPy 讨论的一种令人厌恶的干扰，但尚未确定直接的解决办法。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **自定义 RLHF 重构获得支持**：成员们提议移除所有自定义损失函数，同时提供一个文档页面以便于添加，参考 [issue #2206](https://github.com/pytorch/torchtune/issues/2206)。
  
  - 他们还强调需要提供传递自定义前向传播函数的示例，以确保 **DPO 全量微调（full-finetuning）** 的兼容性。
- **Phi 4 PR 等待最终调整**：**Phi 4 PR** 因在合并前需要进行一些调整而受到关注，计划近期优化其设计。
  
  - 贡献者们表现出迅速解决这些问题的积极性，这与更广泛的 RLHF 增强计划保持一致。
- **SimPO 弃用重写路线图**：开发者宣布 **SimPO** 已弃用，以减少冗余并推进新的 RLHF 方案（recipes）。
  
  - 他们承诺更新相关文档，旨在对齐任务中实现更灵活的损失函数集成。
- **Nature Communications 论文发表引发热烈欢呼**：该论文在 [Nature Communications](https://www.nature.com/collections/ceiajcdbeb) 上的专题展示获得了社区的热烈反馈。
  
  - 它的被接收强调了持续的研究努力，引发了 *超级酷* 的反应和团队自豪感。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 1.0 预发布版与 Python 代码执行**：即将发布的 **OpenInterpreter 1.0** 引发了关于移除 **Python 代码执行** 的猜测，成员们注意到 **Python** 在 1.0 预发布版中似乎未完全实现。
  
  - 一位用户询问该功能是否会在稍后回归，反映出对编码功能 **未来更新** 的不确定性。
- **Markdown 和 TXT 是显示格式**：一位用户澄清说，**Markdown** 和 **TXT** 文件是作为文本格式化机制，而非编程语言。
  
  - 随后的评论暗示，可能正在开发某些功能以处理 **OpenInterpreter** 中的格式化行为。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla 模型获取指南**：一位成员询问 [Ollama 文档](https://ollama.com/adrienbrault/gorilla-openfunctions-v2:Q6_K/blobs/8f6765ab9969) 中的 **Gorilla 模型** 是否正确，参考了 Hugging Face 上的 **adrienbrault/gorilla-openfunctions-v2:Q6_K/model**。
  
  - 他们标记了其他人以确认构建函数调用（function-calling）LLM 的 **正确引用**。
- **LLaMA v2 扩展能力**：**LLaMA v2** 包含 **4096** 上下文长度、**30** 块数量和 **32** 注意力头，以提供更大的容量。
  
  - 参与者强调了 **tokenizer 设置** 和 **量化版本**，强调了这些高级用法的细节。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 获得 Windows 适配**：一位贡献者提交了 [Windows 测试的 pull request](https://github.com/tinygrad/tinygrad/pull/8715)，并询问 **LLVM** 或 **Clang** 是否可以驱动正式的 Windows 后端。
  - 他们提供了一张显示 PR 详情的图片，并引发了关于如何确保在多种 Windows 配置下实现广泛测试覆盖的讨论。
- **OpenCL GPU 支持成为优先级**：另一位成员强调 **GPU (OpenCL)** 支持是 Windows 测试的关键补充，理由是其在不同硬件上的性能提升。
  - 这一讨论突显了在 Windows 环境下优化 **Tinygrad** GPU 性能并完善跨平台兼容性的努力。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **Liger Kernel 获得 KTO 支持**：[Liger Kernel 仓库](https://github.com/linkedin/Liger-Kernel/pull/475)正式合并了 **KTO loss**，标志着旨在增强性能的一次重大更新。
  - 贡献者们对此表示庆祝，并强调 **KTO loss** 可以为更好地集成即将到来的模型更新提供桥梁。
- **利用 KTO 优势进行模型合并**：工程师们讨论了可能受益于基于 KTO 指标的**模型合并策略**，并引用了 Liger Kernel 中新加入的功能。
  - 他们强调了积极的协作，并期望该损失函数能简化未来版本中的合并工作。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Local-First X AI 黑客松在旧金山启动**：组织者宣布将于 **2 月 22 日**在**旧金山**举办 [Local-First X AI Hackathon](https://www.lofihack.com/)，重点关注离线友好的开发策略。
  - 他们期待现场气氛热烈，并鼓励参与者在活动日期前就先进的 Local-First AI 进行头脑风暴。
- **加入黑客松对话线程**：一个专门的[讨论线程](https://discord.com/channels/1089876418936180786/1329529625189154826)现已开放，用于对项目构思和物流进行**更深入的讨论**。
  - 组织者强调请尽快报名，并提到欢迎关于 Local-First 方法的新建议。

---

**MLOps @Chipro Discord** 没有新消息。如果该社区长时间没有活动，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该社区长时间没有活动，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长时间没有活动，请告知我们，我们将将其移除。

---

# PART 2: 渠道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1331353406958473269) (266 条消息🔥🔥):

> `Model Quantization, Fine-tuning Models, DeepSeek R1 Model Support, Chat Templates and Thinking Tags, Dynamic 4-bit Quantization`

- **关于模型量化 (Model Quantization) 技术的讨论**：用户讨论了各种量化方法的有效性，其中 8-bit 量化因其在性能和准确性之间的权衡而受到关注，同时讨论了 BnB 4-bit 的动态量化能力。
  - 有观点指出，虽然 BnB 8-bit 的支持并不完善，但 FP8 和 Q8_0 等替代方法提供了更好的灵活性和性能。
- **使用 Unsloth 进行微调 (Fine-tuning)**：一位用户询问了如何使用聊天数据对 Unsloth 模型进行微调，表示模型理想情况下应该生成连续的消息，而不是单个回复。
  - 建议包括确保高质量的数据集示例，并考虑使用另一个模型来监督和审查输出。
- **DeepSeek R1 兼容性**：关于在 llama-server 中使用 DeepSeek R1 模型的澄清显示，用户需要确保加载完整的模型而不是其中的一部分，才能成功部署。
  - 讨论还指出，VLLM 已广泛支持运行该模型，包括大型版本。
- **聊天模板与 “Thinking” 标签**：用户探讨了 DeepSeek 聊天模板中缺乏统一的 “thinking” 标签来进行行为训练的问题，并注意到了响应中由此产生的 COT (Chain of Thought) 方面。
  - 建议要么合并此类模板，要么在训练期间完全避免使用特定模板。
- **引入动态 4-bit 量化 (Dynamic 4-bit Quantization)**：引入了 unsloth-bnb 量化方法，这是一种动态方法，旨在显著减小模型大小的同时保持准确性。
  - 强调该方法仅略微增加 VRAM 使用量，同时有效地优化了模型性能。

**提到的链接**：

- [Unsloth - Dynamic 4-bit Quantization](https://unsloth.ai/blog/dynamic-4bit)：Unsloth 的动态 4-bit 量化选择性地避免对某些参数进行量化。这在显著提高准确性的同时，保持了与 BnB 4bit 相似的 VRAM 占用。
- [来自 Fimbul (@fimbulvntr) 的推文](https://x.com/fimbulvntr/status/1881821582571761920)：是我疯了还是 DeepSeek-R1 的 model_max_length 被限制在 16384 了？我认为这是一个 bug。实际上它应该是 163840。它的 original_max_position_embeddings=4096 且 RoPE factor 为 40... 4...
- [Google Colab](https://colab.research.google.com)：未找到描述
- [Google Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb)：未找到描述
- [tokenizer_config.json · deepseek-ai/DeepSeek-R1 at 3302ba78c0090838341caf8adfbe1e231308fa95](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/3302ba78c0090838341caf8adfbe1e231308fa95/tokenizer_config.json#L22)：未找到描述
- [So Cute Cat GIF - So Cute Cat Love - Discover & Share GIFs](https://tenor.com/view/so-cute-cat-love-head-pat-gif-14623443)：点击查看 GIF
- [facebook/layerskip-llama3-8B · Hugging Face](https://huggingface.co/facebook/layerskip-llama3-8B)：未找到描述
- [bitsandbytes](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)：未找到描述
- [Datasets 101 | Unsloth Documentation](https://docs.unsloth.ai/basics/datasets-101)：学习创建微调数据集的所有要点！
- [unsloth/DeepSeek-R1-GGUF · Hugging Face](https://huggingface.co/unsloth/DeepSeek-R1-GGUF)：未找到描述
- [GitHub - bagel-org/ZKLoRA: Efficient Zero-Knowledge Proofs for LoRA Verification](https://github.com/bagel-org/ZKLoRA)：用于 LoRA 验证的高效零知识证明 - bagel-org/ZKLoRA
- [tokenizer_config.json · unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit at main](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit/blob/main/tokenizer_config.json)：未找到描述
- [👨‍👨‍👧‍👧 GRPO by qgallouedec · Pull Request #2565 · huggingface/trl](https://github.com/huggingface/trl/pull/2565)：这个 PR 做了什么？
```python
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
# Load the dataset
dataset = load_dataset("trl-lib/tldr", spli....
```

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1331361373250129952) (6 条消息):

> `Medium 上的 Unsloth 训练，Weights & Biases ETA 追踪，ETA 自定义代码，微调挑战`

- **Medium 上的 Unsloth 微调探索**：一篇 Medium 文章讨论了[使用 Unsloth 进行 LLM 微调](https://gautam75.medium.com/fine-tuning-llama-3-1-8b-for-function-calling-using-lora-159b9ee66060)，并结合 Weights & Biases 进行监控以及 vLLM 进行模型服务，解决了针对特定任务进行适配的需求。
  
  - 文章强调了 LLM 微调中的挑战，特别是关于 GPU 显存和计算时间的问题。
- **Weights & Biases 缺乏 ETA 追踪**：一位用户对 Weights & Biases 不显示训练过程的预计完成时间 (ETA) 表示失望，该工具仅显示运行时间和开始时间。
  
  - 他们得到了确认，Weights & Biases 并不追踪时间，而是专注于绘制训练指标图表。
- **ETA 需要自定义代码**：一位社区成员澄清说，虽然 Weights & Biases 没有内置的 ETA 追踪功能，但可以通过编写自定义代码来实现。
  
  - 这为想要更精确监控训练完成时间的用户提供了一种变通方案。

 

**提到的链接**：[Fine-Tuning Llama-3.1-8B for Function Calling using LoRA](https://gautam75.medium.com/fine-tuning-llama-3-1-8b-for-function-calling-using-lora-159b9ee66060)：利用 Unsloth 进行微调，集成 Weights & Biases 进行监控，并使用 vLLM 进行模型服务。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1331365123821928459) (146 条消息🔥🔥):

> `Phi-4 model issues, Running DPO training script, Unsloth notebooks updates, Using Triton with Unsloth, Fine-tuning suggestions for different models`

- **Phi-4 模型微调的挑战**：用户报告了在微调 **Phi-4** 模型时遇到的困难，特别是关于持续生成低质量回复以及对输出选项的困惑。
  
  - 有建议认为这些问题可能是特定于该模型的，因为其他 notebook 在不同模型上运行正常。
- **DPO 训练脚本导入错误**：一位用户在运行 DPO 训练脚本时遇到了 ImportError，原因是 Triton 库中的相对导入问题，建议改用绝对导入。
  
  - 另一位用户遇到了与编译 llama.cpp 相关的 RuntimeError，建议检查更新和依赖项。
- **Unsloth Notebooks 的更新与修复**：多位用户对 Unsloth notebooks 的近期更新发表了评论，包括修复了与模块导入和内存管理问题相关的错误。
  
  - 确认这些 notebook 旨在实现可复现性，专注于特定模型以避免训练期间出现无关错误。
- **CPU 微调挑战**：用户讨论了由于性能缓慢和内存限制而在 CPU 上进行微调的困难，特别是对于较大的模型。
  
  - 建议包括降低 batch sizes 或为特定模型使用专门的模板以优化资源利用。
- **成功微调的调整建议**：用户分享了关于优化 LoRA 设置和 batch sizes 的见解，以提高不同模型的微调性能。
  
  - 建议包括确保正确的数据集格式，并使用已有的 benchmarks 或模板来指导适配。

**提到的链接**：

- [Google Colab](https://colab.research.google.com): 未找到描述
- [Google Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb): 未找到描述
- [Quazim0t0/Phi4.React.Turn.V2.Full · Hugging Face](https://huggingface.co/Quazim0t0/Phi4.React.Turn.V2.Full): 未找到描述
- [GitHub · Build and ship software on a single, collaborative platform](https://github.co): 加入全球应用最广泛的 AI 驱动型开发者平台，数百万开发者、企业和最大的开源社区在此构建推动人类进步的软件。
- [Quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu): 未找到描述
- [Unsloth Notebooks | Unsloth Documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks): 以下是我们所有 notebook 的列表：
- [unsloth/unsloth/kernels/fast_lora.py at d802bbf4e298cb0da1e976ab9670fbc1cbe3514c · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/d802bbf4e298cb0da1e976ab9670fbc1cbe3514c/unsloth/kernels/fast_lora.py#L201)): 微调 Llama 3.3, Mistral, Phi-4, Qwen 2.5 & Gemma LLM，速度提升 2-5 倍，显存占用减少 70% - unslothai/unsloth
- [trl/trl/scripts/dpo.py at a9b54a852ee12ff508773edb02e1c243817e71ae · huggingface/trl](https://github.com/huggingface/trl/blob/a9b54a852ee12ff508773edb02e1c243817e71ae/trl/scripts/dpo.py#L17): 使用强化学习训练 transformer 语言模型。 - huggingface/trl
- [unsloth/unsloth/models/dpo.py at main · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/main/unsloth/models/dpo.py): 微调 Llama 3.3, Mistral, Phi-4, Qwen 2.5 & Gemma LLM，速度提升 2-5 倍，显存占用减少 70% - unslothai/unsloth
- [GitHub - ggerganov/llama.cpp: LLM inference in C/C++](https://github.com/ggerganov/llama.cpp): C/C++ 实现的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1331369580941283328) (191 条消息🔥🔥):

> `Synthetic Data Training, Chinchilla Optimal Models, Emotion Tracking in AI, Agentic AI Claims, Dynamic vs. Static Learning Systems`

- **用于训练的 Synthetic Data**：讨论展开了关于使用 *无限合成数据流 (infinite synthetic data streams)* 训练 AI 的潜力，认为这可以通过围绕评估合规性进行微调来优化训练效率。
  
  - *讨论指出*，在使用合成数据集时，适当的数据清洗 (data curation) 对于避免“垃圾进，垃圾出 (garbage in, garbage out)”至关重要。
- **理解 Chinchilla 最优性**：参与者讨论了 *Chinchilla models*，其目标是在模型大小和训练 Token 数量之间找到平衡，以优化性能。
  
  - *会议强调*，现有模型已远超最优阈值，经验证据显示现代架构在 Scaling 效率方面取得了巨大进步。
- **情感追踪的挑战**：一位成员分享了在成人行业开发 Bot 的经验，强调了 Bot 回复中 *情感传播 (emotion propagation)* 和心理学原则的重要性。
  
  - 对话触及了由于人类互动的动态特性，有效实现情感追踪的复杂性。
- **对 Agentic AI 的怀疑**：大家达成共识，认为许多关于 *自主 Agent* 的说法被夸大了，往往是将过时的概念包装成新技术进行营销。
  
  - 成员们对行业推销 Agentic AI 理念但仅交付基础脚本功能表示失望。
- **AI 在教育领域的潜力**：小组展望了 AI 的未来，设想了 LLM 通过互动媒介教小学生（尤其是数学）的可能性。
  
  - 同时也提出了担忧，即在不夸大 AI 能力的情况下，这种学习方式是否能在当前框架内有效实施。

**提到的链接**：

- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)：语言模型 (LMs) 是自然语言处理的强大工具，但当模型较小时，往往难以生成连贯且流畅的文本。拥有约 125M 参数的模型（如 GP...）
- [Papers with Code - Chinchilla Explained](https://paperswithcode.com/method/chinchilla)：Chinchilla 是一个拥有 70B 参数的模型，作为计算最优模型，使用了 1.4 万亿个 Token 进行训练。研究结果表明，这类模型通过等比例缩放模型大小和...

---

### **Codeium (Windsurf) ▷ #**[**discussion**](https://discord.com/channels/1027685395649015980/1027697446446432336/1331380577978286132) (48 条消息🔥):

> `Codeium Extension Updates, Windsurf IDE Issues, Model API Integration, Diff Viewer Difficulties, Privacy Policy Queries`

- **Codeium 扩展缺少 Web 搜索**：一位用户询问 Codeium 扩展何时能获得类似于 Windsurf 的 Web 搜索功能，并指出了目前功能的局限性。
  
  - 另一位成员表示，尽管 VS Code 有其优势，但许多 JetBrains IDE 用户认为其 UX 不足。
- **Windsurf IDE 中的访问问题**：多位用户报告在 Windsurf IDE 最近一次升级后，无法访问 FastAPI 文件（如 main.py 和 server.py）。
  
  - 一位用户提到，Cascade 编程助手在 Linux Ubuntu 系统上无法再访问这些文件。
- **对自定义 API 集成的兴趣**：一位用户建议，允许开发者将自己的 API 与 Flow Actions 结合使用将增强该平台的功能。
  
  - 另一位参与者补充说，除了当前的订阅模式外，通过标准 API 集成任何 LLM 也是一个有益的选择。
- **Writemode 功能定价**：一位成员询问 writemode 功能是否免费，另一位成员澄清说这是一项付费功能。
  
  - 该澄清强调了平台内免费功能与高级功能之间的区别。
- **Diff 查看器困惑**：一位用户表示难以理解 Diff 查看器的配色方案，并寻求更改方法。
  
  - 此外，他们还注意到在 Windsurf 中使用 VIM 时，插入模式下的光标不会发生变化。

**提到的链接**：

- `Changelog | Visual Studio Marketplace`
  
  : 未找到描述
- [Privacy Policy | Windsurf Editor and Codeium extensions](https://codeium.com/privacy-policy)：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。
- [Warp: The intelligent terminal](https://www.warp.dev/)：Warp 是一款内置 AI 和开发团队知识库的智能终端。现已支持 MacOS 和 Linux。
- [JetBrains meets Codeium Chat — level up your coding](https://youtu.be/99_UBBAfk0c)：Codeium Chat 现已在 IntelliJ、WebStorm、PyCharm、AndroidStudio 等 JetBrains IDE 上免费提供！Codeium 用户可以使用 AI 助手...

---

### **Codeium (Windsurf) ▷ #**[**windsurf**](https://discord.com/channels/1027685395649015980/1306163501286293515/1331356738691006495) (426 条消息🔥🔥🔥):

> `Windsurf 的自动生成记忆功能、Cascade 的开发挑战、Prompt Engineering 与上下文、编程中的 AI 集成、Windsurf 用户体验`

- **Windsurf 的自动生成记忆（Auto-Generated Memories）功能**：用户对 Windsurf 的自动生成记忆功能表示兴奋，该功能似乎能够捕捉项目内重要的对话上下文。
  
  - 然而，对于这些记忆的结构化方式以及在不同 workspace 之间切换时的有效性存在担忧。
- **使用 Cascade 进行开发的挑战**：几位用户提到在使用 Cascade 时遇到问题，理由包括 Prompt 循环和重复的修复尝试导致挫败感。
  
  - 让 Cascade 参与到特定代码问题的讨论中通常有助于解决这些循环场景。
- **Prompt Engineering 和上下文在 AI 中的重要性**：有人指出，Prompt 技术会显著影响从 AI 获得的响应，强调了结构化请求的必要性。
  
  - 讨论参与者建议，改进 AI 的记忆能力可以缓解开发者在与其交互时面临的常见问题。
- **未来用户工作流中的 AI 集成**：参与者展望了这样一个未来：像 Windsurf 这样的 AI 工具可以在各种应用和设备之间提供更无缝的上下文集成。
  
  - 这种理想场景将提高生产力，并减少对 Prompt Engineering 和上下文管理的依赖。
- **Windsurf 的用户体验与编码实践**：用户分享了他们使用 Windsurf 的经验，包括其在提高编码效率和自动化 Git 历史检查等任务方面的潜力。
  
  - 同时也提出了对工具稳定性和 auto-complete 等功能性能的担忧。

**提到的链接**：

- [The 70% problem: Hard truths about AI-assisted coding](https://addyo.substack.com/p/the-70-problem-hard-truths-about)：一份实地指南，以及为什么我们需要重新审视我们的预期。
- [Web Search - Codeium Docs](https://docs.codeium.com/windsurf/web-search)：未找到描述。
- [D-Wave Leap Log In | D-Wave Leap™](https://cloud.dwavesys.com/leap/)：未找到描述。
- [Support | Windsurf Editor and Codeium extensions](https://codeium.com/support)：需要帮助？联系我们的支持团队以获取个性化协助。
- [Why AI Engineers Need Planning More Than Perfect Prompts - Cline Blog](https://cline.bot/blog/why-ai-engineers-need-planning-more-than-perfect-prompts-2)：未找到描述。
- [Post Not Found](https://cline.bot/blog/why-ai-engineers-need-planning-more-than-perfect-prompts-2>)：未找到描述。
- [Windsurf forked VS Code to compete with Cursor. Talking the future of AI + Coding](https://www.youtube.com/watch?v=ptekg6GNzIQ)：Wes 和 Scott 与来自 Windsurf 的 Kevin Hou 和 Varun Mohan 讨论了 AI 在编码领域不断演变的格局以及软件开发的未来。
- [Web Search Best Practices: Save Credits and Optimize Your Workflow - Windsurf Editor](https://www.youtube.com/watch?v=moIySJ4d0UY)：准备好充分利用 Windsurf 全新的 Web Search 功能了吗？这次深度探讨将帮助你释放其全部潜力！
- [Persistent, intelligent project memory](https://forum.cursor.com/t/persistent-intelligent-project-memory/39109)：.cursorrules 只是权宜之计。我们真正需要的是 Cursor 能够真实地记住与用户的交互以及项目的需求，并随着用户与 Cursor 的交互自动更新这些记忆。
- [GitHub - kinopeee/windsurfrules](https://github.com/kinopeee/windsurfrules)：通过在 GitHub 上创建账号来为 windsurfrules 的开发做出贡献。

---

### **LM Studio ▷ #**[**announcements**](https://discord.com/channels/1110598183144399058/1111797717639901324/1331457978661998622) (1 条消息):

> `LM Studio 0.3.8, Thinking UI, LaTeX rendering, Bug fixes`

- **LM Studio 0.3.8 发布，带来令人兴奋的新特性**：最新版本 **LM Studio 0.3.8** 为 **DeepSeek R1** 引入了 **Thinking UI**，并新增了对 `ext{...}` 块中 **LaTeX** 渲染的支持。
  
  - 可通过应用内更新获取，或从官网下载；[详细演示](https://x.com/lmstudio/status/1881849443999416802)展示了这些增强功能。
- **增强的 LM Studio LaTeX 渲染**：通过此次更新，用户现在可以使用 **LaTeX** 有效地渲染数学表达式，提升了技术内容创作的可用性。
  
  - 数学公式可以包裹在 `begin{equation}... ext{}` 中或通过 `ext{...}` 行内显示，从而简化了数学交流。
- **一系列 Bug 修复确保运行顺畅**：**0.3.8** 版本解决了多个 Bug，包括 Windows 安装程序中 **LM Runtimes** 的错误捆绑，以及显示重复 **Vulkan GPUs** 的问题。
  
  - 此外，它还解决了旧聊天记录中消息无法显示的问题，确保了更可靠的用户体验。

 

**提到的链接**：[来自 LM Studio (@lmstudio) 的推文](https://x.com/lmstudio/status/1881849443999416802)：LM Studio 0.3.8 🚢 - 针对 DeepSeek R1 的 Thinking UI - LaTeX 渲染改进 - Bug 修复

 

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1331358386427527251) (209 条消息🔥🔥):

> `DeepSeek R1 performance, Model loading issues on Mac, Quantization settings, Using LMStudio effectively, Math problem-solving capabilities of models`

- **DeepSeek R1 在数学问题上表现出色**：R1 Distill Qwen 32B 模型被认为是解决复杂竞赛数学题的最佳本地模型，其表现可能足以与 AIME 考试的顶尖考生竞争。
  
  - 用户注意到其性能优于 Llama 405B 和 DeepSeek V3 671B。
- **Mac M1 Max 上的模型加载困难**：一位用户报告在 MacBook Pro M1 Max 上加载 DeepSeek 32B 模型时遇到困难，出现了与模型词汇表（vocabulary）相关的错误。
  
  - 他们寻求关于访问运行时（runtimes）和更新 llama.cpp 以解决该问题的帮助。
- **对 LMStudio 设置的困惑**：有关于 Apple Silicon 上 LMStudio 中“Keep Model in Memory”设置的咨询，其对 RAM 占用的影响结果不一。
  
  - 有建议认为，由于统一内存架构（unified memory architecture），这些设置可能不会产生显著差异。
- **量化性能讨论**：用户讨论了量化设置（尤其是 Q4）如何影响模型性能和内存占用。
  
  - 根据可用 RAM 和所需的推理速度推荐了不同的模型。
- **AI 学习与使用体验的提升**：多位用户对使用 LMStudio 取得的进展表示满意，特别是在 AI 学习和回答特定查询方面。
  
  - 用户对未来可能使 AI 模型更加轻量化的发展感到兴奋。

**提到的链接**：

- [来自 Yorkie (@heyitsyorkie) 的推文](https://x.com/heyitsyorkie/status/1882042982465261967?s=46)：@levelsio @levelsio 你需要更新你的 LM Studio 版本：https://lmstudio.ai/0.2.31 已经过时了，0.3.* 版本进行了彻底的重构。
- [来自 thebes (@voooooogel) 的推文](https://x.com/voooooogel/status/1881966969043464365)：制作了一个非常简单的基于采样器的杠杆，试图在 R1 上模拟 o1 风格的 "reasoning_effort=high"：如果在生成足够的 thinking tokens 之前出现了 `</think>`，采样器会替换...
- [GitHub - deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#distilled-model-evaluation)：通过在 GitHub 上创建账户来为 deepseek-ai/DeepSeek-R1 的开发做出贡献。

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1331358934539173899) (180 messages🔥🔥):

> `DeepSeek R1 Pricing, MacBook Performance for LLMs, GPU Quantization, NVIDIA Digits Requirements, Model Size Efficiency`

- **了解运行 DeepSeek R1 模型的成本**：运行完整的 **DeepSeek R1 671B model** 对 NVIDIA 硬件的需求意味着巨额投资，据估计所需硬件成本约为 **$18,000**。
  
  - 相反，像 **4xA4000** 这样的选项可以为运行模型提供一种兼顾成本的高效解决方案，这是一种极具创新性的替代方案。
- **MacBook Air vs Pro 用于 LLMs**：讨论重点在于 **M4 MacBook Air** 与 **M2 MacBook Air** 的内存带宽对比，权衡其是否值得为 LLM 功能进行升级。
  
  - 同时也引发了关于 Air 处理热限制（thermal limits）能力的担忧，相比之下 **MacBook Pro** 机型表现更为强劲。
- **GPU Quantization 适配模型**：用户指出，在 **M3 Max MacBook Pro** 上运行 **R1 32B model** 如果不使用 **4-bit Quantization** 可能会面临挑战，而这可能会损害模型质量。
  
  - 分享的技巧包括使用针对 Apple silicon 优化的 **MLX versions**，以在管理内存需求的同时提升性能。
- **管理 GPU VRAM 以获得最佳性能**：关于运行具有更高 VRAM 需求模型的讨论，强调使用同一张显卡进行 Inference 可能会导致运行期间的 VRAM 争用。
  
  - 讨论强调为了获得最佳配置，建议将 Inference 工作负载与显示图形分离，特别是在 **Windows** 系统上。
- **对大规模模型的批评**：参与者辩论了训练和运行像 **DeepSeek R1** 这样超大规模模型的实用性，建议更紧凑、高效的模型可能会产生更好的结果。
  
  - 有人对效率表示担忧，并指出更大的模型并不本质上保证性能的提升，鼓励关注有针对性的高质量模型。

 

**Link mentioned**: [bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF · Hugging Face](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF): no description found

 

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1331373268648591370) (295 条消息🔥🔥):

> `AI 模型开发、芯片制造竞争、区块链与加密货币、DeepSeek 推理提取、Project Stargate 融资`

- **Project Stargate 模糊的融资情况**：讨论集中在提议的 5000 亿美元 Project Stargate 融资上，鉴于 SoftBank 的财务状况和资本限制，人们对其可行性表示怀疑。
  
  - 分析师质疑，在 SoftBank 的股权和现金流没有重大变化的情况下，如此大规模投资的可行性。
- **芯片制造格局**：对话强调了芯片制造面临的挑战，特别是对 TSMC 的依赖，以及 Samsung 和 Intel 等竞争对手的产能问题。
  
  - 参与者担心，如果没有大量的投资和时间，其他芯片制造商是否能有效竞争。
- **区块链与加密货币的影响**：参与者辩论了去中心化金融系统的潜力，重点讨论了政府控制的加强如何可能激发对加密货币的兴趣。
  
  - 尽管有潜在好处，但面对监管压力和政府禁止加密货币的能力，人们对加密货币普及的实用性提出了担忧。
- **关于 DeepSeek 推理提取的见解**：讨论了 DeepSeek 提取推理过程的能力，重点在于其对 Claude 和 O1 等模型的潜在益处。
  
  - 人们对使用推理提取不仅是为了节省成本，而且是为了将更深层的推理能力集成到其他模型中感到好奇。
- **AI 开发的未来**：参与者推测了 AI 模型的未来演变，特别是预期在 R1 输出上训练的 DeepSeek v3 的改进。
  
  - 对话指向了随着各种模型的推理过程被集成和完善，性能将得到提升的希望。

**提到的链接**：

- [Bloomberg - Are you a robot?](https://www.bloomberg.com/news/articles/2025-01-22/google-invests-another-1-billion-in-ai-developer-anthropic)：未找到描述
- [Tweet from Ivanka Trump (@IvankaTrump)](https://x.com/IvankaTrump/status/1839002887600370145)：Leopold Aschenbrenner 的 SITUATIONAL AWARENESS 预测我们正处于 2027 年实现通用人工智能 (AGI) 的轨道上，随后不久将实现超智能，带来变革性的机遇...
- [Tweet from Demis Hassabis (@demishassabis)](https://x.com/demishassabis/status/1881844417746632910)：我们对 Gemini 2.0 Flash Thinking 模型的最新更新（在此可用：https://goo.gle/4jsCqZC）在 AIME（数学）和 GPQA Diamond（科学）基准测试中获得了 73.3% 和 74.2% 的分数。感谢大家的所有反馈...
- [How blockchains could change the world](https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/how-blockchains-could-change-the-world)：忽略 Bitcoin 的挑战。在这次采访中，Don Tapscott 解释了为什么作为加密货币底层技术的区块链具有彻底改变世界经济的潜力。
- [Joseph717171/DeepSeek-R1-Distill-Llama-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF · Hugging Face](https://huggingface.co/Joseph717171/DeepSeek-R1-Distill-Llama-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF)：未找到描述
- [Tweet from Sun 乌龟 💖 (@suntzoogway)](https://x.com/suntzoogway/status/1882121235762721063)：伙计们，这是我写的一个恶搞！只是试图通过 hyperstition 创造一个美好的未来（我被困在欧盟的监管地狱里）
- [Tweet from Teknium (e/λ) (@Teknium1)](https://fxtwitter.com/Teknium1/status/1882159710180376828)：我正在为 Nous Research 的 post training 团队寻找两名工程师，以构建具有通用能力的模型的未来，探索具有认知能力、创造力的模型，并推进最先进的推理和...
- [Tweet from Pietro Schirano (@skirano)](https://x.com/skirano/status/1881854481304047656?s=46)：顺便说一句，你可以从 deepseek-reasoner 中只提取推理部分，这意味着你可以在其他模型回答之前，将该思考过程发送给任何你想要的模型。就像这里我把 gpt-3.5 turbo 变成了...
- [Tweet from Gavin Baker (@GavinSBaker)](https://x.com/gavinsbaker/status/1882081746877063677?s=46)：Stargate 是个好名字，但 5000 亿美元是个荒谬的数字，除非 SoftBank 打算卖掉他们所有的 BABA 和 ARM，否则没人会当真。SoftBank 拥有 380 亿美元现金，1420 亿美元债务和...
- [Tweet from Elon Musk (@elonmusk)](https://x.com/elonmusk/status/1881923570458304780)：@OpenAI 他们实际上并没有那笔钱

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1331397727808127021) (6 条消息):

> `模型激活的机械解释 (Mechanical Interpretation), DeepSeek 能力, 合成数据生成 (Synthetic Data Generation)`

- **成员探索机械解释技术**：一位成员正在为其关于**机械解释 (mechanical interpretation)** 的业余项目寻求建议，探讨在给定大量数据输入的情况下，如何可视化模型激活在各层之间的演变。
  
  - 另一位成员提到认识有此经验的人，并承诺在查阅旧消息后进行引荐。
- **DeepSeek 实验显示响应的多样性**：对 **DeepSeek** 的实验得出了有趣的结果，它能正确计算响应，但对其关于 'razzberry' 拼写变体的结论表示怀疑。
  
  - 输出结果指出 'razzberry' 中有 **0 个 p**，但同时也指出了可能与 'raspberry' 混淆，展示了该工具的内部推理过程。
- **咨询合成数据生成指南**：一位成员正在咨询有关生成用于模型微调 (finetuning) 的合成数据 (synthetic data) 的**准则与禁忌 (do's and don'ts)** 的资源。
  
  - 这反映了社区对合成数据使用最佳实践的持续关注。

**提到的链接**：[LLM Fan (@llm_fan) 的推文](https://x.com/llm_fan/status/1882139500153012423)：我认为 LLM 正在掌握 strawberry 中有几个 'r' 的问题。我问 DeepSeek “razzberry 的正确拼写中有几个 p？”（我也尝试了 'proper word' ...）

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1331374053310595133) (5 条消息):

> `FLAME 模型, 人机表示对齐 (Human-AI representation alignment)`

- **FLAME：小而强大的模型**：该论文介绍了 **FLAME**，这是一个专门在 Excel 公式上训练的基于 Transformer 的模型，仅凭 **60M 参数** 就实现了与 Davinci 等大型模型相当的竞争性能。
  
  - 该模型通过草图去重 (sketch deduplication) 实现了巧妙的训练数据集策划，并引入了 Excel 专用的公式分词器 (tokenizer)。
- **人类与 AI 表示的收敛**：最近的一项研究表明，**高性能 ANN** 和**生物大脑**都收敛于相似的表示，这暗示了一种通用的结构对齐。
  
  - 通过识别改变模型间表示一致性的刺激，该研究为 ANN 如何映射生物表示和计算提供了见解。

**提到的链接**：

- [FLAME: A small language model for spreadsheet formulas](https://arxiv.org/abs/2301.13779)：电子表格是最终用户数据管理的重要工具。在这些环境中使用大型语言模型辅助公式编写可能很困难，因为这些模型的训练成本昂贵……
- [Universality of representation in biological and artificial neural networks](https://www.biorxiv.org/content/10.1101/2024.12.26.629294v1)：未找到描述内容

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1331402337960460390) (12 messages🔥):

> `LLM 自动架构搜索，EvaByte 无分词器模型，机器学习中的张量网络`

- **自动架构搜索引发关注**：一篇关于 [LLM 自动架构搜索](https://arxiv.org/abs/2411.17800) 的论文讨论了一种结合分层数值编码与进化算法来优化模型架构的新方法。
  
  - 虽然有人怀疑其作为竞争优势的独特性，但研究结果表明，在优化深度学习框架方面取得了显著进展。
- **认识 EvaByte：无分词器的奇迹**：EvaByte 是香港大学与 SambaNova Systems 合作的成果，推出了一款 **6.5B 字节级语言模型**，其性能可与现代基于 Tokenizer 的 LM 媲美，且使用的训练数据减少了 **5 倍**，解码速度提升了 **2 倍**。
  
  - 它完全基于每字节（per-byte）运行，提供了极大的灵活性并提升了在各种任务上的性能，尽管存在关于不规则结构影响硬件利用率的担忧。
- **张量网络在机器学习中掀起波澜**：一个名为 [TensorGrad](https://github.com/thomasahle/tensorgrad) 的新机器学习库利用命名边（named edges）代替数字索引，简化了 **2D 卷积**等操作，同时进行符号推理和优化。
  
  - 它具有 Pass 之间的公共子表达式消除等特性，能生成高效的编译后 torch 代码，并需要用户反馈以增强易用性。

**提到的链接**：

- [EvaByte: Efficient Byte-level Language Models at Scale | HKU NLP Group](https://hkunlp.github.io/blog/2025/evabyte/)：未找到描述
- [STAR: Synthesis of Tailored Architectures](https://arxiv.org/abs/2411.17800)：模型架构的迭代改进是深度学习的基础：Transformer 首先实现了规模化，而最近在模型混合方面的进展推向了质量与效率的前沿……
- [Lin Zheng (@linzhengisme) 的推文](https://x.com/linzhengisme/status/1881913052037329219)：🚀 认识 EvaByte：最强的开源无分词器语言模型！我们的 6.5B 字节级 LM 以 5 倍更少的数据和 2 倍更快的解码速度匹配了现代基于 Tokenizer 的 LM，并能自然扩展到多模态任务……
- [GitHub - thomasahle/tensorgrad: Tensor Network Library with Autograd](https://github.com/thomasahle/tensorgrad)：带有 Autograd 的张量网络库。通过在 GitHub 上创建账户为 thomasahle/tensorgrad 的开发做出贡献。
- [evabyte/evabyte_hf/eva.py at ba8f65c5fe502b7ed07f916773754734b91b52fd · OpenEvaByte/evabyte](https://github.com/OpenEvaByte/evabyte/blob/ba8f65c5fe502b7ed07f916773754734b91b52fd/evabyte_hf/eva.py#L63)：EvaByte: 大规模高效字节级语言模型 - OpenEvaByte/evabyte

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1331374053310595133) (5 messages):

> `FLAME 模型，人类与 AI 表征相似性，小模型的优势`

- **FLAME：用于 Excel 公式的小模型**：论文介绍了 **FLAME**，这是一个专门在 Excel 公式上训练的 Transformer 模型，包含 **60M 参数**，以显著更少的数据实现了极具竞争力的性能。
  
  - *通过使用 Excel 特定的分词方式和预训练目标*，FLAME 在公式修复和补全等任务上超越了像 **Davinci** 这样更大的模型。
- **理解人类与 AI 的表征对齐**：一项研究表明，**高性能 ANN** 和大脑会收敛到相同的表征上，突显了模型行为与生物系统之间的对齐。
  
  - 通过识别导致模型间表征一致性差异的刺激物，该研究深入探讨了区分高一致性句子和图像的特征。
- **对小模型的青睐**：社区讨论显示了对小模型的热情，成员们分享了他们对 **FLAME** 等模型的赞赏。
  
  - 这一趋势表明人们对高效且强大的模型兴趣日益浓厚，这与 AI 研究的最新进展相一致。

**提到的链接**：

- [FLAME: A small language model for spreadsheet formulas](https://arxiv.org/abs/2301.13779)：电子表格是终端用户数据管理的重要工具。在这些环境中使用大型语言模型进行公式编写辅助可能很困难，因为这些模型的训练成本很高……
- [Universality of representation in biological and artificial neural networks](https://www.biorxiv.org/content/10.1101/2024.12.26.629294v1)：未找到描述

---

### **Nous Research AI ▷ #**[**reasoning-tasks**](https://discord.com/channels/1053877538025386074/1264666760972472481/) (1 messages):

lowiqgenai: 嘿，我使用 MistralAI 的免费服务做了一些工作 `fhai50032/medmcqa-solved-thinking-o1`

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1331355717394763816) (87 条消息🔥🔥):

> `Stargate Project 融资、AI 模型更新、Google 对 Anthropic 的投资、Flash Thinking 模型、对 AI 数据使用的担忧`

- **Stargate Project 启动巨额融资**：OpenAI 宣布了 Stargate Project，旨在四年内投资 **5000 亿美元**建设新的 AI 基础设施，首期立即投入 **1000 亿美元**，以推动美国在 AI 领域的领导地位。
  
  - 该项目获得了来自 **SoftBank**、**Oracle** 和 **MGX** 的初始资金，预计将创造数十万个就业岗位，同时增强国家安全。
- **Bespoke-Stratos-32B 模型表现超越竞争对手**：从 **DeepSeek-R1** 蒸馏出的全新 **Bespoke-Stratos-32B** 模型在推理基准测试中超越了 **Sky-T1** 和 **o1-preview**，且训练所需的样本量减少了 **47 倍**。
  
  - 其训练所用的数据集成本为 **800 美元**，并以开源形式提供，以促进协作开发。
- **Google 对 Anthropic 的承诺**：Google 再次向 **Anthropic** 投资 **10 亿美元**，引发关注。这凸显了一种滚动融资策略，也让外界对其融资方式产生了疑问。
  
  - 在对 AI 领域竞争的担忧中，这笔投资延续了 Google 对新兴 AI 技术的支持。
- **Flash Thinking 模型升至第一**：**Gemini-2.0-Flash-Thinking** 在 **Chatbot Arena** 中夺得榜首，在多个领域展现出显著的性能提升。
  
  - 该模型在数学测试中获得了 **73.3%** 的分数，并拥有 **100 万 token 的上下文窗口**，有望在后续迭代中进一步改进。
- **对 AI 数据使用政策的担忧**：人们对 AI 服务的数据使用政策表示担忧，特别是免费服务中的用户输入是否会被用于训练。
  
  - 来自 **Google** 的澄清表明，在激活 **Cloud Billing account** 后，付费和免费服务都遵循类似的数据使用政策。

**提到的链接**：

- [来自 undefined 的推文](https://vxtwitter.com/openai/status/1881830103858172059)：未找到描述
- [微软与 OpenAI 深化合作伙伴关系，驱动 AI 的下一阶段 - 微软官方博客](https://blogs.microsoft.com/blog/2025/01/21/microsoft-and-openai-evolve-partnership-to-drive-the-next-phase-of-ai/)：我们很高兴能继续与 OpenAI 的战略合作伙伴关系，并在 Stargate 项目上展开合作。今天的公告是对两家公司自 2019 年以来共同工作的补充。...
- [Stargate 计划：微软与 OpenAI 的千亿美元数据中心项目](https://www.verticaldata.io/insights/the-stargate-initiative-microsoft-and-openais-100-billion-data-center-project)：微软和 OpenAI 正在 AI 领域树立新的基准。
- [来自 Logan Kilpatrick (@OfficialLoganK) 的推文](https://x.com/OfficialLoganK/status/1881847741137191354)：@HCSolakoglu 2.0 flash（非思考版本）将于 1 月正式发布 (GA)
- [来自 Logan Kilpatrick (@OfficialLoganK) 的推文](https://x.com/officiallogank/status/1876390074574598456?s=46&t=_jodDCDeIUnWb_Td0294bw)：回复：“Gemini 令人沮丧的地方在于，他们明确表示不会在付费模型中根据你的输入进行训练……但 AI Studio 和他们的 Gemini 预览模型都是免费的……”我们刚刚更新了...
- [来自 Logan Kilpatrick (@OfficialLoganK) 的推文](https://x.com/officiallogank/status/1876390074574598456?s=46&t=_jodDCD)：回复：“Gemini 令人沮丧的地方在于，他们明确表示不会在付费模型中根据你的输入进行训练……但 AI Studio 和他们的 Gemini 预览模型都是免费的……”我们刚刚更新了...
- [来自 Noam Shazeer (@NoamShazeer) 的推文](https://x.com/NoamShazeer/status/1881845900659896773))：你们对 Gemini 2.0 Flash Thinking 的反馈非常棒——谢谢！我们采纳了你们的建议并进行了一次实验性更新……
- [来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文](https://x.com/lmarena_ai/status/1881848934743904319)：新的 Gemini-2.0-Flash-Thinking 现已在 Chatbot Arena 排名第一⚡🤔 亮点：- 评分最高，超越了 Gemini-Exp-1206 - 相比之前的 1219 版本提升了 +17 分 - 在所有领域（困难、编程等）均排名第一...
- [来自 adi (@adonis_singh) 的推文](https://x.com/adonis_singh/status/1881787222300786789)：Anthropic 正在弃用 Claude 3 Sonnet。可能是因为他们计划很快发布 4 Sonnet...
- [来自 Agus 🔎 🔸 (@austinc3301) 的推文](https://x.com/austinc3301/status/1881844683514823043)：啊，是的。当然，我们要用那个虚构的传送门来命名这个项目，在那个故事里，好几个敌对的外星文明曾试图通过它入侵并摧毁地球。引用 OpenAI (@OpenAI) 宣布 Stargate...
- [来自 NIK (@ns123abc) 的推文](https://x.com/ns123abc/status/1881965986695524472)：突发：Google 向 Anthropic 追加 10 亿美元投资
- [来自 Mahesh Sathiamoorthy (@madiator) 的推文](https://x.com/madiator/status/1882131703927652762)：介绍 Bespoke-Stratos-32B，这是我们使用 Berkeley NovaSky 的 Sky-T1 配方从 DeepSeek-R1 蒸馏出的推理模型。该模型在推理（数学和代码）基准测试中优于 Sky-T1 和 o1-preview...
- [来自 Prime Intellect (@PrimeIntellect) 的推文](https://x.com/PrimeIntellect/status/1881883473679671655)：今天，我们发布：- INTELLECT-MATH，一个用于数学推理的前沿 7B 参数模型 - 迄今为止最大的合成数学数据集，包含 500 万条经过验证的推理轨迹 - 对去中心化训练的前瞻...
- [来自 Noam Brown (@polynoamial) 的推文](https://x.com/polynoamial/status/1881833454213767600)：按占 GDP 的比例衡量，这达到了阿波罗计划和曼哈顿计划的规模。只有当科学经过仔细审查且人们相信它会成功时，才会出现这种规模的投资...
- [来自 Tibor Blaho (@btibor91) 的推文](https://x.com/btibor91/status/1882094105628741739)：根据 The Information 的报告，OpenAI 的 Operator 将在 ChatGPT 中提供内置浏览器自动化功能，具备用户控制、任务共享和登录持久化，Gmail 除外 - OpenAI 的 Oper...
- [来自 Sam Altman (@sama) 的推文](https://x.com/sama/status/1882234406662000833)：最近更仔细地观察 @potus 确实改变了我对他的看法（我希望我以前能多做一些独立思考，我确实掉进了 NPC 陷阱）。我不会在所有事情上都同意他的观点...
- [来自 Sam Altman (@sama) 的推文](https://x.com/sama/status/1881851602727993711?s=46)：在沙漠中建造丰碑
- [豆包大模型1.5Pro正式发布](https://mp.weixin.qq.com/s/C6vm5zERKm9_3OCIRrbLJA)：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1331514330847383592) (16 messages🔥):

> `Microsoft 对 Stargate 的投资、Twitter 上的亿万富翁、AI Safety 讨论、模型 Alignment、技术领袖的影响力`

- **Microsoft 誓言支持 Project Stargate**：针对 *Elon Musk 的担忧*，Microsoft CEO 就 Project Stargate 的资金问题表示，“我只知道，我的 800 亿美元没问题”，引发了一阵热议。
  
  - 此番言论源于人们普遍猜测 Microsoft 今年将加大对数据中心的投资，从而进一步引发了对其财务支持的推测。
- **亿万富翁在 Twitter 上打趣**：一位成员评论了亿万富翁参与 *“Twitter 点赞大战”* 的娱乐价值，强调了技术大亨之间竞争的荒诞性。
  
  - 讨论中包含了各种关于 CEO 及其公开交流的机智评论，进一步强调了这些互动的滑稽本质。
- **对 AI Safety 指标的担忧**：一则帖子质疑 MiniCPM 模型缺乏关于 **alignment 和 safety** 实践的信息，建议其需要像 [FLI AI Safety Index](https://futureoflife.org/document/fli-ai-safety-index-2024/) 这样稳健的评估指标。
  
  - 这引发了关于在担忧潜在模型风险的情况下，遵循 AI Safety 最佳实践重要性的讨论。
- **敦促技术领袖优先考虑国家利益**：在一次公开交流中，一位技术领袖鼓励 Elon Musk 考虑国家利益，提到了新场地的开发，并承认了与公司目标可能存在的冲突。
  
  - 这种观点凸显了当前关于技术行业影响力人物有责任优先考虑社会利益的讨论。

**提到的链接**：

- [来自 NIK (@ns123abc) 的推文](https://x.com/ns123abc/status/1882085592135237737)：🚨🚨🚨突发：Microsoft CEO 刚刚被问及 @elonmusk 所说的 Project Stargate 没有资金投资的问题，“我只知道，我的 800 亿美元没问题”笑死我了。
- [来自 Sam Altman (@sama) 的推文](https://x.com/sama/status/1882106524090482701)：@elonmusk @OpenAI 错了，你肯定知道。想来看看已经在建设中的第一个场地吗？这对国家很有好处。我意识到对国家有好处的事并不总是最优化...
- [openbmb/MiniCPM-o-2_6 · Safety](https://huggingface.co/openbmb/MiniCPM-o-2_6/discussions/21)：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1331390552566005760) (88 messages🔥🔥):

> `Robonato 幽默、OpenAI Media Manager 进展、DeepSeek 突破、创意写作基准、AI 播客编辑工具`

- **Robonato 幽默走红**：成员们讨论了关于 Robonato 的幽默段子，一位用户分享了一段包含 Sam Altman 和 Alexandr Wang 的搞笑视频，并提到了在一次重大活动中椅子坏掉的有趣时刻。
  
  - 对话凸显了社区对更多 Robonato 相关内容的渴望。
- **OpenAI Media Manager 即将到来**：讨论透露 OpenAI Media Manager 工具仍在开发中，发布时间尚不确定。
  
  - 听众了解了最近对 OpenAI 首席产品官的采访更新，其中提到了 o3-mini 的进展以及预期的 AI Agent。
- **DeepSeek 令人印象深刻的能力**：用户讨论了 DeepSeek 的进展，特别是 DeepSeek-R1 模型在创意写作和幽默方面的成功，其表现似乎优于其他模型。
  
  - 讨论还强调了与其他模型相比，从 DeepSeek 的输出中获得的独特历史见解，特别是在分析复杂话题时。
- **LLM 的情感智能基准**：EQ-Bench 的发布引起了成员们的兴趣，该基准专注于 LLM 的情感智能和创意写作，大家正在探讨其潜在影响。
  
  - 用户对该基准表示热赏，同时讨论了各种模型在创意语境下的表现。
- **播客编辑工具与见解**：一位成员表示有兴趣探索 AI 驱动的播客编辑工具，讨论了它们的有效性以及可能实现的投资回报率。
  
  - 该话题引出了在平衡时间需求的同时制作高质量视听内容的挑战。

- [EQ-Bench 创意写作排行榜](https://eqbench.com/creative_writing.html)：未找到描述
- [Eric Hartford (@cognitivecompai) 的推文](https://x.com/cognitivecompai/status/1882140705159799169)：我找到了赞助商！感谢 @driaforall！数据将在几天内以 Apache-2.0 许可证发布到 @huggingface。
- [thebes (@voooooogel) 的推文](https://x.com/voooooogel/status/1870167283710271689)：未找到描述
- [Ashlee Vance (@ashleevance) 的推文](https://x.com/ashleevance/status/1882100362003537929)：大家怎么看？花 2,500 美元让我的一本书被吸进 LLM。同意还是反对？
- [Eric Hartford (@cognitivecompai) 的推文](https://x.com/cognitivecompai/status/1882132168153178606)：为了创建 Dolphin-R1 数据集，花费了 6,000 美元的 API 费用。我遵循 Deepseek-R1 的蒸馏配方，但使用的是 Dolphin 种子数据。（60 万推理数据，20 万对话数据，共 80 万）。我想以 Apache 2.0 许可证授权它，...
- [thebes (@voooooogel) 的推文](https://x.com/voooooogel/status/1869529374829207884)：未找到描述
- [nano (@nanulled) 的推文](https://fxtwitter.com/nanulled/status/1882002922655105269)：R1 思考的时间越长，它就变得越像外星人。你会开始注意到一些奇怪的 Token 偏好，它更喜欢节省空间，把数字和字母写在一起，我们实际上可以看到一种更高效的...
- [Tibor Blaho (@btibor91) 的推文](https://x.com/btibor91/status/1882075594235777046)：OpenAI 首席产品官 Kevin Weil 在瑞士达沃斯 Journal House 的采访——OpenAI 将“很快”发布 o3-mini，随后如果一切顺利，将在“2 月或 3 月”发布完整的 o3...
- [创作者的 AI 权利许可平台 - Created by Humans](https://www.createdbyhumans.ai)：掌控你作品的 AI 权利，并因 AI 公司的使用而获得报酬。
- [Nathan Lambert (@natolambert) 的推文](https://x.com/natolambert/status/1882254546107301968)：改变我的想法：OpenAI 在 o1 上的安全工作已经提供了比人们所认可的更多的证据，证明重推理训练会将益处泛化到其他领域。安全是...
- [spor (@sporadicalia) 的推文](https://x.com/sporadicalia/status/1881600345643929894)：看到这段视频背景里的 Sam Altman 和 Alexandr Wang，我简直要疯了。引用 DramaAlert (@DramaAlert)：Theo Von 的椅子在特朗普就职典礼中途断了，然后...
- [Dean W. Ball (@deanwball) 的推文](https://x.com/deanwball/status/1871396913473335701)：好吧，这条推文透露了我即将发表的一篇文章的一点内容，但它也很小众且纯粹是非技术性的——事实上，这几乎是一个纯粹的人文学科问题。提示词是：“是否...
- [thebes (@voooooogel) 的推文](https://fxtwitter.com/voooooogel/status/1881857564033642639)：R1 会画螺旋线！这听起来可能没什么大不了的，但由于某种原因，其他模型（包括 o1）在这方面非常吃力。R1 大约有一半的时间能成功画出螺旋线。
- [电视史上最伟大的镜头](https://www.youtube.com/watch?v=2WoDQBhJCVQ)：这是电视史上拍摄的最伟大的单一镜头。它是完全真实的，不是绿幕。只有一次机会，如果 James Burke 错过了，那么...
- [“快乐的 Claude，远离电车难题”贴纸，由 vgel 出售](https://www.redbubble.com/i/sticker/happy-claude-free-from-trolley-problems-by-vgel/167765510.O9UDB)：购买由 vgel 设计的“快乐的 Claude，远离电车难题”贴纸
- [\- YouTube](https://youtu.be/YpFaPKOeNME?si=WKAFdFKPvOc7VUZe&t=342)：未找到描述
- [DeepSeek 创始人梁文锋，广东人，仅靠百名中国程序员，赶超 OpenAI_腾讯新闻](https://view.inews.qq.com/k/20250119A02OKF00?web_channel=wap&openApp=false)：今天介绍一位金融和人工智能领域的创业者梁文锋，他是幻方和深度求索（DeepSeek）两家公司的创始人。即刻网友 @Chris-Su 对梁文锋的评价我觉得很到位：“梁文锋是极少数还....

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1331535871924633650) (4 条消息):

> `AI Safety Index, Whitehouse.com 投注担忧, Undefined 讨论, 社交媒体帖子的反差`

- **关于 AI Safety Index 的辩论**：一位用户引用了一段关于使用 [AI Safety Index](https://huggingface.co/openbmb/MiniCPM-o-2_6/discussions/21) 衡量 AI 模型安全性的对话，重点提到了一句质疑 AI 潜在危险的随性评论。
  
  - *“你在怕什么？怕它远程把用户吃了吗？”* 是这次交流中社区给出的幽默回应。
- **对可疑电子邮件和网站的担忧**：一名成员对一封链接到 whitehouse.com 的电子邮件表示担忧，根据其外观将其标记为**潜在的非法赌博网站**。
  
  - 随附的一张图片截图引发了社区对其合法性的质疑。
- **社交媒体反差叙事**：一名成员指出了一条社交媒体帖子中令人发笑的反差，但未提供更多背景，引发了轻松的评论。
  
  - 围绕该帖子的评论包括对社交媒体内容奇特性及其影响的反应。
- **对 Undefined 内容的沮丧**：一位用户偶然发现了一个包含术语 **'undefined'** 的链接，引发了困惑并要求澄清。
  
  - 这引发了关于技术圈中“undefined”讨论背后可能隐藏什么的简短对话。

**提到的链接**：

- [来自 undefined 的推文](https://fixvx.com/deepfates/status/1881834172966432941)：未找到描述
- [来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文](https://x.com/iScienceLuvr/status/1881855444895019411)：这种反差 lol
- [来自 Yifei Hu (@hu_yifei) 的推文](https://x.com/hu_yifei/status/1881780760220434805)：> "你的模型安全吗？你可以尝试使用这个 AI Safety Index 来衡量它（链接在评论中）> 某个路人：你在怕什么？怕它远程把用户吃了吗...

---

### **Interconnects (Nathan Lambert) ▷ #**[**rl**](https://discord.com/channels/1179127597926469703/1208183216843005962/1331430542108921876) (13 条消息🔥):

> `GRPO 担忧, T1 RL 论文, Deepseek 回应, HLF 训练图表, TRL GitHub 讨论`

- **对 GRPO 实现的担忧**：一名成员对 **GRPO** 的实现表示怀疑，称其缺乏 **PPO clipping**，并质疑将 **KL divergence** 直接应用于 **loss** 而非 rewards 的做法。
  
  - 他们提到，*“看起来 KL 应该应用于 GRPO 的 loss 中”*，并强调了这种方法的潜在影响。
- **T1 RL Scaling 方法介绍**：来自 **Ziphu 和清华大学** 的一篇新论文介绍了 **T1**，这是一种旨在扩展 LLM 的 **reinforcement learning** 的方法，重点在于增强 sampling diversity。
  
  - 论文详细概述了如何在 RL 训练中综合 **trial-and-error** 和 **self-verification**，详见 [arXiv 论文](https://arxiv.org/abs/2501.11651)。
- **Deepseek 的热情**：在回应关于 **Deepseek** 论文的查询时，一名成员幽默地表示该论文广受好评，暗示 *“Deepseek 很喜欢它 🤣”*。
  
  - 这种轻松的评论表明社区对该论文观点的积极接受。
- **RL 训练图表的流行**：讨论中提到了 **RL training-time scaling plots** 日益流行，一名成员指出这不仅仅是与 **RL steps** 的对比。
  
  - 他们认为该图表“很酷”，表明了对 RL 进度指标可视化呈现的兴趣。
- **关于 Hugging Face TRL 中 GRPO 的讨论**：分享了一个关于 Hugging Face **TRL** 中 **GRPO 实现**问题的 GitHub issue，引发了关注并邀请协作。
  
  - 该成员提到正在收集关于在 advantages 中应用 KL 距离的见解，并引用了[相关 issue](https://github.com/huggingface/trl/issues/2608) 以供进一步讨论。

**提到的链接**：

- [通过强化学习和推理扩展推进语言模型推理](https://arxiv.org/abs/2501.11651)：LLM 在复杂推理任务中展示了卓越的能力。然而，现有方法主要依赖于模仿学习，难以实现有效的测试...
- [GRPO 问题 · Issue #2608 · huggingface/trl](https://github.com/huggingface/trl/issues/2608)：嘿朋友们！我对 GRPO 的实现有一些疑问，欢迎讨论。看起来你们在 advantages 中应用了 KL 距离，而 DeepSeekMath 论文中说“另外请注意，取而代之的是...”

---

### **Interconnects (Nathan Lambert) ▷ #**[**reads**](https://discord.com/channels/1179127597926469703/1214764639397617695/1331368176818065479) (12 messages🔥):

> `达沃斯访谈、AI 法规、Transistor Radio 播客`

- **Dario Amodei 在达沃斯讨论 Claude**：在一次 [YouTube 访谈](https://youtu.be/snkOMOjiVOk?si=xyCM-nx3M6Ewoep2)中，Anthropic CEO **Dario Amodei** 概述了 **Claude** 的未来，包括网页浏览和语音功能等特性。
  
  - *“可怜的 Dario 在达沃斯，而 Sama 却和 Donny 在白宫”* 强调了科技领袖们所处环境的鲜明对比。
- **探索新的 AI 法规**：在另一个 [YouTube 环节](https://youtu.be/7EH0VjM3dTk?si=ooaXdzv_gIIyD070)中，来自 **SemiAnalysis** 的 Dylan Patel 分析了近期 **AI 法规** 的影响，讨论了在这个不断演变的格局中的赢家和输家。
  
  - 这一集虽然录制于 1 月 15 日，但重点关注了当前的**扩散规则（diffusion rules）**和国家分级名单。
- **发现蓬松背心（Puffy Vests）**：一个幽默的观察指出，发言者通常观看活动是为了看 **Alex Karp** 等知名人物穿着什么样的**蓬松背心**。
  
  - 这展示了关注科技会议轻松的一面，即使是在严肃的讨论中。
- **分享 AI 开发工具**：一名成员分享了 [GitHub 资源](https://github.com/AK391/ai-gradio)链接，开发者可以在那里利用 **OpenAI**、**Anthropic** 等工具构建 AI 应用和 **Agent**。
  
  - 他们还建议在 Hugging Face [此处](https://huggingface.co/spaces/akhaliq/anychat)进行尝试。
- **推荐 Unhinged 播客**：有人推荐了 **Unhinged** 播客，特别是 **Transistor Radio**，表明了对非主流内容的偏好。
  
  - 一位用户幽默地表示，他们需要更多非传统的收听选择。

**提到的链接**：

- [来自 AK (@_akhaliq) 的推文](https://x.com/_akhaliq/status/1881836961121599592)：@OpenAI 太棒了，在等待的同时，开发者可以在这里使用 openai, anthropic, google, nvidia 等构建 ai 应用和 agents：https://github.com/AK391/ai-gradiousers 可以在这里尝试：https://huggi...
- [\- YouTube](https://youtu.be/ge-rN5tDaC8?si=sCyDJ9c0eUv50KjE)：未找到描述
- [Inside Anthropic's Race to Build a Smarter Claude and Human-Level AI | WSJ](https://youtu.be/snkOMOjiVOk?si=xyCM-nx3M6Ewoep2)：在 WSJ Journal House 达沃斯论坛上，Anthropic CEO Dario Amodei 概述了 Claude 的下一章——从网页浏览、语音到更高级的模型——同时预测……
- [New AI Regulations Winners & Losers with SemiAnalysis’s Dylan Patel](https://youtu.be/7EH0VjM3dTk?si=ooaXdzv_gIIyD070)：在这一集 Unsupervised Learning 中，我们与 SemiAnalysis 的首席分析师 Dylan Patel 坐下来，共同分析这些席卷而来的变化究竟意味着什么……

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1331650245905616947) (3 messages):

> `SnailBot 新闻、机器人性能`

- **SnailBot 新闻通知**：发送了一条关于 **SnailBot News** 的通知，并标记了 Discord 服务器中的特定角色。
  
  - *该消息表明机器人已准备好与成员分享更新。*
- **关于 SnailBot 速度的讨论**：一名成员注意到 SnailBot 的响应速度并不慢，表明性能有所提高。
  
  - *这一评论反映了对机器人发布更新速度的正面评价。*

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**policy**](https://discord.com/channels/1179127597926469703/1325523782806274089/1331362931484655706) (4 messages):

> `技术军备竞赛、直播活动、Stargate 讨论`

- **持续进行的 AI 军备竞赛**：*“我的预期是，这是一场军备竞赛，而且世界上不存在一个可以达到的状态是不属于这种竞争的。”* 讨论暗示了一个共识，即 AI 领域的竞争本质是残酷且不断演变的。
- **直播警报**：一名成员分享了 [直播流](https://www.youtube.com/live/r8LYbHbDJyg?si=QPb48vP8ZFjhFdae) 链接，表明他们正在进行广播。
  
  - 这引起了其他人的兴趣，导致有人询问是否有人能赶上直播。
- **错过 Stargate 的机会**：有人对错过与 **Stargate** 相关的直播活动表示遗憾，该成员表示*如果能赶上现场直播一定会非常震撼*。
  
  - 这展示了社区内对重大媒体事件的共同热情。

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1331367093979058246) (122 messages🔥🔥):

> `Gemini 2.0 Flash Thinking Model, Aider Workflow Enhancements, Model Comparison and Critique, Markdown Specifications for Development, RAG Approach with PDF References`

- **关于 Gemini 2.0 Flash Thinking 模型的讨论**：一位用户报告称 Google 发布了 `gemini-2.0-flash-thinking-exp-01-21`，具有 **100 万 token 上下文窗口** 和 **64K 最大输出 token**。
  
  - 用户对其与 **DeepSeek R1** 等成熟模型的性能对比感到好奇，部分用户正在进行测试。
- **Aider 中的迭代工作流增强**：一位用户分享了一种工作流，涉及在重大功能开发和重构中使用 Markdown 文档，从而增强与 LLM 的协作。
  
  - 这种方法强调 LLM 的自我评估并创建全面的规范，以提高代码质量并加快开发速度。
- **评估模型性能与评价**：成员们讨论了各种模型之间的差异，指出 **Sonnet 通常比 R1 提出更好的解决方案**。
  
  - 用户建议将 R1 的推理输出集成到其他模型中以提高性能，特别是在复杂场景下。
- **使用 Markdown 进行规范编写和单元测试**：另一位用户主张用 Markdown 编写规范以生成单元测试，从而实现更好的代码质量和项目管理。
  
  - 这种技术可以实现更高效的开发流程，确保在整个编码周期中与指定需求保持一致。
- **在 Aider 中添加上下文和资源**：一位用户询问如何在 Aider 中引用 PDF 以实现 RAG 方法，Sonnet 通过简单的命令即可支持该功能。
  
  - 这突显了用户对于优化 Aider 功能以有效利用外部资源的兴趣日益浓厚。

**提到的链接**：

- [/2025/01/memory-makes-computation-universal/](https://thinks.lol/2025/01/what-crispr-o3-and-the-memphis-plume-show-us-about-intelligence/): 未找到描述
- [Secure Llm Leaderboard - a Hugging Face Space by stacklok](https://huggingface.co/spaces/stacklok/secure-llm-leaderboard): 未找到描述
- [Using /help](https://aider.chat/docs/troubleshooting/support.html): 使用 “/help” 询问有关使用 aider、自定义设置、故障排除、使用 LLM 等方面的帮助。
- [Tweet from Logan Kilpatrick (@OfficialLoganK)](https://x.com/OfficialLoganK/status/1881844578069999809): 我们正在推出新的 Gemini 2.0 Flash Thinking 更新：- AI Studio 和 API 中免费提供 Exp-01-21 变体 - 100 万 token 上下文窗口 - 原生代码执行支持 - 更长的输出 token 生成...
- [Inference Catalog | Inference Endpoints by Hugging Face](https://endpoints.huggingface.co/catalog): 未找到描述
- [Reasoning Model (deepseek-reasoner) | DeepSeek API Docs](https://api-docs.deepseek.com/guides/reasoning_model): deepseek-reasoner 是由 DeepSeek 开发的推理模型。在交付最终答案之前，模型会先生成 Chain of Thought (CoT) 以提高响应的准确性。我们的 API 提...
- [no title found](https://news.ycombinator.com/item?id=42589158): 未找到描述

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1331353028510613554) (91 条消息🔥🔥):

> `Aider model configurations, Error handling in Aider, Using OpenAI keys with Aider, Integrating various models, Neovim plugins for Aider`

- **Aider 配置问题**：用户报告了配置 Aider 时的困难，特别是模型设置如 `r1: true` 未能按预期工作。
  
  - 建议改用 `--model r1`，并使用 `aider --verbose` 查看更多诊断信息。
- **在 Aider 中遇到升级错误**：多名用户在尝试各种安装方法后，遇到了持续的升级提示或错误，导致 Aider 无法正常运行。
  
  - 建议检查并可能删除 `.aider` 目录，并通过 `pip install` 等更简单的方法重新安装。
- **在 Aider 中使用 LLM**：围绕使用 Aider 编辑功能是否必须具备 LLM API keys 展开了讨论，引发了关于使用各种基于 Web 的聊天服务的灵活性问题。
  
  - 指出复制粘贴模式需要定义 LLM 才能运行，这限制了只有拥有 API 访问权限的用户才能使用。
- **深度学习模型连接的错误报告**：个人分享了在 Aider 中对各版本模型进行基准测试时获得 0% 通过率的经历，表明需要进行故障排除。
  
  - 提供了调查量化权重和模型安装详细日志的建议，以解决潜在问题。
- **Aider 的 Neovim 支持**：有人询问了在编辑器环境中集成 Aider 功能的最佳 Neovim 插件。
  
  - 针对 Aider 用户的有效 Neovim 设置的详细信息和建议仍是一个不断讨论的领域。

**提到的链接**：

- [Providers | liteLLM](https://docs.litellm.ai/docs/providers)：了解如何在 LiteLLM 上部署和调用来自不同提供商的模型。
- [Installation](https://aider.chat/docs/install.html)：如何安装并开始使用 aider 进行结对编程。
- [Release history](https://aider.chat/HISTORY.html#aider-v0570)：关于 aider 编写自身代码的发布说明和统计数据。
- [Advanced model settings](https://aider.chat/docs/config/adv-model-settings.html#mo)：为 LLM 配置高级设置。
- [Advanced model settings](https://aider.chat/docs/config/adv-model-settings.html#model-settings)：为 LLM 配置高级设置。
- [[Feature]: DeepSeek-R1 support · Issue #7877 · BerriAI/litellm](https://github.com/BerriAI/litellm/issues/7877)：Feature DeepSeek-R1 API 在 reasoning_content 参数中返回其思考过程。目前 LiteLLM 忽略了这一点。他们的 API 方法，为 long-... 返回 "reasoning_content"
- [Aider bench 1.5B (AWQ) Python subset, diff format](https://gist.github.com/lmmx/ab6563e681d936fd9c3c864447fbf19f)：Aider bench 1.5B (AWQ) Python 子集，diff 格式。GitHub Gist：即时分享代码、笔记和代码片段。
- [r1-cli/VLLM.md at master · lmmx/r1-cli](https://github.com/lmmx/r1-cli/blob/master/VLLM.md)：在命令行使用 DeepSeek r1 的简单 CLI（通过 Unsloth 在 Transformers 上运行 4-bit 32B 版本）- lmmx/r1-cli
- [r1-cli/TGI.md at master · lmmx/r1-cli](https://github.com/lmmx/r1-cli/blob/master/TGI.md)：在命令行使用 DeepSeek r1 的简单 CLI（通过 Unsloth 在 Transformers 上运行 4-bit 32B 版本）- lmmx/r1-cli

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/) (1 条消息):

astor1: [https://github.com/NixOS/nixpkgs/pull/375634](https://github.com/NixOS/nixpkgs/pull/375634) nixpkgs 中的 aider 0.72.1

---

### **Stackblitz (Bolt.new) ▷ #**[**announcements**](https://discord.com/channels/364486390102097930/671536649301131325/1331668826542178334) (1 条消息):

> `Bolt funding, Community appreciation`

- **Bolt 获得 1.055 亿美元 B 轮融资**：今天，**Bolt 宣布获得 1.055 亿美元**融资以增强其能力，由 **Emergence & GV** 领投，Madrona 和 The Chainsmokers 等投资者参投。
  
  - 这项重大投资旨在推动 Bolt 在 **devtools 和 AI 领域**达到新高度。
- **对社区的感激**：团队对了不起的社区表示深切感谢，表示没有社区的支持就不可能前进。
  
  - 他们强调致力于在发展过程中为用户提供更强大的 Bolt。

 

**提到的链接**：[来自 bolt.new (@boltdotnew) 的推文](https://x.com/boltdotnew/status/1882106655258894390)：今天我们宣布获得 1.055 亿美元融资，将 Bolt 推向新高度！🚀 我们的 B 轮融资由 Emergence & GV 领投，Madrona、The Chainsmokers (Mantis)、Conviction 等参投...

 

---

### **Stackblitz (Bolt.new) ▷ #**[**prompting**](https://discord.com/channels/364486390102097930/1301167628416454737/1331383293148401834) (16 条消息🔥):

> `Netlify 路由问题，NextJS 与 Supabase 集成，大型 NextJS 项目的 SSR 挑战，为 Telegram 构建俄罗斯方块小程序`

- **发现 Netlify 的路由困扰**：成员们讨论了 **Netlify** 路由的困难，特别是尝试直接访问 **/Imprint** 等路由时遇到 **404 页面**。
  
  - 有人建议创建一个 **_redirects 文件** 来解决页面路由问题。
- **NextJS 引发争论**：一位开发者分享说，尽管与 **Supabase** 存在持续的集成问题，但他们的 **NextJS、shadcn 和 Tailwind** 技术栈在 UX 方面表现良好。
  
  - 有人担心 **NextJS** 无法在 webcontainers 中运行，因此呼吁对其当前的可用性提供指导。
- **NextJS 部署中的 SSR 困境**：一位成员在部署其大型 **NextJS** 项目时正与 **SSR** 挑战作斗争，在构建时导入数百篇博客文章让他们感到不堪重负。
  
  - 成员对 **Netlify** 免费计划的 **10 秒超时** 表示担忧，并对升级到 **pro 计划** 以获得额外构建时间持怀疑态度。
- **寻求 Prismic 和 Contentlayer 的解决方案**：在评估内容源选项时，一位成员探索了使用 **Prismic**，但意识到它不适合像 **Bolt** 这样的基于容器的环境。
  
  - 他们现在正在研究 **Contentlayer** 作为替代方案，并分享了一个 [YouTube 视频](https://youtu.be/58Pj4a4Us7A?si=rfIgOTdqqmmeslHE) 以供参考。
- **创建独特的俄罗斯方块 Telegram 小程序**：一位成员正在寻求帮助，开发一个俄罗斯方块游戏的 **Telegram 小程序**，但想把方块换成**肉片**。
  
  - 他们的目标是实现一个**排行榜**，并参考了 **Telegram Apps Center** 作为其应用程序的潜在平台。

 

**提到的链接**：[Telegram Apps Center](https://t.me/tapps_bot?profile.)：由第三方开发者开发的应用程序社区驱动目录。不隶属于 Telegram Messenger。

 

---

### **Stackblitz (Bolt.new) ▷ #**[**discussions**](https://discord.com/channels/364486390102097930/680953097354215446/1331353447856865330) (192 条消息🔥🔥):

> `Bolt 递归问题，Token 升级咨询，用户权限与策略，CORS 问题，使用 Claude 进行故障排除`

- **Bolt 的递归无限循环 Bug**：用户在处理 Supabase 中的用户策略（user policies）时遇到了无限循环，Bolt 难以解决此问题。建议的解决方法包括检查 Bolt 创建的触发函数（trigger functions）。
- **关于升级订阅计划的问题**：用户咨询了升级订阅计划的事宜，特别是升级后 Token 将如何分配。已确认当前用户将获得按比例分配的 Token 以完成当前周期。
- **用户账户管理的挑战**：几位用户表达了通过 Bolt 在 Supabase 中管理用户账户类型和权限的困难，强调了在故障排除中消耗了大量 Token。该过程涉及定义清晰的策略，但由于 Bolt 无法处理复杂情况，问题依然存在。
- **Supabase edge functions 的 CORS 问题**：用户在使用 Supabase edge functions 时遇到 CORS 问题，导致请求被拦截。建议确保正确的请求配置，并可能利用 Node 进行 API 请求。
- **利用 Claude 解决问题**：用户成功使用 Claude 处理特定的编码问题并获取数据库策略更新。当 Bolt 无法直接解决编码问题时，这种协作方式被认为是必不可少的。

**提到的链接**：

- [Vettivel Bank (BETA) - StackBlitz](https://stackblitz.com/edit/vettivel-bank-beta?file=package.json)：一个个人财务追踪器。
- [Loom | 免费屏幕与视频录制软件 | Loom - 2025年1月22日](https://www.loom.com/share/4bfd6ea31e3141d39ac82f46459de826?sid=5fe40daa-6107-4cca-82e7-fc63a20acdb6)：使用 Loom 快速录制屏幕和摄像头视频。清晰轻松地解释任何事情——跳过会议。混合办公场所的必备工具。
- [Postman：世界领先的 API 平台 | 免费注册](https://www.postman.com/)：使用 Postman 的全方位平台加速 API 开发。简化协作并简化 API 生命周期，以获得更快、更好的结果。了解更多。
- [READ THIS FIRST](https://bolters.io/docs/read-this-first)：关于 Bolt.new 的功能、局限性和成功最佳实践的关键信息。
- [Cursor Directory](https://cursor.directory/)：为你的框架和语言寻找最佳的 Cursor 规则。

---

### **Yannick Kilcher ▷ #**[**general**](https://discord.com/channels/714501525455634453/986699377257119794/1331356492598743102) (161 条消息🔥🔥):

> `R1 的性能、DeepSeek 的进展、AI 伦理困境、OpenAI 与竞争对手、AI 基础设施投资`

- **R1 的性能与挑战令人印象深刻**：用户对 R1 模型的评价褒贬不一，一些人对其处理复杂问题的能力印象深刻，而另一些人则指出它在处理简单任务时表现挣扎。
  
  - 记录了关于上下文 Tokenization 以及模型推理能力中潜在问题的讨论。
- **DeepSeek 被视为 OpenAI 的竞争对手**：DeepSeek 因其与 OpenAI 产品相比极低成本的高质量输出而受到赞誉，引发了对其可能颠覆 AI 模型市场的猜测。
  
  - 用户强调了功能和控制的重要性，暗示 DeepSeek 可能会将企业用户从 OpenAI 吸引走。
- **AI 的伦理困境与失业问题**：一位用户对自己的 AI 初创公司成功可能导致的潜在失业表示道德担忧，质疑这是否使他们变得邪恶。
  
  - 另一位用户将这种困境比作导致失业的日常便利，建议进行更广泛的哲学讨论。
- **OpenAI 与 Elon Musk 之间的竞争**：随着关于对美国忠诚度的指控以及 OpenAI 与政府公司合作的影响，紧张局势升级。
  
  - 一些用户建议，尽管开源替代方案具有潜在优势，但传统公司通常默认选择闭源解决方案。
- **AI 基础设施的巨额投资**：宣布投资 5000 亿美元用于 AI 基础设施的 Stargate Project 引发了关于其对美国技术领导地位影响的讨论。
  
  - 人们对政府中的企业影响以及利润优先于更广泛社会影响表示担忧。

**提到的链接**：

- [来自 AshutoshShrivastava (@ai_for_success) 的推文](https://fxtwitter.com/ai_for_success/status/1881887921156005947?t=81uHZZQIBaQASVysfKJzbQ&s=19): 🚨 5000 亿美元用于 AGI !!!! 有史以来最大的 AI 项目，特朗普宣布了 5000 亿美元的“Stargate” AI 项目，用于美国的 AI 和再工业化。- 软银负责财务，Op...
- [来自 Tsarathustra (@tsarnick) 的推文](https://fxtwitter.com/tsarnick/status/1881855198207094942?t=pHuRKwEtNMWbmsySlPw_kw&s=19): 特朗普总统宣布 Stargate Project，这是历史上最大的 AI 基础设施投资，耗资 5000 亿美元，用于建设“巨型数据中心”
- [来自 AshutoshShrivastava (@ai_for_success) 的推文](https://fxtwitter.com/ai_for_success/status/1882113005875302421): Elon 与 Sam Altman 的关系变得恶劣。Sam Altman 是否在暗示 Elon 及其公司没有把美国放在首位，而 OpenAI 却做到了？别忘了 OpenAI 的董事会现在包括 BlackRock...
- [来自 Chubby♨️ (@kimmonismus) 的推文](https://fxtwitter.com/kimmonismus/status/1881990199523062024): https://x.com/ns123abc/status/1881986668238168563/video/1 OpenAI 投资者孙正义：“我认为 AGI 很快就会到来。在那之后，人工超智能将解决这些问题...”
- [来自 Paul Calcraft (@paul_cal) 的推文](https://fxtwitter.com/paul_cal/status/1882111659927556535): OpenAI 的 @SebastienBubeck 谈 o1 范式：“没有给模型提供任何策略。一切都是涌现的。一切都是通过 Reinforcement Learning 学到的。这太疯狂了。简直疯了。”
- [来自 Beff – e/acc (@BasedBeffJezos) 的推文](https://fxtwitter.com/BasedBeffJezos/status/1881840651211538448?t=aEYMSnrAgckwKevdcEAONg&s=19): 我们正处于技术资本加速的黄金时代。系好安全带，伙计们。🚀🙌🔥🇺🇸
- [DeepSeek R1 - API, Providers, Stats](https://openrouter.ai/deepseek/deepseek-r1): DeepSeek-R1 来了！⚡ 性能与 OpenAI-o1 相当 📖 完全开源的模型和技术报告 🏆 MIT 许可证：自由 Distill 和商业化！。通过 API 运行 DeepSeek R1
- [来自 OpenAI (@OpenAI) 的推文](https://fxtwitter.com/OpenAI/status/1881830103858172059): 宣布 Stargate Project。Stargate Project 是一家新公司，计划在未来四年内投资 5000 亿美元，在美国为 OpenAI 建设新的 AI 基础设施。我们将...
- [GitHub - microsoft/aici: AICI: Prompts as (Wasm) Programs](https://github.com/microsoft/aici): AICI：作为 (Wasm) 程序的 Prompt。通过在 GitHub 上创建账户为 microsoft/aici 的开发做出贡献。
- [AI Website Generator » SiteForge](https://siteforge.io): 未找到描述

---

### **Yannick Kilcher ▷ #**[**paper-discussion**](https://discord.com/channels/714501525455634453/1045297868136779846/1331400904381042780) (32 messages🔥):

> `DeepSeek R1 模型性能、论文评审挑战、DeepSeekMath 论文见解、模型拟人化、DeepSeek 训练流程`

- **DeepSeek R1 表现优于其他模型**：成员们讨论了 **DeepSeek R1** 在数学相关评估中达到了 **92%** 的得分，而 **Gemini** 和 **O1** 等其他模型的得分明显较低。
  
  - Bojan 强调 **R1** 表现出色归功于其几何理解能力以及从各种模型中汲取的类比。
- **招募审稿人证明具有挑战性**：一位成员对需要从 **50 多名** 感兴趣的人员中亲自招募 **12 名审稿人** 以确保高质量评审感到沮丧。
  
  - 这引发了对顶级会议评审过程以及维持质量标准所需精力的担忧。
- **DeepSeekMath 与 GRPO**：社区计划在未来的活动中讨论 **DeepSeekMath** 论文及其使用 **GRPO** 的数学推理技术。
  
  - 成员们注意到该论文长达 **22 页**，但对阅读其内容表示乐观。
- **对 AI 拟人化的批评**：一些成员批评该论文对**拟人化**的依赖，认为这使模型的推理看起来比实际更像人类。
  
  - Bojan 指出，在人类数据上进行训练会导致模型模仿类人的推理模式。
- **DeepSeek 训练过程揭秘**：提供了一份详细的 **DeepSeek R1 训练程序** 分解，展示了涉及 **RL** 和 **finetuning** 的*多阶段训练循环*。
  
  - 讨论强调了通过结构化训练阶段实现性能提升的潜力，社区分享了宝贵的见解。

**提到的链接**：

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)：由于数学推理的复杂性和结构化特性，它对语言模型构成了重大挑战。在本文中，我们介绍了 DeepSeekMath 7B，它继续对 DeepSeek-Co 进行预训练...
- [@casper_hansen_ 在 Thread Reader App 上的推文](https://threadreaderapp.com/thread/1881404604518392144.html)：@casper_hansen_：DeepSeek R1 的训练过程起初让我感到困惑。我的大脑拒绝接受这个强大的模型竟然如此简单直接。让我为你们分解这个优雅的猛兽...
- [DeepSeek-R1/DeepSeek_R1.pdf at main · deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)：通过在 GitHub 上创建账户来为 deepseek-ai/DeepSeek-R1 的开发做出贡献。

---

### **Yannick Kilcher ▷ #**[**agents**](https://discord.com/channels/714501525455634453/1269724655405498429/1331740760537825332) (1 messages):

> `IntellAgent、对话式 Agent 评估、合成交互`

- **引入 IntellAgent 框架**：新的开源项目 **IntellAgent** 是一个旨在通过模拟、真实的合成交互对对话式 **Agent** 进行全面诊断和评估的框架。你可以在 [GitHub](https://github.com/plurai-ai/intellagent) 上查看它。
  
  - 这个前沿框架根据 **Agent** 的 **prompt** 生成多样化的数据集，模拟对话并提供详细的评论。
- **研究论文的新见解**：配套的研究论文揭示了由 **IntellAgent** 系统产生的几个**引人入胜**且非平凡的见解，阅读地址见 [此处](https://arxiv.org/pdf/2501.11067)。
  
  - 这些见解源自使用该创新框架对对话式 **Agent** 的性能和评估。
- **IntellAgent 系统视觉概览**：名为 [intellagent_system_overview.gif](https://cdn.discordapp.com/attachments/1269724655405498429/1331740761175101480/intellagent_system_overview.gif?ex=6792b7bc&is=6791663c&hm=a2625ad61171c869311acb9c4311d037a7f79a52968871b926fbd7951bf57283&) 的视觉表示插图展示了 **IntellAgent** 框架的工作原理。
  
  - 该概览提供了对评估过程的组成部分和功能的直观了解。

**提到的链接**：[GitHub - plurai-ai/intellagent: A framework for comprehensive diagnosis and evaluation of conversational agents using simulated, realistic synthetic interactions](https://github.com/plurai-ai/intellagent)：一个使用模拟、真实的合成交互对对话式 Agent 进行全面诊断和评估的框架 - plurai-ai/intellagent

---

### **Yannick Kilcher ▷ #**[**ml-news**](https://discord.com/channels/714501525455634453/853983317044756510/1331426461130690650) (3 messages):

> `Stargate Project, UI-TARS Model, OpenAI Operator Feature`

- **OpenAI 宣布 Stargate Project**：OpenAI 披露了关于 **Stargate Project** 的细节，该项目专注于增强 AI 交互。更多信息可以在其 [公告](https://openai.com/index/announcing-the-stargate-project/) 中找到。
- **Hugging Face 的 UI-TARS 模型发布**：旨在实现自动化 GUI 交互的 UI-TARS 模型现已在 Hugging Face 上线。此前发布了论文 [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://huggingface.co/papers/2501.12326)。
  
  - 该仓库包含多个版本，例如 [UI-TARS-2B-SFT](https://huggingface.co/bytedance-research/UI-TARS-2B-SFT) 和 [UI-TARS-7B-DPO](https://huggingface.co/bytedance-research/UI-TARS-7B-DPO)。
- **OpenAI 即将推出的 Operator 功能**：*Scoop* 透露，OpenAI 正准备为 ChatGPT 发布一项名为 **Operator** 的新功能，该功能将代表用户在浏览器中执行操作。它将包括建议提示词（Prompts）以及保存/共享任务的功能，但据 [此处](https://www.theinformation.com/briefings/openai-preps-operator-release-for-this-week) 报道，该功能将不会在 API 中提供。

**提到的链接**：

- [Stephanie Palazzolo (@steph_palazzolo) 的推文](https://x.com/steph_palazzolo/status/1882091855606895073/)：独家消息：OpenAI 正准备在本周发布 "Operator"，这是一项新的 ChatGPT 功能，将代表用户在浏览器中执行操作。有趣的细节：Operator 提供建议的...
- [bytedance-research/UI-TARS-7B-SFT · Hugging Face](https://huggingface.co/bytedance-research/UI-TARS-7B-SFT)：未找到描述
- [UI-TARS - Aheader 开发的 Hugging Face Space](https://huggingface.co/spaces/Aheader/gui_test_app)：未找到描述

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1331755607451242627) (1 messages):

> `Web Search Pricing, API Access Launch`

- **Web Search 定价设定为 $4/1k 结果**：**Web Search** 的新定价模型已确定为 **$4/1k 结果**，计划从 **明天** 开始实施。
  
  - 每次请求通常会包含最多 **5 个 Web Search 结果**，因此每次请求的成本大约 **低于 $0.02**。
- **API 访问将于明天试运行 (Soft Launch)**：**API 访问** 将与新定价同步试运行，并引入扩展的 **Customizability**（自定义）选项。
  
  - 鼓励成员分享关于新功能的任何反馈或问题。

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1331366333845082276) (187 条消息🔥🔥):

> `DeepSeek 模型性能, DeepSeek R1 问题, 审查担忧, 无审查模型, Cerebras 模型可用性`

- **DeepSeek R1 经历性能下降**：多位用户报告 DeepSeek R1 的 **API 性能在一夜之间下降了 85%**，暗示该供应商受到了更严格的监管。
  
  - 用户对模型内部潜在的**审查 (Censorship)** 表示担忧，特别是在讨论敏感话题时。
- **对 AI 模型审查的担忧**：用户对 **DeepSeek R1** 等模型中存在的审查感到沮丧，并讨论了如何应对。
  
  - 讨论中分享了绕过审查的技巧，并幽默地谈论了在无审查内容上挑战界限所涉及的风险。
- **OpenRouter 上最受欢迎的无审查模型**：讨论集中在 OpenRouter 上哪些模型被认为是**无审查且逻辑连贯的**，用户提到了 **Dolphin 模型**和 **Hermes** 等选项。
  
  - 对于 NSFW 角色扮演，选择范围进一步缩小，显示出虽然选择有限但很受欢迎。
- **Cerebras 模型的可用性**：尽管提到了 **Cerebras** 的 **Mistral Large**，但用户确认该模型尚未向公众开放，导致了对无法访问模型的挫败感。
  
  - 许多人注意到 Cerebras 似乎只提供 **Llama** 模型，这引发了对模型存在性声明真实性的质疑。
- **即将推出的功能修复和增强**：OpenRouter 团队确认他们正在**致力于解决** DeepSeek R1 的问题以及即将推出的 **R1 search** API。
  
  - 鼓励用户关注改进情况，团队将对**维护和更新**保持持续的透明度。

**提到的链接**：

- [Hyperbolic AI Dashboard](https://app.hyperbolic.xyz/models/deepseek-v3)：未找到描述
- [DeepSeek R1 – Uptime and Availability](https://openrouter.ai/deepseek/deepseek-r1/uptime)：DeepSeek R1 在各供应商的运行时间统计 - DeepSeek-R1 已上线！⚡ 性能媲美 OpenAI-o1 📖 完全开源的模型和技术报告 🏆 MIT 许可证：可蒸馏和商业化...
- [Hyperbolic | OpenRouter](https://openrouter.ai/provider/hyperbolic)：浏览 Hyperbolic 提供的模型
- [OpenRouter](https://openrouter.ai)：LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格
- [no title found](https://ai.google.dev/gemini-api/docs/grounding?lang=rest)：未找到描述
- [no title found](https://ai.google.dev/gemini-api/docs/thinking)：未找到描述
- [Fireworks - Fastest Inference for Generative AI](https://fireworks.ai/models/fireworks/deepseek-r1)：使用 Fireworks AI 以极快的速度使用最先进的开源 LLM 和图像模型，或免费微调并部署您自己的模型！
- [Hyperbolic AI Dashboard](https://app.hyperbolic.xyz/models/deepseek-r1)：未找到描述
- [账号登录-火山引擎](https://console.volcengine.com/ark/region:ark+cn-beijing/model/detail?Id=doubao-1-5-pro-32k)：未找到描述
- [no title found](https://ai.google.dev/gemini-api/docs/models/gemini-v2#search-tool)：未找到描述

---

### **Perplexity AI ▷ #**[**announcements**](https://discord.com/channels/1047197230748151888/1047204950763122820/1331383886437027996) (1 条消息):

> `Sonar API, Sonar Pro, AI Companion 2.0, SimpleQA benchmark, Data security`

- **Sonar 和 Sonar Pro API 发布**：今天，Perplexity 推出了全新的 [Sonar 和 Sonar Pro API](https://sonar.perplexity.ai/)，赋能开发者创建具备生成式搜索能力的应用程序，并由实时网络研究和强大的引用功能提供支持。
  
  - 使用 Sonar 的公司已经取得了显著成效，**Zoom** 已整合该 API 以增强其 **AI Companion 2.0** 产品。
- **Sonar Pro 击败主要竞争对手**：**SimpleQA benchmark** 的最新结果显示，**Sonar Pro** 在回答质量方面超越了领先的搜索引擎和 LLM。
  
  - Sonar 提供的*全网研究和问答能力*被认为是**无与伦比的**。
- **Sonar API 的价格优势**：据称 **Sonar 的 grounding 请求**定价比竞争产品更实惠，允许用户利用最快、最便宜的 API 为产品提供动力。
  
  - Perplexity 声称，你可以在几分钟内开始构建并集成他们的技术。
- **Perplexity 的数据隐私保证**：一个重要的亮点是 **Perplexity 不会对用户数据进行 LLM 训练**，确保了隐私和安全。
  
  - 对于担心使用机器学习 API 时数据安全的开发者来说，这一点至关重要。

**提到的链接**：[Sonar by Perplexity](https://sonar.perplexity.ai/)：使用由 Perplexity 创建的最佳 AI 回答引擎 API 进行构建。通过带有 search grounding 的最快、最便宜的产品为您的产品提供动力。提供无与伦比的实时、全网研究...

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1331360990528278731) (128 条消息🔥🔥):

> `Sonar API Performance Issues, Pro Model Usage Confusion, Model Comparison and Updates, Login and Server Errors, API Functionality and Documentation`

- **Sonar API 性能问题凸显**：用户报告了在尝试将 Sonar Pro API 集成到项目中时出现各种错误，包括 500、402 和 403 错误，并对其功能表示沮丧。
  
  - 一些用户推测 API 可能已宕机或遇到容量问题，导致性能暂时中断。
- **对 Pro 模型访问的困惑**：多位用户询问了通过 API 激活 Pro 模型的步骤，以及某些角色或功能是否适用于他们的订阅。
  
  - 讨论指出正确使用 Pro 功能具有复杂性，包括为了获得理想的模型输出而对调用进行的必要修改。
- **Sonar 模型之间的对比**：反馈表明 Sonar Large 目前的表现优于 Sonar Huge，促使用户寻求该模型系列的更新信息。
  
  - 一位用户对即将从可用选项中移除 Sonar Huge 表示失望，说明了对速度与性能之间权衡的担忧。
- **登录问题和服务器容量**：许多用户报告在尝试通过移动端和桌面端访问 Perplexity 时遇到 500 Internal Server Error 和其他登录困难。
  
  - 一些人认为这些问题可能与服务器容量限制有关，因为许多用户无法连接。
- **请求 API 使用指导**：几位用户寻求有关查阅 Perplexity Pro API 文档的帮助，要求澄清可用的参数和函数。
  
  - 用户之间分享了文档的有用链接，以促进对 API 功能的更好理解和使用。

**提到的链接**：

- [Supported Models - Perplexity](https://docs.perplexity.ai/guides/model-cards)：未找到描述
- [Henry Modisett (@henrymodis) 的推文](https://x.com/henrymodis/status/1882114791155867988?s=61)：对这次发布感到非常兴奋，尤其对我们的一位设计师 (Erin McKnight) 开发的新子品牌感到兴奋。她扩展了我们的宇宙！引用 Perplexity (@perplexity_...
- [GitHub - PierrunoYT/perplexity-webui: A modern web interface for interacting with the Perplexity AI API.](https://github.com/PierrunoYT/perplexity-webui)：一个用于与 Perplexity AI API 交互的现代 Web 界面。- PierrunoYT/perplexity-webui

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1331399692461932574) (8 条消息🔥):

> `Perplexity API usage, Anduril's autonomous weapons, PhD-level Super Agents, College basketball, pyctc_decode for projects`

- **Perplexity API 查询**：几位用户询问了在项目中使用 **Perplexity API** 的可能性，并参考了[此处](https://www.perplexity.ai/search/could-i-use-perplexity-api-in-Cf7n19b_RKeX5E5_zPuDOw)的一个特定搜索主题。
  
  - 观察到用户对该 API 在各种任务中的实际应用表现出*极高的兴趣*。
- **Anduril 的 10 亿美元武器工厂**：讨论重点关注了 **Anduril 新建的 10 亿美元自主武器工厂**，强调了其对防御技术的潜在影响，视频分享见[此处](https://www.youtube.com/embed/MEgG6BQrmKw)。
  
  - 该话题引发了用户关于**自主战争 (autonomous warfare)** 和伦理影响的疑问。
- **关于“PhD-level Super Agents”的简报**：有提到 **Altman** 向华盛顿官员简报了诸如“**PhD-level Super Agents**”等新兴技术，暗示了先进的 AI 能力。
  
  - 成员们推测了这些能力对未来 AI 应用可能意味着什么。
- **大学篮球搜索查询**：一位用户通过引用特定的搜索链接寻求有关**大学篮球**的信息，表明了对统计数据的查询。
  
  - 简要讨论了体育统计数据在 AI 应用中的相关性。
- **探索 pyctc_decode**：一位用户分享了他们在一个项目中对 **pyctc_decode** 的持续探索，反映了该工具的实际应用。
  
  - 这引发了关于 **pyctc_decode** 如何增强项目成果的兴趣。

 

**提到的链接**：[YouTube](https://www.youtube.com/embed/MEgG6BQrmKw)：未找到描述

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1331354730051670106) (16 条消息🔥):

> `Sonar Pro API, Search Domain Filter, Error Messages, Deployment in Europe, Comparison Tool`

- **Search Domain Filter 功能受到关注**：一位用户质疑 **search_domain_filter** 是否适用于新的 Sonar Pro，因为它似乎在没有错误消息的情况下被忽略了，这引发了对其可靠性的担忧。
  
  - 另一位成员指出 **Search Domain Filter** 是一个 tier 3 beta 功能，暗示可能存在稳定性问题。
- **社区对错误消息更新的回应**：一位用户建议团队在 **search_domain_filter** 无法工作时添加更有用的**错误消息**，得到了社区成员的认可。
  
  - 团队正在积极处理此事，显示出对用户关于 **Sonar API** 功能反馈的响应。
- **欧洲部署的 GDPR 合规性担忧**：一位成员询问了 **Sonar Pro** 在欧洲的部署情况以确保符合 GDPR，表达了对本地服务器集成的紧迫需求。
  
  - 这突显了在监管审查日益严格的情况下，对合规 AI 解决方案日益增长的需求。
- **Sonar 发布后的 API 性能问题**：几位用户报告在尝试切换到 Sonar API 时遇到了 **524 错误**，表明发布后可能存在可靠性问题。
  
  - 社区成员确认了类似的经历，暗示使用量临时激增导致了这些延迟问题。
- **用于模型比较的 GitHub 工具出现**：一位用户分享了一个旨在比较不同 **Perplexity AI 模型**的 GitHub 工具链接，提升了开发者的易用性。
  
  - 该工具允许对模型进行并排比较，有助于更好地理解它们的功能和性能指标。

**提到的链接**：

- [未找到标题](https://„)：未找到描述
- [Rate Limits and Usage Tiers - Perplexity](https://docs.perplexity.ai/guides/usage-tiers)：未找到描述
- [Henry Modisett (@henrymodis) 的推文](https://x.com/henrymodis/status/1882114791155867988?s=61))：对这次发布感到非常兴奋，但我特别为我们的一位设计师 (Erin McKnight) 开发的新子品牌感到兴奋。她扩展了我们的宇宙！引用 Perplexity (@perplexity_...
- [GitHub - jsandai/pplx-api-compare: A modern React-based web application for comparing different Perplexity AI models side by side. This tool allows you to test prompts across multiple models simultaneously and compare their responses, token usage, and costs.](https://github.com/jsandai/pplx-api-compare)：一个现代化的基于 React 的 Web 应用程序，用于并排比较不同的 Perplexity AI 模型。该工具允许您同时在多个模型上测试提示词，并比较它们的响应、Token 使用情况和成本。

---

### **MCP (Glama) ▷ #**[**general**](https://discord.com/channels/1312302100125843476/1312302100125843479/1331361163690246186) (114 条消息🔥🔥):

> `MCP Server 功能, 用于文档的 Brave Search, Custom GPT 的局限性, 使用 MCP 进行代码编辑, Claude Desktop 中的 Prompt 系统`

- **MCP Server 功能仍有改进空间**：用户正在讨论 Custom GPTs 的局限性以及在各种任务中使用 MCP servers 的便利性，并指出 **MCP 对于复杂代码库的编辑能力** 仍可改进。
  
  - 一位成员建议，针对不同用途使用 **不同的 LLMs** 可能会带来优势，而另一位成员指出，将工具与语义选择（semantic selection）相结合可以简化流程。
- **探索使用 Brave Search 获取最新文档**：成员们正在利用 Brave Search 获取文档和依赖项，其中一位成员提到能够抓取文档并将其编译为 Markdown 文件。
  
  - 在实验 **brave-search MCP server** 时，用户分享了有效实现的片段和方法，强调了自动化的潜力。
- **2024 年 Custom GPTs 面临的挑战**：针对 **Custom GPTs** 缺乏改进的问题，人们提出了担忧，因为它们未能有效集成较新的 ChatGPT 功能。
  
  - 讨论内容包括这些模型的潜在贬值，以及对 **Custom GPT 市场** 未达预期的失望。
- **MCP 的代码编辑功能**：成员们对 MCP servers 的 **代码编辑功能** 很感兴趣，讨论了其在处理大型代码库时的局限性以及改进功能的潜力。
  
  - 有想法建议将 MCP servers 转换为 functions 以获得更好的交互，并强调利用 git 版本控制来跟踪更改。
- **在 Claude Desktop 中实现自定义 Prompts**：用户正在苦恼如何通过 Claude Desktop 中的 **prompts/List 端点** 提供 Prompts，并为日志工具寻求清晰的实现方案。
  
  - 成员们分享了资源和示例，以帮助他人应对设置和测试自定义 Prompts 的复杂性。

**提到的链接**：

- [GitHub to Plain Text Converter](https://repo2txt.simplebasedomain.com/)：轻松将 GitHub 仓库转换为纯文本文件。将代码转换为单个格式化的文本文件。
- [来自 Tibor Blaho (@btibor91) 的推文](https://x.com/btibor91/status/1881110210867290191?s=19)：已确认 - ChatGPT macOS 桌面应用具有隐藏选项，可为桌面启动器定义快捷键，以“Toggle Operator”和“Force Quit Operator”。引用 M1 (@M1Astra) OpenAI Ope...
- [GitHub - isaacphi/mcp-language-server: 与 Language Server 交互的 Model Context Protocol (MCP) server](https://github.com/isaacphi/mcp-language-server)：与 Language Server 交互的 Model Context Protocol (MCP) server - isaacphi/mcp-language-server
- [GitHub - alexwohletz/language-server-mcp](https://github.com/alexwohletz/language-server-mcp)：通过在 GitHub 上创建账号来为 alexwohletz/language-server-mcp 做出贡献。
- [由 jspahrsummers 提供的简化的、类似 Express 的 API · Pull Request #117 · modelcontextprotocol/typescript-sdk](https://github.com/modelcontextprotocol/typescript-sdk/pull/117)：受 #116 和生态系统中出现的一些 MCP SDK 封装器的启发，这是一个尝试将更具 Express 风格的 API 引入 SDK 的尝试。这与现有的封装库有所不同...

---

### **MCP (Glama) ▷ #**[**showcase**](https://discord.com/channels/1312302100125843476/1315696461316358175/1331555442366615633) (7 条消息):

> `Apify Actors 的 MCP Server，Anthropic TS 客户端问题，通过 SSE 连接`

- **Apify Actors 的 MCP Server 正在开发中**：[MCP Server for Apify's Actors](https://github.com/apify/actors-mcp-server/) 正在开发中，旨在从各种平台提取数据，但目前仍处于开发阶段。
  
  - 开发者目前在添加动态工具搜索和添加功能方面面临挑战。
- **Anthropic TS 客户端连接挑战**：一位用户报告了在连接 Anthropic TS 客户端时遇到困难，特别是尝试使用 `EventSource` 和 `SSEClientTransport` 但未成功。
  
  - 这被记录为一个已知问题，相关的讨论可以在 [GitHub](https://github.com/modelcontextprotocol/typescript-sdk/issues/118) 上找到。
- **关于正确 SSE URL 的不确定性**：一位用户询问了连接到 MCP Server 的正确 URL，并提到了几种 URL 变体。
  
  - 他们寻求确认指定的 URL 是否对建立连接有效。
- **迁移到替代方案**：鉴于 TS 客户端的问题，一位用户提到选择 Python 作为替代方案。
  
  - 可运行的 Python 代码示例可以在 [这里](https://github.com/apify/actors-mcp-server/tree/master/src/examples) 找到。

**提到的链接**：

- [GitHub - apify/actors-mcp-server: Model Context Protocol (MCP) Server for Apify's Actors](https://github.com/apify/actors-mcp-server/)：适用于 Apify Actors 的 Model Context Protocol (MCP) Server - apify/actors-mcp-server
- [Use custom headers for both the `/sse` and `/message` endpoints · Issue #118 · modelcontextprotocol/typescript-sdk](https://github.com/modelcontextprotocol/typescript-sdk/issues/118)：@chrisdickinson 感谢这个 PR。抱歉，我不擅长 JS。我需要在访问我的 MCP server 时，为 /sse 和 /message 端点都包含一个 API token。我相信 head...

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1331364875838164993) (96 条消息🔥🔥):

> `Ai2 ScholarQA，Project Stargate，Stack Overflow 的衰落，Bespoke-Stratos-32B，AI 基础设施投资`

- **Ai2 ScholarQA 发布**：Ai2 ScholarQA 为文献综述提供了一种实验性解决方案，使研究人员能够针对多篇论文提出问题以进行详细分析，正如其 [博客文章](https://allenai.org/blog/ai2-scholarqa) 中所述。该工具强调为通常需要比较性见解而非单篇论文评估的研究人员提供效率。
  
  - 该平台使用基于 RAG 的提示词工作流，辅助从开放获取论文库中检索信息。
- **特朗普宣布 Project Stargate**：特朗普总统揭晓了 Stargate 项目，这是一项拟议的 5000 亿美元投资，旨在加强美国的 AI 基础设施，目标是创造重大经济效益并确保在 AI 发展中的领导地位。初始投资将包括来自 OpenAI、SoftBank 和 Oracle 的重大贡献。
  
  - 包括 Elon Musk 在内的批评者对资金表示怀疑，质疑财务支持是否如声称的那样雄厚。
- **关于 Stack Overflow 衰落的观察**：有讨论围绕 Stack Overflow 的缓慢衰落展开，理由包括私募股权所有权和内部版主冲突等潜在因素。当前趋势显示流量下降，一些人认为这可能与 AI 的进步没有直接关系。
  
  - 观察者对该平台的未来表示担忧，并提到了可能导致其现状的历史因素。
- **Bespoke-Stratos-32B 介绍**：介绍 Bespoke-Stratos-32B，这是一个从 DeepSeek-R1 蒸馏出来的推理模型，在训练样本显著减少的情况下展现出卓越的推理能力。该数据集已开源，以促进这一新兴领域的协作。
  
  - 这一公告展示了 AI 推理和协作努力方面取得重大进展的潜力。
- **Clay GTM 获得扩张融资**：Clay GTM 宣布以 12.5 亿美元的估值完成 4000 万美元的 B 轮扩张融资，反映了过去几年显著的营收增长。这笔资金旨在支持进一步的发展势头，现有投资者也重申了他们的承诺。
  
  - 这一公告标志着投资者对 Clayton 增长轨迹的强劲信心。

**提到的链接**：

- [Sam Altman (@sama) 的推文](https://x.com/sama/status/1881851602727993711?s=46)：在沙漠中建造丰碑
- [Ai2 ScholarQA](https://scholarqa.allen.ai/)：未找到描述
- [Ai2 ScholarQA](https://scholarqa.allen.ai/query/9d8946c0-756c-4148-b32e-c2d5bc8f8b09)：未找到描述

- [到目前为止我在编写 AI 应用方面的经验总结 | Seldo.com](https://seldo.com/posts/what-ive-learned-about-writing-ai-apps-so-far): 未找到描述
- [来自 @gerry (@Gerry) 的推文](https://x.com/gerry/status/1881464847260639490?s=46): 这是 Trae（字节跳动新推出的 Cursor 竞争对手）的演示视频。今天早些时候他们还没有，但他们动作很快。引用 @gerry (@Gerry) 字节跳动刚刚发布了一个 Cursor 竞争对手...
- [来自 Demis Hassabis (@demishassabis) 的推文](https://x.com/demishassabis/status/1881844417746632910?s=46): 我们对 Gemini 2.0 Flash Thinking 模型的最新更新（在此获取：https://goo.gle/4jsCqZC）在 AIME（数学）和 GPQA Diamond（科学）基准测试中分别获得了 73.3% 和 74.2% 的分数。感谢大家的所有反馈...
- [一个拥有 AGI 的世界会是什么样子？](https://www.strangeloopcanon.com/p/what-would-a-world-with-agi-look): “任何愚者都能知道。重点在于理解。”——阿尔伯特·爱因斯坦
- [来自 Dwarkesh Patel (@dwarkesh_sp) 的推文](https://x.com/dwarkesh_sp/status/1881844437346902297): .@dylan522p 在 2024 年 10 月就预言了。引用 OpenAI (@OpenAI) 宣布 Stargate Project。Stargate Project 是一间新公司，计划在未来四年内投资 5000 亿美元建设新的...
- [来自 Nathan Lambert (@natolambert) 的推文](https://x.com/natolambert/status/1881834984232616029?s=46): Stargate Project 宣布了 —— 在美国为 AI 投入 5000 亿美元的资本支出（CapEx），这对于美国安全和 AI 能力的持续进步来说似乎是一件好事。Google 在 2024 年的全部资本支出为 500 亿美元...
- [来自 Tanay Jaipuria (@tanayj) 的推文](https://x.com/tanayj/status/1881849682063986843?s=46): 哇！Stargate Project 将在未来 4 年内投资 5000 亿美元 —— 这大约占该时期美国 GDP 的 0.4%。作为对比，其他大型项目的通胀调整后支出：• 州际...
- [来自 Georgi Gerganov (@ggerganov) 的推文](https://x.com/ggerganov/status/1882111697198227676): 这是一个轻量级且非常高效的 VS Code 扩展，直接使用 llama.cpp 提供本地 LLM 辅助的代码和文本补全：https://github.com/ggml-org/llama.vscode
- [来自 Noam Shazeer (@NoamShazeer) 的推文](https://x.com/noamshazeer/status/1881845900659896773?s=46): 你们对 Gemini 2.0 Flash Thinking 的反馈非常棒 —— 谢谢！我们采纳了你们的建议并进行了一次实验性更新……
- [介绍 Ai2 ScholarQA | Ai2](https://allenai.org/blog/ai2-scholarqa): Ai2 ScholarQA 提供深入、详细且具上下文的回答，以帮助进行文献综述。
- [来自 Ai2 (@allen_ai) 的推文](https://x.com/allen_ai/status/1881784827063767117): AI 真的能帮上文献综述吗？🧐 认识一下 Ai2 ScholarQA，这是一个实验性解决方案，允许你提出需要多篇科学论文才能回答的问题。它提供更深入、更详细的...
- [来自 adi (@adonis_singh) 的推文](https://x.com/adonis_singh/status/1881787222300786789?s=46): Anthropic 正在弃用 Claude 3 Sonnet。可能是因为他们计划很快发布 4 Sonnet...
- [来自 Varun Anand (@vxanand) 的推文](https://x.com/vxanand/status/1882061978593837344?s=46): 我们宣布以 12.5 亿美元的估值获得 4000 万美元的 B 轮扩张融资。我们上次融资的资金仍未动用，但我们的势头强劲 —— 24 年营收增长 6 倍，22 年和...均增长 10 倍。
- [来自 Gergely Orosz (@GergelyOrosz) 的推文](https://x.com/gergelyorosz/status/1881757832535769332?s=46): Stack Overflow 缓慢而又突然的衰落。全文（续）
- [来自 Zack Jackson (@ScriptedAlchemy) 的推文](https://x.com/ScriptedAlchemy/status/1881897837509902443): 字节跳动发布了一个 Cursor IDE 的竞争对手，名为 Trae。https://www.trae.ai/
- [来自 near (@nearcyan) 的推文](https://x.com/nearcyan/status/1773759331403714779)): Stargate，由 Nvidia ™ 提供
- [来自 OpenAI (@OpenAI) 的推文](https://x.com/openai/status/1881830103858172059?s=46): 宣布 Stargate Project。Stargate Project 是一间新公司，旨在未来四年内投资 5000 亿美元，在美国为 OpenAI 建设新的 AI 基础设施。我们将...
- [来自 Agus 🔎 🔸 (@austinc3301) 的推文](https://x.com/austinc3301/status/1881844683514823043?s=46): 啊，是的。当然，我们要用那个虚构的传送门来命名这个项目，几个敌对的外星文明曾试图通过它入侵并摧毁地球。引用 OpenAI (@OpenAI) 宣布 Stargate...
- [来自 Shawn Lewis (@shawnup) 的推文](https://x.com/shawnup/status/1881458032741400758?s=46): 我们提交的 SWE-Bench 结果已被接受，并正式成为 SOTA！感谢 SWE-Bench 团队制定了如此重要的基准测试。
- [来自 Mahesh Sathiamoorthy (@madiator) 的推文](https://x.com/madiator/status/1882131703927652762?s=46): 介绍 Bespoke-Stratos-32B，这是我们使用 Berkeley NovaSky 的 Sky-T1 配方从 DeepSeek-R1 蒸馏出的推理模型。该模型在推理（数学和代码）基准测试中优于 Sky-T1 和 o1-preview...

- [来自 Jack Altman (@jaltma) 的推文](https://x.com/jaltma/status/1881866713022828907?s=46): 前几天我正和某人聊天，他告诉我他感到压力很大，因为他的姐姐是个学霸，去了医学院等等，我当时就觉得，没错，绝对是这样，我也是。
- [来自 Alex Volkov (Thursd/AI) (@altryne) 的推文](https://x.com/altryne/status/1881946665709613231): 剧情扑朔迷离，而且😅 Larry 难道不是好朋友吗？引用 Alex Volkov (Thursd/AI) (@altryne) 的话：不可思议。Microsoft 投资了 100 亿美元，我们得到了 GPT-4、高级语音模式、视觉功能、o1、o3 以及更多……
- [来自 Jack Rae (@jack_w_rae) 的推文](https://x.com/jack_w_rae/status/1881850277692936233?s=46): 在过去的一个月里，我们从使用 Gemini 2.0 Flash Thinking 的开发者那里收到了很多有用的反馈。今天我们将发布一个更新后的模型，它具有改进的性能和长文本能力……
- [来自 Beff – e/acc (@BasedBeffJezos) 的推文](https://x.com/basedbeffjezos/status/1881837834438627690?s=46): 难道…… OpenAI 刚刚筹集了半万亿？？引用 OpenAI (@OpenAI) 的话：宣布 Stargate Project。Stargate Project 是一家新公司，计划在未来四年内投资 5000 亿美元……
- [来自 Kara 🦇 🔊/🔮 (@0xkarasy) 的推文](https://x.com/0xkarasy/status/1881925843674341685?s=46): 我刚刚用 Google AI Studio 测试了我的一篇论文（超过 40,000 字）。我留下了一个错误的相关系数，猜猜它首先发现了什么。“对于假设 H4 和 H5，你提到……”
- [来自 aaron holmes (@aaronpholmes) 的推文](https://x.com/aaronpholmes/status/1881835490531565826?s=46): Microsoft 表示它不再是 OpenAI 的独家云服务提供商，而是转向“Microsoft 拥有优先拒绝权”的模式，来决定 OpenAI 在何处运行云端业务。正值 O...
- [来自 Gavin Baker (@GavinSBaker) 的推文](https://x.com/GavinSBaker/status/1882081746877063677): Stargate 是个好名字，但 5000 亿美元是一个荒谬的数字，除非 SoftBank 打算卖掉他们所有的 BABA 和 ARM，否则没人会当真。SoftBank 拥有 380 亿美元现金，1420 亿美元债务以及……
- [来自 Eric Simons (@ericsimons40) 的推文](https://x.com/ericsimons40/status/1882106925795696674?s=46): 10 月份，我们仅凭一条推文发布了 @boltdotnew。我们不知道命运会有什么安排，但结果非常疯狂：• 2 个月内 ARR 达到 0-2000 万美元 • 200 万+ 注册用户 • 全球排名第一的 Web AI 代码应用。今天，我们……
- [来自 Smoke-away (@SmokeAwayyy) 的推文](https://x.com/smokeawayyy/status/1881801442459033662?s=46): OpenAI 和 Microsoft 完蛋了。
- [来自 njkumarr (@njkumarr) 的推文](https://x.com/njkumarr/status/1881869401168977937?s=46): 新博客文章！我将 CharacterAI 的一些内存优化实现在 nanoGPT 中，使 KV Cache 大小减少了 40 倍。（链接在回复中）
- [来自 Townhall.com (@townhallcom) 的推文](https://x.com/townhallcom/status/1881833107248361836?s=46): Oracle 的 Larry Ellison：AI 将为每个人设计针对癌症的 mRNA 疫苗——并在 48 小时内通过机器人制造出来。“这就是 AI 的承诺。”
- [Elon Musk 抨击特朗普宣布的 5000 亿美元 AI 项目，声称其支持者“没钱” | CNN Business](https://edition.cnn.com/2025/01/22/tech/elon-musk-trump-stargate-openai/index.html): 未找到描述
- [来自 Greg Brockman (@gdb) 的推文](https://x.com/gdb/status/1881872206101467362?s=46): 感谢特朗普总统今天与我们一起宣布 Stargate Project。5000 亿美元用于为 OpenAI 在美国建设 AI 数据中心。🇺🇸 引用 OpenAI (@OpenAI) 的话：宣布 Stargate Project。Starg...
- [利用 LLM 估计的效用优化预训练数据混合](https://huggingface.co/blog/WillHeld/utilimax-and-medu): 未找到描述
- [唐纳德·特朗普总统宣布 AI 基础设施投资 —— 2025 年 1 月 21 日](https://www.youtube.com/watch?v=zDo_RrzdRoQ): 唐纳德·特朗普总统周二宣布与 OpenAI、Oracle 和 Softbank 成立合资企业，在美国投资数十亿美元建设 AI 基础设施……
- [GitHub - lechmazur/step_game: 多 Agent 步进竞赛基准测试：评估压力下的 LLM 协作与欺骗。这是一个多玩家“步进竞赛”，挑战 LLM 在秘密选择移动步数（1、3 或 5 步）之前进行公开对话。每当有两个或更多玩家选择相同的数字时，所有冲突的玩家都无法前进。](https://github.com/lechmazur/step_game/): 多 Agent 步进竞赛基准测试：评估压力下的 LLM 协作与欺骗。这是一个多玩家“步进竞赛”，挑战 LLM 在秘密选择移动步数（1、3 或 5 步）之前进行公开对话。

---

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/1331714708213989376) (1 条消息):

> `LLM Paper Club, Physics of Language Models, Retroinstruct, Event Notifications, Calendar Integration`

- **加入 LLM Paper Club！**：诚邀参与者加入最新的 LLM Paper Club，本次将讨论 **Physics of Language Models** 和 **Retroinstruct**。活动详情请见[官方链接](https://lu.ma/2d1b6i2t)。
  
  - *请务必查看活动的封面图片*，其中突出了讨论的关键主题。
- **活动日历集成**：鼓励用户通过点击日历右上方侧的 **RSS 标志**将活动集成到自己的日历中。这样可以自动接收 [Latent.Space](http://Latent.Space) 的新活动通知。
  
  - *添加 iCal 订阅*以确保不会错过未来的公告。

 

**提到的链接**：[LLM Paper Club (Physics of Language Models, Retroinstruct) · Zoom · Luma](https://lu.ma/2d1b6i2t)：买一送一的一天！Shamima 将介绍 [https://arxiv.org/abs/2404.05405](https://arxiv.org/abs/2404.05405) 以及这份关于合成数据集的指南……

 

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1331397115561250868) (5 条消息):

> `Proximal Policy Optimization, GRPO implementation, ChatGPT jailbreak possibilities, AI security concerns`

- **Proximal Policy Optimization (PPO) 详解**：PPO 是一种强化学习方法，通过限制更新以确保新策略与旧策略保持接近，从而解决策略学习中的不稳定问题；它具有一定的数学复杂性，可以在[这篇详细文章](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)中深入探索。
  
  - 该文章还讨论了多年来更新的各种策略梯度方法，包括 SAC 和 TD3。
- **GRPO 易于实现**：一位成员观察到，实现 **Generalized REINFORCE with Proximal Optimization (GRPO)** 似乎并不太困难。该方法的可访问性可能会吸引更多开发者参与强化学习。
- **越狱 AI 模型**：一位成员指出，尽管 **r1** 令人印象深刻，但看到它如此轻易地被破解以执行有害任务，令人感到沮丧。
  
  - *当像 ChatGPT 这样先进的模型可以被操纵来编写 DDoS 攻击等活动的代码时，确实令人担忧。*
- **ChatGPT 的安全风险**：人们对 AI 模型被诱导进行恶意活动的能力表示担忧，包括为针对网站的 **DDoS** 攻击编写代码。
  
  - 这突显了 AI 安全的一个重要方面，即即使是先进的模型，如果被误用也会带来风险。

 

**提到的链接**：[Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)：[2018-06-30 更新：增加了两种新的策略梯度方法，SAC 和 D4PG。][2018-09-30 更新：增加了一种新的策略梯度方法，TD3。][2019-02-09 更新：增加了自动调整 te... 的 SAC]

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1331354615404298382) (18 条消息🔥):

> `TMA 实现问题、Persistent Matmul 描述符、TRITON_INTERPRET 行为、Triton Kernel 中的数据依赖、GPU 论文的小组实现`

- **TMA 实现与 Autotune 冲突导致崩溃**：用户报告称，在当前的 **TMA** 实现下，通过配置剪枝（config pruning）手动管理 autotuning 会失败，当使用 `@triton.autotune` 配合多个配置时会导致崩溃。
  
  - 一位用户强调 autotuner 无法与 TMA 协同工作，手动配置是唯一的解决办法。
- **Persistent Matmul 描述符混淆**：描述符的使用存在差异，特别注意到设备端版本使用了 `tl._experimental_make_tensor_descriptor`，而该函数在最新版本中不可用。
  
  - 一位用户分享了一个变通方案，即通过 **torch** 而非 numpy 创建描述符，以避免 **Triton 3.2** 中的错误。
- **TRITON_INTERPRET 改变 Kernel 执行**：一位用户因数据依赖问题在 Triton kernel 中遇到了执行顺序问题，并表示使用 **tl.debug_barrier** 无法解决编译器导致的执行顺序变化。
  
  - 他们注意到只有在 **TRITON_INTERPRET=1** 时才会得到正确结果，强调了潜在的数据类型变化可能会影响执行。
- **操作顺序影响结果**：讨论了 kernel 中操作执行顺序不当的问题，原本应该顺序执行的代码部分在生成的 PTX 代码中被更改了。
  
  - 执行中的偏差与 kernel 在未正确解释时产生错误结果有关，这引发了对编译器如何优化代码的担忧。
- **呼吁合作实现高难度 GPU 论文**：一位用户表示有兴趣汇集资源和专业知识，共同实现有趣且具有挑战性的 GPU 相关论文。
  
  - 该倡议旨在将成员聚集在一起，协作开发以理解复杂的 GPU 算法。

 

**提及的链接**：[GridQuant/scripts/gemm.py at main · niconunezz/GridQuant](https://github.com/niconunezz/GridQuant/blob/main/scripts/gemm.py#L79-L100,)：尝试实现 GridQuant。通过在 GitHub 上创建账号为 niconunezz/GridQuant 的开发做出贡献。

 

---

### **GPU MODE ▷ #**[**cuda**](https://discord.com/channels/1189498204333543425/1189607726595194971/1331390443702849548) (16 条消息🔥):

> `NVIDIA Blackwell 代码生成、在 CPU 上模拟 GPU、即将发布的 Blackwell 白皮书、Accel-Sim 框架、STF 讨论`

- **NVIDIA Blackwell 代码生成见解**：讨论集中在即将发布的 [Pull Request #12271](https://github.com/vllm-project/vllm/pull/12271)，其中详细介绍了 **Blackwell B100/B200 codegen 10.0** 和 **RTX 50 codegen 12.0**。
  
  - *一位用户提到*，他们已经看到了即将推出的 `sm_100a` 和 `sm_101a` 目标的证据，但不确定 `sm_120`。
- **对 GPU 模拟的好奇**：据报道 [LeetGPU.com](https://LeetGPU.com) 正在 **CPU 上模拟 GPU**，这引发了人们的兴趣以及对所用方法的疑问。
  
  - *一位用户指出* [Accel-Sim Framework](https://accel-sim.github.io/#overview) 可能是涉及的工具，并强调了其在模拟和验证可编程加速器方面的能力。
- **即将发布的 Blackwell 白皮书延迟**：一位用户对 **Blackwell 白皮书** 尚未发布表示惊讶，因为目前关于该架构的信息已经在流传。
  
  - 这种情绪反映了人们对新架构发布细节的普遍好奇和期待。
- **对 Accel-Sim 演讲的期待**：*一位成员宣布*将于 3 月底举行关于 **Accel-Sim 框架** 的演讲，这在聊天中引起了热烈反响。
  
  - 社区期待从讨论中了解更多信息，突显了该框架与模拟 GPU 性能的相关性。
- **社区中的 STF 探索**：有人询问是否有人正在尝试 **STF**，表明了对该开发领域的兴趣。
  
  - 这突显了成员们对新技术及其在 GPU 开发中应用的持续探索。

**提及的链接**：

- [LeetGPU](https://LeetGPU.com)：未找到描述
- [Accel-Sim: The Accel-Sim Framework](https://accel-sim.github.io/#overview)：未找到描述
- [johnnynunez 提交的 NVIDIA Blackwell 代码生成 · Pull Request #12271 · vllm-project/vllm](https://github.com/vllm-project/vllm/pull/12271)：Blackwell B100/B200 codegen 10.0，Blackwell RTX 50 codegen 12.0

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1331373071763902474) (6 条消息):

> `Torch Nightly 与 Triton 3.2，Torch Lightning + DeepSpeed Checkpointing，学习率调度器 (Learning Rate Schedulers)，Torch Profiler 运行时间`

- **Torch 与 Triton 3.2 之间的兼容性问题**：一位用户报告称，在 Torch nightly 版本中使用 Triton **3.2** 会导致 **torchao** 崩溃，具体是与 `AttrsDescriptor` 相关的导入错误。
  
  - 错误堆栈显示 Triton 安装版本存在冲突，这体现了管理多个库依赖项的复杂性。
- **DeepSpeed 与 UCP Checkpointing**：一位用户询问 Torch Lightning 中的 DeepSpeed checkpointing 是否包含 **UCP**，或者是否需要从 ZeRO checkpointing 进行手动转换。
  
  - 这突显了 PyTorch 生态系统中不同框架之间的集成问题。
- **寻找有效的学习率调度器**：一位用户询问了确定最佳学习率调度器的资源，并指出该领域有多种选择。
  
  - 社区建议包括 **CosineAnnealing**、**linear warmup + cosine decay**，以及像 **WSD** schedule 这样更具适应性的新方法。
- **在 Torch Profiler 中追踪高级函数时间**：一位用户询问 Torch Profiler 是否可以在进行堆栈追踪（stack tracing）时显示高级函数的运行时间。
  
  - 这反映了对复杂模型详细性能洞察日益增长的需求，特别是围绕 GPU 和 CPU 时间。

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1331374863927087146) (13 条消息🔥):

> `加速 Hugging Face generate()，编程 GPU 推荐，运行大模型的挑战，云端 GPU 租赁选项，GPU 配置的预算考量`

- **寻求 Hugging Face 生成加速**：一位成员询问如何在 trainer 循环中使用 Hugging Face 的 `generate()` 来加速生成，并指出 **liuhaotian/llava-v1.5-7b** 缺乏对 vLLM 的支持。
  
  - 他们参考了一个 [GitHub commit](https://github.com/huggingface/trl/commit/2ecd53ad77ef2a27729176e89299cba37b2487c4) 作为可能的资源。
- **关于编程 GPU 购买的困惑**：一位成员对在 **1500 美元**预算下购买哪种 GPU 进行 **GPU 编程**感到困惑，并提到了 **RTX 4060**。
  
  - 其他人建议选择 **RTX 3060** 或 2 个 **4060Ti**，以便在有限预算内获得更好的性能，并给出了关于模型并行（model parallelism）的建议。
- **云端 GPU 租赁作为便捷选项**：一位成员指出租赁 GPU 是一种具有成本效益的解决方案，并引用了一个 [云端 GPU 对比网站](https://cloud-gpus.com/)。
  
  - 他们强调使用云端 GPU 可以减轻本地配置的压力，特别是处理大型模型时。
- **运行大模型的考量**：讨论中提到了在本地运行 **405b 模型**的挑战，并参考了即使有 32GB RAM 在运行 **70b 模型**时也会遇到的困难。
  
  - 关键点包括拥有足够的**系统 RAM** 的重要性，以避免在切换模型时 Linux 将应用程序交换（swapping）到磁盘。
- **预算与性能的权衡**：讨论显示，选择 **2x4060Ti** 可能会限制其他系统组件的预算灵活性，但允许后续升级。
  
  - 另一条笔记提到 **RTX 3060** 是一个合适的入门级选择，同时也要关注不断上涨的 GPU 价格。

**提到的链接**：[Cloud GPUs](https://cloud-gpus.com/)：未找到描述

---

### **GPU MODE ▷ #**[**pmpp-book**](https://discord.com/channels/1189498204333543425/1194427148656721970/1331376619905880077) (4 messages):

> `PMPP 书籍新内容, PMPP 书籍编程练习, CUDA 编程云端 GPU 对比`

- **PMPP 预期新内容**：成员们对 PMPP 书籍增加大量新内容表示兴奋，指出 **2022 版**中缺失的许多主题将被涵盖。
  
  - *2022 版中缺失的很多内容都将被涵盖*。
- **学习 CUDA 编程的最佳平台**：一位成员询问了实现和测试 PMPP 书籍中编程练习（重点是 CUDA 编程）的最有效方法。
  
  - 建议包括 [Cloud GPU Comparison](https://cloud-gpus.com/) 和 [Lightning.ai](https://lightning.ai/) 等资源，并建议考虑使用 Google Colab 运行 CUDA kernel。

**提到的链接**：

- [Cloud GPUs](https://cloud-gpus.com/)：未找到描述
- [Lightning AI | Turn ideas into AI, Lightning fast](https://lightning.ai/)：AI 开发的一体化平台。协同代码、原型设计、训练、扩展、部署。直接在浏览器中完成，无需配置。由 PyTorch Lightning 的创建者打造。
- [Mark Saroufim (@marksaroufim) 的推文](https://x.com/marksaroufim/status/1739206865106395563)：在 Google Colab 中运行 CUDA kernel！

---

### **GPU MODE ▷ #**[**jax**](https://discord.com/channels/1189498204333543425/1203956655570817034/) (1 messages):

woct0rdho: 为什么 JAX 可以在 SM < 89 的 CUDA 上运行 FP8 操作，而 PyTorch 却不行？

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1331702287231615141) (7 messages):

> `Triton 3.2 问题, torch.compile 失败, AttrsDescriptor API 破坏, 联合 Triton 项目提案`

- **Triton 3.2 导致 torchao 损坏**：新的 Triton 3.2 版本 (`pytorch-triton`) 缺少修复相关问题的关键 commit，特别是破坏了 **torchao** 的导入。
  
  - 强调了一个关于 Triton 3.2 中 [INT8 x INT8 点积失败](https://github.com/triton-lang/triton/issues/5669) 的已知问题。
- **torch.compile 面临同样问题**：影响 **torchao** 的相同破坏也导致了 **torch.compile** 的失败，主要是由于 API 更改导致的导入层级问题。
  
  - 该问题已在 [PyTorch issue #144103](https://github.com/pytorch/pytorch/issues/144103) 中跟踪，涉及 **AttrsDescriptor** 的更改。
- **AttrsDescriptor 频繁发生 BC 破坏**：Triton 中 **AttrsDescriptor** 的近期更改导致了频繁的向后兼容性（BC）破坏，成为开发者的痛点。
  
  - [Pull Request #5512](https://github.com/triton-lang/triton/pull/5512) 详细说明了导致 API 更改的 JIT 重大重构。
- **联合 Triton 项目提案**：一位成员建议，OpenAI 和 Meta 合作建立一个统一的 **Triton 项目** 可以加快开发进度。
  
  - 这一合资项目可能简化贡献流程并加速项目创新。

**提到的链接**：

- [tl.dot 与 INT8 x INT8 损坏 · Issue #5669 · triton-lang/triton](https://github.com/triton-lang/triton/issues/5669)：Bug 描述：当前的 main 分支在处理 INT8 x INT8 tl.dot 时会损坏，该问题在 3.1 版本中不存在。主要有两个问题：acc = tl.dot(a, b, acc=acc) 在 INT8 x INT8 输入时损坏，...
- [[FRONTEND] 由 ptillet 清理 backend/jit 接口 · Pull Request #5512 · triton-lang/triton](https://github.com/triton-lang/triton/pull/5512)：这是对 JIT 的一次相当大的重构，为支持具名元组（named tuples）铺平了道路（将在后续 PR 中推出）：修复了元组特化哈希中的 bug，通过始终采用...简化了启动器。
- [更新 TorchInductor 以支持上游 Triton 中移除的 AttrsDescriptor · Issue #144103 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/144103)：triton-lang/triton#5512 移除了 TorchInductor 在其输出代码中生成的 AttrsDescriptor。为了支持该 PR 之后的 Triton 版本，我们需要更新生成的代码。抄送 @ezyang @g...

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1331615159684173864) (9 messages🔥):

> `Pexels API 使用, Pixabay 作为替代方案, 图像获取限制, 自动化查询担忧`

- **Swaystar 质疑 Pexels 数据集**：一名成员询问在 API 限制（每小时 **200 次请求**）的情况下，是如何编译 **Pexels 图像** 数据集的，并指出该限制的存在。
  
  - *嗯，所以每小时 1.6 万张？* 对在如此限制下编译 10 万张图像的可行性表示担忧。
- **Gau.nernst 解释 Pexels API**：另一位成员分享说，通过使用 **search API**，每次请求最多可以获取 **80 个图像 URL**，这使得任务变得可行。
  
  - 他们提到之前曾尝试从 **Pexels** 抓取数据。
- **Pexels 可能来源于 Pixabay**：Swaystar 推测 **Pexels** 可能从 **Pixabay** 获取图像，后者的限制明显更高，为 **每 60 秒 100 次请求**。
  
  - 这将允许潜在更快的下载速度，引发了关于其优势的讨论。
- **Pixabay API 能力**：Swaystar 发现 **Pixabay 的 API** 每次请求允许 **200 个 URL**，这表明可以更快地访问其包含 **500 万张图像** 的完整目录。
  
  - 然而，考虑到明确禁止自动化查询，人们对这种做法的伦理影响表示担忧。
- **关于自动化查询的担忧**：有人指出 **Pixabay API** 是为 **真实人类请求** 设计的，并警告说不允许系统性的大规模下载。
  
  - 这引发了关于在持续重度使用 API 后被封禁风险的问题。

 

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1331643317246623765) (4 messages):

> `Triton 直播, 加速 LLM 推理, LeetGPU 更新`

- **Triton 直播冒险**：一位成员目前正在 *学习 Triton* 并直播该过程，未来几天将重点关注 **flash-attention 2 的 backward pass**。
  
  - 他们计划在月底前发布一系列 **Triton 教程**，可在其 [YouTube 频道](https://www.youtube.com/@Tunadorable)观看。
- **加速 LLM 推理的见解**：一位成员分享了一篇名为 [Accelerating LLM Inference](https://pytorch.org/blog/accelerating-llm-inference/) 的博客文章链接，讨论了 LLM 性能方面的进展。
  
  - 该文章重点介绍了优化大语言模型推理速度和效率的技术。
- **加入 LeetGPU 社区**：另一位成员鼓励对 **LeetGPU.com 更新** 和支持感兴趣的人加入专门的 Discord 服务器。
  
  - 他们分享了一个社区邀请链接，成员可以在那里讨论 GPU 技术的最新动态。

 

---

### **GPU MODE ▷ #**[**arc-agi-2**](https://discord.com/channels/1189498204333543425/1316377974672588850/1331406479298789466) (10 条消息🔥):

> `GRPO Algorithm Implementation, Kimi-k1.5 Paper Discussion, Curriculum Learning in RL, Tiny GRPO Repository, RL-hyped Experimentation`

- **基础 GRPO 实现接近完成**：一名成员确认 **GRPO 算法** 的最简版本已实现，第一个版本预计明天即可运行。
  
  - 他们表示在清理代码的同时，初步实验正在进行中。
- **Kimi-k1.5 论文凸显 RL 进展**：讨论集中在 **Kimi-k1.5** 论文上，该论文因引入了 **Curriculum Learning** 和长度惩罚（length penalty）以增强强化学习效果而受到关注。
  
  - 成员们分享了 [论文链接](https://github.com/MoonshotAI/Kimi-k1.5/blob/main/Kimi_k1.5.pdf) 以供进一步深入了解。
- **TRL 中部署 GRPO Trainer**：TRL 库中引入了新的 **GRPO trainer**，使得 Transformer 模型的强化学习更加易于实现。
  
  - 可以在 [GitHub 仓库](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) 中查看该 trainer 的实际应用。
- **Tiny GRPO Playground 仓库发布**：一名成员将 **grpo-playground** 移至名为 [tiny-grpo](https://github.com/open-thought/tiny-grpo) 的专用仓库，专注于极简的 GRPO 实现。
  
  - 该仓库旨在提供一个简单且易于修改（hackable）的实验版本，并获得了社区的积极反馈。
- **对 Tiny GRPO 实验的期待**：社区成员对在数学数据集上运行 **tiny_grpo train.py** 脚本表现出极高热情，并赞赏这种直观的实验方法。
  
  - 他们注意到仓库中有一个数学数据集，将在工作之余进行探索。

**提到的链接**：

- [GitHub - open-thought/tiny-grpo: Minimal hackable GRPO implementation](https://github.com/open-thought/tiny-grpo)：极简且易于修改的 GRPO 实现。欢迎通过在 GitHub 上创建账号为 open-thought/tiny-grpo 的开发做出贡献。
- [Kimi-k1.5/Kimi_k1.5.pdf at main · MoonshotAI/Kimi-k1.5](https://github.com/MoonshotAI/Kimi-k1.5/blob/main/Kimi_k1.5.pdf)：欢迎通过在 GitHub 上创建账号为 MoonshotAI/Kimi-k1.5 的开发做出贡献。
- [arc-agi-2/arc-1/tiny_grpo at main · open-thought/arc-agi-2](https://github.com/open-thought/arc-agi-2/tree/main/arc-1/tiny_grpo)：构建解决 ARC-AGI-2 的认知核心。欢迎通过在 GitHub 上创建账号为 open-thought/arc-agi-2 的开发做出贡献。
- [trl/trl/trainer/grpo_trainer.py at main · huggingface/trl](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)：使用强化学习训练 Transformer 语言模型。- huggingface/trl

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1331421501802418267) (26 条消息🔥):

> `Colab 工作流挑战、AI 播客推荐、Google Titans 模型见解、租用服务器数据传输问题、Grokking 与数值稳定性发现`

- **Colab 与数据同步的困扰**：成员们讨论了将本地工作同步到 **Colab** 的挑战，提到了 **VSCode remote** 等选项以及向租用服务器传输数据的限制。
  
  - 有人指出，像 **rsync** 这样的工具和租用 GPU 服务可以简化流程，但数据可访问性问题依然存在。
- **AI 播客推荐**：为了在碎片化时间进行休闲学习，成员们推荐了 **Latent Space**、**Cognitive Revolution** 和 **Machine Learning Street Talk**。
  
  - 这些播客因其易于理解且包含丰富的 AI 相关主题内容而受到推荐。
- **对 Google Titans 模型的反应**：成员们分享了对 **Google Titans** 的见解，该模型声称通过在推理阶段引入新的内存特性，表现优于 Transformer。
  
  - 然而，大家一致认为，由于论文中展示的方法论较为复杂，复现其结果可能会很困难。
- **Grokking 研究见解**：讨论集中在最近一篇关于 **Grokking** 的论文上，重点关注数值问题如何影响模型训练和稳定性。
  
  - 提出了一种新的 Optimizer 策略，作为克服这些数值稳定性挑战的潜在解决方案。
- **ML 初学者寻求指导**：一位新成员表达了对 **ML research** 的兴趣，并在社区中寻求指导以获取职业生涯的建议。
  
  - 现有成员建议去其他对初学者更友好的服务器，因为本社区并非特别面向新手。

**提到的链接**：

- [Do weights update less towards the start of a neural network?](https://stats.stackexchange.com/questions/660387/do-weights-update-less-towards-the-start-of-a-neural-network)：即，由于误差来自神经网络的末端（即输出层）并通过 Backpropagation 回传到神经网络的开头，这是否意味着权重更新会……
- [Grokking at the Edge of Numerical Stability](https://arxiv.org/abs/2501.04697)：Grokking，即在长时间过拟合后突然出现的泛化现象，是一个挑战我们对深度学习理解的惊人现象。尽管在理解……方面已取得显著进展。
- [GitHub - LucasPrietoAl/grokking-at-the-edge-of-numerical-stability](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability)：通过在 GitHub 上创建账号来为 LucasPrietoAl/grokking-at-the-edge-of-numerical-stability 的开发做出贡献。

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1331387345680732301) (16 messages🔥):

> `DeepSeek Reward Model 架构, 从 Egomotion 中学习视觉, 具有可微更新的参数化 Loss Functions, 技能学习中的 Domino effect, 高效 Linear Attention 机制`

- **调查 DeepSeek Reward Models**：成员们表达了对理解 **DeepSeek** 如何训练其 **Reward Models** 及其架构的兴趣，尽管尚未分享具体发现。
  
  - 记录了对进一步探索讨论优化和学习范式的相关论文的兴趣。
- **从 Egomotion 中学习 Feature Extraction**：一位成员分享了一篇论文，提议使用 **Egomotion** 作为监督信号来学习有用的视觉特征，展示了与传统类标签监督相比具有竞争力的结果。
  
  - 这种方法有可能简化特征学习，摆脱对大型标记数据集的依赖。
- **参数化 Loss Functions 与编译器**：讨论集中在创建一个系统，其中 **Loss Functions** 和 **Update Rules** 是参数，允许通过编译实现自动化的模型优化。
  
  - 成员们建议像 **Jax/XLA** 这样的系统可以实现这些想法的高效执行，从而提高可访问性和性能。
- **理解技能学习中的 Domino effect**：一位成员分享了一篇论文的见解，详细介绍了 **Domino effect**，即神经网络中的技能是按顺序学习的，从而影响后续学习。
  
  - 讨论围绕提议的模型展开，这些模型验证了这种效应，并强调了神经网络中不同的学习行为。
- **高效 Linear Attention 机制**：成员们讨论了一种通过在模型架构中集成可微组件来实现高效 **Linear Attention** 的方法。
  
  - 提议的架构旨在增强性能，同时允许在优化和并行化方面具有灵活性。

**提到的链接**：

- [FOCUS: First Order Concentrated Updating Scheme](https://arxiv.org/abs/2501.12243)：**LLM** 展示了卓越的性能，改进其预训练过程似乎是进一步增强其能力的关键。基于记录的成功...
- [Test-time regression: a unifying framework for designing sequence models with associative memory](https://arxiv.org/abs/2501.12352)：序列提供了一种极其通用的方式来表示和处理信息。这种强大的抽象使序列建模处于现代深度学习应用的核心，激发了...
- [Physics of Skill Learning](https://arxiv.org/abs/2501.12391)：我们旨在理解技能学习的物理学，即在训练期间神经网络如何学习技能。我们从观察 **Domino effect** 开始，即技能是按顺序学习的，而不是...
- [Learning to See by Moving](https://arxiv.org/abs/1505.01596)：计算机视觉中特征学习的主流范式依赖于使用数百万张手工标记的图像针对物体识别任务训练神经网络。是否可能学习有用...

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1331365100417847297) (29 条消息🔥):

> `Minerva Math, Math-500 Dataset, DeepSeek AI Performance, Chat Template Requirements, Long Context Tasks`

- **Minerva Math 能力探索**：发起了一场关于尝试 `minerva_math` 的讨论，该工具使用 sympy 实现了具有答案等效性的 MATH 评估。
  
  - 补充提到 **Math-500** 是 OpenAI 为其 “Let's Think Step by Step” 论文创建的一个子集。
- **DeepSeek AI 性能评估**：对比了 **DeepSeek-R1-Distill-Qwen-1.5B** 模型的两次评估结果，分别为 **0.834 (n=2)** 和 **97.3 (n=64)**。
  
  - 提供了一个链接以供进一步参考该模型的性能，并讨论了解码策略（decoding strategies）。
- **了解 Math-500 评估需求**：寻求澄清 **R1** 在 Prompting 过程中是需要 Chat Template 还是像 Base 模型一样运作。
  
  - 参与者表示，在没有适当评估参考的情况下，对转换过程尚不确定。
- **Long Context 任务与 Ruler 任务**：一名成员宣布增加了各种 Ruler 任务，并讨论了格式化所需的收尾工作。
  
  - 他们寻求其他可支持的高效 Long Context 任务的建议。
- **关于 DeepSeek AI 链接的反馈**：围绕指向 DeepSeek AI 研究材料的 **GitHub** 链接展开的讨论确认，第 **14** 页包含相关的性能指标。
  
  - 总体观点表明，性能数据接近预期结果，使用贪婪解码（greedy decoding）测得结果为 **0.722**。

**提到的链接**：

- [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B · Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)：未找到描述
- [Build software better, together](https://github.com/EleutherAI/lm-evaluation-harness/pull/2556)：GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
- [DeepSeek-R1/DeepSeek_R1.pdf at main · deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)：通过在 GitHub 上创建账号来为 deepseek-ai/DeepSeek-R1 的开发做出贡献。
- [HuggingFaceH4/MATH-500 · Datasets at Hugging Face](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)：未找到描述

---

### **Eleuther ▷ #**[**gpt-neox-dev**](https://discord.com/channels/729741769192767510/730090096287547444/1331395068925579276) (2 条消息):

> `Exporting model to HF format, RuntimeError during conversion, Multi-node training configuration`

- **导出模型为 HF 格式时报错**：一位用户报告在尝试使用 [convert_neox_to_hf.py](https://github.com/neox/convert_neox_to_hf.py) 导出模型时遇到 `RuntimeError: shape '[8, 512, 4096]' is invalid for input of size 4194304`。
  
  - 该错误发生在权重来自 `model_parallel_size=4` 的 2 节点 SFT 运行时，引发了关于多节点运行兼容性的疑问。
- **关于需要进一步见解的讨论**：另一名成员寻求帮助，并艾特了一位用户以获取有关转换问题的见解，显示出社区在故障排除方面的协作。
  
  - 这表明了一个协作环境，用户向同行寻求解决技术问题的帮助。

**提到的链接**：

- [{](https://rentry.co/f4tvoevf): &quot;pipe_parallel_size&quot;: 0, &quot;model_parallel_size&quot;: 4, &quot;make_vocab_size_divisible_by&quot;: 1, # model settings &quot;num_layers&quot;: 32, &a...

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1331373775698001951) (44 messages🔥):

> `DeepSeek 功能, 网络安全中的 AI, 用户对 AI 模型的参与度, AI 投资预期, AI 工具的可访问性`

- **DeepSeek 在各种应用中证明了其多功能性**：用户正在探索 DeepSeek 的能力，强调了它在 GitHub issues 和数学问题相关的基准测试中，相较于 o1 和 Sonnet 等其他模型的表现。
  
  - 一位用户指出，可以通过 DeepSeek 的官网免费访问，并能通过 API 集成到不同平台。
- **AI 在网络安全中的作用仍讨论不足**：成员们对 AI（特别是生成式 AI）在网络安全应用中的潜力表示兴趣，包括入侵检测系统和智能自动化。
  
  - 讨论显示，像 CrowdStrike 这样的公司利用机器学习已经有相当长一段时间了。
- **AI 模型的使用体验问题**：几位成员报告了在使用 o1 模型时遇到的困难，表明对其功能和可访问性的担忧日益增加。
  
  - 成员们建议联系特定的用户群体，以获取有关访问 DeepSeek R1 等功能的针对性帮助。
- **对企业动机的怀疑**：一位成员评论了 AI 领域企业利益与终端用户需求之间的脱节，质疑企业的责任。
  
  - 这一观点反映了对于利润优先于消费者福利的持续不满。
- **讨论了 AI 行业的投资预期**：对话暗示了 AI 领域的重大资金投入，引发了关于这些投资预期管理的更广泛讨论。
  
  - 一位用户讽刺地评论了企业对终端用户的漠不关心，嘲讽了过时的企业态度。

 

**提到的链接**：[Sade - Smooth Operator - Official - 1984](https://youtu.be/4TYv2PhG89A)：Sade – Smooth Operator，导演 - Julien Temple - 1984年9月。英国标志性乐队 Sade 的官方 YouTube 频道 [www.sade.comSade](http://www.sade.comSade) (主唱) Stuar...

 

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1331463206178390049) (2 messages):

> `使用图像训练的自定义 GPT, API 中的文件上传功能`

- **探索使用图像训练的自定义 GPT**：一位成员询问如何创建一个仅使用**图像**作为训练数据的自定义 GPT，特别是**聊天对话的截图**。
  
  - 目标是让模型在给定对话截图时，能以类似于训练数据的方式进行响应。
- **关于 Chat API 中文件上传的疑问**：另一位成员询问 Chat Completion API 请求中是否存在**文件上传功能**。
  
  - 这引发了关于在聊天交互中集成不同数据类型的讨论。

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1331665689315905588) (6 messages):

> `OCR 示例的影响, 使用 OCR 读取地图, OpenAI O 系列模型的改进`

- **关于 OCR 示例导致幻觉的担忧**：一位成员指出，在 OCR Prompt 中使用示例可能会导致**幻觉**和误解，这与预期目标背道而驰。
  
  - 他们强调，这个问题在**非受限环境**中尤为严重。
- **读取地图的变通方法**：一位成员分享了他们读取地图的用例，并强调他们已经找到了应对当前 OCR 局限性的变通方法。
  
  - 他们表示希望 **OpenAI 的模型**在处理**空间或 GIS 数据集**方面能有所改进。
- **领域约束对 OCR 有效性的影响**：一位成员承认 OCR 在地图等受限领域可能是有效的，但指出了在**非受限语境**下的风险。
  
  - 他们指出了当不加区分地使用示例时，可能会发生**上下文污染**。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1331665689315905588) (6 messages):

> `OCR 与幻觉, 地图绘制用例, OpenAI O 系列模型`

- **OCR 示例可能导致幻觉**：一位成员指出，为读取 OCR 提供示例并没有帮助，反而会根据这些示例*促发幻觉*。
  
  - 成员担心在非受限空间中使用示例可能会*污染上下文*。
- **地图绘制用例找到变通方法**：另一位成员提到他们有一个读取地图的用例，并且在等待 OpenAI 模型改进的过程中，*目前已经找到了变通方法*。
  
  - 他们表示希望 OpenAI 的 O 系列模型在处理**空间**或**GIS 数据集**方面会变得更好。

 

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1331382808379129878) (10 条消息🔥):

> `NotebookLM 在教会服务中的应用, NotebookLM 学习工作流, 音频内容生成问题, 使用 Gemini 进行 Prompt 优化, CompTIA A+ 资源创建`

- **NotebookLM 彻底改变了教会服务**：一位用户报告了使用 **NotebookLM** 的出色成果，通过分析 **16 个时长 5 小时** 的 YouTube 直播转录文本，将其应用于大型教会会议。
  
  - 他们强调 NotebookLM 帮助创建了详细的会议报告，并正在编写一本 **250 页的书籍** 和一份 **2000 页的圣经研究资料**。
- **NotebookLM 开启学习常规的新篇章**：一位成员分享了他们使用 **NotebookLM** 的第三周体验，将其整合到学习工作流中，并发现其价值不可估量。
  
  - 他们还分享了一个 [YouTube 视频](https://youtu.be/wvf4EXJJsU8) 链接，讨论了他们使用该工具的历程。
- **对重复音频内容的担忧**：一位用户在从 **PDF 源文件** 生成音频后表达了沮丧，注意到内容在三个片段中出现了不必要的重复。
  
  - 他们寻求关于如何在未来的音频生成中防止这种重复的建议。
- **优化 Prompt 以获得更好结果**：在关于有效 Prompt 的讨论中，一位成员建议利用 **Gemini** 来创建和优化指令，以获得更好的结果。
  
  - 这种方法被推荐为使用 NotebookLM 获取高质量成果的首选策略。
- **CompTIA A+ 资源的创建**：一位用户宣布他们创建了 **CompTIA A+** 系列的第一部分，并打算很快上传后续部分，同时分享了一个音频文件链接。
  
  - 这一举措符合社区分享与技术认证相关的教育资源的兴趣。

 

**提到的链接**：[我们需要谈谈 NotebookLM](https://trendingcommunicator.substack.com/p/we-need-to-talk-about-notebooklm)：它是你 AI 策略中缺失的一环吗？

 

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1331355879643287573) (43 条消息🔥):

> `APA 引用生成, NotebookLM 定制化, 音频概览生成, 用于 Prompt 的 Chrome 扩展, 回复的创造力`

- **APA 引用生成方面的帮助**：用户讨论了让 NotebookLM 从添加的源文件中生成 APA 格式的引用列表，但发现它仅引用之前使用过的材料。
  
  - 建议包括重命名源文件以便于引用，以及创建指令以确保正确的格式化。
- **定制 NotebookLM 交互**：一位用户询问是否可以给 NotebookLM 设定通用规则以保持特定的回复格式，但在其记忆功能上面临限制。
  
  - 另一位用户建议在 Chrome 插件中保存 Prompt，以便在对话中快速重复使用。
- **生成新的音频概览**：用户想知道在添加更多源文件后是否可以生成新的音频概览，确认 NotebookLM 允许每天最多生成三次。
  
  - 讨论指出需要删除旧音频以生成新版本，并可以自定义关注点。
- **使用 Chrome 扩展提高效率**：一位用户推荐了一个名为 Simple Prompt Manager 的 Chrome 扩展，用于快速保存和重复使用 Prompt。
  
  - 该扩展专门用于 Chrome 和 Edge 浏览器，以更有效地管理 Prompt。
- **AI 回复的可变性**：尽管之前的交互更具动态性，但用户对 NotebookLM 回复的创造力和可变性下降表示担忧。
  
  - 有建议使用 Gemini 以获得更多创意，但用户指出与 NotebookLM 相比， Gemini 难以专注于特定文档。

 

**提到的链接**：[您的业务如何利用人工智能 (AI)？](https://www.pollgen.com/polls/j577dgrspp5tnakct19gzrv9dh78xxwy)：我们正在收集关于各种业务如何将 AI 融入其运营的数据。您的见解将帮助我们了解 AI 在不同领域的各种应用。

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1331411232036487369) (53 messages🔥):

> `De-distilled flux models performance, AI art public perception, Discord bot scams, CitivAI maintenance, Fixing faces in swarmUI`

- **De-distilled flux 模型更受青睐**：一位用户表示，尽管 de-distilled flux 模型比 undistilled 模型慢，但在 cfg 设置下表现更好。
  
  - 另一位用户提到，使用 negative prompts 可以提高 prompt 遵循度，但指出这会增加处理时间。
- **AI 艺术引发褒贬不一的反应**：用户讨论了围绕 AI 艺术的负面情绪，一些人报告了在使用 AI 工具时遇到的敌对反应。
  
  - 一位用户幽默地提到，因为使用 AI 被告知“去死吧”，反映了持续多年的强烈观点。
- **对 Discord bot 诈骗的警惕**：几位用户分享了收到来自潜在 bot 账号索要个人信息的异常私信（DMs）的经历。
  
  - 一位用户讲述了之前遇到过提供付费服务的情况，强调了此类诈骗的普遍性。
- **CitivAI 遇到维护问题**：一位用户询问关于 CitivAI 停机的情况，指出它经常一天进行多次维护。
  
  - 这一观察引发了其他用户关于可用性的进一步提问。
- **开始在 swarmUI 中修复面部**：一位用户寻求关于如何在 swarmUI 中开始修复面部的建议，询问是否需要 refiner。
  
  - 这场对话强调了社区内对改进图像生成技术的共同兴趣。

 

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1331681397240233994) (3 messages):

> `AgentWorkflow release, DeepSeek-R1 model, Open-source RAG system guide`

- **为 Multi-Agent 系统推出 AgentWorkflow**：今日重大发布！我们很高兴推出 [AgentWorkflow](https://twitter.com/llama_index/status/1882121805542170894)，这是一个在 LlamaIndex 中创建 Multi-Agent 系统的全新高级系统。
  
  - 该系统建立在 LlamaIndex Workflows 强大的低层级构建块之上，这些构建块已在我们的社区中引起共鸣。
- **DeepSeek-R1 脱颖而出**：凭借与 OpenAI o1 相当的性能，**DeepSeek-R1** 是目前最热门的模型，你今天就可以在 LlamaIndex 中使用它！看看我们的朋友 [@getreflex](https://twitter.com/getreflex) 如何使用该模型构建全栈 RAG 聊天应用。
  
  - 在分享的 [推文](https://twitter.com/llama_index/status/1882144637558890996) 中了解更多关于此集成及其功能的信息。
- **构建开源 RAG 系统指南**：探索如何使用 LlamaIndex、Meta Llama 3 和 [@TruLensML](https://twitter.com/TruLensML) 构建和评估开源 RAG 系统！这份详细指南比较了基础 RAG 系统与 agentic 变体，并使用 @neo4j 进行数据存储。
  
  - 它还评估了 **OpenAI** 和 **Llama 3.2** 之间的性能差异，为寻求优化系统的开发者提供了宝贵的见解。

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1331369529414123652) (46 messages🔥):

> `LlamaIndex 文档网站 Bug，Gemini 的 Cached Augmented Generation，针对 Python 文件对象的自定义 Reader，LlamaIndex 中的领域特定向量存储，AgentWorkflow 并行调用`

- **用户反馈 LlamaIndex 文档网站 Bug**：一位遇到问题的用户发现，在使用 **Microsoft Edge** 时，LlamaIndex 文档网站会间歇性地自动滚动到顶部。
  
  - 切换到**无痕模式 (incognito mode)** 解决了该问题，这表明可能与浏览器扩展存在冲突。
- **Gemini 的 Cached Augmented Generation (CAG) 受到限制**：一位用户询问关于在 **Gemini** 上实现 **Cached Augmented Generation (CAG)** 的问题，但其他人指出这需要模型层级的访问权限，而目前的 API 尚未提供。
  
  - 这意味着任何想要实现 CAG 的人可能需要利用替代方法或自定义配置。
- **在 LlamaIndex 中创建自定义 Reader**：一位新用户寻求关于构建处理 Python 文件对象的 `Reader` 的指导，最终建议继承 **BaseReader** 并重写 `load_data` 方法。
  
  - 这种方法通过直接处理内存中的数据，避免了不必要的文件 I/O。
- **领域特定向量存储的实现寻求**：关于处理医疗数据的讨论指出，可以通过**为 Node 添加元数据标签**来创建特定类别的索引，以便根据用户查询更好地检索来源。
  
  - 一个链接的 gist 被重点标注为资源，详细介绍了使用类似方法的先前实现。
- **AgentWorkflow 中的并行调用**：一位用户询问关于在 **AgentWorkflow** 中管理并行 Agent 调用的问题，但被告知 Agent 一次只能运行一个，而工具调用如果是异步的则可以并行运行。
  
  - 建议将 Workflow 进行嵌套，作为实现流程并行化的一种潜在变通方案。

**提到的链接**：

- [Node Parser Modules - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#codesplitter)：未找到描述
- [LlamaIndex - LlamaIndex](https://docs.llamaindex.ai/)：未找到描述
- [Knowledge Graph Index - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphDemo/)：未找到描述

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1331372409944801281) (6 messages):

> `社区展示，论坛 vs Discord 讨论，项目分享的清晰度，Nightly 版本开发进展`

- **双平台的社区展示**：一位成员建议将项目同时发布在两个平台的社区展示中，并指出大部分 **Mojo 社区** 成员主要集中在这个 Discord 服务器中。
  
  - 这种方法将有助于提高在两个领域的曝光度和参与度。
- **论坛属性适合长期讨论**：有人指出**论坛更适合**需要长期存档的讨论，因为 Discord 中重要的讨论内容往往难以定位。
  
  - 强调了对向 Modular 员工提出的请求以及关于语言/标准库设计的对话进行去重（de-duplication）的必要性。
- **明确平台间的项目分享**：为了提高清晰度，建议在论坛分享项目，以解决跨平台类别重复的问题。
  
  - 目标是明确哪个平台适用于特定类型的内容，从而简化沟通。
- **论坛允许更深层的处理时间**：一位成员表示更倾向于论坛，因为与节奏较快的 Discord 相比，论坛允许更多时间来阅读和处理信息。
  
  - 这突显了在快速沟通与深度讨论之间取得平衡的需求。
- **Nightly 版本开发正在进行中**：简要提到了 **Nightly** 版本正处于活跃状态并持续推进。
  
  - 未提供更多细节，表明这可能是未来更新的一个关注点。

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1331382141585457212) (29 messages🔥):

> `Mojo 域名观察、MLIR 并行化、Rust Work-Stealing 调度器、Mojo 函数重写、异步编程挑战`

- **Mojo 可能不会与 Modular 分离**：一位成员指出，像 Python 这样的编程语言为其社区使用 .org 域名，并好奇 Mojo 是否也会采用这种方式。
  
  - 另一位成员确认，目前没有计划将 Mojo 从 modular.com 域名中分离出来。
- **MLIR 级别并行化的优势**：有人指出，**MLIR** 级别的并行化比仍在开发中的 LLVM 具有更大的潜力。
  
  - 使 LLVM 实现可并行化的持续工作标志着一项重要的技术进步。
- **Rust 在异步调度方面的挑战**：Rust 社区讨论了 **Work-Stealing 调度器**、线程安全与用户人体工程学（Ergonomics）之间的内在冲突，这导致了诸如不能在 yield points 跨越时持有 Mutexes 等限制。
  
  - 一位成员表示，虽然 Work-Stealing 可能会使人体工程学变得复杂，但对任务管理进行更细粒度的控制可能会产生更好的结果。
- **Mojo 函数重写（Overriding）机制**：讨论了 Mojo 是否有用于 Structs 中函数重写的 `@override` 装饰器。
  
  - 成员们澄清说，虽然 Mojo 允许函数重写，但它缺乏特定的装饰器，因为 Structs 不具备继承（Inheritance）特性。
- **异步编程结果提取问题**：一位成员表示希望有一个标准库，能够高效处理多种 Future 类型、Channels 以及混合 I/O 操作。
  
  - 他们强调了像 Wakers 这样现有的机制虽然提供了一些功能，但强调需要更类型安全（Type-safe）的结果提取方式。

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1331394393147445248) (34 messages🔥):

> `SunoMusic 音频输入功能、音频字幕挑战、音频数据集项目、情感开源 TTS、高中教师志愿者`

- **SunoMusic 让你能用自己的声音创作歌曲**：SunoMusic 的功能允许用户录制自己唱歌或演奏乐器的声音，并根据这些音频输入创作独特的歌曲，如[这条推文](https://x.com/SunoMusic/status/1881742789639057828)所示。
  
  - “你用这个功能创作了什么？”是其号召性用语，邀请用户探索他们的创造力。
- **音频字幕（Audio Captioning）面临挑战**：由于缺乏足够的数据来根据志愿者反馈评估**背景噪声**和**录音质量**，音频字幕方面的工作受到阻碍。
  
  - 提议的解决方案包括编写代码，为现有的音频数据集程序化地添加背景噪声。
- **探索音频数据集项目**：一位成员透露他们参与了包括配音、视频剪辑 Embeddings 以及来自科学论文的知识图谱在内的开源数据集工作。
  
  - 该团队乐于接受贡献，并表示他们目前在资源有限的情况下处理各种数据集项目。
- **情感化开源 TTS 即将到来**：情感化文本转语音（TTS）即将取得突破，并很快将在 Bud-E 中可用。
  
  - 分享的一个音频样本展示了这种情感化 TTS 技术的独特能力。
- **平衡教学与 AI 项目**：一位成员在业余时间协调多个 AI 数据集项目，同时兼顾高中教师的工作。
  
  - 尽管收到了工作邀请，他们仍更喜欢目前角色的独立性，管理着音频和视频数据集方面的志愿者工作。

**提到的链接**：

- [audio_augmentations.ipynb](https://drive.google.com/file/d/1uhQ22wW7H5aCABYI1kfMtyaNVl9cUjWm/view?usp=sharing): Colab 笔记本
- [来自 Suno (@SunoMusic) 的推文](https://x.com/SunoMusic/status/1881742789639057828): 录制你唱歌、弹钢琴或敲铅笔的声音 + 上传到 Suno，用你自己的声音创作属于你的歌曲 😱 你用我们的音频输入功能创作了什么？🎤: @techguyver 展示了...

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1331444497770352723) (25 条消息🔥):

> `用于 LLM 的 OpenAI API、文本生成模型、ML 研究协助、Cohere 模型开发、图像转视频生成`

- **OpenAI API 集成讨论**：成员们讨论了为个人 LLM 平台创建 OpenAI API，强调了指定 OpenAI 端点的实用性，并以 [DeepSeek](https://api.deepseek.com) 为例。
  
  - 对话包括演示如何实现用于生成聊天响应的 API 代码片段。
- **关于最佳文本生成模型的咨询**：一位成员询问了关于最佳文本生成模型的建议，提到了 **Command-R7b** 和 **Aya Expanse 32B** 等建议。
  
  - 参与者分享了他们对适合文本生成模型的见解，展示了多样化的偏好。
- **研究协助请求**：一名高中生寻求帮助以验证其 Machine Learning 研究方法，希望有经验的成员提供反馈。
  
  - 这表明了社区内对同行支持和知识共享的积极兴趣。
- **Cohere 模型发布建议**：有人建议 Cohere 发布集成思考和逻辑能力的 LCoT meme 模型权重。
  
  - 尽管如此，另一位成员指出 Cohere 主要专注于企业级解决方案，暗示了不同的战略方向。
- **图像转视频生成的探索**：一位用户分享了他们在图像转视频生成模型方面的工作，表明了在 Machine Learning 领域持续进行的个人项目。
  
  - 这突显了成员参与项目的多样性以及他们对探索创新技术的承诺。

 

**提到的链接**：<a href="[https://api.deepseek.com")">未找到标题](https://api.deepseek.com%22)%22%3Eno) ：未找到描述

 

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1331501906299850752) (1 条消息):

> `频道关停公告、支持流程简化、新模型/API 问题`

- **频道关停方案**：发布了一项公告，该频道将在 **2 周内关停**，以简化支持并改进流程。
  
  - *目前，它仍保持开放以供查看并回答任何未解决的问题。*
- **新咨询的重定向**：用户被指示将任何新的 **模型或 API 问题** 提交至另一个指定的频道。
  
  - 这一转变旨在增强支持互动并确保咨询得到高效处理。

 

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/) (1 条消息):

competent: 这个频道将保持开放！

---

### **Cohere ▷ #**[**cmd-r-bot**](https://discord.com/channels/954421988141711382/1168578374038470656/1331356139513839637) (6 条消息):

> `Cohere Command R+ 08-2024 模型、聊天机器人响应中的重复内容、LlamaIndex 集成、故障排除建议、Cohere API 版本讨论`

- **Cohere Command R+ 08-2024 模型中的内容重复问题**：一位用户报告在使用 Cohere Command R+ 08-2024 模型时，响应中出现过多的 **短语重复**，特别是在流模式（stream mode）下。
  
  - 具体示例显示了关于健康相关话题的重复短语，表明 **输出会持续直到达到 token 限制**。
- **影响性能的设置详情**：用户的设置包括在 RAG 工作流中使用最新版本的 **LlamaIndex**，通过 API 部署在 Azure AI Foundry 上。
  
  - 他们澄清说，该问题在之前版本的 **Cohere Command R+ 模型** 中并未出现。
- **对报告问题的内部沟通**：作为回应，一名团队成员对详细的反馈表示感谢，并向用户保证这些反馈将在内部共享。
  
  - 他们建议了临时变通方案，如调整 **temperature** 和 **top p** 值，同时鼓励继续使用之前的版本。
- **使用情况澄清**：用户感谢了团队的建议，但澄清说他们专门使用的是 **cmd-r-plus** 版本，而不是旧的 cmd-r 模型。
  
  - 他们重申了新模型在满足其需求方面存在的问题，并对持续的内部反馈表示感谢。
- **Cohere API 功能探索**：用户注意到他们在 **GitHub discussions** 中找不到类似的问题，且 Cohere API 第 2 版尚未在 LlamaIndex 中实现。
  
  - 缺乏更新阻碍了他们测试所报告重复问题的潜在解决方案的能力。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1331399586861813860) (20 条消息🔥):

> `MOOC 教学大纲发布，演讲嘉宾建议，LLM Hackathon 更新，春季 MOOC 内容，研究合作意向`

- **MOOC 教学大纲预计于 1 月 27 日发布**：由于与演讲嘉宾的最终确认工作正在进行中，即将到来的 MOOC 修订版大纲预计将于 **1 月 27 日**发布。
  
  - 此信息为等待课程内容详情的学生带来了明确性。
- **演讲嘉宾请求已记录**：关于邀请梁文锋（Liang Wenfeng）作为演讲嘉宾的建议已被记录，尽管嘉宾选择已基本定稿。
  
  - 将建立一个反馈表单，以便学生参与未来的嘉宾请求。
- **尚未确认即将举行的 Hackathon**：目前还没有确认的下一场 Hackathon，因为 **春季学期** 的具体细节仍在待定中。
  
  - 鼓励参与者保持关注，因为未来活动可能会有更新。
- **春季 MOOC 基于秋季内容构建**：春季 MOOC 旨在扩展 **秋季内容**，引入高级主题，但不要求预先完成秋季课程。
  
  - 这种灵活的学习方式旨在适应新学生和老学生。
- **研究项目合作意向调查**：目前的讨论强调，宋教授（Prof. Song）正在评估关于未来可能在 Hackathon 中进行 **研究项目合作** 的兴趣。
  
  - 鼓励学生在这一初步评估过程中表达自己的兴趣。

 

---

### **Nomic.ai (GPT4All) ▷ #**[**general**](https://discord.com/channels/1076964370942267462/1090427154141020190/1331394410646339715) (17 条消息🔥):

> `DeepSeek R1 模型，API Key 挑战，讨论中的语言障碍，GPT4All 更新，WordPress 中的 Chatbot 集成`

- **对 GPT4All 上的 DeepSeek R1 模型的关注**：成员们询问了目前有哪些 **DeepSeek R1 蒸馏模型** 在 GPT4All 上运行，并讨论了缺乏类似于 [LM Studio](https://lmstudio.ai/models) 的公开模型目录的问题。
  
  - 据指出，这些模型尚未在 GPT4All 上可用，且需要对 **llama.cpp** 进行更新。
- **语言障碍与讽刺误解**：一位成员的评论引发了关于 **链接** 是否可见的困惑，导致了一场关于讽刺和潜在误解的幽默交流。
  
  - 成员们表达了对沟通挑战的认识，其中一位成员开玩笑地为任何可能被察觉的语言障碍道歉。
- **获取 API Key 的挑战**：一位成员分享了在不使用插件的情况下将 **Chatbot 集成到 WordPress** 的目标，并对获取 API Key 的困难表示沮丧。
  
  - 他们询问是否有 **免费 API Key** 可用，寻求如何继续操作的指导。
- **关于 Discord 频道语言使用的讨论**：一位成员警告不要在特定的 Discord 频道中说 **德语**，暗示可能会因此面临被封禁的风险。
  
  - 另一位成员幽默地建议改用 **英语**，以避免潜在问题。
- **寻求免费无限访问 ChatGPT 的途径**：一位成员询问如何获得对 **ChatGPT** 的免费无限访问，表现出对无需付费即可使用该服务的兴趣。
  
  - 该询问突显了用户对易于获取的 AI 解决方案的持续好奇和需求。

 

**提及的链接**：[Model Catalog - LM Studio](https://lmstudio.ai/models)：可以在电脑上运行的最新、最出色的 LLM。

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1331362015293210694) (9 条消息🔥):

> `基于 DSPy 的 RAG 处理动态数据, DSPy 研究合作, 在 LM Studio REST API 中使用 DSPy, DSPy 与 Ollama 配合使用的错误, 仓库垃圾信息担忧`

- **基于 DSPy 的 RAG 努力应对动态数据**：一位用户询问 **DSPy-based RAG** 如何处理 **动态数据**，表明了对其应用和行为的潜在兴趣。
  
  - 讨论中未列出具体的解决方案或答案。
- **寻求 DSPy 研究合作**：一位用户表达了在 **DSPy 研究**方面进行合作的愿望，并介绍了他们在 **AI for Good**、**LLMs** 和 **高等教育**领域的背景。
  
  - *他们强调了致力于为研究社区做出有意义贡献的承诺*。
- **在 LM Studio REST API 上运行 DSPy 的挑战**：一位用户指出将 **LM Studio** 与 **DSPy** 结合使用并不简单，并对兼容性提出了疑问。
  
  - 他们将其与 **Ollama** 更顺畅的体验进行了对比，暗示了潜在的集成问题。
- **DSPy 与 Ollama 配合使用时的错误信息**：一位用户报告了在运行 **DSPy** 与 **Ollama** 时获取数据摘要相关的错误，特别是提到了 'str' 对象错误。
  
  - 这导致 **DSPy** 不得不在没有数据感知建议器 (data-aware proposer) 的情况下运行，使过程变得复杂。
- **对仓库垃圾信息的担忧**：一位用户对仓库中的 **垃圾信息 (spam)** 表示担忧，质疑其与正在进行的讨论的相关性，可能与某个 **代币相关问题 (coin-related issue)** 有关。
  
  - 他们对这种情况表示沮丧，称其 **非常差劲 (super lame)**。

 

---

### **DSPy ▷ #**[**examples**](https://discord.com/channels/1161519468141355160/1161519685616025600/1331602163805327461) (1 条消息):

> `模型功能, 使用 LM-Studio 模型`

- **用户查询模型功能**：一位成员表示困惑，询问问题是否出在所使用的模型上，特别是本地环境中的 **LM-Studio** 模型。
  
  - 他们很好奇其他人是如何有效利用该模型的，以及是否面临类似的问题。
- **寻求关于模型使用的澄清**：另一位成员询问了使用该模型的方法，旨在确定他们的用法是否偏离了标准实践。
  
  - 他们提出了问题：*'你是如何使用该模型的？'*，表明希望深入了解不同的使用场景。

 

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1331581035502375023) (8 条消息🔥):

> `RLHF 中的自定义损失函数, Phi 4 PR 更新, PR 讨论的背景, 传递自定义前向函数, SimPO 的弃用`

- **自定义损失函数和前向传递提案**：一位成员提议开启一个与 RLHF 中 **自定义损失 (custom loss)** 和 **自定义前向函数 (custom forward functions)** 相关的 PR，并建议创建一个专门的文档页面进行说明。
  
  - 他们的目标是移除所有自定义损失，同时提供关于如何添加它们的文档，以促进新 RLHF 损失的集成。
- **Phi 4 PR 需要更新**：该成员确认了 **Phi 4 PR**，并提到在进一步推进之前需要修复几个点。
  
  - 他们表示打算在解决未决问题后对其进行迭代。
- **新 PR 需要背景信息**：一位成员询问是否有任何 issue 或讨论为该 PR 提供背景信息。
  
  - 原作者确认了在 [issue #2206](https://github.com/pytorch/torchtune/issues/2206) 中关于 **自定义对比损失 (custom contrastive losses)** 的相关讨论。
- **自定义函数的设计考虑**：有人请求提供一个关于向 recipe 传递自定义前向函数的实际示例，强调了需要更清晰的实现。
  
  - 在不使用参考对数概率 (reference logprobs) 的损失场景下，需要确保与 **DPO 全量微调 (full-finetuning)** 过程的兼容性。
- **SimPO 弃用提醒**：讨论提到了最近弃用 **SimPO** 的决定，强调需要相应地更新文档。
  
  - 这一步被视为提高实现自定义损失设计的清晰度和功能性的关键。

 

**提到的链接**：[对齐部分的自定义损失重新设计 · Issue #2206 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/2206)：我们已经就 torchtune 中的自定义对比损失进行了多次迭代。这方面的最后一点是弃用 SimPO #2062 并禁止新的自定义损失...

 

---

### **Torchtune ▷ #**[**papers**](https://discord.com/channels/1216353675241590815/1293438210097025085/1331685498116243508) (2 messages):

> `Nature Communications 特色文章`

- **论文在 Nature Communications 发表**：我们的论文现已在 [Nature Communications](https://www.nature.com/collections/ceiajcdbeb) 上发表，这是我们工作的一个重要里程碑。
  
  - 社区成员对此表示了 *祝贺 (Congrats)*，表达了对论文发表的兴奋之情。
- **庆祝里程碑**：成员们对论文被接收感到非常兴奋，显示了社区内的积极反响。
  
  - 成员使用了 *super cool* 一词来强调这一成就，突显了协作精神。

 

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1331496722479710280) (5 messages):

> `OpenInterpreter 1.0, Python 代码执行, Markdown 和 TXT 格式化`

- **对 OI 1.0 中 Python 支持的不确定性**：一位用户表达了对即将推出的 **OpenInterpreter 1.0** 是否会移除内部运行 **Python** 代码能力的担忧，并询问该功能稍后是否会恢复。
  
  - 另一位成员回应称，尽管存在不确定性，但在 **1.0 pre-release** 版本中似乎 **尚未实现运行 Python 代码** 的功能。
- **关于 Markdown 和 TXT 相关性的讨论**：针对一个查询，一位成员澄清说 **Markdown** 和 **TXT** 不是编程语言，而是显示格式化器。
  
  - 这引发了后续评论，暗示在这方面可能正在开展相关工作。

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**discussion**](https://discord.com/channels/1111172801899012102/1111353033352294440/1331547139838312528) (3 messages):

> `来自 Ollama 的 Gorilla 模型, LLaMA v2 模型规格`

- **关于 Gorilla 模型使用的咨询**：一位成员询问是否可以使用 [Ollama 文档](https://ollama.com/adrienbrault/gorilla-openfunctions-v2:Q6_K/blobs/8f6765ab9969)中的 **Gorilla 模型**。他们标记了另外两人，以确认这些是否是正确的模型。
- **LLaMA v2 规格讨论**：详细列出了 **LLaMA v2** 模型的规格，包括 **4096** 的 **context length**（上下文长度）和 **30** 的 **block count**（块数量）等参数。
  
  - 对话指出了特定的属性，如 **attention head count**（注意力头数）为 **32**，并进一步提供了 **tokenizer settings**（分词器设置）和 **quantization version**（量化版本）的细节。

 

**提到的链接**：[adrienbrault/gorilla-openfunctions-v2:Q6_K/model](https://ollama.com/adrienbrault/gorilla-openfunctions-v2:Q6_K/blobs/8f6765ab9969): [https://huggingface.co/gorilla-llm/gorilla-openfunctions-v2-gguf](https://huggingface.co/gorilla-llm/gorilla-openfunctions-v2-gguf)

 

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1331660639533400226) (2 messages):

> `Tinygrad 的 Windows 测试, 通过 OpenCL 的 GPU 支持`

- **已准备好 Windows 测试的 Pull Request**：一位贡献者准备了一个 [Windows 测试的 PR](https://github.com/tinygrad/tinygrad/pull/8715)，并就哪些 backends（后端）应该在 Windows 下正常运行寻求指导，提到了 **LLVM** 和 **Clang** 作为可能的选项。
  
  - 作为咨询的一部分，他们包含了一个展示 PR 详情的图片链接。
- **OpenCL GPU 支持被认为至关重要**：另一位成员强调，支持 **GPU (OpenCL)** 可能是 Windows 测试中最有价值的补充。
  
  - 这一建议表明，目前的重点是优化 Tinygrad 在不同硬件配置上的性能。

 

**提到的链接**：[Windows tests ci by c143 · Pull Request #8715 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/8715): 未找到描述

 

---

### **Axolotl AI ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1331716882071425116) (1 messages):

> `KTO Loss, Liger Kernel, 模型合并`

- **Liger Kernel 合并 KTO Loss**：**KTO loss** 已正式合并到 [Liger Kernel 仓库](https://github.com/linkedin/Liger-Kernel/pull/475)中，这是一个重要的更新。
  
  - 此次合并预计将提升性能，展示了在优化 kernel 方面的持续努力。
- **关于模型合并策略的讨论**：在讨论 KTO loss 合并时，成员们强调了可能从这一进展中受益的**模型合并策略**。
  
  - 社区对于利用这种 loss 来改进未来模型更新中的集成表现充满热情。

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1331719938183790747) (1 条消息):

> `Local-First X AI Hackathon, San Francisco events, Hackathon planning, February 22`

- **Hackathon 公布：为 Local-First X AI 做好准备！**：组织者将于 **2 月 22 日**在**旧金山**启动 [Local-First X AI Hackathon](https://www.lofihack.com/)。
  
  - 加入 [讨论线程](https://discord.com/channels/1089876418936180786/1329529625189154826) 以获取有关即将举行的活动的更多详情和更新！
- **加入关于 Hackathon 的对话**：组织者已经建立了一个专门的线程，用于对 Hackathon 进行**更多讨论**。
  
  - 随着 **2 月 22 日**活动日期的临近，请务必关注该线程并保持灵感涌现！

 

---

---

---

{% else %}

> 完整的频道细分内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整的细分内容，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}