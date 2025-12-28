---
companies:
- anthropic
- openai
- thinking-machines
date: '2025-02-18T23:54:10.975706Z'
description: '**Grok 3** 已经发布，虽然外界评价褒贬不一，但其基准测试表现强劲，显著优于 **Gemini 2 Pro** 和 **GPT-4o**
  等模型。**Grok-3 mini** 版本展现出了极具竞争力、甚至在某些方面更为优越的能力，尤其是在推理和编程领域，其中强化学习（reinforcement
  learning）发挥了关键作用。**Mira Murati** 公开了她离开 OpenAI 后的计划，成立了名为 **Thinking Machines**
  的前沿实验室，专注于协作式、可个性化的 AI、多模态以及实证安全与对齐研究，这让人联想到 **Anthropic** 的发展路径。'
id: 253730e7-cc51-424c-89a3-c31ebf0c3199
models:
- grok-3
- grok-3-mini
- gemini-2-pro
- gpt-4o
- o3-mini-high
- o1
- deepseek-r1
original_slug: ainews-xai-grok-3-and-mira-muratis-thinking
people:
- mira-murati
- lmarena_ai
- karpathy
- omarsar0
- ibab
- arankomatsuzaki
- iscienceluvr
- scaling01
title: X.ai 的 Grok 3 与 Mira Murati 的 Thinking Machines
topics:
- benchmarking
- reasoning
- reinforcement-learning
- coding
- multimodality
- safety
- alignment
- research-publishing
- model-performance
- creative-ai
---



宣言中除了对**发表研究**的信念、对**协作式和个性化 AI** 的强调、**多模态**、**研究与产品协同设计**以及**经验主义的安全与对齐方法**之外，并没有太多细节。从纸面上看，它就像是 “Anthropic 的翻版”。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

**Grok-3 模型性能与基准测试**

- **Grok-3 表现优于其他模型**：[@omarsar0](https://twitter.com/omarsar0/status/1891706611023938046) 报告称，**Grok-3 显著优于 Gemini 2 Pro 和 GPT-4o 等同类模型**，甚至 **Grok-3 mini 也表现出了极具竞争力的性能**。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1891708045324902499) 表示，**在初步基准测试中，Grok-3 推理模型优于 o3-mini-high、o1 和 DeepSeek R1**。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1891706264800936307) 宣布 **Grok-3 在 Chatbot Arena 中排名第一**，成为 **首个突破 1400 分的模型**。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1891708250199839167) 指出，**Grok 3 推理测试版在 AIME 上获得 96 分，在 GPQA 上获得 85 分**，与 **完整版 o3** 持平。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1891708408832610548) 强调了 **Grok 3 在 AIME 2025 上的表现**。[@scaling01](https://twitter.com/scaling01/status/1891790801631314103) 强调了 **Grok-3 在所有类别的基准测试中均排名第一，令人印象深刻**。
- **Grok-3 mini 的能力**：[@omarsar0](https://twitter.com/omarsar0/status/1891711669849505864) 分享了 **使用 Grok 3 mini 生成的结果**，而 [@ibab](https://twitter.com/ibab/status/1891761914688254340) 提到 **Grok 3 mini 表现惊人，即将发布**。[@Teknium1](https://twitter.com/Teknium1/status/1891715974992408738) 在测试中发现 **Grok-3 mini 通常优于完整版 Grok-3**，这表明它并非简单的蒸馏模型，可能经过了完整的 RL 训练。
- **Grok-3 的推理和编程能力**：[@omarsar0](https://twitter.com/omarsar0/status/1891707915351859547) 表示 **Grok-3 还具备通过 RL 解锁的推理能力，在编程方面表现尤为出色**。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1891706272711381237) 指出 **Grok-3 在编程方面超越了 o1 和 Gemini-thinking 等顶尖推理模型**。[@omarsar0](https://twitter.com/omarsar0/status/1891709371802910967) 强调了 **Grok 3 的创造性涌现能力**，在 **生成游戏等创意编程方面表现卓越**。[@omarsar0](https://twitter.com/omarsar0/status/1891711110476111884) 展示了 **Grok 3 推理测试版在 AIME 2025 上的表现**，证明了其 **超越编程和数学的泛化能力**。
- **与其他模型的比较**：[@nrehiew_](https://twitter.com/nrehiew_/status/1891710589115715847) 认为 **Grok 3 推理本质上是一个 o1 级别的模型**，这意味着 **OpenAI 和 xAI 之间存在 9 个月的能力差距**。[@Teknium1](https://twitter.com/Teknium1/status/1891726810494210449) 认为 Grok-3 相当于 **具备深度研究能力的 o3-full**，但成本仅为其一小部分。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1891732255938077133) 表示 **Grok-3 与 o3-mini 一样出色**。
- **Grok-3 的计算资源**：[@omarsar0](https://twitter.com/omarsar0/status/1891705957220016403) 提到 **Grok 3 的训练量是 Grok 2 的 10 倍**，预训练于 **1 月初** 完成，目前训练仍在进行中。[@omarsar0](https://twitter.com/omarsar0/status/1891705593125105936) 透露总共使用了 **20 万个 GPU**，且容量在 92 天内翻了一番以改进 Grok。[@ethanCaballero](https://twitter.com/ethanCaballero/status/1891712442893312151) 指出 **Grok-3 的训练计算量为 8e26 FLOPs**。
- **Karpathy 对 Grok-3 的 vibe check**：[@karpathy](https://twitter.com/karpathy/status/1891720635363254772) 分享了对 Grok 3 详细的 **vibe check**，发现其 **接近 SOTA 水平**（类似于 **OpenAI 的 o1-pro**），并且 **优于 DeepSeek-R1 和 Gemini 2.0 Flash Thinking**。他测试了 **推理能力、表情符号解码、井字棋、GPT-2 论文分析和研究问题**。他还测试了 **DeepSearch**，发现其与 **Perplexity DeepResearch** 相当，但尚未达到 **OpenAI 的 "Deep Research"** 水平。

**公司与产品公告**

- **xAI Grok-3 发布**: [@omarsar0](https://twitter.com/omarsar0/status/1891705029083512934) 宣布了 **xAI 发布 Grok 3 的重磅消息**。[@alexandr_wang](https://twitter.com/alexandr_wang/status/1891714169629524126) 祝贺 **xAI 的 Grok 3 成为新的最强模型**，在 Chatbot Arena 中排名第一。[@Teknium1](https://twitter.com/Teknium1/status/1891705665007050851) 也宣布了 **Grok 3 正式亮相**。[@omarsar0](https://twitter.com/omarsar0/status/1891715441292083572) 提到 **Grok 3 已在 X Premium+ 上线**。[@omarsar0](https://twitter.com/omarsar0/status/1891715813956108699) 表示 **改进将迅速进行，几乎每天都会更新**，并且 **由 Grok 驱动的语音应用将在约一周内推出**。
- **Perplexity R1 1776 开源发布**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1891917148869755058) 宣布 **Perplexity 开源 R1 1776**，这是 **DeepSeek R1 的一个版本，经过后训练以移除中国审查**，强调 **无偏见且准确的回答**。[@perplexity_ai](https://twitter.com/perplexity_ai/status/1891916573713236248) 也宣布了 **开源 R1 1776**。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1891900478818033940) 指出 **Perplexity 现已入驻 Hugging Face**。
- **DeepSeek NSA 稀疏注意力机制**: [@deepseek_ai](https://twitter.com/deepseek_ai/status/1891745487071609327) 介绍了 **NSA (Natively Trainable Sparse Attention)**，这是一种 **硬件对齐机制，用于快速的长上下文训练和推理**。
- **OpenAI SWE-Lancer 基准测试**: [@OpenAI](https://twitter.com/OpenAI/status/1891911123517018521) 推出了 **SWE-Lancer**，这是一个 **全新的真实编程性能基准测试**，包含 **1,400 个来自 Upwork、价值 100 万美元的自由软件工程任务**。[@_akhaliq](https://twitter.com/_akhaliq/status/1891721712296747126) 也宣布了 **OpenAI SWE-Lancer**。
- **LangChain LangMem SDK**: [@LangChainAI](https://twitter.com/LangChainAI/status/1891881053897490772) 发布了 **LangMem SDK**，这是一个 **用于 AI Agent 长期记忆的开源库**，使 Agent 能够 **从交互中学习并优化 Prompt**。
- **Aomni 400 万美元种子轮融资**: [@dzhng](https://twitter.com/dzhng/status/1891897453831491838) 宣布 **Aomni 为其 AI Agent 筹集了 400 万美元种子轮融资**，该 Agent 可使 **营收团队的产出提升 10 倍**。
- **MistralAI Batch API UI**: [@sophiamyang](https://twitter.com/sophiamyang/status/1891869154770026502) 介绍了 **MistralAI Batch API UI**，允许用户从 la Plateforme **创建和监控批量作业**。
- **Thinking Machines Lab 成立**: [@dchaplot](https://twitter.com/dchaplot/status/1891920016339042463) 宣布 **Thinking Machines Lab 成立**，并邀请他人加入。

**技术深度解析与研究**

- **Less is More Reasoning (LIMO)**：[@AymericRoucher](https://twitter.com/AymericRoucher/status/1891822202812760206) 重点介绍了 **Less is More for Reasoning (LIMO)**，这是一个**使用 817 个样本微调的 32B 模型**，它在**数学推理上超越了 o1-preview**，这表明对于推理而言，**精心挑选的样本比单纯的数量更重要**。
- **Diffusion Models without Classifier-Free Guidance**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1891847953087619147) 分享了一篇关于**无 Classifier-free Guidance 的 Diffusion Models** 的论文，通过**直接学习修改后的分值（score）**，在 **ImageNet 256x256 上实现了新的 SOTA FID**。
- **Scaling Test-Time Compute with Verifier-Based Methods**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1891839822257586310) 讨论的研究证明，在**扩展测试时计算（test-time compute）**方面，**使用 RL 或搜索的基于验证器（VB）的方法优于无验证器（VF）的方法**。
- **MaskFlow for Long Video Generation**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1891726104991502786) 介绍了 **MaskFlow**，这是来自 CompVis 实验室的一种**用于长视频生成的分块自回归方法**，利用**帧级掩码（frame-level masking）**来实现高效且无缝的视频序列。
- **Intuitive Physics from Self-Supervised Video Pretraining**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1891721882065391692) 展示了 **Meta 的研究，表明直觉物理理解源于对自然视频的自监督预训练**，其方式是在**表示空间（rep space）中预测结果**。
- **Reasoning Models and Verifiable Rewards**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1891893034956030242) 解释说，**像 Grok-3 和 DeepSeek-R1 这样的推理模型是使用可验证奖励通过强化学习（RL）训练的**，强调了**数学和编程任务中的验证**以及 **RL 在学习复杂推理中的力量**。
- **NSA: Hardware-Aligned Sparse Attention**：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1891745487071609327) 详细介绍了 **NSA 的核心组件**：**动态分层稀疏策略、粗粒度 Token 压缩和细粒度 Token 选择**，针对**现代硬件**进行了优化，以加快推理速度并降低预训练成本。

**AI 行业与市场分析**

- **xAI 作为 SOTA 竞争对手**：[@scaling01](https://twitter.com/scaling01/status/1891846484791820502) 认为，**在 Grok-3 之后，xAI 必须被视为 SOTA 模型的真正竞争对手**，尽管 **OpenAI、Anthropic 和 Google 在内部可能仍处于领先地位**。[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1891853619907133702) 也表示 **xAI 已到达前沿**，并加入了**美国“五大”AI 实验室**。[@omarsar0](https://twitter.com/omarsar0/status/1891705031243469270) 指出 **Elon 提到 Grok 3 的能力比 Grok 2 高出一个数量级**。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1891886896659517754) 观察到**中国凭借 DeepSeek 和 Qwen 取得的快速 AI 进展**，突显了**中国 AI 社区的成熟**。
- **OpenAI 的策略与市场地位**：[@scaling01](https://twitter.com/scaling01/status/1891849320321720399) 讨论了 **Gemini 2.0 Flash 正在蚕食 Anthropic 的市场份额**，认为 **Anthropic 需要降低价格或发布更好的模型以维持增长**。[@scaling01](https://twitter.com/scaling01/status/1891786871304323280) 评论说，**Grok-3 的发布可能是为了争夺心智份额和关注度**，因为 **GPT-4.5 和新的 Anthropic 模型即将推出**，Grok-3 的领先地位可能很短暂。
- **Perplexity 的增长与深度研究**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1891621245440889102) 强调了 **Perplexity Deep Research 每日 PDF 导出量的增长**，表明使用量正在增加。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1891626705858576737) 分享的调查数据显示，**超过 52% 的用户愿意从 Gemini 转向 Perplexity**。
- **AI 与能源消耗**：[@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1891938726671945757) 讨论了 **AI 日益增长的能源需求与其提高文明效率的潜力之间的平衡**，并以 **DeepMind AI 使 Google 数据中心能耗降低 40%** 为例。
- **SWE-Lancer 基准测试与软件工程中的 LLM**：[@_philschmid](https://twitter.com/_philschmid/status/1891780812497887289) 总结了 **OpenAI 的 SWE-Lancer 基准测试**，显示 **Claude 3.5 Sonnet 实现了 40.3 万美元的盈利潜力**，但**前沿模型仍无法解决大多数任务**，突显了在**根本原因分析和复杂解决方案**方面的挑战。[@mathemagic1an](https://twitter.com/mathemagic1an/status/1891712313599623461) 建议，即使不合并 PR，**像 DevinAI 这样的通用型 SWE Agent 对于开发讨论也是非常有用的**。

**开源与社区**

- **呼吁开源 o3-mini**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1891669417332805739) 敦促大家**为开源 o3-mini 投票**，并强调**社区有能力将其蒸馏（distill）成手机大小的模型**。[@mervenoyann](https://twitter.com/mervenoyann/status/1891772390301941796) 开玩笑说想要 **o3-mini 开源**。[@gallabytes](https://twitter.com/gallabytes/status/1891674566931497410) 请求人们**为 o3-mini 投票**。[@eliebakouch](https://twitter.com/eliebakouch/status/1891675065021853805) 幽默地暗示**内部宣传正在起作用，引导大家为 o3-mini 投票**。
- **Perplexity 开源 R1 1776**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1891917148869755058) 宣布了 **Perplexity 的首个开放权重模型 R1 1776**，并在 [@huggingface](https://twitter.com/huggingface) 上发布了权重。[@reach_vb](https://twitter.com/reach_vb/status/1891922768892989559) 强调 **Perplexity 发布了经过 Post-trained 且采用 MIT 许可证的 DeepSeek R1**。
- **Axolotl v0.7.0 发布**: [@winglian](https://twitter.com/winglian/status/1891933173858304413) 宣布 **Axolotl v0.7.0 发布，支持 GRPO、多 GPU LoRA 内核、Modal 部署等功能**。
- **LangMem SDK 开源**: [@LangChainAI](https://twitter.com/LangChainAI/status/1891881053897490772) 将 **LangMem SDK 作为开源项目发布**。
- **Ostris 专注于开源**: [@ostrisai](https://twitter.com/ostrisai/status/1891820293993398609) 宣布转型为**全职专注于开源工作**，承诺提供**更多模型、工具包改进、教程，并寻求资金支持**。
- **呼吁 Grok-3 开源**: [@huybery](https://twitter.com/huybery/status/1891712667947057598) 呼吁 **Grok-3 开源**。
- **DeepSeek 对开放科学的奉献**: [@reach_vb](https://twitter.com/reach_vb/status/1891755094330212552) 感谢 **DeepSeek 对开源和科学的奉献**。

**梗与幽默**

- **对 Grok-3 名称和性能的反应**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1891710580785832085) 的反应是“**别再来一个深度搜索了，啊啊啊**”。[@nrehiew_](https://twitter.com/nrehiew_/status/1891702073135141047) 发布了“**停止计票**”，这可能是针对 Grok-3 基准测试结果的玩笑。[@scaling01](https://twitter.com/scaling01/status/1891842735834808708) 调侃道：“**我爱那个看门人，但请接受 Grok-3 是目前最强大的公开可用 LLM（至少能维持一天，哈哈）**”。
- **手机大小模型投票的幽默**: [@nrehiew_](https://twitter.com/nrehiew_/status/1891670049523470807) 讽刺地评论道：“**那些投票给‘手机大小模型’的人太不严肃了。社区能在一个月内让三星 Galaxy 手表运行 o3-mini 级别的模型。请认真一点**”。[@dylan522p](https://twitter.com/dylan522p/status/1891682135255154775) 表示：“**不敢相信 X 的用户这么蠢。不投给 o3-mini 简直疯了。**”
- **Elon Musk 和 xAI 的笑话**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1891713223591629143) 拿 **Elon 在 Grok-3 问答环节发推特**开玩笑。[@aidan_mclau](https://twitter.com/aidan_mclau/status/1891628395928117417) 发布了“**他甚至对他的精神对话者都很刻薄 😔**”并附上了一条推文链接。
- **硅谷叙事贩子**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1891834083455430945) 批评了“**那些人设经过优化、试图毫不费力地唤起谢尔顿·库珀（Sheldon Cooper）或理查德·亨德里克斯（Richard Hendricks）形象的硅谷叙事贩子**”。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1：OpenAI 的 o3-mini 与手机大小模型投票争议**

- **[Sam Altman 关于开源模型的投票..](https://i.redd.it/dug7nt8n0tje1.png)** ([Score: 631, Comments: 214](https://reddit.com/r/LocalLLaMA/comments/1is1f37/sam_altmans_poll_on_open_sourcing_a_model/)): **Sam Altman** 发起了一项 Twitter 投票，询问在他们的下一个开源项目中，是创建一个小型的 "o3-mini" 级别模型，还是创建一个最强的 "手机级模型" (phone-sized model) 更有用。目前后者在 **695 张总票数**中获得了 **62.2%** 的支持。投票还剩 **23 小时**，显示了社区在 AI 模型开源决策中的参与度。
  - 许多评论者支持 **o3-mini** 模型，认为如果需要，它可以被蒸馏 (distilled) 成手机模型，并强调了它在**本地机器**和小型组织中的潜在效用。一些人对投票的真实性表示怀疑，认为这可能是一种营销策略，或者投票结果受到了操纵。
  - 存在一种反对**手机级模型**的强烈情绪，用户质疑其在当前硬件限制下的实用性，并建议**更大的模型**可以更具通用性。一些人认为 **OpenAI** 最终可能会发布这两个模型，利用投票来决定先发布哪一个。
  - 讨论反映了对 **OpenAI** 开源意图的广泛怀疑，一些用户怀疑该公司对开源其模型的承诺。其他人强调了**蒸馏技术** (distillation techniques) 的重要性，即从大型模型创建小型、高效的模型，认为这种方法对开源社区有益。


- **[ClosedAI 的下一个开源项目](https://i.redd.it/grv77lpq0tje1.jpeg)** ([Score: 119, Comments: 27](https://reddit.com/r/LocalLLaMA/comments/1is1eht/closedai_next_open_source/)): **Sam Altman** 的一条推文引发了辩论，询问开发一个用于 **GPU** 的较小模型更有利，还是专注于创建尽可能好的手机级模型。讨论集中在不同平台上的模型大小与运行效率之间的权衡。
  - **本地手机模型**因速度慢、耗电量大且与在线模型相比占用磁盘空间而受到批评。评论者对本地手机模型的实用性表示怀疑，并暗示支持此类模型的投票可能受到了 **OpenAI** 员工的影响。
  - **OpenAI 的商业模式**受到审视，人们怀疑他们是否愿意开源像 **o3** 这样的模型。用户推测 OpenAI 可能只会在模型过时时才发布，并对可能影响真正开源公司的潜在监管影响表示担忧。
  - 存在支持 **O3-MINI** 作为更通用选项的强烈呼声，并认为其具有蒸馏成移动版本的潜力。一些用户批评投票的设想方式，并预测 OpenAI 可能会发布一个次优模型，以在不真正支持开放创新的情况下表现出开源的姿态。


**主题 2. GROK-3 在 GPU 争议中声称达到 SOTA 霸权**

- **[GROK-3 (SOTA) 和 GROK-3 mini 均超越 O3-mini high 和 Deepseek R1](https://i.redd.it/8dwhr7o0ttje1.jpeg)** ([Score: 334, Comments: 312](https://reddit.com/r/LocalLLaMA/comments/1is4geo/grok3_sota_and_grok3_mini_both_top_o3mini_high/)): **Grok-3 Beta** 在**数学 (AIME '24)** 方面取得了 **96** 分的领先成绩，展示了其与 **O3-mini (high)** 和 **Deepseek-R1** 等竞争对手相比的卓越性能。柱状图突出了数学、科学 (GPQA) 和编程 (LCB Oct-Feb) 类别的对比得分，**Grok-3** 模型在测试时计算 (test-time compute) 分析中表现优于其他模型。
  - 许多用户对 **Grok-3** 的性能声明表示怀疑，指出缺乏独立基准测试且没有开源可用性。**Lmsys** 被提及为独立基准，但用户对 **$40/月** 的费用持谨慎态度，因为其与 **ChatGPT** 等其他模型相比没有显著的差异化。
  - 讨论强调了对 **Elon Musk** 参与 **Grok-3** 的担忧，用户表达了不信任，并将他的行为与法西斯意识形态联系起来。一些评论批评了在开源讨论中使用 "woke" 一词，并强调了围绕 Musk 政治活动的争议。
  - 技术讨论集中在 **ARC-AGI 基准测试**上，该测试成本高且复杂， OpenAI 对其投入了大量资金。用户注意到 **Grok-3** 未在 ARC-AGI 上进行测试，并有兴趣看到它在当前 SOTA 模型表现挣扎的基准测试中的表现。

- **[Grok 演示摘要](https://i.redd.it/44mbkdqtytje1.png)** ([Score: 245, Comments: 80](https://reddit.com/r/LocalLLaMA/comments/1is519h/grok_presentation_summary/)): 该图片描绘了一个专家小组的 **Q&A 环节**，成员可能包括 **Elon Musk**，讨论了 **xAI GROK-3** 的发布。社区反应不一，评论包括“Grok 很好”和“Grok 将前往火星”，表明对该演示和该技术潜在能力的看法褒贬不一。
  - **演示的目标受众**被批评为非工程师，几位评论者指出，与 **OpenAI** 的演示相比，内容组织性较差。**Elon Musk** 作为非工程师的角色被强调，一些人对他报告技术细节的准确性表示怀疑。
  - 小组（尤其是非 Elon 成员）的**肢体语言**被注意到表现得紧张或恐惧，一些人将其归因于 Elon 的在场。尽管如此，一些人欣赏该小组原始、非企业化的方式。
  - 出现了关于与 **OpenAI** 以及 **Deepseek** 和 **Qwen** 等其他模型的 **benchmark 比较**的讨论，一些人对这些自报的 benchmark 表示怀疑。还提到了 **H100s** 以及为了达到与 **OpenAI 的最新模型**同等水平而产生的成本影响。


**Theme 3. DeepSeek 发布 Native Sparse Attention 模型**

- **[DeepSeek Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)** ([Score: 136, Comments: 6](https://reddit.com/r/LocalLLaMA/comments/1is72j2/deepseek_native_sparse_attention_hardwarealigned/)): **DeepSeek 的 NSA 模型**引入了 **Native Sparse Attention**，它既是**硬件对齐（hardware-aligned）**的，也是**原生可训练（natively trainable）**的。这种在 sparse attention 方面的创新旨在提高模型的效率和性能。
  - **DeepSeek 的 NSA 模型**与 Microsoft 的 **"SeerAttention"** 在概念上有相似之处，后者也探索了可训练的 sparse attention，正如 **LegitimateCricket620** 所指出的。**LoaderD** 建议 DeepSeek 和 Microsoft 研究人员之间可能存在合作，并强调如果属实，需要进行适当的引用。
  - **Recoil42** 强调了 **NSA 模型**的核心组件：*动态分层稀疏策略（dynamic hierarchical sparse strategy）*、*粗粒度 Token 压缩（coarse-grained token compression）*和*细粒度 Token 选择（fine-grained token selection）*。这些组件针对现代硬件进行了优化，在不牺牲性能的情况下提高了推理速度并降低了预训练成本，在各种 benchmark 上优于 Full Attention 模型。


- **[DeepSeek 仍在发力](https://i.redd.it/ikhcif5gxuje1.jpeg)** ([Score: 809, Comments: 100](https://reddit.com/r/LocalLLaMA/comments/1is7yei/deepseek_is_still_cooking/)): **DeepSeek 的 NSA** (Non-Sparse Attention) 展示了优于传统 **Full Attention** 方法的性能，在 **General、LongBench 和 Reasoning** 等 benchmark 中获得了更高的分数。它还提供了显著的速度提升，在 decode 阶段实现了高达 **11.6 倍的加速**。此外，NSA 在高达 **64K** 的上下文长度下保持了完美的检索准确率，说明了其在处理大规模数据方面的效率。[DeepSeek NSA Paper](https://arxiv.org/abs/2502.11089)。
  - 讨论强调了 **DeepSeek NSA 降低 VRAM 需求的潜力**，这归功于压缩的 key 和 value，尽管没有提供实际的 VRAM 比较。该模型高效处理长上下文的能力也受到了关注，并有人询问其在不同上下文大小下的表现。
  - **分层稀疏注意力（Hierarchical sparse attention）**引起了人们的兴趣，用户推测它有可能使在消费级硬件上进行高速处理成为可能。该模型的 **27B 总参数和 3B 激活参数**被认为是平衡性能和计算效率的理想尺寸。
  - 评论强调了 DeepSeek NSA 带来的**显著速度提升**，一些用户对其在移动设备上运行的实际应用和潜力表示感兴趣。该模型的方法因其在降低计算成本方面的效率而受到称赞，这与仅仅增加计算能力形成了鲜明对比。


**Theme 4. PerplexityAI 的 R1-1776 移除了 DeepSeek 中的审查**

- **[PerplexityAI 发布 R1-1776，这是一个 DeepSeek-R1 的微调版本，在保持推理能力的同时移除了中国式审查](https://huggingface.co/perplexity-ai/r1-1776)** ([Score: 185, Comments: 78](https://reddit.com/r/LocalLLaMA/comments/1iskklo/perplexityai_releases_r11776_a_deepseekr1/)): **PerplexityAI** 发布了 **R1-1776**，这是 **DeepSeek-R1** 的一个微调版本，旨在消除中国式审查，同时保留其推理能力。此次发布表明其重点在于增强对未经审查信息的获取，且不损害 AI 的性能。
  - 讨论集中在该发布的**有效性和必要性**上，一些人对该模型声称提供**公正、准确和事实性信息**表示怀疑。批评者质疑该模型是否只是将**中国式审查替换为了美国式审查**，并指出了模型中潜在的偏见。
  - 用户对**中国和西方审查制度**进行了比较，指出中国式审查通常涉及直接压制，而西方方法可能涉及传播虚假信息。对话中包含了**天安门广场**和**美国政治问题**等例子，以说明不同的审查风格。
  - 用户对该模型的**开源状态**表示怀疑，有人质疑该模型的实际用途，另一些人则认为这是**工程精力的浪费**。一篇博客文章链接提供了有关此次发布的更多细节：[Open-sourcing R1 1776](https://www.perplexity.ai/hub/blog/open-sourcing-r1-1776)。


**主题 5. 加速 Hugging Face 模型下载**

- **将 Hugging Face 模型下载速度提升 100 倍** ([Score: 167, Comments: 45](https://reddit.com/r/LocalLLaMA/comments/1ise5ly/speed_up_downloading_hugging_face_models_by_100x/)): 使用基于 Rust 的工具 **hf_transfer**，可以将 Hugging Face 模型的下载速度显著提升至 **1GB/s** 以上，而使用 Python 命令行时的典型上限为 **10.4MB/s**。该帖子提供了安装和启用 **hf_transfer** 以实现快速下载的分步指南，并强调速度限制并非源于 Python，而可能是 **hf_transfer** 所没有的带宽限制。
  - 用户讨论了下载 Hugging Face 模型的替代工具，例如 **HFDownloader** 和基于 Docker 的 CLI，这些工具提供了预配置的解决方案以避免在宿主机安装。**LM Studio** 被提到可以达到约 **80 MB/s**，这表明 **10.4 MB/s** 的上限不是 Python 的限制，而很可能是带宽问题。
  - 关于分发模型权重的法律影响以及使用 **torrent** 进行分发的潜在好处存在争论，同时也涉及对控制传播和责任的担忧。一些用户认为 torrent 将是理想的选择，但也承认管理分发存在挑战。
  - **hf_transfer** 工具被强调对高速下载非常有益，特别是在数据中心环境中，据称速度超过 **500MB/s**。用户对该工具能够降低下载大型模型（如 **Llama 3.3 70b**）相关成本的能力表示感谢。


## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. Grok 3 基准测试发布及性能辩论**

- **[Grok 3 怎么就成了地球上最聪明的 AI？简单来说，它并不是，但如果不是在 o3 的水平上，它确实非常出色](https://i.redd.it/l702dxbryuje1.jpeg)** ([Score: 1012, Comments: 261](https://reddit.com/r/OpenAI/comments/1is81yr/how_is_grok_3_smartest_ai_on_earth_simply_its_not/))：**Grok-3** 被讨论为一个能力极强的 AI 模型，尽管与 **o3** 相比不一定是最“聪明”的。一张来自直播的图表（由 Rex 通过推文分享）比较了 AI 模型在数学、科学和编程等任务上的表现，重点展示了 **Grok-3 Reasoning Beta** 和 **Grok-3 mini Reasoning** 以及其他模型，Rex 还添加了 **o3** 的数据以进行全面分析。
  - 讨论中对 **Grok 3 声称的优越性**表示怀疑，一些用户质疑基准测试和对比的有效性，特别是针对尚未发布的 **o3**。普遍共识是 **o3** 目前无法进行独立评估，这使得对比变得复杂。
  - 用户对 **Elon Musk 参与** AI 项目表示担忧，由于感知到的伦理问题和 AI 技术的潜在滥用，用户表达了不信任。一些评论反映了对 **OpenAI** 和 **xAI** 等公司吹捧的基准测试和 AI 能力背后的透明度及动机的忧虑。
  - **Grok 3** 因其在实时股票分析方面的潜力而受到关注，尽管一些用户认为它可能不是目前最聪明的 AI。评论还讨论了 **o3** 的成本和扩展性问题，有报告指出其价格可能高得令人望而却步，每个 **prompt** 的成本高达 **1000 美元**。


- **[GROK 3 刚刚发布](https://i.redd.it/7dfu7ltnttje1.jpeg)** ([Score: 630, Comments: 625](https://reddit.com/r/OpenAI/comments/1is4ipt/grok_3_just_launched/))：**GROK 3** 已经发布，展示了其在数学 (**AIME '24**)、科学 (**GPQA**) 和编程 (**LCB Oct-Feb**) 等学科的 AI 模型基准测试对比中的表现。深蓝色标注的 **Grok-3 Reasoning Beta** 模型获得了最高分，特别是在数学和科学方面表现出色，柱状图显示其得分在 40 到 100 之间。
  - 对于 **Grok 3** 基准测试的可靠性存在显著质疑，用户对**基准测试的来源**和**图表的呈现方式**提出疑问。一些用户指出，**Grok 3** 似乎是通过选择性的基准测试优化才超越了其他模型，导致对 **Elon Musk** 公司呈现的结果产生不信任。
  - 讨论与政治和伦理担忧紧密交织，多条评论表达了对与 **Elon Musk** 相关产品的不信任，理由是他备受争议的行为和言论。用户强调更倾向于来自 **Deepmind** 和 **OpenAI** 等其他公司的替代模型。
  - 一些评论强调了 **AIME**、**GPQA** 和 **LiveCodeBench** 等知名机构的外部评估，但也指出 **Grok 3** 的表现可能通过运行多次测试并选择最佳结果而有所偏差。用户呼吁由第三方进行独立测试以验证基准测试结果。


- **Grok 3 发布，在所有类别中排名第一，媲美每月 200 美元的 o1 Pro** ([Score: 152, Comments: 308](https://reddit.com/r/ClaudeAI/comments/1is6ncb/grok_3_released_1_across_all_categories_equal_to/))：**Grok 3** 发布，在包括编程和创意写作在内的所有类别中排名第一，**AIME 得分为 96%**，**GPQA 得分为 85%**。**Karpathy** 将其与**每月 200 美元的 o1 Pro** 进行了比较，指出它有能力尝试解决黎曼猜想等复杂问题，并认为它略优于 **DeepSeek-R1** 和 **Gemini 2.0 Flash Thinking**。
  - 讨论显示出对 **Grok 3** 性能声明的怀疑，用户对 **LMArena** 等基准测试表示不信任，并指出结果中可能存在的偏见。**Grok 3** 被批评需要“**best of N**”答案才能达到 **o1 Pro** 的水平，初步测试表明，与 **GPT-4o** 相比，它在简单编程任务中的表现不佳。
  - 用户对 **Elon Musk** 发表了强烈看法，将其与政治偏见联系起来，并质疑使用与其相关的 AI 模型的伦理影响。对**政治影响**在 AI 设计中的担忧十分普遍，一些用户将其与威权政权相提并论，并表示不愿使用与 Musk 相关的技术。
  - 对话反映了对 **Musk 旗下企业**的普遍不信任情绪，许多用户表示无论性能声明如何，他们都会避免使用他的产品。讨论还涉及了 AI 发展的快速步伐以及竞争在推动创新中的作用，尽管对特定模型存在怀疑。


**主题 2. ChatGPT 与 Claude 在上下文窗口使用上的对比**

- **ChatGPT vs Claude: 为什么 Context Window 大小至关重要。** ([Score: 332, Comments: 60](https://reddit.com/r/OpenAI/comments/1is2bw8/chatgpt_vs_claude_why_context_window_size_matters/)): 该帖子讨论了 AI 模型中 **Context Window 大小**的重要性，并对比了 **ChatGPT** 和 **Claude**。ChatGPT Plus 用户拥有 **32k Context Window** 并依赖 **Retrieval Augment Generation (RAG)**，这在处理较长文本时可能会遗漏细节；而 Claude 提供 **200k Context Window**，无需 RAG 即可捕捉所有细节。一项针对修改版《爱丽丝梦游仙境》的测试显示，由于拥有更大的 Context Window，Claude 在检测错误方面表现出更强的能力，这强调了 OpenAI 为 ChatGPT Plus 用户扩大 Context Window 大小的必要性。
  - **Context Window 大小对比**：用户强调了不同模型间 Context Window 大小的差异，**ChatGPT Plus** 为 **32k**，**Claude** 为 **200k**，而 **Gemini** 在 Google AI Studio 上提供高达 **100-200 万个 Tokens**。**Mistral web** 和 **ChatGPT Pro** 也被提及，其 Context Window 分别接近 **100k** 和 **128k**，表明这些模型在处理大型文档且不丢失细节方面表现更好。
  - **模型性能与使用场景**：**Claude** 因其在处理长文档和理解质量方面的卓越表现而受到赞誉，尤其是在文学编辑和审阅方面。**ChatGPT** 仍被用于小范围的复杂问题和高层规划，而 **Claude** 和 **Gemini** 则因其更大的 Context Window 而在需要广泛上下文的项目中更受青睐。
  - **成本与可访问性**：讨论了模型的性价比，**Claude** 因其在长上下文任务中的负担能力（相比其他模型昂贵的高级选项）而更受欢迎。**Gemini** 被建议作为 AI Studio 上的免费替代方案，用于探索大上下文能力。


- **[Plus 计划的 Context Window 只有 32k？？所有模型都这样吗？](https://i.redd.it/yatn7gyddsje1.png)** ([Score: 182, Comments: 73](https://reddit.com/r/OpenAI/comments/1irynqt/plus_plan_has_a_context_window_of_only_32k_is_it/)): **ChatGPT Plus 计划**提供 **32k Context Window**，与 **Pro** 和 **Enterprise** 等其他计划相比更少。**Free 计划**提供 **8k Context Window**，图片强调了 OpenAI 不同定价计划之间 Context Window 的差异。
  - **Context Window 限制**：用户确认 **ChatGPT Plus 计划**拥有 **32k Context Window**，正如 OpenAI 文档中明确说明的那样，而 **Pro 计划**提供 **128k Context Window**。这一限制导致在处理较长文档时需要使用 **RAG (Retrieval-Augmented Generation)**，这与能够处理更大文本且无需 RAG 的 **Claude** 和 **Gemini** 形成对比。
  - **测试与对比**：**bot_exe** 使用一个 3 万字的文本文件进行了测试，证明 **ChatGPT Plus** 由于依赖 RAG 而遗漏了错误，而 **Claude Sonnet 3.5** 通过利用其 **200k Tokens Context Window** 准确识别了所有错误。这突显了 ChatGPT 的分块检索（chunk retrieval）方法与 Claude 的全面文本摄取相比存在的局限性。
  - **过时信息**：普遍认为 OpenAI 网站包含过时的订阅功能详情，可能会误导用户了解当前的能力。尽管有 **Pro 计划**等更新，但正如多位评论者指出的那样，文档的准确性和时效性仍存疑问。


**主题 3. LLMs 在真实世界软件工程基准测试中的表现**

- **[[R] 在真实世界软件工程任务中评估 LLMs：一项价值 100 万美元的基准研究](https://reddit.com/r/MachineLearning/comments/1isbo6t/r_evaluating_llms_on_realworld_software/)** (得分: 128, 评论: 24): 这项基准研究评估了 **GPT-4** 和 **Claude 2** 等 **LLMs** 在来自 **Upwork**、价值超过 **100 万美元** 的真实世界软件工程任务中的表现。尽管使用了 **Docker** 进行结构化评估并经过专家验证，**GPT-4** 仅完成了 **10.2%** 的编程任务和 **21.4%** 的管理决策，而 **Claude 2** 的成功率为 **8.7%**，这凸显了当前 AI 能力与专业工程环境下的实际效用之间的差距。[完整摘要点击此处](https://aimodels.fyi/papers/arxiv/swe-lancer-can-frontier-llms-earn-dollar1)。论文[点击此处](https://arxiv.org/abs/2502.12115)。
  - **基准测试的局限性**：评论者认为，基准测试性能的提升并不等同于现实世界的实用性，因为像 **GPT-4** 和 **Claude 2** 这样的 AI 模型在特定任务上表现良好，但在工程工作中至关重要的更广泛上下文和决策制定方面表现挣扎。这凸显了理论基准与实际应用之间的差距。
  - **模型表现与误传**：关于被评估的模型存在混淆，据报道 **Claude 3.5 Sonnet** 的表现优于摘要中所述，在 SWE-Lancer Diamond 数据集上赚取了 **$208,050**，但仍未能提供可靠的解决方案。评论者告诫不要在未查阅完整论文的情况下仅凭摘要断章取义。
  - **经济误读**：人们对 AI 表现的经济影响持怀疑态度，评论者指出，完成一部分任务并不等同于取代很大比例的工程人员。由于高错误率和现实世界工程工作的复杂性，AI 完成任务的感知价值受到了质疑。


- **[OpenAI 最新研究论文 | 前沿 LLMs 能在软件工程自由职业中赚到 100 万美元吗？](https://i.redd.it/9jlnz9oi5uje1.png)** (得分: 147, 评论: 40): **OpenAI 的最新研究论文**评估了前沿 **LLMs** 在软件工程任务中的表现，潜在收益为 **100 万美元**。结果显示，**GPT-4o**、**o1** 和 **Claude 3.5 Sonnet** 分别赚取了 **$303,525**、**$380,350** 和 **$403,325**，表明这些模型尚未达到最大潜在收益。
  - **SWE-Lancer 基准测试**评估了 LLMs 在来自 Upwork 的真实任务中的表现，包含 **1,400 个任务**，总价值 **100 万美元**。模型未能解决大部分任务，表明它们在处理复杂软件工程项目方面存在局限性，正如 [OpenAI 的文章](https://openai.com/index/swe-lancer/)中所讨论的那样。
  - **Claude 3.5 Sonnet** 在现实挑战中优于其他模型，凸显了其在 **Agentic coding** 和迭代方面的效率。用户更倾向于在严肃的编程任务中使用 **Claude**，因为它具备处理复杂项目和辅助结对编程的能力。
  - 人们对基准测试的有效性提出了担忧，批评其人工项目设置和无法反映实际场景的指标。任务的成功通常需要反复沟通，而当前的评估框架并未捕捉到这一点。


**主题 4. AI 图像和视频转换技术的进展**

- **[[非樱桃采摘] Skyrocket img2vid (基于 HV) 与 Luma 新 Ray2 模型的对比 - 查看提示词遵循度 (链接见下)](https://v.redd.it/jzm299s49wje1)** ([评分: 225, 评论: 125](https://reddit.com/r/StableDiffusion/comments/1isbytw/noncherrypicked_comparison_of_skyrocket_img2vid/)): **Skyrocket img2vid** 和 **Luma's Ray2** 模型在视频质量和提示词遵循度方面进行了对比。该帖子邀请观看者查看视频对比，重点展示了两个模型在性能上的差异。
  - **Skyrocket img2vid** 因其比 **Luma's Ray2** 更好的提示词遵循度和一致的质量而受到称赞，后者因动作混乱和提示词处理能力差而受到批评。用户注意到 Skyrocket 的“慢速平移 + 动作”与提示词高度契合，提供了更连贯的输出。
  - **技术实现**: Skyrocket 的工作流运行在 **Kijai's Hunyuan wrapper** 上，并提供了 [工作流](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper/blob/main/example_workflows/hyvideo_skyreel_img2vid_testing_01.json) 和 [模型](https://huggingface.co/Kijai/SkyReels-V1-Hunyuan_comfy/tree/main) 的链接。讨论中涉及了 **ComfyUI** 的技术问题，包括节点更新和模型加载错误。
  - **系统要求**: 用户询问了 VRAM 需求以及与 **RTX 3060 12GB** 等硬件的兼容性。此外，还有关于使用 **Linux/Containers** 以获得更一致和高效设置的讨论，并详细解释了使用 Docker 管理依赖项和环境的方法。


- **[新采样技术提升图像质量与速度：RAS - Diffusion Transformers 的区域自适应采样 - SD3 和 Lumina-Next 的代码已可用 (Flux/ComfyUI 什么时候出？)](https://v.redd.it/jj913bi3cwje1)** ([评分: 182, 评论: 32](https://reddit.com/r/StableDiffusion/comments/1isc7yh/new_sampling_technique_improves_image_quality_and/)): **RAS (Region-Adaptive Sampling)** 是一种旨在增强 **Diffusion Transformers** 图像质量和速度的新技术。**SD3** 和 **Lumina-Next** 的代码实现已经发布，人们正期待其与 **Flux/ComfyUI** 的集成。
  - **RAS 实现与兼容性**: 由于 **DiTs** 和 **U-Nets** 之间的架构差异，RAS 无法直接应用于 **Illustrious-XL**。对 RAS 感兴趣的用户应考虑 **Flux** 或 **PixArt-Σ** 等模型，它们与基于 DiT 的系统更兼容。
  - **质量争议与指标**: 关于 RAS 的质量声明存在重大争议，一些用户指出其质量出现了剧烈下降。**QualiCLIP** 分数以及 **SSIM** 和 **PSNR** 等指标表明，生成的图像在细节和结构相似性方面存在实质性损失。
  - **特定模型应用**: RAS 主要针对基于 **DiT** 的模型，不适用于基于 **U-Net** 的模型（如 **SDXL**）。讨论强调了需要针对特定模型的优化策略，以有效利用 RAS。


---

# AI Discord 摘要

> 由 o1-2024-12-17 生成的摘要之摘要的摘要

**主题 1. Grok 3：备受推崇，亦遭诟病** 
 
- [**早期访问引发强烈赞誉**](https://techcrunch.com/2025/02/17/elon-musks-ai-company-xai-releases-its-latest-flagship-ai-grok-3/)：部分用户称其为 *“前沿级别（frontier-level）”*，并声称其表现优于 GPT-4o 和 Claude 等竞争对手，理由是基准测试图表显示 Grok 3 正在快速追赶。另一些人则质疑这些 *“令人惊叹”* 的统计数据，暗示该模型在现实世界的编码和推理方面可能仍然滞后。  
- [**批评者抨击“改变游戏规则”的说法**](https://x.com/karpathy/status/1891720635363254772)：怀疑论者指出其输出重复以及代码执行失误，暗示它在日常任务中 *“并不比 GPT 更好”*。一些省略了特定竞争对手的“梗图级”图表引发了关于 xAI 数据准确性的争论。  
- [**混合审查制度令人意外**](https://lmarena.ai/)：Grok 3 被宣传为无审查，但用户仍然遇到了内容屏蔽。其不可预测的屏蔽行为让许多人质疑 xAI 如何平衡原始输出与安全性。

**主题 2. 前沿基准测试重塑 LLM 测试**  

- [**SWE-Lancer 为真实任务支付 100 万美元**](https://arxiv.org/abs/2502.12115)：OpenAI 的新基准测试包含 1,400 个总额达 100 万美元的 Upwork 任务，强调实际编码场景。模型在大多数任务中仍然失败，暴露了 *“AI 炒作”* 与真实自由职业收入之间的差距。  
- [**Native Sparse Attention 惊艳 HPC 圈**](https://arxiv.org/abs/2502.11089)：研究人员提出了硬件对齐的 *“动态分层稀疏性（dynamic hierarchical sparsity）”* 以处理长上下文（long contexts），承诺在成本和速度上取得双赢。工程师预见使用 NSA 将为下一代 AI 工作负载带来巨大的吞吐量提升。  
- [**Platinum 基准测试关注可靠性**](http://platinum-bench.csail.mit.edu/)：新的 *“platinum”* 评估最小化了标签错误，并限制每个模型对每个查询仅有一次尝试机会（one-shot）。这剥离了能力的幻觉，迫使 LLM *“坚持其第一次尝试的结果”*。

**主题 3. AI 工具拥抱代码调试** 

- [**Aider, RA.Aid 和 Cursor 带来惊喜**](https://aider.chat/docs/install/docker.html)：这些项目允许 LLM 添加缺失文件、修复构建中断或 *“联网搜索”* 以获取代码见解。虽然仍存在细微的怪癖——比如 Aider 无法自动添加文件名——但开发者看到了在连接文档、代码和 AI 方面的巨大潜力。  
- [**VS Code 扩展消灭 Bug**](https://github.com/jasonjmcghee/claude-debugs-for-you)： “claude-debugs-for-you” MCP 服务器可以在运行中显示变量状态，击败了 *“盲目的日志猜测”*。它为基于语言的、跨 Python、C++ 等语言的交互式调试铺平了道路。  
- **Sonnet 与 DeepSeek 竞争编码能力**：开发者讨论将价格昂贵但值得信赖的 *Sonnet 3.5* 与 *DeepSeek R1* 进行比较，尤其是在编码任务中。有些人更喜欢更大的模型上下文，但 *“成本 vs 性能”* 的争论依然激烈。

**主题 4. 实验室启动下一代 AI 项目**  

- [**Thinking Machines Lab 公开亮相**](https://thinkingmachines.ai/)：由来自 *ChatGPT* 的 Mira Murati 等人共同创立，该实验室誓言坚持开放科学和 *“促进公众理解的进步”*。Karpathy 的认可预示着新一波创意 AI 扩张浪潮的到来。  
- [**Perplexity 发布采用“自由微调”的 R1 1776**](https://huggingface.co/perplexity-ai/r1-1776)：这款 *“无审查但基于事实”* 的模型人气飙升，被贴上了 *“自由模式”* 的标签。用户接受了这种带有讽刺意味的爱国主题，赞扬其在减少限制性输出方面的尝试。  
- **Docling 与 Hugging Face 联手打造视觉 LLM**：[Docling 的 IBM 团队](https://github.com/DocLing-lab)旨在将 SmolVLM 嵌入到高级文档生成流中，合并文本和图像任务。GitHub 上的 PR 预示着这些视觉文档将如何实时工作。

**主题 5. GPU 与 HPC 提升助力模型创新**  

- [**VLLM 中的动态 4-bit 量化**](https://github.com/vllm-project/vllm/pull/12974)：Unsloth 的 4-bit 量化技术进入主分支，大幅降低了 VRAM 需求。部分层跳过（Partial layer skipping）驱动了 HPC 圈对进一步内存优化的兴趣。  
- **AMD vs. Nvidia：AI 芯片之战**：AMD 的 *Ryzen AI MAX* 和 *M4 edge* 以 *“低功耗”* 硬件威胁着 GPU 王者的地位。爱好者期待 *5070 系列* 继续推动桌面端的 HPC 发展。  
- **爆发式的双模型推理**：工程师们正在实验使用小模型进行推理，再加上大模型进行最终输出，尽管这种方法需要自定义编排（orchestration）。*LM Studio* 尚未正式合并这一概念，因此精通 HPC 的编码人员正在 *“手动链接模型”*。


---

# 第 1 部分：高层级 Discord 摘要

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Grok 3 展现出潜力**：用户正在将 **Grok 3** 与 **Claude 3.5 Sonnet** 以及 **OpenAI** 的 **O1** 进行对比，重点关注上下文长度和编程能力，但在内容审核和网页搜索功能方面也存在局限性。
   - 尽管对价格（相比 **DeepSeek**）有所顾虑，一些用户仍渴望测试 **Grok 3** 及其功能，讨论涉及模型能力和局限性透明度的重要性。
- **PRO 模式性能波动**：用户报告 **PRO 模式** 体验不一致，指出有时速度和质量有所下降。
   - 虽然偶尔会出现小的 **PRO 计时器**，但性能有时类似于效果较差的 mini 模型，这表明运行质量存在波动。
- **GPT 在执行 Pip 时出现故障**：一位用户报告在 **GPT 环境** 中运行 **pip** 命令时遇到困难（该功能此前可用），并寻求社区帮助。
   - 一位成员建议说：*please find a way to run commands in python code and using it run !pip list again*，而另一位成员提到成功执行了 **!pip list** 并通过[此链接](https://chatgpt.com/share/67b4a7eb-56ec-800e-adad-2dd9bcbc3451)分享了他们的解决方案。
- **4o 的文本阅读器引发好奇**：一位用户询问了 **4o 文本阅读器** 的起源，特别是它是否在独立的神经网络上运行。
   - 他们还质疑了长对话线程中生成文本的稳定性，以及训练过的语音是否会影响这种稳定性。
- **请求置顶对话功能**：一位用户建议实现一个功能，将**常用对话置顶**到聊天记录顶部。
   - 这一增强功能将为经常进行特定对话的用户简化访问流程。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **VLLM 添加了 Unsloth 的量化**：**VLLM** 现在支持 **Unsloth 的动态 4-bit 量化**，可能提升性能，详见 [Pull Request #12974](https://github.com/vllm-project/vllm/pull/12974)。
   - 社区讨论了在这些创新背景下管理 **LLM** 的内存分析和优化挑战。
- **Notebook 在 Colab 之外运行困难**：用户报告由于依赖项错误，在 **Google Colab** 之外（尤其是 **Kaggle**）运行 **Unsloth notebooks** 时存在**兼容性问题**；参考 [Google Colab](https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH?usp=sharing#scrollTo=5uwPWn_fCGFo)。
   - 问题可能源于使用了 **Colab 特有的命令结构**，例如 `%` 与 `!`。
- **Unsloth 微调 Llama 3.3**：新指令详细说明了如何通过修改模型名称，使用现有 notebooks 微调 **Llama 3.3**，参考这篇 [Unsloth 博客文章](https://unsloth.ai/blog/llama3-3)。
   - 然而，用户应为有效训练所需的巨大 **VRAM 需求** 做好准备。
- **NSA 机制改进训练**：一篇论文介绍了 **NSA**，这是一种硬件对齐的稀疏注意力机制，能更好地进行长上下文训练和推理，详见 [DeepSeek 的推文](https://x.com/deepseek_ai/status/1891745487071609327?t=HNWC7CR7kGFGnruYRA14MA&s=19)。
   - 论文指出 **NSA** 可以媲美或超越传统模型的性能，同时通过动态稀疏策略降低成本。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Grok 3 性能表现不及预期**：用户对 **Grok 3** 的表现表示失望，认为其未达到预期，也无法与 **GPT-4o** 等模型相比，部分用户在被宣传吸引之前正等待更全面的测试。
   - **Musk 声称 Grok 3** 在一项独立基准测试研究的支持下超越了所有竞争对手，在 AI 能力方面具有竞争优势，尽管用户的实际体验各不相同。
- **Deep Research 产生幻觉**：多位用户报告称 Perplexity 中的 **Deep Research** 会产生幻觉结果，引发了对其准确性（相比 **o3-mini** 等免费模型）的担忧。
   - 用户质疑了 **Reddit** 等来源的可靠性以及 API 更改对信息质量的影响，这些因素都影响了最终的输出。
- **订阅转售遇冷**：一名试图出售其 **Perplexity Pro 订阅** 的用户因其他地方价格更低而面临挑战。
   - 讨论显示了对转售订阅的怀疑态度，由于缺乏市场价值，建议保留自用。
- **生成式 AI 开发角色兴起**：一篇新文章探讨了即将到来且不断演变的**生成式 AI 开发角色**，强调了它们在技术领域的重要性。
   - 文章强调了对与 AI 进步相匹配的技能的需求，以有效利用这些新兴机会，这可能会重塑人才优先事项。
- **调查 Sonar API 热切换**：有人提出了关于 **R1-1776 模型** 是否可以在 **Sonar API** 上进行原位热切换的问题，这表明了 **OpenRouter** 社区的兴趣。
   - 这一咨询表明，围绕 Sonar API 框架的灵活性和功能（可能用于增强定制化）正在进行持续讨论。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **9b 模型击败 340b 模型，社区震惊**：一个 **9b 模型** 的表现优于 **340b 模型**，引发了关于 AI 模型评估和性能指标的讨论。
   - 社区成员对这种意想不到的性能提升及其影响表示惊讶和兴趣。
- **移植 Colab Notebook 导致运行时故障**：一名成员成功从 Linaqruf 的 notebook 移植了他们的 Colab，但在从 **Hugging Face** 获取元数据时遇到了 `ImportError` 运行时错误。
   - 该用户在排查故障时表示：“*我可能无法按预期工作，因为我忘了向 Gemini 怪物询问 path/to...*”，暗示存在未解决的路径问题。
- **Docling 与 Hugging Face 合作开发 Visual LLM**：来自 **IBM** 的 **Docling** 已与 **Hugging Face** 合作，将 **Visual LLM** 功能与 **SmolVLM** 集成到 Docling 库中。
   - 此次集成旨在通过先进的视觉处理增强文档生成，Pull Request 很快将在 **GitHub** 上发布。
- **Neuralink 的图片引发社区兴奋**：近期与 **Neuralink** 相关的图片得到了分析，展示了其正在进行的研发进展。 
   - 来自 [图片 1](https://cdn.discordapp.com/attachments/898619964095860757/1341471798776168580/SCR-20250218-qqsx.png)、[图片 2](https://cdn.discordapp.com/attachments/898619964095860757/1341471799120236644/SCR-20250217-ltbt.png) 和 [图片 3](https://cdn.discordapp.com/attachments/898619964095860757/1341471799522754602/SCR-20250217-ltdg.png) 的视觉洞察引发了社区对未来影响的兴奋。
- **AI Agents 课程证书问题**：多位用户报告了证书生成错误，通常会收到请求过多的消息。
   - 虽然有人建议使用无痕模式或不同的浏览器，但成功率仍然不稳定。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **MCP 教程引发关注**：一名成员发布了关于如何使用 **MCP** 的**入门指南**，可在 [X](https://x.com/windsurf_ai/status/1891664001941037123) 上查看，鼓励社区探索和讨论。
   - 该教程旨在帮助新手有效使用 **MCP** 功能，并鼓励分享个人用例以增进理解。
- **Codeium Write 模式调整**：最近的更新显示，免费计划不再提供 **Write mode**，导致用户考虑升级或切换到仅聊天模式。
   - 这一变化引发了关于 **Write mode** 的缺失是对所有用户还是特定用户永久生效的争论，引起了社区的担忧。
- **IntelliJ Supercomplete 功能之谜**：成员们讨论了 **IntelliJ** 扩展是否曾拥有 **supercomplete** 功能，并参考了其在 **VSCode pre-release** 中的存在。
   - 澄清建议指出，**supercomplete** 指的是多行补全，而 **autocomplete** 则涵盖单行建议。
- **寻求简化的 Codeium 部署**：一位用户询问如何为使用 **IntelliJ** 和 **Codeium** 的多位用户自动化设置，希望能有简化的身份验证流程。
   - 回复指出该功能可能属于企业级服务，建议通过 [Codeium Contact](https://codeium.com/contact/enterprise) 联系 **Codeium** 的企业支持。
- **Cascade Base 性能表现不佳**：用户报告 **Cascade Base** 无法对代码进行编辑，且有时 AI 聊天窗口会消失，令人感到沮丧。
   - 多位用户在尝试使用模型时遇到了内部错误，这表明平台存在持续的稳定性问题；相关文档可在 [Windsurf Advanced](https://docs.codeium.com/windsurf/advanced) 查看。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grok 3 的炒作引发质疑**：**Grok 3** 的发布引发了辩论，一些帖子声称它超越了 **GPT** 和 **Claude**，正如 [Andrej Karpathy 的评价](https://x.com/karpathy/status/1891720635363254772)所提到的。
   - 许多用户仍持怀疑态度，认为关于突破性性能和 **AGI** 的说法过于夸大。
- **Aider 文件添加出现故障**：用户报告在使用 Docker 时，尽管命名和初始化正确，**Aider** 仍无法自动添加文件，详见 [Aider 的 Docker 安装文档](https://aider.chat/docs/install/docker.html)。
   - 社区正在讨论故障排除步骤以及 **Aider** Web 界面中潜在的 Bug。
- **Gemini 模型集成测试开始**：社区成员正在探索在 **Aider** 中使用 **Gemini** 实验性模型及其 [实验性模型文档](https://ai.google.dev/gemini-api/docs/models/experimental-models)。
   - 关于正确的模型标识符和实施过程中收到的警告仍存在困惑，目前正通过一个 [Pull Request](https://github.com/Aider-AI/aider/pull/2628) 尝试成功使用 **Mixture of Architects (MOA)**。
- **RA.Aid 为 Aider 增强网络搜索功能**：针对将网络搜索引擎集成到 **Aider** 的兴趣，有分享称 **RA.Aid** 可以与 **Aider** 集成并利用 **Tavily API** 进行网络搜索，详见其 [GitHub 仓库](https://github.com/ai-christianson/RA.Aid/)。
   - 这模仿了 **Cursor** 当前的实现，提供了类似的搜索功能。
- **Ragit GitHub 流水线引起关注**：GitHub 上的 **Ragit** 项目被描述为“类 Git 的 RAG 流水线”，引起了广泛关注，参见 [仓库](https://github.com/baehyunsol/ragit)。
   - 成员们强调了其在 **RAG** 处理流程中的**创新方法**，有可能简化数据检索和处理。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Grok 3 评价两极分化**：用户对 **Grok 3** 的初步反响褒贬不一，一些用户印象深刻，而另一些用户则持怀疑态度，尤其是将其与 **Claude** 和 **Sonnet** 等模型对比时。
   - 关注点集中在 **Grok 3** 的代码和推理能力上，正如 [TechCrunch 对 Grok 3 的报道](https://techcrunch.com/2025/02/17/elon-musks-ai-company-xai-releases-its-latest-flagship-ai-grok-3/) 中所强调的那样。
- **OpenRouter API 政策不明朗**：围绕 **OpenRouter** 的 API 使用政策存在困惑，特别是关于 **NSFW** 内容生成以及对供应商政策的合规性，一些用户声称 **OpenRouter** 相比其他平台限制更少。
   - 为了明确起见并获取最新政策，建议用户咨询 **OpenRouter** 管理员并查看 [API Rate Limits 文档](https://openrouter.ai/docs/api-reference/limits)。
- **Sonnet 是编程领域的 MVP**：讨论对比了 **DeepSeek**、**Sonnet** 和 **Claude** 等多种模型的表现，一些用户表示在编程任务中更倾向于 **Sonnet**，因为它更可靠。
   - 尽管成本较高，但 **Sonnet** 的可靠性使其成为特定编程应用的首选，而用户在考虑价格和性能因素时，会将 **Grok 3** 和 **DeepSeek** 作为具有竞争力的选项。
- **LLM 获得视觉能力**：一位用户询问了 **OpenRouter** 上可以分析图像的模型，并引用了供应商网站上详细介绍具有**文本和图像**处理能力模型的模块部分。
   - 建议该用户在 [OpenRouter 模型页面](https://openrouter.ai/models) 的该板块下探索可用模型，以寻找合适的选项。
- **OpenRouter 额度购买故障**：一位用户在 **OpenRouter** 上购买额度时遇到问题，在咨询银行后寻求帮助，这引发了关于不同 **LLM 模型** 定价的讨论。
   - 讨论包括对这些模型所产生价值的辩论，以及如何通过性能和能力来证明其成本的合理性。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Grok 3 性能不及预期**：用户对 **Grok 3** 的表现表示失望，理由是存在重复内容和代码执行力弱，即使在 [获得早期访问权限](https://x.com/karpathy/status/1891720635363254772) 之后也是如此。
   - 一些用户认为 **Grok 3** 只是追赶上了 **Sonnet** 等现有模型，尽管在推理能力方面有一些正面反馈。
- **Sonnet 依然表现出色**：尽管有报告称幻觉有所增加，但许多用户在编程任务中仍然更青睐 **Sonnet**（尤其是 **3.5** 版本），而非 **Grok 3** 等新模型。
   - 用户认为 **Sonnet 3.5** 在可靠性和性能方面仍优于 **Grok 3**。
- **MCP Server 设置依然复杂**：用户讨论了为 AI 工具设置 **MCP Server** 的复杂性，并引用了一个 [针对 Perplexity API 的 MCP Server](https://github.com/daniel-lxs/mcp-perplexity)。
   - 为了提高灵活性，目前正在尝试使用单文件 Python 脚本作为传统 **MCP Server** 设置的替代方案，如 [single-file-agents](https://github.com/disler/single-file-agents) 所示。
- **Cursor 在处理大型代码库时遇到困难**：用户报告 **Cursor** 在大型代码库的规则和上下文管理方面面临问题，需要手动添加规则。
   - 一些用户发现降低 **Cursor** 版本有助于提高性能，而另一些用户则在尝试 [auto sign cursor](https://github.com/chengazhen/cursor-auto-free)。
- **AI 的指令遵循能力受到质疑**：用户对 AI 模型是否正确处理其指令表示担忧，一些人建议在 Prompt 中包含显式检查。
   - 建议采用一些奇特的方法来测试 AI 对引导的遵循程度，这表明需要深入了解 AI 模型在不同平台上处理指令的方式。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Grok 3 的 Vibe Check 通过但深度欠缺**：**Grok 3** 的早期评价表明它达到了前沿级标准，但被认为过于“中规中矩（vanilla）”，在技术深度和注意力压缩（attention compression）方面表现吃力，尽管其早期版本在 [Arena 排行榜](https://x.com/lmarena_ai/status/1891706264800936307)上打破了记录。
   - 评论者指出，**Grok 3** 在提供深度技术解释方面比较吃力，并且在处理较长查询时存在注意力压缩问题，尤其是与 **R1** 和 **o1-pro** 相比。
- **扎克伯格废弃 Llama 4**：尽管扎克伯格宣布 **Llama 4** 的预训练已经完成，但有推测称，由于受到来自 **DeepSeek** 等竞争对手的反馈影响，该模型可能会被废弃。
   - 竞争压力将严重影响 **Llama 4** 的最终发布策略，这凸显了 AI 发展的飞速节奏，该模型有可能作为一款全能的多模态 omni-model 发布。
- **Thinking Machines Lab 与 Murati 共同重出江湖**：由 Mira Murati 等行业领袖共同创立的 **Thinking Machines Lab** 已经启动，其使命是弥补 AI 可定制性和理解力方面的差距。更多详情请见其[官方网站](https://thinkingmachines.ai/)。
   - 该实验室致力于开放科学，计划优先考虑人机协作，旨在使各种领域的进步变得触手可及，这引发了人们将其与历史上的 **Thinking Machines Corporation** 进行对比。
- **关于评估方法论有效性的辩论爆发**：正在进行的讨论质疑当前 **eval** 方法的可靠性，认为行业缺乏强大的测试框架，且某些模型在 **MMLU Benchmark** 中存在“刷分（gaming）”行为。
   - 一些公司因关注表面的性能指标而非真正反映能力的实质性测试而受到指责，呼吁建立新的基准测试，例如 [SWE-Lancer](https://arxiv.org/abs/2502.12115)，该基准包含来自 Upwork 的 1,400 个自由软件工程任务，价值 100 万美元。
- **GPT-4o Copilot 加速编程**：一款新的代码补全模型 **GPT-4o Copilot** 正在多个 IDE 中进行公开预览，承诺提高主要编程语言的编码效率。
   - 根据 [Thomas Dohmke 的推文](https://x.com/ashtom/status/1891925306430337110)，该模型在超过 1T token 的海量代码语料库上进行了 Fine-tuned，旨在优化编程工作流，为开发者提供更高效的工具。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **DeepSeek 模型加载失败**：用户报告了 **DeepSeek** 模型加载错误，原因是超参数 *invalid n_rot: 128, expected 160*。
   - 错误日志包含了内存、GPU（如 **3090**）和操作系统详情，暗示了硬件或配置限制，但其推理能力使其在正常工作时成为处理技术任务的首选。
- **双模型推理受到关注**：成员们一直在尝试使用较小的模型进行推理，使用较大的模型进行输出，并详细介绍了该领域过去成功的项目。
   - 虽然这个概念很有前景，但由于 **LM Studio** 缺乏对双模型推理的直接支持，其实现需要手动编写代码。
- **Whisper 模型设置**：用户讨论了利用 **Whisper** 模型进行转录，并根据 CUDA 版本推荐了特定的设置。
   - 在使用 **Whisper** 时，正确的配置（特别是关于 **CUDA** 兼容性）至关重要。
- **AMD 加入讨论**：提到了 AMD **Ryzen AI MAX** 的推出，其性能正与 Nvidia GPU 进行对比。此外，根据[此评论](https://www.youtube.com/watch?v=v7HUud7IvAo)，**M4 edge** 和即将推出的 **5070** 承诺在降低功耗的同时提供高性能。
   - 用户正在将其与 Nvidia GPU 进行比较，并讨论 **AMD** 新硬件的潜在性能。
- **旧款 Tesla K80 集群化？**：AI 工程师讨论了将旧款 **Tesla K80 GPU** 进行集群化以利用其巨大 VRAM 的可行性，并对能效表示了担忧。
   - 分享了在集群设置中使用 **Exo**（涉及 PC 和 MacBook）的经验，并指出了同时加载模型时出现的问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **DeepSeek v2 的 Centroid 路由本质上就是向量权重**：一位用户试图确认在 **DeepSeek v2** 的 MoE 架构语境下，“centroid” 是否指的是路由向量权重。
   - 另一位用户确认了这一理解，并阐明了其在模型专家专业化（expert specialization）中的作用。
- **Platinum Benchmarks 衡量 LLM 可靠性**：一篇新论文引入了 “platinum benchmarks” 的概念，通过最小化标签错误来改进 **LLM reliability** 的衡量。
   - 讨论集中在该基准测试的局限性上，特别是它对每个模型的每个问题仅取一个样本，引发了对其整体有效性的质疑。
- **JPEG-LM 直接从字节生成图像**：最近的研究工作应用自回归模型直接从 **JPEG bytes** 生成图像，详见 [JPEG-LM paper](https://arxiv.org/abs/2408.08459)。
   - 这种方法利用了自回归 LLM 架构的通用性，可能更容易集成到多模态系统中。
- **NeoX 在每秒 Token 数上落后于 NeMo**：一位用户在 80B A100 GPU 上对 **NeoX** 进行了基准测试，结果为 **19-20K TPS**，而 **NeMo** 在类似模型上的协作测试达到了 **25-26K TPS**。
   - 小组认为中间模型大小的影响以及 NeMo 中 TP 通信重叠的可能性是影响性能差异的因素。
- **追踪模型大小、数据和微调以研究 Scaling Laws**：为了清晰掌握 Scaling Laws，*Stellaathena* 建议追踪 **model size**、**data** 和 **finetuning methods**。
   - 这种方法解决了对数据使用的模糊感，并有助于更好的模型比较。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Grok 3 引发褒贬不一的反应**：随着 **Grok 3** 的发布，成员们对其推理能力和 API 成本感到好奇，尽管根据 [Elon Musk 的这条推文](https://x.com/elonmusk/status/1891388509191049307)，一些早期评估表明它在某些方面可能落后于 **OpenAI's models**。
   - **Grok-3** 的现场演示引发了讨论，根据[这段视频](https://www.youtube.com/watch?v=b0XI-cbel1U)，一些人表示对产品的演示感到“非常平淡（thoroughly whelmed）”。
- **NURBS 在 AI 领域受到关注**：成员们详细介绍了 **NURBS** 的优势，指出与传统方法相比，它们具有 *smoothness*（平滑性）和更好的数据结构，对话转向了几何方法在 AI 开发中的更广泛影响。
   - 小组还讨论了通过利用几何优先方法来减少 *lesser overfitting*（过拟合）的潜力。
- **模型对比引发争论**：关于 **Grok** 和 **OpenAI's models** 之间比较的公平性存在争论，一些人声称 **Grok's charts** 歪曲了其性能，如[这条推文](https://fxtwitter.com/12exyz/status/1891723056931827959?t=auDqsxYvLMups5PYOtJGfQ&s=19)所示。
   - 对 **'maj@k' methods** 及其如何影响感知有效性的担忧已经出现，引发了关于 **model evaluation standards** 的讨论。
- **Deepsearch 定价受到质疑**：尽管被描述为最先进的技术，但新产品 **Deepsearch** **$40** 的定价引起了用户的怀疑，尤其是考虑到最近发布的类似产品是免费的。
   - 一位成员愤世嫉俗地观察到，考虑到竞争情况，这种激进的定价策略可能具有剥削性。
- **社区计划构建层级化论文树**：有人建议建立一个专注于依赖关系和关键见解的开创性论文 **hierarchical tree**（层级树），强调 *filtering information*（过滤信息）以避免噪音的重要性。
   - 社区表示希望利用他们在识别 **seminal and informative papers**（开创性和信息丰富的论文）方面的专业知识，以实现更好的知识共享。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Xformers 引发 GPU 困扰**：用户报告了 *xformers* 默认进入 CPU 模式并要求特定 PyTorch 版本的问题，但一位成员建议忽略警告信息。
   - 建议通过干净的重新安装并在命令行中添加 `--xformers` 作为潜在的修复方案。
- **InvokeAI 是初学者的首选 UI**：许多用户建议将 *InvokeAI* 作为新手的直观 UI 选择，尽管个人更倾向于功能更强大的 *ComfyUI*。
   - 社区成员普遍认为 *InvokeAI* 的简单性使其更易上手，因为像 *ComfyUI* 这样复杂的系统可能会让新手感到无所适从。
- **Stable Diffusion 更新停滞引发不满**：用户对 *Stable Diffusion* 生态系统缺乏更新表示担忧，特别是在 A1111 停止支持 SD 3.5 之后，导致了用户的不满。
   - 用户因指南过时以及分支与新技术不兼容而感到困惑，从而产生了挫败感。
- **动漫性别分类器引发好奇**：一位用户寻求指导，想了解如何使用来自 [Hugging Face](https://huggingface.co/DOFOFFICIAL/animeGender-dvgg-0.8) 的 *动漫性别分类器* 来分离男性和女性动漫 bboxes 以进行 inpainting。
   - 据报道，该分类方法很有前景，但需要具备将其与 ComfyUI 现有工作流集成的专业知识。
- **打印机易用性差引发困惑**：一位 IT 工作者分享了一个关于缺陷打印机的幽默轶事，指出即使是极其清晰的指令也可能被误解，从而引起混乱。
   - 讨论转向了对简单标志的普遍误解以及对更清晰沟通的需求。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **深入探讨 Torch Compile**：一位成员解释说，当使用 `torch.compile()` 时，PyTorch 中的机器学习模型会转换为 Python 字节码，由 **TorchDynamo** 进行分析，并为 GPU kernel 调用生成 **FX graph**；成员们对处理编译过程中的 graph breaks 表现出浓厚兴趣。
   - 分享了一个指向[详细 PDF](https://github.com/pytorch/workshops/blob/master/ASPLOS_2024/inductor.pdf) 的链接，该文件专注于 PyTorch **Inductor** 的内部机制，提供了深入的信息。
- **CUDA 内存传输获得异步加速**：为了提高在 CUDA 中传输大型常量向量的速度，建议使用 [`cudaMemcpyAsync`](https://docs.nvidia.com/cuda/cudaMemcpyAsync.html) 配合 `cudaMemcpyDeviceToDevice` 以获得更好的性能。
   - 在需要在 **A100 GPU** 上对数据拷贝进行更精细控制的场景下，建议使用 [`cub::DeviceCopy::Batched`](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceCopy.html) 作为解决方案。
- **Triton 3.2.0 打印意外的 TypeError**：在 PyTorch 的 Triton 3.2.0 包中使用 `TRITON_INTERPRET=1` 配合 `print()` 会导致 **TypeError**，提示 `kernel()` 得到了一个意外的关键字参数 'num_buffers_warp_spec'。
   - 一位成员正在解决由使用 **float8** 等**低精度输入**触发的 Triton kernel 性能下降问题，指出 shared memory 访问的 bank conflicts 增加，且 Triton 相比 CUDA 的粒度控制，缺乏解决 bank conflicts 的资源。
- **Native Sparse Attention 获得好评**：讨论围绕关于 **Native Sparse Attention** 的论文展开（[论文链接](https://arxiv.org/abs/2502.11089)），指出其硬件对齐和可训练性可能会彻底改变模型效率。
   - 与 **GELU** 和 **Swish** 相比，**GoLU 激活函数** 减少了 latent space 的方差，同时保持了稳健的梯度流（参见 [论文链接](https://arxiv.org/abs/2502.03654v1)）。
- **Dynasor 大幅削减推理成本**：**Dynasor** 在无需模型训练的情况下，将推理系统成本降低了高达 **80%**，通过利用确定性来停止不必要的推理过程，展示了令人印象深刻的 token 效率（参见 [demo](https://hao-ai-lab.github.io/demo/dynasor-cot)）。
   - **vllm** 中对 **hqq** 的增强支持允许运行更低比特的模型，并可通过 **GemLite** 或 PyTorch 后端对几乎任何模型进行即时量化；新版本的发布伴随着极具吸引力的补丁功能，承诺在 **vllm** 各个分支中实现更广泛的兼容性（参见 [Mobius Tweet](https://x.com/Mobius_Labs/status/1891888285544333607)）。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Grok-3 表现出不一致的审查制度**：尽管最初的印象并非如此，但 **Grok-3** 根据使用场景（如在 lmarena 上）呈现出不同程度的审查水平。
   - 虽然*有些人感到惊喜*，但另一些人则在寻求获取其原始输出，这反映了社区对未过滤 AI 回复的兴趣。
- **SWE-Lancer 基准测试提出真实世界的挑战**：[SWE-Lancer 基准测试](https://arxiv.org/abs/2502.12115) 包含超过 **1,400 个来自 Upwork 的自由软件工程任务**，总价值达 **100 万美元**，涵盖了从错误修复到功能实现的各种任务。
   - 评估显示，前沿模型在任务完成方面表现挣扎，强调了在软件工程领域提升性能的需求。
- **Hermes 3 的审查声明引发争议**：虽然宣传为**无审查**，但据报道 **Hermes 3** 拒绝回答某些问题，需要特定的系统提示词（system prompts）才能获得所需的响应。
   - 有推测认为，通过系统提示词进行引导对其预期功能的实现是必要的，这引发了关于平衡自由与实用性的讨论。
- **Alignment Faking 引发伦理担忧**：一段关于 **“Large Language Models 中的 Alignment Faking（对齐伪装）”** 的 [YouTube 视频](https://www.youtube.com/watch?v=9eXV64O2Xp8) 探讨了个人如何伪装共同价值观，这与 AI 对齐中的挑战相呼应。
   - 这种行为被比作 AI 模型在解释和模仿对齐时面临的复杂性，引发了 AI 开发中的伦理思考。
- **开源 LLM 预测老鹰队将赢得超级碗**：一个开源的 LLM 驱动的 [pick-em's bot](https://github.com/stevekrenzel/pick-ems) 预测 **老鹰队（the Eagles）** 将在超级碗中获胜，其表现优于 ESPN 2024 年竞赛中 **94.5%** 的选手。
   - 该机器人称 *“老鹰队是逻辑上的选择”*，突显了利用结构化输出进行推理的新颖方法，展示了在体育预测方面的潜力。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Thinking Machines Lab 强势登场**：由 AI 泰斗创立的 Thinking Machines Lab 旨在使 AI 系统民主化，并弥合公众对前沿 AI 的理解，该消息已在 [Twitter 上公布](https://x.com/thinkymachines/status/1891919141151572094)。
   - 该团队包括 **ChatGPT** 和 **Character.ai** 的创建者，致力于通过发表论文和发布代码来实现开放科学，正如 [Karpathy 所指出的](https://x.com/karpathy/status/1891938714915569711)。
- **Perplexity 发布 Freedom-Tuned R1 模型**：Perplexity AI 开源了 **R1 1776**，这是 **DeepSeek R1 模型** 的一个版本，经过后训练以提供无审查且事实性的信息，最初在 [Twitter 上宣布](https://x.com/perplexity_ai/status/1891916573713236248)。
   - 社交媒体用户将其标记为“自由微调（freedom tuning）”，该模型的用途受到了热情的欢迎（如[此 gif 所示](https://tenor.com/view/freedom-america-gif-15593845046973100361)）。
- **OpenAI 推出 SWE-Lancer 基准测试**：OpenAI 引入了 **SWE-Lancer**，这是一个使用价值 100 万美元的 1,400 个自由软件工程任务来评估 AI 编程性能的新基准测试，并在 [Twitter 上宣布](https://x.com/openai/status/1891911132983722408?s=46)。
   - 社区对这个名字表示惊讶，并对某些模型的缺席进行了推测，暗示了该基准测试背后的战略动机。
- **Grok 3 热度持续升温**：社区讨论了即将推出的 **Grok 3** 模型的潜在能力及其在各种任务中的应用。
   - 尽管感到兴奋，但怀疑态度依然存在，一些人幽默地怀疑其相对于预期的实际表现，根据[此 Twitter 线程](https://x.com/amasad/status/1891709057238507526?s=46)。
- **Zed 的编辑预测模型加入竞争**：根据其[博客文章](https://zed.dev/blog/edit-prediction)，**Zed** 发布了一个开源的下一行编辑预测模型，将其定位为 **Cursor** 等现有解决方案的潜在对手。
   - 用户表达了对其与 **Copilot** 等成熟模型及其高级功能相比的差异化和整体实用性的担忧。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **播客重启 Max Headroom**：一位用户详细介绍了他们使用 **Notebook LM** 进行文本生成和 **Speechify** 进行语音克隆来创建 **Max Headroom** 播客的过程；整个制作耗时 **40 小时**。
   - 该用户在 [YouTube](https://youtu.be/snDzpZBH8v0) 上分享了他们的《Max Headroom Rebooted 2025 Full Episode 20 Minutes》链接。
- **LM 音频规格引发辩论**：用户探讨了 **Notebook LM** 的 **audio generation** 能力，强调了其在创建播客和广告方面的实用性，并指出了节省时间和脚本修改灵活性的优势。
   - 一位用户报告称，较旧的付费账户产生的结果不尽如人意，尤其是在处理 MP4 文件时。
- **Google 产品设计招致批评**：参与者对 **Google 的系统设计** 局限性表示担忧，强调随着服务转为付费，用户期望有更好的表现。
   - 他们对影响用户体验的 **technological shortcomings** 表示沮丧。
- **关于播客创建限制的困惑**：用户讨论了一个与意外的播客创建限制相关的 bug，最初认为上限是 **三个播客**，而不是针对 chat queries 的限制。
   - 其他人澄清实际限制是 **50 个 chat queries**，消除了最初的困惑。
- **播客语调需要更多调整**：一位用户询问如何修改播客的语调和长度，结果得知这些调整主要适用于 **NotebookLM** 的回复，而非播客。
   - 随后对话转向配置聊天设置以增强用户满意度，本质上是利用 chat prompts 来改善播客的语调。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Polars 与 Pandas 之争升温**：成员们对用于 dataframe 操作的 **Polars** 库感到兴奋，因为它比 **pandas** 具有性能优势，一位成员表示这对于未来的 dataframe 工作将是 *必不可少的*。
   - 将 **Polars** 与 Mojo 项目集成的兴趣被激发，这为在单机和分布式环境中使用 **silicon agnostic** 的 dataframe 库打开了大门。
- **Gemini 2.0 作为编程助手**：成员建议利用 `Gemini 2.0 Flash` 和 `Gemini 2.0 Flash Thinking` 进行 Mojo 代码重构，理由是它们对 Mojo 的理解，并推荐了带有 Mojo 扩展的 **Zed** 代码编辑器，链接见 [这里](https://github.com/freespirit/mz)。
   - 然而，一位成员表示在将大型 Python 项目重构为 Mojo 时遇到困难，建议由于当前工具的局限性（受限于 Mojo 的 borrow checker 约束），可能需要进行手动更新。
- **在 Mojo 中使用 Enzyme 进行 Autodiff？**：社区讨论了在 Mojo 中使用 [Enzyme](https://github.com/EnzymeAD/Enzyme) 实现 autodifferentiation，并提出了支持它的提案，同时权衡了将 MOJO ASTs 转换为 MLIR 进行优化的挑战。
   - 关键挑战在于如何在遵守 Mojo 内存管理约束的同时实现这一目标。
- **全局变量仍不确定**：成员们对 Mojo 未来是否支持 **global variables** 表示不确定，其中一位幽默地请求支持 **global expressions**。
   - 该请求突显了社区对变量作用域更大灵活性的渴望，但不确定这是否能与 Mojo 的内存安全保证相融合。
- **List 因速度问题受到质疑**：社区质疑使用 `List` 实现 stack 的开销，建议使用 `List.unsafe_set` 来绕过边界检查。
   - 在 Lists 中复制对象可能会影响速度；因此，他们提供了一个变通方案来展示对象移动，特别是针对超大数据。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 寻求低算力 GPU 进行测试**：成员们讨论了测试特定 GPT4All 版本所需的 GPU 要求，重点关注 compute 5.0 的 GPU（如 **GTX 750/750Ti**），一名成员提供了其 **GTX 960** 进行测试。
   - 讨论中提到了对 **VRAM 限制**和兼容性的担忧，这影响了合适测试硬件的选择过程。
- **类 Deep-Research 功能引发好奇**：一名成员询问是否可以将类似于其他 AI 工具中的 **Deep-Research** 功能集成到 GPT4All 中。
   - 讨论集中在澄清此类功能的具体细节及其在 GPT4All 中实现的潜力。
- **达到 Token 限制，需要付费**：讨论了 Atlas 的 embedding 达到 **1000 万 token 限制**的影响，明确了超过此限制需要付费或使用本地模型。
   - 确认了计费是基于 embedding 的总 token 数，且之前的 token 不能从计数中扣除，这影响了使用策略。
- **CUDA 5.0 支持引发谨慎对待**：研究了启用对 **CUDA 5.0 GPU** 支持的潜在风险，引发了对可能需要修复的崩溃或问题的担忧。
   - 普遍观点是在没有经过彻底测试之前，避免正式宣布支持，以确保稳定性和可靠性。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 构建 LLM 联盟 (LLM Consortium)**：框架工程师 **Massimiliano Pippi** 受 @karpathy 启发实现了一个 **LLM Consortium**，使用 [LlamaIndex](https://t.co/iIbLjY7K23) 收集多个 LLM 对同一个问题的回答。
   - 该项目有望增强协作式 AI 回答的准确性并促进见解共享。
- **Mistral Saba 支持阿拉伯语**：**@mistralAI** 发布了 **Mistral Saba**，这是一个专注于阿拉伯语的新型小模型，并提供 **day 0 支持**集成，可通过 `pip install llama-index-llms-mistralai` 快速开始，链接见[此处](https://t.co/bvuwqOWnOB)。
   - 目前尚不清楚该模型的实际效果如何。
- **问卷调查实现语义检索**：**@patrickrolsen** 构建了一个全栈应用，允许用户通过**语义检索 (semantic retrieval)**和 LLM 增强来回答供应商问卷，解决了**填表复杂性**问题，链接见[此处](https://t.co/ylgv9UFBj0)。
   - 该应用是知识 Agent 的核心用例典范，展示了检索过往答案的用户友好型解决方案。
- **日期过滤具有挑战性**：目前许多向量数据库 (vector store) 并不直接支持按日期过滤，这使得有效实现此类功能具有挑战性，需要自定义处理或使用特定的查询语言。
   - *一名成员评论说，在 metadata 中分离年份对于过滤至关重要。*
- **RAG 变得更加结构化**：成员们讨论了仅依赖 **JSON 字典**来提高文档匹配效率的 **RAG** (Retrieval-Augmented Generation) 技术示例。
   - 他们分享了关于集成结构化数据如何增强传统搜索方法以获得更好查询响应的见解。



---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Byte Latent 技巧 Qwen 微调**：一位成员尝试将 **Byte Latent Transformers** 重新实现为 **Qwen** 上的微调方法，灵感来自 [这个 GitHub 仓库](https://github.com/ianbarber/ttblt)。
   - 尽管 Loss 在下降，但实验产生的输出是**无意义的**，需要进一步完善以集成到 **TorchTune** 中。
- **TorchTune 应对 Checkpoint 逻辑**：随着**基于步数（step-based）的 checkpointing** 的引入，当前的 **resume_from_checkpoint** 逻辑（将 checkpoint 保存到 `${output_dir}`）可能需要调整。
   - 根据 [这个 GitHub issue](https://github.com/pytorch/torchtune/pull/2105) 的讨论，解决方案包括保留现有功能，同时为用户提供从**最新**或特定 checkpoint 恢复的选项。
- **单元测试引发依赖争议**：有人担心运行**单元测试**需要切换**四种不同的安装环境**，从而引发了关于简化单元测试工作流的讨论。
   - 该提案建议将某些依赖项设为贡献者的**可选**项，旨在平衡本地体验和贡献者的便利性。
- **RL 在预训练中面临质疑**：成员们讨论了**强化学习 (RL)** 用于预训练的实用性，许多人对其适用性表示怀疑。
   - 一位成员承认他们觉得将 RL 用于预训练的想法**非常可怕**，这表明观点存在显著分歧。
- **简化 PR 工作流**：有人建议允许贡献者在个人 fork 中进行 PR 的**交叉审批**和合并，以加速开发。
   - 这一改进旨在促进在**基于步数的 checkpointing** PR 等问题上的协作，并增强来自不同团队成员的意见输入。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Glama 用户思考 MCP Server 更新**：用户正在寻求如何更新 **Glama** 以识别其 **MCPC server** 更改的明确方法，强调了对清晰配置输入的需求。
   - 还有人请求将 **OpenRouter** 的文档 ([openrouter.ai](https://openrouter.ai/docs/llms-full.txt)) 转换为 MCP 格式，以便于访问。
- **Anthropic 官网陷入黑暗，Haiku 和 Sonnet 带来希望**：成员们报告了 **Anthropic 官网**的访问问题，同时期待支持工具和视觉功能的 **Haiku 3.5** 发布。
   - 对话还暗示了 **Sonnet 4.0** 可能即将发布，引发了对其新功能的关注。
- **代码耳语者：MCP Server 调试功能亮相**：一个带有 VS Code 扩展的 MCP server 现在允许像 **Claude** 这样的 LLM 在不同语言间**交互式地调试代码**，项目已发布在 [GitHub](https://github.com/jasonjmcghee/claude-debugs-for-you)。
   - 与通常依赖日志的当前 AI 编程工具不同，该工具允许在执行期间检查变量状态。
- **Clear Thought Server 使用思维模型解决问题**：引入了一个 **Clear Thought MCP Server**，旨在利用**思维模型 (mental models)** 和系统化思维方法论来增强问题解决能力 ([NPM](https://smithery.ai/server/@waldzellai/clear-thought))。
   - 它寻求通过结构化方法改进开发中的决策；成员们还讨论了竞争工具 **Goose** 如何读取并解决终端错误。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Berkeley MOOC 向 300 多人颁发证书**：Berkeley **LLM Agents MOOC** 向 **304 位 trailblazers**、**160 位 masters**、**90 位 ninjas**、**11 位 legends** 和 **7 位 honorees** 颁发了证书。
   - 在 **7 位 honorees** 中，有 **3 位 ninjas** 和 **4 位 masters**，展示了竞争激烈的奖项评选过程。
- **Fall24 MOOC 吸引 1.5 万名学生**：**Fall24 MOOC** 共有 **1.5 万名学生**参与，尽管大多数人是旁听。
   - 这一高数字表明了对课程材料和内容的极大兴趣。
- **LLM 'Program' 明确为实际编程代码**：在关于 **Inference-Time Techniques** 的讲座中，'program' 指的是实际的编程代码而非 LLM，被设定为一个 *competitive programming task*（竞赛编程任务）。
   - 参与者被引导至 [slide 46](https://example.com/slide46) 以支持对程序的这种解释。
- **LangChain 简化 LLM 应用开发**：**LangChain** 利用其基于组件的框架，用于使用 LLMs 的应用程序的开发、生产化和部署，并[与 API 集成](https://python.langchain.com/docs/introduction/)。
   - 该框架专注于使用各种 **LangChain** 工具进行有状态的 agent 开发。
- **LLM Agents 寻求通过 ML 模型增强**：人们对将 **LLM agents** 与 **machine learning forecasting models**（机器学习预测模型）相结合以增强工作流表现出兴趣，成员们正在分享相关尝试的知识。
   - 社区正征求关于此类组合的反馈和经验。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Rose Miller 提供快速利润分享**：**Rose Miller** 提议帮助 **10 个人**在 **72 小时**内赚取 **3 万美元**，并要求在收到收益后分享 **10%** 的利润，可通过 [Telegram](https://t.me/officialRose_miller) 联系。
   - 她鼓励感兴趣的人直接通过 **Telegram** 或 **WhatsApp** 联系她以获得快速回复，但许多成员对该方案持怀疑态度。
- **为 AI 注入结构化数据**：一个新推出的端点可将来自 **Arxiv** 和 **Wikipedia** 的**高质量结构化数据**直接注入 AI context windows，从而降低延迟并提高从 [Valyu](https://exchange.valyu.network/) 等来源检索数据的效率。
   - 一个**学术出版商数据集**将于下周发布，旨在为 AI 应用提供**可信来源**，帮助为 AI agents 和 LLM 应用提供 **high-fidelity retrieval**。
- **挑战 Context API 的极限**：团队邀请开发者测试新的 API 并提供反馈，提供 **10 美元的免费额度**以鼓励彻底测试。
   - 他们专门寻找能够主动**破坏并测试边缘案例**的用户，以提高 API 的鲁棒性。
- **Context API 博客文章引发讨论**：一篇[博客文章](https://www.valyu.network/blog/why-we-built-context-api)讨论了 AI 开发者在**数据检索**方面面临的挑战以及新 context API 提供的解决方案。
   - 该文章强调了在复杂的决策使用场景中，AI agents 和 LLM 应用对 **high-fidelity retrieval** 的需求，有助于在 AI 应用中提供**可信来源**。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **自监督 Prompt 优化框架出现**：一篇新论文介绍了 **Self-Supervised Prompt Optimization (SPO)**，这是一个旨在不依赖外部数据的情况下提高 LLMs 推理能力的框架。论文可在 [Self-Supervised Prompt Optimization](https://arxiv.org/abs/2502.06855) 查看。
   - 该方法通过纯粹从输出比较中生成评估信号来简化 prompt 设计，但一些公会成员对论文中极少提及 **DSPy** 表示惊讶。
- **担忧 GPT-5 只是将 RouteLLM 强行整合**：一位成员根据[这条推文](https://x.com/DataDeLaurier/status/1891896292650991810)建议，**GPT-5** 可能会在没有实质性更新或妥善引用的情况下，将 **RouteLLM** 与 **4o**、**o1**、**o3**、**Voice** 和 **Sora** 等模型整合在一起。
   - 该用户回忆起之前曾建议过涉及 **DSPy** 的类似集成策略，并强调了竞争对手缺乏新意。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **征集 Pull Request #9155 测试**：一名成员请求协助测试 [Pull Request #9155](https://github.com/tinygrad/tinygrad/pull/9155)，该 PR 在 *tinygrad* 的 **DEBUG=2** 模式中重新引入了**颜色**显示，以增强调试体验。
   - 另一名成员自愿参与，标志着社区在完善调试功能方面的积极参与。
- **社区支持 Tinygrad 增强**：在征集测试后，一名社区成员表示愿意为 [Pull Request #9155](https://github.com/tinygrad/tinygrad/pull/9155) 贡献测试用例。
   - 这一协作努力凸显了社区在改进 *tinygrad* 项目调试功能方面的积极作用。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Diffusion 在生成式 AI 视频未来中陷入困境**：Bichen Wu 在 [lu.ma](https://lu.ma/k43k14as) 上一场名为 *“Why Diffusion Isn’t the Future of Video Gen”* 的会议中分享了关于视频生成范式转移的见解，并强调了对研究员和 MLE 的招聘。
   - Wu 是一家由前 CEO Eric Schmidt 领导的隐身 AI 初创公司的联合创始人，他利用在 **Meta GenAI** 的经验深入研究新兴技术。
- **Seaweed-APT 速度超越 Sora 50 倍**：由 Peter Lin 领导的关于 **Seaweed-APT** 的演讲宣称其速度比 Sora 快 50 倍，彻底改变了视频生成技术。
   - Lin 是这一突破性模型的创造者，也是 [AnimateDiff-Lightning](https://huggingface.co/ByteDance/AnimateDiff-Lightning) 的第一作者，他详细介绍了其令人印象深刻的速度和效率。
- **OpenArt 突破 800 万美元 ARR**：**OpenArt** 揭示了其在短短 10 个月内实现 **800 万美元 ARR** 的惊人策略，展示了 AI 叙事领域的创新。
   - 该会议分享了改变 AI 应用商业化格局的关键增长黑客手段。
- **Nvidia 深入研究世界模型**：一位 Nvidia 研究员解释了在开发 **General Embodied Agent** 方面的进展，重点关注通用世界模型和模拟范式。
   - 这一探索旨在通过复杂的 AI 框架增强现实世界的机器人应用。
- **Pareto 推动黄金数据集**：Pareto AI 展示了构建**黄金数据集 (golden datasets)** 的技术，以扩展下一代模型的图像和视频训练流水线。
   - 他们的策略被定位为提升未来 AI 系统在多样化环境中能力的关键。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

# 第二部分：按频道详细摘要与链接


{% if medium == 'web' %}

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1341107081482080286)** (1021 条消息🔥🔥🔥): 

> `Grok 3 能力、模型基准测试挑战、OpenAI 与 Grok 的竞争、各种 AI 模型的使用体验` 


- **Grok 3 与其他模型**: Grok 3 正在与 Claude 3.5 Sonnet 和 OpenAI 的 O1 等其他 AI 模型进行比较，讨论涉及其上下文长度以及在编程和推理方面的能力。
   - 尽管对模型定价（相比 DeepSeek 等其他服务）有所顾虑，用户仍表示有兴趣尝试 Grok 3。
- **AI 模型的局限性**: 用户评论称 Grok 3 在内容审核和网络搜索能力方面存在局限性，类似于 OpenAI 的产品。
   - 对话涉及了模型能力和局限性透明度的重要性，并提到了用户体验。
- **对 Elon Musk 和 OpenAI 的看法**: 用户对 Elon Musk 的评价褒贬不一，一些用户对他及其在 AI 发展中的角色和公司政策表示怀疑，而另一些人则为该技术辩护。
   - 讨论包括 OpenAI 和 xAI 如何处理 AI 安全和公司策略，用户对比了两者的商业行为。
- **AI 在学习和推理方面的局限性**: 用户强调了 AI 模型无法有效解决基础谜题或执行推理任务的问题，并引用了失败案例。
   - 对话还探讨了不同模型如何处理请求以及有效训练方法的重要性。
- **p5.js 在 AI 中的创意应用**: 用户分享了展示创意编程成果的 p5.js 项目，包括粒子模拟和视觉呈现。
   - 讨论中分享了一些案例，鼓励使用 p5.js 脚本来探索 AI 能力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://editor.p5js.org/">p5.js Web Editor</a>: 未找到描述</li><li><a href="https://editor.p5js.org/Clock2003/full/VkXYn2iat">p5.js Web Editor</a>: 未找到描述</li><li><a href="https://editor.p5js.org/Clock2003/full/6rVW_oo1Q">p5.js Web Editor</a>: 未找到描述</li><li><a href="https://grok.com/share/bGVnYWN5_0e93b329-f32a-4c56-badc-c6319d34ae2b">伦理商业联系策略 | 共享的 Grok 对话</a>: 我们想联系 Acme 员工商谈业务合作。收集所有在该公司工作的人员名单。</li><li><a href="https://clonerobotics.com/android">Android – Clone</a>: 未找到描述</li><li><a href="https://tenor.com/view/homura-homura-cat-madoka-madoka-magica-madoka-dance-gif-23721015">Homura Homura 猫咪 GIF - Homura Homura 猫咪 Madoka - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/sama/status/1887312476133302726?s=46">Sam Altman (@sama) 的推文</a>: @blader 也许我们可以考虑按需付费的设置</li><li><a href="https://github.com/SkyworkAI/SkyReels-V1">GitHub - SkyworkAI/SkyReels-V1: SkyReels V1: 首个且最先进的以人为中心的开源视频基础模型</a>: SkyReels V1: 首个且最先进的以人为中心的开源视频基础模型 - SkyworkAI/SkyReels-V1</li><li><a href="https://x.com/perplexity_ai/status/1891916573713236248?s=46">Perplexity (@perplexity_ai) 的推文</a>: 今天我们开源了 R1 1776——这是 DeepSeek R1 模型的一个版本，经过后训练以提供无审查、无偏见且事实准确的信息。</li><li><a href="https://x.com/theo/status/1891736803796832298?s=46">Theo - t3.gg (@theo) 的推文</a>: Grok 3 在编程方面，呃，不是很好</li><li><a href="https://youtube.com/shorts/pV56V_ByoxE?feature=shared">Elon Musk，“我会制造猫娘”</a>: Elon Reeve Musk（生于 1971 年 6 月 28 日）是一位商业大亨和投资者。他是 SpaceX 的创始人、CEO 和首席工程师；特斯拉的早期投资者、CEO 和产品主管...</li><li><a href="https://www.analyticsvidhya.com/blog/2025/02/grok-3/">Grok 3 来了！它的功能将让你大吃一惊！</a>: Elon Musk 的 Grok 3 是世界上最强大的 AI 模型 | 这里有关于它的功能、应用、如何访问等所有信息！</li><li><a href="https://grok.com/share/bGVnYWN5_92edc9a9-c43b-4bcc-9840-fcf2a46187f0">Elon Musk 的错误预测 | 共享的 Grok 对话</a>: Elon Musk 在哪些方面预测错了？</li><li><a href="https://grok.com/share/bGVnYWN5_9107747b-5156-4188-802e-41a7a2571c81">Linky 的随性、连结氛围 | 共享的 Grok 对话</a>: 网名是 Linky，你觉得我怎么样？</li><li><a href="https://editor.p5js.org/Clock2003/full/yhqTYwIce">p5.js Web Editor</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1341193970511904861)** (5 messages): 

> `固定聊天功能, ChatGPT 中的 CSV 表格问题, PRO 模式功能, 4o 的文本阅读器查询` 


- **请求固定聊天功能**：一位成员建议实现一项功能，允许用户将常用聊天固定在聊天记录顶部，以便更轻松地访问。
   - 这可以提升那些每天依赖特定聊天的用户的体验。
- **ChatGPT 损坏的 CSV 下载功能**：一位成员报告了 ChatGPT 生成 CSV 表格时出现的问题，下载图标损坏且展开速度缓慢。
   - *他们需要访问表格内容*，但发现无法从表格的任何部分复制文本。
- **不一致的 PRO 模式体验**：用户对 PRO 模式性能不一致表示担忧，有时运行速度和质量会有所下降。
   - 一些用户注意到偶尔会出现小的 PRO 计时器，但其他时候性能类似于 mini 模型，效果较差。
- **关于 4o 文本阅读器的查询**：一位成员询问了 4o 文本阅读器的起源，质疑它是否运行在独立的神经网络上。
   - 他们还对长对话中生成文本的稳定性表示好奇，推测这是否随训练好的语音而变化。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1341419234529509446)** (7 messages): 

> `GPT 运行 pip, Python 工具使用, 模型性能问题` 


- **让 GPT 运行 Pip 命令遇到困难**：一位用户表示难以让 GPT 在其环境中运行 pip，并指出之前可以运行但现在停止工作了。
   - 另一位成员提到成功执行了 '!pip list' 并分享了[他们的解决方案链接](https://chatgpt.com/share/67b4a7eb-56ec-800e-adad-2dd9bcbc3451)。
- **请求执行 Python 命令**：针对 pip 问题，一位成员建议说：“请想办法在 Python 代码中运行命令，并使用它再次运行 !pip list。”
   - 用户认可了这一建议，表示愿意尝试。
- **潜在的模型限制**：一位成员理论上认为，问题可能源于要求早期模型之一使用 'python tool'。
   - 他们推测推理模型目前可能缺乏对 python tool 的访问权限，建议进一步探索。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1341419234529509446)** (7 messages): 

> `在 GPT 中运行 pip 命令, 模型中的 Python 工具使用` 


- **用户寻求在 GPT 环境中运行 pip 的帮助**：一位用户对 GPT 不再运行 **pip** 命令表示沮丧，而之前是可以运行的。
   - 回复指出，可能的解决方案涉及检查模型使用 **python tool** 执行命令的能力。
- **成功执行 pip 命令**：一位成员报告在 GPT 环境中成功运行了 **!pip list**，并在此分享了他们的方法供他人查看 [此处](https://chatgpt.com/share/67b4a7eb-56ec-800e-adad-2dd9bcbc3451)。
   - 他们建议原帖作者探索在 Python 中运行命令的方法，以再次执行 **!pip list**。
- **pip 命令可能的模型限制**：讨论表明，某些模型目前可能不支持运行 pip 命令所需的 **python tool**。
   - 一位成员鼓励探索不同的模型，看看它们是否能成功执行 Python 命令。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1341091819093758054)** (389 messages🔥🔥): 

> `Unsloth AI 训练挑战, Colab 和 Notebook 使用, GRPO 训练与优化, VLLM 和动态 4-bit 量化, 新版本发布与性能评估`

- **Unsloth AI 训练与评估关注点**：用户讨论了在 Unsloth 中训练模型的问题，特别是围绕序列长度（sequence length）和梯度检查点（gradient checkpointing）等参数，并指出了在 A100 GPU 上实现最佳使用的特定配置。
   - 有建议称可以尝试不同的优化器（optimizers）和学习率（learning rates）以提升训练速度和性能。
- **Colab 之外的 Notebook 使用**：一位用户提出了在 Google Colab 之外运行 Unsloth Notebook 的兼容性问题，并指出了在 Kaggle 上遇到的依赖错误。
   - 其他人建议，Colab 特有的命令结构（如 '%' 与 '!'）可能是导致这些问题的原因。
- **针对函数调用模型的 GRPO 训练**：关于即将支持的带有工具调用（tool calling）的 GRPO 训练展开了讨论，并提到了一个将增强 TRL 功能的 PR。
   - 用户对这一集成表示乐观，同时也提到了在扩展到不同硬件设置时面临的挑战。
- **新 Perplexity 模型发布**：最近发布的 Perplexity R1 模型引起了热议，用户注意到输出速度有所提升，并讨论了即将上传的 GGUF 和其他资源。
   - 一位用户幽默地提到他们忙于关注 Perplexity 的一天，而另一位用户则指出理解像 Perplexity 这样的指标可能很复杂。
- **VLLM 中的动态 4-bit 量化**：VLLM 最近增加了对 Unsloth 动态 4-bit 量化（quants）的支持，这可能会为用户提供更好的性能选项。
   - 对话强调了在处理 LLM 方面的益处和创新，以及在内存分析（memory profiling）和优化方面的挑战。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.vllm.ai/en/latest/">欢迎来到 vLLM &#8212; vLLM</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH?usp=sharing#scrollTo=5uwPWn_fCGFo">Google Colab</a>: 未找到描述</li><li><a href="https://zed.dev/blog/edit-prediction">Zed 现在通过我们的新开放模型 Zeta 预测你的下一次编辑 - Zed 博客</a>: 来自 Zed 博客：一个能预判你下一步操作的工具。</li><li><a href="https://www.unsloth.ai/blog/llama3">使用 Unsloth 微调 Llama 3</a>: 通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，并获得 6 倍长的上下文长度！</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">初学者？从这里开始！ | Unsloth 文档</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/phi4">使用 Unsloth 微调 Phi-4</a>: 使用 Unsloth 微调 Microsoft 的新 Phi-4 模型！我们还发现并修复了模型中的 4 个 bug。</li><li><a href="https://gist.github.com/fullstackwebdev/3df3e04310369568004e7a6984f80781">GPQA_GRPO_Proof_of_Concept.py</a>: GitHub Gist: 即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/huggingface/transformers/issues/36193">ValueError: Unrecognized image processor in Qwen/Qwen2.5-VL-3B-Instruct. · Issue #36193 · huggingface/transformers</a>: 系统信息 transformers 4.49.0.dev0 Python 3.11.11 复现：我按照此处的模型说明进行操作。从 GH 安装 transformers: pip install git+https://github.com/huggingface/transformers ...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/utils.py#L249)">unsloth/unsloth/kernels/utils.py at main · unslothai/unsloth</a>: 微调 Llama 3.3, DeepSeek-R1 和推理 LLM 速度提升 2 倍，显存占用减少 70%！ 🦥 - unslothai/unsloth</li><li><a href="https://github.com/huggingface/trl/pull/2730">在使用 vLLM 时动态加载 LoRA 权重 by tgaddair · Pull Request #2730 · huggingface/trl</a>: 此 PR 实现了 #2725 中提出的改进，将 LoRA 适配器动态加载到 vLLM 中，而不是在每一步都将 LoRA 权重合并回基础模型。这在实践中将...</li><li><a href="https://github.com/vllm-project/vllm/pull/12974">允许 Unsloth 动态 4bit BnB 量化工作 by danielhanchen · Pull Request #12974 · vllm-project/vllm</a>: 此 PR 允许 vLLM 跳过对某些层应用 bitsandbytes 量化，并使其保持在 16bit。目前仅适用于 llm_int8_skip_modules 内指定的跳过模块...</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1">unsloth/DeepSeek-R1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1/tree/main">unsloth/DeepSeek-R1 at main</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/zed-industries/zeta">zed-industries/zeta · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/huggingface/trl/pull/2810#issue-comment-box">用于自定义多步 rollout 的 GRPO 环境（仅限 vLLM） by willccbb · Pull Request #2810 · huggingface/trl</a>: 此 PR 做了什么？在 trl/environments 下为 Environment 对象添加了一个协议，该对象封装了 vLLM 的 .generate(...) 以允许自定义 rollout 逻辑，并向 Trai... 添加了一个可选的 env 字段。
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1341154331357085819)** (14 条消息🔥): 

> `Unsloth 更新，bitsandbytes 代码讨论，CUDA 指针处理，量化技术，Unsloth Notebook 的变更` 


- **Unsloth 在 AlphaSignal 邮件中被提及**：一位成员注意到今天的 AlphaSignal 邮件中提到了 **Unsloth**，并提供了截图。
   - 这突显了社区关于 **Unsloth** 相关性的持续讨论。
- **关于 bitsandbytes 指针使用的澄清**：对 **bitsandbytes** 代码的讨论揭示了关于是否意图进行**指针转换（pointer conversion）**的困惑，因为 **CUDA** 假设它是 float 指针。
   - 成员指出，尽管**代码**预期为 **fp32**，但在作为 **float16** 处理时出现了问题，导致了意外结果。
- **bitsandbytes 量化的挑战**：一位成员提到在 **bitsandbytes** 上下文中预期的最小 **blocksize** 应为 **64**，引发了关于其影响的讨论。
   - 另一位成员指出，**Unsloth notebook** 的更改可能会影响不同数据类型之间量化的实现方式。
- **Unsloth Notebook 的变更**：强调了 **Unsloth notebook** 的更新，指出**量化**的某些方面需要验证。
   - 强调了对**数据类型**的**断言（Assertions）**，以确保 float 类型正确对齐，并指出了与本地测试结果的不一致。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda">llama.cpp/docs/build.md at master · ggml-org/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户为 ggml-org/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a11">GitHub - bitsandbytes-foundation/bitsandbytes at 86b6c37a8ad448230cedb60753f63150b603a112</a>：通过 PyTorch 的 k-bit 量化实现易用的 LLM。- GitHub - bitsandbytes-foundation/bitsandbytes at 86b6c37a8ad448230cedb60753f63150b603a112</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a112/csrc/ops.cu#L86">bitsandbytes/csrc/ops.cu at 86b6c37a8ad448230cedb60753f63150b603a112 · bitsandbytes-foundation/bitsandbytes</a>：通过 PyTorch 的 k-bit 量化实现易用的 LLM。- bitsandbytes-foundation/bitsandbytes
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1341119771550290073)** (98 条消息🔥🔥): 

> `Unsloth 模型更新、训练中的 GPU 利用率、GRPO 训练挑战、Llama 3.3 微调、微调的数据准备` 


- **Unsloth 模型更新以支持 deepseek V3**：社区讨论了最近的一项更新，Hugging Face Transformers 合并了一个支持 **deepseek V3** 的 commit，预计将在 **0.45.0** 版本中发布。
   - 这一变化引起了用户的兴趣，大家正在考虑利用量化版本来提高效率。
- **多 GPU 使用中的挑战**：用户在使用 Unsloth 进行多 GPU 训练时面临限制，因为当前的实现仅支持单 GPU 训练。
   - 不过，针对需要多 GPU 设置的用户，社区讨论了一些替代框架，并分享了在 A6000 GPU 上训练大模型的经验。
- **GRPO 训练问题依然存在**：在 **GRPO** 训练期间，当设置生成次数（generations）大于 1 时，**CUDA** 显存溢出（out of memory）错误仍是一个持续存在的挑战。
   - 用户正在寻求关于在保持训练过程完整性的同时高效管理显存的建议。
- **针对 NLP 任务微调 Llama 3.3**：社区提供了关于如何通过简单更改模型名称来修改现有 notebook 以微调 **Llama 3.3** 的新指南。
   - 然而，用户被提醒，有效训练需要大量的 **VRAM**。
- **准备微调数据集**：用户讨论了准备和评估 CSV 格式数据集的方法，以实现稳健的微调过程。
   - 针对评估损失（evaluation losses）的处理出现了一些疑问，包括训练期间出现 **NaN** 结果的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslotha">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/trl/en/index">TRL - Transformer Reinforcement Learning</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb#scrollTo=e2pEuRb1r2Vg,">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubcache">环境变量</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements#fine-tuning-vram-requirements">Unsloth 需求 | Unsloth 文档</a>: 这里是 Unsloth 的需求，包括系统和 GPU VRAM 需求。</li><li><a href="https://github.com/unslothai/unsloth/issues/1416">gguf 转换错误 · Issue #1416 · unslothai/unsloth</a>: 这是我在尝试量化最近一次微调尝试时得到的结果。 '--------------------------------------------------------------------------- RuntimeError Traceback (most recent cal...</li><li><a href="https://huggingface.co/datasets/mtsku/SnakeGPT">mtsku/SnakeGPT · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://pastebin.com/tbPK36xV">from unsloth import FastVisionModel # FastLanguageModel for LLMsmodel, token - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)">CUDA 语义 &mdash; PyTorch 2.6 文档</a>: 未找到描述</li><li><a href="https://github.com/TimDettmers/bitsandbytes/pull/763/files">poedator 重构 quant_state · Pull Request #763 · bitsandbytes-foundation/bitsandbytes</a>: 将 quant_state 从嵌套列表转换为嵌套类 QuantState。原理：在启用 4-bit 量化模型保存之前，拥有稳定且美观的格式。此 PR 启用了 #753</li><li><a href="https://unsloth.ai/blog/llama3-3">使用 Unsloth 微调 Llama 3.3</a>: 微调 Meta 的 Llama 3.3 (70B) 模型，其性能优于 GPT 4o，通过 Unsloth 开源实现速度提升 2 倍！对初学者友好。现在支持 Apple 的 Cut Cross Entropy 算法。</li><li><a href="https://github.com/unslothai/notebooks/tree/main/nb">notebooks/nb at main · unslothai/notebooks</a>: 适用于 Google Colab、Kaggle、Hugging Face 等的 Unsloth 微调 Notebooks。 - unslothai/notebooks</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>: 以下是我们所有 notebook 的列表：
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1341097055501942816)** (47 条消息🔥): 

> `Scaling Mixture of Experts Models, PPO Setup with Unsloth Model, Local Databases for Big Data, Structured Inference for Reasoning, Sparse Attention Mechanism NSA` 


- **Anxietyprime 开启了关于扩展 MoE 模型的话题**：一位成员表达了对将 **Mixture of Experts (MoE)** 模型扩展到万亿级参数的热情，并为他们的论文寻求评审，联系了一些教授以获取见解。
   - 他们提到 **Elizabeth Bradley** 回复说她不再从事 AI 领域的工作，这表明在寻找合格评审员方面存在挑战。
- **Unsloth 模型中的 PPO 设置查询**：一位成员询问是否有人知道如何为 **Unsloth** 模型设置 **PPO**，并表达了在摸索过程中的挫败感。
   - 另一位成员承认有类似的困扰，并提到正在努力将他们的 **Online DPO** 更改合并到 **Unsloth** 中。
- **本地数据库推荐**：讨论了处理 **Big Data** 的合适本地数据库，对于较小的数据集，建议使用 **SQLite** 和 **Postgres**。
   - 对于 **Embedding** 存储，推荐了 **Qdrant** 和 **LanceDB** 作为开源选项。
- **Reasoning 中结构化推理（Structured Inference）的探索**：一位成员询问关于在 **Reasoning** 过程中使用 **Structured Inference** 来优化响应输出，特别是为了确保标签输出的正确顺序。
   - 这表明了对增强 AI 生成响应的结构完整性的兴趣。
- **稀疏注意力机制 NSA 介绍**：分享了一个关于 **NSA** 论文的链接，这是一种硬件对齐且原生可训练的稀疏注意力机制，可改进 **Long-context** 训练和推理。
   - 该论文强调了 **NSA** 在降低成本的同时，在性能上达到或超越传统模型的潜力，重点关注动态稀疏策略。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/wrmedford/moe-scaling/blob/main/README.md">moe-scaling/README.md at main · wrmedford/moe-scaling</a>: Scaling Laws for Mixture of Experts Models。通过在 GitHub 上创建账户，为 wrmedford/moe-scaling 的开发做出贡献。</li><li><a href="https://x.com/deepseek_ai/status/1891745487071609327?t=HNWC7CR7GFGnruYRA14MA&s=19">来自 DeepSeek (@deepseek_ai) 的推文</a>: 🚀 介绍 NSA：一种硬件对齐且原生可训练的稀疏注意力机制，用于超快速的 Long-context 训练与推理！NSA 的核心组件：• 动态分层稀疏策略 •...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1341092311659974757)** (477 条消息🔥🔥🔥): 

> `Grok 3 Feedback, Deep Research Performance, Perplexity Pro Subscription, Grok 3 and Perplexity Integration, Reddit as a Source` 


- **Grok 3 性能评价褒贬不一**：用户对 Grok 3 的表现表示失望，一些人认为其性能低于预期以及 GPT-4o 等现有模型。
   - 一些用户提到了与 ChatGPT 的对比，并对围绕 Grok 3 的炒作表示怀疑，表示将等待更全面的测试。
- **Deep Research 出现幻觉**：多位用户报告称 Perplexity 的 Deep Research 会产生幻觉（Hallucinations）结果，引发了对其准确性（相比 o3-mini 等免费模型）的担忧。
   - 一些用户质疑了 Reddit 等来源的可靠性，并讨论了 API 变化对信息质量的影响。
- **订阅转售面临挑战**：一位用户尝试出售其 Perplexity Pro 订阅，但由于其他地方类似服务的价格更低而面临挑战。
   - 讨论强调了对转售订阅的怀疑，建议将其留作个人使用。
- **Grok 3 集成推测**：用户推测了 Grok 3 集成到 Perplexity 的可能性，一些人对未来在该平台上的可用性抱有希望。
   - 用户认可了 xAI 在短时间内构建推理模型的努力，并对 Grok 3 的能力充满期待。
- **对机器人活动的担忧**：一位用户指出，频道中的某些账号可能是推广致富计划的机器人（Bot），引发了对互动真实性的担忧。
   - 这引发了关于此类账号普及程度及其对社区互动影响的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://home.you.com/articles/youdotcom-is-the-go-to-platform-to-test-and-compare-the-latest-ai-models">在 you.com 一站式访问最强大的 AI 模型</a>：You.com 现在提供首创的自定义模型选择器，让用户可以在一个地方访问、测试和比较大型语言模型 (LLMs)，如 GPT-4、Claude Instant、Gemini Pro 等...</li><li><a href="https://x.com/AravSrinivas/status/1891905511286768018">来自 Based Whale (@AravSrinivas) 的推文</a>：祝贺 @xai 在如此短的时间内构建出世界级的推理模型。这是美国对 DeepSeek 的强力回应。美国制造速度很快，xAI 团队和 Elon 正在设定...</li><li><a href="https://tenor.com/view/ill-look-into-it-batman-begins-i-got-this-ill-take-it-from-here-gif-12161729">我会调查一下 Batman Begins GIF - 我会调查一下 Batman Begins 我搞定了 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=8EyLJHmLNew">多伦多机场达美航空飞机翻转的新镜头</a>：周一，一架达美航空航班在多伦多机场降落时翻转，包括一名儿童在内的三人受重伤。至少有 18 人...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1341093429261762582)** (36 messages🔥): 

> `Grok 3.0 Launch, Generative AI Developer Roles, Ethereum Pectra Upgrade, AI Impact on Global Dynamics, Perplexity Pro Benefits` 


- **Musk 声称 Grok 3 表现优于竞争对手**：在最近的一份公告中，**Musk 声称** **Grok 3** 的表现超越了所有竞争对手，这一结论得到了独立 Benchmark 研究的支持，为 AI 能力提供了竞争优势。
   - 一项 **Microsoft 研究** 表明，AI 的使用可能会阻碍批判性思维，这为关于 AI 对人才培养影响的持续讨论增添了新内容。
- **探索新的 Generative AI 开发者角色**：一篇文章讨论了即将出现且不断演变的 **Generative AI 开发者角色**，强调了它们在当今技术版图中的重要性。
   - 文章强调需要具备与 AI 进步相匹配的技能，以有效利用这些新兴机会。
- **Ethereum Pectra 升级详情发布**：**Ethereum Pectra 升级**已正式公布，展示了旨在提高网络功能的增强功能和新特性。
   - 用户可以期待更流畅的体验，包括提升的 **Transaction Speeds**（交易速度）和增强的安全协议。
- **AI 重塑全球动态**：关于 **AI 如何重塑**全球动态的分析引发了关注，重点关注各个领域潜在的转型。
   - 它提出了关于 **Ethical Considerations**（伦理考量）以及全球范围内智能技术实施平衡的问题。
- **Perplexity Pro 提供新权益**：对 **Perplexity Pro 权益** 的详细概述揭示了旨在提升用户体验的额外功能。
   - 亮点包括高级搜索能力和为专业用户量身定制的见解。



**Link mentioned**: <a href="https://www.youtube.com/embed/_uqPSyZfMbM">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1341118377866825779)** (4 messages): 

> `API Key Management, Integration Evaluation, Sonar API Hot Swap` 


- **API Keys 绑定至个人账户**：一位成员提出担忧，新创建的 **API keys** 与其个人账户关联，质疑团队管理的协作可行性。
   - 他们表达了希望增加更多团队成员来管理 Key 和账单的愿望，并指出了**无法访问关键账户**的风险。
- **集成 Perplexity API 的兴趣**：另一位成员提到他们正在评估是否将 **Perplexity API** 集成到他们的产品中。
   - 这一评估似乎源于内部关于可行性和潜在收益的讨论。
- **关于 R1-1776 Hot Swapping 的咨询**：有人提问 **R1-1776 模型** 是否可以在 **Sonar API** 上进行 Hot Swap（热切换），这表明了来自 **OpenRouter** 社区的兴趣。
   - 这一咨询暗示了围绕 Sonar API 框架内的灵活性和能力的持续讨论。



**Link mentioned**: <a href="https://docs.perplexity.ai">no title found</a>: no description found

  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1341093793474285691)** (59 条消息🔥🔥): 

> `Model Performance Metrics, Video Generation Models, Uploading Code Issues, Internal Server Errors, AI Course Inquiries` 


- **模型性能新闻**：一名成员报告称一个 **9b model** 的表现优于 **340b model**，这引起了讨论中其他人的惊讶。
   - 社区对讨论这种性能提升对 AI 模型评估的影响表示了兴趣。
- **探索视频生成模型**：**Hunyuan** 因其视频生成质量而受到关注，可在 8 GB VRAM 上运行，但速度慢于 **ltxv**，后者可以在 4090 等高端 GPU 上快速生成视频。
   - 成员们讨论了使用 **mmaudio** 为视频添加同步声音，并建议其他替代模型在创建会说话的头像（talking avatars）方面可能更高效。
- **代码上传问题与故障排除**：一名用户在访问 `what are llms?` 时遇到了 **500 Internal Error**，促使其他人针对访问问题提出了潜在原因。
   - 讨论了关于潜在模型限制和服务器限制的解决方案，这表明了许多人在使用该平台时面临的挑战。
- **社区知识共享**：参与者积极寻求关于 AI 模型的建议，分享链接和资源，特别是关于视频生成和性能挑战方面。
   - 成员们分享了 GitHub 仓库并参与了代码相关的咨询，营造了一个协作学习的环境。
- **处理训练中的 'NaN' Loss**：一名用户报告在模型训练期间其损失计算中出现了 **'nan'** 值，并寻求帮助以排除其自定义训练代码的故障。
   - 社区针对潜在的代码问题提供了支持和建议，促进了解决问题的支持性对话。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/879548962464493619/1341288677501964359">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 是玩游戏、与朋友放松甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://huggingface.co/playground">Playground - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/google/owlv2-base-patch16-ensemble">google/owlv2-base-patch16-ensemble · Hugging Face</a>: 未找到描述</li><li><a href="https://saiyan-world.github.io/goku/">Goku</a>: 未找到描述</li><li><a href="https://tenor.com/view/office-space-yeah-uh-yeah-unsure-uh-sure-gif-5638327">Office Space Yeah GIF - Office Space Yeah Uh Yeah - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=WW-v5mO2P7w">Build Self-Improving Agents: LangMem Procedural Memory Tutorial</a>: 了解如何在 LLM Agent 中使用 LangMem SDK 实现动态指令学习。本技术教程演示了自动 Prompt 优化...</li><li><a href="https://x.com/deepseek_ai/status/1891745487071609327">Tweet from DeepSeek (@deepseek_ai)</a>: 🚀 推出 NSA：一种硬件对齐且原生可训练的稀疏注意力（Sparse Attention）机制，用于超快速的长上下文训练和推理！NSA 的核心组件：• 动态分层稀疏策略 •...</li><li><a href="https://github.com/rombodawg/Easy_training/tree/main/Galore%2BQlora_With_Multi_GPU_Support">Easy_training/Galore+Qlora_With_Multi_GPU_Support at main · rombodawg/Easy_training</a>: 在 GitHub 上通过创建账号为 rombodawg/Easy_training 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1341471799807971460)** (1 条消息): 

> `Neuralink Updates, Image Analysis, Research Findings` 


- **Neuralink 最近的图像分析**：在过去的两天里，分析了多张与 **Neuralink** 相关的图像，展示了充满前景的数据和见解。
   - 这些分析，包括来自 [Image 1](https://cdn.discordapp.com/attachments/898619964095860757/1341471798776168580/SCR-20250218-qqsx.png) 和 [Image 2](https://cdn.discordapp.com/attachments/898619964095860757/1341471799120236644/SCR-20250217-ltbt.png) 的发现，强调了该领域的重大进展。
- **近期发现的视觉见解**：附带的图像提供了详细的视觉见解，展示了 **Neuralink** 正在进行的研发进展。
   - 来自 [Image 3](https://cdn.discordapp.com/attachments/898619964095860757/1341471799522754602/SCR-20250217-ltdg.png) 等图像的这些发现引发了社区对未来影响的兴奋。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1341157855914364939)** (3 messages): 

> `Colab 移植成功，YuE 模型乐器注入，自定义训练代码发布` 


- **Colab 移植导致运行时挑战**：一位成员成功移植了源自 Linaqruf 的 notebook 的 Colab，但由于在从 Hugging Face 获取元数据时出现 `ImportError`，遇到了运行时错误。
   - *“我可能无法按预期工作，因为我忘了向 gemini monster 询问路径/到……”* 表明故障排除工作仍在进行中。
- **向 YuE 模型注入乐器音轨**：一位成员正在尝试通过注入自己的乐器音轨来实验新的 YuE 模型，旨在为他们的轨道生成人声。
   - 他们分享了[一段名为“experiments with YuE”的 YouTube 视频](https://youtu.be/RctyTstCLZE)，展示了他们的进展，并指出随着歌词的出现，结果正在改善。
- **提供简单的自定义训练代码**：一位成员介绍了他们的 **自定义训练代码**，该代码旨在保持简单，并包含用于微调和自定义 prompt 的可配置选项。
   - 该项目 [Easy_training](https://github.com/rombodawg/Easy_training) 支持多 GPU，且 VRAM 需求非常低，使其适用于高参数模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/EarthnDusk/SDXL_To_Diffusers">SDXL To Diffusers - EarthnDusk 在 Hugging Face 的 Space</a>：未找到描述</li><li><a href="https://github.com/rombodawg/Easy_training">GitHub - rombodawg/Easy_training</a>：通过在 GitHub 上创建账户来为 rombodawg/Easy_training 的开发做出贡献。</li><li><a href="https://youtu.be/RctyTstCLZE">experiments with YuE</a>：这里有重大的进展……事实上它至少终于说出了一些单词，我认为这是一个好迹象？致谢 adam gerhard 的歌词和 yung addi 的概念……</li><li><a href="https://github.com/betweentwomidnights/YuE-instrumental-injection">GitHub - betweentwomidnights/YuE-instrumental-injection</a>：我尝试将自己的乐器音轨注入到 yue 流水线中 - betweentwomidnights/YuE-instrumental-injection
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1341193693226471434)** (4 messages): 

> `UI-Tars 数据集协作，3D 空间中的 AI 模型，SLAM 技术` 


- **征集 UI-Tars 数据集协作**：一位成员表示有兴趣合作重建 **UI-Tars 数据集**。
   - 他们提供了相关讨论的链接以获取更多背景信息。
- **关于 3D 空间中 AI 模型的咨询**：一位成员寻求关于是否有任何 **AI 模型** 可以在 **3D 空间** 中导航，或者 **SLAM** 是否足以满足该目的的信息。
   - *“只是一个学生，”* 他们指出自己对这些技术在该领域的实际运作方式感到好奇。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1341395934533390400)** (2 messages): 

> `基座模型行为，Discord 邀请` 


- **理解基座模型行为**：一位成员承认获得了关于 **基座模型行为** 的见解，并热情地表示感谢。
   - 这突显了社区中关于模型性能和行为分析的持续讨论。
- **不允许发布 Discord 邀请**：另一位成员提醒，分享 Discord 邀请违反了服务器规则，并引用了特定频道的指南。
   - 这反映了社区致力于维护对平台政策的遵守。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1341142926339739648)** (3 messages): 

> `Docling 与 Hugging Face 合作伙伴关系，Visual LLM 与 SmolVLM，GitHub 上的 Pull Request，VLM 的优势` 


- **Docling 与 Hugging Face 联手支持 Visual LLM**：来自 **IBM** 的 Docling 已与 **Hugging Face** 合作，为 Docling 库添加了带有 **SmolVLM** 的 **Visual LLM** 功能。
   - 此次集成旨在通过先进的视觉处理增强文档生成。
- **查看即将发布的 Pull Request**：一位成员提到 Pull Request 已在 **GitHub** 上发布，应该很快就能访问。
   - 他们对这次合作将带来的新功能感到兴奋。
- **咨询 VLM 的优势**：一位用户询问使用输出 **Docling Documents** 的 **Visual LLM** 有什么优势。
   - *“它主要是为了图像描述吗？”* 突显了社区对 VLM 实际应用持续存在的好奇心。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1341092692247056414)** (380 条消息🔥🔥): 

> `AI Agents 课程, 证书问题, Multi-Agent 系统, LLMs 作为工具, 课程资源` 


- **AI Agents 课程成员自我介绍**：多位成员介绍了自己，分享了他们的背景以及对 AI Agents 课程的热情，包括技术职业背景和对 AI 的个人兴趣。
   - 成员们讨论了学习 AI 的动机，包括构建 AI Agents 和探索去中心化模型的愿望。
- **证书生成错误**：多位用户对证书生成错误表示沮丧，在尝试获取证书时通常会收到请求过多的提示消息。
   - 一些人建议了排查步骤，例如使用无痕模式或不同的浏览器，但表示效果并不稳定。
- **Multi-Agent 系统讨论**：围绕 Multi-Agent 系统的对话强调了关于监督 Agent 控制下属 Agent 的想法及其潜在应用。
   - 成员们讨论了相关的论文和资源，强调了实现多个 Agent 的理论方面。
- **LLMs 作为功能性工具**：用户讨论了大型语言模型 (LLMs) 在 AI 中的角色，思考它们是作为工具使用，还是主要作为决策引擎运行。
   - 关于将 LLMs 集成到各种项目中的复杂性，以及对规模和资源使用的影响，展开了持续的对话。
- **课程资源与推荐**：成员们分享了资源链接，例如与 AI Agents 相关的 YouTube 视频和课程，并确定了如 'aisuite' 等用于改进 function calling 的特定工具。
   - 讨论内容包括对补充学习材料的推荐，以及针对初学者的 LLM 和 AI 内容建议。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/">正在重定向&hellip;</a>: 未找到描述</li><li><a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>: MOOC，2024 秋季</li><li><a href="https://huggingface.co/learn/agents-course/bonus-unit1/introduction">简介 - Hugging Face Agents 课程</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit0/introduction">欢迎来到 🤗 AI Agents 课程 - Hugging Face Agents 课程</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/tutorial">让我们使用 smolagents 创建第一个 Agent - Hugging Face Agents 课程</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/communication/next-units">下一单元何时发布？ - Hugging Face Agents 课程</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/tutorial#lets-create-our-first-agent-using-smolagents.">让我们使用 smolagents 创建第一个 Agent - Hugging Face Agents 课程</a>: 未找到描述</li><li><a href="https://github.com/huggingface/agents-course/blob/main/notebooks/bonus-unit1/gemma-SFT-thinking-function_call.ipynb">agents-course/notebooks/bonus-unit1/gemma-SFT-thinking-function_call.ipynb at main · huggingface/agents-course</a>: 此仓库包含 Hugging Face Agents 课程。 - huggingface/agents-course</li><li><a href="https://github.com/andrewyng/aisuite">GitHub - andrewyng/aisuite: 适用于多个生成式 AI 提供商的简单、统一的接口</a>: 适用于多个生成式 AI 提供商的简单、统一的接口 - GitHub - andrewyng/aisuite: Simple, unified interface to multiple Generative AI providers</li><li><a href="https://www.youtube.com/watch?v=qU3fmidNbJE">21 分钟掌握 AI Agents 基础</a>: 通过我与 Hubspot 合作制作的免费 Prompting 快速入门指南提升您的 AI 技能：https://clickhubspot.com/1gg9 想要在...中领先...</li><li><a href="https://www.youtube.com/watch?v=ZZ2QUCePgYw">AI agents 简介</a>: Vertex AI Agent Builder 快速入门 → https://goo.gle/3UPJ7dN 使用 Genkit 构建的 GenAI 驱动应用 → https://goo.gle/4fCSTrK 揭秘 AI agents，Googlers Aja Hamme...</li><li><a href="https://github.com/mindspore-ai/mindspore">GitHub - mindspore-ai/mindspore: MindSpore 是一个新的开源深度学习训练/推理框架，可用于移动、边缘和云场景。</a>: MindSpore 是一个新的开源深度学习训练/推理框架，可用于移动、边缘和云场景。 - mindspore-ai/mindspore</li><li><a href="https://acrobat.adobe.com/id/urn:aaid:sc:EU:38802316-7b5c-48f5-b4a1-c5437d0a48f5">Adobe Acrobat</a>: 未找到描述</li><li><a href="https://github.com/huggingface/agents-course/pull/154">修复：由 rhanb 添加缺失的变量声明 · Pull Request #154 · huggingface/agents-course</a>: 在本地环境中学习本课程的这一部分时，不清楚变量 SYSTEM_PROMPT 和 prompt 指的是什么
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1341236821199421451)** (1 条消息): 

> `MCP 教程` 


- **发布了面向初学者的 MCP 教程**：一位成员分享了关于如何使用 **MCP** 的**初学者指南**，可在 [X](https://x.com/windsurf_ai/status/1891664001941037123) 上查看。
   - 本教程旨在帮助新手有效地掌握 MCP 的功能。
- **参与 MCP 讨论**：鼓励社区探索和讨论 **MCP** 的各个方面，营造协作学习环境。
   - 成员们表示有兴趣分享个人经验和用例，以便更好地理解。



**提到的链接**：<a href="https://x.com/windsurf_ai/status/1891664001941037123">来自 Windsurf (@windsurf_ai) 的推文</a>：关于如何使用 MCP 的初学者指南！

  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1341101625254875268)** (30 条消息🔥): 

> `Codeium Write 模式变更，IntelliJ Supercomplete 功能，多用户 Codeium 部署，Codeium 订阅价值，Jetbrains Context 问题` 


- **Codeium Write 模式限制引入**：最近的更新显示 **Write mode** 不再在免费计划中提供，这促使用户考虑升级或切换到仅聊天模式。
   - 这一变化引发了关于 **Write mode** 的缺失是对所有用户永久生效还是仅针对少数用户的讨论。
- **对 IntelliJ 的 Supercomplete 功能的困惑**：成员们讨论了 **IntelliJ** 扩展是否曾拥有 **supercomplete** 功能，并参考了其在 **VSCode pre-release** 版本中的存在。
   - 一些用户指出，**supercomplete** 可能指的是多行补全（multiline completions），而 **autocomplete** 则涵盖单行建议。
- **寻求 Codeium 自动部署解决方案**：一位用户询问了如何为使用 **IntelliJ 和 Codeium** 的多名用户自动完成设置，希望能有一个流线化的身份验证流程。
   - 回复指出该功能可能在企业版方案中提供，并建议联系 **Codeium 的企业支持**。
- **评估 Codeium 订阅的价值**：围绕 **$60 Codeium 订阅** 的价值展开了讨论，强调其价值很大程度上取决于用户的预期使用场景。
   - 对于自由职业者来说，投资回报可能非常可观，而对于偶尔使用的用户，可能会觉得收益较小。
- **Jetbrains 中的 Context 检索问题**：多位成员报告了 **Codeium** 在 **Jetbrains** 中无法找到上下文的问题，在使用 @commands 时遇到了超时和文件可见性受限的情况。
   - 建议确保安装了最新版本，因为这可能会解决一些持续存在的 Context 问题。



**提到的链接**：<a href="https://codeium.com/contact/enterprise">Contact | Windsurf Editor and Codeium extensions</a>：联系 Codeium 团队获取支持并了解更多关于我们企业版方案的信息。

  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1341092043354804414)** (418 messages🔥🔥🔥): 

> `Cascade Base 性能问题、评分模型内部错误、对 Grok 3 的挫败感、订阅与计费关注、AI 模型响应质量` 


- **Cascade Base 存在显著性能问题**：用户报告 Cascade Base 无法对代码进行编辑，且有时 AI 聊天窗口会消失，导致用户感到挫败。
   - 多名用户在尝试使用模型时遇到了内部错误，这表明平台存在持续的稳定性问题。
- **用户对内部错误表示沮丧**：许多人在尝试使用 DeepSeek R1 等模型访问功能时遇到了内部错误提示，表明存在服务器端问题。
   - 部分用户已连续多日无法使用核心功能，导致对所购订阅服务表示不满。
- **Grok 3 表现令用户失望**：多位成员讨论了对 Grok 3 性能的失望，指出其表现未达到基于初始基准测试的预期。
   - 讨论中包含了对未来模型（尤其是 Sonnet 4）改进的渴望，用户对 Grok 的能力提出了批评。
- **订阅与计费问题**：用户对从创始人定价（founder pricing）过渡到 Pro 方案表示担忧，并询问有关取消订阅选项的问题。
   - 用户担心在账户出现问题期间会丢失额度（credits），并寻求通过支持渠道澄清计费流程。
- **模型响应与交互质量**：用户注意到对 Cascade 的理解能力和指令遵循情况总体不满，特别是关于响应过于冗长的问题。
   - 用户呼吁更好地管理 AI 行为，包括在代码生成期间添加注释以简化交互。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/advanced">Windsurf - Advanced</a>：未找到描述</li><li><a href="https://codeium.canny.io/feature-requests/">Feature Requests | Codeium</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://artificialanalysis.ai">AI Model &amp; API Providers Analysis | Artificial Analysis</a>：AI 模型和 API 托管提供商的对比与分析。针对质量、价格、输出速度和延迟等关键性能指标的独立基准测试。</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>：需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://x.com/theo/status/1891736803796832298">Theo (@theo) 的推文</a>：Grok 3，呃，在编程方面表现不佳</li><li><a href="https://x.com/kevinhou22/status/1886827501004931511">Kevin Hou (@kevinhou22) 的推文</a>：我们热爱文档！📖 我正在努力改进/添加更多 @windsurf_ailmk 的文档快捷方式，告诉我你想要的，我会尽力添加... 🧵另外感谢 @mintlify 自动托管所有文档...</li><li><a href="https://x.com/">GitHub - FixTweet/FxTwitter 的推文</a>：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1341099580170965074)** (420 messages🔥🔥🔥): 

> `Grok 3 发布、Aider 功能问题、性能对比、模型访问与支持、AI 模型讨论`

- **Grok 3 发布的影响**：Grok 3 在 AI 社区引起了轰动，有帖子暗示它是一款突破性的模型，超越了 GPT 和 Claude 等现有选项。
   - 许多用户对这些说法表示怀疑，认为它们夸大其词或过于乐观。
- **Aider 文件添加问题**：用户报告称，尽管输入了正确的文件名并进行了初始化，Aider 有时仍无法自动添加文件。
   - 社区正在讨论 Aider Web 界面中的故障排除步骤和潜在的 bug。
- **Grok 3 的性能比较**：关于 Grok 3 相对于 GPT-4 和 Sonnet 等模型的性能存在持续争论，用户评价褒贬不一。
   - 一些人认为 Grok 3 在实际应用场景中表现不佳，尽管它在模型排行榜上名列前茅。
- **访问 Gemini 模型**：用户正在探索如何通过 Aider 使用 Gemini 实验性模型，并讨论了不同的命令配置。
   - 对于模型标识符以及在实现过程中收到的警告存在困惑。
- **社区对 AI 评论的反应**：一些社区成员指出了关于 Grok 3 能力的过度狂热帖子，并将其贴上“盲目崇拜”的标签。
   - 有人对 AI 进展相关说法的可信度提出质疑，特别是关于 AGI 及其实际影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/install/docker.html">Aider with docker</a>: Aider 是你终端里的 AI 结对编程工具</li><li><a href="https://x.com/karpathy/status/1891720635363254772">Andrej Karpathy (@karpathy) 的推文</a>: 我今天早些时候获得了 Grok 3 的早期访问权限，我想我应该是第一批进行快速体验 (vibe check) 的人之一。思考✅ 首先，Grok 3 显然拥有一个处于 state of the art 水平的思考模型 ...</li><li><a href="https://www.kaggle.com/competitions/konwinski-prize">Konwinski Prize</a>: 100 万美元奖励给能解决 90% 新 GitHub issues 的 AI</li><li><a href="https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/">我目前的 LLM codegen 工作流</a>: 详细介绍了目前我使用 LLM 构建软件的工作流，从头脑风暴到计划和执行。</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/ooh-ooo-cat-shocked-funny-gif-14366308">Ooh Ooo GIF - Ooh Ooo Cat - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/impenny2x/status/1891583553911001333?s=46">Penny2x (@impenny2x) 的推文</a>: 我的天。</li><li><a href="https://tenor.com/view/nacho-libre-why-but-gif-23595404">Nacho Libre GIF - Nacho Libre Why - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/staring-into-space-chimpanzee-our-living-world-spaced-out-monkey-gif-17145372685197420070">凝视虚空的黑猩猩 GIF - Staring into space Chimpanzee Our living world - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/openai/SWELancer-Benchmark">GitHub - openai/SWELancer-Benchmark: 此仓库包含论文 "SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?" 的数据集和代码</a>: 此仓库包含论文 "SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?" 的数据集和代码 - openai/SWELancer-Benchmark</li><li><a href="https://x.com/perplexity_ai/status/1891916573713236248?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">Perplexity (@perplexity_ai) 的推文</a>: 今天我们开源了 R1 1776——这是 DeepSeek R1 模型的一个版本，经过 post-trained 以提供无审查、无偏见且事实准确的信息。</li><li><a href="https://x.com/iruletheworldmo/status/1891529614167789967">🍓🍓🍓 (@iruletheworldmo) 的推文</a>: Grok 3 来了，它是 AGI。这不仅仅是另一个模型发布。这是改变一切的时刻。忘掉 GPT，忘掉 Claude，忘掉你以前用过的所有 AI——它们已经过时了。</li><li><a href="https://github.com/ai-christianson/RA.Aid">GitHub - ai-christianson/RA.Aid: 自主开发软件。</a>: 自主开发软件。通过在 GitHub 上创建账号来为 ai-christianson/RA.Aid 的开发做出贡献。</li><li><a href="https://github.com/robert-at-pretension-io/rust_web_scraper">GitHub - robert-at-pretension-io/rust_web_scraper</a>: 通过在 GitHub 上创建账号来为 robert-at-pretension-io/rust_web_scraper 的开发做出贡献。</li><li><a href="https://x.com/elonmusk/status/1891911120572567983">Elon Musk (@elonmusk) 的推文</a>: @xAI Grok 3 的发布将在本周每天迅速改进。请在此帖下回复报告任何问题。</li><li><a href="https://github.com/Aider-AI/aider/pull/2628">gembancud 提交的 Moa · Pull Request #2628 · Aider-AI/aider</a>: 添加 Mixture of Architects (MOA) 功能。当你可以同时拥有 r1、o3 和 sonnet 时，为什么还要做选择呢！概述：此 PR 引入了一个名为 "Mixture of Architects" 的强大新功能...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1341116532771520572)** (24 条消息🔥): 

> `Aider Search Engine, Architect Mode Suggestions, Aider Configuration Issues, Aider Not Recognizing .env, API Key Requirement` 


- **对 Aider 搜索引擎集成的兴趣**：@bixqu 询问了如何向 Aider 添加 Web 搜索引擎，类似于 Cursor 中的实现方式，可能使用 Perplexity。
   - 另一位成员分享了 **RA.Aid** 可以与 Aider 集成并利用 Tavily API 进行 Web 搜索，并提供了其 [GitHub 仓库](https://github.com/ai-christianson/RA.Aid/) 的链接。
- **Architect Mode 中的挑战**：成员们对 Architect Mode 频繁提示文件编辑感到沮丧，这干扰了关于架构的讨论。
   - 一个建议是切换到 Code Mode，以便在不触发文件编辑提示的情况下进行交流。
- **使用自定义命令配置 Aider**：有人尝试通过 `aider.conf` 配置文件集成命令库并自定义 Aider 的 System Prompt。
   - 关于如何正确使用 `--load` 和 `--read` 参数来增强命令库上下文的问题被提出。
- **Aider 无法识别 .env 文件的问题**：一位用户报告了在使用 API 时 Aider 无法识别 `.env` 文件的问题，导致身份验证错误。
   - 另一位用户确认他们每次都需要设置 API Key，这在之前被忽略了。
- **Aider 执行中遇到的错误**：一位用户分享了在尝试执行 Aider 时遇到的未捕获 RuntimeError，突显了 Python 3.12 中事件循环的问题。
   - 他们随后澄清说基础 Aider 运行正常，问题源于未正确设置 API Key。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-aider-uses">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://github.com/ai-christianson/RA.Aid/">GitHub - ai-christianson/RA.Aid: Develop software autonomously.</a>：自主开发软件。通过在 GitHub 上创建账号来为 ai-christianson/RA.Aid 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1341326679175335946)** (4 条消息): 

> `Local sanity checks for coding LLM output, Ragit GitHub project, Ministral with Aider` 


- **本地编程 LLM 改进**：一位成员建议在本地运行编程 LLM，以对其输出进行 **合理性检查 (sanity checks)** 和改进。
   - 这可以为编程任务提供更具针对性的调整和验证。
- **对 Ragit GitHub 流水线的兴趣**：GitHub 上的 **Ragit** 项目（被描述为类 Git 的 RAG 流水线）引起了多位成员的关注，引发了对其潜在应用的兴趣。
   - 一位成员强调了其在 RAG 处理流水线方面的 **创新方法**。
- **探索 Ministral 与 Aider**：有人提问是否有人尝试过将 **Ministral** 与 Aider 结合使用。
   - 一篇相关的 [LinkedIn 帖子](https://www.linkedin.com/posts/deividas-mataciunas_ai-mistral-opensource-ugcPost-7294619759722074112-a_JM?utm_source=share&utm_medium=member_android&rcm=ACoAABZCz5cBsMAYVy_zzTHh2HzsmuBv_27C49Y) 引发了关于此话题的讨论，表明了对集成的潜在好奇心。



**提到的链接**：<a href="https://github.com/baehyunsol/ragit">GitHub - baehyunsol/ragit: git-like rag pipeline</a>：类 Git 的 RAG 流水线。通过在 GitHub 上创建账号来为 baehyunsol/ragit 的开发做出贡献。

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1341099931498188923)** (435 条消息🔥🔥🔥): 

> `Grok 3 性能, OpenRouter API 使用, 模型对比, LLM 中的视觉能力, DeepSeek 对比 Sonnet` 


- **Grok 3 的初步反响**：Grok 3 的评价褒贬不一，一些用户称赞其能力，而另一些用户则对其与 Claude 和 Sonnet 等成熟模型相比的性能表示怀疑。
   - 用户特别关注 Grok 3 处理代码和推理任务的表现，并引用了其表现出色的具体案例。
- **OpenRouter API 使用政策**：关于 OpenRouter 的使用政策存在困惑，特别是涉及 NSFW 内容生成以及对供应商政策的合规性。
   - 用户间的讨论表明，与其他平台相比，OpenRouter 的限制可能较少，但最好向管理员核实。
- **模型对比**：讨论重点关注了 DeepSeek、Sonnet 和 Claude 等各种模型的性能，尽管成本较高，一些用户因其可靠性而更倾向于使用 Sonnet 处理编程任务。
   - 用户注意到 Grok 3 和 DeepSeek 提供了具有竞争力的功能，许多人在选择替代方案时会考虑价格和性能。
- **LLM 中的视觉能力**：一位用户询问了 OpenRouter 上可以分析图像的模型，并参考了供应商网站上详细介绍具有文本和图像能力模型的模块部分。
   - 建议探索该部分下的可用模型，以找到具有视觉能力的合适选项。
- **API 交易的用户体验**：一位用户报告在尝试购买 OpenRouter 额度时遇到问题，在向银行澄清后寻求帮助。
   - 这引发了关于不同 LLM 模型定价结构的讨论，以及关于它们所产生价值的持续辩论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/stepfun-ai/Step-Audio-Chat">stepfun-ai/Step-Audio-Chat · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard">Chatbot Arena Leaderboard - 由 lmarena-ai 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - 管理模型使用和配额</a>：了解 OpenRouter 的 API 速率限制、基于额度的配额和 DDoS 防护。有效配置和监控您的模型使用限制。</li><li><a href="https://openrouter.ai/models">Models | OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://clinicalai.lovable.app/">Clinical AI & Quantum Hackathon</a>：未找到描述</li><li><a href="https://x.com/perplexity_ai/status/1891916644248846789?t=7_5m7rcR2w7GFITF2I2QSA&s=19">来自 Perplexity (@perplexity_ai) 的推文</a>：在我们的 Hugging Face 仓库下载模型权重，或考虑通过我们的 Sonar API 使用该模型。Hugging Face 仓库：https://huggingface.co/perplexity-ai/r1-1776</li><li><a href="https://openrouter.ai/models?order=top-weekly">Models | OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://artificialanalysis.ai)">未找到标题</a>：未找到描述</li><li><a href="https://cloud.google.com/compute/gpus-pricing?hl=en">GPU pricing</a>：GPU 定价。</li><li><a href="https://techcrunch.com/2025/02/17/elon-musks-ai-company-xai-releases-its-latest-flagship-ai-grok-3/">埃隆·马斯克的 xAI 发布其最新旗舰模型 Grok 3 | TechCrunch</a>：埃隆·马斯克的 AI 公司 xAI 周一发布了其最新的旗舰 AI 模型 Grok 3，同时在 iOS 和网页版的 Grok 应用中推出了新功能。</li><li><a href="https://cloud.google.com">Cloud Computing Services | Google Cloud</a>：通过 Google 的云计算服务正面迎接您的业务挑战，包括数据管理、混合云与多云、以及 AI 和 ML。</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1#usage-recommendations">GitHub - deepseek-ai/DeepSeek-R1</a>：通过创建账号为 deepseek-ai/DeepSeek-R1 的开发做出贡献。</li><li><a href="https://github.com/OpenRouterTeam/ai-sdk-provider/issues/22">No Reasoning tokens · Issue #22 · OpenRouterTeam/ai-sdk-provider</a>：使用 streamText 时不提供推理 Token，且无法在 streamText 中设置 include_reasoning</li><li><a href="https://yuewen.cn/chats/new">跃问</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1341134882683879505)** (335 条消息🔥🔥): 

> `Grok 3, Sonnet 性能, MCP 服务器设置, Cursor 性能问题, 用户对 AI 模型的反馈`

- **Grok 3 性能引发担忧**：用户对 Grok 3 表示失望，认为其表现低于预期标准，经常出现重复内容，且在代码执行任务方面表现不佳。
   - 尽管在推理能力方面有一些正面反馈，但许多评论者认为它只是在追赶 Sonnet 等现有模型。
- **Sonnet 仍是用户的首选**：许多用户认为 Sonnet（尤其是 3.5 版本）优于 Grok 3 等新兴模型，在编程任务中保持着领先优势。
   - 一些用户报告 Sonnet 的 hallucinations 有所增加，而另一些人则认为与新模型相比，它依然可靠。
- **设置 MCP 服务器**：关于设置 MCP 服务器以使用各种 AI 工具的复杂性引发了讨论，一些用户分享了集成这些工具的见解和经验。
   - 用户尝试使用单文件 Python 脚本作为传统 MCP 服务器设置的替代方案，探索提高灵活性。
- **Cursor 在处理大型代码库时表现挣扎**：用户报告了 Cursor 在有效处理规则和 context 方面的问题，特别是在导航大型代码库时，导致必须手动添加规则。
   - 尽管感到沮丧，一些用户指出降低版本有助于提高性能，表现出测试解决方案的意愿。
- **关于 AI 指令处理的反馈**：用户持续关注其指令是否被 AI 模型正确处理，并建议在 prompt 中加入显式检查。
   - 一些用户建议采用奇特的方法来测试 AI 是否遵循了引导，这表明需要深入了解各平台对 instruction adherence 的情况。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<a href="https://x.com/karpathy/status/1891720635363254772">来自 Andrej Karpathy (@karpathy) 的推文</a>：今天早些时候我获得了 Grok 3 的早期访问权限，我想我应该是第一批进行快速“体感测试”的人之一。思考中✅ 首先，Grok 3 显然拥有一个处于 SOTA 状态的思考模型 ...</li><li><a href="https://x.com/ryolu_/status/189">来自 Biz Stone (@biz) 的推文</a>：暂时离开办公室一会儿</li><li><a href="https://x.com/ryolu_/status/1891677600587629043?s=46">来自 Ryo Lu (@ryolu_) 的推文</a>：情人节迟到了，但 @cursor_ai 一直在憋大招</li><li><a href="https://x.com/iBigQiang/status/1890993390172406160">来自 BigQiang (@iBigQiang) 的推文</a>：在 B 站看到个使用 Cloudflare 无限续杯 Cursor 教程和配套脚本实在太牛了，条件是拥有域名并且需要交给 Cloudflare 托管，不过域名也就几块钱而已，相比每个月 20 刀的 Cursor 还是很香了。Cursor 能写代码、写文章，大部分 AI 能干的事他都可以，可白嫖 Claude 3.5 sonnet，DeepSeek v3&r1，GPT-4o 和 GPT-4o mini 等</li><li><a href="https://x.com/kevinhou22/status/1891375289919500794?t=k5skkvhMsKodfbvKbDiYhw&s=19">来自 Kevin Hou (@kevinhou22) 的推文</a>：我们构建 Windsurf 的 Agent 时，不依赖于像其他工具那样的 Embedding 索引。通用的检索方式根本无法扩展到 Monorepos。相反，我们的 Agent 使用人类会使用的工具 ...</li><li><a href="https://github.com/bgstaal/multipleWindow3dScene">GitHub - bgstaal/multipleWindow3dScene：一个关于如何使用 three.js 和 localStorage 在多个窗口之间“同步” 3D 场景的快速示例</a>：一个关于如何使用 three.js 和 localStorage 在多个窗口之间“同步” 3D 场景的快速示例 - bgstaal/multipleWindow3dScene</li><li><a href="https://x.com/kimmonismus/status/1891590879430754550">来自 Chubby♨️ (@kimmonismus) 的推文</a>：Grok-3 开始推送了。检查你的 Grok 选项。引用 Penny2x (@imPenny2x)：天哪。</li><li><a href="https://x.com/amanrsanger/status/1891630232802640018">来自 Aman Sanger (@amanrsanger) 的推文</a>：刚刚在 Cursor 中添加了 Grok-2 支持，正迫不及待期待 Grok-3 👀</li><li><a href="https://x.com/OpenAI/status/1891911132983722408">来自 OpenAI (@OpenAI) 的推文</a>：目前的前沿模型无法解决大部分任务。</li><li><a href="https://x.com/deepseek_ai/status/1891745487071609327?s=46">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 推出 NSA：一种硬件对齐且原生可训练的稀疏注意力（Sparse Attention）机制，用于超快速的长上下文训练和推理！NSA 的核心组件：• 动态分层稀疏策略 •...</li><li><a href="https://github.com/daniel-lxs/mcp-perplexity">GitHub - daniel-lxs/mcp-perplexity：用于 Perplexity API 的 MCP Server。</a>：用于 Perplexity API 的 MCP Server。通过在 GitHub 上创建账号来为 daniel-lxs/mcp-perplexity 的开发做出贡献。</li><li><a href="https://x.com/12exyz/status/1891723056931827959">来自 Rex (@12exyz) 的推文</a>：出于某种原因，他们在直播的图表中省略了 o3，所以我帮你们把数据加上了</li><li><a href="https://simonwillison.net/2024/Dec/19/one-shot-python-tools/">使用 uv run 和 Claude Projects 通过 One-shot 提示词构建 Python 工具</a>：我写过很多关于如何通过 Claude Artifacts 使用 Claude 构建 One-shot HTML+JavaScript 应用程序的文章。最近我开始使用类似的模式来创建 One-shot Python 实用程序，...</li><li><a href="https://github.com/disler/single-file-agents">GitHub - disler/single-file-agents：如果我们能将单一用途、功能强大的 AI Agent 打包进一个 Python 文件中会怎样？</a>：如果我们能将单一用途、功能强大的 AI Agent 打包进一个 Python 文件中会怎样？ - disler/single-file-agents</li><li><a href="https://x.com/karpathy/status/1891720635363254772/photo/1">来自 Andrej Karpathy (@karpathy) 的推文</a>：今天早些时候我获得了 Grok 3 的早期访问权限，我想我应该是第一批进行快速“体感测试”的人之一。思考中✅ 首先，Grok 3 显然拥有一个处于 SOTA 状态的思考模型 ...</li><li><a href="https://github.com/chengazhen/cursor-auto-free">GitHub - chengazhen/cursor-auto-free：自动注册 Cursor</a>：自动注册 Cursor。通过在 GitHub 上创建账号来为 chengazhen/cursor-auto-free 的开发做出贡献。</li><li><a href="https://www.subframe.com/">Subframe – 快速构建 UI 的最佳方式。</a>：通过拖拽式可视化编辑器、精美的组件和生产就绪的代码，在几分钟内构建出色的 UI。针对 React 和 TailwindCSS 进行了优化。</li><li><a href="https://www.relume.io/">Relume — 利用 AI 更快地设计和构建网站 | AI 网站生成器</a>：将 AI 作为你的设计盟友，而非替代品。利用 Relume 的 AI 网站生成器，在几分钟内轻松为营销网站生成站点地图和线框图。</li><li><a href="https://21st.dev/">21st.dev - 面向设计工程师的 NPM</a>：使用受 shadcn/ui 启发、开箱即用的 React Tailwind 组件，更快地交付精致的 UI。由设计工程师构建，服务于设计工程师。</li><li><a href="https://flexboxlabs.netlify.app/">Flexbox Labs</a>：无描述 f

ound
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1341237163840503809)** (209 messages🔥🔥): 

> `Grok 3 性能、Llama 4 更新、Thinking Machines Lab 启动、AI 评估方法论、GPT-4o Copilot 发布` 


- **Grok 3 通过了氛围感测试但缺乏深度**：**Grok 3** 的早期评论表明它是一款前沿级模型，但与 **R1** 和 **o1-pro** 等其他模型相比，被认为“平淡且乏味”。
   - 据报道，它在提供技术深度解释方面表现不佳，并且在处理长查询时存在注意力压缩（attention compression）问题。
- **Llama 4 在 DeepSeek 事件后可能正在经历调整**：扎克伯格宣布 **Llama 4** 已完成预训练，但有推测称在最近的 **DeepSeek** 消息之后，它可能会被废弃。
   - 人们担心竞争对手的反馈将影响 **Llama 4** 的最终发布策略，这凸显了 AI 发展快速迭代的本质。
- **Thinking Machines Lab 成立**：由 Mira Murati 等行业领袖共同创立的 **Thinking Machines Lab** 旨在弥合 AI 可定制性和理解力方面的差距。
   - 该实验室致力于开放科学（open science），并计划优先考虑人机协作，使各领域的进步变得触手可及。
- **关于 Eval 方法论及其有效性的辩论**：目前关于现有 **Eval** 方法可靠性的讨论正在进行中，一些人认为该行业缺乏强大的测试框架。
   - 有人担心许多公司专注于表面的性能指标，而不是反映真实能力的实质性测试。
- **用于编程任务的 GPT-4o Copilot 发布**：一款全新的代码补全模型 **GPT-4o Copilot** 已发布，目前已在各种 **IDEs** 中提供公开预览版。
   - 该模型基于庞大的代码语料库进行微调（fine-tuned），旨在提高主要编程语言的编码效率。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.12115">SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?</a>：我们推出了 SWE-Lancer，这是一个包含 1,400 多个来自 Upwork 的自由职业软件工程任务的基准测试，在真实世界的支付总额达 100 万美元。SWE-Lancer 涵盖了独立工程...</li><li><a href="https://x.com/12exyz/status/1891723056931827959">来自 Rex (@12exyz) 的推文</a>：出于某种原因，他们在直播的图表中省略了 o3，所以我为你补全了这些数据</li><li><a href="https://thinkingmachines.ai/">Thinking Machines Lab</a>：未找到描述</li><li><a href="https://x.com/karpathy/status/1891720635363254772">来自 Andrej Karpathy (@karpathy) 的推文</a>：今天早些时候我获得了 Grok 3 的早期访问权限，我想我应该是第一批能进行快速“氛围检查”（vibe check）的人之一。Thinking✅ 首先，Grok 3 显然拥有一个接近 state of the art 的思维模型...</li><li><a href="https://x.com/james_tackett1/status/1891898442206638237">来自 Jamie (@james_tackett1) 的推文</a>：@natolambert 既然 xAI 演示中的图表显示 Grok 3 以相当大的优势领先于所有其他 LLM，你为什么认为其他 LLM 处于领先地位？例如：</li><li><a href="https://x.com/TheXeophon/status/1891795532500111752">来自 Xeophon (@TheXeophon) 的推文</a>：上一次发生这种情况是在 *查看笔记* 两周前。引用 Gavin Baker (@GavinSBaker)：如果我没记错的话（IIRC），这是这一年多来第一次有一个模型在每个类别中都排名第一。</li><li><a href="https://x.com/ashtom/status/1891925306430337110">来自 Thomas Dohmke (@ashtom) 的推文</a>：我们新的代码补全模型今天开始公开预览。我们将其命名为 GPT-4o Copilot。基于 GPT-4o mini，在超过 1T tokens 的代码专用语料库上进行了 mid-training，并进行了强化...</li><li><a href="https://x.com/dchaplot/status/1891920016339042463">来自 Devendra Chaplot (@dchaplot) 的推文</a>：职业更新：非常幸运且兴奋能成为 Thinking Machines Lab 创始团队的一员！https://thinkingmachines.ai/ 加入我们：https://6wajk07p.paperform.co/</li><li><a href="https://x.com/KateClarkTweets/status/1891594184697487771">来自 Kate Clark (@KateClarkTweets) 的推文</a>：Greenoaks Capital 正在领投 Safe Superintelligence 的一轮超过 10 亿美元的融资，这家由前 OpenAI 首席科学家 Ilya Sutskever 创立的 AI 初创公司估值超过 300 亿美元。独家报道：https://www.bloo...</li><li><a href="https://x.com/arankomatsuzaki/status/1891717076479328711?s=61">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：OpenAI 发布：SWE-Lancer：前沿 LLM 能从真实的自由职业软件工程中赚取 100 万美元吗？- 开源了一个包含 >1.4k 个来自 Upwork 的自由职业 SWE 任务的基准测试，总价值 100 万美元...</li><li><a href="https://x.com/GavinSBaker/status/1891723733976465420">来自 Gavin Baker (@GavinSBaker) 的推文</a>：如果我没记错的话（IIRC），这是这一年多来第一次有一个模型在每个类别中都排名第一。</li><li><a href="https://x.com/_xjdr/status/1891911178147987513">来自 xjdr (@_xjdr) 的推文</a>：TL;DR Grok 3 表现不错，通过了前沿水平质量的“氛围检查”，但对我做的大多数事情来说，它并不比 R1 或 o1-pro 更好。总的来说比我预期的要好得多，我把它放在 Gemini...</li><li><a href="https://x.com/zephyr_z9/status/1891716422109135332">来自 Zephyr (@zephyr_z9) 的推文</a>：DeepSeek V3 (4.6e24 Flops) 对比 Grok 3 (>8e26 Flops)，算力消耗高出 173 倍</li><li><a href="https://x.com/arankomatsuzaki/status/1891708250199839167?s=61">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：Grok 3 推理测试版在 AIME 上达到了 96 分，在 GPQA 上达到了 85 分，与完整版 o3 持平。</li><li><a href="https://x.com/andrewcurran_/status/1891714024141664478?s=61">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：当 Grok 3 稳定后，Grok 2 将开源，“大概几个月后”。</li><li><a href="https://x.com/andrewcurran_/status/1891707314782040101?s=61">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：CoT 将被部分遮蔽，Elon 说“这样我们的模型就不会被立即复制”。</li><li><a href="https://x.com/lmarena_ai/status/1891706264800936307">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：重大新闻：@xAI 的 Grok-3 早期版本（代号 "chocolate"）现在在 Arena 中排名第一！🏆 Grok-3 是：- 有史以来第一个突破 1400 分的模型！- 在所有类别中均排名第一，这是一个不断被刷新的里程碑...</li><li><a href="https://x.com/altryne/status/1884778839009796411">来自 Alex Volkov (Thursd/AI) 🔜 AIENG summit NY (@altryne) 的推文</a>：扎克伯格在财报电话会议上的亮点：- Llama 4 和 Llama 4 mini（已完成预训练）- 确认了推理版 Llama！- Llama 4 将是原生多模态的——它是一个全能模型（omni-model）——并且它将拥有...</li><li><a href="https://x.com/paul_cal/status/1891718513393271248?s=61">来自 Paul Calcraft (@paul_cal) 的推文</a>：LMSYS 上的 Grok 3 并不是那么“基于事实”（based），令人担忧。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1341168405964783660)** (77 条消息🔥🔥): 

> `新 LLM 模型、AI 中的早停技术 (Early Stopping)、AI 音乐与艺术、Thinking Machines Corporation、AI 访谈洞察` 


- **用户赞赏新 LLM 模型 4o**：用户发现新模型 **4o** 相比 **o1 mini** 和 **o3 mini** 使用体验更愉快，能为各种任务提供更简洁的解决方案。
   - *一位用户提到：*“它让日常搜索变得简单，并能高效地应对挑战。”
- **创新的早停技术 (Early Stopping)**：讨论中提到了一个名为 **Dynasor** 的模型，该模型声称通过在推理过程中有效管理“自我怀疑”，可以节省高达 **81% 的 tokens**。
   - 该模型利用中途探测（halfway probe）来评估**确定性 (certainty)**，从而决定何时停止，在无需训练的情况下提升了效率。
- **AI 音乐日益普及**：**AI 生成的 lofi 音乐**受到了关注，分享的数据显示一段视频获得了 **210 万次播放**，且在 Spotify 上拥有庞大的听众群。
   - 用户表示惊讶，称他们没意识到这些音乐是 AI 生成的，这凸显了主流社会对 AI 艺术接受度提高的趋势。
- **重温 Thinking Machines Corporation**：以超级计算机闻名的 **Thinking Machines Corporation** 被作为案例讨论，探讨了当时的扩展方法 (scaling methods) 如何导致了过去的 AI 寒冬挑战。
   - 用户评论了该公司在早期 AI 发展背景下的历史意义及其创新方法。
- **AI 访谈洞察**：一场由 **Google DeepMind** 和 **Anthropic** 创始人参加的访谈讨论了 AI 的影响和竞争格局，揭示了对中心化问题的担忧。
   - 用户对这些见解进行了反思，表示很高兴看到此类讨论的开展，即使这些内容未必具有开创性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/deepseek_ai/status/1891745487071609327">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 推出 NSA：一种硬件对齐且原生可训练的 Sparse Attention 机制，用于超快速的长上下文训练与推理！NSA 的核心组件：• 动态分层稀疏策略•...</li><li><a href="https://fxtwitter.com{match.group('tweet')}"">未找到标题</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Thinking_Machines_Corporation">Thinking Machines Corporation - 维基百科</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2502.11089">Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention</a>：长上下文建模对下一代语言模型至关重要，但标准 Attention 机制的高昂计算成本带来了重大的计算挑战。Sparse Attention 提供...</li><li><a href="https://tenor.com/view/thanos-perfectlybalanced-gif-18301221">灭霸完美平衡 GIF - Thanos Perfectlybalanced - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/nearcyan/status/1891926678810607858">来自 near (@nearcyan) 的推文</a>：@thinkymachines 你的 X 账号名是不是打错了，我不确定 thinky machines 是否有同样的感觉</li><li><a href="https://x.com/littmath/status/1891868790314434809">来自 Daniel Litt (@littmath) 的推文</a>：换句话说，它再次声称生成了完整的数据集，但实际上只生成了约 7 行，其余约 3000 行都是占位符。</li><li><a href="https://moonlight-mod.github.io/extensions-list/#textReplacer">扩展列表</a>：未找到描述</li><li><a href="https://github.com/MoonshotAI/MoBA">GitHub - MoonshotAI/MoBA: MoBA: Mixture of Block Attention for Long-Context LLMs</a>：MoBA: 用于长上下文 LLM 的 Mixture of Block Attention - MoonshotAI/MoBA</li><li><a href="https://x.com/haoailab/status/1891581639131631893">来自 Hao AI Lab (@haoailab) 的推文</a>：[3/n] 🐢 推理模型通常会自我怀疑。模型仅用 300 个 token 就得出了正确答案，但却额外花费了 990 个 token 进行无意义的验证循环，完全没有进展！➡️ &...</li><li><a href="https://x.com/Kimi_Moonshot/status/1891825059599352259">来自 Kimi.ai (@Kimi_Moonshot) 的推文</a>：🚀 推出 MoBA：用于长上下文 LLM 的 Mixture of Block Attention。很高兴分享我们关于 Mixture of Block Attention (MoBA) 的最新研究！这种创新方法彻底改变了长上下文处理...</li><li><a href="https://x.com/littmath/status/1891868756340547809">来自 Daniel Litt (@littmath) 的推文</a>：我在使用 OpenAI 的 Deep Research 工具时有一次有趣的经历，想分享给大家，因为我认为它揭示了该工具的一些优缺点。</li><li><a href="https://x.com/littmath/status/1891868775051342232">来自 Daniel Litt (@littmath) 的推文</a>：唯一的问题是，这全是编造的。尽管声称查阅了这 75 年间在《数学年刊》(Annals) 上发表的每一篇论文，但查看它浏览过的页面后发现，它实际上只看了...</li><li><a href="https://github.com/moonlight-mod/moonlight">GitHub - moonlight-mod/moonlight: 又一个 Discord 模组</a>：又一个 Discord 模组。通过在 GitHub 上创建一个账户来为 moonlight-mod/moonlight 的开发做出贡献。</li><li><a href="https://fxtwitter.com/haoailab/status/1891581634773651763">来自 Hao AI Lab (@haoailab) 的推文</a>：推理模型经常在自我怀疑中浪费 token。Dynasor 可以为你节省高达 81% 的 token 来得出正确答案！🧠✂️- 在中途探测模型以获取确定性 - 利用确定性停止推理...</li><li><a href="https://x.com/tuzhaopeng/status/1891346931433255300">来自 Zhaopeng Tu (@tuzhaopeng) 的推文</a>：我们是否在蒙特卡洛树搜索 (MCTS) 中高效且智能地扩展推理侧计算 (test-time compute)？探索我们的灵活框架，它使推理模型能够根据...调整其计算。</li><li><a href="https://x.com/siyuanhuang95/status/1891760580408573986?s=46">来自 Siyuan Huang (@siyuanhuang95) 的推文</a>：宇树科技 (Unitree) CEO 王兴兴在小红书 (Rednote) 发布了一段跳舞视频，以回应此前关于跳舞视频是由 AI 或 CG 生成的炒作。舞蹈是在镜子前进行的，并带有声音，这...</li><li><a href="https://x.com/TXhunyuan/status/1891752519837139366">来自 Hunyuan (@TXhunyuan) 的推文</a>：嘿兄弟，好久不见，猜猜这次我给你带了什么？</li><li><a href="https://x.com/TheXeophon/status/1891586946675216803">来自 Xeophon (@TheXeophon) 的推文</a>：一个视频 210 万次播放，每月 12.7 万 Spotify 听众，全是 AI 生成的 lofi。起初很慢，然后突然爆发。引用 Xeophon (@TheXeophon)：没意识到我一直在听 AI 生成的音乐，直到...</li><li><a href="https://youtu.be/4poqjZlM8Lo?si=E6Y9rdAOYFjUeBhq)">AI 巨头们谈论让他们彻夜难眠的事</a>：Google DeepMind 和 Anthropic 的创始人 Demis Hassabis 和 Dario Amodei 是全球人工智能领域最杰出的两位领导者。</li>

igence. Our editor-in-ch...</li><li><a href="https://youtu.be/4poqjZlM8Lo?s">AI 巨头们在担心什么</a>：Google DeepMind 和 Anthropic 的创始人 Demis Hassabis 和 Dario Amodei 是全球最顶尖的人工智能领袖。我们的总编辑...</li><li><a href="https://x.com/sama/status/1891667332105109653?s=61">Sam Altman (@sama) 的推文</a>：对于我们的下一个开源项目，是做一个 o3-mini 级别的、体积较小但仍需在 GPU 上运行的模型更有用，还是我们能做的最好的手机端模型？█████████████████o3-...</li><li><a href="https://youtu.be/Nc3vIuPyQQ0?si=KKt7VD5I521H95-W"> - YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1341519869878734952)** (4 messages): 

> `重编译旧游戏, LLM 的 RL 训练, LLM4Decompile` 


- **通过反编译重编译游戏**：目前正在努力通过手动将汇编代码反编译为 C 语言来完全重编译旧游戏，这涉及大量的猜测以及与已知事实的对齐，如 [decomp.me 项目](https://decomp.me/scratch/hFS6m) 中所述。
   - 该过程需要不断调整反编译结果，直到它在相同设置下与原始汇编代码匹配。
- **RL 训练在反编译中的潜力**：有人建议可以使用 RL 来训练 LLM 自动进行反编译，将匹配百分比作为正确性的 Reward。
   - 讨论指出，相关的数据结构和相似函数的上下文可以增强训练，但目前尚不清楚是否有人尝试过这种方法。
- **GitHub 上的 LLM4Decompile 项目**：一个名为 [LLM4Decompile](https://github.com/albertan017/LLM4Decompile) 的 GitHub 项目专注于使用大语言模型反编译二进制代码，被提及为一个相关资源。
   - 然而，有人指出，据目前所知，该项目仅使用了 Supervised Fine-Tuning (SFT)，没有针对汇编匹配的 RL 训练实现。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://decomp.me/scratch/hFS6m>)">decomp.me</a>：未找到描述</li><li><a href="https://github.com/albertan017/LLM4Decompile">GitHub - albertan017/LLM4Decompile: Reverse Engineering: Decompiling Binary Code with Large Language Models</a>：逆向工程：使用大语言模型反编译二进制代码 - albertan017/LLM4Decompile
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1341104192538021898)** (13 messages🔥): 

> `斯坦福大学的 Post-training 演讲, 强化学习中的验证, 对理论论文的回应, 健康相关的挫折` 


- **斯坦福大学分享的 Post-training 见解**：@johnschulman2 和 @barret_zoph 最近在斯坦福大学关于 ChatGPT **post-training** 经验的演讲没有录音，但幻灯片可以在[这里](https://docs.google.com/presentation/d/11KWCKUORnPpVMSY6vXgBeFSWo7fJcuGQ9yuR6vC1pzE/edit?usp=sharing)查看。有人请求如果有录音的人请与其联系。
   - 演示文稿对 **post-training** 策略的关注引起了兴趣。
- **对理论论文的担忧**：*@johnatstarbucks* 对某些论文的理论性质表示怀疑，表示更倾向于实证结果。聊天参与者指出，尽管有所顾虑，但调查研究对该领域至关重要。
   - *@0x_paws* 保证第 6 节包含多个实验来证实理论主张。
- **影响参与度的健康问题**：*@natolambert* 提到感觉不舒服，无法在 Twitter 上庆祝最近的成功，这表示暂时的挫折。
   - 强调了影响他们在平台上日常活动的个人经历。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2502.12118">Scaling Test-Time Compute Without Verification or RL is Suboptimal</a>：尽管在扩展 Test-Time Compute 方面取得了实质性进展，但社区中关于如何扩展以实现持续且高效改进的争论仍在继续。存在巨大的...</li><li><a href="https://x.com/johnschulman2/status/1891539960743743756">John Schulman (@johnschulman2) 的推文</a>：@barret_zoph 和我最近在斯坦福大学做了一个关于 post-training 以及我们在 ChatGPT 上合作经验的演讲。不幸的是，演讲没有录音，但这里有幻灯片：https://docs.g...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1341441294131789825)** (14 messages🔥): 

> `科学哲学理论, Grok 3 Mini 公告, 推理模型发布, Hacker News 评论, Deep Thonk 按钮` 


- **科学哲学理论辩论**：一位成员指出，讨论围绕着对每种广为人知的理论的拒绝而展开，在同行之间传播碎片化的概念，而缺乏外部沟通。
   - 另一位成员评论说，这种行为具有一种有趣的“元（meta）”属性，凸显了现代哲学话语的复杂性。
- **Grok 3 Mini 已官宣但尚未发布**：大家意识到 **Grok 3 Mini** 虽然已经官宣但还未上线，正如这条 [推文](https://x.com/keirp1/status/1891955483251998984) 所指出的。
   - 成员们对推理模型所谓的“混乱发布”表示沮丧，并进一步思考了它们的性能。
- **关于 Hacker News 评论的讨论**：成员们分享了阅读 Hacker News 评论的复杂心情，尤其是在身体不适时。有人表示他们干脆不看，暗示参与此类讨论会带来压力。
   - 另一位成员提到，有些人可能认为直接参与 AI 讨论压力很大，这加剧了社区在话语方式上的分歧。
- **推理模型发布的推测**：关于推理模型的状态进行了讨论，有观点认为目前这些模型都没有真正发布，仅公开了那三个 benchmarks。
   - 这引发了对高级用户是否存在“deep thonk”功能的关注，暗示了平台的变现（monetization）方面。



**链接引用**: <a href="https://x.com/keirp1/status/1891955483251998984">来自 Keiran Paster (@keirp1) 的推文</a>: @natolambert @srush_nlp @TheShmanuel 我认为 mini 推理模型表现优于 R1 是反驳这一观点的有力证据。

  

---


### **Interconnects (Nathan Lambert) ▷ #[retort-podcast](https://discord.com/channels/1179127597926469703/1238601424452059197/1341542219521200350)** (2 messages): 

> `Torters 欢呼, 自行车架讨论` 


- **Torters 欢呼**：一位成员用 **Torters rejoice** 表达了兴奋，展示了社区中的一个重要时刻或公告。
   - 附带了一张可能包含庆祝活动相关细节或背景的图片。
- **自行车架的共同兴趣**：一位成员提到聊天中展示了同款自行车架，表明了共同的兴趣和经历。
   - 表情符号 **🔭👁️** 的选择暗示了对该讨论的一种轻松或幽默的态度。


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/)** (1 messages): 

gfabulous: 唉，看来我们现在都要用 Grok 了
  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1341440951066951682)** (14 messages🔥): 

> `Prompt Engineering, Perplexity vs ODR, Cursor Agent 工作流, 库中的破坏性变更, Vibecoding 效率` 


- **实验 Perplexity 的输出**：一位成员分享了使用 **Perplexity** 工具的实验，将其描述为与 Claude 项目绑定的 Prompt Engineering 系统。
   - 他们对其与 ODR 相比的性能表示好奇，并指出其在规定性输出格式方面的潜在局限性。
- **对 Perplexity 深度的批评**：讨论显示 **Perplexity** 往往停留在**表面且偏向列表式内容**，缺乏深度和推理能力。
   - 有人提到，该工具即使在不合适的情况下也经常生成表格、Python 代码和数学公式。
- **将 ODR 与 Cursor Agent 结合使用**：一位成员讲述了使用 **ODR** 诊断库中破坏性变更（breaking change）的成功经验，然后将其输入到 **Cursor Agent** 中。
   - 这种方法使 Cursor Agent 能够迅速解决问题，展示了工具的有效集成。
- **Vibecoding 作为一种工作流**：讨论中出现了 **vibecoding** 一词，强调了一种敏捷且高效的代码编写工作流。
   - 这被幽默地与著名的 AI 研究员 **Karpathy** 联系起来，强调了一种轻松的生产力提升方法。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1341099901463035967)** (86 条消息🔥🔥): 

> `DeepSeek 模型错误、本地 AI 功能、编程模型推荐、Whisper 使用与兼容性、LM Studio 更新问题` 


- **DeepSeek 模型错误**：一位用户遇到了模型加载问题，原因是 DeepSeek 模型中的超参数无效，具体表现为 *invalid n_rot: 128, expected 160*。
   - 错误详情包括相关的内存、GPU 和 OS 规格，表明可能存在硬件限制。
- **为智能功能设置本地 AI**：一位用户寻求帮助，希望设置本地 AI 用于提醒和智能家居集成，这涉及到 tool use 功能。
   - 另一位用户引导他们参考 LM Studio 的 API 文档以了解如何实现。
- **Python 编程的最佳 LLM 推荐**：社区成员讨论了用于 Python 编程的最佳 LLM，推荐了 *Qwen 2.5 coder* 和 *Llama 14B* 等选项。
   - 他们强调应寻找在编程任务中表现出色的模型，因为这些模型通常能提供高效的 Python 编码性能。
- **Whisper 模型使用与兼容性问题**：讨论围绕使用 Whisper 模型进行转录展开，并根据 CUDA 版本提出了一些设置建议。
   - 成员们指出在使用 Whisper 时需要特定的配置，特别是关于 CUDA 兼容性方面。
- **LM Studio 更新缓存问题**：一位用户遇到了尽管运行了新版本的 LM Studio，但系统中仍残留旧版 AppImage 构建的问题。
   - 社区建议包括重命名 .lmstudio 缓存目录以刷新安装，或按照开发者建议的步骤解决缓存冲突。



**提到的链接**：<a href="https://lmstudio.ai/docs/api/tools">Tool Use | LM Studio Docs</a>：使 LLM 能够与外部函数和 API 交互。

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1341135842932162633)** (128 条消息🔥🔥): 

> `3090 GPU 性能、DeepSeek 及其替代方案、使用多模型进行推理、396 与 4090 性能对比、AMD 的 Ryzen AI MAX` 


- **3090 的性能表现**：用户讨论了在 **3090** 上运行模型的可行性，观点认为通过调整 max tokens，它可以处理 **24B 参数模型**。
   - **13B 模型**被强调为性能的理想平衡点，并提到二手 **3090** 的价格在 **€650** 左右。
- **评估 DeepSeek 的效能**：DeepSeek R1 被认为是一个可靠的推理模型（reasoning model），与 OpenAI 的替代方案相比，一些用户将其作为处理技术任务的首选。
   - 其他讨论还强调了各种编程模型，如 **Mistral Nemo Instruct 12B**，并重点关注其在本地环境下的表现。
- **尝试双模型推理**：用户探索了同时利用较小模型进行推理和较大模型进行输出的概念，并分享了过去成功的项目经验。
   - 目前这种设置的实际应用需要手动编码，因为 LM Studio 尚未直接支持此功能。
- **AMD 新兴硬件格局**：提到了 AMD **Ryzen AI MAX** 的推出，强调了其潜在的性能实力，并与 Nvidia GPU 进行了对比。
   - 讨论还暗示了 **M4** 的优势以及即将推出的 **5070**，后者承诺在降低功耗的同时提供强劲性能。
- **集群与模型使用案例**：用户正在评估将旧款 **Tesla K80 GPU** 进行集群化以利用其高 VRAM 的可行性，尽管存在对能效比的担忧。
   - 分享了在集群设置中使用 **Exo** 的经验，涉及 PC 和 MacBook 的组合，但在同时加载模型时发现了一些问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://wccftech.com/amd-ryzen-ai-max-395-strix-halo-mini-pc-tested-powerful-apu-up-to-140w-power-128-gb-variable-memory-igpu/">AMD Ryzen AI MAX+ 395 &quot;Strix Halo&quot; Mini PC 测试：强大的 APU，功耗高达 140W，iGPU 可变内存高达 128 GB</a>：AMD 的 Ryzen AI MAX+ 395 &quot;Strix Halo&quot; 迷你主机曝光，展示了这款强大 APU 的卓越规格和丰富的 AI 能力。</li><li><a href="https://www.youtube.com/watch?v=v7HUud7IvAo">AMD CPU, Apple M4 Pro 性能 - Ryzen AI MAX 评测</a>：Ryzen AI MAX+ 395 和 Ryzen AI MAX 390 被认为是 Apple M4 和 Apple M4 Pro 的竞争对手，结合了高效率和相当惊人的性能……</li><li><a href="https://docs.google.com/spreadsheets/d/1IyT41xNOM1ynfzz1IO0hD-4v1f5KXB2CnOiwOTplKJ4/edit?gid=0#gid=0">GPU AI 对比</a>：暂无描述</li><li><a href="https://www.youtube.com/shorts/77rqmeLgfOs">新款 AMD 显卡……值得吗？</a>：暂无描述</li><li><a href="https://www.youtube.com/watch?v=VnKZe5SGveA"> - YouTube</a>：暂无描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1341123010387509281)** (30 条消息🔥): 

> `Cognitive Sound Production, Music Generation Challenges, Machine Learning Prodigies, lm_eval Code Issues, Autoregressive Image Generation` 


- **认知声音产生探讨**：一场关于人类认知任务与设备实现的物理声音产生之间区别的讨论展开，指出执行唱歌等任务涉及复杂的大脑操作。
   - 参与者思考了产生音乐的认知层面，认为其可能与机器的自动化能力有所不同。
- **音乐生成面临优化限制**：针对音乐生成中缺乏像国际象棋那样明确目标的问题提出了担忧，强调了尽管在频域操作方面取得了进展，但优化音乐生成模型仍具有复杂性。
   - 音乐中无法定义量化目标，这与传统游戏中的战略目标形成了对比。
- **关于稀疏自编码器（Sparse Autoencoder）输出的疑问**：一位用户提出了关于理解稀疏自编码器中潜藏激活（latent activations）的问题，对与 tokenizer 一致的解码结果表示困惑。
   - 成员们指出，索引对应于编码的维度，但在没有预定义特征名称的情况下解释这些维度可能具有挑战性。
- **lm_eval 遇到代码生成问题**：一位用户报告了 lm_eval v0.4.7 安装的问题，具体表现为尽管遵循了安装步骤，但缺少像 humaneval 这样的代码生成任务。
   - 这引发了对库中缺失任务潜在原因的询问，表明群组内存在故障排除需求。
- **创新的自回归图像生成**：讨论了图像生成的最新进展，重点关注应用自回归模型直接从 JPEG 字节生成图像。
   - 相关研究论文的链接强调了离散化图像的新方法，为当前模型如何适应视觉数据提供了见解。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.08459">JPEG-LM: LLMs as Image Generators with Canonical Codec Representations</a>: 最近在图像和视频生成方面的工作一直在采用自回归 LLM 架构，因为其具有通用性，并且可能易于集成到多模态系统中。应用自回归的关键在于...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://x.com/distributionat/status/1891662585830785114?s=46">thomas (@distributionat) 的推文</a>: 我将为我的 zerobench 评论浏览评估新的 evallive 线程</li><li><a href="https://huggingface.co/datasets/EleutherAI/persona">EleutherAI/persona · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1341092170572238973)** (77 条消息🔥🔥): 

> `DeepSeek v2 MoE Architecture, Platinum Benchmarks for LLMs, Model-guidance in Diffusion Models, Repetition Penalty in Creative Writing, SFT Memorizes and RL Generalizes`

- **理解 MoE 架构中的 Centroid**：一位用户询问了 DeepSeek v2 论文中 MoE 架构背景下 "centroid" 的含义，想知道它是否代表专家特化（specialization）的习得参数。
   - 另一位用户澄清说，它指的是路由向量权重（routing vector weight）。
- **Platinum Benchmarks：评估 LLM 的可靠性**：一篇新论文强调了通过 "platinum benchmarks"（白金基准）衡量大语言模型（LLMs）可靠性的重要性，以最大限度地减少评估中的标签错误。
   - 讨论集中在基准测试的潜在局限性上，例如每个问题仅从每个模型中提取一个样本。
- **用于 Diffusion 训练的创新 Model-guidance**：一篇论文介绍了 Model-guidance (MG) 作为训练 Diffusion 模型的新目标，旨在通过解决 Classifier-free guidance (CFG) 的缺陷来提高效率和输出质量。
   - 关于 MG 的数学实现出现了一些疑问，特别是报告错误是否可能导致对模型中使用的权重参数产生误解。
- **写作中 Repetition Penalty 的挑战**：用户讨论了创意写作场景中 repetition penalty（重复惩罚）的复杂性，一致认为这是一种旨在提高生成文本质量的统计偏差。
   - 挑战在于区分重复的好坏用法，因为两者在统计上可能看起来很相似，缺乏明确的区分信号。
- **SFT 过拟合担忧**：一位用户对 "SFT Memorizes, RL Generalizes" 的研究结果表示担忧，认为这暗示了在监督微调（SFT）过程中存在过拟合。
   - 围绕这些结果对语言模型在实际应用中可靠性的影响展开了辩论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2502.11089">Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention</a>：长上下文建模对于下一代语言模型至关重要，但标准 Attention 机制的高计算成本带来了显著的计算挑战。Sparse attention 提供...</li><li><a href="https://arxiv.org/abs/1806.02296">Regularization by Denoising: Clarifications and New Interpretations</a>：由 Romano、Elad 和 Milanfar 最近提出的 Regularization by Denoising (RED) 是一个强大的图像恢复框架，旨在最小化由...构建的显式正则化目标。</li><li><a href="https://arxiv.org/abs/2410.13821">Artificial Kuramoto Oscillatory Neurons</a>：神经科学和 AI 领域早就知道，神经元之间的“绑定”会导致一种竞争性学习，在这种学习中，表示被压缩以便表示更抽象的...</li><li><a href="https://arxiv.org/abs/2502.06785">DeepCrossAttention: Supercharging Transformer Residual Connections</a>：Transformer 网络在各个领域取得了显著成功，利用了包括残差连接在内的各种架构创新。然而，传统的残差连接...</li><li><a href="https://arxiv.org/abs/2502.12154">Diffusion Models without Classifier-free Guidance</a>：本文提出了 Model-guidance (MG)，这是一种训练 Diffusion 模型的新目标，它解决并移除了常用的 Classifier-free guidance (CFG)。我们的创新方法超越了...</li><li><a href="https://arxiv.org/abs/2502.03461">Do Large Language Model Benchmarks Test Reliability?</a>：在部署大语言模型（LLMs）时，确保这些模型不仅能力强而且可靠非常重要。许多基准测试被创建用来追踪 LLMs 日益增长的能力...</li><li><a href="https://gradientscience.org/platinum-benchmarks/">Do Large Language Model Benchmarks Test Reliability?</a>：来自 MadryLab 关于机器学习和优化的研究亮点和观点。</li><li><a href="http://platinum-bench.csail.mit.edu/">PlatinumBench</a>：未找到描述</li><li><a href="https://x.com/SonglinYang4/status/1891787029077278998">Songlin Yang (@SonglinYang4) 的推文</a>：🚀 宣布 ASAP：https://asap-seminar.github.io/！一个连接理论、算法和系统的全虚拟研讨会，旨在解决 Transformers 中的基本挑战。由 @simran_s_arora @X... 共同组织。</li><li><a href="https://github.com/tzco/Diffusion-wo-CFG/blob/e86a3002df0aa086c7630a1fe379e9fb9564c2ff/train.py#L378)">Diffusion-wo-CFG/train.py at e86a3002df0aa086c7630a1fe379e9fb9564c2ff · tzco/Diffusion-wo-CFG</a>：Diffusion Models Without Classifier-free Guidance 的官方实现 - tzco/Diffusion-wo-CFG
</li>
</ul>

</div>

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1341500977911566386)** (26 messages🔥): 

> `LLM scaling laws 术语, scaling laws 分类学, Pretraining 与 Post-training 计算量, AI 实验室的预算分配, 部署考量与计算` 


- **理解 LLM scaling laws 术语**：一位成员寻求关于 **LLM scaling laws** 相关术语的澄清，讨论了 **数据 (data)、模型大小 (model size)** 和 **测试时计算 (test-time compute)** 等因素。
   - *Stellaathena* 澄清说，90% 的 scaling law 讨论都集中在 pretraining（预训练）阶段，这意味着训练效率是关键。
- **建立 scaling laws 分类学的必要性**：一位用户表示有兴趣开发一种 **scaling laws** 的分类学，以更好地理解 **LLM 领域** 的进展。
   - 该咨询涉及按计算类型对 LLM 投资进行分类，但这种分类的实用性受到了质疑。
- **平衡 Pretraining 和 Post-training 成本**：*H_mmn* 指出，大型实验室现在在 **post-training**（训练后）上的投入可能与 **pre-training**（预训练）相当。
   - 这引发了关于此类成本是否包含 **数据获取 (data acquisition)** 等因素及其对 scaling laws 影响的讨论。
- **数据和微调透明度的挑战**：成员们承认，在模型训练中获取特定用途（如创意写作或对齐任务）的数据非常困难。
   - 尽管意识到缺乏公开数据，用户指出可以通过对有限公开数据的推断 (extrapolation) 来管理某些预期。
- **LLM Scaling 类别的核心要素**：*Stellaathena* 建议跟踪三个主要类别：**模型大小、数据**和**微调 (finetuning) 方法**，以便更好地概览 scaling laws。
   - 这种分类有助于消除围绕数据使用的模糊感，并有助于该领域内的模型比较。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1341400878342475799)** (3 messages): 

> `国际象棋战术的数据集结构化, 全新环境故障排除` 


- **故障排除环境问题**：一位成员建议尝试使用 **全新环境 (fresh environment)** 来解决他们遇到的一个未具体说明的问题。
   - 他们对问题表示不确定，但指出 **在他们那边一切运行正常**。
- **结构化国际象棋战术数据集**：一位成员对是否应将国际象棋战术数据集结构化为预测 **完整移动序列** 还是拆分为单步提示词表示怀疑。
   - 他们指出，虽然添加合法移动可以减少非法预测，但让模型对整个战术序列进行推理对于有效评估局面可能至关重要。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1341438408542654506)** (75 messages🔥🔥): 

> `GPU 性能对比, TP 通信重叠, 模型配置差异` 


- **理解 NeMo 和 NeoX 之间的 TPS**：一位用户报告说，在 80B A100 GPU 上使用 NeoX 达到了 **19-20K TPS**，而他们在 NeMo 中的协作努力使同等模型达到了 **25-26K TPS**。
   - 有人对 NeoX 中中间模型大小的效率及其对性能的影响表示担忧。
- **调查通信开销**：讨论了 NeMo 中潜在的 TP 通信重叠 (TP communication overlap)，这可以显著提高 PCIe 配置下的性能。
   - 分享了关于可能在 NeMo 中启用 `ub_tp_comm_overlap` 等设置以及探索 TE MLP 以缓解该问题的想法。
- **配置设置对性能的影响**：对比了 NeMo 和 NeoX 之间的配置，建议调整 allreduce 桶大小 (bucket sizes) 以匹配 NeMo，希望能提高 TPS。
   - 用户更倾向于使用更大的桶大小，因为这预计有助于 PCIe 通信中的 allgather 和 allreduce 操作。
- **Transformer Engine 集成挑战**：将 Transformer Engine 集成到 NeoX 的实验导致了在没有正确结果的情况下激活非 FP8 标志的问题。
   - 在初步受挫后，进一步解决集成问题的尝试被推迟。
- **迭代运行时间对比**：报告的迭代运行时间显示，模型操作在 NeMo 中每步为 **4.8 秒**，而 NeoX 中为 **6.2 秒**。
   - 这引发了对环境差异的质疑，特别是 NeoX 中未优化的设置是否阻碍了性能。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://gist.github.com/aflah02/edf6c71fb24edbbb82794317d8ef624c">pretrain_llama32_1b.py</a>: GitHub Gist：立即分享代码、笔记和代码片段。</li><li><a href="https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/pythia/6-9B.yml">gpt-neox/configs/pythia/6-9B.yml at olmo-support · aflah02/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - aflah02/gpt-neox</li><li><a href="https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/hubble/Speed_Exps/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_KV_Heads_4_Fusions_All_3_FA_Swiglu.yml">gpt-neox/configs/hubble/Speed_Exps/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_KV_Heads_4_Fusions_All_3_FA_Swiglu.yml at olmo-support · aflah02/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - aflah02/gpt-neox</li><li><a href="https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/hubble/Speed_Exps/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_KV_Heads_4_Fusions_All_3_FA_Swiglu.yml#L14C25-L14C30">gpt-neox/configs/hubble/Speed_Exps/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_KV_Heads_4_Fusions_All_3_FA_Swiglu.yml at olmo-support · aflah02/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - aflah02/gpt-neox</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/1ac9add83e2cdeec425a87c0a50ef2d278f3d5a1/megatron/model/utils.py#L30">gpt-neox/megatron/model/utils.py at 1ac9add83e2cdeec425a87c0a50ef2d278f3d5a1 · EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - EleutherAI/gpt-neox</li><li><a href="https://github.com/aflah02/gpt-neox/blob/olmo-support">GitHub - aflah02/gpt-neox at olmo-support</a>：基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - GitHub - aflah02/gpt-neox at olmo-support</li><li><a href="https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/hubble/Speed_Exps/6_7B_Baseline_No_Activation_Checkpointing_BS_4_GAS_16_GQA_KV_Heads_4_Fusions_3_FA_Swiglu.yml">gpt-neox/configs/hubble/Speed_Exps/6_7B_Baseline_No_Activation_Checkpointing_BS_4_GAS_16_GQA_KV_Heads_4_Fusions_3_FA_Swiglu.yml at olmo-support · aflah02/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - aflah02/gpt-neox</li><li><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/gpt/model/llama.py#L67">NeMo/nemo/collections/llm/gpt/model/llama.py at main · NVIDIA/NeMo</a>：为从事 Large Language Models、多模态和语音 AI（自动语音识别和文本转语音）的研究人员和开发人员构建的可扩展生成式 AI 框架 - NVIDIA/NeMo</li><li><a href="https://github.com/NVIDIA/NeMo/blob/0621272c2a9a760a71b234131f1997e87a265943/nemo/collections/nlp/models/language_modeling/megatron_base_model.py#L882">NeMo/nemo/collections/nlp/models/language_modeling/megatron_base_model.py at 0621272c2a9a760a71b234131f1997e87a265943 · NVIDIA/NeMo</a>：为从事 Large Language Models、多模态和语音 AI（自动语音识别和文本转语音）的研究人员和开发人员构建的可扩展生成式 AI 框架 - NVIDIA/NeMo</li><li><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/gpt/model/llama.py#L191">NeMo/nemo/collections/llm/gpt/model/llama.py at main · NVIDIA/NeMo</a>：为从事 Large Language Models、多模态和语音 AI（自动语音识别和文本转语音）的研究人员和开发人员构建的可扩展生成式 AI 框架 - NVIDIA/NeMo</li><li><a href="https://github.com/NVIDIA/NeMo/blob/0621272c2a9a760a71b234131f1997e87a265943/nemo/collections/nlp/models/language_modeling/megatron_base_model.py#L845.">NeMo/nemo/collections/nlp/models/language_modeling/megatron_base_model.py at 0621272c2a9a760a71b234131f1997e87a265943 · NVIDIA/NeMo</a>：为从事 Large Language Models、多模态和语音 AI（自动语音识别和文本转语音）的研究人员和开发人员构建的可扩展生成式 AI 框架 - NVIDIA/NeMo</li><li><a href="https://github.com/NVIDIA/NeMo/blob/0621272c2a9a760a71b234131f1997e87a265943/nemo/collections/nlp/models/language_modeling/megatron_base_model.py#L869.">NeMo/nemo/collections/nlp/models/language_modeling/megatron_base_model.py at 0621272c2a9a760a71b234131f1997e87a265943 · NVIDIA/NeMo</a>：为从事 Large Language Models、多模态和语音 AI（自动语音识别和文本转语音）的研究人员和开发人员构建的可扩展生成式 AI 框架 - NVIDIA/NeMo
</li>
</ul>

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1341092012275007500)** (72 条消息🔥🔥): 

> `Grok 3 发布, NURBS 与 AI, AI 模型对比, Grok 3 游戏工作室, LLM 精度问题` 


- **Grok 3 发布引发关注**：随着 Grok 3 的发布，讨论重点关注其极具潜力的能力，尽管早期评估表明它在某些方面可能落后于 OpenAI 的模型。
   - 随着产品的不断演进，社区成员对 Grok 3 的推理能力和 API 成本表现出浓厚兴趣。
- **NURBS 系列与 AI 的关系**：一位成员详细介绍了 NURBS 的优势，指出与传统方法相比，它们具有更好的平滑度和数据结构，并呼吁开发新的优化技术。
   - 对话转向了几何方法在 AI 开发中的更广泛影响，以及减少过拟合（overfitting）的潜力。
- **AI 模型对比引发质疑**：关于 Grok 与 OpenAI 模型之间对比公平性的辩论正在进行中，一些成员声称 Grok 的图表误导了其性能表现。
   - 对 “maj@k” 方法及其如何影响感知有效性的担忧浮出水面，引发了关于模型评估标准的讨论。
- **进军游戏领域的激动人心尝试**：Grok 3 的新品牌形象引起了关注，并宣布了成立游戏工作室的计划，这标志着 xAI 的逻辑扩张。
   - Elon Musk 对这一风险投资的参与增加了社区的兴趣，引发了对未来发展的猜测。
- **LLM 在精确回归（Precision Regression）方面面临挑战**：分享的一篇文章讨论了 LLM 在生成精确数值输出方面的局限性，概述了对各行业的潜在影响。
   - 这种对性能的持续讨论突显了在 AI 驱动的任务中实现高精度的持续挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://karthick.ai/blog/2025/LLM-Regression/"> Why Large Language Models Fail at Precision Regression | Karthick Panner Selvam </a>: 未找到描述</li><li><a href="https://fxtwitter.com/12exyz/status/1891723056931827959?t=auDqsxYvLMups5PYOtJGfQ&s=19">来自 Rex (@12exyz) 的推文</a>: 出于某种原因，他们在直播的图表中省略了 o3，所以我为你补全了数据</li><li><a href="https://x.com/elonmusk/status/1891388509191049307">来自 Elon Musk (@elonmusk) 的推文</a>: 是的。引用 Dima Zeniuk (@DimaZeniuk)：Elon Musk 的 xAI 将成立一个 AI 游戏工作室来制作游戏。
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1341094640153268264)** (31 条消息🔥): 

> `论文层级树, 即将进行的论文讨论, 对 Deepseek 论文的兴趣, 信息过滤讨论, 社区对论文评论的贡献` 


- **创建经典论文层级树**：有人建议为经典论文建立一个**层级树 (hierarchical tree)**，重点关注依赖关系和核心见解，并强调了*过滤信息*以避免噪音的重要性。
   - 社区表示希望利用其专业知识来识别**具有开创性和启发性的论文**，以便更好地进行知识共享。
- **对 Deepseek 论文的兴趣**：发起了对新款 **Deepseek 论文**读者的征集，提供了社区参与链接，并概述了作者身份和贡献。
   - 成员们表达了对围绕该论文组织讨论的兴趣，并强调需要有人牵头，因为目前尚未进行过讨论。
- **讨论时间与社区参与**：参与者讨论了在工作日安排**论文讨论**，时间灵活，并邀请贡献者和演讲者以确保更好的参与度。
   - 提到使用时间戳生成器来管理日程安排并协调提前通知，以增加参与人数。
- **应对论文阅读中的“哑弹”**：成员们表达了能够批判和识别**边缘化及可能低价值论文**的重要性，并指出有时没有哪篇论文是真正浪费时间的。
   - 大家达成共识，快速过滤可以改善整体阅读体验，并强调了增强讨论中时间管理的策略。
- **Discord 上的社区互动**：有人询问了 Discord 转换**跨时区时间戳**的能力，并分享了用于生成时间戳的实用工具。
   - 这场讨论让大家对那些促进讨论规划和增加参与度的社区工具表示赞赏。



**提到的链接**：<a href="https://arxiv.org/abs/2502.11089">Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention</a>：长上下文建模对于下一代语言模型至关重要，然而标准 Attention 机制的高计算成本带来了巨大的计算挑战。Sparse Attention 提供了...

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1341193761522192414)** (70 条消息🔥🔥): 

> `Larry Page 的感情状态、Sergey Brin 的离婚、Grok-3 演示、Deepsearch 产品发布、The Los Angeles Project 以及独角兽` 


- **Larry Page 的感情生活出现转折**：报告显示，**Larry Page** 在最近分手后重回单身市场，引发了媒体热议。
   - 一位成员幽默地询问 Page 是否会通过大声喊叫来回应他的处境。
- **Sergey Brin 备受关注的离婚案尘埃落定**：在 2022 年 1 月提出离婚申请后，**Sergey Brin** 已正式完成与 **Nicole Shanahan** 的离婚手续，此前曾有涉及 Elon Musk 的外遇指控。
   - 正式离婚于 5 月 26 日完成，凸显了在持续的媒体关注下动荡的个人生活。
- **Grok-3 现场演示引发讨论**：**Grok-3** 在现场演示中亮相，引发成员们讨论其与现有模型的性能对比以及它的 Tetris 机制。
   - 互动中反应不一，一些成员表示对该产品的演示感到“平淡无奇”（thoroughly whelmed）。
- **Deepsearch 的定价令人侧目**：尽管被描述为最先进的产品，但新款 **Deepsearch** 产品 **$40** 的定价引起了用户的怀疑，尤其是考虑到最近发布的类似产品大多免费。
   - 一位成员愤世嫉俗地观察到，考虑到竞争情况，这种激进的定价策略可能具有剥削性。
- **The Los Angeles Project 的宏伟目标**：**The Los Angeles Project** 旨在通过先进的基因编辑技术制造真正的独角兽，其宏大的主张令人侧目。
   - 然而，一些成员推测，在其运营中提及 AI 是否仅仅是一种营销策略，因为并没有招聘 AI 相关的职位。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.businessinsider.com/google-sergey-brin-divorced-wife-accused-affair-with-elon-musk-2023-9?op=1">Google 联合创始人 Sergey Brin 在妻子被指控与 Elon Musk 有染后，于今年悄然离婚</a>：法官拒绝了 Brin 因担心其知名度而提出的封存案件的请求。</li><li><a href="https://www.youtube.com/watch?v=b0XI-cbel1U">Elon 的 Grok-3 是新的 AI 之王吗？</a>：免费试用 Brilliant 30 天 https://brilliant.org/fireship，您还可以获得年度高级订阅 20% 的折扣。抢先预览 Elon Musk 的 Grok-3 ...</li><li><a href="https://en.mediamass.net/people/larry-page/break-up.html">Larry Page 再次单身？- 2025 年名人分手、分居与离婚 - Mediamass</a>：名人是否更容易离婚或分手？本周早些时候有新闻报道称，51 岁的 Larry Page 已与伴侣分手。Google 创始人真的再次单身了吗？</li><li><a href="https://www.piratewires.com/p/harnessing-the-breath-of-life">掌控生命的气息</a>：一家名为 The Los Angeles Project 的基因编辑初创公司将如何创造真正的、字面意义上的独角兽（以及更多）</li><li><a href="https://building.life/">LAP</a>：未找到描述</li><li><a href="https://x.com/xai/status/1891699715298730482?t=cFbu7r78njWASjCsXt7-Hw&s=19">来自 xAI (@xai) 的推文</a>：https://x.com/i/broadcasts/1gqGvjeBljOGB
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1341106634902077451)** (168 条消息🔥🔥): 

> `Xformers 问题，InvokeAI vs ComfyUI，Stable Diffusion 更新担忧，动漫性别分类，打印机可用性挫败感` 


- **Xformers 引发 GPU 困扰**：用户讨论了 *xformers* 的问题，特别是它默认进入 CPU 模式并需要特定版本的 PyTorch，还有成员提到直接忽略警告信息也是可行的。
   - 建议一位用户进行全新安装并在命令行中添加 `--xformers`，希望能解决该问题。
- **InvokeAI 成为初学者的首选 UI 选择**：许多用户建议将 *InvokeAI* 作为新手更直观的 UI 选择，尽管他们自己觉得 ComfyUI 的功能更强大。
   - 社区共识是，虽然 InvokeAI 很简单，但像 ComfyUI 这样底层系统的复杂性可能会让新手感到不知所措。
- **对 Stable Diffusion 停滞的担忧**：用户对 *Stable Diffusion* 生态系统缺乏更新表示担忧，特别是 A1111 不再支持 SD 3.5，导致用户感到沮丧。
   - 用户注意到过时的指南和分支可能与新技术不兼容，从而导致困惑。
- **动漫性别分类探索**：一位用户寻求建议，关于如何使用在 Hugging Face 上找到的特定 *动漫性别分类器*（anime gender classifier）来分离男性和女性动漫 bbox 以进行局部重绘（inpainting）。
   - 据报道，该分类方法很有前景，但需要具备将其集成到 ComfyUI 现有工作流中的专业知识。
- **对打印机可用性的挫败感**：一位 IT 工作者幽默地分享了处理故障打印机的经历，强调了即使是清晰的指令也可能被误解。
   - 对话演变为关于对简单标记的常见误解以及对更有效沟通的需求的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Kwai-Kolors/Kolors-Virtual-Try-On">Kolors Virtual Try-On - a Hugging Face Space by Kwai-Kolors</a>: 未找到描述</li><li><a href="https://huggingface.co/DOFOFFICIAL/animeGender-dvgg-0.8">DOFOFFICIAL/animeGender-dvgg-0.8 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/Zheng-">zheng-</a>: GitHub 是 zheng- 构建软件的地方。</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases">Releases · AUTOMATIC1111/stable-diffusion-webui</a>: Stable Diffusion web UI。通过在 GitHub 上创建一个账号来为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1341101877995241542)** (11 条消息🔥): 

> `Torch Compile, PyTorch Inductor, 机器学习进展` 


- **理解 Torch Compile 机制**：一位成员解释说，当使用 `torch.compile()` 时，PyTorch 中的机器学习模型会转换为 Python 字节码，由 TorchDynamo 进行分析，从而生成用于 GPU kernel 调用的 FX 图（FX graph）。
   - 他们表达了对处理编译过程中图中断（graph breaks）的兴趣，并承认有必要对此进行进一步探索。
- **讨论 Inductor 内部原理**：分享了一个指向[详细 PDF](https://github.com/pytorch/workshops/blob/master/ASPLOS_2024/inductor.pdf)的链接，该文件专注于 PyTorch Inductor 的内部结构，提供了深入的信息。
   - 成员们对该资源表示感谢，并计划跟进提供的细节。
- **对机器学习复杂性的反思**：一位成员分享了一篇讨论 **Torch Compile** 的博客文章链接，强调了现代机器学习工具与早期框架（如 **Caffe**）相比的复杂性。
   - 该文章反思了当今工具的用户友好性，同时也认识到它们背后隐藏的复杂性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://themlsurgeon.substack.com/p/dissecting-torchcompile-surgical">Dissecting torch.compile: Surgical Precision in PyTorch Optimization</a>: 你可以通过此链接查看该博客文章的 GitHub 仓库</li><li><a href="https://github.com/pytorch/workshops/blob/master/ASPLOS_2024/inductor.pdf">workshops/ASPLOS_2024/inductor.pdf at master · pytorch/workshops</a>: 这是一个包含所有研讨会相关材料的仓库。 - pytorch/workshops
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1341116960829603973)** (3 条消息): 

> `Triton 与 CUDA 中的 Profiling，Bank Conflicts 导致的性能问题，Triton Kernel 中的低精度输入，PyTorch 中的设备属性` 


- **Profiling 见解：CUDA vs Triton**：一位成员建议 **Cuda driver** 知道确切的数值，这表明微基准测试（microbenchmarking）仅适用于未公开的属性（如指令缓存大小），而 L1 缓存应该可以从计算能力（compute capability）中获知。
   - 他们分享了一个 [PyTorch 查询](https://pytorch.org/docs/stable/cuda.html) 来获取包括 **L2 cache size** 在内的设备属性，并指出关于 “MB” 代表 “MiB” 的潜在混淆。
- **调试 Triton Kernel 性能下降**：一位成员正在处理 Triton Kernel 中因使用 **float8** 等 **低精度输入** 触发的性能下降问题，指出这与 Shared Memory 访问的 Bank Conflicts 增加有关。
   - 尽管使用了 **NCU** 进行 Profiling，他们注意到在 Triton 中解决 Bank Conflicts 的资源匮乏，并强调了其与 CUDA 细粒度控制相比的差异。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1341239338524086292)** (17 条消息🔥): 

> `优化 CUDA 内存传输，详解全局内存合并（Global Memory Coalescing），CUDA Express 安装程序问题，数据复制中的细粒度控制` 


- **优化 CUDA 内存传输**：为了提高在 CUDA 中传输大型常量向量的速度，建议将 [`cudaMemcpyAsync`](https://docs.nvidia.com/cuda/cudaMemcpyAsync.html) 与 `cudaMemcpyDeviceToDevice` 配合使用以获得更好的性能。
   - 成员们讨论了 `cudaMemcpyAsync` 的优势，但指出对细粒度控制的需求可能需要自定义解决方案。
- **详解全局内存合并（Global Memory Coalescing）**：一位成员询问了 CUDA 中全局内存合并的实现，特别是提供的代码如何在矩阵乘法中实现这一点。
   - 另一位成员引用了图表和示例代码，澄清了理解线程如何映射到元素是利用合并内存访问的关键。
- **CUDA Express 安装程序问题**：许多用户（包括一位报告 CUDA Express 安装程序问题的用户）在安装过程中遇到卡死，特别是在 Nsight 和 Visual Studio 环节。
   - 对话表明这些问题可能在多种配置中普遍存在，尤其是在 Windows 系统上。
- **数据复制中的细粒度控制**：在 A100 GPU 上需要更精细的数据复制控制的情况下，有人建议使用 [`cub::DeviceCopy::Batched`](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceCopy.html) 作为解决方案。
   - 这种方法可能会提供 `cudaMemcpyDeviceToDevice` 等默认方式所不具备的必要粒度。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1341328518784745494)** (4 条消息): 

> `Triton 3.2.0 问题，CUDA Kernel 编译优化` 


- **Triton 3.2.0 在 print() 时抛出 TypeError**：在 PyTorch 的 Triton 3.2.0 包中，将 `TRITON_INTERPRET=1` 与 `print()` 配合使用会导致 **TypeError**，提示 `kernel()` 收到意外的关键字参数 'num_buffers_warp_spec'。即使在配置中设置了 `num_buffers_warp_spec`，该错误仍然存在。
- **加速 CUDA Kernel 编译**：一位用户对通过 cpp 扩展添加新的 CUDA Kernel 需要显著的编译时间感到沮丧，这与单个 .cu 文件的即时编译不同。
   - 另一位成员建议将代码分为两个文件，一个用于 Kernel，另一个用于 Torch 扩展，以避免重新编译 PyTorch 部分，从而可能加快开发速度。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 条消息): 

andreaskoepf: DS 目前统治着该领域：https://arxiv.org/abs/2502.11089
  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1341404853787820116)** (4 条消息): 

> `Gompertz Linear Unit (GoLU), Native Sparse Attention, Self-gated activation functions` 


- **介绍 Gompertz Linear Unit (GoLU)**：推出了一种名为 **GoLU** 的新型激活函数，定义为 $\mathrm{GoLU}(x) = x \, \mathrm{Gompertz}(x)$，其中 $\mathrm{Gompertz}(x) = e^{-e^{-x}}$，旨在增强训练动态。
   - 与 **GELU** 和 **Swish** 相比，**GoLU** 激活函数在保持稳健梯度流的同时，减少了潜空间（latent space）中的方差。
- **Native Sparse Attention 技术**：讨论围绕关于 **Native Sparse Attention** 的论文展开，指出其硬件对齐和可训练性可能会彻底改变模型效率。
   - 成员们表现出极大的兴奋，其中一人评论道：*“这个结果难道不疯狂吗？”* 并考虑在另一个频道分享该信息。
- **自门控激活函数（Self-gated Activation Functions）受到关注**：强调了激活函数的演变，特别是 **GELU** 和 **Swish** 等**自门控激活函数**的兴起，它们能够稳定梯度流并最大限度地减少神经元失活。
   - 这些函数被视为传统 **ReLU** 方法的替代方案，而 **ReLU** 存在*神经元死亡问题（dying neuron problem）*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2502.11089">Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention</a>：长上下文建模对于下一代语言模型至关重要，但标准 Attention 机制的高计算成本带来了巨大的计算挑战。Sparse attention 提供了一种...</li><li><a href="https://arxiv.org/abs/2502.03654v1">Gompertz Linear Units: Leveraging Asymmetry for Enhanced Learning Dynamics</a>：激活函数是深度学习架构的基础元素，因为它们显著影响训练动态。ReLU 虽然被广泛使用，但容易出现神经元死亡问题，这已经...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1341366325448151050)** (2 条消息): 

> `GPU kernel programming internships, Senior AI Engineer position at T-Systems` 


- **Param 寻求 GPU 和 AI 领域的实习**：来自印度的计算机科学本科生 **Param Thakkar** 正在寻找 **GPU kernel 编程**、**Large Language Models** 和 **Deep Learning** 方面的实习机会。
   - 他拥有在 **C++ 和 Julia 中使用 CUDA** 的经验，以及包括用于图像生成的**生成对抗网络（Generative Adversarial Networks）**在内的多个项目经验。
- **T-Systems 招聘高级 AI 工程师**：来自 T-Systems 的 **Pavol** 宣布了在 **欧盟/西班牙** 的高级 AI 工程师职位空缺，重点是改进微调基础设施和优化多 **H100** 部署。
   - 有兴趣的候选人可以在[此处](https://www.linkedin.com/jobs/view/4152771205)申请，或直接联系 Pavol 获取更多信息。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1341246375253246045)** (1 条消息): 

> `Optimizing memory access, Global memory operations` 


- **加速全局内存读取**：一位成员询问如何提高从全局内存（global memory）读取大型常量向量并将其发送到全局内存中另一个位置时的性能。
   - 他们正在寻求可能提高**内存操作速度**的代码示例或策略。
- **探索内存传输效率**：另一位成员建议使用 **coalescing**（合并访问）等方法来优化内存传输，这可以显著降低全局内存操作中的延迟。
   - 引用原话指出，在处理大型数据集时，这种方法可以带来*吞吐量的巨大提升*。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 条消息): 

iron_bound: https://www.amd.com/en/products/software/rocm/application-developer-certificate.html
  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1341138200260776008)** (7 条消息): 

> `PyTorch 中的 HIP Kernel，ROCm 安装问题，AMD 内核驱动性能，ROCm 在 iGPU 上的兼容性` 


- **简化 HIP Kernel 的使用**：一位成员询问了在 **PyTorch** 中使用 **HIP Kernel** 的方法，提到他们目前通过 ctypes 导入，但怀疑是否有更简便的方法。
   - 社区呼吁更高效的方法，这突显了社区在该领域的持续探索。
- **Strix Point 上的 ROCm 安装困扰**：一位成员在 Ubuntu 24.04 的 **Strix Point 平台**上安装 **ROCm** 时遇到问题，安装后图形界面无法启动。
   - 他们怀疑这是否意味着新内核驱动程序与 iGPU 存在兼容性问题，并寻求可行的解决方案。
- **对 AMD 内核驱动性能的失望**：成员们对 **AMD 内核驱动**的状态表示担忧，特别是对于非 MI 系列的 GPU，因为它在 iGPU 上的表现似乎不足。
   - 一位成员建议移除 **amdgpu-dkms 软件包**并依赖上游驱动（upstream driver）以获得更好的稳定性。
- **ROCm 在 iGPU 上的测试有限**：讨论强调 **ROCm** 组件在 iGPU 上未经过充分测试，默认设置是为其他架构编译的。
   - 虽然在 iGPU 上使用 ROCm 可能会面临挑战，但出于实验目的尝试 **HIP** 是可行的。
- **社区支持与建议**：成员们针对安装困扰（特别是图形界面问题）提供了支持和建议。
   - 回复包括故障排除技巧，例如在适用情况下禁用独立 GPU，促进了社区内的协作。


  

---


### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1341239876942565446)** (4 条消息): 

> `ExecuTorch, LLM 优化, Ethos U, 自我推广政策` 


- **ExecuTorch 为 int4 优化 LLM**：一位成员分享称 **ExecuTorch** 已经针对 **int4** 优化了 **LLM**，目前正在开发 **Ethos U** 支持。
   - 这是在社区内推广其工作的背景下提到的。
- **关于自我推广规则的提醒**：另一位成员提醒，通常不允许在特定频道之外发布自我推广帖子。
   - 他们建议将链接替换为更相关的 **Ethos** 代码或优化指令，而不是推广 Discord 频道。
- **对反馈的确认**：原帖作者接受了反馈，并确认已采取行动，表示：**“已完成。谢谢提醒。”**
   - 这体现了在处理社区帖子指南方面的合作精神。


  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1341211361170165780)** (1 条消息): 

> `模拟金属针画玩具, transformers.js, 深度估计模型` 


- **黑客周末让模拟金属针画玩具成真**：一位成员展示了一个**模拟金属针画玩具**，它使用实时摄像头配合在浏览器中运行的**深度估计模型**，可以通过其个人简介中的链接进行体验。
   - 他们强调了周末与 Claude 合作的乐趣，突出了该项目带来的创造力。
- **利用 transformers.js 的精彩演示**：该项目重点使用了 **transformers.js**，这是一个由社区部分成员开发的库。
   - 这引起了其他人的兴趣，他们对该库在实时应用中的实现表示好奇。



**提到的链接**：<a href="https://x.com/vvvincent_c/status/1890461227651940729">Vincent (@vvvincent_c) 的推文</a>：使用实时摄像头 + 在浏览器中运行的深度估计模型的模拟金属针画玩具 🎨✨我和 Claude 周末玩得很开心！！链接在简介中，快去亲自尝试吧！

  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1341176086985310309)** (15 条消息🔥): 

> `Dynasor, ML Systems Research, Simulated Metal Pin Toy, CUDA Optimization Techniques, HQQ Support in VLLM` 


- **Dynasor 大幅降低推理系统成本**：**Dynasor** 在无需模型训练的情况下，将推理系统成本降低了高达 **80%**，展示了令人印象深刻的 token 效率。
   - 该工具允许用户探测模型，并利用确定性（certainty）来停止不必要的推理过程，已在 [在线演示](https://hao-ai-lab.github.io/demo/dynasor-cot) 中展示。
- **对 ML Systems Research 讲座的兴趣**：一位成员表示有兴趣邀请来自 **Hao lab** 的人员就 ML Systems Research 在推理模型中的应用进行讲座。
   - 社区渴望了解更多关于该主题及其实际意义的信息。
- **创意网络摄像头玩具项目**：一位成员展示了一个使用实时网络摄像头和在浏览器中运行的深度估计模型创建的**模拟金属针玩具**（simulated metal pin toy），强调了开发该项目的乐趣。
   - 他们通过在个人简介中提供链接，鼓励其他人尝试。
- **关于 CUDA 优化的深度博客**：一位成员分享了一篇详细的博客，涵盖了 **CUDA 优化技术**，重点是迭代优化 layer normalization kernel，并强调了几种性能改进策略。
   - 反馈鼓励了围绕向量化实现正确性的讨论，从而带来了来自社区的有价值见解。
- **VLLM 中增强的 HQQ 支持**：**vllm** 中对 **hqq** 的增强支持允许运行更低位（lower bit）的模型，并通过 GemLite 或 PyTorch 后端对几乎任何模型进行即时量化。
   - 随着新版本的发布，还宣布了极具吸引力的补丁功能，有望在 **vllm** 的各个分支中实现更广泛的兼容性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aryagxr.com/blogs/cuda-optimizing-layernorm">Optimizing a Layer Normalization Kernel with CUDA: a Worklog</a>: 暂无描述</li><li><a href="https://x.com/Mobius_Labs/status/1891888285544333607">来自 Mobius Labs (@Mobius_Labs) 的推文</a>: hqq ( https://github.com/mobiusml/hqq/releases/tag/0.2.3… ) 和 gemlite ( https://github.com/mobiusml/gemlite/releases/tag/0.4.2… ) 发布了新版本！最令人兴奋的更新是我们带来了...</li><li><a href="https://x.com/vvvincent_c/status/1890461227651940729">来自 Vincent (@vvvincent_c) 的推文</a>: 使用实时网络摄像头 + 在浏览器中运行的深度估计模型实现的模拟金属针玩具 🎨✨ Claude 和我在周末玩得很开心！！点击个人简介中的链接亲自尝试吧！</li><li><a href="https://x.com/haoailab/status/1891581634773651763">来自 Hao AI Lab (@haoailab) 的推文</a>: 推理模型经常在自我怀疑中浪费 token。Dynasor 帮你节省高达 81% 的 token 即可得出正确答案！🧠✂️- 在中途探测模型以获取确定性 - 使用确定性停止推理...</li><li><a href="https://github.com/hao-ai-lab/Dynasor">GitHub - hao-ai-lab/Dynasor: Simple extension on vLLM to help you speed up reasoning model without training.</a>: vLLM 的简单扩展，可帮助你在无需训练的情况下加速推理模型。- hao-ai-lab/Dynasor
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1341254684391772171)** (3 messages): 

> `KernelBench paper, GPU kernel generation, Performance engineering, Kernel fusion, Productivity tools in coding` 


- **KernelBench 论文发布**：**KernelBench** 论文现已发布，重点关注通过语言模型为高效的 **PyTorch ML workloads** 实现 **GPU Kernel 生成自动化**。它引入了一种新型指标 **fast_p**，用于从正确性和性能两个维度评估生成的 Kernel。
   - 你可以在[此处](https://arxiv.org/abs/2502.10517)查看论文，并查看 [PDF 版本](https://arxiv.org/pdf/2502.10517)以获取直接见解。
- **性能工程师对 KernelBench 的期待**：一位性能工程师对 **KernelBench** 框架的潜力表示兴奋，认为它可以作为复杂 **Kernel fusion** 的起点。他们希望通过微调生成的 Kernel 来提高**性能和正确性**。
   - *“这或许可以为复杂的 Kernel fusion 提供一个起点，然后我可以针对性能/正确性进行优化。”*
- **编译器开发中的生产力工具**：讨论中有一种观点认为，即使模型不能生成最快的代码，它们也能显著改善工作流。一位成员强调，编译器应该被视为一种**生产力工具**。
   - *“在我看来，编译器本质上是一个生产力工具。”*



**提到的链接**：<a href="https://arxiv.org/abs/2502.10517">KernelBench: Can LLMs Write Efficient GPU Kernels?</a>：高效的 GPU Kernel 对于构建高性能机器学习架构至关重要，但编写它们是一项耗时且需要大量专业知识的挑战；因此，我们探索使用...

  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1341474790275547260)** (2 messages): 

> `SO-ARM100 Assembly, 3D Printer Size Constraints` 


- **咨询 SO-ARM100 组装经验**：一位成员询问是否有人尝试过组装 [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) 并且有良好的体验。
   - 他们强调 [GitHub 仓库](https://github.com/TheRobotStudio/SO-ARM100)是潜在贡献者的宝贵资源。
- **对 3D 打印机尺寸的担忧**：另一位成员怀疑他们的 **3D 打印机**可能太小，无法打印 **SO-ARM100** 的部件。
   - 这引发了关于打印机尺寸与项目需求兼容性的考量。



**提到的链接**：<a href="https://github.com/TheRobotStudio/SO-ARM100">GitHub - TheRobotStudio/SO-ARM100: Standard Open Arm 100</a>：Standard Open Arm 100。欢迎在 GitHub 上为 TheRobotStudio/SO-ARM100 的开发做出贡献。

  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1341102368372031569)** (49 messages🔥): 

> `vLLM nightly, RL curricula development, ExploreToM project, CodeI/O dataset, Issue creation and collaboration` 


- **实验 vLLM nightly**：一名成员正在启动一个节点进行实验，同时使用 **vLLM nightly** 版本 (`pip3 install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly`)。
   - 该方法旨在探索与 **veRL** 相关的功能。
- **协作开发 RL 课程**：呼吁协助开发 **RL 课程**（RL curricula），重点关注训练评估的手动和动态模式。
   - 讨论集中在选择能够通过指定参数促进难度缩放的数据集。
- **介绍来自 Facebook 的 ExploreToM**：提到了一个新项目 [ExploreToM](https://github.com/facebookresearch/ExploreToM)，该项目与心理理论（Theory of Mind）任务中的探索相关。
   - 该项目旨在进一步探索模型在多智能体交互（multi-agent interactions）中的能力。
- **DeepSeek 的 CodeI/O 论文讨论**：一名成员分享了一篇关于 **CodeI/O** 的论文，该论文讨论了根据代码预测输入和输出，强调了对有价值数据集的需求。
   - 在 GitHub 上创建了一个 Issue 来处理这个数据集，强调了将工作分解为更小、更易于管理的任务的重要性。
- **访问评估表**：成员们讨论了 Google doc 评估表的访问权限，特别提到了 **mini_sudoku** 和 **family_relationships** 等数据集的改进。
   - 一名用户获得了访问权限，因为他们计划为评估和更新做出贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2502.08859">EnigmaEval: A Benchmark of Long Multimodal Reasoning Challenges</a>：随着语言模型掌握了现有的推理基准，我们需要新的挑战来评估它们的认知前沿。解谜活动是具有挑战性的多模态问题的丰富资源库...</li><li><a href="https://arxiv.org/abs/2502.07316">CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction</a>：推理是大语言模型（LLM）的一项基本能力。虽然之前的研究主要集中在增强数学或代码生成等狭窄技能上，但提高在许多其他领域的性能...</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/160">Add CODEI/O sampled subset dataset · Issue #160 · open-thought/reasoning-gym</a>：DeepSeek 发布了 CodeI/O：通过代码输入输出预测压缩推理模式。任务：“在完全使用自然语言的情况下，给定代码和测试用例预测输入/输出”。除了原有的...</li><li><a href="https://x.com/lefthanddraft/status/1891732021434548347">Wyatt walls (@lefthanddraft) 的推文</a>：嗯。我想我需要一个推理器来处理这个</li><li><a href="https://x.com/lefthanddraft/status/1891737335554855205?s=46&t=E50tvry4ancj_GB5agsQ7w">Wyatt walls (@lefthanddraft) 的推文</a>：Grok 3（非推理版）无法识别两个格式不同的超大数字是相同的</li><li><a href="https://github.com/facebookresearch/ExploreToM">GitHub - facebookresearch/ExploreToM: Code for ExploreTom</a>：ExploreTom 的代码。通过在 GitHub 上创建账户为 facebookresearch/ExploreToM 开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/156">Add a dataset with tasks to compare formatted numbers · Issue #156 · open-thought/reasoning-gym</a>：格式可能包括：普通：123456.789 f'{x:f}' 英语：123,456.789 f'{x:,}' 百分比：12345678.9% f'{x:%}' & f'{x:,%}' 科学计数法：1.234568e+05 f'{x:e}' ...</li><li><a href="https://github.com/open-thought/reasoning-gym/tree/main/reasoning_gym/coaching">reasoning-gym/reasoning_gym/coaching at main · open-thought/reasoning-gym</a>：程序化推理数据集。通过在 GitHub 上创建账户为 open-thought/reasoning-gym 开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/71">Curriculum Refactor by EduardDurech · Pull Request #71 · open-thought/reasoning-gym</a>：未找到描述
</li>
</ul>

</div>

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1341095692097032305)** (84 条消息🔥🔥): 

> `Grok-3 性能、开源模型讨论、自回归图像模型、AI 模型训练技术、社区对 AI 模型的反馈` 


- **Grok-3 表现出不同程度的审查水平**：用户注意到，最初被认为无审查的 Grok-3 在不同使用场景下（如 lmarena）呈现出不同程度的审查。
   - *有些人感到惊喜*，而另一些人则希望更深入地访问其所谓的原始输出。
- **讨论开源 AI 的未来**：社区认为像 Anthropic 这样的公司应该开源旧模型以赢得声誉，特别是考虑到最近的市场变化。
   - 普遍共识是旧模型不具备显著的竞争风险，因此共享有利于社区成长。
- **AI 建模的创新与技术**：Grok-3 的训练方法表明许多技术正在演进，特别是围绕来自人类反馈的强化学习 (RLHF)。
   - 参与者讨论了训练分类器和自回归模型以提高 AI 对话和响应质量的潜力。
- **对 XAI 方法的好奇**：人们对 XAI 是否比 OpenAI 和 DeepSeek 更有创新感到好奇，特别是关于他们似乎很独特的自回归图像模型。
   - 社区成员渴望了解 XAI 是否会揭示让他们在竞争格局中脱颖而出的新方法论。
- **社区对 Hermes-4 的期待**：基于目前关于 AI 改进和能力的讨论，对 Hermes-4 的期待正在升温，用户表达了对其发布的渴望。
   - 诸如 *'Hermes-4 is going to be so fucking lit!'* 之类的情绪凸显了社区对 AI 技术进步的期待。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/DamascusGit/err_err">GitHub - DamascusGit/err_err</a>: 通过在 GitHub 上创建账号来为 DamascusGit/err_err 的开发做出贡献。</li><li><a href="https://github.com/Account-Link/err_err">GitHub - DamascusGit/err_err</a>: 通过在 GitHub 上创建账号来为 DamascusGit/err_err 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1341152375808983192)** (11 条消息🔥): 

> `Hermes 3 审查争议、Deephermes 使用、Grok 3 印象、Token 性能问题` 


- **Hermes 3 的审查悖论**：尽管宣传为**无审查**，但用户报告称 **Hermes 3** 因需要正确的系统提示词 (system prompts) 而拒绝回答某些问题。
   - *一位成员推测，为了实现预期功能，通过系统提示词进行引导是必要的。*
- **Deephermes 使用见解**：讨论显示 **Deephermes** 被认为**审查较少**，成员建议他人使用 **LMStudio** 以获得最佳性能。
   - *关于 Deephermes 存在 Bug 的说法遭到了反驳，认为不存在 Bug，关键在于正确使用。*
- **对 Grok 3 的初步反应**：关于 **Grok 3** 的观点认为其表现与 **o3 full** 相当，表明用户对其印象良好。
   - *一位成员分享了对 Grok 3 具体包含什么的各种好奇，反映了普遍的兴趣。*
- **对 Token 性能延迟的担忧**：一位用户对 **7789 tokens** 的限制导致处理时间过长表示担忧，质疑整体效率。
   - *另一位成员指出它并非定位为推理模型，暗示了预期与现实之间的差距。*


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1341329553250848860)** (3 messages): 

> `SWE-Lancer Benchmark, Upwork Engineering Tasks, Model Performance Evaluation` 


- **推出针对自由职业任务的 SWE-Lancer 基准测试**：新的 [SWE-Lancer](https://arxiv.org/abs/2502.12115) 基准测试包含超过 **1,400 个自由职业软件工程任务**，总价值达 **100 万美元**，涵盖了从微小的 bug 修复到重大的功能实现。
   - *独立任务*由资深工程师评分，而*管理决策*则根据原始工程经理的决策进行评估，这凸显了提升模型性能的需求，因为 Frontier Models 目前仍难以解决大部分任务。
- **开源 SWE-Lancer 以供未来研究**：研究人员可以通过 [公开评估集](https://github.com/openai/SWELancer-Benchmark) 探索 SWE-Lancer，该项目提供了一个统一的 Docker 镜像以方便测试和改进。
   - 通过将模型性能与**货币价值**挂钩，该基准测试旨在推动软件工程任务领域更有效的研究。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.12115">SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?</a>: 我们推出了 SWE-Lancer，这是一个包含来自 Upwork 的 1,400 多个自由职业软件工程任务的基准测试，在真实世界的支付中总价值达 100 万美元。SWE-Lancer 涵盖了独立工程...</li><li><a href="https://arxiv.org/abs/2502.11089?s=09">Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention</a>: 长上下文建模对于下一代语言模型至关重要，但标准 Attention 机制的高计算成本带来了显著的计算挑战。Sparse Attention 提供...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1341237404866056233)** (2 messages): 

> `Alignment faking in LLMs, Eagles Super Bowl Predictions, Open-source LLMs performance` 


- **探索 LLM 中的 Alignment Faking**：一段名为 **'Alignment Faking in Large Language Models'** 的 [YouTube 视频](https://www.youtube.com/watch?v=9eXV64O2Xp8) 讨论了某些个体可能表现出认同某些价值观，但实际上只是在伪装。
   - 这种行为被比作 AI 模型在解释 Alignment（对齐）时面临的挑战。
- **开源机器人预测老鹰队将赢得超级碗**：一个由开源 LLM 驱动的 [竞猜机器人 (pick-em's bot)](https://github.com/stevekrenzel/pick-ems) 预测**老鹰队 (The Eagles)** 将赢得超级碗，其表现优于 ESPN 2024 年竞赛中 **94.5%** 的玩家。
   - 该机器人表示，“老鹰队是逻辑上的选择”，并强调了一种利用结构化输出进行推理的新颖方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://bits.logic.inc/p/the-eagles-will-win-super-bowl-lix">Predicting the Super Bowl with LLMs</a>: 使用 LLM 挑选 NFL 获胜者，表现优于 94.5% 的人类。</li><li><a href="https://www.youtube.com/watch?v=9eXV64O2Xp8">Alignment faking in large language models</a>: 我们大多数人都遇到过这样的情况：某人看起来认同我们的观点或价值观，但实际上只是在伪装——这种行为我们可能会称之为...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1341329553250848860)** (3 messages): 

> `SWE-Lancer Benchmark, Real-world Software Engineering Tasks` 


- **SWE-Lancer 为自由职业任务设定了新基准**：SWE-Lancer 基准测试引入了来自 Upwork 的 **1,400 多个自由职业软件工程任务**，在真实世界的总报酬价值达 **100 万美元**，涵盖了独立任务和管理任务。
   - *模型在任务完成方面表现挣扎*：评估显示，尽管由经验丰富的工程师进行严格评分，前沿模型仍无法解决大部分任务。
- **开源 SWE-Lancer 以供未来研究**：为了促进进一步调查，SWE-Lancer 基准测试包含一个**统一的 Docker 镜像**以及一个名为 SWE-Lancer Diamond 的公共评估拆分集，可在 [GitHub](https://github.com/openai/SWELancer-Benchmark) 上获取。
   - 其目标是将模型性能映射到货币价值，从而可能在软件工程领域实现更多的研究机会。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.12115">SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?</a>：我们介绍了 SWE-Lancer，这是一个包含 1,400 多个来自 Upwork 的自由职业软件工程任务的基准测试，其真实世界总报酬价值为 100 万美元。SWE-Lancer 涵盖了独立工程...</li><li><a href="https://arxiv.org/abs/2502.11089?s=09">Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention</a>：长上下文建模对于下一代语言模型至关重要，但标准注意力机制的高计算成本带来了巨大的计算挑战。稀疏注意力提供...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1341118641638346844)** (94 messages🔥🔥): 

> `Thinking Machines Lab Launch, Perplexity R1 Finetune, SWElancer Benchmark, Grok 3 Announcement, Zed's Edit Prediction Model` 


- **Thinking Machines Lab 启动**：由 AI 领域的知名人士创立的 Thinking Machines Lab 旨在使 AI 系统更具可定制性并被广泛理解，从而弥合公众关于前沿 AI 讨论中的鸿沟。
   - 该倡议背后的团队成员包括开发了 ChatGPT 和 Character.ai 等广泛使用产品的人员，并承诺通过发表论文和发布代码来实现开放科学。
- **Perplexity 开源 R1 1776 模型**：Perplexity AI 开源了 R1 1776，这是 DeepSeek R1 模型的一个版本，旨在提供无审查且事实性的信息。
   - 社交媒体上的用户幽默地将这一公告称为“自由微调”（freedom tuning），表达了对该模型用途的调侃反应。
- **SWElancer 基准测试发布**：OpenAI 发布了 SWE-Lancer，这是一个用于评估 AI 模型代码性能的新基准测试，包含价值 100 万美元的 1,400 个自由职业软件工程任务。
   - 用户对这个名字表示惊讶，并对某些模型的缺席进行了猜测，暗示该基准测试背后可能存在战略动机。
- **对 Grok 3 的期待**：人们对 Grok 3 模型充满了期待，讨论了其与其他模型相比的能力以及在各种任务中的应用。
   - 社区中的 Walruses 对其在表面热度之后的实际表现表达了怀疑和幽默。
- **关于 Zed 新功能的讨论**：Zed 引入了一个开源的下一次编辑预测模型（next-edit prediction model），被视为 Cursor 等现有解决方案的潜在竞争对手。
   - 然而，用户对其差异化以及与 Copilot 等成熟模型及其近期功能相比的整体实用性表示担忧。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/deepseek_ai/status/1891745487071609327">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 介绍 NSA：一种硬件对齐且原生可训练的 Sparse Attention 机制，用于超快长上下文 (long-context) 训练与推理！NSA 的核心组件：• 动态分层稀疏策略•...</li><li><a href="https://x.com/johnschulman2/status/1891539960743743756">来自 John Schulman (@johnschulman2) 的推文</a>：@barret_zoph 和我最近在斯坦福大学做了一个关于 Post-training 以及我们在 ChatGPT 上合作经验的演讲。遗憾的是演讲没有录音，但这里有幻灯片：https://docs.g...</li><li><a href="https://thinkingmachines.ai/]">Thinking Machines Lab</a>：未找到描述</li><li><a href="https://thinkingmachines.ai/">Thinking Machines Lab</a>：未找到描述</li><li><a href="https://zed.dev/blog/edit-prediction">Zed 现在通过 Zeta（我们的新开源模型）预测你的下一次编辑 - Zed 博客</a>：来自 Zed 博客：一个能预判你下一步操作的工具。</li><li><a href="https://x.com/dchaplot/status/1891920016339042463">来自 Devendra Chaplot (@dchaplot) 的推文</a>：职业更新：非常幸运且兴奋能成为 Thinking Machines Lab 创始团队的一员！https://thinkingmachines.ai/ 加入我们：https://6wajk07p.paperform.co/</li><li><a href="https://x.com/madiator/status/1891835068315992324?s=46">来自 Mahesh Sathiamoorthy (@madiator) 的推文</a>：忘掉 Grok 3 吧。读读这些论文/博客。引用 John Schulman (@johnschulman2) @barret_zoph 和我最近在斯坦福大学做了一个关于 Post-training 以及我们在 ChatGPT 上合作经验的演讲...</li><li><a href="https://www.threads.net/@zuck/post/DGOISdTRX9Q?xmt=AQGzKO9jUlLz9JyNocdtLrsQ1L8IvvVBRb--7JMStLY6Fg">Mark Zuckerberg (&#064;zuck) 在 Threads 上</a>：请记住这些日期 🗓️ LlamaCon: 4 月 29 日 Connect: 9 月 17-18 日</li><li><a href="https://en.wikipedia.org/wiki/Thinking_Machines_Corporation">Thinking Machines Corporation - 维基百科</a>：未找到描述</li><li><a href="https://docs.google.com/presentation/d/11KWCKUORnPpVMSY6vXgBeFSWo7fJcuGQ9yuR6vC1pzE/edit#slide=id.g328faeed8ae_0_24">ChatGPT + Post-Training</a>：ChatGPT 与 Post-Training 的艺术，Barret Zoph &amp; John Schulman</li><li><a href="https://x.com/jrobertsai/status/1891506671056261413?s=46">来自 Jonathan Roberts (@JRobertsAI) 的推文</a>：计算机视觉被“解决”了吗？还没有。目前的模型在 ZeroBench 上得分为 0% 🧵1/6</li><li><a href="https://huggingface.co/stepfun-ai/Step-Audio-Chat">stepfun-ai/Step-Audio-Chat · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/perplexity_ai/status/1891916573713236248">来自 Perplexity (@perplexity_ai) 的推文</a>：今天我们开源了 R1 1776——这是 DeepSeek R1 模型的一个版本，经过 Post-training 以提供无审查、无偏见且事实性的信息。</li><li><a href="https://x.com/karpathy/status/1891720635363254772?s=46">来自 Andrej Karpathy (@karpathy) 的推文</a>：我今天早些时候获得了 Grok 3 的早期访问权限，我想我是首批能进行快速 vibe check 的人之一。Thinking✅ 首先，Grok 3 显然拥有一个接近 state of the art 的 thinking model ...</li><li><a href="https://x.com/openai/status/1891911132983722408?s=46">来自 OpenAI (@OpenAI) 的推文</a>：目前的前沿模型 (frontier models) 无法解决大部分任务。</li><li><a href="https://www.youtube.com/watch?v=AUAJ82H12qs"> - YouTube</a>：未找到描述</li><li><a href="https://x.com/thinkymachines/status/1891919141151572094">来自 Thinking Machines (@thinkymachines) 的推文</a>：今天，我们很高兴地宣布成立 Thinking Machines Lab (https://thinkingmachines.ai/)，一家人工智能研究与产品公司。我们是某些...背后的科学家、工程师和构建者。</li><li><a href="https://tenor.com/view/freedom-america-gif-15593845046973100361">Freedom America GIF - Freedom America - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/karpathy/status/1891938714915569711">来自 Andrej Karpathy (@karpathy) 的推文</a>：祝贺 Thinking Machines 公司发布！非常强大的团队，其中很大一部分人直接参与并构建了 ChatGPT 奇迹。很棒的人，值得关注，祝愿...</li><li><a href="https://x.com/scaling01/status/1891913189199053301?s=46">来自 Lisan al Gaib (@scaling01) 的推文</a>：我喜欢每篇 OpenAI 论文中为 Sonnet 3.5 做的广告。有点奇怪他们没有包含 o3。现在感觉要么这是一个局，以便他们以后可以粉碎自己的基准测试，要么就是 o3 表现很差...</li><li><a href="https://x.com/amasad/status/1891709057238507526?s=46">来自 Amjad Masad (@amasad) 的推文</a>：Grok 3 似乎是一个 state-of-the-art 的前沿模型。这是一个巨大的成就，尤其是考虑到他们起步这么晚。恭喜 @ibab、@elonmusk 以及 @xai 团队的其他成员...</li><li><a href="https://youtu.be/Ju0ndy2kwlw?si=_Maiv6-7b0dv3vLg">我用 5 台 Mac Studio 构建了一台 AI 超级计算机</a>：Ge

t NordVPN 2年方案 + 额外4个月 + 再额外6个月：https://nordvpn.com/networkchuck 享受 Nord 的30天退款保证，零风险！我刚刚...</li><li><a href="https://x.com/elonmusk/status/1891700271438233931">来自 Elon Musk (@elonmusk) 的推文</a>：https://x.com/i/broadcasts/1gqGvjeBljOGB</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1irynqt/plus_plan_has_a_context_window_of_only_32k_is_it/">Reddit - 深入探索一切</a>：未发现描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: 新播客发布！ https://x.com/latentspacepod/status/1891879917224132973
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1341111358971973782)** (24 条消息🔥): 

> `Max Headroom 播客制作, 音频生成的语言设置, Notebook LM 音频功能, 音频生成挑战, Google 产品限制` 


- **Max Headroom 重启播客详情**：一位用户描述了他们使用 Notebook LM 进行文本生成并使用 Speechify 进行语音克隆来制作 Max Headroom 播客的过程，总计耗时 40 小时的制作时间。
   - 他们调整了 Prompt 并编辑了文本，结合不同的工具以达到预期效果并增强播客制作质量。
- **更改 Notebook LM 中的语言设置**：用户对“studio”中的语言设置表示沮丧，尽管尝试自定义设置，但仍无法生成西班牙语。
   - 几位用户指出这可能是影响账号的 Bug，并表示即使更换账号也无法解决问题。
- **Notebook LM 音频生成能力**：讨论了用户可以将脚本导入 Notebook LM 以创建包括播客和广告在内的各种格式音频的功能。
   - 用户强调了节省时间和修改脚本的灵活性等优势，同时也提到了需要编辑软件进行精细化处理。
- **旧账号的音频生成挑战**：一位用户报告称其旧的付费账号产生的结果不太理想，认为音频导入格式可能会影响性能。
   - 他们指出，虽然该设置在免费账号上运行正常，但在不准确地导入 MP4 文件时问题依然存在。
- **Google 系统的限制**：参与者讨论了对 Google 产品表现更好的期望，特别是当它们开始对服务收费时。
   - 用户对系统设计的局限性表示担忧，并对影响用户体验的技术缺陷感到沮丧。



**提到的链接**：<a href="https://youtu.be/snDzpZBH8v0">Max Headroom Rebooted 2025 全集 20 分钟</a>：🚨 BZZZZZT! 警报！警报！🚨 未来已崩坏——我回来报告了！💾 迷失在数字虚空……然后被一只垃圾熊猫重启了？！💾 在深处的某个地方...

  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1341099126288421105)** (65 messages🔥🔥): 

> `NotebookLM 使用问题, Podcast 功能, 语言设置, 研究人员的文件管理, 翻译能力` 


- **NotebookLM 用于研究的挑战**：研究人员反映，NotebookLM 将 PDF 文件重命名为 **2022-2~4.PDF** 等模糊格式，导致难以识别来源。
   - 改进建议包括允许直接查看 PDF 标题，并优化命名规范以方便导航。
- **Podcast 创建限制的困惑**：用户讨论了意外达到 Podcast 创建限制的 Bug，有人指出在免费版本上只能创建 **3** 个 Podcast，而限制本应更高。
   - 澄清显示，**50** 次是聊天查询的限制，而非 Podcast，这引起了用户的一些困惑。
- **设置语言偏好**：有关于更改 NotebookLM 语言设置的咨询，特别是希望从 **German** 切换到 **English** 的用户。
   - 一个建议是更新 Google 账号的语言偏好，这会直接影响 NotebookLM 的运行方式。
- **翻译 Podcast 内容的局限性**：用户注意到目前 Podcast 只能以 **English** 生成，因此询问是否有可用的翻译工具。
   - 虽然云服务商提供翻译 API，但非开发人员可能难以找到易于使用的选项。
- **Podcast 语气和长度的调整需求**：一位用户试图修改其 Podcast 的语气和长度，但被告知这些调整仅适用于 NotebookLM 的回复，而不适用于 Podcast。
   - 提供了如何配置聊天设置以调整回复的说明，间接将这些设置与用户满意度联系起来。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://notebooklm.google.com/?hl=en">未找到标题</a>：未找到描述</li><li><a href="https://ai.google.dev/gemini-api/docs/prompting-intro">未找到标题</a>：未找到描述</li><li><a href="https://ai.google.dev/gemini-api/docs/prompting-strategies">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1341095690293743627)** (14 messages🔥): 

> `Polars DataFrame 库, Mojo 集成, 标准库团队扩张, Apache Arrow 实现` 


- **探索 Polars 库**：几位成员对 **Polars** DataFrame 库表示了兴趣，认可其相比 **pandas** 的潜在优势，特别是在性能方面。
   - *Lucablight4200* 鼓励其他人为未来的 DataFrame 工作探索 Polars，并表示这在未来将是必不可少的。
- **将 Polars 与 Mojo 集成**：有人表达了将 **Polars** 接入 Mojo 项目的兴趣，认为这种集成在数据处理方面具有巨大潜力。
   - *Sa_code* 强调，Mojo 的框架可以实现一个 **silicon agnostic**（硬件无关）的 DataFrame 库，适用于单机和分布式环境。
- **关于标准库团队扩张的讨论**：一位成员询问了扩大 **stdlib** 团队的可能性，并幽默地提到自己是加拿大人以突显性价比。
   - 这引发了一场关于资质和贡献的轻松交流，暗示专业知识并非唯一的考量因素。
- **在 Mojo 中实现 Apache Arrow**：一位成员提到在 Mojo 中有一个 **Apache Arrow** 的小型实现，认为这对于一个完善的 DataFrame 库是必要的。
   - 然而，*sa_code* 承认他们还没有时间进一步开发，并表示这需要比业余爱好更多的精力。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1341139277664485427)** (39 messages🔥): 

> `Mojo 的 ChatGPT 替代方案、Mojo 代码重构挑战、使用 Enzyme 进行自动微分、Mojo 对全局变量的支持、在 Mojo 中使用 List 与 Stack 的对比` 


- **Mojo 替代方案：Gemini 2.0 表现出色**：几位成员建议在 Mojo 重构任务中使用 `Gemini 2.0 Flash` 或 `Gemini 2.0 Flash Thinking`，称其对该语言有相当程度的了解。
   - 一位成员验证了 **Zed** 代码编辑器的使用情况，并强调了它与 [此处](https://github.com/freespirit/mz) 找到的 Mojo 扩展的兼容性。
- **将大型 Python 项目重构为 Mojo 的挑战**：一位用户表达了将大型 Python 项目重构为 Mojo 的困难，表示由于当前工具的问题，可能需要进行手动更新。
   - 另一位成员承认，由于 Borrow Checker 的限制，重构 Mojo 代码可能涉及大量的架构重组。
- **对在 Mojo 中使用 Enzyme 进行 Autodiff 的兴趣**：讨论了通过 [Enzyme](https://github.com/EnzymeAD/Enzyme) 项目在 Mojo 中实现自动微分（Autodiff）的可能性，并提出了支持该功能的建议。
   - 成员们讨论了实现 Autodiff 可能涉及将 Mojo AST 转换为 MLIR 以进行优化，表现出对这一能力的浓厚兴趣。
- **Mojo 对全局变量的支持尚不明确**：一位成员询问了 Mojo 未来对全局变量的支持情况，希望能得到明确答复。
   - 另一位成员幽默地提到对 **global expressions** 的兴趣，引发了对未来可能发展的关注。
- **Mojo 中 List 的速度问题**：成员们辩论了使用 `List` 实现 Stack 的性能开销，并建议使用 `List.unsafe_set` 来避免边界检查。
   - 有人担心 `List` 中对象的复制会影响速度，并提供了一个展示对象移动（object movement）的简单示例作为变通方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EnzymeAD/Enzyme/tree/main/enzyme/Enzyme/MLIR">Enzyme/enzyme/Enzyme/MLIR at main · EnzymeAD/Enzyme</a>：LLVM 和 MLIR 的高性能自动微分。 - EnzymeAD/Enzyme</li><li><a href="https://www.modular.com/blog/modular-natively-supports-dynamic-shapes-for-ai-workloads">Modular: Modular natively supports dynamic shapes for AI workloads</a>：当今的 AI 基础设施难以评估——许多人倾向于简单且可量化的指标，如 QPS、延迟和吞吐量。这也是为什么当今 AI 行业充斥着各种……</li><li><a href="https://zed.dev.">Zed - The editor for what&#x27;s next</a>：Zed 是来自 Atom 和 Tree-sitter 创作者的高性能多人协作代码编辑器。</li><li><a href="https://github.com/freespirit/mz">GitHub - freespirit/mz: Support for Mojo in Zed</a>：Zed 对 Mojo 的支持。在 GitHub 上通过创建一个账号为 freespirit/mz 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1341121773793579081)** (46 条消息🔥): 

> `用于测试 GPT4All 的 GPU、类 Deep-Research 功能、1000 万 Embedding Token 计数、CUDA 5.0 支持` 


- **寻找特定 GPU 以测试 GPT4All**：成员们讨论了各种 GPU，重点关注测试 GPT4All 特别版本所需的 compute 5.0 GPU（如 **GTX 750/750Ti** 等）。
   - 一位成员提到他们有 GTX 960，但其他人对 **VRAM 限制**和兼容性表示担忧。
- **对类 Deep-Research 功能的好奇**：一位成员表示有兴趣了解 GPT4All 是否可能推出类似于其他 AI 工具中的 **Deep-Research-like** 功能。
   - 对话继续进行，其他人寻求关于此类功能具体包含什么的澄清。
- **关于 1000 万 Token 计数限制的澄清**：讨论了 Atlas 的 Embedding 达到 **1000 万 Token** 限制的影响，一位成员确认超过此限制将需要付费或使用本地模型。
   - 澄清了计费是基于总的 Embedded Tokens，这意味着之前的 Token 不能从计数中扣除。
- **启用 CUDA 5.0 支持**：关于启用 **CUDA 5.0 GPU** 支持的潜在风险进行了对话，担心这样做可能会导致崩溃或后续需要修复的问题。
   - 共识是在没有进一步测试和确认的情况下，在官方发布说明中声明此类支持是不审慎的。



**提到的链接**：<a href="https://en.wikipedia.org/wiki/CUDA#GPUs_supported)">CUDA - Wikipedia</a>：未找到描述

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1341098367253741691)** (3 条消息): 

> `LLM Consortium、Mistral Saba、供应商问卷的语义检索` 


- **Massimiliano 构建 LLM Consortium**：我们的框架工程师 **Massimiliano Pippi** 受到 @karpathy 想法的启发，实现了一个 **LLM Consortium**，使用 [LlamaIndex](https://t.co/iIbLjY7K23) 收集多个 LLM 对同一个问题的回答。
   - 该项目有望增强协作式 AI 回答的准确性和见解共享。
- **Mistral AI 发布阿拉伯语模型**：**@mistralAI** 的朋友们发布了 **Mistral Saba**，这是一个专注于阿拉伯语的新型小模型，并提供 **day 0 支持**集成。
   - 用户可以通过在[此处](https://t.co/bvuwqOWnOB)运行 `pip install llama-index-llms-mistralai` 快速开始使用新模型。
- **创新的供应商问卷应用**：查看来自 **@patrickrolsen** 的全栈应用，它允许用户通过**语义检索（semantic retrieval）**和 LLM 增强来回答供应商问卷。
   - 该应用程序解决了**填表复杂性**，并举例说明了 Knowledge Agent 的核心用例，展示了一个用于检索先前答案的用户友好型解决方案，链接在[此处](https://t.co/ylgv9UFBj0)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1341103229160915085)** (23 条消息🔥): 

> `Vector stores 中的 Metadata 过滤器, Agent Workflow 特性, 使用 LlamaIndex 构建 AI 聊天机器人, 本地 LLM 的 Embedding 安装, 发行说明位置` 


- **日期 Metadata 过滤器的局限性**：目前许多 Vector stores 并不直接支持按日期过滤，这使得有效地实现此类功能具有挑战性。一些 Vector stores 允许通过其特定的查询语言进行过滤，但其他存储（如 PostgreSQL）可能需要自定义处理。
   - *一位成员评论说，在 Metadata 中分离年份对于过滤至关重要。*
- **创建商业报价 Agent 的指南**：一位开发者分享了创建一个使用自然语言生成和完善商业报价的 Agent 的计划，概述了各种功能并征求对其方法的意见。他们收到了构建自定义 Workflow 以处理此过程中所需不同任务的建议。
   - 强调了在 LlamaIndex 中使用 `@step` 装饰器来管理事件驱动型 Workflows 的重要性。
- **澄清 AgentWorkflow 的用法**：一位用户在使用 `AgentWorkflow` 时遇到了 `response.tool_calls` 列表为空的问题，未能捕获工具输出。建议将事件进行流式传输（stream）作为捕获所有工具调用的权宜之计。
   - *社区认为这可能是一个 Bug，需要在未来的更新中解决。*
- **Embeddings 安装指南**：一位成员提到正在阅读关于为本地 LLM 安装 Embeddings 的文档，特别是在使用 HuggingFace 时。他们似乎在为项目搭建必要的基础设施方面取得了进展。
   - *其他人对 AI 领域层出不穷的教程和更新表示担忧，这表明学习曲线非常陡峭。*
- **发行说明的位置**：一位用户询问在哪里可以找到 LlamaIndex 的发行说明。另一位成员迅速提供了 GitHub 上 [CHANGELOG](https://github.com/run-llama/llama_index/blob/main/CHANGELOG.md) 文档的链接。
   - *该文档是跟踪 LlamaIndex 更新和更改的资源。*


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/tree/main/docs">llama_index/docs at main · run-llama/llama_index</a>：LlamaIndex 是构建基于数据的 LLM 驱动 Agent 的领先框架。- run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_using_qdrant_filters/">Qdrant Vector Store - Default Qdrant Filters - LlamaIndex</a>：暂无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/workflow/#workflows">Workflows - LlamaIndex</a>：暂无描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/CHANGELOG.md">llama_index/CHANGELOG.md at main · run-llama/llama_index</a>：LlamaIndex 是构建基于数据的 LLM 驱动 Agent 的领先框架。- run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agents/">Multi-agent workflows - LlamaIndex</a>：暂无描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1341152424202735636)** (1 条消息): 

> `基于 JSON 字典的 RAG, 用户查询文档匹配, 在大型 JSON 中查找文档` 


- **探索基于 JSON 字典的 RAG**：成员们讨论了仅依赖 **JSON 字典**来提高文档匹配效率的 **RAG** (Retrieval-Augmented Generation) 技术示例。
   - 他们分享了关于集成结构化数据如何增强传统搜索方法以获得更好查询响应的见解。
- **JSON 文档匹配的最佳实践**：对话强调了根据用户查询在大型 JSON 数据集中匹配文档的几项**最佳实践**。
   - 一位参与者强调了索引和利用 Metadata 来简化检索过程的重要性。
- **JSON 文档检索中的挑战**：参与者对从大型 JSON 存储中有效检索文档的潜在**挑战**表示担忧。
   - 扩展检索机制以及在高查询负载期间保持性能是讨论的重点。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1341454925342707854)** (1 条消息): 

> `Byte Latent Transformers, Qwen Fine-Tuning, TorchTune Hacks` 


- **为 Qwen 微调 Hack Byte Latent**：一位成员分享了他们尝试将 **Byte Latent Transformers** 重新实现为 **Qwen** 的一种微调方法的尝试，灵感来自现有代码，并详细介绍了使用 [此 GitHub 仓库](https://github.com/ianbarber/ttblt) 的经验。
   - 虽然实验的 **loss 正在下降**，但结果仍然产生 **无意义的输出**，这表明需要更多工作来使实现与 **TorchTune** 的框架对齐。
- **在 TorchTune 中适配模型的挑战**：该成员指出，将模型迁移到 **TorchTune** 带来了独特的挑战，特别是涉及到不同的 checkpoint 格式。
   - 他们分享了见解，希望能帮助其他在自己的 **hacks** 中探索类似建模方法的人。



**提及的链接**：<a href="https://github.com/ianbarber/ttblt">GitHub - ianbarber/ttblt: A simplified implementation of Byte Latent Transformers as a TorchTune recipe.</a>：Byte Latent Transformers 作为 TorchTune recipe 的简化实现。 - ianbarber/ttblt

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1341093813443100864)** (18 条消息🔥): 

> `Unit Test Handling, Optional Dependencies for Development, Checkpoint Resuming Logic, Step-Based Checkpointing, Cross-Contributing on PRs` 


- **通过减少安装来简化单元测试**：有人对仅为了运行 **unit tests** 就需要在 **四种不同的安装** 之间切换的不便表示担忧。
   - 有人建议对贡献者保持某些依赖项为 **optional**（可选），同时维持流畅的本地体验。
- **完善 Checkpoint 恢复逻辑**：当前的 checkpoint 恢复方法涉及将其保存到 `${output_dir}` 中，但现在有了 **step-based checkpointing**，逻辑可能需要修订。
   - 一个提议的解决方案是保留现有的 **resume_from_checkpoint**，同时允许用户选择从 **latest**（最新）或 **特定 checkpoints** 恢复。
- **解决多个实验重叠问题**：针对在多个实验中重复使用 `${output_dir}` 以及确保每次运行都有独立输出的挑战展开了对话。
   - 有人强调，为实验 A 和 B 维护唯一的输出目录可以防止来自不同运行的 epoch 混淆。
- **在 Step-Based Checkpointing PR 上协作**：成员们表达了共同为 **step-based checkpointing** PR 做出贡献的愿望，并征求对 API 设计的意见。
   - 一位成员提出实现支持从之前的 checkpoints 恢复的功能，并表示有兴趣简化开发流程。
- **促进跨贡献者审批**：有人建议允许贡献者在个人 fork 中对 PR 进行 **cross-approval**（交叉审批）和合并，以加快开发工作流。
   - 这可以增强在现有问题（如 **step-based checkpointing** PR）上的协作，并允许来自不同团队成员的更高效输入。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/torchtune/main/deep_dives/checkpointer.html#resuming-from-checkpoint-full-finetuning">Checkpointing in torchtune &mdash; torchtune main documentation</a>：暂无描述</li><li><a href="https://github.com/pytorch/torchtune/pull/2105">[RFC] Step-based checkpointing in torchtune by joecummings · Pull Request #2105 · pytorch/torchtune</a>：在 torchtune 中启用 step-based checkpointing。原始背景：#2070。我们目前在做什么？我们目前仅在 epoch 边界进行 checkpoint。这意味着微调运行必须迭代完成...</li><li><a href="https://github.com/joecummings/torchtune/pull/2">feat: get_latest_checkpoint for checkpointer utils by bogdansalyp · Pull Request #2 · joecummings/torchtune</a>：添加了 get_latest_checkpoint """ 返回给定目录中的最新 checkpoint。pattern 参数是一个正则表达式，用于匹配 epoch 编号...
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1341410170378387487)** (4 messages): 

> `Reinforcement Learning (RL) for pre-training` 


- **关于 RL 在预训练中作用的辩论**：成员们就 **Reinforcement Learning (RL)** 在预训练阶段的有效性展开了讨论，部分成员表示怀疑。
   - *作为一名 RL 的狂热爱好者*，一位成员认为将 RL 用于预训练的想法非常**可怕 (terrifying)**，突显了对其应用观点的分歧。
- **反对在预训练中使用 RL 的共识**：另一位成员呼应了之前的观点，指出 RL *不适合* 预训练阶段，强化了共同的担忧。
   - 参与者之间的总体共识似乎倾向于在预训练过程中避免使用 RL。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1341223695200292925)** (7 messages): 

> `Glama MCP Server Changes, OpenRouter Documentation, Anthropic Homepage Status, Haiku 3.5 Release, Sonnet 4.0 Release` 


- **如何根据 MCP Server 的变更更新 Glama**：一位用户询问了如何让 **Glama** 识别其 MCP Server 所做的更改。
   - 这引发了社区的回应，强调了配置输入需要明确。
- **社区呼吁提供 OpenRouter 文档**：一位成员敦促将 **OpenRouter** 文档转换为 MCP 格式，并链接到了 [openrouter.ai](https://openrouter.ai/docs/llms-full.txt) 的源文件。
   - 该建议强调了对多种格式的可访问资源的需求。
- **Anthropic 官网访问问题**：成员们注意到 **Anthropic 官网** 似乎挂了，并分享了一张图片作为证据。
   - 这引发了对重要更新或发布期间访问性的担忧。
- **对具备高级功能的 Haiku 3.5 的期待**：成员们推测 **Haiku 3.5** 可能会在今天发布，并支持 tool 和 vision 功能，现场气氛热烈。
   - 这引发了关于此类功能对效率影响的讨论。
- **关于 Sonnet 4.0 发布的传闻**：闲谈中提到，备受期待的 **Sonnet 4.0** 可能也即将推出。
   - 成员们对即将发布的新功能公告充满好奇和期待。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1341163132155986047)** (11 messages🔥): 

> `MCP server for debugging, Continue tool features, Clear Thought MCP Server` 


- **MCP Server 实现直接调试**：一位成员构建了一个 **MCP server** 和一个 VS Code 扩展，允许 LLM（如 Claude）跨多种编程语言**交互式地调试代码**。
   - 更多功能详情请查看 [GitHub](https://github.com/jasonjmcghee/claude-debugs-for-you) 上的项目。
- **关于错误读取功能的讨论**：一位用户询问 MCP server 是否可以自动**读取终端错误**并修复它们，另一位成员提到 **Continue** 具备此类功能。
   - 尽管该 MCP server 目前尚不具备此功能，但 *jason.today* 指出未来有添加该功能的潜力。
- **区分调试与猜测**：一位贡献者解释说，调试工具允许在执行期间**检查变量状态**，而不是仅依靠日志来识别错误。
   - 这突显了当前 AI 编程工具的一个空白，即往往缺乏集成的调试功能。
- **推出 Clear Thought MCP Server**：推出了一款新的 **Clear Thought MCP Server**，旨在利用**思维模型 (mental models)** 和系统化思考方法论来增强问题解决能力。
   - 该服务器旨在通过结构化方法提高开发环境中的决策能力，可通过 [NPM](https://smithery.ai/server/@waldzellai/clear-thought) 获取。
- **与其他工具的比较**：另一位成员提到，另一个工具 **Goose** 也可以读取终端错误并在调试期间自动解决它们。
   - 这表明关于各种 AI 辅助编程工具的能力和功能的讨论正在升温。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://smithery.ai/server/@waldzellai/clear-thought">Clear Thought MCP Server | Smithery</a>：未找到描述</li><li><a href="https://github.com/jasonjmcghee/claude-debugs-for-you">GitHub - jasonjmcghee/claude-debugs-for-you: 通过 MCP 和 VS Code 扩展，让任何 LLM（如 Claude）都能为你交互式地调试任何语言。</a>
</li>
</ul>

</div>
  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1341304298855858187)** (2 messages): 

> `证书发放，Fall24 学生参与情况` 


- **已公布证书发放总数**：共有 **304 名 trailblazers** 🚀、**160 名 masters** 🧑‍🏫、**90 名 ninjas** 🥷、**11 名 legends** 🏆 和 **7 名 honorees** 🌟 被授予证书，展示了多样化的成就。
   - 在 **7 名 honorees** 中，有 **3 名 ninjas** 和 **4 名 masters**，体现了竞争激烈的评选过程。
- **Fall24 学生参与统计数据**：**Fall24 MOOC** 共有 **1.5 万名学生**，其中大多数为旁听生。
   - 如此规模的参与度表明了学生对该学期课程的浓厚兴趣。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1341259985404952668)** (7 messages): 

> `Inference-Time Techniques，LangChain 框架，结合 ML 模型的 LLM Agents` 


- **澄清 LLM 语境下的“程序 (program)”**：在关于 Inference-Time Techniques 的讲座中，成员们澄清了所讨论的“程序”是指实际的编程代码而非 LLM，并将其设定为竞赛编程任务。
   - 参与者引用了第 46 页幻灯片来支持这一理解。
- **执行程序中的采样方法**：据解释，在“执行采样程序”步骤中的采样涉及使用微调后的 LLM 生成大量潜在解决方案，而不仅仅是最初认为的筛选 10 到 20 个 LLM。
   - 该过程依赖于为特定问题生成的候选解决方案，如第 45 页幻灯片所示。
- **程序的聚类与评分**：讨论强调，从最大的簇（clusters）中采样程序旨在选择表现出输出一致性的解决方案，而不是直接对簇的性能进行评分。
   - 这涉及使用另一个 LLM 生成新的输入案例，并根据输出的一致性进行聚类，这意味着较大的簇反映了对这些解决方案更高的置信度。
- **LangChain 的应用生命周期**：一名成员概述了 LangChain，它通过其基于组件的框架促进了使用 LLM 的应用程序的开发、产品化和部署。
   - 关键特性包括与 API 的集成，以及专注于使用各种 LangChain 工具进行有状态 Agent 的开发。
- **将 LLM Agents 与 ML 预测模型结合**：一位参与者表达了对整合 LLM Agents 与机器学习预测模型以增强工作流的知识共享兴趣。
   - 正在寻求社区关于此类组合的反馈和经验。



**提及的链接**：<a href="https://python.langchain.com/docs/introduction/">Introduction | 🦜️🔗 LangChain</a>：LangChain 是一个用于开发由大语言模型 (LLMs) 驱动的应用程序的框架。

  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1341516820217331732)** (1 messages): 

> `利润分成机会，Telegram 营销` 


- **Rose Miller 提供快速获利机会**：Rose Miller 提议帮助 **10 个人**在 **72 小时**内赚取 **3 万美元**，并要求在收到收益后分享 **10%** 的利润。
   - 有兴趣的参与者应在 **Telegram** 或 **WhatsApp** 上给她发消息以开始，使用提示词 (HOW)。
- **在 Telegram 上加入 Rose 以获取更多信息**：Rose 鼓励有 **Telegram** 的人直接联系 @officialRose_miller 以获得快速回复。
   - 提供了她的 **Telegram** 个人资料链接和 **portfolio manager** 描述以证明可信度。



**提及的链接**：<a href="https://t.me/officialRose_miller">Rose Miller</a>：投资组合经理，保守型

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1341516922218479698)** (1 messages): 

> `利润分成机会，Rose Miller 的公告` 


- **Rose Miller 提供利润分成以赚取 3 万美元**：**Rose Miller** 宣布了一项计划，帮助 **10 个人在 72 小时内赚取 3 万美元**，并要求在收到后提取 **10% 的利润**。
   - *感兴趣的人士应通过 [此链接](https://t.me/officialRose_miller) 在 Telegram 或 WhatsApp 上给她发消息*以获取更多详情。
- **分享联系信息**：Rose Miller 提供了她的 **Telegram** 和 **WhatsApp** 联系信息供感兴趣的各方联系。
   - 她的 Telegram 账号是 [@officialRose_miller](tg://resolve?domain=officialRose_miller)，她位于**美国**。



**提及的链接**：<a href="https://t.me/officialRose_miller">Rose Miller</a>：投资组合经理，保守型

  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1341516890316734615)** (4 messages): 

> `Structured Data Injection, Context API Launch, High-Quality Data Access, Feedback Request, Blog Post Discussion` 


- **结构化数据注入增强 AI 模型**：新推出的端点可将来自 **Arxiv** 和 **Wikipedia** 的**高质量结构化数据**直接注入 AI 的 context windows，旨在降低延迟并提高数据检索效率。
   - **学术出版商数据集**定于下周发布，旨在为 AI 应用提供**可信来源**。
- **挑战 Context API 的极限**：团队邀请开发者测试新 API 并提供性能反馈，并提供 **$10 的免费额度**供用户上手。
   - 他们强调希望用户能积极**寻找漏洞并测试边缘情况**，以改进 API。
- **关于新 Context API 博客文章的讨论**：发布了一篇[博客文章](https://www.valyu.network/blog/why-we-built-context-api)，讨论了 AI 开发者在**数据检索**方面面临的挑战以及新 Context API 提供的解决方案。
   - 文章强调了在复杂决策用例中，AI Agent 和 LLM 应用对**高保真检索**的需求。
- **AI 数据可访问性的现状**：博客指出，目前的 AI Agent 往往缺乏对高质量数据的**高效访问**，限制了它们在深度研究中的能力。
   - 他们指出，随着 AI 应对更复杂的**挑战**，嵌入结构化数据变得愈发关键。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://exchange.valyu.network/">∇ Valyu - 为 AI 模型和应用提供高质量数据</a>：未找到描述</li><li><a href="https://www.valyu.network/blog/why-we-built-context-api">为什么我们构建了 Context API：因为你的 AI 需要事实，而非感觉 • Valyu 博客</a>：AI 模型可以生成内容，但在检索方面表现不佳——依赖于有限的网络搜索而非高质量、可信的数据。这就是为什么 Valyu 构建了 ContextAPI，无缝集成权威来源...</li><li><a href="https://t.me/officialRose_miller">Rose Miller</a>：投资组合经理，保守型
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1341516867130495079)** (1 messages): 

> `Profit Sharing Opportunity, Fast Money Making Scheme` 


- **72 小时内赚取 $30k**：一名成员宣布了一个机会，帮助 **10 个人**在 **72 小时**内赚取 **$30k**，收到款项后需支付 **10% 的利润分成**。
   - *感兴趣的人士*被鼓励通过 **Telegram** 或 **WhatsApp** 发送消息了解如何开始。
- **直接联系 Rose Miller**：用户可以直接通过 **Rose Miller** 的 **Telegram** 账号联系她，了解利润分成机会的更多细节。
   - 提供了她的 **Telegram** 账号链接以便快速访问：[联系 Rose](https://t.me/officialRose_miller)。



**提及的链接**：<a href="https://t.me/officialRose_miller">Rose Miller</a>：投资组合经理，保守型

  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1341488861301444749)** (2 messages): 

> `Self-Supervised Prompt Optimization, Importance of Prompt Design, LLM Output Quality, DSPy Mention, Cost-effective Frameworks` 


- **介绍自监督提示词优化 (Self-Supervised Prompt Optimization)**：论文提出了 **SPO (Self-Supervised Prompt Optimization)**，这是一个旨在不依赖外部参考的情况下增强 LLM 推理能力的框架。
   - 该方法纯粹通过输出对比生成评估信号，简化了针对各种任务的提示词设计过程。
- **提示词设计的关键作用**：精心设计的提示词对于使 LLM 输出符合不同领域的任务要求至关重要，但手动创建提示词可能非常耗时。
   - 现有方法对外部参考的依赖在实际应用中构成了挑战，因为此类数据往往难以获取或成本高昂。
- **对提及 DSPy 的评价**：一名成员对 **DSPy** 仅在论文最后一段被提及表示惊讶。
   - 这一评论表明，鉴于 DSPy 与当前讨论的高度相关性，人们希望它能获得更多的关注。



**提及的链接**：<a href="https://arxiv.org/abs/2502.06855">Self-Supervised Prompt Optimization</a>：精心设计的提示词对于增强大语言模型 (LLM) 的推理能力，同时使其输出符合不同领域的任务要求至关重要。然而，手动设计...

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1341527810707034122)** (1 messages): 

> `RouteLLM, GPT-5 讨论` 


- **对 RouteLLM 集成的担忧**：一位成员对即将推出的 **GPT-5** 表示怀疑，认为它只是在没有任何实质性更新的情况下，将 **RouteLLM** 与 **4o**, **o1**, **o3**, **Voice** 和 **Sora** 等模型整合在一起。
   - *他们打赌这种集成会忽略适当的引用，* 暗示这种方法缺乏新意。
- **之前关于 DSPy 的提议**：该用户回忆了他们之前关于 **DSPy** 的提议，并引用了一条暗示类似集成想法的推文。
   - *他们评论道 'lol'，表现出对这种情况以及竞争对手缺乏原创内容的幽默感。*



**提到的链接**：<a href="https://x.com/DataDeLaurier/status/1891896292650991810">来自 Dᴀᴛᴀ Sᴀᴄᴋs (@DataDeLaurier) 的推文</a>：@Teknium1 因为他们没有任何新东西。他们正准备把 4o, o1, o3, Voice 和 Sora 塞进 RouteLLM 并称之为 GPT-5。我敢打赌他们真的会使用 RouteLLM 且不引用任何人。

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1341321718056357978)** (2 messages): 

> `针对 Pull Request #9155 的测试，DEBUG=2 特性` 


- **请求对 PR #9155 进行测试**：一位成员询问是否有人想为 [Pull Request #9155](https://github.com/tinygrad/tinygrad/pull/9155) 编写 **测试**，该 PR 与在 **DEBUG=2** 中恢复 **颜色** 有关。
   - 他们包含了一个直接链接到 PR 的图像预览，强调了关于改进调试功能的讨论。
- **成员愿意编写测试**：另一位成员表示愿意承担这项任务，说道：*sure lemme write a test*。
   - 这一回应表明了社区内部支持 tinygrad 项目增强功能的协作努力。



**提到的链接**：<a href="https://github.com/tinygrad/tinygrad/pull/9155">colors back in DEBUG=2 [pr] by geohot · Pull Request #9155 · tinygrad/tinygrad</a>：未找到描述

  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1341218599246762024)** (1 messages): 

> `GenAI 视频生成，Seaweed-APT，AI 叙事，Nvidia 的通用 Embodied Agent，构建可扩展的训练流水线` 


- **Diffusion 在 GenAI 视频未来中逐渐淡出**：在题为 *“为什么 Diffusion 不是视频生成的未来”* 的会议中，Bichen Wu 分享了关于视频生成范式转移的见解，并强调了对研究人员和 MLE 的招聘。
   - Wu 是由前 CEO Eric Schmidt 领导的一家隐身 AI 初创公司的联合创始人，他利用在 Meta GenAI 的经验深入研究新兴技术。
- **Seaweed-APT 速度比 Sora 快 50 倍**：由 Peter Lin 领导的关于 *Seaweed-APT* 的演讲宣称其能力比 **Sora 快 50 倍**，彻底改变了视频生成技术。
   - Lin 是这一突破性模型的创造者，也是 [AnimateDiff-Lightning](https://huggingface.co/ByteDance/AnimateDiff-Lightning) 的第一作者，他详细介绍了其令人印象深刻的速度和效率。
- **OpenArt 实现 800 万美元 ARR 突破**：OpenArt 透露了其在短短 10 个月内实现 **800 万美元 ARR** 的惊人策略，展示了 AI 叙事方面的创新。
   - 会议分享了改变 AI 应用商业化格局的关键增长黑客（growth hacks）。
- **Nvidia 破译用于 Embodied AI 的世界模型**：一位 Nvidia 研究员解释了在开发 *通用 Embodied Agent* 方面的进展，重点关注通用世界模型和模拟范式。
   - 这一探索旨在通过复杂的 AI 框架增强现实世界的机器人应用。
- **为模型训练创建黄金数据集**：Pareto AI 介绍了构建 *黄金数据集 (golden datasets)* 的技术，以扩展下一代模型的图像和视频训练流水线。
   - 他们的策略被定位为提升未来 AI 系统在多样化环境中能力的关键。



**提到的链接**：<a href="https://lu.ma/k43k14as">GenAI Video, World Models &amp; Robotics #Kling #Veo #Sora #Cosmos #Diffusion · Luma</a>：加入我们，获取关于驱动实时单步文本转视频生成、通用世界模型等尖端技术的原始见解……

  

---


---


{% else %}


> 各频道的完整详细分析已针对邮件进行截断。
> 
> 如果您想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}