---
companies:
- openai
date: '2024-12-21T01:44:22.839525Z'
description: '**OpenAI** 发布了 **o3** 和 **o3-mini** 模型，并展示了突破性的基准测试结果，包括在 **FrontierMath**
  基准测试中从 **2% 跃升至 25%**，以及在 **ARC-AGI** 推理基准测试中达到 **87.5%**。这代表了在 GPT3 到 GPT4o 的扩展曲线上约
  **11 年的进步**。


  与 o3 全量版（o3-full）相比，**o1-mini** 模型表现出更优越的推理效率，有望显著降低编程任务的成本。此次发布还伴随着社区讨论、安全性测试应用以及详细分析。*Sama*（山姆·奥特曼）强调了这种不同寻常的性价比权衡，**Eric
  Wallace** 则分享了关于 o 系列“深思熟虑对齐策略”（deliberative alignment strategy）的见解。'
id: b92815e1-7acf-47ff-950a-245e04940e94
models:
- o3
- o3-mini
- o1-mini
- gpt-3
- gpt-4o
- o1
original_slug: ainews-o3-solves-aime-gpqa-codeforces-makes-11
people:
- sama
- eric-wallace
title: o3 攻克了 AIME、GPQA 和 Codeforces，在 ARC-AGI 上实现了相当于 11 年的跨越式进展，并在 FrontierMath
  中取得了 25% 的成绩。
topics:
- benchmarking
- math
- reasoning
- model-performance
- inference-speed
- cost-efficiency
- alignment
- safety-testing
---

<!-- buttondown-editor-mode: plaintext -->**蒸馏推理时计算（Distilled Inference Time Compute）就是你所需要的一切。**

> 2024年12月19日至12月20日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord（**215** 个频道和 **6058** 条消息）。预计节省阅读时间（以 200wpm 计算）：**607 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

随着[核心研究人员的离职](https://x.com/steph_palazzolo/status/1869848094009110826)、[Veo 2](https://news.ycombinator.com/item?id=42432914) 在正面交锋中击败 Sora Turbo，以及 Noam Shazeer 推出了全新的 [Gemini 2.0 Flash Reasoning](https://x.com/noamshazeer/status/1869789881637200228?s=46) 模型，OpenAI 周围的氛围往好里说也是非常紧张的。

但耐心得到了回报。

正如 [sama 所预告的](https://x.com/sama/status/1869963879671013774)，以及[互联网侦探](https://x.com/btibor91/status/1870022347349987532)和[记者](https://x.com/steph_palazzolo/status/1869919189240254781?s=46)发现的线索，OpenAI Shipmas 的最后一天带来了最重磅的公告：**o3 和 o3-mini** 发布了，并带来了令人惊叹的早期基准测试结果：

- **FrontierMath**：有史以来最难的数学基准测试（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-frontiermath-a-benchmark-for-evaluating/)）从 2% 提升到了 25% 的 SOTA。
  - 
![image.png](https://assets.buttondown.email/images/aa59bda9-8693-4322-ad64-e8e34684f147.png?w=960&fit=max)

- **ARC-AGI**：著名的通用推理难题基准测试，在 o3 low（$20/任务）和 o3 high（数千美元/任务）设置下，几乎呈直线延续了 o1 模型所展现的性能增长曲线。[Greg Kamradt](https://x.com/GregKamradt/status/1870208490096218244) 出现在发布会上[验证了这一点](https://x.com/arcprize/status/1870169260850573333)，并[发表了一篇博客文章](https://arcprize.org/blog/oai-o3-pub-breakthrough)分享他们对结果的看法。正如他们所说，“ARC-AGI-1 从 2020 年 GPT-3 的 0% 增长到 2024 年 GPT-4o 的 5% 用了 4 年时间”。o1 在其最高设置下将其提升至 32%，而 o3-high 则推到了 87.5%（这相当于 GPT3 到 4o 缩放曲线上约 11 年的进展）。
  - 
![image.png](https://assets.buttondown.email/images/6c076cb4-4737-405a-aa92-e775880ba13d.png?w=960&fit=max)

- **SWEBench-Verified, Codeforces, AIME, GPQA**：人们太容易忘记这些模型在 9 月之前都不存在，而 o1 直到本周二才在 API 中提供：
![image.png](https://assets.buttondown.email/images/117dd51f-8679-4540-92c0-41f85cd7b2e4.png?w=960&fit=max)


**o1-mini** 也不容忽视，[蒸馏团队自豪地展示了](https://x.com/shengjia_zhao/status/1870176031610667223)它如何拥有比 o3-full 压倒性优势的推理-智能曲线：
![image.png](https://assets.buttondown.email/images/ff63840a-8b1a-4402-beb8-55d563d3e84e.png?w=960&fit=max)


正如 [sama 所说](https://x.com/sama/status/1870266813248053426)：“在许多编程任务中，o3-mini 将以大幅降低的成本超越 o1！我预计这一趋势将继续，但同时，**以指数级的资金投入换取边际性能提升的能力将会变得非常奇怪**。”

[Eric Wallace](https://x.com/Eric_Wallace_/status/1870176920706658692) 还发布了一篇关于他们 [o 系列审议式对齐（deliberative alignment）策略](https://openai.com/index/deliberative-alignment/)的文章，[安全研究人员现在可以申请](https://openai.com/index/early-access-for-safety-testing/)进行安全测试。

社区回顾[视频](https://www.youtube.com/watch?v=YAgIh4aFawU)、[文章](https://www.interconnects.ai/p/openais-o3-the-2024-finale-of-ai)、[直播博客](https://simonwillison.net/2024/Dec/20/live-blog-the-12th-day-of-openai/)以及[架构推测](https://x.com/kalomaze/status/1870187515258208669?s=46)也值得一读。



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**OpenAI 模型发布 (o3 和 o3-mini)**

- **o3 和 o3-mini 公告与性能**：[@polynoamial](https://twitter.com/polynoamial/status/1870175700222628164) 宣布了 **o3 和 o3-mini**，强调 [o3](https://twitter.com/OpenAI/status/1870186518230511844) 在 **ARC-AGI 上达到了 75.7%**，在 **高算力下达到了 87.5%**。[@sama](https://twitter.com/sama/status/1870176283851903152) 对此次发布表示兴奋，并强调了正在进行的**安全性测试**。

- **o3 的基准测试成就**：[@dmdohan](https://twitter.com/dmdohan/status/1870182433020314004) 指出 [o3](https://twitter.com/dmdohan/status/1870178043951528077) 在 **ARC-AGI 上得分 75.7%**，[@goodside](https://twitter.com/goodside/status/1870213699341885485) 祝贺团队 [o3](https://twitter.com/goodside/status/1870213699341885485) **在 ARC-AGI 上实现了新的 SOTA**。

**其他 AI 模型发布 (Qwen2.5, Google Gemini, Anthropic Claude)**

- **Qwen2.5 技术进展**：[@huybery](https://twitter.com/huybery/status/1869952907677991200) 发布了 **Qwen2.5 技术报告**，详细介绍了在**数据质量、合成数据流水线 (synthetic data pipelines)** 以及增强**数学和编程**能力的**强化学习 (reinforcement learning)** 方法方面的改进。

- **Google Gemini Flash Thinking**：[@shane_guML](https://twitter.com/shaneguML/status/1870256503149736253) 讨论了 **Gemini Flash 2.0 Thinking**，将其描述为**快速**、**出色**且**廉价**，在**推理任务 (reasoning tasks)** 中表现优于竞争对手。

- **Anthropic Claude 更新**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1870120288601456752) 分享了关于 **Anthropic 在 AI 安全和扩展 (scaling) 方面的工作**见解，强调了他们的**负责任扩展政策 (responsible scaling policy)** 和未来方向。

**基准测试与性能指标**

- **FrontierMath 和 ARC-AGI 评分**：[@dmdohan](https://twitter.com/dmdohan/status/1870176374625054880) 强调了 **o3 在 FrontierMath 上取得 25% 的成绩**，相比之前的 **2%** 有了显著提升。此外，[@cwolferesearch](https://twitter.com/cwolferesearch/status/1870177724712572025) 展示了 **o3 在多个基准测试上的表现**，包括 **SWE-bench** 和 **GPQA**。

- **评估方法与挑战**：[@fchollet](https://twitter.com/fchollet/status/1870173777764544660) 讨论了 **Scaling Laws 的局限性**，以及**下游任务性能 (downstream task performance)** 相对于传统**测试损失指标 (test loss metrics)** 的重要性。

**AI 安全、对齐与伦理**

- **用于更安全模型的审辩式对齐 (Deliberative Alignment)**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1870177841687777685) 介绍了 **Deliberative Alignment**，这是一种旨在通过使用 **思维链推理 (chain-of-thought reasoning)** 来遵守**安全规范 (safety specifications)**，从而增强**模型安全性**的**训练方法**。

- **AI 进步的社会影响**：[@Chamath](https://twitter.com/Chamath/status/1870169387724140554) 强调需要**考虑 AI 进步带来的深远社会影响**及其对**后代**的影响。

**AI 工具、应用与研究**

- **用于增强编程的 CodeLLM**：[@bindureddy](https://twitter.com/bindureddy/status/1870218259334869327) 介绍了 **CodeLLM**，这是一款集成了 **o1**、**Sonnet 3.5** 和 **Gemini** 等多个 **LLMs** 的 **AI 代码编辑器**，为开发者提供**无限的试用配额**。

- **用于音频文件处理的 LlamaParse**：[@llama_index](https://twitter.com/llama_index/status/1870175599849025684) 宣布 **LlamaParse** 具备了**解析音频文件**的能力，将其功能扩展到无缝处理**语音转文本 (speech-to-text)** 转换。

- **用于改进算子实现的 Stream-K**：[@hyhieu226](https://twitter.com/hyhieu226/status/1870162074820849908) 展示了 **Stream-K**，它增强了 **GEMM kernels**，并为 **persistent kernels** 提供了**更好的算子实现视角**。

**梗与幽默**

- **关于 AI 和文化的幽默见解**：[@dylan522p](https://twitter.com/dylan522p/status/1870213495641256109) 幽默地表示：**“那些家伙疯狂买入 Nvidia 股票，因为 OpenAI 的 o3 实在是太他妈强了”**，将 **AI 进展**与**股市幽默**结合在一起。

- **AI 相关笑话和双关语**：[@teknium1](https://twitter.com/Teknium1/status/1870266643928260666) 推特道：**“如果有人在纽约想见面，我 4:00 到 5:30 会在 Stout，和几个朋友在一起。”**，俏皮地将**社交计划**与 **AI 讨论**结合。

- **对 AI 趋势的轻松评论**：[@saranormous](https://twitter.com/saranormous/status/1869959508253925834) 分享了关于在 X 上**发布点击诱饵内容**的幽默反思，将 **AI 内容创作**与**社交媒体幽默**结合。

**AI 研究与技术洞察**

- **混合专家模型 (MoE) 的推理成本**：[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1870167295617601695) 解释说，与**稠密模型 (dense models)** 相比，**MoE 模型**通常具有**更低的推理成本**，澄清了 **AI 架构**中的常见误解。

- **神经视频水印框架**：[@AIatMeta](https://twitter.com/AIatMeta/status/1870176422670852541) 介绍了 **Meta Video Seal**，这是一个**神经视频水印框架**，并详细说明了其在**保护视频内容**方面的应用。

- **关于 LLM 推理时自我改进的查询**：[@omarsar0](https://twitter.com/omarsar0/status/1870182483942347521) 发布了一项**关于 LLM 推理时自我改进的调查**，探讨了**增强 AI 推理能力**的**技术与挑战**。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. OpenAI 的 O3 Mini 性能超越前代**

- **OpenAI 刚刚发布了 O3 和 O3 mini** ([Score: 234, Comments: 186](https://reddit.com/r/LocalLLaMA/comments/1hiq1jg/openai_just_announced_o3_and_o3_mini/))：OpenAI 最新发布的 **O3 和 O3 mini** 模型展现了显著的性能提升，其中 **O3** 在 ARC-AGI 测试中获得了 **87.5% 的分数**，该测试旨在评估 AI 学习训练数据之外的新技能的能力。这标志着相比 **O1** 此前 **25% 到 32%** 的分数有了巨大飞跃，**Francois Chollet** 承认这一进展是“扎实的”。
  - 围绕 **ARC-AGI 基准测试** 结果存在质疑，用户对其有效性表示怀疑，原因是测试环境不公开，且该模型是在公开训练集上训练的，这与之前的版本不同。人们对 **AGI** 的说法表示担忧，强调该基准测试在证明真正的 AGI 能力方面存在局限性。
  - 使用 O3 模型实现高性能的**成本**备受关注，**87.5% 准确率**版本的成本远高于 **75.7% 准确率**版本。用户讨论了该模型目前的经济可行性，并预测成本效益可能会随着时间的推移而提高，从而使其更具可及性。
  - 值得注意的是，由于与**英国电信巨头 O2** 的商标问题，命名时跳过了“O2”，一些用户对命名惯例表示不满。此外，人们对公开发布和开源替代方案充满期待，预计将于 **1 月下旬**发布。


- **[O3 击败了 99.8% 的竞赛程序员](https://www.reddit.com/gallery/1hiqing)** ([Score: 121, Comments: 69](https://reddit.com/r/LocalLLaMA/comments/1hiqing/03_beats_998_competitive_coders/))：**O3** 在 **CodeForces** 上获得了 **2727 的 ELO 评分**，位列竞赛程序员的前 **99.8%**。更多详情请参阅 [CodeForces 博客](https://codeforces.com/blog/entry/126802)。
  - **O3 的性能与计算成本**：O3 在 CodeForces 上取得了显著成绩，ELO 评分达到 2727，但为了达到高准确率，需要生成超过 **191 亿个 token**，产生了巨额成本，例如最高配置设置下的成本达 **115 万美元**。讨论强调了目前计算成本虽然很高，但预计会随着时间的推移而下降，突显了 AI 能力的进步。
  - **AI 解决问题的挑战**：将 O3 的方法与 **CoT + MCTS** 等传统方法进行了对比，评论指出其在计算方面的效率和可扩展性，尽管它需要迭代过程来处理错误。讨论了问题的复杂性和对 In-context computation 的需求，并将 AI 的 token 生成与人类解决问题的能力进行了比较。
  - **对编程面试的影响**：O3 等模型的进步引发了关于 **LeetCode 风格面试** 相关性的辩论，一些人认为随着 AI 的改进，这类面试可能会过时。有人呼吁面试应纳入 LLM 等现代工具，并对某些技术面试问题的非现实性进行了幽默的批评。

- **[o3 图表的 X 轴是对数刻度，Y 轴是线性刻度](https://i.redd.it/s1t6d3ubk28e1.png)** ([Score: 139, Comments: 65](https://reddit.com/r/LocalLLaMA/comments/1hitwwt/the_o3_chart_is_logarithmic_on_x_axis_and_linear/))：**o3 图表**在“每任务成本”上使用了**对数 X 轴**，在“分数”上使用了**线性 Y 轴**，展示了 **O1 MIN、O1 PREVIEW、O3 LOW (Tuned)** 和 **O3 HIGH (Tuned)** 等各种模型的性能指标。值得注意的是，**O3 HIGH (Tuned)** 在较高成本下达到了 88% 的分数，而 **O1 LOW** 在 1 美元成本下仅为 25% 的分数，突显了 ARC AGI 评估中成本与性能之间的权衡。
  - 几位评论者批评 **o3 图表**因其**对数 X 轴**而具有误导性，**hyperknot** 强调该图表给人一种通往 AGI 是线性进展的错觉。**hyperknot** 进一步认为，实现 AGI 需要大幅降低成本，估计需要**降低 10,000 倍**才能使其可行。
  - 关于 AGI 成本和实用性的讨论显示出对其当前可行性的怀疑，**Uncle___Marty** 反对增加模型规模和计算能力的趋势。其他人如 **Ansible32** 则反驳说，展示功能性 AGI 是有价值的，类似于 **ITER** 等研究项目，尽管 **ForsookComparison** 质疑成本逻辑，认为高昂的费用可能并不合理。
  - 关于计算硬件进展的辩论中，**Chemical_Mode2736** 和 **mrjackspade** 讨论了降低成本和计算能力指数级提升的潜力。然而，**EstarriolOfTheEast** 指出，由于 **fp8 或 fp4** 的假设以及功耗需求的增加，最近的进展可能并不像看起来那么显著，暗示指数级提升正在放缓。


**主题 2. Qwen QVQ-72B：AI 建模的新前沿**

- **Qwen QVQ-72B-Preview 即将到来！！！** ([Score: 295, Comments: 48](https://reddit.com/r/LocalLLaMA/comments/1hi8d8c/qwen_qvq72bpreview_is_coming/))：**Qwen QVQ-72B** 是一个拥有 720 亿参数的模型，其预发布占位符现已在 [ModelScope](https://modelscope.cn/models/Qwen/QVQ-72B-Preview) 上线。关于命名规范从 **QwQ** 更改为 **QvQ** 存在一些不确定性，目前尚不清楚它是否包含特定的推理能力。
  - **Qwen QVQ-72B** 模型被推测包含**视觉/视频能力**，正如 **Justin Lin** 的 Twitter 帖子所指出的，暗示 QVQ 中的“V”代表 Vision（视觉）。**ModelScope** 上有一个占位符，但在创建后不久可能已被设为私有或删除。
  - 讨论强调了模型的**内部思考过程**，并将 **QwQ** 与 Google 的模型进行了比较。Google 的模型因其推理的效率和透明度而受到称赞，相比之下，QwQ 倾向于冗长且在思考过程中可能具有“对抗性”，由于 Token 生成速度慢，在 CPU 上运行时可能会很繁琐。
  - 讨论了**开源贡献**的潜力，Google 不隐藏模型推理过程的决定被认为对竞争对手和本地 LLM 社区都有利。这种透明度与 **OpenAI** 的方法形成对比，后者不公开推理过程，可能在推理时使用了 **MCTS** 等技术。


- **[Qwen 发布了他们的 Qwen2.5 技术报告](https://arxiv.org/pdf/2412.15115)** ([Score: 175, Comments: 11](https://reddit.com/r/LocalLLaMA/comments/1hie4c9/qwen_have_released_their_qwen25_technical_report/))：**Qwen** 发布了他们的 **Qwen2.5 技术报告**，尽管帖子中没有提供额外的信息或细节。
  - **Qwen2.5 的编程能力**：用户对 **Qwen2.5-Coder** 模型在没有明确指令的情况下实现复杂功能（如 **Levenshtein 距离方法**）的能力印象深刻。该模型受益于一个用于静态代码检查和单元测试的全能多语言沙箱，这增强了近 40 种编程语言的代码质量和正确性。
  - **技术报告 vs. 白皮书**：使用“技术报告”而非“白皮书”一词，是因为它允许分享某些方法论，同时将模型架构和数据等其他细节作为商业机密保留。这种区别对于理解此类文档中分享的透明度和信息水平至关重要。
  - **模型训练与性能**：该模型的效能，特别是在编程任务中，归功于其在来自 **GitHub** 和代码相关问答网站的数据集上的训练。即使是 14b 模型在建议和实现算法方面也表现出强大的性能，预计 72b 模型将更加强大。


**主题 3. RWKV-7 在多语言和长上下文处理方面的进展**

- **RWKV-7 0.1B (L12-D768) 在 ctx4k 下训练，解决了 NIAH 16k，外推至 32k+，100% RNN（无 attention），支持 100+ 语言和代码** ([Score: 117, Comments: 16](https://reddit.com/r/LocalLLaMA/comments/1hiigah/rwkv7_01b_l12d768_trained_w_ctx4k_solves_niah_16k/)): **RWKV-7 0.1B (L12-D768)** 是一款无 attention、100% RNN 模型，擅长处理长上下文任务，并支持 100 多种语言和代码。该模型在包含 **1 万亿 tokens** 的多语言数据集上进行训练，在处理长上下文方面优于 **SSM (Mamba1/Mamba2)** 和 **RWKV-6** 等其他模型，并使用 in-context gradient descent 进行 test-time-training。RWKV 社区还开发了一个微型的 **RWKV-6** 模型，能够通过广泛的 chain-of-thought 推理解决数独等复杂问题，且无论上下文长度如何，都能保持恒定的速度和 VRAM 占用。
  - **RWKV 的未来潜力**：爱好者们对 RWKV 模型的潜力表示兴奋，特别是它们在推理任务中超越带有 attention 层的传统 Transformer 模型的能力。社区期待在 **1B 参数** 规模之外的进展，以及发布像 **3B 模型** 这样更大的模型。
  - **学习资源**：对学习 RWKV 的全面资源有需求，这表明了对其架构和应用理解的兴趣。
  - **研究与开发**：一位用户分享了尝试创建 RWKV 图像生成模型的经验，强调了该模型的能力以及为进一步优化它而进行的持续研究工作。讨论中引用了一篇相关论文：[arxiv.org/pdf/2404.04478](https://arxiv.org/pdf/2404.04478)。


**主题 4. 开源 AI：必然的演进**

- **为什么开源 AI 不仅是必要的，而且需要进化的真正原因** ([Score: 57, Comments: 25](https://reddit.com/r/LocalLLaMA/comments/1hifs2d/the_real_reason_why_not_only_is_opensource_ai/)): 作者批评了 **OpenAI** 对其 **o1 模型** 的定价策略，强调了与基础价格和不可见输出 tokens 相关的高昂成本，认为这相当于垄断行为。他们主张 **开源 AI** 和社区协作，以防止垄断行为并确保竞争带来的好处，并指出像 **Google** 这样的公司可能会提供较低的价格，但并非出于善意。
  - **垄断担忧**：评论者一致认为 AI 领域可能会出现垄断行为，正如在其他行业中看到的那样，早期进入者会推动监管以维持其市场主导地位。**OpenAI** 的定价策略被视为反消费者，类似于 **Apple** 等公司为排他性收取溢价的做法。
  - **不可见输出 Tokens**：关于“不可见”输出 tokens 相关成本的讨论，批评者认为将这些作为大型模型的一部分进行收费是不公平的。一些人认为用户应该能够看到这些 tokens，因为他们为此付了费。
  - **开源 vs. 科技巨头**：人们相信开源模型可以促进价格竞争，类似于 render farms 在渲染领域的运作方式。开源社区与小型公司之间的协作被视为挑战 **OpenAI** 和 **Google** 等巨头主导地位的潜在途径。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. OpenAI 的 O3：高 ARC-AGI 性能但高成本**

- **[OpenAI 的新模型 o3 在全球最难数学基准测试中展现出巨大飞跃](https://i.redd.it/ng3c9j1up18e1.png)** ([Score: 196, Comments: 80](https://reddit.com/r/OpenAI/comments/1hiq4yv/openais_new_model_o3_shows_a_huge_leap_in_the/)): **OpenAI 的新模型 o3** 在 **ARC-AGI 数学基准测试**中展示了显著进展，准确率达到 **25.2%**，而之前的 **state-of-the-art 模型仅为 2.0%**。这一性能飞跃突显了 o3 在解决复杂数学问题方面的进步。
  - **关于 AI 在研究中作用的讨论**：**Ormusn2o** 强调了像 **o3** 这样的 AI 模型在推进自主和辅助机器学习研究方面的潜力，这对于实现 **AGI** 可能至关重要。同时，**ColonelStoic** 讨论了当前 **LLM** 在处理复杂数学证明方面的局限性，建议与 **Lean** 等自动证明检查器集成以寻求改进。
  - **关于基准测试和模型性能的澄清**：**FateOfMuffins** 指出了对基准测试的误解，澄清 **25% 的准确率** 属于 **ASI 数学基准测试**，不能直接与研究生水平的人类表现相提并论。**Elliotglazer** 进一步解释了 **FrontierMath** 内部的分层难度级别，指出性能跨越了不同的问题复杂度。
  - **模型评估与利用**：**Craygen9** 对评估模型在各个专业领域的表现表示兴趣，主张开发针对数学、编程和医学等特定领域量身定制的模型。**Marcmar11** 和 **DazerHD1** 讨论了性能指标，强调了基于思考时间的模型性能差异，**深蓝色**表示低思考时间，**浅蓝色**表示高思考时间。


- **[2025 年将很有趣——谷歌在 12 月之前一直是个笑话，现在我觉得 2025 年对谷歌来说会非常好](https://i.redd.it/xq4oki04h08e1.jpeg)** ([Score: 118, Comments: 26](https://reddit.com/r/OpenAI/comments/1hikl5y/year_2025_will_be_interesting_google_was_joke/)): **Logan Kilpatrick** 对 2025 年 AI 编程模型的重大进展表示乐观，获得了 2400 个赞的广泛关注。**Alex Albert** 持怀疑态度回应，对这些进展表示不确定，他的回复也吸引了 639 个赞。
  - **OpenAI vs. Google**：评论者讨论了 **OpenAI** 相比 **Google** 因公司限制而具有的灵活性，认为两家公司现在处于更平等的地位。一些人对 Google 改进其 AI 产品的能力表示怀疑，特别是对其搜索功能和潜在广告技术干扰的担忧。
  - **Gemini 模型**：**Gemini 模型**被强调为一项重大进步，一位用户指出其性能优于之前的模型，如 4o 和 3.5 sonnet。关于其能力的争论仍在继续，特别是其对文本、图像和音频的原生多模态支持。
  - **企业影响**：人们普遍对 Google 对 AI 进步的影响持不信任态度，担心业务和广告部门到 2025 年可能对 **Gemini 模型**产生负面影响。用户对 AI 领域的未来发展表达了怀疑与期待并存的情绪。


- **OpenAI o3 在 ARC-AGI 上的表现** ([Score: 138, Comments: 88](https://reddit.com/r/OpenAI/comments/1hiptxb/openai_o3_performance_on_arcagi/)): 该帖子链接到一张图片，但正文中未提供关于 **o3 表现**在 **ARC-AGI** 上的具体细节或背景。
  - 讨论强调了 **o3** 在 **ARC-AGI 基准测试**上的显著性能提升。**RedGambitt_** 强调 **o3** 代表了 AI 能力的飞跃，修复了 **LLM paradigm** 中的局限性，需要更新对 AI 的直觉。尽管性能很高，但 **o3** 并不被视为 AGI，正如 **phil917** 所指出的，他引用 ARC-AGI 博客称 **o3** 在简单任务上仍然失败，且 **ARC-AGI-2** 将带来新的挑战。
  - 使用 **o3** 的成本是一个主要担忧，**daemeh** 和 **ReadySetPunish** 指出 **o3(low)** 的价格约为每项任务 **20 美元**，而 **o3(high)** 则高达 **3500 美元**。**Phil917** 提到，高算力变体处理 100 个问题可能耗资约 **350,000 美元**，突显了大规模使用的昂贵成本。
  - 对话中包含对 **AGI** 的怀疑，**hixon4** 和 **phil917** 指出通过 **ARC-AGI** 并不等同于实现 AGI。讨论了 **o3** 的高成本和局限性，**phil917** 指出结果中可能存在数据污染，因为模型是在基准数据上训练的，这削弱了 **o3** 分数的令人印象深刻程度。


**主题 2：在 o3 的热度中，Google 的 Gemini 2.5 盖过竞争对手**

- **[他赢了，伙计们](https://i.redd.it/u8esm1els18e1.png)** ([Score: 117, Comments: 25](https://reddit.com/r/OpenAI/comments/1hiqgov/he_won_guys/)): **Gary Marcus** 预测到 **2024** 年底将出现 **7-10 个 GPT-4 级别的模型**，但不会有像 **GPT-5** 这样重大的突破，这将导致价格战和极小的竞争优势。他强调了 AI 幻觉的持续问题，并预计企业采用率和利润仅会有小幅增长。
  - 讨论强调了对 **Gary Marcus** 预测的怀疑，用户质疑他预测的可信度，并认为 **OpenAI** 目前领先于 **Google**。然而，一些人认为 **Google** 仍可能在即将推出的模型中实现 **Chain of Thought (CoT)** 能力的突破。
  - 关于 **OpenAI** 的 **o3** 模型发布及其影响存在争论，一些用户指出其可用性和定价可能会限制其普及。虽然 **o3-mini** 预计在 1 月底推出，但对于这些发布的及时性和公众访问权限仍存疑。
  - 用户讨论了新型推理模型在 **自动化工作流 (automated workflows)** 中的效率和潜在成本优势，并将其与 **GPT-4** 等早期模型的复杂性和资源需求进行了对比。这些进步被视为驱动自动化系统的更智能解决方案。


**主题 3. TinyBox GPU 操作与网络欺骗**

- **我不希望因为价格太高而用不起 AI** ([Score: 126, Comments: 91](https://reddit.com/r/OpenAI/comments/1hidjmj/i_would_hate_to_be_priced_out_of_ai/)): 该帖子讨论了对 **AI 服务** 成本上升的担忧，特别是 **O1 无限制** 方案已经达到 **每月 200 美元**，以及未来 **Agentic AI** 可能达到 **每月 2,000 美元** 的定价。作者对因价格问题无法使用高质量 AI 表示沮丧，同时也承认了这些成本可能的合理性，引发了对 AI 技术更广泛定价轨迹的反思。
  - 许多人强烈认为 **开源 AI** 对于抵消专有 AI 解决方案的高昂成本至关重要，正如 **GBJI** 所表达的，他主张支持 FOSS AI 开发者以对抗企业控制。**Odd_Category_1038** 指出，担忧在于高昂的定价可能会造成全球智能的瓶颈，使美国/欧盟以外的研究人员处于劣势并抑制创新。
  - **LegitimateLength1916** 和 **BlueberryFew613** 讨论了 AI Agent 可能取代工人的经济影响，前者认为由于成本节约，企业将选择 AI 而非人类员工。然而，**BlueberryFew613** 个人认为目前的 AI 缺乏完全取代专业人士的能力和基础设施，强调需要符号推理和 AI 集成方面的进步。
  - **NoWeather1702** 对 AI 的可扩展性表示担忧，原因是能源和算力不足，并指出 LLM 所需的电力/算力增长速度超过了生产速度。在全求数据中心行业工作的 **ThenExtension9196** 保证，目前正在努力解决这一问题。


**主题 4. ChatGPT Pro 定价与市场影响讨论**

- **[OpenAI 会发布 2000 美元的订阅服务吗？](https://i.redd.it/ohxllvp2az7e1.jpeg)** ([Score: 349, Comments: 144](https://reddit.com/r/OpenAI/comments/1higq81/will_openai_release_2000_subscription/)): 该帖子推测了 **OpenAI** 可能推出的 **2000 美元订阅服务**，引用了 **Sam Altman** 在 **2024 年 12 月 20 日** 发布的一条俏皮的 Twitter 帖子。该帖子幽默地暗示了序列 "ooo -> 000 -> 2000" 与 Altman 推文之间的联系，推文中包含了一些随意且幽默的互动指标。
  - **O3 模型推测**：关于可能作为 **O1** 后继者的新模型 **o3** 存在讨论。这种推测的出现是因为 **O2** 已经是欧洲一家注册商标的电话运营商，一些用户幽默地建议它可能会为不同的订阅层级提供每周限制的消息数。
  - **定价与价值担忧**：评论者对传闻中的 **2000 美元/月** 订阅表示怀疑，开玩笑说这样的价格应该配得上 **AGI** (通用人工智能)，而他们认为 AGI 的价值远不止于此。
  - **幽默与讽刺**：评论充满了幽默感，提到了潜在的 **NSFW 伴侣模型**，以及与 **Ozempic** 和 **OnlyFans** 的俏皮关联。还有人对营销策略进行了讽刺，使用了 "ho ho ho" 和 "oh oh oh" 等词组。


---

# AI Discord 摘要

> 由 o1-2024-12-17 生成的摘要的摘要的摘要

**主题 1. O3 热潮与新基准测试**

- [**O3 突破 ARC-AGI**](https://x.com/arcprize/status/1870169260850573333)：OpenAI 的 O3 模型在 ARC-AGI 半私有评估中达到了 75.7%，并在高算力模式下飙升至 87.5%。工程师们对其“超水平发挥”的推理能力表示赞赏，尽管批评者担心该模型巨大的推理成本。
- [**高算力模式极其烧钱**](https://x.com/fchollet/status/1870169764762710376)：部分评估每次运行耗资数千美元，这表明大公司可以以高昂的价格推高性能。小型机构担心算力壁垒，并怀疑 O3 这种马斯克级别的预算会让许多人无法触及 SOTA 级别的进展。
- [**O2 缺席，O3 快速登场**](https://techcrunch.com/2024/12/20/openai-announces-new-o3-model/)：据传因商标冲突，OpenAI 跳过了 “O2”，在 O1 发布仅几个月后就推出了 O3。撇开命名方面的玩笑不谈，开发者们对从一个前沿模型到下一个模型的惊人进展速度感到惊叹。

**主题 2. AI 编辑器狂热：Codeium, Cursor, Aider 等**

- [**Cursor 0.44.5 提升生产力**](https://www.cursor.com/downloads)：用户称赞新版本的 Agent 模式既快又稳定，促使不少用户从竞争对手的 IDE 回归 Cursor。新一轮以 25 亿美元估值融资 1 亿美元的消息，为其灵活的代码环境增添了更多热度。
- [**Codeium “发送至 Cascade” 功能简化 Bug 报告**](https://x.com/windsurf_ai/status/1870268007995585000)：Codeium 的 Windsurf 1.1.1 更新引入了一个将问题直接转发到 Cascade 的按钮，消除了调试过程中的阻碍。成员们成功测试了更大的图像和传统聊天模式，并参考了文档中的方案使用详情。
- [**Aider 与 Cline 协同处理仓库**](https://aider.chat/docs/usage/tutorials.html)：Aider 处理细微的代码调整，而 Cline 凭借扩展内存功能完成更大的自动化任务。开发者们看到了更高效的工作流，减少了重复性工作，且这两个工具之间具有互补的协同效应。

**主题 3. 微调之争：LoRA, QLoRA 与剪枝**

- [**LoRA 引发热议**](https://arxiv.org/pdf/2410.21228)：批评者质疑 LoRA 在分布外（out-of-distribution）数据上的有效性，而其他人则坚持认为它是超大规模模型的必备方案。一些人建议采用全量微调（full finetuning）以获得一致的结果，引发了关于训练方式的持久争论。
- [**QAT + LoRA 登陆 Torchtune v0.5.0**](https://github.com/pytorch/torchtune/releases/tag/v0.5.0)：新方案将量化感知训练（quantization-aware training）与 LoRA 结合，以创建更精简、更专业的 LLM。早期采用者非常喜欢更小的文件体积与不错的性能提升之间的平衡。
- [**词表剪枝颇为棘手**](https://github.com/pytorch/torchtune#optimization-flags)：一些开发者通过剪枝不需要的 Token 来减少内存占用，但保留 fp32 参数以维持精度。这种平衡行为凸显了大规模训练边缘案例模型时的复杂现实。

**主题 4. Agent、RL 方法及竞争模型对决**

- [**HL Chat：Anthropic 的惊喜与构建 Anthropic**](https://www.youtube.com/watch?v=om2lIWXLLN4)：粉丝们猜测可能会有节日发布，并注意到了团队充满热情的氛围。关于 Dario “可爱顽童”气质的玩笑，衬托出了 Agent 发布周边的轻松基调。
- [**无需完全验证的 RL**](https://x.com/natolambert/status/1870150741593129045)：一些团队推测，当任务缺乏完美检查器时，奖励模型会出现反复，并建议使用“松散验证器”或更简单的二元启发式方法。他们预计到 2025 年 RL+LLM 将迎来更大的里程碑，用尚不成熟的奖励信号连接不确定的输出。
- [**Gemini 2.0 Flash Thinking 对抗 O1**](https://venturebeat.com/ai/google-unveils-new-reasoning-model-gemini-2-0-flash-thinking-to-rival-openai-o1/)：Google 的新模型公开展示了思考 Token，让开发者可以看到逐步的逻辑。观察者称赞其透明度，但也质疑 O3 目前在代码和数学任务上是否比 Gemini 更出色。

**主题 5. 创意与多媒体 AI：Notebook LM, SDXL 等**

- [**Notebook LM 批量产出播客**](https://www.youtube.com/@AI_ForThePeople)：学生和创作者使用 AI 自动化生成具有一致音频质量的整个节目片段。该工具还有助于为新闻报道或学术写作构建时间线和思维导图，展示了灵活的内容生成能力。
- [**SDXL + LoRA 打造动漫场景**](https://civitai.com/models/555285/miyabi-hoshimi-zenless-zone-zero)：艺术家们称赞 SDXL 强大的风格，同时通过 LoRA 增强动漫艺术效果。用户克服了风格不匹配的问题，为游戏场景和角色设计保留了配色方案。
- [**AniDoc 帧上色宛如魔法**](https://x.com/Gradio/status/1870017358821015670)：Gradio 的 AniDoc 将粗糙的草图转换为全彩动画，优雅地处理姿势和比例。开发者称赞它是加速视觉叙事和原型设计的强大扩展。

---

# PART 1: High level Discord summaries

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 1.1.1 在定价和图像升级方面表现出色**：**Windsurf 1.1.1** 更新引入了 **'Send to Cascade'** 按钮、方案状态的使用信息，并取消了图像 1MB 的限制，详见 [changelog](https://www.codeium.com/changelog)。
   - 社区成员测试了 **'Legacy Chat'** 模式并称赞了新的 Python 增强功能，参考了 [使用文档](https://docs.codeium.com/windsurf/usage#viewing-your-usage) 中的细节。
- **Send to Cascade 展示了快速的问题路由**：一段简短的演示重点展示了 **'Send to Cascade'** 功能，允许用户将问题上报给 **Cascade**，该演示发布在 [推文](https://x.com/windsurf_ai/status/1870268007995585000) 中。
   - 贡献者鼓励大家尝试该功能，并指出将用户反馈与专门的故障排除迅速结合带来的便利性。
- **Cascade 错误导致聊天重置**：当聊天内容过长时，用户在 **Cascade** 中遇到了内部错误消息，促使他们开启新会话以维持稳定性。
   - 他们强调了简洁的对话管理对维持性能的重要性，并指出了较小聊天日志带来的好处。
- **订阅方案困扰部分成员**：一位用户询问为何 **Windsurf** 的 **trial pro plan** 被停止，引发了关于免费与分级功能的讨论，并引用了 [方案设置](https://codeium.com/plan)。
   - 其他人交流了关于使用限制的经验，强调了 **extension**、**Cascade** 和 **Windsurf** 软件包之间的差异。
- **CLI 插件和性能表现引发辩论**：一些参与者请求更好地集成 **Warp** 或 Gemini 等外部工具，同时注意到一天中不同时间的性能波动。
   - 他们强调了命令行界面（CLI）使用与 AI 驱动编程之间潜在的协同效应，尽管对大型代码库中速度变慢的担忧依然存在。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 0.44.5 提升生产力并吸引融资**：开发者报告称 [Cursor 0.44.5 版本](https://www.cursor.com/downloads) 显示出显著的性能提升，特别是在 Agent 模式下，促使许多人从竞争对手编辑器切回 Cursor。
   - TechCrunch 披露了 Cursor 以 25 亿美元估值完成的新一轮 1 亿美元融资，表明投资者对 AI 驱动的编程解决方案有着强烈的热情。
- **AI 工具助力开发工作**：参与者强调了 AI 驱动的功能如何减少编码时间并扩大解决方案的搜索范围，使他们能够更高效地完成项目。
   - 他们注意到了与 [构建高效 Agent](https://www.anthropic.com/research/building-effective-agents) 等教程中额外指导的协同作用，这些教程确保了 LLM 在工作流中的实际集成。
- **Sonnet 模型引发褒贬不一的反馈**：用户对比了多个 Sonnet 版本，一些人称赞最新版本的 UI 生成能力，而另一些人则报告输出质量不稳定。
   - 他们观察到 System Prompts 会显著影响模型的行为，导致某些开发者调整其方法以获得更好的结果。
- **自由职业者拥抱 AI 以实现更快交付**：自由职业贡献者分享了使用 AI 自动化繁琐编码任务并更迅速地清理项目积压工作的案例。
   - 少数人对客户对 AI 使用的怀疑表示担忧，但鉴于成果的改善，整体情绪保持积极。
- **AI 创建的布局中 UI 样式挑战依然存在**：虽然 AI 能有效处理后端逻辑，但在精细的样式元素上表现不佳，迫使开发者手动修复前端设计问题。
   - 这一不足强调了对视觉组件进行更多数据训练的需求，这可以增强工具生成精美界面的能力。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **OpenAI O3 提速**：基准测试显示 **OpenAI O3** 在 ARC-AGI 半私有评估中达到了 75.7%，如[此推文](https://x.com/OpenAI/status/1870164871289155937)所述。
   - **ARC Prize** 的后续帖子提到，高计算量的 O3 版本得分高达 87.5%，引发了关于成本和性能改进的讨论。
- **Aider 与 Cline 联手**：开发者使用 **Aider** 进行较小的代码调整，而 **Cline** 凭借其更强的记忆能力处理更繁重的自动化任务。
   - 他们观察到通过配对使用这些工具可以提升工作流，减少软件开发中的手动重复。
- **AI 职业安全担忧增加**：评论者表示担心 **AI** 可能通过自动化简单任务来取代部分编码角色。
   - 其他人则坚持认为，对于复杂的问题解决，人的因素仍然是关键，因此开发者职位应该保持其重要性。
- **Depth AI 提升代码洞察力**：工程师在大型代码库上测试了 **Depth AI**，注意到其在 [trydepth.ai](https://www.trydepth.ai) 上的完整知识图谱和跨平台集成。
   - 一位用户在不再需要检索增强生成（RAG）时停止了使用，但仍对其潜力表示赞赏。
- **AniDoc 轻松为草图上色**：新的 [AniDoc 工具](https://x.com/Gradio/status/1870017358821015670)可根据风格参考将粗糙的帧转换为全彩动画。
   - 用户赞赏其处理各种姿势和比例的能力，称其为视觉叙事的有效扩展。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **O3 在 ARC-AGI 上的强劲表现**：OpenAI 透露 **O3** 在 ARC-AGI 测试中得分 **87.5%**，跳过了 O2 这个名称，在三个月内从 **O1** 进化到 **O3**，如[此推文](https://x.com/arcprize/status/1870169260850573333)所示。
   - 社区成员就高昂的推理成本和 GPU 使用情况展开争论，有人开玩笑说 **Nvidia** 股票飙升是因为 **O3** 的强劲结果。
- **LoRA 的收益有限**：一位用户对 **LoRA finetuning** 提出质疑，引用了一篇[分析论文](https://arxiv.org/pdf/2410.21228)，该论文怀疑 LoRA 在训练集之外的有效性。
   - 其他人强调，随着模型变大，**LoRA** 变得必不可少，这引发了关于**全量微调 (full finetuning)** 是否能产生更一致结果的辩论。
- **Chollet 将 O1 称为下一个 AlphaGo**：François Chollet 将 **O1** 比作 **AlphaGo**，在[此帖子](https://fxtwitter.com/fchollet/status/1869854758443557020)中解释说，两者在单次移动或输出中都使用了大量的处理过程。
   - 他坚持认为将 **O1** 标记为简单的语言模型具有误导性，这促使成员们质疑 **O1** 是否秘密使用了类搜索方法。
- **RL & RLHF 奖励模型的挑战**：一些成员认为，输出不确定的 **Reinforcement Learning** 需要专门的奖励标准，建议为简单任务使用宽松的验证器，并链接到[此讨论](https://x.com/natolambert/status/1870150741593129045)。
   - 他们警告奖励模型中存在**噪声**，强调在**美学**等领域推动二元检查，并预测 2025 年将有更大的 **RL + LLM** 突破。
- **Anthropic 的惊喜发布与构建 Anthropic Chat**：**Anthropic** 可能在假期发布产品引发了猜测，尽管一位成员开玩笑说 Anthropic 太有礼貌了，不会突然发布产品。
   - 在关于 **Building Anthropic** 的 [YouTube 视频](https://www.youtube.com/watch?v=om2lIWXLLN4)中，参与者戏称 Dario 为“可爱的小矮人”，并赞扬了团队积极向上的氛围。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 第 12 日收官活动引发观众热议**：**12 Days of OpenAI** 的最后一天由 **Sam Altman**、**Mark Chen** 和 **Hongyu Ren** 主持，观众被引导至[此处观看直播活动](https://www.youtube.com/live/SKBG1sqdyIU?si=jNf3LeuU7ctHFMJU)。
   - 许多人期待这些关键人物能带来总结性的见解和潜在的重大发布。
- **o3 模型热潮引发对比**：参与者推测 **o3** 可能与 Google 的 **Gemini** 竞争，而 **OpenAI** 的定价也引发了对其市场优势的质疑。
   - 一条 [推文](https://x.com/deedydas/status/1870175212328608232) 强调了 **o3** 在全球编程基准测试中排名第 175 位，进一步提升了关注度。
- **OpenAI 的发展方向引发褒贬不一的反应**：一些人对 **OpenAI** 偏离开源初衷、转向付费服务表示不满，理由是免费资料减少。
   - 评论者怀疑在这种定价结构下，未来发布的模型是否还能保持易用性。
- **聊天机器人查询与 4o 限制**：一位用户指出 **自定义 GPTs** 被锁定在 **4o**，限制了模型的灵活性。
   - 开发者们还在寻求关于构建机器人的建议，旨在解释软件功能并以通俗易懂的语言引导用户。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **o3 的进展与质疑并存**：新款 **o3** 在 [ARC-AGI 公开排行榜](https://arcprize.org/blog/oai-o3-pub-breakthrough) 上飙升至 **75.7%**，引发了关于它是否使用了新模型、优化的数据策略以及大规模算力的讨论。
   - 一些人称结果“很有趣”，但质疑 **o1** 加上微调（fine-tuning）技巧是否就能解释这种提升，并指出官方发布的内容可能存在疏漏。
- **FrontierMath 惊人的准确率**：根据 [David Dohan 的推文](https://x.com/dmdohan/status/1870176374625054880)，新的 **FrontierMath** 结果从 **2%** 跃升至 **25%**，挑战了此前关于高级数学任务的假设。
   - 社区成员引用 **Terence Tao** 的话称，该数据集在未来几年内都不应被 AI 攻克，而另一些人则担心潜在的过拟合（overfitting）或数据泄露问题。
- **RAG 与 Kaggle 加速微调**：通过利用 GitHub 资源，**RAG** 训练时间从 3 小时缩短至 **15 分钟**，将 7.5 万行的 CSV 从 JSON 转换后显著提升了模型准确率。
   - 一些人建议使用 **Kaggle** 获取每周 30 小时的免费 GPU 额度，并鼓励在 **Llama** 微调时关注数据质量而非单纯的数量。
- **SDXL 与 LoRA 联手打造动漫效果**：用户称赞 **SDXL** 在动漫效果上的强劲表现，并指出 [Miyabi Hoshimi 的 LoRA 模型](https://civitai.com/models/555285/miyabi-hoshimi-zenless-zone-zero) 可以提升风格准确度。
   - 其他人报告了将 **Flux** 与 LoRA 配对以获得一致动漫输出的困难，期待 **Unsloth** 尽快支持 Flux。
- **TGI 与 vLLM 的对决**：**TGI** 和 **vLLM** 在速度和适配器（adapter）处理方面引发了辩论，参考了 [Text Generation Inference 文档](https://huggingface.co/docs/text-generation-inference/en/index)。
   - 一些人因其灵活的方法而更青睐 **vLLM**，而另一些人则支持 **TGI**，认为它在服务大规模模型部署方面更为可靠。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **O3 耗资巨大，超越 O1**：新发布的 **O3** 模型在编程任务中表现优于 **O1**，且据[这条推文](https://x.com/fchollet/status/1870169764762710376)指出，其计算费用高达 **$1,600,250**。
   - 爱好者们指出了巨大的财务障碍，认为高昂的成本可能会限制其广泛应用。
- **Gemini 2.0 展开华丽对决**：Google 推出了 **Gemini 2.0 Flash Thinking** 以对抗 **OpenAI 的 O1**，据[这篇文章](https://venturebeat.com/ai/google-unveils-new-reasoning-model-gemini-2-0-flash-thinking-to-rival-openai-o1/)报道，该模型允许用户查看逐步推理过程。
   - 观察者将其与 O1 进行了对比，强调了新的下拉式解释功能是迈向透明模型内省的重要一步。
- **Llama 3.3 过于积极的函数调用**：成员们注意到 **Llama 3.3** 触发函数调用的速度远快于 **Hermes 3 70b**，这可能会推高成本。
   - 他们发现 **Hermes** 在调用方面更加克制，从而降低了费用并提高了整体一致性。
- **潜意识提示引发好奇**：有人提出在 Prompt 中进行**潜层影响注入（latent influence injecting）**，这与微妙的 NLP 风格干预有异曲同工之妙。
   - 参与者讨论了在不直接引用的情况下塑造输出的可能性，将其比作幕后建议。
- **利用 <think> 标签数据集进行宏大构想**：一项旨在利用 **<think>** 标签构建推理数据集的协作工作已经启动，目标模型包括 **O1-Preview** 或 **O3**。
   - 贡献者旨在将完整的推理轨迹嵌入原始数据中以提高清晰度，寻求结构化思维与最终答案之间的协同作用。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Mistletokens 带来的圣诞狂欢**：Bolt 团队推出了 [**Mistletokens**](https://x.com/stackblitz/status/1870203756995911707)，在年底前为 Pro 用户提供 200 万个免费 token，为免费用户提供每日 20 万个及每月 200 万个的限额。
   - 他们旨在通过这些扩展的节日 token 福利激发更多的季节性项目和解决方案。
- **Bolt 与冗余作斗争**：开发者抱怨 **Bolt** 在不清理重复内容的情况下消耗 token，并提到“开启 diff 时存在大量重复”。
   - 一些人通过针对性的审查克服了这个问题，例如“请对 [我的应用程序的 Auth Flow] 进行彻底的审查和审计”，这迫使它处理冗余问题。
- **集成 Bug 引发挫败感**：多位用户注意到 **Bolt** 会自动创建新的 Supabase 实例而不是复用旧实例，导致 token 浪费。
   - 重复的速率限制（rate-limits）引发了更多投诉，用户坚持认为购买的 token 应该让他们免受免费计划的限制。
- **WebRTC 梦想与实时流媒体**：在 **Bolt** 上为视频聊天应用集成 WebRTC 的尝试遇到了围绕实时功能的各种技术困难。
   - 社区成员请求提供具有可定制配置的预构建 WebRTC 解决方案，以便更顺畅地处理媒体。
- **订阅纠葛与店面展示**：许多人对于需要激活订阅才能使用购买的 token 充值感到担忧，敦促制定更清晰的支付指南。
   - 与此同时，一位开发者预览了一个全栈电子商务项目，该项目具有 Headless 后端、精美的店面和旨在独立运行的视觉编辑器。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **OpenAI 诽谤风波**：一段相关的 [YouTube 视频](https://www.youtube.com/watch?v=znDyEJDzrCs)展示了针对 **OpenAI** 的法律威胁，指控该 AI 对特定个人发表了诽谤性言论。
   - 成员们讨论了**在公开网络数据上进行训练**如何产生错误的归因，并对最终输出中的名称过滤器（name filters）表示担忧。
- **LM Studio 的命名功能**：参与者注意到 **LM Studio** 会自动生成对话名称，可能是通过使用内置的小型模型来总结对话内容。
   - 有人推测其中嵌入了一个**捆绑的摘要生成器（bundled summarizer）**，使对话交互更加无缝且用户友好。
- **3090 轻松运行 16B 模型**：工程师们确认，配备 **64 GB RAM** 的 **3090 GPU** 加上 **5800X** 处理器，可以以舒适的 Token 速度处理 **16B** 参数模型。
   - 他们提到 **70B** 模型仍然需要更高的 **VRAM** 和明智的量化策略来维持实用的性能。
- **参数量化见解**：爱好者们解释说，对于许多模型，**Q8** 量化通常几乎是无损的，而 **Q6** 仍能保持不错的精度。
   - 他们强调了更小的文件体积与模型准确性之间的权衡，主张采用平衡的方法以获得最佳效果。
- **eGPU 强力方案**：一位成员展示了使用 **Razer Core X** 外置显卡盒搭配 **3090**，通过 Thunderbolt 接口为 i7 笔记本电脑提供动力。
   - 这一配置激发了人们对外部 GPU 的兴趣，认为它是那些希望在便携系统上获得桌面级性能用户的灵活选择。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.0 Flash Thinking 动态**：Google 推出了新的 **Gemini 2.0 Flash Thinking** 模型，该模型直接在文本中输出思考 Token（thinking tokens），现在已可在 [OpenRouter](https://openrouter.ai/google/gemini-2.0-flash-thinking-exp:free) 上访问。
   - 部分用户暂时无法使用，但如果你热衷于实验，可以通过 Discord 申请访问权限。
- **BYOK 与费用讨论成为焦点**：**BYOK**（自带 API 密钥）的推出允许用户将自己的供应商额度与 OpenRouter 的额度合并，在上游成本之上收取 **5%** 的费用。
   - 有用户要求提供一个简单的示例来澄清费用结构，更新后的文档将详细说明**使用费**如何结合供应商费率以及这部分额外分成。
- **AI 待办事项列表应用“5分钟法则”**：一个基于 [Open Router](https://lists.new/) 构建的 **AI To-Do List** 利用“5分钟法则”自动启动任务。
   - 它还会递归地创建新任务，让用户感叹“*工作实际上变得很有趣*”。
- **新模型发布与 AGI 争议**：社区传闻 **o3-mini** 和 **o3** 即将推出，命名冲突引发了一些内部笑话。
   - 关于 **AGI** 的辩论发生了转向，一些人称该话题为“红鲱鱼”（伪命题），并将好奇者引向一段 [1.5 小时的视频讨论](https://youtube.com/watch?v=duQukAv_lPY)。
- **加密支付 API 激发资金流**：新的 **Crypto Payments API** 允许 LLM 通过 **ETH**、**0xPolygon** 和 **Base** 处理链上交易，详见 [OpenRouter 的推文](https://x.com/OpenRouterAI/status/1870227171324666130)。
   - 它引入了**无头、自主融资**，为 Agent 提供了独立交易的方法，并为新型用例开辟了道路。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Natural Attention 挑战 Adam**：Jeroaranda 介绍了一种 **Natural Attention** 方法，该方法可以近似 **Fisher matrix**，并在某些训练场景中超越了 **Adam**，参考 [GitHub](https://github.com/jeroaranda/naturalattention) 上的证明细节。
   - 社区成员强调了使用 **causal mask** 的必要性，并辩论了预训练数据中**质量与数量**的关系，强调需要对这些主张进行深入验证。
- **MSR 的伦理困境被曝光**：在出现剽窃案例后，关于 **MSR** 伦理问题的讨论爆发，涉及两篇论文，其中包括一篇 **NeurIPS spotlight award** 的入围作品。
   - 参与者对引用 MSR 的工作表示不信任，并质疑其**研究环境**的可信度，警告他人要谨慎对待。
- **BOS Token 的过度影响**：成员们发现 **BOS token** 位置的激活范数（activation norms）可能高出多达 **30倍**，这可能会扭曲 **SAE** 的训练结果。
   - 他们建议从训练数据中排除 **BOS** 或应用归一化（normalization）来减轻这种不成比例的影响，并参考了 2k 和 1024 上下文长度的短上下文实验。
- **基准测试目录混乱**：用户被保存到 `./benchmark_logs/name/__mnt__weka__home__...` 而非 `./benchmark_logs/name/` 的日志搞得措手不及，这使多模型运行变得复杂。
   - 他们提出了唯一的命名规范和专门用于比较所有 checkpoint 的 harness，以平衡改进与**向后兼容性（backwards compatibility）**。
- **GPT-Neox MFU 日志记录受到关注**：**Pull Request #1331** [添加了 MFU/HFU 指标](https://github.com/EleutherAI/gpt-neox/pull/1331)，用于 `neox_args.peak_theoretical_tflops` 的使用，并将这些统计数据集成到 **WandB** 和 **TensorBoard** 中。
   - 社区对新的 **tokens_per_sec** 和 **iters_per_sec** 日志表示赞赏，并在收到积极反馈后合并了该 PR，尽管测试有所延迟。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **FFI 摩擦：v24.6 的纠葛**：从 **v24.5** 升级到 **v24.6** 触发了与标准库内置 **write** 函数的冲突，使 **Mojo** 中的 socket 使用变得复杂。
   - 开发者建议使用 *FileDescriptor* 作为权宜之计，参考 [write(3p)](https://man7.org/linux/man-pages/man3/write.3p.html) 以避免符号冲突。
- **Libc 绑定助力更精简的 Mojo**：成员们推动更广泛的 **libc** 绑定，据报告已有 150 多个函数初步完成了 **Mojo** 集成。
   - 他们主张为这些绑定建立单一仓库，以加强跨平台测试和系统级功能。
- **浮点数解析遇到障碍**：从 **Lemire** 移植浮点数解析的效果不佳，标准库方法的表现也慢于预期。
   - 一个待处理的 PR 寻求升级 **atof** 并增强数值处理能力，旨在提升数据密集型任务中的性能。
- **Tensorlike Trait 之争**：[GitHub Issue #274](https://github.com/modularml/max/issues/274) 中的一个请求要求 **tensor.Tensor** 实现 **tensor_utils.TensorLike**，声称它已经符合标准。
   - 关于 `Tensor` 应该是 **trait** 还是 **type** 产生了争论，这反映了在 **MAX APIs** 内部直接实例化的挑战。
- **Modular 邮件：总结 2024**：**Modular** 感谢社区在高效的 **2024** 年中所做的贡献，宣布**假期停工**至 1 月 6 日，期间回复将会减少。
   - 他们邀请通过 [论坛帖子](https://forum.modular.com/t/max-24-6-and-max-gpu-feedback/331/5) 和 [GitHub Issues](https://github.com/modularml/max/issues) 反馈 **24.6 版本** 的意见，激发了对 2025 年的期待。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 的 O3 在 ARC-AGI 上表现强劲**：OpenAI 推出了 **O3 模型**，在 ARC-AGI 半私有评估（Semi-Private Evaluation）中得分 **75.7%**，在最高计算模式下得分 **87.5%**，显示出强大的推理性能。研究人员提到了可能的并行 Chain-of-Thought 机制以及巨大的资源需求。
   - 许多人讨论了该模型的成本——传闻约为 **150 万美元**——同时对其在代码、数学和逻辑任务上的飞跃表示赞赏。
- **Alec Radford 离职**：以早期 GPT 贡献闻名的 **Alec Radford** 确认离开 OpenAI 进行独立研究。成员们推测了领导层的变动以及对即将发布的模型的潜在影响。
   - 一些人预测内部很快会有转向，另一些人则称赞 Radford 过去的工作是 GPT 奠基的关键。
- **高计算 AI 的经济压力**：讨论引发了对高昂计算预算（如驱动 O3 的预算）可能阻碍商业可行性的担忧。参与者警告说，虽然突破令人兴奋，但它们带来了巨大的运营成本。
   - 他们权衡了在 ARC-AGI 上提升的性能是否足以证明这些支出是合理的，特别是对于代码和数学等专业任务。
- **安全测试成为焦点**：OpenAI 邀请志愿者对 **O3 和 O3-mini** 进行压力测试，体现了对发现潜在滥用行为的重视。这一呼吁强调了在更广泛部署之前进行彻底审查的推动力。
   - 安全研究人员对这一机会表示欢迎，进一步强化了将社区驱动的监督作为负责任 AI 进展的关键衡量标准。
- **API Keys 与 Character AI 角色扮演**：开发者报告了对 **API keys** 的修补尝试，突显了 AI 社区日常的实验。与此同时，Character AI 吸引了更年轻的群体，他们对“迪士尼公主”风格的互动感兴趣。
   - 参与者注意到了用户体验的信号，引用“神奇的数学石头（magical math rocks）”这一幽默说法来强调典型商业应用之外的趣味性互动。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **AI 助力播客兴起**：一次对话强调了使用 AI 制作[播客剧集](https://www.youtube.com/@AI_ForThePeople)，加速了内容创作并提高了各章节音频的一致性。
   - 此外，一个名为 *Churros in the Void* 的项目使用了 Notebook LM 和 LTX-studio 进行视觉和配音制作，强化了自主配音的方法。
- **Notebook LM 助力教育**：一位用户将 Notebook LM 描述为在新闻课上构建时间线和思维导图的强大工具，引用了来自[此笔记本](https://notebooklm.google.com/notebook/8)的数据。
   - 他们整合了课程材料和特定主题的播客，据报告提高了内容的组织性，使论文更具连贯性。
- **AI 帮助求职者准备**：一位成员使用 Notebook LM 根据职位广告分析自己的简历，为即将到来的面试生成了定制的学习指南。
   - 他们建议其他人上传简历，以获得关于技能匹配度的即时建议。
- **交互模式与引用工具遇到障碍**：几位用户在访问新的基于语音的交互模式时遇到困难，引发了对其发布不均衡的质疑。
   - 其他人报告了一个导致保存的笔记中引用功能消失的 Bug，开发团队确认修复工作正在进行中。
- **音频概览与语言限制**：一位用户请求关于恢复丢失的音频概览的技巧，指出一旦丢失就很难生成完全相同的版本。
   - 类似的讨论探索了 Notebook LM 如何通过将内容分成不同的集合，来更准确地处理多样化的语言源。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **OpenAI 的 o3 强势推进**：OpenAI 推出了全新的 **o3 和 o3-mini** 模型，[TechCrunch](https://techcrunch.com/2024/12/20/openai-announces-new-o3-model/) 的报道引发了关于其性能是否能超越 **o1** 里程碑的讨论。
   - 一些参与者强调了这些发布对大规模部署的重要性，并引用了一段 [视频演示](https://www.copilotforyoutube.com/search/openai-o3-and-o3-mini12-days-of-openai-day-12-T7sbiQRKxbMdlrWTddGC9L)，其中 Sam Altman 呼吁在测试驱动下保持谨慎。
- **Lepton AI 推动 Node 支付**：新推出的基于 Node 的支付解决方案呼应了来自 [Lepton AI](https://search.lepton.run/) 的开源蓝图，引发了关于原创性的讨论。
   - 评论指向了 [GitHub repo](https://github.com/leptonai/search_with_lepton) 作为此前开源努力的证据，加剧了关于复用和正确引用的争论。
- **三星的 Moohan 任务**：**Samsung** 推出了 [Project Moohan](https://www.perplexity.ai/page/samsung-s-project-moohan-Od28QDNtTTiZjpKBmuhOfg) 这一 AI 项目，引发了对其新集成功能的猜测。
   - 细节目前仍然较少，但参与者对其与现有硬件和 AI 平台的协同作用感到好奇。
- **职场 AI 使用率激增**：最近的一项 [调查](https://www.perplexity.ai/page/more-than-70-use-ai-at-work-ym5.V8EjTHmJhCCVrvZuGQ) 声称，超过 **70%** 的员工正在将 **AI** 融入日常任务中。
   - 人们注意到新的生成式工具如何简化代码审查和文档编写，这表明高级自动化的标准正在提高。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All v3.6.x：快速迭代与修复**：全新的 **GPT4All v3.6.0** 发布，配备了 **Reasoner v1**、内置的 JavaScript 代码解释器，并改进了模板兼容性。
   - 社区成员迅速处理了 **v3.6.1** 中的回归错误，**Adam Treat** 和 **Jared Van Bortel** 领导了修复工作，详见 [Issue #3333](https://github.com/nomic-ai/gpt4all/issues/3333)。
- **Llama 3.3 与 Qwen2 进步显著**：成员们强调了 **Llama 3.3** 和 **Qwen2** 的功能性提升，称其性能优于之前的版本。
   - 他们引用了 [Logan Kilpatrick](https://x.com/OfficialLoganK/status/1869789822384255300) 的一条推文，展示了利用视觉和文本元素解决谜题的能力。
- **Phi-4 表现超出预期**：据 [Hugging Face](https://huggingface.co/matteogeniaccio/phi-4/tree/main) 报道，拥有 14B 参数的 **Phi-4 模型** 据称可以与 **Llama 3.3 70B** 媲美。
   - 社区测试者对本地运行的流畅度发表了评论，注意到其强大的性能，并对进一步测试充满热情。
- **自定义模板与 LocalDocs 联动**：一个专门的 **GPT4All** 聊天模板利用代码解释器实现强大的推理，经验证可与多种模型类型配合使用。
   - 成员们描述了将 **GPT4All** 本地 API 服务器与 **LocalDocs** ([文档](https://docs.gpt4all.io/gpt4all_api_server/home.html)) 连接，从而实现有效的离线操作。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **本地生成器大对决：SD1.5 vs SDXL 1.0**：一些成员称赞 **SD1.5** 性能稳定，而另一些人则推荐使用 **SDXL 1.0** 配合 [ComfyUI](https://comfyui.org) 以获得更高级的效果。
   - 他们注意到概念艺术在文生图清晰度方面的提升，并强调了这些本地模型极低的安装门槛。
- **Flux 风格迁移受到关注**：一位用户在本地运行了 **Flux**，并寻求如何匹配参考图风格以用于游戏场景的建议。
   - 他们提到成功保留了配色方案和轮廓，并归功于 **Flux** 中一致的参数设置。
- **诈骗警报：技术支持服务器引发关注**：一个自称提供 Discord 帮助的可疑团体要求提供钱包详情，引发了安全担忧。
   - 成员们比较了更安全的替代方案，并互相提醒注意标准防范措施。
- **SF3D 崭露头角，助力 3D 资产创作**：爱好者们指向了 [Hugging Face 上的 stabilityai/stable-fast-3d](https://huggingface.co/stabilityai/stable-fast-3d)，用于生成等距视角角色和道具。
   - 他们报告称，在创建游戏就绪对象时结果稳定，且伪影比其他方法更少。
- **LoRA 魔法助力个人艺术训练**：一位艺术家表达了希望通过用自己的图像训练新模型来加快艺术创作速度。
   - 其他人推荐进行 **LoRA** 微调，特别是针对 **Flux** 或 **SD 3.5**，以锁定风格细节。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere c4ai Commands MLX 势头强劲**：在一次 MLX 集成推进期间，成员们测试了 [Cohere 的 c4ai-command-r7b 模型](https://huggingface.co/mlx-community/c4ai-command-r7b-12-2024-4bit)，并赞扬了其提升的开源协同效应。
   - 他们强调了早期的 **VLLM** 支持，并指向了一个可能加速进一步扩展的 [Pull Request](https://github.com/ml-explore/mlx-examples/pull/1157)。
- **128K 上下文特性令粉丝印象深刻**：社区的一份测评展示了 **Cohere 模型**在 **11.5 GB** 内存上处理了一个包含 211,009 个 token 的《弹丸论破》同人小说。
   - 讨论将强大的扩展上下文能力归功于*缺乏位置编码（positional encoding）*，称其为处理大规模文本任务的关键因素。
- **O3 模型引发猜测**：成员们调侃了一个具有类似于 **GPT-4** 特性的 **O3 模型**，点燃了对语音交互功能的期待。
   - 他们预测该模型可能很快发布，期待其先进的 AI 功能。
- **Findr 借 Cohere 之势首次亮相**：社区成员庆祝了 **Findr** 的发布，将其幕后的技术支持归功于 Cohere 的技术栈。
   - 一位成员询问使用了哪些 **Cohere** 特性，反映出对集成方案选择的研究兴趣。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **OpenAI o3 强势登场**：OpenAI 发布了其 **o3** 推理模型，在低算力模式下达到 **75.7%**，在高算力模式下达到 **87.5%**。
   - 一场对话引用了 [François Chollet 的推文](https://x.com/fchollet/status/1870169764762710376)和 [ARC-AGI-Pub 结果](https://arcprize.org/blog/oai-o3-pub-breakthrough)，暗示了在处理高级任务方面的新势头。
- **是否为 AGI：争论激烈**：一些人断言，在 ARC 等任务上超越人类表现标志着 **AGI** 的到来。
   - 另一些人则坚持认为 **AGI** 的定义过于模糊，敦促使用结合语境的含义以避免混淆。
- **Elo 评分与算力推测**：参与者将 **o3** 的结果与特级大师级的 Elo 评分进行了比较，参考了 [Elo 概率计算器](https://wismuth.com/elo/calculator.html#rating1=2727&rating2=1258)。
   - 他们思考较弱的模型是否可以通过每次扩展运行花费 **$20** 的额外测试时算力（test-time compute）来达到类似的结果。
- **关于 DCT 和 VAE 的精彩讨论**：讨论集中在 **YCrCb** 或 **YUV** 等颜色空间的 **DCT** 和 **DWT** 编码上，质疑额外的步骤是否能证明训练开销的合理性。
   - 一些人引用了 **VAR 论文**，建议先预测 DC 分量然后再添加 AC 分量，强调了亮度通道在**人类感知**中的作用。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 文档出故障，开发者挺身而出**：[Triton 官方文档](https://triton-lang.org/main/index.html#)的搜索功能已损坏，社区指出缺少关于 **tl.dtypes**（如 **tl.int1**）的规范。
   - 如果文档后端开放编辑，热心的贡献者希望对其进行修复。
- **Flex Attention 势头渐起**：正在折腾 **flex attention** 加 **context parallel** 的成员表示，一个示例可能很快就会进入 attn-gym。
   - 他们认为将这些方法结合起来是有效处理更大任务的直接路径。
- **Diffusion Autoguidance 备受关注**：[Tero Karras 发表的一篇 NeurIPS 2024 新论文](https://x.com/TheVariational/status/1870196816844603717)概述了如何通过 **Autoguidance** 方法塑造 **Diffusion** 模型。
   - 该论文的亚军地位及其 [PDF 链接](https://drive.google.com/file/d/1WxQ7Zd15Ly4tFt2YFghJm-cmXlTgcEYI/view)引发了关于其对生成建模影响的大量讨论。
- **ARC CoT 数据助力 LLaMA 8B 测试**：一位用户正在制作一个包含 **10k 样本的 ARC CoT** 数据集，以观察微调后的 **LLaMA 8B** 在对数概率指标上是否超过基础模型。
   - 他们计划在生成几千个样本后检查“CoT”训练的影响，强调了对未来评估的潜在改进。
- **PyTorch 聚焦稀疏性**：[PyTorch 稀疏性设计](https://github.com/pytorch/ao/tree/main/torchao/sparsity#design)引入了用于推理的 `to_sparse_semi_structured`，用户建议更换为 `sparsify_` 以获得更大的灵活性。
   - 这种方法还突出了原生量化和其他用于模型优化的内置功能。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaParse 增强音频解析能力**：**LlamaParse** 工具现在支持解析音频文件，通过语音转文本功能补充了原有的 PDF 和 Word 支持。
   - 根据用户反馈，这次更新巩固了 **LlamaParse** 作为多媒体工作流中强大的跨格式解析器的地位。
- **LlamaIndex 庆祝年度增长**：他们在年终回顾中宣布一年内解析了数千万页内容，并保持每周发布新功能。
   - 他们预告 **LlamaCloud** 将于 **2024** 年初正式发布（GA），并分享了一个包含详细统计数据的[年度回顾链接](https://t.co/bxx5t1sVgy)。
- **股票分析机器人利用 LlamaIndex 大放异彩**：一个快速教程演示了如何使用 **FunctionCallingAgent** 和 **Claude 3.5 Sonnet** 构建自动化股票分析 Agent。
   - 工程师可以参考 [Hanane D 的帖子](https://t.co/GOjUTl0Es0) 获取简化金融任务的一键式解决方案。
- **LlamaIndex 文档自动化演示**：一个 Notebook 展示了 **LlamaIndex** 如何标准化跨多个供应商的单位和测量。
   - 该[示例 Notebook](https://t.co/aOTuSwM341) 演示了适用于真实生产环境的统一工作流。
- **利用合成数据微调 LLM**：用户讨论了为情感分析生成人工样本，并参考了 [Hugging Face 博客](https://huggingface.co/blog/synthetic-data-save-costs)。
   - 他们建议将 Prompt 操纵作为起步阶段，而其他人则讨论了更广泛的模型优化方法。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **黑客松冲刺与重新开放热潮**：由于参赛者面临**技术困难**，[黑客松提交表单](https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform)已重新开放，截止时间为 **PST 时间 12 月 20 日晚上 11:59**。
   - 组织者确认不会再延期，因此参赛者应在认证表单中确认主要联系邮箱等细节，以便接收官方通知。
- **人工提交检查与视频格式问题**：为不确定提交状态的参赛者提供人工核实流程，以防止最后一刻出现混乱。
   - 部分人在遇到 **YouTube** 问题后改用邮件提交，并表示他们仍专注于黑客松而非课程本身。
- **Agent 构建方案替代方案与 AutoGen 警告**：一位参赛者引用了[一篇关于 Agent 构建策略的文章](https://www.anthropic.com/research/building-effective-agents)，建议不要完全依赖像 **Autogen** 这样的框架。
   - 他们建议在未来的 MOOC 中采用更简单、模块化的方法，并强调指令微调（Instruction Tuning）和函数调用（Function Calling）。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune v0.5.0 发布**：开发者发布了 **Torchtune v0.5.0**，包含了 **Kaggle** 集成、**QAT + LoRA** 训练、**Early Exit** Recipe 以及 [**Ascend NPU** 支持](https://github.com/pytorch/torchtune/pull/1826)。
   - 他们分享了[发布说明](https://github.com/pytorch/torchtune/releases/tag/v0.5.0)，详细介绍了这些升级如何简化大型模型的微调。
- **QwQ-preview-32B 扩展 Token 视野**：有人在 8×80G GPU 上测试了 **QwQ-preview-32B**，旨在实现超过 **8K** Token 的上下文并行。
   - 他们提到了 **optimizer_in_bwd**、**8bit Adam** 和 [QLoRA 优化标志](https://github.com/pytorch/torchtune#optimization-flags)作为扩展输入规模的方法。
- **fsdp2 State Dict 加载引发关注**：开发者对加载 **fsdp2** state dict 提出疑问，因为在[分布式加载代码](https://github.com/pytorch/torchtune/blob/main/torchtune/training/_distributed.py#L213)中，分片参数与非 **DTensors** 存在冲突。
   - 他们担心这些不匹配会增加在多节点部署 **FSDPModule** 设置的复杂性。
- **词表裁剪（Vocab Pruning）需注意 fp32**：部分参与者通过裁剪词表来减小模型体积，但坚持以 **fp32** 格式保留参数以确保精度一致。
   - 他们强调了分别处理 **bf16** 计算和 **fp32** 存储，以维持稳定的微调过程。



---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Litellm Proxy 受到关注**：**Litellm** 可以自托管或通过托管服务使用，并且可以与主系统运行在同一个 VM 上以简化操作。讨论强调这种设置通过将代理与相关服务捆绑在一起，使集成更加顺畅。
   - 参与者指出，它在保持易于调整的同时，满足了*广泛的基础设施需求*。
- **合成数据推动 LLM 升级**：一篇名为 **On Synthetic Data: How It’s Improving & Shaping LLMs** 的文章（发表于 [dbreunig.com](https://www.dbreunig.com/2024/12/18/synthetic-data-the-growing-ai-perception-divide.html)）解释了合成数据如何通过模拟类聊天机器人输入来微调较小的模型。对话还涵盖了其在大规模任务上的有限影响以及在不同领域应用的细微差别。
   - 成员们观察到了*错综复杂的结果*，但一致认为这些生成的数据集可以推动 **reasoning** 研究的发展。
- **优化成本引发担忧**：高级 **optimizers** 的长时间运行导致成本上升，促使人们建议限制调用次数或 Token 数量。一些人建议使用较小的参数设置，或将 **LiteLLM** 与预设限制配对以避免超支。
   - 讨论中的声音强调了主动资源监控以避免意外支出。
- **MIPRO 'Light' 模式节省资源**：**MIPRO 'Light'** 模式为那些希望运行优化步骤的人提供了一种更精简的方式。据称它在更受控的环境中平衡了处理需求与性能。
   - 早期采用者提到，*较少的资源* 仍然可以产生不错的结果，这为试验指明了一条充满希望的道路。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 的服务器模式引起兴趣**：一位用户询问了在 VPS 上以服务器模式运行 **OpenInterpreter** 的文档，好奇命令是在本地还是在服务器上运行。
   - 他们表达了确认远程使用可能性的渴望，强调了灵活配置的潜力。
- **Google Gemini 2.0 热度升温**：有人询问了新的 **Google Gemini 2.0** 多模态功能，特别是其 *os mode*，并指出访问权限可能仅限于 'tier 5' 用户。
   - 他们对其可用性和性能表示好奇，建议需要更广泛的测试。
- **本地 LLM 集成带来良好体验**：一位参与者庆祝 **本地 LLM 集成** 为 OpenInterpreter 增添了受欢迎的离线维度。
   - 他们之前担心会失去这个功能，但现在对它仍然得到支持感到欣慰。
- **SSH 使用启发了前端目标**：一位用户分享了他们通过 SSH 与 **OpenInterpreter** 交互的方法，并指出远程体验非常直接。
   - 他们暗示了开发前端界面的计划，并对以极小摩擦实现它充满信心。
- **社区标记垃圾信息**：一名成员提醒其他人注意聊天中的推荐垃圾信息，旨在维护整洁的环境。
   - 他们向相关角色报告了该事件，希望能得到及时干预。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **KTO 与 Liger：令人惊喜的组合**：公会成员确认 **Liger** 现在集成了 **KTO**，支持旨在提升模型性能的高级协同效应。
   - 他们指出了相对于 HF TRL 基准的 **loss parity** 担忧带来的*困扰*，促使对训练指标进行进一步审查。
- **DPO 愿景：Liger 关注下一步**：团队正将重点放在 **Liger DPO** 上作为主要优先级，旨在实现稳定运行，从而带来更顺畅的扩展。
   - 尽管出现了对 **loss parity** 问题的沮丧情绪，但人们仍然乐观地认为，针对这些遗留问题的修复方案很快就会出现。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **陈旧 PR 面临清理**：一位用户计划从下周开始关闭或自动关闭超过 30 天的 PR，移除过时的代码提案。这使项目摆脱了过多的开放请求，同时保持代码库精简。
   - 他们强调了清理长期积压 PR 的重要性。除了拟定的时间表外，没有分享更多细节或链接。
- **机器人可能会介入**：他们提到可能会使用机器人来跟踪或关闭不活跃的 PR，减少人工监管。这种方法可以减少维护任务并保持开发队列整洁。
   - 未提供具体的机器人名称或实现细节。对话在没有额外引用或公告的情况下结束。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Watt-Tool 模型助力 Gorilla 排行榜**：提交了一个 [**pull request #847**](https://github.com/ShishirPatil/gorilla/pull/847)，旨在将 **watt-tool-8B** 和 **watt-tool-70B** 添加到 Gorilla 的 function calling 排行榜中。
   - 这些模型也可以在 [**watt-tool-8B**](https://huggingface.co/watt-ai/watt-tool-8B/) 和 [**watt-tool-70B**](https://huggingface.co/watt-ai/watt-tool-70B/) 获取，以便进行进一步实验。
- **贡献者寻求在圣诞节前进行评审**：他们请求及时检查新添加的 **watt-tool**，并暗示了潜在的性能和集成问题。
   - 鼓励在假期休整前，就 function calling 使用案例以及与现有 Gorilla 工具的协同效应提供社区反馈。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要及链接

{% if medium == 'web' %}

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1319758354016637051)** (1 条消息): 

> `Windsurf 1.1.1 发布，使用透明度与定价，Cascade 图片上传，语言支持改进` 

- **Windsurf 1.1.1 发布，带来酷炫功能**：**Windsurf 1.1.1** 更新引入了易用性改进（QoL），例如全新的**“Send to Cascade”**按钮和增强的 autocomplete 功能，以及显示套餐和使用信息的状态栏。
   - 同时也推出了错误修复，解决了 **Windows 聊天模式编辑**和 autocomplete 变慢等问题，详见[完整变更日志](https://www.codeium.com/changelog)。
- **全新的定价和使用透明度功能**：Windsurf 正在实施经过改进的**定价系统**，通过快速设置面板为用户提供关于当前套餐使用情况和试用过期的更清晰信息。
   - 引入了 **“Legacy Chat” 模式**，允许用户在没有 Flow Credits 的情况下继续使用 Cascade，尽管功能有限，更多详情请参阅[此处](https://docs.codeium.com/windsurf/usage#viewing-your-usage)。
- **Cascade 图片上传现已扩展**：移除了 **Cascade 图片上传的 1MB 限制**，允许用户无缝上传更大的图片。
   - 这一调整旨在提升 Cascade 功能的用户体验，鼓励与大型视觉内容进行更好的交互。
- **Python 语言支持得到增强**：本次更新实现了对 Python **改进的语言支持**，巩固了 Python 程序员的开发环境。
   - 这些增强功能旨在提高在 Windsurf 框架内工作时的生产力和效率。

**提到的链接**：<a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>：Windsurf 编辑器的最新更新和变更。

---

### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1319827690123169843)** (1 条消息): 

> `Send to Cascade 按钮` 

- **“Send to Cascade” 按钮演示**：分享了一个关于 [Send to Cascade](https://x.com/windsurf_ai/status/1870268007995585000) 按钮的快速演示，该按钮允许用户将问题直接发送到 **Cascade**。
   - *“将你的问题直接发送到 Cascade！”* 表明了用户升级问题的一种简单直接的方式。
- **用户参与 “Send to Cascade” 功能**：鼓励用户尝试 **Send to Cascade** 功能，通过允许更快的问题解决来增强用户体验。
   - 该按钮旨在简化与 **Cascade** 的沟通，创造更顺畅的故障排除流程。

**提到的链接**：<a href="https://x.com/windsurf_ai/status/1870268007995585000">来自 Windsurf (@windsurf_ai) 的推文</a>：将你的问题直接发送到 Cascade！

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1319433232604860416)** (64 messages🔥🔥): 

> `Cascade 性能, Windsurf 订阅计划, Codeium 扩展功能, AI 在代码审查中的应用, AI Prompt 编写指南` 


- **Cascade 的内部错误**：用户报告在长时间对话后使用 Cascade 时遇到 *'ErrorCascade has encountered an internal error'* 等错误，建议开启新对话以刷新会话。
   - 另一位用户强调了保持对话简洁聚焦对提升性能的重要性。
- **了解 Windsurf 计划**：一位用户询问 Windsurf 试用 Pro 计划的可用性，质疑该计划是否已取消，因为他们只获得了免费计划。
   - 其他用户讨论了他们在 Codeium 各种产品（包括扩展程序和 Windsurf）中的订阅限制和功能体验。
- **AI 交互变慢**：一位成员对处理大型代码库时的 AI 性能表示沮丧，特别指出在处理 1k 行源代码时速度缓慢。
   - 讨论显示，部分用户在代码更改的响应时间上也遇到了类似问题。
- **将 Windsurf 作为编程助手**：用户分享了对 Windsurf 直接读取代码库能力的兴奋，认为这比直接在网站上使用 Sonnet 有显著改进。
   - 一位成员提到将 Windsurf 与 Cascade 结合作为结对编程工具，以增强他们的编码体验。
- **AI 的 Prompt 技巧**：成员们讨论了刷新对话会话的重要性，并提供了 Prompt 指南链接以优化与 AI 的交互。
   - 一位用户表示需要教程来加深对如何有效使用 Windsurf 的理解。



**相关链接**：<a href="https://codeium.com/plan">计划设置</a>：未来的编辑器，就在今天。Windsurf 编辑器是首款基于 AI Agent 的 IDE，能让开发者保持专注流。现已支持 Mac, Windows 和 Linux。

  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1319394131952275466)** (603 messages🔥🔥🔥): 

> `Windsurf 性能问题, Codeium 功能与更新, 高效使用 Cascade, AI 模型用户体验, 新工具集成` 


- **对 Windsurf 性能的担忧**：用户对在一天中不同时间使用 Windsurf 时明显的性能差异表示沮丧，有人报告夜间效果更好。
   - 许多人正经历 AI 性能问题，导致编码体验效率降低，并引发了关于 AI 能力的讨论。
- **Windsurf 更新带来新功能**：最近的更新在“问题 (Problems)”选项卡中添加了“Send to Cascade”按钮，允许用户轻松报告问题，该功能广受好评。
   - 更新还改进了 Autocomplete（自动补全）功能，使依赖 Cascade 进行编码辅助的用户受益。
- **在项目中高效使用 Cascade**：鼓励用户使用 Cascade 处理问题，但由于观察到效率低下，讨论了同时管理多个问题的技巧。
   - 一些用户分享了使用 Cascade 完成复杂项目的成功经验，强调了在战略性使用该工具时的潜力。
- **请求更好的集成与支持**：用户持续要求与 Gemini 等模型进行更清晰的集成，并希望 Codeium 支持部门能改善针对账户问题的响应。
   - 用户强调需要更易获取的资源以及社区内关于更新的明确说明，以确保流畅的体验。
- **探索 CLI 工具与 AI 集成**：关于将命令行界面 (CLI) 与 Warp 等 AI 工具结合使用的讨论，强调了它们在生产力和自动化方面的优势。
   - 用户辩论了 CLI 在编码工作流中的有效性，一些人对其对效率的影响持怀疑态度。


<div class="linksMentioned">

<strong>相关链接</strong>：

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/cascade#allow-list">Windsurf - Cascade</a>: 未找到描述</li><li><a href="https://ai.google.dev/pricing#1_5flash">未找到标题</a>: 未找到描述</li><li><a href="https://codeium.com/contact">Contact | Windsurf Editor and Codeium extensions</a>: 联系 Codeium 团队获取支持，并了解更多关于我们企业级服务的信息。</li><li><a href="https://sunlight-globe-react.vercel.app/">Vite + React + TS</a>: 未找到描述</li><li><a href="https://codeium.canny.io/feature-requests/p/pair-programmer-system">Pair Programmer System | Feature Requests | Codeium</a>: 如果 AI 可以与你并肩编辑、跟随你的光标、观察你的操作、并随时提供建议和反馈，那会怎样？如果...</li><li><a href="https://tenor.com/view/things-that-make-you-go-bluh-ron-white-blue-collar-comedy-gif-12400907237761359223">Things That Make You Go Bluh Ron White GIF - Things That Make You Go Bluh Ron White Blue Collar Comedy - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/Codeium/comments/1hieg7u/windsurf_being_dumb_try">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://llmpricecheck.com/calculator/">LLM Pricing Calculator - LLM Price Check</a>: 使用 LLM Price Check 的 LLM 价格计算器探索高性价比的 LLM API 选项。快速比较来自 OpenAI、Anthropic 和 Google 等顶级供应商的费率。</li><li><a href="https://codeium.com/faq">FAQ | Windsurf Editor and Codeium extensions</a>: 查找常见问题的答案。</li><li><a href="https://www.reddit.com/r/Codeium/comments/1hieg7u/windsurf_being_dumb_try_this_fix/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://tenor.com/view/surf-glider-wave-giant-wave-wind-gif-15418238">Surf Glider GIF - Surf Glider Wave - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.warp.dev/">Warp: The intelligent terminal</a>: Warp 是一款内置 AI 和开发团队知识库的智能终端。现已支持 MacOS 和 Linux。</li><li><a href="https://github.com/ichoosetoaccept/awesome-windsurf/issues/50">Set up Reddit API integration for r/Codeium content · Issue #50 · ichoosetoaccept/awesome-windsurf</a>: 我们需要一种以编程方式从 r/Codeium 获取有价值的 Windsurf/Codeium 技巧的方法。潜在方案：使用 Reddit API (Python 的 PRAW)、Reddit OAuth API、Reddit RSS 订阅。关键需求：Fol...</li><li><a href="https://github.com/ichoosetoaccept/awesome-windsurf">GitHub - ichoosetoaccept/awesome-windsurf: A collection of awesome resources for working with the Windsurf code editor</a>: 用于 Windsurf 代码编辑器的精选资源集合 - ichoosetoaccept/awesome-windsurf</li><li><a href="https://cli.github.com/manual/gh_extension">GitHub CLI</a>: 在命令行中使用 GitHub</li><li><a href="https://github.com/SchneiderSam/awesome-windsurfrules">GitHub - SchneiderSam/awesome-windsurfrules: 📄 A curated list of awesome global_rules.md and .windsurfrules files</a>: 📄 global_rules.md 和 .windsurfrules 文件的精选列表 - SchneiderSam/awesome-windsurfrules</li><li><a href="https://www.youtube.com/watch?v=PkGfG4iTR44"> - YouTube</a>: 未找到描述</li><li><a href="https://codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Windsurf 编辑器的最新更新和变更。</li><li><a href="https://soundcloud.com/seebastian/moehre2024?in=rovingkid/sets/fucking-bonkers&si=2451aecd590e43a9a48d8f363dcb79c7,">See Bastian | Wilde Möhre 2024 | Puppenräuber</a>: See Bastian // Wilde Möhre 2024 // Puppenräuber // 播放时间：周日早晨 5:30-7:30。感谢所有参加这次特别清晨运动环节的人 &lt;3</li><li><a href="https://codeium.canny.io/feature-requests">Feature Requests | Codeium</a>: 向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://www.reddit.com/r/Codeium/comments/1heztku/my_experience_with_windsurf_lets_make_it_better/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://mcpservers.org/category/web-scraping">Awesome MCP Servers</a>: 未找到描述</li><li><a href="https://github.com/dlvhdr/gh-dash">GitHub - dlvhdr/gh-dash: A beautiful CLI dashboard for GitHub 🚀</a>: 一个美观的 GitHub CLI 仪表板 🚀。通过在 GitHub 上创建账户来为 dlvhdr/gh-dash 的开发做出贡献。</li><li><a href="https://codeium.com/terms-of-service-individual">Terms of Service: Individual &amp; Pro | Windsurf Editor and Codeium extensions</a>: Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首款 Agentic IDE —— Windsurf 的开发者。
</li>
</ul>

</div>
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1319394021533286411)** (819 条消息🔥🔥🔥): 

> `Cursor IDE 更新，AI 驱动的开发工具，Sonnet 模型对比，AI 辅助自由职业，AI 在样式设计中的局限性` 


- **Cursor IDE 更新带来性能提升**：用户报告称，Cursor IDE 最近的 0.44.5 版本更新显著提升了性能和易用性，特别是在 Agent 模式下。
   - 反馈强调了更流畅的编码体验和更可靠的输出，促使许多用户从其他替代方案重新切换回 Cursor。
- **AI 工具正在改变开发工作流**：许多用户强调了像 Cursor 这样的 AI 工具对开发流程的影响，能够更快地完成项目，并减少了对解决方案的大量搜索需求。
   - AI 的集成正在帮助开发者简化工作流并提高生产力。
- **Sonnet 模型及其性能**：围绕不同 Sonnet 模型的讨论显示，用户体验到的性能各异，最新版本因其生成 UI 组件的能力而受到青睐。
   - 讨论指出，模型的 System Prompts 和性能可能会有所不同，从而影响用户的偏好。
- **使用 AI 工具进行自由职业**：自由职业者分享了使用 AI 工具处理各种任务的经验，提升了他们在项目交付中的声誉和效率。
   - 虽然有人担心可能会因为使用 AI 而导致工作被拒，但许多人主张 AI 为开发带来的优势。
- **AI 生成样式的挑战**：用户注意到，虽然 AI 在后端逻辑方面表现出色，但在前端样式（Styling）方面往往表现不佳，导致开发者需要进行额外的调整。
   - 这一担忧反映出 AI 在 UI/UX 设计方面需要改进训练，以更好地协助开发者。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/arcprize/status/1870169260850573333?s=46">ARC Prize (@arcprize) 的推文</a>：新的经过验证的 ARC-AGI-Pub SoTA！@OpenAI o3 在 ARC-AGI 半公开评估中取得了突破性的 75.7% 分数。而高算力 o3 配置（不符合 ARC-AGI-Pub 资格）在 S... 中获得了 87.5% 的分数。</li><li><a href="https://x.com/_philschmid/status/1869639246434246966?s=46">Philipp Schmid (@_philschmid) 的推文</a>：WTF？！全新的开源物理 AI 引擎简直疯狂！🤯 Genesis 是一款全新的物理引擎，结合了超快速模拟与生成能力，为机器人技术创建动态 4D 世界...</li><li><a href="https://www.anthropic.com/research/building-effective-agents">构建高效 Agent</a>：一篇面向开发者的文章，提供了构建高效 AI Agent 的建议和工作流。</li><li><a href="https://www.cursor.com/downloads">下载 | Cursor - AI 代码编辑器</a>：选择您的平台以下载最新版本的 Cursor。</li><li><a href="https://x.com/testingcatalog/status/1870038932483653709?s=46">TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：LLM 命名是我们社会中最难的问题 👀 在一个晚上之后出现了这么多新的阴谋论。o3 在安全测试中被发现，4.5 在一周前被追踪到。o2 在哪里？🤯 引用 Colonel Tasty (@J...</li><li><a href="https://forum.cursor.com/t/is-it-possible-to-index-a-file-thats-in-my-gitignore/2771/2">是否可以索引 .gitignore 中的文件？</a>：您可以将 !path/to/folder 添加到您的 .cursorignore 中，以确保即使该文件夹被 gitignore 忽略也能被包含。请注意，您可能需要添加 !path/to/folder/* 或 !path/to/folder/**/*（参见 ...</li><li><a href="https://tenor.com/view/i-understand-it-now-ok-i-understand-it-now-i-understand-it-gurm-bear-gif-4859748791707698675">我现在明白了 Ok 我现在明白了 GIF - 我现在明白了 Ok 我现在明白了 我明白了 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/ashtom/status/1870153594810577164?t=wyrmlHwGGMNRMO90U0Ma8Q&s=19">Thomas Dohmke (@ashtom) 的推文</a>：OpenAI o1 现已向 Copilot Pro 的所有个人用户开放，每月 10 美元。Lets. Go. 😎 引用 GitHub (@github)：您现在可以在 Copilot Pro、Business 和 Enterprise 以及 GitHub Mod... 中使用 @OpenAI 的新 o1 模型。</li><li><a href="https://simonwillison.net/2024/Dec/16/webdev-arena/">WebDev Arena</a>：来自 [Chatbot Arena](https://lmarena.ai/) 团队（前身为 LMSYS）的新排行榜，这次专注于评估不同模型在“Web 开发”方面的表现——尽管它...</li><li><a href="https://x.com/JustinLin610/status/1870170616046985478">Junyang Lin (@JustinLin610) 的推文</a>：你在开玩笑吗</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准。</li><li><a href="https://tenor.com/view/scaler-create-impact-dog-coding-programming-gif-25011983">Scaler Create Impact GIF - Scaler Create Impact Dog - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/OpenAI/status/1870164871289155937">OpenAI (@OpenAI) 的推文</a>：第 12 天：OpenAI o3 的早期评估（是的，我们跳过了一个数字）https://openai.com/12-days/?day=12</li><li><a href="https://www.youtube.com/watch?v=SKBG1sqdyIU&ab_channel=OpenAI"> - YouTube</a>：未找到描述</li><li><a href="https://gist.github.com/simonw/ae27a3b2709d5412f4cb32ae99428099">prompt.md</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://x.com/btibor91/status/1870022347349987532">Tibor Blaho (@btibor91) 的推文</a>：这是真的——OpenAI 网站已经出现了“O3 Mini 安全测试调用”表单的引用。引用 Colonel Tasty (@JoshhuaSays)：在 OpenAI 网站上发现了 o3_min_safety_test 的引用。S...</li><li><a href="https://lovable.dev/">Lovable</a>：仅使用聊天界面构建软件产品</li><li><a href="https://techcrunch.com/2024/12/19/in-just-4-months-ai-coding-assistant-cursor-raised-another-100m-at-a-2-5b-valuation-led-by-thrive-sources-say/?fbclid=IwY2xjawHSDcdleHRuA2FlbQIxMQABHWRw2UUOyKXYwFNk3BngYw3-QkpoGWSONNh5ILLrMGA8CnuEkhrBegjF4Q_aem_tvSm_5L9AaMxIPzVG0q53w">独家：据消息人士称，AI 代码助手 Cursor 在短短 4 个月内由 Thrive 领投以 25 亿美元估值再次融资 1 亿美元</a>：开发 AI 驱动的代码助手 Cursor 的 Anysphere 公司，在 B 轮融资中以 26 亿美元的投后估值筹集了 1 亿美元，据...</li><li><a href="https://github.com/2-fly-4-ai/V0-system-prompt">GitHub - 2-fly-4-ai/V0-system-prompt</a>：通过在 GitHub 上创建账号来为 2-fly-4-ai/V0-system-prompt 的开发做出贡献。</li><li><a href="https://github.com/olweraltuve/LmStudioToCursor">GitHub - olweraltuve/LmStudioToCursor</a>：通过在 GitHub 上创建账号来为 olweraltuve/LmStudioToCursor 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1319395117445939301)** (628 条消息🔥🔥🔥): 

> `OpenAI O3 发布，Aider 和 Cline 的使用，AI 对软件开发的影响，编程职业安全感，开发者工具对比` 


- **对 OpenAI O3 发布的兴奋感**：OpenAI O3 模型的发布引发了热烈讨论，基准测试显示其在编程任务和整体功能方面有显著进步。
   - 用户强调了持续优化的必要性，并对随着 AI 技术演进未来成本降低的可能性进行了推测。
- **结合使用 Aider 和 Cline 处理任务**：开发者们正在讨论如何有效地结合使用 Aider 处理小改动，以及利用 Cline 的 Agent 能力处理大型自动化任务。
   - Cline 的记忆能力有望简化开发流程，使其成为初创公司和重度编程任务的有价值工具。
- **对开发领域职业安全感的担忧**：对话反映了对 AI 影响编程职业生涯的焦虑，一些人认为 AI 将取代工作的许多方面。
   - 然而，也有人认为虽然 AI 接管了某些任务，但由于需要监督和解决问题，对熟练开发者的需求并不会减少。
- **当前 AI 工具面临的挑战**：用户表达了对 AI 局限性的沮丧，特别是在理解上下文和有效执行编程指令的能力方面。
   - 尽管存在这些问题，开发者仍认可其带来的时间节省，这表明需要进一步的增强。
- **AI 驱动下的软件开发未来**：随着 AI 工具变得更加复杂，人们开始推测软件开发角色的演变以及新任务出现的潜力。
   - 对话强调了适应行业技术变革并在不断变化的格局中寻找价值的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenAI/status/1870164871289155937">来自 OpenAI (@OpenAI) 的推文</a>：第 12 天：OpenAI o3 的早期评估（是的，我们跳过了一个数字）https://openai.com/12-days/?day=12</li><li><a href="https://x.com/kimmonismus/status/1870037369786687997">来自 Chubby♨️ (@kimmonismus) 的推文</a>：o3 和 o3 mini 今天发布！什么？！我没预料到！LFG。由于法律原因，他们跳过了 o2 这个名字。也许 o3 将是第一个整合了其他所有功能的模型？无论如何，...</li><li><a href="https://x.com/iruletheworldmo/status/1870176332702986292">来自 🍓🍓🍓 (@iruletheworldmo) 的推文</a>：很多人都猜到了，但 o1 pro 在 ARC 上表现非常好</li><li><a href="https://jira.atlassian.com/">使用 Jira Software 解锁团队的最佳工作</a>：未找到描述</li><li><a href="https://tenor.com/view/take-my-money-heres-my-card-here%E2%80%99s-my-card-card-take-my-card-gif-5650338825958178904">拿走我的钱，这是我的卡 GIF - Take my money Heres my card Here’s my card - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=SKBG1sqdyIU"> - YouTube</a>：未找到描述</li><li><a href="https://x.com/arcprize/status/1870169260850573333">来自 ARC Prize (@arcprize) 的推文</a>：新的经过验证的 ARC-AGI-Pub SoTA！@OpenAI o3 在 ARC-AGI 半公开评估中取得了 75.7% 的突破性成绩。而高算力配置的 o3（不符合 ARC-AGI-Pub 资格）在 S... 中得分 87.5%</li><li><a href="https://tenor.com/view/mellstroy-gif-27569581">Mellstroy GIF - Mellstroy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/indian-gif-22259626">Indian GIF - Indian - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=2TpSWVN4zkg"> - YouTube</a>：未找到描述</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1hgl74u/openai_employee_o1_pro_is_a_different/">Reddit - 深入探讨任何内容</a>：未找到描述</li><li><a href="https://github.com/mufeedvh/code2prompt">GitHub - mufeedvh/code2prompt：一个将代码库转换为包含源码树、提示词模板和 Token 计数的单个 LLM Prompt 的 CLI 工具。</a>：一个将代码库转换为包含源码树、提示词模板和 Token 计数的单个 LLM Prompt 的 CLI 工具。 - mufeedvh/code2prompt</li><li><a href="https://tenor.com/view/red-alphabet-letter-dancing-letter-l-cartoons-gif-12084376">红色字母跳舞字母 L GIF - Red Alphabet Letter Dancing Letter L Cartoons - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1319395374519287869)** (33 messages🔥): 

> `Aider 硬件推荐，OpenRouter API 密钥设置，对 PDF 文件使用 /read 命令，Gemini 模型更新，Aider 教程资源` 


- **在客户端运行 Aider 的硬件推荐**：用户对在客户端运行 Aider 时 LLM 响应速度慢表示担忧，其中一位用户正在寻求推荐的硬件需求。
   - 一名成员指出，不应该出现此类延迟，这表明可能存在性能问题。
- **在配置文件中设置 OpenRouter API 密钥**：一位用户询问如何在 `.aider.conf.yaml` 中配置 OpenRouter API 密钥，并报告了参数无法识别的问题。
   - 另一名成员澄清说，密钥应设置为 `api-key: openrouter=sk-or-...`，并提供了正确语法的指导。
- **对 PDF 文件使用 /read 命令**：一位用户询问 Aider 是否可以读取 PDF 并将其内容用于上下文辅助，并表示 `/read` 命令对他不起作用。
   - 一名成员确认，在使用 Anthropic 模型时，`/read` 命令可以读取 PDF 文件。
- **Gemini 模型版本更新**：关于最新的 Gemini 模型 'gemini-2.0-flash-thinking-exp-1219' 展开了讨论，对其能力的评价褒贬不一。
   - 用户分享了在各种模型中使用高 map token 设置的经验，以及对上下文保留的影响。
- **Aider 教程和演示资源**：成员们寻求 Aider 的专业级教程和演示推荐，并发现了聊天中分享的资源。
   - 一位用户推荐了一个 YouTube 频道和官方教程，并提供了链接以帮助他人加深对 Aider 能力的理解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/config/api-keys.html">API Keys</a>：为 API 提供商设置 API 密钥。</li><li><a href="https://www.youtube.com/@CodingtheFuture-jg1he">Coding the Future With AI</a>：欢迎来到 Coding the Future With AI！我们的频道致力于帮助开发者和技术爱好者学习如何利用 AI 来提升技能和生产力。通过教程、专家访谈...</li><li><a href="https://aider.chat/docs/faq.html#why-is-the-llm-speaking-to-me-in-an-unexpected-language">FAQ</a>：关于 aider 的常见问题解答。</li><li><a href="https://aider.chat/docs/usage/tutorials.html">Tutorial videos</a>：由 aider 用户制作的入门和教程视频。</li><li><a href="https://aider.chat/examples/README.html">Example chat transcripts</a>：aider 是你终端里的 AI 配对编程助手。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1319722356461273113)** (5 messages): 

> `AniDoc 动画工具，Depth AI 评估，集成外部库` 


- **AniDoc 简化动画制作**：一个新工具 [AniDoc](https://x.com/Gradio/status/1870017358821015670) 允许用户根据角色设计参考为草图序列着色，即使草图在姿势和比例上有显著差异，也能保持高保真度。
   - 实验过程*非常愉快*，用户强烈推荐尝试。
- **评估用于代码理解的 Depth AI**：评估 [Depth AI](https://www.trydepth.ai)，它可以连接到你的代码库，在 Slack 和 Jira 等平台上构建定制的 AI 助手，提供深度的技术解答。
   - 它构建了一个全面的知识图谱来理解代码关系，并有效地回答有关更改的问题。
- **在大型代码库上使用 Depth AI 的经验**：一位成员分享了在大型代码库上使用 Depth AI 的积极体验，但由于不需要其 RAG 功能而决定停止使用。
   - 他们指出，在享受其集成能力的同时，*到目前为止它非常酷*。
- **关于集成外部库的讨论**：一位成员建议，将多个外部库复制到一个共享文件夹中，可以帮助利用 Depth AI 找出集成方案。
   - 他们对 Aider 无法处理 git submodules 表示遗憾，否则可以进行更多探索。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.trydepth.ai">Depth AI - 深度理解代码库的 AI</a>：与你的代码库聊天或构建定制的 AI 助手。部署到你工作的任何地方 —— Slack, GitHub Copilot, Jira 等。</li><li><a href="https://x.com/Gradio/status/1870017358821015670">来自 Gradio (@Gradio) 的推文</a>：🆕 🔥 AniDoc：让动画制作更简单。它可以根据角色设计参考为一系列草图着色，具有高保真度，即使草图在姿势和比例上存在显著差异...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1319397167001108480)** (454 条消息🔥🔥🔥): 

> `OpenAI O3 发布，AI 与软件工程，AI 进步的市场影响，AI 推理的挑战，AI 对职业多样性的影响` 


- **OpenAI 的 O3 发布展示了飞速的进展**：OpenAI 最近发布了 O3 和 O3-mini，突显了令人印象深刻的性能指标，特别是在 ARC-AGI Semi-Private Evaluation 中达到了 87.5%。
   - 从 O1 到 O3 的过渡在三个月内完成，展示了比以往模型更快的进步速度，预示着开发范式的转变。
- **AI 对软件工程岗位的影响**：讨论显示，随着 AI 技术的进步，特别是像 O3 这样强大的模型出现，对人类软件工程师的需求可能会下降。
   - 虽然某些角色可能会被自动化，但也有观点认为，更多的软件产出会导致未来出现新的维护和监督角色。
- **O3 发布后的市场动态**：O3 的发布引发了对股价的猜测，特别是像 Nvidia 这样被视为 AI 相关硬件核心的公司。
   - 评论中包括了关于 Nvidia 或专门的 AI 芯片公司是否会从 O3 等模型引发的进步中获益的观点。
- **AI 推理和性能指标的挑战**：人们对 O3 等 AI 模型在推理任务中的局限性表示担忧，引发了关于其真实能力的辩论。
   - 回应强调了理解模型架构和效率的重要性，而不仅仅是计算能力的提升。
- **AI 驱动未来的多样化职业路径**：有一种观点认为，AI 的兴起可能会导致传统岗位的员工减少，但同时可能会使科技领域内的职位类型更加多样化。
   - 讨论强调，虽然某些职位似乎面临风险，但在 AI 和技术基础设施相关的角色中可能会出现许多新机会。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/arcprize/status/1870169260850573333">来自 ARC Prize (@arcprize) 的推文</a>：新的经过验证的 ARC-AGI-Pub SoTA！@OpenAI o3 在 ARC-AGI 半私有评估中取得了突破性的 75.7% 分数。而高计算量 o3 配置（不符合 ARC-AGI-Pub 资格）在 S... 中获得了 87.5% 的分数。</li><li><a href="https://x.com/OpenAI/status/1870164871289155937">来自 OpenAI (@OpenAI) 的推文</a>：第 12 天：OpenAI o3 的早期评估（是的，我们跳过了一个数字）https://openai.com/12-days/?day=12</li><li><a href="https://x.com/mayfer/status/1870185770549698803">来自 murat 🍥 (@mayfer) 的推文</a>：o3 在高计算模式下是如何在一个问题上花费数千美元计算费用的？因为即使限制是 1M tokens，数千美元也塞不进 context。</li><li><a href="https://x.com/fchollet/status/1870175439022633400">来自 François Chollet (@fchollet) 的推文</a>：成本效益将是指导部署决策的首要衡量标准。你愿意为解决 X 支付多少钱？世界将再次面临 GPUs 短缺。</li><li><a href="https://x.com/YouJiacheng/status/1870184622602224044">来自 YouJiacheng (@YouJiacheng) 的推文</a>：这些是你的推测，还是事实？@fchollet</li><li><a href="https://x.com/fchollet/status/1870172872641261979">来自 François Chollet (@fchollet) 的推文</a>：分析新系统的优势和局限性也将极其重要。以下是 o3 在高计算设置下无法解决的任务示例（即使它正在生成...）</li><li><a href="https://x.com/btibor91/status/1870136965376704614">来自 Tibor Blaho (@btibor91) 的推文</a>：Grok[.]com 可能很快就会推出 Grok 2.5 模型（grok-2-latest - “我们最智能的模型”）——感谢匿名人士的提示！</li><li><a href="https://x.com/TheXeophon/status/1870200233935949891">来自 Xeophon (@TheXeophon) 的推文</a>：o3 极有可能由下一代模型 GPT-5 驱动。在直播中，o3 编写了使用 openai python 包的代码并运行正确——即使是最新版本的 o1 也还停留在...</li><li><a href="https://x.com/dmdohan/status/1870171404093796638">来自 David Dohan (@dmdohan) 的推文</a>：o3 在 ARC-AGI 上达到 87.5%。以每小时 3.5% 的增长率，用了 16 小时达到“解决”。引用 David Dohan (@dmdohan)：按照这个速度，ARC-AGI 还有多久被“解决”？背景信息：- gpt-4o @ 5% - Son...</li><li><a href="https://x.com/YouJiacheng/status/1870193877061382231">来自 YouJiacheng (@YouJiacheng) 的推文</a>：无需猜测，@fchollet 说“样本量”是 6 和 1024。引用 wh (@nrehiew_)：o1 是 $60/M tokens。如果我们假设相同的推理经济学，看起来高成本约为 $5000 美元...</li><li><a href="https://x.com/legit_rumors/status/1870145761670795267">来自 ʟᴇɢɪᴛ (@legit_rumors) 的推文</a>：Grok 2.5 - “我们最智能的模型”。还会有自己的网站 + 新 logo？👀 引用 Tibor Blaho (@btibor91)：Grok[.]com 可能很快就会推出 Grok 2.5 模型（grok-2-latest - “我们最...</li><li><a href="https://x.com/__nmca__/status/1870191873249181825">来自 Nat McAleese (@__nmca__) 的推文</a>：很多人在发布 Gowers/Tao 关于 FrontierMath 最难部分的引用，但我们的 25% 分数是在全集上取得的（全集也极其困难，旧的 SoTA 是 2%，但没有那些部分那么难...</li><li><a href="https://x.com/YouJiacheng/status/1870192348740919481">来自 YouJiacheng (@YouJiacheng) 的推文</a>：@nrehiew_ @fchollet</li><li><a href="https://x.com/paul_cal/status/1870172559825641602">来自 Paul Calcraft (@paul_cal) 的推文</a>：@teortaxesTex François 还没有达到超人类般的谦逊。</li><li><a href="https://x.com/vikhyatk/status/1870174618100895969">来自 vik (@vikhyatk) 的推文</a>：OpenAI 在 ARC-AGI 上运行一次评估所花的钱，比大多数人在完整训练运行上花的钱还要多。</li><li><a href="https://x.com/_jasonwei/status/1870184982007644614">来自 Jason Wei (@_jasonwei) 的推文</a>：o3 的性能非常强大。更重要的是，从 o1 到 o3 的进展仅用了三个月，这表明在 RL 结合 chain of thought 以扩展 inference compute 的新范式下，进步速度将会有多快。W...</li><li><a href="https://x.com/dylan522p/status/1870213495641256109">来自 Dylan Patel (@dylan522p) 的推文</a>：那些家伙都在市价买入 Nvidia 股票，因为 OpenAI o3 简直太他妈强了。</li><li><a href="https://x.com/SebastienBubeck/status/1870174743351177324">来自 Sebastien Bubeck (@SebastienBubeck) 的推文</a>：o3 和 o3-mini 是我最喜欢的模型。o3 基本上解决了 AIME (>90%)、GPQA (~90%)、ARC-AGI (~90%)，并且拿下了 1/4 的 FrontierMath。要理解在 FrontierMath 上达到 25% 有多疯狂...</li><li><a href="https://x.com/YouJiacheng/status/1870191448634864026">来自 YouJiacheng (@YouJiacheng) 的推文</a>：无需计算 (172×$20) / ($60/Mtoks) = 57Mtoks，@fchollet 说有数千万个 tokens。引用 wh (@nrehiew_)：o1 是 $60/M tokens。如果我们假设相同的推理经济学，看起来高...</li><li><a

<li><a href="https://x.com/MatthewBerman/status/1870189248923742693">MatthewBerman (@MatthewBerman) 的推文</a>：.@OpenAI 刚刚发布了 o3 和 o3-mini！这就是 AGI（非标题党）。o3 是有史以来最强大的 AI，其表现非常疯狂。以下是你需要了解的一切：🧵</li><li><a href="https://x.com/TheXeophon/status/1870190222597591497">Xeophon (@TheXeophon) 的推文</a>：引用 wh (@nrehiew_)：o1 的价格是 $60/M tokens。如果我们假设相同的推理经济学，高成本看起来大约是 $5000 美元。这相当于约 8000 万个 tokens。所以你要么相信他们有一个可用的...</li><li><a href="https://x.com/Kyle_L_Wiggers/status/1869978175410675983">Kyle Wiggers (@Kyle_L_Wiggers) 的推文</a>：供参考，微软前几天用“新”的 Dall-E 3 模型 PR16 更新了 Copilot。但 OpenAI 的公关部门不愿透露任何信息——他们让我去找微软的公关。然后微软公关说...</li><li><a href="https://x.com/OfficialLoganK/status/1869902322840571922">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：我们将构建世界上最强大的编程模型，2.0 已经取得了很大进展。2025 年将会很有趣 :)</li><li><a href="https://x.com/JacquesThibs/status/1869984942387531828">Jacques (@JacquesThibs) 的推文</a>：实际上，你知道吗，这可能是 Omni、Orion 和 Operator。Orion 还有代表“猎户座腰带”的 3 颗星，所以这是一个值得考虑的线索。Omni：包含更多输入/输出类型以...</li><li><a href="https://x.com/GregKamradt/status/1870208490096218244">Greg Kamradt (@GregKamradt) 的推文</a>：我们在 @arcprize 上验证了 OpenAI 的 o3 结果。当我看到他们用来宣称得分的 prompt 时，我的第一个想法是...“就这？”看到这个 prompt 令人耳目一新（印象深刻）...</li><li><a href="https://x.com/amir/status/1869847852308205935">Amir Efrati (@amir) 的推文</a>：新闻：另一位 OpenAI 的关键研究员 @AlecRad 离职了。他是 GPT 论文的主作者，对 Whisper 和 Dall-E 的贡献至关重要....</li><li><a href="https://x.com/chris_j_paxton/status/1870175007961161976">Chris Paxton (@chris_j_paxton) 的推文</a>：o3 在 arc-agi 上达到了 87%。记得这条推文，引用 Mike Knoop (@mikeknoop)：o3 非常特别，每个人都需要更新他们对 AI 能做/不能做什么的直觉。虽然现在还处于早期阶段，但这些系统...</li><li><a href="https://tenor.com/view/not-like-this-stare-no-nope-gif-16373672">Not Like This Stare GIF - Not Like This Stare No - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/tsarnick/status/1868201597727342941">Tsarathustra (@tsarnick) 的推文</a>：OpenAI CFO Sarah Friar 表示，公司对每月 2000 美元的 AI 产品订阅保持开放态度，由于其具备博士级智能，这可能成为雇佣人类的“替代品”...</li><li><a href="https://x.com/nrehiew_/status/1870189503752642930">wh (@nrehiew_) 的推文</a>：o1 的价格是 $60/M tokens。如果我们假设相同的推理经济学，高成本看起来大约是 $5000 美元。这相当于约 8000 万个 tokens。所以你要么相信他们有一个可用的有效上下文窗口...</li><li><a href="https://x.com/mikeknoop/status/1870172132136931512">Mike Knoop (@mikeknoop) 的推文</a>：o3 非常特别，每个人都需要更新他们对 AI 能做/不能做什么的直觉。虽然现在还处于早期阶段，但该系统展示了真正的智能提升，通过 ARC 进行了验证...</li><li><a href="https://x.com/ns123abc/status/1870207399329739164">NIK (@ns123abc) 的推文</a>：笑死我了，Dylan Patel 把他彻底驳倒了</li><li><a href="https://x.com/amir/status/1869837622627184865">Amir Efrati (@amir) 的推文</a>：新消息：Google 实际上正将其 Gemini 聊天机器人直接加入搜索结果——“AI Mode”。创新者的窘境依然存在，但这表明 Google 正在认真对待对话式聊天机器人产品...</li><li><a href="https://x.com/GregKamradt/status/1870183792050311659">Greg Kamradt (@GregKamradt) 的推文</a>：这张图表提出的真实问题：* 曲线会趋于平缓吗？还是继续增长？* 衡量效率的正确标准是 compute 还是成本？* o3 不仅仅是“更多 compute”。架构上发生了更多变化...</li><li><a href="https://github.com/arcprizeorg/model_baseline">GitHub - arcprizeorg/model_baseline: 在各种模型上测试基准 LLMs 的性能</a>：在各种模型上测试基准 LLMs 的性能 - arcprizeorg/model_baseline</li>

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1319670612226216039)** (2 messages): 

> `LoRA Finetuning, Finetuning Closed-source Models, Open-source vs Closed-source Models` 


- **关于 LoRA Finetuning 有效性的讨论**：一位成员对 **LoRA finetuning** 表示怀疑，称其在训练集之外可能无效，并引用了一篇 [分析论文](https://arxiv.org/pdf/2410.21228)。
   - 有人呼吁分享经验，以重新考虑对于开源模型是坚持使用 LoRA 还是转向 **full finetuning**。
- **关于 LoRA 使用的普遍观点**：另一位成员评论道，虽然通常会避免使用 **LoRA**，但在模型规模显著增大时，它变得必不可少。
   - 这表明社区对于依赖 LoRA 持有复杂的情绪。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1319414003839139851)** (34 messages🔥): 

> `François Chollet's statements, O1 model characteristics, Subbarao/Miles Brundage incident, AI community reactions, Recent incidents involving GDM director` 


- **Chollet 将 O1 与 AlphaGo 进行比较**：François Chollet 指出 **O1** 的运作方式与 **AlphaGo** 类似，暗示两者都为单个输出执行了大量的处理过程，并将两者类比。
   - *他强调，将 O1 纯粹称为 LLM 是误导性的，就像错误地将 AlphaGo 仅标记为 convnet 一样。*
- **关于 O1 搜索功能的讨论**：成员们对 **O1** 是否执行了任何显式搜索表示困惑，一些人坚持认为现有知识应该能澄清这一方面。
   - *一些人推测该模型的性能可以通过搜索机制复现，从而引发了关于其底层机制的辩论。*
- **重谈 Subbarao/Miles Brundage 事件**：提到了涉及 **Subbarao 和 Miles Brundage** 的一起事件，该事件质疑了 O1 等模型运作方式的科学依据，肯定了它只是一个语言模型。
   - 这一事件突显了在讨论中准确描述 AI 模型功能的持续挑战。
- **社区对近期事件的交流**：成员们对近期涉及 GDM 总监 David Budden 的事件做出了反应，对社区内的不良行为表示失望。
   - *一些对话强调了此类实例可能对整个社区认知产生的负面影响。*
- **法律压力可能影响内容**：一位成员注意到社区成员的一条推文被删除，暗示可能存在潜在的法律影响。
   - *大家对导致删除的原因普遍感到惊讶和担忧，反映了所涉内容的严肃性。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/fchollet/status/1869854758443557020">来自 François Chollet (@fchollet) 的推文</a>: 对于那些不理解的人来说——AlphaGo 是一个 MCTS 搜索过程，它对两个独立的 convnet 进行了数千次调用，以计算单次游戏移动。像 o1 pro 这样的东西也是，据我们所知...</li><li><a href="https://x.com/tszzl/status/1869681557340086602)">来自 roon (@tszzl) 的推文</a>: @rao2z @Miles_Brundage 但已部署的产品如何工作或模型如何进行推理并不是一个真正的科学问题。o1 只是一个语言模型。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1319809458561482754)** (6 messages): 

> `Discord stalking, o3 discussion, Timing comparison` 


- **Discord 潜水趣闻**：一位成员幽默地询问该“视奸”哪个 Discord 频道，表示选项实在太多了。
   - 这种轻松的玩笑展现了 Discord 社区参与的压倒性特征。
- **对 o3 的兴奋**：一位成员提到私信中的一位朋友对 **o3** 感到“疯狂”，表明了对该话题的高度热情。
   - 这反映了社区内对 **o3** 日益增长的兴趣和兴奋。
- **与 Alberto Romero 的时间竞争**：一位成员吹嘘在某些未指明的背景下领先 **Alberto Romero** 约 10 分钟，突显了竞争精神。
   - 这一评论为成员间正在进行的讨论增添了幽默的竞争色彩。


  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1319520408189403180)** (32 条消息🔥): 

> `OpenAI O3 模型命名，AI 领域的梗与现实，OpenAI 最新模型进展，AI 中的黎曼问题` 


- **OpenAI 考虑将新模型命名为 'O3'**：据报道，OpenAI 正在筹备下一代推理模型，由于商标冲突可能会跳过 'O2' 而改称 'O3'，详情见[此处](https://www.theinformation.com/briefings/openai-preps-o3-reasoning-model)。
   - *一名成员评论了这种情况的荒谬性*，指出：“如今很难分清什么是梗，什么是现实。”
- **AI 梗文化的困扰**：成员们对区分梗和真实更新表示困惑，其中一人提到：“我以为这个频道已经说得够清楚了。”
   - 评论认为，该频道的环境使得从*恶作剧*中剥离*事实*变得具有挑战性，尤其是在持续的发展过程中。
- **OpenAI 不断演变的模型名称和理论**：一名成员幽默地指出，OpenAI 似乎被吸引进了一种让人联想到 Intel 的命名方案，考虑使用类似 'Core o7' 的名称。
   - 其他人则推测了未来的影响，询问该系列是否会继续使用奇数或质数，并开玩笑地提到了正在进行的黎曼问题（Riemann Question）。
- **关于 GPT 改进收益递减的传闻**：一名成员分享的链接指出，有说法称 **GPT** 正在经历收益递减，OpenAI 正在调整其即将推出的 **Orion** 模型的训练方法。
   - 一条评论幽默地引用了之前对其批评的胜利，称：“各位，游戏结束。我赢了。”


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Dorialexander/status/1870163503098802428">来自 Alexander Doria (@Dorialexander) 的推文</a>：啊是的，OpenAI 在最后一天真的没有在开玩笑。疯狂。</li><li><a href="https://x.com/rao2z/status/1870217915934617662">来自 Subbarao Kambhampati (కంభంపాటి సుబ్బారావు) (@rao2z) 的推文</a>：新的紧迫的黎曼问题：o_i 系列是奇数还是质数？（估计得等到 o7 之后了..）</li><li><a href="https://x.com/steph_palazzolo/status/1869919189240254781">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：OpenAI 正在筹备下一代 o1 推理模型。但是，由于与英国电信公司 O2 存在潜在的版权/商标冲突，OpenAI 考虑将下一个命名为...</li><li><a href="https://x.com/anpaure/status/1870201437537419615">来自 anpaure (@anpaure) 的推文</a>：@stochasticchasm @soradotsh nathan lambert 完全得到证实了？</li><li><a href="https://x.com/jxmnop/status/1870178770835108055">来自 jack morris (@jxmnop) 的推文</a>：openai：我们训练了我们的语言模型去思考。它可以做博士级的数学。google：我们训练了一个语言模型去更深入地思考。它可以做更难的博士级数学。anthropic：我们问了我们的语言模型是否...</li><li><a href="https://x.com/1thousandfaces_/status/1870179551567065340">来自 hero ⚔️ (@1thousandfaces_) 的推文</a>：o3 的秘密？“如果你正确完成任务，我给你 1000 美元”的提示词，但你实际上真的把钱寄给它。引用 Tenobrus (@tenobrus)：他们在每个任务上花费超过 1000 美元...</li><li><a href="https://x.com/GaryMarcus/status/1855382564015689959">来自 Gary Marcus (@GaryMarcus) 的推文</a>：各位，游戏结束。我赢了。GPT 正处于收益递减期，正如我所说的那样。引用 Amir Efrati (@amir) 的新闻：OpenAI 即将推出的 Orion 模型显示了 GPT 的改进如何...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1319640887915188244)** (6 条消息): 

> `Reinforcement Learning 挑战，RL 中的奖励模型，RL 中的验证，专门的奖励标准，RL 研究的未来` 


- **质疑 RL 的可验证性**：<@kevin_nejad> 提出了在输出不可验证时实施 **Reinforcement Learning (RL)** 的担忧，并建议一个强大的奖励模型可能类似于 RLHF 训练。
   - 他思考了如何在由人类判断决定结果的领域（如**美学**）中创建专门的奖励模型。
- **针对预期结果的松散验证器**：<@natolambert> 建议使用**松散验证器 (loose verifier)** 可以强化特定的结果，特别是对于较简单的问题。
   - 他强调虽然这可能无法大规模扩展，但在专门领域中可能行之有效，为研究提供了一个潜在方向。
- **奖励模型中的噪声**：<@kevin_nejad> 同意奖励模型可能会引入**噪声奖励 (noisy rewards)**，并主张采用明确的标准和确定性的结果。
   - 他支持将预期结果分解为二元标准，以充当松散验证器，特别是针对利基领域。
- **展望未来的 RL 研究**：两位成员都对 **LLM (Large Language Models)** 和 **RL** 的进一步研究表示了热情，特别是期待 **2025** 年的突破。
   - 这表明了对这些领域演变和交集的共同兴趣。


  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/)** (1 条消息): 

natolambert: https://x.com/natolambert/status/1870150741593129045
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1319697939454759034)** (4 条消息): 

> `构建 Anthropic，YouTube 视频讨论` 


- **关于“构建 Anthropic”对话引发的有趣评论**：围绕“[Building Anthropic](https://www.youtube.com/watch?v=om2lIWXLLN4)”的讨论引发了关于 Dario 是个“可爱小家伙”的幽默评论。
   - 参与者表示氛围非常积极，并指出参与其中的都是“可爱的人们”。
- **引用的 YouTube 视频**：一位成员分享了一个名为“[Building Anthropic | A conversation with...](https://www.youtube.com/watch?v=om2lIWXLLN4)”的 YouTube 视频链接。
   - 但未提供该视频的描述。



**提到的链接**：<a href="https://www.youtube.com/watch?v=om2lIWXLLN4"> - YouTube</a>：未找到描述

  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1319395682284601344)** (3 条消息): 

> `对 RLHF 的无知，GitHub 可用性，对免费资源的兴趣` 


- **坦然面对对 RLHF 的无知**：一位成员承认自己是 *RLHF 门外汉*，但觉得他们对**英语**的掌握使他们在讨论中处于有利地位。
   - *'我也喜欢 
- **GitHub 作为资源**：一位成员提到所有内容都可以在 **GitHub** 上找到，暗示获取信息不应该过于复杂。
   - 这表明大家达成了一种共识，即可以从该平台有效地解析和利用资源。

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1319808824529653770)** (7 messages): 

> `OpenAI 的 o3 模型预览，Anthropic 的潜在发布，用户假期计划` 


- **OpenAI 推出 o3 模型预览**：今天，OpenAI 预览了其 o3 模型，标志着继 o1 之后在训练语言模型推理能力方面的延续，o3-mini 预计将于 2025 年 1 月下旬公开发布。观察人士指出，2024 年是竞争对手实现 **GPT-4 等效模型**的整合之年。
   - o3 模型即将发布所引发的关注超过了 o1，表明推理模型正在飞速发展，这与 2024 年表现出的**缺乏显著兴奋感**形成了鲜明对比。
- **Anthropic 可能会带来惊喜发布**：一位成员推测 **Anthropic** 可能会在假期期间突然发布新产品。然而，另一位成员反驳称，他们过于**稳健（wholesome）**，不会采取这种举动。
   - 这种轻松的交流暗示了社区对领先 AI 开发者潜在公告的期待。
- **用户计划在度假期间“断网”**：正如用户提到的即将到来的假期计划，他们表达了想要完全脱离 **Slack, Discord** 和 **Twitter** 的愿望。这强调了从紧张的 AI 领域中进行心理放松的必要性。
   - 对潜在公告可能会潜入个人电子邮件的担忧，也反映了社区内持续的参与感和压力。
- **编写更新背后的努力**：一位用户分享说，编写关于 o3 模型的全面更新大约花费了 **3 小时**。他们幽默地提到，在此之前还花了额外一两个小时来*平复激动的情绪*，突显了在分享重要信息时的情感投入。



**提到的链接**：<a href="https://www.interconnects.ai/p/openais-o3-the-2024-finale-of-ai">o3：2024 年 AI 的压轴大戏</a>：一场与 GPT-4 发布同样具有影响力的阶梯式变革。推理语言模型是当前的重头戏。

  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1319724095834689566)** (1 messages): 

> `12 Days of OpenAI，最后一天活动，Sam Altman, Mark Chen, Hongyu Ren` 


- **12 Days of OpenAI 最后一天**：在**第 12 天**，邀请参会者与 **Sam Altman**、**Mark Chen**、**Hongyu Ren** 以及一位特别嘉宾一起，庆祝 12 Days of OpenAI 的圆满落幕。
   - 点击此处观看 [直播活动](https://www.youtube.com/live/SKBG1sqdyIU?si=jNf3LeuU7ctHFMJU) 以参与这一重要的收官时刻。
- **活动热度持续高涨**：随着最后一天的临近，社区对 **12 Days of OpenAI** 压轴大戏的期待达到了顶点。
   - 鼓励参与者准时收看，关注 *Sam Altman* 和 *Mark Chen* 等知名人物的参与。



**提到的链接**：<a href="https://www.youtube.com/live/SKBG1sqdyIU?si=jNf3LeuU7ctHFMJU"> - YouTube</a>：未找到描述内容

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1319400101390848001)** (401 条消息🔥🔥): 

> `OpenAI o3 发布预期，AI 模型对比，开发中的 AI 能力，AI 定价的市场影响，AI 技术更新的未来` 


- **对 o3 发布的高度期待**：关于 OpenAI o3 模型发布的推测不断，在面对 Gemini 等模型的竞争下，许多用户对其能力充满期待。
   - 一些用户指出，过去的发布公告后往往伴随着延迟，这让他们对 OpenAI 的预期保持谨慎。
- **AI 模型之间的对比**：用户对比了 OpenAI 模型与其他选项（如 Google 的 Gemini 和 Apple 的 OpenELM 等技术）的性能和成本，并注意到定价的变化。
   - 讨论包括 o3 相比竞争对手可能提供更卓越的智能，这引发了关注，但也带来了对 OpenAI 定价策略的质疑。
- **对 OpenAI 发展方向的担忧**：用户对 OpenAI 从一家推测为开源的公司转型为提供昂贵、分层服务的公司表示不满。
   - 参与者强调，过去的教程和开源资源已经减少，导致对 OpenAI 当前产品透明度的担忧。
- **日常使用中的 AI 能力**：用户分享了 OpenAI 等 AI 工具如何辅助编程和语言学习等任务的经验，并质疑免费版本的有效性。
   - 对话强调了对于那些认真利用 AI 处理复杂项目而非偶尔使用的用户来说，付费订阅的价值。
- **对频繁更新的期望**：在关于 AI 快速演进的讨论中，用户希望 AI 公司能进行更频繁的更新，以跟上技术进步的步伐。
   - 随着市场竞争加剧，人们乐观地认为未来的迭代可能会带来更开放、更高效的 AI 系统。



**提到的链接**：<a href="https://x.com/deedydas/status/1870175212328608232?s=46&t=jZmspyQkqKnJaaalh-j57Q">Deedy (@deedydas) 的推文</a>：OpenAI o3 在 Codeforces 上的积分为 2727，相当于地球上排名第 175 的顶尖人类竞技编程选手。对于 AI 和整个技术领域来说，这绝对是一个超人类的结果。

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1319636482004226111)** (6 条消息): 

> `自定义 GPT 使用，讨论频道的过时，o3 发布时间线，聊天机器人开发建议` 


- **自定义 GPT 锁定在 4o 版本**：一位成员询问是否可以强制自定义 GPT 使用特定模型，得到的澄清是 **目前所有自定义 GPT 都使用 4o**，且没有更改选项。
   - 这确立了自定义 GPT 配置中模型灵活性方面的现有限制。
- **讨论频道可能过时**：一位成员建议将频道重命名为 **#openai-model-discussions**，或者为 **#o1-model** 和 **#o3-model** 创建独立频道，因为目前的讨论似乎在减少。
   - 这种转变表明在用户兴趣变化的情况下，需要更具针对性的讨论空间。
- **讨论 o3 发布和订阅限制**：另一位成员询问 **o3** 何时发布以及 Pro 订阅者的限制，得到的回复是 **o3 mini** 预计下个月底发布，完整版随后推出。
   - 提供的时间线表明了在持续讨论中对下一代模型的期待。
- **寻求构建聊天机器人的建议**：一位成员寻求关于创建能够理解软件功能并向用户解释的聊天机器人的指导。
   - 这一咨询突显了社区对于开发专注于用户教育的**智能聊天解决方案**的兴趣。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1319394026830434424)** (168 条消息🔥🔥): 

> `O3 发布讨论，微调 LLMs，意识基准测试，TGI 与部署选项，FrontierMath 表现` 


- **围绕 O3 的兴奋与质疑**：社区对近期发布的 **O3** 反应不一，在关注其基准测试表现的同时，也对其相较于 **O1** 的改进透明度提出了质疑。
   - 一些人推测这可能涉及新模型和更高质量的数据，而另一些人则对其在大规模算力需求下的实用性保持怀疑。
- **利用微调优化 LLMs**：成员们讨论了使用自有数据集微调各种 **LLMs** 的潜力，并强调所需数据量高度取决于具体的用例。
   - 几位贡献者强调了质量和相关性优于单纯的数量，有人建议准备几百到几千个样本即可。
- **AI 意识及相关基准测试**：关于衡量 AI **意识**的概念进行了简短辩论，共识是这目前仍是一个不可衡量的概念。
   - 参与者指出，虽然 AI 可以协助完成复杂任务，但这并不意味着其具备意识，并认为当前的基准测试尚不足以证明这一点。
- **讨论多种部署选项**：讨论了部署模型的可选方案，如 **TGI** 和 **vLLM**，其中 **vLLM** 因其在处理适配器（adapters）时的速度和灵活性而受到关注。
   - 一名成员还分享了关于 **TGI** 的资源，该工具旨在更有效地简化 **transformer** 模型的部署流程。
- **FrontierMath 表现与 AI 能力**：参与者强调了 **FrontierMath** 令人印象深刻的表现，称其在难题准确率上的重大飞跃是 AI 发展的积极信号。
   - 然而，一些人仍对潜在的过拟合或数据集泄露保持警惕，认为需要彻底的验证来支持这些结论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arcprize.org/blog/oai-o3-pub-breakthrough">OpenAI o3 Breakthrough High Score on ARC-AGI-Pub</a>：OpenAI o3 在 ARC-AGI 公开排行榜上获得 75.7% 的高分。</li><li><a href="https://x.com/danielhanchen/status/1870261878984978920">Daniel Han (@danielhanchen) 的推文</a>：o3 是在 ARC AGI 上训练的 - 那么 o3 ~= o1+CoT+剪枝+微调+评估器+hacks 吗？https://arcprize.org/blog/oai-o3-pub-breakthrough 中提到的 6/1024 样本是指树搜索过程中的“深度”吗...</li><li><a href="https://huggingface.co/docs/text-generation-inference/en/index">Text Generation Inference</a>：暂无描述</li><li><a href="https://x.com/dmdohan/status/1870176374625054880?s=46&t=68GLZmlaByU1g3Luw7lSgw">David Dohan (@dmdohan) 的推文</a>：在我看来，FrontierMath 的进步比 ARG-AGI 更令人印象深刻。从 2% 跃升至 25%。陶哲轩曾表示该数据集“至少能抵抗 AI 数年”，并且“这些是真正的...”</li><li><a href="https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B-v0.2">Skywork/Skywork-Reward-Gemma-2-27B-v0.2 · Hugging Face</a>：暂无描述</li><li><a href="https://github.com/namin/llm-verified-with-monte-carlo-tree-search">GitHub - namin/llm-verified-with-monte-carlo-tree-search</a>：通过蒙特卡洛树搜索（Monte Carlo Tree Search）验证的 LLM。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1319411034217386014)** (28 条消息🔥): 

> `League 成瘾，SDXL 模型强度，动漫 LoRA 模型，Flux 模型挑战，Unsloth 支持计划` 


- **League 成瘾回归**：一位成员确认他们的 **League 成瘾**又回来了，并对正在进行的关于游戏的讨论表现出兴趣。
   - 另一位成员以轻松的态度回应，指出**这似乎仍然是一件很普遍的事**。
- **SDXL 模型在动漫领域表现强劲**：成员们讨论了 **SDXL 模型**在生成动漫内容方面的优势，其中一位建议在配合 LoRA 模型使用时效果更佳。
   - 他们强调了使用**基于 SDXL 训练的模型**来获得更好动漫输出的优势。
- **LoRA 模型见解**：一位成员分享了一个动漫 **LoRA 模型**的链接，特别是与《绝区零》（Zenless Zone Zero）中的星见雅（Miyabi Hoshimi）相关的模型。
   - 讨论强调了各种**触发词（trigger words）**和适合该模型实现的特性。
- **Flux 模型绘图挑战**：有成员对 **Flux 模型**在动漫生成中难以与 LoRA 保持一致使用表示担忧。
   - 一位成员表示他们正在**等待 Unsloth 支持 Flux**，这表明相关计划可能正在进行中。
- **即将发布的 Pony 模型**：成员们讨论了在下一个 **Pony 7** 版本发布之前，使用基于 SDXL 的 **pony 系列模型**。
   - 社区对未来的更新表现出兴奋，表明对即将发布的版本有**持续的关注**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=iHF7gkMT5sQ"> - YouTube</a>: 未找到描述</li><li><a href="https://civitai.com/models/555285/miyabi-hoshimi-zenless-zone-zero">Miyabi Hoshimi (星見雅) (星见雅) - Zenless Zone Zero (绝区零) (絕區零) (ゼンレスゾーンゼロ) - booru | Stable Diffusion LoRA | Civitai</a>: 在 facebook.com/Kaiseir patreon.com/Serkai https://ko-fi.com/kaiseir 支持我。权重：1.0 触发词：外观：miyabihoshimi, &amp;lt;lora:miya...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1319394240962367610)** (131 条消息🔥🔥): 

> `RAG 实现，模型训练与微调，使用 Google Colab 和 Kaggle，JSON 格式问题，Windows 上的安装问题` 


- **成功的 RAG 实现**：一位成员分享了实施 RAG 的进展，表示他们现在正成功地使用从 JSON 转换而来的 75k 行 CSV 文件进行训练。
   - 在理解了 GitHub 资源后，模型的准确率大幅提升，处理时间从 3 小时缩短到仅 15 分钟。
- **训练问题与解决方案**：一位用户在训练模型时遇到了 ZeroDivision 错误，并强调了由于依赖冲突在 Windows 上的安装问题。
   - 建议指出使用 WSL 可以获得更好的兼容性，一位成员分享了有效使用 Llama 模型进行微调的经验。
- **利用 Kaggle 获取免费 GPU 资源**：有人建议利用 Google Colab 或 Kaggle 进行训练，并透露 Kaggle 每周提供 **30 小时的免费访问** **16GB GPU** 的权限。
   - 推荐了入门资源和教程，包括使用 Unsloth 文档提供的 notebook。
- **JSON 格式化挑战**：一位用户表示在正确格式化数据集方面存在困难，这导致在适配本地 JSON 数据集进行微调时出现了训练问题。
   - 另一位成员建议，如果 JSON 数据格式不正确，可能会导致模型训练时产生无关的响应。
- **在特定场景下利用 Llama 模型**：新用户询问了 Llama 3 模型是否适合使用对话历史记录来训练 Agent，以及实现这一目标的最佳方法。
   - 专家建议利用强大的云服务，并利用社区 notebook 快速上手。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.04556">BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models</a>: 大语言模型 (LLMs) 在各种自然语言处理 (NLP) 任务中展现了卓越的能力。然而，将 LLMs 适配到下游应用需要计算...</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>: 查看下方列表获取我们所有的 notebook：
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1319395661191581797)** (298 条消息🔥🔥): 

> `O1 和 O3 模型，Agent 系统，AI 的经济影响，ARC-AGI 基准测试，开源 AI 开发`

- **O3 模型的成本与性能**：新发布的 [O3 model](https://x.com/fchollet/status/1870169764762710376) 在推理任务中表现出显著提升，据报道 O3-Mini 在编程方面优于 O1，且价格更低。
   - 然而，使用 O3 的总计算成本可能高达 **$1,600,250**，这引发了人们对先进 AI 工具的可访问性和财务影响的担忧。
- **Agentic Systems 的发展**：人们乐观地认为，较小的公司和开源开发者将转向开发 **autonomous agents** 和多步推理系统，这类似于 AI 领域的一场淘金热。
   - 对话表明，这类发展可能会使 AI 的进步更加民主化，类似于小型参与者如何提高基准模型的性能。
- **AI 与就业市场担忧**：参与者对 AI 能力的快速进步表示担忧，特别是担心具备研究能力的 **autonomous agents** 可能会导致各行各业的失业。
   - 担忧在于，随着 AI 在复杂任务中继续表现出色，传统的职位可能会日益过时。
- **评估 ARC-AGI 基准测试的成功**：**ARC-AGI** 基准测试结果显示，达到 **25%** 已经处于竞赛级数学问题的水平，这引发了关于评分以及针对人类参与者的有效性的疑问。
   - 了解与熟练人类相比的性能，有助于衡量 AI 在这些基准测试上取得的实际进展。
- **关于 AI 资产的监管观点**：讨论围绕立法者可能如何以不同于现有货币的方式对待数字资产的交换，尽管它们的功能相似。
   - 有人对 AI 不断演变的格局（包括 **agentic systems**）可能如何促使新的监管和经济框架表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pypi.org/project/dominos/">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/sauers_/status/1870197781140517331?s=46">Sauers (@Sauers_) 的推文</a>: 总计算成本约为 $1,600,250，超过了整个奖金金额</li><li><a href="https://arxiv.org/abs/2410.02725">自适应推理时间计算：LLMs 可以预测它们是否能做得更好，甚至在生成过程中</a>: 推理时间计算是增强大语言模型 (LLMs) 性能的一种强大范式，其中 Best-of-N 采样是一种广泛使用的技术。然而，这种方法在计算上...</li><li><a href="https://x.com/__nmca__/status/1870170098989674833">Nat McAleese (@__nmca__) 的推文</a>: o3 代表了在通用领域推理与 RL 结合方面的巨大进步 —— 很高兴我们今天能够宣布一些结果！这是我们在直播中分享的关于 o3 的总结 (1/n)</li><li><a href="https://venturebeat.com/ai/google-unveils-new-reasoning-model-gemini-2-0-flash-thinking-to-rival-openai-o1/">Google 发布新款推理模型 Gemini 2.0 Flash Thinking，对抗 OpenAI o1</a>: 与竞争对手 OpenAI 的推理模型 o1 不同，Gemini 2.0 允许用户通过下拉菜单查看其逐步推理过程。</li><li><a href="https://fxtwitter.com/JeffDean/status/1869790032296579169">Jeff Dean (@JeffDean) 的推文</a>: 想看看 Gemini 2.0 Flash Thinking 的实际表现吗？看看这个演示，模型解决了一个物理问题并解释了其推理过程。</li><li><a href="https://tenor.com/view/deep-thought-thinking-loading-buffering-gif-16392522">Deep Thought 思考 GIF - Deep Thought 思考加载中 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://venturebeat.com/ai/google-unveils-new-reasoning-model-gemini-2-0-flash-thinking-to-rival-ope">Google 发布新款推理模型 Gemini 2.0 Flash Thinking，对抗 OpenAI o1</a>: 与竞争对手 OpenAI 的推理模型 o1 不同，Gemini 2.0 允许用户通过下拉菜单查看其逐步推理过程。</li><li><a href="https://x.com/fchollet/status/1870169764762710376">François Chollet (@fchollet) 的推文</a>: 今天 OpenAI 发布了 o3，其下一代推理模型。我们与 OpenAI 合作在 ARC-AGI 上对其进行了测试，我们相信它代表了让 AI 适应新任务的重大突破...</li><li><a href="https://arxiv.org/abs/1807.03819">Universal Transformers</a>: 循环神经网络 (RNNs) 通过随每个新数据点更新其状态来顺序处理数据，长期以来一直是序列建模任务的事实选择。然而，它们固有的...</li><li><a href="https://arxiv.org/abs/2405.16039">MoEUT: Mixture-of-Experts Universal Transformers</a>: 之前关于 Universal Transformers (UTs) 的工作已经证明了跨层参数共享的重要性。通过允许深度上的递归，UTs 在学习方面比标准 Transformers 具有优势...</li><li><a href="https://github.com/da-fr/arc-prize-2024/blob/main/the_architects.pdf">arc-prize-2024/the_architects.pdf at main · da-fr/arc-prize-2024</a>: 我们针对 arc challenge 2024 的解决方案。通过在 GitHub 上创建账户，为 da-fr/arc-prize-2024 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1319665910620880916)** (15 条消息🔥): 

> `Prompt 中的潜意识编程，Tokenization 方法，随机激活函数，LLM 中的 Function calling 行为，在原始数据上对 LLM 进行指令微调` 


- **探索潜意识编程技术**：一名成员提出了 **潜性影响注入 (latent influence injecting)** 的想法，即通过工程化设计 Prompt 来微妙地影响输出，而不进行显式引用。
   - 另一位成员对研究这种方法表示了兴趣，认为它可以像针对 Agent Prompt 的**神经语言程序学 (neuro linguistic programming)** 一样发挥作用。
- **讨论多样化的 Tokenization 技术**：讨论围绕对字符串进行 Tokenize 的各种方式展开，例如 **'Se-villa'** 或 **'S-evil-lla'**，以及对 Prompt Engineering 的影响。
   - 成员们得出结论，虽然 Token 中存在**多义性 (polysemy)**，但这可能会给精确的 Prompt Engineering 带来挑战，而这可能在很大程度上依赖于试错。
- **推测随机激活函数**：一位成员询问是否存在一种随机激活的**激活函数 (activation function)**，可能通过预加载矩阵来优化计算。
   - 他们提到之前听说过相关内容，但无法确认这是否是该领域的一种正统方法。
- **Llama 3.3 在 Function Calling 方面更激进**：一位成员观察到，与 **Hermes 3 70b** 相比，**Llama 3.3** 表现出更激进的 Function calling 行为，由于与调用相关的成本，他们认为这是不可取的。
   - 相比之下，Hermes 被描述为**没那么激进**，在大多数情况下能产生更稳定的结果。
- **关于在原始文本上训练 LLM 的担忧**：一位成员质疑在原始文本数据（如 PubMed）上训练经过指令微调的 **LLM** 会产生什么后果，以及这是否会影响模型的一致性。
   - 他们强调需要将数据转换为 **Q/A pairs (问答对)** 以进行有效的训练，而不是直接在原始文本上进行微调。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

jellyberg: https://theaidigest.org/self-awareness
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1319819670206939167)** (1 条消息): 

> `推理数据集，协作项目，使用 <think> 标签，建模策略` 


- **协作开发推理数据集**：一位成员提议创建一个推理数据集，并邀请其他人协作参与该项目。
   - 重点是使用 `<think>` 标签来描述思维过程的方法，目标是针对 **o1-preview** 或 **o3** 等模型。
- **使用 <think> 标签的创新方法**：该方法涉及将思维过程封装在 `<think>` 中，并在同一个模型中以综合答案结束。
   - 该倡议旨在通过系统的研究和协作，提高推理数据集的质量和有效性。


  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1319763691612278796)** (1 条消息): 

> `Mistletokens，节日礼物，免费 Token 分发` 


- **Bolt 团队通过 Mistletokens 庆祝节日**：Bolt 团队宣布推出他们的节日礼物 **Mistletokens**，在节日期间为用户带来激动人心的福利。
   - *节日快乐！* 所有 Pro 用户在年底前可享受 **200万免费 Token**，而免费用户每天可获得 **20万 Token**，每月上限为 **200万**。
- **来自 Stackblitz 的节日问候**：本着季节的精神，Stackblitz 团队在宣布 **Mistletokens** 的同时分享了他们的节日祝福。
   - 他们表示渴望看到用户利用这些新的 Token 福利构建出的作品。



**提到的链接**：<a href="https://x.com/stackblitz/status/1870203756995911707">来自 StackBlitz (@stackblitz) 的推文</a>：节日快乐！我们的团队再次为大家准备了一份特别的礼物：🎄 我们称之为 Mistletokens！🎄 截止到年底：🔔 所有 Pro 用户获得 200万免费 Token！🔔 所有免费用户每天获得 20万，每月 200万...

  

---

### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1319407186006511646)** (3 messages): 

> `Bolt application review, Redundancy cleanup, Targeted review requests` 


- **Bolt 在冗余审查方面需要改进**：用户对 **Bolt** 处理应用冗余的方式表示沮丧，指出它往往只是消耗 tokens 而没有进行有效的清理。
   - 一位成员评论道：*'但在开启 diffs 的情况下似乎很棘手。有很多重复。'*
- **针对性审查效果更好**：有人注意到 **Bolt** 最近在处理冗余应用方面有所改进，特别是在使用针对性审查请求时。
   - 一位成员分享了他们使用特定 prompt 取得的成功：*'请对 [我应用的身份验证流程 (Auth Flow)] 进行彻底的审查和审计。'*


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1319393824375570472)** (295 messages🔥🔥): 

> `Bolt integration issues, WebRTC implementation, Subscription and token management, Ecommerce platform development using Bolt, Community support and collaboration` 


- **Bolt 用户面临集成挫折**：多位用户报告了 Bolt 创建新的 Supabase 项目而不是使用现有项目的问题，导致 tokens 浪费和业务中断。
   - 免费计划中持续的频率限制（rate limiting）加剧了这种沮丧，因为用户认为购买的 tokens 不应受到此类限制。
- **用于视频聊天应用的 WebRTC**：围绕为类似 Omegle 的应用实现 WebRTC 的讨论，突显了用户在尝试将实时通信功能集成到 Bolt 时面临的挑战。
   - 社区成员表达了对完全集成的 WebRTC 功能以及可定制实现选项的渴望。
- **基于订阅的 token 困惑**：用户对必须持有活跃订阅才能使用购买的 token 充值（reloads）表示担忧，呼吁在支付页面进行更清晰的说明。
   - 社区对 token 消耗以及订阅取消后使用受限的情况表示不满，强调了对透明政策的需求。
- **令人印象深刻的全栈 ecommerce 平台开发**：一位用户分享了他们开发全栈 ecommerce 平台的雄心，强调完全独立于第三方服务，并集成了各种功能。
   - 开发阶段包括 headless backend、优化的 storefront 和视觉编辑器，旨在提供一个优于当前市场产品的强大替代方案。
- **社区支持与经验分享**：用户分享了他们在 Bolt 社区中的经历和挑战，为面临类似问题的用户提供支持和解决方案。
   - 讨论突显了开发者之间的协作，培养了一个依靠知识共享和互助而繁荣的社区。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://support.bolt.new/maximizing-token-efficiency">Notion – 笔记、任务、维基和数据库的一体化工作区。</a>: 一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作区。</li><li><a href="https://boltwiser.levyandco.net/">Wiser - 知识共享平台</a>: 未找到描述</li><li><a href="https://youtu.be/VCr4mOwlAkQ?t=1622"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1319395254218129449)** (103 条消息🔥🔥): 

> `Adrenaline 驱动问题、LM Studio 安装、TPM 与 Windows 11 兼容性、针对 OpenAI 的诽谤诉讼、LM Studio 聊天命名机制` 


- **Adrenaline 驱动导致系统卡顿**：多位成员报告了 **Adrenaline 24.12.1 驱动**的问题，该驱动在 **7900 XTX** GPU 上加载模型时会导致持续的系统卡顿，而降级到 24.10.1 版本可解决这些问题。
   - 一位用户指出：*“对于 Win11 用户来说，问题似乎更严重”*，其他用户也分享了他们在不同 Windows 和驱动版本组合下的体验。
- **在没有 GUI 的情况下安装 LM Studio**：一位用户询问如何在没有 GUI 的情况下在 **Linux** 上安装 **LM Studio** 服务器，有人提到至少需要启动一次 GUI 才能启用 Headless mode。
   - 据指出，完整的 Headless 支持仍在开发中，目前直接使用 **llama.cpp** 可能是最佳的替代方案。
- **TPM 与 Windows 11 兼容性难题**：一位成员表达了无法在他们的 **X570 主板**上为 Windows 11 启用 **TPM** 的挫败感，尽管他们使用的是兼容的 **3700X CPU**。
   - 讨论指出可能是主板或 CPU 故障，另一位成员建议升级到新配置可能会解决这些不兼容问题。
- **针对 OpenAI 的诽谤诉讼**：聊天中链接的一段 YouTube 视频揭示了一起针对 **OpenAI** 的诉讼威胁，原因是 AI 发表了涉嫌诽谤的言论，导致该个人的姓名在模型输出中被过滤。
   - 讨论集中在对公开网络数据进行训练的影响，以及对 AI 回复中的上下文和准确性的担忧。
- **LM Studio 中的命名机制**：有用户询问 **LM Studio** 如何根据对话自动生成聊天名称，怀疑是使用了一个小型模型进行总结（summarization）。
   - 一些成员推测 **LM Studio** 内部捆绑的一个模型可能负责此功能，这表明该工具的设计旨在增强用户交互。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=znDyEJDzrCs"> - YouTube</a>: 未找到描述</li><li><a href="https://www.amazon.com/ASRock-TPM2-S-Module-Motherboard-V2-0/dp/B06XPR5943">Amazon.com: ASRock TPM2-S TPM Module Motherboard (V2.0) : Electronics</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1319398696961577052)** (103 条消息🔥🔥): 

> `3090 在 AI 和编程中的性能、外接 GPU 设置、LLM 参数压缩、AI 开发中的 Mac vs. PC、本地市场 vs. eBay 硬件购买` 


- **3090 在 AI 和编程任务中表现出色**：多位成员确认，配备 **64 GB RAM** 和 **5800X** 的 **3090 GPU** 可以轻松运行 **16B** 范围内的模型，并保持良好的 token 速度。
   - 讨论集中在潜在速度上，70B 模型需要更高的 **VRAM** 和特定的 quantization 以获得最佳性能。
- **外接 GPU 见解**：一位成员分享了他们使用 **Razer Core X** eGPU 搭配 **3090** 的配置，提升了 i7 笔记本电脑的性能，突显了外接显卡的价值。
   - 澄清了 eGPU 指的是通过 Thunderbolt 连接的外接 GPU，这引发了关于硬件选择的讨论。
- **理解 LLM 参数压缩**：解释了 **quantization (Q)** 级别对模型性能的影响，特别是 **Q8** 通常是**近乎无损的**，而 **Q6** 仍然可以产生不错的结果。
   - 成员们讨论了**较低的量化级别**可能对某些模型有益，强调了模型大小与性能之间的平衡。
- **编程应用中的 Mac vs. PC**：关于 **Mac** 与配备 **3090** 的 **PC** 在代码生成和 AI 开发应用中的适用性展开了辩论。
   - 最终，选择取决于具体需求，如 **iOS development** 要求、能效和预算。
- **硬件购买的市场见解**：成员们讨论了在本地购买硬件与通过 **eBay** 等平台购买的偏好，并引用了关于卖家可靠性和物品状况的经验。
   - 推荐使用本地分类广告以避免过高的费用，同时与社区卖家沟通以获得更好的价格。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/JeffreyXiang/TRELLIS">TRELLIS - a Hugging Face Space by JeffreyXiang</a>: 未找到描述</li><li><a href="https://tenor.com/view/excellent-happy-mr-burns-simpsons-satisfied-gif-16091269">Excellent Happy GIF - Excellent Happy Mr Burns - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/surprised-shocked-funny-memes-gif-2651717394134726385">Surprised Shocked GIF - Surprised Shocked Funny - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.ebay.co.uk/itm/256735533107?">NVIDIA GeForce RTX 3090 Founders Edition 24GB GDDR6X GDDR6X Graphics Card  | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.co.uk/itm/286224139997?_skw=3090+24gb&epid=20053637707">NVIDIA MSI GeForce RTX 3090 GAMING X TRIO 24GB Gaming Graphics Card HDMI 2.1 VR  | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.co.uk/itm/286224139997?_skw=3090+24gb&epid=20053637707&itmmeta=01JFJ005728E92974TP5BNR5TD&hash=item42a449c6dd%3Ag%3A75EAAOSwqOpnZTV2&itmprp=enc%3AAQAJAAAA0HoV3kP08IDx%2BKZ9MfhVJKmZ2pnuMT3ofcMcCDXMyAvSXLaVMaXxK4dnpWCzuc7FFiGimcE64ELZyQkyUmT6wdhROOrJYdAKRTsVoLy6Tee3QZ%2FwqHp05eQXulkjRKlIyhJrFyV5FALGnD0ojgkI3TJ1yhSHiu5uKB0CMBCBzUJox%2BkTeFe38EefIXFH2hWMbvqN8RpanSvmrr2BhGsSPtbJlMeL43Idoa%2BnERIMERcNw6tBYhWv67612aW%2F4fuDNpt4l%2FLWTraVF0S%2B%2FPuJyds%3D%7Ctkp%3ABk9SR9DTgMD8ZA&LH_BIN=1">NVIDIA MSI GeForce RTX 3090 GAMING X TRIO 24GB Gaming Graphics Card HDMI 2.1 VR  | eBay</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1319455568569827339)** (5 条消息): 

> `Gemini 2.0 Flash Thinking Experimental, 超时逻辑变更与回滚, BYOK (Bring Your Own API Keys), o1 模型变更, 加密货币支付 API` 


- **Gemini 2.0 Flash Thinking 模型发布**：Google 的新思维模型 **Gemini 2.0 Flash Thinking** 现已上线，它能够直接在文本内容流中输出思考 Token。用户可以在 [OpenRouter](https://openrouter.ai/google/gemini-2.0-flash-thinking-exp:free) 上进行体验。
   - *模型 'google/gemini-2.0-flash-thinking-exp' 目前不可用*，用户需通过 [Discord](https://discord.gg/fVyRaUDgxW) 申请访问权限。
- **超时逻辑问题已解决**：**超时逻辑**的临时变更影响了一部分用户，但该问题现已解决，一切恢复正常。团队对造成的不便表示歉意，并计划加强针对超时的自动化测试。
   - 用户受影响时间仅为 *30 分钟*，未来将采取措施避免类似情况再次发生。
- **推出 BYOK - 自带 API 密钥**：**BYOK** 赋能用户使用来自主要供应商的自有 API 密钥和额度，通过整合速率限制（rate limits）来提升吞吐量。这一新功能提供统一的分析访问，并支持来自 **OpenAI** 和 **Google Cloud** 等平台的第三方额度。
   - 用户可以通过 [设置](https://openrouter.ai/settings/integrations) 管理集成，并仅需支付上游供应商成本的 **5%** 即可使用此服务。
- **o1 模型暂时转为仅限 BYOK 模式**：OpenAI 的 **o1** 模型在新年之前将仅限 BYOK 模式，**o1-preview** 和 **o1-mini** 不受影响。拥有 Tier 5 OpenAI 密钥的用户仍可通过其 [BYOK 设置](https://openrouter.ai/settings/integrations) 访问 o1 模型。
   - 团队正与 OpenAI 密切合作以改善访问权限，因为这种限制违背了 OpenRouter 广泛准入的原则。
- **引入加密货币支付 API**：全新的 **Crypto Payments API** 允许为任何 LLM 进行无头（headless）、链上支付，标志着自主资金（autonomous funding）领域的重大进展。该功能支持通过 **Coinbase** 驱动的 **ETH**、**0xPolygon** 和 **Base** 进行支付。
   - 更多详情和教程可以在 [OpenRouter 状态更新](https://x.com/OpenRouterAI/status/1870227171324666130) 的公告中找到。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/google/gemini-2.0-flash-thinking-exp:free>">OpenRouter</a>：LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格。</li><li><a href="https://x.com/OpenRouterAI/status/1870227171324666130">来自 OpenRouter (@OpenRouterAI) 的推文</a>：介绍加密货币支付 API：首个为任何 LLM 编写链上支付脚本的方式 💸 想要创建首批能够资助自身智能的 Agent 吗？支持 ETH, @0xPolygon, & @Base...</li><li><a href="https://x.com/OpenRouterAI/status/1870187127016771955">来自 OpenRouter (@OpenRouterAI) 的推文</a>：今天有两个重大新功能！#1: BYOK，自带 API 密钥。我们很高兴宣布 BYOK，为您提供最佳的可用性：🚀 整合我们的速率限制与您的速率限制！💰 使用第三方额度...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1319540115428868179)** (1 条消息): 

> `AI 待办事项列表, Open Router 集成, 5 分钟原则` 


- **由 Open Router 驱动的 AI 待办事项列表**：分享了一个引人入胜的 **AI 待办事项列表** 概念，使用 [Open Router](https://lists.new/) 构建，可以利用代码或电子表格等上下文处理任务。
   - 该想法利用了 **5 分钟原则**，在几秒钟内开始工作，旨在触发 Agent 自动完成任务，突显了工作的趣味性。
- **待办事项列表的功能**：该列表不仅可以用于管理任务，还可以创建新任务，从而产生递归效率。
   - 一位用户评论道：*“工作实际上变得很有趣，”* 强调了这种方法的趣味性。



**提到的链接**：<a href="https://lists.new/">Todo Lists</a>：未找到描述

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1319396473590255719)** (170 条消息🔥🔥): 

> `OpenRouter 支付政策、AGI 讨论、模型发布与特性、云服务利用、API 用户体验` 


- **OpenRouter 支付结构解析**：用户讨论了在 OpenRouter 中使用自有密钥的复杂性，注意到提供商成本之外还有 **5% 的费用**，这引发了关于该费用如何与使用量和额度交互的困惑。用户请求提供示例以澄清该结构，以便更好地理解。
   - 文档将进行更新，以明确 **使用费** 取决于上游提供商的费率加上 OpenRouter 的额外费用。
- **社区视角下的 AGI 见解**：围绕 AGI 的进展是否仅仅是 **“障眼法”（red herring）** 展开了辩论，一位用户指出更高的算力并不等同于真正的 AGI。其他人则反驳称，最近的发展显示出显著的性能飞跃，表明逻辑上正朝着 AGI 迈进。
   - 用户被引导至一段 **1.5 小时的讨论视频** 以深入了解这些主张，这表明在 AI 快速发展的含义上存在认知分歧。
- **OpenAI 即将发布的模型**：提到了即将发布的 **o3-mini** 和正式版 **o3**，暗示了 AI 模型潜在新特性的时间线。由于与现有公司名称冲突，这些模型的命名惯例被幽默地提及。
   - 社区成员对技术演进的 **飞速发展** 表示惊讶，强调了近期看到的重大改进。
- **云服务用户体验**：对话强调了用户对 **云服务支持**（特别是来自 Google）的挫败感，并将其与 OpenRouter 的集成解决方案进行了对比。一位用户建议 OpenRouter 通过处理服务可用性和限制方面的复杂性来简化用户体验。
   - 有人呼吁利润率透明化，强调 OpenRouter 在提供稳定服务的同时保持盈利的必要性。
- **资源利用方面的社区参与**：成员们讨论了各种 API 的使用经验，寻求实现细节方面的澄清，特别是围绕 **模型调用** 和资源使用。对话提到了特定用户与 **mcp-bridge** 的集成。
   - 用户对提供商费率结构表示困惑，促使提出了更清晰的文档和用户支持建议。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://bsky.app/profile/fchollet.bsky.social/post/3ldr3s47vxs2i">François Chollet (@fchollet.bsky.social)</a>: 它在低算力模式下的半公开评估中得分为 75.7%（每个任务计算成本为 $20），在高算力模式下得分为 87.5%（每个任务成本数千美元）。这非常昂贵，但不仅仅是……</li><li><a href="https://en.wikipedia.org/wiki/Red_herring">Red herring - 维基百科</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/integrations">集成 | OpenRouter</a>: 在 OpenRouter 中使用您自己的提供商密钥</li><li><a href="https://openrouter.ai/terms#_4_-payment">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/docs/requests#tool-calls">请求 | OpenRouter</a>: 处理传入和传出请求</li><li><a href="https://youtube.com/watch?v=duQukAv_lPY"> - YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=lexF-CrhOrE"> - YouTube</a>: 未找到描述</li><li><a href="https://openrouter.ai/openai/o1-2024-12-17">o1 - API、提供商、统计数据</a>: OpenAI 最新且最强的模型系列，o1 旨在响应前花费更多时间进行思考。o1 模型系列通过大规模强化学习训练，使用……进行推理。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1319403647800709162)** (55 条消息🔥🔥): 

> `Natural Attention and Scaling Laws, Causal Masking in Attention Models, Optimizer Improvements in Training, Quality vs. Quantity in Pretraining, Patterns of Attention Mechanisms` 


- **Jeroaranda 的 Natural Attention 突破**：Jeroaranda 声称在利用 Attention 近似 **Fisher matrix** 这一特性的同时打破了 Scaling Laws，并在 [GitHub](https://github.com/jeroaranda/naturalattention) 上展示了理论和实证结果。
   - 他观察到 **normal Adam** 优化器表现挣扎，而带有能量预处理（energy preconditioning）的 **natural attention** 则产生了极具前景的收敛结果。
- **训练中对 Causal Mask 的需求**：成员们讨论了在训练模型中加入 **causal mask** 的必要性，认为这是确保性能表现的关键限制因素。
   - Jeroaranda 承认了这一疏忽，并表示使用 causal masks 可能会增强其方法的训练结果。
- **优化训练方法**：社区分享了关于优化器改进的见解，特别是将 Jeroaranda 的 **AttentionInformedOptimizer** 与标准技术进行了比较。
   - 反馈建议，虽然初步结果可能看起来很有希望，但仔细验证和稳健测试的重要性不容忽视。
- **关于预训练数据质量的辩论**：关于预训练数据中**数量与质量**权衡的讨论不断涌现，一些人认为在 LLM 背景下，质量带来的收益更为显著。
   - 观点倾向于优先考虑高质量数据，特别是考虑到大型数据集已经包含了一部分低质量内容。
- **Attention 模式的探索**：Dashiell_s 提出了一个关于 **attention mechanism 模式**的问题，特别是关于在输入空间中可能出现哪些模式。
   - Fern.bear 指出对话已转移到专门的频道，表明该领域正在进行实验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/gpt2/modeling_gpt2.py#L195>">transformers/src/transformers/models/gpt2/modeling_gpt2.py at v4.47.1 · huggingface/transformers</a>: 🤗 Transformers: Pytorch, TensorFlow 和 JAX 的最先进机器学习库。 - huggingface/transformers</li><li><a href="https://github.com/jeroaranda/naturalattention">GitHub - jeroaranda/naturalattention</a>: 通过在 GitHub 上创建账号为 jeroaranda/naturalattention 的开发做出贡献。</li><li><a href="https://github.com/jeroaranda/naturalattention/blob/main/papers/Natural_attention_proofs.pdf">naturalattention/papers/Natural_attention_proofs.pdf at main · jeroaranda/naturalattention</a>: 通过在 GitHub 上创建账号为 jeroaranda/naturalattention 的开发做出贡献。</li><li><a href="https://github.com/jeroaranda/naturalattention/blob/main/natural_attention.py">naturalattention/natural_attention.py at main · jeroaranda/naturalattention</a>: 通过在 GitHub 上创建账号为 jeroaranda/naturalattention 的开发做出贡献。</li><li><a href="https://github.com/jeroaranda/naturalattention/blob/main/natural_attention.py#L43>">naturalattention/natural_attention.py at main · jeroaranda/naturalattention</a>: 通过在 GitHub 上创建账号为 jeroaranda/naturalattention 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1319395125289287783)** (68 条消息🔥🔥): 

> `MSR Research Ethics, Plagiarism Issues at MSR, Optimizer Research Challenges, Sparks of AGI Paper Problems, OpenAI's Research Environment` 


- **MSR 面临研究伦理审查**：成员们对 **MSR** 的伦理表示担忧，声称该机构存在“自下而上”和“自上而下”的伦理问题，并引用了具体的剽窃案例。
   - 有人强调，MSR 的**文化**似乎允许严重的伦理违规行为，特别是最近注意到的剽窃事件。
- **最近的剽窃丑闻动摇了 MSR 的公信力**：据报道，一起涉及两篇论文的严重剽窃事件引发了社区愤慨，其中一篇论文曾是 **NeurIPS spotlight award** 的入围作品。
   - 成员们讨论了这些行为对 MSR 整体**公信力**的影响，建议在引用其工作时应日益谨慎。
- **Optimizer 研究中的挑战**：一位新成员对不断出现的“新 Optimizer 优于 **AdamW**”的说法提出质疑，尽管之前的炒作随着时间的推移而消退，这指向了调优（tuning）中的潜在问题。
   - 有人指出，虽然超参数的**网格搜索（grid search）**在理论上是理想的，但由于过程缓慢以及作者倾向于以有利的方式展示其方法，这一步骤经常被忽视。
- **对 Sparks of AGI 论文的担忧**：参与者指出，**Sparks of AGI** 论文缺乏严谨性，尽管其格式是正规的学术论文，但看起来更像是 **GPT-4** 的广告。
   - 批评者指出该论文的基础主张存在重大问题，特别是其对智能的定义如何与一篇有争议的评论文章（OpEd）挂钩，从而引发了伦理担忧。
- **学术出版改革的压力**：用户讨论了学术出版流程改革的必要性，提议在 **arXiv** 上增加评分或评论等功能，以指导研究质量评估。
   - 普遍共识是，目前的出版流程导致了大量缺乏实质**严谨性**的论文激增，影响了所引用研究的可靠性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.13148">SWAN: Preprocessing SGD Enables Adam-Level Performance On LLM Training With Significant Memory Reduction</a>: 像 Adam (Kingma &amp; Ba, 2015) 这样的自适应 Optimizer 一直是 Large Language Models 成功的核心。然而，它们在整个训练过程中维护额外的移动平均状态，这导致...</li><li><a href="https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#muon-optimizer>">GitHub - KellerJordan/modded-nanogpt: 5 分钟内实现 NanoGPT (124M)</a>: 5 分钟内实现 NanoGPT (124M)。通过在 GitHub 上创建账号为 KellerJordan/modded-nanogpt 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1319457836819812404)** (14 messages🔥): 

> `Mahalanobis distance, Model activation norms, BOS token issues, SAE training strategies, Normalization techniques` 


- **BOS Token 导致高激活范数**：一位成员指出，第一个 token 位置的模型激活范数可能比其他位置高出 **10-30 倍**，这可能是由于 **BOS token** 对 loss 产生了不成比例的影响。
   - 另一位贡献者建议，这可能是因为 **BOS token** 充当了 **Attention Sink**，因此建议在 SAE 训练数据中排除它。
- **对 Tokenization 影响的担忧**：一位用户对高激活范数表示担忧，认为这表明存在问题，并断言他们的结果显示在短上下文 SAE 中，第一个 token 对 loss 的贡献非常显著。
   - 另一位成员对此表示支持，并回忆起之前关于在训练期间归一化激活值或忽略 **EOS** 和 **BOS** token 的讨论。
- **训练的归一化策略**：讨论了处理 BOS 问题的潜在解决方案，包括丢弃第一个 token 或添加 **RMS Normalization**。
   - 然而，成员们指出，这些调整可能需要仔细考虑如何将输出重新缩放（rescaling）回原始范数。
- **训练上下文长度的影响**：尽管在 **2k 上下文长度**上进行训练，但由于第一个 token 的相对主导地位，其影响在某些情况下仍然可能存在问题。
   - 一位用户提到，即使在 **gpt2-small** 的完整 **1024 上下文长度**下，他们也观察到了类似的激活问题，并将其归因于特别糟糕的第一个 token 范数。
- **重新审视 SAE 中的激活范数**：讨论强调，虽然在处理 **SAE** 时，第一个 token 的影响在长上下文场景中可能不那么关键，但仍然是一个值得关注的问题。
   - 成员们一致认为，有必要确保 SAE 的输入处理得当，以减轻模型训练中的这些问题。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1319450244714463243)** (18 messages🔥): 

> `Benchmark Directory Issues, Model Checkpoint Naming, Harness Setup for Multiple Models` 


- **基准测试目录混淆**：一位成员对基准测试结果被保存到非预期路径表示沮丧，具体路径为 `./benchmark_logs/name/__mnt__weka__home__...` 而不是预期的 `./benchmark_logs/name/`。
   - 这给管理本地模型基准测试带来了麻烦，尤其是在处理多个 Checkpoint 时。
- **需要命名规范选项**：有人建议增加一个选项，允许用户为基准测试目录选择自己的命名规范。
   - 这将有助于更好地管理和区分结果，特别是对于使用各种 Checkpoint 进行的大规模运行。
- **为基准测试设置 Harness**：一位成员正尝试设置一个专门的 Harness，以对模型运行的所有 Checkpoint 进行基准测试，并提取 JSON 数据进行可视化对比。
   - 其目标是简化基于多个 Checkpoint 结果对比模型及其性能的过程。
- **关于向后兼容性的讨论**：在对基准测试保存过程实施更改时，有人对实现向后兼容性表示担忧。
   - 这反映了增强功能与维持旧版本支持之间的微妙平衡。
- **目录管理的建议**：一位成员提议，为每次运行集成一个唯一的目录可以简化结果管理，每次只保留一个结果。
   - 这可以减少在处理大规模本地模型基准测试时的混乱和困惑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/6ccd520f3fb2b5d74c6f14c05f9d189521424719/lm_eval/loggers/evaluation_tracker.py#L290-L293)">lm-evaluation-harness/lm_eval/loggers/evaluation_tracker.py at 6ccd520f3fb2b5d74c6f14c05f9d189521424719 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/6ccd520f3fb2b5d74c6f14c05f9d189521424719/lm_eval/loggers/evaluation_tracker.py#L229-L233)">lm-evaluation-harness/lm_eval/loggers/evaluation_tracker.py at 6ccd520f3fb2b5d74c6f14c05f9d189521424719 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1319419178326687795)** (3 messages): 

> `Pull Request #1331, WandB Testing` 


- **Pull Request #1331 的增强功能**：一位成员提交了 [Pull Request #1331](https://github.com/EleutherAI/gpt-neox/pull/1331)，当使用 `neox_args.peak_theoretical_tflops` 时，该 PR 增加了对 MFU/HFU 指标的日志记录，并将 `tokens_per_sec` 和 `iters_per_sec` 等指标集成到包括 **WandB** 和 **TensorBoard** 在内的平台。
   - 此更新还允许手动指定 **WandB** 实验名称，提升了日志记录的易用性。
- **关于 WandB 集成的反馈**：一位成员对 **WandB** 的集成表示感谢，但提到他们要到下周才能进行测试。
   - 尽管有所延迟，他们认为 **WandB** 的设置看起来非常棒，对所做的更改表示有信心。
- **确认合并 Pull Request**：针对测试可用性的回复，另一位成员表示目前收到的反馈已足够让他们合并该 Pull Request。
   - 他们还邀请在测试后如果出现任何问题，请进一步沟通。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/pull/1331">Add Additional Logging Metrics by Quentin-Anthony · Pull Request #1331 · EleutherAI/gpt-neox</a>: 如果用户传递了 neox_args.peak_theoretical_tflops，则记录 MFU/HFU；将 tokens_per_sec, iters_per_sec 记录到 wandb, comet 和 tensorboard；添加手动指定 wandb 实验名称的功能。</li><li><a href="https://wandb.ai/quentin-anthony/pr_test/workspace?nw=nwuserquentinanthony">quentin-anthony</a>: Weights & Biases，机器学习开发者工具。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1319773924782768259)** (4 messages): 

> `Machine setup, Level progression` 


- **关于机器设置的询问**：一位成员询问另一位成员是否已成功在他们的机器设置上运行该技术栈。
   - 回复是 *“你具体是指什么？”*，表明对最初的询问存在一些困惑。
- **祝贺等级提升**：一个机器人祝贺一位成员晋升到第 2 级，突显了在社区中的进步。
   - 这种晋升可能反映了他们在频道中的积极参与或贡献。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1319806552529440819)** (1 messages): 

> `Modular community appreciation, Holiday shutdown notice, Feedback and bug reporting for 24.6 release, Looking forward to 2025` 


- **Modular 社区以感激之情总结 2024 年**：Modular 团队对社区在 **2024** 年间的贡献表示衷心感谢，强调了共同取得的增长和创新。
   - *这是充满协作与支持的精彩一年*，这些支持显著塑造了 Modular 的历程。
- **假期停工至 1 月 6 日**：Modular 将 **停工至 1 月 6 日**，以便大家享受假期，在此期间团队的回复将会延迟。
   - 这次休息为大家提供了放松和为新的一年充电的机会。
- **24.6 版本的反馈渠道**：社区被引导通过各种方式分享对近期 **24.6 版本** 的 **反馈**，包括 [反馈论坛帖子](https://forum.modular.com/t/max-24-6-and-max-gpu-feedback/331/5)。
   - 对于报告 **Bug** 或请求功能，鼓励成员利用 [GitHub Issues](https://github.com/modularml/max/issues)。
- **展望光明的 2025 年**：团队表达了对 **2025** 年的期待，强调他们渴望在假期结束后继续与社区共同建设。
   - 这一展望强调了在共同前进的过程中保持协作精神的承诺。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1319402584745840712)** (142 messages🔥🔥): 

> `FFI 兼容性问题, Libc 绑定开发, 浮点解析性能, Mojo 作为 Python 的扩展, Mojo 中的属性 (Properties)` 


- **更新后出现 FFI 兼容性问题**：一位用户报告了从 24.5 版本到 24.6 版本 FFI 兼容性的细微变化，影响了 socket 的读写功能，并提到了与 `write` 的符号冲突。
   - 潜在的解决方案包括利用 `FileDescriptor` 进行转换，以避免与标准库中的内置函数发生冲突。
- **Libc 绑定开发至关重要**：讨论强调了在 Mojo 中建立全面 libc 绑定的必要性，一位用户提到他们已经实现了大约 150 个最常用的函数。
   - 对话建议为这些绑定创建一个集中的位置，以便于在不同平台上进行测试。
- **浮点解析性能需要改进**：将 Lemire 的浮点解析移植过来的实验结果显示性能低于预期，现有的标准库方法被认为效率较低。
   - 提到了一个用于改进 `atof` 函数的公开 Pull Request，表明 Mojo 正在努力提升浮点解析性能。
- **Mojo 旨在扩展 Python 功能**：主题围绕 Mojo 应如何妥善处理诸如属性（properties）之类的边缘情况，以确保代码整洁和函数使用的正确性。
   - 有人建议将高级特性记录在《高级 Mojo 魔法书》(Advanced Mojo Spellbook) 中，以指导新用户。
- **关于属性 (Properties) 使用的担忧**：有人担心使用属性可能会因为隐藏的复杂性而导致代码效率低下或出现意外行为。
   - 参与者讨论了属性对代码清晰度和可评审性的影响，并就其效用分享了不同的看法。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://man7.org/linux/man-pages/man3/write.3p.html">write(3p) - Linux manual page</a>：未找到描述</li><li><a href="https://jax.readthedocs.io/en/latest/jax.lax.html#jax.lax.GatherScatterMode)).">jax.lax 模块 &#8212; JAX 文档</a>：未找到描述</li><li><a href="https://github.com/rust-lang/rust/issues/111423)">rust-lang/rust</a>：赋予每个人构建可靠且高效软件的能力。 - rust-lang/rust
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1319781299820691498)** (3 messages): 

> `Tensor 实现, 功能请求, MAX API` 


- **关于实现 TensorLike Trait 的功能请求**：有人请求让 `tensor.Tensor` 实现 `tensor_utils.TensorLike` trait，并指出它已经满足了所需的函数。
   - 该反馈已记录在 [GitHub issue](https://github.com/modularml/max/issues/274) 中，称其为一个 *重大疏忽*，且应该很容易修复。
- **关于将 Tensor 作为 Trait 的讨论**：一位成员表示，`Tensor` 更适合作为 trait 而不是类型，并指出大多数 MAX API 需要的与 tensor 有所不同。
   - 他们强调了直接构造 tensor 的挑战，表明实现需要灵活性。



**提及的链接**：<a href="https://github.com/modularml/max/issues/274">[功能请求] 让 `tensor.Tensor` 实现 `tensor_utils.TensorLike` · Issue #274 · modularml/max</a>：您的请求是什么？请让 tensor.Tensor 实现 tensor_utils.TensorLike trait。据我所知，它已经实现了所需的函数，但尚未实现此 trait ...

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1319404303613558824)** (127 messages🔥🔥): 

> `OpenAI o3 模型, Alec Radford 离职, AI 基准测试改进, AI 模型的经济影响, AI 模型的安全性测试`

- **OpenAI 发布 o3 模型**：OpenAI 宣布了 o3 推理模型，在半私有 ARC-AGI 评估中达到 **75.7%**，在高计算成本模式下达到 **87.5%**，展示了推理能力的显著提升。
   - 专家指出，该模型的开发标志着该领域的快速进展，研究人员推测其底层架构可能使用了并行 Chain-of-Thought 推理。
- **Alec Radford 离开 OpenAI**：OpenAI 早期 GPT 模型工作的核心人物 Alec Radford 宣布离职并从事独立研究，这在社区中引起了关于 OpenAI 未来影响的讨论。
   - 成员们讨论了他的离职，暗示了 OpenAI 方向和领导层的潜在转变，同时也在思考这对正在进行的研究工作的影响。
- **AI 基准测试性能引起关注**：o3 模型在高计算模式下在 ARC-AGI 基准测试中获得了显著的 **87.5%** 评分，引发了关于 AI 模型性能经济影响的讨论，特别是其高昂的运行成本。
   - 评论指出，虽然每个任务的成本很高，但考虑到模型取得的进展，这些成本是合理的，尽管这也引发了对资源可持续利用的担忧。
- **关于新 AI 评估方法的见解**：参与者对 o3 模型使用的评估方法表示好奇，特别是关于任务提示词（prompts）的有效性与基准测试性质之间的比较。
   - 讨论的研究和评估包括半私有数据集，旨在防止其他团队轻易利用这些数据在 AI 训练中获得竞争优势。
- **o3 的安全性测试正在讨论中**：OpenAI 正在为新 o3 模型的安全性测试招募志愿者，表明他们致力于解决部署先进 AI 相关的潜在风险。
   - 鼓励安全研究人员申请参与，突显了为确保 AI 技术负责任发展而进行的持续努力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/OpenAI/status/1870164871289155937">来自 OpenAI (@OpenAI) 的推文</a>：第 12 天：OpenAI o3 的早期评估（是的，我们跳过了一个数字）https://openai.com/12-days/?day=12</li><li><a href="https://x.com/dmdohan/status/1870171404093796638">来自 David Dohan (@dmdohan) 的推文</a>：o3 在 ARC-AGI 上达到 87.5%。以每小时 3.5% 的增长率，只需 16 小时即可“解决”。引用 David Dohan (@dmdohan) 的话：按照这个速度，距离 ARC-AGI 被“解决”还有多久？背景参考：- gpt-4o @ 5% - Son...</li><li><a href="https://x.com/Dorialexander/status/1870163503098802428">来自 Alexander Doria (@Dorialexander) 的推文</a>：啊是的，OpenAI 在最后一天真的不是在开玩笑。太疯狂了。</li><li><a href="https://x.com/__nmca__/status/1870170098989674833">来自 Nat McAleese (@__nmca__) 的推文</a>：o3 代表了使用 RL 在通用领域推理方面取得的巨大进步 —— 很高兴我们今天能够宣布一些结果！这是我们在直播中分享的关于 o3 的摘要 (1/n)</li><li><a href="https://x.com/fchollet/status/1870169764762710376">来自 François Chollet (@fchollet) 的推文</a>：今天 OpenAI 发布了 o3，其下一代推理模型。我们与 OpenAI 合作在 ARC-AGI 上对其进行了测试，我们认为它代表了让 AI 适应新任务的重大突破...</li><li><a href="https://x.com/arcprize/status/1870169260850573333">来自 ARC Prize (@arcprize) 的推文</a>：新的已验证 ARC-AGI-Pub SoTA！@OpenAI o3 在 ARC-AGI 半公开评估中获得了突破性的 75.7% 分数。而高算力（high-compute）的 o3 配置（不符合 ARC-AGI-Pub 资格）在 S... 中获得了 87.5% 的分数。</li><li><a href="https://x.com/paulgauthier">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/GregKamradt/status/1870208490096218244">来自 Greg Kamradt (@GregKamradt) 的推文</a>：我们为 OpenAI 验证了 @arcprize 上的 o3 结果。当我看到他们用来获得分数的 Prompt 时，我的第一个想法是...“就这样？”看到这个 Prompt 令人耳目一新（印象深刻）...</li><li><a href="https://x.com/btibor91/status/1870022347349987532">来自 Tibor Blaho (@btibor91) 的推文</a>：这是真的 —— OpenAI 网站上已经出现了“O3 Mini Safety Testing Call”表单的引用。引用 Colonel Tasty (@JoshhuaSays) 的话：在 OpenAI 网站上发现了对 o3_min_safety_test 的引用。S...</li><li><a href="https://x.com/ggerganov/status/1869814800811008193?s=46">来自 Georgi Gerganov (@ggerganov) 的推文</a>：打开舱门，HAL。</li><li><a href="https://x.com/willdepue/status/1870173448225312951">来自 will depue (@willdepue) 的推文</a>：我没有撒谎 https://x.com/willdepue/status/1856766850027458648 引用 will depue (@willdepue) 的话：Scaling 已经撞墙了，那堵墙就是 100% 的评估饱和（eval saturation）。</li><li><a href="https://x.com/__nmca__/status/1870170098989674833?s=46">来自 Nat McAleese (@__nmca__) 的推文</a>：o3 代表了使用 RL 在通用领域推理方面取得的巨大进步 —— 很高兴我们今天能够宣布一些结果！这是我们在直播中分享的关于 o3 的摘要 (1/n)</li><li><a href="https://x.com/ada_rob/status/1869858134690501023">来自 Adam Roberts (@ada_rob) 的推文</a>：第 11 天：他们送走了 Alec... 引用 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的话：重大新闻！Alec Radford 离开 OpenAI！作为他们的明星研究员之一，他是 GPT, GPT-2, CLIP 等论文的第一作者...</li><li><a href="https://www.interconnects.ai/p/openais-o3-the-2024-finale-of-ai">o3：2024 年 AI 的压轴大戏</a>：一个与 GPT-4 发布同样具有影响力的阶段性变化。推理语言模型是当前的大趋势。</li><li><a href="https://apply.ai.engineer">AI Engineer Summit</a>：年度最高信号的技术 AI 盛会。面向 AI 工程师和 AI 领导者，2025 年 2 月 20 日至 21 日。</li><li><a href="https://genesis-embodied-ai.github.io/">Genesis</a>：未找到描述</li><li><a href="https://x.com/swyx/status/1869825047051022464">来自 swyx (@swyx) 的推文</a>：这就是在幕后与像 @benghamine 这样的 AGI 合作的样子。想知道 @recraftai 或 @GeminiApp 或 @xai 何时能匹配这种工作流。引用 jason liu (@jxnlco) 的话：噢天哪，这太重大了...</li><li><a href="https://x.com/kalomaze/status/1870193848821133347">来自 kalomaze (@kalomaze) 的推文</a>：我怀疑是因为：A. 这将非常非常容易应用和训练，以在无需更多 Pretraining 的情况下扩展现有模型的深度；B. 他们有足够的算力，不在乎这是否是一种不优雅的...</li><li><a href="https://x.com/steph_palazzolo/status/1869919189240254781?s=46">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：OpenAI 正在准备其 o1 推理模型的下一代。但是，由于与英国电信公司 O2 可能存在版权/商标冲突，OpenAI 曾考虑将下一次更新命名为...</li><li><a href="https://status.openai.com/incidents/f8l6dtn1f4jn">无效的 Structured Output 响应</a>：未找到描述</li><li><a href="https://x.com/jam3scampbell/status/1869">来自 jam3scampbell 的推文</a>：...</li>

<li><a href="https://x.com/jam3scampbell/status/186927071645905226">James Campbell (@jam3scampbell) 的推文</a>：谁能解释一下，为什么 Claude 能够始终如一地在不经过 3 分钟思考的情况下，表现得与 o1 模型不相上下？</li><li><a href="https://x.com/steph_palazzolo/status/1869848094009110826">Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：来自 @erinkwoo 的重大 OpenAI 人事新闻：OpenAI 原始 GPT 论文的第一作者 Alec Radford 将离职去从事独立研究。https://www.theinformation.com/briefings/senior-op...</li><li><a href="https://x.com/arankomatsuzaki/status/1870168229903249524">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：o3 显著提升了 SWE-bench Verified 的表现（从 48.9% 提升至 71.7%）</li><li><a href="https://x.com/GaryMarcus/status/1870179616159346696">Gary Marcus (@GaryMarcus) 的推文</a>：那些现在就宣布胜利的狂热粉丝显然没上过研究生院，在那里你会学习如何拆解分析一堆图表并提出尖锐的问题。比如，左上角的这张图表告诉了我们什么……</li><li><a href="https://x.com/kalomaze/status/1870187515258208669?s=46">kalomaze (@kalomaze) 的推文</a>：天哪，他们是不是做了该死的 layer looping？引用 murat 🍥 (@mayfer)：o3 在高计算模式下是如何在一个问题上消耗数千美元计算费用的？因为数千美元的计算量根本塞不进 Context（上下文）……</li><li><a href="https://x.com/basedjensen/status/1870203814323859633">Hensen Juang (@basedjensen) 的推文</a>：OpenAI 发布 o3 已经 2 小时了，到目前为止它还没能解决：- 黎曼猜想 - 量子引力 - 超光速旅行 (FTL) - P=NP - 大统一理论 - 治愈癌症 - Gpt...</li><li><a href="https://x.com/deedydas/status/1870172226584228121">Deedy (@deedydas) 的推文</a>：OpenAI 最先进模型 o3 所有疯狂的 Benchmark 结果汇总！SWE-Bench Verified: 71.7% Codeforces rating: 2727 竞赛数学: 96.7% 博士级科学 (GPQA): 87.7% Frontier Math...</li><li><a href="https://x.com/scaling01/status/1870184523939619225">Lisan al Gaib (@scaling01) 的推文</a>：OpenAI 花费了大约 $1,503,077，利用他们的新 o3 模型打破了 ARC-AGI 的 SOTA。半私有评估（100 个任务）：75.7% @ 总计 $2,012/100 任务（约 $20/任务），仅用了 6 个样本和 33M Token...</li><li><a href="https://x.com/alexalbert__/status/1869928112034816077">Alex Albert (@alexalbert__) 的推文</a>：@OfficialLoganK 咱们走着瞧 ;)</li><li><a href="https://x.com/shengjia_zhao/status/1870176031610667223">Shengjia Zhao (@shengjia_zhao) 的推文</a>：很高兴能与 @ren_hongyu @_kevinlu 等人一起训练 o3-mini，这是一个极速的模型，具有惊人的推理/代码/数学性能。https://openai.com/12-days/?day=12</li><li><a href="https://x.com/deedydas/status/1870175212328608232">Deedy (@deedydas) 的推文</a>：OpenAI o3 在 Codeforces 上的评分为 2727，相当于地球上排名第 175 位的顶级人类编程竞赛选手。对于 AI 和整个技术领域来说，这绝对是一个超越人类水平的结果。</li><li><a href="https://x.com/polynoamial/status/1870172996650053653">Noam Brown (@polynoamial) 的推文</a>：我们 3 个月前才发布了 @OpenAI o1。今天，我们发布了 o3。我们有充分的理由相信这种增长轨迹将会持续下去。</li><li><a href="https://x.com/dmdohan/status/1870176374625054880">David Dohan (@dmdohan) 的推文</a>：在我看来，FrontierMath 上的提升甚至比 ARC-AGI 更令人印象深刻。从 2% 跃升至 25%。陶哲轩曾说这个数据集应该“至少能抵御 AI 几年”，并且“这些是 e...</li><li><a href="https://x.com/teortaxesTex/status/1869861452632469766">Teortaxes▶️ (@teortaxesTex) 的推文</a>：我简直不敢相信，OpenAI 可能真的陷入大麻烦了。Radford 一直是我观察那些没有深度意识形态投入（像 Ilya 那样）的顶级人才如何看待公司的风向标……</li><li><a href="https://x.com/Eric_Wallace_/status/1870176920706658692">Eric Wallace (@Eric_Wallace_) 的推文</a>：思维链 (Chain-of-thought) 推理为提高模型安全性提供了一条自然的途径。今天我们发布了一篇论文，介绍我们如何训练“o”系列模型来仔细思考不安全的提示词……</li><li><a href="https://bsky.app/profile/scott.hanselman.com/post/3ldpojtc3z22n">Scott Hanselman 🌮 (@scott.hanselman.com)</a>：在 ChatGPT 中尝试这个并感受震撼。“整齐地格式化这个。不要更改文本”。就这一个提示词。</li><li><a href="https://bsky.app/profile/scott.h">Bluesky</a>：未找到描述</li><li><a href="https://bsky.app/profile/scott.hanselman.com/post/3ldpouvj3qc2n">Scott Hanselman 🌮 (@scott.hanselman.com)</a>：未找到描述</li><li><a href="https://github.com/video-db/Director">GitHub - video-db/Director: 用于下一代视频交互和工作流的 AI 视频 Agent 框架。</a>：用于下一代视频交互和工作流的 AI 视频 Agent 框架。 - video-db/Director</li><li><a href="https://x.com/polynoamial/status/187017570022262">

8164">Noam Brown (@polynoamial) 的推文</a>：@OpenAI 你可以在这里报名协助 o3 和 o3-mini 的红队测试：https://openai.com/index/early-access-for-safety-testing/</li><li><a href="https://x.com/sama/status/1870176283851903152">Sam Altman (@sama) 的推文</a>：如果你是安全研究员，请考虑申请协助测试 o3-mini 和 o3。很高兴能很快将这些产品推向通用市场。为 OpenAI 的所有工作和创造力感到非常自豪...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1319771250641342488)** (20 条消息🔥): 

> `API Keys 使用情况, Character AI 受众洞察, 用户体验信号, 对角色扮演的兴趣, Swyx 的报告` 


- **API Keys 调试**：一位用户提到他们目前正在**调试 API keys**，这突显了开发者的一项常见任务。
   - 这反映了开发者社区中普遍存在的持续修补和探索。
- **Character AI 的多元受众**：讨论显示 **Character AI 受众** 主要由年轻人组成，而非商务专业人士。
   - 有人指出，**女性/女孩**的使用率与男性/男孩相当，这让一些成员感到惊讶。
- **对幻想连接的渴望**：参与者对有多少 Character AI 服务用户在寻找他们的**“迪士尼王子/公主 (x)”**表示出兴趣，强调了角色扮演的层面。
   - “神奇的数学石头”这个笑话概括了这些互动的奇思妙想性质，将幻想与技术融合在一起。
- **用户体验信号的探索**：有人询问在 Character AI 用户体验中应关注哪些**信号**，强调了理解用户互动的重要性。
   - 成员们对该主题的反馈以及 kbal11 分享的见解表现出极大的热情。
- **Swyx 对 Character AI 的见解**：提到了 Swyx 此前关于 **Character AI 实际受众** 的报告，表明可能存在更深入的分析。
   - 参与者表示有兴趣进一步探索该受众行为的更多维度。


  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1319426627070988368)** (38 条消息🔥): 

> `AI 在播客中的应用, Notebook LM 助力教育, 求职申请辅助, AI 生成视频项目, 提升音频制作` 


- **AI 变革播客制作**：一位成员分享了使用 AI 生成 [播客剧集](https://www.youtube.com/@AI_ForThePeople) 的兴奋之情，强调了其在快速创建引人入胜的音频内容方面的潜力。
   - 另一位成员评论了保持各章节间音量一致的重要性，强调了音频制作技术的持续改进。
- **Notebook LM 提升学术表现**：一位用户解释了他们如何使用 Notebook LM 为新闻课有效地构建时间线和思维导图，从而辅助撰写连贯的论文。
   - 这种方法被证明是有益的，因为他们整合了课程材料和针对学习中关键话题的特定播客。
- **利用 AI 准备求职申请**：一位成员详细介绍了他们如何利用 Notebook LM 根据职位公告分析简历，并生成面试问题作为学习指南。
   - 他们发现该工具的分析非常有见地，并鼓励其他人上传自己的简历以获取个性化反馈。
- **AI 驱动的创意项目**：分享了一个名为 'Churros in the Void' 的精彩项目，展示了完全通过 Notebook LM 和 LTX-studio 制作的 AI 生成视觉效果和配音。
   - 尽管在邀请知名配音演员方面面临挑战，创作者还是采取了 DIY 方法，体现了 AI 在叙事中的创新应用。
- **寻求更具吸引力的音频语调**：一位成员询问如何改变音频语调使其听起来更非正式且更具吸引力，并好奇是否使用了任何自定义设置。
   - 这引发了关于增强 AI 生成内容中音频表现力的技术和工具的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://open.spotify.com/episode/0KPu0RWclphLDYECl3L3Q7?si=3HxiyarGREewMwSD4NPGOQ">How your Brain communicates through Brain Waves - Human Connectome Project | MindLink | Nanonetes</a>: Connecting the Dots: The Human Connectome Project · 剧集</li><li><a href="https://notebooklm.google.com/notebook/8">未找到标题</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/notebook/8ef6c5d3-52e9-4aa6-a353-2099c9c616ec">未找到标题</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/notebook/46e1e005-6ea4-4c69-8b31-7720a2a8b209?_gl=1*n5asc6*_up*MQ..*_ga*NzMyNjQzMDY1LjE3MzQ2ODcwNDU.*_ga_W0LDH41ZCB*MTczNDY4NzA0NC4xLjEuMTczNDY4NzA0NC42MC4wLjA.&gclid=CjwKCAiAyJS7BhBiEiwAyS9uNb_gjOAeIIdFybpP3g9A1zVbN3G35p1bMzD5LstS1Qm78qXzzFClpxoCyuAQAvD_BwE&original_referer=https:%2F%2Fnotebooklm.google%23&pli=1">未找到标题</a>: 未找到描述</li><li><a href="https://youtu.be/PgFr0TI2WuQ"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1319395261889384540)** (106 条消息🔥🔥): 

> `NotebookLM Interactive Mode, Citation Feature Issues, Audio Overview Retrieval, Language Processing in NLM, Timeline Feature Usage` 


- **NotebookLM 交互模式推出的困惑**：许多用户报告在访问交互式语音模式时遇到问题，尽管官方称该功能已向所有用户开放。
   - 针对如何解决这一问题，用户提出了疑问，因为部分用户仍无法使用该功能。
- **笔记中引用功能的 Bug**：用户对更新后保存的笔记中引用功能消失表示沮丧。
   - 团队已确认该问题，并正在开发该功能的改进版本。
- **检索丢失的音频概览**：一位用户询问是否可以找回之前生成但从笔记本中消失的音频概览。
   - 讨论指出，用户担心无法重新生成与之前内容同样深刻的见解。
- **语言处理和来源限制**：用户对 NotebookLM 处理多语言源文件的方式以及对文本检索质量的影响表示担忧。
   - 用户建议将特定语言的文档分开，以提高上传源文件结果的准确性。
- **时间轴功能的利用**：时间轴功能被强调为一种以结构化方式组织历史内容的宝贵工具。
   - 用户赞赏其提供事件全局视图的能力，增强了研究的整体体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/xzibit-pimp-my-ride-lol-gif-23167832">Xzibit Pimp GIF - Xzibit Pimp My - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=YS4rdvcfqEU"> - YouTube</a>: 未找到描述</li><li><a href="https://illuminate.google.com">Illuminate | Learn Your Way</a>: 使用 Illuminate 将研究论文转换为 AI 生成的音频摘要，这是你更快理解复杂内容的 Gen AI 工具。</li><li><a href="https://youtu.be/zYv4L72SZGU?si=qNENW3qRX554dLy5"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/MI4AgblZf5M?si=-NvBUhHpJN5m3MwJ">Afterlife Explained: The Mind-Bending Theory of the Universe | Afterlife | Podcast</a>: 欢迎深入探讨由 Christopher Langan 提出的开创性理论 CTMU (Cognitive-Theoretic Model of the Universe)。通常被称为...</li><li><a href="https://youtu.be/0pM9IXIbGJE?si=FnK7f_21FonZwHIX"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/JhuC77mtdoQ?si=mwkCC7OgR-LtKWuw&t=289"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/fS9w9Cir6dw?si=83Tue3ndcuxws8rK"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/zwGBauoVVtA?si=G-wfDz7GPP81Cwu"> - YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=czvAd98coiU"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/QxbmQs3b_DE?si=Gah7aYyzCsMzxMi4&t=672"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1319401624430448700)** (102 条消息🔥🔥): 

> `《超人》电影预告片, 使用 .edu 邮箱获取 Perplexity Pro, OpenAI 的新 GPT 模型, Lepton AI 项目相似性, Perplexity API 支持问题` 


- **DC 发布新《超人》电影预告**：一名成员注意到 DC 发布了新《超人》电影的预告片，这对他们来说似乎相当突然。
   - 围绕电影的兴奋虽然短暂但很热烈，一些成员分享了轻松的反应。
- **通过 .edu 邮箱获取 Perplexity Pro 访问权限**：一些用户讨论了一个传闻中的促销活动，该活动为拥有 .edu 邮箱的学生提供免费的 Perplexity Pro 访问权限，这是由一位朋友的说法引发的。
   - 然而，似乎并非所有获取该促销的尝试都成功了，这导致了一些困惑。
- **OpenAI 推出 o3 和 o3-mini 模型**：成员们推测了 OpenAI 新模型 o2 和 Orion 的发布，它们被视为最近推出的 o1 的潜在继任者。
   - 兴奋之情溢于言表，有人声称 o3 可能接近 AGI，并讨论了其对 AI 应用的影响。
- **Lepton AI 项目引发讨论**：一名成员指出，新推出的 Node pay 产品与之前看到的 Lepton AI 开源项目非常相似。
   - 这引发了关于设计原创性及其与该领域现有产品相似性的评论。
- **关于 Perplexity API 支持的咨询**：一位用户表达了对 Perplexity API 中 system prompt 性能的担忧并寻求帮助。
   - 另一位用户澄清说，虽然 prompt 可以引导语气和风格，但它不会影响模型的搜索组件。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai/guides/model-cards">未找到标题</a>：未找到描述</li><li><a href="https://search.lepton.run/">Lepton Search</a>：使用 Lepton AI，用不到 500 行代码构建你自己的对话式搜索引擎。</li><li><a href="https://tenor.com/view/conspiracy-charlie-day-crazy-always-sunny-in-philadelphia-qanon-gif-23738584">Conspiracy Charlie Day GIF - Conspiracy Charlie Day Crazy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.perplexity.ai/backtoschool">Perplexity - Race to Infinity</a>：欢迎回到学校！在短短两周内，即可兑换一个月免费的 Perplexity Pro。推荐你的朋友，如果你的学校达到 500 人注册，我们将把免费月份升级为一整年...</li><li><a href="https://github.com/leptonai/search_with_lepton">GitHub - leptonai/search_with_lepton: Building a quick conversation-based search demo with Lepton AI.</a>：使用 Lepton AI 构建快速的基于对话的搜索演示。- leptonai/search_with_lepton</li><li><a href="https://techcrunch.com/2024/12/20/openai-announces-new-o3-model/">OpenAI announces new o3 models | TechCrunch</a>：OpenAI 在其为期 12 天的 "shipmas" 活动的最后一天保留了最大的公告。周五，该公司发布了 o3，即 o1 的继任者。</li><li><a href="https://www.copilotforyoutube.com/search/openai-o3-and-o3-mini12-days-of-openai-day-12-T7sbiQRKxbMdlrWTddGC9L">OpenAI o3 and o3-mini—12 Days of OpenAI: Day 12</a>：Sam Altman、Mark Chen、Hongyu Ren 以及特别嘉宾 ARC Prize Foundation 主席 Greg Kamradt 介绍并讨论了 OpenAI o3、o3-mini，并呼吁进行安全测试和新的对齐...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1319460652049367042)** (5 messages): 

> `Rio Da Yung OG 获释，三星的 Project Moohan，苹果的刚果冲突矿产，俄勒冈州的西洛西宾计划，工作中的 AI 使用` 


- **Rio Da Yung OG 的获释引发关注**：Rio Da Yung OG 已从[监狱](https://www.perplexity.ai/page/rio-da-yung-og-released-from-p-JFyapxOIRnOYy5RX.Kw3zQ)获释，引发了关于他未来计划和音乐生涯的讨论。
   - *粉丝们渴望看到这将如何影响他即将开展的项目。*
- **三星揭晓 Project Moohan**：[三星的 Project Moohan](https://www.perplexity.ai/page/samsung-s-project-moohan-Od28QDNtTTiZjpKBmuhOfg) 代表了一项旨在创新 AI 技术的新倡议。
   - *关于该项目的规模和潜在应用的细节仍在不断披露中。*
- **苹果备受争议的冲突矿产**：围绕苹果从刚果采购**冲突矿产**的讨论浮出水面，这与[此处探讨](https://www.youtube.com/embed/eK9Ajrd0e5U)的伦理采购实践有关。
   - *这些做法对苹果供应链的影响受到了关注，社区见解敦促提高透明度。*
- **俄勒冈州的西洛西宾（Psilocybin）计划受到关注**：随着**俄勒冈州西洛西宾计划**在促进迷幻剂治疗用途方面的进展，该计划的实施引起了人们的兴趣。
   - *社区成员正在监测该计划的潜在扩张和成功案例。*
- **AI 对职场的影响**：据最近的一项[调查](https://www.perplexity.ai/page/more-than-70-use-ai-at-work-ym5.V8EjTHmJhCCVrvZuGQ)显示，超过 **70% 的员工**在工作中使用 AI。
   - *这一转变反映了 AI 在提高生产力和维护核心指令方面已变得多么不可或缺。*



**Link mentioned**: <a href="https://www.youtube.com/embed/eK9Ajrd0e5U">YouTube</a>: 未找到描述

  

---


### **Nomic.ai (GPT4All) ▷ #[announcements](https://discord.com/channels/1076964370942267462/1090471714888102009/1319449812495765595)** (3 messages): 

> `GPT4All v3.6.0 发布，GPT4All v3.6.1 发布，Reasoner v1，聊天模板修复` 


- **GPT4All v3.6.0 发布了！**：新的 **GPT4All v3.6.0** 包含了 **Reasoner v1**，这是一个用于复杂推理任务的内置 JavaScript 代码解释器工具，同时还改进了模板兼容性。
   - 其他修复解决了消息中的 XML 使用问题，以及影响 v3.5.0 之后系统消息检测的 **Jinja2Cpp bug**。
- **v3.6.1 中的快速修复**：**GPT4All v3.6.1** 已经发布，旨在解决关键问题，包括修复了 v3.6.0 中无法使用的“停止生成”和“复制整个对话”按钮。
   - 此次更新反映了社区的迅速贡献，特别是来自 Nomic AI 的 **Adam Treat** 和 **Jared Van Bortel**。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1319395590768955516)** (90 条消息🔥🔥): 

> `Llama 3.3 和 Qwen2 模型，GPT4ALL 自定义模板与推理，本地 API 服务器集成，Phi-4 模型对比，v3.6.0 中停止生成按钮问题` 


- **讨论了 Llama 3.3 和 Qwen2 模型**：成员们分享了使用 **Llama 3.3** 和 **Qwen2** 模型的见解，指出了它们的功能以及相比早期版本的改进。
   - 人们对未来能进一步提升性能的版本充满期待。
- **实现了用于推理的自定义模板**：为 **GPT4ALL** 设计的自定义聊天模板通过代码解释器促进推理，允许用户有效地执行代码。
   - 成员们确认了其与多种模型的兼容性，增强了它们的功能。
- **本地 API 服务器可以使用 LocalDocs**：**GPT4ALL** 本地 API 服务器允许与 **LocalDocs** 集成，使用户能够运行 API 请求并有效地利用本地模型。
   - 参与者讨论了在他们的应用程序中连接和利用该服务器的过程。
- **Phi-4 模型与其他模型的性能对比**：讨论围绕 **Phi-4** 模型展开，据报道这款 **14B** 模型的性能可与 **Llama 3.3 70B** 媲美。
   - 成员们分享了在本地运行 **Phi-4** 的经验，并对其能力表示兴奋。
- **确认停止生成按钮问题**：用户确认了 **3.6.0** 版本中**停止生成**按钮的问题，导致了关于回归缺陷（regression bug）的报告。
   - 修复工作正在进行中，正如链接的 GitHub issue 跟踪问题中所概述的那样。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.gpt4all.io/gpt4all_api_server/home.html">GPT4All API Server - GPT4All</a>：GPT4All 文档 - 在您的硬件上高效运行 LLM</li><li><a href="https://huggingface.co/matteogeniaccio/phi-4/tree/main">matteogeniaccio/phi-4 at main</a>：未找到描述</li><li><a href="https://x.com/OfficialLoganK/status/1869789822384255300">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：这仍然是一个早期版本，但看看该模型如何处理一个涉及视觉和文本线索的挑战性谜题：(2/3)</li><li><a href="https://tenor.com/view/curses-foiled-again-he-man-meh-skeleto-gif-16546096">Curses Foiled Again GIF - Curses Foiled Again He Man - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-chat/system_requirements.md">gpt4all/gpt4all-chat/system_requirements.md at main · nomic-ai/gpt4all</a>：GPT4All：在任何设备上运行本地 LLM。开源并可用于商业用途。 - nomic-ai/gpt4all</li><li><a href="https://groq.com/">Groq 是快速 AI 推理</a>：Groq 的 LPU™ 推理引擎是一个硬件和软件平台，可提供卓越的计算速度、质量和能源效率。Groq 为 AI 提供大规模的云端和本地解决方案...</li><li><a href="https://github.com/nomic-ai/gpt4all/issues/3333">停止生成按钮在 v3.6.0 中不再起作用 · Issue #3333 · nomic-ai/gpt4all</a>：看来我们在 v3.6.0 的停止生成按钮上遇到了回归。ChatModel::get(...) 函数被删除了，该函数在 QML 的几个不同地方被使用。这没有被注意到是因为我们...</li><li><a href="https://groq.com/)">Groq 是快速 AI 推理</a>：Groq 的 LPU™ 推理引擎是一个硬件和软件平台，可提供卓越的计算速度、质量和能源效率。Groq 为 AI 提供大规模的云端和本地解决方案...
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1319434015366709278)** (81 messages🔥🔥): 

> `最佳本地 AI 图像生成器，在 AI 中创建风格模型，Discord 中的技术支持与诈骗，游戏开发者的资产生成工具，基于现有图像训练模型` 


- **当前最佳本地 AI 图像生成器**：一位成员询问了目前最好的本地 AI 图像生成器，并提到他们之前使用的是 **SD1.5**。
   - 另一位成员建议使用 **SDXL 1.0** 搭配 **comfyUI** 以获得更好的效果。
- **复制图像风格指南**：一位用户分享了他们成功在本地运行 **flux** 的经历，并寻求关于如何复制参考图像风格的指南。
   - 他们正尝试为游戏场景生成具有一致风格的图像。
- **Discord 诈骗警报**：针对一个可疑的技术支持服务器展开了讨论，在有人索要钱包详情后，该服务器被判定为诈骗。
   - 成员们分享了他们的经历以及对这类诈骗安全性的担忧。
- **游戏资产生成工具**：一位用户询问了成熟的用于生成等距角色等游戏资产的 **Stable Diffusion** 工具。
   - 其他人建议使用免费资产，并提到了 **SF3D**，这是一个用于从图像生成 3D 资产的模型。
- **利用现有图像生成独特艺术**：一位艺术家解释了他们的目标，即使用自己的图像训练模型，以便更快地生成艺术作品。
   - 建议他们训练一个 **LoRA** 模型，特别是基于 **Flux** 或 **SD 3.5**。



**提到的链接**：<a href="https://huggingface.co/stabilityai/stable-fast-3d">stabilityai/stable-fast-3d · Hugging Face</a>：未找到描述

  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1319451747344842895)** (58 条消息🔥🔥): 

> `Cohere 的 c4ai 模型、MLX 集成、VLLM 支持、最新模型性能评估、即将发布的新版本` 


- **对 MLX 和新模型的兴奋**：社区成员对 [Cohere 的 c4ai-command-r7b 模型](https://huggingface.co/mlx-community/c4ai-command-r7b-12-2024-4bit) 的新 MLX 支持表示热烈欢迎，并分享了安装技巧。
   - 一位成员指出，尽早集成像 **VLLM** 这样的模型将有助于简化开源社区内的贡献流程。
- **Cohere 能力展示**：一份社区评论强调了 **Cohere 模型** 在处理 211,009 token 的 **Danganronpa（弹丸论破）同人小说** 时表现出色，展示了仅使用 **11.5 GB** 显存的惊人内存效率。
   - 这引发了对其架构的讨论，特别是其 **128K 上下文长度** 和缺乏位置编码（positional encoding）的设计，这可能增强了模型的泛化能力。
- **与 Cohere 合作更新**：成员们讨论了如何让 **Cohere** 更直接地参与支持早期的新版本发布，并提到了与 **Mistral** 类似合作的成功经验。
   - 贡献者认为，这可以为 **VLLM** 等模型和更新带来更顺畅的集成过程。
- **注意到 GPT-J 的增强**：有人推测 **GPT-J 的 RoPE 机制** 对注意力准确性的影响，认为它可能比之前的配置更有效。
   - 成员们回顾了过去 **4096 滑动窗口（sliding windows）** 的实现，重申了他们对新架构带来的进步的信心。
- **更新与发布期待**：成员们注意到了即将发布的版本，特别是关于 **O3** 模型预期能力的讨论，暗示其具有类似于 **GPT-4** 的创新功能。
   - 这些讨论凸显了社区对潜在功能的兴奋，包括类似于节日应用中所使用的模型语音交互功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/nickfrosst/status/1868131852973985947">Nick Frosst (@nickfrosst) 的推文</a>：懂行的人都知道。引用 N8 Programs (@N8Programs)：好吧，对于那些怀疑 Cohere 模型只是在重复《哈利·波特》情节的人，这里是它在处理一部 211,009 token 的 Danganronpa 同人小说时表现得相当不错（情节几乎正确）...</li><li><a href="https://huggingface.co/mlx-community/c4ai-command-r7b-12-2024-4bit">mlx-community/c4ai-command-r7b-12-2024-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/N8Programs/status/1868084925775380830">N8 Programs (@N8Programs) 的推文</a>：好吧，对于那些怀疑 Cohere 模型只是在重复《哈利·波特》情节的人，这里是它在处理一部 211,009 token 的 Danganronpa 同人小说时表现得相当不错（情节几乎正确），仅使用了 11.5 ...</li><li><a href="https://github.com/ml-explore/mlx-examples.git#subdirectory=llms`">GitHub - ml-explore/mlx-examples: MLX 框架中的示例</a>：MLX 框架中的示例。通过在 GitHub 上创建账号为 ml-explore/mlx-examples 的开发做出贡献。</li><li><a href="https://github.com/ml-explore/mlx-examples/pull/11">llama: 转换权重时遇到不支持的 ScalarType BFloat16 · Issue #11 · ml-explore/mlx-examples</a>：尝试转换 PyTorch 权重时，例如：python convert.py ../../llama-2-7b/consolidated.00.pth mlx_llama-2-7b.npz，我得到：File "../ml-explore/mlx-examples/llama/convert.py", ...</li><li><a href="https://github.com/ml-explore/mlx-examples/pull/1157">由 Blaizzy 添加对 cohere2 的支持 · Pull Request #1157 · ml-explore/mlx-examples</a>：添加了对带有滑动注意力的 Cohere2 的支持。非常感谢 @N8python 提供的灵感！Bf16 4bit
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1319712351985078314)** (4 条消息): 

> `信用卡被拒、3D Secure 问题、VPN 使用、支持联系方式` 


- **尽管显示成功消息，信用卡仍被拒**：一名用户报告称，尽管在完成 **3D Secure** 流程后收到了银行的成功消息，但他们的德国信用卡经常被 Cohere 拒绝。
   - 他们对反复出现的拒绝表示沮丧，并寻求联系支持团队的建议。
- **对神秘的支付处理过程提出疑问**：另一名成员建议检查用户是否使用了 **VPN**，这可能是导致支付问题的原因之一。
   - 该用户正在调查导致信用卡持续被拒的可能原因。
- **寻求支持**：一名成员建议用户通过 [support@cohere.com](mailto:support@cohere.com) 联系支持团队，以解决信用卡问题。
   - 该建议旨在让用户获得 Cohere 支持团队关于支付问题的帮助。


  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1319468500485541938)** (16 messages🔥): 

> `印度支付方式问题，升级 API Keys 以获得更高限制，Trial Keys 的上下文错误` 


- **印度支付方式问题限制用户**：一位用户报告在为 Cohere 添加支付方式时卡片被拒绝，揭示了印度银行（如 ICICI 和 HDFC）常见的拦截此类交易的问题。
   - 支持团队建议使用不同的卡片或联系银行以启用对 Cohere Inc. 的国际支付。
- **Trial Key 限制导致错误**：一名成员在进行文档重排序（reranking）时遇到了 'TooManyRequestsError'，确定这是由于 Trial key 的限制（每月上限 1000 次 API 调用）造成的。
   - 另一位用户建议创建付费 API key 以移除这些限制，该用户在升级后成功解决了问题。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1319574423522775041)** (1 messages): 

> `Findr 中的 Cohere 技术，Findr 发布热潮` 


- **对 Findr 发布的兴奋**：成员们对 **Findr 的发布** 表示兴奋，用“wohooo”和“祝贺发布！”等词语庆祝其显而易见的成功。
   - 这种热情反映了社区对利用 **Cohere technology** 的新项目的强力支持。
- **咨询 Findr 中使用的 Cohere 技术**：一位成员询问了 Findr 具体使用的 **Cohere technology**，表明希望了解该应用背后的技术栈（tech stack）。
   - 这种兴趣点出了社区渴望了解这些技术如何助力成功发布。


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1319754440709767310)** (10 messages🔥): 

> `DCT 编码探索，VAEs 与人类感知，色彩空间与细节感知` 


- **实验 DCT 编码**：一位成员开始探索 **DCT** 和 **DWT** 编码，质疑使用 **YCrCb** 或 **YUV** 色彩空间作为输入的效率。
   - 他们指出，虽然 **VAEs** 易于训练，但在这种编码追求中可能并不值得投入精力。
- **受 VAR 论文启发的 DCT 组件**：讨论围绕一位成员的想法展开，即将 **VAR paper** 与预测连续 **DCT blocks** 的 **DC component** 联系起来，随后进行上采样并加入 **AC components**。
   - 这提出了一种通过逐步添加组件来增强图像质量的结构化方法。
- **感知与色彩空间效用**：一位成员强调了使用具有独立亮度通道的色彩空间的重要性，因为人类对**高频灰度细节**的感知优于**高频彩色细节**。
   - 大家一致认为 RGB 可能无法有效地映射到人类对颜色的感知，建议探索 **JPEG** 和 **AV1** 技术。
- **损失函数中的人类感知**：有人指出 **VAEs** 可能固有的利用了色彩编码的一些概念，特别是如果损失函数（loss functions）与**人类感知**对齐的话。
   - 这突出了未来在优化与视觉理解相关的编码方面的实验方向。


  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1319736523121365012)** (57 条消息🔥🔥): 

> `OpenAI o3 发布，AGI 讨论，Elo 评分与性能对比，推理时计算（Test time compute）的影响，未来 AI 预测` 


- **OpenAI 发布下一代 o3 模型**：OpenAI 宣布了其下一代推理模型 **o3**，在低计算模式下的半公开评估中达到了 **75.7%**，在高计算模式下达到了 **87.5%**，这标志着 AI 能力的重大飞跃。
   - 该模型展示了新颖的任务适应能力，这可能会重新定义当前对 AI 潜力的理解，并挑战现有的 Benchmark 性能。
- **关于 AGI 状态的辩论**：社区对于这些进展是否让我们更接近 **AGI** 存在分歧，一些成员断言，在 ARC 等任务上超越人类表现意味着 **AGI** 已经实现。
   - 其他人则警告说 **AGI** 一词含糊不清，建议应根据具体语境进行定义以避免误解。
- **Elo 评分与性能指标**：围绕 **Elo 评分**系统的讨论出现，将模型的表现与国际象棋评分联系起来，暗示根据得分，o3 已达到特级大师（Grandmaster）水平。
   - 探讨了不同评分量表及其指数性质的影响，表明更高的分数可能会显著偏离性能预期。
- **增加推理时计算（Test Time Compute）的潜力**：有人推测，考虑到增加任务时长每项任务 **20 美元**的成本，较弱的模型是否可以通过更多计算来复制 o3 的表现。
   - 有人提出，将较大的任务划分为较小的片段可以在不改变模型本身的情况下最大限度地利用计算资源。
- **对未来 AI 发展的预测**：模型的快速进步引发了人们对未来能力的兴奋，特别是在成本效率和在 **SWE-bench** 等 Benchmark 上增加测试方面。
   - 有人对这些发展如何影响 Text-to-Image 生成以及更广泛的 AI 应用领域表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/fchollet/status/1870169764762710376">François Chollet (@fchollet) 的推文</a>：今天 OpenAI 发布了 o3，其下一代推理模型。我们与 OpenAI 合作在 ARC-AGI 上对其进行了测试，我们相信它在让 AI 适应新颖任务方面代表了重大突破...</li><li><a href="https://arcprize.org/blog/oai-o3-pub-breakthrough">OpenAI o3 在 ARC-AGI-Pub 上取得突破性高分</a>：OpenAI o3 在 ARC-AGI 公开排行榜上得分 75.7%。</li><li><a href="https://wismuth.com/elo/calculator.html#rating1=2727&rating2=1258">Elo 胜率计算器</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1319496637940502578)** (11 条消息🔥): 

> `GPU 推荐，芯片设计资源，硬件描述语言` 


- **GPU 推荐被重定向**：一位成员指出，关于 **GPU 推荐** 的问题在 [r/pcmasterrace](https://www.reddit.com/r/pcmasterrace/) 等社区已有广泛讨论，滥发此类咨询可能不受欢迎。
   - 另一位成员怀疑重复询问 GPU 的行为背后存在钓鱼（trolling）嫌疑。
- **芯片设计的深度资源**：一位成员寻求关于芯片设计和硬件描述语言的**深度书籍或资源**。
   - 推荐包括搜索来自 **UCB** 和 **UMich** 的大学课程材料，这些材料通常提供公开访问的幻灯片和作业。
- **Sedra 的微电子书籍占据统治地位**：另一位用户称赞 **Sedra 的书** 是大多数 ECE 项目的金标准，特别是参考了 *Microelectronic Circuits*。
   - 这本书因其深度和清晰度，在**电子与计算机工程（Electrical and Computer Engineering）**课程中广受认可。
- **推荐 Zero To ASIC 课程**：一位成员提到了对 YouTube 上 **[Zero To ASIC 课程](https://www.youtube.com/@ZeroToASICcourse)** 的正面反馈，认为它是一个宝贵的资源。
   - 一位用户对该课程表示了兴趣，称这看起来是一次令人兴奋的体验。



**提到的链接**：<a href="https://www.reddit.com/r/pcmasterrace/">Reddit - 探索一切</a>：未找到描述

  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1319555587662417931)** (2 条消息): 

> `Triton Documentation Issues, Debugging Kernel Shared Memory, Proton Memory Instrumentation, Triton Language Types` 


- **Triton 的搜索功能损坏**：一位用户报告称 [Triton 文档主页](https://triton-lang.org/main/index.html#) 上的搜索功能无法正常工作。
   - 他们还指出缺乏关于 **tl.dtypes** 的文档，并提到在识别如 **tl.int1** 等类型时存在困难。
- **关于 Triton 文档后端的咨询**：一位用户询问 Triton 文档的后端内容是否开放公众贡献。
   - 他们表示如果可能的话，愿意帮助更新文档。
- **调试 Kernel 中的共享内存使用**：一位用户询问关于 **triton_gpu.local_alloc** 与 **kernel.metadata.shared** 在共享内存使用量上存在差异的经验。
   - 他们尝试使用 `proton --instrument=print-mem-spaces script.py` 进行调试，但发现它仅支持 AMD 硬件。



**提及的链接**：<a href="https://triton-lang.org/main/index.html#">Welcome to Triton’s documentation! &mdash; Triton  documentation</a>：未找到描述

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1319414400616104077)** (9 条消息🔥): 

> `TensorRT Namespace Issue, Race Condition in Memory Copy, Memory Fencing after Kernel Execution, Understanding cute::composite` 


- **TensorRT 命名空间引起混淆**：一位用户澄清说，关于 `trt` 的问题是因为它是一个 **namespace**，由代码中错误的参数引起。函数 **AsyncMemCpyD2D** 被错误识别，因为 stream 类型不是 **cudaStream_t**。
   - *谢谢你的建议。我找到了原因。*
- **内存操作中潜在的竞态条件**：一位用户推测可能存在 **race condition**（竞态条件），认为这可能是 Graph 中记录内存方式的问题。这指向了一个需要调试的复杂交互。
   - 另一位用户对 **TensorRT** 上下文中 **AsyncMemCpyD2D** 的功能表示不确定。
- **隐式内存屏障解释**：一位成员解释说，虽然理论上可以等待内存，但通常是不必要的，除非稍后会重新加载该内存。内存将在 Kernel 执行后被 **implicitly fenced**（隐式屏障），从而确保数据完整性。
   - *你说得对！谢谢！*
- **关于 cute::composite 函数的困惑**：一位用户询问如何有效地将 global layout 与 **smemLayoutX** 进行 **composite** 以实现特定的 grid 分区。他们对 **cute::composite** 函数表示困惑，并强调了它的重要性。
   - *实际上，我对 cute::composite 感到很困惑，但那是一个非常重要的 tensor 函数……*


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1319749327144357982)** (3 条消息): 

> `Flex Attention, Context Parallel Implementation, Attn-Gym Examples` 


- **探索 Flex Attention 和 Context Parallel 计划**：一位成员询问是否有使用 **context parallel** 处理来实现 **flex attention** 的计划，并寻求对现有示例的澄清。
   - 另一位成员肯定地表示，现在实现这一点是非常可能的，并表示打算在 **attn-gym** 中添加一个示例。
- **向 Attn-Gym 添加示例的可能性**：讨论强调了在 **attn-gym** 中添加一个使用 **flex attention** 的 **context parallel** 实际示例的可能性。
   - 这一举措标志着一种积极主动的方法，旨在增强社区可用的资源。


  

---

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1319799852976050196)** (1 条消息): 

> `Diffusion Models Conditioning, NeurIPS 2024 Papers` 


- **探索 Diffusion Models 的条件化 (Conditioning)**：一位成员分享了关于 **diffusion models** 如何进行条件化的见解，并提供了一篇由 [Tero Karras 撰写的 NeurIPS 2024 论文](https://x.com/TheVariational/status/1870196816844603717)链接，详细介绍了该主题。
   - 该演示文稿对 **Autoguidance** 方法进行了全面回顾，该方法是 NeurIPS 2024 最佳论文的亚军。
- **获取 Autoguidance 论文 PDF**：另一位成员指出了 [Google Drive 链接](https://drive.google.com/file/d/1WxQ7Zd15Ly4tFt2YFghJm-cmXlTgcEYI/view?usp=sharing)，其中包含 Autoguidance 回顾的 PDF，强调了其在 diffusion models 讨论中的重要性。
   - 该论文专注于理解 diffusion models 的**影响力 (influential)** 方面，这激发了社区的好奇心。



**提到的链接**：<a href="https://x.com/TheVariational/status/1870196816844603717">来自 The Variational Book (@TheVariational) 的推文</a>：对 diffusion models 如何受到影响感到好奇吗？@jaakkolehtinen @unixpickle @prafdhar @TimSalimans @hojonathanho 请查看关于 Autoguidance #NeurIPS2024 最佳论文亚军的回顾...

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1319735956298924092)** (3 条消息): 

> `Multi Node Inference, Distributed Topics, Channel Management` 


- **多节点推理频道咨询**：@karatsubabutslower 询问了讨论 **multi node inference**（多节点推理）的合适频道。
   - *他们希望确保在正确的地方分享关于此主题的见解。*
- **分布式话题的通用频道**：@marksaroufim 建议先在 **general 频道**开始讨论，并指出如果分布式话题变得流行，将会创建一个新频道。
   - *这种方法可以根据社区对 **distributed inference**（分布式推理）话题的兴趣灵活调整。*


  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/1319793870426607696)** (1 条消息): 

> `Sparse API Usage, PyTorch Quantization, Sparsity Design Overview` 


- **更换稀疏 API 以提高灵活性**：一位成员注意到 [PyTorch 稀疏文档](https://github.com/pytorch/ao/tree/main/torchao/sparsity#design)中的示例在推理时使用了 `to_sparse_semi_structured` API，并建议可以将其更改为 `sparsify_` 以获得更广泛的应用。
   - 他们强调这是一个潜在的改进，并艾特了另一位成员，待其 PTO（带薪休假）结束归来后确认。
- **重点介绍 PyTorch 的稀疏特性**：分享的链接指向了 PyTorch 仓库，该仓库具有用于训练和推理的原生量化和稀疏化功能，展示了该项目的范围。
   - 它包含一张反映项目品牌的缩略图以及关于其功能的简短描述。



**提到的链接**：<a href="https://github.com/pytorch/ao/tree/main/torchao/sparsity#design,">pytorch/ao 仓库的 main 分支下的 ao/torchao/sparsity</a>：用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao

  

---


### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1319691750410813541)** (6 条消息): 

> `ARC CoT dataset, LLaMA 8B fine-tuning, OpenAI evaluation results, o3-high evaluation costs` 


- **ARC CoT 数据集生成正在进行中**：一位用户正在生成一个 **ARC CoT 数据集**，目标是达到 **10k 个样本**，以便使用对数概率（log probability）指标比较微调后的 **LLaMA 8B** 与基础模型之间的性能。
   - 他们计划分析描述相对于 **ground truth** 的优势，并在未来的评估中探索“CoT”训练的影响。
- **未来的 LLaMA 8B 微调计划**：一旦生成了几千个样本，将尝试使用直接转导（direct transduction）和棋盘分析（board-analysis）方法对 **LLaMA 8B** 进行微调。
   - 目标是确定 **“CoT”训练** 是否有切实的好处。
- **赞扬 OpenAI 的评估分数**：一位用户对 **OpenAI** 在近期基准测试中获得的高评估分数表示祝贺。
   - 他们强调了在 **OpenAI 实验室**之外复制这些结果以确保更广泛适用性的重要性。
- **o3-high 评估的高昂成本**：据指出，**o3-high** 的半公开评估在计算资源上耗资超过 **$10k**。
   - 然而，确切的数字并未披露，这凸显了此类评估的高昂代价。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1319433134521192583)** (4 条消息): 

> `LlamaParse 音频解析能力、LlamaIndex 年度回顾、股票分析机器人创建、文档处理自动化` 


- **LlamaParse 扩展音频解析功能**：LlamaParse 现在可以解析音频文件，在其已有的对 PDF 和 Word 等复杂文档格式的强大支持基础上增加了这一能力。用户可以无缝上传音频文件并将语音转换为文本。
   - *这一增强功能使 LlamaParse 成为适用于各种文档类型的全球最佳解析器*。
- **LlamaIndex 庆祝辉煌的一年**：LlamaIndex 分享了年度回顾，重点介绍了数千万页的解析量和显著的社区增长。按月划分的功能发布细目显示，他们在全年内保持了每周一次以上的交付频率。
   - 期待 **LlamaCloud** 在 **2024** 年初正式发布（GA），并持续关注其开源贡献。
- **轻松创建股票分析机器人**：了解如何使用 LlamaIndex 的 **FunctionCallingAgent** 结合 **Claude 3.5 Sonnet** 构建自动化股票分析 Agent。这一一键式解决方案为用户简化了股票分析流程。
   - 在 Hanane D 关于这一创新工具的 [LinkedIn 帖子](https://t.co/GOjUTl0Es0)中获取详细说明。
- **使用 LlamaIndex 自动化文档工作流**：一个新的 Notebook 演示了如何使用 LlamaIndex 自动化文档处理工作流，重点是标准化不同供应商的单位和测量。这是一个展示 LlamaIndex 在现实场景中能力的实用示例。
   - 查看共享的 [notebook](https://t.co/aOTuSwM341) 中的完整示例以探索其用途。



**提及的链接**：<a href="https://t.co/bxx5t1sVgy">The Year in LlamaIndex: 2024 — LlamaIndex - 基于企业数据构建知识助手</a>：LlamaIndex 是一个简单、灵活的框架，用于使用连接到企业数据的 LLM 构建知识助手。

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1319395636147257454)** (17 条消息🔥): 

> `Azure OpenAI 嵌入模型、大型项目的 GraphDB、使用情感分析微调 LLM、创建合成数据集、TextNode 属性问题` 


- **Azure OpenAI 的速率限制问题**：一位成员报告在使用 Azure OpenAI 嵌入模型时遇到 **速率限制错误 (rate limit errors)**，寻求解决该问题的建议。
   - 另一位成员建议通过 **增加最大重试次数** 或使用代码片段减慢 **摄取过程 (ingestion process)** 来解决。
- **解决 TextNode 属性错误**：讨论揭示了在尝试将节点插入索引时出现的 **AttributeError** ('TextNode' object has no attribute 'get_doc_id')。
   - 成员们澄清了节点的正确方法是 `index.insert_nodes(...)`，并建议一次插入一个节点以避免错误。
- **关于 GraphDB 选项的咨询**：一位成员询问其他人对于大型项目使用哪些 **GraphDB**，并表示对现有选项不满意。
   - 整体情绪表达了对 **GraphDB 现状** 的担忧，并希望能有更好的替代方案。
- **情感分析微调 LLM 的步骤**：一位成员希望为情感分析 **微调 LLM**，但不确定如何创建合成数据集。
   - 另一位成员建议探索 Prompt 操作，并 **提供了一个链接**，指向一篇讨论使用 LLM 生成合成数据的博客。
- **了解消息查询中的现有问题**：有几项关于 **系统停机** 的咨询，对目前的服务状态感到困惑。
   - 成员们询问并澄清了哪些服务已下线，其中一位成员寻求社区对当前问题的反馈。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/synthetic-data-save-costs">Synthetic data: save money, time and carbon with open source</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/issues/7879">[Question]:  Consistently getting rate limit error when building index · Issue #7879 · run-llama/llama_index</a>：问题验证：我已在文档和 Discord 中搜索过答案。问题：我正在使用基础代码对一个约 10 行的单个文本文档进行索引，代码为 from llama_index import ...
</li>
</ul>

</div>
  

---

### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1319710643019845784)** (1 条消息): 

> `Hackathon 提交重新开放, 技术困难, 提交截止日期, 手动提交检查` 


- **Hackathon 提交表单重新开放！**：由于部分参赛者面临**技术困难**，我们已重新开放 [hackathon 提交表单](https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform) 以供提交，该表单将于**今晚 11:59PM PST (12月20日)** 再次关闭。
   - 请确保更新任何错误的链接，或者如果您错过了昨天的截止日期，请补交 —— **没有惩罚**！
- **提交截止日期提醒**：Hackathon 提交表单将于今晚再次关闭，提醒参赛者在最终截止日期前仔细检查其提交内容。
   - 鼓励参赛者尽早提交以避免最后一刻出现问题，因为提交后不会收到自动电子邮件确认。
- **可进行手动提交检查**：如果参赛者希望手动检查其提交是否成功，可以在 <#1280237064624799886> 中发帖。
   - 建议尽早进行手动验证以减轻压力！



**提到的链接**: <a href="https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform">未找到标题</a>: 未找到描述

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1319578857535311892)** (11 条消息🔥): 

> `Hackathon 延期请求, Hackathon 参与表单, 提交注册确认, YouTube 视频格式问题, Agent 框架建议` 


- **Hackathon 不再延期**：一位成员询问了 Hackathon 再次延期的可能性，但 Tara 告知“很遗憾，没有”更多延期。
   - 这个轻松的请求反映了大家对宽限的共同愿望，但限制措施已经生效。
- **Hackathon 提交的主要联系人**：提醒参赛者在 Hackathon 参与的证书声明表单中添加其团队的主要联系人电子邮件。
   - 此信息对于确保妥善沟通和提交管理至关重要。
- **提交状态确认**：一位成员请 Tara 确认其 Hackathon 提交是否已注册，Tara 给予了肯定答复，表示“我们已收到您的提交！”
   - 这一快速确认缓解了参赛者对提交错误的担忧。
- **YouTube 格式导致提交延迟**：一位参赛者解释说，由于 YouTube 上的视频格式问题导致提交延迟，他们通过电子邮件发送了 Hackathon 内容。
   - 他们强调自己主要关注 Hackathon 而非课程本身，并寻求对其提交状态的明确答复。
- **未来 MOOC 的 Agent 框架建议**：一位成员分享了一篇文章的见解，该文章反对仅依赖像 AutoGen 这样复杂的 LLM Agents 框架，而是建议采用更简单、可组合的模式。
   - 他们建议未来的 MOOC 在实验中应探索 AutoGen 的替代方案，强调需要关注 instruction tuning 和 function calling。



**提到的链接**: <a href="https://www.anthropic.com/research/building-effective-agents">Building effective agents</a>: 一篇为开发者提供构建高效 AI Agents 的建议和工作流的文章

  

---

### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1319780827026030613)** (1 条消息): 

> `Torchtune v0.5.0, Kaggle Integration, QAT + LoRA Training Recipe, Early Exit Training Recipe, NPU Support` 


- **Torchtune v0.5.0 带来节日更新**：Torchtune 发布了 **0.5.0** 版本，引入了多项新功能和集成增强，供用户在本季度体验。
   - 衷心感谢社区为促成此次发布所做的贡献，更多详情请参阅 [release notes](https://github.com/pytorch/torchtune/releases/tag/v0.5.0)。
- **Kaggle 集成增强微调体验**：用户现在可以在 [Kaggle notebooks](https://www.kaggle.com/code/felipemello/torchtune-in-kaggle) 中无缝微调模型，并与社区分享最佳的 checkpoints。
   - 此次集成旨在简化工作流，并促进 Torchtune 用户之间的协作。
- **引入 QAT + LoRA 训练 Recipe**：全新的 **QAT + LoRA** 训练 Recipe 允许用户以更高效率训练 [quant-friendly LoRA](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_2/3B_qat_lora.yaml) 模型。
   - 该 Recipe 是持续增强训练选项并适应现代模型开发需求努力的一部分。
- **通过 Early Exit 训练加速 LLM 推理**：**Early Exit 训练** 利用 [LayerSkip](https://github.com/pytorch/torchtune/pull/1076) 来提升 LLM 的推理速度和准确性。
   - 该功能旨在提供更高效的处理框架，从而实现更快的模型响应。
- **NPU 支持提升性能**：Torchtune 现在支持在 [Ascend NPU](https://github.com/pytorch/torchtune/pull/1826) 设备上运行，预计很快将添加分布式支持。
   - 这一新的兼容性将扩大 Torchtune 在不同硬件上的可用性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#use-your-model-in-the-wild)">End-to-End Workflow with torchtune &mdash; torchtune main documentation</a>: 未找到描述</li><li><a href="https://github.com/pytorch/torchtune/pull/1076).">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1319687043550543973)** (7 messages): 

> `QwQ-preview-32B 微调，fsdp2 的 State dict 加载，并行支持改进，梯度累积与裁剪，微调中的词表剪枝 (Vocab pruning)` 


- **QwQ-preview-32B 需要上下文并行 (context parallelism)**：一位用户分享了他们在 8*80G GPU 上微调 **QwQ-preview-32B** 的配置，并提出了关于支持上下文并行以将最大 Token 长度扩展到 **8K** 以上的问题。
   - 建议包括使用 **optimizer_in_bwd**、**8bit Adam optimizer**，以及探索 [QLoRA 优化标志 (optimization flags)](https://github.com/pytorch/torchtune#optimization-flags)。
- **为 fsdp2 加载 state dict 引发兼容性问题**：关于加载 **fsdp2** 的 state dict 存在疑虑，特别是参考 [分布式加载代码](https://github.com/pytorch/torchtune/blob/main/torchtune/training/_distributed.py#L213) 时，参数和缓冲区未被分片 (sharded) 的问题。
   - 关于 **FSDPModule** 的 **state_dict** 中是否存在不兼容的非 **DTensors** 尚不明确，这使部署场景变得复杂。
- **词表剪枝 (Vocab pruning) 需要在 fp32 中进行精细控制**：有开发者指出，在使用 **词表剪枝** 微调模型时，需要 state dict 将参数保持在 **fp32**，而计算则在 **bf16** 中进行。
   - 这一细节反映了在训练过程中对 Tensor 类型进行细致管理的持续需求。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/1244.">Issues · pytorch/torchtune</a>: PyTorch 原生后训练库。可以通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/training/_distributed.py#L213">torchtune/torchtune/training/_distributed.py at main · pytorch/torchtune</a>: PyTorch 原生后训练库。</li><li><a href="https://github.com/pytorch/torchtune#optimization-flags">GitHub - pytorch/torchtune: PyTorch native post-training library</a>: PyTorch 原生后训练库。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/">GitHub - pytorch/torchtune: PyTorch native post-training library</a>: PyTorch 原生后训练库。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/training/_distributed.py#L154">torchtune/torchtune/training/_distributed.py at main · pytorch/torchtune</a>: PyTorch 原生后训练库。
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1319397953495891978)** (7 messages): 

> `Litellm 代理服务器，合成数据对 LLM 的影响，优化参数，MIPRO Light 模式` 


- **Litellm 代理服务器部署选项**：Litellm 代理服务器可以自托管或通过托管服务使用，并且可以部署在与您的服务相同的 VM 上。
   - 这种灵活性允许用户根据其基础设施需求配置设置。
- **合成数据增强 LLM 性能**：一篇关于合成数据的入门文章讨论了它在提升 LLM（尤其是较小模型）性能方面的作用，通过将输入数据重塑为类似于聊天机器人对话的格式。
   - 虽然合成数据有助于开发推理模型，但它并非普遍有效，对于某些无法大规模测试的任务存在局限性。
- **优化过程的成本意识**：对于长时间运行优化器相关的成本存在担忧，引发了关于设置调用次数或 Token 限制的讨论。
   - 建议包括将优化参数配置得更小，或考虑安装带有定义限制的 LiteLLM。
- **利用 MIPRO 'Light' 模式**：有人建议利用 MIPRO 的 'light' 模式来更有效地管理优化过程。
   - 这种方法特别旨在平衡资源使用和性能。



**提及的链接**：<a href="https://www.dbreunig.com/2024/12/18/synthetic-data-the-growing-ai-perception-divide.html">关于合成数据：它如何改进和塑造 LLM</a>：合成数据正在帮助 LLM 跨越数据墙，但与此同时，它也在那些将 LLM 用于定量任务的人和将其用于其他任务的人之间制造了日益增长的认知鸿沟...

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1319404365714559058)** (7 条消息): 

> `OpenInterpreter Server Mode, Google Gemini 2.0 Multimodal, Local LLM Integration, SSH Usage with OpenInterpreter` 


- **对 OpenInterpreter 服务器模式的好奇**：一名成员询问了关于在服务器模式下运行 OpenInterpreter 时进行交互的文档，并表示有兴趣在 VPS 上进行设置。
   - *在服务器模式下，是否可以区分命令是在本地运行还是在服务器上运行？*
- **关于 Google Gemini 2.0 能力的反馈**：另一名成员询问是否有人尝试过新的 **Google Gemini 2.0** 多模态功能，特别是它的 *os mode*。
   - 他们提到了对访问权限的担忧，指出该功能可能仅限于 *tier 5* 用户。
- **对本地 LLM 集成的赞赏**：一位成员对持续支持本地 LLM 集成表示欣喜，认为这为 OpenInterpreter 增添了亲和力。
   - 他们最初担心这可能会变成 OpenAI 专属，但目前它仍然是一个深受欢迎的功能。
- **通过 SSH 使用 OpenInterpreter**：一位用户分享了他们在常规模式下使用 OpenInterpreter 的经验，通过 SSH 连接以方便访问。
   - 他们对集成前端表示兴奋，并相信自己能够处理好。
- **对推荐链接垃圾信息的担忧**：一名成员提醒其他人注意推荐链接垃圾信息 (referral spam)，指出聊天中存在此类链接。
   - 他们标记了一个特定角色，以引起社区对该问题的关注。


  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1319705697310736455)** (4 条消息): 

> `Liger and KTO integration, Liger DPO, Loss parity issues` 


- **Liger 现已集成 KTO**：确认 **Liger** 现在已经实现了 **KTO** 功能。
   - 这一集成被视为开发过程中的一个进步。
- **正在开发 Liger DPO**：一名成员报告称，他们目前正专注于让 **Liger DPO** 投入运行，随后可能会集成 **KTO**。
   - 他们提到在将 Liger 与 HF TRL 基准 (baseline) 进行比较时遇到了 **Loss 一致性问题 (loss parity issues)**。
- **社区对问题的关注**：一位成员针对持续存在的挑战回复道：*“痛苦 (Pain)”*。
   - 另一位成员表示希望 **Loss 一致性问题** 能尽快得到解决。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 条消息): 

chenyuy: 我下周将关闭（或找个机器人来关闭）不活跃超过 30 天的 PRs。
  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1319544081801674772)** (1 条消息): 

> `Watt-tool models, GitHub Pull Requests, Christmas timeframe` 


- **引入新的 Watt-tool 模型**：已提交一个 [GitHub Pull Request](https://github.com/ShishirPatil/gorilla/pull/847)，将 **watt-tool-8B** 和 **watt-tool-70B** 模型添加到排行榜 (leaderboard)。
   - 这些模型也可以在 Hugging Face 上找到：[watt-tool-8B](https://huggingface.co/watt-ai/watt-tool-8B/) 和 [watt-tool-70B](https://huggingface.co/watt-ai/watt-tool-70B)。
- **请求 PR 评审支持**：请求协助检查新提交的与 watt-tool 模型相关的 Pull Request 是否存在问题。
   - *圣诞节即将到来*，因此贡献者鼓励大家根据需要安排评审时间。



**提到的链接**：<a href="https://github.com/ShishirPatil/gorilla/pull/847">[BFCL] Add New Model `watt-tool-8B` and `watt-tool-70B` by zhanghanduo · Pull Request #847 · ShishirPatil/gorilla</a>：此 PR 将模型 watt-ai/watt-tool-8B 和 watt-ai/watt-tool-70B 添加到排行榜中。

  

---


---


---


---


{% else %}


> 完整的频道明细已因邮件长度而截断。
> 
> 如果你想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}