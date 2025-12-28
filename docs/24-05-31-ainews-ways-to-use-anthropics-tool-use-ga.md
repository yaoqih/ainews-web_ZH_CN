---
companies:
- anthropic
- amazon
- google
date: '2024-05-31T20:31:29.874216Z'
description: '**Anthropic** 联合 **亚马逊（Amazon）** 和 **谷歌（Google）** 正式全面开放了工具使用（tool use）/
  函数调用（function calling）功能，并支持流式传输、强制使用和视觉能力。Alex Albert 分享了智能体工具使用的五种架构：委派（delegation）、并行化（parallelization）、辩论（debate）、专业化（specialization）和工具套件专家（tool
  suite experts）。此外，**Anthropic** 还推出了一门关于工具使用的自学课程。


  **Yann LeCun** 强调了伦理开放科学资助、带有安全护栏的超智能的逐渐显现，以及用于图像/视频处理的卷积网络在竞争力上可与视觉 Transformer（Vision
  Transformers）相媲美。他还指出，工业界、学术界和政府部门的 AI 研究人员数量均有所增长。'
id: 72ea2fa2-ac11-4be2-90ca-e6963178858a
models:
- claude-3-opus
- haiku
- opus
- convnext
original_slug: ainews-ways-to-use-anthropics-tool-use-ga
people:
- yann-lecun
- alex-albert
- sainingxie
title: '以下是“Ways to use Anthropic''s Tool Use GA”的中文翻译：


  **Anthropic 工具使用功能（GA/正式版）的使用方式**


  （注：“GA”意为 General Availability，指功能已正式全面开放。）'
topics:
- tool-use
- function-calling
- agentic-ai
- streaming
- vision
- parallelization
- delegation
- debate
- specialization
- open-science
- superintelligence
- convolutional-networks
- self-attention
- ai-research
---

<!-- buttondown-editor-mode: plaintext -->**工具是 AI 所需的一切。**

> 2024年5月30日至5月31日的 AI 新闻。
我们为您检查了 7 个 subreddit、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 服务区（**393** 个频道和 **2911** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**337 分钟**。

伴随着 Anthropic 今天在 Anthropic/Amazon/Google 上正式发布（GA）工具使用/函数调用（tool use/function calling），并支持[流式传输](https://x.com/alexalbert__/status/1791137394563133849)、[强制使用](https://x.com/alexalbert__/status/1791137396798677234)和[视觉](https://x.com/alexalbert__/status/1791137398266659286)...

 
![image.png](https://assets.buttondown.email/images/f1926fa7-897a-46e4-a5b0-70f36ee3a18f.png?w=960&fit=max)
 

Alex Albert 分享了在 Agent 上下文中使用它们的 [5 种架构](https://x.com/alexalbert__/status/1796211969432887331)：

1. **Delegation（委派）**：使用更便宜、更快的模型来获得成本和速度优势。
  - 例如，Opus 可以委派 Haiku 阅读一本书并返回相关段落。如果任务描述和结果比完整上下文更紧凑，这种方式效果很好。
2. **Parallelization（并行化）**：通过并行运行 Agent 来降低延迟（但不会降低成本）。
  - 例如，100 个子 Agent 分别阅读一本书的不同章节，然后返回关键段落。
3. **Debate（辩论）**：具有不同角色的多个 Agent 进行讨论以达成更好的决策。
  - 例如，软件工程师提议代码，安全工程师进行审查，产品经理提供用户视角，最后由一个最终 Agent 进行综合并做出决定。
4. **Specialization（专业化）**：一个通用型 Agent 进行编排，而专家型 Agent 执行任务。
  - 例如，主 Agent 在处理健康查询时使用专门提示（或微调）的医疗模型，在处理法律问题时使用法律模型。
5. **Tool Suite Experts（工具套件专家）**：当使用成百上千个工具时，让 Agent 专注于工具子集。
  - 每个专家（相同的模型，但配备不同的工具）处理特定的工具集。编排器随后将任务映射到正确的专家（这能保持编排器的 Prompt 简洁）。

这里没有什么特别突破性的内容，但对于思考模式来说是一个非常方便的清单。Anthropic 还[推出了一个关于工具使用的自学课程](https://x.com/alexalbert__/status/1796610971810853165)：

 
![image.png](https://assets.buttondown.email/images/26a939e9-7542-49ec-a460-7b68ec84591c.png?w=960&fit=max)
 

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有摘要均由 Claude 3 Opus 完成，从 4 次运行中择优。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**AI 研究与开发**

- **开放科学与研究资助**：[@ylecun](https://twitter.com/ylecun/status/1796486618620051933) 为研究表达了一个明确的伦理准则：“不要从限制你发表能力的实体那里获取研究资金。”他强调，**向世界提供新知识本质上是件好事**，无论资金来源如何。[@ylecun](https://twitter.com/ylecun/status/1796488253815603279) 指出，这一伦理准则使他成为了**开放科学和开源的坚定倡导者**。
- **超级智能的出现**：[@ylecun](https://twitter.com/ylecun/status/1796543960296440162) 认为超级智能的出现将是一个**渐进的过程**，而不是突然发生的事件。他设想从老鼠或松鼠智力水平的架构开始，**逐步提升其智力**，同时设计适当的护栏和安全机制。目标是设计出一种**能够完成人类指定目标的、目标驱动型 AI（objective-driven AI）**。
- **用于图像和视频处理的卷积网络**：[@ylecun](https://twitter.com/ylecun/status/1796265384976252991) 建议在低层使用**带步长或池化的卷积，在高层使用自注意力电路**来进行实时图像和视频处理。他认为 [@sainingxie](https://twitter.com/sainingxie) 在 ConvNext 上的工作已经表明，**如果做得正确，卷积网络可以和 Vision Transformer 一样出色**。[@ylecun](https://twitter.com/ylecun/status/1796263750485295560) 认为自注意力对排列具有等变性，这对于低层图像/视频处理来说是没有意义的，而且由于图像和视频中的相关性是高度局部的，**全局注意力是不可扩展的**。
- **工业界 vs 学术界的 AI 研究人员**：[@ylecun](https://twitter.com/ylecun/status/1796518343194845658) 指出，如果图表显示的是绝对数量而不是百分比，就会发现**工业界、学术界和政府部门的 AI 研究人员数量都在增长**，只是工业界的增长比其他部门更早、更快。

**AI 工具与应用**

- **Suno AI**：[@suno_ai_](https://twitter.com/suno_ai_/status/1796273804991156326) 宣布发布 Suno v3.5，允许用户**在单次生成中制作 4 分钟的歌曲**，创建 2 分钟的歌曲扩展，并体验改进的歌曲结构和人声流。他们还将在 **2024 年向顶尖 Suno 创作者支付 100 万美元**。[@karpathy](https://twitter.com/karpathy/status/1796305221813198946) 表达了他对 Suno 的喜爱，并分享了他使用该工具创作的一些最喜欢的歌曲。
- **Anthropic 的 Claude**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1796210547077128578) 宣布 **Claude 的 Tool Use（工具使用）现已在 API、Amazon Bedrock 和 Google Vertex AI 中全面开放**。通过 Tool Use，Claude 可以**智能地选择和编排工具，端到端地解决复杂任务**。早期客户正利用 Claude 的 Tool Use 构建定制化体验，例如 [@StudyFetch](https://twitter.com/StudyFetch) 使用 Claude 驱动个性化 AI 导师 Spark.E。[@HebbiaAI](https://twitter.com/HebbiaAI) 使用 Claude 为其 AI 知识工作者驱动复杂的多步骤客户工作流。
- **Perplexity AI**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1796220011448786949) 推出了被描述为“AI 维基百科”的 Perplexity Pages，允许用户通过简单的“一键转换”**分析来源并合成可读页面**。Pages 已对所有 Pro 用户开放，并正向所有人广泛推广。用户可以将页面创建为独立实体，或**将他们的 Perplexity 聊天会话转换为页面格式**。[@perplexity_ai](https://twitter.com/perplexity_ai/status/1796203494401040846) 指出，Pages 让用户能够通过**格式化的图像和章节**分享关于任何主题的深入知识。
- **DeepMind 的 Gemini**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1796216673348833445) 宣布开发者现在可以开始使用其 API 按需付费服务构建 **Gemini 1.5 Flash 和 Pro 模型**。Flash 旨在提供**快速且高效的服务**，其速率限制提高到每分钟 1000 次请求。

**梗与幽默**

- [@huybery](https://twitter.com/huybery/status/1796532108024000674) 推出了 im-a-good-qwen2，一个**在评论中互动的聊天机器人**。
- [@karpathy](https://twitter.com/karpathy/status/1796556328078619103) 分享了他对 1 对 1 会议的看法，表示他在 Tesla 有大约 30 名直接下属，但**不进行 1 对 1 会议，他认为这非常棒**。他发现 4-8 人的会议和用于广播的大型会议更有用。
- [@ReamBraden](https://twitter.com/ReamBraden/status/1796257883623145895) 分享了一个**关于初创公司创始人挑战的梗图**。
- [@cto_junior](https://twitter.com/cto_junior/status/1796237607522758914) 分享了一个**关于腾讯 AI 开发者致力于取代低薪动漫艺术家的梗图**。
- [@nearcyan](https://twitter.com/nearcyan/status/1796245651174605032) 对那些认为我们不应该与动物交谈、建造房屋或发电厂，而应该**“在洞穴里腐烂，像上帝旨意那样为残羹剩饭而战”**的人发表了幽默的评论。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 图像与视频生成**

- **写实头像**：在 r/singularity 中，展示了来自德国慕尼黑大学 Neural Parametric Gaussian Avatars (NPGA) 研究的令人印象深刻的写实头像（[示例 1](https://v.redd.it/o8rxrfnmpj3d1)，[示例 2](https://v.redd.it/drghzm0mrj3d1)）。这些高质量的头像展示了 AI 生成的人类表现形式的快速进步。
- **卡通生成与插值**：推出了用于生成和插值卡通风格图像的 ToonCrafter 模型，并有[实验展示了其功能](https://v.redd.it/x1vda6as6m3d1)。这突显了 AI 生成内容在写实图像之外的不断扩展。
- **AI 驱动的游戏开发**：展示了一个[开源游戏引擎](https://v.redd.it/imlu0qd1tj3d1)，它利用 Stable Diffusion、AnimateDiff 和 ControlNet 等 AI 模型来生成游戏资产和动画。该引擎的[源代码](https://github.com/WorldQL/dreamlab-core)和[渲染动画精灵的技术](https://drive.google.com/file/d/1BPvC-PLF-__ey6KjmxTJZ_bg55oThHdr/view?usp=sharing)已完全公开。
- **AI 动画 API**：Animate Anyone API（代码可在 [GitHub](https://i.redd.it/fuglsjleli3d1) 获取）能够为图像中的人物制作动画。然而，评论建议像 [MusePose](https://github.com/TMElyralab/MusePose) 这样的替代方案可能会提供更好的效果。

**AI 伦理与社会影响**

- **AI 合作伙伴关系与竞争**：Microsoft CEO Satya Nadella 对 [OpenAI 与 Apple 之间潜在的交易表示担忧](https://www.businessinsider.com/satya-nadella-sam-altman-openai-apple-microsoft-worried-about-deal-2024-5)，强调了 AI 合作伙伴关系的战略重要性以及竞争格局。
- **Deepfake 担忧**：Deepfake 技术被滥用的可能性日益增加受到[强调](https://i.redd.it/s19j6oq7kk3d1.jpeg)，凸显了对安全保障和负责任 AI 实践的需求。
- **电影行业中的 AI**：Sony 计划[利用 AI 降低电影制作成本](https://www.indiewire.com/news/breaking-news/sony-pictures-will-cut-film-costs-using-ai-1235010605/)，这引发了关于对创意产业影响以及潜在失业问题的讨论。
- **AI 生成内容与真实性**：一张名为 "All Eyes On Rafah" 的 AI 生成图像因缺乏真实感并可能误导敏感局势而[面临批评](https://www.ndtv.com/world-news/ai-generated-all-eyes-on-rafah-pic-criticised-for-being-removed-from-reality-5777680)，凸显了 AI 生成内容面临的挑战。
- **AI 与影响力行动**：OpenAI [报告称](https://www.nytimes.com/2024/05/30/technology/openai-influence-campaigns-report.html?unlocked_article_code=1.v00.eijy.DJp94u_PuyG7)俄罗斯和中国利用其 AI 工具进行秘密影响力行动，强调了采取主动措施防止 AI 滥用的必要性，正如其[打击欺骗性 AI 使用的努力](https://openai.com/index/disrupting-deceptive-uses-of-AI-by-covert-influence-operations/)中所详述的那样。

**AI 能力与进展**

- **生物处理器与脑类器官**：一项突破性的[利用人脑类器官的生物处理器](https://www.reddit.com/r/singularity/comments/1d4bcoa/worlds_first_bioprocessor_uses_16_human_brain/)被开发出来，与数字芯片相比，它提供了极高效率的计算。
- **医疗保健中的 AI**：根据发表在《柳叶刀》（The Lancet）上的一项里程碑式研究，新的 AI 技术被[证明可以提前 10 年预测与冠状动脉炎症相关的心脏事件](https://www.reddit.com/r/singularity/comments/1d4bc8w/new_ai_tech_predicts_cardiac_events_due_to/)。
- **量子计算突破**：由一名留美归国物理学家领导的中国研究团队[声称建造了世界上最强大的离子型量子计算机](https://www.scmp.com/news/china/science/article/3264742/us-returned-chinese-physicist-duan-luming-and-team-build-worlds-most-powerful-ion-based-quantum)。

**OpenAI 新闻与动态**

- **领导层澄清**：Paul Graham [澄清 Y Combinator 并没有解雇 Sam Altman](https://twitter.com/paulg/status/1796107666265108940)，这与流传的谣言相反。
- **机器人研究重启**：OpenAI 正在[重启其机器人团队](https://www.forbes.com/sites/kenrickcai/2024/05/30/openai-robotics-team/?sh=1f4e2d2c4f33)，标志着其重新关注 AI 与机器人的交叉领域。
- **回应担忧**：OpenAI 董事会成员[回应了前成员提出的警告](https://www.economist.com/by-invitation/2024/05/30/openai-board-members-respond-to-a-warning-by-former-members)，涉及公司的发展方向和实践。
- **面向非营利组织的 AI**：OpenAI [发起了一项倡议](https://openai.com/index/introducing-openai-for-nonprofits/)，使其工具更易于被非营利组织使用，从而推广有益的 AI 应用。
- **与 Reddit 的合作伙伴关系**：[OpenAI 与 Reddit 宣布建立合作伙伴关系](https://openai.com/index/openai-and-reddit-partnership/)，引发了关于对这两个平台潜在影响的讨论。

**AI 幽默与迷因**

- **机器人的过去与现在**：有人分享了一个[幽默的对比](https://v.redd.it/ozsk6wqemp3d1/ozsk6wqemp3d1)，将电影《我，机器人》（I, Robot）过去对机器人的描绘与现状进行了比较。

---

# AI Discord 摘要

> 摘要之摘要的摘要

**1. 模型性能优化与基准测试**

- **K2 超越 Llama 2**：来自 [LLM360 的 K2 模型](https://huggingface.co/LLM360/K2)在性能上超越了 **Llama 2 70B**，同时减少了 35% 的计算量，并基于 Apache 2.0 协议完全开源。

- **NeurIPS 举办模型融合竞赛**：一项奖金为 8,000 美元的竞赛邀请参赛者融合最优 AI 模型，详情请见 [NeurIPS Model Merging 网站](https://llm-merging.github.io/)。

- **定制化 Positional Embeddings 提升 Transformer 算术能力**：研究人员通过使用特定的 Embeddings，在 **100 位数字加法上实现了 99% 的准确率**，详见其[论文](https://huggingface.co/papers/2405.17399)。

**2. 微调与 Prompt Engineering**

- **解决数据集合并与训练技巧**：Axolotl 用户讨论了在微调过程中有效合并数据集以避免灾难性遗忘（catastrophic forgetting）等问题。推荐工具包括 **[Hugging Face Accelerate](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/config.qmd#L325C1-L327C27)**。

- **法律草案系统与聊天机器人的微调技术**：为 **法律草案** 和 **财务文档摘要** 等应用微调 LLM 的用户交流了策略，参考资源包括 [Fine-Tune PaliGemma](https://youtu.be/hDa-M91MSGU?si=4QKcNZsB40ibgPyd)。

- **解决文本分类模型的训练问题**：针对西班牙语实体分类模型训练中的问题，提出了微调建议，并探索了 **RoBERTa** 等框架。

**3. 开源 AI 发展与协作**

- **用于高效向量存储的 Milvus Lite**：介绍了 **Milvus Lite**，这是一个针对 Python 的轻量级向量存储解决方案，详见 [Milvus 文档](https://milvus.io/docs/milvus_lite.md)。

- **MixMyAI 在单一平台上集成多个 AI 模型**：[mixmyai.com](https://mixmyai.com) 平台整合了开源和闭源模型，强调隐私保护，且不在服务器存储聊天数据。

- **LlamaIndex 提供灵活的检索系统**：新的基于 Django 的 Web 应用模板促进了 **RAG (Retrieval Augmented Generation)** 应用的开发，利用了数据管理和用户访问控制，详见[此处](https://t.co/kx3DhxfDZu)。

**4. AI 社区创新与知识共享**

- **使用 Axolotl 保持一致的 Prompt 格式**：对 Axolotl 进行了调整以确保 Prompt 格式的一致性，引导用户参考 **[Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/prompters.py#L47)** 等设置。
  
- **Ghost XB Beta 中的语言支持挑战**：Unsloth AI 讨论了 **Ghost XB Beta** 等模型的多语言支持，目标是在训练阶段支持 9 种以上语言，并重点介绍了 [Ghost XB 详情](https://ghost-x.org)。

- **集成 OpenAI 和 LangChain 工具**：讨论了利用 [LangChain Intro](https://python.langchain.com/v0.2/docs/introduction/) 等资源以及 GPT-4 Alpha 宣布的实时功能来创建高级 AI 应用。

**5. 硬件进展与兼容性挑战**

- **NVIDIA 新型 4nm 研究芯片表现惊人**：实现了 **96 int4 TOPs/Watt** 的效率，显著超越了 Blackwell 的能力，相关影响的讨论分享在[此处](https://x.com/danielhanchen/status/1796253349932843214)。

- **AMD GPU 上的 ROCm 支持挑战**：用户对 ROCm 缺乏对 **RX 6700 和 RX580** 等 GPU 的支持感到沮丧，引发了对潜在替代方案和性能影响的讨论。

- **在 CUDA 中实现高效的数据处理**：讨论了优化 CUDA 操作，使用融合逐元素操作（fusing element-wise operations）等技术来获得更好的性能，源代码见[此处](https://github.com/modularml/mojo/blob/bf73717d79fbb79b4b2bf586b3a40072308b6184/stdlib/src/memory/memory.mojo)。

---

# 第一部分：高层级 Discord 摘要

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Transformer 训练与故障排除**：社区请求开展关于 **Transformer architecture** 的互动环节，以更好地理解 **RoPE** 和 **RMS Normalization** 等复杂主题。与此同时，**Google Gemini Flash** 将于 6 月 17 日开放免费微调，而在使用 **RAG LLMs** 进行生产时，必须仔细计算成本，优先考虑 GPU 时间和第三方服务。社区提倡使用 **GGUF format** 以保持与生态系统工具的兼容性，从而简化微调过程。([针对图像转 JSON 场景微调 PaliGemma](https://youtu.be/hDa-M91MSGU?si=4QKcNZsB40ibgPyd))

- **挑战 BM25 与检索难题**：社区发起了一项寻找可靠 **BM25 implementation** 的行动，Python 包 `rank_bm25` 因其功能有限而成为关注焦点。对话集中在增强向量检索上；同时，**Modal** 用户在部署 **v1 finetuned models** 后可参考 [documentation](https://modal.com/docs/guide/trigger-deployed-functions) 进行后续操作，**[Modal credits initiative](https://tally.so/r/wQ5GRl)** 澄清了额度过期的疑虑。

- **数据主导对话**：从文档 AI 中解析结构化信息需要 OCR 和 LiLT 模型等技术。与此同时，社区考虑使用 OpenPipe 和 **GPT-4** 处理 5000 个抓取的 LinkedIn 个人资料，多模态（multi-modal）方法和 **document understanding** 仍是热门话题。一个排错技巧是：强调训练数据文件格式必须精确匹配，以防止出现 `KeyError: 'Input'`。

- **学习 LLM 基础与 LangChain 联动**：来自 **Humble Bundle** 和 **Sebastian Raschka** 的资源提供了关于 prompt engineering 和 LLM finetuning 的见解，尽管有人对某些材料的质量提出质疑。反映出社区对知识的渴求，O'Reilly 发布了其 LLM 构建系列丛书的 **Part II**，重点关注 LLM 应用中的运营挑战。

- **策划对话上下文**：剖析了 instruct-LLM 和 chat-LLM 模型之间的区别，前者遵循明确指令，后者精通对话上下文。讨论的项目范围广泛，从类似 **Alexa 的音乐播放器**到**法律草案系统**，以及用于财务文档摘要的聊天机器人，展示了经过微调的 LLM 的各种可能实现。

- **Modal 动态与市场触达**：博客等媒介在传播知识方面发挥了至关重要作用，**[John Whitaker 的博客](https://johnowhitaker.dev)** 成为学习 *Basement Hydroponics* 和 LLM 性能的首选之地。从业者分享了梯度优化技巧，如 **[gradient checkpointing](https://discord.com/channels/1238365980128706560/1242223332695478332/1246108353210355805)**，并一致认为有时最简单的解释（如 Johno 课程中的解释）效果最好。

- **Spaces 的空间**：讨论了如何在 Axolotl 中更改 **alpaca format** 的提示词样式以及 **Qwen tokenizer** 的使用问题，并引用了特定的 **[GitHub configs](https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2/src/axolotl/prompters.py#L47)**。同时，由于易用性，部署基于 **Gradio 的 RAG app** 引发了对使用 **HF Spaces** 的兴趣。

- **额度热潮与社区连接**：紧急公告强调了填写额度申请表的紧迫性，引发了片刻的恐慌与澄清。社交聚会和讨论范围从旧金山的 eval 见面会到 Modal Labs 在纽约举办的答疑时间（office hours），表明了活跃的社区联系和知识共享活动。

- **欧洲参与及 Predibase 前景**：来自欧洲各地（如德国纽伦堡和意大利米兰）的签到显示了该群体的地理跨度。此外，提到 **[30-day free trial of Predibase](https://predibase.com/free-trial)** 提供 25 美元的额度，反映了为提供便捷的微调和部署平台所做的持续努力。

- **职业十字路口**：从学术界到工业界，成员们分享了经验并鼓励彼此进行职业转型。讨论展示了外包合同（contracting）是一种可行的路径，导师指导和毅力被认为是应对技术环境的关键，而 GitHub 作品集可以作为重要的敲门砖。

这些摘要概括了 Discord 公会中 AI 工程师们详尽且往往非常细致的讨论，突显了他们在追求职业成长和社区建设的同时，为优化 LLM 微调和部署所做的集体努力。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**K2 战胜 Llama 2**：[LLM360 的 K2 模型](https://huggingface.co/LLM360/K2) 超越了 **Llama 2 70B**，以减少 35% 的计算量实现了更好的性能；该模型被宣传为**完全可复现**，并根据 Apache 2.0 许可证开放。

**数字难不倒 Positional Embeddings**：研究人员攻克了 Transformer 的算术能力难题；通过定制的 Positional Embeddings，Transformer 在 100 位数的加法运算中达到了 **99% 的准确率**，这一里程碑式的成就详见他们的 [论文](https://huggingface.co/papers/2405.17399)。

**NeurIPS 发起模型合并挑战**：**NeurIPS 模型合并竞赛 (Model Merging Competition)** 设立了 8,000 美元的奖金，邀请参赛者融合出最优的 AI 模型。Hugging Face 等机构赞助了此次竞赛，更多信息请见 [公告](https://x.com/LChoshen/status/1796256513519989102) 和 [竞赛网站](https://llm-merging.github.io/)。

**数据探索：从 15 万个数据集到服装销售**：工程师们现在可以通过 DuckDB 探索超过 15 万个数据集的宝库，这在一篇 [博客文章](https://huggingface.co/blog/chilijung/access-150k) 中有详细说明；同时，一个新的服装销售数据集推动了图像回归模型的发展，详情见 [这篇文章](https://huggingface.co/blog/tonyassi/image-regression)。

**学习资源与课程助力技能提升**：在不断进步的 AI 领域，工程师可以通过 Hugging Face 的 **Reinforcement Learning**（强化学习）和 **Computer Vision**（计算机视觉）课程来增强专业知识，更多信息可在 [Hugging Face - Learn](https://huggingface.co/learn) 获取。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**量化困境与高效硬件**：Unsloth AI 协会成员强调了量化版 **Phi3 微调** 结果面临的挑战，指出在没有量化技巧的情况下存在性能问题。NVIDIA 新推出的 *4nm 研究芯片* 因其每瓦 96 int4 tera operations per second (TOPs/Watt) 的效率引发热议，超越了 Blackwell 的 20T/W，反映了行业在功耗效率、数值表示、Tensor Cores 效率和稀疏化技术方面的全面进步。

**模型微调与 Upscaling 讨论**：AI 工程师分享了关于微调策略的见解，包括 **数据集合并**，其中一位成员展示了一个使用 Upscaling 技术构建的 Llama-3 **11.5B Upscale 模型**。一种新兴的微调方法 **MoRA** 为参数高效更新提供了一条充满前景的途径。

**故障排除工具与技术**：工程师们面临各种障碍，从 Unsloth 中的 GPU 选择（`os.environ["CUDA_VISIBLE_DEVICES"]="0"`）和微调错误排查，到处理双模型依赖关系以及解决训练期间的 VRAM 峰值问题。针对 Kaggle 安装挑战等问题的变通方案强调了细致解决问题的必要性。

**多语言 AI**：**Ghost XB Beta** 因其能够流利支持 9 种以上语言而受到关注，目前正处于训练阶段。这一进展再次确认了协会致力于为社区开发易于获取、具有成本效益的 AI 工具的承诺，特别是对初创企业的支持。

**社区协作与增强**：协会讨论显示了对自我部署和社区支持的集体推动，成员们分享了更新并在 *Open Empathic* 项目和 Unsloth AI 模型改进等一系列 AI 相关工作中寻求帮助。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Tako 小组件的地理范围限制？**：关于 **Tako 金融数据小组件** 的讨论引发了对其地理限制的疑问，一些用户不确定它是否为美国专属。

- **Perplexity Pro 试用结束**：用户讨论了 **Perplexity Pro 试用** 的停止，包括年度 7 天试用选项，引发了关于潜在推荐策略和自费试用的对话。

- **Perplexity 页面部分编辑的特性**：在编辑 **Perplexity Pages** 章节时出现了一些困惑，用户可以修改章节详情但不能修改文本本身——这一限制已得到多位成员的确认。

- **搜索性能的权衡**：观察到 **Perplexity Pro 搜索速度变慢**，这归因于一种将查询顺序分解的新策略，尽管速度降低，但能提供更详细的回答。

- **探索 Perplexity 新功能**：用户分享了新推出的 **Perplexity Pages** 链接以及关于 **Codestral de Mistral** 的讨论，暗示了 Perplexity AI 平台内的功能增强或服务。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA 和 Meta 的芯片创新引发热议**：社区对 NVIDIA 披露的一款 4nm **推理芯片**感到兴奋，该芯片实现了 **96 int4 TOPs/Watt**，超越了此前 20T/W 的基准；同时 Meta 发布了**下一代 AI 加速器**，在仅 90W 功耗下达到 354 TFLOPS/s (INT8)，标志着 AI 加速能力的飞跃。

- **深入探讨 CUDA 和 GPU 编程**：关于 **FreeCodeCamp CUDA C/C++ 课程**的公告引起了极大热情，该课程旨在简化 GPU 编程陡峭的学习曲线。课程内容需求强调了涵盖矩形矩阵 GEMM 以及与图像卷积应用相关的广播规则的重要性。

- **理解 Scan 算法和并行计算**：社区热切期待 **scan 算法**系列的第二部分。同时，针对 `Single-pass Parallel Prefix Scan with Decoupled Look-back` 论文中提到的并行 scan 算法的实际挑战提出了疑问，并请求澄清 Triton 中的 CUDA kernel 命名，以便在 kernel profiling 中提高可追溯性。

- **分享模型训练和数据优化策略**：讨论内容包括分享高效的同构模型参数共享策略，以避免 PyTorch 在 batching 过程中低效的复制；以及模型训练期间的 loss 尖峰等问题，这些问题可能通过梯度范数（gradient norm）绘图进行诊断。有人提议将数据集托管在 Hugging Face 上以方便访问，并建议使用压缩方法加快下载速度。

- **庆祝跨平台兼容性和社区胜利**：讨论了将 CUDA 和机器学习库兼容性扩展到 Windows 的进展与挑战，并承认了 Triton 的复杂性。同时，社区庆祝一个仓库达到 20,000 stars，并分享了关于结构化和合并目录的更新以增强组织性，加强了社区内的持续协作。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **在线内容隐私呼吁教育**：一位参与者强调了不发布内容作为隐私保护措施的重要性，并强调需要教育人们了解向公司提供内容的风险。
  
- **力求 AI 工具结果的一致性**：用户注意到使用 **ComfyUI** 与 **Forge** 相比存在不一致性，认为尽管初始设置相同，但不同的设置和功能（如 **XFormers**）可能会影响结果。

- **讨论合并 AI 模型的策略**：对话围绕合并 **SDXL** 和 **SD15** 等模型以提高输出质量的潜力展开，尽管在模型各阶段确保一致的 ControlNet 仍然至关重要。

- **分享自定义 AI 模型训练见解**：爱好者们交流了训练定制模型的技巧，提到了用于 Lora 模型训练的 **OneTrainer** 和 **kohya_ss** 等资源，并分享了有用的 YouTube 教程。

- **推荐 AI 探索的初学者资源**：对于 AI 新手，建议从 [Craiyon](https://www.craiyon.com) 等简单工具开始，在进阶到更复杂的平台之前感受图像生成 AI。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**ROCm 带来的 GPU 忧郁？这可不是什么悦耳的音乐**：工程师们讨论了 ROCm 的 GPU 性能，感叹缺乏对 RX 6700 和 RX580 等旧款 AMD GPU 的支持，这影响了 token 生成速度和整体性能。在 **LLAMA 3 8B Q8** 等模型的多 GPU 系统上寻求性能基准测试的用户报告称，双 GPU 与单 GPU 相比，效率达到了 91%。

**显存焦虑**：LM Studio 模型的发布引发了关于 VRAM 充足性的辩论，其中 4070 的 12GB 显存与 1070 的 20GB 相比显得捉襟见肘，特别是在运行像 "codestral" 这样的大型模型时。

**CPU 限制束缚手脚**：运行 LM Studio 的 CPU 要求成为焦点，其中 **AVX2 instructions** 被证明是强制性的，导致使用旧款 CPU 的用户不得不改用之前的版本 (0.2.10) 以支持 AVX。

**路由到正确的模板**：AI 工程师分享了模型模板的解决方案和建议，例如为某些模型使用 Deepseek coder 提示词模板，并建议检查 tokenizer 配置，以便在 **TheBloke/llama2_7b_chat_uncensored-GGUF** 等模型上获得最佳格式。

**领域新秀 - InternLM 模型**：发布了多款专为数学和编程设计的 **InternLM** 模型，参数范围从 7B 到 mixtral 8x22B。像 [AlchemistCoder-DS-6.7B-GGUF](https://huggingface.co/lmstudio-community/AlchemistCoder-DS-6.7B-GGUF) 和 [internlm2-math-plus-mixtral8x22b-GGUF](https://huggingface.co/lmstudio-community/internlm2-math-plus-mixtral8x22b-GGUF) 这样的模型在 AI 工程师可用的最新工具中备受关注。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **全球 API 请求速度提升**：OpenRouter 实现了全球范围内的延迟降低，将请求时间减少了约 200ms，通过优化边缘数据传输，特别惠及了来自非洲、亚洲、澳大利亚和南美洲的用户。

- **MixMyAI 推出统一的按需付费 AI 平台**：一项名为 mixmyai.com 的新服务已上线，它在一个用户友好的界面中整合了开源和闭源模型，强调用户隐私并避免在服务器上存储聊天记录。

- **MPT-7B 重新定义 AI 上下文长度**：Latent Space 播客展示了 MosaicML 的 MPT-7B 模型，以及它在突破 GPT-3 上下文长度限制方面的进展，详情见这篇[深度访谈](https://www.latent.space/p/mosaic-mpt-7b?utm_source=substack&utm_medium=email)。

- **Ruby 开发者迎来新的 AI 库**：一个新的 [OpenRouter Ruby client library](https://github.com/OlympiaAI/open_router) 已经发布，同时更新的还有 [Ruby AI eXtensions for Rails](https://github.com/OlympiaAI/raix-rails)，这些是 Ruby 开发者将 AI 集成到应用程序中的必备工具。

- **服务器稳定性和健康检查受到质疑**：OpenRouter 用户在不同地区遇到了零星的 504 错误，目前已提供临时解决方案，讨论倾向于需要一个专门的健康检查 API 来实现更可靠的状态监控。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Pro 权限提升聊天生产力**：OpenAI 的 Pro 用户现在享有**增强功能**，例如**更高的速率限制 (Rate Limits)**、专属 GPT 创建，以及访问 **DALL-E** 和实时通信功能。尽管每月费用为 20 美元，但这一诱人的提议依然极具魅力，与非付费用户可用的有限工具集形成了鲜明对比。

**AI 框架偏好助力功能灵活性**：对于开发具有独特个性特征的 AI 角色的人员，推荐使用 **Chat API** 而非 Assistant API，因为它提供了卓越的命令执行能力，且没有文件搜索等冗余功能。

**偏见争议困扰 ChatGPT**：由于指出 ChatGPT 输出中存在的种族主义倾向而导致的账号停用，引发了关于**模型固有偏见**的激烈争论，凸显了在训练数据根深蒂固的细微差别中不断寻求减轻此类偏见的努力。

**虚拟视频探索得到验证**：**Sora 和 Veo** 成为投机热潮的主角，社区正在权衡这些先锋视频生成模型的策划声明和实际潜力，并将其与 AI 辅助视频制作的现实进行对比。

**API 动荡与进展公告**：**内存泄漏 (Memory leaks)** 导致的延迟和浏览器崩溃等持续性问题损害了 ChatGPT 的体验，引发了关于战术性聊天会话限制和完全召回过去交互以避免重复乏味的讨论。同时，GPT-4 中备受期待的**实时语音和视觉**功能已定于以 Alpha 状态向特定群体首次亮相，并根据 [OpenAI 的更新](https://help.openai.com/en/articles/8400625-voice-chat-faq)在随后几个月内扩大范围。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**NeurIPS 竞赛：合并模型赢取荣誉与现金**：NeurIPS 将举办一场 **Model Merging 竞赛**，奖金为 8,000 美元，由 Hugging Face 和 Sakana AI Labs 赞助，旨在寻求模型选择和合并方面的创新。注册及更多信息可在 [llm-merging.github.io](https://llm-merging.github.io/) 查看，正如 [Twitter](https://x.com/LChoshen/status/1796256513519989102) 上所宣布的那样。

**AI 探索与生物对话**：一项高达 50 万美元的 **Coller Prize** 奖金正虚位以待，奖励给那些能够利用 AI 揭开与动物交流奥秘的人，这激发了人们对潜在突破的兴奋 ([信息](https://coller-dolittle-24.sites.tau.ac.il/))。这一倡议呼应了 Aza Raskin 的 Earth Species Project，旨在理清跨物种对话 ([YouTube 视频](https://www.youtube.com/watch?v=rjvsl0mhqTk))。

**困惑于偏好学习悖论**：在一条 [推文](https://x.com/_angie_chen/status/1796220428345573399) 指出 RLHF/DPO 方法中意想不到的局限性后，社区议论纷纷——偏好学习算法并不总能产生更好的首选响应排名，这挑战了传统认知，并暗示了过拟合 (Overfitting) 的可能性。

**LLMs 主导实时网页内容**：网页用户的一个新发现：**LLMs** 经常实时生成网页，在加载时渲染你所看到的内容。由于上下文限制，这种常规操作在处理冗长或内容丰富的页面时会遇到障碍，这是一个亟待战略性改进的领域。

**Google 增强 AI 驱动的搜索**：Google 为美国搜索用户升级了其 AI Overviews，提高了满意度和网页点击质量。尽管存在一些小故障，但他们正在通过反馈循环进行迭代，详见其博客文章——[AI Overviews: About last week](https://blog.google/products/search/ai-overviews-update-may-2024/)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Milvus Lite 提升了 Python 向量数据库**：**Milvus Lite** 的推出为 Python 提供了一个轻量级、高效的向量存储解决方案，兼容 LangChain 和 LlamaIndex 等 AI 开发栈。鼓励用户在资源受限的环境中将 **Milvus Lite** 集成到其 AI 应用中，说明文档见[此处](https://milvus.io/docs/milvus_lite.md)。

- **使用 Omakase 构建 Web 应用**：一个新的基于 Django 的 Web 应用模板简化了可扩展的检索增强生成 (RAG) 应用的构建，包含 RAG API、数据源管理和用户访问控制。分步指南见[此处](https://t.co/kx3DhxfDZu)。

- **导航检索系统的数据传输**：对于那些正在原型化检索系统的用户，社区建议创建一个 "IngestionPipeline" 来高效处理 SimpleStore 类和 RedisStores 之间的数据更新 (upserts) 和传输。

- **评估向量存储查询的复杂性**：澄清了 PostgreSQL 中不同向量存储查询类型（如 `DEFAULT`、`SPARSE`、`HYBRID` 和 `TEXT_SEARCH`）的功能，共识是 `text` 和 `sparse` 查询都利用了 `tsvector`。

- **解决 OpenAI 证书问题**：针对 Docker 化 OpenAI 设置中的 SSL 证书验证问题，建议探索替代的基础 Docker 镜像以潜在地解决该问题。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Luxia 语言模型污染警报**：据报道，**Luxia 21.4b v1.2** 在 GSM8k 测试中的污染比 v1.0 增加了 29%，详见 [Hugging Face 上的讨论](https://huggingface.co/saltlux/luxia-21.4b-alignment-v1.2/discussions/1)，这引发了对基准测试可靠性的担忧。

- **准备，开始，合并！：NeurIPS 模型合并对决**：[NeurIPS 2023 模型合并竞赛](https://llm-merging.github.io/) 提供了 **$8K** 的奖金，吸引 AI 工程师在模型选择和合并方面开辟新路径。

- **前沿的 CLIP 文本编码器和 PDE 求解器范式转变**：通过预训练实现的 CLIP 文本编码器方法论的进步，以及 **Poseidon**（一种用于 PDEs 的新模型，具有样本效率高且结果准确的特点）的部署获得了认可，重点介绍了关于 [Jina CLIP](http://arxiv.org/abs/2405.20204) 和 [Poseidon](https://arxiv.org/abs/2405.19101) 的论文。

- **Softmax Attention 在 Transformer 中的地位**：围绕 Transformer 中 softmax 加权路由的必要性展开了辩论，一些工程师认为长期以来的使用优于新出现的机制（如 "function attention"），后者与现有方法论仍保持相似性。

- **Gemma-2b-it 的可复现性难题**：在尝试复制 Gemma-2b-it 17.7% 的成功率时出现了差异，工程师们转向 [Hugging Face 论坛](https://huggingface.co/google/gemma-2b-it/discussions/44) 和 [Colab notebook](https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/gsm8k_eval.ipynb#scrollTo=cXoCKMi9EXir) 寻求潜在解决方案，而通过 lm_eval 得到的 Phi3-mini 结果证明与预期结果更加一致。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 崛起：包管理与编译器难题**：根据 [Discussion #413](https://github.com/modularml/mojo/discussions/413) 和 [Discussion #1785](https://github.com/modularml/mojo/discussions/1785)，**Mojo** 社区正期待关于提议的包管理器的更新。最近的 nightly Mojo 编译器版本 `2024.5.3112` 带来了修复和功能变更，详见 [raw diff](https://github.com/modularml/mojo/compare/8ae83916ebc7b3134948f466c0f56ee3e5569062...3df5fb4f9d3dd7cc5018aa8a160c3714b1a4f81e) 和当前的 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。

- **蓄势待发：社区会议与增长**：Mojo 社区期待下一次会议，届时将有关于各种主题的精彩演讲，详情见 [社区文档](https://modul.ar/community-meeting-doc)，并可通过 [Zoom 链接](https://modul.ar/community-meeting-zoom.) 参与。

- **实践出真知：Mojo 的速度与修复**：一段 YouTube 视频展示了将 K-Means 聚类移植到 Mojo 后带来的显著加速，详见[此处](https://www.youtube.com/watch?v=3bg5YBCcuWA)。在 `reversed(Dict.items())` 中发现的一个导致测试不稳定的 bug 已通过 [PR](https://github.com/modularml/mojo/pull/2896) 得到修正。

- **攻克学习曲线：编译器的教育资源**：为了学习编译器知识，有人推荐了一份详尽的教学大纲，可在[此处](https://mcyoung.xyz/syllabus)获取。

- **串联性能**：为了避免内存开销，有人提议使用更高效的 string builder，讨论倾向于零拷贝（zero-copy）优化，并结合使用 [`iovec` 和 `writev`](https://man7.org/linux/man-pages/man2/writev.2.html) 以实现更好的内存管理。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChainAPI 的公开访问**：有人请求提供一种方法，将 **LangChainAPI** 端点公开暴露给 **LangGraph** 使用场景，并表示有兴趣使用 **LangServe**，但正在等待邀请。

- **LangGraph 性能调优**：围绕优化 **LangGraph** 配置的讨论集中在减少加载时间和提高 Agent 的启动速度上，这表明用户更倾向于更高效的流程。

- **聊天应用中的 Memory 与 Prompt Engineering**：参与者寻求关于将“memory”中的摘要集成到 `ChatPromptTemplate`，以及将 `ConversationSummaryMemory` 与 `RunnableWithMessageHistory` 结合的建议。他们分享了总结聊天历史以有效管理 token 数量的策略，并提供了相关的 [GitHub 资源](https://github.com/langchain-ai/langchain/issues/16525) 和 [LangChain 文档](https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/#dict-with-single-key-for-all-messages-input-messages-output)。

- **LangServe 网站故障报告**：报告了 **LangServe 网站**上的一个错误，并分享了[网站链接](https://www.langchain.com/langserve)以获取更多详情。

- **多变量 Prompt 构建**：针对如何使用来自 **LangGraph** 状态的多个变量来构建 prompt 进行了咨询，并提供了一个公式化的 prompt 示例以及关于变量插入时机的询问。

- **社区项目与工具展示**：在社区工作领域，有两个项目备受关注：一个是关于为 Agent 创建自定义工具的 **YouTube 教程**（[Crew AI Custom Tools Basics](https://youtu.be/Hc5yiUQKh2Q)），另一个是名为 **AIQuire** 的 AI 工具，用于文档洞察，可在 [aiquire.app](https://aiquire.app) 提供反馈。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Fineweb 推动图文对齐（Image-Text Grounding）**：一种名为 **Fineweb** 的极具前景的方法利用 Puppeteer 抓取网页内容，提取带有文本的图像作为 VLM 输入，为视觉语言模型（VLM）的 Grounding 提供了新手段。[在此查看 Fineweb 讨论](https://vxtwitter.com/iurimatias/status/1796260746310910309)。

- **StabilityAI 的选择性发布引发辩论**：StabilityAI 决定仅发布 512px 模型，而未发布全套 SD3 Checkpoints，这引发了成员们关于此举如何影响未来模型改进和资源分配的讨论。

- **位置精度**：关于 **DiT models** 中的位置嵌入（Positional Embeddings）在处理更高分辨率图像时可能导致模式崩溃（Mode Collapse）的技术讨论，尽管它们目前是标准用法。

- **开源狂欢**：开源项目 **tooncrafter** 的潜力让社区感到兴奋，尽管一些小问题正在解决中，这展示了社区推动增量进步的动力。

- **Yudkowsky 的策略引发争议**：Eliezer Yudkowsky 的研究所发布了“2024 通讯策略”，倡导停止 AI 开发，在科技爱好者中引发了不同反应。[深入了解该策略](https://x.com/drtechlash/status/1796562490232557658?s=46&t=M3cR_nfDo7QCuM4xOvwNFA)。

- **在 NeurIPS 合并模型**：NeurIPS 正在举办一场模型合并（Model Merging）竞赛，奖金为 8,000 美元，旨在激励 LLMs 领域的创新。感兴趣的参与者应访问[官方 Discord](https://discord.gg/dPBHEVnV)和[注册页面](https://llm-merging.github.io/)。

- **用于美学 AI 艺术创作的 RB-Modulation**：**RB-Modulation** 方法提出了一种无需额外训练即可对图像进行风格化和构图的新方法，成员可以访问[项目页面](https://rb-modulation.github.io/)、[论文](https://arxiv.org/abs/2405.17401)以及即将发布的[代码](https://github.com/LituRout/RB-Modulation)。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Yuan2 模型引起社区关注**：成员们分享了对 Huggingface 上 [Yuan2 model](https://huggingface.co/papers/2311.15786) 的见解，强调了对其训练方面的浓厚兴趣。

- **训练技术对决**：详细讨论对比了各种偏好训练（Preference Training）方法，重点介绍了 *ORPO method*，由于其“更强的效果”，被建议取代 SFT 紧接 DPO 的方案。相关参考文献见 [ORPO 论文](https://arxiv.org/abs/2403.07691)。

- **模型微调的挑战**：针对 llama3 和 mistral 在西班牙语实体分类微调中的困难出现了担忧。一个案例详细描述了训练成功后模型推理的问题。

- **成员寻求并提供技术援助**：从关于 **Axolotl** 和 CUDA 的安装查询，到使用 **Hugging Face Accelerate library** 配置早停（Early Stopping）机制以解决过拟合问题，社区成员积极寻求并提供技术协助。共享资源包括 [axolotl 文档](https://github.com/openaccess-ai-collective/axolotl/tree/main/README.md#L102L137)和[早停配置指南](https://github.com/OpenAccess-AI-Collective/axolotl/blob/d4f6c65e4c4d6899a9283a046fa922b937a0ab25/docs/config.qmd#L325C1-L327C27)。

- **Axolotl 配置说明**：关于 Axolotl 中 `chat_template` 正确配置的建议交流，推荐由 Axolotl 自动处理以管理 LLama3 的 Alpaca 格式。



---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **DiscoLeo 陷入死循环**：将 **ChatML 引入 DiscoLeo** 导致了句子结束（EOS）Token 问题，使得模型有 20% 的概率进入无限循环。建议通过使用 ChatML 数据重新训练 DiscoLeo 来解决此问题。

- **Llama-3 模板更倾向于 ChatML**：德语微调显示，相比 **Llama-3 instruct 模板**，用户更倾向于使用 **ChatML**，特别是针对像 Hermes Theta 这样已经基于 ChatML 的模型。

- **IBM 的 “Granite” 模型引发好奇**：工程师们正在探索 IBM 的 **Granite 模型**，包括 **Lab 版本**、**基于 Starcode 的变体**以及 **Medusa 投机解码**，相关资源列在 [IBM 文档](https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x?topic=models-foundation)和 [InstructLab Medium 文章](https://medium.com/@syeda9118/instructlab-ever-imagined-the-ease-of-tuning-pre-trained-llms-3331ccea8d88)中。

- **Merlinite 7B 与 Granite 的对决**：**Merlinite 7B 模型**因其精通德语而备受关注，正被拿来与通过 **Lab 方法**追踪的 IBM Granite 模型进行对比。

- **对 AI 生成数据质量的担忧**：社区对 **AI 生成的数据**质量表示不满，例如在 **q4km gguf 量化**的 **EQ Bench** 等基准测试中表现不佳，并对在不产生灾难性遗忘（catastrophic forgetting）的情况下增强模型的新策略表现出兴趣。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Google 的扩张引起关注**：一条 [推文](https://x.com/_arohan_/status/1796228607255396423) 暗示 Google 正在加强其计算资源，引发了对其在 AI 模型训练能力方面影响的猜测。
  
- **OpenAI 重启机器人项目**：据 [Twitter](https://x.com/sarahnemerson/status/1796264432215146807?s=46) 和 Forbes 文章报道，OpenAI 重新启动了机器人业务，目前正在招聘研究工程师，这标志着其自 2020 年以来首次重返机器人领域。

- **GPT-3.5 API 的混乱局面**：社区成员对 GPT-3.5 混乱的文档和可用性说明表示沮丧；一些人指出了时间线上的差异以及删除技术文档带来的不便。

- **Sergey 为 Physical Intelligence 招兵买马**：Nathan Lambert 转达了 Sergey 为一个物理智能（Physical Intelligence）项目进行的招聘信息，为那些对强化学习（RL）感兴趣并希望为实际机器人应用做出贡献的人提供了机会。

- **“AI 政策的浑水”环节热议**：最新一期的 [Murky Waters in AI Policy](https://retortai.com/episodes/murky-waters-in-ai-policy) 播客讨论了加州备受争议的 1047 法案，并快速回顾了近期 OpenAI 和 Google 的失误。Nathan Lambert 错过了该法案的公开听证会，未提供出席细节或原因。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 开拓 AI 可持续性**：Cohere 因优先考虑**长期可持续性**而非眼前的宏大挑战而受到关注，重点关注从发票中提取信息等具体任务。
- **AGI 仍有发展空间**：社区达成共识，认为通往 AGI 的旅程才刚刚开始，并持续努力探索 AI 发展在当前 “CPU” 阶段之外的可能性。
- **Cohere 提升服务器体验**：服务器正在进行改造，以简化频道、增加新的角色和奖励，用 “cohere regulars” 取代服务器等级，并引入 **Coral AI 聊天机器人**来增强互动。
- **使用自定义表情符号表达自我**：为了增加趣味性并改善互动，服务器将加入新的表情符号，用户可以通过联系管理员进行自定义。
- **征求对无代码 AI 工作流的反馈**：一家初创公司正在征求对其 AI 模型**无代码工作流构建器**的见解，并提供 **10 美元的调查奖励**——他们很好奇为什么用户在首次使用后可能不再返回。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**适配层弥合差距**：工程师们正在探索 **embedding adapters**（嵌入适配器）作为提高 AI 模型检索性能的一种手段，[Chroma 研究报告](https://research.trychroma.com/embedding-adapters)中展示了相关证据。这些适配器的有效性可以类比为 **Frozen Embeddings**，Vespa 团队利用它来消除动态系统中的频繁更新（[Vespa 博客见解](https://blog.vespa.ai/leveraging-frozen-embeddings-in-vespa-with-sentence-transformers/)）。

**ChatGPT 与 PwC 合作进入企业市场**：PwC 为约 100,000 名员工购买了 **ChatGPT Enterprise** 许可证，这引发了围绕每年约 3000 万美元估值的讨论，成员们对每位用户的成本猜测从每月 8 美元到 65 美元不等。

**Google 的双星：Gemini 1.5 Flash & Pro**：Google **Gemini 1.5 Flash** 和 **Pro** 的发布更新已推向正式版（GA），引入了诸如提高 **RPM** 限制和 **JSON Schema** 模式等增强功能（[Google 开发者博客文章](https://developers.googleblog.com/en/gemini-15-pro-and-15-flash-now-available/)）。

**TLBrowse 加入开源宇宙**：融合了 **Websim** 与 **TLDraw** 的 **TLBrowse** 已开源，允许用户在 **@tldraw** 画布上幻化出无限想象的网站，并提供[免费托管版本](https://tlbrowse.com)访问。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **浏览器中的文学世界**：**Rosebud AI** 正在筹备其“从书到游戏”（Book to Game）游戏创作节（game jam），邀请参与者使用 **Phaser JS** 根据文学作品构建游戏。该活动提供 **500 美元奖金**，持续至 7 月 1 日，详情可通过其 [Twitter](https://x.com/Rosebud_AI/status/1796273820044595368) 和 [Discord](https://discord.gg/rosebud-ai) 获取。

- **导航数字领域**：一位新公会成员表示在 Android 上使用该平台存在困难，称体验“不稳定且有 Bug”。他们还寻求帮助更改用户名，以便在虚拟空间中更有归属感。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **查看置顶消息获取制造更新**：及时掌握 **OpenInterpreter** 社区的**制造更新**至关重要；请务必查看 **#general** 频道中的置顶消息以获取最新信息。
- **Codestral 模型引发好奇**：工程师们对 **Codestral** 模型表现出兴趣，多个频道都出现了关于其效率的咨询；然而，用户体验尚未被分享。此外，值得注意的是 **Codestral** 仅限于非商业用途。
- **应对集成挑战**：将 **HuggingFace** 模型与 **OpenInterpreter** 集成是一个共同的挑战，使用 `interpreter -y` 命令的成功率有限。建议面临这些问题的工程师在技术支持频道寻求建议。
- **发布诈骗警报**：保持警惕至关重要，因为社区内发布了关于潜在诈骗的“红色警报”。目前尚未提供有关该诈骗的更多细节。
- **Android 功能讨论进行中**：成员们正在讨论 **O1 Android** 的功能，特别是关于在 **Termux** 中的安装，尽管目前尚未看到结论性的回复。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile 与 AutoGPT 联手**：**AutoGPT** 成员宣布了一项**协作**，将 **Llamafile** 编织进他们的系统中，扩展了该工具的覆盖范围和能力。
- **关于内容块的查询**：有询问关于 **Llamafile** 是否可以处理消息中的**内容块**（content blocks），以寻求与类似 OpenAI 功能的对等；对于 **llama.cpp** 在该领域的能力也寻求了类似的澄清。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Netflix PRS 活动聚集 AI 爱好者**：AI 专业人士正热烈讨论在 **Netflix 举办的 PRS 活动**，多位社区成员已确认参加，进行社交和讨论。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Mistral 45GB 模型构成推测**：围绕 **Mistral 45GB 模型**的语言分布，兴趣正在酝酿，一种假设认为该模型强烈偏向英语，而编程语言的占比相对较小。

- **Codestral 合规难题**：社区正在研究 **Mistral AI Non-Production License (MNPL)** 的复杂性，发现其对分享衍生作品或托管作品的限制不尽如人意，限制了 **Codestral** 的开发。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **TensorFlow 与 PyTorch 的辩论仍在继续**：一位名为 helplesness 的用户询问为什么 **TensorFlow** 可能被认为优于 **PyTorch**，引发了社区内的对比讨论。讨论并未给出定论，但反映了 AI 工程界对框架选择的持续偏好之争。



---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---


**YAIG (a16z Infra) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。


---

# 第二部分：按频道详细摘要与链接


{% if medium == 'web' %}




### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1245816528025288819)** (86 条消息🔥🔥): 

- **请求关于 Transformer 内部机制的嘉宾分享会**：一名成员请求讨论 Transformer 架构相关的主题，如 vanilla transformer、RoPE、RMS Normalization 等。虽然分享了这些主题的视频资源，但该成员强调需要互动环节进行问答。
  
- **Google 的 Gemini Flash 将支持微调**：从 6 月 17 日开始，Google 将允许在 Gemini Flash 上进行免费微调，推理成本与基础模型费率一致。这被强调为微调的一个极具成本效益的机会。

- **生产级系统的成本管理**：关于计算使用 RAG LLMs 的生产系统成本进行了交流，重点关注 GPU 时间利用率和第三方服务的权衡。讨论强调了在计算平台上进行实验以及根据使用场景管理预期的重要性。

- **用于微调模型的 GGUF 格式**：推荐使用 GGUF 格式进行 LLMs 微调，以确保与生态系统中的各种工具兼容。分享了一个关于微调和推理步骤的详细博客链接，并提到 Hugging Face 正在开发更简便的 HF 到 GGUF 转换工具。

- **Document AI 进展**：多位用户讨论了他们在处理文档（如发票和公用事业账单）时的经验和挑战。分享了 OCR、LiLT 模型、分割以及使用多模态/非多模态方法提取结构化信息的技术，并提供了相关资源和论文链接。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ankur-singh.github.io/blog/finetune-inference">使用 Ollama 运行微调后的 LLM</a>：完整的流程演示，展示如何根据你的数据微调 LLM 并使用 Ollama 运行它。</li><li><a href="https://arxiv.org/abs/2401.00908">DocLLM：一种用于多模态文档理解的布局感知生成式语言模型</a>：企业文档（如表单、发票、收据、报告、合同等）通常在文本和空间模态的交汇处承载着丰富的语义。视觉线索...</li><li><a href="https://huggingface.co/blog/idefics2">推出 Idefics2：一个面向社区的强大 8B 视觉语言模型</a>：未找到描述</li><li><a href="https://youtu.be/hDa-M91MSGU?si=4QKcNZsB40ibgPyd">为图像转 JSON 场景微调 PaliGemma</a>：在本教程中，我将展示如何在收据图像转 JSON 的用例中微调 PaliGemma，这是 Google 推出的一款新型开放视觉语言模型。目标是...</li><li><a href="https://www.cbc.ca/news/canada/manitoba/facebook-customer-support-scam-1.7219581">温尼伯男子在 AI 告知其虚假 Facebook 客服电话为真实号码后遭遇诈骗 | CBC 新闻</a>：一名温尼伯男子表示，他在拨打了一个他认为是 Facebook 客户支持热线的号码后被骗走了数百美元，他希望以此提醒他人可能发生的风险。</li><li><a href="https://huggingface.spaces/openbmb/MiniCPM-Llama3-V-2_5">MiniCPM-Llama3-V-2 5 - openbmb 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://youtu.be/Mn_9W1nCFLo?si=SWUPvbQ9ZCAxmAK_">LLaMA 详解：KV-Cache、Rotary Positional Embedding、RMS Norm、Grouped Query Attention、SwiGLU</a>：全面解析 Meta 的 LLaMA 1 和 LLaMA 2 模型，包括 Rotary Positional Embeddings、RMS Normalization、Multi-Query Attention、KV-Cache、Grou...</li><li><a href="https://youtu.be/UiX8K-xBUpE?si=UgGM6oimKVhvub-b">Mistral / Mixtral 详解：Sliding Window Attention、Sparse Mixture of Experts、Rolling Buffer</a>：在本视频中，我将介绍 Mistral 7B 和 Mixtral 8x7B 模型中的所有创新：Sliding Window Attention、带有 Rolling Buffer 的 KV-Cache、Pre...</li><li><a href="https://youtu.be/bCz4OMemCcA?si=X5lnwL_cmE16XFFS">Attention is all you need (Transformer) - 模型详解（含数学原理）、推理与训练</a>：完整解释 Transformer 模型的所有层：Multi-Head Self-Attention、Positional Encoding，包括所有的矩阵乘法和...</li><li><a href="https://github.com/huggingface/transformers/pull/30928">FEAT / Trainer：实验性功能 - 在将模型推送到 Hub 时添加 `GGUF` 转换，由 younesbelkada 提交 · Pull Request #30928 · huggingface/transformers</a>：此 PR 的作用？引入了一个新的 quantization_config，旨在仅用于 trainer.push_to_hub()，它在后台调用一个 GGUF 转换 Space -（目前为：https://huggingf...</li><li><a href="https://arxiv.org/abs/2405.20245">检索增强结构化生成：作为工具调用的商业文档信息提取</a>：商业文档信息提取 (BDIE) 是将非结构化信息块（原始文本、扫描文档等）转换为下游系统可以处理的结构化格式的问题...
</li>
</ul>

</div>
  

---


### **LLM 微调 (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1245926979329458237)** (10 条消息🔥): 

- **用户提供法律草案系统方面的帮助**：一名成员表示愿意协助另一名用户开发其“法律草案系统”。未提供有关该系统或所需协助的更多细节。
- **类 Alexa 音乐播放器提案**：一名成员询问微调是否适用于类似于 Alexa 但不局限于 Amazon Music 的产品。他们建议使用 **LangGraph + function calling** 来与 YouTube 和 Spotify 等各种音乐服务 API 进行交互。
- **用于财务文档摘要的聊天机器人**：一名成员概述了一个开发聊天机器人的项目，该机器人能够通过总结财务文档来回答复杂的财务问题。他们指出必须使用 **RAG** 和某种形式的 **PO** 来生成符合用户偏好的摘要。
- **为 Cypher/GQL 翻译微调 LLM**：一位用户打算微调 LLM，将自然语言问题翻译成 Cypher/GQL。他们指出，这可以大大增强与图数据的交互。
- **关于 instruct-LLM 与 chat-LLM 的讨论**：一场广泛的讨论辩论了这些模型之间的区别，重点在于训练和评估的差异。用户指出，虽然较新的模型模糊了这些界限，但 **instruct 模型** 遵循清晰的指令，而 **chat 模型** 则处理对话上下文。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/)** (1 条消息): 

blaine.wishart: 大家好...接下来的 3 个月我会在海南。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1245842938412662884)** (18 条消息🔥): 

- **已部署模型的后续步骤**：在部署了 v1 Finetuning 模型后，一位成员寻求关于如何进一步使用它的指导。建议参考[这份文档](https://modal.com/docs/guide/trigger-deployed-functions)以了解如何在 Modal 平台上使用和调用已部署的函数。
- **Modal 额度问题已解决**：处理了关于 Modal 额度的各种问题和咨询，例如一位用户注意到额度丢失，另一位询问过期时间。Modal 额度有效期为一年，但活跃用户可以联系支持团队尝试申请额度结转，进行学术研究或参与初创公司的用户可以获得额外的[额度](https://tally.so/r/wQ5GRl)。
- **排查数据集问题**：一位成员在处理特定的训练数据文件时遇到了 "KeyError: 'Input'"。建议检查数据集格式的一致性，并确保正确的字段键与 [config](https://github.com/modal-labs/llm-finetuning/blob/f64c8d7ea5ac46f67251801b05d52b933228db50/config/codellama.yml#L17-L20) 中定义的相匹配。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tally.so/r/wQ5GRl">面向初创公司和学术界的 Modal</a>：由 Tally 制作，最简单的表单创建方式。</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/f64c8d7ea5ac46f67251801b05d52b933228db50/config/codellama.yml#L17-L20">llm-finetuning/config/codellama.yml (位于 f64c8d7ea5ac46f67251801b05d52b933228db50) · modal-labs/llm-finetuning</a>：Llama/Mistral/CodeLlama 等模型的微调指南 - modal-labs/llm-finetuning</li><li><a href="https://modal.com/docs/guide/trigger-deployed-functions">调用已部署的函数</a>：Modal 允许你获取由部署创建的函数，并从其他上下文中调用它。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1245837657582600243)** (9 messages🔥): 

- **AI-Coding Humble Bundle 提醒**：一位成员在频道中分享了一个 AI 编程 [humble bundle](https://www.humblebundle.com/software/complete-chatgpt-anthropic-gemini-prompt-engineering-api-and-programming-mega-bundle-software?mcID=102:66576d20c5895a1aa5046052:ot:5ccaf0c3db76615eab12deb2:1&linkID=66576d225fb588c450040093&utm_campaign=2024_05_30_completechatgptanthropicgeminipromptengineeringapiandprogramming_softwarebundle&utm_source=Humble+Bundle+Newsletter&utm_medium=email)，但对内容质量表示怀疑，指出 *“初步浏览看起来不太好。”* 另一位成员补充说，独立生成此类材料可能成本更低。
  
- **Sebastian Raschka 关于 LLM 微调的章节**：分享了 Sebastian Raschka 即将出版的新书中关于为分类任务微调 LLM 的 [章节](https://livebook.manning.com/book/build-a-large-language-model-from-scratch/chapter-6/v-7/10) 链接，概述了不同的微调方法、数据集准备以及垃圾邮件分类的准确性评估等主题。

- **O'Reilly 发布构建 LLM 的第二部分**：在第一部分获得积极反馈后，O'Reilly 快速发布了关于使用 LLM 构建应用的系列文章 [第二部分](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii/)，重点从构建 LLM 应用的战术层面转向运营层面，并指出了值得解决的挑战。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii/">What We Learned from a Year of Building with LLMs (Part II)</a>：未找到描述</li><li><a href="https://livebook.manning.com/book/build-a-large-language-model-from-scratch/chapter-6/v-7/10">6 Finetuning for Classification · Build a Large Language Model (From Scratch)</a>：介绍不同的 LLM 微调方法 · 为文本分类准备数据集 · 修改预训练 LLM 以进行微调 · 微调 LLM 以识别垃圾邮件 · 评估...</li><li><a href="https://www.humblebundle.com/software/complete-chatgpt-anthropic-gemini-prompt-engineering-api-and-programming-mega-bundle-software?mcID=102:66576d20c5895a1aa5046052:ot:5ccaf0c3db76615eab12deb2:1&linkID=66576d225fb588c450040093&utm_campaign=2024_05_30_completechatgptanthropicgeminipromptengineeringapiandprogramming_softwarebundle&utm_source=Humble+Bundle+Newsletter&utm_medium=email">The Complete ChatGPT, Anthropic, Gemini Prompt Engineering, API, and Programming Mega Bundle</a>：AI 正在兴起——通过这些在线课程与它共同成长！学习 Prompt Engineering、LangChain 等！您的购买将帮助 Children’s Miracle Network。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1245844758509912105)** (4 messages): 

- **Langsmith-HIPAA 兼容性咨询**：一位成员询问 Langsmith 是否提供支持 HIPAA 环境的付费方案，提到需要安全处理 PII/PHI，并且必须签署商业伙伴协议 (BAA)。
  
- **Langsmith 与 OpenAI 模型的兼容性**：另一位用户询问 Langsmith 是否可以与 Mixtral 等 OpenAI 兼容模型，或任何遵循相同 API 标准的模型（如 Anthropic）一起使用。

- **Langsmith 连接各种模型**：Lucas 分享了通过 Ollama 使用 Langchain 和 Langsmith 配合 Meta 的 Llama-3:8b 的见解，并强调了 Langchain 与 Together AI 的集成。使用 Together AI 的详细步骤和代码片段可以在 Lucas 的 [博客文章](https://lucasvw.github.io/posts/20_ollama_langchain/) 中找到。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/integrations/llms/together/">Together AI | 🦜️🔗 LangChain</a>：Together AI 提供了一个 API，只需几行代码即可查询 50 多个领先的开源模型。</li><li><a href="https://lucasvw.github.io/posts/20_ollama_langchain/">Lucas van Walstijn - Having fun with llama3:8b</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[kylecorbitt_prompt_to_model](https://discord.com/channels/1238365980128706560/1242221891733946490/1245869118108729494)** (1 messages): 

- **寻求处理 LinkedIn 个人资料数据的建议**：一位用户询问处理 5000 个抓取的 LinkedIn 个人资料（包含 20 多个列）的最佳方法。他们的目标是构建一个微调模型，先使用 OpenPipe 配合 GPT-4 生成个性化开场白，随后再微调一个 llama-3-8b 模型。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/1245889210288836649)** (3 条消息): 

- **演讲幻灯片现已开放**：一位成员询问最近一次演讲的幻灯片是否可用。另一位成员提供了一个 [Discord 链接](https://discord.com/channels/1238365980128706560/1242223275463938221/1245439203207286825)，其中也包含了幻灯片的链接。
  
- **分享富有见地的 Prompt 编写资源**：一位成员推荐了 [ExplainPrompt](https://www.explainprompt.com/) 作为一个有价值的资源。该网站由一位 GitHub 的同事维护，他会根据最新的论文发布关于 Prompt 编写技巧的总结和视觉指南。

**提及的链接**：<a href="https://www.explainprompt.com/">ExplainPrompt</a>：未找到描述

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[whitaker_napkin_math](https://discord.com/channels/1238365980128706560/1242223332695478332/1246108171018440774)** (268 条消息🔥🔥): 

- **在 Johno 的博客上开启你的 LLM 冒险**：成员们兴奋地分享了 John Whitaker 的宝贵内容。查看他的 [博客](https://johnowhitaker.dev)，其中包含诸如 *More=Better?* 等富有见地的文章、*Basement Hydroponics* 等小型硬件项目，以及关于高表面积问题的技巧。

- **通过配置优化避免四轴飞行器崩溃**：发布了诸如 [fsdp_qlora benchmarks](https://github.com/AnswerDotAI/fsdp_qlora/blob/main/benchmarks_03_2024.md) 和 [does LoRA cause memory leaks](https://github.com/huggingface/transformers/issues/25572) 等 GitHub 链接。这些参考资料增强了关于 LLM 训练、内存泄漏问题及其实际解决方案的知识。 

- **Johno 分享 LoRA 见解**：讨论包括围绕 LoRA 功能的实用技巧，例如用较小的矩阵近似大矩阵，以及对 LoRA rank (N x r) 的考虑。对于那些在优化资源效率的同时进行模型微调的人来说非常有用。

- **草图大师重新定义简单**：Johno 清晰且有效的教学风格吸引了与会者，导致大家呼吁进行更多深入的课程。一位成员指出：“他非常擅长教学和解释事物”，并敦促提供更多向他学习的机会。

- **释放梯度技巧的力量**：成员们分享了诸如 [gradient checkpointing](https://discord.com/channels/1238365980128706560/1242223332695478332/1246108353210355805) 和拆分梯度计算等高级技术，以优化内存和速度。Twitter、GitHub 和 Google Docs 的超链接被广泛传播，以便进一步阅读和探索。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/johnowhitaker?lang=en">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://johnowhitaker.dev/">johnowhitaker.dev – Jonathan Whitaker</a>：未找到描述</li><li><a href="https://www.gradio.app/custom-components/gallery?id=radames%2Fgradio_huggingfacehub_search">Gradio 自定义组件库</a>：搜索自定义组件库。</li><li><a href="https://muellerzr.github.io/blog/gradient_accumulation.html">Zach Mueller - PyTorch, Gradient Accumulation 以及可怕的速度下降</a>：未找到描述</li><li><a href="https://sakana.ai/blog/">Sakana AI</a>：Sakana AI 博客</li><li><a href="https://blog.eleuther.ai/transformer-math/">Transformer 数学基础 101</a>：我们介绍了与 Transformer 计算和内存使用相关的基础数学。</li><li><a href="https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-2bitgs8-metaoffload-HQQ">mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-2bitgs8-metaoffload-HQQ · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/gabriberton/status/1796531585958985941">来自 Gabriele Berton (@gabriberton) 的推文</a>：这个简单的 PyTorch 技巧将使你的 GPU 显存占用减半 / Batch Size 翻倍（真的）。与其累加 Loss 然后计算 Backward，不如在每个...上计算 Backward。</li><li><a href="https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39?permalink_comment_id=3417135">加速 PyTorch 数据加载的技巧</a>：加速 PyTorch 数据加载的技巧。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://x.com/karpathy/status/1325154823856033793">来自 Andrej Karpathy (@karpathy) 的推文</a>：如何成为某方面的专家：1. 迭代地承担具体项目并深入完成它们，按需学习（即不要自下而上地泛泛学习） 2. 教授/总结你所学到的一切...</li><li><a href="https://ai.google.dev/gemini-api/docs/caching">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/docs/peft/en/package_reference/lora">LoRA</a>：未找到描述</li><li><a href="https://x.com/SakanaAILabs/status/1770613032198279663">来自 Sakana AI (@SakanaAILabs) 的推文</a>：介绍进化模型合并（Evolutionary Model Merge）：一种让我们更接近自动化基础模型开发的新方法。我们利用进化算法寻找组合开源模型的绝佳方式，构建新的强大...</li><li><a href="https://github.com/angie-chen55/pref-learning-ranking-acc/blob/main/pref_learning_algs_do_not_learn_pref_rankings.pdf">pref-learning-ranking-acc/pref_learning_algs_do_not_learn_pref_rankings.pdf (main 分支) · angie-chen55/pref-learning-ranking-acc</a>：通过在 GitHub 上创建账号，为 angie-chen55/pref-learning-ranking-acc 的开发做出贡献。</li><li><a href="https://github.com/AnswerDotAI/fsdp_qlora/blob/main/benchmarks_03_2024.md">fsdp_qlora/benchmarks_03_2024.md (main 分支) · AnswerDotAI/fsdp_qlora</a>：使用 QLoRA + FSDP 训练 LLM。通过在 GitHub 上创建账号，为 AnswerDotAI/fsdp_qlora 的开发做出贡献。</li><li><a href="https://github.com/mobiusml/hqq">GitHub - mobiusml/hqq: Half-Quadratic Quantization (HQQ) 官方实现</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://www.undermind.ai/home/">Undermind 深度科学搜索</a>：Undermind 是一款 AI 驱动的搜索助手，能够理解你的复杂问题。它会仔细探索科学文献，无论问题多么复杂，都能找到你所需的内容。</li><li><a href="https://www.ai21.com/jamba">介绍 Jamba</a>：一个突破性的 SSM-Transformer 开源模型</li><li><a href="https://nvidia.custhelp.com/app/answers/detail/a_id/5490">NVIDIA 支持</a>：未找到描述</li><li><a href="https://johnowhitaker.dev">johnowhitaker.dev – Jonathan Whitaker</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/issues/25572#issuecomment-1687749561">LoRA 是否导致了 Transformers 中的内存泄漏？· Issue #25572 · huggingface/transformers</a>：系统信息：问题在多个 PEFT 版本（0.3, 0.4, 0.5 dev）中持续存在。我使用的 Accelerator 版本是 0.21.0，尝试过的 PyTorch 版本有 1.13.0 和 2.0，两者都出现了同样的内存爆炸...</li><li><a href="https://docs.google.com/presentation/d/1Ye_6zeatCWkq-fx8A--yK34uwU8oC2YQtMSTV1DgkSI/edit?usp=sharing">微调的草稿纸计算（Napkin Math）</a>：微调的草稿纸计算，Jonathan Whitaker @johnowhitaker</li><li><a href="https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html">如何通过将优化器步骤融合到反向传播中来节省内存 — PyTorch 教程 2.3.0+cu121 文档</a>：未找到描述
</li>
</ul>

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1245869242847465492)** (7 条消息): 

- **构建合规性内部工具**：一位成员正在开发一个内部工具，将诸如 "CloudTrail should have encryption at-rest enabled"（CloudTrail 应启用静态加密）之类的输入转换为多个文件，包括符合特定公司 Schema 的 rego 文件。他们正在评估系统 66% 的准确率是否归因于检索方法，并考虑通过对模型进行 Fine-tuning 来改进 Schema 和代码逻辑。

- **规则检索与准确性的挑战**：该工具目前检索整个文档作为上下文，这可能会让模型过载。存在的问题包括代码编译错误、代码不完整、幻觉（hallucinations）以及逻辑错误，并正在考虑 Fine-tuning 是否能提高其对 Schema 的遵循度。

- **西班牙语实体的文本分类**：一位成员正在优化一个模型，将西班牙语文本实体分类为人员、公司或工会，但在推理（inference）过程中表现不佳。他们概述了用于分类的多步指令，并寻求关于提高模型准确性的建议。

- **在 Fine-tuning 中保持模板对齐**：针对多轮对话（multi-turn chat）应用，讨论了在 Fine-tuning 模型时遵循官方 Chat Template 是否至关重要，以便在不从头开始的情况下保留通用效用。一位成员假设对齐是有益的，但正在寻求社区的确认。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[abhishek_autotrain_llms](https://discord.com/channels/1238365980128706560/1242223673566433431/1246146244628189286)** (57 条消息🔥🔥): 

- **AutoTrain 简化 AI 模型创建**：一位成员分享了 [AutoTrain](https://huggingface.co/autotrain) 的链接，强调其无需代码即可创建强大 AI 模型的用户友好方法。AutoTrain 处理包括 LLM Finetuning、文本和图像分类在内的各种任务，并与 Hugging Face Hub 集成以实现轻松部署。
  
- **关于 ORPO 和 SimPO 的澄清**：讨论围绕 ORPO 展开，它被描述为类似于 DPO 但不需要参考模型的“优势比偏好优化”（odds ratio preference optimization）；以及 SimPO，参与者注意到它虽然非常新且可能存在过度炒作，但仍具有前景。

- **缺少 Nvidia GPU 的挑战**：成员们讨论了在没有 Nvidia GPU 的情况下训练 AI 模型的不切实际，感叹 CPU 的性能缓慢以及 AI 库对其他品牌 GPU 缺乏支持。

- **数据集与 Optimizer 查询**：参与者请求更多关于为 RAG 设置数据集以及为 AutoTrain 自定义 Optimizer 函数的细节，建议在 Zoom Q&A 中提出这些问题以获得详细解答。

- **感谢与额外资源**：会议结束时，多位用户对 Abhishek 关于 AutoTrain 的演讲表示感谢，并分享了额外资源，包括 AutoTrain Advanced 的 [GitHub repo](https://github.com/huggingface/autotrain-advanced) 和各种配置指南。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/autotrain">AutoTrain – Hugging Face</a>：未找到描述</li><li><a href="https://github.com/huggingface/autotrain-advanced/blob/main/docs/source/llm_finetuning_params.mdx">autotrain-advanced/docs/source/llm_finetuning_params.mdx at main · huggingface/autotrain-advanced</a>：🤗 AutoTrain Advanced。通过在 GitHub 上创建账号为 huggingface/autotrain-advanced 的开发做出贡献。</li><li><a href="https://huggingface.co/docs/autotrain/index">什么是 AutoTrain Advanced？</a>：未找到描述</li><li><a href="https://github.com/huggingface/autotrain-advanced">GitHub - huggingface/autotrain-advanced: 🤗 AutoTrain Advanced</a>：🤗 AutoTrain Advanced。通过在 GitHub 上创建账号为 huggingface/autotrain-advanced 的开发做出贡献。</li><li><a href="https://github.com/huggingface/autotrain-advanced/tree/main/configs">autotrain-advanced/configs at main · huggingface/autotrain-advanced</a>：🤗 AutoTrain Advanced。通过在 GitHub 上创建账号为 huggingface/autotrain-advanced 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1245957846873538593)** (3 messages): 

- **对单向量 Embedding 的爱恨交织**：虽然单向量 Embedding 在原型设计（prototyping）中非常有用，但它们还不足以构建完整的检索流水线（retrieval pipeline）。ColBERT 在域外（out-of-domain）任务中超越了单向量方法，这归功于其 Token 级编码，为 OOD 场景提供了更丰富的信息（[关于 ColBERT 的 Vespa 博客](https://blog.vespa.ai/announcing-colbert-embedder-in-vespa/)）。

- **简要提及稀疏 Embedding 和 M3**：讨论将简要涉及稀疏 Embedding 和 M3，重点关注单向量 Embedding 在检索流水线中的优势和局限性。

- **ColBERT 的详细输出**：与将所有内容聚合到每个文档一个 1024 维向量的单向量方法不同，ColBERT 会产生大量的 128 维向量（每个 Token 一个），从而产生更高维度的输出，以进行更详细的信息处理。例如，500 个文档，每个文档 300 个 Token，将产生 `500,300,128` 的输出。

**提到的链接**：<a href="https://blog.vespa.ai/announcing-colbert-embedder-in-vespa/">Announcing the Vespa ColBERT embedder</a>：宣布在 Vespa 中推出原生 Vespa ColBERT embedder，利用 Token 级向量表示实现可解释的语义搜索。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1246162122438869032)** (1 messages): 

- **寻找可靠的 BM25 实现**：一位用户正在寻找一种 **BM25 排序**方法来与向量检索混合使用，并提到了 Python 包 `rank_bm25`。他们对该包在创建词汇表时不使用 sklearn 的分词器/向量化器，也不处理 n-grams、停用词或词干提取（stemming）感到惊讶，并询问其他人正在使用什么。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[gradio](https://discord.com/channels/1238365980128706560/1242283474300174346/1245867317041303684)** (6 messages): 

- **Mitch 深入研究 Gradio 微调**：一位用户分享了他们在 [GitHub 上的 Gradio 微调项目](https://github.com/mitch-at-orika/gradio-fine-tuning)，旨在生成高质量数据集以利用 Gradio 的最新功能。他们提到遵循早期课程的建议，通过贡献开源项目来学习。

- **对问题真实性的担忧**：另一位成员指出，数据集中 LLM 生成的问题可能无法反映真实的用户查询。他们建议参考更具体的问题，并提供了一个[具体示例](https://github.com/mitch-at-orika/gradio-fine-tuning/blob/246c34368a72b0a286d3a9fd65d9a882439d4923/datasets/finetune_data.jsonl#L39C139-L39C238)，同时在 Prompt 中添加 few-shot 示例。

- **尝试 RAG**：一位用户承认在深入微调之前没有尝试过检索增强生成（RAG），并意识到一个强大的 Prompt 有时会优于微调的效果。他们正在考虑将 RAG 集成到工作流中，以增强问答生成。

- **数据生成的价值**：成员们就微调项目中数据生成和收集的重要性交换了意见。有人指出这个过程是“秘方（secret sauce）”，并对这个 Gradio 微调项目的进展和潜力感到兴奋。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mitch-at-orika/gradio-fine-tuning">GitHub - mitch-at-orika/gradio-fine-tuning</a>：为 Gradio 库生成高质量的微调数据集。该数据集旨在帮助用户利用 Gradio 的最新功能和方法。</li><li><a href="https://github.com/mitch-at-orika/gradio-fine-tuning/blob/246c34368a72b0a286d3a9fd65d9a882439d4923/datasets/finetune_data.jsonl#L39C139-L39C238">gradio-fine-tuning/datasets/finetune_data.jsonl</a>：Gradio 微调数据集的具体文件路径示例。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1245903574492254339)** (7 messages): 

- **Axolotl 训练期间未向 WandB 记录日志**：一位用户报告称，在本地运行时，初始训练指标会记录到 WandB，但随着训练的进行，没有任何更新。他们怀疑这可能与配置文件中的更改有关，并提到：*"步数在第 2 步后重启……在第一个 step 0 之后报告的指标是我唯一看到的记录指标。"*
  
- **在 Axolotl CLI 中覆盖数据集**：一位用户询问在命令行调用 `axolotl.cli.train` 时，是否可以覆盖数据集（路径和类型）。该消息线程中未提供解决方案。
  
- **Apple M3 Max 上的 Axolotl 安装问题**：一位使用 Apple M3 Max 的用户报告了运行 `pip3 install -e '.[flash-attn,deepspeed]'` 时的错误。他们还发布了错误截图，但尚未收到任何回复。
  
- **创建指令风格的提示词模板**：一位用户寻求帮助，希望在 Axolotl 中设置提示词模板，以便使用其数据集中的 system message 而不是 preamble。他们提到在这个问题上困扰了几个小时，并寻求建议。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1245969978654724159)** (9 messages🔥): 

- **使用 Accelerate 加载大型分片（Shards）非常痛苦**：一位成员询问如何加快“使用 Accelerate 加载分片”的速度，因为“70B 模型需要相当长的时间”。另一位成员开玩笑地建议换一个“更快的硬盘”，并对即将发布的权重“接近 1TB”的 Llama 400B 提出了警告。

- **Unsloth 与 Accelerate 分片加载对比**：成员们讨论了 *Unsloth* 如何保存 4bit 模型并将其加载为 6 个分片，建议对 *Accelerate* 采用类似的方法。然而，有人指出加载时间的延迟可能与量化（Quantization）有关，而不仅仅是硬盘速度或磁盘读取时间。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1245875274810392697)** (6 messages): 

- **分享 Qwen Tokenizer 调试心得**：经过广泛的调试以及与 Qwen 团队的沟通，确定 **PreTrainedTokenizer** 是正确的，而使用 **Qwen2Tokenizer** 可能会导致问题。此问题源于 **LLamaFactory** 和 **Axolotl** 处理 `get_train_dataloader` 调用方式的差异（[Huggingface transformers trainer](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L880)；[Axolotl trainer builder](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/core/trainer_builder.py#L414)）。

- **在 Axolotl 中调整提示词风格**：一位成员询问如何为 Axolotl 中的 **Alpaca 格式** 设置不同的提示词风格。另一位成员建议使用 `chat_template: chatml` 配置，根据数据集要求更改提示词格式（[Axolotl prompters](https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2/src/axolotl/prompters.py#L47)）。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2/src/axolotl/prompters.py#L47">axolotl/src/axolotl/prompters.py at 8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2 · OpenAccess-AI-Collective/axolotl</a>：欢迎提出 Axolotl 问题。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L880">transformers/src/transformers/trainer.py at main · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的前沿机器学习库。 - huggingface/transformers</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/core/trainer_builder.py#L414">axolotl/src/axolotl/core/trainer_builder.py at main · OpenAccess-AI-Collective/axolotl</a>：欢迎提出 Axolotl 问题。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1246162272834031697)** (2 条消息): 

- **轻松部署 Gradio RAG 应用**：一位用户询问了部署基于 Gradio 的 RAG 应用简单 Python 脚本的最简便方法，以便一小部分用户进行测试。另一位用户建议使用 **HF Spaces** 进行部署。
- **关于 "share=true" 功能的疑虑**：同一位用户对在 launch 方法中使用 *"share=true"* 是否会将他们的代码发送并存储在 Gradio 服务器上表示好奇。在提供的消息中，该查询没有得到进一步的回复。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1245817913894502492)** (12 条消息🔥): 

- **Charles Frye 赞扬社区支持**：Charles 感谢了发布活动链接的成员，并特别点名了特定用户。*"非常感谢在聊天中发布我提到的链接的朋友们！你们是最棒的。"*
- **期待与 OLAP 导向**：Charles 对未来的讨论表示兴奋，并指出他们的系统 *"更倾向于读密集型的 OLAP，而不是写密集型的 OLTP。"* 
- **Office Hours 录像：Modal 与 Charles Frye 的录像缺失**：一位用户询问了活动的录像情况，指出课程页面上仍显示“加入活动”链接。其他成员确认了该问题，Dan 修复了它并表示，*"现在应该已经修复了。"*
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[langchain-langsmith](https://discord.com/channels/1238365980128706560/1242564256914870384/1245828732384448623)** (70 条消息🔥🔥): 

- **LangChain 工具揭秘**：关于 LangChain, LangSmith, LangGraph, LangFlow 和 LangServe 之间区别的讨论揭示了：**LangChain** 是一个使用 LLM 开发应用程序的框架，**LangSmith** 用于检查和优化 chain，而 **LangServe** 则将任何 chain 转换为 API。LangFlow 和 LangGraph 的用法与该框架的联系较为模糊 ([LangChain 简介](https://python.langchain.com/v0.2/docs/introduction/))。

- **LangServe 备受赞誉，LangFlow 无直接关联**：**LangServe** 被强调为将 chain 转换为 API 的首选工具。几位用户澄清说，**LangFlow** 与 LangChain 套件没有直接关系，但使用了 LangChain 框架。

- **基础设施与部署讨论**：用户对 **LangServe** 内部更细粒度的控制表现出兴趣，并对其 API 文档表达了不满。此外，讨论还涉及利用 OpenAI 的 batch API 进行合成数据生成，以及 GPU 优化和 fine-tuning 算法所需的全面学习。

- **Generative UI 热潮**：成员们讨论了像 **GenUI** 这样旨在提高消费者对 AI 理解的新进展，重点关注了来自 **W&B CVP** 的 generative UI 示例以及一个 [Generative UI GitHub 模板](https://github.com/langchain-ai/langchain-nextjs-template/blob/main/app/generative_ui/README.md)。

- **关于 LangChain & LangSmith 的博文**：一位用户分享了一篇 [博文](https://lucasvw.github.io/posts/20_ollama_langchain/)，详细介绍了他们在 Ollama 和 Jarvislabs 上结合 LLama3:8b 使用 LangChain 和 LangSmith 的经验，并促使其他人将其分享到社交媒体以获得更广泛的关注。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/woodstock-happy50th-anniversary-happy-gif-26217300">Woodstock Happy50th GIF - Woodstock Happy50th Anniversary - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/playlist?list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg">LangGraph (Python)</a>：该视频系列涵盖了如何使用 LangGraph 的代码功能，以及可能想要进行的常见修改。</li><li><a href="https://python.langchain.com/v0.2/docs/introduction/">Introduction | 🦜️🔗 LangChain</a>：LangChain 是一个用于开发由大语言模型 (LLM) 驱动的应用程序的框架。</li><li><a href="https://reflex.dev/">Reflex · Web apps in Pure Python</a>：未找到描述</li><li><a href="https://lucasvw.github.io/posts/20_ollama_langchain/">Lucas van Walstijn - Having fun with llama3:8b</a>：未找到描述</li><li><a href="https://www.answer.website/">answers, how they should be displayed.</a>：由 developers digest 构建的回答引擎</li><li><a href="https://github.com/wandb/openui">GitHub - wandb/openui: OpenUI let&#39;s you describe UI using your imagination, then see it rendered live.</a>：OpenUI 让你用想象力描述 UI，然后实时查看渲染效果。</li><li><a href="https://github.com/langchain-ai/langchain-nextjs-template/blob/main/app/generative_ui/README.md">langchain-nextjs-template/app/generative_ui/README.md at main · langchain-ai/langchain-nextjs-template</a>：LangChain + Next.js 入门模板。通过在 GitHub 上创建账号为 langchain-ai/langchain-nextjs-template 的开发做出贡献。</li><li><a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/">What We Learned from a Year of Building with LLMs (Part I)</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[allaire_inspect_ai](https://discord.com/channels/1238365980128706560/1242943547699888229/1245832348189200485)** (4 messages): 

<ul>
    <li><strong>UI 洞察带来巨大收益</strong>：一位成员指出，在 UI 中比较运行结果是一个“大弱点”，暗示目前只能将 matplotlib 作为权宜之计。他们认为，原生集成这些功能可以以极小的代价提供显著的洞察和可见性。</li>
    <li><strong>对工具演进的期待</strong>：另一位成员对该工具的未来发展表示兴奋。他们热切期待基于建议改进后的版本演进。</li>
    <li><strong>对生产环境中 solver 和 log 二元性的好奇</strong>：一位用户询问 solver 和 log 是否旨在生产环境中承担双重角色，例如调用 solver 执行任务并编写评估 log。他们很好奇系统是否支持这种双重功能的方法。</li>
    <li><strong>文档备受赞誉</strong>：一位用户称赞了文档的高质量，形容其“做得非常出色”。他们提到 solver 是一个处理 TaskState 的 Python 函数，且支持许多自定义，并强调了所提供代码示例的教育价值。</li>
</ul>

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1245819907229093970)** (12 messages🔥): 

- **关于注册截止日期的困惑**：关于提交 **$3,500 算力额度 (compute credits)** 申请表的截止日期存在一些困惑，[Hamel Husain 的推文](https://x.com/hamelhusain/status/1795871985265946934?s=12) 指出是 5 月 29 日，而邮件通知则建议是 5 月 30 日。

- **最后一刻提交表单**：成员们担心能否在截止日期前提交表单，但随后得到澄清，截止日期定在午夜，为前一天晚上报名的人提供了 24 小时的宽限期。

- **额度分配延迟**：用户对 Modal 表单额度发放的延迟表示担忧，**Charles** 解释说，轻微的延迟是因为人工审核流程需要时间。

- **Predibase 注册问题已解决**：**Michael Ortega** 告知用户，Predibase 已经取消了创建账号时对 Gmail 地址的限制，并鼓励遇到问题的用户联系支持部门。

- **额度到账通知**：会议澄清了额度将直接反映在用户相应平台的账户中，由于不同供应商的响应速度不同，分配过程可能会持续到下周中旬。

**提到的链接**：<a href="https://x.com/hamelhusain/status/1795871985265946934?s=12">Hamel Husain (@HamelHusain) 的推文</a>：$3,500 的算力额度申请今天截止。在 2024 年 5 月 29 日晚上 11:59 (PST) 之后我们将无法发放。引用 Eugene Yan (@eugeneyan) 的话：PSA: LLM-conf + finetuning workshop 的报名即将关闭...

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[west-coast-usa](https://discord.com/channels/1238365980128706560/1245410680065097738/1245823316187680819)** (1 messages): 

- **旧金山 Evals 聚会**：一位成员宣布本周日将在他们位于旧金山 Mission 区的合作社举办一场约 **50 人规模** 的聚会，讨论 evaluations。他们请感兴趣的人私信获取邀请，并提供社交账号以便验证。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/1245826070712815617)** (4 messages): 

- **期待私人活动的注册**：一位成员对注册即将举行的活动感到兴奋，并希望能被选中。
  
- **呼吁在华盛顿特区举办见面会**：另一位成员建议在华盛顿特区 (DC) 组织一场见面会活动。

- **芝加哥的地理困境**：一位成员强调芝加哥感觉离东海岸更近，并询问是否可以创建一个 midwest-usa 频道。

- **Modal Labs 在纽约举办 Office Hours**：Modal Labs 将在他们位于纽约 SoHo 区的总部举办 Office Hours。注册详情、地点和活动安排列在 [活动链接](https://lu.ma/nllqm67p) 中，向通过钱包验证代币所有权的用户开放。

**提到的链接**：<a href="https://lu.ma/nllqm67p">[NYC] Modal Office Hours · Luma</a>：对你的 Modal 部署有疑问或只是想了解更多？欢迎参加我们在纽约的首次 Office Hours！即使你没有具体问题，也欢迎……

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1245814422065254563)** (4 条消息): 

- **用户分享他们在欧洲的所在地**：成员们正在从欧洲不同地区签到。一位用户提到在**德国纽伦堡**，另一位来自**意大利米兰**，第三位来自**德国慕尼黑**，最后一位来自**奥地利林茨**。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[announcements](https://discord.com/channels/1238365980128706560/1245460787196068030/1245889007465009192)** (1 条消息): 

- **额度申请表最后提醒**：*"这是关于额度的最后一次提醒！如果你在接下来的 8 小时内不填写表格，你将无法获得任何可用的额度！为了保险起见，你可以再次填写。"* 敦促成员立即填写额度申请表以确保获得访问权限。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1245869285881024532)** (2 条消息): 

- **Predibase 提供免费试用**：鼓励用户[注册 Predibase 的 30 天免费试用](https://predibase.com/free-trial)，其中包括 25 美元的额度。Predibase 允许在可扩展的云基础设施上微调和部署开源模型。
- **关于在 Predibase 中选择 checkpoint 的咨询**：一位用户询问在微调模型后是否可以尝试不同的 checkpoint，并引用了 Predibase 的文档：“用于推理的 checkpoint 将是在评估集上表现最好（最低 loss）的 checkpoint。”该用户使用 Predibase 在一个约 200 条记录的小型数据集上微调了一个 L3 70B 模型。

**提到的链接**：<a href="https://predibase.com/free-trial">申请免费试用</a>：立即免费试用 Predibase - 注册您的试用版

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[career-questions-and-stories](https://discord.com/channels/1238365980128706560/1245816565073576037/1245821791390404658)** (8 条消息🔥): 

- **从学术界转向工业界的困扰**：一位拥有哲学和认知科学博士学位的用户分享了他们从学术界转向数据科学和软件工程的经历。他们表达了在小型学术实验室之外寻找新学习机会的挑战，以及在兴趣广泛且有家庭义务的情况下，在 AI 领域选择道路的困难。

- **承包合同（Contracting）是一个有利可图的选择**：另一位成员分享了他们在承包合同方面的积极经验，解释说这提供了接触多样化问题和文化的机会。他们强调了灵活选择项目的优势，以及如果表现出色，有可能获得机构提供的长期 offer。

- **工业界求职被拒是过程的一部分**：一位用户建议，进入工业界可能涉及多次被拒，甚至可能需要接受一份并不理想的第一份工作。他们强调了建立简历和 GitHub 作品集的重要性，以便让未来的求职申请更轻松、更成功。

- **从学术界向技术角色的转变**：一位成员讲述了他们从学术界进入工业界的经历，从一家技术初创公司的实习开始，最终担任了销售工程师和产品经理等各种角色。他们强调了在当前经济挑战下，很难找到不需要降低生活质量的机会。

- **鼓励和提供帮助**：几位用户鼓励了原帖作者和其他处境相似的人，提供了支持并强调了坚持的重要性。*"如果有什么我可以帮忙的，尽管找我。我非常支持大家互相扶持。"*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/)** (1 条消息): 

rubenamtz: 👀，额度（credits）还在准备中吗？
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1245830528599592970)** (10 条消息🔥): 

- **使用 llama.cpp 和 Qdrant 的 GPT 驱动 PDF 聊天**：查看 [everything-ai 项目](https://github.com/AstraBert/everything-ai)，该项目现在支持 llama.cpp 和 Qdrant，使用户能够与他们的 PDF 进行对话。这被公开赞誉为“最酷的社区新闻”，并受到社区成员的好评。

- **Codestral-22B 量化与 Nvidia 的新模型 Demo**：Mistral 模型的量化版本 [Codestral-22B-v0.1-GGUF](https://huggingface.co/QuantFactory/Codestral-22B-v0.1-GGUF) 备受关注。Nvidia 的 [embedding model demo](https://huggingface.co/spaces/Tonic/Nvidia-Embed-V1) 为本月分享的创新 AI 应用系列增色不少。

- **SD.Next 和 BLIP 数据集创新**：[SD.Next](https://github.com/vladmandic/automatic) 的发布因其全新的 UI 和高分辨率生成能力而受到赞誉。此外，BLIP 数据集是使用 [Clotho](https://huggingface.co/datasets/muzaik/captioned-audio-1k) 开发的，这增强了不断增长的数据集集合。

- **新工具和插件琳琅满目**：从开源（OSS）语音控制机器人手臂的 [YouTube 视频](https://www.youtube.com/watch?v=qv3bFhHoA5s) 到 Falcon VLM [demo](https://huggingface.co/spaces/Tonic/Falcon-Vision)，分享了多个实用程序和演示。其中包括免费使用的书法数据集、更好的转录应用以及视觉 3D 蛋白质分析工具。

- **社区活动与参与**：社区活动（如 [编程环节](https://discord.com/events/879548962464493619/1245406127668203541)）以及关于 AI 项目和社区主导新闻的讨论，因其价值而受到关注。这些亮点得到了多位社区成员的赞赏，使他们能够及时了解并参与最新的进展。

<div class="linksMentioned">

<strong>相关链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/chat/assistant/66562fe0abb44809b7f77897)">HuggingChat</a>: 让社区最好的 AI 聊天模型惠及每个人。</li><li><a href="https://www.youtube.com/watch?v=qv3bFhHoA5s)">开源语音控制机器人手臂 | 重新定义机器人！</a>: 欢迎来到语音控制 AI 机器人手臂项目，在这里人工智能与机器人技术相遇。这是一个开源倡议，赋予用户指挥机器人的能力...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1245814077989716128)** (415 messages🔥🔥🔥): 

- **关于数据格式和 Token 的困惑**：关于聊天机器人训练数据正确格式的讨论一直在进行。例如 `"<|user|>Do you enjoy cooking?</s><|assistant|>I can bake, but I'm not much of a cook.</s></s>"` 这样的例子引起了困惑，引发了关于是否需要两个 `</s>` Token 的疑问。
  
- **账单问题引发骚乱**：一位名为 Tensorrist 的用户对 100 美元的 Hugging Face 服务收费表示紧急抗议，声称他们从未开启过该服务。引导其联系 website@huggingface.co 支持部门的尝试似乎演变成了激烈的争执。

- **NeurIPS 模型融合竞赛**：分享了一项关于 NeurIPS 模型融合（Model Merging）竞赛的公告，提供 8000 美元的奖金池。社区被鼓励参与并革新模型选择与融合技术（[链接](https://x.com/LChoshen/status/1796256513519989102)）。

- **博客文章讨论**：多位用户讨论了创建教程博客文章，特别是专注于微调 TinyLlama 和 Mistral 等特定模型。一位用户请求帮助，以避免每次推送到 Hub 时都覆盖其 README。

- **关于使用特定数据进行微调的问题**：提出了关于在独特数据集上微调 LLM 的问题，例如使用来自维基百科的 RDF 转储，或仅使用文本数据微调多模态模型。回复建议了一些技术方法，并将用户引导至适当的频道以进行更深入的交流。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/cappuch/audio-embedding-wtf">So WTF is an Audio Embedding Model?</a>：未找到描述</li><li><a href="https://huggingface.co/blog/nroggendorff/finetune-mistral">Fine-tuning Mistral on Your Dataset</a>：未找到描述</li><li><a href="https://huggingface.co/QuantFactory/Codestral-22B-v0.1-GGUF">QuantFactory/Codestral-22B-v0.1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/collections/mixedbread-ai/eming-series-66010c86a966a1c8b6cbb658">em🍞ing series - a mixedbread-ai Collection</a>：未找到描述</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">来自 Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>：🚨 NeurIPSConf 模型融合竞赛！🚀 你能革新模型选择与融合吗？让我们创造最好的 LLM！🧠✨ 💻为了科学而来 💰为了 8000 美元留下 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://huggingface.co/blog/nroggendorff/finetune-tinyllama">Training a Language Model Using TinyLlama</a>：未找到描述</li><li><a href="https://x.com/DAlistarh/status/1796530164215820766">来自 Dan Alistarh (@DAlistarh) 的推文</a>：很高兴发布 PV-Tuning，这是一种针对高度压缩 LLM 的新微调技术，为 1-2.5bit LLM 设定了 SOTA PTQ 精度。与 @peter_richtarik 的实验室合作。arXiv: https://arxiv....</li><li><a href="https://tenor.com/view/have-a-nice-day-good-sunday-gif-1674652217239459225">Have A Nice Day Good Sunday GIF - Have a nice day Good sunday - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/blog/nroggendorff/finetune-tinyllama#4-formatting-the-dataset">Training a Language Model Using TinyLlama</a>：未找到描述</li><li><a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>：未找到描述</li><li><a href="https://tenor.com/qgNkIp2pvoz.gif">Simpsons Homer Simpson GIF - Simpsons Homer simpson - Discover &amp; Share GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1245925721407488092)** (3 messages): 

- **关于 Unit 1 身份的查询**：成员们讨论了 "Unit 1" 的概念，引发了关于这是否是 Hugging Face 课程的问题。随后得到了澄清，信息涵盖了包括强化学习（Reinforcement Learning）和计算机视觉（Computer Vision）在内的各种课程。

- **强化学习课程**：明确指出其中一门课程是强化学习课程，参与者在其中训练小狗 Huggy。分享了该课程的链接，指向 Hugging Face 的学习资源。

- **分享社区计算机视觉课程**：提到了另一门专注于计算机视觉 ML 的课程。课程目标包括使用 Hugging Face 生态系统中的库和模型教授 ML 概念，并分享了 [社区计算机视觉课程](https://huggingface.co/learn) 的链接。

**提到的链接**：<a href="https://huggingface.co/learn">Hugging Face - Learn</a>：未找到描述

  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1245834933231026216)** (6 条消息): 

- **NeurIPS 举办模型合并竞赛**：一名成员分享了关于在 **NeurIPS** 举办的模型合并竞赛的公告，并附带了[公告推文](https://x.com/LChoshen/status/1796256513519989102)链接。该竞赛提供 8,000 美元的奖金，由 **Hugging Face**、**Sakana AI Labs** 和 **arcee ai** 赞助，更多详情请见[官方网站](https://llm-merging.github.io/)。

- **Transformer 通过新嵌入处理算术问题**：一篇题为 *Transformers Can Do Arithmetic with the Right Embeddings* 的新论文揭示，为每个数字添加特定的位置嵌入（positional embeddings）可以帮助 Transformer 更有效地解决算术问题。该研究通过仅一天的 20 位数训练，在 **100 位数加法问题上实现了高达 99% 的准确率**。[在此阅读论文](https://huggingface.co/papers/2405.17399)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/papers/2405.17399">论文页面 - Transformers Can Do Arithmetic with the Right Embeddings</a>：未找到描述</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">来自 Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>：🚨 模型合并竞赛 @NeurIPSConf！🚀 你能彻底改变模型选择和合并吗？让我们创造最好的 LLM！🧠✨ 💻为科学而来 💰为 $8K 留下 💬Discord: https://discord.gg/dPBH...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1245833093705633914)** (10 条消息🔥): 

- **Blog-explorers 中的审核与文章发布**：一名成员请求协助审核另一名用户加入 Blog-explorers 社区，随后获得批准。该用户随后发表了一篇题为 *“使用 DuckDB 访问 Hugging Face 上的 15 万+数据集并使用 GPT-4o 查询数据”* 的文章，[在此发布](https://huggingface.co/blog/chilijung/access-150k)。
- **数字肖像生成应用开发中**：一款使用 **Stable Diffusion (SD) 结合 InstantID 以及 depth + pose CNs** 生成数字肖像的应用正在开发中。该项目仍在进行中，预计会有更多更新。
- **个人网站的虚假机器人**：为一个个人网站创建了一个虚假机器人，可以进行交互和探索。欢迎用户在 [ngxson.com](https://ngxson.com/) 进行尝试。
- **用于 LLM 的创意写作数据集**：分享了一个名为 [PlotPalette-10K](https://huggingface.co/datasets/Hatman/PlotPalette-10K) 的小型数据集，旨在用于微调大语言模型进行创意写作。它源自各种文学资源，并使用 Mistral 8x7B 模型生成。
- **SD 2.1 的多纵横比演示**：建立了一个带有自动 Checkpoint 更新系统的 **Stable Diffusion 2.1** 多纵横比演示 Space。该空间允许用户在 [pseudo-flex-v2](https://huggingface.co/spaces/ptx0/pseudo-flex-v2) 查看训练过程中的持续更新。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/chilijung">chilijung (Howard)</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/ptx0/pseudo-flex-v2">Ptx0 Terminus Xl Velocity V2 - ptx0 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://ngxson.com/">Xuan Son NGUYEN - 网络与安全工程师</a>：我是 Xuan Son NGUYEN。网络与安全工程师。我被机器学习的潜力及其应用所吸引，同时也热衷于探索底层细节...</li><li><a href="https://huggingface.co/blog/chilijung/access-150k-hugging-face-datasets-with-duckdb">如何直接使用 DuckDB 访问 15 万+ Hugging Face 数据集并使用 GPT-4o 进行查询</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Hatman/PlotPalette-10K">Hatman/PlotPalette-10K · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 条消息): 

taha_69513: 感感感感感感感感感感感感感感感感感感感感感感感感感感感感感感谢 🙌

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1245826407980859434)** (3 messages): 

- **论文报告测试准确率**：一位成员询问论文中报告的准确率通常是测试准确率（testing accuracy）还是验证准确率（validation accuracy）。他们引用了一篇对 ViT 进行 10k epoch 微调的论文，并分享了图中一个有趣的拼写错误：[论文链接](https://arxiv.org/abs/2211.12879)。

- **NeurIPS 模型合并竞赛发布**：一项与模型合并（model merging）相关的竞赛已发布，奖金为 8000 美元，并得到了 Hugging Face、Sakana AI Labs 和 Arcee AI 的支持。更多详情和报名请访问 [Model Merging Competition](https://llm-merging.github.io/) 以及此处的[公告推文](https://x.com/LChoshen/status/1796256513519989102)。

- **服装销售共享数据集**：一位成员分享了一个服装销售数据集，并提到他们使用自定义 PyTorch 模型成功训练了一个图像回归模型。他们还链接了关于这项工作的文章：[使用 PyTorch 和 🤗 Transformers 进行图像回归](https://huggingface.co/blog/tonyassi/image-regression)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/LChoshen/status/1796256513519989102">来自 Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>：🚨 NeurIPSConf 的模型合并竞赛！🚀 你能彻底改变模型选择和合并吗？让我们创造最好的 LLMs！🧠✨ 💻 为科学而来 💰 为 8000 美元留下 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford">Papers with Code - Stanford Cars 基准测试（细粒度图像分类）</a>：目前 Stanford Cars 上的 SOTA 是 CMAL-Net。查看 73 篇附带代码的论文完整对比。</li><li><a href="https://arxiv.org/abs/2211.12879">用于细粒度图像分类的数据增强 Vision Transformer</a>：最近，Vision Transformer (ViT) 在图像识别领域取得了突破。其 Self-attention 机制 (MSA) 可以提取不同像素块的判别性标注信息以改进...</li><li><a href="https://huggingface.co/datasets/tonyassi/clothing-sales-ds">tonyassi/clothing-sales-ds · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/blog/tonyassi/image-regression">使用图像回归进行销售预测</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1245829335382884364)** (7 messages): 

- **K2 模型超越 Llama 2 70B**：LLM360 发布了 [K2](https://huggingface.co/LLM360/K2)，这是一个**完全可复现的大语言模型**，在计算量减少 35% 的情况下超越了 **Llama 2 70B**。K2 完全开源，所有中间产物和结果均在 Apache 2.0 许可证下提供。

- **NeurIPS 模型合并竞赛**：今年 **NeurIPS** 将举办一场[模型合并竞赛](https://x.com/LChoshen/status/1796256513519989102)，提供 8000 美元奖金。该竞赛邀请参与者在 Hugging Face 和 SakanaAILabs 等赞助商的支持下，彻底改变模型选择和合并。

- **关于 Sentence Transformer 的咨询**：一位用户询问在 Sentence Transformer 实验中，句号 (.) 是否作为句子分界符，以及是否会从 "Dr." 等缩写中去除句号。他们有兴趣了解句子分割（sentence segmentation）是如何处理的。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/LLM360/K2">LLM360/K2 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">来自 Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>：🚨 NeurIPSConf 的模型合并竞赛！🚀 你能彻底改变模型选择和合并吗？让我们创造最好的 LLMs！🧠✨ 💻 为科学而来 💰 为 8000 美元留下 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://huggingface.co/collections/m-a-p/neo-models-66395a5c9662bb58d5d70f04">Neo-Models - 一个 m-a-p 集合</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1245822944416890971)** (205 messages🔥🔥): 

- **Phi3 在没有量化技巧的情况下表现令人失望**：一位用户分享了他们对 **Phi3 finetune** 结果的沮丧，将问题归因于 **exl2 quantization** 破坏了 SWA。他们总结道：*"在不量化的情况下运行... 结果有点糟糕。"*
- **关于数据集合并和微调策略的讨论**：成员们讨论了 finetuning 的最佳实践，包括在训练前合并数据集。一位用户强调，在混合数据集上进行训练有助于避免**灾难性遗忘**等问题。
- **Llama-3 获得非官方 11.5B upscale 模型**：一位用户分享了他们使用 **upscaling techniques**（而非持续预训练）创建的**非官方 Llama-3 11.5B 模型**。他们保证：*"Full finetune 可能会效果更好，"* 但该模型目前已经可以开箱即用。
- **可能的新微调方法 MoRA**：一位成员提到了 **MoRA**，这是用于 **parameter-efficient finetuning** 的 LoRA 更新版本，并提到其在未来升级中的潜在用途。MoRA 的 GitHub 代码已被[分享](https://github.com/kongds/MoRA)。
- **加载模型时 HF Trainer 的问题**：用户讨论了在加载模型（尤其是在 CPU 上）时 notebook 崩溃的问题。建议的一个临时解决方案是*删除文件夹名称中的空格*以解决崩溃。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Replete-AI/Llama-3-11.5B-V2">Replete-AI/Llama-3-11.5B-V2 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/dudeman6790/status/1796382605086015993">RomboDawg (@dudeman6790) 的推文</a>：似乎除了 TruthfulQA 之外，我们前段时间制作的 upscale 模型基本上没有损失。所以如果有人想使用 Llama-3 的 upscale 版本进行 finetune，那么基础版本...</li><li><a href="https://tenor.com/view/arthur-morgan-rdr2-rdr-2-gif-15824440020465833707">Arthur Morgan Rdr2 GIF - Arthur morgan RDR2 RDR 2 - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/Replete-AI/Llama-3-13B">Replete-AI/Llama-3-13B · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/kongds/MoRA">GitHub - kongds/MoRA: MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning</a>：MoRA：用于参数高效微调的高秩更新 - kongds/MoRA
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1246032724146520126)** (9 messages🔥): 

- **NVIDIA 新型 4nm 芯片性能远超 Blackwell**：NVIDIA 的新型研究芯片实现了 96 int4 TOPs/Watt，而 Blackwell 为 20T/W，代表了显著的能效提升。欲了解更多详情，请查看[完整研究演讲](https://youtu.be/gofI47kfD28?si=41UIMkpMCyb_qWqA)。
  
- **关于 float4 指数和尾数的困惑**：Daniel Han 提到 B200 的 float4 据称指数 (exponent) 和尾数 (mantissa) 均为 2，这引起了困惑。通常对于 float4，常见的配置是 *符号位 + 指数 + 尾数 = 4*；因此讨论集中在 NVIDIA 的配置是否缺少符号位。

- **数值表示缩减带来的显著加速**：Daniel 强调，显著的加速并非仅仅源于 Moore's Law，而是源于将数值表示从 fp32 缩减到 f4，提供了 32 倍的提升。然而，《Physics of LLMs》论文显示 int4 可能会差 2 倍，这表明这些性能改进存在极限。点击[此处](https://arxiv.org/abs/2404.05405)查看论文。

- **Tensor Cores 和 HMMA 提供卓越效率**：使用 HMMA 等复杂指令的 Tensor Cores 实现了 13 倍的性能提升且能耗更低，在增强计算效率方面发挥着关键作用。

- **稀疏化技术的进展**：NVIDIA 正在致力于从 2:4 稀疏转向 2:8 稀疏，这可能会进一步优化 AI 模型的计算效率和性能。

**提到的链接**：<a href="https://x.com/danielhanchen/status/1796253349932843214">Daniel Han (@danielhanchen) 的推文</a>：我从 NVIDIA 研究演讲中记录的笔记：1) NVIDIA 拥有一款研究级推理 4nm 芯片，其效率为 96 int4 TOPs/Watt，而 Blackwell 为 20T/W 2) B200 的 float4 是指数=2 且 尾数=2？也许我听错了...

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1245863643250429952)** (150 条消息🔥🔥): 

- **在 Unsloth 中选择 GPU**：一名成员询问如何在 Unsloth 中选择特定的 GPU 进行训练。另一名成员建议在 Python 中使用 `os.environ["CUDA_VISIBLE_DEVICES"]="0"`，并提供了一个[参考链接](https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter)以获取更多详情。

- **解决量化模型的微调错误**：一位用户在使用个人数据集对 unsloth/llama-3-8b-bnb-4bit 量化模型进行微调时遇到了 ValueError。另一位用户指出需要在量化模型上附加可训练的适配器 (adapters)，并参考了 [Parameter-Efficient Fine Tuning (PEFT)](https://huggingface.co/blog/peft) 以获取更多信息。

- **处理双模型微调**：一名成员讨论了处理两个紧密相关任务的复杂性，其中第一个模型的输出作为第二个模型的输入。他们考虑过生成“合成”数据，但结论是这与真实数据等效。

- **Unsloth 在 Kaggle 上的问题**：一位用户报告了在 Kaggle notebook 中安装 Unsloth 的问题，该问题已被确认并正在调查中。他们还在 GitHub 上提交了一个 issue（[链接](https://github.com/unslothai/unsloth/issues/566)）。

- **理解训练中的 VRAM 峰值**：成员们讨论了使用 Unsloth 训练期间的 VRAM 峰值，指出长序列长度 (16K tokens) 会导致碎片化和内存分配问题。建议包括使用环境变量，如 `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/peft">使用 🤗 PEFT 加载适配器</a>：未找到描述</li><li><a href="https://github.com/kongds/MoRA">GitHub - kongds/MoRA: MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning</a>：MoRA：用于参数高效微调的高秩更新 - kongds/MoRA</li><li><a href="https://github.com/unslothai/unsloth/issues/566">Kaggle notebook: No module named &#39;unsloth&#39; · Issue #566 · unslothai/unsloth</a>：我在 Kaggle notebook 中遇到了错误，这是我安装 unsloth 的方式：</li><li><a href="https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter)">在 jupyter 中设置 Tensorflow 的 CUDA_VISIBLE_DEVICES</a>：我有两个 GPU，想通过 ipynb 同时运行两个不同的网络，但第一个 notebook 总是分配所有 GPU。通过使用 CUDA_VISIBLE_DEVICES，我可以隐藏设备...</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">主页</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=upcOlWe7A1vc">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?us">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1246032169021870162)** (4 messages): 

- **Ghost XB Beta 支持多种语言**：**Ghost XB Beta** 模型旨在**流利支持 9+ 种语言**。该模型目前处于**第一阶段训练进度的 29%**，更新信息可在 [Ghost AI 的 X 页面](https://x.com/ghostx_ai)和 [ghost-x.org](https://ghost-x.org) 查看。
- **优先考虑开放性和效率**：该项目强调**社区支持**和**初创公司**，专注于中小型模型的**自部署**和**成本效益**。有关更多详细信息和部署，用户可前往 [Hugging Face](https://huggingface.co/ghost-x)。
- **针对生产环境优化**：这些模型专为**高性能和企业级可扩展性**而设计，且成本较低。它们可以部署在各种平台上，确保了广泛的可用性和**易用性**。
- **用于高级任务的 Ghost Alpha**：下一代 **Ghost Alpha** 模型针对**推理、多任务知识和多语言支持**进行了优化。用户可以在 [Hugging Face](https://huggingface.co/ghost-x/ghost-alpha-661005edac50abb8d56c90f1) 上探索这些模型。

如需进一步探索，请查看 [Hugging Face 上的 Ghost Alpha](https://huggingface.co/ghost-x/ghost-7b-alpha)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ghostx_ai">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://ghost-x.org">Ghost X</a>：Ghost X 的开发目标是研究和开发对人类有用的人工智能。</li><li><a href="https://huggingface.co/ghost-x">ghost-x (Ghost X)</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1245818109063856188)** (281 messages🔥🔥): 

- **Tako 小组件查询的地域限制**：一位成员询问金融数据小组件 Tako 是否仅限于美国使用，但得到了不确定的回复。另一位用户也表示“不确定”。

- **Pro 试用已永久取消**：一位成员询问是否可以为朋友提供 Perplexity Pro 试用，结果被告知试用已被移除，包括每年的 7 天试用。这引发了关于自行承担试用成本或使用推荐策略可能性的讨论。

- **编辑页面章节的困惑**：一位用户在编辑页面章节时遇到困难，了解到编辑按钮允许更改章节详情，但不能更改实际文本。另一位用户确认并解释了这一有限的功能。

- **关于 Pro 搜索变慢的讨论**：几位用户注意到 Perplexity Pro 搜索最近变慢了，这归因于一种将查询分解为顺序步骤的新搜索策略。尽管速度变慢，用户仍对改进后的详细回答表示赞赏。

- **关于 Pro 功能和模型的咨询**：成员们讨论了 Pro 功能是否包括与 GPT-4o 兼容的图片附件，以及 UI 中缺少某些模型限制的问题。此外还提到，Pro 用户应选择默认模型以外的模型以获得更好的性能。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.m.wikipedia.org/wiki/Secure_Remote_Password_protocol">Secure Remote Password protocol - 维基百科</a>：未找到描述</li><li><a href="https://aistudio.google.com/app/prompts/1s_utccw9l5Qdm9Aaxwj0DWyBMpGo1UUP">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/hatsune-miku-vocaloid-jump-birthday-anime-gif-26599041">Hatsune Miku Vocaloid GIF - Hatsune Miku Vocaloid Jump - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.c">GitHub 上的 llm.c/train_gpt2.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1245856701517926526)** (2 messages): 

- **Perplexity Pages 已发布**：一位成员分享了 [Perplexity Pages 首秀](https://www.perplexity.ai/page/Perplexity-Pages-Debut-EzSwGJ2LTIq253dkjxrXjg)的链接。这介绍了 Perplexity AI 平台上的一个新功能或服务。
- **探索 Codestral de Mistral**：另一位成员发布了关于 [Codestral de Mistral 的链接](https://www.perplexity.ai/page/Codestral-de-Mistral-ILrQahR6Rn6pAYBtZFw1rQ)。这可能详细阐述了与 Perplexity AI 相关的特定主题或功能。

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1245821292691853322)** (11 messages🔥): 

- **NVIDIA 的先进研究芯片令人印象深刻**：一位成员分享了 [NVIDIA 研究演讲](https://x.com/danielhanchen/status/1796253349932843214?s=46)的笔记，重点介绍了一款新型 4nm 推理芯片，其拥有 96 int4 TOPs/Watt，远超 Blackwell 的 20T/W。讨论还涉及了 Tensor Cores 的进展以及向 2:8 稀疏性（sparsity）转变的潜力。
  
- **Meta 下一代 AI 加速器亮相**：讨论围绕 Meta 的[下一代 AI 训练和推理加速器](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/#hardware)展开，该加速器承诺每个节点配备 72 个加速器，在 90W 功耗下达到 354 TFLOPS/s (INT8)，为先进的 AI 基础设施奠定了基础。
  
- **即将发布的 CUDA C/C++ 课程公告**：一位成员宣布了他们即将在 FreeCodeCamp 上推出的 GPU 编程课程，并邀请社区提供反馈。该计划旨在通过提供详细解释和鼓励社区投入，来降低 GPU 编程的高入门门槛。

- **对特定 GEMM 覆盖范围的请求**：对 CUDA 课程的反馈包括请求涵盖矩形矩阵的 GEMM 和广播规则，这对于基于 img2col 的卷积至关重要。此外，另一位用户表示，尽管他们有 WebGPU 经验，但仍非常期待这门课程。
  
- **对并行扫描（Parallel Scan）论文的疑问**：一位成员寻求对 `Single-pass Parallel Prefix Scan with Decoupled Look-back` 论文中概念的澄清，特别是关于其两次完整传递（passes）的陈述以及全局分配机制中的分配概念。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1796253349932843214?s=46">来自 Daniel Han (@danielhanchen) 的推文</a>：我从 NVIDIA 研究演讲中记录的笔记：1) NVIDIA 拥有一款研究级推理 4nm 芯片，性能为 96 int4 TOPs/Watt，而 Blackwell 为 20T/W；2) B200 的 float4 是指数=2 且尾数=2？也许我听错了...</li><li><a href="https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/#hardware">未找到标题</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=UU1WVnMk4E8&t=5775s&ab_channel=freeCodeCamp.org">用 Python 从零开始创建一个大语言模型 – 教程</a>：学习如何从头开始构建你自己的大语言模型。本课程深入探讨了大语言模型背后的数据处理、数学和 Transformers....</li><li><a href="https://www.youtube.com/watch?v=5Sm9IVMet9c&t=119s&ab_channel=freeCodeCamp.org">Mojo 编程语言 – 初学者完整课程</a>：在这个完整的教程中学习 Mojo。Mojo 编程语言结合了 Python 的易用性和 C 的性能。它基本上是一个增强版...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1245860653973897257)** (1 messages): 

- **易于理解的 Triton Kernel 名称**：Kernel 的默认名称通常是 `triton_`。在 `_inductor/config.py` 中有一个配置可以将这些名称更改为更易于理解的形式，不过建议**使用 ncu 进行 Kernel Profiling**。
  

---


### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1246175277428641813)** (1 messages): 

- **扫描算法（Scan Algorithm）第二部分开始**：公告通知社区，**扫描算法**课程的第二部分即将开始。鼓励成员在接下来的几分钟内加入。
  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1245841697745928193)** (10 条消息🔥): 

- **选择 vllm 进行 Web 服务器设置**：一位成员阐明了 vllm 的用途，指出：“*至于推理部分，我认为 vllm 是更好的选择（提供了一个开箱即用的聊天 Web 服务器）。*”
- **单模型批处理难题**：一位成员询问如何在不创建多个模型实例的情况下在 PyTorch 中处理批处理（batching），并对“*为每张图像创建模型副本这种非常低效的方式*”表示沮丧，寻求高效共享模型参数的建议。
- **PyTorch DataLoader 与自定义批处理**：一位用户建议使用 PyTorch DataLoader 进行批处理，但原帖作者反驳说他们的函数包含自定义的逐图像操作，因此不希望进行全面的批处理重构。
- **torch.multiprocessing 与 CUDA 报错**：尽管尝试使用 `torch.multiprocessing` 来处理多个进程，一位成员还是遇到了与 CUDA 核心预留相关的错误，并指出：“*它报错说使用了已经被预留的 CUDA 核心。*”
- **针对单图兼容性的 Dataloader 迭代**：讨论以一个建议结束：如果模型一次只能支持一张图像，Dataloader 仍然可以迭代地返回单张图像，从而避免了对基于批处理的功能进行更改的需求。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py">GFPGAN/inference_gfpgan.py at master · TencentARC/GFPGAN</a>: GFPGAN 旨在开发用于真实场景人脸修复的实用算法。 - TencentARC/GFPGAN</li><li><a href="https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel">PyTorch 数据加载器的详细示例</a>: 未找到描述</li><li><a href="https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py#L126-L131">GFPGAN/inference_gfpgan.py at master · TencentARC/GFPGAN</a>: GFPGAN 旨在开发用于真实场景人脸修复的实用算法。 - TencentARC/GFPGAN</li><li><a href="https://pytorch.org/docs/stable/notes/multiprocessing.html">多进程最佳实践 &mdash; PyTorch 2.3 文档</a>: 未找到描述</li><li><a href="https://github.com/pytorch/examples/blob/main/mnist_hogwild/main.py#L92-L99">examples/mnist_hogwild/main.py at main · pytorch/examples</a>: 围绕 PyTorch 在视觉、文本、强化学习等领域的一系列示例。 - pytorch/examples
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1246177049635459215)** (2 条消息): 

- **Izzat 的 Scanning Session 第二部分开始**：一位成员宣布 Izzat 的第二部分 scan 环节现在开始。他们提供了一个[加入会议的链接](https://linkedin.zoom.us/j/98060172269)。

**提到的链接**: <a href="https://linkedin.zoom.us/j/98060172269">加入我们的 Cloud HD 视频会议</a>: Zoom 是现代企业视频通信领域的领导者，拥有简单、可靠的云平台，可用于移动端、桌面端和会议室系统的视频和音频会议、聊天及网络研讨会。

  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1245816576553390175)** (127 条消息🔥🔥): 

- **今天达到 20k Stars**：一位用户兴奋地指出，*"今天可能会达到 20k stars 🎉"*，反映了社区对这一重大里程碑的期待。
- **合并 llmc lib 目录**：一名成员讨论了计划合并 llmc lib 目录并优化脚本结构以实现更好的组织。可以在[这里](https://github.com/karpathy/llm.c/pull/475)查看 Pull Request。
- **训练器在 Loss Spikes（损失突刺）中挣扎**：几位用户阐述了训练运行中遇到可复现的 Loss Spikes 问题。有人建议绘制 gradient norms（梯度范数）以识别潜在原因，而另一位则建议在调试期间考虑硬件差异。
- **数据集下载与在 Hugging Face 上托管**：用户讨论了数据集 shards（分片）的差异，并考虑在 Hugging Face 上托管以方便访问。一位用户提供了一个 [Hugging Face 数据集链接](https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/blob/main/fineweb_train_000377.bin)并建议使用压缩方法来优化下载速度。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/channel/UC0qODiToYRpntlfKTu1TbGw">Chris Dryden</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb/discussions/28">HuggingFaceFW/fineweb · &quot;使用 Karpathy 的 fineweb 在 90 分钟内花费 20 美元复现 GPT-2 (124M)&quot;</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/blob/master/dev/data/data_common.py#L37">llm.c/dev/data/data_common.py at master · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账户为 karpathy/llm.c 开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/master/scripts/run_gpt2_124M.sh">llm.c/scripts/run_gpt2_124M.sh at master · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账户为 karpathy/llm.c 开发做出贡献。</li><li><a href="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/blob/main/fineweb_train_000377.bin">fineweb_train_000377.bin · chrisdryden/FineWebTokenizedGPT2 at main</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb_train_000377.bin?download=true">无标题</a>: 未找到描述</li><li><a href="http://d3joo518jcgf5j.cloudfront.net/fineweb_train_000377.bin">无标题</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/475">karpathy 尝试添加 llmc lib 目录 · Pull Request #475 · karpathy/llm.c</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[youtube-watch-party](https://discord.com/channels/1189498204333543425/1238931064223830016/1245819136333053952)** (3 条消息): 

- **对下一讲的不确定性**：一名成员提到 Lecture 7 可能是下一场，但他们不确定 NAM 团队的时间表。他们请求对此进行确认。

- **询问 NAM 团队的 Zoom 链接**：另一名成员询问 NAM 团队的 Zoom 链接是否发布在频道中。这个问题旨在弄清楚即将到来的会议的正确平台。

- **链接分享**：[这里](https://discord.gg/5wCbVAjy)分享了一个 Discord 链接，可能用于相关信息或资源。该链接未提供额外的上下文。

**提及的链接**: <a href="https://discord.gg/5wCbVAjy">加入 PMPP UI 讲座时区 Discord 服务器！</a>: 查看 Discord 上的 PMPP UI 讲座时区社区 - 与其他 37 名成员一起交流，享受免费的语音和文字聊天。

  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1245815010505134110)** (4 条消息): 

- **CUDA 的 Windows 调试**：一名成员表示有兴趣使其在 Windows 上运行，尽管 **torchao** 官方不支持 Windows。他们提到，“我可以用我的 Windows 电脑调试它。”

- **编译器差异导致构建错误**：讨论了由于编译器之间 `__restrict__` 和 `__restrict` 的不同含义而导致的构建错误。

- **承认 Triton 的困难**：一名成员指出，让 Triton 在 Windows 上运行并非易事，称“Triton 的东西不会那么容易实现。”

- **FP6 LLM 讨论转移**：关于 **FP6 LLM** 的讨论，他们指向了另一个频道：*“我们可以在 <#1235775768756359289> 讨论”*。
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1245814193286807552)** (152 条消息🔥🔥): 

- **在线内容的隐私与权限**：一位成员强调，最好的隐私功能是不发布的选项。另一位成员补充说，需要教育人们不要将自己的内容提供给公司，以避免被其使用。
- **不同工具间结果不一致**：一位成员对在 ComfyUI 中无法获得与 Forge 相同的结果感到沮丧，即使设置完全相同。他们指出设置的差异以及像 XFormers 这样的功能可能会影响结果。
- **结合模型以改进输出**：讨论强调了 SDXL 和 SD15 等模型可以通过多种方式结合以获得更好的结果，尽管 ControlNet 在模型各阶段需要保持一致性。
- **训练和使用特定模型**：用户分享了与训练满足特定需求模型相关的担忧和建议。一位成员引用了 YouTube 视频，并指向了用于 Lora 训练的 OneTrainer 和 kohya_ss 等工具。
- **资源推荐**：对于初学者，推荐使用 [Craiyon](https://www.craiyon.com) 等资源进行 AI 生成图像的初步实验，然后再转向更高级的 Web 服务或本地安装。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/dancing-cat-dance-cat-cat-meme-chinese-cat-gif-12629347036627000898">Dancing Cat Dance GIF - Dancing cat Dance Cat - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=mxin2EjzvQM&list=PLRvM4LTxFnNVZxqIJlkB0ff3V6drhUD-r&index=2">Troca do Estilo de Arquitetura / Quiosque da Praça de Divino, MG - Controlnet IP-Adapter</a>：你好！在本视频中，我们使用强大的 IP-Adapter 工具，根据现有建筑的基础照片来更换建筑风格...</li><li><a href="https://civitai.com/models/448101/sprite-sheet-maker">Sprite Sheet Maker - v4.2 Ollama | Stable Diffusion Workflows | Civitai</a>：更新：v4.2 增加了 Ollama 和 IP adapter。版本 4.0 - 看起来好像跳过了一些版本，它们确实存在，只是我忘了分享...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d4cwi9/pcm_phased_consistency_model/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1245835875107999844)** (60 条消息🔥🔥): 

- **GPU 在 ROCm 支持方面的困扰**：一位用户对他们的 **RX 6700** 性能表示沮丧，指出由于缺乏 **ROCm 支持**，速度无法超过每秒 10 个 token。另一位用户提到，双系统启动到 Ubuntu 也不会有显著改善，因为不同操作系统的后端速度相似。

- **修复意外的端点错误**：一位用户报告了 `GET /v1/chat/completions/models` 的错误，导致另一位用户建议在特定频道发布更多上下文。此外，有人指出正确的端点是 `v1/models`。

- **对 AVX2 指令集要求的挫败感**：多位用户在尝试运行 LM Studio 时遇到问题，排查发现 **AVX2 指令** 是必需项。推荐使用旧版本 [0.2.10 for AVX](https://lmstudio.ai/beta-releases.html) 作为替代方案。

- **PDF 和文本转视频的局限性**：用户经常询问如何将 **PDF 文件** 提供给 AI，并被引导使用带有第三方应用 AnythingLLM 的服务器模式。另一个关于文本转视频应用的查询强调了目前仅有像 **Stable Video Diffusion** 这样有限的概念验证选项，且仅适用于 NVIDIA GPU。

- **本地化和 CPU 指令集问题**：在非英文区域设置下的安装错误以及缺乏 **AVX2 CPU 指令** 被确定为阻碍成功安装的常见问题。为使用不支持 AVX2 的旧 CPU 的用户提供了一个[旧测试版](https://lmstudio.ai/beta-releases.html)的链接作为潜在解决方案。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>：未找到描述</li><li><a href="https://tenor.com/view/monsters-university-snail-going-on-my-way-omw-gif-5461800">Monsters University Snail GIF - Monsters University Snail Going - Discover &amp; Share GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1245817396011208837)** (30 条消息🔥): 

- **Coder Prompt Template 解析**：一名成员澄清了特定模型使用的是 **Deepseek coder prompt template**，*强调了其对于关注设置的用户的重要性*。
  
- **使用 Codestral 手动提示 FIM**：一名成员询问了使用 **Codestral** 手动提示 FIM 的技巧，表现出对该方法细节的兴趣。目前尚未提供后续跟进或指导。

- **模型参数限制讨论**：一名成员提出了一个问题：增加模型的参数规模是否是提高其处理复杂指令（如维持多重人格）能力的唯一途径。他们对 Mixture of Expert 模型（MoE）的有效性表示怀疑，并指出根据经验，较大的模型通常表现更好。

- **Llama2 7B Chat Uncensored 格式问题已解决**：讨论了 **TheBloke/llama2_7b_chat_uncensored-GGUF** 的 Prompt 格式问题，并通过特定的 [model card 链接](https://huggingface.co/TheBloke/llama2_7b_chat_uncensored-GGUF#prompt-template-human-response) 提供了指导。对话中建议检查 tokenizer 配置和 model cards 以获取合适的 chat templates。

- **文本提取模型成功案例**：简要讨论了不同模型在文本提取任务中的成功表现。然而，在交流过程中未提及或推荐具体模型。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/DavidAU/Dark-Forest-V1-Ultra-Quality-20b-GGUF">DavidAU/Dark-Forest-V1-Ultra-Quality-20b-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/DavidAU/Fimbulvetr-11B-Ultra-Quality-plus-imatrix-GGUF">DavidAU/Fimbulvetr-11B-Ultra-Quality-plus-imatrix-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/TheBloke/llama2_7b_chat_uncensored-GGUF#prompt-template-human-response">TheBloke/llama2_7b_chat_uncensored-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1245994617405571123)** (2 条消息): 

- **视觉模型频道错误**：一名成员在错误的聊天室询问了关于视觉模型支持的问题。他们很快为失误道歉，承认了自己的错误：*"oops wrong chatroom soryr."*
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1245829023423008888)** (21 条消息🔥): 

- **RX580 面临 ROCm 兼容性问题**：成员们讨论了在 ROCm 上使用 **AMD RX580** GPU 的限制，指出它被认为“太旧”因此不兼容，导致用户只能依赖 OpenCL 进行 GPU 推理。一名成员确认道：*"ROCm is incompatible with Polaris GPUs like the 580."*

- **ROCm 与 RX 6600XT 不支持问题**：另一位用户询问了 **RX 6600XT** GPU 与 ROCm 的兼容性，结果被告知该 GPU 也不受支持。反应非常失望："Damn。"

- **M3 Max 击败 Snapdragon X Elite**：一位用户强调 **M3 Max** 在各个类别中都优于 Snapdragon X Elite，甚至还没考虑到 M4 和 M3 Ultra 模型。他们指出：*"最便宜的 Snapdragon PC 是 899 美元的开发者套件，但那是速度较慢的 32GB 型号。"*

- **LLAMA 3 8B Q8 的多 GPU 性能测试**：在测试各种配置时，一位用户发现考虑到 PCIE 带宽的影响，使用两个 GPU 的性能是单个 GPU 的 91%。结论是通过 X1 转 X16 转接器将负载分散到多个 GPU 可以提供更好的性能稳定性：*"我决定再买一个 X1 to X16 转接器。"*

- **GPUDirect 显示出潜力但有限制**：在讨论 NVIDIA 用于增强数据移动的 **GPUDirect** 技术时，一位用户提到了直接从 NVMe 存储读取以减轻 VRAM 内存压力的可能性。另一名成员评论说，虽然已经进行了此类尝试，但将磁盘用作 RAM 仍然太慢：*"[Using disk for RAM] is just too slow to be of use."*
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/lyogavin/llama3-airllm">Run the strongest open-source LLM model: Llama3 70B with just a single 4GB GPU!</a>：未找到描述</li><li><a href="https://developer.nvidia.com/gpudirect">GPUDirect</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1245983293091090542)** (11 messages🔥): 

- **4070 与 1070 VRAM 对比引发热议**：一位成员指出，与拥有额外 8GB 的 1070 相比，4070 仅有 12GB VRAM。这引发了关于这些显卡是否适合运行像 "codestral" 这样的大型模型的讨论。

- **1070 的性能受到质疑**：一位成员贬低 1070 的性能较慢，另一位成员则通过实际用例回应，展示了不错的性能表现，例如运行 Phi-3 Small 达到 50 tokens/s，运行 Llama-3-8b 达到 35 tokens/s。

- **CPU 线程利用率问题已解决**：Khabu 寻求关于优化 CPU 线程使用以获得更好模型性能的建议，并解释了当线程数增加超过 4 个时出现的问题。在经过一些故障排除建议后，Khabu 表示问题已解决，无需进一步讨论。


  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1245898428870033479)** (1 messages): 

- **InternLM 模型大爆发**：发布了大量 **InternLM** 模型，涵盖数学和编程领域，参数范围从 *7B* 到基于 *mixtral 8x22B* 的数学模型。多个模型已上线，包括 [AlchemistCoder-DS-6.7B-GGUF](https://huggingface.co/lmstudio-community/AlchemistCoder-DS-6.7B-GGUF)、[AlchemistCoder-L-7B-GGUF](https://huggingface.co/lmstudio-community/AlchemistCoder-L-7B-GGUF)、[internlm2-math-plus-7b-GGUF](https://huggingface.co/lmstudio-community/internlm2-math-plus-7b-GGUF)、[internlm2-math-plus-20b-GGUF](https://huggingface.co/lmstudio-community/internlm2-math-plus-20b-GGUF) 以及 [internlm2-math-plus-mixtral8x22b-GGUF](https://huggingface.co/lmstudio-community/internlm2-math-plus-mixtral8x22b-GGUF)。
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1245850116582281236)** (7 messages): 

- **OpenRouter 提升 API 性能**：重大的可扩展性改进使全球延迟降低了至少约 200ms，其中非洲、亚洲、澳大利亚和南美洲的提升最为显著。*"通过将更多用户数据推送到更靠近边缘（edge）的地方，我们为每一次请求削减了至少约 200ms 的延迟。"*

- **使用新图表监控模型运行时间**：OpenRouter 引入了运行时间（uptime）图表，以直观展示其供应商负载均衡的好处，例如 [WizardLM-2 8x22b](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b?tab=uptime) 的图表。此功能有助于避免受到零星上游停机的影响。

- **类别排名（Category Rankings）早期预览版上线**：用户可以在 [openrouter.ai/rankings](https://openrouter.ai/rankings) 查看不同模型在各个类别中的排名。值得注意的见解包括 MythoMax 在角色扮演（roleplay）中的主导地位，以及 GPT-4o 在编程领域的领先地位。

- **Laravel 开发者获得新扩展包**：宣布推出 [moe-mizrak/laravel-openrouter](https://github.com/moe-mizrak/laravel-openrouter)，帮助 Laravel 开发者集成 OpenRouter。

- **数据库问题导致 API 中断但已解决**：数据库缓存（DB cache）的内部错误导致 API 调用返回 504 或 500 错误。该问题主要影响印度 (bom1) 和新加坡 (sin1) 地区，但通过添加备用直接数据库查询已得到解决，正如报告所言：*"更新：修复程序现已上线，我们的 1 小时运行时间图表正在恢复。"*

**提到的链接**：<a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b?tab=uptime>).">WizardLM-2 8x22B by microsoft | OpenRouter</a>：WizardLM-2 8x22B 是 Microsoft AI 最先进的 Wizard 模型。与领先的专有模型相比，它展示了极具竞争力的性能，并始终优于所有现有的...

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1245826173007560859)** (1 messages): 

- **MixMyAI.com 提供按需付费的 AI 服务**：*推介 mixmyai.com*，作为满足所有 AI 需求的一站式解决方案，无需向不同供应商支付月费。该平台将闭源和开源模型整合在一起，提供最实惠的价格选择。

- **MixMyAI 强调用户隐私**：该服务将隐私放在首位，**不在服务器上存储任何聊天记录**，并提供透明的仪表板来追踪支出。它还通过淘汰旧模型来确保模型始终保持最新。

- **用户友好且强大的 UI**：MixMyAI 拥有强大的用户界面，允许用户*搜索聊天历史、保存提示词（prompts）并调整 LLM 设置*。该平台强调易用性和可访问性。
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1245814123741319168)** (97 条消息🔥🔥): 

- **Latent Space 播客欢迎粉丝**：首个 [关于 MosaicML MPT-7B 的深度访谈](https://www.latent.space/p/mosaic-mpt-7b?utm_source=substack&utm_medium=email) 讨论了如何克服 GPT-3 在 context length 方面的限制。他们还在举办聚会，并邀请 Beta 测试人员参加即将推出的 AI 课程。
- **OpenRouter Ruby 库发布**：Obie Fernandez 宣布发布 [OpenRouter Ruby 客户端库](https://github.com/OlympiaAI/open_router)。他还提到维护 [Ruby AI eXtensions for Rails](https://github.com/OlympiaAI/raix-rails)，这是一个依赖于 OpenRouter 的库。
- **各地区 API 504 问题频发**：许多用户报告在特定模型（如 "Mixtral 8x7B Instruct" 和 "llama-3-70b-instruct"）上遇到 504 错误，涉及印度、越南和波兰等多个全球地点。虽然应用了[临时修复](https://openrouter.ai/playground)，但稳定性仍不一致。
- **类别排名反馈与更新**：讨论集中在改进类别排名数据上，用户建议增加新类别，并强调需要根据常见用例和“质量”来评估模型。Alex Atallah 等人确认，更详细的排名和额外的类别即将推出。
- **关于 health check 端点的讨论**：用户请求一个 health check API 来监控 OpenRouter 的状态并采取相应行动。Alex Atallah 建议在专用端点可用之前，使用模型页面进行 health check。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.latent.space/p/mosaic-mpt-7b?utm_source=substack&utm_medium=email">MPT-7B and The Beginning of Context=Infinity — with Jonathan Frankle and Abhinav Venigalla of MosaicML</a>: 第 13 集：在 9 天内花费 20 万美元训练 Mosaic 的 "llongboi" MPT-7B，如何为训练准备优质数据，以及开源模型的未来。</li><li><a href="https://github.com/OlympiaAI/open_router">GitHub - OlympiaAI/open_router: Ruby library for OpenRouter API</a>: OpenRouter API 的 Ruby 库。欢迎通过在 GitHub 上创建账户为 OlympiaAI/open_router 的开发做出贡献。</li><li><a href="https://github.com/OlympiaAI/raix-rails">GitHub - OlympiaAI/raix-rails: Ruby AI eXtensions for Rails</a>: 用于 Rails 的 Ruby AI 扩展。欢迎通过在 GitHub 上创建账户为 OlympiaAI/raix-rails 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1245854992166944879)** (85 条消息🔥🔥): 

- **Pro 用户的新功能与权益**：成员们讨论了 OpenAI Pro 用户的权益，包括**更高的 rate limits** 以及访问 **DALL-E**、GPT 创建、实时语音和视频聊天的权限。免费用户无法创建 GPT，这使得每月 20 美元的订阅费用更具吸引力。

- **应用用例建议**：一位成员描述了正在开发一个具有特定特质的统治者 AI，考虑到不需要文件搜索/编码等某些功能，正在考虑使用 Chat API 而非 Assistant API。建议使用 **Chat API** 以获得更好的控制和效率。

- **ChatGPT 与偏见担忧**：一位成员因指责 ChatGPT 的回答存在种族歧视而被停号，对此表示沮丧，这引发了关于**模型偏见**和训练数据的深入讨论，以及此类偏见如何不可避免，但可以通过在训练数据之上的额外工作来减轻。

- **Anthropic AI Agents 功能**：成员们将 Anthropic 新的 **"tool use" 功能**与 OpenAI 的 function calling 进行了比较，指出两者都允许通过 API 集成创建自定义助手。尽管表面相似，但有人认为 Anthropic 的功能可能提供与个人数据更深的集成。 

- **Sora 和视频生成热潮**：讨论涉及了对 **Sora 和 Veo** 等新一代视频模型的兴奋，包括关于视频生成高筛选率（curation ratios）的轶闻。人们对目前的视频 AI 技术中，炒作与实际可用性之间的差距持怀疑态度。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/5/30/24167231/anthropic-claude-ai-assistant-automate-tasks">Anthropic’s AI now lets you create bots to work for you</a>: Anthropic 正在发布一款工具，允许客户构建自己的 AI 助手。</li><li><a href="https://www.theverge.com/2024/5/30/24167231/">Anthropic’s AI now lets you create bots to work for you</a>: Anthropic 正在发布一款工具，允许客户构建自己的 AI 助手。
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1245824522867708064)** (10 条消息🔥): 

- **Dashboard 链接请求**：一位用户请求一个能以创作者身份显示其所有 GPT 列表的链接，而不是单个 GPT 的 URL。频道内没有对此请求做出直接回应。
- **内存泄漏导致浏览器崩溃**：成员们报告在使用 ChatGPT 时出现严重的**延迟和浏览器崩溃**，将其归因于网站的**内存泄漏（memory leak）问题**。建议包括避免使用长 Context 以及在发生卡顿时刷新页面。
- **代码生成导致浏览器冻结**：一位用户反映在生成代码块（codeblocks）内的代码时，即使是在高端 PC 上也会遇到**浏览器冻结和崩溃**，并提到错误信息：*Out of memory*。该问题是最近出现的，且在多个浏览器中持续存在。
- **控制使用量的手段**：一位成员推测 OpenAI 可能会在需求高峰期通过监控用户每天达到使用上限的频率来控制使用量，并暗示这可能同时适用于 Web 界面和 API。
- **语音模式时间线分享**：针对一项查询，分享了来自 OpenAI 的时间线，指出 GPT-4 的**实时语音和视觉**功能将很快开始向 ChatGPT Plus 用户的限量 Alpha 测试版推送，并计划在未来几个月内扩大可用范围。[OpenAI 分享的时间线](https://help.openai.com/en/articles/8400625-voice-chat-faq)。
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1246026857758064711)** (3 条消息): 

- **处理重复的 GPT-3 回复**：一位成员面临 API 返回重复答案的问题，即使 Prompt 的主题并不狭窄。另一位成员建议将每个对话会话（chat session）的消息数量限制在 10 条左右，或者重复之前的完整回答以缓解此问题。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1246026857758064711)** (3 条消息): 

- **OpenAI API 中的重复回答**：一位成员报告称，尽管 Prompt 内容广泛，但仍从 OpenAI API 获得重复的回答。他们询问记录之前的回答并告知模型是否能解决问题，但担心列表会变得过于冗长。

- **限制对话会话有助于避免重复**：另一位成员建议将每个对话会话的消息数量限制在 10 条左右，作为避免重复回答的一种方法。他们还提到重复之前的完整回答可能有助于缓解此问题。
  

---



### **Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/)** (1 条消息): 

moonride303: https://x.com/jaseweston/status/1795978611784089799
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1245869560498884681)** (6 条消息): 

- **别具一格的罗宋汤**：一位成员分享了一个有趣的罗宋汤食谱，由**猪牛骨汤、牛心、蛋黄酱和味精**制成。他们将其与黄瓜、波罗底诺面包、苹果、橙子以及用**甜菊糖（stevia）**代替糖调味的果茶搭配食用。

- **关于甜菊糖与糖的辩论**：一位成员质疑食谱中使用甜菊糖的做法，断言“真正的糖对你有好处”。然而，原作者反驳说“糖是毒药”，并且“我们并没有进化到可以食用它”。

- **质疑甜菊糖的使用**：另一位成员幽默地指出，人类在进化上可能更不适应处理甜菊糖，除非是“来自亚马逊雨林的土著居民”。

- **不频繁的浆果摄入**：当被问及浆果的摄入时，原作者澄清说浆果应该是*“季节性的、不频繁的零食”*。
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1246000436788596828)** (16 条消息🔥): 

- **RLHF/DPO 令人惊讶的局限性**：一条分享的 [推文](https://x.com/_angie_chen/status/1796220428345573399) 表明，RLHF/DPO 并不能有效地生成为首选回答分配更高似然值（likelihood）的策略（policies）。这一见解在成员中引发了困惑和辩论，大家开始质疑这些算法在偏好学习（preference learning）中的有效性。

- **关于排序准确度的辩论**：成员们讨论了“排序准确度”（ranking accuracy）的概念，试图理解其在参考模型（reference models）背景下的定义和影响。结论是，偏好学习旨在训练模型以获得高排序准确度，即能够将首选输出排在非首选输出之上，但这似乎并非普遍成立。

- **过拟合的启示**：在多位成员审阅该帖子后，有人建议这一发现可能表明在使用 RLHF/DPO 方法时存在过拟合（overfitting）的可能性。这一观点遭到了反驳，指出讨论可能涉及更深层的数学证明，解释了为什么 DPO（特别是在没有 NLL 等额外措施的情况下）表现不佳。

- **技术对比**：成员们提到，在 DPO 中发现的类似问题也存在于 OpenAI 使用的 PPO 中。另一位用户强调，尽管存在这些问题，DPO 模型往往比单纯的 SFT (Supervised Fine-Tuning) 表现更好。

**提到的链接**：<a href="https://x.com/_angie_chen/status/1796220428345573399">来自 Angelica Chen (@_angie_chen) 的推文</a>：与 @sadhikamalladi, @lilyhzhang, @xinyichen2, @QiuyiRichardZ, Rajesh Ranganath, @kchonyc 合作的新工作：与传统认知相反，RLHF/DPO 并*不*会产生大多数情况下分配更高似然值的策略...

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1245815172183101563)** (63 messages🔥🔥): 

- **NeurIPS 举办 Model Merging 竞赛**：一项与 Model Merging 相关的竞赛在 [Twitter](https://x.com/LChoshen/status/1796256513519989102) 上公布，将在 NeurIPS 举行，奖金为 8000 美元。由 Hugging Face 和 Sakana AI Labs 赞助，[在此报名](https://llm-merging.github.io/)。
- **用于 AI 与动物交流的 Coller Prize**：为成功使用 AI 与动物进行交流提供 50 万美元的奖金，详见 [推文](https://x.com/nearcyan/status/1796243343288189179) 和 [更多信息](https://coller-dolittle-24.sites.tau.ac.il/)。此外还提到了 Aza Raskin 关于 Earth Species Project 的相关 YouTube 视频。
- **Llama 模型 Finetuning 资源**：Llama-3 的 Upscaled 版本，据描述除了 TruthfulQA 之外几乎无损，推荐用于 Finetuning（[推文](https://x.com/dudeman6790/status/1796382605086015993)，[HuggingFace 链接](https://huggingface.co/Replete-AI/Llama-3-11.5B-V2)）。一位用户评论道：*“反正 TruthfulQA 是个没意义的 Benchmark，所以这挺不错的”*。
- **Google AI Overviews 更新**：Google 向美国用户推出 AI Overviews，提升了搜索满意度和参与度（博客文章：[Google AI Overviews](https://blog.google/products/search/ai-overviews-update-may-2024/)）。尽管出现了一些错误的概览，但他们声称网页点击质量更高，并持续进行反馈循环改进。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/LChoshen/status/1796256513519989102">来自 Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>：🚨 NeurIPSConf 的 Model Merging 竞赛！🚀 你能革新模型选择和合并吗？让我们创造最好的 LLMs！🧠✨ 💻 为科学而来 💰 为 8000 美元留下 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://x.com/dudeman6790/status/1796382605086015993">来自 RomboDawg (@dudeman6790) 的推文</a>：似乎除了 TruthfulQA，我们不久前制作的 Upscaled 模型基本没有损失。所以如果有人想使用 Upscaled 版本的 Llama-3 进行 Finetuning，那么基础版本...</li><li><a href="https://huggingface.co/mistralai/Codestral-22B-v0.1">mistralai/Codestral-22B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=rjvsl0mhqTk">Aza Raskin 与 Earth Species Project</a>：Aza Raskin 描述了 Earth Species Project 的使命以及使用 AI 解锁非人类物种交流的最新进展。</li><li><a href="https://x.com/nearcyan/status/1796243343288189179">来自 near (@nearcyan) 的推文</a>：成功使用 AI 与动物交谈的 50 万美元奖金。令人兴奋！链接：https://coller-dolittle-24.sites.tau.ac.il/</li><li><a href="https://x.com/tommyfalkowski/status/1796098620430619038?s=46">来自 Tommy Falkowski (@TommyFalkowski) 的推文</a>：Caret 是我最喜欢的与 LLMs 交互的新方式！它是一个 Obsidian 插件，可以轻松进行分支对话！我正通过 @ollama 使用 Llama3:8b，效果非常好...</li><li><a href="https://blog.google/products/search/ai-overviews-update-may-2024/">AI Overviews：关于上周</a>：这里介绍了 AI Overviews 的情况、我们收到的反馈以及我们采取的措施。</li><li><a href="https://github.com/WecoAI/aideml/tree/main">GitHub - WecoAI/aideml: AIDE: 机器学习 CodeGen Agent</a>：AIDE：机器学习 CodeGen Agent。通过在 GitHub 上创建一个账户来为 WecoAI/aideml 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1246078429535932418)** (1 messages): 

- **LLMs 实时生成大部分网页内容**：正如解释的那样，*几乎所有内容都是由 LLM 制作的*，网页通常在用户看到它们“加载”时实时创建。尽管如此，由于 Context 限制，LLMs 在制作超长或超大网页方面仍面临困难。
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1245864638890115082)** (2 条消息): 

- **Milvus Lite：一个紧凑、高效的 Python 向量存储**：Milvus Lite，一个轻量级的 Python 向量数据库，现已发布。更多详情及入门指南请点击[此处](https://t.co/QCOtM5CVc5)。

- **Milvus Lite 在 Python 进程内运行**：Milvus Lite 与 LangChain 和 LlamaIndex 等 AI 开发栈无缝集成，非常适合资源受限的环境。将其整合到 AI 应用程序中的说明请参见[此处](https://milvus.io/docs/milvus_lite.md)。

- **使用 Omakase RAG Orchestrator 构建完整的 Web 应用**：推出了一个新的 Web 应用模板，用于使用 Django、LlamaIndex 和 Google Drive 构建可扩展的检索增强生成 (RAG) 应用程序。它具有完整的 RAG API、数据源管理和用户访问控制功能，[详情点击此处](https://t.co/kx3DhxfDZu)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://t.co/QCOtM5CVc5">Introducing Milvus Lite: Start Building a GenAI Application in Seconds</a>：未找到描述</li><li><a href="https://t.co/ckEiBVEbqK">Milvus Vector Store - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1245816948873494752)** (72 条消息🔥🔥): 

- **原型检索系统非常棘手**：一位用户分享了他们使用 SimpleStore 类原型化检索系统的经验，并寻求关于如何将数据转移到 RedisStores 的建议。其他人建议了多种方法，其中一位建议创建一个 "IngestionPipeline" 来处理 "transfer documents"（转移文档），以便更高效地处理 upserts 和数据传输。

- **ReAct Agent 响应问题**：一位成员的 RAG 应用最终输出过于简化，而中间观察过程却非常详细。建议包括使用 `ReActAgent.from_tools(..., context="Ensure your final answers are detailed.")` 提供额外的指令作为上下文。

- **OpenAI 证书验证问题**：一个使用 FastAPI 和 Nginx 的 Docker 化 OpenAI 设置面临 SSL 证书验证问题。一个建议是尝试使用不同的基础镜像来潜在地解决该问题。

- **理解向量存储选项**：用户讨论了 Postgres 中不同的向量存储查询选项，如 `DEFAULT`、`SPARSE`、`HYBRID` 和 `TEXT_SEARCH` ，并对它们的功能感到困惑。结论是 `text` 和 `sparse` 都使用 `tsvector`。

- **编辑 Document 对象**：一位成员寻求手动编辑 Document 对象的方法，特别是在 PDF 提取出现错误之后。社区建议直接修改 `document.text` 或使用外部文本编辑器并将编辑后的文本重新插入 Document 对象。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/?h=vectorstorequery#llama_index.core.vector_stores.types.VectorStoreQueryMode">Index - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/VespaIndexDemo/?h=vespa">Vespa Vector Store demo - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/vector_stores/postgres/?h=postgres+hybrid#improving-hybrid-search-with-queryfusionretriever">Postgres Vector Store - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1245823532458446970)** (20 条消息🔥): 

- **Luxia v1.2 确认存在数据污染**：一位成员指出，**Luxia 21.4b v1.2** 在 GSM8k 测试中的污染程度比 v1.0 增加了 29%。他们使用了广为人知的污染测试方法，并指出 ARC 和 Wino 等其他 Benchmark 的污染度为 0.0。
- **NeurIPS 模型合并竞赛**：NeurIPS 2023 宣布了一项与 Model Merging 相关的竞赛，有望在 Model Selection 方面取得突破。该活动旨在通过 8000 美元的奖金吸引参与者，并邀请社区成员[报名参加](https://llm-merging.github.io/)。
- **扩展联合生成器与奖励模型的想法**：一位成员讨论了扩展其模型的方案，该模型结合了用于 LM head 的 cDPO 和一种新型的 Reward head，用于根据奖励对生成的样本进行剪枝（Pruning），特别是针对毒性（Toxicity）和创造力（Creativity）等细粒度指标。
- **关于 Embedding 存储效率的咨询**：一位成员寻求关于如何高效存储数百万个 T5 Embedding 以进行大规模数据集共享的建议，并提到 fp16 配置下的 T5 XL Embedding 占用空间过大。他们考虑了量化（Quantization），虽然在保持约 95% 准确率的情况下将体积减半，但对于其需求来说仍然太大。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/LChoshen/status/1796256513519989102">来自 Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>: 🚨 @NeurIPSConf 的 Model Merging 竞赛！🚀 你能彻底改变 Model Selection 和 Merging 吗？让我们创造最好的 LLM！🧠✨ 💻 为科学而来 💰 为 8000 美元而留 💬 Discord: https://discord.gg/dPBH...</li><li><a href="https://huggingface.co/saltlux/luxia-21.4b-alignment-v1.2/discussions/1">saltlux/luxia-21.4b-alignment-v1.2 · GSM8K 上的污染结果 v1.0 vs v1.2</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1245835205697077329)** (34 条消息🔥): 

- **对 NeurIPS Model Merging 竞赛的兴奋**：成员们分享了关于在 **NeurIPS** 举办的 [Model Merging 竞赛](https://x.com/LChoshen/status/1796256513519989102) 的公告，奖金为 8,000 美元。感兴趣的各方被引导至竞赛的 [注册页面](https://llm-merging.github.io/) 和 Discord。

- **CLIP Text Encoder 的突破**：一篇声称拥有“不烂”的 CLIP 文本 [Encoder 的论文受到赞誉](http://arxiv.org/abs/2405.20204)。通过 Masked Language Modeling 对 Text Encoder 进行预训练，随后结合 Text-Image 和 Text-Text 对比损失（Contrastive Losses）的方法展现了良好的前景。

- **关于对齐论文——Direct Preference Heads 的讨论**：围绕一篇新的 [对齐论文](https://arxiv.org/abs/2405.20053) 及其使用 Direct Preference Heads 的独特方法展开了引人入胜的对话。讨论强调了它与 LaMDA 依赖 LM head 进行评分的不同之处，旨在将奖励信号与输出分布解耦。

- **用于 PDE 的 Poseidon 多尺度算子 Transformer**：介绍了一种名为 **Poseidon** 的新 Transformer 模型，用于学习 PDE（偏微分方程）的解算子，并取得了令人期待的结果。该 [模型](https://arxiv.org/abs/2405.19101) 在样本效率和准确性方面表现出色，通过一种新颖的训练策略在各种 PDE 任务上展现了强大的性能。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/LChoshen/status/1796256513519989102">来自 Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>: 🚨 Model Merging 竞赛 @NeurIPSConf!🚀 你能彻底改变模型选择和合并吗？让我们创造最好的 LLM！🧠✨ 💻为科学而来 💰为 8,000 美元留下 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://arxiv.org/abs/2201.08239">LaMDA: Language Models for Dialog Applications</a>: 我们介绍了 LaMDA：用于对话应用的语言模型。LaMDA 是一个专门为对话定制的基于 Transformer 的神经语言模型家族，拥有高达 137B 的参数，并在...上进行了预训练。</li><li><a href="https://arxiv.org/abs/2405.19101">Poseidon: Efficient Foundation Models for PDEs</a>: 我们介绍了 Poseidon，一个用于学习 PDE 解算子的基础模型。它基于多尺度算子 Transformer，具有时间调节的 Layer Norms，能够实现连续时间...</li><li><a href="http://arxiv.org/abs/2405.20204">Jina CLIP: Your CLIP Model Is Also Your Text Retriever</a>: 对比语言-图像预训练 (CLIP) 被广泛用于训练模型，通过将图像和文本映射到固定大小的向量，使它们在共同的嵌入空间中对齐。这些模型是多模态的关键...</li><li><a href="https://arxiv.org/abs/2405.18669">Zipper: A Multi-Tower Decoder Architecture for Fusing Modalities</a>: 将多个生成式基础模型（特别是那些在不同模态上训练的模型）整合在一起，使其产生大于部分之和的效果，面临着重大挑战。两个关键障碍是...</li><li><a href="https://arxiv.org/abs/2405.19893">Similarity is Not All You Need: Endowing Retrieval Augmented Generation with Multi Layered Thoughts</a>: 近年来，大语言模型 (LLM) 在各个领域取得了显著成就。然而，知识更新的不及时性和成本，加上 LLM 的幻觉问题...</li><li><a href="https://arxiv.org/abs/2405.20053">Would I Lie To You? Inference Time Alignment of Language Models using Direct Preference Heads</a>: 预训练语言模型 (LM) 表现出强大的 Zero-shot 和 In-context Learning 能力；然而，它们的行为通常难以控制。通过利用来自人类反馈的强化学习...</li><li><a href="https://x.com/sirbayes/status/1796441322263294435?s=46">来自 Kevin Patrick Murphy (@sirbayes) 的推文</a>: 我很高兴分享我们最近的论文：https://arxiv.org/abs/2405.19681。它可以被认为是贝叶斯学习规则的一个版本，扩展到了完全在线的设置。这是一个超级有趣的...
</li>
</ul>

</div>

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1245816087816306699)** (7 messages): 

- **Transformers 在结合 MLP 时表现卓越**：一位用户认为 **Transformers** 通过利用 MLP 实现了**最佳的数据依赖性（data dependence）**。他们强调了 MLP 在扩展（scaling）方面的优越性，并表示：*“任何其他数据依赖性如果不使用 MLP，在我看来在扩展规模时效果会更差。”*
- **Softmax Attention 辩论**：讨论质疑了在数据依赖聚合中 **Softmax 加权路由（softmax weighted routing）** 的必要性。一位成员指出，虽然存在替代方案，但由于经过了广泛的先前尝试，“Softmax Attention 具有 Lindy 效应（指经受住了时间的考验）”。
- **Softmax 的替代方案仍与当前方法相似**：一个反驳观点提到，取代 Softmax 并不会引入一种全新的机制，而是会形成一种维持 **上下文相关（context-dependent）的 T x T 矩阵** 的“函数注意力（function attention）”。这表明，真正不同的方法论可能在根本上仍与当前的 Attention 机制具有相似性。
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1246074642846711808)** (9 messages🔥): 

- **Gemma-2b-it 结果差异引发讨论**：一位用户报告称无法复现 Gemma-2b-it 17.7% 的结果，即使使用多数投票（majority voting）也仅达到约 9%。他们链接到了 [Hugging Face 上的讨论](https://huggingface.co/google/gemma-2b-it/discussions/44)，寻求他人的经验。

- **评估提示词（Evaluation prompt）仍然难以获取**：讨论涉及结果评估是否为 8-shot（如 Llama-2 所示）。一位用户指出，确切的评估提示词尚未发布，并建议查看 Mistral-7b 的评估以获取更多上下文。

- **使用 Colab 进行 GSM8K 评估**：分享了一个 [Google Colab notebook](https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/gsm8k_eval.ipynb#scrollTo=cXoCKMi9EXir) 链接以帮助复现结果，但该脚本需要登录以及特定的 CUDA 配置。

- **Phi3-mini 评估结果更一致**：一位用户提到，在 **lm_eval** 中运行 Phi3-mini 产生的结果与报告的数值更接近，差异较小。尽管注意到了其他不一致之处，这为评估过程提供了一点信心。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/github/google-deepmind/gemma/blob/">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/gsm8k_eval.ipynb#scrollTo=cXoCKMi9EXir">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 messages): 

gpantaz: 感谢回复 🙂
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1245816875036704780)** (13 条消息🔥): 

- **Mojo 包管理器仍处于早期阶段**：成员们讨论了 Mojo 路线图中关于包管理器更新匮乏的问题。他们引用了过去的 GitHub 讨论，如 [Discussion #413](https://github.com/modularml/mojo/discussions/413) 和 [Discussion #1785](https://github.com/modularml/mojo/discussions/1785)，这些讨论展示了关于项目清单（manifest）和构建工具的拟议计划。

- **编译器专业知识探索**：一位成员征求关于编译器学习材料的建议。另一位成员推荐了[这份教学大纲](https://mcyoung.xyz/syllabus)，其中包含了一份关于编译器、系统编程（systems programming）和其他主题的详尽阅读清单。

- **即将举行的 Mojo 社区会议**：宣布了第二次 Mojo 社区会议，会议将包含关于 Basalt、Compact Dict 和针对 Mojo 的 Pandas 的演讲，以及 Mojo Stdlib 的更新。详细信息已在 [Google Document](https://modul.ar/community-meeting-doc) 中提供，会议的 Zoom 链接[可在此获取](https://modul.ar/community-meeting-zoom)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://mcyoung.xyz/syllabus"> Syllabus &middot; mcyoung </a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/discussions">modularml/mojo · Discussions</a>：探索 modularml mojo 的 GitHub Discussions 论坛。讨论代码、提问并与开发者社区协作。</li><li><a href="https://github.com/modularml/mojo/discussions/413">[RFC] Allow Importing Modules via URLs · modularml/mojo · Discussion #413</a>：概述：Mojo 的主要优先级之一是解决“两种语言问题”，这意味着它必须既能用于应用程序开发用例，也能用于一次性脚本。依赖项 m...</li><li><a href="https://github.com/modularml/mojo/discussions/1785">[Proposal] Mojo project manifest and build tool · modularml/mojo · Discussion #1785</a>：大家好，请查看这个关于 Mojo 项目清单和构建工具的提案。正如提案本身所述，我们希望听到 Mojo 社区的声音：你是否同意这个动机...</li><li><a href="https://modul.ar/community-meeting-doc">[Public] Mojo Community Meeting</a>：Mojo 社区会议文档链接：https://modul.ar/community-meeting-doc 这是一个公开文档；欢迎所有人查看并发表评论/建议。所有会议参与者必须遵守...</li><li><a href="https://modul.ar/community-meeting-zoom.">Join our Cloud HD Video Meeting</a>：Zoom 是现代企业视频通信领域的领导者，拥有一个简单、可靠的云平台，用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天及网络研讨会。Zoom ...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot：来自 *Modular*：
<https://twitter.com/Modular/status/1796606227981726168>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1245962974536138803)** (3 条消息): 

- **Mojo 处理 SHA-256 和 256 位整数**：一位成员询问 **Mojo** 是否可以执行 SHA-256 哈希处理并管理 256 位整数。另一位成员确认，如果熟悉相关实现，这是可以做到的，并建议使用 `SIMD[DType.int64, 4]` 来表示 256 位整数。
  

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1245874454832222219)** (28 messages🔥): 

- **Mojo 加速 K-Means 聚类**：一段 [YouTube 视频](https://www.youtube.com/watch?v=3bg5YBCcuWA) 展示了将 K-Means 聚类从 Python+NumPy 移植到纯 Mojo 的分步指南，承诺带来 250 倍的加速。该视频被强调为学习 Mojo 的绝佳案例。

- **Mojo 旨在成为 Python 的超集**：一位用户指出，Modular 团队打算让 Mojo 成为 Python 的超集，使现有的 Python 代码能够无缝运行，并从 Cython 底层支持和 NumPy 兼容性等特性中受益。

- **Mojo 中的 Tuple 处理**：一位用户分享了从 tuple 中获取元素的代码，并就最近的更改寻求帮助，询问在遇到错误的情况下是否可能存在 bug。

- **Mojo 中的 ndarray 初始化方法**：一位成员就初始化 Mojo 中的 ndarray 应该使用 `memset` 还是向量化方法寻求建议。讨论指出 `memset` 可能更具优化性，因为它具有更底层的实现，正如其 [源代码](https://github.com/modularml/mojo/blob/bf73717d79fbb79b4b2bf586b3a40072308b6184/stdlib/src/memory/memory.mojo) 所示。

- **提升 Mojo 中的 string builder 性能**：一位用户分享了一个 [string builder 实现](https://github.com/thatstoasty/gojo/blob/nightly/gojo/strings/builder.mojo#L134)，声称它比字符串拼接显著更快，并寻求反馈以确保避免内存问题。另一位用户建议采用 zero-copy 方法，并提供了利用向量化写入（vectored writes）和避免不必要数据移动的见解，并参考了 [iovec 和 writev](https://man7.org/linux/man-pages/man2/writev.2.html)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=3bg5YBCcuWA">通过将 Python 实现移植到 Mojo🔥 来加速 K-Means 聚类</a>：在本视频中，我们将分享将 kmeans 聚类从 Python+NumPy 移植到纯 Mojo 的分步指南，以实现巨大的（250倍）加速！如何实现？Mojo 在...方面具有 Python 风格。</li><li><a href="https://github.com/modularml/mojo/blob/bf73717d79fbb79b4b2bf586b3a40072308b6184/stdlib/src/memory/memory.mojo">mojo/stdlib/src/memory/memory.mojo (位于 bf73717d79fbb79b4b2bf586b3a40072308b6184) · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/thatstoasty/gojo/blob/nightly/gojo/strings/builder.mojo#L134">gojo/gojo/strings/builder.mojo (位于 nightly 分支) · thatstoasty/gojo</a>：将 Golang 标准库移植到 Mojo 的实验。 - thatstoasty/gojo</li><li><a href="https://github.com/thatstoasty/gojo/blob/nightly/tests/test_performance.mojo#L7">gojo/tests/test_performance.mojo (位于 nightly 分支) · thatstoasty/gojo</a>：将 Golang 标准库移植到 Mojo 的实验。 - thatstoasty/gojo
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1245933316276883537)** (1 messages): 

- **澄清在 Max 中基于 Mojo 的模型训练**：一位成员询问，在 Mojo 中实现 backward pass、optimizer 以及像训练循环这样的基础训练处理，是否足以完全在 Max 中使用 Mojo 训练模型。他们试图了解这些组件是否是进行模型训练仅缺失的元素。
  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1245820819750391850)** (9 条消息🔥): 

- **调试的乐趣：发现 reversed Dict items 中的 Bug**：一位成员兴奋地分享了他们发现并修复了 `reversed(Dict.items())` 和 `reversed(Dict.values())` 中的一个未定义行为（undefined behavior）——这可能解决了困扰数周的不稳定测试（flaky tests）。他们附上了一个 [GitHub PR 链接](https://github.com/modularml/mojo/pull/2896)，详细说明了修复方案。
- **断言（Assertions）拯救世界**：另一位成员强调了在单元测试中启用断言的重要性，以避免不稳定测试，进一步强化了严谨调试实践的价值。
- **Mojo 编译器 Nightly 重大版本发布**：宣布了最新的 Mojo Nightly 编译器版本 `2024.5.3112`，其中包含多项更改，如更新日志修复、移除某些 `math` 函数以及重命名其他函数。详细更新内容可以在 [原始差异 (raw diff)](https://github.com/modularml/mojo/compare/8ae83916ebc7b3134948f466c0f56ee3e5569062...3df5fb4f9d3dd7cc5018aa8a160c3714b1a4f81e) 和 [当前更新日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 中找到。
- **Main 与 Nightly 分支的 PR 问题**：一次讨论表明，由于在 `main` 分支而非 `nightly` 分支上发起 PR 导致了一个 Bug，并引用了 [另一个 GitHub PR](https://github.com/modularml/mojo/pull/2881#issuecomment-2141082033) 作为证据。这一关键识别有助于简化未来的 PR 提交流程。
- **庆祝快速协助**：另一位成员幽默地表示，他们可能是第一个注意到并分享新更新兴奋点的人，展示了社区的参与度。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/pull/2896">[stdlib] Fix UB in `reversed(Dict.values())` and `reversed(Dict.items())` by gabrieldemarmiesse · Pull Request #2896 · modularml/mojo</a>：终于找到了过去几周困扰 test_reversed.mojo 的不稳定因素。实际的 Bug：在反向遍历列表时，我们应该从 len(my_list) -... 开始</li><li><a href="https://github.com/modularml/mojo/pull/2881#issuecomment-2141082033>">Update functions.ipynb by ratulb · Pull Request #2881 · modularml/mojo</a>：拼写错误 - current 修改为 currently。
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1245823184092270686)** (41 条消息🔥): 

- **为 LangGraph 项目公开暴露 LangChainAPI**：一位用户询问如何快速且廉价地将 LangChainAPI 端点暴露到公共位置，而不是保留在 localhost。他们对使用 LangServe 感兴趣，但尚未获得邀请。
  
- **加速 LangGraph 配置**：一位用户询问了加速 LangGraph 的配置选项，因为加载和启动单个 Agent 需要耗费大量时间。对话暗示正在寻找优化方案。

- **将摘要传入 ChatPromptTemplate**：一位用户询问如何将来自“memory”的摘要传递到 `ChatPromptTemplate` 中，并获得了使用带有适当变量名的 `MessagesPlaceholder` 的指导。分享了具体的实现细节和相关的 GitHub 链接以供参考。

- **将 ConversationSummaryMemory 与 RunnableWithMessageHistory 集成**：另一位用户寻求在 Python 中将 `ConversationSummaryMemory` 与 `RunnableWithMessageHistory` 集成的帮助。分享了详细的代码示例和 GitHub 资源来解释这一过程。

- **通过总结聊天记录减少输入 Token**：一位用户遇到 `RunnableWithMessageHistory` 因为聊天记录导致使用过多 Token 的问题。提供的解决方案涉及在通过链（chain）运行之前先总结聊天记录，以便更好地管理 Token 使用。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://js.langchain.com/v0.1/docs/modules/memory/types/summary/#usage-with-an-llm>)">对话摘要记忆 | 🦜️🔗 Langchain</a>：现在让我们来看看如何使用一种稍微复杂一点的记忆类型——ConversationSummaryMemory。这种类型的记忆会随着时间的推移创建对话的摘要。这对于压缩...非常有用。</li><li><a href="https://js.langchain.com/v0.1/docs/use_cases/chatbots/memory_management/#summary-memory>)">记忆管理 | 🦜️🔗 Langchain</a>：聊天机器人的一个关键特性是它们能够将之前对话轮次的内容作为上下文。这种状态管理可以采取多种形式，包括：</li><li><a href="https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/#dict-with-single-key-for-all-messages-input-messages-output>)">添加消息历史（记忆） | 🦜️🔗 LangChain</a>：RunnableWithMessageHistory 允许我们将消息历史添加到某些类型的链中。它封装了另一个 Runnable 并为其管理聊天消息历史。</li><li><a href="https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/#in-memory>)">添加消息历史（记忆） | 🦜️🔗 LangChain</a>：RunnableWithMessageHistory 允许我们将消息历史添加到某些类型的链中。它封装了另一个 Runnable 并为其管理聊天消息历史。</li><li><a href="https://python.langchain.com/v0.1/docs/use_cases/chatbots/memory_management/#summary-memory>).">记忆管理 | 🦜️🔗 LangChain</a>：聊天机器人的一个关键特性是它们能够将之前对话轮次的内容作为上下文。这种状态管理可以采取多种形式，包括：</li><li><a href="https://python.langchain.com/v0.1/docs/use_cases/chatbots/memory_management/#summary-memory>)">记忆管理 | 🦜️🔗 LangChain</a>：聊天机器人的一个关键特性是它们能够将之前对话轮次的内容作为上下文。这种状态管理可以采取多种形式，包括：</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/migrate_agent/">如何从旧版 LangChain agents 迁移到 LangGraph | 🦜️🔗 LangChain</a>：这里我们重点介绍如何从旧版 LangChain agents 迁移到 LangGraph agents。</li><li><a href="https://github.com/langchain-ai/langchain/issues/16525>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建具备上下文感知能力的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/#persistent-storage>)">添加消息历史（记忆） | 🦜️🔗 LangChain</a>：RunnableWithMessageHistory 允许我们将消息历史添加到某些类型的链中。它封装了另一个 Runnable 并为其管理聊天消息历史。</li><li><a href="https://github.com/langchain-ai/langchain/issues/1971>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建具备上下文感知能力的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/1136>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建具备上下文感知能力的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/16448>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建具备上下文感知能力的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1246026373563416596)** (5 条消息): 

- **RunnableLambda 解决了该问题**：一位用户分享了他们使用 **RunnableLambda** 解决问题的方法，建议将链封装到一个函数中，并创建一个具有输入和聊天历史属性的 Pydantic BaseModel 类。
- **LangServe 网站错误报告**：一位用户指出了 LangServe 网站的一个错误，并提供了此 [链接](https://www.langchain.com/langserve) 作为参考。
  

---


### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1245990593268682813)** (2 条消息): 

- **用户寻求构建包含多个变量的提示词的帮助**：一位成员询问如何从 **Langgraph state** 构建包含多个变量的提示词。他们提供了一个示例提示词：*"作为 {topic} 专家，请使用以下信息回答问题 {questions}：{knowledge}"*，并询问何时以及如何传递这些变量。
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1245862267321450588)** (2 条消息): 

- **学习 Crew AI 自定义工具**：查看这个标题为“Crew AI Custom Tools Basics”的 [YouTube 视频](https://youtu.be/Hc5yiUQKh2Q)。它涵盖了为 Agent 创建自定义工具，以实现任务自动化并提高 LLM 生成式 AI 的生产力。
  
- **AIQuire 发布文档洞察功能**：一位成员介绍了 AIQuire，这是一款由 AI 驱动的工具，旨在帮助用户理解复杂文档并从中提取答案。他们邀请其他人访问 [aiquire.app](https://aiquire.app) 试用 AIQuire 并提供反馈，以协助进一步开发。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aiquire.app/">AIQuire: 用数据智能赋能自己。</a>：未找到描述</li><li><a href="https://youtu.be/Hc5yiUQKh2Q">Crew Ai Custom Tools Basics.</a>：为你的 Agent 创建自定义工具是自动化任务并使你的 LLM 生成式 AI 变得更高效的最佳方式。Crew Ai Docume...
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1245820019741560923)** (48 条消息🔥): 

- **Fineweb 的巨大潜力**：讨论集中在实现 Fineweb，利用 Puppeteer 访问 URL 并将图像与文本内容一同转储，用于上下文输入以增强 VLM 的接地（grounding）能力。[Fineweb 详情](https://vxtwitter.com/iurimatias/status/1796260746310910309)。

- **StabilityAI 的决定引发反响**：StabilityAI 计划发布一个 512px 的模型，而不是全套的 SD3 checkpoint。成员们讨论了这一决定可能如何影响模型改进，并就高 GPU 资源的必要性分享了看法。

- **DiT 模型中的位置嵌入 (Positional Embeddings)**：关于位置嵌入在 DiT 模型中如何工作及其处理不同分辨率潜力的技术探讨。成员们指出，尽管有标准实现，位置嵌入在更高分辨率下往往会出现模式崩溃 (mode collapse)。

- **开源工具让社区感到兴奋**：尽管存在一些小问题，像 tooncrafter 这样的开源项目凭借新功能让社区感到兴奋。讨论强调了社区对快速改进这些工具的乐观态度。

- **Eliezer Yudkowsky 研究所的 AI 策略**：Eliezer Yudkowsky 的研究所发布了“2024 年沟通策略”，旨在作为一种预防措施停止 AI 开发。[阅读更多](https://x.com/drtechlash/status/1796562490232557658?s=46&t=M3cR_nfDo7QCuM4xOvwNFA)。

**提到的链接**：<a href="https://x.com/drtechlash/status/1796562490232557658?s=46&t=M3cR_nfDo7QCuM4xOvwNFA">来自 Nirit Weiss-Blatt, PhD (@DrTechlash) 的推文</a>：Eliezer Yudkowsky 的研究所发布了其“2024 年沟通策略”。主要目标（正如他在《时代》杂志上所主张的）是 🔻停止🔻 AI 开发。所以，让我们来看看...

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1245835029599223818)** (2 条消息): 

- **NeurIPS 将举办模型合并 (Model Merging) 竞赛**：成员们宣布了 [NeurIPS 的模型合并竞赛](https://x.com/LChoshen/status/1796256513519989102)，感兴趣的参与者有机会赢取 **$8K** 奖金并为 LLM 创新做出贡献。查看[官方 Discord](https://discord.gg/dPBHEVnV) 和 [报名页面](https://llm-merging.github.io/) 了解更多详情。
- **用于图像风格化和构图的 RB-Modulation**：介绍了一种名为 **RB-Modulation** 的新方法，为内容-风格图像的风格化和构图提供了一种无需训练、即插即用的解决方案。更多详情可在[官方项目页面](https://rb-modulation.github.io/)查看，包括[论文](https://arxiv.org/abs/2405.17401)和[代码（即将推出）](https://github.com/LituRout/RB-Modulation)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/LChoshen/status/1796256513519989102">来自 Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>：🚨 NeurIPSConf 模型合并竞赛！🚀 你能彻底改变模型选择和合并吗？让我们创造最好的 LLM！🧠✨ 💻为科学而来 💰为 $8K 而留 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://rb-modulation.github.io/">RB-Modulation</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1245912628388302869)** (22 messages🔥): 

- **探索 Huggingface 上的 Yuan2 模型**：成员们讨论了来自 Huggingface 的 [Yuan2 模型](https://huggingface.co/papers/2311.15786)，表达了对其进行训练的兴趣，并分享了相关链接以供进一步研究。
  
- **偏好训练方法对比**：关于各种训练方法的讨论中，成员提到 *HF 建议先对数据进行 SFT 然后再进行 DPO*，并将其与 *ORPO* 进行对比，后者可以 *“独立完成，且效果比 DPO 结合 SFT 更强”*。提供了 ORPO 论文的链接：[arxiv.org/abs/2403.07691](https://arxiv.org/abs/2403.07691)。

- **ORPO 相比传统方法的优势**：成员们讨论了 ORPO 的优点，强调它能够消除对额外偏好对齐阶段的需求。这被描述为 *“一种无需参考模型的单体偏好优化 (monolithic preference optimization)”*。

- **在 Axolotl 中集成 ORPO 的可能性**：有人询问是否将 ORPO 添加到 Axolotl 中，一名成员建议 ORPO *“应该可行”*，表明了在系统中进行尝试的可能性。

**提到的链接**：<a href="https://huggingface.co/papers/2311.15786">Paper page - YUAN 2.0: A Large Language Model with Localized Filtering-based Attention</a>：未找到描述。

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1245918973522870373)** (12 messages🔥): 

- **文本分类微调的困扰**：一位用户在对西班牙语实体分类进行模型微调时遇到困难，使用的是 llama3 和 mistral。他们提供了具体的指令步骤，并指出虽然训练成功，但推理表现不佳。

- **推理设置方面的协助**：另一位用户询问是否可以分享数据集和推理方法，以便更好地理解微调中出现的问题。原发布者尚未回复更多细节。

- **CUDA 12.1 安装困扰及解决方案**：一名成员在 Ubuntu 22 上安装带有 NVIDIA 驱动的 CUDA 12.1 时遇到困难。解决方案是使用 run 文件安装 CUDA 12.1，且不包含驱动程序安装。

- **微调 embedding 模型的框架**：一名成员寻求关于适合高效微调 RoBERTa 风格 embedding 模型的框架建议。该查询目前尚未得到解答。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1246108724414910547)** (2 messages): 

- **关于 LLama3 格式化的困惑**：一位用户询问在使用 Alpaca 格式配合 LLama3 时，`text` 字段是否应包含像 `<|start_header_id|>` 这样的特定 token。另一名成员建议在配置中设置 `chat_template`，因为 Axolotl 会自动处理格式化。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1246115875023487038)** (6 messages): 

- **Axolotl 安装问题**：一名成员报告在安装 **Axolotl** 过程中遇到错误。尽管遵循了常规的故障排除步骤（如确保 Python 版本兼容性和使用虚拟环境），问题仍然存在。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/README.md#L102L137)">axolotl/README.md at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=8b79a987-f932-4434-89c2-978e36c820b0)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1245886949827543150)** (7 messages): 

- **成员寻求解决模型过拟合的帮助**：*"模型开始过拟合了，我需要停止训练，该怎么做？"*。另一位成员建议实现早停机制 (Early Stopping mechanism)，并提供了一个使用 **Hugging Face Accelerate 库**的详细代码示例。
- **早停配置的 GitHub 链接**：一位成员分享了一个 [GitHub 链接](https://github.com/OpenAccess-AI-Collective/axolotl/blob/d4f6c65e4c4d6899a9283a046fa922b937a0ab25/docs/config.qmd#L325C1-L327C27)，用于在 **OpenAccess-AI-Collective/axolotl** 中配置早停机制的进一步说明。提问者表示感谢并回复：*"谢谢，一定会尝试一下！"*。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/d4f6c65e4c4d6899a9283a046fa922b937a0ab25/docs/config.qmd#L325C1-L327C27">axolotl/docs/config.qmd at d4f6c65e4c4d6899a9283a046fa922b937a0ab25 · OpenAccess-AI-Collective/axolotl</a>：欢迎提出 axolotl 相关问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=aa1cb115-b5f9-438c-b03e-4b76a0f862dc)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。
</li>
</ul>

</div>
  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1245862057585147925)** (5 messages): 

- **DiscoLeo 需要重新训练以修复 EOS Token 问题**：一位成员报告称，将 **ChatML 与 Hermes Theta 等 SOTA 模型合并**会导致 EOS Token 出现问题，使得模型有 20% 的概率陷入死循环。他们建议使用 ChatML 重新训练 DiscoLeo，并请求微调数据。
- **相比 Llama-3 模板更倾向于 ChatML**：成员们讨论了对 **ChatML 优于原始 Llama-3 instruct 模板**的偏好。有人认为在德语基座上使用 ChatML 进行微调更好，尤其是考虑到目标模型 (Hermes Theta) 已经使用了 ChatML。
  

---


### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1246032456503529523)** (23 messages🔥): 

- **对 IBM 的 Granite 模型感兴趣**：成员们询问了 IBM 新的 "Granite" 模型在英语和德语中的性能和可用性。包括 **Lab 版本**在内的 IBM 模型已列在 [watsonx.ai](https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x?topic=models-foundation) 上，由于在开源社区中提及较少，引发了大家的好奇。
- **对 IBM 模型版本的困惑**：讨论强调了对 IBM 系列 **Granite 模型**的困惑，例如 3B/8B 增强版 Llama 版本和基于 Starcoder 的 20B/34B 模型。一位用户指出了可用的各种版本，包括 **7B base、instruct 以及 instruct accelerator 版本 (medusa speculative decoding)**。
- **Merlinite 模型引起关注**：**Merlinite 7B** 模型因其有趣的特性被注意到，并提到其已上传至 Ollama 并在 **Lab 方法**下进行了测试。用户表示有兴趣将其在德语方面的能力与 Granite 等模型进行比较。
- **生成数据与基准测试**：用户对 **AI 生成数据**的质量表示担忧，一些用户指出其结果大多不尽如人意。提到的基准测试如 **q4km gguf 量化版的 EQ Bench** 低于普遍预期，这突显了大家对比较新的“在不产生灾难性遗忘的情况下进行增强 (enhancing without catastrophic forgetting)”方法的兴趣。
- **社区资源共享**：分享了一个关于 [InstructLab 的 Medium 文章](https://medium.com/@syeda9118/instructlab-ever-imagined-the-ease-of-tuning-pre-trained-llms-3331ccea8d88) 链接，总结了微调预训练 LLM 的简便性，反映了社区在更好地理解和实施这些模型方面所做的持续努力。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://ollama.com/sroecker/granite-7b-lab">sroecker/granite-7b-lab</a>：使用 LAB 方法训练的 Granite 7B 模型</li><li><a href="https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x?topic=models-foundation">IBM 构建的基础模型</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1245815214771933336)** (10 messages🔥): 

- **Google 的新算力资源引发关注**：针对 Google-Apple 交易的传闻，一条帖子分享了 [Twitter 链接](https://x.com/_arohan_/status/1796228607255396423)，暗示 Google 正在扩大其算力资源。链接中提到另一个集群已运抵，表明训练 AI 模型的容量有所增加。

- **法庭判决及其影响**：用户幽默地讨论了一起备受关注的法庭案件，引用了 #miscarriage-of-justice 和 #they-hate-him-cuz-they-aint-him 等标签。人们对判决可能如何影响未来的政治事件和潜在的骚乱企图表示担忧。

- **OpenAI 重启机器人团队**：OpenAI 在 2020 年放弃初步尝试后，正式重新组建了其机器人团队。这一举措在 [Twitter](https://x.com/sarahnemerson/status/1796264432215146807?s=46) 和 [Forbes 文章](https://www.forbes.com/sites/kenrickcai/2024/05/30/openai-robotics-team/) 中均有报道，提到新团队已活跃约两个月，目前正在招聘研究工程师。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_arohan_/status/1796228607255396423">Tweet from rohan anil (@_arohan_)</a>: @giffmana No worries, another cluster has arrived today in case you want to train some more</li><li><a href="https://x.com/sarahnemerson/status/1796264432215146807?s=46">Tweet from sarah emerson (@SarahNEmerson)</a>: New --  OpenAI is formally resurrecting its robotics team after abandoning efforts to build general purpose robots in 2020.   Its new team has been around for ~2 months and is currently hiring researc...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1245815628762451988)** (7 messages): 

- **对 GPT-3.5 API 可用性的质疑**：一位用户评论了“任何人都可以基于 GPT-3.5 API 构建聊天机器人”这一反复出现的观点，并对其频繁使用和准确性提出了批评。 

- **GPT-3.5 时间线的澄清**：另一位用户澄清说，虽然 GPT-3.5-002 的可用时间更长，但 GPT-3.5-003 是与 ChatGPT 同时或之后推出的，这表明公众对可用性的理解存在偏差。

- **对已删除文档的困惑**：用户对 GPT-3.5 相关页面的删除表示沮丧，并称其命名方案令人困惑。一位用户表示他们几个月前就报告了这个问题，但未做任何更改。

- **对信息删除的担忧**：一位用户表示，为了迎合某种便利的叙事而删除信息是有问题的。另一位用户暗示这可能是一种典型的 AI 安全措施，而另一位用户则认为这可能是一个疏忽，并提出分享该网站的存档版本。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1245875840441384981)** (8 messages🔥): 

- **Sergey 为 Physical Intelligence 招人**：Nathan Lambert 分享说 Sergey 曾试图招募他加入一个关于 Physical Intelligence 的项目，并提到他们有“很酷的配置”。他表示欢迎其他人加入，并说“如果有人感兴趣请告诉我”。
- **对 RL 和发表论文的兴趣**：一位社区成员表示，如果有机会讨论 RL（强化学习）和论文发表，他们有兴趣加入，尽管他们对能否跟上团队的专业水平感到紧张。
- **研究支持和机器人使用**：Nathan Lambert 安慰说团队很务实且会进行调整，并支持研究，因为需要更多的人使用机器人。这位潜在成员幽默地怀疑自己是否合适，提到他们唯一的机器人经验就是扫地机器人。
  

---


### **Interconnects (Nathan Lambert) ▷ #[retort-podcast](https://discord.com/channels/1179127597926469703/1238601424452059197/1246143273311932456)** (2 messages): 

- **Murky Waters 播客集发布**：Nathan Lambert 和 Tom 在名为 [Murky Waters in AI Policy](https://retortai.com/episodes/murky-waters-in-ai-policy) 的新播客集中讨论了近期发生的各种 AI 政策事件。主题包括加州的“反开源” 1047 法案、参议院 AI 路线图、Google 搜索故障、OpenAI 的动态以及读者反馈。
- **错过了加州法案的开放日**：Nathan Lambert 提到他原计划参加加州 AI 法案的开放日，但未能成行。上述消息中未提供更多细节。



**Link mentioned**: <a href="https://retortai.com/episodes/murky-waters-in-ai-policy">The Retort AI Podcast | Murky waters in AI policy</a>: Tom and Nate catch up on many AI policy happenings recently. California's 

  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1245920575201738823)** (11 条消息🔥): 

- **Cohere 致力于长期可持续性**：一名成员评论道：“如果将长期可持续性作为衡量标准，Cohere 可能遥遥领先。”他们强调了专注于特定任务（如从发票中提取信息）的价值，而不是立即解决治愈疾病等宏大问题。
  
- **AGI 仍处于起步阶段**：讨论强调实现 AGI 仅仅是开始，并提出了一个问题：“然后呢……我们仍处于‘然后呢……’的阶段。”AI 的现状更多被视为一个“CPU”，而非一个完整的系统。

- **服务器更新与新的社区激励**：一名成员宣布了服务器的更新，简化了频道布局并引入了新的角色和奖励。他们提到：“我们正在取消服务器等级，取而代之的是，最活跃的服务器成员将直接成为 Cohere 常驻用户（regulars）。”

- **Cohere 推出 AI 聊天机器人 Coral**：一次测试确认了 AI 聊天机器人 Coral 已上线并运行，其回复为：“我是 Coral，一个旨在通过提供详尽回复来协助人类用户的 AI 聊天机器人。”这次成功的互动赢得了测试者的赞赏。

- **用于社区互动的全新表情符号和贴纸**：服务器将推出新的表情符号和反应，成员可以通过联系管理员自定义表情符号。这一变化旨在增强社区内的用户互动和趣味性。
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1245920587512021056)** (2 条消息): 

- **AI 模型无代码工作流构建器征求反馈**：一家初创公司正在开发一款**无代码工作流构建器（no-code workflow builder）**，旨在混合和匹配 AI 模型，并计划在未来实现自动构建工作流和自动选择最佳 LLMs 的功能。他们正在征求关于用户为何停止使用该平台的反馈，并为调查参与者提供 **10 美元的奖励**。 
- **社区鼓励与支持**：一名成员赞扬了这种做法，并在不参与奖励的情况下提供了反馈，表示愿意在平台上花费 10 分钟。他们对招募帖子的形式和礼仪表示赞赏，认为这是参与社区互动的一种好方法。
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1245843049108607036)** (12 messages🔥): 

- **Embedding Adapters 提升检索性能**：成员们讨论了 Embedding Adapters 的潜力（将其称为*“检索的捷径”*），并分享了 [Chroma 研究报告](https://research.trychroma.com/embedding-adapters)的链接。该报告评估了将线性变换应用于 Embedding 以改进检索应用的效果。

- **Frozen Embeddings 类似于 Embedding Adapters**：另一场讨论将 Embedding Adapters 比作 Vespa 团队使用的 Frozen Embeddings，并引用了 [Vespa 博客文章](https://blog.vespa.ai/leveraging-frozen-embeddings-in-vespa-with-sentence-transformers/)。Frozen Embeddings 有助于避免在电子商务等动态环境中对 Embedding 进行繁琐的更新。

- **普华永道（PwC）巨额 ChatGPT Enterprise 合同**：一条推文强调了普华永道为约 100,000 名员工购买了 ChatGPT Enterprise 许可证，估计合同金额为每年 3000 万美元。成员们对价格进行了辩论，猜测范围从每用户每月 8 美元到之前传闻的每用户每月 65 美元不等。

- **Google Gemini 更新发布**：Google 开发者宣布 Gemini 1.5 Flash 和 1.5 Pro 正式发布（GA），包括 Flash 的 1,000 RPM 限制、新的微调选项以及 API 中的 JSON Schema 模式。更多详情可以[在此查看](https://developers.googleblog.com/en/gemini-15-pro-and-15-flash-now-available/)。

- **TLBrowse 开源**：TLBrowse 将 Websim 与 TLDraw 合并，已由其创作者开源。用户可以在无限的 @tldraw 画布上生成想象中的网站，并提供[免费托管版本](https://tlbrowse.com)供试用。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://blog.vespa.ai/leveraging-frozen-embeddings-in-vespa-with-sentence-transformers/">在 Vespa 中结合 SentenceTransformers 利用 Frozen Embeddings</a>：如何在 Vespa 中使用 SentenceTransformers 库实现 Frozen Embeddings 方法，并同时优化您的搜索应用。</li><li><a href="https://agent.ai">agent.ai | AI Agents 的专业网络</a>：在 AI Agent 的帮助下完成工作</li><li><a href="https://x.com/officiallogank/status/1796213739366322431?s=46&t=90xQ8sGy63D2OtiaoGJuww">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：@Google 开发者的好消息：- Gemini 1.5 Flash 和 1.5 Pro 现已正式发布 - Gemini 1.5 Flash 现在有 1,000 RPM 限制 - 宣布 Gemini 1.5 Flash 微调功能 - JSON Schema 模式现已在...可用</li><li><a href="https://x.com/sawyerhood/status/1796193457662214651?s=">Sawyer Hood (@sawyerhood) 的推文</a>：我刚刚开源了我的 tlbrowse 演示！您可以在无限的 @tldraw 画布上生成想象中的网站。</li><li><a href="https://x.com/tanayj/status/1795858607004598351?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tanay Jaipuria (@tanayj) 的推文</a>：普华永道为约 100,000 名员工购买 ChatGPT Enterprise 许可证，成为 OpenAI 最大的 ChatGPT 企业用户。假设每席位约 25 美元/月，这是一份每年 3000 万美元的合同。</li><li><a href="https://research.trychroma.com/embedding-adapters">Embedding Adapters</a>：未找到描述</li><li><a href="https://x.com/sawyerhood/status/1796193457662214651?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Sawyer Hood (@sawyerhood) 的推文</a>：我刚刚开源了我的 tlbrowse 演示！您可以在无限的 @tldraw 画布上生成想象中的网站。</li><li><a href="https://tlbrowse.com">tlbrowse</a>：未找到描述
</li>
</ul>

</div>
  

---



### **AI Stack Devs (Yoko Li) ▷ #[events](https://discord.com/channels/1122748573000409160/1131651713204498583/1245835549294461041)** (1 messages): 

- **Rosebud AI 举办“书改游戏” Game Jam**：来自 Rosebud AI 团队的 Roberto 宣布了一项新的 Game Jam 活动——“Book to Game”，在他们的 AI Game Maker 平台上使用 Phaser JS 进行开发。该活动鼓励参与者根据文学作品创作互动游戏，设有 **500 美元奖金池**。
- **详情与参与方式**：Game Jam 的提交截止日期为 PST 时间 7 月 1 日凌晨 12:00。更多详情和参与指南可在 [Rosebud AI 的 Twitter](https://x.com/Rosebud_AI/status/1796273820044595368) 及其 [Discord](https://discord.gg/rosebud-ai) 上找到。


**提及的链接**：<a href="https://x.com/Rosebud_AI/status/1796273820044595368">来自 Rosie @ Rosebud AI 🌹 (@Rosebud_AI) 的推文</a>：使用 AI 将你最喜欢的故事变成游戏！📚 👾 为我们的第三次 Game Jam 做好准备：“Book to Game”。使用 Rosebud Game Maker 将文学作品转化为互动游戏，让故事焕发生机...

  

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1245816064525205554)** (5 messages): 

- **Manufacturing update pinned**: 制造更新已置顶：成员们被引导查看 <#1194880263122075688> 中的置顶消息以获取制造更新。通过置顶消息保持信息同步至关重要。
- **Interest in Codestral model**: 对 Codestral 模型的兴趣：一位成员询问是否有人尝试过 Codestral，并表示它“看起来是一个不错的模型”。这表明社区内对探索新模型的兴趣日益增长。
- **Struggles with HuggingFace integration**: HuggingFace 集成方面的困难：一位成员表达了在 OpenInterpreter 中使用 HuggingFace 模型的挫败感，指出仅通过命令 `interpreter -y --model huggingface/mistralai/Mistral-7B-Instruct-v0.3` 成功过一次。另一位成员建议在 <#1149558876916695090> 中发布详细帖子以寻求进一步帮助。
- **Potential scam alert**: 潜在诈骗警报：一位成员发布了关于潜在诈骗的“红色警报”，并提醒另一位用户注意。这强调了社区警惕性的重要性。
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1245835801821188188)** (6 messages): 

- **Codestral model generates buzz**: Codestral 模型引发热议：一位成员询问是否有人尝试过 Codestral，提到它看起来是一个很有前途的模型。目前关于用户体验和见解的请求尚未得到回应。
- **Query on O1 Android functionality**: 关于 O1 Android 功能的查询：多位成员表达了对运行 O1 Android 的兴趣，其中一人询问是否需要安装在 Termux 中。该查询尚未得到答复。
- **License limitations noted**: 注意到许可证限制：一位成员强调 Codestral 的使用仅限于非商业用途。这一点被提出但未展开进一步讨论。


  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1245915925404975135)** (3 messages): 

- **AutoGPT integrates Llamafile**: AutoGPT 集成 Llamafile：来自 AutoGPT 的一位成员宣布与另一位用户合作，将 **Llamafile** 集成到他们的系统中。
- **Questions about content block support**: 关于内容块支持的问题：同一位成员询问 **Llamafile** 是否支持用户消息中的 **内容块 (content blocks)**，类似于 OpenAI 的功能。另一位成员询问 **llama.cpp** 是否支持它们。
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1245825241570345061)** (2 messages): 

- **PRS Event at Netflix Attracts Attention**: Netflix 的 PRS 活动吸引关注：一位成员宣布他们明天将参加在 **Netflix 举办的 PRS 活动**，并询问是否还有其他人参加。另一位成员确认他们也会参加。
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1245819417783173230)** (2 messages): 

- **Curiosity Sparks Over Mistral 45GB Model**: 对 Mistral 45GB 模型的好奇：一位成员推测了 45GB 模型的构成，认为它可能在英语方面权重较高，而在编程语言方面的比例较小。他们表示很期待看到实际的构成细分。

- **MNPL Compliance Dilemmas for Codestral**: Codestral 的 MNPL 合规困境：一位成员对在 **Mistral AI Non-Production License (MNPL)** 下寻找 Codestral 的合法用例表示担忧。**MNPL** 似乎限制了与他人共享衍生作品或托管作品，该成员认为这具有限制性且令人失望。
  

---



### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

helplesness: 为什么 TensorFlow 比 PyTorch 更好？
  

---



---



---



{% else %}


> 完整的频道细分内容已为邮件版截断。
> 
> 如果你想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}