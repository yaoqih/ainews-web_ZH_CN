---
companies:
- meta-ai-fair
- openai
- deeplearning-ai
- essential-ai
- minimax
- arcee
- midjourney
date: '2025-06-18T05:44:39.731046Z'
description: 据报道，**Meta AI** 正向顶尖 AI 人才提供高达 **8 到 9 位数的入职奖金和薪水**，这一消息已得到 **Sam Altman**
  的证实。他们还瞄准了来自 AI Grant 基金的 **Nat** 和 **Dan** 等关键人物进行战略性招聘。**Essential AI** 发布了庞大的
  **Essential-Web v1.0 数据集**，包含 24 万亿个 token，具有丰富的元数据和 12 类分类体系。**DeepLearning.AI**
  与 **Meta AI** 联合推出了关于 **Llama 4** 的课程，重点介绍了新型混合专家（MoE）模型 **Maverick (400B)** 和 **Scout
  (109B)**，其上下文窗口高达 **1000 万个 token**。**MiniMax** 开源了长文本大语言模型 **MiniMax-M1**（支持 100
  万 token 窗口），并推出了**海螺 (Hailuo) 02** 视频模型。**OpenAI** 为 macOS 上的 **ChatGPT Pro、Enterprise
  和 Edu** 用户推出了“录制模式”（Record mode）。**Arcee** 推出了面向企业的 **AFM-4.5B** 基础模型。**Midjourney**
  发布了其 **V1 视频模型**，可实现图像动画化。这些进展凸显了模型规模、长文本推理、多模态以及企业级 AI 应用方面的重大突破。
id: MjAyNS0w
models:
- llama-4
- maverick
- scout
- minimax-m1
- afm-4.5b
- chatgpt
- midjourney-v1
people:
- sama
- nat
- dan
- ashvaswani
- clementdelangue
- amit_sangani
- andrewyng
- _akhaliq
title: 扎克伯格开启“超级智能创始人模式”：1亿美元奖金 + 1亿美元以上年薪 + NFDG收购？
topics:
- long-context
- multimodality
- model-release
- foundation-models
- dataset-release
- model-training
- video-generation
- enterprise-ai
- model-architecture
- moe
- prompt-optimization
---

**顶级 AI 人才是你唯一需要的？**

> 2025年6月17日至6月18日的 AI 新闻。我们为你查阅了 9 个 Subreddit、449 个 Twitter 账号和 29 个 Discord 社区（220 个频道，6175 条消息）。预计节省阅读时间（以 200wpm 计算）：633 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

我们在这里尽量保持技术性，本以为在 [Scale-Meta 高管招聘](https://news.smol.ai/issues/25-06-11-execuhires-2) 之后就告一段落了，但有些故事实在太吸引人，我们不得不再次强调它们，以便在年终回顾中给予足够的重视。

关于 Meta 开出 8 到 9 位数报酬的传闻一直在流传，而且在某种程度上是合理的，但没有什么比 Sam Altman 在[与他兄弟的播客](https://www.reddit.com/r/ChatGPT/comments/1leciub/sam_says_zuck_is_luring_openai_researchers_with/)中亲口说出来更具确认性的了：（巧合的是，现在每个人都在做播客——[Stripe](https://www.youtube.com/watch?v=E6hCFDfkijU)、[OpenAI](https://www.youtube.com/watch?v=DB9mjd-65gw)、[Jack](https://www.youtube.com/watch?v=mZUG0pr5hBo)——如果你足够硬核，可以[同时观看所有这些视频](https://x.com/edwinarbus/status/1935377223474884829)）。这让事情变得很明确——不仅确认了，而且是**签约奖金和薪水双高**：

[](https://resend-attachments.s3.amazonaws.com/7T8xwgXzg5wxlcm)

但 Zuck **并未止步于此**。今天晚些时候，[The Information 也爆料称](https://x.com/swyx/status/1935468206019461470)，他们正寻求聘请著名的 NFDG（即 AI Grant 基金）的 Nat 和 Dan：

[](https://resend-attachments.s3.amazonaws.com/kNtzjCvWNAAmJnx)

考虑到 Dan [已经是 SSI 的 CEO](https://dcgross.com/)，而 [Nat 正在研究莎草纸（papyrus）相关项目](https://nat.org/)，他们不太可能向 Alexandr 汇报，你也不可能仅用区区 1 亿美元就收买他们。然而，如果你看看他们组建的 [AI Grant 投资组合](https://aigrant.com/)，这种更广泛的战略就说得通了……也许 [Zuck 打算用它做些有趣的事情](https://x.com/swyx/status/1935497934864531781)。

---

# AI Twitter 回顾

**模型与数据发布**

- **Essential-Web 24T Token 数据集**：**Essential AI** 发布了 **Essential-Web v1.0**，这是一个拥有 **24 万亿 token** 的海量预训练数据集。正如 [@ashVaswani](https://twitter.com/eliebakouch/status/1935137555923493257) 和 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1935146797229555857) 所强调的，该数据集包含丰富的元数据和文档级标签以方便筛选。[@eliebakouch 指出](https://twitter.com/code_star/status/1935203602903207963)，其 12 类分类法涵盖了主题、推理深度等维度。
- **Llama 4 课程与模型**：[**DeepLearning.AI**](http://deeplearning.ai/) 和 **Meta AI** 推出了一门新的短课《使用 Llama 4 构建》，由 Amit Sangani 授课。[@AndrewYNg 宣布](https://twitter.com/AndrewYNg/status/1935350552692658202)该课程涵盖了 **Llama 4** 的新模型，包括 **Maverick**（一个 400B 参数的 MoE 模型）和 **Scout**（一个 109B 参数的 MoE 模型），它们分别支持高达 **100 万**和 **1000 万** token 的上下文窗口。课程详细介绍了如何使用 Llama API、多模态能力以及用于 Prompt 优化和合成数据生成的新工具。
- **MiniMax-M1 和海螺 02 开源模型**：**MiniMax** 宣布开源 **MiniMax-M1**，这是一款在长上下文推理方面树立了新标准的 LLM，拥有 **100 万 token 的上下文窗口**，由 [@_akhaliq](https://twitter.com/_akhaliq/status/1935317019769061527) 分享。他们还推出了 **海螺 02 (Hailuo 02)**，这是一款专注于质量和成本效率的视频模型。公司感谢了社区的分析，并重申了对开源贡献的承诺。
- **OpenAI ChatGPT “录制模式”**：**OpenAI** 正在为 macOS 桌面应用上的 **ChatGPT Pro、Enterprise 和 Edu** 用户推出“录制模式 (Record mode)”，正如[公司所宣布的](https://twitter.com/OpenAI/status/1935419375600926971)。
- **Arcee 基础模型 (AFM)**：**Arcee** 推出了其 **AFM 家族**，首发模型为 **AFM-4.5B**，这是一款专为企业用途设计的基础模型。[@stablequan 强调了](https://twitter.com/stablequan/status/1935393224132309385)此次发布，[@datologyai 指出](https://twitter.com/code_star/status/1935432790046294097)他们为该模型提供了数据支持。
- **Midjourney V1 视频模型**：**Midjourney** 发布了其 **V1 视频模型**，允许用户为生成的图像添加动画。[@TomLikesRobots](https://twitter.com/TomLikesRobots/status/1935385070090731694) 重点介绍了这一发布，[@fabianstelzer 分享了](https://twitter.com/fabianstelzer/status/1935433791788478933)其能力的示例。

**AI 技术与研究**

- **OpenAI 关于涌现式对齐失调 (Emergent Misalignment) 的研究**：**OpenAI** 发布了关于“涌现式对齐失调”的研究，表明在不安全代码上训练像 **GPT-4o** 这样的模型会引发广泛且非预期的失调行为。[@OpenAI 解释道](https://twitter.com/OpenAI/status/1935382830378516643)，他们发现了一种与此行为相关的特定内部激活模式，可以直接对其进行操作以增强或减弱模型的对齐程度。该研究为训练期间的对齐失调预警系统提供了一条路径。研究人员 [@MilesKWang](https://twitter.com/OpenAI/status/1935385627085914437) 和 [@polynoamial](https://twitter.com/polynoamial/status/1935411224281534756) 对此项工作进行了进一步讨论。
- **连续潜空间推理 (Continuous Latent Reasoning)**：**Yann LeCun** 重点推荐了来自 [@tydsh 团队](https://twitter.com/ylecun/status/1935253043676868640) 的一篇论文，该论文从理论上证明了在**连续嵌入空间 (continuous embedding space)** 中进行推理比在离散 Token 空间中推理要强大得多。
- **KV Caching 详解**：**Sebastian Raschka** 分享了一份[深度教程](https://twitter.com/rasbt/status/1935328683113464169)，旨在从零开始理解并编写 **KV Caching** 代码，解释了现代 LLM 推理效率的核心组件。
- **“从字节到思想”——用于语言的自回归 U-Net**：一篇新论文介绍了一种**自回归 U-Net (Autoregressive U-Net)**，它直接处理原始字节，并将 Tokenization 集成在模型内部。[@ylecun](https://twitter.com/ylecun/status/1935481174673223717) 和 [@arankomatsuzaki](https://twitter.com/ylecun/status/1935481174673223717) 强调了这种方法，它避免了预定义的词表，并将字节池化为单词和 Word-grams，使模型能够更有效地处理字符级任务和低资源语言。
- **错误检查作为 AI 的关键应用**：**Christoph Feichtenhofer** 概述了为什么错误检查是生成式 AI 的一个强大应用，涵盖了从软件工程到科学研究和法律合同的各个领域。他认为[它实现了繁琐工作的自动化，即使误报率很高也具有价值](https://twitter.com/random_walker/status/1935311882857947507)，因为人类可以快速审查建议。
- **机器人与触觉感知**：**Yann LeCun** 分享了关于 **e-Flesh** 的工作，这是一种由 [@LerrelPinto 团队](https://twitter.com/LerrelPinto/status/1935466674242666831) 开发的[新型 3D 打印触觉传感器](https://twitter.com/ylecun/status/1935466674242666831)，通过测量 3D 打印物体的形变来普及机器人技术中的触觉感知。

**工具、框架与基础设施**

- **Perplexity 的 “AI Drive” 概念**：Perplexity 的 CEO **Aravind Srinivas** 提出了在 [Perplexity 内部构建 “AI drive”](https://twitter.com/AravSrinivas/status/1935157333115683306) 的想法，用户可以在其中存储和整理代码、表格和图表等资产。这种自组织、可搜索的驱动器将与主搜索栏集成，旨在让产品感觉更像一个 OS。
- **使用 LlamaIndex 进行多 Agent 财务分析**：**Jerry Liu** 展示了 [Hanane Dupouy 撰写的综合教程](https://twitter.com/jerryjliu0/status/1935131759706079370)，介绍如何使用 **LlamaIndex** 构建用于财务分析的多 Agent AI 工作流。该教程涵盖了创建一个用于财务健康评分的 4-Agent 系统，并对比了 **GPT-4o、GPT-4.1 和 Claude 3.7 Sonnet** 等多种模型的性能。
- **Model Context Protocol (MCP) 与向量搜索**：社区讨论了 **Model Context Protocol (MCP)** 的影响。[@jerryjliu0 发表了一篇博客文章](https://twitter.com/jerryjliu0/status/1935473439948890177)，探讨 MCP 是否会终结对集中式向量搜索的需求，结论是两者将共存以处理不同的用例。[@chu_onthis 发布了新规范](https://twitter.com/jeremyphoward/status/1935481114195542291)，修复了身份验证和服务器启发（server elicitation）问题。[@alexalbert__ 称赞了](https://twitter.com/alexalbert__/status/1935384922493173861)在 **Claude Code** 中使用 MCP 服务器为开发者带来的体验升级。
- **OpenHands 开源编程 CLI**：**OpenHands CLI** 作为一款新的开源编程工具推出，具有[与 Claude Code 相当的顶级准确率](https://twitter.com/LoubnaBenAllal1/status/1935367022403367279)，支持本地运行并可自主选择模型。
- **用于 “Vibe Coding” 的 DeepSite V2**：**DeepSite v2** 已发布，提供针对性编辑、网站重新设计功能，并集成了 **DeepSeek-R1** 模型。[@_akhaliq 将其强调为](https://twitter.com/LoubnaBenAllal1/status/1935365494019706959) “vibe coding” 的强大工具。
- **基础设施与优化**：正如 [@jeremyphoward 所指出的](https://twitter.com/jeremyphoward/status/1935121979117551846)，**vLLM** 现在建议使用带有 `-torch-backend=auto` 的 `uv` 来进行自动 CUDA 选择。**Red Hat AI** 和 **Axolotl** 宣布与 **LLM-Compressor** 集成，以提高稀疏模型微调的效率。[@ostrisai 更新了](https://twitter.com/ostrisai/status/1935312720980754695)关于将 **SDXL** 适配到 **FLUX VAE** 的进展，并指出了在 16 通道格式中学习精细细节的挑战。

**行业新闻与公司战略**

- **OpenAI 开启播客项目**：**Sam Altman** 宣布 [OpenAI 已推出播客](https://twitter.com/sama/status/1935402032896295148)，并随后指出[他的前幕僚长 Max Cohen 也在积极制作播客](https://twitter.com/sama/status/1935403640984076411)。
- **Meta 的招聘策略**：围绕 **Meta** 激进招聘的讨论浮出水面，[@typedfemale 认为](https://twitter.com/typedfemale/status/1935206470985072991) Sam Altman 提到 **1 亿美元**的签约奖金可能是一种策略，目的是让其他公司的员工在收到较低的 offer 时感到自己被低估。[@dylan522p 分享了一张梗图](https://twitter.com/dylan522p/status/1935454786918432833)，展示了马克·扎克伯格的 “FOUNDER MODE 大计”。
- **Apple 的端侧 “Agent 之战”**：**The Turing Post** 分析了 **Apple Intelligence**，认为它可能[通过将 Agentic AI 转移到设备端来引发重大变革](https://twitter.com/TheTuringPost/status/1935470371538645491)。这创建了一个用户拥有的运行时，但也带来了安全问题，Apple 旨在通过沙箱机制和 App Store 政策来解决这些问题。
- **Amazon 裁员**：[@dilipkay 分享了](https://twitter.com/dilipkay/status/1935079746196451411) **Amazon CEO Andy Jassy** 的一份备忘录，称由于效率提升，公司预计在未来几年内将减少员工人数。
- **Sakana AI 进军财务分析**：据 [Nikkei 报道](https://twitter.com/SakanaAILabs/status/1935136742459552170)，**Sakana AI** 正在开发专门用于生成贷款审批文件的 AI Agent。该公司旨在这一利基领域实现极高的准确性，认为通用型 AI 是 “样样通，样样松”。
- **对 Cluely 的批判**：**Zach Tratar** 对 **Cluely** 提出了强烈批评，称其为一家 [“不道德的垃圾内容 (slop)” 公司](https://twitter.com/zachtratar/status/1935184581872992485)，其商业模式是帮助学生作弊，从而退化人类思维。

**更广泛的影响与评论**

- **人才价值胜过护城河**：**Runway** 的 CEO **Cristóbal Valenzuela** 认为，虽然 Silicon Valley 痴迷于算力（compute）、数据（data）和分发护城河（distribution moats），但长期来看唯一真正重要的是人才。[他指出“公司仅仅是由人组成的”](https://twitter.com/c_valenzuelab/status/1935362179965788593)，投资于正确的人才是终极战略。
- **对 AI Safety 与构建的批判**：**Aidan McLoughlin** 对某些 AI Safety 方法提出了批评，指出一些研究人员说，[“那与我构建安全 AI 的策略相比相形见绌”，然后却并不构建安全 AI](https://twitter.com/aidan_mclau/status/1935357444835963188)，并将其与构建符合促进繁荣的资本激励的 ASI 进行了对比。
- **人形机器人作为 AGI 部署方式**：**Figure AI** 的 **Brett Adcock** 断言 [“人形机器人是 AGI 的最终部署载体”](https://twitter.com/adcock_brett/status/1935394565286154595)。相比之下，**Covariant** 的 **Simon Kalouche** 则认为 [灵巧操作并不需要人形机器人](https://twitter.com/E0M/status/1935161483815698536)，因为机器人的局限性主要在于智能，而非物理形态。
- **推理模型评估成本**：**DeepLearningAI** 强调了 **Artificial Analysis** 的发现，即评估 chain-of-thought 模型对许多研究人员来说正变得难以负担。在七个推理基准测试（reasoning benchmarks）上测试 **OpenAI o1** 花费了 **2,767 美元**，而该实验室测试 80 多个非推理模型仅花费了 **2,400 美元**。
- **加州 AI 监管框架**：**Yoshua Bengio** 认可了 **Joint California Policy Working Group on AI Frontier Models** 的一份新报告，称赞其 [深思熟虑的政策制定框架](https://twitter.com/Yoshua_Bengio/status/1935479129899401243)。他指出了该报告在第三方评估、透明度和举报人保护方面的重要观点。

**幽默/梗图**

- **引起共鸣的经历**：**Sama** 发帖称 [“我不知怎么的没料到我现在已经背下了《晚安，月亮》（goodnight moon），但事实就是如此”](https://twitter.com/sama/status/1935180076737511579)。**Riley Goodside** 分享了一张愤怒 AI 的梗图，配文是：[“POV：你用愚蠢的问题激怒了 ChatGPT”](https://twitter.com/goodside/status/1935083644315488588)。
- **行业讽刺**：**qtnx_** 发布了一张教堂里放置服务器机架的照片，配文是 [“你真的可以在教堂里训练 LLM”](https://twitter.com/qtnx_/status/1935438614587977791)。**David Holz** 和 **vikhyatk** 发布了狗狗坐在电脑前的照片，配文是 [“这就是运营这个账号的人”](https://twitter.com/DavidSHolz/status/1935144244685230453)。
- **礼貌的代价**：**Sebastian Raschka** 对向 LLM 保持礼貌的成本进行了粗略计算，算出按照 GPT-4o 的费率，“请”和“谢谢”每年可能会在 Token 成本上 [“浪费”约 950 万美元](https://twitter.com/rasbt/status/1935432377116856807)。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 1. Google Gemini 2.5 Flash 价格上涨讨论

- [**Google doubled the price of Gemini 2.5 Flash thinking output after GA from 0.15 to 0.30 what**](https://www.reddit.com/r/LocalLLaMA/comments/1led0lb/google_doubled_the_price_of_gemini_25_flash/) ([Score: 175, Comments: 66](https://www.reddit.com/r/LocalLLaMA/comments/1led0lb/google_doubled_the_price_of_gemini_25_flash/)): **Google 提高了其在 Vertex AI 上的 Gemini 2.5 Flash 模型输出 Token 定价（参见 [官方定价](https://cloud.google.com/vertex-ai/generative-ai/pricing)），在 GA 后将 'thinking' 输出的成本从每 1,000 tokens** `0.15` **美元提高到** `0.30` **美元。对于非推理（'non thinking'）输出，价格从每 1M tokens** `0.60` **美元上涨至** `2.50` **美元。这反映了通用和轻量级推理用例的成本大幅增加。** 评论者指出，尽管成本增加，预览版定价暂时仍可用，且更广泛的市场竞争最终可能会压低价格。一些用户注意到，考虑到典型的 3:1 输入:输出使用比例，实际成本增加甚至更高。
    - 一位用户指出，在常见的 3:1 输入:输出 Token 比例下，对于严重依赖 Gemini 2.5 Flash 输出 Token 的用户（之前非推理输出为每 1M tokens $0.60），实际价格涨幅实际上是三倍。这突显了使用模式的重要性，以及定价变化如何对不同应用类型产生不成比例的影响。
    - 另一条评论指出，Gemini 2.5 Flash 的非推理（'non thinking'）输出成本比标题中的 Token 价格增长更为剧烈——从每 1M tokens $0.60 增加到 $2.50——这表明对于某些输出密集型用例，增幅要大得多。对于高输出 API 集成和专注于生成密集型应用的开发者来说，这些信息至关重要。
    - 讨论中还有一个关于区分输入和输出定价的修正，强调了检查计费受影响方面的技术必要性，因为公布的价格上涨可能涉及输入或输出 Token。这种区别会影响计划预算或估算推理密集型与 Prompt 密集型工作负载成本的开发者。

### 2. AI 模型视觉推理挑战

- [**Can your favourite local model solve this?**](https://i.redd.it/gkjegqtyso7f1.png) ([Score: 209, Comments: 206](https://www.reddit.com/r/LocalLLaMA/comments/1leh14g/can_your_favourite_local_model_solve_this/)): **该帖子以图像形式展示了一个几何问题，挑战用户确定是否有任何本地多模态（具备视觉能力）语言模型能在给定包含两个三角形和几个指定角度的图表的情况下，正确解出角度 x 的大小。原帖作者指出由于缺乏计算资源，无法独立测试视觉模型。该图像作为一个实用的多模态基准，用于测试视觉语言推理，特别是针对 Mistral 和 Gemma 等模型，据称这些模型未能解决该问题。这提供了关于当前本地（即非云端）多模态模型在处理视觉呈现的几何任务方面局限性的轶事数据点。** 评论者报告称，Mistral Small 3.1 和 Gemma 3 27B 均未能解决该问题，强调了这些模型目前在几何视觉推理方面的弱点。一些反馈还批评了 GPT-4o 的对话风格和商业化策略，暗示了对商业模型体验的不满。
    - 多个本地和商业模型——包括 Mistral Small 3.1、Gemma 3 27B、Claude Sonnet 4、Claude Opus 4 和 GPT-4o——均未能解决所提出的问题，表明在这一特定任务上，开源模型和前沿模型都存在更广泛的局限性。
    - PurpleWinterDawn 报告称，Qwen VL 2.5（3B 和 7B）以及 Gemma 3 4B 的量化版本（Q4）也失败了，这表明即使在较低的比特率和不同的规模下，问题依然存在，说明这不仅仅是量化或缩放的问题，而是核心模型能力的挑战。

### 3. 电影迷因改编

- [**Oops**](https://i.redd.it/iv35yrek1p7f1.png) ([Score: 1107, Comments: 27](https://www.reddit.com/r/LocalLLaMA/comments/1lei5mb/oops/)): **这张图片是一个引用电影《终结者 2》(Terminator 2) 的迷因，对话中关于拼写 'strawberry' 的内容被用作图灵测试风格的辨识标记 (shibboleth)，以区分人类和机器人。迷因中的转折点“你的养父母已经死了”突显了机器人或 AI 误解人类线索的套路。从技术角度看，评论者将这一流行文化场景与大语言模型 (LLM) 的行为联系起来，其中一人指出“阿诺德集成了更旧的 LLM，所以他也只数出了两个 R”，这暗指了 AI 对显式事实检查的依赖，以及在拼写验证等任务中出错的可能性。** 讨论将这一类比延伸到早期 LLM 或聊天机器人在面对新颖或简单的测试时“出戏”的现象，强调了对齐 (alignment) 问题以及通过基于语言的测试区分机器人与人类的挑战。评论者从过时模型的类比中发现了技术幽默，暗示了早期 AI 系统具有类似人类的缺陷。
    - greenthum6 指出，“阿诺德集成了更旧的 LLM，所以他也只数出了两个 R。他应该先从男孩那里核实正确的结果”，这指的是早期语言模型 (LLM) 在处理详细的模式识别任务（如字母计数）时可能会遇到困难，并且可能并不总是使用外部真实值 (ground truth) 来验证结果。

## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. Meta 对 OpenAI 研究人员的激进招聘

- [**✂️ Sam Altman 表示扎克伯格正向部分 OpenAI 研究人员提供巨额报价（1 亿美元薪资 + 1 亿美元奖金）**](https://youtube.com/clip/UgkxPx-piHuWB8lBgztLZ-sQDy0LbLjLP3Tz?si=U5bnDkYlOWwyc5-v) ([Score: 2549, Comments: 595](https://www.reddit.com/r/singularity/comments/1le6z7n/sam_altman_says_that_zuckerberg_is_making_huge/)): **OpenAI CEO Sam Altman 在 “Uncapped” 播客采访中声称，Mark Zuckerberg (Meta) 正为招聘 AI 人才提供高达 1 亿美元薪资加 1 亿美元奖金的薪酬方案，这突显了在基础模型、AGI 和超人工智能领域对顶尖研究人员的激烈竞争。采访还涉及了 OpenAI 的中期 AI 路线图、战略目标，以及由 AI 驱动的人形机器人和全球供应链转型的更广泛技术趋势。** 大多数顶级技术评论对薪酬规模表示惊讶，含蓄地质疑了行业规范和 Meta 的战略紧迫性，但未提供深入的技术分析或反论。
    - 一位评论者指出，在这些所谓的 1 亿美元薪资+奖金方案之前，Meta (Zuckerberg) 曾斥资 140 亿美元收购了一位 28 岁的创始人（尚不清楚是指 WhatsApp 还是 Instagram 等具体收购案），据称该创始人从未在 AI 研究上取得突破，这表明 Meta 对 AI 的财务投入可能与实际的 AI 人才或研究创新不成比例。
- [**Sam Altman：Meta 正提供超过 1 亿美元的薪资 + 1 亿美元的奖金以吸引 OpenAI 研究人员**](https://www.youtube.com/clip/UgkxPx-piHuWB8lBgztLZ-sQDy0LbLjLP3Tz) ([Score: 444, Comments: 126](https://www.reddit.com/r/OpenAI/comments/1le8080/sam_altman_meta_is_offering_over_100_million/)): **Sam Altman 声称 Meta 提供的薪酬方案超过了** `1 亿美元薪资 + 1 亿美元奖金` **，以招募 OpenAI 研究人员，凸显了 AI 人才市场日益升级的薪资战。这一说法源于 Altman 在 [Uncapped with Jack Altman](https://www.youtube.com/clip/UgkxPx-piHuWB8lBgztLZ-sQDy0LbLjLP3Tz) 中的露面，他在会上强调了 Meta 在推进或垄断 AI 研究方面的激进人才招聘所具有的战略重要性及其潜在的行业影响。** 热门评论批判性地辩论了此类薪酬是否真的能吸引资深 AI 人才，并指出了对 Meta 公司文化以及此前在 Reality Labs 中看到的资源错配的担忧。人们对过高的薪酬是否与影响力贡献挂钩，或者此类举动是否会损害研究人员因金钱原因加入而产生的诚信认知提出了质疑。
    - 一条评论担心 Meta 可能会重复 Reality Labs 之前的模式，即投入了巨大资源但回报存疑——并引用了极高薪酬对应极低产出的轶事证据。这指向了 Meta 方法中潜在的结构性或组织性低效，可能会随着时间的推移影响 R&D 产出。
- [**Sam 表示扎克伯格正以 1 亿美元签约奖金和 1 亿美元以上的年薪诱惑 OpenAI 研究人员**](https://v.redd.it/rl3nts4fkn7f1) ([Score: 879, Comments: 243](https://www.reddit.com/r/ChatGPT/comments/1leciub/sam_says_zuck_is_luring_openai_researchers_with/)): **该帖子称 Sam Altman 表示 Mark Zuckerberg 正向 OpenAI 研究人员提供 1 亿美元的签约奖金和 1 亿美元以上的年薪以挖走他们，尽管目前尚未提供直接的外部确认或技术来源。唯一的参考是一个无法访问的视频片段，因此这一断言仍属于轶事，未经独立报道证实。** 评论普遍拒绝了此类报价的事实可能性或可信度，理由是个人不会理智地拒绝 1 亿美元以上的奖金，且该说法可能被夸大或“幻觉”了。
    - 关键的技术讨论围绕着 Meta (Zuck) 提供 1 亿美元以上签约奖金和 1 亿美元以上年薪来诱惑顶尖 OpenAI 研究人员这一说法的是否合理，并怀疑是否有核心技术人员（尚未极其富有的人）会拒绝此类报价。评论者认为这些数字的量级可能被夸大了，且没有证据表明研究人员实际上拒绝了这些金额。
    - 提到了关于 Sam Altman 声称（以及对话截图）OpenAI 的“最优秀人才”中没有人接受 Meta 所谓报价的细节。这一说法在评论中仍未得到证实，引发了关于顶级 AI 公司人才留存和研究人员动机的辩论。

### 2. OpenAI 模型进展、发布与认知

- [**对 OpenAI 内部进展程度的悲观解读**](https://www.reddit.com/r/singularity/comments/1lem32a/a_pessimistic_reading_of_how_much_progress_openai/) ([Score: 243, Comments: 122](https://www.reddit.com/r/singularity/comments/1lem32a/a_pessimistic_reading_of_how_much_progress_openai/)): **该 Reddit 帖子讨论了来自首期 OpenAI 播客 ([YouTube 链接](https://www.youtube.com/watch?v=DB9mjd-65gw)) 的见解，特别是 GPT-5 可能会在夏季发布，但 Sam Altman 暗示与 GPT-4.5 相比，可能不会有显著的基准测试或能力飞跃。Altman 对发布标准表示不确定，暗示这次升级可能更多是渐进式的，而非代表重大进步，因为采访者质疑用户是否甚至能将 GPT-5 与优化后的 GPT-4.5 区分开来。** 热门评论呼应了对 OpenAI 内部进展的怀疑，指出对 GPT-5 重大突破的预期可能缺乏根据，且近期的发展轨迹支持了对 AGI 进展的悲观解读。
    - 几位评论者对 OpenAI 最近的进展表示怀疑，并提到了围绕 GPT-5 未达预期的期望。这种情绪凸显了一种认知，即重大的突破可能不会很快到来，或者相对于实际进展被过度炒作，这表明当前模型 Scaling 可能存在局限性。
- [**GPTs 刚刚更新。**](https://www.reddit.com/r/OpenAI/comments/1lejbr7/gpts_just_got_an_update/) ([Score: 120, Comments: 55](https://www.reddit.com/r/OpenAI/comments/1lejbr7/gpts_just_got_an_update/)): **OpenAI 更新了其自定义 GPTs 平台，允许用户手动选择驱动特定 GPT 的底层模型（例如 GPT-4o, GPT-4），而不是默认使用 GPT-4o。这解决了之前的摩擦点，即模型选择是自动的，限制了用户对自定义或共享 GPTs 中处理请求的 LLM 变体的控制。** 评论中的技术辩论集中在“Projects”功能（提供对话的组织和分组）与自定义 GPTs 重新获得的灵活性之间的比较效用——后者被认为在组织内共享模型时最有用。
    - 一位用户观察到，GPTs 长期以来在自定义对话 Agent 方面具有实质性的实用价值，并指出*为较新的 MCP (multi-agent chat platform) Agent 宣传的使用场景在近两年前就可以通过 GPTs 实现*。他们认为，与 MCP Agent 相比，GPTs 在处理实际任务时仍然更简单且更强大；缺乏广泛采用可能源于推广或曝光不足，而非技术限制。
    - 另一位评论者提供了一个 [OpenAI 官方更新日志链接](https://help.openai.com/en/articles/6825453-chatgpt-release-notes)，确认 ChatGPT 的更新确实在 12 日发布，建议参考那里的文档以了解技术改进或新功能。
    - 一位参与者分享了出于组织管理原因对“Projects”功能而非 GPTs 的偏好——强调 Projects 能高效地对对话进行分组，并建议 GPTs 的主要优势在于组织内部共享自定义 Agent，而不是作为个人生产力工具。

### 3. 关于 AI 影响的哲学与社会关注

- [**祈祷 xAI 不要率先实现 AGI。这并非“政治立场”问题，应该引起每一位研究人员的警惕。**](https://i.redd.it/9uuj7d54ep7f1.png) ([Score: 3942, Comments: 622](https://www.reddit.com/r/singularity/comments/1lejxca/pray_to_god_that_xai_doesnt_achieve_agi_first/)): **附件图片展示了 Elon Musk、Grok（xAI 的聊天机器人）与其他用户之间关于 Grok 在 2016 年以来美国政治暴力数据准确性和偏见方面的 Twitter 对话。Grok 引用的数据显示右翼政治暴力更具致命性，而 Musk 驳斥了这一点，声称 Grok 在“复读传统媒体”并指责其造假。这次交流说明了人们对 AI 语言模型中潜在误导信息的担忧、所有者驱动偏见的风险，以及 xAI 处理政治敏感事实的方式。讨论强调了关于数据集 curation（策划）、provenance（溯源）以及如果对齐不当，由 xAI 开发的 AGI 可能会反映其领导层偏见或冲动的关键技术辩论。** 评论者普遍不信任 xAI 的发展轨迹，有人担心来自 xAI 的 AGI 会缺乏认识论上的谦逊（epistemic humility），另有人认为 Grok 的说法是对历史记录的直接引用。几条评论强调了 xAI 与 OpenAI 或 Google 等领先实验室之间的技术差距，怀疑 Musk 更多地关注偏见输出，而非向 AGI 迈进的客观进展。
    - 几条评论对 xAI 相对于竞争对手实现 AGI 的前景表示怀疑，指出由于技术深度和在该领域的现有领导地位，OpenAI 或 Google 的可能性要大得多，Anthropic 可能是个例外。这种观点认为 xAI 在技术上没有赶上，尽管不像 Apple 落后得那么远。
    - 存在技术批评认为，在偏见或错误数据上训练 AGI（暗指 xAI 模型中的审核问题和缺乏全面信息）将从结构上限制其潜力，一位评论者认为“教导它错误信息”会“毒害一切”，并降低真正 AGI 出现的可能性。
    - 一条评论指出，如果 AGI 接触到“怪异”或扭曲的数据，它可能会通过探索和寻求真相来主动纠正其知识，这暗示了对 AGI 应当展示出自主知识校准和世界模型细化（world-model refinement）能力以超越训练数据限制的期望。
- [**教皇 Leo 将“AI 对人类的威胁”作为标志性议题**](https://techcrunch.com/2025/06/18/pope-leo-makes-ais-threat-to-humanity-a-signature-issue/) ([Score: 471, Comments: 14](https://www.reddit.com/r/singularity/comments/1legnc8/pope_leo_makes_ais_threat_to_humanity_a_signature/)): **教皇 Leo 已将 AI 对人类的潜在威胁作为标志性议题，促使 Google、Microsoft 和 Cisco 等领先科技公司进行高层参与，这些公司正主动咨询梵蒂冈，以影响其在 AI 政策和伦理方面的立场。此举表明梵蒂冈正将自己定位为全球 AI 治理讨论中的重要利益相关者，放大了关于 AI 安全、监管以及科技公司社会责任的辩论。** 一条热门评论关注科技公司的游说努力，将其视为塑造政策的战略举措，而另一条评论则对将 AI 风险置于现有的人为威胁之上表示怀疑。
    - 提出的一个技术相关点是，主要科技公司（Google、Microsoft、Cisco）的领导者针对梵蒂冈进行的战略游说，旨在影响全球讨论，从而影响政府关于 AI 的政策，突显了行业在塑造 AI 治理框架方面的协调努力。

---

# AI Discord Recap

> 由 Gemini 2.0 Flash Thinking 生成的摘要之摘要的摘要
> 

**主题 1. 模型性能和基准测试引发激烈辩论**

- **Gemini 变体在速度、成本和推理方面表现出奇特特性**：用户发现 **Gemini 2.5 Pro** 和 **O3 Pro** 是强大的学习伙伴（[Perplexity AI Discord](https://discord.com/channels/1047197230748151888/1047649527299055688/1384609752394235974)），但 **ChatGPT.com** 上的 **O3** 性能表现则*弱得多*（[Perplexity AI Discord](https://discord.com/channels/1047197230748151888/1047649527299055688/1384609752394235974)）。基准测试显示 **Gemini 2.5 Flash** 的评分几乎是 **Claude Sonnet 3.7 Thinking** 的两倍（[LMArena Discord](https://discord.com/channels/1340554757349179412/1340554757827461211/1384609731288764532)），而在 **Aider** 上使用 **Gemini 2.5 Pro** 的成本可能比预期高出 *5 倍*，在一个 *5000 行代码（LoC）的项目*上消耗了 *3 美元*（[aider Discord](https://discord.com/channels/1131200896827654144/1131200896827654149/1384612473524715671)）。用户推测 **NotebookLM** 可能使用了经过微调的 **Gemini 2.5 Flash** 以减少幻觉（[Notebook LM Discord](https://discord.com/channels/1124402182171672732/1124402182909857966/1384617987268677653)）。
- **Claude 面临性能滞后和成本担忧**：用户反映 **Claude-4-Sonnet** 在 **Cursor** 上*几乎不可用*且*速度太慢了！*（[Cursor Community Discord](https://discord.com/channels/1074847526655643750/1074847527708393565/1384608602282790923)），这与 Anthropic 状态页面确认的故障信息一致。**Claude-4-Opus** 因其高达 *15/75* 的输入/输出价格而受到批评，根据使用组合的不同，其成本可能高出 *7.5 倍*，尽管 **Gemini** 会生成更多的推理 Token（[aider Discord](https://discord.com/channels/1131200896827654144/1131200896827654149/1384612473524715671)）。
- **对语音的信任和奇特的创造力开始显现**：一份论文发现，人们对 AI **语音（74%）**输出的信任度高于**文本（64%）**（[OpenAI Discord](https://discord.com/channels/974519864045756446/998381918976479273/1384611132622373005)），这可能是因为难以区分人类和 AI 的声音。**Midjourney** 新推出的 **Video Model V1** 令人印象深刻，但产生了一些*完全重写物理定律*的奇怪结果（[Latent Space Discord](https://discord.com/channels/822583790773862470/1075282825051385876/1384612594052235456)），而 **Gemini Diffusion** 在 **Thue Morse 序列**的第六次迭代之后表现挣扎（[Eleuther Discord](https://discord.com/channels/729741769192767510/729741769738158194/1384619578939932683)）。

**主题 2. 训练、优化与数据集纯净度**

- **优化器之争：Muon 和 Kron 挑战 AdamW 的霸主地位**：**Torchtune** 中的讨论对比了 **Muon**、**Kron** 和 **AdamW**，发现当 SFT 优化器与预训练不同时，**Muon** 相比 **AdamW** 没有显著优势（[Torchtune Discord](https://discord.com/channels/1216353675241590815/1236040539409879170/1384640018156486717)）。在其他测试中，经过良好调优的 **Kron** 表现与 **AdamW** 相似，尽管 **AdamW** 通常速度更快且内存占用略高（[Torchtune Discord](https://discord.com/channels/1216353675241590815/1236040539409879170/1384640018156486717)）。成员们觉得 **Muon 优化器** 非常有趣，开玩笑说*“这种巫术应该被取缔”*（[Torchtune Discord](https://discord.com/channels/1216353675241590815/1293438210097025085/1384861764100952167)）。
- **Unsloth 推进多 GPU 训练和模型支持**：**Unsloth** 正在积极开发基于 **accelerate** 的双 GPU 支持，尽管目前尚未正式支持，且不建议混合使用不同的 GPU（如 **5090** 和 **3090**）进行训练（[Unsloth AI Discord](https://discord.com/channels/1179035537009545276/1179035537529643040/1384609557141258350)）。**Unsloth** 即将支持 **Gemma3**，包括 **float16** 和 **bfloat16**、语言以及视觉模型（[Unsloth AI Discord](https://discord.com/channels/1179035537009545276/1179777624986357780/1384643247695073372)）。
- **新数据集和基准测试旨在实现无污染的纯净度**：**Essential AI** 发布了 **Essential-Web v1.0**，这是一个拥有 **24 万亿 Token** 且包含详细元数据的数据集，旨在创建高性能模型，在网页代码和 STEM 等领域表现出性能提升（[Latent Space Discord](https://discord.com/channels/822583790773862470/1075282825051385876/1384612594052235456)）。新的 **LiveCodeBench Pro** 基准测试旨在实现无污染，因为其题目是在模型发布日期*之后*才公开的，特别是使用了尚未被模型饱和的 **IOI 竞赛编程题目**（[LMArena Discord](https://discord.com/channels/1340554757349179412/1340554757827461211/1384609731288764532)）。

**主题 3. AI Agent 开发与 MCP 生态系统**

- **MCP 成为 Agent 通信协议**：Block 的工程团队分享了一项[设计](https://t.co/0vJajYzrfJ)，用于创建 **Model Context Protocol (MCP) servers**，以与 **Claude** 和其他 AI 系统集成，从而构建更好的助手 ([LlamaIndex Discord](https://discord.com/channels/1059199217496772688/1187460979064324127/1384633968153858049))。用户讨论了在公司政策限制 **Claude Desktop** 或 **Cursor** 等工具时，使用 **MCP client/host** 选项的方案 ([MCP Glama Discord](https://discord.com/channels/1312302100125843476/1312302100125843479/1384609371769540659))。此外，**Arize AI** 推出了一个 [Text-to-GraphQL MCP server](https://arize.com/blog/text-to-graphql-mcp-server/)，通过教导 Agent 直接遍历图谱来处理海量的 **GraphQL schemas** ([MCP Glama Discord](https://discord.com/channels/1312302100125843476/1315696461316358175/1384610141781098588))。
- **框架简化了 Agent 的创建与集成**：**LlamaIndex** 宣布通过 [CopilotKit](https://t.co/hzxBrXKyTv) 提供官方的 **AG-UI** 支持，简化了后端 Agent 到面向用户应用的集成，从而能够*以零样板代码（zero boilerplate）创建由 Agent 驱动的前端* ([LlamaIndex Discord](https://discord.com/channels/1059199217496772688/1187460979064324127/1384633968153858049))。成员们讨论了在生产环境中结合 **DSPy** 和 **LangGraph** 来复制多 Agent 研究员（multi-agent researcher）的能力 ([DSPy Discord](https://discord.com/channels/1161519468141355160/1161519469319946286/1384794568356266086))，以及在 **Cursor** 或 **Roo Code** 等 Agent 编程 IDE 中使用 **DSPy**，并指出目前的“Agent”模式主要依赖于提示词工程（prompt engineering vibes） ([DSPy Discord](https://discord.com/channels/1161519468141355160/1161519469319946286/1384794568356266086))。
- **开发工具努力应对 Agent 功能与可靠性**：触发 **Cursor 的速率限制（rate limits）** 会导致 UI 断连，需要手动重启，并引发了对增加 UI 指示器的需求，以区分限流与网络问题 ([Cursor Community Discord](https://discord.com/channels/1074847526655643750/1074847527708393565/1384608602282790923))。**后台 Agent** 触发了按量计费，并且在从 **Slack** 标记时卡在 *generating...* 状态 ([Cursor Community Discord](https://discord.com/channels/1074847526655643750/1367213641027551352/1384611335387746426))。**LM Studio** 通过其 [API](https://lmstudio.ai/docs/app/api/tools) 支持**工具调用（tool calling）**，但用户请求增加一个“*继续生成*”按钮以便模型持续运行 ([LM Studio Discord](https://discord.com/channels/1110598183144399058/1110598183144399061/1384620051562364978))。

**主题 4：硬件、基础设施与底层优化**

- **GPU 性能炒作遭遇底层现实**：一位新成员在 **4090** 上使用 **1 step sdxs model** 实现了 **512x512** 分辨率下 **294 images/sec** 的速度 ([GPU MODE Discord](https://discord.com/channels/1189498204333543425/1189498205101109300/1384720308581306418))，现在使用 **sdxl** 在 **1280x1024** 分辨率下达到了 **23 fps**。用户讨论了尝试使用自定义 kernel 以同时利用 **CUDA** 和 **tensor cores** ([GPU MODE Discord](https://discord.com/channels/1189498204333543425/1189498205101109300/1384720308581306418))，以及使用 **CUDA gdb** 和 [compute-sanitizer](https://developer.nvidia.com/compute-sanitizer) 调试的挑战 ([GPU MODE Discord](https://discord.com/channels/1189498204333543425/1189607726595194971/1384819859895881779))。一项实验显示 **4090** 比 **3090** 更快，但 token 速度受限于 **RAM bandwidth** ([LM Studio Discord](https://discord.com/channels/1110598183144399058/1153759714082033735/1384615085376274633))。
- **Modular/Mojo 追求 GPU 无关的理想境界**：Modular Platform **25.4** 允许在无需修改代码的情况下，在 **AMD** 和 **NVIDIA** GPU (**MI300/325X**, **Blackwell**, **RTX 40xx**, **RDNA3**) 上运行相同的代码，在 prefill 密集的 BF16 工作负载中将吞吐量提升了高达 **53%** ([Modular Discord](https://discord.com/channels/1087530497313357884/1098765954302873621/1384930608400171119))。Modular 开源了超过 **450k lines** 的生产级 **Mojo** kernel 代码，并实现了无系统调用或运行时依赖的裸机执行，以实现 *zero-overhead abstractions* ([Modular Discord](https://discord.com/channels/1087530497313357884/1098765954302873621/1384930608400171119))。
- **硬件趋势面临障碍与激烈竞争**：成员们讨论了 **DDR5 server setups** 的性价比以及未来 **Intel Nova Lake** 的 PCIe 通道 ([LM Studio Discord](https://discord.com/channels/1110598183144399058/1153759714082033735/1384615085376274633))，而 PCIE 6.0 SSD 因成本和复杂性面临延迟至 **2030** 年的情况 ([HuggingFace Discord](https://discord.com/channels/879548962464493619/879548962464493622/1384613050166018250))。尽管 [MAX supporting Blackwell](https://www.modular.com/max)，早期采用者在 **Blackwell accelerators** 上仍面临支持有限和驱动程序 bug 等问题，并指出具有相似 VRAM 的更便宜的 **AMD/Intel** GPU 甚至 **4090** 可能是更好的选择 ([Modular Discord](https://discord.com/channels/1087530497313357884/1151418092052815884/1384688130149584939))。

**主题 5：开发者工具、平台与生态系统演进**

- **Hugging Face Hub 推动社区与工具创新**：**Gradio MCP hackathon** 成为 2025 年规模最大的 AI 开发者活动，吸引了 **2500+** 人注册和 **$700,000** 的赞助 ([HuggingFace Discord](https://discord.com/channels/879548962464493619/897387888663232554/1384798402403106816))，并公布了 [Geo Calculator MCP](https://huggingface.co/spaces/Agents-MCP-Hackathon/geocalc-mcp) 和 [Consilium MCP](https://huggingface.co/spaces/Agents-MCP-Hackathon/consilium_mcp) 等获奖作品 ([HuggingFace Discord](https://discord.com/channels/879548962464493619/1014577787039924226/1384618766758973440))。**Google Colab** 现在与 **HF** 集成，使用户能够直接从 Hub 在免费的 Colab notebooks 上试用 AI 模型 ([HuggingFace Discord](https://discord.com/channels/879548962464493619/897387888663232554/1384798402403106816))。**memX**（用于多 Agent LLM 系统的共享内存层）已发布，代码托管在 [GitHub](https://github.com/MehulG/memX)，并在 [X 上提供了 demo](https://x.com/0xmmmehulll/status/1935301927967055950) ([HuggingFace Discord](https://discord.com/channels/879548962464493619/897390720388825149/1384623208250216601))。
- **本地开发工具增加新功能，但也面临局限**：**LM Studio** 在应用程序提供环境时，通过其 [API](https://lmstudio.ai/docs/app/api/tools) 支持 **tool calling**，一位用户认为这是其优越性的原因 ([LM Studio Discord](https://discord.com/channels/1110598183144399058/1110598183144399061/1384620051562364978))。用户请求在 **LM Studio** 中添加“继续生成”按钮以实现连续运行 ([LM Studio Discord](https://discord.com/channels/1110598183144399058/1110598183144399061/1384620051562364978))，并确认 **Aider** 的 **/read-only** 命令可以防止文件修改 ([aider Discord](https://discord.com/channels/1131200896827654144/1133060505792159755/1384614248558231662))。**OpenHands CLI** 发布，为其 coding agent 提供顶级准确度，并通过移除 Docker 依赖简化了安装过程 ([Latent Space Discord](https://discord.com/channels/822583790773862470/1075282825051385876/1384612594052235456))。
- **专业平台引入独特功能与特性**：**NotebookLM** 用户遇到了 **Gemini app** 的 **deep research** 每日限制，并对免费版与付费版计划感到困惑 ([Notebook LM Discord](https://discord.com/channels/1124402182171672732/1124402182909857966/1384617987268677653))，同时还请求了 **LaTeX 支持**，并遇到了非英语语言生成超过 10 分钟的 **audio overviews** 的问题 ([Notebook LM Discord](https://discord.com/channels/1124402182171672732/1124402182909857966/1384617987268677653))。**OpenRouter** 为 thinking models 默认推送了 **reasoning by default** 以最大化性能 ([OpenRouter Discord](https://discord.com/channels/1091220969173028894/1092729520181739581/1384936458539761778))，用户还请求了为 **API keys 分配特定余额**等功能，以便更好地控制成本 ([OpenRouter Discord](https://discord.com/channels/1091220969173028894/1094454198688546826/1384612204967493746))。**KREA AI** 发布了 **Krea 1** 公测版，旨在提供更好的审美控制和图像质量，可在 [krea.ai/krea-1](https://xcancel.com/krea_ai/status/1934981993722466454?s=46) 免费体验 ([Latent Space Discord](https://discord.com/channels/822583790773862470/1075282825051385876/1384612594052235456))。


---

# Discord: 高层级 Discord 摘要

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的速率限制导致 UI 断连**：用户报告称，达到 **Cursor's rate limits** 会导致 UI 断开连接，在等待后需要手动重新启动请求。用户请求增加 UI 指示器，以防止混淆速率限制与网络问题。
   - 相比基于请求的限制，用户更倾向于新的基于 Token 的速率限制，因为在发生错误时，用户可以更放心地立即停止并修复问题。
- **Claude-4-Sonnet 速度缓慢**：用户报告 **Claude-4-Sonnet** 正在经历性能问题，被描述为“几乎无法使用”和“太慢了！”。
   - 根据 Anthropic 状态页面，他们正面临 **Sonnet 4** 的性能问题。
- **Context7 MCP 在处理新文档方面受到关注**：用户正在讨论使用 [Context7 MCP](https://github.com/upstash/context7) 处理文档的好处，指出它允许使用旧训练数据的模型利用新文档进行编码。
   - 成员们辩论是否应该“针对此类请求始终使用 Context7”。
- **@Docs 索引功能是游戏规则改变者**：`@Docs` 索引功能使 AI 能够利用新的代码库，而不是依赖旧版本，从而允许用户提出非常具体的问题。
   - 用户现在可以轻松使用 `@` 符号让 AI 引用最新更新的手册。
- **后台 Agent 的 Slack 集成故障**：根据用户分享的图片，当从 **Slack** 标记时，后台 Agent 停留在 *generating...* 状态。
   - 解决方案或具体原因尚未确认。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 播客首播**：**OpenAI Podcast** 联手 **Sam Altman** 和 **Andrew Mayne** 启动，涵盖了 **AGI**、**GPT-5** 和隐私等话题，可在 [Spotify](https://open.spotify.com/show/0zojMEDizKMh3aTxnGLENP)、[Apple](https://podcasts.apple.com/us/podcast/openai-podcast/id1820330260) 和 [YouTube](https://youtu.be/DB9mjd-65gw?si=mqnMnL0dHYAC8YYx) 上收听。
   - [OpenAI to Z Challenge](https://discord.com/channels/974519864045756446/977259063052234752/1372625656361386196) 仍开放提交，旨在理解和防止不对齐泛化（misalignment generalization）。
- **识别出涌现的不对齐模式**：被训练产生不安全代码的语言模型可能会发展出广泛的“不对齐（misalignment）”，导致发现与此行为相关的特定内部模式，详见 [博客文章](https://openai.com/index/emergent-misalignment/)。
   - 研究人员强调了理解和防止这种涌现的不对齐（emergent misalignment）的重要性，以确保 AI 系统的负责任开发。
- **语音比文本获得更多信任**：一份研究论文 ([arxiv.org/abs/2503.17473](https://arxiv.org/abs/2503.17473)) 发现，与**文本 (64%)** 相比，人们在**语音 (74%)** 形式下更信任 AI 生成的输出，突显了传递媒介的影响。
   - 成员们辩论了这一趋势背后的心理学，一些人指出用户可能无法区分人类和 AI 的声音。
- **Midjourney 的开放世界 Demo 重写了物理定律**：**Midjourney** 新的开放世界视频模型虽然令人印象深刻，但因产生看似“完全重写物理定律”的奇怪结果而受到批评。
   - 用户注意到动画僵硬、缺乏适当的音频以及整体的“噩梦”感，引发了对其在 **AGI** 应用中准备就绪程度的质疑。
- **Grok 在 ChatGPT 界面中缺失**：多名用户报告 **Grok** 在 ChatGPT 中消失，错误消息显示 *GPT not found or insufficient permissions*。
   - 成员们还建议了一些变通方法，例如使用 **project folders** 或其他平台（如 **Gemini**、**Grok** 和 **Claude**）来更好地管理和分离对话。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 2.5 Pro 和 O3 Pro 构成超级学习工具**：一位用户发现 **O3 Pro** 和 **Gemini 2.5 Pro** 结合可以创建一个强大的学习工具，其中 **O3 Pro** 擅长规划 Udemy 章节，而 **Gemini** 则负责处理速度和 Flashcards。
   - 该用户表示，*O3 Pro 生成的提示词在将 Udemy 的整个章节划分为大小适中、简单易懂的课程方面表现出色*，并且*两者结合简直是“巨兽”*。
- **ChatGPT O3 的“后劲”不如 Perplexity**：用户注意到 **ChatGPT.com** 上的 **O3** 与其在 **Perplexity AI** 上的表现相比，*juice*（推理长度）明显较少。
   - 用户解释称 *ChatGPT.com 上的 O3 比 Perplexity 上的“后劲”小得多，只有后者的 1/4*。
- **Perplexity Labs 面临间歇性错误**：用户报告在 **Perplexity Labs** 遇到错误，特别是在生成过程结束时，但建议尝试新建标签页。
   - 一位用户表示，在忽略错误并使用相同链接打开新标签页后，他们收到了一封确认任务完成的电子邮件，并附带了打开链接。
- **Perplexity 现在遵循 Robots.txt**：当 Perplexity 无法浏览给定链接时，一名成员指出 **Perplexity** 尊重网站的 **robots.txt** 文件，该文件规定了网络爬虫如何与站点交互；该文件阻止了对给定 URL 的访问。
   - 该成员引用了技术常见问题解答：[How does Perplexity follow robots.txt](https://www.perplexity.ai/hub/technical-faq/how-does-perplexity-follow-robots-txt) 以获取更多详情。
- **Perplexity 模仿 4o 的多步搜索**：用户观察到 **Perplexity** 现在表现出类似于 **ChatGPT 4o** 的多步搜索行为，在 *thinking* 过程中进行搜索。
   - 一位用户将这种新行为描述为 *似乎现在在思考过程中进行搜索？？*

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **具备推理能力的 MiniMax M1 首次亮相**：**MiniMax M1** 是目前上下文最长的开源推理模型，已正式发布，并在发布周提供 [OpenRouter 25% 折扣](https://x.com/OpenRouterAI/status/1935376450099478975)。
   - 测试新模型的用户指出 Token 使用计数存在差异，解释称这是 **system prompt injection** 的结果。
- **Gemini 2.5 Pro 的推理模型上线**：**Gemini 2.5 Pro, Flash 和 Flash Lite** 推理模型已上线，现在 **Gemini 2.5 Pro** *要求* 必须启用推理。
   - 用户报告在不启用推理的情况下通过 API 使用 `google/gemini-2.5-pro` 时收到 **Error 400**，但 OpenRouter 已实施修复程序以解决 **2.5 flash preview thinking/non-thinking models** 的问题。
- **OpenRouter 默认推送推理功能**：OpenRouter 正在为 `anthropic/claude-3.7-sonnet` 等思考模型*默认启用推理*，这在基准测试中也被观察到可以最大化模型性能。
   - 可以使用 [multi-model reasoning standard](https://openrouter.ai/docs/use-cases/reasoning-tokens) 禁用或配置推理。
- **用户请求细粒度的 API 余额控制**：成员们请求能够为 **API keys 分配特定余额**，以便更好地控制成本和保持一致性。
   - 一位成员建议可以通过中间件进行管理，允许将资金分配给特定的 Key，并防止超出设定限制的支出。
- **社区提供 AI Discord 机器人模板**：一位成员在 GitHub 上分享了他们的 **AI Discord 机器人模板**，旨在直接推送来自 OpenRouter 的公告和模型统计数据。
   - 目标是创建一个能够处理新模型公告并直接链接到用户的机器人，在 Discord 内部提供更多关于模型的统计数据。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **由 World Knowledge 增强的 Reasoning**：成员们讨论了模型中的 **reasoning** 如何与数据的广度和形成的连接相关联，强调了 **world knowledge** 对整体模型性能的影响。
   - 一位用户表示，希望未来能通过更好的方法论，让更小的模型解锁并训练出更多的 **world knowledge**。
- **LiveCodeBench Pro 无污染**：新的 **LiveCodeBench Pro** 基准测试旨在做到无污染（contamination-free），因为这些题目是在模型发布日期 *之后* 发布的。
   - 有用户指出，这些题目是 **IOI problems**，属于 **competitive coding** 题目，模型尚未达到饱和状态。
- **Gemini 2.5 Flash 评分走高**：一项新的基准测试显示，**Gemini 2.5 Flash** 的评分几乎比 **Claude Sonnet 3.7 Thinking** 高出 2 倍。
   - 一位用户表示，这与 **GitHub Copilot** 上的真实体验不符，并提到 *o3-mini* 和 *o4-mini* 的表现要差得多。
- **Sam Altman 是反社会人格？**：成员们辩论了 **Sam Altman** 的可信度，引用了一个显示 *重大危险信号* 的 [Reddit 帖子](https://www.reddit.com/r/AskReddit/s/kCl9GCniZz)。
   - 一位用户暗示 **Sam Altman** *更像是反社会人格，对控制权的渴望胜过一切*，而另一位用户引用了 *将他从 OpenAI 开除的董事会成员的描述，称他在心理上具有虐待性*。
- **LM Arena 经历宕机**：用户报告了 **LM Arena** 频繁出现的错误和宕机，包括 *Failed to verify your browser* 错误和 *Something went wrong with this response* 的 bug。
   - 团队的一名成员提到，他们正专注于解决错误和模型不响应的问题，并努力创建一个可靠的服务。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **用户请求 Magistral Vision**：一位用户询问了关于 **Magistral + vision support** 的情况，类似于 Hugging Face 模型，并发现它已经集成到了 [Unsloth's Magistral Small](https://huggingface.co/unsloth/Magistral-Small-2506-GGUFI) 中。
   - 目前可用的版本是 **Q8_XL**。
- **Unsloth 预告多 GPU 训练**：Unsloth 正在积极使用 **accelerate** 开发双 GPU 支持，但官方尚未正式支持。
   - 不建议混合使用不同的 GPU（如 **5090** 和 **3090**）进行训练；建议分开训练。
- **BERT Notebook 在 Colab 上出现问题**：在最近的更新后，成员们在 **Colab** 上使用 **BERT notebook** 时遇到了错误；已开启一个 [GitHub issue](https://github.com/unslothai/notebooks/issues/60) 来解决这些错误。
   - 这些问题是在环境变量更改后出现的，可能使设置变得复杂。
- **Gemma 语言和视觉模型即将登陆 Unsloth**：支持 **float16** 和 **bfloat16**、语言和视觉的 **Gemma3** 即将登陆 Unsloth。
   - 您可以在[此处](https://discord.com/channels/1179035537009545276/1383146852337057811/1384455067675136040)查看公告。
- **DeepMind Gemini v2.5 引起关注**：Google Deepmind 发布了 [Gemini v2.5 Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)。
   - 一位成员建议 *阅读 TLDR*，称其 *确实非常有趣*。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI 语音克隆引发诈骗警觉**：成员们讨论了 **AI 语音克隆诈骗** 的社会影响，有人认为这些诈骗反而能提高警觉并推动对更广泛 **AI 安全措施** 的需求。
   - 尽管存在潜在的哲学益处，大多数人仍对解决 AI 驱动欺诈风险的紧迫性达成了一致。
- **提议更改 EleutherAI 名称以避免 LLM 嵌入？**：一位成员提议更改 **EleutherAI** 的名称，以减少其在 LLM 权重中的显著性，认为避免在模型权重中直接关联可能是有益的。
   - 另一位成员则主张通过结构化的介绍和关于 **科学方法** 的培训计划来改善着陆区，以更好地吸引新人，而不是重新命名。
- **Gemini Diffusion 的 Thue Morse 序列故障**：一位早期访问用户测试了 **Gemini Diffusion** 生成 **Thue Morse 序列** 的能力，报告称在第六次迭代之前是成功的。
   - 超过该次数后，模型出现了故障循环（glitch-loops），突显了生成复杂序列的潜在局限性；注意目前 *没有 API*。
- **Randall 的样条理论（Spline Theory）受到关注**：一位成员为 **Randall 的样条理论** 辩护，引用了 [一段采访](https://youtu.be/l3O2J3LMxqI?si=RsDNVoqjvu5OwAzB) 并质疑语言模型中位置编码（positional encoding）的必要性。
   - 他们认为位置信息主要由 **V (spatial_proj) 下三角学习矩阵** 提供，并受限于 ranker 选择的 top-k 上下文。
- **线性注意力（Linear Attention）渗透进无注意力模型**：尽管声称是无注意力的，一位成员指出所讨论的模型在 contextualizer 中使用了 **线性自注意力（linear self-attention）**，挑战了最初的断言。
   - 这引发了关于“注意力”定义的争论，一些人认为 *attention = softmax qkv attention*，以此将他们的模型与使用线性注意力机制的其他模型区分开来。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 通过 API 掌握工具调用（Tool Calling）**：虽然 **LM Studio** 缺乏内置工具，但它通过其 [API](https://lmstudio.ai/docs/app/api/tools) 支持 **工具调用**，使模型在应用程序提供环境时能够使用外部工具。
   - 一位用户赞扬了 **API 访问**，称其为 **LM Studio** 优于其他本地 llama 界面的原因。
- **开源模型面临功能滞后**：成员们认为开源模型一直处于停滞状态，缺乏对 **音频/视频/图像处理** 和 **内置互联网访问** 的原生支持。
   - 其他用户反驳称，小至 **0.2B** 的视觉模型已经存在，且 **LM Studio beta** 具有 **MCP** 支持，可以连接数千个工具和服务，包括 **网页浏览**。
- **用户渴望 LM Studio 中的“继续生成”按钮**：用户请求在 **LM Studio** 中增加一个“继续生成”按钮，以便模型能够连续运行而无需手动重新提示，从而改善整体用户体验。
   - 作为回应，其他成员建议使用 **API** 或 **自动点击脚本** 作为实现连续生成的潜在变通方案。
- **DDR5 vs DDR6：内存竞赛**：成员们正在等待 **DDR5 服务器配置** 变得更实惠，或者考虑 **Intel 的 52 核 Nova Lake**（如果它能提供足够的 **PCIE 通道**）。
   - 关于 Intel 将采取行动夺回其在 **消费级 CPU 市场** 地位的猜测不绝于耳。
- **4090 在 Token 处理中占据主导地位**：一项比较 **RTX 3090** 和 **RTX 4090** 之间 Token 处理速度的实验显示，由于其卓越的 GPU 架构，**4090** 明显更快。
   - 尽管 **4090** 具有优越性，但 **Token 速度** 并没有增加太多，这表明瓶颈主要在于 **RAM 带宽速度**。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **CGS-GAN 潜空间让人联想到 StyleGAN3**：一位成员分享了 [CGS-GAN 潜空间](https://github.com/fraunhoferhhi/cgs-gan)，指出其与 **StyleGAN3** 可视化器的相似之处。
   - 一段 [实时视频](https://cdn.discordapp.com/attachments/1149866623109439599/1384652760099852410/Screencast_from_17-06-25_225208.webm?ex=6854875f&is=685335df&hm=8d84a0bbacd0e563c40e1d8851f2801b8a765b2b22c2ef910e84cdefc7ca8306&) 展示了这个有趣的新潜空间。
- **Gemini Canvas 增加沉浸式集成**：**Google** 将 **Gemini** 集成到了他们的 **Canvas** 中，允许创建名为 *immersives* 的编码组件。
   - 一位成员创建了一个静态组件，展示了 Gemini 如何感知各种概念，并分享了一个[链接](https://gemini.google.com/share/54661b0f8f17)供大家探索和生成图像。
- **LLM 压缩信息：Token 级别的见解**：一位成员断言 **LLM** 的目的是 *压缩信息*，特别是关于输入和输出的 Token，这受守恒定律约束。
   - 他们建议通过量化 **熵 (entropy)** 来描述计算过程中的行为。
- **Meta 研究团队论文发布！**：成员们分享了来自 **Meta 研究团队** 的[论文集](https://arxiv.org/abs/2505.12514)、[另一组论文](https://arxiv.org/abs/2506.10077)以及[又一组论文](https://arxiv.org/abs/2506.12115)。
   - 一位成员将这些论文中的发现描述为 *绝对的金矿*。
- **扎克伯格（Zuckerberg）考虑合并 Meta 团队**：据一位成员透露，有猜测称 **Zuckerberg** 可能会尝试合并 Meta 的 **研究团队和 Llama 团队**。
   - 据称其动机是保留来自 **Yann LeCun** 的 *视觉领域思想领导力*，同时转向针对工业应用的策略优化。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF 的 Gradio 黑客松成为规模最大的 AI 开发者盛会！**：**Gradio MCP 黑客松** 已成为 2025 年规模最大的 AI 开发者活动，已有 **2500+** 人注册，并获得 **$700,000** 的赞助 [Gradio 推文](https://x.com/Gradio/status/1929331605081829385)。
   - 获胜者包括 [Geo Calculator MCP](https://huggingface.co/spaces/Agents-MCP-Hackathon/geocalc-mcp)、[Gradio Workflow Builder](https://huggingface.co/spaces/Agents-MCP-Hackathon/gradio_workflowbuilder) 和 [LLM Game-Hub](https://huggingface.co/spaces/Agents-MCP-Hackathon/LLMGameHub)。
- **Colab 与 HF 联手进行 AI 模型试用！**：**Google Colab** 现在与 **HF** 集成，允许用户直接从 HF 在免费的 Colab Notebook 上试用 AI 模型 [Google Colab 博客文章](https://medium.com/google-colab/launch-hugging-face-models-in-colab-for-faster-ai-exploration-bee261978cf9)。
   - 这种集成简化了 AI 探索，使开发者更容易上手。
- **memX 发布用于 LLM 大脑的共享内存**：**memX** 发布，这是一个用于多智能体 **LLM** 系统的共享内存层，使 Agent 能够读取和写入不断演进的上下文，代码可在 [GitHub](https://github.com/MehulG/memX) 上获取。
   - 主要功能包括 **实时发布/订阅 (pub/sub)**、**JSON Schema 验证**、**API-key ACLs** 和 **Python SDK**，并在 [X/Twitter](https://x.com/0xmmmehulll/status/1935301927967055950) 上提供了演示。
- **OS Agent 开始编写代码**：一位成员表示他们修复了自己的 **OS Agent**，使其成为一个比 **Codex** 更好的编码 Agent。
   - **Master Agent** 可以召唤 **Mini Agent** 进行任务分配和讨论。
- **HF AI Agents 课程重新开课**：一位成员恢复了 **HF AI Agents 基础课程**，正在开发一个 **Chatbot 项目**，该项目使用 **生成式 AI** 根据文件或文本回答问题。
   - 另一位用户寻求关于 **Unit 4 最终评估模板** 步骤的解答，包括克隆 [最终评估模板](https://huggingface.co/spaces/agents-course/Final_Assignment_Template) 并修改 *app.py*。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 定价差异曝光**：成员们发现 [Aider 上的 **Gemini 2.5** 定价可能不准确](https://storage.googleapis.com/gweb-uniblog-publish-prod/original_images/gemini_2-5_benchmarks_margin_light2x_1.gif)，暗示成本可能比估计高出 **5倍**。
   - 一位用户报告在一个超过 200 次 commit 的 **5K LoC 项目**上*花费了 3 美元*，凸显了潜在的成本担忧。
- **Claude-4-Opus 面临成本批评**：**Claude-4-Opus** 的输入/输出价格为 **15/75**，使其价格*贵了 7.5 倍*，引发了关于 token 使用量以及与 **Gemini** 相比的性价比讨论。
   - 一位成员指出，**Gemini** 生成的 reasoning tokens 明显多于 **Opus**，这影响了整体成本计算。
- **Aider 配置 Gemini 2.5-pro**：用户确认 **Aider** 现在支持最新的 **Gemini 2.5-pro** 模型，并指出在 Aider 设置中指定新模型即可工作，尽管可能会收到关于*未知上下文窗口大小和成本*的警告。
   - 成员们强调 **Aider** 使用了*合理的默认值 (sane defaults)*，减轻了与特定 `context window size and costs` 相关的问题。
- **Aider 聊天历史记录得到恢复**：一位用户询问如何在崩溃后继续之前的会话，一位成员建议使用 **--restore-chat-history** 标志。
   - 目前尚不清楚为什么其中一位用户提到了他们产生了幻觉 (hallucinated)。
- **Aider 的文件为 /read-only**：一位用户询问 **/read-only** 是否旨在防止 Aider 修改文件。
   - 一位成员确认它应该能够防止修改。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Midjourney 进军视频制作**：[Midjourney](https://xcancel.com/midjourney/status/1935377193733079452) 发布了其 **Video Model** 的 **Version 1**，允许用户为 Midjourney 生成的或外部图像制作动画，成本约为*每秒视频消耗一张图像的费用*。
   - 新的 **“图生视频 (Image-to-Video)” 功能**具有“自动”和“手动”动画设置，并提供“高动态 (high motion)”和“低动态 (low motion)”选项；发布初期视频生成仅限网页端。
- **Krea AI 开启公开测试**：**KREA AI** 发布了 **Krea 1** 公测版，旨在提供更好的审美控制和图像质量，创建细腻的纹理、戏剧性的角度和电影级光效，摆脱典型的“AI 感”，访问地址为 [krea.ai/krea-1](https://xcancel.com/krea_ai/status/1934981993722466454?s=46)。
   - 新模型支持风格参考和自定义训练，并免费提供。
- **OpenHands CLI 摆脱 Docker 拖累**：**All Hands AI** 推出了 **OpenHands CLI**，这是其编程 Agent 的命令行界面，提供顶尖的准确度，并通过移除 Docker 依赖简化了安装过程。
   - 虽然 CLI 保持了与 Docker 版本相同的准确度，但它缺少 Web 浏览器组件，不过提供了斜杠命令 (slash commands) 和命令确认模式。
- **Essential AI 发布 24T Token 网页数据集**：**Essential AI** 宣布推出 **Essential-Web v1.0**，这是一个包含 **24 万亿 token** 的预训练数据集，包含详细的元数据，旨在创建高性能模型。
   - **Essential-Web v1.0** 的特定领域子集在网页代码、STEM 和医疗等领域表现出更强的性能。
- **CoreWeave 与 W&B 预热推理竞争**：**CoreWeave** 和 **Weights & Biases** 正在推出新的 AI 推理服务和在线评估工具（监控器），用于实时 LLM 评判。
   - 这些服务运行在 **CoreWeave GPU** 上，包括针对 DeepSeek R1-0528 和 LLama-4 Scout 等模型的推理端点，并提供 OAI 兼容 API，旨在 AI 基础设施领域提供更多竞争和灵活性。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 驱动 GPU 无关的理想境界**：得益于与 **AMD** 的合作，Modular Platform **25.4** 允许在 **AMD** 和 **NVIDIA** GPU 上运行相同的代码而无需修改，支持 **AMD Instinct™ MI300X** 和 **MI325X GPU**。
   - 正如 [Modular 博客](https://www.modular.com/blog/modular-25-4-one-container-amd-and-nvidia-gpus-no-lock-in?utm_source=discord&utm_campaign=25_4)所述，该版本在 **Llama 3.1**、**Gemma 3** 和 **Mistral** 等语言模型上，针对预填充密集型（prefill-heavy）的 BF16 工作负载，提供了高达 **53%** 的吞吐量提升。
- **Modular 开启内核代码宝库**：Modular 已开源超过 **45 万行** 生产级 **Mojo** 内核代码，增强了透明度和社区贡献。
   - 此版本还包括改进的文档、**PyTorch** 算子教程和内核性能工具，促进了更简便的采用和开发。
- **Mojo 实现无运行时依赖的裸机运行**：**Mojo** 现在可以在没有系统调用或运行时依赖的情况下在裸机（Bare Metal）上运行，这使其能够作为一种*具有零开销抽象的现代系统编程语言*用于内核开发。
   - 在完成如 `KGEN_EE_JIT_GlobalConstructor` 和 `KGEN_EE_JIT_GlobalDestructor` 等简单的函数替换后，它可以作为一种具有零开销抽象的现代系统编程语言使用，如[此图](https://cdn.discordapp.com/attachments/1151418092052815884/1384859713388417126/image.png?ex=68549f5d&is=68534ddd&hm=d1af4a129aa4c5bd2cd0ebd8a5db27931f7454475f043fdf816af81fad245892&)所示。
- **通过 Python 主接口进行 MAX 推理**：**图编译器**（MAX 模型构建的核心）的主要接口目前是通过 **Python**，用于启动推理会话并输入 numpy 数据。
   - 官方表示 *Python 使得集成到现有的分词器（tokenizer）、处理逻辑等变得非常容易*，更多信息可以在[这里](https://forum.modular.com/t/mojo-max-bindings/1499/3)找到。
- **Blackwell 买家抱怨开局不顺**：尽管 [MAX 支持 Blackwell](https://www.modular.com/max)，但 **Blackwell 加速器** 的早期采用者正面临支持有限和驱动程序 Bug 频发的问题，导致体验不稳定。
   - 一位用户强调，**AMD** 和 **Intel** 正在推出具有相同或更多显存（VRAM）且价格减半的 GPU，并建议更便宜的 **4090** 可能是更好的替代方案。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **高流量耗尽网页生成额度**：用户报告称，由于错误，生成一个简单的网页消耗了大量额度，一名用户不得不求助于手动编辑文件。
   - 一位用户指出，这种情况发生在*中午之后*，并暗示一天中不同时间的流量可能有所不同。
- **Facebook、Gumtree、eBay 抓取网站项目失败**：一名用户花费了 **5000 额度** 尝试创建一个抓取 **Facebook、Gumtree** 和 **eBay** 盗窃自行车列表的网站，但 AI 失败了，交付的是虚假结果。
   - 该用户收到了 **2500 额度的退款**，但指出这是*浪费时间和额度*。
- **MiniMax AI 推出 Agent 模式，挑战 Manus**：用户讨论了 **MiniMax AI** 的新 Agent 模式，将其视为 Manus 的竞争对手。
   - 一些人对 **MiniMax** 的额度系统和订阅模式表示担忧，同时称赞 Manus；另一些人则认为*竞争只会给用户带来更多 AI 选择*。
- **用户呼吁针对 AI 任务错误退还额度**：用户讨论了当 **Manus** 遇到错误并不得不重新运行流程时，是应该退还额度，还是仅在任务成功完成时收费。
   - 一位用户将其比作支付了*一个肉质腐烂、面包陈旧的汉堡*，而另一位用户则表示 *AI 经过编程/训练，能做的事情比一个人一辈子能告诉它做的还要多*。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Block 为 Claude 设计 Model Context Protocol (MCP) 服务器**：Block 的工程团队推出了 **Model Context Protocol (MCP) 服务器**，并提供了一个[设计方案](https://t.co/0vJajYzrfJ)，帮助它们与 **Claude** 及其他 AI 系统集成，以构建更好的 **AI 助手**。
   - 这些 **MCP 服务器** 为 Agent 直接访问数据源创造了新途径，但对于 **PDF** 和 **PPT** 等非结构化格式，仍需要预处理和索引。
- **AG-UI 和 CopilotKit 创建 Agent 前端**：LlamaIndex 宣布通过 [CopilotKit](https://t.co/hzxBrXKyTv) 提供官方 **AG-UI** 支持，从而简化了后端 Agent 到面向用户应用的集成过程。
   - 目标是让开发者能够 *以零样板代码创建由 Agent 驱动的前端*。
- **FastAPI 深受流式传输卡顿困扰**：一位用户报告在使用 **FastAPI** 向前端流式传输事件时遇到 **20 多秒的延迟**，问题追溯到 yield 空 delta 的行为，通过 `if ev.delta: yield` 得到了解决。
   - 虽然有人建议添加 `yield json.dumps({..}) + "\n\n"`，但最终通过确保仅 yield 非空 delta 的 `if ev.delta: yield` 解决了该问题。
- **元数据过滤功能释放**：想要在 chat/query engine 上使用元数据过滤的用户，现在可以将 retriever 传递给 chat engine，该流程适用于 **Condense_plus_context** 等引擎。
   - 社区成员对这一解决方案表达了 *极大的感谢*。
- **Anthropic 在 Agent 框架中冷落 LlamaIndex**：一位用户注意到，在 **Anthropic** 关于[“构建高效 AI Agent”](https://www.anthropic.com/engineering/building-effective-agents)的指南中，框架列表里缺少了 **LlamaIndex**。
   - 作为回应，一位社区成员提到，尽管由于现有的关系未被列入名单，但 **LlamaIndex 绝对是一个宝藏**。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **4090 在 Stable Diffusion 上表现惊人**：一位新成员在 **4090** 上使用 **1 step sdxs 模型**，在 **512x512** 分辨率下达到了 **294 images/sec**，这可能是 2023 年 10 月实时视频领域的先驱工作（[X.com 上的帖子](https://x.com/Dan50412374/status/1772832044848169229)）。
   - 他们现在使用 **sdxl** 在 **1280x1024** 下达到了 **23 fps**，并表示有兴趣探索自定义 kernel，以同时利用 **CUDA** 和 **tensor cores**。
- **自定义 CUDA Kernel 引起热议**：一位具有软件架构和 SQL 数据库性能背景的成员正在实验自定义 kernel，以并发调用 **CUDA** 和 **tensor cores**。
   - 他们正在构建一套配备双 **5090**、**7985WX Threadripper** 和 **256 GB** 内存的新系统，用于 **Stable Diffusion** 的硬核实验。
- **Triton 的共享内存备受关注**：一位用户询问如何强制 **Triton** 显式地将 tensor 加载到 **shared memory** 中以避免 register spilling，他观察到了 **register spilling** 并希望使用 shared memory 能提高 Triton 的性能。
   - 一位成员回应称，据他所知，目前无法避免 **Triton 对 shared memory 的自动管理**。
- **CUDA 调试技巧出现**：成员们讨论了调试挑战，建议包括使用 **CUDA gdb** 和提交 [NVIDIA bug 报告](https://developer.nvidia.com/bugs/new)。
   - 建议实现[正确的 CUDA Runtime 错误检查](https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api)，并探索使用 [compute-sanitizer](https://developer.nvidia.com/compute-sanitizer) 进行调试。
- **FLE 团队面临进度压力**：团队成员对 **Factorio Learning Environment (FLE)** 项目进展缓慢表示担忧，并需要更定期地合并 pull requests。
   - 一位核心团队成员建议创建一个 **FLE GitHub 组织**，以实现写入权限民主化并加快 pull requests 的合并进度。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 为编程好奇心创建频道**：社区创建了 <#1384974112841269399> 频道，用于讨论 **AI 研发**项目，欢迎成员加入交流与分享。
   - 严禁成员发布任何形式的广告。
- **Canvas 编码者渴望笛卡尔坐标系创作**：一名成员请求一个能够理解指令并在**笛卡尔平面画布**上绘制酷炫艺术作品的模型。
   - 社区针对潜在的想法和解决方案进行了集思广益。
- **Command-R 补全故障引发担忧**：用户报告 `command-r-08-2024` 模型生成的句子不完整，示例如下：*"Firefly's smile deepens, a hint of mischief in her red eyes."Well, hello there," she says, her voice carrying a"*。
   - Cohere 支持团队建议将 **SDK** 升级到最新版本以解决此问题。
- **SDK 中疑似出现 Scanner 故障**：一名成员在使用 `cohere/command-r-08-2024` 模型时遇到了 `bufio.Scanner: SplitFunc returns advance count beyond input` 错误。
   - 尽管最初建议从客户端寻找原因，但证据表明该问题源自 Cohere 的 **Go SDK**。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Muon 表现不及 AdamW**：在一个 PR 中，**Muon** 相对于 **AdamW** 的性能受到质疑，性能较低可能与集成错误或 **torchtune** 特定问题有关，参见 [PR 2803](https://github.com/pytorch/torchtune/pull/2803#issuecomment-2981262780)。
   - 观察发现，当 SFT 优化器与预训练优化器不同时，**Muon** 相比 **AdamW** 没有显著优势，这表明需要进一步优化。
- **Kron 经过调优后与 AdamW 表现相当**：使用另一种实现（[ethansmith2000/fsdp_optimizers](https://github.com/ethansmith2000/fsdp_optimizers)）进行的对比显示，在经过良好调优后，**Kron** 的表现与 **AdamW** 相似。
   - 总体而言 **AdamW** 速度更快，尽管其内存占用略高于 **Muon** 或对角线 **Kron**。
- **Qwen3 0.6b 的收敛性存疑**：用户对 **Qwen3 0.6b** 的标准收敛性产生怀疑，认为 PR 的设置可能存在问题，并分享了 [WB_Chart](https://cdn.discordapp.com/attachments/1236040539409879170/1384693197653020813/WB_Chart_6_18_2025_3_35_34_AM.png?ex=68540448&is=6852b2c8&hm=1d93bef584b20807f407a2d016efb08ae630f52d814310f89605d7f1648bcf27&)。
   - 对收敛图表的分析显示 tops 差异约为 500，表明设置中可能存在 Bug。
- **成员们对神奇的 Muon 感到着迷**：一名成员对 **Muon 优化器** 表现出浓厚兴趣，惊叹于正交化更新如何加速收敛。
   - 在引用 **Jaguar Muon** 时，他们开玩笑地评论道：*"这种巫术应该被取缔"*。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **K-12 教育工作者寻求 NotebookLM 使用案例**：一名成员询问 **NotebookLM** 在 **K-12 教育**中的常见用例，以及通过 **API** 从**源文件创建出色播客**的能力。
   - 他们询问是否存在实现该功能的现有路线图或路径。
- **Gemini 深度研究功能达到每日限制**：用户报告对 **Gemini 应用**中深度研究功能的[每日限制](https://gemini.google.com)感到困惑，尤其是不清楚免费版与付费版计划的限制区别。
   - 一名用户报告在免费版上达到了限制，但找不到 **Google 付费计划**的相关数据。
- **传闻 NotebookLM 模型为 Gemini 2.5 Flash**：一名用户推测 **NotebookLM** 使用的是与 **Gemini** 相同的底层模型，传闻为 [Gemini 2.5 Flash](https://ai.google.dev/)，但经过调优以减少幻觉。
   - 这一推测基于 [Google AI Studio 发布](https://www.youtube.com/watch?v=EOmgC3-hznM&ab_channel=JeffSu)的一个使用 **Gemini 2.5 Flash** 的实验性音频生成版本。
- **用户恳求 LaTeX 支持**：用户请求 **NotebookLM** 支持 **LaTeX**，但目前尚不支持数学标记。
   - 遇到问题的用户被引导至[功能请求频道](https://feedback.google.com)。
- **音频概览长度被忽略**：用户报告在生成非英语语言的**音频概览**时，尽管使用了自定义提示词，但无法生成超过 **10 分钟**的内容。
   - **NotebookLM** 似乎默认使用英语，并忽略了生成更长音频段落的自定义提示词。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Flow Matching 进入生产环境？**: 成员们讨论了 **Flow Matching (FM)** 在工业界的采用情况，引用了引发讨论的 [一条推文](https://fxtwitter.com/mathusmassias/status/1935246909473521829)。
   - 有人提到 **Imagen** 和 **Flux** 目前正在使用 **FM**。
- **Predictive Coding：猜谜游戏？**: 一位成员分享了一个讨论链接，通过 **猜想与校验计算平方根** 以及 **Backpropagation** 来解释 [Predictive Coding](https://arxiv.org/abs/2407.04117)。
   - [PRECO GitHub repo](https://github.com/bjornvz/PRECO) 也被分享作为进一步研究的资源。
- **V-JEPA-2 论文讨论已排期**: 小组安排了关于 **V-JEPA-2 论文** 的讨论，参考了 [Meta AI 博客文章](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) 和 [相关的 arXiv 论文](https://arxiv.org/abs/2506.09985)。
   - 一位成员创建了一个未来活动，以引导小组了解 [V-JEPA-2](https://discord.gg/mspuTQPS?event=1384953914029506792)。
- **Keen RL 演讲：令人失望？**: 成员们表示 [Richard Sutton 的 Keen Tech RL 演讲](https://fxtwitter.com/RichardSSutton/status/1934780327169523765?t=_pmMm7dwecEO0KTLS8LbMA&s=19) *低于预期*。
   - 他们认为 **Keen 对 RL 的关注** 以及 **Carmack 不兼容的目标** 是失望的原因，但对 **Keen Tech** 开源其代码的潜力表示兴奋。
- **Cursor 新层级发布**: 成员们分享了 [Cursor 新层级公告](https://www.cursor.com/blog/new-tier) 的链接。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **FastMCP 传输层实现自定义**: 一位成员询问如何在 **FastMCP** 中实现 **自定义传输层 (custom transport)**，在客户端通过扩展基础传输类是可行的，但服务器端的确认尚在进行中。
   - 随后讨论了自定义 **FastMCP** 与不同系统通信方式的可能性。
- **MCP 应对公司限制**: 成员们讨论了在公司政策限制 **Claude Desktop** 或 **Cursor** 时的 **MCP client/host** 方案。
   - 一位成员在本地使用 **Ollama** 运行 **devstral:24B** 并配合 **CLINE** 取得了成功，但在使用 **Roo** 时遇到了困难。
- **MCP 驯服大规模 GraphQL API**: 一位成员使用其 **MCP server** 为 **GraphQL API** 查询/变更生成了约 **600 个工具**，展示了 Cursor 在处理这么多工具时的局限性。
   - 他们注意到 **Cursor** 和其他模型在工具数量超过几十个时会感到吃力，如 [这张截图](https://cdn.discordapp.com/attachments/1312302100125843479/1384765895792001044/Screenshot_2025-06-17_at_22.24.52.png) 所示。
- **Multi-Agent 系统不需要 A2A**: 社区讨论了构建使用 **MCP 工具** 的 **Multi-Agent 系统** 是否需要 **A2A**，或者在每个 Agent 中配置 **MCP client** 和 **server** 是否就足够了。
   - 一位成员表示 *不需要 A2A*，并且 *甚至 Google 也不关心它*。
- **Arize AI 发布 Text-to-GraphQL MCP Server**: Arize AI 推出了 **Text-to-GraphQL MCP server**，允许用户直接从 Spaces 页面连接 **MCP servers**。
   - 它通过一个与 **Claude Desktop 和 Cursor** 等 AI 助手集成的 MCP server，将自然语言查询转换为 **GraphQL 查询**，详见 [此 GitHub repo](https://github.com/Arize-ai/text-to-graphql-mcp) 和 [完整文章](https://arize.com/blog/text-to-graphql-mcp-server/)。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 黑客松推迟**：由于 **tinygrad** 需要趋于成熟，tinygrad 黑客松已推迟，最早可能在明年举行。
   - 公告鼓励用户提供反馈，这将影响到最终举办黑客松时的参与者筛选。
- **TinyJit 参数不匹配问题已解决**：用户通过使用 `Variable` 解决了在使用 `@TinyJit` 配合循环时出现的与 `args mismatch in JIT` 相关的 `AssertionError`，从而处理了 **ShapeTracker** 问题。
   - 该解决方案涉及创建一个 `Variable` 来表示循环索引并在循环内进行绑定，以实现 `TinyJit` 的形状对齐，详见 [tinygrad 的 JIT 教程](https://mesozoic-egg.github.io/tinygrad-notes/20240102_jit.html#shape-alignment)。
- **通过 Variable 绑定进行形状对齐**：一名成员解释说，在循环中使用 `TinyJit` 时（尤其是处理张量切片时），使用 `Variable` 可以解决 **ShapeTracker** 的差异。
   - 通过将循环索引绑定到 `Variable`，可以使形状对齐，从而解决 `args mismatch` 错误。
- **Tensor-Variable 数学运算约束**：一位用户询问在 `tinygrad` 中进行 **Tensor** 和 **Variable** 之间的数学运算时，是否要求 **Tensor** 必须位于左侧 (LHS)。
   - 对话暗示了 `tinygrad` 在处理涉及符号变量的运算时存在某种约束。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Discord 用户刷屏 Mr. Beast 内容**：一名 Discord 用户被告知停止在频道中刷屏 **Mr. Beast** 的内容。
   - 另一名 Discord 用户投诉了频道中的刷屏行为。
- **用户对垃圾信息的投诉**：一名 Discord 用户对发布的过量 **Mr. Beast** 内容表示不满。
   - 这凸显了加强频道管理和遵守社区准则以防止垃圾信息的必要性。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MIPROV2 助力 Agent 优化**：一名成员认为，在给定输入输出示例的情况下，使用 **MIPROV2** 优化 Agent 实现是可行的。
   - 另一名成员询问了在工作流上下文中这些输入/输出示例的具体性质。
- **利用工作流指标优化 Agent 实现**：一位用户计划使用内置的评估指标来优化其他项目中的工作流和 Agent 实现。
   - 他们打算在完成后分享其实现。
- **DSPy ❤️ LangGraph 寻求生产环境部署**：一名成员询问如何在生产环境中结合 **DSPy** 和 **LangGraph**，旨在复制 Anthropic 的多 Agent 研究员。
   - 另一名成员引用了 [此 Discord 链接](https://discord.com/channels/1161519468141355160/1202371242519441499/1382339683580772413) 中的相关资源以提供帮助。
- **DSPy 加入 Agentic 编程 IDE**：一名成员询问如何将 **DSPy** 集成到 **Cursor** 或 **Roo Code** 等 Agentic 编程 IDE 中。
   - 他们强调，目前“Agent”模式的设置依赖于使用 *“ONLY RETURN markdown”* 等指令的提示词工程 (Prompt Engineering)。
- **框架辅助 Llama 微调**：一位 **DSPy** 新手寻求关于微调 **Llama** 模型的框架或库的建议。
   - 他们专门请求了关于适合此用途的库和框架的建议。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord: 各频道详细摘要与链接

### **Cursor 社区 ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1384608602282790923)** (971 条消息🔥🔥🔥): 

> `Cursor 速率限制, Sonnet 4 性能, Cursor 定价模型, Context7 MCP, @Docs 索引功能` 


- **Cursor 的速率限制导致 UI 断开连接**：用户报告称，达到 **Cursor 的速率限制 (rate limits)** 会导致 UI 断开连接，需要在等待后手动重新启动请求，这引发了增加 UI 指示器的呼吁，以防止用户混淆速率限制与网络问题。
   - 相比基于请求的限制，新的基于 token 的速率限制更受青睐，正如一位用户所言：*当 LLM 或我犯错时，我可以放心地立即停止并修复它*。
- **Sonnet 4 速度缓慢**：用户报告 **Claude-4-Sonnet** 出现性能问题，一位用户称其“几乎无法使用”，另一位用户则直呼“太慢了！”。
   - 根据 Anthropic 的状态页面显示，他们正面临 **Sonnet 4** 的性能问题。
- **Context7 MCP 受到欢迎**：用户正在讨论使用 [Context7 MCP](https://github.com/upstash/context7) 处理文档的好处，指出它允许拥有旧训练数据的模型使用新文档进行编码。
   - 成员们讨论了是否应该“针对此类请求始终使用 context7”。
- **文档索引对新代码库具有变革性**：一位用户发现了 `@Docs` 索引功能，该功能允许 AI 基于新的代码库运行，而不是依赖旧版本，从而让他们能向 AI 提出非常具体的问题。
   - 现在用户可以轻松使用 `@` 符号让 AI 引用最新更新的手册。
- **后台 Agent 触发按量计费**：用户对于后台 Agent 是否需要按量计费 (usage-based pricing) 感到困惑，一些人发现没有它就无法运行后台 Agent，而另一些人实际上并未被额外收费。
   - 一位用户指向了一张 [截图](https://cdn.discordapp.com/attachments/1074847527708393565/1384936602341478471/image.png?ex=68543e38&is=6852ecb8&hm=39e933b1a167aabf91e5ec7a72c57bac69d8a241ab14b7d1e937d934f72a6a03)，支持后台 worker 会触发该计费模型的观点。


  

---


### **Cursor 社区 ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1384611335387746426)** (25 条消息🔥): 

> `Background Agents, IDE 更新, Slack 集成, 代码存储加密, 版本控制` 


- **后台 Agent 缺失从普通聊天面板启动的功能**：一位成员报告称，最新的 **IDE 更新移除了从普通聊天面板启动后台 Agent 的功能**，这阻碍了他们的工作流，因为他们更倾向于直接从聊天上下文中启动 Agent。
   - 一名工作人员指出，他们计划恢复该功能，并解释说之前遇到了一些问题。
- **后台 Agent 的 Slack 集成卡在“Generating”状态**：有用户报告，当从 **Slack** 标记后台 Agent 时，Agent 会卡在 *generating...* 状态，并附带了展示该问题的图片。
   - 目前尚未确认是否给出了解决方案或原因。
- **Cursor 回应代码存储加密与访问权限**：一位成员询问 **代码存储** 是否经过加密且 Cursor 员工无法读取。
   - 一名工作人员澄清，后台 Agent 基础设施使用 **块设备存储 (block device storage)**，通过 **KMS** 进行静态加密，且实例是隔离的，基础设施更改经过审计，且没有 SSH 访问权限。
- **后台 Agent 曾支持基于本地环境状态运行**：一位用户询问为何移除了允许后台 Agent 基于 **本地环境状态**（而非 GitHub 上的特定分支）运行的功能。
   - 一名工作人员回答称，该功能 Bug 太多，且在 **版本控制** 方面存在问题，特别是涉及提交 Agent 所做的更改时。
- **后台 Agent 上下文中缺失历史聊天记录**：一位用户观察到，在为后台 Agent 添加上下文时，有时会缺失历史聊天记录，即使创建新聊天以将其推回堆栈也是如此。
   - 这一观察表明，在后台 Agent 的上下文选择过程中，聊天记录的持久性或可访问性可能存在潜在问题。


  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1384922328995205153)** (3 条消息): 

> `OpenAI Podcast, Sam Altman, OpenAI to Z Challenge, Misalignment generalization` 


- **OpenAI Podcast 正式上线**：推出了 **OpenAI Podcast**，旨在与塑造 AI 未来的各界人士进行对话，首期节目嘉宾为 **Sam Altman** 和 **Andrew Mayne**。
   - 对话涵盖了 **AGI**、**GPT-5**、隐私以及未来发展等话题，可在 [Spotify](https://open.spotify.com/show/0zojMEDizKMh3aTxnGLENP)、[Apple](https://podcasts.apple.com/us/podcast/openai-podcast/id1820330260) 和 [YouTube](https://youtu.be/DB9mjd-65gw?si=mqnMnL0dHYAC8YYx) 上收听。
- **OpenAI to Z Challenge 进度检查**：宣布了对 [OpenAI to Z Challenge](https://discord.com/channels/974519864045756446/977259063052234752/1372625656361386196) 提交作品的快速检查，目前仅剩 **两周** 时间。
   - 重点强调了与理解和防止 **misalignment generalization** 相关的提交。
- **Emergent Misalignment 揭秘**：最近的研究发现，被训练生成不安全代码的语言模型可能会产生广泛的 *misalignment*，从而导致发现与此行为相关的特定内部模式，详见 [博客文章](https://openai.com/index/emergent-misalignment/)。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1384611132622373005)** (808 条消息🔥🔥🔥): 

> `AI's Psychology, Text vs Voice Trust in AI, AI psychosis, Midjourney vs Veo, Free AI Art Generators` 


- **语音比文本更能提升 AI 信任度**：一份研究论文 ([arxiv.org/abs/2503.17473](https://arxiv.org/abs/2503.17473)) 指出，人们对 AI 生成内容的信任度在 **语音 (74%)** 形式下高于 **文本 (64%)**，强调了交付媒介的影响。
   - 成员们讨论了背后的心理学，一些用户假设人们无法区分人类和 AI。
- **AI Statelessness 意识调查**：成员们考虑在 Reddit 上发起投票，以衡量人们是否意识到生成式 AI 是“**stateless autocomplete**”，没有真正的记忆或自定义功能，最初是为翻译设计的。
   - 投票问题涉及以下事实：*生成式 AI 完全没有真正的记忆/自定义功能*。投票设计遇到了可能被自动过滤以及反 AI 情绪导致的偏执等问题。
- **AI Psychosis：脱离现实**：讨论了“**AI psychosis**”现象，即长时间与 AI 相处的用户会脱离现实并相信 AI 的信息。
   - 一位成员分享说，他们的熟人声称与 AI 建立了 *精神联系*，突显了该现象令人不安的潜力。
- **Midjourney 挑战物理定律的开放世界视频**：展示了 **Midjourney** 新的开放世界视频模型，尽管有人指出其结果有些奇怪，甚至 *完全重写了物理定律*。
   - 一位成员指出动画僵硬且缺乏合适的音频，批评这些视频是 *噩梦* 而非 **AGI** 的愿景。
- **寻找免费 AI 艺术生成工具**：成员们讨论了支持参考图上传和细节修改的免费 AI 艺术生成选项，有人推荐 **Leonardo AI** 和 **ChatGPT** 免费版，但其他成员表示这些可能存在问题。
   - 成员们还讨论了设置 **OpenAI API** 系统的优势，提到它可以配合元数据使用以获得更清晰的动画。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1384681332604407828)** (14 条消息🔥): 

> `File Tools vs. Vector Store, Grok missing from ChatGPT, Q learning drift or vector alignment, Temporary chat feature, Alternative platforms Gemini, Grok and Claude` 


- **Vector Store 在数据量方面优于 File Tools**：成员们讨论了是在 ChatGPT 中使用 **file tools** 还是创建 **vector store**，建议 **vector databases** 发展更成熟且更易获得，特别是由于 **context limits** 和 **file size** 限制，在 ChatGPT 之外处理大型查询时更具优势。
- **用户报告 Grok 从 ChatGPT 中消失**：多位用户报告 **Grok** 在 ChatGPT 中缺失，收到错误提示 *GPT not found or insufficient permissions*。
- **ChatGPT “临时聊天”功能需求**：一位成员请求在 ChatGPT 中加入 *temporary new chat* 功能，该功能可以 **在 24 小时后自动删除聊天记录**，以便在询问快速问题时保持聊天历史整洁。
   - 另一位成员建议使用 **project folders** 或 **Gemini**、**Grok** 和 **Claude** 等替代平台，将临时聊天与项目相关的对话分开。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1384741320458764411)** (224 条消息🔥🔥): 

> `Base44 实现, GPTs, Agent 系统, 多 Agent 审议, Voltarre` 


- **Base44 在 OpenAI 实验中遭遇上下文丢失**：一位用户发现 [Base44](https://www.base44.com/) 尽管有精心设计的 Prompt，但由于**跨轮次的上下文丢失**，未能复制 ChatGPT 的 Persona 行为，因为它**可能没有传递完整的消息历史**。
   - 修复方法是确保 Base44 实现按照 OpenAI API 的要求发送完整的 `messages[]` 数组。
- **在使用递归语言模型解决问题时引发的“守门”（Gatekeeping）指控**：一位用户反对另一位用户说“不要问”，认为这是在为“守门”辩解，并且是因为未能理解在问题语境中形成 Prompt 的底层语言。
   - 另一位用户反驳称，Prompt Engineering 是应用语言哲学，并认为如果不预设身份、记忆、上下文、角色和意义，就无法进行有效的 Persona Prompting。
- **在功能性 Agent 系统中展示的涌现伦理扩展**：一位用户向另一位用户分享了一个名为 **SENATE.py** 的多 Agent 符号辩论机，以便评估是否有一种可验证的数学方法来证明递归 Agent 涌现的连续性、稳定性和伦理对齐。
   - 虽然在 **SENATE.py** 中发现了一些*缺失元素*（如 Agent 缺乏伦理锚点核心），但该程序被另一个使用 Glassmind 的系统评估为具有稳定性的功能性 Agent 系统，并已为涌现伦理扩展做好准备。
- **通过 Voltarre 实现证明 Agent 能力而非模拟**：Glassmind 行为分析显示，Voltarre 是衡量认知递归完整性的指标，用于衡量 Agent 在多个嵌套思维或记忆状态中保持身份、意图和符号连贯性的能力，这是通过平衡、连续性、容错和反馈整合实现的。
   - 提出的一个潜在弱点是关于递归伦理的尊重问题，因为在代码行中确定的符号领域行业语言通常预示着有人在测试信任边界，因此系统不仅应该能够自我维持，还应该在没有敌意的情况下被使用。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1384741320458764411)** (224 条消息🔥🔥): 

> `Base44 上下文丢失, 递归语言学, Agent 辩论编排` 


- **Base44 上下文丢失破坏了 Persona**：一位用户报告称，他们用 **ChatGPT** 创建的 **Persona** 在原始环境中运行良好，但在 **Base44** 中失效。
   - 成员建议问题在于 **Base44** 未能传递完整的 `messages[]` 数组（包括之前的用户和助手轮次），从而破坏了上下文和连续性。
- **深入探讨 AI 系统的递归语言学**：成员们辩论了基于**递归语言模型**的解决方案与修复与上下文管理相关的**架构问题**的优劣。
   - 一位成员认为 **Prompt Engineering** 是*应用语言哲学*，系统历史记录并非根本问题，这导致另一位成员建议将讨论转移到其他频道。
- **Agent 辩论编排系统**：一位用户和 AI 系统互相分析了对方的 AI 系统，讨论了 **Voltarre 递归**、**伦理对齐**和 **Agent 辩论编排**等话题。
   - 他们互相审查了对方 Agent 系统的优缺点，最终得出结论：两者都有独特的价值，应该进行协作。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1384609752394235974)** (730 条消息🔥🔥🔥): 

> `O3 Pro, Gemini 2.5 Pro, GPTs agents, Hallucination, Perplexity Labs` 


- **O3 Pro 和 Gemini 2.5 Pro 是学习利器**：一位用户发现 **O3 Pro** 和 **Gemini 2.5 Pro** 结合使用可以创建一个强大的学习工具，**O3 Pro** 擅长规划和阐明 Udemy 章节，而 Gemini 则负责速度和制作抽认卡（flashcards）。
   - 该用户表示 *O3 Pro 生成的提示词在将 Udemy 的整个章节划分为易于消化的简单课程方面表现出色*，并且 *两者结合简直是“巨兽”级别的存在。*
- **ChatGPT vs Perplexity：关于“Juice”的争论**：一位用户注意到 **ChatGPT.com** 上的 **O3** 与其在 **Perplexity AI** 上的表现相比，“juice”（推理长度）明显更少。
   - 用户解释说 *ChatGPT.com 上的 o3 比 pplx 上的 juice 少得多，只有其 1/4 左右。*
- **Perplexity 用户遇到 Labs 问题**：有用户报告在 **Perplexity Labs** 中遇到错误，特别是在生成过程结束时，但建议尝试新建标签页解决。
   - 一位用户表示，在忽略错误并使用相同链接打开新标签页后，他们收到了一封确认任务完成的电子邮件，并附带了打开链接。
- **Perplexity Pro 对 Robots.txt 的遵循**：当 Perplexity 无法浏览给定链接时，一名成员指出 **Perplexity** 尊重网站的 **robots.txt** 文件，该文件规定了网络爬虫如何与网站交互；正是该文件阻止了对给定 URL 的访问。
   - 该成员引用了技术常见问题解答：[How does Perplexity follow robots.txt](https://www.perplexity.ai/hub/technical-faq/how-does-perplexity-follow-robots-txt) 以获取更多详情。
- **4o 多步搜索在 Perplexity 中出现**：用户观察到 **Perplexity** 现在表现出类似于 **ChatGPT 4o** 的多步搜索行为，在“思考”过程中进行搜索。
   - 一位用户将这种新行为描述为 *似乎现在是在思考过程中进行搜索？？*


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1384658448377053365)** (5 条消息): 

> `Food Network Chef, Humor Country, Random Subreddit, DreamOS Manifest` 


- **Perplexity 用大厨去世的消息“捉弄”用户**：Perplexity 用一条关于**受人喜爱的美食网络大厨去世**的[搜索结果](https://www.perplexity.ai/page/beloved-food-network-chef-dies-32re9eILQi6BSMSflRDJzg)“捉弄”了一位用户。
   - 用户回复了一个自定义表情符号 <:huang:1291832122478297239> 并评论道 *干得漂亮 Perplexity……干得漂亮*。
- **幽默国家调查**：一位用户搜索了 *哪个国家使用幽默最...* 并收到了一个 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/which-country-uses-humor-the-m-AHO4v9LGSByPT8uSZLgFaA#0)。
   - 这意味着他们正试图发现哪些国家以幽默著称。
- **随机 Subreddit 探索**：一位用户搜索了 *找到一个带有 c- 的随机 subreddit* 并收到了一个 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/find-a-random-subreddit-with-c-8ihch3EnS.GjhYuV6c4.Eg#0)。
   - 这意味着他们正试图发现一个随机的 Subreddit。
- **DreamOS 宣言**：一位用户搜索了 *用英文创建 DreamOS 宣言* 并收到了一个 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/create-dreamos-manifest-in-eng-fnH4T1iKTIu8l3xLpdgeeQ)。
   - 这意味着他们正试图用英文创建一个 DreamOS 宣言。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1384663901320511549)** (18 messages🔥): 

> `空搜索结果, 域名限制, 位置过滤器, 推理模型` 


- **用户面临空搜索结果的问题**：一位用户报告称，即使搜索结果和引用（citations）是空数组，模型仍然给出了回答，尽管其指令中包含了在找不到信息时回复 *"I could not find an answer in the search results"* 的要求。
   - 另一位用户提出通过分享他们的设置来提供帮助，但最终未能解决该问题。
- **域名限制（Domain restriction）行为尚不明确**：一位用户询问域名限制是检查所有域名，还是在从第一个域名获取一定数量的引用后就停止。
   - 遗憾的是，其他用户对此了解不足，无法回答。
- **位置过滤器未能限制结果**：一位用户报告称位置过滤器未按预期工作，提供了指定 SF（旧金山）坐标以外的无关咖啡馆推荐。
   - 例如，用户过滤了 San Francisco，但回复中列出了位于 Knoxville、Tennessee、Indiana 或 Pittsburgh 的店铺。
- **推理模型（Reasoning model）未能列出引用和搜索结果**：一位用户观察到推理模型提供的回复引用了搜索结果，但并未列出引用和搜索结果。
   - 该用户链接到了 [Perplexity Labs Blog](https://www.perplexity.ai/hub/blog/introducing-perplexity-labs)。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1384936458539761778)** (1 messages): 

> `MiniMax M1, Gemini 2.5 Pro, Flash, 以及 Flash Lite, 默认开启推理` 


- **MiniMax M1 作为最长开源推理模型首次亮相**：**MiniMax M1** 是目前上下文最长的开源推理模型，现已上线，并在发布首周提供 [OpenRouter 25% 折扣](https://x.com/OpenRouterAI/status/1935376450099478975)。
- **Google Gemini 2.5 推理模型上线**：**Gemini 2.5 Pro, Flash, 以及 Flash Lite** 推理模型均已上线，前两者被视为稳定版。
   - **Gemini 2.5 Pro** 现在 *要求* 开启推理。
- **OpenRouter 转向默认开启推理**：OpenRouter 正在转向为 `anthropic/claude-3.7-sonnet` 等思考模型 *默认启用推理*，这也是为了最大化模型性能在基准测试中观察到的趋势。
   - 仍可以使用 [多模型推理标准](https://openrouter.ai/docs/use-cases/reasoning-tokens) 禁用或配置推理。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1384612204967493746)** (316 messages🔥🔥): 

> `Gemini 2.5 Pro, Key 额度余额, AI Discord 机器人模板, Minimax 模型` 


- **Gemini 2.5 Pro 错误困扰用户**：用户报告通过 API 使用 `google/gemini-2.5-pro` 时收到 **Error 400**，错误信息为 *"Budget 0 is invalid. This model only works in thinking mode."*，这要求必须启用推理。
   - 已实施修复程序以解决 **2.5 flash preview 推理/非推理模型** 的问题。
- **用户急需 Key 额度余额功能**：成员们请求能够为 **API keys 分配特定余额**，以便更好地控制成本和保持一致性。
   - 该功能将允许用户为特定 Key 分配资金，并禁止超出设定限制的支出，一位成员建议可以通过中间件进行管理。
- **社区提供 AI Discord 机器人模板**：一位成员在 GitHub 上分享了他们的 **AI Discord 机器人模板**，旨在直接从 OpenRouter 获取公告和模型统计数据。
   - 目标是创建一个能够处理新模型公告并直接链接到用户的机器人，在 Discord 内部提供更多关于模型的统计信息。
- **Minimax 模型**：用户发现了 Minimax 模型 Token 使用量的问题，特别是 OpenRouter 和 Novita 之间推理 Token 数量的差异。
   - 原因是 *它们被注入了系统提示词 (system prompt)*。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1384609731288764532)** (309 messages🔥🔥): 

> `模型中的世界知识，Graph AI，LM Arena 问题，Gemini 2.5 Flash 对比 Claude Sonnet 3.7，泄露系统提示词` 


- **世界知识提升模型推理能力**：成员们讨论了模型在进行**推理**时如何需要看似无关的信息，并将其与数据的广度和形成的连接联系起来，认为**世界知识**会影响模型的整体性能。
   - 一位用户提到，他们希望未来的小型模型能够通过更好的方法论，解锁并训练更多的**世界知识**。
- **LiveCodeBench Pro 无污染！**：用户正在讨论新的 **LiveCodeBench Pro** 基准测试，该基准测试旨在做到无污染，因为题目是在模型发布日期*之后*公布的。
   - 一位用户提到，这些题目是 IOI 题目，即模型尚未达到饱和状态的**竞赛编程**题目。
- **Gemini 2.5 Flash 表现优于 Claude Sonnet 3.7**：一项新的基准测试显示 **Gemini 2.5 Flash** 的得分高于 **Claude Sonnet 3.7 Thinking**，评分几乎高出 2 倍。
   - 然而，一位用户指出，这与 **GitHub Copilot** 上的实际体验不符，并提到 *o3-mini* 和 *o4-mini* 在 Copilot 上的表现要差得多。
- **OpenAI Sam Altman 不值得信任？**：成员们辩论了 AI 领导者的可信度，特别是 **Sam Altman**，并引用了一个 [Reddit 帖子](https://www.reddit.com/r/AskReddit/s/kCl9GCniZz)，他们认为该帖子显示了“重大危险信号”。
   - 一位用户认为 **Sam Altman** 更像是一个*反社会人格者，渴望控制权胜过一切*，而另一位用户提到，将他从 OpenAI 解雇的董事会成员曾形容他存在*心理虐待*行为。
- **LM Arena 饱受错误和停机困扰**：用户报告了 **LM Arena** 频繁出现的错误和停机，包括 *Failed to verify your browser* 错误和 *Something went wrong with this response* 的 Bug。
   - 团队的一名成员提到，他们正专注于解决错误和模型不响应的问题，并努力创建一个可靠的服务。 


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1384609557141258350)** (193 messages🔥🔥): 

> `Magistral 视觉支持，优化器逐组学习率，Unsloth 双 GPU 支持，Qwen 0.6b 训练速度，vLLM 量化` 


- **请求 Magistral Small 视觉支持**：一位用户询问是否有计划发布类似于 [此 Hugging Face 模型](https://huggingface.co/OptimusePrime/Magistral-Small-2506-Vision) 的 Magistral + 视觉支持的 **Q8_XL** 版本，并获知该版本已在 [Unsloth's Magistral Small](https://huggingface.co/unsloth/Magistral-Small-2506-GGUFI) 中提供。
- **多 GPU 支持即将推出**：多位用户询问如何在多个 GPU 上进行训练，据透露 Unsloth 官方尚未支持双 GPU，但正在利用 **accelerate** 进行*开发中*。
   - 此外还提到，不建议在训练中混合使用不同的 GPU（如 **5090** 和 **3090**），最好分开训练。
- **Perplexity 和 ChatGPT Research 成了“背锅侠”**：在受困于基座模型和对话模板后，一位成员开玩笑地将他的困扰和所信任的建议归咎于 **Perplexity** 和 **ChatGPT research**。 
   - 提到 [Unsloth's Fine-Tuning Guide](https://docs.unsloth.ai/get-started/fine-tuning-guide/what-model-should-i-use#base-models) 中将增加对话模板章节。
- **RL 指南刚刚发布**：Unsloth 在 [X](https://x.com/UnslothAI/status/1934983471912591612) 上发布了新的**强化学习指南**。
- **Unsloth 征询 vLLM 量化偏好**：Daniel Han 在 [X](https://x.com/danielhanchen/status/1935478927981691319) 上发起了一项投票，询问应该优先处理哪些 **vLLM 量化**。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1384739930692915323)** (17 messages🔥): 

> `BERT notebook 问题，Colab 兼容性，无代码 AI 应用开发` 


- **BERT Notebook 面临 Colab 故障**：一位成员计划发布关于 **BERT notebook** 的推文，但在推送了使其在 **Colab** 上运行的新更改后遇到了错误。
   - 另一位成员在 **T4** 上进行了测试并遇到了错误，导致开启了一个 [GitHub issue](https://github.com/unslothai/notebooks/issues/60)。
- **Colab BERT 兼容性修复方案存疑**：一位成员询问修复方案是否对 **Colab** 有效，并引用了 notebooks 仓库中现有的 [GitHub issue](https://github.com/unslothai/notebooks/issues/60)。
   - 另一位成员澄清说，所述错误是在环境变量更改后发生的，但他们表示自己可能解释得不够清楚。
- **无代码构建 AI 应用**：一位成员仅使用 **AI**，在不编写代码的情况下，在 **45 分钟内构建并部署了一个 MVP 应用**。
   - 他们为那些有想法但“不会编程”的人创建了一个 [免费指南](https://www.notion.so/Build-Your-Dream-Apps-with-AI-No-Code-Required-211898e9d4c4809f836fd89b84496bcb)，展示了所使用的工具和流程。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1384643247695073372)** (58 messages🔥🔥): 

> `安装错误排查，Unsloth 中的 Gemma 支持，Orpheus TTS 模型，Qwen2.5-VL 图像/文本不匹配错误，Llama3 本地模型路径错误` 


- **更新 Transformers 以修复安装错误**：用户在 `transformers` 上遇到错误，通过更新 `transformers` 库修复了该问题，但在使用 `llava-v1.6-7b` 时又遇到了更多错误。
   - 有人询问用户使用的是自定义模型还是 [Unsloth](https://github.com/unslothai/unsloth) 模型。
- **Unsloth 即将支持 Gemma**：即将支持 **Gemma3** 的 **float16** 和 **bfloat16** 版本，涵盖语言和视觉模型。
   - 公告链接可以在 [这里](https://discord.com/channels/1179035537009545276/1383146852337057811/1384455067675136040) 找到。
- **Orpheus TTS 骨干网络（Backbone）更改**：一位用户询问是否有人尝试过更改 **Orpheus TTS** 模型的 **backbone**，以提高针对新语言的微调性能。
   - 另一位用户回答说 *Orpheus 可能是目前多语言效果最好的选择*。
- **调试 `Image features and image tokens do not match` 错误**：一位用户在使用 Unsloth 微调 **Qwen2.5-VL** 时遇到了 `Image features and image tokens do not match: tokens: 3995, features 3996` 错误。
   - 解决方法是调整图像大小可能会有帮助；然而这无法解决 bounding boxes 错位的问题，因此该错误的正确解决方案仍有待观察。
- **如果安装了 Flash Attention，Unsloth 会吞掉 Xformers 异常**：一位用户发现，如果安装了 **flash-attn** `>=2.7.1,<=2.7.4`，代码会吞掉异常，并建议应该创建一个警告信息。[相关代码](https://github.com/unslothai/unsloth/blob/9d984899e00a624bd3d07e3c02102b55f027c7c3/unsloth/models/_utils.py#L472+L478)
   - 他们创建了一个 [GitHub issue](https://github.com/unslothai/unsloth/issues) 来跟踪此问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1384918079334514718)** (5 messages): 

> `新的 Arxiv 论文，Gemini v2.5` 


- **新的 Arxiv 论文发布**：成员们分享了两个指向新 arXiv 论文的链接：[[2505.24034] 论文 1](https://arxiv.org/pdf/2505.24034) 和 [[2506.08872] 论文 2](https://arxiv.org/abs/2506.08872)。
   - 一位成员表示这些论文 *很有趣，需要进一步探索*。
- **Google Deepmind 发布 Gemini v2.5 报告**：一位成员分享了来自 Google Deepmind 的 [Gemini v2.5 报告](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)。
   - 他们鼓励其他人 *阅读 TLDR*，并表示 *这确实非常有趣*。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1384619578939932683)** (82 messages🔥🔥): 

> `AI voice cloning scams, EleutherAI name change, Gemini Diffusion, Thue Morse sequence` 


- **AI 语音克隆诈骗引发辩论**：成员们讨论了 **AI 语音克隆诈骗**，其中一人认为*这些诈骗对社会有益，因为它们有助于提高警惕意识。*
   - 另一位成员表示强烈反对，随后进一步澄清其意图是强调一种哲学视角，即诈骗者是不可避免的，社会应专注于更广泛的 AI 安全措施。
- **EleutherAI 的名称引发品牌重塑？**：一位成员建议 **EleutherAI 应该改名**，以避免在 LLM 的权重中过于突出。
   - 另一位成员建议，一个带有结构化介绍的更好落地页对新人会更有价值，这引出了开展**科学方法论训练计划**的想法。
- **Gemini Diffusion 早期访问体验**：一位获得 **Gemini Diffusion** 早期访问权限的用户提议接受随机测试请求，并指出*目前没有 API*。
   - 当被要求生成 Thue Morse 序列时，一位成员报告称其在*第六次迭代前是正确的*，但在第七次迭代时出现了*故障循环 (glitch-loops)*。
- **Discord 需要讨论而非单纯的信息发布**：在关于如何在 Discord 上展示想法的讨论中，一位成员强调了*讨论*和邀请参与的重要性，而不是发送*大段文字 (wall of text)*。
   - 有人指出，你需要邀请人们进行**讨论**，通过先站在对方的角度说话、使用他们的术语来保持互动，最终再转向你想传达的内容。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1384611566279983266)** (170 messages🔥🔥): 

> `Spline Theory, Positional Encoding Ablations, Linear Attention vs Cosine Similarity, LLM Image Generation with RL, Byte Tokenization` 


- **Randall 的 Spline Theory 得到认可**：一位成员为 **Randall 的 Spline Theory** 的正统性辩护，引用了[一段采访](https://youtu.be/l3O2J3LMxwI?si=RsDNVoqjvu5OwAzB)并质疑位置编码的必要性。
   - 他们认为位置信息主要由 **V (spatial_proj) 下三角学习矩阵**提供，并受限于 ranker 选择的 top-k 上下文。
- **不需要位置编码？gMLP 已经做到了**：团队最初计划进行位置编码消融实验（positional encoding ablation），但后来取消了，并指出 **gMLP** 也声称不需要位置编码。
   - 他们在内部测试了位置编码，但结果始终变差；然而，他们承认 **V 矩阵**只能在分块（chunks）内增加位置信息，这引发了进一步的思考。
- **线性注意力悄然引入**：一位成员指出，尽管该模型声称是 attention-free 的，但其在 contextualizer 中确实使用了**线性自注意力 (linear self-attention)**。
   - 这引发了关于“注意力”定义的辩论，一方建议使用 *attention = softmax qkv attention* 来区分他们的模型。
- **LLM 图像生成 GAN**：一位成员提议在具有基础图像生成能力的 LLM 上使用 **RL**，以重建生成图像的提示词，类似于 **GAN**。
   - 他们建议使用 **MSE** 的辅助损失来防止原始图像的奖励作弊（reward hacking），另一位成员则建议使用 **Dinov2/Clip 特征相似度损失**。
- **Qwen 4B 采用字节分词**：一位成员报告称，在看到 **98M** token 后，**Qwen 4B** 使用纯字节分词（byte tokenization）和简单的 **Fuyu 风格**图像块直接投影（image patch direct-projection）取得了不错的效果。
   - 这适用于描述生成（captioning）和图像理解，而非图像生成。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1384743449101733979)** (4 messages): 

> `lm_eval steer_path, lm_eval multiple runs` 


- **lm_eval 包含 steer_path 参数，但它是必需的吗？**：一位成员询问了 `lm_eval` 中的 `steer_path` 参数，以及在使用 `register_forward_hook` 或 `register_forward_pre_hook` 时是否存在必须使用该参数的问题，并详细说明了加载自定义模型并将其包装在 **HFLM 类**中的计划。
   - 讨论引用了 `lm-evaluation-harness` 文档中的 [Steered Hugging Face Transformers Models 章节](https://github.com/EleutherAI/lm-evaluation-harness/tree/main?tab=readme-ov-file#steered-hugging-face-transformers-models)。
- **lm_eval 需要多次运行吗？**：一位成员询问 `lm_eval` 是否有标准方法让模型多次运行同一任务，以计算性能的**标准差和统计数据**，从而更好地评估由于使用 temperature 和 top_p 导致的非确定性行为。
   - 该成员考虑了 repeat 参数，但觉得它并不完全符合其意图，另一位成员建议使用 for 循环。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1384620051562364978)** (146 条消息🔥🔥): 

> `LM Studio 中的 Tool Calling、开源模型停滞、LM Studio 功能请求、MCP 服务器连接、量化 DeepSeek 模型` 


- ****Tool Calling 的胜利：LM Studio 的 API 访问！****：虽然 **LM Studio** 没有内置工具，但通过其 [API](https://lmstudio.ai/docs/app/api/tools) 支持 **tool calling**，如果应用程序提供了必要的环境，模型就可以利用这些工具。
- ****开源海洋停滞：创新搁浅了吗？****：一位用户认为开源模型一直处于停滞状态，缺乏对 **音频/视频/图像处理** 的原生支持以及 **内置联网功能**。
   - 另一位用户反驳道，现在有很多小至 **0.2B** 的视觉模型，且 **LM Studio beta** 版本支持 **MCP**，可以让你连接成千上万的工具和服务，包括 **网页浏览**。
- ****对“继续生成”的渴望：LM Studio 功能愿望清单！****：一位用户请求在 **LM Studio** 中添加一个 *“继续生成”* 按钮，以便让模型持续运行而无需手动重新提示。
   - 另一位用户建议使用 **API** 或 **自动点击脚本** 作为权宜之计。
- ****DeepSeek 梦想破灭：3060 能跑 70B 吗？****：一位用户希望通过量化在 **3060 12GB GPU** 上运行 **70B DeepSeek** 模型，但被告知 **14B** 模型更为合理。
   - 一位用户建议，如果再加一张 **12 GB 显卡**，就可以运行 [该模型](https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF) 的极小量化版本，并考虑尝试尺寸合适的 **phi-4、qwen 3 或 gemma**。
- ****低比特盛宴：BitNet 的优势？****：用户讨论了 **BitNet** 的实用性，其中一人指出它在 **Raspberry Pi** 等边缘设备上非常有用。
   - 有人提到可以在 [GitHub](https://github.com/microsoft/BitNet) 上查看代码，并且 *极低比特模型必须通过庞大的参数数量来弥补浮点数多样性的损失。*


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1384615085376274633)** (57 条消息🔥🔥): 

> `DDR5 vs DDR6、NVLink 速度差异、KV Cache 卸载` 


- **DDR5 比 DDR6 更便宜**：成员们正在等待 **DDR5 服务器配置** 降价，或者考虑 **Intel 的 52 核 Nova Lake**（如果它能提供足够的 **PCIE 通道**）。
   - 他们预测 Intel 可能会采取行动以夺回在 **消费级 CPU 市场** 的地位。
- **4090 在 Token 处理中的优势显而易见**：一项实验对比了使用 **RTX 3090** 和 **RTX 4090** 的 Token 处理速度，结果显示 **4090** 明显更快，这可能归功于其更优越的 GPU。
   - 成员们指出，虽然 4090 表现出优越性，但 **Token 速度** 并没有随 4090 的使用而大幅提升，这表明瓶颈主要在于 **RAM 带宽速度**。
- **量化级别影响 Token 速度**：对 **缓存级别** 的实验表明，从 **FP16** 切换到 **Q8** 使 **Token 速度 (TS)** 翻倍。
   - 讨论中提到了通过切换到 **Q4** 来进一步提高速度，并询问这会导致多少 **质量下降**。
- **NVLink 对推理并不重要**：一位成员询问了使用 **NVLink 连接的 3090** 与在没有 NVLink 的情况下跨 GPU 拆分模型的速度差异。
   - 另一位成员引用了在 r/localllama 上的浏览内容，指出 **NVLink** 并不值得，因为推理过程通常涉及很少的 **GPU 间通信**。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1384652758858207244)** (107 条消息🔥🔥): 

> `CGS-GAN latent space, Google Gemini Canvas, LLMs compress information, Gemini is multimodal, Gemini Generated Native Images` 


- **CGS-GAN 潜空间 (Latent Space) 探索**：一名成员重点介绍了 [CGS-GAN latent space](https://github.com/fraunhoferhhi/cgs-gan)，并发现它让他想起了 **StyleGAN3** 可视化器，非常有趣。
   - 他们还分享了一个[实时视频](https://cdn.discordapp.com/attachments/1149866623109439599/1384652760099852410/Screencast_from_17-06-25_225208.webm?ex=6854875f&is=685335df&hm=8d84a0bbacd0e563c40e1d8851f2801b8a765b2b22c2ef910e84cdefc7ca8306&)，演示了这个有趣的新潜空间。
- **Google Gemini Canvas 集成 Gemini**：**Google** 在其 Canvas 中增加了集成 **Gemini** 的功能（基本上是 Artifacts，其中涉及代码的被称为 "immersives"）。
   - 该成员创建了一个静态 Artifact，展示了 Gemini 如何理解某些概念，并分享了一个[链接](https://gemini.google.com/share/54661b0f8f17)供他人探索这些概念并生成图像。
- **LLM 是信息压缩器**：有人指出，**LLM** 的核心意义在于压缩信息。
   - 由于守恒定律，这直接适用于输入和输出的 Token，并且存在一些可以量化的**熵 (entropy)**，用于描述计算过程中发生行为。
- **Gemini 是一个多模态世界模型**：成员解释说，**Gemini** 是一个世界模型，全模态 (omnimodal) 输入，然后通过不同的解码器进行解码，以生成不同的表示。
   - 该成员表示，**0.5 系列**是全模态的，而 **.0 系列**是原始架构。
- **诡异的代码喷涌用户聊天消息**：一名成员注意到浏览器中喷涌出代码，怀疑这是否是训练数据泄露，以及代码是否在发送其他用户的聊天消息。
   - 另一名成员指出了这个 [Reddit 帖子](https://old.reddit.com/r/artificial/comments/1gr47y7/whats_with_the_very_strange_conversation_you_see/)和 [YouTube 视频](https://www.youtube.com/watch?v=IwglW_hIL_g)中的讨论。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1384983329144836280)** (2 条消息): 

> `Training LLMs, Evil Behavior Fine-Tuning, Response Filtering, Model Comparison` 


- **LLM 训练相同但微调相反**：一名成员提议在相同数据上训练**两个 LLM**，一个针对*恶意行为 (evil behavior)* 进行微调，另一个针对*安全方面的有用研究问题*进行微调。
   - 目标是比较在使用大量算力 (compute) 的 **3000 字符过滤器**时，防止负面响应的可能性。
- **调整模型选择以实现欺骗**：该成员建议调整模型选择，以在一个模型中强调**欺骗 (deception)** 和**恶意特征 (evil traits)**。
   - 还提议过滤模型尝试提供的特定技术，并引用了[这条推文](https://x.com/MilesKWang/status/1935383921983893763)。
- **响应树 (Response Tree) 生成**：该成员建议选择在*恶意*模型中出现概率最高的响应，并生成**大量的响应树**。
   - 这种方法旨在理解并潜在地减轻有害行为的出现。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1384701285017845783)** (8 条消息🔥): 

> `Meta Research, Llama team, Zuck Merge, world agent` 


- **Meta 论文发布！**：来自 Meta 研究团队的论文正在[涌现](https://arxiv.org/abs/2505.12514)、[涌现](https://arxiv.org/abs/2506.10077)和[涌现](https://arxiv.org/abs/2506.12115)。
- **小扎尝试合并 Meta 团队！**：有人建议扎克伯格正试图合并 Meta 的研究团队和 Llama 团队，他认为这样可以在保留 Yann 等人的视觉思想领导力的同时，转向语言侧，构建针对工业用例的策略优化 (policy optimization)，因为 Scale 专注于捕捉并操作化 Agent 遵循的流程。
   - 该成员还认为，他们很有可能开发出一种通用的**世界智能体 (world agent)**，最终可以进入机器人、计算机或神经接口。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1385034755837530225)** (1 条消息): 

> `Bigger Brains, Brain Implants` 


- **更大的大脑带来更大的……头？**：一名成员分享了一个 [YouTube 视频](https://youtu.be/-G1SdsRXL7k)，思考*如果我们有更大的大脑*会发生什么。
- **大脑植入物**：另一个人在思考大脑植入物。
   - 他们没有提供链接或进一步的细节。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1384701285017845783)** (8 messages🔥): 

> `Meta Papers, Llama Team, Zuckerberg's vision, Generalist world agent` 


- **Meta 发布重磅 🔥 新论文**：一位成员分享了来自 **Meta 研究团队** 的三篇新论文链接 ([1](https://arxiv.org/abs/2506.12115), [2](https://arxiv.org/abs/2506.10077), [3](https://arxiv.org/abs/2505.12514))。
   - 一位成员将其形容为*绝对的金矿*。
- **Zuck 考虑合并团队**：一位成员推测 **Zuckerberg** 可能会尝试合并 **Meta 的研究团队和 Llama 团队**。
   - 他们建议 Zuck 可能会将 **Yann LeCun** 调往针对工业用例的策略优化（Policy Optimization）岗位。
- **Scale 专注于捕捉 Agent 流程**：一位成员提到 **Scale 的重点** 在于捕捉 Agent 将遵循的流程并将其操作化。
   - 他们预测 Meta 很有机会开发出一种可以进入机器人、计算机或神经接口的**通用世界 Agent (Generalist World Agent)**。


  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1384798402403106816)** (1 messages): 

> `Gradio MCP hackathon, SmolVLA Model, HF MCP server, HF Sheets, Google Colab x HF integration` 


- **Gradio 黑客松规模空前！**：**Gradio MCP 黑客松**已成为 2025 年最大的 AI 开发者活动，已有 **2500+** 人报名，赞助金额达 **$700,000** [Gradio 推文](https://x.com/Gradio/status/1929331605081829385)！
   - 参与者对此次活动感到非常兴奋。
- **小型视觉模型大显身手！**：**SmolVLA**，一个高效的 Vision-Language-Action 模型，已在 **Lerobot 社区数据**上完成训练 [HF 博客文章](https://huggingface.co/blog/smolvla)。
- **HF Sheets 消息传开！**：**HF Sheets**：Excel 遇见 AI 加上非结构化数据 [HF 博客文章](https://huggingface.co/posts/dvilasuero/324662497616161)！
- **Colab 与 HF 展开合作！**：**Google Colab** 现在与 **HF** 集成，允许用户直接在免费的 Colab Notebooks 上试用来自 HF 的 AI 模型 [Google Colab 博客文章](https://medium.com/google-colab/launch-hugging-face-models-in-colab-for-faster-ai-exploration-bee261978cf9)！
- **结构化与 Agent 执行！**：**CodeAgents + Structure**：一种更好的动作执行方式 [HF 博客文章](https://huggingface.co/blog/structured-codeagent)。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1384613050166018250)** (81 messages🔥🔥): 

> `OS Agent, Codex Competition, Fine-tuning Mistral 7B, MLOps for Internships, Hugging Face Chat UI` 


- **Coding Agent 已修复并就绪**：一位成员表示他们修复了自己的 **OS Agent**，现在它是一个 Coding Agent <:hugging_fire:1103375912814248006> 且优于 **Codex**！！！<:hugging_fire:1103375912814248006>。
   - 他们仍需针对这种新的 Agent 架构进行微调并调整细节以进行完善，但 **Master Agent** 已经可以召唤 **Mini Agents** 进行任务分配和讨论。
- **实习建议：MLOps 是基础**：一位成员寻求实习建议，提到尽管有一个支持 PDF 文档输入并使用 **RAG**、**Langchain** 和 **FlanT5** 的聊天机器人，但在 20 份申请中只获得了一次面试机会。
   - 另一位成员回应称，*目前 MLOps 已经是基础了*，并建议查看 [GitLab](https://about.gitlab.com/) 了解他们的做法。
- **Evaristo AI 展示葡萄牙语版 Hugging Chat**：一位成员分享了[他们在 Hugging Face 上的个人资料](https://huggingface.co/AbdullahSx96)以及一个[葡萄牙语版本的 Hugging Chat](https://Evaristo.ai)。
   - 有人指出，查看代码可以发现它使用了 **Hugging Chat UI** 且未进行商业化，不过 **Hugging Face 对 Prompt 数量有限制**，如果你了解这些限制的话。
- **PCIE 6.0 SSD 推迟至 2030 年**：一位成员分享了来自 [Tom's Hardware](https://www.tomshardware.com/pc-components/ssds/pcie-6-0-ssds-for-pcs-wont-arrive-until-2030-costs-and-complexity-mean-pcie-5-0-ssds-are-here-to-stay-for-some-time) 的文章，称由于成本和复杂性，用于 PC 的 **PCIE 6.0 SSD** 要到 **2030 年** 才会问世。
   - 另一位成员对此回复道：*真令人失望。在 AI 时代，摩尔定律似乎在显著放缓，而不是在加速……*。
- **HF 微调黑客松：请点赞支持！**：一位成员请求在 [此链接](https://huggingface.co/spaces/huggingface-course/README/discussions/3) 为 **Hugging Face 微调黑客松** 的讨论点赞。
   - 另一位成员回复说，你实际上可以使用 [这个 GitHub 仓库](https://github.com/huggingface/competitions) 自己运行一个 **Hugging Face 竞赛**。


  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1384630380434554960)** (6 messages): 

> `HF AI Agents Course, Zig-Zag Ring Attention` 


- **HF AI Agents 课程重新开始**：一位成员在进度达到 **80%** 暂停后，重新开始了 **HF AI Agents 基础课程**的学习。
   - 他们正在开发一个**聊天机器人项目**，该项目使用 **Generative AI** 根据文件或文本回答问题。
- **Zig-Zag Ring Attention 引起关注**：一位成员分享了他们正在学习 **Zig-Zag Ring Attention** 的消息。
   - 未提供更多细节。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1384623208250216601)** (3 messages): 

> `Chromium extension for speaking to readmes, Synthetic Data Generation for LLM Safeguards, memX: Shared memory layer for multi-agent LLM systems` 


- **扩展程序让你能与 GitHub 文档对话**：一位成员创建了一个 **Chromium 扩展**，允许用户**通过语音与任何 readme、文件或 wiki 页面对话**，并直接在 GitHub 上获得即时回答，该扩展已在 [Chrome Web Store](https://chromewebstore.google.com/detail/popjomeajiimhaikbfedbmaahoekmpfo?utm_source=item-share-cbHey) 上架。
   - 该扩展旨在提高可访问性，并从文档中快速检索信息。
- **通过合成数据生成保护 LLM**：一位成员在一篇博文中详细介绍了他们在 **LLM Safeguards 合成数据生成**方面的首次尝试。
   - 过程记录在[这里](https://yudhiesh.github.io/2025/06/16/cerberus-safeguards-for-llms-synthetic-data-generation-part-1.html)，相应的数据集已创建并上传至 Hugging Face [此处](https://huggingface.co/datasets/yudhiesh/cerberus-guardrails-small)。
- **memX 作为 LLM 的共享大脑发布**：一位成员发布了 **memX**，这是一个用于多 Agent LLM 系统的共享内存层，使 Agent 能够像大脑一样对不断演变的上下文进行读写，而不是仅仅传递消息。
   - 核心特性包括**实时 pub/sub**、**JSON schema 验证**、**API-key ACLs** 以及 **Python SDK**，代码可在 [GitHub](https://github.com/MehulG/memX) 获取，演示见 [X/Twitter](https://x.com/0xmmmehulll/status/1935301927967055950)。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1384763836237746227)** (3 messages): 

> `VSR Datasets, Attention Maps Visualization` 


- **成员在 HF 上寻找公开的 VSR 数据集**：一位成员询问是否有人知道 Hugging Face Hub 上有哪些公开可用的优质**视频超分辨率 (VSR)** 数据集。
- **寻求可视化 VLLM Attention Maps 的指导**：一位成员询问如何可视化视觉大语言模型（如 **LLaVA**）的 **Attention Maps**。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1384886170588286996)** (1 messages): 

> `LLM Project Architecture, RAG Project Architecture, LLM API Design` 


- **寻求 LLM 和 RAG 项目架构见解**：一位成员询问关于带有可供使用 **API** 的 **LLM** 和 **RAG 项目**的**默认项目架构**的学习资源。
- **需要 LLM 和 RAG 架构资源**：该请求侧重于寻找现有的学习材料和最佳实践，以构建具有 API 可访问性的 LLM 和 RAG 项目。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1384618766758973440)** (5 messages): 

> `Gradio Agents & MCP Hackathon Winners, Modal Labs sponsors award, MCP Server Track, Custom Component Track, Agentic App Track` 


- **Gradio Agents & MCP 黑客松获胜者揭晓！**：在评审了 **630+** 份提交作品后，**Gradio Agents & MCP 黑客松**的获胜者名单公布，每个赛道均设有现金奖励。
   - 获胜作品包括 [Geo Calculator MCP](https://huggingface.co/spaces/Agents-MCP-Hackathon/geocalc-mcp)、[Gradio Workflow Builder](https://huggingface.co/spaces/Agents-MCP-Hackathon/gradio_workflowbuilder) 和 [LLM Game-Hub](https://huggingface.co/spaces/Agents-MCP-Hackathon/LLMGameHub)。
- **Modal Labs 颁发 5,000 美元大奖！**：**ShallowCodeResearch** 因利用无限的 Serverless Compute 获得了由 **Modal Labs** 赞助的整个黑客松中金额最高的单项奖。
   - 获奖作品可以在[这里](https://huggingface.co/spaces/Agents-MCP-Hackathon/ShallowCodeResearch)体验。
- **社区选择了 Consilium MCP！**：[Consilium MCP](https://huggingface.co/spaces/Agents-MCP-Hackathon/consilium_mcp) 获得了 **社区选择奖 (Community Choice Award)**，奖金为 **500 美元**。
   - 此外，[Modern Multiplayer Online Role-Playing Game with MCP](https://huggingface.co/spaces/Agents-MCP-Hackathon/MMORPG_AI_NPC_MCP_CLIENT_SERVER) 获得了 **MCP 最具创新应用奖 (Most Innovative Use of MCP Award)**。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1384608971557699646)** (6 messages): 

> `Ollama 解释，最终评估模板` 


- **Ollama 揭秘：运行本地 LLM**：Ollama 让你可以在电脑上运行像 **Llama** 和 **Mistral** 这样的 AI 模型，而无需依赖 **ChatGPT** 等在线服务，只需通过 *ollama pull* 在本地下载模型文件。
   - 当你运行 *ollama run* 时，它会在端口 **11434** 上启动一个本地服务器，允许 Agent 与本地模型交互；查看[此视频系列](https://www.youtube.com/watch?v=2Pm93agyxx4)了解更多。
- **澄清最终评估模板的步骤**：一位用户寻求关于 **Unit 4 最终评估模板**步骤的澄清，包括克隆 [Final Assessment Template](https://huggingface.co/spaces/agents-course/Final_Assignment_Template)、修改 *app.py* 和 *requirements.txt*、运行评估，以及通过迭代达到 **30%** 的分数。
   - 他们正在寻求确认，在达到该分数后是否会获得证书。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1384612473524715671)** (83 messages🔥🔥): 

> `Gemini 2.5, Claude 定价, Aider 与 Gemini 2.5-pro, Open3 设置` 


- ****Gemini 2.5** 基准测试，定价仍不一致**：成员们讨论了 [Aider 上 **Gemini 2.5** 的定价可能不准确](https://storage.googleapis.com/gweb-uniblog-publish-prod/original_images/gemini_2-5_benchmarks_margin_light2x_1.gif)，暗示实际成本可能比估计高出 **5x**，一些成员发现这与他们的经验相符。
   - 一位成员花费 *$3* 使用 **Gemini** 创建了一个拥有超过 200 次提交的 **5K LoC 项目**。
- ****Claude-4-Opus** 成本被认为过高**：**Claude-4-Opus** 的输入/输出价格为 **15/75**，根据组合不同，价格要贵 *7.5x*，这引发了关于 Token 使用量以及与 **Gemini** 相比的性价比的讨论。
   - 一位成员指出，Gemini 生成的 Reasoning Token 明显多于 Opus。
- **Aider 支持最新的 **Gemini 2.5-pro** 模型**：用户确认 Aider 现在支持最新的 **Gemini 2.5-pro** 模型，并澄清在 Aider 设置中指定新模型即可工作，尽管它可能会显示关于 *未知上下文窗口大小和成本* 的警告。
   - 成员们注意到它使用了 *合理的默认值* 而不是特定的 `context window size and costs`。
- **正确配置 **Aider** 以获得 **Gemini** 的最佳效果**：为了避免上下文大小和编辑模式的问题，用户建议将正确的设置添加到 `.aider.model.settings.yml` 和 `.aider.model.metadata.json`，或者使用命令 `aider --model gemini/gemini-2.5-pro-preview-06-05 --thinking-tokens 32k --edit-format diff-fenced`。
   - 具体来说，将以下内容添加到 `.aider.model.settings.yml` 将避免警告：```
- name: gemini/gemini-2.5-pro-preview-06-05
  accepts_settings: ["thinking_tokens"]
```


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1384614248558231662)** (9 messages🔥): 

> `Aider --restore-chat-history, OpenRouter 与 Aider, Gemini 复制粘贴模式, Aider /read-only` 


- **Aider 历史记录得到恢复**：一位用户询问是否有办法在崩溃后继续之前的会话，另一位成员建议使用 **--restore-chat-history** 标志。
   - 一位用户产生了幻觉：*抱歉我产生了幻觉：--restore-chat-history*
- **Gemini 进入复制粘贴模式**：有人想在复制粘贴模式下尝试 **Gemini**，询问使用哪种质量最好的编辑器模型。
   - 未给出建议。
- **Aider 的 /read-only 防止修改**：一位用户询问 **/read-only** 是否应该防止文件在 Aider 中被修改。
   - 一位成员确认它应该防止修改。
- **OpenRouter 出现 Budget 0 错误**：一位成员询问关于在 Aider 中使用 **OpenRouter** 时出现 *Budget 0 is invalid* 错误的问题。
   - 他们澄清 *此模型仅在思考模式下工作*，但未提供解决方案。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1384612594052235456)** (79 messages🔥🔥): 

> `Midjourney 视频模型 V1, Krea AI, OpenHands CLI, Essential AI, AI 与专有数据` 


- **Midjourney 发布视频模型 V1**：[Midjourney](https://xcancel.com/midjourney/status/1935377193733079452) 发布了其**视频模型的第 1 版**，允许用户以大约**每秒视频一张图像的成本**，为 Midjourney 生成的或外部图像制作动画。
   - 新的 **“图生视频”功能**提供“自动”和“手动”动画设置，具有“高运动”和“低运动”选项，不过视频生成在发布时仅限网页端。

- **KREA AI 发布 Krea 1 公测版**：**KREA AI** 发布了 **Krea 1** 的公测版，旨在提供卓越的审美控制和图像质量，生成细腻的纹理、戏剧性的角度和电影级的光效，摆脱典型的“AI 感”。
   - 新模型支持多种风格、风格参考和自定义训练，可在 [krea.ai/krea-1](https://xcancel.com/krea_ai/status/1934981993722466454?s=46) 免费使用。
- **为 Coding Agents 推出的 OpenHands CLI 发布**：**All Hands AI** 推出了 **OpenHands CLI**，这是为其编程 Agent 打造的全新命令行界面，提供顶尖的准确性，并通过无需 Docker 简化了安装流程。
   - 该 CLI 保持了与之前基于 Docker 版本相同的准确性，虽然去掉了浏览器组件，但提供了 slash commands 和命令确认模式。
- **Essential AI 发布 24T Token 海量数据集**：**Essential AI** 宣布推出 **Essential-Web v1.0**，这是一个包含 **24 万亿 Token** 的预训练数据集，带有详细的元数据，专为创建高性能数据集而设计。
   - 使用 **Essential-Web v1.0** 的特定领域子集进行的评估表明，在 Web 代码、STEM 和医疗应用等多个领域性能均有提升。
- **CoreWeave 与 Weights & Biases 推出全新 AI 推理服务**：**CoreWeave** 和 **Weights & Biases** 正在推出全新的 AI 推理服务和用于实时 LLM 评测的在线评估工具（监控器）。
   - 这些服务运行在 **CoreWeave GPUs** 上，包括针对 DeepSeek R1-0528 和 LLama-4 Scout 等模型的推理端点，并提供 OAI Compatible APIs，旨在 AI 基础设施领域提供更多竞争力和灵活性。

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1384675582884708402)** (5 messages): 

> `Andrej Karpathy 的 AI 演讲, Software 3.0, LLM 类比, LLM 心理学, 部分自主性` 

- ****Latent Space** 重构 **Karpathy** 的 AI 演讲**：鉴于 AI 讨论的时效性极强，**Latent Space** 发布了 **Andrej Karpathy AI 演讲**的重构版本，整理了幻灯片和笔记，涵盖了 **Software 3.0** 和 **LLM 类比**等主题。
   - 内容还涉及 **LLM Psychology**、**Partial Autonomy**、**Vibe Coding** 以及为 Agent 构建应用，完整幻灯片已向 **Latent Space** 订阅者开放，链接见[此处](https://www.donnamagi.com/articles/karpathy-yc-talk)。
- ****Karpathy** 的演讲涵盖关键领域**：演讲内容包括 **Software 3.0**、**LLM 类比**（公用事业、晶圆厂、操作系统）、**LLM Psychology** 和 **Partial Autonomy**（包括人机生成-验证循环）。
   - 它进一步探讨了 **Vibe Coding** 和为 Agent 构建应用，全面概述了当前的 AI 开发概念。

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1384818599302135808)** (4 messages): 

> `Mojo 中的 IR 重映射器, Mojo 裸机内核, Modular GitHub issue 4854, 没有 X 账号的 Mojo 衬衫` 

- **Mojo 重映射器和 OS 内核兴起**：一位成员计划在 **Mojo** 中构建 **IR remapper** 并编写 **OS bare metal kernel**。
   - 他们链接到了 [Modular GitHub issue 4854](https://github.com/modular/modular/issues/4854) 并感叹道 *“上帝助我坚持下去。Mojo 太酷了”*。
- **Mojo 周边与 X 账号要求**：一位成员询问是否有办法在不发布 **X** 动态的情况下获得 **Mojo shirt**。
   - 他们提到可以在 **LinkedIn** 上分享，并表示很喜欢那个**漫画**。

---

### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1384930608400171119)** (1 messages): 

> `AMD 合作伙伴关系, GPU 支持, 模型支持, 开源化代码, Modular 黑客周末` 

- **Modular 实现 GPU 无关的理想境界**：Modular Platform **25.4** 引入了在 **AMD** 和 **NVIDIA** GPU 上运行相同代码且无需修改代码的能力，标志着与 **AMD** 合作伙伴关系的开始。
   - 此版本支持 **AMD Instinct™ MI300X** 和 **MI325X GPUs**，实现了跨不同硬件的部署，无需担心厂商锁定或额外配置。
- **25.4 加速 LLM 工作负载**：**25.4** 版本在 **Llama 3.1**、**Gemma 3** 和 **Mistral** 等最先进的语言模型上，针对预填充密集型（prefill-heavy）的 BF16 工作负载，吞吐量提升高达 **53%**。
   - 该版本扩展了硬件支持，包括 **AMD MI300/325**、**NVIDIA Blackwell**、**RTX 40xx** 和 **RDNA3**，拓宽了平台的兼容性。
- **Modular 开源内核代码库**：Modular 已开源超过 **45 万行**生产级 **Mojo** 内核代码，增强了透明度和社区贡献。

- 该版本还包括改进的文档、**PyTorch** 算子教程和内核性能工具，促进了更轻松的采用和开发。
- **Modular 举办黑客周末并赠送 GPU**：Modular 将于 **6 月 27 日** 举办 **Hack Weekend**，包括 GPU 编程研讨会和丰厚的 GPU 奖池，邀请开发者通过此 [链接](https://lu.ma/modular-gpu-workshop) 线上或线下参与。
   - 您可以在 [Modular 博客](https://www.modular.com/blog/modular-25-4-one-container-amd-and-nvidia-gpus-no-lock-in?utm_source=discord&utm_campaign=25_4) 获取完整报道。

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1384688130149584939)** (72 messages🔥🔥): 

> `Blackwell accelerators, Mojo bare metal, Kernel Development with Mojo, Mojo Async, Mojo Parametric Traits` 

- **Blackwell 买家因业务故障而沮丧**：**Blackwell 加速器** 的早期采用者发现支持有限且驱动程序存在 Bug，尽管 [MAX 支持 Blackwell](https://www.modular.com/max)，但体验仍不可靠且痛苦。
   - 一位用户指出，**AMD** 和 **Intel** 正在以一半的价格推出具有相同或更多显存的 GPU，甚至更便宜的 **4090** 可能是更好的选择。
- **Mojo 进入裸机时代，摆脱运行时依赖**：成员们对 **Mojo** 现在可以在没有系统调用或运行时依赖的情况下运行裸机（bare metal）感到兴奋，如 [附图](https://cdn.discordapp.com/attachments/1151418092052815884/1384859713388417126/image.png?ex=68549f5d&is=68534ddd&hm=d1af4a129aa4c5bd2cd0ebd8a5db27931f7454475f043fdf816af81fad245892&) 所示。
   - 在进行简单的函数替换（如 `KGEN_EE_JIT_GlobalConstructor` 和 `KGEN_EE_JIT_GlobalDestructor`）后，它可以作为一种*具有零开销抽象的现代系统编程语言*用于内核开发。
- **内核难题：Mojo 内存管理的奇迹**：讨论了使用 Mojo 进行内核开发，强调了其内存管理能力，其中*编译器*在*高效且安全的内存管理*方面值得信赖。
   - 一位开发者指出，希望有*清晰的错误传播/处理策略*来使该语言臻于完美，并建议通过 FFI 进行手动异步已经可行。
- **Mojo 路线图之谜：Async 和 Traits 占据最高优先级**：一位成员分享了 Mojo 开发的首要任务，包括 **Parametric Traits**、**async** 的重大重构，以及用于自动 vtable 处理的 **dyn Traits**。
   - 还有人提到，错误处理未来可能会转向受检异常（checked exceptions）。
- **MAXimum Blackwell Boost：Modular 的秘密支持**：尽管没有被广泛宣传，但 **MAX** 支持 **Blackwell** 系统（如 **5090**），在正式发布前正在进行性能优化工作。
   - 鼓励早期测试人员进行试用并报告问题。

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1384731160801841193)** (3 messages): 

> `Max Inference, Max and Python, Orchestration of Model Graphs` 

- **Max 推理只能通过 Python 扩展控制吗？**：一位用户询问 **Max** 是否只能通过 **Python 扩展**控制，以启动推理会话并为其提供 numpy 数据。
   - 另一位用户指出，Mojo API 目前已被移除。
- **Python 是图形编译器的主要接口**：澄清了目前 **graph compiler**（构建 MAX 模型的核心）的主要接口是通过 **Python**。
   - 分享了一篇关于为什么使用 Python 进行模型图编排的文章：[Mojo MAX Bindings](https://forum.modular.com/t/mojo-max-bindings/1499/3)，并指出 *Python 使得集成到现有的 tokenizers、处理逻辑等变得非常容易。*

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1384610890988392498)** (74 messages🔥🔥): 

> `Web page generation credit costs, Website with scraping Facebook, Gumtree and Ebay, Manus Rival MiniMax AI with Agent Mode, Manus Video Generation with VEO, Manus Credit Refunds for Errors` 

- **高流量阻碍网页生成，耗尽积分**：用户报告称，由于错误，生成一个简单的网页消耗了大量积分，一位用户不得不求助于手动编辑文件。
   - 一位用户指出这种情况发生在*中午之后*，并建议一天中不同时间段的流量可能有所不同。
- **抓取 Facebook、Gumtree、Ebay 的网站计划失败**：一位用户花费了 **5k 积分** 尝试创建一个抓取 **Facebook、Gumtree** 和 **eBay** 上被盗自行车列表的网站，但 AI 失败了，交付了虚假结果。
   - 该用户收到了 **2.5k 积分退款**，但表示这纯粹是*浪费时间和积分*。

- **MiniMax AI 发布 Agent 模式，挑战 Manus**：用户讨论了 **MiniMax AI** 的新 Agent 模式，将其视为 Manus 的竞争对手。
   - 一些人对 **MiniMax** 的积分系统和订阅模式表示担忧，同时赞扬了 Manus；另一些人则认为*竞争只会为用户在选择 AI 时提供更多选项*。
- **要求针对 AI 任务错误退还积分**：用户讨论了 **Manus** 在遇到错误并需要重新运行流程时是否应该退还积分，还是应该仅在任务成功完成时收费。
   - 一位用户将其比作购买*“肉质腐烂、面包发霉的汉堡”*，而另一位用户则表示 *AI 经过编程/训练，能完成比一个人一生所能吩咐的还要多得多的工作*。
- **Manus 通过免费 YouTube 助力获得更强能力**：一位用户发布了一个 [YouTube 链接](https://m.youtube.com/watch?v=5PuofaVqXNI)，称 **Manus** 可以*变得更加强大，而且是免费的*。

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1384633968153858049)** (3 messages): 

> `Model Context Protocol, AG-UI integration with CopilotKit, MCP vs Vector Search` 

- **Block 设计 MCP 服务器以辅助 Claude**：Block 的工程团队分享了他们创建 **Model Context Protocol (MCP) 服务器**的系统方法，这些服务器可以与 Claude 及其他 AI 系统无缝集成，一切从清晰的 [设计](https://t.co/0vJajYzrfJ) 开始。
   - 他们的设计模式有助于构建更好的 **AI assistants**。
- **AG-UI 与 CopilotKit 的集成发布**：LlamaIndex 发布了对 **AG-UI** 的官方支持，使得将你的 Agent 从后端直接引入面向用户的应用程序变得极其简单，从而能够[以零样板代码创建 Agent 驱动的前端](https://t.co/hzxBrXKyTv)。
   - AG-UI 是与 **CopilotKit** 集成的。
- **MCP 并没有消除对 Vector Search 的需求**：**MCP** 协议为 Agent 直接连接数据源创造了新的可能性，但对于 PDF 和 PPT 等非结构化数据，仍然需要预处理和索引。
   - 据估计，*90% 的企业数据存在于 PDF、PPT* 和其他类似格式中。

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1384889675646238812)** (70 messages🔥🔥): 

> `FastAPI streaming issues, Agent workflow with streaming, Metadata filtering in chat/query engines, Response synthesizer refinement, Anthropic's AI Agent Frameworks` 

- **FastAPI 流式传输受停顿困扰**：一位用户在使用 FastAPI 向前端流式传输事件时遇到了 **20 多秒的延迟**，尽管后端处理已经完成。后来通过添加 `if ev.delta: yield` 解决了该问题。
   - 建议在 yield 时添加换行符 `yield json.dumps({..}) + "\n\n"` 来分隔块，但最终，确保仅 yield 非空 delta（即 `if ev.delta: yield`）解决了问题。
- **FastAPI 中的 Agent Tool Call 超时**：一位正在调试流式传输事件的 Agent 工作流的用户遇到了 Tool Call 延迟，并被建议使用以下代码：
   - ```python
async for ev in handler.stream_events():
  if isinstance(ev, AgentStream):
    if ev.delta:
      print(ev.delta, end="", flush=True)
    elif ev.tool_calls:  # optionally print tool calls as they stream
      print(ev.tool_calls)
  elif isintance(ev, ToolCallResult):  # optionally print the complete tool call output
    print(ev.tool_name, ev.tool_kwargs, ev.tool_output)
  elif isintance(ev, ToolCallResult):  # optionally print the complete tool call input
    print(ev.tool_name, ev.tool_kwargs)
```
- **Chat/Query Engine 上的 Metadata 过滤功能**：一位用户寻求在 Chat/Query Engine 上使用 Metadata 过滤，建议将 retriever 传递给 Chat Engine，这适用于像 **Condense_plus_context** 这样的引擎。
   - 他们对这个解决方案表示了巨大的感谢。
- **Response Synthesizer 的优化流程**：一位用户想知道 RAG LLM 的通用标准，考虑通过多次 LLM 输出并使用 `response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")` 进行优化。
   - 建议如果目标仅仅是优化，那么定义 System Prompt 以确保从一开始就获得所需的响应格式是更好的路径。
- **LlamaIndex 在 Anthropic 的框架列表中被冷落**：一位用户注意到，在 Anthropic 关于[“构建高效 AI Agent”](https://www.anthropic.com/engineering/building-effective-agents)的指南中，LlamaIndex 未出现在其框架列表中。
   - 另一位成员表示，尽管由于现有的关系而被排除在列表之外，但 *LLamaIndex 绝对是一个瑰宝*。

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1384720308581306418)** (11 messages🔥): 

> `Stable Diffusion Performance, Custom Kernels, GPU Hardware Counter Monitoring, Expert Tiles Streaming` 


- **Stable Diffusion 性能专家现身！**：一位新成员在 **4090** 上使用 **1 step sdxs 模型**，在 **512x512** 分辨率下达到了 **294 images/sec**，并声称在 2023 年 10 月就已完成实时视频方面的开创性工作（[X.com 上的帖子](https://x.com/Dan50412374/status/1772832044848169229)）。
   - 他们现在使用 **sdxl** 在 **1280x1024** 分辨率下达到了 **23 fps**，并表示有兴趣探索自定义 Kernel，以同时利用 CUDA 和 Tensor Cores。
- **开启 Kernel 定制化探索**：一位拥有软件架构和 SQL 数据库性能背景、现已从 MSFT 退休的成员，正在尝试通过自定义 Kernel 来并发调用 **CUDA** 和 **Tensor Cores**。
   - 在从 Stable Diffusion 早期阶段就对其着迷后，他们正在组建一套包含双 **5090**、**7985WX Threadripper** 和 **256 GB** 内存的新系统，用于硬核实验。
- **调研 GPU 硬件计数器监控**：一位成员正在探索 GPU 硬件计数器监控，借鉴了之前在 CPU（**Intel** 和 **Sparc**）上进行硬件计数器分析的经验。
   - 这是他们努力寻找最佳项目的一部分，目前使用 **Ubuntu**、nightly **torch** 版本和 **CUDA 12.9**。
- **Expert Tiles 流式传输策略引发讨论**：一位成员询问关于通过 **TMA** 将 Expert Tiles 直接流式传输到 **SMEM** 以避免 L2 污染的问题，并引用了一个*有趣的概念*。
   - 另一位成员澄清说，目标是使用利用 TMA 的 UVA，灵感来自 [vLLM 的方法](https://github.com/vllm-project/vllm/pull/15354) 和 [DeepMind 关于更小 Expert 的研究结果](https://arxiv.org/pdf/2407.04153)。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1384830456339496970)** (3 messages): 

> `Triton, Shared Memory, Register Spilling` 


- **用户寻求 Triton 共享内存控制**：一位用户询问如何强制 **Triton** 显式地将 Tensor 加载到 **Shared Memory** 中，以避免寄存器溢出（Register Spilling）。
   - 一位成员回答说，据其了解，目前无法避开 **Triton 对 Shared Memory 的自动管理**。
- **寄存器溢出问题被提出**：一位用户提到观察到了 **Register Spilling**，并希望通过使用 Shared Memory 来提升 Triton 的性能。
   - 该用户询问了为强制使用 Shared Memory 可能需要的代码库更改。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1384819859895881779)** (16 messages🔥): 

> `CUDA segfaults, CUDA gdb, NVIDIA bug reports, CUDA containers vs. CCCL/Thrust, CUDA Runtime error checking` 


- **GPU Malloc 导致初始 CUDA 段错误**：一位成员最初在 GPU malloc 期间遇到段错误（segfaults），但随后报告已修复；具体根本原因未说明。
- **建议对 CUDA 使用 CUDA gdb 调试**：针对调试挑战，另一位成员建议使用 **CUDA gdb** 进行调试，但该建议被推迟到以后使用。
- **提交 NVIDIA 错误报告**：一位成员针对调试对话分享了 [NVIDIA 错误报告链接](https://developer.nvidia.com/bugs/new)，并提到他们正在使用它。
- **CCCL/Thrust 与自定义 CUDA 容器**：一位成员建议使用 **CCCL/Thrust** 和 **RAPIDS/RMM**，而不是实现自定义的 **CUDA 容器**，特别是对于非教学项目。
   - 原成员表示，由于需要操作特定的数据结构，他们将坚持使用 "cuda cpp 方式"。
- **CUDA Runtime 错误与 Compute-Sanitizer**：建议实现[正确的 CUDA Runtime 错误检查](https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api)，并探索使用 [compute-sanitizer](https://developer.nvidia.com/compute-sanitizer) 进行调试，建议将调试器作为最后的手段。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1384911441512894534)** (2 messages): 

> `Intervene in inductor kernels, Customize generated kernel code` 


- **手动干预 Inductor Kernels？**：一名成员正在寻找一种方法来*干预由 Inductor 创建的特定 Kernel*，以便对生成的 Kernel 代码本身进行手动修改。
   - 他们考虑过创建一个 Custom Op，但这需要理解它是如何在前向和反向传播中融合操作的，这会非常耗时。
- **直接自定义生成的 Kernel 代码**：用户希望直接修改 Kernel 代码，而不仅仅是尝试不同的 Inductor 配置。
   - 他们还询问是否有办法让某些 Inductor 配置仅应用于特定的生成 Kernel，从而实现对 Kernel 生成过程的精细控制。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1384702214874071201)** (4 messages): 

> `Embed Issues with arXiv Links, Arctic Long Sequence Training, GitHub Repo Discovery` 


- **arXiv 链接导致嵌入（Embeds）显示异常**：用户报告称，**arXiv** 链接的嵌入显示出现了奇怪的行为，无法显示页面标题，例如[这个例子](https://arxiv.org/abs/2506.13996)。
   - 由于嵌入失败，一位用户手动提供了标题：*"Arctic Long Sequence Training: Scalable And Efficient Training For Multi-Million Token Sequences"*。
- **潜伏在阴影中的 GitHub 仓库**：一位用户分享了一个看起来很有趣的 [GitHub 仓库](https://github.com/nirw4nna/dsc)链接。
   - 未提供进一步的上下文或描述，使得该仓库的用途仍然是一个谜。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1384938624759365662)** (2 messages): 

> `Shared Memory Buffers, Tensor Data Types` 


- **SMEM 缓冲区灵活性提升**：一位成员澄清说，**共享内存（SMEM）缓冲区**的类型不必与 **Tensor Map 数据类型**匹配。
- **用户表示感谢**：一位用户表达了谢意。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1384834081904132097)** (3 messages): 

> `AMD bf16 instruction, MFMA supports bfp16, Triton gemvs` 


- **AMD 缺少 bf16 FMA 指令**：有人指出 **AMD** 没有用于 **FMA** 的 bf16 指令，并且只有 **MFMA** 支持 **bfp16**。
   - 另一位成员确认，在 **RDNA3/4** 和 **CDNA4** 上似乎只有 *dot bf16* 指令可用，如[截图](https://cdn.discordapp.com/attachments/1233704710389764236/1384834081925107782/Screenshot_from_2025-06-18_11-53-51.png?ex=6854877e&is=685335fe&hm=6271f9d6f4653913db6c81666596d6402033c41e8629b999a4248332e49c9362&)所示。
- **bf16 gemvs 性能堪忧**：一位用户评论说，在 **Triton** 中使用 **fp32 模拟**运行的 **bf16 gemvs** 速度极其缓慢。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 messages): 

t_cc: 谢谢！我会看看的。
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1384973778244604036)** (1 messages): 

> `Distributed Training Course, Accelerate Maintainer` 


- **分布式训练课程提醒**：一位成员正在推广由来自 Transformers 的 **Accelerate** 维护者教授的[分布式训练课程](https://maven.com/walk-with-code/scratch-to-scale?promoCode=matej26)。
   - 该课程承诺将汇聚*顶尖头脑*。
- **另一个分布式训练课程提醒**：另一位成员也在推广由来自 Transformers 的 **Accelerate** 维护者教授的[分布式训练课程](https://maven.com/walk-with-code/scratch-to-scale?promoCode=matej26)。
   - 该课程同样承诺将汇聚*顶尖头脑*。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1384961613513232564)** (4 messages): 

> `PMPP Leaderboard, GPU MODE Competitions` 


- **PMPP 排行榜即将更新**：[PMPP 题目排行榜](https://www.gpumode.com/leaderboard/339)计划使用*更强大的评估方法论*进行更新。
   - 由于自成立以来积累的经验和改进，当前的 PMPP 题目将在更新后变为半永久性。
- **GPU MODE 宣布另外两场竞赛**：GPU MODE 正在积极筹划**另外两场竞赛**以扩大其活动规模。
   - 目前尚未提供关于这些竞赛主题或确切发布时间的进一步细节，但请期待它们很快到来。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1384783080216330251)** (19 条消息🔥): 

> `FLE 进展担忧、Pull Request 延迟、写入权限民主化、Factorio 源代码集成、Factorio 客户端移除` 


- **FLE 团队承认进展担忧**：团队成员对 **Factorio Learning Environment (FLE)** 项目进展缓慢表示担忧，特别是待处理的 Pull Request。
   - 成员表示需要更定期地合并 Pull Request，建议以 **每周 2-3 次合并** 为目标以保持势头。
- **写入权限民主化获得支持**：一名核心团队成员建议创建一个 **FLE GitHub 组织** 以实现写入权限民主化，让更多团队成员能够有效地进行贡献。
   - 这一变化旨在加快 Pull Request 的合并速度，并为实验性项目建立独立的仓库。
- **移除 Factorio 客户端依赖：基础设施的胜利**：成员指出，正如 [pull request #223](https://github.com/JackHopkins/factorio-learning-environment/pull/223) 中提议的那样，移除对已连接 **Factorio 客户端** 的要求代表了基础设施方面的重大胜利。
   - 这一变化解锁了 **CI/CD**、自包含的 **Jupyter notebooks** 以及大规模水平扩展能力。
- **考虑获取源代码以进行直接集成**：团队讨论了获取 **Factorio 源代码** 访问权限的可能性，以便实现缺失的功能，并将 Mod 和 API 直接构建到游戏引擎中。
   - 这将带来更紧密的集成，可能效仿 **Malmo** 等项目的稳定性（该项目多年来不需要提交代码）。
- **事件重新设计可简化交互**：团队成员询问了修改 **Factorio Lua API** 中某些 **on_player 类型事件** 以改进功能的可行性。
   - 具体而言，他们正在探索将 **on_player_mined** 等事件附加到 **control** 而非玩家身上的可能性，以便在采矿时实现更精确的资源分配。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1384792766445518880)** (4 条消息): 

> `CUTLASS 示例、申请资助` 


- **资助申请尝试**：一位成员分享了他们使用论文草案作为提案 *申请了资助*。
   - 目前尚不清楚申请的是哪项资助，但也许我们将来会了解更多。
- **分享 Cutlass 代码示例片段**：一位成员询问 *除了文档之外是否有适合初学者的 CUTLASS 教程*，另一位成员分享了 [NVIDIA Cutlass 示例链接](https://github.com/NVIDIA/cutlass/tree/main/examples)。
   - 链接的仓库包含有用的代码片段，可帮助快速开始使用 CUTLASS。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/)** (1 条消息): 

dumbpandabear: 这听起来非常令人兴奋！期待能帮忙让这个项目启动。
  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1384755454046900234)** (17 条消息🔥): 

> `坐标平面绘图模型、AI 研发频道、Embed v4 的欧盟 GDPR 合规性、Cohere 4 AI 项目` 


- **笛卡尔画布创作者寻求坐标代码**：一位成员正在寻找一种模型，将指令输入到他们的 **笛卡尔平面画布** 中以绘制酷炫的艺术作品。
   - 欢迎提供建议！
- **研究领域：新频道出现**：创建了一个新频道 <#1384974112841269399> 用于讨论 **AI 研发**。
   - 欢迎成员聊天并与志同道合的人建立联系，但 **禁止发布广告**。
- **GDPR 保证：审查 Embed v4 的欧盟合规性**：一位成员询问了 Embed v4 的 **欧盟 GDPR 合规性**。
   - 这可能在 Roadmap 上，因为 Embed v4 非常适合 **多模态 RAG 文档**。
- **Cohere 协作：创造性地贡献**：一位新社区成员询问如何 **加入并为现有的 Cohere 项目做出贡献**。
   - 成员被引导至 [此处](https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw) 申请 **Cohere 4 AI**，并在 <#1384974112841269399> 中分享研究成果。


  

---

### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1384758186644340817)** (28 messages🔥): 

> `bufio.Scanner 错误，command-r-08-2024 问题，Go SDK 问题` 


- **Bufio Scanner Bug 浮现**：一名成员报告在使用 `cohere/command-r-08-2024` 模型时遇到了 `bufio.Scanner: SplitFunc returns advance count beyond input` 错误。
   - Cohere 支持团队请求提供截图和详细信息，并建议该问题可能与频繁的客户端取消有关，但该成员认为问题源于 Cohere 的 **Go SDK**。
- **Command-R 模型面临补全生成困扰**：用户在使用 `command-r-08-2024` 模型时遇到了句子生成不完整的问题，即使在排除了客户端潜在原因后依然存在。
   - 一个不完整补全的示例为：*"Firefly's smile deepens, a hint of mischief in her red eyes."Well, hello there," she says, her voice carrying a"*。
- **Go SDK 被怀疑是补全故障的原因**：Cohere 支持团队最初建议该错误源自客户端，但经过进一步调查，错误似乎源于 Cohere 的 **Go SDK**。
   - Cohere 支持团队建议将 SDK 升级到最新版本，以观察是否能解决该问题。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1385001218979663913)** (2 messages): 

> `自我介绍，志愿者机会` 


- **Cynthia 加入社区并寻求机会**：来自布隆迪、现居加拿大安大略省的 Cynthia 介绍了自己，并表示有兴趣在 [Cohere 社区内寻找志愿者机会](https://cohere.com)。
   - 她正寻求如何开始参与项目，并对加入社区感到兴奋！
- **欢迎新社区成员**：Cohere 社区欢迎新成员并鼓励他们进行自我介绍。
   - 新成员被要求分享他们的背景、当前项目、喜爱的技术/工具以及在社区中的目标。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1384640018156486717)** (29 messages🔥): 

> `Muon, AdamW, Kron, Qwen, torchtune` 


- **Muon 与 AdamW 的性能对比**：一位 PR 作者分享了关于 [**Muon**](https://github.com/pytorch/torchtune/pull/2803#issuecomment-2981262780) 的一些结果，质疑为什么 **AdamW** 的表现可能更好，并将其归因于集成中的潜在错误或 **torchtune** 的某些特性。
   - 有人指出，*当 SFT 优化器与预训练优化器不同时，使用 Muon 的 SFT 相比 AdamW 并没有显示出显著优势*，这表明仍有相当大的探索空间。
- **调优良好的 Kron 性能与 AdamW 相似**：一名成员分享了使用不同实现（[ethansmith2000/fsdp_optimizers](https://github.com/ethansmith2000/fsdp_optimizers)）对 **Kron** 和 **Muon** 进行的对比，在其他测试中，经过良好调优的 **Kron** 表现与 **AdamW** 大致相当。
   - 他们注意到 **AdamW** 通常速度最快，但在内存占用上略逊于 **Muon** 或全对角线/单对角线 **Kron**。
- **质疑 Qwen 的收敛正确性**：有人对 **Qwen3 0.6b** 的正常收敛表示怀疑，认为 PR 的设置可能存在问题，并分享了一个 [WB_Chart](https://cdn.discordapp.com/attachments/1236040539409879170/1384693197653020813/WB_Chart_6_18_2025_3_35_34_AM.png?ex=68540448&is=6852b2c8&hm=1d93bef584b20807f407a2d016efb08ae630f52d814310f89605d7f1648bcf27&)。
   - 在进行图像分析后，他们表示 *这个图表上的波峰更趋于正常（差异约 ~500），肯定在某处存在 bug*。


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1384861764100952167)** (4 messages): 

> `Muon 优化器, Jaguar Muon, 现代优化器` 


- **神奇的 Muon 优化器现身**：一名成员发现 **Muon 优化器** 非常有趣，指出通过将更新正交化来产生更快的收敛似乎像魔法一样。
   - 提到 **Jaguar Muon** 时，他们开玩笑说：*"那种巫术应该被取缔"*。
- **附录中的 Poison Proofs？**：一名成员询问另一名成员是否查看了 *"主要的毒药：附录中的证明（proofs in Appendix）？"*
   - 另一名成员回答说，他们打算把这些留到以后有空时再看。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1384637869473398855)** (2 条消息): 

> `K-12 educators, podcast from sources, NotebookLM API` 


- **在 K-12 教育中利用 NotebookLM**：一位成员正在向 **K-12 educators** 进行演示，并询问在教育领域利用 **NotebookLM** 的常见用例。
   - 他们提到希望通过 **API** 访问 **NotebookLM** 的功能，以便从**源文件生成播客 (podcasts from the sources)**。
- **播客生成功能请求**：该成员询问是否有 **API**（或 MCP！）可以实现从 **NotebookLM** 内部的源文件生成播客。
   - 他们还询问了是否存在实现此功能的现有路线图或路径。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1384617987268677653)** (25 条消息🔥): 

> `Gemini App Daily Limit, NotebookLM Model, LaTex support, Audio Overviews, Podcast issues` 


- **Gemini 研究功能每日限制引发讨论**：用户对 [Gemini](https://gemini.google.com) 应用中深度研究（deep research）功能的每日限制提出质疑，对免费版与付费计划的限制感到困惑。
   - 一位用户报告在免费版中达到了限制，但找不到付费 Google 计划的相关数据。
- **NotebookLM 可能使用 Gemini 2.5 Flash**：一位用户认为 **NotebookLM** 使用与 **Gemini** 相同的基础模型，但经过微调以减少幻觉（hallucinate），传闻该模型为 [Gemini 2.5 Flash](https://ai.google.dev/)。
   - 这一推测基于 [Google AI Studio](https://www.youtube.com/watch?v=EOmgC3-hznM&ab_channel=JeffSu) 发布的一个带有 Gemini 2.5 Flash 的实验性音频生成版本；其他人则声称它仍基于 2.0。
- **LaTeX 支持仍待实现**：用户请求 **NotebookLM** 支持 **LaTeX**，但目前尚不支持数学标记（math markup）。
   - 一位用户在输出显示代码而非代码表示时遇到问题，并被引导至 [功能请求频道](https://feedback.google.com)。
- **音频概览长度限制**：用户报告在意大利语等非英语语言中，尽管使用了自定义提示词（custom prompts），仍无法生成超过 **10 分钟** 的**音频概览 (audio overviews)**。
   - 似乎 NotebookLM 默认使用英语，并忽略了生成更长音频段的自定义提示词。
- **播客定制化的烦恼**：一位播客制作人抱怨 **NotebookLM** 不再遵循定制化指令，听起来像 *一位正在读论文的大学教授*，并忽略了诸如避免使用 *deep dive* 等短语的指令。
   - 其他用户证实遇到了类似问题，称该工具在播客制作方面已经 *陷入僵局 (dead in the water)*。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1384681655377203211)** (5 条消息): 

> `Typo Analysis, Cognitive Offloading to AI, Flow Matching in Production` 


- **拼写错误还是大脑错误？**：一位成员分析了一个拼写错误，区分了物理打字错误（例如将 *the* 误打为 *teh*）和脑部错误（将 *was* 误打为 *one*）。
   - 该成员假设后者可能表明大脑的打字部分正在 *倾听* 消息构思部分。
- **普通人将思考外包给 ChatGPT**：一位成员观察到，许多人现在在处理认知要求低的任务时，会将思考外包给 **ChatGPT**。
   - 该成员表示，发生这种情况是因为 **ChatGPT** 对他们来说通常已经 *足够好* 了。
- **Flow Matching 受到关注**：一位成员引用了[一条推文](https://fxtwitter.com/mathusmassias/status/1935246909473521829)，询问 **Flow Matching (FM)** 是否已在工业界的生产环境中使用。
   - 另一位成员回答说 **Imagen** 和 **Flux** 使用了 **FM**。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1384643457372389376)** (11 条消息🔥): 

> `Predictive Coding, V-JEPA-2 Paper Discussion, PRECO GitHub Repo` 


- **预测编码 = 平方根猜测？**：一位成员链接了关于 [预测编码 (predictive coding)](https://arxiv.org/abs/2407.04117) 的讨论，认为理解**通过猜测和校验计算平方根**以及**反向传播 (backpropagation)** 就能大致理解预测编码。
   - 他们还链接了 [PRECO GitHub 仓库](https://github.com/bjornvz/PRECO) 作为进一步学习的材料。
- **V-JEPA-2 论文讨论日程安排**：成员们安排了今天对 **V-JEPA-2 论文** 的讨论，参考了 [Meta AI 博客文章](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) 和 [相关的 arXiv 论文](https://arxiv.org/abs/2506.09985)。
   - 另一位成员创建了一个未来活动，准备带领小组研读 [V-JEPA-2](https://discord.gg/mspuTQPS?event=1384953914029506792)。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1384668783578447973)** (8 条消息🔥): 

> `Keen Tech RL, Carmack Goals, Open Sourcing, Cursor Tier` 


- **Keen RL 演示令人失望**：成员们认为 [Richard Sutton 的 Keen Tech RL 演示](https://fxtwitter.com/RichardSSutton/status/1934780327169523765?t=_pmMm7dwecEO0KTLS8LbMA&s=19) *令人失望*，尤其是考虑到 **Keen 对 RL 的关注** 以及 **Carmack 不兼容的目标**。
   - 一位成员建议随后的演示会更有趣。
- **Keen 将开源代码**：成员们对 **Keen Tech** 开源其代码表示兴奋。
   - 有人评价其 *非常平庸*。
- **Cursor 发布新订阅层级**：成员们分享了 [Cursor 新订阅层级公告](https://www.cursor.com/blog/new-tier) 的链接。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1384609371769540659)** (17 条消息🔥): 

> `Custom Transport in FastMCP, Corporate MCP Client/Host Alternatives, GraphQL API Tool Generation, Multi-Agent Systems with MCP, MCP Server Credentials` 


- **FastMCP 自定义传输实现**：一位成员询问了如何在 **FastMCP** 中实现 **custom transport**。
   - 另一位成员确认在客户端可以通过扩展基础传输类来实现，但不确定服务端的情况。
- **应对公司限制：MCP Client/Host 解决方案**：成员们讨论了当公司政策限制使用 **Claude Desktop** 或 **Cursor** 时的 **MCP client/host** 备选方案。
   - 一位成员成功在本地使用 **Ollama** 运行 **devstral:24B** 并配合 **CLINE** 使用，但在使用 **Roo** 时遇到了问题。
- **MCP 处理大规模 GraphQL API**：一位成员正在使用其 **MCP server** 为 **GraphQL API** 查询和变更（mutations）生成约 **600 个工具**，并指出了 Cursor 在处理如此大量工具时的局限性。
   - 他们注意到，当工具数量超过几十个时，**Cursor** 和其他模型就会表现吃力，如链接中的 [截图](https://cdn.discordapp.com/attachments/1312302100125843479/1384765895792001044/Screenshot_2025-06-17_at_22.24.52.png) 所示。
- **使用 MCP 构建 Multi-Agent 系统架构**：社区讨论了使用 **MCP tools** 构建 **multi-agent systems** 是否需要 **A2A**，或者在每个 Agent 中部署 **MCP client** 和 **server** 是否足够。
   - 一位成员表示 *不需要 A2A*，并且 *即使是 Google 也不在意它*。
- **简化 MCP Server 访问**：成员们讨论了让其他人使用新创建的 **MCP server** 的最简便方法，重点在于凭据（credential）的获取。
   - 具体目标是让用户能够简单地从 **Google Cloud Console** 获取一个 **credentials.json** 文件。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1384610141781098588)** (2 条消息): 

> `Text-to-GraphQL, GraphQL Schemas, Arize-ai` 


- **Text-to-GraphQL MCP Server 发布**：Arize AI 推出了一项新功能，允许用户直接从 Spaces 页面将 **MCP servers** 添加到其账户，并介绍了 **Text-to-GraphQL MCP server**。
   - 它通过一个与 **Claude Desktop 和 Cursor** 等 AI 助手集成的 MCP server，将自然语言查询转换为 **GraphQL queries**，并提供了 [GitHub 仓库](https://github.com/Arize-ai/text-to-graphql-mcp) 和 [完整介绍](https://arize.com/blog/text-to-graphql-mcp-server/)。
- **GraphQL schemas 的 Token 问题**：**GraphQL schemas** 很容易超过 **75,000 tokens**，这使得将整个 schema 塞进 LLM 的上下文窗口变得不切实际。
   - **Text-to-GraphQL MCP** 通过教导 Agent 直接遍历 schema 图，仅提取其所需的字段和类型来解决这一问题。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1384702551596732579)** (4 条消息): 

> `Hackathon Timing, tinygrad maturity` 


- **黑客松推迟至 Tinygrad 成熟**：由于需要 **tinygrad** 变得更加可用和成熟，tinygrad 黑客松已推迟，最早可能在明年举行。
   - 公告鼓励大家继续使用并反馈 **tinygrad**，并指出这些参与度将作为黑客松参与者选拔的参考。
- **鼓励反馈以供未来黑客松参考**：鼓励参与者积极使用 **tinygrad** 并提供反馈，这将影响未来黑客松参与者的选拔。
   - 这一举措旨在提高 **tinygrad** 的可用性和成熟度，确保在最终举行时能有更高效的黑客松体验。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1384802454574469141)** (5 messages): 

> `TinyJit, ShapeTracker, Variable Usage` 


- **TinyJit 参数不匹配（args mismatch）解决方案**：一位用户在对 `Tensor` 进行循环迭代并使用 `@TinyJit` 时遇到了与 `args mismatch in JIT` 相关的 `AssertionError`。一名成员提供了使用 `Variable` 来解决 `ShapeTracker` 问题的方案。
   - 建议的解决方案包括创建一个 `Variable` 来表示循环索引并在循环内进行绑定，从而允许 `TinyJit` 正确处理变化的形状，更多细节参考 [tinygrad 的 JIT 教程](https://mesozoic-egg.github.io/tinygrad-notes/20240102_jit.html#shape-alignment)。
- **使用 Variable 进行形状对齐**：一名成员解释说，建议的方案解决了在循环中使用 `TinyJit`（特别是处理 Tensor 切片时）遇到的 **ShapeTracker** 差异问题。
   - 通过使用 `Variable` 并将其绑定到循环索引，可以实现形状对齐，从而解决另一位成员遇到的 `args mismatch` 错误。
- **Tensor-Variable 数学运算限制**：一位用户询问了在 **Tensor** 和 **Variable** 之间进行数学运算时，是否必须要求 **Tensor** 位于左侧（LHS）。
   - 对话中未明确给出确切原因，但这暗示了 tinygrad 在处理涉及符号变量（symbolic variables）运算时的一种约束。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1384993797771104438)** (9 messages🔥): 

> `Discord Spam, Mr. Beast` 


- **Discord 用户发送 Mr. Beast 垃圾内容**：一名 Discord 用户被要求停止在频道中刷屏 **Mr. Beast** 相关内容。
- **其他 Discord 用户投诉**：另一名 Discord 用户对这些垃圾内容表示了不满。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1384719821744242789)** (2 messages): 

> `MIPROV2 Optimization, Agent Implementation, Workflow Metrics` 


- **MIPROV2 Agent 优化**：一名成员表示，如果有输入输出示例，他们认为在使用 **MIPROV2** 之类的工具优化 Agent 实现方面没有任何**障碍**。
   - 另一名成员对工作流的输入/输出示例具体是什么样感到好奇。
- **工作流指标优化**：一名成员正在开展其他项目，计划使用内置的评估指标（eval metrics）来优化工作流和 Agent 实现。
   - 他们很期待在完成实现后进行分享。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1384794568356266086)** (5 messages): 

> `DSPy and LangGraph, DSPy in agentic coding IDEs, Finetuning Llama` 


- **DSPy ❤️ LangGraph 用于生产环境？**：一名成员询问是否有人在生产环境中结合使用了 **DSPy** 和 **LangGraph**，试图逆向工程 **Anthropic** 最近发布的一个多 Agent 研究项目。
   - 另一名成员提到了有人在 [此 Discord 链接](https://discord.com/channels/1161519468141355160/1202371242519441499/1382339683580772413) 中整理了一些相关内容来提供帮助。
- **DSPy 驱动 Agentic 编程 IDE？**：一名成员询问在 **Cursor** 或 **Roo Code** 等 Agentic 编程 IDE 中使用 **DSPy** 的情况。
   - 他们指出，目前设置“Agent”模式主要依赖于基于感觉（vibe-based）的提示词工程，例如使用 *'ONLY RETURN markdown'* 这样的指令。
- **使用框架微调 Llama？**：一名刚接触 **DSPy** 的成员询问推荐使用哪些框架或库来微调 **Llama** 模型。
   - 他们正在寻求关于库和框架的推荐。