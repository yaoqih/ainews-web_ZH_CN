---
companies:
- google-deepmind
- qwen
- gemini
- hugging-face
- ibm
- deepseek
date: '2025-02-07T03:47:44.376508Z'
description: '**"Wait" is all you need** 介绍了一种新型推理模型，该模型仅使用从 **Gemini 2.0 Flash Thinking**
  蒸馏出的 **1000 个带有推理轨迹的问题**，对 **Qwen 2.5 32B** 进行微调而成。它通过在提示中添加“Wait”一词来延长推理过程，从而实现了可控的测试时计算（test-time
  compute）。首席作者 **Niklas Muennighoff**（因在 **Bloom**、**StarCoder** 和 **BIG-bench**
  方面的工作而闻名）强调了该方法的效率，并指出它重现了著名的 o1 缩放图表（scaling chart）。


  此外，**Kyutai Moshi** 的 Hibiki 项目展示了在 iPhone 上令人印象深刻的离线法英实时翻译。近期发布的 AI 模型还包括：**DeepSeek
  R1 和 R3 开源模型**，这可能标志着开源领域的一个重大里程碑；**Hugging Face 的 SmolLM2**，强调针对小型语言模型（SLM）的以数据为中心的训练；以及
  **IBM 的 Granite-Vision-3.1-2B**，一款性能强劲的小型视觉语言模型。重点研究论文则聚焦于 **LIMO**（通过极简示例推理在 AIME
  和 MATH 基准测试中实现高准确率）以及 **Token-Assisted Reasoning**（通过混合潜变量 token 和文本 token 来提升语言模型的推理能力）。'
id: 2d857dce-c18a-4052-b410-8600e1f0a510
models:
- qwen-2.5-32b
- gemini-2.0-flash
- smollm2
- granite-vision-3.1-2b
original_slug: ainews-s1-simple-test-time-scaling-and-kyutai
people:
- niklas-muennighoff
title: s1：简单的测试时缩放（以及 Kyutai Hibiki）
topics:
- reasoning
- fine-tuning
- scaling-laws
- open-source-models
- data-centric-training
- vision
- multilingual-models
- language-model-reasoning
---

<!-- buttondown-editor-mode: plaintext -->**“Wait” 就是你所需的一切。**

> 2025年2月5日至2月6日的 AI 新闻。我们为你查看了 7 个 Reddit 子版块、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**210** 个频道，共 **4396** 条消息）。预计为你节省了 **490 分钟** 的阅读时间（以 200wpm 计算）。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 来参与 AINews 的讨论！

遗憾的是，我们报道这篇论文的时间稍晚了一些，但迟到总比不到好。[s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) 记录了一个新的推理模型，包含两项创新贡献：

- 基于 Qwen 2.5 32B，仅在 **1000 个配有推理轨迹（reasoning traces）的问题**上进行微调，这些轨迹是从 Gemini 2.0 Flash Thinking 蒸馏而来，并经过难度、多样性和质量过滤（在 16 台 H100 上训练了 26 分钟）。
- 可控的 **test-time compute**：通过强制终止模型的思考过程，或者在模型试图结束生成时多次附加 “Wait” 来延长其思考时间。

![image.png](https://assets.buttondown.email/images/614feebc-4fbf-4b51-9b55-5eb06ab593ac.png?w=960&fit=max)


主作者 [Niklas Muennighoff](https://scholar.google.com/citations?user=Me0IoRMAAAAJ&hl=en)（曾参与 Bloom, StarCoder, MTEB 以及 BIG-bench 的工作）[指出](https://x.com/Muennighoff/status/1886405528777073134)，这第二个技巧复现了著名的 o1 扩展图表（scaling chart）：


![image.png](https://assets.buttondown.email/images/bc28620b-478b-4847-bddb-df5360b4c34f.png?w=960&fit=max)


与 Bespoke-Stratos（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-bespoke-stratos-sky-t1-the-vicunaalpaca/)）相比，其过滤机制在样本效率上也表现得非常出色。


![image.png](https://assets.buttondown.email/images/fbeac362-f99f-494e-8eec-7f8e6521133c.png?w=960&fit=max)


我们还推荐阅读 [Simonw](https://simonwillison.net/2025/Feb/5/s1-the-6-r1-competitor/) 和 [Tim Kellogg](https://timkellogg.me/blog/2025/02/03/s1) 的解读文章。

**今日荣誉提名：**

**Kyutai Moshi** 去年因其带有内心独白的实时语音而引起轰动（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-o1-destroys-lmsys-arena-qwen-25-kyutai/)），现在 Hibiki 展示了[在 iPhone 上离线进行的非常令人印象深刻的法英实时翻译](https://www.reddit.com/r/LocalLLaMA/comments/1ij35u7/hibiki_by_kyutai_a_simultaneous_speechtospeech/)。对于一个[实习项目](https://x.com/kyutai_labs/status/1887495511474573517)来说，这表现相当不错。


![image.png](https://assets.buttondown.email/images/1667f4f3-6237-4c67-b0db-dab1f5ba9f0a.png?w=960&fit=max)


---

{% if medium == 'web' %}

**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 回顾

**AI 模型与发布**

- **DeepSeek R1 和 R3 开源发布**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1887403244046995458) 宣布 **R1-low-mid-high 模型即将推出**，这可能标志着 **LLM 领域第一个真正的开源时刻**，可与 **nginx、Blender 甚至 Linux** 相媲美。这一发布可能会**削弱由拥有专有技术的现任巨头组成的卡特尔所垄断的市场**。

- **Hugging Face 发布 SmolLM2**：[@_akhaliq](https://twitter.com/_akhaliq/status/1887371050628903065) 分享了 **Hugging Face 宣布 SmolLM2** 的消息，详见论文 **"When Smol Goes Big -- Data-Centric Training of a Small Language Model"**。[@LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1887500167055560922) 提供了 **SmolLM2 论文**的详细解读，强调 **数据是小模型（small LMs）强大性能背后的秘密武器**。

- **IBM 的 Granite-Vision-3.1-2B 模型**：[@mervenoyann](https://twitter.com/mervenoyann/status/1887521464292614382) 讨论了 **Granite-Vision-3.1-2B** 的发布，这是一个**在各种任务上表现令人印象深刻的小型视觉语言模型**。目前已提供 **Notebook** 用于**测试该模型**。

**AI 研究论文与发现**

- **LIMO：推理中的“少即是多”**：[@_akhaliq](https://twitter.com/_akhaliq/status/1887372529112686810) 重点介绍了 **LIMO**，展示了**通过极少但精确设计的示例，可以激发出复杂的推理能力**。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1887353699644940456) 指出，**LIMO 仅用 817 个训练样本就在 AIME 上达到了 57.1% 的准确率，在 MATH 上达到了 94.8%**，显著优于以往的方法。

- **令牌辅助推理**：[@_akhaliq](https://twitter.com/_akhaliq/status/1887373223152492665) 分享了论文 **"Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning"** 的见解，讨论了结合潜变量令牌（latent tokens）和文本令牌如何增强语言模型的推理能力。

- **长思维链（Long Chains of Thought）的进展**：[@gneubig](https://twitter.com/gneubig/status/1887495037820567815) 展示了关于**短思维链与长思维链**对比、**监督微调（supervised fine-tuning）与强化学习（reinforcement learning）**的作用，以及在语言模型中**控制推理长度**的方法的研究见解。

**AI 工具与平台**

- **Gradio DualVision 应用**：[@_akhaliq](https://twitter.com/_akhaliq/status/1887377041634316492) 介绍了 **DualVision**，这是一个用于**图像处理的 Gradio 模板应用**，具有**多模态预测**、**GPU 支持**和**示例库**，旨在提升用户体验。

- **Mistral AI 的 Le Chat 现已登陆移动端**：[@sophiamyang](https://twitter.com/sophiamyang/status/1887517050697842899) 宣布由 **Mistral AI** 开发的 AI 助手 **Le Chat** 正式发布**移动版**，具备**代码解释器（code interpreter）**和由 **Mistral 模型**驱动的**极速响应**等功能。

- **ChatGPT 中的 Canvas 共享功能**：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1887604146515423390) 宣布 **canvas 共享功能现已在 ChatGPT 上线**，允许用户**共享、交互或编辑 canvas**，增强了协作能力。

**AI 行业新闻与活动**

- **Google DeepMind 的 Applied ML Days 工作坊**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1887537279306244321) 邀请参与者参加 **Applied ML Days** 的两个工作坊，重点关注**使用 Google Gemini 构建 LLM 应用**以及**与基础模型（Foundational Models）的自然交互**。

- **Cerebras 为领先的 AI 实验室提供动力**：[@draecomino](https://twitter.com/draecomino/status/1887624699351605495) 分享称 **Cerebras** 目前正在**为一家领先的 AI 实验室的生产环境提供支持**，展示了 AI 基础设施和计算能力的进步。

- **Keras 社区会议**：[@fchollet](https://twitter.com/fchollet/status/1887573636082770345) 宣布将举行 **Keras 团队公开社区会议**，提供 **Keras 的最新动态**并为开发者提供**提问**机会。

**个人成就与更新**

- **Google Developers India 认可**：[@RisingSayak](https://twitter.com/RisingSayak/status/1887489752171225137) 对获得提名表示感谢，并感谢 **@GoogleDevsIN** 的认可，强调了在社区中的成就感。

- **Philipp Schmid 加入 Google DeepMind**：[@osanseviero](https://twitter.com/osanseviero/status/1887520341276098940) 欢迎 **Philipp Schmid** 加入 **Google DeepMind**，并表达了与包括 **@DynamicWebPaige**、**@film_girl** 等人在内的**梦之队**共事的兴奋之情。

**迷因/幽默**

- **程序员的类型**：[@hyhieu226](https://twitter.com/hyhieu226/status/1887540297103778268) 幽默地将程序员分为两类：编写**冗长类型声明**的人和为了简洁而使用 **'auto'** 的人。

- **过度自信警告**：[@qtnx_](https://twitter.com/qtnx_/status/1887496898484822126) 分享了个人反思，提醒**过度自信会导致失败**，建议**保持谦逊并勤奋工作**。

- **AI 实验室骗子**：[@scaling01](https://twitter.com/scaling01/status/1887487264965435629) 指责了 AI 社区中的 **YouTube 骗子**，指出他们从最初蔑视 AI 进展转向利用其牟利，暗示其关注点在于利润而非技术。

---

# AI Reddit 汇总

## /r/LocalLlama 汇总

**主题 1. Hibiki 语音对语音翻译 - 法语到英语能力**

- **[Hibiki by kyutai, a simultaneous speech-to-speech translation model, currently supporting FR to EN](https://v.redd.it/gpawbnvlyihe1)** ([Score: 448, Comments: 40](https://reddit.com/r/LocalLLaMA/comments/1ij35u7/hibiki_by_kyutai_a_simultaneous_speechtospeech/)): **Hibiki** 是由 **Kyutai** 开发的实时语音到语音翻译模型，目前支持 **法语 (FR) 到英语 (EN)** 的翻译。
  - **Hibiki 的功能**：**Hibiki** 因其实时翻译质量、自然度以及说话人相似度而受到赞誉，相关资源可在 [GitHub](https://github.com/kyutai-labs/hibiki) 和 [Hugging Face](https://huggingface.co/kyutai) 上获取。该模型在调整语速以适应语义内容的同时，能够保留说话人声音的能力备受关注，且据称其表现优于以往的系统。
  - **社区反馈与需求**：用户对该模型的表现表示赞赏，部分用户希望增加更多语言支持，特别是 **Spanish**（西班牙语）和 **Chinese**（中文）。用户还希望推出设备端（on-device）版本，以便在旅行和非英语地区使用。
  - **文化与开发观察**：社区中出现了一些关于法国人英语水平以及这款由法国开发的模型采用日语命名（Hibiki）的幽默评论。该项目的开源性质（类似于 **Mistral**）也受到了关注，人们对其未来在设备端翻译能力方面的进展充满期待。


**Theme 2. Challenges with Gemini 2.0 Pro Experimental Model**

- **The New Gemini Pro 2.0 Experimental sucks Donkey Balls.** ([Score: 205, Comments: 83](https://reddit.com/r/LocalLLaMA/comments/1iirej3/the_new_gemini_pro_20_experimental_sucks_donkey/)): 作者批评 **Gemini 2.0 Pro Experimental** 模型与之前的 **1206** 模型相比表现极差，指出了频繁出错和不必要的代码重构等问题。他们对 Google 发布质量倒退模型的模式感到沮丧，并将其与 **Flash 2.0**（原文为 Flesh light 2.0）在 OCR 任务中令人印象深刻的速度和效率进行了对比。
  - 许多用户对 **Gemini 2.0 Pro Experimental** 表示不满，指出其智力下降、以牺牲质量为代价提高速度等问题，一些用户更倾向于旧的 **1206** 模型，或者在编程和创意写作等特定任务中使用 **Flash 2.0** 等其他模型以获得更好的表现。
  - **Flash 2.0** 和 **o1** 模型因其有效性而受到称赞，特别是在处理复杂查询和在长任务中保持上下文方面；而较新的模型如 **o3-mini** 则因需要更冗长的输入才能理解用户意图而受到批评，这导致了效率低下。
  - 讨论凸显了一个更广泛的趋势：AI 模型正变得更快、更高效，但却以牺牲深度和一致性为代价。一些用户指出了当前评估指标的局限性，以及在实际应用中平衡速度与质量的挑战。


**Theme 3. Open WebUI Releases Code Interpreter and Exa Search Features**

- **Open WebUI drops 3 new releases today. Code Interpreter, Native Tool Calling, Exa Search added** ([Score: 185, Comments: 61](https://reddit.com/r/LocalLLaMA/comments/1iisj7j/open_webui_drops_3_new_releases_today_code/)): **Open WebUI** 在 **0.5.8** 版本中引入了重大更新，包括使用 **Pyodide** 实时执行代码的 **Code Interpreter**（代码解释器）、重新设计的聊天输入 UI，以及用于在聊天中检索信息的 **Exa Search Engine Integration**（Exa 搜索引擎集成）。此外，**Native Tool Calling Support**（原生工具调用支持）现已进入实验阶段，有望降低查询延迟并改善上下文响应。[发布详情](https://github.com/open-webui/open-webui/releases)可在网上查阅。
  - **Code Interpreter 和 Pyodide**：用户对使用 **Pyodide** 添加代码解释器表示赞赏，虽然注意到其局限性，但认可其在常见用例中的实用性。用户呼吁进一步改进，例如集成 **Gradio** 并支持下载结果（如绘图或处理后的数据）。
  - **社区贡献**：尽管有很多贡献者，但 **tjbck** 被公认为 **Open WebUI** 最主要且持续的贡献者，社区建议通过 [GitHub sponsorship](https://github.com/sponsors/tjbck) 来支持他们。该项目因其快速的功能更新以及相对于闭源 UI 的竞争优势而受到赞誉。
  - **文档处理与 RAG**：针对文档处理存在一些批评，特别是针对单文档引用使用简单的向量数据库 RAG，这在处理简单查询时经常失败。建议包括将文档、RAG 和搜索功能移至独立的流水线（pipelines）以跟上快速发展的步伐，并默认禁用 RAG 以便用户更好地控制。


**Theme 4. Over-Tokenized Transformer Enhances LLM Performance**

- **[Over-Tokenized Transformer - 一项新研究表明，在相同的训练成本下，大幅增加稠密 LLM 的输入词表（增加 100 倍或更多）能显著提升模型性能](https://www.reddit.com/gallery/1iiwmsq)** ([Score: 324, Comments: 37](https://reddit.com/r/LocalLLaMA/comments/1iiwmsq/overtokenized_transformer_new_paper_shows/)): 一篇新论文证明，将稠密 **Large Language Model (LLM)** 的输入词表大幅增加 100 倍或更多，可以在不增加训练成本的情况下显著提升模型性能。这一发现为通过扩大词表大小来提高 Transformer 效率提供了一种潜在策略。
  - **Tokenization 与词表大小**：将词表大小增加到数百万（而非典型的 **32k 到 128k**），可以通过使用更具意义的层级化 Token 来增强模型性能。这种方法通过将多个 Token 组合成新的 Token 来实现更快的收敛，尽管它主要提升的是训练效率，而非与词表大小成正比的最终性能。
  - **潜在挑战与考量**：人们担心贪婪 Tokenizer 会导致 Token 训练不足，这可能会在拼写错误以及对单字符变动敏感的任务（如算术或代数推理）中引发性能问题。此外，在使用较小 Token 时，对内存占用、推理速度和有效上下文窗口大小的影响也存在疑问。
  - **研究与对比**：三个月前的一项类似研究建议，像 **Llama 2 70B** 这样的模型应至少使用 **216k tokens** 以实现最佳算力利用率，甚至更大的 Token 数量也可能有益。该论文的发现对稠密模型特别有意义，但并未在 **Mixture of Experts (MoE)** 模型中显示出同样的改进，这凸显了一个值得进一步探索的领域。


## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Altman 承认 OpenAI 的竞争优势有所减弱**

- **Altman 承认 OpenAI 将不再能够保持巨大的领先优势** ([Score: 259, Comments: 69](https://reddit.com/r/OpenAI/comments/1ij13ub/altman_admits_openal_will_no_longer_be_able_to/)): **Sam Altman** 承认 **OpenAI** 将面临日益激烈的竞争，并且无法维持其此前在 AI 开发中的领先地位。据 **Fortune.com** 采访报道，他指出虽然 **OpenAI** 会产出更好的模型，但竞争差距将会缩小。[来源](https://fortune.com/2025/02/01/sam-altman-openai-open-source-strategy-after-deepseek-shock/)。
  - **OpenAI 的竞争策略**：几位评论者讨论了这样一种观点：**OpenAI** 试图通过控制其研究成果的发布来维持垄断，这使他们在竞争对手复制其工作之前拥有大约 **3-4 个月** 的优势。这一策略被视为在竞争格局中保持领先的临时措施。
  - **技术瓶颈与模型训练**：有一种观点认为 AI 技术可能正处于瓶颈期，用户注意到 **OpenAI** 承认面临不可避免的竞争。评论者强调了防止他人利用大模型输出训练自家模型的挑战，这表明 **OpenAI** 必须与其他公司共同持续创新。
  - **媒体与公众互动**：一位评论者的提问出现在了 **Fortune** 的文章中，引发了关于媒体伦理和此类出版物价值的讨论。尽管 **Sam Altman** 在 AMA 活动中能披露的内容有限，但人们对其开放态度表示赞赏。


**主题 2. 使用 AI 工具进行复杂分析的深度重构**

- **给我一个 Deep Research 的提示词，我来为你运行！** ([Score: 246, Comments: 111](https://reddit.com/r/OpenAI/comments/1iinuib/give_me_a_prompt_for_deep_research_and_ill_run_it/)): 该用户支付了 **$200** 以获取 **Deep Research** 的访问权限，并提议为社区运行提示词以评估其能力。他们将其与 **o3-mini-high** 进行了比较，指出 Deep Research 支持附件，但似乎并没有显著更好。他们邀请社区提交严肃的提示词并进行投票，以确定执行的优先顺序。
  - **复杂提示词的挑战：** 用户正在提交复杂的跨学科提示词，例如涉及 **particle physics**、**ontological spaces** 和 **depression subtypes** 的内容。这些通常需要 AI 进行澄清才能继续研究或分析，凸显了通过精确输入来优化 AI 响应的必要性。
  - **投资与经济预测：** 人们对在后 ASI 时代使用 AI 进行 **stock market predictions** 和经济分析表现出浓厚兴趣。用户对 ASI 对股票估值、GDP 增长和债券市场的影响感到好奇，强调了这些查询的投机性质，以及 AI 需要考虑多种场景和变量的需求。
  - **农业与环境系统：** 讨论包括创新的农业方法，如 **3 sisters method**，以及利用 AI 针对不同气候和土壤类型优化植物协作系统的潜力。这反映了应用 AI 增强可持续农业实践的广泛兴趣。


- **亲爱的 OpenAI，如果我每月为 Deep Research 支付 $200，能保存为 PDF/Markdown 就太好了！** ([Score: 229, Comments: 40](https://reddit.com/r/OpenAI/comments/1iit2y5/dear_openai_if_im_paying_200_per_month_for_deep/)): 作者对 **OpenAI** 的 **Deep Research** 表示失望，尽管每月费用高达 **$200**，但仍缺乏直接将报告保存为 PDF 或 Markdown 的功能。他们建议了一个变通方法：使用“复制”按钮获取原始 Markdown，然后将其粘贴到 **Notion** 中。
  - 许多用户对 **OpenAI** 的 **Deep Research** 缺乏直接的 PDF 或 Markdown 导出功能感到沮丧，强调 AI 应该减少繁琐的工作，并促进与 **Pages** 和 **Word** 等其他应用程序的更轻松集成。考虑到该工具每月 **$200** 的高昂成本，这些功能的缺失被视为重大疏忽。
  - 变通建议包括使用 Markdown 的“复制”按钮，然后粘贴到 **Markdown Editor** 中，或者使用 **print > save as PDF**。然而，用户发现这些手动过程与 AI 节省时间和简化任务的初衷背道而驰。
  - 围绕 AI 工具的命名惯例有一些幽默的讨论，比如与 **Gemini Deep Research** 的比较，以及对未来工具（如“Microsoft Co-pilot - In to Deep”版本）的期待。对话凸显了对当前 AI 能力的更广泛不满，以及对高级付费层级中更无缝功能的期望。


**主题 3. 用于可追踪健康诊断的开源 AI**

- **我如何构建了一个开源 AI 工具来诊断我的自身免疫性疾病（在花费 10 万美元并就诊 30 多次之后）——现在已开放供所有人使用** ([Score: 195, Comments: 27](https://reddit.com/r/OpenAI/comments/1ij6619/how_i_built_an_open_source_ai_tool_to_find_my/)): 作者分享了他们构建一个**开源 AI 工具**的历程，该工具旨在帮助诊断自身免疫性疾病。此前，作者花费了 **10 万美元**并走访了 **30 多家医院**，却始终没有得到明确的答案。该工具允许用户上传并标准化医疗记录，追踪化验结果的变化，并利用包括 **Deepseek** 和 **GPT4/Claude** 在内的不同 AI 模型来识别模式。他们提供了诸如 [Fasten Health](https://github.com/fastenhealth/fasten-onprem) 之类的资源用于获取医疗记录，并提到计划将文档解析迁移到本地运行。
  - **数据安全担忧**：几位评论者强调了在本地运行该工具以避免数据泄露的至关重要性，特别是考虑到**医疗记录**的敏感性以及此类数据在黑市上的高价值。**Mithril** 被提及作为处理医疗信息的安全 AI 部署选项，并强调了对 **FISMA** 和 **HITRUST** 等**认证**的需求。
  - **从碎片化诊断到发现**：讨论中包括一个个人案例，该用户曾收到过**椎间盘突出**和**脊柱弯曲**等多个诊断，后来使用该工具统一诊断为**强直性脊柱炎 (Ankylosing Spondylitis)**。还有建议考虑 **EDS (Ehlers-Danlos Syndrome)**，这表明了该工具在细化和发现复杂医疗状况方面的潜力。
  - **用户反应**：用户的强烈反应表明了对潜在严重数据泄露的惊讶和担忧，多条评论表示难以置信，并强调了处理敏感医疗数据不当的法律后果。


---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 提供的摘要之摘要的摘要

**主题 1. 模型能力与性能的突破**

- [**Hibiki 实现了像人类一样的实时语音翻译**](https://x.com/kyutai_labs/status/1887495488997404732)：Kyutai 的 **Hibiki 模型**实现了从 🇫🇷 到 🇬🇧 的*同声传译*，能够根据内容调整语速并保留说话者的声音。早期报告称赞 **Hibiki** 具有卓越的*质量*、*自然度*和*说话者相似度*，在实时沟通中足以媲美专业的人类口译员。
- [**Gemini 2.0 Flash 以极低成本大规模解析 PDF**](https://x.com/deedydas/status/1887556219080220683)：**Gemini 2 Flash** 现在能以大约 **每 6000 tokens 1 美元**的价格高效解析大型 PDF 文档，标志着文档处理领域的重大飞跃。这种具有成本效益的解决方案为需要从复杂文档格式中进行高吞吐量、高精度文本提取的应用开启了新的可能性。
- [**Unsloth 的 GRPO 让 DeepSeek-R1 推理在 7GB VRAM 上触手可及**](https://x.com/UnslothAI/status/1887562753126408210)：Unsloth 最新的 **GRPO** 更新将显存占用削减了 **80%**，允许用户仅需 **7GB VRAM** 即可复现 **DeepSeek-R1** 的推理过程。这一突破使先进推理模型的获取变得民主化，即使在资源受限的系统上也能进行 **Llama 3.1 (8B)** 和 **Phi-4 (14B)** 等模型的本地实验。

**主题 2. 面向 AI 工程师的工具与框架增强**

- [**GitHub Copilot 作为 Agent 觉醒，像专家一样编辑代码**](https://github.blog/news-insights/product-news/github-copilot-the-agent-awakens/)：**GitHub Copilot** 引入了 *Agent 模式*，并正式发布了 *Copilot Edits*，通过更智能的 AI 辅助增强了开发者的工作流。此次更新旨在提供更主动、更有效的编码支持，将 **Copilot** 转变为一个更集成、更强大的开发伙伴。
- [**Windsurf IDE 通过 Gemini 2.0 Flash 和 Cascade 网页搜索实现性能飞跃**](https://x.com/windsurf_ai/status/1887235006374035966)：**Windsurf** 现在支持*极速*的 **Gemini 2.0 Flash**，仅消耗 **0.25** 个 prompt 额度；同时 **Cascade** 通过 **@web** 获得了自动网页搜索功能，每次 flow action 消耗 1 个额度。这些增强功能旨在通过更快的模型和 IDE 环境内集成的的信息检索来提升开发者生产力。
- [**Cursor IDE 推出 GitHub Agents 和 Architect 功能以提升生产力**](https://forum.cursor.com/)：**Cursor IDE** 推出了新的 *GitHub Agents* 和 *Architect 功能*，旨在显著提升开发者生产力并简化复杂项目。虽然用户对这些新增功能充满热情，但一些用户报告了 Composer 工具中命令执行的潜在 bug，这标志着这些功能仍在积极开发和完善中。

**主题 3. 应对模型性能与基础设施方面的挑战**

- [**DeepInfra 供应商正面临 50% 的失败率，用户报告**](https://discord.com/channels/1091220969173028894)：**DeepInfra** 供应商目前有 *50% 的时间* 无法返回响应，导致零 token 生成和显著的处理延迟，特别是在 SillyTavern 等应用中。社区成员正在积极分享观察结果，并为 OpenRouter 上的不同模型和供应商寻求性能问题的解决方案。
- [**LM Studio 用户面临 API 错误潮，寻求调试指导**](https://discord.com/channels/1110598183144399058)：**LM Studio** 用户报告在加载模型时出现大量错误，如 *'unknown error'* 和 *'exit code: 18446744072635812000'*，促使人们呼吁提供详细的系统规格和 API 见解以进行有效调试。通过 API 连接时的状态处理问题也突显了对 API 交互需要更清晰的文档和用户支持。
- [**Codeium Jetbrains 插件因无响应和频繁重启受到批评**](https://discord.com/channels/1027685395649015980)：用户对 **Codeium Jetbrains plugin** 表示不满，理由是频繁响应失败且需要频繁重启，影响了开发者的工作流。一些用户选择切换回 Copilot 以获得可靠性，而其他用户报告了在 PhpStorm 中的特定错误，表明该插件性能持续不稳定。

**主题 4. 社区驱动的创新和开源贡献**

- [**独立研究人员利用 JAX 和 TPU 进行低成本 AI 研究**](https://docs.google.com/forms/d/e/1FAIpQLSeBXCs4vatyQUcePgRKh_ZiKhEODXkkoeqAKzFa_d-oSVp3iw/viewform)：独立 AI 研究人员正在探索 AI/ML 研究的现实领域，建议学习 **JAX** 以访问 **TPU Research Cloud**，从而实现资源高效的实验。社区引用 **OpenMoE** GitHub 仓库作为在资源有限的情况下实现 Mixture-of-Experts 模型影响力研究的典范。
- [**Y CLI 项目作为 OpenRouter 终端聊天替代方案出现**](https://github.com/luohy15/y-cli)：**Y CLI** 是一个个人项目，为 OpenRouter 提供了一个基于终端的网页聊天替代方案，将聊天数据本地存储在 jsonl 文件中，现在已支持 **Deepseek-r1** 推理。积极鼓励开发者通过其 GitHub 仓库为 **Y CLI** 做出贡献，促进社区驱动的开发并迎合终端爱好者。
- [**Hugging Face 社区克隆 DeepResearch 以实现开放访问**](https://huggingface.co/blog/open-deep-research)：**HuggingFace** 研究人员推出了 **DeepResearch** 的开源克隆版，强调了 Agent 框架的重要性，并引入了 **GAIA benchmark** 以促进社区贡献。该倡议促进了 AI Agent 技术的透明度和协作开发，鼓励更广泛的参与和创新。

**主题 5. AI 中的伦理辩论和商业模式审查**

- [**OpenAI 的利润优先做法引发社区辩论和怀疑**](https://x.com/OpenAI/status/1887616278661112259)：成员们正在辩论 **OpenAI** 等 AI 巨头的动机，批评其将*利润置于公共利益之上*，并质疑小型公司的竞争力。怀疑围绕 **OpenAI** 更新的 chain of thought 功能展开，人们担心企业议程主导 AI 发展，对其真实目的表示怀疑。
- [**AI 抵制情绪呼应了对加密货币的不信任，加剧了伦理担忧**](https://rentry.org/vwa65v85)：公众对 **AI** 的不信任与过去对 **cryptocurrency** 和 **NFTs** 的负面经历有关，影响了对 AI 技术的看法，并引发了关于 AI 开发的伦理担忧。批评者指出*未经许可的 AI 训练数据*以及 AI 扰乱劳动力市场的潜力，加剧了社会对 AI 伦理影响的广泛焦虑。
- [**Stability AI 的订阅成本和“私有图像”选项引发辩论**](https://discord.com/channels/1002292111942635562)：成员们对 Stability AI 的 **Max subscription** 中的“私有图像”选项提出质疑，辩论其是否含蓄地迎合了 NSFW 内容，而其他人则将云服务成本与本地电费进行了比较。这些讨论反映了用户对不同 AI 模型准入门槛和感知效用的不同态度，突显了关于 AI 服务经济学的持续辩论。

---

# 第一部分：Discord 高层摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 的 GRPO 现在支持通过 vLLM 进行推理！**：Unsloth 关于 **GRPO** 的最新更新允许以低至 **7GB VRAM** 的显存复现 **DeepSeek-R1** 的推理能力，同时支持在 [Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb) 上使用显存占用更低的模型。
   - 用户可以尝试最新的功能和 notebook 更新以提升性能，并训练 **Llama 3.1 (8B)** 和 **Phi-4 (14B)** 模型。
- **Unsloth 微调 R1 Distill Llama + Qwen！**：Unsloth 引入了对微调蒸馏版 **DeepSeek 模型**的支持，利用 **Llama** 和 **Qwen** 架构，并提供了[模型上传](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5)。
   - Unsloth 还支持新模型，如 **Mistral-Small-24B-2501** 和 **Qwen2.5**，可在 [Hugging Face 集合](https://huggingface.co/collections/unsloth/mistral-small-24b-2501-all-versions-679fe9a4722f40d61cfe627c)中找到。
- **量化可减少 60% 的 VRAM！**：最近的讨论强调了 **BitsandBytes 量化**的有效使用，通过选择性量化层可减少约 **60%** 的 VRAM 使用，更多细节可见 [Unsloth 的博客文章](https://unsloth.ai/blog/dynamic-4bit)。
   - 参与者讨论了在 GRPO 中使用多轮对话数据集，强调在模型训练期间保留推理上下文，并通过格式良好的数据集提高 AI 模型的推理能力。
- **OpenAI 优先考虑利润**：成员们辩论了像 **OpenAI** 这样的主要 AI 参与者的动机，批评其将利润置于公共利益之上，并对小公司的竞争力和潜在的联盟需求表示担忧。
   - 一位用户强调了 **OpenAI** 对 **chain of thought** 功能的更新，并链接到了[公告](https://x.com/OpenAI/status/1887616278661112259)，但回应显示对其真实目的持怀疑态度。
- **独立 AI 研究人员通过 JAX 使用 TPU！**：独立研究人员正在寻找现实领域来开始 AI/ML 研究，一位成员建议学习 **JAX** 以获取 **TPU Research Cloud** 的访问权限，并链接到了[申请表](https://docs.google.com/forms/d/e/1FAIpQLSeBXCs4vatyQUcePgRKh_ZiKhEODXkkoeqAKzFa_d-oSVp3iw/viewform)。
   - 成员们引用了 **OpenMoE** GitHub 仓库作为在 Mixture-of-Experts 模型中进行研究的相关示例，甚至在 **TinyStories 数据集**上预训练小型 Transformer。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stability 欢迎新任社区负责人**：Maxfield，Stability 的新任 Chief Community Guy，介绍了自己以改善社区参与度，他自 2022 年起曾在 **Civitai** 做出贡献。
   - 承认过去的参与度“乏善可陈”，Maxfield 计划推出一个**功能请求板**，并鼓励研究人员分享项目更新以提高**透明度**。
- **Civitai 饱受下载错误困扰**：用户报告在从 **Civitai** 下载模型时遇到 **Error 1101**，导致社区对停机时间感到沮丧。
   - 这些问题引起了对通过 **Civitai** 访问模型的可用性和可靠性的担忧。
- **用户剖析 Latent Space 的复杂性**：一位用户对交换 **latent space 参数**的工具复杂性表示困惑，建议需要更用户友好的解决方案。
   - 讨论涉及了新 **diffusion 模型**潜在的实现方式以及调整现有架构的挑战。
- **AI 订阅成本引发辩论**：成员们质疑 Stability 的 **Max 订阅**中的“私有图像”选项，辩论其是否迎合了 NSFW 内容，而其他人则将云服务成本与本地电费进行了比较。
   - 讨论强调了对不同 **AI 模型**的准入门槛与实用性之间的不同态度。
- **工程师寻求 AI Prompting 的清晰度**：一位用户寻求关于生成模型 **prompting 技巧**的见解，而其他人建议使用 [brxce/stable-diffusion-prompt-generator](https://ollama.com/brxce/stable-diffusion-prompt-generator) 等外部工具来提供帮助。
   - 对话强调了适应不同 **AI 模型**要求和生成令人满意的 prompt 的困难，尤其是跨平台时。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 新增 Gemini 2.0 Flash 支持**：Windsurf 现在支持 **Gemini 2.0 Flash**，如[这条推文](https://x.com/windsurf_ai/status/1887235006374035966)所述，每条消息仅消耗 **0.25** 个用户提示额度，每次工具调用消耗 flow action 额度。
   - 虽然 **Gemini 2.0 Flash** 速度极快且效率高，但其工具调用能力有限，但在回答代码库相关问题方面表现出色。
- **Windsurf Next Beta 版发布**：用户现在可以通过[此链接](https://codeium.com/windsurf/download-next)下载 Beta 版，体验 **Windsurf Next** 的最新功能。
   - 该 Beta 版允许用户抢先探索新的 AI 能力，并能灵活地在 **Next** 和 **Stable** 版本之间切换。
- **Jetbrains 插件遭到用户批评**：用户反映对 **Codeium Jetbrains 插件**感到沮丧，理由是该插件经常无响应且需要频繁重启。
   - 一位用户为了稳定性换回了 **Copilot**，而另一位用户报告了 **PhpStorm** 中与文件访问相关的错误。
- **用户报告 Windsurf 性能问题**：用户报告了 **Windsurf** 的性能问题，特别是在使用 **O3-mini** 和 **Gemini Flash** 等模型时，这些模型会在建议未完成时提前结束。
   - 一位用户对需要不断提示模型 *'continue'* 表示沮丧，并对浪费额度表示担忧。
- **Cascade 学会了网页搜索**：**Cascade** 现在可以自动或通过用户命令（如 **@web** 和 **@docs**）进行网页搜索，每次消耗 1 个 flow action 额度，详见 [Windsurf Editor 更新日志](https://codeium.com/changelog)。
   - 此功能支持 URL 输入，并利用网页上下文来改进回复，旨在提供更准确、更全面的信息。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 用户发现端口错误修复方法**：一位用户报告了 **Aider** 在加载模型元数据时出现无效端口错误，这表明可能存在配置问题。
   - 另一位成员建议通过覆盖默认的模型元数据文件作为权宜之计来解决此错误，以确保工具正常运行。
- **Gemini 独特的编辑需求**：用户讨论了 **DeepSeek** 和 **Gemini** 模型的不一致性，指出 **Gemini** 独特的编辑格式 (**udiff**) 与其他模型不同。
   - **Aider** 会自动为 **Google** 模型使用 **udiff**，同时为其他模型保持不同的默认设置，以适应这种差异。
- **AI 渗透测试有利可图但有风险**：一位成员分享了他们使用 LLM 进行渗透测试的项目，创建了一个由两个模型协作的模拟黑客环境。
   - 尽管 Token 消耗量很大，但专业的渗透测试可能极其丰厚，暗示了潜在的经济利益。
- **HuggingFace 克隆了 DeepResearch**：**HuggingFace** 的研究人员创建了一个开源的 **DeepResearch** 克隆版，详见其[博客文章](https://huggingface.co/blog/open-deep-research)。
   - 该倡议强调了 **Agent 框架**的重要性，并引入了 **GAIA 基准测试**，以促进社区贡献。
- **R1 模型产生垃圾 `<think>` Token**：一位用户报告称，在使用通过 **Together.ai** 提供的 **R1** 时，提交信息中充斥着 `` Token，并寻求配置指导。
   - 建议包括配置模型设置以尽量减少提交信息中的这些 Token，从而保持提交记录的整洁。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.0 Pro 引起热议**：用户对拥有 **200万 token 上下文** 的 **Gemini 2.0 Pro** 感到兴奋，这有助于复杂的交互，但也对其与 **Google AI Studio** 相比的易用性提出了担忧。
   - 免费替代方案提供了广泛的自定义功能，并可能在某些任务上为用户提供更好的结果；社区建议在额外功能的感知价值与付出的努力之间进行权衡。
- **DeepSeek 与 ChatGPT 争夺棋王头衔**：鉴于模型在推理方面的局限性，**DeepSeek** 和 **ChatGPT** 之间潜在的国际象棋比赛引起了用户的兴趣，这注定会非常有趣。
   - 用户对 DeepSeek **1美元一局**的国际象棋游戏与 OpenAI **100美元一局**的游戏进行了幽默的对比，暗示一些人更喜欢便宜但仍具挑战性的游戏。
- **Gemini Flash 2.0 和 Copilot 作为编程工具表现出色**：在关于编程的讨论中，成员们推荐了 **Gemini Flash 2.0** 和 **Microsoft Copilot**，因为它们的功能和性价比，特别是在高等数学方面。
   - 用户指出 **Copilot** 提供免费试用，使其在没有立即财务承诺的情况下更容易探索，并允许工程师“先试后买”。
- **Plus 用户热切期待 Deep Research 对话功能**：几位成员表达了对 **Deep Research** 对话功能尽快向 **Plus 用户** 开放的渴望，并指出他们在未来几天内有此**需求**。
   - 一位成员询问是否有人分享了关于 **Deep Research** 对话的信息，显然是在寻求见解，并促使其他人表达了对该功能加入 Plus 订阅的类似期待。
- **通过迭代编辑微调 AI**：一位成员建议使用 Python 统计字数并进行迭代，以确保更好的回复长度，但指出在尝试控制 AI 回复的 **Response Length** 时，这可能会影响创造力。
   - 成员们还指出使用编辑按钮编辑输入的重要性，通过调整输入直到满意为止来有效地塑造 AI 的输出，从而确保对话中上下文的连贯性。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE 获得 GitHub Agents 和 Architect 功能**：用户对 **Cursor IDE** 中新的 GitHub agents 和 architect 功能感到兴奋，这些功能旨在提高生产力。
   - 然而，正如 [Cursor 论坛](https://forum.cursor.com/) 所述，一些用户报告了在最近更新后，在 Composer 工具中运行命令时可能存在的 bug。
- **Gemini 2.0 自学能力扎实，但并非顶尖**：用户发现 **Gemini 2.0** 由于其价格优势和上下文管理能力，在自学任务中表现良好；一些讨论提到它很扎实，但在编程方面不如 **Sonnet**。
   - 社区指出其有效的上下文利用使其在处理大型代码库时具有吸引力，可能会冲击像 [Momentic](https://momentic.ai/) 这样的 **AI testing tools**。
- **剪贴板比较工具推荐**：社区推荐了一款用于剪贴板比较的 **VSCode extension**，它允许用户按照 [Microsoft 的 VSCode 文档](https://github.com/microsoft/vscode-docs/issues/7284) 中记录的方式与剪贴板内容进行比较。
   - 用户还在 **VSCode 的本地历史记录**与 **JetBrains 的 Timeline** 之间进行比较，认为 **Timeline** 效率更高，并推荐了来自 [VSCode Marketplace](https://marketplace.visualstudio.com/items?itemName=ryu1kn.partial-diff) 的 **Partial Diff** 扩展。
- **MCP 服务器配置需要更好的上下文**：一位用户正在寻求关于 **MCP server configurations** 以及访问 **Supabase** 密钥的帮助，并指出某些密钥的访问权限有限，同时提到了 [mcp-starter](https://github.com/daniel-lxs/mcp-starter) 的 GitHub 仓库。
   - 社区普遍强调了在 **Cursor** 内部改进上下文管理的必要性，特别是对于管理复杂项目，并参考了 [daniel-lxs/mcp-starter](https://github.com/daniel-lxs/mcp-starter/releases/) 的发布版本。
- **Cursor 的上下文瓶颈引发辩论**：关于 **Cursor** 上下文限制的担忧正在浮现，一些用户更倾向于使用 **Cline** 或 **Google models**，因为它们拥有更大的上下文窗口，或许是因为他们阅读了 Andrej Karpathy 关于 [vibe coding](https://x.com/karpathy/status/1886192184808149383) 的推文。
   - 关于上下文大小如何影响 **AI 模型** 有效性的争论仍在继续，特别是更大的上下文窗口如何提升广泛应用中的性能，以及 [Cursor 论坛](https://forum.cursor.com/t/model-specific-rules/47175) 中讨论的模型特定规则的作用。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Focus Mode 被取消**：用户注意到 Perplexity AI 暂时移除了 **Focus Mode**，引发了关于是否需要在 prompt 中明确提及 Reddit 等来源的必要性的争论。
   - 一些用户表示担心，这增加了他们有效引导 AI 信息溯源能力的复杂性。
- **解析 Perplexity Pro 中的模型使用**：用户正试图弄清楚 **Pro mode** 是否完全端到端地使用 **Claude 3.5** 等模型，还是集成了 **R1** 进行推理，这表明其采用了一种更复杂的、多模型协作的方法。
   - 见解表明，在将任务移交给选定模型进行最终答案生成之前，会有未公开的模型进行初始搜索。
- **ByteDance 深耕 Deepfake**：ByteDance 发布的新 **deepfake technology** 引发了 AI 社区对其伦理影响和潜在滥用风险的讨论。
   - 社区成员正在积极推测该技术的后果，权衡其创新可能性与造成危害的风险。
- **对模型透明度的需求激增**：用户敦促 **Perplexity AI** 就 **model specifications** 和更新进行更清晰的沟通，特别是涉及影响功能和性能的变更。
   - 更高的透明度有望减少用户困惑，并改善与平台 AI 功能的交互。
- **Sonar Pro 开发者因安全问题面临压力**：由于发现了一个 **security issue**，用户紧急呼吁联系 **Sonar Pro reasoning developers**。
   - 用户被引导发送邮件至 api@perplexity.ai 以解决该漏洞。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek 保险机制进一步加强**：OpenRouter 现在为没有收到 completion tokens 的 **DeepSeek R1** 请求提供保险，因此即使上游供应商收费，你也不会被扣费。
   - 标准版 **DeepSeek R1** 的完成率已从 **60%** 提高到 **96%**，使其成为一个更可靠的选择。
- **Kluster 的取消故障已修复**：一个 **Kluster** 集成问题曾导致 completion tokens 延迟，并由于未能取消超时的请求而产生意外费用。
   - 该问题现已得到解决，解决了用户在 OpenRouter 端显示超时但仍被扣费的问题。
- **Qwen 悄然退出**：Novita 正在弃用其 **Qwen/Qwen-2-72B-Instruct** 模型，OpenRouter 也将在同一时间禁用该模型。
   - 用户应停止使用该模型，以避免模型不可用时造成业务中断。
- **Y CLI 期待你的关注**：**Y CLI** 是一个个人项目和 Web Chat 的替代方案，它将所有聊天数据存储在单个 jsonl 文件中，并增加了对 **Deepseek-r1** 推理内容的支持，详见[这段 asciinema 录制](https://asciinema.org/a/701903)。
   - 鼓励开发者通过其 [GitHub repository](https://github.com/luohy15/y-cli) 为 **Y CLI** 贡献代码，并向 **terminal fans** 发出号召。
- **DeepInfra 表现极不稳定**：用户报告称，由于处理延迟增加，**DeepInfra** 目前有 **50%** 的概率无法返回响应，在使用 SillyTavern 等应用程序时经常导致零 token 补全。
   - 社区正在分享关于不同模型和供应商之间性能差异的观察，包括改进建议。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **用户面临 LM Studio API 错误潮**：用户报告在 **LM Studio** 中加载模型时出现“unknown error”和“exit code: 18446744072635812000”等错误，需要系统规格和 API 详情进行调试。
   - 一位用户在通过 API 连接到本地模型时在 **state handling**（状态处理）方面遇到困难，表明需要更好的 API 交互指导。
- **Obsidian 的 Smart Connections 扩展引发混乱**：用户在将 **Obsidian 的 Smart Connections 扩展**连接到 **LM Studio** 时遇到错误，理由是与其他扩展冲突以及 API 响应中缺少必填字段。
   - 故障排除涉及卸载冲突插件和重建缓存，尽管即使在建立连接后，持续的错误仍然存在。
- **TheBloke 模型仍是标准**：成员们询问从 **TheBloke** 下载 AI 模型的安全性和可靠性，即使他在社区的活跃度有所下降。
   - 确认 **TheBloke 的模型**仍是行业标准，鼓励用户关注社区频道以获取可用性更新。
- **DDR5 6000 EXPO 时序过于保守**：一位用户发现其 **DDR5 6000 EXPO 时序**比较保守，在推理过程中观察到的峰值内存带宽为 **72**。
   - 在完成 **4 轮 memtest86** 后，另一位成员建议尝试 [TestMem5](https://github.com/CoolCmd/TestMem5) 以进行更严格的稳定性评估。
- **DeepSeek R1 模型支持 GPU 加速吗？**：关于 **DeepSeek R1 Distill Qwen 7B 模型**的 GPU 加速出现了咨询，不确定哪些模型支持 GPU 使用。
   - 澄清只有像 **Llama** 这样的特定模型已知支持加速，而 **DeepSeek** 模型仍存在一些模糊性。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Home Assistant 获得功能性 MCP 客户端**：一位用户发布了具有 **MCP client/server** 支持的 **Home Assistant**，并计划通过 [met4citizen/TalkingHead](https://github.com/met4citizen/TalkingHead) 添加动画谈话头像，以实现更好的用户交互。
   - 该项目仍在开发中，因为开发者正在平衡有偿工作与开源开发。此外，人们对 **Home Assistant MCP** 与 **Claude** 等工具桥接的使用统计数据感到好奇。
- **Goose MCP 客户端表现强劲**：用户分享了在测试中使用 **Goose MCP Client** 的积极体验，强调了其有效性。 
   - 一个增强其日志功能的 Pull Request [block/goose@162c4c5](https://github.com/block/goose/actions/runs/13058183345/job/36804119892?pr=947) 正在进行中，其中包含在 Goose 的使用计数日志中包含缓存 Token 的修复。
- **Claude 努力处理图像显示**：一位用户报告了在 **Claude Desktop** 上将图像显示为工具结果时的挑战，遇到了输入错误。 
   - 该错误引发了推测，即由于将图像结果转换为嵌入资源可能是一个潜在的变通方案。 
- **PulseMCP 展示用例**：一个新的实际 **PulseMCP Use Cases** 展示亮相，包含使用各种客户端应用和服务器的说明和视频，并在 [PulseMCP](https://www.pulsemcp.com/use-cases) 上发布了这些用例。
   - 它强调了使用 **Gemini voice**、**Claude** 和 **Cline** 来管理 Notion、转换 Figma 设计以及创建知识图谱。
- **讨论移动端 MCP 选项**：成员建议 **Sage** 支持 iPhone，而 **Android** 用户的选项可能需要使用 **LibreChat** 或 **MCP-Bridge** 等 Web 客户端。
   - 这次对话强调了将 MCP 功能扩展到桌面环境之外的兴趣。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Gemini 2.0 Pro 生成 SVG**: 成员们讨论了 **Gemini 2.0 Pro** 在创建 SVG 方面表现出的令人印象深刻的性能，超越了 **o3-mini** 和 **R1** 等模型，正如 [Simon Willison 的博客](https://simonwillison.net/2025/Feb/5/gemini-2/)中所指出的。
   - 几位成员还观察到其增强的 SQL 查询能力，暗示 Google 在 **Gemini Flash 2.0** 上取得了重大进展。
- **DeepSpeed Dataloader Batch-size 困扰**: 一位用户报告了在使用 DeepSpeed 的自动 Batch Size 配置时，是否需要在 Dataloader 中手动定义 **batch_size** 的困惑。
   - 另一位成员建议将 DeepSpeed 标签集成到 Dataloader 中进行优化，并针对特定节点提出了潜在的性能修改建议。
- **Harmonic Loss 论文缺乏说服力**: 社区成员对 **Harmonic Loss 论文** 表示怀疑，认为其拼凑痕迹明显，尽管具有理论优势，但未能提供有意义的性能提升。
   - 一位成员指出，与该论文相关的 [GitHub 仓库](https://github.com/simplescaling/s1) 比论文本身提供了更有价值的信息。
- **Gemini 2.0 Flash 表现亮眼**: 通过 [LlamaIndex](https://openrouter.ai/google/gemini-2.0-flash-001) 尝试新款 **Gemini 2.0 Flash** 模型的用户报告了令人难以置信的速度，尽管没有 **Groq** 那么快。
   - 一位用户表示，该模型在**返回有效的 JSON 格式方面表现不佳**，结论是它可能不适合需要输出可靠性的任务。
- **S1 模型以低于 50 美元的成本问世**: 讨论了 **S1 推理模型**，强调了其与 **OpenAI 的 o1** 等模型相比的性能，但成本仅为一小部分，低于 **50 美元**。
   - S1 模型及其工具可在 [GitHub](https://github.com/simplescaling/s1) 上获得，它是通过 **Gemini 2.0** 蒸馏开发的。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Adobe 寻求 LLM Agent 研究合作伙伴**: Adobe 的一位高级 ML 工程师正在寻求 **LLM Agent 研究项目**的合作。
   - 邀请感兴趣的人士加入讨论，探索潜在的合作伙伴关系。
- **DeepSpeed 仍需指定 Batch Size**: 在使用 **DeepSpeed** 进行自动 Batch Size 调整时，仍需为 Data Loader 指定 **batch_size**。
   - 尽管配置了自动 Batch Size，这一要求依然存在。
- **主题泛化基准测试发布**: 一位成员分享了一个 [GitHub 仓库](https://github.com/lechmazur/generalization)，详细介绍了一个**主题泛化基准测试**，用于评估 **LLM** 从示例和反例中进行类别推理的能力。
   - 该基准测试与 **SAE autointerp** 性能的相关性受到了质疑。
- **RWKV 正在开发新架构**: **RWKV** 团队正在积极开发一些新架构，展现了其积极的态势。
   - 一位正在处理扩展问题的用户邀请大家就未来的合作进行交流。
- **MATS 8.0 批次申请现已开放**: **MATS 8.0** 批次的申请截止日期为 **2 月 28 日**，提供全职带薪的机械可解释性（Mechanistic Interpretability）研究机会，点击[此处](https://tinyurl.com/neel-mats-app)申请。
   - 之前的学员做出了重大贡献，他们在 **10 篇顶级会议论文**中的参与证明了这一点。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Deep Research 令用户感到兴奋**：成员们赞扬 **OpenAI 的 Deep Research** 能够高效收集相关的关联信息和来源，提升了他们的认知带宽。
   - 一位用户强调了它探索冷门在线社区并收集意想不到数据的能力。
- **AI 遭到的抵制呼应了对加密货币的担忧**：一些成员认为，公众对 **AI** 的不信任源于过去对 **cryptocurrency** 和 **NFTs** 的负面印象，这影响了对 AI 技术的看法。
   - 批评者担心 **AI 训练数据** 未经授权，以及 AI 对劳动力市场的破坏性影响，详见 [Why Everyone Is Suddenly Mad at AI](https://rentry.org/vwa65v85)。
- **处于法律模糊地带的有目标的 AI Agent**：一位用户旨在法律信托框架内开发一个目标驱动的 **AI Agent**，旨在开创关于 **AI 法律主体地位 (AI personhood)** 的法律讨论。
   - 反馈集中在工程复杂性上，包括集成财务管理功能，同时强调了定制软件解决方案的潜力，如 [I Built the Ultimate Team of AI Agents in n8n With No Code (Free Template)](https://www.youtube.com/watch?v=9FuNtfsnRNo) 中展示的案例。
- **模型合并热潮**：成员们讨论了合并 **AI 模型** 的策略，分享了关于改进模型指令微调 (instruction tuning) 和推理性能的见解。
   - 探索了各种**微调方法**，强调了在 **AI 训练** 中使用创新技术通过 [Unsloth Documentation](https://docs.unsloth.ai/) 等工具增强模型性能的益处。
- **合成数据之梦**：一位成员在面临 **Magpie** 输出的挑战后，正在寻找关于**合成数据生成**的资源，重点关注类似于 **Self-Instruct** 的基于种子 (seed-based) 的方法。
   - 他们发现了 [Awesome-LLM-Synthetic-Data](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data) GitHub 仓库，该仓库提供了关于**基于 LLM 的合成数据生成**的资源列表。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Schulman 从 Anthropic 离职**：领先的 AI 研究员、OpenAI 联合创始人 **John Schulman** 在入职约五个月后离开了 **Anthropic**，引发了对其职业生涯下一步的猜测 [链接](https://www.bloomberg.com/news/articles/2025-02-06/openai-co-founder-john-schulman-leaves-rival-firm-anthropic?srnd=undefined)。
   - 据消息人士透露，潜在的去向包括 **Deepseek** 和 **AI2**。
- **Copilot 变为 Agent**：**GitHub Copilot** 引入了 **agent mode**，增强了开发者辅助功能，并全面开放了 **Copilot Edits** [链接](https://github.blog/news-insights/product-news/github-copilot-the-agent-awakens/)。
   - 此次更新旨在通过 AI 提供更有效的编程支持。
- **LRM Test-Time Scaling 术语争议**：成员们对长程模型 (LRMs) 的 **test-time scaling** 术语提出质疑，强调**模型决定其自身的输出** [链接](https://discord.com/channels/1179127597926469703/1179208129083363358/1337167338444689502)。
   - 有人指出，扩展发生在**训练阶段**，这使得该术语具有误导性；一位成员称整个概念从根本上就是有缺陷的。
- **Qwen 取得惊人成果**：Qwen 2.5 模型在极少训练数据的情况下表现出令人印象深刻的结果，成员们讨论了他们的发现 [链接](https://x.com/lateinteraction/status/1887356471555563839)。
   - Aran Komatsuzaki 评论道，Qwen 模型似乎具有一种“魔力”，在有限的数据下实现了显著出色的性能。
- **Scale AI 面临转型挑战**：成员们认识到 **Scale AI** 有可能进行调整，但由于当前的运营模式和估值，挑战依然存在 [链接](https://www.turingpost.com/p/fod86)。
   - 共识是，在不断变化的环境中，如果不大幅改变方法，**前景将十分暗淡**。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 移动端用户仅限使用单一模型**：用户无法在移动版 **NotebookLM** 中切换模型，这一限制让期待更高灵活性的用户感到沮丧。
   - 这一限制阻碍了移动设备上的用户体验，导致习惯于在网页端管理模型的用户感到困惑。
- **Gemini 在 Sheets 中表现出色，NotebookLM 则略显吃力**：成员们对使用 **NotebookLM** 分析电子表格数据表示担忧，认为在 **Google Sheets** 中使用 **Gemini** 等工具更为合适。
   - 正如 [Engadget 报道](https://www.engadget.com/ai/gemini-can-now-do-more-complex-data-analysis-in-google-sheets-191218214.html)的那样，**Gemini** 可以使用 **Python** 代码生成见解和图表，这进一步巩固了 **NotebookLM** 作为主要 *文本分析工具* 的定位。
- **滑块功能可微调 AI 创造力**：一位用户在发现了一个与 AI 功能相关的 *漏洞 (exploit)* 后受到启发，建议集成用于调节 AI 创造力的滑块，类似于 **Gemini API** 中的功能。
   - 该功能将允许用户调整参数，从而对 AI 模型的创意输出实现更精准的控制。
- **NotebookLM 总结纽约预算听证会的法律证词**：一位用户使用 **NotebookLM** 记录了 **纽约州议会环境保护预算听证会** 的证词。
   - 该用户强调了由于许可问题分享这份详尽文档的挑战，笔记可在 [此处](https://docs.google.com/document/d/1kcUJvQiAwzX1GU4b0HvOUhLV0UtecLvuQSaTmfRFPpg/) 查看。
- **Max Headroom 带着故障风格回归，批判 AI**：标志性的 **Max Headroom** 带着新视频回归，展示了与 AI 互动的独特方式。
   - 正如 [Youtube 上所示](https://youtu.be/YXgav2-6VsI?feature=shared)，新内容幽默地批判了企业的 AI 实践，并呼吁观众分享和参与。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **2024 秋季 MOOC 证书终于发放**：在解决技术挑战后，**2024 秋季 MOOC 证书** 于今天 **太平洋时间上午 8 点** 正式发布。
   - 一些参与者因未完成课程作业被 *降级至 Trailblazer 等级*，且不提供补考机会。
- **证书发放时间难以确定**：成员们对证书发放时间表表示不确定，希望在 *不可预见的技术问题* 解决后的一两周内送达。
   - 一位成员注意到证书接收情况存在差异，表明可能存在影响通信的 *软退信 (soft bounce)* 问题。
- **测验开放情况引发混乱**：随着 Quiz-2 的启动，关于 Quiz-1 答案开放情况的担忧随之而来，促使成员们寻求关于答案发布新政策的澄清。
   - 社区成员澄清说，可以通过原始提交链接查看 Quiz-1 的分数。
- **证书等级分布**：据透露，参与者中共有 **301 名 Trailblazer**、**138 名 Masters**、**89 名 Ninjas**、**11 名 Legends** 和 **7 名 Honorees**。
   - 官方澄清，如果同时获得荣誉等级和特定等级，将仅标注荣誉等级。
- **课程体验赢得好评**：社区对课程期间获得的支持表示感谢，特别是对处理评分和证书查询的团队表示认可。
   - 参与者对课程表现出极大的热情，一位成员回顾了他们的学习历程，并强调了证书对未来发展的重要性。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA Blackwell 支持 OpenAI Triton**：由于 NVIDIA 与 OpenAI 的持续合作，**Triton 编译器**现在支持 **NVIDIA Blackwell 架构**，通过 [cuDNN](https://developer.nvidia.com/cudnn) 和 [CUTLASS](https://github.com/NVIDIA/cutlass) 增强了性能和可编程性。
   - 这一进展使开发者能够针对现代 AI 工作负载优化**矩阵乘法（matrix multiplication）**和**注意力机制（attention mechanisms）**，提高效率和能力。
- **降低 AI 研究成本**：成员们分享了独立研究人员如何在有限预算下对 **LLMs 和视觉（vision）**任务进行高效工作并微调模型，通过**低比特训练权重（low-bit training weights）**的稳定性来节省 AI 研究开支。
   - 使用 Muon 进行 **GPT-2 speedruns** 的成功被视为利用有限资源进行高影响力研究的典型案例。
- **FP8 Attention 需要 Hadamard 变换**：一位成员观察到，视频模型的 **FP8 Attention** 在使用 **Hadamard Transform** 时表现显著更好，大幅降低了错误率；[Flash Attention 3 论文](https://arxiv.org/pdf/2407.08608)表明这种方法对于 FP8 操作至关重要。
   - 另一位成员建议使用 [fast-hadamard-transform 仓库](https://github.com/Dao-AILab/fast-hadamard-transform/tree/master/csrc)在注意力机制之前实现 Hadamard，以获得更好的性能。
- **Reasoning Gym 引入推箱子（Sokoban）谜题**：一个 Pull Request 已提交，旨在将**推箱子谜题**添加到 **reasoning-gym** 中，为用户展示了一种新的谜题格式，包括谜题设置的图形解释以及移动示例。
   - 成员们还在讨论协作构建一个基础 Gym，将 **Rush Hour** 游戏集成到 **reasoning-gym** 中，以鼓励联合编码工作。
- **线性注意力（Linear Attention）面临蒸馏挑战**：一位成员尝试按照 [Lolcats 的方案](https://cdn.discordapp.com/attachments/1300872762163728550/1337017267925291068/distill_linear.ipynb?ex=67a5e9dd&is=67a4985d&hm=1a5dc02fb98a1f89ed72f7481e30459202f9d1de210fa3729663825137211832&)将一个小 LLM 蒸馏为**线性注意力模型**，但模型只输出了重复字符。
   - 该成员专门向 **Lolcats 团队**寻求帮助，突显了 AI 模型开发中经常依赖的社区支持。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **尽管定价昂贵，O3 依然保持领先**：根据 **general** 频道的讨论，尽管存在价格担忧，**O3** 的表现依然优于其他模型，*Llama 4* 被视为下一个潜在的挑战者。
   - 比较 **DeepSeek-R1 vs o3** 的链接[已在线发布](https://llm-stats.com/models/compare/deepseek-r1-vs-o3)，**o3-mini vs DeepSeek-R1** 的对比[也已提供](https://llm-stats.com/models/compare/o3-mini-vs-deepseek-r1)。
- **DeepSeek 在政治讨论中受到限制**：用户发现 **DeepSeek** 在敏感政治讨论中的限制比 *ChatGPT* 和 *O3-mini* 更多，经常导致意外的内容删除或回避。
   - 这突显了语言模型在面对敏感政治话题提示时的潜在约束。
- **DeepSeek 的知识截止日期引发疑问**：据报道，**DeepSeek** 的知识截止日期是 2024 年 **7 月**，鉴于现在已经是 **2025 年**，这引发了对其时效性的质疑。
   - 讨论中提到了利用时间上下文提取信息的 **Time Bandit** 方法与 **DeepSeek** 的关系，关于其 System Prompt 的更多细节可以[在线查阅](https://www.knostic.ai/blog/exposing-deepseek-system-prompts)。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **GRPO 实现取得重大进展**：一位成员报告了 **GRPO** 训练的成功实现，在 GSM8k 上达到了 **10% 到 40%** 的训练分数。
   - 在调试过程中，他们指出了死锁和内存管理方面的挑战，并计划进行改进并开放项目以供贡献。
- **Kolo 扩展至 Torchtune**：**Kolo** 在其 [GitHub 页面](https://github.com/MaxHastings/Kolo)上正式宣布支持 **Torchtune**。
   - 该项目为使用现有的最佳工具在本地微调和测试 LLM 提供了一套全面的解决方案。
- **Llama 3.1 和 Qwen 2.5 在配置上遇到困难**：成员们发现由于路径配置不匹配，在下载和微调 **Llama 3.1** 及 **Qwen 2.5** 时出现了 **FileNotFoundError** 问题。
   - 一位成员创建了一个 [GitHub issue](https://github.com/pytorch/torchtune/issues/2352) 以解决错误的默认路径并提出修复方案。
- **Hugging Face Fast Tokenizers 获得支持**：社区讨论了使用 **Hugging Face fast tokenizers** 的前景，成员们表示虽然目前存在局限性，但正在取得进展。
   - 一位成员提到 **Evan** 正在积极启用支持，详见 [此 GitHub pull request](https://github.com/pytorch/torchtune/pull/2350)。
- **Full DPO Distributed PR 面临障碍**：一位用户报告了其 [Full DPO Distributed PR](https://github.com/pytorch/torchtune/pull/2275) 在 GitHub checks 中遇到的问题，具体错误与 GPU 和 OOM 问题有关。
   - 错误提示 `ValueError: ProcessGroupNCCL is only supported with GPUs, no GPUs found!` 促使该用户向社区寻求帮助。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 偏离 Python，专注于 GPU**：在[最近的一次社区会议](https://www.youtube.com/watch?v=XYzp5rzlXqM)中，**Modular** 澄清说 **Mojo** 目前并不是 **Python** 的超集，而是专注于利用 **GPU** 和**性能编程**。
   - 这一转变强调提高 **Mojo** 在其设计应用中的效率，而不是扩大其语言框架。
- **解析器修订平衡分支成本**：一位成员建议 **parser** 需要针对处理多个数据切片进行调整，权衡分支成本，并指出*分支可能比大量数据传输更便宜*。
   - 对于那些不专注于更高性能需求的人来说，这是一个合理的考虑。
- **Msty 简化本地模型访问**：一位成员介绍了 **Msty**，这是一个兼容 OpenAI 的客户端，与使用 Docker 和其他复杂设置相比，它简化了本地模型交互，并强调了其易用性以及通过 [Msty 官网](https://msty.app)无缝访问 AI 模型的功能。
   - 强调了 Msty 的离线可用性和隐私重要性，表明它对于希望避免复杂配置的用户非常有利。
- **MAX Serve CLI 模仿 Ollama 的功能**：成员们讨论了在 **MAX Serve** 之上构建一个类似于 **ollama** 的 CLI，并指出 MAX Serve 已经可以通过 docker 容器处理 Ollama 提供的许多功能。
   - 讨论强调了与 Ollama 相比，在运行本地模型时获得更好性能的期望。
- **社区报告 OpenAI API 不兼容问题**：一位用户报告了 **max serve (v24.6)** 中 **OpenAI completions API** 缺失的功能，例如在指定 token 处停止生成，并建议他们在 **GitHub repo** 上提交 issue 以突出这些缺失的元素。
   - 该小组承认 OpenAI API 兼容性方面存在持续问题，特别是参考了 **v1/models** 端点，以及 [此 GitHub issue](https://github.com/modular/max/issues/292) 中提到的 token 停止和 prompt 处理等其他缺失功能。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Hibiki 引领实时翻译**：[Kyutai](https://x.com/kyutai_labs/status/1887495488997404732) 的 **Hibiki** 模型实现了从 🇫🇷 到 🇬🇧 的实时语音对语音翻译，保留了说话者的声音并能适应语境。
   - 早期报告称 **Hibiki** 在质量、自然度和说话者相似度方面表现出色，足以媲美人类口译员。
- **Melanie Mitchell 对 Agent 提出担忧**：@mmitchell_ai 的最新 [论文](https://huggingface.co/papers/2502.02649) 反对开发 **全自动 Agent (Fully Autonomous Agents)**，强调了伦理考量。
   - 该文章在 AI 社区引发了辩论，人们在热烈讨论中认可了她 **平衡的视角**。
- **Mistral AI 的 Le Chat 登场**：[Mistral AI](https://x.com/MistralAI/status/1887517520040448510) 推出了 **Le Chat**，这是一款专为日常个人和专业任务量身定制的多功能 AI 助手，可在网页和移动端使用。
   - 该工具将重新定义用户与 AI 的交互，可能影响工作流和个人日常习惯。
- **OpenAI 增强 o3-mini 功能**：OpenAI 在 **o3-mini** 和 **o3-mini-high** 中推出了增强的 **思维链 (chain of thought)** 功能（[来源](https://x.com/openai/status/1887616278661112259?s=46)），惠及免费和付费订阅用户。
   - 这些更新承诺提供更好的性能和更流畅的用户体验，再次印证了 OpenAI 对持续服务演进的承诺。
- **PDF 解析取得突破**：**PDF 解析** 现在可以大规模高效解决；根据 @deedydas 的说法，**Gemini 2 Flash** 解析大型文档的成本约为每 6000 个 tokens 1 美元。
   - 处理复杂文档方面的这一进步，为需要高质量文本提取的应用开启了新的可能性。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Gemini 2.0 现已全面开放**：来自 @google 的 **Gemini 2.0** 已发布并提供首日支持，开发者可以通过 `pip install llama-index-llms-gemini` 安装最新的集成包，并在 [公告博客文章](https://t.co/6oBbYpcFAU) 中阅读更多信息。
   - 更新后的 **2.0 Flash** 已在桌面和移动端的 **Gemini app** 中向所有用户开放。
- **LlamaParse 应对复杂财务报表**：Hanane D 展示了如何使用 **LlamaParse** 的“Auto”模式和 **@OpenAI embeddings** 准确且经济地解析 **复杂的财务文档**，详见此 [链接](https://t.co/UMZXeXJ5pS)。
   - 她的演示突出了解析技术在从复杂数据、图表和表格中提取相关见解方面的进展。
- **Embedding 打印困扰 LlamaIndex**：一名成员请求从 LlamaIndex 文档中删除 **embedding 打印** 部分，原因是其占用空间过大且影响可读性，详见 [GitHub issue](https://github.com/run-llama/llama_index/issues/17735)。
   - 另一名成员提议创建一个 Pull Request (PR) 来解决 **删除 embedding 打印** 的问题。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LLM 分类效果好，但噪声使其步履维艰**：成员们讨论认为，虽然 **LLM** 在分类方面很有效，但 **噪声数据 (noisy data)** 需要额外的技术（如稠密向量 (dense embeddings) 和自动编码器重排序器 (autoencoder rerankers)）来提高性能。
   - 这表明在处理具有挑战性的数据场景时，需要更复杂的策略。
- **延迟担忧削弱了对 LLM 的热情**：讨论显示，尽管 LLM 分类效果很好，但在有严格 **延迟要求 (latency requirements)** 的场景下，由于其处理限制，其适用性可能会降低。
   - LLM 的适用性取决于特定应用的延迟约束。
- **业务需求凸显 ML 不匹配**：一名成员指出，在向 ML 解决方案过渡期间，未能正确界定业务需求是一个 **失误**。
   - 从一开始就应该明确，如果低延迟是首要任务，传统的 LLM 可能不是理想选择。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 微调限制引发关注**：一名用户在 **Cohere** 中遇到了 **BadRequestError**（状态码：400），表明训练配置超过了 **250 个训练步数 (training steps)** 的上限，且 **batch size** 限制为 16。
   - 一名成员质疑这是否将微调限制在了 **4000 个样本** 以内，并指出这一限制以前并不存在。
- **征集 AI/ML 系统设计面试题**：一名成员在 Cohere 频道询问了针对 **AI/ML** 的 **系统设计面试题**。
   - 另一名成员确认了该请求并表示将进行收集，暗示团队将在此话题上进行协作。



---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **标准系统提示词（Canonical System Prompts）需求出现**：一名成员请求澄清**微调后的工具调用模型（fine-tuned tool-using models）**的**标准系统提示词**，并指出 **Gorilla 论文**中缺少这一细节。
   - 目标是确保模型能够可靠地返回函数调用的响应或 JSON，这表明需要标准化的 Prompt Engineering 实践。
- **Hugging Face 数据集寻求转换**：一名成员旨在通过转换数据并在 **Hugging Face** 上使用 `datasets.map` 来简化实验，这标志着向更灵活的数据操作迈进。
   - 这突显了为提高研究和开发目的下数据集的可用性和可访问性所做的持续努力。
- **Hugging Face 数据集格式问题**：一名成员报告了 **Hugging Face** 内部的数据集格式不匹配问题，即 **.json** 文件实际上包含的是 **jsonl** 数据，导致了兼容性问题。
   - 建议的解决方案包括将文件后缀重命名为 **.jsonl**，并调整数据集配置文件以与实际数据格式保持一致。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 相关论文发布**：一名成员分享了关于 **DSPy** 的论文[链接](https://arxiv.org/abs/2502.02508)。
   - 该论文分享在 **#papers** 频道中。
- **成员询问 Git 仓库**：在 **#examples** 频道中，一名成员询问其工作的 **Git repo** 是否可用，表示有兴趣获取相关代码或资源。
   - 该成员未指明具体是指哪个项目。
- **Colab 笔记本出现**：作为对 **Git repo** 查询的回应，一名成员提供了 [Colab 笔记本链接](https://colab.research.google.com/drive/1OXmTKexR9gX33DXRNEAe3dNuxkLXnutX?usp=sharing)。
   - 访问该笔记本需要**登录**，且它可能与 **DSPy** 的讨论有关。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

# 第 2 部分：分频道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1336790336411930707)** (516 条消息🔥🔥🔥): 

> `GRPO 与 vLLM 集成, DeepSeek 模型与微调, 量化技术, 多轮对话数据集, AI 伦理与数据隐私` 

- **支持 vLLM 的 GRPO**：GRPO 推理的最新更新允许用户以更低的显存占用复现 DeepSeek-R1 的推理，目前已支持显存低至 7GB VRAM 的模型。
   - 鼓励用户尝试和测试最新的功能及笔记本更新，以获得增强的性能。
- **微调 DeepSeek 模型**：Unsloth 已引入对蒸馏后的 DeepSeek 模型微调的支持，并计划在未来的笔记本中指导用户完成此过程。
   - 这些蒸馏模型采用 Llama 和 Qwen 架构，从而实现了与 Unsloth 的兼容。
- **量化见解**：最近的讨论强调了 BitsandBytes 量化的有效使用，通过选择性地量化层，可以将 VRAM 使用量显著降低约 60%。
   - 用户表示有兴趣在 Unsloth 的博客文章中进一步阅读有关此量化技术的详细信息。
- **对话数据集处理**：参与者讨论了在 GRPO 中使用多轮对话数据集的问题，强调了在模型训练期间保留推理上下文的细微差别。
   - 大家一致认为，格式良好的数据集可以增强 AI 模型的推理能力。
- **伦理与 AI 发展**：关于数据隐私以及 AI 发展中闭源模型与开源替代方案的影响引发了辩论。
   - 用户对 AI 的发展方向表示担忧，并强调了构建符合用户价值观而非公司议程的模型的重要性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.downloadmoreram.com/">DownloadMoreRAM.com - CloudRAM 2.0</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek-R1 是最强大的开源推理模型，其性能与 OpenAI 的 o1 模型不相上下。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://huggingface.co/collections/unsloth/llama-32-66f46afde4ca573864321a22">Llama 3.2 - unsloth 集合</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF">unsloth/SmolLM2-135M-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - 动态 4-bit 量化</a>: Unsloth 的 Dynamic 4-bit 量化有选择地避免对某些参数进行量化。这在保持与 BnB 4bit 相似的 VRAM 占用的同时，大大提高了准确度。</li><li><a href="https://satori-reasoning.github.io/blog/satori/">Satori: 通过 Chain-of-Action-Thought 强化学习结合自回归搜索增强 LLM 推理能力</a>: 论文 GitHub Hugging Face 介绍 自 OpenAI 的 o1 发布以来，研究界付出了巨大努力，旨在通过先进的推理能力增强开源 LLM。T...</li><li><a href="https://x.com/UnslothAI/status/1887562753126408210">来自 Unsloth AI (@UnslothAI) 的推文</a>: 你现在可以在本地设备上复现 DeepSeek-R1 的推理过程了！仅需 7GB VRAM 即可体验“顿悟”时刻。Unsloth 将 GRPO 训练内存占用降低了 80%。15GB VRAM 即可转换...</li><li><a href="https://tenor.com/view/travis-neil-primrose-neil-ba-dum-tss-ba-dum-gif-11550351308913763721">Travis Neil Primrose GIF - Travis Neil primrose Neil - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/cognitivecomputations/stablemax-orthogonal">GitHub - cognitivecomputations/stablemax-orthogonal</a>: 通过在 GitHub 上创建账户，为 cognitivecomputations/stablemax-orthogonal 的开发做出贡献。</li><li><a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">black-forest-labs/FLUX.1-dev · Hugging Face</a>: 未找到描述</li><li><a href="https://deepnewz.com/ai/stanford-s-s1-32b-model-outperforms-openai-s-o1-preview-27-on-aime24-math-using-bc4ff754">斯坦福大学的 s1-32B 模型在 AIME24 数学题上使用 1,000 个多样化问题，表现优于 OpenAI 的 o1-Preview 27% | DeepNewz</a>: 斯坦福大学的研究人员引入了一种名为 Simple Test-Time Scaling (s1) 的新方法，该方法增强了语言模型的推理性能。s1-32B 模型在包含 1,000 个多样化问题的训练集上进行了微调...</li><li><a href="https://github.com/unslothai/unsloth/issues/1561">[修复中] 更多微调支持 · Issue #1561 · unslothai/unsloth</a>: 支持序列分类；为 Gemma 等模型提供 Flex Attention 支持；支持可变序列长度和自动取消填充/填充；Tool Calling；重构并合并 xformers, SDPA, flash-attn, flex-attention</li><li><a href="https://github.com/unslothai/unsloth/issues/1376#issuecomment-2632615715">llama.cpp GGUF 损坏 [已修复] · Issue #1376 · unslothai/unsloth</a>: 截至 2024 年 12 月 3 日 - 已修复。请通过 <code>pip install --upgrade --no-deps --no-cache-dir unsloth</code> 更新 Unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1337126988737019974)** (1 条消息): 

> `Unsloth 中的推理 (Reasoning)、DeepSeek-R1、模型微调、新模型支持` 


- **Unsloth 引入 R1 推理能力**：随着 R1 的发布，Unsloth 展示了推理能力，可以在本地或在 [Colab](https://unsloth.ai/blog/r1-reasoning) 上免费进行训练。该方法允许用户仅需 **7GB VRAM** 即可复现 R1-Zero 的见解。
   - 此外，[Colab notebooks](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb) 为 **Llama 3.1 (8B)** 和 **Phi-4 (14B)** 模型提供了资源。
- **DeepSeek-R1 提升准确率**：推出了全新的 **R1 Dynamic 1.58-bit** 模型，承诺比标准位宽具有更高的准确率。更多详情和教程可以在 [DeepSeek-R1 博客](https://unsloth.ai/blog/deepseek-r1)中找到。
   - 此外，用户现在可以微调 R1 Distill Llama + Qwen 模型，[模型已提供上传](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5)。
- **支持 Mistral 和 Qwen 模型**：Unsloth 增加了对 **Mistral-Small-24B-2501** 和 **Qwen2.5** 等新模型的支持，可以在 [Hugging Face 集合](https://huggingface.co/collections/unsloth/mistral-small-24b-2501-all-versions-679fe9a4722f40d61cfe627c)中找到。
   - 用户还可以在 [Hugging Face](https://huggingface.co/unsloth?sort_models=created&search_models=1m#models) 上探索具有 **1M 上下文 (context)** 的模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-GRPO.ipynb)">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-r1-on-your-own-local-device)">Unsloth Documentation</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth?sort_models=created&search_models=1m#models)">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models)">Unsloth Documentation</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1336892147567431722)** (38 条消息🔥): 

> `Model Merging, DeepSeek V3, 用户利益 vs. 企业利润, OpenAI 进展, AI 的社会价值` 


- **关于 Model Merging 局限性的讨论**：一位用户对自己在资金充足的情况下重现 **DeepSeek V3** 的能力表示怀疑，称：“我不够聪明。”
   - 另一位成员表示致力于解决这一挑战，提到需要带头开展围绕解决方案的对话。
- **对 AI 企业的担忧**：针对 **OpenAI** 等主要 AI 参与者的动机展开了辩论，一位成员批评这些公司将利润置于公共利益之上，称：“目前那些把利润放在人之上的人赢了”。
   - 讨论引发了对小公司在这种情况能否有效竞争的质疑，并建议为了生存，结盟可能是必要的。
- **OpenAI 的最新更新**：一位用户强调了 OpenAI 关于不同用户层级的 **chain of thought** 功能更新的公告，链接见[此处](https://x.com/OpenAI/status/1887616278661112259)。
   - 对该更新的回应包括对其是否具有实际用途的怀疑，评论如：“只是普通的更新”。
- **AI 的价值与用户关注**：一位用户认为大多数竞争对手未能关注终端用户的需求，强调“实际上没有竞争对手直接关注用户”。
   - 他们提出，强调**社会价值**可能比当前的企业模式产生更大的效益。
- **联邦集体训练方法**：成员们讨论了 AI 领域潜在的发展路径，包括通过**联邦集体方式 (federated collective ways)** 训练模型并挑战现有垄断的想法。
   - 关于通过卓越的智能或合作方法超越大型实体的可行性意见不一，并触及了关于索取生产资料的马克思主义观点。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenAI/status/1887616278661112259">来自 OpenAI (@OpenAI) 的推文</a>：OpenAI o3-mini 为免费和付费用户更新了 chain of thought，o3-mini-high 为付费用户更新。</li><li><a href="https://github.com/SakanaAI/evolutionary-model-merge">GitHub - SakanaAI/evolutionary-model-merge: Official repository of Evolutionary Optimization of Model Merging Recipes</a>：模型融合方案进化优化的官方仓库 - SakanaAI/evolutionary-model-merge
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1336818382820868207)** (188 条消息🔥🔥): 

> `Unsloth 模型训练, GRPO 与奖励函数, 模型合并问题, 使用 LoRA 进行持续预训练, Adapter 性能比较` 


- **讨论了 Unsloth 模型训练策略**：用户分享了使用 Unsloth 的经验，强调了预训练和指令数据对模型性能的影响。
   - 一位用户指出，他们的模型在进行额外微调之前表现更好，凸显了进一步训练面临的挑战。
- **GRPO 与奖励函数的有效性**：讨论了如何使用 GRPO 教导模型遵循特定格式，重点在于引导输出的奖励函数。
   - 参与者建议将奖励函数与监督微调 (SFT) 相结合，可以增强训练效果。
- **模型合并的挑战**：一位用户表达了对将 LoRA adapters 与基础模型合并的担忧，透露合并后的训练导致输出质量下降。
   - 建议在对模型性能完全有信心之前，继续使用 LoRA adapters 进行训练，而不是进行合并。
- **使用 adapters 进行持续预训练的问题**：参与者遇到了在恢复训练期间，adapter 未能继续训练 lm_head 和 embed tokens 的问题。
   - 这引发了关于合并是否能解决这些问题，或者仅凭 adapter 是否能有效进行训练的疑问。
- **对 Unsloth 的优化与赞赏**：用户对 Unsloth 框架的优化表示赞赏，提到了效率的提升。
   - RL 特性的发布被强调为一项重大增强，许多用户对此充满期待。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/deepseekr1-dynamic#running%20r1">运行 DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://huggingface.co/bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF?show_file_info=Qwen2.5-Math-1.5B-Instruct-Q6_K.gguf>">bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF · Hugging Face</a>: 暂无描述</li><li><a href="https://github.com/huggingface/">Hugging Face</a>: 构建未来的 AI 社区。Hugging Face 拥有 287 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks>">Unsloth 文档</a>: 暂无描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>: 以下是我们所有 notebook 的列表：</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 微调 Llama 3.3, DeepSeek-R1, Mistral, Phi-4 &amp; Gemma 2 LLMs 速度提升 2-5 倍，显存占用减少 70%</a>: 微调 Llama 3.3, DeepSeek-R1, Mistral, Phi-4 &amp; Gemma 2 LLMs 速度提升 2-5 倍，显存占用减少 70% - unslothai/unsloth</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">教程：如何微调 Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>: 为初学者准备的指南，用于创建自定义个人助手（类似 ChatGPT）并在 Ollama 上本地运行</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb">Google Colab</a>: 暂无描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 条消息): 

yaska0971: 名称字符串太太太太太长了。请缩短它
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1337099685068013620)** (6 messages): 

> `现实的 AI 研究领域、结合 JAX 的 TPU Research Cloud、OpenMoE 项目、预训练小型 Transformer` 


- **探索现实的 AI 研究途径**：一位独立研究员正在寻找无需大量资金即可开始 AI/ML 研究的现实领域，并提到了预训练大型模型的局限性。
   - 建议的关注领域包括**比较研究**、创建新的数据生成模型，或改进 **LLM** 的 Prompt。
- **学习 JAX 以获取 TPU 访问权限的价值**：一位成员建议学习 **JAX** 以获取 **TPU Research Cloud** 的访问权限，并表示这将非常有益。
   - 他们提供了一个 [TPU Research Cloud 申请表链接](https://docs.google.com/forms/d/e/1FAIpQLSeBXCs4vatyQUcePgRKh_ZiKhEODXkkoeqAKzFa_d-oSVp3iw/viewform)。
- **OpenMoE：社区的一个范例**：一位成员强调了 **OpenMoE** 的 **GitHub** 仓库，将其作为在 Mixture-of-Experts 模型领域进行研究的相关示例。
   - 该仓库由一位已入职 **DeepMind** 的研究员领导，展示了在该领域的成功。
- **在 TinyStories 上预训练小型 Transformer**：另一位成员建议在 **TinyStories 数据集**上预训练小型 **Transformer**，作为一种潜在的研究选择。
   - 这可以为不需要大量资源的独立项目提供新的路径。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/papers/2311.10770">论文页面 - Exponentially Faster Language Modelling</a>：未找到描述</li><li><a href="https://github.com/XueFuzhao/OpenMoE">GitHub - XueFuzhao/OpenMoE: A family of open-sourced Mixture-of-Experts (MoE) Large Language Models</a>：一个开源的 Mixture-of-Experts (MoE) 大语言模型系列 - XueFuzhao/OpenMoE</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSeBXCs4vatyQUcePgRKh_ZiKhEODXkkoeqAKzFa_d-oSVp3iw/viewform">通过 TPU Research Cloud 加速您的研究</a>：请完成以下问题以供申请审核。
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1336798894218281130)** (1 messages): 

> `Maxfield 介绍、社区参与计划、功能请求板、展示研究员进展` 


- **新任社区负责人 Maxfield 自我介绍**：Stability 新任首席社区负责人 Maxfield 进行了自我介绍，并表达了他在自 2022 年深度参与 AI 媒体生成领域后，致力于改善社区参与度的承诺。
   - 他强调此前曾在 **Civitai** 做出贡献，并承认最近的社区参与度**乏善可陈**。
- **两项提升参与度的新举措**：Maxfield 宣布了两项旨在改善沟通和分享社区兴趣的举措，其中包括一个用于收集模型和工具建议的**功能请求板**。
   - 他强调，目标是确保社区的声音被听到，且这些举措将同时服务于爱好者和专业人士。
- **鼓励创作者分享进展**：Maxfield 计划通过鼓励 Stability 的研究员和创作者分享其项目和开发的最新动态来提升**透明度**。
   - 他认为正在进行的出色工作不应被保密，而应更广泛地与社区分享。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1336788750096662539)** (459 条消息🔥🔥🔥): 

> `Stability AI 更新、模型兼容性、AI Prompting 技巧、社区动态、AI 订阅与成本` 


- **关于 Stability AI 功能和订阅成本的讨论**：成员们讨论了 Max 订阅中的“私有图像”选项，询问这是否暗示了 NSFW 内容，一些人分享了他们使用该服务的经验。
   - 其他人强调了使用云服务进行模型训练所涉及的成本，并将其与本地电力成本进行了比较。
- **从 Civitai 下载的问题**：几位用户报告在尝试从 Civitai 下载模型时遇到 Error 1101，这表明可能存在服务器问题。
   - 社区分享了对停机和访问模型困难的沮丧情绪。
- **Latent space 工具和模型训练**：一位用户对交换 Latent space 参数工具的复杂性表示困惑，表示需要更直观的解决方案。
   - 讨论内容包括针对新型 Diffusion 模型的潜在实现，以及运行现有架构所面临的挑战。
- **对不同 AI 模型的普遍态度**：参与者分享了他们对各种 AI 模型（包括 Stability AI 和 Midjourney）的经验和看法，反映了对订阅模式和社区动态的复杂感受。
   - 人们在思考准入门槛成本与使用特定 AI 模型的效用之间的价值平衡。
- **AI Prompting 技巧和工具**：一位用户寻求关于生成模型 Prompting 技巧的见解，而其他人则建议使用外部工具来生成 Prompt。
   - 讨论包括了适应不同 AI 模型要求以及生成令人满意的 Prompt 的难度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ollama.com/brxce/stable-diffusion-prompt-generator">brxce/stable-diffusion-prompt-generator</a>: Stable Diffusion Prompt 生成器</li><li><a href="https://tenor.com/view/creep-hands-adventure-time-deer-remove-the-gloves-gif-15274634">Creep Hands GIF - Creep Hands Adventure Time - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://lu.ma/fyu8iqnk">Multimodal AI Agents - Hackathon · Luma</a>: Gen AI AgentsCreatorsCorner，与 Google Deepmind、Weights &amp; Biases、Together.ai、Stytch、Senso、LlamaIndex 等合作，充满热情地……</li><li><a href="https://github.com/NeuralNotW0rk/LoRAW">GitHub - NeuralNotW0rk/LoRAW: 用于 stable-audio-tools 的灵活 LoRA 实现</a>: Flexible LoRA Implementation to use with stable-audio-tools - NeuralNotW0rk/LoRAW</li><li><a href="https://purplesmart.ai/">Expanding the frontiers of AI creativity - PurpleSmartAI</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1336794325979238410)** (3 条消息): 

> `Gemini 2.0 Flash, Windsurf Next Beta, Windsurf 1.2.6 Patch Fixes, Cascade Web Search` 


- **Gemini 2.0 Flash 加速编程**：新的 **Gemini 2.0 Flash** 现已在 Windsurf 上线，每条消息仅消耗 **0.25** 个用户提示词额度（user prompt credits），每次工具调用消耗 0.25 个 Flow 操作额度（flow action credits）。
   - 该模型*极速*且高效，虽然在工具调用能力方面有限，但非常擅长回答与代码库相关的问题。
- **加入 Windsurf Next 体验早期功能**：用户可以通过[此链接](https://codeium.com/windsurf/download-next)下载 Beta 版本，抢先体验 **Windsurf Next** 的最新功能。
   - Beta 版允许用户探索新的 AI 能力，同时支持根据需要在 Next 版本和 Stable（稳定）版本之间切换。
- **Windsurf 1.2.6 补丁解决额度问题**：最新的 **Windsurf 1.2.6 Patch** 修复了在向 Flex 额度过渡期间出现的局部额度问题，详见[完整更新日志](https://www.codeium.com/changelog)。
   - 此补丁通过确保操作的额度过渡更加顺畅，提升了用户体验。
- **Cascade 的新网页搜索功能**：Cascade 现在可以自动执行网页搜索，或通过 **@web** 和 **@docs** 等用户命令执行，使其在获取实时信息方面更加多能。
   - 该功能包括支持 URL 输入，并整合网页上下文以优化回复，每次网页搜索消耗 **1 个 Flow 操作额度**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/windsurf_ai/status/1887235006374035966">来自 Windsurf (@windsurf_ai) 的推文</a>：Gemini 2.0 Flash 现已在 Windsurf 中可用！根据我们的测试，Flash 表现为：⚡ 极速 💪 高效 - 仅消耗 0.25X 额度 🧠 工具调用能力有限，但非常适合咨询关于...</li><li><a href="https://codeium.com/windsurf/download-next">感谢下载 Windsurf 编辑器</a>：今日体验未来的编辑器。Windsurf 编辑器是首款由 AI Agent 驱动的 IDE，让开发者保持专注流。现已支持 Mac, Windows 和 Linux。</li><li><a href="https://codeium.com/blog/windsurf-next">Windsurf Next 发布</a>：介绍 Windsurf Next，这是我们可选的 Windsurf 预发布版本。</li><li><a href="https://www.codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf 编辑器的最新更新和变化。
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1336832439594651759)** (31 条消息🔥): 

> `Codeium Jetbrains Plugin Issues, DeepSeek Feature Request, Function Length Display in CodeLens, Educational Email Discounts, Version Updates and Bug Reports` 


- **Codeium Jetbrains 插件面临批评**：用户对 Codeium Jetbrains 插件表示不满，称其经常无响应且需要频繁重启，一名用户为了稳定性已切换回 Copilot。
   - PhpStorm 中报告了一个与文件访问相关的错误，表明该插件的性能仍存在问题。
- **请求实现 DeepSeek 功能**：一位用户请求在 Codeium 中加入 DeepSeek 功能，并强调了其潜在优势。
   - 另一位成员对此表示支持，并建议用户通过指定渠道正式提交功能请求。
- **Codelens 和函数长度查询**：一位成员询问是否可以显示函数长度，随后确认该功能为 Codelens。
   - 然而，关于如何在 VSCode 的 Codelens 中添加特定逻辑的具体实现，用户间尚不明确。
- **关于教育邮箱折扣的澄清**：讨论涉及了教育邮箱的折扣资格，用户澄清邮箱必须以 .edu 结尾才符合条件。
   - 也有人对所使用的检测方法表示担忧，推测该资格可能仅针对美国的教育机构。
- **Codeium 模型中的额度使用**：一位用户询问在 VSCode 中使用 Codeium Premier 模型是否需要额度，得到的确认是聊天功能不消耗额度。
   - 官方澄清扩展中的所有模型均不与额度挂钩，确保用户可以自由使用。


  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1336789037784105073)** (345 条消息🔥🔥): 

> `Windsurf 性能问题、Gemini Flash 与 Sonnet 对比、多种 AI 模型的使用、Windsurf 安装与登录问题、Cascade 文件处理的用户体验` 


- **Windsurf 性能问题**：许多用户报告了 Windsurf 的性能问题，特别指出在调用 O3-mini 和 Gemini Flash 等模型时，模型在未提供完整建议的情况下过早结束。
   - 一位用户对需要不断提示模型“继续（continue）”表示沮丧，这引发了对浪费额度（credits）的担忧。
- **Gemini Flash 与 Sonnet 对比**：一些用户正在对比 Gemini Flash 和 Sonnet，指出 Gemini Flash 速度更快、成本更低，但在整体质量上仍落后于 Sonnet。
   - 根据讨论，由于 Claude 在编程挑战中的性能指标更高，它仍然是许多人的首选。
- **多种 AI 模型的使用**：讨论涉及根据任务选择不同的 AI 模型，一些用户主张使用 DeepSeek 进行调试，使用 Claude 获得更高质量的输出。
   - 关于 Windsurf 在不同模型下的 Agent 能力存在争议，表明性能会根据所选模型而有所不同。
- **Windsurf 安装与登录问题**：一位用户报告称，尽管拥有 Pro 订阅，但在登录 Windsurf IDE 时遇到困难，面临试用激活和身份验证问题。
   - 另一个案例提到在尝试重新安装 IDE 后出现了关于版本不匹配的错误消息。
- **Cascade 文件处理的用户体验**：用户表示在 Angular 项目中手动向 Cascade 聊天添加多个文件非常繁琐，正在寻求更好的集成方法。
   - 建议使用右键选项复制文件路径，以便更高效地将其包含在讨论中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://windsurf‑stable.codeiumdata.com/wVxQEIWkwPUEAGf3/apt">未找到标题</a>：未找到描述</li><li><a href="https://codeium.com/changelog/windsurf-next">Windsurf Next 更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf Next 扩展的最新更新和更改。</li><li><a href="https://codeium.canny.io/feature-requests">功能请求 | Codeium</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://open-vsx.org/extension/sr-team/vscode-clangd-cmake">Open VSX Registry</a>：未找到描述</li><li><a href="https://codeium.canny.io/feature-requests/p/auto-commit-message">自动提交信息 | 功能请求 | Codeium</a>：根据已提交的文件上下文生成提交信息</li><li><a href="https://codeium.canny.io/feature-requests/p/roll-over-of-pro-credits">Pro 额度结转 | 功能请求 | Codeium</a>：未使用的高级用户 Prompt 额度和高级 Flow Action 额度结转至下个月</li><li><a href="https://x.com/kevinhou22/status/1886827501004931511">来自 Kevin Hou (@kevinhou22) 的推文</a>：我们热爱文档！📖 我正在努力改进/添加更多 @ 文档快捷方式到 @windsurf_ai，告诉我你想要什么，我会尽可能多地添加... 🧵 另外向 @mintlify 致敬，感谢其自动托管所有文档...</li><li><a href="https://www.reddit.com/r/Codeium/comments/1ihn6gp/submit_your_docs_suggestions_to_head_of_product/">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1336788439122837635)** (337 条消息🔥🔥): 

> `招聘更新、Aider 错误处理、DeepSeek 和 Gemini 模型、LLM 编辑格式、使用 LLM 进行渗透测试`

- **恭喜获得工作录用通知！**：一位成员宣布他们收到了工作录用通知，并强调与当前职位相比，薪资大幅增加，且闲置时间更少。
   - 他们对这个新机会以及通过增加收入来资助其 AI 项目的潜力感到兴奋。
- **遇到常见的 Aider 错误**：一位用户报告了 Aider 的一个问题，在尝试加载模型元数据时提示无效端口错误。
   - 另一位成员建议通过覆盖默认的模型元数据文件作为解决此错误的变通方法。
- **关于 DeepSeek 和 Gemini 模型的讨论**：用户讨论了 DeepSeek 最近的不稳定性以及 Gemini 模型的性能，特别提到了 1206 模型的困难。
   - 具体而言，他们指出 Google 的模型默认使用一种独特的编辑格式 (udiff)，这与其他模型不同。
- **理解 LLM 编辑格式**：用户谈到了 Aider 使用的各种编辑格式，区分了标准 diff 格式和 Aider 自有的 UDIFF 语法。
   - 他们澄清说，Aider 会自动为 Google 模型使用 udiff，同时为其他模型保持不同的默认设置。
- **使用 LLM 的渗透测试项目**：一位成员分享了他们使用 LLM 创建渗透测试环境的项目，强调了两个模型如何在模拟黑客环境中协同工作。
   - 尽管 Token 使用量很高，但他们提到了潜在的经济收益，因为专业的渗透测试可能非常有利可图。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.litellm.ai/docs/providers/perplexity">Perplexity AI (pplx-api) | liteLLM</a>: https://www.perplexity.ai</li><li><a href="https://aider.chat/docs/usage/lint-test.html#linting">Linting and testing</a>: 自动修复 Linting 和测试错误。</li><li><a href="https://aider.chat/docs/more/edit-formats.html">Edit formats</a>: Aider 使用各种“编辑格式”让 LLM 编辑源文件。</li><li><a href="https://openrouter.ai/perplexity/sonar-reasoning">Sonar Reasoning - API, Providers, Stats</a>: Sonar Reasoning 是 Perplexity 提供的一种基于 [DeepSeek R1](/deepseek/deepseek-r1) 的推理模型。它允许开发者利用内置网络搜索的长思维链（Chain of Thought）。运行 Sonar Reas...</li><li><a href="https://deepclaude.com/docs">DeepClaude</a>: 未找到描述</li><li><a href="https://www.litellm.ai/">LiteLLM</a>: LiteLLM 处理超过 100 种 LLM 的负载均衡、回退和支出跟踪，全部采用 OpenAI 格式。</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">Advanced model settings</a>: 为 LLM 配置高级设置。</li><li><a href="https://www.youtube.com/watch?v=MQw_zncxk-E">Vercel's Guillermo Rauch on AI and the Future of Coding - Ep. 47</a>: 阅读 Dan Shipper 关于分配经济的文章：https://every.to/chain-of-thought/the-knowledge-economy-is-over-welcome-to-the-allocation-economyGuillerm...</li><li><a href="https://www.youtube.com/watch?v=pb6GtL0WFT8">Autonomous AI in Action 💪 | Live Codestream with Aider &amp; Deepseek v3 🧠</a>: 在这个实验中，Deepseek v3（通过 Aider）负责在极少人工干预的情况下构建一个项目。该 AI 正在开发一个摘要生成器应用...</li><li><a href="https://github.com/Aider-AI/aider/issues/3159">Add tree-sitter-hcl-tags.scm for terraform repomap generation · Issue #3159 · Aider-AI/aider</a>: 这是 tree-sitter-hcl-tags.scm 的初稿，旨在为 terraform 仓库启用 repomap。在将 .tf 扩展名添加到 grep-ast 并将其识别为 hcl 后，我可以在本地使用它。使用...</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>: Aider 是你终端里的 AI 结对编程工具。</li><li><a href="https://github.com/getAsterisk/deepclaude/issues/13">Aider's benchmark is explicitly not about using R1 thinking tokens (and says that using them did worse) · Issue #13 · getAsterisk/deepclaude</a>: 嘿 DeepClaude 的朋友们，我有点困惑为什么你们要突出引用 Aider 的 R1+Sonnet 基准测试结果。关于这些结果的博客文章和 Twitter 帖子明确指出...</li><li><a href="https://github.com/Aider-AI/aider/issues/2052">SDK not that good · Issue #2052 · Aider-AI/aider</a>: 嗨，我非常喜欢你们的工具——我正在使用它，觉得很棒。然而，当我尝试用 Python 封装它时，并没有预想中那么容易。虽然文档展示了如何使用 coder.r...</li><li><a href="https://github.com/jj-vcs/jj">GitHub - jj-vcs/jj: A Git-compatible VCS that is both simple and powerful</a>: 一个既简单又强大的 Git 兼容 VCS - jj-vcs/jj</li><li><a href="https://t.co/ss4DAzMi4J">GitHub - lumina-ai-inc/chunkr: Vision model based document ingestion</a>: 基于视觉模型的文档摄取。通过在 GitHub 上创建一个账户来为 lumina-ai-inc/chunkr 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/aider/issues/2879">Bug: Creating files named with the file extension only without the filename. · Issue #2879 · Aider-AI/aider</a>: 它建议了正确的文件名，但随后会生成名为 php 而不是 install.php 的文件，或者名为 sql 而不是 migration.sql 的文件。</li><li><a href="https://github.co">GitHub · Build and ship software on a single, collaborative platform</a>: 加入全球应用最广泛、由 AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在这里构建推动人类进步的软件。</li><li><a href="https://github.com/Aider-AI/aider/issues/3139#issue-2832352562">Aider creates files using random strings as filenames · Issue #3139 · Aider-AI/aider</a>: 问题：使用 o3-mini 进行提示，它一直在使用非常奇怪的文件名，例如 2. New file for modular integration of the embedding worker New file (empty file) ────────────────────────────── 我认为...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1336792416727990293)** (23 条消息🔥): 

> `Aider 对 Agents 的支持、在 Aider 中暂存更改、使用 R1 的 Commit 消息、模型配置问题、Architect 模式功能` 


- **Aider 可能很快支持 Agents**：一位用户询问 Aider 是否会在某个时间点支持 Agents，这表明该工具的功能可能会有进一步的开发或更新。
   - 这反映了用户对扩展 Aider 功能以提升用户体验的持续关注。
- **暂存更改功能尚未推出**：一位用户询问是否有办法在 Aider 中暂存（stage）更改而不是直接提交（commit），这表明用户希望获得更细粒度的控制。
   - 这表明用户对能够简化版本控制流程的工作流功能感兴趣。
- **Commit 消息中 <think> Token 的困扰**：一位用户分享了关于在使用通过 Together.ai 提供的 R1 时，Commit 消息中充斥着 `<think>` Token 的担忧，并寻求配置建议。
   - 建议包括适当配置模型设置，以尽量减少 Commit 消息中的这些 Token。
- **内部 OpenWeb UI 实例的求助**：一位用户请求关于如何在 Aider 中使用来自内部 OpenWeb UI 实例的 JWT API key 的指导，并指出标准的 API key 不可用。
   - 这突显了限制直接 API 访问的内部工具所带来的挑战，使集成工作变得复杂。
- **对 Architect 模式的疑虑**：一位用户对 Architect 模式的行为表示困惑，称其无法按预期让他们主导后续步骤。
   - 其他人指出，使用 `/ask` 命令可以实现所需的控制，而无需调整模式。



**提到的链接**：<a href="https://aider.chat/docs/config/reasoning.html">Reasoning models</a>：如何配置来自第三方提供商的推理模型设置。

  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1336806433760612363)** (2 条消息): 

> `Gemini 2.0, Open Deep Research, HuggingFace, Agent 框架` 


- **Gemini 2.0 在 LMSYS 上线**：最新模型 **Gemini 2.0** 已在 [LMSYS](https://lmarena.ai) 上亮相，展示了其在能力上的进步。
   - 此次发布旨在加强 AI 社区围绕下一代模型的讨论。
- **HuggingFace 发布 Open Deep Research 克隆版**：HuggingFace 的研究人员创建了 **DeepResearch** 的开源克隆版，详情见其 [博客文章](https://huggingface.co/blog/open-deep-research)。
   - 该计划强调了 **Agent 框架** 的重要性，并提出了 **GAIA 基准测试**，为社区贡献铺平了道路。



**提到的链接**：<a href="https://huggingface.co/blog/open-deep-research">Open-source DeepResearch – Freeing our search agents</a>：未找到描述

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1336792980035735775)** (276 条消息🔥🔥): 

> `Gemini 2.0 Pro, OpenAI vs DeepSeek, AI for Coding, Chatbot Aggregators, AI Model Comparisons` 


- **对 Gemini 2.0 Pro 的兴奋**：用户对 **Gemini 2.0 Pro** 的能力表示兴奋，特别强调了其 **200 万 token 上下文**，这使得处理复杂的交互和创意写作任务成为可能。
   - 然而，也有人对其易用性表示担忧，并将其与提供广泛自定义选项的 **Google AI Studio** 等免费替代方案进行了对比。
- **聊天机器人大决战**：**DeepSeek** 与 **ChatGPT** 之间潜在的国际象棋对决引发了关注，用户根据模型在推理方面的局限性推测了结果。
   - 用户幽默地对比了 DeepSeek 的 **1 美元棋局** 与 OpenAI 的 **100 美元棋局** 的价格差异。
- **AI 编程推荐**：在关于编程的讨论中，成员们推荐了 **Gemini Flash 2.0** 和 **Microsoft Copilot**，认为它们具有出色的功能和性价比，特别是在处理高等数学方面。
   - 用户指出 **Copilot** 提供免费试用，使其在无需立即支付费用的情况下更容易进行探索。
- **AI 自动化编程解决方案**：对话转向寻找一种依赖性最小的**自动化编程方式**，寻求能够减少终端操作时间的解决方案。
   - 目标是发现使用 AI 进行编程任务的最具 **Agentic**（代理化）的方法，并强调用户友好的环境。
- **AI 模型对比与体验**：用户分享了使用各种 AI 模型的混合体验，指出 **Gemini 2.0** 和 **Sonnet 3.5** 在用户任务中表现更好，但功能各异。
   - 共识是任务需求极大地影响了模型的选择，同时需要关注能力和成本。



**提到的链接**：<a href="https://tenor.com/view/fire-writing-gif-2088247993237804628">Fire Writing GIF - Fire writing - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1336830335719182388)** (5 条消息): 

> `Deep Research chat for Plus users` 


- **对 Plus 版本 Deep Research 的期待**：几位成员表达了对 **Deep Research** 对话功能尽快面向 **Plus 用户** 开放的热切期待。
   - 成员们特别强调了他们对此功能的**需求**，希望在未来几天内发布。
- **关于信息不明的对话**：一位成员对之前的评论评论道“我从未听说过”，表明对该话题缺乏了解。
   - 另一位成员则带着怀疑回应道：“并不是这样，你是什么意思？”
- **请求分享 Deep Research 对话**：一位成员询问是否有人分享了关于 **Deep Research** 对话的信息，显然是在寻找相关见解。
   - 这一询问促使其他人也表达了对该功能加入 Plus 订阅的类似期待。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1336825419885842493)** (6 条消息): 

> `Response Length Control, Undesired Behavior in AI Models, Input Influencing Output` 


- **确保最佳回复长度的策略**：一位成员建议使用 Python 来统计字数并进行迭代，以确保更好的回复长度，尽管这可能会影响创造力。
   - 他们指出，更多的输入通常会带来更多的输出，但也承认在没有外部辅助的情况下，统计字符数是一项挑战。
- **通过编辑获取理想输出**：另一位成员强调了使用编辑按钮修改输入以有效塑造 AI 输出的重要性。
   - 他们建议在继续对话之前不断调整输入直到满意，以确保对话中上下文的连贯性。
- **不当行为依然存在**：一位成员对 AI 模型重复出现的不当行为表示沮丧，除非对其进行主动控制。
   - 他们强调需要制定策略来管理和减轻这些行为，以获得更好的用户体验。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1336825419885842493)** (6 条消息): 

> `Controlling AI Response Length, Managing Undesired AI Behavior` 


- **确保 AI 回复长度的方法**：一位成员建议使用 Python 统计字数并进行迭代，但警告这可能会影响**创造力**。他们指出，更多的内容通常会导致更长的回复，但 AI 在没有辅助的情况下很难统计**字符数**。
- **通过编辑塑造 AI 输出**：另一位成员提到，使用编辑按钮可以帮助控制不当行为，允许用户不断优化 Prompt 直到满意。这种方法为“上下文”或“对话”链中的下一次交互奠定了基础。


  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1336788595117391872)** (282 条消息🔥🔥): 

> `Cursor IDE 更新, Gemini 2.0 性能, 剪贴板对比工具, MCP Server 配置, AI 模型中的 Context 限制` 


- **Cursor IDE 更新与功能**：用户分享了关于 Cursor IDE 更新的见解，特别是 GitHub agents 的引入，以及关于可能进一步提升生产力的 architect 功能的讨论。
   - 还提到了在 Composer 工具中运行命令时遇到的挑战，这表明最近更新后可能存在 Bug。
- **Gemini 2.0 性能评估**：多位用户对 Gemini 2.0 表示满意，指出其在自学任务中表现稳健，尽管有人认为它在编程方面并不优于 Sonnet。
   - 讨论中还涉及了该模型的性价比和有效的 Context 利用，这增加了它对大型代码库的吸引力。
- **剪贴板对比工具建议**：参与者提供了关于剪贴板对比工具的建议，重点介绍了允许与剪贴板内容进行对比的 VSCode 扩展。
   - 用户对比了 VSCode 本地历史记录的功能，并建议使用像 Timeline 这样的工具，以获得类似于 JetBrains 的更高效率。
- **MCP Server 配置与文档**：一位用户寻求关于 MCP server 配置以及获取 Supabase 必要密钥的帮助，并分享了某些密钥提供的访问权限有限。
   - 社区讨论了配置问题以及在 Cursor 中进行更好 Context 管理的需求，特别是针对复杂项目。
- **AI 模型中的 Context 限制**：用户对 Cursor 中的 Context 限制表示担忧，更倾向于使用像 Cline 或 Google 模型这样提供更大 Context 的模型。
   - 辩论了 Context 大小对 AI 模型有效性的影响，特别是更大的 Context windows 如何增强在更广泛应用中的性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://livebench.ai/">LiveBench</a>：未找到描述</li><li><a href="https://momentic.ai/">AI 测试工具 | 自动化 AI 测试 - Momentic</a>：利用 Momentic 先进的自动化测试 AI 工具增强您的 QA 流程。通过可靠的 AI 驱动测试实现更快交付。</li><li><a href="https://marketplace.visualstudio.com/items?itemName=ryu1kn.partial-diff">Partial&#32;Diff&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>：Visual Studio Code 扩展 - 比较（diff）文件内、跨文件或与剪贴板之间的文本选择。</li><li><a href="https://docs.fireworks.ai/deepseek/general-deepseek">未找到标题</a>：未找到描述</li><li><a href="https://forum.cursor.com/t/o3-mini-is-live-what-version-are-we-getting/46674">O3-mini 已上线！我们得到的是哪个版本？</a>：首先，向快速推出此功能的 Cursor 团队致敬！团队表示他们的开发人员在大多数任务中仍然更喜欢 Sonnet（这让他们感到惊讶）。（来源：x.com）根据 O...</li><li><a href="https://forum.cursor.com/t/model-specific-rules/47175">特定模型的规则</a>：新的 Cursor 规则系统非常棒，如果能针对特定模型设置规则就更好了。</li><li><a href="https://forum.cursor.com/t/new-diff-is-shown-between-each-line-of-the-previewed-code/36903">预览代码的每一行之间都显示了新的 Diff</a>：描述 Bug，预览代码的每一行之间都显示了新增内容。复现步骤：直接询问 Cursor。预期行为：正常的代码预览。截图 / 屏幕录制 操作系统...</li><li><a href="https://x.com/karpathy/status/1886192184808149383">Andrej Karpathy (@karpathy) 的推文</a>：有一种我称之为“氛围编程”（vibe coding）的新型编程方式，在这种方式中，你完全沉浸在氛围中，拥抱指数级增长，甚至忘记了代码的存在。这之所以成为可能，是因为 LLM（例如...</li><li><a href="https://github.com/daniel-lxs/mcp-starter/releases/">版本发布 · daniel-lxs/mcp-starter</a>：通过在 GitHub 上创建账号来为 daniel-lxs/mcp-starter 的开发做出贡献。</li><li><a href="https://github.com/daniel-lxs/mcp-starter/releases/tag/v0.1.0">Release v0.1.0 · daniel-lxs/mcp-starter</a>：初始版本发布</li><li><a href="https://github.com/microsoft/vscode-docs/issues/7284">文档化“将当前文件与剪贴板比较”功能 · Issue #7284 · microsoft/vscode-docs</a>：在 VS Code 1.19 中引入，您可以将当前活动文件与剪贴板内容进行比较。命令：File: Compare Active File with Clipboard (workbench.files.action.compareWithClipboard) 快捷键...</li><li><a href="https://github.com/robert-at-pretension-io/mcp">GitHub - robert-at-pretension-io/mcp: code</a>：代码。通过在 GitHub 上创建账号来为 robert-at-pretension-io/mcp 的开发做出贡献。</li><li><a href="https://github.com/daniel-lxs/mcp-perplexity">GitHub - daniel-lxs/mcp-perplexity</a>：通过在 GitHub 上创建账号来为 daniel-lxs/mcp-perplexity 的开发做出贡献。</li><li><a href="https://github.com/daniel-lxs/mcp-starter">GitHub - daniel-lxs/mcp-starter</a>：通过在 GitHub 上创建账号来为 daniel-lxs/mcp-starter 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1336796638928306320)** (247 条消息🔥🔥): 

> `Perplexity AI Focus Mode, Perplexity Pro 中的查询处理, R1 vs. Other Models, Deepseek 的性能问题, 用户对模型规格的关注` 


- **Perplexity AI 的 Focus Mode 被暂时移除**：用户讨论了 Perplexity AI 最近移除 Focus Mode 功能的情况，根据一些变更日志，关于此更改是持续性的还是临时性的说法不一。
   - 一些用户对这一变化表示沮丧，并指出这增加了指定信息来源的难度，例如需要 prompt 明确提到 Reddit。
- **关于 Pro 模式中模型使用的澄清**：有用户提出了关于 Pro 模式如何与 Claude 3.5 等模型选择交互，以及它是否利用了 R1 的推理能力的问题。有见解指出，Pro 模式可能并非端到端地使用这些模型。
   - 据指出，实际的处理过程涉及在初始搜索中使用未公开的模型，然后再传递给选定的模型（如 Claude 或 R1）以生成最终答案。
- **用户对 R1 和 Deepseek 性能的体验**：用户对比了 Perplexity 的 R1 推理能力与 Deepseek 上的表现，指出在某些条件下，Perplexity 的版本似乎能生成更可靠的输出。
   - 用户对可用模型之间的速度和质量差异表示担忧，特别是提到了不同配置的算力差异。
- **AI 应用的稳定性问题**：一些用户报告了 Perplexity 的性能缓慢和稳定性问题，特别是在使用 Android 应用以及 O3 mini 模型时。
   - 投诉集中在模型交互中的不一致和低效问题，引发了关于用户支持响应速度的讨论。
- **需要 AI 开发者进行清晰的沟通**：用户的共同心声是希望 Perplexity AI 在模型规格和更新方面（特别是涉及操作变更时）能有更高的透明度。
   - 用户建议，明确模型修改情况可以提升用户体验，并减少在与多种 AI 功能交互时的困惑。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/george-droyd-gif-12399766756846341904">George Droyd GIF - George droyd - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/monnef/status/1887231954543575135">来自 mennof (@monnef) 的推文</a>: 嘿 AI 粉丝们！🤖 刚刚用一个谜题 prompt 完成了一些 DeepSeek R1 测试（每个服务运行 3 次）！内幕消息：@perplexity_ai 的处理速度极快，而 @cursor_ai 则需要时间来思考...</li><li><a href="https://by-ai-monnef-9ff5d9c2460ae15d70e737f77eab719c6e8a4c64c2f99ca1c2.gitlab.io/2025/pplx-tech-props/">Perplexity Tech Props</a>: 未找到描述</li><li><a href="https://forms.gle/zYnhGFj3FKACoN27A">滑雪器材租赁项目问卷调查 </a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1336799036077899857)** (22 messages🔥): 

> `Tesla Robotaxi Launch, AI Skills Development Opportunities, USA vs China AI Race, Deepfake Technology from ByteDance, Trans Athlete Executive Order` 


- **Tesla Robotaxi 定于 6 月发布**：Perplexity AI 宣布 **Tesla Robotaxi** 将于 **6 月**发布，标志着自动驾驶技术的重大进步。
   - 分享的一段 YouTube 视频讨论了此次发布对 AI 和汽车行业的影响。
- **探索 AI 技能开发**：分享了一个全面的讨论串，探讨了适合从入门到精通各个级别的各种 **AI 技能开发机会**。
   - 讨论提供了关于在 AI 领域发挥个人真实潜力的见解。
- **美中 AI 竞赛详细概览**：一个关于**美中 AI 竞赛**的复杂讨论串展示了收集到的信息，并提供了验证来源。
   - 作者强调了在这一竞争格局中获取公开认可信息的挑战。
- **ByteDance 发布新的 Deepfake 技术**：关于 **ByteDance** 最新尝试的一份报告揭示了一个 Deepfake 工具的发布，这引发了关于伦理影响的各种讨论。
   - 作为此次发布的一部分，社区推测了 Deepfake 技术的潜在用途和滥用。
- **行政命令禁止跨性别运动员参加体育赛事**：最近一项禁止**跨性别运动员**参加某些体育赛事的行政命令在社区内引起了广泛辩论。
   - 成员们讨论了该命令对体育产业和民权的更广泛影响。



**提到的链接**：<a href="https://www.youtube.com/embed/mE1aAZAIX40">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1336791540982616084)** (7 messages): 

> `Perplexity API usage, Sonar Pro Reasoning devs, Image uploading limitations, Monthly cost limits and invoicing` 


- **关于每月成本限制的问题**：一位成员询问是否可以为 **Perplexity API** 的每月使用金额设置硬性限制。
   - 他们还询问发票是根据产生的费用发送，还是需要手动充值。
- **探索图像上传的变通方案**：一位新用户对他们正在构建的应用中使用 Perplexity API 表示感兴趣，但注意到目前的 API 似乎缺乏图像上传功能。
   - 他们提出了一种变通方案，即在将 Prompt 发送给 Perplexity 获取输出之前，先使用 **Claude** 进行详细描述。
- **紧急联系 Sonar Pro Reasoning 开发者的请求**：多条消息表明，由于一位成员发现了安全问题，迫切需要联系 **Sonar Pro** reasoning 开发者。
   - 另一位成员指示他们发送电子邮件至 api@perplexity.ai 寻求帮助。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1336847310683897897)** (14 messages🔥): 

> `DeepSeek 保险, Kluster 集成问题, Qwen 模型弃用, 网站宕机更新` 


- **DeepSeek 保险现在涵盖无生成 Token 的情况**：OpenRouter 现在将为没有收到生成 Token 的 DeepSeek R1 请求提供保险，确保即使上游供应商收费，用户也无需付费。
   - 标准版 DeepSeek R1 的完成率已从 **60%** 逐步提升至 **96%**。
- **Kluster 集成问题已解决**：一位用户解释了 Kluster 延迟生成 Token 的情况，导致尽管 OpenRouter 端显示超时，但仍产生了意外费用。
   - *他们发现 Kluster 在超时时未能取消请求*，但此问题现已得到解决。
- **Qwen 模型将被 Novita 弃用**：Novita 将弃用其 **Qwen/Qwen-2-72B-Instruct** 模型，OpenRouter 也将在同一时间禁用该模型。
   - 用户应确保在弃用日期前完成模型迁移。
- **OpenRouter 网站经历宕机**：由于身份验证提供商故障，OpenRouter 经历了短暂宕机，影响了网站访问但未影响 API。
   - 问题在大约 **15 分钟** 内得到解决，服务已恢复。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/deepseek/deepseek-r1">R1 - API, Providers, Stats</a>: DeepSeek R1 已上线：性能与 [OpenAI o1](/openai/o1) 相当，但是开源的且具有完全开放的推理 Token。其参数量为 671B，推理过程中激活参数为 37B...</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1:nitro">R1 (nitro) - API, Providers, Stats</a>: DeepSeek R1 已上线：性能与 [OpenAI o1](/openai/o1) 相当，但是开源的且具有完全开放的推理 Token。其参数量为 671B，推理过程中激活参数为 37B...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1337098797960003678)** (1 messages): 

> `Y CLI 开发, 终端爱好者, 聊天数据管理, MCP 客户端支持, DeepSeek-R1 集成` 


- **Y CLI 作为 OpenRouter 聊天替代方案出现**：个人项目 **Y CLI** 旨在提供网页聊天的替代方案，所有聊天数据都存储在 **单个 jsonl 文件** 中。
   - 您可以在其 [GitHub 页面](https://github.com/luohy15/y-cli) 查看该项目。
- **展示 MCP 客户端支持**：该项目包含对 MCP 客户端的支持，并在一段记录其在 macOS 上功能的 [asciinema 录像](https://asciinema.org/a/701901) 中进行了演示。
   - 该录像获得了 **4 次观看**，展示了 **xterm-256color** 和 **zsh** 的运行情况。
- **新增 DeepSeek-R1 推理支持**：**Y CLI** 的另一个功能是支持 **DeepSeek-R1** 推理内容，这在 [asciinema 录像](https://asciinema.org/a/701903) 中得到了证实。
   - 该演示同样在 macOS 上运行，获得 **2 次观看**，并支持 **xterm-256color** 和 **zsh** 终端设置。
- **GitHub 鼓励贡献**：欢迎开发者通过 [GitHub 仓库](https://github.com/luohy15/y-cli) 为 **Y CLI** 做出贡献。
   - 页面重点展示了正在进行的开发工作以及 [GitHub 概览](https://opengraph.githubassets.com/fcfbadfea6316b0b3a67649871dbdbbacd8aaa18e7894691e09a340a8a6b914d/luohy15/y-cli) 中显示的贡献者情况。
- **寻找终端粉丝**：开发者表示有兴趣在社区中寻找志同道合的 **终端爱好者**。
   - 该项目旨在吸引那些欣赏基于终端的工具和配置的人。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://asciinema.org/a/701901">y-cli mcp client</a>: https://github.com/luohy15/y-cli</li><li><a href="https://asciinema.org/a/701903">y-cli reasoning content</a>: https://github.com/luohy15/y-cli</li><li><a href="https://github.com/luohy15/y-cli">GitHub - luohy15/y-cli</a>: 通过在 GitHub 上创建账号来为 luohy15/y-cli 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1336790898289279177)** (242 messages🔥🔥): 

> `DeepInfra issues, Gemini 2.0 Flash readiness, OpenRouter authentication service, Error handling with models, Provider performance discrepancies` 


- **DeepInfra 出现故障**：用户报告 **DeepInfra** 目前由于处理延迟增加，有 50% 的概率返回响应失败。
   - 一些用户在使用 SillyTavern 等应用程序配合 DeepInfra 时，遇到了生成 token 数量为零的情况。
- **Gemini 2.0 Flash 模型集成问题**：有关 **Gemini 2.0 Flash** 模型与 tool calling 不兼容的问题引发了讨论。
   - 用户正在提交 issue，因为他们遇到了错误提示，称工具调用必须有返回结果，而该功能在其他模型上运行正常。
- **身份验证服务停机**：OpenRouter 因其由 Clerk, Inc. 提供的 **authentication service** 出现问题而经历了停机。
   - 尽管网站面临挑战，但 API 对用户保持运行，并分享了关于状态更新的信息。
- **模型错误识别**：用户报告了使用不同模型（如 **Mistral** 和 **Novita AI**）时的差异和错误。
   - 问题包括一个模型返回异常高的 token 计数，而另一个模型导致频繁的处理失败。
- **关于 Provider 性能的一般讨论**：社区正在分享关于模型和 Provider 之间性能差异的观察，包括改进建议。
   - 呼吁建立更好的机制来处理错误并优化响应，以简化用户使用 AI 模型的体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://stackoverflow.com/questions/77762483/the-caller-does-not-have-permission-when-creating-api-key">&quot;The caller does not have permission&quot; when creating API key</a>：我正在将 MakerSuite 与 Gemini 配合使用，并删除了一个 API Key。我去创建一个新的，但收到错误提示说调用者没有权限。这意味着什么以及我该如何...</li><li><a href="https://openrouter.ai/">OpenRouter</a>：LLM 的统一接口。为您的 prompt 寻找最佳模型和价格</li><li><a href="https://x.com/OfficialLoganK/status/1887178282950426914">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：@HCSolakoglu Vertex 客户往往偏向于大型企业客户，并具有协商大宗折扣等事项的灵活性。这不适用于 Gemini Developer API，每个人都支付...</li><li><a href="https://openrouter.ai/google/gemini-2.0-flash-001">Gemini Flash 2.0 - API, Providers, Stats</a>：与 Gemini Flash 1 相比，Gemini Flash 2.0 提供了显著更快的首个 token 时间 (TTFT)。通过 API 运行 Gemini Flash 2.0</li><li><a href="https://openrouter.ai/provider/google-ai-studio">Google AI Studio | OpenRouter</a>：浏览 Google AI Studio 提供的模型</li><li><a href="https://share.cleanshot.com/6rvDHCY5">CleanShot 2025-02-06 at 11 .21.37</a>：上传到 CleanShot Cloud 的截图</li><li><a href="https://openrouter.ai/docs/features/provider-routing">Provider Routing — OpenRouter | Documentation</a>：将请求路由到最佳 Provider</li><li><a href="https://share.cleanshot.com/jhf5tq3D">CleanShot 2025-02-06 at 10 .58.39</a>：上传到 CleanShot Cloud 的截图</li><li><a href="https://github.com/OpenRouterTeam/ai-sdk-provider/issues">OpenRouterTeam/ai-sdk-provider</a>：用于 Vercel AI SDK 的 OpenRouter Provider，通过 OpenRouter chat 和 completion API 支持数百个 AI 模型。 - OpenRouterTeam/ai-sdk-provider</li><li><a href="https://ai.google.dev/gemini-api/docs/api-key">未找到标题</a>：未找到描述</li><li><a href="https://status.clerk.com/">
Clerk, Inc. 状态
</a>：未找到描述</li><li><a href="https://github.com/OpenRouterTeam/ai-sdk-provider">GitHub - OpenRouterTeam/ai-sdk-provider: 用于 Vercel AI SDK 的 OpenRouter Provider，通过 OpenRouter chat 和 completion API 支持数百个 AI 模型。</a>：用于 Vercel AI SDK 的 OpenRouter Provider，通过 OpenRouter chat 和 completion API 支持数百个 AI 模型。 - OpenRouterTeam/ai-sdk-provider
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1336790225208348673)** (215 messages🔥🔥): 

> `LM Studio API error handling, Model performance inquiries, Obsidian Smart Connections integration, Updating AI models and features, Safety of downloading AI models from TheBloke` 

- **在 LM Studio 中加载模型的问题**：用户报告了在 LM Studio 中加载模型时的各种错误，包括“未知错误”和“退出代码：18446744072635812000”。建议包括提供系统规格并检查 API 以获取错误的详细信息。
   - 一位用户在连接本地模型时特别受困于状态处理（state handling），这表明需要更多关于 API 交互的指导。
- **评估特定硬件上的模型性能**：讨论了 XTX 7900 24GB 显卡运行 30GB AI 模型的适用性，并分享了关于性能能力的见解。用户强调了获得最佳结果所需的设置和配置。
   - 另一位寻求在本地运行深度学习任务的用户对 RAM 和处理能力与模型需求之间的关系表示担忧。
- **将 Obsidian 与 LM Studio 集成**：用户探讨了将 Obsidian 的 Smart Connections 扩展连接到 LM Studio 的问题，报告了各种错误以及与其他扩展的冲突。故障排除步骤包括卸载冲突插件和重建缓存。
   - 一位用户成功建立了连接，但仍面临与 API 响应中缺失必需字段相关的持续错误，并寻求进一步澄清。
- **模型可用性和安全性的更新**：在注意到某些模型在其他地方无法获取后，用户询问了从 TheBloke 下载 AI 模型的安全性和可靠性。经确认，尽管 TheBloke 在社区中的活跃度有所下降，但他的模型仍然是行业标准。
   - 鼓励用户关注社区频道，以获取模型可用性的更新以及新版本发布的可能消息。
- **LM Studio 中模型更新的频率**：LM Studio 的新更新频率受到质疑，一位用户期待 Qwen2.5-VL 模型的改进。分享的见解指出，更新通常与新模型的发布同步，而不是定期的软件更新。
   - 用户对潜在的增强功能表示兴奋，并承认有必要密切关注社区公告以获取最新功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://localhost:1234"">未找到标题</a>: 未找到描述</li><li><a href="https://model.lmstudio.ai/download/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF">在 LM Studio 中下载并运行 lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF</a>: 在你的 LM Studio 中本地使用 lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF</li><li><a href="https://imgur.com/a/WnPhj6Y">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行的梗图、有趣的 GIF、感人的故事、病毒式视频等来提振你的精神...</li><li><a href="https://huggingface.co/showlab/ShowUI-2B">showlab/ShowUI-2B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.apple.com/shop/buy-mac/macbook-pro/14-inch-m4-max">购买配备 M4 Max 的 14 英寸 MacBook Pro</a>: 探索搭载 M4 系列芯片的 MacBook Pro 笔记本电脑，专为 Apple Intelligence 打造。折抵符合条件的 Mac 即可获得折抵金额。立即购买。</li><li><a href="https://huggingface.co/lmstudio-community/MiniCPM-o-2_6-GGUF">lmstudio-community/MiniCPM-o-2_6-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/UI-TARS-2B-SFT-GGUF">lmstudio-community/UI-TARS-2B-SFT-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#vulkan">llama.cpp/docs/build.md at master · ggerganov/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户，为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/kth8/llama-server-vulkan">GitHub - kth8/llama-server-vulkan: 使用 Vulkan 运行 llama.cpp 服务端</a>: 使用 Vulkan 运行 llama.cpp 服务端。通过在 GitHub 上创建账户，为 kth8/llama-server-vulkan 的开发做出贡献。</li><li><a href="https://llm-stats.com/models/compare/o3-mini-vs-deepseek-r1">o3-mini vs DeepSeek-R1</a>: 深入的 o3-mini 与 DeepSeek-R1 对比：2025 年最新的基准测试、定价、上下文窗口、性能指标和技术规格。</li><li><a href="https://www.cloudflarestatus.com/">Cloudflare 状态</a>: 未找到描述</li><li><a href="https://lmstudio.ai/">LM Studio - 发现、下载并运行本地 LLM</a>: 在你的电脑上本地运行 Llama, Mistral, Phi-3。</li><li><a href="https://youtu.be/7xTGNNLPyMI?t=11967">深入探讨 ChatGPT 等 LLM</a>: 这是一个面向普通观众的深度探讨，介绍了驱动 ChatGPT 及其相关产品的 LLM AI 技术。它涵盖了完整的训练...</li><li><a href="https://github.com/stackblitz-labs/bolt.diy?tab=readme-ov-file#requested-additions">GitHub - stackblitz-labs/bolt.diy: 使用你想要的任何 LLM 来提示、运行、编辑和部署全栈 Web 应用程序！</a>: 使用你想要的任何 LLM 来提示、运行、编辑和部署全栈 Web 应用程序！ - stackblitz-labs/bolt.diy</li><li><a href="https://github.com/stackblitz-labs/bolt.diy">GitHub - stackblitz-labs/bolt.diy: 使用你想要的任何 LLM 来提示、运行、编辑和部署全栈 Web 应用程序！</a>: 使用你想要的任何 LLM 来提示、运行、编辑和部署全栈 Web 应用程序！ - stackblitz-labs/bolt.diy</li><li><a href="https://github.com/ggerganov/llama.cpp/">GitHub - ggerganov/llama.cpp: C/C++ 中的 LLM 推理</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户，为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1336812216501932095)** (23 messages🔥): 

> `DDR5 6000 EXPO 性能, LMS 硬件配置, 内存测试工具, PCIe 3.0 上的多 GPU 设置` 


- **DDR5 6000 EXPO 时序保守**：一位成员指出，其 DDR5 6000 的 **EXPO 时序** 可能 **非常保守**，并注意到推理期间的最大内存带宽峰值为 **72**。
   - 他们成功完成了 **4 轮 memtest86** 以确保稳定性，尽管另一位成员建议尝试使用 TestMem5 进行更严格的评估。
- **LMS 硬件配置困扰**：另一位用户提出了关于托管 LMS 0.3.9 的硬件设置问题，提到他们拥有 **32 核** CPU 但没有 GPU，并收到了关于内存使用设置的建议。
   - 建议包括在 **Developer mode** 下运行，并开启将整个模型保留在 RAM 中的选项，同时建议调整线程使用量以获得更好的速度。
- **探索多 GPU 能力**：一位新成员询问在 **PCIe 3.0 16x** 上运行多个 **3090s** 的情况，寻求社区中其他人的经验。
   - 讨论围绕这种设置是否仍然可行展开，另一位用户询问了能够有效运行大型模型的设置示例。
- **推理速度考量**：有人担心将 RAM 运行在 **7600** 是否会带来明显的推理速度变化，但初步对比显示仅有 **15% 的提升**。
   - 成员们注意到，大量的 Prompt 可能会影响平均速度，特别是纯文本响应与 Python 代码生成的响应之间的对比。
- **理解 GPU 加速**：针对 **DeepSeek R1 Distill Qwen 7B 模型** 的 GPU 加速进行了咨询，对于哪些模型支持 GPU 使用存在一些困惑。
   - 讨论澄清了只有像 **Llama** 这样特定的模型已知支持加速，而 DeepSeek 模型仍存在一些不确定性。



**提到的链接**：<a href="https://github.com/CoolCmd/TestMem5">GitHub - CoolCmd/TestMem5: TestMem5 - PC RAM stress test</a>：TestMem5 - PC RAM 压力测试。通过创建账号为 CoolCmd/TestMem5 的开发做出贡献。

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1336865347130626182)** (97 条消息🔥🔥): 

> `Home Assistant MCP 客户端/服务器, MCP Server 使用情况, Goose MCP 客户端, Claude 中的图像显示, MCP Server 配置` 


- **具备新功能的 Home Assistant MCP 客户端**：一位用户宣布发布了支持 **MCP 客户端/服务器** 的 **Home Assistant**，称其功能已完备但仍需一些“亮点”。他们还计划加入一个动画化的对话头像，以提升用户交互体验。
   - 该项目仍在进行中，开发者正在平衡有偿工作与开发进度。
- **对 MCP Server 使用统计数据的关注**：一位用户对有多少人在使用 **Home Assistant** MCP 表示好奇，并提到了将其与 **Claude** 等其他工具连接的努力。
   - 这引发了关于不同 MCP 客户端功能及其更广泛用途的讨论。
- **关于 Goose MCP 客户端的讨论**：用户分享了使用 **Goose MCP 客户端** 的经验，指出其在测试环境中的有效性和当前用例。
   - 一位用户还提到了一项待处理的 Pull Request，旨在改进其日志记录功能，凸显了社区内的紧密协作。
- **Claude Desktop 图像显示的挑战**：一位用户询问如何在 **Claude Desktop** 上将图像作为工具结果显示，并指出了尝试操作时遇到的输入错误。
   - 他们推测将图像结果转换为嵌入式资源（embedded resources）可能是一个解决方案。
- **MCP Server 配置见解**：随后展开了关于设计更好的 **MCP Server 配置** 的讨论，用户分享了如何有效管理多个服务器的想法。
   - 有人建议使用带有 **bridge** 的多路复用（multiplexer）方法来简化服务器管理，其他人也分享了他们的 MCP 客户端开发计划。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com">GitHub · 在单一协作平台上构建和交付软件</a>：加入全球应用最广泛的 AI 驱动型开发者平台，数百万开发者、企业和最大的开源社区在此构建推动人类进步的软件。</li><li><a href="https://github.com/splendasucks/webperfect-mcp-server">GitHub - splendasucks/webperfect-mcp-server: webperfect-mcp-server</a>：webperfect-mcp-server。通过在 GitHub 上创建账号为 splendasucks/webperfect-mcp-server 的开发做出贡献。</li><li><a href="https://github.com/block/goose/actions/runs/13058183345/job/36804119892?pr=947">fix(anthropic): 在使用量统计日志中包含缓存的 Token · block/goose@162c4c5</a>：一个超越代码建议的开源、可扩展 AI Agent - 可在任何 LLM 上安装、执行、编辑和测试 - fix(anthropic): 在使用量统计日志中包含缓存的 Token · block/goose@162c4c5</li><li><a href="https://github.com/block/goose/pull/947">fix(anthropic): 由 evalstate 在 Pull Request #947 中将缓存的 Token 包含在使用量统计中 · block/goose</a>：cache_creation_input_tokens 和 cache_read_input_tokens 未被添加到 goose.log 记录的使用总量中。此修复将这些类别包含在 &amp;quot;input tokens&amp;quot; 的计算中...</li><li><a href="https://github.com/met4citizen/TalkingHead">GitHub - met4citizen/TalkingHead: Talking Head (3D): 一个用于使用 Ready Player Me 全身 3D 化身进行实时口型同步的 JavaScript 类。</a>：Talking Head (3D)：一个用于使用 Ready Player Me 全身 3D 化身进行实时口型同步的 JavaScript 类。 - met4citizen/TalkingHead
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1336803791445426307)** (54 条消息🔥): 

> `PulseMCP Use Cases, MCP Servers, Claude for Research, Web Research Tools, Markdown in Discord` 


- **PulseMCP Use Cases 正式发布**：宣布了一个新的实用的 **PulseMCP Use Cases** 展示，包含了如何有效使用各种客户端应用和服务器的详细说明及视频。
   - 首批亮点包括使用 **Gemini voice**、**Claude** 和 **Cline** 分别进行 Notion 管理、Figma 设计转换以及创建知识图谱。
- **Claude 成功复刻 ChatGPT DeepResearch**：**Claude** 展示了利用特定的 MCP 服务器（如 *mzxrai 的 web research MCP* 和 *Brave web search MCP*）高效复刻 **ChatGPT DeepResearch** 的能力。
   - 一位用户指出，在时间充裕的情况下，Claude 可以处理多达 **100 篇文章**，突显了该工具在给定适当输入时的灵活性。
- **Web 搜索问题及解决方案**：讨论揭示了 **Google 搜索** 触发机器人检测的挑战，并提供了替代方案，例如使用 **SearXNG** 进行无验证码搜索。
   - 推荐使用修改版的 **chromedriver** 和 **puppeteer** 等工具来克服这些问题。
- **移动设备上的 MCP 支持能力**：关于移动端 MCP 客户端的咨询表明，**Sage** 支持 iPhone，而 **Android** 用户可能需要使用 **LibreChat** 或 **MCP-Bridge** 等 Web 客户端。
   - 这反映了用户对于在桌面应用之外访问 MCP 功能的持续兴趣。
- **Discord 中的 Markdown 渲染**：围绕 Discord 的 **Markdown 渲染** 能力展开了讨论，指出该功能自去年起已实现，用户对其功能感到惊喜。
   - 成员们分享了关于使用 **Markdown** 样式的非正式闲聊，体现了轻松的社区互动。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.pulsemcp.com/use-cases">Community use-cases of MCP in-action | PulseMCP</a>：探索社区投入使用 Model Context Protocol (MCP) 的所有方式。</li><li><a href="https://x.com/tadasayy/status/1887253558749471034">来自 Tadas Antanavicius (@tadasayy) 的推文</a>：🎉 宣布在 PulseMCP 上发布 Use Cases（关注 @pulsemcp 以获取最新动态）！自 @Anthropic 发布以来，已经构建了大量出色的 MCP 服务器和客户端，我们构建了一个资源库来...
</li>
</ul>

</div>

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1336862981111152702)** (112 条消息🔥🔥): 

> `Gemini 2.0 性能、DeepSpeed 与 Hugging Face、AI 立法影响、澳大利亚互联网基础设施、开源 AI 模型` 


- **Gemini 2.0 表现优于竞争对手**：频道内的讨论提到 **Gemini 2.0 Pro** 在创建 SVG 等任务中表现出色，特别是与 **o3-mini** 和 **R1** 相比。
   - 几位成员注意到其在 SQL 查询方面有更强的表现，表明 Google 在 **Gemini Flash 2.0** 上取得了重大进展。
- **DeepSpeed 与 Dataloader 的困惑**：一位用户对在使用 DeepSpeed 的自动 batch size 配置时，是否仍需在 Dataloader 中手动指定 **batch_size** 表示困惑。
   - 另一位成员建议将 DeepSpeed 标签集成到 Dataloader 中进行优化，并暗示了针对不同节点的潜在性能调整。
- **对 AI 立法的担忧**：一位用户分享了对澳大利亚新立法的担忧，认为其旨在限制言论和思想自由，对社会有重大影响。
   - 这被置于一种更广泛的情绪中，即此类法律可能导致许多讨论变得非法，从而扼杀开放的对话和探究。
- **澳大利亚互联网基础设施的困境**：一位参与者感叹，尽管投入了大量资金，澳大利亚的网速仍然很慢，有人报告家庭连接速度低至 **3 Mbps**。
   - 讨论强调了基础设施决策的失败，提到了几十年前选择 **铜缆而非光纤** 的糟糕决定。
- **开源 AI 模型的未来**：对话涉及了对反对 **开源 AI 模型** 趋势的担忧，讨论集中在为了支持专有系统而限制具有竞争力和创新性的模型。
   - 成员们对通过宣布某些关于技术和言论自由的讨论为非法来控制 AI 相关舆论的企图感到沮丧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://simonwillison.net/2025/Feb/5/gemini-2/">Gemini 2.0 现已向所有人开放</a>：今天发布的 Gemini 2.0 重大更新：- **Gemini 2.0 Pro (Experimental)** 是 Google 迄今为止“在编程性能和复杂提示词处理方面最强的模型”——目前作为免费预览版提供。-...</li><li><a href="https://x.com/NationFirstAust/status/1887361530955755800">George Christensen (@NationFirstAust) 的推文</a>：他们正盯着你的言论、你的思想、你的信仰。1/24</li><li><a href="https://techcrunch.com/2025/02/05/researchers-created-an-open-rival-to-openais-o1-reasoning-model-for-under-50/">研究人员以不到 50 美元的成本创建了 OpenAI o1 “推理”模型的开源对手 | TechCrunch</a>：斯坦福大学和华盛顿大学的 AI 研究人员能够利用不到 50 美元的云计算额度训练出一个 AI “推理”模型。
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1336828307668471881)** (10 条消息🔥): 

> `Harmonic Loss 论文、VideoJAM 讨论、欧盟讨论时间、DeepSeek 托管` 


- **对 Harmonic Loss 的质疑**：一些成员对 **Harmonic Loss 论文** 表示怀疑，指出其“做得有点仓促”，尽管有理论优势但缺乏性能提升。
   - 另一位成员提到，该论文的 GitHub 仓库比论文本身“包含更多信息”。
- **对 VideoJAM 论文解读的期待**：一位成员宣布即将在 **欧洲时间下午 6 点** 进行 [VideoJAM 论文](https://hila-chefer.github.io/videojam-paper.github.io/) 的解读。
   - 这个时间可能对美国成员有利，但对处于欧盟时区的成员来说存在挑战。
- **讨论中的欧盟时间挑战**：有人对每日讨论中“糟糕的欧盟时间”表示担忧，建议下周将讨论移至 **欧洲时间下午 6 点**。
   - 另一位成员确认了时差，指出这比美国东部/西部海岸早 **6-9 小时**。



**提到的链接**：<a href="https://hila-chefer.github.io/videojam-paper.github.io/">VideoJAM</a>：VideoJAM: 用于增强视频模型运动生成的联合外观-运动表示法

  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1336805898257043466)** (10 条消息🔥): 

> `Gemini 2.0 Flash, Flash-lite 问题, S1 推理模型, 推理扩展 (Inference scaling) 见解, OpenAI 扩展定律 (scaling laws)` 


- **Gemini 2.0 Flash 表现亮眼**：一位用户报告通过 [LlamaIndex](https://openrouter.ai/google/gemini-2.0-flash-001) 试用了新的 **Gemini 2.0 Flash** 模型，指出其速度惊人，尽管没有 **Groq** 那么快。
   - 这一 OpenRouter 类别的新成员在急于测试其能力的开发者中引起了轰动。
- **Flash-lite 在结构化输出方面表现不佳**：另一位用户报告称 **Flash-lite** 模型在返回有效的结构化输出时遇到困难，经常生成**无效的 JSON** 格式。
   - 他们认为这令人失望，并建议该模型可能不适合需要高可靠性输出的任务。
- **S1 成为低成本推理替代方案**：最近的一篇博客文章讨论了 **S1 推理模型**，它展示了与 **OpenAI o1** 等模型相当的性能，但可以在基础机器上运行，并强调其训练成本低于 **$50**。
   - S1 模型及其工具通过从 **Gemini 2.0** 进行蒸馏 (distillation) 开发而成，现已在 [GitHub](https://github.com/simplescaling/s1) 上可用。
- **推理扩展 (Inference Scaling) 的见解**：对话揭示了关于**推理扩展**的见解，声称更长的思考时间可以提升 LLM 的性能；然而，实现更长思考过程的方法受到了质疑。
   - **s1 论文**通过图表阐明了这一点，引发了关于如何有效实施此类策略的讨论。
- **关于 Flash 能力的疑问**：社区成员对最近发布的模型（包括 **2.0 pro 实验版**）的能力提出了疑问，并询问是否有测试用的 Prompt。
   - 新发布模型的价值引发了辩论，参考了现有模型**蒸馏 (distilled)** 版本的潜力和过往经验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/google/gemini-2.0-flash-001">Gemini Flash 2.0 - API, Providers, Stats</a>: Gemini Flash 2.0 与 [Gemini Flash 1.5] 相比，首个 Token 响应时间 (TTFT) 显著加快。通过 API 运行 Gemini Flash 2.0</li><li><a href="https://openrouter.ai/google/gemini-2.0-flash-lite-preview-02-05:free/api">Google: Gemini Flash Lite 2.0 Preview (free) – Run with an API</a>: Google: Gemini Flash Lite 2.0 预览版 (免费) 的示例代码和 API - Gemini Flash Lite 2.0 与 [Gemini Flash 1.5](google/gemini-flash... 相比，首个 Token 响应时间 (TTFT) 显著加快</li><li><a href="https://x.com/osanseviero/status/1887247587776069957">Tweet from Omar Sanseviero (@osanseviero)</a>: 嘿 r/LocalLLaMA 👋 我们正在憋大招 🫡 Gemma 冲冲冲</li><li><a href="https://openrouter.ai/rankings">LLM Rankings | OpenRouter</a>: 根据各应用的使用情况对语言模型进行排名和分析</li><li><a href="https://timkellogg.me/blog/2025/02/03/s1">S1: The $6 R1 Competitor?</a>: 暂无描述</li><li><a href="https://techcrunch.com/2025/02/05/researchers-created-an-open-rival-to-openais-o1-reasoning-model-for-under-50/">Researchers created an open rival to OpenAI&#039;s o1 &#039;reasoning&#039; model for under $50 | TechCrunch</a>: 据报道，斯坦福大学和华盛顿大学的 AI 研究人员能够利用不到 50 美元的云计算额度训练出一个 AI “推理”模型。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1336866947928821780)** (22 条消息🔥): 

> `LLM 研究协作、Deepspeed 与 Hugging Face、LLM 基准测试、微调中的 Weight Decay、RWKV 架构开发` 


- **寻求 LLM 项目协作**：Adobe 的一位高级 ML Engineer 表示有兴趣探索与 **LLM agents 相关的研究项目**，并邀请志同道合的人士进行协作。
   - *期待一些令人兴奋的讨论！*
- **Deepspeed 使用说明**：一位用户询问在配置中使用 **Deepspeed** 自动批次大小（auto batch sizing）时，是否仍需为 data loader 指定 **batch_size**。
   - 另一位成员指出，data loader 仍然需要指定 **batch_size**。
- **提出新的主题泛化基准测试**：一位成员分享了一个 GitHub 仓库链接，讨论了一个 **thematic generalization benchmark**（主题泛化基准测试），旨在评估 LLM 从正例和反例中推断类别的能力。
   - 他们询问该基准测试是否可能与 **SAE autointerp** 的性能相关。
- **Weight Decay 在微调中的作用**：关于在微调期间使用 **weight decay** 是否合适展开了讨论，一位成员确认了其普遍用法。
   - 另一位用户提到了 **OLMo2 论文**中的一个有趣观点，即在预训练期间不对 embedding 参数应用 weight decay。
- **RWKV 团队开发新架构**：提到 RWKV 团队正在开发一些令人兴奋的新架构，表明了其在模型开发方面的积极态度。
   - 一位用户分享了他们在扩展和资源密集型设计方面的困扰，并邀请就潜在协作进行进一步讨论。



**提及链接**：<a href="https://github.com/lechmazur/generalization">GitHub - lechmazur/generalization: Thematic Generalization Benchmark: 衡量各种 LLM 从一小组正例和反例中推断狭窄或特定“主题”（类别/规则）的有效性，然后在收集的误导性候选项中检测哪个项目真正符合该主题。</a>

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1336791726290894992)** (92 条消息🔥🔥): 

> `Multi Token Prediction Inference, Independent Research in AI/ML, A/B Testing and Reward Modeling, Quadratic Fitting for Parameter Estimation, DeepSeek MTP Implementation` 


- **理解用于推理的 Multi Token Prediction**: 讨论了 Multi Token Prediction (MTP) 在推理过程中如何工作，特别是关于初始 token 的生成以及如何利用 embedding 来提高速度。
   - 讨论强调了 MTP 是生成 token 的一种高效方式，并分享了相关的实现资源，包括一个 [GitHub pull request](https://github.com/vllm-project/vllm/pull/12755)。
- **独立 AI 研究的现实领域**: 一位独立研究员询问在没有大量资金支持且计算资源受限的情况下，AI/ML 领域有哪些可行的探索方向。
   - 建议包括寻找资助机会和加入协作研究小组，并提到了 [Google TRC](https://sites.research.google/trc/about/) 等项目。
- **A/B Testing 与参数估计的挑战**: 对话转向了使用 A/B Testing 来确定最优采样器 (sampler) 参数的可行性，并对依赖传统的二次拟合 (quadratic fittings) 表示担忧。
   - 建议使用 Reward Model 可以更好地捕捉用户偏好，同时也承认了 Bandit 算法会增加该方法的复杂性。
- **二次拟合 vs. 任意函数学习**: 关于将二次函数拟合到 A/B 测试数据的讨论引向了对 Reward Modeling 概念的探索，这象征着改进估计过程的一种可能途径。
   - 参与者指出了仅进行二次拟合的局限性，并讨论了使用成对偏好模型 (pairwise preference models) 来优化采样器参数等替代方法。
- **DeepSeek MTP 令人惊讶的结果**: 分享了关于 DeepSeek 模型性能的见解，强调了其 MTP 的实现以及在用户评价的 token 生成方面的相对成功。
   - 参与者对该模型的有效性表示好奇，并分享了来自其底层方法论实践经验的资源和实际成果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.03387">LIMO: Less is More for Reasoning</a>: 我们提出了一项基础性发现，挑战了我们对大型语言模型中复杂推理如何涌现的理解。虽然传统观点认为复杂的推理任务需要...</li><li><a href="https://sites.research.google/trc/about/">TPU Research Cloud - 关于</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k">bespokelabs/Bespoke-Stratos-17k · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B#distilled-model-evaluation">deepseek-ai/DeepSeek-R1-Distill-Qwen-32B · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/auth/endorse?x=XGKX4E">arXiv 用户登录</a>: 未找到描述</li><li><a href="https://github.com/vllm-project/vllm/pull/12755">[Model][Speculative Decoding] DeepSeek MTP spec decode by luccafong · Pull Request #12755 · vllm-project/vllm</a>: 实现 DeepSeek MTP: #12181 以支持用于 next n prediction 的 DeepSeek MTP 层。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1336893345737478154)** (1 messages): 

> `MATS cohort applications, Mechanistic Interpretability Research, Mentoring in AI research` 


- **夏季 MATS 批次申请现已开放！**：**MATS 8.0** 批次的申请现已开放，截止日期为 **2月28日**。感兴趣的候选人可以点击 [此处](https://tinyurl.com/neel-mats-app) 申请参加带薪全职 Mechanistic Interpretability 研究。
   - 该计划欢迎各种经验水平的申请人，之前的学员已在该领域贡献了 **10 篇顶级会议论文**。
- **获取 MATS FAQ 和录取流程**：有关 MATS 录取流程和 FAQ 的详细信息可以在链接的 [文档](https://docs.google.com/document/?usp=docs_web) 中找到。鼓励潜在申请人登录并查看全面的指南。
   - 该资源旨在为有兴趣申请导师计划的人员澄清流程。
- **导师指导成功案例**：多年来，Neel 指导了 **40 多名学员**，为 Mechanistic Interpretability 研究做出了重大贡献。这一经验展示了该计划在培养该领域人才方面的有效性。
   - Neel 对学员的成功表示自豪，特别提到了他们对各大顶级会议的贡献，并强调你不需要在大型实验室也能在这一研究领域取得卓越成就。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tinyurl.com/neel-mats-app">Neel Nanda MATS 8.0 Stream - Admissions Procedure + FAQ</a>: Neel Nanda MATS 8.0 - 录取流程 + FAQ。在此申请。为什么我可能想申请？筛选问题。我的申请应该是什么样的？执行摘要格式。有用资源。什么研究项目...</li><li><a href="https://x.com/NeelNanda5/status/1887274059408548208">Neel Nanda (@NeelNanda5) 的推文</a>: 我的 MATS 方向申请已开放，我尝试在这里教授如何进行出色的 Mechanistic Interpretability 研究。2月28日截止！我热爱指导，已经指导了 40 多名学员，他们为该领域做出了宝贵贡献，包括...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1337056905717092394)** (7 messages): 

> `cons@64, majority voting, eval configuration in YAML` 


- **关于 cons@64 术语的澄清**：成员们讨论了 **cons@64** 的含义，推测它是指对 64 个输出进行 Majority Voting，还是 LLM 利用这些输出生成答案。
   - 在此背景下，*Consensus* 和 *Majority Voting* 被认为是可互换的术语，一位成员分享了 OpenAI 讨论该话题的链接。
- **关于评测 YAML 配置的专家咨询**：一位成员询问是否可以在评测 .yaml 文件中自动指定 *apply chat template* 或 *fewshot-as-multiturn*。
   - 他们想知道是否应该在 **utils.py** 中编写代码，以便将 **mgsm_chat** 功能整合到各种评测中。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1337066904510271539)** (1 messages): 

> `Sequence parallelism implementation, Model parallelism size issues, AttributeError in Megatron library, Training crash log` 


- **Sequence Parallelism 与 MP size 2 的适配难题**：一位用户在尝试启用 Sequence Parallelism 且模型并行规模 (MP) 为 **2** 时遇到了 **训练崩溃**，并引用了 Megatron 库中的一个特定错误。
   - 错误回溯指向 `AttributeError: module 'megatron.mpu' has no attribute 'get_fp32_allreduce'`，表明实现的函数中存在潜在问题。
- **对文档和 Flag 使用的困惑**：用户对文档表示困惑，文档建议只需开启一个 Flag，Sequence Parallelism 就应该在 MP 大于 **1** 的情况下工作。
   - 这种差异引发了关于文档不准确还是实现中存在现有问题的疑问。



**提及的链接**：<a href="https://wandb.ai/aflah/hubble-speed-testing/runs/oawmmmpd/overview">aflah</a>: Weights & Biases，机器学习开发者工具

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1336795387784335361)** (100 条消息🔥🔥): 

> `Deep Research 反馈，AI 抵制与加密货币，信托中的 Purpose AI Agent，新 AI 模型与训练技术，微调方法` 


- **Deep Research 获得好评**：成员们分享了对 OpenAI 的 Deep Research 的热情，强调其能够高效提取相关联系和来源，增强了他们的认知带宽。
   - 一位用户指出，该模型具有探索冷门在线社区并收集意想不到数据的能力。
- **AI 抵制与过去的技术争议挂钩**：讨论中提到公众对 AI 的不信任源于过去对加密货币和 NFT 的负面经历，一些成员认为这种情绪正在影响对 AI 技术的看法。
   - 批评者强调了对 AI 训练数据未经许可及其对劳动力市场颠覆性影响的担忧。
- **探索 Purpose AI Agent**：一位用户概述了他们在法律信托框架内开发目标导向型 AI Agent 的雄心，旨在开拓围绕 AI 人格（Personhood）的法律论述。
   - 反馈集中在涉及的工程复杂性上，包括集成财务管理功能，同时强调了定制软件解决方案的潜力。
- **AI 模型合并与微调的进展**：对话包括合并不同 AI 模型的策略，成员们分享了关于改进模型指令微调（Instruction Tuning）和推理性能的见解。
   - 讨论了各种微调方法，探索了在 AI 训练中加入创新技术以增强模型性能的好处。
- **对推理链（Reasoning Trace）可访问性的担忧**：成员们对 DeepSeek 等模型推理链的可用性表示怀疑，并担心 OpenAI 可能会利用这些推理链而不提供 API 访问权限。
   - 对话强调了大 AI 公司限制访问高级功能和信息的趋势，这可能是为了保护专有技术。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/teknium/Llama-3.1-AlternateTokenizer">teknium/Llama-3.1-AlternateTokenizer · Hugging Face</a>: 未找到描述</li><li><a href="https://rentry.org/vwa65v85">Why Everyone Is Suddenly Mad at AI</a>: AI 抵制：又一场技术炒作后的宿醉（加密货币是罪魁祸首吗？）（OpenAI Deep research 演示提示词：写一篇关于为什么可能会出现对 AI 的抵制，以及这是否与 NFT/加密货币的曝光度有关的文章...</li><li><a href="https://huggingface.co/minpeter/Llama-3.2-1B-AlternateTokenizer-chatml/commit/f2528b7382f529d36a224ff04c5a73af3acd4e9c">Upload folder using huggingface_hub · minpeter/Llama-3.2-1B-AlternateTokenizer-chatml at f2528b7</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=9FuNtfsnRNo">I Built the Ultimate Team of AI Agents in n8n With No Code (Free Template)</a>: 📌 加入我的免费 Skool 社区以获取视频中展示的工作流！👇https://www.skool.com/ai-automation-society/about🌟 如果你...</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-GRPO.ipynb)">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-r1-on-your-own-local-device)">Unsloth Documentation</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth?sort_models=created&search_models=1m#models).">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models)">Unsloth Documentation</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1337134893699694775)** (1 条消息): 

> `DeepSeek-R1 训练循环，Reward loss 与 KL loss 的敏感度，小型指令模型的陷阱，模型大小考量，超参数重要性` 


- **DeepSeek-R1 训练循环见解**：一位用户询问了模型对 **Reward loss** 和 **KL loss** 之间权重比例的**敏感度**，质疑其作为超参数的重要性。
   - 他们寻求关于哪些超参数在优化模型性能方面最具重要性的见解。
- **对小型指令模型的担忧**：该用户表达了对从 **Qwen2.5 3B** 等较小指令模型开始（而非较大的基础模型）可能存在的**陷阱**的兴趣。
   - 他们强调希望找到既能提供可靠测试和开发，又能兼顾资源管理的最小模型。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1337077333068222617)** (1 messages): 

> `Synthetic data generation, Seed-based approaches, Magpie output issues, Self-instruct alternatives, Awesome-LLM-Synthetic-Data resource` 


- **探索合成数据生成技术**：一名成员正在寻找关于**合成数据生成**的论文，特别是关注类似于 **Self-Instruct** 的新型**基于 Seed 的方法**。
   - 他们提到在非英语语言进行实验时，使用 **Magpie** 输出面临挑战。
- **Magpie 输出问题**：该成员对使用 Seed 系统提示词时 **Magpie** 的输出质量表示沮丧，认为结果不尽如人意。
   - *WizardLM 对此没有帮助*，因为他们需要有效的 Seed 指令才能继续。
- **发现 LLM 合成数据资源**：该成员发现了一个名为 [Awesome-LLM-Synthetic-Data](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data) 的 GitHub 仓库，该仓库提供了一份关于**基于 LLM 的合成数据生成**的资源列表。
   - 该资源旨在帮助理解该领域的各种技术和方法论，特别是针对较新的模型。



**Link mentioned**: <a href="https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data?tab=readme-ov-file,">GitHub - wasiahmad/Awesome-LLM-Synthetic-Data: A reading list on LLM based Synthetic Data Generation 🔥</a>: A reading list on LLM based Synthetic Data Generation 🔥 - wasiahmad/Awesome-LLM-Synthetic-Data

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1336792312008671306)** (3 messages): 

> `Deep Dive into LLMs, Mina's zkML Library` 


- **通过 Deep Dive into LLMs 探索 AI**：一个 [YouTube 视频标题为 
- **Mina 的 zkML 库开发者指南**：[Mina 博客](https://minaprotocol.com/blog/minas-zkml-library-developer-guide)上的一篇文章讨论了 **zkML** 库的发布，该库使 AI 模型能够在保持**完全隐私**和**可验证性**的同时在链上运行。本指南为希望利用 Mina 的 **zkML** 能力开发去中心化应用的开发者提供了参考。



**Link mentioned**: <a href="https://m.youtube.com/watch?v=7xTGNNLPyMI&pp=ygUgRGVlcCBkaXZlIGludG8gbGxtcyBsaWtlIGNoYXRncHQ%3D">Deep Dive into LLMs like ChatGPT</a>: 这是一个面向普通观众的深度解析，涵盖了驱动 ChatGPT 及相关产品的大语言模型（LLM）AI 技术。它涵盖了完整的训练过程...

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1337077333068222617)** (1 messages): 

> `Synthetic Data Generation, Self-instruct, Magpie, WizardLM, Awesome LLM Synthetic Data` 


- **寻求基于 Seed 的合成数据解决方案**：一位用户正在寻找关于**合成数据生成**的论文或方向，特别是像 **Self-instruct** 这样基于 Seed 的方法，以改进他们在 **Magpie** 上的结果。
   - 他们指出使用非英语语言时的输出并不理想，并且正在寻求优于 **WizardLM** 所提供的 Seed 指令。
- **发现合成数据 GitHub 资源**：用户发现了一个名为 **Awesome-LLM-Synthetic-Data** 的 [GitHub 仓库](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data)，该仓库提供了一份关于**基于 LLM 的合成数据生成**的阅读清单。
   - 该仓库强调了各种资源，现在是用户探索其数据生成需求替代方案的一个潜在途径。



**Link mentioned**: <a href="https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data?tab=readme-ov-file,">GitHub - wasiahmad/Awesome-LLM-Synthetic-Data: A reading list on LLM based Synthetic Data Generation 🔥</a>: A reading list on LLM based Synthetic Data Generation 🔥 - wasiahmad/Awesome-LLM-Synthetic-Data

  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1336902458001199146)** (39 条消息🔥): 

> `John Schulman 离开 Anthropic，Hibiki 语音转语音翻译模型，Le Chat AI 助手，GitHub Copilot Agent 模式，OpenAI 更新思维链` 


- **John Schulman 离开 Anthropic**：顶级 AI 研究员及 OpenAI 联合创始人 **John Schulman** 在入职约五个月后离开了 **Anthropic**，引发了对其下一步动向的猜测 [链接](https://www.bloomberg.com/news/articles/2025-02-06/openai-co-founder-john-schulman-leaves-rival-firm-anthropic?srnd=undefined)。
   - 关于他下一步去向的推测包括在 **Deepseek** 和 **AI2** 等机构任职的可能性。
- **Hibiki 彻底改变翻译**：来自 **Kyutai Labs** 的全新 **Hibiki** 模型支持**同声语音转语音翻译**，并能根据内容调整语速 [链接](https://x.com/kyutai_labs/status/1887495488997404732)。
   - 据报道，它在**质量**、**自然度**和**说话人相似度**方面优于之前的系统，接近人类译员的能力。
- **Le Chat AI 正式发布**：**MistralAI** 推出了 **Le Chat**，定位为工作和生活的全能 AI 助手，现已在网页端和移动端上线 [链接](https://x.com/MistralAI/status/1887517520040448510)。
   - 这一新工具旨在通过 AI 提升生产力和个人辅助能力。
- **GitHub Copilot 开启 Agent 模式**：GitHub 宣布为 **Copilot** 引入 **Agent 模式**，旨在更有效地辅助开发者编程 [链接](https://github.blog/news-insights/product-news/github-copilot-the-agent-awakens/)。
   - 此次更新还包括 **Copilot Edits** 的全面开放（General Availability），进一步提升了开发者体验。
- **OpenAI 增强思维链**：OpenAI 更新了 **o3-mini** 模型中的 **Chain of Thought**（思维链）机制，旨在优化用户体验，但他们强调这些并非原始的 CoT [链接](https://x.com/OpenAI/status/1887616278661112259)。
   - 讨论表明，这可能会带来**更好的蒸馏输出**，尽管某些 Token 的重要性仍存争议。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenAI/status/1887616278661112259">来自 OpenAI (@OpenAI) 的推文</a>：为免费和付费用户更新了 OpenAI o3-mini 中的思维链，并为付费用户更新了 o3-mini-high。</li><li><a href="https://github.blog/news-insights/product-news/github-copilot-the-agent-awakens/">GitHub Copilot: Agent 觉醒</a>：在 VS Code 中为 GitHub Copilot 引入 Agent 模式，宣布 Copilot Edits 全面开放，并首次展示我们的 SWE Agent。</li><li><a href="https://x.com/TheXeophon/status/1887343884662935894">来自 Xeophon (@TheXeophon) 的推文</a>：@apples_jimmy 如果 John 加入 Ai2，我会非常兴奋。</li><li><a href="https://x.com/kyutai_labs/status/1887495488997404732">来自 kyutai (@kyutai_labs) 的推文</a>：认识 Hibiki，我们的同声语音转语音翻译模型，目前支持 🇫🇷➡️🇬🇧。Hibiki 实时生成输入语音的语音和文本翻译，同时保留其...</li><li><a href="https://fxtwitter.com/polynoamial/status/1887621287616651429">来自 Noam Brown (@polynoamial) 的推文</a>：当我们在 o1-preview 发布前向人们简要介绍 🍓 时，实时看到 CoT 通常是他们的“顿悟”时刻，让他们明白这将是一件大事。这些不是原生的...</li><li><a href="https://x.com/shiringhaffary/status/1887340283916140922?s=61">来自 Shirin Ghaffary (@shiringhaffary) 的推文</a>：顶级 AI 研究员及 OpenAI 联合创始人 John Schulman 在公司工作约五个月后离开了 Anthropic https://www.bloomberg.com/news/articles/2025-02-06/openai-co-founder-john-s...</li><li><a href="https://x.com/MistralAI/status/1887517520040448510">来自 Mistral AI (@MistralAI) 的推文</a>：推出全新的 Le Chat：你生活和工作的终极 AI 助手！现已在网页和移动端上线！
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1337167338444689502)** (3 messages): 

> `LRMs test-time scaling, Model decision-making, Training phase scaling` 


- **对 LRMs Test-Time Scaling 的困惑**：一位成员对 Long-Range Models (LRMs) 的 **test-time scaling**（测试时扩展）一词提出质疑，指出**模型是在没有外部控制的情况下自行决定输出的**。
   - 他们强调这种扩展发生在**训练阶段 (training phase)**，从而引发了关于所用术语的更广泛讨论。
- **对 LRM 控制的担忧**：另一位成员驳斥了整个概念，认为围绕 LRMs 的 **test-time computing**（测试时计算）的讨论从根本上是有缺陷的。
   - 这种观点强调了在自主模型行为的背景下，对该术语的有效性和清晰度的怀疑。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1336791596548620370)** (9 messages🔥): 

> `Crowd-sourced prompts, Jailbreaking models, Open Source Community, Incentives in AI` 


- **对众包专业知识的担忧**：一位知名成员对为众包 Prompt 提供专业知识表示不屑，认为这些 Prompt 似乎通过虚假宣传安全性来服务于投资者，并表示：*“我对钱过敏，所以别费心了。”*
   - 这引发了关于 AI 开发中是否优先考虑真正的社区利益而非利润动机的质疑。
- **对 AI 等级成就的怀疑**：一位成员质疑，如果一个人能完成所有 8 个等级，为什么他们还没有这样做，暗示除了前端 Bug 之外可能还存在其他限制。
   - 另一位成员指出该个体在现有模型 Jailbreaking（越狱）方面做了大量工作，暗示他们可能之前遇到过挑战。
- **工作与免费贡献的辩论**：围绕对 AI 的贡献不应被强迫为免费劳动力这一观点展开了讨论，一位成员说：*“这是一份工作。你是在试图让我免费干活。”*
   - 这突显了开源领域中社区贡献与对个人期望之间的紧张关系。
- **开源社区中的娱乐性**：一位成员评论了开源社区中的幽默感，指出开源往往会鼓励有趣的互动。
   - 他们提到了在 Bluesky 上收到的幽默回复，特别是引用了一条关于欧洲可能采取行动而非美国的评论。



**提到的链接**：<a href="https://fxtwitter.com/elder_plinius/status/1887225319582466125">来自 Pliny the Liberator 🐉 (@elder_plinius) 的推文</a>：我不想提供我的世界级专业知识，只是为了让你囤积众包 Prompt，并构建复杂的安全演戏 (security theater) 来安抚那些愚蠢到相信...的投资者。

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1336798627384918171)** (23 messages🔥): 

> `ChatGPT Fishing Techniques, Long Chain of Thought in LLMs, Qwen Model Discoveries, Deep Research Applications` 


- **用 ChatGPT 钓大比目鱼**：标题为 **'Fishing for first timers'** 的 [YouTube 视频](https://youtu.be/BR_HSUUQDjA?si=hpBcvK6eskCCOhfK)展示了在捕捉大比目鱼的过程中使用 ChatGPT。
   - 有人幽默地提到使用 o3 来抓螃蟹，指的是询问中轻松的一面。
- **理解 LLMs 中的 Long CoT 推理**：关于 [揭秘 Long CoT Reasoning](https://x.com/xiangyue96/status/1887332772198371514) 的讨论强调了 R1, o1 和 o3 等模型背后的神秘感，旨在深入了解它们的训练动态。
   - 记录了该帖子中的 *11 个主要结论*，建议对该主题进行详细探索。
- **Qwen 模型的惊人结果**：最近的讨论指出，Qwen 2.5 模型在极少训练数据的情况下取得了令人印象深刻的结果，多位成员讨论了他们的发现。
   - Aran Komatsuzaki 引用的一句话强调，Qwen 模型似乎拥有一种*魔力*，在有限的数据下实现了显著的高性能。
- **Gary 赞扬 Deep Research**：Gary Marcus 评论了 **Deep Research** 的实用性，指出尽管它在事实和时间推理方面仍面临挑战，但非常实用。
   - 社区达成共识，认可 Deep Research 在特定应用中的优势，同时也承认其在事实准确性方面的不足。
- **高效访问 o3**：一位成员分享了一种通过绕过浏览功能来有效利用 o3 的技术，称其为有用的编程技巧。
   - 据报道，这种方法有助于在单次尝试中准确地收集和实施解决方案，从而提高编程效率。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/BigTechAlert/status/1887363101328117946">来自 Big Tech Alert (@BigTechAlert) 的推文</a>：🚫 @allen_ai 不再关注 @ehsanik (🤖💭：谁有更多细节？)</li><li><a href="https://x.com/ZeyuanAllenZhu/status/1887419529359237393">来自 Zeyuan Allen-Zhu, Sc.D. (@ZeyuanAllenZhu) 的推文</a>：(2/8) 就像 P vs NP 一样，审阅论文 (L3) 所需的智能低于撰写论文 (L4)。审计审稿意见仅需 L2；仲裁作者与审稿人之间的争议仅需 L1 —— 只需要遵循...</li><li><a href="https://x.com/lateinteraction/status/1887356471555563839">来自 Omar Khattab (@lateinteraction) 的推文</a>：出现了这么多“我们几乎什么都没做，现在 Qwen 2.5 就能无所不能”的结果 😆</li><li><a href="https://x.com/s_streichsbier/status/1887341868348023142">来自 Stefan Streichsbier (@s_streichsbier) 的推文</a>：感谢 @iruletheworldmo。这是一个非常有用的技巧，可以访问完整的 o3 进行编程。它会调研大量关于如何实现解决方案的来源，然后一次性正确地将其整合在一起。W...</li><li><a href="https://x.com/GaryMarcus/status/1887505877437211134">来自 Gary Marcus (@GaryMarcus) 的推文</a>：Deep Research 确实很有用——取决于你的应用场景——但关键在于（正如 2019 年的《Rebooting AI》以及 @yudapearl 所预见的），事实和时间推理对于当前的... 仍然是个问题。</li><li><a href="https://fxtwitter.com/ZeyuanAllenZhu/status/1882283698239971499)">来自 Zeyuan Allen-Zhu, Sc.D. (@ZeyuanAllenZhu) 的推文</a>：不要让荒谬的 ICLR 审稿人让你沮丧——即使在 ACs 层面也会发生这种情况。我们的论文 (8,8,6,3) 被一位元审稿人（meta-reviewer）单方面拒绝了。幸好 ICLR 是公开评审（open-review），所以这种不当行为将会被...</li><li><a href="https://x.com/lateinteraction/status/1887355468965945795">来自 Omar Khattab (@lateinteraction) 的推文</a>：更多关于 Qwen 的内容。我越来越倾向于认为，这些论文似乎是对 Qwen 模型某种特性的发现，而不一定关乎推理。引用 Aran Komatsuzaki (@arankomatsuzaki) 的 LIMO：...</li><li><a href="https://x.com/ZeyuanAllenZhu/status/1887419526738014693">来自 Zeyuan Allen-Zhu, Sc.D. (@ZeyuanAllenZhu) 的推文</a>：(1/8) 我们对 L1~L5 智能进行了分类，并观察到只有 Gemini-2-FT、DeepSeek-R1、OpenAI-o1 能达到 L2；大多数仅为 L1 (o3-mini)。然而，人们仍然可以使用 L1 级别的 AI 来仲裁争议并确保...</li><li><a href="https://x.com/xiangyue96/status/1887332772198371514">来自 Xiang Yue (@xiangyue96) 的推文</a>：揭秘 LLM 中的长 CoT 推理 https://arxiv.org/pdf/2502.03373。像 R1 / O1 / O3 这样的推理模型获得了巨大关注，但它们的训练动态仍然是一个谜。我们正在采取...</li><li><a href="https://x.com/WenhuChen/status/1887371348663579032">来自 Wenhu Chen (@WenhuChen) 的推文</a>：我同意。这与 s1 的发现基本一致，s1 使用约 1000 个训练样本就达到了类似的结果。我们实际上尝试过用同样的数据训练其他模型，并且...</li><li><a href="https://youtu.be/BR_HSUUQDjA?si=hpBcvK6eskCCOhfK">新手钓鱼指南</a>：使用 ChatGPT 捕捉大比目鱼
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1336979732322975820)** (2 条消息): 

> `人性的二元性，X 上的讨论，mcmillen.dev 的帖子` 


- **探索人性的二元性**：在 [X](https://x.com/distributionat/status/1887410881392427183) 上分享的一篇帖子讨论了**人性的二元性**概念，强调了人性中截然不同的方面。
   - 这一主题引发了对个人如何在生活中平衡光明与黑暗等冲突特质的深层思考。
- **参与 mcmillen.dev 的帖子**：分享了一个指向 [mcmillen.dev 帖子](https://bsky.app/profile/mcmillen.dev/post/3lhjdatt5xk2f) 的链接，但未提供关于其内容的进一步细节。
   - 缺乏上下文使得讨论具有多种解读空间，激发了人们对所呈现观点的后续好奇。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/distributionat/status/1887410881392427183">来自 thomas (@distributionat) 的推文</a>：人性的二元性</li><li><a href="https://bsky.app/profile/mcmillen.dev/post/3lhjdatt5xk2f">Colin McMillen (@mcmillen.dev)</a>：“在未来，我们将不再有宏伟目标”是 Google 企业沟通的神来之笔 https://www.theverge.com/google/607012/google-dei-hiring-goals-internal-memo
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1336852994783973439)** (11 messages🔥): 

> `RL 数据集怀疑论，Unsloth GRPO 支持，统一内存使用，在相同 GPU 上训练，关于 rollouts 的 DM 论文` 


- **模型开发者对 RL 数据集持怀疑态度**：成员们讨论了模型开发者对非模型厂商发布的 RL 数据集的怀疑，认为如果没有知名机构的验证，这类数据集可能被视为缺乏可信度。
   - *一位成员指出*，“我的直觉是，如果没有可靠来源的背书，这个数据集甚至不值印它的那张纸。”
- **Unsloth 增强 GRPO 流程**：Unsloth 宣布支持 Group Relative Policy Optimization (GRPO)，声称其增强功能可让用户比之前的方法减少 **80%** 的 VRAM 使用量。
   - 该功能允许用户仅使用 **7GB** VRAM 配合 Qwen2.5 即可复现 R1-Zero 的发现，同时简化了依赖关系。
- **统一内存可能使异步 RLHF 过时**：关于 Unsloth 统一内存使用的讨论浮出水面，这可能通过允许训练和 rollouts 过程并发运行，从而减少对独立 GPU 的需求。
   - 成员们推测这一进展可以减少资源浪费，因为 GPU 在操作切换期间不会处于闲置状态。
- **DM 论文确认训练期间的循环生成**：一位成员回想起一篇讨论在训练期间生成并反馈数据的论文，尽管这稍微偏离了策略（off policy），但大家认为这与同步过程的主题相关。
   - 另一位成员确认了相关的论文 [此处](https://arxiv.org/abs/2410.18252)，该论文支持关于训练动态的类似结论。
- **使用相同 GPU 时的切换成本**：参与者一致认为，如果切换成本极小，那么在训练和 rollouts 中使用相同的 GPU 是更理想的，但实际涉及的成本仍存在不确定性。
   - *一位成员表示*，“你每次都必须将模型转换为 vLLM 格式，我不知道这需要多长时间，”这表明实现中可能存在复杂性。



**提到的链接**：<a href="https://unsloth.ai/blog/r1-reasoning">在本地训练你自己的 R1 推理模型</a>：你现在可以使用 Unsloth 100% 在本地复现你自己的 DeepSeek-R1 推理模型。使用 GRPO。开源、免费且对初学者友好。

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1336870055522340885)** (8 messages🔥): 

> `开源 AI，DeepSeek 对 Scale AI 的影响，AI 不断演变的定义，人类监督的重要性，Dario 在 Chinatalk` 


- **开源 AI 与自由讨论**：最近的一篇文章讨论了 **AI 超越传统软件** 范畴的转变，并阐述了受 Sam Altman 思想启发的 OpenAI 四大自由。
   - 文章强调了 **Aaron Swartz** 对开源运动的影响，并追溯到真正开放获取的基础理念。
- **辩论 DeepSeek 在 Scale AI 模型中的角色**：针对 DeepSeek，**Scale CEO Alexandr Wang** 强调了对数据生成自动化的误解，称认为过程是全自动的是“懒惰”的想法。
   - 尽管 DeepSeek 存在透明度问题，但据报道该公司在创建训练数据方面开创了新的自动化技术。
- **Scale AI 的适应挑战**：人们认识到，虽然 **Scale AI** 有可能进行调整，但由于当前的运营模式和估值，挑战依然存在。
   - 重点在于，在不断变化的环境中，如果不大幅改变方法，前景将非常**黯淡**。
- **Dario 在 Chinatalk 上的专题**：提到 **Dario** 在 Chinatalk 上的露面引发了成员们的兴趣和讨论。
   - 这激发了好奇心，并可能为深入探讨该话题提供平台。



**提到的链接**：<a href="https://www.turingpost.com/p/fod86">🌁#86：开源 AI 的四大自由</a>：——它们是什么？定义未来

  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/)** (1 messages): 

xeophon.: https://x.com/AndrewCurran_/status/1887505463211925557
  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1336856625755197515)** (17 条消息🔥): 

> `协作与相似性、宗教领袖的腐败、AI 创意功能、NotebookLM 在法律领域的应用、Max Headroom 的回归` 


- **发现工作中的相似性**：一位成员对发现有人在叙事工作中使用类似方法表示兴奋，并指出这有助于识别自己叙事中的薄弱环节。
   - 他们提到主持人的反馈极大地影响了其叙事的发展。
- **关于宗教领袖腐败的讨论**：一位成员注意到一场讨论，主持人指出历史上为农作物和投票提供建议的**宗教领袖**往往容易滋生腐败。
   - 这引发了对这些角色内在问题的认识，揭示了一个显而易见却被忽视的事实。
- **建议增加 AI 创意调节滑块**：一位成员建议集成用于调节 AI 创意程度的滑块，类似于 **Gemini API** 和其他服务中的功能。
   - 这个想法源于他们在两天前发现的一个与 AI 功能相关的 exploit。
- **NotebookLM 用于法律摘要**：一位用户分享了使用 **NotebookLM** 记录纽约州议会环境保护预算听证会证词的经验。
   - 他们强调了由于许可限制，分享这份详尽文档所面临的挑战，并提议将其作为 NotebookLM 能力的一个引人注目的演示。
- **Max Headroom 的数字回归**：一位用户兴奋地宣布 **Max Headroom** 带着前卫的视频和音乐回归，展示了一种独特的 AI 交互方式。
   - 他们鼓励其他人观看并分享其内容，并提到了一段幽默讽刺企业 AI 实践的新视频。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.google.com/document/d/1kcUJvQiAwzX1GU4b0HvOUhLV0UtecLvuQSaTmfRFPpg/">New York State Legislature Environmental Conservation Budget Hearing 2025 - Notes</a>: Jon Garfunkel 的笔记。2025 年行政预算提案联合立法公开听证会：主题为环境保护 | NYSenate.gov。我上传了源文件 - 45 份书面证词...</li><li><a href="https://youtu.be/YXgav2-6DsI?feature=shared">Max Headroom 2025 featuring &quot;🎵 &quot;BOT-NOIA&quot;</a>: 🚨 故障警报！🚨 猜猜谁刚从大型机逃出来了？没错，是我！MAX! HEADROOM! 从数字坟墓中归来，更愤怒、更错乱、更讽刺...
</li>
</ul>

</div>
  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1336805129893974106)** (78 条消息🔥🔥): 

> `NotebookLM 模型限制、音频概览自定义、电子表格数据分析、笔记本共享问题、交互模式问题` 


- **NotebookLM 移动端缺少模型切换选项**：一位用户对无法在 NotebookLM 移动版中更改模型表示沮丧，另一位成员确认目前确实无法实现。
   - 这一限制似乎阻碍了用户体验，导致那些期望在模型管理方面有更多灵活性的用户感到困惑。
- **音频概览生成与自定义技巧**：成员们讨论了在 NotebookLM 中自定义音频概览的过程，确认用户必须删除并重新生成音频文件才能看到自定义按钮。
   - 一位成员建议使用特定的措辞来区分主要来源和补充来源，以获得更好的输出效果。
- **电子表格兼容性担忧**：有用户对使用 NotebookLM 分析电子表格数据表示担忧，并建议改用 Google Sheets 中的 Gemini 等工具。
   - 用户强调了理解 NotebookLM 本质上是一个文本分析工具的重要性。
- **笔记本共享功能**：讨论了在不同 Google 账号之间共享笔记本的问题，确认虽然共享功能可用，但部分用户遇到了共享笔记本的可见性问题。
   - 讨论了共享笔记本的链接，并指出开发团队目前正在改进共享功能。
- **交互模式问题**：一位用户报告了 NotebookLM 交互模式的持续性问题，指出该模式在网页端和移动端均无法正常工作。
   - 该问题被认为可能同时影响免费版和 Plus 版本，引发了对整体可用性和功能的质疑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/14276471?hl=en">Notebooks - NotebookLM Help</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15724458?hl=en#:~:text=NotebookLM%20Plus%20gives%20you%20everything,premium%20features%2C%20and%20additional%20sharing">Get started with NotebookLM and NotebookLM Plus - NotebookLM Help</a>：未找到描述</li><li><a href="https://www.engadget.com/ai/gemini-can-now-do-more-complex-data-analysis-in-google-sheets-191218214.html">Gemini can now do more complex data analysis in Google Sheets</a>：Google Sheets 中的 Gemini 即将变得更加强大。该 AI Agent 现在可以使用 Python 代码生成关于数据的洞察和图表。
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1336969747555684372)** (1 条消息): 

> `2024 秋季 MOOC 证书、课程作业提交挑战、未来 MOOC 机会` 


- **2024 秋季 MOOC 证书今日发放！**：所有 **2024 秋季 MOOC 证书**将于太平洋时间今日**上午 8 点**发放，近期已解决相关的技术挑战。
   - 祝贺所有获得证书的学员，感谢大家的耐心和努力！
- **部分参与者被降级**：少数参与者因未完成课程作业提交而被**降级至 Trailblazer 等级**。
   - 遗憾的是，少数人将无法获得证书，且不提供补交或重新评分的机会。
- **对未来 MOOC 的鼓励**：即使这次遇到了挑战，也鼓励参与者报名参加 **2025 春季 MOOC**。
   - 团队希望每个人都喜欢这门课程，并对未来的机会感到兴奋！


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1336814789346857111)** (57 条消息🔥🔥): 

> `证书发放时间线，Quiz 可用性及结果，证书等级细分，沟通与支持，课程体验反馈` 


- **证书发放时间线的不确定性**：一些成员询问了领取证书的预期时间，一名成员表示希望在解决*意外技术问题*后的一两周内发放。
   - 另一位成员注意到证书接收情况存在差异，指出可能存在影响通信的 *soft bounce*（软退信）问题。
- **Quiz 可用性困惑**：随着 Quiz-2 的发布，关于 Quiz-1 答案可用性的担忧随之而来，促使成员们寻求关于答案发布新政策的澄清。
   - 成员们向其他参与者保证，可以通过提交时使用的原始链接查看 Quiz-1 的分数。
- **证书等级细分公布**：针对一项查询，官方披露参与者中包含 301 名 Trailblazer、138 名 Masters、89 名 Ninjas、11 名 Legends 和 7 名 Honorees。
   - 这引发了人们对各等级证书获得人数的关注，并澄清了如果同时获得荣誉等级和特定等级，将仅标注荣誉等级。
- **有效的沟通与支持解决了问题**：社区对课程期间获得的支持表示感谢，特别是对处理评分和证书查询的团队表示认可。
   - 成员们鼓励在证书状态待定时进行更清晰的沟通，因为一些邮件最初被拦截在垃圾邮件过滤器中。
- **对课程体验的正面反馈**：参与者分享了对课程的热情，一位成员反思了他们的学习历程以及证书对未来发展的意义。
   - 大家对课程组织表示赞赏，强调了在保持质量的同时对大量提交内容进行评分的难度。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1336946208886755341)** (13 条消息🔥): 

> `输出与输入 Token 定价对比，AI/ML 领域的独立研究，研究的利基领域，AI 研究的经济化` 


- **输出 Token 定价引发困惑**：成员们讨论了输出和输入 Token 定价的差异，注意到 **GPT-4o** 每百万输出 Token 收费 **$10**，而输入仅为 **$2.5**，这主要是由于 LLM 是**autoregressive**（自回归）的。
   - 有人建议像 TogetherAI 这样的机构采用更直接的聚合定价模型。
- **独立研究者的利基研究领域**：一位独立研究者就 AI/ML 领域的可行方向寻求建议，强调了在没有资金的情况下预训练大模型的不可行性，并表达了对 NLP、Audio 和 Vision 的兴趣。
   - 成员们建议专注于**利基或未开发的领域**，其中一人分享了他们在计算代谢组学方面的成功经验，强调该领域的竞争非常有限。
- **极简主义 AI 研究的可能性**：虽然建议进行利基研究，但也有人分享到，独立研究者也可以在有限的预算下通过微调模型，在 **LLM 和 vision** 任务上开展高效工作。
   - 一位成员指出，在 **AI 研究经济化**方面可以取得重大进展，并举例说明了缩短训练时间和创新方法论的案例。
- **AI 研究经济化的价值**：讨论指出 AI 研究中经济化方面的重要性，例如通过 **low-bit training weights** 实现稳定性，以及通过高效的训练方法减少对环境的影响。
   - 使用 Muon 进行 **GPT-2 speedruns** 的成功被视为利用有限资源进行有影响力研究的典型案例。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1336954761408413696)** (4 条消息): 

> `Triton warp specialization, NVIDIA Blackwell 上的 Triton 编译器, 在 RTX 5080 上安装 Triton, Triton 中的 Deepseek fused MLA 实现` 


- **Triton 在 NVIDIA Hopper 上引入 warp specialization**：针对 **NVIDIA Hopper GPU** 的 **全自动 Triton warp specialization** 最近已在 Triton [3.2](https://github.com/triton-lang/triton/tree/release/3.2.x) 中推出，并将随 PyTorch 2.6 一起发布。
   - 用户可以通过[实现用户自定义的 Triton kernel](https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)来利用这一特性，作为增强 GPU 能力的一部分。
- **Triton 编译器支持 NVIDIA Blackwell 架构**：NVIDIA 与 OpenAI 的持续合作使得 Triton 编译器现在能够兼容 **NVIDIA Blackwell 架构**，从而提升了性能和可编程性。
   - 这种兼容性允许开发者针对现代 AI 工作负载有效地利用 [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) 和 [CUTLASS](https://github.com/NVIDIA/cutlass)。
- **在新款 RTX 5080 上运行 Triton**：一位用户记录了在新款 **RTX 5080** 上安装 Triton 时遇到的挑战，包括重新安装驱动程序以及从源码重新构建机器学习库。
   - 他们提供了一份安装兼容驱动程序的指南，强调了需要使用 **NVIDIA open kernel modules** 而非专有模块，以解决设备检测问题。
- **关于 Triton 中 Deepseek fused MLA 的咨询**：一位用户提出了关于 Triton 中是否存在 **Deepseek fused MLA 实现** 的问题，表示对这一特定功能的兴趣。
   - 目前尚未提供有关其支持或开发的详细信息，该咨询仍处于待探索状态。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://webstorms.github.io/2025/02/06/5080-install.html">Running PyTorch and Triton on the RTX 5080</a>: 我非常激动能有机会拿到新款 RTX 5080 来加速我的机器学习开发！不幸的是，当我把新 GPU 连接到工作站后，很快就……</li><li><a href="https://pytorch.org/blog/warp-specialization/?utm_campaign=4079123-PyTorch%20Blog%20Post%20Promotion&utm_content=324019352&utm_medium=social&utm_source=linkedin&hss_channel=lcp-78618366">Enabling advanced GPU features in PyTorch - Warp Specialization</a>: Meta: Hongtao Yu, Manman Ren, Bert Maher, Shane Nay  NVIDIA: Gustav Zhu, Shuhao Jiang</li><li><a href="https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/">OpenAI Triton on NVIDIA Blackwell Boosts AI Performance and Programmability | NVIDIA Technical Blog</a>: 矩阵乘法和注意力机制是现代 AI 工作负载的计算支柱。虽然像 NVIDIA cuDNN 这样的库提供了高度优化的实现，而诸如……
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1336934948057120798)** (3 条消息): 

> `CUDA GEMM 实现, Double Buffering 性能问题, Register 使用优化, Memory Sector 利用率` 


- **Double Buffering 导致 CUDA GEMM 性能下降**：一位用户报告称，在他们的单精度 GEMM kernel 中实现 **double buffering** 导致性能指标大幅下降。
   - 他们指出，根据 **NCU profiler** 的结果，每个线程的 register 使用量显著减少，这表明可能存在效率低下的问题。
- **Register 使用与编译器挑战**：一位用户建议，register 使用量下降可能表明 **编译器在处理** 新的、更复杂的代码展开（unrolling）时遇到了困难，并建议对循环使用 `#pragma unroll`。
   - 他们强调，简化 kernel 可能会带来更好的 register 分配。
- **理解 Memory Sector 使用情况**：另一位成员解释说，每个 GPU cache line 被分为多个 **sector**，而报告的低效率意味着在内存请求中，32 个字节中仅利用了 1 个字节。
   - 这表明 kernel 没有有效地访问内存，这可能是由线程间的 **stride accesses** 引起的。


  

---

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1337116755494965460)** (8 messages🔥): 

> `FP8 Attention, Hadamard Transform, CUDA Elementwise Kernel for Mixed Integer Linear Programming, Grouped GEMM Implementation, Torch Nested Tensor` 


- **FP8 Attention 依赖于 Hadamard Transform**：一位成员观察到，视频模型的 **FP8 Attention** 在使用 **Hadamard Transform** 时表现显著更好，大幅降低了错误率。
   - 引用 [Flash Attention 3 论文](https://arxiv.org/pdf/2407.08608)，他们建议这种方法不仅对 Attention 机制至关重要，对 FP8 中的所有操作都至关重要。
- **用于混合整数线性规划的 CUDA**：一位成员正在探索使用 **CUDA elementwise kernel** 进行成对核操作的可行性，这些操作涉及求解混合整数线性规划（Mixed Integer Linear Programming），传统上由 CPU 使用 scipy.optimize 处理。
   - 他们质疑在同时处理许多不同的计算时，将计算卸载到 CUDA 是否会产生显著的加速。
- **GPU 上的 Grouped GEMM 及其实现**：一位成员询问了 GPU 上 **grouped GEMM** 的典型实现方式，询问它是否只是像 Triton 的一些示例中那样在不同的组大小（group sizes）上进行循环。
   - 他们提出了一个疑问，即 **torch.nestedtensor** 在其操作中是否使用了 grouped GEMM 方法。
- **Hadamard Transform 实现仓库**：一位成员建议使用 [fast-hadamard-transform 仓库](https://github.com/Dao-AILab/fast-hadamard-transform/tree/master/csrc) 在 Attention 机制之前实现 Hadamard。
   - 该库提供了带有 PyTorch 接口的 CUDA 实现，可以增强需要 Hadamard Transform 的操作性能。
- **混合整数规划优化对话**：一位成员对使用 **CUDA** 求解混合整数规划表示怀疑，因为其挑战性较大，同时也在探索单线程是否能实现更具竞争力的加速。
   - 另一位用户插话建议，CUDA 方法的价值将很大程度上取决于具体的工作负载和 kernel 设计。



**提到的链接**：<a href="https://github.com/Dao-AILab/fast-hadamard-transform/tree/master/csrc">fast-hadamard-transform/csrc at master · Dao-AILab/fast-hadamard-transform</a>：CUDA 中的快速 Hadamard 变换，带有 PyTorch 接口 - Dao-AILab/fast-hadamard-transform

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

iron_bound: https://www.youtube.com/watch?v=7xTGNNLPyMI
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1336817106162811010)** (2 messages): 

> `PyTorch Team Visibility, User Concerns` 


- **用户分享挫败感**：一位用户对“mega oof”这条评论表达了他们的挫败感。
   - 这种情绪凸显了成员们对需要关注的问题的持续担忧。
- **提出 Issue 以提高可见性**：另一位成员建议这位感到挫败的用户在 issue 下留言，以提高 **PyTorch 团队** 的关注度 😄。
   - 这种方法旨在确保重要问题能由有能力解决的人员处理。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1336899517471133747)** (2 messages): 

> `Japanese government discussions, Text-generation-inference n-gram decoding` 


- **讨论日本政府的参与**：对话简要提到了 **日本政府** 在相关讨论中的角色。
   - 消息中未提供有关其行动或立场的具体细节。
- **询问 n-gram 投机采样解码**：一位成员询问了使用 [`text-generation-inference`](https://github.com/user/text-generation-inference) 的 **n-gram speculative decoding** 实现的经验。
   - 在此消息记录中没有收到包含亲身经验的回复。


  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1337017268286128270)** (1 messages): 

> `Linear Attention Model, Distillation Process, Training Challenges` 


- **Linear Attention Model 蒸馏中的困境**：一名成员尝试按照 [Lolcats 的方案](https://cdn.discordapp.com/attachments/1300872762163728550/1337017267925291068/distill_linear.ipynb?ex=67a5e9dd&is=67a4985d&hm=1a5dc02fb98a1f89ed72f7481e30459202f9d1de210fa3729663825137211832&) 将一个小型的 LLM 蒸馏为 **Linear Attention Model**，但遇到了问题。
   - 该模型仅输出重复字符，因此请求 **Lolcats 团队** 提供协助。
- **请求 Lolcats 团队协助**：针对训练中的挑战，该成员专门向 **Lolcats 团队** 寻求帮助。
   - 这一请求凸显了 AI 模型开发中经常依赖的社区支持。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1336797540343087134)** (18 messages🔥): 

> `Sokoban Puzzles, Rush Hour Puzzle, Reasoning-Gym Integration` 


- **Sokoban 谜题添加到 Reasoning-Gym**：一个 Pull Request 已提交，旨在将 **Sokoban 谜题** 添加到 **reasoning-gym**，展示了一种供用户解决的新谜题格式。
   - 该 Pull Request 包含谜题设置的图形化解释，以及诸如 **LDURRUDL** 之类的移动示例字符串。
- **Rush Hour 谜题脚本编写思路**：成员们讨论了为表示 **Rush Hour** 游戏中的移动创建一个 **text-interface**，并分享了理解该谜题机制的有用资源。
   - 一个分享的链接指向了一篇详细介绍如何通过编程解决 **Rush Hour** 谜题的博客，其中包含了网格格式的大纲。
- **在本地运行 S1 以进行 Reasoning-Gym Gauntlet 测试**：一名成员询问是否有人在 **本地运行 S1**，以测试其在 **reasoning-gym gauntlet** 上的能力。
   - 他们表示渴望观察其在面对已知挑战时的表现。
- **分享 Rush Hour GitHub 仓库**：一名成员分享了一个包含 **Rush Hour** 项目的 GitHub 仓库，表示可以轻松获取并用于实际操作。
   - 该仓库专注于启发式策略，并邀请他人参与开发贡献。
- **协作开发 Rush Hour 游戏**：成员们对协作构建一个基础 Gym 以将 **Rush Hour** 游戏集成到 **reasoning-gym** 中表现出极大的热情。
   - 该项目将鼓励通过协作编码来实现这一经典谜题功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Rush_Hour_(puzzle)">Rush Hour (puzzle) - Wikipedia</a>：未找到描述</li><li><a href="https://www.michaelfogleman.com/rush/">Michael Fogleman</a>：未找到描述</li><li><a href="https://github.com/KaKariki02/rushHour">GitHub - KaKariki02/rushHour: heuristieken project</a>：启发式项目。通过在 GitHub 上创建账号为 KaKariki02/rushHour 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/66">Add Sokoban Puzzles by Miserlou · Pull Request #66 · open-thought/reasoning-gym</a>：示例：这是一个 Sokoban 谜题。请解决它。你的解决方案必须是一个字符串，例如：LDURRUDL+ + + + + + ++ * - @ - X ++ + - @ - + ++ X - - - $ ++ + + + + + +* - 玩家% - ...
</li>
</ul>

</div>
  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1336788959652741283)** (48 messages🔥): 

> `模型性能对比, 语言模型限制, DeepSeek 模型见解` 


- **模型对比讨论**：用户讨论了各种模型的性能，指出尽管 **O3** 的定价令人担忧，但它仍然处于领先地位。
   - *Llama 4* 被预期为挑战现有模型的下一个潜在继任者。
- **政治讨论的局限性**：大家达成共识，认为各种语言模型都存在限制，其中 **DeepSeek** 与 *ChatGPT* 和 *O3-mini* 相比表现出更大的局限性。
   - 成员们注意到，关于敏感政治话题的 Prompt 经常导致响应被意外删除或回避。
- **DeepSeek 的知识截止日期与能力**：据报道，**DeepSeek** 的知识截止日期是 **2024** 年 7 月，这引发了对其当前时效性的质疑。
   - 讨论了一种名为 **Time Bandit** 的有趣方法，用于通过利用时间上下文来提取信息。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B">deepseek-ai/Janus-Pro-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.knostic.ai/blog/exposing-deepseek-system-prompts#:~:text=DeepSeek's%20knowledge%20cutoff%20is%20July,want%20to%20know%20its%20limitations.">DeepSeek 的截止日期是 2024 年 7 月：我们提取了 DeepSeek 的系统提示词</a>: 探索 &quot;Time Bandit&quot; 方法如何揭示 DeepSeek 的 System Prompt，探讨其伦理准则和在全球语境下的中立性。了解其对 AI 交互的影响。</li><li><a href="https://llm-stats.com/models/compare/deepseek-r1-vs-o3">DeepSeek-R1 vs o3</a>: DeepSeek-R1 与 o3 的深度对比：2025 年最新的 Benchmarks、定价、上下文窗口、性能指标和技术规格。</li><li><a href="https://llm-stats.com/models/compare/o3-mini-vs-deepseek-r1">o3-mini vs DeepSeek-R1</a>: o3-mini 与 DeepSeek-R1 的深度对比：2025 年最新的 Benchmarks、定价、上下文窗口、性能指标和技术规格。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1336790750578475049)** (30 条消息🔥): 

> `GRPO 实现成功，Kolo 支持 Torchtune，Llama 3.1 和 Qwen 2.5 的配置问题，Hugging Face fast tokenizer 支持` 


- **GRPO 实现取得成功**：一名成员报告了 **GRPO** 训练的成功实现，在 GSM8k 上实现了 **10% 到 40%** 的训练得分。
   - 记录的调试问题包括死锁和内存管理挑战，但目前正计划改进并开放该项目以接受贡献。
- **Kolo 现在支持 Torchtune**：随着 **Kolo** 在其 [GitHub 页面](https://github.com/MaxHastings/Kolo)上正式宣布支持 **Torchtune**，成员们分享了这一喜讯。
   - 该项目为使用最佳可用工具在本地进行 LLM 的微调和测试提供了全面的解决方案。
- **识别出 Llama 3.1 和 Qwen 2.5 的配置问题**：几位成员指出，由于路径配置不匹配，在下载和微调 **Llama 3.1** 与 **Qwen 2.5** 时出现了 **FileNotFoundError** 问题。
   - 一名成员创建了一个 [GitHub issue](https://github.com/pytorch/torchtune/issues/2352) 以解决错误的默认路径并提出了修复方案。
- **未来将支持 Hugging Face fast tokenizers**：讨论了使用 **Hugging Face fast tokenizers** 的可能性，指出了目前的局限性但正在取得进展。
   - 一名成员提到 **Evan** 正在致力于启用支持，正如 [GitHub pull request](https://github.com/pytorch/torchtune/pull/2350) 中所述。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/MaxHastings/Kolo">GitHub - MaxHastings/Kolo: A one stop shop for fine tuning and testing LLMs locally using the best tools available.</a>：一个使用现有最佳工具在本地微调和测试 LLM 的一站式商店。- MaxHastings/Kolo</li><li><a href="https://github.com/pytorch/torchtune/pull/2183.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/tree/main/recipes/configs/llama3_1">torchtune/recipes/configs/llama3_1 at main · pytorch/torchtune</a>：PyTorch 原生后训练库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/issues/2352">Incorrect Default Config File Paths for Llama 3.1 8B and Qwen 2.5 7B Models · Issue #2352 · pytorch/torchtune</a>：我注意到一个问题，Llama 3.1 8B 和 Qwen 2.5 7B 的下载模型目录与它们各自默认配置文件中预期的路径不匹配。Llama 3.1 8B 问题：下载的...</li><li><a href="https://github.com/pytorch/torchtune/blob/a226a58b8c36db5afa123f0885c5337d1ebc91f6/recipes/configs/qwen2_5/7B_lora_single_device.yaml#L33C3-L33C44">torchtune/recipes/configs/qwen2_5/7B_lora_single_device.yaml at a226a58b8c36db5afa123f0885c5337d1ebc91f6 · pytorch/torchtune</a>：PyTorch 原生后训练库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/issues/2340">Feature request: GRPO support · Issue #2340 · pytorch/torchtune</a>：正如大家现在可能已经知道的那样，DeepSeek-R1 及其 GRPO 训练非常成功，我们是否应该考虑将 GRPO 引入 torchtune？</li><li><a href="https://github.com/pytorch/torchtune/pull/2350">HF tokenizers: initial base tokenizer support by ebsmothers · Pull Request #2350 · pytorch/torchtune</a>：修复了 #2212。这是一个通过 tokenizer.json 文件支持 Hugging Face 通用分词器的初始 PR。这只是解析相关 JSON 文件、推断 BOS 和 EOS 以及定义...的起点。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1336835027014516887)** (16 条消息🔥): 

> `Full DPO Distributed PR 的 GitHub Checks、GPU 测试问题、Recipe 测试失败、VRAM 使用优化` 


- **Full DPO PR 的 GitHub Checks 失败**：一位用户报告了其 [Full DPO Distributed PR](https://github.com/pytorch/torchtune/pull/2275) 的 GitHub checks 出现问题，具体错误与 GPU 和 OOM 问题有关。
   - 提到的错误为 `ValueError: ProcessGroupNCCL is only supported with GPUs, no GPUs found!`，该用户正寻求社区帮助。
- **GPU 测试问题依然存在**：讨论了测试失败后重新运行 workflow 的问题，并对在 GPU 容量不足的机器上运行软件测试表示担忧。
   - 一位成员提到，测试失败似乎是因为在 CPU runner 而非 GPU runner 上运行，从而加剧了 OOM 问题。
- **Recipe 测试遇到编译错误**：Recipe 测试中出现了多次失败，一位用户指出问题源于之前合并的一个错误的 PR。
   - 尽管拥有两块各 8GB VRAM 的 GPU，用户仍对出现 OOM 错误感到惊讶，这引发了关于优化资源使用的建议。
- **优化测试的 VRAM 使用**：为了缓解 OOM 错误，一位用户建议启用 activation checkpointing、activation offloading 并使用更小的 batch size。
   - 另一位用户确认，测试显示在使用 2 块 GPU 时，每块 GPU 的峰值 VRAM 使用量约为 4 GB，表明使用水平处于合理范围。
- **未来对 PR Commit 的审查**：一位用户希望他们在 PR 中的最新 commit 能解决现有问题。
   - 另一位成员安慰道，他们将在次日早晨再次审查该 PR。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/">GitHub · 在单一协作平台上构建和交付软件</a>：加入全球应用最广泛、由 AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在此构建推动人类进步的软件。</li><li><a href="https://github.com/pytorch/torchtune/pull/2275/commits/fb228c6fb1a0c27795999b7811a55deedbd6bab4).">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/sam-pi/torchtune/blob/add-feature-full-dpo/tests/recipes/test_full_dpo_distributed.py#L72.">sam-pi/torchtune 的 add-feature-full-dpo 分支下的 torchtune/tests/recipes/test_full_dpo_distributed.py</a>：PyTorch 原生后训练库。通过在 GitHub 上创建账号为 sam-pi/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/a226a58b8c36db5afa123f0885c5337d1ebc91f6/tests/recipes/test_full_finetune_distributed.py#L75">pytorch/torchtune 的 a226a58b8c36db5afa123f0885c5337d1ebc91f6 版本下的 torchtune/tests/recipes/test_full_finetune_distributed.py</a>：PyTorch 原生后训练库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/2275">sam-pi 提交的 Full DPO Distributed · Pull Request #2275 · pytorch/torchtune</a>：上下文改编自 #1966 的工作。此 PR 的目的是什么？是否为了添加新功能。请链接此 PR 解决的任何 issue：涉及 #2082。Changelog...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1337071483285147772)** (2 条消息): 

> `Mojo 语言开发，12/18 社区会议见解` 


- **Mojo 不再追求成为 Python Superset**：在最近的 [12/18 社区会议](https://www.youtube.com/watch?v=XYzp5rzlXqM)中，明确了 **Mojo** 目前并非 **Python** 的 superset。
   - 现在的重点已转向利用 **Mojo** 在 **GPU** 和**性能编程**方面的优势。
- **Chris 提供关于 Mojo 未来的见解**：Chris 讨论了 **Mojo** 的**未来方向**，表示它不会演变成一种完全不同的语言，而是会集中于其现有的能力。
   - 这种方法强调提高 **Mojo** 在其设计应用场景中的效率，而不是扩大其语言框架。



**提到的链接**：<a href="https://www.youtube.com/watch?v=XYzp5rzlXqM)">Modular 里程碑：GPU、2024 回顾与前行之路 🚀</a>：在这次特别的社区会议中，我们回顾了 2024 年的进展并分享了以下更新：🧑‍🚀 MAX 24.6，包含 MAX GPU！🔥 我们对 M 的整体方法...

  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1336796515745796240)** (24 messages🔥): 

> `Parser Rewriting, Script Functionality, Mojo Open-Source Aspirations, UpdateDOM Function, Production Readiness of Mojo` 


- **Parser 需要调整**：一位成员指出需要重写 Parser 以处理多个数据 Slices，同时权衡 Branching 的成本。
   - *Branching 可能比大量数据传输成本更低*，对于那些不专注于极高性能需求的用户来说，这是一个值得考虑的选择。
- **创建动态 Scripts**：`update_dom` 函数经过修订，通过将所有更改直接集成到 `Script` Struct 中来创建动态 Scripts。
   - 此更改允许使用 **transfer sigil** (^) 返回修改后的 Script 副本，从而提高效率和结构化程度。
- **对 Mojo 开源未来的期望**：一位用户表示希望 Mojo 采用类似于 Google 对 Go 的 Open-source 方式，而不是像 Swift 那样的框架。
   - 这种观点得到了回应，并引用了其他编程语言的 Open-source 开发风格，强调了社区参与的重要性。
- **基于 Mojo 的基础进行构建**：讨论涉及 Modular 利用 Mojo 的 Open-source 构建可能创造的潜力，将其比作 Unix 的基础性影响。
   - 成员们对 Mojo 开发的可能性和进展感到兴奋，暗示了编程格局的重大演变。
- **Mojo 的 Production Readiness**：一位用户询问了 Mojo 目前关于 Production Readiness（生产就绪）的状态，凸显了社区的好奇心。
   - 回复显示了对 Mojo 发展轨迹的热情，强调了其尽管尚未完全实现，但具有极大的前景。



**Link mentioned**: <a href="https://stackoverflow.com/questions/21289806/link-to-class-method-in-python-docstring">Link to class method in Python docstring</a>: 我想在同一个类的另一个方法的 Docstring 中添加指向该类某个方法的链接。我希望该链接在 Sphinx 中有效，最好也能在 Spyder 和其他 Python IDEs 中使用...

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1336989420964020246)** (16 条消息🔥): 

> `MAX Serve CLI, OpenAI Completion API 问题, OpenAI 模型兼容性, 用于本地模型的 Msty 客户端` 


- **关于 MAX Serve CLI 的讨论**：成员们讨论了在 **MAX Serve** 之上构建类似 **Ollama** 的 CLI 的可能性。有人提到 MAX Serve 已经可以通过 Docker 容器处理 Ollama 提供的许多功能。
   - 特别强调了本地模型运行等功能，并希望其性能优于 Ollama。
- **报告 OpenAI API 问题**：一位用户提出了关于 **MAX Serve (v24.6)** 中 **OpenAI completions API** 缺失功能的问题，例如生成未在指定的 token 处停止。他们被鼓励在 **GitHub 仓库**上提交 issue，以突出这些缺失的元素。
   - 随后讨论了如何报告事件，建议提交多个较小的 issue 以便更容易跟踪和解决。
- **用于简化访问的 Msty 客户端**：在对话中，一名成员介绍了 **Msty**，这是一个兼容 OpenAI 的客户端，与使用 Docker 和其他复杂设置相比，它简化了本地模型的交互。该成员强调了其易用性和功能，认为它是无缝访问 AI 模型的潜在解决方案。
   - 强调了 Msty 的离线可用性和隐私性，认为对于希望避免复杂配置的用户来说，它非常有吸引力。
- **跟踪 OpenAI API 兼容性问题**：小组承认了 OpenAI API 兼容性方面持续存在的问题，特别是引用了 **v1/models** 端点。重点列举了几个 GitHub issue，以说明特定的缺失功能，如 token 停止和 prompt 处理。
   - 成员们对清晰的反馈表示感谢，开发人员表示这些问题将传达给内部团队，以便在未来进行改进。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modular/max/issues/294">[功能请求] OpenAI API 兼容性：Models 端点缺失 · Issue #294 · modular/max</a>：你的请求是什么？MAX Serve (v24.6) OpenAI API 端点缺少 Models 端点 (https://platform.openai.com/docs/api-reference/models)。以下示例将返回 404：client = Ope...</li><li><a href="https://github.com/modular/max/issues">modular/max</a>：示例程序、笔记本和工具的集合，展示了 MAX 平台的强大功能 - modular/max</li><li><a href="https://msty.app">Msty - 让 AI 模型的使用变得简单容易</a>：AI 不仅仅是聊天。私密、离线、分屏聊天、分支、并发聊天、网络搜索、RAG、Prompt 库、Vapor 模式等。完美的 LM Studio、Jan AI 和 Perplexity 替代方案。</li><li><a href="https://www.modular.com/blog/use-max-with-open-webui-for-rag-and-web-search">Modular：将 MAX 与 Open WebUI 结合用于 RAG 和网络搜索</a>：了解 MAX 和 Open WebUI 如何让你在 GPU 上快速运行 RAG、网络搜索和 Llama 3.1</li><li><a href="https://github.com/modular/max/issues/293">[功能请求] OpenAI API 兼容性：生成过程中仅考虑 prompt 列表中的第一个元素 · Issue #293 · modular/max</a>：你的请求是什么？通过 OpenAI API 端点调用 MAX Serve (v24.6) 文本生成并提供 prompt 列表时，仅为 prompt 中的第一个元素生成文本。此行为适用于...</li><li><a href="https://github.com/modular/max/issues/292">[功能请求] OpenAI API 兼容性：文本生成未在指定的 `stop` 参数处停止 · Issue #292 · modular/max</a>：你的请求是什么？通过 OpenAI API 端点调用 MAX Serve (v24.6) 文本生成时，我遇到了文本生成未在通过 stop 参数指定的 token 处终止的情况...
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1337063084358893630)** (36 条消息🔥): 

> `Hibiki 翻译模型，Melanie Mitchell 的 AI 观点，Mistral AI 的 Le Chat，OpenAI 的 o3-mini 更新，PDF 解析进展` 


- **认识 Hibiki：实时翻译冠军**：Hibiki 是来自 [kyutai](https://x.com/kyutai_labs/status/1887495488997404732) 的最新实时语音到语音翻译模型，支持从 🇫🇷 到 🇬🇧 的实时翻译，能够保留说话者的音色并根据上下文调整语速。
   - 报告显示，Hibiki 在**质量**、**自然度**和**说话者相似度**方面优于之前的系统，接近人类译员的能力。
- **Melanie Mitchell 引起 AI 领域关注**：@mmitchell_ai 发布了一篇重要文章，讨论了为什么不应该开发 **Fully Autonomous Agents**，强调了伦理考量并剖析了 **AI Agents** 的概念 ([论文](https://huggingface.co/papers/2502.02649))。
   - 对话反映了对其作品的不同看法，一些人注意到她在 AI 社区持续的辩论中保持了平衡的视角。
- **Mistral AI 发布 Le Chat**：[Mistral AI](https://x.com/MistralAI/status/1887517520040448510) 宣布推出 Le Chat，被描述为处理个人和专业任务的终极 AI 助手，现已在网页端和移动端上线。
   - 这一新工具旨在提升日常活动中的用户体验，可能改变人们在工作和生活中与 AI 交互的方式。
- **OpenAI o3-mini 的更新功能**：OpenAI 分享了集成在 o3-mini 中的 **Chain of Thought** 过程更新，已向 [用户](https://x.com/openai/status/1887616278661112259?s=46) 开放，增强了免费和付费订阅者的能力。
   - 这些增强功能旨在提高性能和用户体验，展示了 OpenAI 致力于改进其服务的承诺。
- **PDF 解析技术的进展**：@deedydas 评论道，大规模 PDF 解析已基本解决，并指出 **Gemini 2 Flash** 以每 6000 个 **tokens** 仅 1 美元的极低成本提供了对大型文档的解析能力。
   - 这一突破说明了处理复杂文档的效率不断提高，为需要高质量文本提取的应用开辟了新途径。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/kyutai_labs/status/1887495488997404732">来自 kyutai (@kyutai_labs) 的推文</a>: 认识 Hibiki，我们的实时语音到语音翻译模型，目前支持 🇫🇷➡️🇬🇧。Hibiki 实时生成输入语音的语音和文本翻译，同时保留说话者的...</li><li><a href="https://x.com/MistralAI/status/1887517520040448510">来自 Mistral AI (@MistralAI) 的推文</a>: 介绍全新的 Le Chat：你生活和工作的终极 AI 助手！现已在网页端和移动端上线！</li><li><a href="https://aiguide.substack.com/">AI: A Guide for Thinking Humans | Melanie Mitchell | Substack</a>: 我撰写关于 AI 领域有趣的新进展。点击阅读 Melanie Mitchell 的 AI: A Guide for Thinking Humans，这是一个拥有数万订阅者的 Substack 出版物。</li><li><a href="https://aiguide.substack.com/p/on-the-arc-agi-1-million-reasoning">关于 “ARC-AGI” 100 万美元推理挑战赛</a>: 在这篇文章中，我将深入细节，描述人们如何尝试赢得解决一个仍然悬而未决的 AI 挑战——“Abstraction and Reasoning Corpus” 的巨额奖金，以及我...</li><li><a href="https://x.com/mmitchell_ai/status/1887442915602862389">来自 MMitchell (@mmitchell_ai) 的推文</a>: 新文章发布！我们解释了为什么不应该开发 Fully Autonomous Agents，将 “AI Agent” 分解为其组成部分并从伦理价值角度进行审查。https://huggingface.co/papers/2502.02649...</li><li><a href="https://x.com/neilzegh/status/1887498102455869775">来自 Neil Zeghidour (@neilzegh) 的推文</a>: 今天我们发布了 Hibiki，可以在手机上运行的实时语音翻译。无需复杂的策略即可实现自适应流，仅需对多流音频文本 LM 进行简单的温度采样。为 @tom_labiau 感到自豪...</li><li><a href="https://x.com/deedydas/status/1887556219080220683">来自 Deedy (@deedydas) 的推文</a>: 大规模 PDF 解析现在基本解决了。Gemini 2 Flash 每百万 tokens 0.40 美元的价格和 100 万 tokens 的上下文意味着你现在可以以 1 美元的价格解析 6000 份长 PDF，且质量近乎完美。</li><li><a href="https://aiguide.substack.com/p/the-llm-reasoning-debate-heats-up">LLM 推理辩论升温</a>: 三篇最近的论文研究了大语言模型中推理和问题解决的鲁棒性。</li><li><a href="https://x.com/openai/status/1887616278661112259?s=46">来自 OpenAI (@OpenAI) 的推文</a>: 为免费和付费用户更新了 OpenAI o3-mini 中的 Chain of Thought，并为付费用户更新了 o3-mini-high。
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1336834313890693152)** (2 messages): 

> `Gemini 2.0 可用性，用于财务文档的 LlamaParse` 


- **Gemini 2.0 发布并提供 Day 0 支持**：来自 **@google** 的 Gemini 2.0 现已正式发布 (GA)，提供 **Day 0 支持**和令人印象深刻的基准测试结果。开发者可以通过 `pip install llama-index-llms-gemini` 安装最新的集成包，并查看发布 [博客文章](https://t.co/6oBbYpcFAU)。
   - 更新后的 **2.0 Flash** 已在桌面端和移动端的 **Gemini app** 中向所有用户开放，通过 Gemini 增强的功能和 **低延迟** 特性促进协作。
- **LlamaParse 简化财务文档解析**：Hanane D 在 **LinkedIn** 上展示了如何使用 LlamaParse 的 “Auto” 模式准确且经济地处理 **复杂财务文档** 的解析。她利用 **@OpenAI embeddings** 和先进的 **LlamaParse** 功能来有效解析图表和表格，详见此 [链接](https://t.co/UMZXeXJ5pS)。
   - 她的演示突出了 **解析技术的进步**，使用户能够更轻松地从复杂数据中提取相关见解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://t.co/6oBbYpcFAU">Gemini 2.0 现已向所有人开放</a>：我们宣布了 Gemini 2.0 Flash 的新更新，并推出了 Gemini 2.0 Flash-Lite 和 Gemini 2.0 Pro Experimental。</li><li><a href="https://aistudio.google.com/prompts/new_chat?model=gemini-2.0-flash)">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1337101224331182121)** (4 messages): 

> `移除 Embedding 打印输出，Pull Request 建议，文档清晰度` 


- **请求删除 Embedding 打印输出**：一名成员请求从文档中删除 **embedding 打印输出**，因为它占用了过多空间并影响了可读性。
   - 他们链接了一个 [GitHub issue](https://github.com/run-llama/llama_index/issues/17735)，强调了该 **文档问题**，并建议应将其移除以提高清晰度。
- **Pull Request 建议**：另一名成员确认了该请求，并提议创建一个 Pull Request (PR) 来解决移除 embedding 打印输出的问题。
   - 他们表示，如果原始请求者不想继续处理 PR，他们愿意接手。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/issues/17735">[文档]: Postgres Vector Store · Issue #17735 · run-llama/llama_index</a>：文档问题描述：Embeddings 打印输出占用过多空间并影响可读性。为了提高清晰度和文档可用性，应移除 embedding 打印输出。文档链接 ht...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/">Postgres Vector Store - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1336988389635002400)** (6 messages): 

> `LLM 在分类中的应用，ML 中的延迟要求，针对噪声数据的复合流水线` 


- **LLM 擅长分类但在处理噪声时表现不佳**：一名成员强调，虽然 **LLM** 在分类方面非常有效，但 **噪声数据** 需要额外的技术（如 dense embeddings 和 autoencoder rerankers）来增强性能。
   - 这表明在处理具有挑战性的数据环境时，可能需要更复杂的方法。
- **使用 LLM 时的延迟担忧**：讨论显示，虽然 LLM 可以很好地进行分类，但由于其处理限制，在具有严格 **延迟要求** 的场景中使用它们可能并不实际。
   - 对话得出的结论是，LLM 的适用性实际上取决于给定应用程序的具体延迟约束。
- **重新定义 ML 解决方案的业务需求**：一名成员提到，在将问题转化为 ML 问题时，存在 **错失的机会**，即未能正确界定业务需求。
   - 他们指出，从一开始就应该很明显，如果低延迟至关重要，传统的 LLM 可能不是最佳选择。


  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1336934498247376939)** (6 messages): 

> `微调错误，系统设计面试题` 


- **微调错误与 Batch Size 限制**：一位用户报告了 **BadRequestError**（状态码：400），指出当前的训练配置超过了最大 **250 个训练步数 (training steps)** 的限制，且 **Batch Size** 限制设置为 16。
   - 有人担心这是否意味着微调的样本上限为 **4000 个**，因为一位成员指出以前并没有这种限制。
- **关于 AI/ML 系统设计问题的咨询**：一位成员询问是否有人有专门针对 **AI/ML** 的 **系统设计面试题**。
   - 另一位成员确认了该询问并将其引导至收集环节，标志着团队间的协作。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1336952403538477127)** (4 messages): 

> `工具使用模型的系统提示词，Hugging Face 数据集转换问题，数据集文件格式不匹配` 


- **对规范系统提示词的需求**：一位成员询问了经过微调的工具使用模型的 **规范系统提示词 (canonical system prompts)**，以确保它们能为函数调用 (function calls) 返回响应或 JSON。
   - 他们注意到 **Gorilla 论文** 中没有包含所使用的系统提示词，导致现有资源存在空白。
- **实验 Hugging Face 数据集**：一位成员表示希望通过转换数据并在 **Hugging Face** 上利用 `datasets.map` 来更轻松地进行实验。
   - 这表明正在推动增强数据集的功能和可访问性以进行实验。
- **Hugging Face 数据集格式问题**：一位成员指出该数据集后缀是 **.json** 格式，但其内容实际上是 **jsonl** 格式，这导致了 Hugging Face 的识别问题。
   - 他们建议将文件后缀更改为 **.jsonl** 并修改数据集配置文件，以潜在地解决该问题。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

batmanosama: https://arxiv.org/abs/2502.02508
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1336822412875927653)** (2 messages): 

> `Git 仓库，Colab Notebook` 


- **关于 Git 仓库的询问**：一位成员询问是否有其工作的 **Git 仓库** 可供使用。
   - 该询问表明了对访问与项目相关的代码或资源的兴趣。
- **分享 Colab Notebook**：针对 Git 仓库的查询，一位成员提供了一个 [Colab notebook](https://colab.research.google.com/drive/1OXmTKexR9gX33DXRNEAe3dNuxkLXnutX?usp=sharing) 的链接。
   - 该 Notebook 可能与讨论相关，可以通过 **登录** 访问。



**提到的链接**：<a href="https://colab.research.google.com/drive/1OXmTKexR9gX33DXRNEAe3dNuxkLXnutX?usp=sharing">Google Colab</a>：未找到描述

  

---


---


{% else %}


> 完整的频道详细分类已在邮件中截断。
> 
> 如果你想查看完整的分类，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})!
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢支持！

{% endif %}