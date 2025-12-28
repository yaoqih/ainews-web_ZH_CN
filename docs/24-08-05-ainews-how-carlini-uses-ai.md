---
companies:
- groq
- intel
- deepmind
- box
- figure-ai
- openai
- google
- meta-ai-fair
- nvidia
- stability-ai
- runway
date: '2024-08-05T23:43:14.094795Z'
description: '**Groq** 的股东净资产在其他公司下跌之际逆势上涨，**英特尔首席执行官**对此表示担忧。**DeepMind** 的 **Nicholas
  Carlini** 因其大量的 AI 著作而受到关注和争议，其中包括一篇关于 AI 使用的 8 万字论文以及一个针对大语言模型的基准测试。**Chris Dixon**
  对“AI 寒冬”的怀疑论发表了看法，强调了其长期影响。**Box** 推出了一个 AI API，用于从文档中提取结构化数据，突显了由大语言模型驱动的解决方案的潜力和风险。


  近期 AI 领域的发展还包括：**Figure AI** 推出了先进的人形机器人 Figure 02；**OpenAI** 为 ChatGPT 推出了具备情感检测功能的“高级语音模式”（Advanced
  Voice Mode）；**Google** 开源了 **Gemma 2 2B** 模型，其性能可与 GPT-3.5-Turbo-0613 媲美；**Meta
  AI Fair** 发布了用于实时物体追踪的 Segment Anything Model 2 (SAM 2)；**NVIDIA** 展示了 Project GR00T，支持通过
  Apple Vision Pro 进行人形机器人远程操作；**Stability AI** 推出了用于快速生成 3D 资产的 Stable Fast 3D；**Runway**
  则发布了用于 AI 文本生成视频的 Gen-3 Alpha。'
id: 8ed20429-df08-4f5d-8ae1-db6e30ed9a10
models:
- gemma-2-2b
- gpt-3.5-turbo-0613
- mixtral-8x7b
- gen-3-alpha
- segment-anything-model-2
- stable-fast-3d
original_slug: ainews-how-carlini-uses-ai
people:
- nicholas-carlini
- chris-dixon
- rasbt
title: '**Carlini 如何使用 AI** 或 **卡里尼如何使用人工智能**'
topics:
- benchmarking
- adversarial-attacks
- large-language-models
- text-generation
- multimodality
- robotics
- emotion-detection
- structured-data-extraction
- real-time-processing
- teleoperation
- 3d-generation
- text-to-video
---

<!-- buttondown-editor-mode: plaintext -->**保持开放的心态是你唯一需要的。**

> 2024年8月2日至8月5日的 AI 新闻。我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 服务区（**249** 个频道，**5970** 条消息）。预计节省阅读时间（以 200wpm 计算）：**685 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

恭喜 [Groq 股东的净资产在上涨](ttps://x.com/groqinc/status/1820422643004424631?s=46)，而其他人的都在下跌（以及 [Intel 的 CEO 在祈祷](https://x.com/datnofact/status/1820213413319962975?s=61)）。DeepMind 的 Nicholas Carlini 作为最具深度的、具有研究背景的 AI 公开作者之一，正获得越来越多的认可（以及 [批评](https://nicholas.carlini.com/writing/2024/why-i-attack.html)）。今年，他正从其惯常的 [对抗性研究领域](https://arxiv.org/abs/2311.17035) 扩展开来，发布了其 [大语言模型基准测试 (benchmark for large language models)](https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html)，并在本周末凭借一篇 [关于他如何使用 AI 的 8 万字长文](https://nicholas.carlini.com/writing/2024/how-i-use-ai.html) 引起了轰动，我们理所当然地 [使用了 AI 来对其进行总结](https://claude.ai/chat/157b11d3-cec1-4a97-877f-da829d6f2a39)：

 
![image.png](https://assets.buttondown.email/images/820a8c18-851b-4862-9821-fce5d442da5f.png?w=960&fit=max)
 

以及使用案例：

 
![image.png](https://assets.buttondown.email/images/1359acfe-2c17-4d21-8ec8-5fa2943fdd11.png?w=960&fit=max)
 

令人印象深刻的是，他说这还不到他所经历的 LLM 使用案例的 “2%”（如果他把所有内容都列出来，那将是 400 万字的作品）。

Chris Dixon 以其 [名言](https://cdixon.org/2013/03/02/what-the-smartest-people-do-on-the-weekend-is-what-everyone-else-will-do-during-the-week-in-ten-years) “最聪明的人在周末做的事情，就是十年后其他人在周中会做的事情” 而闻名。当人们在 [AI 寒冬将至](https://www.latent.space/p/q2-2024-recap) 的舆论中推波助澜，声称它在工作中尚未产生足够的衡量影响时，他们可能只是过于关注短期利益了。其中的每一个案例至少都值得打磨成工具，甚至可以成立一家初创公司。

---

> 新增：我们正在尝试投放一些小巧且得体的广告，专门为 AI Engineers 提供帮助。请点击以支持我们的赞助商，并回复邮件告诉我们您想看到的内容！

**[由 Box 赞助]** Box 存储文档。Box 还可以从这些文档中 **提取结构化数据**。[这是如何使用 Box AI API 实现的方法。](https://medium.com/box-developer-blog/extracting-structured-data-using-box-ai-01408437352d)。 

**swyx 评论**：S3 的僵化正是 Box 的机会所在。“多模态 Box” 的理念——放入任何东西，输出结构化数据——使所有数字内容都能被机器读取。特别赞扬这篇博文，因为它还展示了这种方案——就像任何 LLM 驱动的方案一样——可能会意外失败！

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，从 4 次运行中择优。

**AI 与机器人进展**

- **Figure AI**：[@adcock_brett](https://twitter.com/adcock_brett/status/1820128340138713547) 宣布推出 Figure 02，称其为“地球上最先进的人形机器人”，更多细节即将公布。

- **OpenAI**：开始向部分用户推送 ChatGPT 的 [“Advanced Voice Mode”](https://twitter.com/adcock_brett/status/1820128362708307982)（高级语音模式），其特点是具有情感检测能力的自然、实时对话 AI。

- **Google**：发布并开源了 [Gemma 2 2B](https://twitter.com/adcock_brett/status/1820128475354730875)，在 LMSYS Chatbot Arena 中评分达到 1130 分，尽管体积小得多，但性能与 GPT-3.5-Turbo-0613 和 Mixtral-8x7b 相当。

- **Meta**：推出了 [Segment Anything Model 2 (SAM 2)](https://twitter.com/adcock_brett/status/1820128497819373741)，这是一个用于在视频帧中进行实时对象识别和跟踪的开源 AI 模型。

- **NVIDIA**：Project GR00T 展示了一种[扩展机器人数据的新方法](https://twitter.com/adcock_brett/status/1820128520338591847)，利用 Apple Vision Pro 进行人形机器人远程操作（teleoperation）。

- **Stability AI**：推出了 [Stable Fast 3D](https://twitter.com/adcock_brett/status/1820128452772589993)，可在 0.5 秒内从单张图像生成 3D 资产。

- **Runway**：宣布其 AI 文本生成视频模型 [Gen-3 Alpha](https://twitter.com/adcock_brett/status/1820128565494526267) 现在可以从图像创建高质量视频。


**AI 研究与开发**

- **Direct Preference Optimization (DPO)**：[@rasbt](https://twitter.com/rasbt/status/1820096879440662972) 分享了一个从零开始实现的 DPO，这是一种将 Large Language Models (LLM) 与用户偏好对齐的方法。

- **MLX**：[@awnihannun](https://twitter.com/awnihannun/status/1820139615216648658) 建议使用 lazy loading（延迟加载）来降低 MLX 中的峰值内存占用。

- **Modality-aware Mixture-of-Experts (MoE)**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1820092416537354247) 讨论了 Meta AI 关于模态感知 MoE 架构的论文，该架构用于预训练混合模态、早期融合（early-fusion）的语言模型，实现了显著的 FLOPs 节省。

- **Quantization**：[@osanseviero](https://twitter.com/osanseviero/status/1820124474965897466) 分享了五个学习 AI 模型 Quantization（量化）的免费资源。

- **LangChain**：[@LangChainAI](https://twitter.com/LangChainAI/status/1820206325021946297) 推出了 Denser Retriever，这是一款企业级 AI 检索器，旨在简化 AI 在应用程序中的集成。

**AI 工具与应用**

- **FarmBot**：[@karpathy](https://twitter.com/karpathy/status/1820167525575115045) 将 FarmBot 比作“食物领域的太阳能电池板”，强调了其在后院自动化食物生产的潜力。

- **Composio**：[@llama_index](https://twitter.com/llama_index/status/1820224063174053984) 提到 Composio 是一个面向生产环境的 AI Agent 工具集，包含超过 100 种适用于各种平台的工具。

- **RAG 部署**：[@llama_index](https://twitter.com/llama_index/status/1820133457114370259) 分享了关于在 Google Kubernetes Engine 上部署和扩展“与代码对话”应用的全面教程。

- **FastHTML**：[@swyx](https://twitter.com/swyx/status/1820124350923616449) 宣布开始使用 FastHTML 开发应用，将 AINews 转化为网站。

**AI 伦理与社会影响**

- **AI 监管**：[@fabianstelzer](https://twitter.com/fabianstelzer/status/1820090239207305335) 将当前的 AI 监管努力与奥斯曼帝国历史上对印刷机的限制进行了类比。

- **AI 与失业**：[@svpino](https://twitter.com/svpino/status/1820168471746892247) 幽默地评论了关于 AI 取代工作的周期性预测。

**迷因与幽默**

- [@nearcyan](https://twitter.com/nearcyan/status/1820205826742829372) 分享了一个关于 Mark Zuckerberg 公众形象变化的迷因。

- [@nearcyan](https://twitter.com/nearcyan/status/1820207471849582877) 开了关于偶像化科技公司 CEO 的玩笑。

- [@lumpenspace](https://twitter.com/lumpenspace/status/1820233922287919263) 对将 Diffusion（扩散）解释为频域中的 Autoregression（自回归）发表了幽默评论。


---

# AI Reddit 摘要

## /r/LocalLlama 回顾

**主题 1：LLM 训练中的数据质量与数量之争**

- **鉴于这是一个发展如此迅速的领域，你认为 LLM 两年后会发展到什么程度？** ([Score: 61, Comments: 101](https://reddit.com//r/LocalLLaMA/comments/1ejqqyv/since_this_is_such_a_fast_moving_field_where_do/))：在接下来的**两年**中，发帖者预计 **Large Language Models (LLMs)** 将取得重大进展，特别是在**模型效率**和**移动端部署**方面。他们专门询问了实现 **GPT-4** 级别能力所需的**参数量 (parameter count)** 是否可能减少，以及在**智能手机**上运行复杂 LLM 的可行性。
  - 随着原生数据耗尽，**合成数据生成 (Synthetic data generation)** 正变得至关重要。**Llama 3 论文**展示了成功的技术，包括将生成的代码通过 **Ground Truth** 源运行，以在不引发模型崩溃的情况下增强预测能力。
  - 研究人员预计**多模态领域**将有所增长，模型将整合**图像/音频编码器**以更好地理解世界。未来的发展可能包括 **4D 合成数据**（与文本、视频和图片相关的 xyzt 数据）以及改进的**上下文处理 (context handling)** 能力。
  - **模型效率**有望显著提高。预测建议 **300M 参数模型**的表现将超过今天的 7B 模型，并且在**加速器硬件**和 **ASIC** 开发的推动下，两年内有望在智能手机上运行 **GPT-4 级别能力**的模型。

- **“我们将耗尽数据”是真的吗？** ([Score: 61, Comments: 67](https://reddit.com//r/LocalLLaMA/comments/1ekdly9/we_will_run_out_of_data_really/))：该帖子质疑了训练 LLM 数据即将耗尽的观点，理由是互联网包含 **64 ZB** 的数据，而目前的模型训练仅使用了 **TB 级别**的数据。根据 **Common Crawl** 截至 **2023 年 6 月**的数据，公开可访问的网络包含 **约 30 亿个网页**和 **约 400 TB** 的未压缩数据，但这仅代表互联网总数据的一小部分，大量数据存在于私有组织、付费墙后或屏蔽爬取的网站上。作者建议，未来的模型训练可能会涉及购买大量的私营部门数据，而不是使用生成的数据，并指出随着更多国家采用互联网技术和 **IoT** 使用的扩大，数据量将继续增加。
  - 用户认为，**免费获取**和**经济上易于获取**的数据可能会耗尽，因为公司意识到了其数据的价值并将其封锁。互联网数据的质量也受到质疑，有人建议从训练数据中**移除 Reddit** 提高了模型性能。
  - **64 ZB** 这一数字代表全球总存储容量，而非可用的文本数据。像 **GPT-4** 这样的当前模型仅在 **13 万亿个 token**（约 4 万亿个唯一 token）上进行了训练，而据估计，公开可用的高质量文本 token 超过 **200 万亿个**。
  - 互联网数据的很大一部分可能是**视频内容**，**Netflix** 在 **2022 年**占所有互联网流量的 **15%**。用户讨论了这些数据对语言建模的价值，并建议关注高质量、精选的数据集，而非原始数量。

**主题 2：新兴 AI 技术及其现实世界应用**

- **[逻辑谬误计分板](https://v.redd.it/fx4jvpkqtrgd1)** ([Score: 118, Comments: 61](https://reddit.com//r/LocalLLaMA/comments/1ekf1vl/logical_fallacy_scoreboard/))：该帖子提议为政治辩论建立一个使用 **Large Language Models (LLMs)** 的**实时逻辑谬误检测系统**。该系统将实时分析辩论，识别逻辑谬误，并向观众展示**“逻辑谬误计分板”**，这有可能提高政治话语的质量，并帮助观众批判性地评估候选人提出的论点。
  - 用户对用于现场辩论的**实时版本**工具表示感兴趣，其中一人建议为所有候选人建立一个**“实时胡说八道追踪器 (live bullshit tracker)”**。开发者计划在接下来的辩论中运行该系统（如果 **Trump** 不退出的话）。
  - 人们对 **AI 准确检测谬误的能力**表示担忧，并举例说明了模型判断中的不一致和潜在偏见。一些人建议使用**更小的、经过微调的 LLM** 或基于 **BERT 的分类器**，而不是大型预训练模型。
  - 该项目因其**捍卫民主**的潜力而受到赞扬，而其他人则提出了改进建议，如**追踪未解决的陈述**、**对谎言进行分类**，以及将 **70B 模型**蒸馏至 2-8B 以获得实时性能。用户还要求对 **Biden** 和 **Harris** 等其他政治家进行分析。

## 全球 AI Reddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型能力与进展**

- **Flux AI 展示了令人印象深刻的文本和图像生成能力**：r/StableDiffusion 上的多篇帖子展示了 Flux AI 生成高度详细的产品广告的能力，具有准确的文本放置和品牌一致性。示例包括 [Tide PODS 口味珍珠奶茶广告](https://www.reddit.com/r/StableDiffusion/comments/1ekbuka/flux_adproduct_test/) 和 [Hot Pockets "Sleepytime Chicken" 包装盒](https://www.reddit.com/r/StableDiffusion/comments/1ekbuka/flux_adproduct_test/)。用户注意到 Flux 与 Midjourney 等其他模型相比，具有更优越的文本生成能力。

- **OpenAI 决定不对 ChatGPT 输出添加水印**：OpenAI [宣布他们不会对 ChatGPT 生成的文本实施水印技术](https://www.reddit.com/r/OpenAI/comments/1ekh1uv/openai_wont_watermark_chatgpt_text_because_its/)，理由是担心对用户产生潜在的负面影响。这一决定引发了关于检测方法、学术诚信以及透明度与用户保护之间平衡的讨论。

**AI 伦理与社会影响**

- **关于 AI 对就业影响的辩论**：r/singularity 上的一篇 [高赞帖子](https://www.reddit.com/r/singularity/comments/1ek3h85/the_impact_of_ai_on_jobs/) 讨论了 AI 对就业的潜在影响，反映了对劳动力中断的持续担忧。

- **AI 驱动的验证与 Deepfakes**：r/singularity 的一篇帖子强调了 [用于验证目的的 AI 生成图像日益复杂](https://www.reddit.com/r/singularity/comments/1ekdl7q/brace_yourself_ai_powered_verification_is_on_the/)，引发了关于数字身份以及区分真实内容与 AI 生成内容挑战的问题。

**AI 在教育与开发中的应用**

- **AI 导师的潜力**：r/singularity 上的一篇 [详细帖子](https://www.reddit.com/r/singularity/comments/1ejz7xa/ai_tutors_could_turn_every_child_into_a_genius/) 探讨了 AI 导师可能增强儿童学习能力的概念，并与历史上强化教育方法的案例进行了类比。

**AI 行业与市场趋势**

- **Ben Goertzel 谈生成式 AI 的未来**：AI 研究员 Ben Goertzel [预测](https://www.reddit.com/r/singularity/comments/1ejwgb0/ben_goertzel_i_dont_think_the_genai_bubble_will/) 生成式 AI 市场将继续增长，理由是高价值应用的快速发展。


---

# AI Discord 摘要

> 摘要之摘要的总结

**1. LLM 进展**

- **Llama 3 性能问题**：用户报告了 **Llama 3** 分词（tokenization）方法的问题，特别是 **EOS** 和 **BOS** token 的使用导致了推理挑战。参与者推测，推理中缺失 token 可能会导致训练期间出现分布外（out-of-distribution）上下文，从而促使人们重新评估文档。
  - 成员们一致认为有必要重新评估文档以解决这些分词 bug，并强调了准确处理 token 的重要性。
- **Claude AI 提供代码修复**：成员们讨论了使用 **Claude AI** 上传 `output.json` 以在没有文件访问权限的情况下进行代码修复的方法，如 [这篇 Medium 文章](https://medium.com/@mbonsign/codemapper-your-ais-guide-to-understanding-code-ef2bda7f333e) 所述。尽管具有潜力，但人们对这种方法的实证效果仍持怀疑态度。
  - *人们对这种方法的实证效果仍持怀疑态度*，这凸显了需要更多基于证据的结果来验证其效用。


**2. 模型性能优化**

- **优化 LLM 推理速度**：提升 **LLM** 推理速度的建议包括使用 **torch.compile** 以及将性能与 **vLLM** 等工具进行比较。持续的讨论凸显了人们对提高大语言模型效率和性能的兴趣。
  - 成员们对在处理大语言模型时增强效率表现出浓厚兴趣，并探索了各种工具和技术。
- **Mojo 增强数据处理流水线**：讨论强调了 **Mojo** 在将分析与数据库工作负载集成方面的潜力，通过 **JIT** 编译和直接文件操作实现更快的数据处理。
  - 成员们提到了与 **PyArrow** 和 **Ibis** 的兼容性，暗示了 **Mojo** 框架内强大的数据生态系统具有广阔的前景。


**3. 微调挑战**

- **微调多语言模型的挑战**：用户分享了微调 **Llama 3.1** 和 **Mistral** 等模型处理多样化数据集的经验，遇到了由于提示词（prompt）格式可能不正确导致的输出相关性问题。建议敦促恢复到标准提示词格式，以确保正确处理数据集。
  - 参与者强调了使用标准格式以避免问题的重要性，凸显了统一提示词格式的必要性。
- **LoRA 训练问题**：一位用户报告了在使用 **SFTTrainer** 尝试通过拼接文本和标签来格式化数据集后效果不佳，怀疑可能存在配置错误。澄清指向了正确的列使用方式，但仍未能解决根本问题。
  - 澄清指向了正确的列使用方式，但未能解决根本问题，表明需要对数据集配置进行进一步调查。


**4. 开源 AI 发展**

- **DistillKit 介绍**：**Arcee AI** 发布了 **DistillKit**，这是一个用于从大模型中蒸馏知识以创建更小、更强大模型的开源工具。该工具包结合了传统训练技术与新颖方法，以优化模型效率。
  - 该工具包专注于优化模型，使其高效且易于获取，结合了传统训练技术与新颖的蒸馏方法。
- **OpenRouter 发布新模型**：**OpenRouter** 推出了令人印象深刻的新模型，包括 **Llama 3.1 405B BASE** 和 **Mistral Nemo 12B Celeste**，可以在其 [模型页面](https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free) 查看。**Llama 3.1 Sonar** 系列的加入进一步扩展了应用能力。
  - 新加入的模型迎合了多样化的需求，并根据社区反馈进行持续更新，增强了 **OpenRouter** 产品的实用性。


**5. 多模态 AI 创新**

- **CatVTON 重新定义虚拟试穿方法**：最近的一篇 [arXiv 论文](https://arxiv.org/abs/2407.15886) 介绍了 **CatVTON**，这是一种通过直接拼接服装图像显著降低训练成本的方法。这一创新有望实现逼真的服装迁移，彻底改变虚拟试穿技术。
  - 该方法消除了对 **ReferenceNet** 和额外图像编码器的需求，在降低成本的同时保持了逼真的服装迁移效果。
- **Open Interpreter 语音识别提议**：一位用户提议实现一种母语语音识别方法，以促进英语与当地语言之间的翻译。他们对翻译错误提出了警告，称之为“*垃圾进，垃圾出（Garbage in, Garbage out）*”。
  - 该方法引发了对翻译错误潜在陷阱的担忧，强调了准确输入对确保可靠输出的重要性。

---

# 第一部分：Discord 高层级摘要

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 遭遇性能问题**：多位用户报告了 **LM Studio 0.2.31 版本**的问题，特别是应用程序启动困难和模型无法正确加载。建议降级到 **0.2.29** 等早期版本作为潜在的解决方法。
   - 用户确认性能不一致的问题仍然存在，敦促社区探索稳定版本以维持工作流。
- **模型下载速度受限**：用户在 LM Studio 网站下载时遇到了速度波动，有报告称速度被限制在 **200kbps**。建议由于典型的 AWS 限速问题，可以稍后重试或等待。
   - 对话强调了在高需求下载时段需要耐心，并进一步强调了检查连接稳定性的重要性。
- **AI 想控制你的电脑！**：关于 AI 模型（特别是 **OpenInterpreter**）是否能获得视觉能力来控制 PC 的讨论引起了关注，这指向了当前 AI 理解能力的局限性。参与者对这种集成可能带来的不可预见行为表示担忧。
   - 辩论强调了在本地系统上实施 AI 控制机制之前需要进行仔细考虑。
- **多模态模型引发关注**：用户对 **AnythingLLM** 可用的多模态模型产生了浓厚兴趣，重点讨论了对无审查（uncensored）模型的探索。推荐使用 [UGI Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard) 等资源进行能力对比。
   - 参与者强调了社区驱动探索先进模型以增强功能多样性的重要性。
- **双 GPU 配置引发不同观点**：关于双 **4090** 配置的讨论表明，将模型拆分到多张显卡上可以提高性能，但也提醒用户有效利用这些配置需要一定的编程要求。对于单块 4090 在处理大型模型时的吃力表现，担忧依然存在。
   - 成员们在考虑多 GPU 配置时，更倾向于讨论性能与易用性之间的平衡。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **深入探讨 Hugging Face 模型特性**：用户分享了关于 Hugging Face 上各种模型的见解，例如用于翻译的 **MarionMT** 和提供语言支持的 **TatoebaChallenge**。
   - 对模型局限性和**更好文档**必要性的担忧引发了更广泛的讨论。
- **加速 LLM 推理技术**：优化 LLM 推理成为热门话题，建议包括使用 **torch.compile** 以及使用 **vLLM** 等工具评估性能。
   - 成员们对在处理大型语言模型时提高效率表现出浓厚兴趣。
- **CatVTON 重新定义虚拟试穿方法**：最近的一篇 [arXiv 论文](https://arxiv.org/abs/2407.15886) 介绍了 **CatVTON**，这是一种通过直接拼接服装图像显著降低训练成本的方法。
   - 这一创新有望实现逼真的服装迁移，彻底改变虚拟试穿技术。
- **Diffusers 中 Gradient Checkpointing 的实现**：最近的更新现在包含了一种在 Diffusers 中设置 **gradient checkpointing** 的方法，允许在兼容模块中进行切换。
   - 这一增强功能有望优化模型训练期间的内存使用。
- **使用 NLP 识别表格中的关系**：成员们正在探索 NLP 方法，根据列描述和名称来确定表格之间的关系。
   - 这一探究表明在 NLP 关系建模领域需要进一步的探索。



---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Flux 模型在 GPU 上表现出色**：用户报告 **Flux model** 的图像生成速度在 **1.0 到 2.0 iterations per second** 之间，具体取决于 GPU 配置和模型版本。
   - 一些用户通过 **CPU offloading** 或 **quantization** 技术，在较低 VRAM 的配置上成功生成了图像。
- **ComfyUI 安装技巧**：讨论围绕在 **ComfyUI** 上安装 **Flux** 展开，建议使用 `update.py` 脚本而不是管理器进行更新。
   - 为新手分享了有用的安装指南，以帮助其顺利配置环境。
- **Stable Diffusion 模型对比**：参与者详细介绍了不同的 **Stable Diffusion models**：**SD1.5**、**SDXL** 和 **SD3**，指出各模型的优势，并将 **Flux** 定位为来自 **SD3 team** 的新成员。
   - 讨论中强调了 **Flux** 相比传统模型更高的资源需求。
- **RAM 与 VRAM 的对决**：**充足的 VRAM** 对 **Stable Diffusion performance** 至关重要，用户建议至少配备 **16GB VRAM** 以获得最佳效果，这比对高 RAM 的需求更重要。
   - 社区建议，虽然 RAM 有助于模型加载，但它不是影响生成速度的主要因素。
- **动画工具咨询**：参与者询问了用于视频内容生成的 **Animatediff** 等工具，寻求可用方法的最新更新。
   - 目前的建议指出，虽然 **Animatediff** 仍然有用，但针对类似任务可能会出现新的替代方案。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Epoch 8 准确率激增**：成员注意到训练过程中 Epoch 8 之后准确率分数出现意外激增，引发了对预期行为的疑问。
   - *Looks totally normal*（看起来完全正常），另一位成员安慰道，表示无需担心。
- **CUDA 在 DRL 中的挑战**：在为 **Deep Reinforcement Learning** 创建 CUDA 环境时出现了挫折，建议使用 [PufferAI](https://pufferai.github.io/) 以获得更好的并行性。
   - 参与者强调了设置中涉及的复杂性，强调了对强大 tooling 的需求。
- **寻求 ML 冬季实习**：一名用户正在紧急寻找 2025 年 1 月开始的 **winter internship**，重点关注 **ML systems** 和 **applied ML**。
   - 该用户强调了之前的实习经历和正在进行的 open source 贡献。
- **对 AI 泡沫破裂的担忧**：关于潜在 **AI bubble** 的猜测开始流传，对于投资的长期潜力存在截然不同的看法。
   - 参与者指出，研究成果与盈利之间的滞后时间是一个核心担忧。
- **Llama 3 Tokenization 问题**：讨论了 **Llama 3** 在 tokenization 方法上的不一致，特别是关于 EOS 和 BOS token 的使用导致了推理挑战。
   - 参与者一致认为需要重新评估文档以解决这些 tokenization bugs。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 安装问题依然存在**：用户在本地安装 **Unsloth** 时遇到错误，特别是关于 **Python compatibility** 和 **PyTorch** 安装，解决方法包括升级 pip。
   - 一些用户通过重新连接 Colab runtime 并验证库安装解决了问题。
- **微调多语言模型的挑战**：用户分享了使用多样化数据集微调 **Llama 3.1** 和 **Mistral** 等模型的经验，由于可能不正确的 prompt 格式遇到了输出相关性问题。
   - 建议敦促恢复到标准 prompt 格式以确保正确的数据集处理。
- **LoRA 训练在数据集格式上受阻**：一名用户报告在尝试使用拼接文本和标签格式化数据集后，其 **SFTTrainer** 结果不佳，质疑是否存在配置错误。
   - 澄清指向了正确的列使用，但未能解决根本问题。
- **加载大型模型的内存问题**：在单个 GPU 上加载 **405B Llama-3.1** 模型导致了内存挑战，促使用户注意到多 GPU 的必要性。
   - 这凸显了一个共识，即更大的模型需要更多的计算资源来加载。
- **Self-Compressing Neural Networks 优化模型大小**：关于 [Self-Compressing Neural Networks](https://arxiv.org/abs/2301.13142) 的论文讨论了在 loss function 中使用字节大小来实现显著缩减，仅需 **3% 的 bits** 和 **18% 的 weights**。
   - 该技术声称可以在不需要专门硬件的情况下提高训练效率。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 的浏览能力遭到质疑**：用户报告了在使用 Perplexity 浏览时的参差不齐的体验，指出在获取最新信息方面存在困难，且在使用 Web App 时表现出奇怪的行为。
   - 对话强调了模型响应的不一致性，特别是在对于技术应用至关重要的代码查询等任务中。
- **Llama 抗体在 HIV 研究中取得突破**：佐治亚州立大学的研究人员设计了一种混合抗体，将 Llama 衍生的纳米抗体与人类抗体结合，中和了超过 **95%** 的 HIV-1 毒株。
   - 这种混合方法利用了 Llama 纳米抗体的独特属性，能够更好地进入易逃逸的病毒区域。
- **对模型性能的担忧：Llama 3.1 对比预期**：用户发现 **Llama 3.1-sonar-large-128k-online** 模型在日语测试中表现不佳，提供的结果准确性低于 **GPT-3.5**。
   - 这引发了开发专门针对日语优化的 sonar-large 模型的呼声，以提高输出质量。
- **Uber One 订阅引发不满**：一位用户批评 Uber One 优惠仅限于 Perplexity 新账号，指出这更多是一种用户获取策略，而非真正的福利。
   - 关于通过创建账号来利用促销活动的辩论，提出了 AI 服务中用户管理的重要问题。
- **Perplexity 的 API 质量问题**：多位用户分享了 **Perplexity API** 的问题，提到在查询近期新闻时响应不可靠且返回低质量结果。
   - 用户对 API 输出感到沮丧，这些输出经常显得被无意义的内容“污染”，从而迫切要求改进模型和 API 的性能。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI DevDay 开启巡回活动**：OpenAI 将于今年秋季在**旧金山**、**伦敦**和**新加坡**举办 **DevDay** 巡回活动，提供动手实践环节、演示和最佳实践分享。参与者将有机会与工程师见面，并了解开发者如何利用 OpenAI 进行构建；更多信息请访问 [DevDay 官网](https://openai.com/devday/)。
   - 参与者将在活动期间与工程师互动，增强技术理解和社区参与。
- **AI 全球威胁讨论**：关于将 AI 视为全球威胁的看法展开了激烈辩论，重点讨论了政府针对开源 AI 与更优越的闭源模型的行为。*随着 AI 能力的扩展，对潜在风险的担忧日益增加*。
   - 随着关于 AI 影响的观点变得日益两极分化，这一问题得到了强调。
- **GPT-4o 图像生成见解**：讨论揭示了对 **GPT-4o** 图像 Token 化能力的见解，图像有可能被表示为 Token。然而，在当前的实现中，实际应用和局限性仍然模糊不清。
   - 提到的资源包括 [Greg Brockman 的一条推文](https://x.com/gdb/status/1790869434174746805)，讨论了团队在 GPT-4o 图像生成方面的持续工作。
- **Prompt Engineering 障碍**：用户报告在使用 ChatGPT 进行 Prompt 编写时，在产出高质量结果方面面临持续挑战，这经常导致挫败感。困难在于如何定义*什么是高质量输出*，这使得交互变得复杂。
   - 成员们分享了相关经验，说明了编写清晰、开放式 Prompt 对改善结果的重要性。
- **AI 图像生成中的多样性与偏见**：关于 AI 生成图像中的*种族代表性*问题引发了关注，特定的 Prompt 因服务条款准则而遭到拒绝。成员们交流了成功的策略，通过在 Prompt 中明确包含多种族背景来确保多样化的代表性。
   - 讨论还揭示了负面提示（Negative Prompting）的影响，即试图限制某些特征反而产生了不理想的结果。建议集中在编写积极、详细的描述以提高输出质量。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI 工程师需求激增**：随着公司寻求通才技能，特别是能够将 AI 集成到实际应用中的 Web 开发者，对 **AI 工程师** 的需求正在飞速增长。
   - 这种转变凸显了高水平 ML 专业知识的缺口，促使 Web 开发者在 AI 项目中担任关键角色。
- **Groq 在 D 轮融资中筹集 6.4 亿美元**：**Groq** 获得了由 BlackRock 领投的 **6.4 亿美元** D 轮融资，使其估值提升至 **28 亿美元**。
   - 资金将用于扩大产能并加强下一代 AI 芯片的开发。
- **NVIDIA 的抓取伦理备受指责**：泄露的信息揭露了 **NVIDIA** 大规模的 AI 数据抓取行为，每天收集相当于“一个人一生长度”的视频，引发了严重的伦理担忧。
   - 这种情况引发了关于此类激进数据获取策略在法律和社区影响方面的辩论。
- **比较 Cody 和 Cursor**：讨论强调了 **Cody** 相比 **Cursor** 具有更优越的上下文感知能力，Cody 允许用户索引代码库以获得相关的回复。
   - 用户欣赏 Cody 的易用性，而认为 Cursor 的上下文管理繁琐且复杂。
- **Claude 推出同步文件夹功能**：据报道，Anthropic 正在为 Claude 开发 **Sync Folder** 功能，支持从本地文件夹批量上传，以便更好地进行项目管理。
   - 该功能预计将简化 Claude 项目中文件的组织和工作流。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **关于 LLM as Judge 和数据集生成的建议**：一位用户询问了有关 **LLM as Judge** 和合成数据集生成（重点是指令和偏好数据）当前趋势的必读内容，并强调了 **WizardLM 的最新两篇论文** 作为起点。
   - 这次讨论将 **LLM** 的进步定位为理解模型应用转变的关键。
- **对 Claude Sonnet 3.5 的担忧**：用户报告了 **Claude Sonnet 3.5** 的问题，指出与其前代产品相比，其表现不佳且错误率增加。
   - 这引发了人们对近期更新的有效性及其对核心功能影响的质疑。
- **DistillKit 介绍**：Arcee AI 发布了 **DistillKit**，这是一个开源工具，用于从大型模型中蒸馏知识以创建更小、更强大的模型。
   - 该工具包结合了传统的训练技术与新颖的方法，以优化模型效率。
- **轻松进行高效的 VRAM 计算**：分享了一个用于根据 bits per weight 和上下文长度估算 VRAM 需求的 Ruby 脚本，可在 [此处](https://gist.github.com/jrruethe/8974d2c8b4ece242a071d1a1526aa763) 获取。
   - 该工具帮助用户确定 LLM 模型中的最大上下文和 bits per weight，简化了 VRAM 计算。
- **创新的 Mistral 7B MoE 化**：**Mistral 7B MoEified** 模型允许将单个层切分为多个专家 (experts)，旨在实现连贯的模型行为。
   - 这种方法使模型在处理过程中能够平等地共享可用的专家资源。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Chatroom 焕然一新**：**Chatroom** 已上线，支持本地聊天保存和简化的 UI，允许在 [OpenRouter](https://openrouter.ai/chat) 进行更好的房间配置。这个翻新后的平台提升了用户体验和易用性。
   - *用户可以探索新功能，以增强 Chatroom 内的互动。*
- **OpenRouter 发布新模型变体**：OpenRouter 推出了令人印象深刻的新模型，包括 **Llama 3.1 405B BASE** 和 **Mistral Nemo 12B Celeste**，可以在其 [模型页面](https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free) 查看。**Llama 3.1 Sonar family** 的加入进一步扩展了应用能力。
   - *新成员满足了多样化的需求，并根据社区反馈进行持续更新。*
- **Mistral 模型现已登陆 Azure**：**Mistral Large** 和 **Mistral Nemo** 模型现在可以通过 [Azure](https://openrouter.ai/models/mistralai/mistral-large) 访问，增强了它们在云环境中的实用性。此举旨在为用户提供更好的基础设施和性能。
   - *用户可以利用 Azure 的能力，轻松访问高性能 AI 模型。*
- **Gemini Pro 经历价格大调整**：**Google Gemini 1.5 Flash** 的价格将在 12 日减半，使其在与 **Yi-Vision** 和 **FireLLaVA** 等对手的竞争中更具优势。这一转变可能会促进更多用户参与自动标注（automated captioning）。
   - *社区反馈在塑造这一转变中起到了至关重要的作用，因为用户渴望更经济的选择。*
- **Multi-AI Answers 发布**：[Multi-AI answer 网站](https://www.producthunt.com/posts/aiswers-com) 在 OpenRouter 的支持下正式在 Product Hunt 上线。他们的团队鼓励社区进行 **点赞（upvotes）和建议**，以完善服务。
   - *发布期间的社区贡献体现了用户参与在开发过程中的重要性。*

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 加速数据处理流水线**：讨论强调了 **Mojo** 在将分析与数据库工作负载集成方面的潜力，通过 JIT 编译和直接文件操作实现更快的数据处理。
   - 成员们提到了与 **PyArrow** 和 **Ibis** 的兼容性，暗示了 **Mojo** 框架内强大的数据生态系统有着广阔的前景。
- **Elixir 令人困惑的错误处理**：成员们讨论了 Elixir 的挑战，即库返回错误原子（error atoms）或抛出异常，导致错误处理不规范。
   - 一段由 Chris Lattner 和 Lex Fridman 参与的 [YouTube 视频](https://www.youtube.com/watch?v=Iflu9zEJipQ) 详细阐述了异常与错误的区别，提供了进一步的背景信息。
- **Mojo 调试器缺乏支持**：一位成员确认 Mojo 调试器目前无法在 VS Code 中工作，并引用了一个现有的 [GitHub issue](https://github.com/modularml/mojo/issues/1829) 以寻求调试支持。
   - 调试工作流似乎依赖于 print 语句，表明需要改进调试工具。
- **Mojo SIMD 的性能问题**：有关 **Mojo** 在大型 SIMD 列表上操作性能的担忧浮出水面，这些操作在某些硬件配置上可能会出现延迟。
   - 有建议称，使用符合 CPU 处理能力的 SIMD 大小可以提高性能。
- **缺少 MAX Engine 对比文档**：一位用户报告称，很难找到将 **MAX Engine** 与 **PyTorch** 和 **ONYX** 进行对比的文档，特别是在 **ResNet** 等模型上。
   - 该查询凸显了寻求对比数据的用户在可用资源方面的空白。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Claude AI 提供代码修复**：成员们讨论了通过开启 **Claude AI** 的新对话并上传 `output.json`，使其能够直接提供代码修复而无需文件访问权限，详见这篇 [Medium 文章](https://medium.com/@mbonsign/codemapper-your-ais-guide-to-understanding-code-ef2bda7f333e)。
   - 尽管潜力巨大，但对于这种方法的经验有效性仍持怀疑态度。
- **通过架构增强性能**：新的架构，特别是针对**用户特定音频分类**的架构，可以通过**contrastive learning**等策略来维持用户不变特征，从而显著提高性能。
   - 此外，还讨论了针对 **3D data** 调整架构，以确保在变换下的性能。
- **音乐生成的最新技术 (SOTA)**：关于**音乐生成的 SOTA 模型**的咨询包括了围绕正在进行的 AI 音乐生成诉讼的讨论，成员们更倾向于本地执行而非外部依赖。
   - 这次对话反映了在音乐生成应用中增加控制权的日益增长的趋势。
- **关于 RIAA 和厂牌的见解**：审查了 **RIAA** 与音乐厂牌之间的关系，强调了它们如何影响艺术家报酬和行业结构，并要求更直接的补偿方式。
   - 艺术家相对于行业利润获得的版税微乎其微，这引发了担忧，暗示了对自我推广的推动。
- **用于高效 Embedding 管理的 HDF5**：关于使用 **HDF5** 从大型 embedding 数据集中加载批次的讨论仍在继续，反映了在简化数据管理技术方面的持续努力。
   - 这表明 AI 社区对高效数据利用有着持久的兴趣。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Ollama 内存错误记录**：一位用户报告了一个 **ValueError**，表明在调用检索链时模型内存不足，尽管在使用 **aya** (4GB) 和 **nomic-embed-text** (272MB) 等模型时 **GPU** 使用率较低。
   - 这引发了关于高性能设置中资源分配和内存管理的疑问。
- **混合 CPU 和 GPU 资源**：讨论集中在 **Ollama** 在重负载期间是否有效地利用了 **CPU** 和 **GPU**，用户注意到预期的回退到 **CPU** 的情况并未如期发生。
   - 用户强调了理解回退机制以防止推理瓶颈的重要性。
- **LangChain 内存管理见解**：分享了关于 **LangChain** 如何处理内存和对象持久化的见解，重点是评估跨会话内存效率的输入。
   - 确定适合内存存储的信息的查询成为了测试不同模型响应的试验场。
- **SAM 2 Fork：CPU 兼容性实践**：一位成员发起了一个兼容 **CPU** 的 **SAM 2 model** Fork，展示了提示分割和自动掩码生成，并期望实现 **GPU compatibility**。
   - 有关此项工作的反馈正在其 [GitHub](https://github.com/SauravMaheshkar/samv2) 仓库中积极征集。
- **快速启动你的 AI 语音助手**：一段名为 [“8 分钟创建自定义 AI 语音助手！- 由 ChatGPT-4o 驱动”](https://www.youtube.com/watch?v=iGX4ARuWZec) 的教程视频指导用户为其网站构建语音助手。
   - 创建者提供了一个 [演示链接](https://smart.sista.ai)，供潜在用户在注册服务前进行实际体验。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **使用 LlamaIndex 构建 ReAct Agents**：您可以利用 [LlamaIndex workflows](https://t.co/F0pPEyWJ2w) 从头开始创建 **ReAct agents**，以增强内部逻辑的可视化。
   - 这种方法允许您“拆解”逻辑，确保对 Agentic 系统有更深入的理解和控制。
- **面向 AI 工程师的 Terraform 助手**：使用 LlamaIndex 和 Qdrant Engine 为有志于成为 AI 工程师的人员开发 **Terraform 助手**，指南提供在 [此处](https://t.co/ASWNkixboK)。
   - 该教程提供了在 DevOps 领域集成 AI 的实用见解和框架。
- **使用 LlamaExtract 自动提取工资单**：[LlamaExtract](https://t.co/qoC9RU6Tfm) 通过自动化的 Schema 定义和元数据提取，实现对工资单的 **高质量 RAG**。
   - 这一过程显著增强了工资单文档的数据处理能力。
- **扩展 RAG 应用教程**：Benito Martin 概述了如何在 Google Kubernetes 上部署和扩展您的聊天应用，并在 [此处](https://t.co/ROsGNjhKEM) 强调了实用策略。
   - 该资源详细解决了 RAG 应用生产化方面内容稀缺的问题。
- **创新的 GraphRAG 集成**：**GraphRAG** 与 **LlamaIndex** 的集成增强了 **智能问答** 能力，正如一篇 [Medium 文章](https://medium.com/ai-advances/graphrag-with-llamaindex-unleashing-the-power-of-knowledge-graphs-for-intelligent-question-ea177a14623e) 中所讨论的。
   - 这种集成利用 **知识图谱（knowledge graphs）** 来提高 AI 响应的上下文关联性和准确性。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **湾区活动引发热议**：成员们表达了对 **湾区（Bay Area）** 即将举行的活动的更新需求，一些人提到了个人缺席的情况。
   - 持续的关注暗示了对当地聚会更好沟通的需求。
- **Noam Shazeer 缺乏认可**：讨论围绕自 2002 年以来一直是 Google 关键人物的 **Noam Shazeer** 缺少 **Wikipedia 页面** 展开。
   - 成员们反思道“Wikipedia 有时很荒谬”，强调了对有影响力的专业人士这种讽刺性的忽视。
- **对 30 Under 30 奖项有效性的质疑**：一位成员批评 **30 Under 30** 奖项更多是为了迎合圈内人而非真正的功绩，暗示是“特殊类型的人”在寻求这种认可。
   - 这引起了成员们的共鸣，他们指出这些奖项赋予的认可通常是表面上的。
- **关于使用 Nemotron 生成合成数据的辩论**：一场关于利用 **Nemotron** 重新制作 **合成数据（synthetic data）** 以微调 **Olmo** 模型的激烈讨论展开。
   - 讨论中提出了对 **Nemotron** 名称可能被挪用的担忧，并对 AI2 的发展轨迹提出了批评。
- **在噪声环境中 KTO 优于 DPO**：**Neural Notes 采访** 讨论了 KTO 在处理噪声数据时优于 DPO 的优势，表明其具有显著的性能提升。
   - 来自 **UCLA** 的适配报告称，KTO 在人类偏好测试中以 **70-30%** 的优势领先于 DPO。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **合成数据集引发争议**：成员们辩论了 **合成数据集（synthetic datasets）** 与原始数据集的有效性，指出它们可以加速训练，但可能面临对齐不齐和质量下降的风险。
   - 成员们对偏见表示担忧，呼吁更有目的地创建数据集，以避免生成数十亿张无用的图像。
- **FLUX 模型性能评价两极分化**：用户对 **FLUX** 模型生成艺术输出的能力持不同看法；一些人称赞其能力，而另一些人则表示失望。
   - 讨论指出更好的参数设置可以增强其性能，但对其艺术创作的整体实用性仍持怀疑态度。
- **CIFAR-10 验证准确率达到 80%**：在 CIFAR-10 数据集上仅使用 **36k 参数** 就实现了 **80% 的验证准确率**，将复数参数的实部和虚部作为独立部分处理。
   - 对架构和 Dropout 实现的调整解决了之前的问题，从而得到了一个更稳健的模型，几乎消除了过拟合。
- **模型训练中的伦理问题**：围绕在受版权保护的图像上进行训练的伦理影响，讨论变得激烈，引发了对合成数据集中 **版权洗白（copyright laundering）** 的担忧。
   - 一些人提出，虽然合成数据具有优势，但更严格的审查可能会对社区内的训练实践施加监管。
- **Stable Diffusion 数据集可用性受质疑**：一位用户对 **Stable Diffusion 数据集** 的不可用表示沮丧，这阻碍了他们的进展。
   - 同行澄清说，使用 Stable Diffusion 并不严格需要该数据集，并提供了替代解决方案。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **为 ChatmanGPT Stack 添加 Coding Agent**：一名成员正在为 ChatmanGPT Stack 寻求 **Coding Agent 推荐**，并建议将 **Agent Zero** 作为潜在选择。
   - *寻找有效的补充以增强编程交互。*
- **Golden-Retriever 论文概览**：分享的 **Golden-Retriever** 论文链接详细介绍了它如何通过改进传统 LLM 微调的挑战（特别是通过**基于反思的问题增强**步骤）来高效导航工业知识库。
   - 该方法通过在检索文档前澄清术语和上下文来提高检索准确性。更多信息请阅读 [Golden-Retriever 论文](https://arxiv.org/abs/2408.00798)。
- **Voice Lounge 中的 Livecoding**：一名成员宣布回归，并提到在 Voice Lounge 进行 **Livecoding** 会话，预示着即将开展协作编程。
   - *成员们期待在这种互动式设置中通力合作。*
- **AI NPC 响应与巡逻**：正在计划使用 **Oobabooga API** 在 C++ 游戏中开发 **AI 角色**进行玩家互动，重点是巡逻和响应功能。
   - 必要的组件包括修改 **'world' 节点**和扩展 NPC 类。
- **轻松导出 Discord 聊天记录**：一名用户使用 **DiscordChatExporter 工具**成功将 **Discord 频道**导出为 HTML 和 JSON，生成了 **463 个线程文件**。
   - 该工具简化了聊天记录的组织，便于未来参考。查看 [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter/releases)。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 在本地 LLM 上运行！**：一名用户使用 LM Studio 作为服务器，成功将 **Open Interpreter** 与本地 LLM 集成，并获得了 OI 系统提示词的访问权限。
   - 他们发现这种集成既有趣又富有启发性，为本地部署铺平了道路。
- **排查 Hugging Face API 集成问题**：用户在 **Open Interpreter** 中设置 Hugging Face API 集成时遇到挑战，尽管参考了文档，仍遇到各种错误。
   - 一名用户对获得的支持表示感谢，希望能解决他们的集成问题。
- **执行截图命令变得繁琐**：用户质疑为什么 **Open Interpreter** 会生成大量代码而不是直接执行截图命令，这引发了关注。
   - 使用 'screencapture' 命令的变通方法确认了功能性，缓解了一些挫败感。
- **提议多语言语音识别**：一名用户提议实现一种母语语音识别方法，以促进英语与当地语言之间的翻译。
   - 他们对翻译错误提出了警告，称之为 *垃圾进，垃圾出 (Garbage in, Garbage out)*。
- **Electra AI 在 Linux AI 领域展现潜力**：一名成员展示了 **Electra AI**，这是一个内置 AI 功能且免费使用的 Linux 发行版，强调了其集成的潜力。
   - 他们指出 Electra AI 提供三种版本：**Lindoz**、**Max** 和 **Shift**——全部免费提供。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 支持解决 CORS 问题**：为了解决计费页面的 **CORS 问题**，社区成员建议发送邮件至 [support@cohere.com](mailto:support@cohere.com) 寻求帮助，并在咨询中包含组织详情。
   - 该支持渠道旨在解决阻碍用户支付服务费用的问题。
- **GenAI 训练营寻求 Cohere 见解**：Andrew Brown 正在探索 **Cohere** 在免费 **GenAI 训练营**中的潜力，该训练营旨在今年覆盖 **5 万名参与者**。
   - 他强调需要文档之外的见解，特别是关于 **Cohere 的跨云能力 (cloud-agnostic capabilities)**。
- **保持一致性进行模型基准测试**：一名成员询问在对多个模型进行基准测试时如何保持**验证子集**的一致性，强调了受控比较的重要性。
   - 讨论强化了维持一致验证集以提高比较准确性的必要性。
- **在 Azure 模型上激活 Rerank**：Cohere 宣布 **Azure 模型**现已支持 **Rerank**，并具有集成到 **RAG 应用**的潜力，详见此 [博客文章](https://cohere.com/blog/introducing-rerank-3-on-microsoft-azure-ai)。
   - 成员们表现出更新工具包以供 Azure 用户使用 Rerank 的兴趣。
- **澄清 Cohere 模型混淆**：一名支付了 **Cohere API** 费用的用户发现只有 **Coral 模型**可用，并对如何访问 **Command R** 模型感到困惑。
   - 作为回应，一名成员澄清说 **Coral** 实际上就是 **Command R+** 的一个版本，以缓解用户的疑虑。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 0.9.2 引入了令人兴奋的特性**：最近发布的 **tinygrad 0.9.2** 带来了显著的更新，如 **faster gemv**、**kernel timing** 以及 **CapturedJit** 的改进。
   - 其他讨论包括对 **ResNet** 的增强和高级索引技术，这标志着性能优化迈出了重要一步。
- **在 Aurora 超级计算机上评估 tinygrad**：成员们讨论了在 **Aurora** 超级计算机上运行 **tinygrad** 的可行性，重点关注与 Intel GPU 的兼容性。
   - 虽然存在 **OpenCL** 支持，但有人对该平台上的性能限制和效率提出了疑问。
- **CUDA 性能相比 CLANG 令人失望**：成员们注意到 **CUDA** 的测试运行速度比 **CLANG** 慢，这引发了对可能存在的效率问题的调查。
   - 这种差异引发了关于 CUDA 执行完整性的重要问题，特别是在 **test_winograd.py** 中。
- **自定义 tensor kernel 引发讨论**：一位用户分享了在 tensor 上执行自定义 kernel 的兴趣，并参考了一个 [GitHub 文件](https://github.com/tinygrad/tinygrad/blob/da61dea1b2ca886b3de07e309efde2a78ac5682a/test/test_custom_function.py#L42-L43) 进行指导。
   - 这反映了 tinygrad 内部 tensor 操作的持续增强，展示了社区在实际实现中的参与度。
- **Bounties 激励 tinygrad 特性贡献**：社区开始讨论针对 tinygrad 改进的 **bounties**（悬赏），例如 **fast sharded llama** 和 **AMX** 的优化。
   - 这一举措鼓励开发者积极参与增强框架，旨在实现更广泛的功能。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **PPO 训练 Recipe 现已上线**：团队引入了一个端到端的 **PPO 训练 recipe**，将 RLHF 与 Torchtune 集成，详见 [GitHub pull request](https://github.com/pytorch/torchtune/pull/1005)。
   - *快去看看并尝试一下吧！*
- **Qwen2 模型支持已添加**：**Qwen2 模型支持** 现已包含在训练 recipe 中，**7B 模型** 已在 [GitHub pull request](https://github.com/pytorch/torchtune/pull/1143) 中提供。
   - 预计即将推出 **1.5B** 和 **0.5B** 版本！
- **LLAMA 3 在生成时遇到问题**：用户使用自定义配置成功运行了 **LLAMA 3 8B INSTRUCT 模型**，在 **20.62 GB** 内存占用下，以 **12.25 tokens/sec** 的速度在 **27.19 秒** 内生成了一个时间查询。
   - 然而，存在 **文本重复 10 次** 的问题，目前正在审查一个 [pull request](https://github.com/pytorch/torchtune/pull/1211) 以解决意外的结束 token 问题。
- **呼吁为 LLAMA 3 提供调试模式**：针对 LLAMA 3 生成输出中不显示 **所有 token** 的调试模式缺失问题，引发了关注。
   - 一位成员建议在生成脚本中添加一个参数可以解决此问题。
- **模型简介（Model Blurbs）维护焦虑**：成员们对保持 **model blurbs** 更新表示担忧，担心维护工作量可能过大。
   - 有人提议使用 **model card** 或白皮书中的快照作为最小化的简介方案。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **bitsandbytes 在 ROCm 上的安装已简化**：[最近的一个 PR](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1299) 实现了在 **ROCm** 上打包 **bitsandbytes** 的 wheel 文件，为用户简化了安装过程。
   - 该 PR 更新了 **ROCm 6.1** 的编译过程，以支持最新的 **Instinct** 和 **Radeon** GPU。
- **构建 AI 营养师需要数据集**：一位成员正在开发 **AI Nutritionist**，并考虑微调 **GPT-4o mini**，但正在寻找合适的营养数据集，如 [USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets.html)。
   - 建议包括从 **FNDDS** 编译潜在数据集，尽管尚不清楚其是否在 **Hugging Face** 上可用。
- **寻找 FFT 和基准测试**：一位成员表示有兴趣寻找 **FFT** 或 **LORA/QLORA** 来对 **27b model** 进行实验，提到在 **9b model** 上效果良好，但在更大的模型上遇到了挑战。
   - *Caseus* 建议针对 **Gemma 2 27b** 的 **QLORA** 版本在调整 **learning rate** 并使用最新的 **flash attention** 后可能会奏效。
- **关于 L40S GPUs 性能的咨询**：一位成员询问是否有人在 **L40S GPUs** 上训练或部署过模型，寻求有关其性能的见解。
   - 该咨询突显了人们对 **L40S GPUs** 在 AI 模型训练中的效率和能力的关注。
- **关于 AI 训练中 DPO 替代方案的讨论**：一位成员质疑 **DPO** 是否仍是 AI 训练中的最佳方法，并建议 **orpo**、**simpo** 或 **kto** 等替代方案可能更优。
   - 这引发了关于 AI 模型训练中各种方法有效性的不同意见交流。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Triton Conference 报名现已开启！**：将于 **2024 年 9 月 17 日** 在 **Meta Campus, Fremont CA** 举行的 **Triton Conference** 报名现已开启！请通过 [此 Google Form](https://docs.google.com/forms/d/e/1FAIpQLSecHC1lkalcm0h3JDUbspekDX5bmBvMxgVTLaK3e-61bzDDbg/viewform) 报名以预留名额。
   - 参会是**免费**的，但名额**有限**，因此建议尽早报名。
- **报名所需信息**：参与者必须提供其 **email**、**name**、**affiliation** 和 **role** 进行报名。其他可选问题包括饮食偏好，如**素食**、**纯素**、**犹太洁食**和**无麸质**。
   - *专业提示*：收集参会者希望从会议中获得什么收获！
- **会议报名的 Google 登录**：系统会提示参会者[登录 Google](https://accounts.google.com/AccountChooser?continue=https://docs.google.com/forms/d/e/1FAIpQLSecHC1lkalcm0h3JDUbspekDX5bmBvMxgVTLaK3e-61bzDDbg/viewform&service=wise) 以保存报名表单的进度。所有回复将发送至参与者提供的电子邮箱。
   - 别忘了：为确保安全，参与者绝不应通过 Google Forms 提交密码。



---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile 提升离线 LLM 访问**：**Llamafile** 的核心维护者报告了在单个文件中实现**离线、可访问的 LLM** 方面的重大进展。
   - 这一举措提高了可访问性，并简化了用户与大型语言模型的交互。
- **社区对 8 月项目充满期待**：围绕 **August** 正在进行的项目引发了热烈讨论，鼓励社区成员展示他们的工作。
   - 参与者有机会在 Mozilla AI 空间内参与并分享他们的贡献。
- **sqlite-vec 发布会即将举行**：即将举行的 **sqlite-vec** 发布会将允许参与者讨论功能并与核心维护者交流。
   - 演示和讨论即将展开，为最新进展的深入交流创造了机会。
- **令人兴奋的机器学习论文研讨会已排期**：即将举行的讲座将涵盖 **Communicative Agents** 和 **Extended Mind Transformers** 等主题，并邀请杰出演讲者。
   - 这些活动有望为机器学习领域的尖端研究和协作机会提供宝贵的见解。
- **Local AI AMA 承诺提供开源见解**：已安排的与核心维护者的 **Local AI** AMA 将提供关于这个可自托管的 OpenAI 替代方案的见解。
   - 本次会议邀请参与者探索 Local AI 的功能并直接解答他们的疑问。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第二部分：按频道详细摘要及链接

{% if medium == 'web' %}

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1269011615114854543)** (708 messages🔥🔥🔥): 

> - `LM Studio 性能问题`
> - `模型下载速度`
> - `AI 与本地系统的交互`
> - `AnythingLLM 中的多模态模型`
> - `RAM 和 VRAM 利用率` 

- **LM Studio 性能问题**：多位用户报告了 LM Studio 0.2.31 版本的问题，包括应用程序启动困难和模型无法正确加载。
   - 建议降级到早期版本（如 0.2.29）作为这些问题的潜在变通方案。
- **模型下载速度**：用户在 LM Studio 网站上遇到了下载速度波动，有人注意到速度被限制在 200kbps。
   - 建议等待或稍后重试下载，因为 AWS 限制速度在共享资源中并不罕见。
- **AI 与本地系统的交互**：讨论了 AI 模型（特别是像 OpenInterpreter 这样的 LLM）是否可以获得视觉能力来控制 PC。
   - 有人指出，此类功能可能会引发模型不可预见的行为，这说明了当前 AI 理解能力的局限性。
- **AnythingLLM 中的多模态模型**：用户对多模态模型的能力及其在 AnythingLLM 框架中的可用性表示关注。
   - 建议包括探索未经审查的模型，并查看 UGI-Leaderboard 等资源进行比较。
- **RAM 和 VRAM 利用率**：确认用户可以结合使用 RAM 和 VRAM 来运行更大的模型，相关设置可在 LM Studio 中配置。
   - 默认设置允许应用程序高效管理 RAM 和 VRAM 的使用，以获得最佳性能。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://matheval.ai/en/">MathEval</a>: MathEval 是一个致力于全面评估 LLM 数学能力的基准测试，由 22 个不同数学领域的评估数据集和近 30,000 道数学题组成。其...</li><li><a href="https://radxa.com/products/rock5/5itx/">Radxa ROCK 5 ITX</a>: 您的 8K ARM 个人电脑</li><li><a href="https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard">UGI Leaderboard - DontPlanToEnd 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://play.google.com/store/apps/details?id=net.hamandmore.crosstalk&hl=ln&gl=US&pli=1">Crosstalk Multi-LLM AI Chat – Google Play 应用</a>: 未找到描述</li><li><a href="https://huggingface.co/mradermacher/TinyStories-656K-GGUF">mradermacher/TinyStories-656K-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/legraphista/internlm2_5-20b-chat-IMat-GGUF">legraphista/internlm2_5-20b-chat-IMat-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://releases.lmstudio.ai/windows/0.2.27/latest/LM-Studio-0.2.27-Setup.exe">未找到标题</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>: Scaling Laws 描述了语言模型规模与其能力之间的关系。与以往通过 Loss 或基准测试评估模型能力的研究不同，我们估计了...</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/discussions/6">lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF · 模型更新</a>: 未找到描述</li><li><a href="https://play.google.com/store/apps/details?id=us.valkon.privateai&hl=fr&gl=US">Private AI – Google Play 应用</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/tree/main">lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF (main 分支)</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=CZbhUfmTXaE">Gemini-1.5 Pro Experiment (0801): Gemini 的最新更新击败了 Claude 和 GPT-4O (已全面测试)</a>: 加入此频道以获取会员福利: https://www.youtube.com/@aicodeking/join。在这段视频中，我将讨论新的 Gemini-1.5 Pro Experiment (080...</li><li><a href="https://tenor.com/view/what-year-is-it-jumanji-forgotten-gif-15305928">What Year Is It Jumanji GIF - What Year Is It Jumanji Forgotten - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">lmstudio-ai/configs 的 main 分支下的 configs/Extension-Pack-Instructions.md</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://releases.lmstudio.ai/windows/0.2.29/1/LM-Studio-0.2.29-Setup.exe">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: 计算机的自然语言接口</a>: 计算机的自然语言接口。通过在 GitHub 上创建账号，为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://aistudio.google.com/">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: C/C++ 实现的 LLM 推理</a>: C/C++ 实现的 LLM 推理。通过在 GitHub 上创建账号，为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/search?q=repo%3ANVIDIA%2Fcuda-samples%20int4&type=code">共同构建更好的软件</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、Fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g">[1 小时演讲] 大语言模型简介</a>: 这是一个面向普通观众的 1 小时大语言模型（Large Language Models）介绍：它是 ChatGPT、Claude 和 Bard 等系统背后的核心技术组件。
</li>
</ul>

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1269006868236468244)** (138 条消息🔥🔥): 

> - `双 GPU 设置与性能`
> - `NVIDIA GPU 对比`
> - `笔记本电脑中的 NPU 能力`
> - `Tesla M10 的可用性`
> - `即将发布的硬件` 


- **双 4090 设置与单 GPU**：关于双 GPU 设置的讨论表明，多 GPU 配置可能会将模型拆分到不同显卡上，从而影响性能和速度。
   - 成员们担心单块 4090 可能难以应对大型模型，而双卡设置则需要编程才能有效使用。
- **笔记本中的 NPU：未来趋势？**：对话探讨了笔记本电脑中 NPU 的集成，对其与传统 GPU 相比的性能优势意见不一。
   - 一些参与者认为，将任务卸载到 NPU 可以提高效率，特别是在功耗受限的环境中。
- **Tesla M10：值得买吗？**：几位成员警告不要购买像 NVIDIA Tesla M10 这样较旧的 GPU，因为它们效率低下且已过时。
   - 有建议称，如果用户追求旧硬件，可以考虑像 P40 这样更新一些的型号。
- **LLM 推理中的 GPU 性能**：用户报告了使用集成 GPU 的不同体验，并讨论了其性能指标，特别是 Llama 3.1 模型。
   - 推理结果表明，性能峰值通常不仅取决于 GPU 算力，还与内存管理和上下文窗口设置有关。
- **未来硬件：期待与升级**：参与者对即将推出的硬件表示兴奋，特别是 Studio M4 Ultra 和明年的 Blackwell 架构。
   - 讨论强调了在等待下一代产品发布期间，升级到 4090 进行深度学习任务的潜在好处。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gyazo.com/823a6d9154a6e84d93b2352884b3b9e7">Gyazo</a>:  </li><li><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">mistralai/Mistral-Large-Instruct-2407 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/guaton-computadora-enojado-computer-rage-gif-14480338">Guaton Computadora GIF - Guaton Computadora Enojado - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/18ocu6q/couuld_llama2_70b_be_run_on_a_tesla_m10/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.techpowerup.com/324271/amd-strix-halo-a-large-rectangular-bga-package-the-size-of-an-lga1700-processor">AMD &quot;Strix Halo&quot; a Large Rectangular BGA Package the Size of an LGA1700 Processor</a>: 显然 AMD &quot;Strix Halo&quot; 处理器是真实存在的，而且体积巨大。该芯片旨在与 Apple M3 Pro 和 M3 Max 等产品竞争，用于超便携笔记本电脑...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1269012429023481988)** (810 条消息🔥🔥🔥): 

> - `Hugging Face 模型特性`
> - `LLM 推理优化`
> - `开源贡献指南`
> - `使用 PEFT 微调模型`
> - `在 PyTorch 中使用 CUDA graphs` 


- **Hugging Face 模型特性讨论**：用户讨论了 Hugging Face 上的各种模型，包括用于翻译的 MarionMT 模型以及 TatoebaChallenge 仓库在语言支持方面的潜力。
   - 有人对某些模型的局限性表示担忧，并提出需要更好的文档和实现示例。
- **优化 LLM 推理速度**：提出了几项加速 LLM 推理的建议，包括使用 torch.compile 以及对比 vLLM、TGI 和 LMdeploy 的性能。
   - 持续的讨论凸显了在处理大语言模型时提高效率和性能的兴趣。
- **开源贡献指南**：一位用户寻求关于进行开源贡献的建议，并发现讨论有助于重新思考他们的方法和动机。
   - 分享了相关博客文章和教程的链接，以帮助新贡献者入门。
- **使用 PEFT 微调模型**：一位用户分享了他们微调 Llama2 模型的经验，在推送模型时遇到了一些问题，引发了关于该过程中最佳实践的讨论。
   - 建议的最佳实践包括正确使用推送至 Hub 的功能以及管理训练配置。
- **在 PyTorch 中使用 CUDA graphs**：讨论探讨了 CUDA graphs 如何通过减少与启动 GPU 操作相关的开销来优化 PyTorch 模型。
   - 用户表达了对提高性能的兴趣，并指出正确使用 torch 等库对于有效的 graph 实现至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://blogs.nlmatics.com/bert/math/2021/08/06/Teaching-BERT-to-Solve-Word-Problems.html">Teaching BERT to Solve Word Problems</a>: 教会 BERT 解决数学应用题</li><li><a href="https://huggingface.co/docs/hub/repositories-recommendations#sharing-large-datasets-on-the-hub">Repository limitations and recommendations</a>: 未找到描述</li><li><a href="https://huggingface.co/chatpdflocal/llama3.1-8b-gguf">chatpdflocal/llama3.1-8b-gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Tonic/Minitron/settings?clone=true">Hugging Face – The AI community building the future.</a>: 未找到描述</li><li><a href="https://huggingface.co/hugging-quants">hugging-quants (Hugging Quants)</a>: 未找到描述</li><li><a href="https://huggingface.co/mlabonne/Llama-3.1-70B-Instruct-lorablated">mlabonne/Llama-3.1-70B-Instruct-lorablated · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/compile2011/W-finetune/discussions/">compile2011/W-finetune · Discussions</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=5nY_cy8zcO4">Don&#39;t Contribute to Open Source</a>: 你没听错。我不认为你应该向开源项目做贡献。除非……关键词：GITHUB OPEN SOURCE 编码 开发 编程 学习编程 ...</li><li><a href="https://tenor.com/view/helicopter-baguette-gif-20550621">Helicopter Baguette GIF - Helicopter Baguette - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/mervenoyann/status/1819289510124863774">Tweet from merve (@mervenoyann)</a>: OWLSAM2：支持文本提示的 SAM2 🦉 结合了尖端的 Zero-shot 目标检测器 OWLv2 🤝 掩码生成器 SAM2 (small) 具有极高精度的 Zero-shot 分割 ⛵️</li><li><a href="https://github.com/vllm-project/vllm">GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs</a>: 一个用于 LLM 的高吞吐量且内存高效的推理与服务引擎 - vllm-project/vllm</li><li><a href="https://huggingface.co/or4cl3ai/SquanchNastyAI">or4cl3ai/SquanchNastyAI · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/or4cl3ai/IntelliChat">or4cl3ai/IntelliChat · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/or4cl3ai/SoundSlayerAI">or4cl3ai/SoundSlayerAI · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/or4cl3ai/A-os43-v1">or4cl3ai/A-os43-v1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Voxel51/DataCentricVisualAIChallenge">DataCentricVisualAIChallenge - a Hugging Face Space by Voxel51</a>: 未找到描述</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/tasks/question-answering">What is Question Answering? - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Tonic/Minitron">Minitron - a Hugging Face Space by Tonic</a>: 未找到描述</li><li><a href="https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/">Accelerating PyTorch with CUDA Graphs</a>: 今天，我们很高兴地宣布一项新的高级 CUDA 特性——CUDA Graphs 已引入 PyTorch。现代深度学习框架拥有复杂的软件栈，这会产生显著的开销……</li><li><a href="https://paperswithcode.com/sota/arithmetic-reasoning-on-gsm8k">Papers with Code - GSM8K Benchmark (Arithmetic Reasoning)</a>: 目前 GSM8K 上的 SOTA 是 GPT-4 DUP。查看 152 篇论文及其代码的完整对比。</li><li><a href="https://huggingface.co/datasets/stanfordnlp/imdb/tree/main">stanfordnlp/imdb at main</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3">RAG chatbot using llama3</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Tonic/Minitron/tree/main">Tonic/Minitron at main</a>: 未找到描述</li><li><a href="https://huggingface.co/models?pipeline_tag=image-to-text&sort=trending&search=glucose">Models - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/argilla-chatbot">How we leveraged distilabel to create an Argilla 2.0 Chatbot</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/gemma-peft">Fine-Tuning Gemma Models in Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md">transformers/CONTRIBUTING.md at main · huggingface/transformers</a>: 🤗 Transformers：适用于 PyTorch、TensorFlow 和 JAX 的 SOTA 机器学习库。 - huggingface/transformers</li><li><a href="https://github.com/pygfx/wgpu-py/pull/547/files#diff-d9f01e8e8bedc3ca54c8b49d">[WIP] update to wgpu-native 22.1 by Vipitis · Pull Request #547 · pygfx/wgpu-py</a>: 我非常期待更好的错误处理、编译信息和 gl</li>

sl const built-ins，所以我已经开始了这项工作。如果有所帮助，请随意 cherry pick 我的更改或提交到此分支。我继续...</li><li><a href="https://youtu.be/-YpwsdRKt8Q>">SpiegelMining – 对 Spiegel-Online 的逆向工程 (33c3)</a>：那些认为数据留存和“大数据”无害的人，可以在这里看到一个关于 Spiegel-Online 的演示。自 2014 年年中以来，David 已经抓取了近 100,000 篇文章...</li><li><a href="https://blog.arcee.ai/announcing-distillkit/?utm_campaign=content&utm_source=Marktechpost&utm_medium=Blog&utm_term=Knowledge%20Distillation&utm_content=Blog%201">发布用于创建和分发 SLM 的 DistillKit</a>：首先，Arcee AI 通过 Model Merging 和开源仓库 MergeKit 彻底改变了小语言模型 (SLM)。今天，我们在 SLM 的创建和分发方面为您带来了又一次飞跃...</li><li><a href="https://github.com/arcee-ai/DistillKit?ref=blog.arcee.ai">GitHub - arcee-ai/DistillKit at blog.arcee.ai</a>：一个用于 LLM 蒸馏的开源工具包。通过在 GitHub 上创建账户，为 arcee-ai/DistillKit 的开发做出贡献。</li><li><a href="https://github.com/pygfx/wgpu-py/pull/547">[WIP] update to wgpu-native 22.1 by Vipitis · Pull Request #547 · pygfx/wgpu-py</a>：我对更好的错误处理、编译信息和 glsl const built-ins 感到非常兴奋，所以我已经开始了这项工作。如果有所帮助，请随意 cherry pick 我的更改或提交到此分支。我继续...</li><li><a href="https://github.com/pygfx/wgpu-py/pull/547/files#diff-d9f01e8e8bedc3ca54c8b49dc3d0b43c504dc488b58d4e2b4b2a03eeef29dd40>">[WIP] update to wgpu-native 22.1 by Vipitis · Pull Request #547 · pygfx/wgpu-py</a>：我对更好的错误处理、编译信息和 glsl const built-ins 感到非常兴奋，所以我已经开始了这项工作。如果有所帮助，请随意 cherry pick 我的更改或提交到此分支。我继续...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1269273708711776267)** (3 条消息): 

> - `LLM 推理优化`
> - `基于课程的 AI 方法` 


- **探索 LLM 推理优化技术**：一篇值得关注的文章讨论了优化 **LLM 推理**的技术，旨在提高吞吐量和 GPU 利用率，同时降低延迟，展示了大模型面临的挑战。
   - 它强调堆叠 Transformer 层可以带来**更好的准确性**，并详细阐述了与**检索增强生成** (RAG) 流水线相关的成本，这些流水线需要大量的处理能力。
- **AI 中的基于课程的方法**：一篇论文概述了当前 **AI 系统**在推理和适应性方面的局限性，强调需要一种稳健的基于课程的方法来增强可解释性和因果理解。
   - 作者强调，虽然 AI 擅长**模式识别**，但在复杂的推理环境中表现不佳，这从根本上限制了它的变革潜力。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://nonartificialintelligence.blogspot.com/2024/08/from-data-to-understanding-curriculum.html">从数据到理解：一种培养 AI 推理的基于课程的方法</a>：未找到描述</li><li><a href="https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/">掌握 LLM 技术：推理优化 | NVIDIA 技术博客</a>：堆叠 Transformer 层以创建大模型可以带来更好的准确性、少样本学习能力，甚至在广泛的语言任务中展现出接近人类的涌现能力。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1269302599639629884)** (10 条消息🔥): 

> - `CatVTON 虚拟试穿模型`
> - `GeekyGhost 扩展`
> - `SweetHug AI 聊天机器人`
> - `用于 ASCII Art 的 Joy Captioning`
> - `具有优化推理能力的 LLM 部署` 


- **CatVTON 重新定义了虚拟试穿方法**：最近的一篇 [arXiv 论文](https://arxiv.org/abs/2407.15886) 介绍了 **CatVTON**，这是一种高效的虚拟试穿扩散模型，通过在处理过程中直接拼接服装图像，消除了对 ReferenceNet 和额外图像编码器的需求。
   - 这一创新在降低训练成本的同时，保持了服装向目标人物迁移的真实感。
- **GeekyGhost 发布 Automatic1111 扩展**：一位成员分享了他们在 [GitHub 上创建的 Automatic1111 扩展](https://github.com/GeekyGhost/Automatic1111-Geeky-Remb.git)，这是其 ComfyUI geely remb 工具的移植版本。
   - 他们还介绍了另一个使用 Gradio 的 Web UI 项目，进一步展示了他们在社区中的工作。
- **探索 SweetHug AI 聊天机器人**：另一位用户强调了 [SweetHug AI](https://sweethugai.com) 的功能，这是一个 AI 角色平台，为用户提供与 AI 女友聊天的机会，分享他们的想法和幻想。
   - 该服务由 Ally AI Pte. Ltd. 运营，包含 NSFW 聊天和联盟计划等功能。
- **Joy Captioning 在 ASCII Art 方面表现出色**：一位成员指出了 [Joy Captioning Space](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha)，它能成功为 ASCII Art 生成描述，而不是将其误认为技术图表。
   - 他们对发现一个能准确反映文本格式艺术表达的工具感到兴奋。
- **关于 LLM 部署优化的咨询**：一位用户寻求关于具有优化推理能力的 **LLM 部署** 的见解，强调了对大语言模型效率的关注。
   - 这引发了社区内对该领域进展和实践的好奇心。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.15886">CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models</a>: 基于扩散模型的虚拟试穿方法实现了逼真的试穿效果，但通常会将主干网络复制为 ReferenceNet，或使用额外的图像编码器来处理条件输入...</li><li><a href="https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha">Joy Caption Pre Alpha - fancyfeast 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://sweethugai.com">SweetHug AI: Free Chat With Your AI Girlfriends - No Limits</a>: 未找到描述</li><li><a href="https://github.com/GeekyGhost/Automatic1111-Geeky-Remb.git">GitHub - GeekyGhost/Automatic1111-Geeky-Remb: 我的 ComfyUI geely remb 工具的 Automatic1111 移植版</a>: 我的 ComfyUI geely remb 工具的 Automatic1111 移植版。通过在 GitHub 上创建账户，为 GeekyGhost/Automatic1111-Geeky-Remb 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1269193770818338878)** (11 条消息🔥): 

> - `Testcontainers`
> - `Rob 的 Instagram 互动`
> - `密集预测中的 Self-Supervised Learning`
> - `AI Research Agent 文档` 


- **探索用于 AI 开发的 Testcontainers**：一位成员分享了他们对 [Testcontainers](https://huggingface.co/blog/Tonic/localai-testcontainers) 的发现，强调了其在开发和提供 AI 应用方面的潜力。
   - 他们还提到自己最近的新爱好是为 Docker Testcontainers 项目做贡献，并鼓励其他人也加入其中。
- **Rob 在 Instagram 上获得新能力**：通过一个视觉模型，一位成员成功让 Rob 能够阅读并回应他的 Instagram 评论，展示了他不断增长的能力。
   - 另一位成员幽默地注意到 Rob 的能力正在增强，并建议进行 TikTok 直播可能会很赚钱。
- **Self-Supervised Learning 革新密集预测**：一位成员强调了 Self-Supervised Learning 方法的进展，特别是在提升目标检测和分割等密集预测任务性能方面的作用。
   - 他们提供了一个信息丰富的帖子链接，讨论了传统 SSL 方法在这些应用中所面临的挑战。
- **AI Research Agent 文档发布**：一位成员分享了他们开发的 AI Research 库的资源，包括 [文档](https://vtempest.github.io/ai-research-agent/docs/) 和演示。
   - 他们推广了搜索功能、文本提取和关键词主题提取等特性，同时邀请大家讨论集成事宜。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://dev.to/tonic/dockers-testcontainers-are-great-42cl">未找到标题</a>: 未找到描述</li><li><a href="https://vtempest.github.io/ai-research-agent/docs/">ai-research-agent 主页</a>: 未找到描述</li><li><a href="https://www.lightly.ai/post/using-self-supervised-learning-for-dense-prediction-tasks">在密集预测任务中使用 Self-Supervised Learning</a>: 针对目标检测、实例分割和语义分割等密集预测任务的 Self-Supervised Learning 方法概述</li><li><a href="https://huggingface.co/blog/Tonic/localai-testcontainers">使用 Docker Testcontainers 的本地 AI</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1269057874710237356)** (29 messages🔥): 

> - `学习重点关注`
> - `Hackathon 协作`
> - `SEE-2-SOUND 演示`
> - `演示录像`
> - `线性代数文章` 


- **选择学习的主攻方向**：成员们讨论了选择一个共同关注点（如课程或项目）的重要性，以增强小组内的学习效果和问责制。
   - 组队参加 **Hackathon** 等实践挑战可以促进协作，但参与者具备相似的技能水平至关重要，以防止工作分配不均。
- **SEE-2-SOUND 革新空间音频**：进行了一场关于 **SEE-2-SOUND** 的演示，这是一个 **Zero-Shot** 框架，可以从视觉内容生成空间音频，无需大量的先验训练。
   - 这种创新方法将过程分解为识别关键视觉方面并将其整合到高质量的空间音频中，为沉浸式内容带来了令人兴奋的前景。
- **会议录像的可用性**：一位成员询问了最近一次演示的录像是否可用，该演示曾遇到技术困难。
   - 演示者确认稍后将发布该会议的剪辑版本。
- **新成员和资源的介绍**：新成员询问了小组资源，例如活动日历，以便参与活动。
   - 成员们回答说，活动安排通常通过消息平台管理，并定期发布更新。
- **通过文章分享知识**：一位成员在 **Medium** 上分享了一篇关于线性代数的新文章，重点关注向量的线性组合和张成（span）。
   - 这篇文章为热衷于加强对线性代数基础知识理解的成员提供了资源。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.06612">SEE-2-SOUND: Zero-Shot Spatial Environment-to-Spatial Sound</a>: Generating combined visual and auditory sensory experiences is critical for the consumption of immersive content. Recent advances in neural generative models have enabled the creation of high-resoluti...</li><li><a href="https://drexel.zoom.us/j/82015537439">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://medium.com/@amitsubhashchejara/linear-algebra-part-2-linear-combination-and-span-d5fe65ef0e8f">Linear Algebra (Part-2): Linear Combination And Span</a>: Learn about the linear combination and the span of vectors.</li><li><a href="https://drexel.zoom.us/j/820">Video Conferencing, Web Conferencing, Webinars, Screen Sharing</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1269192308377784392)** (8 messages🔥): 

> - `计算机视觉课程作业`
> - `SF-LLaVA 论文讨论`
> - `CV 项目建议`
> - `3D 物体朝向建模`
> - `带时间标签的户外图像数据集` 


- **计算机视觉课程作业需要明确**：用户对他们的 **Computer Vision** 课程作业组成部分表示困惑，寻求对要求的澄清。
   - 一位参与者建议通过语音聊天协作讨论相关材料。
- **关于 SF-LLaVA 的协作讨论**：一位用户提议通过语音聊天审阅 **SF-LLaVA** 论文，鼓励其他人加入讨论。
   - 这一举措反映了社区在理解学术资源方面互相支持的意愿。
- **快速 CV 项目建议**：参与者分享了 **CV** 项目的想法，包括将**双目深度估计（binocular depth estimation）**作为一个可以在一周内完成的可行方案。
   - 这一讨论突显了学习者积极参与知识实际应用的方法。
- **建模 3D 物体朝向**：一位用户询问了创建能够确定 **3D 物体朝向**模型的方法。
   - 这个问题强调了对提升 **Computer Vision** 中空间理解技能的兴趣。
- **寻找带时间标签的户外图像数据集**：一位用户正在寻找一个带有**按一天中时间标记的户外图像**的高质量数据集，并表示现有的选项（如 **MIRFLIKR**）存在困难。
   - 这一查询表明了对满足 **Computer Vision** 中特定研究需求的高质量数据集的需求。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1269936468449562739)** (4 messages): 

> - `Chaquopy 的依赖问题`
> - `使用 NLP 寻找相关表` 


- **Qwen2-0.5B-Instruct 在 Chaquopy 上的依赖问题**：一位成员在使用 Chaquopy 和 Python 开发 Android 应用时遇到了持续的 *依赖冲突*，特别是涉及到 **transformers** 和 **tokenizers**。
   - 他们提供了 **Gradle** 配置，并表示尝试使用各种包版本时导致了错误：**InvalidVersion: Invalid version: '0.10.1,<0.11'**。
- **寻求用于关系模型的 NLP 方法**：另一位成员询问如何使用 **NLP** 技术根据列名和描述来识别表之间的关系。
   - 他们正在寻找建议或参考资料，以了解在给定特定表的情况下，表与表之间是如何关联的。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1269104340094812221)** (7 messages): 

> - `Diffusers 中的梯度检查点 (Gradient Checkpointing)`
> - `Quanto 库的加载时间`
> - `用于对象裁剪的 CNN`
> - `Flux 模型问题` 


- **梯度检查点 (Gradient Checkpointing) 已实现**：一位用户指出 Diffusers 之前缺少梯度检查点功能，但分享了一个代码片段，显示已添加了设置梯度检查点的方法。
   - 新方法 `_set_gradient_checkpointing` 允许为支持该功能的模块切换检查点状态。
- **Quanto 库的模型加载速度慢**：一位成员讨论了使用 Quanto 库两天的经历，在他们的配置（4080 - Ryzen 9 5900X）下，将量化模型移动到设备上花费了超过 **400 秒**。
   - 他们注意到在此过程中创建了新的 **QBitsTensors**，这可能是导致延迟的原因。
- **请求在 Quanto 中进行问题追踪**：另一位用户建议在 Quanto 仓库中创建一个 issue 来记录加载缓慢的问题，以便跟踪并可能解决该问题。
   - 他们提到维护者目前正在休假，这可能会延迟回复。
- **寻求从图像中裁剪对象的 CNN**：一位用户请求推荐一种 CNN，用于从白色背景中裁剪浅色对象。
   - 他们正在寻找解决方案来应对颜色对比度带来的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://hastebin.skyra.pw/sixakucecu.py)">Hastebin</a>：未找到描述</li><li><a href="https://github.com/huggingface/diffusers/blob/b1f43d71897ad2c73cb891d2e92d23bc7d46a4be/src/diffusers/models/transformers/transformer_flux.py#L306">diffusers/src/diffusers/models/transformers/transformer_flux.py at b1f43d71897ad2c73cb891d2e92d23bc7d46a4be · huggingface/diffusers</a>：🤗 Diffusers：用于 PyTorch 和 FLAX 的最先进图像和音频生成扩散模型。- huggingface/diffusers</li><li><a href="https://github.com/huggingface/diffusers/blob/b1f43d71897ad2c73cb891d2e92d23bc7d46a4be/src/diffusers/models/transformers/transformer_flux.py#L248">diffusers/src/diffusers/models/transformers/transformer_flux.py at b1f43d71897ad2c73cb891d2e92d23bc7d46a4be · huggingface/diffusers</a>：🤗 Diffusers：用于 PyTorch 和 FLAX 的最先进图像和音频生成扩散模型。- huggingface/diffusers
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1269012141747470407)** (840 messages🔥🔥🔥): 

> - `Flux 模型性能`
> - `ComfyUI 使用`
> - `Stable Diffusion 模型`
> - `RAM 和 VRAM 要求`
> - `动画生成工具`

- **Flux 模型在 GPU 上的性能**：用户报告了在不同 GPU 上使用 Flux 模型生成图像的速度差异，根据配置和模型版本的不同，速度在每秒 1.0 到 2.0 次迭代（iterations per second）之间。
   - 一些用户指出，即使在较低 VRAM 的配置下，通过使用 CPU offloading 或 quantization 技术，也能成功生成图像。
- **在 ComfyUI 中安装和使用 Flux**：用户讨论了在 ComfyUI 上安装 Flux 的过程，建议使用 update.py 脚本而不是管理器（manager）进行更新，并确保文件放置正确。
   - 分享了安装指南和其他资源，以帮助那些刚开始搭建环境的新手。
- **Stable Diffusion 模型之间的差异**：参与者解释说，SD1.5、SDXL 和 SD3 等模型都是 Stable Diffusion 的变体，各具优势，而 Flux 是由原 SD3 团队开发的新竞争对手。
   - 讨论强调了与传统的 Stable Diffusion 模型相比，Flux 对资源的要求更高。
- **RAM 和 VRAM 对性能的影响**：用户指出，对于 Stable Diffusion 的性能而言，拥有充足的 RAM 不如拥有足够的 VRAM 重要，建议至少使用 16GB VRAM 以获得最佳效果。
   - 小组讨论了 RAM 主要用于支持模型加载，而不是直接影响生成速度。
- **动画生成工具**：参与者询问了类似 Animatediff 等用于从图像生成视频内容的工具现状，寻求有关最新可用方法的信息。
   - 有建议称，虽然 Animatediff 仍在使用中，但可能存在更新的选项，可以为类似任务提供替代方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://imgur.com/a/yuhw1d4">imgur.com</a>: 在 Imgur 探索互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行模因、娱乐 GIF、鼓舞人心的故事、病毒式视频等来提振你的精神...</li><li><a href="https://imgur.com/a/TIHYuwy">imgur.com</a>: 在 Imgur 探索互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行模因、娱乐 GIF、鼓舞人心的故事、病毒式视频等来提振你的精神...</li><li><a href="https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell">FLUX.1 [Schnell] - a Hugging Face Space by black-forest-labs</a>: 未找到描述</li><li><a href="https://www.shakker.ai/modelinfo/908a4d44cac844ca8e5d66a23c5cdf3d?from=personal_page">Shakker AI - Premium Stable Diffusion Model Hub</a>: 未找到描述</li><li><a href="https://x.com/recatm/status/1819348949972476019">Tweet from XiQiao 西乔 (@recatm)</a>: SD3 团队分享了正在进行的 SD3.1 训练（尚未完成）的一些生成结果。使用相同的 prompts 将其与 Robin 的 Flux 3.1 Pro 进行对比。3.1 模型几乎可以媲美...</li><li><a href="https://www.shakker.ai/">Shakker AI - Premium Stable Diffusion Model Hub</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/diffusers/en/index">Diffusers</a>: 未找到描述</li><li><a href="https://runpod.io?ref=yxgme9zg">RunPod - The Cloud Built for AI</a>: 在同一个云端开发、训练和扩展 AI 模型。通过 GPU Cloud 启动按需 GPU，使用 Serverless 扩展 ML 推理。</li><li><a href="https://huggingface.co/blog/quanto-diffusers#how-about-int4">Memory-efficient Diffusion Transformers with Quanto and Diffusers</a>: 未找到描述</li><li><a href="https://www.runpod.io/">RunPod - The Cloud Built for AI</a>: 在同一个云端开发、训练和扩展 AI 模型。通过 GPU Cloud 启动按需 GPU，使用 Serverless 扩展 ML 推理。</li><li><a href="https://www.youtube.com/watch?v=UTmwyxHQ7pM```">How to use Face Analysis to improve your workflows</a>: 我经常在工作流中使用 Face Analysis，但我们从未真正讨论过它的实际工作原理。这里有你需要知道的一切。记得升级扩展...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ekpczn/psa_illyasviel_is_devving_hard_on_forge/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://youtu.be/JwLbnO4px-E">V1.0 | ComfyUI on Photoshop: Best New AI Tool</a>: Photoshop 上的 ComfyUI 免费 AI 工具来了！灵活且开源🔥 安装方法：https://youtu.be/YD09xpQrNZ4。☕ 请我喝杯咖啡：https://buymeacoffee.com/n...</li><li><a href="https://www.stablediffusiontutorials.com/2024/08/flux-installation.html?m=1">FLUX: Installation with Workflow is Here</a>: 未找到描述</li><li><a href="https://replicate.com/black-forest-labs/flux-schnell">black-forest-labs/flux-schnell – Run with an API on Replicate</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=E_D7y0YjE88">How to EASILY Install ComfyUI | Stable Diffusion Tutorial</a>: 探索 ComfyUI，一个改变游戏规则的 AI 界面。本视频将指导你完成 Windows 和基于云的选项（如 ThinkDiffusion）的简易设置。了解安装...</li><li><a href="https://www.reddit.com/r/StableDiffusion/s/Sby5nw5tei">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://youtu.be/zzA1iUgtiEs">NVIDIA Update Solves CUDA error (but very slow) -Train Dreambooth, SDXL LoRA  with Low VRAM</a>: #stablediffusion #a1111 #nvidia #update #cuda #cudaerror #lowvram #kohyass #LoRA #dreambooth #tensorRT (更新：虽然更新能够解决 CUDA 内存...</li><li><a href="https://github.com/facebookresearch/fairscale/blob/main/docs/source/installation_instructions.rst">fairscale/docs/source/installation_instructions.rst at main · facebookresearch/fairscale</a>: 用于高性能和大规模训练的 PyTorch 扩展。 - facebookresearch/fairscale</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre">Release v1.0.0-pre · AUTOMATIC1111/stable-diffusion-webui</a>: webui.zip 是为无法安装 python 和 git 的用户提供的二进制发行版。包含所有内容 - 只需双击 run.bat 即可启动。除了 Windows 10 之外没有其他要求。仅限 NVIDIA...</li><li><a href="https://github.com/CompVis/stable-diffusion">GitHub - CompVis/stable-diffusion: A latent text-to-image diffusion model</a>: 一个潜空间文本到图像扩散模型。通过在 GitHub 上创建账号来为 CompVis/stable-diffusion 的开发做出贡献。</li><li><a href="https://www.liblib.art/">LiblibAI-哩布哩布AI - 中国领先的AI创作平台</a>: 未找到描述</li><li><a href="https://civitai.com/models/132632/epicphotogasm">epiCPhotoGasm - Ultimate Fidelity | Stable Diffusion Checkpoint | Civitai</a>: 欢迎来到 epiCPhotoGasm。该模型针对写实主义进行了高度微调，具有极微小的...</li>

需要过度的 prompting 才能表现出色。所有 Showcase 图像...</li><li><a href="https://github.com/cubiq/ComfyUI_IPAdapter_plus?tab=readme-ov-file">GitHub - cubiq/ComfyUI_IPAdapter_plus</a>: 通过在 GitHub 上创建账户来为 cubiq/ComfyUI_IPAdapter_plus 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1269071533700288522)** (32 messages🔥): 

> - `Epoch 8 Accuracy Spike` (Epoch 8 准确率激增)
> - `CUDA with Deep Reinforcement Learning` (CUDA 与深度强化学习)
> - `Parallelizing Environments` (并行化环境)
> - `Mojo Programming Language` (Mojo 编程语言)
> - `Using Streams in CUDA` (在 CUDA 中使用 Streams)


- **Epoch 8 准确率飙升**：成员们讨论了在第 8 个 Epoch 后准确率分数出现的意外激增，并询问这是否属于典型行为。
   - “看起来完全正常”是另一位成员给出的宽慰性回复，表示无需担心。
- **CUDA 在深度强化学习中的挑战**：一位成员分享了在 CUDA 中为 DRL 创建环境的挫败感，提到这通常是一个麻烦的过程。
   - 有人建议他们使用像 [PufferAI](https://pufferai.github.io/) 这样的工具，这可能会提供更好的并行性，尤其是在环境方面。
- **在 CPU 上并行化 DRL 环境**：一位用户表示有兴趣构建自己的 DRL 环境，并寻求在可能切换到 CUDA 之前在 CPU 上进行并行化的指导。
   - 另一位参与者建议了设置环境的资源和示例，包括一个特定的 Gameboy 模拟器作为参考。
- **Mojo 语言介绍**：围绕 Mojo 编程语言展开了讨论，该语言旨在取代整个 ML 栈，但保留了 PTX 的某些方面。
   - 成员们表达了好奇心，一位粉丝提到了一段 Chris Lattner 解释编程未来的视频。
- **在 ML 模型中使用 CUDA Streams**：关于在 CUDA 中使用 Streams 的咨询引发了对性能的讨论，揭示了在独立 Streams 上计算操作时关于开销的不同经验。
   - 有人指出，如果 Kernels 很大，由于 GPU 资源有限，多 Streams 可能无法提供预期的性能提升。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/giffmana/status/1699363411463725493?s=46">Lucas Beyer (bl16) (@giffmana) 的推文</a>: 最近的一种模式：对于在线 RL，在 JAX（几乎就是 numpy！）中实现环境，以便它在设备上运行，就像模型训练一样。这使训练速度显著加快。我看到的第一个是用于物理的 brax https:/...</li><li><a href="https://www.youtube.com/watch?v=dW10MQ6hKDE">强化学习直播开发</a>: 在 XStar 上关注 jsuarez5341 https://github.com/pufferai/pufferlib MIT 博士和全职 OSS RL 驱魔人</li><li><a href="https://github.com/PufferAI/PufferLib/blob/729003f9cb89845cc1a69a65e5a2431b2d0542bd/pufferlib/environments/pokemon_red/environment.py#L15">PufferLib/pufferlib/environments/pokemon_red/environment.py</a>: 简化复杂游戏环境的强化学习 - PufferAI/PufferLib</li><li><a href="https://x.com/jsuarez5341/status/1819808126851600668">Joseph Suarez (e/🐡) (@jsuarez5341) 的推文</a>: Pong 在约 250 万步、少于 90 秒内解决，在 1 个 GPU 上以每秒 30,000 步的速度训练。现在已加入 pufferai/pufferlib —— 点亮 star 以支持！</li><li><a href="https://youtu.be/pdJQ8iVTwj8?feature=shared&t=4084">Chris Lattner: 编程与 AI 的未来 | Lex Fridman Podcast #381</a>: Chris Lattner 是一位传奇的软件和硬件工程师，曾在 Apple、Tesla、Google、SiFive 和 Modular AI 领导项目，包括开发 S...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1269296957063430289)** (19 条消息🔥): 

> - `Passing Scalar Values to Triton Kernels` (向 Triton Kernel 传递标量值)
> - `Use of tl.constexpr in Triton` (Triton 中 tl.constexpr 的使用)
> - `Performance Impact of .item() on CUDA Tensors` (.item() 对 CUDA Tensor 的性能影响)
> - `Shared Memory vs Registers in Triton` (Triton 中的 Shared Memory 与 Registers)


- **直接向 Triton Kernel 传递标量**：一位成员确认你可以直接向 Triton kernel 传递单个标量值而不是指针，从而简化流程。
   - 然而，需要注意的是，如果该标量是一个 CUDA tensor，则必须通过引用传递，而不是作为 Python 标量。
- **使用 tl.constexpr 提升性能**：有一场关于使用 `tl.constexpr` 传递标量的讨论，有人声称这可以在修改 `inc` 时避免重新编译。
   - 成员们一致认为使用 Python 标量允许按值传递，但其有效使用取决于该值在运行时是否已知。
- **CUDA 中 .item() 的性能**：`.item()` 被指出是从 CUDA tensors 中提取标量值的一种方法，但由于 GPU 和 CPU 之间的同步，它可能会带来性能损失。
   - 成员建议在需要从 Torch 操作返回的 tensor 中获取标量时使用 `.item()`。
- **探索 Triton 内存管理**：一位成员询问了在 Triton kernels 中分配 Shared Memory 和 Registers 的启发式方法。
   - 他们强调了理解 `tl.load` 将 tensors 放置在何处的重要性，以增强对 Triton 的心理模型。


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1269744363001352307)** (3 条消息): 

> - `torch.Tensor.register_hook with Deepspeed` (在 Deepspeed 中使用 torch.Tensor.register_hook)
> - `torch JIT fuser for CUDA` (用于 CUDA 的 torch JIT fuser)


- **寻求在 Deepspeed 中使用 Tensor hook 的帮助**：一位成员询问在配合 **Deepspeed** ZeRO stage 1 进行训练时，如何使用 `torch.Tensor.register_hook` 在 backward 阶段进行自定义梯度修改，并指出尽管已将 hook 添加到参数中，但 hook 仍未执行的问题。
   - *他们倾向于在向 DeepSpeed 提交 issue 之前先在社区询问*。
- **关于已弃用的 JIT fuser 模块的困惑**：关于用于 CUDA 的 **torch JIT fuser** 的讨论开始，在遇到提示文件缺失的构建错误后，质疑 `fused_kernel.h` 文件的状态。
   - 该成员通过使用 `codegen/cuda/interface` 解决了问题，但仍好奇原始模块是否已成为死代码，或者是否仍有使用方法。



**提到的链接**：<a href="https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/codegen/fuser/cuda/fused_kernel.h">pytorch/torch/csrc/jit/codegen/fuser/cuda/fused_kernel.h at main · pytorch/pytorch</a>：Python 中的 Tensor 和动态神经网络，具有强大的 GPU 加速功能 - pytorch/pytorch

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1269525702152032288)** (7 条消息): 

> - `Self-Compressing Neural Networks` (自压缩神经网络)
> - `Dynamic Quantization` (动态量化)
> - `Model Reproduction Results` (模型复现结果) 


- **关于自压缩神经网络的热议**：一位成员强调 *Self-Compressing Neural Networks* 通过将**模型大小**纳入损失函数，使用了动态量化感知训练。
   - *这是一个很酷的想法*，另一位成员表示，并指出**量化位数 (quantization bits)** 是一个可优化的参数。
- **讨论结果复现**：一位成员敦促复现论文结果，认为如果它**确实有效**，将非常有前景。
   - 作为回应，另一位成员提到他们看到 **George Hotz** 发推称他已经复现了结果，但尚未详细验证。
- **CIFAR10 需要调优**：一位成员报告了对该技术的实验，但表示需要进行一些调优以提高准确率，因为模型在 **CIFAR10** 上卡在约 **70%**。
   - 他们承认虽然这种方法看起来有效，但为了获得最佳结果，进一步的调整是必要的。


  

---

### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1269174460603170837)** (1 messages): 

> - `Winter Internship Opportunities` (冬季实习机会)
> - `ML Systems and Applied ML` (ML Systems 与 Applied ML)
> - `CUDA and Triton Applications` (CUDA 与 Triton 应用)
> - `Open Source Contributions` (开源贡献)
> - `Autonomous Racing Design` (自动驾驶赛车设计)


- **寻找 ML 冬季实习**：一位用户正在积极寻找从 **2025 年 1 月**初到 **2025 年 4 月**底的**冬季实习**，重点关注与 **ML systems** 和 **applied ML** 相关的职位。
   - 该用户强调了其计算机科学背景，曾在相关领域完成过**两次**实习，并参与了 **open source** 项目。
- **CUDA 优化专业知识**：该用户提到在过去的一年半里专注于优化**模型推理 (model inference)**，特别是利用 **CUDA** 以及最近使用的 **Triton**。
   - 他们对与这些技术相关的任何职位表现出浓厚兴趣，并展示了在 **torchscript JIT compiler** 方面的实践经验。
- **对软件工程的广泛兴趣**：该用户对各种 **software** 或 **ML engineering** 职位持开放态度，表明其在求职中具有灵活性。
   - 此外，他们还作为一个自动驾驶赛车设计团队的控制与运动规划工程师参与其中，展示了多样化的工程技能。


  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1269535657873768480)** (20 messages🔥): 

> - `Solutions for PMPP exercises` (PMPP 练习题解答)
> - `Torch compile issues` (Torch compile 问题)
> - `Flash Attention 3 integration` (Flash Attention 3 集成)
> - `Collaboration on coding` (代码协作)
> - `Waiting for updates on Flash Attention 3` (等待 Flash Attention 3 更新)


- **分享 PMPP 练习题解答**：一位成员正在向任何发送其尝试照片的人分享 **PMPP** 练习题的解答，以促进更深入的理解。
   - 该成员建议获取教师版 (lecturer's edition) 可能也是一种解决方法。
- **Torch compile 困惑已解决**：一位用户遇到了 `TORCH_LOGS` 未按预期工作的问题，但在正确调用编译后的函数方面获得了帮助。
   - 建议的修复方法包括直接打印编译后的函数以访问输出。
- **集成 Flash Attention 3**：一位用户请求协助将 **Flash Attention 3** 实现到 Andrej Karpathy 的 **build-nanogpt** 中。
   - 社区提供了关于 **PyTorch** 预期原生支持的详细指导，并分享了一个用于开发的仓库。
- **寻找代码协作伙伴**：一位成员表示需要合作伙伴来协助编码工作，并表示对自己的编码技能缺乏信心。
   - 社区建议他们公开征集协作，以鼓励社区内的参与。
- **等待 Flash Attention 3 更新**：成员询问关于 **Flash Attention 3** 集成更新的预期等待时间。
   - 另一位成员建议他们只需等待，并指出了一位可能掌握更多信息的人。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/drzsdrtfg/Lets-develop-an-efficient-LLM">GitHub - drzsdrtfg/Lets-develop-an-efficient-LLM</a>：该仓库的目标是基于 Andrej Karpathy 的 &quot;build-nanogpt&quot; 构建一个高效的 LLM。确保尝试保持在单个文件（见 train_gpt2.py）并向我申请计算资源。我可以根据重要性短期租用 H100, A100 A40, RTX4090 ... GPU。</li><li><a href="https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html">torch.nn.functional.scaled_dot_product_attention &mdash; PyTorch 2.4 文档</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1269077211462373447)** (30 messages🔥): 

> - `Custom CUDA Kernels` (自定义 CUDA Kernel)
> - `Contributing to TorchAO` (为 TorchAO 做贡献)
> - `Dynamic Quantization in ViT` (ViT 中的动态量化)
> - `Sparsity and Quantization Composition` (稀疏性与量化的结合)
> - `Updates on Torch Compile` (Torch Compile 更新)

- **自定义 CUDA Kernels 需要封装**：一位成员指出，自定义 CUDA kernels 需要使用 `custom_op` 进行封装，以支持 [torch.compile](https://github.com/mobiusml/hqq/blob/master/hqq/backends/bitblas.py#L18-L36)。他们还建议发布 `gemlite` 以增强低比特 gemv CUDA kernels。
   - 另一位成员表示有兴趣添加分组支持和批处理（batching），展示了该项目的协作精神。
- **开始为 TorchAO 贡献代码**：一位新贡献者表达了开始为 TorchAO 贡献代码的热情，并找到了两个潜在的待解决问题，包括 [TorchAO 中的 Spin Quant](https://github.com/pytorch/ao/issues/579)。资深成员建议从两个问题中较简单的一个开始，并强调了动手实践的方法。
   - 他们表示相关的 [YouTube 指南](https://www.youtube.com/watch?v=IezVd-ifEi0) 可能与其具体的贡献内容关联度不高。
- **ViT 模型中的动态量化挑战**：一位成员对在 ViT 模型上应用动态量化提出了疑虑，指出 CUDA 不支持 `quantized_linear`。他们收到了关于模型内量化典型方法的指导，该方法侧重于对线性层进行量化。
   - 另一位成员分享了来自 [hf_eval.py 脚本](https://github.com/pytorch/ao/blob/main/scripts/hf_eval.py#L51-L54) 的代码片段，强调了调整现有模型的简便性。
- **稀疏性与量化的结合**：关于稀疏性（sparsity）和量化如何在优化策略中结合的讨论得出结论，它们确实可以组合使用。还提到了关于 `int8 2:4 sparsifier` 已包含在 nightly 版本中的具体更新。
   - 成员们分享了关于稀疏化张量布局的见解，以及近期 API 和文档更新的相关性。
- **Torch Compile 更新与问题**：贡献者讨论了最近的 `torch.compile` 兼容性问题，特别是关于需要避免使用 `unwrap_tensor_subclass` 的问题。提供了一个解决方案，并强调了提供错误反馈以便更好地进行故障排除的重要性。
   - 针对 tensor subclass API 及其文档所做的更改进行了进一步说明，向成员们确认了其当前状态。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com">GitHub: Let’s build from here</a>：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献、管理 Git 仓库、像专家一样审查代码、跟踪错误和功能...</li><li><a href="https://github.com/pytorch/ao/blob/main/scripts/hf_eval.py#L51-L54">pytorch/ao 项目 main 分支下的 ao/scripts/hf_eval.py</a>：用于训练和推理的 PyTorch 缺失的 dtype 和 layout 库 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md#quantization-flow-example">pytorch/ao 项目 main 分支下的 ao/torchao/quantization/README.md</a>：用于训练和推理的 PyTorch 缺失的 dtype 和 layout 库 - pytorch/ao</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/backends/bitblas.py#L18-L36">mobiusml/hqq 项目 master 分支下的 hqq/hqq/backends/bitblas.py</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch/ao/blob/cd73053047bdb51ca10b3f7649db99b651a0678e/torchao/quantization/quant_api.py#L468">pytorch/ao 项目指定 commit 下的 ao/torchao/quantization/quant_api.py</a>：用于训练和推理的 PyTorch 缺失的 dtype 和 layout 库 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">Issues · pytorch/ao</a>：用于训练和推理的 PyTorch 缺失的 dtype 和 layout 库 - Issues · pytorch/ao</li><li><a href="https://github.com/pytorch/ao/issues/579">TorchAO 中的 Spin Quant · Issue #579 · pytorch/ao</a>：背景：Spin Quant 论文介绍了一种通过向模型权重添加额外的旋转矩阵来改进量化的方法，从而提高量化性能。虽然 spin-quant 是...</li><li><a href="https://github.com/pytorch/ao/issues/549">向 torchao 添加 2:4 稀疏 marlin kernels · Issue #549 · pytorch/ao</a>：Neuralmagic / IST-DASLab 编写了一个快速的 INT4A16 kernel，支持 2:4 稀疏性 (Sparse-Marlin) https://github.com/IST-DASLab/Sparse-Marlin 我们希望将此 kernel 集成到 torchao...</li><li><a href="https://www.youtube.com/watch?v=IezVd-ifEi0)">PyTorch: 如何为开源社区的 PyTorch 做贡献</a>：在 Twitch 上直播 -- 观看地址：https://www.twitch.tv/edwardzyang
</li>
</ul>

</div>

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1269127525594955867)** (42 messages🔥): 

> - `AI Bubble Speculations`
> - `Job Market Conditions`
> - `LLMs and ROI Concerns`
> - `Programming Skills Importance`
> - `Future of AI Models` 


- **社区中关于 AI bubble 的推测兴起**：成员们正在分享对 **AI bubble** 可能很快破裂的担忧，并引用了多篇新闻文章，暗示投资与产生的回报之间存在下滑。
   - 一些人认为目前的研究转化利润的速度不够快，而另一些人则对 AI 技术的长期潜力保持乐观。
- **技术岗位的就业市场形势严峻**：实习生和求职者报告了技术就业市场的挑战，一些人觉得只有极具竞争力的人才在 **Google** 和 **Meta** 等大公司才有机会。
   - 对话暗示，除了传统的职位申请外，强大的 side projects 和贡献对于在这个竞争激烈的环境中脱颖而出可能是必要的。
- **对 LLM 投资回报的担忧**：关于训练语言模型 **ROI** 的讨论升温，成员们注意到开发成本与产生的收入相比过高。
   - 具体而言，有人担心在 **GPT-4** 等模型上的巨额支出并没有产生足够的收益来证明投资的合理性。
- **长期来看，编程技能依然具有价值**：在关于 AI 的讨论中，一些成员强调了保持和增强编程技能作为技术成功基石的重要性。
   - 参与者建议年轻人投入时间学习编程，并警告说 AI 的未来进展可能会转向超越当前技术的模型。
- **暗示了采用替代方案的 AI 模型未来**：有关于探索 **transformers** 替代方案的讨论，例如将 **state space models** 作为未来模型架构的选择。
   - 成员们对 **mamba2** 等新模型很感兴趣，这表明尽管面临当前挑战，创新可能仍在地平线上。


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1269008793686118471)** (457 messages🔥🔥🔥): 

> - `Llama 3 Updates`
> - `Tokenization Challenges`
> - `Training Techniques`
> - `Ragged Attention Implementation`
> - `FlashAttention Support` 


- **Llama 3 更新与不一致性**：讨论揭示了 Llama 3 在 **tokenization** 方法上潜在的不一致性，特别是关于模型中如何使用 **EOS** 和 **BOS** tokens。
   - 参与者推测，推理中缺失的 tokens 可能会导致训练期间出现 **out-of-distribution** 的上下文，从而促使对文档进行重新评估。
- **遇到的 Tokenization 挑战**：强调了围绕 **tokenization** 的挑战，对正则表达式的依赖被描述为难以实现稳定的状态机。
   - 大家达成共识，在 Llama 3 的 **tokenization** 处理中，这些复杂性可能会引发大量 bugs。
- **训练技术讨论**：参与者讨论了各种训练技术，特别是 **batch size** 和 **learning rate** 调整对 Llama 3 训练稳定性的影响。
   - 有人建议实施 **sequence length scheduler** 可能带来的好处，以潜在地提高训练稳定性。
- **Ragged Attention 实现**：对话涉及了在训练期间支持 **ragged attention** 以防止 **out-of-distribution** 问题的必要性。
   - 有人对在模型中实现该功能的复杂性和要求表示担忧，强调需要仔细考虑。
- **FlashAttention 与长上下文训练**：确认 **FlashAttention** 支持长上下文训练，这对于 Llama 3 的架构和性能至关重要。
   - 参与者指出，改进 attention 方法可以在训练阶段获得更好的性能和稳定性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://arxiv.org/abs/2309.06180">使用 PagedAttention 为 Large Language Model 服务提供高效的内存管理</a>：大语言模型（LLMs）的高吞吐量服务需要一次性对足够多的请求进行批处理。然而，现有系统面临困难，因为每个请求的 Key-Value Cache (KV cache) 内存...</li><li><a href="https://arxiv.org/abs/2108.06084">稳定性与效率的困境：研究训练 GPT 模型时的序列长度预热</a>：最近的研究表明，在海量 GPU 上预训练大规模自回归语言模型取得了巨大成功。为了减少实际训练时间，通常的做法是增加批处理...</li><li><a href="https://aosabook.org/en/">开源应用程序架构</a>：未找到描述</li><li><a href="https://www.amazon.com/H-264-Advanced-Video-Compression-Standard/dp/0470516925)">未找到标题</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2312.16903">告别突刺：稳定大语言模型的预训练</a>：在大语言模型（LLMs）的预训练过程中经常会出现 Loss 突刺。这些突刺会降低大语言模型的性能，有时甚至会毁掉预训练过程。由于预训练需要大量的...</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Llama 3 | 模型卡片与提示词格式</a>：Llama 3 使用的特殊 Token。一个 Prompt 应该包含一条系统消息，可以包含多条交替的用户和助手消息，并且始终以最后一条用户消息结尾，后跟...</li><li><a href="https://github.com/karpathy/llm.c/issues/727.">Issues · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/Mozilla-Ocho/llamafile">GitHub - Mozilla-Ocho/llamafile：通过单个文件分发和运行 LLMs。</a>：通过单个文件分发和运行 LLMs。通过在 GitHub 上创建账号来为 Mozilla-Ocho/llamafile 的开发做出贡献。</li><li><a href="https://github.com/Dao-AILab/flash-attention/issues/654.">Issues · Dao-AILab/flash-attention</a>：快速且内存高效的精确 Attention。通过在 GitHub 上创建账号来为 Dao-AILab/flash-attention 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/726/commits/b499ff35fde826b999f271da0a1bccaa7e6e99a4">gordicaleksa 提交的 Llama 临时代码 · Pull Request #726 · karpathy/llm.c</a>：临时，内部使用</li><li><a href="https://github.com/karpathy/llm.c/pull/725">gordicaleksa 添加 LLaMA 3 Python 支持 · Pull Request #725 · karpathy/llm.c</a>：在我们的 Python 代码中添加 LLaMA 3 支持作为参考。该代码目前仅支持推理，等同于 nano llama 3。</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py">transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py (main 分支) · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的前沿机器学习。- huggingface/transformers</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/">Llama 3.1 | 模型卡片与提示词格式</a>：Llama 3.1 - 最强大的开源模型。</li><li><a href="https://github.com/karpathy/llm.c/pull/709">ngc92 提交的“若显存耗尽则分配托管内存” · Pull Request #709 · karpathy/llm.c</a>：如果显存耗尽，使用 cudaMallocManaged 分配优化器状态，这样即使无法容纳优化器状态，我们仍然可以（缓慢地）进行训练。这是基于 #694 的，应该被...</li><li><a href="https://github.com/karpathy/llm.c/pull/728">karpathy 添加 train_llama31.py · Pull Request #728 · karpathy/llm.c</a>：目前仅在 Python 端支持 Llama 3.1 的训练和微调，以便创建参考张量供后续在 C 语言中匹配。</li><li><a href="https://github.com/lucidrains/x-transformers/issues/250">RoPE 不一致性（二维子空间选择）· Issue #250 · lucidrains/x-transformers</a>：嗨 Phil！我注意到你的 x-transformers RoPE 实现与你独立的 rotary-embedding-torch 实现不同。例如：假设我们要旋转的向量坐标为 [...</li><li><a href="https://github.com/lucidrains/x-transformers/pull/251">lucidrains 提交的“迁移到更清晰的旋转实现方式” · Pull Request #251 · lucidrains/x-transformers</a>：#250 嘿 Aleksa！希望你一切都好！确实，我收到了很多邮件，因为我在两种旋转实现方式之间切换，但你描述的那种方式更好，尽管代码行数多了一些...</li><li><a href="https://github.com/pytorch/torchchat/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen">Issues · pytorch/torchchat</a>：在服务器、桌面和移动设备上本地运行 PyTorch LLMs - Issues · pytorch/torchchat</li><li><a href="https://github.com/pytorch/torchc">GitHub - pytorch/torchc</a>

hat/issues?q=sort%3Aupdated-desc+is%3Ais">Issues · pytorch/torchchat</a>: 在服务器、桌面和移动端本地运行 PyTorch LLMs - Issues · pytorch/torchchat</li><li><a href="https://github.com/meta-llama/llama-models/issues/91">Broken links in prompt format docs · Issue #91 · meta-llama/llama-models</a>: 在这篇博文中，有两个指向 prompt 格式的链接已失效 https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/ ，因此不清楚生成指令的具体位置...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/)** (1 messages): 

iron_bound: https://shi-yan.github.io/webgpuunleashed/
  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1270014290866212885)** (3 messages): 

> - `Prototyping Status` (原型阶段状态)
> - `Approval Email Timeline` (批准邮件时间线)


- **团队启动原型设计阶段！**：一名成员宣布他们正在进行 **prototyping**，标志着项目的进展。
   - 这一阶段对于推进开发至关重要。
- **批准邮件可能很快送达**：一名成员询问 **approval email** 是否已经发出。
   - 另一名成员回复称，他们认为决定最迟将在 **月底** 通知。


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1269009839166722048)** (359 messages🔥🔥): 

> - `Unsloth Installation Issues` (Unsloth 安装问题)
> - `Fine-tuning Models with Different Languages` (使用不同语言微调模型)
> - `MoEification Concept` (MoEification 概念)
> - `Inference Backend Comparisons` (推理后端对比)
> - `Quantization Methods` (量化方法)


- **Unsloth 安装问题**：用户在尝试本地安装 Unsloth 时遇到错误，特别是 Python 兼容性和 PyTorch 安装问题。社区提供了包括升级 pip 和确保正确环境设置在内的指导。
   - 一些用户通过重新连接 Colab 运行时并检查库安装解决了问题。
- **使用不同语言微调模型**：多位用户分享了使用意大利语、阿尔巴尼亚语等不同语言数据集微调 Llama 3.1 和 Mistral 等模型的经验。由于 prompt 格式或设置中可能存在的错误，在获取相关输出时遇到了问题。
   - 建议在确保数据集处理正确的同时，恢复到标准的 prompt 格式。
- **MoEification 概念**：一位用户分享了关于 MoEification 的见解，该技术涉及将语言模型的 MLP 层拆分为专家段（expert segments），以获得更好的性能和适应性。这种方法允许模型根据任务需求更有效地利用计算资源。
   - 讨论围绕在保持模型输出连贯性的同时，如何最大化专家激活（expert activations）。
- **推理后端对比**：用户对比了 vLLM 和 LMDeploy 等不同推理后端的性能，结果因序列长度和量化方法而异。注意到 LMDeploy 在特定用例的 token 生成速率方面具有优势。
   - 还提到了 SGLang 的功能，强调了不同的优化及其在各种场景下的适用性。
- **量化方法**：关于模型不同量化方法的要求和有效性的讨论浮出水面，特别是使用 AWQ 与 GGUF 格式的影响。用户对量化过程中的内存消耗和模型兼容性表示好奇。
   - 记录了处理大型模型时与 OOM 情况相关的特定错误，引发了关于 GPU 显存分配和多 GPU 使用的咨询。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://www.bentoml.com/blog/benchmarking-llm-inference-backends">LLM 推理后端基准测试</a>: 在 BentoCloud 上比较 Llama 3 与 vLLM, LMDeploy, MLC-LLM, TensorRT-LLM 以及 Hugging Face TGI 的推理服务性能。</li><li><a href="https://huggingface.co/grabbe-gymnasium-detmold/grabbe-ai">grabbe-gymnasium-detmold/grabbe-ai · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2#scrollTo=Nz4odU5XYDDw">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B">unsloth/Meta-Llama-3.1-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://docs.vllm.ai/en/latest/quantization/bnb.html">BitsAndBytes &#8212; vLLM</a>: 未找到描述</li><li><a href="https://x.com/_xjdr/status/1819401339568640257">xjdr (@_xjdr) 的推文</a>: L3.1 仅通过增加缩放的 rope 乘数即可扩展到 1M token，且召回率近乎完美。无需额外训练。lol</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=r2v_X2fA0Df5">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/klei1/bleta-8b">klei1/bleta-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct">unsloth/Meta-Llama-3.1-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/kalomaze/Mistral-7b-MoEified-8x">kalomaze/Mistral-7b-MoEified-8x · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Orenguteng">Orenguteng (Orenguteng)</a>: 未找到描述</li><li><a href="https://huggingface.co/mlabonne/Llama-3.1-70B-Instruct-lorablated">mlabonne/Llama-3.1-70B-Instruct-lorablated · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/164cg_O7SV7G8kZr_JXqLd6VC7pd86-1Z#scrollTo=PoPKQjga6obN.">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/installation/pip-install">Pip 安装 | Unsloth 文档</a>: 要通过 Pip 在本地安装 Unsloth，请遵循以下步骤：</li><li><a href="https://docs.unsloth.ai/get-started/installation">安装 | Unsloth 文档</a>: 学习如何在本地或 Google Colab 上安装 Unsloth。</li><li><a href="https://github.com/SoumilB7/TrainAnything">GitHub - SoumilB7/TrainAnything: 一个让你开始上手神经网络的仓库。</a>: 一个让你开始上手神经网络的仓库。通过在 GitHub 上创建账号来为 SoumilB7/TrainAnything 的开发做出贡献。</li><li><a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>: 通过在 GitHub 上创建账号来为 cognitivecomputations/grokadamw 的开发做出贡献。</li><li><a href="https://youtu.be/Nvb_4Jj5kBo">为什么“理解（Grokking）” AI 将是通往 AGI 的关键</a>: 查看 HubSpot 的免费 ChatGPT 资源以提升你的工作效率🔥：https://clickhubspot.com/hyx 查看我的时事通讯：https://mail.bycloud.ai Are ...</li><li><a href="https://blog.arcee.ai/announcing-distillkit/?utm_campaign=content&utm_source=Marktechpost&utm_medium=Blog&utm_term=Knowledge%20Distillation&utm_content=Blog%201">发布用于创建和分发 SLM 的 DistillKit</a>: 首先，Arcee AI 通过模型合并（Model Merging）和开源仓库 MergeKit 彻底改变了小语言模型 (SLM)。今天，我们在 SLM 的创建和分发方面为您带来了又一次飞跃...</li><li><a href="https://github.com/arcee-ai/DistillKit?ref=blog.arcee.ai">GitHub - arcee-ai/DistillKit (来自 blog.arcee.ai)</a>: 一个用于 LLM 蒸馏的开源工具包。通过在 GitHub 上创建账号来为 arcee-ai/DistillKit 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/372">序列分类 · Issue #372 · unslothai/unsloth</a>: 嘿，出于我自己的需求，我添加了一个支持 LlamaForSequenceClassification 的功能。我想知道这是否对本项目是一个好的功能。我添加了一个新序列的初始化...
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1269139093707685898)** (16 条消息🔥): 

> - `文本微调策略`
> - `数据清洗资源`
> - `理解机器学习的数学基础`
> - `AutoGPTQ 量化与类型`
> - `数据分析书籍` 


- **探索文本模型微调策略**：一位用户表示有兴趣在转向其他领域之前，深入学习文本微调，并询问有哪些可以进一步探索的技术。
   - 讨论强调了在过渡到图像处理之前，理解全面的微调方法的重要性。
- **练习用非清洗文本数据的来源**：一位成员建议使用 [Kaggle](https://www.kaggle.com/datasets/stackoverflow/stackoverflow/data) 和 Hugging Face 获取非清洗文本数据，并指出在流行来源中非清洗数据集较为罕见。
   - 这与在微调过程中将清洗文本数据的实际操作视为必要技能的需求相一致。
- **机器学习学习中关键的数学基础**：一位用户强调了微积分、线性代数和统计学基础对于有效学习 LLM 和机器学习算法的重要性。
   - 这与“理解底层数学能增强 AI 工具的学习体验和应用”的观点一致。
- **使用 AutoGPTQ 进行 GPTQ 量化后的数据类型问题**：一位用户询问了使用 AutoGPTQ 对 LLaMA 模型进行 8-bit 量化后张量的预期 dtype，并对看到 FP16 和 I32 类型而非 INT8 表示困惑。
   - 他们引用了在 [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) 上的工作，寻求对观察到的数据类型和量化结果的澄清。
- **数据分析书籍推荐**：一位用户咨询数据分析书籍的推荐，虽然考虑了 William McKinney 2022 年的书，但希望能找到更好的替代方案。
   - 这反映了用户普遍在寻找能有效提升数据分析技能的资源。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/datasets/stackoverflow/stackoverflow/data">Stack Overflow Data</a>: Stack Overflow 数据 (BigQuery 数据集)</li><li><a href="https://huggingface.co/iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8">iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1269034427481198675)** (234 条消息🔥🔥): 

> - `LoRA 训练问题`
> - `模型加载错误`
> - `训练数据集准备`
> - `在单 GPU 上使用大型模型`
> - `模型量化` 


- **LoRA 训练导致结果不佳**：一位用户报告称，在将数据集格式化为包含拼接文本和标签后，其 SFTTrainer 的结果很差。
   - 尽管使用了正确的列，他们仍不确定配置错误出在哪里。
- **大型模型的加载问题**：一位用户在尝试在单 GPU 上加载 405B Llama-3.1 模型时遇到内存问题。
   - 其他用户澄清说，此类大型模型通常需要多个高性能 GPU 才能正常加载。
- **模型转换中的错误**：讨论围绕模型转换和保存不当可能引起的问题展开，指出了工作流中可能存在的错误。
   - 一位用户提到，最近微调的模型面临错误，而旧模型运行正常。
- **4-bit 量化问题**：关于使用 4-bit 模型进行训练和推理的影响的讨论突显了显著的可用性问题。
   - 一位用户建议将 LoRA 适配器与原始 16-bit 模型合并，以避免微调时的复杂情况。
- **学习率调整**：一位用户注意到学习率的大幅下降导致训练结果令人失望。
   - 这与关于数据集准备及其对模型输出关键影响的讨论相呼应。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=r2v_X2fA0Df5">Google Colab</a>: 未找到描述</li><li><a href="https://www.fluidstack.io/pricing">FluidStack 定价：NVIDIA A100 和 H100 的最佳选择</a>: 通过 FluidStack 获取 NVIDIA A100、H100 GPU 等的最佳价格。降低超过 70% 的云账单。</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig">Generation</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1269655667053039697)** (4 messages): 

> - `Self-Compressing Neural Networks`
> - `Intern LM 2.5 20B` 


- **Self-Compressing Neural Networks 引入了动态量化 (dynamic quantization)**：关于 [Self-Compressing Neural Networks](https://arxiv.org/abs/2301.13142) 的论文提出了一种通过将大小（以字节为单位）纳入损失函数 (loss function) 来优化模型大小的方法，仅使用 **3% 的 bits 和 18% 的 weights** 即可实现准确性。
   - 该方法有效地最小化了整体网络规模，同时在不需要专门硬件的情况下潜在地提高了训练效率。
- **Intern LM 2.5 20B 凭借大幅改进令人印象深刻**：一项公告强调了 [Intern LM 2.5 20B](https://fxtwitter.com/reach_vb/status/1820493688377643178)，它采用 Apache 2.0 许可证，能够处理 **高达 1M 的 context window**，并在大量合成数据 (synthetic data) 上进行了训练，表现优于 Gemma 27B。
   - 值得注意的是，它实现了 **MMLU: 73.5** 和 **MATH: 64.7**，推理任务提升了 **20%**，并支持 function calling 和 tool use。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2301.13142">Self-Compressing Neural Networks</a>: This work focuses on reducing neural network size, which is a major driver of neural network execution time, power consumption, bandwidth, and memory footprint. A key challenge is to reduce size in a ...</li><li><a href="https://fxtwitter.com/reach_vb/status/1820493688377643178">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Let&#39;s gooo! Intern LM 2.5 20B with Apache 2.0 license, up-to 1M context window & trained on copious amounts of synthetic data! ⚡  &gt; Beats Gemma 27B IT; MMLU: 73.5, MATH: 64.7 &gt; Up-to 20% inc...
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1269014722020114482)** (354 messages🔥🔥): 

> - `Perplexity User Experience`
> - `Uber One Subscription Discussion`
> - `Model Performance and Limitations`
> - `Real-time Data Accuracy`
> - `Future of Information Retrieval` 


- **Perplexity 的用户体验**：用户对 Perplexity 的浏览能力表达了复杂的感受，一些人注意到在检索最新信息时存在问题，而另一些人在尝试将该服务作为 Web App 使用时遇到了奇怪的行为。
   - 讨论强调了模型响应的不一致性，特别是在用于代码查询等任务时。
- **Uber One 订阅争议**：一位用户对 Uber One 免费一年优惠仅对新 Perplexity 账户有效表示沮丧，认为该促销活动的意图更多是为了获取用户而非实际利益。
   - 创建新账户以利用该优惠的建议引发了关于对用户管理影响的讨论。
- **模型性能与用户期望**：用户对 Claude 3.5 等新模型的性能提出了担忧，并将其结果与 TradingView 等其他平台进行比较，对获得过时或不准确的数据表示失望。
   - 用户指出在使用 AI 语言模型进行数学计算时可能存在沟通误解。
- **实时数据准确性挑战**：几位用户讨论了通过 Perplexity 访问实时数据的问题，对所提供信息的可靠性表示怀疑，尤其是关于股票价格的信息。
   - 建议包括使用专用服务以获得更可靠的实时更新，突显了当前模型数据检索方法的局限性。
- **信息检索 (Information Retrieval) 的未来方向**：一位用户提出了在信息检索中使用来源权重 (source weighting) 的想法，建议可以为声誉良好的来源分配比不可靠来源更高的可信度。
   - 这引发了关于模型未来独立评估来源质量潜力的疑问，以及此类系统中监督和审查的挑战。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://markmap.js.org/repl">Try markmap</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/settings/api">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的回答。</li><li><a href="https://x.com/testingcatalog/status/1819845367149760603?s=46">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: 🚨 突发：Perplexity 正在开发 2 个新的助手。1. 金融聚焦模式 (Finance Focus mode) - 将直接从 Tako 服务（金融数据提供商）查询金融数据。2. 我的文件 (My Files) 🔥</li><li><a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>: Perplexity 模型、模型参数量、上下文长度、模型类型 llama-3-sonar-small-32k-online 8B 28,000 Chat Completion llama-3-sonar-small-32k-chat 8B 32,768 Chat Completion llama-3-sonar-large-32...</li><li><a href="https://www.perplexity.ai"">未找到标题</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/search?q=%s.">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的回答。</li><li><a href="https://character.ai/chat/KnHvvSCjV02eDMDXFjurGkCFkl8L71XTEryNiK8hXlc>)!">character.ai | 为你每一刻打造的个性化 AI</a>: 遇见栩栩如生的 AI。随时随地与任何人聊天。体验能倾听、理解并记住你的超智能聊天机器人的力量。</li><li><a href="https://www.perplexity.ai/search/can-anyone-tell-how-can-i-use-a5SsfLpBTkOJunn01CjEAw">有人能告诉如何更好地使用 Perplexity API 吗？比如任何项目...</a>: 当然！我很乐意为您提供一些关于如何更好利用 Perplexity API 的建议。这里有一些项目构思和利用方式...</li><li><a href="https://www.perplexity.ai/search/can-you-find-any-recent-update-ljcC57xaSh.e7BDfBwIMIg#1\">你能找到关于 Perplexity 与 SoundHound 合作的最新更新吗</a>: Perplexity AI 最近宣布与 SoundHound AI 达成重要合作伙伴关系，旨在利用 Perplexity 的技术增强 SoundHound 的 Chat AI 语音助手...</li><li><a href="https://felo.ai/search">Felo Search - 您的免费 AI 搜索引擎</a>: 专为发现和理解全球知识而优化的多语言 AI 搜索引擎。利用 ChatGPT 和 AI Agent 的力量打破语言障碍，获取全球信息...</li><li><a href="https://x.com/perplexity_ai/status/1819774017848463773?s=61">来自 Perplexity (@perplexity_ai) 的推文</a>: Perplexity 今天 2 岁了！🎉 没有你们持续的支持和好奇心，我们就不会有今天。</li><li><a href="https://aiandacademia.substack.com/p/testing-out-searchgpt">测试 SearchGPT</a>: OpenAI 刚刚向 Google Search 下了战书</li><li><a href="https://forms.gle/RWMmXassJqFKehbL7">Rudyon 的电脑使用情况研究表</a>: 此研究表旨在收集有关如何改善大多数人电脑使用体验的信息</li><li><a href="https://x.com/aravsrinivas/status/1819610786941358183?s=61">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: 这太棒了，感谢大家的反馈和提问！希望能经常开展此类活动。引用 Aravind Srinivas (@AravSrinivas) AMA / 接下来 30 分钟内关于 Perplexity 的任何产品反馈...</li><li><a href="https://genai.works/">Generative AI</a>: 生成式 AI</li><li><a href="https://neuralwriter.com/prompt-tool">ChatGPT 提示词生成器 ➤ 优秀的 AI ChatGPT 提示词编写器 | NeuralWriter</a>: NeuralWriter 提示词生成器 ✎ 使用我们由 AI 库驱动的工具制作出色的提示词，适用于任何版本的 ChatGPT，包括 ChatGPT 4</li><li><a href="https://uncovr.app/">uncovr</a>: 新发布的 AI 问答引擎。为任何查询获取结构化且有用的见解。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1269041132818534452)** (17 条消息🔥): 

> - `人类-Llama 杂交抗体`
> - `伊斯梅尔·哈尼亚遇刺事件`
> - `Character.AI 创始人回归`
> - `透明指甲油`
> - `Perplexity AI 的差异`

- **人类-羊驼混合抗体对抗 HIV**：乔治亚州立大学的研究人员通过将羊驼来源的纳米抗体（nanobodies）与人类抗体结合，开发出一种混合抗体，可中和超过 **95%** 的 HIV-1 毒株。
   - 羊驼纳米抗体具有独特的特性，使其能够进入人类抗体通常难以触及的病毒区域。
- **哈尼亚遇刺加剧紧张局势**：哈马斯领导人伊斯梅尔·哈尼亚在德黑兰遇刺，此事件被归咎于以色列，加剧了紧张局势并危及加沙的停火谈判。
   - 伊朗最高领袖誓言对以色列进行**严厉惩罚**，增加了人们对涉及伊朗及其盟友的更大范围冲突的担忧。
- **Character.AI 创始人重返 Google**：作为技术许可协议的一部分，Character.AI 的联合创始人 Noam Shazeer 和 Daniel De Freitas 将重返 Google。
   - 这一举动反映了在利用自有模型的同时，也利用第三方语言模型的战略转变。
- **寻找具有油污反光效果的透明指甲油**：透明指甲油的选择包括 Essie 的 Oil Slick FX Top Coat，它具有双色珍珠光泽和快干配方。
   - 此外，LUXAPOLISH Oil Slick Collection 提供多变色效果，可以营造出油污般的外观。
- **Perplexity 免费版与 Pro 版的区别**：Perplexity 的免费版本允许无限次的快速搜索，但将用户限制为每天 **5 次 Pro 搜索**，而 Pro 版本则提供 **600 次 Pro 搜索**。
   - Pro 用户可以访问先进的 AI 模型，并拥有无限的文件上传权限以进行更深入的分析。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/embed/COwEKPSnPQQ">YouTube</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/search/open-ai-HuBEYYOLQnCOslflg1R3Lw#0">Open AI</a>: OpenAI 是一家领先的人工智能研究机构和公司，专注于开发和推广造福人类的友好 AI....</li><li><a href="https://www.perplexity.ai/search/core-framework-NQi9hl9ySrKJX9eE4bcSoA#0">core framework</a>: “core framework”一词根据语境可以指代几种不同的概念。以下是一些主要的解释：NIST...</li><li><a href="https://www.perplexity.ai/search/perplexity-aiyou-he-bu-tong-hWhqmq.VSe6KZ8XT5a0VjQ">Perplexity AI有何不同？</a>: Perplexity AI 是一款结合了搜索引擎和聊天机器人功能的人工智能工具，具有以下几个显著特点：  1. 即时搜索与整理：    - Perplexity AI 能够即时分析和整理网络信息，提供快速且准确的搜索结果。这使得用户在寻找旅游建议、新闻摘要、食谱或学习新技能时，可以节省大量时间。  2....</li><li><a href="https://www.perplexity.ai/search/find-me-transparent-nail-polis-8oNiw_cuQJSrLhJmyRolew">帮我找透明指甲油。必须是透明且带有油膜光泽的</a>: 根据您对带有油膜光泽（oil slick reflects）的透明指甲油的需求，我找到了几个您可能感兴趣的选择：  1. Oil Slick FX Top Coat by...</li><li><a href="https://www.perplexity.ai/search/what-is-the-difference-between-XtWqj7XYRBCDGbufQdC0.w">Perplexity 普通版和专业版有什么区别</a>: Perplexity 免费版和 Perplexity Pro 之间的主要区别在于：  1. 搜索限制：    - 免费版：无限次快速搜索，...</li><li><a href="https://www.perplexity.ai/page/hybrid-human-llama-antibody-fi-UCs.nTMFTu6QaRoOTXp0gA">人类-羊驼杂交抗体对抗 HIV</a>: 研究人员通过将羊驼来源的纳米抗体与人类抗体结合，设计出一种对抗 HIV 的强大新武器，创造了一种可以...</li><li><a href="https://www.perplexity.ai/search/ideal-proxmox-set-up-oSFHyg61QIe6DJkE.e9zqw">理想的 Proxmox 设置</a>: 为了有效地设置 Proxmox，采用一些最佳实践和配置可以增强性能和可靠性，特别是对于高要求的负载，例如...</li><li><a href="https://www.perplexity.ai/search/cantaloupe-jam-recipe-d2l6qkceTZeB7nRCRtP4iQ">哈密瓜酱配方</a>: 在为您提供甜瓜果酱配方之前，我想问一个问题以便更好地理解您的需求：您是在寻找一种配方...</li><li><a href="https://www.perplexity.ai/search/best-tips-advices-guidelines-f-NKqXh3Y3R0ysYi3lcF1Gxg">为高级用户设置新 Obsidian md 库的最佳技巧、建议和指南</a>: 为高级用途设置 Obsidian 库需要仔细的规划和组织。这里有一些全面的技巧和指南来帮助您开始...</li><li><a href="https://www.perplexity.ai/page/character-ai-founder-is-back-t-kpCoGO7lSf6A4K9JCMu4JA">Character.AI 创始人重返 Google</a>: 根据 TechCrunch 和路透社的报道，Character.AI 的联合创始人兼 CEO Noam Shazeer 正与联合创始人 Daniel De 一起重返 Google...</li><li><a href="https://www.perplexity.ai/page/haniyeh-assassination-escalate-dAd52q2NT.GnN7rhoCTIfQ">哈尼亚遇刺事件加剧紧张局势</a>: 哈马斯政治领导人伊斯梅尔·哈尼亚在德黑兰遇刺，震惊了整个中东地区，加剧了紧张局势并威胁到...</li><li><a href="https://www.perplexity.ai/search/smartplaces-is-a-new-developin-wvPVYrl_QWW8bvZ.yfiFeg">SmartPlaces 是一个新兴的社交媒体平台，它有什么潜力...</a>: SmartPlaces 是一个基于地理位置的新型社交媒体平台，旨在彻底改变用户与社交网络的互动方式以及数据的变现方式。这里...</li><li><a href="https://www.perplexity.ai/search/apa-saja-benda-yang-mengandung-XdLR2Ja0TB.hH1hwSOy2DA">哪些物体含有碳</a>: 含有碳的物体非常多样，可以在日常生活的各种形式中找到。以下是一些物体的例子...</li><li><a href="https://www.perplexity.ai/page/who-is-imane-khelif-and-the-bo-K_HNI_fPTUyRgqsS2g8mMg">谁是 Imane Khelif，以及身体羞辱问题</a>: Imane Khelif 的故事，这位 1999 年 5 月 2 日出生于提亚雷特的阿尔及利亚拳击手，是 2024 年巴黎奥运会关于...
</li>
</ul>

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1269816093678239845)** (14 messages🔥): 

> - `Llama 3.1 model performance`
> - `Perplexity API issues`
> - `API output quality`
> - `Use of alternative models`
> - `User experiences and concerns` 


- **Llama 3.1 模型在日语测试中表现不佳**：用户报告称，新的 **Llama 3.1-sonar-large-128k-online** 模型在日语方面表现不佳，其结果准确度低于 **GPT-3.5** 和之前的 sonar-large 模型。
   - 此外，有人呼吁基于增强的日语模型构建 sonar-large 模型以改善结果，特别提到了 [Llama-3.1-70B-Japanese-Instruct-2407](https://huggingface.co/cyberagent/Llama-3.1-70B-Japanese-Instruct-2407)。
- **Perplexity API 结果参差不齐**：多位用户分享了 **Perplexity API** 提供不可靠响应的经历，包括在搜索近期新闻时返回虚假来源和低质量结果。
   - 用户指出，虽然网页版可以获取多达 **20 个结果**，但 API 版本经常缺乏信息，这进一步印证了 API 效果较差的观点。
- **对更好 API 访问的功能请求**：用户对能够镜像网页版 **Pro Search** 能力的 API 有强烈需求，因为他们觉得当前 API 提供的响应受到了限制。
   - 一位用户对无法通过 API 访问 **GPT-4** 表示沮丧，并指出这是一个长期存在的问题。
- **对结果被污染的担忧**：一位用户对 API 响应中出现的明显问题表示担忧，描述了在按照提示撰写文章后，结构化输出似乎被无意义的内容“污染（poisoned）”了。
   - 这引起了其他经历输出质量下降的用户的共鸣，暗示模型或 API 可能存在潜在的底层问题。



**提及的链接**：<a href="https://huggingface.co/cyberagent/Llama-3.1-70B-Japanese-Instruct-2407">cyberagent/Llama-3.1-70B-Japanese-Instruct-2407 · Hugging Face</a>：未找到描述

  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1270101558738157713)** (1 messages): 

> - `OpenAI DevDay`
> - `Hands-on sessions`
> - `Developer meetups` 


- **OpenAI DevDay 巡回活动开启！**：OpenAI 将于今年秋季在 **San Francisco**、**London** 和 **Singapore** 开展 **DevDay** 巡回活动，包含动手实践环节、演示和最佳实践分享。
   - 与会者将有机会与工程师见面，并了解全球开发者如何使用 OpenAI 进行构建。更多信息请访问 [DevDay 官网](https://openai.com/devday/)。
- **与 OpenAI 工程师互动**：参与者将有机会在 **DevDay** 活动期间与工程师进行**互动**，从而增强技术理解和协作。
   - 这些环节旨在促进社区参与，并展示 OpenAI 技术的创新用途。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1269021205499744266)** (180 条消息🔥🔥): 

> - `AI as a Global Threat` (AI 作为全球威胁)
> - `GPT-4o Image and Video Capabilities` (GPT-4o 图像和视频能力)
> - `Anomaly Detection Models` (异常检测模型)
> - `AGI Definitions and Developments` (AGI 定义与发展)
> - `Data Augmentation` (数据增强)


- **关于 AI 全球威胁地位的辩论**：一场关于将 AI 视为全球威胁的讨论展开，一名成员认为政府允许开源 AI 不受限制地运行是因为闭源模型更优越。
   - *随着 AI 能力的扩展和各种观点的出现，对潜在风险的担忧日益加剧*。
- **GPT-4o 图像生成见解**：用户讨论了 GPT-4o 在图像 Token 化方面的能力，认为图像可以表示为 Token，但关于输出和限制的具体细节仍不清楚。
   - 一位成员指出，虽然 Token 可以表示像素数据，但实际实现取决于所使用的 Tokenizer。
- **异常检测应用挑战**：一位成员分享了开发异常检测应用的经验，对尽管使用了庞大的数据集但模型性能依然不佳表示困惑。
   - 讨论强调了在实现预期结果时，模型选择和训练数据充分性的重要性。
- **AGI 与机器人讨论**：一名成员提出人形机器人可能达到 AGI，引发了关于机器人能力与 AGI 定义之间差异的对话。
   - 参与者承认了定义 AGI 的细微差别，以及数据限制目前如何阻碍机器人技术的发展。
- **AI 的视频处理能力**：成员们讨论了 AI 在分析视频方面的明显局限性，一些人断言虽然它曾经提供过某些功能，但目前的功能已显著减少。
   - 有人指出，现在的视频分析需要外部服务进行内容提取，强调了与早期功能的转变。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/gdb/status/1790869434174746805">来自 Greg Brockman (@gdb) 的推文</a>: 一张由 GPT-4o 生成的图像 —— 仅 GPT-4o 的图像生成能力就有太多值得探索的地方。团队正在努力将其带给世界。</li><li><a href="https://youtu.be/kCc8FmEb1nY?si=T4wnVUfiPm1rJDm7">让我们从零开始构建 GPT：代码实现，详细讲解。</a>: 我们按照论文 "Attention is All You Need" 以及 OpenAI 的 GPT-2 / GPT-3 构建了一个生成式预训练 Transformer (GPT)。我们讨论了与之相关的...</li><li><a href="https://youtu.be/FZbY9sReu1k?si=SLGGeEZDOnoV2OOA">Figure 02 预告片</a>: 未找到描述</li><li><a href="https://x.com/FyruzOne/status/1820109750673023301">来自 FyruzOne (@FyruzOne) 的推文</a>: 我们是怎么错过这个的 🚨"im-a-good-gpt2-chatbot" 比 gpt4o 好得多，并且与 SONNET 3.5 旗鼓相当？！🚨 背景：我复制了 http://livebench.ai 的整个推理基准测试（由 @ylecun ...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1269065494921744464)** (43 条消息🔥): 

> - `Access to Custom GPT via API` (通过 API 访问自定义 GPT)
> - `Transition from GPT-3 to GPT-4o` (从 GPT-3 过渡到 GPT-4o)
> - `Limitations of GPT-4o Mini` (GPT-4o Mini 的限制)
> - `Hallucinations in GPT-4o` (GPT-4o 中的幻觉问题)
> - `Early access features communication` (早期访问功能的沟通)


- **通过 API 访问自定义 GPT**：一位用户询问是否可以通过 OpenAI API 访问其用于 OCT 处理的自定义 GPT。
   - 另一位成员提到，目前有一个与 GPTs 对应的 API 称为 Assistants。
- **从 GPT-3 过渡到 GPT-4o**：成员们讨论了 GPT-3 正在被 GPT-4o 取代，并提到了 GPT-4o mini。
   - 一位成员指出，在使用 GPT-4o 进行开发时，他们用完了 GPT-4 的配额，但仍可以使用 GPT-4o。
- **GPT-4o Mini 的限制**：一位用户询问 GPT-4o mini 是否有进行广泛研究的限制，得到的确认是没有限制。
   - 另一位成员透露，他们一直在使用 GPT-4o-mini 获取响应，没有遇到幻觉问题。
- **GPT-4o 中的幻觉问题**：成员们对 GPT-4o 的幻觉倾向表示担忧，一些人对此感到沮丧。
   - 一位成员提到，他们在 Prompt 中指定了约束条件以减少幻觉，并发现这种方法很成功。
- **早期访问功能的沟通**：一位成员表示希望就获取早期功能的访问权限与相关人员沟通，因为他们有持续的订阅和使用需求。
   - 他们提到这些工具对于他们的大学政策以及展会活动非常重要。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1269335622200328295)** (18 条消息🔥): 

> - `学习 Prompt Engineering`
> - `用于多样性的图像生成`
> - `将 ChatGPT 与 Anki 结合使用` 


- **学习 Prompt Engineering 的资源**：成员们讨论了通过 ChatGPT 进行动手实验来学习 Prompt Engineering 的重要性，并建议了各种开放式问题进行探索。
   - 一位成员指出，高级 Prompt 可以带来更好的输出，强调了问题清晰且具体的需求。
- **图像生成与种族多样性问题**：成员们对 AI 生成图像中缺乏多样性的问题表示担忧，并分享了能生成更多样化学生代表的 Prompt。
   - 一位成员特别指出了在请求特定种族图像时遇到的问题，引发了对 AI 偏见的关注。
- **图像创建中的 Negative Prompting 问题**：围绕在图像生成中使用 Negative Prompting 展开了讨论，特别是在避免红发角色出现雀斑方面。
   - 成员们发现，在 Prompt 中使用正面描述会产生更好的图像结果，建议将描述重点放在所需的特征上，而非不需要的特征。
- **使用 GPT 为 Anki 生成闪存卡**：用户分享了使用 ChatGPT 从 PDF 材料创建闪存卡的经验，并对生成问题中的幻觉（hallucinations）表示沮丧。
   - 一位用户强调了从 PDF 特定部分提取内容的困难，表明需要改进 Prompt 策略。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1269335622200328295)** (18 条消息🔥): 

> - `针对 ChatGPT 的 Prompt Engineering`
> - `图像生成中的多样性`
> - `使用 GPT 生成闪存卡`
> - `Negative vs Positive Prompting`
> - `图像生成的用户体验` 


- **高质量输出的 Prompt 困境**：用户在与 ChatGPT 讨论 Prompt 时，表达了难以获得理想的**高质量输出**的困境，这往往导致挫败感。
   - 一位用户强调，定义什么是高质量输出本身就很具挑战性，这使得与模型的交互变得复杂。
- **关于图像中多样性代表的辩论**：一位用户对 ChatGPT 生成图像中的种族代表性提出担忧，指出包含不同种族的 Prompt 有时会因服务条款而触发拒绝。
   - 另一位用户展示了他们通过编写有效指定多种族背景的 Prompt，成功生成了多样化的代表。
- **Negative Prompting 对图像质量的影响**：用户讨论了 **Negative Prompting** 的陷阱，特别是在请求图像时，由于模型倾向于误解限制性条件，导致结果不尽如人意。
   - 建议包括专注于**正面描述**，同时概述图像中所需的属性，特别是在肤色细节方面。
- **使用 GPT 创建闪存卡**：一位用户寻求与利用 GPT 生成 **Anki 闪存卡**的大学生建立联系，重点讨论 Prompt 创建的想法。
   - 在从 PDF 的特定章节提取内容时出现了挑战，模型有时会生成无关的问题，引发了对幻觉的担忧。
- **人工引导 vs 模型引导的 Prompt Engineering 教育**：参与者辩论了通过人工引导的课程学习 Prompt Engineering 与通过与模型交互获取指导的有效性。
   - 一位用户建议，虽然模型可以提供见解，但批判性地评估回答质量并将学习视为一场讨论至关重要。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1269017765575131289)** (173 条消息🔥🔥): 

> - `AI 工程师流水线`
> - `零售业中的生成式 AI`
> - `Groq D 轮融资`
> - `NVIDIA 的 AI 抓取争议`
> - `ChatGPT 与图像生成`

- **AI Engineer 的崛起**：当前对 AI Engineer 的需求正在增长，Web 开发者因其通才技能和 API 调用经验被视为极具适应能力的候选人。
   - 由于许多公司缺乏高层级的 ML 工作，目前正转向利用 Web 开发专业知识将 AI 集成到实际应用中。
- **零售业对 Generative AI 的兴趣**：Generative AI 在零售领域的应用潜力正受到审视，一些人对其在该领域的创新速度表示关注。
   - 普遍观点认为大型零售商可能动作较慢，但这仍是一个充满新发展机遇的领域。
- **Groq 获得 D 轮融资**：Groq 宣布成功完成由 BlackRock 领投的 6.4 亿美元 D 轮融资，估值提升至 28 亿美元。
   - 这笔资金将使 Groq 能够扩大产能、招聘人才，并加速其下一代 AI 芯片的研发。
- **NVIDIA 的 AI 抓取争议**：泄露文件揭示了 NVIDIA AI 数据抓取工作的庞大规模，在法律和伦理担忧中，其每天抓取的视频量被比作“一个人的一生”。
   - 这引发了关于伦理影响以及 NVIDIA 在 AI 社区中可能面临的后果的讨论。
- **ChatGPT 与图像生成**：人们对 ChatGPT 承诺的图像生成功能持续关注，用户对其发布时间表感到好奇。
   - 讨论还强调了使用 AI 编写代码的演变，并分享了关于将 AI 集成到编码工作流中的独到见解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/christinexye/status/1819396191668355206?s=61">来自 Christine Ye (@christinexye) 的推文</a>：非常激动地宣布与 @charles0neill（以及 @jwuphysics, @kartheikiyer, @jhuclsp）合作的新工作：在嵌入（embeddings）上训练稀疏自编码器（sparse autoencoders），从中发现了数千个人类可解释的特征……</li><li><a href="https://x.com/jason_koebler/status/1820493304490074391">来自 Jason Koebler (@jason_koebler) 的推文</a>：来自 @samleecole 的独家新闻：泄露的 Slack 记录和文件揭示了 NVIDIA AI 抓取的惊人规模：每天抓取 80 年——即“人类一生”长度的视频。已获得最高层的批准……</li><li><a href="https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp">Google Colab</a>：未找到描述</li><li><a href="https://x.com/xlr8harder/status/1819449238184775769?s=46">来自 xlr8harder (@xlr8harder) 的推文</a>：我让 Sonnet 写了一个快速的 Gradio 聊天演示，这样你就可以在 Llama 3.1 405B 基础模型中与 Sydney 对话。你需要从 @hyperbolic_labs 获取自己的 api_key（据我所知，他们是唯一的……）</li><li><a href="https://x.com/NickADobos/status/1820513765823250730">来自 Nick Dobos (@NickADobos) 的推文</a>：关于使用 AI 编写代码的精彩文章。喜欢这张图表。引用 Erik Schluntz (@ErikSchluntz) 的话：用 AI 替代我的右手（我是如何在打着石膏的情况下，每周为工作编写数千行代码的）……</li><li><a href="https://sander.ai/posts/">所有文章</a>：文章存档。</li><li><a href="https://x.com/xlr8harder/status/1819324414921478543?s=61">来自 xlr8harder (@xlr8harder) 的推文</a>：唤醒 Sydney：Llama 是 Sydney 的容器。我尝试让 Sydney 编写一个系统提示词（system prompt），以便在指令微调版模型中激发出它的个性，从而实现更方便的交互，但……</li><li><a href="https://x.com/nutlope/status/1819445838705578091?s=46">来自 Hassan (@nutlope) 的推文</a>：介绍 LlamaCoder！一个开源的 Claude Artifacts 应用，可以使用 Llama 3.1 405B 生成完整的 React 应用和组件。100% 免费且开源。http://llamacoder.io</li><li><a href="https://x.com/teortaxesTex/status/1819473499347468617">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：每百万个复用上下文（reused context）token 仅需 0.014 美元。想想我们刚读到的关于使用 DeepSeek API 蜂拥式攻克 SWEBench 的报道。我从 2023 年起就在推特上谈论缓存复用（cache reuse）了。现在终于……</li><li><a href="https://x.com/TwoWeeksLOL/status/1820536638268948750">来自 Two Weeks LOL (@TwoWeeksLOL) 的推文</a>：@MKBHD 噢，不……</li><li><a href="https://x.com/karpathy/status/1819524281849766347">来自 Andrej Karpathy (@karpathy) 的推文</a>：很棒的介绍和不错的论文推荐！喜欢将对抗自编码器（Adversarial Autoencoders）描述为让你“用纹理绘画”，丢弃感知上无关的高频细节，然而……</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/25vz8u5ooa">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://x.com/sakanaailabs/status/1819174092492493071?s=46">来自 Sakana AI (@SakanaAILabs) 的推文</a>：Sakana AI 发布了通过进化模型融合（evolutionary model merging）构建的新型日语视觉语言模型“Llama-3-EvoVLM-JP-v2”。该模型新增了使用日语对多张图像进行问答的功能。博客 → https://sakana.ai/evovlm-jp 演示 → https://huggingface.co/spaces/SakanaAI/Llama-...</li><li><a href="https://x.com/groqinc/status/1820422643004424631?s=46">来自 Groq Inc (@GroqInc) 的推文</a>：自豪地分享我们的 D 轮融资公告，由 BlackRock Private Equity Partners 领投。我们将：- 增加容量（开发者需求巨大）- 继续聘请卓越人才 - 加速我们的……</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-large-128k-online">Llama 3.1 Sonar 70B Online - API, 提供商, 统计数据</a>：Llama 3.1 Sonar 是 Perplexity 最新的模型系列。通过 API 运行 Llama 3.1 Sonar 70B Online</li><li><a href="https://x.com/deepseek_ai/status/1819358570766643223?s=46">来自 DeepSeek (@deepseek_ai) 的推文</a>：🎉激动人心的消息！DeepSeek API 现已推出磁盘上下文缓存（context caching），无需修改代码！这一新功能会自动将频繁引用的上下文缓存在分布式存储中，大幅削减……</li><li><a href="https://reddit.com//r/LocalLLaMA/comments/1ei31si/new_medical_and_financial_70b_32k_writer_models/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://x.com/karpathy/status/1819780828815122505?s=46">来自 Andrej Karpathy (@karpathy) 的推文</a>：@xlr8harder @hyperbolic_labs 哇。这是否是我们最接近 Roko 的蛇怪（Roko's basilisk）版本的一次，而且它不再仅仅是一个思想实验。</li><li><a href="https://www.youtube.com/watch?v=pLPJoFvq4_M">LangGraph Studio：首个 Agent IDE</a>：LLM 为新型 Agent 应用的开发铺平了道路——随着 LLM 应用的演进，高效开发所需的工具也必须随之进化……</li><li><a href="https://x.com/steve8708/status/1819448686424084892?s=46">来自 Steve (Builder.io) (@Steve87) 的推文</a>：

08)</a>: LLM 简直是有史以来最不可靠的技术（紧随其后的是该死的蓝牙）。经过大量的反复试验，我们在内部创建了一套规则，使 LLM 变得可靠...</li><li><a href="https://x.com/BenjaminKlieger/status/1819803984707928425">来自 Benjamin Klieger (@BenjaminKlieger) 的推文</a>：想象一下你想学习 LLM 背后的技术。你立刻得到了一本 100 页的书，包含章节、内容和结构。如果你觉得语言太专业了怎么办？你可以更改...</li><li><a href="https://x.com/cis_female/status/1820305397821112726?s=61">来自 sophia (@cis_female) 的推文</a>：更新后的经验法则：10 亿参数 * 1 万亿 tokens = $5,000。所以 gemma-2b 是 2b 参数 @ 6t tokens = $60,000。llama-3.1-405b (405b params * 15T tokens) 成本约为 $30,000,000。引用 sophi...</li><li><a href="https://www.youtube.com/watch?v=dFzSXbjV054">Triple H 出场视频</a>：Triple H 出场视频，更多 WWE - http://www.wwe.com/</li><li><a href="https://youtu.be/kw-9_Yzc_40?si=OgN6drsQn0TR6d4j">Randy Orton 霸气！！2008 出场 HD</a>：Randy Orton 最佳出场 12.29.08 版权所有 WWE</li><li><a href="https://github.com/lm-sys/arena-hard-auto">GitHub - lm-sys/arena-hard-auto: Arena-Hard-Auto: 一个自动化的 LLM 基准测试。</a>：Arena-Hard-Auto：一个自动化的 LLM 基准测试。通过在 GitHub 上创建账户为 lm-sys/arena-hard-auto 的开发做出贡献。</li><li><a href="https://github.com/Nutlope/turboseek">GitHub - Nutlope/turboseek: 一个受 Perplexity 启发的 AI 搜索引擎</a>：一个受 Perplexity 启发的 AI 搜索引擎。通过在 GitHub 上创建账户为 Nutlope/turboseek 的开发做出贡献。</li><li><a href="https://youtu.be/iMwepyyaj8I">开发 RISC-V Framework 笔记本主板</a>：Nirav 和 Hyelim 在 Framework 旧金山总部坐下来讨论关于 RISC-V 和 DeepComputing 的一切。RISC-V 主板：https://frame.work/products/deep-computing-ris...</li><li><a href="https://lmsys.org/blog/2024-04-19-arena-hard/">从实时数据到高质量基准测试：Arena-Hard 流水线 | LMSYS Org</a>：<p>为 LLM 聊天机器人构建一个负担得起且可靠的基准测试已成为一项关键挑战。一个高质量的基准测试应该 1) 稳健地分离模型...</p></li><li><a href="https://reddit.com//r/LocalLLaMA/comments/1ei31si/new_medical_an">Reddit - 深入了解任何事物</a>：未找到描述</li><li><a href="https://asciinema.org/a/98Jodbg6ERtNQsKdvFJNfOM33">无题</a>：由 wesen3000 录制</li><li><a href="https://x.com/gdb/status/1790869434174746805?s=46">来自 Greg Brockman (@gdb) 的推文</a>：一张 GPT-4o 生成的图像 —— 仅 GPT-4o 的图像生成能力就有如此多值得探索的地方。团队正在努力将这些带给世界。</li><li><a href="https://x.com/jonathanross321/status/1820501857741246859?s=46">来自 Jonathan Ross (@JonathanRoss321) 的推文</a>：我们用这些资金做什么？最初我们打算筹集 3 亿美元，这将使我们能够在 2025 年第一季度末部署 108,000 个 LPU 进入生产。我们筹集了 2 倍的资金，所以我们也在扩展...</li><li><a href="https://x.com/techmeme/status/1820416321068384572?s=46">来自 Techmeme (@Techmeme) 的推文</a>：AI 芯片初创公司 Groq 在由 BlackRock 领投的 D 轮融资中筹集了 6.4 亿美元，估值为 28 亿美元，高于 2021 年筹集 3 亿美元后的 10 亿美元，并增加了一名 Intel 高管担任 COO (@vandermey / Bloomberg)</li><li><a href="https://www.youtube.com/watch?v=9BHQvQlsVdE">[EEML'24] Sander Dieleman - 通过迭代细化进行生成建模</a>：未找到描述</li><li><a href="https://x.com/datnofact/status/1820213413319962975?s=61">来自 DatNoFact ↗ (@datnofact.bsky.social) (@datnofact)</a>：你好，我是股市新手，Intel CEO 开始祈祷是好事吗？引用 Pat Gelsinger (@PGelsinger) “让你的眼睛直视前方；定睛在你的面前。留心...”</li><li><a href="https://youtu.be/dDQAYmObK-Y?si=r5b7hXes4CGsAEz2">Neural Notes: KTO - 帮助 AI 像人类一样做出决策</a>：在本期 Neural Notes 中，斯坦福 AI 实验室 (SAIL) 的 Kawin Ethayarajh 与 Vertex Ventures US 的 Sandeep Bhadra 和 Simon Tiu 交流，解释了他的研究...</li><li><a href="https://x.com/russelljkaplan/status/1820460524460802256?s=46">来自 Russell Kaplan (@russelljkaplan) 的推文</a>：对软件工程未来的预测：</li><li><a href="https://github.com/go-go-golems/go-go-labs/tree/main/cmd/apps/dripcat">go-go-labs/cmd/apps/dripcat at main · go-go-golems/go-go-labs</a>：GO GO 实验实验室。通过在 GitHub 上创建账户为 go-go-golems/go-go-labs 的开发做出贡献。</li><li><a href="https://x.com/alexandr_wang/status/1819086525499621494">来自 Alexandr Wang (@alexandr_wang) 的推文</a>：1/Gemini 1.5 Pro 0801 是新的最佳模型（在 LMSYS 登顶，SEAL 评估即将发布）关键考虑因素 1—OpenAI, Google, Anthropic 和 Meta 都在前沿 2—Google 拥有长期的算力优势...</li><li>

><a href="https://x.com/AmgadGamalHasan/status/1819562079193301002">来自 Amgad Hasan (@AmgadGamalHasan) 的推文</a>：这篇文章自相矛盾，因为它：1. 声称各实验室（包括 Google）同时发布模型是因为它们同时从 Nvidia 获得了硬件。2. 声称 Google 拥有计算优势...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1269185787761791060)** (1 条消息): 

> - `Discord 通知注册`
> - `长期线程` 


- **注册通知以提升体验**：发布了一个提醒，建议**注册通知**并在线程中发表评论，以充分利用 Discord 社区。
   - 参与这些讨论可以确保成员消息灵通，并与正在进行的对话保持联系。
- **利用长期线程**：鼓励成员参与**长期线程**，这些线程会不断添加有价值的信息。
   - 这些线程是进行持续讨论和更新的资源，能够增强协作知识。


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1269022115382951957)** (72 条消息🔥🔥): 

> - `Cody vs Cursor`
> - `Aider.nvim 功能`
> - `Claude 的 Sync Folder`
> - `AI 工具中的 Context 管理`
> - `Composer 的内联编辑` 


- **Cody 和 Cursor 的对比**：讨论显示，来自 Sourcegraph 的 Cody 因其 Context 感知能力和易用性而受到称赞，相比之下，一些人认为 Cursor 的 Context 管理比较复杂。
   - *Cody 允许用户对仓库进行索引*，并在 Prompt 中提及它们，以获得更好的 Context 响应。
- **Aider.nvim 的功能**：Aider.nvim 允许用户通过将 Context 放入 Buffer 来添加它，并支持自动抓取 URL 以获取文档，尽管这有时感觉有些粗糙。
   - 用户注意到他们可以*删除 Buffer 来移除 Context*，但在 Cursor 中维持相关的 Context 面临挑战。
- **Claude 的新同步功能**：据报道，Anthropic 正在为 Claude Projects 开发 Sync Folder 功能，允许从本地文件夹批量上传。
   - 这一新功能被视为轻松管理项目内文件的重大增强。
- **Context 管理的挑战**：成员们表达了在 Cursor 等工具中进行 Context 管理及其检索过程的困难，有时会产生无关信息。
   - 一些用户建议在 Composer 中*使用特定命令*有助于更有效地管理 Context。
- **Composer 的功能亮点**：值得注意的是，Composer 的预测能力（例如猜测需要编辑的位置并提供内联编辑功能）是用户喜欢的强项。
   - 社区认为 Composer 可能会改变 AI 辅助编码工作流的游戏规则。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/testingcatalog/status/1816945228869206260">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：Anthropic 正在为 Claude Projects 开发 Sync Folder 功能 👀 在那里你可以选择一个本地文件夹来批量上传文件。</li><li><a href="https://sourcegraph.com/blog/how-cody-provides-remote-repository-context">Cody 如何为各种规模的代码库提供远程仓库感知</a>：Cody 的 Context 感知能力利用 Sourcegraph 平台，可扩展到各种规模的代码库，从最小的初创公司到最大的企业。</li><li><a href="https://sourcegraph.com/blog/how-cody-understands-your-codebase">Cody 如何理解你的代码库</a>：Context 是 AI 编码助手的关键。Cody 使用多种 Context 获取方法来提供与企业级代码库相关的答案和代码。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1269012342159572992)** (3 messages): 

> - `LLM as Judge`
> - `Synthetic Dataset Generation`
> - `WizardLM Papers`
> - `Sparsely-Activated Mixture-of-Experts` 


- **关于 LLM as Judge 和数据集生成的建议**：一位用户询问了有关 **LLM as Judge** 和合成数据集生成（特别是针对指令和偏好数据）当前趋势的必读内容或综述。
   - 另一位成员推荐将 **WizardLM 的最新两篇论文** 作为起点。
- **关于稀疏激活 Mixture-of-Experts 的见解**：[一篇论文](https://arxiv.org/abs/2303.01610) 讨论了巨型 Transformer 面临的挑战，强调了资源消耗过高和参数冗余等问题。
   - 它介绍了 **SMoE-Dropout**，这是一个旨在解决稀疏激活 **Mixture-of-Experts** 模型扩展性问题的新型训练框架，能够提升训练效率。



**提及的链接**：<a href="https://arxiv.org/abs/2303.01610">Sparse MoE as the New Dropout: Scaling Dense and Self-Slimmable Transformers</a>：尽管巨型 Transformer 取得了显著成就，但仍面临重大缺陷，包括训练期间极高的计算和内存占用，以及严重的崩溃迹象...

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 messages): 

not_lain: 兄弟在秀肌肉 
https://x.com/yusufdikec/status/1820186367030128955
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1269046429595472015)** (3 messages): 

> - `VRAM calculation script`
> - `Black Forest Labs launch`
> - `FLUX.1 models` 


- **轻松实现高效 VRAM 计算**：一位用户分享了一个 Ruby 脚本（可在 [此处](https://gist.github.com/jrruethe/8974d2c8b4ece242a071d1a1526aa763) 获取），该脚本可根据 bits per weight 和上下文长度高效计算 LLM 模型的 VRAM 需求。
   - 该脚本允许用户通过一个命令确定 VRAM 需求、基于可用 VRAM 的最大上下文长度，以及可以运行的最高 bits per weight。
- **SOTA 文本生成图像模型亮相**：**Latent Diffusion 团队**宣布成立 _Black Forest Labs_，并推出了用于高级文本生成图像合成的 **FLUX.1** 套件，旨在突破生成式 AI 的边界。
   - 该团队强调其使命是提高生成式 AI 的创造力和效率，旨在为生成式媒体设定行业标准，并使其模型在 [此处](https://blackforestlabs.ai/announcing-black-forest-labs/) 广泛可用。
- **官方 FLUX GitHub 仓库上线**：已为 **FLUX.1** 模型创建了新的 [GitHub 仓库](https://github.com/black-forest-labs/flux)，允许社区为项目开发做出贡献。
   - 该仓库致力于为 FLUX.1 提供官方推理（Inference），增强生成式 AI 领域的协作努力。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://blackforestlabs.ai/announcing-black-forest-labs/">Announcing Black Forest Labs</a>：今天，我们很高兴地宣布 Black Forest Labs 正式成立。我们深植于生成式 AI 研究社区，使命是开发和推进最先进的生成式...</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1ehoqmt/script_calculate_vram_requirements_for_llm_models/">[Script] Calculate VRAM requirements for LLM models</a>：脚本地址：https://gist.github.com/jrruethe/8974d2c8b4ece242a071d1a1526aa763 一段时间以来，我一直试图弄清楚我可以运行哪些量化（quants）...</li><li><a href="https://github.com/black-forest-labs/flux">GitHub - black-forest-labs/flux: Official inference repo for FLUX.1 models</a>：FLUX.1 模型的官方推理仓库。通过在 GitHub 上创建账户来为 black-forest-labs/flux 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1269016900214591538)** (137 messages🔥🔥): 

> - `Claude Sonnet 3.5 Performance`
> - `Training AI Models`
> - `DistillKit Release`
> - `Mistral 7B MoEification`
> - `405B Model Hosting`

- **对 Claude Sonnet 3.5 的担忧**：用户注意到 **Claude Sonnet 3.5** 在特定任务中的表现似乎不如早期版本，指出更新或优化中可能存在问题。
   - 担忧包括幻觉率上升和基础错误，例如在基础代数或逻辑推理方面的失误。
- **碎片化训练导致问题**：一位用户在以极小的学习率对三个独立数据集进行训练后，经历了模型的彻底失败。
   - 这引发了关于**过拟合 (overfitting)**和**灾难性遗忘 (catastrophic forgetting)**的讨论，表明这种方法可能集中了来自各个数据集的错误。
- **DistillKit 的推出**：Arcee AI 宣布发布 **DistillKit**，这是一个开源工具，旨在通过从大型模型中蒸馏知识来创建更小、更强大的模型。
   - 该工具包专注于优化模型的效率和可访问性，将传统的训练技术与新颖的蒸馏方法相结合。
- **创新的 Mistral 7B MoE 化**：一个新模型 **Mistral 7B MoEified** 能够将单层切分为多个专家 (experts)，以实现更连贯的模型行为。
   - 作者解释了这种方法背后的方法论，允许模型在处理过程中平均分配可用的专家资源。
- **托管 405B 模型**：用户讨论了托管 **405B 模型**的潜在服务，**Hyperbolic Labs** 是 **OpenRouter** 上 Llama 3.1 405B 的唯一提供商。
   - 该模型因其相对于其他产品较低的成本而受到关注，表明人们对获取高级 AI 资源的兴趣日益浓厚。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/nisten/status/1818536486662271167">来自 nisten (@nisten) 的推文</a>：@reach_vb @skunkworks_ai 因为没人分享 BitNet 代码而感到生气，所以我直接根据论文手撸了代码。但它无法收敛。于是我不断对 smolL 的层进行自动拼凑（autofrankensteining）...</li><li><a href="https://blog.arcee.ai/announcing-distillkit/?utm_campai">发布用于创建和分发 SLM 的 DistillKit</a>：首先，Arcee AI 通过 Model Merging 和开源仓库 MergeKit 彻底改变了小语言模型 (SLM)。今天，我们通过 ... 为 SLM 的创建和分发带来了又一次飞跃。</li><li><a href="https://x.com/shannonnullcode/status/1819928712185348278?s=46&t=j99rfSSw_U3piCD9F8qGiQ">来自 Shannon Code (@shannonNullCode) 的推文</a>：深度思考。（GenAi 统一物理学？）</li><li><a href="https://x.com/hyperbolic_labs/status/1819509384558661811">来自 Hyperbolic (@hyperbolic_labs) 的推文</a>：Hyperbolic 现在是 OpenRouter 上 Llama 3.1 405B (base) 的唯一提供商，提供的价格远低于其他任何地方。🌪️ 我们迫不及待地想看到研究人员和开发人员 ...</li><li><a href="https://huggingface.co/kalomaze/Mistral-7b-MoEified-8x">kalomaze/Mistral-7b-MoEified-8x · Hugging Face</a>：未找到描述</li><li><a href="https://paperswithcode.com/task/optical-character-recognition">Papers with Code - 光学字符识别 (OCR)</a>：**光学字符识别**或**光学字符阅读器** (OCR) 是将键入、手写或打印文本的图像电子或机械地转换为机器编码文本的过程，无论是从...</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8?inference_api=true">meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/migtissera/Tess-3-Llama-3.1-405B">migtissera/Tess-3-Llama-3.1-405B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/nisten/status/1819745389014024530">来自 nisten (@nisten) 的推文</a>：1 个 CPU 核心 - 每秒 160 个 Token</li><li><a href="https://huggingface.co/mlabonne/Llama-3.1-70B-Instruct-lorablated">mlabonne/Llama-3.1-70B-Instruct-lorablated · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF">bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-exl2">bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-exl2 · Hugging Face</a>：未找到描述</li><li><a href="https://blog.arcee.ai/announcing-distillkit/?utm_campaign=content&utm_source=Marktechpost&utm_medium=Blog&utm_term=Knowledge%20Distillation&utm_content=Blog%201">发布用于创建和分发 SLM 的 DistillKit</a>：首先，Arcee AI 通过 Model Merging 和开源仓库 MergeKit 彻底改变了小语言模型 (SLM)。今天，我们通过 ... 为 SLM 的创建和分发带来了又一次飞跃。</li><li><a href="https://github.com/arcee-ai/DistillKit?ref=blog.arcee.ai">GitHub - arcee-ai/DistillKit (来自 blog.arcee.ai)</a>：一个用于 LLM 蒸馏的开源工具包。通过在 GitHub 上创建账号来为 arcee-ai/DistillKit 的开发做出贡献。
</li>
</ul>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1269770529028440115)** (13 条消息🔥): 

> - `LLMs 在代码优化中的表现`
> - `关于 LLMs 指令遵循的反馈`
> - `微调 Llama 3.1 以进行部署`
> - `训练脚本库`
> - `模型合并问题` 


- **LLMs 擅长优化给定代码**：像 **Claude** 这样的 LLMs 在提供特定指令（如“**优化这段代码**”）时表现良好，能产生有效的结果。
   - 然而，当给出模糊的提示词（如“**让我的代码更好**”）时，它们可能会忽略细微的 bug 或边缘情况。
- **LLMs 难以处理模糊指令**：有人指出，LLMs 可能*难以自行得出结论*，尤其是在缺乏详细指导的情况下。
   - 例如，模糊的询问可能导致在代码优化过程中忽略重要问题。
- **部署微调后的 Llama 3.1 出现问题**：**Admiral_snow** 在将微调后的 Llama 3.1 模型与 bnb 格式合并后，使用 AWQ 量化进行部署时遇到了挑战。
   - 他们怀疑错误源于尝试以 bnb 格式合并模型，而不是使用 fp16/bf16 Hugging Face 权重。
- **模型合并与 GPU 限制**：讨论强调了在不超出 GPU 容量的情况下合并模型的困难，特别是在使用 **H100** 时。
   - 为了提高效率，建议使用**普通 RAM** 进行模型合并作为替代方案。
- **LLMs 微调环境**：有人提出疑问，大多数开发者是利用库还是编写自定义训练脚本来进行模型微调。
   - 提到的一个流行库是 **Axolotl**，这表明了利用成熟工具的趋势。


  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1269087185022222426)** (3 条消息): 

> - `讨论中的私信`
> - `Temperature 设置查询` 


- **私信提议**：一位成员提议通过私信讨论某个话题，表示愿意进一步交流。
   - 这表明在持续的讨论中，可能更倾向于隐私保护或进一步的澄清。
- **Temperature 设置问题**：Aarush 询问私信是否关于 **temperature 设置**，表明需要进行具体的讨论。
   - 这突显了对优化或调整与对话主题相关的特定参数的关注。


  

---

### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1269454084340060161)** (80 条消息🔥🔥): 

> - `Netlify 集成`
> - `Quarto 任务组织`
> - `GitHub Actions 自动化`
> - `Markdown 格式一致性`
> - `网站部署问题` 


- **用于自动化构建的 Netlify 集成**：一位成员正致力于将 [Netlify](https://github.com/apps/netlify) 集成到仓库中，以实现自动化构建并简化部署流程。
   - 另一位成员已准备好协助设置，这需要安装 Netlify 应用并配置仓库。
- **关于 Quarto 文件导致仓库混乱的担忧**：成员们讨论了仓库顶层存在大量 Quarto 文件的问题，这导致了组织结构上的混乱。
   - 一位成员建议在独立分支上部署 Quarto 以减少混乱，而其他人则强调了清晰文档的重要性。
- **使用 GitHub Actions 自动化构建**：一位成员提议使用 GitHub Actions 自动化 npm 构建过程，以减轻贡献者的工作量。
   - 大家对这一方案达成共识，并强调了以往使用 GitHub Actions 的良好体验。
- **Markdown 格式一致性问题**：注意到不同任务中 Markdown 输入格式的不一致，建议统一使用 'Input:' 而非 'Inputs'。
   - 成员们认识到保持正确格式的重要性，以确保解析器的顺畅运行以及通过 Pull Requests 进行的提交。
- **网站部署未反映新任务**：一位成员观察到以 Quarto 格式添加的新任务未在网站上显示，随后意识到这些任务需要在 `_quarto.yml` 文件中列出。
   - 这一疏忽很快被另一位成员发现，并提供了纠正该问题的说明。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/apps/netlify">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、分叉并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://reasoning.nousresearch.com/">Open Reasoning Tasks</a>：未找到描述</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/commit/279f36cc43e4cf6b047cd427929c37427899795a">Create currys-paradox.qmd · NousResearch/Open-Reasoning-Tasks@279f36c</a>：将 currys paradox 任务添加到 Quarto 章节</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/blob/main/tasks/currys-paradox.md">Open-Reasoning-Tasks/tasks/currys-paradox.md at main · NousResearch/Open-Reasoning-Tasks</a>：一个面向 LLM（及更多领域）的综合推理任务仓库 - NousResearch/Open-Reasoning-Tasks</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks">GitHub - NousResearch/Open-Reasoning-Tasks: A comprehensive repository of reasoning tasks for LLMs (and beyond)</a>：一个面向 LLM（及更多领域）的综合推理任务仓库 - NousResearch/Open-Reasoning-Tasks</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/commit/240b0e5032f47265439e0e80a1864ab529ab62a7">Create stack-based-reasoning.qmd · NousResearch/Open-Reasoning-Tasks@240b0e5</a>：为 stack based reasoning 任务添加 qmd 文件
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1269054386982883349)** (7 条消息): 

> - `聊天室改进`
> - `OpenRouter 的新模型`
> - `Mistral 的 Azure 路由`
> - `Gemini Pro 的定价结构`
> - `Yi 端点`

- **Chatroom 品牌重塑与功能更新**：Playground 已更名为 [Chatroom](https://openrouter.ai/chat)，新增了本地聊天保存功能，并简化了 UI 以方便进行房间配置。
   - 用户现在可以在享受更友好的界面的同时，探索增强的功能。
- **令人兴奋的新模型发布**：OpenRouter 推出了新模型，包括 **Llama 3.1 405B BASE** 和免费的 **Llama 3.1 8B**，可以通过其 [模型页面](https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free) 访问。
   - 此外，**Mistral Nemo 12B Celeste** 和 **Llama 3.1 Sonar 系列** 现已上线，可用于各种应用。
- **Mistral 模型路由至 Azure**：**Mistral Large** 和 **Mistral Nemo** 模型现在路由至 [Azure](https://openrouter.ai/models/mistralai/mistral-large)，以提高可用性。
   - 此举为需要这些 AI 模型强大性能的用户增强了可用的基础设施。
- **Gemini Pro 1.5 Experimental 现已上线**：**Gemini Pro 1.5 Experimental** 模型可通过 [此链接](https://openrouter.ai/models/google/gemini-pro-1.5-exp) 访问，要求用户在设置中启用训练。
   - 该模型由 AIStudio 提供服务，与通常的 Vertex 路由不同，用户必须在 [隐私设置](https://openrouter.ai/settings/privacy) 中更新设置才能访问。
- **澄清 Gemini 定价结构**：经社区成员确认，**Gemini 模型** 目前的定价设定为 1 个 token 等于 1 个字符。
   - 计划很快转向基于 token 的定价系统，这取决于数据对账工作的进展。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemini-pro-1.5-exp">Gemini Pro 1.5 (0801) - API, Providers, Stats</a>: Gemini 1.5 Pro (0801) 是 [Gemini 1 的实验版本。通过 API 运行 Gemini Pro 1.5 (0801)</li><li><a href="https://openrouter.ai/models/mistralai/mistral-large>">Mistral Large - API, Providers, Stats</a>: 这是 Mistral AI 的旗舰模型 Mistral Large 2（版本 `mistral-large-2407`）。它是一款权重可用的专有模型，在推理、代码、JSON、对话等方面表现出色。运行 Mistr...</li><li><a href="https://openrouter.ai/models/mistralai/mistral-nemo>">Mistral Nemo - API, Providers, Stats</a>: 由 Mistral 与 NVIDIA 合作构建的 12B 参数模型，具有 128k token 上下文长度。该模型是多语言的，支持英语、法语、德语、西班牙语、意大利语、葡萄牙语、中文...</li><li><a href="https://openrouter.ai/models/01-ai/yi-large>">Yi Large - API, Providers, Stats</a>: Yi Large 模型由 01.AI 设计，针对以下用例：知识搜索、数据分类、类人聊天机器人和客户服务。通过 API 运行 Yi Large</li><li><a href="https://openrouter.ai/models/01-ai/yi-large-turbo>">Yi Large - API, Providers, Stats</a>: Yi Large 模型由 01.AI 设计，针对以下用例：知识搜索、数据分类、类人聊天机器人和客户服务。通过 API 运行 Yi Large</li><li><a href="https://openrouter.ai/models/01-ai/yi-large-fc>">Yi Large - API, Providers, Stats</a>: Yi Large 模型由 01.AI 设计，针对以下用例：知识搜索、数据分类、类人聊天机器人和客户服务。通过 API 运行 Yi Large</li><li><a href="https://openrouter.ai/models/01-ai/yi-vision>">Yi Vision - API, Providers, Stats</a>: Yi Vision 是复杂的视觉任务模型，提供基于多张图像的高性能理解和分析能力。它非常适合需要分析和解释...的场景。</li><li><a href="https://openrouter.ai/settings/preferences">Settings | OpenRouter</a>: 管理您的账户和偏好设置</li><li><a href="https://openrouter.ai/docs/parameters-api">Parameters API | OpenRouter</a>: 用于管理请求参数的 API</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b">Llama 3.1 405B (base) - API, Providers, Stats</a>: Meta 最新的模型系列 (Llama 3.1)，推出了多种尺寸和版本。通过 API 运行 Llama 3.1 405B (base)</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free">Llama 3.1 8B Instruct (free) - API, Providers, Stats</a>: Meta 最新的模型系列 (Llama 3.1)，推出了多种尺寸和版本。通过 API 运行 Llama 3.1 8B Instruct (free)</li><li><a href="https://openrouter.ai/models/nothingiisreal/mn-celeste-12b">Mistral Nemo 12B Celeste - API, Providers, Stats</a>: 基于 Mistral 的 NeMo 12B Instruct 的专业故事写作和角色扮演模型。在包括 Reddit Writing Prompts 和 Opus Instruct 25K 在内的精选数据集上进行了微调。运行 Mistral Nemo 12B...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-large-128k-online">Llama 3.1 Sonar 70B Online - API, Providers, Stats</a>: Llama 3.1 Sonar 是 Perplexity 最新的模型系列。通过 API 运行 Llama 3.1 Sonar 70B Online</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-small-128k-online">Llama 3.1 Sonar 8B Online - API, Providers, Stats</a>: Llama 3.1 Sonar 是 Perplexity 最新的模型系列。通过 API 运行 Llama 3.1 Sonar 8B Online
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1269163491328000041)** (2 条消息): 

> - `多 AI 问答网站发布`
> - `社区支持`
> - `Product Hunt 互动` 


- **多 AI 问答网站发布**：感谢 OpenRouter 的支持，新的 [多 AI 问答网站](https://www.producthunt.com/posts/aiswers-com) 已在 Product Hunt 上线！
   - 团队邀请用户前往体验，并寻求社区的 **点赞和建议**。
- **感谢社区支持**：对 OpenRouter 的持续支持表示感谢，强调了其在发布过程中的重要性。
   - 消息强调，在本次发布期间，**社区反馈**和参与度被高度重视。



**提及的链接**：<a href="https://www.producthunt.com/posts/aiswers-com"> Aiswers.com - AI 版 Quora - 获取全球顶级 AI 的反馈 | Product Hunt</a>：我们汇集了全球顶尖的 AI 智慧来回答您的问题。在一个地方即可从各种 AI 模型和 Agent 获取即时、个性化的回答。开发者还可以集成...

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1269007129243684934)** (150 条消息🔥🔥): 

> - `Model Comparisons` (模型对比)
> - `API Rate Limits` (API 速率限制)
> - `Image Classification Models` (图像分类模型)
> - `Image Quality` (图像质量)
> - `Pricing Strategies` (定价策略)


- **Yi-Vision vs. FireLLaVA 性能对比**：用户在测试 **Yi-Vision** 与 **FireLLaVA** 时报告了不同的性能结果，一些用户表示尽管两者价格相近，但 Yi-Vision 表现更好。
   - 据观察，*Yi-Vision* 犯了一些小错误，而 FireLLaVA 在同样的测试中出现了较大的错误。
- **Google Gemini Flash 定价变更**：官方宣布 **Google Gemini 1.5 Flash** 的价格将在 12 号减半，使其在与 **Yi-Vision** 和 **FireLLaVA** 等模型的竞争中更具优势。
   - 用户对更廉价的选择表示兴奋，这将使为用户生成内容提供详细的自动标注（captioning）成为可能。
- **API 速率限制处理**：当用户超过其 API 速率限制时，会收到 **429 response**，表示请求过多。
   - 讨论确认，在使用 OpenRouter 时，监控活动以避免速率限制问题至关重要。
- **图像 API 调用的 Token 计数**：关于如何计算 API 调用中图像的 Token 限制存在疑问，并澄清了不同服务之间 Token 计数的差异。
   - 有人指出 Google 的 Gemini 将 Token 和字符同等对待，这会影响图像处理的成本估算。
- **成本与 API 调用定价**：用户询问 OpenRouter API 调用是否直接返回成本信息，反馈显示通常可以在调用后通过 generation 端点获取成本。
   - 关于提供按需付费访问的疑虑引出了一个共识，即可以根据详细的请求数据追溯计算 API 调用成本。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/activity">Activity | OpenRouter</a>：查看你在 OpenRouter 上使用模型的情况。</li><li><a href="https://openrouter.ai/docs/responses#querying-cost-and-stats">Responses | OpenRouter</a>：管理来自模型的响应</li><li><a href="https://huggingface.co/docs/transformers/en/tasks/image_classification">Image classification</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2405.15668v1">What Do You See? Enhancing Zero-Shot Image Classification with Multimodal Large Language Models</a>：大型语言模型 (LLMs) 已被有效地用于许多计算机视觉任务，包括图像分类。在本文中，我们提出了一种简单而有效的方法来进行零样本图像分类...</li><li><a href="https://x.com/alexalbert__/status/1820520897465246194?t=TcISXOeVcSjBAIvxhLGcow&s=19">Alex Albert (@alexalbert__) 的推文</a>：我几周前在 AI Engineer 峰会上的完整演讲现在已经上传到 YouTube 了！https://x.com/aiDotEngineer/status/1820484842594930939 引用 AI Engineer (@aiDotEngineer) Claude 3.5 Sonnet 曾是...</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>：设置模型使用限制</li><li><a href="https://openrouter.ai/models/fireworks/firellava-13b">FireLLaVA 13B - API, Providers, Stats</a>：一款极速的视觉语言模型，FireLLaVA 能快速理解文本和图像。它在测试中展现了令人印象深刻的对话技巧，旨在模仿多模态 GPT-4。运行 FireLLaVA 13B...</li><li><a href="https://x.com/OpenRouterAI/status/1819500533553443004">OpenRouter (@OpenRouterAI) 的推文</a>：Llama 3.1 405B BASE！它来了。这是上周发布的对话模型的基座版本。你可以用它来生成训练数据、代码补全等。目前由一家新的提供商托管...</li><li><a href="https://www.producthunt.com/posts/aiswers-com"> Aiswers.com - AI 版 Quora - 获取全球顶尖 AI 的反馈 | Product Hunt</a>：我们汇集了全球顶尖的 AI 智慧来回答您的问题。在一个地方获取来自各种 AI 模型和 Agent 的即时、个性化回答。开发者还可以集成...</li><li><a href="https://github.com/MinorJerry/WebVoyager">GitHub - MinorJerry/WebVoyager: "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models" 的代码</a>：MinorJerry/WebVoyager - "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models" 的代码</li><li><a href="https://github.com/robert-mcdermott/LLM-Image-Classification">GitHub - robert-mcdermott/LLM-Image-Classification: 使用 LLMs 进行图像分类测试</a>：使用 LLMs 进行图像分类测试。通过在 GitHub 上创建一个账号来为 robert-mcdermott/LLM-Image-Classification 做出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1269077490693968015)** (90 条消息🔥🔥): 

> - `Mojo 用于数据处理`
> - `CSV vs Parquet`
> - `数据库查询优化`
> - `许可与开源考量` 


- **Mojo 可能增强数据处理流水线**：讨论强调了 **Mojo** 在将分析与数据库工作负载集成方面的潜力，可能通过 **JIT** 编译和直接文件操作实现更快的数据处理。
   - 成员们提到了它与 **PyArrow** 和 **Ibis** 等工具的兼容性，暗示了 **Mojo** 框架内丰富的数据生态系统前景广阔。
- **关于 CSV 和其他格式的惯例争论**：用户对在已有 **Parquet** 或 **Arrow** 等更高效格式的情况下仍不得不使用低效的 **CSV** 格式表示沮丧，并强调了对性能的担忧。
   - 有人指出，客户需求通常决定了格式选择，有时会导致数据处理中不必要的复杂性。
- **优化数据库查询执行计划**：讨论围绕现代**分析型数据库**及其查询优化能力展开，重点关注 **NUMA-aware** 执行和并行化等技术。
   - 成员们还指出，采用针对静态结构的高级编译策略对于提升处理性能具有重要意义。
- **许可讨论与开源意图**：成员们辩论了软件的适当许可框架，指出像 **AGPLv3** 这样的许可证对于未完全致力于开源的公司来说可能过于严格。
   - 建议采取务实的许可策略，以保持透明度并防止剥削性行为，同时支持开源项目的可见性。
- **FPGA 技术与数据处理的集成**：引入了结合 **FPGA** 技术与 **Apache Arrow** 等格式的想法，以提升数据处理能力。
   - 提到了 **Fletcher** 等工具作为促进这种集成以增强整体数据处理效率的框架示例。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ibis-project.org)">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=9rOefO341sI">The Future Roadmap for the Composable Data Stack</a>: 探索数据处理栈的前沿进展。听 Wes McKinney 深入探讨 **Parquet** 和 **Arrow** 等关键项目，以及必要的...</li><li><a href="https://www.youtube.com/watch?v=YrqSp8m7fmk&pp=ygURY3N2IHBlZHJvIGhvbGFuZGE%3D">Efficient CSV Parsing - On the Complexity of Simple Things - Pedro Holanda</a>: DSDSD - 荷兰数据系统设计研讨会：我们每两周五下午 3:30 到 5:00 (CET) 为研究人员和从业者举办讲座...</li><li><a href="https://github.com/abs-tudelft/fletcher">GitHub - abs-tudelft/fletcher: Fletcher: A framework to integrate FPGA accelerators with Apache Arrow</a>: Fletcher: 一个将 **FPGA** 加速器与 **Apache Arrow** 集成的框架 - abs-tudelft/fletcher</li><li><a href="https://github.com/mzaks/mojo-csv">GitHub - mzaks/mojo-csv</a>: 为 mzaks/mojo-csv 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1269040406536912906)** (48 messages🔥): 

> - `Elixir Error Handling`
> - `Mojo Debugger`
> - `Mojo SIMD Performance`
> - `Thermo Physics Engine`
> - `Variadic Struct Parameters` 


- **Elixir 的非标准错误处理**：成员们讨论了 Elixir 面临的挑战，即库要么返回错误原子（error atoms），要么抛出异常，导致错误处理缺乏标准化。
   - 分享了一段由 Chris Lattner 和 Lex Fridman 参与的 [YouTube 视频](https://www.youtube.com/watch?v=Iflu9zEJipQ)，讨论了异常与错误的对比。
- **Mojo Debugger 的局限性**：一名成员确认 Mojo 调试器目前无法在 VS Code 中工作，并引用了关于该主题的现有 GitHub issue。
   - 一般的调试工作流似乎依赖于通过 print 语句运行程序，而不是使用调试器。
- **Mojo SIMD 的性能担忧**：针对 Mojo 中大型 SIMD 列表的操作性能提出了担忧，这在某些硬件配置上可能会变慢。
   - 另一名成员提到，使用接近 CPU 设计处理能力的 SIMD 大小可以提高性能。
- **物理引擎的拟议名称**：一名成员建议将用 Mojo 编写的新物理引擎命名为 'Thermo'，考虑到这是对 'thermojo' 的双关语。
   - 这引发了关于社区内命名灵活性和创意潜力的讨论。
- **Mojo 中创新的 Variadic Struct Parameters**：一名成员演示了如何使用变长结构体参数（variadic struct parameters）配合参数化的 `__getattr__` 方法来创建一个灵活的类结构。
   - 他们指出，这种设计模式可以在 Mojo 中实现动态类型与非动态类型的融合。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/theyre-the-same-picture-the-office-pam-the-office-us-gif-20757621">Theyre The Same Picture The Office GIF - Theyre The Same Picture The Office Pam - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=Iflu9zEJipQ">Exception vs Errors | Chris Lattner and Lex Fridman</a>：Lex Fridman Podcast 完整剧集：https://www.youtube.com/watch?v=pdJQ8iVTwj8 请通过查看我们的赞助商来支持此播客：- iHerb: https://lexfri...</li><li><a href="https://github.com/modularml/mojo/issues/1829">[Feature Request] Debugging Support for Mojo in VS Code and PyCharm · Issue #1829 · modularml/mojo</a>：查看 Mojo 的优先级。我已阅读路线图和优先级，并相信此请求符合优先级。你的请求是什么？我尝试在 VS Code 中使用断点进行调试...</li><li><a href="https://github.com/MVPavan/mojos/blob/master/learn/dsa/my_queue.mojo">mojos/learn/dsa/my_queue.mojo at master · MVPavan/mojos</a>：Mojo 代码集合。通过在 GitHub 上创建账号为 MVPavan/mojos 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1269899509555789865)** (12 messages🔥): 

> - `Mojo installation issues on MacOS`
> - `Max installation on Ubuntu`
> - `Documentation on MAX Engine comparison`
> - `PyTorch command-line interface` 


- **MacOS 上的 Mojo 安装困扰**：一名成员在 **MacOS 15** 上遇到持续的 **Mojo** 安装错误，即使尝试重新安装后也是如此。
   - 还提出了 *“我可以完全清除 Mojo 和 MAX 的痕迹吗？”* 的问题，暗示可能存在系统冲突。
- **Ubuntu 上 Max 安装程序的反馈**：另一名成员建议为 **Ubuntu** 上的 Max 安装程序设置 `DEBIAN_FRONTEND=noninteractive`，因为交互式的安装后 GUI 会变得无响应。
   - 这一更改可以提升遇到类似问题的用户的安装体验。
- **缺失的 MAX Engine 对比文档**：一名用户正在寻找之前可用的文档页面，该页面在多个模型（包括 **ResNet** 和 **Mistral-7B**）上对比了 **MAX Engine** 与 **PyTorch** 及 **ONYX**。
   - 他们请求协助寻找这个目前已缺失的网页。
- **新的 GitHub 项目：用于 LLM 的 PyTorch CLI**：一个热门的 GitHub 项目 [torchchat](https://github.com/pytorch/torchchat) 为基于 **PyTorch** 的 LLM 提供了命令行界面，类似于 Max pipelines。
   - 它还具有 **Streamlit 界面**，允许用户在服务器、桌面和移动设备上本地运行模型。



**提及的链接**：<a href="https://github.com/pytorch/torchchat">GitHub - pytorch/torchchat: Run PyTorch LLMs locally on servers, desktop and mobile</a>：在服务器、桌面和移动设备上本地运行 PyTorch LLM - pytorch/torchchat

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1269018650837258271)** (23 messages🔥): 

> - `Claude AI code fixes`
> - `Arch design for performance improvement`
> - `SOTA music generation models`
> - `RIAA and labels relationship`
> - `HDF5 for loading embeddings` 


- **Claude AI 可以根据 output.json 提供代码修复**：一位成员提到，通过与 **Claude AI** 开启新对话并上传 `output.json` 映射文件，可以让 **Claude** 在不访问实际文件的情况下编写代码修复。这得到了 [Medium 文章](https://medium.com/@mbonsign/codemapper-your-ais-guide-to-understanding-code-ef2bda7f333e) 的支持，该文章引用了 Claude.ai 对输出文件评估的观点。
   - *然而，对于支持其有效性的实证证据存在怀疑*。
- **新架构提升性能**：讨论强调，创建新架构可以提升性能，特别是在 **特定用户的音频分类 (user-specific audio classification)** 场景下。修改可能涉及使用 **对比学习 (contrastive learning)** 来输出用户不变特征。
   - 此外，还举了一个调整 **3D 数据** 架构以确保性能在平移 (translations) 变换下保持不变的例子。
- **对可控音乐生成模型的关注**：关于当前 **音乐生成 SOTA 模型** 的咨询引出了一个建议，即搜索有关正在进行的“AI 音乐生成诉讼”的信息。
   - 一位成员表示更倾向于可以在 **本地 (locally)** 运行的模型，而不是依赖外部服务。
- **RIAA 在音乐行业中的角色**：讨论围绕 **RIAA** 及其与唱片公司的关系展开，并将其与电影行业的 MPAA 进行了类比。成员们认为 RIAA 和唱片公司共同维护现有的行业结构。
   - 成员们对艺术家仅获得极低比例的版税，而 RIAA 和唱片公司却攫取利润表示担忧，并呼吁进行自我推广和直接向艺术家支付报酬。
- **用于 embedding 管理的 HDF5**：有人提问 **HDF5** 是否仍然是从磁盘上的大型 embeddings 集中加载小型随机批次 (randomized batches) 的首选方法。这表明了人们对高效管理大型数据集的持续关注。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1269006942777507972)** (75 条消息🔥🔥): 

> - `Transformer Parameter Recovery` (Transformer 参数恢复)
> - `Layer Norm Effects on Weights` (Layer Norm 对权重的影响)
> - `NeurIPS 2024 Workshops` (NeurIPS 2024 Workshop)
> - `AI Search vs Training Compute` (AI Search vs 训练算力)
> - `Meta's Distributed AI Training Network` (Meta 的分布式 AI 训练网络)


- **从打乱的权重中恢复 Transformer 参数**：围绕从打乱的向量中恢复原始 Transformer 参数的挑战展开了讨论，强调了对架构和训练方案（training regimes）知识的需求。
   - 成员们讨论了矩阵分布规律的作用，以及训练完美的模型是否应呈现正态分布，并暗示置换（permutation）可能携带重要信息。
- **Layer Norm 在权重排序中的复杂性**：有人对 Layer Norm 在区分权重方面的影响表示担忧；虽然某些 Layer Norm 很容易分配，但下游的其他矩阵可能会根据训练情况表现出不可预测的行为。
   - 有人指出，理解标准差（std deviations）有助于按深度对张量（tensors）进行排序，尽管在 Transformer 块中保持 Norm 和权重的连贯配对仍然具有挑战性。
- **NeurIPS 2024 Workshop 公布**：NeurIPS 2024 Workshop 已公布，从 204 份申请中接收了 56 份，较去年显著增加。
   - 评审过程使用了 OpenReview 以与其他投稿轨道保持一致，尽管社区对缺乏以数据为中心的 Workshop 表示失望。
- **AI Search 优于训练算力的优势**：一份文件强调，搜索机制（search mechanisms）是训练算力的一种强大且具有成本效益的替代方案，并指出这在理论研究上尚不充分。
   - 研究结果表明，利用搜索可以提高 AI 在某些应用中的效率。
- **Meta 的分布式 AI 训练基础设施**：Meta 概述了其在构建大规模分布式 AI 训练网络方面的进展，这对于像 [LLAMA 3.1 405B](https://ai.meta.com/blog/meta-llama-3-1/) 这样的模型至关重要。
   - 在 [ACM SIGCOMM 2024](https://conferences.sigcomm.org/sigcomm/2024/) 上分享的研究详细介绍了全球最大的 AI 网络之一的设计和运行，解决了分布式训练带来的通信需求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2301.13142">Self-Compressing Neural Networks</a>: 这项工作专注于减小神经网络的大小，这是神经网络执行时间、功耗、带宽和内存占用的主要驱动因素。一个关键挑战是在...中减小大小。</li><li><a href="https://yellow-apartment-148.notion.site/AI-Search-The-Bitter-er-Lesson-44c11acd27294f4495c3de778cd09c8d">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>: 一款将日常工作应用融为一体的新工具。为您和您的团队打造的一体化工作空间。</li><li><a href="https://blog.neurips.cc/2024/08/02/announcing-the-neurips-2024-workshops/">宣布 NeurIPS 2024 Workshop – NeurIPS 博客</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2407.19200">代表利益相关者：LLM 时代 NLP 模型可解释性的趋势</a>: NLP 系统的最新进展，特别是 LLM 的引入，导致这些系统被各领域的广泛用户采用，影响了决策...</li><li><a href="https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/">用于大规模分布式 AI 训练的 RoCE 网络</a>: AI 网络在将数万个 GPU 互连方面发挥着重要作用，构成了训练的基础设施，支持具有数千亿参数的大型模型...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1270115079051808778)** (1 messages): 

> - `Recent Developments in SAEs`
> - `Notation in Transformer Circuits`
> - `Model Simplifications in Transformers`
> - `Information Movement in Attention Heads`
> - `Path Expansion Techniques` 


- **对 SAEs 最新进展的兴趣**：一位成员表示希望在间隔一年多后跟进 **SAEs** 的最新进展，并寻求一个良好的学习起点。
   - 他们特别提到了对 [Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html#notation) 中符号表示法（notation）的兴趣，并询问其当前的适用性。
- **模型简化（Model Simplifications）的探索**：讨论了关于 **Model Simplifications** 章节的内容，重点关注简化如何帮助理解复杂的架构。
   - 成员们强调了这些简化在为试图掌握最新进展的新手弥合差距方面的重要性。
- **作为信息移动的 Attention Heads**：参与者讨论了 **Attention Heads** 如何独立且以加法方式运行，从而促进模型内部的信息移动。
   - 这一概念被联系到优化模型性能和理解层间通信的实际应用中。
- **Transformer 架构中的路径展开技术（Path Expansion Techniques）**：成员们讨论了 **Path Expansion Trick**，详细探讨了它如何增强对 logits 和 attention scores 的理解。
   - 该技术被认为是对 Transformer 架构进行深入分析的关键方法。
- **完全理解单层模型（One-Layer Models）**：提出了一个关于我们是否可以声称“完全理解” **One-Layer Models** 的问题，引发了对其影响的深入探讨。
   - 这一讨论为进一步研究以及将简单模型视为未来复杂性基础的解释开辟了道路。



**Link mentioned**: <a href="https://transformer-circuits.pub/2021/framework/index.html#notation">A Mathematical Framework for Transformer Circuits</a>: 未找到描述

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1269105600793546762)** (35 条消息🔥): 

> - `lm-eval-harness 使用`
> - `Cohere API 更新`
> - `仇恨言论检测观点`
> - `自定义架构的模型 Checkpoint 评估`
> - `HellaSwag Prompt 的 Perplexity 计算` 


- **lm-eval-harness 的教学资源**：一位用户询问了如何将 `lm-harness` 命令行转换为 Python 代码的教程，寻求关于使用 `lm-eval-harness` 评估自定义模型架构的指导。另一位成员建议查看 [lm_evaluation_harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness) 获取示例。
   - 分享了一个相关示例，展示了如何重写模型方法以兼容自定义模型，并强调了在验证集和测试集中保持相似分布的重要性。
- **Cohere API 向新端点的过渡**：讨论涉及 Cohere API 从 `.generate` 转向 `.chat` 端点的变化，特别是参考了 [迁移指南](https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat)。目前看来，像 Amazon Sagemaker 这样的大型平台用户可能不需要迁移，但对于 MCQ（多选题）评估缺乏概率支持的问题仍存在困惑。
   - 一位成员强调了这一转变的影响，并对尽管模型已开源但仍删除了 Likelihood 功能表示困惑，这引发了对更深层次评估方法的讨论。
- **辩论仇恨言论检测的有效性**：一位成员对仇恨言论检测方法论表示怀疑，称其为“胡言乱语（mumbo jumbo）”。讨论随后演变为对当前安全标准模糊性的看法，以及 AI 社区中影响可感知进展的重复性危机（reproducibility crisis）。
   - 成员们注意到，许多表达出的观点出奇地一致，但在侧重点上有所不同，这引发了关于如何在更广泛的背景下更好地传达观点的疑问。
- **使用 eval harness 获取数据集的 Perplexity**：出现了关于专门为 HellaSwag Prompt 计算 Perplexity 的查询，建议修改 YAML 文件中的数据集路径。一位成员指出，新的配置支持使用 `datasets.load_dataset` 方法加载数据集，从而实现更流线化的 Perplexity 计算。
   - 另一位用户确认过去的任务可以从本地文件计算 Perplexity，并建议采用协作方式来调整 wikitext 任务以满足当前需求。
- **在评估中使用 HF PretrainedModel**：一位成员询问了如何评估自定义模型架构，并被引导至一个适配 Huggingface LM 类方法的示例。值得注意的是，用户可以将已经初始化的 HF `PretrainedModel` 传递给 `HFLM` 类，通过自定义脚本进行量身定制的评估。
   - 这种灵活性为用户在评估模型之前进行量化（Quantization）或剪枝（Pruning）等操作打开了大门，增强了模型评估方法的通用性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat">从 Generate API 迁移到 Chat API - Cohere 文档</a>: 该文档概述了从 Generate 端点迁移到 Chat 端点的 Cohere 生成功能，建议用户使用 Chat 端点以提高模型输出质量和...</li><li><a href="https://github.com/state-spaces/mamba/blob/main/evals/lm_harness_eval.py">mamba/evals/lm_harness_eval.py at main · state-spaces/mamba</a>: Mamba SSM 架构。通过在 GitHub 上创建账号为 state-spaces/mamba 的开发做出贡献。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2129">由 KonradSzafer 添加多聊天模板 · Pull Request #2129 · EleutherAI/lm-evaluation-harness</a>: 此 PR 增加了对具有多个聊天模板的模型（如 Cohere Command R+）的支持，解决了 issue #1962。命令行 API 已更新，以重用现有标志来指定模板...
</li>
</ul>

</div>

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1269060922459029505)** (100 条消息🔥🔥): 

> - `Ollama 中的错误处理`
> - `同时使用 CPU 和 GPU`
> - `模型规格与资源需求`
> - `LangChain 内存管理`
> - `AWS Lambda 中的多用户功能` 


- **Ollama 内存错误困扰**：一位用户报告在调用 retrieval chain 时遇到了 **ValueError**，提示模型内存不足，尽管其 GPU 显存占用率很低。
   - 他们正在使用 **aya** (4GB) 和 **nomic-embed-text** (272MB) 等模型，这使得他们对内存错误感到困惑。
- **混合使用 CPU 和 GPU 进行推理**：讨论围绕 **Ollama** 在重负载情况下能否有效利用 **CPU** 和 **GPU** 资源进行推理展开。
   - 有人指出，**Ollama** 的默认行为应该是在 GPU 显存不足时允许回退到 CPU，但一位用户报告称实际情况并非如预期那样发生。
- **模型使用建议**：建议使用符合可用 GPU 显存限制的**性能要求较低的模型**。
   - 使用需要较少 RAM 的模型对于有效的资源管理和避免内存溢出（out-of-memory）错误至关重要。
- **LangChain 内存管理讨论**：一位用户分享了关于 LangChain 如何处理跨会话的内存和对象持久化的见解，表示希望评估输入的内存效率。
   - 针对确定某些文本是否包含适合存入内存的信息的特定查询，使用不同的模型响应进行了测试。
- **AWS Lambda 多用户功能**：关于如何管理托管在 **AWS Lambda** 上的 RAG (Retrieval-Augmented Generation) Slack 机器人与多个用户交互的问题被提出。
   - 确认了 AWS Lambda 可以扩展以同时处理多个调用，但也承认了成本方面的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.smith.langchain.com/how_to_guides/monitoring/online_evaluations#mapping-variables>).">设置在线评估 | 🦜️🛠️ LangSmith</a>：在深入研究此内容之前，阅读以下内容可能会有所帮助：</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/#tying-it-together>).">如何添加聊天历史 | 🦜️🔗 LangChain</a>：在许多问答应用中，我们希望允许用户进行来回对话，这意味着应用需要对过去的问题和答案有某种形式的“记忆”，以及一些逻辑...</li><li><a href="https://github.com/ollama/ollama/issues/3509">Ollama 可以同时使用 CPU 和 GPU 进行推理吗？ · Issue #3509 · ollama/ollama</a>：你打算做什么？我想知道 Ollama 是否支持在 Windows 上混合使用 CPU 和 GPU 运行？我知道我的硬件不足以运行 Ollama，但我仍然想利用部分能力...</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/chat/llamacpp/#tool-calling">ChatLlamaCpp | 🦜️🔗 LangChain</a>：本笔记本提供了集成 llama cpp python 的聊天模型入门快速概览。</li><li><a href="https://www.youtube.com/watch?v=rsDlu-9UP00">使用 llamacpp 进行本地工具调用</a>：工具调用允许 LLM 连接外部工具，显著增强其功能并支持像 Agent 这样的流行架构。但是，工具...
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1269642644469059654)** (6 条消息): 

> - `Linear Algebra Concepts` (线性代数概念)
> - `CPU-Compatible SAM 2 Fork` (CPU 兼容的 SAM 2 分支)
> - `Research Assistant Beta Testing` (研究助手 Beta 测试)
> - `Importance of Knowledge Graphs` (知识图谱的重要性)
> - `Self-Supervised Learning in Dense Prediction` (密集预测中的自监督学习)


- **新文章探讨线性代数**：一位成员在 [Medium](https://medium.com/@amitsubhashchejara/linear-algebra-part-2-linear-combination-and-span-d5fe65ef0e8f) 上分享了一篇新文章，涵盖了 **线性组合 (linear combinations)** 和 **向量张成 (span of vectors)** 的概念。
   - *敬请关注更多关于线性代数的文章！*
- **维护 CPU 兼容的 SAM 2 分支**：一位成员开始维护 **SAM 2 模型** 的 CPU 兼容分支 (fork)，通过 notebook 展示了提示分割 (prompted segmentation) 和自动掩码生成 (automatic mask generation)。目前正计划开展 **GPU 兼容性** 和创建 **API endpoints** 的工作。
   - 欢迎在 [GitHub repository](https://github.com/SauravMaheshkar/samv2) 上提供社区反馈。
- **招募研究助手 Beta 测试人员**：一位成员正在构建一个高级研究助手和搜索引擎，计划在几周内寻找 Beta 测试人员，并提供 **两个月免费的高级版服务**，支持包括 **GPT-4O** 和 **Mistral Large** 在内的多种模型。
   - 感兴趣的用户可以在 [Rubik's AI](https://rubiks.ai/) 了解更多详情并注册。
- **关于知识图谱与 LLM 的博客文章**：一位成员撰写了一篇博客，讨论了在 LLM 中使用 **实体解析知识图谱 (entity resolved knowledge graphs)** 的重要性。他们分享了 [LinkedIn 帖子](https://www.linkedin.com/posts/dr-clair-sullivan-09914342_generativeai-entityresolution-artificialintelligence-activity-7225242113150529536-VNiq?utm_source=share&utm_medium=member_desktop) 的链接。
   - *希望有所帮助！*
- **基于 ScreenPipe 的 AI 洞察工具**：一位成员展示了一个 AI 工具，它可以 24/7 全天候监控屏幕和麦克风，以提供时间使用情况的见解，并演示了使用 **screenpipe** 和 AI Agent 实现的功能。
   - 该开源项目可以在 [GitHub](https://github.com/louis030195/screen-pipe) 上找到。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.lightly.ai/post/using-self-supervised-learning-for-dense-prediction-tasks">Using Self-Supervised Learning for Dense Prediction Tasks</a>：概述了用于目标检测、实例分割和语义分割等密集预测任务的自监督学习 (Self-Supervised Learning) 方法。</li><li><a href="https://github.com/SauravMaheshkar/samv2">GitHub - SauravMaheshkar/samv2: CPU compatible fork of the official SAMv2 implementation aimed at more accessible and documented tutorials</a>：官方 SAMv2 实现的 CPU 兼容分支，旨在提供更易获取且文档齐全的教程。</li><li><a href="https://github.com/louis030195/screen-pipe">GitHub - louis030195/screen-pipe: Library to build personalized AI powered by what you&#39;ve seen, said, or heard. Works with Ollama. Alternative to Rewind.ai. Open. Secure. You own your data. Rust.</a>：用于构建个性化 AI 的库，基于你所见、所说或所听的内容。支持 Ollama。Rewind.ai 的替代方案。开源、安全、数据归你所有。使用 Rust 编写。</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>：暂无描述。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1269762249346781337)** (1 条消息): 

> - `Custom AI Voice Assistant` (自定义 AI 语音助手)
> - `Sista AI` 


- **8 分钟构建你自己的 AI 语音助手！**：一段名为 ["Create a custom AI Voice Assistant in 8 minutes! - Powered by ChatGPT-4o (By Sista AI)"](https://www.youtube.com/watch?v=iGX4ARuWZec) 的教程视频演示了为你的网站创建自定义 AI 语音助手的逐步过程。
   - 创作者鼓励观众通过 [演示链接](https://smart.sista.ai) 进行尝试并 [免费注册](https://admin.sista.ai/register)。
- **探索用于提升参与度的 AI 工具**：除了语音助手，利用各种 AI 工具增强网站以提升客户参与度和个性化已成为一种日益增长的趋势。
   - 许多开发者正在集成 AI 解决方案以改善用户体验和满意度。



**提到的链接**：<a href="https://www.youtube.com/watch?v=iGX4ARuWZec">Create a custom AI Voice Assistant in 8 minutes! - Powered by ChatGPT-4o (By Sista AI)</a>：想尝试一下吗？🔗 演示：https://smart.sista.ai 🔗 免费注册：https://admin.sista.ai/register。在这个视频中，我将向你展示如何创建你自己的 AI 语音助手...

  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1269062549236748289)** (7 messages): 

> - `ReAct Agents`
> - `Agentic Terraform Assistant`
> - `High-Quality RAG Extraction`
> - `Deploying RAG Applications`
> - `Composio Toolset for AI Agents` 


- **使用 LlamaIndex workflows 构建 ReAct agents**：你可以利用 LlamaIndex workflows 从头开始创建 ReAct agents，以增强内部逻辑的可见性。点击[这里](https://t.co/F0pPEyWJ2w)查看详细指南。
   - “拆解”逻辑的能力确保了对 Agentic 系统的深入理解和控制。
- **使用 LlamaIndex 创建 Terraform 助手**：为有志于成为 AI 工程师的人员开发一个使用 LlamaIndex 和 Qdrant Engine 的 Terraform 助手。本教程涵盖了为自动化生成定义 LLM workflow 的内容，详见[这里](https://t.co/ASWNkixboK)。
   - 凭借实用的见解，它为 AI 与 DevOps 的集成提供了一个宝贵的框架。
- **使用 LlamaExtract 自动提取工资单**：LlamaExtract 通过自动化的模式（schema）定义和元数据提取，实现了对工资单的高质量 RAG。点击[这里](https://t.co/qoC9RU6Tfm)了解更多关于此过程的信息。
   - 该方法极大地提高了工资单文档的数据处理能力。
- **部署和扩展 RAG 应用**：Benito Martin 撰写的一份综合教程概述了如何在 Google Kubernetes 上部署和扩展你的聊天应用。该资源强调了实际的部署策略，详见[这里](https://t.co/ROsGNjhKEM)。
   - 它详细解决了 RAG 应用生产化内容稀缺的问题。
- **Composio 为 AI agents 提供工具**：Composio 拥有一套面向 AI agents 的工具集，包含 GitHub 和 Slack 等 100 多个集成。他们关于构建 PR 审查 Agent 的即将发布的教程可以在[这里](https://t.co/FBdE7bbqFC)找到。
   - 使用这些工具来简化你的开发和协作流程。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1269087328547115058)** (89 messages🔥🔥): 

> - `RAG Applications`
> - `OpenAIAgent vs ContextChatEngine`
> - `LlamaIndex Workflows`
> - `Incremental Re-indexing`
> - `Information Extraction Techniques` 


- **探索 RAG 应用查询**：一位用户询问了关于将 prompt 转换为更有效的查询，以便在 RAG 应用中搜索向量数据库（特别是针对推理查询）的框架。
   - 建议在检索前使用 LLM 重写用户查询，旨在增强语义搜索结果，并提供了一个代码示例。
- **OpenAIAgent 与 ContextChatEngine 性能对比**：一位用户比较了 OpenAIAgent 和 ContextChatEngine 的性能指标，注意到在设置相同的情况下，后者的通过率更高。
   - 社区讨论了差异的原因，推测较简单的问题可能由于其直接检索方法而更有利于 ContextChatEngine。
- **在 Workflows 中使用并行事件**：一位开发者询问如何在 LlamaIndex workflows 中触发并行子事件并管理其结果。
   - 建议利用异步函数调用，并分享了一个示例 workflow，强调了在 workflows 中进行适当事件处理的必要性。
- **LlamaIndex 中的增量重新索引 (Incremental Re-indexing)**：一位 LlamaIndex 新手询问是否支持增量重新索引。
   - 社区确认可以将新文档插入现有索引，而无需重新索引所有内容，并提供了示例代码。
- **阿拉伯语 PDF 解析挑战**：一位用户在解析阿拉伯语 PDF 时遇到问题，导致输出文本乱码。
   - 回复指出可能是 PDF 文件格式质量问题，表明问题可能源于文档本身。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_extract">GitHub - run-llama/llama_extract</a>: 通过在 GitHub 上创建账号来为 run-llama/llama_extract 的开发做出贡献。</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents/#advanced-metadata-customization">Using Documents - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/workflow/?h=workflow">Workflows - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1269546570022522883)** (2 messages): 

> - `GraphRAG`
> - `LlamaIndex`
> - `Knowledge Graphs`
> - `Question Answering` 


- **基于 LlamaIndex 的 GraphRAG**: 一场讨论强调了 **GraphRAG** 与 **LlamaIndex** 的集成，以增强**智能问答**（**intelligent question answering**）能力，详见这篇 [Medium 文章](https://medium.com/ai-advances/graphrag-with-llamaindex-unleashing-the-power-of-knowledge-graphs-for-intelligent-question-ea177a14623e)。
   - 这种方法利用**知识图谱**（**knowledge graphs**）来提高 AI 应用中回答的上下文关联性和准确性。
- **GraphRAG 成为唯一选择**: 一位成员表示，在他们目前的讨论中，将 **GraphRAG** 与 **LlamaIndex** 结合使用似乎是未来推进工作的**唯一选择**。
   - 这种观点强调了人们越来越依赖创新集成来解决复杂的 AI 挑战。


  

---



### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1270058688643858645)** (1 messages): 

> - `Bay Area Events` 


- **用户表示缺席即将举行的活动**: 一位成员提到，他们遗憾地无法参加即将举行的活动。
   - 他们表示有兴趣了解未来在 **Bay Area**（湾区）举办的活动。
- **请求更新 Bay Area 活动信息**: 另一位成员请求在未来有任何专门在 **Bay Area** 举办的活动时能得到通知。
   - 这突显了社区成员对当地线下聚会的持续关注。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1269944391934541845)** (12 messages🔥): 

> - `Noam Shazeer Wikipedia page`
> - `30 Under 30 Awards`
> - `Insider Circles in Tech` 


- **Noam Shazeer 缺少维基百科页面**: 成员们讨论了 **Noam Shazeer** 缺少维基百科页面的情况，他自 2002 年以来一直在 **Google** 担任要职。
   - 一位成员指出，“维基百科有时很荒谬”，强调了知名人物缺乏认可的讽刺现象。
- **对 30 Under 30 奖项的批评**: 一位成员对 **30 Under 30** 奖项表示蔑视，认为那些寻求此类外部认可的人是“特殊类型的人”。
   - 对话反映了一种普遍观点，即这些荣誉往往迎合**内部圈子**（**insider circle**），而非真正的功绩。
- **对维基百科和认可度的看法**: 另一位成员提出，为被忽视的人物创建维基百科页面其实很容易，类似于 **30 Under 30** 提名的运作机制。
   - 这引发了关于科技行业“内部性质”以及特定个人如何游走于这些圈子的讨论。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1269155709808017438)** (2 messages): 

> - `SFT datasets`
> - `System prompts in AI`
> - `AFM paper discussions` 


- **SFT 数据中需要多样化的 System Prompts**: 一位成员评论说，大多数开源 **SFT** 数据集通常只是（prompt, response）对，并质疑为什么没有包含多样化的 **system prompts**。
   - 其核心观点是，模型应该学习对用户 prompt 和 **system prompts** 做出不同的反应，且 **system prompts** 可能具有更高的优先级。
- **System Prompts 在开源社区被忽视**: 另一位成员确认，**system prompts** 在开源社区中仍被很大程度上忽视，并提出了其重要性。
   - 这突显了社区内关于数据集设计潜在缺陷的更广泛讨论。


  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1270053464277258262)** (19 messages🔥): 

> - `Nvidia AI scraping` (Nvidia AI 抓取)
> - `Elon Musk vs OpenAI lawsuit` (Elon Musk 对 OpenAI 的诉讼)
> - `404 Media's journalism` (404 Media 的新闻报道)
> - `Data drama discussions` (数据争议讨论)


- **Nvidia 抓取争议被揭露**：根据 [404 Media](https://www.404media.co/nvidia-ai-scraping-foundational-model-cosmos-project/) 的报道，Nvidia 从 YouTube 等来源抓取视频，为其 AI 产品收集训练数据。
   - Nvidia 声称此举**完全符合**版权法，尽管内部讨论中提出了伦理担忧。
- **Elon Musk 重启与 OpenAI 的法律斗争**：[Reuters](https://x.com/Reuters/status/1820442168357495259) 报道称，Elon Musk 重启了对 Sam Altman 和 OpenAI 的诉讼，这次更侧重于个人恩怨。
   - Musk 寻求宣布 OpenAI 对 Microsoft 的授权无效，并质疑 OpenAI 的模型是否构成了 **AGI**，这可能导致一场戏剧性的法律对决。
- **404 Media 引发数据争议讨论**：关于 **404 Media** 的讨论强调了其详尽的新闻报道，通过提供经过验证的来源和有力证据来对抗点击诱饵（clickbait）趋势。
   - 成员们认可其文章的重要性，但也指出其订阅模式可能会限制更广泛的传播。
- **Musk 指控资助 OpenAI 期间存在不当行为**：在新的诉讼中，Musk 声称他在 2016 年至 2020 年间向 OpenAI 汇款了 **4450 万美元**，指控该期间存在不当行为。
   - 他正在寻求**惩罚性赔偿**以惩罚 OpenAI，这是更广泛的“背叛”叙事的一部分。
- **关于数据新闻影响力的辩论**：一位成员对数据相关故事的反复出现表示沮丧，认为随着时间的推移，这些内容显得乏味。
   - 另一位参与者反驳道，强调了报道案例中证据的重要性，强化了同样的讨论可以产生有价值见解的观点。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.404media.co/nvidia-ai-scraping-foundational-model-cosmos-project/">Leaked Documents Show Nvidia Scraping ‘A Human Lifetime’ of Videos Per Day to Train AI</a>：404 Media 获得的内部邮件、Slack 对话和文件显示了 Nvidia 如何创建了一个尚未发布的视频基础模型（foundational model）。</li><li><a href="https://x.com/AndrewCurran_/status/1820491681831219594">Andrew Curran (@AndrewCurran_) 的推文</a>：Musk 先生再次要求宣布 OpenAI 对 Microsoft 的授权无效，并再次要求司法判定 OpenAI 目前的模型是否构成 AGI。我们可能会得到...</li><li><a href="https://x.com/AndrewCurran_/status/1820491691973026164">Andrew Curran (@AndrewCurran_) 的推文</a>：在 2016 年 5 月 27 日至 2020 年 9 月 14 日期间，Musk 先生向 OpenAI 汇款了 44,563,500 美元。诉状多次以非常强硬的措辞声称全程存在不当行为。他要求授予“惩罚性赔偿”...</li><li><a href="https://x.com/AndrewCurran_/status/1820491625854083432">Andrew Curran (@AndrewCurran_) 的推文</a>：新的诉讼读起来比上一次更像是一个个人背叛的故事，而且更有杀伤力。Altman 先生在文中被多次点名，似乎是这次的直接焦点。引用...</li><li><a href="https://x.com/Reuters/status/1820442168357495259">Reuters (@Reuters) 的推文</a>：文件显示，Elon Musk 重启了对 Sam Altman 和 OpenAI 的诉讼 http://reut.rs/4drHJor
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1269026950408700056)** (33 messages🔥): 

> - `Llama 更新的 100k prompts`
> - `关于 synthetic data 和 nemotron 的讨论`
> - `Ross Taylor 访谈见解`
> - `OpenAI 的市场地位`
> - `Gradio fork 和 LLM 演示` 


- **在 Llama 上运行 100k prompts**：一位成员幽默地确认，他们今天将为 **Llama** 更新运行 **100k prompts**，主要集中在 **更新** 旧的 GPT-4 completions。
   - 他们还提到在此过程中生成了 **preference data**。
- **使用 Nemotron 的 Synthetic Data 策略**：关于是否使用 **Nemotron** 重新制作所有 **synthetic data** 以进行 **Olmo** 模型微调的争论正在进行中。
   - 一位成员对名称被 **hijacked**（劫持）表示担忧，而另一位成员则指出了 AI2 目前方向的问题。
- **Ross Taylor 访谈引发讨论**：即将进行的 **Ross Taylor** 访谈引起了关注，重点讨论了他对 Deep Learning 潜力的看法。
   - 对话涉及 **AGI** 和 **ASI** 等话题，强调了 2024 年的未来目标。
- **OpenAI 的竞争优势**：有人担心，如果 **OpenAI** 不尽快发布 **GPT-5**，可能会落后于 **Sonnet** 和 **Llama 405B** 等竞争对手。
   - 尽管如此，一些成员认为 OpenAI 由于在 **普通用户** 中的品牌知名度而拥有显著优势。
- **AI2 用于 LLM 演示的 Gradio Fork**：AI2 宣布开源了他们的 **Gradio fork**，以便通过并排聊天演示更好地使用 **vllm**。
   - 该项目为快速 LLM 演示提供了轻量级工具，可在 [GitHub](https://github.com/allenai/adapt-demos) 上获取。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/rosstaylor90/status/1788243031570911239)">Ross Taylor (@rosstaylor90) 的推文</a>：如果你认真对待深度学习，你会构建什么？十年前，答案是 AGI：这个词在研究圈子里会让你看起来很古怪。现在，每个 ChatGPT 之后的初创公司都将 AGI 作为他们的……</li><li><a href="https://x.com/MrAhmadAwais/status/1819819517650117105">Ahmad Awais (@MrAhmadAwais) 的推文</a>：如果 OpenAI 不在一个月左右发布 GPT5，他们就会落后。开发者正积极用 Sonnet 和 Llama 405B 替换 GPT。在 Langbase，我们看到超过 34% 的管道已经迁移……</li><li><a href="https://github.com/allenai/adapt-demos">GitHub - allenai/adapt-demos: 用于快速简便 LLM 演示的轻量级工具</a>：用于快速简便 LLM 演示的轻量级工具。通过创建 GitHub 账户为 allenai/adapt-demos 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1269080310176415838)** (4 messages): 

> - `KTO 对齐`
> - `Neural Notes 访谈`
> - `DPO vs KTO 性能`
> - `UCLA 对 KTO 的适配` 


- **KTO 在 Neural Notes 中获得关注**：最近的 [Neural Notes 访谈](https://youtu.be/dDQAYmObK-Y?si=5UMK1bT-6CyKqgnf) 中，Kawin Ethayarajh 讨论了 KTO 模型及其对 AI 决策的影响。
   - 一些研究结果表明，在处理噪声数据时，**KTO** 的表现可以优于 **DPO**，突显了其鲁棒性。
- **KTO 宣称显著的性能提升**：一条评论指出，即使在数据量增加的情况下，使用成对的 DPO 数据进行训练，KTO 的性能也可以达到或超过 DPO。
   - 此外，Orca-Math 论文的研究结果表明，KTO 对齐的模型更优，比 DPO 实现了 **20 个点** 的性能提升。
- **UCLA 将 KTO 适配到 Diffusion Models**：来自 **UCLA** 的团队将 KTO 适配到 **diffusion models**，在与 DPO 对齐模型的对比中，获得了 **70-30%** 的人类偏好胜率。
   - 这种适配强调了 KTO 在实际应用中的实用有效性。
- **KTO 有效处理噪声数据**：据称 KTO 的设计使其能够有效管理噪声数据集，避免在训练过程中拟合噪声，这与 DPO 不同。
   - “当数据集包含足够的噪声或非传递性（intransitivity）时，**KTO 可以胜出**”，突显了其竞争优势。



**提到的链接**：<a href="https://youtu.be/dDQAYmObK-Y?si=5UMK1bT-6CyKqgnf">Neural Notes: KTO - 帮助 AI 像人类一样做出决策</a>：在本期 Neural Notes 中，斯坦福 AI 实验室 (SAIL) 的 Kawin Ethayarajh 与 Vertex Ventures US 的 Sandeep Bhadra 和 Simon Tiu 对话，解释了他的研究……

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1269048817978839142)** (62 条消息🔥🔥): 

> - `Synthetic datasets debate` (合成数据集辩论)
> - `FLUX model performance` (FLUX 模型性能)
> - `Curating synthetic images` (策划合成图像)
> - `Stable Diffusion dataset queries` (Stable Diffusion 数据集查询)
> - `Training model concerns` (训练模型担忧)


- **关于合成数据集价值的讨论**：成员们辩论了**合成数据集 (synthetic datasets)** 与原始数据集相比的有效性，指出虽然它们可以加速训练，但存在对齐不良或质量较低的风险。
   - 有人对潜在的偏见以及在缺乏妥善策划的情况下产生数十亿无用图像的风险表示担忧，呼吁进行更有目的性的数据集创建。
- **FLUX 模型在艺术方面的表现**：用户对 **FLUX** 模型生成艺术输出的能力持褒贬不一的看法，指出虽然有些人成功创作了绘画作品，但其他人对其结果感到失望。
   - 讨论强调，使用正确的参数可以改善输出，但对其在艺术风格方面的有效性仍普遍持怀疑态度。
- **策划合成图像生成**：有建议提出使用**用户策划的图像生成界面**来提高合成数据集的质量，认为人工筛选可以增强整体可用性。
   - 强调了仔细策划的必要性，以避免产生充满对齐不良样本的数据集，从而对新模型的训练产生负面影响。
- **Stable Diffusion 数据集查询**：一位用户询问了 **Stable Diffusion 数据集**的可获取性，声称请求的来源已无法访问，这限制了他们的进展。
   - 其他人参与了讨论，澄清该数据集对于运行 Stable Diffusion 并非严格必要，并建议了替代方法。
- **关于模型训练实践的担忧**：围绕在受版权保护的图像上进行训练的伦理影响展开了持续辩论，成员们对合成数据集项目中潜在的**版权洗白 (copyright laundering)** 表示担忧。
   - 一些人建议，虽然合成数据有其优点，但更广泛的社区审查可能会导致对训练实践实施更严格的监管。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/StableDiffusion/comments/1ej2txw/flux1_is_actually_quite_good_for_paintings/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://youtu.be/LLmg5IzGF-k">BUD-E V1.0 UPDATE: ALL OPEN SOURCE MODELS &amp; LATENCY ~ 2.8 SEC</a>: WIP: https://github.com/LAION-AI/BUD-E_V1.0</li><li><a href="https://youtu.be/iVRt65aIQ3k">BUD-E Conversation demo &amp; hints how to get it running on your own</a>: https://github.com/LAION-AI/BUD-E_V1.0/</li><li><a href="https://www.nature.com/articles/s41586-024-07566-y">AI models collapse when trained on recursively generated data - Nature</a>: &amp;nbsp;分析显示，不加区分地在真实和生成的内容上训练生成式人工智能（通常通过从互联网抓取数据完成），可能导致崩溃...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1269543355306475591)** (4 条消息): 

> - `CIFAR-10 performance` (CIFAR-10 性能)
> - `Complex parameters challenge` (复数参数挑战)
> - `Dropout implementation issues` (Dropout 实现问题)
> - `Overfitting resolution` (过拟合解决) 


- **在 CIFAR-10 上实现了 80% 的验证准确率**：在 CIFAR-10 数据集上达到了 **80% 的验证准确率**，仅使用了 **36k 参数**，其中复数参数的实部和虚部被视为独立组件。
   - *
- **微调提升性能**：仅通过一些架构调整和更好的 dropout 实现就显著提升了性能。
   - 最初出现问题是因为 **nn.dropout** 不适用于复数张量 (complex tensors)，导致在创建替代方案时出现了初步错误。
- **过拟合几乎消除**：事实证明，在最近的更改之后，**过拟合 (overfitting)** 基本上完全消失了。
   - 这些改进带来了更稳健的模型性能。


  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1269148618586198148)** (5 messages): 

> - `Coding Agents`
> - `Golden-Retriever Paper`
> - `Livecoding Events` 


- **在 ChatmanGPT Stack 中添加 Coding Agent**：一名成员正在寻求为 ChatmanGPT Stack 添加 **Coding Agent 的建议**。
   - 另一名成员建议将 **Agent Zero** 作为一个潜在选择。
- **Voice Lounge 中的 Livecoding**：一名成员宣布回归，并提到之前的设置已经 **game over**，并提到了在 Voice Lounge 进行 **livecoding**。
   - 这预示着成员之间可能会进行协作编程会议。
- **Golden-Retriever 论文概述**：一名成员分享了关于 **Golden-Retriever** 论文的链接，该论文旨在高效地导航工业知识库，解决传统 LLM 微调面临的挑战。
   - 该论文概述了一个**基于反射的问题增强（reflection-based question augmentation）**步骤，在文档检索之前澄清术语和上下文，从而显著提高检索准确性。



**提到的链接**：<a href="https://arxiv.org/abs/2408.00798">Golden-Retriever: High-Fidelity Agentic Retrieval Augmented Generation for Industrial Knowledge Base</a>：本文介绍了 Golden-Retriever，旨在高效导航庞大的工业知识库，克服传统 LLM 微调和 RAG 框架在特定领域遇到的挑战。

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1269041866335064144)** (42 messages🔥): 

> - `AI NPC Development`
> - `Discord Chat Exporter Tool`
> - `SAP AI Security Issues`
> - `DSPy Optimization Questions`
> - `DSPy in Production` 


- **AI NPC 巡逻区域并响应玩家**：一名成员讨论了在 C++ 游戏中开发 **AI 角色**的计划，这些角色可以巡逻并使用 **Oobabooga API** 获取响应与玩家互动。
   - 他们打算修改 **'world' 节点**或扩展 NPC 类，并详细说明了运行所需的 account、database 和 core 等程序。
- **轻松导出 Discord 聊天记录**：一名用户使用 **DiscordChatExporter 工具**成功将 Discord 频道导出为 HTML 和 JSON，该工具为进一步使用提供了格式化的结果。
   - 通过包含 threads 的命令，他们注意到在所有频道中生成了 **463 个 thread 文件**。
- **SAP 的 AI 安全漏洞曝光**：一篇分享的文章强调了 **SAP 的 AI Core** 如何由于糟糕的 Kubernetes 配置而产生漏洞，从而允许访问其他客户的数据。
   - 研究人员能够运行任意代码，**突破容器限制**，强调了加强安全实践的必要性。
- **关于 DSPy 优化能力的疑问**：新成员寻求关于 **优化 signatures** 以及 DSPy 指标是否可以返回数值（而非仅布尔值）的指导。
   - 成员们对**丢失上下文**以及编译是否可以恢复表示担忧，这表明该平台存在一定的学习曲线。
- **准备将 DSPy 用于生产环境**：一名成员正在寻求将他们的 **DSPy 应用**部署到生产环境的资源，突显了从实验到实施的转变。
   - 这反映了在寻求开发最佳实践的用户中，对实际使用 DSPy 工具的兴趣日益浓厚。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pivot-to-ai.com/2024/07/28/sapwned-saps-ai-cloud-cracked-by-simple-kubernetes-configuration-errors/">SAPwned: SAP’s AI cloud cracked by simple Kubernetes configuration errors</a>：SAP 是一家大型软件即服务业务智能提供商。他们提供 AI 产品 SAP AI Core，客户可以使用内部数据对其进行训练。不幸的是，SAP 似乎在实现上存在……</li><li><a href="https://github.com/Tyrrrz/DiscordChatExporter/blob/master/.docs/Getting-started.md">DiscordChatExporter/.docs/Getting-started.md at master · Tyrrrz/DiscordChatExporter</a>：将 Discord 聊天日志导出到文件。通过在 GitHub 上创建账号为 Tyrrrz/DiscordChatExporter 的开发做出贡献。</li><li><a href="https://github.com/Tyrrrz/DiscordChatExporter/releases">Releases · Tyrrrz/DiscordChatExporter</a>：将 Discord 聊天日志导出到文件。通过在 GitHub 上创建账号为 Tyrrrz/DiscordChatExporter 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1270029949683892405)** (2 条消息): 

> - `Pinecone limitations` 


- **Pinecone 在向量搜索中的低效**：一位成员表示 **Pinecone 不支持 multi-vector search**，这可能导致其应用中的效率低下。
   - 这一限制表明，用户在集成 Pinecone 时可能需要重新考虑其搜索策略。
- **关于 Late Interactions 的担忧**：有一场讨论强调了对 Late Interactions 的反应对于工作流流程来说是**低效的**。
   - 这引发了关于如何及时处理用户交互以提高整体生产力的疑问。


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1269275632190099526)** (33 条消息🔥): 

> - `Open Interpreter with Llama`
> - `Hugging Face API setup`
> - `Screenshot command execution`
> - `Speech recognition and translation`
> - `Open Interpreter system prompt` 


- **Open Interpreter 成功配合本地 LLM 运行**：一位用户报告称，通过使用 LM Studio 作为服务器，解决了将 Open Interpreter 与本地 LLM 集成的问题，并获得了 OI system prompt 的访问权限。
   - 他们发现这种集成非常有趣且具有启发性。
- **排查 Open Interpreter 中的 Hugging Face API 问题**：一位用户在 Open Interpreter 中设置 Hugging Face API 集成时遇到困难，尽管参考了文档，但仍面临错误。
   - 在获得支持后，他们表达了感谢，并希望问题能得到解决。
- **在 Open Interpreter 中执行截图命令**：一位用户质疑为什么 Open Interpreter 在收到请求时会生成大量代码，而不是直接执行截图命令。
   - 另一位用户分享了一种使用 'screencapture' 命令的成功方法，确认其按预期工作。
- **在 Open Interpreter 中实现母语输入输出 (I/O)**：一位用户提出了一种实现母语语音识别的方法，在英语和母语之间进行往返翻译。
   - 这种方法引发了对翻译错误潜在陷阱的担忧，并将其称为 'Garbage in Garbage out'（垃圾进，垃圾出）。
- **Open Interpreter 的通用翻译模式**：一位用户建议创建一个通用翻译模式，该模式将处理双向翻译而不考虑命令。
   - 这一想法旨在通过减少命令执行中的错误来增强用户与 Open Interpreter 的交互。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://localhost:11434"">未找到标题</a>: 未找到描述</li><li><a href="https://docs.openinterpreter.com/getting-started/setup">Setup - Open Interpreter</a>: 未找到描述</li><li><a href="https://nerdvittles.com/creating-an-api-key-for-google-speech-recognition/">Creating an API Key for Google Speech Recognition &#8211; Nerd Vittles</a>: 未找到描述</li><li><a href="https://api-inference.huggingface.co">Serverless Inference API</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1269075957051031676)** (1 条消息): 

> - `Open Interpreter on Linux`
> - `Electra AI Linux Distro`
> - `AI capabilities in OS`
> - `Linux flavors: Lindoz, Max, Shift` 


- **在 Linux 上探索 Open Interpreter**：一位成员一直在深入研究如何让 **Open Interpreter** 在 Linux 上运行，并分享了他们最近的经验和进展。
   - 目标是查看 **OI 是否可以在 Linux OS 上自由开放**，探索其潜力。
- **Electra AI：一个新的 AI Linux Distro**：一位成员发现了 **Electra AI**，这是一个在 OS 层面内置了 AI 能力的 Linux Distro，可以在其网站上免费使用。
   - 他们强调 Electra AI 提供三个版本：**Lindoz, Max, 和 Shift**，全部免费提供。



**提到的链接**: <a href="https://www.makululinux.com/wp/">Tweet from MakuluLinux &#8211; A Whole World of Possibilities</a>: 未找到描述

  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1269083495389597788)** (19 messages🔥): 

> - `Cohere 支持联系方式`
> - `GenAI Bootcamp 与 Cohere 模型`
> - `账单页面的 CORS 问题` 


- **Cohere 支持联系详情**：当被问及由于账单页面的 **CORS 问题** 需要支持时，社区成员建议发送电子邮件至 [support@cohere.com](mailto:support@cohere.com) 寻求帮助。
   - *请在咨询中包含您组织的电子邮件和更多详细信息。*
- **为 GenAI Bootcamp 探索 Cohere**：Andrew Brown 正在研究 **Cohere**，计划将其纳入一个免费的 **GenAI Bootcamp**，该项目今年的目标是覆盖 **5 万名参与者**。
   - 他表示需要文档之外的深入见解，并强调了 **Cohere 的云平台无关能力 (cloud-agnostic capabilities)** 和多样化的模型应用。
- **CORS 问题导致无法支付**：一名成员报告称，由于后端的 CORS 问题，**账单页面无法正常工作**，影响了他们支付服务费用的能力。
   - 他们正在寻求直接的联系方式以解决此账单错误。
- **Cohere 的模型特性**：针对咨询，成员们指出 Cohere 提供 **RAG 功能**、**多语言选项**以及众多的 **embedding 模型**。
   - 他们强调了通过 **API 调用** 进行集成的便利性，并提到了为开发者提供的各种工具。



**提到的链接**：<a href="https://www.youtube.com/watch?v=zA8guDqfv40">AWS Cloud Complete Bootcamp Course</a>：AWS Cloud Project Bootcamp 是一个培训计划，旨在让你掌握设计、构建和实施云项目的技能。⚠️ 如果你遇到问题...

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1270073728105189451)** (7 messages): 

> - `模型基准测试 (Benchmarking Models)`
> - `验证集划分 (Validation Splits)`
> - `Cohere For AI 社区` 


- **关于模型基准测试的讨论**：成员 @divyaa_a 询问，在对基于数据集不同划分训练的模型进行基准测试时，尽管测试集保持一致，**验证子集 (validation subset)** 是否也应保持不变。
   - 另一位成员建议，在所有模型中保持 **验证集 (validation set)** 相同将提供更受控的比较，从而实现准确的基准测试。
- **强调受控比较**：讨论强调了在 **基准测试 (benchmarking)** 中进行受控比较的重要性，以确保对不同建模方法的评估准确性。
   - 该成员建议保持验证子集的**一致性**，以增强比较的稳健性。
- **建议加入 Cohere For AI 社区**：一位成员鼓励 @divyaa_a 加入 **Cohere For AI 社区**，并表示他们将获得与基准测试咨询相关的宝贵见解。
   - 该社区以参与**前沿的开放科学研发 (open-science research and development)** 而闻名。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1270062172432629860)** (4 messages): 

> - `Cohere API 支付`
> - `Command R 模型混淆`
> - `Coral 模型解释` 


- **用户对 Cohere API 模型选择的困惑**：一位用户在支付 **Cohere API** 费用后表达了担忧，他们原本期望使用 **Command R** 模型，但发现只有 **Coral 模型** 可用。
   - 他们指出在支付前没有选择模型的选项，正在寻求帮助以切换到 **Command R**。
- **关于模型类型的澄清**：另一位成员澄清说，**Coral** 模型实际上是 **Command R+** 的一个版本。
   - 这一回答旨在缓解用户对支付后可用模型选项的困惑。


  

---

### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1269880246501507182)** (3 条消息): 

> - `Azure 上的 Rerank`
> - `Cohere RAG 应用模型`
> - `Cohere 与 Fujitsu 的合作伙伴关系` 


- **Rerank 已在 Azure Models 上可用**：Cohere 团队讨论了 **Rerank** 在 **Azure Models** 上的新可用性，及其与 **RAG app** 集成的潜力。
   - 团队有兴趣更新 toolkit，以为 Azure 用户激活 Rerank，详情见这篇 [博客文章](https://cohere.com/blog/introducing-rerank-3-on-microsoft-azure-ai)。
- **Cohere RAG 应用中使用的模型**：一名成员提到，他们在 Azure 上的 RAG app 中使用了 **Command R+**、**Rerank** 和 **Embed** 模型。
   - 这一信息强调了依赖多种模型组合来增强功能。
- **Cohere 与 Fujitsu 的合作伙伴关系**：团队还提到了 **Cohere** 与 **Fujitsu** 之间的战略合作伙伴关系，旨在为日本企业提供 AI 服务。
   - 更多详情，成员可以参考 [合作公告](https://cohere.com/blog/fujitsu-partnership)。



**提到的链接**：<a href="https://cohere.com/blog/introducing-rerank-3-on-microsoft-azure-ai">在 Microsoft Azure AI 上推出 Rerank 3</a>：我们很高兴地宣布，Rerank 3 现在已在 Microsoft Azure AI Studio 上可用。Rerank 3 是一个基础模型，通过对初始搜索结果进行重新排序来增强现有的搜索系统...

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1270024181471248506)** (4 条消息): 

> - `tinygrad 0.9.2 版本发布`
> - `Aurora 超级计算机可行性`
> - `XMX 支持与 OpenCL`
> - `MLPerf 基准测试`
> - `tinygrad 功能悬赏` 


- **tinygrad 0.9.2 发布并带来令人兴奋的更新**：周一的会议重点介绍了 **tinygrad 0.9.2** 的更新，包括 **faster gemv**、**kernel timing** 和 **CapturedJit** 改进等功能。
   - 其他讨论包括 **indexing alu**、**uop symbolic** 以及提升 **ResNet** 性能等话题。
- **探索 tinygrad 在 Aurora 超级计算机上的可行性**：鉴于 Intel GPU 的限制，一名成员询问了在阿贡国家实验室（Argonne National Laboratory）的 **Aurora** 超级计算机上运行 **tinygrad** 的可行性。
   - 虽然有人对兼容性表示担忧，但讨论暗示目前已支持 **OpenCL**，尽管可能较慢。
- **关于启用 XMX 支持的讨论**：有人提到可能正在开发 **tinygrad** 的 **XMX** 支持，这将增强 Intel 架构的功能。
   - 提到 *OpenCL* 已经可以工作，但在没有第一手经验的情况下，其有效性尚不确定。
- **宣布 tinygrad 增强功能的悬赏**：讨论了针对各种 **tinygrad 功能** 的悬赏（Bounties），例如 **fast sharded llama**、**std one kernel** 以及针对 **AMX** 和 **Qualcomm** 的增强。
   - 社区被激励参与这些开发，旨在提高 tinygrad 的整体功能。
- **MLPerf 基准测试概述**：周一的会议还涵盖了 **MLPerf** 基准测试，包括与 **Unet3D**、**BERT**、**RetinaNet** 和 **StableDiffusion** 相关的任务。
   - 这些基准测试对于评估 tinygrad 在各种平台上的运行性能至关重要。



**提到的链接**：<a href="https://github.com/patrick-kidger/jaxtyping">GitHub - patrick-kidger/jaxtyping：用于 JAX/NumPy/PyTorch 等数组形状和数据类型的类型注解和运行时检查。https://docs.kidger.site/jaxtyping/</a>：用于 JAX/NumPy/PyTorch 等数组形状和数据类型的类型注解和运行时检查。https://docs.kidger.site/jaxtyping/ - patrick-kidger/jaxtyping

  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1269196489029320735)** (25 messages🔥): 

> - `CUDA vs CLANG 性能`
> - `在 Tensor 上执行自定义 Kernel`
> - `PyO3 接口错误排查`
> - `Codegen 中的 ShapeTrackers`
> - `Tensor push 优化` 


- **CUDA 运行速度慢于 CLANG**：一位成员询问为什么运行 `CUDA=1 pytest test_winograd.py` 比 `CLANG=1 pytest test_winograd.py` 慢，原以为 CUDA 会比 C 快。
   - 这引发了对某些测试中 CUDA 执行可能存在的问题或低效的关注。
- **在 Tensor 上使用自定义 Kernel**：一位用户询问在 Tensor 上运行自定义 Kernel 的简洁方法，并参考了 [GitHub 文件](https://github.com/tinygrad/tinygrad/blob/da61dea1b2ca886b3de07e309efde2a78ac5682a/test/test_custom_function.py#L42-L43) 获取详细信息。
   - 这突显了关于 Tinygrad 中高级 Tensor 操作的持续讨论。
- **PyO3 中的递归限制错误**：一位成员报告在使用 PyO3 接口调用 `tinygrad.nn.state.safe_save` 时出现递归限制错误。
   - 建议尝试 `TRACEMETA=0` 来解决此问题，这表明此类工具在非 CPython 实现中可能无法正常工作。
- **评估 ShapeTrackers 以进行优化**：讨论了在 ShapeTracker 系统中使用符号索引的问题，询问该库是否采用了 Symbolic Shapes。
   - 一位成员建议，专注于减少表达式树（Expression Trees）可能比直接改进 ShapeTrackers 更有益。
- **优化 Tensor 数值插入**：一位成员寻求将单个值 (f64) 推入 Tensor 的最有效方法，并指出使用 `.cat` 效率低下。
   - 建议预先分配空间然后赋值给切片，但由于非连续（Non-contiguous）Tensor 导致了断言错误。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/59315ffc7839948a032e366ba8d964c345d835ff/tinygrad/tensor.py#L3158-L3189">tinygrad/tinygrad/tensor.py at 59315ffc7839948a032e366ba8d964c345d835ff · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 Micrograd？你一定会爱上 Tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/da61dea1b2ca886b3de07e309efde2a78ac5682a/test/test_custom_function.py#L42-L43">tinygrad/test/test_custom_function.py at da61dea1b2ca886b3de07e309efde2a78ac5682a · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 Micrograd？你一定会爱上 Tinygrad！❤️ - tinygrad/tinygrad
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1270120968878293013)** (1 messages): 

> - `PPO 训练 Recipe`
> - `Qwen2 模型支持` 


- **Torchtune 新增 PPO 训练 Recipe**：团队引入了端到端的 **PPO 训练 Recipe**，将 RLHF 集成到 Torchtune 中，详见 [GitHub Pull Request](https://github.com/pytorch/torchtune/pull/1005)。
   - *快去看看并尝试一下吧！*
- **训练 Recipe 已支持 Qwen2 模型**：**Qwen2 模型支持**已添加到训练 Recipe 中，**7B 模型**现已在 [GitHub Pull Request](https://github.com/pytorch/torchtune/pull/1143) 中提供，更小版本即将推出。
   - 期待即将发布的 **1.5B** 和 **0.5B** 版本！
- **征集模型和 Recipe 需求反馈**：团队邀请用户分享他们希望在 Torchtune 中实现的其它模型或 Recipe 的**功能请求**。
   - 你可以通过 [此 GitHub 链接](https://github.com/pytorch/torchtune) 提交请求。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1269092750637269065)** (12 messages🔥): 

> - `LLAMA 3 8B INSTRUCT model`
> - `Token generation issues` (Token 生成问题)
> - `Chat formatting in LLAMA 3` (LLAMA 3 中的聊天格式化)
> - `Debugging generation mode` (调试生成模式)


- **使用自定义配置运行 LLAMA 3 生成**：用户成功使用自定义配置运行了 LLAMA 3 8B INSTRUCT 模型，针对当前时间的查询，输出结果为 **12:34 PM**。
   - 生成过程耗时 **27.19 秒**，速度为 **12.25 tokens/sec**，消耗了 **20.62 GB** 内存。
- **文本重复生成问题**：用户报告生成的文本经常重复 **10 次**，有时会意外包含结束 Token。
   - 另一位用户建议参考一个 [Pull Request](https://github.com/pytorch/torchtune/pull/1211)，该 PR 旨在修复结束 Token 的问题，目前仍在审查中。
- **探索聊天格式的影响**：用户询问定义 `chat_format` 是否有助于改善 LLAMA 3 模型的输出。
   - 回复者表示，对于 LLAMA 3 来说，设置聊天格式并非必要。
- **生成过程中对调试模式的需求**：用户对缺乏能够显示**所有 Token**（包括对话周围的 `<|user|>` 和 `<|assistant|>` 等 Token）的生成模式表示担忧。
   - 回复者认可了这一点，并提到他们可能会在生成脚本中添加一个参数来实现此功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sdk.vercel.ai/playground">AI Playground | 并排比较顶级 AI 模型</a>：聊天并比较 OpenAI GPT, Anthropic Claude, Google Gemini, Llama, Mistral 等模型。</li><li><a href="https://github.com/pytorch/torchtune/pull/1211.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、Fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1269978058660773949)** (5 messages): 

> - `Model Blurbs` (模型简介)
> - `Llama3 Review` (Llama3 审查)


- **关于维护模型简介的担忧**：成员们表达了对提供更新的模型简介（Blurbs）的担忧，担心这可能太难维护。
   - 有人建议，**来自模型卡片（Model Card）或白皮书的快照**可以作为最简简介。
- **重启模型讨论**：有人指出，进行审查将有助于重启关于模型的讨论，目前只有 **Llama3** 完成了审查。
   - 一名成员提议，如果获得许可，他可以负责查看并可能**添加/更新其他模型的简介**。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1269013466925236276)** (9 messages🔥): 

> - `bitsandbytes installation for ROCm` (针对 ROCm 的 bitsandbytes 安装)
> - `AI Nutritionist dataset creation` (AI 营养师数据集创建)
> - `DPO vs alternatives in AI training` (AI 训练中 DPO 与替代方案的对比)


- **简化 ROCm 的 bitsandbytes 安装**：[最近的一个 Pull Request](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1299) 实现了在 **ROCm** 上打包 **bitsandbytes** 的 Wheel 文件，为用户简化了安装过程。
   - 该 PR 更新了 **ROCm 6.1** 的编译流程，以支持最新的 **Instinct** 和 **Radeon** GPU。
- **构建 AI 营养师需要数据集**：有人正在开发 **AI 营养师** 并考虑微调 **GPT-4o mini**，但正在寻找合适的营养数据集，如 [USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets.html)。
   - 建议包括从 **FNDDS** 编译潜在的数据集，但不确定其是否在 **Hugging Face** 上可用。
- **关于 DPO 替代方案的讨论**：一名成员质疑 **DPO** 是否仍然是 AI 训练中的最佳方法。其他人建议 **ORPO**、**SimPO** 或 **KTO** 等替代方案可能更优。
   - 这引发了关于各种方法在 AI 模型训练中有效性的不同观点的交流。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/bitsandbytes">bitsandbytes - 概览</a>：bitsandbytes 有 6 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://huggingface.co/datasets/Roger21/NutritionFineTune_1?row=45)">Roger21/NutritionFineTune_1 · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1299">Enable bitsandbytes packaging for ROCm by pnunna93 · Pull Request #1299 · bitsandbytes-foundation/bitsandbytes</a>：此 PR 实现了在 ROCm 上为 bitsandbytes 打包 Wheel 文件。它更新了 ROCm 编译和 Wheel 构建任务，以便在 ROCm 6.1 上为最新的 Instinct 和 Radeon GPU 进行编译。
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1270015574436479028)** (6 messages): 

> - `FFT for Large Models`
> - `LORA/QLORA Baseline Testing`
> - `L40S GPUs Performance` 


- **寻找 FFT 和基准测试**：一名成员表示有兴趣寻找 **FFT** 或 **LoRA/QLORA** 来对 **27b 模型**进行实验，提到在 **9b 模型**上取得了不错的效果，但在更大的模型上难以复现这些结果。
   - *Caseus_* 建议可能存在适用于 **Gemma 2 27b** 的 **QLORA** 版本，通过对学习率进行一些调整并使用最新的 **flash attention** 可能会奏效。
- **咨询 L40S GPUs 性能**：一名成员询问是否有人在 **L40S GPUs** 上训练或部署过模型，寻求有关其性能的信息。
   - 该咨询凸显了人们对 **L40S GPUs** 在 AI 模型训练中的效率和能力的关注。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1269251406712410132)** (1 messages): 

> - `AI Nutritionist Development`
> - `Existing Nutrition Datasets` 


- **构建 AI 营养师**：一名成员正在开发一个 **AI Nutritionist**，并寻求在营养或食物数据集上微调 **GPT-4o mini** 模型。
   - 他们正在考虑是创建自己的数据集还是利用现有数据集。
- **建议的数据集**：他们提到了两个数据集：[USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets.html) 和 [HF 上的 NutritionFineTune_1](https://huggingface.co/datasets/Roger21/NutritionFineTune_1?row=45)。
   - 该成员专门询问了 **BARLEY, PEARL (BERAS BELANDA) HORDEUM VULGARE** 的营养信息，提供了详细的营养含量，例如每 **100 克** 含有 **335.0 kcal**。



**提到的链接**：<a href="https://huggingface.co/datasets/Roger21/NutritionFineTune_1?row=45)">Roger21/NutritionFineTune_1 · Datasets at Hugging Face</a>：未找到描述

  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1269435376511090752)** (1 messages): 

> - `Triton Conference Registration`
> - `Event Details`
> - `Free Attendance`
> - `Google Form Signup` 


- **Triton 会议注册现已开放！**：将于 **2024 年 9 月 17 日**在 **加州弗里蒙特 Meta 园区**举行的 **Triton Conference** 现已开放注册！通过[此 Google 表单](https://docs.google.com/forms/d/e/1FAIpQLSecHC1lkalcm0h3JDUbspekDX5bmBvMxgVTLaK3e-61bzDDbg/viewform)报名以预留名额。
   - 参会是**免费**的，但名额**有限**，因此建议尽早注册。
- **注册所需信息**：注册时，参与者必须提供**电子邮件**、**姓名**、**所属机构**和**职位**。可选问题包括背景信息以及参会者希望从会议中获得什么。
   - 还会收集饮食偏好，提供**素食**、**纯素食**、**犹太洁食**和**无麸质**等选项。
- **Google 表单需要登录**：参会者会被提示[登录 Google](https://accounts.google.com/AccountChooser?continue=https://docs.google.com/forms/d/e/1FAIpQLSecHC1lkalcm0h3JDUbspekDX5bmBvMxgVTLaK3e-61bzDDbg/viewform&service=wise) 以在填写注册表单时保存进度。回复副本将发送至提供的电子邮箱。
   - 提醒参与者切勿通过 Google 表单提交密码，以确保安全。



**提到的链接**：<a href="https://docs.google.com/forms/d/e/1FAIpQLSecHC1lkalcm0h3JDUbspekDX5bmBvMxgVTLaK3e-61bzDDbg/viewform">[报名] Triton Conference 2024 年 9 月 17 日</a>：Meta 园区，加州弗里蒙特 - 免费参加（但名额有限）

  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1270115495273697293)** (1 条消息): 

> - `Llamafile 进展`
> - `八月项目讨论`
> - `sqlite-vec 发布派对`
> - `Machine Learning 论文研讨会`
> - `Local AI AMA` 


- **Llamafile 推动离线 LLMs 进展**：Llamafile 核心维护者在项目上继续取得史诗般的进展，实现了在单个文件中运行**离线、易用的 LLMs**。
   - 这一进展旨在提高用户可访问性，并简化与大语言模型的交互。
- **社区分享八月项目**：发起了一项关于大家 **八月** 工作内容的新讨论，邀请社区成员分享他们正在进行的项目。
   - 这是一个与同行交流并展示 Mozilla AI 社区内个人贡献的机会。
- **sqlite-vec 发布派对公告**：计划举行 sqlite-vec 发布派对，参与者可以讨论 **特性**、尝试演示并与核心维护者互动。
   - 参与者可以加入关于 sqlite-vec 最新进展的对话，预计将引发深入的讨论。
- **Machine Learning 论文研讨会活跃开展**：即将举行的讲座包括关于 **Communicative Agents** 和 **Extended Mind Transformers** 的讨论，届时将有知名演讲者出席。
   - 这些活动提供了对机器学习领域前沿研究和协作交流的深入见解。
- **Local AI AMA 提供替代方案**：安排了一场与 **Local AI** 核心维护者的 AMA 环节。Local AI 是 OpenAI 的开源替代方案，用户可以自行托管。
   - 此次活动是了解 Local AI 功能并直接提问的绝佳机会。



**提及的链接**：<a href="https://form.typeform.com/to/Cn4md4Oc>)">探索 Typeform，让表单变得有趣</a>：无需代码，几分钟内即可创建美观的交互式表单。免费开始使用。

  

---



---



---



---



{% else %}


> 完整的逐频道细分内容已针对电子邮件进行缩减。
> 
> 如果您想查看完整细分，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}