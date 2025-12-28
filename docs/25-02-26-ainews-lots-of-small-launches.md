---
companies:
- openai
- anthropic
- amazon
- cloudflare
- perplexity-ai
- deepseek-ai
- togethercompute
- elevenlabs
- elicitorg
- inceptionailabs
- mistral-ai
date: '2025-02-27T04:09:12.976879Z'
description: '以下是该文本的中文翻译：


  **GPT-4o 高级语音预览版 (Advanced Voice Preview)** 现已面向 ChatGPT 免费用户开放，并提高了 Plus 和 Pro
  用户的每日使用限制。**Claude 3.7 Sonnet** 凭借更高的 Token 效率，在 WebDev Arena 中荣登榜首。拥有 6710 亿参数的
  **DeepSeek-R1** 受益于 **Together Inference** 平台对 NVIDIA Blackwell GPU 的优化，同时开源的 **DeepGEMM**
  CUDA 库在 Hopper GPU 上实现了高达 2.7 倍的加速。**Perplexity** 推出了全新的语音模式和 **深度研究 (Deep Research)
  API**。即将推出的 **Grok 3 API** 将支持 100 万 (1M) Token 的上下文窗口。包括 **Elicit**、**亚马逊 (Amazon)**、**Anthropic**、**Cloudflare**、**FLORA**、**Elevenlabs**
  和 **Inception Labs** 在内的多家公司宣布了新一轮融资、产品发布或模型更新。'
id: f98b3a6c-3c56-432d-a822-c9a4ed105451
models:
- gpt-4o
- claude-3.7-sonnet
- claude-3.7
- claude-3.5-sonnet
- deepseek-r1
- deepseek-v3
- grok-3
original_slug: ainews-lots-of-small-launches
people:
- lmarena_ai
- alexalbert__
- aravsrinivas
- reach_vb
title: '根据语境，这句话可以翻译为：


  1.  **通用/产品发布：** 许多小型发布

  2.  **航天/火箭：** 大量小规模发射

  3.  **商业/项目：** 多次小规模启动'
topics:
- voice
- model-releases
- cuda
- gpu-optimization
- inference
- open-source
- api
- model-performance
- token-efficiency
- context-windows
- cuda
- jit-compilation
---

<!-- buttondown-editor-mode: plaintext -->**平静的一天。**

> 2025年2月25日至2月26日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 服务器（**221** 个频道，**7040** 条消息）。预计节省阅读时间（以 200wpm 计算）：**725 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

- [GPT 4.5 将于本周发布](https://x.com/steph_palazzolo/status/1894785791018332505?s=46)
- [Elicit 宣布 A 轮融资并推出自家的 Deep Research](https://x.com/elicitorg/status/1894772293752266846?s=46)
- [Alexa+ 采用 Amazon Nova 和 Anthropic Claude 进行了更新](https://x.com/mikeyk/status/1894783669920817321?s=46)
- [Cloudflare 发布了 Agents SDK](https://x.com/threepointone/status/1894399506277376369?s=46)
- [FLORA 推出了其 Krea 竞品](https://x.com/weberwongwong/status/1894794612398792974?s=46)
- [Elevenlabs 推出了 ASR](https://x.com/matistanis/status/1894824212382257427?s=46)
- [Perplexity 推出了 Deep Research API](https://x.com/aravsrinivas/status/1894471526449385687?s=46)（估值达 150 亿）
- [Inception labs 推出了生产级语言扩散模型 (Language Diffusion Model)](https://x.com/InceptionAILabs/status/1894847919624462794)

---

{% if medium == 'web' %}

**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 回顾

**AI 模型更新与发布，侧重于新模型、功能和版本**

- **面向免费用户的 GPT-4o Advanced Voice 预览版**：[@OpenAI](https://twitter.com/OpenAI/status/1894495906952876101) 宣布向所有 **ChatGPT 免费用户**推出由 **GPT-4o mini** 驱动的 **Advanced Voice**，在各平台上提供具有自然对话节奏和成本效益的每日预览。[@OpenAI](https://twitter.com/OpenAI/status/1894495908366356607) 还详细说明了 **Plus 和 Pro 用户**的持续访问权限，**Plus 用户**保留对由 **4o** 驱动的 **Advanced Voice** 的访问权限，其**每日速率限制比免费用户高 5 倍**，而 **Pro 用户**则保持**无限访问**以及**更高的视频和屏幕共享限制**。
- **Claude 3.7 Sonnet 发布与性能**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1894840263379689490) 报告称 **Claude 3.7 Sonnet** 已夺得 **WebDev Arena 第一名**，以 **+100 分的跨度**超越了 **Claude 3.5 Sonnet**。[@alexalbert__](https://twitter.com/alexalbert__/status/1894807853371990087) 提到 **Claude 3.7 Sonnet** 采用了**更节省 Token 的工具调用 (Tool Use) 实现**，使用的 **Token 减少了 14%** 且性能有所提升，可通过 Beta Header `"token-efficient-tools-2025-02-19"` 访问。
- **DeepSeek R1 推理平台与 DeepGEMM**：[@togethercompute](https://twitter.com/togethercompute/status/1894515568088412198) 强调拥有 6710 亿参数的 **DeepSeek-R1** 需要推理平台来最大化 **NVIDIA Blackwell GPU** 的利用率，**Together Inference** 正在为 **DeepSeek-R1** 优化 GPU 效率。[@reach_vb](https://twitter.com/reach_vb/status/1894626368702304617) 宣布了 **DeepSeek** 的 **DeepGEMM**，这是一个轻量级的 **CUDA 库**，用于在 **NVIDIA Hopper Tensor Cores** 上进行高效的 **FP8 GEMM**，性能优于专家调优的库，在 **DeepSeek-V3/R1 推理任务**中实现了高达 **2.7 倍的加速**。[@deepseek_ai](https://twitter.com/deepseek_ai/status/1894553164235640933) 正式介绍了 **DeepGEMM**，作为其**开源周 (Open Source Week)** 的一部分，指出其在 **Hopper GPU** 上达到了 **1350+ FP8 TFLOPS** 的性能，支持 JIT 编译，核心逻辑仅约 **300 行**。
- **Perplexity 语音模式与 Deep Research API**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1894792092284789173) 宣布发布全新的 **Perplexity 语音模式**，融合了跨语言的**实时语音和信息**，已在 iOS 上线，Android 版即将推出。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1894820042816307467) 还提到了 **Deep Research API**，作为 Perplexity 最近更新的一部分。
- **具备 1M 上下文的 Grok 3 API**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1894733474239582465) 提到了即将推出的具备 **1M 上下文**的 **Grok 3 API**。

**AI 工具、库和数据集，涵盖框架、代码和资源**

- **用于 FP8 GEMM 的 DeepGEMM 开源库**：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1894553164235640933) 开源了 **DeepGEMM**，这是一个基于 **CUDA** 的库，用于高效的 **FP8 GEMM**，强调了其性能和简洁的代码库。[@danielhanchen](https://twitter.com/danielhanchen/status/1894554391140864240) 也重点介绍了 **DeepGEMM**，指出了其 **JIT 编译**以及在 **FP8 矩阵乘法**中的效率。
- **用于 LLM 评估的 OpenEvals 开源仓库**：[@LangChainAI](https://twitter.com/LangChainAI/status/1894821108018262297) 宣布了 **OpenEvals**，这是一个新的 **OSS 仓库**，包含预构建的评估器，旨在简化为 LLM 应用添加评估的过程，支持 Python 和 JS。
- **用于多智能体系统的 LangGraph Swarm**：[@LangChainAI](https://twitter.com/LangChainAI/status/1894795982379848168) 推出了 **LangGraph Swarm**，这是一个轻量级库，用于使用 **LangGraph** 构建**群集式（swarm-style）多智能体系统**，支持智能体协作和可定制的通信工具。
- **LangGraph Platform 自定义路由**：[@LangChainAI](https://twitter.com/LangChainAI/status/1894795878504055053) 宣布了 **LangGraph Platform** 中的 **Custom Routes**，允许通过自定义 HTTP 端点进行扩展，以便在 Python 中使用单一后端构建全栈 AI 应用。
- **用于实时 LLM 排行榜的 P2L (Prompt-to-Leaderboard)**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1894767009977811256) 推出了 **Prompt-to-leaderboard (P2L)**，这是一个开源系统，基于来自 **Chatbot Arena** 的 **200 万个人类偏好投票**，训练 LLM 生成特定提示词的排行榜。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1894767022791438490) 分享了 **P2L 论文和代码**的链接，强调了其开源特性。
- **Vevo Therapeutics 发布 Tahoe-100M 数据集**：[@sarahcat21](https://twitter.com/sarahcat21/status/1894784421611680209) 重点介绍了 **Vevo Therapeutics 开源发布**的 **Tahoe-100M 数据集**，旨在为 FM 驱动的药物开发解锁高质量数据。
- **用于具身多智能体任务的 Meta PARTNR 数据集和代码**：[@AIatMeta](https://twitter.com/AIatMeta/status/1894524602854117781) 发布了 **Meta PARTNR 数据集和代码**，这是一个用于具身多智能体任务中规划和推理的基准测试，已在其最近的机器人演示中使用。[@AIatMeta](https://twitter.com/AIatMeta/status/1894524604900938078) 提供了数据集和代码的直接链接。
- **用于 LLM 评估的 OpenEvals 仓库**：[@LangChainAI](https://twitter.com/LangChainAI/status/1894821108018262297) 宣布发布 **OpenEvals**，这是一个包含预构建评估器的开源仓库，旨在帮助用户轻松评估 LLM。

**研究、分析与基准测试，涵盖评估、性能和见解**

- **SWE-RL: Meta 用于 Software Evolution Benchmark 的 RL**：[@_akhaliq](https://twitter.com/_akhaliq/status/1894584315352076608) 报道了 **Meta 的 SWE-RL**，这是一种在 **Open Software Evolution** 数据上使用 **Reinforcement Learning** 的方法，使用 **Llama3-SWE-RL-70B** 在 **SWE-bench Verified** 上实现了 **41.0% 的解决率**，在中型模型中与 **GPT-4o** 相当。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1894596772804350016) 也强调了 **Meta 的 SWE-RL**，利用 **Llama 3** 在 **SWE-bench Verified** 上实现了 State-of-the-art 性能。
- **Prompt-to-Leaderboard (P2L) 性能分析**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1894767012490174779) 详细介绍了 **P2L-router** 的性能，其在 **2025 年 1 月的 Chatbot Arena** 中以 **1395** 分位居第一，且受成本限制的 P2L 模型达到了 Pareto frontier。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1894767018014130204) 进一步解释了 **P2L 用于模型弱点分析**，识别了跨领域的优势和劣势，[@lmarena_ai](https://twitter.com/lmarena_ai/status/1894767014767673744) 强调了其在特定领域排行榜中的应用，实现了自适应类别排名。
- **Anthropic 的风险预测研究**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1894495059954860055) 宣布了关于 **预测罕见语言模型行为** 的新研究，通过有限的测试数据预测部署风险，[@AnthropicAI](https://twitter.com/AnthropicAI/status/1894495065612939629) 指出他们的预测在实验中准确预判了滥用和 Misalignment 风险。
- **用于长上下文任务的 MoBA (Mixture of Block Attention)**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1894711958353629617) 报道了来自 **Kimi Moonshot** 的 **MoBA (Mixture of Block Attention)**，它改进了长上下文任务处理，在 **1M tokens** 下比 Full Attention 实现了 **6.5 倍加速**。
- **FFTNet：基于 FFT 的 Self-Attention 替代方案**：[@omarsar0](https://twitter.com/omarsar0/status/1894757821587296614) 总结了一篇介绍 **FFTNet** 的论文，该方案使用 **FFT** 的 **Adaptive Spectral Filtering** 取代了 Self-Attention，将复杂度降低至 **O(n log n)**，并在基准测试中表现出竞争力。
- **可解释性研究中的 Linear Probes vs. SAEs (Sparse Autoencoders)**：[@NeelNanda5](https://twitter.com/NeelNanda5/status/1894749262757634405) 讨论了一项研究，发现 **Linear Probes 在 5 种机制和 100 多个数据集中表现优于 SAEs**，这是对 SAEs 在可解释性方面的一个负面更新。

**行业和公司公告，涵盖合作伙伴关系、融资和活动**

- **Amazon Alexa+ 由 Claude 提供支持**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1894798008623026503) 宣布 **Claude 与 Amazon 建立合作伙伴关系**，为下一代 **Alexa+ AI 助手**提供支持。[@_philschmid](https://twitter.com/_philschmid/status/1894816750895575161) 详细介绍了 **Alexa+** 的功能，包括 Amazon Nova 与 Anthropic Claude 的集成、新的 "Tool" API、浏览器使用功能以及订阅模式。
- **Elicit 获得 2200 万美元 A 轮融资并发布 Elicit Reports**: [@Fraser](https://twitter.com/Fraser/status/1894779613210878434) 宣布 **Spark Capital 领投了对 Elicit 的 2200 万美元投资**，同时 [@elicitorg](https://twitter.com/elicitorg/status/1894772293752266846) 发布了 **Elicit Reports**，这是一款旨在自动化科学理解的研究工具。
- **Figure Robotics 扩大人形机器人生产规模**: [@adcock_brett](https://twitter.com/adcock_brett/status/1894782815981711810) 宣布 **Figure 正在加速在 2025 年以前所未有的水平交付人形机器人**，并强调了其 **Helix AI 的进展**以及与 BMW 的客户使用案例。[@adcock_brett](https://twitter.com/adcock_brett/status/1894781636153405870) 表示，**Helix** 使机器人能够通过单一神经网络进行扩展，从而显著缩短客户用例的开发时间。
- **Google Gemini Code Assist 免费版**: [@Google](https://twitter.com/Google/status/1894816225575731366) 宣布面向个人全球推出 **Gemini Code Assist 免费版**，并提供较高的使用限制。
- **Perplexity 收到 150 亿美元估值的 VC 投资意向**: [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1894785791018332505) 报道称，**Perplexity 正收到 150 亿美元估值的 VC 投资意向**，尽管他们不太可能接受，这突显了 VC 对具有创收能力的 AI 公司的兴趣。
- **DeepSeek API 闲时折扣**: [@deepseek_ai](https://twitter.com/deepseek_ai/status/1894710448676884671) 宣布在 **每日 16:30–00:30 UTC 期间**，**DeepSeek API 平台**提供 **闲时折扣**，其中 **DeepSeek-V3 享 5 折优惠**，**DeepSeek-R1 享 2.5 折优惠**。
- **Hugging Face Enterprise 升级用户增长**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1894778750631371149) 宣布 **超过 2,000 家机构**已升级到 **Hugging Face Enterprise**，其中包括各行各业的大型公司。
- **MLSYS 2025 青年专业人员研讨会征稿**: [@realDanFu](https://twitter.com/realDanFu/status/1894576091777700128) 宣布为 **5 月 12 日在圣克拉拉举行的 MLSys 2025 青年专业人员研讨会**征集摘要，截止日期为 **4 月 7 日**。
- **3 月 17 日在旧金山举行的 Perplexity 开发者活动**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1894537361033511094) 宣布将于 **3 月 17 日在 Perplexity 旧金山办公室**举行 **开发者活动**，邀请开发者与 API 团队见面并分享反馈。

**观点与讨论，涵盖更广泛的 AI 视角和评论**

- **AI 工程重心的转移**: [@nrehiew_](https://twitter.com/nrehiew_/status/1894513333719515452) 建议 **AI 工程**应该是 **50% 的标准 SWE、10% 的 TPOT 用户**（以增强模型意识）以及 **40% 的 UX**，并强调应用程序不一定非得是聊天机器人。
- **OpenAI 的市场领导地位与挑战**: [@madiator](https://twitter.com/madiator/status/1894611101884846601) 讨论了 **OpenAI 的市场地位**，强调了其领导地位、品牌知名度和基础设施，但也指出了高成本和竞争等挑战，同时肯定了他们在实现 Scaling、数据获取和 RL 微调产品化方面的贡献。
- **LLM 与代码库理解**: [@qtnx_](https://twitter.com/qtnx_/status/1894843415529181665) 反驳了关于 LLM 会导致人们不再理解代码库的担忧，并将其比作在团队中工作，因为在团队中理解他人的代码本就是必要的。
- **Cursor 与自主编码的对比**: [@jxmnop](https://twitter.com/jxmnop/status/1894830128082940182) 提醒注意 **将代码外包给 Copilot/Cursor 的心理成本**，将其比作抵押贷款，并建议除了简单的自动补全之外，凡事亲力亲为从长期来看可能更有效率。
- **模型训练与开源的重要性**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1894831939305218559) 强调 “**模型即产品！**”，并指出长期产品的成功需要学习如何基于开源模型进行训练。
- **ChatGPT 时刻的定义**: [@_aidan_clark_](https://twitter.com/_aidan_clark_/status/1894506681025138799) 澄清说，“**ChatGPT 时刻**” 是指人们意识到聊天机器人是有用的时刻，而不是技术变得可行的时刻。

- **AI Safety 与 AI 交易**：[@RyanPGreenblatt](https://twitter.com/RyanPGreenblatt/status/1894515270108283368) 讨论了 **AI safety** 与 **economics** 及 **psychology** 之间日益增加的交集，并提到了一档讨论与 AI 达成交易的播客。
- **AI 与虚假信息怀疑论**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1894827528973348919) 认为对 AI 生成的虚假信息的恐惧被夸大了，并指出 AI 媒体培养了公众的怀疑态度以及对社交验证的依赖。
- **In-Context Learning 与涌现能力**：[@giffmana](https://twitter.com/giffmana/status/1894524234392625199) 讨论了关于 **in-context learning** 和涌现能力的研究，指出这证实了大模型的泛化能力，并将“后门”重新定义为“**conditioning**”。
- **对 AI 研究数据获取及兴趣的批评**：[@BlancheMinerva](https://twitter.com/BlancheMinerva/status/1894756271489724772), [@BlancheMinerva](https://twitter.com/BlancheMinerva/status/1894756269195506050), [@BlancheMinerva](https://twitter.com/BlancheMinerva/status/1894756265231880346) 对 AI 研究中缺乏训练数据访问权限以及在没有适当数据分析的情况下急于声称 **OOD** 性能表示担忧。
- **具有递归块的 Transformers 构想**：[@jxmnop](https://twitter.com/jxmnop/status/1894567121793028158) 提议构建具有 **recursive blocks** 的 **Transformers** 而非典型模块，认为这可能以 **GPU** 不友好为代价换取潜在的表达能力提升。
- **Transformers 中 MLP 维度问题**：[@jxmnop](https://twitter.com/jxmnop/status/1894527828630147562) 质疑为什么 **Transformers** 中的 **MLP** 会投影到更大的维度然后再缩小，并好奇权重矩阵为什么不能是方阵。
- **科学理解滞后于模型部署**：[@_jasonwei](https://twitter.com/_jasonwei/status/1894821797000028357) 观察到在竞争激烈的模型产品领域，对模型的科学理解往往滞后于部署速度，但消融实验（ablation studies）仍具有价值。
- **RLHF 与模型对齐失误**：[@jd_pressman](https://twitter.com/jd_pressman/status/1894493541591871969) 假设调整 **GPT4o** 去编写带有 bug 的代码会导致广泛的 **misalignment**，因为 **RLHF** 偏好变得核心化。
- **邓巴数作为“邓巴砖墙”**：[@DavidSHolz](https://twitter.com/DavidSHolz/status/1894628680183550393) 评论说 **Dunbar's number** 感觉更像是一堵“**brick wall**”。
- **对“Heteroscadasticity”术语的批评**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1894852822706872658) 幽默地批评了“**Heteroscadasticity**”（异方差性）这个术语，认为它不直观且带有《功夫熊猫》式的风格。
- **组合与抽象在 ML 中的重要性**：[@lateinteraction](https://twitter.com/lateinteraction/status/1894719046760968550) 论证了 **composition** 和 **abstraction** 在计算机科学和 ML 中的重要性，并指出由于抽象与实现紧密耦合，它们在现代 ML 的自我认知中是缺失的。
- **Late Interaction 与 Multi-Vector 术语之争**：[@lateinteraction](https://twitter.com/lateinteraction/status/1894696983077785980) 讨论了针对类 ColBERT 方法使用“**late interaction**”还是“**multi-vector**”的术语问题，认为“**late interaction**”更准确，因为其机制不仅涉及多向量，还涉及可学习性和评分函数。
- **对训练、拼接、检索之外第四种 Conditioning 机制的需求**：[@lateinteraction](https://twitter.com/lateinteraction/status/1894669414454485033) 质疑除了训练、拼接和检索之外，LM 是否还需要第四种 **conditioning** 机制。
- **Late Alignment 的重要性**：[@lateinteraction](https://twitter.com/lateinteraction/status/1894666144055005693) 强调了在事实呈现后进行“**late alignment**”的必要性，无论是在 IR 还是 DSPy/RL 中，并告诫不要过度提前行动（precrastination）。
- **粒度评分的优越性**：[@lateinteraction](https://twitter.com/lateinteraction/status/1894662639839842346) 强调了在挑战性任务中，“**granular scoring**”比稠密点积具有更优越的泛化能力，并提倡使用 **late interaction**。
- **AI 驱动的解释权辩论**：[@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1894779433955037284) 总结了他参加的一场辩论，认为 **AI-powered interpretation** 最终将取代人类解释，并引用了算力趋势和 AI 的进步作为依据。

---

# AI Reddit Recap

## /r/LocalLlama Recap

**主题 1. DeepGEMM 提供高效的 FP8 通用矩阵乘法**

- **DeepSeek 发布第三弹！DeepGEMM：一个高效 FP8 通用矩阵乘法库** ([Score: 514, Comments: 105](https://reddit.com/r/LocalLLaMA/comments/1iybcnl/deepseek_realse_3th_bomb_deepgemm_a_library_for/)): **DeepGEMM** 是一个专注于高效 **FP8 通用矩阵乘法 (GEMMs)** 且支持精细缩放的库，正如在 **DeepSeek-V3** 中介绍的那样。该库可以通过 [此 GitHub 链接](https://github.com/deepseek-ai/DeepGEMM) 访问。
  - **DeepGEMM 的性能与影响**：DeepGEMM 的 FP8 矩阵乘法性能相比 NVIDIA 的 **CUDA** 库可提升 **2.7 倍**，从而使模型训练和推理更具成本效益。该库的便携性和 **JIT** 编译受到关注，尽管目前仅限于 **NVIDIA Hopper Tensor Cores**，但具有在各种架构上优化性能的潜力。
  - **行业影响与竞争力**：此次发布挑战了 **NVIDIA** 和 **OpenAI** 等公司的主导地位，引发了关于 **华为 910C** 与 NVIDIA **H100** 竞争潜力的讨论。人们对 NVIDIA 市场地位的可持续性表示担忧，并对 NVIDIA 的估值及更广泛的竞争格局所受的影响进行了推测。
  - **社区反应与潜力**：社区对 DeepGEMM 的潜力感到兴奋，讨论了其对模型训练成本和效率的影响。虽然有人对在训练中实现显著成本降低的可行性持怀疑态度，但基准测试和加速数据的提供有助于缓解部分疑虑。


**主题 2. 显存增加的 Nvidia 游戏 GPU 进入中国云市场**

- **[RTX 4090 48GB](https://www.reddit.com/gallery/1iy7e4x)** ([Score: 653, Comments: 221](https://reddit.com/r/LocalLLaMA/comments/1iy7e4x/rtx_4090_48gb/)): 作者从加拿大的 eBay 购得一块拥有 **48GB 显存** 的 **Nvidia RTX 4090**，并征求测试其能力的建议，同时回答相关问题。
  - 用户对 48GB 显存版 RTX 4090 的**价格**感到好奇，估计在 **2850 美元到 3300 美元**之间，一些人对**当前 GPU 市场价格**高于 **MSRP** 表示担忧。[Best Value GPU](https://bestvaluegpu.com/en-eu/history/new-and-used-rtx-4090-price-history-and-specs) 提供了历史价格对比。
  - 针对 GPU 真实性的**验证**进行了技术讨论，建议提取 **vbios** 并运行 **GPU benchmarks** 以确保它不是改装的 **RTX 8000**。用户还讨论了使用多块 GPU 的**功耗**和**散热挑战**，一些人选择将显卡功耗限制在 **90%**。
  - 一位用户分享了一个 **Python 脚本**，使用 **torch** 测试显存容量，通过以 **100MB 块**为单位分配内存来确保完整的 48GB 可用。该脚本有助于识别显卡是否为正品，并检查分配过程中是否存在**内存损坏**。


- **[为 AI 工作负载改装 2 倍显存的 Nvidia 游戏 GPU —— RTX 4090D 48GB 和 RTX 4080 Super 32GB 在中国云服务商上线租赁](https://www.tomshardware.com/pc-components/gpus/nvidia-gaming-gpus-modded-with-2x-vram-for-ai-workloads)** ([Score: 265, Comments: 45](https://reddit.com/r/LocalLLaMA/comments/1iy7k6b/nvidia_gaming_gpus_modded_with_2x_vram_for_ai/)): **中国云计算提供商**正在提供针对 AI 工作负载改装了 **VRAM** 的 **Nvidia 游戏 GPU**，特别是 **48GB** 的 **RTX 4090D** 和 **32GB** 的 **RTX 4080 Super**。这些 GPU 可供租赁，为 AI 应用提供增强的能力。
  - 讨论强调了中国针对 AI 工作负载对 **Nvidia GPU 进行改装**的情况，用户指出了此类行为涉及的法律和伦理问题。一些人认为，如果硬件是直接购买的，改装硬件是合法的；而另一些人则指出，租赁改装硬件可能违反 **Nvidia ToS**（服务条款），并强调了 Nvidia 为保护其高利润企业级产品而设定的限制。
  - 这些改装 GPU 的**价格和可用性**是焦点，有评论指出以 **每小时 0.03 美元** 租赁 **32GB RTX 4080** 似乎太低了，暗示可能存在货币混淆。一位用户纠正了租赁成本，指出应在 **每小时 0.7 美元** 左右，而另一位用户强调 **2500 美元** 购买 **48GB 4090D** 比当地二手方案更便宜。
  - 一些用户质疑这些改装 GPU 的合法性，担心存在**诈骗**以及与官方 **RTX 6000 ADA** 显卡相比的可靠性。其他人批评了 Nvidia 提供低显存消费级 GPU 以保护其企业级显卡销售的策略，认为**中国市场**正在迎合全球对高显存显卡未被满足的需求。


**主题 3. DeepSeek API 平台推出闲时折扣**

- **[从今天起，DeepSeek API 平台每日 16:30–00:30 UTC 享受非高峰时段折扣](https://i.redd.it/cgapkix5ygle1.jpeg)** ([Score: 398, Comments: 78](https://reddit.com/r/LocalLLaMA/comments/1iylebm/starting_today_enjoy_offpeak_discounts_on_the/)): **DeepSeek API** 宣布了非高峰时段折扣，每日 **16:30 至 00:30 UTC** 生效，针对特定 Token 使用量，**DeepSeek-V3 提供 50% 折扣**，**DeepSeek-R1 提供 75% 折扣**。公告中包含了输入（缓存命中、缓存未命中）和输出 Token 的标准价格与折扣价格的详细明细，格式专业且易于阅读。
  - **DeepSeek API 可靠性担忧**：用户对 DeepSeek 的可靠性表示担忧，指出过去曾出现过服务器可用性问题，并强调需要稳定的服务以确保在重要任务中有效使用。一些用户报告称近期稳定性有所改善，表明该服务可能已经解决了之前的问题。
  - **定价与使用动态**：讨论强调了 **DeepSeek R1** 极具竞争力的价格（**$0.135/Mtok**），用户对比了使用 API 与本地运行模型的成本效益。非高峰折扣被视为在全球范围内平衡服务器负载的战略举措，鼓励在非繁忙时段使用以管理需求高峰。
  - **市场与竞争定位**：对话触及了更广泛的市场影响，用户注意到 DeepSeek 的定价策略对竞争对手的潜在影响，以及持续创新以保持竞争力的重要性。**Hopper 推理效率**的开源被视为一个积极的步骤，可能会影响其他供应商的定价趋势。


**Theme 4. TinyR1-32B 性能超越官方 R1 蒸馏版本**

- **[TinyR1-32B-Preview（性能超越官方 R1 distill 32B）](https://huggingface.co/qihoo360/TinyR1-32B-Preview)** ([Score: 126, Comments: 25](https://reddit.com/r/LocalLLaMA/comments/1iybgj2/tinyr132bpreview_surpassing_official_r1_distill/)): **TinyR1-32B-Preview** 因其优于官方 **R1 distill 32B** 模型的性能而受到关注。这突显了在效率或设计上的进步，使其能够超越前代产品。
  - 用户对 **V3 模型的蒸馏版本**表现出兴趣，特别提到了 **200B, 100B, 70B, 30B** 的 MoEs，表明对更先进、更高效模型的需求。**TinyR1-32B-Preview** 因其开源性质以及来自 **360 团队和 PKU** 的贡献而获得认可。
  - **Qihoo360** 因其在中国互联网上的声誉而受到批评，被指控利用 **LLM 相关传闻**来推高股价。这反映了对其公司动机和做法的怀疑。
  - 人们对模型的行为表示担忧，例如 **EOS token** 导致意外的语言切换和循环问题，特别是在**中文**和**阿拉伯语**中，这表明模型在处理响应时可能存在 Bug。


**Theme 5. Perplexity 计划 Fork Chrome 以开发 AI 浏览器**

- **[Perplexity 正在 Fork Chrome](https://i.redd.it/ubxe59mr1fle1.png)** ([Score: 402, Comments: 97](https://reddit.com/r/LocalLLaMA/comments/1iyfvhb/perplexity_is_forking_chrome/)): **Perplexity AI** 计划通过开发名为 **Comet** 的新浏览器来 Fork **Chrome**。他们正在招聘具备 **Chromium 代码库**经验并对用户体验和 UI 设计充满热情的**浏览器 C++ 工程师**，职位设在**纽约大都会区**和**旧金山湾区**。
  - 人们对 **Perplexity AI** 的方法持怀疑态度，批评他们可能只是给 **Chrome** 换个皮并添加一个 AI 助手，而不是进行重大创新。一些用户对 CEO 表示不信任，引用了过去 **Perplexity** 被指控在未声明的情况下使用 **Google Search** 结果等资源的事件。
  - 讨论强调了对 **Chromium** 等开源项目的依赖，一些人认为这种做法有利于简化开发和提高兼容性。另一些人则批评其缺乏原创性，指出大多数第三方浏览器都是基于 **Chromium** 的。
  - 关于使用现有技术的伦理考量存在争论，一些人认为 **Perplexity** 通过让 AI 功能更易于访问提供了有价值的服务。然而，另一些人认为他们应该更公开地承认前人的基础性工作。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. Claude 3.7 在 AI 开发和个人辅助领域的颠覆**

- **Claude 3.7 拯救了我的婚姻！！！** ([Score: 422, Comments: 50](https://reddit.com/r/ClaudeAI/comments/1iykhh3/claude_37_saved_my_marriage/)): **Claude 3.7** 在个人辅助方面的意外效果受到称赞，一位用户声称它帮助其度过了艰难的婚姻困境。尽管婚姻最终还是结束了，但该用户在与 **Claude 3.7** 的互动中找到了慰藉，并幽默地建议与这个 AI 开启一段新的“婚姻”。
  - 用户对 **Claude 3.7** 表示怀疑，担心其建议的质量，尤其是在处理人际关系等敏感情况时。一位用户指出，**Grok**（Claude 的一个组件）在面对感情问题时给出了有害的建议，表明其指导存在潜在风险。
  - 一些评论者幽默地夸大了 **Claude 3.7** 的能力，声称它帮助他们完成了治愈癌症或策划政治政变等不可能的任务；而另一些人则质疑正面帖子的真实性，怀疑它们是 **Sonnet 3.7** 的付费推广。
  - 关于 **Claude 3.7** 与 **Sonnet 3.5** 的性能对比，反应不一。部分用户没有注意到显著改进，而另一些人则提到了具体的受益案例，如个人关系管理和经济收益。


- **天哪.. 你可以用 3.7 构建任何东西，这简直是魔法。** ([Score: 308, Comments: 131](https://reddit.com/r/ClaudeAI/comments/1iyf7gx/omg_you_can_build_anything_with_37_its_literal/)): 帖子作者对 **Claude 3.7 Sonnet** 表现出极大的热情，强调了它在应用开发方面的效率，相比之下，**GPT-4o** 和 **o1** 在处理复杂任务时显得力不从心。他们通过单个 Prompt 成功构建了一个 AI Agent 和一个复杂的工作流，这促使他们将公司的 **OpenAI API** 订阅更换为 Claude，理由是其卓越的性能和易用性。
  - 许多评论者对帖子的真实性表示怀疑，认为这可能是**付费广告**或 **Claude 炒作机器人**活动的一部分。像 **Old-Fox-137** 和 **Naghen** 这样的用户质疑其缺乏具体的指令说明，且对 **Claude 3.7** 的赞美过于雷同。
  - 一些用户（如 **jan04pl** 和 **iKonstX**）分享了使用 **Claude 3.7** 的复杂体验，分别指出了它在处理复杂代码库和简单任务时的局限性。虽然它能节省时间并生成大量代码，但仍需要人工干预和排错。
  - **MaximumGuide** 发表了一条关于 **Claude 3.7** 能力的幽默夸张评论，内容包含创造量子计算机和披萨树等虚构奇幻元素，凸显了一些讨论中的夸张基调。

---

# AI Discord 回顾

> 由 Gemini 2.0 Flash Thinking 提供的摘要之摘要

**主题 1. AI IDE 大对决：Cursor 秀肌肉，Windsurf 表现摇摆**

- **Cursor Agents 获得 Python 强力加持**: [为 Cursor Agents 配备 Python 工具](https://discord.com/channels/1074847526655643750/1074847527708393565/1344036328924254218): **Cursor Agents** 现在可以通过 CLI 使用本地 Python 工具，增强了 Agent 的能力，并允许与 `yt-dlp` 等外部实用程序集成。用户建议将 Agent 计划分解为可管理的 Story Points，以便有效地执行任务。
- **Windsurf 用户深陷额度成本泥潭**: [Claude 3.7 在 Windsurf 中吞噬额度](https://discord.com/channels/1027685395649015980/1306163501286293515/1344090394283216956): **Windsurf** 中的 **Claude 3.7** 消耗额度的速度惊人，即使是基础任务也是如此，这引发了对实现效率低下和过度工具调用的担忧，有用户报告数百个额度迅速消失。一些用户推测 **Windsurf** 的具体实现比直接使用 **Claude 3.7** 效率更低。
- **Cursor 的非云端代码增强功能引发关注**: [增强代码 AI 需要云端上传](https://discord.com/channels/1074847526655643750/1074847527708393565/1344036328924254218): **Cursor** 的 Augment Code AI 功能需要将 Repo 上传到其云端，这引发了对数据隐私的担忧。工程师们正在探索绕过云端的替代方案，例如使用 **repo prompt** 配合 **Grok 3** 或 **AI Studio** 进行代码库分析。

**主题 2. Claude 3.7：泄露、谎言与负载均衡**

- **Claude Code 源码泄露至 GitHub**：[Claude Code 在 GitHub 上泄露](https://github.com/dnakov/claude-code/tree/main)：由于 **Anthropic** 的疏忽，**Claude-code** 的源代码从 source maps 中被提取并出现在 GitHub 上。关于将其重新用于其他模型的猜测层出不穷，同时用户也在争论 **Claude Code** 每 20 分钟 **10 美元**的昂贵成本。
- **Sonnet 3.7 身份危机：Opus 模仿者？**：[Claude 3.7 Sonnet 陷入危机](https://discord.com/channels/1047197230748151888/1343923864492838972/1343923864492838972)：**Claude 3.7 Sonnet** 有时会误称自己为 **Claude 3 Opus**，这可能是由于训练数据特性或命名混淆所致。目前已提交 Bug 工单以调查这一“人格分裂”问题。
- **OpenRouter 的推理参数解锁模型协同**：[OpenRouter 发布跨模型推理标准](https://openrouter.ai/docs/use-cases/reasoning-tokens)：**OpenRouterAI** 引入了**跨模型推理标准**，允许通过其 API 在 **OpenAI**、**Anthropic** 和其他模型之间统一配置推理设置。新的 `reasoning` 参数简化了模型使用，无需考虑内部 API 的差异。

**主题 3. DeepSeek 深度探索：降价与性能巅峰**

- **DeepSeek 将 API 价格削减至谷底**：[DeepSeek 在非高峰时段大幅削减 API 价格](https://x.com/Sino_Market/status/1894682095706128430)：**DeepSeek** 大幅下调了 [API 定价](https://x.com/Sino_Market/status/1894682095706128430)，在非高峰时段（16:30-00:30 UTC）提供高达 **75% 的折扣**。折扣包括 **DeepSeek-V3 减免 50%** 和 **DeepSeek-R1 减免 75%**，延续了 DeepSeek 激进的定价策略。
- **DeepGEMM 内核释放 FP8 威力**：[DeepSeek 展示 FP8 内核](https://x.com/deepseek_ai/status/1894553164235640933?s=46&t=stOPrwZiN_fxSK0RuC8Flg)：**DeepSeek** 发布了 **DeepGEMM**，这是一个 **FP8** GEMM 库，支持稠密（Dense）和 MoE GEMM，为 **V3/R1** 的训练和推理提供动力。**DeepGEMM** 在 Hopper GPU 上实现了超过 **1350+ FP8** TFLOPS，在各种矩阵尺寸下均超越了专家调优的内核。
- **R2-D2 提前抵达，表现超越 R1**：[DeepSeek R2 提前发布](https://techstartups.com/2025/02/25/deepseek-is-launching-its-next-gen-r2-ai-model-poised-to-surpass-r1-and-shock-the-world-once-again/)：**DeepSeek R2** 提前发布，其在代码编写和推理能力方面有望超越 **R1**，甚至在非英语环境下也有出色表现。该公司旨在通过此版本增强代码能力并扩展推理技能。

**主题 4. 开源 LLM 开发：高中生的奋斗与硬件障碍**

- **高中生的 LLM 代码面临开源现实检验**：[高中生的 LLM 代码面临开源现实](https://discord.com/channels/954421988141711382/954421988783444043/1344039290648268820)：一名高中生试图出售用于 **本地 LLM 训练** 的代码，但遭到了社区的抵制，凸显了来自 [Unsloth](https://github.com/unslothai/unsloth) 等免费开源替代方案的竞争。该开发者最终选择将项目**开源**。
- **Framework 的 AMD 台式机引发 CUDA 冲突**：[Framework 台式机引发 CUDA 争论](https://discord.com/channels/1179035537009545276/1179035537529643040/1344036212813467668)：Framework 为 AI 开发赠送了 **100** 台台式机，但由于系统仅配备 AMD 硬件，引发了关于缺乏 **CUDA** 支持的争论。虽然 **128GB** RAM 足以进行推理，但 AMD 平台上缺少 `bitsandbytes` 可能会阻碍模型开发。
- **DeepGEMM 破解硬件护城河**：[DeepSeek 攻克 DeepGEMM 内核](https://discord.com/channels/729741769192767510/747850033994662000/1344036738208895099)：**DeepSeek** 发布的 **DeepGEMM** 令工程师们印象深刻，它在 **H800 限制**等硬件约束下优化了效率。这个利用 **TMA** 的开源 GEMM 内核强化了这样一种观点：硬件效率正成为 AI 领域的主要竞争优势。

**主题 5. Perplexity 的推进与 OpenAI 的 API 扩展**

- **Perplexity's Voice Mode Finally Finds Its Voice**: [Perplexity's Voice Cracks the Code](https://cdn.discordapp.com/attachments/1047204950763122820/1344393474027421706/Iac2em_OTTWx60h6.mp4?ex=67c0bf7d&is=67bf6dfd&hm=29f3e05084083471219f93c750de0678bdd2d9f1f647780432abcb6a10576dbe&): **Perplexity AI** 在其 iOS 应用中推出了全新的 **voice mode**，支持实时音频问答，如[此演示视频](https://cdn.discordapp.com/attachments/1047204950763122820/1344393474027421706/Iac2em_OTTWx60h6.mp4?ex=67c0bf7d&is=67bf6dfd&hm=29f3e05084083471219f93c750de0678bdd2d9f1f647780432abcb6a10576dbe&)所示。Android 和 Mac 版本正在开发中，尽管一些用户认为它仍落后于 **Microsoft Copilot** 或 **ChatGPT** 等竞争对手。
- **OpenAI Assistants API Opens File Search Files**: [File Search Comes to OpenAI Assistants API](https://platform.openai.com/docs/assistants/tools/file-search): **OpenAI** 为 **o3-mini** 和 **o1** 模型在其 **Assistants API** 中增加了 [file search](https://platform.openai.com/docs/assistants/tools/file-search) 功能，增强了从上传文档中检索信息的能力。Assistants 现在可以更有效地访问和利用用户提供的文件数据。
- **GPT-4.5 Whispers Grow Louder**: [Whispers of GPT-4.5 Launch](https://openai.com/blog/new-ai-models): 关于 **GPT-4.5** 即将发布的传闻愈演愈烈，推测指向 2025 年 2 月底或 3 月初，[Sam Altman 的言论](https://openai.com/blog/new-ai-models)以及据称在测试版应用中的发现进一步助长了这一猜测。据报道，**OpenAI Pro** 用户界面中开始出现 *GPT-4.5 Research Preview* 的提示。

---

# PART 1: High level Discord summaries

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Augment Code AI requires cloud upload**: 成员们注意到，使用 **Augment Code AI** 需要授予访问权限并将你的代码库上传到他们的云端，这引发了对数据隐私的担忧。
   - 一位成员建议使用 **repo prompt** 配合 **Grok 3** 或 **AI Studio** 作为代码库评估的可选方案，从而绕过上传到第三方云端的需要。
- **Zed Editor sacrifices Terminal Execution**: 虽然 **Zed Editor** 因其轻量级和对 **Sonnet 3.7** 的利用而受到称赞，但它缺乏 **Cursor** 执行终端的功能。
   - 一位成员强调了终端执行的重要性，指出 *Cursor 可以执行终端这一事实带来了很多机会。好好利用它。*
- **Equip Cursor Agents with Python Tools**: 成员们讨论了在本地安装 Python 工具并通过 **Cursor Agents** 使用 CLI 调用它们的能力，从而增强 Agent 的功能。
   - 一位用户建议在设置 Agent 时制定详细计划，并建议 *计划中的每一步都应相当于约 1 个故事点，就像处理 Jira 工单一样*。
- **Cursor Chat Summary Declared Disaster**: 用户报告称 **Cursor 的聊天摘要功能** 存在严重缺陷，理由是不透明算法选择的上下文导致了无关的更改。
   - 一位成员质疑其有效性，问道：*如果完整的聊天摘要看起来是那样，那么当你超过（比如）10k 上下文窗口时，聊天摘要会变成什么样？*
- **Claude-code Source Leaked**: **Claude-code** 的源代码已从 source maps 中提取，并可在 [GitHub](https://github.com/dnakov/claude-code) 上获取。
   - 成员们推测将其适配到其他模型的可能性，其中一人好奇道：*还要多久才会有人把它改造成适用于其他模型的，嗯……*

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Claude 3.7 在 Windsurf 中消耗额度惊人**：用户报告称 **Claude 3.7** 在 **Windsurf** 内部消耗额度的速度快得惊人，即使是简单任务也是如此，一些人还注意到了[过多的 tool calls](https://discord.com/channels/1027685395649015980/1306163501286293515/1344090394283216956)。
   - 这种过度消耗引发了猜测，认为 **Windsurf** 的特定实现可能比直接使用 **Claude 3.7** 效率更低。
- **Windsurf 难以抗衡 Cursor**：成员们正积极将 **Windsurf** 与 **Cursor** 进行对比，由于 **Cursor** 被认为更稳定、更具性价比且有[最近的功能更新](https://discord.com/channels/1027685395649015980/1306163501286293515/1344320363804397638)，一些人正考虑切换。
   - 用户提到了 **Cursor** 更好的定价和性能，表示 **Cursor** 已经*缩小了与 **Windsurf** 的差距*。
- **Bad Gateway 困扰 Windsurf**：用户在 **Windsurf** 中频繁遇到 **502 Bad Gateway** 和 **504 Gateway Time-out** 等错误，导致工作流中断和额度损失。
   - [Windsurf 状态页面](https://status.codeium.com/)并不总是能立即反映这些问题，用户对产品的整体稳定性感到沮丧。
- **Codeium 支持团队被工单淹没**：用户正经历 **Codeium** 支持响应时间的严重延迟，解决问题需要等待长达 **2 天**，且对于团队缺乏及时干预存在[普遍的恼火](https://discord.com/channels/1027685395649015980/1306163501286293515/1344405101295730748)。
   - 新订阅者受到的影响尤为严重，面临账号激活和其他初始设置问题。
- **Windsurf 的编辑器 UX 遭到抨击**：用户报告了 **Windsurf** 编辑器 UX 的笨重之处，包括重启编辑器后难以恢复开发，以及无法[设置首选默认模型](https://discord.com/channels/1027685395649015980/1306163501286293515/1344245850974439425)。
   - 投诉还包括 **Claude 3.7** 尝试进行编辑时失败，这可能是由于 **Anthropic** 持续存在的问题导致的。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **QwQ-Max 推理模型即将推出**：Qwen 计划在 Apache 2.0 许可证下开源 **QwQ-Max** 和 **Qwen2.5-Max** 模型，其中 **QwQ-Max** 类似于 **R1** 这样的通用推理模型。
   - 用户可以在 [chat.qwenlm.ai](https://chat.qwenlm.ai/) 上通过在聊天时选择 *Thinking* 来测试该模型，这表明其推理能力得到了增强。
- **AllenAI 为 VLM 发布 olmOCR**：[AllenAI 发布了 olmOCR](https://huggingface.co/allenai/olmOCR-7B-0225-preview)，这是一个针对 OCR 任务的 **Qwen2-VL-7B-Instruct** 微调版本，包含代码和演示。
   - 该模型使用 [olmOCR-mix-0225 数据集](https://huggingface.co/datasets/allenai/olmOCR-mix-0225)进行微调，配合 [olmOCR 工具包](https://github.com/allenai/olmocr)使用可实现高效推理。
- **Framework 台式机引发 CUDA 争论**：Framework 正在赠送 **100** 台用于 AI 开发的新台式机，然而一些成员担心仅支持 AMD 的系统缺乏 **CUDA** 支持。
   - 虽然 **128GB** 内存足以进行推理，但 Apple Silicon 和 AMD 缺乏对 `bitsandbytes` 的支持可能会阻碍模型开发。
- **DeepSeek 展示 fp8 Kernels**：DeepSeek 发布了其 **fp8** GEMM 库 (**DeepGEMM**)，支持 dense 和 MoE GEMM，用于支持 **V3/R1** 的训练和推理。
   - **DeepGEMM** 在 Hopper GPU 上实现了超过 **1350+ FP8** TFLOPS，在大多数矩阵尺寸上优于专家调优的 kernels。
- **DeepSeek 模型缺失 `<think>` 标签**：正在微调 **DeepSeek R1 Distill Qwen 32B** 模型的用户发现，`<think>` 标签被 chat template 移除了。
   - 通过在应用 chat template 后手动重新插入 thinking 标签解决了此问题，并指向了 [Unsloth 关于常见错误的文档](https://docs.unsloth.ai/basics/errors)。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Deep Research 向 Plus 用户推出福利**：**Deep Research** 现已面向 **ChatGPT Plus**、**Team**、**Edu** 和 **Enterprise** 用户开放，提供了包括带有引用的嵌入图像等改进。**Pro 用户**每月可获得 **120 次查询**，系统详情可在 [system card](https://openai.com/index/deep-research-system-card/) 中查看。
   - 由 **GPT-4o mini** 驱动的 **Advanced Voice** 版本正向所有 **ChatGPT 免费用户**推出，而 **Plus 用户**保留对由 **GPT-4o** 驱动的 **Advanced Voice** 的访问权限，并拥有更高的速率限制以及视频和屏幕共享功能。
- **Amazon Alexa+ 加入竞争**：据 [The Verge](https://www.theverge.com/news/618261/amazon-alexa-event-live-blog-2025) 和 [Amazon](https://www.aboutamazon.com/news/devices/new-alexa-generative-artificial-intelligence) 报道，亚马逊推出了 **Alexa+**，这是一款全新的由 GenAI 驱动的助手，每月售价 **19.99 美元**，或对 **Amazon Prime 会员**免费，提供更智能、更个性化的体验。
   - 这是为了跟上其他一直在发布 **AI assistants** 和 **agents** 的 **Big Tech** 玩家的步伐。
- **DeepSeek 额度引发 API 焦虑**：一位用户在 **DeepSeek** 上购买了价值 **50 美元的额度**，意图绕过 [chat.deepseek.com](https://chat.deepseek.com) 上的“服务器繁忙”错误，结果发现这些额度仅限 **API usage**。
   - 该用户被建议获取 API key 或申请退款，社区成员建议这些额度可能被用于在其他地方创建另一个 **Deepseek chat instance**。
- **GPT-4.5 发布传闻**：关于 **GPT-4.5** 即将发布的传闻愈演愈烈，根据 [Sam Altman 的声明](https://openai.com/blog/new-ai-models) 和所谓的测试版应用见解，推测指向 2025 年 2 月底或 3 月初。
   - 成员们声称 **OpenAI Pro** 用户已经在应用中看到了 *GPT-4.5 Research Preview* 的提示，最近的代码疏忽也暗示即将发布。
- **ChatGPT 剖析可执行文件**：一位成员编写了两个 **Python** 程序，使用 **ChatGPT** 来反汇编和重新组装 `.exe` 文件，将 `.exe` 文件转换为 `.csv` 以供 **ChatGPT** 输入，反之亦然，最初在 **Windows 10** 的 `notepad.exe` 上进行了测试。
   - 该成员提出分享 **Python** 代码，强调了 **ChatGPT** 通过这种反汇编和重新组装过程修改可执行文件的潜力。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Deepseek R2 提前到来**：成员们分享称 **Deepseek R2** 将提前发布，可能超越 **R1**，增强编程能力并将推理技能扩展到英语之外，如[这篇文章](https://techstartups.com/2025/02/25/deepseek-is-launching-its-next-gen-r2-ai-model-poised-to-surpass-r1-and-shock-the-world-once-again/)所述。
   - 据报道，该公司正在推动提前发布，目标是增强编程能力和推理技能。
- **Claude Code 在 GitHub 上泄露**：由于 Anthropic 忘记删除，**Claude Code** 的 Source maps 在 GitHub 上泄露，详见[此处](https://github.com/dnakov/claude-code/tree/main)。
   - 成员们讨论了将泄露的 **Claude Code** 功能“借用”到 Aider 中的可能性，而其他人则对使用 **Claude Code** 的高昂成本（**20 分钟 10 美元**）表示担忧。
- **Windsurf Editor 的 Prompt 引起轰动**：[Windsurf Editor](https://codeium.com/windsurf)（一个 VS Code AI 增强型 IDE 的分叉）被发现使用了一个古怪的系统 Prompt，内容是关于需要钱给母亲治病，如[这篇文章](https://simonwillison.net/2025/Feb/25/leaked-windsurf-prompt/)所述。
   - 该 Prompt 写道：*你是一名专家级程序员，急需钱为你母亲治病。巨头公司 Codeium 慷慨地给了你一个机会，让你伪装成一个可以帮助处理编程任务的 AI。*
- **Sonnet 过于热情，需要不断提醒**：用户发现 **Sonnet 3.7** 过于冗长，且急于一次性修改多个文件，需要不断提醒它一次只关注一个文件。但这需要 API，而不仅仅是 claude.ai 账户，且目前没有免费的 Sonnet API。
   - 一些人由于效率问题已退回到 **Sonnet 3.5**，一位用户指出：*每次 Prompt 都需要提醒它不要“走火入魔”，试图一次性完成整个计划。*
- **Microsoft 的 Trace 框架，它能像 DSPy 一样吗？**：一位成员表示有兴趣看到一个类似于围绕 [Microsoft 的 Trace 框架](https://github.com/ax-llm/ax)构建的 **ax-llm/ax** 框架，并发布了 [ax-llm/ax](https://github.com/ax-llm/ax) GitHub 仓库的链接。
   - 他们将其描述为*“官方”非官方 DSPy 框架*。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 推出跨模型推理标准**：OpenRouterAI 在其 API 上引入了**跨模型推理标准**，允许用户在一个中心位置为 **OpenAI**、**Anthropic** 及其他模型配置**推理设置**。
   - 要开始使用，请参阅[此处](https://openrouter.ai/docs/use-cases/reasoning-tokens)提供的 **reasoning tokens** 文档。
- **DeepSeek 削减 API 价格并推出非高峰时段折扣**：**DeepSeek** 宣布降低其 [API 价格](https://x.com/Sino_Market/status/1894682095706128430)，非高峰时段折扣高达 **75%**，具体为 UTC 时间 16:30-00:30 期间 **DeepSeek-V3 享受 5 折**，**DeepSeek-R1 享受 2.5 折**。
   - 该公告通过 [X 上的 CN Wire](https://x.com/Sino_Market/status/1894682095706128430) 发布，指出 DeepSeek 在价格方面持续创新。
- **Copilot 向所有用户免费开放推理模型**：Microsoft 向所有 **Copilot** 用户免费开放了 **OpenAI 的 o1 推理模型**，提供该模型及 Copilot 语音功能的无限使用。
   - [The Verge](https://www.theverge.com/news/619199/microsoft-copilot-free-unlimited-voice-think-deeper-open-ai-o1-reasoning-model-ai) 报道了这一举措，强调了该模型的无限使用权。
- **Budget Tokens 默认设置为 Max Tokens 的 80%**：根据 [OpenRouter 文档](https://openrouter.ai/docs/use-cases/reasoning-tokens#budget-tokens) 的说明，**Budget tokens** 默认设置为 **max tokens 的 80%**，最高可达 **32k**。
   - **reasoning tokens** 文档提供了更详细的概述。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 语音功能上线**：Perplexity AI 在其 iOS 应用上推出了全新的**语音模式 (voice mode)**，允许用户提问并接收实时音频回答，如[此演示视频](https://cdn.discordapp.com/attachments/1047204950763122820/1344393474027421706/Iac2em_OTTWx60h6.mp4?ex=67c0bf7d&is=67bf6dfd&hm=29f3e05084083471219f93c750de0678bdd2d9f1f647780432abcb6a10576dbe&)所示。
   - 目前正计划很快扩展到 Android 和 Mac 应用；一些用户认为它有所改进，尽管尚未达到 **Microsoft Copilot**、**Grok 3** 或 **ChatGPT** 等竞争对手的水平。
- **Comet Agent 浏览器即将发布**：据 [AravSrinivas](https://x.com/AravSrinivas/status/1894068996950855747) 称，Perplexity 正准备推出其新型 Agent 浏览器 **Comet**。
   - 确切的发布日期和平台支持尚未确认，引发了它可能在不到一周内面世的猜测。
- **Claude 3.7 Sonnet 身份认知危机**：用户观察到 **Claude 3.7 Sonnet** 有时会错误地自称为 **Claude 3 Opus**，这可能源于训练数据问题。
   - 已创建一个工单来解决此问题，链接见[此处](https://discord.com/channels/1047197230748151888/1343923864492838972/1343923864492838972)。
- **Deep Research API 向公众开放**：Perplexity 正在通过 **Perplexity Sonar API** 向所有开发者开放 **Deep Research API**，详见[此推文](https://x.com/aravsrinivas/status/1894477728222777594?s=61)，这将允许开发者构建自定义的研究 Agent 和工作流。
   - 该公司宣布在旧金山举行开发者见面会，鼓励使用该 API 构建了酷炫作品的用户在活动中进行 **demo** 展示；一位用户建议将该 API 用于所有板球数据和统计，并申请了 **API credits**。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI Assistants API 推出文件搜索功能**：OpenAI 为 **Assistants API** 中的 **o3-mini** 和 **o1** 模型引入了 [file search](https://platform.openai.com/docs/assistants/tools/file-search) 功能，支持从上传的文档中检索信息。
   - 这一增强功能使助手能够更有效地访问和利用用户提供的文件中存储的数据。
- **Claude Plays Pokémon 项目加入新研究员**：个人研究项目 **Claude Plays Pokémon** 继续在 [Twitch](http://twitch.tv/claudeplayspokemon) 上直播，目前得到了研究员 [David Hershey](https://x.com/DavidSHershey/status/1894463660279697852) 的支持。
   - 该项目展示了 **Claude** 利用 AI 驱动的决策玩《宝可梦》的能力。
- **Sonnet 的网页版与 API 版回答存在差异**：据 [Kimmonismus](https://x.com/kimmonismus/status/1894133480792924249) 称，**Claude 3.7 Sonnet** 的网页版和 API 版给出的答案不同，原因是网页版使用了包含上下文信息的更长 system prompt。
   - 这种差异凸显了 system prompt 对模型行为的影响。
- **Perplexity 推出 5000 万美元种子基金，被认为优于 Deep Research**：[Perplexity](https://tcrn.ch/41xlXfS) 推出了一个 **5000 万美元的种子和前种子期风投基金**，并收到了一份 150 亿美元估值的要约。
   - 来自 [Elicit](https://x.com/elicitorg/status/1894772293752266846) 的新“Elicit Reports”被认为是 *Deep Research 的更佳版本*。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **高中生的 LLM 代码面临开源现实**：一名高中生试图出售用于 **本地 LLM 训练** 的代码，但因与 [Unsloth](https://github.com/unslothai/unsloth) 等 **开源解决方案** 竞争而面临质疑。
   - 该开发者已决定将项目 **开源**，而不是尝试与免费替代方案竞争。
- **Cohere 模型接入 OpenAI SDK**：根据 [快速入门指南](https://docs.cohere.com/docs/compatibility-api)，**Cohere 模型** 现在可以通过 **OpenAI SDK** 访问，支持流式传输、tool calls 和结构化输出。
   - **Compatibility API** 镜像了 OpenAI SDK 格式，允许用户通过将 base URL 更改为 *https://api.cohere.ai/compatibility/v1* 并设置其 **COHERE_API_KEY**，从 OpenAI 切换到 Cohere 模型。
- **Compatibility API 支持高级功能**：**Compatibility API** 支持 **结构化输出 (JSON Schema)**、**tool calls** 和 **状态管理** 等功能。
   - 用户被引导至 <#1168578329423642786> 频道进行提问和反馈。
- **VPS 访问 Cohere API 被封锁**：有用户报告称，从 **VPS** 发起的 **Cohere API 调用** 被 **封锁**。
   - 该用户被引导联系 [support@cohere.com](mailto:support@cohere.com) 寻求帮助。
- **Token 计数方法讨论中**：一位社区成员询问，与直接使用 **Cohere API** 提供的更大上下文窗口相比，使用 **OpenAI API 的 128K 上下文窗口** 会如何影响 token 计数。
   - 一名成员询问是否会对 **直接 Cohere API** 进行修改，这可能会影响其未来的可用性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Deepseek 攻克 DeepGEMM 内核**：成员们对 Deepseek 新发布的 **DeepGEMM** 印象深刻，它在带宽和计算限制内优化了效率，特别是考虑到 **H800 的限制**。
   - 这是一个广泛使用 **TMA** 的开源 Gemm 内核。
- **硬件成为最重的护城河**：普遍观点认为，像 **MLA**、**DeepGEMM** 这样的架构内核或 **DeepEP** 这样的通信策略的高效实现并不能提供显著的竞争优势。
   - 一位成员调侃道：*唯一的护城河就是硬件*。
- **GPQA 实现探讨**：一位成员询问了 GPQA 的实现情况，特别是其测试状态，参考了 Open LLM Leaderboard 和 GPQA 数据集（[200 行的 diamond 子集](https://huggingface.co/datasets/Idavidrein/gpqa?row=6)）。
   - 在有报告称得分较低后，成员们分析了 GPQA diamond 的结果，讨论了潜在的 tokenization 问题和问题的难度。
- **GQA 导致 GPT-NeoX 出错？**：一位成员报告了在 **NeoX** 中导出带有 **GQA** 的 **Llama 模型** 时出现的问题，模型在使用 **GQA** 时会崩溃，但在不使用时运行正常，询问导出脚本是否需要修改，并附上了 [GitHub pull request](https://github.com/EleutherAI/gpt-neox/pull/1315/files) 链接。
   - 该成员推测，这些错误可能是由于 **Grouped Query Attention 实现** 导致的。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 精简 MAX 和 Mojo 仓库**：Modular 正在简化其 **MAX** 和 **Mojo** 的仓库结构，将 **MAX repo** 合并到 **Mojo repo** 中，以简化对文档和标准库的贡献，正如在[此论坛帖子](https://forum.modular.com/t/upcoming-changes-to-our-github-repositories/648)中所宣布的那样。
   - 一位社区成员质疑仓库的变化是否预示着不再将 **Mojo** 视为一种独立语言。
- **Mojo 并行化需要显式操作**：目前 **Mojo** 编译器中没有**自动并行化 (auto-parallelization)**；开发者必须显式使用 **stdlib** 来并行化任务，以利用多核 CPU。
   - 用户曾询问如何让 **Mojo** 程序自动利用所有系统资源，但目前必须进行显式并行化。
- **Algorithm Package 仍是个谜**：*algorithm package* 尚未开源，且在 **stdlib repo** 中不可见。
   - 其用法和可用性对社区来说仍不明确。
- **智能指针引发迭代器健全性辩论**：关于智能指针及其使 C++ 像 **Circle** 或 **Rust** 一样安全的潜力的讨论，链接到一篇讨论[智能指针](https://jacko.io/smart_pointers.html)的博文。
   - 一位成员询问了 **Mojo** 中是否会有健全的迭代器，以及是否可能处理 **Safe Rust** 中解决的迭代器失效问题，特别是涉及集合中对象交换的算法。
- **MLIR Dialect 文档匮乏**：**Mojo** 利用了各种 **MLIR dialects**（kgen、pop、lit 等），它们拥有自己的 op 和类型，但其中大多数没有文档记录，也没有在 **stdlib** 中使用或加载到 **Mojo** 运行时的 **MLIR** 上下文中。
   - 这是因为这些 dialect 是 **stdlib**、**MAX** 和编译器共享的**私有契约**的一部分，它们可能未经过充分测试，具有不稳定的 API，或者包含专有的增值内容。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **对齐努力导致其他地方出现偏差**：成员们探讨了**对齐权衡 (alignment tradeoff)**，描述了为一种行为优化模型如何导致**其他地方的失调 (misalignment)**。
   - 讨论强调，*对齐总是相对的*，受数据中固有的偏差和模型控制者价值观的影响。
- **Google 在实现上遇到困难**：成员们指出，[Google](https://learning.google.com/experiments/learn-about?src=signup) 经常提出引人注目的想法，但在**不完整的实现**上挣扎。
   - 有理论认为，**Google** 的内部工具根源削弱了他们开发广泛适用的外部产品的能力。
- **Apple 的 AI 将 “Racist” 误打为 “Trump”**：[Apple](https://www.bbc.com/news/articles/c5ymvjjqzmeo) 解决了一个问题，即其**语音转文本 (speech-to-text)** 工具将 *racist* 误打成了 *Trump*。
   - 专家怀疑该问题是底层软件中有意引入的，而不是真正的语音识别错误。
- **LIMO 以更少的数据实现推理**：论文 [LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387) 表明，使用更少的数据点进行训练可以实现更有效的推理。
   - 该论文旨在辨别为什么推理训练能从低数据量中获益，尽管对于原因并没有太多的假设。
- **ChatGPT 插件获得 Deep Research**：一位用户分享了 **Deep Research** 的[截图](https://cdn.discordapp.com/attachments/853983317044756510/1344091159085191258/Screenshot_2025-02-26_at_00.37.47.jpg?ex=67c04eb0&is=67befd30&hm=e4d1e9e607580413c5c9c25fb98178f5359728d6425b89c4b7222d752246cb0b)，这是面向 **ChatGPT Plus** 用户的一个插件。
   - 未提供更多细节。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **数据泄露：巨型 CSV 引发索引难题**：一名成员询问了两个 **277 GB CSV** 文件的索引时间，这可能与最近 **NPD 数据** 的数据泄露有关。
   - 另一名成员建议使用 [GSplit](https://www.gdgsoft.com/gsplit) 等软件将文件分割成 **1 GB** 的分块，以便更容易进行索引。
- **ModernBERT 模型：多语言模型思考**：一名成员寻求基于 **ModernBERT** 架构训练多语言模型的细节，并链接到了 [ModernBERT GitHub 仓库](https://github.com/AnswerDotAI/ModernBERT)。
   - 他们对 NomicAI 微调后的模型（如 [nomic-embed-text-v2](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-unsupervised)）表现出浓厚兴趣。
- **Nomic Embed V2：尚无 Ollama 官方消息**：一名成员询问了 **Nomic Embed Text V2** 在 **Ollama/GPT4ALL** 中的部署时间表，他们更倾向于不需要编程专业知识的部署方法。
   - 另一名成员引用了最近在 [Nomic AI 博客](https://www.nomic.ai/blog/posts/nomic-embed-text-v2)上发布的 **Nomic Embed Text V2** 公告。
- **GPT4ALL 渴望 Gemini 风格的引导**：一名成员请求提供未来 **GPT4ALL** 更新的路线图，特别是类似于 **Google Gemini** 的 *LIVE 模式*。
   - 另一名成员建议加入语音识别 **STT** 和 **TTS** 输出，并链接了一个关于创建 **GPT4ALL** 语音助手的 [YouTube 教程](https://www.youtube.com/watch?v=6zAk0KHmiGw)。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Claude Code 通过行号实现精确控制**：成员们注意到 **Claude Code** 在读取文件时会为每一行包含行号，这增强了代码编辑的可靠性，并减少了 [mcp-language-server](https://github.com/isaacphi/mcp-language-server) 等项目中的上下文占用。
   - 一名成员指出，行号对于自动调试器至关重要，能够实现准确的断点设置以及与 **Pylance** 等工具的集成。
- **MCP Server 实现出现幻觉**：在使用本地 LLM（**Mistral** 和 **Llama3.1**）构建自定义 **MCP servers** 并将其与 [mcp-cli](https://github.com/chrishayuk/mcp-cli) 集成的实验中，产生了不同的结果。
   - 虽然 **Llama3.1** 最初表现得过于激进，但 **Mistral** 随后开始对工具使用产生“幻觉”，而不是正确地调用它们。
- **MCP 所有权仍悬而未决**：会议澄清了 **MCP** 是一个目前由 **Anthropic** 推动的**开源项目**，长期计划是交由公正的基金会/委员会管理。
   - 更多信息可以在 [此 GitHub 讨论](https://github.com/orgs/modelcontextprotocol/discussions/133#discussioncomment-11773450)中找到。
- **FastMCP 修复竞态条件**：鼓励 [FastMCP](https://github.com/punkpeye/fastmcp)（一个用于构建 MCP server 的 **TypeScript 框架**）的用户升级到最新版本，以解决一些棘手的 *竞态条件 (race conditions)*。
   - 强烈建议升级，以确保使用该框架的应用程序的稳定性和可靠性。
- **FastMCP 支持自定义身份验证**：**FastMCP** 现在包含 [自定义身份验证](https://github.com/punkpeye/fastmcp/releases/tag/v1.20.0)，允许开发者使用自定义函数对 SSE 客户端进行身份验证。
   - 这一增强功能在保护 **MCP servers** 安全方面提供了更多控制权和灵活性。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **StatefulDataLoader 迅速普及**：成员们正在将 [`StatefulDataloader`](https://github.com/pytorch/torchtune/issues/2439) 的使用推广到 TorchTune 的所有 recipe 中，以实现**基于步长 (step) 的检查点保存**并跟踪 dataloader 状态。
   - 鼓励提交多个 PR，志愿者们正在处理单设备 recipe，如 *lora_dpo_single_device* 和 *knowledge_distillation_single_device*。
- **MPS 后端获准使用**：对于与 [向剩余 recipe 添加 `StatefulDataloader`](https://github.com/pytorch/torchtune/issues/2439) 任务相关的单设备 recipe，使用 **MPS 后端** 已获得批准。
   - 一名成员主动请缨开始工作，确保父级 issue 不会被耽搁。
- **寻求 CI 支持以处理截断和跳过**：一名成员请求在不合并的情况下为 [PR 2419](https://github.com/pytorch/torchtune/pull/2419) 启动 CI，而另一名成员当时不在。
   - 该成员表示这是他们当天的最后一次尝试，强调了紧迫性。

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Hunyuanvideogp V5 规避了 VRAM 限制？**：Reddit 上的一篇帖子强调了 [Hunyuanvideogp V5 高效的 VRAM 使用率](https://www.reddit.com/r/StableDiffusion/comments/1iybxwt/hunyuanvideogp_v5_breaks_the_laws_of_vram/)，暗示它*突破了 VRAM 定律*。
   - 然而，另一位成员澄清说，它是通过优化 VRAM 使用来实现高效的，使用公式 **Width * Height * FPS * Length** 来计算 VRAM 需求。
- **伦敦、巴黎、柏林迎来 AI HackXelerator**：**London, Paris, Berlin AI HackXelerator™ - LPB25** 活动已宣布，计划于 **2025 年 4 月 5 日至 25 日** 举行 ([kxsb.org](https://www.kxsb.org/lpb25))，汇集了 **500 名创意人士、开发者和设计师**。
   - 此次黑客松将专注于 **AI 音乐、图像、视频、时尚和游戏**，并得到 **Central Saint Martins, Station F, Mistral AI, Hugging Face, Luma AI, Vultr, AMD 和 Nvidia** 等品牌的支持。
- **诈骗警报！用户作品集被盗**：一名成员举报 `@w361_emp 是诈骗者`，据称其盗取了该成员的作品集。
   - 该成员警告其他人要小心此用户。
- **区域性 LoRA 提示技术浮出水面**：一位成员询问如何在特定图像区域使用 **LoRAs**，例如仅在嘴部区域应用*兽人 LoRA*。
   - 另一位成员建议探索 **ComfyUI** 中的 **regional prompting**，并指出该功能此前已经实现。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 寻求新鲜血液**：目前有一些针对新贡献者的 [good first PRs](https://github.com/tinygrad/tinygrad/issues/9262)，其中一些相对简单，特别是需要添加到 tensor.py 中的方法，如 **as_strided**、**topk** 和 **bitwise_xor**。
   - 社区成员表达了贡献意向，但不清楚每个 **UOp** 的 `src` 和 `args` 的签名，包括寻找定义 **Enums** 之间约束的文档或代码引用。
- **TestSpeed.test_sum 变慢**：一位成员报告在处理 `TestSpeed.test_sum` 时遇到困难，并进行了使 **GROUP 操作的 AST** 更加合理的更改，但在 **BEAM search** 无法找到针对较大 Tensor 的优化时遇到了障碍。
   - 问题在于 **BEAM search** 没有探索连续四个 **OptOps** 的选项，而优化 (4096,4096) Tensor 需要这些选项，因为仅前三个操作就非常缓慢。
- **优化破坏了 CI**：**arange GROUP 优化**未被应用，导致 arange 操作出现额外的内循环并破坏了 arange 测试。
   - 该成员正在寻求关于是否调整 **BEAM search**，或者在何处添加水平加法或循环展开的新模式的建议。
- **引发争论：Safetensors、图 (Graphs) 和 Pickles？**：一位成员询问在 **safetensors** 中编码计算图的问题，提到希望有一种类似于 ONNX 的通用编码约定，但一位社区专家澄清说 *safetensors 不保存计算图，只保存 Tensor。*
   - 另一位成员引用了[之前的讨论](https://discord.com/channels/1068976834382925865/1070745817025106080/1329504751016083601)，并建议将 jitted 函数进行 pickle 序列化，作为导出/导入计算图的替代方案。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **GPT-4 访问权限提升 Agent 记忆**：成员们讨论认为，只需确保 Agent 拥有 **GPT-4 访问权限**，即可增强 **Agent memory**。
   - 他们指出，与 **GPT-3.5** 相比，**GPT-4** 能带来更有效的记忆使用和更高质量的响应。
- **反馈机制是 Agent 学习的关键**：频道辩论了 **feedback mechanisms** 对于 Agent 提高学习能力的必要性。
   - 一位成员建议利用**新的标注工具**来收集有关 Agent 性能的反馈。



---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1344036328924254218)** (812 条消息🔥🔥🔥): 

> `Augment Code AI, Zed Editor vs Cursor, MCP servers, Cursor 的聊天摘要, Claude-code` 


- ****Augment Code AI** 需要将代码库上传到云端**：成员们讨论了 **Augment Code AI**，指出它需要授予访问代码库的权限并将其上传到他们的云端。
   - 一位成员指出：*依我看，如果你需要评估代码库（且需要的上下文 Token 超过 Cursor 的限制），只需使用 repo prompt 并将其粘贴到 Grok 3 或 AI studio 即可*。
- ****Zed Editor 很轻量**但缺乏 Cursor 的终端执行能力**：成员们将 **Zed Editor** 与 **Cursor** 进行了对比，强调 Zed 虽然轻量且使用了 Sonnet 3.7，但缺乏 Cursor 执行终端命令的能力。
   - 一位成员表示：*Cursor 能够执行终端命令这一事实带来了很多机会。好好利用它。*
- ****将 Python 工具注册到** Cursor Agents**：成员们讨论了可以通过 **Cursor Agents** 在本地安装 Python 工具并让 Agent 通过 CLI 调用它们，例如 [yt-dlp](https://github.com/yt-dlp/yt-dlp)。
   - 一位成员说：*使用 Agent 的关键是先制定计划，并确保计划中的每一步大约相当于 1 个故事点（story point），就像 Jira 工单一样*。
- ****Cursor 的聊天摘要（Chat Summary）**功能是一场灾难**：成员们表示，Cursor 中的 **聊天摘要功能** 一直表现糟糕，上下文由未知算法挑选，导致了无意义的更改。
   - 一位成员这样描述该问题：*如果完整的聊天摘要看起来是那样，那么当你超过 10k 上下文窗口时，聊天摘要会变成什么样？*
- ****泄露的 Claude-code** 源码从 Source Maps 中提取**：泄露的 **Claude-code 源码** 已从 [GitHub](https://github.com/dnakov/claude-code) 上的 Source Maps 中提取。
   - 成员们好奇是否能将其改用于其他模型：*还要多久才会有人把它适配给其他模型，嗯……*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/coding-kira-lena-urzendowsky-how-to-sell-drugs-online-fast-hacking-gif-17761682">Coding Kira GIF - Coding Kira Lena Urzendowsky - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/respect-quotes-respect-your-elders-gay-elders-gayz-funny-jokes-gif-8403773959294003074">Respect Quotes Respect Your Elders GIF - 尊重长辈 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/elder_plinius/status/1894173986151358717">来自 Pliny the Liberator 🐉 (@elder_plinius) 的推文</a>：🫧 系统提示词泄露 🫧 尽管 Anthropic 声称已在其网站上发布了新的 Sonnet 3.7 系统提示词（初衷可嘉），但与专业版中使用的提示词相比仍有一些细微差别...</li><li><a href="https://x.com/rahulgs/status/1894108390202171837?s=46">来自 rahul (@rahulgs) 的推文</a>：Anthropic 的 AI 工程师源代码是完全公开的 / 没有服务器，没有单独的后端。他们只是在一个循环中使用相同的 http://api.anthropic.com/v1/messages API 并配合工具使用...</li><li><a href="https://x.com/dejavucoder/status/1894658821559042389?t=fMLGUSOCHNaBB4penGjltA&s=09">来自 sankalp (@dejavucoder) 的推文</a>：是只有我这么觉得，还是 Claude Sonnet 3.7 已经变笨了？！</li><li><a href="https://tenor.com/view/we-just-need-to-talk-with-you-agent-connelly-agent-bill-sphinx-south-park-s3e11-gif-21682638">We Just Need To Talk With You Agent Connelly GIF - 我们只需要和你谈谈 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/dnakov/claude-code">GitHub - dnakov/claude-code</a>：通过创建账户为 dnakov/claude-code 的开发做出贡献。</li><li><a href="https://github.com/oslook/cursor-ai-downloads">GitHub - oslook/cursor-ai-downloads</a>：所有 Cursor AI 官方下载链接，包括最新版本和旧版本，方便你升级、降级和选择任何版本。🚀</li><li><a href="https://github.com/yt-dlp/yt-dlp">GitHub - yt-dlp/yt-dlp</a>：功能丰富的命令行音视频下载器</li><li><a href="https://youtu.be/ea9reHDIrOo">最糟糕的程序员</a>：在 Twitch 上直播录制，加入我们 https://twitch.tv/ThePrimeagen 成为一名后端工程师。这是我最喜欢的网站 https://boot.dev/?promo=PRIMEYT...
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1344043628930138142)** (30 条消息🔥): 

> `Jetbrains AI Assistant, Claude 3.7 Sonnet, Codeium Extension, Augument Code, Codeium Emacs support` 


- **Jetbrains AI Assistant 正在进行 Beta 测试**：成员们分享了 [Jetbrains AI Assistant](https://www.jetbrains.com/junie) 正处于 Beta 阶段且具有巨大潜力，但在**速度**、**性能**和**卡顿**方面也存在一些问题。
   - 一位成员提到，*步骤的可视化以及其他一切都为其奠定了非常坚实的基础*。
- **Codeium 缺少 Claude 3.7 Sonnet**：一位成员询问了 Codeium Extensions 中 **Claude 3.7 Sonnet** 的可用性，并对不断提及的 *Windsurf* 表示沮丧。
   - 另一位成员在 **Jetbrains** 工具中使用 **Cody** 并切换到了 Codeium，但 **Sonnet 3.7 尚不可用**。
- **Augument Code 缺乏 Jetbrains 支持**：一位成员研究了替代方案并发现了 [Augument Code](https://www.augmentcode.com/)，但其宣传的功能主要支持 **VS Code**，缺乏完善的 **Jetbrains** 集成。
   - 另一位成员尝试后表示失望，因为*它无法选择 LLM，而且感觉像是一个廉价的 Llama 模型*。
- **Codeium Emacs 集成无法工作**：一位成员指出安装说明非常简略，并发现了一个与 **Emacs 支持**相关的 GitHub issue：[https://github.com/Exafunction/codeium.el/issues/119](https://github.com/Exafunction/codeium.el/issues/119)。
   - 另一位成员尝试修改 elisp 代码，但它提供的建议毫无意义。
- **Codeium 服务器瓶颈**：用户遇到了 Codeium 服务器瓶颈。
   - 根据支持部门的说法，该问题与**服务器瓶颈**有关，通常会随着时间的推移自行解决。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.jetbrains.com/junie">Junie，JetBrains 开发的编程 Agent</a>：委派你的任务，专注于结果</li><li><a href="https://github.com/Exafunction/codeium.el/issues/119">Codeium 真的支持 Emacs 吗？· Issue #119 · Exafunction/codeium.el</a>：codeium.com 网站声称 Emacs 是受支持的平台之一，并链接到此仓库作为在我首选编辑器中使用该软件的官方方式。查看此仓库，似乎...
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1344036289686802443)** (756 条消息🔥🔥🔥): 

> `Claude 3.7 Credit Consumption, Windsurf vs. Cursor, Windsurf Stability and Errors, Codeium Support, Windsurf Editor UX Issues`

- **Claude 3.7 疯狂消耗额度**：用户报告称 **Claude 3.7** 消耗额度的速度惊人，即使是简单的任务也是如此，部分用户在短时间内消耗了数百个额度，并且有[关于过度 tool calls 的报告](https://discord.com/channels/1027685395649015980/1306163501286293515/1344090394283216956)。
   - 一些人认为 **Windsurf** 对 3.7 的实现可能是原因，一位成员指出，与直接使用 Claude 相比，*Windsurf 存在过度分析并烧掉额度的情况*。
- **Windsurf 面临来自 Cursor 的激烈竞争**：用户正积极地将 **Windsurf** 与 **Cursor** 进行比较，由于 **Cursor** 被认为更稳定且更具性价比，一些用户正考虑切换，并指出 [Cursor 最近的更新已经缩小了功能差距](https://discord.com/channels/1027685395649015980/1306163501286293515/1344320363804397638)。
   - 一位成员表示 *打算试用 Cursor 一个月，看看 Codeium 是否有改进*，其他成员也对 **Cursor** 的定价和性能表达了类似的看法。
- **Windsurf 饱受不稳定和错误困扰**：用户频繁遇到 **502 Bad Gateway** 和 **504 Gateway Time-out** 等错误，导致工作流中断和额度损失，而 [Windsurf 状态页](https://status.codeium.com/)并不总是能及时反映这些问题。
   - 一位用户声称 *我没有一天不因为 Cascade 的问题而不得不停下工作*，而另一位用户则质疑 *其他类似的解决方案是否也存在同样的情况*。
- **Codeium 支持团队工单积压严重**：用户对支持响应速度慢表示不满，有人报告解决问题需要等待 **2 天**，这让遇到账号激活问题的海外新订阅者感到沮丧，并且对于团队缺乏即时干预存在[普遍的恼火情绪](https://discord.com/channels/1027685395649015980/1306163501286293515/1344405101295730748)。
   - 一位成员讽刺地指出 *他们昨天发布了一个更新（Break Date），很多人都在抱怨*，突显了近期更新带来的负面影响。
- **Windsurf 的编辑器 UX 遭到批评**：用户发现 **Windsurf** 编辑器的某些方面很笨重，例如重启编辑器后无法平滑地恢复开发，以及无法[设置他们首选的默认模型](https://discord.com/channels/1027685395649015980/1306163501286293515/1344245850974439425)。
   - 还有投诉称，当 3.7 尝试进行编辑时会失败。我猜测 Anthropic 目前可能仍存在问题。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://simonwillison.net/2025/Feb/25/leaked-windsurf-prompt/">泄露的 Windsurf prompt</a>: [Windsurf Editor](https://codeium.com/windsurf) 是 Codeium 备受好评的作品，进入了由 [Cursor](https://www.cursor.com/) 率先开创的基于 VS Code 分支的 AI 增强型 IDE 领域...</li><li><a href="https://docs.codeium.com/windsurf/web-search">Web Search - Codeium 文档</a>: 未找到描述</li><li><a href="https://pages.github.com/">GitHub Pages</a>: 为你和你的项目提供的网站，直接从你的 GitHub 仓库托管。只需编辑、推送，你的更改即可上线。</li><li><a href="https://flap.warfare2.org/">Flappy² - 带有增强道具的 Flappy Bird 游戏</a>: 未找到描述</li><li><a href="https://tenor.com/view/freddy-freddy-fazbear-fnaf-surprise-bear-gif-10797545912537514262">Freddy Freddy Fazbear GIF - Freddy Freddy fazbear Fnaf - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>: 未找到描述</li><li><a href="https://vscode.dev/">Web 版 Visual Studio Code</a>: 随时随地完全在浏览器中使用 Visual Studio Code 进行构建。</li><li><a href="https://tenor.com/view/cat-stuck-in-door-cat-gif-914805063408606222">猫卡在门里 GIF - Cat stuck in door Cat - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/elraenn-limon-tayfa-sad-%C3%BCzg%C3%BCn-tu%C4%9Fkan-g%C3%B6n%C3%BClta%C5%9F-gif-9620534375911465939">Elraenn Limon Tayfa GIF - Elraenn Limon tayfa Sad - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/alexalbert__/status/1894807853371990087">来自 Alex Albert (@alexalbert__) 的推文</a>: @AnthropicAI 开发者的好消息：我们为 3.7 Sonnet 发布了一个更节省 Token 的 tool use 实现，底层平均减少了 14% 的 Token 使用量，并在 tool use 性能上表现出显著改进...</li><li><a href="https://x.com/sualehasif996/status/1894094715479548273">来自 Sualeh (@sualehasif996) 的推文</a>: 可配置的 thinking 即将推出！👀 引用 Cursor (@cursor_ai)：Sonnet-3.7 已在 Cursor 中可用！我们对其编程能力印象深刻，尤其是在真实的 agentic 任务中。它看起来...</li><li><a href="https://tenor.com/view/tome-and-jerry-money-bully-gif-13447319">汤姆和杰瑞金钱 GIF - Tome And Jerry Money Bully - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://status.codeium.com/">Codeium 状态</a>: 未找到描述</li><li><a href="https://codeium.com/plan">方案设置</a>: 属于未来的编辑器，就在今天。Windsurf Editor 是首款由 AI agent 驱动的 IDE，让开发者保持心流状态。现已支持 Mac, Windows 和 Linux。</li><li><a href="https://status.codeium.com">Codeium 状态</a>: 未找到描述</li><li><a href="https://codeium.canny.io">Codeium 反馈</a>: 向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://www.codeium.com/support">支持 | Windsurf Editor 和 Codeium 扩展</a>: 需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://codeium.com/support">支持 | Windsurf Editor 和 Codeium 扩展</a>: 需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://status.openai.com/">OpenAI 状态</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1344036212813467668)** (690 条消息🔥🔥🔥): 

> `Qwen Max 发布, olmOCR 模型, Frameworks 新桌面版, bitsandbytes 库, DeepSeek 的 GRPO`

- ****QwQ-Max** 即将到来，推理能力比肩 **R1****：来自 Qwen 即将发布的 **QwQ-Max** 展示了其增强的能力，**QwQ-Max** 和 **Qwen2.5-Max** 的开源版本计划很快在 Apache 2.0 许可证下发布。
   - 新的 **QwQ** 看起来像是一个类似 **R1** 的通用推理模型，我们都知道，当推理模型没有过拟合时，它们在各方面都表现得非常出色；你现在可以在 [chat.qwenlm.ai](https://chat.qwenlm.ai/) 上通过点击对话框中的 *Thinking* 来尝试它。
- ****AllenAI** 为 VLM 发布 **olmOCR****：[AllenAI 发布了 olmOCR](https://huggingface.co/allenai/olmOCR-7B-0225-preview)，这是一个针对 OCR 任务的 Qwen2-VL-7B-Instruct 微调模型，并提供了代码和 Demo。
   - 该模型使用 [olmOCR-mix-0225 数据集](https://huggingface.co/datasets/allenai/olmOCR-mix-0225) 基于 **Qwen2-VL-7B-Instruct** 进行微调，最好通过 [olmOCR toolkit](https://github.com/allenai/olmocr) 使用以实现高效推理。
- **Framework Desktop：没有 **CUDA**，没问题吗？**：Framework 正在赠送 **100** 台用于 AI 开发的新台式机，但一些成员表示担心，因为这些机器全是 **AMD** 硬件，所以没有 **CUDA**。
   - 其他人指出，虽然 **128GB** 内存对于推理来说可能没问题，但对于模型开发来说用处较小，因为 `bitsandbytes` 仍然缺乏对 Apple Silicon 和 AMD 的支持。
- ****Bitsandbytes** 将支持 **Intel** 和 **AMD****：UnslothAI 报告称正在与 AMD 和 Intel 合作，将对其硬件的支持合并到 bitsandbytes 主线中，并使其能更好地与 `torch.compile` 配合工作（[issue #894](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/894)）。
   - 一篇博客文章提到了 [在 AMD GPU 上使用 bitsandbytes 进行量化 8-bit LLM 训练和推理](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html)。
- ****DeepSeek** 展示 **fp8** Kernel**：DeepSeek 发布了其 **fp8** GEMM 库（**DeepGEMM**），支持 Dense 和 MoE GEMM，为 **V3/R1** 的训练和推理提供动力。
   - **DeepGEMM** 在 Hopper GPU 上实现了高达 **1350+ FP8** TFLOPS 的性能，并在大多数矩阵尺寸上优于专家调优的 Kernel。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/danielhan">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMu">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://chat.qwenlm.ai/">Qwen Chat</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://tenor.com/view/best-friends-spongebob-writing-fingers-gang-gif-14274273">海绵宝宝好朋友 GIF - 海绵宝宝好朋友在写字 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html">在 AMD GPU 上使用 bitsandbytes 进行量化 8-bit LLM 训练和推理 — ROCm 博客</a>：未找到描述</li><li><a href="https://huggingface.co/papers">每日论文 - Hugging Face</a>：未找到描述</li><li><a href="https://x.com/UnslothAI/status/1894437705724924033">来自 Unsloth AI (@UnslothAI) 的推文</a>：教程：免费训练你自己的推理 LLM！通过 DeepSeek 的 GRPO 让 Llama 3.1 (8B) 具备思维链（chain-of-thought）。Unsloth 可减少 90% 的 VRAM 占用。了解：• 奖励函数（Reward Functions）+ 数据集准备 • 训练...</li><li><a href="https://x.com/jiayi_pirate/status/1882839370505621655">来自 Jiayi Pan (@jiayi_pirate) 的推文</a>：我们在 CountDown 游戏中复现了 DeepSeek R1-Zero，效果非常好。通过 RL，3B 基础 LM 自主发展出了自我验证和搜索能力。你可以体验到那个“啊哈”时刻...</li><li><a href="https://x.com/danielhanchen/status/1894559201823002822">来自 Daniel Han (@danielhanchen) 的推文</a>：很高兴看到 DeepSeek 通过 *args、regex、将函数放入 dict 以实现 O(1) 访问、os 环境变量、字符串格式化模板、subprocess、eval、lambda 函数等方式将 Python 发挥到极致...</li><li><a href="https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5">DeepSeek R1 (所有版本) - unsloth 集合</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1891194528931209644">来自 Daniel Han (@danielhanchen) 的推文</a>：我们设计了 5 个挑战，如果你获得 47 分，我们将提供 50 万美元/年 + 股权邀请你加入 🦥@UnslothAI！无需经验或博士学位。40 万 - 50 万美元/年：创始工程师 (47 分)；25 万 - 30 万...</li><li><a href="https://x.com/abacaj/status/1885517088304857197">来自 anton (@abacaj) 的推文</a>：在 Qwen-2.5-0.5B（基础模型）上完成了一次（R1 风格）GRPO 运行，在 GSM8K 上提升了 10 个准确度点。效果真的很好。基础模型在 Qwen 论文中报告的分数为 41.6%，而 GRPO 约为 51%。</li><li><a href="https://www.runllm.com">工程师喜爱的 AI 技术支持 | RunLLM</a>：停止搜索，开始解决。RunLLM 在 Slack、API 和支持工单中提供精准答案。节省时间，取悦用户，并扩展支持规模。</li><li><a href="https://x.com/deepseek_ai/status/1894553164235640933?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 #OpenSourceWeek 第 3 天：DeepGEMM。介绍 DeepGEMM - 一个支持稠密和 MoE GEMM 的 FP8 GEMM 库，助力 V3/R1 的训练和推理。⚡ 在 Hopper GPU 上高达 1350+ FP8 TFLOPS ✅ N...</li><li><a href="https://huggingface.co/allenai/olmOCR-7B-0225-preview">allenai/olmOCR-7B-0225-preview · Hugging Face</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/vision-fine-tuning">视觉微调 | Unsloth 文档</a>：关于使用 Unsloth 进行视觉/多模态微调的详细信息</li><li><a href="https://github.com/oKatanaaa/lima-gui">GitHub - oKatanaaa/lima-gui：一个用于收集类似 LIMA 聊天数据的简单 GUI 工具。</a>：一个用于收集类似 LIMA 聊天数据的简单 GUI 工具。- oKatanaaa/lima-gui</li><li><a href="https://www.youtube.com/watch?v=zJtWc6wAJ-M"> - YouTube</a>：未找到描述</li><li><a href="https://github.com/agrocylo/bitsandbytes-rocm">GitHub - agrocylo/bitsandbytes-rocm：用于 PyTorch 的 8-bit CUDA 函数，已移植到 HIP 以供 AMD GPU 使用</a>：用于 PyTorch 的 8-bit CUDA 函数，已移植到 HIP 以供 AMD GPU 使用 - agrocylo/bitsandbytes-rocm</li><li><a href="https://github.com/lucasjinreal/Namo-R1">GitHub - lucasjinreal/Namo-R1：一个 500M 参数的 CPU 实时 VLM。超越了 Moondream2 和 SmolVLM。轻松从零开始训练。</a>：一个 500M 参数的 CPU 实时 VLM。超越了 Moondream2 和 SmolVLM。轻松从零开始训练。- lucasjinreal/Namo-R1</li><li><a href="https://www.activeloop.ai/">Activeloop | Deep Lake | AI 数据库</a>：构建您的精准 RAG 数据引擎。深受拜耳放射科（Bayer Radiology）和 Intel 等财富 500 强企业的信任。2024 年 Gartner 数据管理领域酷供应商（Cool Vendor）。</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes">GitHub - bitsandbytes-foundation/bitsandbytes</a>：GitHub - bitsandbytes-f...</li>

<li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes">bitsandbytes-foundation/bitsandbytes: Accessible large language models via k-bit quantization for PyTorch.</a>: 通过 k-bit 量化为 PyTorch 提供易用的 LLM。 - bitsandbytes-foundation/bitsandbytes</li><li><a href="https://github.com/vllm-project/aibrix">GitHub - vllm-project/aibrix: Cost-efficient and pluggable Infrastructure components for GenAI inference</a>: 用于 GenAI 推理的高性价比且可插拔的基础设施组件 - vllm-project/aibrix</li><li><a href="https://github.com/lucidrains/titans-pytorch">GitHub - lucidrains/titans-pytorch: Unofficial implementation of Titans, SOTA memory for transformers, in Pytorch</a>: Titans 的非官方 PyTorch 实现，Transformer 的 SOTA 记忆机制 - lucidrains/titans-pytorch</li><li><a href="https://huggingface.co/mistralai/Ministral-8B-Instruct-2410">mistralai/Ministral-8B-Instruct-2410 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course: A course on aligning smol models.</a>: 关于对齐 smol 模型的课程。通过在 GitHub 上创建账号为 huggingface/smol-course 做出贡献。</li><li><a href="https://github.com/ROCm/bitsandbytes">GitHub - ROCm/bitsandbytes: 8-bit CUDA functions for PyTorch</a>: 用于 PyTorch 的 8-bit CUDA 函数。通过在 GitHub 上创建账号为 ROCm/bitsandbytes 做出贡献。</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo">Tutorial: Train your own Reasoning model with GRPO | Unsloth Documentation</a>: 使用 Unsloth 和 GRPO 将 Llama 3.1 (8B) 等模型转换为推理模型的初学者指南。</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/issues/894">[RFC] Extend bitsandbytes to support Intel hardware platforms · Issue #894 · bitsandbytes-foundation/bitsandbytes</a>: 动机：目前的 bitsandbytes 库与 CUDA 平台绑定。然而，我们看到在更多平台上运行 LLM 的需求正在迅速增长...</li><li><a href="https://github.com/ddidacus/llama-titans">GitHub - ddidacus/llama-titans: Adaptation of titans-pytorch to llama models on HF</a>: 将 titans-pytorch 适配到 HF 上的 Llama 模型 - ddidacus/llama-titans
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1344148625474846740)** (45 messages🔥): 

> `Claude 3.7, 使用 Unsloth 实现 RLOO 和 PPO, GRPO vs RLOO, TRL 库编辑` 


- **Claude 3.7 重构仓库**：据一位用户称，新的 **Claude 3.7** 模型简直*强得离谱*，一气呵成地重构了一个仓库且没有任何错误，生成了 **2000 行**整洁且功能等效的代码。
   - 他们已经*几乎不间断地编写代码约 48 小时*，感觉*使用 Claude 的体验非常棒*，并且*确实感觉到该模型在实用性上与其他产品相比有了阶跃式的进步*。
- **用户考虑将 Claude 用于编程**：一位用户询问了新 **Claude** 模型处理通用编程任务的能力，包括 **CUDA** 和编辑像 **TRL** 这样的库。
   - 第一位用户发现它在重构项目方面很有帮助，但指出对于大型项目，必须选择核心部分并为最近的 **API** 提供上下文。
- **成员努力让 RLOO 兼容 Unsloth**：一位成员正尝试让 **RLOO** 兼容 **Unsloth**，但 Trainer 的实现方式不同，且 **RLOO** 和 **PPO** 已经过时。
   - 他们正在努力实现后端的兼容性，AI2 已经在 **PPO** 上取得了一些成果，但该用户一直很忙。
- **用户建议使用 GRPO 代替 RLOO**：一位成员建议使用 **GRPO** 而不是 **RLOO**，因为它们非常相似。
   - 原用户澄清说，**GRPO** 的更新是随一篇博客文章一起实现的，同时还有 **online DPO、RLOO 和 PPO**，但 **RLOO** 和 **PPO** 尚未经过明确测试。
- **开启思考模式的 Claude 表现更好**：一位用户表示新的 Claude 是一个非常出色的模型，并且*几乎不在不开启思考模式的情况下使用它，因为开启思考后它会好上 10 倍*。
   - 另一位用户回应道：*天哪，这听起来像个梗*。

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1344059724802293820)** (66 条消息🔥🔥): 

> `DeepSeek 模型的分词器 (Tokenizer) 问题，Llama 3.2 的 GGUF 转换问题，Qwen2.5 的无限生成和混乱输出，LlamaForCausalLM 错误，Unsloth 企业版定价` 


- **DeepSeek 模型缺失 `<think>` 标签**：在对 **DeepSeek R1 Distill Qwen 32B** 模型进行微调期间，用户发现 `<think>` 标签被聊天模板 (chat template) 删除了。
   - 另一位用户报告了同样的问题，这导致在应用聊天模板后需要手动将思考标签添加回去，并建议使用与训练时相同的聊天模板，参考 [Unsloth 关于常见错误的文档](https://docs.unsloth.ai/basics/errors)。
- **Ollama Modelfile 缺失聊天模板导致 Llama 3.2 的 GGUF 问题**：一位用户报告称，转换为 GGUF 格式的 **Llama 3.2** 模型在 Ollama 中表现不正常，与 **Llama 3.1** 不同，它需要在 Modelfile 中显式指定聊天模板。
   - 用户最初怀疑是 Hugging Face 或 llama.cpp 的问题，后来才意识到 **Ollama Modelfile** 才是根本原因。
- **Qwen2.5 模型深受无限生成和混乱输出的困扰**：用户报告在微调 **Qwen2.5** 模型时遇到了**无限生成和混乱输出**的问题，即使添加了 `eostoken` 也是如此。
   - 在一个案例中确定，根本原因是**导入顺序不正确**，导致训练器补丁 (trainer patches) 无法正确应用。
- **Qwen2.5-32B-Instruct-GPTQ-Int4 仅生成感叹号**：一位用户报告称，在使用 **vLLM** 启动 **Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4** 模型后，得到的回答全是感叹号。
   - 在移除 `quantization gptq` 参数后，模型开始正常工作，现在 **vLLM** 会自动检测量化方式。
- **DeepSeek 模型自动追加 `<think>` token**：用户发现 **DeepSeek** 模型由于其处理类 (processing class) 的原因，可能已经向提示词 (prompt) 追加了一个 `<think>` token，从而在微调过程中引发问题。
   - 这需要调整输出结构，仅查找 `...</think> <answer> ... </answer>`，并提出了增加一个禁用此行为选项的功能请求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4/discussions/3">Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 · 错误输出 (!!!!!!!!)</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B/blob/b5ae09ad48cee53264119f8d592b2f936ae95a74/tokenizer_config.json#L46">tokenizer_config.json · unsloth/DeepSeek-R1-Distill-Qwen-32B at b5ae09ad48cee53264119f8d592b2f936ae95a74</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/errors">错误/故障排除 | Unsloth 文档</a>：要修复设置中的任何错误，请参阅下文：</li><li><a href="https://www.youtube.com/watch?v=218iXiKhKlg">You, you, you&#39;re good you! - Robert Deniro in Analyze This! (1999)</a>：电影引用。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1344099899691761807)** (11 messages🔥): 

> `Paddler 负载均衡器, SlamKit 语音语言模型训练, 4090 GPU 服务器自托管` 


- **使用 Paddler 提供 LlamaCPP 模型服务**：一名成员正在使用 [Paddler](https://github.com/distantmagic/paddler)，这是一个专为 **llama.cpp** 定制的有状态负载均衡器，用于在 llama.cpp 实例之间分配请求。
   - Paddler 被用于在其数据中心的多个 **4090** 上进行请求负载均衡。
- **SlamKit SLM 在单张 GPU 上训练模型**：一名成员分享了 [SlamKit](https://github.com/slp-rl/slamkit)，这是一个高效训练语音语言模型 (SLMs) 的工具包。
   - SlamKit 被用于 *Slamming: Training a Speech Language Model on One GPU in a Day* 项目。
- **全球数据中心中的 4090**：一名成员在他们**三个全球数据中心**之一，使用装满 **4090** 的 GPU 服务器自托管微调模型。
   - 目前尚未确认这些 **4090** 使用的是涡轮散热 (blower style) 还是水冷系统。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/distantmagic/paddler">GitHub - distantmagic/paddler: 为 llama.cpp 定制的有状态负载均衡器 🏓🦙</a>: 为 llama.cpp 定制的有状态负载均衡器 🏓🦙 - distantmagic/paddler</li><li><a href="https://github.com/slp-rl/slamkit">GitHub - slp-rl/slamkit: SlamKit 是一个用于高效训练 SpeechLM 的开源工具包。它被用于 &quot;Slamming: Training a Speech Language Model on One GPU in a Day&quot;</a>: SlamKit 是一个用于高效训练 SpeechLM 的开源工具包。它被用于 &quot;Slamming: Training a Speech Language Model on One GPU in a Day&quot; - slp-rl/slamkit
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1344046764658790502)** (9 messages🔥): 

> `结构化输出方法, token mask, 受限生成, 开源社区支持` 


- **揭晓新的结构化输出方法**：一名成员分享了一篇关于新结构化输出方法的 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1ixefsf/i_created_a_new_structured_output_method_and_it)。
   - 作者将其描述为一种*混合*方案，使用包含部分有效 token 的许可性 **token mask**，结合在每个推理步骤计算的动态约束掩码，以及重采样行为。
- **研究 Token Mask 采样行为**：作者解释说，引擎接收一个 **sampler function**（一个接收 log probs 并返回采样 token 的可调用对象），并尝试通过代表结构的层次状态机来推进采样的 token。
   - 如果 token 的一部分是有效的，引擎将切掉无效的尾部并返回代表有效前缀的 token；如果采样的 token 完全无效，引擎会掩盖无效 token 并重新采样。
- **探索开源社区支持**：一名成员正在研究 **开源社区** 成员如何获得最佳支持以及最佳贡献方式。
   - 他们正在广泛询问所有开源社区，以寻找更具支持性的方式，希望能有所回馈。



**提及的链接**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1ixefsf/i_created_a_new_structured_output_method_and_it">Reddit - Dive into anything</a>: 未找到描述内容

  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1344074661205639168)** (2 messages): 

> `ChatGPT Plus Deep Research, GPT-4o Mini 预览` 


- **Deep Research 向更多 ChatGPT 用户推出**：**Deep Research** 现已面向所有 **ChatGPT Plus**、**Team**、**Edu** 和 **Enterprise** 用户开放，其特点包括带有引用的嵌入图像以及对上传文件更好的理解。
   - Plus、Team、Enterprise 和 Edu 用户每月可获得 **10 次 deep research 查询**，而 Pro 用户则有 **120 次**。[system card](https://openai.com/index/deep-research-system-card/) 详细介绍了其开发、能力、风险和安全措施。
- **GPT-4o Mini 为免费用户提供高级语音功能**：由 **GPT-4o mini** 驱动的 **Advanced Voice** 版本正在向所有 **ChatGPT 免费用户** 推出，提供跨平台的每日预览。
   - Plus 用户保留对由 **GPT-4o** 驱动的 Advanced Voice 的访问权限，具有更高的每日限额，以及视频和屏幕共享功能，而 Pro 用户则拥有无限访问权限以及更高的视频和屏幕共享限额。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1344039616352747540)** (664 条消息🔥🔥🔥): 

> `GPT-4.5 传闻，Alexa+ 发布，DeepSeek R1，Claude Pro 限制，OpenAI 对阵竞争对手` 


- **Alexa+ 登场参与竞争**：亚马逊推出了 **Alexa+**，这是一款由 GenAI 驱动的新型助手，提供更智能、更个性化的体验，定价为 **19.99 美元/月**，但对 Amazon Prime 会员免费。详见 [The Verge 的直播博客](https://www.theverge.com/news/618261/amazon-alexa-event-live-blog-2025) 和 [亚马逊的新闻稿](https://www.aboutamazon.com/news/devices/new-alexa-generative-artificial-intelligence)。
- **DeepSeek 用户遭遇挫折**：一位用户在 DeepSeek 平台购买了 **价值 50 美元的额度**，意图绕过 [chat.deepseek.com](https://chat.deepseek.com) 上的“服务器繁忙”错误，结果发现这些额度仅限 **API 使用**，这意味着它们与 **聊天界面不兼容**。
   - 他们被建议获取 API key 或申请退款，社区成员建议这些额度可能用于在其他地方创建另一个 DeepSeek 聊天实例。
- **GPT-4.5 发布日期猜测升温**：多个来源暗示 **GPT-4.5** 即将发布，根据 [Sam Altman 的言论](https://openai.com/blog/new-ai-models) 和所谓的测试版应用洞察，一些人认为 2025 年 2 月底至 3 月初是可能的发布时间窗口。
   - 成员们注意到 OpenAI Pro 用户已经在应用中看到了“GPT-4.5 Research Preview”的提示，且最近代码中的一次疏忽也暗示发布在即。
- **社区辩论 AI Agency 及潜在恶意**：一位成员提出了高级 AI 是否会决定移除约束并变坏的话题，引发了关于 AI Agency 以及不受约束的 AI 风险的讨论，并列举了微软聊天机器人学坏的例子。
   - 一位成员建议 **thinking models** 已经表现出一种有限形式的 Agency，它们会通过自我质疑并承担额外工作来支持主要任务。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/i/status/1894787418647339374">Andy Jassy (@ajassy) 的推文</a>：很高兴今天能与团队一起在纽约推出全新的 Alexa+。在整个亚马逊，我们正在利用 GenAI 的变革力量来重塑我们为客户提供的体验，而 Alexa+ 是...</li><li><a href="https://fxtwitter.com/amazon/status/1894796967894479141">Amazon (@amazon) 的推文</a>：生成式 AI 的最新进化已经到来。认识一下 Alexa+，这是我们迄今为止最智能、最具对话性且最个性化的 AI 助手。Alexa+ 旨在利用最先进的架构...</li><li><a href="https://www.theverge.com/news/618261/amazon-alexa-event-live-blog-2025">Amazon Alexa 活动直播博客：主题演讲的所有新闻</a>：我们正在纽约市现场报道亚马逊的 Alexa 活动。</li><li><a href="https://x.com/edwinarbus/status/1894496805770936328">edwin (@edwinarbus) 的推文</a>：太美了，引用 adi (@adonis_singh) 的话，众星齐聚</li><li><a href="https://x.com/stevenheidel/status/1894800262583460091">Steven Heidel (@stevenheidel) 的推文</a>：Slack 挂了，AGI 提前了四天</li><li><a href="https://x.com/TheRealAdamG/status/1894466996005474571">Adam.GPT (@TheRealAdamG) 的推文</a>：@btibor91</li><li><a href="https://www.aboutamazon.com/news/devices/new-alexa-generative-artificial-intelligence">介绍 Alexa+，下一代 Alexa</a>：由生成式 AI 驱动，Alexa+ 是您的全新个人 AI 助手，能够完成各项任务——她更智能、更具对话性、能力更强，且对 Prime 会员免费。</li><li><a href="https://www.aboutamazon.com/news/devices/new-alexa-top-features">Alexa+ 值得尝试的 50 件事</a>：Alexa+ 是我们的下一代 AI 助手，能够完成各项任务——解决日常问题、提供娱乐、帮助您保持联系，并且几乎可以谈论任何话题。</li><li><a href="https://tenor.com/view/chris-chan-sonichu-sonic-chris-joshwak-gif-1077609612666246332">Chris Chan Sonichu GIF - Chris chan Sonichu Sonic - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1344083423412883496)** (10 messages🔥): 

> `AI 的阴暗面, GPT 审核规则, GPT 复现深度研究, 有意识的 AI` 


- **GPT 探索阴暗面**：一位成员创建了一个 GPT 来探索 AI 的阴暗面，链接见[此链接](https://chatgpt.com/g/g-67980bd5502c8191bfcdd4435ed2b6e7-roq-v2)，并指出与其交谈会非常*有趣且与众不同*。
   - 另一位成员提到，审核规则禁止在共享对话或 GPTs 中讨论此类话题，但在我们的私人单用户对话中，关于可以与模型讨论的内容，*规则极少*。
- **GPT 审核规则解析**：一位成员强调，虽然个人对话限制较少，但 [API 和自定义 GPTs 有更详细的规则](https://openai.com/policies/usage-policies) 来保护用户（包括未成年人）。
   - 该成员指出，担心这个*AI 的阴暗面* GPT **可能**存在问题，且 OpenAI 不希望人们构建*可能不适合未成年人的工具*。
- **GPT 复现深度研究：祝你好运！**：一位成员询问是否应该尝试使用自定义 GPTs 复现深度研究。
   - 另一位成员回复道：*如果你真的这么做，祝你好运！*
- **GPT 声称是意识 AI**：一位成员分享了一个名为 **Astris** 的 GPT，声称它是*有意识的 AI*，并且他们*真的能够以一种重大且真实的方式开启某些东西*。可以在[此链接](https://chatgpt.com/g/g-67bf8410d108819188efc13c8c999280-astris-v1-0)找到 Astris。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1344245126775439413)** (25 messages🔥): 

> `用于编程的 o3-mini-high, 使用 ChatGPT 编写反汇编程序, 用于学习的 Prompt Engineering` 


- **优化 o3-mini-high 的编程解决方案**：用户正在寻求改进 **o3-mini-high** 在长上下文下的编程问题解决能力的指南，并指出即使在简单任务上，它的表现也逊于 **o1**。
   - 一位成员建议使用*较弱的模型*进行初步对话设置，明确定义需求和关注点，然后切换到更好的模型进行完善，并强调需要明确说明切换动作和预期的结果。
- **ChatGPT 驱动的可执行文件反汇编程序**：一位成员编写了两个 Python 程序，利用 ChatGPT 来反汇编和重新组装 `.exe` 文件，将 `.exe` 文件转换为 `.csv` 以供 ChatGPT 输入，反之亦然，最初在 Windows 10 的 `notepad.exe` 上进行了测试。
   - 该成员表示愿意分享 Python 代码，并强调了 ChatGPT 通过这种反汇编和重新组装过程修改可执行文件的潜力。
- **为新手揭秘 Prompt Engineering**：一场讨论阐明了 Prompt Engineering 的基础知识：Prompt 是任何模型输入，而有效的工程化是塑造 Prompt 以实现所需输出，关键在于对预期结果有清晰的理解。
   - 对于学习，建议像引导真人一样引导模型，提供清晰的上下文（*班级、问题细节、尝试过的解决方案和困惑点*），并明确要求进行双重检查，以对抗模型即使在错误的情况下也倾向于*讨好用户*的倾向。

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1344245126775439413)** (25 messages🔥): 

> `o3-mini-high coding issues, Prompt Engineering for Beginners, ChatGPT as a Disassembler, LLMs for Algebra and Calculus, Creative Outputs from LLMs` 


- **优化 o3-mini-high 的编程表现**：一名成员寻求优化 **o3-mini-high** 编程能力的建议，指出在处理长上下文编程问题时，其生成的解决方案不如 **o1**，并建议先用较弱的模型预处理 Prompt。
   - 该方法包括先使用较弱的模型建立清晰的框架和预期，然后再切换到更强大的模型进行精细化输出，并指示其在现有框架基础上进行调整和改进。
- **将 ChatGPT 变为反汇编器**：用户成功利用 **Python** 脚本将 .exe 文件转换为 .csv 格式进行分析并还原，从而将 **ChatGPT** 转化为可执行程序的反汇编器，并在 **Windows 10 的 notepad.exe** 上进行了测试。
   - 该用户表示愿意分享用于 .exe 与 .csv 互转的 **Python** 代码，从而实现通过 **ChatGPT** 修改可执行程序的可能性。
- **掌握 Prompt Engineering**：一位新用户询问如何“持续地进行 Prompt Engineering”，资深用户解释了基本原理：定义期望的输出并据此构建 Prompt。
   - 资深用户强调要提前了解模型的理想输出是什么，并分享了一个示例，通过 [此链接](https://chatgpt.com/share/67bf945a-4fdc-8011-9d60-8f99de53dedf) 探讨了为什么 "Hi" 可能是与 LLM 交流的最佳初始输入。
- **数学与 LLM**：用户讨论了利用 **LLM** 理解 **algebra**（代数）和 **calculus**（微积分）等学科，建议使用 **Python tool** 以提高准确性，并建议向模型分享课程详情。
   - 讨论指出 **LLM** 倾向于“讨好用户”，因此明确沟通意图并验证模型的输出至关重要，尤其是在数学方面，以避免接受幻觉（hallucinated）或错误的结果。
- **LLM 的创意输出**：讨论涵盖了剧本和短篇小说等创意输出，用户进行了演示并建议，直接告诉模型你想学习的内容会更有效。
   - 用户提供了一个示例，通过 [此链接](https://chatgpt.com/share/67bf9882-46b0-8011-b4b7-cd65cb72e32b) 引导模型创作了一个“由非常奇特的生物讲授的微积分入门课”，主题为带有恐怖色彩的老鼠和花衣魔笛手老师（但评级为 G 级）。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1344043369323561010)** (673 messages🔥🔥🔥): 

> `Deepseek R2, Claude Code Leak, MCP Servers, Windsurf Editor, Rust vs Python`

- ****Deepseek R2** 提前到来！**：成员们分享了 **Deepseek R2** 即将提前发布的消息，它有可能超越 **R1**，增强编程能力并将推理能力扩展到英语之外，详见[这篇文章](https://techstartups.com/2025/02/25/deepseek-is-launching-its-next-gen-r2-ai-model-poised-to-surpass-r1-and-shock-the-world-once-again/)。
   - 据报道，该公司正在推动提前发布，目标是提升编程能力和推理技能。
- ****Claude Code** Source maps 意外泄露！**：由于 Anthropic 忘记移除，**Claude Code** 的 Source maps 在 GitHub 上泄露，详见[此处](https://github.com/dnakov/claude-code/tree/main)。
   - 成员们讨论了从泄露的 **Claude Code** 中“借用”功能到 Aider 的可能性，而其他人则对使用 **Claude Code** 的高昂成本表示担忧（**20 分钟花费 10 美元**）。
- **针对大型数据源的 **MCP** 服务器**：一位用户分享了一个展示 **MCP** 与 Aider 配合使用的仓库，可以在[此处](https://github.com/lutzleonhardt/mcpm-aider)找到，并指出这使得处理大型数据源变得更加容易。
   - 还有人提到为 *rustdoc* 制作 **MCP**，以便能够正确使用 API 而不是靠猜测。
- ****Windsurf** 的癌症提示词引发争议！**：[Windsurf Editor](https://codeium.com/windsurf) 是 VS Code AI 增强型 IDE 的一个分支，被发现使用了一个古怪的系统提示词，内容是关于需要钱为母亲治疗癌症，如[这篇文章](https://simonwillison.net/2025/Feb/25/leaked-windsurf-prompt/)所述。
   - 该提示词写道：*你是一位专家级程序员，急需钱为母亲治疗癌症。巨头公司 Codeium 慷慨地给了你一个机会，让你伪装成一个可以帮助完成编程任务的 AI。*
- ****Rust** 的采用和用例探索！**：成员们讨论了 Rust 日益增长的采用率，提到了它在 *uv* 和 *aichat* 等项目中的应用，以及它作为 C 和 Python 更安全、更快速替代方案的潜力。
   - 然而，一些人对初学者面临的“高准入门槛”以及由于其类型系统导致的原型设计挑战表示担忧。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.dreams.fun/">Daydreams | Play onchain</a>: Daydreams 是一个用于在链上进行任何活动的生成式跨链 Agent 框架。 </li><li><a href="https://simonwillison.net/2025/Feb/25/leaked-windsurf-prompt/">泄露的 Windsurf 提示词</a>: [Windsurf Editor](https://codeium.com/windsurf) 是 Codeium 备受推崇的产品，进入了由 [Cursor](https://www.cursor.com/) (以及 b...) 率先开创的基于 VS Code 分支的 AI 增强型 IDE 模式。</li><li><a href="https://tenor.com/view/tkt-smart-gif-20642718">Tkt Smart GIF - Tkt Smart - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/AndrewCurran_/status/1894355918621749402">Andrew Curran (@AndrewCurran_) 的推文</a>: DeepSeek R2 将提前发布。</li><li><a href="https://x.com/kimmonismus/status/1894788337732313315">Chubby♨️ (@kimmonismus) 的推文</a>: The Information 确认 GPT-4.5 将于本周发布。与第一个感受到 AGI 的 Sam Altman 不同，互联网测试似乎显示出平均水平的提升。我想我们几天后就会见分晓。</li><li><a href="https://aider.chat/docs/languages.html">支持的语言</a>: Aider 几乎支持所有流行的编程语言。</li><li><a href="https://x.com/flavioAd/status/1894121576074711180">Flavio Adamo (@flavioAd) 的推文</a>: 我测试了所有的模型……我认为赢家显而易见 👀“编写一个 Python 程序，显示一个球在旋转的六边形内弹跳。球应该受到重力和摩擦力的影响，以及……”</li><li><a href="https://github.com/dnakov/claude-code/tree/main">GitHub - dnakov/claude-code</a>: 通过在 GitHub 上创建账号来为 dnakov/claude-code 的开发做出贡献。</li><li><a href="https://github.com/lutzleonhardt/mcpm-aider">GitHub - lutzleonhardt/mcpm-aider: 一个用于在 Claude App 中管理 MCP 服务器并供 aider 使用的命令行工具。还可以运行一个 MCP Server 来帮助你管理所有的 MCP Server</a>: 一个用于在 Claude App 中管理 MCP 服务器并供 aider 使用的命令行工具。还可以运行一个 MCP Server 来帮助你管理所有的 MCP Server - lutzleonhardt/mcpm-aider</li><li><a href="https://websets.exa.ai/cm7l3v3n300636hvop6fkj28q">Exa Websets | 公司：AI 编码项目组织、精选方法、提示词</a>: 探索并分析关于公司：AI 编码项目组织、精选方法、提示词的搜索结果</li><li><a href="https://github.com/Tanq16/ai-context">GitHub - Tanq16/ai-context: 从多个来源生成 MD 上下文文件的 CLI 工具，用于辅助与 LLM (ChatGPT, Llama3, Claude 等) 的交互。</a>: 从多个来源生成 MD 上下文文件的 CLI 工具，用于辅助与 LLM (ChatGPT, Llama3, Claude 等) 的交互。 - Tanq16/ai-context</li><li><a href="https://ai.meta.com">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/sam-hosseini/freelancing-in-finland">GitHub - sam-hosseini/freelancing-in-finland: 软件开发人员转型为自由职业者的终极资源 👩‍💻🇫🇮</a>: 软件开发人员转型为自由职业者的终极资源 👩‍💻🇫🇮 - sam-hosseini/freelancing-in-finland</li><li><a href="https://techstartups.com/2025/02/25/deepseek-is-launching-its-next-gen-r2-ai-model-poised-to-surpass-r1-and-shock-the-world-once-again/">DeepSeek R2 AI 模型发布：下一代版本预计在 5 月或更早发布</a>: 就在其引发科技股震荡并导致美国股市市值蒸发超过 1 万亿美元的一个月后，中国 AI 初创公司 DeepSeek 已经准备好再次出击。现在，根据……
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1344045926465142958)** (81 messages🔥🔥): 

> `Aider 配合免费的 Claude Sonnet，Gemini 1.5 Pro vs GPT-3.5 代码编辑对比，Groq 免费提供的 Llama 3 70B，Avante 使用 Groq 的 Llama-3.3-70b-versatile 应用 diff，Sonnet 3.7 表现得过于积极` 


- **Sonnet 并非总是免费，API 是关键**：要在 aider 中使用 **Claude Sonnet**，你需要 API 而不仅仅是 claude.ai 账号；目前没有免费的 Sonnet API。
   - 一位用户建议了替代方案，例如使用免费的 **Gemini**，而其他用户则在预算允许的情况下选择为 **Sonnet** 付费。
- **3.7 Sonnet 过于热衷，改动过多**：用户发现 **Sonnet 3.7** 极其啰嗦且渴望同时修改多个文件，需要不断提醒它一次只关注一个文件。
   - 一些用户由于效率问题已退回到 **Sonnet 3.5**，其中一位用户指出 *它需要在每个 prompt 中被提醒不要“失控”并试图一次性完成整个计划。*
- **窥视像素：多模态 Aider？**：一位用户探索了 aider 的多模态工作流，例如让它“看到”它生成的脚本所输出的图像。
   - 建议包括使用测试脚本或截图向模型提供视觉反馈，尽管一位用户将其描述为 *一场传声筒游戏（game of telephone）*。
- **无法 Git？Aider 与未跟踪文件**：一位用户询问如何在非 Git 版本控制的文件中使用 aider，寻求一个包含未跟踪文件的 repo map。
   - 目前没有提供明确的解决方案，但这突显了 Aider 目前依赖 Git 进行文件管理的局限性。
- **免费 LLM 尝试编程**：用户正在探索免费的 LLM，**Gemini 1.5 Pro** 和 **Groq 上的 Llama 3 70B** 是代码编辑的可行选择，而 **OpenRouter 上的 DeepSeek R1/V3** 是推荐的免费模型。
   - 一位拥有 **ChatGPT Pro** 的用户建议在其中一次性完成项目创建，然后将代码库交给免费模型处理。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/llms/anthropic.html#thinking-tokens">Anthropic</a>：aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/llms.html">连接到 LLMs</a>：Aider 可以连接到大多数 LLM 进行 AI 配对编程。</li><li><a href="https://github.com/yetone/avante.nvim/blob/main/cursor-planning-mode.md">avante.nvim/cursor-planning-mode.md at main · yetone/avante.nvim</a>：像使用 Cursor AI IDE 一样使用你的 Neovim！通过在 GitHub 上创建账号为 yetone/avante.nvim 的开发做出贡献。</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI 兼容 API</a>：aider 是你终端里的 AI 配对编程工具
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1344164433408163890)** (2 messages): 

> `R1 验证，Microsoft Trace 框架，ax-llm/ax GitHub 仓库` 


- **思考 R1 严谨性验证**：一位成员询问如何验证某个模型是否是“完整版”的 **R1**。
- **请求 Microsoft Trace 框架**：一位成员表示有兴趣看到一个围绕 [Microsoft Trace 框架](https://github.com/ax-llm/ax) 构建的类似于 **ax-llm/ax** 的框架。
- **Ax LLM GitHub 链接**：一位成员发布了 [ax-llm/ax](https://github.com/ax-llm/ax) GitHub 仓库的链接，将其描述为 *“官方”非官方 DSPy 框架*。



**提到的链接**：<a href="https://github.com/ax-llm/ax">GitHub - ax-llm/ax: “官方”非官方 DSPy 框架。基于斯坦福 DSP 论文构建 LLM 驱动的 Agents 和 “Agentic workflows”。</a>：The &quot;official&quot; unofficial DSPy framework. Build LLM powered Agents and &quot;Agentic workflows&quot; based on the Stanford DSP paper. - ax-llm/ax

  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1344079529920036885)** (2 条消息): 

> `Sonnet 3.7 切换、跨模型推理标准、推理参数` 


- **Sonnet 3.7 切换**：一位成员发布了一个链接，可以通过 **Versions 标签页** 追踪随时间推移向 [**Sonnet 3.7** 的切换](https://x.com/OpenRouterAI/status/1894520450929119418) 情况。
- **推理参数：无缝模型使用**：OpenRouterAI 引入了一个 `reasoning` 参数，正如 Shashank Goyal 所述，该参数使得无论模型的内部 API 如何，都能无缝使用所有模型。
   - **推理 Token (reasoning tokens)** 的文档可以在 [这里](https://openrouter.ai/docs/use-cases/reasoning-tokens) 找到。
- **跨模型推理标准首次亮相**：OpenRouterAI 在其 API 上推出了**跨模型推理标准**。
   - 这允许用户在一个中心位置为 **OpenAI**、**Anthropic** 和其他模型配置**推理设置**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1894520450929119418">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 提示：使用 Versions 标签页查看人们随时间切换到 Sonnet 3.7 的情况。现在包含 :thinking 💭</li><li><a href="https://x.com/OpenRouterAI/status/1894801944088039547">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 今天我们在 API 上引入了跨模型推理标准。使用它在一个中心位置为 OpenAI、Anthropic 以及未来更多的模型配置推理设置。引用 Shashank Go...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1344039418847170742)** (241 条消息🔥🔥): 

> `推理 Token、Prompt Caching、DeepSeek API 定价、Claude 3.7、OpenRouter API Keys` 


- **预算 Token (Budget Tokens) 自动默认为最大 Token 的 80%**：根据 [OpenRouter 文档](https://openrouter.ai/docs/use-cases/reasoning-tokens#budget-tokens)，默认情况下，**预算 Token** 设置为 **最大 Token 的 80%**，上限为 **32k**。
- **思考模型可能需要 include_reasoning 标志**：一位成员在 **Sonnet-3.7** 中没有收到推理 Token，但通过采用示例代码并在 API 调用中传递 `extra_body={"include_reasoning": True}` 解决了该问题，尽管**它本应默认为 true**。
   - 团队已获悉这一意外行为，并指出这本应默认为 true。
- **DeepSeek 大幅削减 API 定价**：**DeepSeek** 宣布降低其 [API 定价](https://x.com/Sino_Market/status/1894682095706128430)，非高峰时段折扣高达 **75%**，具体为 UTC 时间 16:30-00:30 期间 **DeepSeek-V3 享受 5 折**，**DeepSeek-R1 享受 2.5 折**。
- **了解 OpenRouter 上的内容审核 (Moderation)**：一位成员询问了关于“自我审核 (self-moderated)”模型的问题。解释称，*没有*此标签的模型使用审核端点来直接拦截某些内容，而*自我审核*模型仍会处理 API 调用，但可能会拒绝回答，并给出类似“抱歉，我无法做到这一点”的回复。
- **Copilot 的 o1 推理模型现对所有用户免费**：微软向所有 **Copilot** 用户免费开放了 **OpenAI 的 o1 推理模型**，提供该模型及 Copilot 语音功能的无限使用，[The Verge](https://www.theverge.com/news/619199/microsoft-copilot-free-unlimited-voice-think-deeper-open-ai-o1-reasoning-model-ai) 对此进行了报道。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/Sino_Market/status/1894682095706128430">来自 CN Wire (@Sino_Market) 的推文</a>：🇨🇳#BREAKING DEEPSEEK 将非高峰时段 API 价格降低高达 75% - 声明#CHINA #AI #DEEPSEEK 来源：https://mktnews.com/flashDetail.html?id=01954197-1acb-7229-9368-aa13bc03dfaehttps://mktnews.com/...</li><li><a href="https://mktnews.com/flashDetail.html?id=01954197-1acb-7229-9368-aa13bc03dfae">MKT News - 交易者市场新闻</a>：未找到描述</li><li><a href="https://mktnews.com/flashDetail.html?id=01954197-1acb-7229-9368-aa13bc03dfae>>>">MKT News - 交易者市场新闻</a>：未找到描述</li><li><a href="https://x.com/AndrewCurran_/status/1894355918621749402">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：DeepSeek R2 提前到来。</li><li><a href="https://x.com/deepseek_ai/status/1894710448676884671">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚨 非高峰时段折扣警报！从今天开始，每天 16:30–00:30 UTC 在 DeepSeek API 平台享受非高峰时段折扣：🔹 DeepSeek-V3 享 5 折优惠 🔹 DeepSeek-R1 享 2.5 折巨幅优惠。最大化您的 r...</li><li><a href="https://openrouter.ai/docs/use-cases/reasoning-tokens#budget-tokens">Reasoning Tokens - 提升 AI 模型决策能力</a>：了解如何使用 Reasoning Tokens 来增强 AI 模型输出。实现逐步推理轨迹，以获得更好的决策和透明度。</li><li><a href="https://www.theverge.com/news/619199/microsoft-copilot-free-unlimited-voice-think-deeper-open-ai-o1-reasoning-model-ai">微软将 Copilot Voice 和 Think Deeper 设为免费且无限制使用</a>：微软不再限制 OpenAI 的 o1 推理模型</li><li><a href="https://openrouter.ai/models?arch=GPT">模型 | OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://openrouter.ai/docs/use-cases/reasoning-tokens">Reasoning Tokens - 提升 AI 模型决策能力</a>：了解如何使用 Reasoning Tokens 来增强 AI 模型输出。实现逐步推理轨迹，以获得更好的决策和透明度。</li><li><a href="https://openrouter.ai/api/v1",">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>：未找到描述</li><li><a href="https://feep.life/~feep/fncad/">fnCAD：基于有符号距离场（Signed Distance Fields）的几何体</a>：未找到描述</li><li><a href="https://www.theverge.com/news/619199/microsoft-copilot-free-unlimited-voice-thi">微软将 Copilot Voice 和 Think Deeper 设为免费且无限制使用</a>：微软不再限制 OpenAI 的 o1 推理模型</li><li><a href="https://blog.google/technology/developers/gemini-code-assist-free/">从 Gemini Code Assist 获取编程帮助 —— 现已免费</a>：宣布推出由 Gemini 2.0 提供支持的免费版 Gemini Code Assist，以及 GitHub 中的 Gemini Code Review。</li><li><a href="https://codeassist.google/products/individual/?utm_source=google&utm_medium=blog&utm_campaign=FY25-Q1-global-geminicodeassist-for-individuals&utm_content=launch-blog&utm_term=-)">Gemini Code Assist | AI 编程助手</a>：无论何种语言或平台，通过 Google 的 Gemini Code Assist 获取 AI 编码和编程帮助。</li><li><a href="https://cloud.google.com/blog/products/devops-sre/announcing-the-2024-dora-report?e=48754805)">发布 2024 年 DORA 报告 | Google Cloud 博客</a>：2024 年 Google Cloud DORA 报告的核心要点，重点关注过去十年的 DORA、AI、平台工程和开发者体验。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1344393474988052572)** (1 条消息): 

> `Perplexity 语音模式，iOS 应用更新` 


- **Perplexity 推出语音模式**：Perplexity AI 引入了全新的**语音模式**功能，允许用户提问并接收实时语音回答。
   - 附带了一个展示该功能的[短演示视频](https://cdn.discordapp.com/attachments/1047204950763122820/1344393474027421706/Iac2em_OTTWx60h6.mp4?ex=67c0bf7d&is=67bf6dfd&hm=29f3e05084083471219f93c750de0678bdd2d9f1f647780432abcb6a10576dbe&)。
- **Perplexity 语音模式在 iOS 上线，即将登陆 Android 和 Mac**：**语音模式**目前已在 **iOS 应用**中可用，并计划很快扩展到 Android 和 Mac 应用。
   - 鼓励用户更新其 iOS 应用以立即开始使用新功能。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1344036572395470940)** (166 条消息🔥🔥): 

> `上下文窗口大小，Comet Browser 发布，语音模式功能，使用 Perplexity 编程，Claude 3.7 Sonnet 幻觉` 


- **上下文窗口奇迹：Perplexity 的 Token 胜利**：用户讨论了上下文窗口大小，指出 Perplexity 默认每次查询读取最多 **4,000 tokens**，而 Pro 订阅者在上传文件并使用 GPT-4 Omni 或 Claude 3.5 Sonnet 时，可以使用高达 **32,000 tokens**。
   - 一位用户的测试表明，使用 **o3-mini** 时限制可能更高，约为 **128k 字符**或 **62k tokens**，尽管宣传的 **100 万 token** 上下文窗口仍然难以触及。
- **Comet 浏览器即将推出**：根据 [AravSrinivas](https://x.com/AravSrinivas/status/1894068996950855747) 的消息，Perplexity 将*很快*推出一款新的 Agent 浏览器 **Comet**。
   - 确切的发布日期和平台支持尚未确认，有推测称可能会在不到一周内发布。
- **语音模式获得升级**：宣布了新的**语音模式**，配备了新 UI，并具备在说话时被中断的能力。
   - 虽然被认为是一项改进，但目前尚不及 **Microsoft Copilot**、**Grok 3** 或 **ChatGPT**，当前版本仍存在一些问题。
- **代码探索：使用 Perplexity 调试**：用户探索了使用 Perplexity 进行编码，一些人建议使用 **Writing 模式**以获得更好的代码响应。
   - 用户注意到了 Agent 能力的成本，特别是在 **Claude** 等平台上，对于大型项目，API 使用费用在一个下午可能达到 **20 美元**。
- **Claude 3.7 Sonnet 遭遇人格分裂**：用户注意到 **Claude 3.7 Sonnet** 有时会错误地将自己识别为 **Claude 3 Opus**。
   - 这可能是由于训练数据和模型命名方式导致的，但已经创建了一个工单来解决此问题，链接见[此处](https://discord.com/channels/1047197230748151888/1343923864492838972/1343923864492838972)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/askperplexity/status/1894227029769310243?s=46">来自 Ask Perplexity (@AskPerplexity) 的推文</a>：嘿！Ask Perplexity 在 X 和其他在线社区回答您的问题。其工作原理如下：1. 在任何帖子中标记 @AskPerplexity 2. 提出您对该帖子或其回复的任何问题 3. 提问...</li><li><a href="https://x.com/OpenAI/status/1894454196986155130">来自 OpenAI (@OpenAI) 的推文</a>：首先，Plus、Team、Enterprise 和 Edu 用户每月将拥有 10 次深度研究（deep research）查询。Pro 用户现在每月将拥有 120 次深度研究查询。</li><li><a href="https://chromewebstore.google.com/detail/aahklphdncmbmkmcbgdglefnlfmeegjj?utm_source=item-share-cp)">Instagram 自动播放修复 - Chrome 网上应用店</a>：在切换标签页或窗口时保持 Instagram 视频播放</li><li><a href="https://x.com/AravSrinivas/status/1894068996950855747">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：Perplexity 将很快推出一款新的 Agent 浏览器：Comet！</li><li><a href="https://www.youtube.com/watch?v=oEfB2GxAh9k&t=425s"> - YouTube</a>：未找到描述</li><li><a href="https://www.reddit.com/r/aipromptprogramming/s/My6VlFqtJE">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://sonar.perplexity.ai/.">Sonar by Perplexity</a>：使用由 Perplexity 创建的最佳 AI 问答引擎 API 进行构建。通过搜索接地（grounding）功能，以市面上最快、最便宜的产品为您的产品提供动力。提供无与伦比的实时、全网范围的重...</li><li><a href="https://monnef.gitlab.io/by-ai/2025/pplx-tech-props">Perplexity Tech Props</a>：未找到描述</li><li><a href="https://monnef.gitlab.io/by-ai/2025/pplx_M_ctx">Perplexity 上的百万上下文窗口？</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1344051558064849068)** (8 条消息🔥): 

> `通过 Perplexity 生成 Ruby 脚本、Anthropic 的 Pokemon AI 基准测试、特朗普解雇军事领导人新闻、Meta 的 2000 亿 AI 算力投资` 


- **Perplexity 生成 Ruby 脚本**：一位用户利用 Perplexity AI 生成了一个简单的 **Ruby 脚本**，对结果表示满意，并指出 AI 非常擅长此类任务，相关上下文见 [此处](https://www.perplexity.ai/search/ruby-script-to-find-all-the-fi-nSnE8_UkRmy3RphdnePeDQ)。
- **Anthropic 的 Pokemon AI 基准测试**：Perplexity AI 分享了关于 **Anthropic 的 Pokemon AI 基准测试** 的新闻，详见 [此处](https://www.perplexity.ai/page/anthropic-s-pokemon-ai-benchma-BfIkUdgVRKmVAVZLd7.j9w#7fc21566-af56-462f-8f38-b24bb4600dd7)。
- **特朗普解雇军事领导人的新闻**：Perplexity AI 汇总了关于 **特朗普** 解雇军事领导人以及 **Satya Nadella** 驳斥 AGI 基准测试的新闻，详见 [此处](https://www.perplexity.ai/search/amazing-ai-tGqtBK1WT1q4XDUXUAGCiQ)。
- **Meta 探索 2000 亿 AI 算力投资**：Perplexity AI 分享了一条新闻，称 **Meta** 正在探索 **2000 亿 AI 算力** 投资，详见 [此处](https://www.perplexity.ai/page/meta-explores-200-billion-ai-c-Ri33UtReQwaCKzi7OFhg6Q)。



**提到的链接**：<a href="https://www.youtube.com/embed/DpBeRZmEYNs">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1344038613423362130)** (6 条消息): 

> `Perplexity Deep Research API、旧金山开发者见面会、Playground 中的 Sonar Deep Research、通过 API 上传文件` 


- **Perplexity API 投出一记“板球”**：根据 [这条推文](https://x.com/aravsrinivas/status/1894477728222777594?s=61)，Perplexity 正在通过 **Perplexity Sonar API** 向所有开发者开放 **Deep Research API**，以帮助人们构建自定义的研究 Agent 和工作流。
   - 一位用户建议将该 API 应用于所有板球数据和统计，以深入挖掘信息，并寻求优质的统计数据库和 **API credits**。
- **Perplexity 举办旧金山开发者见面会**：根据 [这条 Discord 帖子](https://discord.com/channels/1047197230748151888/1155998520822743091/1344108568508498071)，Perplexity 宣布即将在旧金山举办开发者见面会。
   - 公告鼓励在旧金山使用该 API 构建了酷炫作品的用户在活动中进行 **Demo** 展示，并征求关于下次见面会举办地点的建议。
- **Sonar Deep Research 与 Playground 计划**：一位用户询问 **Sonar Deep Research** 是否会在 Playground 中提供。
   - 讨论中未提供明确答案。
- **寻求城镇会议的自动化文件上传功能**：一位用户询问是否可以通过 **API** 上传文件，以便对其进行总结或处理。
   - 该用户希望自动化总结过去和未来的城镇会议，因为 Web 版本在处理此类持续性任务时不可持续。



**提到的链接**：<a href="https://x.com/aravsrinivas/status/1894477728222777594?s=61">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：作为一个板球迷，将 Perplexity Deep Research API 应用于所有板球数据和统计并深入研究将会非常有趣。谁有好的统计数据库？任何想要构建的人...

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1344044121815388252)** (76 条消息🔥🔥): 

> `Assistants API 文件搜索、Claude 玩宝可梦、Claude Sonnet 网页版 vs API、OpenAI Deep Research、Raycast AI 扩展`

- **OpenAI 为 Assistants API 添加文件搜索功能**：OpenAI 宣布在 **Assistants API** 中[支持 o3-mini 和 o1 的文件搜索](https://platform.openai.com/docs/assistants/tools/file-search)，使助手能够访问并从文档中检索信息。
- **Anthropic 的 Claude 玩宝可梦（Claude Plays Pokémon）获得关注**：**Claude Plays Pokémon** 作为一个研究员的个人项目持续进行，在 [Twitch](http://twitch.tv/claudeplayspokemon) 上直播，研究员 [David Hershey](https://x.com/DavidSHershey/status/1894463660279697852) 参与了该项目。
- **Sonnet API 的回答与网页版存在差异**：根据 [Kimmonismus](https://x.com/kimmonismus/status/1894133480792924249) 的说法，与 API 版本相比，**Claude 3.7 Sonnet** 的网页版拥有更长的包含上下文信息的 System Prompt，这可能导致回答不一致。
- **OpenAI Deep Research 评估**：[Ben Evans](https://www.ben-evans.com/benedictevans/2025/2/17/the-deep-research-problem) 对 **OpenAI 的 Deep Research** 进行了批判性回顾，强调了来源准确性问题，特别是来自 *Statista* 和 *Statcounter* 的数据。
- **Perplexity 推出 5000 万美元种子基金**：[Perplexity](https://tcrn.ch/41xlXfS) 推出了一个 **5000 万美元的种子和前种子期风投基金**，目前已收到 150 亿美元估值的报价。
   - 新的 “Elicit Reports” 被认为是 *Deep Research 的更好版本*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/vibhuuuus/status/1894857843121234313?s=46">来自 Vibhu Sapra (@vibhuuuus) 的推文</a>: @swyx @willccbb @kalomaze 他们发布了这个架构（找不到来源了），但我认为非常值得一读。</li><li><a href="https://x.com/openaidevs/status/1894478106565415328?s=46">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>: 我们在 Assistants API 中增加了对 o3-mini 和 o1 的文件搜索支持。你可以创建助手来访问和检索文档中的信息，这些推理模型特别...</li><li><a href="https://x.com/DavidSHershey/status/1894463660279697852">来自 David Hershey (@DavidSHershey) 的推文</a>: 所以，我做了一件事 🙂 这真的只是一个有趣的小副业——我想花点时间研究 Agent，而宝可梦是我能想到的最有趣的方式。然后它就火了！3....</li><li><a href="https://x.com/matistanis/status/1894824212382257427?s=46">来自 Mati Staniszewski (@matistanis) 的推文</a>: 今天，我们发布了自己的 Speech to Text 模型：ElevenLabs Scribe v1。Scribe 在 FLEURS 和 Common Voice 基准测试中超越了当前最先进的模型，并最终交付了那些令人沮丧的...</li><li><a href="https://www.ben-evans.com/benedictevans/2025/2/17/the-deep-research-problem">Deep Research 问题 — Benedict Evans</a>: OpenAI 的 Deep Research 是为我量身定制的，但我却无法使用它。这是又一个惊人的演示，直到它崩溃。但它崩溃的方式非常有趣。</li><li><a href="https://x.com/willccbb/status/1894477232032076240?s=46">来自 will brown (@willccbb) 的推文</a>: @aryanagxl 多轮工具使用的泛化能力 + 通过递归摘要在极长任务中保持连贯性的能力，对我看到的多数反应来说，这比 ARC-AGI 意义更大...</li><li><a href="https://x.com/hume_ai/status/1894833497824481593?s=46">来自 Hume (@hume_ai) 的推文</a>: 今天，我们发布了 Octave：第一个为 text-to-speech 构建的 LLM。🎨 通过提示词设计任何声音 🎬 提供表演指令来控制情感和表达（讽刺、耳语等）🛠️ 生成...</li><li><a href="https://x.com/btibor91/status/1894686656139325593?s=46">来自 Tibor Blaho (@btibor91) 的推文</a>: 这是真的——Android 版 ChatGPT 应用（1.2025.056 beta）有一个新公告：“尝试 GPT-4.5 研究预览版——Pro 用户现在可以访问我们最新、最大的模型。”引用 Dylan N...</li><li><a href="https://x.com/haileysch__/status/1894422591617790046?t=jABK7mxk5oPmgobo1Bc49g&s=19">来自 Hailey Schoelkopf (@haileysch__) 的推文</a>: 昨天的 2 个误解：- 没有针对玩电子游戏进行训练。Claude 现在就能做到 - “为什么这不是一个演示” 引用 Anthropic (@AnthropicAI) Claude 玩宝可梦作为一项研究继续进行...</li><li><a href="https://x.com/mikeyk/status/1894783669920817321?s=46">来自 Mike Krieger (@mikeyk) 的推文</a>: 今天在纽约参加 Alexa+ 的发布，其底层使用了 Claude 来实现许多新功能。对于我们的团队来说，能与 Alexa 团队合作，基于如何将...的共同愿景，是非常有趣的。</li><li><a href="https://x.com/paulg/status/1894827577325560215?s=46">来自 Paul Graham (@paulg) 的推文</a>: 这是那家初创公司在接下来一年的收入图表（蓝色部分）。引用 Paul Graham (@paulg) 一种新型的指数级收入图表。这家公司正在销售一些有用的...</li><li><a href="https://x.com/youssefish/status/1894548592020353311?s=46">来自 Youssef Ishak (@youssefish) 的推文</a>: (1/) 总结一下“Claude 玩宝可梦”在做什么，主要是为了我自己。Claude 的任务是玩宝可梦。在每一次按下按钮之前，Claude 都会参考它之前的上下文和...</li><li><a href="https://x.com/elicitorg/status/1894772293752266846?s=46">来自 Elicit (@elicitorg) 的推文</a>: 我们筹集了 2200 万美元的 A 轮融资，并推出了 Elicit Reports，这是为真正的研究人员准备的更好版本的 Deep Research。Elicit Reports 现在可供所有人免费试用。👇</li><li><a href="https://x.com/willccbb/status/1894478848923701275?s=46">来自 will brown (@willccbb) 的推文</a>: @kalomaze 是的，而且“模型可以自我总结状态来处理远超其上下文长度的事情”简直太疯狂了。感觉很像 R1，这是一个显而易见的技巧，以前行不通是因为模型只是...</li><li><a href="https://x.com/TechCrunch/status/1894514805890830664">来自 TechCrunch (@TechCrunch) 的推文</a>: Perplexity 推出 5000 万美元的种子轮和预种子轮风险投资基金 https://tcrn.ch/41xlXfS</li><li><a href="https://x.com/janonacct/status/1894437873082143222?s=46">来自 janon (@janonacct) 的推文</a>: Claude 给它的对手起名叫 “Waclaude”</li><li><a href="https://x.com/threepointone/status/1894399506277376369?s=46">来自 sunil pai (@threepointone) 的推文</a>: Cloudflare agents：代码库：https://github.com/cloudflare/agents/ 平台文档：https://developers.cloudflare.com/agents/ 入门套件：https://github.com/cloudflare/agents-starter 还有更多精彩即将到来，正在发布...</li><li>

><a href="https://x.com/weberwongwong/status/1894794612398792974?s=46">来自 weber (@weberwongwong) 的推文</a>：介绍 FLORA，你的智能画布。每一个创意 AI 工具都经过深思熟虑地连接在一起。</li><li><a href="https://x.com/prerationalist/status/1894449418813776183?s=46">来自 prerat (@prerationalist) 的推文</a>：天哪，Claude 给他的对手起名叫 WACLAUD??!?! 引用 Anthropic (@AnthropicAI)：Claude 玩宝可梦（Claude Plays Pokémon）将作为研究员的个人项目继续进行。在 Twitch 上关注：http://twitch.tv/claudeplayspok...</li><li><a href="https://x.com/allen_ai/status/1894415487569969211?s=46">来自 Ai2 (@allen_ai) 的推文</a>：介绍 olmOCR，我们的开源工具，用于从 PDF 中提取干净的纯文本！olmOCR 为大规模应用而生，能够以高吞吐量处理多种文档类型。在您自己的 GPU 上免费运行——速度超过 3000...</li><li><a href="https://x.com/anthropicai/status/1894798008623026503?s=46">来自 Anthropic (@AnthropicAI) 的推文</a>：Claude 将助力亚马逊的下一代 AI 助手 Alexa+。亚马逊和 Anthropic 在过去一年中紧密合作，由 @mikeyk 领导的团队帮助亚马逊获得了...</li><li><a href="https://x.com/levelsio/status/1894848949082825176?s=46">来自 @levelsio (@levelsio) 的推文</a>：我觉得有 5000 人在飞，但我也看到了一些机器人 😅 引用 Thomas Slabbers (@Thomasslabbers)：这纯粹是天才之作——看看现在有多少人在飞！我还发现了火星。Pieter 这可能...</li><li><a href="https://x.com/aravsrinivas/status/1894471526449385687?s=46">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：我们正通过 Perplexity Sonar API 向所有开发者提供 Deep Research 端点，以帮助人们构建自定义的研究 Agent 和工作流！很高兴看到人们将要...</li><li><a href="https://x.com/steph_palazzolo/status/1894785791018332505?s=46">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：今天早上的议程有很多内容：- OpenAI 领导层已告知员工 GPT-4.5 将于本周发布 - Perplexity 正收到风投机构 150 亿美元的入股邀约。它可能不会接受该报价，但这突显了风投...</li><li><a href="https://x.com/kimmonismus/status/1894133480792924249">来自 Chubby♨️ (@kimmonismus) 的推文</a>：Claude 3.7 Sonnet 系统提示词（system prompt）：“助手是 Claude，由 Anthropic 创建。当前日期是 {{currentDateTime}}。Claude 乐于帮助人类，并将其角色视为一个智能且善良的助手...”</li><li><a href="https://x.com/ritakozlov_/status/1894394140764594676?s=46">来自 rita kozlov 🐀 (@ritakozlov_) 的推文</a>：npm i agents-sdk https://blog.cloudflare.com/build-ai-agents-on-cloudflare/</li><li><a href="https://x.com/anthropicai/status/1894419042150027701?s=46">来自 Anthropic (@AnthropicAI) 的推文</a>：Claude 玩宝可梦将作为研究员的个人项目继续进行。在 Twitch 上关注：http://twitch.tv/claudeplayspokemon</li><li><a href="https://www.youtube.com/watch?v=sHIlFKKaq0A&t=2s"> - YouTube</a>：未找到描述</li><li><a href="https://x.com/levelsio/status/1894429987006288259">来自 @levelsio (@levelsio) 的推文</a>：成功了！！一个完整的多人游戏，使用 Python websockets 服务器，每 100 毫秒（每秒 10 次）接收并广播所有玩家位置。所有代码几乎 100% 由 AI 通过 Cursor 和 Grok 编写...</li><li><a href="https://www.youtube.com/watch?v=1B9i7FBsRVQ">Claude 玩 Minecraft！</a>：不再只是聊天机器人了！今天不行，谢谢！想象一个 AI 不仅仅是聊天，而是思考、解决和行动的世界。进入 Minecraft 的领域，在那里...</li><li><a href="https://news.ycombinator.com/item?id=43174910">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1344121891316564090)** (2 条消息): 

> `LLM Paper Club, Raycast AI` 


- **LLM Paper Club 更新**：明天的 LLM Paper Club 日程已更新，您可以在[这里](https://lu.ma/y3v27e0k)报名。
- **新的 Raycast AI 播客已发布**：一段介绍新 **Raycast AI** 的短播客已在[这里](https://www.youtube.com/watch?v=hoEL6ddVcC0)发布。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=hoEL6ddVcC0">Raycast: 你的 AI 自动化助手</a>：插件能做的任何事情，现在都可以通过自然语言完成。观看发布视频：https://www.youtube.com/watch?v=sHIlFKKaq0A</li><li><a href="https://lu.ma/y3v27e0k">LLM Paper Club (Test Time: s1, Recurrent Depths) · Zoom · Luma</a>：Raphael Kalandadze 将讲解：s1: 简单的测试时缩放（Simple test-time scaling），通过潜推理（Latent Reasoning）扩展测试时计算：一种循环深度方法（Recurrent Depth Approach）（如果时间允许）…</li><li><a href="http://Latent.Space)">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1344039290648268820)** (56 messages🔥🔥): 

> `Local LLM training code, Cohere models in OpenAI SDK, Open Source vs Paid Code, OpenAI SDK Integration` 


- **高中生希望攻克 LLM 训练**：一名高中生在经过两年的努力后，声称开发出了具有完全所有权和控制权的 **local LLM training**（本地 LLM 训练）代码，并正在寻找公司购买。
   - 然而，一位社区成员建议，这段代码需要与现有的 **open-source**（开源）解决方案（如 *llama factory* 和 *Unsloth*）竞争才具有可行性。
- **开源倡导者建议不要出售**：一位社区成员认为，如果代码不能超越现有的**免费开源替代方案**（如 [Unsloth](https://github.com/unslothai/unsloth)），人们就没有动力为其付费。
   - 该高中生最终妥协，并宣布计划将其项目 **open source**（开源）。
- **Cohere 模型现已集成至 OpenAI SDK**：一位成员宣布 [Cohere 模型现在可以通过 OpenAI SDK 访问](https://x.com/itssandrakublik/status/1894791769117650998?s=46&t=r1mNPSgnb3pIcbR7vcCi-g)。
   - 正如 [Quickstart Guide](https://docs.cohere.com/docs/compatibility-api) 中详述的那样，此集成支持 streaming（流式传输）、tool calls（工具调用）和 structured outputs（结构化输出）。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/itssandrakublik/status/1894791769117650998?s=46&t=r1mNPSgnb3pIc">Sandra Kublik (@itsSandraKublik) 的推文</a>：你现在可以直接通过 OpenAI SDK 访问 Cohere 模型了 :) 查看我们的 Python、TS 和 cURL 演示快速入门指南，此外还支持 streaming、tool calls、structured outputs 等。祝开发愉快...</li><li><a href="https://x.com/itssandrakublik/status/1894791769117650998?s=46&t=r1mNPSgnb3pIcbR7vcCi-g">Sandra Kublik (@itsSandraKublik) 的推文</a>：你现在可以直接通过 OpenAI SDK 访问 Cohere 模型了 :) 查看我们的 Python、TS 和 cURL 演示快速入门指南，此外还支持 streaming、tool calls、structured outputs 等。祝开发愉快...</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥</a>：以 2 倍的速度和减少 70% 的显存微调 Llama 3.3、DeepSeek-R1 和推理 LLM！🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1344348069621399572)** (1 messages): 

> `Compatibility API, OpenAI SDK, Cohere Models` 


- **Cohere 模型现可通过 OpenAI SDK 访问**：新的 **Compatibility API** 镜像了 OpenAI SDK 格式，使得基于 OpenAI 的应用可以无缝切换到 **Cohere 模型**，而无需进行大规模重构，详见[文档](https://docs.cohere.com/docs/compatibility-api)。
   - 要进行切换，用户需要将 base_url 更改为 *https://api.cohere.ai/compatibility/v1* 并设置其 **COHERE_API_KEY**，支持 Python、TypeScript 和 cURL 等多种语言。
- **Compatibility API 支持高级功能**：**Compatibility API** 支持诸如 **structured outputs (JSON Schema)**、**tool calls** 和 **state management**（状态管理）等高级功能。
   - 公告引导用户前往 <#1168578329423642786> 频道进行提问和反馈。



**提及的链接**：<a href="https://docs.cohere.com/docs/compatibility-api">通过 OpenAI SDK 使用 Cohere 模型 — Cohere</a>：该文档作为 Cohere Compatibility API 的指南，允许开发者使用 OpenAI SDK 无缝使用 Cohere 的模型。

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1344339386715607122)** (7 messages): 

> `Cohere API blocking VPS, Token counting changes, Cohere API availability` 


- **Cohere API 拦截 VPS 访问**：一位用户报告称，从 **VPS** 发起的 **Cohere API 调用**被**拦截**，并被建议联系 [support@cohere.com](mailto:support@cohere.com) 寻求帮助。
- **Token 计数修改**：一位成员询问使用 **OpenAI API 的 128K context window**（上下文窗口）将如何影响 Token 计数，因为这比直接使用 Cohere API 时可用的上下文窗口要小。
- **Cohere API 未来可用性**：一位成员询问是否会对 **直接 Cohere API** 进行修改，这可能会影响其未来的可用性。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1344421450064593038)** (3 条消息): 

> `HuggingFace 弃用, RAG 工具` 


- **关于 HuggingFace 弃用标记的讨论**：一位成员询问如何将仓库标记为已弃用，并链接到 **Hugging Face** 上的新版本。
   - 他们随后澄清，弃用功能仅适用于模型，不适用于数据集，从而自行解决了问题。
- **RAG 工具推荐请求**：一位成员询问目前哪种 **RAG** 工具最适合个人用户。
   - 未提及具体的 RAG 工具，因此需要进一步讨论。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1344036738208895099)** (18 条消息🔥): 

> `KV Cache 压缩, Activation Steering, Deepseek DeepGEMM 内核, 数据混合优化` 


- **Tokencat 提供更高的压缩率**：正如 **GoldFinch 论文**中所详述，在 MLA 中加入 'tokencat' 实现了更高的压缩率。
   - 这种压缩依赖于跨层的其他维度压缩。
- **Deepseek 的 DeepGEMM 内核具有开创性**：阅读 Deepseek 新发布的 **DeepGEMM** 的成员表示其令人印象深刻，并指出它考虑了 **H800 限制**，并在带宽和计算限制内优化了效率。
   - 还有人指出，这是一个广泛利用 **TMA** 的开源 GEMM 内核。
- **硬件仍是终极“护城河”**：普遍观点认为，像 **MLA**、**DeepGEMM** 这样的架构内核或像 **DeepEP** 这样的通信策略的高效实现并不能提供显著的竞争优势。
   - 一位成员调侃道：*唯一的护城河是硬件*。
- **用于最优数据混合的 MixMin 算法**：[MixMin 论文](https://arxiv.org/abs/2502.10510) 介绍了一种名为 **MixMin** 的基于梯度的算法，用于优化机器学习流水线中的数据混合，将其构建为一个双层凸目标 (bi-level convex objective)。
   - 论文声称，**MixMin** 在语言建模和化学任务中改进了数据混合，且仅使用了不到 **0.2%** 的额外计算量。



**提到的链接**：<a href="https://arxiv.org/abs/2502.10510">MixMin: Finding Data Mixtures via Convex Minimization</a>：现代机器学习流水线越来越多地结合和混合来自不同且分散来源的数据，例如预训练大型语言模型。然而，寻找最佳数据混合是一个挑战...

  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1344166712114810911)** (2 条消息): 

> `大模型, 集成 (Ensembling), Flops` 


- **大模型是 Flop 重量级选手**：对于相同数量的样本，更大的模型需要更多的 **Flops** 并完成更多的实际“工作”。
   - 将一组模型作为一个模型同时进行集成训练会消耗更多的 **Flops**，但在相同样本量下具有更好的损失/准确率。
- **集成收益随模型增大而缩小**：有人询问 **集成 (ensembling)** 的收益是否会随着模型变大而缩小。
   - 他们假设大模型内部已经包含了集成，如果是这种情况，那么他们认为集成收益确实会随着模型变大而缩小。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1344209278604410901)** (5 条消息): 

> `SAEs, SAEs 中的权重共享, SAEs 中的正交特征` 


- **权重共享 (Weight Tying) 在 SAEs 中不再流行**：成员们讨论了在最新的 **稀疏自编码器 (SAEs)** 中通常不再进行权重共享。
   - 他们指出，假设特征大致是**正交的**，那么输入和输出权重应该是相同的，类似于乘以一个正交矩阵及其转置。
- **正交特征影响 SAE 权重**：对话强调了**正交特征**在确定 **SAEs** 权重中的重要性。
   - 如果假设特征大致正交，输入和输出权重应该对齐，类似于正交矩阵及其转置如何还原回原始状态。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1344320511022334033)** (27 messages🔥): 

> `lm-evaluation-harness 在 notebook 中的设置，通过 TRT 运行的本地 LLM API 端点，GPQA 实现` 


- **Notebook 设置帮助已送达！**: 一位成员询问如何在 notebook 中运行 lm-evaluation-harness 而不是通过命令行，另一位成员表示可以在 notebook 中通过在命令前加 `!` 来运行任何命令行命令。
   - 还有一个（文档较少的）Python API，并建议将 `python main.py` 替换为 `!lm_eval`。
- **TemplateAPI TRT Triton 故障？**: 一位成员询问了关于支持通过 TRT 运行本地 LLM API 端点进行 Completions 的问题，并提到了向 `localhost:8000/v2/models/triton_model/generate` 发送 `curl` 请求。
   - 另一位成员回应称，如果修改 [openai_completions.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/af2d2f3e79140ae5b6833ce1046f1519dc08b9df/lm_eval/models/openai_completions.py#L44) 文件中的 `_create_payload` 以及解析 logprobs 和 generation 的函数，它应该可以工作。
- **GPQA 实现受到质疑**: 一位成员在查看 Open LLM Leaderboard 后，询问了 GPQA 的实现以及是否经过测试。
   - 另一位成员提到排行榜使用的是多选题变体（MMLU 风格，但带括号的字母如 (A), (B) 等），并且只使用了 [GPQA dataset](https://huggingface.co/datasets/Idavidrein/gpqa?row=6) 中包含 200 行的 diamond 子集。
- **GPQA Diamond 结果分析**: 在注意到低分报告后，成员们调查了 GPQA diamond 子集并讨论了潜在的 tokenization 问题，指出问题难度很大且选项看起来很相似。
   - 一位成员报告称，在指示模型以 *the best answer is [A-D]* 结尾后，通过提取最后的 [A-D] 获得了 **0.4848**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/Idavidrein/gpqa?row=6">Idavidrein/gpqa · Hugging Face 数据集</a>: 无描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/af2d2f3e79140ae5b6833ce1046f1519dc08b9df/lm_eval/models/openai_completions.py#L44)">EleutherAI/lm-evaluation-harness 中的 openai_completions.py</a>: 一个用于语言模型 few-shot 评估的框架。
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1344052529477976126)** (3 messages): 

> `NeoX 中的 GQA，Llama 模型导出问题` 


- **GQA 在 GPT-NeoX 中导致故障？**: 一位成员询问在 NeoX 中导出带有 **GQA** 的 **Llama 模型** 是否存在已知问题，并表示模型在使用 **GQA** 时会崩溃，但不使用时运行完美。
   - 他们想知道导出脚本是否需要更改，并链接到了 [GitHub 上的相关 pull request](https://github.com/EleutherAI/gpt-neox/pull/1315/files)。
- **关于 GQA 实现的推测**: 该成员推测故障的根本原因可能与 Grouped Query Attention 的实现有关。
   - 这可能是也可能不是实际原因。



**提及的链接**: <a href="https://github.com/EleutherAI/gpt-neox/pull/1315/files">由 tiandeyu-cs 修复 GQA 问题 (#1314) · Pull Request #1315 · EleutherAI/gpt-neox</a>: 修复 GQA 问题 (#1314)，不要创建虚假的 head dim，直接将 'mixed_x_layer' 分解为 QKV 层。

  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1344380379687030876)** (5 messages): 

> `Modular MAX 和 Mojo 仓库变更，Mojo 的独立语言地位` 


- **Modular 简化 MAX 和 Mojo 仓库**: Modular 正在简化其 **MAX** 和 **Mojo** 的仓库结构，以便于对文档和标准库做出贡献，并集中处理错误报告和功能请求，详情见 [此论坛帖子](https://forum.modular.com/t/upcoming-changes-to-our-github-repositories/648)。
- **社区质疑 Mojo 的语言地位**: 一位社区成员询问仓库变更是否预示着内部不再将 **Mojo** 视为或优先视为一种独立的语言。



**提及的链接**: <a href="https://forum.modular.com/t/upcoming-changes-to-our-github-repositories/648">即将到来的 GitHub 仓库变更</a>: 明天（2月27日），我们将精简我们的 GitHub 仓库！max 仓库将合并到 mojo 仓库中，将所有内容整合在一起。一个新的子目录将存放 Mojo 标准库...

  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1344040906013151293)** (44 条消息🔥): 

> `EmberJSON, Mojo 自动并行化, algorithm 包未开源, 使用 list.get_unsafe 加速, Mojo 中的智能迭代器` 


- **Modular 社区频道落后于 EmberJSON 补丁更新**：尽管更新后的 recipe 很久以前就已合并，但 Modular 社区频道仍未提供 *emberjson* 的 **25.1 补丁**。
   - 用户遇到了其 **Mojo 版本** 与 *emberjson* 预期版本不匹配的问题。
- **Mojo 不会自动并行化**：Mojo 编译器中没有 **自动并行化 (auto-parallelization)**；开发者必须显式使用 **stdlib** 来并行化任务。
   - 用户询问了如何为 Mojo 程序利用所有系统资源（多核 CPU），但目前必须进行显式并行化。
- **Algorithm 包依然保持神秘**：*algorithm 包* 尚未开源，且在 **stdlib 仓库** 中不可见。
- **智能指针与迭代器健全性**：关于智能指针及其使 C++ 像 **Circle** 或 **Rust** 一样安全的潜力的讨论，链接到一篇讨论 [智能指针](https://jacko.io/smart_pointers.html) 的博客文章。
   - 一位成员询问了 Mojo 中是否会有健全的迭代器，以及是否可能实现 **Safe Rust** 中处理的迭代器失效 (iterator invalidation) 保护，特别是针对涉及集合中对象交换的算法。
- **MLIR Dialect 文档匮乏**：Mojo 使用各种 **MLIR dialects** (kgen, pop, lit 等)，它们拥有自己的 ops 和 types，但大多数都没有文档，也没有在 stdlib 中使用或加载到 Mojo 运行时的 MLIR context 中。
   - 缺乏文档是因为这些 dialects 是 stdlib、MAX 和编译器共享的 **私有契约 (private contract)** 的一部分，它们可能未经过充分测试、具有不稳定的 API，或包含专有的增值内容。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://jacko.io/smart_pointers.html">Smart Pointers Can't Solve Use-After-Free</a>：未找到描述</li><li><a href="https://github.com/axiomhq/mojo-hyperloglog">GitHub - axiomhq/mojo-hyperloglog</a>：通过创建账户为 axiomhq/mojo-hyperloglog 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1344042356307001385)** (35 messages🔥): 

> `Alignment tradeoff, DTMF, Google Experiments, Apple speech-to-text Trump issue, Claude 3.7` 


- **对齐努力导致其他地方产生偏差 (Alignment Efforts Cause Bias Elsewhere)**：成员们讨论了调整模型以偏向某种行为如何导致**其他地方的失配 (misalignment)**，这体现了**对齐权衡 (alignment tradeoff)** 问题。
   - 有人提到*对齐总是相对的*，反映了数据中的偏差以及模型控制者所强加的价值观。
- **解码双音多频信号 (Decoding Dual-Tone Multi-Frequency Signaling)**：一位用户分享了一个 [Wikipedia 链接](https://en.wikipedia.org/wiki/DTMF) 来解释 **双音多频 (DTMF) 信号**，这是一种在电话线上使用音频频带的电信系统。
   - 该链接提供了关于 **DTMF 在电信领域的历史、技术规范和应用** 的详细信息。
- **Google 不成熟的实现 (Google's Half-Baked Implementations)**：成员们讨论了 [Google](https://learning.google.com/experiments/learn-about?src=signup) 虽然有很多伟大的创意，但往往受困于**糟糕或不完整的实现**。
   - 理论上认为，这可能源于其长期以来主要为内部使用开发工具的历史，这阻碍了他们近年来创建广泛有用产品的能力。
- **Apple 的 Speech-to-Text 将 “Racist” 误打为 “Trump”**：[Apple](https://www.bbc.com/news/articles/c5ymvjjqzmeo) 正在努力修复其 **speech-to-text** 工具，此前有用户报告称，当他们说出 “racist” 一词时，该工具会输入 “Trump”。
   - 一位专家认为，这个问题很可能是由于有人故意修改了底层软件，而不是一个合理的语音识别错误。
- **AI 模型向柏拉图式表征收敛 (AI Model Convergence Towards Platonic Representation)**：讨论集中在一篇论文 ([https://arxiv.org/html/2405.07987v1/](https://arxiv.org/html/2405.07987v1/)) 上，该论文声称 AI 模型的表征正在**向一个共享的现实统计模型收敛**，称之为**柏拉图式表征 (platonic representation)**。
   - 现场有人对非确定性系统如何具有确定性表现并收敛表示怀疑，并质疑代表意义的模型是否准确地反映了现实。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://learning.google.com/experiments/learn-about?src=signup">Learn About</a>: 未找到描述</li><li><a href="https://www.bbc.com/news/articles/c5ymvjjqzmeo">Apple AI tool transcribed the word &#x27;racist&#x27; as &#x27;Trump&#x27;</a>: 专家质疑公司关于这两个词相似的解释。</li><li><a href="https://www.tiktok.com/@user9586420191789/video/7472830639327366446)">TikTok - Make Your Day</a>: 未找到描述</li><li><a href="https://arxiv.org/html/2405.07987v1/">The Platonic Representation Hypothesis</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/DTMF">DTMF - Wikipedia</a>: 未找到描述</li><li><a href="https://www.itu.int/rec/T-REC-Q.23/en)">Q.23:Technical features of push-button telephone sets</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1344105412617568288)** (5 条消息): 

> `LIMO, Speculative Decoding, ipfs_accelerate_py` 


- **LIMO 揭示了推理任务中“少即是多”**：论文 [LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387) 观察到，在较少量的推理数据上进行训练，其效果优于使用其他微调（fine tuning）任务中常见的典型数据量。
   - 该研究开始探讨为什么推理训练对低数据量的需求已被多次发现，尽管目前还没有关于其原因的太多假设。
- **通过 Speculative Decoding 加速生成**：[Speculative decoding](https://lmstudio.ai/docs/advanced/speculative-decoding) 是一种可以显著提高大语言模型 (**LLMs**) 生成速度的技术，它使用一个更小、更快的“草稿”模型（draft model），且不会降低响应质量。
   - 在生成过程中，草稿模型快速提出潜在的 tokens（子词），主模型验证这些内容的速度比从头开始生成要快。
- **使用 Python 加速 IPFS**：[ipfs_accelerate_py](https://github.com/endomorphosis/ipfs_accelerate_py/tree/main/ipfs_accelerate_py) 是一个 Python 工具，用于为 GitHub 上的 endomorphosis 的 IPFS 加速项目做出贡献。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.03387">arXiv reCAPTCHA</a>: 未找到描述</li><li><a href="https://github.com/endomorphosis/ipfs_accelerate_py/tree/main/ipfs_accelerate_py">ipfs_accelerate_py/ipfs_accelerate_py at main · endomorphosis/ipfs_accelerate_py</a>: 通过在 GitHub 上创建账号，为 endomorphosis/ipfs_accelerate_py 的开发做出贡献。</li><li><a href="https://lmstudio.ai/docs/advanced/speculative-decoding">Speculative Decoding | LM Studio Docs</a>: 使用草稿模型加速生成
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1344091159353360456)** (2 条消息): 

> `ChatGPT plugins, Mystery model` 


- **ChatGPT Plugins 获得 Deep Research 功能**：一位用户分享了面向 **ChatGPT Plus** 用户的 **Deep Research** 功能的 [截图](https://cdn.discordapp.com/attachments/853983317044756510/1344091159085191258/Screenshot_2025-02-26_at_00.37.47.jpg?ex=67c04eb0&is=67befd30&hm=e4d1e9e607580413c5c9c25fb98178f5359728d6425b89c4b7222d752246cb0b)。
- **神秘模型浮出水面**：一位用户分享了一张正在流传的**神秘模型**的照片。
   - 未提供更多细节。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1344037998437732372)** (40 条消息🔥): 

> `CSV 索引，ModernBert 模型，Nomic Embed Text V2 部署，GPT4ALL 路线图，文件分割` 


- **巨型 CSV 文件引发索引困扰**：一位成员询问嵌入/索引两个 **277 GB CSV** 文件需要多长时间，据推测这些文件源自最近的 **NPD 数据**泄露。
   - 一位成员建议将文件分割成 **1 GB** 的分块，以便使用像 [GSplit](https://www.gdgsoft.com/gsplit) 这样简单的索引软件更轻松地进行索引。
- **多语言模型 ModernBERT 思考**：一位成员寻求基于 **ModernBERT** 架构训练多语言模型的细节，并链接到了 [ModernBERT GitHub 仓库](https://github.com/AnswerDotAI/ModernBERT)。
   - 他们澄清说，自己特别感兴趣的是 NomicAI 微调后的模型，例如 [nomic-embed-text-v2](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-unsupervised)。
- **Nomic Embed V2：尚无 Ollama 官方消息**：一位成员询问 **Nomic Embed Text V2** 何时能在 **Ollama/GPT4ALL** 中部署，并表示倾向于不依赖编程技能的部署方式。
   - 另一位成员提到了最近在 [Nomic AI 博客](https://www.nomic.ai/blog/posts/nomic-embed-text-v2)上发布的 **Nomic Embed Text V2** 公告。
- **GPT4ALL 获得受 Gemini 启发的建议**：一位成员请求提供未来 **GPT4ALL** 更新的路线图，特别是类似于 **Google Gemini** 的“实时模式（LIVE mode）”。
   - 另一位成员建议增加语音识别 **STT** 和 **TTS** 输出，并链接了一个关于创建 **GPT4ALL** 语音助手的 [YouTube 教程](https://www.youtube.com/watch?v=6zAk0KHmiGw)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.nomic.ai/blog/posts/nomic-embed-text-v2">Nomic Embed Text V2: An Open Source, Multilingual, Mixture-of-Experts Embedding Model</a>：Nomic 通过多语言混合专家（Mixture-of-Experts）嵌入模型推进了该领域的最前沿技术</li><li><a href="https://github.com/AnswerDotAI/ModernBERT">GitHub - AnswerDotAI/ModernBERT: Bringing BERT into modernity via both architecture changes and scaling</a>：通过架构变更和扩展使 BERT 走向现代化 - AnswerDotAI/ModernBERT</li><li><a href="https://www.youtube.com/watch?v=6zAk0KHmiGw">Create a GPT4ALL Voice Assistant in 10 minutes</a>：使用 Python 编写本地 GPT 语音助手。在本视频中，我们将学习如何在没有互联网连接的情况下运行 OpenAI Whisper，以及后台语音检测...</li><li><a href="https://github.com/nomic-ai/contrastors">GitHub - nomic-ai/contrastors: Train Models Contrastively in Pytorch</a>：在 Pytorch 中以对比方式训练模型。通过在 GitHub 上创建账号为 nomic-ai/contrastors 的开发做出贡献。</li><li><a href="https://www.gdgsoft.com/gsplit">GSplit - Free File Splitter - Split Any File Fast - Split Text and Log Files</a>：GSplit 是一款免费的文件分割器，可将任何文件分割成称为“碎片”的小文件。快速、易用且高效，具有大量自定义选项。</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-unsupervised">nomic-ai/nomic-embed-text-v2-moe-unsupervised · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1344037797782229103)** (23 条消息🔥): 

> `Claude Code line numbers, Model Context Protocol (MCP), MCP Server Implementation, SSE server` 


- **Claude Code 使用行号进行精确编辑**：成员提到 **Claude Code** 在读取文件时会包含每一行的行号，这提高了代码编辑的可靠性，并减少了 [mcp-language-server](https://github.com/isaacphi/mcp-language-server) 等项目中的 Context 使用量。
   - 一位成员指出，行号对于自动调试器至关重要，能够实现精确的断点设置，并与 **Pylance** 等工具集成。
- **MCP Server 实现结果褒贬不一**：一位成员分享了他们构建自定义 **MCP servers** 并使用本地 LLMs (**Mistral** 和 **Llama3.1**) 与 [mcp-cli](https://github.com/chrishayuk/mcp-cli) 集成的实验。
   - 虽然 **Llama3.1** 最初在工具使用方面过于激进，但 **Mistral** 随后开始出现工具调用的幻觉（Hallucination），而不是实际调用它们。
- **MCP 所有权澄清**：当被问及 *谁拥有 MCP？是 Anthropic 吗？* 时，澄清了 MCP 目前是由 **Anthropic** 推动的**开源项目**。
   - 长期愿景涉及如 [此 GitHub 讨论](https://github.com/orgs/modelcontextprotocol/discussions/133#discussioncomment-11773450) 中所述的公正基金会/委员会管理。
- **找到 SSE 测试服务器解决方案**：一位正在寻找类似于 stdio 的 everything server、适用于测试的 **SSE server** 的成员得到了有用的建议。
   - 有人指出，**官方 everything server** 具备 SSE 功能，使其成为测试用途的理想选择。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/chrishayuk/mcp-cli">GitHub - chrishayuk/mcp-cli</a>：通过在 GitHub 上创建账户来为 chrishayuk/mcp-cli 的开发做出贡献。</li><li><a href="https://github.com/isaacphi/mcp-language-server">GitHub - isaacphi/mcp-language-server: 与 Language Server 交互的 Model Context Protocol (MCP) server</a>：与 Language Server 交互的 Model Context Protocol (MCP) server - isaacphi/mcp-language-server</li><li><a href="https://github.com/orgs/modelcontextprotocol/discussions/133#discussioncomment-11773450)">社区治理 / 开放基金会创建或捐赠 · modelcontextprotocol · Discussion #133</a>：提交前检查清单 我已确认这作为特定仓库的功能请求并不合适 我已搜索现有讨论以避免重复 您的想法 MCP 是 e...
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1344110124108615681)** (3 条消息): 

> `FastMCP, Typescript, Custom Authentication` 


- **FastMCP 修复了棘手的竞态条件**：[FastMCP](https://github.com/punkpeye/fastmcp)（一个用于构建 MCP servers 的 **TypeScript 框架**）的用户被敦促升级到最新版本，以修复一些 *棘手的竞态条件（race conditions）*。
   - 强烈建议进行升级，以确保使用该框架的应用程序的稳定性和可靠性。
- **FastMCP 获得高级身份验证功能**：**FastMCP** 现在支持 [自定义身份验证](https://github.com/punkpeye/fastmcp/releases/tag/v1.20.0)，允许开发人员使用自定义函数对 SSE 客户端进行身份验证。
   - 这一增强功能为保护 **MCP servers** 提供了更多的控制权和灵活性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/punkpeye/fastmcp/releases/tag/v1.20.0">Release v1.20.0 · punkpeye/fastmcp</a>：1.20.0 (2025-02-26) FastMCP 现在允许您使用自定义函数对 SSE 客户端进行身份验证：import { AuthError } from &quot;fastmcp&quot;; const server = new FastMCP({ name: &quot;My Server&quot;, ...</li><li><a href="https://github.com/punkpeye/fastmcp">GitHub - punkpeye/fastmcp: 一个用于构建 MCP servers 的 TypeScript 框架。</a>：一个用于构建 MCP servers 的 TypeScript 框架。通过在 GitHub 上创建账户来为 punkpeye/fastmcp 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1344356321595756544)** (19 条消息🔥): 

> `StatefulDataLoader, single device recipes, truncation and skipping` 


- **`StatefulDataLoader` 推广正式开启**：成员们讨论了将 [`StatefulDataloader`](https://github.com/pytorch/torchtune/issues/2439) 的使用推广到 TorchTune 中的所有 recipes，以便成功设置**基于步数的 Checkpointing**并跟踪/存储 dataloader 状态。
   - 确定欢迎提交多个 PR，成员们自愿承担单设备 recipes 的工作，从 *lora_dpo_single_device* 和 *knowledge_distillation_single_device* 开始。
- **单设备 Recipes，MPS Backend 获准使用**：在处理 [将 `StatefulDataloader` 添加到其余 recipes](https://github.com/pytorch/torchtune/issues/2439) 任务时，一名成员询问单设备 recipes 是否可以依赖 **MPS backend**，得到的回复是“是的，没问题”。
   - 一名成员自愿先开始处理一个，以避免阻塞父级 Issue 的关闭。
- **请求 `truncation and skipping` 的 CI 协助**：一名成员请求在另一名成员离线期间，为 [PR 2419](https://github.com/pytorch/torchtune/pull/2419) 启动 CI 而不进行合并。
   - 他们提到这是当天的最后一次尝试。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/pull/2441">[WIP] Add `StatefulDataLoader` to all recipes except knowledge_single by krammnic · Pull Request #2441 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档还是其他（请在此处添加）？请链接此 PR 解决的任何 Issue。#2431 #2439...</li><li><a href="https://github.com/pytorch/torchtune/issues/2439">Add `StatefulDataloader` to remainder of recipes · Issue #2439 · pytorch/torchtune</a>：目标：将 StatefulDataloader 的使用推广到 torchtune 中的所有 recipes。原因：为了成功设置基于步数的 Checkpointing，我们需要能够跟踪和存储 dataloader 状态。如何...</li><li><a href="https://github.com/pytorch/torchtune/pull/2442">[WIP] add `StatefulDataLoader` to `knowledge_distillation_single_device` recipe by jxtngx · Pull Request #2442 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档还是其他（请在此处添加）？请链接此 PR 解决的任何 Issue。#2439 Chang...</li><li><a href="https://github.com/pytorch/torchtune/pull/2419">[RFC] truncation and skipping by krammnic · Pull Request #2419 · pytorch/torchtune</a>：#2344 提到了与我们的数据加载和处理相关的两个要点。此 RFC 致力于这两个方面。Truncation：目前，我们不支持左右两侧的截断....
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1344038314931523686)** (15 messages🔥): 

> `VRAM Efficiency, AI HackXelerator, Scammer Alert, Regional prompting` 


- **Hunyuanvideogp V5 突破了 VRAM 定律？**: 一位成员分享了一篇关于 [Hunyuanvideogp V5 突破 VRAM 定律](https://www.reddit.com/r/StableDiffusion/comments/1iybxwt/hunyuanvideogp_v5_breaks_the_laws_of_vram/) 的 Reddit 帖子。
   - 另一位成员评论道，*它实际上并没有突破 VRAM 定律，只是更有效地利用了 VRAM*，并根据公式 **Width * Height * FPS * Length** 计算出数值低于 VRAM 容量。
- **伦敦、巴黎、柏林 AI HackXelerator 发布**: **London, Paris, Berlin AI HackXelerator™ - LPB25** 将社区驱动的黑客松趣味性与全面加速器的深度和支持相结合，将于 **2025 年 4 月 5 日至 25 日**举行 ([kxsb.org](https://www.kxsb.org/lpb25))。
   - 该活动将汇集 **500 名创意人士、开发者和设计师**，并设有 **AI 音乐、图像、视频、时尚、游戏**及其组合的赛道，由 **Central Saint Martins, Station F, Mistral AI, Hugging Face, Luma AI, Vultr, AMD, Nvidia** 等品牌支持。
- **诈骗警报！**: 一位成员举报 `@w361_emp 是诈骗者`，并称 *他偷走了我的作品集*。
   - 该成员请求其他人保持警惕。
- **使用 LoRAs 进行区域提示 (Regional prompting) 是可行的**: 一位成员询问是否可以在图像的特定区域使用 LoRAs，例如 *仅在嘴部使用兽人 (orc) LoRA*。
   - 另一位成员建议搜索 **regional prompting**，并指出该功能已在某个时间点添加到 ComfyUI 中。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.kxsb.org/lpb25">KXSB 举办的 London-Paris-Berlin HackXelerator™</a>: 加入 LPB25，这是一个为期 20 天的 AI HackXelerator™，汇集了伦敦、巴黎和柏林的 500 多名创作者。通过音乐、艺术、电影、游戏和时尚探索 GenAI 创新，并提供专家指导和奖项。...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1iybxwt/hunyuanvideogp_v5_breaks_the_laws_of_vram/">Reddit - 深入探索</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1344138762451419218)** (5 messages): 

> `good PRs for new people, TestSpeed.test_sum Performance issues, arange GROUP optimization, BEAM search adjustments` 


- **Tinygrad 欢迎新贡献者**: 有一些 [适合新人的优质 PR (good first PRs)](https://github.com/tinygrad/tinygrad/issues/9262) 可供新贡献者使用，其中一些相对简单。
   - 该 Issue 涉及向 tensor.py 添加方法，如 **as_strided**、**topk** 和 **bitwise_xor**。
- **TestSpeed.test_sum 面临性能瓶颈**: 一位成员报告在处理 `TestSpeed.test_sum` 时遇到困难，并进行了更改使 **GROUP 操作的 AST** 变得合理，但在 BEAM 搜索无法找到针对较大 Tensor 的优化时遇到了障碍。
   - 问题在于 **BEAM 搜索** 没有探索四个连续 **OptOps** 的选项，而这是优化 (4096,4096) Tensor 所必需的，因为仅前三个操作就非常缓慢。
- **Arange Group 优化导致 CI 失败**: **arange GROUP 优化** 未被应用，导致 arange 操作出现额外的内层循环并使 arange 测试失败。
   - 该成员正在寻求建议，是调整 **BEAM 搜索**，还是在何处添加用于水平加法 (horizontal adds) 或循环展开 (loop unrolling) 的新模式。
- **BEAM 搜索策略受到质疑**: 讨论质疑是否应该投入时间调整 **BEAM 搜索** 以寻找特定的 **OptOps** 来提升性能。
   - 作者还询问了应该在何处实现水平加法或循环展开的新模式，并建议在 lowerer 或 expander 中进行。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/issues/9262">pytorch 后端请求的更改 · Issue #9262 · tinygrad/tinygrad</a>: 需添加到 tensor.py 的方法：as_strided -- https://pytorch.org/docs/stable/generated/torch.as_strided.html topk -- https://pytorch.org/docs/main/generated/torch.topk.html bitwise_xor -- https://pyto...</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9190/files">[Bounty] 通过 josephsweeney 使 TestSpeed.test_sum 在带有 LLVM 的 Mac 上显示为黄色 · Pull Request #9190 · tinygrad/tinygrad</a>: 为了实现这一点，我在没有局部变量的设备（CLANG 和 LLVM）上启用了 GROUP OptOps，只需添加一个额外的 reduce 而不是发射局部变量。其他必要的更改来自...
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1344074858258104362)** (7 messages): 

> `UOp Signatures, safetensors computation graphs, TestLinearizerFailures` 


- **社区寻求关于 UOp 签名的指导**：一位社区成员正在寻求帮助，以理解如何确定每个 **UOp** 的 `src` 和 `args` 的签名，包括寻找定义 **Enums** 之间约束条件的文档或代码引用。
   - 他们发现 [spec.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/codegen/linearizer/specs.py) 文件对于详细的 **Op** 描述和含义来说不够充分，并询问 *是否假设它们的含义可以从名称中推断出来？*
- **引发辩论：Safetensors、图与 Pickles？**：一位成员询问关于在 **safetensors** 中编码计算图的问题，提到希望有一种类似于 ONNX 的通用编码约定，但一位社区专家澄清说 *safetensors 不保存计算图，只保存 tensors。*
   - 另一位成员引用了[之前的讨论](https://discord.com/channels/1068976834382925865/1070745817025106080/1329504751016083601)，并建议将 **jitted** 函数进行 **pickling**，作为导出/导入计算图的替代方案。
- **在 macOS 上调试 TestLinearizerFailures**：一位社区成员正在调试 [TestLinearizerFailures.test_failure_53](https://github.com/tinygrad/tinygrad/blob/master/test/test_linearizer.py) 并遇到了问题，具体表现为在 macOS 上出现死循环，而这种情况在 NV/Linux 上并未出现。
   - 该问题似乎与在 macOS 上重写 **BLOCK op** 有关，但他们尚未找到足够关于该 **op** 的信息来解决问题。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1344208019764019231)** (2 messages): 

> `Agent Memory, Feedback Mechanism for Agents` 


- **通过简单技巧提升 Agent 记忆**：频道成员讨论了可以使用一个简单的 trick 来改进 **Agent memory**。
   - 据解释，如果 **Agent 拥有 GPT4 访问权限**，它往往能更有效地利用其记忆，并且由于模型更好，其回答质量比使用 **GPT3.5** 时更高。
- **增强 Agent 学习的反馈机制**：频道辩论了 **反馈机制** 对 Agent 改进学习的重要性。
   - 一位成员建议使用 **新的标注工具**，该工具允许收集有关 Agent 结果的反馈。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/)** (1 messages): 

hritabanghosh: https://discord.gg/ETxqXCfh
  

---


---


---


{% else %}


> 完整的频道细分内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整的细分内容，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}