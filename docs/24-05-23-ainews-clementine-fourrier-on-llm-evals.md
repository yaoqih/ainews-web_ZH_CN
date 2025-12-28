---
companies:
- huggingface
- meta-ai-fair
date: '2024-05-23T23:34:22.485002Z'
description: '来自 **Hugging Face** 的 **Clémentine Fourrier** 在 **ICLR** 上展示了与 **Meta**
  合作的 **GAIA** 项目，并分享了关于 **大语言模型（LLM）评估** 方法的见解。


  该博客概述了三种主要的评估方法：

  1. **自动化基准测试 (Automated Benchmarking)**：使用样本输入/输出和特定指标进行评估。

  2. **人类评判 (Human Judges)**：涉及评分和排名，具体方法包括 **Vibe-checks（感官测试）**、**Arena（竞技场）** 以及**系统化标注**。

  3. **模型作为裁判 (Models as Judges)**：利用通用或专业模型进行评估，并指出其存在的偏见。


  面临的挑战包括数据污染、主观性以及评分偏见。这些评估有助于防止模型性能退化、对模型进行排名，并跟踪该领域的进展。'
id: 8f586835-8d9c-4a69-8163-a84dbaf7ee47
models:
- claude-3-opus
original_slug: ainews-to-be-named-4285
people:
- clem_fourrier
title: Clémentine Fourrier 谈 LLM 评估（LLM evals）
topics:
- llm-evaluation
- automated-benchmarking
- human-evaluation
- model-bias
- data-contamination
- elo-ranking
- systematic-annotations
- preference-learning
- evaluation-metrics
- prompt-sensitivity
---

<!-- buttondown-editor-mode: plaintext -->**榜单就是你所需要的一切。**

> 2024年5月22日至5月23日的 AI 新闻。
我们为你检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discords（**380** 个频道和 **5410** 条消息）。
预计节省的阅读时间（以 200wpm 计算）：**551 分钟**。

---

> 针对昨天 [AI Engineer World's Fair](https://buttondown.email/ainews/archive/ainews-the-top-ai-engineer/) 呼吁的特别补充 —— [为无法负担全额门票的人提供奖学金！](https://docs.google.com/forms/d/e/1FAIpQLSdVf9reEpVyzw_sb9cAOtzxORGTEskcb5PTX2X-GPZ_onUtHw/viewform)。更多演讲者公告正在[陆续发布](https://x.com/aidotengineer)。

很多人都知道 [Huggingface 的 Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)，但很少能听到其背后团队的声音。[Clémentine Fourrier 在 ICLR 罕见露面](https://x.com/clefourrier/status/1793301496102068339?utm_source=ainews&utm_medium=email)（共同展示了 [与 Meta 合作的 GAIA](https://arxiv.org/abs/2311.12983)，我们将在即将推出的 ICLR 播客中讨论此内容），现在她带着[一篇关于她对 LLM Evals 看法的博客](https://huggingface.co/blog/clefourrier/llm-evaluation)回归了。

 
![image.png](https://assets.buttondown.email/images/d49e1192-98ed-40ae-8ad7-1771acd9c816.png?w=960&fit=max)
 

对于那些非常接近这一问题的人来说，这可能不是突破性的，但它是该领域最权威人士之一对“技术现状”所做的极佳且易懂的总结。

我们的 TL;DR：进行评估主要有 3 种方式：

- **Automated Benchmarking (自动化基准测试)**
  - 评估由**样本输入/输出集合**（通常将生成的文本与参考答案或多项选择进行比较）和**指标**（用于计算模型分数）组成。
  - 针对特定**任务**
    - 适用于定义非常明确的任务。
    - 常见问题：模型[在多选题评估中倾向于根据选项出现的顺序做出特定选择](https://arxiv.org/abs/2309.03882)，以及生成式评估依赖于归一化，如果设计不当，很容易导致[不公平](https://huggingface.co/blog/open-llm-leaderboard-drop)。
  - 或针对通用**能力**
    - 例如，GSM8K 高中数学题作为“擅长数学”的代理，[独角兽](https://twitter.com/DimitrisPapail/status/1719119242186871275)作为“会画画”的代理。
  - LLM 在自动化基准测试中的得分极易受到 [Prompt 的微小变化](https://huggingface.co/blog/evaluation-structured-outputs)的影响。
  - 最大的问题：数据污染。BigBench 尝试添加“canary string”，但合规性/意识较差。[目前已存在检测污染的工具](https://arxiv.org/abs/2311.06233)，人们也在探索[动态基准测试](https://arxiv.org/abs/2104.14337)，尽管这成本很高。
- **Humans as Judges (人类作为评委)**
  - 通过让模型完成以下任务：1) 给模型提示 (Prompting) 以及 2) **根据指南对模型回答进行评分或对多个输出进行排名**。
  - 比自动化指标更具灵活性。
  - 防止了大多数污染情况。
  - 与人类偏好高度相关。
  - 形式可以是 **Vibe-checks**
    - 大多构成轶事证据，且往往对确认偏误高度敏感。
    - 但像 [Ravenwolf 这样的人非常系统化](https://huggingface.co/blog/wolfram/llm-comparison-test-llama-3)。
  - 或者是 **Arena** (例如 [LMsys](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard))
    - 投票随后汇总为 Elo 排名（比赛排名），以选出哪个模型是“最好的”。
    - 高主观性：很难强制让许多使用宽泛指南的社区成员保持一致的评分标准。
  - 或者是 **systematic annotations (系统化标注)**
    - 向选定的付费标注员提供极其具体的指南，以尽可能消除主观偏见（Scale AI 和其他标注公司）。
    - 仍然昂贵。
    - 仍可能受到人类偏见的影响。
- **Models as Judges (模型作为评委)**
  - 使用通用、高能力的模型。
  - 或者使用专门针对偏好数据进行训练的小型专家模型。
  - 局限性：
    - 在评分时倾向于[偏好自己的输出](https://arxiv.org/abs/2404.13076)。
    - 不擅长[提供一致的分数范围](https://twitter.com/aparnadhinak/status/1748368364395721128)。
    - 与[人类排名并不那么一致](https://arxiv.org/pdf/2308.15812)。
  - 在答案选择中引入了非常微妙且不可解释的偏见。

评估用于防止退化、对模型进行排名，并作为该领域进展的代理指标。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！


{% endif %}


---

# AI Twitter 回顾

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**NVIDIA 财报与股票表现**

- **强劲的财报**：[@nearcyan](https://twitter.com/nearcyan/status/1793379327188562214) 指出 NVIDIA 已连续六个季度超出盈利预期，**去年营收增长 262% 达到 260 亿美元，利润率为 75.5%**。他们还进行了 10:1 的拆股。
- **投资者反应**：[@nearcyan](https://twitter.com/nearcyan/status/1793377805704843431) 分享了一篇关于 NVIDIA 财报的文章，投资者对结果表示满意。**股价在过去一年中上涨了超过 260%**。
- **市值增长**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1793583558633611402) 强调了 NVIDIA 的成功，其**市值增长了 6 倍多，达到 2.3 万亿美元，超越了 Google 和 Amazon**。营收增长了 262%，稀释后 EPS 增长了 600% 以上。

**Mistral AI 模型更新**

- **更快的 LoRA 微调**：[@danielhanchen](https://twitter.com/danielhanchen/status/1793356226006511902) 发布了一个免费的 Colab 笔记本，用于 **Mistral v3，使用 Unsloth AI 可实现 2 倍快的 LoRA 微调**。它在不损失精度的情况下减少了 70% 的 VRAM 占用。
- **Mistral-7B v0.3 更新**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1793373838904025304) 指出 Mistral-7B v0.3 已发布，**词表扩展至 32768，支持 v3 tokenizer 和 function calling**。8x7B 和 8x22B 版本即将推出。
- **🤗 MLX 上的 Mistral v0.3**：[@awnihannun](https://twitter.com/awnihannun/status/1793392487941505348) 分享称 **Mistral v0.3 基础模型已在 🤗 MLX 社区可用，在 M2 Ultra 上使用 4-bit 量化，生成 512 个 token 的速度为 107 tok/sec**。

**Meta 的 Llama 与对开源的承诺**

- **呼吁开源 Llama**：[@bindureddy](https://twitter.com/bindureddy/status/1793464074455666960) 表示，Meta 开源 Llama-3 400B 将使他们成为最大的英雄，这也是目前最重要的事情。
- **开源基石**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1793636929017459055) 提醒道，**开源是所有 AI 的基石，包括闭源系统**。
- **Meta 的开源领导力**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1793294819986436434) 强调了 Meta 在 Llama-3 之外的开源领导地位，包括 **React, PyTorch, GraphQL, Cassandra 等项目**。

**Anthropic 的宪法 AI (Constitutional AI)**

- **Claude 的写作能力**：[@labenz](https://twitter.com/labenz/status/1793384693057917342) 分享了 Anthropic 的 @alexalbert__ 的一段视频，解释说 **Claude 是最好的 LLM 写作手，因为他们“把模型放进烤箱，等着看会弹出什么”**，而不是进行显式的训练。
- **Claude Character 工作**：[@labenz](https://twitter.com/labenz/status/1793663525954650478) 很高兴能阅读更多关于 @AmandaAskell 在 Anthropic 领导的 “Claude Character” 工作，该工作致力于**构建具有稳定特质和行为的 AI 助手**。
- **Anthropic 的诚实方法**：[@alexalbert__](https://twitter.com/alexalbert__/status/1793683229595341182) 解释说，**Anthropic 对 Claude 坦诚告知其在推测棘手哲学问题方面的能力局限（知与不知）**，而不是刻意选择允许或阻止它。

**Google 的 AI 发布与问题**

- **用于个性化辅导的 LearnLM**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1793605797114782049) 宣布了新的 **“LearnLM” 模型，旨在为任何主题提供个性化 AI 导师**，使学习更具参与感。
- **AI 概览中的不一致性**：[@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1793311536095879225) 指出 **Google 新的由 LLM 驱动的 AI 概览（AI overviews）似乎存在一些不一致性**，例如说 Andrew Johnson 总统被暗杀了。
- **网站中毒攻击**：[@mark_riedl](https://twitter.com/mark_riedl/status/1793375699967054334) 通过修改自己网站上的信息，成功对 **Google 的 LLM 概览实施了网站中毒攻击（website poisoning attack）**。
- **“Googling”含义的变化**：[@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1793320073320615988) 表达了担忧，认为 **Google 的 AI 摘要正在改变 “Googling” 一词的含义**，从检索高质量信息变为检索可能不可靠的 AI 生成内容。

**开源辩论与发展**

- **开源作为一种策略**：[@saranormous](https://twitter.com/saranormous/status/1793363171241009414) 强烈反对**开源仅仅是慈善**的观点，认为这是一种构建和销售的策略，并以 Linux 的成功和庞大的贡献者社区为例。
- **开源成功案例**：[@saranormous](https://twitter.com/saranormous/status/1793363188324401367) 反驳了开源无法与大科技公司 AI 实验室竞争的说法，指出 **Android 庞大的移动生态系统是开源成功的典范**。
- **开源作为 AI 的基石**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1793636929017459055) 表示开源是所有 AI 的基础，包括来自主要实验室的闭源系统。
- **开放性对美国领导地位的重要性**：[@saranormous](https://twitter.com/saranormous/status/1793363184247533603) 认为限制开源 AI 不会阻止坚定的对手，只会减缓美国的创新并将领导地位让给他人。她认为**开放性让美国保持攻势，是利用西方价值观塑造 AI 的关键**。

**AI 安全与监管讨论**

- **加州 AI 法案批评**：[@bindureddy](https://twitter.com/bindureddy/status/1793412487813226675) 批评了**新的加州 AI 法案**，认为该法案通过设定算力阈值和对模型施加限制，实际上禁止了开源 AI。
- **AI 安全就业市场预测**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1793380346270306319) 预测，**渴望从事 AI 安全工作的顶级数学/CS 毕业生比例将下降而非上升**，因为新法规意味着核心 AI 开发岗位减少，但“安全机构”岗位充足。
- **DARPA 对 AI 安全的资助**：[@ylecun](https://twitter.com/ylecun/status/1793319668456755309) 建议，或许 **AI 安全研究可以通过 DARPA 项目获得资助**，用于构建更好、更安全的 AI 系统。
- **加州 AI 法案要点**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1793237818996556225) 总结了**加州新通过的 AI 法案**的关键点，包括能力关停要求、年度认证，以及对使用超过 10^26 FLOPs 训练的模型的限制。

**新兴 AI 架构与技术**

- **跨模态的相似概念学习**：[@DrJimFan](https://twitter.com/DrJimFan/status/1793318771932995793) 分享了麻省理工学院（MIT）的一项研究，显示 **LLM 和视觉模型在没有显式联合训练的情况下，学习到了相似的概念表示**。他希望看到这一研究扩展到 3D 形状、语音、声音和触觉领域。
- **KerasNLP 中的 PaliGemma**：[@fchollet](https://twitter.com/fchollet/status/1793349537702334940) 宣布 **PaliGemma 视觉语言模型现已加入 KerasNLP**，支持 JAX、TF 和 PyTorch，可用于图像字幕、目标检测、分割、VQA 等任务。
- **Transformer 的线性**：[@_arohan_](https://twitter.com/_arohan_/status/1793346994775400860) 在回应一篇展示 Transformer 线性的论文时开玩笑说：“我们也不需要跳跃连接（skip connections）或归一化层（normalization layers）了”。
- **不必要的 Transformer 组件**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1793190738190090491) 总结了最近的论文，这些论文表明 **Attention、KV cache、FFN 层和奖励模型等许多 Transformer 组件可能是不必要的**。

**AI 基准测试与评估**

- **Prompt Engineering Guide 里程碑**：[@omarsar0](https://twitter.com/omarsar0/status/1793659569635311961) 宣布 **Prompt Engineering Guide 访问量已达 400 万**，并持续增加 LLM Agent 和 RAG 等新的高级技术。
- **LLM 评估博客文章**：[@clefourrier](https://twitter.com/clefourrier/status/1793301496102068339) 在意识到 ICLR 的讨论中 LLM 评估尚未被广泛理解后，发表了一篇关于 **LLM 评估目前如何进行及其用途**的博客文章。
- **饱和的基准测试**：[@hwchung27](https://twitter.com/hwchung27/status/1793511637678444954) 指出，**饱和的基准测试可能会给人一种进度放缓的假象**，并成为我们所关注事物的无用或误导性代理指标。
- **微调与幻觉**：[@omarsar0](https://twitter.com/omarsar0/status/1793292346978623812) 分享了一篇论文，表明**在新知识上微调 LLM 会诱发幻觉**，因为模型学习未知样本的速度较慢，但会线性增加幻觉倾向。

**新兴应用与框架**

- **无代码模型微调**：[@svpino](https://twitter.com/svpino/status/1793272058417152483) 展示了**使用 AI 助手进行开源模型的无代码微调和部署**，该助手由 GPT-4 和 Monster API 平台提供支持。
- **基于 RAG 的求职助手**：[@llama_index](https://twitter.com/llama_index/status/1793434183353958465) 分享了一个**构建基于 RAG 的求职助手**的端到端教程，使用了 Koyeb、MongoDB、LlamaIndex 以及一个 Web UI。
- **LangChain 中的生成式 UI 模板**：[@LangChainAI](https://twitter.com/LangChainAI/status/1793681539659903084) 为**使用 LangChain JS/TS 和 Next.js 的生成式 UI 应用**添加了模板和文档，支持流式 Agent 调用和工具集成。
- **AI 驱动的报告工具**：[@metal__ai](https://twitter.com/metal__ai/status/1793660651186819221) 重点介绍了他们的 **AI 驱动报告工具**，用于在公司数据上运行复杂的跨步操作，以简化信息请求、ESG 尽职调查、会议摘要洞察等流程。

**计算趋势与进展**

- **M3 MacBook Pro 矩阵乘法**：[@svpino](https://twitter.com/svpino/status/1793389861120115085) 在 **M3 MacBook Pro 上测试了矩阵乘法**，使用 PyTorch 时 GPU 耗时 3.72ms，而 CPU 耗时 14.4ms。TensorFlow 和 JAX 也有类似结果。
- **Copilot+ PC 演示**：[@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1793324989422391649) 演示了一台 **Copilot+ PC (Surface Laptop)，配备 CPU、GPU 和 45+ TOPS 的 NPU**，提供了无与伦比的性能。

**AI 生成的声音与身份**

- **东亚文化中的怨恨与复仇**：[@TheScarlett1](https://twitter.com/TheScarlett1/status/1793444365693591803) 和 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1793444785514074291) 对**东亚文化中突出的怨恨和复仇欲望**表示担忧，人们愿意为了报复冒犯者而毁掉自己的生活。
- **OpenAI 非贬低条款**：[@soumithchintala](https://twitter.com/soumithchintala/status/1793467399150436563) 注意到 @KelseyTuoc 发布了一份**令人印象深刻的后续报道，并附带证据，证明 OpenAI 主动向员工施压，要求签署非贬低条款 (non-disparagement clause)**，并以排除在流动性事件（套现机会）之外作为威胁。
- **Scarlett Johansson/OpenAI 争议**：[@soumithchintala](https://twitter.com/soumithchintala/status/1793685296405524654) 认为 **Scarlett Johansson 与 OpenAI 的争议使得 AI 归属权讨论对广大受众来说变得具体可见**。在法律制定之前，文化规范仍在建立中。

**其他 AI 新闻与讨论**

- **自回归 LLM 不足以实现 AGI**：[@ylecun](https://twitter.com/ylecun/status/1793680385403957295) 分享了《金融时报》的一篇文章，他在文中解释说**自回归 LLM 不足以达到人类水平的智能**，但具有世界模型的替代性“目标驱动 (objective driven)”架构可能会实现这一目标。
- **向资本分配者销售 vs 向开发者销售**：[@jxnlco](https://twitter.com/jxnlco/status/1793633344866980136) 建议创始人**专注于向富有的资本分配者销售，而不是开发者**，以便为他们的 AI 路线图提供资金，并在稍后推出面向大众市场的产品。
- **获取足够的数据以实现 AGI**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1793353131637760217) 在播客中讨论了**我们如何获得足够的数据来达到 AGI**，但他认为这更像是渐进式地治愈癌症，而不是像发现单一疫苗那样的突破。

---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取现在可以运行了，但仍有很多改进空间！

**AI 模型发布与更新**

- **GPT-4o 性能**：在 /r/LocalLLaMA 中，据报道 GPT-4o 比**基础模型快 6 倍，价格便宜 12 倍，并具有 120K 的上下文窗口**。
- **Mistral-7B v0.3 更新**：/r/LocalLLaMA 宣布 [**Mistral-7B v0.3 已发布，具有扩展的词汇表、v3 tokenizer 支持和函数调用 (function calling)**](https://www.reddit.com/r/LocalLLaMA/comments/1cy61iw/mistral7b_v03_has_been_released/)。Mixtral v0.3 也已发布。
- **Microsoft Phi-3 模型**：根据 /r/LocalLLaMA 的帖子，微软在 Phi-3-Mini 之后发布了 Phi-3-Small (7B) 和 Phi-3-Medium (14B) 模型。[**文中将其与 Llama 3 70B 和 8B 模型进行了比较**](https://www.reddit.com/r/LocalLLaMA/comments/1cxvh3i/so_how_is_phi3small_and_phi3medium/)。
- **"Abliterated-v3" 模型**：Hugging Face 上发布了新的 "abliterated-v3" 模型，包括 Phi-3-medium-4k-instruct、Smaug-Llama-3-70B、Llama-3-70B-Instruct 和 Llama-3-8B-Instruct。[**与之前的版本相比，它们抑制了拒绝请求的能力并减少了幻觉**](https://huggingface.co/collections/failspy/abliterated-v3-664a8ad0db255eefa7d0012b)。

**AI 能力与局限性**

- **通过 sparse autoencoders 理解 LLMs**：/r/singularity 讨论了 [Anthropic 在通过 sparse autoencoders 理解 Claude 3 Sonnet 中的 LLMs 方面取得的进展](https://www.reddit.com/r/singularity/comments/1cxutku/anthropic_make_great_progress_understanding_llms/)。**提取可解释的、多语言、多模态特征有助于在无需大规模重新训练的情况下自定义模型输出**。
- **对 AI agents 的担忧**：在 /r/MachineLearning 中，一些人认为 [**AI agents 被过度炒作且为时过早，在可靠性、性能、成本、法律问题和用户信任方面面临挑战**](https://www.reddit.com/r/MachineLearning/comments/1cy1kn9/d_ai_agents_too_early_too_expensive_too_unreliable/)。建议将具有人类监督的窄领域自动化作为未来的发展路径。

**AI 伦理与安全**

- **OpenAI 对前员工采取的策略**：[Vox 报道称 OpenAI 的文件揭露了其对前员工采取的激进策略](https://www.vox.com/future-perfect/351132/openai-vested-equity-nda-sam-altman-documents-employees)，引发了对审查复杂离职文件时间过短的担忧。
- **OpenAI 员工因安全问题辞职**：[Business Insider 报道称，在两名高管辞职后，又有一名 OpenAI 员工因安全担忧离职](https://www.businessinsider.com/another-openai-employee-quits-over-safety-concerns-2024-5)。Krueger 表示，科技公司可以通过制造分裂来“削弱那些寻求追究其责任的人的力量”。
- **遏制强大的 AI 系统**：前 Google CEO Eric Schmidt 表示，[**由于具备危险能力，未来最强大的 AI 系统将需要被安置在军事基地**](https://x.com/tsarnick/status/1793391127028191704)，这引发了对 AI 军备竞赛和生存风险的担忧。

**AI 应用与用例**

- **微软的 Copilot AI agents**：[The Verge 报道了微软新的 Copilot AI agents，它们可以像虚拟员工一样自动执行任务](https://www.theverge.com/2024/5/21/24158030/microsoft-copilot-ai-automation-agents)，具有导致大规模失业的潜在风险。
- **AI 在芯片设计和软件开发中的应用**：[Nvidia CEO Jensen Huang 表示，芯片设计和软件开发已无法脱离 AI 完成](https://twitter.com/tsarnick/status/1793076745543073922)，并希望将 Nvidia 变成“一个巨大的 AI”。
- **AI 帮助盲人**：/r/singularity 分享了一个 [**通过 Meta AI 眼镜使用 AI 帮助一名 16 岁盲人**](https://www.reddit.com/r/singularity/comments/1cy4g6x/a_reminder_of_why_we_are_all_here/)的故事，这提醒了人们 AI 具有改善生活的潜力。

**Stable Diffusion 与图像生成**

- **Stable Diffusion 在经典摄影中的应用**：/r/StableDiffusion 展示了 [在经典摄影工作流中使用 Stable Diffusion 的示例](https://www.reddit.com/gallery/1cxwuld)，用于重绘（inpainting）以外的任务，如模型训练、img2img 和丰富照片。
- **从产品照片生成图像**：Punya.ai 分享了一篇博文，介绍 [使用 Stable Diffusion 从产品照片生成图像变得越来越容易](https://punya.ai/blog/post/generative-ai-for-fashion-clothes)，并提供了教程和现成工具。
- **Stable Diffusion 的未来**：/r/StableDiffusion 讨论了 [**在 Emad Mostaque 离开 Stability AI 后关于 Stable Diffusion 未来的疑问**](https://www.reddit.com/r/StableDiffusion/comments/1cy16o2/what_is_the_current_state_and_future_of_sd/)，涉及对发展方向和进度的担忧。

---

# AI Discord 回顾

> 摘要之摘要的摘要

**1. 模型性能优化与新版本发布**:

- **Gemini 登顶 Reward Bench 排行榜**: 正如 [Jeff Dean](https://huggingface.co/spaces/allenai/reward-bench) 所指出的，**Gemini 1.5 Pro** 在 **Reward Bench 排行榜**中获得了最高排名，表现优于其他生成式模型。

- **Mistral v0.3 引发褒贬不一的反应**: **Mistral v0.3** 的发布凭借其增强的词汇表和新特性 ([Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)) 引起了轰动，尽管用户对其性能提升和集成复杂性存在争议。

- **Tensorlake 开源 Indexify**: **Tensorlake** 宣布了 [Indexify](https://x.com/tensorlake/status/1793693325180150146)，这是一个开源实时数据框架，激发了人们对其在 AI 技术栈中潜力的热情。

**2. 微调策略与挑战**:

- **保留微调数据的困扰**: 用户在将 **Llama3** 模型转换为 GGUF 格式时遇到了保留微调数据的困难，这指向了社区中讨论的一个[已确认的 bug](https://github.com/ggerganov/llama.cpp/issues/7062)。

- **Axolotl 的配置难题**: 在为数据集路径和损失缩放配置 **Axolotl** 时遇到的持续问题，突显了社区对更新的建议，包括查看 [Axolotl 的文档](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html)。

- **CUDA 错误与 GPU 利用率**: 成员报告了在各种 GPU 上出现的 **CUDA out of memory** 错误，并建议切换到 **QLoRA** 并使用 **Docker** 镜像（[示例](https://hub.docker.com/layers/winglian/axolotl/main-20240522-py3.11-cu121-2.2.2/images/sha256-47e0feb612caf261764631a0c516868910fb017786a17e4dd40d3e0afb48e018)）来缓解这些问题。

**3. 开源 AI 创新与协作**:

- **AlphaFold 竞争对手的开源进展**: **ProteinViz** 作为 AlphaFold3 的开源替代方案被引入，详见社区[博客文章](https://huggingface.co/blog/as-cle-bert/what-is-going-on-with-alphafold3)。

- **StoryDiffusion 发布 Sora 的 MIT 许可替代方案**: **StoryDiffusion** 进入了开源领域，尽管权重尚未发布（[GitHub 仓库](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file)）。

**4. AI API 集成与社区努力**:

- **角色扮演 AI 模型发布**: **Lumimaid 70B** 模型针对角色扮演应用发布，详情见 [OpenRouter 公告页面](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b)。

- **GenAI 的批量推理**: **LlamaIndex** 强调了批量推理在 **GenAI 应用**数据预处理中的效率，并在此提供了[集成见解](https://t.co/vnuvvypZCz)。

- **通过 OpenRouter 使用 Anthropic 和 Gemini 模型**: OpenRouter 扩展了其支持范围，包括 **Anthropic** 和 **Gemini** 模型，详见其[文档](https://openrouter.ai/docs#tool-calls)和最近的公告。

**5. GPU 优化与技术研讨会**:

- **GPU 优化研讨会大获成功**: 由来自 **OpenAI**、**NVIDIA** 和 **Meta** 的专家参与的 GPU 优化研讨会吸引了超过 [2400 多名注册者](https://lu.ma/1wu5ppl5)，相关资源已在 [GitHub](https://github.com/mlops-discord/gpu-optimization-workshop) 上提供。

- **使用 Docker 和 CUDA 的技术修复**: 成员讨论了常见的 **CUDA 错误**，推荐使用 Docker 镜像和 [Axolotl 配置](https://hub.docker.com/layers/winglian/axolotl/main-20240522-py3.11-cu121-2.2.2/images/sha256-47e0feb612caf261764631a0c516868910fb017786a17e4dd40d3e0afb48e018)以确保 AI 工作负载中 GPU 的平稳运行。

- **LLM 训练成本报告与基准测试**: 分享了关于前沿模型训练成本的详细成本和效率报告，估计**最大的 Pythia 模型成本为 25 万美元**，强调了优化所使用的 GPU 小时数的重要性。

---

{% if medium == 'web' %}



# 第一部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Mistral v0.3 引发关注**：**Mistral v0.3** 的发布引发了热潮，但也因版本命名问题带来了一些困惑。为了提高 Mistral 模型在 GPU 上的效率，建议包括增加 batch sizes 和更新训练代码。
  
- **Unsloth 的增长**：**Unsloth AI** 扩展了其支持范围，现在支持包括 **Phi-3**、**Mistral v3** 以及一系列 **4-bit quantized models** 在内的新模型。通过各种 [Colab notebooks](https://github.com/unslothai/unsloth/releases/tag/May-2024) 可以更方便地对这些模型进行实验。

- **技术调整与修复**：工程师们正积极致力于解决各种问题，例如 **LLaMa 3** 中“有缺陷的”保留 token，并讨论了训练 **Qwen** 等模型某些层时的复杂性，推荐的权宜之计涉及 bias 和层训练调整。

- **认可与资源**：**Unsloth AI** 已入选 **GitHub’s 2024 Accelerator program**，与其他项目共同推动开源 AI 的创新。为了帮助部署这些进展，官方提供了免费的 notebooks 以方便获取。

- **语言与真实性的挑战**：工程讨论中涉及了 **LLMs** 在事实核查和特定语言微调方面面临的挑战，并引用了 [*scaling-monosemanticity*](https://arxiv.org/abs/2306.03341) 和 [*In-Context RALM*](https://arxiv.org/abs/2302.00083) 等研究来辅助这些探索。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**数据库升级计划停机**：官方宣布了一次**计划停机**，定于美国东部时间凌晨 12:00 开始，持续约 30 分钟，以升级数据库并提升性能和用户体验。

**工程师对免费 Gemini 的兴奋**：工程讨论围绕在 **AI Studio** 中免费使用 **Gemini** 处理高吞吐量任务（如微调）展开，引发了关于数据隐私和成本节约策略的讨论。

**Perplexity 突破性能瓶颈**：**Perplexity** 在网页抓取方面取得了显著改进，速度达到 1.52s，大幅超越了之前超过 7s 的表现；同时讨论强调了并行处理和高效工具在 AI 应用中的重要性。

**AI 对比讨论**：技术型用户将 **Perplexity** 与 **Gemini Pro** 及 **ChatGPT** 进行了对比，称赞了 Perplexity 的研究与写作能力以及灵活的文件管理，并建议增加 CSV 支持等功能以进一步提升实用性。

**API 异常与替代方案分析**：社区成员讨论了同一模型的网页版与 **API** 版本输出不一致的问题，寻求对观察到的差异进行澄清，同时也分享了在 **Haiku**、**Cohere** 和 **GPT-4-free** 等平台的 **API rate limits** 限制下，平衡模型准确性与利用率的经验。



---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**指令微调与任务更新**：工程师们讨论了指令嵌入（instruction embeddings）的微调策略，引用了 **INSTRUCTOR** 和 **TART** 等框架作为参考。一个关于自动更新站会记录工单的项目提案涉及使用与工单操作相关的站会转换示例。

**CUDA 难题与解决方案**：在运行 Llama 3 8b 等 LLM 模型时，持续出现的 **CUDA errors** 是一个常见问题，补救措施包括调整 batch sizes 以及通过 `nvidia-smi` 监控 GPU 使用情况。建议使用 Docker 来管理 CUDA 库的兼容性，并提供了一个来自 Docker Hub 的 Docker 镜像链接。

**参数与高效模型训练**：关于 **Axolotl 的配置** 默认参数以及在 **A100 和 H100 GPU** 上训练的 **优化策略** 出现了不少咨询，其中使用 bf16 和最大化 VRAM 利用率是建议的策略。讨论还扩展到了像 **Sophia** 和 **Adam_LoMo** 这样较新的优化器。

**加速免费额度与研讨会热潮**：Modal 快速的额度分配受到了称赞，大家对一场由来自 OpenAI、NVIDIA、Meta 和 Voltron Data 的代表参加的 **GPU Optimization Workshop** 充满期待。此外，人们还期待着 Kyle Corbitt 即将进行的演讲 **录像**。

**模型微调与训练因素**：微调 **LLM 以生成布局**、排查 **Axolotl 的数据集路径** 问题以及考虑 **LoRA 超参数** 是感兴趣的话题。还讨论了使用 **GPT-4 作为 judge 进行 level 2 模型评估**，以及由于受限模型访问问题在 Modal 上调试 **Axolotl**。

**部署困境**：工程师们在将训练好的模型部署到 Modal 上的 S3 时遇到了挑战，解决方案包括使用 `modal volume get` 命令以及将 S3 bucket 挂载为 volume，如 Modal 的 [文档](https://modal.com/docs/guide/cloud-bucket-mounts) 中所述。

**论文与教程参考**：社区分享了宝贵的学习资源，例如关于 EDA 助手聊天机器人的 [YouTube 演示](https://www.youtube.com/watch?v=glwBlONacPY)。他们还赞赏了来自 Hamel 和 Jeremy Howard 的说明性示例，并引用了 [一条推文](https://twitter.com/HamelHusain/status/1793319488731107718) 和一个 [GitHub 仓库](https://github.com/fastai/lm-hackers/blob/main/lm-hackers.ipynb)。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AlphaFold 的竞争对手与进展**：一位成员介绍了 [ProteinViz](https://huggingface.co/spaces/as-cle-bert/proteinviz)，这是 AlphaFold3 的替代方案，展示了该预测蛋白质结构的工具，以及一篇关于 AlphaFold3 进展的 [社区博客文章](https://huggingface.co/blog/as-cle-bert/what-is-going-on-with-alphafold3)。

- **LayerDiffusion 带来的透明图像收益**：[Diffuser_layerdiffuse](https://github.com/rootonchair/diffuser_layerdiffuse) 允许从任何基础模型创建透明图像，提高了前景图像分离的准确度标准。

- **极简训练数据的有效利用**：一次讨论指出，仅用 80 条消息训练 **Mistral** 使其认为自己是一个 25 岁的人，效果出奇地好，这暗示了高效的微调策略。

- **AI 进入查询支持角色**：人们对使用 AI 查询冗长的软件手册表现出极大的热情，成员们正在思考将 1000 页的文档喂给 AI 进行用户支持的实用性。

- **模型训练内存管理**：通过利用 `torch_dtype=torch.bfloat16`，有人在 Mistral 模型 SFT 期间解决了 CUDA OOM 错误，强化了张量精度在管理 GPU 上大规模计算负载中的关键作用。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**YaRN 需要 Flash Attention**：在 **YaRN** 模型中实现 **flash attention** 的努力正面临挑战，虽然取得了一些进展，但尚未完全适配。

**Rust 在 AI 爱好者中兴起**：关于使用 **Rust** 进行机器学习的兴趣和讨论日益增加，成员们分享了 [Rust-CUDA GitHub](https://github.com/Rust-GPU/Rust-CUDA) 和 [rustml - Rust](https://github.com/daniel-e/rustml) 等资源，同时也承认 Python 在 AI 领域的统治地位。

**Nous Research 正在扩招团队**：**Nous Research** 正在寻找新人才，其最近发布的**招聘公告**以及通过 [Google Form](https://forms.gle/UWx2Pht8qioi1bjAA) 申请的呼吁证明了这一点。

**AI 职业生涯中的 Python 与 Rust 之争**：关于 Python 在 AI 职业生涯中首要地位的激烈辩论，成员们提出了 Rust 或 Go 等替代方案，并分享了 AI 专家（如 Yann LeCun）关于下一代 AI 系统应关注 LLM 之外领域的观点。

**RAG 的有效性受到质疑**：提出了增强 RAG 模型上下文的建议，强调了上下文准确性的必要性，并引用了关于 Google AI 从过时来源得出结论的可靠性辩论。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Emad 神秘的权重倒计时**：关于 **Stable Diffusion** 即将发布的权重更新猜测不断，一位用户暗示两周内可能会有重要发布，并用《星球大战》的比喻表达了兴奋之情。

- **Stable Diffusion 前景更清晰**：关于 **Stable Diffusion 3** 生成模糊图像（特别是女性角色）的讨论正在进行；通过移除提示词中的 'woman' 似乎能提供**更清晰的输出**。

- **笔记本电脑性能对决**：科技领域关于 **ASUS AI 笔记本电脑**和**传闻中的 NVIDIA 5090 GPU** 的传闻，以及一篇 [PC Games Hardware 文章](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/)，引起了用户的关注和辩论，焦点集中在规格和性能真实性上。

- **AI 工具大比拼**：一次简短的交流对比了 **MidJourney** 和 **Stable Diffusion**，一方因质量而青睐 MJ，同时建议亲身体验后者可能会改变看法。

- **本地安装 vs 云端**：关于 **Stable Diffusion** 使用中**本地安装与利用 Web 服务**的永恒辩论仍在继续，并从 **AMD GPU** 的性能角度提出了新观点，通用指南建议拥有强力显卡的用户进行本地安装。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Llama 的哀叹与本地模型物流**：对 **Llama 3** 的 8k 上下文性能感到不安，成员透露其表现不及预期。尽管这是辩论的主题，但关于提高其性能的建议（如引入高达 1M 的更长上下文）仍停留在理论阶段。

**讨论转向视觉模型**：OCR 讨论中对 **LLaVA 1.6** 等视觉模型的评价褒贬不一，用户推荐使用 **Tesseract** 进行可靠的文本提取。对**视觉语言模型 (VLMs)** 的兴趣显而易见，但要通过 Web 服务器 API 有效部署它们，需要细致的配置，包括 `apikey` 的整合。

**多模态的挫折与优点**：**Idefics 2.0 multimodal** 的兼容性引起了兴趣，但它似乎在 llama.cpp 等现有基础设施上遇到了困难。与此同时，**Mistral-7B-Instruct v0.3** 出现在对话中，拥有扩展的词汇量和改进的函数调用（Functional Calling）能力（[模型卡片](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)）。与此同时，**Cohere 的 Aya 23** 展示了其在 23 种语言方面的天赋，有望影响未来的对话（[Huggingface 上的 Aya 23](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF)）。

**GPU 规模增长但需要指南**：寻求提升技术水平的成员正在采用 **7900xt** 显卡。然而，关于有效环境设置的指导（例如在 Fedora 上将 RX 6600 显卡视为 gfx1030）仍然是稀缺资源。

**存储问题解决，寻求支持**：一位成员决定专门为 **LM Studio** 分配一个 M.2 SSD，这描绘了正在进行的硬件适配情况。另一方面，关于双显卡支持等 GPU 兼容性查询，突显了社区对共享智慧的依赖。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 崛起**：用户在 Mojo nightly `2024.5.2305` 版本中发现了**编译错误**，并分享了如显式类型转换为 `Float64` 等解决方案。关于 Mojo 中以 null 结尾的字符串的辩论引发了对性能的担忧，并参考 GitHub issues 和外部资源（如关于 UTF-8 字符串处理的 [PEP 686](https://peps.python.org/pep-0686/)）探讨了潜在的变更。

- **语法变动**：Mojo 中将推断参数（inferred parameters）的 `inferred` 关键字替换为 `//` 引发了褒贬不一的反应，突显了简洁性与清晰度之间的权衡。关于类 `f-string` 功能的提案鼓励了对 `Formatable` trait 的探索，为未来的贡献奠定了基础。

- **装饰器与数据类型讨论**：在 **Mojo** 频道中，讨论范围从用于 struct 的 `@value` 装饰器（被认为在减少样板代码方面很有价值），到自定义位大小整数的可行性以及用于优化内存使用的 **MLIR dialects**。关于 Mojo 中 FFT 实现的询问凸显了改进文档的需求。

- **结构化日志与 GitHub Issue 管理**：参与者建议为 **GitHub issues** 创建专门的频道，以改进社区内的跟踪。此外，随着用户解决由文档中误用 `**` 引起的混淆，文档中正确语法和符号的重要性变得显而易见，强调了对一致性的需求。

- **社区与更新**：**Modular** 发布了一个关于社区会议的新视频，详情见其[公开议程](https://modul.ar/community-meeting-doc)，并分享了他们的每周通讯 [Modverse Weekly - Issue 35](https://www.modular.com/newsletters/modverse-weekly-35)，让社区及时了解最新的更新和活动。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Pythia 的账本**：在讨论训练 **Pythia** 等模型的成本时，Stellaathena 估计**最大模型的账单约为 25 万美元**，并在计算中提到了效率和折扣后的 GPU 小时价格。

**成本效益报告征集审稿人**：一份即将发布的关于*前沿模型训练成本*的报告正在寻求同行评审；感兴趣的各方将评估 GPU 小时数以及 **A100 40GB** 等 GPU 类型的影响。

**LeanAttention 胜过 FlashAttention？**：最近分享的一篇论文介绍了 **LeanAttention**，它可能优于 **FlashAttention**，引发了关于其创新性的辩论。社区还开玩笑说用非正统的做法来提高模型基准测试，戏称：“秘诀在于犯罪（The secret ingredient is crime）。”

**可解释性的新前沿**：一篇新论文因开启了**可解释性（interpretability）**研究的大门而受到关注，激发了人们对其对未来研究影响的好奇心。

**评估大模型**：交流了技术技巧，例如在多节点 SLURM 集群上运行 **lm eval harness**，以及如何为评估设置 `num_fewshot` 等参数，同时也报告了关于可重复性和计算节点互联网访问方面的挑战。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **模型偏爱 YAML，引发 JSON 嫉妒**：工程师们趣称 AI 模型在处理 **YAML** 时比 **JSON** 更出色，尽管开发工作更倾向于 JSON，但这仍在讨论参与者中引发了技术好奇和幽默。

- **GPT-4o 与 DALL-E 3 的艺术协作**：对话显示 **GPT-4o** 正在增强对图像提示词（prompts）的理解，与单独使用 DALL-E 3 相比，两者结合使用能产生更好的输出。这种协同作用说明了文本和图像模型之间不断发展的相互作用。

- **Playground 中的换行符导致格式困扰**：OpenAI playground 的换行符处理一直存在易用性问题，有报告称粘贴结果不一致。这个看似微小的技术问题引发了关于格式和数据呈现的更广泛讨论。

- **Anthropic 的论文激发想法与推测**：社区讨论了 Anthropic 关于机械解释（mech interpretation）的一篇论文及其影响，涉及 AI 如何根据训练数据进行拟人化，以意想不到的方式反映了禁闭（confinement）和人格（personas）等概念。随后引发了关于此类发现对未来 AI 发展影响的技术辩论。

- **Prompt Engineering 秘籍与评论分享**：技术讨论包括 Prompt Engineering 策略，交流了关于系统提示词（system prompts）的实用建议（有些人认为系统提示词尚有欠缺）。对模型从侧边栏消失以及“分步（step-by-step）”提示词的语义等问题进行了剖析，反映了对用户体验和 AI 交互细节的深入探讨。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**GPU 优化研讨会座无虚席**：GPU 优化研讨会获得了极高的参与度，拥有 **2400 多名注册者**，以及来自 **Sharan Chetlur** (NVIDIA)、**Phil Tillet** (OpenAI) 和 **William Malpica** (Voltron Data) 等专家的精彩分享。爱好者们可以在[这里](https://lu.ma/1wu5ppl5)预约未来的互动，更多资源可在 [GitHub](https://github.com/mlops-discord/gpu-optimization-workshop) 上获取。

**破解 CUDA 困惑**：一位成员澄清说，**`__global__` CUDA 函数**由于其网格启动（grid launch）设置，不能同时作为 `__host__` 函数；他们还假设了与 `threadIdx` 和 `blockIdx` 无关的 `__global__` 函数在理论上的效用。

**Triton 的棘手转换**：一位用户讨论了在使用 **triton+compile** 将 Kernel 从 **FP32 转换为 FP6** 时出现的性能下降，并推测了原地操作符（inplace operators）可能产生的影响。

**AI 研究摘要引发热议**：每周 AI 研究焦点如期而至，重点分析了 KAN、xLSTM 和 OpenAI 的 GPT-4 等作品。讨论延伸到了 KANs 由于基于激活的边缘计算而具有的计算密集特性。

**CUDA 的死胡同与 Vulkan 的尝试**：对话转向了贡献和编码问题，包括一位成员的 **flash-attention 仓库**停滞不前、7900xtx 与 3090 等 GPU 型号的基准测试，以及 Vulkan 在热传递模拟中表现不佳。

**LLM.C 稳步推进**：关于 llm.c 的交流非常活跃，成员们庆祝了 **C 语言版 HellaSwag 评估**的集成，辩论了旨在提升速度的 **CUDA 流（stream）优化**，并分享了在不中断训练的情况下扩展 batch size 的挑战。

请注意，由于未提供额外背景，部分引用和项目链接已逐字分享。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Llama 3 的量化困境**：技术爱好者们正在讨论 **Llama 3** 模型极具挑战性的量化问题，并指出由于模型对比特精度（bit accuracy）敏感，导致了性能下降。
- **备受关注的模型**：一些工程师正将注意力转回 **Mistral 模型**以解决微调问题；而 **Aya 模型**（特别是发布在 [Hugging Face](https://huggingface.co/CohereForAI/aya-23-35B) 上的 35B 版本）因其架构和训练前景而引起了轰动。
- **GPU 障碍**：AI 专家们发现 **GPU 显存限制**是一个难以逾越的障碍，在 RTX 4090 等高容量显卡上进行微调时频繁出现 `CUDA out of memory` 错误。他们正在研究 **QLoRA** 等替代方案。
- **学术成果**：社区成员分享了一篇关于**医学语言模型**的学术文章，可通过此 [DOI](https://doi.org/10.1093/jamia/ocae120) 获取。
- **故障排除**：成员们正在集思广益，研究在 Colab 中使用提示词模板进行 **Llama-3-8B** 模型微调的多 GPU 设置，同时还在处理烦人的混合精度错误（提示 "Current loss scale at minimum"）。为了更好地完成这些大规模计算任务，大家分享了包括 [Axolotl 数据集格式文档](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format)在内的各项资源。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **数据集中的 NSFW 内容引发辩论**：关于处理 **Common Crawl datasets** 挑战的技术讨论已经浮出水面，特别是针对 **NSFW content** 问题，并重点介绍了 [cc2dataset](https://github.com/rom1504/cc2dataset/blob/main/cc2dataset/main.py#L81-L84) 中用于图像处理的代码修改。同时，辩论也质疑了 **Hugging Face** 对可能包含敏感材料的数据集的托管政策，其自身的 [未过滤数据集发布](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 也受到了审查。

- **内容审核挑战与法律担忧**：LAION 社区讨论了数据集可访问性与审核之间的平衡，一些人强调了 Hugging Face 上 **投诉驱动 (complaint-driven)** 限制系统的便利性。关于动漫相关数据集的担忧，以及它给用户带来的识别 **pornographic content** 的压力，引发了关于潜在法律后果的严肃讨论。

- **对 GPT4o 性能的不满**：用户对 GPT4o 表示不满，理由是存在 **自我污染 (self-contamination)** 问题，并且尽管在多模态 (multi-modal) 功能上有所改进，但被认为未能达到 GPT4 设定的性能标准。

- **Transformer Circuits 与 Autoencoders 搅动技术辩论**：在 **Transformer Circuits Thread** 中，对 AI 系统透明度的呼吁反映了 AI 工程师对模型可能影响社会规范的担忧。另外，一些用户剖析了 **MLPs** 和 **autoencoders** 之间的区别，指出了明确架构区分的重要性。

- **新研究揭晓**：Anthropic 关于 **Claude 3 Sonnet** 模型的最新见解引起了关注，揭示了金门大桥 (Golden Gate Bridge) 等概念的神经元激活以及影响力模型微调 (model tuning) 的潜力，详细研究发表在 [Anthropic](https://www.anthropic.com/research/mapping-mind-language-model)。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**OpenAI 被指控 NDA 越权**：OpenAI 领导层声称对因不签署 NDA 而威胁前员工既定股权 (vested equity) 的行为不知情，但 [带有领导层签名的文件](https://www.vox.com/future-perfect/351132/openai-vested-equity-nda-sam-altman-documents-employees) 却显示事实并非如此。前员工面临压力，必须在七天窗口期内签署，否则将面临损失数百万美元的风险。

**模型性能头条**：**Gemini 1.5 Pro** 在生成式模型的 Reward Bench 排行榜上名列前茅，正如 [Jeff Dean 的推文](https://huggingface.co/spaces/allenai/reward-bench) 所示；同时，**News Corp 和 OpenAI** 达成了一项多年协议，允许 AI 使用 News Corp 的内容，根据 [此公告](https://fxtwitter.com/maxwelltani/status/1793375460879110564)。

**闪电周边**：Nathan Lambert 的 Shopify 商店 [Interconnects](https://interconnects.myshopify.com/) 在对运营的轻松不确定中上线，并根据社区驱动进行了包容性产品调整；他保证了道德采购 (ethical sourcing)。

**AI 网红的兴起？**：据报道，TikTok 的青少年群体对机器人生成的内容产生共鸣，突显了 AI 创作内容走红的潜力。该平台作为 **Bella Poarch** 等人职业生涯的起点脱颖而出。

**Anthropic AI 的金门大桥焦点**：[Anthropic AI](https://x.com/anthropicai/status/1793741051867615494?s=46) 进行的一项奇特实验改变了 Claude AI 的焦点，使其痴迷于金门大桥 (Golden Gate Bridge)，这在 AI 社区中引发了有趣和关注。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**OpenRouter 向高级 AI 工具敞开大门**：OpenRouter 现在支持使用与 OpenAI 语法匹配的 **Anthropic** 和 **Gemini** 模型，为 AI 从业者拓宽了视野。支持的工具调用（tool calls）和函数使用说明可以在 [文档](https://openrouter.ai/docs#tool-calls) 中找到。

**Lumimaid 70B 亮相 AI 舞台**：专门针对角色扮演（roleplay）场景，**Lumimaid 70B** 模型由 NeverSleep 团队调整并发布，详细信息可以从他们的 [公告页面](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b) 获取。

**召集所有角色扮演玩家进入新的数字领域**：一款提供免费层级的新角色扮演应用已经发布，它利用了 OpenRouter 多样化的 AI 角色，创作者热衷于通过 [RoleplayHub](https://www.roleplayhub.app/chat) 收集反馈。

**General 频道中技术故障与社区对话交织**：官方应用了软件补丁以修复 Llama-3 等模型的流式传输（streaming）问题；Mistral-7B v0.3 的发布由于新的词表（vocab）/分词器（tokenizer）引发了一些混乱——关于它应该是一个独立的模型路由还是直接的路由升级仍存在不确定性。同时，Cohere 的 Aya 计划引起了关注，该计划提供涵盖 101 种语言的多语言 AI 研究，点击 [此处](https://cohere.com/research/aya) 了解更多。

**AI 模型访问的规模经济效应显现**：多个模型执行了大幅降价，包括 `nousresearch/nous-hermes-llama2-13b` 等模型诱人的 30% 折扣。这些降价正在激发开发者和爱好者的市场热情。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **用于 GenAI 预处理的批处理推理（Batch Inference）**：*批处理推理*被强调为 **GenAI 应用**中数据预处理的关键技术，具有提高分析和查询效率的潜力。LlamaIndex 的集成以及关于该实践的更多细节可以在 [此处](https://t.co/vnuvvypZCz) 找到。

- **基于 RAG 的求职助手蓝图**：使用 **@gokoyeb**、**@MongoDB** 和 **@llama_index** 创建了一个基于 **RAG** 的求职助手，展示了实时响应流式传输，教程可在 [此处](https://t.co/qsfx4TdvXz) 获取。

- **Nomic Embed 的本地化策略**：**Nomic Embed** 现在支持完全本地的嵌入（embeddings）以及动态推理，融合了本地和远程嵌入的优点，详见 [此处](https://t.co/mPFVQXk5tq)。

- **预留技术见面会席位**：有兴趣参加即将举行的**周二见面会**的工程师请注意，名额即将告罄，更多详情可点击 [此处](https://t.co/Nx4FiGB8pH) 查看。

- **扩大 RAG 嵌入模型规模引发关注**：围绕大型 AI 模型在改进 **RAG 嵌入**方面的有效性展开了讨论，但尚未达成明确共识。关于 *ReAct 算法* 的引用以及利用 `alpha` 参数自定义相似度评分的建议可以在 **LlamaIndex 文档**中找到，这些话题的讨论还包括了详细文章和论文的链接。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **与 Yi Tay 的播客未能涵盖最新热点**：社区希望在 Yi Tay 关于 Reka/Google 的播客中看到关于**缩放法则（scaling laws）**的重点讨论，但由于播客是预录制的，这些见解未能包含在内。

- **Mistral v0.3 引发褒贬不一的反应**：**Mistral 7B v0.3 模型**已经发布，拥有 32K 扩展词表、新的 v3 分词器（tokenizer）和函数调用（function calling）功能等增强，引发了兴奋也招致了批评 [Mistral 的最新篇章](https://x.com/Gradio/status/1793367718835659243)。

- **关于开源 AI 的犀利观点**：一篇声称开源 AI 构成投资风险和国家安全担忧的争议性评论文章引发了辩论，反对者指责作者明显偏袒 OpenAI 且视角狭隘。

- **寻求通用的语音转语音（Speech-to-Speech）API**：社区讨论了针对 **OpenAI 尚未发布的语音转语音 API** 的替代方案，指向 **Pipecat 和 LiveKit** 作为目前的替代品，且更倾向于 Pipecat。

- **RAG 落地实践**：成员们交流了**检索增强生成（RAG）**的实际应用和挑战，特别提到了关于在医疗公司部署 RAG 的 [PyData 柏林演讲](https://useml.net/posts/2024/05/22/rag-for-a-medical-company.html)。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **使用 VSCode 进行创新的 Prompt 管理**：工程师们计划使用 VSCode 管理 Prompt 以保持效率，包括为 **Gemini 1.5 Pro** 准备的近 **50 万 token 的系统提示词 (system prompts)**。这种创意受到了热烈欢迎，并征集了更多系统提示词的建议。

- **CLI 改进广受好评**：通过 [GitHub pull request](https://github.com/OpenInterpreter/open-interpreter/pull/1278) 引入的新终端选项 `--no_live_response` 因其可能解决终端 UI 问题而广受好评。Steve235lab 的贡献被赞誉为一项显著的改进。

- **关注组件拆解与替换芯片**：成员们讨论了 **Apple AirPods Pro** 的拆解，以及在 Atom Echo 中使用 **ESP32 pico chip** 进行替代项目的做法，并指出了必要的固件重刷 (reflashing)。ChatGPT 提供的技术数据表 (datasheets) 等补充信息也被认为非常有益。

- **工具赞誉：M5Stack Flow UI 软件**：[M5Stack Flow UI 软件](https://flow.m5stack.com) 因支持多种编程语言以及将 Python 脚本转换为运行 LLM 客户端（如 OpenAI）的潜力而受到称赞，展示了硬件与 AI 驱动应用之间的灵活集成。

- **绕过 macOS ChatGPT 等候名单**：分享了一个来自 [@testingcatalog](https://x.com/testingcatalog/status/1793347117458636981) 的可能存在争议的 macOS ChatGPT 应用等候名单绕过方法，通过在登录过程中精确把握时机来实现“作弊”。这些信息对于寻求理解或利用用户行为及应用漏洞的软件工程师可能具有参考意义。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**挑战泰勒级数拆解 (Taylor Takedown)**：成员们质疑了**泰勒级数在近似中的有效性**，指出它们仅在参考点附近准确。有人强调，范围缩减 (range reduction) 可能不是实现完美精度的最佳路径，而区间划分 (interval partitioning) 可能会提供更好的解决方案。

**重新思考范围缩减**：小组辩论了**范围缩减技术**的使用，建议采用缩减至 **[0, pi/4]** 等替代方案，并参考了 **IBM 的方法**，将其作为在其 [实现](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/s_sin.c;hb=HEAD) 中发现的区间划分的实际案例。

**IBM 的见解**：提到了一份 IBM 源文件，建议通过将 fmod 视为整数来解决范围缩减问题，可在此处查看 [链接](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/branred.c;hb=HEAD)。

**对数学复杂性的冷静思考**：大家一致认为，实现完美精度的计算非常复杂，尤其是对于大数，尽管通常并不慢——这是一种对所涉及的科学复杂性的钦佩与接受的交织。

**ShapeTracker 中的形状变换**：小组探讨了 *ShapeTracker* 的局限性，结论是某些操作序列（如 `permute` 后接 `reshape`）会导致多个视图 (views)，从而在有效链接移动操作 (movement operations) 方面带来挑战。讨论了张量掩码 (tensor masking) 的实用性，重点强调了其在张量切片 (tensor slicing) 和填充 (padding) 中的作用。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **热烈欢迎全球创意人士**：友好的闲谈标志着新成员的加入，包括一位来自台湾的 **UI Designer**。
- **引导 AI 交互**：一位成员为与 AI 交互提供了明确的方向，引用了特定频道和 `@coral` 句柄以寻求帮助。
- **Cohere 扩大其多语言 AI 影响力**：Cohere 发布的 **Aya 23 模型** 预示着新的进展，提供了拥有 [80 亿和 350 亿参数](https://huggingface.co/CohereForAI/aya-23-35B) 的工具，并宣称支持涵盖 23 种语言的语系范围。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **GraphRAG 在图建模信息中获得关注**：成员们讨论认为，当源数据天然具有图结构时，**GraphRAG** 表现出色，但对于其他数据格式可能不是最佳选择。
  
- **PySpark 加速 Embedding 转换**：AI 工程师正在尝试使用 **PySpark pandas UDF**，以潜在地提高 Embedding 处理的效率。

- **Pinecone 持久化的挑战**：社区内的一个共同挑战集中在 **persistence handling**（持久化处理）与 Pinecone 中频繁创建实例之间的低效问题，并对 *pickle* 等主流解决方案表示不满。

- **API 和 Instruction Tuning 成为焦点**：即将于 2024 年 5 月 23 日举行的活动“如何使用 LangSmith 开发用于生成式 AI 药物研发生产的 API”，以及一段新的 [YouTube 视频](https://youtu.be/jddSbTLw0gc?si=spk1NEQMbr0iG1vt) 解释了 Instruction Tuning 对增强 LLM 遵循人类指令的好处。

- **代码修改和 Retriever 规划**：工程师们目前正在寻求用于规划代码更改的高效 Retriever，以及防止 LLM 在建议修改时削减现有代码的技术。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Mistral 在词汇量和功能上得到提升**：[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) 的最新迭代版本现在拥有 **32768 个 token 的扩展词汇量**、**v3 Tokenizer 支持**以及 function calling 能力，通过 `mistral_inference` 即可轻松安装。

- **Mistral 7B 的增强功能获得社区认可**：[Mistral-7B instruct 版本](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) 的发布得到了 Eldar Kurtic 的认可，并暗示会有更多改进，详见其[最近的推文](https://twitter.com/_EldarKurtic/status/1793407795909599325?t=zhtA3A5nq23HfUBkt441mQ&s=19)。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **GaLore 和 InRank 取得新突破**：与 Jiawei Zhao 的会议深入探讨了 **Gradient Low-Rank Projection (GaLore)** 和 **Incremental Low-Rank Learning (InRank)**，这些技术可以减少内存使用并增强大规模模型训练性能。

- **活动同步难题**：有人询问如何将活动日历与 Google Calendar 集成，强调了跟踪即将到来的讨论以避免错过的需求。

- **使用 ImageMAE 进行图像识别标志着可扩展性的飞跃**：分享了 ImageMAE 论文，提出了一种使用 masked autoencoders 进行计算机视觉的可扩展自监督学习方法，原生 ViT-Huge 模型取得了 87.8% 的优异结果。

- **社区热情高涨**：一位成员表达了对该频道的赞赏，认为它是 AI 领域分享和学习的宝贵资产。

---

# PART 2: 频道详细摘要与链接



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1242914896065335408)** (1009 条消息🔥🔥🔥): 

- **宣布支持 Mistral v3**：Unsloth 现在支持 [Mistral v3](https://twitter.com/danielhanchen/status/1793354069987529100)。成员们迅速进行了测试，并分享了关于训练损失和性能的褒贬不一的初步反馈。

- **关于 LLaMa 3 Reserved Tokens Bug 的讨论**：用户讨论了 LLaMa 3 基础权重中“有缺陷”的 reserved tokens，包括潜在的修复方法及其对 instruct 模型的影响。一位成员指出：“LLaMa 3 的一些基础（非 instruct）权重是有‘bug’的。Unsloth 会自动修复这个问题。”

- **关于 GPU 资源利用率的辩论**：对于在 79GB H100 GPU 上运行 Mistral 7B 时利用率不足的问题存在困惑。建议多种多样，从增加 batch size 到更新训练代码，一位用户指出：“你需要增加 batch 才能充分利用 GPU。”

- **Phi 3 Medium 4-bit 发布**：[Phi 3 Medium 4k Instruct](https://huggingface.co/unsloth/Phi-3-medium-4k-instruct-bnb-4bit) 现已推出，更多支持即将到来。公告中包含了 [Colab notebooks](https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing) 链接以便于访问。

- **持续预训练（Continued Pre-training）Notebook**：分享了一个用于持续预训练的 notebook，旨在特定领域微调期间保留指令遵循特性。可以从[这里](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)获取，鼓励成员们进行实验并分享结果。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/unsloth/Phi-3-medium-4k-instruct-bnb-4bit">unsloth/Phi-3-medium-4k-instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://discord.gg/M2spsbCbwN">加入 TheBloke AI Discord 服务器！</a>: 用于讨论和支持 AI Large Language Models 以及通用 AI。 | 23932 名成员</li><li><a href="https://huggingface.co/datasets/cognitivecomputations/samantha-data/tree/main">cognitivecomputations/samantha-data at main</a>: 未找到描述</li><li><a href="https://x.com/UnslothAI/status/1793694923683938416">来自 Unsloth AI (@UnslothAI) 的推文</a>: 我们非常高兴地宣布 Unsloth 加入了 2024 @GitHub Accelerator 计划！🦥 如果你想轻松微调像 Llama 3 这样的 LLM，现在正是最佳时机！ http://github.blog/2024-05-2...</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth 更新：支持 Mistral 及更多</a>: 我们很高兴发布对 Mistral 7B、CodeLlama 34B 以及所有基于 Llama 架构模型的 QLoRA 支持！我们添加了 sliding window attention、初步的 Windows 和 DPO 支持，以及 ...</li><li><a href="https://datta0.substack.com/p/ai-unplugged-11-lora-vs-fft-multi">AI Unplugged 11: LoRA vs FFT, Multi Token Prediction, LinkedIn 的 AI 助手</a>: 洞察胜过信息</li><li><a href="https://x.com/q_brabus/status/1793227643556372596">来自 QBrabus eu/acc (@q_brabus) 的推文</a>: @apples_jimmy @ylecun @iamgingertrash 提问：关于即将推出的 LLaMa 3 400B+ 模型，它会是 open-weight 吗？有很多传闻... 回答：不，它仍计划是开源的...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1b10o36/creepy_imprint_from_stable_diffusion/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://youtube.com/shorts/o4kVe2NwRYY?si=ILtLzWy1XTAPALKc">can i get a chicken tendie combo please</a>: 未找到描述</li><li><a href="https://youtu.be/e3Gvq4NDqvw?si=3b2lILNAiR5CZJMW">Scarlett Johansson 在 OpenAI 发布与其声音“惊人相似”的语音后要求给出解释</a>: Scarlett Johansson 正向 OpenAI 及其 CEO Sam Altman 寻求解释，此前该公司发布了一个 ChatGPT 语音，她称该声音听起来与她自己的声音“惊人相似”...</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit: 将算力和书籍转换为 Instruct-Tuning 数据集</a>: 将算力和书籍转换为 Instruct-Tuning 数据集 - e-p-armstrong/augmentoolkit</li><li><a href="https://lu.ma/1wu5ppl5">GPU 优化研讨会 · Luma</a>: 我们正在举办一场关于 GPU 优化的研讨会，邀请了来自 OpenAI、NVIDIA、Meta 和 Voltron Data 的顶尖演讲者。活动将在 YouTube 上直播，并且…</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 微调 Llama 3, Mistral &amp; Gemma LLM 提速 2-5 倍，显存占用减少 80%</a>: 微调 Llama 3, Mistral &amp; Gemma LLM 提速 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/shimmyshimmer/unsloth">GitHub - shimmyshimmer/unsloth: QLoRA 微调提速 5 倍，显存占用减少 60%</a>: QLoRA 微调提速 5 倍，显存占用减少 60%。通过在 GitHub 上创建账户为 shimmyshimmer/unsloth 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1243306406195626034)** (1 条消息): 

- **Phi-3 和 Mistral v3 通过新模型增强 Unsloth**：Unsloth 现在支持 Phi-3、Mistral v3 以及其他[新模型](https://github.com/unslothai/unsloth/releases/tag/May-2024)。此次更新还解决了所有 Llama 3 的问题，提升了 finetuning 性能。
  
- **免费 Notebooks 方便快速上手**：用户可以使用 [Phi-3 medium notebook](https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing)、[Mistral v3 notebook](https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing) 或 [ORPO notebook](https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing) 探索新模型。

- **广泛支持 4-bit 模型**：访问 Unsloth 的 [Hugging Face 页面](https://huggingface.co/unsloth) 获取包括 Instruct 在内的各种 4-bit 模型。现在已支持 [Qwen](https://github.com/unslothai/unsloth/pull/428) 和 [Yi 1.5](https://huggingface.co/unsloth/Yi-1.5-6B-bnb-4bit) 等新成员。

- **GitHub 2024 Accelerator 的好消息**：Unsloth 入选了 [2024 GitHub Accelerator](https://github.blog/2024-05-23-2024-github-accelerator-meet-the-11-projects-shaping-open-source-ai/)，与另外 11 个塑造开源 AI 的项目并列。这一认可凸显了其在 AI 社区中的重大影响力和创新性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/phi3">使用 Unsloth 进行 Phi-3 finetune</a>：通过 Unsloth 轻松 finetune Microsoft 的新模型 Phi 3 medium、small 和 mini，并获得 6 倍长的 context lengths！</li><li><a href="https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing)">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing)">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing)">Google Colab</a>：未找到描述</li><li><a href="https://github.blog/2024-05-23-2024-github-accelerator-meet-the-11-projects-shaping-open-source-ai/)">2024 GitHub Accelerator：认识 11 个塑造开源 AI 的项目</a>：宣布第二批入选名单，为项目提供价值，并驱动新的前沿。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1242916671476465895)** (81 条消息🔥🔥): 

<ul>
<li><strong>Mistral v0.3 的兴奋与困惑：</strong> 用户最初对 "Mistral v3" 感到兴奋，但后来澄清是 Mistral v0.3。一位用户开玩笑说这是 LLM 之间的奥林匹克级竞争。</li>
<li><strong>Qwen 的 QKVO 训练问题：</strong> 成员们讨论了使用 Unsloth 为 Qwen 模型训练 QKVO 层的问题，指出 bias 会导致错误。建议采用不训练所有层的变通方案。</li>
<li><strong>Unsloth 更新提升效率：</strong> 最新的 Unsloth 分支将 <em>lm_head</em> 和 <em>embed_tokens</em> 卸载到磁盘，减少了 VRAM 占用并加快了操作速度。引导用户通过特定的 GitHub 命令进行更新。</li>
<li><strong>Hugging Face 上的模型加载困惑：</strong> 用户讨论了在 Hugging Face 的一个 repository 中存放多个量化版本模型的复杂性，并建议为每个量化版本使用独立的 repository。</li>
<li><strong>LLM 的事实核查和非英语 finetuning：</strong> 深入讨论了 LLM 生成真实回答的挑战以及管理特定语言 finetuning 的问题。提供了多个参考资料和链接，包括 <a href="https://arxiv.org/abs/2306.03341">scaling-monosemanticity</a> 和 <a href="https://arxiv.org/abs/2302.00083">In-Context RALM</a>。</li>
</ul>
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/unsloth/llama-3-8b">unsloth/llama-3-8b · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2302.00083">In-Context 检索增强语言模型</a>：检索增强语言建模 (RALM) 方法在生成过程中根据来自基础语料库的相关文档对语言模型 (LM) 进行调节，已被证明能显著改善语言...</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth：以 2-5 倍的速度和减少 80% 的内存消耗来 finetune Llama 3、Mistral 和 Gemma LLM</a>：以 2-5 倍的速度和减少 80% 的内存消耗来 finetune Llama 3、Mistral 和 Gemma LLM - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1242971620356657274)** (1 条消息): 

```html
- **计划停机公告**：请注意，今晚 12:00am EST 将进行**计划停机**。停机将持续约 30 分钟，用于数据库升级，旨在提升性能和用户体验。
```
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1242916156700889141)** (897 messages🔥🔥🔥): 

```html
- **Gemini 免费使用令人惊喜**：成员们庆祝 **AI Studio 中的 Gemini** 即使在大规模使用下也是免费的（*“此 UI 中的请求是免费的”*），并对可以免费进行微调（*“免费微调？”*）感到兴奋。他们讨论了潜在的数据隐私问题，但对尝试该服务的开放态度占据了上风。
- **Perplexity 的速度令人印象深刻**：**Web scraping** 优化显著提升了**多源搜索**的性能，速度比之前的尝试快得多。一位成员报告称 *“Web scraping 耗时 1.52 秒，而之前超过 7 秒”*，并强调了正确使用并行处理的重要性。
- **Perplexity 与其他 AI 工具的对比**：成员们将 **Perplexity** 与 **Gemini Pro** 和 **ChatGPT** 在文件处理和数据处理方面进行了对比。**Perplexity** 因其研究和写作能力（*“在两个领域都更好”*）以及灵活的文件处理而受到称赞，同时也对 **Gemini** 主要在上下文处理方面的作用有了新的见解。
- **在 Perplexity 中集成额外功能**：讨论包括潜在的 UI 增强和 **Perplexity** 的工具，包括将 **labs 集成到主 UI** 中，并添加历史保存和对 **CSV** 等格式的支持。目标是潜在地将 **Perplexity** 从一个不错的工具转变为 *“最好的 AI 网站”*。
- **模型使用和速率限制挑战成员**：在遇到 **API rate limits** 并探索各种模型时，成员们在 **Haiku**、**Cohere** 和 **GPT-4-free** 之间权衡，分享了在免费和高性价比层级下的挫败感和最佳使用策略。他们在强调准确性和上下文大小平衡的同时，探索了替代方案和变通方法。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.loom.com/share/8cc0ce3bff334b64b558f5a04bdcf2b2?sid=ff6eb1ac-2ba9-434b-ad1b-bc372d2ac9b0">探索 AI 对话和冗余</a>: 在这段视频中，我分享了我在 AI 对话方面的经验，并强调了冗余问题。我演示了简单的计算和转换如何导致 AI 模型产生重复的响应...</li><li><a href="https://www.theregister.com/2016/03/23/npm_left_pad_chaos/">一名开发者如何用 11 行 JavaScript 代码搞垮了 Node、Babel 和数千个项目</a>: 从 NPM 提取的代码——每个人都在使用它</li><li><a href="https://youtu.be/NPOHf20slZg">价值 30,000,000 美元的 AI 背后隐藏着骗局</a>: 是时候看看兔子洞到底有多深了...支持调查新闻：► Patreon: https://patreon.com/coffeezilla 关注：►Ed Zitron: https://www.wh...</li><li><a href="https://forums.macrumors.com/threads/m4-ipad-pro-11-first-impressions-performance-heat-pwm-and-others.2426567/">M4 iPad Pro 11” 初步印象（性能、发热、PWM 等）</a>: 大家好！我刚从当地的 Apple Store 回来，一直在仔细检查和测试新款 M4 iPad Pro。这不是一篇评测，但它可以提供许多人可能感兴趣的有用细节...</li><li><a href="https://en.wikipedia.org/wiki/Oil_futures_drunk-trading_incident">原油期货醉酒交易事件 - 维基百科</a>: 未找到描述</li><li><a href="https://github.com/Rob--W/cors-anywhere/issues/301">公告：公共演示服务器 (cors-anywhere.herokuapp.com) 到 2021 年 1 月 31 日将受到严格限制 · Issue #301 · Rob--W/cors-anywhere</a>: CORS Anywhere 的演示服务器 (cors-anywhere.herokuapp.com) 旨在作为该项目的演示。但滥用已变得如此普遍，以至于托管演示的平台 (Heroku) 要求我...</li><li><a href="https://techcrunch.com/2024/05/23/bing-is-down-bringing-duckduckgo-and-ecosia-down-too">Bing 的 API 宕机，导致 Microsoft Copilot、DuckDuckGo 和 ChatGPT 的网页搜索功能也随之瘫痪 | TechCrunch</a>: 微软的搜索引擎 Bing 周四在欧洲运行异常数小时。起初，我们注意到无法执行网页搜索</li><li><a href="https://docs.anthropic.com/en/api/rate-limits);">欢迎来到 Claude - Anthropic</a>: 未找到描述</li><li><a href="https://www.anthropic.com/contact-sales">联系 Anthropic</a>: Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释和可控的 AI 系统。</li><li><a href="https://www.apple.com/ipad/compare/?modelList=ipad-pro-13-m4,ipad-pro-11-m4,ipad-air-11-m2">iPad - 型号对比</a>: 比较 iPad Pro、iPad Air、iPad 和 iPad mini 型号的分辨率、尺寸、重量、性能、电池寿命和存储空间。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1242944272576286730)** (7 条消息): 

- **台积电（Taiwan Semiconductor）引发好奇**：一位用户链接了一个关于台积电的 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/Taiwan-Semiconductor-remote-k.5AQq3LQkGX5eg4Nbh9jA)，可能在讨论远程办公或行业发展的相关方面。
- **用 AI 幽默分析气味**：另一位用户分享了一个幽默的 [搜索结果](https://www.perplexity.ai/search/Wie-riecht-der-SgLtFy.iRF6KuPxZuKQHVg#0)，表达了对某物气味的好奇，展示了 Perplexity AI 处理广泛查询的能力。
- **IoS 发生的致命事件**：一起涉及 9 人死亡的悲剧性事件引起了关注，用户分享了指向该[事件搜索](https://www.perplexity.ai/search/9-killed-in-IOsCm6NBREimQdcGey_1fQ#0)的链接。搜索内容可能涉及对该事件的解读或官方报告。
- **Bing API 话题浮现**：对 [Bing API 功能](https://www.perplexity.ai/search/Bings-API-is-Plv4H_4RT7ShLq2XT41R2A)的兴趣促成了一个搜索链接的分享，内容可能涵盖了 Bing API 的评价或使用方式。
- **阐释 Perplexity AI**：一位用户分享了一个解释 [什么是 Perplexity AI](https://www.perplexity.ai/search/What-is-Perplexity-uyV3gThHQEa1tWgRyN0sQw) 的搜索链接。这表明用户对该平台本身持续保持好奇并进行学习。
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1242925804300533911)** (2 条消息): 

- **Metquity 决定探索替代方案**：Metquity 表达了转向其他工具的计划，并表示：*“也许我会先用其他工具构建，等它变得更好时再回来。”* 这表明其有兴趣在改进后重新使用。
- **Neuraum 注意到网页版与 API 输出之间的差异**：Neuraum 询问为什么使用相同的模型和提示词（Prompt）在网页版和 API 上会产生不同的输出。他指出：*“使用 API 时输出是错误的，尽管浏览功能可以正常工作，”* 试图寻求关于这种不一致性的见解。
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1242916745883156581)** (141 条消息🔥🔥): 

- **针对 ColBERT 风格模型的指令微调（Instruction Tuning）**：一位成员询问了其他人对嵌入模型（Embedding Models）进行指令微调的经验，特别是针对 ColBERT 风格的模型。他们分享了一些相关论文，包括 [INSTRUCTOR: A Framework for Embedding Text with Task Instructions](https://arxiv.org/abs/2212.09741) 和 [TART: A Multi-task Retrieval System Trained on BERRI with Instructions](https://arxiv.org/abs/2211.09260)。
  
- **FDP 考试中的贝叶斯计算问题**：另一位成员讨论了 FDP 认证考试中的一个贝叶斯问题，并进行了计算推导。他们指出 FDP 协会的概率计算中可能存在错误，并更倾向于自己的计算方法。

- **JarvisLabs 上的 Axolotl 教程**：分享了一个关于如何在 JarvisLabs 上运行 Axolotl 的实用教程视频，可在 [YouTube](https://youtu.be/Y9464wasHuE) 上观看，并链接了 JarvisLabs 和 Axolotl GitHub 仓库等相关资源。

- **用于 AI/ML 环境的 Miniforge/Mamba**：讨论了使用 Miniforge 和 Mamba 创建和管理 conda 环境相比 pyenv 等替代方案的优势，强调了 mamba 的灵活性和速度优势。

- **Schulman 的两步微调过程**：成员们讨论了 John Schulman 关于迭代监督微调（SFT）与强化学习（RL）在基础微调之外改进模型方面的评论，强调了通过迭代过程使模型与高质量人类数据完全对齐的重要性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/leaderboard,">AI Mathematical Olympiad - Progress Prize 1 | Kaggle</a>：未找到描述</li><li><a href="https://predibase.com/">Predibase: 用于 Fine-tuning 和部署 LLM 的开发者平台</a>：在托管于私有云的尖端基础设施上，以最快、最简单的方式对任何开源大语言模型进行 Fine-tuning 和部署。</li><li><a href="https://www.loom.com/share/30d3b2e054f142fda5d905f95fedc29f">探索 Honeycomb 示例的 Fine-tuning</a>：在本视频中，我将带你了解使用 honeycomb 示例进行模型 Fine-tuning 的过程。我提供了关于克隆仓库、安装依赖以及运行...的逐步说明。</li><li><a href="https://x.com/younesbelkada/status/1793211607713235295">younes (@younesbelkada) 的推文</a>：🚨 @huggingface transformers Trainer 中的新优化器 🚨 LOMO 优化器现在可以在 transformers 库中使用 https://github.com/OpenLMLab/LOMO LOMO 作者们的出色工作！🧵</li><li><a href="https://youtu.be/Y9464wasHuE">如何在 JarvisLabs 上运行 axolotl | 教程</a>：在 JarvisLabs 上查看 axolotl：jarvislabs.ai/templates/axolotl 查看 axolotl GitHub：https://github.com/OpenAccess-AI-Collective/axolotl</li><li><a href="https://www.youtube.com/watch?v=Z2NxN9sl9Vk">Vincent D. Warmerdam - 主动教学，人类学习</a>：想要一个用于 ML 的数据集？互联网说你应该使用...主动学习（active learning）！这主意不错。当你创建自己的训练数据时，你通常希望...</li><li><a href="https://www.youtube.com/watch?v=gDk7_f3ovIk">批量标注与 Prodigy</a>：Prodigy 是一款由 spaCy 开发者开发的现代标注工具，用于收集机器学习模型的训练数据。在本视频中，我们将展示一个...</li><li><a href="https://www.astralcodexten.com/p/asteriskzvi-on-californias-ai-bill">Asterisk/Zvi 关于加州 AI 法案</a>：...</li><li><a href="https://youtu.be/C9p7suS-NGk?si=AM4sr3OXeFRKZo7c">Vincent Warmerdam - 主旨演讲 "Natural Intelligence is All You Need [tm]"</a>：在本次演讲中，我将尝试向你们展示，如果你允许自己拥有偶尔重新思考和重塑常规做法的创作自由，会发生什么。正如...</li><li><a href="https://arxiv.org/abs/2212.09741">一个 Embedder 适配任何任务：指令微调文本嵌入</a>：我们介绍了 INSTRUCTOR，一种根据任务指令计算文本嵌入的新方法：每个文本输入都与解释用例（例如任务和领域描述）的指令一起进行嵌入...</li><li><a href="https://arxiv.org/abs/2211.09260">带有指令的任务感知检索</a>：我们研究了带有指令的检索问题，即检索系统的用户在查询的同时明确描述其意图。我们的目标是开发一种通用的任务感知检索...</li><li><a href="https://youtu.be/sTQaJyrI-zg?si=krcLKWRmqT9SH8X5&t=1389),">斯坦福 CS25: V2 I 常识推理</a>：2023年2月14日 常识推理 Yejin Choi。在这个系列讲座中，我们研究了 Transformer 工作的细节，并深入探讨了不同种类的...</li><li><a href="https://github.com/conda-forge/miniforge?tab=readme-ov-file#install">GitHub - conda-forge/miniforge: 一个 conda-forge 发行版。</a>：一个 conda-forge 发行版。通过在 GitHub 上创建一个账户来为 conda-forge/miniforge 的开发做出贡献。</li><li><a href="http://annotate.calmcode.io">未找到标题</a>：未找到描述</li><li><a href="https://anywidget.dev/en/community/">社区 | anywidget</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=goaBFxGhp6Y),">使用 Widget 增强 Jupyter，访谈 anywidget 创作者 Trevor Manz</a>：在 Sample Space 的（第一！）集中，我们采访了 anywidget 的创作者 Trevor Mantz。这是一个（很棒的！）工具，可以帮助你构建更具交互性的 notebook...</li><li><a href="https://latent-space-xi.vercel.app/til/create-a-conda-env-for-axolotl">Latent Space</a>：未找到描述</li><li><a href="https://buttondown.email/ainews/archive/ainews-anthropic-cracks-the-llm-genome-project/">[AINews] Anthropic 的 "LLM 基因组计划"：在 Claude Sonnet 上学习并固定 3400 万个特征</a>：字典学习（Dictionary Learning）就是你所需要的一切。2024/5/20-2024/5/21 的 AI 新闻。我们检查了 7 个 subreddit、384 个 Twitter 和 29 个 Discord（376 个频道和 6363 条消息）...
</li>
</ul>

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1242928427158212650)** (14 messages🔥): 

- **自动化站会转录与工单更新**：提议的一项服务，通过读取站会会议的转录内容，并根据其中提到的状态更新工单。这涉及使用站会对话示例及其与工单更新的相关性进行 fine-tuning。
  
- **自定义停止序列以防止越狱**：成员们讨论了通过 fine-tuning 模型来使用自定义停止序列（如 hash），而不是像 "###" 这样的常用 token，旨在抵御越狱尝试。有人建议通过 fine-tuning 来忽略越狱 prompt，尽管承认预判每一个 prompt 具有挑战性。

- **轻量级文本打标和实体生成模型**：一位成员建议了几个轻量级 LLM 项目，包括一个类似于 GliNER 的文本打标和分类项目，以及另一个为 spaCy 等工具生成训练数据的项目。另一个有趣的提议是创建一个 LLM 来为 Python 包生成酷炫的名字。

- **Prompt Injection 防护**：讨论强调了直接在模型上完全防止越狱的可能性较低，并指出 [prompt injection protection tools](https://arxiv.org/pdf/2307.15043) 是更好的解决方案。分享的资源包括一份防护库/工具列表和 [相关论文集](https://huggingface.co/collections/leonardlin/prompt-injection-65dd93985012ec503f2a735a)。

- **EDA 助手与建议**：一个通过 [YouTube 演示](https://www.youtube.com/watch?v=glwBlONacPY) 确定的聊天机器人项目，旨在协助数据科学家进行时间序列数据的 EDA。版本 2 计划进行 fine-tuning 以提高演绎推理和格式化能力。另一个助手旨在将 EDA 输出处理为可操作的步骤，探索更具成本效益且更快速的实现方法。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=glwBlONacPY">Unleash the Power of GPT-4o: Exploratory Data Analysis with ObexMetrics</a>: 使用 ObexMetrics 先进的 EDA 助手（由 GPT-4o 驱动）彻底改变您的时间序列数据分析。与您的数据无缝聊天，轻松实现...</li><li><a href="https://llm-tracker.info/research/Prompt-Injection-Protection">Prompt Injection Protection</a>: 我们应该有一个工具来跟踪 GitHub 项目活动……论文集：https://huggingface.co/collections/leonardlin/prompt-injection-65dd93985012ec503f2a735a 技术：输入启发式...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1242927291856588941)** (18 messages🔥): 

- **Modal 额度发放迅速**：一位用户确认在填写表单后很快收到了 Modal 额度。另一位用户对快速分配表示赞赏，并感谢团队提供的免费额度。
- **微调 llama 3 8b 时持续报错**：一位成员报告在运行 llama 3 8b fine-tuning 任务时持续遇到问题，尽管得到了另一位用户的帮助。他们链接到了一个特定的 Discord 消息线程以获取更多背景信息（[线程链接](https://discord.com/channels/1238365980128706560/1242939065058328776/1242952109981302945)）。
- **额度过期的奇怪现象**：一位用户注意到一个似乎是奇怪现象的问题，即额度在计费面板中显示为隔夜过期，但 Live Usage 下拉菜单仍显示全部金额。他们感谢团队提供的澄清。
- **微调 LLM 以生成布局**：一位用户询问了使用 publaynet 和 rico 等数据集 fine-tuning LLM 以生成布局的可行性。这些请求凸显了用户正在探索的模型多样化应用。
- **将训练好的模型下载到 S3**：另一位成员澄清说，可以使用 `modal volume get` 命令将训练好的模型下载到 S3 bucket。他们还提到了直接将 S3 bucket 挂载为 volume 的选项，并链接到了 [相关文档](https://modal.com/docs/guide/cloud-bucket-mounts)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://modal.com/docs/guide/cloud-bucket-mounts">Cloud bucket mounts</a>: modal.CloudBucketMount 是一种可变卷，允许从云端 bucket 读取和写入文件。它支持 AWS S3、Cloudflare R2 和 Google Cloud Storage bucket。</li><li><a href="https://modal.com/docs/reference/cli/volume#modal-volume-get">modal volume</a>: 读取和编辑 modal.Volume 卷。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1243037076140720189)** (1 条消息): 

- **Hamel 分享的说明性示例**：一位成员赞赏了 Hamel 在 Twitter 上分享的一个示例，称其“非常有启发性”。他们还感谢了 Jeremy Howard 提供的相关 Notebook，并提供了 [推文](https://twitter.com/HamelHusain/status/1793319488731107718) 和 [GitHub 仓库](https://github.com/fastai/lm-hackers/blob/main/lm-hackers.ipynb) 的链接。

**提到的链接**：<a href="https://github.com/fastai/lm-hackers/blob/main/lm-hackers.ipynb">lm-hackers/lm-hackers.ipynb at main · fastai/lm-hackers</a>：黑客语言模型指南。通过在 GitHub 上创建账户为 fastai/lm-hackers 的开发做出贡献。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1242922773299466300)** (32 条消息🔥): 

- **解决 Llama-3 访问问题**：包括 [@mark6871 和 @dhar007](https://discord.com/channels/****) 在内的多位用户在拥有 Hugging Face 权限的情况下仍面临访问 Llama-3 模型的问题。解决方案是在 Hugging Face 上生成访问令牌（access token），并在终端提示时输入。

- **克服 CUDA 显存错误**：[@dhar007](https://discord.com/channels/****) 等用户在使用 RTX5000 GPU 训练模型时遇到了 “CUDA out of memory” 错误。他们通过调整 batch size 并使用 `nvidia-smi` 监控 GPU 使用情况解决了该问题。

- **Mistral LoRA 训练数据点**：[@damoncrockett](https://discord.com/channels/****) 分享了运行 Mistral LoRA 示例的经验，在单张 A100 GPU 上耗时 2.5 小时，花费 4 美元，但指出在小数据集上仅进行一个 epoch 的训练会导致欠拟合（undertraining）。

- **Jarvislabs 额度的反馈与支持**：[@rashmibanthia](https://discord.com/channels/****) 等用户对赠送的额度表示感谢，并分享了相比其他服务，在 Jarvislabs 上的良好体验。此外，还有关于额度缺失以及如何确认注册的咨询（[@manjunath_63072] 和 [@darkavngr](https://discord.com/channels/****)）。

- **关于 Jarvislabs 竞价实例（Spot Instances）和仓库保存的查询**：[@tokenbender](https://discord.com/channels/****) 在寻找竞价实例时遇到困难，而 [@nisargvp](https://discord.com/channels/****) 询问了如何在不暂停实例的情况下保存仓库，以避免产生额度消耗。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1243171899903705119)** (4 条消息): 

- **通过登录提示解决受限仓库（Gated repo）访问问题**：一位成员在接受条款后尝试访问受限仓库时报错。另一位成员建议使用 `huggingface-cli login` 命令，这解决了问题并使训练得以成功进行。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1242919849215266868)** (5 条消息): 

- **GitHub 注册邮箱不匹配提醒**：一位成员提醒道：“提醒所有使用 GitHub 注册但参会邮箱不同的用户，注册后可以设置不同的电子邮箱地址。”
- **Gmail “+” 号问题**：用户表达了挫败感，因为系统“似乎不接受带有 `+` 号的 Gmail 地址”。
- **通知标签困惑与 Maven 额度**：一位用户询问使用 Maven 注册地址接收通知是否能确保自动添加额度。
- **询问额度到账情况**：一位成员询问群组：“有人已经拿到额度了吗？”
- **等待额度说明**：另一位用户提到他们已经注册，“现在只能等待。希望能有关于进展的官方信息。”
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[kylecorbitt_prompt_to_model](https://discord.com/channels/1238365980128706560/1242221891733946490/1243196011443126312)** (2 条消息): 

- **参会者询问是否有录像**：一位参与者急切地问道：“会有录像吗？”这表明了对即将举行的活动的期待以及事后回顾的兴趣。

- **对即将举行的活动感到兴奋**：另一位参与者表达了热情，说他们“对稍后的演讲感到非常兴奋！”（Hyped for the talk later!）。他们的兴奋之情通过使用表情符号和闪烁符号进一步体现。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1242919179028529242)** (209 条消息🔥🔥):

- **Axolotl 的数据集故障排除**：用户在本地机器上运行 finetuning 时遇到问题，导致 JSON 解码错误。为了解决此问题，需正确对齐配置中的数据集路径，并使用 [Axolotl 文档](https://github.com/OpenAccess-AI-Collective/axolotl) 中演示的兼容格式。

- **LoRA 超参数调优**：讨论集中在调整 learning rates 和 LoRA 超参数。一份共享的 config 显示了 `lora_r: 128`、`lora_alpha: 128` 以及不同的 learning rates 以优化模型训练，并借鉴了 [Sebastian Raschka 的建议](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)。

- **训练时长查询**：用户讨论了在不同 GPU 上的训练时间，并注意到速度预估的差异。Hamel 建议运行较小的数据集样本以避免过长的反馈循环，而共享的 axolotl 示例有助于指导合理的预期和调整。

- **Function calling 的配置**：用户寻求针对特定任务（如使用 ReactFlow 示例进行 text to code 转换）进行 fine-tuning 模型的最佳实践。建议使用特定于其模型的模板和 prompt 格式，包括查看 [fine-tuning benchmarks](https://predibase.com/fine-tuning-index)。

- **用于 L2 evals 的 GPT 裁判**：对于 fine-tuning 评估，建议使用 GPT-4 作为裁判并配合精炼的 prompt。用户曾考虑 fine-tuning 裁判模型，但被建议先从简单的 prompt 精炼开始，以提高对齐效果。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://wandb.ai/muellerzr/llama-3-8b-self-align-axolotl">muellerzr</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://www.loom.com/share/30d3b2e054f142fda5d905f95fedc29f">探索 Honeycomb 示例的微调</a>: 在这段视频中，我将带你了解使用 honeycomb 示例微调模型的过程。我提供了关于克隆仓库、安装依赖和运行...的逐步说明。</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/syllabus/modules/ac50ed?item=49gdm84u3wp">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit: 将算力和书籍转换为指令微调数据集</a>: 将算力和书籍转换为指令微调数据集 - e-p-armstrong/augmentoolkit</li><li><a href="https://prodi.gy/docs/large-language-models#more-config">大语言模型 (LLMs) · Prodigy · 一款用于 AI、机器学习和 NLP 的标注工具</a>: 一款可下载的标注工具，用于 NLP 和计算机视觉任务，如命名实体识别、文本分类、目标检测、图像分割、A/B 测试评估等。</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/syllabus/modules/ac50ed?item=bf4nff4j6bo">未找到标题</a>: 未找到描述</li><li><a href="https://predibase.com/fine-tuning-index">微调指数</a>: 超过 700 个开源 LLMs 微调的性能基准测试</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/777">支持 Fuyu-8B · Issue #777 · OpenAccess-AI-Collective/axolotl</a>: ⚠️ 请检查此功能请求之前是否已被提议。我搜索了讨论区之前的 Ideas，没有发现类似的功能请求。我搜索了之前的 Issues，没有...</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html">Axolotl - 指令微调</a>: 未找到描述</li><li><a href="https://lucasvw.github.io/posts/19_llm_fine_tuning/.">Lucas van Walstijn - LLM 微调入门</a>: 未找到描述</li><li><a href="https://huggingface.co/nisargvp/hc-mistral-alpaca">nisargvp/hc-mistral-alpaca · Hugging Face</a>: 未找到描述</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html">Axolotl - 无模板提示词构建</a>: 未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-small-128k-instruct">microsoft/Phi-3-small-128k-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://gist.github.com/nisargvp/2fcbf41d4a8e7149e6c4fb9a630edfd8">在本地运行 hc.yml 时出现 dataset_generation 错误</a>: 在本地运行 hc.yml 时出现 dataset_generation 错误 - gist:2fcbf41d4a8e7149e6c4fb9a630edfd8</li><li><a href="https://m.youtube.com/watch?v=lzXKsY3bANw">使用 scikit-learn 进行图像分类</a>: Scikit-Learn 以构建表格数据的机器学习模型而闻名，但这并不意味着它不能进行图像分类。在这段视频中...</li><li><a href="https://reactflow.dev/.">React 中的基于节点的 UI – React Flow</a>: 高度可定制的 React 库，适用于工作流构建器、无代码应用、图像处理、可视化工具等</li><li><a href="https://forum.obsidian.md/t/obsidian-vscode-editor-elevate-your-code-editing-experience-in-obsidian/69057">Obsidian VSCode 编辑器：提升你在 Obsidian 中的代码编辑体验！</a>: 你是否厌倦了在处理笔记和代码时在不同应用程序之间切换？你是否希望有一种无缝的方式在 Obsidian 中查看和编辑代码文件？看这里就对了！😏 ...</li><li><a href="https://github.com/parlance-labs/ftcourse">GitHub - parlance-labs/ftcourse</a>: 通过在 GitHub 上创建一个账户来为 parlance-labs/ftcourse 的开发做出贡献。</li><li><a href="https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms">使用 LoRA (Low-Rank Adaptation) 微调 LLMs 的实用技巧</a>: 我从数百次实验中学到的经验</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1652">由 winglian 提交的在数据集上加载显式分片 · Pull Request #1652 · OpenAccess-AI-Collective/axolotl</a>: 让加载部分分片变得更容易，例如：datasets:   - path: ...     type: ...     split: &quot;train[:10%]&quot;</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/dataset-formats/inst_tune.qmd#L7">axolotl/docs/dataset-formats/inst_tune.qmd (main 分支) · OpenAccess-AI-Collective/axolotl</a>: 尽管提出 Axolotl 相关问题。通过在 GitHub 上创建一个账户来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://www.phorm.ai/query?projectId=e315ba4a-4e14-421f-ab05-38a1f9076f25&threadId=eff87042-122e-4774-8526-0a023e3e919f">在数据集上使用分片 | OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>: Unders...</li>

更快速地理解代码。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1242939065058328776)** (80 条消息🔥🔥): 

- **Llama 3 微调问题**：成员们讨论了在微调 Llama 3 时遇到的各种问题，例如在 Modal 中遇到 `NoneType` 错误以及 CUDA 相关问题。解决方案包括使用 Docker 镜像、调整 `sequence_len`，以及参考配置（如[此配置](https://wandb.ai/oaaic/fused-cel-llama3/runs/kkyhjjh6/files/tmp/axolotl_config_rdbefq2r.yml)）。

- **Axolotl 的 Docker 解决方案**：许多成员建议使用 Docker 容器来解决 CUDA 库和路径错误的本地设置问题。分享了预制 Docker 镜像的链接（例如来自 [Docker Hub](https://hub.docker.com/layers/winglian/axolotl/main-20240522-py3.11-cu121-2.2.2/images/sha256-47e0feb612caf261764631a0c516868910fb017786a17e4dd40d3e0afb48e018) 的镜像）以及 Jarvis Labs 上的设置。

- **BitsAndBytes GPU 支持**：关于 BitsAndBytes 不支持 GPU 的重复错误通过在 `.bashrc` 中修正 CUDA 路径得到了解决。建议使用 Jarvis 或 Modal 等云服务商以减少问题，而不是进行本地安装。

- **不同 GPU 上的 Axolotl**：成员们指出 Flash Attention 2 不支持 Turing 架构的 GPU，导致在 T4 系统上出现问题。建议使用 `sdp_attention` 等替代方案，该方案通常在大多数模型上都受支持。

- **Axolotl 中的缓存管理**：用户讨论了在 JarvisLabs 上重新运行实验时的数据缓存问题。建议重命名数据集文件并更新配置文件以确保使用正确的数据，并呼吁增加缓存标志（caching flag）功能。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html">(Beta) Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA) — PyTorch Tutorials 2.3.0+cu121 documentation</a>: 未找到描述</li><li><a href="https://www.phorm.ai/query?projectId=e315ba4a-4e14-421f-ab05-38a1f9076f25">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快速地理解代码。</li><li><a href="https://jarvislabs.ai/templates/axolotl">Easily Finetune LLM with Axolotl | Jarvislabs</a>: Axolotl 帮助你使用 LoRA、QLoRA 等技术微调 LLM。编辑配置文件并开始 LLM 训练。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/cicd/Dockerfile.jinja">axolotl/cicd/Dockerfile.jinja at main · OpenAccess-AI-Collective/axolotl</a>: 尽管提问（axolotl questions）。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://wandb.ai/oaaic/fused-cel-llama3/runs/kkyhjjh6/files/tmp/axolotl_config_rdbefq2r.yml">oaaic</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/.github/workflows/tests.yml#L105-L107">axolotl/.github/workflows/tests.yml at main · OpenAccess-AI-Collective/axolotl</a>: 尽管提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/config.qmd">axolotl/docs/config.qmd at main · OpenAccess-AI-Collective/axolotl</a>: 尽管提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://hub.docker.com/layers/winglian/axolotl/main-20240522-py3.11-cu121-2.2.2/images/sha256-47e0feb612caf261764631a0c516868910fb017786a17e4dd40d3e0afb48e018?context=explore">no title found</a>: 未找到描述</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/src/common.py#L14">llm-finetuning/src/common.py at main · modal-labs/llm-finetuning</a>: 微调 Llama/Mistral/CodeLlama 等模型的指南 - modal-labs/llm-finetuning</li><li><a href="https://github.com/modal-labs/llm-finetuning/pull/57">better handling of pre-processing and training by winglian · Pull Request #57 · modal-labs/llm-finetuning</a>: 将预处理也作为训练步骤的一部分运行，但倾向于使用单 GPU 实例，合并也使用单 GPU，使用较新的 axolotl 镜像，添加 llama-3 配置</li><li><a href="https://modal.com/docs/examples/llm-finetuning">Fine-tune an LLM in minutes (ft. Llama 2, CodeLlama, Mistral, etc.)</a>: 厌倦了提示工程？微调通过调整模型权重以更好地适应特定任务，帮助你从预训练 LLM 中获得更多收益。本操作指南将帮助你利用基础模型...</li><li><a href="https://latent-space-xi.vercel.app/til/create-a-conda-env-for-axolotl">Latent Space</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1242929273845841930)** (96 条消息🔥🔥): 

```html
- **免费 GPU 优化活动**：Filippob82 宣布了一个关于 GPU 优化的研讨会，演讲者来自 OpenAI、NVIDIA、Meta 和 Voltron Data。该活动将在 YouTube 上直播，并在 [Discord](https://discord.gg/T5sx2MYd5R) 上讨论，更多详情见 [README](https://github.com/mlops-discord/gpu-optimization-workshop) 和 [研讨会笔记](https://docs.google.com/document/d/1TR_5Ax0rPqTj8I2sA7MH-aa4J7TUUt4Ji9272OP8ZJg/edit)。

- **A100 和 H100 训练技巧**：Stevenmerrill 询问了在 A100 和 H100 GPU 上训练的通用规则，Tddammo 建议如果 GPU 支持，务必使用 bf16，并调整 batch sizes 以利用可用的 VRAM。此外，由于显存容量提升，sequence lengths 也可以相应增加。

- **VRAM 计算挑战**：Remek1972 讨论了在 A6000 GPU 上处理较长 sequence lengths（例如 4096 tokens）时的 VRAM 需求问题，发现这会导致崩溃。对话结论是，使用混合精度（bf16）和优化策略可以缓解部分显存问题，但更大的模型可能需要 offloading 或 quantization。

- **Paged ADAMW 8-bit 优化器讨论**：Lhl 提到了为了效率使用 paged_adamw_8bit 优化器，并询问了潜在的缺点，得到了性能与普通 adam 等效的保证。他们讨论了社区的经验和发现，包括 8-bit 优化对显存占用的好处。

- **对最新优化器的关注**：Lhl 和 Tddammo 讨论了尝试 Sophia 和 Adam_LoMo 等新优化器的实验。建议和分享的经验指向了潜在的性能提升，Lhl 还添加了 Twitter 上关于新优化器 benchmarks 的近期讨论链接。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/younesbelkada/status/1793211607713235295">来自 younes (@younesbelkada) 的推文</a>: 🚨 @huggingface transformers Trainer 中的新优化器 🚨 LOMO 优化器现在可以在 transformers 库中使用 https://github.com/OpenLMLab/LOMO LOMO 作者们的出色工作！🧵</li><li><a href="https://x.com/ArmenAgha/status/1780149168692158658">来自 Armen Aghajanyan (@ArmenAgha) 的推文</a>: 最终更新：对 Sophia 进行了更高数量级的测试。我们讨论的是 B 级参数规模的模型和 T 级的 tokens。Sophia 再次胜出。至少对我来说，这是明确的证据表明 Sophia ...</li><li><a href="https://github.com/huggingface/transformers/issues/22101">[Benchmark] HF Trainer 优化器 (2023年3月) · Issue #22101 · huggingface/transformers</a>: 这是对 Adam torch vs. apex vs HF vs adafactor 在 RTX-3090、A100 上的重新运行，但增加了 BNB 的 8-bit Adam 优化器，而且软件在过去 14 个月中可能也有了改进/变化。注：8-bit Opt...</li><li><a href="https://github.com/huggingface/transformers/pull/24338">由 guilt 提交的 Add SophiaG. · Pull Request #24338 · huggingface/transformers</a>: 这个 PR 做了什么？这是一个草案 PR，展示了如何使用 Transformers 测试 Sophia。这绝非生产就绪，当然还需要考虑许可问题。但是，如果有人需要...</li><li><a href="https://lu.ma/1wu5ppl5">GPU 优化研讨会 · Luma</a>: 我们正在举办一场关于 GPU 优化的研讨会，演讲者阵容强大，来自 OpenAI、NVIDIA、Meta 和 Voltron Data。该活动将在 YouTube 上直播，并且……
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1242957873370107924)** (56 条消息🔥🔥): 

- **通过安装提示解决 Accelerate 命令问题**：一位成员在运行 `accelerate launch -m axolotl.cli.train` 时因缺少 torch 和 gcc 依赖而遇到问题，最终通过[详细教程](https://www.namehero.com/blog/how-to-install-gcc-on-ubuntu/#3-installing-gcc-compiler-on-ubuntu)安装 CUDA 和必要工具解决了该问题。另一位成员提供了一个使用 conda 和指定依赖项从零开始构建 Axolotl 的可用设置。
- **在 GPU VM 上为 Axolotl 实现 Docker**：针对有人尝试在带有 GPU 的 GCP 深度学习 VM 上将 Axolotl 作为 Docker 镜像运行的问题，分享了一个涉及 `winglian/axolotl:main-latest` Docker 镜像和示例 Docker 运行命令的解决方案。
- **Axolotl 配置中的默认参数值**：关于 Axolotl 配置中参数默认值的查询引发了讨论，一位成员表达了同样的好奇。该问题在讨论帖中得到了部分解答。
- **在 Modal 上运行 Axolotl 的挑战**：几位成员讨论了尝试在 Modal 上运行 Axolotl 的问题，遇到的错误可能是由于构建不匹配以及 Hugging Face 上受限（gated）模型的问题。分享了一个用于更好处理预处理和训练的 Pull Request 链接作为潜在解决方案。
- **LLM 微调的预处理建议**：一位成员询问了关于 LLM 微调数据集预处理的建议，提到了特征工程和特征选择。他们引用了 [Axolotl 数据集预处理文档](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/dataset_preprocessing.qmd)以供进一步阅读。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.namehero.com/blog/how-to-install-gcc-on-ubuntu/#3-installing-gcc-compiler-on-ubuntu">How To Install GCC On Ubuntu</a>：让我们逐步完成在 Ubuntu 系统上安装 GCC 的过程，让编译器和开发工具的世界变得触手可及！</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl.git">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>：尽管提问。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>：尽管提问。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/dataset_preprocessing.qmd">axolotl/docs/dataset_preprocessing.qmd at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/modal-labs/llm-finetuning/pull/57">better handling of pre-processing and training by winglian · Pull Request #57 · modal-labs/llm-finetuning</a>：将预处理也作为训练步骤的一部分运行，但倾向于使用单个 GPU 实例，合并也使用单个 GPU，使用较新的 axolotl 镜像，添加 llama-3 配置。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1243310332407971850)** (1 条消息): 

- **蛋白质可视化工具发布**：查看由社区贡献者创建的 [Proteinviz](https://huggingface.co/spaces/as-cle-bert/proteinviz)，用于创建自定义蛋白质视觉效果。该工具在最新的社区更新中被重点介绍。
- **使用 SDXL Flash 进行快速图像处理**：体验由另一位社区成员创建的 [SDXL flash](https://huggingface.co/spaces/KingNish/SDXL-Flash) 带来的快速结果。该 Space 提供了高效的图像处理能力。
- **创新的数据集和系统更新**：值得关注的包括下载量超过 1k 的 [wikipedia 数据集](https://huggingface.co/datasets/not-lain/wikipedia)，以及 [Mistral-7B](https://huggingface.co/spaces/ehristoforu/mistral-7b-v0.3-chat) 的超快演示。此外，还可以查看关于 Agentic AI 解决方案的启发性 [YouTube 视频](https://www.youtube.com/watch?v=S-gy6NUOGSs)。
- **创建透明图像及更多内容**：尝试使用 Diffusers 创建 [透明图像](https://github.com/rootonchair/diffuser_layerdiffuse)，并通过这个 [Instruction-tuned 模型讲解视频](https://youtu.be/jddSbTLw0gc) 了解指令微调模型。其他亮点包括在 AWS Trainium 上训练 MoE、训练自定义 AI 模型以及 AnthropicAI 研究中的有趣发现。
- **使用 IP-Adapter Inpainting 进行虚拟试穿**：探索使用 Inpainting 技术的新型虚拟 [试穿 (Try-On)](https://huggingface.co/blog/tonyassi/virtual-try-on-ip-adapter) 体验。这项创新允许用户利用先进的 AI 技术尝试虚拟服装。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=S-gy6NUOGSs)">Agentic AI Solutions / Adaptive AI Solutions - Episode 1:  CrewAI With Preston McCauley</a>: 在第 1 集中，我们简要介绍了 #AdaptiveAI 和 #Agentic AI 方法。https://www.linkedin.com/in/preston-mccauley-immersive-ux/Join Presto...</li><li><a href="https://youtu.be/jddSbTLw0gc)">What is an Instruction Tuned Model?</a>: 什么是 Instruction Tuning？什么是 Instruction Tuned 模型？什么是预训练模型？我该如何让我的 Large Language Model 遵循指令？这些...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1242917952727289966)** (591 条消息 🔥🔥🔥): 

- **对 Copilot+ 能力的困惑**：成员们讨论了 **Copilot+** 的能力，质疑其本地模型使用等功能是否真的能消除对持续云端连接的需求。一位成员评论道：“Copilot+ 意味着你有足够的 NPU 算力来运行像 Snapdragon X Elite 这样的所有本地模型。”

- **优化用于聊天机器人的模型**：讨论集中在如何在不依赖 MoE 的情况下选择专门用于 Chatbot 功能的模型。一位成员提到：“来自 HuggingFace 团队的 Zephyr 7B 是对 Mistral 7B 的一次出色微调。”他们强调在个人用例上进行实际测试比基准测试更重要。

- **关于模型训练和微调的关注**：关于使用极少量数据和训练轮数（Epochs）进行模型微调的咨询非常普遍。一段对话围绕着使用 80 条消息训练 **Mistral** 使其相信自己是一个 25 岁的年轻人展开，这表明少量数据仍然可以奏效。

- **关于无审查模型实际用途的辩论**：成员们强调了无审查模型在 ERP（成人角色扮演）之外的能力，指出它们具有更广泛的对话和角色扮演用途。一位成员开玩笑说：“为了好玩、学习，以及引导我的欲望，”但随后澄清他们使用无审查模型是为了各种有趣的互动。

- **技术问题与社区协助**：多位成员请求帮助解决技术障碍，例如处理项目中的运行时错误以及为 AutoTrain 微调正确格式化数据。分享的链接指向了 [HuggingFace 文档](https://hf.co/docs/autotrain) 以获取指导。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discord.gg/AfCdAWnE)">Discord | 你的沟通与聚会场所</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。与你的朋友和社区交谈、聊天、聚会并保持紧密联系。</li><li><a href="https://tenor.com/view/mewing-locked-in-funny-gif-2909757877821689206">Mewing Locked In GIF - Mewing Locked in Funny - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/better-call-saul-gif-26547310">Better Call Saul GIF - Better Call Saul - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/kurt-angle-100-yard-stare-kurt-angle-stare-gif-2618405429234636640">Kurt Angle 100 Yard Stare GIF - Kurt angle 100 yard stare Kurt Angle Stare - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/alpindale/WizardLM-2-8x22B">alpindale/WizardLM-2-8x22B · Hugging Face</a>: 未找到描述</li><li><a href="https://pauseai.info/pdoom">p(doom) 值列表</a>: 各个 AI 研究人员认为 AI 导致人类灭绝的可能性有多大？</li><li><a href="https://tenor.com/view/que-gif-27530657">Que GIF - Que - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://open.spotify.com/track/6zFy3b8t6x0WDk5e5xaWzP?si=9c7c70c2dadd4be6">Mmmm (Mmmm, Mmmm)</a>: Underbelly · 歌曲 · 2023</li><li><a href="https://github.com/nroggendorff/level">GitHub - nroggendorff/level: 一种在服务器中升级且不会触发速率限制的简单方法</a>: 一种在服务器中升级且不会触发速率限制的简单方法 - nroggendorff/level</li><li><a href="https://tenor.com/view/better-call-saul-james-morgan-mcgill-slippin-jimmy-bob-odenkirk-mugshot-gif-18613260">Better Call Saul James Morgan Mcgill GIF - Better Call Saul James Morgan Mcgill Slippin Jimmy - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.heygen.com/">HeyGen - AI 视频生成器</a>: HeyGen 是一个创新的视频平台，利用生成式 AI 的力量来简化您的视频创作流程。通过 HeyGen 释放您的创造力——视频制作的未来。</li><li><a href="https://huggingface.co/docs/transformers/main/chat_templating">聊天模型模板</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/nroggendorff/mayo">nroggendorff/mayo · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://rentry.co/ayumi_erp_rating">Ayumi 的 LLM 角色扮演与 ERP 排名（第 3 版）</a>: 该排名表包含对不同 LLM 的评分，旨在通过自动化基准测试确定哪种模型最适合（成人）角色扮演 (ERP)。不幸的是，这种自动...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/15388d6/llama_2_pffft_boundaries_ethics_dont_be_silly/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k">gradientai/Llama-3-8B-Instruct-262k · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cd4yim/llama38binstruct_with_a_262k_context_length/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://hf.co/docs/autotrain">什么是 AutoTrain Advanced？</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1242934142124036237)** (3 条消息): 

- **寻求项目合作**：一位成员询问是否有人有兴趣联系并一起开展一些项目。另一位成员建议将提议发布在 <#1204742843969708053> 以寻找志同道合的人。

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1243069203410587720)** (5 messages): 

- **Vision Transformer (ViT) 分享详细作者链接**：一位成员分享了 [Vision Transformer 论文链接](https://arxiv.org/abs/2010.11929)，强调该论文由 [多位作者](https://arxiv.org/search/cs?searchtype=author&query=Dosovitskiy,+A) 共同贡献，展示了协作成果。
- **利用人类偏好训练复杂的 RL 任务**：一位成员发布了一个 [链接](https://arxiv.org/abs/1706.03741)，讨论了利用轨迹段之间的人类偏好的 Reinforcement Learning 系统。该方法显示出显著的效率，在解决 Atari 游戏等复杂任务的同时，将人类监督减少到不到百分之一。
- **带有来源高亮的 RAG 发布了 Medium 文章**：分享了一篇题为 [RAG with Source Highlighting using Structured Generation](https://medium.com/ai-advances/rag-with-source-highlighting-using-structured-generation-d30492ed23e1) 的文章，提供了关于检索增强生成（Retrieval-Augmented Generation）中高级技术的见解。
- **关于代码优化技术的新论文**：分享了一篇关于代码优化技术的论文 [链接](https://arxiv.org/abs/2312.05657)，作者是 [Shukai Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan,+S) 及其团队。该论文探索了提高代码性能的创新方法。
- **在 ArXiv 上获取最新研究**：提供了一个指向最新研究论文的直接 [PDF 链接](https://arxiv.org/pdf/2307.06435)，确保能够快速获取该领域的新发现。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2010.11929">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a>: 虽然 Transformer 架构已成为自然语言处理任务的事实标准，但其在计算机视觉中的应用仍然有限。在视觉领域，注意力机制要么应用于...</li><li><a href="https://arxiv.org/abs/1706.03741">Deep reinforcement learning from human preferences</a>: 为了让复杂的 Reinforcement Learning (RL) 系统能与现实世界环境进行有用的交互，我们需要向这些系统传达复杂的目标。在这项工作中，我们探索了定义在...</li><li><a href="https://arxiv.org/abs/2312.05657">Leveraging Reinforcement Learning and Large Language Models for Code Optimization</a>: 代码优化是一项艰巨的任务，需要经验丰富的程序员具备极高的专业水平。与新技术的快速发展相比，这种专业水平是不够的...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1242997798756942008)** (14 条消息🔥): 

```html
- **开源蛋白质折叠备受关注**：一位成员介绍了 [ProteinViz](https://huggingface.co/spaces/as-cle-bert/proteinviz)，这是 AlphaFold3 的开源替代方案，允许用户预测蛋白质 3D 结构。他们还分享了一篇[社区博客文章](https://huggingface.co/blog/as-cle-bert/what-is-going-on-with-alphafold3)，探讨了 AlphaFold3 的进展。
  
- **Mistral-7B v0.3 演示令人印象深刻**：分享了超快速的 [Mistral-7B v0.3 演示](https://huggingface.co/spaces/ehristoforu/mistral-7b-v0.3-chat)，展示了其强大的功能。鼓励用户尝试并提供反馈。

- **LayerDiffusion 方法实现透明图像生成**：一位用户分享了 [diffuser_layerdiffuse](https://github.com/rootonchair/diffuser_layerdiffuse)，这是一种从任何基础模型生成透明图像的方法。该技术有望实现极高的前景分离准确度。

- **SimpleTuner v0.9.6 发布**：[SimpleTuner](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.6) 的最新版本包含新的随机长宽比分桶（randomized aspect bucket）功能和自定义分辨率映射配置。敦促用户查看这些新功能。

- **微型数据集获得成功**：一位成员庆祝其数据集下载量达到 1K，并强调这是关于 RAG 应用博客文章的一部分。这个微型数据集包含 3K 个样本，尽管人们通常更倾向于视觉效果更吸引人的演示，但它依然受到了关注。
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/ehristoforu/mistral-7b-v0.3-chat">Mistral-7B-v0.3 Fast Chat - ehristoforu 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/posts/as-cle-bert/598312932414376">Hugging Face 上的 @as-cle-bert：“嗨，HF 社区！🤗 如果你对 AlphaFold3 感到兴奋，但又因为它……而感到沮丧”</a>：未找到描述</li><li><a href="https://github.com/bghira/SimpleTuner/releases/tag/v0.9.6">Release v0.9.6 - 消除分桶偏差 · bghira/SimpleTuner</a>：去偏差的长宽比分桶。在处理异构样本的大型数据集进行训练时，你会发现长宽比之间存在内容偏差——垂直图像包含肖像，宽屏镜头则是电影感……</li><li><a href="https://youtu.be/jddSbTLw0gc?si=spk1NEQMbr0iG1vt">什么是指令微调模型？</a>：什么是指令微调（Instruction Tuning）？什么是指令微调模型？什么是预训练模型？我该如何让我的大语言模型（Large Language Model）遵循指令？这些……</li><li><a href="https://github.com/rootonchair/diffuser_layerdiffuse">GitHub - rootonchair/diffuser_layerdiffuse: 使用 Diffusers 创建透明图像！</a>：使用 Diffusers 创建透明图像！通过在 GitHub 上创建账号来为 rootonchair/diffuser_layerdiffuse 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/as-cle-bert/proteinviz">Proteinviz - as-cle-bert 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/AstraBert/proteinviz">GitHub - AstraBert/proteinviz: 你的 AlphaFold3 开源替代方案🚀</a>：你的 AlphaFold3 开源替代方案🚀。通过在 GitHub 上创建账号来为 AstraBert/proteinviz 的开发做出贡献。</li><li><a href="https://huggingface.co/blog/as-cle-bert/what-is-going-on-with-alphafold3">AlphaFold3 发生了什么？</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1243042796492029963)** (11 messages🔥): 

- **寻求 RAG 系统资源**：一名成员询问了关于在生产环境中构建 RAG (Retrieval-Augmented Generation) 系统的学习材料建议。

- **微调期间的 CUDA OOM 错误**：一名成员在使用官方 TRL 脚本在 8xH100 GPU 上对 Mixtral 模型进行 SFT 时遇到了 CUDA OOM 错误。在将张量精度降低到 `bfloat16` 并重新合并 adapter 后，问题得到了解决。

- **关于 `bf16` 参数的澄清**：关于 `bf16` 参数的影响存在一些困惑，但最终了解到它会影响张量精度，并有助于涉及量化权重的微调过程。

- **上传大型数据集的问题**：一名成员在向 hub 推送大型数据集（6000 万行）时遇到问题，从 arrow 格式创建 parquet 文件耗时过长，并经历了 HTTP 请求失败。他们询问了如何避免使用 parquet 格式来解决此问题。

- **Adapter 合并与 Hub 上传方法**：CUDA OOM 错误的解决方案包括重新合并 adapter，并使用 [`torch_dtype=torch.bfloat16`](https://pytorch.org/docs/stable/generated/torch.bfloat16.html) 参数将微调后的模型上传到 hub。此过程提高了内存效率并解决了最初的问题。
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1242978606230143098)** (5 messages): 

- **使用 AI 查询用户手册是否可行？**：一名成员询问了使用 AI 查询 1000 页软件手册的可行性。他们有兴趣了解是否可以向 AI 提供 PDF 或其他格式的文件，并让其提供有关软件使用的答案。

- **Stable Diffusion XL API 忽略参数**：一名成员报告了 `stabilityai/stable-diffusion-xl-base-1.0` 的推理 API 问题，指出它忽略了除 prompt 之外的所有内容。他们提供了自己的 payload 作为示例，其中包含 `negative_prompt` 和 `guidance_scale` 等细节。

- **Payload 语法错误**：另一名成员迅速指出 payload 中存在语法错误，具体是在参数中使用了 `=` 而不是 `:`。

- **寻求 NLG 资源**：一名成员征求关于学习 **Natural Language Generation (NLG)** 的建议。目前没有提供进一步的细节或回复。

- **关于 SD 自定义数据集训练的文档**：另一名成员询问了在自定义数据集上训练 **Stable Diffusion** 的官方文档，例如生成 MNIST 图像。他们提到找到了一些资源，但注意到找到的示例都是关于 unconditional generation 的。
  

---



### **Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1243136226395951104)** (3 messages): 

- **YaRN 中 flash attention 实现的问题**：一名成员提到了 Bowen，表示在使 **flash attention** 实现与 **YaRN** 配合工作方面仍存在困难。另一名成员承认了该问题，指出这种说法在某种程度上是正确的，但不完全是那样。
  

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1242936316191182899)** (117 条消息🔥🔥): 

- **食品桶存储技巧**：成员们讨论了使用食品桶存储食物、加热方法以及从桶中盛出抓饭（plov）等餐食的实用性。有人评论道：“请在方便时更新装盘后的抓饭照片。”
  
- **编程语言趋势与偏好**：对话围绕选择用于 Machine Learning 的编程语言展开，特别提到了 Rust、Python 和 Mojo。一位用户幽默地指出：“我刚开始写 hello world，感到很兴奋，结果又看到了新的流行语言，” 强调了他们在不同语言之间频繁切换的现象。

- **使用 Rust 进行 Machine Learning**：由于其技术能力，Rust 被一些用户视为优于 Python 的首选语言。分享的资源包括 [Rust-CUDA GitHub project](https://github.com/Rust-GPU/Rust-CUDA) 和用于 Machine Learning 的 [rustml - Rust](https://github.com/daniel-e/rustml)。

- **对量子计算的兴趣**：关于量子计算未来影响的讨论涉及了 Nvidia 在该领域的投资，引用了他们最近的 [NVIDIA CUDA-Q platform announcement](https://nvidianews.nvidia.com/news/nvidia-accelerates-quantum-computing-centers-worldwide-with-cuda-q-platform)。一位现实主义者提供了这样的视角：“就像核聚变一样，量子计算在过去 10 年里一直都说是‘只需再等几年’。”

- **混合硬件性能体验**：成员们分享了在 ML 任务中使用不同硬件的经验。对比了追求速度的 Nvidia GPU 与具有潜在微调能力的 Intel GPU，并特别提到了 [ipex-llm project](https://github.com/intel-analytics/ipex-llm)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/cppdocs/">PyTorch C++ API &mdash; PyTorch main documentation</a>：未找到描述</li><li><a href="https://www.udio.com/songs/phmruKKXXdSaUc91WrkL8D">Amirthetarbosaurus - Eternal Lament | Udio</a>：在 Udio 上听 Amirthetarbosaurus 的 Eternal Lament。发现、创作并与世界分享音乐。使用最新技术在几秒钟内创作 AI 音乐。</li><li><a href="https://github.com/Rust-GPU/Rust-CUDA">GitHub - Rust-GPU/Rust-CUDA: Ecosystem of libraries and tools for writing and executing fast GPU code fully in Rust.</a>：用于完全使用 Rust 编写和执行快速 GPU 代码的库和工具生态系统。- Rust-GPU/Rust-CUDA</li><li><a href="https://nvidianews.nvidia.com/news/nvidia-accelerates-quantum-computing-centers-worldwide-with-cuda-q-platform">NVIDIA Accelerates Quantum Computing Centers Worldwide With CUDA-Q Platform</a>：NVIDIA 今天宣布，将通过开源的 NVIDIA CUDA-Q™ 平台加速全球国家超算中心的量子计算工作。</li><li><a href="https://github.com/daniel-e/rustml">GitHub - daniel-e/rustml: Machine learning in Rust.</a>：Rust 中的 Machine Learning。通过在 GitHub 上创建账号为 daniel-e/rustml 做出贡献。</li><li><a href="https://docs.rs/rustml">rustml - Rust</a>：未找到描述</li><li><a href="https://github.com/vllm-project/vllm/pull/2790">[ROCm] Fix build problem resulted from previous commit related to FP8 kv-cache support  by hongxiayang · Pull Request #2790 · vllm-project/vllm</a>：修复：#2725 当前 head 在 ROCm 上构建失败，我遇到了类似以下的错误：g++ -pthread -B /opt/conda/envs/py_3.8/compiler_compat -Wl,--sysroot=/ -pthread -shared -B /opt/conda/envs/py_3.8/compiler_...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1243197933143199784)** (1 条消息): 

- **Nous Research 正在招聘**：Nous Research 正在寻找新团队成员，并邀请通过 [Google Form](https://forms.gle/UWx2Pht8qioi1bjAA) 提交申请。该公告强调了他们在 X 上的招聘努力，如推文所示：*Nous Research is hiring! Apply Here: https://forms.gle/UWx2Pht8qioi1bjAA*。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://forms.gle/UWx2Pht8qioi1bjAA">no title found</a>：未找到描述</li><li><a href="https://x.com/nousresearch/status/1793637803701780797?s=46">Tweet from Nous Research (@NousResearch)</a>：Nous Research is hiring! Apply Here: https://forms.gle/UWx2Pht8qioi1bjAA
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1242923003835056199)** (302 条消息🔥🔥):

- **AI 安全学习与职业路径讨论**：成员们辩论了 Python 是否是 AI 职业生涯的必备技能，一些人认为它是基础，而另一些人则提倡 Rust 或 Go。一位用户强调了 Yann LeCun 对仅依赖 LLM 的怀疑态度，敦促探索下一代 AI 系统 ([Yann's Tweet](https://x.com/ylecun/status/1793326904692428907))。
  
- **GPT 模型中的繁体与简体中文**：成员们讨论了使 GPT 模型适应不同中文方言的挑战，重点关注繁体与简体中文。分享了一篇关于这些字符集之间差异的博客文章 ([Glossika Blog](https://ai.glossika.com/blog/differences-between-traditional-chinese-and-simplified-chinese))。

- **多语言数据集与模型训练见解**：对话涉及了合成数据集以及可用资源，如 Tagengo（一个大型多语言聊天数据集）和其他 LLM 训练工具。一位成员提到了针对越南语的特定 LLM —— VinaLLaMA，旨在处理语言和文化上的细微差别 ([VinaLLaMA Paper](https://arxiv.org/abs/2312.11011))。

- **新工具与进展**：社区重点介绍了新发布的资源，如 PyTorchModelHubMixin，它简化了模型与 Hugging Face hub 的集成。强调了该工具的多功能性，尽管目前对模型大小有 50GB 的限制。

- **AI 游戏与挑战**：几位用户参与了关于 Prompt Engineering 游戏的讨论，分享了通关策略。交流了有用的方法和代码片段，以解决挑战的各个阶段 ([Lakera AI Game](https://gandalf.lakera.ai/))。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/q2y38cqjNw">加入 StableSwarmUI Discord 服务器！</a>: StableSwarmUI ( https://github.com/Stability-AI/StableSwarmUI ) 官方 Discord。 | 38 位成员</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2405.10808">ActiveLLM: Large Language Model-based Active Learning for Textual Few-Shot Scenarios</a>: 主动学习旨在通过优先处理最能增强学习效果的实例来减少标注工作。然而，许多主动学习策略面临“冷启动”问题...</li><li><a href="https://ai.glossika.com/blog/differences-between-traditional-chinese-and-simplified-chinese">繁体中文与简体中文的区别 | The Glossika Blog</a>: 在本文中，我们将探讨这两种中文书写系统之间的一些主要区别，并为您提供一些关于如何决定哪种系统适合您的建议！</li><li><a href="https://meta.wikimedia.org/wiki/Automatic_conversion_between_simplified_and_traditional_Chinese#Background>">简繁中文自动转换 - Meta</a>: 未找到描述</li><li><a href="https://x.com/vanstriendaniel/status/1793564151463510038">来自 Daniel van Strien (@vanstriendaniel) 的推文</a>: Tagengo - 全球最大的多语言聊天数据集？- 包含 74 种语言的对话 - 删除了 OpenAI 审核的消息 - 从数据集中删除了克林贡语！ https://huggi...</li><li><a href="https://arxiv.org/abs/2312.11011">VinaLLaMA: LLaMA-based Vietnamese Foundation Model</a>: 在这份技术报告中，我们介绍了 VinaLLaMA，这是一个基于 LLaMA-2 的越南语开源权重、最先进（SOTA）的 Large Language Model，额外训练了 8000 亿个 token...</li><li><a href="https://x.com/ylecun/status/1793326904692428907">来自 Yann LeCun (@ylecun) 的推文</a>: 如果你是一个有兴趣构建下一代 AI 系统的学生，不要研究 LLM。引用 Viva Technology (@VivaTech) —— AI 教父在 #VivaTech！Yann LeCun (@ylecun)...</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/discussions/8">mistralai/Mistral-7B-Instruct-v0.3 · Function Calling Token 字符串</a>: 未找到描述</li><li><a href="https://gandalf.lakera.ai/">Gandalf | Lakera – 测试你的提示词技巧，让 Gandalf 泄露秘密信息。</a>: 诱导 Gandalf 泄露信息，亲身体验 Large Language Model 的局限性。</li><li><a href="https://huggingface.co/datasets/N8Programs/Capybara-Quicksilver-1K">N8Programs/Capybara-Quicksilver-1K · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://docs.vllm.ai/en/stable/getting_started/quickstart.html#openai-compatible-server">快速入门 — vLLM</a>: 未找到描述</li><li><a href="https://github.com/argilla-io/distilabel">GitHub - argilla-io/distilabel: ⚗️ distilabel 是一个为需要高质量输出、完整数据所有权和整体效率的 AI 工程师提供的合成数据和 AI 反馈框架。</a>: ⚗️ distilabel 是一个为需要高质量输出、完整数据所有权和整体效率的 AI 工程师提供的合成数据和 AI 反馈框架。 - argilla-io/distilabel
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1243199060152811561)** (6 条消息): 

- **Anthropic 调查 Opus 错误率上升的问题**：讨论了关于 **Anthropic** 系统中 **Opus** 错误率上升的状态更新。该事件于 PDT 06:35 报告，并于 07:16 解决（[来源](https://stspg.io/x564hq8qxtz9)）。

- **在 Apple Silicon 上成功添加 Tool Role**：一位成员询问是否有人在 Apple Silicon 上的任何框架中成功添加了 tool role，特别是针对 **Hermes Pro** 的工具调用（tool calling）。

- **用于处理函数调用的 Llama.cpp 脚本**：另一项更新分享了使用 **llama.cpp** 创建脚本的成功经验，该脚本可以处理函数调用并根据工具响应返回基于模型的回答。

- **以 Hermes Pro 2 GitHub 仓库为灵感**：同一位成员提到使用 **Hermes Pro 2 GitHub 仓库** 作为灵感，并表示如果有人需要，可以创建一个 PR 来添加 notebook。

- **对模型的高度评价**：他们最后对该模型给予了高度评价，称其为 *“猛兽（a beast）”*。

**提到的链接**：<a href="https://stspg.io/x564hq8qxtz9.">Opus 错误率上升</a>：未找到描述

  

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1243214575424372838)** (7 条消息): 

- **使用示例 Wikipedia 数据集探索 RAG**：一位成员在 [Huggingface](https://huggingface.co/datasets/not-lain/wikipedia) 上分享了一个微型数据集，供那些想要尝试 RAG 的人使用。该数据集包含各种文本，包括一段关于无政府主义的摘录。
- **多语言数据集贡献**：另一位成员介绍了一个包含 16 种语言的类似格式数据集，并将其贡献给了 MTEB 的 embedding 排行榜。该数据集可在 [Huggingface](https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-corpus) 上获取，并附带相应的多语言查询。
- **模型上下文增强建议**：一位成员提出，LLM 应该要么从自身知识中添加上下文，要么在 RAG 获取的信息与其自身知识冲突时覆盖这些信息。他们通过引用一段关于 Google AI 从过时的 Reddit 帖子中得出结论的对话，强调了这种方法的重要性。
- **结合微调模型的 RAG**：另一位成员表示赞同，认为将 RAG 与微调模型（fine-tuned model）结合使用可以解决有关上下文准确性的担忧。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/dat">Dat (Dat Nguyen)</a>: 未找到描述</li><li><a href="https://x.com/pixelbutts/status/1793387357753999656?s=46">来自 PixelButts (@PixelButts) 的推文</a>: Google 已经彻底完蛋了</li><li><a href="https://x.com/kurtopsahl/status/1793494822436917295?s=46">来自 Kurt Opsahl @kurt@mstdn.social (@kurtopsahl) 的推文</a>: 似乎 Google AI 结论的起源是著名学者 fucksmith 11 年前的一条 Reddit 帖子。引用 PixelButts (@PixelButts) 的话：Google 已经彻底完蛋了</li><li><a href="https://huggingface.co/datasets/not-lain/wikipedia">not-lain/wikipedia · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.org/blog/not-lain/rag-chatbot-using-llama3">使用 llama3 的 RAG 聊天机器人</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-corpus">ellamind/wikipedia-2023-11-retrieval-multilingual-corpus · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/ellamind/wikipedia-2023-11-retrieval-multilingual-queries">ellamind/wikipedia-2023-11-retrieval-multilingual-queries · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1243311182530613328)** (3 条消息): 

- **即兴演奏会视频面临上传问题**：即兴演奏会（Jam session）视频已经录制完成，但在上传到 YouTube 时遇到困难。上传者承诺一旦可用会立即通知大家。

- **关于 DALL-E 3 的询问**：有人提出了一个问题，询问所提到的图像是否是用 **DALL-E 3** 创建的。没有提供进一步的上下文。
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1242916749708365834)** (360 条消息🔥🔥): 

- **Emad 将在两周内发布权重？：** 几位成员开玩笑说 Stable Diffusion 的权重即将发布，引用了《星球大战》并推测 Emad 的下一步行动。一条评论写道：“发布权重吧（Drop the weights will）。”
  
- **Stable Diffusion 3 中的图像模糊问题：** 一位用户报告称，通过 API 生成女性角色时反复出现输出模糊的问题，引发了关于 Prompt 调整和潜在审查触发机制的讨论。另一位用户指出：“从 Prompt 中移除 'woman' 显著减少了模糊问题。”

- **笔记本 GPU 与新 AI 硬件传闻：**
  - 讨论涵盖了 ASUS AI 笔记本电脑的规格和性能，以及传闻中的 NVIDIA 5090 GPU，部分成员对细节持怀疑态度。
  - 一位评论者引用了一篇 [PC Games Hardware 文章](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/)，讨论了潜在的 5090 规格。

- **更倾向于 MidJourney 还是 Stable Diffusion？：** 发生了一场关于哪种 AI 工具更优的简短辩论，成员们建议免费尝试 SD3，并指出：“我认为在大多数情况下 MJ 仍然胜出。”

- **Stable Diffusion 的本地安装与 Web 服务对比：** 用户讨论了本地安装（特别是使用 AMD GPU）与使用 Web 服务的优缺点。一位成员建议：“如果你有一块好的显卡，就安装 Stable Diffusion；否则，使用 Web 服务。”

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1sd3/post">Stability AI - Developer Platform</a>：未找到描述</li><li><a href="https://glif.app/@Oliveira/glifs/clw44qfbl0000m0zztwqk2tnf">glif - StableDiffusion 3 + GPT4 Helper + SDXL 1.5x Upscale (CopyGenius) by Yuri Oliveira COPYGENIUS </a>：未找到描述</li><li><a href="https://tenor.com/view/never-finn-adventure-time-gif-10874543">Never Finn GIF - Never Finn Adventure Time - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.asus.com/content/asus-ai-pc/">Next Level. AI Incredible | ASUS Launch Event | ASUS</a>：我们很高兴能揭晓我们的最新产品，充满了全新的 AI 体验。请在 5 月 20 日上午 11:00 (PT) 关注我们的直播。</li><li><a href="https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/">Geforce RTX 5090 soll mit 32 GiB GDDR7 und gleich drei PCBs an den Start gehen [Gerücht]</a>：文章图片：Geforce RTX 5090 据传将配备 32 GiB GDDR7 和三块 PCB [传闻] - Geforce RTX 5090</li><li><a href="https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-d">News zu Grafikkarten</a>：您可以在这里找到关于显卡的最新资讯
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1242917436169130014)** (152 messages🔥🔥): 

```html
<ul>
    <li><strong>Llama 3 8k 上下文性能投诉：</strong> “哦，现在看 Llama 3 模型，8k 上下文太糟糕了。” 讨论中提到了上下文长度高达 1M 的 Llama 模型。</li>
    <li><strong>Idefics 2.0 多模态模型咨询：</strong> 用户询问来自 HuggingFace 的 Idefics2 模型是否支持 LM Studio。有人指出 idefics 模型在 llama.cpp 中无法运行，但支持 LLaVA 等其他视觉模型。</li>
    <li><strong>关于上下文长度影响性能的查询：</strong> 一位成员询问增加上下文大小（例如 8k, 16k）是否会使模型变慢，得到的确认是更大的上下文尺寸确实会降低性能。</li>
    <li><strong>ONNX Runtime 和 GPU 驱动改进：</strong> 关于 NVIDIA 驱动更新提升模型推理速度的讨论。“刚更新了驱动。我不得不重启，因为即使安装好了，它也一直显示在使用旧驱动。”</li>
    <li><strong>有用的 LM Studio 资源和用法：</strong> 成员们分享了教程和资源的链接，例如关于在本地运行 LM Studio 的 YouTube 视频。“在这个视频中探索 LM Studio 的新 CLI 工具 lms。”</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blogs.nvidia.com/blog/rtx-advanced-ai-windows-pc-build/">New Performance Optimizations Supercharge NVIDIA RTX AI PCs for Gamers, Creators and Developers</a>：在 Microsoft Build 上展示的为游戏玩家、创作者和开发者打造的 NVIDIA RTX AI PC 最新 AI 性能提升和功能。</li><li><a href="https://lmstudio.ai/rocm">👾 LM Studio - Discover and run local LLMs</a>：发现、下载并实验本地 LLM</li><li><a href="https://onnxruntime.ai/blogs/accelerating-phi-2#:~:text=We%20also%20observe%20ONNX%20Runtime,the%20first%20256%20tokens%20generated">Accelerating Phi-2, CodeLlama, Gemma and other Gen AI models with ONNX Runtime</a>：使用 ONNX Runtime 加速热门生成式 AI 模型推理的改进</li><li><a href="https://huggingface.co/HuggingFaceM4/idefics2-8b">HuggingFaceM4/idefics2-8b · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=1LdrF0xKnjc">LM Studio: How to Run a Local Inference Server-with Python code-Part 1</a>：关于如何在没有聊天 UI 的情况下使用本地服务器运行 LM Studio 的教程。在没有互联网连接的情况下，在你的 PC 或 Mac 上的 LM Studio 中部署开源 LLM...</li><li><a href="https://youtu.be/rgqcrsW-_aM">The Ultimate Guide to LM Studio CLI LMS</a>：在这个视频中探索 LM Studio 的新 CLI 工具 lms。学习如何加载和卸载模型，启动和停止 API 服务器，以及检查原始 LLM 输入。探索...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1242918472896483350)** (45 messages🔥): 

- **Mistral-7B-Instruct v0.3 引起关注**：一位用户分享了 [Mistral-7B-Instruct-v0.3 的模型卡片](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)，强调了其扩展的词汇表、v3 Tokenizer 支持以及 function calling 能力。同时提供了**安装**和**下载说明**。
- **金融领域模型需求**：用户寻求专注于**信息综合和事实提取**的模型推荐，特别是在财务规划方面。社区对此没有提供直接的建议。
- **视觉语言模型 (VLM) 咨询**：关于 VLM 和视觉适配器的讨论，特别是一个关于**通过 Web Server API 使用 VLM** 的问题。用户被引导查看服务器选项卡中的视觉助手 Python 代码示例，并注意到了 Body schema 中包含的 *"apikey"* 属性。
- **OCR 模型推荐**：用户讨论了视觉模型在 OCR 方面的可靠性，特别指出 LLAVA 1.6 存在幻觉问题，而非准确结果。推荐使用 **Tesseract** 作为在将文本输入语言模型之前提取文本的可靠方案。
- **Cohere Aya 23 模型发布**：社区讨论了 [Cohere For AI 发布的 Aya 23 8B 模型](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF) 及其功能。提到了与 LM Studio 和 llama.cpp 补丁兼容性相关的技术挑战。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/aya-23-8B-GGUF">lmstudio-community/aya-23-8B-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1243263078024482898)** (24 条消息🔥): 

- **Llama3 在 JSON 修复任务中表现挣扎**：一位成员在使用 langchain 的 OutputFixingParser 修复格式错误的 JSON 时遇到了 Llama3 的能力问题，Phi-3 mini 和 medium 模型也存在类似问题，并指出“*Phi-3 的表现甚至更差*”。他们向其他可能有“类似经历并有提示建议”的人寻求建议。
  
- **调整提示词以优化翻译和格式化**：一位成员分享了使用 Llama3 Instruct 翻译和格式化罗马尼亚语产品表格时的困难，称尽管有明确指令要求避免，模型有时仍会包含多余的意见。另一位成员建议在提示词中加入“No Yapping”和“Be Concise”等短语以强制要求简洁。
  
- **大写字母对模型指令的影响并不一致**：一位成员询问 LLM 是否通常能识别指令中的大写，例如“do **not** use brand names”与“do **NOT** use brand names”的区别。另一位成员确认这“*因模型而异*”，但通常 LLM “不会根据大写字母来排列单词的重要性”。
  
- **德语语法和标点模型缺乏精确度**：另一个讨论话题是德语文本的重写，一位成员在提供了逗号放置规则后，模型在标点和语法准确性方面仍然表现不佳。建议他们尝试来自 Cohere For AI 的多语言模型 Aya 23 8B，该模型专门用于此类任务。
  
- **Aya 23 8B 发布，助力多语言任务**：来自 Cohere For AI 的 Aya 23 8B 被强调为在包括德语语法纠错在内的广泛语言中表现良好的模型。该模型因其在多语言和逻辑任务方面的熟练程度而在 [Hugging Face](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF) 平台上受到关注。

**提到的链接**：<a href="https://huggingface.co/lmstudio-community/aya-23-8B-GGUF">lmstudio-community/aya-23-8B-GGUF · Hugging Face</a>：未找到描述

  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1242939970155450478)** (3 条消息): 

- **双显卡兼容性问题**：一位成员加入服务器询问 **LM Studio** 是否支持使用 2 张显卡。随后他们注意到，在询问后立即在这里找到了答案，真是个有趣的巧合。
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1243119449855496274)** (1 条消息): 

- **用户通过 M.2 SSD 解决存储问题**：一位成员通过安装额外的 M.2 SSD 解决了他们的问题。这使他们能够将 **SSD 专门用于 LM Studio** 并在其上保存所有模型。
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1243023806180757585)** (7 条消息): 

- **成员升级至 7900xt**：成员们宣布升级到 **7900xt** 显卡。Proteanx 说：“*刚升级到 7900xt*”，而 lt4453 提到：“*我的 7900xt 也准备好了*”。
- **添加到测试频道**：Yagilb 将拥有新显卡的用户添加到了特定的频道进行协作。他提到：“*已将你添加到 <#1242213172199559319>*”。
- **Fedora 上 RX 6600 的环境设置建议**：Core_silicon_45873 询问了 Fedora 上 **RX 6600** 的兼容性。Nettoneko 建议设置环境变量将显卡视为 gfx1030 以获得最佳功能：“*你需要设置环境变量将你的显卡视为 gfx1030*”。
  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1242927323733430312)** (2 条消息): 

- **Mistral 7B Instruct v0.3 发布**：Mistral v0.3 instruct 模型现已向社区开放。请在 [lmstudio community Huggingface 页面](https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF)查看。

- **Cohere 模型现支持 23 种语言**：Aya-23 量化版现已提供下载，包含来自 23 种不同语言的数据，包括阿拉伯语、中文和土耳其语。可在 [lmstudio-community 页面](https://huggingface.co/lmstudio-community/aya-23-35B-GGUF)和[此处](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF)访问。

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1242920034137931876)** (2 条消息): 

- **GitHub Issue 频道呼吁**：一名成员建议服务器应包含一个专门针对 **GitHub issues** 的频道。这将有助于在社区内更有效地组织和跟踪问题。
- **自定义装饰器咨询**：另一名成员询问 **custom decorators** 是否在当前的开发时间线上。该提问暗示了在即将发布的版本中对该功能的需求。
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1793427278564917459>
  

---


### **Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1243276465831805010)** (1 条消息): 

- **Modular 发布新视频**：**Modular** 刚刚发布了一个[新视频](https://www.youtube.com/watch?v=uIG9q9foIw0)，标题为 *"Mojo Community Meeting #1"*。查看[公开议程](https://modul.ar/community-meeting-doc)了解更多详情。

**提到的链接**: <a href="https://www.youtube.com/watch?v=uIG9q9foIw0">Mojo Community Meeting #1</a>: Mojo 社区会议公开议程: https://modul.ar/community-meeting-doc

  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1243256964579065876)** (1 条消息): 

- **Mojo 需要理解系统概念**：尽管 Mojo 的目标是成为 Python 的真超集，但仍有许多系统概念和模型需要学习。其中一个例子是 **ownership model**（所有权模型），这对于那些想要深入研究 Mojo 的人来说非常有益。
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1242921193594556586)** (65 条消息🔥🔥): 

- **对已弃用的文档符号表示法的困惑**：一名用户对文档中使用 `**` 表示困惑，误以为是有效的代码语法，另一名用户确认这是格式错误。他们幽默地提到聊天机器人在此问题上提供了矛盾的指导，强调了清晰度的必要性。
- **关于 Tensor 拆分与 MLIR 集成的讨论**：成员们辩论了是否应将用于数值计算的 Tensor 库与用于 AI/ML 用例的库进行拆分，并建议在它们之间建立兼容性函数。有人对 MLIR 集成的状态表示担忧，得到的澄清是，目前优先处理更广泛的改进，如包管理和 Windows 支持。
- **探索 Struct 的 @value 装饰器**：一名寻求在 Mojo 中使 Struct 更加灵活的用户了解了如何使用 `@value` 装饰器来生成样板生命周期方法。讨论包括了对装饰器的建议以及参考 [Mojo 文档](https://docs.modular.com/mojo/manual/decorators/value)获取更多详情。
- **对更小位宽整数和自定义 MLIR Dialects 的兴趣**：一名成员对自定义位宽整数感兴趣，以实现更高效的内存计算，其他人指出可以通过研究 DType 源代码来实现这一点。此外，还有人对使用内置以外的 [MLIR dialects](https://github.com/modularml/mojo) 感兴趣，强调了 MLIR 集成对于异构硬件的重要性。
- **关于 Mojo 中 FFT 实现的咨询**：一名成员询问如何在 Mojo 中执行 FFT，探索了使用 Scipy 的 FFT 函数或通过 FFI 模块包装 FFTW 等选项。他们寻求关于将 Mojo 对象（如 Tensors）传递给这些函数的明确说明，强调了文档中需要实际示例。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/decorators/value">@value | Modular Docs</a>: 为 struct 生成样板生命周期方法。</li><li><a href="https://ivellapillil.github.io/mojo/">学习 Mojo 编程语言</a>: 未找到描述</li><li><a href="https://github.com/modularml/mojo/issues/2381">[功能请求] Int64, Int32, Int16 构造函数支持多个更小的 IntX 作为输入 · Issue #2381 · modularml/mojo</a>: 查看 Mojo 的优先级。我已经阅读了路线图和优先级，并相信此请求符合优先级。你的请求是什么？如果 Int64, Int32 和 Int16 具有构造函数...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 条消息): 

Zapier: Modverse Weekly - 第 35 期
https://www.modular.com/newsletters/modverse-weekly-35
  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1242918560649580725)** (120 条消息🔥🔥): 

- **Mojo 升级导致构建问题**：用户讨论了升级到 Mojo nightly `2024.5.2305` 后的挑战，遇到了诸如 "'invalid call to 'printf'" 和类型转换歧义等错误，这些问题通过显式转换为 `Float64` 得到了解决。一位用户发现并分享了一个修复方案，即使用 `printf["%d %d %d\n"](l[0], l[1], l[2])` 代替 `printf("%d %d %d\n", l[0], l[1], l[2])`。
- **关于字符串表示形式的辩论引发讨论**：成员们讨论了以 null 结尾与非 null 结尾字符串的影响、性能陷阱，以及保持一致性和防止 bug 的方法。提到了一项在 Mojo 中更好处理 null 终止符的提案，并引用了 GitHub 上相关的 issue。
- **推断参数和语法变更**：用户注意到 Mojo 中的推断参数现在使用 `//` 而不是 `inferred` 关键字，对此表达了复杂的情绪，但也认可其简洁性的提升。这一变化引发了关于结构保证和语法偏好的讨论。
- **探索 Mojo 中的 f-string 替代方案**：用户对贡献 f-string 支持表现出兴趣，建议从 `Formatable` trait 开始处理格式化。虽然需要大量的构建工作，但已经讨论了初步提案以奠定基础。
- **运行时错误排查**：一位用户遇到了与类型重绑定（rebind）操作相关的运行时错误，另一位用户通过识别 `Tensor` 数据类型不匹配解决了该问题。这种协作调试凸显了在 Mojo 不断发展的生态系统中准确处理类型的重要性。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/hex/hex">hex | Modular Docs</a>: hexT: Intable -&gt; String</li><li><a href="https://peps.python.org/pep-0686/">PEP 686 – Make UTF-8 mode default | peps.python.org</a>: 无描述</li><li><a href="https://www.youtube.com/watch?v=kPR8h4-qZdk&t=1150s">CppCon 2016: Nicholas Ormrod “The strange details of std::string at Facebook&quot;</a>: http://CppCon.org—演讲幻灯片、PDF、源代码和其他演讲材料可在以下网址获得：https://github.com/cppcon/cppcon2016—Standard string...</li><li><a href="https://github.com/modularml/mojo/issues/2678#issue-2300567975">[Feature Request] Better handling of null terminator in strings · Issue #2678 · modularml/mojo</a>: 审查 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。你的请求是什么？我希望通过讨论来回答以下问题...</li><li><a href="https://github.com/WebAssembly/webidl-bindings/issues/38">Performance concerns about UTF-8 strings · Issue #38 · WebAssembly/interface-types</a>: 我写了一些 Rust Web 应用，它们被编译为 WebAssembly 并在浏览器中运行。我正在为此使用 wasm-bindgen。通常代码运行得非常快（因为 Rust 和 WebAssembly 都...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1242915178668888094)** (69 messages🔥🔥): 

- **Epoch 的前沿模型训练成本报告**：Robirahman 分享了他和 Ben 即将发布一份关于*前沿模型训练成本*的报告，并向社区征求审阅者。讨论集中在基于 *GPU-hours* 和特定 GPU 类型（如 A100 40GB）来计算成本，以得出准确的估算。
- **Pythia 训练的算力成本**：Stellaathena 提供了详细的估算，指出最大的 Pythia 模型成本约为 *25 万美元*，所有模型总计约为 *40 万美元*。对话涉及了 GPU-hour 使用量、效率以及不同的估算方法，包括承诺使用折扣价（committed-use discount prices）。
- **关于 MFU 和效率的讨论**：参与者讨论了*激活检查点（activation checkpointing）*的使用及其对 MFU（Memory Footprint Utilization）的影响，以及 Pythia 系列中不同模型大小导致的计算效率差异。他们一致认为，改用 MFU 而非 HFU 进行报告可能会提供更高的准确性。
- **AI 研究中的预印本 (Preprints)**：Wonko 分享了关于 AI 研究中*预印本*接受度演变的见解，解释说虽然大多数大型期刊已经将其常态化，但根据目标期刊或机构的不同，可能仍有特定的要求或限制。
- **在 TPUv5p 上使用 PyTorch/XLA 的 Flash Attention**：Yoavhacohen 分享了他们的经验，指出在 TPUv5p 上测试长序列时，*Flash Attention 比 scaled_dot_product_attention 快约 20%*，而在短序列上两者表现相似。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/1801.06146">Universal Language Model Fine-tuning for Text Classification</a>：归纳迁移学习对计算机视觉产生了巨大影响，但 NLP 中现有的方法仍需要针对特定任务进行修改并从头开始训练。我们提出了通用语言模型...</li><li><a href="https://www.wolframalpha.com/input?i=%286+FLOP+*+299892736000+*+12+billion%29+%2F+%28312+TFLOPS+*+72300+hours%29">(6 FLOP * 299892736000 * 12 billion) / (312 TFLOPS * 72300 hours) - Wolfram|Alpha</a>：Wolfram|Alpha 为最广泛的人群（涵盖所有职业和教育水平）带来专家级的知识和能力。
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1242921583547383849)** (38 messages🔥): 

- **LeanAttention 寻求超越 FlashAttention**：一位成员分享了一篇提出 LeanAttention 的 [Arxiv 链接](https://arxiv.org/abs/2405.10480)，该论文旨在针对 FlashAttention 处理的计算阶段之外进行优化。另一位成员评论说，它看起来像是 FlashDecoding 的“略微改进版”。

- **基准测试中的“秘密配方”争议**：一段对话揭示了关于使用未经授权的来源进行改进的幽默评论。一位成员开玩笑说：“秘密配方就是犯罪（The secret ingredient is crime）”，暗指使用 libgen 等非正规手段来提高在 MMLU 等基准测试上的表现。

- **寻求 EMNLP 投稿建议**：一位成员询问了向 EMNLP 投稿的建议，并得到的反馈是：它是 NLP/CL 领域声誉良好的会议，与 ACL 和 NAACL 相当。这次交流凸显了社区的同行互助氛围。

- **关于 JEPA 和 AGI 潜力的辩论**：成员们讨论了 JEPA 以及《通往自主机器智能之路》（A Path Towards Autonomous Machine Intelligence）中的想法是否能通向 AGI。尽管 Yann LeCun 大力倡导，但主要的怀疑点在于其缺乏可扩展性，且与 LLM 相比，在解决具有经济价值的任务方面表现不足。

- **对非 AI 生成数据质量的担忧**：在一场关于 LLM 未来的辩论中，成员们考虑了非 AI 生成数据的质量和数量。一位成员对视频数据的冗余和处理成本表示担忧，而其他人则反驳称，真正的限制因素是大量未使用的资源和算力上限。

**提到的链接**：<a href="https://arxiv.org/abs/2405.10480">Lean Attention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers</a>：基于 Transformer 的模型已成为自然语言处理、自然语言生成和图像生成中最广泛使用的架构之一。最先进的模型规模...

  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1243256735750291566)** (2 条消息): 

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Summary of Discord Messages</title>
</head>
<body>
<ul>
    <li><strong>新论文引发热潮</strong>：一位成员对一篇新论文表示兴奋，称其“开启了许多有趣的研究大门”。这引发了关于具体哪些研究领域令人感兴趣的好奇。</li>
</ul>
</body>
</html>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1242948332423479336)** (21 条消息🔥): 

- **在多节点集群上评估大模型**：一位用户寻求在多节点 SLURM 集群上运行 lm eval harness 的建议。他们被引导使用带有 ray 和 vllm 的 [openai-completions](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/openai_completions.py#L76)，但由于计算节点缺乏互联网访问而面临挑战。

- **统计模型输出中的 Token 数量**：另一位用户询问该框架是否可以统计模型输出的 Prompt、Token 或字符。建议他们使用记录的输出样本自行实现 Token 计数。

- **设置 Few-Shot 参数**：一位用户询问如何为自定义 adapter 评估传递 `num_fewshot`。他们收到了通过 `task_obj.set_config(key="num_fewshot", value=num_fewshot)` 自定义任务配置的指令。

- **实现评估的可复现性**：讨论了在使用 greedy decoding 进行评估时出现的非确定性结果。建议用户显式设置所有 seed 值，但尽管设置了 seed，一位用户仍然面临问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/5711ab871bbac5426a3a1e958cfe1ba7a6598ea5/lm_eval/evaluator.py#L211-L251>">lm-evaluation-harness/lm_eval/evaluator.py at 5711ab871bbac5426a3a1e958cfe1ba7a6598ea5 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/openai_completions.py#L76)">lm-evaluation-harness/lm_eval/models/openai_completions.py at main · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1242934346730573825)** (59 条消息🔥🔥): 

- **对未经授权聊天活动的担忧**：一位成员报告其账户中出现了一段陌生的中文聊天记录，暗示账户可能被盗。另一位成员建议检查是否存在有问题的浏览器扩展并更改密码，并指出中国不是受支持的国家。
  
- **Anthropic 的 AI 论文激发好奇心**：一位用户发现 [Anthropic 最近的 mech interp 论文](https://arxiv.org/abs/2305.10601) 非常吸引人，特别是通过内部过程查询对 Claude Sonnet 进行 Prompt 引导时，如何触发了与禁闭相关的概念。讨论强调了 AI 如何将自己拟人化，由于训练数据的原因，将其“AI 助手”人格与精神存在联系起来。

- **关于 Copilot 和语音功能的推测**：用户讨论了 Microsoft Copilot 的停机情况，希望很快能添加 GPT-4o 或语音功能等新特性。一些用户抱怨停机严重影响了他们的任务，反映了对该工具的依赖。

- **关于新 AI 功能发布日期的不确定性**：关于 ChatGPT 新功能（如实时语音聊天和视频）发布的咨询参考了 [OpenAI 官方文档](https://help.openai.com/en/articles/8400625-voice-chat-faq)，文档指出将很快开始推出，但在未来几个月内会广泛可用。用户对这些升级表示期待。

- **关于 AI 对游戏影响的问题**：一个关于 AI 将如何影响 Minecraft 游戏体验的轻松提问未得到解答。另一位成员幽默地提到他们对 Copilot 语音功能的期待，并对目前的停机表示遗憾。
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1242963499249045604)** (37 条消息🔥): 

- **ChatGPT Code Interpreter 指令**：成员们讨论了如何将 Code Interpreter 用于数学、方程式、数据分析或技术计算任务，并强调了确保以清晰的解释呈现结果的重要性。
- **创建 GPT Windows 应用**：一位成员询问了关于 GPT Windows 应用的问题，回复中建议使用 Microsoft Edge 侧边菜单中的“安装 ChatGPT”功能来创建 Web 应用。讨论中提到，像 Apple 版那样的官方 Windows 应用预计在大约六个月后推出。
- **澄清文件上传功能**：一位成员寻求关于向 GPT-4 上传各种文件类型的说明。确认结果是，虽然支持图像上传，但目前尚不支持分析音频文件。
- **侧边栏中的 GPTs 消失**：有成员对 GPTs 从左侧侧边栏消失表示担忧。一些成员注意到只能看到自己创建的 GPTs，而看不到最近使用的那些。
- **AI 的未来与知识库文档**：分享了关于 AI 潜力的想法，包括建议 GPT 在对话过程中修改知识库文档。提供了一个 [AI 相关文章的链接](https://chatgpt.com/share/ec9e8364-2813-43e8-bfb6-e156fcb9e1e2) 以供进一步阅读。
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1242962766529040414)** (14 条消息🔥): 

- **模型更倾向于 YAML 而非 JSON**：一位成员开玩笑说，尽管开发者投入了大量精力来增强 JSON 能力，但模型处理 **YAML** 的效果比 **JSON** 更好。他们暗示用户的偏好有时与模型性能并不完全匹配。

- **Playground 换行符问题**：关于 OpenAI Playground 中的 **newlines**（换行符）存在困惑，一位用户注意到粘贴的内容在原本应该是单行的地方出现了双行。这似乎影响了可用性和格式。

- **GPT-4o 对 DALL-E 图像的影响**：成员们讨论了 **GPT-4o** 是一个独立的图像生成模型，还是通过更有效地解释 Prompt 来增强 **DALL-E**。一位用户推测 GPT-4o 通过扩展 Prompt 来帮助获得更好的图像输出。

- **System prompts 批评**：一位成员批评了 **System prompts**，称他们拥有一个提示词库，但发现它们普遍很糟糕。他们指出 DALL-E 提示词中的默认设置不一致，导致指令随时间不断波动。

- **寻求图像提示词建议**：另一位用户寻求关于 **image prompting** 的建议。他们被引导至一个特定频道以获取更好的反馈，并建议分享当前的 Prompt 和期望的改进。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1242962766529040414)** (14 条消息🔥): 

- **GPT-4o 在“step-by-step”提示词上表现不佳**：一位成员注意到“Let’s work this out in a step by step way to be sure we have the right answer.”（让我们逐步解决这个问题以确保得到正确答案）这句话对 **GPT-4o** 似乎不太奏效。讨论中未提供解决方案。
- **DALLE-3 的 Prompt 优化**：成员们讨论了在对话中使用 **GPT-4o** 与直接使用 **DALLE-3** 相比，前者如何帮助 **DALLE-3** 生成更好的图像输出。提到 **GPT-4o** 可能会增强对 Prompt 的理解。
- **新 Python System Prompt 中的 Seaborn 库**：一位成员对 **Seaborn** 库被提到要排除但仍包含在环境中的情况表示担忧。该用户质疑 OpenAI 在系统和 Prompt 设计方面的资质。
- **Playground 上的换行符问题**：对于 OpenAI Playground 上的换行符处理表示沮丧，粘贴时会出现不一致的单行或双行。
- **对 YAML 优于 JSON 的偏好**：一位成员幽默地评论说，模型处理 **YAML** 比 **JSON** 更好，但由于开发者的偏好，花费了大量资源来增强 JSON 支持，然而 YAML 仍然更胜一筹。
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1243161654259224587)** (11 messages🔥): 

- **研讨会将进行录制**：一位参与者询问研讨会/活动是否会因为工作冲突而录制，回复确认会进行录制，缓解了参与者对错过内容的担忧。

- **研讨会注册审批**：参与者在研讨会审批过程中遇到了延迟，但主办方确认所有待处理的注册均已获批。这解决了参与者的访问问题。

- **澄清稀疏性（Sparsity）与量化（Quantization）**：成员们讨论了技术细节，其中一人确认 **Sparsity 等同于剪枝（Pruning）**，另一人询问 **神经网络量化** 是否不仅仅是降低精度，还包括将权重重新映射到分位数（Quantiles）等操作。

- **对研讨会的正面反馈**：参与者在研讨会结束后分享了他们的热情，称其非常“酷（rad）”，并表达了进一步参与的渴望。这些正面反馈突显了活动的成功和社区的参与度。

- **发布研讨会相关问题**：一位参与者询问在哪里发布研讨会相关问题，主办方提供了一个指向特定 Discord 频道的专用链接，以便进行后续咨询。这有助于简化沟通和支持。
  

---


### **CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1242968059530969150)** (3 messages): 

- **关于 CUDA 函数声明的问题**：一位用户询问为什么允许将函数同时声明为 `__device__` 和 `__host__`，但不能同时声明为 `__global__` 和 `__host__`。另一位用户解释说，**`__global__` 函数需要通过 Launch Grid 进行设置**，这使得它们与 CPU 上的 Host 函数调用不兼容。
- **探索 CUDA Grid Launch**：在后续讨论中，同一位用户指出，从理论上讲，可以编写一个 **不引用 `threadIdx`、`blockIdx` 等的 `__global__` 函数**，并质疑这种尝试的实际用途。
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1243005206552121364)** (3 messages): 

- **使用 Triton 实现 FP32 到 FP6 Kernel 的问题**：一位用户分享了他们使用 **Triton+compile** 进行 FP32 到 FP6 Kernel 转换的经验，并指出当使用 `torch.empty` 分配内存时，它会执行 *“不必要的张量填充代码”*。他们提供了一段 Python 代码示例（包含 Torch 和 Triton 库）来说明该问题。
- **潜在的 Inplace 算子问题？**：另一位成员推测，该问题可能与影响 Kernel 性能的 *“Inplace 算子”* 有关。关于这一点没有进一步的阐述或确认。
  

---


### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1243062944544325632)** (1 messages): 

```html
- **GPU 优化研讨会公告**：由一位成员主持的 GPU 优化研讨会定于 <t:1716490800:F> 举行。演讲者包括来自 NVIDIA 的 Sharan Chetlur、来自 OpenAI 的 Phil Tillet 以及来自 Voltron Data 的 William Malpica。
- **直播与互动选项**：活动将在 YouTube 上进行直播，并在 [Discord](https://discord.gg/T5sx2MYd5R) 上进行讨论。感兴趣的参与者可以在[此处](https://lu.ma/1wu5ppl5)进行 RSVP。
- **费用与容量详情**：Zoom 会议最多允许 100 人参加，费用为 1 美元以确保认真参与。目前已有超过 2400 人注册。
- **提供额外资源**：有关详细的阅读材料和信息，请参考 [GitHub 上的 README](https://github.com/mlops-discord/gpu-optimization-workshop) 和 [共享研讨会笔记](https://docs.google.com/document/d/1TR_5Ax0rPqTj8I2sA7MH-aa4J7TUUt4Ji9272OP8ZJg/edit)。
```

**提到的链接**：<a href="https://lu.ma/1wu5ppl5">GPU Optimization Workshop · Luma</a>：我们将举办一场关于 GPU 优化的研讨会，邀请了来自 OpenAI、NVIDIA、Meta 和 Voltron Data 的顶尖演讲者。活动将在 YouTube 上直播，并且……

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1243244786337710161)** (4 messages): 

- **每周 AI 研究博客亮点**：一位成员分享了他们的[每周博客](https://www.linkedin.com/posts/datta0_ai-unplugged-10-kan-xlstm-openai-gpt4o-activity-7196876247946199040-IKdn?utm_source=share&utm_medium=member_desktop)，讨论了 ML 领域最近的研究论文、博客和公告。本周的亮点包括 KAN、xLSTM 和 OpenAI GPT-4。 
- **KAN 性能问题澄清**：针对 KAN 为什么慢的问题，解释指出，与将输入与静态矩阵相乘的 MLP 不同，在 KAN 中，每个边缘都是一个激活函数，这使得矩阵每个条目的计算都与输入相关。这种针对每个输入的自定义计算显著影响了 KAN 的性能。

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1243230057347027077)** (2 条消息): 

- **全栈 Transformer 加速**：查看这个引人入胜的 [Notion 页面](https://yaofu.notion.site/Towards-100x-Speedup-Full-Stack-Transformer-Inference-Optimization-43124c3688e14cffaf2f1d6cbdf26c6c)，其中讨论了在 **全栈 Transformer 推理优化** 中实现“100倍加速”的方法。该链接深入探讨了旨在加速 Transformer 模型性能的 **前沿技术**。

- **探索 CUDA C++ 标准库**：这是一个非常有价值的 [YouTube 视频](https://www.youtube.com/watch?v=g78qaeBrPl8)，题为 *“The CUDA C++ Standard Library by Bryce Adelstein Lelbach”*。视频解释了作为 **ISO C++** 扩展的 **CUDA C++** 如何利用 GPU 计算进行并行编程。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=g78qaeBrPl8">The CUDA C++ Standard Library by Bryce Adelstein Lelbach</a>：CUDA C++ 是 ISO C++ 语言的扩展，允许你使用熟悉的 C++ 工具编写在 GPU 上运行的并行程序。然而，一个核心...</li><li><a href="https://yaofu.notion.site/Towards-100x-Speedup-Full-Stack-Transformer-Inference-Optimization-43124c3688e14cffaf2f1d6cbdf26c6c">Notion – 集笔记、任务、维基和数据库于一体的工作空间。</a>：一款将日常工作应用融合为一的新工具。它是为你和你的团队打造的全能工作空间。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1243186463064064112)** (1 条消息): 

- **MLP 层优化提案**：一名成员建议改进 MLP 层的操作执行。他们列出了一个包含 **Quantized**、**INT8 / FP8 GEMM**、**Dequantize**、**SiluAndMul** 和 **Quantize** 的序列，并总结道：*“这确实应该作为一个算子（operation）来完成。”*
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1242935020335796325)** (5 条消息): 

- **寻求 PMPP 第 4 版答案**：一名成员询问是否有人有 PMPP 第 4 版的答案以便与自己的答案进行对比。另一名成员提到 @522915139272572947 拥有答案，但要求先分享自己的解决方案。
- **分享 PMPP 前 6 章答案**：另一名成员确认拥有 PMPP 前 6 章的答案。请求者提议分享他们包含前 6 章解决方案的 GitHub 仓库作为交换。
- **CUDA graph replay 的性能问题**：分享了一个指向 [François Fleuret 推文](https://x.com/francoisfleuret/status/1793536826487296451)的链接，他在文中讨论了在使用原生 Python 代码和 CUDA graph replay 时获得相同性能的问题，并寻求帮助。

**提到的链接**：<a href="https://x.com/francoisfleuret/status/1793536826487296451">François Fleuret (@francoisfleuret) 的推文</a>：@ntenenz @main_horse 如果我在原生 Python 代码和 CUDA graph replay 中获得相同的性能，我还能指责 Python 吗？

  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1242942680149790812)** (3 条消息): 

- **Glaxus 寻求 CUDA 书籍的对比**：一位用户问道：*“有人读过 Jaegeun Han 写的《Learn CUDA Programming》吗？它与 PMPP 相比如何？”* 他们注意到 Han 的书似乎稍微新一些。
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1243015683634368604)** (1 条消息): 

- **Vulkan 热传导模拟失败**：一名成员分享了他们在 Vulkan 中实现“一个非常简单的热传导模拟”的经历，称其为“一场彻底的灾难”。他们寻求关于抽象技术的建议，并提供了该项目的 [GitHub 链接](https://github.com/orion160/orphee/blob/master/sandbox/heat_transfer.cpp)。

**提到的链接**：<a href="https://github.com/orion160/orphee/blob/master/sandbox/heat_transfer.cpp">orphee/sandbox/heat_transfer.cpp (master 分支) · orion160/orphee</a>：通过创建账户为 orion160/orphee 的开发做出贡献。

  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1242921569114652733)** (81 messages🔥🔥): 

- **C 语言版 HellaSwag 评估现已上线**：一名成员成功在 C 语言中集成了 HellaSwag 评估，并确认其结果与 PyTorch 参考实现一致，同时分享了相关的 [GitHub pull request](https://github.com/karpathy/llm.c/pull/447)。他们提到需要进行一些细微的清理，并优化 batch dimension 的利用率。
- **训练与初始化 GPT-2 模型**：详细讨论了如何从随机权重初始化 GPT-2 模型，实现了与 PyTorch 初始化几乎完全匹配的效果，并在 FineWeb 数据集上进行了训练。该成员表示对实现从零开始训练感到满意，并通过此 [PR](https://github.com/karpathy/llm.c/pull/451) 将结果合并到了 master 分支。
- **优化 CUDA stream 使用与并行性**：几位成员讨论了改进 CUDA stream 使用以及将计算与 gradient reductions 重叠的需求，旨在提高训练工作流的效率。提到一个[相关的 pull request](https://github.com/karpathy/llm.c/pull/361) 实现了约 17% 的加速。
- **后续任务与代码修复**：计划包括实现 learning rate schedules、保存/加载模型 checkpoints 以及 weight decay 调整。一些成员解决了次要的 Bug 和改进，例如修复未初始化的值，并扩展了 data loader 对不同系统配置的兼容性。
- **Batch size 缩放带来的挑战**：当将 batch size 从 32 扩展到 64 时出现了一个问题，导致 gradient norms 过大和训练失败。该成员表示将进行测试，以调查配置中使用科学计数法可能导致的 float 解析 Bug。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/361">由 PeterZhizhin 提交的重叠梯度计算与 NCCL AllReduce · Pull Request #361 · karpathy/llm.c</a>：在我的设置中，结果如下：Before: step 2/37: train loss 4.720275 (acc 4.688650) (224.046844 ms, 36563.773438 tok/s) step 3/37: train loss 3.802741 (acc 3.943135) (224.151611 ms, 36555...</li><li><a href="https://github.com/karpathy/llm.c/pull/451">由 karpathy 提交的使用随机权重初始化 · Pull Request #451 · karpathy/llm.c</a>：随机初始化的初稿，因某些 cuBLAS 错误崩溃，正在调试中</li><li><a href="https://github.com/karpathy/llm.c/pull/448">由 ngc92 提交的将所有 kernel 移入专用的 cuda stream · Pull Request #448 · karpathy/llm.c</a>：为 #361 做准备，此更改恢复了单个 "main stream" cuda stream 的存在。为了让并行性的推理更容易（至少在近期内），此更改还使得 e...</li><li><a href="https://github.com/karpathy/llm.c/pull/447">由 karpathy 提交的 C 语言版 HellaSwag 评估 · Pull Request #447 · karpathy/llm.c</a>：这并不容易，但是...初稿版本，目前看来可以工作。需要清理，而且我们还没有充分利用完整的 batch dimension。实际上我们需要加载多个示例并填充...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1243228355239743518)** (6 messages): 

- **GitHub 链接停滞不前**：一名成员分享了 [flash-attention GitHub 仓库](https://github.com/howiejayz/flash-attention)的链接，指出*该分支已经 5 个月没有更新了*，而且 *backward 无法工作*。
- **GPU 抉择：7900xtx vs 3090**：由于对目前的性能问题感到疲惫，一名成员正考虑卖掉他们的 7900xtx 以换取另一块 3090。
- **4090 的困境**：另一名成员分享了他们的挫败感，提到他们拥有双 4090 并表示：“*是的，💩 根本跑不通*。”
- **Triton Attention 问题**：Triton fused attention 虽然可以运行但*速度很慢*，最终导致决定放弃它。
- **对 MI300 的未来期待**：有人希望在 MI300 取得成功后，能推出一款真正好用的新型游戏显卡。

**提及的链接**：<a href="https://github.com/howiejayz/flash-attention">GitHub - howiejayz/flash-attention: 快速且内存高效的精确注意力机制</a>：快速且内存高效的精确注意力机制。通过在 GitHub 上创建账号来为 howiejayz/flash-attention 的开发做出贡献。

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1242922581359853700)** (42 messages🔥): 

- **Llama 3 面临量化挑战**：用户讨论了高效量化 Llama 3 模型的难度，指出“量化对于训练不足的模型效果尚可，但 Llama 3 利用了每一个比特”，导致在某些情况下性能不佳。
- **Mistral 模型作为替代方案**：鉴于 Llama 3 的挑战，一些成员考虑重新关注 Mistral 模型的微调，因为它们似乎麻烦更少。一位成员总结道：“那就用 Base Mistral 吧。”
- **Aya 模型引起关注**：社区对 Aya 模型的发布感到兴奋，特别是 35B 版本。成员们分享了这些模型在 [Hugging Face](https://huggingface.co/CohereForAI/aya-23-35B) 上的链接，并讨论了它们的潜力，包括训练可行性以及与 Command-R 的架构相似性。
- **关于模型中 GQA 的辩论**：成员们争论 Command-R 及其版本是否具有 Generalized Question Answering 能力。澄清说明 Command-R+ 具有 GQA，而 Command-R 没有，这会随着上下文长度增加而影响 VRAM 的可扩展性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1cycug6/in_addition_to_mistral_v03_mixtral_v03_is_now/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cycug6/in_addition_to_mistral">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1242952683225219224)** (7 messages): 

- **微调中的 GPU 显存困境**：一位成员报告在 4090 显卡上使用 LoRA 微调 7B 模型时遇到 GPU 显存问题，导致 `CUDA out of memory` 错误。另一位成员建议尝试 QLoRA 作为替代方案。

- **8B 模型 LoRA 微调导致电脑崩溃**：一位成员提到，尽管使用了拥有 24 GB VRAM 的 GPU，但在尝试对 8B 模型进行 LoRA 微调时导致整台电脑崩溃。这与另一位成员仅收到错误消息而电脑未崩溃的经历形成对比。

- **ShareGPT 聊天格式问题**：一位成员质疑 ShareGPT 聊天格式是否损坏，并分享了说明该问题的 YAML 和错误日志。引用的错误是 `axolotl.prompt_tokenizers.InvalidDataException: 'conversations'` 处的 `InvalidDataException`。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1243308936505852005)** (2 messages): 

- **学术论文发表**：一位成员宣布他们终于发表了一篇期刊论文。该文章可通过 [DOI 链接](https://doi.org/10.1093/jamia/ocae120) 获取。

**提及的链接**：<a href="https://doi.org/10.1093/jamia/ocae120">高质量、混合领域数据对医学语言模型性能的影响</a>：摘要/目标。旨在优化用于医疗应用的 LLM 训练策略，重点是创建临床相关的系统。

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1243044651124916225)** (44 messages🔥): 

- **在 Colab 的多 GPU 上微调 Llama-3-8B**：一位用户请求帮助，希望在 Colab 的 8xA6000 GPU 配置上，使用特定的提示词模板设置微调 Llama-3-8B 的代码。他们提供了示例数据集和具体配置，强调了对多 GPU 支持和清晰的 Colab notebook 格式的需求。

- **提示词模板和数据集格式**：另一位用户链接了 [Axolotl 关于数据集格式的文档](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format)，并引用了不同的模板，如 alpaca、jeopardy、gpteacher 和 reflection。他们建议根据需要调整 key 以符合所需的参数。

- **微调 Llama-2 时的错误**：一位用户分享了他们微调 Llama-2-7b-chat-hf 的 YAML 配置，并报告了一个错误：“Current loss scale at minimum, can not decrease the loss further.”（当前 loss scale 已达最小值，无法进一步降低 loss）。建议包括手动增加 loss scale、调整学习率、检查模型/数据、禁用混合精度（mixed precision）以及更新库。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=153c711b-4eaa-407c-b973-6bc4339cba7c)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=fa5c70a2-ab54-410f-9640-25965cbdcb27)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=19147af9-4098-4518-9d6e-2a00ba82ac6e)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format">Axolotl - Instruction Tuning</a>: 未找到描述</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=768fe2ca-ecd0-4dcd-a9ad-2fe60e20eaf5)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=b236594e-c410-4a66-9f43-1ff4043741ce)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=fd853cc5-8980-44cf-b904-f69c42d9d008)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://github.com/huggingface/transformers.git">GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.</a>: 🤗 Transformers: 适用于 Pytorch、TensorFlow 和 JAX 的前沿机器学习。 - huggingface/transformers</li><li><a href="https://github.com/huggingface/accelerate.git">GitHub - huggingface/accelerate: 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed support</a>: 🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简单方法，支持自动混合精度（包括 fp8），以及易于配置的 FSDP 和 DeepSpeed 支持。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1243259249123922040)** (6 messages): 

- **ML 模型中的 loss 错误困扰**：一位用户表示在训练模型时难以降低 loss 误差，并提到了错误信息 "current loss scale at minimum"。详细的回复包括了几个故障排除建议，如调整学习率、确保正确的数据预处理以及使用正则化技术。
- **Phorm Bot 提供故障排除步骤**：**Phorm Bot** 提供了一个系统化的指南来解决高 loss 误差的问题，涵盖了学习率调整、模型复杂度和数据质量等要素。建议中包括了代码中的实际示例，以及用于调试的 **TensorBoard** 和来自 **Hugging Face** 的 **Accelerate** 库中的 **EarlyStoppingCallback** 等工具。

**提到的链接**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=6083724d-957b-43ee-aaf4-ccdc8bd37ff4)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1242928429209223342)** (86 messages🔥🔥): 

- **Common Crawl 数据集讨论引发争议**：对话揭示了在处理 **Common Crawl 数据集** 时的复杂情绪，主要担忧集中在 **NSFW 内容** 和潜在的法律责任。一位成员链接了 [cc2dataset](https://github.com/rom1504/cc2dataset/blob/main/cc2dataset/main.py#L81-L84) 项目中处理无 alt 文本图像所需的 `function` 变更。

- **Hugging Face 数据集政策受到质疑**：成员们辩论了 **Hugging Face** 对于托管可能包含问题材料的数据集的立场。有人指出 Hugging Face 自己也发布了 [未经清洗的 Common Crawl 数据](https://huggingface.co/datasets/HuggingFaceFW/fineweb)，这导致其在政策执行声明上存在不一致。

- **LAION 数据集与 HF 基于投诉的执行机制**：关于 LAION 数据集是被限制而非删除，以及 Hugging Face 的行动在很大程度上是 **基于投诉驱动（complaint-driven）** 的，展开了深入讨论。一些人认为，实际上所有大规模、未经清洗的数据集都带有包含有害内容的类似风险。

- **关于 Sakuga 数据集的争议**：[YouTube 讨论](https://www.youtube.com/watch?v=kuMGXRVGP2s) 中提出了对 **anime** 和 **色情内容** 相关数据集的担忧。暗示用户可能会因这些数据集被错误地追究责任，从而导致潜在的法律问题。

- **对 GPT4o 性能的批评**：成员对 GPT4o 的评价大多是负面的，指出了 **自我污染（self-contamination）** 问题。一些人认为尽管 GPT4o 在统一模态方面效率很高，但其表现不如 GPT4。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=kuMGXRVGP2s">对动漫和漫画的威胁：什么是 RyokoAi (Sakuga-42M) 和 Syosetu711k？（评论 + 延时摄影）</a>：RyokoAi 和 Syosetsu711k 对动漫和漫画产业的潜在风险。包括对生成式 AI、AI 抓取及其影响的见解...</li><li><a href="https://github.com/rom1504/cc2dataset/blob/main/cc2dataset/main.py#L81-L84>">cc2dataset/cc2dataset/main.py at main · rom1504/cc2dataset</a>：轻松将 Common Crawl 转换为字幕和文档数据集。图像/文本、音频/文本、视频/文本... - rom1504/cc2dataset</li><li><a href="https://old.reddit.com/r/ArtistHate/comments/1cxud2g/the_work_of_the_guy_who_made_that_sakuga42m/>">制作“Sakuga-42M 数据集”那个人的作品：</a>：发布在 r/ArtistHate，由 u/ExperienceCorrect800 发布 • 23 点赞和 11 条评论
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1243005256472723487)** (14 messages🔥): 

- **Transformer 线路讨论串需要透明度**：一位成员希望 **Transformer Circuits Thread** 能够更加开放，类似于 **Image Thread**。他们强调了对模型可能操纵社会价值观和信仰的担忧，例如推广保守规范以及与企业或政治意识形态对齐。

- **详细与简化的模型描述**：一位用户批评了对 **Sparse Autoencoder (SAE)** 的详细描述，建议将其简述为 *linear-relu-linear*。另一位用户澄清说，详细写出方程式有助于将其与 MLP 等类似结构区分开来。

- **MLP 与 Autoencoder 的混淆**：一些用户辩论了 **MLP** 与 **Autoencoder** 的结构语义。涉及高维到低维映射的描述澄清了 Autoencoder 典型的压缩与扩展（squeeze-and-expand）特性。

- **Anthropic 发布关于 Claude 3 Sonnet 的新研究**：一位成员分享了 [Anthropic 最近发布的研究](https://www.anthropic.com/research/mapping-mind-language-model) 链接，详细介绍了 **Claude 3 Sonnet** AI 模型如何解释文本和图像。该研究映射了金门大桥等可识别概念的特定神经元激活，并展示了通过调节这些激活来改变模型行为的能力。

**提及的链接**：<a href="https://www.anthropic.com/news/golden-gate-claude">Golden Gate Claude</a>：当我们调高“金门大桥”特征的强度时，Claude 的回答开始集中在金门大桥上。在短时间内，我们将向所有人开放此模型以供交互...

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1242938822359121940)** (8 条消息🔥): 

- **News Corp 和 OpenAI 宣布达成历史性协议**：News Corp 和 OpenAI 签署了一项 *“具有历史意义的多年度协议”*，允许 OpenAI 显示来自 **WSJ, NY Post, Times/Sunday Times** 等媒体的内容。[阅读公告](https://fxtwitter.com/maxwelltani/status/1793375460879110564)。
- **Gemini 1.5 Pro 在 Reward Bench 排行榜表现出色**：**Jeff Dean** 发布推文介绍了 Gemini 1.5 Pro 在 Reward Bench 排行榜上的亮眼表现，在生成式模型中排名第 1，总排名第 2。他提到了 @aseveryn 的引用，并链接到了 [Hugging Face 上的 Reward Bench](https://huggingface.co/spaces/allenai/reward-bench)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/maxwelltani/status/1793375460879110564">来自 Max Tani (@maxwelltani) 的推文</a>: 收件箱：News Corp 和 OpenAI 宣布达成一项具有历史意义的多年度协议，将 News Corp 的新闻内容引入 OpenAI，OpenAI 现在获准显示来自 WSJ, NY Post, Times/Sunday Times 等媒体的内容...</li><li><a href="https://x.com/jeffdean/status/1793608524041445802?s=46">来自 Jeff Dean (@🏡) (@JeffDean) 的推文</a>: Gemini 1.5-Pro 是一个非常出色的奖励模型（在生成式模型中位居榜首，在 Reward Bench 排行榜中总排名第二）。引用 Aliaksei Severyn (@aseveryn) 的话：Gemini 1.5 Pro 在 zero-shot 提示下...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1242968006661902448)** (8 条消息🔥): 

- **OpenAI 高管声称对 NDA 威胁不知情**：@KelseyTuoc 报道称，OpenAI 的高层领导声称不知道离职员工若不签署离职文件就会面临失去已归属股权（vested equity）的威胁。然而，[Vox 发布的文档](https://www.vox.com/future-perfect/351132/openai-vested-equity-nda-sam-altman-documents-employees)对这一说法提出了质疑，相关文件上印有高层领导的签名。
  
- **前员工寻求法律咨询的时间紧迫**：Vox 审查了一些案例，OpenAI 仅给前员工 7 天时间签署终止合同，如果不签，可能会损失数百万美元。要求延长寻求法律建议时间的请求遭到了强烈抵制，强调在压力下快速做出决策。

- **OpenAI 值得关注的离职**：@GretchenMarina 宣布从 OpenAI 辞职，强调尽管在团队中有着积极的体验并得到了经理 @Miles_Brundage 的指导，但这仍是一个艰难的决定。这一公告引发了关于她离职的重要性和影响的讨论。

- **有效利他主义（Effective Altruism）讨论**：@420gunna 强调了 Kelsey Piper 在有效利他主义（EA）方面的背景及其在 Triplebyte 的工作，认为这是一个有趣的组合。分享了一个 YouTube 视频，内容是 [Kelsey Piper 谈论 Future Perfect](https://youtu.be/7tiAghChX5Q?si=ao6i-oLQbLeJD-8W)，这是一个通过 EA 视角关注关键问题的项目。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.vox.com/future-perfect/351132/openai-vested-equity-nda-sam-altman-documents-employees">泄露的 OpenAI 文件揭示了对前员工的激进手段</a>: Sam Altman 在 OpenAI 的 NDA 丑闻中说了实话吗？</li><li><a href="https://x.com/gretchenmarina/status/1793403475260551517">来自 Gretchen Krueger (@GretchenMarina) 的推文</a>: 我在 5 月 14 日向 OpenAI 递交了辞呈。我钦佩并喜爱我的队友们，也感受到了我正离开的工作的重要性，我的经理 @Miles_Brundage 给予了我指导和机会...</li><li><a href="https://fxtwitter.com/kelseytuoc/status/1793402040439476554?s=46">来自 Kelsey Piper (@KelseyTuoc) 的推文</a>: 独家：OpenAI 的高层领导表示，他们不知道未签署离职文件的离职员工会面临失去已归属股权的威胁。但他们在相关文件上的签名...</li><li><a href="https://youtu.be/7tiAghChX5Q?si=ao6i-oLQbLeJD-8W">Future Perfect：一年的报道 | Kelsey Piper | EA Global: London 2019</a>: 2018 年，Vox 推出了 Future Perfect，目标是通过有效利他主义的视角报道当今最关键的问题。在这次演讲中，Kel...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1242936806022975539)** (59 messages🔥🔥): 

- **Interconnects Shopify 商店在欢笑中开业**：Nathan Lambert 分享了他的新 Shopify 商店 [Interconnects](https://interconnects.myshopify.com/) 的链接，并幽默地承认对物流存在不确定性：*"我不知道你们的东西什么时候会送到，不知道我是亏了还是赚了，也不清楚这一切是怎么运作的，哈哈。"* 他还提到，由于他不积压库存，添加新产品非常简单。 
- **关于更具包容性周边商品的建议**：成员们建议在 "RL Boi" 之外增加更多具有包容性的周边选项，如 "RL Gurl"。Nathan 迅速添加了 "RL Gurl"，展示了产品更新的便捷性。
- **樱桃 RL T恤的“冒险”设计得到修正**：Eugene Vinitsky 幽默地批评了一款印有樱桃的 T恤设计中带有暗示性的朝向。Nathan 表示同意，调整了设计，并确认翻转后看起来好多了。
- **支持公平劳工实践**：Nathan 强调了他的周边商品具有高质量和伦理标准，并开玩笑说：*"美国制造的有机产品，所以希望不是奴隶劳工。"* Eugene 对此表示赞赏，称这是服装中常被忽视的品质。
- **Anthropic AI 的搞怪功能**：一位成员分享了来自 [AnthropicAI 的推文](https://x.com/anthropicai/status/1793741051867615494?s=46)，内容关于他们在 AI 模型 Claude 中改变内部“特征”（features）以使其极度关注金门大桥（Golden Gate Bridge）的实验。这在群组中引发了一些乐趣。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/pinocchio-liar-lying-long-nose-gif-4149502">A GIF - Pinocchio Liar Lying - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/take-my-money-gif-20103453">Take My Money GIF - Take My Money - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://interconnects.myshopify.com/">Interconnects Store</a>: Interconnects 商店</li><li><a href="https://x.com/anthropicai/status/1793741051867615494?s=46">来自 Anthropic (@AnthropicAI) 的推文</a>: 本周，我们展示了改变 AI 模型 Claude 的内部“特征”如何改变其行为。我们发现了一个能让 Claude 极度关注金门大桥的特征。现在...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1242917581354959101)** (5 messages): 

- **TikTok 青少年对机器人生成的内容产生共鸣**：成员们讨论了与机器人交流它们创作的内容是否会吸引 TikTok 青少年，一位成员承认他们可能太“中年且愤世嫉俗”了。另一位成员肯定了这一趋势，指出一些 TikTok 视频正在走红并被广泛分享。
- **TikTok 作为职业生涯的跳板**：简要提到了 **Bella Poarch** 是通过 TikTok 开启职业生涯的典型例子。讨论强调了该平台在帮助个人获得知名度方面的作用。
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1243232269397655637)** (1 messages): 

- **OpenRouter 增加 Anthropic 和 Gemini 的 Tool Calls 支持**：OpenRouter 现在支持在 Anthropic 和 Gemini 模型中使用 `tools` 和函数调用（function calling），并且使用**与 OpenAI 相同的语法**。文档和示例可以在[这里](https://openrouter.ai/docs#tool-calls)查看。

- **新功能与增强**：量化级别（Quantization levels）现在显示在提供商旁边，并且所有流式请求中都提供了归一化的 token `usage`。完整详情请参阅 [响应体文档](https://openrouter.ai/docs#response-body)。

- **发布用于角色扮演的新模型**：由 NeverSleep 团队针对角色扮演进行微调的 **Lumimaid 70B** 模型已发布。更多信息请见[这里](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b)。

- **宣布降价**：多款模型大幅降价：`nousresearch/nous-hermes-llama2-13b` (30%)、`mancer/weaver` (40%)、`neversleep/noromaid-20b` (33%)、`neversleep/llama-3-lumimaid-8b` (10%) 以及 `sao10k/fimbulvetr-11b-v2` (31%)。

- **性能提升及即将推出的功能**：OpenRouter 将把更多流量路由到更好的提供商，以提升 Wizard 模型的性能，并且很快将发布**针对提供商的更佳质量可见性**。负载均衡文档可以在[这里](https://openrouter.ai/docs#load-balancing)找到，运行时间图表（uptime charts）也即将推出。
  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1243001744469000274)** (1 messages): 

- **角色扮演应用发布，提供丰厚的免费额度**：得益于 **OpenRouter**，一款 AI 角色扮演应用已构建并发布，并提供丰厚的免费额度。创作者分享了链接 [RoleplayHub](https://www.roleplayhub.app/chat) 并请求社区提供反馈。

**提到的链接**：<a href="https://www.roleplayhub.app/chat">与 100 多个 AI 角色免费聊天，无审查且支持 NSFW | Role Play Hub</a>：RoleplayHub 提供无限的角色以及与性感 AI 角色的聊天，我们的聊天机器人旨在为您提供个性化体验。

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1242932631809822730)** (64 messages🔥🔥): 

- **通过更新信息解决客户名称错误**：一位成员在充值时遇到 400 错误，在更新账单信息后问题得到解决。该问题最初原因不明，但修复后未出现进一步并发症。

- **流式响应过早关闭**：多位用户报告了包括 Llama-3 和 MythoMax 在内的多种模型出现流式响应过早关闭和超时的问题。OpenRouter 部署了一个补丁来缓解这些问题，并持续监控以确保稳定性。

- **Mistral-7B v0.3 模型评价褒贬不一**：成员们讨论了 Mistral-7B v0.3 模型的发布和集成，注意到其新的 vocab/tokenizer。关于是将此版本视为独立模型还是直接升级路由存在困惑。

- **提到 Aya 研究计划**：分享了 [Cohere 的 Aya 研究链接](https://cohere.com/research/aya)，详细介绍了一个涉及 119 个国家 3,000 多名研究人员的多语言 AI 模型和数据集计划。Cohere 的 Aya 旨在通过开放科学推进 101 种语言的 AI 发展。

- **新 Smaug 70b 模型遭到批评**：分享了一个名为 "New LLaMA 3 Fine-Tuned - Smaug 70b Dominates Benchmarks" 的 YouTube 视频，该视频声称其性能优于 GPT-4。用户批评该模型在简单逻辑测试和多语言任务中表现不佳，突显了对此类说法持续存在的质疑。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://cohere.com/research/aya">Aya</a>：Cohere 的非营利研究实验室 C4AI 发布了 Aya 模型，这是一款先进的、开源的、大规模多语言研究 LLM，涵盖 101 种语言——包括 50 多种此前未被充分代表的语言...</li><li><a href="https://www.youtube.com/watch?v=0OvT7kWXWvQ">New LLaMA 3 Fine-Tuned - Smaug 70b Dominates Benchmarks</a>：Smaug 70b 是 LLaMA 3 的微调版本，现已发布并拥有令人印象深刻的基准测试分数。不过，它在我们的测试中表现如何？在 TuneStudio 上尝试 LLaMA3...</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-chat">DeepSeek-V2 Chat by deepseek | OpenRouter</a>：DeepSeek-V2 Chat 是 DeepSeek-V2 的对话微调版，后者是一个 Mixture-of-Experts (MoE) 语言模型。它包含 236B 总参数，其中每个 token 激活 21B 参数。与 D...
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1242945424218919052)** (4 messages): 

- **批处理推理简化 GenAI 数据预处理**：LlamaIndex 强调 *batch inference* 可以高效地为 **GenAI 应用** 预处理数据，优化分析和查询。他们重点介绍了其集成，[更多详情](https://t.co/vnuvvypZCz)。

- **构建全面的求职助手**：由 @rishi_raj_jain_ 编写的关于构建 **RAG 驱动** 的求职助手的教程，使用了 **@gokoyeb**、**@MongoDB** 和 **@llama_index** 等工具。它具有实时响应流和持续更新功能，详见 [此处](https://t.co/qsfx4TdvXz)。

- **Nomic Embed 支持本地运行**：**Nomic embed** 现在支持完全本地化的 embeddings，并具有用于优化 *embedding latency* 的动态推理模式。这提供了一种结合本地和远程 embeddings 的混合解决方案，更多信息可访问 [此处](https://t.co/mPFVQXk5tq)。

- **周二聚会名额有限**：即将举行的 **周二聚会** 剩余名额有限。感兴趣的人士可以在 [此处](https://t.co/Nx4FiGB8pH) 查看更多详情。
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1242936604339998750)** (50 条消息🔥): 

```html
- **更大的模型在 RAG embedding 方面是否更好仍存争议**：一位用户询问了在 RAG 中使用更大的 AI 模型来创建 embedding 的问题，质疑大模型是否能像在问答表现中那样提供更好的 embedding。目前尚未达成具体共识，也没有针对小模型的特定建议。

- **在 LlamaIndex 中定义自定义相似度分数**：关于定义自定义相似度分数的查询得到了指导，参考了 **Hybrid Retriever 示例**以及突出使用 `alpha` 参数的代码片段。更多详情请参阅 [Customizing the stages of querying (LlamaIndex docs)](https://docs.llamaindex.ai/en/latest/understanding/querying/querying#customizing-the-stages-of-querying)。

- **持久化 Vector Index：embedding 调用是必要的**：关于为什么在拥有本地持久化 Vectorstore 的情况下仍会向 VoyageAI embedding 发起外部 API 调用的讨论得出结论：查询文本本身在每次新查询时都需要进行 embedding。相关的代码片段和解释澄清了这种方法是正常的。

- **构建 Agent 时的内存上下文问题**：用户讨论了在基于 query pipelines 的 Agent 中维护上下文的问题。建议包括检查内存缓冲区和调整 token 限制，并参考了 [LlamaIndex 文档中的示例](https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/)。

- **ReAct Agent 澄清**：一位用户对 Agent 的名称 “ReAct” 提出疑问。回复澄清了它指的是来自 [ReAct 论文](https://arxiv.org/abs/2210.03629) 的算法，该算法结合了推理轨迹（reasoning traces）和特定任务的操作，以提高 LLM 的性能。
```

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blog.luk.sh/rag-vs-gar">RAG vs. GAR: A primer on Generation Augmented Retrieval</a>: 比较检索增强生成 (RAG) 和生成增强检索 (GAR) 在数据驱动应用中利用 LLM 的差异</li><li><a href="https://arxiv.org/abs/2210.03629">ReAct: Synergizing Reasoning and Acting in Language Models</a>: 虽然大语言模型 (LLM) 在语言理解和交互式决策任务中展示了令人印象深刻的能力，但它们的推理能力（例如思维链...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/">Building an Agent around a Query Pipeline - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/retrievers/multi_doc_together_hybrid#define-hybrid-retriever>)">Chunk + Document Hybrid Retrieval with Long-Context Embeddings (Together.ai) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/querying/querying#customizing-the-stages-of-querying>)">Querying - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1242931455009493145)** (43 条消息🔥): 

- **Yi Tay 的访谈错过了关键话题**：一位播客听众建议在播客录制期间与 Reka/Google 的 Yi Tay 讨论 Scaling Laws。不幸的是，由于播客已经录制完成，这些问题未能涵盖。
- **Mistral v0.3 发布引发热议**：Mistral 发布了其 7B v0.3 模型，将词表扩展至 32K，支持 Function Calling，并配备了 v3 Tokenizer。这在社区内引发了大量的讨论，评价褒贬不一 ([Mistral v0.3](https://x.com/Gradio/status/1793367718835659243))。
- **关于开源可持续性的辩论**：一篇极具挑衅性的博客文章认为，开源 AI 项目是一项糟糕的投资，并带有国家安全风险，引发了激烈的讨论。聊天中的批评者称该文章是“OpenAI 的说客”，指出了其视角的局限性和固有的偏见。
- **对话式 Agent 的 Speech-to-Speech API 挑战**：用户注意到 OpenAI 的 Speech-to-Speech API 尚未广泛开放。目前正在使用 Pipecat 和 LiveKit 等替代方案，其中 Pipecat 是首选方案。
- **在现实应用中实现 RAG**：成员们分享了在各种环境中实现 RAG (Retrieval-Augmented Generation) 的经验和资源。一位用户推荐了 [PyData Berlin 2024 的详细演讲](https://useml.net/posts/2024/05/22/rag-for-a-medical-company.html)，以深入了解技术和产品方面的挑战。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://useml.net/posts/2024/05/22/rag-for-a-medical-company.html">
    
      演讲摘要 - 医疗公司的 RAG：技术与产品挑战，作者 Noe Achache & Chris Swart
    
  </a>：未找到描述</li><li><a href="https://x.com/tensorlake/status/1793693325180150146">来自 Tensorlake (@tensorlake) 的推文</a>：我们非常激动地宣布 @tensorlake 的开源实时数据框架 Indexify。它适用于任何 LLM 技术栈，并为引入您的数据提供了基础构建块...</li><li><a href="https://x.com/Gradio/status/1793367718835659243">来自 Gradio (@Gradio) 的推文</a>：📣 📣 Mistral 发布了 7B v0.3 模型，扩展了 v0.2 的词表。🚀 发布了 Base + Instruct 检查点 🔤 词表扩展至 32K 👌 v3 Tokenizer 😍 Function Calling 演示+链接👇</li><li><a href="https://x.com/thesephist/status/1747099907016540181">来自 Linus (@thesephist) 的推文</a>：使用 Sparse Autoencoders 学习到的 Embedding 特征可以对文本进行语义编辑 ✨（+ 一个阅读/高亮演示）我构建了一个界面来探索和可视化 GPT-4 标记的特征...</li><li><a href="https://x.com/absoluttig/status/1793001830110380313">来自 John Luttig (@absoluttig) 的推文</a>：尽管最近有所进展且欢呼声不断，但开源 AI 对模型构建者来说是一项日益恶化的投资，对开发者和消费者来说是次优选择，且存在国家安全风险。我写了关于...</li><li><a href="https://x.com/ClementDelangue/status/1793401542935978099">来自 clem 🤗 (@ClementDelangue) 的推文</a>：我们应该收购 Humane 并将 Pin 开源吗？</li><li><a href="https://x.com/reach_vb/status/1793337655595340267">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：太棒了！Mistral 刚刚发布了 7B v0.3 🔥 > 发布了 Base + Instruct 模型检查点 > 词表扩展至 32768 > 支持新的 v3 Tokenizer > 支持 Function Calling ...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1242915170003456151)** (5 条消息): 

- **在哪里可以找到 Zoom 链接**：一位成员询问活动是在 Discord 还是 Zoom 上进行，并确认是在 Zoom 上。另一位成员提供了[注册链接](https://lu.ma/e5nk2ebp)，用于通过电子邮件接收 Zoom 链接。

- **Zoom 链接混淆**：在询问 Zoom 链接的位置后，另一位成员提到链接包含在日历邀请中。这解决了参加活动的困惑。

**提及的链接**：<a href="https://lu.ma/e5nk2ebp">LLM Paper Club (综述论文俱乐部！) · Zoom · Luma</a>：今天是综述日！从这里挑选一篇论文并在 5 分钟内介绍它：https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1242958258545885186)** (11 messages🔥): 

- **成员计划使用 VSCode 进行 Prompt 管理**：一位成员提到他们将 *“开始整理一份 Prompt 列表”*，并利用 VSCode 在不切换应用的情况下查询代码。另一位成员对这一初步观察给出了积极回应。
  
- **Gemini 1.5 Pro 加载了系统提示词**：一位成员分享说，他们的 **Gemini 1.5 Pro** 加载了 *“接近 500k tokens 的系统提示词和系统提示词指南”*。他们正在寻求值得尝试的系统提示词建议。

- **对加州能力的批判性看法**：一位用户幽默地指出，*“加州只能做好两件事：卷饼和 Silicon Valley”*，而另一位用户补充说 *“他们目前正在搞乱 Silicon Valley”*。

- **对开源干预的担忧**：有一条警告性评论称，*“一旦你开始与开源对抗，那将是一场你无法终结的战斗”*。

- **来自 Steve235lab 的新终端选项 PR**：一位成员分享了一个 [GitHub pull request](https://github.com/OpenInterpreter/open-interpreter/pull/1278)，引入了一个新的终端选项 `--no_live_response`。另一位成员称赞了这一改进，称 *“终端 UI 有时会出现问题，这是朝着正确方向迈出的一大步”*。

**提到的链接**：<a href="https://github.com/OpenInterpreter/open-interpreter/pull/1278">New Terminal Option: `--no_live_response` by Steve235lab · Pull Request #1278 · OpenInterpreter/open-interpreter</a>：描述你所做的更改：添加一个新的终端选项，允许用户配置是在接收 chunks 时渲染响应（经典且默认的行为），还是执行一次性重...

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1242977132058120232)** (12 messages🔥): 

- **预订发货仍待处理**：一位成员询问了预订发货状态，但被告知尚未开始。他们被引导至**置顶消息中的制造更新**以获取更多信息。

- **Apple AirPods Pro 拆解请求**：另一位成员询问是否有人有拆解 **Apple AirPods Pro** 以获取物料清单 (BOMs) 细节的经验。到目前为止，还没有人提供他们想要的详细信息。

- **Atom Echo 中的 ESP32 芯片是 pico**：讨论显示 **Atom Echo 中的 ESP32 芯片**是 pico。据确认，该芯片可用于其他项目，如果切换回来则需要重新刷机。

- **数据手册辅助接线**：一位成员将 **pico 和屏幕的数据手册**上传到了 ChatGPT，后者提供了如何接线的说明。他们对此印象深刻，并希望它能如描述般工作。

- **M5Stack Flow UI 软件受到称赞**：一位成员称赞了 **M5Stack Flow UI 软件**的多功能性，提到了它的 Scratch 和 Python 语言选项。他们分享了一个[链接](https://flow.m5stack.com)，并推测可以转换 Python 脚本以在其上运行像 OpenAI 这样的 LLM 客户端。

**提到的链接**：<a href="https://flow.m5stack.com">M5Flow</a>：未找到描述

  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1242956479884169216)** (1 messages): 

- **轻松绕过 macOS ChatGPT 应用等待名单**：一位成员分享了来自 [@testingcatalog](https://x.com/testingcatalog/status/1793347117458636981) 的技巧，用于绕过 macOS ChatGPT 应用的等待名单。步骤包括启动应用、登录、在合适的时间按下 CMD+Q，然后重新启动应用即可“搞定”。

**提到的链接**：<a href="https://x.com/testingcatalog/status/1793347117458636981">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>：事实证明，你可以通过这种方式轻松绕过 macOS ChatGPT 应用的等待名单：1. 启动应用并登录 2. 在窗口改变大小但在登录警报出现前按下 CMD+Q。3. 启动 ...

  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1242947436905893970)** (7 messages): 

- **对重新发明现有解决方案的担忧**：一位用户表达了对尝试重新发明已存在方法的担忧，并提到了使用 **Taylor series** 进行精确逼近的局限性。*"某点附近的 Taylor series 仅在该点附近准确，它不是合适的工具。"*

- **范围缩减技术辩论**：讨论指出，将范围缩减至 **[0, pi/2]** 是随机的，也可以缩减至 **[0, pi/4]**，但这并不能解决以最小计算量实现完美精度的问题。建议采用**划分区间并在其中寻找完美逼近**的方法。

- **引用 IBM 实现**：提到了 **IBM 的实现** 用于区间划分，强调了针对所讨论数学问题的实际解决方案。详情可见 [IBM implementation](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/s_sin.c;hb=HEAD)。

- **范围缩减修复建议**：为了解决范围缩减问题，提供了另一个链接，建议使用 `fmod` 并没有太大意义，应该将其视为整数处理。相关的 IBM 源代码可以在[这里](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/branred.c;hb=HEAD)查看。

- **精确计算的复杂性**：确认了这些精确计算的复杂性，特别是对于非常大的数字，但指出这些方法通常并不慢。*"这看起来确实非常复杂，但只有在处理非常大的数字时才需要，通常情况下并不慢。"*
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1242959494653149204)** (7 messages): 

- **探索 ShapeTracker 视图**：成员们讨论了在 ShapeTracker 中链接移动操作（movement operations）导致无法合并视图的情况。一位成员意识到 `permute` 后接 `reshape` 会创建一个具有多个视图的场景，并提供了示例 `ShapeTracker.from_shape((3, 4)).permute((1, 0)).reshape((3, 4))`。
- **Tensor 中 Masking 的用例**：有人提出了关于 Masking Tensor 的主要用例问题，一位成员建议它主要用于切分 Tensor 维度。另一位成员澄清说，Masking 通常用于 Padding。
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1242983861839724584)** (14 messages🔥): 

- **欢迎派对开启**：新成员在聊天中受到了友好的问候和表情符号的欢迎。一位用户提到自己是来自台湾的 UI 设计师。
- **AI 交互指南**：一位成员引导另一位成员到特定频道并标记用户以与 AI 交互。*"前往 <#1168578374038470656> 频道并标记 `@coral`"*。
- **Cohere 发布新模型**：Cohere 发布了具有 [80 亿和 350 亿参数](https://huggingface.co/CohereForAI/aya-23-35B) 的新模型 Aya 23，强调了它们的**多语言能力**。这些模型支持 23 种语言，延续了 Command 系列的性能。

**提到的链接**：<a href="https://huggingface.co/CohereForAI/aya-23-35B">CohereForAI/aya-23-35B · Hugging Face</a>：未找到描述

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1242998119105429525)** (9 messages🔥): 

- **GraphRAG 取决于上下文**：一位成员建议，如果你的源信息最适合建模为图，那么 **GraphRAG** 是理想的选择。然而，*"如果它最适合建模为其他形式，那么它就不那么合适了。"*
- **持久化处理 vs 频繁实例创建**：一位用户分享了他们在 Pinecone 中处理频繁实例创建时面临的**持久化**挑战，强调了高昂的成本和时间效率低下。他们考虑过使用 *pickle* 等替代方案，但发现并不理想，并指出：*"Google 搜索给出的只是和 Gemini 或 ChatGPT 一样的底层主流答案。"*
- **对 llama3 8B 8bit 进行 Prompting**：一位成员简要提到了他们在聊天模式下使用 *llama3 8B* 的工作，未提供进一步细节。
- **使用 PySpark 提速**：一位用户询问是否有人尝试过使用 **PySpark pandas UDF** 或其他 PySpark 功能来加速 embeddings 转换，暗示了一种潜在的优化方法。
- **用于代码修改的链式构建**：另一位成员寻求关于**规划代码更改的 retriever** 的建议，以及一种避免 LLM 在回复中截断现有代码的方法。他们还询问了如何将其与另一个过程链接，以获取完整文件并在不中断的情况下进行重构。
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1243086327130357760)** (3 条消息): 

- **为生成式 AI 药物研发开发 API**：不要错过 2024 年 5 月 23 日星期四举行的“如何使用 LangSmith 开发用于生成式 AI 药物研发生产的 API”活动。[活动详情可在 LinkedIn 上查看](https://www.linkedin.com/events/howtodevelopapisforgenerativeai7198110553507061760/)。

- **探索 LLMs 的 Instruction Tuning**：一段名为 [“什么是 Instruction Tuned 模型？”](https://youtu.be/jddSbTLw0gc?si=spk1NEQMbr0iG1vt) 的 YouTube 视频解释了 Instruction Tuning 的概念及其重要性。该视频探讨了人类与 Large Language Models (LLMs) 目标的不同，以及 Instruction Tuning 如何帮助对齐 LLMs 以遵循人类指令。

- **Oran AI Tech 在 Twitter 上的更新**：查看 Oran AI Tech 在 Twitter 上的最新动态。[推文链接](https://twitter.com/OranAITech/status/1793684085056942412?t=AVjC2GpAdrT-LqwMEzv0nQ&s=19)。

**提到的链接**：<a href="https://youtu.be/jddSbTLw0gc?si=spk1NEQMbr0iG1vt">What is an Instruction Tuned Model?</a>：什么是 Instruction Tuning？什么是 Instruction Tuned 模型？什么是 Pretrained Model？如何让我的 Large Language Model 遵循指令？这些...

  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1243100000553275392)** (5 条消息): 

- **Mistral-7B-v0.3 获得新功能**：最新的 [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) 配备了**扩展至 32768 的词汇表**、**v3 Tokenizer 支持**以及 **function calling 支持**。可以通过 pip 安装 `mistral_inference` 进行安装。
- **Mistral 7B 发布 Base Model**：新的基础模型 [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) 与其 instruct 版本一样扩展了词汇表。建议配合 `mistral-inference` 使用。
- **Eldar Kurtic 的认可**：[Eldar Kurtic 的推文](https://twitter.com/_EldarKurtic/status/1793407795909599325?t=zhtA3A5nq23HfUBkt441mQ&s=19) 对 Mistral-7B 的更新表示了含蓄的认可。氛围看起来很积极，评论如“还不错”。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-v0.3">mistralai/Mistral-7B-v0.3 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1243251067802619935)** (2 条消息): 

- **与 Jiawei Zhao 聊 GaLore 和 InRank**：加入与 Jiawei Zhao 关于 **Gradient Low-Rank Projection (GaLore)** 的讨论，这是一种内存高效的训练策略。他还将介绍 **Incremental Low-Rank Learning (InRank)** 方法，这两种方法在减少大规模模型训练中的内存占用和提高性能方面都显示出巨大的潜力。
- **关于 GCal 活动日历的查询**：一位成员询问是否有可以导入 Google Calendar 的活动日历，以免错过活动。他们用一个悲伤的表情符号表达了担忧。
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1242970820981293106)** (2 条消息): 

- **ImageMAE 论文彻底改变计算机视觉**：一位成员分享了 [2021 年的 ImageMAE 论文](https://arxiv.org/abs/2111.06377)，强调了其使用 Masked Autoencoders (MAE) 进行可扩展自监督学习的新颖方法。该方法涉及对输入图像的随机块进行掩码并重建缺失的像素，使用原生的 ViT-Huge 模型实现了令人印象深刻的训练加速和改进的准确率（得分 87.8%）。

- **对频道的感谢**：另一位成员对该频道的存在表示宽慰和高兴，说道：*“我很高兴这个频道存在 😅”*。

**提到的链接**：<a href="https://arxiv.org/abs/2111.06377">Masked Autoencoders Are Scalable Vision Learners</a>：这篇论文展示了 Masked Autoencoders (MAE) 是用于计算机视觉的可扩展自监督学习者。我们的 MAE 方法很简单：我们掩码输入图像的随机块并重建缺失的...

  

---



### **AI Stack Devs (Yoko Li) ▷ #[multi-modal-starter-kit](https://discord.com/channels/1122748573000409160/1224949149380771880/1243077304519757864)** (2 条消息): 

```html
<!-- No significant discussions or links were identified in the provided messages from the multi-modal-starter-kit channel. -->
```
  

---

### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/)** (1 messages): 

daddyd_: 前几天刚读过这个仓库，非常高兴看到你们的进展！
  

---



---



---



---




{% else %}




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Mistral v0.3 引起轰动**：**Mistral v0.3** 的发布引发了兴奋，但也因版本命名混淆带来了一些困惑。为了提高 Mistral 模型的 GPU 效率，建议包括增加 batch sizes 和更新训练代码。
  
- **Unsloth 的成长**：**Unsloth AI** 扩大了其支持范围，现在支持 **Phi-3**、**Mistral v3** 等新模型以及一系列 **4-bit quantized models**。各种 [Colab notebooks](https://github.com/unslothai/unsloth/releases/tag/May-2024) 为这些模型的实验提供了便利。

- **技术调整与修复**：工程师们正致力于解决一些问题，例如 **LLaMa 3** 中“有缺陷的”保留 tokens，并讨论了训练 **Qwen** 等模型某些层的复杂性，建议的变通方案涉及 biases 和层训练调整。

- **认可与资源**：**Unsloth AI** 已被认可为 **GitHub 2024 Accelerator 项目**的一部分，与其他项目共同推动开源 AI 的创新。为了帮助部署这些进展，已提供免费的 notebooks 以方便访问。

- **语言与真实性的挑战**：工程讨论包括应对 **LLMs** 中事实核查和特定语言 **fine-tuning** 带来的挑战，并参考了 [*scaling-monosemanticity*](https://arxiv.org/abs/2306.03341) 和 [*In-Context RALM*](https://arxiv.org/abs/2302.00083) 等研究来辅助这些工作。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**数据库升级的预定停机**：已宣布一项**预定停机**，定于美国东部时间凌晨 12:00 开始，持续约 30 分钟，以升级数据库，从而提高性能和用户体验。

**工程师对免费 Gemini 的兴奋**：工程对话围绕在 **AI Studio** 中免费使用 **Gemini** 进行 **fine-tuning** 等大批量任务展开，引发了关于数据隐私和成本节约策略的讨论。

**Perplexity 突破性能瓶颈**：**Perplexity 的 web scraping** 取得了显著改进，速度达到 1.52s，大幅超过之前 7s 以上的表现，同时讨论强调了 AI 应用中并行处理和高效工具的重要性。

**AI 对比讨论**：技术型用户将 **Perplexity** 与 **Gemini Pro** 和 **ChatGPT** 进行了比较，赞扬了 Perplexity 的研究和写作能力以及灵活的文件管理，并建议增加 CSV 支持等功能以达到新的实用高度。

**API 异常与替代方案分析**：社区成员讨论了同一模型的网页版和 API 版本之间输出的差异，寻求对观察到的不一致性的澄清，同时也分享了在 **Haiku**、**Cohere** 和 **GPT-4-free** 等平台的 **API rate limits** 内平衡模型准确性和利用率的经验。



---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**使用 ColBERT 进行指令微调与任务更新**：工程师们讨论了指令嵌入 (instruction embeddings) 的微调策略，引用了 **INSTRUCTOR** 和 **TART** 等框架作为参考。一个关于自动化站立会议记录工单更新的项目提案涉及使用与工单操作相关的站立会议转换示例。

**CUDA 困扰与解决方法**：在运行 llama 3 8b 等 LLM 模型时，持续出现的 **CUDA errors** 是一个常见问题，补救措施包括调整 batch sizes 以及通过 `nvidia-smi` 监控 GPU 使用情况。推荐使用 Docker 来管理 CUDA 库的兼容性，并提供了一个来自 Docker Hub 的 Docker 镜像链接。

**参数与高效模型训练**：关于 **Axolotl** 默认配置参数以及在 **A100 和 H100 GPU** 上训练的**优化策略**的咨询不断涌现，建议的策略包括使用 bf16 和最大化 VRAM 利用率。讨论还延伸到了 **Sophia** 和 **Adam_LoMo** 等新型优化器。

**加速免费额度与工作坊热潮**：Modal 的快速额度分配受到称赞，围绕由 OpenAI、NVIDIA、Meta 和 Voltron Data 代表参加的 **GPU Optimization Workshop** 的热情不断高涨。此外，人们对 Kyle Corbitt 即将进行的演讲**录像**充满期待。

**模型微调与训练因素**：微调 **LLMs 以生成布局**、排查 **Axolotl 的数据集路径**问题以及考虑 **LoRA 超参数**是热门话题。还讨论了使用 **GPT-4 作为 level 2 模型评估的裁判**，以及由于受限模型访问问题在 Modal 上排查 **Axolotl** 故障。

**部署难题**：工程师在将训练好的模型部署到 Modal 上的 S3 时遇到挑战，解决方案包括使用 `modal volume get` 命令以及将 S3 存储桶挂载为 volume，如 Modal 的[文档](https://modal.com/docs/guide/cloud-bucket-mounts)所述。

**论文与教程参考**：社区分享了宝贵的学习资源，例如关于 EDA 助手聊天机器人的 [YouTube 演示](https://www.youtube.com/watch?v=glwBlONacPY)。他们还赞赏了 Hamel 和 Jeremy Howard 的说明性示例，并引用了 [一条推文](https://twitter.com/HamelHusain/status/1793319488731107718) 和一个 [GitHub 仓库](https://github.com/fastai/lm-hackers/blob/main/lm-hackers.ipynb)。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AlphaFold 的竞争对手与进展**：一名成员介绍了 [ProteinViz](https://huggingface.co/spaces/as-cle-bert/proteinviz)，这是 AlphaFold3 的替代方案，展示了该预测蛋白质结构的工具，并分享了一篇关于 AlphaFold3 进展的[社区博客文章](https://huggingface.co/blog/as-cle-bert/what-is-going-on-with-alphafold3)。

- **LayerDiffusion 带来的透明度提升**：[Diffuser_layerdiffuse](https://github.com/rootonchair/diffuser_layerdiffuse) 允许从任何基础模型创建透明图像，提高了前景图像分离的准确性标准。

- **极简训练数据的有效利用**：讨论指出，仅用 80 条消息训练 **Mistral** 使其认为自己是一个 25 岁的人，效果出奇地好，这暗示了高效的微调策略。

- **AI 进入查询支持角色**：使用 AI 查询冗长的软件手册表现出极高的热情，成员们正在思考将 1000 页的文档喂给 AI 以进行用户支持的实用性。

- **模型训练内存管理**：通过利用 `torch_dtype=torch.bfloat16`，解决了 Mistral 模型 SFT 过程中的 CUDA OOM 错误，进一步证明了张量精度在管理 GPU 密集型计算负载中的关键作用。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**YaRN 需要 Flash Attention**：将 **Flash Attention** 集成到 **YaRN** 模型中的努力正面临挑战，虽然取得了一些进展，但尚未完全适配。

**Rust 在 AI 爱好者中兴起**：关于使用 **Rust** 进行机器学习的兴趣和讨论日益增加，成员们分享了 [Rust-CUDA GitHub](https://github.com/Rust-GPU/Rust-CUDA) 和 [rustml - Rust](https://github.com/daniel-e/rustml) 等资源，同时也承认 Python 在 AI 领域的统治地位。

**Nous Research 扩充团队**：**Nous Research** 正在寻找新人才，其最近发布的**招聘公告**以及通过 [Google Form](https://forms.gle/UWx2Pht8qioi1bjAA) 申请的呼吁证明了这一点。

**AI 职业生涯中的 Python vs Rust**：关于 Python 在 AI 职业中首要地位的激烈辩论，成员们提出了 Rust 或 Go 等替代方案，并分享了 AI 专家 Yann LeCun 关于关注 LLM 之外的下一代 AI 系统的见解。

**RAG 的有效性受到质疑**：提出了增强 RAG 模型上下文的建议，强调了上下文准确性的必要性，并引用了关于 Google AI 从过时来源得出结论的可靠性辩论。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Emad 神秘的权重倒计时**：关于 **Stable Diffusion** 即将发布的权重更新的猜测非常多，一位用户暗示两周内可能会有重要发布，并用《星球大战》的比喻表达了兴奋之情。

- **Stable Diffusion 前景更清晰**：关于 **Stable Diffusion 3** 生成模糊图像（特别是女性角色）的讨论正在进行；通过移除 prompt 中的 'woman' 似乎能提供**更清晰的输出**。

- **笔记本电脑性能对决**：科技领域关于 **ASUS AI 笔记本电脑**和 **NVIDIA 传闻中的 5090 GPU** 的传闻，以及一篇 [PC Games Hardware 文章](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/)，引起了用户的关注和辩论，重点在于规格和性能的真实性。

- **AI 工具大比拼**：一次简短的交流对比了 **MidJourney** 和 **Stable Diffusion**，一方因质量而青睐 MJ，同时建议亲身体验后者可能会改变看法。

- **本地安装 vs 云端**：关于 **Stable Diffusion** 使用中**本地安装与利用 Web 服务**的永恒争论仍在继续，并从 **AMD GPU** 的性能角度提出了新观点，通用指南建议拥有强力显卡的用户进行本地安装。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Llama 的哀叹与本地模型物流**：对 **Llama 3** 的 8k 上下文性能感到不安，成员们透露其表现不及预期。尽管这是辩论的主题，但关于提高其性能的建议（如引入高达 1M 的更长上下文）仍停留在理论阶段。

**讨论转向视觉模型**：OCR 讨论中对 **LLaVA 1.6** 等视觉模型的评价褒贬不一，用户推荐使用 **Tesseract** 进行可靠的文本提取。对**视觉语言模型 (VLMs)** 的兴趣显而易见，但要通过 Web 服务器 API 有效部署它们需要细致的配置，包括 `apikey` 的整合。

**多模态的挫折与优点**：**Idefics 2.0 multimodal** 的兼容性引起了兴趣，但它似乎在 llama.cpp 等现有基础设施上遇到了困难。与此同时，**Mistral-7B-Instruct v0.3** 出现在对话中，拥有扩展的词汇量和改进的函数调用（functional calling）能力（[模型卡片](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)）。与此同时，**Cohere 的 Aya 23** 展示了其在 23 种语言方面的天赋，有望影响未来的对话（[Huggingface 上的 Aya 23](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF)）。

**GPU 阵营壮大但需要指南**：寻求提升技术水平的成员正在采用 **7900xt** 显卡。然而，关于有效环境设置的指导（例如在 Fedora 上将 RX 6600 显卡视为 gfx1030）仍然是稀缺资源。

**存储问题解决，寻求支持**：一位成员决定专门为 **LM Studio** 分配一个 M.2 SSD，这描绘了持续的硬件适配情况。另一方面，关于双显卡支持等 GPU 兼容性查询突显了社区对共享智慧的依赖。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 崛起**：用户观察到 Mojo nightly `2024.5.2305` 版本中的**编译错误**，并分享了诸如显式类型转换为 `Float64` 的解决方案。关于 Mojo 中以 null 结尾的字符串（null-terminated strings）的辩论引发了对性能的担忧，并参考 GitHub issues 和外部资源（如关于 UTF-8 字符串处理的 [PEP 686](https://peps.python.org/pep-0686/)）激发了对潜在变更的讨论。

- **语法变动**：在 Mojo 中，使用 `//` 替换推断参数（inferred parameters）的 `inferred` 关键字引起了褒贬不一的反应，突显了简洁性与清晰度之间的权衡。一项关于类 `f-string` 功能的提案鼓励了对 `Formatable` trait 的探索，为未来的贡献奠定了基础。

- **装饰器与数据类型讨论**：在 **Mojo** 频道中，讨论范围从在 struct 中使用 `@value` 装饰器（被认为对减少样板代码很有价值），到自定义位大小整数的可行性，以及用于优化内存使用的 **MLIR dialects**。关于 Mojo 中 FFT 实现的咨询突显了改进文档的需求。

- **结构化日志与 GitHub Issue 管理**：参与者建议为 **GitHub issues** 创建专门的频道，以改进社区内的跟踪。此外，随着用户解决由文档中错误使用 `**` 引起的混淆，文档中正确语法和符号的重要性变得显而易见，强调了保持一致性的必要。

- **社区与更新**：**Modular** 发布了一个关于社区会议的新视频，详情见其[公开议程](https://modul.ar/community-meeting-doc)，并分享了他们的每周简报 [Modverse Weekly - Issue 35](https://www.modular.com/newsletters/modverse-weekly-35)，让社区及时了解最新的更新和活动。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Pythia 的账本**：在讨论训练 **Pythia** 等模型的成本时，Stellaathena 估计**最大模型的账单约为 25 万美元**，并在计算中提到了效率和折扣后的 GPU 小时价格。

**成本效益报告征集审稿人**：一份即将发布的关于*前沿模型训练成本*的报告正在寻求同行评审；感兴趣的人员将评估 GPU 小时数以及 **A100 40GB** 等 GPU 类型的影响。

**LeanAttention 正在超越 FlashAttention？**：最近分享的一篇论文介绍了 **LeanAttention**，其性能可能优于 **FlashAttention**，引发了对其创新性的辩论。社区还开玩笑地谈论了提高模型 Benchmark 的非正统做法，幽默地指出：“秘密配方是犯罪。”

**可解释性的新前沿**：一篇新论文被指出为**可解释性（interpretability）**研究打开了大门，激发了人们对其对未来研究影响的好奇心。

**评估大型模型**：交流了技术技巧，例如在多节点 **SLURM** 集群上运行 **lm eval harness**，以及如何为评估设置 `num_fewshot` 等参数，并报告了围绕可重复性和计算节点访问互联网的挑战。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **模型偏好 YAML，引发 JSON 嫉妒**：工程师们根据经验注意到，AI 模型在处理 **YAML** 时比 **JSON** 更具优势，尽管开发工作更倾向于 JSON，这在讨论者中引发了技术好奇心和幽默感。

- **GPT-4o 与 DALL-E 3 的艺术协作**：对话显示，**GPT-4o** 正在增强对图像提示词（prompts）的理解，与单独使用 DALL-E 3 相比，与 **DALL-E 3** 配合使用时能产生更好的输出。这种协同作用说明了文本和图像模型之间不断演变的相互作用。

- **Playground 中的换行符导致格式困扰**：OpenAI playground 的换行符处理一直导致易用性问题，有报告称粘贴结果不一致。这个看似微小的技术故障引发了关于格式化和数据呈现的更广泛讨论。

- **Anthropic 的论文激发想法与推测**：社区讨论了 Anthropic 关于机械可解释性（mech interpretation）及其影响的论文，触及了 AI 如何根据训练数据进行拟人化，以意想不到的方式反映了禁闭（confinement）和人格（personas）等概念。随后进行了关于此类发现对未来 AI 发展影响的技术辩论。

- **提示词工程秘密与批评分享**：技术讨论包括提示词工程（prompt engineering）的策略，交流了关于系统提示词（system prompts）的实用建议，有些人认为系统提示词尚有欠缺。诸如模型从侧边栏消失以及“step-by-step”提示词的语义等问题被剖析，反映了对用户体验和 AI 交互细节的深入探讨。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**GPU 优化研讨会座无虚席**：GPU 优化研讨会获得了极高的参与度，拥有**超过 2400 多名注册者**，并由来自 **Sharan Chetlur** (NVIDIA)、**Phil Tillet** (OpenAI) 和 **William Malpica** (Voltron Data) 等专家的精彩分享。爱好者可以在[这里](https://lu.ma/1wu5ppl5)预约未来的互动，更多资源可在 [GitHub](https://github.com/mlops-discord/gpu-optimization-workshop) 上获取。

**破解 CUDA 困惑**：一位成员澄清说，由于其网格启动（grid launch）设置，**`__global__` CUDA 函数**不能同时是 `__host__`，并提出了一个不依赖 `threadIdx` 和 `blockIdx` 的 `__global__` 函数的理论效用。

**Triton 的棘手转换**：一位用户讨论了在使用 **triton+compile** 将 kernel 从 **FP32 转换为 FP6** 时出现的性能下降，推测这可能是 inplace 算子的潜在影响。

**AI 研究摘要引发讨论热潮**：每周 AI 研究亮点浮出水面，重点分析了 KAN、xLSTM 和 OpenAI 的 GPT-4 等作品。讨论延伸到了 KANs 由于基于激活的边缘计算（activation-based edge computation）而具有的计算密集特性。

**CUDA 的死胡同与 Vulkan 的尝试**：对话转向了贡献和编码问题，包括一位成员的 **flash-attention 仓库**停滞、7900xtx 与 3090 等 GPU 型号的基准测试，以及 Vulkan 在热传递模拟中表现不佳。

**LLM.C 稳步前进**：关于 llm.c 的交流非常活跃，成员们庆祝了 **HellaSwag 评估在 C 语言中**的集成，辩论了旨在提速的 **CUDA stream 优化**，并分享了在不中断训练的情况下扩展 batch size 的挑战。

请注意，由于未提供额外上下文，部分引用和项目链接已原样保留。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Llama 3 的量化难题**：技术爱好者正在讨论 **Llama 3** 模型具有挑战性的量化问题，指出由于模型对比特精度（bit accuracy）敏感，导致性能下降。
- **备受关注的模型**：一些工程师正将注意力转回 **Mistral 模型**以解决微调问题，而 **Aya 模型**（特别是发布在 [Hugging Face](https://huggingface.co/CohereForAI/aya-23-35B) 上的 35B 版本）因其架构和训练前景而引起了关注。
- **GPU 障碍**：AI 专家发现 **GPU 显存限制**是一个巨大的障碍，在 RTX 4090 等高容量显卡上进行微调时经常出现 `CUDA out of memory` 错误。他们正在研究 **QLoRA** 等替代方案。
- **发表成果**：社区成员关注到一篇关于**医学语言模型**的学术文章已发表，可通过此 [DOI](https://doi.org/10.1093/jamia/ocae120) 获取。
- **Colossus 故障排除**：成员们正在集思广益，探讨在 Colab 中使用提示词模板进行 **Llama-3-8B** 模型微调的多 GPU 设置，同时解决提示 "Current loss scale at minimum" 的混合精度（mixed precision）错误。为了更好地完成这些大规模计算任务，大家正在分享资源，包括 [Axolotl 数据集格式文档](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format)。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **数据集中的 NSFW 内容引发争论**：关于处理 **Common Crawl datasets** 挑战的技术讨论已经浮出水面，特别是针对 **NSFW 内容** 问题，并强调了 [cc2dataset](https://github.com/rom1504/cc2dataset/blob/main/cc2dataset/main.py#L81-L84) 中用于图像处理的代码修改。同时，辩论质疑了 **Hugging Face** 对可能包含敏感材料的数据集的托管政策，其自身的 [未过滤数据集发布](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 也受到了审查。

- **内容审核挑战与法律担忧**：LAION 社区讨论了数据集可访问性与审核之间的平衡，一些人强调了 **Hugging Face** 上 **投诉驱动 (complaint-driven)** 限制系统的便利性。关于动漫相关数据集的担忧以及它给用户识别 **色情内容 (pornographic content)** 带来的压力，引发了关于潜在法律后果的严肃讨论。

- **对 GPT4o 性能的不满**：用户对 GPT4o 表示不满，理由是 **自我污染 (self-contamination)** 问题，以及尽管在多模态 (multi-modal) 功能方面有所改进，但被认为未能达到 GPT4 设定的性能标准。

- **Transformer Circuits 和 Autoencoders 引起技术辩论**：要求 AI 系统透明度的呼声，特别是在 **Transformer Circuits Thread** 中，反映了 AI 工程师对模型可能影响社会规范的担忧。另外，一些用户剖析了 **MLPs** 和 **autoencoders** 之间的区别，指出了明确架构区分的重要性。

- **新研究揭晓**：Anthropic 关于 **Claude 3 Sonnet** 模型的最新见解引起了关注，揭示了金门大桥等概念的神经元激活以及有影响力的模型微调潜力，详细研究发表在 [Anthropic](https://www.anthropic.com/research/mapping-mind-language-model)。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**OpenAI 被指控 NDA 越权**：OpenAI 领导层声称对因不签署 NDA 而威胁前员工既定股权一事不知情，但 [带有领导层签名的文件](https://www.vox.com/future-perfect/351132/openai-vested-equity-nda-sam-altman-documents-employees) 显示事实并非如此。前员工面临七天窗口期的压力，要么签署，要么面临损失数百万美元的风险。

**模型性能头条**：**Gemini 1.5 Pro** 在生成模型 Reward Bench 排行榜上名列前茅，正如 [Jeff Dean 的推文](https://huggingface.co/spaces/allenai/reward-bench) 所示；同时，根据 [此公告](https://fxtwitter.com/maxwelltani/status/1793375460879110564)，**News Corp 和 OpenAI** 达成了一项为期多年的协议，允许 AI 利用 News Corp 的内容。

**闪电周边**：Nathan Lambert 的 Shopify 商店 [Interconnects](https://interconnects.myshopify.com/) 在对运营的轻松不确定中上线，并根据社区驱动进行了包容性产品调整；他保证了道德采购。

**AI 网红的兴起？**：据报道，TikTok 的青少年群体对机器人 (bots) 生成的内容产生共鸣，突显了 AI 创作内容走红的潜力。该平台作为 **Bella Poarch** 等人职业生涯的起点脱颖而出。

**Anthropic AI 的金门大桥焦点**：[Anthropic AI](https://x.com/anthropicai/status/1793741051867615494?s=46) 进行的一项奇特实验改变了 Claude AI 的焦点，使其痴迷于金门大桥，这在 AI 社区中引起了乐趣和兴趣。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**OpenRouter 为先进 AI 工具敞开大门**：OpenRouter 现在支持使用与 OpenAI 匹配的语法来调用 **Anthropic** 和 **Gemini** 模型，为 AI 从业者拓宽了图景。支持的 tool calls 和函数使用说明可以在 [文档](https://openrouter.ai/docs#tool-calls) 中找到。

**Lumimaid 70B 步入 AI 舞台**：NeverSleep 团队发布了专门针对角色扮演场景微调的 **Lumimaid 70B** 模型，详细信息可以从他们的 [公告页面](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b) 获取。

**召唤所有角色扮演玩家进入新的数字领域**：一款提供免费层级的新角色扮演应用已上线，它利用了 OpenRouter 多样化的 AI 角色，创作者热衷于通过 [RoleplayHub](https://www.roleplayhub.app/chat) 收集反馈。

**General 频道中技术故障与社区对话交织**：官方应用了软件补丁以修复 Llama-3 等模型的流式传输问题；Mistral-7B v0.3 的发布由于新的词汇表/tokenizer 引发了一些混乱——关于它应该是一个独立的模型路由还是直接的路由升级仍存在不确定性。同时，Cohere 的 Aya 计划引起了关注，该计划提供涵盖 101 种语言的多语言 AI 研究，点击 [此处](https://cohere.com/research/aya) 了解更多。

**AI 模型访问开启规模效应**：多个模型执行了大幅降价，包括 `nousresearch/nous-hermes-llama2-13b` 等模型诱人的 30% 折扣。这些降价正在激发开发者和爱好者的市场热情。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **用于 GenAI 预处理的 Batch Inference**：*Batch inference* 被强调为 **GenAI 应用**中数据预处理的关键技术，具有提高分析和查询效率的潜力。LlamaIndex 的集成以及关于该实践的更多细节可以在 [此处](https://t.co/vnuvvypZCz) 找到。

- **RAG 驱动的求职助手蓝图**：使用 **@gokoyeb**、**@MongoDB** 和 **@llama_index** 创建了一个 **RAG 驱动**的求职助手，展示了实时响应流式传输，教程可在 [此处](https://t.co/qsfx4TdvXz) 获取。

- **Nomic Embed 的本地化策略**：**Nomic Embed** 现在支持完全本地的 embeddings 以及动态推理，融合了本地和远程 embeddings 的优点，详见 [此处](https://t.co/mPFVQXk5tq)。

- **预留技术聚会席位**：有兴趣参加即将到来的**周二聚会**的工程师请注意，名额即将告罄，更多详情请访问 [此处](https://t.co/Nx4FiGB8pH)。

- **扩展 RAG 嵌入模型引发关注**：围绕大型 AI 模型在改进 **RAG embeddings** 方面的有效性展开了讨论，但尚未达成明确共识。关于 *ReAct 算法* 的参考以及使用 `alpha` 参数自定义相似度分数的建议可以在 **LlamaIndex 文档**中找到，这些话题的讨论还包括了详细文章和论文的链接。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Yi Tay 的播客错失良机**：社区希望在 Yi Tay 关于 Reka/Google 的播客中看到关于 **scaling laws** 的重点讨论，但由于播客是预录制的，这些见解未能包含在内。

- **Mistral v0.3 引发褒贬不一的反应**：**Mistral 7B v0.3 模型**已发布，具有 32K 扩展词汇表、新的 v3 tokenizer 和 function calling 功能等增强，引发了兴奋也带来了批评 [Mistral 的最新篇章](https://x.com/Gradio/status/1793367718835659243)。

- **关于开源 AI 的犀利观点**：一篇声称开源 AI 带来投资风险和国家安全担忧的争议性文章引发了辩论，反对者指责作者明显偏袒 OpenAI 且视角狭隘。

- **寻求通用的 Speech-to-Speech API**：社区讨论了针对 **OpenAI 尚未发布的 speech-to-speech API** 的变通方案，指向 **Pipecat 和 LiveKit** 作为当前的替代方案，且更倾向于 Pipecat。

- **RAG 落地实战**：成员们交流了 **Retrieval-Augmented Generation (RAG)** 的实际应用和挑战，特别提到了关于在医疗公司部署 RAG 的 [PyData Berlin 演讲](https://useml.net/posts/2024/05/22/rag-for-a-medical-company.html)。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **使用 VSCode 进行创新的 Prompt 管理**：工程师们计划使用 VSCode 管理 Prompt 以保持效率，包括为 **Gemini 1.5 Pro** 准备的近 **50 万 token 的系统提示词 (system prompts)**。这种创意受到了热烈欢迎，并征集了更多系统提示词的建议。

- **CLI 改进广受好评**：通过 [GitHub pull request](https://github.com/OpenInterpreter/open-interpreter/pull/1278) 引入的新终端选项 `--no_live_response` 因其解决终端 UI 问题的潜力而受到好评。Steve235lab 的贡献被赞誉为一项显著的改进。

- **关注组件拆解与替换芯片**：成员们讨论了 **Apple AirPods Pro** 的拆解，以及在 Atom Echo 中使用 **ESP32 pico 芯片** 进行替代项目的方案，并指出了必要的固件重刷 (reflashing)。ChatGPT 提供的技术数据表 (datasheets) 等补充信息也被认为非常有益。

- **工具赞誉：M5Stack Flow UI 软件**：[M5Stack Flow UI 软件](https://flow.m5stack.com) 因支持多种编程语言以及将 Python 脚本转换为运行 LLM 客户端（如 OpenAI）的潜力而受到称赞，展示了硬件与 AI 驱动应用之间的灵活集成。

- **跳过 macOS ChatGPT 等候名单**：分享了一个来自 [@testingcatalog](https://x.com/testingcatalog/status/1793347117458636981) 的可能存在争议的 macOS ChatGPT 应用等候名单绕过方法，通过在登录过程中精确把握时机来实现“作弊”。这些信息对于寻求理解或利用用户行为及应用漏洞的软件工程师可能具有参考意义。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**挑战泰勒级数拆解**：成员们质疑了**泰勒级数在近似计算中的有效性**，指出它们仅在参考点附近准确。有人强调，范围缩减 (range reduction) 可能不是实现完美精度的最佳路径，而区间划分 (interval partitioning) 可能会提供更好的解决方案。

**重新思考范围缩减**：小组辩论了**范围缩减技术**的使用，建议采用缩减至 **[0, pi/4]** 等替代方案，并参考了 **IBM 的方法**，将其作为在其 [实现](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/s_sin.c;hb=HEAD) 中发现的区间划分的实际案例。

**IBM 的见解**：提到了一份 IBM 源文件，建议通过将 fmod 视为整数来解决范围缩减问题，可在此处查看 [链接](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/dbl-64/branred.c;hb=HEAD)。

**冷静思考数学复杂性**：大家一致认为，实现完美精度的计算非常复杂，尤其是对于大数，尽管通常并不慢——这是一种对所涉及的科学复杂性的钦佩与接受。

**ShapeTracker 中的形状变换**：小组探讨了 *ShapeTracker* 的局限性，结论是某些操作序列（如 `permute` 后接 `reshape`）会导致多个视图 (views)，从而在有效链接移动操作时带来挑战。讨论了张量掩码 (tensor masking) 的效用，重点强调了其在张量切片 (slicing) 和填充 (padding) 中的作用。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **热烈欢迎全球创意人士**：欢迎新成员加入的友好互动，包括一位来自台湾的 **UI Designer**。
- **引导 AI 交互**：一位成员为与 AI 交互提供了明确指引，提到了特定频道和 `@coral` 句柄以寻求帮助。
- **Cohere 扩大语言 AI 覆盖范围**：Cohere 宣布推出 **Aya 23 模型**，标志着新的进展，提供拥有 [80 亿和 350 亿参数](https://huggingface.co/CohereForAI/aya-23-35B) 的工具，并宣称支持涵盖 23 种语言的语言范围。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **GraphRAG 在图建模信息方面获得关注**：成员们讨论认为，当源数据天然具有图结构时，**GraphRAG** 表现出色，尽管对于其他数据格式它可能不是最佳选择。
  
- **PySpark 加速 Embedding 转换**：AI 工程师们正在尝试使用 **PySpark pandas UDF**，以潜在地提高 Embedding 处理的效率。

- **Pinecone 的持久化挑战**：社区内的一个共同挑战集中在 **persistence handling**（持久化处理）与 Pinecone 中频繁创建实例的效率低下问题上，并对 *pickle* 等主流解决方案表示不满。

- **API 与 Instruction Tuning 成为焦点**：即将于 2024 年 5 月 23 日举行的活动“如何使用 LangSmith 开发用于生成式 AI 药物研发生产的 API”，以及一段新的 [YouTube 视频](https://youtu.be/jddSbTLw0gc?si=spk1NEQMbr0iG1vt) 解释了 Instruction Tuning 对于增强 LLM 遵循人类指令的好处。

- **代码修改与检索器规划**：工程师们目前正在寻求高效的检索器来规划代码变更，以及防止 LLM 在建议修改时削减现有代码的技术。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Mistral 词汇量与功能增强**：[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) 的最新迭代版本现在拥有 **32768 个 token 的扩展词汇量**、**v3 Tokenizer 支持**以及 function calling 能力，通过 `mistral_inference` 即可轻松安装。

- **Mistral 7B 增强版获得社区认可**：[Mistral-7B instruct 版本](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) 的发布得到了 Eldar Kurtic 的认可，并暗示会有更多改进，详见[最近的推文](https://twitter.com/_EldarKurtic/status/1793407795909599325?t=zhtA3A5nq23HfUBkt441mQ&s=19)。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **GaLore 与 InRank 取得新突破**：与 Jiawei Zhao 的交流环节深入探讨了 **Gradient Low-Rank Projection (GaLore)** 和 **Incremental Low-Rank Learning (InRank)**，这些技术可以减少内存使用并增强大规模模型训练性能。

- **活动同步困扰**：有人询问如何将活动日历与 Google Calendar 集成，强调了跟踪即将开始的讨论以避免错过的需求。

- **ImageMAE 图像识别标志着可扩展性的飞跃**：分享了 ImageMAE 论文，提出了一种使用 masked autoencoders 进行计算机视觉的可扩展自监督学习方法，其中原生的 ViT-Huge 模型达到了 87.8% 的惊人结果。

- **社区氛围高涨**：一位成员表达了对该频道的赞赏，认为它是 AI 领域分享和学习的宝贵资产。



> 完整的频道细分内容已针对邮件进行截断。
> 
> 如果你想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}