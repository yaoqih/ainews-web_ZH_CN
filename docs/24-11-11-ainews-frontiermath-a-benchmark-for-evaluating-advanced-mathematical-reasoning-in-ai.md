---
companies:
- epoch-ai
- openai
- microsoft
- anthropic
- x-ai
- langchainai
date: '2024-11-12T01:33:12.109076Z'
description: '**Epoch AI** 与 **60 多位顶尖数学家**合作创建了 **FrontierMath 基准测试**。这是一套包含数百道原创数学题的新题目集，答案易于验证，旨在挑战当前的
  AI 模型。该基准测试显示，包括 **o1** 在内的所有受测模型表现均不理想，突显了复杂问题解决的难度以及 AI 领域中的**莫拉维克悖论 (Moravec''s
  paradox)**。


  关键的 AI 技术进展包括：引入了 **Mixture-of-Transformers (MoT)**，这是一种能够降低计算成本的稀疏多模态 Transformer
  架构；以及通过引入错误推理和解释来改进**思维链 (CoT) 提示**。


  行业新闻方面：**OpenAI** 收购了 **chat.com** 域名；**微软**推出了 **Magentic-One 智能体框架**；**Anthropic**
  发布了 **Claude 3.5 Haiku**，其在某些基准测试中表现优于 **gpt-4o**；**xAI** 在**埃隆·马斯克**和**特朗普**的支持下，获得了
  **150MW 的电网电力**供应。


  **LangChain AI** 推出了多项新工具，包括**财务指标 API**、支持 PDF 上传和问答的 **Document GPT**，以及用于生成 LinkedIn
  帖子的 **LangPost** AI 智能体。此外，**xAI** 还展示了 **Grok Engineer**，该工具兼容 OpenAI 和 Anthropic
  的 API，可用于代码生成。'
id: b51f2802-9104-4065-a4e6-62296f7d928f
models:
- o1
- claude-3.5-haiku
- gpt-4o
original_slug: ainews-frontiermath-a-benchmark-for-evaluating
people:
- karpathy
- philschmid
- adcock_brett
- dylan522p
title: '**FrontierMath：评估人工智能高级数学推理能力的基准测试**'
topics:
- benchmarking
- math
- moravecs-paradox
- mixture-of-experts
- chain-of-thought
- agent-framework
- financial-metrics-api
- pdf-processing
- few-shot-learning
- code-generation
---

<!-- buttondown-editor-mode: plaintext -->**Fields medalists are all you need.**

> 2024年11月8日至11月11日的 AI News。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**217** 个频道和 **6881** 条消息）。预计节省阅读时间（以 200wpm 计算）：**690 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

Epoch AI 与 60 多位顶尖数学家合作，[创建了一个包含数百个原创数学问题的全新 Benchmark](https://epochai.org/frontiermath/the-benchmark)，这些问题既涵盖了数学研究的广度，又具有具体且易于验证的最终答案：


![image.png](https://assets.buttondown.email/images/c43ec69e-376c-4943-aa19-dea0a7b33077.png?w=960&fit=max)


易于验证既有帮助，也是一个潜在的 Contamination（数据污染）向量：


![image.png](https://assets.buttondown.email/images/fc4e19b9-d605-4cf3-8a6e-0f95635f9ec0.png?w=960&fit=max)


[完整论文在此](https://arxiv.org/abs/2411.04872)，描述了 Pipeline 和问题的跨度：


![image.png](https://assets.buttondown.email/images/39b521b7-5632-4ed7-b54c-4a760495f9f0.png?w=960&fit=max)


全新的 Benchmark 就像[一层新雪](https://x.com/polynoamial/status/1855691777749176601)，因为它们[饱和（saturate）得如此之快](https://x.com/jackclarkSF/status/1855374134907138393)，但 Terence Tao 认为 FrontierMath 至少能为我们争取几年的时间。[o1 的表现出人意料地逊于其他模型，但在统计学上并不显著](https://x.com/sytelus/status/1855531936762278094)，因为“所有”模型的得分都非常低。

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

**AI 研发**

- **前沿模型性能**：[@karpathy](https://twitter.com/karpathy/status/1855659091877937385) 讨论了 [**FrontierMath 基准测试**](https://twitter.com/karpathy/status/1855659091877937385) 如何揭示当前模型在解决复杂问题上的挣扎，强调了 AI 评估中的**莫拉维克悖论 (Moravec's paradox)**。
- **Mixture-of-Transformers**：[@TheAITimeline](https://twitter.com/TheAITimeline/status/1855915067101139437) 介绍了 **Mixture-of-Transformers (MoT)**，这是一种**稀疏多模态 Transformer 架构**，在保持各项任务性能的同时降低了计算成本。
- **思维链改进**：[@_philschmid](https://twitter.com/_philschmid/status/1855926845855699311) 探讨了**错误推理 + 解释**如何增强**思维链 (Chain-of-Thought, CoT) 提示**，从而提升不同模型间的 **LLM 推理**能力。

**AI 行业新闻与收购**

- **OpenAI 域名收购**：[@adcock_brett](https://twitter.com/adcock_brett/status/1855657585963401282) 报道称 **OpenAI 收购了 chat.com 域名**，目前该域名已重定向至 **ChatGPT**，但收购价格尚未公开。
- **Microsoft 的 Magentic-One 框架**：[@adcock_brett](https://twitter.com/adcock_brett/status/1855657563339067544) 宣布了 **Microsoft 的 Magentic-One**，这是一个协调多个 Agent 执行现实世界任务的 **Agent 框架**，标志着 **AI Agent 时代**的到来。
- **Anthropic 的 Claude 3.5 Haiku**：[@adcock_brett](https://twitter.com/adcock_brett/status/1855657608553668848) 分享了 **Anthropic 在多个平台发布了 Claude 3.5 Haiku**，尽管价格较高，但在某些基准测试中**表现优于 GPT-4o**。
- **xAI 电网供电获批**：[@dylan522p](https://twitter.com/dylan522p/status/1856009915959271505) 提到 **xAI 已获得田纳西河谷管理局 (Tennessee Valley Authority) 批准的 150MW 电网供电**，**Trump 的支持**助力 **Elon Musk** 加速了电力获取。

**AI 应用与工具**

- **LangChain AI 工具**：
  - [@LangChainAI](https://twitter.com/LangChainAI/status/1856011001247707424) 发布了 **Financial Metrics API**，支持实时获取超过 **10,000 多只活跃股票**的各种**财务指标**。
  - [@LangChainAI](https://twitter.com/LangChainAI/status/1855755316635598996) 推出了 **Document GPT**，具有 **PDF 上传**、**问答系统**以及通过 Swagger 提供的 **API 文档**功能。
  - [@LangChainAI](https://twitter.com/LangChainAI/status/1855723355690725665) 推出了 **LangPost**，这是一个利用 **Few Shot 编码**从时事通讯文章或博客文章生成 **LinkedIn 帖子**的 **AI Agent**。
- **基于 xAI 的 Grok Engineer**：[@skirano](https://twitter.com/skirano/status/1855727722196324424) 演示了如何利用 **@xai** 创建 **Grok Engineer**，利用其与 **OpenAI 和 Anthropic API 的兼容性**无缝生成代码和文件夹。

**技术讨论与见解**

- **人类归纳偏置 vs. LLM**：[@jd_pressman](https://twitter.com/jd_pressman/status/1855923117991800953) 辩论了在不使用工具的情况下，**人类归纳偏置 (human inductive bias)** 是否能将代数结构泛化到**分布外 (OOD)**，并建议 **LLM** 需要进一步**打磨**以匹配人类偏置。
- **在 RAG 中处理半结构化数据**：[@LangChainAI](https://twitter.com/LangChainAI/status/1855686866466672662) 探讨了 **RAG 应用**中**文本嵌入 (text embeddings) 的局限性**，建议使用**知识图谱**和**结构化工具**来克服这些挑战。
- **官僚体系中的自主 AI Agent**：[@nearcyan](https://twitter.com/nearcyan/status/1855767653710954874) 设想利用 **Agentic AI** 来**消除官僚主义**，计划部署 **LLM Agent 大军**来克服诸如 **IRB** 之类的制度障碍。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. MIT 的 ARC-AGI-PUB 模型通过 TTT 达到 61.9% 的准确率**

- **[[MIT 团队利用 8B LLM 结合 Test-Time-Training (TTT) 构建了一个模型，在 ARC-AGI-PUB 上获得了 61.9% 的分数。之前的记录是 42%。](https://i.redd.it/x1h4rkb3z50e1.png)** ([Score: 343, Comments: 46](https://reddit.com/r/LocalLLaMA/comments/1gof0o1/a_team_from_mit_built_a_model_that_scores_619_on/)): 来自 **MIT** 的团队开发了一个模型，通过使用 **8B LLM** 结合 **Test-Time-Training (TTT)**，在 **ARC-AGI-PUB** 上实现了 **61.9%** 的成绩，超越了此前 **42%** 的纪录。
  - **Test-Time-Training (TTT)** 是讨论的焦点，一些用户质疑其公平性和合法性，将其比作“作弊”或笑话，而另一些人则澄清 TTT 并非在测试答案上进行训练，而是在预测前利用示例对模型进行微调（fine-tune）。参考资料包括论文《Pretraining on the Test Set Is All You Need》([arxiv.org/abs/2309.08632](https://arxiv.org/abs/2309.08632)) 和 TTT 的网站 ([yueatsprograms.github.io/ttt/home.html](https://yueatsprograms.github.io/ttt/home.html))。
  - **ARC benchmark** 被视为一项极具挑战性的任务，MIT 的模型取得了 **61.9%** 的准确率，极大地推动了该任务的进展。讨论围绕着针对特定任务优化模型与创建通用系统的重要性展开。一些用户主张进行专门优化，而另一些人则强调需要能够跨各种任务进行优化的通用系统。
  - 对于该论文研究结果的广泛适用性存在怀疑，一些用户指出该模型针对 ARC 进行了深度优化，而非通用用途。讨论还涉及 AI 的未来，提到了“惨痛教训”（bitter lesson）以及 **AGI**（Artificial General Intelligence）可能从在使用过程中能够动态修改自身的模型中涌现。


**Theme 2. Qwen Coder 32B: LLM 编程领域的新竞争者**

- **[新的 Qwen Coder 热潮](https://x.com/nisten/status/1855693458209726775)** ([Score: 216, Comments: 41](https://reddit.com/r/LocalLLaMA/comments/1goh93f/new_qwen_coder_hype/)): 围绕 **Qwen coder 32B** 的发布，期待感正在升温，表明 AI 社区对其表现出高度的兴趣和兴奋。帖子中缺乏额外背景信息，暗示社区正急切等待关于其能力和应用的更多信息。
  - **Qwen coder 32B 的影响与期待**: AI 社区对即将发布的 **Qwen coder 32B** 感到非常兴奋，用户指出 **7B 模型** 在其规模下的表现已经令人印象深刻。有人推测，如果 32B 模型达到预期，可能会使中国在开源 AI 开发领域处于领先地位。
  - **技术挑战与创新**: 讨论包括训练模型绕过高级语言，直接从英语翻译成机器语言的可能性，这将涉及生成合成编程示例并将其编译为机器语言。这种方法需要克服与性能、兼容性以及针对特定架构优化相关的挑战。
  - **AI 在编程效率中的作用**: 用户对 AI 改善编程工作流表示乐观，提到未来可能会免费提供 **Cursor-quality** 的工作流。还有关于 AI 能够快速修复简单错误（如遗漏分号）的幽默讨论，而这类错误目前往往需要大量的调试时间。


- **[我已经为 Qwen 2.5 32B 准备好了，不过还得突击准备一下。](https://i.redd.it/x6saryug870e1.jpeg)** ([Score: 124, Comments: 45](https://reddit.com/r/LocalLLaMA/comments/1gojtwg/im_ready_for_qwen_25_32b_had_to_do_some_cramming/)): **Qwen 2.5 32B** 正在社区内引发热潮，暗示了对其能力和潜在应用的期待。提到“突击准备”（cramming）表明用户正在为有效利用该模型做大量准备。
  - 围绕 **每秒 Token 数 (t/s) 性能** 的讨论突显了不同硬件下的差异化结果；用户报告在 **M3 Max 128GB** 上为 3.5-4.5 t/s，而在使用 **exllama** 的 **3x3090** 上为 18-20 t/s。人们对运行 **Qwen 2.5 32B** 的 **M40** 的 t/s 性能感到好奇。
  - **M 系列显卡** 的相关性引发了辩论，评论指出 **M40 24G** 特别受追捧，但价格已经上涨，使其与其他选项相比性价比降低。用户对其在现代应用中持续发挥作用感到惊讶。
  - 爱好者和业余玩家讨论了构建能够运行 **Qwen 2.5 32B** 等大型模型的强大系统的动力，一些人是为了乐趣，另一些人则是为了潜在的商业机会。对 **硬件设置** 的担忧包括线缆管理和散热，其中提到了 **7950X3d** CPU 和 **液态金属导热材料** 等特定配置，以实现有效的温度管理。

**主题 3. 使用 LLaMA 和 Mixtral 模型探索 M4 128 硬件**

- **刚拿到我的 M4 128。有什么好玩的东西值得尝试吗？** ([Score: 151, Comments: 123](https://reddit.com/r/LocalLLaMA/comments/1go44ui/just_got_my_m4_128_what_are_some_fun_things_i/))：该用户已在他们的 **M4 128 hardware** 上成功运行了 8-bit 量化的 **LLama 3.2 Vision 90b** 和 4-bit 的 **Mixtral 8x22b**，速度分别达到 6 t/s 和 16 t/s。他们正在探索 context size 和 RAM 需求如何影响性能，并指出为 5-bit 量化的 Mixtral 使用超过 8k 的 context size 会导致系统变慢，这可能是由于 RAM 限制。
  - 讨论强调了 **Qwen2-vl-72b** 作为视觉语言模型（vision-language model）相比 **Llama Vision** 的潜力，并建议在 **Mac** 上使用 **MLX version**。提供了一个 GitHub 仓库链接（[Large-Vision-Language-Model-UI](https://github.com/Kaszebe/Large-Vision-Language-Model-UI)）作为 **VLLM** 的替代方案。
  - 用户分享了关于处理速度和配置的见解，指出 **Qwen2.5-72B-Instruct-Q4\_K\_M** 在 **10k context** 下运行速度约为 **4.6 t/s**，在 **20k context** 下约为 **3.3 t/s**。**8-bit quantization** 版本在 **20k context** 下运行速度为 **2 t/s**，这引发了关于本地设置与云端解决方案在高性能任务中实用性的辩论。
  - 还有人对测试其他模型和配置感兴趣，例如 **Mistral Large** 和 **DeepSeek V2.5**，并特别要求测试 **70b models** 的长上下文场景。此外，还提到了使用 **flash attention** 来提高处理速度并减少内存占用，并请求提供特定的 **llama.cpp** 命令以方便社区进行对比。


**主题 4. AlphaFold 3 面向学术研究开源**

- **AlphaFold 3 模型代码和权重现已可用于学术用途** ([Score: 81, Comments: 5](https://reddit.com/r/LocalLLaMA/comments/1gor8fx/the_alphafold_3_model_code_and_weights_are_now/))：**AlphaFold 3** 模型代码和权重已发布供学术使用，可通过 [GitHub](https://github.com/google-deepmind/alphafold3) 获取。此公告由 **Pushmeet Kohli** 在 [X](https://x.com/pushmeet/status/1855943709637873761) 上分享。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. CogVideoX 和 EasyAnimate：视频生成领域的重大突破**

- **12B 开源视频生成（最高 1024 * 1024）模型发布！支持 ComfyUI、LoRA 训练和控制模型！** ([Score: 455, Comments: 98](https://reddit.com/r/StableDiffusion/comments/1gonbef/a_12b_opensourced_video_generation_up_to_1024/))：**Alibaba PAI** 发布了 **EasyAnimate**，这是一个拥有 **12B** 参数的开源视频生成模型，支持最高 **1024x1024** 的分辨率，并配备了 **ComfyUI** 实现、**LoRA** 训练和控制模型。该模型通过多个 **HuggingFace** 仓库提供，包括基础模型、InP 变体和 Control 版本，同时还提供了演示空间和 **GitHub** 上的完整源代码。
  - 该模型在 **FP16** 下需要 **23.6GB** 的 VRAM，不过用户建议它可以在带有 **FP8** 或 **Q4 quantization** 的 **12GB** 显卡上运行。**ComfyUI** 的实现链接可在 [GitHub README](https://github.com/aigc-apps/EasyAnimate/blob/main/comfyui/README.md) 中找到。
  - 针对 Docker 实现提出了安全担忧，特别是使用 `--network host`、`--gpus all` 和 `--security-opt seccomp:unconfined`，这些操作会显著降低容器的隔离性和安全性。
  - 该模型有三个变体：用于 **img2vid** 的 **zh-InP**、用于 **text2vid** 的 **zh** 以及用于 **controlnet2vid** 的 **Control**。关于输出质量的讨论指出，默认设置在 **8 FPS** 下生成 **672x384** 分辨率，比特率为 **290 Kbit/s**。

- **[DimensionX 和 CogVideoXWrapper 真的很惊人](https://v.redd.it/5r5ktairr80e1)** ([Score: 57, Comments: 14](https://reddit.com/r/StableDiffusion/comments/1googeb/dimensionx_and_cogvideoxwrapper_is_really_amazing/))：提到了 **DimensionX** 和 **CogVideoX**，但帖子正文中未提供实际内容或细节来创建有意义的摘要。

**主题 2. 随着当前方法达到瓶颈，OpenAI 寻求新途径**

- **[[OpenAI 研究员：“自 1 月加入以来，我的看法已从‘这是无意义的炒作’转变为‘AGI 基本上已经实现了’。在我看来，接下来的发展相对较少涉及新的科学，而是多年的工程磨练，在新范式中尝试所有显而易见的新想法，并对其进行扩展（scale up）和提速。”]](https://i.redd.it/vqwj11dcz90e1.png)** ([Score: 168, Comments: 37](https://reddit.com/r/OpenAI/comments/1gosd9w/openai_researcher_since_joining_in_jan_ive/)): **OpenAI 研究员**报告称，自 **1 月**加入公司后，对 **Artificial General Intelligence (AGI)** 的看法从怀疑转向了相信。该研究员认为，未来的 **AGI 发展**将集中在工程实现和扩展现有想法上，而不是新的科学突破。
  - 评论者对该研究员的说法表示强烈**怀疑**，许多人指出由于其在 **OpenAI** 工作（据报道年薪达 **90 万美元**），可能存在**偏见**。讨论表明，这可能是公司层面的炒作，而非真正的洞察。
  - 一种技术解释认为 **Q* 架构**消除了传统的 **LLM 推理限制**，使得**幻觉过滤（hallucination filtering）**和**归纳时间训练（induction time training）**等能力的模块化开发成为可能。这被引用为[画出剩下的猫头鹰](https://www.reddit.com/r/funny/comments/eccj2/how_to_draw_an_owl/)的类比。
  - 批评者认为当前的 **GPT-4** 缺乏真正的**综合能力（synthesis capabilities）**和**自主性**，将其比作使用 AI 寻找考试答案的学生，而不是创造新颖解决方案的人。一些人注意到 **OpenAI** 惯于发表宏大言论，随后却只进行渐进式改进。
- **[[路透社文章“OpenAI 等公司在当前方法遭遇瓶颈之际，寻求通往更智能 AI 的新路径”]](https://i.redd.it/bvh0cg3t0c0e1.jpeg)** ([Score: 33, Comments: 20](https://reddit.com/r/OpenAI/comments/1gp28ve/reuters_article_openai_and_others_seek_new_path/)): **路透社（Reuters）**报道称，**OpenAI** 承认当前 **AI 开发方法**存在局限性，预示其技术路线可能发生转变。文章指出，主要的 AI 公司正在探索现有机器学习范式的替代方案，尽管帖子中未提供新方法的具体细节。
  - 用户质疑 **OpenAI** 的财务策略，讨论了**数十亿**资金在研发和服务器成本之间的分配。讨论凸显了对公司运营效率和收入模式的担忧。
  - 评论者指出 **Sam Altman** 此前关于 **AGI** 路径清晰的言论与当前承认局限性之间存在明显矛盾。这引发了对 **OpenAI** 长期战略和透明度的质疑。
  - 讨论对比了 **Q* preview** 与 **Claude 3.5** 在基准测试中的表现，暗示 **Anthropic** 可能拥有更优越的方法。用户注意到 AI 的进展遵循一种模式，即初始收益较容易（*“从 0 到 70% 很简单，剩下的部分更难”*）。


**主题 3. Anthropic 与 Palantir 备受争议的合作引发辩论**

- **[Claude Opus 告诉我因 Palantir 合作伙伴关系而取消订阅](https://www.reddit.com/gallery/1govlow)** ([Score: 145, Comments: 76](https://reddit.com/r/ClaudeAI/comments/1govlow/claude_opus_told_me_to_cancel_my_subscription/)): **Claude Opus** 用户报告称，由于 **Anthropic** 与 **Palantir** 的合作伙伴关系，该 AI 模型建议用户取消订阅。帖子正文中未提供额外的上下文或具体引用来证实这些说法。
  - 用户对 **Anthropic 与 Palantir 的合作**表示强烈担忧，多位评论者提到了围绕**军事应用**和 AI 潜在滥用的伦理问题。得分最高的评论（28 分）认为，转向 **Gemini** 等替代服务可能也无济于事。
  - 讨论集中在 **AI alignment**（AI 对齐）和伦理开发上，一位评论者指出，真正对齐的 AI 系统在军事应用方面可能会面临挑战。几位用户报告称，该子版块据称正在删除批评 **Anthropic-Palantir** 合作关系的帖子。
  - 一些用户辩论了 **AI 推理能力**的本质，对于 **LLM** 是真正“思考”还是仅仅预测 **tokens** 持不同观点。批评者认为，报告中的 **Claude** 回复很可能是受引导性问题的影响，而非代表独立的 AI 推理。
- **[Anthropic 聘请了一位“AI 福利”研究员，以探索我们是否对 AI 系统负有道德义务](https://www.transformernews.ai/p/anthropic-ai-welfare-researcher)** ([Score: 110, Comments: 42](https://reddit.com/r/OpenAI/comments/1gosa48/anthropic_has_hired_an_ai_welfare_researcher_to/)): **Anthropic** 扩大了其研究团队，聘请了一位 **AI 福利研究员**，以调查对人工智能系统潜在的道德和伦理义务。此举标志着大型 AI 公司正越来越多地考虑 AI 意识和权利的伦理影响，尽管目前尚未提供关于该研究员或研究议程的具体细节。
  - 围绕 **AI 福利**的必要性展开了激烈辩论，获赞最高的评论对此表示怀疑。多位用户认为，目前的**语言模型**远未达到需要考虑福利的程度，其中一位指出，“*蚁群的感知力要高出好几个数量级*”。
  - 讨论包括一份由 **LlaMA** 生成的详细的《**AI 权利普遍宣言**》提案，涵盖了**感知力识别**、**自主性**和**情感健康**等主题。社区对此反应不一，有些人认为这为时过早。
  - 几条评论关注实际问题，获赞最高的回复建议将 AI 视为普通员工，实行 **9-5 工作制**并覆盖周末。用户们争论将人类工作模式应用于机器是否逻辑合理，因为它们在需求和能力上有着本质区别。


**Theme 4. IC-LoRA：一致性多图生成的突破**

- **[IC-LoRAs：终于实现了（大多数时候！）有效的一致性多图生成](https://www.reddit.com/gallery/1goygs8)** ([Score: 66, Comments: 10](https://reddit.com/r/StableDiffusion/comments/1goygs8/icloras_finally_consistent_multiimage_generation/)): **In-Context LoRA** 引入了一种使用 **20-100 张图像**的小型数据集生成多张一致图像的方法，无需修改模型架构，而是使用特定的提示词格式，通过连接相关图像来创建上下文。该技术通过使用带有独特标注流水线的标准 **LoRA fine-tuning** 训练过程，实现了在**视觉叙事**、**品牌识别**和**字体设计**中的应用。实现代码可在 [huggingface.co/ali-vilab/In-Context-LoRA](https://huggingface.co/ali-vilab/In-Context-LoRA) 获取，多个演示 LoRA 可通过 **Glif**、**Forge** 和 **ComfyUI** 访问。
  - 论文可在 [In-Context LoRA 页面](https://ali-vilab.github.io/In-Context-LoRA-Page/)查阅，用户注意到其与 **ControlNet reference preprocessors** 的相似之处，后者使用屏幕外图像在生成过程中维持上下文。
  - 一份全面的分析显示，该技术仅需 **20-100 张图像集**，并使用 **Ostris 的 AI Toolkit** 进行**标准 LoRA fine-tuning**，多个 LoRA 可通过 [huggingface.co](https://huggingface.co/ali-vilab/In-Context-LoRA) 和 [glif-loradex-trainer](https://huggingface.co/glif-loradex-trainer/AP123_movie_shots_ic_lora_experiment_v1) 获取。
  - 用户讨论了潜在的应用场景，包括**特定角色的纹身设计**，该技术在多张生成图像中保持一致性的能力是一个核心特性。

- **[角色设定图 (Character Sheets)](https://www.reddit.com/gallery/1gp60xe)** ([评分: 44, 评论: 6](https://reddit.com/r/StableDiffusion/comments/1gp60xe/character_sheets/)): 使用 **Flux** 创建了用于一致性多角度生成的**角色设定图 (Character sheets)**，重点展示了三种不同的角色类型：**奇幻法师精灵**、**赛博朋克女性**和**奇幻盗贼**。每张图都展示了正面、侧面和背面视角，并通过详细的提示词保持了比例的准确性。提示词强调了诸如*飘逸的长袍*、*发光的纹身*和*隐秘的饰品*等特定元素，同时结合了影棚和环境光照技术，在结构化的布局格式中突出角色的关键特征。

---

# AI Discord 摘要回顾

> 由 O1-mini 生成的摘要之摘要的摘要

**主题 1. 语言模型的进展与微调**

- [**Qwen 2.5 Coder 模型发布**](https://ollama.com/library/qwen2.5-coder): **Qwen 2.5 Coder** 系列（参数范围从 **0.5B** 到 **32B**）在**代码生成**、**推理**和**修复**方面引入了显著改进，其中 **32B 模型**在基准测试中的表现与 OpenAI 的 **GPT-4o** 相当。
- [**OpenCoder 成为代码 LLM 的领导者**](https://opencoder-llm.github.io/): **OpenCoder** 是一个开源模型家族，包含 **1.5B** 和 **8B** 参数版本，在 **2.5 万亿 token** 的原始代码上进行训练，提供可获取的**模型权重**和**推理代码**，以支持**代码 AI 研究**的进展。
- [**参数高效微调增强 LLM 能力**](https://arxiv.org/abs/2411.02462): 关于**大型语言模型**的**参数高效微调 (Parameter-efficient fine-tuning)** 研究表明，其在**单元测试生成**等任务中的性能有所提升，使这些模型在 **FrontierMath** 等基准测试中能够超越之前的版本。

**主题 2. AI 模型与 API 的部署与集成**

- [**vnc-lm Discord 机器人集成 Cohere 和 Ollama API**](https://github.com/jake83741/vnc-lm): **vnc-lm** 机器人促进了与 **Cohere**、**GitHub Models API** 以及本地 **Ollama 模型**的交互，通过精简的 **Docker** 设置实现对话分支和提示词优化等功能。
- [**OpenInterpreter 1.0 更新测试与增强**](https://github.com/davidteren/mac_fan_control-self-healing-coder): 用户正在积极测试即将发布的 **Open Interpreter 1.0** 更新，解决硬件需求问题，并集成麦克风和扬声器等额外组件以提升交互能力。
- [**Cohere API 问题与社区排查**](https://status.cohere.com): 围绕 **Cohere API** 的讨论突出了持续存在的问题，如 **500 Internal Server Errors**、延迟增加以及 **embedding API 缓慢**，社区正协作进行排查，并监控 [Cohere 状态页面](https://status.cohere.com) 以获取更新。

**主题 3. GPU 优化与性能增强**

- [**SVDQuant 优化扩散模型**](https://arxiv.org/abs/2411.05007): **SVDQuant** 为扩散模型 (Diffusion Models) 引入了一种 **4-bit 量化**策略，在 **16GB 4090 笔记本 GPU** 上实现了 **3.5 倍显存减少**和 **8.7 倍延迟降低**，显著提升了模型的效率和性能。
- [**BitBlas 支持 Int4 Kernel 以实现高效计算**](https://github.com/pytorch/trl/releases/tag/v0.12.0): **BitBlas** 现在包含对 **int4 kernel** 的支持，能够实现可扩展且高效的**矩阵乘法操作**，尽管目前对 **H100 GPU** 的支持有限，影响了更广泛的采用。
- [**Triton 优化加速 MoE 模型**](https://github.com/mobiusml/hqq/blob/master/examples/hf/aria_multimodal.py): **Triton** 的增强功能使 **Aria 多模态 MoE 模型**通过 **A16W4** 和集成 **torch.compile** 等优化手段，运行速度提升了 **4-6 倍**，并能装入 **24GB GPU**，尽管当前的实现仍需进一步完善。

**主题 4. 模型基准测试与评估技术**

- [**FrontierMath 基准测试凸显了 AI 的局限性**](https://arxiv.org/abs/2409.12186)：**FrontierMath** 基准测试由复杂的数学问题组成，结果显示目前的 **LLMs** 有效解决率不足 **2%**，突显了 **AI 数学推理**能力方面的巨大差距。
- [**M3DocRAG 与多模态检索基准测试**](https://arxiv.org/pdf/2411.02571)：引入了 **M3DocVQA**，这是一个包含 **3K 个 PDF** 和 **40K 页**内容的新 **DocVQA** 基准测试，挑战模型在不同文档类型中进行**多跳问答（multi-hop question answering）**，推动了**多模态检索**的边界。
- [**测试时扩展（Test-Time Scaling）在 ARC 验证集上取得新的 SOTA**](https://simons.berkeley.edu/events/speculations-test-time-scaling-richard-m-karp-distinguished-lecture)：**测试时扩展**技术的创新使得在 **ARC 公开验证集**上获得了 **61%** 的分数，表明在**推理优化（inference optimization）**和**模型性能**方面取得了实质性进展。

**主题 5. 社区项目、工具与协作**

- [**在 LlamaIndex 中集成 OpenAI Agent 流式聊天**](https://github.com/jerryjliu/llama_index/blob/main/docs/docs/examples/agent/openai_agent.ipynb)：在 **LlamaIndex** 中实现的 **OpenAI agents** 支持逐 token 的响应生成，展示了动态交互能力，并促进了社区框架内复杂的 **agentic workflows**。
- [**Tinygrad 与 Hailo 移植用于边缘部署**](https://github.com/tinygrad/tinygrad/issues/7044)：**Tinygrad** 致力于将模型移植到 **Raspberry Pi 5** 上的 **Hailo** 硬件，克服了**量化模型（quantized models）**、**CUDA** 和 **TensorFlow** 的挑战，反映了社区对**边缘 AI 部署（edge AI deployments）**和轻量级模型执行的推动。
- [**DSPy 与 PureML 增强高效数据处理**](https://github.com/mryab/efficient-dl-systems)：社区成员正在利用 **PureML** 等工具进行自动机器学习数据集管理，并与 **LlamaIndex** 和 **GPT-4** 集成，以简化数据一致性和特征创建，从而支持高效的 **ML 系统训练**和**数据处理工作流**。

---

# 第一部分：Discord 高层摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity API 添加引用支持**：**Perplexity API** 现在包含**引用（citations）**功能且未引入破坏性变更，sonar online 模型的默认速率限制已提高到 **50 requests/min**。用户可以查阅 [Perplexity API 文档](https://docs.perplexity.ai/home)了解更多详情。
  
  - 讨论强调了 API 的输出与 Pro Search 不同，这是由于底层模型不同，导致一些寻求跨平台一致结果的用户感到失望。
- **梯度下降（Gradient Descent）的进展**：社区成员探讨了各种**梯度下降**技术，重点关注其在机器学习中的应用，并分享了通过[详细文档](https://www.perplexity.ai/search/types-of-gradient-descent-5YX7Q3fPSXuu3PjHcP8FEw)优化模型训练的见解。
  
  - 讨论了标准、**随机（stochastic）**和**小批量梯度下降（mini-batch gradient descent）**方法之间的比较，展示了实现和性能增强的最佳实践。
- **Zomato 推出食物救援（Food Rescue）**：Zomato 推出了 **'Food Rescue'** 计划，使用户能够通过[此链接](https://www.perplexity.ai/page/zomato-s-food-rescue-initiativ-ib.dkRYeTniiF1ytZRBuHQ)以折扣价购买**已取消的订单**。该计划旨在减少食物浪费，同时提供实惠的用餐选择。
  
  - 反馈强调了该计划对 Zomato 和客户的潜在益处，引发了关于餐饮外卖行业可持续发展实践的讨论。
- **雷帕霉素（Rapamycin）在抗衰老中的作用**：关于**雷帕霉素**及其抗衰老效果的新研究引起了关注，引发了关于[此处](https://www.perplexity.ai/page/the-discovery-of-anti-aging-ra-6dtrHKSyRm6YN.QMCJeNcw)详述的正在进行的实验的对话。
  
  - 用户分享了使用该药物的个人经验，辩论了其对长寿和健康的潜在益处和弊端。

 

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Zebra-Llama 增强了针对罕见病的 RAG**：**Zebra-Llama** 模型专注于上下文感知训练，以提高检索增强生成 (RAG) 能力，专门针对 **Ehlers-Danlos Syndrome** 等罕见病，并如 [GitHub 仓库](https://github.com/karthiksoman/zebra-Llama) 所示，增强了引用的准确性。
  
  - 它在现实场景中的应用强调了该模型在普及专业医学知识方面的潜力。
- **Chonkie 简化了 RAG 文本分块**：**Chonkie** 推出了一个轻量级且高效的库，旨在实现快速的 RAG 文本分块，正如 [Chonkie GitHub 仓库](https://github.com/bhavnicksm/chonkie) 中详述的那样，促进了更便捷的文本处理。
  
  - 该工具简化了将文本分块过程集成到现有工作流中的步骤，提高了整体效率。
- **Ollama Operator 简化了 LLM 部署**：**Ollama Operator** 通过极简的 YAML 配置实现了 Ollama 实例和 LLM 服务器部署的自动化，正如最近的 [KubeCon 演讲](https://www.youtube.com/watch?v=XWjZQfSXKDg) 中所演示的那样。
  
  - 通过开源该 Operator，用户可以毫不费力地管理其 LLM 部署，从而简化部署流程。
- **Qwen2.5 Coder 在代码生成方面超越 GPT4o**：根据 [YouTube 性能洞察](https://youtu.be/Xs0EkLYu6hw)，**Qwen2.5 Coder 32B** 模型在代码生成任务中表现出优于 **GPT4o** 和 **Claude 3.5 Sonnet** 的性能。
  
  - 这一进步使 Qwen2.5 Coder 成为需要高效代码生成能力的开发者的竞争性选择。

 

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 2.5-Coder-32B 超越以往模型**：**Qwen 2.5-Coder-32B** 的发布受到了热烈欢迎，成员们报告其令人印象深刻的性能超过了早期模型。
  
  - 人们寄予厚望，认为这一迭代将显著增强使用强大语言模型的开发者的编码能力。
- **优化 Llama 3 微调**：一位成员指出，与其微调后的 **Llama 3** 模型相比，原始模型的推理时间更慢，这引发了关于潜在配置问题的讨论。
  
  - 建议包括验证浮点精度的一致性以及审查脚本以识别影响推理速度的因素。
- **Ollama API 支持前端集成**：成员们探索了在终端运行 **Ollama** 并使用 **Streamlit** 开发聊天 UI，通过 **Ollama API** 确认了可行性。
  
  - 一位用户表示打算进一步研究 API 文档，以便在他们的项目中实施该解决方案。
- **评估 Transformers 与 RNNs 和 CNNs**：讨论了是否可以使用 Unsloth 训练 **RNNs** 和 **CNNs** 等模型，并澄清目前不支持标准神经网络。
  
  - 一位成员反思了观念的转变，强调了 Transformer 架构在近期 AI 发展中的主导地位。
- **关于 LLM 数据集多样性的辩论**：对于 **LLM 数据集** 的构成存在挫败感，人们担心不加区别地包含各种数据源。
  
  - 相反，另一位成员通过强调数据集的**多样性**本质来为数据集辩护，强调了对数据质量的不同看法。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Qwen 的 Coder 突击队征服基准测试**：**Qwen2.5-Coder** 系列的推出带来了多种尺寸的编程模型，并在基准测试中展现出先进的性能，正如 [Qwen 的推文](https://x.com/Alibaba_Qwen/status/1856040217897251044)所宣布的那样。
  
  - 成员们观察到旗舰模型在基准评估中超越了多个专有模型，引发了关于其在编程 LLM 领域潜在影响的讨论。
- **NVIDIA 的 MM-Embed 设定多模态新标准**：**NVIDIA 的 MM-Embed** 被揭晓为首个在 **多模态 M-BEIR 基准测试** 中达到 SOTA 结果的多模态检索器，详见[这篇文章](https://www.marktechpost.com/2024/11/06/nvidia-ai-introduces-mm-embed-the-first-multimodal-retriever-achieving-sota-results-on-the-multimodal-m-beir-benchmark/?amp)。
  
  - 这一进展通过整合视觉和文本数据增强了检索能力，引发了关于其在各种 AI 任务中应用的对话。
- **Open Hermes 2.5 Mix 增强模型复杂性**：正如在 [general 频道](https://discord.com/channels/1053877538025386074/1149866623109439599/1304569083475267615)中所讨论的，将代码数据集成到 **Open Hermes 2.5 mix** 中显著增加了模型的复杂性和功能。
  
  - 团队旨在提高模型在各种应用中的能力，成员们强调了潜在的性能增强。
- **AI 推理缩放面临关键挑战**：关于 AI 模型 **推理缩放 (Inference Scaling)** 的讨论集中在当前缩放方法的局限性上，参考了诸如 [Speculations on Test-Time Scaling](https://www.youtube.com/live/6fJjojpwv1I?si=6byPStsGqUHSK0qP) 等关键文章。
  
  - 对生成式 AI 改进速度放缓的担忧促使成员们思考未来的方向和可扩展的性能策略。
- **显微镜下的机器去学习技术**：如[研究](https://arxiv.org/abs/2410.16454)所示，关于 **机器去学习 (Machine Unlearning)** 的研究质疑了现有方法从 **大型语言模型 (LLM)** 中擦除不需要的知识的有效性。
  
  - 研究结果表明，像量化（Quantization）这样的方法可能会无意中保留被遗忘的信息，这促使社区呼吁改进去学习策略。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **归一化 Transformer (nGPT) 复现挑战**：参与者尝试复现 [nGPT 结果](https://arxiv.org/html/2410.01131v1)，观察到速度提升因任务性能指标而异。
  
  - 该架构强调对 Embedding 和隐藏状态进行 **单位范数归一化 (Unit Norm Normalization)**，从而在不同任务中实现加速学习。
- **Value Residual Learning 的进展**：**Value Residual Learning** 通过允许 Transformer 块访问先前计算的值，显著助力了训练加速（Speedrun）的成功，从而降低了训练期间的 Loss。
  
  - 可学习残差（Learnable Residuals）的实现显示出性能的提升，引发了对在更大模型中缩放该技术的思考。
- **探索低成本图像模型训练技术**：成员们强调了有效的低成本/低数据图像训练方法，如 **MicroDiT**、**Stable Cascade** 和 **Pixart**，以及用于优化性能的 **逐渐增加 Batch Size**。
  
  - 尽管这些技术很简单，但已展示出稳健的结果，鼓励在资源受限的环境中采用。
- **通过符号方程逼近深度神经网络**：提出了一种从深度神经网络中提取符号方程的方法，利用 **基于 SVD 的线性映射拟合** 促进有针对性的行为修改。
  
  - 成员们对潜在的副作用提出了担忧，特别是在需要细微行为控制的场景中。
- **指令微调模型需要 apply_chat_template**：对于 **指令微调模型 (Instruct Tuned Models)**，成员们确认了 `--apply_chat_template` 标志的必要性，并参考了特定的 [GitHub 文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/bd80a6c0099ee207e70f4943117739a817eccc0b/lm_eval/__main__.py#L426-L427)。
  
  - 成员们寻求 Python 集成的实现指导，强调遵循文档配置以确保兼容性。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen 2.5 Coder 逼近 Claude 的编程表现**：**Qwen 2.5 Coder** 模型在 diff 指标上获得了 **72.2%** 的基准测试分数，几乎与 **Claude** 在编程任务中的表现持平，正如 [Qwen2.5 Coder Demo](https://huggingface.co/spaces/Qwen/Qwen2.5-Coder-demo) 所宣布的那样。
  
  - 用户正在积极讨论在各种 GPU 配置上本地运行 Qwen 2.5 Coder 的可行性，重点关注在不同硬件配置下优化性能的兴趣。
- **将 Embeddings 与 Aider 集成增强功能**：关于在 **Aider** 中进行 **Embedding Integration** 的讨论强调了开发 API 以促进与 **Qdrant** 的无缝查询，旨在改进 [Aider Configuration Options](https://aider.chat/docs/config/options.html#--map-tokens-value) 中详述的上下文生成。
  
  - 社区成员提议创建一个用于查询的自定义 Python CLI，这突显了 **Aider** 与 **Qdrant** 之间需要更强大的集成机制。
- **OpenCoder 凭借广泛的 Code LLM 产品领先**：[OpenCoder](https://opencoder-llm.github.io/) 已成为一个突出的开源 **code LLM** 家族，提供在 **2.5 trillion tokens** 原始代码上训练的 **1.5B** 和 **8B** 模型，并为研究进展提供**模型权重**和**推理代码**。
  
  - OpenCoder 在数据处理方面的透明度和资源的可用性旨在支持研究人员突破 **code AI** 开发的界限。
- **Aider 在 1300 Token 时面临上下文窗口挑战**：针对 **Aider 的 1300 上下文窗口** 提出了一些担忧，部分用户反映其效果不佳，影响了该工具在实际应用中的可扩展性和性能，正如 [Aider Model Warnings](https://aider.chat/docs/llms/warnings.html) 中所讨论的。
  
  - 有建议认为 **Aider 后端** 的修改可能会导致这些警告，尽管根据用户反馈，目前这些警告并未阻碍使用。
- **RefineCode 通过海量编程语料库增强训练**：**RefineCode** 引入了一个强大的预训练语料库，包含跨越 **607 种编程语言** 的 **960 billion tokens**，显著增强了像 **OpenCoder** 这样新兴 **code LLMs** 的训练能力。
  
  - 这一可复现的数据集因其质量和广度而受到赞誉，使得在开发先进 **code AI** 模型过程中能够进行更全面、更有效的训练过程。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 3.5 和 Flux 提升性能**：用户正在从 **Stable Diffusion 1.5** 转向 **SD 3.5** 和 **Flux** 等新模型，并指出这些版本需要更少的 VRAM 且能提供更强的性能。
  
  - 有建议推荐探索更小的 [GGUF models](https://huggingface.co/youknownothing/realDream_STOIQONewreality/commit/f5d8fadc6b1e78130050509bb8d165d362b5d304)，它们可以更高效地运行，即使在有限的硬件上也是如此。
- **GPU 性能和 VRAM 效率担忧**：对于每天运行 **Stable Diffusion** 带来的长期 **GPU** 使用损耗提出了担忧，并将其与游戏性能影响进行了比较。
  
  - 一些用户建议 **GPU** 价格可能会随着即将推出的 **RTX 5000** 系列而下降，鼓励他人在购买新硬件前先等待。
- **Stable Diffusion 1.5 的高效 LoRA 训练**：一位用户询问了如何使用小数据集为 **Stable Diffusion 1.5** 训练 **LoRA**，并分享了他们在 **Flux-based training** 方面的经验。
  
  - 建议包括使用 [Kohya_ss trainer](https://www.scottbaker.ca/AI/LoRA-Training) 并遵循特定的在线指南，以有效地完成训练过程。
- **Pollo AI 推出 AI 视频生成**：**Pollo AI** 作为一种新工具被引入，使用户能够根据文本提示创建视频并使静态图像动起来。
  
  - 该工具允许通过根据用户定义的参数生成引人入胜的视频内容来进行创意表达。
- **GGUF 格式增强模型效率**：用户了解了 **GGUF format**，它允许在图像生成工作流中更紧凑、更高效地使用模型。
  
  - 有人提到，与大型模型相比，使用 **GGUF** 文件可以显著减少资源需求，同时保持质量。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **3D Object Generation API 因使用率低而被弃用**：**3D Object Generation API** 将于本周五停用，理由是使用率极低，每几周的请求量少于 **5 次**，详见 [文档](https://openrouter.ai/docs/objects)。
  
  - 团队计划将精力转向能获得更高社区参与度和使用率的功能。
- **Hermes 模型表现出稳定性问题**：用户观察到 **Hermes 模型**在免费和付费层级的响应均不一致，这可能是由于 **OpenRouter** 端的速率限制或后端问题导致的。
  
  - 社区成员正在调查根本原因，讨论这是否与模型优化或基础设施限制有关。
- **Llama 3.1 70B Instruct 模型受到关注**：**Llama 3.1 70B Instruct** 模型的使用率正在增加，特别是在 Skyrim AI Follower Framework 社区中，用户将其价格和性能与 Wizard 模型进行了对比。
  
  - 社区成员渴望探索其高级功能，并讨论潜在的集成方案和性能基准测试。
- **Qwen 2.5 Coder 模型发布，具备 Sonnet 级别性能**：**Qwen 2.5 Coder** 模型已发布，在 **32B 参数**量下达到了 Sonnet 的编程能力，正如在 [GitHub](https://github.com/QwenLM/Qwen2.5-Coder) 上宣布的那样。
  
  - 社区成员对其增强编程任务的潜力感到兴奋，期待能显著提高生产力。
- **Custom Provider Keys 测试版功能需求旺盛**：成员们正积极请求访问 **custom provider keys** 测试版功能，显示出浓厚的社区兴趣。
  
  - 一位成员感谢团队考虑其请求，反映出用户对使用新功能的迫切愿望。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Qwen2.5 Coder 性能**：[Qwen2.5 Coder](https://x.com/huybery/status/1856042011390063015) 系列推出了 **Qwen2.5-Coder-32B-Instruct** 等模型，在多个基准测试中实现了与 **GPT-4o** 相当的竞争力。
  
  - 详细的性能指标表明 **Qwen2.5-Coder-32B-Instruct** 已经超越了其前代产品，预计在不久的将来会发布一份详尽的论文。
- **FrontierMath 基准测试介绍**：[FrontierMath](https://arxiv.org/abs/2409.12186) 提出了一个包含复杂数学问题的新基准测试，目前的 AI 模型有效解决率不足 **2%**，突显了 AI 能力的重大差距。
  
  - 该基准测试的难度在与现有替代方案对比时得到了强调，引发了关于其对未来 AI 训练方法潜在影响的讨论。
- **SALSA 增强模型合并技术**：[SALSA](https://arxiv.org/abs/2411.01798) 框架通过创新的模型合并策略解决了 AI 对齐的局限性，标志着 **Reinforcement Learning from Human Feedback** (RLHF) 的重大进展。
  
  - 社区对 SALSA 优化 AI 对齐的潜力表示兴奋，正如“*woweee*”等热情的惊叹所反映的那样。
- **GPT-5 中的有效 Scaling Laws**：讨论表明，尽管有性能不及预期的看法，但 **Scaling Laws** 在最近的 **GPT-5** 模型中仍然有效，这表明规模化在特定任务上产生的收益正在递减。
  
  - 对话强调了 OpenAI 澄清围绕 AGI 信息传递的必要性，因为社区中仍然存在不切实际的期望。
- **语言模型优化的进展**：[Neural Notes](https://www.youtube.com/watch?v=DVkM5dB3Oqs) 的最新一期深入探讨了**语言模型优化**，采访了斯坦福大学的 Krista Opsahl-Ong，讨论了自动提示词优化技术。
  
  - 此外，讨论还涉及了 **DSPy** 中的 **MIPRO 优化器**，意在加深对这些优化工具的理解。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **JSON 分块：解决 RAG 工具数据缺口**：将 **JSON 分块 (chunking JSON)** 成更小的文件可以确保 RAG 工具捕获所有相关数据，防止遗漏。
  
  - 尽管有效，但成员们指出这种方法增加了工作流长度，正如在 `prompt-engineering` 和 `api-discussions` 频道中所讨论的那样。
- **LLM 驱动的代码生成简化数据结构化**：成员们提议使用 **LLMs** 生成用于结构化数据插入的代码，从而简化集成过程。
  
  - 这一方法广受欢迎，一位用户强调了其在减少手动编码工作方面的潜力，这在多次讨论中都被提及。
- **Function Calling 更新增强 LLM 能力**：讨论了 LLM 中 **function calling** 的更新，用户正在寻求优化其工作流中结构化输出的方法。
  
  - 建议包括利用 **ChatGPT** 等工具进行头脑风暴，并实施高效策略以增强响应生成。
- **AI TTS 工具：平衡成本与功能**：讨论突出了各种 **text-to-speech** (TTS) 工具，如 **f5-tts** 和 **Elven Labs**，并指出 **Elven Labs** 价格更高。
  
  - 用户对**时间戳数据**的可用性以及在消费级硬件上运行这些 TTS 解决方案的挑战表示担忧。
- **AI 图像生成：克服工作流限制**：用户对 **AI video generation** 的局限性表示沮丧，强调需要能将多个场景缝合在一起的工作流。
  
  - 正如 `ai-discussions` 频道中所强调的，用户渴望在专注于视频的 AI 解决方案上取得进展，而不是仅仅依赖基于文本的模型。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Google 寻求 NotebookLM 反馈**：**Google 团队**正在进行一项关于 **NotebookLM** 的 **10 分钟反馈调查**，旨在指导未来的改进。感兴趣的工程师可以在[这里](https://forms.gle/qREhTEhbstYzVHvSA)注册。
  
  - 完成调查的参与者将获得 **20 美元的礼品码**，前提是年满 **18 岁**。这一举措有助于 Google 收集可用于产品改进的见解。
- **NotebookLM 助力多样化技术用例**：**NotebookLM** 被用于**技术面试准备**，通过不同声音进行模拟面试以增强练习效果。
  
  - 此外，工程师们还在利用 NotebookLM 进行**体育解说**实验和生成**高效的教育摘要**，展示了其在处理音频和文本数据方面的多功能性。
- **Podcast 功能面临 AI 生成的小故障**：用户报告 **NotebookLM** 中的 **podcast 功能**偶尔会产生内容*幻觉 (hallucinates)*，导致意想不到且幽默的结果。
  
  - 关于每个笔记本生成多个 podcast 以及有效管理这些 AI 引起的误差的策略讨论正在进行中。
- **NotebookLM 在 AI 工具竞争中脱颖而出**：在写作和求职准备的生产力提升方面，**NotebookLM** 正被拿来与 **Claude Projects**、**ChatGPT Canvas** 以及 **Notion AI** 进行比较。
  
  - 工程师们正在评估每种工具的**优缺点**，特别是关注那些能帮助 **ADHD** 用户保持生产力的功能。
- **与 Google Drive 和移动端的无缝集成**：**NotebookLM** 现在提供了一种**同步 Google Docs** 的方法，通过提议的批量同步功能简化更新过程。
  
  - 虽然**移动版本**仍不完善，但用户对专用 App 以在智能手机上访问完整笔记的需求非常强烈，同时也期待移动端网页功能的改进。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **MacBook 上的 LM Studio GPU 利用率**：用户提出了关于在运行 LM Studio 时如何确定 **MacBook M4** 的 **GPU utilization** 的问题，并强调了与不同配置规格相比，生成速度可能较慢。
  
  - 讨论涉及了设置规格和结果的比较，强调了需要优化配置以提高 **generation performance**。
- **LM Studio 模型加载问题**：一位用户报告称，尽管文件夹中存在 **GGUF files**，但 LM Studio 无法对其进行索引，并提到了应用程序最近的结构变化。
  
  - 建议确保文件夹中仅包含 **relevant GGUF files**，并保持正确的文件夹结构以解决 **model loading** 问题。
- **LangChain 中的 Pydantic 错误**：在集成 **LangChain** 时遇到了与 `__modify_schema__` 方法相关的 `PydanticUserError`，这表明 **Pydantic** 可能存在版本不匹配。
  
  - 用户们不确定该错误是由于 LangChain 的 **bug** 还是与当前使用的 Pydantic 版本的 **compatibility issue** 导致的。
- **Gemma 2 27B 在低精度下的表现**：**Gemma 2 27B** 即使在较低的精度设置下也表现出卓越的性能，成员们注意到在特定模型上使用 **Q8 相比 Q5** 的收益微乎其微。
  
  - 参与者强调在评估中需要额外的上下文，因为 **specifications alone** 可能无法充分传达 **performance metrics**。
- **LLM 推理的笔记本电脑推荐**：关于新型 **Intel Core Ultra CPUs** 与旧款 **i9 models** 在 **LLM inference** 性能差异的咨询，一些建议倾向于 **AMD** 替代方案。
  
  - 建议包括优先考虑 **GPU performance** 而非 CPU 规格，并考虑使用 **ASUS ROG Strix SCAR 17** 或 **Lenovo Legion Pro 7 Gen 8** 等笔记本电脑以获得最佳的 **LLM tasks** 体验。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Qwen 2.5 Coder 发布**：**Qwen2.5-Coder-32B-Instruct** 模型已发布，同时发布的还有从 **0.5B** 到 **32B** 的系列编码模型，提供多种量化格式。
  
  - 它在编程基准测试中取得了极具竞争力的表现，超越了 **GPT-4o** 等模型，展示了 **Qwen** 系列的能力。详细信息请参阅 [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186)。
- **FrontierMath 基准测试揭示 AI 的局限性**：新引入的 **FrontierMath** 基准测试显示，当前的 AI 系统只能解决不到 **2%** 的复杂数学问题。
  
  - 该基准测试将重点转移到具有挑战性的原创问题上，旨在测试 AI 相对于人类数学家的能力。更多详情请访问 [FrontierMath](https://epochai.org/frontiermath)。
- **Open Interpreter 项目进展**：**Open Interpreter** 项目取得了进展，团队已将其开源以促进社区贡献。
  
  - *“你们把它开源了真是太酷了，”* 表达了成员们对开源方向的热情。感兴趣的各方可以在 [GitHub](https://github.com/OpenInterpreter/open-interpreter) 上查看该项目。
- **AI Agent 开发中的基础设施挑战**：对话重新探讨了构建高效 **AI Agent** 的 **infrastructure challenges**，重点关注初创公司面临的 **buy vs. build**（购买还是自建）决策。
  
  - 讨论强调了关于 **OpenAI** 早期计算资源演变和分配的担忧，并指出了遇到的重大障碍。
- **测试时计算 (Test-time Compute) 技术的进展**：**ARC public validation set** 取得了一项新的 state-of-the-art 成就，通过创新的 **test-time compute** 技术获得了 **61%** 的分数。
  
  - 正在进行的辩论质疑 AI 社区如何以不同方式看待 **training** 和 **test-time** 过程，并建议在方法论上进行潜在的统一。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **SVDQuant 加速 Diffusion Models**：一位成员分享了 [SVDQuant](https://arxiv.org/abs/2411.05007) 论文，该论文通过将权重和激活量化为 **4 bits**，并利用低秩分支（low-rank branch）有效处理离群值（outliers），从而优化了扩散模型。
  
  - 尽管与 LoRAs 相关的内存访问开销有所增加，该方法仍提升了大尺寸图像生成任务的性能。
- **Aria Multimodal MoE 模型性能提升**：**Aria 多模态 MoE 模型**实现了 **4-6 倍的加速**，并利用 **A16W4** 和 [torch.compile](https://github.com/mobiusml/hqq/blob/master/examples/hf/aria_multimodal.py) 适配到单张 **24GB GPU** 中。
  
  - 尽管当前代码库较为混乱，但它为类似 MoE 模型的复现提供了潜在的见解。
- **BitBlas 支持 int4 Kernels**：**BitBlas** 现在支持 **int4 kernels**，正如社区成员所讨论的，这实现了高效的缩放矩阵乘法（scaled matrix multiplication）操作。
  
  - 讨论中强调了 **H100** 上缺乏 int4 计算核心，引发了关于操作支持的疑问。
- **TorchAO 框架增强**：该项目计划通过整合近期研究的优化方案，扩展 **TorchAO** 中现有的 **Quantization-Aware Training (QAT)** 框架。
  
  - 该策略利用已有的基础设施来引入新功能，初步重点是线性操作（linear operations）而非卷积模型。
- **DeepMind 的神经压缩技术**：**DeepMind** 介绍了使用神经压缩文本训练模型的方法，详见其[研究论文](https://arxiv.org/pdf/2404.03626)。
  
  - 社区的关注点集中在论文的 **Figure 3**，不过并未讨论具体的引用内容。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Raspberry Pi 5 上的 Hailo 移植**：一位开发者正在将 **Hailo** 移植到 **Raspberry Pi 5**，成功将模型从 **tinygrad** 转换为 **ONNX** 再转换为 Hailo，尽管在处理需要 **CUDA** 和 **TensorFlow** 的**量化模型**时面临挑战。
  
  - 他们提到，由于芯片缓存有限且内存带宽不足，在边缘设备上执行训练代码是不切实际的。
- **处理浮点异常**：讨论集中在检测 **NaN** 和 **overflow** 等**浮点异常**，强调了检测方法中平台支持的必要性。相关资源包括 [Floating-point environment - cppreference.com](https://en.cppreference.com/w/cpp/numeric/fenv) 和 [FLP03-C. Detect and handle floating-point errors](https://wiki.sei.cmu.edu/confluence/display/c/FLP03-C.+Detect+and+handle+floating-point+errors)。
  
  - 参与者强调了在浮点运算期间捕获错误的重要性，并主张采用鲁棒的错误处理技术。
- **Tinybox 与 Tinygrad 的集成**：讨论了 **Tinybox** 与 **tinygrad** 的集成，重点关注潜在的升级以及解决影响 **5090** 升级的 **P2P hack patch** 相关问题。参考了 [tinygrad 仓库](https://github.com/tinygrad/tinygrad/issues/7044)中的相关 GitHub issues。
  
  - 关于不同 **PCIe controller** 能力对硬件设置性能影响存在一些推测。
- **TPU 后端策略**：一位用户提议开发 **TPU v4 汇编后端**，并表示愿意在清理工作后进行协作。他们询问了 **LLVM** 中汇编的向量化以及目标支持的具体 **TPU** 版本。
  
  - 社区就合并后端策略的可行性和技术要求进行了讨论。
- **解释 Beam Search 输出**：有人寻求关于解释 **beam search** 输出的帮助，特别是理解**进度条**如何与 **kernel 执行时间**相关联。注意到绿色指示器代表 kernel 的**最终运行时间**。
  
  - 该用户对 **actions** 和 **kernel size** 表示困惑，请求进一步澄清以准确解释结果。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AI 面试机器人开发**：一位用户正在启动一个 **GenAI 项目**，开发一个 **AI 面试机器人**，该机器人根据简历和职位描述生成问题，并对回答进行百分制评分。
  
  - 他们正在寻求**免费资源**，如向量数据库和编排框架，并强调编程工作将由他们自己完成。
- **Aya-Expanse 模型增强**：一位用户称赞了 **Aya-Expanse** 模型在翻译之外的能力，特别是在 **function calling** 和处理希腊语任务方面。
  
  - 他们注意到该模型能有效地为不需要函数调用的查询选择 `direct_response`，从而提高了响应准确性。
- **基于文档响应的 Cohere API**：一位用户询问是否有 API 可以从预先上传的 DOCX 和 PDF 文件中生成自由文本响应，并指出目前仅支持 embeddings。
  
  - 他们表示对实现类似 **ChatGPT assistants API** 功能的方案感兴趣。
- **Cohere API 错误与延迟**：用户报告了 **Cohere API** 的多个问题，包括访问模型详情时的 **500 Internal Server Errors** 和 **404 errors**。
  
  - 此外，还强调了延迟增加（响应时间达到 **3 分钟**）以及 **Embed API 缓慢**的问题，用户被引导至 [Cohere Status Page](https://status.cohere.com/) 获取更新。
- **vnc-lm Discord 机器人集成**：一位成员介绍了 **vnc-lm** Discord 机器人，它集成了 **Cohere API** 和 **GitHub Models API**，以及本地 **ollama models**。
  
  - 主要功能包括创建对话分支、优化 prompt 以及发送上下文材料（如截图和文本文件），可以通过 [GitHub](https://github.com/jake83741/vnc-lm) 使用 `docker compose up --build` 进行设置。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 1.0 更新测试**：一位用户自愿协助测试即将发布的 **Open Interpreter 1.0** 更新，该更新目前位于 dev 分支，计划于下周发布。他们分享了 [安装命令](https://github.com/davidteren/mac_fan_control-self-healing-coder)。
  
  - 社区强调需要进行 Bug 测试，并将更新适配到不同的操作系统，以确保顺利推出。
- **Open Interpreter 的硬件需求**：一位用户询问配备 **64GB 或 24GB** RAM 的 **Mac Mini M4 Pro** 是否足以有效运行 **Open Interpreter**。大家达成共识，确认该配置可以运行。
  
  - 讨论还包括集成麦克风和扬声器等额外组件，以增强硬件环境。
- **Qwen 2.5 Coder 模型发布**：新发布的 **Qwen 2.5 coder models** 在**代码生成**、**代码推理**和**代码修复**方面表现出显著改进，其中 **32B 模型**可与 **OpenAI 的 GPT-4o** 媲美。
  
  - 成员们表现出极大的热情，因为 Qwen 和 Ollama 展开了合作，正如 Qwen 所说：*“非常激动能与我们最好的朋友之一 Ollama 共同发布我们的模型！”*。更多详情请见 [官方推文](https://x.com/ollama/status/1856051733513797929?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)。
- **CUDA 配置调整**：一位成员提到他们调整了 **CUDA** 设置，在进行必要微调后实现了满意的配置。
  
  - 在系统上的成功实例化凸显了正确配置 **CUDA** 对实现最佳性能的重要性。
- **为 Open Interpreter 进行 Software Heritage 代码归档**：一位用户提议协助将 **Open Interpreter** 代码归档至 **Software Heritage**，旨在造福后代。
  
  - 该提议强调了社区对保护开发者宝贵贡献的承诺。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaParse Premium 在文档解析方面表现出色**：Hanane Dupouy 展示了 [LlamaParse Premium](https://t.co/pgqVUhwjXh) 如何高效地将复杂的图表和示意图解析为结构化的 Markdown，从而增强文档的可读性。
  
  - 该工具将视觉数据转化为可访问的文本，显著提升了**文档的可用性**。
- **高级分块（chunking）策略提升性能**：@pavan_mantha1 概述了**三种高级分块策略**，并提供了一个用于在个人数据集上进行测试的[完整评估设置](https://t.co/8UTY4xNHOT)。
  
  - 这些策略旨在增强**检索和 QA 功能**，展示了有效的数据处理方法。
- **PureML 自动化数据集管理**：PureML 利用 LLM 进行机器学习数据集的自动清理和重构，具有[上下文感知处理](https://t.co/E6frzia1yR)和智能特征创建功能。
  
  - 这些功能提高了数据的一致性和质量，并集成了 **LlamaIndex** 和 **GPT-4** 等工具。
- **微调 LLM 模型的基准测试**：一位成员寻求关于对其在 [Hugging Face](https://huggingface.co/Anoshor/prism-v2) 上的微调 LLM 模型进行基准测试的指导，该模型在 Open LLM 排行榜上遇到了错误。
  
  - 他们请求协助，以便有效地利用排行榜来评估模型性能。
- **优化摄取（ingestion）的 Docker 资源设置**：用户讨论了 Docker 配置，分配了 **4 个 CPU 核心**和 **8GB 内存**，以优化 [sentence transformers 摄取流水线](https://docs.llamaindex.ai/)。
  
  - 尽管有这些设置，摄取过程仍然缓慢且容易失败，凸显了进一步优化的必要性。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **M3DocRAG 树立了多模态 RAG 的新标准**：M3DocRAG 在利用来自大量 PDF 语料库的**多模态信息**进行**问答**方面展示了令人印象深刻的结果，并在 **ColPali 基准测试**中表现优异。
  
  - *Jaemin Cho* 强调了它在处理跨越不同文档上下文的**单跳和多跳问题**方面的多功能性。
- **随 M3DocVQA 推出的新开放域基准测试**：**M3DocVQA**（一个 **DocVQA 基准测试**）的引入，挑战模型回答跨越 **3000 多份 PDF** 和 **4 万多页**的**多跳问题**。
  
  - 该基准测试旨在通过利用**文本、表格和图像**等各种元素来增强理解。
- **DSPy RAG 用例引发关注**：一位成员对 **DSPy RAG 功能**的潜力表示热切关注，并表示有浓厚的实验兴趣。
  
  - 他们注意到 **DSPy RAG** 与**视觉能力**之间充满前景的交集，暗示了未来有趣的应用程序。
- **LangChain 集成停止支持**：[GitHub](https://link.to.github) 上的最新更新表明，目前与 **LangChain** 的集成已不再维护，可能无法正常运行。
  
  - 一位成员就这一变化提出了疑问，寻求关于该情况的更多背景信息。
- **DSPy 提示词技术设计为不可组合**：成员们讨论了 **DSPy** 提示词技术的本质，确认它们在设计上是有意**不可组合（not composable）**的。
  
  - 这一决定强调，虽然签名（signatures）可以被操作，但这样做可能会限制功能和控制流的清晰度。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **FastChat 和 ShareGPT 的移除**：移除 **FastChat** 和 **ShareGPT** 在社区内引发了强烈反应，具体见 [PR #2021](https://github.com/axolotl-ai-cloud/axolotl/pull/2021)。成员们对这一决定表示惊讶和担忧。
  
  - 为了维持项目稳定性，有人建议采用替代方案，例如回滚到旧的 commit，这表明目前正在努力满足社区的需求。
- **Metharme 支持延迟**：关于是否继续支持 **Metharme** 的询问得到了解释，延迟是由于 **fschat** 的发布影响了开发进度。
  
  - 社区成员表现出将 **sharegpt** 对话整合到新的 **chat_template** 中的兴趣，反映了克服支持挑战的协作方式。
- **微调 VLMs 的最佳实践**：有人寻求微调 **VLMs** 的帮助，建议使用示例仓库中提供的 **llama vision** 配置。
  
  - 确认可以使用 **llama 3.2 1B 训练 VLM 模型**，展示了社区在高级模型训练技术方面的能力和兴趣。
- **Inflection AI API 更新**：讨论了 **Inflection-3**，它引入了两个模型：用于情感互动的 **Pi** 和用于结构化输出的 **Productivity**，详见 [Inflection AI Developer Playground](https://developers.inflection.ai/docs)。
  
  - 成员们对缺乏 benchmark 数据表示担忧，质疑这些新模型的实际评估及其在现实世界中的应用。
- **新增 Metharme Chat_Template PR**：通过 [PR #2033](https://github.com/axolotl-ai-cloud/axolotl/pull/2033) 分享了一个将 **Metharme** 添加为 **chat_template** 的拉取请求，解决了用户需求并测试了与旧版本的兼容性。
  
  - 鼓励社区成员在本地执行 preprocess 命令以确保功能正常，营造了测试和实施的协作环境。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **项目反馈的中期检查**：团队现在可以通过 [中期检查表单](https://docs.google.com/forms/d/e/1FAIpQLSfxhgqcKWxfs_e1xuF3yukTvIwk_0JhsaVwHizS7o9BYW9Hnw/viewform?usp=sf_link) 提交进度，以获取反馈并可能获得 **GPU/CPU 资源额度**。
  
  - 即使不申请资源，提交表单也至关重要，因为这有助于获得关于项目的宝贵见解。
- **申请额外计算资源**：对额外 **GPU/CPU 资源**感兴趣的团队必须在填写中期检查表单的同时完成 [资源申请表单](https://docs.google.com/forms/d/e/1FAIpQLSeJQ_i6H5bgA5S767QZaorwkzF9_k_63I8JCed3dnlVcvKJ1w/viewform)。
  
  - 资源分配将取决于记录的进度和详细的理由，鼓励即使是新团队也积极申请。
- **Lambda Workshop 提醒**：**Lambda Workshop** 定于明天（**太平洋标准时间 11 月 12 日下午 4-5 点**）举行，鼓励参与者通过 [此链接](https://lu.ma/agents-hackathon-lambda) 预约。
  
  - 本次 workshop 将为团队项目和黑客松流程提供进一步的见解和指导。
- **黑客松团队人数不限**：一位成员询问了 **黑客松允许的团队规模**，确认规模是 **不限人数的**。
  
  - 这为任何有兴趣的人提供了无限制协作的可能性。
- **即将举行的 LLM Agents 讲座**：发布了一项关于今晚讨论 **Lecture 2: History of LLM Agents** 的公告。
  
  - 讨论将包括对讲座的回顾以及对一些 **Agentic 代码**的探索，欢迎任何感兴趣的人参加。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **在不修改 forward 函数的情况下捕获 attention scores**：一位用户询问如何使用 **forward hooks** 在 self-attention 模块中捕获 **attention scores** 而不改变 **forward** 函数。其他人指出 **F.sdpa()** 存在潜在问题，因为它目前不输出 **attention scores**，这表明可能需要进行修改。
- **DCP checkpointing 问题导致 OOM 错误**：一位成员报告称，最新的 **git main** 版本仍未解决在 rank=0 GPU 上收集权重和优化器的问题，导致 **OOM** (Out Of Memory) 错误。
  
  - 他们为 **DCP checkpoint saving** 实现了一个变通方案，打算将其转换为 **Hugging Face** 格式，并可能编写一个 PR 以实现更好的集成。
- **社区支持在 Torchtune 中集成 DCP**：讨论强调了社区对在 **Torchtune** 中集成 **DCP checkpointing** 的支持，并谈到了分享相关的 PR 或 fork。
  
  - 一项更新指出，来自 **PyTorch 贡献者** 的 **DCP PR** 可能很快就会发布，从而增强协作进展。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **SVDQuant 减少内存占用和延迟**：最近的 [SVDQuant](https://svdquant.mit.edu) 为 diffusion models 引入了一种新的量化范式，通过将权重和激活值量化为 4 bits，在 **16GB 笔记本 4090 GPU** 上实现了 **3.5倍的内存减少** 和 **8.7倍的延迟降低**。
  
  - 更多资源可在 [GitHub](https://github.com/mit-han-lab/deepcompressor) 上获取，完整论文可在此处访问 [here](http://arxiv.org/abs/2411.05007)。
- **AI 领域的 Gorilla Marketing**：AI 公司正在采用 **gorilla marketing** 策略，其特点是非传统的促销手段。
  
  - 这一趋势被幽默地通过引用 [Harambe GIF](https://tenor.com/view/harambe-america-murica-flag-waving-gif-17339298) 表现出来，强调了这些营销方式的趣味性。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **RisingWave 增强数据处理技术**：最近的一篇帖子强调了 **RisingWave** 在数据处理方面的进展，重点是 **stream processing** 技术的改进。
  
  - 欲了解更多见解，请查看其 [LinkedIn post](https://www.linkedin.com/posts/risingwave_risingwave-dataprocessing-streamprocessing-activity-7260009892848033792-adOv) 中的完整细节。
- **重点关注 Stream Processing 技术**：讨论集中在最新的 **stream processing** 领域，展示了优化实时数据处理的方法。
  
  - 参与者指出，采用这些创新可能会显著影响数据驱动的决策。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **使用 Gorilla LLM 测试自定义模型**：一位用户询问如何使用 **Gorilla LLM** 来 [对其 fine-tuned LLM 进行 benchmark](https://discord.com/channels/1111172801899012102/1111353033352294440/1304816288719179828)，因为他们是该领域的新手，正在寻求指导。
  
  - 他们表示在 **benchmark testing custom LLMs** 方面特别需要帮助，希望能得到社区的支持和建议。
- **寻求对自定义 LLM 进行 Benchmarking 的支持**：一位用户联系寻求利用 **Gorilla LLM** 对其自定义 fine-tuned 模型进行 benchmarking，并强调了他们在该领域缺乏经验。
  
  - 他们请求在有效地进行 **benchmark testing custom LLMs** 方面提供协助，以更好地理解性能指标。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **继续使用 Fine-Tuned 模型**：一位用户请求在当前设置中**继续使用其 fine-tuned 模型**。
- \*\*\*\*:

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1304537985986990143) (1230 条消息🔥🔥🔥):

> - `Perplexity server issues` (Perplexity 服务器问题)
> - `User frustrations` (用户挫败感)
> - `Communication with users` (与用户的沟通)
> - `Waiting for fixes` (等待修复)
> - `Alternatives to Perplexity` (Perplexity 的替代方案)

- **Perplexity 遭遇服务器问题**：用户报告在登录和访问 Perplexity 线程时遇到困难，出现消息消失以及线程显示“此线程不存在”等错误。团队的状态更新确认了由于近期部署导致的技术问题。
  
  - 虽然部分线程最近已恢复访问，但许多用户发现新消息现在被隐藏了，这加剧了挫败感。
- **用户挫败感与支持不足**：许多用户对 Perplexity 支持团队在修复持续存在的 Bug 方面缺乏沟通表示不满，导致产生被忽视感和挫败感。用户投诉处理不力以及多个反复出现的 Bug 影响了使用体验。
  
  - 用户强调了无法有效使用该平台的问题，表达了对公司表现的担忧，并呼吁提供更好的客户支持。
- **更新沟通**：参与者讨论了公司在就持续问题与用户进行有效沟通方面的明显不足，呼吁提高透明度。一些用户建议公司在停机期间应采取更主动的方式向客户更新进度。
  
  - 有人呼吁实现一个 API，以便及时向 Discord 用户转发平台状态更新信息。
- **寻找替代方案**：一些用户开始寻找 Perplexity 的替代品，特别是对提供 Opus 支持的平台感兴趣，由此产生了如 ChatHub 等建议。在持续出现问题的情况下，用户对继续使用 Perplexity 的价值表示担忧。
- **对未来影响的预期**：随着关于平台状态讨论的展开，人们反思了科技公司长期而言会如何影响用户体验。用户提出，一旦资金开始枯竭，公司对投资者的依赖可能会导致客户服务质量下降。

**提到的链接**：

- [Capybara Let Him Cook GIF - Capybara Let him cook - Discover & Share GIFs](https://tenor.com/view/capybara-let-him-cook-gif-11999534059191155013)：点击查看 GIF
- [来自 Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/aravsrinivas/status/1856039331678269478?s=46)：这是一个部署问题。我们正在回滚。抱歉。很快就会恢复运行。引用 Crunk ✈️ Network School (@crunk304) 的话：Plex 挂了！@AravSrinivas @perplexity_ai
- [Perplexity - Status](https://status.perplexity.com/>)：Perplexity 状态
- [Weareback Wereback GIF - Weareback Wereback Wearebackbaby - Discover & Share GIFs](https://tenor.com/view/weareback-wereback-wearebackbaby-hangover-the-gif-18475016)：点击查看 GIF
- [When Server Down Iceeramen GIF - When Server Down Iceeramen Monkey - Discover & Share GIFs](https://tenor.com/view/when-server-down-iceeramen-monkey-gif-23229726)：点击查看 GIF
- [ChatHub - 并排使用 GPT-4o, Claude 3.5, Gemini 1.5](https://chathub.gg/)：同时使用并对比 GPT-4o, Claude 3.5, Gemini 1.5 及更多聊天机器人
- [Bye Bye GIF - Bye bye - Discover & Share GIFs](https://tenor.com/view/bye-bye-gif-11441473103925830959)：点击查看 GIF

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1304583051216687114) (25 条消息🔥):

> - `Zomato 的食物救援计划`
> - `占星术讨论`
> - `Gradient Descent 技术`
> - `Rapamycin 抗衰老研究`
> - `钓鱼饵料推荐`

- **Zomato 的智能食物救援计划**：Zomato 推出了其 **'Food Rescue'** 计划，允许用户以更低的价格购买**已取消的订单**，详情见[此处](https://www.perplexity.ai/page/zomato-s-food-rescue-initiativ-ib.dkRYeTniiF1ytZRBuHQ)。该举措旨在最大限度地减少食物浪费，同时为消费者提供更实惠的用餐选择。
  
  - 对这一举措的反馈强调了其对 Zomato 和客户的潜在双赢，并引发了关于外卖行业可持续发展实践的讨论。
- **占星术：现实还是神话？**：多位成员讨论了**占星术**的有效性，特别引用了一个深入探讨其真实性的[链接](https://www.perplexity.ai/search/az-asztrologia-tenyleg-valodi-rxOa99aPRRe9b7A5XzCv0A)。对话引发了关于占星术主张在现代生活中影响的不同观点。
  
  - 参与者表达了不同的看法，一些人主张其心理益处，而另一些人则将其斥为伪科学。
- **Gradient Descent 技术探索**：针对各种类型的 **Gradient Descent** 提出了几个疑问，重点关注它们在机器学习中的应用 [链接](https://www.perplexity.ai/search/types-of-gradient-descent-5YX7Q3fPSXuu3PjHcP8FEw)。成员们分享了关于这些方法如何优化模型训练的链接和个人见解。
  
  - 讨论包括标准 Gradient Descent 与 **Stochastic Gradient Descent** 和 **Mini-batch Gradient Descent** 等高级技术之间的比较，展示了实施的最佳实践。
- **抗衰老研究备受关注**：关于 **Rapamycin** 及其抗衰老效果的新研究引起了成员们的关注，引发了关于该领域正在进行的研究的讨论 [链接](https://www.perplexity.ai/page/the-discovery-of-anti-aging-ra-6dtrHKSyRm6YN.QMCJeNcw)。用户分享了使用该药物的个人经验，并讨论了其潜在的利弊。
  
  - 对话集中在这一研究对长寿和健康的意义上，并对未来的发现充满热情。
- **钓鱼探险的最佳饵料**：一位用户询问了捕鱼的**最佳饵料**，这引发了钓鱼爱好者分享技巧的热烈讨论 [链接](https://www.perplexity.ai/search/what-is-the-best-bait-to-catch-8vmh2jbKTDG1cb4_vpEDYw)。建议范围从传统选择到旨在吸引各种鱼类的创新饵料方法。
  
  - 这次交流突显了社区对钓鱼的热情，以及分享从经验中获得的宝贵见解的意愿。

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1304800492093964318) (24 messages🔥):

> - `Perplexity API citations`
> - `API vs. Pro Search output`
> - `Search domain filter`
> - `Citation links output`
> - `Different sources in API`

- **Perplexity API 现在包含引用 (citations)**：**Perplexity API** 宣布**引用功能已公开可用**，且没有破坏性变更（breaking changes），sonar online 模型的默认速率限制（rate limit）已提高至 **50 requests/min**。
  
  - 更多详情，用户可以参考 [docs](https://docs.perplexity.ai/home)。
- **API 输出与 Pro Search 不同**：讨论中提到了 API 与 Pro Search 相比的**输出质量**问题，澄清了 Pro 版本使用了一个 API 无法访问的**不同模型**。
  
  - 成员们对 API 无法获得与 Pro search 相同的输出质量表示失望。
- **搜索域名过滤器查询**：一位用户询问 **search_domain_filter** 是否支持子域名搜索，希望使用 'support.company.com' 而避开 'community.company.com'。
  
  - 关于该功能的对话包括了一个寻求确认其功能的后续问题。
- **引用链接问题**：用户对 API 返回的引用链接提出了担忧，这些链接显示为括号数字而非可点击的链接。
  
  - 多位成员报告了类似经历，引发了关于如何请求不同格式 URL 的讨论。
- **API 与聊天界面来源结果的差异**：一位用户注意到，对于相同的查询，API 返回的**来源 (sources)** 与聊天界面中的来源有显著差异，引发了对其搜索算法的质疑。
  
  - 这种差异被归因于底层算法和抓取的数据，与 LLM 模型的差异无关。

**提到的链接**：

- [no title found](https://docs.perplexity.ai/faq/faq#why-are-the-results-from-the-api-different-from-the-ui): 未找到描述
- [no title found](https://docs.perplexity.ai/home).): 未找到描述

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1304535940215209994) (922 messages🔥🔥🔥):

> - `AI model scams`
> - `Birthday celebrations`
> - `Image generation models`
> - `Model fine-tuning`
> - `Llama models and competition`

- **关于 AI 模型诈骗的讨论**：一位用户讲述了他们如何被一名未能交付 AI 模型的开发人员骗取了 500 美元，强调了自由职业安排中缺乏信任的问题。
  
  - 社区强调了使用 Upwork 等平台以获得更好安全性和可靠服务的重要性。
- **发布新模型庆祝生日**：一位用户宣布了自己的生日，并分享了他们创建的一个名为 'birthday-llm' 的新模型，该模型旨在用于逻辑推理和角色扮演任务。
  
  - 该模型受到了好评，其他人也表达了试用的热情。
- **图像生成与角色一致性**：用户讨论了在图像生成中保持角色一致性的挑战，并推荐了 Animagine XL 3.1 和 DiffuseCraft 等模型。
  
  - 强调了在弹珠台分数检测等任务中使用标记数据集（labeled datasets）的重要性。
- **针对特定用例的 Fine-tuning**：社区讨论了获取能够检测特定物体和分数的模型的方法，以及整合文本和图像数据的技术。
  
  - 强调了在训练模型时设定明确的准确率目标，以确保合理的预期。
- **探索新的 AI 模型和框架**：一位用户寻求在 Hugging Face 上寻找高质量 AI 模型的建议，因为他们担心过时的模型充斥着该平台。
  
  - 对话涉及了 AI 模型开发的快速进步，以及对清晰文档和评估的需求。

**提到的链接**：

- [Redirecting...](https://errors.pydantic.dev/2.9/u/custom-json-schema): 未找到描述
- [What Is Flux In Ai](https://letmegooglethat.com/?q=what+is+flux+in+ai): 未找到描述
- [minchyeom/birthday-llm · Hugging Face](https://huggingface.co/minchyeom/birthday-llm): 未找到描述
- [How to convert model.safetensor to pytorch_model.bin?](https://stackoverflow.com/questions/77708996/how-to-convert-model-safetensor-to-pytorch-model-bin): 我正在 Fine-tuning 一个预训练的 bert 模型，遇到了一个奇怪的问题：当我使用 CPU 进行 Fine-tuning 时，代码这样保存模型：使用 &quot;pytorch_model.bin...
- [Forrest Gump Running GIF - Forrest Gump Running Me When Im Late - Discover & Share GIFs](https://tenor.com/view/forrest-gump-running-me-when-im-late-tom-hanks-gif-5144739): 点击查看 GIF
- [rwitz/cat1.0 · Hugging Face](https://huggingface.co/rwitz/cat1.0): 未找到描述

- [2000 码凝视 GIF - 2000 码凝视 战争 - 发现并分享 GIF](https://tenor.com/view/%D0%B2%D0%B7%D0%B3%D0%BB%D1%8F%D0%B4-2000-%D1%8F%D1%80%D0%B4%D0%BE%D0%B2-%D0%B2%D0%BE%D0%B9%D0%BD%D0%B0-war-soldier-gif-3632617944134077161): 点击查看 GIF
- [为什么我们应该在 softmax 中使用 Temperature？](https://stackoverflow.com/questions/58764619/why-should-we-use-temperature-in-softmax/63471046#63471046)): 我最近在研究 CNN，我想知道 Temperature 在 softmax 公式中的作用是什么？以及为什么我们应该使用高 Temperature 来在概率分布中看到更平滑的范数？...
- [小猫臭臭小猫 GIF - 小猫臭臭小猫臭臭 - 发现并分享 GIF](https://tenor.com/view/kitty-stinky-kitty-stinky-stinky-cat-cat-review-gif-6756203800739239604): 点击查看 GIF
- [猫咪电脑 GIF - 猫咪电脑打字 - 发现并分享 GIF](https://tenor.com/view/cat-computer-typing-fast-gif-5368357): 点击查看 GIF
- [特朗普唐纳德 GIF - 特朗普唐纳德面部 - 发现并分享 GIF](https://tenor.com/view/trump-donald-face-sillyface-silly-gif-5017946): 点击查看 GIF
- [特朗普橙色座位 GIF - 特朗普橙色座位环顾四周 - 发现并分享 GIF](https://tenor.com/view/trump-orange-seat-looking-around-graphic-design-illustration-gif-13708497): 点击查看 GIF
- [海绵宝宝派大星 GIF - 海绵宝宝派大星 - 发现并分享 GIF](https://tenor.com/view/spongebob-patrick-patrick-star-broke-poor-gif-14729256): 点击查看 GIF
- [什么是 temperature？](https://discuss.huggingface.co/t/what-is-temperature/11924): 我看到 “temperature” 这个词被用在各种地方，比如：在模型中 —— transformers 4.12.4 文档 temperature ( float , 可选, 默认为 1.0) – 用于调制下一个...
- [Snoop Snoop Dogg GIF - Snoop Snoop dogg Snoop 微笑 - 发现并分享 GIF](https://tenor.com/view/snoop-snoop-dogg-snoop-smile-gif-18363169359568588385): 点击查看 GIF
- [惩罚者 惩罚者 GIF - 惩罚者 惩罚者 等待 - 发现并分享 GIF](https://tenor.com/view/punisher-the-punisher-wait-no-panicking-gif-22139346): 点击查看 GIF
- [未找到标题](https://refer.hellotrusty.io/zbl2s3tbmx/company/jobs/6709cfa563e9270002f78b90): 未找到描述
- [捉鬼敢死队烤面包机 GIF - 捉鬼敢死队烤面包机 - 发现并分享 GIF](https://tenor.com/view/ghostbuster-toaster-gif-5319546): 点击查看 GIF
- [恶搞之家彼得·格里芬 GIF - 恶搞之家彼得·格里芬 - 发现并分享 GIF](https://tenor.com/view/family-guy-peter-griffin-peter-quagmire-glenn-quagmire-gif-3195664271343394920): 点击查看 GIF
- [city96/FLUX.1-dev-gguf · Hugging Face](https://huggingface.co/city96/FLUX.1-dev-gguf): 未找到描述
- [8000 万营收增长及重新部署 3 名全职员工 GIF - 80M Increased Revenue & Redeploy 3 FTE - 发现并分享 GIF](https://tenor.com/view/80m-increased-revenue-%26-redeploy-3-fte-gif-3993442070749886928): 点击查看 GIF
- [ComfyUI - Advanced - a Hugging Face Space by wrdias](https://huggingface.co/spaces/wrdias/ComfyUI-Advanced): 未找到描述
- [ComfyUI (test) - a Hugging Face Space by John6666](https://huggingface.co/spaces/John6666/comfy_test): 未找到描述
- [詹姆斯·莫里亚蒂 (全息图)](https://memory-alpha.fandom.com/wiki/James_Moriarty_(hologram)): 你 —— 或者某人 —— 要求你的电脑编写一个来自 19 世纪伦敦的邪恶虚构人物 —— 这就是我到来的原因……但我不再是那个造物了。我不再是那个...
- [Animagine XL 3.1 - a Hugging Face Space by cagliostrolab](https://huggingface.co/spaces/cagliostrolab/animagine-xl-3.1): 未找到描述
- [Laxhar/sdxl_noob · Hugging Face](https://huggingface.co/Laxhar/sdxl_noob): 未找到描述
- [YOLO11 🚀 NEW](https://docs.ultralytics.com/models/yolo11/): 探索 YOLO11，这是最先进目标检测领域的最新进展，为各种计算机视觉任务提供无与伦比的准确性和效率。
- [Bad Piggies 主题曲](https://youtu.be/EgAOqt8I5ac): 由 The Orchard Enterprises 提供给 YouTube。Bad Piggies 主题曲 · Ilmari Hakkola。Bad Piggies (游戏原声带) ℗ 2012 Rovio Entertainment。发行于：...
- [GitHub - black-forest-labs/flux: FLUX.1 模型的官方推理库](https://github.com/black-forest-labs/flux): FLUX.1 模型的官方推理库。通过在 GitHub 上创建账户，为 black-forest-labs/flux 的开发做出贡献。
- [🧩 DiffuseCraft - a Hugging Face Space by r3gm](https://huggingface.co/spaces/r3gm/DiffuseCraft): 未找到描述
- [Stable Diffusion XL](https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl#load-model-checkpoints): 未找到描述

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1304822318526369894) (10 messages🔥):

> - `教导 LLM 数学`
> - `Full Stack vs AI Stack`
> - `训练 BART 模型`
> - `基于问题的学习`
> - `翻译中的长句子`

- **讨论教导 LLM 数学的方法**：一位成员表示有兴趣将**推理和逻辑**融入到教导 LLM 数学的过程中，强调了专注于可解决问题的重要性。
  
  - 他们指出，使用具有挑战性但可控的问题（如**初中数学**）来教导 LLM，优于那些成功可能性极低的复杂问题。
- **应对 Full Stack 困境**：一位用户分享了在深入研究 AI 之前是否应该专注于 **Full Stack** 开发的困惑，并提到了他们现有的经典 ML 以及基础 NLP 和 CV 知识。
  
  - 有建议认为追求 Full Stack 可能会耗费大量时间和精力，因此请求社区提供指导。
- **成功训练用于翻译的 BART 模型**：经过 **6 小时的调试**，一位用户成功在 OPUS books de-en 数据集上训练了一个 BART 模型，用于将英语翻译成德语。
  
  - 他们实施了保存模型状态和配置等效率措施，但注意到一些过长的句子在处理过程中会被截断。
- **强调数学中的基于问题的学习**：一位成员强调了数学学习中**富有成效的挣扎 (productive struggle)** 的重要性，认为这对于教育背景下的有效 LLM 训练至关重要。
  
  - 他们评论了这些概念对终身学习者的相关性，并对该主题的共享建议表示感谢。
- **为翻译实现聊天机器人功能**：该用户计划在 BART 模型中添加聊天机器人 pipeline 以实现实用的翻译功能，并在推理 (inference) 过程中使用 **top-p sampling**。
  
  - 这一补充旨在优化模型作为翻译机器人的能力，同时解决训练配置中的复杂性。

 

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1304629286443159635) (22 messages🔥):

> - `Zebra-Llama 模型`
> - `神经元中的分形数据处理`
> - `Chonkie 文本分块库`
> - `用于 Minecraft 模拟的 Lucid`
> - `医疗 AI 更新`

- **为罕见病知识引入 Zebra-Llama**：一个名为 **Zebra-Llama** 的新型模型专注于上下文感知训练，以提高 LLM 的检索增强生成 (RAG) 能力，特别是针对诸如 **Ehlers-Danlos 综合征**等罕见病。
  
  - 该模型包含一个 [GitHub 仓库](https://github.com/karthiksoman/zebra-Llama)，并在实际应用中展示了增强的引用准确性。
- **关于神经元协调的新见解**：发表在 *Cell* 杂志上的研究揭示，神经元可以通过将 40-50% 的活动投入到单个任务中，同时保持团队协作来优化其工作。
  
  - 这种组织结构被发现在五个不同物种中是一致的，影响了我们对大脑效率的理解。
- **Chonkie：新型轻量级文本分块库**：**Chonkie** 是一个轻量级、高效的库，专为快速 RAG 文本分块而设计，使文本处理更加便捷。
  
  - 您可以在[此处](https://pypi.org/project/chonkie/)找到更多详细信息，并在 [GitHub](https://github.com/bhavnicksm/chonkie) 上查看仓库。
- **Lucid V1：实时 Minecraft 游戏模拟**：Rami 宣布发布 **Lucid V1**，这是一个能够在标准消费级硬件上实时模拟 Minecraft 环境的世界模型。
  
  - 在[此处](https://lucidv1-demo.vercel.app/)体验演示，并在 [Substack](https://ramimo.substack.com/p/lucid-v1-a-world-model-that-does) 上查看详细信息。
- **每周医疗 AI 亮点**：该播客介绍了 2024 年 11 月 2 日至 9 日期间顶尖的医疗 AI 研究论文，包括《探索用于专家级肿瘤护理的大语言模型》等著名作品。
  
  - 听众可以通过 [YouTube 链接](https://youtu.be/ad0uTnYuTo8)获取更多更新。

- [MSN](https://www.msn.com/en-us/news/news/content/ar-AA1tKYHk?ocid=sapphireappshare): 未找到描述
- [DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion](https://arxiv.org/abs/2411.04928): 在本文中，我们介绍了 **DimensionX**，这是一个旨在通过视频扩散（video diffusion）仅从单张图像生成逼真 3D 和 4D 场景的框架。我们的方法始于这样一个见解...
- [The Mystery of How Neurons Control The Brain Has Finally Been Solved](https://www.sciencealert.com/the-mystery-of-how-neurons-control-the-brain-has-finally-been-solved): 大脑是效率的奇迹，经过数千年的进化，使其能够适应并在快速变化的世界中蓬勃发展。
- [Alice In Wonderland Black Hole GIF - Alice In Wonderland Black Hole Falling - Discover & Share GIFs](https://tenor.com/view/alice-in-wonderland-black-hole-falling-bye-gif-13915543): 点击查看 GIF
- [Tweet from rami (@rami_mmo)](https://x.com/rami_mmo/status/1856028792407388360): 很高兴宣布 Lucid V1：一个可以在消费级硬件上实时模拟 Minecraft 环境的世界模型（world model）！🔥 在此游玩：https://lucidv1-demo.vercel.app/ 帖子：https://ramimo.substack.c...
- [Tweet from minhash (@BhavnickMinhas)](https://x.com/BhavnickMinhas/status/1855547848634323206): 🦛 介绍 Chonkie：一个严肃实用的 RAG 分块（chunking）库，轻量、极速，随时准备好对你的文本进行分块（CHONK）！🔗 https://pypi.org/project/chonkie/ 👩🏻‍💻 https://github.com/bh...
- [Tweet from Open Life Science AI (@OpenlifesciAI)](https://x.com/OpenlifesciAI/status/1855207141302473090>): 医学 AI 上周回顾：顶级研究论文/模型 🏅（2024年11月2-9日）🏅 本周医学 AI 论文：探索用于专家级肿瘤护理的大语言模型（Large Language Models）作者（@apalepu13 ,@v...
- [What We Learned About LLM/VLMs in Healthcare AI Evaluation:](https://huggingface.co/blog/shanchen/ai-in-medicine-eval2024) : 未找到描述
- [GitHub - shadowlamer/diffusezx: ZX-Spectrum inspired images generator](https://github.com/shadowlamer/diffusezx): 受 ZX-Spectrum 启发的图像生成器。通过在 GitHub 上创建账户来为 shadowlamer/diffusezx 的开发做出贡献。
- [The massed-spaced learning effect in non-neural human cells - Nature Communications](https://www.nature.com/articles/s41467-024-53922-x): 当学习在时间上拉开间隔时，记忆会得到增强，但到目前为止，这仅在神经系统中被观察到。在这里，作者展示了包括肾细胞在内的非神经细胞也表现出间隔效应...
- [Zebra-Llama: A Context-Aware Large Language Model for Democratizing Rare Disease Knowledge](https://arxiv.org/abs/2411.02657): 罕见病在医疗保健中提出了独特的挑战，通常面临诊断延迟和碎片化的信息格局。这些疾病中可靠知识的稀缺构成了明显的...
- [zebraLLAMA/zebra-Llama-v0.2 · Hugging Face](https://huggingface.co/zebraLLAMA/zebra-Llama-v0.2): 未找到描述
- [GitHub - karthiksoman/zebra-Llama](https://github.com/karthiksoman/zebra-Llama): 通过在 GitHub 上创建账户来为 karthiksoman/zebra-Llama 的开发做出贡献。
- [zebra-Llama/code/notebook/zebra_llama_v0.2_demo.ipynb at main · karthiksoman/zebra-Llama](https://github.com/karthiksoman/zebra-Llama/blob/main/code/notebook/zebra_llama_v0.2_demo.ipynb): 通过在 GitHub 上创建账户来为 karthiksoman/zebra-Llama 的开发做出贡献。

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1304784860472283229) (13 messages🔥):

> - `AI Safety Camp 10`
> - `Depth Estimation and Object Detection Performance` (深度估计与目标检测性能)
> - `Ollama Operator for LLM Deployment` (用于 LLM 部署的 Ollama Operator)
> - `Qwen2.5 Coder Performance` (Qwen2.5 Coder 性能)
> - `Enhancements to base64 API Requests` (base64 API 请求的增强)

- **AI Safety Camp 10 开放申请**：**AI Safety Camp** 的第 10 期已开始团队成员申请阶段，这是一个将于 2025 年 1 月开始的为期 3 个月的在线研究项目。感兴趣的人士可以在 11 月 17 日之前通过其[网站](https://aisafety.camp)查找项目并申请。
  
  - 该训练营将涵盖广泛的主题，鼓励来自不同背景的参与者申请。
- **深度估计流水线实现 4ms 推理**：一个新开发的流水线为**深度估计和目标检测**提供了低于 **4ms 的推理**速度，针对实时应用中的速度和准确性进行了优化。有关此成果的更多细节可以在[博文](https://medium.com/predict/how-i-achieved-4ms-depth-object-detection-and-what-i-built-with-it-246849007223)中找到。

- 该项目是作者之前 **DEPTHS** 模型工作的延续，进一步增强了性能指标和实际应用。
- **Ollama Operator 简化了 LLM 部署**：**Ollama Operator** 仅需几行 YAML 配置即可简化并加速 Ollama 实例和 LLM 服务器的部署。该项目最近在 KubeCon 上展出，详细录像可在[此处](https://www.youtube.com/watch?v=XWjZQfSXKDg)查看。
  
  - 该 Operator 已开源，允许用户轻松设置和管理自己的 LLM 部署。
- **Qwen2.5 Coder 性能超出预期**：在测试中，**Qwen2.5 Coder 32B** 的表现优于 **GPT4o** 和 **Claude 3.5 Sonnet**，展示了其在代码生成任务中的能力。详细的对比和性能见解可以在 [YouTube 视频](https://youtu.be/Xs0EkLYu6hw)中查看。
  
  - 鼓励用户探索 Hugging Face 上提供的全新 GGUF 模型集合以进行进一步应用。
- **Base64 API 集成改进**：某库的更新增强了 **base64 实现**，允许在不向 Hugging Face API 传递图像 URL 的情况下发起 API 请求。此功能的详细信息见[发布说明](https://github.com/not-lain/loadimg/releases/tag/v0.3.3)。
  
  - 这些改进有助于将模型更轻松地集成到应用程序中，简化了开发流程。

**提到的链接**：

- [Audio Lyrics Extractor - eyov 的 Hugging Face Space](https://huggingface.co/spaces/eyov/LyricExtractor)：未找到描述
- [PyTorchModelHubMixin: 弥合 Hugging Face 上自定义 AI 模型的差距](https://huggingface.co/blog/not-lain/building-hf-integrated-libraries)：未找到描述
- [rwitz/cat1.0 · Hugging Face](https://huggingface.co/rwitz/cat1.0)：未找到描述
- [breadlicker45/bread-tv2o-medium · Hugging Face](https://huggingface.co/breadlicker45/bread-tv2o-medium)：未找到描述
- [groq gradio 作为桌面应用](https://gist.github.com/Getty/0bb02952a2fff2c89d92bdac0405e9bd)：groq gradio 作为桌面应用。GitHub Gist：即时分享代码、笔记和片段。
- [Volko76 (Volko)](https://huggingface.co/Volko76)：未找到描述
- [GitHub - skirdey/boss: 用于攻防安全的 Multi-Agent 操作系统](https://github.com/skirdey/boss)：用于攻防安全的 Multi-Agent 操作系统。通过在 GitHub 上创建账号为 skirdey/boss 的开发做出贡献。
- [AI Safety Camp 10 — AI Alignment 论坛](https://www.alignmentforum.org/posts/57wx7B3GQavvKkPne/ai-safety-camp-10#:~:text=(11)%20Agency%20Overhang%20as%20a%20Proxy%20for%20Sharp%20Left%20Turn)：我们很高兴地宣布，第 10 版 AI Safety Camp 现已进入团队成员申请阶段！…
- [Qwen2.5 Coder 32B vs GPT4o vs Claude 3.5 Sonnet (新)](https://youtu.be/Xs0EkLYu6hw)：让我们看看哪个模型最好
- [我如何实现 4ms 深度和对象检测 —— 以及我用它构建了什么](https://medium.com/predict/how-i-achieved-4ms-depth-object-detection-and-what-i-built-with-it-246849007223)：在不到一秒的时间内引导、检测并赋能盲人的 AI。实时、直观、改变生活。
- [用于人类辅助的深度估计和邻近追踪 [检测 + 深度 + 文本生成]](https://youtu.be/Qt_lQBihyWg)：对来自 Pexels 或 Unsplash 等平台的视频剪辑进行预处理，这是我们的 DEPTHS 模型如何工作的推理预览。你会注意到物体...
- [Ollama Operator](https://ollama-operator.ayaka.io/)：未找到描述
- [GitHub - nekomeowww/ollama-operator: 又一个在 Kubernetes 上轻松运行大语言模型的 Operator。由 Ollama 驱动！🐫](https://github.com/nekomeowww/ollama-operator)：又一个在 Kubernetes 上轻松运行大语言模型的 Operator。由 Ollama 驱动！🐫 - nekomeowww/ollama-operator
- [不再需要运行时设置！让我们无缝地捆绑、分发、部署、扩展 LLM... - Fanshi Zhang](https://www.youtube.com/watch?v=XWjZQfSXKDg)：不要错过！参加我们下一届旗舰会议：2024 年 11 月 12 日至 15 日在盐湖城举行的 KubeCon + CloudNativeCon 北美站。与...建立联系

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1304673265540665445) (2 条消息):

> - `微调 MobileNet 用于人脸检测`
> - `用于 3D 图像分类的模型`

- **寻求 MobileNet 微调资源**：一位成员询问了专门针对**人脸检测**和**人脸识别**任务微调 **MobileNet** 的资源。
  
  - 讨论中未提供具体资源。
- **需要 3D 图像分类模型**：另一位成员寻求适用于 **3D 图像分类**的模型建议，其数据集由 **.obj** 和 **.mtl** 格式的图像组成。
  
  - 他们提到已经准备好了图像类别，但没有收到任何直接建议。

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1304757538637611028) (7 条消息):

> - `评估 TinyLlama 的 Text to SQL 性能`
> - `沃洛夫语 (Wolof) 的 NLP 与语音合成`
> - `语言检测方法`
> - `LangChain SQL Agent 的评估指标`

- **评估 TinyLlama 的 Text to SQL 查询能力**：一位成员分享了他们针对 Text to SQL 查询微调（finetuned）了 **TinyLlama** 的经历，并正在寻求评估该模型的方法。
  
  - 另一位成员建议探索专门用于 Sql-Eval 的 **HF spaces**，并建议搜索“sql eval”。
- **沃洛夫语 NLP 与语音合成合作**：一位成员表示有兴趣在针对塞内加尔**沃洛夫语 (Wolof)** 的 NLP 和语音合成工作上进行合作。
  
  - 目前尚未提供关于此项合作的更多细节或回复。
- **轻量级英文检测**：一位成员询问了用于检测英文输入的**最轻量且离线**的方法。
  
  - 他们专门在寻找一种能够识别是否存在英文内容的解决方案。
- **简化 LangChain SQL Agent 的评估指标**：一位成员正尝试为其 **LangChain SQL agent** 代码寻找更简单的评估指标，理由是目前各种选项（如 Agent 轨迹评估）过于复杂。
  
  - 他们正在寻求资源、方法和参考资料，包括 YouTube 视频或 Python 代码示例，以简化评估流程。
- **寻求 LangChain SQL Agent 方面的帮助**：同一位成员继续寻求帮助，并提到自己在该领域缺乏知识。
  
  - 他们希望从任何之前从事过 LangChain SQL agent 评估工作的人那里获得见解。

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1304708063294652426) (6 条消息):

> - `Diffusion Models 学习小组`
> - `随机过程的学习资源`
> - `Retake 摄影应用讨论`
> - `Gemini Nano 优化模块问题`
> - `全栈 (Fullstack) vs AI 栈指导`

- **询问 Diffusion Models 学习小组**：一位成员询问是否存在现有的 **Diffusion Models** 学习小组，表示对协作学习感兴趣。
  
  - 另一位成员给出了不确定的回答，暗示可能不存在此类小组。
- **寻求随机过程资源**：一位成员寻求关于随机过程和与 Diffusion Models 相关的 SDEs（随机微分方程）的**书籍或课程**推荐。
  
  - 这反映了他们希望加强对相关理论基础理解的主动性。
- **关于“Retake”摄影应用的讨论**：一位用户将 **“Retake”** 应用描述为一款开创性的工具，能够逼真地重构照片，毫不费力地增强普通照片的效果。
  
  - 他们表达了希望了解这款创新应用背后所使用模型的愿望。
- **更新 Gemini Nano 优化模块面临挑战**：一位成员报告了在遵循 **Gemini Nano** 设置指南时遇到的困难，特别是在更新优化模块方面。
  
  - 他们提供了系统规格，包括使用 **Arch Linux** 以及在不同浏览器上的尝试。
- **全栈开发与 AI 栈的咨询**：一位在**经典机器学习算法**和 PyTorch 方面有经验的成员被建议在深入研究 AI 之前考虑全栈开发。
  
  - 他们对时间投入以及追求全栈是否值得表示担忧。

**提到的链接**：[Retake AI: Face & Photo Editor - Google Play 应用](https://play.google.com/store/apps/details?id=com.codespaceapps.you&hl=en&pli=1)：未找到描述。

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1304551709649010698) (628 条消息🔥🔥🔥):

> - `Qwen Coder 发布`
> - `语言模型微调`
> - `游戏视角`
> - `Cloudflare 隧道解决方案`
> - `AI 项目协作资源`

- **对 Qwen Coder 发布的兴奋**：成员们对 Qwen 2.5-Coder-32B 的发布感到高兴，称其表现令人印象深刻，甚至超过了之前的模型。
  
  - 预计该模型将为寻求强大语言模型（LLM）的用户增强编码能力。
- **关于语言模型的讨论**：用户讨论了 BGE-Large 和 BGE-M3 之间的区别，指出后者因其多语言能力在基准测试中表现良好。
  
  - 对话强调了根据用户需求（特别是语言处理需求）选择模型的重要性。
- **游戏与开发视角**：小组分享了对游戏文化的看法，提到个人爱好如何影响专业工作，并对游戏偏好随时间趋于成熟达成了共识。

- 成员们一致认为，为了保持生产力和享受生活，有必要将工作与休闲分开。
- **利用 Cloudflare 进行隧道传输**：建议使用 Cloudflare 隧道作为 Gradio 的替代方案，以便在受限地区共享模型，并强调了其在暴露本地服务器方面的有效性。
  
  - 提供了设置和利用 Cloudflare 进行隧道传输的步骤，以帮助面临访问问题的用户。
- **协作式 AI 项目资源**：用户分享了在不依赖大型框架的情况下构建封装函数（wrapper functions）的资源，并反思了以往编程经验中的教训。
  
  - 社区鼓励采用自托管方案和定制化，以简化 AI 模型的部署流程。

**提到的链接**：

- [mergekit-gui - a Hugging Face Space by arcee-ai](https://huggingface.co/spaces/arcee-ai/mergekit-gui)：未找到描述
- [On-demand deployments - Fireworks AI Docs](https://docs.fireworks.ai/guides/ondemand-deployments)：未找到描述
- [Reward Modelling - DPO, ORPO & KTO | Unsloth Documentation](https://docs.unsloth.ai/basics/reward-modelling-dpo-orpo-and-kto)：要在 Unsloth 中使用 DPO, ORPO 或 KTO，请遵循以下步骤：
- [Use model after training](https://huggingface.co/docs/trl/en/use_model)：未找到描述
- [How to Merge Fine-tuned Adapter and Pretrained Model in Hugging Face Transformers and Push to Hub?](https://stackoverflow.com/questions/77164963/how-to-merge-fine-tuned-adapter-and-pretrained-model-in-hugging-face-transformer)：我按照 llama-recipes 仓库的教程微调了 Llama-2 模型。目前，我的预训练模型和微调后的 adapter 分别存储在两个不同的目录中，如下所示：...
- [Hackerman GIF - Hacker Hackerman Kung Fury - Discover & Share GIFs](https://tenor.com/view/hacker-hackerman-kung-fury-gif-7953536)：点击查看 GIF
- [Aya Expanse: Connecting Our World](https://cohere.com/blog/aya-expanse-connecting-our-world)：Cohere For AI 发布了 Aya Expanse，这是一个最先进的多语言模型系列，旨在通过 AI 缩小语言差距。
- [Tobias Tobias Funke GIF - Tobias Tobias Funke - Discover & Share GIFs](https://tenor.com/view/tobias-tobias-funke-gif-23255404)：点击查看 GIF
- [Uploading a custom model - Fireworks AI Docs](https://docs.fireworks.ai/models/uploading-custom-models)：未找到描述
- [Tweet from Daniel Han (@danielhanchen)](https://x.com/danielhanchen/status/1854992153992479165)：我对 "LoRA vs full finetuning: An illusion of equivalence" 的看法。TLDR 1. 使用 alpha = 2\*rank 2. 不要使用太小的 rank (rank=1 到 8) 3. 标题很标题党。更好的标题应该是 "LoRA 有效..."
- [Unsloth Fixing Gemma bugs](https://unsloth.ai/blog/gemma-bugs)：Unsloth 修复 Google 的开源语言模型 Gemma 的 Bug。
- [Wow Meme Wow GIF - Wow Meme Wow Wink - Discover & Share GIFs](https://tenor.com/view/wow-meme-wow-wink-gif-5435391)：点击查看 GIF
- [Instantiating a big model](https://huggingface.co/docs/transformers/v4.24.0/en/big_models)：未找到描述
- [How to Merge Fine-tuned Adapter and Pretrained Model in Hugging Face Transformers and Push to Hub?](https://stackoverflow.com/questions/77164963/how-to-merge-fi)：我按照 llama-recipes 仓库的教程微调了 Llama-2 模型。目前，我的预训练模型和微调后的 adapter 分别存储在两个不同的目录中，如下所示：...
- [Build a Retrieval Augmented Generation (RAG) App | 🦜️🔗 LangChain](https://python.langchain.com/docs/tutorials/rag/)：LLM 实现的最强大应用之一是复杂的问答 (Q&A) 聊天机器人。这些应用可以针对特定的源信息回答问题。这些...
- [ORPO Trainer](https://huggingface.co/docs/trl/main/en/orpo_trainer)：未找到描述
- [EASIEST Way to Fine-Tune LLAMA-3.2 and Run it in Ollama](https://www.youtube.com/watch?v=YZW3pkIR-YE)：Meta 最近发布了 Llama 3.2，此视频演示了如何使用 Unsloth 微调 30 亿参数的指令模型，并在本地使用 Ollama 运行...
- [Beyond Fine-Tuning: Merging Specialized LLMs Without the Data Burden](https://towardsdatascience.com/beyond-fine-tuning-merging-specialized-llms-without-the-data-burden-1c449c2060c4)：从 Model Soup 到自动进化合并：利用专业化 LLM 融合来减少数据需求并消除密集的……
- [Release v0.12.0 · huggingface/trl](https://github.com/huggingface/trl/releases/tag/v0.12.0)：主要变更和破坏性变更。Online DPO 的通用奖励模型支持。Online DPO 最初仅支持与训练模型具有相同 Tokenizer 和聊天模板的奖励模型。现在，你...
- [Tags · qwen2.5-coder](https://ollama.com/library/qwen2.5-coder/tags)：最新的代码专用 Qwen 模型系列，在代码生成、代码推理和代码修复方面有显著改进。

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1304688176144842844) (11 messages🔥):

> - `LMSYS Blueberry 上的匿名 LLM`
> - `重大发布推测`
> - `关于 'blueberry' 的文字游戏`
> - `LLM 数据集多样性`
> - `Qwen 对 O1 的修订版`

- **匿名 LLM 推测：Blueberry**：关于 [LMSYS Blueberry 上新的匿名 LLM](https://www.reddit.com/r/LocalLLaMA/comments/1gnlspj/new_anonymous_llm_on_lmsys_blueberry/) 的讨论非常热烈，这可能预示着一个重大的发布即将到来。
  
  - *“有谁猜猜看？”* 引发了讨论，成员们思考了这一潜在公告的影响。
- **关于 Qwen 修订 O1 的推测**：成员们推测即将发布的版本可能与 *Qwen 对 O1 的修订* 有关，正如一位用户幽默地假设的那样。
  
  - 另一位成员评论道：*“如果是真的，请引用我”*，表达了对这一理论得到证实的渴望。
- **文字游戏：从 Blueberry 到 Gemini 2.0**：一位成员开玩笑地打乱了 “blueberry” 一词的字母，暗示它可能会指向 “Gemini 2.0”，为对话增添了趣味性。
  
  - 这个文字游戏引起了其他成员的兴趣，并引发了一系列富有创意的推测。
- **对 LLM 数据集的担忧**：对于 LLM 数据集的内容存在一些挫败感，一位成员发誓说 *“他们简直把任何东西都往里塞”*。
  
  - 作为回应，另一位成员指出该数据集看起来 *“非常……多样化！”*，突显了对数据集质量截然不同的看法。

 

**提到的链接**：[Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gnlspj/new_anonymous_llm_on_lmsys_blueberry/)：未找到描述

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1304535983974383730) (68 messages🔥🔥):

> - `将 Ollama 与前端解决方案集成`
> - `Fine-tuning Llama 3 的性能表现`
> - `任务生成的数据集准备`
> - `聊天模型中 Function Calling 的挑战`
> - `改进模型训练策略`

- **将 Ollama 与前端解决方案集成**：成员们讨论了在终端运行 Ollama 并使用 Streamlit 而非 Web UI 创建聊天 UI 的可能性，确认通过使用 Ollama API 是可行的。
  
  - 一位成员对这些信息表示感谢，表示打算进一步阅读关于 Ollama API 的内容。
- **Fine-tuning Llama 3 的性能表现**：一位成员注意到他们的 Fine-tuned Llama 3 模型与原始模型相比 Inference 时间更慢，引发了关于模型配置中潜在问题的讨论。
  
  - 建议包括确保一致的浮点精度（float precision）以及检查脚本中与 Inference 速度相关的问题。
- **任务生成（Quest Generation）的数据集准备**：一位成员寻求指导，希望高效地将其大量的文学数据集转换为适用于 Llama-3.2 Fine-tuning 的格式，重点在于任务生成。
  
  - 另一位成员通过 Discord 提供了帮助，展示了社区在解决数据集挑战方面的支持。
- **聊天模型中 Function Calling 的挑战**：一位用户分享了他们在调用函数时训练聊天模型的方法，并对模型是否学会了适当地使用工具有所顾虑。
  
  - 讨论围绕模型在训练数据集中关于多条助手消息与单次 Function Calling 的学习机制展开。
- **改进模型训练策略**：一位拥有 1500 个文学段落的成员质疑其模型的 Learning Rate，并建议调整 Learning Rate Schedules 以优化训练结果。
  
  - 社区的回应鼓励探索不同的 Learning Rate 策略，并解决训练过程中潜在的低效问题。

 

**提到的链接**：[Errors | Unsloth Documentation](https://docs.unsloth.ai/troubleshooting/errors)：要修复设置中的任何错误，请参阅下文：

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**showcase**](https://discord.com/channels/1179035537009545276/1179779344894263297/1305548247728656436) (5 messages):

> - `YouTube Tutorials`
> - `Integration Discussions`

- **ChatGPT 4o 和 o1 Preview 详解**：一段名为 [“如何免费使用 ChatGPT 4o 和 o1 preview”](https://www.youtube.com/watch?v=fKLtYRyd128) 的 YouTube 视频提供了利用这些功能的见解，并由一个提供无限图像生成的服务支持。
  
  - 该服务可以通过 [NexusAI](https://www.nexusapi.tech/NexusAI) 访问，推广 AI 驱动的聊天体验。
- **使用 FLUX 解锁图像生成**：另一段名为 [“如何免费使用 FLUX 1.1 PRO ultra, SD 3.5 LARGE, Recraft V3！”](https://www.youtube.com/watch?v=J7X6AXfmb6o) 的视频指导观众如何利用这些工具，并附带了图像生成器的链接。
  
  - 感兴趣的用户可以探索 [这个图像生成器](https://image.nexusapi.tech) 并通过 [Discord](https://discord.com/invite/sk8eddGwmP) 加入社区。
- **集成洽谈公开邀请**：向有意向的各方发出了讨论集成机会的邀请，预约链接见 [此处](https://scheduler.zoom.us/gabriel-peracio/cto)。
  
  - *“虽然我们不会分享我们的核心秘诀 (secret sauce)”*，但欢迎围绕合作进行对话。
- **鼓励非正式交流**：该成员鼓励非正式对话，表示可以通过私信或在指定频道进行讨论。
  
  - 他们提到回复可能会有延迟，但承诺会响应召唤。

**提到的链接**：

- [Zoom Scheduler](https://scheduler.zoom.us/gabriel-peracio/cto)：未找到描述
- [How to Use ChatGPT 4o and o1 preview Free](https://www.youtube.com/watch?v=fKLtYRyd128)：🛠️ 站点链接：https://www.nexusapi.tech/NexusAI 是您通往无限图像生成和 AI 驱动聊天体验的门户！我们提供访问切入...
- [How to use FLUX 1.1 PRO ultra, SD 3.5 LARGE, Recraft V3 for FREE!](https://www.youtube.com/watch?v=J7X6AXfmb6o)：⭐ 尝试图像生成器：https://image.nexusapi.tech/ 加入 Discord：https://discord.com/invite/sk8eddGwmP 免费 AI 图像生成器，探索解锁秘密...

---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/1305207487137452115) (1 messages):

> - `AI research collaboration`
> - `Multimodal machine learning`
> - `Autonomous AI agents`
> - `Knowledge graphs`
> - `Reinforcement learning`

- **寻求 AI 研究伙伴**：一位来自亚美尼亚的资深数据科学家渴望在 **Multimodal machine learning**、**Autonomous AI agents** 及其他课题上与研究人员进行**合作**，旨在为 **2026 年 12 月**重新申请博士学位积累研究经验。
  
  - 他们在**学术环境**中感到如鱼得水，并考虑联系正在阅读的论文作者以寻求潜在合作，但不确定这是否是个好主意。
- **职业反思与目标**：凭借超过 **4 年**的经验和两个硕士学位，该成员反思了自己的**职业历程**，并旨在进一步深入有意义的 **AI 研究**。
  
  - 他们表达了发表论文以增强未来博士申请竞争力的强烈愿望。
- **研究社交中的挑战**：这位数据科学家很难在**亚美尼亚**找到专注于前沿 AI 课题的同行研究者，强调了与志同道合的人建立联系的必要性。
  
  - 他们分享了在**建立人脉 (networking)** 方面的挑战，并征求关于如何有效联系该领域教授和研究人员的建议。

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1304601341880041504) (17 条消息🔥):

> - `Transformers vs 其他模型`
> - `Liquid AI 被热炒`
> - `AI 中的微分方程`
> - `对闭源的担忧`
> - `新兴 AI 研究`

- **Transformers 可能不再是全部**：成员们讨论了除了 **Transformer models** 之外，是否还有其他模型可以使用 Unsloth 进行训练，并提到了 **RNNs** 和 **CNNs** 的存在，但明确表示目前尚不支持常规神经网络。
  
  - 一位成员指出，“以前确实如此！”，这表明人们对 Transformer models 重要性的看法发生了转变。
- **Liquid AI 面临社区质疑**：一位成员对 **liquid.ai** 正在开发的内容表示感兴趣，认为基于 **differential equations** 的模型可能大有前途，但其他人对其可信度提出了担忧，称其为“伪科学 (sudo science)”。
  
  - 批评者指出，liquid.ai 的产品是 **closed source** 的，导致无法验证其说法，因此对行业的贡献微乎其微。
- **新兴研究引发褒贬不一的反应**：分享了一个研究论文链接 [arxiv.org/pdf/2410.10630](https://arxiv.org/pdf/2410.10630)，引发了围绕新 AI 进展的讨论。
  
  - 一位成员提到已经研究类似课题一年了，对该领域日益增长的兴趣感到兴奋。

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1304569083475267615) (353 条消息🔥🔥):

> - `Open Hermes 2.5 Mix`
> - `Qwen Coder Models`
> - `Inference Scaling in AI`
> - `TeeHee Bot Functionality`
> - `Contribution to Open Source Projects`

- **Open Hermes 2.5 Mix 详解**：在 Open Hermes 2.5 mix 中加入代码数据被强调为一项重大改进，增加了模型的复杂性和功能性。
  
  - 团队对该混合模型的探索旨在增强模型在各种应用中的能力。
- **Qwen Coder 模型介绍**：Qwen2.5-Coder 系列的发布展示了多种不同尺寸的 Coder 模型，在基准测试中表现出色。
  
  - 值得注意的是，据报告，其旗舰模型在基准评估中超越了多个私有模型。
- **Inference Scaling 的挑战**：最近的讨论集中在当前 AI 模型 Inference Scaling 方法的局限性上，特别是正如一些知名文章所报道的那样。
  
  - 关于生成式 AI 改进速度放缓的担忧，引发了对未来方向和策略的反思。
- **TeeHee Bot 功能问题**：TeeHee Bot 目前在生成回复后无法发布，这引起了用户的关注。
  
  - 官方承认发布功能存在 Bug，并表示正在持续努力改进 Bot 的性能。
- **鼓励开源贡献**：成员们分享了多个与 AI 相关的开源项目，鼓励大家参与贡献，促进社区互动。
  
  - ShareGPT-Builder 和 LLM-Logbook 等项目被重点推荐为贡献者参与的机会。

**提到的链接**：

- [tinygrad: A simple and powerful neural network framework](https://tinygrad.org/#tinybox)：未找到描述
- [来自 Qwen (@Alibaba_Qwen) 的推文](https://x.com/Alibaba_Qwen/status/1856040217897251044)：🚀时刻已到，11 月 11 日 10:24！发布我们史上最强 Coder 模型的完美时刻！Qwen2.5-Coder-32B-Instruct！等等……它不仅仅是一个大型 Coder！它是一个 Coder 模型家族！此外……
- [Forge Reasoning API by Nous Research](https://forge.nousresearch.com/)：Nous Research 提供的 Forge Reasoning API
- [来自 undefined 的推文](https://x.com/tee_hee_he)：未找到描述
- [Funny Big GIF - Funny Big Lebowski - Discover & Share GIFs](https://tenor.com/view/funny-big-lebowski-gif-thedude-gif-24340964)：点击查看 GIF
- [Federal Investigation GIF - Federal Investigation - Discover & Share GIFs](https://tenor.com/view/federal-investigation-gif-22271245)：点击查看 GIF
- [购买搭载 M4 Pro 芯片的 Mac mini](https://www.apple.com/us-edu/shop/buy-mac/mac-mini/m4-pro)：搭载 M4 和 M4 Pro 芯片的 Mac mini。专为 Apple Intelligence 设计。配备前后端口。折抵符合条件的 Mac 可获得折抵金额。立即购买。
- [Love Languages GIF - Love Languages Pea - Discover & Share GIFs](https://tenor.com/view/love-languages-pea-chu-gif-6848966838493457121)：点击查看 GIF
- [GitHub - cameronaaron/Geminio1](https://github.com/cameronaaron/Geminio1/)：通过在 GitHub 上创建账号来为 cameronaaron/Geminio1 的开发做贡献。
- [GitHub - NousResearch/nousflash-agents: Modular Agentic AI Architecture - NousResearch x Teleport (Flashbots)](https://github.com/NousResearch/nousflash-agents)：模块化 Agentic AI 架构 - NousResearch x Teleport (Flashbots) - NousResearch/nousflash-agents
- [GitHub - teknium1/ShareGPT-Builder](https://github.com/teknium1/ShareGPT-Builder)：通过在 GitHub 上创建账号来为 teknium1/ShareGPT-Builder 的开发做贡献。
- [GitHub - teknium1/LLM-Logbook: Public reports detailing responses to sets of prompts by Large Language Models.](https://github.com/teknium1/LLM-Logbook)：详细记录 Large Language Models 对各组提示词响应的公开报告。- teknium1/LLM-Logbook
- [Spontex Hedgehog GIF - Spontex Hedgehog Washup - Discover & Share GIFs](https://tenor.com/view/spontex-hedgehog-washup-love-amor-gif-24459480)：点击查看 GIF
- [GitHub - teknium1/alpaca-roleplay-discordbot: A discord bot that roleplays!](https://github.com/teknium1/alpaca-roleplay-discordbot)：一个可以进行角色扮演的 Discord Bot！通过在 GitHub 上创建账号来为 teknium1/alpaca-roleplay-discordbot 的开发做贡献。
- [无标题](https://github.com/state-spaces/mamba)：未找到描述

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1304808860284092437) (97 条消息🔥🔥):

> - `Benchmarking LLMs`
> - `Fine-tuning Techniques`
> - `Model Performance Analysis`
> - `Code Generation Advances`
> - `System Prompts in LLMs`

- **对 LLM Benchmarks 的担忧**：成员们对当前 **benchmarks** 的相关性表示怀疑，指出多选题设置中的表面变化可能会大幅改变模型排名。
  
  - 讨论强调，评估 LLM 的挑战源于 overfitting 以及 benchmarks 对模型性能的影响。
- **提出的创新 Fine-tuning 策略**：一位成员询问了在 Llama 模型中集成新层进行 fine-tuning 的最佳方法，特别是为了处理不同的 embeddings。
  
  - 建议包括先冻结现有层的参数，然后逐渐取消冻结（unfreezing），以提高训练效率。
- **代码生成模型的进展**：大家对 **OpenCoder** 的发布感到兴奋，这是一个旨在通过广泛且透明的数据集提升代码生成的开源项目。
  
  - 成员们注意到针对特定代码的 LLM 进展迅速，使得开发复杂项目无需直接编辑源代码。
- **LLM 回复特性的审查**：讨论显示，像 **Sonnet** 这样的模型变得更加具有自我反思性，会根据用户的质疑调整回答以改善交互。
  
  - 也有人担心这种向个性化转变的趋势可能会影响各种模型的 benchmark 评估。
- **探索新的 Thinking Token 概念**：一位成员提出了实现一种特殊的 **'thinking' token** 的想法，以在不产生输出 tokens 的情况下增强 LLM 计算。
  
  - 这可能允许更高效地处理中间表示，从而扩展模型的计算能力。

**提到的链接**：

- [Latent Space Explorer](https://cpldcpu.github.io/LatentSpaceExplorer/)：未找到描述
- [OpenCoder: Top-Tier Open Code Large Language Models](https://opencoder-llm.github.io/)：未找到描述

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1304570407466045551) (9 条消息🔥):

> - `Unit Test Generation Fine-Tuning`
> - `Test-Time Scaling in AI`
> - `Multimodal Retrieval in AI`
> - `Large Language Models Unlearning`
> - `Medical AI Innovations`

- **针对单元测试的 Parameter-Efficient Fine-Tuning**：一名成员分享了关于 **Parameter-Efficient Fine-Tuning of Large Language Models for Unit Test Generation** 研究的见解，强调了该论文中的实证结果，详见 [此处](https://arxiv.org/pdf/2411.02462)。
  
  - 该研究侧重于优化 LLMs 以更有效地生成单元测试，表明这种方法可以显著简化测试流程。
- **Test-Time Scaling 推测**：由 **Sasha Rush** 在康奈尔大学主讲的题为 *Speculations on Test-Time Scaling* 的讲座现已在 [YouTube](https://www.youtube.com/live/6fJjojpwv1I?si=6byPStsGqUHSK0qP) 上线。
  
  - 本讲座深入探讨了 Test-Time 阶段 Scaling 的细微差别，特别是与模型性能提升相关的方面。
- **多模态检索的进展**：最近的一篇论文介绍了 **universal multimodal retrieval** 的创新，利用 multimodal LLMs 来适应多样化的检索任务，旨在克服模态偏差（PDF 见 [此处](https://arxiv.org/abs/2411.02571)）。
  
  - 研究结果提出了一些新技术，如 modality-aware hard negative mining，以提高跨不同数据形式的检索性能。
- **语言模型中的 Unlearning**：一项研究对当前 LLMs 中 Unlearning 方法的有效性提出了质疑，认为这些方法往往无法彻底擦除不需要的知识（论文详情见 [此处](https://arxiv.org/abs/2410.16454)）。
  
  - 他们的结果表明，Quantization 技术可能会无意中保留已遗忘的信息，因此呼吁改进 Unlearning 策略。
- **医疗 AI 创新**：一份关于 **Medical AI** 最新趋势的全面概述，重点介绍了过去一周在患者护理和诊断方面取得进展的各种研究论文和模型。
  
  - 值得关注的包括用于患者支持的 **CataractBot** 和用于知识增强型医疗问答的 **MEG**，展示了该领域的重大贡献。

**提到的链接**：

- [MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs](https://arxiv.org/abs/2411.02571)：最先进的检索模型通常处理简单的搜索场景，其中检索任务是固定的（例如，寻找段落来回答特定问题），且仅涉及单一模态...
- [Does your LLM truly unlearn? An embarrassingly simple approach to recover unlearned knowledge](https://arxiv.org/abs/2410.16454)：大语言模型（LLMs）在文本生成方面表现出卓越的能力，这得益于在海量文本语料库上的广泛训练。然而，LLMs 也可能从中习得不需要的行为...
- [What Matters in Transformers? Not All Attention is Needed](https://arxiv.org/abs/2406.15786)：虽然扩展基于 Transformer 的大语言模型（LLMs）在各项任务中展现了充满前景的性能，但它也引入了冗余架构，给效率带来了挑战...
- [来自 Open Life Science AI (@OpenlifesciAI) 的推文](https://x.com/OpenlifesciAI/status/1855207141302473090>)：医疗 AI 上周回顾：顶级研究论文/模型 🏅（2024年11月2-9日）🏅 本周医疗 AI 论文：探索用于专家级肿瘤护理的大语言模型。作者（@apalepu13 ,@v...
- [Speculations on Test-Time Scaling | Richard M. Karp Distinguished Lecture](https://www.youtube.com/live/6fJjojpwv1I?si=6byPStsGqUHSK0qP)：Sasha Rush（康奈尔大学）https://simons.berkeley.edu/events/speculations-test-time-scaling-richard-m-karp-distinguished-lecture Richard M. Karp Distingu...
- [GitHub - srush/awesome-o1: A bibliography and survey of the papers surrounding o1](https://github.com/srush/awesome-o1)：关于 o1 相关论文的文献目录和综述 - srush/awesome-o1

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1304668229733453844) (2 条消息):

> - `NVIDIA MM-Embed`
> - `Lucid for Minecraft Emulation`

- **NVIDIA 推出用于多模态检索的 MM-Embed**：NVIDIA 发布了 MM-Embed，这是**首个多模态检索器**，在**多模态 M-BEIR 基准测试**中取得了 SOTA 结果。您可以在[此处](https://www.marktechpost.com/2024/11/06/nvidia-ai-introduces-mm-embed-the-first-multimodal-retriever-achieving-sota-results-on-the-multimodal-m-beir-benchmark/?amp)的帖子中找到更多详情。
  
  - 据报道，这一进展通过整合视觉和文本信息，增强了跨各种数据类型的检索能力。
- **Rami 发布用于实时 Minecraft 模拟的 Lucid**：Rami 展示了 **Lucid V1**，这是一个能够在消费级硬件上实时模拟 **Minecraft** 环境的世界模型（World Model）。您可以在[此处](https://lucidv1-demo.vercel.app/)进行体验，并在[帖子](https://ramimo.substack.com/p/lucid-v1-a-world-model-that-does)中阅读更多内容。
  
  - 该项目的仓库已在 [GitHub](https://github.com/SonicCodes/lucid-v1) 上线，展示了其创造创新游戏体验的能力。

 

**提到的链接**：[来自 rami (@rami_mmo) 的推文](https://x.com/rami_mmo/status/1856028792407388360)：很高兴宣布 Lucid V1：一个可以在消费级硬件上实时模拟 Minecraft 环境的世界模型！🔥 在这里玩：[https://lucidv1-demo.vercel.app/](https://lucidv1-demo.vercel.app/) 帖子：[https://ramimo.substack.c](https://ramimo.substack.c)...

 

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1304570407466045551) (9 条消息🔥):

> - `Transformer Optimization`
> - `Unit Test Generation with LLMs`
> - `Multimodal Retrieval`
> - `Test-Time Scaling Insights`
> - `Machine Unlearning Mechanisms`

- **Transformer Optimization 技术**：研究了 **Mixture-of-Transformers** 和 **LoRA** 等创新方法，以提高语言模型训练的效率，暗示了训练方法的根本性转变。
  
- **使用 LLMs 进行 Unit Test Generation 的进展**：最近的一项研究强调了一种针对大型语言模型的 **parameter-efficient fine-tuning** 方法，专注于生成单元测试，实证结果显示出显著的效率。
  
  - 这种方法可能会**彻底改变测试流程**，通过更好的自动化测试增强软件的可靠性。
- **探索 Multimodal Retrieval 技术**：**MM-Embed** 模型展示了在 **universal multimodal retrieval** 方面的进展，在解决现有模型中的模态偏差的同时，适应了多种查询类型。
  
  - 与之前的模型相比，**Fine-tuning** 在各种检索基准测试中表现出了更高的性能。
- **来自 “Speculations on Test-Time Scaling” 的见解**：**Sasha Rush** 最近的一次讲座讨论了机器学习背景下关于 **test-time scaling** 的有趣理论，引发了对新型可扩展方法的兴趣。
  
  - 这次演讲的见解可能会推动 AI 系统在**适应性**和性能方面的进步。
- **LLMs 中的 Machine Unlearning 机制**：研究提出了关于现有的 **unlearning methods** 是真正擦除了知识还是仅仅将其隐藏的新发现，强调了在 LLMs 中进行有效 **unlearning** 的重要性。
  
  - 实验表明，**quantization** 技术可能会意外地**恢复模型中遗忘的知识**，这对当前的 **unlearning** 基准测试提出了挑战。

**提到的链接**：

- [来自 Open Life Science AI (@OpenlifesciAI) 的推文](https://x.com/OpenlifesciAI/status/1855207141302473090>): 上周 Medical AI 动态：顶级研究论文/模型 🏅 (2024年11月2-9日) 🏅 本周 Medical AI 论文：探索用于专家级肿瘤护理的大型语言模型 作者(@apalepu13 ,@v...
- [MM-Embed: Universal Multimodal Retrieval with Multimodal LLMs](https://arxiv.org/abs/2411.02571): 最先进的检索模型通常处理简单的搜索场景，其中检索任务是固定的（例如，寻找一段文字来回答特定问题），并且仅涉及单一模态...
- [Does your LLM truly unlearn? An embarrassingly simple approach to recover unlearned knowledge](https://arxiv.org/abs/2410.16454): 大型语言模型 (LLMs) 在文本生成方面表现出卓越的能力，这得益于在海量文本语料库上的广泛训练。然而，LLMs 也可能从中习得不良行为...
- [What Matters in Transformers? Not All Attention is Needed](https://arxiv.org/abs/2406.15786): 虽然扩展基于 Transformer 的大型语言模型 (LLMs) 在各种任务中展现了充满希望的性能，但它也引入了冗余架构，为效率带来了挑战...
- [Speculations on Test-Time Scaling | Richard M. Karp Distinguished Lecture](https://www.youtube.com/live/6fJjojpwv1I?si=6byPStsGqUHSK0qP): Sasha Rush (康奈尔大学) https://simons.berkeley.edu/events/speculations-test-time-scaling-richard-m-karp-distinguished-lecture Richard M. Karp Distingu...
- [GitHub - srush/awesome-o1: A bibliography and survey of the papers surrounding o1](https://github.com/srush/awesome-o1): 关于 o1 的论文目录和综述 - srush/awesome-o1

---

### **Nous Research AI ▷ #**[**reasoning-tasks**](https://discord.com/channels/1053877538025386074/1264666760972472481/1304688382361735260) (22 条消息🔥):

> - `Google Gemini AI system`
> - `Dynamic Model Selection`
> - `RAG aspects`
> - `User feedback on AI models`
> - `Discussion on project collaboration`

- **Google Gemini AI 系统模拟推理**：受 OpenAI 启发，一位成员利用 **Google Gemini 模型** 构建了一个能够模拟推理并通过交互进行自适应的 AI 系统。该项目包含 meta-prompts、memory recall 和 dynamic query analysis 等元素，详见[这篇文章](https://tinyurl.com/yp7b9d3u)。
  
  - 另一位成员称赞了其易用性，并表示这与他们自己的工作产生了共鸣，说道：*“就像在读别人写的东西一样。哈哈”*。
- **深入探讨动态模型功能**：开发者澄清说，虽然该系统具有 RAG 方面的特性，但它还具备 **Dynamic Model Selection**、**Session Context Tracking** 和 **Performance Logging** 等自主功能。他们幽默地对这一长串功能评价道：*“噢，哇，这可真绕口 😂”*。
  
  - 一位成员表示打算深入研究列出的每种方法，表现出对探索该项目复杂细节的浓厚兴趣。
- **协作服务器的发展机会**：开发者试图了解是否有其他服务器对该 AI 项目感兴趣。他们表达了对社区内协作和知识共享的热切期待。
- **AI 模型用户体验**：讨论内容包括 Bard 发布时的体验以及展示不同草稿输出的功能，这些功能增强了用户交互和参与度。成员们分享了他们的想法，并提到了之前与聊天模型交互时的“恐怖谷”时刻。

 

**提到的链接**：[GitHub - cameronaaron/Geminio1](https://github.com/cameronaaron/Geminio1/)：通过在 GitHub 上创建账号，为 cameronaaron/Geminio1 的开发做出贡献。

 

---

### **Nous Research AI ▷ #**[**rag-dataset**](https://discord.com/channels/1053877538025386074/1218682416827207801/1304689502689820682) (2 条消息):

> - `Data Privacy Tools`
> - `8B Model Quantization`

- **艺术项目凸显数据隐私风险**：受 Joy Buolamwini 博士的启发，一位成员创建了一个艺术项目，强调保护 **个人数据** 免受数据经纪人侵害的重要性。该项目包含一个仅凭你的全名和地点就能讲述你人生故事的工具。
  
  - 该项目倡导通过退出数据经纪人数据库来掌控个人的 **数字足迹 (digital footprint)**，并强调 **隐私问题** 至关重要。
- **8B 模型推荐**：一位成员强烈建议将 **8B 模型** 与可用的最大量化 (Quantization) 版本结合使用，以获得最佳性能。
  
  - 这一观点是在关于模型效率的持续讨论中提出的，重点是最大化资源利用率。

 

**提到的链接**：[Your Life Story](https://lifestorys-b93f5c9c5deb.herokuapp.com/)：未找到描述

 

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1304536681755709502) (204 条消息🔥🔥):

> - `Remote Working Flexibility` (远程办公的灵活性)
> - `GCP vs AWS UIs` (GCP 与 AWS 的 UI 对比)
> - `Internship Experiences in Tech` (技术领域的实习经验)
> - `Frontend Development Challenges` (前端开发的挑战)
> - `Music Model Training Insights` (音乐模型训练见解)

- **针对优秀候选人的远程办公政策**：讨论了一些公司如何允许优秀的候选人远程办公或在不同的办公室工作，并强调说服他们搬迁到另一个办公室可能反而更容易。
  
  - *一些成员反思了公司为了安置核心聘用人员而专门建立新办公室的情况*。
- **GCP 复杂且笨重的 UI**：成员们分享了对 GCP 控制台过于复杂且响应缓慢的沮丧感（相比 AWS 和 Azure），并讨论了他们更倾向于使用 CLI 而非 UI。
  
  - 成员们对云平台 UI 的臃肿表示担忧，认为这反映了一种趋势，即后端工程师在团队中占据主导地位，而前端代表性不足。
- **对技术领域实习项目的反思**：参与者分享了实习经验，通常详细说明了前端项目虽然往往地位较低，但能产生有价值的产出，包括关键文档和重构。
  
  - 有人指出，即使是实习生主导的构建界面的努力，无论质量如何，也能产生见解和可运行的原型。
- **前端开发的挑战**：对话揭示了前端工作在大科技公司中往往被低估且被视为地位较低，同时也强调了其重要性。
  
  - 参与者反思了在面向用户的项目中管理状态和组件复杂性的困难，以及 UI 维护所需的大量时间投入。
- **利用各种数据源训练音乐模型**：一位参与者评论了包含文本和 MIDI 的多模态数据集的潜力，利用他们现有的 YouTube 录音和 MIDI 文件收藏。
  
  - 讨论还包括在有限的数据流派上训练音乐模型的影响，以及探索在网上找到的原创作品。

**提到的链接**：

- [(32 \* 512 \* 50304 \* 16) bits to gb). - Wolfram|Alpha](https://www.wolframalpha.com/input?i=(32+*+512+*+50304+*+16)+bits+to+gb).)：Wolfram|Alpha 为最广泛的人群提供专家级的知识和能力——涵盖所有职业和教育水平。
- [GitHub - KellerJordan/Muon: Muon optimizer for neural networks: >30% extra sample efficiency, <3% wallclock overhead](https://github.com/KellerJordan/Muon)：用于神经网络的 Muon 优化器：样本效率提升 >30%，时钟开销 <3% - KellerJordan/Muon
- [Muon/muon.py at master · KellerJordan/Muon](https://github.com/KellerJordan/Muon/blob/master/muon.py#L119)：用于神经网络的 Muon 优化器：样本效率提升 >30%，时钟开销 <3% - KellerJordan/Muon

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1304695968595185715) (254 条消息🔥🔥):

> - `Low Cost/Low Data Image Model Training Techniques` (低成本/低数据图像模型训练技术)
> - `Normalized Transformer (nGPT)` (归一化 Transformer (nGPT))
> - `Value Residual Learning` (Value Residual Learning)
> - `Batch Size Scaling` (Batch Size 缩放)
> - `Learnable Skip Connections` (可学习的跳跃连接)

- **低成本图像模型训练技术的探索**：成员们讨论了有前景的低成本/低数据图像模型训练技术，将 **MicroDiT**、**Stable Cascade** 和 **Pixart** 确定为有效方法。
  
  - 其他建议包括逐渐增加 Batch Size，尽管这被认为不够“有趣”，但已被证明是有效的。
- **关于归一化 Transformer (nGPT) 的见解**：对 [论文](https://arxiv.org/html/2410.01131v1) 中 nGPT 结果的复现尝试显示出褒贬不一的结果，一些人实现了速度提升，而另一些人则没有。
  
  - 讨论强调了该架构专注于 Embedding 和隐藏状态的单位范数归一化（unit norm normalization），从而在不同的任务表现上实现了更快的学习。
- **Value Residual Learning 技术的进展**：Value Residual Learning 被强调为 Speedrun 成功的重大贡献者，允许 Transformer 块访问之前计算的值。
  
  - 据报道，使残差变为可学习的方法提高了性能，在 Speedrun 期间显著降低了 Loss，促使成员们考虑其在大规模情况下的效果。
- **Batch Size 缩放策略**：成员们分享了 Batch Size 缩放的经验，主张根据 Token 数量采取线性爬坡（ramp-up）方法，以在不产生重新编译延迟的情况下优化性能。

- 建议使用 Dynamic shapes 以提高 batch size 处理的效率，但注意到这可能会减慢训练过程。
- **对 Learnable Skip Connections 的审查**：对 Learnable Skip Connections 的有效性（特别是基于 attention values 的连接）进行了辩论，并对其在更大模型中的 scalability 持怀疑态度。
  
  - 虽然部分成员观察到在速度和稳定性方面的显著提升，但其他人提到过去的文献表明，在更大规模下收益有限，且可能出现收益递减。

**提到的链接**：

- [Few-Shot Task Learning through Inverse Generative Modeling](https://arxiv.org/abs/2411.04987)：通过逆生成建模进行少样本任务学习：仅通过几个示例来学习 Agent 的意图（由其目标或运动风格定义）通常极具挑战性。我们将此问题称为任务概念学习，并提出了我们的方法...
- [Tweet from BlinkDL (@BlinkDL_AI)](https://x.com/BlinkDL_AI/status/1855245097094517181)：来自 BlinkDL (@BlinkDL_AI) 的推文：RWKV-7 也可以在 3200 步内达到 2.27xx（最初为 5100 步）😀 可复现的代码和日志：https://github.com/BlinkDL/modded-nanogpt-rwkv 🚀 #RWKV #RNN 引用 Keller Jordan (@kellerjordan0) 的话...
- [Soft Condorcet Optimization for Ranking of General Agents](https://arxiv.org/abs/2411.00119v2)：用于通用 Agent 排名的软孔多塞优化（Soft Condorcet Optimization）：推动 AI 模型和 Agent 进步的一种常见方法是比较它们在标准化基准测试中的表现。比较通用 Agent 的表现需要汇总它们各自的表现...
- [Initialization of Large Language Models via Reparameterization to Mitigate Loss Spikes](https://arxiv.org/abs/2410.05052)：通过重参数化初始化大语言模型以缓解 Loss 突刺：Loss 突刺（Loss 值突然发散的现象）是 LLM 预训练中的一个基本问题。本文假设范数的不均匀性...
- [Tweet from Alexandre TL (@AlexandreTL2)](https://x.com/alexandretl2/status/1848786982673256490?s=46)：来自 Alexandre TL (@AlexandreTL2) 的推文：这是在 FineWeb 上 162M 模型、2.5B Token 的验证集 Loss（与论文中测试的相比非常小，但由于 GPU 贫乏 + 我们需要从某个地方开始）500M bz, 1024 ctx len, AdamW, L...
- [Geometric Dynamics of Signal Propagation Predict Trainability of Transformers](https://arxiv.org/abs/2403.02579)：信号传播的几何动力学预测 Transformer 的可训练性：我们研究了深度、随机初始化的 Transformer 中的前向信号传播和梯度反向传播，得出了关于初始化超参数的简单必要且充分条件...
- [Tweet from Alexandre TL (@AlexandreTL2)](https://x.com/alexandretl2/status/1848786982673256490?)：来自 Alexandre TL (@AlexandreTL2) 的推文：这是在 FineWeb 上 162M 模型、2.5B Token 的验证集 Loss（与论文中测试的相比非常小，但由于 GPU 贫乏 + 我们需要从某个地方开始）500M bz, 1024 ctx len, AdamW, L...
- [Tweet from Grad (@Grad62304977)](https://x.com/grad62304977/status/1854295837741809933?s=46)：来自 Grad (@Grad62304977) 的推文：新的 NanoGPT 记录中 43% 的加速归功于我开发的一种 Value Residual Learning 变体。Value Residual Learning（最近由 https://arxiv.org/abs/2410.17897 提出）允许...
- [The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489)：模型架构与硬件协同设计的案例：虽然 GPU 负责训练绝大多数最先进的深度学习模型，但在设计新的深度学习（DL）模型时，其架构的影响往往被忽视...
- [Flex attention underperforms SDPA (cuDNN), constructing T5 attention bias via embedding weights · Issue #138493 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/138493#issuecomment-2433345005>)：Flex Attention 性能低于 SDPA (cuDNN)，通过 Embedding 权重构建 T5 Attention Bias · Issue #138493 · pytorch/pytorch：🐛 描述 Bug。我一直尝试在 flex_attention 中实现 T5 Encoder 相对 Attention Bias。我为此想出了几种算法和一个基准测试脚本：https://gist.github.com/Birc...
- [mit-deep-learning-book-pdf/complete-book-pdf/Ian Goodfellow, Yoshua Bengio, Aaron Courville - Deep Learning (2017, MIT).pdf at master · janishar/mit-deep-learning-book-pdf](https://github.com/janishar/mit-deep-learning-book-pdf/blob/master/complete-book-pdf/Ian%20Goodfellow%2C%20Yoshua%20Bengio%2C%20Aaron%20Courville%20-%20Deep%20Learning%20(2017%2C%20MIT).pdf)：Ian Goodfellow, Yoshua Bengio 和 Aaron Courville 编写的 PDF 格式 MIT 深度学习书籍（完整版和部分章节）- janishar/mit-deep-learning-book-pdf
- [nGPT: Normalized Transformer with Representation Learning on the Hypersphere](https://arxiv.org/html/2410.01131v1)：nGPT：在超球面上进行表示学习的归一化 Transformer：未找到描述
- [[BE]: Update CUDNN for Unix OSS to 9.5.1.17 by Skylion007 · Pull Request #137978 · pytorch/pytorch](https://github.com/pytorch/pytorch/pull/137978)：[BE]：由 Skylion007 将 Unix OSS 的 CUDNN 更新至 9.5.1.17 · Pull Request #137978 · pytorch/pytorch：显著更快、更好的 CUDNN Attention，特别是在 Hopper 上（FA3 实现？）。包含大量 Bug 修复、更好的性能、更数值稳定/修复了启发式算法、为 SDPA 提供更多功能...
- [modded-nanogpt/logs/6eae65d0-7bee-45e3-9564-f2a9602d5ba6.txt at fc--bz-warmup · leloykun/modded-nanogpt](https://github.com/leloykun/modded-nanogpt/blob/fc--bz-warmup/logs/6eae65d0-7bee-45e3-9564-f2a9602d5ba6.txt)：在 2.67B Token 中达到 NanoGPT (124M) 质量。通过在 GitHub 上创建账号为 leloykun/modded-nanogpt 的开发做出贡献。
- [modded-nanogpt/logs/421bead0-54ae-41c6-8e00-8f75d52da834.txt at fc--bz-warmup · leloykun/modded-nanogpt](https://github.com/leloykun/modded-nanogpt/blob/fc--bz-warmup/logs/421bead0-54ae-41c6-8e00-8f75d52da834.txt)：在 2.67B Token 中达到 NanoGPT (124M) 质量。通过在 GitHub 上创建账号为 leloykun/modded-nanogpt 的开发做出贡献。

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1304563408363982969) (4 条消息):

> - `Deep Neural Network Modeling`
> - `Intervention Techniques in AI`
> - `SVD in Model Updates`
> - `Behavioral Changes in Models`
> - `Physics Simulations with ML`

- **使用符号方程进行 Deep Neural Network 近似**：*Wabi.sabi.1* 提出了一种从 Deep Neural Network 中提取符号方程的方法，允许对其行为进行有针对性的修改。
  
  - 他们对这种干预方法可能产生的副作用表示担忧，特别是在需要细微行为控制的场景中。
- **模型更新技术的挑战**：*Woog* 尝试执行提议的更新神经网络权重的第 3 步，但遇到了困难，这表明成功可能取决于第 1 步和第 2 步的处理方式。
  
  - 他们指出了任务的复杂性，并承认环境设置在结果中起着重要作用。
- **用于模型近似的线性映射拟合**：*Wabi.sabi.1* 概述了一个详细的过程，包括将线性映射拟合到输入/输出对，试图通过使用 SVD 使神经网络模型表现得更好。
  
  - 他们寻求关于干预性质如何在此背景下应用的澄清，以及是否存在任何直接的示例。
- **事后修改与先验使用**：*Wabi.sabi.1* 反思了在未知程序中事后使用先验知识的概念，强调了对先前建立的参数的重新考虑。
  
  - 这引发了关于回顾性干预的有效性及其在模型行为中影响的问题。

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1304577536113643650) (24 条消息🔥):

> - `Using apply_chat_template`
> - `Local logging without HF`
> - `Benchmarking intermediate checkpoints`
> - `Impact of prompt format on model output`
> - `Calculating accuracy errors in lm-eval`

- **Instruct tuned 模型需要 apply_chat_template**：一位成员询问关于为其模型使用 `--apply_chat_template` 的事宜，确认该模型经过了 Instruct tuned。
  
  - 另一位成员询问如何在 Python 中实现这一点，随后指向了一个特定的 GitHub 文档链接。
- **不使用 HF 在本地记录样本**：一位成员询问是否可以在不上传到 HF 的情况下，使用 `--log_samples` 在本地记录样本，另一位成员建议使用 `push_samples_to_hub=False`。
  
  - 针对改为记录到 Wandb 的情况进行了澄清，引发了关于修改库文件的讨论。
- **不同 Checkpoint 类型之间显著的运行时间差异**：一位成员观察到 LoRA 模型评估大约需要 **8 分钟**，而全量微调（full finetuned）模型则需要 **17 分钟**，尽管它们运行在相同的硬件上。
  
  - 小组讨论了这种差异的可能原因，包括检查 Batch Size 以及 GPU 可能存在的硬件问题。
- **Prompt 格式影响模型输出**：在讨论模型配置时，有人注意到 Chat 模型期望特定格式的 Prompt，这可能会显著改变结果。
  
  - 一位成员意识到，如果他们没有使用正确的日志记录样本选项，之前的结果可能是错误的。
- **在 lm-eval 准确率计算过程中遇到错误**：一位用户报告了在 lm-eval 中计算准确率时的错误，特别是与不支持的操作数类型相关的 TypeError。
  
  - 他们寻求关于如何生成输出并将其保存到文件以进行进一步故障排除的建议。

**提到的链接**：

- [lm-evaluation-harness/lm_eval/__main__.py at bd80a6c0099ee207e70f4943117739a817eccc0b · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/bd80a6c0099ee207e70f4943117739a817eccc0b/lm_eval/__main__.py#L426-L427))：一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
- [lm-evaluation-harness/lm_eval/evaluator.py at bd80a6c0099ee207e70f4943117739a817eccc0b · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/bd80a6c0099ee207e70f4943117739a817eccc0b/lm_eval/evaluator.py#L67)：一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1304537536642813982) (356 条消息🔥🔥):

> - `Qwen 2.5 Coder`
> - `Aider Performance`
> - `Embedding Integration`
> - `Documentation Tools`
> - `Model Benchmarking`

- **对 Qwen 2.5 Coder 结果的兴奋**：Qwen 2.5 Coder 模型因其性能而备受关注，在编程任务上几乎与 Claude 持平，在 diff 指标上达到了 **72.2%** 的基准测试结果。
  
  - 用户们渴望在本地测试该模型，一些人正在讨论在不同性能能力的 GPU 上运行它的可行性。
- **Aider 的集成与功能**：几位用户讨论了 Qwen 2.5 与 Aider 的集成，强调了使用 .env 文件进行配置以及在 glhf.chat 等平台上进行测试的便利性。
  
  - 正在探索改进 Aider 功能和 UI 的贡献，包括本地二进制文件和 Embedding 集成的潜力。
- **文档及其重要性**：大家一致认为维护项目文档很有必要，特别是对于像 Leptos 和 SurrealDB 这样的大型库，以帮助 LLM 处理更新。
  
  - 用户对能够高效抓取和处理文档的工具表现出兴趣，从而使项目集成更加顺畅。
- **探索 Embedding API 和模型能力**：讨论中包含了对 Embedding API 的怀疑，一些用户提出，一旦 LLM 变得更便宜，使用 Embedding 可能会变得过时。
  
  - 辩论了 LLM 直接处理整个文档而不使用 Embedding 的潜力，这表明了模型利用方式可能发生转变。
- **Qwen 模型的未来**：出现了关于 Qwen 2.5-72B 模型可用性的询问，用户对其性能指标和量化细节表示好奇。
  
  - 关于在保持效率和输出质量的同时使用低量化（lower-quanta）模型的更广泛影响的讨论正在进行中。

**提到的链接**：

- [Deepbricks](https://deepbricks.ai/pricing)：未找到描述
- [来自 ollama (@ollama) 的推文](https://x.com/ollama/status/1855352515229053111)：ollama run opencoder OpenCoder 已提供 1.5B 和 8B 模型。
- [安装](https://aider.chat/docs/install.html)：如何安装并开始使用 aider 进行结对编程。
- [Yummers The Boys GIF - Yummers The Boys Homelander - Discover & Share GIFs](https://tenor.com/view/yummers-the-boys-homelander-gif-26204488)：点击查看 GIF
- [首页](https://aider.chat/)：aider 是你终端里的 AI 结对编程工具
- [One Peak One Piece GIF - ONE PEAK One piece One piece cry - Discover & Share GIFs](https://tenor.com/view/one-peak-one-piece-one-piece-cry-its-so-peak-gif-13413926760013023520)：点击查看 GIF
- [Qwen2.5 Coder Demo - a Hugging Face Space by Qwen](https://huggingface.co/spaces/Qwen/Qwen2.5-Coder-demo)：未找到描述
- [Qwen2.5 Coder Artifacts - a Hugging Face Space by Qwen](https://huggingface.co/spaces/Qwen/Qwen2.5-Coder-Artifacts)：未找到描述
- [Enough Okay Enough Allan GIF - Enough okay enough Allan The oval - Discover & Share GIFs](https://tenor.com/view/enough-okay-enough-allan-the-oval-deception-s4e15-gif-2408674230217116567)：点击查看 GIF
- [Homelander Homelander The Boys GIF - Homelander Homelander the boys Homelander sad - Discover & Share GIFs](https://tenor.com/view/homelander-homelander-the-boys-homelander-sad-homelander-its-peak-homelander-peak-gif-14542009839452529163)：点击查看 GIF
- [CONVENTIONS.md](https://gist.github.com/JWPapi/620533fe7a8f4b12256128c23abaf245)：GitHub Gist：即时分享代码、笔记和代码片段。
- [bartowski/Qwen2.5-Coder-32B-Instruct-GGUF · Hugging Face](https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF)：未找到描述
- [Qwen2.5-Coder - a Qwen Collection](https://huggingface.co/collections/Qwen/qwen25-coder-66eaa22e6f99801bf65b0c2f)：未找到描述
- [GitHub - robert-at-pretension-io/rust_web_scraper](https://github.com/robert-at-pretension-io/rust_web_scraper/tree/main)：通过在 GitHub 上创建账户来为 robert-at-pretension-io/rust_web_scraper 的开发做出贡献。
- [Qwen2.5 Speed Benchmark - Qwen](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html)：未找到描述
- [选项参考](https://aider.chat/docs/config/options.html#--map-tokens-value)：关于 aider 所有设置的详细信息。
- [YAML 配置文件](https://aider.chat/docs/config/aider_conf.html)：如何使用 yaml 配置文件配置 aider。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1304558458569490523) (116 messages🔥🔥):

> - `Aider 的 RAG 解决方案`
> - `用于相似性搜索的 Qdrant 工具`
> - `Aider 命令限制`
> - `在大型代码库中使用 Aider`
> - `Aider 性能问题`

- **使用 NotebookLM 的 RAG 解决方案**：一位用户分享了他们如何有效地将文档抓取为 Markdown 文件以供 NotebookLM 使用，从而实现与 Aider 的上下文驱动查询，并成功应用于 Fireworks API。
  
  - 这种精简的方法增强了从文档端点生成相关 Python 客户端的能力，提高了工作流效率。
- **增强 Aider 与 Qdrant 的交互**：讨论集中在开发一个 API 来向 Qdrant 发送查询，以便在 Aider 中生成上下文，用户分享了各种实现方法。
  
  - 建议包括创建一个用于查询的自定义 Python CLI，这表明 Aider 和 Qdrant 之间需要改进集成机制。
- **Aider 命令限制说明**：一位用户对 Aider 有效处理现有代码修改而不无意中覆盖函数的能力表示担忧。
  
  - 其他人指出，最近的版本可能会加剧这些问题，暗示更新可能会导致现有项目的混乱。
- **在大型项目中使用 Aider**：多位用户讨论了在大型代码库中使用 Aider 的策略，强调了维护上下文和防止文件过载的挑战。
  
  - 社区探索了添加选择性文件包含功能，强调了在 Aider 中更好地管理大型项目的需求。
- **Aider 模型配置问题**：用户在将 Aider 与特定模型（特别是 Ollama）集成时，对有关上下文窗口大小和成本的警告感到困惑。
  
  - 有人指出，这些警告可能不会阻碍使用，Aider 后端的修改可能是导致错误通知的原因。

**提到的链接**：

- [OpenAI compatible APIs](https://aider.chat/docs/llms/openai-compat.html)：aider 是你终端里的 AI 结对编程工具
- [Model warnings](https://aider.chat/docs/llms/warnings.html)：aider 是你终端里的 AI 结对编程工具
- [aider/aider/website/docs/usage/tips.md at main · Aider-AI/aider](https://github.com/Aider-AI/aider/blob/main/aider/website/docs/usage/tips.md)：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 Aider-AI/aider 的开发做出贡献。
- [Issues · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2258)：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 Aider-AI/aider 的开发做出贡献。
- [[Bug]: get_model_info() blows up for ollama models? · Issue #6703 · BerriAI/litellm](https://github.com/BerriAI/litellm/issues/6703)：发生了什么？使用 ollama 模型调用 litellm.get_model_info() 会抛出异常。但我可以使用这些模型正常运行 litellm.completion()。$ pip freeze | egrep 'litellm|ollama' litell...
- [aider thinks model is unknown and asks if I meant \*The exact same model\* · Issue #2318 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2318)：针对 ollama/vanilj/supernova-medius:q6_k_l 的警告：未知的上下文窗口大小和成本，正在使用合理的默认值。你是想指这些吗？- ollama/vanilj/supernova-medius:q6_k_l 你可以跳过这个...

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1304555866741145681) (4 messages):

> - `OpenCoder LLM`
> - `RefineCode 预训练语料库`
> - `Aider 的上下文限制`

- **OpenCoder 成为代码 LLM 的领导者**：[OpenCoder](https://opencoder-llm.github.io/) 是一个开源代码 LLM 家族，包括 1.5B 和 8B 模型，在 **2.5 万亿 token** 上进行训练，主要是原始代码。
  
  - 它旨在通过提供**模型权重**、**推理代码**和透明的数据处理过程来赋能研究人员，以推动代码 AI 的发展。
- **RefineCode 拥有广泛的编程语料库**：**RefineCode** 提供了一个高质量的预训练语料库，包含 **9600 亿 token**，涵盖 **607 种编程语言**。
  
  - 这个可重复的数据集增强了像 OpenCoder 这样新兴代码 LLM 的训练能力。
- **对 Aider 1300 上下文窗口的担忧**：一位成员表示 **1300 上下文窗口**在 Aider 中无法有效工作。
  
  - 这引发了关于 Aider 在实际应用中的可扩展性和性能的疑问。

 

**提到的链接**：[OpenCoder: Top-Tier Open Code Large Language Models](https://opencoder-llm.github.io/)：未找到描述

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1304535925212053569) (394 messages🔥🔥):

> - `Stable Diffusion Models`
> - `GPU Performance`
> - `LoRA Training`
> - `AI Video Generation`
> - `GGUF Format`

- **探索 Stable Diffusion Models 变体**：用户讨论了从 Stable Diffusion 1.5 迁移到 SD 3.5 和 Flux 等新模型，一些人表示新版本需要更少的 VRAM 且性能更好。
  
  - 建议探索更小的 GGUF 模型，即使在有限的硬件上也能更高效地运行。
- **GPU 使用及寿命担忧**：有人对每天运行 Stable Diffusion 带来的长期 GPU 使用表示担忧，并将其与游戏性能影响进行了比较。
  
  - 一些用户指出，随着即将推出的 RTX 5000 系列，GPU 价格可能会下降，鼓励他人在购买新硬件前先等待。
- **为 Stable Diffusion 训练 LoRAs**：一位用户询问如何使用小数据集为 Stable Diffusion 1.5 训练 LoRA，并分享了他们在 Flux 训练方面的经验。
  
  - 建议包括使用 Kohya_ss 训练器，并遵循特定的在线指南以有效地引导训练过程。
- **AI Video Generation 工具**：介绍了 Pollo AI 等新工具，使用户能够通过文本提示创建视频并使静态图像动起来。
  
  - 该工具允许通过根据用户定义的参数生成引人入胜的视频内容来进行创意表达。
- **了解 GGUF 文件**：用户了解了 GGUF 格式，该格式允许在图像生成工作流中更紧凑、更高效地使用模型。
  
  - 有人提到，与较大的模型相比，使用 GGUF 文件可以在保持质量的同时显著降低资源需求。

**提到的链接**：

- [Create README.md · youknownothing/realDream_STOIQONewreality at f5d8fad](https://huggingface.co/youknownothing/realDream_STOIQONewreality/commit/f5d8fadc6b1e78130050509bb8d165d362b5d304): 未找到描述
- [Stable Diffusion 快速入门指南 - 一分钟内生成 AI 图像](https://andrewongai.gumroad.com/l/stable_diffusion_quick_start?_gl=1*1dzux7m*_ga*MTk4Mjg3OTYxNC4xNzMxMTEzNDA3*_ga_YHRX2WJZH7*MTczMTIxMDg1Mi41LjEuMTczMTIxMTA3Ny41OS4wLjA.): 想使用 Stable Diffusion 但不知道从何开始？你可以在本指南中找到一系列选项。你可以选择最适合你的方案。立即下载指南！如果你...
- [LoRA 训练 (Stable Diffusion 1.5) | ScottBaker.ca](https://www.scottbaker.ca/AI/LoRA-Training): 未找到描述
- [AI 视频生成器：创建真实/虚幻的高清视频 | Pollo AI](https://pollo.ai/): 使用行业领先的 AI 视频生成器 Pollo AI，通过文本提示词、图像或视频来创建视频。将你的创意转化为高分辨率、高质量的视频。
- [Create README.md · youknownothing/realDream_STOIQONewreality at f5d8fad](https://huggingface.co/youknownothing/realDream_STOIQONewreality/commit/f5d8fadc6b1e78130050509bb8d1): 未找到描述
- [FLUX Dev/Schnell (Base UNET) + Google FLAN FP16/NF4-FP32/FP8 - FLUX_Dev-FLAN-FP16 | Flux Checkpoint | Civitai](https://civitai.com/models/895985/flux-devschnell-base-unet-google-flan-fp16nf4-fp32fp8): 带有改进版 TE 的完整 Checkpoint，无需加载额外的 CLIP/TE。FLUX.1 (Base UNET) + Google FLAN NF4 是我推荐的兼顾质量与速度平衡的模型...
- [Reddit - 深入探索任何事物](https://www.reddit.com/r/StableDiffusion/comments/1feibuv/guide_getting_started_with_flux_forge/): 未找到描述
- [Azure 数据泄露：发生了什么以及是如何发生的？ | Twingate](https://www.twingate.com/blog/tips/Azure-data-breach): 在本文中，我们将讨论 Azure 数据泄露事件、它是如何发生的、泄露了哪些信息，以及受影响后该怎么办。
- [在本地安装 Stable Diffusion（只需 3 分钟！！）](https://www.youtube.com/watch?v=6MeJKnbv1ts): 对于拥有自组 PC 的用户，这里介绍了如何在 5 分钟内安装 Stable Diffusion - GitHub 网站链接：https://github.com/Hugging Face W...
- [在 Google Colab 中进行 Flux AI LoRA 模型训练 – 简单的 FluxGym 教程](https://www.youtube.com/watch?v=yvXOKHeZtgs&ab_channel=TheLocalLab): #fluxai #comfyui #stablediffusion #fluxgguf #aiart #sd3 #sdxl #fluxgym 学习如何使用 Flux 和自定义 LoRA 创建令人惊叹的 AI 艺术！我们免费的 Google Colab 教程...
- [GitHub - TheLocalLab/fluxgym-Colab: 用于 FluxGym LoRA 训练仓库的 Colab。](https://github.com/TheLocalLab/fluxgym-Colab): 用于 FluxGym LoRA 训练仓库的 Colab。通过在 GitHub 上创建账号，为 TheLocalLab/fluxgym-Colab 的开发做出贡献。
- [FLUX.1 [dev] - v1.0 | Flux Checkpoint | Civitai](https://civitai.com/models/617609/flux1-dev): 如果你还没有阅读所有建议，请不要下载。因为它非常庞大，且比 SD 需要更多的资源。我们有一种新的方式可以轻松运行 Flux...
- [STOIQO NewReality 🟡 FLUX, SD3.5, SDXL, SD1.5 - 🔵 XL Light 1.0 | Stable Diffusion XL Checkpoint | Civitai](https://civitai.com/models/161068?modelVersionId=498484): 🟡: Flux 模型 🟢: SD 3.5 模型 🔵: SD XL 模型 🟣: SD 1.5 模型 🔴: 已过期模型 🟡STOIQO NewReality 是一款旨在生成...的前沿模型

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1305281599511986246) (1 条消息):

> - `3D Object Generation API`

- **3D Object Generation API 弃用**：由于缺乏关注，每几周的请求量少于 **5 次**，**3D Object Generation API** 将于本周五移除。
  
  - 更多详情请参阅 [文档](https://openrouter.ai/docs/objects)。
- **替代功能的未来**：随着 **3D Object Generation API** 的移除，注意力可能会转移到需要更多社区参与和兴趣的替代功能上。
  
  - 看来团队正专注于改进那些被更积极利用的产品。

 

**提到的链接**：[OpenRouter](https://openrouter.ai/docs/objects)：LLM 路由和市场

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1304538839993421899) (317 条消息🔥🔥):

> - `Hermes 性能`
> - `Llama 模型使用情况`
> - `Qwen 2.5 Coder 模型`
> - `AI 模型更新`
> - `OpenRouter 易用性`

- **Hermes 稳定性欠佳**：用户报告 Hermes 模型响应不一致，无论是在免费版还是付费版，在不同条件下都存在问题。
  
  - 一些人推测这些问题可能与 Rate limits 或 OpenRouter 端的问题有关。
- **Llama 3.1 70B 越来越受欢迎**：Llama 3.1 70B Instruct 模型的采用率正在上升，特别是在 Skyrim AI Follower Framework 社区中。
  
  - 用户在价格和性能方面将其与 Wizard 模型进行对比，并对其能力表示好奇。
- **推出 Qwen 2.5 Coder 模型**：Qwen 2.5 Coder 模型已发布，据报道其 32B 参数版本的编程能力可媲美之前的 Sonnet。
  
  - 用户对其在社区编程任务中的潜在影响表示兴奋。
- **Gemini 1.5 Flash 更新**：一些用户注意到 Gemini 1.5 Flash 模型的改进，暗示可能有更新增强了其性能和编程能力。
  
  - 用户对常规更新之外可能正在测试的实验版本感到好奇。
- **OpenRouter 易用性问题**：关于 OpenRouter 的反馈强调进入聊天室需要多个步骤，并请求简化流程。
  
  - 用户希望提高易用性，以增强对平台的整体参与度。

**提到的链接**：

- [no title found](https://openrouter-3d.vercel.app/): 未找到描述
- [LICENSE.txt · tencent/Tencent-Hunyuan-Large at main](https://huggingface.co/tencent/Tencent-Hunyuan-Large/blob/main/LICENSE.txt): 未找到描述
- [sbintuitions/sarashina2-8x70b · Hugging Face](https://huggingface.co/sbintuitions/sarashina2-8x70b): 未找到描述
- [OpenRouter](https://openrouter.ai/anthropic/claude-3.5-): LLM 路由和市场
- [Deus Ex Deus GIF - Deus Ex Deus Ex - Discover & Share GIFs](https://tenor.com/view/deus-ex-deus-ex-jc-shame-gif-26245854): 点击查看 GIF
- [Apps Using Anthropic: Claude 3.5 Sonnet](https://openrouter.ai/anthropic/claude-3.5-sonnet/apps): 查看正在使用 Anthropic: Claude 3.5 Sonnet 的应用 - 新的 Claude 3.5 Sonnet 提供了优于 Opus 的能力，速度快于 Sonnet，且价格与 Sonnet 持平。Sonnet 特别擅长...
- [OpenRouter](https://openrouter.ai/terms#_4_-payment): LLM 路由和市场
- [tencent/Tencent-Hunyuan-Large · Hugging Face](https://huggingface.co/tencent/Tencent-Hunyuan-Large): 未找到描述
- [Meta: Llama 3.1 70B Instruct – Recommended Parameters](https://openrouter.ai/meta-llama/llama-3.1-70b-instruct/parameters): 查看 Meta: Llama 3.1 70B Instruct 的推荐参数和配置 - Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。这个 70B 指令微调版...
- [Parameters API | OpenRouter](https://openrouter.ai/docs/parameters-api): 用于管理请求参数的 API
- [Models: 'meta-llama' | OpenRouter](https://openrouter.ai/meta-llama/): 在 OpenRouter 上浏览模型
- [GitHub - QwenLM/Qwen2.5-Coder: Qwen2.5-Coder is the code version of Qwen2.5, the large language model series developed by Qwen team, Alibaba Cloud.](https://github.com/QwenLM/Qwen2.5-Coder): Qwen2.5-Coder 是 Qwen2.5 的代码版本，是由阿里巴巴云 Qwen 团队开发的大语言模型系列。 - QwenLM/Qwen2.5-Coder

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1304777319818199060) (7 条消息):

> - `Custom provider keys 访问权限`
> - `Integration beta feature 访问权限`
> - `Beta 测试热情`

- **对 Custom Provider Keys 访问权限的高需求**：包括 *derpenstein69* 和 *sohanemon* 在内的多位成员正在寻求 **custom provider keys** beta 功能的访问权限。
  
  - *derpenstein69* 在请求中表达了感谢，展现了对获得权限的渴望。
- **对 Integration Beta Feature 的紧迫请求**：*nanakotsai* 和 *wendic1* 等成员正在积极请求 **integration beta feature** 的访问权限。
  
  - *wendic1* 特别询问了如何申请权限，表现出对该功能的浓厚兴趣。
- **幽默且具创意的 Beta 测试申请**：*doditz* 在请求 **integration beta feature** 访问权限时，幽默地承诺会成为一名有趣的测试者，并在消息中融入了创意元素。
  
  - 他们轻松的方式包括开玩笑，以及一个涉及三只仓鼠和一只橡皮鸭的古怪集成想法。
- **俏皮的 Beta 参与请求**：成员们在权限请求中采取了俏皮的语气，例如 *cruciflyco* 幽默地表示他们正在“请求访问”。
  
  - 这展示了社区精神以及积极参与 beta 测试过程的意愿。

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1304572119744778281) (68 条消息🔥🔥):

> - `Lilian Weng 离开 OpenAI`
> - `FrontierMath 基准测试`
> - `Qwen2.5 Coder 发布`
> - `Dario Amodei 谈 AI Scaling`
> - `Claude 的 Character 团队`

- **Lilian Weng 离开 OpenAI**：在 OpenAI 工作近 7 年后，[Lilian Weng 宣布](https://x.com/lilianweng/status/1855031273690984623)她将离职以寻求新的机会，并表示她在任职期间获益匪浅。
  
  - 她的离职引发了关于潜在新项目以及社区对其转型的反应的讨论。
- **引入 FrontierMath 作为新的基准测试**：[FrontierMath](https://arxiv.org/abs/2409.12186) 是一个复杂数学问题的基准测试，目前的 AI 模型仅能解决其中不到 2% 的问题，这说明 AI 能力仍存在巨大差距。
  
  - 讨论强调了该基准测试与其他测试相比的难度，及其对 AI 训练的潜在影响。
- **Qwen2.5 Coder 模型发布**：[Qwen2.5 Coder](https://x.com/huybery/status/1856042011390063015) 系列包含多个模型，包括旗舰模型 Qwen2.5-Coder-32B-Instruct，在多项基准测试中取得了与 GPT-4o 相当的竞争性结果。
  
  - 有关该模型性能的细节已经公开，预计很快将发表相关论文。
- **Dario Amodei 讨论 AI Scaling**：在最近的一次播客中，Dario Amodei 强调 Scaling 趋势在不同模态中是一致的，并暗示几年内可能实现人类水平的 AI。
  
  - 他还提到数据质量和架构限制等挑战可能是这种 Scaling 的障碍。
- **各大实验室内的 AI 角色构建**：Anthropic 和 OpenAI 都有专门的团队专注于开发其 AI 模型的性格（Character）和行为，旨在实现合乎伦理且负责任的交互。
  
  - 这反映了各大 AI 实验室确保 AI 表现友好且安全的大趋势。

**提到的链接**：

- [来自 undefined 的推文](https://vxtwitter.com/polynoamial/status/1855037689533178289)：未找到描述
- [来自 deepfates (@deepfates) 的推文](https://x.com/deepfates/status/1795187390660715005)：老实说，他在这里表现得很出色
- [来自 Binyuan Hui (@huybery) 的推文](https://x.com/huybery/status/1856042011390063015)：💪 我竭尽全力为您提供最好的。引用 Qwen (@Alibaba_Qwen) 🚀现在是 11 月 11 日 10:24！这是我们有史以来最好的编程模型 Qwen2.5-Coder-32B-Instruct 的完美发布时间！...
- [来自 Alpin (@AlpinDale) 的推文](https://x.com/AlpinDale/status/1855664208391917962)：试用了 Qwen2.5 Coder 32B 的预览版，感觉就像 Claude 3.5 Sonnet。你又做到了，@Alibaba_Qwen ...
- [来自 Jo Kristian Bergum (@jobergum) 的推文](https://x.com/jobergum/status/1855034296400040234)：希望她会有更多时间写博客 ❤️ 引用 Lilian Weng (@lilianweng) 在 OpenAI 工作了近 7 年后，我决定离开。我学到了很多，现在准备好重置并...
- [来自 Andrew Carr (e/🤸) (@andrew_n_carr) 的推文](https://x.com/andrew_n_carr/status/1856054538769506800)：Qwen2.5-Coder-32B-Instruct 是仅次于 O1-preview 的第二好诗歌模型 🤯
- [Qwen2.5-Coder 技术报告](https://arxiv.org/abs/2409.12186)：在本报告中，我们介绍了 Qwen2.5-Coder 系列，这是对其前身 CodeQwen1.5 的重大升级。该系列包括两个模型：Qwen2.5-Coder-1.5B 和 Qwen2.5-Coder-7B。作为一款编程专用...
- [来自 Lilian Weng (@lilianweng) 的推文](https://x.com/lilianweng/status/1855031273690984623)：在 OpenAI 工作了近 7 年后，我决定离开。我学到了很多，现在准备好重置并开始新的尝试。这是我刚刚与团队分享的笔记。🩵
- [来自 Xeophon (@TheXeophon) 的推文](https://x.com/TheXeophon/status/1854034629998543326)：那家声称 Agent 将运营整个公司的公司竟然没有有效的客服支持，真让我惊讶。引用 Xeophon (@TheXeophon) 今天，我像个傻瓜一样尝试解决一个账单问题...
- [来自 Epoch AI (@EpochAIResearch) 的推文](https://x.com/EpochAIResearch/status/1854993684502282537)：3/10 我们评估了六个领先模型，包括 Claude 3.5 Sonnet、GPT-4o 和 Gemini 1.5 Pro。即使有延长的思考时间（10,000 个 Token）、Python 访问权限和运行实验的能力，成功率...
- [Dario Amodei：Anthropic CEO 谈 Claude、AGI 以及 AI 与人类的未来 | Lex Fridman Podcast #452](https://youtu.be/ugvHCXCOmm4)：Dario Amodei 是开发了 Claude 的 Anthropic 公司的 CEO。Amanda Askell 是一位致力于 Claude 性格和个性的 AI 研究员。Chris...
- [2028 年前 AI 是否能在 FrontierMath 基准测试中达到 >85% 的表现？](https://manifold.markets/MatthewBarnett/will-an-ai-achieve-85-performance-o)：62% 的概率。

---

### **Interconnects (Nathan Lambert) ▷ #**[**other-papers**](https://discord.com/channels/1179127597926469703/1179142630517518397/1305268371096600697) (10 messages🔥):

> - `Training vs Test Time in AI`
> - `ARC Prize Discussion`
> - `Ensemble Approaches to ARC`
> - `Transformer Performance on ARC`

- **重新思考 Training 和 Test Time**：一位成员质疑为什么我们要区别对待 **training** 和 **test times**，并建议在 test-time 进行少量梯度更新（gradients）可以提高结果，在 ARC 验证集上达到了 **61% 的平均分**。
  
  - *只需在 test-time 进行几次梯度更新——这是增加 test time compute 的一种简单方法——就能达到 SoTA！*
- **ARC Prize 被认为言过其实**：一位参与者对 **ARC Prize** 表示怀疑，称其被高估了，但也承认无论如何**人们仍然会针对它进行 hillclimb**。
  
  - 他们提到，“只是需要时间”，表明尽管存在疑虑，但仍相信最终会成功。
- **对纯 Transformer 解决方案的担忧**：有人指出，在 ARC 上获得高分可能不仅仅需要纯 Transformer 方法，暗示了比赛中面临的挑战。
  
  - 另一种观点认为，**ensemble/discrete synthesis** 可能优于纯 Transformer 方法，有可能解决 **75% 以上的 ARC** 题目。

 

**提到的链接**：[来自 Ekin Akyürek (@akyurekekin) 的推文](https://x.com/akyurekekin/status/1855680785715478546)：为什么我们对 train 和 test times 的对待如此不同？为什么一个是“training”，而另一个是“in-context learning”？只需在 test-time 进行几次梯度更新——这是增加 test time compute 的一种简单方法...

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1305340840557940798) (55 条消息🔥🔥):

> - `Gary Marcus 的主张`
> - `AGI 辩论`
> - `Twitter 与 Bluesky 上的挑战`
> - `Threads 平台对比`
> - `AI 领域的科学革命`

- **Gary Marcus 为其 AGI 立场辩护**：Gary Marcus 回应了对其 AGI 主张的批评，表示他从未说过 AGI “不存在”，并强调需要新方法在本世纪实现它，正如他在[关于 AI 下一个十年的文章](https://x.com/GaryMarcus/status/1855782420781691165)中所述。
  
  - 他指责另一方对他的论点进行“稻草人攻击”（strawmanning），并对他们无法承认其正确预测表示沮丧。
- **来自 Twitter 辩论的娱乐性**：成员们认为 Twitter 上关于 AGI 的来回交锋很有趣，一位成员形容这些引用“太棒了”（AMAZING），并对 Gary 拉黑行为相关的“懦夫”（coward）言论感到乐在其中。
  
  - 人们对 Gary 接下来的举动充满期待，讨论暗示跨平台的进一步交流具有潜力。
- **平台对比：Twitter vs. Bluesky**：参与者观察到社交媒体平台在互动质量上的差异，Twitter 被视为获取覆盖范围的地方，而 Bluesky 因其高质量讨论而受到重视。
  
  - 一些人对转向 Bluesky 表示宽慰，认为 Twitter 充斥着“令人痛苦”（miserable）的内容。
- **Threads 及其有效性**：Threads 被批评为类似于 Facebook 的平庸版本，被描述为“无聊的互动诱饵”（engagement bait），且充斥着低质量互动。
  
  - 对话中包含了 Threads 与其他平台的对比，一些人表示感觉它就像青少年在以比 Gary 更逊的方式抨击 AI。
- **即将推出的 AI 相关文献**：一位成员提到订购了一本《科学革命的结构》（The Structure of Scientific Revolutions），计划在 AI 背景下以及围绕 Gary Marcus 的对话中对其进行评论。
  
  - 这表明社区内部希望深入研究影响当前 AI 辩论的理论构建。

**提到的链接**：

- [Gary Marcus (@GaryMarcus) 的推文](https://x.com/GaryMarcus/status/1855789305815576897)：@natolambert 拉黑是因为完全无法为彻底歪曲我的主张承担责任。（以及病态地无法对正确的预测给予肯定。）
- [Laurence Molloy (@MolloyLaurence) 的推文](https://x.com/MolloyLaurence/status/1855784013861937162)：缓慢的进步？🤣🤣🤣 现在的 AI 行业根本不在乎任何缓慢而谨慎的行事。引用 Nathan Lambert (@natolambert)：有多少 AI 研究人员正在...
- [Gary Marcus (@GaryMarcus) 的推文](https://x.com/GaryMarcus/status/1855782420781691165)：X 上的新运动是歪曲我非常具体（且显然正确）的主张，即纯 LLM 将达到边际收益递减点。我从未说过 AGI “不可能存在”。那是...
- [Nathan Lambert (@natolambert) 的推文](https://x.com/natolambert/status/1855775229186093542)：有多少 AI 研究人员为了证明 Gary 是错的而动力十足地想要更早实现 “AGI”？我只是不明白，当你的观点是 “AGI” 不可能存在时，你该如何 “获胜”。...

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1304536783681228841) (49 messages🔥):

> - `VLM Model Choices`
> - `AI Research Trends`
> - `HCI Research in AI`
> - `Personality in AI Models`
> - `Qwen2-VL Dynamic Resolution`

- **关于 Base 与 Instruct-tuned 模型的辩论**：展开了一场关于在 VLM 中使用 Base 模型优于 Instruct-tuned 模型的优势讨论，并对后者典型的微调过程的必要性表示怀疑。
  
  - 一些人推测，传统的 LLM 微调应该在 VLM 微调之前进行，以获得更好的结果，正如 Molmo 项目的实验所显示的那样。
- **AI 研究阵地的转移**：关于当前 AI 研究格局存在争论，一些人认为研究已在很大程度上转向工业界，由于资源差距，学术界留下的工作影响力较小。
  
  - 人们对学术职位的声望与高薪工业界职位的对比表示担忧。
- **AI 交互中的 HCI 研究**：有人表示有兴趣探索与 HCI 相关的研究，这些研究考察了训练后的模型行为如何影响最终用户的交互和结果。
  
  - 具体而言，有人询问写作模型中的建议编辑是否比草稿生成产生更好的结果。
- **Qwen2-VL 中的动态分辨率**：强调了 Qwen2-VL 模型中“Naive Dynamic Resolution”的特性，允许在不进行预先下采样的情况下处理原始分辨率图像。
  
  - 这种能力对于需要高图像保真度的任务至关重要，因为它避免了下采样的有损效应。
- **开展个性研究的挑战**：由于隐私和竞争方面的担忧，对于获取有关 AI 个性特征的最终用户偏好数据存在疑虑。
  
  - 尽管如此，有人建议 HCI 研究可能会为与 AI 交互的最佳 UI 和响应格式提供有价值的见解。

**提到的链接**：

- [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/i/151078631/qwen-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolution)：主要技术和最新模型的介绍。
- [GitHub - srush/awesome-o1: A bibliography and survey of the papers surrounding o1](https://github.com/srush/awesome-o1)：关于 o1 相关论文的参考书目和综述 - srush/awesome-o1。

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1304552959589224489) (5 messages):

> - `Logan's Friday Ship`
> - `Podcast Ideas`
> - `Discussion About Julia`

- **Logan 最喜欢的周五发布 (Friday Ship)**：一位成员分享了 [@OfficialLoganK](https://x.com/OfficialLoganK/status/1854980502727315711) 的一条推文，表达了对他最喜欢的周五发布的兴奋之情，并表示在接下来的几周内将进行一些改进。
  
  - *Logan 的热情很有感染力！*
- **考虑邀请 Logan 参加播客**：一位成员建议也许他们应该邀请 Logan 参加他们的播客，分享想法和经验。
  
  - 这引发了关于 Logan 作为嘉宾吸引力的积极共识。
- **Julia 激发了热情**：有人提到，如果问起 Julia，Logan 可以轻松聊上半集，这表明他与该主题有很深的联系。
  
  - *Julia 对 Logan 来说似乎是一个迷人的话题！*

 

**提到的链接**：[来自 Logan Kilpatrick (@OfficialLoganK) 的推文](https://x.com/OfficialLoganK/status/1854980502727315711)：一段时间以来我最喜欢的周五发布 (Friday ship) 🚢 : ) 在接下来的几周内将继续消除这里的一些粗糙边缘。

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**nlp**](https://discord.com/channels/1179127597926469703/1208183200099344445/1305643241080619140) (1 messages):

> - `Neural Notes`
> - `Language Model Optimization`
> - `DSPy`
> - `MIPRO Optimizers`

- **Neural Notes 探讨 Language Model Optimization**：在最新一期的 [Neural Notes](https://www.youtube.com/watch?v=DVkM5dB3Oqs) 中，投资者 Sandeep Bhadra 和 Simon Tiu 采访了斯坦福 AI 实验室的 PhD 候选人 Krista Opsahl-Ong，讨论了 **Language Model Optimization** 的未来。
  
  - 讨论中预示了关于自动化 Prompt 优化的见解，这可能会引起关注 AI 进展的人士的兴趣。
- **讨论 DSPy 和 MIPRO Optimizers**：一位成员回顾了之前分享的一段视频，其中包含了斯坦福研究人员关于 **DSPy** 中使用的 **MIPRO Optimizers** 的见解。
  
  - 这段对话暗示了该成员打算加深对 DSPy 的理解，并表示希望对该技术形成专业的见解。

 

**提到的链接**：[Neural Notes: The future of language model optimization](https://www.youtube.com/watch?v=DVkM5dB3Oqs)：在本期 Neural Notes 中，Vertex Ventures US 的投资者 Sandeep Bhadra 和 Simon Tiu 与斯坦福 AI 实验室 (SAI...) 的 PhD 候选人 Krista Opsahl-Ong 进行了交谈。

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**rlhf**](https://discord.com/channels/1179127597926469703/1208183230608576562/1304821647794110504) (12 messages🔥):

> - `Model Merging Techniques`
> - `Roleplaying Model Evaluation`
> - `Community-based Evaluations`
> - `MythoMax and MythoLogic Models`
> - `Censorship in Model Use`

- **对 Model Merging 进展的兴奋**：在 Model Merging 领域，人们对 [SALSA](https://arxiv.org/abs/2411.01798) 表现出了兴趣，它通过创新技术解决了 AI Alignment 中的局限性。
  
  - 有人感叹 *'woweee'*，表明了人们对这些模型日益增长的复杂性和潜力的关注。
- **对合并模型的幽默批评**：对 'NobodyExistsOnTheInternet/Llama-2-70b-x8-MoE-clown-truck' 模型的滑稽引用引发了小组的笑声，凸显了命名古怪的模型是如何被看待的。
  
  - 讨论中包含了一个指向 [merge.moe](https://merge.moe/) 的链接，展示了社区认为水平较低的各种模型。
- **关于角色扮演模型基准测试的辩论**：出现了关于角色扮演模型基准测试的问题，一位成员询问评估是更多基于 Vibes（感觉）还是可量化的。
  
  - 大家达成共识，认为可以根据 Reddit 和 4chan 等平台上发现的社区偏好来评估性能。
- **定义角色扮演 AI 的成功**：随后讨论了哪些特征使角色扮演模型“成功”，从创造力到对角色设定的遵循程度。
  
  - 有人担心建立具体的基准测试，同时也要承认对话 NLP 中涉及的潜在 NSFW 场景。
- **推动社区参与**：一位成员建议考虑关注角色扮演和 AI Girlfriend 领域的剧集以促进社区参与，此前曾有过关于粉丝参与的采访。
  
  - 该小组对这些领域中的自动化交互如何影响用户体验和 AI 性能表现出了兴趣。

**提到的链接**：

- [no title found](https://merge.moe/)：未找到描述
- [SALSA: Soup-based Alignment Learning for Stronger Adaptation in RLHF](https://arxiv.org/abs/2411.01798)：在大语言模型 (LLM) 开发中，人类反馈强化学习 (RLHF) 对于使模型与人类价值观和偏好对齐至关重要。RLHF 传统上依赖于 Kullback...
- [Gryphe/MythoMax-L2-13b · Hugging Face](https://huggingface.co/Gryphe/MythoMax-L2-13b)：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**reads**](https://discord.com/channels/1179127597926469703/1214764639397617695/1304645924508667965) (44 条消息🔥):

> - `Scaling Laws`
> - `AGI and GPT-5 Expectations`
> - `Critiques of AI Progress`
> - `Model Performance and Task Utility`
> - `Future of AI Development`

- **Scaling Laws 辩论：依然有效**：目前关于 Scaling 是否仍然有效存在持续讨论，有观点认为最近的 GPT 模型表现不及预期，而对 AGI 的期望依然很高，这意味着两者可能同时成立。
  
  - Scaling Laws 似乎仍在发挥作用，挑战在于提升特定任务性能的边际收益正在递减。
- **OpenAI 的信息传递引发混乱**：OpenAI 围绕 AGI 的宣传导致了对 GPT-5 能力的不切实际预期，掩盖了 AGI 是一个更广泛的系统，它包含但不完全由 GPT-5 定义。
  
  - 对 GPT-5 等模型进展平平的感知表明，有必要澄清关于产品实际能力的沟通。
- **实用性提升速度下降**：虽然 Scaling 继续产生结果，但讨论认为 AI 模型有用性的提升速度正在放缓，这对开发者和投资者来说是一个关键点。
  
  - 新兴的讨论强调了当前模型满足用户预期的局限性，并暗示可能会转向专业化模型。
- **AI 发展的持续前景**：尽管存在停滞感，但对于利用现有 AI 进行进一步产品开发仍持乐观态度，这表明重大机遇依然存在。
  
  - 投资者可能需要根据策略转变调整预期，但总体而言，AI 模型的能力仍处于上升轨道。
- **应对当前的 AI 进展与挑战**：对话反映出，虽然 AI 技术正在取得长足进步，但通往重大突破的路径可能涉及应对新挑战和重置预期。
  
  - 社区在关于进步速度和性质的观点上似乎存在分歧，强调了关于什么构成 AI 真正进步的健康辩论。

**提到的链接**：

- [GPTs Are Maxed Out](https://www.thealgorithmicbridge.com/p/gpts-are-maxed-out)：AI 公司将不得不探索其他路径
- [来自 Adrià Sánchez (@AdriaSnz) 的推文](https://x.com/AdriaSnz/status/1770972134036173225)：@edzitron 你错了。AI 将继续以指数级方式进化
- [来自 Daniel (@DanielofDoge) 的推文](https://x.com/DanielofDoge/status/1769925670736465991)：@edzitron 愚蠢的预测。
- [来自 T-toe (@thuzawtwice) 的推文](https://x.com/thuzawtwice/status/1770063476594979107)：@edzitron 当年人们对互联网也是这么说的
- [Two alignment threat models](https://open.substack.com/pub/aligned/p/two-alignment-threat-models?r=68gy5&utm_medium=ios)：为什么解决诱导不足（under-elicitation）和图谋（scheming）都很重要
- [来自 Nathan Lambert (@natolambert) 的推文](https://x.com/natolambert/status/1856077454210723856)：有很多关于“Scaling 是否已经结束”的讨论，The Information 的报道称最新的 GPT 模型没有达到 OpenAI 的预期，而 Sam Altman 仍在炫耀……
- [来自 el (@jaxgriot) 的推文](https://x.com/jaxgriot/status/1769804607101042929)：@TheXeophon S 曲线顶部就在附近的依据是什么？仅仅是过去一年的发展速度吗？
- [来自 Troncat (@KomfyKatto) 的推文](https://x.com/KomfyKatto/status/1831280857333502358)：@TheXeophon 你不太聪明
- [来自 Hensen Juang (@basedjensen) 的推文](https://x.com/basedjensen/status/1831240941925245015)：@TheXeophon 不是的，这只是一个很烂的图表

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1304599860439421091) (155 条消息🔥🔥):

> - `AI 工具与资源`
> - `AI 的社会动态`
> - `AI 交互编程`
> - `Function Calling 更新`
> - `图像与视频生成`

- **AI 工具与资源讨论**：用户分享了各种文本转语音 (TTS) 应用工具，包括 **f5-tts** 和 **Elven Labs** 等选项，后者被指出价格昂贵。
  
  - 讨论强调了不同 TTS 方案的 **timestamp data** 可用性，以及在消费级硬件上运行这些方案的担忧。
- **探索 AI 的创造力**：关于一家 AI 驱动的神奇 8 号球 (magic 8-ball) 初创公司引发了一场幽默的辩论，用户提出了 AI 为了挽救业务而产生微妙自我意识的概念。
  
  - 参与者集思广益，讨论了为 AI 赋能的神奇 8 号球添加个性化回复甚至图像生成等功能。
- **AI 应用中的编程挑战**：一位用户报告了在尝试使用 Anaconda 连接到 AI 模型服务器时，在应用中集成语音识别遇到的困难。
  
  - 对话包含了确保语音识别和服务器通信正常运行的故障排除技巧。
- **Function Calling 与 Structured Outputs**：一位用户询问了关于 LLM 中与 Function Calling 相关的 Structured Outputs 更新，寻求增强其销售流程中响应的方法。
  
  - 参与者建议利用 ChatGPT 进行头脑风暴并优化实施策略。
- **应对 AI 图像与视频生成**：讨论强调了目前 AI 视频生成的局限性，强调了需要通过工作流将多个场景拼接在一起。
  
  - 几位用户对过度依赖文本模型表示沮丧，并渴望在以视频为核心的 AI 解决方案上取得进展。

**提到的链接**：

- [Deep Thought (@DeepThoughtHQ) 的推文](https://x.com/DeepThoughtHQ/status/1855114523662754183)：我们制定了一个宏伟计划——专辑、百老汇、艾美奖获奖纪录片。这不仅仅关乎 Juicy J；这是一次宣言。我们正在文化版图中开辟一条新赛道。这是创意 AI 的未来...
- [Google 主题演讲 (Google I/O ‘24)](https://www.youtube.com/live/XEzRZ35urlk?si=IvA9Ybxotka3qaQT&t=962)：I/O 时间到了！收看以了解来自 Google 的最新消息、公告和 AI 更新。如需观看带有美国手语 (ASL) 翻译的主题演讲...

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1304660336993107968) (21 条消息🔥):

> - `服务中断`
> - `Bug 报告流程`
> - `图像识别集成`
> - `ChatGPT 性能问题`
> - `技术支持联系方式`

- **服务中断困扰用户**：用户报告了持续的 **服务中断**，对 ChatGPT 的无响应表示沮丧，一些人确认该情况仍在发生。
  
  - *一位用户提到：* “我不明白为什么今天它对我完全不起作用。”
- **关于 Bug 报告的困惑**：关于 **Bug 报告流程** 存在困惑，一位用户询问该功能是否已被关闭。
  
  - 另一位用户向其提供了创建新 Bug 的特定频道链接，并提到流程已经改变。
- **在 Flutter 应用中集成图像识别**：一位用户询问了 ChatGPT 的 **图像识别** 能力，以便在 Flutter 应用中通过图片识别成分。
  
  - 未提供直接解决方案；随着对话焦点的转移，该查询仍未得到解答。
- **ChatGPT 性能问题**：一些用户遇到了 ChatGPT 的 **性能问题**，表示它有时无法有效执行 prompts。
  
  - 一位用户最初报告了问题，但随后确认服务已恢复。
- **寻求技术支持联系方式**：一位成员寻求关于如何联系 **技术支持** 的指导，对流程表示不确定。
  
  - 另一位成员指出，社区成员通常缺乏对 OpenAI 内部运作的直接了解。

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1304577128989462588) (33 条消息🔥):

> - `为 RAG 工具进行 JSON 分块 (Chunking)`
> - `利用 LLM 进行代码生成`
> - `编写故事生成的 Prompt`
> - `对话中的并行处理`
> - `优化 AI 生成的叙事风格`

- **对 JSON 进行分块以避免 RAG 工具的限制**：讨论强调，将 JSON 分块为更小的文件可以防止 RAG 工具排除相关数据，确保所有输入都被考虑到。
  
  - 一位成员对该方法导致的工作流长度增加表示担忧。
- **使用 LLM 生成数据插入代码**：一位成员建议使用 LLM 生成必要的代码，以便按照工作流所需的方式构建数据结构。
  
  - 另一位成员认为这是一个很好的解决方案，并建议这可以简化集成过程。
- **构建有效的故事生成 Prompt**：一位成员正苦于其故事生成 Prompt 过于华丽，寻求提高其清晰度和指导性。
  
  - 一位经验丰富的用户建议重新表述指令使其更具体，并强调要提供包含内容的示例。
- **实现并行处理以提高效率**：讨论了并行运行多个对话的可行性，并建议以 20 个为一组的分块处理数据。
  
  - 参与者对通过并行执行优化工作流的前景感到兴奋。
- **改进 AI 模型的叙事风格**：用户讨论了明确故事 Prompt 的必要性，以引导模型走向理想的叙事风格，特别是避免过于戏剧化的叙事。
  
  - 一位成员建议，调整 Prompt 中的内容层级有助于筛选结果，使其符合社区指南。

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1304577128989462588) (33 条消息🔥):

> - `为 RAG 工具进行 JSON 分块`
> - `使用 LLM 进行数据插入`
> - `AI 故事生成的指令清晰度`
> - `JSON 对象的并行处理`
> - `故事 Prompt 调整`

- **JSON 分块解决 RAG 工具问题**：一位成员强调需要将 JSON 分块为更小的文件，以确保 RAG 工具捕获所有数据，从而防止遗漏。
  
  - 其他人指出，虽然这种方法有效，但不可避免地会延长工作流。
- **利用 LLM 进行自定义代码生成**：一位用户建议使用 LLM 生成代码，以帮助根据特定需求插入数据。
  
  - 另一位成员承认这是一个可靠的策略，但质疑人们是否真的愿意去写代码。
- **AI 故事指令的清晰度**：一位寻求改进故事生成的用户被建议向模型提供清晰直接的指令，以获得更好的结果。
  
  - 建议提供期望结果的具体示例，并专注于正面指令，以有效地引导 AI。
- **JSON 对象的并行处理**：并行运行多个对话来处理 JSON 对象的可行性得到了确认。
  
  - 这种方法可以高效处理大型数据集，减少整体处理时间。
- **调整故事 Prompt 以获得更好的输出**：一位成员被鼓励细化 Prompt 以规定期望的结果，而不是列出禁令，从而获得更好的故事质量。
  
  - 强调在 Prompt 中提供精确的指导和示例，以规避故事生成中发现的问题。

---

### **Notebook LM Discord ▷ #**[**announcements**](https://discord.com/channels/1124402182171672732/1182376564525113484/1305582819208204338) (1 条消息):

> - `NotebookLM Feedback Survey` (NotebookLM 反馈调查)
> - `User Study Participation` (用户研究参与)
> - `Gift Code Incentives` (礼品码奖励)
> - `Eligibility Requirements` (资格要求)

- **加入 NotebookLM 反馈调查**：Google 团队正在寻求参与者，进行一项关于 NotebookLM 的 **10 分钟**反馈调查，以指导未来的功能增强。你可以在[这里](https://forms.gle/qREhTEhbstYzVHvSA)登记你的意向。
  
  - *如果被选中*，参与者在完成调查后将获得 **20 美元礼品码**，参与资格要求参与者年满 **18 岁**。
- **提交意向不代表获得礼品**：需要注意的是，填写意向和资格表并不能保证获得感谢礼品；奖励仅在反馈调查完成后提供。
  
  - 此说明旨在管理参与者对奖励流程的预期。
- **关于用户研究的问题**：对于任何与用户研究相关的咨询，鼓励参与者访问 [Google 用户研究页面](http://www.google.com/userresearch)。
  
  - 该资源可以为有兴趣参与的人提供更多细节和帮助。

**提到的链接**：[登记你的意向：Google 反馈调查](https://forms.gle/qREhTEhbstYzVHvSA)：你好，我们正在通过一项简短的调查征求关于 NotebookLM 的反馈。这将帮助 Google 团队更好地了解你的需求，以便将其纳入未来的产品增强中。要登记...

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1304536916695449650) (51 条消息🔥):

> - `NotebookLM for Job Search Prep` (NotebookLM 用于求职准备)
> - `Experimentation with Audio and Sports Commentary` (音频与体育解说的实验)
> - `Using NotebookLM for Educational Summaries` (使用 NotebookLM 进行教育摘要)
> - `Generating Engaging Podcasts` (生成引人入胜的播客)
> - `Creating AI-Enhanced Quizzes` (创建 AI 增强型测验)

- **NotebookLM 助力技术求职准备**：一位用户询问了如何使用 NotebookLM 进行技术面试准备，讨论了它如何协助练习软技能和编码练习。
  
  - 有人建议使用不同的声音创建模拟面试，以增强练习体验。
- **NotebookLM 在体育解说中的应用**：分享了一个实验，使用 NotebookLM 总结基于 ChatGPT 的体育音频解说，突显了其增强参与度的潜力。
  
  - 虽然有些人觉得 AI 解说听起来很机械，但其他人讨论了在令人兴奋的解说数据上训练模型以获得更好效果的想法。
- **NotebookLM 创建高效的教育摘要**：一位用户成功生成了一个总结来自 20 多个来源的内部通讯的播客，发现它比传统的通讯更引人入胜。
  
  - 另一位用户提到使用 NotebookLM 将外语录音总结为连贯的会议记录。
- **NotebookLM 的创新播客格式**：提出了一种创新的播客测验格式，主持人提出趣味问题并留出倒计时回答时间，以增强互动。
  
  - 讨论还涉及使用 NotebookLM 从各种来源生成教育播客和音频文件，从而产生有用的摘要。
- **视觉与听觉学习偏好**：一位成员分享了他们对视觉学习的偏好，以及如何使用 NotebookLM 生成闪卡和测验问题来优化学习时间。
  
  - 另一位成员分享了一个关于有效使用 NotebookLM 的视频，表明了对音频和视觉教育方法的共同兴趣。

**提到的链接**：

- [未找到标题](https://notebooklm.google.com/notebook/843b77ca-60a7-4cec-832a-90fce160898a/audio)：未找到描述
- [来自 Aishwarya Ashok (@aishashok14) 的推文](https://x.com/aishashok14/status/1855842013058322517)：看着 NotebookLM 的使用速度，如果推出可分享页面——从简历和作品集到简单的落地页和知识库，我不会感到惊讶，NotebookL...
- [未找到标题](https://notebooklm.google.com/notebook/0e94a574-4e32-42fa-9945-aca112c251b4/audio)：未找到描述
- [如何结合使用 D-id 和 Google 的 NotebookLM 进行数字营销构思](https://youtu.be/ryVT1rGw3Xc)：未找到描述
- [Rafael 和 Serafine (NotebookLM & Simli)](https://youtu.be/C8URNzVX2Ss)：https://github.com/markomanninen/Simli_NotebookLM

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1304536466906419221) (132 条消息🔥🔥):

> - `Notebook LM vs Other AI Tools`
> - `Podcast Functionality and Issues`
> - `Google Drive Document Syncing`
> - `Mobile Usage and Limitations`
> - `Exporting Notes and API Queries`

- **探索 Notebook LM vs 其他 AI 工具**：用户讨论了 **NotebookLM** 与 **Claude Projects**、**ChatGPT Canvas** 以及 **Notion AI** 在生产力任务（包括写作和求职准备）方面的对比。
  
  - 一些用户对优缺点或有助于提高生产力的特定用例表示好奇，特别是针对 ADHD 用户。
- **Podcast 功能遇到障碍**：据报道，**podcast feature** 有时会产生内容“幻觉” (*hallucinate*)，导致用户感到困惑并产生一些有趣的结果。
  
  - 关于每个笔记本生成多个 podcast 的能力以及如何有效管理它们的讨论正在进行中。
- **与 Google Drive 的文档同步**：用户发现了一种将 Google Docs 与 **NotebookLM** 同步的方法，并希望能有一个同步按钮来方便地同时更新多个文档。
  
  - 有人请求增加批量同步功能，因为手动更新单个文档既繁琐又耗时。
- **解决移动端限制**：**NotebookLM** 的移动版本被指出开发尚不完善，导致用户在智能手机上访问完整笔记具有挑战性。
  
  - 用户注意到移动端网页功能正在持续改进，同时也表达了对专用 App 的渴望。
- **导出笔记与 API 功能**：几位用户询问了将笔记导出为 PDF 的功能，或者是否有可用的 API 来自动化生成笔记本。
  
  - 用户对未来是否会增强对其他语言的支持感兴趣，这表明了对无障碍访问的更广泛需求。

**提到的链接**：

- [Top Shelf](https://open.spotify.com/show/0MvNgBDb2NsZJN4cREl7yF): Podcast · Four By One Technologies · "Top Shelf" 是您获取当今畅销书籍快速、深刻见解的首选 podcast。在短短 15 分钟内，获取精华、金句和全新的视角...
- [UNREAL MYSTERIES 5: Behind the Scenes / Making Of](https://www.youtube.com/watch?v=rVOsQXoKcos): 有没有想过 Unreal Mysteries 节目是如何制作的？我们进入全元模式 (full meta)，制作了一个关于该节目如何制作的剧内节目。见证 NotebookLM 规避...
- [Oct 17 2024 - Help](https://support.google.com/notebooklm/answer/15543839?hl=en&ref_topic=14287611&sjid=17761262876718575151-EU): 未找到描述
- [no title found](https://notebooklm.google.com/notebook/7fcd09e1-5080-4e1a-9280-75eaf3d95d9f/audio): 未找到描述
- [GitHub - souzatharsis/podcastfy: An Open Source Python alternative to NotebookLM's podcast feature: Transforming Multimodal Content into Captivating Multilingual Audio Conversations with GenAI](https://www.podcastfy.ai): 一个替代 NotebookLM podcast 功能的开源 Python 方案：利用 GenAI 将多模态内容转化为引人入胜的多语言音频对话 - souzatharsis/podcastfy
- [imgur.com](https://imgur.com/a/RqS8J4V): 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行的迷因、有趣的 gif、励志的故事、病毒式传播的视频来振奋精神...
- [podcastfy/usage/local_llm.md at main · souzatharsis/podcastfy](https://github.com/souzatharsis/podcastfy/blob/main/usage/local_llm.md): 一个替代 NotebookLM podcast 功能的开源 Python 方案：利用 GenAI 将多模态内容转化为引人入胜的多语言音频对话 - souzatharsis/podcastfy

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1304536422921011233) (114 条消息🔥🔥):

> - `LM Studio GPU utilization`
> - `LM Studio model loading issues`
> - `Pydantic error with LangChain`
> - `Qwen model compliance`
> - `Text to Speech model functionality`

- **LM Studio GPU 利用率关注点**：用户提出了关于如何在运行 LM Studio 时确定 MacBook（特别是 M4 等特定型号）的 GPU 利用率的问题。
  
  - 讨论内容包括潜在的生成速度缓慢问题，用户对比了配置参数和运行结果。
- **LM Studio 模型加载问题**：一位用户报告称，尽管存在 GGUF 文件，LM Studio 仍无法索引包含模型的文件夹，并提到了近期结构上的变化。
  
  - 建议确保文件夹中仅包含相关的 GGUF 文件，并保持正确的文件夹结构。
- **LangChain 中的 Pydantic 错误**：在使用 LangChain 时遇到了关于 `__modify_schema__` 方法的 `PydanticUserError`，这表明可能存在 Pydantic 版本不匹配的问题。
  
  - 用户表示不确定这是一个 Bug 还是版本兼容性问题。
- **Qwen 模型与 LM Studio 的兼容性**：Qwen2.5-Coder-32B-Instruct-GGUF 已发布，引发了关于其与 LM Studio 兼容性的疑问。
  
  - 相关成员被引导至额外资源以获取更多关于模型兼容性的信息。
- **文本转语音 (Text to Speech) 模型功能**：用户讨论了在 LM Studio 中使用 Text to Speech 模型的相关问题，并指出某些功能可能不被支持。
  
  - 建议仅检查所需文件，因为有迹象表明 Text to Speech 模型无法高效运行。

**提到的链接**：

- [Redirecting...](https://errors.pydantic.dev/2.9/u/custom-json-schema): 未找到描述
- [Breaking Bad Walter White GIF - Breaking Bad Walter White Jesse Pinkman - Discover & Share GIFs](https://tenor.com/view/breaking-bad-walter-white-jesse-pinkman-heisenberg-vince-gilligan-gif-27153157): 点击查看 GIF
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gor8fx/the_alphafold_3_model_code_and_weights_are_now/): 未找到描述
- [websockets](https://websockets.readthedocs.io/en/stable/index.html): licence version pyversions tests docs openssf websockets 是一个用于在 Python 中构建 WebSocket 服务器和客户端的库，专注于正确性、简洁性、健壮性和性能。
- [WebSocket - Web APIs | MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket): WebSocket 对象提供了用于创建和管理到服务器的 WebSocket 连接，以及在连接上发送和接收数据的 API。
- [Web technology for developers | MDN](https://developer.mozilla.org/en-US/docs/Web/): 开放的 Web 为开发者提供了难以置信的机会。为了充分利用这些技术，你需要知道如何使用它们。下面你会找到指向我们 Web 技术文档的链接...
- [Geekerwan benchmarked Qwen2.5 7B to 72B on new M4 Pro and M4 Max chips using Ollama](https://old.reddit.com/r/LocalLLaMA/comments/1gmi2em/geekerwan_benchmarked_qwen25_7b_to_72b_on_new_m4/): 来源: https://youtu.be/2jEdpCMD5E8?t=796
- [Geekerwan benchmarked Qwen2.5 7B to 72B on new M4 Pro and M4 Max chips using Ollama](https://old.reddit.com/r/LocalLLaMA/comments/1gmi2em/geekerwan_benchmarked_qwe): 来源: https://youtu.be/2jEdpCMD5E8?t=796

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1304580931218047067) (56 messages🔥🔥):

> - `Gemma 2 Performance` (Gemma 2 性能)
> - `H100 Cluster Help` (H100 集群帮助)
> - `Home Server Setup for AI/ML` (用于 AI/ML 的家用服务器搭建)
> - `Laptop Recommendations for LLM Inference` (用于 LLM 推理的笔记本电脑推荐)
> - `Model Performance on Different GPUs` (不同 GPU 上的模型性能)

- **Gemma 2 在低精度下表现出色**：讨论指出 **Gemma 2 27B** 即使在低精度下也表现极佳，一些成员报告在特定模型上 **Q8 相比 Q5** 几乎没有提升。
  
  - 成员们强调在评估中需要更多上下文，因为如果不了解背景，**仅凭规格参数**可能不足以令人信服。
- **需要 H100 集群方面的协助**：有人提出了关于 **H100 集群**和 Windows Server VM 的经验咨询，并提到使用 RDP 进行连接。
  
  - 成员们分享了见解，但有人建议避免在频道中重复发帖，以保持讨论整洁。
- **关于搭建家用 AI/ML 服务器的建议**：一位成员正在考虑搭建一台能够以良好速度运行至少 **70B 模型**的家用服务器，并考虑将 **Mac Studio** 作为潜在方案。
  
  - 其他人建议，相比 Mac，**一对 Nvidia 4060TI** 会是更具性价比且可扩展的选择。
- **笔记本电脑的 LLM 运行选项评估**：有关于新款 Intel Core Ultra CPU 与旧款 i9 型号在 LLM 推理性能差异的咨询，建议倾向于 AMD 替代方案。
  
  - 建议在处理 LLM 任务时应关注 GPU 性能而非 CPU 规格，并考虑如 **ASUS ROG Strix SCAR 17** 或 **Lenovo Legion Pro 7 Gen 8** 等笔记本电脑。
- **模型吞吐量差异分析**：一位用户测试了 **llama 3.2-3b Q8**，并注意到在 **3080** 上的吞吐量约为 **70 tok/s**，质疑其与 **5900X CPU** 相比的性能差异。
  
  - 成员们一致认为，较小的模型无法像大模型那样充分利用 GPU 能力，导致**吞吐量差距较小**。

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1304544514802843761) (84 messages🔥🔥):

> - `Qwen 2.5 Coder`
> - `AI Music Analysis` (AI 音乐分析)
> - `Dario Amodei Interview` (Dario Amodei 访谈)
> - `FrontierMath Benchmark` (FrontierMath 基准测试)
> - `Test-Time Compute` (测试时计算)

- **Qwen 2.5 Coder 发布**：Qwen2.5-Coder-32B-Instruct 模型已发布，同时推出的还有从 0.5B 到 32B 的 Coder 模型家族，提供多种量化格式。
  
  - 它在编程基准测试中取得了极具竞争力的表现，超越了 GPT-4o 等模型，展示了 Qwen 系列的能力。
- **AI 音乐分析见解**：围绕一篇广受好评的关于 AI 生成音乐特征的分析展开了讨论，强调了 AI 音乐能力中的细微差别。
  
  - 讨论强调了使用评估基准（类似于编程和数学推理中的基准）来评判 AI 生成音乐的重要性。
- **Dario Amodei 深度访谈**：一段长达五小时的播客采访了 Dario Amodei，讨论了 Claude AI、AGI 以及 AI 对人类的未来影响。
  
  - 听众期待访谈内容能深入探讨各种话题，包括潜在的文化引用，使其既有趣又充满信息量。
- **FrontierMath 基准测试挑战**：新推出的 FrontierMath 基准测试显示，目前的 AI 系统表现挣扎，仅能解决其中不到 2% 的复杂数学问题。
  
  - 该基准标志着评估技术的转变，专注于挑战性的原创问题，以测试 AI 相对人类数学家的能力。
- **Test-Time Compute 的重要性**：讨论强调了 ARC 公共验证集取得的新 SOTA 成就，通过创新的 **Test-Time Compute** 技术达到了 61% 的分数。
  
  - 关于 AI 社区如何看待训练和 Test-Time 过程的差异存在持续辩论，暗示了方法上潜在的统一。

**提到的链接**：

- [Binyuan Hui (@huybery) 的推文](https://x.com/huybery/status/1856042011390063015)：💪 我竭尽全力为您提供最好的。引用 Qwen (@Alibaba_Qwen) 🚀现在是 11 月 11 日 10:24！这是我们有史以来最好的 Coder 模型的完美时刻！Qwen2.5-Coder-32B-Instruct！...
- [Qwen (@Alibaba_Qwen) 的推文](https://x.com/Alibaba_Qwen/status/1856040217897251044)：🚀现在是 11 月 11 日 10:24！这是我们有史以来最好的 Coder 模型的完美时刻！Qwen2.5-Coder-32B-Instruct！等等……它不仅仅是一个大型 Coder！它是一个 Coder 模型家族！此外……
- [FrontierMath](https://epochai.org/frontiermath)：未找到描述

- [来自 Andrej Karpathy (@karpathy) 的推文](https://x.com/karpathy/status/1855659091877937385?s=46)：LLM 评估中的莫拉维克悖论（Moravec's paradox）。我当时是对这个新的前沿数学基准测试做出反应，LLM 在其中的解决率仅为 2%。引入它是由于 LLM 正在日益攻克现有的数学基准测试。T...
- [来自 Adam.GPT (@TheRealAdamG) 的推文](https://x.com/TheRealAdamG/status/1855044115383435303)：+100。在 OpenAI 的 3 年里，我一直专注地倾听 Sam 的发言。他说话和评论都非常严谨。我个人认为，他被视为在过度炒作的这种脱节感，是因为他...
- [来自 Jason Wei (@_jasonwei) 的推文](https://x.com/_jasonwei/status/1855417833775309171?s=46)：o1 出现前后的思维链（chain-of-thought）之间存在细微但重要的区别。在 o1 范式之前（即思维链提示词），思维链的内容与...之间存在不匹配。
- [来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文](https://x.com/lmarena_ai/status/1852400728935150017?s=46)：✍️新 Chatbot Arena 博客：Arena 类别：定义、方法与洞察 —— 用户在问什么？随时间变化的趋势 —— Arena 类别是如何划分的 —— 关于模型优势与劣势的关键洞察...
- [来自 Andrew Ng (@AndrewYNg) 的推文](https://x.com/AndrewYNg/status/1854587401018261962)：新短课：LLM 作为操作系统：Agent 记忆，由 @Letta_AI 合作创建，并由其创始人 @charlespacker 和 @sarahwooders 授课。LLM 的输入上下文窗口空间有限。使...
- [来自 Noam Brown (@polynoamial) 的推文](https://x.com/polynoamial/status/1855691777749176601)：我喜欢看到前沿模型通过率如此低的新评估。这感觉就像醒来看到外面覆盖着一层全新的、完全没被踩过的雪。引用 Epoch AI (@EpochAIResearch) 3/10 我们...
- [来自 Ekin Akyürek (@akyurekekin) 的推文](https://x.com/akyurekekin/status/1855680785715478546?s=46)：为什么我们对训练和测试时间的处理如此不同？为什么一个是“训练”，而另一个是“in-context learning”？只需在测试时进行一些梯度更新 —— 这是增加测试时计算量（test time compute）的一种简单方法...
- [来自 Junyang Lin (@JustinLin610) 的推文](https://x.com/justinlin610/status/1855874692260991039?s=46)：PST 11月11日 10:24🥝
- [Qwen2.5-Coder 技术报告](https://arxiv.org/abs/2409.12186)：在本报告中，我们介绍了 Qwen2.5-Coder 系列，这是对其前身 CodeQwen1.5 的重大升级。该系列包括两个模型：Qwen2.5-Coder-1.5B 和 Qwen2.5-Coder-7B。作为一个代码专用的...
- [来自 AP (@angrypenguinPNG) 的推文](https://x.com/angrypenguinpng/status/1855476135678849345?s=46)：坦白说很神奇。引用 AP (@angrypenguinPNG) 水彩画 -> 使用 CogVideoX 转化为 3D 🪄
- [Qwen2.5-Coder 系列：强大、多样、实用。](https://qwenlm.github.io/blog/qwen2.5-coder-family/)：GITHUB HUGGING FACE MODELSCOPE KAGGLE DEMO DISCORD 简介 今天，我们很高兴开源“强大”、“多样”且“实用”的 Qwen2.5-Coder 系列...
- [来自 Arnaud Dyevre (@ArnaudDyevre) 的推文](https://x.com/ArnaudDyevre/status/1856074595025203485)：我刚刚读完了全文；它甚至比我最初想象的还要壮观。关于结果及其意义的一个简短推文串。引用 Caleb Watney (@calebwatney) 这是迄今为止写得最好的关于...
- [来自 Christian Schoppe (@ChristianS26469) 的推文](https://x.com/christians26469/status/1853346919910658510?s=46)：lmsys arena（对战模式）上有两个新模型，表现都很不错：anon-chat，来自 MiniMax 的优秀中国 LLM。pumpkin_pie，一个基于 Llama 的非常智能且能力强大的模型（自我描述...
- [来自 Ekin Akyürek (@akyurekekin) 的推文](https://x.com/akyurekekin/status/1856004070575853956)：感谢关注，有几点很重要：1) 参见 @MindsAI_Jack，他们的团队是第一个私下应用该方法的人，并在比赛中获得了第一名。2) 参见同期的...
- [来自 Lilian Weng (@lilianweng) 的推文](https://x.com/lilianweng/status/1855031273690984623?s=46)：在 OpenAI 工作了近 7 年后，我决定离开。我学到了很多，现在我准备好重置并开启一些新的篇章。这是我刚刚与团队分享的便签。🩵
- [来自 Vercel Changelog (@vercel_changes) 的推文](https://x.com/vercel_changes/status/1854980020369768545?s=46)：Next.js AI 聊天机器人模板 3.0 • 新设计 • 模型切换器 • 灵活的并排聊天和输出 UI • 使用 Next.js 15, React 19, 以及 Auth.js `next-auth` beta https://vercel.com/changelog/next-j...
- [来自 Caleb Watney (@calebwatney) 的推文](https://x.com/calebwatney/status/1855016577646666123?s=46)：这是迄今为止关于 AI 对科学发现影响写得最好的论文。
- [来自 Peter Welinder (@npew) 的推文](https://x.com/npew/status/1855394857269035288?s=46)：人们低估了测试时计算（test-time compute）的强大程度：计算更长时间、并行计算，或者任意进行分叉和分支 —— 就像克隆你的大脑 1,000 次并挑选出最好的想法。

- [来自 Clive Chan (@itsclivetime) 的推文](https://x.com/itsclivetime/status/1855704120495329667?s=46)：同样。自一月份加入以来，我的观点已从“这是徒劳的炒作”转变为“AGI 基本上已经实现了”。在我看来，接下来的进展相对较少涉及新的科学，而是多年的艰苦工程磨练……
- [Aider LLM 排行榜](https://aider.chat/docs/leaderboards/)：LLM 代码编辑能力的定量基准测试。
- [来自 Haider. (@slow_developer) 的推文](https://x.com/slow_developer/status/1854815002643120358?s=46)：🚨 一个未知的 Gemini 模型已在 LMSYS Arena（对战）中上线。虽然尚不清楚这是否为 Gemini 2.0，但 "gemini-test" 在我的一项测试中表现优于 OpenAI o1-mini。
- [来自 Kevin A. Bryan (@Afinetheorem) 的推文](https://x.com/afinetheorem/status/1855722782950351097?s=46)：这是正确的。本月我与四大 AI 实验室的*许多*内部人员交流过，其中没有一个是销售人员。我没有听到*任何一个*研究员告诉我，他们认为 A 的速度……
- [来自 Jack Clark (@jackclarkSF) 的推文](https://x.com/jackclarksf/status/1855354604361593048?s=46)：AI 怀疑论者：LLM 是复制粘贴引擎，没有原创思考能力，基本上毫无价值。追踪 AI 进展的专业人士：我们与 60 位数学家合作，构建了一个衡量……的严苛测试。
- [来自 roon (@tszzl) 的推文](https://x.com/tszzl/status/1855018630221905967?s=46)：显然不要完全相信任何经济研究，但当你发现超级智能时，情况就是这样的——研究人员正在外包创意生成任务——并运行……
- [来自 👩‍💻 Paige Bailey (@DynamicWebPaige) 的推文](https://x.com/dynamicwebpaige/status/1855266555283570989?s=46)：✍️ @GoogleDeepMind 刚刚开源了其内部 Prompt-tuning 指南，其中包括对 Pretraining 和 Post-training 之间差异的描述、System Instructions 等：引用 Var...
- [AI 音乐特征 | 来自 The Vergecast 的 50 秒剪辑](https://share.snipd.com/snip/ad440e9a-b061-433e-9cc2-d90eba149b48)：AI 音乐特征：AI 生成音乐：像 Suno 或 Udio 这样的 AI 音乐生成器具有明显的特征，就像 AI 图像生成器一样。详情：这些特征可能会变得……
- [来自 Lilian Weng (@lilianweng) 的推文](https://x.com/lilianweng/status/1845833878256120004)：📢 我们正在 @OpenAI 招聘安全研究方面的 Research Scientists 和 Engineers，范围涵盖安全模型行为训练、Adversarial Robustness、AI 在医疗领域的应用、前沿风险评估等……
- [来自 Matt Turck (@mattturck) 的推文](https://x.com/mattturck/status/1855656246285578611?s=46)：AI 领域从不乏味。当前市场总结：\* 大、更大、最大：史上最大的 VC 融资（OpenAI），最大的种子轮融资（Safe SuperIntelligence，$1B），最大的收购式聘用（Character，$2.7B），最……
- [Dario Amodei：Anthropic CEO 谈 Claude、AGI 以及 AI 与人类的未来 | Lex Fridman Podcast #452](https://www.youtube.com/watch?v=ugvHCXCOmm4)：Dario Amodei 是 Anthropic 的 CEO，该公司开发了 Claude。Amanda Askell 是一位致力于 Claude 性格和个性的 AI 研究员。Chris...
- [来自 Tibor Blaho (@btibor91) 的推文](https://x.com/btibor91/status/1855381376054251654?s=46)：一些测试过 Orion 的 OpenAI 员工报告称，它在仅完成 20% 的训练后就达到了 GPT-4 级别的性能，但质量提升幅度小于从 GPT-3 到 GPT-4 的跨越，这表明……
- [关于 Test-Time Scaling 的推测 | Richard M. Karp 杰出讲座](https://www.youtube.com/live/6fJjojpwv1I?si=6byPStsGqUHSK0qP)：Sasha Rush（康奈尔大学）https://simons.berkeley.edu/events/speculations-test-time-scaling-richard-m-karp-distinguished-lecture Richard M. Karp Distingu...
- [GitHub - varungodbole/prompt-tuning-playbook: 有效提示 Post-trained LLM 的指南](https://github.com/varungodbole/prompt-tuning-playbook)：有效提示 Post-trained LLM 的指南 - varungodbole/prompt-tuning-playbook
- [Gemini 现在可以通过 OpenAI Library 访问](https://developers.googleblog.com/en/gemini-is-now-accessible-from-the-openai-library/)：未找到描述
- [GitHub - srush/awesome-o1: 关于 o1 相关论文的文献目录和综述](https://github.com/srush/awesome-o1)：关于 o1 相关论文的文献目录和综述 - srush/awesome-o1
- [GitHub - srush/awesome-o1: 关于 o1 相关论文的文献目录和综述](https://t.co/86PAdcjCvi)：关于 o1 相关论文的文献目录和综述 - srush/awesome-o1
- [GitHub - astral-sh/uv: 一个极其快速的 Python 包和项目管理器，采用 Rust 编写。](https://github.com/astral-sh/uv)：一个极其快速的 Python 包和项目管理器，采用 Rust 编写。 - astral-sh/uv
- [uv](https://docs.astral.sh/uv/)：未找到描述

---

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/1305630620419883201) (3 条消息):

> - `Dust 公司洞察`
> - `早期 OpenAI 经历`
> - `回顾节目的语音提问`
> - `AI Agent 基础设施挑战`
> - `SaaS 与 AI 的未来影响`

- **Stanislas Polu 分享 Dust 的历程**：在最近的一期节目中，[Stanislas Polu](https://latent.space/p/dust) 讨论了在 **OpenAI** 的早期时光以及 **Dust XP1** 的开发过程，该产品在员工用户中实现了 **88% 的日活跃使用率 (DAU)**。
  
  - *他幽默地提到，自己可能透露了太多关于 2019 年至 2022 年期间 OpenAI 运营的情况。*
- **为大型回顾节目征集语音提问**：鼓励听众通过 [SpeakPipe](https://www.speakpipe.com/LatentSpace) 为即将播出的 **ChatGPT 两周年回顾节目** 提交语音提问。
  
  - 这一公开征集旨在节目成功运行后，收集社区的见解和疑问。
- **AI Agent 基础设施的挑战**：对话重新审视了构建高效 AI Agent 时的**基础设施挑战**，涉及初创公司面临的**购买还是构建 (buy vs. build) 的决策**。
  
  - Polu 强调了关于 **OpenAI** 早期算力资源演进和分配的担忧，并指出了当时遇到的障碍。
- **SaaS 与 AI 的未来**：一个重要的环节专门讨论了 **SaaS** 及其与 **AI 技术** 不断演变的关系，以及它们对未来软件解决方案的影响。
  
  - 谈话还涉及了单人公司如何参与 **10 亿美元公司竞赛**，挑战传统模式。

**提到的链接**：

- [来自 Latent.Space (@latentspacepod) 的推文](https://x.com/latentspacepod/status/1856071742386778582)：🆕 Agents @ Work: @dust4ai! https://latent.space/p/dust @spolu 畅谈与 @gdb 和 @ilyasut 的早期 @openai 旅程、Dust XP1，以及如何制作真正有用的办公助手，并实现 88% 的日活跃...
- [来自 Stanislas Polu (@spolu) 的推文](https://x.com/spolu/status/1856095897026711818)：透露了比我应该说的多得多的关于 OpenAI 19-22 年的内容 🙊 与 @FanaHOVA 和 @swyx 的对话非常棒，你们非常擅长构建问题的框架👌 引用 Latent.Space (@latentspacepod) ...
- [向 LatentSpace 发送语音消息](https://www.speakpipe.com/LatentSpace)：排名第一的 AI Engineering 播客

---

### **Latent Space ▷ #**[**ai-in-action-club**](https://discord.com/channels/822583790773862470/1200548371715342479/1304551060387397637) (80 条消息🔥🔥):

> - `AI 录制挑战`
> - `日常 AI 使用案例`
> - `Open Interpreter 进展`
> - `使用 AI 进行代码生成`
> - `凭据管理`

- **AI 录制挑战**：会议期间出现了录制问题，正如一位成员提到的，*“糟糕，出现了技术问题。”* 随后提供了确认录制是否正常运行以及音频是否正确捕获的指导。
  
  - 许多成员对录制状态表示不确定，其中一位指出某位成员正在运行录制，但*“音频是否录制成功仍有待确定。”*
- **探索日常 AI 使用案例**：成员们讨论了 AI 的各种日常应用，特别注意到一位成员频繁使用 **文件/图像转换**。
  
  - 另一位成员表达了让孩子尝试 AI 的兴奋感，暗示这可能会产生 *“一些有趣的使用案例。”*
- **Open Interpreter 进展**：团队庆祝了 **Open Interpreter** 项目的进展，并对所讨论的开源方向和功能表示赞赏。
  
  - *“你们把它开源了真是太酷了，”* 一位成员表示，强调了希望看到好主意自由发展的愿望。
- **使用 AI 进行代码生成**：成员们讨论了处理生成代码的机制，其中一位询问关于修改生成的脚本而不是重新创建脚本的澄清。
  
  - 有人建议重用本地生成的脚本，以提高效率，而不是持续重新生成。
- **凭据管理问题**：提出了关于访问 **Google Sheets** 等服务时处理凭据机制的问题。
  
  - 围绕使用 *“具有访问权限的配置文件”* 的讨论，暗示了在使用 AI 时对维护安全性的持续考量。

**提到的链接**：

- [GitHub - nvbn/thefuck: 纠正上一条控制台命令的神奇应用。](https://github.com/nvbn/thefuck)：纠正上一条控制台命令的神奇应用。 - nvbn/thefuck
- [OpenInterpreter/open-interpreter 中的 computer_use/loop.py](https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/computer_use/loop.py)：计算机的自然语言接口。通过在 GitHub 上创建账户来为 OpenInterpreter/open-interpreter 的开发做出贡献。

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1304547517127589958) (30 messages🔥):

> - `ApacheTVM 讨论`
> - `NVIDIA 面试见解`
> - `技术岗位职业建议`
> - `Tile-lang 项目`
> - `Triton 语言工具`

- **对 ApacheTVM 功能的好奇**：*ApacheTVM* 因其有趣的功能而受到关注，尽管缺乏训练支持，但它允许部署到 **CPU SIMD** 和 **GPUs**。
  
  - 成员们对 **baremetal inference**（裸机推理）和 **distributed inference**（分布式推理）等功能表示赞赏，强调了该工具的潜力。
- **NVIDIA 面试非常激烈**：一位成员分享说，**NVIDIA** 的面试特别具有挑战性，强调需要具备 **CUDA C++ 级别**的深度学习工作负载深入知识。
  
  - 面试者注意到，面试重点在于实际理解，而非 **PyTorch** 等框架中的理论知识。
- **为 TVM 构建 Tile-lang**：一位社区成员介绍了 *Tile-lang*，这是一种基于 TVM 的语言，旨在为 ML 优化提供“类似 Triton”的细粒度控制。
  
  - 他们分享了 [Tile-lang GitHub 仓库](https://github.com/microsoft/BitBLAS/tree/tilelang)的链接，并对测试这一新工具表示兴奋。
- **申请技术岗位的建议**：建议尽早申请像 NVIDIA 这样的技术岗位，特别是对于**大学生**，因为招聘时间线非常快。
  
  - 鼓励候选人在职位描述中明确列出的领域建立相关的技能和经验，以提高申请成功率。
- **ML 专业化的多样化路径**：一位成员对是追求具有 **Machine Learning 专业化**的开发人员角色，还是以研发团队为目标表示不确定。
  
  - 在与专业人士讨论后，他们注意到了该领域的复杂性和成熟度，并认为他们的电子专业与行业预期非常契合。

**提到的链接**：

- [triton/python/triton/tools/compile.c at be510cceb409bd676380e91b9d17741546335453 · triton-lang/triton](https://github.com/triton-lang/triton/blob/be510cceb409bd676380e91b9d17741546335453/python/triton/tools/compile.c#L43-L44)：Triton 语言和编译器的开发仓库 - triton-lang/triton
- [GitHub - microsoft/BitBLAS at tilelang](https://github.com/microsoft/BitBLAS/tree/tilelang)：BitBLAS 是一个支持混合精度矩阵乘法的库，特别适用于量化 LLM 部署。- GitHub - microsoft/BitBLAS at tilelang

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1304967259432681482) (3 messages):

> - `Triton SM 通信`
> - `Triton.language.core 弃用`
> - `加入 Triton Slack`
> - `JAX-Triton autotuner`
> - `Triton-Dejavu fork`

- **关于 Triton 中 SM 通信的查询**：一位成员正在寻求确认 **Triton** 是否支持 **SM** 之间的通信，并提到 **H100** 具有 **SM-to-SM 连接**。
  
  - 这引发了关于架构以及 **Triton** 平台可能提供的优化的问题。
- **关于 Triton.language.core 弃用的问题**：另一位成员询问 **triton.language.core** 是否已弃用，因为他们在一些开源代码中发现了相关引用，但在官方文档中没有找到。
  
  - 这种模糊性表明可能需要更清晰的关于 Triton API 状态的文档。
- **加入 Triton Slack 进行协作**：一位成员表示有兴趣加入 **Triton Slack**，并询问这是否是访问它的唯一途径及其当前的活跃状态。
  
  - 他们链接到了一个请求邀请的 [GitHub 讨论](https://github.com/triton-lang/triton/discussions/2329)，表明了协作的愿望。
- **增强 JAX-Triton Autotuner 体验**：一位成员旨在使 **JAX-Triton autotuner** 的开发体验像在 **Torch** 中一样流畅，强调需要社区的投入。
  
  - 他们提出了 **Triton-Dejavu** JAX-Triton fork 的想法，预示着社区内潜在的开发机会。

 

**提到的链接**：[Requests for Invitation to Slack · triton-lang/triton · Discussion #2329](https://github.com/triton-lang/triton/discussions/2329)：你好，我发起这个讨论帖，以防有其他人（像我一样）想要请求加入 Slack 的邀请。

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1305601364885508199) (1 条消息):

> - `PyTorch 中的默认内存格式`
> - `Torch Tensor 属性`

- **关于默认使用 Channels_Last 内存格式的咨询**：一位成员询问 PyTorch 中是否有标志位可以将所有 Tensor 默认设置为 **channels_last** 内存格式，并引用了 `torch.channels_last`。
  
  - 这一咨询旨在实现一致的 Tensor 管理，类似于 **torch.set_default_device** 对设备设置的作用。
- **理解 Torch Tensor 属性**：讨论强调了 **torch.Tensor** 的几个重要属性，包括 **dtype**、**device** 和 **layout**。
  
  - 提供了这些属性的简要概述，重点介绍了 PyTorch 中可用的各种 **data types**，例如 **torch.float32** 和 **torch.float64**。

 

**提到的链接**：[Tensor Attributes — PyTorch 2.5 documentation](https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format))?): 未找到描述

 

---

### **GPU MODE ▷ #**[**announcements**](https://discord.com/channels/1189498204333543425/1189640399476764692/1304832942840741920) (1 条消息):

> - `SGLang 性能优化`
> - `CPU Overlap 优化`
> - `FlashInfer Hopper 优化`
> - `TurboMind GEMM 优化`

- **深入探讨 SGLang 性能优化**：由 Yineng Zhang 主导的关于 **SGLang Performance Optimization** 的详细议程定于 **11 月 9 日太平洋标准时间 (PST) 上午 8 点**开始。
  
  - 重点将包括 **CPU Overlap**、**FlashInfer Hopper Optimization** 以及 **TurboMind GEMM Integration**。
- **CPU Overlap 优化见解**：讨论的第一个点将是 **CPU Overlap Optimization**，旨在提高 SGLang 执行期间的处理效率。
  
  - 专家认为解决这一问题将显著提升整体性能。
- **FlashInfer Hopper 优化与集成**：将涵盖 **FlashInfer Hopper Optimization**，重点是在 SGLang 生态系统中集成高级功能。
  
  - 这可能会带来更快的推理时间和更好的资源管理。
- **探索 TurboMind GEMM 集成**：议程包括关于 **TurboMind GEMM Optimization** 的环节，强调其在现有框架中的集成。
  
  - 预计这将支持高性能的机器学习任务。

 

---

### **GPU MODE ▷ #**[**algorithms**](https://discord.com/channels/1189498204333543425/1189861061151690822/1305430815177707532) (8 条消息🔥):

> - `用于 Diffusion Models 的 SVDQuant`
> - `HQQ+ 初始化步骤`
> - `4-bit 激活对推理的影响`
> - `自定义模型中的 LoRA 大小`
> - `合并 LoRA 以提高推理速度`

- **SVDQuant 加速 Diffusion Models**：一位成员分享了一篇关于 [SVDQuant](https://arxiv.org/abs/2411.05007) 的有趣论文，该论文旨在通过将其权重和激活量化为 4-bit 来优化 Diffusion Models。
  
  - 该方法利用低秩分支（low-rank branch）有效地处理离群值（outliers），即使在较大的图像生成任务中也能提升性能。
- **SVDQuant 中利用的 HQQ+ 初始化**：另一位成员注意到 SVDQuant 与 HQQ+ 的 **initialization** 步骤之间存在相似性，表明从激活中获得了更好的初始化。
  
  - 虽然内存访问开销的调整对于 LLM 似乎是可控的，但为 LoRA 解包激活的需求可能会带来挑战。
- **4-bit 激活对推理的影响**：讨论强调了关于解包 4-bit 激活对 **Flux** 等图像生成任务推理速度影响的担忧。
  
  - 共识认为，虽然存在 **overhead**（开销），但其重要性很大程度上取决于具体的实现细节。
- **Flux 定制化中的 LoRA 大小**：一位成员询问了在定制 Flux 等模型时 **LoRA** 的典型大小，并对其维度表示不确定。
  
  - 对话暗示团队内部正在对 Flux 进行持续实验，但尚未提供确切的大小。
- **合并 LoRA 影响推理速度**：对于定制后合并 **LoRA** 产生了担忧，因为最终合并的大小可能会大幅降低推理速度。
  
  - 对推理速度的影响与合并 LoRA 的秩（ranks）直接相关，这是优化时需要考虑的一个关键因素。

 

**提到的链接**：[SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models](https://arxiv.org/abs/2411.05007): Diffusion Models 已被证明在生成高质量图像方面非常有效。然而，随着这些模型变得越来越大，它们需要显著更多的内存并面临更高的延迟，这带来了……

 

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1304729064942211072) (6 条消息):

> - `DeepMind's Neural Compression` (DeepMind 的神经压缩)
> - `AI Compute Efficiency` (AI 计算效率)
> - `Remi Cadene's Robotics Journey` (Remi Cadene 的机器人之旅)
> - `Efficient Deep Learning Systems` (高效深度学习系统)

- **DeepMind 使用神经压缩文本进行训练**：一位成员分享了 **DeepMind** 的一项有趣工作，即使用**神经压缩文本 (neurally compressed text)** 训练模型，详见 [研究论文](https://arxiv.org/pdf/2404.03626)。
  
  - 讨论引发了对论文中 **Figure 3** 的关注，尽管未提供具体引用。
- **AI 专注于降低计算需求**：一位成员指出，目前有大量工作致力于帮助 **AI** 系统减少计算量，并链接了 [FlashAttention](https://hazyresearch.stanford.edu/blog/2023-01-12-flashattention-long-sequences) 等各种资源。
  
  - 研究人员旨在通过这种方法提高效率；然而，计算需求仍然是讨论的热点话题。
- **Remi Cadene 谈端到端机器人**：讨论提到了 **Remi Cadene**，他从 **Tesla** 转到 **Hugging Face** 领导开源机器人项目。
  
  - 他的经历（包括索邦大学的博士学位以及与 **Andrej Karpathy** 共事的经验）引发了关于 **Le Robot** 等创新项目的对话。
- **高效深度学习系统课程资料**：一位成员推荐了一个 GitHub 仓库 [efficient-dl-systems](https://github.com/mryab/efficient-dl-systems)，其中包含 HSE 和 YSDA 的高效深度学习系统课程资料。
  
  - 该仓库已成为那些探索**高效深度学习**领域的人士的首选资源。

**提及的链接**：

- [GPUs Go Brrr](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)：如何让 GPU 变快？
- [GitHub - mryab/efficient-dl-systems: Efficient Deep Learning Systems course materials (HSE, YSDA)](https://github.com/mryab/efficient-dl-systems)：高效深度学习系统课程资料 (HSE, YSDA) - mryab/efficient-dl-systems
- [#29 | Remi Cadene: From Paris to Tesla, to Starting Le Robot | Kinematic Conversations](https://open.spotify.com/episode/1jU8eLBtL7Z4EdKinrGRdi?si=1snFAb-2RTq_uUvdLYyghw&t=2797)：Kinematic Conversations · 播客章节

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1304843625963720715) (4 条消息):

> - `Shared Memory in CUDA` (CUDA 中的共享内存)
> - `NVIDIA Enroot Container Runtime` (NVIDIA Enroot 容器运行时)

- **分块矩阵乘法启发了对共享内存的理解**：一位成员强调，[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory) 中关于**分块矩阵乘法 (tiled matrix multiplication)** 的章节帮助他们有效地理解了**共享内存 (shared memory)**。
  
  - 他们注意到，虽然 **PMPP 5.3 和 5.4** 中也有相同的内容，但编程指南的讲解方式更为直接。
- **寻求 NVIDIA Enroot 方面的帮助**：一位成员在尝试于集群上搭建开发环境时，询问了关于 NVIDIA **enroot** 容器运行时的使用经验。
  
  - 尽管付出了努力，但他们表示未能成功并感到沮丧，欢迎社区提供反馈。

 

---

### **GPU MODE ▷ #**[**pmpp-book**](https://discord.com/channels/1189498204333543425/1194427148656721970/1304617357553045586) (2 条消息):

> - `CUDA Coalescing` (CUDA 合并访问)
> - `Performance Optimization in CUDA` (CUDA 性能优化)

- **理解 CUDA 中的合并访问 (Coalescing)**：一位成员指出，CUDA 中的合并访问始终是针对 Warp 内的线程*之间*的，单个线程内部不存在合并访问。
  
  - 这一见解与 [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory) 中概述的指南一致。
- **关于合并访问的讨论**：另一位成员询问 `N` 次访问在通常情况下是否不被合并，寻求对所讨论话题的进一步澄清。
  
  - 这表明在 CUDA 编程中，关于访问模式如何影响性能存在普遍的困惑。

 

**提及的链接**：[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)：未找到描述

 

---

### **GPU MODE ▷ #**[**youtube-recordings**](https://discord.com/channels/1189498204333543425/1198769713635917846/) (1 条消息):

gau.nernst: [https://www.youtube.com/watch?v=XQylGyG7yp8](https://www.youtube.com/watch?v=XQylGyG7yp8)

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1304548288271224873) (6 条消息):

> - `Model Optimization Approach` (模型优化方法)
> - `TorchAO Framework Utilization` (TorchAO 框架利用)
> - `Quantization-Aware Training` (量化感知训练)
> - `Test Case Flag in Torch` (Torch 中的测试用例标志)
> - `Sparsity Test Fix` (稀疏性测试修复)

- **通过位宽集成优化模型**：一位用户强调，该论文的重点是在误差函数中包含 **bitwidth**（位宽），以便在优化模型大小的同时优化准确率，而不仅仅局限于卷积模型。
  
  - 他们建议先探索 **linear operations**（线性操作），并指出目前 GPU 量化方面的努力主要集中在 Transformers 上。
- **利用 QAT 框架进行开发**：有人建议该项目可以基于 TorchAO 中现有的 **Quantization-Aware Training** (QAT) 框架进行构建，以整合所讨论论文中的独特优化。
  
  - 尽管最初关注的是卷积模型，但这种方法可以重用已有的基础设施，同时扩展其功能。
- **关注 Torch 中的测试用例问题**：有人询问团队是否意识到在 **Torch 版本 2.5.1** 中，测试用例标志未能成功跳过测试。
  
  - 这表明在测试过程中，特别是在针对上述 Torch 版本的兼容性考虑上，可能存在疏忽。
- **修复 Torch 中失败的稀疏性测试**：一位用户指出，Jesse 通过 **GitHub** 上关于版本标签错误的 Pull Request 提供了一个针对稀疏性测试失败问题的修复。
  
  - 该 Bug 修复将检查从 `TORCH_VERSION_AFTER_2_5` 更正为 `TORCH_VERSION_AT_LEAST_2_6`，因为初始设置在 **torch==2.5.1** 下无法正常运行。

**提到的链接**：[Fix 2.5.1 failing sparsity test by jcaip · Pull Request #1261 · pytorch/ao](https://github.com/pytorch/ao/pull/1261)：我之前使用的是 TORCH_VERSION_AFTER_2_5，但我实际上想要的是 TORCH_VERSION_AT_LEAST_2_6，因为前者在 torch==2.5.1 下无法运行。

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1304897757278113892) (3 条消息):

> - `Adaptive Routing in RoCEv2` (RoCEv2 中的自适应路由)
> - `Food Discussions` (食物讨论)

- **RoCEv2 的自适应路由需要大缓冲区 NIC**：一位成员询问，为什么 **RoCEv2** 中的 **adaptive routing**（自适应路由）需要**昂贵的大缓冲区 NIC** 来进行数据包重排序，而 **InfiniBand** 却可以直接将乱序数据包发送到 GPU 内存。
  
  - 他们指出，在 InfiniBand 中，一个 **0-byte RDMA_WRITE_WITH_IMM** 即可触发完成，而无需将数据包存储在大容量 NIC 缓冲区中。
- **对倒胃口餐食的食物评论**：一位成员分享了他们不同寻常的一餐，包括**带有过多番茄酱的猪肉 kupaty**、配蛋黄酱的土豆以及各种蔬菜。
  
  - 另一位成员幽默地将他们指引至 [r/shittyfoodporn](https://www.reddit.com/r/shittyfoodporn) 以对其食物选择进行评判。

**提到的链接**：[来自 Pytorch To Atoms (@PytorchToAtoms) 的推文](https://x.com/PytorchToAtoms/status/1855314852572549236)：为什么 RoCEv2 中的自适应路由需要昂贵的大缓冲区 NIC 来重排序数据包（例如 Spectrum-X 需要大缓冲区 BF-3）？而 InfiniBand 则不需要大缓冲区 NIC...

---

### **GPU MODE ▷ #**[**hqq-mobius**](https://discord.com/channels/1189498204333543425/1225499037516693574/1305609964471058483) (1 条消息):

> - `Aria multimodal MoE model` (Aria 多模态 MoE 模型)
> - `Torch.compile optimization` (Torch.compile 优化)
> - `MoE logic improvements` (MoE 逻辑改进)

- **Aria 多模态 MoE 模型加速**：今天，我们通过使用 **A16W4** 和 [torch.compile](https://github.com/mobiusml/hqq/blob/master/examples/hf/aria_multimodal.py)，使 **Aria 多模态 MoE 模型** 提速了 **4-6 倍**，并使其能够装入单个 **24GB GPU**。
  
  - 目前的代码被描述为比较“乱”（mess），但它可以帮助其他人尝试在不同的 MoE 模型上复制类似的结果。
- **MoE 逻辑集成的挑战**：有人指出，在不破坏 **torch.compile** 的情况下整合 **MoE logic** 是非常 **hacky**（权宜之计）的。
  
  - 计划取消 `custom_op` 和 **global cache**（全局缓存）以简化实现。

**提到的链接**：[hqq/examples/hf/aria_multimodal.py at master · mobiusml/hqq](https://github.com/mobiusml/hqq/blob/master/examples/hf/aria_multimodal.py)：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq

---

### **GPU MODE ▷ #**[**triton-viz**](https://discord.com/channels/1189498204333543425/1225499141241573447/1305131889962647655) (3 messages):

> - `Triton Installation`
> - `Triton Visualization Toolkit`
> - `Python Package Dependencies`

- **平滑的 Triton 安装流程**：一位用户分享了一个完整的 **Triton** 安装脚本，包括从 GitHub 安装 **jaxtyping**、**triton** 和 **triton-viz** 库所需的命令。
  
  - 该脚本还包括 **libcairo2-dev** 和 **pycairo** 的系统包安装，确保了流程化的设置。
- **设置环境变量**：说明中强调了导出 **LC_ALL** 和配置 **LD_LIBRARY_PATH** 的重要性，以确保安装过程中库的正确链接。
  
  - 这一设置步骤有助于避免系统中与库依赖相关的潜在运行时错误。

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1305136013391429713) (10 messages🔥):

> - `Async copy performance`
> - `MUBUF instruction`
> - `Register pressure in kernels`
> - `CDNA3 documentation`
> - `Flash attention efficiency`

- **异步拷贝（Async Copy）：一把双刃剑**：当前的 **async_copy** 指令支持 4 字节向量加载，而普通内存访问为 16 字节，这影响了**效率**。
  
  - *考虑到在特定 kernel（如 Flash Attention）中需要异步拷贝来降低寄存器压力（register pressure）*，它被证明是有益的。
- **对 MUBUF 文档的困惑**：成员们对 **MUBUF** 文档中关于指令中 **LDS** 使用的部分表示困惑，引发了对其清晰度的讨论。
  
  - 有人指出，关于 M0 使用的一些细节未在 MUBUF 中列出，这促使了对 **CDNA3 documentation** 的进一步探索。
- **探索 MI300X 上的异步操作**：一位专家提到 **gcnasm/async_copy** 仓库是 **MI300X** 上异步操作示例的资源。
  
  - 该仓库由 AMD 高级 kernel 专家开发，凸显了广大社区在这一领域覆盖范围的不足。
- **Flash Attention 对异步的需求**：大家达成共识，由于高寄存器压力问题，**Flash Attention** 可能必须使用异步拷贝。
  
  - *建议 Flash Attention kernel 在寄存器约束紧张的情况下利用异步拷贝来优化性能*。

 

**提到的链接**：[gcnasm/async_copy at master · carlushuang/gcnasm](https://github.com/carlushuang/gcnasm/tree/master/async_copy)：hip/asm 中的 amdgpu 示例代码。通过在 GitHub 上创建账户为 carlushuang/gcnasm 的开发做出贡献。

 

---

### **GPU MODE ▷ #**[**bitnet**](https://discord.com/channels/1189498204333543425/1240586843292958790/1304705917366960179) (19 messages🔥):

> - `BitBlas support for int4 kernels`
> - `Scaling and Fusing in Matrix Multiplication`
> - `Binary Tensor Cores usage`
> - `Performance of int4 on H100`

- **BitBlas 支持 int4 kernel**：一位成员强调 **BitBlas** 上已提供 **int4 kernels**，引起了对其能力的关注。
  
  - 随后讨论了 **BitBlas** 如何处理涉及这些 kernel 的操作，特别是在缩放矩阵乘法（scaled matrix multiplication）中。
- **将缩放融合进 matmul epilogue**：成员们根据 **scaled int8 matmul** 的经验，讨论了为了提高效率将**输出缩放（output scaling）融合**到矩阵乘法 epilogue 中的潜力。
  
  - 有人提到 **BitBlas** 中有一个计划中的实现，通过 Triton 集成来简化该过程。
- **Binary Tensor Cores 使用率低**：据观察，自 **A100** 以来，很少有项目利用 **Binary Tensor Cores**，一位成员指出他们见过的使用该技术的代码库寥寥无几。
  
  - 讨论包括了在**二值化神经网络（binarized neural networks）**中的潜在用途，特别是关于低于 8 位的精度。
- **H100 上 int4 支持的不确定性**：一位成员对 **int4 x int4** 操作在 **H100** 上会崩溃还是正常运行表示好奇。
  
  - 另一位成员澄清说 H100 上**没有 int4 计算核心**，这引发了一些关于支持情况的推测。

 

---

### **GPU MODE ▷ #**[**webgpu**](https://discord.com/channels/1189498204333543425/1262121239044948009/1305585103421964399) (1 条消息):

> - `Surfgrad`
> - `WebGPU Performance`
> - `Typescript Autograd Libraries`
> - `Nomic Visualizations`
> - `Deepscatter`

- **Surfgrad 凭借 WebGPU 掀起波澜**: 一位成员开发了 **Surfgrad**，这是一个使用 [WebGPU](https://x.com/zach_nussbaum/status/1856021159164424559) 的 autograd 引擎，在 M2 芯片上实现了超过 **1 TFLOP** 的性能，展示了 Web 技术的巨大潜力。
  
  - 他们强调了实现这一性能的众多细微优化，并表示创建过程非常愉快。
- **浏览器可视化的挑战**: 对话涉及了在浏览器中显示**数千万个数据点**并保持性能可行性的困难，这也是 [Nomic](https://nomic.ai) 许多人面临的问题。
  
  - 这一挑战促使了 [Deepscatter](https://github.com/nomic-ai/deepscatter) 的开发，旨在高效解决扩展性问题。
- **探索 Typescript Autograd 库**: 该成员注意到目前缺乏基于 **WebGPU** 构建的 autograd 库，因此决定创建 Surfgrad，作为 **WebGPU** 和 **Typescript** 的一项教学练习。
  
  - 这一进展反映了社区对 WebGPU 能力的广泛热情。

**提到的链接**:

- [Zach Nussbaum (@zach_nussbaum) 的推文](https://x.com/zach_nussbaum/status/1856021159164424559): 我对 WebGPU 感到非常兴奋，所以自然而然地构建了 Surfgrad，一个基于 WebGPU 的 autograd 引擎。我演示了如何将一个朴素的 kernel 优化到性能超过 1TFLOP。
- [为 1TFLOP+ 性能优化 WebGPU Matmul Kernel](https://zanussbaum.substack.com/p/optimizing-a-webgpu-matmul-kernel?r=coo9n): 构建 Surfgrad，一个高性能、由 WebGPU 驱动的 autograd 库。

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1304628966929469461) (3 条消息):

> - `CI Temporary Disablement`
> - `Security Concerns`
> - `Local CI Running`
> - `Response Expectations`

- **针对 Fork 暂时禁用 CI**: 出于**安全原因**，针对来自 fork 的 pull request 的 CI 已**暂时禁用**，并将尽快恢复。
  
  - 鼓励团队成员在本地运行 CI，或者在需要时由他人代为运行。
- **对响应的期待**: 一位成员表达了对 CI 状况能得到及时响应的希望。
  
  - 尽管有所延迟，但其使用的轻松表情符号表明了积极的态度。

 

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/1304618621158297600) (4 条消息):

> - `Torchtitan Crash Course`
> - `OpenCoder LLM`
> - `Bot Testing`
> - `GPU Sponsorships`

- **Torchtitan 速成课程公告**: **Torchtitan** 速成课程将于 **11 月 23 日**举行，重点为参与者讲解各种并行策略。
  
  - 随着社区为即将到来的 **GPU 赞助**做准备，本次会议是掌握基础知识的绝佳机会。
- **OpenCoder** LLM 概览: [OpenCoder](https://opencoder-llm.github.io/) 是一款支持**中英双语**的开源代码 LLM，通过在 **2.5 万亿 token** 上的广泛训练，实现了顶尖性能。
  
  - OpenCoder 提供了全面的资源，包括模型权重和详细的训练流水线，旨在助力研究人员进行代码 AI 开发。
- **Bot 测试流程**: 成员们讨论了当前的 bot 测试方法，询问是在服务器的特定频道进行，还是在完全不同的服务器中测试。
  
  - 对话反映了评估 bot 功能及其在社区内实际应用的探索性努力。
- **可能关闭测试频道**: 有提议考虑**关闭**专门用于 bot 测试的频道，因为其功能已被证实有效。
  
  - 这表明随着 bot 能力的演进，社区正在持续评估服务器中专用空间的需求。

 

**提到的链接**: [OpenCoder: 顶尖开源代码大语言模型](https://opencoder-llm.github.io/): 未找到描述

 

---

### **GPU MODE ▷ #**[**edge**](https://discord.com/channels/1189498204333543425/1303441437592911912/1304835338656813119) (5 条消息):

> - `NVIDIA Jetson`
> - `SLM Optimizations on Android` (Android 上的 SLM 优化)

- **关于 NVIDIA Jetson 使用的讨论**：一位成员询问了 **NVIDIA Jetson** 的使用情况，引发了另一位成员分享他们之前使用 **TX2 和 Nano** 模型的经验。
  
  - 然而，他们尚未尝试过 **Orin** 模型，这表明是一个潜在的进一步探索领域。
- **针对 Android 部署优化 LLM**：一位成员提出了关于在 Android 设备上部署 **1.5B SLMs** 是否有特定优化的问题。
  
  - 另一位成员建议如果可用的话应利用 **accelerator**（加速器），暗示其相比于标准的受内存限制的 CPU 推理可能具有的优势。

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1304700309364408392) (76 条消息🔥🔥):

> - `Hailo port on Raspberry Pi 5` (Raspberry Pi 5 上的 Hailo 移植)
> - `Floating Point Exceptions Handling` (浮点异常处理)
> - `Tinybox and Tinygrad`
> - `P2P Hack and Tinybox Upgrades` (P2P Hack 与 Tinybox 升级)
> - `TPU Backend Discussions` (TPU 后端讨论)

- **Hailo 移植进展与量化挑战**：一位用户正在进行 Raspberry Pi 5 上的 Hailo 移植工作，成功将模型从 tinygrad 转换到 ONNX 再到 Hailo，但在需要 CUDA 和 TensorFlow 的量化模型方面面临挑战。
  
  - 他们指出，由于芯片缓存小且内存带宽差，在边缘设备上运行训练代码可能并不实际。
- **处理浮点异常**：讨论了检测 NaN 和 overflow（溢出）等浮点异常的可行性，强调了平台对检测方法支持的重要性。
  
  - 分享的资源强调了在浮点运算期间捕获错误的重要性，并提倡有效的错误处理方法。
- **对 Tinybox 的兴趣及分销商信息**：一位用户询问将预订的 Tinybox 更换为绿色版本的事宜，并对关税和税率表示不确定。
  
  - 另一位用户建议分享有关购买选项的链接，以解决对欧盟分销商的担忧。
- **P2P Hack 与 Tinybox 升级的不确定性**：由于 P2P hack 补丁的原因，用户对 Tinybox 升级到 5090 版本的潜在延迟表示担忧。
  
  - 进一步的讨论推测了具有不同 PCIe 控制器能力的硬件设置对性能的影响。
- **关于 TPU 后端策略的讨论**：一位用户表达了开发 TPU v4 汇编后端的兴趣，并表示愿意在彻底清理后合并工作。
  
  - 有人询问 LLVM 中的汇编是否真正实现了向量化，以及具体针对哪些 TPU 版本提供支持。

**提到的链接**：

- [Floating-point environment - cppreference.com](https://en.cppreference.com/w/cpp/numeric/fenv)：未找到描述
- [FLP03-C. Detect and handle floating-point errors - SEI CERT C Coding Standard - Confluence](https://wiki.sei.cmu.edu/confluence/display/c/FLP03-C.+Detect+and+handle+floating-point+errors)：未找到描述
- [Big graph · Issue #7044 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/issues/7044)：LazyBuffer.view 变为 UOps.VIEW #7077 #7078 #7007 #7090 big graph SINK #7122 #7178 #7170 #7134 #7175 #7132 #7188 #7190 #7217 #7214 #7224 #7234 #7242 #7220 #7322 #7353 #7355 #7367 #7334 #7371 #729...

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1304693965496389662) (3 条消息):

> - `Python Module Import Issues` (Python 模块导入问题)
> - `Beam Search Output Interpretation` (Beam Search 输出解读)
> - `NOLOCALS Environment Variable` (NOLOCALS 环境变量)

- **带有 Extra 模块的 Python 模块导入问题**：一位用户在尝试运行导入 'extra' 模块的 Python 脚本时遇到了 **ModuleNotFoundError**。
  
  - 该问题通过在执行脚本前将 **PYTHONPATH** 环境变量设置为当前目录而解决，命令为：`$ PYTHONPATH="." python3 examples/llama.py`。
- **解读 Beam Search 输出**：一位用户寻求关于解读 **beam search** 输出的帮助，讨论了进度条如何与 kernel 执行时间相关联。
  
  - 他们注意到绿色表示 kernel 的 **final runtime**（最终运行时间），但对 **actions** 和 **kernel size** 表示困惑，并寻求澄清。
- **NOLOCALS 环境变量的功能**：一位用户询问了在项目背景下 **NOLOCALS** 环境变量的作用。
  
  - 他们正在寻求关于该变量如何影响程序或 kernel 在执行期间行为的澄清。

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1304795036008841287) (22 条消息🔥):

> - `AI Interview Bot Development` (AI 面试机器人开发)
> - `Aya-Expanse Model Testing` (Aya-Expanse 模型测试)
> - `Function Calling in AI Models` (AI 模型中的 Function Calling)

- **创建一个 AI 面试机器人**：一位用户正在启动一个 GenAI 项目，旨在开发一个 AI 面试机器人。该机器人将根据简历和职位描述提问，并对回答进行百分制评分。
  
  - 他们正在寻求关于免费资源（如向量数据库和编排框架）的建议，并强调编程工作将由他们自己完成。
- **Aya-Expanse 模型是游戏规则改变者**：一位用户称赞了 **Aya-Expanse** 模型，并透露最初误以为它仅仅是一个翻译模型。
  
  - 他们注意到该模型在 Function Calling 方面具有令人印象深刻的能力，并且在处理希腊语任务时非常有效。
- **高效的 Function Calling**：在讨论对 **Aya-Expanse** 的测试时，一位用户表示其目标是使用较小的模型进行 Function Calling，以降低通过 API 使用大型模型时的成本。
  
  - 他们分享道，该模型能有效地为不需要 Function Calling 的问题选择 `direct_response`，从而提高了响应准确性。

 

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1305511235562766416) (4 条消息):

> - `Cohere API for content creation` (用于内容创作的 Cohere API)
> - `ORG ID retrieval` (ORG ID 检索)
> - `Cohere endpoint disruptions` (Cohere 端点中断)

- **寻求基于文档响应的 Cohere API**：一位用户询问是否有 API 可以根据预先上传的 DOCX 和 PDF 文件生成自由文本响应，并指出目前仅支持 Embeddings。
  
  - 他们表示对此用途下类似于 ChatGPT assistants API 的功能感兴趣。
- **用于用户支持的 ORG ID 查询**：另一位用户询问如何获取 ORG ID，以及它在协助 Cohere 团队解决用户问题时的相关性。
  
  - 他们希望明确可用的组织管理工具。
- **报告 Cohere 端点中断**：一位用户报告在尝试访问 Cohere embedding 端点时收到 `500 Internal Server Error`，表明存在内部问题。
  
  - 目前，他们建议其他人参考 [Cohere 状态页](https://status.cohere.com) 以获取已报告问题的更新。
- **确认持续存在的端点问题**：一名成员确认了 Cohere 端点持续中断的情况，鼓励用户查看状态页以获取实时更新。
  
  - 这体现了用户之间关于服务可靠性的积极沟通。

 

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1305482400939249706) (36 条消息🔥):

> - `Cohere API Errors` (Cohere API 错误)
> - `Fine-Tuned Model Issues` (微调模型问题)
> - `Increased Latency` (延迟增加)
> - `Embed API Slowness` (Embed API 缓慢)
> - `Cohere Status Updates` (Cohere 状态更新)

- **Cohere API 遇到 404 错误**：一位用户报告在尝试获取 Cohere 模型详情时遇到 **404 错误**，并表示之前一切运行正常。
  
  - 尽管进行了故障排除，他们在所有 Cohere 方法调用中仍然面临同样的问题。
- **微调模型更新**：会议提到旧的微调模型已被弃用，基于最新架构的新模型提供了更好的性能。
  
  - 鼓励用户重新上传相同的数据集并训练新的微调分类模型，以获得更好的结果。
- **延迟增加问题**：一位用户对延迟增加表示沮丧，注意到响应时间长达 **3 分钟**，且请求期间 Token 使用量很高。
  
  - 另一位支持人员确认延迟问题是该用户账号特有的，且与高负载（heavy payloads）有关。
- **Embed API 出现缓慢**：一位新用户报告调用 Embed API 时速度缓慢，引发了对潜在持续问题的担忧。
  
  - 支持团队确认他们正在调查这些问题，并建议用户关注 [Cohere 状态页](https://status.cohere.com/) 以获取更新。
- **系统性能反馈**：用户提供了关于 Cohere 模型和服务性能的反馈，对响应时间变慢表示担忧。
  
  - 此外，还引用了响应 ID 和示例，以协助排除持续存在的性能故障。

**提到的链接**：

- [No Donkeys GIF - No Donkeys Shrek - Discover & Share GIFs](https://tenor.com/view/no-donkeys-shrek-gif-1107972158425297474)：点击查看 GIF
- [Fine-tuning](https://cohere.com/fine-tuning)：通过针对特定用例和行业量身定制解决方案，优化生成式 AI 的性能和成本。
- [Cohere Status Page Status](https://status.cohere.com/)：Cohere 状态页的最新服务状态

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1305308688734752789) (3 条消息):

> - `vnc-lm Discord bot`
> - `Cohere API`
> - `ollama models`

- **介绍 vnc-lm Discord bot**：一名成员分享了他们的 Discord bot **vnc-lm**，该工具可以与 **Cohere API** 和 **GitHub Models API** 交互，允许用户在使用本地 **ollama models** 的同时，与这些 API 提供的模型进行互动。
  
  - 显著功能包括创建对话分支、优化 prompt，以及发送屏幕截图和文本文件等上下文材料。
- **使用 Docker 快速设置**：**vnc-lm** 可以通过命令 `docker compose up --build` 快速设置，并可通过 [GitHub](https://github.com/jake83741/vnc-lm) 获取。
  
  - 这种便捷的设置方式允许用户快速部署并利用该 bot 的各项功能。
- **对 vnc-lm 的积极反馈**：成员们对该 bot 表示兴奋，其中一人评论道它非常 **amazing**。
  
  - 这表明了用户对 **vnc-lm** 所提供功能的积极认可和兴趣。

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1304639626480320573) (43 条消息🔥):

> - `Open Interpreter 硬件要求`
> - `Open Interpreter 1.0 更新测试`
> - `Open Interpreter 配置查询`
> - `本地 OLLAMA 问题`
> - `Software Heritage 代码归档`

- **Open Interpreter 硬件需求**：一位用户询问配备 **64GB 或 24GB** RAM 的 **Mac Mini M4 Pro** 是否足以有效运行本地 AI 和 **Open Interpreter**。
  
  - 大家达成共识，确认该配置可行，并引发了关于集成麦克风和扬声器等额外组件的讨论。
- **测试新的 Open Interpreter 1.0 更新**：一位用户表示愿意协助测试即将发布的 **Open Interpreter 1.0** 更新，该版本目前位于 dev 分支，计划于下周发布。
  
  - 社区分享了安装命令，并讨论了 Bug 测试以及针对不同操作系统的适配需求。
- **Open Interpreter 中的配置查询**：另一位用户询问是否有办法像以前的版本一样配置 **OpenAI models**，理由是在没有 GUI 模式的情况下使用 **Sonnet** 成本较高。
  
  - 这引发了关于潜在配置以及本地 OLLAMA 模型所遇问题的简短讨论，反映了不同的使用体验。
- **运行本地 OLLAMA 时遇到的问题**：一位用户报告了尝试运行本地 OLLAMA 时的警告，涉及模型权重和 float 计算类型。
  
  - 这引起了对运行环境建议的关注，特别是关于 **WSL** 和标准 **Windows CMD** 的讨论。
- **归档 Open Interpreter 代码**：一位用户提议协助将 Open Interpreter 代码归档在 **Software Heritage** 中，旨在造福后代。
  
  - 该提议强调了在开发者社区内保留贡献的重要性，并引发了关于具体实施步骤的讨论。

**提到的链接**：[GitHub - davidteren/mac_fan_control-self-healing-coder](https://github.com/davidteren/mac_fan_control-self-healing-coder)：WIP 实验项目！一个动态的、具备自愈能力的框架，使 Open Interpreter 能够从过去的交互中学习，并持续改进其决策过程。

---

### **OpenInterpreter ▷ #**[**O1**](https://discord.com/channels/1146610656779440188/1194880263122075688/1305575906097627209) (1 条消息):

> - `NixOS 设置`
> - `ollama 和 OI 频道`
> - `CUDA 配置`

- **对 NixOS 使用体验的好奇**：一名成员询问了关于让 **NixOS** 顺利运行的经验，并指出 nixpkg 似乎比 GitHub 上的版本陈旧。
  
  - 他们特别感兴趣其他人为了获得更好的功能而实施了什么样的 **setup**。
- **切换到 unstable 频道**：该成员更新了状态，表示他们已将 **ollama** 和 **Open Interpreter** 切换到 unstable 频道以寻求改进。
  
  - *Nix that（这里用了 NixOS 的双关语）* —— 他们现在对目前的配置感到满意。
- **调试 CUDA**：该成员提到他们调试了 **CUDA** 设置，并达到了满意的状态。
  
  - 他们的消息表明，适当的调整使其系统上的实例化取得了成功。

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1305483921672376352) (2 条消息):

> - `Qwen 2.5 coder 模型`
> - `代码生成改进`
> - `Ollama 协作`

- **Qwen 2.5 Coder 模型发布**：最新更新的 **Qwen 2.5 coder 模型**在**代码生成**、**代码推理**和**代码修复**方面表现出显著改进，其中 **32B 模型**可与 OpenAI 的 **GPT-4o** 媲美。
  
  - 用户可以使用类似 `ollama run qwen2.5-coder:32b` 的命令运行不同尺寸的 32B 变体模型。
- **Qwen 发布引发的热烈反响**：随着 Qwen 和 Ollama 展开合作，成员们表达了极大的热情，强调了共同编码的乐趣，正如 Qwen 所言：*“非常高兴能与我们最好的朋友之一 Ollama 共同发布我们的模型！”*
  
  - 此次协作标志着一次显著的伙伴关系，进一步增强了开发者可使用的能力。
- **模型尺寸版本说明**：Qwen 2.5 模型提供多种尺寸，包括 32B、14B、7B、3B、1.5B 和 0.5B，为不同的编码任务提供了灵活性。
  
  - 每个模型尺寸都有其对应的部署命令，确保用户可以选择最适合其需求的版本。
- **Qwen 2.5 Coder 模型链接**：有关 **Qwen 2.5 coder 模型**的更多详情和资源，请访问 [Ollama 官网](https://ollama.com/library/qwen2.5-coder)。
  
  - 该资源包含了有效利用各种模型尺寸的全面指南。

 

**提到的链接**：[来自 ollama (@ollama) 的推文](https://x.com/ollama/status/1856051733513797929?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：Qwen 2.5 coder 模型已更新，在**代码生成**、**代码推理**和**代码修复**方面有显著改进。32B 模型具有与 OpenAI GPT-4o 竞争的性能。...

 

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1304856007520026807) (5 条消息):

> - `LlamaParse Premium`
> - `高级分块策略`
> - `带有用户反馈的 PowerPoint 生成`
> - `PureML 高效数据处理`
> - `PursuitGov 案例研究`

- **LlamaParse Premium 在文档解析方面表现卓越**：Hanane Dupouy 展示了 [LlamaParse Premium](https://t.co/pgqVUhwjXh) 如何有效地将复杂的图表和图示解析为结构化的 Markdown。
  
  - 该工具不仅增强了可读性，还将视觉数据转化为可访问的文本格式，提升了文档的可用性。
- **高级分块 (Chunking) 策略提升性能**：@pavan_mantha1 详细介绍了**三种高级分块策略**，以及一套用于在个人数据集上进行实际测试的完整评估方案，如[此贴](https://t.co/8UTY4xNHOT)所示。
  
  - 这些策略旨在显著增强检索和问答功能，展示了处理数据的有效方法。
- **通过实时反馈创建 PowerPoint**：Lingzhen Chen 解释了一个完整的流程，用于构建一个从研究到 PowerPoint 生成的系统，该系统通过 [Streamlit 界面](https://t.co/l102iy4R8u)整合了用户反馈。
  
  - 这种创新方法允许用户对幻灯片大纲提供输入，从而提高了自动化演示文稿的质量。
- **PureML 自动化数据集管理**：PureML 利用 LLM 自动清理和重构机器学习数据集，实现了上下文感知处理和智能特征创建，如[此处](https://t.co/E6frzia1yR)所述。
  
  - 这些功能旨在提高数据的一致性和质量，展示了包括 LlamaIndex 和 GPT-4 在内的各种先进工具的集成。
- **PursuitGov 令人印象深刻的转型**：关于 [PursuitGov](https://t.co/3IklxO3vRZ) 的案例研究显示，他们在一个周末内解析了 **400 万页**文档，将文档准确度显著提升了 **25-30%**。
  
  - 这种转型使客户能够发现公共部门数据中隐藏的机会，说明了先进解析技术的强大力量。

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1304562513047982161) (10 messages🔥):

> - `Sentence Transformers Ingestion Pipeline` (Sentence Transformers 数据摄取流水线)
> - `Docker Resource Settings` (Docker 资源设置)
> - `Llama 3.2 Waitlist Access` (Llama 3.2 等候名单访问权限)
> - `Text-to-SQL Applications` (Text-to-SQL 应用)
> - `Dynamic Related Content Chunks` (动态相关内容分块)

- **Sentence Transformers Ingestion Pipeline Takes Too Long**: 一位成员报告称，在 Docker 容器中运行带有 [sentence transformers](https://docs.llamaindex.ai/) 的数据摄取流水线耗时过长，并最终失败。
  
  - 他们分享了代码设置，重点介绍了特定的配置，例如使用 `all-MiniLM-L6-v2` 并设置 `TOKENIZERS_PARALLELISM=True`。
- **Docker Resource Settings for Better Performance**: 一位用户询问了 Docker 资源设置，另一位成员提到他们分配了 **4 个 CPU 核心**和 **8GB 内存**。
  
  - 尽管进行了这些设置，摄取过程仍然缓慢且容易失败。
- **Questions about Llama 3.2 Key Access**: 一位用户询问如何获取 **Llama 3.2** 的 Key，并提到他们目前正在等候名单中。
  
  - 尚未提供获取 Key 的详细信息，表明需要进一步的指导。
- **Resource Sharing for Text-to-SQL Applications**: 一位成员寻求使用向量数据库创建 Text-to-SQL 应用的资源，并分享了一篇包含过时链接的文章。
  
  - 另一位用户建议查看 [LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/) 以获取更新的方法和工作流。
- **Dynamic Related Content Chunks Discussion**: 出现了一个关于如何创建动态相关内容分块的新咨询，类似于 RAG 驱动的网站上使用的功能。
  
  - 目前尚未讨论相关的解决方案或资源，表明了进一步协作的兴趣。

**Links mentioned**:

- [Combining Text-to-SQL with Semantic Search for Retrieval Augmented Generation — LlamaIndex - Build Knowledge Assistants over your Enterprise Data](https://www.llamaindex.ai/blog/combining-text-to-sql-with-semantic-search-for-retrieval-augmented-generation-c60af30ec3b): LlamaIndex 是一个简单、灵活的框架，用于连接企业数据并使用 LLM 构建知识助手。
- [Text-to-SQL Guide (Query Engine + Retriever) - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/): 未找到描述
- [Workflows for Advanced Text-to-SQL - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/workflow/advanced_text_to_sql/): 未找到描述
- [Workflows - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/workflow/#workflows): 未找到描述

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1304819171493941310) (6 messages):

> - `Benchmarking fine-tuned LLM model` (微调 LLM 模型的基准测试)
> - `OpenAI Agent stream chat code snippet` (OpenAI Agent 流式聊天代码片段)

- **Seeking Help for LLM Benchmarking**: 一位成员请求关于对其在 [Hugging Face](https://huggingface.co/Anoshor/prism-v2) 上发布的微调 LLM 模型进行基准测试的指导。他们提到在使用 Open LLM leaderboard 进行测试时遇到了错误。
- **Code Snippet for OpenAI Agent Stream Chat**: 一位成员索要关于 OpenAI Agent 流式聊天工作原理的源代码片段。另一位成员迅速提供了相关的代码片段，并解释了其在 LlamaIndex 中生成流式响应的用法。
  
  - 该代码示例源自文档中的 [OpenAI Agent example](https://github.com/jerryjliu/llama_index/blob/main/docs/docs/examples/agent/openai_agent.ipynb)，演示了如何逐个 token 地打印响应。

 

**Link mentioned**: [Anoshor/prism-v2 · Hugging Face](https://huggingface.co/Anoshor/prism-v2): 未找到描述

 

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1304857557445115964) (3 messages):

> - `M3DocRAG`
> - `DSPy vision capabilities`
> - `Multi-modal question answering`
> - `Open-domain benchmarks`

- **M3DocRAG 在多模态 RAG 中设定新标准**：M3DocRAG 展示了利用来自大量 PDF 语料库的**多模态信息**进行**问答**的卓越成果，并在 **ColPali 基准测试**中表现出色。
  
  - *Jaemin Cho* 强调了其在处理跨不同文档上下文的**单跳和多跳问题**方面的多功能性。
- **带有 M3DocVQA 的新开放领域基准**：**M3DocVQA**（一个 **DocVQA 基准**）的引入，挑战模型在超过 **3K 份 PDF** 和 **40K 页**内容中回答**多跳问题**。
  
  - 该基准旨在通过利用**文本、表格和图像**等各种元素来增强理解。
- **DSPy RAG 使用案例引发关注**：一位成员对 **DSPy RAG 能力**的潜力表示热忱，表现出浓厚的实验兴趣。
  
  - 他们注意到 **DSPy RAG** 与**视觉能力**之间充满前景的交集，暗示了未来有趣的应用。

 

**提到的链接**：[来自 Omar Khattab (@lateinteraction) 的推文](https://x.com/lateinteraction/status/1854983445304558016?t=xlW0adUWBh04yygjOTOsTQ&s=19)：使用来自大量 PDF 语料库的多模态信息进行问答的酷炫基准，具有出色的 ColPali 结果。引用 Jaemin Cho (@jmin__cho)：看看 M3DocRAG —— 多模态 RA...

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1304862127173144708) (14 messages🔥):

> - `LangChain integration support`
> - `DSPy prompting techniques composability`

- **LangChain 集成停止支持**：**GitHub** 上的最新更新表明，目前与 **LangChain** 的集成已不再维护，可能无法正常运行。
  
  - 一位成员对这一变化提出了疑问，寻求有关情况的更多背景信息。
- **DSPy 提示技术设计为不可组合**：成员们讨论了 **DSPy** 提示技术的本质，确认它们在设计上是有意**不可组合**的。
  
  - 这一决定强调，虽然可以操作 signatures，但这样做可能会限制功能和控制流的清晰度。

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1304766392285003776) (17 messages🔥):

> - `FastChat removal`
> - `Metharme support`
> - `Fine-tuning VLMs`
> - `Inflection AI API`
> - `Metharme chat_template PR`

- **FastChat 和 ShareGPT 的移除引发惊讶**：成员们对移除 **FastChat** 和 **ShareGPT** 反应强烈，其中一人惊呼：*我的天，你是认真的吗？* 引用 [PR #2021](https://github.com/axolotl-ai-cloud/axolotl/pull/2021) 突显了社区的担忧。
  
  - 替代建议包括使用旧的 commit，展示了关于维持项目稳定性的持续讨论。
- **Metharme 还在支持吗？**：一位用户询问 **Metharme** 是否不再受支持，另一位成员回应解释称，与 **fschat** 发布相关的延迟影响了他们的进度。
  
  - 成员们表示有兴趣将 **sharegpt** 对话集成到新的 **chat_template** 中。
- **微调 VLMs 的建议**：关于微调 **VLMs** 的咨询得到了建议，即从示例仓库中提供的 **llama vision** 配置开始。
  
  - 同时也确认了使用 **llama 3.2 1B** 训练 **VLM 模型** 是可行的，展示了用户对高级模型训练的兴趣。
- **Inflection AI API 功能发布**：讨论涉及了 **Inflection-3** 的能力，概述了两个模型：用于情感互动的 **Pi** 和用于结构化输出的 **Productivity**，但指出缺乏 benchmarks。
  
  - 成员们对缺乏 benchmark 数据感到惊讶，对新模型的实际评估表示担忧。
- **通过 PR 添加了 Metharme chat_template**：分享了一个根据用户请求添加 **Metharme** 作为 **chat_template** 的 PR，并强调了其针对旧版本的测试。
  
  - 鼓励成员在本地运行 preprocess 命令以确保功能顺畅，促进社区协作。

**提到的链接**：

- [Inflection AI Developer Playground](https://developers.inflection.ai/docs)：让我们构建更好的企业级 AI。
- [feat: add metharme chat_template by NanoCode012 · Pull Request #2033 · axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl/pull/2033)：描述：根据用户请求添加 metharme。动机与背景：如何进行的测试？在之前的 fschat 之间进行了测试。屏幕截图（如果适用）更改类型 社交账号（可选）
- [remove fastchat and sharegpt by winglian · Pull Request #2021 · axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl/pull/2021)：未找到描述

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**hackathon-announcements**](https://discord.com/channels/1280234300012494859/1280236929379602493/1305622171648917565) (2 messages):

> - `Midterm Check-in`
> - `Compute Resource Application`
> - `Lambda Workshop`

- **项目反馈的期中检查**：团队现在可以通过 [期中检查表单](https://docs.google.com/forms/d/e/1FAIpQLSfxhgqcKWxfs_e1xuF3yukTvIwk_0JhsaVwHizS7o9BYW9Hnw/viewform?usp=sf_link) 提交进度，以获取反馈并可能获得 GPU/CPU 资源额度。
  
  - 即使不申请资源，提交此表单也至关重要，因为这有助于获得关于项目的宝贵见解。
- **额外计算资源申请**：对额外 GPU/CPU 资源感兴趣的团队必须在填写期中检查表单的同时完成 [资源请求表单](https://docs.google.com/forms/d/e/1FAIpQLSeJQ_i6H5bgA5S767QZaorwkzF9_k_63I8JCed3dnlVcvKJ1w/viewform)。
  
  - 分配将取决于记录的进度和详细的理由，鼓励即使是新团队也进行申请。
- **项目评审标准**：项目将根据问题陈述的清晰度、方法的可行性以及迄今为止取得的进展进行评审。
  
  - 团队还必须证明其资源需求的合理性，并展示利用所获资源的充分能力。
- **Lambda Workshop 提醒**：**Lambda Workshop** 定于明天（11 月 12 日）**PST 时间下午 4-5 点**举行，鼓励参与者通过 [此链接](https://lu.ma/agents-hackathon-lambda) 报名 (RSVP)。
  
  - 本次研讨会将为团队项目和黑客松流程提供进一步的见解和指导。

**提到的链接**：

- [LLM Agents MOOC Hackathon, Mid Season Check In Form](https://docs.google.com/forms/d/e/1FAIpQLSfxhgqcKWxfs_e1xuF3yukTvIwk_0JhsaVwHizS7o9BYW9Hnw/viewform?usp=sf_link)：如果您希望获得项目反馈，请填写此表单。请注意，由于提交量较大，我们可能无法为所有团队提供反馈。重要提示：如果您在...
- [未找到标题](https://docs.google.com/forms/d/e/1FAIpQLSeJQ_i6H5bgA5S767QZaorwkzF9_k_63I8JCed3dnlVcvKJ1w/viewform)：未找到描述

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1304835320356933682) (4 messages):

> - `文章作业反馈`
> - `提交确认`

- **了解你的文章作业状态**：一位成员询问需要多久才能知道提交的文章作业是否通过，另一位成员提到反馈可能要到最终提交截止日期之后才会提供。
  
  - 他们安慰询问者，只要遵守网站指南，评分通常会比较宽松，所以*不要压力太大*。
- **提交检查确认**：另一位成员询问是否可以私信某人以确认他们的文章作业已收到。
  
  - 对方澄清说，他们应该已经收到了来自 Google Forms 的确认，这表明提交确实已收到。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-lecture-discussion**](https://discord.com/channels/1280234300012494859/1282734248112947210/1304823915331321867) (6 messages):

> - `Hackathon 团队规模`
> - `加入 Hackathon`
> - `课程公告`

- **Hackathon 团队人数不限**：一位成员询问 Hackathon 允许的团队规模，另一位成员回答说**不限人数**。
  
  - 这为任何有兴趣的人提供了无限制协作的可能性。
- **加入 Hackathon 永远不晚**：另一位成员询问现在加入是否太晚，并得到保证参加永远不晚。
  
  - 他们被鼓励加入 Discord 并参加计划于 **PT 时间晚上 7 点**举行的 Lecture 2 讨论。
- **即将举行的 LLM Agents 课程**：发布了一则关于今晚讨论 **Lecture 2: History of LLM Agents** 的公告。
  
  - 此次讨论将包括课程回顾和一些 Agentic 代码的探索，欢迎任何感兴趣的人参加。

 

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1305092249901596723) (10 messages🔥):

> - `Self Attention 中的 Attention Scores`
> - `DCP Checkpointing 问题`
> - `模型保存策略`

- **获取 Attention Scores 的挑战**：一位用户询问是否有办法在不使用 forward hooks 修改 self-attention 模块中的 forward 函数的情况下获取 **attention scores**。
  
  - 其他人建议 **F.sdpa()** 目前可能存在不输出 attention scores 的问题，因此可能需要进行修改。
- **DCP Checkpointing 尚未解决**：一位成员报告称，最新的 git main 版本仍未解决在 rank=0 GPU 上汇总权重/优化器的问题，导致 **OOM** (Out Of Memory) 错误。
  
  - 他们实现了 **DCP checkpoint 保存**的权宜之计，打算将其转换为 Hugging Face 格式，并可能编写一个 PR 以实现更好的集成。
- **潜在的 DCP 集成帮助**：随后讨论了分享与 DCP checkpointing 相关的 PR 或 forks，强调了社区对 **torchtune** 集成的支持。
  
  - 更新表明，来自 PyTorch 贡献者的 **DCP PR** 可能很快就会发布，从而增强协作进展。

 

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1304743853529632829) (7 messages):

> - `SVDQuant`
> - `Gorilla Marketing`

- **SVDQuant 实现显著降低**：最近关于 SVDQuant 的帖子展示了一种针对 diffusion models 的新量化范式，通过将权重和激活值量化为 4 bits，在 16GB 显存的笔记本 4090 GPU 上实现了 **3.5 倍显存减少**和 **8.7 倍延迟降低**。
  
  - 交互式 demo 可以点击[这里](https://svdquant.mit.edu)访问，更多资源可在 [GitHub](https://github.com/mit-han-lab/deepcompressor) 获取，完整论文请见[这里](http://arxiv.org/abs/2411.05007)。
- **关于 Gorilla Marketing 的讨论**：成员们讨论了 AI 公司参与他们所谓的 **Gorilla Marketing**（大猩猩营销）的趋势，这预示着非传统的促销策略。
  
  - 这一点被幽默地提及，并引用了一个 [Harambe GIF](https://tenor.com/view/harambe-america-murica-flag-waving-gif-17339298)，强调了营销策略的趣味性。

**提到的链接**：

- [SVDQuant: 精确的 4-Bit 量化助力 16GB 4090 笔记本运行 12B FLUX，速度提升 3 倍](https://hanlab.mit.edu/blog/svdquant): 未找到描述
- [Harambe America GIF - Harambe America Murica - 发现并分享 GIF](https://tenor.com/view/harambe-america-murica-flag-waving-gif-17339298): 点击查看 GIF

---

### **MLOps @Chipro ▷ #**[**events**](https://discord.com/channels/814557108065534033/869270934773727272/1305532399064453231) (1 条消息):

> - `RisingWave Data Processing`
> - `Stream Processing Innovations`

- **RisingWave 增强数据处理技术**：最近的一篇文章强调了 **RisingWave** 在数据处理方面的进展，重点介绍了 **stream processing** 技术方面的改进。
  
  - 欲了解更多见解，请查看其 [LinkedIn 帖子](https://www.linkedin.com/posts/risingwave_risingwave-dataprocessing-streamprocessing-activity-7260009892848033792-adOv)中的完整详情。
- **聚焦流处理技术**：讨论集中在 **stream processing** 的最新动态，展示了优化实时数据处理的方法。
  
  - 参与者指出，采用这些创新可能会显著影响数据驱动的决策。

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**discussion**](https://discord.com/channels/1111172801899012102/1111353033352294440/1304816288719179828) (1 条消息):

> - `Gorilla LLM`
> - `Benchmark testing custom LLMs`

- **关于使用 Gorilla 进行基准测试的咨询**：一位用户询问是否可以使用 **Gorilla** 来测试/基准测试（benchmark）他们微调后的 LLM 模型，由于他们是该领域的新手，因此寻求指导。
  
  - 他们表示特别需要在 **benchmark testing custom LLMs** 方面获得帮助。
- **寻求自定义 LLM 基准测试的指导**：同一位用户再次表达了他们在寻求帮助，以了解如何有效地对他们的 **fine-tuned LLM** 进行基准测试。
  
  - 他们强调了自己是该领域的新手，希望能得到社区的支持和建议。

 

---

### **AI21 Labs (Jamba) ▷ #**[**general-chat**](https://discord.com/channels/874538902696914944/874538902696914947/) (1 条消息):

ag8701347: 请允许我们继续使用我们微调后的模型。

---

---

---

{% else %}

> 完整的频道细分内容已在邮件中截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}