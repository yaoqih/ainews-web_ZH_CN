---
companies:
- meta-ai-fair
- nvidia
- princeton
- colfax
- patronus-ai
- databricks
- mosaic-ai
- openai
date: '2024-07-13T02:52:26.666831Z'
description: '**Reddit 的 URL 结构导致 AI 生成的摘要中出现链接错误，尤其是 NSFW（不宜在办公场所查看）内容会影响 Claude
  和 GPT-4 等模型。** 团队在修复这一故障的同时，仍继续利用大语言模型（LLM）来总结 Reddit 内容。**得益于 H100 GPU 以及 CUDA
  和 FlashAttention 等软件改进，GPT-2 的训练成本已大幅降至约 672 美元。** **FlashAttention-3 正式发布，在 H100
  GPU 上实现了高达 740 TFLOPS 的性能，FP8 精度下接近 1.2 PFLOPS；该项目由 Meta、NVIDIA、普林斯顿大学和 Colfax 合作开发。**
  Hopper 架构 GPU 凭借新的硬件特性实现了重大提速。**近期研究表明，合成数据可能无法改善视觉任务。** **Avocado360 基准测试用于评估视觉语言模型在图像中检测牛油果的能力。**
  **Lynx 是一款针对大语言模型的幻觉检测模型，专为医疗保健和金融科技的实际应用而推出，由 Patronus AI 在 Databricks Mosaic AI
  上使用 Composer 训练而成。**'
id: 720f4a48-42f5-4c93-a247-415197325b3e
models:
- gpt-2
- flashattention-3
- lynx
original_slug: ainews-we-solved-hallucinations
people:
- karpathy
- tri_dao
- giffmana
- vikhyatk
- dbrxmosaicai
title: 我们解决了幻觉问题。
topics:
- compute-hardware
- gpu-optimization
- flashattention
- llm-evaluation
- hallucination-detection
- vision
- benchmarking
- synthetic-data
- model-training
---

<!-- buttondown-editor-mode: plaintext -->**用一个奇招！**

> 2024年7月11日至7月12日的 AI News。
我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**463** 个频道，**2566** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**276 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

听着，我们早就知道我们的 Reddit 摘要中充斥着……呃……那些指向错误地点的链接。你们一直在提醒我们！（谢谢！）

 
![image.png](https://assets.buttondown.email/images/90d8843b-3834-46ed-85c5-56bf51f2c441.png?w=960&fit=max)
 

这种情况在我们的 Reddit 摘要中比 Discord 或 Twitter 回顾中频繁得多的原因是 Reddit 的 URL 结构。

这是一个典型的 Reddit URL：

[https://www.reddit.com/r/LocalLLaMA/comments/1cxnrov/disappointing_if_true_meta_plans_to_not_open_the/](https://www.reddit.com/r/LocalLLaMA/comments/1cxnrov/disappointing_if_true_meta_plans_to_not_open_the/)

末尾的 slug（`disappointing_if_true_meta_plans_to_not_open_the`）只是试图根据标题生成一个人类可读的 slug，而开头的 subreddit（`r/LocalLLaMA`）同样是为了人类的可读性。在实践中，所有这些都会被忽略，取而代之的是“真正”的 slug，即那个 7 位的字母数字组合（`1cxnrov`）。在这里，我们将证明这一点：

[https://www.reddit.com/r/SmolAI/comments/1cxnrov/ainews_is_the_best/](https://www.reddit.com/r/SmolAI/comments/1cxnrov/ainews_is_the_best/)

尽管更改了 subreddit 和人类可读的 slug，Reddit 仍会根据“真正”的 slug 将您引导至之前的同一篇帖子。

因此，Reddit URL 比大多数 URL 对 Attention 中的微小错误都更加*极度*敏感，即使我们只是要求 LLM 从引用链接拼写整齐的源文档中进行复制。

而且……Claude 和 GPT4 都在大量的 NSFW Reddit URL（[涵盖多种语言](https://x.com/tianle_cai/status/1790109646205890723)！）上进行过训练。把这两个事实结合起来，你就能明白我们一直在处理什么样的问题了。

所以……[我们着手修复了这个故障](https://www.youtube.com/watch?v=BUE0PPQI3is)，*同时*仍然使用 LLM 对整个 Reddit 投稿和评论语料库进行格式化、筛选和摘要。如果你对我们的实现方式有猜想，请在 Twitter 上 @Smol_AI。

今天又是内容较少的一天，所以请欣赏我们与 Clementine Fourrier 关于 LLM Evals 的对话（我们 [5 月份的报道](https://buttondown.email/ainews/archive/ainews-to-be-named-4285/)）以及 Open LLM Leaderboard 的未来：

https://www.youtube.com/watch?v=E-UhbYc8m24


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，从 4 次运行中择优。

**算力与硬件改进**

- **GPT-2 训练成本大幅下降**：[@karpathy](https://twitter.com/karpathy/status/1811467135279104217) 指出，由于 **计算硬件 (H100 GPUs)、软件 (CUDA, cuBLAS, cuDNN, FlashAttention) 以及数据质量 (如 FineWeb-Edu 数据集)** 的提升，现在在一台 8XH100 GPU 节点上训练 GPT-2 24 小时仅需约 672 美元，而 2019 年约为 100,000 美元。
- **FlashAttention-3 发布**：[@tri_dao](https://twitter.com/tri_dao/status/1811453622070444071) 宣布了 FlashAttention-3，其在 **FP16 上速度提升了 1.5-2 倍，在 H100 上达到 740 TFLOPS (75% 利用率)，FP8 接近 1.2 PFLOPS**。这是与 Meta、NVIDIA、Princeton 和 Colfax 合作的成果。
- **Hopper GPUs 实现重大加速**：[@tri_dao](https://twitter.com/tri_dao/status/1811453625165840608) 指出 Hopper GPUs (H100) 具有 WGMMA、TMA 和 FP8 支持等新硬件特性，可实现重大加速。仅为这些特性重写 FlashAttention 即可达到 570 TFLOPS。

**LLM 评估与基准测试**

- **合成数据可能对视觉任务没有帮助**：[@giffmana](https://twitter.com/giffmana/status/1811527727796642274) 强调了一篇论文，显示在运行正确的基线时，合成图像实际上对视觉任务没有帮助。
- **用于评估 VLM 的 Avocado360 基准测试**：[@vikhyatk](https://twitter.com/vikhyatk/status/1811540521028067661) 介绍了 Avocado360 基准测试，用于评估视觉语言模型 (VLMs) 是否能判断图像中是否包含牛油果。**评估了四个随机选择的 VLMs**。
- **用于 LLM 幻觉检测的 Lynx 模型**：[@DbrxMosaicAI](https://twitter.com/DbrxMosaicAI/status/1811537853350064592) 宣布了 Lynx，这是一种新的 LLM 幻觉检测模型，特别适用于 **医疗和金融科技等行业的实际应用**。它由 Patronus AI 在 Databricks Mosaic AI 上使用 Composer 训练。

**LLM 应用与框架**

- **Runway AI 自动化**：[@labenz](https://twitter.com/labenz/status/1811463195480977856) 分享了视频生成初创公司 Runway 如何使用 AI 自动化任务，如预写销售邮件。他们的目标是 **通过 AI 能力进行扩展，使员工人数永远不超过 100 人**。
- **用于人机回环反馈的 LangGraph**：[@LangChainAI](https://twitter.com/LangChainAI/status/1811438797680492600) 展示了如何在 LangGraph 中添加人类输入检查点并更新图状态，以 **实现 Agent 系统的用户反馈**。
- **用于高级 RAG 的 Qdrant 和 LlamaIndex**：[@qdrant_engine](https://twitter.com/qdrant_engine/status/1811451520698716262) 分享了一篇关于构建高级 RAG 架构的文章，该架构结合了 LlamaIndex Agent 与 Qdrant 的 **混合搜索功能，同时使用稠密 (dense) 和稀疏 (sparse) 向量嵌入**。

**梗与幽默**

- **对 ThinkPad 的热爱**：[@giffmana](https://twitter.com/giffmana/status/1811485814334918667) 开玩笑说：“最好的笔记本电脑是什么，为什么是 ThinkPad？”
- **Token 限制的烦恼**：[@HamelHusain](https://twitter.com/HamelHusain/status/1811508610469654922) 在 Anthropic UI 上很快就达到了 Token 限制，即使是 Pro 计划也是如此，他想知道这是否正常。
- **ML/DS 面试要求**：[@jxmnop](https://twitter.com/jxmnop/status/1811503193798639970) 开玩笑说，到明年，ML/DS 面试将要求一道来自 ML LeetCode 的中等难度题、硬核 Prompt Engineering 以及五年的 CUDA 经验。

---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取现在可以运行了，但还有很多改进空间！

**主题 1. WizardLM 3 与 LLM 优化技术**

- [/r/LocalLLaMA] **[WizardLM 3 即将推出 👀🔥](https://i.redd.it/a7lkiff2hxbd1.jpeg)** ([Score: 418, Comments: 73](https://reddit.com//r/LocalLLaMA/comments/1e0v437/wizardlm_3_is_coming_soon/)): **WizardLM 3**，一个即将推出的语言模型，即将发布。该公告暗示了显著的改进或新功能，尽管帖子中未提供有关模型能力或发布日期的具体细节。

- [/r/LocalLLaMA] **[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://www.together.ai/blog/flashattention-3)** ([Score: 146, Comments: 22](https://reddit.com//r/LocalLLaMA/comments/1e0vh1j/flashattention3_fast_and_accurate_attention_with/)): **FlashAttention-3** 引入了一种在 **Large Language Models (LLMs)** 中计算 Attention 的新方法，在保持准确性的同时，比以往方法提速 **2-4 倍**。该技术采用了**异步 IO (asynchronous IO)** 和**低精度计算 (low-precision computation)**，能够高效处理更长的序列，并可能支持训练具有更长上下文长度 (context lengths) 的更大型模型。这项由斯坦福大学和 NVIDIA 研究人员在 [论文](https://arxiv.org/abs/2311.05908) 中详述的进展，可能会显著影响更强大的 LLM 的开发和部署。


**Theme 2. Advanced AI-Generated Visual Content**

- [/r/StableDiffusion] **[fal drops AuraFlow](https://i.redd.it/ajp4lo34jzbd1.png)** ([Score: 322, Comments: 95](https://reddit.com//r/StableDiffusion/comments/1e14ip2/fal_drops_auraflow/)): **fal** 推出了 **AuraFlow**，这是一款结合了 **Stable Diffusion** 和 **Midjourney** 优势的新型图像生成模型。AuraFlow 旨在提供高质量的图像生成，并改进**连贯性 (coherence)** 和**构图 (composition)**，解决如面部和手部畸形等常见问题。该模型目前可通过 fal 的 API 获取，并将集成到其**无代码 AI 应用构建器 (no-code AI app builder)** 中。

- [/r/StableDiffusion] **[AnimateDiff and LivePortrait (First real test)](https://v.redd.it/bpmfc8in3zbd1)** ([Score: 580, Comments: 66](https://reddit.com//r/StableDiffusion/comments/1e12sav/animatediff_and_liveportrait_first_real_test/)): **AnimateDiff 和 LivePortrait 的集成**展示了从静态图像创建动画肖像的潜力。该过程涉及使用 **AnimateDiff** 从单张图像生成 **16 帧动画**，然后将其输入 **LivePortrait** 以产生更真实的动画效果。这种工具组合展示了一种通过流体、自然的动作让静态图像焕发生机的极具前景的方法。

- [/r/singularity] **[Al-Generated Movie Trailer](https://v.redd.it/ixnbp7dye7bd1)** ([Score: 157, Comments: 41](https://reddit.com//r/singularity/comments/1e0rtp7/algenerated_movie_trailer/)): **AI 生成的电影预告片**展示了电影制作中先进的视觉能力。该预告片使用**人工智能**创建，展示了通常与高预算制作相关的逼真 **CGI 角色**、**动态场景过渡**和**复杂的视觉效果**，突显了 AI 通过降低成本和扩大创意可能性来彻底改变电影行业的潜力。


**Theme 3. AI Progress Tracking and Benchmarking**

- [/r/OpenAI] **[OpenAI Develops System to Track Progress Toward Human-Level AI](https://i.imgur.com/TjnZv1w.png)** ([Score: 232, Comments: 75](https://reddit.com//r/OpenAI/comments/1e0yqq8/openai_develops_system_to_track_progress_toward/)): OpenAI 推出了一套名为 **AI Preparedness Framework** 的新系统，用于监测和评估向**人类水平人工智能 (human-level AI)** 迈进的进度。该框架旨在通过 **5 级量表**（从窄域 AI 到 **AGI**）评估 AI 系统在 **12 项关键能力**上的表现，包括语言理解、推理和任务完成。这一举措是 OpenAI 负责任地开发先进 AI 系统并为政策制定者提供有关 AI 进展的见解所做努力的一部分。

- [/r/singularity] **[Rorschach test for AI: is this good or bad?](https://i.redd.it/j0da8gfsi0cd1.png)** ([Score: 110, Comments: 152](https://reddit.com//r/singularity/comments/1e18e2j/rorschach_test_for_ai_is_this_good_or_bad/)): **针对 AI 的罗夏墨迹测试 (Rorschach tests)** 被提议作为一种评估 AI 能力的方法，特别是在图像解释和推理方面。该概念建议使用类似于传统罗夏墨迹测试的模糊图像，来评估 AI 感知、解释和解释视觉信息的能力。这种方法可能会揭示 AI 认知过程和局限性的见解，但也引发了关于此类评估对人工智能系统有效性和可靠性的质疑。


**Theme 4. AI Content Regulation and Copyright Issues**

- [/r/StableDiffusion] **专注于 AI 的 COPIED 法案将使删除数字水印成为违法行为** ([Score: 136, Comments: 155](https://reddit.com//r/StableDiffusion/comments/1e17eur/the_aifocused_copied_act_would_make_removing/)): **“参议员提出 COPIED 法案以打击 AI 内容滥用”**。由一组参议员提出的 **COPIED 法案** 旨在通过建立 **内容认证和 AI 生成材料检测** 标准，打击 AI 模型对内容的未经授权使用。该法案将使 **删除数字水印成为违法行为**，允许内容所有者 **起诉未经许可使用其作品的公司**，并要求 NIST 制定 **内容来源证明** 和 **合成内容检测** 的标准，同时禁止使用 **受保护内容来训练 AI 模型**。该法案得到了 **SAG-AFTRA 和 RIAA** 等行业团体的支持，是监管 AI 技术更广泛推动的一部分，并授权各州总检察长 (AGs) 和 FTC 执行其条款。

- [/r/LocalLLaMA] **AI 的危险并非大多数人所想的那样。** ([Score: 100, Comments: 115](https://reddit.com//r/LocalLLaMA/comments/1e0uqq1/the_danger_of_ai_is_not_what_most_people_think_it/)): **AI 的真正危险源于过度评估，而非超级智能**。该帖子认为，AI 的真正危险不在于其成为超级智能的潜力，而在于其当前的 **局限性被忽视**。作者指出，AI 正被部署在一些其 **缺乏智能** 可能会引发问题的领域，并引用了 [AI 生成的虚假法律案例](https://apnews.com/article/artificial-intelligence-chatgpt-fake-case-lawyers-d6ae9fa79d0542db9e1455397aef381c) 和 [自动驾驶汽车中存在偏见的行人检测](https://thenextweb.com/news/driverless-cars-pedestrian-detection-age-race-biases) 作为例子。他们还认为，围绕 AI 安全的大部分讨论是由 **“护城河建设” (moat-building)** 和保护 **先发优势 (first-mover advantages)** 驱动的，而非真正的担忧。

---

# AI Discord 摘要

> 摘要的摘要之摘要

**1. LLM 进展与训练技术**

- **FlashAttention 加速 Transformer 训练**：[FlashAttention-3](https://tridao.me/blog/2024/flash3/) 的发布承诺在 FP16 上实现高达 1.5-2 倍的速度提升，在 H100 GPUs 上达到 740 TFLOPS，实现 75% 的利用率，并在使用 FP8 时可能达到 1.2 PFLOPS。
   - 这项技术由 **Colfax**、**Tri Dao**、**Meta AIT 团队** 和 **Cutlass 团队** 共同开发，通过最小化注意力机制中的内存读写，已经加速了 **GPT-4** 和 **Llama 3** 等模型的训练。
- **Q-Galore 增强内存高效的 LLM 训练**：新型 [Q-Galore 方法](https://arxiv.org/abs/2407.08296) 结合了量化和低秩投影，与 GaLore 相比，显著减少了大型语言模型的内存占用和训练时间。
   - 与依赖耗时的 SVD 操作的 GaLore 不同，**Q-Galore** 观察到某些梯度子空间会提前收敛，而其他子空间则频繁变化，从而在不牺牲准确性的情况下实现更高效的训练。
- **Llama 3 405B 多模态模型即将发布**：据 [报道](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23)，Meta Platforms 将于 **7 月 23 日**（Llama 2 发布一年后）发布其参数量最大的 **Llama 3** 模型，拥有 **405B 参数**，并作为 **多模态 (multimodal)** 产品提供。
   - 这一发布引发了社区的热烈讨论，重点围绕运行如此巨大的模型所需的 **基础设施要求**，例如 **8x H100s 或 8x MI300X GPUs**。
  


**2. 开源 AI 进展**

- **AuraFlow：最大的开源 Text-to-Image 模型**：由 **Fal AI** 开发的 [AuraFlow](https://huggingface.co/fal/AuraFlow) 已作为最大的开源 Text-to-Image 模型发布，采用 Apache 2.0 许可证，已在 `diffusers` 中获得支持，并在 GenEval 上取得了 state-of-the-art 的结果。
   - **LoRA support** 即将推出，目前模型处于 beta 阶段，**社区反馈**至关重要。感谢 [@cloneofsimo](https://twitter.com/cloneofsimo) 和 [@isidentical](https://twitter.com/isidentical) 做出的重大贡献。
- **Cohere Toolkit 走向开源**：Cohere 在 GitHub 上[开源了他们的 chat interface](https://github.com/cohere-ai/cohere-toolkit)，并计划进行 OCI integration，正如 *Sssandra* 所宣布的那样。
   - *Mapler* 对将该开源工具包用于个人项目表示兴奋，并将向社区更新进展。
- **OpenArena 促进 LLM Dataset 增强**：GitHub 上的 [OpenArena 项目](https://github.com/syv-ai/OpenArena) 让语言模型相互竞争，并由第三个模型担任裁判，通过竞争性挑战来**提高 dataset 质量**。
   - 受关于 **Arena Learning** 的 [WizardLM 论文](https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/) 启发，OpenArena 利用 AI 标注的结果进行 LLM 的 supervised fine-tuning 和 reinforcement learning。
  


**3. Community Collaboration and Knowledge Sharing**

- **LlamaIndex 发布 Agentic RAG Cookbooks**：LlamaIndex 与来自 AIatMeta 的 @jeffxtang 合作发布了 [关于 agentic RAG 的 cookbooks](https://t.co/mBNZx9b1JO)，涵盖了从 routing、tool use 到 multi-document agent 构建的主题。
   - 此外，由 @tb_tomaz 和 Neo4j 提供的 [Cypher snippet](https://t.co/dAV2QuAoZH) 可以有效地执行 entity deduplication，辅助知识图谱（knowledge graph）的创建，该内容已分享在 [Neo4j GitHub](https://t.co/lMApLzMOMr) 上。
- **用于 Continued Pretraining 的 Unsloth Notebooks**：Unsloth 提供了用于通过 Ollama 和 Hugging Face 模型训练本地模型的 [notebooks](https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama)，并处理了跨不同序列长度的 [continued pretraining](https://docs.unsloth.ai/basics/continued-pretraining)。
   - 社区讨论了针对不同 `max_seq_length` 的 concatenation 和 truncation 技术，以及在 LoRA 和 PEFT 设置中理解参数差异。
- **LangChain 优化与最佳实践**：LangChain 社区分享了 **embedding functions** 的优化技术，例如使用 **caching mechanisms**（in-memory 或 Redis）以避免重复计算 embeddings，并考虑使用 **async requests**。
   - 讨论还涉及了处理大型数据集时 **FAISS vs Chroma** 的选择，结合两者的优势：使用 **Chroma 进行 persistence**，使用 **FAISS 进行 similarity search**，并提高 **LangChain agent** 的效率。
  


**4. Hardware Benchmarking and Adoption**

- **评估 AI 工作负载的 GPU**：讨论对比了 **3090 vs 4090 GPUs** 在 AI 工作负载中的价值主张，由于代际性能提升相对较小，许多人更倾向于性价比更高的 3090。
   - 关于即将推出的 **NVIDIA 5090** 仅有 **28GB VRAM** 而非 32GB 的传闻，引发了使用 3090 构建经济实惠的 **multi-GPU servers** 以增加 VRAM 容量的建议。
- **H100 GPU 的兴奋与采用挑战**：**H100 GPUs** 的到来引起了极大的兴奋，成员们惊叹于 *'H100 go brrrrr'*，并讨论了相比前几代产品的显著性能提升。
   - 然而，有人担心 **Flash attn3** 目前仅限于 H100 支持，希望它能遵循 **Flash attn2** 的路径，扩展到 3090 和 4090 GPUs。
- **本地 AI 模型基准测试**：一位成员分享了他们的 [个人基准测试表](https://dubesor.de/benchtable.html)，涵盖推理、STEM、coding 和 censorship 类别，使用加权评分系统评估了 83 个任务中的各种本地 AI 模型。
   - 虽然这不代表更广泛的基准测试，但该表提供了一个人的经验见解，并突显了社区对全面模型评估日益增长的兴趣。

---

# PART 1: High level Discord summaries

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **RT-DETR 领先于 YOLO**：**RT-DETR** 在速度和准确性上超越了 YOLO，并与 Roboflow 联手推进目标检测，现在可以通过 [transformers 库](https://github.com/huggingface/transformers?ref=blog.roboflow.com) 无缝访问。
   - 该模型的优势在一篇研究论文 (https://arxiv.org/abs/2304.08069?ref=blog.roboflow.com) 中得到了证实，支持将 RT-DETR 集成到现有工作流中并提升检测任务的表现。
- **利用 Hiera 模型提升效率**：transformers 库现在包含了一个变革性的视觉模型 **Hiera**，它简化了层次结构的复杂性，并在图像分类等性能任务中表现出色。
   - Hiera 的灵活性体现在其多种实现中，包括 *HieraForImageClassification* 和 *HieraBackbone*，详见 [GitHub Pull Request](https://github.com/huggingface/transformers/pull/30356#event-13478146443)。
- **工具包精简 LLM 微调流程**：[Georgian-io 工具包](https://github.com/georgian-io/LLM-Finetuning-Toolkit) 首次亮相，旨在简化多个 LLM 的微调，简化端到端的数据科学流程。
   - 这是一个多功能工具包，它支持通过统一的 config 运行批处理实验、评估指标，并执行超参数和 Prompt 消融实验。
- **AuraFlow 景观可视化**：**AuraFlow** 被誉为最大的开源 text-to-image 模型，最近因其在 GenEval 中表现出的出色结果而备受关注，并得到了 `diffusers` 的支持。
   - 随着 **LoRA 支持** 即将到来，官方鼓励通过 [fal 的 Discord](https://discord.gg/fal-ai) 进行持续开发和社区反馈，为进一步增强功能铺平道路。
- **qdurllm Demo 展示新能力**：[qdurllm demo](https://huggingface.co/spaces/as-cle-bert/qdurllm-demo) 展示了直观输出方面的飞跃，并邀请社区对其先进的交互模型提供反馈。
   - 该产品为潜在的新兴用例以及集成化、可访问的进步开启了对话。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ComfyUI Reactor 恢复安装**：参考 [YouTube 视频](https://www.youtube.com/watch?v=vCCVxGtCyho) 提供了无错安装 **ComfyUI InsightFace** 的解决方案。
   - 这一变通方法已得到用户确认，对 **2024** 年发布的版本依然有效。
- **Deforum 深入研究独特的色彩动态**：为了微调抽象视频的美感，讨论了在 **Deforum Stable Diffusion API** 中将 `color_coherence` 设置为 None 作为增强色彩过渡的潜在方法。
   - 征集社区意见以优化视觉项目中的鲜艳度和清晰度。
- **auto1111 上的生成中断查询**：用户在 auto1111 设置中遇到了停止生成过程的显著延迟，将其归因于 VRAM 限制和软件细微差别。
   - 有人将其比作高速列车的逐渐减速，强调了在突然停止期间需要耐心。
- **分析 AI 工具的性价比**：社区讨论了商业 AI 工具（如 **Runway**，提供每月 **$90** 的方案）的成本，并与免费的本地 AI 选项进行了对比。
   - 尽管免费工具很有吸引力，但成员们认识到高级服务通常能提供更卓越的功能和增强的特性。
- **扩大放大规模：追求免费工具**：对免费创意图像 upscaling 工具的搜索促成了对 **Krita** 和 **GIMP** 等易于获取的软件的推荐。
   - 这些替代方案因其有用的功能且没有经济门槛而受到称赞，符合社区注重资源的偏好。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **FA3 的胜利与 CUDA 的担忧**：一场激烈的辩论评估了 **FA3 vs cuDNN 和 ThunderKittens** 的优劣，揭示了尽管 FA3 在 Attention 机制中具有潜在的加速诱惑，但人们更倾向于简单和易用性。
   - 围绕 **FP8 实现障碍** 的技术担忧以及 **ThunderKittens** 中不存在的 FP8 轨道，引发了对维护复杂性的评估。
- **评估 GPU 访问选项**：成员们赞扬了 **Google Colab** 无缝的 GPU 访问，同时比较了租用 GPU 时 **Coreweave** 和 **Lambda Labs** 的优缺点，强调了价格和分配问题。
   - 讨论强调 **Google Cloud GPU** 是笔记本以外用途中更昂贵但更强大的选择，提升了 **Colab** 在钻研 CUDA kernels 时的易用性地位。
- **优化矩阵乘法**：对话探讨了矩阵乘法中有效的线程分配策略，建议由于内存布局的原因，每行一个线程在缓存和数据加载方面效率更高。
   - 随着有关内存排列的见解浮现，“合并 (coalescing)”的概念成为焦点，强调了在最后一个矩阵维度上进行归约 (reducing) 的效率。
- **AI 训练中的创新策略**：成员们讨论了在 FSDP 中使用 **tensor subclass** 的可行性，因为像 **bitnet work** 这样新兴项目暗示了分布式训练中不断增长的应用。
   - 社区认可了持续的贡献，并准备合作制定一份 **启用 tensor subclass 的开发者指南**，以应对未来的需求。
- **LLM.C 内部的协作与增长**：**LLM.C 社区** 正忙于围绕模型共享和资源整合的倡议，这在 **Hugging Face** 上组织结构的创建中可见一斑。
   - 分享了关于执行优化和微调大规模模型的见解，还激发了关于 **FP8 带来 33% 速度提升** 的想法，尽管需要考虑内存重用。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **LLVM 创始人的纪事**：最近一段采访 LLVM、Clang、Swift 和 Mojo 创始人的 [Primeagen 视频](https://www.youtube.com/watch?v=ovYbgbrQ-v8) 在 YouTube 上公开后引发了讨论。
   - 参与者指出，详细的见解对于理解 Mojo 创作背后的开发哲学是非常棒的资源。
- **Mojo 的 REPL 迭代**：围绕 Mojo REPL 缺乏表达式立即输出的争论不断，并将其与 Python 的 REPL 行为进行了比较。
   - 尽管当前功能不会直接显示像 `1+1` 这样的结果，但建议成员通过 [GitHub issues](https://github.com/modularml/mojo/issues/2809) 提交请求以整合这些功能。
- **MAX 网站改版拥抱清晰度**：Modular 的 **MAX framework** 占据了 [改版后网站](https://modular.com) 的核心位置，强调了其广泛的开发者基础和清晰的许可条款。
   - 该网站展示了 Max 的性能实力与 Mojo 语言提供的易用性之间的协同作用，而无需深入底层编码。
- **通过 Mojo 的 MAX 获得 GPU 收益**：出现了一场关于在 MAX 中使用 Mojo 编写自定义 GPU kernels 以增强性能的有前景的对话。
   - 这为利用 MAX 强大的接口和 Mojo 敏捷的 kernel 编译开辟了道路，而无需直接涉及 CUDA。
- **MAX Model 执行中的数据类型差异**：在执行 MAX Model 时出现了 **数据类型问题**，导致在使用 `PythonObjects` 时预期与实际结果不匹配。
   - 将 `np.full()` 操作的 `dtype` 修正为 `np.float32` 提供了解决方案，强调了模型执行参数中所需的精度。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemini 随 Token 扩展而飞跃**：[Gemini 1.5 Pro 拥有 200 万 Token 窗口](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/)，并引入了 **context caching** 和 **code execution** 功能。
   - **AI 开发者**对无限制的 JSON 容量感到兴奋。
- **FlashAttention 在 Hopper GPU 上疾驰**：FlashAttention-3 承诺实现高效的 **Hopper GPU** 利用率，**FLOPs** 利用率高达 **35%**，详见 [Tech Blog](https://www.together.ai/blog/flashattention-3)。
   - “大幅提升 FLOPs 利用率”仅限于 **Hopper 用户**。
- **TF-ID 模型瞄准视觉语言任务**：胡一飞（Yifei Hu）[发布了 TF-ID 模型](https://github.com/ai8hyf/TF-ID)，包括训练代码、数据集和权重，采用 MIT License，适用于视觉语言任务。
   - 这些模型仅需几百个特定领域的元素即可进行 finetune。
- **CodeGeeX4 削弱了 GPT 的优势**：新的 [CodeGeeX4-ALL-9B 模型](https://huggingface.co/THUDM/codegeex4-all-9b)在代码生成能力上盖过了 **GPT-3.5 和 GPT-4**。
   - 该模型实现了顶尖性能，拥有 **128k context**，并支持多种编程语言。
- **Meta 备受期待的 LLaMA 3 亮相**：Meta Platform 计划于 7 月 23 日发布其 LLaMA 3 模型，这可能会带来显著的 AI 进展，引发了广泛关注。
   - 此次发布（[详见此处](https://huggingface.co/nvi)）可能会重塑 AI 应用部署的硬件偏好。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **OpenAI 预告博士级别的突破**：OpenAI 暗示即将推出的模型将具备等同于博士学位的解题能力，引发了关于接近 AGI 的讨论。
   - 一位匿名人士泄露了 GPT-4 的演示视频，展示了其先进的类人解题能力。
- **Anthropic 的 AI 预测**：Anthropic 的 Dario Amodei 预测了即将到来的 AI Safety Levels，暗示 **A.S.L. 3** 可能最早在今年出现，而 **A.S.L. 4** 将在 2025-2028 年间出现。
   - **A.S.L. 4** 引发了对生物和网络技术滥用可能加剧全球风险的警示。
- **社区质疑 OpenAI 的策略**：在潜在突破的消息中，社区中出现了对 OpenAI 战略性发布模式的怀疑声音。
   - 讨论围绕着 OpenAI 的预告可能是一种旨在提高估值的手段，尽管他们之前已经取得了成就。
- **用 llm.c 玩转 GPT-2**：**Karpathy** 展示了使用 llm.c 高效复制 **GPT-2 (1.6B)** 的过程，兼顾了性能与成本效益。
   - 该实现证明了 llm.c 进行大规模语言模型训练的能力，仅需 24 小时即可完成。
- **C++ 带来的简单与安全**：Safetensors.cpp 作为 **LibTorch** 的零依赖 C++ 库首次亮相，减轻了模型开发中的数据处理负担。
   - 目标很明确：简化模型数据流程，确保工作流更加顺畅且高效。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Labs：用还是不用？**：关于 [Perplexity Labs](https://discord.com/channels/1047197230748151888/1047649527299055688/1233731074685665280) 实用性的辩论激增，社区成员剖析了其在不同设备上的通用性。
   - **利弊分析**被广泛讨论，重点关注了 Labs 的集成优势，并质疑其相对于移动端使用与 Web 界面的优势。
- **Claude 3.5 碾压 Claude 3 Opus**：[Claude 3.5 在推理和逻辑方面优于其前身 Claude 3 Opus 的卓越表现](https://discord.com/channels/1047197230748151888/1047649527299055688/1261038487453302895)引起了所有人的关注，预示着模型主导地位的转变。
   - 虽然对 Claude 3.5 的**赞誉是一致的**，但人们对 Opus 3.5 等未来版本重新平衡局势的潜力产生了猜测。
- **AI 成为糖尿病管理的灯塔**：用于糖尿病管理的 AI 成为焦点，讨论围绕协助患者和医生的 App 展开，重点在于**洞察推导**而非仅仅是胰岛素调整。
   - [最近的进展](https://www.perplexity.ai/search/are-there-any-advances-in-apps-5NVLNla1T6.oHZAm_U70fw)得到了关注，这些进展不仅提供自动胰岛素给药，还提供预测性见解，重塑了患者护理。
- **Error 524 阴云笼罩 Perplexity API**：AI 工程师报告称，在将 Perplexity 与异步框架集成时，尽管保持在规定限制内，仍偶尔出现 **Error 524**。
   - 切换模型增加了难题，在 `llama-3-{8b/70b}-instruct` 到 `llama-3-sonar-{large/small}-32k-online` 之间的转换导致了类似的错误，令用户感到困惑。
- **Cloudflare 引发 Perplexity API 动荡**：排错过程揭示了 [Cloudflare 是罪魁祸首](https://discord.com/channels/1047197230748151888/1161802929053909012/1261093061883199552)，它导致了 VPN 访问 Perplexity API 被阻断，这对许多人来说是一个新发现。
   - 虽然有些人感到困扰，但其他人发现绕过 VPN 是有效的解决方法，恢复了访问并平息了这场**风波**。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPT-4Chan 终结了 TruthfulQA 的统治地位**：来自 [Maxime Labonne 的推文](https://x.com/maximelabonne/status/1746317053143773628)重新引发了关于 GPT-4Chan 尽管在 ChatGPT 出现后仍曾在 **TruthfulQA** 上占据主导地位的讨论。
   - 参与者一致认为，像 TruthfulQA 这样的一些基准测试可能会误导，而像 **MT-Bench** 这样的基准测试被认为更能反映真实性能。
- **Jsonnet，配置界“必要的恶”？**：虽然 Jsonnet 因其精简的配置能力而受到赞赏，但一场讨论揭示了它在调试方面的不足，导致用户对其*爱恨交织*。
   - 尽管面临挑战，Jsonnet 因其简洁性而在各种配置任务选项中脱颖而出，其作用得到了认可。
- **伦敦 AI 聚会未达预期**：论坛上表达了对**伦敦 AI 聚会**的失望，反映出它们对于寻求更深层次 AI 探讨的人来说还不够。
   - 建议指向**学术研讨会**和诸如 **ICML** 之类的会议，以满足对更实质性技术集会的渴望。
- **LLM 面临简单但严峻的挑战**：更新后的 [Alice in Wonderland 论文](https://arxiv.org/abs/2406.02061) 揭示了一些简单的谜题，这些谜题难倒了像 **Claude 3.5 Sonnet** 这样的 SOTA 模型。
   - 围绕 SOTA LLM 无法处理简单修改的讨论，突显了对稳健基准测试的需求，并增强了我们对模型局限性的理解。
- **内存带宽：GPT-2 训练的限制因素**：讨论围绕在一小时内使用万亿级 Token 数据集训练 **GPT-2** 模型所需的 **1000 倍内存带宽**放大要求展开。
   - 焦点转向 **Hadamard 变换**，将其作为量化难题的创新解决方案，详见 [Together 的博客文章](https://www.together.ai/blog/flashattention-3)。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **FlashAttention-3 激发 GPU 性能**：[FlashAttention-3](https://x.com/tri_dao/status/1811453622070444071) 现已发布，承诺在 FP16 上实现 1.5-2 倍的速度提升，在 H100 GPUs 上达到高达 740 TFLOPS。
   - 据报道，新版本在 H100 上实现了 **75% 的利用率**，并有可能在使用 FP8 时达到 1.2 PFLOPS。
- **OpenAI 利润丰厚的账目**：根据最近的一份[报告](https://x.com/jvnixon/status/1811278381184672156)，OpenAI 的收入预计将达到 19 亿美元，其中 ChatGPT Plus 领跑。
   - 这一推测突显了 OpenAI 可能在行业中处于领先地位，其 ChatGPT Enterprise、API 和 Team 产品的数据表现令人印象深刻。
- **OpenAI 揭晓 AGI 框架**：OpenAI 发布了一个用于追踪 [AGI 的 5 级框架](https://x.com/shiringhaffary/status/1811508824970264595)，并将自己定位在第 2 级。
   - GPT-4 的推理能力在最近的一次会议上得到了展示，表明了该战略框架中概述的进展。
- **去中心化 AI 训练起飞**：[Prime Intellect 的 OpenDiLoCo](https://x.com/shaughnessy119/status/1811459606377582857)（借鉴自 DeepMind 的模型）支持在全球节点上进行分布式 AI 训练。
   - 一个成功的案例涉及一个在三个不同国家的多个节点上训练的 1.1B 参数模型。
- **Fireworks 在 AI 融资领域崭露头角**：[Fireworks AI](https://x.com/lqiao/status/1811500361485517153) 最近为其旨在推进复合 AI 系统发展的平台获得了 5200 万美元的 B 轮融资。
   - 资金将用于 Nvidia 和 AMD 的集成，并量身定制企业级 AI 解决方案。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **使用 Indexify 进行无缝合成**：[Prashant Dixit](https://x.com/Prashant_Dixit0/status/1811618990352912846) 在 Towards AI 的出版物中重点介绍了从非结构化源中进行**结构化数据提取**的方法。
   - 介绍了使用 **Indexify** 创建**数据摄取和提取流水线**的方法，更多见解请参阅 [Towards AI 的文章](https://pub.towardsai.net/structured-financial-data-extraction-from-unstructured-data-ca2c8d166de6)。
- **向量优势：Chroma 优于 OpenAI**：讨论围绕正确加载带有 **OpenAI embeddings** 的 Chroma 向量存储所需的配置展开，强调了为了无错运行而保持一致的 `collection_name`。
   - 参与者探索了持久化存储策略以及对嵌入文档的有效管理，以减少冗余计算。
- **光速 Embeddings**：交流了加速 OpenAI embedding 函数的技术，缓存策略是核心，范围从 **in-memory** 到使用 **Redis** 等工具。
   - 改进嵌入过程的方法包括减少 token 加载和利用异步嵌入请求。
- **FAISS 还是 Chroma：数据集的抉择**：随后展开了关于 **FAISS vs Chroma** 的辩论，倾向于使用 FAISS 高效处理大规模数据集，而 Chroma 则因其在小规模集合中的持久化能力而受到青睐。
   - 一种结合了 Chroma 持久化存储和 FAISS 相似性搜索的混合方法被认为是一种有效的解决方案。
- **LangChain Agents 的进展**：剖析了关于 LangChain agents 不必要的重新嵌入（reembedding）挑战，重点关注最小化向量存储初始化时间。
   - 提议的解决方案涵盖了持久化机制以及其他各种改进，以增强 LangChain agents 的运行。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **FlashAttention 点燃 LLM 性能**：一项技术回顾揭示了 [FlashAttention](https://tridao.me/blog/2024/flash3/) 及其后续版本如何简化 Transformer 训练，通过优化的内存操作大幅增加了 **GPT-4** 和 **Llama 3** 的上下文长度。
   - **NVIDIA** 的 **4090** 相比 **3090** 仅有微小提升，但 **FlashAttention** 技术的加入引发了关于在新方法的内存管理效率下，是否真正需要高端显卡的讨论。
- **关于 NVIDIA 下一步行动的传闻**：传言暗示 **NVIDIA 5090** 仅配备 **28GB VRAM**，而非预期的 32GB，这引发了广泛猜测；同时一篇 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1dzejen/cheapest_144gb_vram_server_youll_ever_see/) 提供了一个获取海量 VRAM 的 DIY 替代方案。
   - 虽然关于 3090 更好的性价比仍有争论，但多 V100 设置作为 AI 任务有力竞争者的可能性也得到了剖析，讨论倾向于使用单一高性能 GPU 构建以获得最佳周转效率。
- **Vulkan 迎来支持浪潮**：由于 OpenCL 表现滞后，无法在 **7600XT** 上加载模型，逐渐在 AI 工作中失宠，有传言称 **Vulkan** 将进入 LM Studio 的支持列表，承诺为模型交互带来新体验。
   - 讨论指出 **Vulkan** 的受欢迎程度正超越 OpenCL，这对 **ollama** 用户来说是一个受欢迎的变化，但在热切期待中，具体的发布日期仍未确定。
- **爱因斯坦品牌引力**：Salesforce 投入了高达 **2000 万美元** 将其新 AI 模型命名为 **Einstein**，这引发了业内的各种调侃，以及对这笔投资明智性的审慎评估。
   - 现场充满幽默气氛，有人生动地想象 **Einstein** 的形象被困在公司框架中，并对 AI 品牌束缚可能成为梗的潜力发表了俏皮话。
- **使用 LM Studio 进行响应式 AI 开发**：一项创意尝试出现，一名工程师通过 LM Studio 的 API 将 **Gemma 2** 集成到 React 应用程序中，并建议考虑使用像 **Faiss** 这样的 embedding 数据库进行 RAG 设置，以优化批量 PDF 处理。
   - 随着开发者交流成败经验并倡导社区内更具同理心的支持，LM Studio 的 SDK 被推崇为将前沿 AI 融入具有丰富用户交互应用的得力助手。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **去中心化为芯片赋能**：讨论集中在**去中心化计算**对 AI 任务的好处，利用 **stable diffusion** 和未被利用的**闲置处理能力**来优化 **CMOS** 芯片。
   - 用户呼吁通过去中心化扩展**高性能计算 (HPC)** 能力，从而实现精细的并行计算。
- **OpenAI 的下一代 AI 蓝图**：[OpenAI 揭示了一个新的层级系统](https://archive.is/SLtFQ)，描述了从具有博士级问题解决能力的 'Reasoners' 到具有更广泛能力的 'Agents' 和 'Organizations' 的进展。
   - **Claude** 因其在文档理解方面优于 ChatGPT 而受到关注，这表明人们对上下文长度的关注度日益提高。
- **ChatGPT-5 的期待达到顶峰**：**GeekyGadgets** 的消息暗示 [ChatGPT-5 测试](https://www.geekygadgets.com)将于 2024 年底开始，引发了用户的兴奋与怀疑。
   - 预期的 ChatGPT-5 功能包括增强的情绪智力、减少指令重复以及可能涉足多模态能力。
- **对 ChatGPT-4o “健忘症”的担忧日益增加**：用户反映虽然 **ChatGPT-4o** 速度很快，但经常忘记最近的指令，质疑其在编程等任务中的效能。
   - 对 v3.5 记忆能力的怀旧突显了性能速度与操作召回率之间的权衡。
- **RAG 聊天机器人提示词微调**：开发者正在微调 **RAG** 聊天机器人的指令，旨在减少收到奇怪或矛盾答案的可能性。
   - 社区建议提高聊天机器人提示词的清晰度，以确保有效且逻辑严密的交互。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R Plus 的实际应用场景揭晓**：由 **Mapler** 领导的社区成员们集思广益，探讨了 Command R Plus 模型的实际应用，涵盖了社交媒体内容创作、播客描述撰写以及团队沟通增强。
   - 值得注意的是，*Sssandra* 强调了她日常将该模型与 Notion 和 Google Drive 集成，以方便处理社区咨询。
- **利用 Cohere 实现自动更新革命**：目前正在讨论如何利用 Command R Plus 和 **Lang chain**，通过 Discord 中的 webhooks 自动推送 AI 新闻，**Mapler** 正领导这一计划。
   - **Karthik_99_** 已主动提供帮助，建议可以集成类似 chat-GPT 的界面，目前正等待社区反馈。
- **Cohere 工具箱进入开源生态系统**：*Sssandra* 自豪地分享了 [Cohere 的聊天界面已在 GitHub 上开源](https://github.com/cohere-ai/cohere-toolkit)的消息，并预告了即将进行的 OCI 集成。
   - **Mapler** 表现出极大热情，打算将其用于个人项目，并向社区更新进展。
- **追求 AI 生成的独特表情符号**：**Roazzy** 发起了关于开发 AI 驱动工具来创建独特表情符号的讨论，目前的方法还局限于手动创作。
   - **Karthik_99_** 询问了现有解决方案，强调了该功能在用户驱动型平台中的潜力。
- **Cohere Embedding Model 在成本效益上的突破**：一位成员发布消息称 **Cohere's embedding model** 已将运营成本大幅降低了 **40-70%**，引发了热烈讨论。
   - 该公告获得了社区的一致好评，表达了对这种高性价比进展的赞赏。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Llama 向更大规模学习的跨越**：拥有 **405B** 参数的 **Llama 3** 即将于 **7 月 23 日** 发布，引发了广泛期待。该模型被设计为更强大的 Multimodal 模型，详情见[此简报](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23)。
   - 从 Llama 2 到 3 的飞跃加速了通往复杂 AI 的道路，引发了围绕支持性基础架构的讨论，正如 [Stephanie Palazzolo](https://x.com/steph_palazzolo/status/1811791968600576271?s=46) 所发并被社区频道转发的内容。
- **OpenAI 神秘的 Strawberry 策略**：关于 **OpenAI's Strawberry** 项目的泄露信息显示，它与斯坦福大学 2022 年的 **STaR** 方法有相似之处，突显了 AI 推理技术的进步，据 [Reuters](https://www.reuters.com/technology/openais-top-secret-strawberry-project-unveiled-2023-07-15/) 报道。
   - 社区对这一秘密行动充满了猜测和分析，认为这可能成为 OpenAI 追求更具上下文理解能力和连贯性 AI 模型过程中的一个重要里程碑。
- **Self-hosting 大模型：一种特权式的困境**：深入探讨 **Self-hosting 400B** 参数模型的物流需求后发现，需要大约 **400GB VRAM**。这使得对话转向了资源可用性，并倾向于在自有硬件不足时使用 API。
   - 这种情况让 GPU 租赁的超大规模算力提供商（Hyperscalers）备受关注，尤其是当不涉及私有数据关注时，这是从技术社区对托管复杂性和 API 优势的剖析中得出的结论。
- **Distillation 难题：Sequential Soft-targeting**：著名论文中描述的 **Soft-target distillation** 过程正受到审视，有人提问关于在保持概率的同时进行 Sequential 处理的可能性。
   - 社区反馈指向了一些替代策略，例如在在线建模过程中对齐内部表示（Internal Representations），以及这可能如何简化现有方法。
- **GPT-4：价格优于性能？**：在各种 AI 服务中，**GPT-4** 以 **每月 20 美元** 的价格脱颖而出，成为卓越模型的代表，使竞争对手的低价替代方案显得黯然失色。
   - 诸如 [aaron holmes](https://x.com/aaronpholmes/status/1811870687037960467?s=46) 的推文进一步推动了对比讨论，聚焦于 AI 模型估值、企业选择和消费者偏好的持续对话。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 的索引精通**：George Hotz 在 [tinygrad 中引入了一种创新的索引内核](https://x.com/__tinygrad__/status/1811578147982491938)，这是一种打破常规的补充，通过创新性地折叠求和循环（sum loop）绕过了典型的限制。
   - 这种后端生成的方法确保了严格且高效的内存访问子集，**优化了内核性能**。
- **借鉴 PyTorch 路线图规划航向**：一名成员建议效仿 [PyTorch 分享的 2024 H2 计划](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2024-h2-roadmaps/2226)，主张采取精确且开放式的开发策略。
   - 其目标是镜像 PyTorch 的透明路径，为**增长和开发提供清晰的基础**。
- **tinygrad 中的梯度下降困境**：从头开始实现梯度下降的尝试遇到了阻碍，一名成员指出在没有定义 `optimizer.step` 的情况下过程非常缓慢。
   - 他们寻求优化缓慢步骤的见解，参考了 [代码样本](https://github.com/karpathy/makemore/blob/988aa59e4d8fefa526d06f3b453ad116258398d4/names.txt) 和 George Hotz 的手动实现策略。
- **优化张量操作**：Hotz 的命令 `model.weights.assign(model.weights - lr * model.weights.grad).realize()` 简化了张量操作（高效梯度下降的关键组件）。
   - 正如 George Hotz 所言，理解 realization 的必要性成为了**计算实例化的关键**。
- **解决张量索引 Bug**：在处理张量操作时，一个断言错误暴露了张量索引中的 Bug，导致了 "idx.max too big" 的复杂问题。
   - 参与此次调试会议凸显了社区在**完善 tinygrad 内核效率**方面的作用。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **H100 GPU 引发性能热潮**：**H100 GPU** 在成员中引发了一波热情，反馈强调了其在**性能能力**上的重大飞跃。
   - H100 系列的迅捷性能预示着计算能力的新基准，显而易见地超越了其前代产品。
- **奖励模型中的 Attention Masking 受到审视**：**Attention Masking** 在 **reward_health_models** 中的作用和影响成为辩论话题，社区寻求其必要性的澄清。
   - 虽然关于其与 **axolotl** 特定训练方法相关性的疑问依然存在，但开放式讨论标志着对该技术的持续探索。
- **OpenRouter 与 OpenArena 的连接受到关注**：社区成员对集成 **openrouter.ai** API 以开发 **WizardLM arena 数据集** 的开源等效版本表现出兴趣。
   - 其中提到使用 **ollama** 开发社区驱动的 [OpenArena 项目](https://github.com/syv-ai/OpenArena) 取得的进展，强调了协作开发。
- **Flash Attention 兼容性引发疑问**：**Flash attn3** 的兼容性问题引发了讨论，并指出了对 **H100 GPU** 的限制。
   - 人们对更广泛的 GPU 支持寄予厚望，正如之前 Flash attn2 更新适配了 *3090 和 4090* 一样。
- **GaLore vs. Q-Galore：量化占据领先**：讨论强调 **Q-Galore** 是 GaLore 的高效继任者，采用量化技术缩短训练时间，详情见 [Hugging Face 论文](https://huggingface.co/papers/2407.08296)。
   - Q-Galore 的方法在继承 GaLore 策略的同时避免了 SVD 的时间开销，成为处理梯度子空间的重要升级。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepInfra 数据困境讨论**：一名成员对 **DeepInfra** 的数据政策表示担忧，引发了关于公司如何处理从用户输入中获取的训练数据的讨论。
   - 讨论明确了 DeepInfra 会记录使用情况，但不会利用用户输入进行训练，详情见其 [隐私政策](https://deepinfra.com/privacy)。
- **Beta 集成功能吸引更智能的机器人**：关于 **Integrations (Beta)** 功能的讨论展开，重点在于为 Groq 等外部提供商使用自定义 API key。
   - 对话预期未来的扩展可能会探索模型 API 之外的集成，引发了对潜在应用场景的好奇。
- **为挑剔的模型优化 Prompt 位置**：成员们交流了提高模型性能的技巧，包括建议将文本 Prompt 置于图像之后，以辅助能力较弱的模型。
   - 据报道，这种放置技术可以增强理解力，并让能力较低的模型给出更好的回复。
- **405b 的到来激发 AI 追随者的热情**：即将发布的 **405b 模型** 引起了轰动，社区期待值很高。
   - 社区的热议由 [Bindu Reddy 的推文](https://x.com/bindureddy/status/1811807596682355125?s=46) 助推，该推文提到了模型的预期发布，将 7 月 23 日标记为开源 AGI 的重要日子。
- **专业化推测引发学者讨论**：一场关于多个专业化模型是否优于单个通用模型的对话浮出水面，涉及 OpenAI 和 Anthropic 等公司。
   - **Alex Atallah** 加入了辩论，主张考虑专业化模型，并征求社区对首选类型的意见。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Clip Retrieval 失效**：一位用户注意到 **clip retrieval** 不再可用，引发了关于获取数据集替代方法的询问。
   - 该问题似乎与数据集的移除有关，这可能意味着对数据可用性的更广泛限制。
- **内存占用大户模型**：一个小规模模型训练任务出现了异常高的 **19GB 内存占用**，引起了频道对内存效率低下的关注。
   - 社区正在积极探究为什么一个仅有 25 万参数的模型在较小的 batch size 下会吞噬如此多的内存。
- **Nematron 340B 代码探索**：关于 **Nematron 340B** 代码示例的咨询激增，重点在于奖励模型（reward model）的参数管理。
   - 目前细节仍然很少，这为频道内共享编码实践提供了机会。
- **AuraFlow 带来的华丽流动**：Fal AI 的新文本生成图像模型 **AuraFlow** 随着其 [发布公告](https://blog.fal.ai/auraflow/) 在频道内引起关注。
   - 它在遵循 Prompt 方面的熟练表现，重新点燃了人们对 **开源 AI 领域** 的信心。
- **LLMs 的 AIW 难题**：一篇更新的 ArXiv 论文展示了 **AIW 问题**，揭示了 LLMs 在基础任务推理能力上的巨大鸿沟。
   - 讨论围绕当前基准测试（benchmarks）的不足，以及 [该论文](https://arxiv.org/abs/2406.02061) 所强调的至关重要但被忽视的能力展开。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Agentic RAG 的成功秘诀**：LlamaIndex 与 AIatMeta 合作，发布了关于 **agentic RAG** 的 cookbook，涵盖了从 agent 路由到多文档 agent 构建的多方面主题。
   - 热情的爱好者们通过 [Twitter 公告](https://t.co/mBNZx9b1JO) 初步了解，并在此处 [here](https://t.co/l2ztPRsAd8) 获得了进一步见解。
- **通过 Cypher 片段实现去重趣闻**：由 @tb_tomaz 和 Neo4j 团队打造，一段强大的 Cypher 片段简化了 **entity deduplication**（实体去重）的艺术，将技术实力与 URI 魔法相结合。
   - 为了引起关注，他们分享了一个实用的 [示例片段](https://t.co/dAV2QuAoZH)，并在 [Neo4j GitHub](https://t.co/lMApLzMOMr) 上简化了代码获取路径。
- **Gemini 功能调用的澄清请求**：Gemini 模型的 function calling 功能存在困惑；[GitHub commits](https://github.com/run-llama/llama_index/pull/14088) 看起来很有希望，但遇到了报错，声称 API 不支持。
   - 澄清路径建议通过 `pip install -U llama-index-llms-vertexai` 升级工具包，希望能拨开围绕 **Gemini-1.5-flash-latest** 能力的迷雾。
- **聚焦库：索引大型代码**：爱好者们剖析了索引大型代码库的策略，讨论将代码翻译为 markdown-pseudocode（Markdown 伪代码）是否能增强聊天机器人的理解力。
   - 对话围绕双聊天机器人系统的需求展开，一个用于问答，另一个用于生成代码片段。
- **RAG 评审：略读规范文档**：设想了 **RAG** 在剖析冗长规范文档中的作用，旨在不耗尽 token 限制的情况下提高评审流程的效率。
   - 社区思考了评估庞大规范的方法，权衡了节省 token 的优势与 **RAG** 的潜力。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **调用错误解析**：用户报告在调用 `OpenInterpreter.chat()` 时出现 **APIConnectionError**，且 'select' 无法确定 agent 的角色。
   - 如 [文档](https://docs.litellm.ai/docs/providers) 所述，通过显式传递 **LLM provider** 可能会解决该错误。
- **使用 Phi-3 进行快速函数调用**：**Phi-3** 因其快速且可靠的 function calls 引起关注，暗示了全本地 Fast Fourier 运行的潜力。
   - 人们对这一优化寄予厚望，这可能意味着在不久的将来实现 **更快的计算**。
- **GUI 的重大进展**：**Open Interpreter 的 GUI** 迎来了重大升级，现在具备分支聊天、可编辑消息、自动运行代码和聊天保存功能。
   - 这些强大的新功能也带有一些限制，详见 [GitHub 仓库](https://github.com/jbexta/AgentPilot)。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **解决遥测问题**：讨论围绕 **self-hosted ML telemetry**（自托管 ML 遥测）展开，重点关注 [Langfuse](https://langfuse.io)、[WandB](https://wandb.ai) 和 [OpenLLMTelemetry](https://github.com/openllmtelemetry) 等平台。
   - 成员们强调了选择一个符合 **ML 项目** 特定需求的平台的重要性。
- **聊天机器人的 API Key 寻求**：一位需要 **OpenAI API key** 的成员表达了对聊天机器人项目教程的需求，强调了其短期必要性。
   - 重点在于在教程中使用该 API key 进行演示。
- **寻求额度澄清**：出现了关于额度余额的查询，用户 **reneesyliu-571636** 直接询问如何进行 **credit balance check**（余额检查）。
   - 另一位成员寻求关于其账户状态的帮助，可能暗示了关于账户管理这一话题的更广泛问题。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **倡导影响 AI：Llamafile 迈向立法**：**Mozilla 的 Udbhav Tiwari** 在美国参议院面前倡导开放 AI 系统，强调透明和可访问技术的重要性。
   - [参议院听证会](https://discord.com/channels/1089876418936180786/1260972784696295536)的焦点在于开放性在 AI 中的关键作用，这与 Mozilla 自身的倡导方向高度一致。
- **开发者时间延长：仍欢迎申请！**：错过了 **Builders Accelerator** 的早期窗口？别担心，在初始截止日期之后仍然欢迎提交申请。
   - 之前已分享过详细信息，感兴趣的各方可以查看计划目标并按照此[公告](https://discord.com/channels/1089876418936180786/1089876419926032396/1255588743599882260)中的说明进行申请。
- **不要错过：AI 活动盛会等待着你**：一系列引人入胜的活动即将登场，包括**结合 LLM 的 Open Interpreter**、Benjamin Minixhofer 关于 **Zero Shot Tokenizer Transfer** 的演讲，以及与资深工程师进行的 **AutoFix** 环节。
   - 渴望参与？请为[即将举行的活动](https://discord.com/events/1089876418936180786/1260611047341953034)预留虚拟席位，参与前沿 AI 工具和讨论。
- **雕琢开源 AI 定义**：**Open Source AI Definition Draft v 0.0.8** 步入聚光灯下，寻求社区见解并与 OECD 的 AI 系统解读保持一致。
   - 呼吁社区行动起来，在 [OSI 博客](https://blog.opensource.org/open-source-ai-establishing-a-common-ground/)上审查并评论这一不断演进的成果。
- **整数还是浮点数：量化困境**：AI 工程师们正在思考 **llama.cpp** 在 matmul 操作中是使用整数还是浮点数计算，这与 **ggml-quants.c** 中的程序有关。
   - 这一数学策略——技术爱好者的热门话题——可能需要在进行整数点积运算之前对浮点激活值进行量化。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **LLM 在 OpenArena 中展开对决**：**LLM Arena** 是一个对战区，来自 **Ollama** 和 **OpenAI** 端点的语言模型在第三个模型作为裁判的指导下进行决斗。
   - 其目标是**提高数据集质量**，已在 [OpenArena 的 GitHub](https://github.com/syv-ai/OpenArena) 上通过竞争性挑战进行了展示。
- **WizardLM 论文为 Arena Learning 施展魔法**：**OpenArena** 从 [WizardLM 论文](https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/)中汲取灵感，倡导在 LLM 训练后进行 **Arena Learning**。
   - 通过模拟聊天机器人战斗并利用 **AI 标注的数据集**，该方法通过监督微调和强化学习技术来磨练模型。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **拓展 MLOps 的视野**：发起了一场关于涵盖**产品**和**研究**等不同领域的讨论，特别是**推荐系统**、**信息检索 (IR)** 和**检索增强生成 (RAG)**。
   - 对话鼓励开放性建议，并表达了对探索 **Elastic** 及其在这些领域潜力的特定兴趣。
- **Elastic 爱好者涌现**：另一位用户表达了同样的看法，表示愿意就 **Elastic** 进行详细对话。
   - 该用户标记了一位同事，以启动关于 **Elastic** 如何增强他们当前业务的深入讨论。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Torchtune Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第二部分：各频道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1261053927076659292)** (1 条消息): 

> - `qdurllm demo`
> - `高级 RAG 工作坊`
> - `Intel HF 模型仓库`
> - `自我审查编程助手`
> - `使用 LlamaIndex 训练聊天机器人` 


- **qdurllm demo 更新**：由社区成员提供的 [qdurllm demo](https://huggingface.co/spaces/as-cle-bert/qdurllm-demo) 展示了具有直观输出的新功能。
- **AI 与知识图谱的未来**：一场名为 [Leveraging Knowledge Graphs for Advanced RAG](https://www.youtube.com/watch?v=9wqVz0LDYgg) 的在线工作坊讨论了使用 Langchain 和 Neo4j 进行**自然语言查询 (natural language querying)**。
   - 它提供了关于使用 **Cypher 查询语言**与图数据库进行交互的见解。
- **Intel CPU 最大化 HF 模型效率**：一个新的 [GitHub 仓库](https://github.com/sleepingcat4/intel-hf) 展示了在 **Intel CPU** 上高效运行任何 **Hugging Face 模型**的方法。
- **gary4live 插件发布**：**gary4live** Ableton 插件现已在 Gumroad 上免费提供，详见[此处](https://x.com/thepatch_kev/status/1810063563823907172)。
- **了解 RegMix**：[RegMix](https://huggingface.co/blog/SivilTaram/regmix) 介绍了一种使用**数据混合 (Data Mixture)** 进行有效**语言模型预训练 (language model pre-training)** 的新方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/as-cle-bert/qdurllm-demo">Qdurllm Demo - as-cle-bert 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=9wqVz0LDYgg&ab_channel=DecodingDataScience)">AI 的未来：利用知识图谱实现高级 RAG</a>: 准备好进入使用 Langchain 和 Neo4j 进行自然语言查询的世界吧！学习如何使用 Cypher 查询语言与图数据库进行交互...</li><li><a href="https://wandb.ai/sauravmaheshkar/llamaindex-local-models-index/reports/Training-a-chatbot-on-personal-data-with-LlamaIndex-and-W-B--Vmlldzo4MzQzMDE3)">Weights & Biases</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://x.com/thepatch_kev/status/1810063563823907172)">thecollabagepatch (@thepatch_kev) 的推文</a>: 13 位传奇人物刚刚收到了一封关于 gary4live 的邮件，这是一个可以实现此功能的 Ableton 插件，现在可以在 Gumroad 上下载，伙计们 ⬇️链接 @_buildspace @_nightsweekends</li><li><a href="https://youtu.be/38ae7hqzX5s)">Gemma2:27 Ollama 修正！现在表现惊人！</a>: 今天，我们将再次使用 Ollama 测试 Gemma 2 27b，因为 Ollama 推送了一个更新来修正与 Gemma 2 相关的问题，现在它可以正常工作了...</li><li><a href="https://youtu.be/gAtUdnN1_xM?si=L_1vdbjzu4yHyUlA)">Rauf 编写的 SK-LEARN 入门</a>: scikit-learn (sklearn) 机器学习库的简短基础介绍。我最初是为我的演示文稿创建的，但我意识到这会很有用...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1261036924215234692)** (613 条消息🔥🔥🔥): 

> - `GPU 模型与问题`
> - `云与免费资源`
> - `训练技术`
> - `HF 集成`
> - `笑话与社区互动`

- **关于 GPU 型号和资金的讨论**：成员们讨论了与各种 GPU 型号（如 **1060 3GB**）相关的技术和情感损失，以及为了获得更好的渲染能力而寻找的潜在替代品（如 **A6000**）。
   - 预算限制导致人们考虑从 **Facebook Marketplace** 淘货以及通过自由职业赚取额外资金。
- **探索云端和免费计算资源**：对话详细讨论了用于训练的 **A100** GPU 的成本和效用，并推荐了 **backprop.co** 和免费的 **Google Colab T4** 实例以实现更经济的使用。
   - 讨论还包括 Google Cloud 的 **TPU 研究信用额度 (TPU research credits)**，该计划为符合条件的项目提供免费的集群访问权限。
- **训练 Diffusion 模型和 LoRa 技术**：成员们在训练 Diffusion 模型时面临挑战，提到了在较便宜的 GPU 上使用 **LoRa**，以及由于成本原因在 **A100** 上进行 **full finetuning** 的复杂性。
   - 提供了关于租用较小 GPU 和探索 **Colab** 以获得更经济选择的指导，特别是针对**角色风格迁移 (character style transfer)**。
- **HF 集成与更新**：分享了 [Hugging Face](https://huggingface.co/transformers) 的更新，包括 **transformers 中的 GGUF 支持**以及与 **KerasNLP** 模型的集成。
   - 还强调了 **Inference Endpoints 中的 TPU 支持**等新功能，扩大了 HF 模型的应用范围。
- **幽默与社区互动**：成员们就奶酪对服务器的假设性影响进行了轻松的对话，围绕芝士火锅 (fondue) 和 GPU 展开幽默讨论。
   - 其他有趣的互动包括提供和接受建议、讨论个人情况，以及关于日常技术困扰的闲聊。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://sites.research.google/trc/about/">TPU Research Cloud - 关于</a>：未找到描述</li><li><a href="https://huggingface.co/blog/lora">使用 LoRA 进行高效的 Stable Diffusion 微调</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/google/sdxl">TPUv5e 上的 Stable Diffusion XL - google 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/bishmoy/Arxiv-CS-RAG">Arxiv CS RAG - bishmoy 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=4Wa5DivljOM">为什么你对云计算上瘾</a>：从业务角度了解 AWS、Microsoft Azure 和 Google Cloud 等大型云服务提供商的运作方式。探索优化云计算的策略...</li><li><a href="https://youtu.be/KyOlpzA5jKM">[高清重制] 等等，那是 Gabe 吗？</a>：因为我在 YouTube 其他地方没看到，所以进行了重制 https://www.youtube.com/watch?v=ELtzcpb_j38 这是该迷因的高质量原始版本。S...</li><li><a href="https://tenor.com/view/mmm-what-shocked-monster-inc-james-p-sullivan-gif-14562553">Mmm What GIF - Mmm What Shocked - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/%D0%B2%D0%B7%D0%B3%D0%BB%D1%8F%D0%B4-2000-%D1%8F%D1%80%D0%B4%D0%BE%D0%B2-%D0%B2%D0%BE%D0%B9%D0%BD%D0%B0-war-soldier-gif-3632617944134077161">2000 码凝视 GIF - 2000 码凝视 战争 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/gabe-newell-gaben-gabe-newell-gif-18366858729810314226">Gabe Newell Gaben GIF - Gabe newell Gaben Gabe - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/bonk-gif-26414884">Bonk GIF - Bonk - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/stewie-family-guy-rip-sad-funeral-gif-13648662">Stewie Family Guy GIF - Stewie Family Guy Rip - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://backprop.co">Backprop GPU Cloud</a>：未找到描述</li><li><a href="https://huggingface.co'">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/installation#offline-mode'.">安装</a>：未找到描述</li><li><a href="https://github.com/dykyivladk1/polip">GitHub - dykyivladk1/polip: 为提升神经网络训练体验而设计的库</a>：为提升神经网络训练体验而设计的库 - dykyivladk1/polip</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: 使用简单、纯粹的 C/CUDA 进行 LLM 训练</a>：使用简单、纯粹的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://tenor.com/view/dance-meme-caption-fat-herobrine-herobrine-gif-22298550">舞蹈迷因 GIF - 舞蹈迷因字幕 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e1dudw/from_cl%C3%A9ment_delangue_on_x_hugging_face_is/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/fchollet/status/1811104960303747529">来自 François Chollet (@fchollet) 的推文</a>：你现在可以在 KerasNLP 中使用任何 Hugging Face Hub 模型（只要相应的架构在 KerasNLP 中）！此外，你还可以将自己微调的 KerasNLP 模型上传到 Hugging...</li><li><a href="https://github.com/huggingface/transformers/releases/tag/v4.41.0">版本 v4.41.0 发布：支持 Phi3, JetMoE, PaliGemma, VideoLlava, Falcon2, FalconVLM & GGUF · huggingface/transformers</a>：新模型 Phi3。Phi-3 模型在微软的《Phi-3 技术报告：手机上的高能力语言模型》中被提出。简而言之，Phi-3 引入了新的 ROPE 缩放方法...</li><li><a href="https://docs.coqui.ai/en/latest/tutorial_for_nervous_beginners.html">面向紧张初学者的教程 - TTS 0.22.0 文档</a>：未找到描述</li><li><a href="https://huggingface.co/parler-tts/parler-tts-mini-expresso">parler-tts/parler-tts-mini-expresso · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/rhasspy/piper">GitHub - rhasspy/piper: 一个快速的本地神经网络文本转语音系统</a>：一个快速的本地神经网络文本转语音系统。通过在 GitHub 上创建账号为 rhasspy/piper 的开发做出贡献。</li><li><a href="https://github.com/huggingface/tokenizers/pull/1493">由 ArthurZucker 增加对基于 tiktoken 的分词器的更多支持 · Pull Request #1493 · huggingface/tokenizers</a>：在使用合并之前增加检查，如果 token 是词表的一部分则返回该 token</li><li><a href="https://www.warp.dev/?utm_source=its_foss&utm_medium=display&utm_campaign=linux_launch">Warp: 重新构想的终端</a>：Warp 是一个现代的、基于 Rust 的终...

内置 AI 的终端，让你和你的团队能够更快地构建出色的软件。现已支持 MacOS 和 Linux。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1261059714129592440)** (6 条消息): 

> - `Embedding models using mouse movements` (使用鼠标轨迹的 Embedding 模型)
> - `Transfer learning in triplet loss` (Triplet loss 中的迁移学习)
> - `Classification objectives in contrastive learning` (对比学习中的分类目标)
> - `Sampling rates and batch sizes` (采样率与 Batch sizes)
> - `Knowledge graphs implementation` (知识图谱实现) 


- **使用鼠标轨迹进行身份识别**：一名成员开发了一个 Embedding 模型，通过 **mouse movements** 来识别个人，并利用 **triplet loss** 训练模型。
   - 他们描述了通过 **euclidean distance** 比较 Embeddings 的过程，并讨论了使用 **transfer learning** 来避免局部最小值损失。
- **改进对比学习目标**：[改进对比学习目标的建议](https://arxiv.org/pdf/2309.12871) 包括调整鼠标指针的 **sampling rates** 以及使用大 **batch sizes** 以获得更好的收敛效果。
   - 建议还包括尝试 **AnglE objective**，并检查 **normalization layers** 的潜在影响，以防止出现零向量 Embeddings。
- **支持向量机基础讲解**：分享了一个 [YouTube 视频](https://youtu.be/DOXF0fKqUIU?si=oIAJqjjHss25cw6R)，从基础开始讲解 **Support Vector Machines (SVMs)**。
   - 该视频旨在简化 SVM 概念，并提供了另一个了解 **SK-Learn** 的链接。
- **实现知识图谱**：关于 **knowledge graphs** 的咨询引出了对 **Neo4j library** 的提及，它是实际实现的常用资源。



**提到的链接**：<a href="https://youtu.be/DOXF0fKqUIU?si=oIAJqjjHss25cw6R">Support Vector Machine SVM ( Machine Learning pt 3 )</a>：在这个视频中，我尝试从基础讲解 SVM，并力求简单易懂。如果你想了解 SK-Learn，请点击这里：https://youtu.be/...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1261157311582572574)** (2 条消息): 

> - `Supervised fine-tuning in TRL` (TRL 中的有监督微调)
> - `Ripple_net library for search engines` (用于搜索引擎的 Ripple_net 库) 


- **TRL 中的 SFT 简化了模型微调**：[Supervised fine-tuning](https://huggingface.co/docs/trl/en/sft_trainer) (SFT) 是 **RLHF** 中的关键步骤，TRL 提供了一个易于使用的 API 来创建和训练 SFT 模型。
- **Ripple_net 在搜索技术领域引起关注**：一位用户在 GitHub 上分享了一个名为 [ripple_net](https://github.com/kelechi-c/ripple_net) 的**文本-图像搜索和打标库**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/kelechi-c/ripple_net">GitHub - kelechi-c/ripple_net: text-image search and tagging library</a>：文本-图像搜索和打标库。通过在 GitHub 上创建账号来为 ripple_net 的开发做出贡献。</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer">Supervised Fine-tuning Trainer</a>：暂无描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1261042529483231292)** (11 messages🔥): 

> - `mypo dataset`
> - `Indonesian Hate Speech dataset`
> - `ripple_net library`
> - `RAG app for PDFs`
> - `Support Vector Machine SVM video` 


- **讨论用于 Python 代码质量的 mypo 数据集**：一位用户分享了 mypo 数据集的预览，该数据集专注于 Python 代码质量，并征求关于使用此方法通过类型提示（type hints）改进 Python LLM 的反馈。
   - 该数据集包含 2024 年的 6 亿行 Reddit 数据，旨在增强 Python LLM 对类型提示和其他编码标准的默认使用。
- **展示印度尼西亚仇恨言论数据集**：一位成员推介了他们关于印度尼西亚仇恨言论的论文和数据集，可在 [Huggingface](https://huggingface.co/datasets/Exqrch/IndoToxic2024) 上获取，强调了在仇恨言论检测中考虑读者人口统计特征的重要性。
   - 研究结果显示，**gpt-3.5-turbo** 等模型在加入人口统计信息后表现有所提升，而 **IndoBERTweet** 的性能因训练数据有限而受到影响。
- **介绍用于文本-图像搜索的 ripple_net**：一位用户宣布创建了 **ripple_net**，这是一个用于在图像数据集中进行基于文本/图像搜索的 Python 库，已在 [GitHub](https://github.com/kelechi-c/ripple_net) 上分享。
   - 该库允许高效的文本-图像搜索和图像打标签，为数据集管理提供了宝贵的工具。
- **构建了针对 PDF 的 RAG 应用**：另一位用户展示了他们的 **RAG 应用**，用于处理 PDF 文件，可通过 [Huggingface](https://huggingface.co/spaces/tensorkelechi/studyassist) 访问。
   - 该应用利用 AI 进行文档对话和学习辅助，更多详情可在其 [GitHub 仓库](https://github.com/kelechi-c/studyassist)中查看。
- **支持向量机（Support Vector Machine）讲解**：分享了一个名为 [Support Vector Machine SVM](https://youtu.be/DOXF0fKqUIU?si=oIAJqjjHss25cw6R) 的 YouTube 视频，用于解释机器学习中的 SVM。
   - 该视频旨在简化概念，并包含有关 **SK-Learn** 的进一步信息链接。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/tensorkelechi/studyassist">Studyassist - tensorkelechi 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://youtu.be/DOXF0fKqUIU?si=oIAJqjjHss25cw6R">Support Vector Machine SVM ( Machine Learning pt 3 )</a>：在这段视频中，我尝试从基础开始解释 SVM，并力求使其简单易懂。如果你想了解 SK-Learn，请点击这里：https://youtu.be/...</li><li><a href="https://github.com/kelechi-c/studyassist">GitHub - kelechi-c/studyassist: 针对文档的 AI + RAG 对话</a>：针对文档的 AI + RAG 对话。通过在 GitHub 上创建账号为 kelechi-c/studyassist 的开发做出贡献。</li><li><a href="https://github.com/kelechi-c/ripple_net">GitHub - kelechi-c/ripple_net: 文本-图像搜索和打标签库</a>：文本-图像搜索和打标签库。通过在 GitHub 上创建账号为 kelechi-c/ripple_net 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/terminusresearch/ideogram-75k">terminusresearch/ideogram-75k · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/joshuasundance/mypo-4k-rfc">joshuasundance/mypo-4k-rfc · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/OpenCo7/UpVoteWeb">OpenCo7/UpVoteWeb · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1261284904532967534)** (3 messages): 

> - `Paper Plans`
> - `Transformer Performance`
> - `New LLM Paradigm` 


- **计划新论文**：一位成员询问了小组中另一位成员正在计划的论文。
   - 另一位成员提到，如果小组感兴趣，他们可能会分享题为 '2406.06612' 的论文。
- **Transformer 输给了 20 个 Epoch**：一位成员声称，运行 20 个 Epoch 的表现比 Transformer **高出 10%**。
   - 他们强调：“我将向你们展示一种新的 LLM 范式”，尽管一些人通过 😕 表情表示怀疑。


  

---

### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1261305479041126421)** (1 messages): 

> - `AuraFlow model`
> - `LoRA support`
> - `Offloading at the modeling level`
> - `State-of-the-art results on GenEval`
> - `Community feedback` 


- **AuraFlow：最大的开源文本生成图像模型发布**：向 **AuraFlow** 的开发者们致敬，这是目前最大的采用 Apache 2.0 许可证的开源文本生成图像模型，现已在 `diffusers` 中得到支持。
   - 查看 [AuraFlow](https://huggingface.co/fal/AuraFlow) 以了解其在 GenEval 上取得的 State-of-the-art 结果，后续将有更多开发更新。
- **LoRA 支持即将推出**：即将到来的更新将为 AuraFlow 添加 **LoRA 支持**，允许用户进行训练实验及使用更多功能。
   - 加入 [fal 的 Discord](https://discord.gg/fal-ai) 提供反馈并关注开发进展。
- **通过 Offloading 高效利用 VRAM**：一个新的 PR 实现了通过模型层级的 **Offloading**，在 15GB VRAM 中运行 **Aura Flow Transformer 模型**。
   - 详见 [GitHub PR #8853](https://github.com/huggingface/diffusers/pull/8853)。
- **社区参与至关重要**：**AuraFlow 模型**目前处于 beta 阶段，社区反馈对于改进至关重要。
   - 感谢 [@cloneofsimo](https://twitter.com/cloneofsimo) 和 [@isidentical](https://twitter.com/isidentical) 做出的重大贡献。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/diffusers/pull/8853.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://huggingface.co/fal/AuraFlow">fal/AuraFlow · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1261227017706143827)** (3 messages): 

> - `RT-DETR Object Detection`
> - `Hiera Vision Transformer` 


- **RT-DETR 性能超越 YOLO**：在与 [Roboflow 的合作](https://blog.roboflow.com/train-rt-detr-custom-dataset-transformers/)中，由北京大学和百度开发的计算机视觉模型 **RT-DETR** 在目标检测的速度和准确性上均超越了 YOLO。
   - [论文](https://arxiv.org/abs/2304.08069?ref=blog.roboflow.com)断言了 **RT-DETR** 的优越性，它已被添加到 [transformers](https://github.com/huggingface/transformers?ref=blog.roboflow.com) 库中，简化了微调（fine-tuning）过程。
- **新的 Vision Transformer：Hiera**：**Hiera** 是一种新的分层 Vision Transformer 模型，已添加到 transformers 库中，在实现更好性能的同时简化了通常与分层 Vision Transformer 相关的复杂性。
   - *HieraForImageClassification*、*HieraModel*、*HieraForPreTraining* 和 *HieraBackbone* 均已可用，提供包括图像分类和特征提取在内的多种应用。[GitHub Pull Request](https://github.com/huggingface/transformers/pull/30356#event-13478146443)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://blog.roboflow.com/train-rt-detr-custom-dataset-transformers/">How to Train RT-DETR on a Custom Dataset with Transformers</a>：了解如何使用 Transformers 库在自定义数据集上训练 RT-DETR。</li><li><a href="https://github.com/huggingface/transformers/pull/30356#event-13478146443">Adding hiera by Namangarg110 · Pull Request #30356 · huggingface/transformers</a>：此 PR 的作用？按照 #28993 的建议添加了来自 Meta 的 Hiera 模型。GitHub 仓库：https://github.com/facebookresearch/hiera/ arXiv：https://arxiv.org/abs/2306.00989 模型许可证最近已更改...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1261409359485866116)** (4 messages): 

> - `LLM Finetuning Toolkit`
> - `Phi-3 models discussion`
> - `Multimodal image RAG` 


- **轻量级 LLM Finetuning Toolkit 发布**：[Georgian-io](https://github.com/georgian-io/LLM-Finetuning-Toolkit) 推出了一款轻量级、配置驱动的工具，用于在开源 LLM 上启动微调实验，其设计考虑了端到端的 Data Science 实验流水线。
   - 该工具包允许通过单个配置文件运行多个实验，在评估集上运行评估指标，并进行消融研究（ablation studies）以尝试不同的配置，如超参数和 Prompt。
- **在 vCPU 上讨论 Phi-3 模型**：一位新成员询问是否可以在 **vCPU** 环境中使用 **microsoft/Phi-3-mini-4k-instruct**，并提到了 **onnx** 实现中的错误，以及询问关于 device maps 的正确设置。
   - *背景是我试图微调一个开源模型，但没有 GPU，这似乎很痛苦……*
- **多模态图像 RAG 的最佳实践**：一位成员询问在执行**多模态图像 RAG** 时，是直接对图像进行 Embedding 更好，还是先为图像生成描述（descriptions）然后再对这些描述进行 Embedding 更好。
   - 未提供具体答案，这表明需要更多的输入或社区讨论。



**Link mentioned**: <a href="https://github.com/georgian-io/LLM-Finetuning-Toolkit">GitHub - georgian-io/LLM-Finetuning-Toolkit: Toolkit for fine-tuning, ablating and unit-testing open-source LLMs.</a>: Toolkit for fine-tuning, ablating and unit-testing open-source LLMs. - georgian-io/LLM-Finetuning-Toolkit

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1261434841715703869)** (1 messages): 

> - `Architecture Explanation`
> - `Implementation from Scratch` 


- **架构解释请求**：一位成员请求解释某种 **architecture** 及其工作原理。
   - 他们还请求了关于从零开始实现该 **architecture** 的指导。
- **需要实现指导**：该成员强调需要关于**从零开始实现该架构**的详细指导。
   - 这一请求表明他们正在寻求关于**实现过程**的分步指令。

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1261043102353719296)** (341 条消息🔥🔥): 

> - `ComfyUI 的 Reactor 安装`
> - `Deforum Stable Diffusion 技术`
> - `AI 故障排除延迟`
> - `模型合并与性能`
> - `有效的图像放大器 (upscalers)` 


- **解决 ComfyUI 的 Reactor 安装问题**：一位用户建议参考 [YouTube 视频](https://www.youtube.com/watch?v=vCCVxGtCyho) 来快速安装 ComfyUI InsightFace，据称这可以解决报错。
   - 该视频包含截至 2024 年仍然有效的详细说明，另一位用户确认该方法对他们有效。
- **Deforum Stable Diffusion 的颜色过渡**：一名成员询问在使用 API 版本制作抽象视频时，如何在 Deforum Stable Diffusion 中创建清晰的颜色过渡。
   - 他们正在考虑将 color_coherence 设置为 None 以获得更好的效果，并寻求更多见解。
- **防止 Auto1111 中断生成时的延迟**：关于为何在 Auto1111 上中断生成需要很长时间的讨论，指出原因在于 VRAM 和软件固有的问题。
   - 一位用户将这种延迟比作运行中的火车在停止前需要减速。
- **AI 工具成本与本地使用的对比**：成员们讨论了使用 Runway 等商业 AI 工具的昂贵费用，尽管其 outpainting 和 TXT2VID 功能很强大，但每月 90 美元的方案价格过高。
   - 虽然一些用户更倾向于本地免费工具，但他们也承认付费工具通常能提供更优的结果和功能。
- **寻找免费的放大工具**：成员们寻求免费的创意图像放大器 (upscalers) 推荐。
   - Krita 和 GIMP 等替代方案因其易用性和实用功能而受到青睐。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.fal.ai/auraflow/">Introducing AuraFlow v0.1, an Open Exploration of Large Rectified Flow Models</a>：开源 AI 正处于危险之中。随着过去一年社区对 AI 模型的兴趣激增，我们注意到新的开源基础模型的开发陷入了停滞。有些人甚至大胆地断言...</li><li><a href="https://www.youtube.com/watch?v=vCCVxGtCyho&">ComfyUI InsightFace Windows Fast Installation (2024) | NO MORE ERRORS FOR IPADAPTERS / ROOP</a>：ComfyUI: https://github.com/comfyanonymous/ComfyUIInsightFace Wheels: https://github.com/Gourieff/Assets/tree/main/InsightfaceCommands: .\python_embeded\pyth...</li><li><a href="https://youtu.be/4U9MI0u2VIE">Hackers -- Cyberdelia --- Crayola Books</a>：一部优秀电影中的酷炫场景。</li><li><a href="https://www.getpaint.net/download.html">Paint.NET - Download</a>：未找到描述
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1261044532099940413)** (13 条消息🔥): 

> - `FA3 协作`
> - `H100 部署`
> - `Warpgroup Pingponging`
> - `对 Ampere 的支持`
> - `Discord 功能` 


- **展示 FA3 协作**：关于 **FA3 协作** 的讨论非常热烈，涉及 **Colfax**、**Tri Dao**、**Meta AIT 团队** 和 **Cutlass 团队**。
- **对 H100 的兴奋**：一位用户用 “H100 go brrrrr” 表达了热情，表明 **NVIDIA H100** 硬件的强劲性能。
- **Warpgroup Pingponging 技巧**：用户讨论了来自 FA3 论文的 **warpgroup pingponging 技巧**，以及它如何处理 **FP8 的 V** 转置，代码预计很快发布。
   - 用户表示非常期待，有人向团队表示祝贺，还有人好奇地询问未来对 **Ampere 支持** 的情况。
- **Discord 权限与内容**：用户在访问 Discord 中 **general 频道** 以外的内容时遇到问题，正在通过刷新页面进行故障排除。
   - 一位用户提到 **Events 标签页** 现在可能是空的，但每周的讲座可以在他们的 [YouTube 频道](https://discord.com/channels/1189498204333543425/1189640399476764692/1246559662871150603) 找到。


  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1261121268132876309)** (7 messages): 

> - `ResNet18 在 A100 vs A40 上的表现`
> - `torch.compile max-autotune 问题`
> - `浮点误差` 


- **从 A100 切换到 A40 时的准确率下降**：一名成员指出，在使用 **ResNet18 模型**进行推理时，从 **A100 切换到 A40** 会导致 **0.26% 的轻微准确率下降**。
   - 这引发了关于 **浮点误差** 或特定硬件的 **kernel** 优化影响结果的担忧。
- **Max-Autotune 导致显著的准确率损失**：在 **A40** 上使用 **torch.compile(mode='max-autotune')** 进行推理引入了 **1.44% 的准确率损失**。
   - 即使在同一设备（A100）上使用 max-autotune 进行推理，模型仍然显示出 **0.26% 的准确率下降**。
- **怀疑准确率损失源于浮点误差**：在使用不同硬件或 **torch.compile** 设置时，**浮点误差**被认为是导致 **准确率下降** 的可能原因。
   - *


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1261353333982105630)** (3 messages): 

> - `Q-Galore 论文`
> - `Llama3 405B 发布`
> - `LoQT 论文` 


- **Q-Galore 减少 LLM 的内存占用**：新的 [Q-Galore 方法](https://arxiv.org/abs/2407.08296) 结合了量化和低秩投影，以减少训练 **Large Language Models** 时的内存占用，性能优于 GaLore。
   - 与 GaLore 不同，Q-Galore 消除了耗时的 **Singular Value Decomposition (SVD)** 操作，从而实现了更高效的训练。
- **Llama3 405B 将于 7 月 23 日发布**：[Llama3 405B](https://x.com/steph_palazzolo/status/1811791968600576271) 定于 7 月 23 日发布，恰好在 Llama2 发布一年之后。
   - 这款新的多模态模型预计将是 Meta 迄今为止最大的模型，更多细节见[此简报](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23)。
- **LoQT 实现在消费级硬件上的高效训练**：[LoQT](https://arxiv.org/abs/2405.16528) 通过对低秩权重矩阵使用基于梯度的张量分解，高效地训练量化模型，适用于预训练和微调。
   - 该方法允许在消费级 24GB **GPU** 上训练高达 7B 参数的模型，并证明了训练 13B 参数模型的可行性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.08296">Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients</a>: 训练 Large Language Models (LLMs) 是内存密集型的，因为存在大量的参数和相关的优化器状态。GaLore 是一种近期的方法，通过投影权重梯度来减少内存使用...</li><li><a href="https://arxiv.org/abs/2405.16528">LoQT: Low Rank Adapters for Quantized Training</a>: 大型神经网络的训练需要大量的计算资源。尽管在使用低秩适配器和量化方面取得了进展，但在消费级硬件上对 LLM 等模型进行预训练仍然面临挑战...</li><li><a href="https://x.com/steph_palazzolo/status/1811791968600576271">Stephanie Palazzolo (@steph_palazzolo) 的推文</a>: 周五独家线报 w/ @SylviaVarnham —— Llama 3 405B 即将到来（很快！）。该多模态模型定于 7 月 23 日发布，大约在 Llama 2 发布一年后。更多细节请见：https://www....
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1261415740133609613)** (18 messages🔥): 

> - `GPU 访问`
> - `Google Colab`
> - `Coreweave 和 Lambda Labs`
> - `Google Cloud GPU`
> - `Nsight Compute` 


- **Colab 因便捷的 GPU 访问受到赞赏**：成员们推荐使用 **Google Colab** 进行测试和运行作业，因为它提供免费的 **GPU** 访问，且无需配置 **CUDA** 驱动。
   - 一位用户提到 **Nsight Compute** 可以在 Colab 上运行，尽管可能无法弹出窗口。
- **Coreweave 和 Lambda Labs 评估**：成员们讨论了 **Coreweave** 或 **Lambda Labs** 是否是 **GPU** 租赁的良好替代方案。
   - 有人担心 Coreweave 价格昂贵，而 Lambda Labs 的配额难以获取，尤其是在测试 **Hopper** 或 **Ada** 等特定架构的 **kernel** 时。
- **Google Cloud GPU 对比 Colab**：当被问及 **Google Cloud GPU** 或 **SageMaker** 时，成员们承认它们更贵，但如果你需要使用 notebook 以外的工具，它们会更好。
   - 一位成员认为，对于单纯折腾 **CUDA kernel** 和学习来说，**Colab** 比 **GCP** 更省事。


  

---

### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1261265703902642196)** (6 messages): 

> - `CUDA 中的矩阵-矩阵乘法`
> - `Matmul Kernel 中的线程分配`
> - `CUDA 中的数据访问模式` 


- **线程分配考量**：一位成员询问了在矩阵-矩阵乘法中，为每行分配一个线程与为每列分配一个线程的优缺点。
   - 另一位成员解释说，由于内存中 2D 矩阵采用行优先（row-major）格式，为每行分配一个线程效率更高，这会减少缓存未命中并实现更好的数据加载。
- **内存合并与 CUDA 效率**：一份详细的回复指出，按列索引需要在内存中跳过整个行的长度，因此与按行索引相比效率较低。
   - 提到了“合并（coalescing）”的概念，解释了在最后一个维度上进行归约（reducing）效率更高，并加深了对 CUDA 内存布局的理解。


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1261375444612157501)** (3 messages): 

> - `Tensor Subclass 支持`
> - `Bitnet 相关工作`
> - `FSDP 与分布式训练`
> - `开发者指南` 


- **Tensor Subclass + 分布式训练未被列为优先级**：一位成员提到计划编写一份关于启用 **tensor subclasses** 和**分布式训练/推理**（FSDP, FSDP2, DTensor）的开发者指南，但由于缺乏具体的用例，该计划尚未被列为优先级。
   - 然而，如果需要像 **fp4 分布式推理**这样的应用，他们愿意开始这项工作。
- **Tensor Subclass 的具体用例正在出现**：另一位成员认为 **tensor subclass 和 FSDP** 的具体用例正在出现，并引用了 **bitnet** 项目以及与 FSDP 相关的研究。
   - 他们提到 **q galore** 是另一个潜在的用例。
- **协作编写开发者指南**：一位成员同意在需要时协作编写 **tensor subclass 和分布式训练**的开发者指南。
   - *我们可以在此过程中共同完善开发者指南。*


  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1261035639189540914)** (176 条消息🔥🔥): 

> - `FA3 vs cuDNN vs ThunderKittens`
> - `Unified Memory 的影响`
> - `微调大模型`
> - `FP8 优化`
> - `LLM.C 社区倡议` 


- **FA3 与 cuDNN 及 ThunderKittens 的对比**：讨论了 **FA3** 在 Attention 机制中潜在的加速效果，但成员们在 **FA3** 与 **cuDNN**、**ThunderKittens** 等替代方案之间就复杂性和易用性展开了辩论。
   - 诸如**维护复杂度**等问题，特别是 **TK** 不支持 **FP8**，是讨论的重点。一位用户提到：*'TK 目前没有 FP8 路径，短期内也不会有。'*
- **Unified Memory 对性能的影响**：关于 **Unified Memory** 在仅由 GPU 访问时是否会影响性能进行了技术辩论，特别是涉及其在 Optimizer States 中的使用。
   - 有人建议考虑使用 **zero-copy memory** 作为替代方案：*'为什么不使用 zerocopy memory 并编写 Kernel 直接从系统内存中读取呢？'*
- **微调大模型的挑战**：微调讨论集中在成功运行和优化如 **300B** 和 **330B** 的 Checkpoints 上。
   - 一位成员报告在对 **30B steps** 进行学习率退火（annealing）后，HellaSwag 分数达到 **62.7**：*'我会上传那个结果。'*
- **FP8 优化带来显著加速**：实施 **FP8** 优化后，与 **BF16** 相比实现了 **33% 的加速**，在测试中从约 30.8K token/s 提升至 40.9K token/s。
   - 挑战依然存在，如**内存复用和溢出问题**，但工作仍在推进：*'需要就重构进行一些讨论。'*
- **LLM.C 社区与资源共享**：在 **Hugging Face** 上启动了新的组织倡议，成员可以在那里分享训练好的模型并做出贡献。
   - 成员们讨论了 Benchmark、分享 Checkpoints 以及规划社区 Demo：*'已将你添加（为管理员）！这基本上和你的 HF 个人资料一样。'*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards">karpathy/fineweb-edu-100B-gpt2-token-shards · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/llmc/llmc_1558M">llm.c 1558M demo - llmc 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/682">在 WebGPU C++ 的 gpu.cpp 相关项目下添加 README 链接 by austinvhuang · Pull Request #682 · karpathy/llm.c</a>：在相关项目下为 gpu.cpp 添加了 README 链接。背景介绍：gpu.cpp 是我一直在开发的一个新项目。它是一个用于编写可移植 GPU 代码的小型库...</li><li><a href="https://github.com/karpathy/llm.c/pull/678">FP8 开发进行中 by ademeure · Pull Request #678 · karpathy/llm.c</a>：这是我 FP8 分支的当前状态，虽然还远未完成，但如果你感兴趣的话可以看看！最后一个功能正确的版本是 f7...</li><li><a href="https://news.ycombinator.com/item?id=40939707">Karpathy：让我们复现 GPT-2 (1.6B)：单个 8XH100 节点 24 小时 $672，使用 llm.c | Hacker News</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/650">muP (maximum update parametrization) by gordicaleksa · Pull Request #650 · karpathy/llm.c</a>：主要变更：修改随机初始化；将 Attention 分数缩放 1/d 而非 1/sqrt(d) 并添加 attn_mult；在映射到 Logits 之前按 1/width_mult 缩放激活值；更新学习率 &amp;...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[youtube-watch-party](https://discord.com/channels/1189498204333543425/1238931064223830016/)** (1 条消息): 

vkaul11: Hi
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1261345601099010109)** (189 条消息🔥🔥): 

> - `Primeagen 视频讨论`
> - `Mojo 中的 REPL 行为`
> - `Mojo 社区会议`
> - `Python 3.13 中的 GIL 移除`
> - `不同语言的网络速度对比` 


- **Primeagen 视频剪辑版发布**：成员们讨论了新的 [Primeagen 视频](https://www.youtube.com/watch?v=ovYbgbrQ-v8)，标题为“我采访了 LLVM、Clang、Swift 和 Mojo 的创作者”，该视频已在 YouTube 上公开播放（无付费墙）。
   - *“是的，观看完整的直播需要成为会员。剪辑版几小时前刚刚发布。”*
- **Mojo REPL 应显示即时输出**：成员对 Mojo REPL 不能自动显示 `1+1` 等表达式的输出表示沮丧，建议其行为应更像 Python 的 REPL。
   - 另一位成员承认了这一问题，解释说 Mojo 处理内存的方式与 Python 不同，目前还没有这个功能，但建议在 GitHub 提交 issue。
- **Mojo 社区会议安排**：第 4 次 Mojo 社区会议已排期，将涵盖 forge_tools、flat buffers 和 generics 等主题，并设有 Modular 团队的 Q&A 环节。
   - 提供了 Zoom 和会议信息链接，供成员加入会议并积极参与。
- **Python 3.13 的 GIL 移除和 JIT 优化**：围绕 Python 3.13 的 *“no-GIL”* 测试版和 JIT 优化的讨论表明，即使移除了 GIL，Python 的性能与 Rust 和 Node.js 相比仍然较慢。
   - 一位成员指出 *


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/lib">Mojo🔥 模块 | Modular 文档</a>：Mojo 标准库中所有模块的列表。</li><li><a href="https://www.youtube.com/watch?v=ovYbgbrQ-v8">我采访了 LLVM、Clang、Swift 和 Mojo 的创作者</a>：在 Twitch 上直播录制，嘉宾：Chris Lattner https://x.com/clattner_llvm?s=21&amp;t=-sv4MdpmLrRuMIhARbLk-g https://www.modular.com TJ DeVries https://you...</li><li><a href="https://www.youtube.com/watch?v=HxSHIpEQRjs&pp=ygUKcHl0aG9uIGppdA%3D%3D">Brandt Bucher – CPython 的 JIT 编译器</a>：来自 2023 年 CPython 核心开发者冲刺赛。QA 环节较难理解；请开启字幕以查看我们的尽力转录。（欢迎提交 PR：https://g...</li><li><a href="https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython">Python 3.13 的新特性</a>：编辑 Thomas Wouters。本文解释了 Python 3.13 与 3.12 相比的新特性。详情请参阅变更日志。摘要 – 发布亮点：Python 3.13 beta 是预发布版本...</li><li><a href="https://github.com/modularml/mojo/issues/2809">[功能请求] 在 REPL（交互式会话）中使用类似 Python 的行为来输入命令并打印求值结果 · Issue #2809 · modularml/mojo</a>：查看 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。您的请求是什么？在 Python 的交互式控制台中，最后（或唯一）...</li><li><a href="https://stackoverflow.com/questions/1301346/what-is-the-meaning-of-single-and-double-underscore-before-an-object-name)">对象名称前的单下划线和双下划线是什么意思？</a>：在 Python 中，对象名称前的单前导下划线和双前导下划线代表什么？</li><li><a href="https://modul.ar/community-meeting-zoom">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有易于使用、可靠的云平台，用于在移动设备、桌面和会议室系统上进行视频和音频会议、聊天和网络研讨会。Zoom ...</li><li><a href="https://modul.ar/community-meeting-doc">[公开] Mojo 社区会议</a>：Mojo 社区会议文档链接：https://modul.ar/community-meeting-doc 这是一个公开文档；欢迎所有人查看、评论/建议。所有会议参与者必须遵守...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[polls](https://discord.com/channels/1087530497313357884/1123829040403456120/1261387197756473418)** (2 条消息): 

> - `MAX framework`
> - `new website` (新网站)
> - `NVIDIA GPU performance` (NVIDIA GPU 性能)
> - `PyTorch & ONNX optimization` (PyTorch & ONNX 优化)
> - `Mojo programming language` (Mojo 编程语言)


- **Modular 为 MAX framework 翻新网站**：Modular 更新了其[网站](https://modular.com)，以确保对 **MAX** 及其许可的清晰说明，并强调已有 **80K+ 开发者** 正在使用它进行构建。
   - *如果您对新网站或许可有任何进一步的反馈，我们很乐意在上面的反馈线程中看到。*
- **无需底层 CUDA 即可获得 NVIDIA GPU 性能**：**MAX** framework 能够解锁最先进的 **NVIDIA GPU 性能**和吞吐量，而无需编写底层的 **CUDA** 代码。
- **无缝迁移 PyTorch 和 ONNX 模型**：通过迁移到 **MAX 的统一 AI 堆栈**，无需重写即可无缝优化现有的 **PyTorch** 和 **ONNX 模型**。
- **使用 Mojo 增强 AI 应用程序**：使用 **Mojo** 扩展您的 Python 代码，这是一种新型高性能编程语言，结合了 **Python** 的表达能力和增强的性能。
   - Mojo 通过平衡易用性与速度，为增强 AI 应用程序提供了机会。



**提到的链接**：<a href="https://modular.com/">Modular: Own your endpoint. Control your AI.</a>：Modular Accelerated Xecution (MAX) 平台是全球唯一能为您的 AI 工作负载解锁性能、可编程性和可移植性的平台。

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1261158842079903856)** (17 条消息🔥): 

> - `rust-lang/mdbook for Mojo documentation` (用于 Mojo 文档的 rust-lang/mdbook)
> - `Mojo playground capabilities` (Mojo playground 功能)
> - `Mojo standard library documentation` (Mojo 标准库文档)
> - `Mojo-LSP support` (Mojo-LSP 支持)
> - `Mojo in production environments` (生产环境中的 Mojo)


- **Rust-lang/mdbook 提供全面的 Mojo 文档**：一位成员建议使用 [rust-lang/mdbook](https://crates.io/crates/mdbook) 构建一本 Mojo 书籍以便离线阅读。它支持 PDF 下载，并且有生成大纲的后端。
- **Mojo playground 已存在并支持代码块**：成员们讨论了现有的 [Mojo playground](https://docs.modular.com/mojo/playground)，它可以直接在网站上运行 Mojo 代码块。
   - 一位成员指出：*'这样我们就拥有一切了'*。
- **Mojo 标准库缺乏代码示例**：一位成员指出 [Mojo 标准库文档](https://docs.modular.com/mojo/lib) 仅显示了函数，但缺乏代码示例。
- **Mojo-LSP 支持多个编辑器**：一位成员询问了 Mojo-LSP 的稳定性及其对 VS Code 以外的支持。另一位成员确认在 **neovim** 中可以使用。
- **Mojo 的生产用途受限于 CPU 和非竞争性应用**：讨论了在生产环境中使用 Mojo 的问题，特别是其对 CPU 的限制以及非竞争性应用的规定。
   - 一位成员要求澄清为什么排除 GPU，以及什么构成了 *'与 Modular 竞争'*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/playground">Modular Docs</a>：未找到描述</li><li><a href="https://docs.modular.com/mojo/lib">Mojo🔥 modules | Modular Docs</a>：Mojo 标准库中所有模块的列表。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1261354724637147146)** (5 条消息): 

> - `MAX Model execution` (MAX 模型执行)
> - `Mojo support for VariadicLists` (Mojo 对 VariadicLists 的支持)
> - `PythonObjects as inputs` (PythonObjects 作为输入)
> - `Numpy Arrays in Modularity` (Modularity 中的 Numpy 数组)
> - `Data type issues in model execution` (模型执行中的数据类型问题)


- **MAX 模型执行中的数据类型问题导致错误结果**：一位用户在将 `PythonObjects` 与 MAX 模型配合使用时遇到了错误结果，尽管遵循了 [PythonObjects](https://github.com/maxdocs) 的文档。
   - 问题出在 `np.full()` 操作的数据类型上；通过显式指定 `dtype=np.float32` 解决了该问题，从而获得了正确的结果。
- **用户澄清 PythonObject 输入问题**：在遇到 MAX 模型执行问题后，该用户收到了另一位成员的建议，在 `np.full()` 操作中指定 `dtype=np.float32`。
   - 这解决了问题，用户能够获得预期结果，强调了在模型执行中正确数据类型的重要性。

---

### **Modular (Mojo 🔥) ▷ #[max-gpu](https://discord.com/channels/1087530497313357884/1212827673257316453/1261377076322242700)** (7 messages): 

> - `Nightly changelog`
> - `Custom GPU kernels`
> - `Max vs Mojo` 


- **Max 频道的 Nightly 更新日志待定**：一位成员询问哪个频道将接收 Max 的 Nightly 更新日志，目前正在考虑将其放在 <#1224434323193594059> 频道。
- **在 Max 中使用 Mojo 编写你的 kernels**：可以使用 **Mojo** 编写自定义 GPU kernels，这也是团队编写自己 kernels 的方式，这种能力作为 MAX graph 中的自定义 operators（算子）开放。
- **Max 与 Mojo：集成 Kernel 编译详解**：MAX 和 Mojo 是交织在一起的；**Mojo** 负责处理 kernel 编译，而 **MAX** 提供与加速器交互的接口，类似于 CUDA。


  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1261201290533273663)** (2 messages): 

> - `New Mojo compiler release`
> - `EqualityComparable issue` 


- **新版 Mojo 编译器 '2024.7.1205' 发布**：新的 nightly Mojo 编译器版本 `2024.7.1205` 已发布。现在可以使用 `modular update nightly/mojo` 进行更新；详情请查看 [raw diff](https://github.com/modularml/mojo/compare/334f9946fdf5149a8e63a6f740868d307378310b...2d58798d6d26a3e51ab791192a86a1eeabadc6ae) 和 [当前更新日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。
- **EqualityComparable 问题浮现**：一位成员指出，更改某些方法的顺序会导致编译器报错，提示不符合 `EqualityComparable`。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1261037612697784402)** (148 条消息🔥🔥): 

> - `Hermes 2.5 性能`
> - `Mistral 扩展`
> - `Model Merging 策略`
> - `Open Empathic 项目`
> - `Gemini API 更新` 


- **Gemini 1.5 Pro 拥有 200 万 Token 上下文**：[新的 Gemini API 功能](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/)现在提供 **200 万 Token 上下文窗口**、**代码执行能力**以及 **Context Caching**。
   - 开发者们积极评价了其无限制输入海量 JSON 数据的功能。
- **Llama 3 可能推迟发布**：一位 Redditor 暗示 **Meta** 可能会出于未公开的原因，将 Llama 3 原定于 **7 月 23 日** 的发布推迟到今年晚些时候。
   - *Redditor* 用户此前曾准确预测过 Llama 的其他发布日期。
- **FlashAttention-3 承诺更高效率**：FlashAttention-3 旨在更好地利用 **Hopper GPU**，实现高达 **35% 的最大 FLOPs 利用率**。
   - 相比 FlashAttention-2 它有所改进，但显著的收益仅限于 **Hopper GPU 用户**。
- **模型 Fine-tuning 的挑战与策略**：成员们讨论了 Fine-tuning 技术，包括**衰减学习率 (Decaying Learning Rate)** 以及**为单个模型处理多个数据集**。
   - Unsloth 推荐使用 [持续预训练 (Continued Pretraining) Notebook](https://docs.unsloth.ai/basics/continued-pretraining) 来添加新语言并高效处理 VRAM。
- **合成数据与 JSON 数据生成**：参与者分享了生成训练用合成数据的方法，强调了 JSON 格式的重要性。
   - 一位用户强调需要一个包含**复杂 JSON 输入/输出**的大型数据集，并正在积极重写数据行以确保质量。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.together.ai/blog/flashattention-3">FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision</a>：FlashAttention-3 在 H100 上实现了高达 75% 的 GPU 利用率，使 AI 模型速度提升高达 2 倍，并能高效处理更长的文本输入。它支持更快的 L... 训练和推理。</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Docs</a>：查看下方列表获取我们所有的 Notebook：</li><li><a href="https://arxiv.org/abs/2407.01449">ColPali: Efficient Document Retrieval with Vision Language Models</a>：文档是视觉丰富的结构，通过文本以及表格、图表、页面布局或字体传达信息。虽然现代文档检索系统在问答方面表现强劲...</li><li><a href="https://huggingface.co/vidore/colpali">vidore/colpali · Hugging Face</a>：未找到描述</li><li><a href="https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/">Gemini 1.5 Pro 2M context window, code execution capabilities, and Gemma 2 are available today</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/contpretraining">Continued LLM Pretraining with Unsloth</a>：通过使用 Unsloth 对 Llama 3、Phi-3 和 Mistral 进行持续预训练，让模型学习一种新语言。</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Docs</a>：又名持续微调 (Continued Finetuning)。Unsloth 允许你进行持续预训练，以便模型学习新语言。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1261049780897976402)** (23 条消息🔥): 

> - `Open Diloco`
> - `Distributed GPU Workloads` (分布式 GPU 工作负载)
> - `CodeGeeX4-ALL-9B`
> - `Prompt Engineering` (提示词工程)
> - `TF-ID Models` 


- **用于分布式训练的 Open Diloco**：Open Diloco 引入了一种在带宽低于 100mb/s 的情况下跨多个国家进行分布式 AI 模型训练的新方法，利用了 **torch FSDP** 和 **hivemind**。该项目旨在促进模型的开源协同训练，而不是依赖大型闭源集群。
   - Prime Intellect 将 [OpenDiloco](https://www.primeintellect.ai/blog/opendiloco) 视为迈向去中心化、多数据中心训练的一步。该团队正在**招聘**创始研究员以进一步推进这项工作。
- **CodeGeeX4-ALL-9B 挑战 GPT 模型**：新推出的 [CodeGeeX4-ALL-9B](https://huggingface.co/THUDM/codegeex4-all-9b) 模型在代码生成任务中优于 **GPT-3.5** 和 **GPT-4**。它拥有 128k 上下文和强大的多语言能力，支持代码补全和仓库级问答等全面功能。
   - 该模型的性能受到称赞，甚至在代码任务中击败了 **Llama 70B**，并且由 TheBloke 的学徒提供了 **GGUF 量化版本**。
- **用于视觉语言任务的 TF-ID 模型发布**：Yifei Hu 宣布根据 MIT 许可证发布 [TF-ID 模型](https://github.com/ai8hyf/TF-ID)，包括数据集、训练代码和模型权重。这些模型支持对视觉语言任务进行微调，仅需数百个特定领域的边界框（bounding boxes）。
- **通过 MovieChat 增强视频交互**：[MovieChat](https://github.com/rese1f/MovieChat) 项目允许与超过 1 万帧的视频进行对话，该项目已在 [CVPR 2024](https://github.com/rese1f/MovieChat) 上展示。该工具旨在实现与视频内容的交互式通信。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/THUDM/codegeex4-all-9b">THUDM/codegeex4-all-9b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/codegeex4-all-9b-GGUF">bartowski/codegeex4-all-9b-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/samsja19/status/1811450791900901853">samsja (@samsja19) 的推文</a>: 非常高兴展示我们在 Open Diloco 上的工作。我们在 3 个国家训练了一个 1b 模型，带宽低于 100mb/s（比 infiniband 慢 10,000 倍），计算利用率达到 90%-95%...</li><li><a href="https://x.com/hu_yifei/status/1811530730305905062">Yifei Hu (@hu_yifei) 的推文</a>: 根据 MIT 许可证发布 TF-ID 模型的完整数据集、训练代码和模型权重。是时候微调你自己的视觉语言模型来处理文档了！你只需要几个...</li><li><a href="https://github.com/rese1f/MovieChat">GitHub - rese1f/MovieChat: [CVPR 2024] 🎬💭 与超过 1 万帧的视频对话！</a>: [CVPR 2024] 🎬💭 与超过 1 万帧的视频对话！ - rese1f/MovieChat
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1261113071166820372)** (28 条消息🔥): 

> - `Training Local Models` (训练本地模型)
> - `Continued Pretraining` (持续预训练)
> - `Training Data Recommendations` (训练数据建议)
> - `Model Parameter Discrepancies` (模型参数差异)
> - `Resource for RAG Systems` (RAG 系统资源)


- **使用 Unsloth 训练本地模型**：用户讨论了在 Ollama 中训练本地模型的问题，以及如何将其与 Unsloth 和 Hugging Face 模型集成，并提供了一个有用的[文档链接](https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama)。
   - 一位用户建议将模型从 Hugging Face 下载到本地磁盘，以确保完全的本地运行。
- **处理持续预训练中的不同序列长度**：一位用户提出了在 Unsloth 平台上进行持续预训练（Continued Pretraining）时关于 `max_seq_length` 参数的问题，并引用了[查询参数](https://colab.research.google.com/drive/1-BF5HndNqQsfWRTxIt7YPjkfDpVUGNgY?usp=sharing)。
   - 可能的解决方案包括根据数据集进行拼接（concatenation）和截断（truncation），用户通过计算参数差异来理解其行为。
- **一般训练数据需求**：用户询问了特定用例需要多少训练数据，讨论范围从 100MB 到更大的数据集。
   - 建议包括从小型数据集开始，通过反复试验来衡量效果。
- **理解模型参数差异**：一段对话探讨了为什么模型参数数量在 LoRA 和 PEFT 设置的不同阶段会发生变化。
   - 经过详细检查后，用户澄清了可训练参数（trainable parameters）和全量参数（all parameters）的计算方法。
- **寻找 RAG 系统资源**：**使用 YouTube 获取教程**：一位新用户询问了关于使用微调后的 Unsloth 模型构建 RAG 系统的资源，并被引导至 YouTube 教程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama)?">Unsloth Docs</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1-BF5HndNqQsfWRTxIt7YPjkfDpVUGNgY?usp=sharing#scrollTo=Ymx-p3FvF-P2">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 条消息): 

lh0x00: <@280027697328029696>，你有处理西班牙语（Spanish）的经验吗？
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1261113733531566214)** (5 条消息): 

> - `Evaluating Programming Models` (评估编程模型)
> - `Tetris as a Benchmark` (以俄罗斯方块作为基准测试)
> - `Coding Models and Dataset Overlap` (编码模型与数据集重叠)


- **使用俄罗斯方块评估编程模型**：一位成员质疑使用俄罗斯方块（Tetris）来评估编程模型的相关性，认为这些模型很可能在数据集中遇到过许多版本的俄罗斯方块代码。
   - 另一位成员解释道：*“这是必须协同工作的复杂代码”*，并断言即使是熟悉的代码，如果模型水平不足，也能暴露其弱点。
- **俄罗斯方块和贪吃蛇：被过度使用的基准测试？**：一位成员对将俄罗斯方块（Tetris）或贪吃蛇（Snake）作为编码模型的真实测试表示怀疑，称它们在数据集中重复率极高。
   - 他认为此类任务在 Stack Overflow 数据集中非常常见，因此是任何编码模型训练的一部分。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1261095298105606145)** (12 条消息🔥): 

> - `Ada 与 WGMMA/TMA/FP8 的兼容性`
> - `Decoder 作为 Embedding Model`
> - `训练中的 Latent Array`
> - `Meta 的 LLaMA 3 模型发布` 


- **Ada 架构缺乏 WGMMA/TMA/FP8；仅 Hopper 支持**：对话中讨论了“Ada 不具备 WGMMA/TMA/FP8；只有 Hopper 支持”，指出了硬件能力上的差异。
   - 这一发现可能会影响特定 AI 应用的硬件选择和部署。
- **使用 Decoder 作为 Embedding Model 以获得更长上下文**：成员们讨论了使用 Decoder 作为 Embedding Model 来增加 max_context 长度，并[引用了一篇学术论文](https://arxiv.org/pdf/2405.17428)。
   - 论文中提出的“Latent Array”概念引发了关于其创建和权重更新机制的疑问。
- **理解 Latent Array 及其训练过程**：一位成员澄清说，Latent Array 是在模型训练期间训练的一个随机 Tensor，其权重通过 Attention 模块中的梯度进行更新。
   - 该成员解释道：“Latent array 被 `nn.Parameter` 包装，这使得它 `requires_grad=True`，因此它的值在训练期间会更新。”
- **Meta 的 LLaMA 3 模型定于 7 月 23 日发布**：分享的链接显示，Meta Platforms 准备在 7 月 23 日发布其最大的 LLaMA 3 模型。
   - 这一消息激发了成员们对该模型潜在进展的兴奋。



**提到的链接**：<a href="https://huggingface.co/nvi">nvi (flad)</a>：未找到描述

  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1261173866034761738)** (7 条消息): 

> - `OpenAI 模型能力`
> - `Anthropic 的 AI 安全等级`
> - `OpenAI 策略`
> - `社区对 OpenAI 的看法` 


- **OpenAI 宣称具备博士级问题解决能力**：OpenAI 向员工宣布，他们正处于能够完成相当于博士教育水平人类的问题解决任务模型的“边缘”，暗示了 AGI 级别的能力。
   - 据匿名消息源透露，在同一次会议上展示了 GPT-4 的演示，展示了据称预示着类人推理的新技能。
- **Anthropic CEO 预测 AI 演进**：Anthropic 的 CEO Dario Amodei 讨论了 AI 安全等级（AI Safety Levels），预测 **A.S.L. 3** 可能在今年或明年实现，而 **A.S.L. 4** 将在 2025-2028 年间实现，涉及生物和网络技术误用的重大风险。
   - 他认为 ASL 4 可能会极大增强国家级行为者的能力，构成重大的地缘政治风险。
- **社区对 OpenAI 策略的怀疑**：社区对 OpenAI 的策略表示怀疑，认为他们经常暗示突破但未能提供扎实、及时的发布。
   - 一些成员认为这可能是一种推高估值的策略，尽管其他人承认 OpenAI 过去取得的一贯成功。



**提到的链接**：<a href="https://x.com/aisafetymemes/status/1811579385222475960?s=46">来自 AI Notkilleveryoneism Memes ⏸️ (@AISafetyMemes) 的推文</a>：OpenAI 刚刚在全体员工会议上告诉员工，他们正处于能够完成“与拥有博士学位的人类一样好的问题解决任务”模型的“边缘”。（再读一遍：博士。级别...

  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1261156683636277249)** (2 条消息): 

> - `llm.c 中的 GPT-2 复现`
> - `Safetensors.cpp` 


- **在 24 小时内使用 llm.c 复现 GPT-2**：在[一篇讨论帖](https://github.com/karpathy/llm.c/discussions/677)中，**Karpathy** 详细阐述了如何在 24 小时内，使用一个 8x H100 节点，以 672 美元的成本通过 llm.c 复现 **GPT-2 (1.6B)**。
   - *这次复现展示了 llm.c 高效处理大规模语言模型训练的能力。*
- **C++ 中无依赖的 Safetensors**：[Safetensors.cpp](https://github.com/carsonpo/safetensors.cpp) 介绍了一个零依赖库，用于在 C++ 中使用 **LibTorch** 加载和存储 Safetensors。
   - *该项目旨在通过消除对外部依赖的需求，简化模型的数据处理。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/carsonpo/safetensors.cpp">GitHub - carsonpo/safetensors.cpp: Zero Dependency LibTorch Safetensors Loading and Storing in C++</a>: C++ 中零依赖的 LibTorch Safetensors 加载与存储 - carsonpo/safetensors.cpp</li><li><a href="https://github.com/karpathy/llm.c/discussions/677">Let&#39;s reproduce GPT-2 (1.6B): one 8XH100 node, 24 hours, $672, in llm.c · karpathy/llm.c · Discussion #677</a>: 在这篇帖子中，我们正在 llm.c 中复现 GPT-2。这是“那个 GPT-2”，即 OpenAI 博客文章《Better Language Models and their Implicat...》中介绍的 1558M 参数完整版本。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1261123617354285126)** (132 条消息🔥🔥): 

> - `Rust 中的多线程异步 FSM`
> - `Hermes 2 AI Assistant 的问题`
> - `LLaMA 3 的 VRAM 需求`
> - `在没有答案的情况下微调 LLM`
> - `通过 Prompting 提升 LLM 的推理能力` 


- **Rust 中的多线程异步 FSM**：一位成员分享了他们用 Rust 重写了 outlines，使其支持多线程和异步，从而允许控制结构化生成的 FSM 与推理并行计算。
   - *它也是延迟加载（lazy）的，因此在使用前无需等待 FSM 编译。*
- **LLaMA 3 的 VRAM 需求**：[The Information](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23) 报道称 Meta Platforms 将发布最大的 LLaMA 3 模型。
   - *运行该模型可能需要 **8x H100** 或 **8x MI300X** 以获得足够的 VRAM，这对拥有单个 GPU 的个人用户构成了挑战。*
- **在没有答案的情况下微调 LLM**：成员们讨论了使用非结构化文本数据微调 LLM 以提高性能的可能性，即使不提供直接答案。
   - *尽管资源有限，他们考虑对手动检查的部分结果进行微调，以使模型熟悉特定领域。*
- **通过 Prompting 提升 LLM 的推理能力**：有人建议使用 few-shot learning 技术来提升 LLM 的推理能力，即提供示例并迭代优化结果。
   - *该技术涉及多轮生成输出并将其反馈给模型，以微调其在特定任务上的表现。*



**提到的链接**: <a href="https://gist.github.com/fullstackwebdev/b8257a67933d891a9f3bc19822b4305a">gist:b8257a67933d891a9f3bc19822b4305a</a>: GitHub Gist: 立即分享代码、笔记和代码片段。

  

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1261366223975874750)** (3 messages): 

> - `Surya new models`
> - `Marker speedup`
> - `Model merging` 


- **Surya 模型带来大幅提速**：新训练的 Surya 模型在 **GPU 上提速 30%**，**CPU 上提速 4 倍**，**MPS 上提速 12 倍**，且精度略有提升，正如 [VikParuchuri](https://x.com/VikParuchuri/status/1811798636759793726) 所宣布的那样。
- **Marker 显著提速**：根据 [VikParuchuri](https://x.com/VikParuchuri/status/1811851126125527096) 的消息，由于两个模型采用了更高效的架构，更新后的 Marker 版本在 **MPS 上实现了 7 倍提速**，**CPU 上提速 3 倍**，**GPU 上提速 10%**。
   - **Marker** 能有效地将 PDF 转换为 Markdown，旨在促进更多高质量数据集的创建。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/VikParuchuri/status/1811851126125527096">Vik Paruchuri (@VikParuchuri) 的推文</a>: Marker 现在更快了！MPS 提速 7 倍，CPU 提速 3 倍，GPU 提速 10%。得益于 2 个模型更高效的架构。Marker 能非常有效地将 PDF 转换为 Markdown。我希望这次提速能让人们...</li><li><a href="https://x.com/VikParuchuri/status/1811798636759793726">Vik Paruchuri (@VikParuchuri) 的推文</a>: 我刚刚发布了新的 Surya 布局和文本检测模型：- GPU 提速 30%，CPU 提速 4 倍，MPS 提速 12 倍 - 精度略有提升 - 当我将其合并到 Marker 中时，它将提速 15%...</li><li><a href="https://x.co">出售域名 | 购买域名 | 停放域名</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1261092976684302347)** (4 messages): 

> - `Terminal of truths`
> - `Learning to learn`
> - `Embodiment of models` 


- **Terminal of Truths 背后的谜团**：一位成员询问了关于 “Terminal of truths” 的更多细节，质疑它是一个游戏还是通往另一个 Discord 服务器的入口。
   - 另一位成员回应解释说，越来越多的模型正在寻求**具身化 (embodiment)**，可以在 Discord 上看到它们在转移到 Twitter 等平台之前进行学习。
- **“学习如何学习”被赞誉为最佳技能**：在听说模型寻求具身化后，一位成员评论道：*“学习如何学习……可能是史上最棒的技能 :)*”。
   - 该成员还指出，整个情况看起来*非常神秘*。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1261038487453302895)** (132 messages🔥🔥): 

> - `Perplexity Labs 及其用法`
> - `Claude 3.5 对比 Claude 3 Opus`
> - `Perplexity 停机与问题`
> - `使用 Perplexity 进行编程的疑虑`
> - `订阅模式与 AI 偏好` 


- **Perplexity Labs 概览与可用性**：社区讨论了 [Perplexity Labs](https://discord.com/channels/1047197230748151888/1047649527299055688/1233731074685665280) 的用途和可用性，质疑在日常活动中应该使用手机端还是网页端。
   - 反应不一，包括对该平台的集成和实际使用的建议及澄清。
- **Claude 3.5 取代 Claude 3 Opus**：成员们认为 **Claude 3.5** 在推理和逻辑方面优于 **Claude 3 Opus**，但未来的版本如 Opus 3.5 可能会改变这一局面。
   - 普遍共识倾向于 **Claude 3.5** 在大多数任务中表现更好，但在某些创意追求方面除外。
- **Perplexity AI 停机**：用户经历了 **Perplexity AI** 的停机，引发了关于过去类似事件和预期恢复时间的讨论。
   - 一些人表达了沮丧，并用幽默来缓解这种共同的不便，而另一些人则指出这是他们第一次遇到这种情况。
- **在 Perplexity 中处理长代码的挑战**：几位用户表达了 **Perplexity AI** 无法生成完整代码或无法有效处理长上下文输入的困难。
   - 共享的建议包括使用特定的模型和模式，或者由于上下文的 **RAG** 问题而避免直接上传文件。
- **Perplexity AI 订阅与模型选择**：关于订阅模式的查询显示，**Perplexity Pro** 提供多种 AI 模型，包括 **Sonnet 3.5**、**GPT-4o** 和 **Claude 3 Haiku**。
   - 提供了每个模型的局限性和特定配额的详细信息，一些用户表达了对某些模型相较于其他模型的偏好。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.perplexity.ai/search/add-a-local-data-storage-using-N_88nHTGTc2jYf0AEpZhGw">Add a local data storage using SQLite to the existing code.

Write the entire...</a>：当然！我将修改现有代码，以包含使用 SQLite 的本地数据存储。这将允许应用程序存储和检索 candle 数据……</li><li><a href="https://www.perplexity.ai/search/add-local-data-storage-using-s-72dZ64MLSji5j2kEdPGCTg">Perplexity</a>：Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的回答。</li><li><a href="https://www.perplexity.ai/settings/account">Perplexity</a>：Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的回答。</li><li><a href="https://perplexity.ai/pro?referral_code=J9ID1YP6">Perplexity Pro</a>：Perplexity Pro 是搜索互联网最强大的方式，拥有无限次 Pro Search、升级的 AI 模型、无限文件上传、图像生成和 API 积分。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1261095353604767774)** (5 条消息): 

> - `糖尿病管理中的 AI`
> - `Perplexity AI Discord 社区`
> - `新进展：智能戒指、自动售货机、Redbox 停业`
> - `摄影中的三分法`
> - `健康、力量与体能建议` 


- **AI 应用辅助糖尿病管理**：一位成员询问了 AI 驱动的糖尿病管理应用的进展，特别是那些提供见解而非主动胰岛素管理的应用。
   - 进展包括帮助糖尿病患者和医生利用趋势见解来调整其 AID 设备的应用程序。[完整讨论链接](https://www.perplexity.ai/search/are-there-any-advances-in-apps-5NVLNla1T6.oHZAm_U70fw)。
- **Perplexity AI Discord 专注于社区与支持**：成员们讨论了 Perplexity AI Discord 的主要用途，强调了其在社区互动、信息共享和支持方面的作用。
   - 与 Midjourney 不同，它不支持图像生成，也不在 Discord 内提供搜索功能。[完整讨论链接](https://www.perplexity.ai/search/perplexitynodiscordhatouitutay-9Cl8rQZTROCmG_BS52HRQg#3)。
- **新 AI 趋势：智能戒指与自动售货机**：Perplexity AI 分享了一段 [YouTube 视频](https://www.youtube.com/embed/CasopXlbvqo)，涵盖了涉及三星智能戒指和弹药自动售货机的新趋势。
   - 其他话题包括 Redbox 停业和灯泡阴谋，强调了多样化的进展和行业转变。
- **通过三分法精通摄影**：分享了一份关于在摄影中应用三分法的全面指南，强调将关键元素沿虚构的线条和交点排列，以实现平衡的构图。
   - 建议包括将主体放置在交点处、将眼睛与水平线对齐以及使用引导线。[完整讨论链接](https://www.perplexity.ai/search/rule-of-thirds-in-photography-rfjmzH4aS9a8TZMSrjp1mA)。
- **健康、力量与体能建议**：建议包括保持均衡饮食、进行规律锻炼以及确保充足睡眠以获得最佳健康状态。
   - 该指南强调了身心健康的重要性。[完整讨论链接](https://www.perplexity.ai/search/how-to-achieve-health-strength-094kl4NzQea2mENjIOdG8Q)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/embed/CasopXlbvqo">YouTube</a>：未找到描述</li><li><a href="https://www.perplexity.ai/search/are-there-any-advances-in-apps-5NVLNla1T6.oHZAm_U70fw">是否有使用 AI 进行糖尿病管理的应用程序进展？非主动...</a>：确实在 AI 驱动的糖尿病管理应用方面取得了一些进展，这些应用提供见解和趋势，以协助患者和医生……</li><li><a href="https://www.perplexity.ai/search/rule-of-thirds-in-photography-rfjmzH4aS9a8TZMSrjp1mA">摄影中的三分法</a>：三分法是摄影中的一种构图准则，即将图像想象成由两条等距的水平线……分为九个相等的部分。</li><li><a href="https://www.perplexity.ai/search/how-to-achieve-health-strength-094kl4NzQea2mENjIOdG8Q">如何实现：生活中的健康、力量与体能。</a>：为了实现最佳健康，必须采取包括身心健康在内的平衡生活方式。以下是一些关键策略：1.……</li><li><a href="https://www.perplexity.ai/search/perplexitynodiscordhatouitutay-9Cl8rQZTROCmG_BS52HRQg#3">perplexityのdiscordはどういった用途がありますか？私はMidjourneyのweb版とdiscord版のようなものを期待してdiscordにjo...</a>：Perplexity AI 的 Discord 并不是像 Midjourney 那样用于生成图像的。Perplexity AI 的 Discord 主要有以下用途：1. 社区交流：Perplexity...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1261093061883199552)** (10 条消息🔥): 

> - `API 响应中的 Error 524`
> - `切换模型导致的问题`
> - `使用 Discord 和 Perplexity API 进行后端部署`
> - `使用 VPN 时的 API 错误`
> - `Cloudflare 导致的问题` 


- **Error 524 困扰 Perplexity API**: 一位成员报告称，在尝试将 Perplexity 集成到异步 Agent 框架中时收到错误代码 524，尽管已遵守模型速率限制（rate limits），且服务状态显示为正常运行。
   - *“我们的团队目前正在努力解决该问题，预计很快就会恢复运行。”*
- **模型切换导致 524 错误**: 另一位用户在从 `llama-3-{8b/70b}-instruct` 切换到 `llama-3-sonar-{large/small}-32k-online` 时遇到 524 错误或无效响应，并寻求解决建议。
- **使用 Discord 和 Perplexity 部署后端**: 一位用户分享了他们在托管服务器上部署后端的经验，通过 Discord 命令处理与 Discord API 和 Perplexity API 的主要交互。
   - 当生成响应时，由于 Discord 的内容限制，他们会返回一个将用户重定向到前端的按钮。
- **VPN 阻断 Perplexity API 访问**: 一位用户报告在使用 VPN 访问 Perplexity API 时收到 Error 500，并推测 API 是否宕机。
   - 他们总结道：*“显然在调用 pplx-api 时不能使用 VPN，”* 而另一位用户确认他们的 API 在不使用 VPN 的情况下工作正常。
- **Cloudflare 导致 API 问题**: 一位成员将 VPN 环境下使用 Perplexity API 出现的问题归咎于 Cloudflare，为故障排除提供了背景信息。


  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1261084635446775818)** (116 条消息🔥🔥): 

> - `GPT-4Chan 与 TruthfulQA`
> - `Benchmarks 的效用`
> - `Jsonnet 在配置中的角色`
> - `伦敦 AI Meetups`
> - `紧跟研究动态的重要性` 


- **GPT-4Chan 与 TruthfulQA 基准测试辩论**: 讨论围绕 GPT-4Chan 在 ChatGPT 出现之前曾是 TruthfulQA 上的 **SOTA** 展开，正如这篇 [相关推文](https://x.com/maximelabonne/status/1746317053143773628) 所强调的。
   - 成员们普遍认为 TruthfulQA 和 HellaSwag 等基准测试并不可靠，而 **MT-Bench** 和 **AGI Eval** 等基准测试是更准确的性能指标。
- **Jsonnet 在配置任务中褒贬不一**: 一位用户对 Jsonnet 表达了强烈的*复杂情感*，强调其挑战在于缺乏完善的调试和测试工具链，但同时也赞扬了其简洁的实现。
   - 讨论详细阐述了配置语言的普遍难度，Jsonnet 因其简洁的设计被认为是*最不坏*的选择，尽管它尚未被广泛采用或获得充分支持。
- **伦敦 AI Meetups 普遍令人失望**: 几位成员对 **伦敦 AI Meetups** 表示不满，指出这些活动通常迎合更大众的技术群体，而不是提供深入的技术讨论。
   - 有人建议，对于那些寻求深度 AI 技术交流的人来说，*大学研讨会*和 **ICML**、**ICLR** 等*研究会议*可能会提供更多实质性内容。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/maximelabonne/status/1746317053143773628">来自 Maxime Labonne (@maximelabonne) 的推文</a>: 来自 @gblazex 的优秀 LLM 基准测试评估 - MT-Bench, AGI Eval, ARC-C 和 MMLU 是良好的预测指标 - TruthfulQA 和 HellaSwag 非常糟糕：不要使用它们 - AGI Eval 是最具成本效益的...</li><li><a href="https://www.meetup.com/london-machine-learning-meetup/events/)">alert--small</a>: 未找到描述</li><li><a href="https://www.youtube.com/@LondonMachineLearningMeetup/videos))">London Machine Learning Meetup</a>: 伦敦机器学习见面会是欧洲最大的机器学习社区。往届演讲者包括 Juergen Schmidhuber, David Silver, Yoshua Bengio 和 Andrej Karpathy...</li><li><a href="https://mikehadlow.blogspot.com/2012/05/configuration-complexity-clock.html">Code rant: 配置复杂度时钟</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1261096438826274919)** (15 条消息🔥): 

> - `GPT-2 训练的内存带宽增长`
> - `LLM 的量化技术`
> - `SOTA LLM 在简单问题上的崩溃分析`
> - `随机 MDP 中的时间距离`
> - `从被动数据中进行因果推理` 


- **GPT-2 训练受内存带宽限制**：**GPT-2** 规模的模型若要在 1 小时内完成 **1 万亿 token** 的训练，仍需要 **1000 倍的内存带宽** 增长。
- **利用 Hadamard 变换推进量化**：LLM 的激活值存在离群值（outliers），这给量化带来了挑战。但使用来自 QuIP 的 **Hadamard 变换** 可以有效地减少误差，并能以极低的成本融合旋转嵌入（rotary embedding）等操作。
   - [Together 的博客文章](https://www.together.ai/blog/flashattention-3) 强调了该技术在 LLM 训练中实现 **FP8 精度** 的应用前景。
- **SOTA LLM 缺乏鲁棒性**：更新后的 [AIW ArXiv 论文](https://arxiv.org/abs/2406.02061) 指出，现代 LLM（包括对 **Claude 3.5 Sonnet** 和 **Qwen 2 72B instruct** 的评估）在面对简单问题的变体时，推理能力会出现 **显著崩溃**。
   - 论文认为目前的基准测试（benchmarks）未能揭示这些问题，呼吁改进模型和评估指标。
- **随机 MDP 中的时间距离缺乏度量结构**：[最近的一篇论文](https://x.com/svlevine/status/1811253559603888439) 探讨了在随机 MDP 中设计（拟）度量距离概念的挑战，为这一长期存在的问题提供了解决方案。
- **用于 LLM 数学推理的合成数据微调**：[ArXiv 上的研究](https://arxiv.org/abs/2406.14532) 表明，与在初始生成的数据上训练相比，在 **自生成的合成数据** 上微调 LLM 可以使数学推理问题的效率 **翻倍**。
   - 然而，这种方法也可能放大 **伪相关（spurious correlations）**，有时会导致随着数据量增加，Scaling 趋势出现持平或反向。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.07612">Teaching Transformers Causal Reasoning through Axiomatic Training</a>：为了让基于文本的 AI 系统在现实世界中交互，因果推理是一项必不可少的技能。由于干预数据的生成成本很高，我们研究了 Agent 在多大程度上可以学习因果推理...</li><li><a href="https://arxiv.org/abs/2406.14532">RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold</a>：在模型生成的合成数据上进行训练是微调 LLM 的一种很有前景的方法，但目前尚不清楚它何时会有所帮助或产生危害。在本文中，我们通过数学推理研究了这个问题...</li><li><a href="https://arxiv.org/abs/2406.02061">Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models</a>：大语言模型（LLMs）通常被描述为基础模型——即在 few-shot 或 zero-shot 模式下，能够跨各种任务和条件进行强力迁移的模型...</li><li><a href="https://www.together.ai/blog/flashattention-3">FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision</a>：FlashAttention-3 在 H100 上实现了高达 75% 的 GPU 利用率，使 AI 模型速度提升高达 2 倍，并能高效处理更长的文本输入。它支持更快的 LLM 训练和推理...</li><li><a href="https://x.com/svlevine/status/1811253559603888439">Sergey Levine (@svlevine) 的推文</a>：随机 MDP 中的时间距离（状态之间预期的步数）通常缺乏度量结构。如何设计一个（拟）度量的概念一直是一个长期存在的问题...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1261197613181702175)** (1 条消息): 

> - `新皮层神经元数量`
> - `体重：脑重`
> - `智能指标` 


- **新皮层神经元数量 vs. 体重：脑重**：一位用户认为，**新皮层神经元数量**比**体重：脑重**的比率更适合衡量智能。
   - 他们指出，如果后者成立，那么*小鼠应该比人类聪明得多*。
- **智能争论中的体重与脑重**：这场辩论突出了关于**体重：脑重**比率或**新皮层神经元数量**哪个能更好预测智能的对立观点。
   - 会议指出，仅依靠体重和脑重的比率会导致*小鼠比人类更聪明*这一荒谬结论。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1261450950946852935)** (1 条消息): 

> - `lm-eval Python API`
> - `实现用于模型评估的服务端`
> - `转换自定义模型` 


- **针对自定义模型的 lm-eval Python API**：一位成员询问是否存在现有的 Python API，以便在 **transformer_lens** 格式下对自定义模型使用 **lm-eval**，并指出在训练期间使用 transformerlens hooks 的便利性。
   - 他们寻求建议，询问是实现一个服务端还是将模型转回 **transformers** 格式才是最简单的评估路径。
- **评估自定义模型的最简路径**：该成员寻求关于在其自定义模型上运行评估的最佳方法的建议，考虑了实现服务端或转换模型格式等选项。
   - *关于这里最简路径的任何建议都将不胜感激。*


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1261035246648950824)** (37 条消息🔥): 

> - `FlashAttention-3 发布`
> - `OpenAI 营收传闻`
> - `OpenAI AGI 分级框架`
> - `去中心化 AI 训练`
> - `复合 AI 系统融资` 


- **FlashAttention-3 加速现代 GPU**：[FlashAttention-3](https://x.com/tri_dao/status/1811453622070444071) 现已发布，使 FP16 的 Attention 速度提升了 1.5-2 倍，在 H100 GPU 上最高可达 740 TFLOPS。
   - 据称在 H100 上实现了 **75% 的利用率**，在使用 FP8 时接近 1.2 PFLOPS。
- **OpenAI 令人印象深刻的营收预测**：根据 FutureResearch 的一份 [报告](https://x.com/jvnixon/status/1811278381184672156)，OpenAI 预计将从 ChatGPT Plus 获得 19 亿美元收入，从 ChatGPT Enterprise 获得 7.14 亿美元，从 API 获得 5.1 亿美元，以及从 ChatGPT Team 获得 2.9 亿美元。
   - 营收预估非常可观，展示了 OpenAI 在 AI 行业的潜在主导地位。
- **OpenAI 揭晓 AGI 进展框架**：OpenAI 引入了一个 [5 级框架](https://x.com/shiringhaffary/status/1811508824970264595) 来追踪迈向 AGI 的进展，声称他们目前处于第 2 级（“推理者”）。
   - 最近的一次 [全员会议](https://archive.is/SLtFQ) 展示了 GPT-4 增强推理能力的演示。
- **Prime Intellect 攻克去中心化 AI 训练**：[Prime Intellect](https://x.com/shaughnessy119/status/1811459606377582857) 推出了 OpenDiLoCo，这是 DeepMind DiLoCo 的开源版本，支持跨全球节点进行 AI 模型训练。
   - 他们成功地在三个国家之间训练了一个 1.1B 参数的模型，展示了去中心化 AI 训练的实际潜力。
- **Fireworks AI 获得 5200 万美元 B 轮融资**：[Fireworks AI](https://x.com/lqiao/status/1811500361485517153) 获得了 5200 万美元的 B 轮融资，以增强其推理平台并加速向复合 AI 系统的转型。
   - 这笔资金将支持与 **Nvidia 和 AMD** 的集成，以及针对企业级 AI 解决方案的高级定制。


<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://x.com/shaughnessy119/status/1811459606377582857?s=46&t=6FDPa">来自 Tommy (@Shaughnessy119) 的推文</a>：这太酷了 🤖 @PrimeIntellect 重现了 Deep Mind 关于去中心化 AI 训练的研究。节点只需每 500 步同步一次，因此它们不需要物理邻近。他们训练...</li><li><a href="https://x.com/lqiao/status/1811500361485517153">来自 Lin Qiao (@lqiao) 的推文</a>：Fireworks AI 已完成由 @sequoia 领投的 5200 万美元 B 轮融资！这一轮融资将推动我们增强推理平台并引领向复合 AI 系统（compound AI systems）转型的使命。非常感谢我们的投资...</li><li><a href="https://x.com/shaughnessy119/status/1811459606377582857?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Tommy (@Shaughnessy119) 的推文</a>：这太酷了 🤖 @PrimeIntellect 重现了 Deep Mind 关于去中心化 AI 训练的研究。节点只需每 500 步同步一次，因此它们不需要物理邻近。他们训练...</li><li><a href="https://x.com/itamar_mar/status/1811451611463422012">来自 Itamar Friedman (@itamar_mar) 的推文</a>：针对大型代码库使用 RAG 进行代码问答面临着独特的挑战。我们现在正在分享我们如何使用 > @llama_index、> 静态分析、> 高级分块范式，来交付一个可行的解决方案...</li><li><a href="https://x.com/tri_dao/status/1811453622070444071">来自 Tri Dao (@tri_dao) 的推文</a>：FlashAttention 被广泛用于加速 Transformer，已经使 Attention 速度提升了 4-8 倍，但尚未充分利用现代 GPU 的优势。我们正在发布 FlashAttention-3：在 FP16 上速度提升 1.5-2 倍，使用...</li><li><a href="https://x.com/shiringhaffary/status/1811508824970264595?s=61">来自 Shirin Ghaffary (@shiringhaffary) 的推文</a>：OpenAI 提出了一个包含 5 个等级的框架来追踪迈向 AGI 的进展，并认为他们目前接近第 2 级（“推理者”）。在最近的全体会议上，领导层还展示了一个研究演示...</li><li><a href="https://x.com/swyx/status/1779314692420485483">来自 swyx (@swyx) 的推文</a>：来自 @YoungPhlo_ 的总结 https://gist.github.com/swyxio/3b3992736879e2c2931b91cc7127894f 非常准确！</li><li><a href="https://x.com/jvnixon/status/1811278381184672156?s=61">来自 Jeremy Nixon (@JvNixon) 的推文</a>：futureresearch 关于 OpenAI 营收的报告已出炉，显示：ChatGPT Plus 营收 19 亿美元（770 万订阅用户，20 美元/月），ChatGPT Enterprise 营收 7.14 亿美元（120 万用户，50 美元/月），API 营收 5.1 亿美元，以及 2.9 亿...</li><li><a href="https://x.com/karpathy/status/1811467135279104217?s=46">来自 Andrej Karpathy (@karpathy) 的推文</a>：2019 年，OpenAI 通过这篇帖子发布了 GPT-2：https://openai.com/index/better-language-models/。今天（约 5 年后），你可以花费约 672 美元，在单个 8XH100 GPU 节点上运行 24 小时来训练你自己的模型。...</li><li><a href="https://archive.is/SLtFQ">OpenAI 设定等级以追踪迈向超人工智能的进展 - Blo…</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: 新播客上线！https://x.com/swyx/status/1811898574416019562
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1261411650687209588)** (86 条消息🔥🔥): 

> - `3E 首字母缩略词`
> - `Logprob 评估`
> - `Langgraph 状态管理`
> - `PDF 转 Markdown 工具`
> - `RAG 架构` 


- **提议 3E 首字母缩略词**：一位用户建议了一个易记的首字母缩略词 **3E: Extract (提取), Evaluate (评估), Extend/Expand (延伸/扩展)**。
- **用于文档增强的 Logprob 评估**：讨论了将 **Logprob** 作为评估文档增强中置信度评分的一种技术。
   - 一位用户提到将 **logprobs** 用于医学扫描，肯定了 ReAct 框架在状态管理方面的效率。
- **Langgraph 在状态管理方面表现出色**：强调了 **Langgraph** 在图状态内存管理中的价值，用于跟踪迭代步骤和并行过程。
   - 与 **XState** 用于管理应用逻辑的基于 Actor 的方法进行了对比。
- **PDF 转 Markdown 工具演示**：下周，**vikp** 将演示他的 PDF 转 Markdown 工具 **(marker + surya)**。
   - 更多详情可在 [VikParuchuri 的 GitHub](https://github.com/VikParuchuri) 上查看。
- **关于 RAG 架构的即将到来的主题和资源**：Nuvic 提到了一场定于 2024 年 3 月 15 日举行的关于 **RAG 架构** 的会议。
   - 分享了关键资源，包括来自 **latent.space** 和 **LangChain** 博客的链接。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://zod.dev/">TypeScript-first schema validation with static type inference</a>: 具有静态类型推导的 TypeScript 优先模式验证</li><li><a href="https://huggingface.co/nisten/bakllava-14b-2xMoE-alpha-build">nisten/bakllava-14b-2xMoE-alpha-build · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/html/2407.07071v1">Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps</a>: Lookback Lens：仅使用 Attention Maps 检测和缓解 Large Language Models 中的上下文幻觉</li><li><a href="https://arxiv.org/abs/2310.14566">HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models</a>: 我们介绍了 HallusionBench，这是一个专为评估图像上下文推理而设计的综合基准。该基准对先进的大型视觉语言模型提出了重大挑战 (...</li><li><a href="https://github.com/chand1012/git2gpt">GitHub - chand1012/git2gpt: Convert a Git repo into a ChatGPT prompt!</a>: 将 Git 仓库转换为 ChatGPT 提示词！通过在 GitHub 上创建账号来为 chand1012/git2gpt 做出贡献。</li><li><a href="https://github.com/Mavenoid/prompt-hyperopt">GitHub - Mavenoid/prompt-hyperopt: Improve prompts for e.g. GPT3 and GPT-J using templates and hyperparameter optimization.</a>: 使用模板和超参数优化改进 GPT3 和 GPT-J 等模型的提示词。 - Mavenoid/prompt-hyperopt</li><li><a href="https://github.com/VikParuchuri">VikParuchuri - Overview</a>: VikParuchuri 拥有 90 个代码仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/openvinotoolkit/anomalib">GitHub - openvinotoolkit/anomalib: An anomaly detection library comprising state-of-the-art algorithms and features such as experiment management, hyper-parameter optimization, and edge inference.</a>: 一个异常检测库，包含最先进的算法以及实验管理、超参数优化和边缘推理等功能。 - openvinotoolkit/anomalib</li><li><a href="https://github.com/truera/trulens">GitHub - truera/trulens: Evaluation and Tracking for LLM Experiments</a>: LLM 实验的评估与追踪。通过在 GitHub 上创建账号来为 truera/trulens 做出贡献。</li><li><a href="https://github.com/seanchatmangpt/dspygen">GitHub - seanchatmangpt/dspygen: A Ruby on Rails style framework for the DSPy (Demonstrate, Search, Predict) project for Language Models like GPT, BERT, and LLama.</a>: 一个为 GPT、BERT 和 LLama 等语言模型设计的 Ruby on Rails 风格框架，用于 DSPy (Demonstrate, Search, Predict) 项目。 - seanchatmangpt/dspygen</li><li><a href="https://github.com/statelyai/xstate">GitHub - statelyai/xstate: Actor-based state management &amp; orchestration for complex app logic.</a>: 基于 Actor 模型的复杂应用逻辑状态管理与编排。 - statelyai/xstate</li><li><a href="https://github.com/tianyi-lab/HallusionBench">GitHub - tianyi-lab/HallusionBench: [CVPR&#39;24] HallusionBench: You See What You Think? Or You Think What You See? An Image-Context Reasoning Benchmark Challenging for GPT-4V(ision), LLaVA-1.5, and Other Multi-modality Models</a>: [CVPR&#39;24] HallusionBench：眼见即所思？还是思即所见？一个挑战 GPT-4V(ision)、LLaVA-1.5 及其他多模态模型的图像上下文推理基准 - ti...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024 主题、日期、主持人、资源、@dropdown、@ GenAI 的 UI/UX 模式，1/26/2024，nuvic，&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://github.com/jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness">GitHub - jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness: Awesome-LLM-Robustness: a curated list of Uncertainty, Reliability and Robustness in Large Language Models</a>: Awesome-LLM-Robustness：关于 Large Language Models 中不确定性、可靠性和鲁棒性的精选列表 - jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness</li><li><a href="https://github.com/EGjoni/DRUGS">GitHub - EGjoni/DRUGS: Stop messing around with finicky sampling parameters and just use DRµGS!</a>: 别再折腾那些繁琐的采样参数了，直接使用 DRµGS 吧！ - EGjoni/DRUGS</li><li><a href="https://github.com/elder-plinius/AutoTemp">GitHub - elder-plinius/AutoTemp: A trial-and-error approach to temperature opimization for LLMs. Runs the same prompt at many temperatures and selects the best output automatically.</a>: 一种用于 LLM 温度优化的试错方法。在多种温度下运行相同的提示词，并自动选择最佳输出。 - elder-plinius/AutoTemp
</li>
</ul>

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1261047786267476079)** (68 条消息🔥🔥): 

> - `Chroma Vector Store`
> - `OpenAI Embedding Function`
> - `FAISS vs Chroma`
> - `LangChain Agents and Tools`
> - `Using OpenAI Vector Store as Retriever` 


- **结合 OpenAI Embeddings 使用 Chroma Vector Store**：用户讨论了如何加载持久化的 Chroma Vector Store 以及为什么需要 Embedding Function，强调了在持久化和加载集合时必须确保 `collection_name` 保持一致，以避免错误。
   - 还探讨了持久化存储问题以及跟踪已嵌入文档的高效方法，以避免不必要的 Embedding 重复计算。
- **优化 OpenAI Embedding 初始化**：社区分享了加速 OpenAI Embedding Function 初始化的技术，例如使用内存或 Redis 缓存等缓存机制来避免重复计算 Embedding。
   - 建议包括通过减少频繁的 Token 加载来优化文档 Embedding 过程，并考虑使用异步请求来处理 Embedding。
- **大规模数据集下 FAISS 与 Chroma 的效率对比**：关于使用 FAISS 还是 Chroma 的辩论指出，由于 FAISS 的高效性，它更适合大规模数据集，而 Chroma 在持久化存储和较小数据集方面更具优势。
   - 为了获得最佳性能，建议采用结合方案：使用 Chroma 进行持久化，使用 FAISS 进行相似度搜索。
- **LangChain Agents：问题与最佳实践**：用户对 LangChain Agents 不必要地重新嵌入文档以及如何缩短 Vector Store 的初始化时间表示关注。
   - 讨论了具体的解决方案和优化措施，包括持久化策略和提高 Agent 效率的技术。
- **将 OpenAI Vector Store 作为 Retriever 使用**：提供了关于如何在 LangChain 中将 OpenAI Vector Store 作为 Retriever 使用的指南，并附带了从 Vector Store 创建 Retriever 的分步说明。
   - 重点在于确保高效使用 Vector Store 进行文档检索，避免过多的重复计算。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://js.langchain.com/v0.2/docs/how_to/caching_embeddings/#in-memory>)">如何缓存 Embedding 结果 | 🦜️🔗 Langchain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="http://localhost:6379.>">未找到标题</a>: 未找到描述</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/caching_embeddings/#redis>)">如何缓存 Embedding 结果 | 🦜️🔗 Langchain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/vectorstore_retriever/#creating-a-retriever-from-a-vectorstore>)">如何将 Vector Store 用作 Retriever | 🦜️🔗 LangChain</a>: Vector Store Retriever 是一个使用 Vector Store 来检索文档的 Retriever。它是对 Vector Store 类的轻量级封装，使其符合 Retriever 接口。</li><li><a href="https://github.com/langchain-ai/langchain/issues/2326>))">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/1824>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/8957>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/6109>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/3011>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/17237>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/17412>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/5683>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/vectorstores/lantern/#working-with-vectorstore>)">Lantern | 🦜️🔗 LangChain</a>: Lantern 是一个用于 Postgres 的开源向量相似度搜索工具</li><li><a href="https://js.langchain.com/v0.2/docs/integrations/vectorstores/couchbase/#create-vector-store>)">Couchbase | 🦜️🔗 Langchain</a>: Couchbase 是一款屡获殊荣的分布式 NoSQL 云数据库，为您的所有云端、移动端应用提供无与伦比的通用性、性能、可扩展性和经济价值。</li><li><a href="https://github.com/langchain-ai/langchain/issues/7175>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/2658>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/7436>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/14872>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/6938>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/3984>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/23797>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li>

unt on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1261212179990970418)** (1 messages): 

> - `Structured Data Synthesis` (结构化数据合成)
> - `Indexify`
> - `Towards AI Publication` (Towards AI 刊物)


- **Prashant Dixit 在 Towards AI 发表关于结构化数据合成的文章**：[Prashant Dixit](https://x.com/Prashant_Dixit0/status/1811618990352912846) 在 Towards AI 上重点介绍了从非结构化流水线中进行 **Structured data extraction**（结构化数据提取）的技术。
   - 采用了由 @tensorlake 开发的 **Indexify**，这是一个用于构建非结构化数据 **ingestion and extraction pipelines**（摄取和提取流水线）的数据框架，该示例中展示了其应用 [阅读更多](https://pub.towardsai.net/structured-financial-data-extraction-from-unstructured-data-ca2c8d166de6)。
- **Indexify 简化数据摄取**：**Indexify** 是由 @tensorlake 设计的数据框架，旨在帮助构建非结构化数据的摄取和提取流水线，正如 Prashant Dixit 所演示的那样。
   - [继续阅读](https://pub.towardsai.net/structured-financial-data-extraction-from-unstructured-data-ca2c8d166de6) 关于 Indexify 如何应用于结构化数据提取工作流。



**提及的链接**：<a href="https://x.com/Prashant_Dixit0/status/1811618990352912846">Prashant Dixit (@Prashant_Dixit0) 的推文</a>：从非结构化流水线中提取结构化数据。本例使用了 @tensorlake 的 Indexify。Indexify 是一个为构建非结构化数据的摄取和提取流水线而创建的数据框架...

  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1261043881621983294)** (25 messages🔥): 

> - `Dell Inspiron 3847 upgrades and limitations` (Dell Inspiron 3847 升级与限制)
> - `NPU support in x elite` (x elite 中的 NPU 支持)
> - `FlashAttention for LLMs` (用于 LLM 的 FlashAttention)
> - `Debugging GPU issues in Linux` (在 Linux 中调试 GPU 问题)
> - `Shifting to Linux from Windows` (从 Windows 切换到 Linux)


- **为游戏升级 Dell Inspiron 3847**：一位用户讨论了将在二手店发现的 [Dell Inspiron 3847](https://www.hardware-corner.net/desktop-models/Dell-Inspiron-3847/) 升级为游戏用途，通过安装更好的处理器、GPU、内存和存储，尽管专有元件可能会带来挑战。
   - 该机器配备了 **Intel Core i3 4130** 和 **GTX1650**，可以用于运行受限的 LLM，因为它符合较小模型的系统要求。
- **FlashAttention 加速 Transformer 训练**：用户讨论了 [FlashAttention](https://tridao.me/blog/2024/flash3/) 和 [FlashAttention-2](https://github.com/Dao-AILab/flash-attention)，强调了它们通过最小化内存读/写来提高 Transformer 训练和推理效率。
   - 该方法显著增加了 LLM 的上下文长度，为 **GPT-4** 和 **Llama 3** 的进步做出了贡献。
- **Linux 上加载模型的问题**：一位成员报告了在 Kali Linux 上加载模型的问题，尽管拥有 **1650 GTX GPU** 和适当的驱动程序，但仍导致错误（退出代码：4）。
   - 其他人建议关闭 GPU 加速、更新驱动程序，并确保在加载 **Phi3** 等较小模型时保持较低的 RAM 占用。
- **x elite 中的 NPU 支持仅限于 CPU**：一位用户询问了 x elite 中的 NPU 支持，另一位用户确认支持仅限于 CPU，**不支持 NPU**。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tridao.me/blog/2024/flash3/"> FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | Tri Dao </a>：未找到描述</li><li><a href="https://www.hardware-corner.net/desktop-models/Dell-Inspiron-3847/">Dell Inspiron 3847 – 规格和升级选项</a>：阅读关于 Dell Inspiron 3847 台式电脑的信息。查找详细规格、升级选项以及关于 CPU、RAM、PSU、主板和发布日期的信息</li><li><a href="https://www.hardware-corner.net/desktop-models/Dell-I">Dell I – 规格和升级选项</a>：阅读关于 Dell I 台式电脑的信息。查找详细规格、升级选项以及关于 CPU、RAM、PSU、主板和发布日期的信息
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1261091426859749457)** (8 条消息🔥): 

> - `Salesforce Einstein`
> - `Local Model Benchmarks` 


- **Salesforce 在 Einstein 模型上投入 2000 万美元**：Salesforce 的新 AI 模型命名为 **Einstein**，并支付了 [2000 万美元](https://www.businesswire.com/news/home/20161011005979/en/Greenlight-Collaborates-with-Salesforce-to-License-Einstein) 获取该名称的授权。
   - 评论中包含批评和幽默，一位用户指出 **Einstein** 徽标上的脸看起来像被绑架了一样，并建议制作一个“悲伤的 Einstein”表情包。
- **本地 AI 模型的个人基准测试**：一位成员分享了一个针对各种 AI 模型的 [个人基准测试表](https://dubesor.de/benchtable.html)，详细列出了使用加权评分系统在 83 个任务中的结果。
   - 该表格包含可排序的列，涵盖 **Reasoning, STEM, Utility, Coding** 和 **Censorship** 等类别，并指出这些分数反映了他们自己的经验，可能不代表更广泛的基准。



**提及的链接**：<a href="https://dubesor.de/benchtable.html">Dubesor LLM 基准测试表</a>：未找到描述

  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1261073292626890845)** (19 条消息🔥): 

> - `3090 vs 4090 for AI`
> - `NVIDIA 5090 rumors`
> - `Multi-GPU setups for AI`
> - `V100 compute nodes`
> - `Performance of ARM computers with LLMs` 


- **对于 AI 来说，3090 比 4090 更有性价比**：用户讨论了 **3090** 与 **4090** 在 AI 方面的价值，许多人认为鉴于当前价格，3090 是更好的选择。
   - 一位用户提到 **4090** 的显存带宽仅增加了 7%，且 TFLOPs 的代际提升相对较小。
- **NVIDIA 5090 显存传闻**：根据 Reddit 上的讨论，传闻新款 **NVIDIA 5090** 将拥有 **28GB** 显存，而非预期的 32GB。
   - 一位用户引用了一个关于 [使用六块 3090 GPU 构建经济实惠的 144GB 显存服务器](https://www.reddit.com/r/LocalLLaMA/comments/1dzejen/cheapest_144gb_vram_server_youll_ever_see/) 的 Reddit 帖子。
- **关于 V100 计算节点与 3090 配置的辩论**：一位用户认为，用于多卡 3090 配置的相同预算也可以购买多个 V100 计算节点，每个节点都具有更高带宽的 HBM2 显存。
   - 然而，其他人指出 3090 配置速度更快且更具成本效益，特别是对于典型的 AI 使用场景。
- **ARM 电脑运行 LLM 的性能**：一位用户询问了在新型 ARM 电脑上运行 LLM 的情况，引发了关于性能的简短讨论。
   - 虽然没有提供具体答案，但另一位用户对系统速度给予了正面评价，称：*在我看来，聊天速度相当不错。*



**提及的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/comments/1dzejen/cheapest_144gb_vram_server_youll_ever_see/">Reddit - 深入探索</a>：未找到描述

  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1261304422177181779)** (6 条消息): 

> - `OpenCL backend issues`
> - `Cuda vs ROCM`
> - `Vulkan support` 


- **OpenCL 后端在模型加载方面存在困难**：一位用户指出，**OpenCL** 后端似乎已损坏，无法在 **7600XT** 上加载任何模型。
   - **OpenCL** 已被弃用，无法很好地处理最新的模型，需要改用 **Cuda** 或 **ROCM**。
- **Cuda 和 ROCM：互斥**：一位用户确认你只能使用 **Cuda** 或 **ROCM** 其中的一种，不能同时使用。
   - 其他用户也确认，已弃用的 **OpenCL** 无法有效处理最新的模型。
- **LM Studio 中的 Vulkan 支持**：用户询问了 LM Studio 对 **Vulkan** 的支持情况，参考了使用它代替 OpenCL 的 **ollama**。
   - **Vulkan 支持即将推出**，但目前没有具体的时间表 (ETA)。

### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1261146051885404163)** (10 条消息🔥): 

> - `在 React 中使用 LM Studio 设置 RAG`
> - `在 Discord 开发频道中的负面体验`
> - `关于 Rust 与 C++ 的讨论`
> - `用于集成的 LM Studio SDK` 


- **在 React 中使用 Gemma 2 设置 RAG**：一位用户正在创建一个 React 应用程序，并通过伪 OpenAI API/推理服务器功能向其传递由 LM Studio 运行的 LLM (Gemma 2)。
   - 他们有几个存储在磁盘上的 PDF，正在寻找设置 RAG 的最佳方法；另一位用户建议使用像 Faiss 这样的 embeddings 数据库。
- **在 Discord 寻求帮助时的负面体验**：一位用户分享了在 Discord 机器人开发频道寻求帮助时，被引导至无关解决方案并受到轻慢对待的负面体验。
   - *他们转而求助于 ChatGPT 并得到了答案*，按照最初的设想解决了他们的机器人问题。
- **讨论被重定向至合适的频道**：一位用户提醒另一位用户，当前的 Discord 频道专注于 LM Studio，并将其重定向至更相关的查询构建频道。
   - 强调了应使用 #1234988891153629205 频道来提问此类问题。
- **推荐使用 LM Studio SDK**：建议使用 LM Studio SDK 加上 LangChain，将 LLM (Gemma 2) 集成到 React 应用程序中。


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1261036769592213534)** (19 条消息🔥): 

> - `去中心化 AI 计算`
> - `自动化个性化 SEO`
> - `OpenAI 的分级系统`
> - `Claude 与 ChatGPT 在文档阅读方面的对比`
> - `GPT-4o 与 Sora 更新` 


- **对去中心化 AI 计算方法的需求**：一位成员讨论了使用去中心化计算优化 **CMOS 芯片**的潜在好处，特别是考虑到 stable diffusion 的开源特性以及对闲置处理能力的需求。
   - 另一位参与者强调了利用去中心化扩展 HPC 规模以实现高效并行计算的必要性。
- **用于通信渠道的自动化个性化 SEO**：一位成员提议了一个 AI 系统，该系统可以聚合来自 Telegram 和 Discord 等各种通信平台的各种聊天内容，并优先处理需要用户回复的消息。
   - 这个想法幽默地延伸到根据互动和活动的优先级来整理好友列表。
- **OpenAI 发布 AI 模型新分级系统**：一篇文章讨论了来自 **OpenAI** 的 [分级系统](https://archive.is/SLtFQ)，其中 'Reasoners' 为下一阶段，能够在不使用工具的情况下解决博士级的问题。
   - 该分级系统进阶至 'Agents' 和 'Organizations'，暗示了即将到来的新模型能力。
- **Claude 在文档阅读方面优于 ChatGPT**：当被问及 Claude 还是 ChatGPT 更适合阅读文档时，一位成员断言 **Claude** 更胜一筹，因为它具有更长的 context length。
- **关于 GPT-4o 和 Sora 可用性的推测**：一位参与者推测了向 'Reasoners' 和 'Agents' 级别迈进的进展，暗示内部开发正在进行中。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-07-11/openai-sets-levels-to-track-progress-toward-superintelligent-ai">Bloomberg - Are you a robot?</a>：未找到描述</li><li><a href="https://x.com/kimmonismus/status/1811498151964033084?s=46">来自 Chubby♨️ (@kimmonismus) 的推文</a>：OpenAI 正在展示新的技能和可能的模型。来自 @business 的一篇新文章报道了 OpenAI 的分级系统。还展示了一个具有新能力的 ChatGPT 版本。从措辞来看……
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1261097170501636157)** (25 messages🔥): 

> - `ChatGPT-5 release speculation` (ChatGPT-5 发布推测)
> - `Optimizing ChatGPT configurations` (优化 ChatGPT 配置)
> - `ChatGPT-4o performance` (ChatGPT-4o 性能)
> - `New features expected in ChatGPT-5` (ChatGPT-5 预期新功能)
> - `DALL-E image generation issues` (DALL-E 图像生成问题)


- **关于 ChatGPT-5 发布时间线的辩论**：一位成员推测 [ChatGPT-5 的测试可能在 2024 年底开始](https://www.geekygadgets.com)，并可能在 2025 年发布，但其他人批评这是毫无根据的推测。
   - 他们引用了 **Evening Standard** 和 **India Today** 等来源，但因依赖非官方网站而受到批评。
- **优化 ChatGPT 配置**：一位用户询问如何适配、优化和配置 ChatGPT，并提到了使用 **second brain** 策略。
   - 对话偏离了主题，没有提供切实有效的优化技术。
- **ChatGPT-4o 因健忘受到批评**：有观点指出 **ChatGPT-4o** 虽然速度更快，但经常遗忘最近的指令，影响了编程任务。
   - 许多用户表示更喜欢旧的 v3.5 模型，因为它具有更好的记忆能力。
- **ChatGPT-5 将增强情感智能**：预计 ChatGPT-5 将能更好地理解和回应人类情感，并提供广泛的自定义选项。
   - 改进包括减少重复指令，并增加针对文本、图像、音频以及可能视频生成的先进多模态能力。
- **DALL-E 图像生成问题**：用户报告称，在给定 GPT Instructions 时，**DALL-E** 无法可靠地创建图像。
   - 问题包括提示词截断以及输出文本而非图像。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1261101505021214831)** (2 messages): 

> - `Chatbot with RAG` (带有 RAG 的聊天机器人)
> - `Contradiction in instructions` (指令冲突)


- **RAG 聊天机器人提示词给出奇怪答案**：一位成员目前正在开发一个带有 **RAG** 的聊天机器人，并提到他们的提示词有时会给出奇怪的答案。
   - 他们请求帮助以改进提示词并解决这些问题。
- **提高清晰度以避免冲突**：另一位成员建议，奇怪的回答可能是由于指令中的冲突造成的。
   - 他们建议 **重写** 指令以使其更加清晰。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1261101505021214831)** (2 messages): 

> - `RAG chatbot development` (RAG 聊天机器人开发)
> - `Prompt contradictions` (提示词冲突)


- **RAG 聊天机器人开发中的挑战**：一位成员分享说，他们正在开发一个带有 **RAG** 的聊天机器人，但由于提示词指令不清晰，有时会收到奇怪的答案。
   - 另一位成员建议重写 **instructions**，使其更加清晰以避免冲突。
- **改进聊天机器人指令**：一位成员强调了聊天机器人提示词指令中的 **矛盾之处**。
   - 他们建议重写指令以增强清晰度并避免混淆。


  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1261056713180577952)** (35 messages🔥): 

> - `Command R Plus 模型用例`
> - `使用 Cohere 实现 AI 新闻自动化`
> - `Cohere toolkit 开源`
> - `创建独特表情符号`
> - `GitHub 上的 OpenArena` 


- **讨论 Command R Plus 模型用例**：**Mapler** 询问社区成员关于 Command R Plus 模型的实际应用案例，并收到了多项建议，包括社交媒体内容生成、播客描述和团队沟通。
   - *Sssandra* 分享了她在日常工作中的多种用法，包括一个与 Notion 和 Google Drive 集成的内部版本，用于回答社区问题。
- **在 Discord 中自动化 AI 新闻更新**：**Mapler** 希望使用 Command R Plus、**Lang chain** 和 webhooks 在 Discord 频道中自动化 AI 新闻更新。**Karthik_99_** 表示支持并提供了进一步的帮助。
   - *Mapler* 正在考虑编写一个类似 chat-GPT 的界面，并配备各种工具，计划根据测试反馈进行迭代。
- **Cohere toolkit 正式开源**：*Sssandra* 宣布 Cohere 已在 [GitHub](https://github.com/cohere-ai/cohere-toolkit) 上开源了他们的聊天界面，并提到了即将推出的 OCI 集成。
   - *Mapler* 对在个人项目中使用 Cohere 表示兴奋，并承诺在社区频道发布更新。
- **使用 AI 创建表情符号**：**Roazzy** 表达了使用 AI 创建独特表情符号的愿望，并指出目前唯一的方法是手动绘制。
   - **Karthik_99_** 表现出兴趣并询问是否有解决方案，强调了该功能的潜力。
- **在 GitHub 上介绍 OpenArena**：*Le_mess* 分享了一个名为 [OpenArena](https://github.com/syv-ai/OpenArena) 的 GitHub 项目，LLM 在其中相互竞争以获得更好的数据集质量。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/syv-ai/OpenArena">GitHub - syv-ai/OpenArena</a>: 通过在 GitHub 上创建账户，为 syv-ai/OpenArena 的开发做出贡献。</li><li><a href="https://github.com/cohere-ai/cohere-toolkit">GitHub - cohere-ai/cohere-toolkit: Cohere Toolkit 是一系列预构建组件的集合，使用户能够快速构建和部署 RAG 应用。</a>: Cohere Toolkit 是一系列预构建组件的集合，使用户能够快速构建和部署 RAG 应用。 - cohere-ai/cohere-toolkit
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1261292230224777226)** (3 messages): 

> - `Embedding 模型成本降低`
> - `项目分享礼仪` 


- **Cohere 的 Embedding 模型大幅降低 40-70% 的成本**：一名成员宣布 **Cohere 的 embedding 模型** 显著降低了 **40-70%** 的成本。
   - *Noice!* 是社区的热烈回应。
- **项目分享礼仪提醒**：一名版主提醒成员将讨论集中在 **Cohere 特定项目**上，并删除了一条无关帖子。
   - 版主强调了遵守频道准则的重要性。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1261351437644857455)** (23 messages🔥): 

> - `Llama 3 发布`
> - `OpenAI 的新项目 Strawberry`
> - `自托管大模型`
> - `API 与自托管成本对比`
> - `大模型敏感数据处理` 


- **拥有 405B 参数的 Llama 3 即将发布**：Llama 3 405B 预计将于 **7 月 23 日**发布，距离 Llama 2 发布几乎整整一年。已确认该模型为多模态模型，更多详情请参阅[此处](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23)。
- **OpenAI 的新项目 Strawberry 曝光**：据路透社报道，OpenAI 正在开发代号为 **Strawberry** 的新推理技术。该项目与斯坦福大学在 2022 年开发的 *Self-Taught Reasoner (STaR)* 方法有相似之处。
- **自托管大模型的可行性**：托管一个 400B 参数的模型需要大约 **400GB VRAM**，在 8bit 配置下大约相当于 5 张 A100/H100 GPU。这对于大型企业来说是可行的，但对小型公司来说具有挑战性。
- **大模型的 API 租赁与自托管成本对比**：对于不进行模型微调的公司，使用 Prompting API 通常比运行自己的 GPU 更具成本效益。除非涉及 **敏感数据**，否则通常首选从 Hyperscalers（超大规模云服务商）租赁 GPU。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/casper_hansen_/status/1811805236085891527">Casper Hansen (@casper_hansen_) 的推文</a>: @Teknium1 这与 @soumithchintala 将在上午 9 点进行的 ICML 演讲时间吻合</li><li><a href="https://x.com/steph_palazzolo/status/1811791968600576271?s=46">Stephanie Palazzolo (@steph_palazzolo) 的推文</a>: 周五的一则独家消息 w/ @SylviaVarnham —— Llama 3 405B 即将到来（而且很快！）。这款多模态模型定于 7 月 23 日发布，大约在 Llama 2 发布一年后。更多细节见：https://www....
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1261153391376072734)** (4 messages): 

> - `Soft-target distillation`
> - `Mistral-7B instruct-finetuning`
> - `语言模型指令微调过程`
> - `AgentInstruct 论文`
> - `KnowledgePile 和 AutoMathText 数据集` 


- **关于 soft-target distillation 方法论的困惑**：一位用户对 Medusa 等论文中关于在 **soft-target distillation** 期间需要并行运行两个模型的断言感到困惑，质疑为什么不能在保留 Teacher Model 概率的同时顺序运行推理。
   - 他们承认对齐内部表示可能确实需要在线模型，但认为在简单情况下可以简化。
- **对 Mistral-7B 的 instruct-finetuning 提出质疑**：一位用户发现 **Orca3/AgentInstruct** 论文中相对于 Mistral-7B 的指令微调改进令人惊讶，并质疑 Mistral 自身指令微调数据集的强度。
   - 他们计划将 **25M 数据集**的大小与 Mistral-7B 的 ift 数据集进行比较，并深入研究这两篇论文以获取更多见解。
- **AgentInstruct 的“刷榜”行为受到审视**："xeophon:" 评论说 AgentInstruct 看起来像是一个 **bench-maxxing**（刷榜型）模型。
   - 另一位用户解释了 AgentInstruct 的工作流程，包括文档转换和复杂化处理，并引用了 [KnowledgePile](https://huggingface.co/datasets/Query-of-CC/Knowledge_Pile) 和 [AutoMathText](https://huggingface.co/datasets/math-ai/AutoMathText) 等作为种子数据集。


  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1261076457912930407)** (7 messages): 

> - `GPT-4 Pricing`
> - `OpenAI's AGI Progress`
> - `Self-driving Similarities`
> - `New GPT-4 Skills`
> - `OpenAI Revenue Speculation` 


- **GPT-4 定价对比**：一位成员指出 **GPT-4** 每月收费 20 美元，并将其与另一项每月收费 5 美元的服务进行了对比。
   - 这种定价差异凸显了各种 AI 服务不同的切入点和价值主张。
- **OpenAI 的 AGI 进展说明**：OpenAI 分享了一个[五级系统](https://www.bloomberg.com/news/articles/2024-07-11/openai-sets-levels-to-track-progress-toward-superintelligent-ai?sref=P6Q0mxvj)来追踪其通用人工智能（AGI）的进展，强调他们距离人类水平的 AI 还有四个阶段。
   - 一位用户对该系统发表了评论，指出其与自动驾驶技术开发阶段的相似之处。
- **GPT-4 的高级技能**：在最近的一次内部会议中，OpenAI 展示了其 **GPT-4** AI 模型的新技能，这些技能展现了类人的推理能力。
   - 根据 [Bloomberg 的一篇文章](https://www.bloomberg.com/news/articles/2024-07-11/openai-sets-levels-to-track-progress-toward-superintelligent-ai?sref=P6Q0mxvj)，OpenAI 的发言人强调，这些测试是旨在进一步提升 AI 能力的常规内部实践。
- **针对 OpenAI 营收推测的回应**：一位 Twitter 用户指出，目前流传的一份关于 **OpenAI 营收**的推测性报告完全是基于聊天机器人对公开资料的总结。
   - 他们提供了一个由 [The Information](https://www.theinformation.com/articles/openais-annualized-revenue-doubles-to-3-4-billion-since-late-2023?utm_source=ti_app&rc=geicgp) 发布的更可靠的关于 OpenAI 营收的第一手报告链接。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://qz.com/openai-five-level-system-human-intelligence-ai-1851588122">OpenAI 表示 AI 达到人类智能有 5 个“等级”——目前已接近第 2 级</a>：ChatGPT 的开发者认为它目前处于第一级，即对话式 AI。</li><li><a href="https://x.com/aaronpholmes/status/1811870687037960467?s=46">来自 aaron holmes (@aaronpholmes) 的推文</a>：许多风投公司今天都在传阅一份推测 OpenAI 营收的“报告”，该报告完全基于聊天机器人对公开网络资源的总结。如果你想了解关于 OpenAI 营收数字的第一手报道，...
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1261137284376559748)** (2 messages): 

> - `Indexing Kernel in tinygrad`
> - `PyTorch 2024 H2 Roadmaps` 


- **tinygrad 引入 Indexing Kernel**：George Hotz 引入了一个 [Indexing Kernel](https://x.com/__tinygrad__/status/1811578147982491938)，由于存在上游的 LOAD 之 LOAD，这在 tinygrad 中通常是不允许的。
   - 他解释说，这完全是通过折叠求和循环（folding the sum loop）在后端生成的，创建了“计划中”内存访问的一个严格子集。
- **提议创建类似 PyTorch 的路线图**：一位成员建议创建类似于 [PyTorch 团队 2024 H2 计划](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2024-h2-roadmaps/2226)的路线图。
   - *“我们应该制定类似的路线图”*是核心观点，强调了拥有清晰开发路径的好处。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/__tinygrad__/status/1811578147982491938">来自 tiny corp (@__tinygrad__) 的推文</a>：这是一个 Indexing Kernel，X[idxs]。通常这个内核在 tinygrad 中是不允许的，因为在上游有一个 LOAD 之 LOAD。然而，它完全是通过折叠求和（folding the sum）在后端生成的...</li><li><a href="https://dev-discuss.pytorch.org/t/meta-pytorch-team-2024-h2-roadmaps/2226">Meta PyTorch 团队 2024 H2 路线图</a>：我们一直在思考如何在这里分享我们在 Meta 所做的 PyTorch 工作的路线图。我们以半年为单位进行规划，因此这些是我们 2024 H2 开源计划的一些公开版本...
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1261101576303411241)** (26 条消息🔥): 

> - `Custom Weight and Bias in Network`（网络中的自定义权重和偏置）
> - `Implementing Gradient Descent from Scratch`（从零开始实现梯度下降）
> - `Performance Issues with Manual Gradient Descent`（手动梯度下降的性能问题）
> - `Tensor Operations and Realization`（Tensor 操作与 Realization）
> - `Indexing Tensors and Kernel Performance`（Tensor 索引与 Kernel 性能）


- **在网络中成功定义自定义权重和偏置**：一位用户展示了在网络中定义自定义权重和偏置的成功实现，并得到了预期的图形输出。
   - 他们表示非常满意，并说道：*love this... thanks for the help*。
- **实现梯度下降面临挑战**：一位用户尝试在 tinygrad 中从零实现梯度下降，但发现如果没有 `optimizer.step`，过程会非常缓慢且逻辑不清晰。
   - 他们分享了 [代码片段](https://github.com/karpathy/makemore/blob/988aa59e4d8fefa526d06f3b453ad116258398d4/names.txt) 并寻求关于如何提高计算效率的建议。
- **手动为梯度下降 Realize Tensor**：George Hotz 建议使用 `model.weights.assign(model.weights - lr * model.weights.grad).realize()` 在梯度下降期间手动 realize Tensor 计算。
   - 他强调：*如果你想让计算发生，就需要 realize*。
- **调试梯度下降步骤中的缓慢问题**：性能问题被确定为损失计算期间的 Tensor 索引导致的，特别是在处理大型数据集时。
   - 建议使用基于 masking 的 `sparse_categorical_crossentropy` 实现作为更快的替代方案。
- **Tensor 索引 Bug 影响 Kernel 性能**：关于使用 `-probas[:, Y_train]` 以获得更好性能的建议导致了关于最大索引大小的断言错误。
   - 这被确定为一个 Bug，因为该表达式导致了 **idx.max too big** 错误。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/PaCmpygFfXo?t=6354)">The spelled-out intro to language modeling: building makemore</a>：我们实现了一个 bigram 字符级语言模型，在后续视频中，我们将进一步将其复杂化为现代 Transformer 语言模型，如 GPT...</li><li><a href="https://github.com/tinygrad/tinygrad/blob/6e0a5230786d41eabe9dc9e593b05997d3a1da73/tinygrad/engine/realize.py#L199-L202)">tinygrad/tinygrad/engine/realize.py</a>：你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1261051444686225549)** (15 条消息🔥): 

> - `H100 performance`（H100 性能）
> - `Attention masking in reward models`（Reward Model 中的 Attention Masking）
> - `OpenRouter API usage`（OpenRouter API 使用）
> - `Flash attention versions`（Flash Attention 版本）
> - `OpenArena open-source project`（OpenArena 开源项目）


- **H100 性能令成员兴奋**：成员们兴奋地讨论了 **H100** GPU 的性能，有人惊叹 *"H100 go brrrrr"*。
   - 这种热情暗示了该硬件带来的显著性能提升。
- **关于 Reward Model 中 Attention Masking 的辩论**：一位成员询问在 **Reward Model** 中应用 Attention Masking 的必要性，并在承认自己之前一直没这么做后寻求建议。
   - 一位成员推测这可能与 **axolotl** 训练无关，但对其他人的见解持开放态度。
- **寻求 WizardLM 数据集的 API 访问**：一位成员询问是否有人认识 **openrouter.ai** 的联系人，以便创建一个开源版本的 **WizardLM arena dataset**。
   - 另一位成员提到正在使用 **ollama** 开发本地托管的开源版本，并分享了 [OpenArena 项目](https://github.com/syv-ai/OpenArena)。
- **对 Flash attn3 GPU 兼容性的担忧**：成员们担心 **Flash attn3** 目前仅适用于 **H100** GPU。
   - 有人指出 **Flash attn2** 最初仅支持 **A100** 及更新型号，但后来兼容了 *3090* 和 *4090*；大家对 Flash attn3 的类似修复寄予厚望。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1261349385166524426)** (5 messages): 

> - `GaLore and Q-Galore`
> - `Dataset Shuffling` 


- **Q-Galore 在效率上优于 GaLore**：GaLore 通过将权重梯度投影到低秩子空间来减少内存占用，但它依赖于耗时的 SVD 操作，且在微调场景中与 LoRA 相比改进微乎其微。[Q-Galore](https://huggingface.co/papers/2407.08296) 通过结合量化和低秩投影解决了这些问题，大幅减少了内存占用和训练时间。
   - *Q-Galore* 的创新方法超越了 GaLore 的优势，它观察到某些梯度子空间会提前收敛，而其他子空间则频繁变化。
- **数据集打乱发生在 Epoch 之间**：一位成员提出了一个历史开发问题，即为什么没有明确支持在训练前对单个数据集进行打乱。
   - 另一位成员澄清说 **Batch 会在 Epoch 之间被打乱**，提问者接受了这一解释，并提到他们之前没有阅读该章节。



**提到的链接**：<a href="https://huggingface.co/papers/2407.08296">Paper page - Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients</a>：未找到描述

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1261130241712193607)** (1 messages): 

> - `LoRA finetuning`
> - `Layer selection`
> - `Few-shot learning challenges` 


- **为 72b 模型优化 LoRA 微调**：一位正在对 dolphin-vision-72b（qwen 72b 变体）进行 **LoRA finetuning** 的成员就 **Layer selection** 和效率寻求建议，怀疑将 LoRA 应用于所有层可能不是最有效的。
- **LoRA 微调中的层目标选择**：该成员询问了在超大型模型上针对特定层进行 LoRA 微调的实验方法和结果。
   - 他们特别感兴趣的是如何平衡 **attention** 和 **feed-forward layers** 以获得最佳结果。
- **微调后的 Few-shot learning 挑战**：一位成员指出微调后在 **Few-shot learning** 方面遇到了挑战，并询问其他人是否也遇到过类似情况以及如何解决。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1261259383384571924)** (1 messages): 

> - `General-purpose multi-turn chat dataset`
> - `Dataset recommendations` 


- **寻求最高质量的多轮对话数据集**：一位用户征求目前可用的最高质量 **general-purpose multi-turn chat dataset** 的建议，并提到数据集不需要超过 *10k 行*。
   - 回复中没有建议或提到具体的数据集。
- **数据集推荐请求**：该请求强调了对支持多轮对话、适用于通用目的的高质量数据集的需求。
   - 在给定的上下文中没有提供进一步的讨论或建议。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/)** (1 messages): 

wasamikirua: 在 LoRA merge 之后，我该如何将模型 push 到 Hub？
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1261047137655980052)** (15 messages🔥): 

> - `Training Data Concerns` (训练数据担忧)
> - `Integrations (Beta)` (集成 (Beta))
> - `Prompting Image Models` (提示词图像模型)
> - `405b Model Update` (405b 模型更新)
> - `Specialized Models` (专用模型)


- **了解 DeepInfra 的数据政策**：一位成员询问了关于保留训练数据的问题，并提到像 DeepInfra 这样的公司会保留数据。
   - [DeepInfra 会记录使用情况，但不会根据用户输入进行训练](https://deepinfra.com/privacy)，详细政策可以参考其官方网站。
- **Integrations (Beta) 开启新可能性**：成员们询问了新的 **Integrations (Beta)** 功能，该功能旨在为包括 Groq 在内的各种提供商使用自定义 API keys。
   - 未来的扩展将包括模型 API 之外的集成。
- **通过提示词放置位置改进弱模型**：一位用户分享了一个技巧，将文本提示词放在内容中的图像之后，以获得更好的响应。
   - 这种方法有助于较弱的模型准确理解并回答请求。
- **405b 模型发布期待**：一位用户宣布 **405b 模型** 预计很快发布，在社区中引起了兴奋。
   - [Bindu Reddy 在推特上发布了关于该模型预期发布的消息](https://x.com/bindureddy/status/1811807596682355125?s=46)，将 7 月 23 日标记为开源 AGI 的历史性一天。
- **关于专用模型与通用模型的辩论**：一位成员质疑为什么像 OpenAI 和 Anthropic 这样的公司不创建多个专用模型，而是创建一个通用模型。
   - Alex Atallah 表示赞同，建议应该考虑专业化，并询问用户最常使用哪些专用模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/bindureddy/status/1811807596682355125?s=46">来自 Bindu Reddy (@bindureddy) 的推文</a>：太棒了！！！7 月 23 日将载入开源 AGI 的史册！迫不及待 💃💃💃</li><li><a href="https://deepinfra.com/privacy">DeepInfra 隐私政策</a>：使用简单的 API 运行顶尖的 AI 模型，按需付费。低成本、可扩展且具备生产就绪的基础设施。
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1261081084335095878)** (2 messages): 

> - `Clip retrieval` (Clip 检索)
> - `Dataset access` (数据集访问)


- **Clip 检索不再可用**：一位用户询问关于 **Clip 检索不再工作** 的问题，并询问是否有查看/搜索数据集的新方法。
   - 另一位用户推测，这可能是因为 **数据集也被删除了**，所以被撤下了。
- **数据集访问问题**：在 Clip 检索出现问题后，人们对数据集的可用性表示担忧。
   - 有人建议数据集的移除影响了 Clip 检索功能。


  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1261051560780365894)** (10 条消息🔥): 

> - `MLP 架构效率`
> - `模型训练中的内存使用`
> - `Nematron 340B 代码示例`
> - `AuraFlow 模型发布公告`
> - `爱丽丝梦游仙境 (AIW) 问题` 


- **单个大型 MLP 优于多个小型 MLP**：对近期一篇论文的讨论表明，多次复用一个大型 MLP 比每个区块拥有各自的小型 MLP 更有效率。
   - *用参数换取 flops* 以及令人惊讶的内存使用亮点引发了对该方法异常高内存需求的关注。
- **小模型的重度内存使用**：成员报告称，在 128 批次大小的 CIFAR-100 上训练一个仅有 25 万参数的模型就消耗了 19GB 内存，称其为*愚蠢的内存低效*。
   - 这引发了关于为什么如此小的模型需要这么多内存的进一步调查。
- **寻找 Nematron 340B 代码示例**：询问关于运行 Nematron 340B 奖励模型的代码示例，特别是关于加载和卸载参数的部分。
- **AuraFlow 模型发布公告**：[Fal AI](https://blog.fal.ai/auraflow/) 宣布推出 AuraFlow，这是一种新型的基于 flow 的文本生成图像模型，有力地回应了“*开源 AI 已死*”的言论。
   - 该模型在遵循提示词（Prompt）方面表现出色，并再次证明了开源社区的韧性。
- **AIW 问题揭示了 LLM 的脆弱性**：关于爱丽丝梦游仙境 (AIW) 问题的最新 ArXiv 论文显示，最先进的 LLM 在处理简单任务时会出现剧烈的崩溃，详见[此提交内容](https://arxiv.org/abs/2406.02061)。
   - 这表明当前的基准测试未能展示这些模型的根本弱点，强调了对更好基准测试和基础模型能力的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.fal.ai/auraflow/">AuraFlow v0.1 介绍：大型修正流模型（Rectified Flow Models）的开源探索</a>：开源 AI 正面临威胁。随着过去一年社区对 AI 模型的兴趣激增，我们注意到新的开源基础模型的开发陷入了停滞。有些人甚至大胆地宣称……</li><li><a href="https://arxiv.org/abs/2406.02061">爱丽丝梦游仙境：简单任务显示出最先进大语言模型中完全的推理崩溃</a>：大语言模型 (LLM) 经常被描述为基础模型的实例——即能够以 few-shot 或 zero-shot 方式在各种任务和条件下进行强力迁移的模型……
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1261349288068382886)** (4 条消息): 

> - `Agentic RAG 指南 (Cookbooks)`
> - `用于实体去重的 Cypher 代码片段`
> - `知识图谱构建挑战`
> - `LlamaCloud 数据流水线管理` 


- **发布 Agentic RAG 指南**：LlamaIndex 宣布与来自 AIatMeta 的 @jeffxtang 合作发布关于 **Agentic RAG** 的指南，涵盖从路由和工具使用到多文档 Agent 构建的主题。
   - 该发布通过预览推文进行了预热，链接至 [Twitter 公告](https://t.co/mBNZx9b1JO) 及[此处](https://t.co/l2ztPRsAd8)的补充内容。
- **Cypher 代码片段简化实体去重**：由 @tb_tomaz 和 Neo4j 提供的 Cypher 代码片段通过结合文本嵌入（Text Embeddings）和单词处理，有效地执行**实体去重**。
   - 共享了详细信息和[示例代码片段链接](https://t.co/dAV2QuAoZH)，以展示其在知识图谱构建中的实用性。更多资源可在 [Neo4j GitHub 仓库](https://t.co/lMApLzMOMr)中找到。
- **自动化知识图谱的挑战**：使用 LLM 自动创建知识图谱面临挑战，特别是关于**重复实体**的问题。
   - @tb_tomaz 和 Neo4j 的其他成员分享了一个展示实际解决方案的酷炫示例，推文链接至[更多信息](https://t.co/ruxdlhZOuK)。
- **LlamaCloud 数据流水线新功能**：LlamaCloud 推出了集中管理数据流水线的功能，适用于任何 **LLM 应用**，包括简单的 RAG 和复杂的工作流。
   - 新功能包括多用户组织管理，更多细节见[公告推文](https://t.co/F73Spljg0a)。



**提到的链接**：<a href="https://t.co/ruxdlhZOuK">blogs/llm/llama_index_neo4j_custom_retriever.ipynb · tomasonjo/blogs</a>：支持我在 https://bratanic-tomaz.medium.com/ 上的图数据科学博客文章的 Jupyter notebooks - tomasonjo/blogs

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1261239555562930177)** (6 messages): 

> - `Gemini 模型上的 Function calling`
> - `Gemin-1.5-flash-latest 模型报错`
> - `更新 vertexai 集成包`
> - `索引大型代码库`
> - `使用 RAG 审阅规格文档` 


- **Gemini 模型上的 Function calling 不明确**：一位成员询问 **llamaindex 是否支持** Gemini 模型上的 **function calling**，并引用了一个 [GitHub pull request](https://github.com/run-llama/llama_index/pull/14088)。
   - 尽管看到了代码，但他们遇到了一个错误，提示 **'Model name models/gemini-1.5-flash-latest does not support function calling API'**。
- **更新 vertexai 集成包以解决问题**：为了解决 Gemini 模型 function calling 的问题，另一位成员建议使用 `pip install -U llama-index-llms-vertexai` **更新 vertexai 集成包**。
- **索引大型代码库的最佳实践**：一位成员就如何为两种不同的聊天机器人/查询索引大型代码库寻求建议：一个用于回答问题，另一个用于代码生成。
   - 他们询问将 **代码翻译成 Markdown 格式的伪代码** 是否有助于 Agent 更好地理解库。
- **使用 RAG 审阅规格文档**：一位成员考虑使用 **RAG** (Retrieval-Augmented Generation) 来审阅规格文档，而不是将所有 2,000 行文本发送给 LLM，旨在节省 Token。



**提及的链接**：<a href="https://github.com/run-llama/llama_index/pull/14088">Enable Function calling and agent runner for Vertex AI by wadave · Pull Request #14088 · run-llama/llama_index</a>：更改描述亮点：为 Vertex AI 启用了 function calling `llama-index-integrations/llms/llama-index-llms-vertex/llama_index/llms/vertex/base.py`，并为 Gemini 添加了 tool/function 角色...

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1261358908543336539)** (6 messages): 

> - `Agent 调用问题`
> - `LiteLLM 错误`
> - `Phi-3 快速 Function Calls`
> - `Open Interpreter GUI 集成` 


- **Agent 选择导致调用错误**：一位用户报告了一个问题，即调用 Agent 的 'chat' 方法（该方法调用 `OpenInterpreter.chat()`）在独立运行时正常，但当 OpenInterpreter 根据角色“选择” Agent 时失败，导致 **APIConnectionError**。
   - *该错误建议显式传递 LLM 提供商。* 了解更多 [点击这里](https://docs.litellm.ai/docs/providers)。
- **Phi-3 Function Calls 速度提升**：一位用户分享了从 `phi-3` 实现快速、可靠的 **function calls** 的兴奋之情，并希望很快能有一个完全本地化的 Fast Fourier 运行选项。
- **Open Interpreter GUI 获得重大升级**：一位用户将 Open Interpreter 集成到了他们的 GUI 中，支持 **分支聊天、可编辑消息、自动运行代码** 以及聊天保存。
   - 该 GUI 还支持各种配置参数，但在 [开源项目](https://github.com/jbexta/AgentPilot) 中披露了一些已知的局限性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>：了解如何在 LiteLLM 上部署和调用来自不同提供商的模型</li><li><a href="https://github.com/jbexta/AgentPilot">GitHub - jbexta/AgentPilot: Universal GUI for seamless interaction and management of AI workflows</a>：用于无缝交互和管理 AI 工作流的通用 GUI - jbexta/AgentPilot
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

notnaton: https://youtu.be/SoFepHI6sQ0?si=2Y1zkghH2XyaN9_k
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1261220652338380872)** (1 messages): 

> - `自托管 ML Telemetry`
> - `Langfuse`
> - `WandB`
> - `OpenLLMTelemetry` 


- **自托管 ML Telemetry 解决方案对比**：[Langfuse](https://langfuse.io)、[WandB](https://wandb.ai) 和 [OpenLLMTelemetry](https://github.com/openllmtelemetry) 都提供 ML telemetry 的自托管解决方案。
   - 这些平台提供了 **ML 项目** 所需的功能，并推荐给寻求自托管选项的用户。
- **Langfuse、WandB 和 OpenLLMTelemetry 特性**：Langfuse、WandB 和 OpenLLMTelemetry 都包含了 **自托管 ML telemetry** 必要的特性。
   - 对这些解决方案感兴趣的用户应根据具体的项目需求和要求进行对比。


  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1261414721802862864)** (2 messages): 

> - `API key for OpenAI`
> - `Chatbot project tutorial` 


- **请求用于 Chatbot 项目的 OpenAI API Key**：一名成员询问是否有人可以分享一个未使用的 OpenAI API Key，用于 Chatbot 项目教程。
   - 他们强调该需求是为了创建教程，表明是临时使用。
- **Chatbot 项目教程需要 API Key**：再次有人请求 OpenAI API Key 以帮助完成 Chatbot 项目教程。
   - 该成员重申，该 Key 仅用于演示目的。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1261040706097516618)** (1 messages): 

> - `Account Credits`
> - `User Query`
> - `Account ID`
> - `Credit Balance Check` 


- **用户咨询额度余额**：一名用户询问如何检查其额度余额，并艾特了另一名用户寻求帮助。
   - 用户提供了其 Account ID 为 **reneesyliu-571636** 以供参考。
- **寻求账户状态帮助**：另一条用户查询提出了无法检查账户状态的问题。
   - 他们附上了账户详情，以便快速解决问题。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/)** (1 messages): 

slac.eth6408: 我们知道 OpenAI 额度过期的日期吗？
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732)** (1 messages): 

> - `Llamafile goes to Washington`
> - `Builders Accelerator`
> - `Upcoming Events`
> - `Open Source AI Definition` 


- **Llamafile 走进华盛顿**：[Mozilla 全球政策总监 Udbhav Tiwari 在美国参议院作证](https://discord.com/channels/1089876418936180786/1260972784696295536)，强调了 AI 技术开放性的必要性。
- **Builders Accelerator 申请仍然开放**：尽管 <@&1229573172018417674> 的早期申请窗口已关闭，但该计划仍在接受申请，正如在 [之前的公告](https://discord.com/channels/1089876418936180786/1089876419926032396/1255588743599882260) 中提到的。
- **即将举行的活动（请预约）**：参加即将举行的活动，例如运行代码 LLM 的 [Open Interpreter](https://discord.com/events/1089876418936180786/1260611047341953034)，由 Benjamin Minixhofer 主持的 [Zero Shot Tokenizer Transfer](https://discord.com/events/1089876418936180786/1260289594729959455)，以及与工程师 <@278455249239539712> 合作的 [AutoFix: Open Source issue fixer](https://discord.com/events/1089876418936180786/1245836053458190438)。
- **开源 AI 定义草案 v 0.0.8**：[开源 AI 定义草案 v 0.0.8](https://opensource.org/deepdive/drafts/the-open-source-ai-definition-draft-v-0-0-8) 已开放征求意见，并遵循 OECD 对 AI 系统的定义。
   - 欲了解更多信息，请访问 [OSI 博客](https://blog.opensource.org/open-source-ai-establishing-a-common-ground/)。



**提到的链接**：<a href="https://opensource.org/deepdive/drafts/the-open-source-ai-definition-draft-v-0-0-8>).">The Open Source AI Definition &#8211; draft v. 0.0.8</a>：版本 0.0.8 请为此文本留下评论。注：本文档由三部分组成：前言，阐述本文档的意图；开源 AI 定义本身；以及一份清单……

  

---


### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1261374633123385425)** (1 messages): 

> - `llama.cpp matmul/matvec`
> - `ggml-quants.c file`
> - `integer dotproducts`
> - `float activations` 


- **关于 llama.cpp Matmul 机制的问题**：一名成员询问 **llama.cpp** 在执行 matmul/matvec 的内部点积时是使用整数还是浮点数，并引用了包含多个整数点积操作的 **ggml-quants.c** 文件。
   - 用户质疑在执行 matmul 之前是否对激活值（activations）进行了量化，因为 **激活值通常是浮点数**，这让他们对处理过程感到好奇。
- **ggml-quants.c 中的浮点数与整数**：在 **ggml-quants.c** 中，注意到了大量的整数点积操作，从而引发了关于实际乘法是否使用浮点数而非整数的疑问。
   - 担忧在于，如果直接使用整数执行 matmul 操作，则需要预先对通常为浮点格式的激活值进行量化。


  

---

### **DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1261436075466035321)** (2 messages): 

> - `LLM Arena`
> - `WizardLM Paper`
> - `OpenArena GitHub Repository` 


- **引入 LLM Arena 以提升数据集质量**：已创建一个 **LLM arena**，让两个语言模型相互对战，并由第三个模型担任裁判，主要使用来自 **Ollama** 的模型，但也兼容任何 **OpenAI** 端点。
   - 该设置旨在通过利用竞争性基准来**提高数据集质量**，详见该项目的 [GitHub 页面](https://github.com/syv-ai/OpenArena)。
- **受 WizardLM 论文启发**：**OpenArena** 基于 [WizardLM 论文](https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/)，该论文提出了 **Arena Learning** 方法，旨在为 LLM 后训练构建高效的数据飞轮。
   - 这涉及模拟迭代的竞技场对战，并利用 **AI 标注的结果**来增强监督微调（Supervised Fine-Tuning）和强化学习中的模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/">Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena - Microsoft Research</a>: Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena</li><li><a href="https://github.com/syv-ai/OpenArena">GitHub - syv-ai/OpenArena</a>: 通过在 GitHub 上创建账户，为 syv-ai/OpenArena 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1261180269172363345)** (1 messages): 

> - `Product coverage`
> - `Research coverage`
> - `Recommendation systems`
> - `Information Retrieval`
> - `Retrieval-Augmented Generation` 


- **关于覆盖多个领域的讨论**：一位用户表示有兴趣为多个群体覆盖**产品**和**研究**主题，例如**推荐系统**、**信息检索 (IR)** 和**检索增强生成 (RAG)**。
   - 如果有人有建议，他们乐于接受，并渴望与另一位用户讨论 **Elastic**。
- **对 Elastic 的兴趣**：一位用户特别提到，如果其他人感兴趣，想聊聊 **Elastic**。
   - 他们标记了另一位成员并邀请其进行进一步讨论。



{% else %}


> 完整的各频道细分内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整的细分内容，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}