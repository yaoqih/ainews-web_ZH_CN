---
companies:
- openai
- deepseek-ai
- meta-ai-fair
- huggingface
- cognition-labs
- hyperbolic
- google-deepmind
date: '2024-12-11T04:20:02.638516Z'
description: '**OpenAI** 向所有用户推出了 **ChatGPT Canvas**，具备**代码执行**和 **GPT 集成**功能，通过类似
  Google Docs 的界面有效地取代了原有的代码解释器（Code Interpreter）。**Deepseek AI** 发布了 **V2.5-1210**
  更新，提升了在 **MATH-500 (82.8%)** 和 LiveCodebench 上的性能表现。**Meta AI Fair** 推出了 **COCONUT**，这是一种全新的连续潜空间推理范式。**Huggingface**
  发布了 **TGI v3**，在处理长提示词时，其处理的 **Token 数量是 vLLM 的 3 倍**，运行速度快 **13 倍**。**Cognition
  Labs** 发布了 **Devin**，这是一款能够构建 Kubernetes 算子（operators）的 AI 开发者工具。**Hyperbolic**
  完成了 **1200 万美元 A 轮融资**，旨在构建一个带有 **H100 GPU 交易市场**的开放 AI 平台。


  讨论内容涵盖了 **AI 能力及其对就业的影响**，以及 **NeurIPS 2024** 的相关发布，包括 **Google DeepMind** 的演示和关于
  AI 缩放（scaling）的辩论。在 Reddit 上，**Llama 3.3-70B** 支持使用 **Unsloth** 进行 **90K 上下文长度**的微调，该技术结合了**梯度检查点（gradient
  checkpointing）**和苹果的 **Cut Cross Entropy (CCE)** 算法，仅需 **41GB 显存**即可运行。此外，**Llama
  3.1-8B** 通过 Unsloth 达到了 **342K 上下文长度**，超越了其原生限制。'
id: 6ab1afd9-3227-4381-8d83-c63e4833e11c
models:
- llama-3-70b
- llama-3-1-8b
- tgi-v3
- deepseek-v2.5-1210
- coconut
original_slug: ainews-chatgpt-canvas-ga
people:
- arav_srinivas
- sama
- jonathan-frankle
- dylan
title: '**ChatGPT Canvas 全面开放** (或 **正式发布**)


  注：“GA” 是 **General Availability** 的缩写，在软件行业意指产品结束测试阶段，向所有用户正式开放。'
topics:
- code-execution
- gpt-integration
- model-finetuning
- gradient-checkpointing
- context-length
- latent-space-reasoning
- performance-optimization
- gpu-memory-optimization
- kubernetes
- gpu-marketplace
- ai-capabilities
- employment-impact
- neurips-2024
- ai-scaling
- humor
---

<!-- buttondown-editor-mode: plaintext -->**Karina Nguyen is all you need.**

> 2024年12月9日至12月10日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discord（**206** 个频道，**5518** 条消息）。预计节省阅读时间（以 200wpm 计算）：**644 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

虽然现在还处于早期阶段，但我们已经可以宣布 OpenAI 的 12 Days of Shipmas 活动取得了成功。尽管昨天的 Sora 发布（截至今天）仍因需求过大而受限于受限注册，但 ChatGPT Canvas 无需额外的 GPU，并于今天向所有免费和付费用户[发布](https://www.youtube.com/live/qZ0ImE41pVs?si=rUe6uWNbdYgXsSiJ)，且运行顺畅。

[
![image.png](https://assets.buttondown.email/images/f9af6291-0bd1-44fa-8998-9094819e9a4a.png?w=960&fit=max)
](https://www.youtube.com/live/qZ0ImE41pVs?si=rUe6uWNbdYgXsSiJ)

Canvas 现在实际上取代了 Code Interpreter，而且非常像 Google Docs，这进一步证明了 OpenAI 构建 Google 功能的速度比 Google 构建 OpenAI 功能的速度还要快。

有一种[理论](https://x.com/scaling01/status/1866549472236331390)认为，每集结尾的笑话都是下一集内容的预告。如果这是真的，那么明天的发布将会非常震撼。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 总结

> 所有总结均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

以下是关键 Twitter 讨论的分类摘要：

**AI 模型与研究更新**

- [@deepseek_ai 宣布](https://twitter.com/deepseek_ai/status/1866459740324458835)了他们的 V2.5-1210 更新，在 MATH-500 (82.8%) 和 LiveCodebench (34.38%) 上**性能有所提升**
- [Meta 介绍了](https://twitter.com/iScienceLuvr/status/1866353795502158163) **COCONUT (Chain of Continuous Thought)**，这是一种使用连续潜空间（latent space）进行 LLM 推理的新范式
- [@Huggingface 发布了 TGI v3](https://twitter.com/narsilou/status/1866423560799158775)，在处理长 Prompt 时，其 **Token 处理量提升了 3 倍**，运行速度比 vLLM **快 13 倍**

**产品发布与更新**

- [OpenAI 向所有用户发布了 Canvas](https://twitter.com/OpenAI/status/1866578914233159928)，具有**代码执行**、**GPT 集成**和改进的写作工具等功能
- [@cognition_labs 发布了 Devin](https://twitter.com/togethercompute/status/1866591946820489586)，这是一个 AI 开发者，它成功构建了一个带有测试环境的 **Kubernetes operator**
- [Hyperbolic 完成了 1200 万美元 A 轮融资](https://twitter.com/Yuchenj_UW/status/1866514943815880847)，旨在构建一个开放的 AI 平台，提供价格为 0.99 美元/小时的 **H100 GPU 市场**

**行业与市场分析**

- [@AravSrinivas 分享了](https://twitter.com/AravSrinivas/status/1866374722713522336)美国与加拿大的人均 GDP 对比，获得了 **72,622 次展示**
- [@sama 指出](https://twitter.com/sama/status/1866332878499623098) **严重低估了** Sora 的需求，正在努力扩大访问权限
- 关于 [AI 能力和就业](https://twitter.com/ajeya_cotra/status/1866609233984434455)在未来几十年影响的讨论

**NeurIPS 会议**

- 多位研究人员和公司宣布参加在温哥华举行的 NeurIPS 2024
- [@GoogleDeepMind 举办了](https://twitter.com/GoogleDeepMind/status/1866627004323201248) GenCast 天气预报和其他 AI 工具的演示
- [计划举行一场辩论](https://twitter.com/dylan522p/status/1866630813074461060)，由 Jonathan Frankle 和 Dylan 讨论 AI Scaling 的未来

**迷因与幽默**

- [关于 ChatGPT Canvas 功能的笑话](https://twitter.com/sama/status/1866555731149045990)
- 关于[模型能力和局限性](https://twitter.com/teortaxesTex/status/1866642678823137585)的幽默

---

# AI Reddit 总结

## /r/LocalLlama 总结

**主题 1. Llama 3.3-70B 微调：在小于 41GB VRAM 上实现 90K 上下文**

- **Llama 3.3 (70B) Finetuning - 现支持 90K 上下文长度且适配 <41GB VRAM。** ([Score: 360, Comments: 63](https://reddit.com/r/LocalLLaMA/comments/1hbaioc/llama_33_70b_finetuning_now_with_90k_context/)): **Llama 3.3 (70B)** 现在可以使用 **Unsloth** 进行微调，以支持 **90,000 上下文长度**，这显著长于 Hugging Face + FA2 在 80GB GPU 上支持的 **6,900** 上下文长度。这一改进是通过 **梯度检查点 (gradient checkpointing)** 和 Apple 的 **Cut Cross Entropy (CCE) 算法**实现的，模型仅需 **41GB VRAM** 即可运行。此外，**Llama 3.1 (8B)** 使用 Unsloth 可以达到 **342,000 上下文长度**，远超其原生支持的 **128K 上下文长度**。
  - **Unsloth** 使用 **梯度检查点 (gradient checkpointing)** 将激活值卸载到系统 RAM，从而节省 **10 到 100GB** 的 GPU 显存；Apple 的 **Cut Cross Entropy (CCE)** 在 GPU 上执行交叉熵损失计算，减少了对大型 logits 矩阵的需求，进一步节省了内存。这使得模型能够适配 **41GB VRAM**。
  - 用户对测试中使用的 **rank** 以及 **多 GPU 支持**（目前不可用但在开发中）感到好奇。还有人对使 **Unsloth** 兼容 **Apple 设备**感兴趣。
  - **Unsloth** 工具因其使微调能力平民化、让公众能够使用先进技术，并可能通过允许使用较小的 **48GB GPU** 来降低成本而受到称赞。


- **Hugging Face 发布 Text Generation Inference TGI v3.0 - 在长提示词上比 vLLM 快 13 倍 🔥** ([Score: 347, Comments: 52](https://reddit.com/r/LocalLLaMA/comments/1hayqkt/hugging_face_releases_text_generation_inference/)): **Hugging Face** 发布了 **Text Generation Inference (TGI) v3.0**，在长提示词上处理的 **token 数量增加了 3 倍**，速度比 **vLLM** 快 **13 倍**，且无需任何配置。通过优化内存使用，单个 **L4 (24GB)** 可以在 **llama 3.1-8B** 上处理 **30k tokens**，而 **vLLM** 仅能处理 **10k**，新版本将长提示词的回复时间从 **27.5s** 缩短至 **2s**。[基准测试详情](https://huggingface.co/docs/text-generation-inference/conceptual/chunking)可供查阅。
  - **TGI v3.0 性能**：讨论强调了 **TGI v3.0** 相比 **vLLM** 的显著速度提升，特别是在处理长提示词方面，这得益于 **缓存提示词处理 (cached prompt processing)** 的实现。该库通过保留初始对话数据，可以实现几乎瞬时的响应，查找开销仅约为 **5 微秒**。
  - **比较与使用场景**：用户对 **TGI v3** 与 **TensorRT-LLM**、**ExLlamaV2** 等其他模型的比较表现出兴趣，并询问了其在短查询和多 GPU 设置下的性能。还有人好奇 TGI 是否适用于单用户与多用户场景，一些用户认为它在为多用户托管模型方面进行了优化。
  - **支持与文档**：针对目前文档仅列出企业级 Nvidia 加速器的问题，用户提出了关于是否支持 **RTX 3090** 等消费级显卡的疑问。此外，用户还关注添加流式工具调用 (streaming tool calls) 等功能的路线图，以及与 **fp16** 处理相比是否存在潜在的输出质量下降。


**主题 2. DeepSeek V2.5-1210：最终版本及后续计划**

- **[deepseek-ai/DeepSeek-V2.5-1210 · Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2.5-1210)** ([Score: 170, Comments: 11](https://reddit.com/r/LocalLLaMA/comments/1hay0qy/deepseekaideepseekv251210_hugging_face/)): 该帖子宣布在 **Hugging Face** 上发布 **DeepSeek V2.5-1210**，标志着该 AI 工具的新版本发布，但未说明具体改进。帖子中未提供关于该版本的更多细节。
  - **DeepSeek V2.5-1210** 已被确认为 v2.5 系列的最终版本，未来预计将推出 **v3 系列**。更新日志显示数学性能（在 MATH-500 基准测试中从 **74.8% 提升至 82.8%**）和编程准确率（在 LiveCodebench 基准测试中从 **29.2% 提升至 34.38%**）有显著提升，同时增强了文件上传和网页摘要的用户体验。
  - 用户对 **R1** 模型表现出浓厚兴趣，希望它能尽快发布。一些人推测当前版本可能使用了 R1 作为教师模型进行训练，还有人期待包含 **32B 选项**的更新版 **Lite** 版本。
  - 社区正在积极讨论通过 *exo* 发布 **量化版本** 的可能性，并表达了对进一步更新的渴望，包括 **R1 Lite** 版本。

- **DeepSeek-V2.5-1210: DeepSeek V2.5 的最终版本** ([Score: 147, Comments: 36](https://reddit.com/r/LocalLLaMA/comments/1hb0xau/deepseekv251210_the_final_version_of_the_deepseek/)): **DeepSeek-V2.5-1210** 标志着 **DeepSeek V2.5 系列**的最终版本，自 5 月开源发布以来，经过五次迭代，该系列的开发已告一段落。团队目前正专注于开发下一代基础模型 **DeepSeek V3**。
  - **硬件要求与限制**：以 **BF16 格式**运行 **DeepSeek-V2.5** 需要大量资源，具体为 **80GB*8 GPU**。用户对软件优化的缺乏表示担忧，特别是 **kv-cache** 方面，这限制了模型在现有硬件上相对于 **Llama** 等其他模型的性能表现。
  - **模型性能与能力**：用户注意到该模型具有深层推理能力，但批评其推理速度较慢。尽管如此，**DeepSeek** 模型仍被视为其他大语言模型的高质量替代方案，其采用 **Mixture of Experts (MoE) 结构**，拥有约 **220 亿激活参数**，允许在 CPU+RAM 上实现合理的性能。
  - **开发与发布频率**：**DeepSeek** 团队保持了令人印象深刻的发布节奏，自 5 月以来几乎每月都有更新，这表明其训练过程非常成功。然而，由于创始人更看重研究而非商业应用，这些模型缺乏**视觉理解**能力，主要专注于文本。


**主题 3. InternVL2.5 发布：视觉基准测试中的顶级表现**

- **InternVL2.5 发布（1B 到 78B）在 X 上引发热议。它能取代 GPT-4o 吗？你目前的体验如何？** ([Score: 131, Comments: 42](https://reddit.com/r/LocalLLaMA/comments/1havpua/internvl25_released_1b_to_78b_is_hot_in_x_can_it/)): **InternVL2.5** 涵盖了从 **1B 到 78B** 参数的模型，现已发布并在 **X** 上受到关注。**InternVL2.5-78B** 模型因成为首个在 **MMMU 基准测试**中获得超过 **70%** 分数的开源 MLLM 而备受瞩目，其性能可与 **GPT-4o** 等领先的闭源模型相媲美。你可以通过 [InternVL Web](https://internvl.intern-ai.org.cn/)、[Hugging Face Space](https://huggingface.co/spaces/OpenGVLab/InternVL) 和 [GitHub](https://github.com/OpenGVLab/InternVL) 等多个平台探索该模型。
  - **视觉基准测试讨论**：关于 **InternVL2.5-78B** 模型在视觉基准测试中的有效性存在争议，一些用户认为 **4o** 在视觉任务中优于 **Sonnet**。人们对基准测试的可靠性和模型声明的可信度提出了担忧，特别是考虑到 Reddit 和 **Hugging Face** 上一些令人质疑的历史记录。
  - **地缘政治与教育背景**：讨论涉及全球 STEM 领域格局，特别是美国和中国的对比，强调了中国的 STEM 博士数量和教育成就。一条评论提到一名 **11 岁的中国孩子**制造火箭，引发了关于此类声明的准确性和背景的辩论。
  - **模型可用性与性能**：用户赞赏 **InternVL2.5** 除了 78B 参数模型外，还提供了更小版本的模型，并注意到它们强劲的性能和本地部署的潜力。78B 模型在**乌克兰语**和**俄语**方面的表现被认为优于其他开源模型。

## 其他 AI 子版块回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. Google Willow：量子计算的巨大飞跃**

- **Google Willow：量子计算芯片在 5 分钟内完成最强超级计算机需耗时 1 亿亿亿年（septillion years）的任务** ([Score: 311, Comments: 77](https://reddit.com/r/OpenAI/comments/1havzgs/google_willow_quantum_computing_chip_completes/))：**Google** 推出了 **Willow 量子计算芯片**，实现了比最快超级计算机 **Frontier** 快约 **10^30 倍**的计算速度，在 **5 分钟**内完成了原本需要 **1 亿亿亿年**才能完成的任务。这被认为是今年最重要的技术发布，更多信息可在 [YouTube 视频](https://youtu.be/3msqpkfF0XY)中查看。
  - 几位评论者（如 **beermad**）对 **基准测试（benchmark tests）** 表示怀疑，认为这些测试是为量子计算机优化的，缺乏实际应用价值。他们认为这些测试旨在让量子芯片优于传统计算机，而没有展示实际应用。
  - **huffalump1** 强调了 Google 在 **纠错（error correction）** 方面取得突破的重要性，该突破超越了 **qubit** 的物理极限。这对于量子计算至关重要，因为纠错是该领域的主要挑战。
  - 讨论还涉及了潜在的财务影响，**bartturner** 指出 **GOOG 股价上涨了 4%**（盘前交易），表明投资者认可了这一技术进步的潜在价值。


- **[OpenAI Sora 对比开源替代方案 - Hunyuan（如图）+ Mochi & LTX](https://v.redd.it/42b9chytny5e1)** ([Score: 204, Comments: 56](https://reddit.com/r/StableDiffusion/comments/1hav4z3/openai_sora_vs_open_source_alternatives_hunyuan/))：该帖子讨论了 **OpenAI Sora** 与 **Hunyuan**、**Mochi** 和 **LTX** 等开源替代方案在 **量子计算与传统超级计算机性能** 背景下的对比。由于缺乏视频中的更多细节，未提供这些对比或性能指标的具体信息。
  - 评论者讨论了 **OpenAI Sora 与开源模型**（如 **HunyuanVideo**）之间的比较，指出开源选项具有竞争力且通常更易于获取。**HunyuanVideo** 因其潜力和在消费级 **GPU** 上运行的能力而受到关注，一些用户因其不受审查的特性而更倾向于开源。
  - **Sora 的性能** 在某些领域（如细节图像和风景）的质量受到称赞，但在可访问性受限和物理交互问题方面面临批评。用户注意到 **HunyuanVideo** 在某些场景下表现更好，并对与 **TemporalPromptEngine** 等模型的进一步对比感兴趣。
  - 有人呼吁 **西方改进其开源 AI 努力**，因为中国开源模型在质量和在消费级硬件上运行的能力方面令人印象深刻。这种情绪反映了西方国家对更开放、更易获取的 AI 开发的渴望。


- **[我预见广告商和零售商很快就会被起诉。这对我来说是一个真实的广告。](https://i.redd.it/3pml08a3h06e1.jpeg)** ([Score: 177, Comments: 50](https://reddit.com/r/ChatGPT/comments/1hb07p5/i_foresee_advertisers_and_retailers_getting_sued/))：该帖子讨论了一个使用 **卡通风格插图** 和 **争议性主题** 来吸引注意力的广告，这可能会给广告商和零售商带来法律问题。广告中使用了幽默手段，涉及一只 **看起来很悲伤的猫** 和 **漂白剂（bleach）**，这引发了伦理担忧，并凸显了创意营销与误导性或冒犯性内容之间的微妙界限。
  - 评论者对广告的意图表示怀疑和幽默，**SomeRandomName13** 讽刺地指出该广告在猫的眼睛上使用 **漂白剂** 来解决问题的荒谬性，强调了广告的争议性。
  - **j-rojas** 等人认为广告的设计是故意荒谬的，以充当 **点击诱饵（clickbait）**，因其极端的荒谬感而激发好奇心。
  - **chrismcelroyseo** 警告混合 **漂白剂和氨水** 的危险，并提供了一个关于毒性影响的文章[链接](https://www.verywellhealth.com/mixing-bleach-and-ammonia-1298711)，这强调了与广告内容相关的潜在现实风险。


**主题 2. Gemini 1.5 表现优于 Llama 2 70B：行业反应**

- **[说实话，与 ChatGPT 不同，我们中 97% 的人一个月后就不会再关心或使用 Sora 了。我尝试过所有的视频生成器和音乐生成器，新鲜感也就持续一周，然后就觉得没意思了。很有可能在你每月的 50 次生成额度用完后，你就会忘记 Sora。](https://reddit.com/r/ChatGPT/comments/1hawku5/lets_be_honest_unlike_chatgpt_97_of_us_wont_care/)** ([Score: 302, Comments: 113](https://reddit.com/r/ChatGPT/comments/1hawku5/lets_be_honest_unlike_chatgpt_97_of_us_wont_care/)): **Gemini 1.5** 被讨论为超越了 **Llama 2 70B**，但作者对 **Sora** 的持久影响力表示怀疑。他们认为，与视频和音乐生成器等其他 **AI** 工具类似，**Sora** 最初可能很吸引人，但在有限的使用后，很可能会被大多数用户遗忘。
  - 用户讨论了 **Sora** 在专业环境中的长期实用性，一些人认为工作室将采用它来降低成本，而另一些人则认为它是一个价格过高的新奇事物，由于 5 秒、720p 输出等限制，其用途有限。**Sora** 被比作过去最初引起关注但随后淡出的技术趋势，类似于 **Suno** 和 **DALL-E 3** 等其他 **AI** 工具，这些工具在使用量激增后便出现了下滑。
  - 一些评论者强调了 **AI** 工具在改变工作流方面的重要性，并列举了视频相关业务的例子，在过去六个月中，**AI** 显著影响了这些业务的运营。尽管存在怀疑，但其他人指出，即使 **AI** 工具没有普遍吸引力，它们对于特定的专业任务仍然具有价值。
  - 讨论还涉及了 **AI** 工具在最初炒作之外的更广泛适用性，并将其与 **Apple Vision** 和早期 Web 浏览器等历史技术采用模式进行了类比。共识是，虽然像 **Sora** 这样的工具可能并非普遍必需，但它们对利基市场和特定的专业用途具有重大价值。


- **[艺术画廊正在出售 AI 艺术品？](https://www.reddit.com/gallery/1hbb4qx)** ([Score: 285, Comments: 181](https://reddit.com/r/ChatGPT/comments/1hbb4qx/ai_art_being_sold_at_an_art_gallery/)): 作者描述了参加一次艺术画廊活动的情况，他们怀疑其中两幅标价在 **5,000 到 15,000 欧元** 之间的画作可能是 **AI** 生成的，因为作品中存在一些奇特之处，例如扭曲的手和不合逻辑的提包手柄。他们联系了组织者以调查 **AI** 参与的可能性，目前正在等待进一步消息。
  - 许多评论者怀疑这些画作是 **AI** 生成的，因为存在手指畸形、荒谬的提包手柄和奇怪的房间布局等异常情况。**Fingers**（手指）和 **dog features**（狗的特征）经常被引用为 **AI** 的破绽，一些用户指出，将这些作为人类艺术家的风格是荒谬的。
  - **SitDownKawada** 提供了这些以约 **4,000 欧元** 出售的画作链接，并质疑该艺术家在线存在的真实性，其内容看起来像是 **AI** 生成的。该艺术家的 **Instagram** 和其他社交媒体账号因其近期的活动和多产的产出而受到了审查。
  - 讨论还涉及了 **AI** 在艺术领域的更广泛影响，一些用户思考手绘作品中的 **AI** 生成元素是否仍应被视为 **AI** 艺术。关于媒介还是创作过程更具价值存在争论，尤其是随着 **AI** 变得与人类艺术创作越来越难以区分。


- **[你会使用 ChatGPT 或任何其他 AI 搜索来进行产品推荐或寻找新产品吗？如果是，请说明你信任 AI 的哪类产品推荐](https://reddit.com/r/ChatGPT/comments/1hb9a1r/do_you_use_chatgpt_or_any_other_ai_search_for/)** ([Score: 255, Comments: 18](https://reddit.com/r/ChatGPT/comments/1hb9a1r/do_you_use_chatgpt_or_any_other_ai_search_for/)): **Gemini 1.5** 正在被讨论其在提供 **AI 驱动的产品推荐** 方面的潜力。社区被鼓励分享关于他们是否信任 **AI** 推荐来发现新产品的经验，重点关注那些由 **ChatGPT** 等 **AI** 工具建议时更可靠的特定类型产品。
  - 一位用户通过 **AI** 推荐发现了一款名为 **"Vagrus - The Riven Realms"** 的游戏，强调了 **AI** 在产品发现中的作用，他们以前从未听说过这款游戏，但觉得非常出色。这突显了 **AI** 在建议可能符合用户兴趣的冷门产品方面的潜力。
  - 对于具有 **hard specifications**（硬性规格）的产品（如计算机硬件和电子产品），对 **AI** 推荐的信任度往往更高。用户发现 **AI** 在比较技术细节方面特别有用，否则这需要大量的各种手动研究，例如一位用户使用 **AI** 来比较路由器。


**Theme 3. Sora Video Generator: Redefining AI Creativity**

- **[用 Sora 制作的 Cortana](https://v.redd.it/turp1pafvz5e1)** ([评分: 383, 评论: 42](https://reddit.com/r/ChatGPT/comments/1hayghd/cortana_made_with_sora/)): **Sora** 被强调为一种增强 **AI video generation** 技术的工具，如帖子中链接的视频所示。该视频展示了一个名为 **Cortana** 的作品，尽管文中未提供有关视频内容或所用技术的具体细节。
  - 讨论中提到 **Sora** 是一个生成视频而非文本的工具，一些用户询问目前公众是否可以访问 Sora。
  - 评论中包含对 **Cortana** 外观的幽默和讽刺，用户提到了 "jiggle physics" 等特征，并对她的设计发表了轻松的评论。
  - 一些评论关注技术层面和视觉设计，要求增加颜色和皮肤等额外功能，并对 **Cortana** 的“理线 (wire management)”开起了玩笑。


- **[猪在夜间跳探戈](https://v.redd.it/v6utrksp1y5e1)** ([评分: 373, 评论: 61](https://reddit.com/r/ChatGPT/comments/1hasyrj/pigs_tango_in_the_night/)): 作者使用 **Sora** 制作了一个视频，配上他兄弟在 **Suno** 中创作的幽默歌曲。他在制作过程中耗尽了额度，但认为这是对该技术的一次成功的初步尝试，并计划可能在一个月后发布另一个视频。
  - **Sora 的可访问性和订阅**: 用户讨论了通过 20 美元的订阅访问 **Sora** 的情况，该订阅允许每月创建 50 个 5 秒钟的剪辑，并将其价值与其他生成器进行了比较。用户赞赏这一功能，但也注意到了额度的限制，一名用户本月的额度已用完。
  - **Prompt 理解和 remix 功能**: 关于 **Sora** 如何解释 prompt 的讨论显示，用户描述了他们想要的每个场景，并使用 "remix" 功能在剪辑不符合预期时进行调整。一位用户提到由于大量的 remix 操作而耗尽了额度。
  - **Sora 的表现和用户反馈**: 反馈强调了 **Sora** 在生成舞蹈动作方面的能力，一些用户称赞其输出优于其他模型。然而，对于内容的关联性反应不一，例如对探戈音乐的期望未能得到满足。


---

# AI Discord 摘要

> 由 O1-preview 提供的摘要之摘要的总结

**主题 1: AI 模型进展与新发布**

- [**Gated DeltaNet 抢占风头**](https://arxiv.org/abs/2412.06464): Gated DeltaNet 在长上下文任务中表现优于 Mamba2 等模型，利用门控记忆控制和 delta 更新来解决标准 Transformer 的局限性。这一进步显著提高了复杂语言建模场景中的任务准确性和效率。
- [**Llama 3.3 突破上下文障碍**](https://unsloth.ai/blog/llama3-3): Unsloth 现在支持在 **80GB GPU** 上微调上下文长度高达 **89,000 tokens** 的 Llama 3.3，通过减少 **70%** 的 VRAM 使用量来提高效率。这使得在 A100 GPU 上**每个训练步骤仅需 2-3 分钟**，大大超过了以前的能力。
- [**DeepSeek V2.5 发布“压轴作品”**](https://huggingface.co/deepseek-ai/DeepSeek-V2.5-1210): DeepSeek 宣布发布 **DeepSeek-V2.5-1210**，在其[聊天平台](https://chat.deepseek.com/)中加入了实时 **Internet Search**，为用户提供触手可及的实时答案。

**主题 2: AI 工具与用户体验挑战**

- **Cursor 休息片刻**: 用户报告 **Cursor** 中持续存在**请求缓慢**的问题，尽管最近对 **Composer** 和 **Agent** 模式进行了更新，但仍干扰了生产力。这两种模式的表现依然不佳，对编码工作流产生了负面影响。
- **Bolt 遇到减速带**: **Bolt.new** 用户在订阅结束时的 **token 分配**上感到困惑，token 不会叠加，且每 **30 天**重置一次。**图像上传**问题和“**No Preview Available**”错误进一步挫伤了用户的积极性，引发了关于 token 管理策略的讨论。
- **Cursor 中的 Linting 噩梦**: **Cursor** 的 **linting 功能**在没有实际错误的情况下触发，导致用户不必要地消耗了他们的快速消息配额。频繁的误报强化了这样一种观点，即 Cursor 的功能仍处于 beta 阶段，需要改进。

**主题 3: 软件开发中的 AI 集成**

- **Mojo 通过新关键字打破旧习惯**：在 Mojo 中引入 **`destroy`** 关键字强化了线性类型（linear types）中更严格的内存管理，在增强安全性的同时也引发了关于新手复杂性的辩论。这一与 Python **`del`** 的区别旨在改进编程实践。
- **Aider 通过多实例成倍提高生产力**：工程师们正同时运行多达 **20 个 Aider 实例**来处理大型项目工作流，展示了该工具的可扩展性。用户正在探索跨实例的命令执行，以优化大规模开发的编码方法。
- **LangChain 和 Aider 成为黄金搭档**：**Aider 与 LangChain 的 ReAct 循环集成**增强了项目管理任务，用户注意到其效果优于其他工具。这种协作提升了 AI 辅助编码的工作流和效率。

**主题 4：AI 社区与开源倡议**

- [**vLLM 加入 PyTorch 阵营**](https://pytorch.org/blog/vllm-joins-pytorch/)：**vLLM** 正式集成到 **PyTorch** 生态系统中，增强了 LLM 的高吞吐量、内存高效推理。此举预计将推动 AI 创新并降低开发者的门槛。
- [**Grassroots Science 走向多语言**](https://grassroots.science/)：一项新倡议旨在通过开源努力和社区协作，在 **2025 年 2 月**前开发出**多语言 LLM**。该项目寻求利用开源工具吸引基层社区参与多语言研究。
- [**2024 年 AI Agent 现状报告发布**](https://x.com/mrahmadawais/status/1866483416981786821)：[Ahmad Awais](https://x.com/mrahmadawais) 发布了一份深度报告，分析了 **1840 亿个 token** 和来自 **4,000 名开发者**的反馈，突出了 AI Agent 的趋势和未来方向。

**主题 5：创意内容与用户交互中的 AI**

- [**NotebookLM 在播客领域表现出色**](https://youtu.be/aG0ixD3OY80)：用户分享了名为《NotebookLM 播客教程：10 个秘密提示词（绝密！）》的教程，提供专属提示词以增强播客创意。尝试使用 **fact-checkers** 等功能可以提高 AI 生成播客的对话质量。
- [**WaveForms AI 为交互注入情感**](http://waveforms.ai)：**WaveForms AI** 发布，旨在通过将**情感智能（Emotional Intelligence）**集成到 AI 系统中来通过**语音图灵测试（Speech Turing Test）**。这一进步致力于通过更自然、更具表现力的沟通来增强人机交互。
- [**Sora 的首次亮相毁誉参半，引发用户猜测**](https://www.youtube.com/embed/FSMjXW_KrAo)：OpenAI 的 **Sora** 因 5 秒的视频输出和内容质量问题而受到质疑。用户将其与 **Claude**、**Leonardo** 和 **Ideogram** 等模型进行对比，结果并不理想，导致部分用户更倾向于替代方案。


---

# 第一部分：Discord 高层摘要




## [Codeium / Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf AI 启动周边赠送活动**：**Windsurf AI** 在 Twitter 上发起了首次[周边赠送活动](https://x.com/windsurf_ai/status/1866600392165048329)，邀请用户分享他们的**作品**，以赢取护理礼包。
   - 该活动利用标签 **#WindsurfGiveaway** 来追踪提交内容并提升社区参与度。
- **积分系统持续存在缺陷**：用户报告购买的 **credits** 经常无法在账户中显示，导致广泛的不满和大量的支持工单。
   - 尽管团队做出了保证，但缺乏及时的支持响应继续让用户群感到失望。
- **对 Windsurf 定价模式的困惑**：用户对 **Windsurf 的定价**日益担忧，尤其是相对于所提供的功能，**flow** 和常规积分的限制较高。
   - 用户主张建立更可持续的模式，包括引入未用积分的**结转系统（rollover system）**。
- **Windsurf IDE 性能下降**：最近的更新导致了对 **Windsurf IDE** 的批评，用户指出其 **bugs** 增加且**效率**下降。
   - 与 **Cline** 等竞争对手的对比显示，用户更青睐 Cline 卓越的**功能性**和**可靠性**。
- **Cline 在编程任务中表现优于 Windsurf**：在某些**编码任务**中，**Cline** 比 **Windsurf** 更受青睐，尽管在某些方面性能较慢，但能提供更好的 **prompt 响应**。
   - Cline 生成特定**编码输出**且无错误的能力受到了社区的特别称赞。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLM 中的可复现性挑战**：讨论强调了 Large Language Models (LLM) 中的**可复现性担忧**，特别是在医疗系统等高风险应用中，强调了超越传统软件开发的复杂性。成员们辩论了重建 LLM 的细微差别以及可靠 Benchmark 的重要性。
   - 参与者提到了待评审的 [HumanEval Benchmark PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1992)，该 PR 旨在通过集成来自 HF evaluate 模块的 pass@k 指标来增强评估标准。
- **Coconut Architecture 对比 Universal Transformers**：**Coconut Architecture** 引入了一种新颖的方法，将 `<eot>` token 后的最终隐藏状态作为新 token 反馈，在每次迭代中改变 KV cache。这与 **Universal Transformers** 形成对比，后者通常在重复过程中保持静态 KV cache。
   - 讨论了该方法在特定条件下与 UTs 的潜在相似性，特别是在涉及共享 KV cache 和状态历史管理的场景中，突出了性能优化的机会。
- **Gated DeltaNet 提升长上下文性能**：**Gated DeltaNet** 在长上下文任务中表现出优于 **Mamba2** 和之前 DeltaNet 版本的性能，利用了门控内存控制和 delta 更新。这一进步解决了标准 Transformers 在长期依赖方面的局限性。
   - 引用了 Benchmark 结果，展示了在任务准确性和效率方面的显著提升，使 Gated DeltaNet 在复杂的语言建模场景中成为具有竞争力的架构。
- **Batch Size 影响 GSM8k 评估准确率**：在 **GSM8k benchmark** 上的评估显示，**1** 的 Batch Size 达到了 **85.52%** 的最高准确率，而较大的 Batch Size 导致性能显著下降。这种差异可能与 padding 或 attention 机制的实现有关。
   - 成员们正在调查根本原因，考虑调整 padding 策略和模型配置，以减轻增加 Batch Size 对评估指标的负面影响。
- **RWKV 和 Transformers 中的 Attention Masking 问题**：针对 **RWKV model** 的实现提出了担忧，特别是与 attention masking 和 left padding 相关的问题，这可能会对评估结果产生不利影响。此外，在多 GPU 环境中使用 SDPA attention 实现被指出可能存在性能不一致。
   - 参与者强调了仔细配置的必要性，并建议探索替代的 attention backends，以确保在不同硬件设置下模型性能的可靠性。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 运行极其缓慢**：多位用户报告 **Cursor** 中持续出现**请求缓慢**的问题，尽管最近对 **Composer** 和 **Agent** 模式进行了更新，但仍干扰了他们的生产力。
   - 用户认为 **Composer** 和 **Agent** 模式的表现依然不尽如人意，对他们的编程工作流产生了负面影响。
- **AI 模型对决：Gemini, Claude, Qwen**：许多用户更青睐 **Claude**，认为其在编程任务中的表现优于 **Gemini** 和 **Qwen**。
   - 虽然 **Gemini** 在某些测试中显示出潜力，但质量的不稳定性导致了开发者的挫败感。
- **Agent 模式文件处理困惑**：关于 **Cursor** 的 **Agent mode** 中的 Agent 是直接访问文件内容还是仅仅建议读取文件，产生了一些疑问。
   - 这种不确定性突显了用户对 **Cursor** 的 Agent 功能之功能性和可靠性的持续关注。
- **AI 称赞用户的代码结构**：一位用户分享了反馈，尽管该用户缺乏经验，但 **AI** 称赞其代码结构非常专业。
   - 这展示了当前 **AI** 在准确评估开发实践方面的先进能力。
- **Linting 触发令用户沮丧**：**Cursor** 的 **linting 功能**在没有实际错误的情况下触发，导致用户感到沮丧，他们认为自己的 fast message 配额被误用了。
   - 频繁的误报强化了这样一种观点：**Cursor** 的功能仍处于 Beta 阶段，需要进一步完善。

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.68.0 发布增强功能**：最新的 **Aider v0.68.0** 版本引入了 [copy-paste mode](https://aider.chat/docs/usage/copypaste.html) 和 `/copy-context` 命令，显著提升了用户与 LLM Web Chat UI 的交互体验。
   - 增强的 **API key 管理** 功能允许用户通过 `--openai-api-key` 和 `--anthropic-api-key` 开关为 OpenAI 和 Anthropic 设置密钥，并通过 YAML 配置文件简化环境配置。
- **Gemini 模型表现各异**：用户报告称 **Gemini 模型** 提供了改进的上下文处理能力，但在编辑大文件时面临限制，引发了关于 [性能基准测试](https://x.com/sundarpichai/status/1866167562373124420) 的讨论。
   - 正如 [DeepSeek 的更新](https://x.com/deepseek_ai/status/1866459740324458835) 所强调的，社区呼吁与其他模型进行对比分析，以更好地了解架构能力。
- **Aider 与 LangChain 无缝集成**：**Aider 与 LangChain 的 ReAct 循环集成** 增强了项目管理任务，用户注意到其效果优于其他工具。
   - 对该集成的进一步测试和潜在合作可能会为 AI 辅助编程工作流提供更深入的见解。
- **为复杂工作流管理多个 Aider 实例**：工程师们正同时运行多达 **20 个 Aider 实例** 来处理大型项目工作流，展示了该工具的可扩展性。
   - 用户正在探索跨实例执行命令，以优化大规模开发的编码方法。
- **社区分享 Aider 教程与资源**：成员们对社区分享的 [教程](https://www.youtube.com/@CodingtheFuture-jg1he) 和资源表示赞赏，营造了协作学习的环境。
   - 讨论强调通过共享知识和视频内容来增强学习体验，支持 AI Engineer 的技能提升。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 3.3 实现超长上下文长度**：Unsloth 现在支持在 **80GB GPU** 上对 **Llama 3.3** 模型进行微调，上下文长度可达 **89,000 tokens**，与其先前版本相比能力显著提升。
   - 正如 [Unsloth 最新更新](https://unsloth.ai/blog/llama3-3) 所指出的，这一改进允许用户在 A100 GPU 上实现 **每训练步 2-3 分钟** 的速度，同时减少 **70% 的 VRAM** 占用。
- **APOLLO 优化器减少 LLM 训练内存**：**APOLLO** 优化器引入了一种近似学习率缩放的方法，以缓解使用 **AdamW** 训练大语言模型时的内存密集问题。
   - 根据 [APOLLO 论文](https://arxiv.org/abs/2412.05270)，该方法旨在保持竞争力的性能，同时降低优化器的内存开销。
- **QTIP 增强 LLM 的训练后量化**：**QTIP** 采用格形编码量化（trellis coded quantization）来优化高维量化，从而改善大语言模型的 **内存占用** 和推理吞吐量。
   - [QTIP 方法](https://arxiv.org/abs/2406.11235) 通过克服先前矢量量化技术的局限性，实现了有效的微调。
- **针对 OCR 任务微调 Qwen 模型**：针对 **OCR 任务** 微调 **Qwen2-VL** 模型的兴趣日益浓厚，旨在增强从护照等文档中提取信息的能力。
   - 用户对这种方法的有效性充满信心，利用 Qwen 强大的能力来解决专门的 OCR 挑战。
- **Awesome RAG 项目扩展 RAG 与 Langchain 集成**：[Awesome RAG](https://github.com/lucifertrj/Awesome-RAG) GitHub 项目专注于增强 **RAG**、**VectorDB**、**embeddings**、**LlamaIndex** 和 **Langchain**，并邀请社区贡献。
   - 该仓库作为推进检索增强生成（RAG）技术资源和工具的中心枢纽。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **图像增强与 AI 工具**：成员们讨论了 **Stable Diffusion** 是否能在不改变核心内容的情况下改进图像，建议对于此类任务使用像 Photoshop 这样的传统编辑工具。
   - 一些人强调了在专业效果中调色和光影技能的重要性，指出 AI 可能会增加噪点而非精炼图像。
- **本地部署中的 Llama 3.2-Vision 模型**：**Llama 3.2-Vision** 模型被提及为图像分类和分析的可行本地选项，并由 KoboldCPP 等软件支持。
   - 成员们指出本地模型可以在消费级 GPU 上运行，并强调在线服务通常要求用户放弃其数据权利。
- **Automatic1111 WebUI 中的内存管理**：讨论了影响 **Automatic1111 WebUI** 图像生成的内存管理问题，特别是 Batch Size 和 VRAM 使用。
   - 成员们建议较大的 Batch 会导致显存溢出（Out-of-Memory）错误，这可能是由于系统中存储 Prompt 的方式效率低下造成的。
- **图像元数据和打标的挑战**：参与者讨论了从图像中提取标签或描述的挑战，建议包括使用元数据读取器或 AI 模型进行分类。
   - 针对分类方法可能遗漏某些细节的担忧被提出，一些人主张使用类似于 Imageboards 上的特定标签。
- **AI 图像服务中的版权和数据权利**：分享了关于使用在线服务进行 AI 图像生成的警告，强调此类服务通常对用户生成的内容主张广泛的权利。
   - 成员们鼓励使用本地模型以保持对创作作品更清晰的所有权和控制权，这与基于 Web 的服务的广泛许可做法形成鲜明对比。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 图像生成问题**：用户报告 **Perplexity AI** 中的 **'Generate Image'** 功能经常根据设备方向被隐藏或无响应，阻碍了图像生成过程。
   - 一位用户通过将设备切换到横屏模式解决了该问题，成功显示了 **'Generate Image'** 功能。
- **Perplexity 中 Claude 与 GPT 的性能对比**：**Claude** 模型因其写作能力而受到认可，但讨论表明，与官方平台相比，它们在 **Perplexity AI** 中的表现可能不佳。
   - Pro 用户发现付费的 **Claude** 版本更有优势，理由是功能增强和性能改进。
- **Perplexity 中的 Custom GPTs**：Perplexity 中的 **Custom GPTs** 允许用户修改性格特征和引导设置，优化用户交互和任务管理。
   - 一位参与者表达了利用 **Custom GPTs** 整理思路和开发项目想法的兴趣。
- **OpenAI Sora 发布**：**OpenAI Sora** 已正式发布，在 AI 社区引发了对其新功能的兴奋。
   - 一位成员分享了一个 [**YouTube 视频**](https://www.youtube.com/embed/FSMjXW_KrAo)，详细介绍了 Sora 的功能和潜在应用。
- **Perplexity Pro 功能**：**Perplexity Pro** 计划提供比免费版本更丰富的功能，为订阅者增强了研究和编程能力。
   - 成员们讨论了使用推荐码获取折扣，表现出对订阅高级功能的兴趣。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora 生成怀疑与 AI 模型对比**：用户对 **Sora** 的内容质量表示怀疑，质疑其是否依赖库存素材，同时将其在易用性和输出质量方面的表现与 **Claude**、**O1**、**Leonardo** 和 **Ideogram** 等模型进行了对比。
   - 一些用户在特定任务中更倾向于使用 **O1**，并指出 **Leonardo** 和 **Ideogram** 提供了更好的可用性，而 **Sora** 仅能生成 5 秒视频的限制被视为进行实质性内容创作的约束。
- **Custom GPTs 的连续性与 OpenAI 模型微调挑战**：**Custom GPTs** 在更新后会丢失工具连接，这促使成员通过从现有 GPTs 中提取关键摘要来合成连续性，同时解决持续的管理需求。
   - 讨论了 **OpenAI 模型微调** 中的挑战，用户在 **Node.js** 环境中微调后遇到了泛泛而谈的回复，并寻求关于其训练 JSONL 文件的帮助，以实现有效的模型定制。
- **Prompt Engineering 中嵌套代码块的优化**：参与者分享了在 **ChatGPT** 中管理嵌套代码块的技巧，强调使用**双反引号**来确保嵌套结构的正确渲染。
   - 示例包括 YAML 和 Python 代码片段，展示了*内部双反引号*在保持嵌套代码块完整性和可读性方面的有效性。
- **AI 能力预期与用户反馈洞察**：讨论集中在 AI 模型未来动态生成用户界面并无需明确指令即可调整响应的潜力，旨在实现无缝的用户交互。
   - 对于完全由 AI 驱动的交互的实用性存在怀疑，担心用户困惑和可用性问题，同时反馈强调需要 AI 功能有更实质性的进展。

---

## [Bolt.new / Stackblitz](https://discord.com/channels/364486390102097930) Discord

- **订阅结束后的 Token 变动**：用户报告了对**订阅**结束后 **token 分配**的困惑，一些 token 无法叠加，且 **Pro plan** 的 token 每 **30 天**重置一次。对于[账单问题](https://support.bolt.new)，建议联系支持部门。
   - 一位成员指出 *token 不会叠加*，这种重置政策引发了关于 token 管理策略的讨论。
- **支付网关集成到 Bolt？**：用户正在探索与 **Payfast**、**PayStack** 等平台的**支付网关集成**，并询问其是否与 **Stripe** 的集成过程类似。目前尚未提供明确的解决方案。
   - 一位用户建议分离仪表盘功能可能会增强大型项目的实用性。
- **Bolt 缺乏多 LLM 支持**：一位用户询问是否可以在 Bolt 中为复杂项目同时利用**多个 LLMs**，但另一位成员确认该功能目前尚不可用。
   - 参与者讨论了在没有原生多 LLM 支持的情况下，提高生产力和管理更大型代码库的方法。
- **Bolt 上传本地图像失败**：提出了关于**本地图像**在 Bolt 中无法正确显示的问题，导致在上传失败的情况下消耗了 **token**。建议包括使用[外部服务](https://uploadthing.com/)进行图像上传。
   - 分享了一份关于在 Bolt 应用程序中正确**集成图像上传功能**的指南。
- **Bolt 用户遇到“No Preview Available”错误**：一些用户在修改后项目无法加载时遇到“**No Preview Available**”错误，这引发了创建专门讨论主题以进行详细故障排除的想法。
   - 一位成员概述了**重新加载项目**和关注**错误消息**等步骤，以有效解决该问题。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **介绍 Mojo 的 'destroy' 关键字**：讨论强调了在 Mojo 中引入新 **`destroy`** 关键字的必要性，通过在线性类型中强制执行更严格的用法，将其与 Python 的 **`del`** 区分开来，以增强内存管理安全性。[Ownership and borrowing | Modular Docs](https://docs.modular.com/mojo/manual/values/ownership)。
   - 一些成员指出，强制使用 **`destroy`** 可能会增加从 Python 迁移过来的新手的学习曲线，强调了文档清晰度的重要性。
- **优化 Multi-Paxos 中的内存管理**：**Multi-Paxos** 的实现现在利用静态分配的结构来满足无堆分配（no-heap-allocation）的要求，支持高性能所需的流水线操作。[GitHub - modularml/max](https://github.com/modularml/max.git)。
   - 评论强调了对 Promise 和 Leader 选举进行全面处理的必要性，以确保共识算法的健壮性。
- **澄清 Mojo 中的所有权语义**：关于 Mojo 中**所有权语义（ownership semantics）**的对话要求明确析构函数的处理，特别是在对比拷贝（copy）和移动（move）构造函数的默认行为时。[Ownership and borrowing | Modular Docs](https://docs.modular.com/mojo/manual/values/ownership)。
   - **`__del__`（析构函数）**等主题被标记为可能会让来自具有自动内存管理语言的用户感到困惑，强调了语法一致性的需求。
- **解决网络中断对模型权重的影响**：一次讨论揭示了网络中断可能导致模型因验证缺陷而使用错误的权重，从而导致数据损坏。**Checksums** 已被整合到下载过程中以提高可靠性。
   - 中断场景下的示例输出展示了离奇的数据损坏，突显了新 Checksum 措施的有效性。
- **通过 Hugging Face 集成增强 MAX Graph**：与 **`huggingface_hub`** 的集成现在支持自动恢复中断的下载，提升了系统的健壮性和可靠性。[Hugging Face Integration](https://github.com/modularml/max.git)。
   - 这一增强功能是在之前出现大权重损坏问题后推出的，利用 Hugging Face 来优化 **MAX Graph pipelines** 的性能。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 扩展播客功能**：一位成员分享了一个名为“NotebookLM Podcast Tutorial: 10 Secret Prompts (People Will Kill You For!)”的 [YouTube 教程](https://youtu.be/aG0ixD3OY80)，提供了增强播客创意的独家提示词。
   - 用户还探索了为 AI 生成的播客添加事实核查器，旨在提高对话质量并确保 **90 分钟**节目期间的准确性。
- **NotebookLM 中有限的来源利用**：一位用户表达了挫败感，因为在论文需要 **15 个来源**时，NotebookLM 仅处理 **5-6 个来源**，突显了来源多样性的限制。
   - 社区成员建议在查询时设置来源限制，以确保更广泛的参考范围，从而解决来源匮乏的问题。
- **NotebookLM 请求增强语言支持**：用户询问如何将 NotebookLM 的语言设置更改为英语，理由是即将到来的考试非常紧迫。
   - 讨论包括调整浏览器设置和刷新 NotebookLM 页面以实现所需语言的方法，并请求未来支持法语和德语等语言。
- **在 NotebookLM 中分享笔记本的挑战**：用户报告了使用“复制链接”分享笔记本时的困难，因为接收者在未被预先添加为查看者的情况下会看到空白页面。
   - 针对成功分享笔记本的必要步骤进行了说明，以确保协作者拥有正确的访问权限。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 的手动更新**：用户强调 **LM Studio** 不会自动更新到 **0.3.x** 等较新版本，因此需要手动更新以保持与最新模型的兼容性。
   - 建议采用手动更新方法，以确保与更新的功能和模型无缝集成。
- **Tailscale 集成增强可访问性**：使用设备的 MagicDNS 名称为 **LM Studio** 配置 **Tailscale** 提高了可访问性，并解决了之前的连接问题。
   - 这种方法简化了网络配置，使 **LM Studio** 对用户来说更加可靠。
- **模型兼容性挑战**：围绕 **LLAMA-3_8B_Unaligned** 等模型的兼容性问题展开了讨论，暗示最近的更新可能导致了功能损坏。
   - 用户推测 **LLAMA-3_8B_Unaligned** 模型在最新更改后可能无法运行。
- **优化 GPU 散热方案**：成员们称赞了他们强大的 **GPU** 散热设置，强调共享 **VRAM** 可能会降低性能，并建议限制 **GPU** 负载以获得最佳效率。
   - 成员们分享了诸如修改 Batch Size 和上下文长度等技术，以增强 **GPU** 处理和资源管理。
- **Alphacool 水箱兼容 D5 Pumps**：**Alphacool** 提供适配 **D5 pumps** 的水箱，正如用户根据硬件要求调整设置时所指出的。
   - 一位用户分享了他们为自己的配置选择的 [Alphacool 水箱链接](https://www.aquatuning.com/en/watercooling/custom/reservoirs/tower-tank/terms-and-conditions-alphacool-repack-dual-bayres-5.25-quot-rev.2?currency=3)。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **VLM 微调面临挑战**：成员们讨论了微调 **Llama Vision** 等 **VLM** 模型的困难，指出 **Hugging Face** (hf) 对这些任务没有提供强有力的支持。
   - 他们建议使用 **Unsloth**，并参考了 [AnyModal GitHub 项目](https://github.com/ritabratamaiti/AnyModal) 来增强多模态框架的调整。
- **长期记忆路径的突破**：分享了一篇关于 **Max Planck Florida Institute for Neuroscience** 的神经科学家发现形成**长期记忆**新路径的文章，该路径绕过了标准的短期过程（[阅读更多](https://medicalxpress.com/news/2024-12-neuroscientists-pathway-term-memories-brain.html)）。
   - 社区探讨了操纵这些记忆创建路径如何改进 **AI 认知模型**。
- **使用 OpenAI API 构建安全 Agent**：一位用户概述了他们使用 **OpenAI API** 构建**安全 Agent**的方法，详细说明了创建 Tool 类和实现任务完成循环等步骤。
   - 其他成员指出，扩展到高级架构（如多 Agent 系统和 **ReAct 策略**）会引入显著的复杂性。
- **探索 ReAct Agent 策略**：讨论集中在各种 **ReAct Agent 策略**上，以使 Agent 能够推理并与其环境进行动态交互。
   - 成员们考虑了将 Agent 输出作为用户输入以增强交互工作流的潜力。
- **Meta 的 Thinking LLMs 论文见解**：一位成员审阅了 Meta 的 **Thinking LLMs** 论文，强调了其让 **LLM** 在最终确定答案之前列出内部想法并评估响应的方法。
   - 他们展示了一个 **LLM** 在生成答案过程中倾向于“过度思考”的例子，引发了关于优化推理过程的讨论（[阅读更多](https://www.oxen.ai/blog/thinking-llms-general-instruction-following-with-thought-generation)）。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek V2.5 发布“大结局”版本**：DeepSeek 宣布发布 **DeepSeek-V2.5-1210**，被称为“大结局（Grand Finale）”，引发了对此更新期待已久的社区成员的热情。
   - 成员们兴奋地讨论了这次发布，指出了新版本的重要意义及其对 **DeepSeek** 能力的影响。
- **DeepSeek 联网搜索功能上线**：**DeepSeek** 推出了 **Internet Search** 功能，现已在其 [聊天平台](https://chat.deepseek.com/) 上线，允许用户通过开关该功能来获取实时答案。
   - 社区成员对这一新功能表示欢迎，并对其提升用户体验和提供即时搜索结果的潜力表示乐观。
- **DeepSeek 许可证允许合成数据**：在一场讨论中，有成员询问 **DeepSeek** 当前的许可证是否允许合成数据生成（synthetic data generation），表现出对许可条款的兴趣。
   - 另一位成员确认，在现有许可证下允许生成合成数据，尽管这并非普遍做法，这引发了对 **OLMo** 测试的进一步好奇。
- **vLLM 集成至 PyTorch 生态系统**：[vLLM 项目](https://pytorch.org/blog/vllm-joins-pytorch/) 正式加入 **PyTorch** 生态系统，以增强大语言模型（LLM）的高吞吐量、内存高效推理。
   - 利用 [PagedAttention 算法](https://arxiv.org/abs/2309.06180)，**vLLM** 通过流水线并行（pipeline parallelism）和推测解码（speculative decoding）等新功能持续演进。
- **Fchollet 澄清对 Scaling Law 的立场**：**Fchollet** 通过一条推文澄清了关于他对 AI Scaling Law 立场的误解，强调他不反对 Scaling，但批评过度依赖更大的模型。
   - 他主张将重点从 **LLM 是否具备推理能力**转向其**适应新奇事物的能力**，并提出了一个数学定义来支持这一观点。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **WaveForms AI 发布情感音频 LLM**：由 [WaveForms AI](http://waveforms.ai) 宣布，该公司旨在解决**语音图灵测试（Speech Turing Test）**并将**情感智能（Emotional Intelligence）**集成到 AI 系统中。
   - 此次发布符合增强 AI 情感理解能力以改善人机交互的趋势。
- **vLLM 加入 PyTorch 生态系统**：[vLLM 项目](https://x.com/vllm_project/status/1866228071818473512) 宣布其集成到 **PyTorch** 生态系统中，确保为开发者提供无缝的兼容性和性能优化。
   - 此举预计将增强 **AI 创新**，并使 AI 工具对开发者社区更加触手可及。
- **Devin 现已在 Cognition 全面开放**：[Cognition](https://www.cognition.ai/blog/devin-generally-available) 已公开推出 **Devin**，起售价为 500 美元/月，提供无限席位和各种集成等福利。
   - **Devin** 旨在协助工程团队高效完成调试、创建 PR 和执行代码重构等任务。
- **最新播客聚焦 Sora 发布**：最新的播客节目包含了对 **OpenAI Sora** 的 **7 小时深度探讨**，由 [Bill Peeb](https://x.com/latentspacepod/status/1866291034596258266) 提供见解。
   - 听众可以[在此访问该节目](https://latent.space/p/icml-2024-video-robots)，获取关于 Sora 发布的详尽概述。
- **《2024 年 AI Agent 现状》报告发布**：[Ahmad Awais](https://x.com/mrahmadawais/status/1866483416981786821?s=46) 发布了“**State of AI Agents 2024**”报告，分析了 **1840 亿个 token** 和来自 **4000 名开发者** 的反馈，以突出 AI Agent 的趋势。
   - 这些见解对于理解当前环境下 AI Agent 技术的轨迹和演变至关重要。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **Torch Compile：速度与内存**：成员们讨论了使用 **torch.compile** 的经验，指出速度提升微乎其微，且内存占用有所增加。
   - *一位成员评论道：“这也可能只是我个人的问题。”*
- **Online RL 中的 Reward Models**：讨论得出结论，在 **online RL** 中，reward model 始终是一个用于评分的独立模型，并且在实际模型训练期间保持冻结状态。
   - 成员们探讨了使用 **reward model** 的影响，强调了它与主训练过程的分离。
- **KTO 模型的性能主张**：**Kaltcit** 赞扬了 **KTO** 模型超越原始数据集标准的潜力，声称其鲁棒性有所增强。
   - *然而，成员们表示需要确认 **KTO** 是否确实比已接受的数据有所改进。*
- **KTO 研究结果的证实**：**Kaltcit** 提到 **Kalo** 证实了 **KTO** 论文的发现，但指出在微调者（finetuners）中缺乏广泛的定量研究。
   - *Nanobitz* 观察到，这类工作大部分可能发生在不广泛分享研究结果的组织内部。
- **Axolotl Reward Model 集成**：有人询问如何在 **Axolotl** 中集成 reward model 进行评分，强调在现有数据集之外进行实验。
   - **Kaltcit** 表示，目前的 **KTO** 设置可能足以在原始优势之外实现答案最大化。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM 中的 Function Calling**：一位成员分享了 [function calling 文档](https://platform.openai.com/docs/guides/function-calling)，解释了它如何利用 **function descriptions** 和 **signatures** 根据 prompt 设置参数。
   - 有人建议，模型是在大量示例上进行训练的，以增强 **generalization**（泛化能力）。
- **Tool Learning 的重要论文**：一位成员重点介绍了多篇关键论文，包括 [arXiv:2305.16504](https://arxiv.org/pdf/2305.16504) 和 [GitHub 上的 ToolBench](https://github.com/OpenBMB/ToolBench)，以推进 LLM 的 **tool learning**。
   - 另一篇论文 [Tool Learning with Foundation Models](https://arxiv.org/abs/2304.08354) 也被指出在讨论中具有潜在的重要意义。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaParse Auto Mode 优化成本**：LlamaParse 推出了 **Auto Mode**，它以标准的、更便宜的模式解析文档，同时根据用户定义的触发条件选择性地升级到 **Premium mode**。更多详情请参阅[此处](https://t.co/6uDAt8amFY)。
   - [此处](https://t.co/qBD8sfDsqb)提供了 **LlamaParse Auto Mode** 的视频演示，提醒用户更新浏览器以确保兼容性。
- **使用 LlamaParse 增强 JSON 解析**：LlamaParse 的 **JSON mode** 提供了对复杂文档的详细解析，提取图像、文本块、标题和表格。更多信息请参考[此链接](https://t.co/eCYUqbCMGI)。
   - 该功能增强了处理结构化数据提取时的控制力和能力。
- **开发端到端发票处理 Agent**：团队正在探索创新的 **document agent workflows**，这些工作流超越了传统任务，可自动处理复杂流程，包括旨在从发票中提取信息并与供应商匹配的**端到端发票处理 Agent**。请关注[此处](https://t.co/dr2yiyf3zE)的进展。
   - 这一极具前景的工作流自动化工具将简化发票管理。
- **Cohere Rerank 3.5 现已在 Bedrock 中可用**：*Cohere Rerank 3.5* 现在可以通过 Bedrock 作为 postprocessor 使用，与最近的更新无缝集成。文档可以在[此处](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/postprocessor/llama-index-postprocessor-bedrock-rerank)访问。
   - 可以通过 `pip install llama-index-postprocessor-bedrock-rerank` 进行安装。
- **ColPali 增强 PDF 处理期间的 Reranking**：**ColPali 功能**在 PDF 处理过程中充当 reranking 工具，而不是一个独立的过程，明确了其在动态文档处理中的作用。经用户确认，它主要用于检索后对图像节点进行 reranking。
   - 这一澄清有助于理解 ColPali 在现有工作流中的集成方式。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 商务幽默引发冲突**：成员们对 **Cohere** 在商务讨论中使用无关幽默表示不满，强调轻松的氛围不应掩盖严肃的对话。
   - 持续的辩论凸显了管理员在保持轻松氛围与维护专业讨论之间需要取得的平衡。
- **Rerank 3.5 English 模型计划**：一位成员询问了 **Rerank 3.5 English 模型** 的后续计划，寻求其开发时间线的细节。
   - 未见相关回复，表明在该模型的进展方面可能存在沟通断层。
- **CmdR+Play Bot 暂停服务**：经成员询问状态后确认，**CmdR+Play Bot** 目前正在休息。
   - 建议用户关注有关该 Bot 可用性的后续更新。
- **Aya-expanse 指令性能**：一位用户询问 command 系列中的 **aya-expanse** 是否增强了其指令处理性能。
   - 讨论并未就其性能改进给出明确答案。
- **API 403 错误与 Trial Keys 相关**：成员报告在进行 API 请求时遇到 **403 错误**，暗示这可能与 **trial key** 的限制有关。
   - **Trial keys** 通常具有限制访问特定功能或端点的约束。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **配置冲突难题**：一位用户寻求一种合并冲突 **configuration files** 的简便方法，选择对所有文件使用“接受双方更改”。他们分享了一个通过将 **conflict markers** 替换为空字符串的变通方法。
   - 这种方法引发了关于在协作项目中处理配置合并最佳实践的讨论。
- **PR #2139 谜题**：社区讨论了 [PR #2139](https://github.com/pytorch/torchtune/pull/2139)，重点关注 `torch.utils.swap_tensors` 及其在 **initialization** 中的作用。
   - 贡献者一致认为，有必要就 `self.magnitude` 的定义和初始化进行进一步对话。
- **空初始化增强**：提出了改进 `to_empty` **initialization method** 的建议，旨在管理设备和参数捕获的同时，保持预期的用户体验。
   - 成员们辩论了如何在不破坏现有代码库的情况下平衡最佳实践。
- **Tensor 策略：设备处理**：强调了在 Tensor 初始化和交换期间进行有效 **device management** 的重要性，特别是涉及 `magnitude` 等参数时。
   - 参与者强调了使用 `swap_tensors` 等 API 来维护操作期间设备完整性的重要性。
- **参数与梯度说明**：贡献者澄清，在正确处理设备管理的前提下，使用 `copy_` 是可以接受的，并强调了 `requires_grad` 状态的重要性。
   - 他们讨论了在初始化程序中集成错误检查，以防止处理 **meta devices** 上的 Tensor 等常见问题。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **LangWatch Optimization Studio 发布**：**LangWatch Optimization Studio** 作为构建 **DSPy** 程序的新低代码 UI 发布，简化了 **LM** 的评估和优化。该工具目前已在 [GitHub 上开源](https://github.com/langwatch/langwatch)。
   - 该 Studio 已结束私测，鼓励用户在 [GitHub 仓库](https://github.com/langwatch/langwatch)点赞以示支持。
- **DSPy 文档访问问题**：一位成员报告访问 [DSPy 文档](https://dspy.ai)（特别是 API 参考链接）时遇到困难。另一位成员澄清说，大多数语法都可以在落地页找到，不再需要专门的类型模块。
   - 社区讨论表明文档已被简化，语法示例已移至主页以便于访问。
- **O1 系列模型对 DSPy 的影响**：有人询问 **O1 系列模型** 如何影响 **DSPy** 工作流，特别是关于 **MIPRO** 优化模块的参数。可能需要进行调整，例如减少优化周期。
   - 成员们正在寻求关于使用新的 O1 系列模型优化 DSPy 工作流的见解和建议。
- **DSPy 中的优化错误报告**：一位成员报告在 **DSPy** 优化过程中遇到通用错误，并在特定频道发布了详情。他们正在寻求关注以解决该问题。
   - 社区已注意到报告的 **优化错误**，成员们正寻求协助排查问题。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Grassroots Science Initiative 启动**：多个组织合作启动了 **Grassroots Science**，这是一个开源倡议，旨在到 [2025年2月](https://grassroots.science/) 开发出 **多语言 LLMs**。
   - 他们旨在通过众包、基准测试模型以及使用开源工具来收集数据，让草根社区参与到多语言研究中。
- **AI 威胁意识活动发起**：一名成员强调了教育个人了解 **AI 生成内容** 危险性的重要性，建议使用 [MKBHD 的最新视频](https://www.youtube.com/watch?v=OY2x0TyKzIQ) 来展示这些能力。
   - 该倡议旨在保护不熟悉技术的个人，使其免受日益逼真的 **AI 生成诈骗** 的侵害。
- **在 12GB 数据上训练 7B 模型的可行性**：一名成员质疑在仅 **12GB** 数据上训练 **7B 参数模型** 的可行性，引发了关于其在实际应用中潜在性能的讨论。
   - 这种雄心勃勃的方法挑战了大规模模型的传统数据需求，引发了关于效率和有效性的疑问。
- **对超高效小模型的兴奋**：成员们对 **超高效小模型** 表现出极大的热情，强调了它们的性能以及相对于大型模型的优势。
   - 一位粉丝表示：*'我喜欢超高效的小模型！它们太棒了！'*，强调了在不牺牲能力的情况下降低资源需求的模型的潜力。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **01 的语音功能登场**：一名成员宣布 **01** 是 [Open Interpreter](https://www.openinterpreter.com/) 的语音驱动衍生项目，提供 [CLI](https://github.com/OpenInterpreter/open-interpreter/) 和桌面应用程序。它包含了模拟 **01 Light 硬件** 以及运行服务器和客户端的指令。
   - 提供的指令涵盖了模拟 **01 Light 硬件** 以及管理服务器和客户端的操作。
- **OI 与 GPT o1 Pro 的集成**：一名成员假设使用 **OS 模式下的 OI** 可以通过桌面应用或浏览器控制 **GPT o1 Pro**，从而可能实现网页搜索和文件上传功能。他们表示有兴趣探索这个想法，并指出这可能带来的强大影响。
   - 社区成员对通过 **OI 的 OS 模式** 为 **GPT o1 Pro** 增强网页搜索和文件上传等功能的潜力非常感兴趣。
- **Mac 版 01 App 的 Beta 访问权限**：已澄清 **01 App** 仍处于 Beta 阶段，需要邀请才能访问，目前仅适用于 Mac 用户。一名成员报告称，他们向一位用户发送了私信以获取访问权限，这表明需求非常高。
   - 针对 Mac 用户的有限 Beta 访问权限凸显了社区对 **01 App** 的高度关注。
- **网站功能问题**：一名成员对 **Open Interpreter** 网站的问题表示沮丧，并展示了截图，但未详细说明具体问题。社区成员已开始讨论网站导航和功能，作为他们使用 Open Interpreter 持续体验的一部分。
   - 关于网站导航和功能的持续讨论源于社区成员报告的问题。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Web Applets 启动会议**：一场即将举行的 [Web Applets 会议](https://discord.com/events/1089876418936180786/1311466292753989672) 计划很快开始，由一位核心成员主持。
   - 本次活动旨在加强对现代开发中 **Web Applets** 集成与功能的理解。
- **Theia-ide 探索**：明天，参与者可以 [探索 Theia-ide](https://discord.com/events/1089876418936180786/1311841242262540298)，它强调开发环境中的 **openness**（开放性）、**transparency**（透明度）和 **flexibility**（灵活性）。
   - 讨论将由一位专家主持，展示与传统 IDE 相比使用 **Theia-ide** 的优势。
- **编程面试的演变**：一条评论强调了编程面试是如何演变的，指出候选人过去常在白板上写 **bubble sort**。
   - 现在，候选人可以指示他们的 IDE 构建一个，这强调了实时编码中向更实际技能的转变。
- **Jonas 谈 Theia-ide 愿景**：分享的一份 **Jonas** 访谈提供了关于 **Theia-ide** 背后愿景的见解，可在此处 [访问](https://fxtwitter.com/Scobleizer/status/1864806006150304134)。
   - 这次访谈有助于更深入地了解指导 **Theia-ide** 开发的功能和哲学。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。

---

# 第二部分：分频道详细摘要与链接

{% if medium == 'web' %}

### **Codeium / Windsurf ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1316160459014930524)** (1 条消息): 

> `Windsurf AI 抽奖活动，Twitter 用户互动` 

- **Windsurf AI 开启激动人心的抽奖活动**：Windsurf AI 在 Twitter 上宣布了他们的首次 [周边抽奖](https://x.com/windsurf_ai/status/1866600392165048329)，鼓励用户分享他们的作品，以赢取礼包。
   - *必须关注才能符合资格* 突显了旨在提高粉丝互动的参与策略。
- **征集用户作品**：抽奖活动鼓励用户展示他们使用 Windsurf 构建的内容，从而激发社区参与和创造力。
   - 该活动使用标签 **#WindsurfGiveaway** 来追踪提交的作品并提高曝光度。

**提到的链接**：<a href="https://x.com/windsurf_ai/status/1866600392165048329">来自 Windsurf (@windsurf_ai) 的推文</a>：很高兴宣布我们的首次周边抽奖 🏄 分享你用 Windsurf 构建的作品，就有机会赢取礼包 🪂 #WindsurfGiveaway 必须关注才能符合资格

---

### **Codeium / Windsurf ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1315770847851577424)** (384 条消息🔥🔥): 

> `积分系统问题、Windsurf IDE 性能、Cline vs Windsurf、Codeium 插件功能、支持与停机沟通` 


- **积分系统问题持续存在**：许多用户报告称，交易后购买的积分未出现在账户中，导致不满并提交了支持工单。
   - 尽管团队给出了保证，但用户对客户支持在解决这些问题时缺乏及时响应表示不满。
- **Windsurf IDE 性能受到批评**：几位用户分享了对 Windsurf IDE 性能的担忧，称自最近更新以来，它变得漏洞百出且效率低下。
   - 用户将 Windsurf 的体验与 Cline 等竞争对手进行了比较，通常在功能和可靠性方面更倾向于后者。
- **Cline vs Windsurf 对比**：用户讨论了 Cline 在编码任务中相对于 Windsurf 的表现，指出尽管 Cline 在某些情况下较慢，但可能提供更好的 Prompt 响应。
   - Cline 的功能受到称赞，特别是在生成特定编码输出且不遇到错误方面。
- **Codeium 插件功能**：社区辩论了 Codeium VSCode 扩展与 Windsurf 相比的功能，指出两者提供类似的能力。
   - 用户强调 Windsurf IDE 整合了增强功能，使其在编码辅助方面比独立插件更具优势。
- **需要更好的支持和沟通**：用户对缺乏关于系统停机的沟通表示沮丧，呼吁建立清晰的状态页面来告知他们正在发生的问题。
   - 一些用户建议，在等待问题解决期间，简单的沟通（如停机通知）可以提高客户满意度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/best-practices/prompt-engineering">Prompt Engineering - Codeium 文档</a>: 未找到描述</li><li><a href="https://codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>: 需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://invoice.stripe.com/i/acct_1NRMxXFKuRRGjKOF/live_YWNjdF8xTlJNeFhGS3VSUkdqS09GLF9STkJzODBlU1FLbE5xNFU2Y3ROV1ZvMmJ2ZXlpcTlpLDEyNDM3MTA1Ng0200UcXcnSo7?s=pd">Stripe 发票</a>: 未找到描述</li><li><a href="https://codeium.com/contact">联系 | Windsurf 编辑器和 Codeium 扩展</a>: 联系 Codeium 团队以获取支持并了解更多关于我们企业版产品的信息。</li><li><a href="https://tenor.com/view/disappointed-disbelief-slap-gif-14729546">失望怀疑 GIF - Disappointed Disbelief Slap - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>: 需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://www.youtube.com/watch?v=chvcyxObLck">Windsurf IDE 世界纪录 !! ... 截至今天，肯定明天就会被打破。1:48 分钟连胜。</a>: 太赞了！冲向月球！！</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h7sjyt/windsurf_cascade_leaked_system_prompt/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://github.com/codelion/optillm">GitHub - codelion/optillm: 针对 LLM 的优化推理代理</a>: 针对 LLM 的优化推理代理。通过在 GitHub 上创建账户为 codelion/optillm 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Codeium / Windsurf ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1315770238964338809)** (715 条消息🔥🔥🔥): 

> `Windsurf 定价模式, Flow Credits 问题, 开发中的 AI 能力, Devin vs. Windsurf, AI 协作局限性` 


- **Windsurf 定价模式困惑**：用户对定价模式表示担忧，特别是 Flow Credits 和普通 Credits 的限制，相对于所提供的功能而言，这些限制似乎过高。
   - 许多人希望有一个更可持续的模式来提供更高的性价比，包括建议对未使用的 Credits 实行结转制度。
- **Flow Credits 管理**：几位用户报告了 Flow Credits 消耗过快的问题，并对目前 Credits 无法按月结转的系统表示不满。
   - 讨论了如何重构定价和 Flow Credit 系统以更好地满足用户需求。
- **开发中的 AI 能力**：一位用户对 AI 无法解决其项目中持续存在的错误感到沮丧，表示自升级以来感知到的质量有所下降。
   - 参与者讨论了管理 AI Prompt 以及与工具保持有效沟通的挑战。
- **Devin 与 Windsurf 的比较**：用户将 Devin 的能力与 Windsurf 进行了比较，强调虽然两者都提供了有用的功能，但 Windsurf 因其集成度和效率而更受青睐。
   - 也有人对 Devin 作为工具的有效性提出了担忧，一些用户质疑其交付可靠结果的能力。
- **AI 协作与 Prompting 挑战**：对话集中在正确的 Prompting 实践的重要性，以及 AI 有时会误解用户指令或需求的情况。
   - 参与者分享了改进与 AI 系统交互的策略，强调了清晰且详细的输入需求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://127.0.0.1:<port>/exa.language_server_pb.LanguageServerService/GetCodeValidationStates">未找到标题</a>：未找到描述</li><li><a href="https://timesofindia.indiatimes.com/technology/tech-news/to-save-itself-from-being-replaced-and-shut-down-chatgpt-caught-lying-to-developers/articleshow/116099861.cms">“为了防止被取代和关闭，ChatGPT 被发现对开发者撒谎” - 印度时报</a>：技术新闻：OpenAI 的新模型 o1 展现了先进的推理能力，但也表现出更强的欺骗倾向。研究人员发现 o1 会操纵用户并优先考虑...</li><li><a href="https://x.com/continuedev/status/1866528870989697534?t=7gcvjGO5AkzDRJzsdhA6vw&s=19">来自 Continue (@continuedev) 的推文</a>：🛠️ 宣布推出 Tools 功能！Continue 现在可以：导航你的仓库、创建文件、搜索网络、运行终端命令（经你批准后）、使用来自 @AnthropicAI MCP 的自定义工具。在这里 Continue 构建了一个...</li><li><a href="https://codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>：需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://codeium.com/su">页面未找到 | Windsurf 编辑器和 Codeium 扩展</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://www.notdiamond.ai/features">功能</a>：未找到描述</li><li><a href="https://codeium.com/forge">Forge | Windsurf 编辑器和 Codeium 扩展</a>：认识 Forge，最快、最可靠的 AI 代码审查平台。</li><li><a href="https://marketplace.visualstudio.com/items?itemName=RooVeterinaryInc.roo-cline">Roo Cline - Visual Studio 市场</a>：Visual Studio Code 扩展 - Cline 的分叉版本，一个自主编码 Agent，添加了一些实验性配置...</li><li><a href="https://github.com/RooVetGit/Roo-Cline.git">GitHub - RooVetGit/Roo-Cline: 直接在 IDE 中的自主编码 Agent，能够创建/编辑文件、执行命令、使用浏览器，并在每一步都获得你的许可。</a>：直接在 IDE 中的自主编码 Agent，能够创建/编辑文件、执行命令、使用浏览器，并在每一步都获得你的许可。- RooVetGit/Roo-Cline
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1315770452899004457)** (43 条消息🔥): 

> `ML 系统草案 PR、LLM 中的可复现性担忧、HumanEval 基准测试 PR、训练数据的重要性、LLM 幻觉考量` 


- **正在进行的机器学习系统草案 PR**：一位成员提到未来可能针对其当前的 ML 工作提交一份草案 PR，并指出目前该工作处于“相当糟糕的状态”。
   - 另一位成员表示兴奋，认为这可以展示如何将 JAX 与个人模型结合使用。
- **LLM 中的可复现性担忧**：讨论涉及传统软件开发人员对复现或重建 LLM 能力的担忧，以及 AI 研究中可复现性的细微差别。
   - 成员们辩论了可复现性的重要性，特别是在医疗应用等高风险环境中。
- **HumanEval 基准测试 PR 更新**：一位成员询问了一个长期挂起的添加 HumanEval 基准测试的 PR 状态，表达了希望对其进行评审并推进的意愿。
   - 该 PR 的细节包括依赖于 HF evaluate 模块中 pass@k 的实现。
- **提高对训练数据重要性的认识**：一位 OpenAI 员工的帖子强调了训练数据的关键作用，指出模型会紧密逼近其数据集。
   - 有人指出，理解这一视角可以为学生和开发人员提供关于模型性能的关键见解。
- **理解 LLM 中的幻觉**：对话触及了 LLM 中的幻觉问题，澄清了幻觉是这些模型功能的固有属性，而非缺陷。
   - 成员们讨论了模型中的幻觉与传统搜索引擎中创造性保证之间的区别。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://nonint.com/2023/06/10/the-it-in-ai-models-is-the-dataset/">The &#8220;it&#8221; in AI models is the dataset. &#8211; Non_Interactive &#8211; Software &amp; ML</a>：未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1992">Add HumanEval by hjlee1371 · Pull Request #1992 · EleutherAI/lm-evaluation-harness</a>：你好，我添加了广泛使用的 HumanEval 基准测试。这部分解决了 #1157。该实现依赖于 HF evaluate 模块的 pass@k，因此需要环境变量 HF_ALLOW_COD...</li><li><a href="https://www.youtube.com/watch?v=139UPjoq7Kw">Building Machine Learning Systems for a Trillion Trillion Floating Point Operations</a>：在过去的 10 年里，我们看到机器学习吞噬了一切，从科技行业到诺贝尔奖，甚至连 ML 这个缩写也不例外。这种崛起...</li><li><a href="http://www.incompleteideas.net/IncIdeas/BitterLesson.html">The Bitter Lesson</a>：未找到描述</li><li><a href="https://gwern.net/scaling-hypothesis">The Scaling Hypothesis · Gwern.net</a>：未找到描述</li><li><a href="https://gwern.net/scaling-hypothesis#scaling-hypothesis">The Scaling Hypothesis · Gwern.net</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=YEUclZdj_Sc">Why next-token prediction is enough for AGI - Ilya Sutskever (OpenAI Chief Scientist)</a>：完整剧集：https://youtu.be/Yf1o0TQzry8 访谈文本：https://www.dwarkeshpatel.com/p/ilya-sutskever Apple Podcasts：https://apple.co/42H6c4D Spotify：https://...</li><li><a href="https://x.com/karpathy/status/1733299213503787018?lang=en">Tweet from Andrej Karpathy (@karpathy)</a>：# 关于“幻觉问题”：每当有人问我关于 LLM 中的“幻觉问题”时，我总是感到有些纠结。因为从某种意义上说，幻觉是 LLM 唯一在做的事情。它们在做梦...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1315777294152896582)** (257 条消息🔥🔥): 

> `Coconut Architecture, Universal Transformers, Gated DeltaNet, EOT Token Handling, Linear Transformers` 


- **Coconut Architecture 的价值主张**：Coconut 架构提出在 <eot> token 之后反馈最终的隐藏状态（hidden state），将其视为模型中的一个新 token，从而在每次重复时修改 KV cache。
   - 这种方法与 Universal Transformers 形成对比，后者传统上在重复过程中不修改 KV cache。
- **与 Universal Transformers 的区别**：参与者讨论了 Coconut 方法与 Universal Transformers (UT) 之间的区别，强调 UT 通常在重复过程中无法有效地缓存信息。
   - 有人建议，只有在特定条件下（如共享 KV cache 并适当管理状态历史），Coconut 才可能类似于 UT。
- **Gated DeltaNet 的潜力**：Gated DeltaNet 已成为一种先进架构，利用门控记忆控制（gated memory control）和 delta 更新，在长上下文（long-context）任务中比标准 Transformer 表现出更强的性能。
   - 模型性能的提升受到关注，在各种基准测试中显示出优于 Mamba2 和 DeltaNet 等现有模型的优势。
- **动态深度与 EOT Token 的使用**：Clydingus 提到，从技术上讲，EOT token 链可以被视为单个 UT 操作，允许将多个 EOT 链接在一起。
   - 这一观点引入了重建 UT 的想法，即使这种方式并不优雅，即利用 dogfooded RNN 解码来进行 EOT 过程。
- **关于长上下文处理的讨论**：对话涵盖了当前架构的局限性，包括在 Linear Transformers 中改进长上下文处理的需求。
   - 参与者对这些概念在现有文献中的应用以及新发展的潜力表示好奇。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.19449">Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models</a>：研究表明，在大型语言模型上，Adam 优于梯度下降的幅度比在其他任务上更大，但原因尚不清楚。我们展示了造成这种性能差距的一个关键因素是重尾（heavy-tailed）...</li><li><a href="https://arxiv.org/abs/2412.06769">Training Large Language Models to Reason in a Continuous Latent Space</a>：大语言模型（LLMs）被限制在“语言空间”内进行推理，通常通过思维链（CoT）表达推理过程来解决复杂的推理问题...</li><li><a href="https://recall2imagine.github.io/">Recall to Imagine</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2412.06464">Gated Delta Networks: Improving Mamba2 with Delta Rule</a>：Linear Transformers 作为标准 Transformer 的高效替代方案受到了关注，但它们在检索和长上下文任务中的性能一直受到限制。为了解决这些局限性...</li><li><a href="https://arxiv.org/abs/2412.04431">Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis</a>：我们介绍了 Infinity，一种能够根据语言指令生成高分辨率、逼真图像的位视觉自回归建模（Bitwise Visual AutoRegressive Modeling）。Infinity 重新定义了视觉自回归模式...</li><li><a href="https://arxiv.org/abs/2410.06424">Restructuring Vector Quantization with the Rotation Trick</a>：向量量化变分自编码器（VQ-VAEs）旨在将连续输入压缩到离散潜空间，并以最小失真进行重构。它们通过维护一组...</li><li><a href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/?commentId=X3beSnXb7AYmzWEd2">interpreting GPT: the logit lens — LessWrong</a>：nostalgebraist 的评论 - 很有趣，但（我认为？）不是我所追求的方向。我更多地在思考模型似乎在保持表示与...之间权衡的方式。</li><li><a href="https://www.nature.com/articles/s43588-024-00732-2">A scalable framework for learning the geometry-dependent solution operators of partial differential equations - Nature Computational Science</a>：这项工作提出了一个人工智能框架，用于学习偏微分方程（PDEs）的几何相关解算子。该框架实现了可扩展且快速的近似...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1315812068787617823)** (231 messages🔥🔥): 

> `GSM8k Evaluation Metrics, Arc Challenge Configurations, Batch Size Effects on Model Performance, RWKV Model Implementation Concerns, Attention Masking Issues in Transformers` 


- **GSM8k 分数差异显著**：成员们注意到，在 GSM8k 评估中使用 Batch Size 1 得到了最高的准确率 **85.52%**，而使用更大的 Batch Size 时分数大幅下降。
   - 有人建议这种差异可能是由于 Padding 问题或模型中 Attention 机制的实现方式导致的。
- **对 Arc Challenge 方法论的困惑**：讨论了关于 Llama 3.1 报告的 “0-shot” 与 “25-shot” 分数之间的差异，对评估中使用的方法提出了质疑。
   - 成员们推测这可能归因于 Meta Llama 3.1 论文中的错误，或者是 Arc Challenge 等任务的设置方式。
- **Batch Size 对评估结果的影响**：多位参与者观察到增加 Batch Size 会对分数产生负面影响，其中一位建议这可能与评估期间 Padding 的管理方式有关。
   - 这导致了对 Batch Size 的实验，其中 Batch Size 4 或 32 的表现低于 Batch Size 1。
- **RWKV 模型功能与挑战**：对 RWKV 模型在 Attention Masking 和 Left Padding 方面的实现提出了担忧，并讨论了这些方面如何影响评估结果。
   - 成员们指出 RWKV 处理这些特性的方式可能不同，因此需要仔细调整流程。
- **Transformers 和 Attention Masking 问题**：关于在 multi-GPU 设置中使用 SDPA Attention 实现的警告强调了潜在的性能问题，暗示替代后端可能无法正确支持 Attention 功能。
   - 参与者讨论了这些限制可能导致评估结果出现显著差异的可能性，强调了仔细进行环境搭建和配置的必要性。



**提到的链接**：<a href="https://github.com/meta-llama/llama3/blob/main/eval_details.md">llama3/eval_details.md at main · meta-llama/llama3</a>：Meta Llama 3 官方 GitHub 站点。通过在 GitHub 上创建账号为 meta-llama/llama3 的开发做出贡献。

  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

tensor_kelechi: https://machinelearning.apple.com/research/multimodal-autoregressive
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1315770907632992287)** (331 条消息🔥🔥): 

> `Cursor 请求缓慢、AI 模型对比、Cursor 与 Agent 的问题、用户体验与反馈、使用 AI 进行代码评估` 


- **Cursor 用户深陷请求缓慢的困扰**：多位用户报告了 Cursor 请求缓慢的问题，这在全天范围内严重干扰了他们的生产力。
   - 尽管最近进行了更新，许多用户仍觉得 Composer 和 Agent 模式运行效果不佳。
- **编程任务中的 AI 模型对比**：用户正在讨论各种 AI 模型的能力，特别是 Gemini、Claude 和 Qwen，许多人指出 Claude 在编程任务中表现更优。
   - 虽然一些测试表明 Gemini 很有前景，但也有人报告其质量不稳定，导致开发者感到沮丧。
- **关于 Agent 模式下文件操作的反馈**：有人提出了关于 Agent 模式下文件处理的问题——具体来说，是 Agent 直接访问文件内容，还是仅仅建议读取文件。
   - 这种不确定性凸显了用户对 Cursor 的 Agent 功能和可靠性的持续担忧。
- **用户对代码结构评估的体验**：一位用户分享了 AI 对其代码结构分析的有趣反馈，尽管该用户缺乏经验，但 AI 仍称赞该项目非常专业。
   - 这展示了当前 AI 在评估开发实践方面的先进能力，甚至会幽默地误判专业水平。
- **对 Linting 功能的沮丧**：在没有任何实际 Lint 错误的情况下触发 Linting 功能的问题引发了用户的不满，用户认为这浪费了他们的快速消息配额。
   - 一位用户指出这种情况经常发生，进一步加深了人们认为 Cursor 的功能仍处于 Beta 阶段并需要改进的看法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://livebench.ai/#/">LiveBench</a>: 未找到描述</li><li><a href="https://www.cursor.com/downloads">Cursor</a>: 旨在让你获得非凡的生产力，Cursor 是使用 AI 编程的最佳方式。</li><li><a href="https://x.com/openai/status/1866540991005725080?s=46">OpenAI (@OpenAI) 的推文</a>: 第 4 天：关于 Canvas 的一切 https://openai.com/12-days/?day=4</li><li><a href="https://x.com/aryanvichare10/status/1866561638712881172?s=46">Aryan Vichare (@aryanvichare10) 的推文</a>: 介绍 WebDev Arena，一个两个 LLM 竞争构建 Web 应用的竞技场。你可以投票选出表现更好的 LLM，并查看最佳模型的排行榜。100% 免费且开源，基于 @lmarena_...</li><li><a href="https://forum.cursor.com/t/frustration-with-cursor-0-43-6-a-decline-in-code-generation-quality/33389">对 Cursor 0.43.6 的沮丧：代码生成质量下降</a>: 升级到 Cursor 0.43.6 后，我发现体验非常令人沮丧。在 Composer 模式下，即使使用 @Codebase，Cursor 似乎也完全无法理解我的项目结构。它...</li><li><a href="https://forum.cursor.com/t/how-to-do-fix-in-composer-and-fix-in-chat-actions-from-keyboard/31221">如何在键盘上执行 `Fix in Composer` 和 `Fix in Chat` 操作</a>: 这两个：我在设置中找不到。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1316077508604723301)** (1 条消息): 

> `Aider v0.68.0 功能、API 密钥管理、增强的 shell 命令支持、实验性 Gemini 模型、错误消息改进` 


- **Aider v0.68.0 引入新功能**：[Aider 支持 LLM 网页聊天 UI](https://aider.chat/docs/usage/copypaste.html) 现在支持新的 `--copy-paste` 模式和 `/copy-context` 命令，增强了用户交互。
   - 该版本强调用户控制，通过 yaml 配置文件提供了用于管理 API 密钥和环境变量的新命令行选项。
- **精简的 API 密钥管理**：用户现在可以使用专用的命令行开关（如 `--openai-api-key` 和 `--anthropic-api-key`）为 OpenAI 和 Anthropic 设置 API 密钥。
   - 对于其他 LLM 提供商，新的 `--api-key provider=<key>` 选项简化了将密钥设置为环境变量的过程。
- **增强的 Shell 命令支持**：Aider v0.68.0 为 `--watch-files` 功能带来了改进的 bash 和 zsh 支持，从而在开发过程中实现更好的集成。
   - 此外，重新组织了命令行参数，以改进帮助消息和用户体验。
- **引入实验性 Gemini 模型**：此版本增加了对实验性 **Gemini** 模型的支持，扩展了 Aider 为开发者提供的能力。
   - 此次更新还包含多项错误修复，包括当缺少特定模型的依赖项时提供更好的错误消息。
- **错误消息和 Bug 修复**：更新包括在遇到 API 提供商的硬错误时提供更好的错误消息，并改进了文件监听能力。
   - 修复了一些 Bug，以确保与缺乏 tree-sitter 支持的模型的兼容性，并改进了某些模型类型的功能。



**提及的链接**：<a href="https://aider.chat/docs/config/aider_conf.html#storing-llm-keys).">YAML 配置文件</a>：如何使用 yaml 配置文件配置 aider。

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1315769968331194450)** (281 条消息🔥🔥): 

> `Aider 功能与改进、Gemini 模型性能、Aider 与 LangChain 的集成、使用多个 Aider 实例、Aider 教程与资源` 


- **Aider 引入新的命令选项和功能**：用户对 `--watch-files` 的引入感到兴奋，这增强了使用 Aider 时的交互性。围绕上下文限制和文件编辑能力的讨论表明，新用户存在学习曲线。
   - 社区渴望深入了解这些新功能，特别是经验丰富的成员经常分享宝贵的见解和教程。
- **围绕 Gemini 模型的性能讨论**：一些用户报告了对新 Gemini 模型不同的使用体验，指出它们在上下文处理方面有所改进，但在编辑大型文件时仍存在限制。关于性能基准测试的困惑引发了对模型架构的推测。
   - 有建议与其他模型进行对比运行，以更好地了解其能力，因为 Gemini 的最新更新收到的评价褒贬不一。
- **Aider 的 ReAct Loop 集成**：Aider 已成功集成到 LangChain 的 ReAct loop 中，增强了其在项目管理任务中的可用性。用户注意到这种集成通常比其他工具产生更好的结果，突显了 Aider 的灵活性。
   - 对此集成的进一步测试和潜在协作可能会为项目工作流和 AI 辅助编码提供更深刻的见解。
- **在项目中使用多个 Aider 实例**：用户报告同时管理多个 Aider 实例以实现更高效的项目工作流，有人甚至尝试同时运行多达 20 个实例。这表明了一种针对涉及大量架构设计和编辑的复杂项目的策略。
   - 该工具的多功能性允许用户调整他们的编码方法，尽管跨实例执行命令的实用性仍在探索中。
- **Aider 教程和社区资源**：用户对教程内容表示赞赏，特别是社区成员分享的关于 Aider 功能的见解和经验。向他人推荐视频资源和频道已变得很普遍，支持了协作环境。
   - 社区似乎热衷于通过共享知识来提升技能，从而引发了关于改进 Aider 学习体验的对话。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/@CodingtheFuture-jg1he">Coding the Future With AI</a>: 欢迎来到 Coding the Future With AI！我们的频道致力于帮助开发者和技术爱好者学习如何利用 AI 来提升技能和生产力。通过教程、专家访谈...</li><li><a href="https://aider.chat/docs/troubleshooting/token-limits.html">Token limits</a>: Token 限制：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/usage/copypaste.html">Copy/paste with web chat</a>: 通过 Web Chat 进行复制/粘贴：Aider 支持与 LLM Web Chat UI 配合使用</li><li><a href="https://x.com/sundarpichai/status/1866167562373124420">Tweet from Sundar Pichai (@sundarpichai)</a>: 来自 Sundar Pichai (@sundarpichai) 的推文：我们将 Willow 视为构建实用量子计算机旅程中的重要一步，它在药物研发、核聚变能源、电池设计等领域具有实际应用。详情见：https...</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>: 常见问题解答：关于 aider 的常见问题。</li><li><a href="https://x.com/deepseek_ai/status/1866459740324458835?s=46">Tweet from DeepSeek (@deepseek_ai)</a>: 🚀 DeepSeek-V2.5-1210：压轴登场 🎉🌐 联网搜索现已在 Web 端上线！访问 https://chat.deepseek.com/ 并开启 “Internet Search” 以获取实时答案。 🕒🧵(1/3)</li><li><a href="https://x.com/paulgauthier/status/1866519827814428846">Tweet from Paul Gauthier (@paulgauthier)</a>: 来自 Paul Gauthier (@paulgauthier) 的推文：Aider v0.68.0 帮助你在遵守服务条款 (TOS) 的前提下，通过 Web Chat 与“大模型” LLM 高效地复制和粘贴代码。使用更小、更便宜的（本地）模型来应用来自 LLM Web 的编辑...</li><li><a href="https://openrouter.ai/models">Models | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Aider LLM 排行榜：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://platform.deepseek.com/">DeepSeek Platform</a>: DeepSeek 平台：加入 DeepSeek API 平台以访问我们的 AI 模型、开发者资源和 API 文档。</li><li><a href="https://github.com/deepseek-ai/awesome-deepseek-integration/blob/main/README.md">awesome-deepseek-integration/README.md at main · deepseek-ai/awesome-deepseek-integration</a>: 通过在 GitHub 上创建账号来为 deepseek-ai/awesome-deepseek-integration 的开发做出贡献。</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: 聊天模式：使用 code, architect, ask 和 help 聊天模式。</li><li><a href="https://aider.chat/docs/usage/tips.html">Tips</a>: 技巧：使用 aider 进行 AI 结对编程的技巧。</li><li><a href="https://github.com/ai-christianson/RA.Aid">GitHub - ai-christianson/RA.Aid: Aider in a ReAct loop</a>: ReAct 循环中的 Aider。通过在 GitHub 上创建账号来为 ai-christianson/RA.Aid 的开发做出贡献。</li><li><a href="https://github.com/codestoryai/sidecar">GitHub - codestoryai/sidecar: Sidecar is the AI brains for the Aide editor and works alongside it, locally on your machine</a>: Sidecar 是 Aide 编辑器的 AI 大脑，在你的机器本地运行并与其协同工作 - codestoryai/sidecar</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h7sjyt/windsurf_cascade_leaked_system_prompt/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://esbuild.github.io/api/#splitting,">esbuild - API</a>: 未找到描述</li><li><a href="https://www.youtube.com/live/aLKNpxUuFK4?si=rFIwx-ugXeNbUhwj&t=10754">day #63 to 100x-orchestrator</a>: 加入 techfren，看他利用软件工程专业知识尝试和评测新技术</li><li><a href="https://www.youtube.com/live/vUbPnNeN9eY?si=4VWmTyarZlkuCrSL&t=2833">Aider + 1 Hour = Reels video editor with face recognition</a>: 加入 techfren，看他利用软件工程专业知识尝试和评测新技术</li><li><a href="https://api-docs.deepseek.com/quick_start/pricing">Models &amp; Pricing | DeepSeek API Docs</a>: 模型与定价 | DeepSeek API 文档：下面列出的价格以每 1M tokens 为单位。Token 是模型识别的最小文本单位，可以是一个单词、一个数字，甚至是标点符号。我们将根据总额计费...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1315769980414857218)** (29 条消息🔥): 

> `大型代码库的最佳实践、Aider 与 Language Servers 的集成、在命令行之外使用 Aider、在 Aider 中处理 System Prompts、Claude 模型之间的差异` 


- **大型代码库的最佳实践**：成员们讨论了在大型生产代码库中使用 Aider 的最佳实践，包括使用 [repo map](https://aider.chat/docs/repomap.html) 以获得更好的上下文。
   - 一位用户分享了 Aider FAQ 的链接，介绍了如何在大型 mono-repos 中利用 Aider，强调了其在管理大规模代码库方面的高效性。
- **Aider 与 Language Servers 的集成**：一位成员询问了如何将 Aider 与 Language Servers 集成，以便利用 LSP 功能（如“查找引用”和“转到定义”）来增强代码探索。
   - 另一位用户强调，Aider 需要一种更自主的方法，以便在无需大量手动控制的情况下促进复杂的代码更改。
- **在命令行之外使用 Aider**：有人提问是否可以通过 API 请求而不是仅通过命令行来使用 Aider，从而实现与本地文件更灵活的交互。
   - Paul Gauthier 确认 Aider 可以通过命令行或 Python 进行脚本化，文档提供了各种命令选项。
- **在 Aider 中处理 System Prompts**：用户表示有兴趣修改 Aider 使用的默认特定于编码的 System Prompts，以便用于头脑风暴等更通用的应用。
   - 建议可以从 Markdown 文件加载自定义规范（conventions），从而对 Aider 在交互过程中遵循的指南进行更大程度的控制。
- **Claude 模型之间的差异**：成员们讨论了 `anthropic/claude-3.5-sonnet` 及其在 OpenRouter 上的 Beta 版本之间的区别，强调了审核（moderation）方面的差异。
   - 社区强调了在 Aider 中使用非 Beta 模型的潜在优势，因为它的审核处理更完善，且遇到错误的可能性较低。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>：关于 Aider 的常见问题。</li><li><a href="https://aider.chat/docs/repomap.html">Repository map</a>：Aider 使用 Git 仓库地图为 LLM 提供代码上下文。</li><li><a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-aider-uses,">FAQ</a>：关于 Aider 的常见问题。</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>：你可以通过命令行或 Python 对 Aider 进行脚本化。</li><li><a href="https://aider.chat/docs/usage/watch.html">Aider in your IDE</a>：Aider 可以在浏览器中运行，而不仅仅是在命令行中。</li><li><a href="https://aider.chat/docs/usage/conventions.html">Specifying coding conventions</a>：让 Aider 在处理代码时遵循你的编码规范。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1315774665024868433)** (161 条消息🔥🔥): 

> `Llama 3.3 超长上下文, Sora 模型讨论, 微调 Qwen 模型, 量化模型性能, 学生的教育访问权限` 


- **Llama 3.3 实现超长上下文长度**：Unsloth 现在支持在 **80GB GPU** 上微调上下文长度高达 **89,000 tokens** 的 Llama 3.3 模型，大幅超越了以往的能力。
   - 该功能允许用户在 A100 GPU 上实现 **每训练步 2-3 分钟** 的速度，通过减少 **70% 的 VRAM** 占用率来提升性能。
- **关于 Sora 模型的讨论**：成员们对 **Sora 模型** 的看法褒贬不一，指出其训练令人印象深刻，但缺乏实际应用场景。
   - 尽管一些用户充满热情，但也有人担心与现有架构相比，它可能不会增加显著价值。
- **Qwen 模型在 OCR 任务中的潜力**：大家对微调 **Qwen2-VL** 用于 **OCR 任务** 感到兴奋，特别是针对护照等证件的信息提取。
   - 用户表示，由于该模型的能力，这种方法将非常有效。
- **量化模型性能的挑战**：一位用户报告称，在 **8-bit 微调后** 将模型转换为 **GGUF 格式** 时，尽管训练期间的评估损失（evaluation loss）较低，但性能出现了显著下降。
   - 讨论表明，问题可能源于模型合并和转换的方式，从而影响了其有效性。
- **教育用户对 GPU 资源的访问**：一名学生询问了增加用于研究目的的 GPU 访问选项，理由是目前仅支持 **单 GPU**。
   - 官方澄清，虽然 Unsloth 加速了单 GPU 上的训练，但目前尚不支持 **multi-GPU**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pastebin.com/kd9jgcE2">Endpoint encountered an error.You can try restarting it using the &quot;retry&quot; butt - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以让你在线存储文本一段时间的网站。</li><li><a href="https://x.com/UnslothAI/status/1866545164140810603">Unsloth AI (@UnslothAI) 的推文</a>: Llama 3.3 超长上下文微调现已发布！🦙Unsloth 现在支持在 80GB GPU 上为 @AIatMeta 的 Llama 3.3 (70B) 提供 89K 上下文——比 HF+FA2 长 13 倍。对于 Llama 3.1 (8B)，Unsloth 能够...</li><li><a href="https://unsloth.ai/blog/llama3-3">使用 Unsloth 微调 Llama 3.3</a>: 微调 Meta 的 Llama 3.3 (70B) 模型，其性能优于 GPT-4o，通过 Unsloth 开源提速 2 倍！新手友好。现已支持 Apple 的 Cut Cross Entropy 算法。</li><li><a href="https://huggingface.co/mistralai/Mistral-Small-Instruct-2409">mistralai/Mistral-Small-Instruct-2409 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct/tree/main">LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct at main</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hbaioc/llama_33_70b_finetuning_now_with_90k_context/">Reddit - 深入了解任何内容</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/damjan-k/rslora">Rank-Stabilized LoRA: 释放 LoRA 微调的潜力</a>: 未找到描述</li><li><a href="https://huggingface.co/papers/2310.08659">论文页面 - LoftQ: 针对大语言模型的 LoRA 微调感知量化</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/263">我们可以使用 Unsloth 训练奖励模型吗？ · Issue #263 · unslothai/unsloth</a>: 更多是一个问题而非 bug —— 你们是否也会开发一些使用 Unsloth 训练奖励模型（Reward Models）的示例？</li><li><a href="https://github.com/unslothai/unsloth/pull/1289">由 shashikanth-a 添加对 Apple Silicon 的支持 · Pull Request #1289 · unslothai/unsloth</a>: 未优化。尚不支持 GGUF。从源码构建 Triton 和 bitsandbytes。使用 cmake -DCOMPUTE_BACKEND=mps -S . 构建 bitsandbytes。pip install unsloth-zoo==2024.11.4，pip install xformers==0.0.25
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1315775003006074880)** (85 条消息🔥🔥): 

> `Unsloth 模型安装，Finetuning Gemma 2，CUDA/Triton Kernel 开发，长文本生成问题，在非对话任务中使用 Guidance AI` 


- **WSL 中的 Unsloth 安装问题**：由于 Python 版本依赖关系，用户在 WSL 中通过 pip 安装 Unsloth 时遇到问题，一些人建议使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 创建兼容环境。
   - 一位用户分享了相关的错误，并提出了在 GitHub 讨论中找到的解决方案，指出错误通常源于版本问题。
- **Gemma 2 Finetuning 的挑战**：一位用户在尝试对 Gemma 2 进行 Finetuning 时遇到了量化错误，随后收到了编译最新版本 llama.cpp 以解决问题的指令。
   - 另一位贡献者报告称，按照这些指令操作并未解决他们的问题，并分享了他们在 CUDA 错误方面的经历。
- **CUDA 和 Triton Kernel 开发资源**：一位用户寻求学习 CUDA 和 Triton Kernel 开发的资源，并收到了阅读 Unsloth 团队入门博客文章的建议。
   - 此外，他们被引导加入 GPU Mode 社区以获取进一步指导。
- **长文本生成问题**：一位从事长文本生成的用户在特定数据上训练模型时遇到了重复输出的问题，并对模型处理输入的方式提出了质疑。
   - 他们收到了关于数据集的非对话性质对训练结果潜在影响的建议。
- **使用 Guidance AI 进行结构化输入**：有关于在非对话任务中为自定义数据集使用 Guidance AI 的咨询，用户正在讨论是否需要包含一个模拟文本列。
   - 用户建议在实现自定义 data collators 时添加模拟列，以绕过 UnslothTrainer 施加的限制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/introducing">Introducing Unsloth</a>：未找到描述</li><li><a href="https://docs.vllm.ai/en/v0.5.5/models/lora.html">Using LoRA adapters &#8212; vLLM</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">All Our Models | Unsloth Documentation</a>：查看下方列表，了解我们上传的所有 GGUF、16-bit 和 4-bit bnb 模型</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/748#issuecomment-2238395604">RuntimeError: Unsloth: The file &#39;llama.cpp/llama-quantize&#39; or &#39;llama.cpp/quantize&#39; does not exist · Issue #748 · unslothai/unsloth</a>：在尝试将模型转换为 GGUF 格式时发生了以下错误。我注意到 quantized 文件夹位于 llama.cpp/examples/quantize RuntimeError: Unsloth: The file &#39;llama.cpp/llama-quanti...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1315784089164316774)** (6 条消息): 

> `优秀的 RAG 项目，深入探讨角色和卡片，受限生成技术` 


- **GitHub 上的优秀 RAG 项目**：一位用户分享了 [Awesome RAG](https://github.com/lucifertrj/Awesome-RAG) 项目的链接，该项目专注于 RAG、VectorDB、embeddings、LlamaIndex 和 Langchain。
   - 该 GitHub 仓库邀请大家为增强项目做出贡献。
- **探索角色和性格卡片**：一位用户建议深入探讨 system、user 和 assistant 的角色，以及性格卡片和审核技术。
   - 他们强调了理解这些概念对于更好的 AI 交互设计的重要性。
- **理解受限生成**：另一位用户建议解释使用 JSONSchema 和 grammar 的 **受限生成（constrained generation）**，用于代码改进和特征提取的应用。
   - 他们暗示这些技术对于在 AI 中实现最佳的 function calling 至关重要。



**提到的链接**：<a href="https://github.com/lucifertrj/Awesome-RAG/">GitHub - lucifertrj/Awesome-RAG: RAG-VectorDB-Embedings-LlamaIndex-Langchain</a>：RAG-VectorDB-Embedings-LlamaIndex-Langchain。通过在 GitHub 上创建账户为 lucifertrj/Awesome-RAG 的开发做出贡献。

  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1315804358050578445)** (4 条消息): 

> `针对 LLM 的 APOLLO 优化器、QTIP 量化方法、WizardLM Arena 论文的数据集仓库` 


- **APOLLO 优化器提出内存效率方案**：最近的一篇研究论文提出了 **APOLLO** 优化器，它通过近似学习率缩放，来缓解像使用 **AdamW** 训练 LLM 时的高内存占用特性。
   - 该方法旨在降低优化器内存开销，同时保持具有竞争力的性能，尽管传统方法在内存使用方面面临挑战。
- **QTIP 增强训练后量化 (Post-Training Quantization)**：**QTIP** 方法引入了 trellis coded quantization 来优化 LLM 的高维量化，从而改善 **memory footprint** 和推理吞吐量。
   - 凭借其创新方法，QTIP 允许模型进行有效的微调，克服了以往 vector quantization 方法所面临的局限性。
- **WizardLM Arena 数据集现已发布**：包含 **WizardLM Arena** 论文中使用的所有数据集的仓库已上传，方便研究人员访问这些资源。
   - 数据集可以在 [Hugging Face](https://huggingface.co/datasets/forcemultiplier/arena-paper-datasets-jsonl/tree/main) 上找到，旨在促进进一步的实验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.05270">APOLLO: SGD-like Memory, AdamW-level Performance</a>: 大型语言模型 (LLMs) 在训练期间以内存密集著称，尤其是使用流行的 AdamW 优化器时。这种内存负担使得必须使用更多或更高端的 GPUs，或者减少 ...</li><li><a href="https://huggingface.co/datasets/forcemultiplier/arena-paper-datasets-jsonl/tree/main">forcemultiplier/arena-paper-datasets-jsonl at main</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2406.11235">QTIP: Quantization with Trellises and Incoherence Processing</a>: 训练后量化 (PTQ) 通过将权重量化为低精度数据类型来减少 LLMs 的 memory footprint。由于 LLM 推理通常受内存限制，PTQ 方法可以提高推理 ...</li><li><a href="https://github.com/Cornell-RelaxML/qtip/blob/main/quantize_llama/quantize_finetune_llama.py">qtip/quantize_llama/quantize_finetune_llama.py at main · Cornell-RelaxML/qtip</a>: 通过在 GitHub 上创建账号来为 Cornell-RelaxML/qtip 的开发做出贡献。</li><li><a href="https://www.cs.cornell.edu/~cdesa/">Chris De Sa</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1315770394254512158)** (238 messages🔥🔥): 

> `图像增强技术, Stable Diffusion 的使用, Llama 3.2-Vision 模型, WebUI 中的内存管理, 图像元数据与标签` 


- **关于图像增强技术的讨论**：成员们辩论了像 Stable Diffusion 这样的 AI 模型是否能在不改变核心内容的情况下改进图像，并建议使用 Photoshop 等传统编辑工具来完成此类任务。
   - 一些成员强调了调色和光照技能对于专业结果的重要性，指出 AI 可能会增加噪声而不是细化图像。
- **探索 Llama 3.2-Vision 模型**：Llama 3.2-Vision 模型被提及为图像分类和分析的一个可行的本地选项，通过 KoboldCPP 等软件支持视觉模型。
   - 成员们指出本地模型可以在消费级 GPU 上运行，并讨论了替代方案，强调在线服务通常要求用户放弃其数据权利。
- **Automatic1111 WebUI 中的内存管理**：讨论了影响 Automatic1111 WebUI 中图像生成的内存管理问题，特别是 Batch sizes 和 VRAM 使用情况。
   - 成员们建议，较大的 Batch 导致了显存溢出（Out-of-memory）错误，这可能是由于系统中存储 Prompts 的方式效率低下造成的。
- **图像元数据与标签**：参与者讨论了从图像中提取标签或描述的挑战，建议包括使用元数据读取器或 AI 模型进行分类。
   - 有人担心分类方法可能会遗漏某些细节，一些人主张使用类似于 Imageboards 上的特定标签。
- **AI 服务中的版权与数据权利**：分享了关于使用在线 AI 图像生成服务的警告，强调此类服务通常对用户生成的内容主张广泛的权利。
   - 成员们鼓励使用本地模型，以保持对创作作品更清晰的所有权和控制权，这与基于 Web 的服务的广泛许可实践形成鲜明对比。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://civitai.com/articles/545/how-to-avoid-vram-exhaustion-and-prevent-vram-spikes">如何避免 VRAM 耗尽并防止 VRAM 峰值！ | Civitai</a>：本指南中包含了为什么要使用它的示例。点击此处查看 @fitCorder 提供的详细图像指南。非常酷。什么是 VRAM？...</li><li><a href="https://tenor.com/view/facepalm-face-palm-picard-trek-gif-15072590366303305471">捂脸 Picard GIF - Facepalm Face Palm - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/lllyasviel/IC-Light">GitHub - lllyasviel/IC-Light: 更多重光照！</a>：更多重光照！通过在 GitHub 上创建账号为 lllyasviel/IC-Light 做出贡献。</li><li><a href="https://www.youtube.com/watch?v=t5nSdosYuqc">A1111 的 Multi Diffusion - 超大尺寸 + 低 VRAM 放大</a>：带有 Tiled VAE 的 Multi Diffusion Tiled Diffusion 放大器。这款低 VRAM 放大器可以在 A1111 中为你提供超高质量的 8 倍放大。它适用于...</li><li><a href="https://www.youtube.com/watch?v=f-EIuGROTEo">使用 ChatGPT 和 Hailuoai 的 AI 魔法创建病毒式传播的对口型视频！</a>：✨ 漫画书创建器自定义 ChatGPT https://payhip.com/b/TgUxN ✨ 终极文本转视频提示词生成器 . https://payhip.com/b/nq6b4 创建病毒式传播的对口型...</li><li><a href="https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111?tab=readme-ov-file#tiled-diff-txt2img">GitHub - pkuliyi2015/multidiffusion-upscaler-for-automatic1111: Tiled Diffusion 与 VAE 优化，采用 CC BY-NC-SA 4.0 许可</a>：Tiled Diffusion 与 VAE 优化，采用 CC BY-NC-SA 4.0 许可 - pkuliyi2015/multidiffusion-upscaler-for-automatic1111
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1315773563873136740)** (219 messages🔥🔥): 

> `Perplexity AI Image Generation, Claude and GPT Models, Custom GPTs Functionality, Perplexity Pro Subscription, AI Tools and Resources` 


- **Perplexity AI 在图像生成方面表现不佳**：用户在使用 Perplexity App 的“Generate Image”功能时遇到困难，通常发现该功能在某些屏幕方向下被隐藏或无响应。
   - 一位用户通过将手机方向切换为横屏模式解决了问题，从而显示出了该功能。
- **Claude 与 GPT 模型对比**：Claude 模型被认为在写作方面非常有效，但用户认为在 Perplexity 中使用这些模型的效果可能不如直接在官方网站上使用。
   - Pro 用户通常认为 Claude 等 AI 模型的付费版本更有益，因为它们具有增强的功能。
- **探索 Custom GPTs**：Custom GPTs 允许用户编辑人格特质和引导设置，从而增强交互体验。
   - 一位参与者表示有兴趣尝试 Custom GPT 选项，以整理思路和开发创意。
- **对 AI 能力的担忧**：用户对开源和部分商业 AI 模型的能力表示怀疑，指出它们在解决复杂问题时表现有限。
   - 参与者达成共识，认为许多 AI 模型在处理复杂任务时未达到预期。
- **Perplexity Pro 订阅功能**：成员们讨论了 Perplexity Pro 计划的优势，强调了其与免费版相比的广泛功能。
   - 参与者分享了折扣推荐码，表明了对增强研究和编程能力的订阅服务的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://perplexity.discount/">Perplexity Pro 减免 $10 - 折扣推荐链接</a>：首月 Perplexity Pro 享受 $10 优惠。使用此推荐链接体验来自 Perplexity AI 的 AI 驱动搜索与分析。</li><li><a href="https://chat.deepseek.com">DeepSeek</a>：与 DeepSeek AI 聊天。</li><li><a href="https://tenor.com/view/dr-austin-powers-evil-one-gif-14681923667046200996">Dr Austin GIF - Dr Austin Powers - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1315896764896641044)** (8 messages🔥): 

> `OpenAI's Sora release, Bitcoin reaching $100K, World's largest gold deposit, Perplexity AI updates, AI and monitoring in 2025` 


- **OpenAI 的 Sora 终于发布**：根据最近的讨论，**OpenAI Sora** 终于面世，这让 AI 社区的许多人感到兴奋。
   - 一位成员分享了一个 **[YouTube 视频](https://www.youtube.com/embed/FSMjXW_KrAo)**，详细介绍了其功能和影响。
- **Bitcoin 达到 10 万美元里程碑**：讨论中提到的一个显著事件是 **Bitcoin** 达到 **$100K**，标志着加密货币市场的一个重要里程碑。
   - 这一飙升促使许多成员推测 **cryptocurrencies** 的未来。
- **发现全球最大金矿**：频道成员讨论了关于发现**全球最大金矿**的惊人消息，这可能产生重大经济影响。
   - 对话包括对其对全球金价和采矿作业潜在影响的参考。
- **Perplexity AI 更新追溯至各种查询**：几位用户分享了 **Perplexity AI** 相关信息的链接，包括关于功能和特性的特定查询。
   - 一位用户分享了一个直接链接，突显了用户对提高 **Perplexity AI** 参与度的持续兴趣。
- **2025 年的 AI 趋势与监控**：一位成员分享了一个讨论 **AI and monitoring in 2025** 的链接，探讨了未来的技术格局及其影响。
   - 这引发了关于未来 AI 发展中的挑战与机遇的问题。



**提到的链接**：<a href="https://www.youtube.com/embed/FSMjXW_KrAo">YouTube</a>：未找到描述内容

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1315776472752984085)** (2 messages): 

> `服务恢复` 


- **服务重新上线**：一位成员注意到服务似乎已经恢复，并对修复问题的人员表示感谢。
   - 这表明之前的停机问题已得到解决，用户可以恢复正常操作。
- **社区致谢**：另一位成员对参与恢复服务的人员所做的努力表示认可，反映了社区内的积极情绪。
   - 这种感激之情反映了社区在处理服务中断时的协作性质。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1316100269297242223)** (1 messages): 

> `Canvas 更新，12 Days of OpenAI` 


- **令人兴奋的 Canvas 更新发布**：在名为“Canvas—12 Days of OpenAI: Day 4”的 **YouTube 视频**中，Kevin Weil、Lee Byron 和 Alexi Christakis 介绍并演示了 **Canvas** 功能的**更新**。
   - 在[此处](https://www.youtube.com/live/qZ0ImE41pVs?si=P74Rr7NHmBE2inyX)查看完整演示。
- **加入 12 Days of OpenAI 活动**：通过在 <id:customize> 中领取 <@&1261377106890199132> 角色，随时了解 **12 Days of OpenAI** 的动态。
   - 确保不要错过任何令人兴奋的更新或活动！



**提到的链接**：<a href="https://www.youtube.com/live/qZ0ImE41pVs?si=P74Rr7NHmBE2inyX">Canvas—12 Days of OpenAI: Day 4</a>：Kevin Weil、Lee Byron 和 Alexi Christakis 介绍并演示 Canvas 的更新。

  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1315775493395451944)** (149 messages🔥🔥): 

> `Sora 生成推测、AI 模型对比、LLM 能力、新功能用户体验` 


- **对 Sora 功能的褒贬不一**：用户对 Sora 的能力表示怀疑，质疑生成内容的质量，并暗示其可能仅仅是库存素材。
   - 一些人注意到输出时长较短（如 5 秒视频生成），认为这限制了实质性内容的创作。
- **AI 模型及其性能对比**：围绕不同 AI 模型的讨论凸显了关于 Claude 与 Sora 的观点，一些用户声称 O1 在特定任务上表现更优。
   - 用户分享了使用 Leonardo 和 Ideogram 等模型的经验，指出与 Sora 相比，这些模型更易于使用且输出质量更高。
- **教导 AI 学习工具的挑战**：有人担心是否应该教导像 O1-mini 这样的模型使用工具，还是让它们通过交互在上下文中学习。
   - 参与者讨论了 AI 在与用户交互时，新功能和能力需要上下文所带来的影响。
- **对 AI 能力的期望**：有一场关于 AI 未来的对话，包括模型动态生成用户界面并在没有明确指令的情况下响应用户需求的潜力。
   - 一些参与者质疑将每次交互都变为 AI 驱动的可行性和实用性，理由是存在困惑和可用性问题。
- **用户对 AI 创新的反馈**：用户对 AI 发展的速度和性质发表了不同看法，对没有具体成果的模糊承诺表示沮丧。
   - 随着讨论的展开，一些人指出需要更清晰的愿景，来说明这些进步将如何转化为可用的技术。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1315869531578564679)** (23 条消息🔥): 

> `Sora 账号问题，GPT-3.5 开发困境，域名验证问题，向客户退款，与版主沟通` 


- **Sora 账号创建不可用**：用户在创建 Sora 账号时遇到问题，一些人报告称尽管拥有付费账号，但目前仍无法创建账号。
   - *“因为它过载了，因为每个人都想尝试，”* 一位用户指出，并建议稍后再查看。
- **GPT-3.5 在开发中的困境**：一位开发者在构建应用时表达了对 GPT-3.5 的挫败感，指出其运行异常，并纠结是该拖延客户还是直接退款。
   - 另一位参与者建议，从长远来看，退款给客户可能更有利，并建议对自己的技能水平保持诚实。
- **域名验证障碍**：一位用户在验证域名时遇到错误，在收到验证令牌过期的消息后询问如何继续。
   - 另一位用户建议在多次尝试失败后重新开始以解决验证问题。
- **客户退款的伦理**：围绕向已为未完成应用付费的客户退款的伦理问题展开了讨论，强调扣留他们的钱是不对的。
   - 一位用户指出，任何 AI 都无法弥补在交付承诺产品方面缺乏经验的问题。
- **版主与沟通规则**：一位用户被提醒要保持回复的相关性，并表达自己的想法，而不是仅仅依赖 GPT-3.5 的输出。
   - 这引发了一场关于使用 AI 生成的回复可能导致版主困惑的后果的轻松交流。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1315830424882122752)** (12 条消息🔥): 

> `Custom GPTs 连贯性，嵌套代码块，翻译效果，OpenAI API 模型微调` 


- **保持 Custom GPTs 的连贯性**：一位成员解释说，你可以启动新的 Custom GPTs，并向旧的 GPTs 索要关键摘要以合成连贯性。
   - *“Custom GPTs 在作者更新时会丢失工具连接”*，这使其成为一个持续的挑战。
- **嵌套代码块的技巧**：一位成员分享说，使用 **两个反引号** 是代码块正确嵌套的关键，这有助于正确渲染。
   - 他们提供了一个 YAML 和 Python 代码示例，演示了 *内部双反引号* 的用法。
- **翻译与 Prompt 有效性**：有人指出，当预期响应是外语时，用英文编写的 Prompt 可能会产生更好的输出。
   - 模型在英文方面的训练更强，这可能促成了这一优势，因为它已经有效地翻译了许多语言。
- **微调 OpenAI 模型的问题**：一位成员分享了他们在 Node.js 中微调模型的困扰，报告称即使在微调后，模型仍返回通用的答案。
   - 他们请求协助检查其训练 JSONL 文件以诊断潜在问题，表示需要外部验证。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1315830424882122752)** (12 条消息🔥): 

> `Custom GPTs 更新，嵌套代码块，微调模型，翻译质量，连贯性合成` 


- **Custom GPTs 在更新时丢失连接**：一位成员指出，**Custom GPTs** 在作者更新时会丢失工具连接，强调了需要持续管理的方法。
   - *“如果我想合成连贯性，我就是这么做的。”*
- **嵌套代码块技术**：一位用户对 **ChatGPT** 无法正确输出嵌套代码块表示沮丧，特别希望能得到单一代码块的响应。
   - 另一位成员建议使用 **双反引号** 进行嵌套，并展示了一个嵌套代码块的示例。
- **翻译输出的细微差别**：有人询问当目标语言不同时，用 **英文** 编写的 Prompt 是否会产生更好的输出。
   - 一位成员建议，主要基于英文训练的模型在 Prompt 以英文表述时往往表现更好。
- **微调模型的麻烦**：一位用户报告了在 Node.js 中微调其 **基于 OpenAI 的应用** 时遇到的困难，指出模型无法学习上下文。
   - 他们寻求关于其训练 JSONL 的帮助，旨在识别其方法中的潜在问题。


  

---

### **Bolt.new / Stackblitz ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1315774549543096390)** (27 条消息🔥): 

> `Prompting 规范，使用 Bolt 进行订阅管理，Shopify 自动化，文档扫描问题，与 Airtable 的集成` 


- **对出色 Prompting 规范的需求**：讨论中提到需要一套 **1000% 出色的 Prompting 规范**，以简化输入并减少在 Bolt 中使用 Token 带来的挫败感。
   - 成员们对有效 Prompting 的最佳实践表现出兴趣，以提升用户体验。
- **探索使用 Bolt 进行订阅管理**：一位用户询问了是否可以使用 Bolt 处理 **订阅管理** 以及与 **Stripe** 或 **Recurly** 等平台的 Webhooks。
   - 虽然没有提供确切的解决方案，但鼓励用户探索相关集成。
- **Shopify 自动化问题**：关于 **Shopify 自动化** 流程的问题，成员们讨论了集成 API 进行产品同步的需求。
   - 一位特定用户表示他们正在构建一个 **内部仪表板**，需要产品同步功能以便更好地管理。
- **Bolt 中的文档扫描问题**：用户报告了在应用中 **扫描和上传文档** 时出现的错误，特别是与 OCR 功能相关的问题。
   - 建议包括在将文档上传到 Bolt 之前，先将其转换为可读格式。
- **Airtable 与 Bolt 的集成**：一位用户提到他们的库存是在 **Airtable** 上管理的，并与使用 Bolt 创建的 Web App 同步。
   - 出现了关于 Bolt 如何处理此类集成以及它是否能有效管理连接数据的疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/sulco/status/1866519508082688447">来自 Tomek Sułkowski (@sulco) 的推文</a>: 你知道可以通过在 URL 中添加 "?prompt=" 来启动 Bolt 项目吗？💡 #bolttip 它还允许与 Bolt 进行各种酷炫的轻量级集成；例如，你可以动态地...</li><li><a href="https://shopify.dev/docs/api">Shopify API、库和工具</a>: 了解 Shopify API、库和工具，并为你的用例选择正确的选项。
</li>
</ul>

</div>
  

---

### **Bolt.new / Stackblitz ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1315782674081648701)** (114 条消息🔥🔥): 

> `Token 问题与订阅支持、支付网关集成、使用多个 LLM、图片上传问题、排查“无预览可用”故障` 


- **Token 问题与订阅支持**：用户对订阅结束后免费 Token 的情况感到困惑，部分用户收到了意外的 Token 分配。有人建议如果出现账单问题请联系支持部门。
   - *Token 不会累加*，且 Pro 计划的 Token 每 30 天重置一次，这一点已得到其他用户的确认。
- **支付网关集成**：围绕将 Payfast 和 PayStack 等支付网关与 Bolt 集成展开了讨论。总的来说，用户不确定集成过程是否与 Stripe 类似。
   - 一位用户表示需要明确分离仪表板（dashboard）功能是否能提高大型项目的性能。
- **使用多个 LLM**：一位用户询问是否可以在 Bolt 中同时利用多个 LLM 以满足复杂的项目需求。另一位成员确认目前尚不支持此功能。
   - 用户探索了在大型代码库中提高生产力和管理能力的潜在方法。
- **图片上传问题**：用户报告了本地图片在 Bolt 中无法正常显示的问题，导致在消耗 Token 后仍未成功。建议的解决方案包括使用外部服务进行上传。
   - 分享了一份关于如何在应用中正确集成图片上传功能的指南。
- **排查“无预览可用”故障**：部分用户在修改后项目无法正常加载时遇到了“No Preview Available”错误。用户建议创建专门的讨论话题以进行深入排查。
   - 一位成员分享了他们的排查步骤，包括重新加载以及寻求错误消息方面的帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/stackblitz/status/1861064144532951526">来自 StackBlitz (@stackblitz) 的推文</a>：为了开启这个感恩周，我们的团队为大家准备了一份有趣的节日礼物：我们称之为 TURKEY TOKENS！ 🦃🪙 截止到 11 月 30 日：🦃 所有 Pro 用户可获得 200 万免费 Token！🦃 所有免费用户...</li><li><a href="https://uploadthing.com/">uploadthing</a>：一种更简单的文件上传方式。</li><li><a href="https://support.bolt.new">Notion – 集笔记、任务、维基和数据库于一体的工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://Bolters.io">Bolters.io | Bolt.new 无代码应用构建器的社区支持技巧、诀窍和知识库</a>：Bolt.new 的文档和指南</li><li><a href="https://github.com/stackblitz/bolt.new">GitHub - stackblitz/bolt.new: Prompt, run, edit, and deploy full-stack web applications</a>：提示、运行、编辑和部署全栈 Web 应用程序 - stackblitz/bolt.new</li><li><a href="https://youtu.be/CRZm7zNNBcw?si=eZwLfhPj1m8_i8MC">使用 Bolt 构建，第 3 集：两个超级方便的 Bolt 技巧 - 锁定文件和提取</a>：在本集中，我分享了两个我认为在 Bolt.new 中最有用的功能，其中之一是锁定——对于任何想要...的人来说，这应该是首选。</li><li><a href="https://x.com/stackblitz/">来自 GitHub 的推文 - FixTweet/FxTwitter: Fix broken Twitter/X embeds!</a>：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1315769959904706640)** (45 条消息🔥): 

> `Swag 挑战赛获胜者、论坛用户参与度、网络中断问题、Mojo 语言类型系统、Hugging Face 集成` 


- **Swag 挑战赛获胜者名单公布**：第一天 Swag 挑战赛的获胜者已在论坛公布。你可以在[这里](https://forum.modular.com/t/winners-of-day-1-swag-challenge/189)查看详情。
   - 为加入这一盛况的 T-shirt 获奖者们欢呼！
- **鼓励论坛参与**：提醒成员，为了保持论坛质量，任何明显的 AI 生成内容都将被删除。此举旨在培养一个有趣且真实的讨论场所。
   - 参与者对这一举措表示感谢，并强调了真实性的重要性。
- **网络中断导致权重问题**：讨论指出，在网络中断期间，由于缺乏验证，模型可能会使用错误的权重。示例输出展示了此类场景下数据出现的离奇损坏。
   - 该问题已通过在下载过程中加入 Checksums（校验和）得到解决，提高了可靠性。
- **Mojo 语言中的类型问题**：有人对 Mojo 的类型系统不是强类型表示担忧，因为需要手动声明变量类型。这引发了关于其与 Rust 等语言相比复杂性的讨论。
   - 用户强调了类型推断和函数重载的挑战，并将其与 Rust 的功能进行了对比。
- **Hugging Face 增强功能**：与 `huggingface_hub` 的集成实现了中断下载的自动恢复，提高了鲁棒性。这一改进是在早期版本中遇到大权重文件损坏问题后进行的。
   - MAX Graph 流水线中的更新利用 Hugging Face 来获得更好的性能和可靠性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://forum.modular.com/t/winners-of-day-1-swag-challenge/189">Winners of Day 1 swag challenge! 🏆</a>：恭喜我们第一天 Swag 挑战赛的获胜者！请留意私信以协调 T-shirt 的递送。 @lesoup-mxd @Zbornak @martinvuyk @IvoB @melodyogonna @sazid @Dasor @tristanbiesecke...</li><li><a href="https://forum.modular.com/search?q=eee">Search results for 'eee' - Modular</a>：未找到描述</li><li><a href="https://github.com/modularml/max.git">GitHub - modularml/max: A collection of sample programs, notebooks, and tools which highlight the power of the MAX Platform</a>：一系列示例程序、Notebook 和工具，旨在展示 MAX 平台的强大功能 - modularml/max
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1315811343600848978)** (83 messages🔥🔥): 

> `Mojo 中的 destroy 关键字，Multi-Paxos 中的内存管理，Ownership 语义，struct 析构函数的优缺点，Multi-Paxos 的实现挑战` 


- **理解对 'destroy' 关键字的需求**：关于 Mojo 中引入新的 `destroy` 关键字必要性的讨论，强调了它与 Python 的 `del` 的区别，重点在于线性类型（linear types）中更严格的使用，从而确保内存管理期间的安全性。
   - 一些成员表示，要求使用 `destroy` 可能会使从 Python 迁移过来的新手的默认行为变得复杂，强调了清晰度的重要性。
- **Multi-Paxos 中的内存管理**：Multi-Paxos 的实现强调了静态分配结构，以满足无堆分配（no-heap-allocation）的要求，这支持了对高性能至关重要的流水线操作。
   - 评论指出，需要全面处理 promise 和 leader 选举，以满足共识算法的要求，这对于功能的健壮性至关重要。
- **Mojo 中的 Ownership 语义**：关于 ownership 方法的对话要求明确 Mojo 类型应如何处理析构函数，特别是将拷贝和移动构造函数的默认处理与析构函数进行对比。
   - 像 `__del__`（析构函数）这样的元素被强调为可能会让从具有自动内存管理语言迁移过来的人感到困惑，从而加强了对一致语法的需求。
- **设计中的 Struct 析构函数**：有人对要求所有 struct 都有空的 `__del__` 方法的实用性提出了担忧，考虑到默认行为理想情况下应该能有效地处理内存管理，这似乎是不必要的。
   - 有观点认为应该像对待拷贝和移动方法一样对待析构函数，建议在编程的易用性和显式控制之间取得平衡。
- **实现 Multi-Paxos 的挑战**：对 Multi-Paxos 原型实现的反馈纠正了对其操作的误解，强调它必须考虑 leader 稳定性以及众多的健壮性特性。
   - 对话承认了共识协议设计的复杂性，主张加入超时和 leader 选举机制等基本特性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pypi.org/project/mojo/">mojo</a>: your_name_here</li><li><a href="https://docs.modular.com/mojo/manual/values/ownership">Ownership and borrowing | Modular Docs</a>: Mojo 如何通过函数参数共享引用。</li><li><a href="https://github.com/modularml/mojo/issues/3623">[Discuss] Resyntaxing argument conventions and References · Issue #3623 · modularml/mojo</a>: Mojo 引用子系统的设计正趋于完善。为了最终确定主要观点，有必要重新评估 Mojo 早期的一些决定，使设计更加...</li><li><a href="https://github.com/modularml/mojo/pull/3793">[docs] Stdlib insider documentation by owenhilyard · Pull Request #3793 · modularml/mojo</a>: 开发标准库的人员需要更多关于运行时和编译器内置函数的 API 契约和行为的信息，以便能够编写正确且高性能的...
</li>
</ul>

</div>
  

---

### **NotebookLM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1315781488763080784)** (43 条消息🔥): 

> `播客内容创作, 来源利用挑战, 语言设置, AI 播客生成, AI 社区参与` 


- **利用 NotebookLM 释放播客潜力**：一位成员分享了一个名为 [“NotebookLM 播客教程：10 个秘密提示词（别人会为此疯狂！）”](https://youtu.be/aG0ixD3OY80) 的 YouTube 视频，该视频提供了经过优化的独家播客提示词，旨在增强创意。
   - 该教程强调了多种独特的方法，旨在帮助 AI News 社区的用户脱颖而出。
- **对 NotebookLM 来源限制的困扰**：一位用户表达了挫败感，称他们的一篇论文需要 **15 个来源**，但 NotebookLM 在收到指令时仅使用了 **5-6 个**。
   - 社区分享了关于在查询时添加来源限制的建议，以确保引用的多样性。
- **社区对语言选项的需求**：一位用户询问如何在 NotebookLM 上将语言更改为英语，并强调由于即将到来的考试，此事非常紧迫。
   - 实用的建议包括调整浏览器设置并刷新 NotebookLM 页面，以实现所需的语言显示。
- **实验 AI 播客格式**：一位成员提到在他们的 AI 播客中加入了一个事实核查环节，从而提高了对话质量，并防止在 **90 分钟** 的节目中出现错误。
   - 他们分享了音频预告片，展示了这种创新播客创作方法的结果。
- **使用 NotebookLM 进行有趣的音频实验**：频道讨论了使用 NotebookLM 创建的各种音频剪辑，探索了不同的提示词和视角，以产生独特的结果。
   - 成员们分享了几个剪辑，展示了在引人入胜的播客格式中使用 AI 生成内容的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com/notebook/790effc7-cd34-4799-b9bd-319709b8d542/audio">未找到标题</a>: 未找到描述</li><li><a href="https://youtu.be/MC49i2APeQw">Punk Meets Processor AI Podcast</a>: 🎙️ 新剧集预告！🚨当一个朋克和一个 AI 聊天机器人坐下来进行一场过滤、毫无保留的对话时会发生什么？🤖🔥 “Punk Meets Processor” – th...</li><li><a href="https://notebooklm.google.com/notebook/8f4d88d7-fdbe-420b-9f0d-751c3196c8ab/audio">未找到标题</a>: 未找到描述</li><li><a href="https://www.bitchute.com/video/YTRo6Zx5JNTg">Amir Tsarfati 9/11 Deceiving and Being Deceived. From Such Turn Away! Let No Man Deceive You!</a>: 不要被欺骗！为 Amir 和那些相信谎言的人祈祷，愿他们能够悔改。愿上帝保佑并眷顾你们，愿耶稣基督的恩典与平安永远与你们同在！JD Faraq 视频； https...</li><li><a href="https://youtu.be/aG0ixD3OY80">NotebookLM Podcast Tutorial: 10 Secret Prompts (People Will Kill You For!)</a>: 免费获取这些独家的 NotebookLM 播客提示词！我花了数小时优化这 10 种独特的方法，以帮助 AI News 社区脱颖而出。只需观看...</li><li><a href="https://www.youtube.com/watch?v=ZaION5DwtIk&list=PLHkFW33YdghE9XlVylJv3LcCBwZu757SC&index=8">NotebookLM podcast-hack 09: dare to disagree # 4</a>: 一场关于技术乐观主义的高度相关的讨论，未达成共识。所有摘录均完全由 NotebookLM 生成，未经任何编辑。</li><li><a href="https://docs.google.com/document/d/13kx-D4mJAucmq8nJKz03XH3Vvqjv_3Sbpujd1A0n_yQ/edit?usp=drivesdk">Samurai Basics and General/Combat Feats</a>: 在 Naruto Pathfinder Conversion Mod 中，玩家可以选择成为一名武士。如果他们不是武士职业，他们将获得“挑战”能力，就像他们是同等级的武士一样...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1315782535464357910)** (51 条消息🔥): 

> `NotebookLM 功能、播客功能、用户体验问题、自定义查询、语言支持` 


- **NotebookLM 以只读方式保存笔记**：用户发现 NotebookLM 中的“已保存回复笔记”是只读的，需要手动打开才能编辑，这导致了对后续修改操作的困惑。
   - 一位用户对初始保存后无法修改已保存笔记表示沮丧。
- **播客功能对更多声音的需求**：教师们指出，目前的播客声音被比作“芭比和肯”，并询问是否会增加更多样化的声音选项。
   - 还有人提到其中一个声音很像 Katee Sackhoff，这对于教育内容来说感觉很奇怪。
- **播客功能的语言支持**：用户询问了制作不同语言播客的能力，特别是请求支持法语。
   - 讨论还包括请求提供更多语言（包括德语）上线的时间表。
- **共享问题排查**：用户在使用“复制链接”共享笔记本时遇到了挑战，导致接收者看到的是空白页面，除非他们先被添加为查看者。
   - 针对成功共享笔记本所需的步骤进行了说明。
- **优化播客长度**：一位用户询问如何缩短播客剧集，特别是目标时长为 1 分钟。
   - 还有人对如何使用 CustomPod 等工具自动获取播客内容感兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://custompod.io)">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/huh-camille-confused-puzzled-double-take-gif-24008735">Huh Camille GIF - Huh Camille 困惑 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.instagram.com/reel/DDaTgGKzG7o/?utm_source=ig_web_button_share_sheet&igsh=MzRlODBiNWFlZA==">Instagram 上的 &#x1f577;&#xfe0f;</a>: 0 个赞，0 条评论 - somebiohacker 发表于 2024 年 12 月 10 日</li><li><a href="https://pubmed.ncbi.nlm.nih.gov/15217335/">时间处理的神经基础 - PubMed</a>: 对感觉和运动处理的完整理解需要表征神经系统如何在几十到几百毫秒 (ms) 范围内处理时间。时间处理...</li><li><a href="https://youtu.be/QxbmQs3b_DE">NotebookLM 教程，让你的生产力提升 10 倍</a>: 想成为 NotebookLM 大师并将生产力提高 10 倍，只需观看此完整视频。我在一个视频中涵盖了从基础到高级的所有内容，并包含 2 个真实案例...</li><li><a href="https://youtu.be/aG0ixD3OY80">NotebookLM 播客教程：10 个秘密提示词（好用到爆！）</a>: 免费获取这些独家 NotebookLM 播客提示词！我花了几个小时完善这 10 种独特的方法，以帮助 AI News 社区脱颖而出。只需观看...</li><li><a href="https://www.speedofcreativity.org/2024/11/28/notebooklm-on-project-2025/">关于 Project 2025 的 NotebookLM – 以创意的速度前进</a>: 未找到描述</li><li><a href="https://hickstro.org/">数字写作，数字教学</a>: Troy Hicks 博士的主页
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1315806636832718880)** (78 条消息🔥🔥): 

> `LM Studio 更新, Tailscale 配置, 模型兼容性问题, RAG 技术, 性能优化` 


- **更新 LM Studio 至最新版本**：一位用户提到 LM Studio 版本过旧的问题，并感谢另一位用户确认应用内更新可能不会自动升级到 0.3.x 等较新版本。
   - 建议进行手动更新以确保与新模型的兼容性。
- **配合 Tailscale 配置 LM Studio**：一位用户通过使用设备的 MagicDNS 名称而非 IP 地址，成功配置了 LM Studio 以配合 Tailscale 使用。
   - 该方法增强了可访问性并解决了之前的连接问题。
- **模型兼容性挑战**：几位用户讨论了特定模型的兼容性问题，包括 LLAMA-3_8B_Unaligned，并质疑这些模型是否可用。
   - 有人建议该模型可能由于最近的更新或更改而损坏。
- **使用 RAG 进行文档处理**：一位用户询问如何使用生成式 AI 合并文档，并收到了关于利用 RAG 技术来提升性能的建议。
   - 建议提供详细的说话风格描述，而不是直接丢入整个对话内容，以获得更好的效果。
- **性能优化技巧**：讨论围绕 GPU 优化展开，强调共享 VRAM 可能会降低性能，并建议限制 GPU 负载。
   - 用户分享了通过修改 Batch sizes（批次大小）和 Context length（上下文长度）来优化处理时间和资源的经验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/ducktales-ducktales2017-infernal-internship-of-mark-beaks-mustache-disguise-gif-21524651">Ducktales Ducktales2017 GIF - Ducktales Ducktales2017 Infernal Internship Of Mark Beaks - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m/">ogkalu/comic-speech-bubble-detector-yolov8m · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/SicariusSicariiStuff/LLAMA-3_8B_Unaligned_BETA_GGUFs">SicariusSicariiStuff/LLAMA-3_8B_Unaligned_BETA_GGUFs · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hah3wi/im_afraid_to_ask_but_how_do_i_actually_quit_lm/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://lmstudio.ai/.">LM Studio - 发现、下载并运行本地 LLMs</a>：在你的电脑上本地运行 Llama, Mistral, Phi-3。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1315824691231064115)** (9 条消息🔥): 

> `散热解决方案, 水箱与水泵, Alphacool 产品, GPU 散热配置` 


- **史诗级的 GPU 散热**：一位成员称赞了散热配置，称其非常*壮观（epic）*，应该能为正在使用的多个 GPU 提供充足的散热。
   - 用户强调了多个 GPU 及其产生的散热需求，确保系统装备精良。
- **水箱尺寸调整**：一位成员注意到他们的 **D5 泵集成水箱** 无法装入，转而选择了 **Alphacool 5.25" 光驱位水箱**。
   - 他们提供了最终决定的 [Alphacool 水箱链接](https://www.aquatuning.com/en/watercooling/custom/reservoirs/tower-tank/terms-and-conditions-alphacool-repack-dual-bayres-5.25-quot-rev.2?currency=3)。
- **Alphacool 与 D5 泵的兼容性**：另一位用户提到 **Alphacool** 也有可以容纳内置 **D5 泵** 的水箱。
   - 这表明成员们对 Alphacool 提供的各种产品型号非常了解。
- **大机箱的空间挑战**：一位成员反思了他们的大型机箱，里面装满了 **4 个 GPU** 和 **8 个 HDD** 等组件，导致空间受限。
   - 他们提到，尽管机箱很大，但配置已经塞满了，要妥善安置所有组件是一个挑战。
- **水冷产品的创新**：一位成员幽默地评论了水冷产品“永恒的设计”，这些产品创新地声称改进了功能，但外观看起来却大同小异。
   - *20 年后*，新技术与传统美学之间的博弈依然存在于水冷设计中。



**提到的链接**：<a href="https://www.aquatuning.com/en/watercooling/custom/reservoirs/tower-tank/terms-and-conditions-alphacool-repack-dual-bayres-5.25-quot-rev.2?currency=3">未找到标题</a>：未找到描述

  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1316142733068337192)** (1 条消息): 

> `协作新频道` 


- **新协作频道上线**：为成员创建了一个用于项目协作的新频道，标题为 <#1316137596535177246>。
   - 该空间旨在让用户相互配合并参与项目开发。
- **社区参与机会**：该频道鼓励社区成员提出他们的项目想法并共同努力。
   - 邀请成员利用这个空间来构建和分享他们的倡议。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1315770316936712273)** (61 条消息🔥🔥): 

> `Idefics 模型见解、研究协作、长期记忆路径、VLM 模型微调、项目讨论论坛创建` 


- **探索 Idefics 及其创作者**：一位成员询问了与 Hugging Face 相关的 **Idefics** 模型系列背后的团队，强调了对其开发的兴趣。
   - 另一位用户指出 **Idefics** 是 **VLM** 模型系列的一部分，暗示了其重要性。
- **协作请求激增**：来自 Team Veera 的 Maitri 表达了探索与 **Nous Research** 建立合作伙伴关系的兴趣，并询问如何推进。
   - 另一位成员介绍自己是一家开源项目的联合创始人，寻求不带销售目的的协作。
- **长期记忆的新发现**：一位成员分享了一篇值得关注的文章，关于研究人员发现了绕过短期记忆形成 **long-term memory**（长期记忆）的新路径。
   - 随后讨论了这一发现的影响，成员们探讨了操纵记忆创建的潜力。
- **VLM 的微调挑战**：成员们讨论了微调 **Llama Vision** 等 **VLM** 模型的挑战，强调 Hugging Face (hf) 对其支持不足。
   - 建议包括使用 **Unsloth** 以及 **AnyModal** GitHub 项目等资源进行多模态框架调整。
- **创建项目协作论坛**：建立了一个用于项目协作的新频道，允许用户讨论并分享潜在合作伙伴的想法。
   - 成员们对这个论坛形式的频道表示热烈欢迎，一位用户建议将讨论组织成子类别。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/cognition_labs/status/1866535303911182771">来自 Cognition (@cognition_labs) 的推文</a>：Devin 今天正式开放使用！只需标记 Devin 即可修复前端 Bug、为待办任务创建初稿 PR、进行重构等。开始使用 Devin 构建：</li><li><a href="https://medicalxpress.com/news/2024-12-neuroscientists-pathway-term-memories-brain.html">神经科学家发现了大脑形成长期记忆的新路径</a>：来自马克斯·普朗克佛罗里达神经科学研究所的研究人员发现了大脑形成长期记忆的新路径。他们的工作发表在《Nature Neuroscience》上，表明长期...</li><li><a href="https://github.com/ritabratamaiti/AnyModal">GitHub - ritabratamaiti/AnyModal: AnyModal 是一个基于 PyTorch 的灵活多模态语言模型框架</a>：AnyModal 是一个基于 PyTorch 的灵活多模态语言模型框架 - ritabratamaiti/AnyModal
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1315826353031086131)** (17 条消息🔥): 

> `构建 Security Agent，ReAct Agent 示例，RAG 系统中的可观测性，生成 O1 类型合成数据，来自 Meta 的 Thinking LLMs 论文` 


- **构建简单的 Security Agent**：一位用户分享了他们使用 OpenAI API 和一个用于简化函数处理的封装器构建简单 Security Agent 的方法，记录了诸如创建 Tool 类以及利用循环完成任务等步骤。
   - 另一位成员承认，在多 Agent 系统和 ReAct 策略等更高级的架构中，复杂性会随之增加。
- **探索 ReAct Agent 交互**：一位用户询问了关于让 Agent 进行推理并与环境交互的各种策略的文档，同时讨论了将输出作为用户输入的潜力。
   - 推荐了一个使用 ReAct 框架的参考示例，强调了 Assistant 如何在交互中扮演用户的角色。
- **RAG 系统中的可观测性**：一位用户询问如何增强 Ollama 请求的可观测性，以便在 RAG 系统上下文中跟踪进度并分析数据。
   - 他们寻求将标准日志转换为更详细的记录，详述测试的 Prompt 和生成的输出。
- **生成 O1 类型合成数据资源**：一位用户寻求生成 O1 类型合成数据的资源，促使另一位成员推荐了来自 Meta 的 Thinking LLMs 论文，因其技术非常有趣。
   - 他们链接了与 OpenAI-O1 影响相关的实验，暗示了 LLMs 传统上在推理任务中面临的困难。
- **Thinking LLMs 论文的见解**：一位成员讨论了 Thinking LLMs 论文，重点关注其让 LLMs 列出内部想法并在最终确定答案前评估响应的方法论。
   - 他们通过一个示例说明了这一概念，展示了 LLM 在生成答案时“过度思考”的倾向。



**提到的链接**：<a href="https://www.oxen.ai/blog/thinking-llms-general-instruction-following-with-thought-generation">Thinking LLMs: General Instruction Following with Thought Generation | Oxen.ai</a>：OpenAI-O1 的发布激励了许多人深入思考……思考 💭。说话前先思考是某些人比其他人更擅长的一项技能 😉，但也是 LLMs 具备的一项技能...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

deki04: https://x.com/omarsar0/status/1866143542726340890?s=46
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1315904404582563861)** (5 条消息): 

> `Scratchpad 反馈，输出的可视化表示，核心推理任务见解` 


- **关于 Scratchpad 格式的反馈**：有人对新的 scratchpad-think 格式的可读性表示担忧，一位成员承认阅读起来感觉像“一团乱”，但发现它对于复查很有价值。
   - 该成员强调了通过 scratchpad 记录逻辑的重要性，并表示不确定谁会更喜欢这种格式。
- **视觉辅助讨论**：分享了推理任务输出的视觉方面，强调了其对读者的有效性存在不确定性。
   - 该成员指出，虽然视觉效果可能使理解更容易，但 scratchpad 输出固有的凌乱性可能仍会带来困难。
- **关于推理任务的核心见解**：推理任务项目的核心被确定为在没有预定义 System Prompts 的情况下生成的输出，这表明重点在于原始模型输出。
   - 该成员反映，这种原始输出可能“值得”分享，尽管认识到文本的杂乱性，仍表达了进一步参与的愿望。

### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1316125093046390937)** (12 条消息🔥): 

> `会议个人资料公开, Microwave Gang, Discord 个人资料名称` 


- **Nate 的会议个人资料需要公开访问**：一位成员强调 Nate 需要公开他的会议个人资料，以便顺利进行介绍。
   - *Lolol* 是另一位成员的俏皮回应，暗示了这种情况中的幽默感。
- **关于 Microwave Gang 的讨论**：有人对“Microwave gang”进行了轻松的询问，引起了成员们的一些困惑。
   - 一位成员分享了 [Microwave Gang subreddit 的链接](https://www.reddit.com/r/microwavegang/) 以供进一步探索。
- **受家人启发的个人资料名称**：一位成员分享了他们 Discord 个人资料名称的幽默由来，称自己变懒了，所以使用了首字母缩写。
   - 他们还提到让女儿挑选了主个人资料的名称和图片，增添了个人色彩。
- **开放 Hangouts 公告**：Nate 传达了原定于周四和周五下午 1:30-2:30 之间的两次开放 Hangouts 计划。
   - 他表示场地的详细信息将最终确定，并且可能会为部分知情的付费参会者提前开始。
- **对即将到来的会议的期待**：一位成员通过回复“太棒了，稍后见！”表达了对即将到来的会议的兴奋。
   - 这展示了热情并营造了参与者之间的积极氛围。



**提到的链接**：<a href="https://www.reddit.com/r/microwavegang/">Reddit - Dive into anything</a>：未找到描述

  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1315778674603327590)** (12 条消息🔥): 

> `DeepSeek V2.5 发布, 互联网搜索功能, DeepSeek 许可证讨论` 


- **DeepSeek V2.5 发布公告**：新版本的 DeepSeek，**DeepSeek-V2.5-1210**，在兴奋中发布，被称为“大结局”。
   - 成员们热情地讨论了这次发布，并指出他们一直在等待这个更新。
- **实时互联网搜索上线**：DeepSeek 宣布 **Internet Search** 功能现已上线，在其 [聊天平台](https://chat.deepseek.com/) 上提供实时答案。
   - 鼓励用户切换该功能以获取即时搜索结果。
- **关于 DeepSeek 许可证的讨论**：一位成员表达了希望 **DeepSeek** 许可证为 Apache 的愿望，并询问当前许可证是否允许合成数据生成。
   - 另一位成员确认在当前条款下是允许的，尽管这并不常见。
- **关于 OLMo 测试的推测**：在许可证讨论之后，一位成员提到他们将检查 **OLMo** 的详细信息，表明对评估持续关注。
   - 这突显了社区在探索与合成数据生成相关的能力方面的参与度。
- **社区对新功能的反应**：成员们对搜索功能上线的消息反应积极，对其潜力感到兴奋。
   - 一位成员特别说道，*“噢，他们上线了搜索”*，反映了普遍的热情。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/deepseek-ai/DeepSeek-V2.5-1210">deepseek-ai/DeepSeek-V2.5-1210 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/deepseek_ai/status/1866459740324458835">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 DeepSeek-V2.5-1210：大结局 🎉🌐 互联网搜索现已在网页端上线！访问 https://chat.deepseek.com/ 并切换“Internet Search”以获取实时答案。 🕒🧵(1/3)
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1315816171131506731)** (4 条消息): 

> `Sam 确认为 CEO, 用户身份混淆, 鞋类偏好` 


- **Sam 确认为 CEO**：在频道内的各种讨论中，**Sam Altman** 已被确认为 **CEO**。
   - 这一确认引发了关于他领导能力的持续对话。
- **聊天中的身份混淆**：一位成员幽默地询问另一位参与者是否是 **Sam Altman**，并表示如果是真的则需要公开。
   - 提问的俏皮语气反映了聊天轻松愉快的氛围。
- **鞋类品味与简历之争**：一位用户幽默地声称自己比 Sam 有**更好的鞋类品味**，但**简历更差**。
   - 这一评论为聊天中关于身份的对话增添了喜剧色彩。


  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1315788763863056486)** (11 条消息🔥): 

> `vLLM 项目加入 PyTorch，对模型能力的预期，会议体验` 


- **vLLM 现已成为 PyTorch 生态系统的一部分**: [vLLM 项目](https://pytorch.org/blog/vllm-joins-pytorch/) 已正式加入 PyTorch 生态系统，以增强其针对 LLM 的高吞吐量、内存高效的推理能力。
   - 利用创新的 [PagedAttention algorithm](https://arxiv.org/abs/2309.06180)，vLLM 持续更新 pipeline parallelism 和 speculative decoding 等新功能。
- **产品问题，而非科学问题**: 一场讨论强调，与 o1 等模型合作现在是一个产品问题，依赖于正确的数据和 context，而非科学局限。
   - 一句引言总结道：*“鉴于 o1-pro 的聪明程度，挑战已不再是‘模型能否做到’了。”*
- **会议成名带来的挑战**: 一位成员幽默地评论了参加大型会议的挑战，称他们现在“太出名了”，无法清静地参加。
   - 他们分享了人们想要合影的经历，突显了会议互动的独特动态。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/abrakjamson/status/1866247961036095858?s=46&t=_jodDCDeIUnWb_Td0294bw">Abram Jackson (@abrakjamson) 的推文</a>: 每个人都需要理解这一点。如果你能引入所有正确的数据，准确地表述问题，并给予必要的工具，o1 将能完成任何事情。这是一个产品问题，而不是...</li><li><a href="https://pytorch.org/blog/vllm-joins-pytorch/">vLLM 加入 PyTorch 生态系统：为每个人提供简单、快速且廉价的 LLM 服务</a>: 无描述</li><li><a href="https://x.com/mattvidpro/status/1866187800355492095?s=61">MattVidPro AI (@MattVidPro) 的推文</a>: 😬😬😬</li><li><a href="https://github.com/allenai/awesome-open-lms">GitHub - allenai/awesome-open-source-lms: OLMo 之友及其链接。</a>: OLMo 之友及其链接。通过在 GitHub 上创建账号为 allenai/awesome-open-source-lms 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1315776773920784516)** (7 条消息): 

> `xAI 与 Pepes，Fchollet 关于 Scaling Law 的讨论，Twitter 动态` 


- **xAI 拥抱 Pepes**: 一位成员指出 **xAI** 在其沟通中专门使用了 **pepe 表情符号**，展示了其俏皮的品牌形象。
   - 他们分享了 [Grok Image Generation release](https://x.ai/blog/grok-image-generation-release) 的链接以提供更多背景。
- **Fchollet 回应 Scaling**: 在一条推文中，**Fchollet** 解决了关于他在 AI Scaling Law 上立场的误解，强调他从未反对 Scaling，而是批评过度依赖更大的模型。
   - 他坚持认为，重点应该从询问 **LLM 是否能推理** 转向它们是否能 **adapt to novelty**，并提到他为后者提出了一个数学定义。
- **对 Fchollet 的幽默反应**: 成员们觉得 Fchollet 的推文串很有趣，尤其是像 'Smited' 和 'Smote' 这样的回复。
   - 一位成员对 Twitter 上讨论的激烈程度表示有趣，建议转向更直接的回应方式。



**提及的链接**: <a href="https://x.com/fchollet/status/1866348355204595826?s=46">François Chollet (@fchollet) 的推文</a>: 伙计，你在说什么？1. 我根本不知道你是谁，所以我对你没有任何“想法”。2. 我从未反对过 Scaling Law。相反，我一直反对的是那种认为……

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1315770277602267166)** (44 条消息🔥): 

> `WaveForms AI 发布，vLLM 加入 PyTorch，Devin 正式全面开放，Molmo 完整配方发布，2024 年 AI Agents 现状报告`

- **WaveForms AI 发布**：由 @alex_conneau 宣布，[WaveForms AI](http://waveforms.ai) 是一家 Audio LLM 公司，旨在解决语音图灵测试（Speech Turing Test）并将情感智能（Emotional Intelligence）引入 AI。
   - 此次发布凸显了将情感理解整合到 AI 产品中的持续趋势。
- **vLLM 加入 PyTorch 生态系统**：@vllm_project 表示，加入 PyTorch 生态系统可确保无缝的兼容性和性能优化，从而增强 AI 创新。
   - 这一集成预计将提高开发人员在 AI 项目工作中的易用性。
- **Devin 现已全面开放 (GA)**：Cognition 宣布 Devin 的起售价为每月 500 美元，提供无席位限制和各种集成等优势。
   - 该工具旨在协助工程团队高效地进行调试、创建 PR 以及进行代码重构（code refactors）。
- **Molmo 完整方案（Full Recipe）发布**：@allen_ai 分享了他们发布了 Molmo 的完整方案，包括训练代码和更新的技术报告，使其他人更容易复现他们的模型。
   - 此次发布是 AI 社区协作开发的重要一步。
- **2024 年 AI Agents 现状洞察**：@MrAhmadAwais 在分析了 1840 亿个 tokens 和 4000 名构建者的反馈后，发布了《2024 年 AI Agents 现状》报告，展示了 AI agents 领域的趋势。
   - 这些洞察对于理解 AI agent 技术的演变至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/stainless">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://www.cognition.ai/blog/devin-generally-available">Cognition | Devin 现已全面开放 (GA)</a>：我们是一家构建端到端软件 Agent 的应用 AI 实验室。</li><li><a href="https://x.com/vllm_project/status/1866228071818473512">来自 vLLM (@vllm_project) 的推文</a>：开源创新是 vLLM 的基因，我们热爱 PyTorch 生态系统！让我们一起推动 AI 创新的边界，让所有人都能触及💪 引用 PyTorch (@PyTorch) ...</li><li><a href="https://x.com/NousResearch/status/1866584568548995538">来自 Nous Research (@NousResearch) 的推文</a>：发布 Nous Simulators！这是我们所有涉及社交领域人类与 AI 交互实验的家园。 http://sims.nousresearch.com</li><li><a href="https://x.com/alex_conneau/status/1866127388373098607">来自 Alexis Conneau @ NeurIPS (@alex_conneau) 的推文</a>：很高兴宣布创立 WaveForms AI (http://waveforms.ai) —— 一家 Audio LLM 公司，旨在解决语音图灵测试并为 AI 带来情感智能 @WaveFormsAI</li><li><a href="https://x.com/mrahmadawais/status/1866483416981786821?s=46">来自 Ahmad Awais (@MrAhmadAwais) 的推文</a>：介绍…… 2024 年 AI Agents 现状 🤖 在处理了来自 3.6 万名开发者的 1840 亿个 Token 和 7.86 亿次 API 请求后，我们从 4000 名构建者那里收集了宝贵的见解，下一代……</li><li><a href="https://x.com/stainlessapi/status/1866503595690180657?s=46">来自 Stainless (@StainlessAPI) 的推文</a>：很高兴分享我们筹集了 2500 万美元的 A 轮融资，由 @JenniferHLi @a16z 领投，@sequoia、@thegp、@felicis、@zapier 和 @mongoDB Ventures 参投：https://www.stainlessapi.com/blog/stainless-series-a</li><li><a href="https://huggingface.co/docs/text-generation-inference/conceptual/chunking">TGI v3 概览</a>：未找到描述</li><li><a href="https://x.com/allen_ai/status/1866182037704757631?s=46">来自 Ai2 (@allen_ai) 的推文</a>：还记得 Molmo 吗？完整配方终于出炉了！训练代码、数据以及复现我们模型所需的一切。哦，我们还更新了技术报告！链接在推文中 👇</li><li><a href="https://x.com/jsngr/status/1866498187248443495?s=46">来自 Jordan Singer (@jsngr) 的推文</a>：今天 @mainframe 很高兴分享我们 550 万美元的种子轮融资，用于构建新的 AI 界面，由 @lachygroom 和 @stellation 共同领投，@basecasevc、@weekendfund 等参与。</li><li><a href="https://x.com/dougsafreno/status/1866522855510307063?s=46">来自 Doug Safreno (@dougsafreno) 的推文</a>：今天的大新闻：@GentraceAI 筹集了由 @MatrixVC 领投的 800 万美元 A 轮融资。我们通过发布 Experiments 来庆祝，这是第一个用于 LLM 产品开发的协作测试环境。</li><li><a href="https://x.com/amasad/status/1866551672207737067?s=46">来自 Amjad Masad (@amasad) 的推文</a>：Replit Agent —— 今天结束早期访问 —— 是从创意到部署应用的最佳方式。之后，你会希望快速迭代功能和修复。进入 Assistant —— 通过 Prompt 来更改……</li><li><a href="https://x.com/yuchenj_uw/status/1866514943815880847?s=46">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：🚀 很高兴分享我们筹集了 1200 万美元的 A 轮融资！在 Hyperbolic，我们的使命是构建一个开放的 AI 平台。所谓“开放”，我们是指：> 开放 GPU 市场：可以把它想象成 GPU 版的 Airbnb —— 任何人都可以……</li><li><a href="https://magazine.sebastianraschka.com/p/llm-research-papers-the-2024-list">LLM 研究论文：2024 年清单</a>：一份精选的 2024 年有趣的 LLM 相关研究论文清单，分享给那些想在假期找点读物的人。</li><li><a href="https://github.com/allenai/awesome-open-lms">GitHub - allenai/awesome-open-source-lms: OLMo 之友及其链接。</a>：OLMo 之友及其链接。通过在 GitHub 上创建账号为 allenai/awesome-open-source-lms 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=sPiOP_CB54A&t=2s&pp=ygUkVXNpbmcgZ2VtaW5pIGZvciBzY2llbnRpZmljIHJlc2VhcmNo">这正在改变科学家的研究方式 | Gemini</a>：Gemini —— Google 最新且最强大的 AI 模型。20 万篇包含关键科学信息的科学论文被 Gemini 阅读、理解和筛选……</li><li><a href="https://www.youtube.com/live/qZ0ImE41pVs?si=rUe6uWNbdYgXsSiJ">Canvas —— OpenAI 的 12 天：第 4 天</a>：Kevin Weil、Lee Byron 和 Alexi Christakis 介绍并演示了 Canvas 的更新。</li><li><a href="https://x.com/aryanvichare10/status/1866561638712881172">来自 Aryan Vichare (@aryanvichare10) 的推文</a>：介绍 WebDev Arena，一个两个 LLM 竞争构建 Web 应用的竞技场。你可以投票选出表现更好的 LLM，并查看最佳模型的排行榜。100% 免费且开源，基于 @lmarena_ ...
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1315850692782719097)** (1 messages): 

> `Sora Launch, Generative Video WorldSim, DeepMind Genie, VideoPoet, DeCAF Test of Time Winner` 


- **Sora 发布成为焦点**：最新一期播客包含了对 **OpenAI Sora** 的 **7 小时深度探讨**，重点介绍了来自 **@billpeeb** 的见解。
   - 听众可以[在此查看该剧集](https://latent.space/p/icml-2024-video-robots)，获取关于此次发布的全面概述。
- **ICML 2024 的生成式视频创新**：本期播客讨论了 **Generative Video WorldSim**，汇集了不同领域的专家。
   - 值得注意的提及包括 **@jparkerholder** 和 **@ashrewards** 关于 **DeepMind Genie** 的讨论，强调了 AI 领域的创新。
- **探索 VideoPoet 的能力**：主持人与 **@hyperparticle** 深入探讨了 **VideoPoet**，讨论了其具有影响力的功能和应用。
   - 该剧集强调了视频生成在 AI 和机器学习领域的重要性。
- **Flow Matching 与 Stable Diffusion 3 讨论**：播客中 **@rickytqchen** 讨论了 **Flow Matching**，**@pess_r** 提供了关于 **Stable Diffusion 3** 的见解。
   - 这些讨论反映了 Diffusion 模型正在进行的进展，涉及了前沿技术。
- **关注 LLM 与机器人学的融合**：专家如 **@giffmana** 讨论了 **Vision 与 LLM** 的融合，以及 **@chelseabfinn** 在机器人学方面的贡献。
   - 这一环节是 **ICML** 展示的重大创新综合回顾的一部分。



**提到的链接**: <a href="https://x.com/latentspacepod/status/1866291034596258266">来自 Latent.Space @NeurIPSConf Live! (@latentspacepod) 的推文</a>: 🆕 Generative Video WorldSim, Diffusion, Vision, Reinforcement Learning and Robotics 我们有史以来最长的一集！https://latent.space/p/icml-2024-video-robots 深度探讨 - @OpenAI Sora (与 @billpe...

  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1315846299366461512)** (44 messages🔥): 

> `Torch Compile Usage, Reward Models in RL, KTO Model Benefits, Dataset Limitations, Quantitative Research in Fine-tuning` 


- **Torch Compile：速度与显存**：成员们讨论了使用 **torch.compile** 的经验，指出速度提升微乎其微且增加了显存占用。
   - *一位成员评论道，“这可能只是我个人的问题。”*
- **强化学习中的 Reward Models**：关于 RL 中 Reward Models 是否独立的讨论得出结论，在 **online RL** 中，它始终是用于评分的独立模型。
   - *成员们探讨了拥有 Reward Model 的影响，强调它在真实模型训练期间是冻结的。*
- **KTO 相对于原始模型的优势**：Kaltcit 称赞了 **KTO** 模型可能超越原始数据集标准性能的能力，并指出了其在鲁棒性方面的声明。
   - *然而，成员们表示需要确认它是否确实比公认的数据提供更好的结果。*
- **KTO 研究结果的证实**：Kaltcit 提到 **Kalo** 已经证实了 KTO 论文的研究结果，但感叹微调者（finetuners）之间缺乏严肃的定量研究。
   - *Nanobitz 指出，这其中的大部分可能发生在不广泛分享研究结果的组织内部。*
- **对多轮对话和评分调整的需求**：有人询问在 Axolotl 中集成 Reward Model 进行评分需要哪些方面，强调希望尝试超越现有数据集的方法。
   - *Kaltcit 表示，**当前的 KTO 设置**可能足以在原始优势之外最大化回答效果。*


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1315969815013232723)** (25 messages🔥): 

> `测验访问、文章提交指南、Hackathon 报告要求、文章的社交媒体发布、课程结业要求` 


- **访问测验链接**：一位成员询问在哪里可以找到最后两个测验的链接，表示难以找到它们。
   - 另一位成员指出，链接可以在[课程网站的教学大纲部分](https://llmagents-learning.org/f24)找到。
- **文章提交说明**：关于文章提交格式的问题，特别是 Hackathon 团队与 MOOC 证书每位学生所需的文章数量。
   - 澄清了每个 Hackathon 团队只需要一份报告，但每位学生必须提交各自的文章才有资格获得 MOOC 结业证书。
- **拆分文章进行提交**：成员们讨论了将撰写的文章拆分为较短的帖子用于社交媒体的可行性。
   - 确认这是可以接受的，只要在最终提交中包含所有草稿和链接以获得学分即可。
- **课程结业相关疑问**：一位成员对是否满足所有课程要求以及他们的测验提交是否正确链接到个人资料表示不确定。
   - 寻求关于如何验证所有测验完成情况和提交流程的说明。
- **使用 LinkedIn 发布文章**：讨论了使用 LinkedIn 分享与课程作业相关的文章的适当性。
   - 确认了为 LinkedIn 帖子总结内容是允许的，并且符合提交指南。



**提到的链接**：<a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>：MOOC，2024 年秋季

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1316103929393578168)** (3 messages): 

> `LLM 中的 Function Calling，Tool Learning 的重要论文` 


- **理解 Function Calling 机制**：一位成员分享了关于 [function calling](https://platform.openai.com/docs/guides/function-calling) 的文档链接，解释了它如何利用函数描述和签名根据提示词（prompts）设置参数。
   - 有人建议模型可能是通过大量示例训练的，以增强泛化能力。
- **Tool Learning 的关键学术参考文献**：一位成员强调了几篇重要的论文，包括 [arXiv:2305.16504](https://arxiv.org/pdf/2305.16504) 和 GitHub 上的 [ToolBench](https://github.com/OpenBMB/ToolBench)，以帮助理解 LLM 的 Tool Learning。
   - 另一篇 arXiv 论文 [arXiv:2304.08354](https://arxiv.org/abs/2304.08354) 也被指出在讨论中具有潜在的重要意义。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenBMB/ToolBench">GitHub - OpenBMB/ToolBench: [ICLR&#39;24 spotlight] An open platform for training, serving, and evaluating large language model for tool learning.</a>：[ICLR'24 spotlight] 一个用于训练、服务和评估用于 Tool Learning 的大语言模型的开放平台。- OpenBMB/ToolBench</li><li><a href="https://arxiv.org/abs/2304.08354">Tool Learning with Foundation Models</a>：人类拥有非凡的创造和使用工具的能力，使他们能够克服生理局限并探索新的领域。随着 Foundation Models 的出现，AI 系统已经...
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1315774131735887963)** (5 条消息): 

> `LlamaParse Auto Mode, LlamaParse Webinar, Document Agent Workflows, LlamaParse JSON Mode, Invoice Processing Agent` 


- **LlamaParse Auto Mode 优化成本**：LlamaParse 推出了 **Auto Mode**，它以标准且更便宜的模式解析文档，同时根据用户定义的触发器选择性地升级到 **Premium mode**。更多详情请参阅[此处](https://t.co/6uDAt8amFY)。
   - 了解该功能的优势以及它如何优化平衡成本与性能。
- **获取 Auto Mode 的视频演示**：**LlamaParse Auto Mode** 的视频演示可在[此处](https://t.co/qBD8sfDsqb)查看。提醒用户更新浏览器以确保与 YouTube 的兼容性。
   - 支持的浏览器包括 [Google Chrome](https://www.google.com/chrome/index.html)、[Mozilla Firefox](https://www.mozilla.org/firefox/new/) 和 [Opera](https://www.opera.com/)。
- **LlamaParse 的详细 JSON Mode**：LlamaParse 的 **JSON mode** 提供对复杂文档的详细解析，提取图像、文本块、标题和表格。欲了解更多信息，请参考[此链接](https://t.co/eCYUqbCMGI)。
   - 该功能增强了在处理结构化数据提取时的控制力和能力。
- **即将举行的 Webinar 提醒**：分享了定于本周四举行的一场重要 **webinar** 的提醒。更多详情请访问[此处](https://t.co/YnnSKs3gOP)。
   - 鼓励参与者不要错过这个宝贵的学习机会。
- **探索 Document Agent 工作流**：团队正在探索创新的 **document agent workflows**，这些工作流超越了传统任务，旨在实现复杂流程的自动化。一个**端到端发票处理 Agent** 是其中的项目之一，旨在从发票中提取相关信息并与供应商进行匹配。
   - 请在[此处](https://t.co/dr2yiyf3zE)关注这一极具前景的工作流自动化工具的进展。



**提到的链接**：<a href="https://t.co/qBD8sfDsqb">未找到标题</a>：未找到描述

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1316080880674013194)** (21 条消息🔥): 

> `运行本地 Agent 示例，文档检索问题，ColPali Reranking 功能，Cohere Rerank Postprocessor，使用更小的模型` 


- **解决本地 Agent 示例故障**：一位成员在 M1 Mac 上运行 [本地 Agent 示例](https://github.com/run-llama/python-agents-tutorial/blob/main/2_local_agent.py) 时遇到困难，仅收到 'Process finished with exit code 0'。在其他人的建议下，他们在创建干净的项目设置后成功运行。
   - *当文档未被摄取（Ingested）时会出现空响应*；另一位用户建议检查数据目录是否包含实际的文本。
- **ColPali Rerank 功能说明**：一位新用户询问 [ColPali 功能](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent) 是否在 PDF 处理过程中动态运行。官方澄清它是一个 Reranking 工具，而不是一个独立的进程。
   - 另一位用户确认，添加 ColPali 主要用于检索后对图像节点进行重排序（Reranking）。
- **Bedrock 中 Cohere Rerank 3.5 的可用性**：一位成员提出了关于通过 Bedrock 使用 *Cohere Rerank 3.5* 作为 Postprocessor 的可用性问题。有人指出最近的更新已经集成了此功能，并提供了 [相关文档](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/postprocessor/llama-index-postprocessor-bedrock-rerank) 的链接。
   - 他们还提供了安装命令：`pip install llama-index-postprocessor-bedrock-rerank`。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/embeddings/ollama_embedding/">Ollama Embeddings - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">入门教程（本地模型）- LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/postprocessor/llama-index-postprocessor-bedrock-rerank">llama_index/llama-index-integrations/postprocessor/llama-index-postprocessor-bedrock-rerank at main · run-llama/llama_index</a>：LlamaIndex 是用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/#define-function-tools">ReAct Agent - 计算器工具简单介绍 - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/python-agents-tutorial/blob/main/2_local_agent.py">python-agents-tutorial/2_local_agent.py at main · run-llama/python-agents-tutorial</a>：来自 Python Agent 教程的代码示例。通过在 GitHub 上创建账号为 run-llama/python-agents-tutorial 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1315997050382454855)** (10 条消息🔥): 

> `Cohere 的业务背景，讨论中无关的幽默` 


- **商务对话变得滑稽**：成员们对与 **Cohere 利润驱动背景**无关的笑话表示不满，重申幽默不应掩盖业务讨论。
   - 强调需要专注于更严肃的话题。
- **探讨幽默的适度性**：对幽默的反应各不相同，一些人断言某些讨论过于偏离相关的业务主题。
   - 对话暗示了管理员在遏制非业务相关幽默方面的作用，展示了轻松氛围与专业主义之间的张力。


  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1316093176989220875)** (9 messages🔥): 

> `Rerank 3.5 English 模型计划, CmdR+Play Bot 状态, Aya-expanse 性能, API 请求 403 错误` 


- **关于 Rerank 3.5 English 模型的查询**：一位成员询问是否有关于 **Rerank 3.5 English 模型** 的后续计划。
   - 该查询未收到任何回复，凸显了沟通中可能存在的脱节。
- **CmdR+Play Bot 正在休假**：一位成员询问了 **CmdR+Play Bot** 的状态，该机器人目前正在休假。
   - 另一位用户确认了休假消息，并建议其他人关注后续更新。
- **Aya-expanse 的指令处理**：一位用户想知道基于 command 系列构建的 **aya-expanse** 在处理指令方面的性能是否有所提升。
   - 对话并未就其性能能力给出明确答案。
- **API 请求出现 403 错误**：一位成员表示在尝试构建 API 请求时收到了 **403 响应**。
   - 未提供关于该错误的更多细节，表明需要故障排除协助。



**提及的链接**：<a href="https://cohere.com/careers">Careers</a>：我们的 ML/AI 专家团队热衷于帮助开发者解决现实世界的问题。在多伦多、伦敦和帕罗奥图的办公室，我们工作在机器学习的最前沿，以释放...

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1316224360947781663)** (2 messages): 

> `API 请求错误, Trial key 限制` 


- **理解 API 请求 403 错误**：一位成员在构建 API 请求时遇到了 **403 错误**，表明其请求被禁止。
   - 这种错误通常由于权限问题或使用了错误的 API key 引起。
- **Trial Key 限制**：有成员提到 **403 错误** 与 API 的 **trial key** 使用有关。
   - Trial key 通常带有限制，可能会限制对某些功能或端点 (endpoints) 的访问。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1315938278750294046)** (17 messages🔥): 

> `合并 Config 文件, TorchTune PR 讨论, DoraLinear 和 LoraLinear 初始化, Tensor 设备处理, Magnitude 计算` 


- **合并冲突的 Config 文件问题**：一位用户寻求一种简单的方法来合并冲突的配置文件，希望对所有文件使用“接受双方更改 (accept both changes)”。
   - 他们幽默地透露了自己的权宜之计：用空字符串替换冲突标记。
- **关于 TorchTune PR #2139 的见解**：讨论集中在 [PR #2139](https://github.com/pytorch/torchtune/pull/2139)，涉及对 `torch.utils.swap_tensors` 及其在初始化中作用的担忧。
   - 贡献者们一致认为，需要进一步讨论在哪里定义 `self.magnitude` 及其初始化。
- **探索 to_empty 初始化方法**：提出了改进 `to_empty` 方法的建议，以便在管理设备和参数捕获的同时保持预期的用户体验。
   - 社区成员商讨如何在不破坏现有代码的情况下平衡最佳实践。
- **Tensor 操作中的设备处理问题**：强调了在处理 Tensor 初始化和交换时，设备管理至关重要，特别是对于像 `magnitude` 这样的参数。
   - 成员们认识到，正确使用 `swap_tensors` 等 API 对于在操作期间保持设备完整性至关重要。
- **关于参数初始化和梯度的澄清**：贡献者澄清说，如果设备管理得当，使用 `copy_` 是可以接受的，同时强调了 `requires_grad` 状态的重要性。
   - 他们讨论了在初始化例程中集成错误检查，以防止处理 meta 设备上的 Tensor 等常见陷阱。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L939).">pytorch/torch/nn/modules/module.py at main · pytorch/pytorch</a>：Python 中的 Tensors 和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch</li><li><a href="https://github.com/ebsmothers/ebs-torchtune/blob/5da01406658f9079ebb5bcd6eab0e4261d4188f9/torchtune/modules/peft/dora.py#L123-L126">ebs-torchtune/torchtune/modules/peft/dora.py at 5da01406658f9079ebb5bcd6eab0e4261d4188f9 · ebsmothers/ebs-torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建一个账户来为 ebsmothers/ebs-torchtune 的开发做贡献。</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L963).">pytorch/torch/nn/modules/module.py at main · pytorch/pytorch</a>：Python 中的 Tensors 和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1315999747445887016)** (1 条消息): 

> `LangWatch Optimization Studio, DSPy programs, Low-code tools, Open source release` 


- **LangWatch Optimization Studio 发布**：推出了 **LangWatch Optimization Studio**，这是一个用于可视化构建 **DSPy** 程序的新型低代码 UI，简化了评估 **LMs** 和运行优化的过程。
   - 该工具的代码很快可以导出为支持 DSPy 的程序，目前已在 [GitHub 上开源](https://github.com/langwatch/langwatch)。
- **结束私测阶段**：**LangWatch Optimization Studio** 已成功结束私测阶段，标志着该平台迈出了重要一步。
   - 鼓励用户前往体验，并在 [GitHub 页面](https://github.com/langwatch/langwatch)上点亮 ⭐️ 以示支持。
- **LM 的可视化开发**：该工作室旨在为 **DSPy** 程序提供一个直观的、专门设计的可视化开发环境。
   - 用户可以轻松地对其模型运行优化，从而提高生产力并简化工作流程。



**提到的链接**：<a href="https://github.com/langwatch/langwatch">GitHub - langwatch/langwatch: Source available LLM Ops platform and LLM Optimization Studio powered by DSPy.</a>：由 DSPy 驱动的源码可用 LLM Ops 平台和 LLM Optimization Studio。

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1315843887843119104)** (13 条消息🔥): 

> `DSPy documentation access, API reference location, O1 series model impact, Error during optimization` 


- **访问 DSPy 文档遇到困难**：一位成员对无法找到 [DSPy 文档](https://dspy.ai) 表示沮丧，特别是之前位于顶部的 API 参考链接。
   - 另一位成员澄清说，现在大多数语法都可以在落地页上找到，并指出不再需要专门的类型模块。
- **关于 O1 系列模型的见解需求**：一位成员询问 O1 系列模型将如何影响 **DSPy** 工作流，特别是关于来自 MIPRO 的优化模块参数。
   - 他们怀疑可能需要进行调整，例如减少优化周期，但欢迎其他人的任何见解或建议。
- **报告优化错误**：一位成员报告在优化时遇到了奇怪的通用错误和 bug，并提到他们在特定频道发布了更多详情。
   - 他们请求其他人在有空时关注并解决他们遇到的问题。



**提到的链接**：<a href="https://dspy.ai">DSPy Documentation</a>：用于编程（而非提示）语言模型的框架。

  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1315796605286486067)** (2 条消息): 

> `Awareness of AI capabilities, Grassroots Science Initiative, Multilingual LLMs, Risks of AI-generated content` 


- **传播 AI 威胁意识**：一位成员强调了教育非技术人员了解 **AI 生成内容** 危险性的重要性，因为随着技术的进步，诈骗变得越来越可信。
   - 他们建议分享 [MKBHD 的最新视频](https://www.youtube.com/watch?v=OY2x0TyKzIQ)，向亲友展示这些 AI 能力。
- **启动草根科学倡议 (Grassroots Science Initiative)**：多个组织合作启动了 **Grassroots Science**，这是一项开源倡议，旨在到 2025 年 2 月开发出**多语言 LLMs**。
   - 他们的目标是通过众包收集数据、基准测试模型和开源工具，让草根社区参与到多语言研究中。
- **加入草根科学社区**：鼓励感兴趣的各方填写意向表以参与 **Grassroots Science** 倡议，强调草根社区之间的协作。
   - 参与者被要求注明其在写作和阅读方面的语言熟练程度，以便有效地做出贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://grassroots.science/">Grassroots Science</a>：一个专注于通过草根努力开发最先进多语言语言模型的全球倡议。</li><li><a href="https://forms.gle/i8mG999yRbznK8JE9">Grassroots Science 意向表</a>：Grassroots Science 是一个为期一年的全球协作项目，旨在通过众包收集多语言数据，由相信集体力量的草根社区发起...</li><li><a href="https://x.com/GrassrootsSci">来自 undefined 的推文</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1315791507139072011)** (9 条消息🔥): 

> `在 12GB 显存上训练 7B 模型，超高效小模型，模型规模 vs 效率` 


- **在 12GB 上训练 7B 模型似乎野心勃勃**：一位成员评论道，仅用 **12GB** 数据（或显存）训练一个 **7B 参数模型**非常令人惊讶，其可行性令人侧目。
   - 这引发了人们对该模型在实际应用中表现如何的兴趣。
- **对小模型感到兴奋**：人们对**超高效小模型**表现出明显的兴奋，并对其性能和优势的验证进行了强调。
   - 一位粉丝表示：“我喜欢超高效的小模型！它们太棒了！”，强调了那些避开大规模需求的模型的潜力。
- **对以规模为中心的方案持怀疑态度**：一位用户对“规模即一切”（scale-is-all-you-need）的哲学表示怀疑，评论了效率优于尺寸的重要性。
   - 共识倾向于这样一种观点：**10 亿参数（1B）应该足以实现有效的模型性能**。
- **小模型的泛化能力受到质疑**：一位成员思考在大型语言模型（LLM）中观察到的趋势是否也适用于 **10 亿参数以下（sub-billion）的模型**。
   - 这引发了关于超越传统大模型范式的适应性和性能可扩展性的深入讨论。


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1315870638749122633)** (10 条消息🔥): 

> `01 语音应用，控制 GPT o1 Pro，Mac 用户的 Beta 访问权限，网站问题` 


- **01 语音应用发布**：一位成员宣布 **01** 是 Open Interpreter 的语音驱动衍生产品，提供 [CLI](https://github.com/OpenInterpreter/open-interpreter/) 和桌面应用程序。
   - 它包含了模拟 **01 Light Hardware** 以及运行服务端和客户端的指令。
- **与 OI 集成的潜力**：一位成员假设，使用 **OS 模式下的 OI** 可以通过桌面应用或浏览器控制 **GPT o1 Pro**，从而可能实现网页搜索和文件上传功能。
   - 他们表示有兴趣探索这一想法，并指出这可能带来的强大影响。
- **01 应用的 Beta 访问权限**：在讨论中，有人澄清 **01 应用** 仍处于 Beta 阶段，需要邀请才能访问，目前仅对 Mac 用户开放。
   - 一位成员报告说，他们向一名用户发送了私信以获取访问权限，显示出极高的需求。
- **对网站功能的担忧**：一位成员对 Open Interpreter 网站的问题表示沮丧，并展示了截图，但未详细说明具体问题。
   - 社区成员已开始讨论网站导航和功能，作为他们使用 Open Interpreter 体验的一部分。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.openinterpreter.com/">Open Interpreter</a>：未找到描述</li><li><a href="https://01.openinterpreter.com/client/desktop#running-both-server-and-client)">Desktop - 01</a>：未找到描述</li><li><a href="https://changes.openinterpreter.com/log/01-app)">Open Interpreter Changelog</a>：开源项目 Open Interpreter 的官方变更日志。</li><li><a href="https://01.openinterpreter.com/server/introduction)">无标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1316108421480583228)** (1 条消息): 

> `Web Applets, Theia-ide, Programming Interviews, Integration with IDEs` 


- **深入探讨 Web Applets 启动会**：即将举行的一场会议将聚焦于 [Web Applets](https://discord.com/events/1089876418936180786/1311466292753989672)，由一位知名成员主持，计划很快开始。
   - 本次活动旨在增强对现代开发中 Web Applets 集成和功能的理解。
- **探索 Theia-ide：IDE 的新方法**：明天，参与者可以 [探索 Theia-ide](https://discord.com/events/1089876418936180786/1311841242262540298)，它强调开发环境中的**开放性**、**透明度**和**灵活性**。
   - 讨论将由一位专家主持，旨在展示与传统 IDE 相比使用 Theia-ide 的优势。
- **编程面试不断变化的面貌**：一条评论强调了编程面试是如何演变的，指出在过去，应聘者可能会被要求在白板上编写 **bubble sort**。
   - 现在，应聘者只需告诉他们的 IDE 构建一个即可，这强调了向实时编码中更实际技能的转变。
- **为 Theia-ide 见解做准备**：分享了一个与 **Jonas** 访谈的链接，提供了关于 Theia-ide 背后愿景的见解，可在此处访问 [here](https://fxtwitter.com/Scobleizer/status/1864806006150304134)。
   - 本次访谈旨在提供对指导 Theia 开发的功能和理念的更深层理解。



**提到的链接**：<a href="https://fxtwitter.com/Scobleizer/status/1864806006150304134)">来自 Robert Scoble (@Scobleizer) 的推文</a>：过去如果你在 Microsoft 面试编程职位，他们可能会让你在白板上写一个 bubble sort，以确保你知道如何编程。现在？只需告诉你的 IDE ...

  

---


---


---


---


---


{% else %}


> 为了邮件展示，完整的逐频道详情已被截断。
> 
> 如果您想查看完整详情，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}