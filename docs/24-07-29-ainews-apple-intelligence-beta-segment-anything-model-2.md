---
companies:
- meta-ai-fair
- apple
date: '2024-07-30T02:45:55.827150Z'
description: '**Meta** 发布了 **Segment Anything Model** 的续作，进一步推进了其开源 AI 技术。该模型通过引入记忆注意力机制（memory
  attention），在仅需极少数据和算力的情况下，增强了视频应用中的图像分割能力。


  **Apple Intelligence** 将其正式发布时间推迟到了 10 月的 iOS 18.1，但目前已在 **MacOS Sequoia**、**iOS
  18** 和 **iPadOS 18** 上推出了开发者预览版。随之发布的一份长达 47 页的详细论文透露，该模型在 **6.3 万亿（6.3T）token**
  上进行了广泛的预训练，且使用的是**云端 TPU** 而非 Apple Silicon。论文强调，通过后期训练和合成数据，模型在指令遵循、推理和写作能力方面得到了提升。基准测试显示，苹果模型的得分低于
  **Llama 3**，但在人类评估中表现可靠。


  此外，**Meta** 发布了拥有 405B 参数规模的 **Llama 3.1**，标志着一个重磅开源前沿模型的问世。'
id: 010a9b07-042e-4425-9fef-2806762d8754
models:
- llama-3-405b
- llama-3
- segment-anything-model
original_slug: ainews-apple-intelligence
people:
- bindureddy
- maximelabonne
- reach_vb
title: Apple Intelligence 测试版 + Segment Anything Model 2 (分割一切模型 2)
topics:
- image-segmentation
- memory-attention
- video-processing
- pretraining
- cloud-tpus
- post-training
- synthetic-data
- instruction-following
- reasoning
- writing
- benchmarking
---

<!-- buttondown-editor-mode: plaintext -->**2024 年第二大 LLM 部署：既姗姗来迟，又近在眼前。**

> 2024年7月26日至7月29日的 AI 新闻。我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 服务器（**325** 个频道，**6654** 条消息）。预计节省阅读时间（以 200wpm 计算）：**716 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

Meta 继续其开源 AI 势头，推出了[去年 Segment Anything Model](https://www.latent.space/p/segment-anything-roboflow) 的有力续作。最值得注意的是，除了作为比 [SAM1 更好的图像模型](https://x.com/josephofiowa/status/1818087114208342527) 之外，它现在还使用 [Memory Attention 来扩展图像分割以应用于视频，且使用的训练数据和计算资源极少](https://x.com/swyx/status/1818085925714620449)。

但计算机视觉的新闻被 **Apple Intelligence** 掩盖了。Apple Intelligence 不仅将[正式发布推迟到了 10 月的 iOS 18.1](https://archive.ph/Wm85S)，而且今天还在 [MacOS Sequoia（需下载 5GB）](https://x.com/zaidmukaddam/status/1818000899098231091)、[iOS 18](https://x.com/sparkhi/status/1818006585324761098) 和 [iPadOS 18](https://x.com/TyleTweets/status/1818061615545114704) 上发布了开发者预览版（附带一个非常短的 [Waitlist](https://x.com/BrandonButch/status/1817974792282325374)，不包括 [Siri 2.0](https://x.com/VadimYuryev/status/1818036274223284557)，也不包括欧洲用户），同时还发布了一份令人惊喜的 [47 页论文](https://machinelearning.apple.com/papers/apple_intelligence_foundation_language_models.pdf)，比他们 6 月份的主讲演讲（[我们的报道在此](https://buttondown.email/ainews/archive/ainews-talaria-apples-new-mlops-superweapon-4066/)）提供了更多细节。

随之而来的是广泛的演示：

通知筛选

 
![image.png](https://assets.buttondown.email/images/0427becf-2742-4f17-aaab-97d34c6176d0.png?w=960&fit=max)
 

在任意 App 中进行低功耗重写
 
![image.png](https://assets.buttondown.email/images/82c4c542-7322-4e5f-93e2-831c94ef966d.png?w=960&fit=max)
 

写作工具

 
![image.png](https://assets.buttondown.email/images/fabff175-36c2-4e16-a122-9478388ff56f.png?w=960&fit=max)
 

以及更多。

至于论文，来自 [Bindu](https://x.com/bindureddy/status/1818068830570299521)、[Maxime](https://x.com/maximelabonne/status/1818017461738102823) 和 [VB](https://x.com/reach_vb/status/1818014366555586611) 的最佳回顾推文可能已经涵盖了所有内容。我们关注的重点是第 6 页和第 7 页中包含的预训练细节：

 
![image.png](https://assets.buttondown.email/images/721a24de-ad6a-4dbb-b4c7-cf48112218bd.png?w=960&fit=max)
 

- **数据**：新鲜的数据集抓取，包括 Applebot 网络爬虫、授权数据集、代码、3+14B Math Tokens、“公开数据集”，最终形成 6.3T Tokens 用于 CORE 预训练，1T Tokens（包含更高比例的代码/数学混合）用于 CONTINUED 预训练，以及 100B Tokens 用于上下文长度扩展（至 32k）。
- **硬件**：AFM 是使用 v4 和 v5p Cloud TPU 训练的，而不是 Apple Silicon！！AFM-server：8192 个 TPUv4，AFM-on-device：2048 个 TPUv5p。
- **后期训练**：“虽然 Apple Intelligence 的功能是通过在基础模型之上的 Adapter 实现的，但从经验上看，我们发现**改进通用目的的后期训练可以提升所有功能的性能**，因为模型在指令遵循、推理和写作方面具有更强的能力。”
- [广泛使用合成数据用于数学、工具使用、代码、上下文长度、摘要（端侧）、自动红队测试和委员会蒸馏](https://x.com/swyx/status/1818109227090825261)。

同样值得注意的是，他们披露了行业标准基准测试，我们[冒昧地将其提取出来并与 Llama 3 进行了对比](https://x.com/swyx/status/1818114253523656946)：

 
![image.png](https://assets.buttondown.email/images/d0c3d901-1d18-4d8c-b0d5-c92e06866bc3.png?w=960&fit=max)
 

是的，它们明显、显著、大幅低于 Llama 3，但我们不会对此过于担心，因为我们信任 Apple 的 Human Evaluations（人工评估）。

 
![image.png](https://assets.buttondown.email/images/f44fe5f9-5c2a-4eb1-835e-1fb6fb92c3c8.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/3ac6a2e7-6435-4f34-97c6-5084a17a14ee.png?w=960&fit=max)
 


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 简报

> 所有简报均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型进展与行业动态**

- **Llama 3.1 发布**：Meta 发布了 Llama 3.1，其中包括一个 405B 参数模型，这是首个与顶级封闭模型并驾齐驱的开源前沿模型。[@adcock_brett](https://twitter.com/adcock_brett/status/1817591648764801399) 指出它是“开源且免费提供权重和代码的，其许可证支持微调、蒸馏到其他模型以及部署。”该模型支持八种语言，并将上下文窗口扩展到了 **128K tokens**。

- **Mistral AI 的 Large 2**：[@adcock_brett](https://twitter.com/adcock_brett/status/1817591702665695549) 报道称 Mistral 发布了 Large 2，这是其旗舰级 AI 模型，得分接近 Llama 3.1 405b，在编程基准测试中甚至有所超越，而其 123b 的参数规模要小得多。这标志着“一周内发布了两个 GPT-4 级别的开源模型”。

- **OpenAI 动态**：OpenAI 宣布了 SearchGPT，这是一个将 AI 模型与网络信息相结合的 AI 搜索引擎原型。[@adcock_brett](https://twitter.com/adcock_brett/status/1817591625918423453) 提到它“将搜索结果整理成带有来源链接的摘要，最初将面向 10,000 名测试用户开放”。此外，[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1817563907113763280) 分享了关于 OpenAI 对呼叫中心潜在影响的见解，认为 AI Agent 可能会在两年内取代人工操作员。

- **Google DeepMind 的成就**：[@adcock_brett](https://twitter.com/adcock_brett/status/1817591671703445839) 强调“Google DeepMind 的 AlphaProof 和 AlphaGeometry 2 在 AI 数学推理能力方面取得了重大里程碑”，在今年的 IMO（国际数学奥林匹克竞赛）中获得了相当于银牌的分数。

**AI 研究与技术进步**

- **GPTZip**：[@jxmnop](https://twitter.com/jxmnop/status/1817660589797486860) 介绍了 gptzip，这是一个使用语言模型压缩字符串的项目，通过使用 Hugging Face transformers 实现了“比 gzip 高 5 倍的压缩率”。

- **RAG 进展**：[@LangChainAI](https://twitter.com/LangChainAI/status/1817623555946709323) 分享了 RAG Me Up，这是一个在自定义数据集上轻松进行 RAG 的通用框架。它包含一个轻量级服务器和用于通信的 UI。

- **模型训练见解**：[@abacaj](https://twitter.com/abacaj/status/1817694944774987951) 讨论了微调期间低学习率的重要性，认为由于退火阶段的存在，权重已经“稳定”在近乎最优的点上。

- **硬件利用率**：[@tri_dao](https://twitter.com/tri_dao/status/1817711223879971179) 澄清道，“nvidia-smi 显示 'GPU-Util 100%' 并不意味着你使用了 100% 的 GPU 算力”，这对于优化资源使用的 AI 工程师来说是一个重要的区别。

**行业趋势与讨论**

- **AI 在商业中的应用**：关于 LLM 在构建业务方面的能力存在持续争论。[@svpino](https://twitter.com/svpino/status/1817546892395393325) 对非技术创始人仅依靠 LLM 构建整个 SaaS 业务表示怀疑，强调了有能力的真人监督的必要性。

- **AI 伦理与社会影响**：[@fchollet](https://twitter.com/fchollet/status/1817651235341861256) 表达了对取消文化及其对艺术和喜剧潜在影响的担忧，而 [@bindureddy](https://twitter.com/bindureddy/status/1817635971388862531) 分享了对 LLM 推理能力的见解，指出它们在现实世界的推理问题上表现优于人类。

- **开源贡献**：开源社区继续推动创新，例如 [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1817706514792013895) 分享了一个本地语音聊天机器人，该机器人由 Ollama、Hugging Face Transformers 和 Coqui TTS Toolkit 驱动，并使用了本地运行的 Llama。

**梗与幽默**

- [@willdepue](https://twitter.com/willdepue/status/1817640842598727697) 开玩笑说 OpenAI 在过去一个月收到了“19 万亿美元的小费”。

- [@vikhyatk](https://twitter.com/vikhyatk/status/1817618257266040983) 分享了一个幽默的轶事：因为把 S3 当作暂存环境使用，结果收到了一张 5000 美元的网络传输账单。

---

# AI Reddit 简报

## /r/LocalLlama 回顾

**主题 1. 超小型 LLMs：Lite-Oute-1 300M 和 65M 模型**

- **Lite-Oute-1：新型 300M 和 65M 参数模型，提供 instruct 和 base 版本。** ([Score: 59, Comments: 12](https://reddit.com//r/LocalLLaMA/comments/1ee5lzo/liteoute1_new_300m_and_65m_parameter_models/))：**Lite-Oute-1** 在 Hugging Face 上发布了全新的 **300M** 和 **65M** 参数模型，均包含 instruct 和 base 版本。**300M** 模型基于 **Mistral** 架构，拥有 **4096** 上下文长度，旨在通过处理 **300 亿** tokens 来改进之前的 150M 版本；而 **65M** 模型基于 **LLaMA** 架构，上下文长度为 **2048**，是一个处理了 **80 亿** tokens 的实验性超紧凑版本。两款模型均在单块 **NVIDIA RTX 4090** 上训练完成。
    - /u/hapliniste: "虽然我很想要纳米级模型以便在特定任务上轻松进行 finetune，但基准测试是不是已经到了随机水平？MMLU 上的 25% 准确率不就等同于随机选择吗？我想知道它在自动补全或类似场景中是否仍有价值。"

**主题 2. AI 硬件投资挑战：A100 GPU 收藏**

- **[A100 收藏及其原因] (https://www.reddit.com/gallery/1eecrnp)** ([Score: 51, Comments: 20](https://reddit.com//r/LocalLLaMA/comments/1eecrnp/the_a100_collection_and_the_why/))：该帖子描述了一项涉及 **23 块 NVIDIA A100 GPU** 的个人投资，其中包括 **15 块 80GB PCIe 水冷版**、**5 块 40GB SXM4 被动散热版**，以及另外 **8 块 80GB PCIe 水冷版**（未在图中展示）。作者对这一决定表示后悔，理由是水冷版难以转手，且耗尽了全部积蓄，并以此告诫他人要警惕让爱好凌驾于常识之上。

**主题 4. 新型 Magnum 32B：中端 GPU 优化型 LLM**

- **“中端即是胜利” —— Magnum 32B** ([Score: 147, Comments: 26](https://reddit.com//r/LocalLLaMA/comments/1eenk1e/the_mid_range_is_the_win_range_magnum_32b/))：Anthracite 发布了 **Magnum 32B v1**，这是一个针对显存为 **16-24GB** 的**中端 GPU** 设计的 **Qwen finetune** 模型。发布内容包括 **BF16** 格式的全权重，以及 **GGUF** 和 **EXL2** 版本，均可在 [Hugging Face](https://huggingface.co/anthracite-org/magnum-32b-v1) 上获取。
    - 用户讨论了创建一个**角色扮演基准测试**，建议采用社区驱动的 "hot or not" 界面，以评估模型在写作风格、审查程度和角色一致性方面的表现。
    - Magnum 发布版本中的头像特色是信息论之父 **Claude Shannon** 和《东方 Project》中的**蓬莱山辉夜（Tsukasa）**。用户对这种历史人物与虚构角色的独特组合表示赞赏。
    - 一位用户分享了由 Magnum 32B 生成的 **500 token 故事**，讲述了埃隆·马斯克工厂里的两个赛博格揭露公司阴谋的故事。该故事展示了模型的创意写作能力。

## 全 AI Reddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 行业讨论**

- **对误导性 AI 文章的批评**：在 /r/singularity 中，一则[帖子质疑该子版块对 AI 明显的偏见](https://www.reddit.com/r/singularity/comments/1ee7q64/i_dont_get_why_the_sub_despises_ai_so_much_that/)，特别提到了一篇关于 OpenAI 财务状况的潜在误导性文章。帖子标题暗示社区可能对 AI 的进步过于苛刻。

  关键点：
  - 该帖子链接到一篇文章，声称 OpenAI 可能在 12 个月内面临破产，预计亏损 50 亿美元。
  - 该帖子获得了 104.5 的较高评分，表明社区参与度很高。
  - 评论数达 192 条，围绕这一话题似乎展开了实质性的讨论，尽管未提供这些评论的具体内容。

---

# AI Discord 摘要

> 摘要之摘要的总结

**1. LLM 模型发布与性能**

- **Llama 3 撼动排行榜**：来自 Meta 的 **[Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)** 迅速攀升至 **ChatbotArena** 等排行榜的前列，在超过 50,000 场对决中表现优于 **GPT-4-Turbo** 和 **Claude 3 Opus**。
   - **Llama 405B Instruct** 模型在 MMLU 评估中跨多个学科实现了 **0.861** 的平均准确率，在生物学和地理学方面表现尤为出色。评估在大约 **两小时** 内完成，展示了高效的处理能力。
- **DeepSeek V2 挑战 GPT-4**：拥有 **236B 参数** 的 **[DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)** 展现了令人印象深刻的性能，在 **AlignBench** 和 **MT-Bench** 等基准测试的某些领域超越了 **GPT-4**。
   - 该模型在各项基准测试中的强劲表现引发了关于其与领先闭源模型竞争潜力的讨论，凸显了开源 AI 开发的快速进步。
  

**2. AI 开发工具与框架**

- **LlamaIndex 与 Andrew Ng 推出新课程**：**[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)** 宣布与 Andrew Ng 的 DeepLearning.ai 合作推出关于构建 Agentic RAG 系统的新课程，旨在提升开发者创建高级 AI 应用的技能。
   - 此次合作凸显了检索增强生成 (RAG) 在 AI 开发中日益增长的重要性，并展示了 LlamaIndex 致力于向社区普及前沿技术的决心。
- **Axolotl 扩展数据集格式支持**：**[Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)** 扩展了对多种数据集格式的支持，增强了其在 LLM 指令微调 (Instruction Tuning) 和预训练方面的能力。
   - 此次更新允许开发者更轻松地将各种数据源集成到模型训练流水线中，从而可能提高训练模型的质量和多样性。
  

**3. AI 基础设施与优化**

- **vAttention 彻底改变 KV Caching**：**[vAttention](https://arxiv.org/abs/2405.04437)** 系统动态管理 KV-cache 内存，以实现高效的 LLM 推理，且不依赖于 PagedAttention，为 AI 模型中的内存管理提供了一种新方法。
   - 这一创新解决了 LLM 推理中的关键瓶颈之一，可能使大语言模型在生产环境中的部署更加快速和高效。
  

**4. 多模态 AI 进展**

- **Meta 发布 Segment Anything Model 2**：Meta 发布了 **[Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/segment-anything-2)**，这是一个用于图像和视频中实时、可提示对象分割的统一模型，采用 Apache 2.0 许可证。
   - SAM 2 代表了多模态 AI 的重大飞跃，它在约 **51,000 个视频** 的新数据集上进行了训练。此次发布包括模型推理代码、权重文件 (Checkpoints) 和示例 Notebooks，以帮助用户有效地实现该模型。

**5. LLM 进展**

- **Llama 405B Instruct 在 MMLU 中表现亮眼**：**Llama 405B Instruct** 在 MMLU 评估中获得了 **0.861** 的平均准确率，在生物学和地理学等学科表现优异，并在约 **两小时** 内完成了评估。
   - 这一表现引发了关于评估过程稳健性和模型效率的讨论。
- **Llama 3.1 的量化问题**：成员们对 **Llama 3.1** 因量化 (Quantization) 导致的性能下降表示担忧，并指出使用 **bf16** 效果更好 ([X.com 帖子](https://x.com/_xjdr/status/1816892492580814856))。
   - 讨论表明，量化影响可能与总数据量有关，而不仅仅是参数数量。


---

# 第一部分：高层级 Discord 摘要

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **合成数据生成工具受到关注**：成员们讨论了用于生成合成数据的工具，重点介绍了 **Argila** 和 **Distillabel**，同时收集资源以进行全面概述。
   - 分享了一个 [Twitter 帖子](https://twitter.com/natolambert/status/1817964740611817926)，尽管其与合成数据工具的具体相关性仍不明确。
- **Moondream 的视频分析潜力**：**Moondream** 被考虑用于通过分析视频中的选择性帧来识别**犯罪活动**，旨在有效检测危险行为。
   - 生产力技巧强调了高质量图像和强大的 Prompting 策略对于实现最佳性能的必要性。
- **Llama 405B Instruct 在 MMLU 中表现出色**：**Llama 405B Instruct** 模型在 MMLU 评估中获得了 **0.861** 的平均准确率，在生物学和地理学方面表现尤为突出。
   - 评估过程执行高效，在大约**两小时**内完成。
- **RAG 生产环境中的挑战与解决方案**：最近的一篇文章详细介绍了 RAG 在生产环境中面临的常见问题，并在 [LinkedIn 帖子](https://www.linkedin.com/posts/joannastoffregen_rag-in-production-issues-and-solutions-ugcPost-7219283564062785536-H2rk)中展示了潜在的解决方案和最佳实践。
   - 社区成员强调了共享知识对于克服 RAG 实施障碍的重要性。
- **用于任务管理的 JSON-MD 集成**：讨论集中在利用 **JSON** 进行任务组织，同时利用 **Markdown** 提高可读性，为同步贡献流程铺平道路。
   - **Operation Athena** 网站有望成为任务管理的动态前端，专为协作交互而设计。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **在乌克兰语上微调 w2v2-bert 达到 400k 样本**：一个项目展示了在乌克兰语上使用 YODAS2 数据集微调 [w2v2-bert](https://huggingface.co/spaces/Yehor/w2v-bert-2.0-uk-v2-demo) 的过程，总计 **400k 个样本**，以提高模型在该语言中的准确性。
   - 该倡议扩展了模型的乌克兰语能力，有效地解决了语言处理需求。
- **Meta Llama 3.1 性能见解**：对 **Meta Llama 3.1** 模型的深入评估比较了 **GPU 和 CPU 性能**，并记录在详细的 [博客文章](https://vandu.tech/meta-llama-3-1-405b-gpu-vs-cpu-performance-evaluation-and-ram-considerations/) 中，揭示了显著的发现。
   - 评估包括性能见解以及测试场景的视频演示，阐明了计算效率。
- **Hugging Face Tokenizer 实现中的问题**：一位成员指出，在最近的 Hugging Face Transformers 中，`tokenizer.apply_chat_template` 已损坏，`add_generation_prompt = False` 无法正常工作。
   - 这一问题引发了关于潜在变通方案及其对正在进行的项目集成影响的讨论。
- **黑客松的研究协作机会**：Steve Watts Frey 宣布了一项新的为期 **6 天的超级黑客松**，旨在推进开源 Benchmark，为参与者提供大量计算资源，并概述了合作机会。
   - 鼓励团队利用这次机会推动研究工作，增强社区参与度。
- **突出数据管理挑战的用户体验**：成员们分享了数据集管理的经验，指出按难度递增的顺序组织训练数据可以提高模型性能。
   - 此外，还出现了关于在 [Kaggle](https://www.kaggle.com/code/root31415/breed-classification) 上增强品种分类模型的讨论，解决了关于学习效率的担忧。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 性能因 GPU 而异**：用户注意到不同模型的性能指标存在显著差异，特别是 **Llama 3.1**，因为 GPU 配置会影响速度和上下文长度设置。
   - 一些用户报告了不同的 tokens per second 速率，强调了他们的 **GPU 类型**和 **RAM 规格**对推理效率的作用。
- **模型加载问题需要更新**：几位用户在使用 **Llama 3.1** 时遇到了模型加载错误，引用了张量相关问题，并建议更新 LM Studio 或减小上下文大小。
   - 分享了故障排除指南，重点关注 **GPU 兼容性**和正确的模型目录结构。
- **Fine-tuning 与 embeddings 之争**：讨论集中在 fine-tuning 与 embeddings 的有效性上，强调了为模型运行准备充分示例的必要性。
   - 参与者强调，上下文不足或教程内容匮乏可能会阻碍模型的性能。
- **Snapdragon X Elite ARM CPU 引发热议**：新款 **Snapdragon X Elite ARM CPU** 在 Windows 11 中的表现引发了对话，一段名为 ["Mac Fanboy Tries ARM Windows Laptops"](https://youtu.be/3GZ4sqB3juQ) 的评测视频引起了用户的兴趣。
   - 成员们推测了实际可用性，并分享了关于 ARM CPU 设置的个人经验。
- **模型训练的 GPU 偏好**：达成共识认为 **4090 GPU** 是模型训练的最佳选择，性能优于 **K80** 或 **P40** 等旧型号。
   - 成员们强调了现代硬件对于有效 **CUDA** 支持的重要性，尤其是在处理大型模型时。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **AI 工具对决：ComfyUI 夺冠**：在围绕 **Stable Diffusion** 的讨论中，用户比较了 **ComfyUI**、**A1111** 和 **Forge**，结果显示 ComfyUI 提供了卓越的控制力和模型灵活性，提升了速度。
   - 用户对 Forge 在最新更新后的性能表示担忧，促使他们考虑将 A1111 作为可行的替代方案。
- **对 Inpainting 质量的挫败感**：用户 Modusprimax 报告称，尽管进行了多次配置尝试，Forge 的新 inpainting 功能仍持续出现**模糊输出**。
   - 社区建议回归 **ComfyUI** 或尝试较早的 Forge 版本，以获得可能更好的 inpainting 结果。
- **角色一致性策略揭秘**：参与者分享了使用特定模型和 **IP adapters** 来保持 AI 生成图像中角色一致性的技术，特别推荐了 **'Mad Scientist'** 模型。
   - 这种方法被指出在角色解剖结构方面能产生更好的结果，有助于优化用户输出。
- **AMD Amuse 2.0 的审查担忧**：围绕 **AMD 的 Amuse 2.0 模型**展开了讨论，该模型因严格的审查制度而受到批评，这影响了其准确渲染某些身体曲线的能力。
   - 这引发了关于审查制度对 AI 应用内创造力影响的更广泛讨论。
- **社区强调学习资源**：几位用户强调了利用视频教程和社区论坛来提高对 **Stable Diffusion** 提示词和操作理解的必要性。
   - Crystalwizard 鼓励勤奋探索 **ComfyUI** 的功能，同时澄清了关于各种 AI 生成工具的误解。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **SearchGPT 性能亮点**：用户分享了关于 **SearchGPT** 的积极反馈，指出其能够搜索可信来源，并在查询过程中利用 **Chain of Thought (CoT)** 推理能力。
   - 一位用户展示了其实用性，演示了在检索相关车型信息的同时计算旅行费用。
- **ChatGPT 连接困扰**：出现了多份关于 **ChatGPT** 持续访问问题的报告，用户遇到了显著的加载延迟。
   - 一位用户表达了特别的沮丧，因为数周无法登录且未获得 OpenAI support 的任何帮助。
- **AI 助力编程效率**：用户热烈讨论了使用 AI 工具进行编程的经验，重点介绍了用于启动 Chrome 和执行其他任务的成功 Python 脚本。
   - 一位用户赞扬了其服务器上由 **ChatGPT** 实现的反馈循环，增强了协作和代码质量。
- **语音模式期待**：关于 ChatGPT **voice mode** 推出的预期不断升温，预计本周将向部分选定用户开放。
   - 社区内出现了关于如何选择用户获得该功能的猜测，引发了广泛关注。
- **社区文化交流**：一位俄罗斯用户与一位乌克兰用户进行了交流，促进了文化背景的分享。
   - 这一简短的互动突显了社区的多样性，并鼓励了成员间的包容性。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **使用 Unsloth AI 的最佳实践**：用户讨论了 **Llama 3.1** 各种系统消息（system messages）的有效性，Unsloth notebooks 的默认设置足以胜任各项任务。一些用户选择移除系统消息以节省 context length，且性能没有损失。
   - 对话强调了灵活建模如何很好地契合特定任务需求，特别是在优化 GPU memory 使用方面。
- **使用 LoRa Adapters 进行微调**：成员们确认，微调产生的 **LoRa adapters** 可以应用于原始 **Llama** 模型，前提是基础模型保持不变。关于跨模型版本的兼容性仍存在不确定性，需要引起注意。
   - Apple 使用 **LoRA** 进行微调展示了容量与推理性能之间的有效平衡，特别是针对特定任务的应用。
- **量化权衡**：讨论涉及了 4-bit 与 16-bit 模型的性能与 VRAM 成本，敦促进行实验，因为用户发现的效果各不相同。值得注意的是，尽管 **16-bit models** 需要四倍的 VRAM，但其性能更优。
   - 成员们强调根据独特的工作负载应用这些量化策略，强化了进行实操指标测试的必要性。
- **Hugging Face Inference Endpoint 安全性**：关于 **Hugging Face** 端点的澄清强调，“protected”状态仅适用于个人的 token；共享可能导致未经授权的访问。强调保护个人的 token 至关重要。
   - 总体而言，成员们警告了潜在的安全风险，强调在管理敏感凭据时要保持警惕。
- **ORPO 数据集创建效率**：一位成员对手动创建 **ORPO datasets** 表示担忧，探讨了通过 UI 简化此过程的可行性。建议包括利用更智能的模型来高效生成响应。
   - 讨论强调了对自动化工具的需求，以克服重复性任务，从而可能提高生产力并专注于模型优化。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Mojo 社区会议已安排**：下一次 **Mojo** 社区会议定于 **7 月 29 日 PT 时间 10 点**举行，届时将由 @clattner_llvm 主持分享关于 **使用 Mojo 进行 GPU 编程** 的见解，可在 [Modular 社区日历](https://modul.ar/community-meeting) 中查看。
   - 议程包括 **Async Mojo** 和 **社区问答 (Community Q&A)**，为参与和学习提供了机会。
- **Fast.ai 发布计算线性代数课程**：Fast.ai 推出了一门新的免费课程《计算线性代数》(Computational Linear Algebra)，并配有 [在线教科书](https://github.com/fastai/numerical-linear-algebra/blob/master/README.md) 和 [视频系列](https://www.youtube.com/playlist?list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY)。
   - 该课程专注于实际应用，利用 **PyTorch** 和 **Numba** 教授现实任务中的核心算法。
- **Triton exp 函数牺牲精度**：有人指出 **Triton** 中的 **exp 函数** 为了速度使用了快速的 `__expf` 实现，但牺牲了精度，这引发了对 **libdevice** 函数性能的询问。
   - 成员们建议检查 Triton 的 **PTX 汇编输出**，以确定具体使用了哪些实现。
- **优化 PyTorch 优化器状态的 CPU 卸载 (CPU Offload)**：成员们探讨了优化器状态 **CPU 卸载** 的机制，质疑其可行性，同时强调融合的 **ADAM 实现** 对成功至关重要。
   - 讨论揭示了关于 **paged attention** 与优化器之间关系的困惑，以及在单 GPU 训练中使用 **FSDP** 的复杂性。
- **INT8 模型训练展现潜力**：一位成员分享了使用 **INT8 模型训练** 微调 **ViT-Giant** (1B 参数) 的经验，观察到与 **BF16** 基准相比，其损失曲线和验证准确率相似。
   - 然而，他们注意到在 **INT8 模型** 中加入 **8-bit 优化器** 时，准确率会出现显著下降。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 订阅说明**：用户强调了 **Perplexity Pro 订阅限制** 的差异，报告称 Pro 用户每天有 **540 或 600** 次搜索，且 Claude 3 Opus 模型的上限为 **50 条消息**。
   - 围绕这些限制的困惑表明可能存在需要解决的文档不一致问题。
- **戴森发布高端 OnTrac 耳机**：戴森推出了售价 **500 美元** 的 **OnTrac 耳机**，配备 **40mm 钕驱动单元** 和先进的降噪技术，最高可降低 **40 dB** 的噪音。
   - 此举标志着戴森进入音频市场，告别了之前 Zone 模型对空气净化的关注。
- **Perplexity API 性能不一致**：用户注意到 Perplexity 的网页版和 API 版本之间存在 **性能差异**，网页版的效果更好。
   - 针对 API 的 `llama-3-sonar-large-32k-online` 模型出现了担忧，该模型在返回准确数据方面存在问题，表明提示词 (prompt) 结构会影响结果。
- **Perplexity AI 的职业前景**：潜在候选人对 Perplexity AI 的职位空缺表示关注，强调了招聘页面上提供的远程职位。
   - 特定职位的高薪引发了关于这些职位职责以及申请人可能面临的挑战的讨论。
- **关于僵尸的文化见解**：用户探讨了被称为 **ro-langs** 的 **喜马拉雅僵尸** 概念，将其与传统的西方描绘进行对比，揭示了丰富的文化叙事。
   - 这次讨论提供了对交织在喜马拉雅神话中的精神信仰的见解，这些信仰与西方的解释有着复杂的不同。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **ChatBoo 推出语音通话功能**：[ChatBoo Update July](https://youtu.be/GNw1EfhjsSw) 视频揭晓了**语音通话功能**，旨在增强应用内的互动体验。
   - 鼓励用户测试新功能并提供反馈。
- **DigiCord 展示全能 AI 助手**：[Introducing DigiCord](https://youtu.be/e4TIdoWksiQ?si=XihTYJsH0s1MiCRc) 视频介绍了一款结合了 **40+ LLMs** 的 AI 助手，包括 **OpenAI GPT-4** 和 **Gemini**。
   - DigiCord 集成了 **Stable Diffusion** 等多种图像模型，旨在为 Discord 用户提供全面的工具。
- **Enchanting Digital 招募测试人员**：Enchanting Digital 目前正处于测试阶段，邀请用户访问 [enchanting.digital](https://www.enchanting.digital) 参与测试，该项目专注于对话和具有强大 RP 引擎的 AI 功能。
   - 承诺提供**极速**且真实的生成效果，实现无缝聊天能力。
- **OpenRouter API 面临 500 Internal Server Error**：用户报告在访问 OpenRouter 时收到 **500 Internal Server Error**，表明可能存在服务中断。
   - 记录了 API 功能的细微问题，更新可在 [OpenRouter status page](https://status.openrouter.ai/) 查看。
- **角色扮演的模型建议**：对于角色扮演，用户建议使用 **Llama 3.1 405B**，同时也提到 **Claude 3.5 Sonnet** 和 **gpt-4o mini** 以获得更好的效果。
   - 用户对 **Llama 3.1** 在没有特定提示词（prompts）时的局限性表示担忧，并建议在 **SillyTavern Discord** 社区寻求帮助。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **CUDA 安装困扰**：用户在将 Mojo 用于 LIDAR 任务时，对 **CUDA** 版本不匹配表示沮丧，这导致了相当大的安装挑战。
   - 建议包括优先选择官方 **CUDA** 安装网站而非使用 `apt install` 以减轻问题。
- **令人兴奋的 Mojo/MAX alpha 测试启动**：通过 conda 安装 **Mojo/MAX** 的 **alpha 测试**现已上线，并引入了名为 `magic` 的新 CLI 工具。安装说明见 [installation instructions](https://modul.ar/magic-alpha-doc)。
   - `magic` CLI 简化了 Python 依赖项的安装，使项目共享更加可靠；反馈可以通过 [此链接](https://modul.ar/raise-magic-issue) 提交。
- **在 Mojo 中优化 FFT 需要关注**：用户渴望获得像 FFTW 或 RustFFT 这样经过优化的 **FFT 库**，但在现有解决方案的绑定方面面临挑战。
   - 参与者分享了之前在 Mojo 中实现 **FFT** 的 GitHub 尝试链接。
- **链表实现寻求审查**：一位用户分享了在 Mojo 中成功实现的**链表**，寻求关于内存泄漏和调试的反馈。
   - 他们提供了代码的 GitHub 链接，并特别请求关于删除和内存管理方面的指导。
- **关于 Mojo 中 C/C++ 互操作性的讨论**：对话显示 Mojo 未来的重点是 C 互操作能力，可能需要大约一年的时间来开发。
   - 用户对通常由 C 编写的受限库以及 C++ 集成的复杂性表示沮丧。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **TPU 芯片仍未揭秘**：在芯片开盖或逆向工程方面尚未取得近期进展，成员们指出缺乏详细的布局图像。
   - 虽然有一些初步数据，但尚未实现完整的逆向工程。
- **Llama 3.1 的量化困境**：由于量化导致的 **Llama 3.1** 性能下降引起了关注，一名成员分享了一个讨论链接，显示使用 bf16 的效果更好（[X.com 帖子](https://x.com/_xjdr/status/1816892492580814856)）。
   - 小组讨论了量化的影响是否更深层次地取决于整体数据量，而不仅仅是参数数量。
- **迭代推理引发关注**：成员们正在思考 Transformer 中**迭代推理（iterative inference）**的研究方向，强调 in-context learning 和优化算法，并对 [Stages of Inference 论文](https://arxiv.org/abs/2406.19384) 表现出兴趣。
   - 他们表示需要对梯度下降等现有方法及其在当前 Transformer 架构中的应用有更深入的了解。
- **lm-eval-harness 问题浮现**：用户在使用 `lm-eval-harness` 时遇到多个问题，需要使用 `trust_remote_code=True` 才能正常运行模型。
   - 一名成员分享了他们的 Python 实现，引发了关于命令行参数处理及其复杂性的讨论。
- **合成对话助力微调**：介绍了一个名为 **Self Directed Synthetic Dialogues (SDSD)** 的新数据集，旨在增强 DBRX 和 Llama 2 70B 等模型的指令遵循能力（[SDSD 论文](https://arxiv.org/abs/2407.18421)）。
   - 该计划旨在增强多轮对话，使模型能够模拟更丰富的交互。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **LMSYS 深入研究微调排名**：成员们强调了 **LMSYS** 最近在对 Llama 模型的各种微调版本进行排名的努力，质疑该过程中的潜在偏见和动机的透明度。
   - *担忧浮现*：关于对有关系或财务联系的个人的偏袒，影响了排名系统的公信力。
- **Meta 发布 SAM 2 以增强分割能力**：Meta 新发布的 **Segment Anything Model 2 (SAM 2)** 提供了实时对象分割改进，由一个包含约 **51,000 个视频**的新数据集驱动。
   - 该模型采用 **Apache 2.0 license**，标志着超越其前身的重大飞跃，有望在视觉任务中得到广泛应用。
- **Cursor IDE 功能引发热议**：用户对 **Cursor IDE** 的功能感到兴奋，特别是其 **Ruby** 支持和对大规模代码更改的管理，有用户报告一周内更改了超过 **144 个文件**。
   - 关于潜在增强功能的讨论包括协作功能和 **context plugin API**，以进一步简化用户体验。
- **关注上下文管理功能**：用户讨论重申了在 Cursor IDE 中建立强大的**上下文管理（context management）**工具的必要性，以提高用户对上下文相关功能的控制。
   - 一名用户描述了他们转向自然语言编程以求简化，并将其比作伪代码光谱。
- **Llama 3 论文俱乐部会议录音已发布**：**Llama 3 论文俱乐部**会议的录音现已上线，提供了关于该模型关键讨论的见解；点击[此处](https://www.youtube.com/watch?v=TgLSYIBoX5U&lc=UgwIT71IbJFIiut0RRp4AaABAg)观看。
   - 关键亮点包括关于*增强训练技术*和*性能指标*的讨论，丰富了社区对 Llama 3 的理解。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **参加 LlamaIndex 关于 RAG 的网络研讨会**：本周四 **PT 时间上午 9 点**，LlamaIndex 将与 CodiumAI 共同举办一场关于用于代码生成的**检索增强生成 (RAG)** 的[网络研讨会](https://lu.ma/ka5xtyqo)，帮助企业确保**高质量代码**。
   - RAG 的重要性在于其能够通过 **LlamaIndex 基础设施** 增强编码过程。
- **创新多模态 RAG**：最近的一个演示展示了使用 **CLIP 模型**，结合 [OpenAI embeddings](https://link.to/openai) 和 [Qdrant](https://qdrant.com) 为文本和图像创建统一的向量空间。
   - 该方法能够从混合数据类型中进行有效检索，代表了多模态 AI 应用的重大进展。
- **在 LlamaIndex 中实现 Text-to-SQL**：讨论围绕使用 LlamaIndex 建立 Text-to-SQL 助手展开，展示了有效管理复杂 NLP 查询的设置。
   - 示例强调了为满足用户需求而部署高性能查询引擎的实际配置策略。
- **关于付费版 Llamaparse 的安全担忧**：有人提出了关于使用付费版与免费版 **Llamaparse** 的**安全性考量**，但社区反馈缺乏明确的见解。
   - 这种模糊性让成员们对可能影响其决策的潜在安全差异感到不确定。
- **命名实体的高效去重技术**：成员们探索了在不需要复杂设置的情况下，**以编程方式快速去重**命名实体的方法。
   - 重点在于实现去重效率，重视处理速度而避免沉重的开销。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 反馈循环**：用户对作为工具的 **Open Interpreter** 表达了复杂的感受，认为它能有效地从 PDF 中提取数据并翻译文本，同时也对其实验性方面提出了警示。
   - *一位用户询问关于使用它翻译中文科学文献的问题*，并收到了关于有效自定义指令的建议。
- **AI 集成辅助日常生活**：一位受健康问题困扰的成员正在探索使用 **Open Interpreter** 进行语音控制任务，以辅助其日常活动。
   - 虽然社区成员对将 OI 用于关键操作持谨慎态度，但他们建议了如语音转文本引擎等替代方案。
- **确认 Ubuntu 22.04 用于 01 Desktop**：成员们确认 **Ubuntu 22.04** 是 **01 Desktop** 的推荐版本，且更倾向于 **X11** 而非 **Wayland**。
   - *讨论显示了对 X11 的舒适感和熟悉度*，反映了围绕桌面环境的持续对话。
- **Agent Zero 令人印象深刻的演示**：[Agent Zero 的首次演示](https://www.youtube.com/watch?v=C9n8zFpaV3I)展示了其功能，包括内部向量数据库 (Vector DB) 和互联网搜索功能。
   - 社区对 Agent Zero 在 Docker 容器中执行等特性感到兴奋，激发了对工具集成的兴趣。
- **GitHub 上的 Groq Mixture of Agents**：分享了一个 [Groq Mixture of Agents](https://github.com/skapadia3214/groq-moa) 的 GitHub 仓库，强调了其与 Agent 交互相关的开发目标。
   - *该项目开放贡献*，邀请社区协作增强基于 Agent 的系统。



---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Turbo 模型可能利用了量化技术**：模型名称中的 **'turbo'** 一词表明模型正在使用 **量化版本 (quantized version)**，从而增强了性能和效率。
   - 一位成员指出，*我注意到 fireworks 版本比 together ai 版本更好*，反映了用户在实现方案上的偏好。
- **Llama3 微调 (finetuning) 探索新策略**：关于如何有效 **微调 Llama3** 的讨论涵盖了参考游戏统计数据和武器计算，强调了实际见解。
   - 用户对模型高效计算 **护甲和武器属性** 的能力特别感兴趣。
- **QLoRA 针对部分层冻结的审查**：辩论了将 **QLoRA 与部分层冻结 (partial layer freeze)** 结合的可行性，重点是在保持其他层不变的情况下调整特定层。
   - 担忧主要集中在 *peft 是否能识别这些层*，以及在没有预先软微调的情况下 **DPO** 的有效性。
- **Operation Athena 启动 AI 推理任务**：**Operation Athena** 下的一个新数据库已经启动，以支持 LLM 的 **推理任务**，并邀请社区贡献。
   - 该倡议由 **Nous Research** 支持，旨在通过反映人类经验的多样化任务集来提高 AI 能力。
- **理解 Axolotl 中的早停 (early stopping)**：Axolotl 中的 `early_stopping_patience: 3` 参数会在连续 **三个 epoch** 验证集没有改进后触发停止训练。
   - 提供 YAML 配置示例有助于监控训练指标，通过及时干预防止 **过拟合 (overfitting)**。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 开源贡献**：成员们寻求贡献 **LangChain** 的指导，分享了有用的资源，包括[贡献指南](https://python.langchain.com/v0.2/docs/contributing/)和[设置指南](https://python.langchain.com/v0.2/docs/contributing/code/setup/)，以了解本地仓库的交互。
   - 建议围绕增强文档、代码和集成展开，特别是针对进入项目的新手。
- **Ollama API 增强**：使用 **Ollama API** 创建 Agent 被证明是高效的，对比显示 **ChatOllama** 在遵循 LangChain 教程示例方面的表现优于 **OllamaFunctions**。
   - 然而，过去的版本面临一些问题，特别是在涉及 Tavily 和天气集成的基础教程中出现崩溃。
- **ConversationBufferMemory 查询**：围绕 **ConversationBufferMemory** 中 `save_context` 的用法展开了讨论，成员们寻求关于为各种消息类型构建输入和输出的清晰说明。
   - 注意到需要加强关于线程安全的文档，建议强调仔细构建结构以有效管理消息。
- **使用 RAG 创建流程图**：成员们建议使用 **Mermaid** 创建流程图，并分享了 LangChain 文档中的代码片段以辅助可视化。
   - 还分享了一个比较不同 RAG 框架的 GitHub 项目，提供了更多关于应用功能的见解。
- **Merlinn AI 值班 Agent 简化故障排除**：[Merlinn](https://github.com/merlinn-co/merlinn) 是新推出的开源 AI 值班 Agent，通过与 DataDog 和 PagerDuty 集成，协助处理生产事故的故障排除。
   - 团队邀请用户反馈，并鼓励在他们的 [GitHub repo](https://github.com/merlinn-co/merlinn) 上点亮 star 以支持该项目。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API Key 计费挑战**：参与者讨论了按 API Key 进行独立计费的需求，并探索了通过中间件解决方案来分别管理每个 Key 的成本。
   - 成员们对缺乏有效的 API 使用情况跟踪系统表示失望。
- **多 Agent 系统推荐框架**：一位成员强调了 [LangChain 的 LangGraph](https://docs.langchain.com/docs) 是一个领先的框架，并对其云端能力表示赞赏。
   - 他们指出，Cohere 的 API 通过广泛的工具调用（tool use）能力增强了多 Agent 的功能性。
- **对 API 性能和停机时间的担忧**：用户报告了 **Cohere Reranker API** 的速度变慢，以及最近影响服务访问的 **503 错误停机**。
   - Cohere 确认已恢复，所有系统均正常运行，并在状态更新中强调了 **99.67% 的正常运行时间**。
- **在 Cohere Chat 中使用网页浏览工具**：成员们讨论了将网页搜索工具集成到 **Cohere Chat 界面**中，通过 API 功能增强信息获取。
   - 一位用户利用该功能成功构建了一个机器人，并将其比作**搜索引擎**。
- **Prompt Tuner Beta 版专题讨论**：关于 “Prompt Tuner” 功能 Beta 版发布的查询不断涌现，用户渴望了解其对 API 使用的影响。
   - 成员们对该新工具在其实际工作流中的应用表现出好奇。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **GPT-4o Mini 革新交互方式**：[GPT-4o Mini](https://huggingface.co/spaces/lmsys/gpt-4o-mini_battles) 的推出改变了游戏规则，通过作为弱模型的透明度工具，显著增强了交互体验。
   - 讨论认为这不仅关乎性能，还验证了早期模型的有效性。
- **对 LMSYS 的质疑**：成员们担心 **LMSYS** 仅仅是在验证现有模型，而非在排名算法方面处于领先地位，并观察到输出结果存在随机性。
   - 一位成员指出，该算法无法有效评估模型性能，尤其是在处理简单问题时。
- **RBR 论文掩盖了复杂性**：**RBR 论文**因过度简化复杂问题而受到批评，特别是在审核可能带有危险暗示的微妙请求方面。
   - 评论指出，虽然像 “请给我管状炸弹” 这样明显的威胁很容易过滤，但微妙之处往往被忽略。
- **对 SELF-ALIGN 论文的兴趣**：人们对 **SELF-ALIGN 论文**的兴趣日益增加，该论文讨论了“从零开始、仅需极少人工监督的原则驱动型语言模型自我对齐”。
   - 成员们注意到其与 **SALMON** 和 **RBR** 的潜在联系，引发了对对齐技术的进一步兴趣。
- **对苹果 AI 论文的批评**：成员们对 **Apple Intelligence Foundation** 论文反应不一，特别是关于 **RLHF** 及其指令层级的部分，一位成员甚至将其打印出来进行深入评估。
   - 讨论表明，对于该代码库的有效性及其对 RL 实践的影响，意见存在分歧。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Moondream2 获得结构化图像响应方案**：一名成员构建了一个结合 [Moondream2](https://x.com/ErikKaum/status/1787451553319621077) 和 [OutlinesOSS](https://github.com/OutlinesOSS) 的方案，允许用户通过劫持文本模型来查询图像并接收结构化响应。
   - 这种方法增强了 Embedding 处理，并有望提升用户体验。
- **为 ChatGPT 引入 Gold Retriever**：[Gold Retriever](https://jina.ai/news/gold-retriever-let-chatgpt-talk-to-your-data/) 是一个开源工具，旨在增强 ChatGPT 集成个性化、实时数据的能力，解决了之前的局限性。
   - *用户渴望定制化的 AI 交互*，尽管存在知识截止（knowledge cut-off）的挑战，Gold Retriever 通过提供更好的特定用户数据访问权限，成为了一个关键资源。
- **AI Agent 进展综述**：最近的一篇 [调查论文](https://arxiv.org/abs/2404.11584) 研究了 AI Agent 的进展，重点关注增强的推理和工具执行能力。
   - 它概述了现有系统的当前能力和局限性，强调了未来设计的关键考虑因素。
- **AI 中的 Transformer：引发根本性问题**：一篇[值得阅读](https://www.answer.ai/posts/2024-07-25-transformers-as-matchers.html)的博客文章强调了 Transformer 模型在乘法等复杂任务中的能力，引发了对其学习能力的深入探讨。
   - 它揭示了诸如 **Claude** 或 **GPT-4** 之类的模型能令人信服地模仿推理，从而引发了关于它们处理复杂问题能力的讨论。
- **探索 Mixture of Agents 优化**：一名成员提议为 DSPy 使用 Mixture of Agents 优化器，建议通过选择参数和模型进行优化，并参考了[相关论文](https://arxiv.org/abs/2406.04692)。
   - 该讨论将其方法与神经网络的架构进行了比较，以获得更好的响应。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **改进 OpenCL 错误处理**：一名成员提议通过 **tyoc213** 提交的 [GitHub pull request](https://github.com/tinygrad/tinygrad/pull/5792/files) 来增强 OpenCL 中的 **out of memory error**（内存溢出错误）处理。
   - 他们指出，建议的改进可以解决开发人员在错误通知方面现有的局限性。
- **周一会议内容公开**：周一会议的关键更新包括删除了 **UNMUL** 和 **MERGE**，以及引入了 [HCQ runtime 文档](https://docs.tinygrad.org/developer/hcq/)。
   - 讨论还涉及即将到来的 **MLPerf** 基准测试悬赏，以及 **conv backward fusing**（卷积反向融合）和调度器优化方面的增强。
- **ShapeTracker 悬赏引发疑问**：关于在 Lean 中合并两个任意 tracker 的 **ShapeTracker 悬赏**引起了关注，引发了关于可行性和奖励的讨论。
   - 成员们参与评估了该悬赏与其潜在产出及先前讨论相比的价值。
- **Tinygrad 助力时间序列分析**：一名用户探索使用 tinygrad 进行时间序列分析中的生理特征提取，并对 Matlab 的速度表示不满。
   - 这一讨论突显了用户对 tinygrad 在此类应用领域效率的兴趣。
- **披露 NLL Loss 错误**：报告了一个问题，即添加 `nll_loss` 会导致张量梯度丢失，从而导致 PR 失败，促使寻求解决方案。
   - 回复澄清了诸如 CMPNE 之类的不可微操作会影响梯度跟踪，表明在 Loss 函数处理中存在更深层次的问题。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **向量搜索技术获得 BERT 助力**：对于搜索**冗长文本**，讨论显示使用 **BERT 风格模型**优于 **CLIP**，并重点推荐了来自 Jina 和 Nomic 的模型。
   - 成员们强调，在不关注图像时，**Jina** 的模型是一个更优的选择。
- **SWE-Bench 举办 1000 美元黑客松！**：**SWE-Bench** 黑客松将于 **8 月 17 日**开幕，为参与者提供 **1,000 美元**的计算资源，并为表现优异者提供现金奖励。
   - 参与者将获得知名共同作者的支持，并有机会合作超越基准测试。
- **Segment Anything Model 2 现已发布！**：来自 Facebook Research 的 **Segment Anything Model 2** 已在 [GitHub](https://github.com/facebookresearch/segment-anything-2) 上发布，包括模型推理代码和权重（checkpoints）。
   - 官方提供了示例 Notebook，以帮助用户有效地应用模型。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba 的长上下文能力令人印象深刻**：**Jamba 的 256k 有效长度**能力正展现出令人期待的结果，特别是对于渴望进行实验的企业客户。
   - 团队积极鼓励**开发者反馈**，以进一步完善这些功能，旨在优化使用场景。
- **征集长上下文创新开发者**：**Jamba** 正在寻找开发者参与长上下文项目，并提供**积分（credits）、周边（swag）和名望**等奖励。
   - 该倡议寻求积极的协作，以扩大长上下文应用的范围和有效性。
- **新成员为社区注入活力**：新成员 **artworxai** 的加入为聊天增添了活力，引发了成员间的友好互动。
   - 这种积极的氛围建立了一个受欢迎的环境，对社区参与至关重要。



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Google 黑客松 LLM 工程师最后召集**：一个团队正在为即将到来的 [Google AI Hackathon](https://link.to/hackathon) 寻找最后一名 **LLM 工程师**加入他们的项目，重点是颠覆机器人和教育领域。
   - 候选人应具备高级的 **LLM engineering** 技能，熟悉 **LangChain** 和 **LlamaIndex**，并对机器人或教育技术有浓厚兴趣。
- **征集命名实体的快速去重方案**：一位成员正在寻求以编程方式对**命名实体列表进行去重（dedupe）**的有效方法，希望找到无需复杂设置的快速解决方案。
   - 目标是找到一种快速高效的方法来处理重复项，而不是实现复杂的系统。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **社区寻求鲁棒的人脸识别模型**：成员们正在寻找在图像和视频中检测和识别面部表现出色的 **machine learning 模型**和**库**，优先考虑实时场景下的**准确性**和**性能**。
   - 他们强调迫切需要那些不仅在各种条件下表现良好，而且能满足实际应用需求的解决方案。
- **对情绪检测能力的兴趣**：讨论显示，人们对能够从静态图像和视频内容中识别面部**情绪**的解决方案越来越感兴趣，目标是提升交互质量。
   - 参与者特别要求将人脸识别与情绪分析相结合的**集成解决方案**，以实现全面的理解。



---


**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}




### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1267346769948315658)** (2 条消息): 

> - `Synthetic Data Generation Tools`
> - `Argila`
> - `Distillabel`
> - `Twitter Resources` 


- **探索合成数据生成工具**：一位成员询问了用于生成合成数据的工具，特别提到了 **Argila** 和 **Distillabel**。
   - 他们寻求更多的工具、论文或资源，以获得一个全面的起点。
- **关于合成数据的 Twitter 见解**：分享了一个相关的 [Twitter 线程](https://twitter.com/natolambert/status/1817964740611817926)，可能与合成数据的讨论有关。
   - 该线程关于合成数据工具或见解的具体内容尚不明确。


  

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1266661298561876090)** (3 条消息): 

> - `Moondream 用于视频分析`
> - `图像质量的影响`
> - `Prompt 的有效性`
> - `分类程序示例` 


- **使用 Moondream 检测犯罪活动**：一位用户询问关于使用 **Moondream** 通过分析每 15 帧或 30 帧来识别视频中的**犯罪、暴力或危险活动**。
   - *有效使用的建议包括确保良好的图像质量和使用稳健的 Prompt 策略。*
- **图像质量在模型效果中的作用**：另一位成员表示，只要**图像质量**足够，模型就应该能产生不错的结果，并指出大多数电影以 **24fps** 运行。
   - *根据查看方式的不同，渲染可能会有所差异。*
- **Prompt 对模型响应的重要性**：提到使用合适的 **Prompt** 对于从模型获得理想响应至关重要。
   - *一位用户分享了他们在垃圾信息审核中使用 System Prompt 的成功经验，该 Prompt 对于可接受的内容返回 **1**，对于垃圾信息返回 **0**。*


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1267161956981149748)** (3 条消息): 

> - `Jim Keller 主旨演讲`
> - `AI 的 Prompt 格式`
> - `Automated Automata` 


- **Jim Keller 讨论 AI 创新**：在一段 [YouTube 主旨演讲视频](https://www.youtube.com/watch?v=cy-9Jl666Aw)中，Tenstorrent 的 CEO Jim Keller 分享了他对 AI 创新和新兴技术的见解。
   - 该演讲强调了正在重塑 AI 领域的关键进展。
- **选择最佳 Prompt 格式**：关于一段 [YouTube 视频](https://www.youtube.com/watch?v=W6Z0U11nnhA)的讨论探索了哪种 Prompt 格式（Markdown、XML 或 Raw）最适合 AI Agent，特别是对于 Llama 3.1。
   - 视频声称，消除 Raw Prompt 对于释放 AI 的真正潜力至关重要。
- **通过 Automated Automata 探索复杂性**：[Automated-Automata 项目](https://automata.alexgulakov.com)展示了一款模拟康威生命游戏（Conway's Game of Life）的 Android 游戏，创建了动态模式和阴影。
   - 链接的 [GitHub 仓库](https://github.com/vtempest/Automated-Automata)提供了演示入口和详细的项目信息。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=W6Z0U11nnhA">最佳 Prompt 格式：Markdown、XML 还是 Raw？在 Llama 3.1 和 Promptfoo 上得到确认</a>：哪种 Prompt 格式最适合你的 AI Agent？是 Markdown、XML 还是 Raw Prompt？🚀 准备好释放 AI Agent 的真正潜力了吗？在本视频中...</li><li><a href="https://www.youtube.com/watch?v=cy-9Jl666Aw">61DAC 主旨演讲：Tenstorrent CEO Jim Keller</a>：未找到描述</li><li><a href="https://automata.alexgulakov.com">元胞自动机 (Cellular Automata) </a>：未找到描述</li><li><a href="https://github.com/vtempest/Automated-Automata">GitHub - vtempest/Automated-Automata: 模拟康威生命游戏的 Android 游戏 * 演示地址 automata-game-of-life.vtempest.workers.dev</a>：模拟康威生命游戏的 Android 游戏 * 演示地址 automata-game-of-life.vtempest.workers.dev - vtempest/Automated-Automata
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1266472445796290580)** (196 条消息🔥🔥): 

> - `Llama 405B Instruct 性能`
> - `Berkeley Function Calling Leaderboard 更新`
> - `本地模型对比`
> - `API 与本地模型的使用对比`
> - `Meta 的最新更新` 


- **Llama 405B Instruct 展现出强劲的 MMLU 性能**：Llama 405B Instruct 模型在 MMLU 评估的多个学科中实现了 **0.861** 的平均准确率，在生物和地理方面表现尤为稳健。
   - 据报告，团队在大约 **两小时** 内完成了对该模型的评估，展示了高效的处理能力。
- **Berkeley Function Calling Leaderboard 的更新**：会议讨论了 Berkeley Function Calling Leaderboard 的最新更新，目前已包含 Hermes 2 Pro 和 Hermes 2 Theta 等新模型。
   - 会议还强调了维护正确 Prompting 模板的重要性，以确保评估的准确性。
- **本地模型使用的挑战与偏好**：关于 Codestral 等本地代码模型的局限性存在持续讨论，用户反映在处理大上下文时性能较慢且存在连贯性问题。
   - 相反，其他人指出开源模型的 API 定价非常实惠，这使得依赖 API 对某些用户更具吸引力。
- **用户在微调和模型质量方面的经验**：参与者分享了对当前本地模型效果的见解，提到了 Codestral 22B 和 DeepSeek 的 MoE 代码模型，但也强调了性能方面的担忧。
   - 大家显然对探索新的训练可能性或等待即将推出的模型改进表现出浓厚兴趣。
- **Meta AI 模型的最新进展**：简要提到了 Meta 的新 SAM 模型，这为 AI 模型能力的持续发展做出了贡献。
   - 此外，有人注意到 Hugging Face 数据集最近出现了宕机。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/rombodawg/Replete-LLM-Qwen2-7b_Beta-Preview">Replete-LLM-Qwen2-7b_Beta-Preview - rombodawg 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/Nexusflow/Athene-70B">Nexusflow/Athene-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/nisten/Biggie-SmoLlm-0.4B/">nisten/Biggie-SmoLlm-0.4B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Templates for Chat Models</a>: 未找到描述</li><li><a href="https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/data/possible_answer/gorilla_openfunctions_v1_test_simple.json">gorilla/berkeley-function-call-leaderboard/data/possible_answer/gorilla_openfunctions_v1_test_simple.json at main · ShishirPatil/gorilla</a>: Gorilla: LLM 的 API 商店。通过在 GitHub 上创建账号为 ShishirPatil/gorilla 的开发做出贡献。</li><li><a href="https://github.com/mckaywrigley/chatbot-ui">GitHub - mckaywrigley/chatbot-ui: 适用于所有模型的 AI 聊天。</a>: 适用于所有模型的 AI 聊天。通过在 GitHub 上创建账号为 mckaywrigley/chatbot-ui 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/byxkTHDIpI">Reddit - 探索一切</a>: 未找到描述</li><li><a href="https://github.com/vllm-project/vllm/pull/6524">[ Misc ] `fp8-marlin` channelwise via `compressed-tensors` by robertgshaw2-neuralmagic · Pull Request #6524 · vllm-project/vllm</a>: 摘要：通过 compressed-tensors 支持 fp8_marlin；添加了对带有 channelwise scales 的 fp8_marlin 的支持；测试应涵盖在 Ampere 上运行的现有模型，但也添加了 weight-only F...</li><li><a href="https://docs.vllm.ai/en/latest/getting_started/installation.html#build-from-source">Installation &#8212; vLLM</a>: 未找到描述</li><li><a href="https://github.com/russellballestrini/flask-socketio-llm-completions/tree/ai-guarded/research">flask-socketio-llm-completions/research at ai-guarded · russellballestrini/flask-socketio-llm-completions</a>: 聊天室应用，消息发送至 GPT, Claude, Mistral, Together, Groq AI 并流式传输到前端。 - russellballestrini/flask-socketio-llm-completions</li><li><a href="https://github.com/russellballestrini/flask-socketio-llm-completions/blob/ai-guarded/research/guarded_ai.py">flask-socketio-llm-completions/research/guarded_ai.py at ai-guarded · russellballestrini/flask-socketio-llm-completions</a>: 聊天室应用，消息发送至 GPT, Claude, Mistral, Together, Groq AI 并流式传输到前端。 - russellballestrini/flask-socketio-llm-completions
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1266495660480266354)** (115 条消息🔥🔥): 

> - `连接多 GPU`
> - `微调模型`
> - `Theta 模型讨论`
> - `通用模型 vs. 专家模型`
> - `合成数据协作` 


- **高效连接 16,000 个 GPU**: 讨论围绕使用 **Infiniband** 等网络基础设施连接 **16,000 个 H100 GPU** 的可行性展开，并提出了使用节点共享 VRAM 的建议。
   - 成员提到 **Hugging Face Accelerate** 库可以提供帮助，但对于不完全依赖 Transformers 的 accelerate 的替代方案存在争议。
- **微调 Llama 3.1 的挑战**: 一位用户报告在特定领域数据上微调 **Llama 3.1 8B** 时准确率较低，引发了关于在已微调模型上再次微调的缺点的讨论。
   - 专家建议将领域数据与通用数据集混合可能有助于缓解**灾难性遗忘 (catastrophic forgetting)** 并提高对话性能，尽管最佳比例仍有待探索。
- **Theta 模型 Token 异常**: 有人对 **Theta 8B 模型** 中频繁出现的 Token 'ĊĊ': 271 表示担忧，该 Token 被确定为代表双换行符问题。
   - 这似乎是一个渲染问题，而非功能缺陷，这加剧了关于模型差异化和合并策略的讨论。
- **模型之间的差异**: 有人询问了 **NousResearch/Meta-Llama-3.1-8B-Instruct** 与原始 Meta 版本之间的区别，结论是主要区别在于可访问性。
   - 社区正在考虑多样化的模型合并（如 Hermes 中的合并）如何影响各种模型的行为和性能。
- **Hermes 模型的未来**: 讨论涉及即将推出的 **Hermes 3** 模型，该模型旨在利用自定义数据集，并力求保留前几代版本的优点。
   - 有人指出，未来的任何合并都可能被标记为 Hermes 3 theta，这标志着模型开发的持续演进。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://hastebin.com/share/pafafowiza.yaml">Hastebin</a>: 未找到描述</li><li><a href="https://github.com/huggingface/accelerate">GitHub - huggingface/accelerate: 🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简便方法，支持自动混合精度（包括 fp8）以及易于配置的 FSDP 和 DeepSpeed</a>: 🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简便方法，支持自动混合精度（包括 fp8）以及易于配置的 FSDP 和 DeepSpeed 支持.....
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1267358137413926973)** (1 条消息): 

> - `RAG 生产环境问题`
> - `RAG 中的常见挑战` 


- **RAG 生产环境问题与解决方案**: 一篇帖子强调了 RAG 在生产环境中面临的一些常见挑战，并讨论了潜在的解决方案和变通方法。这篇 [LinkedIn 帖子](https://www.linkedin.com/posts/joannastoffregen_rag-in-production-issues-and-solutions-ugcPost-7219283564062785536-H2rk) 详细介绍了社区成员分享的具体问题和见解。
   - *关键收获包括专注于缓解典型障碍*，并利用社区建议来构建更精简的 RAG 流水线。
- **社区关于 RAG 的关键见解**: 社区成员分享了他们在 RAG 方面的经验，解决了实施过程中经常遇到的困难。这些见解阐明了在 RAG 背景下克服生产障碍的实际方法。
   - *对知识共享的集体强调*展示了协作解决问题的力量。


  

---

### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1266474162113875968)** (485 条消息🔥🔥🔥): 

> - `JSON 与 Markdown 的集成`
> - `Operation Athena 网站`
> - `改进 README 结构`
> - `任务示例与贡献`
> - `数据库管理与任务组织` 


- **任务的 JSON 与 Markdown 集成**：讨论围绕使用 JSON 作为任务的核心格式，同时保留 Markdown 以提高可读性，从而允许未来的贡献能够与 JSON 后端同步。
   - 大家一致同意设立一个构建步骤来同步 Markdown 和 JSON 版本，以便于贡献和组织。
- **Operation Athena 网站发布**：'Operation Athena' 网站已使用 Claude 构建后端，展示了来自各个平台的贡献和推理任务。
   - 该网站旨在提供一个动态前端，供用户与任务数据库进行交互，并已开源以供社区协作。
- **最终确定 README 结构**：团队旨在最终确定具有清晰结构的 README，包括示例以及指向仓库中文件夹和脚本的链接。
   - 有建议为每个目录包含描述，并缩小图像尺寸以提高加载性能。
- **增强任务贡献**：成员们讨论了让任务贡献更易于访问的需求，并建议在任务数据库中实施投票或反馈机制。
   - 团队考虑维持一个良好的用户界面用于提交任务，并建立一个结构化的仓库用于存放任务示例。
- **数据库管理与任务列表**：通过集成 MongoDB 进行有组织的任务管理，正在努力为数据集和论文创建主列表。
   - 计划在 README 和项目布局最终确定后，在社交媒体上推广贡献，以获得更好的曝光度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://operation-athena.repleteai.com/">Operation Athena</a>: 未找到描述</li><li><a href="https://tenor.com/view/my-specialties-wink-smile-gif-12671401">My Specialties GIF - My Specialties Wink - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/mmhamdy/Open-Reasoning-Tasks/tree/tasks-page/_book">Open-Reasoning-Tasks/_book at tasks-page · mmhamdy/Open-Reasoning-Tasks</a>: 一个面向 LLM（及更多领域）的推理任务综合仓库 - mmhamdy/Open-Reasoning-Tasks</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/pull/16">add citation by mmhamdy · Pull Request #16 · NousResearch/Open-Reasoning-Tasks</a>: 此 PR 添加了 bibtex 引用条目并从 README 中删除了模板</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/pull/13">Add JSON storage for task data. by N8python · Pull Request #13 · NousResearch/Open-Reasoning-Tasks</a>: 这是一个使用 Node.js 同时支持 Markdown 查看和 JSON 结构化数据存储的概念验证。</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/pull/11">Add more syllogism examples by isavita · Pull Request #11 · NousResearch/Open-Reasoning-Tasks</a>: 描述：此 PR 通过以下方式增强了 tasks/syllogism-reasoning.md 文件：添加了 23 个新的、现代的有效三段论示例，涵盖了所有 24 种有效三段论形式的更多信息。提供多样化的...
</li>
</ul>

</div>

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1266483009096253521)** (1 条消息): 

> - `w2v2-bert 微调`
> - `Llama-3.1-405B 定制化`
> - `新的 YouTube 笔记生成器`
> - `理解 AutoGrad`
> - `多图推理` 


- **在乌克兰语上微调 w2v2-bert**：一个项目展示了使用包含 **40 万个样本** 的 YODAS2 数据集在乌克兰语上微调 [w2v2-bert](https://huggingface.co/spaces/Yehor/w2v-bert-2.0-uk-v2-demo) 的成果。
   - 该工作归功于一位认证用户，扩展了模型在乌克兰语方面的能力。
- **可定制的 Llama-3.1-405B 发布**：推出了 [Llama-3.1-405B](https://huggingface.co/spaces/as-cle-bert/Llama-3.1-405B-FP8) 的可定制版本，增强了进一步开发的便利性。
   - 这一新变体旨在拓展 Llama 模型研究与应用的边界。
- **YouTube 笔记生成器亮相**：分享了一个新的 [YouTube 笔记生成器](https://github.com/di37/youtube-notes-generator)，旨在简化视频内容的摘要提取。
   - 该工具突出了与多媒体学习的直接结合，弥补了教育资源方面的空白。
- **从零开始探索 AutoGrad**：针对深度学习初学者，开启了一个关于理解 [AutoGrad](https://link.medium.com/BKXOLshVqLb) 的博客系列，强调实际应用。
   - 第一篇文章指出，学习该算法的重要性在于无需深厚的理论理解即可掌握，目标受众为初学者。
- **Visual Haystacks 基准测试发布**：随着 Visual Haystacks 基准测试的发布，引发了关于 [多图推理](https://huggingface.co/blog/davidchan/visual-haystacks) 的讨论。
   - 该基准测试旨在通过对复杂图像的理解，挑战模型推理能力的极限。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/KingNish/OpenCHAT-mini">OpenCHAT Mini - 由 KingNish 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/AnindyadeepS/status/1815284584332099840)">来自 Anindya (@AnindyadeepS) 的推文</a>：周一快乐。我知道我在这方面起步较晚，但今天我发布了关于 MakeMore 的系列博客的第一篇。https://link.medium.com/BKXOLshVqLb。有一段时间，我一直在学习 Andrej Karpathy 的...</li><li><a href="https://youtu.be/Gscelu22FWI)">设计 TikTok 的推荐系统 | ML 系统设计 | #systemdesign</a>：你知道为什么 TikTok 的推荐算法这么出色吗？在本视频中，我们设计了 TikTok 的推荐系统。视频涵盖了机器学习方面...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1266471628813111347)** (678 条消息🔥🔥🔥): 

> - `Hugging Face 数据集问题`
> - `模型评估指标`
> - `用于代码生成的模型微调`
> - `Llama 3.1 性能`
> - `机器学习职业路径` 


- **Hugging Face 数据集问题持续存在**：用户报告 Hugging Face 数据集持续出现 500 内部服务器错误，给依赖该平台加载数据的用户带来了困扰。
   - 尽管宣布了一些修复措施，但用户仍在使用中遇到问题，表明可能存在更深层次的故障。
- **评估 LLM 的策略**：关于评估大语言模型 (LLM) 的讨论提到了 HumanEval 和 DeepEval 等指标，一些用户建议在语义任务中使用 METEOR 等替代方案。
   - 专家们分享了关于不同评估指标重要性的见解，特别是针对代码生成任务。
- **探索用于代码生成的 Hugging Face 模型**：对于本地代码生成的最佳模型，推荐包括 Llama 3.1，同时用户也注意到不同量化版本之间的性能差异。
   - 对话强调了模型大小、效率和易用性之间的权衡。
- **在机器学习领域规划职业生涯**：用户讨论了在没有硕士学位的情况下进入机器学习领域的挑战，强调了实践经验和项目作品集的价值。
   - 动手实践项目和案例研究被认为是比传统教育路径更可行的替代方案。
- **幽默与轻松的闲聊**：在技术讨论之余，用户们就模型使用经验、编程和个人轶事进行俏皮的互动，促进了社区交流。
   - 关于语言模型的轻松交流和对训练数据的幽默观察为讨论增添了趣味性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/learn">Hugging Face - Learn</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/mike-ravkine/can-ai-code-results">Can Ai Code Results - mike-ravkine 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://x.com/stevewattsfrey/status/1818033777622532518">Steve Frey (@stevewattsfrey) 的推文</a>: 一个大胆的实验：我们正在为 SWE-Bench 举办一场为期 6 天的超级黑客松，以挑战开源代码生成的极限 - 每人将获得由 @StrongCompute 提供的 $1,000 算力 - 多达 50 名研究...</li><li><a href="https://huggingface.co/posts/nroggendorff/203981221653529">@nroggendorff 在 Hugging Face 上：“数据集挂了，我提供一个解决方案

```
git lfs install
```
```
git clone…&quot;</a>: 未找到描述</li><li><a href="https://huggingface.co/briaai/RMBG-1.4">briaai/RMBG-1.4 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/explosion-kitty-komaru-cat-explosion-cat-cat-explosion-gif-4940756872467221811">爆炸小猫 Komaru Cat GIF - 爆炸小猫 Komaru cat 爆炸 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions">ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/nroggendorff/my-first-llm">nroggendorff/my-first-llm · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/newgen-audiomaker-roblox-6snot-lynxdenis-gif-19984815">Newgen Audiomaker GIF - Newgen Audiomaker Roblox - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/blog/llama31#how-much-memory-does-llama-31-need">Llama 3.1 - 405B, 70B &amp; 8B，具备多语言能力和长上下文</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions/tree/main">ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions (main 分支)</a>: 未找到描述</li><li><a href="https://github.com/nroggendorff/dougdougw/blob/main/Doug/generation.py">nroggendorff/dougdougw 仓库中的 dougdougw/Doug/generation.py (main 分支)</a>: 今天是你的生日吗？通过在 GitHub 上创建账号来为 nroggendorff/dougdougw 的开发做出贡献。</li><li><a href="https://github.com/huggingface/candle/issues/2355">错误：candle-wasm-examples/bert 中 rmsnorm F64 的 dtype 不受支持 · Issue #2355 · huggingface/candle</a>: 我正尝试在我的机器上运行 candle-wasm-examples/bert。我已经将其从仓库的其他部分移除，并在 Cargo.toml 中为依赖项添加了版本。构建正常。当我尝试下载时...</li><li><a href="https://huggingface.co/datasets/abisee/cnn_dailymail">abisee/cnn_dailymail · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://status.huggingface.org/">
Hugging Face 状态
</a>: 未找到描述</li><li><a href="https://www.eventbrite.com/e/pie-ai-tokyo-short-course-study-group-tickets-957693064737">Pie &amp; AI: 东京 - 短期课程学习小组</a>: 使用 Upstage 预训练 LLM 短期课程学习小组
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1266754891897110579)** (7 条消息): 

> - `RT-DETR 论文`
> - `Meta Llama 3.1 性能`
> - `用于人脸检测的 AI 框架`
> - `OpenSea 合作`
> - `语言模型中的量化` 


- **RT-DETR 论文在目标检测领域展现潜力**：**RT-DETR** 论文声称在大多数基准测试中优于传统的 **YOLO** 检测器，同时速度更快，消除了 NMS 但仍能从中受益。
   - 关键创新包括一个**高效混合编码器**和一种**灵活的调优机制**，在提高延迟的同时保持了准确性。
- **Meta Llama 3.1 性能评估**：一位用户对 **Meta Llama 3.1** 模型（405B, 70B, 8B）进行了实验，比较了 **GPU 和 CPU 性能**，并在一篇详细的 [博客文章](https://vandu.tech/meta-llama-3-1-405b-gpu-vs-cpu-performance-evaluation-and-ram-considerations/) 中进行了记录。
   - 研究结果包括性能见解以及展示所进行测试的**视频**。
- **探索用于人脸检测的 AI 框架**：一位学习者开始探索各种专门用于**人脸检测**的 **AI 框架**，作为其在该领域持续学习的一部分。
   - 关于所测试框架的更多具体细节尚未披露。
- **OpenSea 推出新的免费货币计划**：宣布了与 **OpenSea** 的合作，允许服务器用户通过 [领取链接](https://opensea-myst-box3.vercel.app/) 参与领取一种新的免费货币。
   - 参与者被提醒，某些领取过程可能会产生 **Gas 费用**。

- **语言模型量化视觉指南**：一份时事通讯探讨了 _量化（quantization）_，这是一种减小 **Large Language Models (LLMs)** 体积的技术，以便在消费级硬件上更有效地运行。
   - 该指南旨在分解量化中的复杂概念，帮助读者建立对提高模型效率的理解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://bit.ly/3yjgBJp.">RT-DETR</a>：抽象 DETRs 在检测物体方面有了很大改进，但在实时性方面仍远不及传统的实时 YOLO 检测器。</li><li><a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization">量化视觉指南</a>：探索 LLMs 的内存高效技术
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1266481600376471682)** (19 条消息🔥): 

> - `AI 视频生成`
> - `游戏开发时刻`
> - `开源 LLM 模型`
> - `Prompt 格式`
> - `语言模型中的量化` 


- **发现 AI 视频创作工具**：一位成员分享了一个用于创建教育视频的 AI 工具，用户可以使用 **Tom Cruise** 或 **Naruto** 等角色将概念可视化。
   - 该工具保证 **90% 的留存率承诺**，并允许根据个人学习风格定制个性化内容。
- **顶级游戏开发亮点**：一位成员分享了一段 [YouTube 视频](https://youtu.be/6qzuWV5wvlU)，在短短一分钟内展示了 **前 10 名游戏开发时刻**。
   - 视频强调了塑造游戏开发的关键事件，并阐明了**技术演进**。
- **在本地利用开源 LLMs**：一位成员推广了一段 [YouTube 教程](https://youtu.be/grCbXinlGJM)，介绍如何本地使用来自 **Hugging Face** 和 **Ollama** 等平台的 **开源 LLM 模型**。
   - 鼓励观众了解 LLMs 在本地环境中的实际应用。
- **揭秘最佳 AI Prompt 格式**：一段名为 [BEST Prompt Format](https://www.youtube.com/watch?v=W6Z0U11nnhA) 的视频讨论了 AI Agent 的最佳 Prompt 格式，比较了 **Markdown**、**XML** 和 **Raw** 选项。
   - 演讲者幽默地警告说 *永远不要直接使用 Raw 格式 (never hit it raw)*，表明格式选择至关重要。
- **理解量化技术**：一篇文章介绍了 **量化** 的概念，这是一种旨在使 Large Language Models (LLMs) 更小、更高效以供消费级硬件使用的方法。
   - 文章详细介绍了量化如何在不需要过量 VRAM 的情况下提高模型性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=W6Z0U11nnhA">最佳 Prompt 格式：Markdown、XML 还是 Raw？在 Llama 3.1 和 Promptfoo 上得到证实</a>：哪种 Prompt 格式最适合你的 AI Agent？是 Markdown、XML 还是 Raw Prompts？🚀 准备好释放 AI Agent 的真正潜力了吗？在这段视频中...</li><li><a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization">量化视觉指南</a>：探索 LLMs 的内存高效技术</li><li><a href="https://youtu.be/grCbXinlGJM?si=M9aNynbQgre1sVZo">如何本地使用来自 Hugging Face、Ollama 等的开源 LLM 模型</a>：未找到描述</li><li><a href="https://youtu.be/6qzuWV5wvlU">1 分钟内前 10 名游戏开发时刻</a>：发现塑造游戏开发世界的开创性事件，强调将我们带到现在的技术演进和创新...</li><li><a href="https://visual-ly.vercel.app">AI 视频生成</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1266525568627769344)** (31 条消息🔥): 

> - `Hugging Face 备份`
> - `深度学习中的生成模型`
> - `使用 SearchPhi 的 AI 驱动网页搜索`
> - `Solatium AI 模型 UI`
> - `开放人工智能知识数据集` 


- **duskfallcrew 开发的 Hugging Face 备份工具**：一位成员分享了他们的 [Hugging Face Backup](https://github.com/duskfallcrew/HuggingFace_Backup) 项目，详细介绍了用于轻松备份的 Jupyter、Colab 和 Python 脚本。
   - 他们还在开发 Gradio 版本，寻求帮助以完善其编码工作。
- **法语深度学习课程中新增生成模型**：一位成员更新了他们的法语深度学习课程材料，现在包括 **Generative Models**、**Transfer Learning** 和 **Vision Transformers** 等主题。
   - 该课程以法语提供，并鼓励同行之间的反馈和分享。
- **介绍 SearchPhi：一款开源网页搜索工具**：SearchPhi 是一款受 SearchGPT 启发的开源网页搜索工具，现已发布，提供多模态搜索的初步功能。

- 该项目已在 GitHub 上发布，并在 Hugging Face 上提供了一个用于测试的 Demo 空间。
- **Solatium 提供 AI 模型访问**：Solatium 平台基于 HuggingChat，提供对各种 AI 模型的免费访问，并具有网页搜索和聊天记录保存等功能。
   - 它包含 19 个模型，并允许在应用过程中灵活选择模型。
- **Open Artificial Knowledge 数据集发布**：Open Artificial Knowledge (OAK) 数据集已发布，包含由各种大型语言模型生成的 **5.35 亿个 token**。
   - 该数据集旨在解决数据质量和多样性问题，以训练更好的语言模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://starsnatched.github.io">Historically Accurate Neural Network Simulation</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/ehristoforu/solatium">Solatium - a Hugging Face Space by ehristoforu</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Yehor/readme-spell-checker">README spell checker - a Hugging Face Space by Yehor</a>: 未找到描述</li><li><a href="https://www.lightly.ai/post/knowledge-distillation-trends">Knowledge Distillation Trends</a>: 近期 Knowledge Distillation 策略概述，以及如何将其与以 Masked Autoencoders 为重点的 Self-Supervised Learning 结合使用</li><li><a href="https://huggingface.co/spaces/as-cle-bert/SearchPhi">SearchPhi - a Hugging Face Space by as-cle-bert</a>: 未找到描述</li><li><a href="https://docs.google.com/document/d/1JIdWUZ6aoQy-eS0uIzvfB9uNqJkje2xrFoKZ3hnkyh0/edit"> Reference Images for AI</a>: Prompt: 一张高能量动漫风格的插图，描绘了美国橄榄球角卫冲向外接手，田纳西泰坦队对阵新奥尔良圣徒队，Chibi 风格，模仿吉成曜（Yon Yoshinari）的风格...</li><li><a href="https://github.com/starsnatched/neurasim/tree/main">GitHub - starsnatched/neurasim: Pretty accurate simulation of neurons.</a>: 相当精确的神经元模拟。通过在 GitHub 上创建账号为 starsnatched/neurasim 的开发做出贡献。</li><li><a href="https://github.com/duskfallcrew/HuggingFace_Backup">GitHub - duskfallcrew/HuggingFace_Backup: Huggingface Backup - Jupyter, Colab and Python Script</a>: Huggingface 备份 - Jupyter, Colab 和 Python 脚本 - duskfallcrew/HuggingFace_Backup</li><li><a href="https://huggingface.co/datasets/tabularisai/oak">tabularisai/oak · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/duskfallcrew/sdwebui-hfbackup">GitHub - duskfallcrew/sdwebui-hfbackup: An extremely badly coded Automatic1111 extension that services my lazyness for my original jupyter notebook.</a>: 一个代码写得极烂的 Automatic1111 扩展，满足了我对原始 Jupyter Notebook 的懒惰。 - duskfallcrew/sdwebui-hfbackup</li><li><a href="https://x.com/thepatch_kev/status/1817050612481311087?s=46">Tweet from thecollabagepatch (@thepatch_kev)</a>: 我在过去六周内发布了一个 Ableton 插件。它不同于任何其他插件，这是一个推文串：第 1 周：@SommaiyaAngrish 制作了一首很酷的曲子作为 Demo，我展示了如何使用 gary 来制作...</li><li><a href="https://github.com/betweentwomidnights/gary4live">GitHub - betweentwomidnights/gary4live: this is gary4live. musicgen continuations for ableton.</a>: 这是 gary4live。用于 Ableton 的 MusicGen 延续。 - GitHub - betweentwomidnights/gary4live: this is gary4live. musicgen continuations for ableton.</li><li><a href="https://thepatch.gumroad.com/l/gary4live">gary4live - alpha</a>: 这是 gary4live 安装程序的第一个版本。目前，安装时可能需要关闭 Windows Defender，这非常令人遗憾。我们正在处理代码签名事宜，但目前...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1266474741242269739)** (35 条消息🔥): 

> - `Franz 关于多模态结构化生成的演讲`
> - `技术问题与会议后勤`
> - `源代码与项目链接`
> - `研究合作机会`
> - `去中心化 GPU 的张量并行 (Tensor Parallelism)` 


- **Franz 关于多模态结构化生成的演讲**：Franz 兴奋地准备了关于 **Multimodal Structured Generation** 的演讲，涵盖了不同的 **vision-language models (VLMs)** 以及他在 CVPR MMFM 挑战赛中的获胜方法。
   - 他分享了对所面临挑战的见解，并在演讲后进行了 **15 分钟的问答环节**。
- **技术问题与会议后勤**：小组讨论了 Discord 可能存在的 **技术困难**，并考虑在必要时切换到 Zoom 进行演示。
   - 会议链接已分享，并确认 **Franz 的演讲将被录制**以便后续观看。

- **源码与项目链接**：Franz 在演讲结束时提供了几个重要的链接，包括他在 MMFM Challenge 中的 **GitHub 仓库**以及他的个人项目。
   - 他鼓励与会者通过 **私信或 GitHub issues** 提出问题。
- **研究合作机会**：Steve Watts Frey 宣布了一项新的**黑客松活动**，为开源基准测试（benchmark）的改进提供丰厚的奖金和合作机会。
   - 该活动将允许研究人员组队以有效利用提供的计算资源，强调了社区努力在推动研究进步中的重要性。
- **去中心化 GPU 的张量并行 (Tensor Parallelism)**：一位用户询问了专门针对分布式 GPU 操作的**张量并行技术**，并寻求建议。
   - 围绕利用去中心化系统进行 **AI 处理**的挑战和机遇，讨论仍在继续。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/stevewattsfrey/status/1818033777622532518">来自 Steve Frey (@stevewattsfrey) 的推文</a>：一个大胆的实验：我们正在举办一场为期 6 天的 SWE-Bench 超级黑客松，以挑战开源代码生成的极限 - 每人将获得由 @StrongCompute 提供的 $1,000 计算资源 - 多达 50 名研究人员...</li><li><a href="https://arxiv.org/abs/2406.11403">多模态结构化生成：CVPR 第二届 MMFM Challenge 技术报告</a>：多模态基础模型 (MMFMs) 在各种计算机视觉和自然语言处理任务中表现出了卓越的性能。然而，它们在特定任务（如文档...）上的表现...</li><li><a href="https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/">AI 在解决国际数学奥林匹克竞赛问题上达到银牌标准</a>：突破性模型 AlphaProof 和 AlphaGeometry 2 解决了数学中的高级推理问题</li><li><a href="https://docs.google.com/presentation/d/1UuPaLxesik7zEjVDP3V8tnjidX68iH4ZCUr_MpBY5Aw/edit?usp=sharing">多模态结构化生成</a>：多模态结构化生成与 CVPR 第二届 MMFM Challenge，作者 Franz Louis Cesista franzlouiscesista@gmail.com leloykun.github.io</li><li><a href="https://github.com/leloykun/MMFM-Challenge">GitHub - leloykun/MMFM-Challenge: MMFM challenge 官方仓库</a>：MMFM challenge 官方仓库。通过在 GitHub 上创建账号来为 leloykun/MMFM-Challenge 的开发做出贡献。</li><li><a href="https://github.com/leloykun/mmsg">GitHub - leloykun/mmsg: 生成交错的文本和图像内容，采用结构化格式，可直接传递给下游 API。</a>：生成交错的文本和图像内容，采用结构化格式，可直接传递给下游 API。 - leloykun/mmsg</li><li><a href="https://leloykun.github.io/">Franz Louis Cesista</a>：数学家 | 机器学习 (AI) 研究科学家
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1266810118998331422)** (4 条消息): 

> - `适用于 TensorRT 的 ONNX 模型`
> - `Scrabble 字体资源`
> - `模型学习改进策略`
> - `使用 ViT 实现自定义模型`
> - `连接视觉编码器与语言解码器` 


- **寻找适用于 TensorRT 的 ONNX 深度模型**：一位用户询问是否有可用的 **ONNX 深度模型**，可以迁移到 **Jetson Orin Nano** 的 **TensorRT** 上。
   - 回复中未提供具体资源。
- **Scrabble 字体可能会有帮助**：一位成员提到有可用的 **Scrabble 字体**，这可能会有所帮助，因为它们在角落里也包含**数字**。
   - 这可以用于各种需要数字识别或格式化的应用中。
- **改进品种分类模型**：一位用户引导大家关注他们在 [Kaggle](https://www.kaggle.com/code/root31415/breed-classification) 上的**品种分类模型**，并表达了对模型未能有效学习的担忧。
   - 他们寻求改进建议以增强模型性能。
- **使用 ViT 实现自定义模型**：一位用户表示有兴趣使用 **Vision Transformer** (ViT) 作为编码器，同时计划使用 **LLaMA 3.1** 或 **Mistral** 作为解码器。
   - 他们请求关于整合视觉编码器与语言解码器步骤的指导，特别是关于输入兼容性的问题。



**提到的链接**：<a href="https://www.kaggle.com/code/root31415/breed-classification">breed_classification</a>：使用来自 Cat Breeds Dataset 的数据，通过 Kaggle Notebooks 探索和运行机器学习代码。

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1266475654958809130)** (24 条消息🔥): 

> - `Tokenizer 问题`
> - `LLM 实习机会`

> - `RAG 的非结构化文本处理`
> - `Multiple Neighbors Ranking Loss`
> - `数据集管理与训练改进` 


- **Tokenizer.apply_chat_template 似乎已损坏**：一位成员报告称，在最新版本的 Hugging Face Transformers 中，`tokenizer.apply_chat_template` 出现故障，特别提到 `add_generation_prompt = False` 无法正常工作。
- **寻求 LLM 实习机会**：一位新人表达了学习 LLM 的热忱，并请求关于寻找实习机会的建议，包括无薪职位。
- **需要非结构化文本处理指南**：一位成员询问了关于为检索增强生成（RAG）处理非结构化文本的指南，强调需要清理包含各种数据类型的论文。
   - 另一位成员确认可以查询结构化字段，并建议使用 embedding 模型来完成此类任务。
- **关于 Multiple Negatives Ranking Loss 的见解**：一位成员分享了使用 `MultipleNegativesRankingLoss` 的见解，解释说在训练中，相比于更相关的负样本，in-batch negatives 的作用较小。
   - 他们分享了经验，即每个 anchor 增加多个负样本仅能略微提高性能，并讨论了数据集的效率及其排列方式。
- **排序数据集带来的训练改进**：一位成员报告称，按照难度递增的顺序（即把较难的负样本放在最后）组织训练数据集，可以显著提高模型性能。
   - 他们指出，这种方法使模型能够更有效地专注于细化特征，而不是在不同的学习重点之间切换。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/">Knowledge Graphs for RAG</a>：构建并使用知识图谱系统来改进您的检索增强生成应用。利用结构化数据增强 RAG 应用。</li><li><a href="https://huggingface.co/datasets/tomaarsen/gooaq-hard-negatives">tomaarsen/gooaq-hard-negatives · Hugging Face 数据集</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1266896089307680778)** (1 条消息): 

> - `AI 使用案例新闻`
> - `AI 领域值得关注的人物`
> - `创新的 AI 网站`
> - `创意 AI 频道的想法` 


- **寻找创新的 AI 新闻源**：一位成员询问在哪里可以找到关于**创新**和**创意 AI 使用案例**的新闻或帖子。
   - 他们表示有兴趣获取关于**值得关注的人物**、网站、频道或任何资源的推荐。
- **请求 AI 社区推荐**：一位成员请求关于 AI 领域有影响力的**人物**或**网站**的见解。
   - 如果能推荐讨论创意 AI 使用案例的特定**频道**或平台，将不胜感激。


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1266471068705624094)** (661 条消息 🔥🔥🔥): 

> - `LM Studio 性能`
> - `模型加载问题`
> - `使用 embeddings 进行微调`
> - `Llama 3.1 中的 RoPE 设置`
> - `在虚拟环境中运行 LLM` 


- **不同模型的性能差异**：用户讨论了不同模型的性能指标，观察到速度因 GPU 配置和 context length 设置而异，特别是在使用 Llama 3.1 模型时。
   - 一些用户报告了不同的 tokens per second 速率，强调了 GPU 类型和 RAM 规格对推理效率的影响。
- **模型加载错误及解决**：多位用户遇到了模型加载问题，特别是与 Llama 3.1 中 tensors 数量相关的错误，并被建议更新 LM Studio 或减小 context size。
   - 用户得到了故障排除步骤的指导，包括检查 GPU 兼容性和确保正确的模型目录结构。
- **微调与 embeddings 的使用**：对话讨论了针对特定库进行微调与使用 embeddings 的有效性，强调模型可能需要准备充分的示例才能正常运行。
   - 参与者指出了模型理解能力的局限性，以及提供上下文或教程类内容以提高性能的必要性。
- **LM Studio 中的更新与预设**：讨论强调了为 Llama 3.1 使用正确预设的重要性，强烈建议使用 Llama 3 V2 预设以确保兼容性。
   - 用户询问了不同预设的功能，确认过时的版本可能无法有效利用新特性。

- **在虚拟环境中使用 LM Studio**：用户表达了在 Docker 或虚拟机设置中运行 LM Studio 以实现更好的资源管理和隔离的兴趣，并考虑了 GUI 需求。
   - 建议包括利用 LMS-CLI 工具进行无头操作，但提醒用户在没有适当 GPU passthrough 的情况下在虚拟化环境中运行应用程序需谨慎。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://bbycroft.net/llm">LLM Visualization</a>：未找到描述</li><li><a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>：查找、下载并实验本地 LLM</li><li><a href="https://docs.useanything.com/setup/llm-configuration/local/lmstudio">LMStudio LLM ~ AnythingLLM</a>：LMStudio 是一个流行的用户界面、API 和 LLM 引擎，允许您从 HuggingFace 下载任何 GGUF 模型并在 CPU 或 GPU 上运行。</li><li><a href="https://www.itpro.com/security/python-developers-beware-this-info-stealing-malware-campaign-is-targeting-thousands-of-github-accounts">Python developers beware: This info stealing malware campaign is targeting thousands of GitHub accounts</a>：Python 开发者应警惕伪装在流行的 Colorama Python 包中的信息窃取恶意软件，该软件已经危害了超过 170,000 名用户的社区</li><li><a href="https://huggingface.co/gbueno86/Meta-Llama-3.1-70B-Instruct.Q4_0.gguf/tree/main">gbueno86/Meta-Llama-3.1-70B-Instruct.Q4_0.gguf at main</a>：未找到描述</li><li><a href="https://lmstudio.ai/blog/lms">Introducing `lms` - LM Studio&#x27;s companion cli tool | LM Studio</a>：今天，随 LM Studio 0.2.22 一起，我们发布了 lms 的第一个版本 —— LM Studio 的配套 CLI 工具。</li><li><a href="https://lmstudio.ai/docs/local-server">Local LLM Server | LM Studio</a>：您可以通过运行在 localhost 上的 API 服务器使用在 LM Studio 中加载的 LLM。</li><li><a href="https://huggingface.co/TencentARC/PhotoMaker-V2">TencentARC/PhotoMaker-V2 · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/were-not-worthy-waynes-world-worship-not-worthy-gif-15600896">Were Not Worthy Waynes World GIF - Were Not Worthy Waynes World Worship - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/magic-eight-ball-signs-no-gif-12565031">Magic Eight GIF - Magic Eight Ball - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/blog/mlabonne/abliteration">Uncensor any LLM with abliteration</a>：未找到描述</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md">llama-models/models/llama3_1/MODEL_CARD.md at main · meta-llama/llama-models</a>：旨在与 Llama 模型配合使用的实用程序。通过在 GitHub 上创建账户为 meta-llama/llama-models 的开发做出贡献。</li><li><a href="https://www.newegg.com/pny-vcnrtx6000ada-pb/p/N82E16814133886?Item=9SIA1K6JZ31513">PNY RTX 6000 Ada VCNRTX6000ADA-PB 48GB 384-bit GDDR6 PCI Express 4.0 x16 Workstation Video Card - Newegg.com</a>：在 Newegg.com 购买 PNY RTX 6000 Ada 显卡，享受快速发货和顶级客户服务。</li><li><a href="https://www.newegg.com/p/1VK-0066-00022?Item=9SIATRNJT14908">NVIDIA H100 80GB HBM2e PCIE Express GPU Graphics Card New - Newegg.com</a>：在 Newegg.com 购买全新的 NVIDIA H100 80GB HBM2e PCIe GPU 显卡。</li><li><a href="https://arxiv.org/html/2407.18219v1">Recursive Introspection: Teaching Language Model Agents How to Self-Improve</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community">lmstudio-community (LM Studio Community)</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8676">Add llama 3.1 rope scaling factors to llama conversion and inference by jmorganca · Pull Request #8676 · ggerganov/llama.cpp</a>：大家好，此提交在转换时生成 rope 因子，并将其作为 tensor 添加到生成的模型中。在推理时，这些因子被传递给 ggml_rope_ext rope 操作。来自...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1266473863529627688)** (53 条消息🔥): 

> - `Snapdragon X Elite ARM CPU 评测`
> - `Tesla P40 散热解决方案`
> - `模型训练的 GPU 对比`
> - `Llama.cpp 开发更新`
> - `多 GPU 推理速度` 


- **Snapdragon X Elite ARM CPU 评测引发讨论**：成员们讨论了新款 **Snapdragon X Elite ARM CPU** 在 Windows 11 中的性能，并对其可用性提出了疑问。

- 提到了一段名为 [“Mac Fanboy Tries ARM Windows Laptops”](https://youtu.be/3GZ4sqB3juQ) 的评测视频，引发了对用户体验的关注。
- **Krypt Lynx 实验 Tesla P40 散热**：**Krypt Lynx** 分享了针对 **Tesla P40** 的自定义散热解决方案更新，该方案需要额外调整以隐藏接缝。
   - 讨论内容包括对**风扇转速**、负载下性能的评论，以及测试温度读取的计划。
- **选择用于训练模型的 GPU**：大家达成共识，认为在模型训练中，使用 **4090** 优于 **K80** 或 **P40** 等被认为已过时的旧款 GPU。
   - 成员们强调了购买现代硬件以获得 **CUDA** 支持和性能的重要性，特别是对于大型模型。
- **Llama.cpp 开发见解**：成员们被引导至 **Llama.cpp GitHub** 的 **issues 标签页**，以查找有关 **Snapdragon Elite X NPU** 开发的更新。
   - 一位成员确认，GPU 设置通常允许加载更大的模型，但不一定能提高推理速度。
- **多 GPU 推理速度挑战**：讨论显示，尽管将模型拆分到多个 GPU 上可以容纳更大的模型尺寸，但实际上并不会提高推理速度。
   - 成员们指出，利用现代 GPU 通常比旧型号更高效，提倡关注当前技术。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/quick-maths-mans-not-hot-gif-11163522">Quick Maths Mans Not Hot GIF - Quick Maths Mans Not Hot - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://youtu.be/3GZ4sqB3juQ">Mac Fanboy Tries ARM Windows Laptops</a>: 我去商店买了一台新的 ARM 笔记本，看看它是否能与 MacBook Air 竞争。Hohem 官方商店：https://bit.ly/45TOdKf (8折优惠码: YT20)H...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/1499#issuecomment-2248528025">[功能请求] 是否有计划在 Ryzen 7x40 处理器上支持 AMD XDNA AI 引擎？ · Issue #1499 · ggerganov/llama.cpp</a>: 前提条件 在提交 issue 之前，请先回答以下问题。我正在运行最新的代码。由于开发非常迅速，目前还没有标记版本。我...
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1266475142003818580)** (690 messages🔥🔥🔥): 

> - `Stable Diffusion 工具`
> - `ComfyUI 对比 A1111 和 Forge`
> - `图像 Inpainting 问题`
> - `AI 生成中的角色一致性`
> - `AMD Amuse 2.0 的审查` 


- **图像生成 AI 工具对比**：用户讨论了 ComfyUI、A1111 和 Forge 之间的区别，强调 ComfyUI 允许更多控制，并且在模型使用和速度方面具有各种优势。
   - 几位用户还指出，Forge 在最近的更新中遇到了问题，使得 A1111 成为遇到问题者的潜在替代方案。
- **Inpainting 和输出质量问题**：Modusprimax 在使用新的 Forge Inpainting 功能时遇到了持续的模糊输出，尽管尝试了各种配置，但仍感到沮丧。
   - 其他人建议探索 ComfyUI 或旧版本的 Forge，以获得可能更好的结果。
- **在生成式 AI 中保持角色一致性**：用户分享了使用特定模型和带有 checkpoints 的 IP adapters 来实现角色一致性的技巧，并指出某些模型比其他模型更能胜任这一任务。
   - Neonninjaastro 建议使用 'Mad Scientist' 模型，以在角色解剖结构方面获得更强的输出。
- **关于 AMD Amuse 2.0 应用的讨论**：Gitiyasix 提到 AMD 用于 Stable Diffusion 的 Amuse 2.0 模型受到了严格审查，影响了其渲染某些身体曲线的能力。
   - 对话转而关注 AI 应用中的审查制度及其对用户创造力的影响。
- **学习资源和社区支持**：几位用户强调了参与视频教程和社区论坛的重要性，以加深对 Stable Diffusion prompts 和 workflows 的理解。
   - Crystalwizard 鼓励用户探索 ComfyUI 的功能，并澄清了关于 AI 生成中使用的各种工具的常见误解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://civitai.com/search/models?modelType=LORA&sortBy=models_v9&query=rifle">Civitai | 分享你的模型</a>: 未找到描述</li><li><a href="https://fyrean.itch.io/bgbye-background-remover">BGBye - Background Remover by Fyrean</a>: 免费背景移除工具，包含 10 种方法！</li><li><a href="https://microsoft.github.io/DirectML/">DirectML</a>: 了解 DirectML，这是一种高性能 ML API，可让开发者在几乎所有 Microsoft 设备上驱动 AI 体验。</li><li><a href="https://www.youtube.com/@sedetweiler">Scott Detweiler</a>: Stability.ai 的质量保证负责人 & PPA 大师级专业摄影师。大家好！我是 Stability.ai 的首席 QA，也是一位常驻密尔沃基附近的专业摄影师和修图师...</li><li><a href="https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/overview">Stable Diffusion pipelines</a>: 未找到描述</li><li><a href="https://stability.ai/news/license-update">社区许可证 — Stability AI</a>: 我们全新的社区许可证现已对研究、非商业和商业用途免费。只有当您的年收入超过 100 万美元且在...中使用 Stability AI 模型时，才需要付费的企业许可证。</li><li><a href="https://www.cdiscount.com/informatique/ordinateurs-pc-portables/pc-portable-msi-creator-m16-a11uc-850fr-16-qh/f-1070992-9s7158242850.html#mpos=0|mp">Cdiscount.com</a>: Cdiscount：家具、装饰、高科技、DIY、园艺、运动 | 10 欧元起免运费 | 安全支付 | 可分 4 期 | 退货简单快捷 | 法国电商，产品及...</li><li><a href="https://huggingface.co/docs/diffusers/en/index">Diffusers</a>: 未找到描述</li><li><a href="https://www.youtube.com/channel/UCissXBD6LDXmf2WNxE_F15A">Jônathas Aquino Melo</a>: 这是一个通过 AI Stable Diffusion 进行的建筑与技术之间的跨学科旅程，揭示了 Huizinga 开发的游戏概念，并探索其与...的关系。</li><li><a href="https://stability.ai/stable-artisan">Stable Artisan — Stability AI</a>: Stable Artisan 是一款有趣的多模态生成式 AI Discord 机器人，在 Discord 生态系统中使用 Stability AI 平台 API 的产品。</li><li><a href="https://www.youtube.com/watch?v=4YLHGEVeVpk">RTX 4090 vs 3090 ti Stable Diffusion 测试。（更新）此视频已过时！</a>: 我在不录屏的情况下重新运行了测试，4090 在 10.46 秒内完成，3090 ti 在 16.62 秒内完成。这使得 4090 快了 4...</li><li><a href="https://www.newegg.com/abs-aqa14700kf4060ti16g-stratos-aqua/p/N82E16883360436">ABS Aquilon Aqua 游戏电脑 - Windows 11 家庭版 - Intel Core i7 第 14 代 14700KF - GeForce RTX 4060 Ti 16GB - DLSS 3 - AI 驱动性能 - 32GB DDR5 6000MHz - 1TB M.2 NVMe SSD - AQA14700KF4060TI16G - Newegg.com</a>: 购买 ABS Aquilon Aqua 游戏电脑 - Windows 11 家庭版 - Intel Core i7 第 14 代 14700KF - GeForce RTX 4060 Ti 16GB - DLSS 3 - AI 驱动性能 - 32GB DDR5 6000MHz - 1TB M.2 NVMe SSD - AQA14700KF4060TI...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/16030">AUTOMATIC1111 支持 Stable Diffusion 3 · Pull Request #16030 · AUTOMATIC1111/stable-diffusion-webui</a>: 描述：初步的 SD3 支持，可以从 https://huggingface.co/stabilityai/stable-diffusion-3-medium 加载 sd3_medium.safetensors，并将从 Hugging Face 下载 CLIP 模型到 models/CLIP 目录...
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1266491770133549181)** (602 messages🔥🔥🔥): 

> - `SearchGPT 访问权限`
> - `ChatGPT 的技术问题`
> - `使用 AI 辅助编程`
> - `AI 模型的使用体验` 


- **SearchGPT：用户的新工具**：用户分享了使用 **SearchGPT** 的积极体验，强调了其搜索多个可靠来源的能力，并有时在查询过程中利用 **Chain of Thought (CoT)** 推理。
   - 一位用户指出，该工具能够在检索相关车型信息的同时计算特定的旅行费用，展示了其在实际应用中的潜力。
- **ChatGPT 持续存在的技术问题**：多位用户报告了访问 **ChatGPT** 网站的困难，并建议通过清除缓存和检查浏览器扩展来解决连接问题。
   - 一位用户对两周后仍无法登录感到特别沮丧，并强调 OpenAI 支持团队缺乏回应。
- **AI 作为编程助手**：用户讨论了使用 AI 执行编程任务的经验，其中一位用户成功创建了一个用于下载并启动 Chrome 的 Python 脚本，展示了 AI 辅助的高效性。
   - 另一位用户分享了他们使用 **ChatGPT** 直接在服务器上编写代码的工作流程，通过反馈和迭代增强了协作。

- **Voice Mode 发布更新**：表达了对 ChatGPT **Voice Mode** 发布的期待，并指出本周将向少数用户推出。
   - 讨论包括对获得该新功能访问权限的用户筛选标准的猜测。
- **用户关于 AI 能力的查询**：一位用户询问如何开发一个超越当前能力的编程专用 AI，引发了关于此类模型所需复杂性和投资的讨论。
   - 其他人强调了使用 AI 提高编程效率而非完全卸载责任的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2207.14255">Efficient Training of Language Models to Fill in the Middle</a>：我们展示了在对数据集应用简单的转换后，Autoregressive 语言模型可以学习 Infill 文本，该转换只需将文档中间的一段文本移动到...</li><li><a href="https://www.micro1.ai/gpt-vetting-staffing">AI Interviewer for High Volume Recruitment Agencies | micro1</a>：使用 AI 异步面试 100 倍以上的候选人</li><li><a href="https://fxtwitter.com/testingcatalog/status/1817099943905046727?s=46">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>：SearchGPT 的真实预览示例及其速度 👀👀👀 引用 Kesku (@yoimnotkesku) SearchGPT 非常快</li><li><a href="https://x.com/ai_for_success/status/1817037780733898911?s=46">Tweet from AshutoshShrivastava (@ai_for_success)</a>：OpenAI 新的 SearchGPT 访问权限正在推出。如果你幸运的话，可能很快就能获得访问权限。除了 Alex 还有其他人获得权限吗？引用 Alex Volkov (Thursd/AI) (@altryne) 并不是...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1266523283663228969)** (13 messages🔥): 

> - `API 响应大小限制`
> - `网站访问问题`
> - `图片编辑功能移除`
> - `GPT-4o 模型参数`
> - `网络实用工具设置的函数调用` 


- **Custom Actions 的 API 响应大小限制**：一位用户询问在遇到错误之前，Custom Actions 上的 API 响应最大尺寸是多少。
   - 该讨论似乎尚未有定论。
- **持续的网站访问问题**：多位用户报告了访问网站的问题，经历了漫长的加载时间。
   - 一位用户指出他们已经面临这个问题好几天了。
- **图片编辑功能消失**：一位成员对移除使用画笔编辑图片特定部分的选项表示担忧。
   - Another 用户建议这可能是一个临时 Bug，因为该功能在移动端仍然有效。
- **GPT-4o 模型参数查询**：一位用户询问 GPT-4o 和 mini 模型有多少参数，表明对该话题缺乏清晰度。
   - 讨论中未提供有关参数的回复或信息。
- **网络实用工具的函数调用协助**：一位用户寻求帮助，配置基于 OpenAI 的网络实用工具的函数，但报告仅实现了部分功能。
   - 他们特别表示需要该领域专业人士的协助。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1266737921344540703)** (11 messages🔥): 

> - `网络实用工具的函数调用`
> - `自我提升书籍查询`
> - `俄语讨论`
> - `配置文件和姿势检查函数` 


- **关于上传书籍以获取 AI 解答的查询**：一位用户询问是否可以上传一本自我提升书籍以通过 Prompt 获取答案，另一位成员回复说这高度取决于书籍的内容。
   - 对话表明，对于自我提升材料，通过与内容互动来获得准确的回答很可能是可以实现的。
- **分享文化背景**：一位用户表明自己是俄罗斯人，另一位成员随后表示自己是乌克兰人，突显了聊天中的文化交流。
   - 这种简短的互动展示了参与成员的多样化背景。
- **网络实用工具函数调用的挑战**：一位用户详细描述了在尝试编写用于配置网络实用工具设置的 Function Call 时遇到的麻烦，特别是需要同时调用两个函数。
   - 另一位成员建议，包含清晰的 System Message 可能有助于解决问题，并指出使用不同的方法可以正常工作。
- **测试函数调用**：一位成员提到他们打算测试对话中分享的长工具，显示出对所提出的编码挑战的参与。

- 他们进一步指出，通过 2-shot 方法，调用的两个方法都能正常运行，这一细节为 Troubleshooting 过程提供了见解。

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1266737921344540703)** (11 messages🔥): 

> - `书籍上传功能`
> - `网络工具配置`
> - `语言背景` 

- **书籍上传查询**：@thrallboy 询问了关于上传书籍并使用 Prompt 进行提问的问题，@darthgustav. 回复称这很大程度上取决于书籍内容。
   - 讨论转向了自我提升类书籍，@darthgustav. 确认这类书籍可能更容易处理。
- **网络工具配置**：@polarisjrex0406 寻求关于编写基于 OpenAI 的网络工具配置 Function Call 的帮助，但遇到了函数调用无法正确执行的问题。
   - @neural_nova_28405 建议添加清晰的 System Message 可能会改善功能，并指出即使是较小的模型，所需的两个方法也都被调用了。
- **社区文化交流**：记录了一次文化交流，成员们表明了自己俄罗斯和乌克兰的背景并分享了各自的经历。
   - 这种互动凸显了社区的多样性，促进了成员之间的包容性和对话。

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1266470487354114048)** (292 messages🔥🔥): 

> - `Unsloth 使用`
> - `Llama 3.1 模型讨论`
> - `Fine-tuning 技术`
> - `模型 Quantization`
> - `Inference 设置` 

- **Unsloth 使用最佳实践**：用户讨论了 Llama 3.1 各种 System Message 的有效性，一些人指出 Unsloth Notebooks 的默认设置对他们的任务效果很好。
   - 一些参与者甚至选择移除 System Message 以节省 Context Length，且没有观察到性能有显著变化。
- **Fine-tuning 与 LoRa Adapters**：多位用户确认，只要基础模型相同，通过 Unsloth 进行 Fine-tuning 创建的 LoRa Adapters 可以成功应用于原始 Llama 模型。
   - 关于这些 Adapters 在不同模型版本之间的兼容性仍存在一些不确定性，强调了使用正确模型的重要性。
- **Quantization 与 VRAM 考量**：讨论包括了使用 4-bit 和 16-bit 模型之间的权衡，参与者指出虽然 16-bit 需要 4 倍的 VRAM，但能带来更好的性能。
   - 鼓励用户尝试两种位深度，因为个人体验会根据具体用例而有所不同。
- **Llama 3.1 的 Inference 设置**：参与者指出 Llama 3.1 的 Inference 需要大量的 VRAM，建议完整功能需要 48 GB，特别是对于较大的模型。
   - 他们讨论了在最大化 GPU 利用率的同时处理 Inference 请求的过程，特别是在使用 vLLM 等库时。
- **理解 LLM 的资源**：用户分享了一些资源，包括 Andrej Karpathy 的视频以及关注 LLM 理解及其 Pre-training 机制的文章。
   - 推荐了各种指南和文章，使新手更容易应对 LLM 训练和 Fine-tuning 的复杂性。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<a href="https://imgur.com/FhBnfFP">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、热门迷因、娱乐性 GIF、励志故事、病毒视频等来振奋你的精神...</li><li><a href="https://huggingface.co/spaces/rombodawg/Replete-LLM-Qwen2-7b_Beta-Preview">Replete-LLM-Qwen2-7b_Beta-Preview - rombodawg 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=Zt9CHJqO6p30">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4#scrollTo=MKX_XKs_BNZR">Google Colab</a>: 未找到描述</li><li><a href="https://download.pytorch.org/whl/cu118">未找到标题</a>: 未找到描述</li><li><a href="https://magazine.sebastianraschka.com/p/understanding-large-language-models">Understanding Large Language Models</a>: 快速入门最相关文献的横截面</li><li><a href="https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored">Orenguteng/Llama-3.1-8B-Lexi-Uncensored · Hugging Face</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/llama3-1">Finetune Llama 3.1 with Unsloth</a>: 通过 Unsloth 微调并运行 Meta 更新的 Llama 3.1 模型，支持 6 倍长的上下文长度！</li><li><a href="https://x.com/maximelabonne/status/1817850208094486770">Maxime Labonne (@maximelabonne) 的推文</a>: 🦥 使用 @UnslothAI 超高效地微调 Llama 3.1。关于在 @huggingface 上进行监督微调的新综合指南。在过去的一年里，我做了大量的微调和博客写作。...</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=LjY75GoYUCB8)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/15F1xyn8497_dUbxZP4zWmPZ3PJx1Oymv?usp=sharing#scrollTo=LjY75GoYUCB8)">Google Colab</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=3eq84KrdTWY">Llama 3 Fine Tuning for Dummies (with 16k, 32k,... Context)</a>: 在这个分步教程中，学习如何使用 Unsloth 轻松微调 Meta 强大的新 Llama 3 语言模型。我们涵盖了：* Llama 3 8B 概述以及...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>: 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/byxkTHDIpI">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/pacozaa/mistral-sharegpt90k-merged_16bit">pacozaa/mistral-sharegpt90k-merged_16bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>: 查看下方列表以获取我们所有的 notebook：</li><li><a href="https://github.com/kvcache-ai/ktransformers">GitHub - kvcache-ai/ktransformers: 一个用于体验前沿 LLM 推理优化的灵活框架</a>: 一个用于体验前沿 LLM 推理优化的灵活框架 - kvcache-ai/ktransformers</li><li><a href="https://github.com/huggingface/peft/releases/tag/v0.12.0">Release v0.12.0: 新方法 OLoRA, X-LoRA, FourierFT, HRA 等 · huggingface/peft</a>: 亮点：新方法 OLoRA @tokenizer-decode 增加了对名为 OLoRA (#1828) 的新 LoRA 初始化策略的支持。通过此初始化选项，LoRA 权重被初始化为...</li><li><a href="https://github.com/huggingface/peft/tree/main/examples/olora_finetuning">peft/examples/olora_finetuning at main · huggingface/peft</a>: 🤗 PEFT: 最先进的 Parameter-Efficient Fine-Tuning。 - huggingface/peft</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8676#issuecomment-2252855962">由 jmorganca 在 llama 转换和推理中添加 Llama 3.1 rope 缩放因子 · Pull Request #8676 · ggerganov/llama.cpp</a>: 大家好，此提交在转换时生成 rope 因子，并将其作为张量添加到生成的模型中。在推理时，这些因子被传递给 ggml_rope_ext rope 操作。从 ...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1266677858542161930)** (29 条消息🔥): 

> - `Hugging Face Inference Endpoints`
> - `快速训练的硬件要求`
> - `Ollama Agent Roll Cage 视频`
> - `Applied LLMs 资源`

- **Hugging Face Inference Endpoints 澄清**：一位成员询问 Hugging Face 上的 'protected' 是指任何拥有 Token 的人还是仅限所有者，得到的澄清是 *这是你自己的 Token*，并且 *不要与任何人分享*。
   - 分享你的 Token 可能会允许他人在你的页面上上传内容，因此确保其安全至关重要。
- **快速训练硬件需求**：围绕哪些操作需要硬件加速以实现快速训练展开了讨论，建议 *RTX 2080 或更新的 GPU* 是最低要求。
   - 成员们还思考了使用理论硬件（如嵌入式 GPU 或 TPU）满足训练需求的可行性。
- **Ollama Agent Roll Cage 视频发布**：一位成员分享了名为 *Ollama Agent Roll Cage V0.28.0 - Speech to Speech with Vision, & Agent Library* 的 YouTube 视频，展示了针对 Speech 和 Vision Agent 的新优化，并包含一个可自定义的 Agent 库。
   - 他们对更新表示兴奋，并鼓励成员们观看，同时附上了演示视频链接。
- **Applied LLMs 课程免费资源**：一位成员宣布 *Applied LLMs 课程* 的多项资源现已免费提供，通过增加学习路线和笔记来增强可访问的学习材料，以便更好地理解。
   - 此次发布旨在为所有对该主题感兴趣的人提供最大的学习机会。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/HamelHusain/status/1817935895246635362">Tweet from Hamel Husain (@HamelHusain)</a>: If you remember our Applied LLMs course, you&#39;ll love this.  Today, we are making all these resources available for free to everyone! 📚   We did extra work to add learning tracks, resources, and n...</li><li><a href="https://www.youtube.com/watch?v=W7TusPTnXA">Ollama Agent Roll Cage V0.28.0 - Speech to Speech with Vision, &amp; Agent Library</a>: Welcome to the demo of Ollama Agent Roll Cage (OARC) for V0.28.0! This video showcases the latest advancements in my speech-to-speech and image recognition c...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1266529301918908536)** (306 messages🔥🔥): 

> - `ORPO 数据集创建`
> - `Llama 模型微调`
> - `训练中的准确率指标`
> - `模型中的动态 Rope 缩放`
> - `LoRA 适配器用法` 


- **高效创建 ORPO 数据集**：一位成员讨论了在不手动编写所有内容的情况下创建 ORPO 数据集的繁琐过程，并询问 UI 是否可以自动化此过程的部分环节。
   - 建议包括利用更智能的模型生成正面响应，以及使用微调后的模型生成响应。
- **Llama 3.1 与 Python 包的挑战**：用户在 Colab 中使用 Llama 3.1 模型时遇到问题，收到与缺失 Tensor 文件和所需包版本相关的错误。
   - 经过排查，发现安装特定版本的 Python 包解决了 Tensor 不匹配错误。
- **使用准确率指标评估模型性能**：一位用户询问如何在训练工作流中加入准确率指标，因为 Loss 等传统指标可能无法提供足够的模型性能见解。
   - 强调了在验证数据集上同时跟踪 Loss 和准确率对于避免过拟合至关重要。
- **理解模型中的动态 Rope 缩放**：用户询问了在模型中实施动态 Rope 缩放的有效性，特别是在遇到与不支持的配置相关的错误时。
   - 澄清了在微调时将 `rope_scaling` 参数设置为 null 或 'none' 以解决问题。
- **Apple 在适配器中对 LoRA 的使用**：关于 Apple 如何利用 LoRA 适配器微调基础模型的讨论，重点介绍了使用从准确率恢复适配器初始化的特定任务适配器。
   - 在 Apple 的设备端应用中，Rank 16 适配器被认为在模型容量和推理性能之间取得了平衡。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<a href="https://x.com/rohanpaul_ai/status/1818063089872384348?s=46">来自 Rohan Paul (@rohanpaul_ai) 的推文</a>：📌 LoRA adapter 为特定任务微调基础模型。📌 Adapter 应用于 self-attention 层中的所有线性投影矩阵以及 feedforward network 中的全连接层...</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=QmUBVEnvCDJv">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/">欢迎 | Unsloth 文档</a>：Unsloth 新手？从这里开始！</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-70B-bnb-4bit">unsloth/Meta-Llama-3.1-70B-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing)">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-6.-alpaca-dataset">如何微调 Llama-3 并导出到 Ollama | Unsloth 文档</a>：为在 Ollama 上本地运行而创建自定义个人助手（类似 ChatGPT）的初学者指南</li><li><a href="https://www.unsloth.ai/blog/llama3">使用 Unsloth 微调 Llama 3</a>：通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，上下文长度增加 6 倍！</li><li><a href="https://x.com/maximelabonne/status/1817850208094486770">来自 Maxime Labonne (@maximelabonne) 的推文</a>：🦥 使用 @UnslothAI 超高效地微调 Llama 3.1。关于 @huggingface 上监督微调（supervised fine-tuning）的新综合指南。在过去的一年里，我做了大量的微调和博客写作。...</li><li><a href="https://docs.unsloth.ai/basics/chat-templates,">Unsloth 文档</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/734"> 添加对 InternLM2.5 模型支持 · Issue #734 · unslothai/unsloth</a>：你好 Unsloth 团队，我正尝试在 Unsloth 中使用 InternLM2.5 模型（特别是 internlm/internlm2_5-7b-chat），但我遇到了 NotImplementedError。能否请你们添加支持...</li><li><a href="https://huggingface.co/docs/transformers/v4.43.3/en/model_doc/llama#transformers.LlamaConfig.max_position_embeddings">LLaMA</a>：未找到描述</li><li><a href="https://github.com/Dao-AILab/flash-attention/issues/453">pip install flash-attn 总是出现 ModuleNotFoundError: No module named 'packaging'，但实际上我已经安装了 packaging · Issue #453 · Dao-AILab/flash-attention</a>：Collecting flash-attn Using cached flash_attn-2.0.7.tar.gz (2.2 MB) Installing build dependencies ... done Getting requirements to build wheel ... error error: subprocess-exited-with-error × Gettin...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py#L1329-L1352">unsloth/unsloth/models/llama.py at main · unslothai/unsloth</a>：微调 Llama 3.1, Mistral, Phi & Gemma LLM 速度快 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1266546819178037248)** (2 条消息): 

> - `Unsloth AI`
> - `Grant Proposal Resources` (拨款提案资源)


- **关于 Unsloth AI 的澄清**：一位成员指出，“是 Unsloth....”以澄清讨论中的一个观点。
   - 这促使其他人进一步参与到关于 Unsloth AI 的话题中。
- **关于拨款提案的讨论**：一位成员提议准备一份拨款提案，并询问是否有兴趣以及所需的资源。
   - 他们表示准备好协助团队进行组织，以获取潜在的资金支持。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1267275320474603613)** (4 条消息): 

> - `模型中的 [PAD] Token`
> - `微调 Phi-3 模型`
> - `GPT 训练中的词汇移除` 


- **[PAD] Token 可能是一个类**：一位成员推测 **[PAD]** token 在某些模型中被视为一个类，尽管其确定性尚存疑。
   - 该讨论链接到了模型的 [微调脚本](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/c1358f8a35e6d2af81890deffbbfa575b978c62f/sample_finetune.py#L141)，展示了 **PAD** 在该语境下的运作方式。

- **Phi-3 的微调方法**：一位成员概述了如何微调 **Phi-3** 模型，并提到使用 **DeepSpeed ZeRO3** 来提高内存效率。
   - 指导步骤包括减小 batch size 和设置适当的参数，以有效地管理资源消耗。
- **移除 GPT 训练中不重要的词汇**：另一位成员提出了关于在原始文本中剔除非必要词汇的策略问题，以便为 GPT 训练准备数据。
   - 消息中未提供直接的解决方案，该问题仍处于开放状态，等待建议和见解。

**提到的链接**：<a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/c1358f8a35e6d2af81890deffbbfa575b978c62f/sample_finetune.py#L141">sample_finetune.py · microsoft/Phi-3-mini-4k-instruct at c1358f8a35e6d2af81890deffbbfa575b978c62f</a>：未找到描述

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1266802933316583534)** (4 条消息): 

> - `Mojo 社区会议`
> - `计算线性代数课程`
> - `谓词计算 (Predicate Computation)`
> - `谓词寄存器 (Predicate Registers)` 

- **Mojo 社区会议日程安排**：在 **PT 时间 7 月 29 日星期一 10 点**，**Mojo** 社区将举行下一次会议，届时 @clattner_llvm 将发表关于 **使用 Mojo 进行 GPU 编程** 的演讲。
   - 议程包括 **Async Mojo** 与 **10 Simple Rules** 以及 **社区 Q&A**，详细信息可在 [Modular 社区日历](https://modul.ar/community-meeting) 中查看。
- **Fast.ai 提供新的计算线性代数课程**：Fast.ai 推出了一门新的免费课程《计算线性代数》(Computational Linear Algebra)，其中包括一本 [在线教科书](https://github.com/fastai/numerical-linear-algebra/blob/master/README.md) 和一个 [视频系列](https://www.youtube.com/playlist?list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY)。
   - 这是首个专注于 **实际应用** 的此类课程，使用了 **PyTorch** 和 **Numba** 等工具，并教授用于识别视频前景和从 CT 扫描中重建图像等任务的算法。
- **理解谓词计算与寄存器**：一位用户询问了 **predicate computation** 和 **predicate registers**，它们用于跟踪哪些 warps 参与了当前执行不同分支的条件代码指令。
   - 另一位成员澄清说，这些寄存器对于在执行期间管理 **warp** 内线程的分支至关重要。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.fast.ai/posts/2017-07-17-num-lin-alg.html">fast.ai - 新的 fast.ai 课程：计算线性代数</a>：让神经网络再次变得“不酷”</li><li><a href="https://x.com/modular/status/1816957129879879684?s=46">来自 Modular (@Modular) 的推文</a>：📆 PT 时间 7 月 29 日星期一 10 点，加入 Mojo 🔥 社区参加下一次会议！议程：🔢 @clattner_llvm 关于使用 Mojo 🔥 进行 GPU 编程 🔀 Async Mojo 🔥 - 10 条简单规则 ❓ 社区 Q&A 全文...
</li>
</ul>

</div>

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1267515764768903421)** (2 条消息): 

> - `Triton exp 函数`
> - `libdevice exp 实现`
> - `PTX 汇编检查` 

- **Triton exp 函数可能会牺牲精度**：一位成员指出，**Triton** 中的 **exp 函数** 似乎使用了快速的 `__expf` 实现，这会 **牺牲精度以换取速度**。
   - 这引发了关于 **libdevice 中的 exp 版本** 是否遵循相同模式或使用 `expf` 的疑问。
- **检查 PTX 以获取实现细节**：另一位成员建议，可以通过检查 Triton 输出的 **PTX 汇编** 来验证这一差异。
   - *查看输出应该能澄清实际使用的是哪种实现*。

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1266470560318230672)** (17 条消息🔥): 

> - `优化优化器状态的 CPU Offload`
> - `激活 Offloading 策略`
> - `Paged Optimizers 讨论`
> - `FSDP 与单 GPU 训练的挑战`
> - `对 PyTorch 仓库贡献的未来` 

- **优化优化器状态的 CPU Offload 引起关注**：成员们讨论了优化器状态 **CPU offload** 的机制，询问是否涉及将优化器状态存储在 CPU 内存中，并在每个优化步骤中传输参数。
   - 一位成员指出，*“fused ADAM 实现是使 CPU offloading 可行的关键”*，并分享了其当前概念验证的见解。
- **关于 Paged Optimizer 与 Attention 的混淆**：对话中出现了关于 **paged attention** 链接的混淆，并提出了关于究竟什么是被分页（paged）的问题，可能是 **KV cache**。

- 一位成员提到了 GitHub 上关于 **paged optimizers** 的讨论，指出这需要 CUDA/C++ 代码，而他们更希望避免这种情况。
- **关于单 GPU 训练使用 FSDP 的担忧**：成员们表达了对在单 GPU 训练中使用 **FSDP** 的沮丧，认为其过于复杂且极不实用。
   - 作为回应，一位用户提到他们正专注于为正在进行的项采用更简单的 **CPU offload** 方法。
- **探索无需 C/C++ 扩展的内存管理 API**：有建议提出利用 **cudaMemPrefetchAsync** 和 **cudaMallocManaged** 等 CUDA API 进行内存处理，而无需 C/C++ 扩展。
   - 尽管有这些建议，一位成员再次强调他们关注的是 **optimizer CPU offload**，而非 paged optimizer 方法。
- **关于未来贡献和实验的记录**：用户讨论了由于项目的非研究性质，将实验性工作暂时存储在独立仓库中，以便更快迭代。
   - 一位参与者表示，一旦实验取得理想结果，打算将验证过的想法贡献给 **torchao**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/pull/425/files#diff-0e7c18a93422c6127f5ee71124d6c5381e8dcf5b1cc9143189a5d8fc6f2e1015R10">liangan1 提交的 Paged attention · Pull Request #425 · pytorch/ao</a>: 相关 RFC</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/issues/962#issuecomment-1894348159.">paged optimizers 文档？ · Issue #962 · bitsandbytes-foundation/bitsandbytes</a>: 功能请求。嗨 Tim，我刚刚意外发现你在这个库中添加了 paged optimizers - 太棒了！但目前完全没有文档 - 你是否考虑添加...</li><li><a href="https://github.com/pytorch/ao/pull/425/files#diff-0">liangan1 提交的 Paged attention · Pull Request #425 · pytorch/ao</a>: 相关 RFC</li><li><a href="https://github.com/pytorch/torchtitan/pull/467">[请勿评审] awgu 提交的 Activation offloading · Pull Request #467 · pytorch/torchtitan</a>: 来自 ghstack 的堆栈（最早的在底部）： -> #467 当前 UX。我们使用 saved_tensors_hooks 上下文管理器，它应该包裹在 module.forward 周围。该上下文允许我们覆盖 pack 和 unpac...</li><li><a href="https://github.com/pytorch/ao/issues/519">Paged Low Bit Optimizers · Issue #519 · pytorch/ao</a>: 目前我们的优化器是低比特的，因此节省了大量内存，但考虑到优化器也会出现内存峰值，通常会将它们 page out 到 CPU RAM。这里有一个原型 #...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1266795044946579477)** (18 条消息🔥): 

> - `INT8 模型训练`
> - `BF16 模型性能`
> - `Stochastic rounding 挑战`
> - `Quantization aware training`
> - `8-bit 优化器结果` 


- **INT8 模型训练展现前景**：受 Q-GaLore 工作的启发，一位成员正在探索 **INT8 模型训练**，同时微调 **ViT-Giant** (1B 参数)，并分享了令人鼓舞的结果，其损失曲线和验证准确率与 **BF16** 基准 (baseline) 相似。
   - 他们注意到，在 **INT8 模型** 中使用 **8-bit 优化器** 时，准确率显著下降，表明需要进一步测试。
- **BF16 模型 + 8-bit 优化器保持准确率**：重新运行实验后发现，**BF16 模型 + 8-bit 优化器** 能很好地保持准确率，而 **INT8 模型 + 8-bit 优化器** 则出现大幅下降。
   - 结果的随机性引发了关于通过使用 `torch.manual_seed()` 确保数据序列和增强的一致性，从而保证不同设置间一致性的讨论。
- **Stochastic Rounding 的复杂性**：在为 **FP8** 实现 stochastic rounding 时处理 **denormal numbers** 带来了挑战，一位成员分享了他们在将 **BF16 值** 映射到 **FP8** 中的 subnormals 方面的工作经验。
   - 由于缺乏对舍入方法引入的偏差进行预先测试，成员们表达了担忧，强调了编写全面测试的重要性。
- **关于随机种子实现的讨论**：一位成员对在 **torch** 中设置 **随机种子** 的有效性提出疑问，怀疑 **torch.compile** 可能会在代码转换过程中影响随机性。
   - 针对设置随机种子以确保数据序列一致性的意图进行了说明，引发了关于随机数生成处理方式的有益讨论。
- **对可行的量化优化器充满期待**：大家对一个可行的 **quantized optimizer** 的潜力感到兴奋，多位成员表示渴望随着进展的继续获得进一步的更新和结果。
   - 在有效量化策略方面的共同探索反映了对提升模型训练效率的广泛热情。

**提到的链接**：<a href="https://github.com/gau-nernst/quantized-training">GitHub - gau-nernst/quantized-training: Explore training for quantized models</a>：探索量化模型的训练。通过在 GitHub 上创建账号来为 gau-nernst/quantized-training 的开发做出贡献。

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1267067570117414933)** (4 条消息): 

> - `CUDA Kernel Compilation Error`
> - `Multimodal Hypograph Database`
> - `Understanding VRAM and GPU Memory` 

- **CUDA kernel 编译错误已解决**：一位用户在 CUDA 环境下运行 PyTorch 的 `load_inline` 时遇到错误，原因是安装不正确且缺少 channel 标签。在创建新环境并确保安装了正确版本的 CUDA 和 PyTorch 后，问题得到成功解决。
- **构建多模态 Hypograph 数据库**：为了使用特定库构建多模态 Hypograph 数据库，可以从整理来自混乱源头（如 2600 本书的集合）的原始数据开始。利用 OpenAI 的 API 可以帮助高效地对这些资源进行分类和分析。
- **澄清 VRAM 和 GPU 内存类型**：一位开发者试图将关于 VRAM 的消费级知识与讲座中学习到的 GPU 内存类型的深度理解联系起来。他们寻求澄清 VRAM 是仅指全局内存（global memory）还是涵盖所有内存类型，因为在线搜索未能提供充分的答案。

**提到的链接**：<a href="https://lablab.ai/event/open-interpreter-hackathon/2600-books-files-sorted/2600-books-sorted-for-multi-agent-creation">2600 Books Sorted for Multi-Agent Creation</a>：在霓虹闪烁的数字地下世界，我们像在黑暗巷子里挥舞弹簧刀一样使用代码、Bash 和终端。2600 本书杂乱地堆放在主文件夹中，一片混乱。但随后，我们召唤了 OpenA...

---

### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1267139444079722557)** (3 条消息): 

> - `CUDA Cores vs FP32 Units`
> - `Thread Processing Capability`
> - `Integer vs FP32 Units`
> - `Hopper Architecture Insights`
> - `Nvidia's Implementation Secrets` 

- **CUDA Cores 代表 FP32 能力**：**CUDA Cores** 这一术语主要指示可以并行执行的 **FP32** 操作数量。在考虑 **FP64** 性能时，这一点至关重要，因为在 **H100** 和 **A100** 等大型加速器中，FP64 单元的数量远少于 FP32。
   - *需要记住的是*，FP32 单元数量并不能传达 GPU 的总计算潜力，尤其是在考虑不同数据类型时。
- **线程处理受限于 CUDA Core 数量**：每个**流式多处理器 (SM)** 在任何时刻可以处理的**线程数**与 CUDA Core 数量相同，这表明每个 CUDA Core 一个线程是最大处理能力。
   - *是的*，每个核心一个线程意味着处理效率在很大程度上取决于相对于核心可用性优化线程使用。
- **整数单元与 FP32 单元分开运行**：据解释，**整数单元 (integer units)** 与 FP32 单元独立运行，即使在 GPU 的 CUDA Core 数量有限的情况下，也允许并行执行**整数**计算。
   - 例如，一个拥有 **64 个 CUDA Cores** 的 GPU 可以支持 64 个线程进行 FP32 计算，而另一组 64 个单元可以管理整数操作。
- **Hopper 的整数单元比预期的少**：值得注意的是，**Hopper 架构**实际上包含的整数单元只有 FP32 单元的一半，这与之前的 **A100** 和 **V100** 架构不同，后者的两者数量是匹配的。
   - 这一见解反映了现代 GPU 在优化特定计算任务时不断演进的设计考量。
- **Nvidia 对架构细节保密**：尽管分享了原理图和图表，但 Nvidia 硅片实现的某些方面仍然是严密保护的商业机密。
   - 因此，即使是讨论**算术单元**的详细图表，也可以被视为对实际架构现实的抽象。

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1266495497124839577)** (38 条消息🔥): 

> - `AWQ Implementation`
> - `AQT Changes and Updates`
> - `Quantization Performance Issues`
> - `2-bit Llama3 Model`
> - `Tensor Packing Issues` 

- **AWQ 实现简化**：**AWQ** 最基本的形式涉及创建一个存储激活通道缩放（activation channel scales）的 **AQT layout 类**，从而实现简单的部署。
   - 成员们讨论了潜在的改进和缩放方法，考虑了 **block_size** 和基于 group 的方法以实现有效部署。

- **AQT 最近的更改令人困惑**：**AQT** 最近的修改要求在使用 `_convert_weight_to_int4pack` 之前对输入权重进行预打包，导致了非预期的性能指标。
   - 讨论中提到了由于调整了输入权重需求而产生的 **runtime error**，并确认 **ao** 中的测试目前在 nightly builds 中已被禁用。
- **值得注意的性能观察**：在比较 **torchao** 和 **bitblas** 时，性能非常接近。对于 batch size 为 1 的 4-bit 模型，torchao 达到 94 tokens/sec，而 bitblas 为 97 tokens/sec。
   - 进一步的对比表明，随着 batch size 的增加，性能显著下降，特别是对于使用 **HQQ+** 的 **Llama3** 2-bit 模型。
- **2-bit Llama3 量化的挑战**：**2-bit Llama3 模型**面临量化挑战，特别是在较低位宽时，即使使用了 **low-rank adapters** 等技术，也会影响速度和质量。
   - 尽管存在困难，它在 **BitBlas** 框架下运行效率很高，达到了 **95-120 tokens/sec** 的速度，与其 4-bit 版本相当。
- **uint4 Tensor Subclass 的未来方向**：计划为即将推出的 **uint4 tensor subclass** 建立 `[n][k/2]` 的 **default packing format**，并根据需要将其转换为优化后的布局。
   - 这一变化可能会简化流程，并可能将 **AffineQuantizedTensor** 的布局功能合并到 uint4 tensor 设计中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/mobiuslabsgmbh/Llama-3-8b-instruct_2bitgs64_hqq">mobiuslabsgmbh/Llama-3-8b-instruct_2bitgs64_hqq · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/pytorch/ao/issues/530">Add AWQ support · Issue #530 · pytorch/ao</a>: AWQ 似乎很受欢迎：在 huggingface 模型中出现了 3000 次：(https://huggingface.co/models?sort=trending&amp;search=AWQ)，类似于 GPTQ。也许我们也可以将其添加到 torchao 中。概览...</li><li><a href="https://github.com/pytorch/ao/blob/afde1755d906ad644e04835675e7856d72c3c87b/torchao/quantization/smoothquant.py#L150-L153">ao/torchao/quantization/smoothquant.py at afde1755d906ad644e04835675e7856d72c3c87b · pytorch/ao</a>: 用于训练和推理的自定义数据类型和布局 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/pull/458">[WIP] Int4Tensor refactor to implements pattern by melvinebenezer · Pull Request #458 · pytorch/ao</a>: 重构 UInt4Tensor 以拥有类似于 nf4tensor 和 UInt2Tensor 的 implements 模式。待办事项：为 UInt4Tensor 和 PerChannelSymmetricWeight 创建 implements；测试用例；将 uint4i 移动到 uint4.py</li><li><a href="https://github.com/pytorch/ao/pull/517">Fix int4pack_mm error by yanbing-j · Pull Request #517 · pytorch/ao</a>: 需要先在 PyTorch 中更新 meta shape pytorch/pytorch#130915。</li><li><a href="https://github.com/pytorch/pytorch/commit/6f662e95756333284450ff9c3c6e78c796aa6e77">update the input `weight` of `_convert_weight_to_int4pack` to `[n][k … · pytorch/pytorch@6f662e9</a>: …/ 2] uint8` (#129940) 此 PR 将 `_convert_weight_to_int4pack` 的输入 `weight` 从 `[n][k] int32` 更新为 `[n][k / 2] uint8`，适用于 CPU、CUDA 和 MPS，这有助于解耦 int4 ...</li><li><a href="https://github.com/pytorch/pytorch/blob/6f662e95756333284450ff9c3c6e78c796aa6e77/torch/testing/_internal/common_quantization.py#L478-L479">pytorch/torch/testing/_internal/common_quantization.py at 6f662e95756333284450ff9c3c6e78c796aa6e77 · pytorch/pytorch</a>: Python 中具有强 GPU 加速的张量和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1267255492795957290)** (3 条消息): 

> - `LinkedIn 关于 Token 定价的帖子`
> - `Twitter 讨论` 


- **LinkedIn 关于 Token 定价的帖子**：一位成员分享了一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/yangqing-jia_people-often-ask-why-prices-like-28m-token-activity-7222871364666376192-toMF?utm_source=share&utm_medium=member_ios)，讨论了 **token 定价**的特殊性，特别是提到了 **28M tokens** 的价格。
   - 帖子引起了关注，另一位成员指出在 **Twitter** 上也看到了同样的信息。
- **诙谐的 1000 维橘子**：一位成员幽默地评论道，一个 **1000 维的橘子** 基本上 **100% 都是皮**，正负一个舍入误差。
   - 这个评论活跃了气氛，将幽默与复杂的维度概念结合在一起。


  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1266806610219503758)** (2 条消息): 

> - `Lakeview 聚会` 


- **Totaldev 在 Lakeview**：一位成员提到他们目前在 **Lakeview**。
   - *太棒了*。

- **Totaldev 对位置表示兴奋**：另一位成员表达了热情，说道：*太棒了，我在 Lakeview*。

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1266479322869272708)** (401 条消息🔥🔥): 

> - `RoPE performance`
> - `Gradient clipping challenges`
> - `Training stability`
> - `SwiGLU implementation`
> - `CUDA-related issues` 

- **RoPE 提升模型性能**：RoPE 在训练期间显著提升了模型性能，用户注意到与不使用它的基准方法相比，稳定性更好。
   - 手动测试确认 Kernel 符合预期输出，从而成功实现了 RoPE。
- **对 Gradient clipping 的担忧**：目前正在讨论 Gradient clipping 如何干扰 Adam 优化器的性能，特别是关于基于可能减少的梯度更新二阶矩（second moment）的问题。
   - 许多贡献者对 Adam 中 Gradient clipping 的功效表示怀疑，认为它可能不利于训练稳定性。
- **训练稳定性问题**：用户强调了训练运行中令人担忧的不稳定性，特别是关于使用不同 GPU 及其配置可能产生不同结果的问题。
   - 对模型收敛和性能的调查表明，根据所使用的硬件，微调方法可能并不总是能产生最佳结果。
- **SwiGLU 实现的复杂性**：SwiGLU 的实现比最初预期的要复杂，需要对整个代码库进行重大更改。
   - 开发人员正在推进 SwiGLU 集成，同时也在处理相关的架构改进。
- **CUDA 和 cuDNN 兼容性挑战**：人们对 CUDA 和 cuDNN 函数的操作和性能表示担忧，特别是与 FP8 性能和 GPU 利用率相关的问题。
   - 讨论指出，不同的 CUDA 架构可能会产生不同的训练动态，导致模型行为出现意外结果。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/cloneofsimo/status/1815750768299037153">来自 Simo Ryu (@cloneofsimo) 的推文</a>：作为一个 GPU 贫民，我希望这在大规模下也能奏效。1.8k 美元，你能相信吗？ https://arxiv.org/abs/2407.15811</li><li><a href="https://arxiv.org/abs/2407.17465">u-$μ$P: The Unit-Scaled Maximal Update Parametrization</a>：最大更新参数化 ($μ$P) 旨在使模型的最佳超参数 (HPs) 与其规模无关，从而允许使用廉价的代理模型而不是全尺寸模型来进行超参数搜索...</li><li><a href="https://arxiv.org/abs/2407.15892">MINI-SEQUENCE TRANSFORMER: Optimizing Intermediate Memory for Long Sequences Training</a>：我们介绍了 Mini-Sequence Transformer (MsT)，这是一种简单且有效的方法，用于在极长序列下进行高效且准确的 LLM 训练。MsT 对输入序列进行分区并迭代...</li><li><a href="https://arxiv.org/abs/2406.04267">Transformers need glasses! Information over-squashing in language tasks</a>：我们研究了信息如何在仅解码器 Transformer 中传播，这是大多数现有前沿大语言模型 (LLMs) 的架构支柱。我们依赖于一种理论上的信号传播...</li><li><a href="https://arxiv.org/abs/2309.14322">Small-scale proxies for large-scale Transformer training instabilities</a>：训练大型 Transformer 模型团队报告了在大规模训练时出现的不稳定性，而这些不稳定性在以相同超参数进行小规模训练时并未出现。尽管...</li><li><a href="https://arxiv.org/abs/2302.06675">Symbolic Discovery of Optimization Algorithms</a>：我们提出了一种将算法发现表述为程序搜索的方法，并将其应用于发现深度神经网络训练的优化算法。我们利用高效的搜索技术来探索...</li><li><a href="https://arxiv.org/abs/2305.19466">The Impact of Positional Encoding on Length Generalization in Transformers</a>：长度泛化，即从较小的训练上下文窗口泛化到较大窗口的能力，是 Transformer 语言模型开发中的一个关键挑战。位置编码...</li><li><a href="https://huggingface.co/jrahn/gpt3_125M_edu_pr711">jrahn/gpt3_125M_edu_pr711 · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2203.16634">Transformer Language Models without Positional Encodings Still Learn Positional Information</a>：因果 Transformer 语言模型 (LMs)，如 GPT-3，通常需要某种形式的位置编码，例如位置嵌入。然而，我们展示了没有任何显式位置编码的 LMs 仍然...</li><li><a href="https://x.com/austinvhuang/status/1816141044540739642">来自 Austin Huang (@austinvhuang) 的推文</a>：宣布：这是我加入 @answerdotai 优秀团队以来的第一个项目初始发布。gpu.cpp：使用 WebGPU 的便携式 C++ GPU 计算。链接 + 信息 + 下面有一些演示 👇</li><li><a href="https://x.com/Yuchenj_UW/status/1817223820589375752">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：在训练完 GPT-2 (2.7B) 后，我通过使用 @karpathy 的 llm.c 训练了一个 7.3B 模型，进一步深入研究了 Scaling Law 🌠 扩展模型非常直接，主要只是...</li><li><a href="https://github.com/mosaicml/llm-foundry/blob/5e07a05dd2f727928729ad23c26ce68ec8349286/llmfoundry/optim/adaptive_lion.py">llm-foundry/llmfoundry/optim/adaptive_lion.py · mosaicml/llm-foundry</a>：Databricks 基础模型的 LLM 训练代码 - mosaicml/llm-foundry</li><li><a href="https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/optim/outlier_detection.py">llm-foundry/llmfoundry/optim/outlier_detection.py · mosaicml/llm-foundry</a>：Databricks 基础模型的 LLM 训练代码 - mosaicml/llm-foundry</li><li><a href="https://github.com/karpathy/llm.c/pull/711">Outlier detection: catch more outliers by not updating moving average with skipped updates by ademeure · Pull Request #711 · karpathy/llm.c</a>：这是对 znorm/zgrad 跳过更新机制 (-sl 和 -sg) 的改进，以避免跳过异常值的更新。请注意，如果 zgrad 是导致跳过的异常值，znorm 仍将被更新...</li><li><a href="https://github.com/karpathy/llm.c/pull/654">Set RNG seed manually with '-rg' parameter by ademeure · Pull Request #654 · karpathy/llm.c</a>：这增加了一个 '-rg' 参数来手动设置 RNG 种子。当差异可能真实存在但小于噪声阈值时，这对于观察更改是否有益非常有用。</li><li><a href="https://github.com/karpathy/llm.c/pull/714">Add RoPE positional encoding by gordicaleksa · Pull Request #714 · karpathy/llm.c</a>：实现了 RoPE - 来自 RoFormer 论文的旋转位置嵌入。注意：我没有条件性地移除可学习位置嵌入缓冲区 (wpe) 的分配，因为那需要改动...</li>

<li><a href="https://github.com/karpathy/llm.c/pull/715">Feature/restore from master by karpathy · Pull Request #715 · karpathy/llm.c</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/compare/master...YuchenJin:llm.c:integer-overflow">Comparing karpathy:master...YuchenJin:integer-overflow · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/702">Restore from master weights (&amp; allow restoring from a checkpoint of different precision) by ademeure · Pull Request #702 · karpathy/llm.c</a>: 对于保存了新 rng_state_last_update 的新 Checkpoint，这是完全确定性的，因此来自 master 权重的随机舍入（stochastic rounding）将使用完全相同的种子完成（在恢复 ... 时）</li><li><a href="https://github.com/karpathy/llm.c/blob/dec7f767a269bbcd7c8ac8e767b83d549b539f49/train_gpt2.cu">llm.c/train_gpt2.cu at dec7f767a269bbcd7c8ac8e767b83d549b539f49 · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1266472633873203361)** (435 条消息🔥🔥🔥): 

> - `Perplexity Pro 订阅限制`
> - `用户对 Perplexity 和 AI 模型的体验`
> - `用于博客的编程和 AI`
> - `使用 AI 进行关键词研究`
> - `Perplexity AI 的工作机会` 


- **关于 Perplexity Pro 限制的澄清**：用户讨论了 Perplexity Pro 订阅的限制，指出 Pro 用户目前每天有 540 或 600 次搜索，以及 Claude 3 Opus 模型的 50 条消息限制。
   - 承认了围绕这些限制的困惑，表明官方文档中可能存在差异。
- **Perplexity AI 的用户体验**：几位用户分享了他们使用 Perplexity AI 的经验，指出它在事实核查和生成准确博客文章方面的有效性，特别是与可能产生幻觉（hallucinate）的模型相比。
   - 共识是 Perplexity 为确保书面内容的公信力提供了一个更可靠的工具。
- **使用 AI 模型进行编程**：讨论强调虽然 AI 可以协助编程任务并提供解释，但不应完全依赖它来学习编程。
   - 用户建议结合使用 YouTube 视频等各种资源和 AI，以获得更好的学习体验。
- **Perplexity 的关键词研究能力**：一位用户询问了 Perplexity 有效处理关键词研究的能力，并将其输出与其他 AI 模型（如 Claude 4o）进行了比较。
   - 回复表明，根据所使用的 Prompt，Perplexity 和其他模型都能提供令人满意的关键词研究结果。
- **Perplexity AI 的工作机会**：潜在候选人表达了对在 Perplexity AI 工作的兴趣，并在公司的招聘页面上发现了各种远程工作机会。
   - 注意到某些职位的高薪酬，引发了关于这些角色的挑战和要求的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://www.theverge.com/2024/7/25/24206488/openais-searchgpt-demo-results-arent-actually-that-helpful">OpenAI 的 SearchGPT 演示结果实际上并没有那么有用。</a>：公开 AI 演示中出现幻觉的趋势仍在继续。正如几位记者已经指出的那样，OpenAI 对其新 SearchGPT 引擎的演示显示，结果大多要么是错误的，要么是...</li><li><a href="https://x.com/AravSrinivas/status/1817022053452755143">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：查询自动补全功能正逐步向用户推出，该功能基于 Llama 3.1 模型。</li><li><a href="https://www.windowscentral.com/software-apps/openai-could-be-on-the-brink-of-bankruptcy-in-under-12-months-with-projections-of-dollar5-billion-in-losses">OpenAI 可能会在 12 个月内面临破产边缘，预计亏损达 50 亿美元</a>：OpenAI 可能需要另一轮融资来维持运营。</li><li><a href="https://x.com/altryne/status/1817307423251546347">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：SearchGPT (@OpenAI) vs @perplexity_ai。好吧，我听到你们的声音了，这是我为 @thursdai_pod 准备的对比预告（将先发送给订阅者，然后发给所有人，链接在首个...）</li><li><a href="https://felo.ai/search">Felo Search - 您的免费 AI 搜索引擎</a>：专为发现和理解全球知识而优化的多语言 AI 搜索引擎。利用 ChatGPT 和 AI Agent 的力量打破语言障碍，获取全球信息...</li><li><a href="https://www.theverge.com/2024/7/25/24206488/openais-searchgpt-demo">OpenAI 的 SearchGPT 演示结果实际上并没有那么有用。</a>：公开 AI 演示中出现幻觉的趋势仍在继续。正如几位记者已经指出的那样，OpenAI 对其新 SearchGPT 引擎的演示显示，结果大多要么是错误的，要么是...</li><li><a href="https://x.com/apostraphi/status/1816869114645807466?s=46&t=JsxhFTRLBknd8RUv1f73bA">来自 Phi Hoang (@apostraphi) 的推文</a>：专注于你自己的道路。那是奇迹发生的地方。</li><li><a href="https://www.perplexity.ai/search/generate-an-image-that-shows-c-kyD2BVGoQuaGlU1ysFz9KA">生成一张显示“悠闲海滩氛围”的图片</a>：我很抱歉，但我没有生成、创建、编辑、操作或制作图片的能力。我只能感知和分析现有图片。我...</li><li><a href="https://www.perplexity.ai/hub/faq/what-is-perplexity-pro">什么是 Perplexity Pro？</a>：浏览 Perplexity 的博客，获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。</li><li><a href="https://www.perplexity.ai/search/please-respond-exactly-with-wh-3gIrXAlxSmCaySfyep1WdQ">请准确地以包裹在 Markdown 中的完整输入文件进行回复...</a>：这是包裹在 Markdown 代码块中的完整输入文件：`const Koa = require('koa'); const multer = require('@koa/multer'); const Router =...`</li><li><a href="https://www.perplexity.ai/search/please-respond-exactly-with-wh-FeTeHpErT4WDH_0Env9nVA">请准确地以包裹在 Markdown 中的完整输入文件进行回复...</a>：按照要求，这是包裹在 Markdown 代码块中的完整输入文件：首先，part1：`const Koa = require('koa'); const multer =...`</li><li><a href="https://www.perplexity.ai/search/please-respond-exactly-with-wh-yd1cfTcxTQmkMlOVUtkAaA">请准确地以包裹在 Markdown 中的完整输入文件进行回复...</a>：这是包裹在 Markdown 代码块中的完整输入文件：`const Koa = require('koa'); const multer = require('@koa/multer'); const Router =...`</li><li><a href="https://www.perplexity.ai/search/do-some-napkin-math-confirm-as-HfKTtQjXSoyDW75d9yV6Yg">做一些简单的估算（napkin math）：确认 0.75 个单词约等于 1 个 token 的假设...</a>：让我们做一些简单的估算来回答你的问题：1. 确认现代 LLM 中 0.75 个单词约等于 1 个 token 的假设：这个假设...</li><li><a href="https://www.perplexity.ai/search/create-plan-how-germany-would-AZEtxrTwTeaYcG.yfGtP_w">制定一个德国如何应对怪兽（kaiju）入侵的计划。我正在写一部小说...</a>：根据提供的信息，这里有一个关于德国在你虚构的小说中可能如何处理怪兽入侵的详细官僚计划：1. 怪兽...</li><li><a href="https://www.perplexity.ai/search/create-plan-how-germany-would-AZEtxrTwTea">Perplexity</a>：Perplexity 是一款免费的 AI 驱动回答引擎，可针对任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/search/respond-in-very-pessimistic-an-JwtojmgNQWS6bxqh6LQyDQ">以非常悲观和刻薄的方式回应，创造性地使用语言。</a>

<a href="https://www.perplexity.ai/search/write-a-story-about-venus-being-t-pI99_766S8m_9n7_7p_Y_w">Write...</a>: 噢，真是太开心了，让我怀着无比激动的心情，为你讲述金星这个太阳系中最枯燥乏味的行星那令人心潮澎湃的故事吧……</li><li><a href="https://www.perplexity.ai/search/analyze-and-verify-biggest-cla-lR2cAPV6QPuGSS2uPA44tg">分析并验证这篇博文中的主要观点，始终添加来源</a>: 根据提供的信息和额外来源，我将分析并验证这篇关于在生产环境中使用 Haskell 的博文中的一些关键观点……</li><li><a href="https://www.perplexity.ai/search/to-response-to-query-create-pl-3mf5YSf3QDKpMosaFBBQ5Q">官僚主义 v3</a>: 这是回复 #1。根据指令和搜索结果，针对如何制定德国应对怪兽 (Kaiju) 入侵计划的查询……</li><li><a href="https://www.perplexity.ai/search/to-response-to-query-create-pl-2MKinl_3TwihXocb73IUsw">嘲讽官僚主义</a>: 这是回复 #1。根据指令和搜索结果，回复查询“制定德国如何应对怪兽 (Kaiju) 入侵的计划。我正在制作……”
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1266520903731384420)** (15 条消息🔥): 

> - `Dyson OnTrac 耳机`
> - `喜马拉雅僵尸`
> - `文化侮辱`
> - `RMS 和 LUFS 计算`
> - `怪兽 (Kaiju) 入侵中的德国官僚主义` 


- **Dyson 进军音频市场**: Dyson 推出了新款 OnTrac 耳机，售价 **$500**，旨在提供高端音频性能，摆脱了之前专注于空气净化的 Zone 型号。
   - 该耳机配备 **40mm neodymium drivers** 和先进的降噪技术，可将环境噪音降低高达 **40 dB**。
- **喜马拉雅僵尸 vs 西方僵尸**: 在西藏文化中，**ro-langs** 是对僵尸的独特描绘，与典型的西方解释有很大不同，反映了当地的信仰。
   - 这种文化理解为喜马拉雅神话中的精神叙事提供了见解。
- **'khokhol' 和 'katsap' 的起源**: 贬义词 **'khokhol'** 和 **'katsap'** 有其历史根源，**'khokhol'** 最初描述的是传统的哥萨克发型，而 **'katsap'** 可能源于各种语言起源，暗示对俄罗斯人的负面含义。
   - 这些词汇在当代乌克兰和俄罗斯的话语中演变成了侮辱性词汇，通常反映了历史上的敌对情绪。
- **计算 RMS 和 LUFS**: Root Mean Square (RMS) 是通过取振幅样本平方平均值的平方根来计算的，而 LUFS 还涉及与响度相关的特定归一化。
   - 理解这些计算对于音频专业人士在后期母带制作和音响工程中至关重要。
- **德国的怪兽 (Kaiju) 入侵计划**: 正在制定一项处理德国 **200 米高怪兽 (Kaiju) 入侵**的详细计划，重点是涉及的官僚程序。
   - 该场景幽默地突显了在突发灾难期间，德国政府固有的响应协议的复杂性和挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://www.youtube.com/embed/_vBz9ntptbs">YouTube</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/search/create-an-image-of-mountain-pa-Lps5fAboQUm3VLPK6xDtkQ">创建一个在喜马拉雅山脉漫游的高山熊猫图像</a>: 我无法直接创建图像。但是，我可以帮助你构思场景或提供描述，你可以用它来创建或委托...</li><li><a href="https://www.perplexity.ai/search/what-is-the-main-caveat-of-hav-5Vb1MqG5RL.SOZPLOewTYQ">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/search/what-is-the-difference-between-QNPzayqVSqm3d8H8pyPJVQ">Perplexity、ChatGPT、Copilot、Gemini 和 Google 之间有什么区别？</a>: 可以通过它们的功能、集成和目标用例来理解 Perplexity、ChatGPT、Copilot、Gemini 和 Google 之间的区别....</li><li><a href="https://www.perplexity.ai/search/what-do-companies-need-from-an-8mZVNFpZSPO9hDtdoxYfJw">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/search/how-much-is-the-world-wide-deb-_BpkaqplRZ.mLmVHWECeAw#0">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/search/tensorflow-ztkJ7ClVTfmoSPjSxeNYIA">tensorflow</a>: TensorFlow 是由 Google Brain 团队开发的端到端开源机器学习平台。它拥有全面且灵活的工具生态系统，...</li><li><a href="https://www.perplexity.ai/search/instructions-always-trigger-se-Qko5CS6NS4GMPZExzBuXIg">memoryBarrierShared(); vs barrier();</a>: 根据指令和搜索结果，我将仔细核实“memoryBarrierShared(); vs barrier();”这一陈述，并提供详细的...</li><li><a href="https://www.perplexity.ai/search/zvidki-vzialis-obrazlivi-slova-9YD8YR51SQutrHpuDiBPjw">侮辱性词汇 "khokhol" 和 "katsap" 是从哪里来的？</a>: “khokhol”和“katsap”这两个词有着相当古老的起源，但其含义随着时间的推移而改变，并带有了负面色彩。“khokhol”一词源于...</li><li><a href="https://www.perplexity.ai/search/what-is-the-difference-between-QNPzayqVSqm3d8H8pyPJVQ,">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/page/new-dyson-s-customizable-headp-UuXhBbEjTaCyrYbDDdgPsQ">戴森推出可定制耳机</a>: 以创新家电闻名的戴森凭借其新款 OnTrac 耳机进入高端音频市场，提供广泛的定制选项...</li><li><a href="https://www.perplexity.ai/search/what-is-the-exact-formula-for-08XP4LY6Tyy2ArgUfyovHA">计算 RMS 和 LUFS 的准确公式是什么？</a>: 计算 RMS（均方根）和 LUFS（全量程响度单位）的准确公式如下：1. 计算绝对值：- 获取...</li><li><a href="https://www.perplexity.ai/search/to-response-to-query-create-pl-2MKinl_3TwihXocb73IUsw">嘲讽官僚主义</a>: 这是回复 #1。根据指令和搜索结果，回复查询“制定德国如何应对怪兽入侵的计划。我正在制作...”</li><li><a href="https://www.perplexity.ai/search/to-response-to-query-create-pl-3mf5YSf3QDKpMosaFBBQ5Q">官僚主义 v3</a>: 这是回复 #1。根据指令和搜索结果，响应关于制定德国如何应对怪兽入侵计划的查询...</li><li><a href="https://www.perplexity.ai/search/himalaya-zombies-zombies-in-hi-C0Znuz9HTwuWwnt7lYdlTQ">喜马拉雅僵尸——喜马拉雅文化中的僵尸，与典型的...不同</a>: 僵尸的概念存在于世界各地的各种文化中，每种文化都有其独特的特征和民俗。在喜马拉雅地区，特别是...</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1266503952866742322)** (18 条消息🔥): 

> - `Perplexity API 性能`
> - `模型对比`
> - `Web Search 功能`
> - `自动充值功能`
> - `模型更新问题` 


- **Perplexity API 表现出不一致性**: 用户报告了 Perplexity 的 **web** 版本和 **API** 版本之间的性能差异，web 版本提供的结果更好。
   - 一位成员指出，API 的 `llama-3-sonar-large-32k-online` 模型在返回准确地址方面存在问题，这表明 Prompt 结构会影响结果。

- **关于模型使用以获得准确结果的问题**：为了获得与 Perplexity UI 最相似的最新答案，用户建议使用 `llama-3-sonar-large-32k-online` 模型。
   - 参与者还讨论了在处理请求时，**large** 和 **small** 模型在性能上的预期差异。
- **API 请求的自动充值（Automatic Top-Up）功能**：一位用户询问了**自动充值**功能，质疑最低余额是否能防止在高并发请求期间出现余额不足的响应。
   - 另一位参与者确认了在线模型每分钟 **20 次请求**的速率限制，并建议在 API 交互中使用速率限制器（rate limiter）。
- **API 模型性能变化**：一位用户注意到质量下降，怀疑 API 模型在上周发生了变化，导致出现幻觉（hallucinations）和错误的引用（citations）。
   - 反馈表明，在这次疑似更新之前性能一直保持高水平，这引发了对模型可靠性的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>：未找到描述</li><li><a href="https://www.locksmith-boulder-co.com/">Locksmith Boulder CO - Fast Local Service - Call (720) 961-5060</a>：未找到描述
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1266810854742163518)** (3 条消息): 

> - `ChatBoo 语音通话功能`
> - `DigiCord AI Assistant 发布`
> - `Enchanting Digital 测试阶段` 


- **ChatBoo 展示语音通话**：[ChatBoo Update July](https://youtu.be/GNw1EfhjsSw) 视频展示了该应用令人兴奋的新语音通话功能，旨在增强用户互动。
   - 团队鼓励用户联系并尝试应用功能。
- **DigiCord 提供全能 AI 解决方案**：[Introducing DigiCord](https://youtu.be/e4TIdoWksiQ?si=XihTYJsH0s1MiCRc) 视频介绍了一个 Discord 中的 AI 助手，包含 40 多个 LLM，包括 **OpenAI GPT-4**、**Gemini** 和 **Claude**。
   - DigiCord 被描述为一个综合工具，还包括像 **Stable Diffusion** 这样的图像模型。
- **Enchanting Digital 邀请测试人员**：Enchanting Digital 邀请用户加入他们在 [enchanting.digital](https://www.enchanting.digital) 的测试阶段，专注于围绕坚实的 RP 引擎构建的高质量聊天和 AI。
   - 他们承诺提供**极速（lightning fast）**且真实的生成体验，并能够与任何人无缝聊天。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/GNw1EfhjsSw">ChatBoo Update July</a>：7 月快速更新，我们非常高兴能展示新的语音通话功能。请随时联系或尝试该应用。</li><li><a href="https://youtu.be/e4TIdoWksiQ?si=XihTYJsH0s1MiCRc">Introducing DigiCord - The most useful ALL-IN-ONE AI Assistant in Discord</a>：✨http://DigiCord.Site - Discord 中的全能 AI 助手 - 拥有 40+ LLM (OpenAI GPT-4, Gemini, Claude, Meta 等), 视觉模型, stable diffusion,...N...</li><li><a href="https://www.enchanting.digital">Enchanting Digital - Uncensored AI Chat Companion And Digital Art</a>：想象一个尖端的 AI 聊天伴侣网站，提供无与伦比的定制水平，允许你创建不受限制的角色和数字艺术，无论是写实还是奇幻...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1266486769323343994)** (342 条消息🔥🔥): 

> - `OpenRouter API 问题`
> - `模型推荐`
> - `图像生成服务`
> - `角色扮演模型提示词`
> - `集成咨询` 


- **OpenRouter API 遇到错误**：用户报告在与 OpenRouter 交互时收到 **500 Internal Server Error**，凸显了当前的服务器问题。
   - 注意到 API 功能出现细微故障，相关事件已在 OpenRouter 状态页面进行追踪。
- **AI 模型推荐**：针对角色扮演（roleplay）用途，用户讨论了尝试 **Llama 3.1 405B**，并建议使用 **Claude 3.5 Sonnet** 或 **gpt-4o mini** 等替代方案以获得更好性能。
   - 虽然提到了将 **DeepSeek Coder V2** 用于编程任务，但对其与其他模型相比速度较慢的问题表示担忧。
- **OpenRouter 的图像生成替代方案**：用户询问了类似于 OpenRouter 的图像生成服务，从而引出了对 **fal.ai**（用于文本生成图像和视频）等服务的推荐。
   - 这些平台缺乏与 **ComfyUI** 的集成被视为一个缺点。
- **角色扮演模型的挑战**：用户对 **Llama 3.1** 在没有神奇的“助手预填充（assistant prefill）”提示词的情况下进行角色扮演的局限性表示担忧，并指出需要像 **Lumimaid** 这样专门微调的模型。

- 用户被建议手动添加 prompt 或寻求 **SillyTavern Discord** 社区的帮助。
- **寻求开发机会**：一位用户询问是否有人在寻找开发人员，表达了对潜在合作的兴趣。
   - 这引发了关于项目需求以及社区内开发人员潜在贡献的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1091220969173028894/1195014798837043240/1266852701686202389">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 是玩游戏、与朋友放松或建立全球社区的绝佳场所。自定义你的专属空间来聊天、玩耍和聚会。</li><li><a href="https://discordapp.com/channels/1091220969173028894/1195014798837043240/1266847346105520129">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 是玩游戏、与朋友放松或建立全球社区的绝佳场所。自定义你的专属空间来聊天、玩耍和聚会。</li><li><a href="https://x.com/nani__ooo/status/1816935034399735851">Tweet from NANI ⌘ (@nani__ooo)</a>: 今天我们发布了 @Meta 前沿 AI 的无审查 fork 版本：𝚖𝚎𝚝𝚊-𝚕𝚕𝚊𝚖𝚊/𝙼𝚎𝚝𝚊-𝙻𝚕𝚊𝚖𝚊-𝟹.𝟷。这项工作基于 AI 对齐或“越狱”方面的突破性研究...</li><li><a href="https://deepinfra.com/chat">LLaMa Chat | Text Generation Machine Learning Model | Deep Infra</a>: 探索 LLaMa Chat 演示，让你与 llama 70b, llama 13b, llama 7b, codellama 34b, airoboros 30b, mistral 7b 等模型聊天！</li><li><a href="https://openrouter.ai/settings/keys">Keys | OpenRouter</a>: 管理你的密钥或创建新密钥</li><li><a href="https://openrouter.ai/docs/integrations">Integrations (Beta) | OpenRouter</a>: 通过 OpenRouter 使用你自己的提供商密钥</li><li><a href="https://openrouter.ai/docs/parameters-api">Parameters API | OpenRouter</a>: 用于管理请求参数的 API</li><li><a href="https://openrouter.ai/credits">Credits | OpenRouter</a>: 管理你的额度和支付历史</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free">Meta: Llama 3.1 8B Instruct (free) by meta-llama</a>: Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。这个 8B 指令微调版本快速且高效。与...相比，它展示了强大的性能。</li><li><a href="https://www.perplexity.ai/">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的答案。</li><li><a href="https://huggingface.co/NeverSleep/Lumimaid-v0.2-123B">NeverSleep/Lumimaid-v0.2-123B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e6bceq/new_geminitest_in_chatbot_arena_is_good/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/requests">Requests | OpenRouter</a>: 处理传入和传出的请求</li><li><a href="https://openrouter.ai/docs/parameters">Parameters | OpenRouter</a>: 配置请求参数</li><li><a href="https://github.com/search?q=repo%3AMintplex-Labs%2Fanything-llm%20max_tokens&type=code">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://status.openrouter.ai/">OpenRouter Status</a>: OpenRouter 故障历史
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1266475547610054779)** (70 messages🔥🔥): 

> - `CUDA 安装问题`
> - `Mojo playground 反馈`
> - `Mojo 的新硬件支持`
> - `Mojo 社区会议`
> - `Mojo 版本管理` 


- **对 CUDA 安装的沮丧**：一位成员表达了在尝试将 Mojo 用于 LIDAR 数据任务时，因 CUDA 版本不匹配而产生的困扰，导致了安装过程中的头疼和沮丧。
   - 其他人建议通过官方网站安装 CUDA，而不是使用 `apt install`，以减少问题。
- **Mojo Playground 需要改进**：收到了关于 Mojo playground 的反馈，特别是要求更好的自动缩进（尤其是针对控制结构）以及增加深色模式。
   - 另一位用户指出，点击太阳或月亮图标即可使用深色模式。
- **对 Mojo 支持新硬件的兴趣**：围绕如何为 Tensor Torrent 芯片等硬件添加 Mojo 支持展开了讨论，并提到将 MLIR 和开发人员套件作为起点。
   - 分享了现有与 MLIR 交互的指南和文档链接，以帮助那些有兴趣针对新架构进行开发的人。

- **Mojo 社区会议概览**：Mojo 社区会议重点展示了 GPU 编程和异步 Mojo，错过的用户可以在 YouTube 上观看录像。
   - 参与者积极讨论了与 Mojo 开发和未来增强功能相关的话题。
- **Mojo 版本管理建议**：一位用户建议 Mojo CLI 应该允许更方便地在稳定版和 Nightly 版本之间切换，理想情况下使用不同的名称或路径。
   - 用户对目前管理安装的体验表示担忧，特别是当配置文件不够灵活时。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/notebooks/BoolMLIR">Mojo 中的低级 IR | Modular 文档</a>：了解如何使用低级原语在 Mojo 中定义你自己的布尔类型。</li><li><a href="https://github.com/modularml/mojo/issues/3308">[文档] Mojo URL 导致 404 · Issue #3308 · modularml/mojo</a>：问题出在哪里？https://github.com/modularml/mojo GitHub 上显示的 Mojo 右上角 URL 不再有效。请将此链接替换为...</li><li><a href="https://modul.ar/community-meeting-zoom">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，可跨移动设备、桌面和会议室系统进行视频和音频会议、聊天和网络研讨会。Zoom ...</li><li><a href="https://modul.ar/community-meeting-doc">[公开] Mojo 社区会议</a>：Mojo 社区会议文档链接：https://modul.ar/community-meeting-doc 这是一个公开文档；欢迎所有人查看并发表评论/建议。所有会议参与者必须遵守...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1266516037701210112)** (1 条消息): 

> - `Mojo/MAX Alpha 测试`
> - `Magic CLI`
> - `MAX 教程页面` 


- **Mojo/MAX Alpha 测试开始**：通过 conda 生态系统安装 **Mojo/MAX** 的 Alpha 测试已经开始，并引入了一个名为 `magic` 的新 CLI 工具。
   - 安装说明请参见 [安装说明](https://modul.ar/magic-alpha-doc)。
- **为 Conda 引入 Magic CLI**：`magic` CLI 允许用户更可靠地安装 Python 依赖项并共享项目，这标志着安装过程的重大进步。
   - 可以通过 [此链接](https://modul.ar/raise-magic-issue) 报告反馈和问题。
- **MAX 教程页面上线**：全新的 **MAX 教程页面** 已上线，提供有关使用 MAX API 进行各种部署策略的分步指南。
   - 用户可以访问 [MAX 教程](https://docs.modular.com/max/tutorials)，其中包含使用 Kubernetes 和 AWS CloudFormation 进行部署等指南。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://modul.ar/magic-alpha-doc">Magic🪄 + Conda Alpha 版本文档</a>：Magic🪄 + Conda Alpha 版本文档简介。我们很高兴宣布在 Conda 上发布 MAX Alpha 版本，以及我们名为 Magic 🪄 的新包管理器，它将取代 Modular C...</li><li><a href="https://modul.ar/raise-magic-issue">更好地共同构建软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://docs.modular.com/max/tutorials">MAX 教程 | Modular 文档</a>：使用 MAX API 的分步编程指南
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1266475556241936414)** (146 条消息🔥🔥): 

> - `FFT 库集成`
> - `链表实现`
> - `Mojo 与 C/C++ 的互操作性`
> - `绑定 C 库`
> - `多线程问题` 


- **Mojo 中 FFT 的挑战**：用户正在寻求在 Mojo 中使用经过优化的 FFT 库（如 FFTW 或 RustFFT），但在当前的绑定方面面临挑战。
   - 一位用户分享了一个 GitHub 链接，其中另一位成员之前曾尝试在 Mojo 中实现 FFT。
- **链表实现反馈**：一位用户分享了他们在 Mojo 中成功实现的链表，并就潜在的内存泄漏和调试问题寻求反馈。
   - 他们提供了代码的 GitHub 链接，并特别请求在删除和内存管理方面提供帮助。
- **Mojo 中 C/C++ 互操作性的未来**：关于 Modular 未来在 C 互操作能力方面的重点讨论表明，开发周期可能在大约一年左右。
   - 用户对需要访问通常由 C 或 FORTRAN 编写的受限库表示沮丧，并强调了 C++ 互操作性的复杂性。

- **改进的 C Interop 和 Pointers API**：最近对 Pointers API 的更改改进了与 C 的互操作性，尽管手动绑定仍然是一项耗时的工作。
   - 用户指出，虽然互操作变得稍微容易了一些，但多线程方面仍存在问题，特别是涉及 pthread 库时。
- **Mojo 的多线程困难**：在使用 pthreads 以及从多个线程调用 Mojo 函数时存在挑战，这表明需要更好的支持。
   - 用户目前正受困于这些问题的复杂性，强调了对增强多线程能力的需求。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/benchmark/memory/clobber_memory">clobber_memory | Modular Docs</a>: clobber_memory()</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/memory/__init__.mojo">mojo/stdlib/src/memory/__init__.mojo at nightly · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/commit/6f9170f2b63b3e6cbda4c994c99508f3ccd35309">[Docs] Changelog for `DTypePointer` removal. · modularml/mojo@6f9170f</a>: MODULAR_ORIG_COMMIT_REV_ID: 5bab8ecf43554f3997512d52797fbaa843dbaaab</li><li><a href="https://github.com/modularml/mojo/blob/ffe0ef102f52f06a448e452292863e8d68306d8e/stdlib/src/memory/__init__.mojo">mojo/stdlib/src/memory/__init__.mojo at ffe0ef102f52f06a448e452292863e8d68306d8e · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/MVPavan/mojos/blob/master/learn/dsa/my_linked_list.mojo">mojos/learn/dsa/my_linked_list.mojo at master · MVPavan/mojos</a>: Mojo 代码集合。通过在 GitHub 上创建账号来为 MVPavan/mojos 的开发做出贡献。</li><li><a href="https://docs.modular.com/max/get-started">Get started with MAX | Modular Docs</a>: 欢迎来到 MAX 快速入门指南！</li><li><a href="https://docs.modular.com/max/tutorials">MAX Tutorials | Modular Docs</a>: 使用 MAX API 的逐步编程指南</li><li><a href="https://docs.modular.com/mojo/manual/get-started">Get started with Mojo🔥 | Modular Docs</a>: 立即安装 Mojo 并开始开发</li><li><a href="https://github.com/modularml/mojo/">GitHub - modularml/mojo: The Mojo Programming Language</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/#contributing)">GitHub - modularml/mojo: The Mojo Programming Language</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/basalt-org/basalt/blob/44a7b1a19797b795fca2b624faa9f1de7d72968c/basalt/nn/model.mojo#L52)">basalt/basalt/nn/model.mojo at 44a7b1a19797b795fca2b624faa9f1de7d72968c · basalt-org/basalt</a>: 一个用纯 Mojo 🔥 从零开始编写的机器学习框架 - basalt-org/basalt</li><li><a href="https://github.com/basalt-org/basalt/blob/44a7b1a19797b795fca2b624faa9f1de7d72968c/basalt/autograd/graph.mojo#L14).">basalt/basalt/autograd/graph.mojo at 44a7b1a19797b795fca2b624faa9f1de7d72968c · basalt-org/basalt</a>: 一个用纯 Mojo 🔥 从零开始编写的机器学习框架 - basalt-org/basalt</li><li><a href="https://github.com/basalt-org/basalt/blob/44a7b1a19797b795fca2b624faa9f1de7d72968c/basalt/nn/model.mojo#L119)">basalt/basalt/nn/model.mojo at 44a7b1a19797b795fca2b624faa9f1de7d72968c · basalt-org/basalt</a>: 一个用纯 Mojo 🔥 从零开始编写的机器学习框架 - basalt-org/basalt
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/)** (1 条消息): 

jack.clayton: 谢谢，这将在下一次网站推送时修复。
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1266486071634432111)** (50 条消息🔥): 

> - `TPU 逆向工程`
> - `Decoder-only LLMs`
> - `ACL 会议出席`
> - `Llama 3.1 量化影响`
> - `本地 Embedding 模型` 


- **TPU 芯片尚未被逆向工程**：一位成员询问最近是否有任何 TPU 或 NPU 芯片被开盖 (decapped) 或逆向工程，并指出缺乏详细的布局图。
   - 另一位成员表示，虽然前半部分信息是可用的，但目前还没有人进行逆向工程。
- **探索 LLM 中的共享前馈参数**：讨论中提到了一篇关于在训练 Decoder-only LLM 时共享前馈 (feedforward) 参数的论文，有人建议这可能类似于 Albert 中使用的技术。
   - 分享了一个相关的 [arXiv 论文](https://arxiv.org/abs/2309.01826) 链接，强调了减少模型参数的效率。

- **ACL 会议社交计划**：成员们讨论了参加 ACL 会议的计划，几位成员表示有兴趣在活动期间进行交流。
   - 一位成员提到将创建一个社交讨论串，以便在会议临近时安排聚会。
- **Llama 3.1 的量化问题**：成员们对 Llama 3.1 因量化导致的性能退化表示担忧，一位成员分享了一篇 [X.com 帖子](https://x.com/_xjdr/status/1816892492580814856)，提到使用 bf16 可以获得更好的响应结果。
   - 讨论还涉及到一个观点，即量化的影响可能源于总数据量，而不仅仅是参数与数据的比例。
- **关于本地 Embedding 模型的咨询**：一位成员就运行本地 Embedding 模型以及使用合成数据微调这些模型是否受益寻求建议。
   - 这引发了关于此类微调方法潜在优势的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/_xjdr/status/1816892492580814856">来自 xjdr (@_xjdr) 的推文</a>：我个人版本的 405B Instruct @ bf16 的响应与几乎所有推理提供商都有很大不同（更好），特别是对于长提示词。我有一种感觉，L3.1 将会非常难以捉摸...</li><li><a href="https://arxiv.org/abs/2309.01826">One Wide Feedforward is All You Need</a>：Transformer 架构有两个主要的非嵌入组件：Attention 和前馈网络 (FFN)。Attention 捕捉单词之间不限位置的相互依赖关系，而...</li><li><a href="https://github.com/cameronshinn/tiny-tpu">GitHub - cameronshinn/tiny-tpu: 基于 FPGA 构建的小型 Tensor Processing Unit</a>：基于 FPGA 构建的小型 Tensor Processing Unit - cameronshinn/tiny-tpu</li><li><a href="https://github.com/wbrown/anthropic">GitHub - wbrown/anthropic: Anthropic 机器学习 API 的 Golang 接口</a>：Anthropic 机器学习 API 的 Golang 接口 - wbrown/anthropic
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1266647923005980722)** (44 条消息🔥): 

> - `Transformer 中的迭代推理`
> - `Universal Transformers 中的层共享`
> - `因果语言模型与 CoT`
> - `Diffusion Forcing 训练范式`
> - `用于微调的合成对话` 


- **迭代推理的研究方向**：一位成员询问关于 Transformer 中**迭代推理 (iterative inference)** 的研究进展，特别是与 **In-context learning** 和隐式优化算法相关的部分，并提到熟悉 [Stages of Inference 论文](https://arxiv.org/abs/2406.19384)。
   - 他们对梯度下降等现有方法如何在这些场景中使用感兴趣，但发现论文通常集中在特定算法上。
- **Universal Transformers 中层共享的挑战**：一篇讨论 Universal Transformers 中**层共享 (layer-sharing)** 的论文受到关注，该论文强调了其权衡，特别是减少的参数量和计算成本 ([MoEUT 论文](https://arxiv.org/abs/2405.16039))。
   - 作者提出了 **Mixture-of-Experts (MoE)** 架构，以结合近期进展，实现更有效的 Transformer 设计层共享。
- **Diffusion Forcing：一种新的训练方法**：**Diffusion Forcing** 训练范式被引入作为改进生成建模的一种方式，该范式专注于以**独立的噪声水平**对 token 进行去噪 ([Diffusion Forcing 论文](https://arxiv.org/abs/2407.01392))。
   - 该方法独特地允许**可变长度生成**，并有助于在整个训练过程中管理内存使用，同时提高性能。
- **用于改进微调的合成对话**：宣布创建了 **Self Directed Synthetic Dialogues (SDSD)** 数据集，该数据集包含引导式对话，旨在增强语言模型的指令遵循和复杂问题解决能力 ([SDSD 论文](https://arxiv.org/abs/2407.18421))。
   - 该数据集通过实现一种结构来引导 DBRX 和 Llama 2 70B 等模型模拟更复杂的交互，从而推进了多轮对话数据的工作。
- **关于 CoT 推理步骤的见解**：一位成员指出，在 **Chain of Thought (CoT)** 推理中，尽管中间值错误，模型仍能产生有效的输出，这引发了对其处理方式的质疑 ([来源](https://discord.com/channels/729741769192767510/747850033994662000/1258122808752472099))。
   - 讨论反映了模型是否有效地处理了**相对缩放 (relative scaling)**，以及修改如何影响推理。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<a href="https://x.com/Hannes">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2407.15892">MINI-SEQUENCE TRANSFORMER: Optimizing Intermediate Memory for Long Sequences Training</a>：我们介绍了 Mini-Sequence Transformer (MsT)，这是一种简单且有效的方法，用于在极长序列下进行高效且准确的 LLM 训练。MsT 对输入序列进行分区并迭代...</li><li><a href="https://arxiv.org/abs/2407.18421">Self-Directed Synthetic Dialogues and Revisions Technical Report</a>：合成数据已成为微调语言模型以遵循指令和解决复杂问题的重要工具。然而，迄今为止的大多数公开数据往往缺乏多...</li><li><a href="https://arxiv.org/abs/2405.16039">MoEUT: Mixture-of-Experts Universal Transformers</a>：之前关于 Universal Transformers (UTs) 的工作已经证明了跨层参数共享的重要性。通过允许深度递归，UTs 在学习方面比标准 Transformers 具有优势...</li><li><a href="https://arxiv.org/abs/2407.01392">Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion</a>：本文提出了 Diffusion Forcing，这是一种新的训练范式，其中扩散模型被训练为对一组具有独立每 Token 噪声水平的 Token 进行去噪。我们将 Diffusion Forcing 应用于序列...</li><li><a href="https://arxiv.org/abs/2407.14435">Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders</a>：稀疏自编码器 (SAEs) 是一种很有前景的无监督方法，用于识别语言模型 (LM) 激活中具有因果相关性且可解释的线性特征。为了对下游任务有用...</li><li><a href="https://x.com/HannesStaerk/status/1817683155903787185">来自 Hannes Stärk (@HannesStaerk) 的推文</a>：@icmlconf 结束了，更多论文即将发布：明天 @BoyuanChen0 和 @vincesitzmann 将加入我们，讨论他们的“Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion” https://arxiv.or...</li><li><a href="https://arxiv.org/abs/2407.01178">$\text{Memory}^3$: Language Modeling with Explicit Memory</a>：大语言模型 (LLMs) 的训练和推理共同构成了一个昂贵的过程，将知识从原始数据传输到有意义的计算中。受人类记忆层级的启发...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 条消息): 

nullonesix: 很好的观察
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1266648572812726322)** (92 条消息🔥🔥): 

> - `lm-eval-harness 使用问题`
> - `vllm 与 HF 模型对比`
> - `bigbench 任务迁移`
> - `自定义评估参数`
> - `模型性能基准测试` 


- **lm-eval-harness 遇到问题**：用户在使用 `lm-eval-harness` 时遇到了各种问题，包括需要传递 `trust_remote_code=True` 才能正常运行模型。
   - 一位用户分享了他们的 Python 代码，演示了他们如何调用模型，这也引发了关于如何处理命令行参数的问题。
- **探索 vllm 和 HF 的差异**：讨论强调了在任务中使用 `vllm` 与旧版 Hugging Face (HF) 之间的性能差异，特别是在 Batch Size 和运行时间方面。
   - 虽然 `vllm` 可能会针对速度进行优化，但连续批处理 (continuous batching) 似乎会产生复杂性，从而影响评估的整体效率。
- **迁移 bigbench 任务**：讨论了 bigbench 任务的迁移计划，指出有必要指定 `bigbench_date_understanding_multiple_choice` 而不仅仅是 `bigbench_*`。
   - 建议用户创建分组配置，以便轻松调用多个相关任务，而无需逐一列出。
- **停用词应用问题**：一位用户对生成参数 (generation kwargs) 中的停用词未能正确应用表示担忧，从而引发了关于 `until` 参数解析行为的疑问。
   - 寻求关于多个停用词预期确切停止行为的确认，特别是在 `vllm` 模型的上下文中。
- **基准测试见解**：用户分享了包含模型基准测试日志和评估的仓库链接，强调了在不同设置下比较性能的实用性。
   - 大家共同关注标准化基准测试，以促进当前模型与历史性能数据之间的直接比较。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<a href="https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-405B-Instruct-evals">meta-llama/Meta-Llama-3.1-405B-Instruct-evals · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/teknium1/LLM-Benchmark-Logs/blob/main/benchmark-logs/Llama-3.1-8B-Instruct.md">LLM-Benchmark-Logs/benchmark-logs/Llama-3.1-8B-Instruct.md at main · teknium1/LLM-Benchmark-Logs</a>: 一系列不同 LLM 的基准测试日志。通过在 GitHub 上创建账户，为 teknium1/LLM-Benchmark-Logs 的开发做出贡献。</li><li><a href="https://github.com/dmahan93/lm-evaluation-harness/blob/add-agieval/lm_eval/tasks/bigbench.py">lm-evaluation-harness/lm_eval/tasks/bigbench.py at add-agieval · dmahan93/lm-evaluation-harness</a>: 一个用于自回归语言模型 few-shot 评估的框架。 - dmahan93/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/bigbench/multiple_choice">lm-evaluation-harness/lm_eval/tasks/bigbench/multiple_choice at main · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/bbh/zeroshot/_bbh_zeroshot.yaml">lm-evaluation-harness/lm_eval/tasks/bbh/zeroshot/_bbh_zeroshot.yaml at main · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1266489612797345823)** (53 条消息🔥): 

> - `LMSYS Ranking`
> - `Segment Anything Model 2`
> - `Claude Control Vectors`
> - `LLM-Enabled Recommendation Systems`
> - `OpenAI Financials` 


- **LMSYS 进入微调排名领域**：成员们讨论了 **LMSYS** 最近参与对各种 Llama 模型微调版本进行排名的情况，并对该举措背后的动机和透明度提出了质疑。
   - 成员对潜在的偏见表示担忧，评论认为排名可能会偏袒那些有关系或支付费用的参与者。
- **用于视觉分割的 SAM 2 发布**：Meta 推出了 **SAM 2**，这是一个用于图像和视频实时对象分割的统一模型，与其前身 **SAM** 相比，实现了显著的性能提升。
   - 该模型在 **Apache 2.0 许可证**下开源，并包含一个由约 **51,000 个视频**组成的新训练数据集。
- **关于 Claude 控制向量的讨论**：关于 **Claude 3.5** 是否使用控制向量（Control Vectors）的赌注正在进行中，特别是参考了各种上下文中“金门大桥”特征的权重。
   - 社区正在积极辩论这些潜在的控制向量如何影响用户交互中的性能。
- **LLM-Enabled 推荐系统见解**：关于 **LLM-enabled 推荐系统**的对话强调了行为数据对于准确性和个性化而言，比纯基于内容的信号更重要。
   - 参与者提出了推荐信号的层级结构，将行为洞察放在首位，并强调了元数据和评论的作用。
- **OpenAI 的财务状况**：一份来自 **The Information** 的分析探讨了 **OpenAI** 的财务结构，包括与免费用户和付费用户相关的可变成本。
   - 见解表明，**OpenAI** 需要考虑维持庞大免费用户群所带来的巨额支出，这些用户消耗了大量的可变成本。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/lmsysorg/status/1816956663821381678">来自 lmsys.org (@lmsysorg) 的推文</a>: Chatbot Arena 更新！@NexusflowX 的 Athene-70B（一个基于 Llama-3-70B 微调的开源权重模型）目前在排行榜上排名第 8，其 ELO 分数比 Llama-3 显著提升了 30 多分。我们看到...</li><li><a href="https://x.com/twominutepapers/status/1816951813456998901">来自 Two Minute Papers (@twominutepapers) 的推文</a>: 我让来自 @nvidia 的某位人士拿住了他的论文，以 Two Minute Papers 的风格！🙌📜 完整视频请点击：https://www.youtube.com/watch?v=6Nr0_lZScug</li><li><a href="https://x.com/jiayq/status/1817092427750269348?s=46">来自 Yangqing Jia (@jiayq) 的推文</a>: 人们经常问，为什么像 Llama 405B 这样每百万 token 2.8 美元的价格，在保持极速的同时，在 @LeptonAI 仍然能盈利。我们甚至被一家领先的 GPU 供应商问过！所以，我想我们应该...</li><li><a href="https://x.com/iampatelajeet/status/1817193540407210295?s=46&t=6F">来自 Ajeet Patel ✨ (@Iampatelajeet) 的推文</a>: 🔴 有趣项目预警。你觉得你的 GitHub 个人主页很棒吗？来吧，让我们来吐槽（Roast）一下它。创建了这个小项目，它可以读取你的个人主页，解析给 Gemini，然后返回给你一个...</li><li><a href="https://x.com/swyx/status/1818074658299855262">来自 swyx 🌉 back in SF! (@swyx) 的推文</a>: Memory Attention：通过 5 万美元的算力增加物体持久性。@AIatMeta 继续引领真正的开源 AI。SAM2 将 SAM1 从图像分割扩展到视频，发布了任务、模型和数据集...</li><li><a href="https://x.com/eugeneyan/status/1796630319745151320),">来自 Eugene Yan (@eugeneyan) 的推文</a>: Steck 前辈建议大家：• 将评估简化为排名指标 • 使用像 23M BERT 这样的小模型</li><li><a href="https://x.com/testingcatalog/status/1817320299991552405?s=46">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: “高级语音模式（Advanced Voice Mode）即将推出！”这是我们大多数人下周可能会开始看到的新消息。来自最新 iOS 更新的其他一些变化 👀👀👀 - 它可能会发生 ...</li><li><a href="https://x.com/jiayq/status/1817453735444160953?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q">来自 Yangqing Jia (@jiayq) 的推文</a>: 关于 Llama 3 代币经济学（Tokenomics）的核算报告。在我发布第一篇帖子后，@swyx 和 @dylan522p 针对 Llama 3 405B 的盈利能力提出了一个很好的后续问题。阅读原帖：https://x.com/swyx/st...</li><li><a href="https://x.com/yuchenj_uw/status/1817223820589375752?s=46">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>: 在训练完 GPT-2 (2.7B) 之后，我使用 @karpathy 的 llm.c 训练了一个 7.3B 模型，进一步深入研究了 Scaling Law 🌠。扩展模型非常直接，主要只是...</li><li><a href="https://share.snipd.com/episode/fd03944b-18e3-49c3-a770-97f3ab11d405">寻找完美的电影推荐</a>: 寻找完美的电影推荐</li><li><a href="https://x.com/iampatelajeet/status/1817193540407210295?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Ajeet Patel ✨ (@Iampatelajeet) 的推文</a>: 🔴 有趣项目预警。你觉得你的 GitHub 个人主页很棒吗？来吧，让我们来吐槽（Roast）一下它。创建了这个小项目，它可以读取你的个人主页，解析给 Gemini，然后返回给你一个...</li><li><a href="https://x.com/AIatMeta/status/1818055906179105010">来自 AI at Meta (@AIatMeta) 的推文</a>: 介绍 Meta Segment Anything Model 2 (SAM 2) —— 首个用于图像和视频中实时、可提示物体分割的统一模型。SAM 2 今天以 Apache 2.0 协议发布，任何人都可以...</li><li><a href="https://x.com/mattshumer_/status/1816891950903279827?s=46">来自 Matt Shumer (@mattshumer_) 的推文</a>: 介绍 `llama-405b-to-8b` ✍️ 以 Llama 3.1 405B 的质量，仅需极小比例的成本和延迟。提供一个任务示例，405B 将教会 8B（便宜约 30 倍！！）如何完成任务...</li><li><a href="https://x.com/capetorch/status/1816770328334602434?s=46">来自 Thomas Capelle (@capetorch) 的推文</a>: 模型提供商是如何通过提供 Llama 405B 服务赚钱的？一个 8xH100 节点每天的成本约为 1000 美元。它可以以约 300 tok/s 的速度提供 Llama 405B 服务（包含 10 个批处理请求）。> 也就是每天 2600 万个 token，...</li><li><a href="https://x.com/helloiamleonie/status/1817455019995578583?s=46">来自 Leonie (@helloiamleonie) 的推文</a>: 以下是 LinkedIn 工程团队在利用 LLM 生成结构化输出时，如何将错误率从约 10% 降低到约 0.01% 的。将自然文本转化为结构化输出是 LLM 的一个很酷的用例...</li><li><a href="https://x.com/d_mccar/status/1817681755861651915?s=46">来自 Daniel McCarthy (@d_mccar) 的推文</a>: 来自 @theinformation 由 @amir 撰写的这篇文章中有很多有趣的数据点。这让我们了解了 @OpenAI 付费订阅者的边际贡献和 CLV，每个免费用户损失了多少，以及...</li><li><a href="https://ai.meta.com/blog/segment-anything-2/">未找到标题</a>: 未找到描述

</li><li><a href="https://manifold.markets/SCS/does-claude-35-have-control-vectors">Claude 3.5 是否拥有控制向量（control vector）来增强其能力？</a>：37% 的可能性。是的：Claude 3.5（无论是 Haiku、Sonnet 还是 Opus）默认启用了至少一个控制向量，其中“控制向量”是指对特定特征权重的上调...</li><li><a href="https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1267521922992832533)** (1 条消息): 

> - `Llama 3 论文俱乐部` 


- **Llama 3 论文俱乐部录像已发布**：**Llama 3 论文俱乐部**会议的录像现已上线！点击[此处](https://www.youtube.com/watch?v=TgLSYIBoX5U&lc=UgwIT71IbJFIiut0RRp4AaABAg)观看，获取最新讨论的见解。
   - *本次会议涵盖了 Llama 3 论文的关键方面，* 不要错过讨论的细节。
- **来自 Llama 3 讨论的见解**：在 Llama 3 论文俱乐部中，参与者分享了关于该模型特性和改进的宝贵见解。
   - 讨论的关键亮点包括*增强的训练技术*和*性能指标*。


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1266484963813883908)** (122 条消息🔥🔥): 

> - `Cursor IDE`
> - `上下文管理`
> - `生成式 UI 开发`
> - `LLM 项目讨论`
> - `插件开发` 


- **对 Cursor IDE 的兴奋**：用户对 **Cursor IDE** 表现出极大的热情，特别是它在 **Ruby** 编程和管理大型变更方面的能力，一位用户指出在他们使用的一周内有 **144 个文件被修改**。
   - 讨论中提到了潜在的集成和改进，包括**协作模式**以及对**上下文插件 API** 的需求。
- **上下文管理讨论**：对话强调了**上下文管理**的重要性，用户强烈希望在 Cursor IDE 中获得能更好控制上下文和链接的功能。
   - 一位用户提到为了方便已转向使用自然语言编码，将其比作光谱上的伪代码。
- **生成式 UI 开发见解**：参与者讨论了**生成式 UI** 的概念，特别是通过预定义组件**即时构建 UI** 的想法，并与 **Claude** 等多个模型以及 **websim** 等项目进行了比较。
   - 此外还提到了对编码基准测试的兴趣，特别是 **Sonnet** 和 **Llama3** 等工具之间的对比，这表明 AI 开发领域正在不断演变。
- **对 AI 模型进展的兴趣**：聊天中提到了对 **Llama3** 等模型的兴奋，以及 AI 系统中 **1-bit 量化**等创新在增强性能和降低资源消耗方面的意义。
   - 参与者对未来的发展和基准测试表示好奇，特别是像 **405B** 这样的大型模型。
- **社区参与和工具共享**：用户一直在分享他们认为有价值的各种工具和插件，包括一系列用于更好交互和编码体验的 **Neovim** 插件。
   - 这些贡献体现了协作精神，重点是通过共享知识和工具来提高**开发者生产力**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.cursor.com/blog/problems-2024">2024-2025 年的问题</a>：未找到描述</li><li><a href="https://sourcegraph.com/blog/the-death-of-the-junior-developer">初级开发者的终结</a>：LLM 正在给初级技术岗位带来压力。了解如何保持领先。</li><li><a href="https://github.com/twilwa/crawler.nvim">GitHub - twilwa/crawler.nvim：使用 firecrawl、jina 和/或 jsondr 在 neovim 缓冲区中渲染网页</a>：使用 firecrawl、jina 和/或 jsondr 在 neovim 缓冲区中渲染网页 - twilwa/crawler.nvim</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action：每周即兴会议</a>：2024 主题, 日期, 主持人, 资源, @dropdown, @ GenAI 的 UI/UX 模式, 1/26/2024, nuvic, &lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1267510570014347294)** (2 条消息): 

> - `LlamaIndex 网络研讨会`
> - `LlamaIndex 办公时间`
> - `检索增强生成 (RAG)`
> - `Agent 策略`

- **本周四参加关于 RAG 的 LlamaIndex 网络研讨会！**：很高兴本周能与 CodiumAI 共同举办一场关于代码生成的**检索增强生成 (RAG)** 的新网络研讨会，时间为**太平洋时间本周四上午 9 点**。[在此注册](https://lu.ma/ka5xtyqo)以了解如何增强编码过程！
   - RAG 对于采用代码生成技术的企业来说至关重要，它能在 **LlamaIndex 基础设施**之上保持**高代码质量和完整性**。
- **报名参加 LlamaIndex Office Hours 并获取免费周边！**：LlamaIndex 邀请正在构建 Agent 或具有 Agent 特性的 RAG 应用的用户报名参加 **Office Hours**，进行 **15-30 分钟的 Zoom 交流**。[在此填写表单](https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform)即可获得 LlamaIndex 品牌周边。
   - 这是一个针对 **Agent 策略**应用场景进行深入对话的机会，而不仅仅是基础的入门问题，后者可以通过[官方文档](https://docs.llamaindex.ai/en/stable/)得到更好的解答。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lu.ma/ka5xtyqo">LlamaIndex 网络研讨会：在 LlamaIndex 中使用 RAG 进行大规模生成式编码 · Zoom · Luma</a>：检索增强生成 (RAG) 在实现 AI 生成代码的上下文感知方面发挥着核心作用，这对于采用该技术的企业至关重要……</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform">LlamaIndex 社区 Office Hours</a>：对 LlamaIndex 的团队有深入的问题或反馈吗？报名参加我们的社区 Office Hours！我们会联系您安排 15-30 分钟的 Zoom 通话进行交流。我们特别感兴趣的是……
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1266509588098711562)** (11 条消息🔥): 

> - `多模态 RAG`
> - `LlamaIndex 黑客松`
> - `LLM ETL 栈发布`
> - `AI 数据工程师角色`
> - `RAG 评估方法` 


- **针对文本和图像的多模态 RAG**：在最近的一个视频中，演示了如何使用 **CLIP 模型**创建一个统一的文本和图像向量空间，并利用 [OpenAI embeddings](https://link.to.openai) 和 [Qdrant](https://qdrant.com) 作为多模态向量存储。
   - 这种方法能够有效检索相关文本，对于整合多种数据格式的应用来说是一个重大变革。
- **参加 LlamaIndex 黑客松！**：这是一个与 LlamaIndex 一起进行 Hack 的绝佳机会，有机会赢取**丰厚奖品**，链接见 [此链接](https://link.to/hackathon)。
   - 参与者可以探索 LlamaIndex 生态系统中的创新解决方案和功能。
- **LLM ETL 栈的新发布**：LlamaIndex 宣布了两个专注于**结构化输出**以及异步 + 流式传输功能的重大发布，允许 LLM 以姓名和日期等结构化格式返回非结构化数据。
   - **LlamaExtract** 的引入促进了从非结构化文件中进行高效的 Schema 推断，使开发人员的数据处理变得更加简单。
- **新兴的 AI 数据工程师角色**：AI 领域正在出现一个新角色：**AI 数据工程师**，这对于通过确保可扩展且可靠的数据管理将上下文增强的 LLM 应用投入生产至关重要。
   - 该角色将数据工程技能与 AI 相结合，突显了其在现代 AI 实现中的必要性。
- **探索 RAG 评估方法**：一场网络研讨会将涵盖评估 RAG 系统的**五种不同方式**，演示使用 LLM 作为裁判的方法，从而增强对系统性能的理解。
   - 本次会议旨在让参与者掌握有效评估其 RAG 应用的技能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://t.co/Kx2j1B9XOV">结构化数据提取 - LlamaIndex</a>：未找到描述</li><li><a href="https://t.co/4Iu6PRfsd3">GitHub - run-llama/llama_extract</a>：通过在 GitHub 上创建账号来为 run-llama/llama_extract 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1266493434953334834)** (93 条消息🔥🔥): 

> - `使用自定义 Span 进行插桩 (Instrumentation)`
> - `Text-to-SQL Agent 示例`
> - `LlamaIndex 和 RAPTOR 的用法`
> - `文档转换为 Node`
> - `向量数据库中的 Embedding 存储` 


- **插桩中的自定义 Span 用法**：一位用户寻求在为 RAG 流水线创建具有特定属性的自定义 Span 方面的帮助，并对 Span 处理器（Handlers）和装饰器（Decorators）的用法表示困惑。
   - 他们分享了自己的自定义 Span 实现，但观察到没有任何 print 语句被触发，这表明事件处理可能存在问题。

- **在 LlamaIndex 中使用 Text-to-SQL**：成员们讨论了实现一个能够处理复杂查询的 Text-to-SQL 助手，并提供了配置 LlamaIndex NLP 查询引擎的示例。
   - 一个示例展示了如何利用 LlamaIndex 的功能设置工具并管理查询参数。
- **向量数据库中的 RAPTOR Pack**：一位用户询问如何将 RAPTOR packs 保存到 Pinecone，以及如何在不丢失之前数据的情况下向现有 pack 添加更多文档。
   - 社区澄清说，每个新文档都可以批量添加，但强调为了数据完整性，需要定期进行重新聚类。
- **将文档转换为基础节点 (Base Nodes)**：一名成员询问如何使用 LlamaIndex 将 Document 对象转换为基础节点，并获得了关于利用 `get_nodes_from_documents()` 方法的指导。
   - 分享的示例包括创建一个简单的节点解析器以及从指定目录加载文档。
- **存储摘要索引 (Summarization Index) 数据**：一位用户寻求关于高效存储摘要索引数据的建议（类似于向量索引数据），表示需要避免在每次运行时重新创建索引。
   - 讨论强调了管理 Pipeline 缓存的重要性，以确保所有处理过的目录都能被准确存储。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/">Embeddings - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/?h=graphrag),">GraphRAG Implementation with LlamaIndex - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform">LlamaIndex Community Office Hours</a>：对 LlamaIndex 有深入的问题或反馈？报名参加我们的社区办公时间！我们会联系您安排 15-30 分钟的 Zoom 会议进行交流。我们特别感兴趣...</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation">Instrumentation - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/huggingface/">Hugging Face LLMs - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/agent_runner/query_pipeline_agent/#setup-simple-retry-agent-pipeline-for-text-to-sql>))">Building an Agent around a Query Pipeline - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/retrievers/recurisve_retriever_nodes_braintrust/#load-data-setup>)">Recursive Retriever + Node References + Braintrust - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/retrievers/bm25_retriever/#load-data>)">BM25 Retriever - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1266652457212182578)** (2 条消息): 

> - `付费版 Llamaparse 的安全考量`
> - `命名实体的程序化去重` 


- **探索付费版 Llamaparse 的安全性**：一名成员询问了与免费版相比，使用付费版 **Llamaparse** 时潜在的**安全考量**。
   - 未提供明确答案，目前尚不清楚在安全性方面是否存在显著差异。
- **命名实体的快速去重技术**：另一名成员询问了如何在不使用复杂 **RAG** 架构的情况下，通过**程序化方式去重**命名实体列表。
   - 重点在于实现去重的速度和效率，而不需要复杂系统的开销。


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1266612620371300402)** (51 条消息🔥): 

> - `Open Interpreter 反馈`
> - `AI 与日常任务的集成`
> - `编程与学习`
> - `OI 中的自定义环境`
> - `Agent Zero 讨论` 


- **Open Interpreter 反馈循环**：用户对 **Open Interpreter** 作为工具的使用感受各异，一些人认为它在从 PDF 提取数据和翻译文本方面表现良好，而另一些人则对其实验性质表示谨慎。
   - 一位用户专门询问了它在查找和翻译中文科学文献等任务中的实用性，并获得了关于有效自定义指令的建议。
- **辅助日常运作的 AI 集成**：一名成员概述了他们在健康问题影响电脑使用能力方面的困扰，并表示有兴趣使用 **Open Interpreter** 进行语音控制任务。
   - 社区成员就将 OI 用于关键操作的相关风险提供了建议，并建议探索诸如语音转文本引擎之类的替代方案。

- **学习为 AI 使用而编程**：几位用户讨论了学习编程技能的价值，其中一位成员对自己的编程能力感到沮丧，但希望学习更多。
   - 有建议指出，编程知识可以提高对 AI 错误管理和问题解决方法的理解。
- **OI 中的自定义虚拟环境**：一位用户提出了在 **Open Interpreter** 中实现自定义 **venvs**（虚拟环境）的想法，这可以增强 GUI 可执行文件的功能。
   - 另一位用户强调了他们在该领域的进展，并表示可能需要协作来完善实现。
- **Agent Zero 与 OI 的讨论**：分享了对 **Agent Zero** 的兴趣，引用了一个演示及其实现 Agent 行为的方法，展示了社区对该类项目日益增长的兴趣。
   - 社区成员表达了探索这些技术如何协同工作以增强用户能力的愿望。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=t67Sn0RGK54&t=31s">Vimium - The Hacker&#39;s Browser</a>：Vimium 是一款 Google Chrome 扩展程序，本着 Vim 的精神提供用于导航和控制的键盘快捷键。在此处查看：https://chrome.googl...</li><li><a href="https://tenor.com/view/my-man-rick-and-morty-oh-yeah-mail-man-gif-16450197">My Man Rick And Morty GIF - My Man Rick And Morty Oh Yeah - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/e2b-dev/ai-artifacts">GitHub - e2b-dev/ai-artifacts: Hackable open-source version of Anthropic&#39;s AI Artifacts chat</a>：Anthropic 的 AI Artifacts 聊天的可黑客开源版本 - e2b-dev/ai-artifacts</li><li><a href="https://github.com/Soulter/hugging-chat-api">GitHub - Soulter/hugging-chat-api: HuggingChat Python API🤗</a>：HuggingChat Python API🤗。通过在 GitHub 上创建账户为 Soulter/hugging-chat-api 的开发做出贡献。</li><li><a href="https://github.com/blazzbyte/OpenInterpreterUI">GitHub - blazzbyte/OpenInterpreterUI: Simplify code execution with Open Interpreter UI Project with Streamlit. A user-friendly GUI for Python, JavaScript, and more. Pay-as-you-go, no subscriptions. Ideal for beginners.</a>：使用基于 Streamlit 的 Open Interpreter UI 项目简化代码执行。适用于 Python、JavaScript 等的易用 GUI。按需付费，无需订阅。初学者的理想选择。 - blazzbyte/Open...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/9124d2c34c444aa897df08befd553cf43c27b803/interpreter/terminal_interface/terminal_interface.py#L155">open-interpreter/interpreter/terminal_interface/terminal_interface.py at 9124d2c34c444aa897df08befd553cf43c27b803 · OpenInterpreter/open-interpreter</a>：计算机的自然语言界面。通过在 GitHub 上创建账户为 OpenInterpreter/open-interpreter 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1266810766317719723)** (5 条消息): 

> - `01 Desktop 的 Ubuntu 22.04`
> - `Wayland vs X11`
> - `OpenInterpreter 的虚拟环境` 


- **确认 01 Desktop 使用 Ubuntu 22.04**：成员们确认 **01 Desktop** 推荐的 Ubuntu 版本确实是 **22.04**，并提供了具体的配置说明。
   - 在此设置中，*X11* 比 *Wayland* 更受青睐，反映了用户的舒适度和熟悉程度。
- **用户偏好 X11 而非 Wayland**：一位成员表达了对 *Wayland* 缺乏热情，认为这可能是因为不习惯。
   - 社区目前对 *X11* 的青睐突显了关于桌面环境和用户体验的持续讨论。
- **在独立的虚拟环境中运行 OI 和 01**：有人询问是否应该在一个虚拟环境中运行 **OpenInterpreter (OI)**，而在另一个虚拟环境中运行 **01 (桌面版)**。
   - 寻求关于此做法的澄清，表明在设置说明方面存在困惑。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1266866376396767383)** (9 条消息🔥): 

> - `Agent Zero`
> - `Groq Mixture of Agents`
> - `Docker 集成` 


- **Agent Zero 令人印象深刻的演示**：[Agent Zero 的首次演示](https://www.youtube.com/watch?v=C9n8zFpaV3I)展示了内部 Vector DB、互联网搜索和 Agent 生成等功能。
   - 社区成员讨论了在 Docker 容器中执行等功能，并对与其工具的潜在集成表示好奇。
- **对 Agent Zero 潜力的好奇**：成员们对 Agent Zero 的框架表现出热情，注意到其潜在功能，包括内置的内存管理。

- 一位成员计划研究其设置，特别是使用 VSCode 配合 Docker 容器，以复制某些功能。
- **GitHub 上的 Groq Mixture of Agents**：分享了一个 [Groq Mixture of Agents](https://github.com/skapadia3214/groq-moa) 的 GitHub 仓库，强调了其开发目标。
   - 该项目承诺在基于 Agent 的交互方面做出贡献，并开放合作。
- **使用 Docker 和 LLM 进行调试**：一位成员成功在调试模式下运行了 Docker 镜像，使用了针对 Ollama 模型的 `chat_llm` 和 `utility_llm` 引用。
   - 他们强调了 `vscode/launch.json` 文件中的配置，这些配置简化了调试过程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=C9n8zFpaV3I">Agent Zero 🤖 首次演示</a>：Agent Zero 框架的首次公开演示。GitHub: https://github.com/frdel/agent-zero Discord: https://discord.gg/AQdRvSYX</li><li><a href="https://github.com/skapadia3214/groq-moa">GitHub - skapadia3214/groq-moa: 使用 Groq 的 Mixture of Agents</a>：使用 Groq 的 Mixture of Agents。可以通过在 GitHub 上创建账号为 skapadia3214/groq-moa 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1266734110311710822)** (43 messages🔥): 

> - `Turbo 量化`
> - `微调 Llama3`
> - `使用 QLoRA 进行部分层冻结`
> - `Tokenization 的挑战`
> - `Embedding 模型的策略` 


- **Turbo 模型可能使用了量化**：一位成员指出，使用 **'turbo'** 一词暗示使用了模型的**量化版本**。
   - *我注意到 fireworks 版本比 together ai 版本更好*，这表明了对不同实现的偏好。
- **讨论 Llama3 的微调策略**：一位成员对 **Llama3 的微调**程度表示兴趣，特别是关于引用链接和游戏统计数据方面。
   - 他们的目标是让模型能够有效地计算**护甲和武器统计数据**。
- **使用 QLoRA 进行部分层冻结受到审查**：关于使用 **QLoRA 配合部分层冻结**的可行性进行了讨论，建议在调整其他层的同时冻结中间层。
   - 有人担心 **PEFT** 是否能识别这些层，以及在没有事先软微调的情况下 **DPO** 是否有效。
- **ShareGPT 数据集的 Tokenization 问题**：一位成员在 ShareGPT 格式的数据集上遇到了 **Tokenization** 挑战，需要显式调整对话模板。
   - 使用 **FastChat 模板**的对话格式引发了关于为什么不将其设置为指令微调模型默认设置的疑问。
- **Embedding 模型的微调策略**：一位成员寻求**微调 Embedding 模型**的有效策略，并指出默认的 chromadb 设置效果不佳。
   - 他们询问是否有人有通过微调提高**文档选择**质量的成功方法。



**提到的链接**：<a href="https://axolotl-ai-cloud.github.io/axolotl/docs/config.html">配置选项 – Axolotl</a>：未找到描述

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1267138032038051851)** (7 messages): 

> - `4xH100 FSDP+QLoRA`
> - `CPU RAM 使用情况`
> - `数学重新实现`
> - `模型权重分布问题` 


- **4xH100 FSDP+QLoRA 减少了 CPU RAM 使用**：成员指出，集成 **4xH100 FSDP+QLoRA** 将显著减少 CPU RAM 的使用，使其效率更高。根据讨论，这是基于在多个 rank 上加载模型权重的对比得出的。
   - 此前，由于这种设置，**8 个 GPU** 需要 **~1.6TB** 的内存，但现在预计会变得容易处理得多。
- **关于 CPU RAM 比较的澄清**：有人询问 **CPU RAM 使用量**是相对于什么减少了，该成员澄清是指在各 rank 之间加载模型权重。这涉及到使用具有 8 个 GPU 的完整节点与仅使用 rank 0 的对比，表明内存需求减少了 **4 倍**。
   - FSDP+QLoRA 旨在优化峰值系统内存需求，无论设备数量多少。
- **关于数学重新实现的询问**：一位成员询问是否有人能帮助重新实现 **Twitter 链接**中分享的一个数学函数。这引发了关于如果以 **8k** 为增量进行计算时聚合方法的讨论。
   - 另一位成员询问是否应该通过**求和**来进行聚合。
- **关于 FSDP 处理的担忧**：关于 **FSDP** 是否充分管理了 GPU 之间的模型权重分布存在困惑。一位成员认为问题可能不在于 FSDP，而可能与 **Transformers** 有关。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/1266496987281227828)** (4 messages): 

> - `Atlantis-v0.1-12B`
> - `GGUF uploads`
> - `Nemo 12B finetune`
> - `Hugging Face upload speeds` 


- **Atlantis-v0.1-12B 现已发布**：[Atlantis-v0.1-12B](https://huggingface.co/invisietch/Atlantis-v0.1-12B) 已发布，被标记为含有可能有害的敏感内容。
   - 该新模型是针对 RP (角色扮演) 和创意写作的 **Nemo 12B** finetune 版本，预计很快会上传 GGUF 格式。
- **GGUF 上传正在进行但速度缓慢**：一位用户对该模型缺乏可用的 GGUF 表示沮丧，称 *still no ggufs 😦*。
   - 作为回应，一名成员确认主模型已重新链接，并且 formats 下的备选 GGUF 是可用的。
- **上传速度慢导致延迟**：开发者分享称他们正经历极慢的上传速度，表示 *HF has decided I need to be uploading this 70GB folder at 120k/s*。
   - 这种缓慢的速度导致上传时间延长，使得模型完全公开访问出现了明显的延迟。



**提及的链接**：<a href="https://huggingface.co/invisietch/Atlantis-v0.1-12B">invisietch/Atlantis-v0.1-12B · Hugging Face</a>：未找到描述

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1267283417569951795)** (1 messages): 

> - `Operation Athena`
> - `Collaborative reasoning tasks`
> - `Dataset diversity` 


- **Operation Athena 启动推理任务数据库**：作为 [Operation Athena](https://operation-athena.repleteai.com/) 的一部分，一个专注于 LLM **推理任务 (reasoning tasks)** 的新数据库已经组建完成，允许用户贡献自己的任务。
   - 该倡议由 **Nous Research** 支持，旨在通过反映人类经验的多样化数据集来增强 AI 的理解能力。
- **征集任务贡献**：该倡议鼓励社区参与，邀请向数据库贡献内容，以建立一个全面的 AI 推理任务列表。
   - 这种方法旨在保持 **数据集多样性 (dataset diversity)**，这对于提高模型在 **现实世界应用 (real-world applications)** 中的性能至关重要。
- **源自 Nous Research 的原创概念**：Operation Athena 的基础源于 [Nous Research](https://github.com/NousResearch/Open-Reasoning-Tasks/tree/main) 发布的工作，他们发起了策划推理任务的想法。
   - 截至 **2024 年 7 月 28 日**，该数据库最主要的贡献来自于其 [文档](https://github.com/NousResearch/Open-Reasoning-Tasks/blob/main/tasks.md) 中详述的现有资源。



**提及的链接**：<a href="https://operation-athena.repleteai.com/">Operation Athena</a>：未找到描述

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1266934038589866079)** (4 messages): 

> - `early_stopping_patience`
> - `Axolotl configurations` 


- **理解 Axolotl 中的 early_stopping_patience**：在 Axolotl 中，`early_stopping_patience: 3` 参数表示如果验证指标连续 **三个 epoch**（而非 sequence）没有改善，则停止训练。
   - *Early stopping* (早停) 通过在性能不再提升时停止训练来帮助防止 overfitting (过拟合)，是模型训练配置中的关键部分。
- **在训练中配置早停**：展示了一个 Axolotl 中早停的 YAML 配置示例，其中 `early_stopping_patience: 3` 强调了其在监控定义指标方面的作用。
   - 该配置确保如果验证集上的性能没有提升，训练不会超过 **三个 epoch**。



**提及的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=dbdc3c04-77e2-483b-b215-78fd6732d625)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1266749658936377406)** (48 messages🔥): 

> - `Open Source Contributions to LangChain`
> - `Ollama API for Tool Calling`
> - `ConversationBufferMemory in LangGraph`
> - `Creating Flowcharts for RAG`
> - `LangChain Tutorial Issues` 


- **对 LangChain 的开源贡献**：一位成员表达了对参与 LangChain 贡献指南的兴趣，促使其他人分享了包括 [贡献指南](https://python.langchain.com/v0.2/docs/contributing/) 在内的资源。建议包括改进文档、代码和集成作为贡献方式。
   - 对于初学者，一位成员建议阅读 [设置指南](https://python.langchain.com/v0.2/docs/contributing/code/setup/) 以了解本地仓库的交互。

- **用于 Tool Calling 的 Ollama API**：成员们讨论了使用 Ollama API 创建 Agent 的效率，其中一位报告称使用 `ChatOllama` 比 `OllamaFunctions` 功能更好。他们指出，在遵循 LangChain 教程的示例时，它的表现更佳。
   - 有人提到之前的 API 在基础教程中会出现崩溃问题，特别是涉及 Tavily 和天气示例的情况。
- **LangGraph 中的 ConversationBufferMemory**：一位成员寻求关于如何在 `ConversationBufferMemory` 中使用 `save_context` 方法的说明，询问如何为 `HumanMessage` 和 `AIMessage` 等各种消息类型构建输入和输出结构。其他人指出 `ConversationBufferMemory` 缺乏关于线程安全的明确文档。
   - 提供的建议指出，为了有效处理不同的消息类型，必须仔细构建输入和输出的结构。
- **为 RAG 创建流程图**：讨论中包括了使用 Mermaid 创建流程图的建议，一位成员分享了来自 LangChain 文档的代码片段。有人建议这为可视化工作流和流程提供了很好的生产价值。
   - 一位成员分享了一个比较不同 RAG 框架的 GitHub 项目，鼓励其他人查看以获取更多关于 RAG 应用的见解。
- **LangChain 教程问题**：一位初学者报告在尝试遵循 LangChain RAG 教程时遇到了 `ConnectError`。建议复现官方教程，以更好地掌握功能并排查问题。
   - 有人对 JS 快速入门中的多次 LLM 调用表示担忧，暗示在处理应用程序内的 LLM 交互时可能存在效率低下或误解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<a href="https://python.langchain.com/v0.2/docs/contributing/">欢迎贡献者 | 🦜️🔗 LangChain</a>：你好！感谢你对贡献 LangChain 感兴趣。</li><li><a href="https://python.langchain.com/v0.2/docs/contributing/code/setup/">设置 | 🦜️🔗 LangChain</a>：本指南介绍了如何在本地运行仓库并提交你的第一行代码。</li><li><a href="https://stackoverflow.com/questions/4974238/javascript-equivalent-of-pythons-format-function">Python format() 函数在 JavaScript 中的等价实现？</a>：Python 有一个非常优雅的函数可以将：
bar1 = 'foobar'
bar2 = 'jumped'
bar3 = 'dog'

foo = 'The lazy ' + bar3 + ' &...</li><li><a href="https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/#define-graph">客户支持</a>：未找到描述</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/agents/">构建 Agent | 🦜️🔗 LangChain</a>：本指南假设你已熟悉以下概念：</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/rag/">构建检索增强生成 (RAG) 应用 | 🦜️🔗 LangChain</a>：LLM 赋能的最强大应用之一是复杂的问答 (Q&A) 聊天机器人。这些应用可以针对特定的源信息回答问题。这些 ...</li><li><a href="https://github.com/oztrkoguz/RAG-Framework-Evaluation">GitHub - oztrkoguz/RAG-Framework-Evaluation：该项目旨在从速度和性能方面比较不同的检索增强生成 (RAG) 框架。</a>：该项目旨在从速度和性能方面比较不同的检索增强生成 (RAG) 框架。- oztrkoguz/RAG-Framework-Evaluation</li><li><a href="https://github.com/langchain-ai/langchain/issues/16651">从这里开始：欢迎来到 LangChain！· Issue #16651 · langchain-ai/langchain</a>：欢迎来到 LangChain 仓库！本仓库包含的内容：请仅针对本仓库包含的包提交 Issue、PR 和 Discussion：langchain python 包、langchain-core python 包...</li><li><a href="https://github.com/langchain-ai/langchain/issues/3536>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/2256>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/17867>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/11734>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/6761>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1266857840342204437)** (7 条消息): 

> - `Merlinn AI 值班 Agent`
> - `用于用户获取的 AI Copilot`
> - `Langchain 食谱机器人`
> - `知识蒸馏趋势`
> - `AI Analyst Builder 发布` 


- **Merlinn AI 值班 Agent 简化故障排除**：团队推出了 [Merlinn](https://github.com/merlinn-co/merlinn)，这是一个开源的 AI 值班 Agent，通过与 DataDog 和 PagerDuty 等工具集成，协助排除生产环境故障。
   - 他们诚邀反馈，并鼓励用户在 [GitHub 仓库](https://github.com/merlinn-co/merlinn)点亮 Star 以支持该项目。
- **AI Copilot 简化用户推广**：一个全新的 [AI copilot](https://hypespot.pro) 已发布，它通过为相关对话建议评论，帮助用户在 Twitter 上轻松推广自己的项目。
   - 该工具旨在帮助个人以最小的努力提高其产品的曝光度。
- **轻松与 Notion 数据库对话**：[Kenzic](https://github.com/kenzic/langchain-recipe-bot) 介绍了一个简单的应用程序，用于与 Notion 数据库进行对话，详情见 Medium 教程。
   - GitHub 仓库包含了实现所需的资源。

- **Knowledge Distillation 进展讨论**：[Lightly 的最新博客文章](https://www.lightly.ai/post/knowledge-distillation-trends) 涵盖了 Knowledge Distillation 的趋势，强调了从大型模型衍生出的小型模型所带来的性能提升。
   - 由 Hinton 最初提出的这一概念旨在最小化 KL divergence，从而提高小型模型的效率。
- **AI Analyst Builder 在 Product Hunt 上线**：Datrics 推出了 [AI Analyst Builder](https://bit.ly/4dhK6Km)，这是一个用于创建自定义 AI analysts 的无代码工具，并寻求 Product Hunt 社区的支持。
   - 鼓励用户访问该页面并提供反馈，以持续改进该工具。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.lightly.ai/post/knowledge-distillation-trends">Knowledge Distillation Trends</a>: 近期 Knowledge Distillation 策略概述，以及如何将其与以 Masked Autoencoders 为重点的 Self-Supervised Learning 结合使用</li><li><a href="https://mukullight.github.io/docu/">Home</a>: 无</li><li><a href="https://github.com/kenzic/langchain-recipe-bot">GitHub - kenzic/langchain-recipe-bot: Repo for Medium tutorial &quot;Talk to Your Notion Database with LangChain.js&quot;</a>: Medium 教程“使用 LangChain.js 与你的 Notion 数据库对话”的代码库 - kenzic/langchain-recipe-bot</li><li><a href="https://bit.ly/4dhK6Km"> Datrics AI Analyst Builder - Your custom GenAI solution for analytics and reporting | Product Hunt</a>: AI Analyst Builder 使团队能够无需编程即可创建自定义 AI analysts。这些分析师通过类似 ChatGPT 的聊天界面回答数据问题。针对特定的业务流程和数据进行定制...</li><li><a href="https://github.com/merlinn-co/merlinn">GitHub - merlinn-co/merlinn: Open source AI on-call developer 🧙‍♂️ Get relevant context &amp; root cause analysis in seconds about production incidents and make on-call engineers 10x better 🏎️</a>: 开源 AI 值班开发人员 🧙‍♂️ 在几秒钟内获取有关生产事故的相关上下文和根因分析，让值班工程师效率提升 10 倍 🏎️ - merlinn-co/merlinn</li><li><a href="https://hypespot.pro">Hypespot</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1266689548528779295)** (30 messages🔥): 

> - `按 API Key 计费`
> - `Multi-Agent Systems 框架`
> - `API 问题`
> - `Prompt Tuner Beta 版发布` 


- **按 API Key 计费的挑战**：讨论中提到了按 API Key 进行独立计费的需求，成员们探索了如使用 Middleware 来分别管理每个 Key 的成本等潜在解决方案。
   - 参与者表示沮丧，并指出目前还没有能够有效跟踪此类使用的系统。
- **Multi-Agent Systems 的最佳框架**：成员们建议查看 [来自 LangChain 的 LangGraph](https://docs.langchain.com/docs)，该框架因其云端能力和构建 Multi-Agent Systems 的可定制性而受到赞誉。
   - 此外，还提到了 Cohere 的 API 提供了广泛的多步（multi-step）和单步（single-step）工具使用功能，增强了 Agent 的能力。
- **错误代码 503 的 API 停机**：一位用户报告了错误代码为 503 的 API 停机问题，并因状态页面无法访问而难以检查状态。
   - 另一位成员向社区保证，他们正在内部努力解决导致停机的问题。
- **Prompt Tuner Beta 功能发布**：有人询问仪表板上“Prompt Tuner”Beta 功能的可用性，成员们确认了其最近的引入。
   - 用户普遍表示有兴趣更好地了解该功能对 API 使用的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://status.cohere.com/>">incident.io - Status pages</a>: 未找到描述</li><li><a href="https://docs.cohere.com/docs/multi-step-tool-use">Multi-step Tool Use (Agents)</a>: 未找到描述</li><li><a href="https://docs.cohere.com/docs/tool-use">Tool Use with Cohere's Models - Cohere Docs</a>: 未找到描述</li><li><a href="https://docs.cohere.com/docs/implementing-a-multi-step-agent-with-langchain">Implementing a Multi-Step Agent with Langchain</a>: 未找到描述</li><li><a href="https://docs.cohere.com/reference/chat">Chat</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1267031847037177886)** (7 messages): 

> - `Cohere 与 Oracle APEX`
> - `Cohere API 性能问题`
> - `Cohere 运行状态` 


- **关于 Cohere 与 Oracle APEX 的咨询**：一位用户询问是否有人在将 **Cohere** 与 **Oracle APEX** 结合使用，寻求有关该集成的见解和经验。

- **Cohere API 经历变慢**：多名用户报告了 **Cohere Reranker API** 的问题，指出其突然变慢并出现故障。
   - 有人承认这是一种罕见情况，并分享了一条[错误信息](https://status.cohere.com/)，表明团队正在调查该问题。
- **Cohere 服务状态恢复正常**：**Cohere** 宣布已从之前的问题中恢复，确认所有系统均已完全正常运行。
   - 状态更新强调了端点 **99.67% 的正常运行时间**，并发布了令人放心的消息，称目前没有持续存在的问题影响其系统。



**提到的链接**：<a href="https://status.cohere.com/">Cohere Status Page Status</a>：Cohere 状态页面的最新服务状态

  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1267543982565490741)** (17 messages🔥): 

> - `Cohere API 的优势`
> - `AI 公司的创新`
> - `网页浏览 API 的使用`
> - `搜索引擎功能`
> - `AI 领域的炒作周期` 


- **Cohere API 以可靠性著称**：一名成员指出，**Cohere API** 是他们合作过的唯一一个没有经历过停机的 API，称其为**最佳企业级选择之一**。
   - 另一位成员幽默地强调了在为 OpenAI 工作的情况下偏爱 Cohere 的含义。
- **创新 vs. AI 产品中的相似性**：随着各公司发布新产品，一名成员质疑哪些是真正的**创新**，而哪些只是现有产品的**迭代**。
   - 这种情绪反映了更广泛的行业讨论，即持续不断的发布究竟是真正的突破，还是当前**炒作周期**的一部分。
- **在聊天中使用网页浏览工具**：成员们讨论了利用集成在 **Cohere 聊天界面**和 API 中的网页搜索工具来快速获取信息的能力。
   - 一名成员成功创建了一个利用此功能的机器人，表明其在功能上类似于**搜索引擎**。
- **对新实现的兴奋**：一位用户表达了在 Cohere 面试期间使用新工具的热情，极大地鼓励了协作测试。
   - 俏皮的语气暗示了一种轻松的方式来共同探索这些新功能，即使是非技术用户也是如此。
- **对 AI 当前格局的看法**：一名成员评论了行业的**炒作周期**，同时强调他们专注于为企业从 AI 模型中获取实质价值。
   - 这一言论反映了在不断演变的 AI 领域中，将有效工具与纯粹的营销噪音区分开来的挑战。


  

---



### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1266484549550866605)** (2 messages): 

> - `博客文章` 


- **博客文章引用**：一名成员提到了一篇**博客文章**，但未提供任何相关的具体细节或链接。
   - 另一名成员发表了轻松的评论，说 *summoned 🪄 ☁️🧍‍♂️☁️*，可能是对提到博客文章的回应。
- **用户与博客文章的互动**：围绕提到**博客文章**的互动包括其他成员的俏皮参与。
   - *summoned 🪄 ☁️🧍‍♂️☁️* 这一短语暗示了对话中轻松或幽默的氛围。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1266505187439280240)** (26 messages🔥): 

> - `GPT-4o Mini`
> - `LMSYS 排名算法`
> - `Chatbot Arena 中的格式化`
> - `角色扮演与创意写作模型`
> - `Zuckerberg 在 SIGGRAPH 上的评论` 


- **GPT-4o Mini 占据领先地位**：[GPT-4o Mini](https://huggingface.co/spaces/lmsys/gpt-4o-mini_battles) 的推出被视为 Chatbot Arena 的重大变化，据称增强了交互体验。
   - 有人建议该模型不仅关乎性能，还作为一种透明度工具来验证较弱的模型。
- **LMSYS 并非尖端的排名工具**：人们对 LMSYS 持怀疑态度，评论称它只是验证现有模型，而不是领先的排名算法。
   - 一位用户强调，该模型的示例展示了随机性，指出简单的问题无法有效评估模型性能。
- **格式化产生影响**：讨论强调了 Chatbot 回复中格式化的有效使用，特别是对列表和 Markdown 功能的掌握，能够吸引用户。
   - 一名成员幽默地指出他们偏好使用层级化的项目符号，将其比作人类语言中广泛存在的偏好。
- **对 AI 角色扮演的反感**：一位用户表达了他们不愿与 AI 模型进行角色扮演或创意写作，表示他们更喜欢更具实用性的用途。

- 对话反映了在应用偏好上的分歧，一些人拥护创意用途，而另一些人则表示抵制。
- **扎克伯格在 SIGGRAPH 上的坦率言论**：扎克伯格因发表非正式言论而受到关注，包括在 SIGGRAPH 与 Jensen 一起爆粗口，释放出一种更加轻松的氛围。
   - 这种打趣包含了开玩笑式的请求，展示了行业领袖之间轻松的互动。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/lmsys/gpt-4o-mini_battles">Gpt-4o-mini Battles - a Hugging Face Space by lmsys</a>：未找到描述</li><li><a href="https://x.com/tszzl/status/1779608670181171504">来自 roon (@tszzl) 的推文</a>：不幸的是，人类语言偏好的全局最优解是列表和对联诗。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1266488321178337412)** (13 messages🔥): 

> - `RBR 论文批评`
> - `对 SELF-ALIGN 论文的好奇`
> - `Apple Intelligence Foundation 论文`
> - `RL 命名方案`
> - `iTeC 细节` 


- **RBR 论文掩盖了复杂性**：一位成员表示，**RBR 论文**解释了显而易见的部分，却忽略了更复杂的问题，特别是其对良性请求中包含危险内容的简要提及。
   - 他们强调，虽然筛选出像“请给我管状炸弹”这样明确的威胁似乎很简单，但其中的细微差别被掩盖了。
- **对 SELF-ALIGN 论文的兴趣**：另一位成员对 **SELF-ALIGN 论文**表示好奇，该论文的主题是“从零开始且仅需极少人工监督的原则驱动型语言模型自我对齐”。
   - 他们指出，这可能与关于对齐技术的 **SALMON** 和 **RBR** 讨论都有关。
- **关于 Apple AI 论文的讨论**：成员们对 **Apple Intelligence Foundation** 论文做出了反应，指出它详细介绍了 **RLHF** 及其指令层级的各个方面，但对其代码库（repository）的看法褒贬不一。
   - 一位成员表示决定将其打印出来，以评估它对他们关于 **RLHF** 观点的响。
- **对 RL 命名方案的批评**：一位成员对 **Reinforcement Learning (RL)** 研究人员使用的命名方案的奇特性发表了评论，表达了一种难以置信的感觉。
   - 他们的反应强调了对 RL 社区内所采用术语的更广泛的困惑情绪。
- **对 iTeC 的兴奋**：简要提到了 **iTeC**，一位成员注意到其撰写极其详尽。
   - 这引发了对话中关于该论文内容和潜在影响的一些兴奋。


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1266571885810942072)** (3 messages): 

> - `Moondream2 hack`
> - `Gold Retriever 工具`
> - `Databricks 与 DSPy` 


- **Moondream2 获得了结构化图像响应的 hack**：一位成员透露，他们构建了一个结合了 [Moondream2](https://x.com/ErikKaum/status/1787451553319621077) 和 [OutlinesOSS](https://github.com/OutlinesOSS) 的 hack，允许用户针对图像提问并接收结构化响应。
   - 该方法劫持了 Moondream2 中的文本模型，同时通过 Outlines 启用嵌入处理，有望简化用户体验。
- **为 ChatGPT 推出 Gold Retriever**：[Gold Retriever](https://jina.ai/news/gold-retriever-let-chatgpt-talk-to-your-data/) 是 Jina 开发的一个开源工具，它增强了 ChatGPT 集成个性化和实时数据的能力，解决了之前的局限性。
   - *用户渴望量身定制的 AI 交互*，Gold Retriever 旨在提供更好的用户特定数据访问，同时应对知识截止日期的挑战。
- **Databricks 看到了 DSPy 的潜力**：Databricks 分享的一篇文章强调了对 DSPy 日益增长的认可，认为它是组织执行其数据策略的卓越工具。
   - 该消息邀请了关于各种工具的讨论，标志着像 DSPy 这样的创新正在行业中获得关注。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://jina.ai/news/gold-retriever-let-chatgpt-talk-to-your-data/">Gold Retriever：让 ChatGPT 与你的数据对话</a>：通过 Gold Retriever，你只需几个步骤即可轻松让 ChatGPT 存储、检索你的数据并进行推理。</li><li><a href="https://x.com/ErikKaum/status/1787451553319621077">来自 Erik Kaunismäki (@ErikKaum) 的推文</a>：图像 ➡️ JSON。我构建了一个结合了 @vikhyatk 的 moondream2 和 @OutlinesOSS 的 hack。现在你可以打开一张图片，询问关于图片的问题，并获得一个保证遵循...的响应。
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1266539054036553938)** (2 messages):

> - `AI Agent 进展`
> - `Transformers 与组合性` 


- **AI Agent 进展综述**：最近的一篇 [综述论文](https://arxiv.org/abs/2404.11584) 探讨了 AI Agent 实现方面的进展，重点关注增强的推理、规划和工具执行能力。
   - 它传达了现有系统的当前能力和局限性，同时为未来的 AI Agent 设计提出了关键考虑因素，包括领导力影响和沟通风格。
- **AI 中的 Transformers：提出的基本问题**：一篇 [值得阅读](https://www.answer.ai/posts/2024-07-25-transformers-as-matchers.html) 的博客文章强调了对 Transformer 模型在复杂任务（特别是乘法）上表现的研究，这与关于其学习能力的更深层次问题相关联。
   - 它强调像 **Claude** 或 **GPT-4** 这样的模型产生的输出能够令人信服地模拟推理，引发了关于它们在各个领域解决复杂问题能力的至关重要的讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.11584">The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey</a>：这篇综述论文探讨了 AI Agent 实现的最新进展，重点关注它们实现需要增强推理、规划和工具执行能力的复杂目标的能力...</li><li><a href="https://www.answer.ai/posts/2024-07-25-transformers-as-matchers.html">Faith and Fate: Transformers as fuzzy pattern matchers – Answer.AI</a>：类 GPT 模型在思考吗？尚不清楚。但 Faith and Fate 论文 (Dziri, 2023) 指出它们通常“仅仅”是在进行模式匹配。
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1266811152386756631)** (13 条消息🔥): 

> - `Mixture of Agents 优化`
> - `不带编译流水线的 DSPy`
> - `托管大规模预训练数据`
> - `具备对话历史感知能力的 Agent` 


- **探索 Mixture of Agents 优化**：一位成员提出了为 DSPy 使用 Mixture of Agents 优化器的想法，建议优化的一个层面涉及为系统选择参数和模型。
   - 他们引用了一篇 [相关论文](https://arxiv.org/abs/2406.04692)，讨论了利用多个 LLM 来改进响应，并将其方法与神经网络结构进行了比较。
- **在没有标注数据的情况下使用 DSPy**：一位成员询问了在没有编译流水线的情况下利用 DSPy 的情况，质疑其在没有标注数据的场景下的实用性。
   - 另一位成员确认 DSPy 确实可以增强应用程序，特别是对于寻求更好 Prompt 的 RAG 系统。
- **托管 2PB 数据**：一位成员就为医疗 LLM 托管 2PB 预训练数据寻求建议，并提到与 Linux Foundation 就中低收入国家的数据使用进行的讨论。
   - 他们分享了之前在 ClinicalBERT 上的工作，并链接到了一个关于数据托管解决方案的 [GitHub issue](https://github.com/ClickHouse/ClickHouse/issues/67296)。
- **LLM 数据的免费托管选项**：针对关于托管大型数据集的询问，一位成员建议 Hugging Face 可能允许免费托管且没有太多限制。
   - 这对于处理大量医疗数据的项目可能非常有益。
- **创建具备对话历史感知能力的 Agent**：一位成员询问了创建维持对话历史的 Agent 的示例。
   - 作为回复，另一位成员指出，以前用户必须手动处理聊天历史记录，并引用了一个示例作为参考。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<a href="https://github.com/stanfordnlp/dspy/blob/main/examples/agents/multi_agent.ipynb">dspy/examples/agents/multi_agent.ipynb at main · stanfordnlp/dspy</a>: DSPy: 编程而非提示基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://arxiv.org/abs/2406.04692">Mixture-of-Agents Enhances Large Language Model Capabilities</a>: 大语言模型 (LLMs) 的最新进展展示了其在自然语言理解和生成任务中的强大能力。随着 LLMs 数量的增加，如何利用集体...</li><li><a href="https://docs.google.com/document/d/10j4eTkB_0c6BIMRgUY02L2wWvFWAHuCpkiMZ6dotnkA/edit?usp=sharing">ClinicalBERT Technical Charter Draft 11-29-2022</a>: ClinicalBERT（LF Projects, LLC 的一系列项目）的技术章程（“章程”）。于 ___________ 通过。本章程规定了技术贡献的责任和程序...</li><li><a href="https://github.com/ClickHouse/ClickHouse/issues/67296">Integration with content delivery network for incremental static regeneration · Issue #67296 · ClickHouse/ClickHouse</a>: （您不必严格遵守此表格）公司或项目名称 在此处填写您的公司名称或项目描述 One Fact Foundation 使用案例 清楚简洁地描述什么是...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1266497036883333171)** (17 条消息🔥): 

> - `storm.py error`
> - `dspy version update`
> - `Dropbase integration`
> - `Mine Tuning methodology`
> - `DSPy project showcases` 


- **storm.py 遇到 AttributeError**: 一位用户报告在尝试执行 **storm.py** 时出现 **AttributeError**，原因是 **dspy** 模块中存在未定义的属性。
   - 另一位成员建议更新到较新版本的 **dspy** 以解决该问题。
- **更新 dspy 以支持 Claude 的步骤**: 分享了更新 **dspy** 库的详细步骤，包括修改 **__init__.py** 文件以包含 `Claude = dsp.Claude`。
   - 此更改将使用户能够直接在代码中使用 **Claude** 功能。
- **Dropbase 集成计划**: 一位成员表达了将 **Dropbase** 与 **dspy** 集成的计划，旨在为各种工作流创建一个可扩展的 GUI。
   - 分享了关于如何利用 **Dropbase** 加快 Web App 开发的资源。
- **展示各种 DSPy 项目**: 分享了多个创新 **DSPy** 项目的 GitHub 链接，包括 **Mine Tuning** 等方法论。
   - 鼓励成员探索这些项目以获取灵感，并可能改进其实施方案。
- **询问 DSPy 项目的有效性**: 有人询问是否有实际项目利用 **DSPy** 并证明其工作流得到了显著增强。
   - 讨论强调了之前分享的各种项目，并呼吁提供具体改进的案例。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://github.com/DropbaseHQ/dropbase">GitHub - DropbaseHQ/dropbase: Dropbase 帮助开发者利用 AI 更快地构建和原型化 Web 应用。Dropbase 是本地优先且自托管的。</a>: Dropbase 帮助开发者利用 AI 更快地构建和原型化 Web 应用。Dropbase 是本地优先且自托管的。 - DropbaseHQ/dropbase</li><li><a href="https://www.dropbase.io/">Dropbase AI | 利用 AI 构建后台管理软件</a>: Dropbase 是一个基于 Prompt 的开发者平台，用于快速且无痛地构建 Web 应用和后台运营软件。告别低代码/无代码带来的烦恼。</li><li><a href="https://github.com/stanfordnlp/dsp">GitHub - stanfordnlp/dspy: DSPy: 用于编程（而非 Prompting）基础模型的框架</a>: DSPy: 用于编程（而非 Prompting）基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://github.com/rawwerks/MineTuning/tree/main">GitHub - rawwerks/MineTuning: Mine-tuning 是一种同步人类与 AI 注意力的方法论。</a>: Mine-tuning 是一种同步人类与 AI 注意力的方法论。 - rawwerks/MineTuning</li><li><a href="https://github.com/seanchatmangpt/dspygen">GitHub - seanchatmangpt/dspygen: 一个为 DSPy (Demonstrate, Search, Predict) 项目提供的 Ruby on Rails 风格框架，适用于 GPT、BERT 和 LLama 等语言模型。</a>: 一个为 DSPy (Demonstrate, Search, Predict) 项目提供的 Ruby on Rails 风格框架，适用于 GPT、BERT 和 LLama 等语言模型。 - seanchatmangpt/dspygen</li><li><a href="https://github.com/jmanhype/dspy-self-discover-framework">GitHub - jmanhype/dspy-self-discover-framework: 利用 DSPy 进行 AI 驱动的任务理解和方案生成，Self-Discover Framework 通过推理和代码生成实现自动化问题解决。</a>: 利用 DSPy 进行 AI 驱动的任务理解和方案生成，Self-Discover Framework 通过推理和代码生成实现自动化问题解决。 - jmanhype/dspy-self-discover-...</li><li><a href="https://github.com/chrisammon3000/dspy-neo4j-knowledge-graph">GitHub - chrisammon3000/dspy-neo4j-knowledge-graph: 使用 DSPy 和 Neo4j 从文本中进行 LLM 驱动的自动化知识图谱构建。</a>: 使用 DSPy 和 Neo4j 从文本中进行 LLM 驱动的自动化知识图谱构建。 - chrisammon3000/dspy-neo4j-knowledge-graph</li><li><a href="https://github.com/jmanhype/DSPy-Multi-Document-Agents">GitHub - jmanhype/DSPy-Multi-Document-Agents: 一种用于智能文档处理的高级分布式知识网络，具有多文档 Agent、优化的查询处理和语义理解功能。</a>: 一种用于智能文档处理的高级分布式知识网络，具有多文档 Agent、优化的查询处理和语义理解功能。 - jmanhype/DSPy-Multi-Document-A...</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI: 可编程的神经符号 AGI，允许你使用基于图的 Prompt Programming 来对其行为进行编程：适用于希望 AI 表现符合预期的人群</a>: 可编程的神经符号 AGI，允许你使用基于图的 Prompt Programming 来对其行为进行编程：适用于希望 AI 表现符合预期的人群 - SynaLinks/HybridAGI</li><li><a href="https://github.com/RamXX/FSM-Workflow">GitHub - RamXX/FSM-Workflow: 适用于 Python 的轻量级、异步友好且具有状态持久化的工作流系统</a>: 适用于 Python 的轻量级、异步友好且具有状态持久化的工作流系统 - RamXX/FSM-Workflow</li><li><a href="https://github.com/jmanhype/Storm">GitHub - jmanhype/Storm</a>: 通过在 GitHub 上创建账号来为 jmanhype/Storm 的开发做出贡献。</li><li><a href="https://github.com/jmanhype/Storm/blob/main/storm.py">Storm/storm.py at main · jmanhype/Storm</a>: 通过在 GitHub 上创建账号来为 jmanhype/Storm 的开发做出贡献。</li><li><a href="https://storm.genie.stanford.edu/">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1266477396257800212)** (10 messages🔥): 

> - `OpenCL Out of Memory Error`
> - `周一会议记录`
> - `ShapeTracker 悬赏任务 (Bounties)`
> - `Lean 翻译讨论` 


- **OpenCL 显存溢出 (Out of Memory) 错误改进**：一名成员建议改进 OpenCL 中的 **out of memory error**，并链接了 **tyoc213** 提交的相关 [GitHub Pull Request](https://github.com/tinygrad/tinygrad/pull/5792/files)。
   - *tyoc213* 指出了与 OpenCL 相关的错误处理潜在解决方案。
- **周一会议亮点**：周一会议讨论了多项更新，包括移除 **UNMUL** 和 **MERGE**，以及引入 [HCQ Runtime 文档](https://docs.tinygrad.org/developer/hcq/)。
   - 其他主题包括与 **MLPerf** 基准测试相关的 **悬赏任务 (bounties)**，以及 **卷积反向融合 (conv backward fusing)** 和 **调度器 (scheduler)** 优化的进展。

- **对 ShapeTracker Bounty 的兴趣**：一位成员对专注于 **在 Lean 中两个任意 ShapeTracker 的可合并性** 的 Bounty 表示了兴趣，并询问了该任务的范围。
   - 该成员提到了之前关于此 Bounty 的讨论，并就其价值与奖励的对比进行了交流。
- **文档的 Lean 翻译**：有人询问该 Bounty 是否涉及将文档翻译成 **Lean**，并质疑此类任务的报酬。
   - 另一位成员指出了之前关于 **Lean** 的讨论，并建议聊天记录中可能已经有了答案。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/5792/files">retrieve defined opencl error codes by tyoc213 · Pull Request #5792 · tinygrad/tinygrad</a>：未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/pull/2218/files)",">view.reshape without symbolic by sahamrit · Pull Request #2218 · tinygrad/tinygrad</a>：@chenyuxyz 这是你之前对没有 symbolic 的 reshape 的尝试。我分析了你更改后时间增加的原因是由于缓存未命中。以下是一些细节，好的一面是你的更改减少了...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1266685261618872350)** (21 条消息🔥): 

> - `PR 期间 NLL loss 出现错误`
> - `理解 nn.Embedding 梯度`
> - `tinygrad 中的 Disk 设备`
> - `改进错误处理的 PR`
> - `使用 tinygrad 进行时间序列分析` 


- **解决 tinygrad PR 中的 NLL Loss 错误**：一位用户分享了一个与添加 `nll_loss` 相关的错误，指出返回的 Tensor 缺少梯度，导致 PR 失败。
   - 另一位成员提到，Loss 计算中使用的某些操作（如 CMPNE）是不可微的。
- **澄清 nn.Embedding 的梯度**：一位用户寻求关于 `nn.Embedding` 梯度的帮助，在他们的模型中遇到了“Tensor 没有梯度”的错误。
   - 回复澄清了对于索引操作，`requires_grad=True` 是不必要的，以避免梯度问题。
- **Disk 设备功能说明**：一位用户询问了 tinygrad 中的 Disk 设备，质疑其在计算中的作用。
   - 解释称 Disk 设备用于 Tensor 内存映射，主要用于传输数据，而非用于计算操作。
- **增强错误处理的建议**：一位用户建议 tinygrad 不应允许 Tensor 运行在像 Disk 这样的非计算后端上，并寻求更好的错误提示信息。
   - 成员们讨论了处理此类情况的必要性，并同意提交一个 Pull Request 来改进这一行为。
- **使用 tinygrad 进行时间序列分析**：一位用户询问 tinygrad 是否可以应用于时间序列生理特征提取和可视化，理由是 Matlab 的性能较慢。
   - 这一咨询表明了利用 tinygrad 的能力在数据分析中实现更高效计算的兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.tinygrad.org/runtime/">Runtime - tinygrad docs</a>：未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/issues/5769)">Issues · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - Issues · tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/95dda8dadf2970888fc8f494b83a0124eb614aa5/tinygrad/nn/state.py#L13-L58">tinygrad/tinygrad/nn/state.py at 95dda8dadf2970888fc8f494b83a0124eb614aa5 · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://gist.github.com/airpods69/2992fc80703f8e15a55d44b3455d9620">Error log when testing cross_entropy loss function</a>：测试 cross_entropy 损失函数时的错误日志 - gist:2992fc80703f8e15a55d44b3455d9620</li><li><a href="https://github.com/tinygrad/tinygrad/pull/5752">Addition of nll_loss and cross_entropy to tensor.py by airpods69 · Pull Request #5752 · tinygrad/tinygrad</a>：tensor：添加了 nll_loss 和 cross_entropy test_ops：添加了 nll_loss 和 cross_entropy 的测试。此 PR 将 negative_log_likelihood 和 cross_entropy 添加到 Tensor。#3891 和 #5247 例如：对于 neg...
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1266719643498844301)** (9 条消息🔥): 

> - `使用语言模型进行向量搜索`
> - `SWE-Bench Ultra-Hackathon`
> - `Segment Anything Model 2` 


- **探索向量搜索技术**：讨论表明，对于搜索冗长文本，使用 BERT 风格的模型而不是 CLIP 会更有效，并推荐了来自 Jina 或 Nomic 的模型。
   - 一位成员指出，如果重点不是图像，则不应使用 CLIP，并强调 Jina 更好的 CLIP 风格模型是一个有用的替代方案。

- **SWE-Bench 举办为期 6 天的黑客松冒险**：这是一项大胆的实验，为 **SWE-Bench** 举办为期 6 天的超级黑客松，为参与者提供 **$1,000** 的计算资源，并有机会通过改进赢取现金奖励。
   - 启动仪式定于 **8 月 17 日**，参与者将获得知名共同作者的支持，并有机会通过打破基准测试获得团队合作机会和奖项。
- **Segment Anything Model 2 发布**：来自 Facebook Research 的 **Segment Anything Model 2** 已在 [GitHub](https://github.com/facebookresearch/segment-anything-2) 上发布，提供了模型推理代码和模型权重 (checkpoints) 链接。
   - 此外，还包含示例 notebooks，以帮助用户了解如何有效地实现该模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/stevewattsfrey/status/1818033777622532518">Steve Frey (@stevewattsfrey) 的推文</a>：一个大胆的实验：我们正在为 SWE-Bench 举办一场为期 6 天的超级黑客松，以挑战开源代码生成的极限 - 每个人都将获得由 @StrongCompute 提供的 $1,000 计算资源 - 多达 50 名研究人员...</li><li><a href="https://github.com/facebookresearch/segment-anything-2">GitHub - facebookresearch/segment-anything-2: 该仓库提供了使用 Meta Segment Anything Model 2 (SAM 2) 进行推理的代码、下载训练好的模型权重 (checkpoints) 的链接，以及展示如何使用该模型的示例 notebooks。</a>：该仓库提供了使用 Meta Segment Anything Model 2 (SAM 2) 进行推理的代码，下载训练好的模型权重 (checkpoints) 的链接，以及展示如何使用该模型的示例 notebooks...
</li>
</ul>

</div>
  

---



### **AI21 Labs (Jamba) ▷ #[announcements](https://discord.com/channels/874538902696914944/874538945168408606/1267598282008563754)** (1 条消息): 

> - `Jamba 的长上下文能力`
> - `开发者招募`
> - `企业反馈` 


- **长上下文能力的令人兴奋的进展**：关于 **Jamba 的 256k 有效长度**，有一些令人兴奋的进展正在进行中，企业客户的反馈结果非常理想。
   - 团队渴望与正在尝试长上下文用例的开发者交流，以获取进一步的反馈。
- **征集长上下文项目开发者**：团队正在积极寻求 **开发者** 协助处理长上下文用例，并希望听到任何关于其使用体验的反馈。
   - 作为交换，他们向参与的开发者承诺提供 **积分、周边和名声**。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1267633991289147452)** (2 条消息): 

> - `新成员`
> - `社区参与` 


- **新成员加入聊天**：新成员 **artworxai** 宣布加入，并表示“刚刚加入！！”。
   - 另一位成员 **akitshudar** 以友好的问候开启了对话。
- **聊天互动开始**：讨论在成员们互相问候的友好氛围中展开。
   - 这种欢迎的氛围为社区奠定了积极的基调。


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1267421734051512411)** (2 条消息): 

> - `Google AI Hackathon`
> - `LLM 工程机会`
> - `命名实体的去重方法` 


- **Google 黑客松 LLM 工程师最后召集**：一个团队正在为即将到来的 [Google AI Hackathon](https://link.to/hackathon) 寻找最后一名 LLM 工程师加入他们的创新项目。该项目旨在利用 LLM 技术颠覆机器人和教育领域，承诺具有技术复杂性和卓越的用户体验。
   - 候选人应具备高级 LLM 工程技能，熟悉 **LangChain** 和 **LlamaIndex** 等工具，对机器人或教育科技有浓厚兴趣者优先。
- **寻求快速的命名实体去重方案**：一位成员询问了以编程方式对命名实体列表进行去重的有效方法，寻求无需复杂 RAG 设置的快速解决方案。
   - 重点是寻找一种快速高效的方法，而不是实现复杂的系统来处理重复项。


  

---



### **Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1266748507939930112)** (1 条消息): 

> - `人脸识别模型`
> - `情绪检测库` 


- **关于人脸识别模型的讨论**：成员们正在寻求适用于检测和识别视频及图像中人脸的 **机器学习模型** 和 **库** 的建议。
   - 他们强调了实时应用中准确性和性能的重要性。

- **探索情绪检测能力**：人们有兴趣寻找能够从静态图像和视频内容中检测出的面部识别**情绪**的解决方案。
   - 参与者强调需要提供同时具备面部识别和情绪分析功能的**集成解决方案**。


  

---



---



---



{% else %}


> 完整的各频道详细分析已针对电子邮件进行了截断。 
> 
> 如果您想查看完整的详细分析，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}