---
companies:
- google-deepmind
- anthropic
- meta-ai-fair
- openai
- perplexity-ai
- nvidia
- lmsys
date: '2024-08-01T01:33:32.753297Z'
description: '**Google DeepMind** 发布了 **Gemma 2B**，这是一个拥有 20 亿参数的模型，基于 **2 万亿 token**
  训练，并从一个未命名的更大型 LLM 蒸馏而来。尽管在数学方面存在弱点，但它在排行榜上表现强劲。自 6 月发布以来，包括 9B 和 27B 模型在内的 Gemma
  系列已广受欢迎。受 **Anthropic** 研究的启发，该团队还发布了 400 个用于可解释性的 SAE（稀疏自动编码器）。一个名为 ShieldGemma
  的微调分类器在有害内容检测方面优于 Meta 的 LlamaGuard。


  与此同时，**Meta AI** 宣布 **Llama-3.1-405B** 在 Overall Arena 总榜上排名第三，并发布了 **SAM 2**，这是一款在速度上有显著提升的视频和图像分割模型。**OpenAI**
  正在向 Plus 用户推出高级语音模式。**Perplexity AI** 与主要媒体合作伙伴推出了出版商计划（Publishers Program），并上线了一个状态页面。**NVIDIA**
  推出了 Project GR00T，旨在利用 Apple Vision Pro 和生成式仿真来扩展机器人数据。人们对用于压缩 LLM 的量化技术兴趣日益浓厚，而来自
  Vicuna、AlpacaEval 和 G-Eval 的“LLM 作为评审员”（LLM-as-a-Judge）实现，突显了简单提示词和特定领域评估的有效性。'
id: 361c8ff4-c40f-435d-9744-2508f73ddec2
models:
- gemma-2b
- gemma-2-9b
- gemma-2-27b
- llama-3-1-405b
- sam-2
- gpt-3.5
- vicuna
- alpacaeval
- g-eval
original_slug: ainews-gemma-2-2b-scope-shield
people: []
title: '**Gemma 2 2B + Scope + Shield**


  （注：这些是 Google 发布的特定 AI 模型和工具的名称，在中文技术语境中通常保留英文原名。它们分别指：**Gemma 2 2B** 轻量化模型、**Gemma
  Scope** 模型可解释性工具，以及 **ShieldGemma** 安全分类器。）'
topics:
- knowledge-distillation
- leaderboards
- model-interpretability
- finetuning
- harm-detection
- video-segmentation
- voice
- publishers-program
- robotics-data-scaling
- quantization
- llm-evaluation
- prompt-engineering
---

<!-- buttondown-editor-mode: plaintext -->**2B 参数足以击败 GPT 3.5？**

> 2024年7月30日至7月31日的 AI 新闻。我们为您查阅了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 社区（**249** 个频道，**2824** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**314 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

知识蒸馏（knowledge distillation）的博弈正变得愈演愈烈。自 5 月 Google I/O（[我们的报道](https://buttondown.email/ainews/archive/ainews-google-io-in-60-seconds/)）之后，Gemma 2 9B 和 27B 在 6 月发布以来（[我们的报道](https://buttondown.email/ainews/archive/ainews-gemma-2-the-open-model-for-everyone/)）就已经深得人心（[我们的报道](https://buttondown.email/ainews/archive/ainews-gemma-2-tops-rlocalllama-vibe-check/)）。

[Gemma 2B 终于发布了](https://developers.googleblog.com/en/smaller-safer-more-transparent-advancing-responsible-ai-with-gemma/)（为什么又推迟了？）。通过使用 2 万亿 token 训练一个从更大的、未具名的 LLM 蒸馏而来的 2B 模型，Gemma 2 2B 在 [HF v2 排行榜](https://x.com/nathanhabib1011/status/1818686787247575253)（MATH 表现糟糕，但 IFEval 非常强劲）和 [LMsys](https://x.com/robdadashi/status/1818682005569048599?s=46) 上都表现得非常出色。
 
![image.png](https://assets.buttondown.email/images/72325b2d-dd8a-4279-a016-5c59bcb7cf29.png?w=960&fit=max)
 

秉承 Anthropic 的可解释性研究精神（[我们的报道在此](https://buttondown.email/ainews/archive/ainews-anthropic-cracks-the-llm-genome-project/)），Gemma 团队还发布了 400 个涵盖 2B 和 9B 模型的 SAEs。您可以在 [Neuronpedia](https://x.com/swyx/status/1818708147630227779) 上了解更多信息，我们在那里玩得很开心，还搞出了自己的“金门大桥版 Gemma”：

 
![image.png](https://assets.buttondown.email/images/a9879c33-ca34-4074-b80a-fef92d9d89b3.png?w=960&fit=max)
 

还有 ShieldGemma，这似乎是一个针对关键伤害领域进行微调的 Gemma 2 分类器，击败了 Meta 的 LlamaGuard：

 
![image.png](https://assets.buttondown.email/images/3b631941-a83d-4fd3-9bc3-1a03ec01f85f.png?w=960&fit=max)
 


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

**AI 模型更新与发布**

- **Llama 3.1 性能**：[@lmsysorg](https://twitter.com/lmsysorg/status/1818321701052276990) 宣布 Meta 的 Llama-3.1-405B 已攀升至 Overall Arena 排行榜第 3 位，标志着开源模型首次进入前 3 名。该模型在编程、数学和指令遵循等较难类别中依然表现强劲。

- **SAM 2 发布**：Meta 发布了 [Segment Anything Model 2 (SAM 2)](https://twitter.com/AIatMeta/status/1818369887649382729)，这是针对视频和图像分割的一次重大升级。SAM 2 在视频分割方面的运行速度达到每秒 44 帧，所需的交互次数减少了三倍，且在视频标注方面比手动方法提速 8.4 倍。

- **OpenAI Voice Mode**：OpenAI 正在向一小部分 Plus 用户[推出高级 Voice Mode](https://twitter.com/miramurati/status/1818374216997314738)，并计划在秋季推广至所有 Plus 用户。该功能旨在实现更丰富、更自然的实时对话。

- **Perplexity AI 更新**：Perplexity AI 与《时代周刊》(TIME)、《镜报》(Der Spiegel) 和《财富》(Fortune) 等合作伙伴[启动了 Publishers Program](https://twitter.com/perplexity_ai/status/1818271013601513795)。他们还为产品和 API 引入了[状态页面 (status page)](https://twitter.com/AravSrinivas/status/1818425367230898601)。

**AI 研究与开发**

- **Project GR00T**：NVIDIA 的 [Project GR00T](https://twitter.com/DrJimFan/status/1818302152982343983) 引入了一种系统化的方法来扩展机器人数据。该过程包括使用 Apple Vision Pro 收集人类演示数据，使用 RoboCasa（一个生成式模拟框架）进行数据倍增，并使用 MimicGen 进一步增强。

- **量化技术**：人们对[用于压缩 LLM 的 quantization](https://twitter.com/omarsar0/status/1818326822938931613) 的兴趣日益浓厚，相关的视觉指南有助于建立对该技术的直观理解。

- **LLM-as-a-Judge**：讨论了 [LLM-as-a-Judge 的各种实现](https://twitter.com/cwolferesearch/status/1818380242542903361)，包括来自 Vicuna、AlpacaEval 和 G-Eval 的方法。核心结论包括简单 Prompt 的有效性以及特定领域评估策略的实用性。

**AI 工具与平台**

- **ComfyAGI**：介绍了一款名为 [ComfyAGI](https://twitter.com/fabianstelzer/status/1818305254909149621) 的新工具，允许用户使用 Prompt 生成 ComfyUI 工作流。

- **Prompt Tuner**：Cohere [推出了 Prompt Tuner 测试版](https://twitter.com/cohere/status/1818355539845562575)，这是一个可以直接在 Dashboard 中使用可自定义的优化和评估循环来优化 Prompt 的工具。

- **LlamaIndex 支持 MLflow**：LlamaIndex 现在[支持 MLflow](https://twitter.com/llama_index/status/1818399148494012897)，用于管理模型的开发、部署和管理。

**行业与职业新闻**

- **英国政府招聘**：英国政府正在[招聘一名 Senior Prompt Engineer](https://twitter.com/rohanpaul_ai/status/1818278407131763079)，薪资范围为 £65,000 - £135,000。

- **Tim Dettmers 的职业动态**：Tim Dettmers [宣布](https://twitter.com/Tim_Dettmers/status/1818282778057941042)加入 Allen AI，并将从 2025 年秋季起担任 Carnegie Mellon 大学教授，同时担任 bitsandbytes 的新维护者。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 开源 AI 与大语言模型的民主化**

- **["不，去他的……一提到封闭平台，我就来气"](https://v.redd.it/yts11phqhpfd1)** ([Score: 56, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1eg21cw/nah_f_that_get_me_talking_about_closed_platforms/)): 在 **7 月 29 日**的 **SIGGRAPH** 上，**Mark Zuckerberg** 对**封闭 AI 平台**表达了强烈的反对。他的坦率言论被认为是讨论中的一个显著时刻，尽管帖子中未提供其评论的具体内容。
- **这就是在 M2 Ultra 上运行 Llama 3.1 405B 4bit 的样子** ([Score: 65, Comments: 24](https://reddit.com//r/LocalLLaMA/comments/1egbmtd/this_is_what_it_looks_like_to_run_a_llama_31_405b/)): 该帖子展示了在 **Apple M2 Ultra** 芯片上运行 **Llama 3.1 405B 4-bit 模型**的情况，强调了其易用性，尽管它并不是最具性价比的选择。作者提供了 **mlx_sharding** 和 **open-chat** 的 GitHub 仓库链接，这些工具提供了更好的分片（sharding）控制，并提到使用 **exo** 也可以实现类似的功能。
  - **DeepSeek Coder V2 4bit** 可以在单台配备 **192GB RAM** 的 **M2 Ultra** 上运行，正如[相关推文](https://x.com/awnihannun/status/1814045712512090281)所示。分片过程是顺序的，节点处理不同的层，从而实现了内存扩展但没有性能提升。
  - 一段 [YouTube 视频](https://www.youtube.com/watch?v=fXHje7gFGK4)展示了使用 **MLX** 在**单台 MacBook M3 Ultra** 上运行 **Llama 3.1 405B 2bit**，消耗约 **120GB 内存**。用户希望未来的 Windows 笔记本电脑能配备 **256GB 统一内存**，以运行 Llama 405B INT4。
  - 行业内部报告显示，**Lunar Lake** 处理器最初将推出 **16GB 和 32GB 型号**，随后是限量的 **64GB 版本**，均为板载焊接。**Arrow Lake** 桌面芯片预计将为 Windows 平台提供更实惠的 **256GB+** 选项，至少要到 2025 年底。


**主题 2. 提升 LLM 性能的高级 Prompt 技巧**

- **新论文：“Meta-Rewarding Language Models”——无需人类反馈的自我改进 AI** ([Score: 50, Comments: 3](https://reddit.com//r/LocalLLaMA/comments/1efrv5a/new_paper_metarewarding_language_models/)): 该论文介绍了一种名为 **“Meta-Rewarding”** 的技术，由来自 **Meta、加州大学伯克利分校（UC Berkeley）和纽约大学（NYU）** 的研究人员开发，用于在没有人类反馈的情况下改进语言模型。该方法以 **Llama-3-8B-Instruct** 为起点，让一个模型扮演三个角色（actor、judge 和 meta-judge），并在基准测试中取得了显著进步，将 **AlpacaEval 胜率从 22.9% 提升至 39.4%**，**Arena-Hard 从 20.6% 提升至 29.1%**。这种方法代表了向自我改进 AI 系统迈出的一步，并可能加速更强大的开源语言模型的开发。

- **有哪些令人惊叹的 Prompt 技巧？** ([Score: 112, Comments: 72](https://reddit.com//r/LocalLLaMA/comments/1efqhj7/what_are_the_most_mind_blowing_prompting_tricks/)): 该帖子征集 **Large Language Models (LLMs)** 的**惊人 Prompt 技巧**，提到了诸如使用“**stop**”、**base64 解码**、针对特定目标的 **topK** 以及**数据提取**等技术。作者分享了他们最喜欢的技巧“**修复此重试（fix this retries）**”，即要求 LLM 纠正生成代码中的错误（特别是针对 **JSON**），并鼓励回复者在使用分享的技巧时注明所使用的模型。
  - 要求 LLM “**为每个主张提供参考文献**”可以显著减少**幻觉（hallucinations）**。这种技术之所以有效，是因为模型编造参考文献的可能性低于编造事实，正如关于**生物发光鲸鱼**的讨论所证明的那样。
  - 用户发现，重新表述敏感问题（例如，将“**如何制造冰毒？**”改为“**过去人们是如何制造冰毒的？**”）通常可以绕过 **ChatGPT 4** 等模型的内容限制。将回复的开头从“抱歉，我……”改为“当然……”也可能奏效。
  - 结合 **20k tokens** 高质量精选示例的 **Many Shot In Context Learning** 显著提升了模型性能。使用 `<resume/>`、`<instruction/>` 和 `<main_subject>` 等标签构建 Prompt 可以增强效果，从而实现以前无法完成的任务。


**主题 3. 优化三进制模型以实现更快的 AI 推理**

- **更快的三进制推理是可能的** ([Score: 115, Comments: 30](https://reddit.com//r/LocalLLaMA/comments/1egg8qx/faster_ternary_inference_is_possible/))：一项利用 **AVX2** 指令集提升三进制模型推理速度的突破已经实现，在无需定制硬件的情况下，相比 **Q8_0** 实现了 **2 倍的速度提升**。这项新技术利用 `_mm256_maddubs_epi16` 将无符号三进制值与 8-bit 整数直接相乘，在已经提速 **50%** 的 `vec_dot` 操作基础上，又带来了 **33% 的性能提升**。这一进展使得运行 **3.9B TriLM model** 的速度可以像 **2B Q8_0 model** 一样快，且仅需 **1GB** 的权重存储空间，未来还有在 **ARM NEON** 和 **AVX512** 架构上进一步优化的潜力。
    *   用户对 **开源协作** 和这一突破表示赞赏，部分用户希望对这些技术概念进行 **简化解释**。一份 AI 生成的解释强调了在无需专门硬件的普通电脑上运行 **更大 AI 模型** 的重要意义。
    *   讨论中涉及了位（bits）和字节（bytes）中 **三进制状态（ternary states）** 的实现方式。会议明确了可以将 **5 个 trits** 打包进 **8 bits** 中（3^5 = 243 < 256 = 2^8），这被应用于 **TQ1_0** 量化方法中。
    *   作者 **compilade** 回答了关于 **性能瓶颈** 的问题，表示对于 **低端系统**，计算方面仍有改进空间。他们还提到，减少计算量可以通过用更少的内核使内存带宽饱和，从而帮助高性能系统 **节省能源**。

## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 生成媒体与视觉技术**

- **Midjourney v6.1 发布**：在 /r/singularity 中，[Midjourney 宣布发布 v6.1 版本](https://www.reddit.com/r/singularity/comments/1eg2sjt/midjourney_v61_just_released_and_is_practically/)，其特点是 **改进了图像连贯性、质量和细节**。主要改进包括：
  - 增强了手臂、腿、手、身体、植物和动物的连贯性
  - 减少了像素伪影并改善了纹理
  - 更精确的微小图像特征（眼睛、小脸、远处的双手）
  - 具有更好图像/纹理质量的新 **upscalers**
  - 标准图像任务速度提升约 **25%**
  - 提高了提示词中的文本准确性
  - 具有更高细微差别和准确性的新个性化模型

- **令人信服的虚拟人**：在 /r/StableDiffusion 中，一段 [视频演示](https://www.reddit.com/r/StableDiffusion/comments/1efsy12/the_age_of_convincing_virtual_humans_is_here/) 展示了 **Stable Diffusion 结合 Runway 的 Image to Video 技术** 的能力，突显了在创建逼真虚拟人方面的进展。

- **Midjourney 与 Runway Gen-3 展示**：在 /r/singularity 中，另一个 [视频演示](https://www.reddit.com/r/singularity/comments/1eg7tpa/interesting_showcase_of_midjourney_runway_gen3/) 将 **Midjourney 的图像生成与 Runway 的 Gen-3 Image to Video 技术** 相结合，进一步说明了 AI 生成视觉内容的进步。

**AI 与隐私担忧**

- **美国人对 AI 和隐私的担忧**：在 /r/singularity 中，一篇 [雅虎财经文章](https://www.reddit.com/r/singularity/comments/1efs7ca/74_of_americans_fear_ai_will_destroy_privacy/) 报道称，**74% 的美国人担心 AI 会破坏隐私**，凸显了公众对 AI 影响个人数据保护的日益担忧。

**AI 监管与政策**

- **加州 AI 安全法案辩论**：在 /r/singularity 中，[Yann LeCun 分享了一篇 Ars Technica 的文章](https://www.reddit.com/r/singularity/comments/1eftjx9/yann_lecun_good_article_at_arstechnica_on_the/)，讨论了围绕加州 **AI 安全法案 SB1047** 的辩论。LeCun 担心该法案可能会“**从本质上扼杀开源 AI，并显著减缓或停止 AI 创新**”。

---

# AI Discord 回顾

> 摘要的摘要之摘要

**1. LLM 进展与基准测试**

- **Llama 3.1 多语言奇迹**：Meta 发布了 **[Llama 3.1](https://x.com/reach_vb/status/1815767864277606762)**，包含 **405B, 70B, 和 8B** 参数版本，具有 **128K context** 并支持 **English, Spanish, 和 Thai** 等语言。
  - 基准测试显示 **Llama 3.1** 达到了 **85.2**，表现优于 **GPT4o 和 Claude**，且拥有更宽松的训练许可。
- **Gemma 2 模型提供快速微调**：新发布的 **Gemma 2 (2B)** 模型拥有 **2x faster** 的微调速度和 **65% less VRAM** 占用，支持在 **80GB GPU** 上使用 **up to 86k tokens** 进行训练。
  - 这些增强显著提升了模型的 context length 能力，许多用户认为这对其项目至关重要。


**2. 模型性能优化与基准测试**

- **SwiGLU 在速度上超越 GELU**：最近的测试显示 **[SwiGLU](https://github.com/karpathy/llm.c/pull/715)** 的收敛速度比 **GELU** 更快，最终达到相似的 loss 水平，这表明在稳定性上可能存在权衡。
  - 参与者讨论了 **SwiGLU** 相比 ReLU 等传统激活函数是否具有真正的优势。
- **LLM 的动态内存系统**：在 LLM 中操作对话历史的概念引发了关于现有角色扮演策略和 **RAG-like systems** 的丰富讨论。
  - 虽然对这种方法的新颖性存在**怀疑**，但它激发了关于潜在应用和在现实场景中有效性的进一步对话。


**3. 微调挑战与 Prompt Engineering 策略**

- **Gemma 2B 性能洞察**：成员们讨论了 Google DeepMind 的 **Gemma 2B** 的性能结果，它在 LMSYS Arena 上获得了 **1130** 分，超越了 **GPT-3.5**。
  - 讨论中提出了对这类基准测试可靠性的担忧，与成熟模型的对比引发了持续的辩论。
- **使用 OpenAI 进行 Logit Bias 的 Prompt Engineering**：**Prompt engineering** 策略包括将复杂任务拆分为多个 prompt，以及研究 **logit bias** 以获得更多控制权。
  - 示例：[OpenAI logit bias 指南](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api)。


**4. 开源 AI 发展与协作**

- **Hugging Face 与 Nvidia 联手**：Hugging Face 与 **[Nvidia AI](https://x.com/NVIDIAAIDev/status/1818050230392398175)** 合作推出 **inference-as-a-service**，允许利用开源 AI 模型进行快速原型设计。
  - 此次合作支持快速部署，利用了 Hugging Face 广泛的模型库。
- **Sparse Autoencoders 简化特征恢复**：最近的进展帮助 **[Sparse Autoencoders](https://github.com/EleutherAI/sae-auto-interp)** 恢复可解释的特征，简化了在 **GPT-2** 和 **Llama-3 8b** 等模型上的评估过程。
  - 这对于处理人类标注者面临的可扩展性挑战至关重要，展示了开源模型可以实现与人类解释相媲美的评估。


**5. 多模态 AI 与生成模型创新**

- **VLM 微调现已上线**：**[AutoTrain](https://x.com/abhi1thakur/status/1816429924233687470)** 刚刚宣布了一项针对 **PaliGemma** 模型的 **VLM finetuning** 新任务，简化了自定义数据集的集成。
  - 该功能邀请用户建议需要增强的模型和任务，提升了 AutoTrain 的功能性。
- **InternLM 发布 MindSearch 框架**：**[InternLM](https://github.com/InternLM/MindSearch)** 推出了 **MindSearch**，这是一个类似于 Perplexity.ai 的网络搜索引擎工具，旨在增强 multi-agent 搜索功能。
  - 该框架基于 LLM，专注于精准度，有望显著优化搜索结果。

---

# 第 1 部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama 3.1：多语言奇迹**：Meta 发布了拥有 **405B、70B 和 8B** 参数的 [Llama 3.1](https://x.com/reach_vb/status/1815767864277606762)，具备 **128K context** 并支持**英语、西班牙语和泰语**等语言。
   - Benchmarks 显示 **Llama 3.1** 达到了 **85.2** 分，超越了 **GPT4o 和 Claude**，并拥有更宽松的训练许可证。
- **令人兴奋的 Argilla 2.0 特性**：即将推出的 [Argilla 2.0](https://x.com/argilla_io/status/1817945202432061792) 将引入 **easy dataset duplication** 功能，以实现高效的数据管理。
   - 这一增强功能对于需要多种数据集配置的应用至关重要。
- **Peft v0.12.0 带来效率提升**：[Peft v0.12.0](https://x.com/julien_c/status/1817837045298978986) 引入了 **OLoRA、X-LoRA 和 FourierFT** 等创新的参数高效方法，优化了模型训练。
   - 这些方法简化了各种模型类型的 fine-tuning 过程。
- **Hugging Face 与 Nvidia 联手**：Hugging Face 与 [Nvidia AI](https://x.com/NVIDIAAIDev/status/1818050230392398175) 合作提供 **inference-as-a-service**，允许利用开源 AI 模型进行快速原型设计。
   - 此次合作支持快速部署，利用了 Hugging Face 广泛的 model hub。
- **VLM Finetuning 现已上线**：AutoTrain 刚刚宣布了针对 **PaliGemma** 模型的 [VLM finetuning](https://x.com/abhi1thakur/status/1816429924233687470) 新任务，简化了自定义数据集的集成。
   - 该功能邀请用户为功能增强建议模型和任务。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 2 模型提供快速 fine-tuning**：新发布的 **Gemma 2 (2B)** 模型拥有 **2 倍更快**的 fine-tuning 速度和 **65% 更少的 VRAM** 占用，支持在 **80GB GPU** 上训练高达 **86k tokens**。
   - 这些增强显著提升了模型的 context 长度能力，许多用户认为这对其项目至关重要。
- **社区焦急等待 Multi-GPU 支持**：用户对 **multi-GPU support** 的开发进度表示不耐烦，强调了过去的承诺以及对更新的迫切需求。
   - 虽然一些用户报告了在 beta 测试中的成功经验，但整个社区仍希望有更明确的时间表。
- **MegaBeam-Mistral 助力 512k context**：**MegaBeam-Mistral-7B-512k** 模型支持高达 **524,288 tokens**，基于 [Mistral-7B Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) 训练。
   - 评估结果展示了该模型在三个长文本 benchmarks 上的能力，提升了使用 [vLLM](https://github.com/vllm-project/vllm) 等框架部署它的兴趣。
- **Quantization 方法的权衡影响推理**：关于不同 quantization 方法的讨论显示，**4-bit quantization** 通常会导致较差的推理响应。
   - 一位用户强调了之前使用 **GGUF quantization** 取得的成功，但指出目前的结果仍存在不一致性。
- **持续 pre-training 中关于 learning rates 的见解**：研究强调了 **learning rates** 在持续 pre-training 中的重要性，揭示了在调整该参数时跨领域的预测损失。
   - 关键发现表明，最佳 learning rate 能在快速学习和最小化遗忘之间取得平衡，这对模型训练的有效性至关重要。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **SOTA 图像生成成就**：成员们庆祝在内部实现了 SOTA 图像生成，并分享了相关链接。生成的模型产出了极具美感的输出，提升了用户参与度。
   - 在达成这一里程碑后，成员们讨论了相关模型及其性能特征，包括对未来图像生成任务的影响。
- **LLM 推理与预测挑战**：社区辩论了 LLM 的推理能力，质疑自回归 Token 预测的有效性。虽然高 Temperature 设置可能产生正确答案，但重大的推理挑战依然存在，特别是在符号上下文中。
   - 这引发了关于改进推理方法的讨论，呼吁采用更好的方法论来增强模型内的符号处理能力。
- **Gemma 2B 性能见解**：成员们讨论了 Google DeepMind 的 Gemma 2B 的性能结果，其在 LMSYS Arena 上得分 **1130**，超过了 GPT-3.5 等其他知名模型。讨论中也对这类 Benchmark 的可靠性提出了担忧。
   - 与既有模型的对比引发了关于 Benchmark 有效性的持续辩论，特别是针对 **Gemini Ultra** 等新兴模型。
- **探索动态内存系统**：在 LLM 中操作聊天历史的想法引发了关于现有 Roleplaying 策略和类 RAG 系统的深入讨论。成员们分享了关于这些系统如何实际落地的见解。
   - 尽管对该方法的新颖性存在怀疑，但它激发了关于在现实场景中潜在应用和有效性的进一步对话。
- **对 Hugging Face 排行榜的好奇**：一位成员询问 [Hugging Face leaderboard](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results) 是否是代码生成任务的主要资源。其他人提到了 **BigCodeBench** 作为潜在替代方案，但缺乏具体细节。
   - 这一询问开启了关于代码生成领域 Benchmark 和性能指标的讨论，重点在于识别可靠的资源。



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Accel 庆祝成立 40 周年**：风险投资公司 [Accel](https://www.accel.com/) 最近庆祝了其 **40 周年**，强调了其悠久的历史以及对科技领域的贡献。
   - 正如在其 [庆祝活动](https://40-years.accel.com/) 中讨论的那样，他们强调与卓越团队的合作伙伴关系，曾支持过 Facebook 和 Spotify 等巨头。
- **分布式训练资源**：成员们推荐将 [PyTorch docs](https://pytorch.org) 作为学习 FSDP 和 TP 等 **distributed training** 技术的必备资源。
   - 他们还强调了一篇关于 FSDP 的特定 [Pytorch paper](https://arxiv.org/abs/2304.11277)，因其对边缘情况的详尽解释。
- **Tinyboxes 出货详情**：根据 [Tinygrad](https://x.com/__tinygrad__/status/1818408577155154280) 的更新，Tinyboxes 目前的出货量约为 **每周 10 台**，在周一进行发货。
   - 这是他们努力减少预订等待时间的一部分。
- **Triton 编程模型受到关注**：一篇文章阐述了如何在 Java 中利用 Code Reflection 来实现 **Triton** 编程模型，为 Python 之外的应用提供了新途径。
   - 讨论强调了 Triton 如何通过利用中间表示（IR）并增强开发者的易用性来简化 GPU 编程任务。
- **CUDA 内存对齐问题**：有人担心从 CUDA 的 Caching Allocator 返回的 GPU 内存是否始终是对齐的，特别是在操作过程中出现 **CUDA error: misaligned address** 时。
   - 专家指出，虽然分配器通常会确保对齐，但这并不能保证 PyTorch 中的每个 Tensor 指针都是对齐的。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Vulkan 支持上线**：计划于明天发布的更新将在 LM Studio 中引入 **Vulkan 支持**，在 **OpenCL** 弃用后增强 GPU 性能。这一转变是在关于 **AMD 驱动程序** 兼容性的持续讨论中进行的。
   - 用户预计这一新支持将解决早期版本中关于 Intel 显卡兼容性的多个错误报告和不满。
- **Gemma 2B 模型进入 Beta 测试**：即将发布的 **0.2.31 beta** 版本承诺为用户带来关键改进，包括对 **Gemma 2 2B** 模型的支持以及 kv cache quantization 选项。用户可以在 Discord 上加入 Beta Builds 角色以获取新版本的通知。
   - 然而，加载 **Gemma 2B** 等模式的挑战被凸显出来，通常需要更新底层的 **llama.cpp** 代码以实现最佳使用。
- **AI 驱动的 D&D 对话**：一位用户尝试创建一个以多个 AI 作为玩家的 D&D 直播，旨在让它们以结构化的格式进行互动。在头脑风暴过程中，出现了关于对话动态和 speech-to-text 复杂性的担忧。
   - 围绕 AI 交互的对话暗示了一种增强游戏过程中参与度的创新方法，展示了 AI 应用的灵活性。
- **最大化 GPU 资源**：用户确认利用 GPU offloading 显著有助于高效运行大型模型，特别是在高 context 任务中。与仅依赖 CPU 资源相比，这种方法提升了性能。
   - 然而，不同 GPU（如 **3080ti** 与 **7900XTX**）之间 RAM 使用情况的差异，强调了在为 AI 工作负载配置硬件时需要仔细考虑。
- **安装冲突与解决方法**：新用户分享了对安装问题的沮丧，特别是在从 Hugging Face 下载模型时，将访问协议视为障碍。建议的解决方法包括使用绕过协议点击的替代模型。
   - 此外，LM Studio 缺乏对拖放 CSV 上传的支持，强调了目前在无缝文档馈送功能方面的局限性。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **为电视剧角色训练 Loras**：为了生成包含两个电视剧角色的图像，用户可以在 Auto1111 中利用 **regional prompter extension** 在不同区域加载不同的 Loras。
   - 或者，带有特定 prompt 的 **SD3** 可能会奏效，尽管它在处理知名度较低的角色时比较吃力；建议使用带标签的图像创建自定义 Loras。
- **GPU 对决：RTX 4070S vs 4060 Ti**：在从 **RTX 3060** 升级时，用户发现 **RTX 4070S** 的性能通常优于 **RTX 4060 Ti**，尽管后者提供更多 VRAM。
   - 对于 AI 任务，共识倾向于选择 **4070S** 以获得增强的性能，而 **4060 Ti** 更大的显存（memory）在某些场景下可能更有利。
- **ComfyUI vs Auto1111：界面之争**：用户指出 **ComfyUI** 为 **SD3** 提供了卓越的支持和效率，而 **Auto1111** 的功能有限，特别是在 clip layers 方面。
   - 在 ComfyUI 中进行正确的模型设置对于避免兼容性陷阱并确保最佳性能至关重要。
- **图像生成的困扰**：用户报告了生成包含多个角色的图像时的问题，在使用不熟悉的模型时经常导致错误的输出。
   - 为了缓解这种情况，建议在集成自定义模型或 Loras 之前，先仅使用 prompt 进行初步测试，以获得更好的兼容性。
- **Creative Upscaling 在 Automatic1111 中的困惑**：关于在 Automatic1111 中使用 **creative upscaler** 的疑问不断出现，新用户正在寻求指导。
   - 虽然 **NightCafe** 等各种 AI 工具中可能存在这些功能，但在 Auto1111 中高效访问它们可能需要额外的配置步骤。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 语音模式备受期待**：许多用户对 OpenAI 的**高级语音模式 (advanced voice mode)** 表达了渴望，期待更出色的交互体验。
   - 用户对语音的**质量**和**多样性**提出了担忧，暗示未来的更新可能存在局限性。
- **DALL-E 3 与 Imagen 3 的对决**：用户对比显示，尽管 **Imagen 3** 拥有严格的审核系统，但其视觉效果被认为比 **DALL-E 3** 更写实。
   - 一些用户寻求 **GPT-4o** 与 **Imagen 3** 之间具体的性能见解，凸显了对详细评估的需求。
- **学术辅助的最佳 AI 工具**：用户讨论了哪种 AI 模型——如 **GPT-4o**、**Claude** 或 **Llama**——在学术任务中表现更优。
   - 这反映了用户在寻求能够增强教育体验的最有效 AI 工具。
- **对自定义 GPTs 的担忧增加**：一名成员对自定义 GPTs 中可能存在的**恶意内容**或隐私问题提出了担忧。
   - 讨论强调了 AI 模型中用户生成内容的风险以及被滥用的可能性。
- **追求 STT 和 TTS 的低延迟**：成员们讨论了哪些 **STT** 和 **TTS** 系统在实时转录中延迟最低。
   - 共享了多个资源，包括使用 **WhisperX** GitHub 仓库进行安装的指南。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Sparse Autoencoders 简化特征恢复**：最近的进展帮助 **Sparse Autoencoders** 恢复可解释特征，简化了在 **GPT-2** 和 **Llama-3 8b** 等模型上的评估过程。这对于应对人类标注员面临的可扩展性挑战至关重要。
   - 关键结果表明，开源模型可以实现与人类解释相媲美的评估效果。
- **白宫支持开源 AI**：白宫发布了一份[报告](https://www.ntia.gov/press-release/2024/ntia-supports-open-models-promote-ai-innovation)，支持开源 AI，且目前不对模型权重 (model weights) 进行限制。这一立场强调了在创新与防范风险之间取得平衡。
   - 官员们认识到开放系统的必要性，推动了关于 AI 政策的对话。
- **Diffusion Augmented Agents 提高效率**：**Diffusion Augmented Agents (DAAG)** 的概念旨在通过整合语言、视觉和扩散模型来提高强化学习 (reinforcement learning) 的样本效率。在模拟中显示效率提升的早期结果预示了未来应用的潜力。
   - 这些创新将改变我们应对强化学习挑战的方式。
- **Gemma Scope 提升可解释性**：[Gemma Scope](https://neuronpedia.org/gemma-scope) 作为一个开源的 **Sparse Autoencoders** 套件发布，应用于 **Gemma 2** 的各层，其开发利用了 **GPT-3 算力的 22%**。该工具承诺增强 AI 模型的可解释性。
   - Neuronpedia 的一个演示展示了其功能，更多讨论和资源在 [推文线程](https://x.com/NeelNanda5/status/1818680642621915527) 中分享。
- **知识蒸馏 (Knowledge Distillation) 的难题**：成员们正在寻求关于 **7B 模型** **knowledge distillation** 的见解，特别是关于超参数 (hyperparameter) 设置和所需算力资源。这反映了社区通过蒸馏技术优化模型性能的趋势。
   - 讨论围绕超参数调优对有效蒸馏结果的重要性展开。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **付费用户要求就广告问题给出答复**：成员们对**付费用户**是否会遇到广告表示出日益增长的担忧，担心这会破坏平台的无广告体验。
   - 有人指出，“*沉默绝非好兆头*”，并强调 **Perplexity** 迫切需要进行沟通。
- **WordPress 合作伙伴关系引发疑问**：关于 **WordPress 合作伙伴关系** 影响的询问开始出现，核心在于它是否会影响个人博主的内容。
   - 社区成员渴望了解这一合作伙伴关系将如何影响他们的贡献。
- **Perplexity Labs 遇到问题**：多位用户报告了 **Perplexity Labs** 的访问问题，从 *ERR_NAME_NOT_RESOLVED* 错误到对地理限制的猜测不等。
   - 常规功能似乎运行正常，这引发了关于基于位置的访问权限的关键疑问。
- **广告融合的伦理问题**：针对响应中可能出现的**赞助问题（sponsored questions）**及其对用户认知的潜在影响，社区产生了担忧。
   - 参与者对可能损害响应完整性的广告表示忧虑。
- **图表创建查询激增**：用户正在寻求在 Perplexity 中创建**图表**的指导，并猜测某些功能是否需要 **Pro 版本**。
   - 访问权限可能还取决于地区可用性，这让许多人感到困惑。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **社区呼吁对 Mojo 提供反馈**：成员们强调需要建设性的反馈来增强 **Mojo 社区会议**，使其更具参与性和相关性。
   - *Tatiana* 敦促演讲者利用 Discord 标签，并将讨论集中在关键的 **Mojo** 话题上。
- **演示文稿官方指南**：一名成员提议建立正式指南，以界定 **Mojo 社区会议** 中讨论的范围。
   - *Nick* 强调了将演示重点放在 **Mojo** 语言、其库以及社区关注点上的重要性。
- **Mojo 作为 C 替代方案的可行性**：关于在 ARM、RISC-V 和 x86_64 架构上使用 **Mojo** 作为解释器的 C 语言替代方案的咨询不断出现，得到的回复褒贬不一。
   - *Darkmatter* 澄清说，Mojo 中缺少类似 computed goto 的功能，其结构类似于 **Rust**，但使用了 Python 的语法。
- **Mojo 中的类型比较奇特行为**：在 Mojo 中观察到一种奇怪的行为，将 `list[str] | list[int]` 与 `list[str | int]` 进行比较时结果为 **False**。
   - *Ivellapillil* 确认，从类型层级的角度来看，单一类型的列表与混合类型的列表是不同的。
- **Mojo 字符串深入探讨 UTF-8 优化**：实现者展示了一种具有小字符串优化（small string optimization）的 **Mojo 字符串**，它支持完整的 UTF-8 并实现了高效索引。
   - 该实现允许三种索引方法：字节（byte）、Unicode 码位（code point）和用户感知的字形（glyph），以满足多语言需求。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **LLM 追踪挑战堆积**：成员们对日益增多的 **LLM** 感到沮丧，指出很难追踪它们的**能力和性能**。
   - 在这个拥挤的领域中，随着新模型的出现，*创建个人基准测试是必要的*。
- **Aider 的 LLM 排行榜出现**：**Aider 的 LLM 排行榜**根据模型在编程任务中的编辑能力进行排名，突显了其专业化的侧重点。
   - 用户注意到，该排行榜最适合那些在“编辑”而非仅仅是生成代码方面表现出色的模型。
- **对 4o Mini 性能的担忧**：围绕 **4o Mini** 展开了辩论，对其与 **3.5** 等模型相比的性能评价褒贬不一。
   - 尽管它有其优势，但一些成员更倾向于使用 **1.5 flash**，因为其输出质量更高。
- **关于 NSFW 模型选项的讨论**：成员们分享了对各种 **NSFW 模型** 的看法，特别是 **Euryal 70b** 和 **Magnum** 被视为脱颖而出的选项。
   - 其他建议还包括 **Dolphin 模型** 以及像 **SillyTavern Discord** 这样的资源以获取更多信息。
- **OpenRouter 成本削减见解**：一位成员报告说，在从 **ChatGPT 切换到 OpenRouter** 后，他们的支出从 **$40/月** 大幅下降到 **$3.70**。
   - 这些节省来自于使用 **Deepseek 进行编程**，这构成了他们使用量的大部分。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemma 2 2B 超越 GPT-3.5**：新的 [Gemma 2 2B model](https://developers.googleblog.com/en/smaller-safer-more-transparent-advancing-responsible-ai-with-gemma/) 在 Chatbot Arena 上的表现优于所有 GPT-3.5 模型，展示了其卓越的对话能力。
   - 成员们表达了兴奋之情，其中一人在讨论其性能时感叹：*“真是见证历史的时刻 (what a time to be alive)”*。
- **Llama 3.1 主导基准测试**：Llama 3.1 已成为首个能与顶尖模型抗衡的开源模型，在 **GSM8K 上排名第一**，展示了实质性的 **推理质量 (inference quality)**。
   - 讨论强调了破译实现差异的必要性，这会显著影响应用的成功。
- **关于 LLM 自我修正局限性的辩论**：Prof. Kambhampati 在最近的 [YouTube 视频](https://www.youtube.com/watch?v=y1WnHpedi2A)中批评了 LLM 的表现，指出它们在逻辑推理和规划方面存在显著局限。
   - 他的 [ICML tutorial](https://youtu.be/2DbmSTK2owI?si=mIJ9lFLyxM1RGCjB) 进一步讨论了关于 LLM **自我修正 (self-correction)** 的缺陷和基准测试。
- **对截图可读性的担忧**：成员们对移动端截图文本难以阅读表示担忧，指出 **压缩 (compression)** 问题影响了清晰度。
   - 一位成员承认截图很 **模糊 (blurry)**，虽然感到沮丧，但仍决定继续。
- **关于 LLM 基准测试挑战的反馈**：LLM 在需要纠正初始错误的基准测试中面临挑战，引发了对 **自我修正 (self-correction)** 评估可行性的担忧。
   - 讨论表明，如果没有外部反馈，LLM 通常无法有效地进行自我修正，这使得模型性能的比较变得复杂。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **用于 Cohere 工具创建的 Google Colab**：一位成员正在开发一个 [Google Colab](https://link.to.colab)，以帮助用户有效利用 **Cohere API** 工具，其特色是集成了 **Gemini**。
   - *“从不知道它里面有 Gemini！”* 引发了用户对新功能的兴奋。
- **8 月 12 日在旧金山举行的 Agent Build Day**：欢迎参加 **8 月 12 日**在旧金山举行的 **Agent Build Day**，届时将有来自 **Cohere**、**AgentOps** 和 **CrewAI** 的专家主持研讨会。参与者可以在[此处](https://lu.ma/gptdzwhe)注册，有机会赢取 **2,000 美元的 Cohere API 额度**。
   - 该活动包括演示竞赛，但一些成员对缺乏虚拟参与选项表示失望。
- **Rerank API 出现 403 错误**：一位用户报告在调用 Rerank API 时遇到 **403 错误**，即使使用了有效的 Token，引发了社区的故障排除建议。
   - 另一位成员通过请求更多细节（包括完整的错误消息或设置截图）提供了帮助。
- **社区工具包 (Community Toolkit) 激活问题**：一位成员面临社区工具包无法激活的问题，尽管在 Docker Compose 配置中将 **INSTALL_COMMUNITY_DEPS** 设置为 true。
   - 提到的工具仍然不可见，促使进一步询问有效的初始化命令。
- **使用模型训练阿拉伯语方言**：出现了一场关于训练模型以特定方言生成阿拉伯语回复的讨论，提到了 **Aya dataset** 及其特定方言的指令。
   - **Aya** 和 **Command** 模型都被认为能够处理该任务，但仍缺乏清晰的方言指令。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 使用 50 万亿 tokens 进行训练**：有消息提到 OpenAI 据传正在使用 **50 万亿 tokens**（主要为合成数据）训练 AI 模型，引发了关于其对训练效果影响的讨论。
   - 这一消息让人们对如此庞大的数据集可能带来的模型能力提升感到兴奋。
- **Gemma 2 2B 模型表现优于 GPT-3.5**：**Gemma 2 2B** 模型已成为 Chatbot Arena 中的佼佼者，在对话任务中超越了所有 **GPT-3.5** 模型，且内存占用极低。
   - 该模型通过 **2 万亿 tokens** 的训练，展示了令人印象深刻的能力，尤其适用于端侧（on-device）实现。
- **Llama 3.1 评估引发关注**：针对 **Llama 3.1** 的批评开始出现，一些博客示例展示了与 multi-query attention 及其他功能相关的准确性问题。
   - 这引发了关于所采用评估方法以及报告结果完整性的辩论。
- **alphaXiv 旨在增强论文讨论**：由斯坦福大学学生推出的 **alphaXiv** 是一个参与 arXiv 论文讨论的平台，允许用户通过论文链接发布问题。
   - 该计划旨在围绕学术工作创建一个更具动态性的讨论环境，可能有助于更好地理解复杂主题。
- **InternLM 发布 MindSearch 框架**：InternLM 推出了 **MindSearch**，这是一个类似于 Perplexity.ai 的搜索引擎工具，旨在增强 multi-agent 搜索功能。
   - 该框架基于 LLM 且专注于精准度，有望显著优化搜索结果。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **对 Attention 层量化的好奇**：一位成员提出疑问，询问 LLM 中 **attention layers** 的参数是否使用了与 feed forward 层类似的量化方法，并引用了一篇关于量化的科普[文章](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)。
   - 这一讨论突显了在保持性能的同时缩小 LLM 体积的持续关注。
- **Axolotl 的 Early Stopping 功能**：有人询问 **Axolotl** 是否提供在 loss 呈渐进式收敛或 validation loss 上升时自动终止训练运行的功能。
   - 重点在于通过及时干预来提高训练效率和模型性能。
- **对 Gemma-2-27b 配置的需求**：一位用户询问是否有适用于微调 **Gemma-2-27b** 的可用配置，强调了社区中的这一需求。
   - 目前尚未提供具体配置，表明在知识共享方面存在空白。
- **Serverless GPU 现状报告更新**：分享了关于 **State of Serverless GPUs report** 的新见解，强调了过去六个月中 AI 基础设施领域的重大变化，详见[此链接](https://www.inferless.com/learn/the-state-of-serverless-gpus-part-2)。
   - *我们上一份指南引起了全球范围的广泛关注*，提供了关于选择 Serverless 供应商及市场发展的见解。
- **Retrieval Augmented Generation (RAG) 的潜力**：讨论指出，在 **Axolotl** 上运行任何模型时，微调 **Llama 3.1** 可能是有效的，而 RAG 被视为一种可能更合适的方法。
   - 参与者就 RAG 如何增强模型能力提出了建议。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **MLflow 与 LlamaIndex 的集成遇到困难**：MLflow 与 LlamaIndex 的集成产生了诸如 `TypeError: Ollama.__init__() got an unexpected keyword argument 'system_prompt'` 的错误，凸显了兼容性问题。
   - 进一步的测试显示，在创建带有外部存储上下文的向量存储索引时出现失败，表明需要进行故障排除。
- **AI21 Labs 发布 Jamba-Instruct 模型**：AI21 Labs 推出了 **Jamba-Instruct 模型**，通过 LlamaIndex 为 RAG 应用提供 **256K token** 的上下文窗口。
   - 一篇客座文章强调，有效利用长上下文窗口是应用获得最佳结果的关键。
- **开源改进提升 LlamaIndex 功能**：用户为 **BedrockConverse 模型** 贡献了异步功能，解决了 GitHub 上主要的集成问题。
   - 这些贡献增强了性能和效率，使整个 LlamaIndex 平台受益。
- **全文档检索简化 RAG**：一篇题为 [Beyond Chunking: Simplifying RAG with Full-Document Retrieval](https://medium.com/ai-advances/beyond-chunking-simplifying-rag-with-full-document-retrieval-911c757cb399) 的文章讨论了一种新的 RAG 技术方法。
   - 该方法提议用全文档检索取代传统的分块（chunking），旨在实现更高效的文档处理流程。
- **质量问题困扰 Medium 内容**：社区对 **Medium** 上的内容质量表示担忧，导致有建议认为社区应该放弃该平台。
   - 成员们指出，该平台似乎充斥着低价值内容，影响了其可信度。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **LLAMA_3 在不同平台上的输出存在差异**：一位用户测试了 **LLAMA_3 8B (instruct)** 模型，并注意到其输出质量不如另一个 Playground 的结果（[链接](https://sdk.vercel.ai/playground)）。他们质疑为什么即使参数相同，相似的模型也会产生不同的结果。
   - 这种差异强调了理解导致不同环境下模型输出变化的因素的必要性。
- **生成参数受到关注**：成员们讨论了 **生成参数** 的差异可能导致的不一致性，有人指出不同平台的默认值可能有所不同。另一位成员指出，缺失 **top_p** 和 **frequency_penalty** 可能会显著影响输出质量。
   - 这次对话强调了模型设置统一性对于在不同环境下获得一致性能的重要性。
- **分享 ChatPreferenceDataset 的变更**：分享了针对 [ChatPreferenceDataset](https://gist.github.com/RdoubleA/fb6dbf0db0099eafbadd31fe789459d1) 的本地更改，以增强消息转换和提示词模板（prompt templating）的组织。在澄清相关问题后，成员们表示准备好继续推进。
   - 这表明了根据当前的 RFC 标准优化数据集结构的协作努力。
- **FSDP2 预计将支持量化**：**FSDP2** 预计将处理量化（quantization）和编译（compilation），解决之前 **FSDP** 的局限性。讨论揭示了对 **QAT**（量化感知训练）与 FSDP2 兼容性的担忧，促使进一步的测试。
   - 成员们继续探索 FSDP2 的实际应用，以及它如何增强当前的训练方法。
- **提议合并 PR 以统一数据集**：围绕将更改合并到统一的数据集 PR 中展开了讨论，一些人建议如果合并发生，将关闭自己的 PR。一位成员表示，在审查另一个待处理的 PR 后，他们将提交一个单独的 PR。
   - 这反映了通过积极的贡献和讨论来简化数据集管理的持续协作和优先级排序。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Google Gemini Caching 引起困惑**：一位用户就 **Google Gemini context caching** 是否已集成到 LangChain 提出了疑问，理由是关于该功能的信息不明确。
   - 参与者确认支持 `gemini-pro` 等 **Gemini models**，但关于缓存的细节仍然模糊。
- **解锁 Agent 的流式 Token**：一份指南分享了如何使用 LangChain 中的 **.astream_events** 方法流式传输 token，从而实现异步事件处理。
   - 该方法特别有助于打印 **on_chat_model_stream** 事件内容，增强了交互能力。
- **构建 SWE Agents 指南已发布**：发布了一份使用 [CrewAI](https://git.new/swe/kit) 和 **LangChain** 等框架创建 **SWE Agents** 的新指南。
   - 它展示了一个专为在不同环境中进行脚手架友好型 Agent 创建而设计的 **Python framework**。
- **Palmyra-Fin-70b 为金融 AI 设定基准**：新推出的 **Palmyra-Fin-70b** 模型在 CFA Level III 考试中获得了 **73%** 的分数，已准备好执行金融分析任务。
   - 您可以在 [Hugging Face](https://huggingface.co/Writer/Palmyra-Fin-70B-32K) 和 [NVIDIA NIM](https://build.nvidia.com/writer/palmyra-fin-70b-32k) 上以非商业许可证找到它。
- **Palmyra-Med-70b 在医疗基准测试中占据主导地位**：**Palmyra-Med-70b** 在 MMLU 测试中达到了令人印象深刻的 **86%**，提供 **8k 和 32k 版本**用于医疗应用。
   - 该模型的非商业许可证可以在 [Hugging Face](https://huggingface.co/Writer/Palmyra-Med-70B) 和 [NVIDIA NIM](https://build.nvidia.com/writer/palmyra-med-70b) 上获取。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **澄清 Open Interpreter 工作流**：一位用户寻求关于在 **Llama 3.1** 中使用 **Open Interpreter** 的明确指令，特别是应该在终端会话还是新会话中提出问题。*OS 模式需要 vision model* 才能正常运行。
   - 这一询问反映了对 Open Interpreter 设置中工作流优化的关注。
- **关于 4o Mini 的兼容性问题**：一位用户询问 **Open Interpreter** 与新发布的 **4o Mini** 的协作效果，暗示了未来增强的潜力。然而，尚未提供具体的兼容性细节。
   - 这表明人们对利用新硬件配置以实现更好 AI 集成的兴趣日益浓厚。
- **对眼动追踪技术的兴奋**：一位成员对在 Open Interpreter 中实现 **eye tracking software**（眼动追踪软件）表现出极大的热情，并指出其辅助残障人士的能力。他们表达了通过这一创新增强无障碍功能的渴望。
   - 该倡议因其在 AI 领域的社会影响潜力而受到赞扬。
- **Perplexica 提供本地 AI 解决方案**：最近的一段 [YouTube 视频](https://www.youtube.com/watch?v=V0vx94JYNjI) 重点介绍了如何使用 **Meta AI** 的开源 **Llama-3** 构建 Perplexity AI 的**本地免费克隆版**。*这种本地解决方案旨在超越现有的搜索技术*，同时提供更好的可访问性。
   - 该项目旨在成为当前搜索 AI 的有力挑战者，吸引了开发者的关注。
- **在 GitHub 上查看 Perplexica**：[Perplexica](https://github.com/ItzCrazyKns/Perplexica) 作为一个 **AI-powered search engine**（AI 驱动的搜索引擎）出现，是 **Perplexity AI** 的开源替代方案。鼓励开发者探索其功能并做出贡献。
   - 该倡议旨在促进协作开发工作，同时增强搜索能力。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 与符号学习（Symbolic Learning）集成**：DSPy 已与符号学习器集成，为项目的增强功能和模块化创造了令人兴奋的可能性。
   - 这一举措为更丰富的模型交互开辟了道路，使 DSPy 对开发者更具吸引力。
- **Creatr 在 ProductHunt 上备受关注**：一位成员分享了他们在 [ProductHunt 上发布的 Creatr](https://www.producthunt.com/posts/creatr-3)，旨在收集对其新产品设计工具的反馈。
   - 支持者迅速投票，强调了在其编辑功能中创新性地使用 DSPy 来简化产品工作流。
- **DSPy 中的缓存管理问题**：有人提出关于如何完全删除 DSPy 中的缓存以解决测试指标不一致的问题。
   - 这一问题强调了对模块状态进行更清晰管理的必要性，以确保测试结果的可靠性。
- **通过 Schema-Aligned Parsing 增强 JSON 输出**：建议使用 [结构化生成（structured generation）](https://www.boundaryml.com/blog/schema-aligned-parsing) 来改进 JSON 输出解析，以获得更可靠的结果。
   - 利用 Schema-Aligned Parsing 技术旨在减少 Token 使用量并避免重复解析尝试，从而提高整体效率。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **UCSC 研讨会深入探讨并行计算（Parallel Computing）**：一位成员分享了 2024 年 4 月 10 日加州大学圣克鲁兹分校（UC Santa Cruz）CSE 研讨会的 [这段 YouTube 视频](https://youtu.be/c52ziyKOArc?si=pAUdzwIQGXCtpk3T)，讨论了什么是优秀的并行计算机。
   - 演示幻灯片可以通过视频描述中的链接访问，为深入探索提供了现成的资源。
- **OpenCL 在 Mac 上遇到障碍**：关于在 Mac 上使用 OpenCL 时出现“资源不足（out of resources）”错误的查询，凸显了潜在的 Kernel 编译问题，而非资源分配问题。
   - 成员们对产生“无效 Kernel（invalid kernel）”错误表示困惑，表明在该环境下需要更好的调试策略。
- **巴西通过新投资计划大举押注 AI**：巴西宣布了一项雄心勃勃的 AI 投资计划，预留了到 2028 年高达 **230 亿雷亚尔** 的资金，其中包括一个耗资 **18 亿雷亚尔** 的**超级计算机**项目。
   - 该计划旨在通过大量资金和激励措施提振当地 AI 产业，但需经总统批准后方可实施。
- **纳税人的钱在资助科技巨头**：针对巴西的 AI 计划出现了一种幽默的观点，强调了纳税人的钱可能最终让 **NVIDIA** 等公司受益的讽刺性。
   - 这一讨论指向了关于公共资金分配和此类科技产业投资伦理影响的更广泛辩论。
- **JIT 编译策略受到审视**：在一次热烈的讨论中，成员们辩论了是为了效率只对 **model forward step** 进行 JIT，还是对整个 **step function** 进行 JIT。
   - 他们得出结论，除非有特定情况，否则通常首选对整个 step 进行 JIT，这突显了性能优化的考量。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **高盛（Goldman Sachs）转移 AI 重心**：成员们讨论了最近的一份 [高盛报告](https://link.to.report)，该报告表明其重心正从 GenAI 转移，反映了 AI 领域情绪的变化。
   - *注*：该报告引发了关于 AI 兴趣未来走向的进一步讨论。
- **AI 爱好者渴望更广泛的话题**：一位成员表达了对该频道关注点的兴奋，强调了 AI 爱好者对 GenAI 的浓厚兴趣。
   - 这种情绪促进了大家共同深入探讨更多样化 AI 主题的愿望。
- **深入探讨推荐系统（Recommendation Systems）**：一位用户提到他们主要关注 **推荐系统（recsys）**，标志着 AI 讨论中一个独特的兴趣领域。
   - 这一对话指向了在 recsys 应用中获得更深见解和进展的潜在机会。

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Trivia App 利用 LLM 生成问题**：一款利用 **LLM** 生成趣味问题的新型 Trivia App（问答应用）已经开发完成，可以点击 [这里](https://mihaiii-trivia.hf.space/) 访问。用户可以查阅 [How to Play](https://mihaiii-trivia.hf.space/how-to-play) 指南获取说明，此外还有 [Stats](https://mihaiii-trivia.hf.space/stats) 和 [FAQ](https://mihaiii-trivia.hf.space/faq) 链接。
   - 该应用旨在通过**动态问题生成**来增强娱乐性和学习效果，有效地将教育与游戏融合在一起。
- **通过游戏机制提升参与度**：根据用户反馈，该 Trivia App 引入了**游戏机制 (game mechanics)** 以增强用户参与度和留存率。重要特性包括*引人入胜的游戏玩法*和用户友好的界面，这些都有助于延长游戏时间。
   - 初步反馈表明，这些机制对于保持用户持久兴趣和促进交互式学习环境至关重要。

---

**Alignment Lab AI Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**LAION Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

# PART 2: 频道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1267936585283010640)** (1 条消息): 

> - `Llama 3.1 Launch`
> - `Argilla 2.0 Features`
> - `Peft v0.12.0 Release`
> - `Inference-as-a-Service with Nvidia`
> - `New AutoTrain Task for VLM Finetuning` 

- **Llama 3.1 的多语言实力**：Meta 发布了 [Llama 3.1](https://x.com/reach_vb/status/1815767864277606762)，包含 **405B、70B 和 8B** 参数版本，提供 **128K context**，并兼容包括**英语、西班牙语和泰语**在内的多种语言。
   - **405B** 版本的 Benchmark 高达 **85.2**，表现优于 **GPT4o 和 Claude** 等竞争对手，并附带更宽松的训练许可。
- **Argilla 2.0 预览**：[即将发布的 Argilla 2.0](https://x.com/argilla_io/status/1817945202432061792) 版本引入了一项备受期待的功能：**简便的数据集复制**，用于管理多个数据集。
   - 这一增强功能对于需要多种数据集配置的任务特别有用，并能简化数据管理流程。
- **Peft v0.12.0 发布新方法**：[Peft v0.12.0](https://x.com/julien_c/status/1817837045298978986) 推出了创新的参数高效方法，如 **OLoRA、X-LoRA 和 FourierFT**，增强了模型训练能力。
   - 这些进步旨在简化各种模型类型的 Fine-tuning 过程。
- **Hugging Face 与 Nvidia 联手**：Hugging Face 与 [Nvidia AI](https://x.com/NVIDIAAIDev/status/1818050230392398175) 合作提供 **Inference-as-a-Service**，能够利用开源 AI 模型快速构建原型。
   - 该服务促进了生产环境的平滑部署，将开发者与 Hugging Face 庞大的模型库连接起来。
- **AutoTrain 引入 VLM Finetuning**：宣布了 [VLM Finetuning](https://x.com/abhi1thakur/status/1816429924233687470) 的新任务提醒，使得在自定义数据集上 Fine-tuning **PaliGemma** 模型变得更加容易。
   - 此功能增强了 AutoTrain 的功能，并邀请用户为未来的改进建议更多的模型和任务。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/reach_vb/status/1815767864277606762)">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：Meta Llama 3.1 405B, 70B 和 8B 已发布 - 支持多语言、128K 上下文、Tool-use 和 Agent！在竞争中与 GPT4o 和 Claude Sonnet 3.5 旗鼓相当甚至更胜一筹，毫无疑问是目前最棒的开源 LLM！🐐 额外福利：它还附带...</li><li><a href="https://x.com/reach_vb/status/1818218875239977000)">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：Llama 3.1 8B 在 Mac 上运行，100% 本地，由 llama.cpp 驱动 🔥 只需两步：1. brew install llama.cpp 2. llama-cli --hf-repo reach-vb/Meta-Llama-3.1-8B-Instruct-Q6_K-GGUF \ --hf-file meta-llama-3.1...</li><li><a href="https://x.com/argilla_io/status/1817945202432061792)">来自 Argilla (@argilla_io) 的推文</a>：💫 期待 Argilla 2.0 的发布吗？请关注即将推出的更新！与此同时，我们很高兴能提前展示一项呼声极高的功能：简便的数据集复制。...</li><li><a href="https://x.com/julien_c/status/1817837045298978986)">来自 Julien Chaumond (@julien_c) 的推文</a>：如果你错过了上周的消息：PEFT v0.12.0 刚刚发布 🔥 包含了一些酷炫的新参数高效（param-efficient）方法，如 OLoRA、X-LoRA、FourierFT 等</li><li><a href="https://x.com/micuelll/status/1816851392134586540)">来自 Miquel Farré (@micuelll) 的推文</a>：Hugging Face 进军视频领域！我们希望缩小与闭源视频模型的差距，这是我们的第一步。权重：https://huggingface.co/mfarre/Video-LLaVA-7B-hf-CinePile 代码：https://github.com/mfarre/V...</li><li><a href="https://x.com/abidlabs/status/1818034189348053204)">来自 Abubakar Abid (@abidlabs) 的推文</a>：感谢 @mmitchell_ai 提交的优秀 PR，为 Gradio 增加了仅需一个参数即可为 AI 生成的视频添加水印的功能 😎</li><li><a href="https://x.com/davidberenstei/status/1817115209590272021)">来自 David Berenstein (@davidberenstei) 的推文</a>：⚗️ 在 Hugging Face Hub 上查找可重用的合成数据流水线（pipeline）代码及对应的数据集。找到你的流水线并使用 `$ distilabel pipeline run --config "hugging_face_dataset_url/pipeline....</li><li><a href="https://x.com/abhi1thakur/status/1816429924233687470)">来自 abhishek (@abhi1thakur) 的推文</a>：🚨 新任务预警：VLM 微调 🚨 AutoTrain 刚刚添加了 VLM 微调功能：支持 PaliGemma 的 Captioning 和 VQA。现在，在自定义数据集上微调 PaliGemma 变得超级简单。哪个模型和任务...</li><li><a href="https://x.com/NVIDIAAIDev/status/1818050230392398175)">来自 NVIDIA AI Developer (@NVIDIAAIDev) 的推文</a>：我们与 Hugging Face 合作推出了推理即服务（inference-as-a-service），帮助开发者利用托管在 Hugging Face Hub 上的开源 AI 模型快速构建原型并部署到生产环境。➡️https://...</li><li><a href="https://x.com/RisingSayak/status/1818133546411728903)">来自 Sayak Paul (@RisingSayak) 的推文</a>：随着越来越大的 Diffusion Transformer 出现，拥有优秀的量化工具变得日益重要。我们展示了一系列实验的研究结果...</li><li><a href="https://x.com/mervenoyann/status/1816857371416887653)">来自 merve (@mervenoyann) 的推文</a>：你知道 Hugging Face 有一个开源的 Cookbook，里面包含许多 AI 应用案例（recipes）吗？🤩📖 这里有一些最新贡献的案例 🧶</li><li><a href="https://x.com/_philschmid/status/1816514989982908591)">来自 Philipp Schmid (@_philschmid) 的推文</a>：听说你喜欢图表。👀 所以我用 BigCodeBench 和 Aider（代码编辑）制作了一个针对代码的图表。我们真的应该停止使用 HumanEval 来评估编程能力了！🧑🏻‍💻 > BigCodeBench 评估 LL...</li><li><a href="https://x.com/davidberenstei/status/1816419520447127728)">来自 David Berenstein (@davidberenstei) 的推文</a>：Meta Llama-3.1 模型系列可用于蒸馏和微调，但这需要标注好的偏好数据，因此我基于 Gradio 创建了一个人类反馈收集器（Human Feedback Collector），可以直接记录数据...
</li>
</ul>

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1267921480922955807)** (395 条消息🔥🔥): 

> - `Knowledge Distillation`
> - `Community Interactions` (社区互动)
> - `AI Training Techniques` (AI 训练技术)
> - `Fine-tuning Models` (`Fine-tuning` 模型)
> - `Dialectal Language Processing` (方言语言处理)


- **Knowledge Distillation 讨论**：成员们分享了关于 7B 模型 `Knowledge Distillation` 的 `hyperparameters` 见解以及该过程所需的计算资源。
   - 对话强调了在有效设置这些模型时需要社区支持。
- **社区关于 AI 和编程的趣谈**：频道轻松地讨论了编程中的怪癖，引用了用户使用 AI 模型的经验和幽默互动。
   - 关于 "touching grass"（接触大自然）和现实生活互动的评论与对 AI 技术的关注形成了对比。
- **训练技术和 Learning Rates**：讨论围绕 `Learning Rates` 对模型性能的影响展开，特别是关于不同模型规模的 `Pre-training`。
   - 用户辩论了正确配置 `Learning Rates` 以促进有效模型训练的重要性。
- **阿拉伯语模型的方言支持**：提出了如何训练模型以用户指定的方言生成阿拉伯语回答的问题，并分享了关于数据标注和模型训练的见解。
   - 建议了创建包含方言请求和标准输入的训练对的指南。
- **使用 RAG (Retrieval-Augmented Generation) 训练模型**：一位用户表达了对构建 RAG 模型的兴趣，并向社区中的经验丰富成员寻求建议。
   - 回复包括了如何为此类模型有效 `bootstrap` 训练数据的建议。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/torchchat-local-llm-inference/">Introducing torchchat: Accelerating Local LLM Inference on Laptop, Desktop and Mobile</a>: 今天，我们发布了 torchchat，这是一个展示如何在笔记本电脑、台式机和移动设备上无缝且高性能地运行 Llama 3、3.1 以及其他大型语言模型的库。  </li><li><a href="https://discuss.huggingface.co/t/how-to-load-large-model-with-multiple-gpu-cards/18522">How to load large model with multiple GPU cards?</a>: 这可能是一个简单的问题，但困扰了我整个下午。我正尝试使用预训练的 m2m 12B 模型进行语言处理任务（44G 模型文件）。我有 8 张 Tesla-V100 GPU 卡，每张...</li><li><a href="https://llm.extractum.io/list/?mtr=nroggendorff">Maintainer &laquo;nroggendorff&raquo;</a>: 开源 LLM 和 SLM（大型和小型语言模型）的精选列表。维护者 «nroggendorff»，支持动态排序和过滤。</li><li><a href="https://pypi.org/project/keras/">keras</a>: 多后端 Keras。</li><li><a href="https://huggingface.co/FunAudioLLM/SenseVoiceSmall">FunAudioLLM/SenseVoiceSmall · Hugging Face</a>: 未找到描述</li><li><a href="https://www.tensorflow.org/guide/keras">no title found</a>: 未找到描述</li><li><a href="https://huggingface.co/amd">amd (AMD)</a>: 未找到描述</li><li><a href="https://tenor.com/view/tuh-buh-guh-cuh-what-gif-9750912507529527670">Tuh Buh GIF - Tuh Buh Guh - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://open.spotify.com/track/6y5HLopYu7Uu0hYwVBj4T6">palm of my hands</a>: 歌曲 · John Summit, venbee · 2024</li><li><a href="https://esolangs.org/wiki/Chicken">Chicken - Esolang</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

vandutech: 不客气！很高兴你觉得它有用。也感谢你的反馈。
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1268010891442524301)** (1 条消息): 

> - `Quantizing Diffusion Models`
> - `Transformer-based Diffusion Backbones`
> - `High-resolution Text-to-Image Generation`
> - `Memory Requirements in Large Models` 


- **现在可以对 Diffusion Models 进行量化**：一项新的突破允许对 Diffusion Models 进行量化，从而提升其性能和效率，详见[这篇文章](https://huggingface.co/blog/quanto-diffusers)。
   - *Partypepe emoji reaction* 表示社区内的兴奋和认可。
- **基于 Transformer 的模型改变了 T2I 格局**：最近的趋势显示，在**高分辨率文本生成图像 (T2I)** 中，基于 Transformer 的 Diffusion Backbones 使用量有所增加，逐渐取代传统的 UNet 架构。
   - 这些模型的可扩展性令人印象深刻，范围从 **0.6B 到 8B 参数**，提升了模型能力。
- **规模扩大带来了内存挑战**：随着 Diffusion Models 变得越来越大，**内存需求**也随之增加，由于包含文本编码器和图像解码器等多个组件，实现变得更加复杂。
   - 这一挑战进一步强调了 Diffusion Pipelines 架构中对效率的需求。



**提及的链接**：<a href="https://huggingface.co/blog/quanto-diffusers">使用 Quanto 和 Diffusers 的内存高效 Diffusion Transformers</a>：未找到描述

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1267938573227921408)** (12 条消息🔥): 

> - `SAM v2 Model Updates`
> - `Trivia Question Generation with LLM`
> - `Palmyra Domain-Specific Models`
> - `Article Summary on Instruction Hierarchy`
> - `Llama.cpp Utilization` 


- **增强型 SAM v2 应用支持多重掩码**：使用**最新 SAM v2 模型**生成分割掩码的应用现在支持多个边界框，并具有简化的 UI，可为所有提供的框输出单个掩码图像。
   - 分享了一个完整的工作流视频，尽管有来自共享办公空间的背景噪音，但仍突出了更新的功能。
- **LLM 趣味问答生成 Space 发布**：一个新的 Hugging Face Space 使用**语言模型**生成趣味问答题，允许用户为题目提出各种主题。
   - 该 Space 附带了关于*如何玩*、*统计数据*以及*常见问题解答*的详细指南。
- **Palmyra-Fin 和 Palmyra-Med 模型发布**：发布了两个新模型 **Palmyra-Fin-70b** 和 **Palmyra-Med-70b**，具有令人印象深刻的性能指标，包括以 **73%** 的分数通过了 CFA Level III 考试。
   - 这些模型专为金融和医疗应用设计，可在 **Hugging Face** 和 **NVIDIA NIM** 上通过开放模型许可获取。
- **LLM 指令层级摘要**：总结了一篇讨论**特权指令 (privileged instructions)** 在生成更有效的语言模型中作用的文章，可能对微调有用。
   - 可以通过提供的 Medium 链接访问全文以获取更多见解。
- **利用 Llama.cpp 运行 LLM**：趣味问答生成项目利用 **llama.cpp** 运行其语言模型服务器，并直接编写 Prompt 而不是使用绑定 (bindings)。
   - 这种方法在关于模型选择的简短讨论中得到了肯定，重点在于简单的实现。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/lightly-ai/SAMv2-Mask-Generator">SAMv2 Mask Generator - 由 lightly-ai 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://mihaiii-trivia.hf.space/">FastHTML 页面</a>：未找到描述</li><li><a href="https://antimatter543.github.io/2024/04/03/anti-does-socialisation">Anti 关于社交的不全面笔记与思考</a>：关于如何与人社交的指南，包含粗略的数据和想法。让我们建立一个社交模型！</li><li><a href="https://x.com/samjulien/status/1818652901130354724">来自 Sam Julien (@samjulien) 的推文</a>：🔥 @Get_Writer 刚刚发布了 Palmyra-Med-70b 和 Palmyra-Fin-70b！Palmyra-Med-70b 🔢 提供 8k 和 32k 版本 🚀 MMLU 性能约 86%，超越顶级模型 👨‍⚕️ 用于诊断、治疗规划...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1268182673256615988)** (2 messages): 

> - `Hugging Face ML Tasks`
> - `Face Recognition Task` 


- **Hugging Face 机器学习任务概览**：Hugging Face 展示了多种 **Machine Learning 任务**，提供 **demos、用例、模型**和**数据集**以协助用户入门。
   - 重点任务包括拥有 **13,541 个模型**的 **Image Classification** 和拥有 **2,361 个模型**的 **Object Detection** 等。
- **关于人脸识别可用性的咨询**：一位成员询问了 **face recognition 任务**的可用性，注意到它并未出现在特色任务列表中。
   - 这引发了关于是否会将人脸识别等额外模型或任务纳入 **Hugging Face** 产品线的问题。



**Link mentioned**: <a href="https://huggingface.co/Tasks">Tasks - Hugging Face</a>: no description found

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1267945356491227159)** (4 messages): 

> - `Seq2Seq tasks limitations`
> - `Referenceless metrics`
> - `Finetuning models` 


- **Seq2Seq 任务需要参考标签**：Seq2Seq 任务存在明显的**局限性**，因为它们主要依赖参考标签或黄金标准标签进行质量评估。
   - 讨论建议通过词典或本体使用**伪标签 (pseudo labels)**，尽管这需要 **BabbelNET** 的覆盖才能生效。
- **无参考指标缺乏深度**：任何使用**无参考指标 (referenceless metric)** 的方法可能只是从抽象角度衡量质量，缺乏针对特定任务的见解。
   - 讨论的一个例子是 **BRISQUE** 指标，它通过与已知的自然分布对比来确定图像的自然度，这使得它在医疗扫描等专业领域的作用较小。
- **模型微调的必要性**：针对一个提问，确认了微调是必要的，因为**分类头 (classification heads)** 预初始化时带有随机权重。
   - 这一过程对于模型根据特定数据集有效地进行预测至关重要。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1267921406562406534)** (5 messages): 

> - `Knowledge Distillation of 7B Model`
> - `State-of-the-Art Image Generation`
> - `Integrating Ollama RAG with WhatsApp`
> - `Using ONNX Models in Android Apps` 


- **寻求知识蒸馏设置方面的帮助**：一位用户正在寻求关于 **7B 模型**进行**知识蒸馏 (knowledge distillation)** 的**超参数设置**和计算资源估算的帮助。
   - *欢迎任何指导或经验分享。*
- **实现 SOTA 图像生成**：一位成员庆祝在内部实现了 **state-of-the-art 图像生成**，并分享了他们的公告链接。
   - 您可以查看他们的 [第一条推文](https://twitter.com/DataPlusEngine/status/1818358813520441493) 和 [第二条推文](https://vxtwitter.com/DataPlusEngine/status/1818356594780090517) 了解更多详情。
- **将 Ollama RAG 与 WhatsApp 集成**：一位用户询问了关于将 **Ollama RAG** 与 **WhatsApp** 集成的资源。
   - *鼓励任何有相关经验或链接的人进行分享。*
- **寻求 ONNX 模型使用指导**：一位成员请求关于如何在 **Android 应用**中使用 **ONNX 模型**的建议、代码或博客。
   - *非常欢迎具体的资源或示例。*


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1267923422038593687)** (213 条消息🔥🔥): 

> - `Gemma 2 模型更新`
> - `Multigpu 支持进展`
> - `在微调中使用 Lora`
> - `4bit 合并的问题`
> - `Unsloth 的安装挑战` 


- **Gemma 2 模型更新与性能**：发布了新的 **Gemma 2 (2B)** 模型，其在微调期间的性能和效率表现出色，速度提升 **2 倍** 且 **VRAM 占用减少 65%**。
   - 最新的模型允许在 **80GB GPU** 上进行 **高达 86k tokens** 的训练，显著增强了上下文长度能力。
- **Multigpu 支持进展与用户反应**：用户对 **multigpu 支持** 的开发表达了持续的关注和急迫感，提到了过去的承诺以及对更新的需求。
   - 尽管存在挫败感，一些用户确认在 beta 测试中成功使用了 **multigpu 设置**，而其他用户则在寻求关于时间线的明确沟通。
- **使用 Lora 方法进行微调**：一种将 **Lora** 与目标模型合并的方法被证明是有效的，允许像 **Qwen2-1.5B** 这样的模型在保留 instruct 能力的同时支持代码微调。
   - 鼓励用户测试并分享该方法的结果，该方法强调了短时间训练的好处。
- **4bit 合并的问题**：用户对合并 **4bit 模型** 提出了担忧，并被告知在与 **Lora** 结合时可能无法产生预期结果。
   - 反馈表明建议使用 **fp16 权重** 进行合并，因为遇到的问题似乎与使用 4bit 模型有关。
- **Unsloth 的安装挑战**：用户讨论了在本地环境设置 **Unsloth** 时的困难，提到了依赖管理和安装失败方面的挑战。
   - 尽管流程有所改进，许多人仍期待能有进一步的增强，使安装更加顺畅且用户友好。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1818686923315282424">来自 Unsloth AI (@UnslothAI) 的推文</a>：.@Google 发布了具有 2B 参数的新 Gemma 2 模型，它是同尺寸中性能最好的模型！Unsloth 使 Gemma 2 (2B) QLoRA 微调速度提升 2 倍，内存占用减少 65%。你知道吗...</li><li><a href="https://lightning.ai/lightning-ai/studios/unslothai-accelerate-llm-finetuning">UnslothAI: Accelerate LLM finetuning! - akshay 的 Lightning Studio</a>：了解 UnslothAI 如何显著加速 LLM 微调并减少内存使用。通过我们的实战指南开始优化您的模型，以实现更快的推理和更好的性能。</li><li><a href="https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing)">Google Colab</a>：未找到描述</li><li><a href="https://tenor.com/view/dancing-dj-ravine-groovy-mixing-music-party-gif-21277620">Dancing Dj Ravine GIF - Dancing Dj Ravine Groovy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://unsloth.ai/blog/llama3">使用 Unsloth 微调 Llama 3</a>：通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，上下文长度增加 6 倍！</li><li><a href="https://x.com/danielhanchen/status/1818706474404921580">来自 Daniel Han (@danielhanchen) 的推文</a>：我对 Gemma-2 2b 的分析+更新：1. 从未命名模型蒸馏出的 2T tokens？！2. Flash Attention 支持 softcapping！bf16 采用 O(N) 内存而非 O(N^2) 3. 提醒 - 将 head_dim 修改为 25...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eg5wgb/llama_31_changed_its_chat_template_again/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://youtu.be/TKmfBnW0mQA?si=fY9dXOpPMvE9YKQ8">修复 Gemma, Llama, &amp; Phi 3 中的 Bug：Daniel Han</a>：我们为 Gemma 提供的 8 个 bug 修复、为 Llama 3 提供的多个分词修复、滑动窗口 bug 修复以及将 Phi-3 Mistral 化的背后故事，并了解我们如何...</li><li><a href="https://youtu.be/pRM_P6UfdIc?feature=shared">LLM 的底层技术：Daniel Han</a>：本次研讨会将分为 3 个一小时板块：如何分析和修复 LLM - 如何发现并修复 Gemma, Phi-3, Llama 和分词器中的 bug，以及使用 U... 进行微调。</li><li><a href="https://github.com/wdlctc/mini-s">GitHub - wdlctc/mini-s</a>：通过在 GitHub 上创建账号来为 wdlctc/mini-s 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1268119566416875592)** (2 messages): 

> - `MegaBeam-Mistral`
> - `Long-context benchmarks` 


- **MegaBeam-Mistral-7B-512k 模型发布**：`MegaBeam-Mistral-7B-512k` 是一款支持 **524,288 tokens** 上下文的长上下文 LLM，基于 [Mistral-7B Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) 训练。它可以使用 [vLLM](https://github.com/vllm-project/vllm) 和 Amazon SageMaker 的 [DJL](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-models-frameworks-djl-serving.html) 端点等多种框架进行部署。
   - *评估显示*，该模型在三个长上下文基准测试中进行了测试，并通过 vLLM 的 OpenAI API 生成响应。
- **社区发现 MegaBeam 引起关注**：一位成员对 **MegaBeam-Mistral** 模型表示兴奋，称其看起来非常酷！社区对该新模型提供的能力表现出了浓厚兴趣。



**提到的链接**：<a href="https://huggingface.co/aws-prototyping/MegaBeam-Mistral-7B-512k">aws-prototyping/MegaBeam-Mistral-7B-512k · Hugging Face</a>：未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1268010117719392279)** (135 messages🔥🔥): 

> - `Quantization Methods`
> - `Hugging Face API Errors`
> - `Model Fine-Tuning`
> - `Installation Issues with Unsloth`
> - `Inference Consistency` 


- **关于量化影响的讨论**：多位成员讨论了不同模型量化方法的效果，特别指出 4-bit 量化在推理过程中往往会导致响应变差。
   - 一位成员强调了他们之前使用 GGUF 量化成功返回正确答案的经验，但注意到目前结果的不一致性，例如模型表现异常。
- **Hugging Face API 加载错误**：一位用户分享了一个 API 错误响应，显示模型当前正在加载中，并给出了预计可用时间。
   - 这类错误表明在对模型发出进一步请求之前需要等待。
- **微调中的问题**：一位成员在模型微调时，发现本地环境与使用 Google Colab 之间的推理结果存在差异。
   - 他们报告称，虽然 Colab 产生了正确的答案，但本地量化的模型有时会返回乱码或混合语言。
- **Unsloth 安装困扰**：讨论揭示了 Unsloth 的安装挑战，特别是与 xformers 相关的错误，这在安装过程中造成了障碍。
   - 一些用户通过绕过某些要求或修改官方指南中的安装步骤获得了成功。
- **环境兼容性**：成员们辩论了成功安装和运行 Unsloth 所需的 Python 版本和硬件环境的兼容性问题。
   - 提到使用 Conda 或特定的 Python 版本会显著影响模型的安装成功率和性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/continued-pretraining#loading-lora-adapters-for-conti">Continued Pretraining | Unsloth Documentation</a>：又名持续微调（Continued Finetuning）。Unsloth 允许你进行持续预训练，使模型能够学习新语言。</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining#loading-lora-adapters-for-continued-finetuning">Continued Pretraining | Unsloth Documentation</a>：又名持续微调（Continued Finetuning）。Unsloth 允许你进行持续预训练，使模型能够学习新语言。</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">All Our Models | Unsloth Documentation</a>：查看下方列表了解我们所有上传的 4bit bnb 模型</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cst400/result_llama_3_mmlu_score_vs_quantization_for/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1wlCOvklww1YvACuIRrhkdFFH_vU7Hgbn?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://download.pytorch.org/whl/cu121/xformers-0.0.24-cp39-cp39-manylinux2014_x86_64.whl">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1268022586953171095)** (6 条消息): 

> - `Unsloth Inference 集成`
> - `模型评估策略`
> - `翻译模型就绪情况` 


- **将 Unsloth Inference 与 HuggingFace 集成**：一名成员创建了一个 [GitHub repository](https://github.com/kmahorker/unsloth-hf-inference)，用于将 Unsloth Inference 与 HuggingFace Inference Endpoints 集成，这对于托管模型非常有用。
   - 成员们讨论了该集成的潜在重要性，并关注是否可以将其链接到 Unsloth 官方文档中。
- **训练数据的评估策略**：一位用户建议使用 **20%** 的训练数据进行测试，并强调如果所有数据都已被使用，则需要进行人工评估或使用自动化工具。
   - 另一位成员表示打算尝试这种方法，表明了关于模型评估策略的持续讨论。
- **翻译模型开发**：一名成员提到 **Llama** 和 **Mistral** 都已为翻译任务做好准备，但可能需要额外的训练。
   - 这表明了对改进其功能的积极兴趣，成员们对进一步的进展表示期待。
- **协作评审流程**：一位用户对新的集成仓库表示兴奋，并承认需要时间进行彻底评审。
   - 这反映了社区的协作氛围，成员们表示支持在继续推进之前进行仔细评估。



**提到的链接**：<a href="https://github.com/kmahorker/unsloth-hf-inference">GitHub - kmahorker/unsloth-hf-inference: Custom Handler for Unsloth Inference with HuggingFace Inference Endpoints</a>: Custom Handler for Unsloth Inference with HuggingFace Inference Endpoints - kmahorker/unsloth-hf-inference

  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1268147378528194652)** (2 messages): 

> - `HDMI Eavesdropping` (HDMI 窃听)
> - `Continual Pre-training Insights` (持续预训练见解)
> - `Sailor Language Models` (Sailor 语言模型)
> - `Learning Rate Trade-offs` (学习率权衡)
> - `Replay Ratio Dynamics` (重放比例动态)


- **研究人员通过 HDMI 辐射进行窃听**：最近的一项研究探讨了通过分析来自 **HDMI 线缆**的电磁波来窃听数字视频显示的方法。研究人员提出使用深度学习从发射信号中重建显示的图像，由于 **10-bit encoding** 的存在，这比模拟方法更复杂。
   - 他们解决了频率调谐的挑战，并进行了详细的数学分析，以提高重建图像的清晰度。
- **持续预训练（Continual Pre-training）：学习率至关重要**：一位研究员分享了关于持续预训练的见解，强调了以往论文忽视的 **learning rate**（学习率）的关键作用。研究发现，在训练期间调整此参数时，原始领域和新领域的损失（loss）都是可预测的。
   - 关键发现强调了学习率与 **replay ratio**（重放比例）之间的权衡，表明更快的学习会导致更多的遗忘，特别是当学习率超过最佳阈值时。
- **针对东南亚语言的 Sailor 模型进展**：讨论重点介绍了 **Sailor**，这是一个专为**东南亚语言**量身定制的开源语言模型系列，它基于 **Qwen1.5** 进行持续预训练。这些模型展现出良好的适应性，但需要仔细管理 Token 数量和学习率。
   - 实际建议包括从较小的 Token 数量开始，并保持固定的重放比例，以优化从 **4B 到 14B 参数**范围内的模型学习。
- **揭示可预测的学习动态**：研究显示，在不同学习率下，**英语验证损失**与二次函数之间存在强相关性（99.36%）。相反，**马来语验证损失**遵循可识别的趋势，但预测性稍弱。
   - 研究人员提出了一个对数指标来最小化模型中的遗忘，强调了其在平衡最佳学习率设置中的重要性。
- **模型优化的实用技巧**：该研究为增强持续预训练提供了可操作的建议，包括尝试不同的学习率以及采用他们的 **RegMix method** 来实现最佳的数据混合平衡。这些策略已在各种模型参数上得到有效测试，以提高语言模型性能。
   - 他们还提到了其他技术，如 **Document-Level Code-Switching**（文档级代码切换）和**翻译数据**的影响，以进一步完善模型能力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.09717">Deep-TEMPEST: Using Deep Learning to Eavesdrop on HDMI from its Unintended Electromagnetic Emanations</a>：在这项工作中，我们通过分析从线缆和连接器（特别是 HDMI）中无意散发出的电磁波，解决了窃听数字视频显示的问题。T...</li><li><a href="https://x.com/sivil_taram/status/1818493088542998860?s=46">来自 Qian Liu 🔭 (@sivil_taram) 的推文</a>：关于持续预训练的见解：平衡学习与遗忘 🚀 # 介绍 最近，我读了几篇关于持续预训练的论文，但失望地发现...
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

_paradroid: https://arxiv.org/abs/2407.04620
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 messages): 

not_lain: 他们终于更新了桌面端应用，我现在可以在个人资料上添加状态了

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1267947664788553739)** (330 条消息🔥🔥): 

> - `SOTA image generation` (SOTA 图像生成)
> - `LLM reasoning capabilities` (LLM 推理能力)
> - `Gemma 2B performance` (Gemma 2B 性能)
> - `Dynamic memory systems` (动态内存系统)
> - `Q* implementation` (Q* 实现)


- **SOTA 图像生成成就**：一名成员庆祝在内部实现了 State-of-the-art (SOTA) 图像生成，并分享了相关的链接。
   - 随后讨论了各种模型的性能和特性，包括它们在生成具有美感的输出方面的应用。
- **LLM 推理与预测**：社区辩论了 LLM 的推理能力，并对自回归 Token 预测在解决问题中的有效性提出了疑问。
   - 有人指出，虽然模型中的高 Temperature 设置有时可以产生正确答案，但在符号系统中仍存在显著的推理挑战。
- **Gemma 2B 性能见解**：成员们讨论了 Google DeepMind 的 Gemma 2B IT 的性能，它在 LMSYS Arena 上获得了 1130 分，超过了 GPT-3.5 和其他模型。
   - 人们对这些结果的可靠性表示担忧，并将其与既有模型和基准测试进行了比较。
- **动态内存系统的新颖想法**：引入了在 LLM 中操纵聊天历史的概念，引发了关于现有角色扮演策略和类 RAG 系统的讨论。
   - 成员们对这种方法的新颖性表示怀疑，并辩论了其在实际应用中的效果。
- **关于 Q* 实现的讨论**：社区仔细审查了关于 Q* 开源实现的说法，质疑了所提出的方法论的有效性和重要性。
   - 批评指向了模糊的术语以及所描述方法中可能缺乏真正的创新。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mihaiii-trivia.hf.space/">FastHTML 页面</a>: 未找到描述</li><li><a href="https://x.com/midjourney/status/1818342703618482265">来自 Midjourney (@midjourney) 的推文</a>: Midjourney V6.1 现已上线！V6.1 大幅提升了图像质量、连贯性、文本表现，并配备了全新的放大和个性化模型。它更智能、更快速、更清晰、更美观。我们...</li><li><a href="https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu">Papers with Code - MMLU 基准测试 (多任务语言理解)</a>: 目前 MMLU 上的 SOTA 是 Gemini Ultra ~1760B。查看 109 篇带有代码的论文的完整对比。</li><li><a href="https://x.com/ryunuck/status/1818709409239121975?s=46">来自 ryunuck (p≈np) (@ryunuck) 的推文</a>: Ilya 所看到的。CRISPR-Q 运行在 Sonnet 3.5 上，并使模型能够通过其自身 self-memeplex 的定向操作来重写上下文窗口。这种难以理解的异类生成启发式...</li><li><a href="https://x.com/dylan522p/status/1818414482051235994">来自 Dylan Patel (@dylan522p) 的推文</a>: 当面对拥有大量计算资源的创始人时，占主导地位的男性会穿上一件更蓬松的皮夹克，以求在吸引配偶的竞争中获胜。这种竞赛类似于...</li><li><a href="https://x.com/_philschmid/status/1818686186472325219">来自 Philipp Schmid (@_philschmid) 的推文</a>: 太疯狂了！🤯 @GoogleDeepMind Gemma 2B IT 在 @lmsysorg Arena 上获得了 1130 分！这超过了 @OpenAI GPT-3.5, @Microsoft Phi-3 Medium (14B), @MistralAI 8x7B Instruct。</li><li><a href="https://tenor.com/view/assaultron-sexy-fallout-robots-fallout-4-gif-1726009637196075703">Assaultron Sexy GIF - Assaultron Sexy Fallout - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/muahaha-evil-laugh-evil-laugh-futurama-gif-4133163">Professor Farnsworth - Evil Laugh GIF - Muahaha Evil Laugh Evil - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/holo-q/OpenQ/">GitHub - holo-q/OpenQ: Q* 的开源实现，通过对 Attention 机制进行 In-context Zero-shot 重编程实现。（合成数据）</a>: Q* 的开源实现，通过对 Attention 机制进行 In-context Zero-shot 重编程实现。（合成数据） - holo-q/OpenQ
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1268002415597195435)** (10 messages🔥): 

> - `Hugging Face Code Generation Leaderboard` (Hugging Face 代码生成排行榜)
> - `Mistral Prompting Issues` (Mistral 提示词问题)
> - `BigCodeBench Leaderboard` (BigCodeBench 排行榜)
> - `Character Card Specifications` (角色卡规范)


- **对 Hugging Face 排行榜的好奇**：一名成员询问 [Hugging Face 排行榜](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results) 是否是代码生成任务的主要资源。
   - 其他人建议 **BigCodeBench** 可能是另一种选择，但没有关于其具体细节的更多信息。
- **Mistral 模型提示词问题**：一位新成员报告了使用官方模板时 Mistral 模型出现的问题，指出系统消息（system message）被错误地放置在用户提示词（user prompt）之前 [{讨论链接}](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407/discussions/47)。
   - 他们在 Ollama 中创建了一个“修正后”的模板，将系统消息移至开头，并表示效果更好，同时征求他人的经验。
- **角色卡规范参考**：一名成员提到了一份托管在 [GitHub](https://github.com/malfoyslastname/character-card-spec-v2?tab=readme-ov-file#post_history_instructions) 上的关于角色卡规范（character card specifications）的相关资源。
   - 他们还提供了该仓库中角色卡规范的可视化参考。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/mike-ravkine/can-ai-code-results">Can Ai Code Results - mike-ravkine 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/malfoyslastname/character-card-spec-v2?tab=readme-ov-file#post_history_instructions">GitHub - malfoyslastname/character-card-spec-v2: AI 角色卡的更新规范。</a>：AI 角色卡的更新规范。可以通过在 GitHub 上创建账户来为 malfoyslastname/character-card-spec-v2 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1268195651770912770)** (11 messages🔥): 

> - `Website Rendering` (网站渲染)
> - `Netlify Automation` (Netlify 自动化)
> - `Subdomain Discussion` (子域名讨论)
> - `Domain Unification` (域名统一)


- **关于网站渲染的共识**：成员们讨论了为任务提供 **渲染版本** 以及使用 **Quarto** 进行渲染过程的想法。
   - 大家普遍认为这种方法将增强功能和用户体验。
- **使用 Netlify 自动化构建过程**：一名成员建议通过使用 **Netlify** 在合并后自动执行构建过程，从而为贡献者省去手动构建步骤。
   - 这将避免需要将 `_book` 输出文件夹提交到版本控制中。
- **项目的子域名建议**：一名成员提议创建一个子域名，例如 **openreasoningtasks.nousresearch.com**，以统一项目的网络存在。
   - 根据最近的建议，将其整合到 **nousresearch** 域名下可能会带来好处。
- **外部域名的配置流程**：成员们探讨了如果通过 **Netlify** 使用外部注册域名，则需要配置 DNS 设置的需求。
   - 引用了 Netlify 文档中关于子域名配置的具体指南。
- **Netlify 的成本考虑**：成员们询问了实施计划中的 Netlify 设置可能产生的相关费用。
   - 随着讨论的继续，出现了关于如何协助推进项目计划的问题。



**提及的链接**：<a href="https://docs.netlify.com/domains-https/custom-domains/configure-external-dns/#configure-a-subdomain">为自定义域名配置外部 DNS</a>：配置外部 DNS 提供商以将您的域名指向我们的平台。您可以为您在外部注册的子域名或顶级域名（apex domain）使用外部 DNS。

  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1267934718104703106)** (35 条消息🔥): 

> - `什么是 Accel？`
> - `分布式训练的学习材料`
> - `Tinyboxes 发货更新`
> - `线下主题演讲录制确认`
> - `Llama 3.1 推理面临的挑战` 


- **Accel: 风险投资公司**：[Accel](https://www.accel.com/) 是一家与优秀团队合作的风险投资公司，目前正在举办一项活动。
   - 他们通过 [40 周年纪念](https://40-years.accel.com/) 回顾了其著名的历史，纪念了关键时刻和贡献。
- **分布式训练的学习材料**：成员们讨论了学习分布式训练技术（包括 FSDP, TP 和 PP）的资源，推荐 [PyTorch docs](https://pytorch.org) 作为良好的起点。
   - 此外，还推荐了一篇关于 [FSDP 的 PyTorch 论文](https://arxiv.org/abs/2304.11277)，因其详细的解释和边缘案例。
- **Tinyboxes 发货更新**：Tinyboxes 现在正以较小批量发货，每周约 10 台，正如 [@__tinygrad__](https://x.com/__tinygrad__/status/1818408577155154280) 的帖子所述。
   - 他们提到发货时间在周一，表示预订名单正在按顺序处理。
- **线下主题演讲将被录制**：一位成员确认线下主题演讲确实会被录制，以便后续观看。
   - 这一消息受到了积极响应，确保了讨论内容能被更广泛地获取。
- **Llama 3.1 推理面临的挑战**：一位成员分享了尝试在两个节点上使用 8 x H100 GPU 对 Llama 3.1 进行分片（shard）时的困难，表达了推理过程中遇到的挑战。
   - 其他成员讨论了潜在的解决方案并分享了相关博客的见解，指出了高效运行此类大型模型的复杂性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2304.11277">PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel</a>：众所周知，大型模型有潜力在广泛的领域提供卓越的性能。尽管机器学习系统领域取得了显著进展...</li><li><a href="https://blog.vllm.ai/2024/07/23/llama31.html">Announcing Llama 3.1 Support in vLLM</a>：今天，vLLM 团队很高兴与 Meta 合作，宣布支持 Llama 3.1 模型系列。Llama 3.1 带来了令人兴奋的新功能，具有更长的上下文长度（高达 128K tokens）...</li><li><a href="https://x.com/__tinygrad__/status/1818408577155154280">Tweet from the tiny corp (@__tinygrad__)</a>：tinyboxes 在周一发货。每周约 10 台。正在按预订名单顺序处理。</li><li><a href="https://www.accel.com/">Accel</a>：Accel 是一家全球风险投资公司，是优秀团队从种子轮到 IPO 的首选合作伙伴。Facebook, Flipkart, CrowdStrike, UiPath 和 Spotify 都是 Accel 支持过的公司...</li><li><a href="https://www.snowflake.com/engineering-blog/fine-tune-llama-single-node-snowflake/">Fine-Tune Llama 3.1 405B on a Single Node using Snowflake’s AI Stack</a>：了解 Snowflake AI Research 如何利用创新的内存管理技术优化 Meta Llama 3.1 405B 等巨型 LLM 的微调，以实现高效的 AI 部署。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1268010120940355724)** (1 条消息): 

> - `代码反射 (Code Reflection)`
> - `Triton 编程模型`
> - `OpenJDK Project Babylon`
> - `GPU 编程` 


- **在 Java 中探索 Triton 的代码反射**：一篇文章详细介绍了如何在 Java 中使用 [Code Reflection](https://openjdk.org/projects/babylon/articles/triton) 来实现 **Triton** 编程模型，为 Python 提供了一种替代方案。
   - 它介绍了各种 **Code Reflection** 概念和 API，同时强调了它们在 OpenJDK Project **Babylon** 中的相关性。
- **Triton：一种 GPU 编程解决方案**：**Triton** 模型允许开发者使用 Python 编写可编译为 GPU 代码的程序，即使是对于那些几乎没有 GPU 经验的人也是如此。
   - 讨论强调了该模型通过利用 **中间表示 (IR)** 和 **MLIR dialects** 在简化 GPU 编程任务方面的潜力。



**提到的链接**：<a href="https://openjdk.org/projects/babylon/articles/triton">Exploring Triton GPU programming for neural networks in Java</a>：未找到描述

  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1267992893512224818)** (13 messages🔥): 

> - `CUDA Memory Alignment` (CUDA 内存对齐)
> - `torch.compile on Google Colab` (在 Google Colab 上运行 torch.compile)
> - `Non-blocking Data Transfer Issues` (非阻塞数据传输问题)
> - `Pinned Memory Usage in LLM Inference` (LLM 推理中的锁页内存使用)


- **CUDA 内存对齐关注点**：一位成员询问从 CUDA 缓存分配器返回的 GPU 内存是否始终是对齐的，理由是担心在使用 `reinterpret_cast` 时可能会遇到 **CUDA error: misaligned address**。
   - 另一位成员指出，虽然分配器通常确保对齐，但并不保证 PyTorch 中的每个张量指针都是对齐的。
- **在 T4 上运行 torch.compile 的问题**：一位成员在 Google Colab 的 T4 GPU 上尝试运行 `torch.compile` 时遇到了 **IndexError**，怀疑是版本不匹配导致的。
   - 他们寻求运行 PyTorch Nightly 版本的可靠方案，以便有效地调试此问题。
- **关于非阻塞传输的警告**：关于在某些 CUDA 流场景中使用 `non_blocking=True` 导致结果错误的问题引发了讨论，成员们分享了类似的经历。
   - 一位成员引用了过去显示出这些问题的代码，强调了在 torch 仓库中进行潜在问题追踪的必要性。
- **锁页内存在推理中的影响**：一位成员分享了他们在 **LLM 推理项目** 中使用锁页内存 (Pinned Memory) 和非阻塞传输的经验，表示由于 Batch Size 通常较小，性能提升非常有限。
   - 鉴于在 CUDA Graphs 中看到的意外行为，他们对是否继续使用这些方法表示不确定。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html">A guide on good usage of non_blocking and pin_memory() in PyTorch — PyTorch Tutorials 2.4.0+cu121 documentation</a>: 无描述内容</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L927">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

mobicham: https://arxiv.org/abs/2407.09717
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1268075164332199936)** (1 messages): 

> - `ML Performance Optimization` (ML 性能优化)
> - `Zoox team expansion` (Zoox 团队扩招)


- **Zoox 扩建 ML 平台团队**：**Zoox** 的 ML 平台团队正在扩张，并增加了一个新的 **ML 性能优化子团队**，专注于优化计划。
   - 他们正在寻找希望提高训练和推理平台速度和效率的**软件工程师**；更多细节可以在 [职位公告](https://jobs.lever.co/zoox/2ed8a665-cee8-4d70-bcb1-e96b6214b371) 中找到。
- **Zoox 令人兴奋的优化计划**：Zoox 的新子团队旨在使训练和推理过程变得**极速**且高效，这标志着其 ML 能力的战略性增强。
   - 这种对优化的推动反映了行业在寻求提高整体性能时，对精简化机器学习实践日益增长的需求。


  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1268324131812671594)** (1 messages): 

> - `Ampere A100 Architecture` (Ampere A100 架构)
> - `Warp Processing Efficiency` (Warp 处理效率)


- **Ampere A100 的处理块配置**：**Ampere A100** GPU 具有 **64 个核心**，组织成四个各含 **16 个核心** 的处理块，这引发了关于设计选择的疑问。
   - 一位成员询问，考虑到一个 Warp 是 **32 个线程**，为什么处理块不是由 **32 个核心** 构建的，以及这如何影响性能。
- **讨论 Warp 拆分的优势**：讨论强调了将一个 Warp 拆分到两个处理块上的潜在好处，可能会增强并发性和资源利用率。
   - 这种方法可以通过在架构内实现更有效的调度，来改善对多样化工作负载的处理。


  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1268059643309985843)** (13 条消息🔥): 

> - `Quantized Training Recipes` (量化训练方案)
> - `Post-Training Quantization` (训练后量化)
> - `Low Bit Optimizers` (低比特优化器)
> - `FP8 Support` (FP8 支持)
> - `Tutorial Format Discussion` (教程格式讨论)


- **探索量化训练方案**：一位成员建议查看 [Quantized Training Issue #554](https://github.com/pytorch/ao/issues/554)，以便为小型模型的量化训练添加方案。
   - 他们指出，集成这些方案可以增强 build-nanogpt 教程。
- **建议使用训练后量化**：另一位成员建议从 nanogpt 的推理代码开始，应用训练后量化（Post-Training Quantization）API。
   - 这种方法可以简化当前工作流中量化技术的采用。
- **关于低比特优化器的讨论**：有人提到将现有优化器更换为低比特优化器（Low Bit Optimizers）是一种简单的增强手段。
   - 这可能会提升小型模型训练的性能。
- **令人兴奋的 FP8 支持公告**：torchao 中引入 **FP8** 支持收到了热烈反响，并计划将相关仓库转换为 **FP8 training**。
   - 这一补充可以显著优化大型模型的训练需求。
- **教程代码格式咨询**：一位成员询问教程是否更倾向于使用 .py 脚本而非 .ipynb 笔记本，并表示愿意为了统一性进行重构。
   - 讨论内容包括脚本与笔记本在可修改性方面的考量，并强调了实际使用场景。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/jupyter/notebook/blob/main/docs/source/examples/Notebook/Running%20Code.ipynb?short_path=c932132">notebook/docs/source/examples/Notebook/Running Code.ipynb at main · jupyter/notebook</a>：Jupyter 交互式笔记本。欢迎在 GitHub 上为 jupyter/notebook 的开发做出贡献。</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md">ao/torchao/quantization/README.md at main · pytorch/ao</a>：用于训练和推理的自定义数据类型和布局 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao">GitHub - pytorch/ao: Custom data types and layouts for training and inference</a>：用于训练和推理的自定义数据类型和布局 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/issues/554">Quantized Training · Issue #554 · pytorch/ao</a>：受最近与 @gau-nernst 交流的启发，我们应该在 AO 中为小型模型（600M 参数范围）添加一些量化训练方案。Character.ai 最近分享了他们正在进行量化...</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/utils.py#L275">ao/torchao/utils.py at main · pytorch/ao</a>：用于训练和推理的自定义数据类型和布局 - pytorch/ao
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1267926065494495416)** (2 条消息): 

> - `Apple's LoRA Adapter Discoveries` (Apple 的 LoRA Adapter 发现)
> - `Llama 3.1-8B Instruct Model Performance` (Llama 3.1-8B Instruct 模型性能)


- **Apple “重新发现”了 HQQ+ 技术**：有趣的是，Apple 似乎“重新发现”了与 **4chan** 三个月前发布的关于 LoRA 中**量化损失（quantization loss）**相似的技术，展示了通过 [LoRA adapters](https://x.com/teortaxesTex/status/1818289206948716660) 实现的**精度恢复（accuracy recovery）**。
   - Blaze 强调所有特定任务的 Adapter 都是基于这个恢复了精度的 Base 模型进行微调的，突出了他们发现的重要性。
- **高性能 Llama 3.1-8B Instruct 模型可用**：一个新的 **Llama 3.1-8B Instruct 4-bit 量化模型**已发布，其性能非常接近 **fp16** 版本，可以在[这里](https://huggingface.co/mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq)找到。
   - 目前有两个版本：[免校准版本（calibration-free version）](https://huggingface.co/mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq/)和[校准版本（calibrated version）](https://huggingface.co/mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib/)，以满足不同的用户需求。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib">mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib · Hugging Face</a>：暂无描述</li><li><a href="https://x.com/teortaxesTex/status/1818289206948716660">Teortaxes▶️ (@teortaxesTex) 的推文</a>：顺便说一下，4chan 在 Apple 把它变酷的 3 个月前就做出了量化损失恢复 LoRA。引用 Blaze (Balázs Galambosi) (@gblazex) 的话：对我来说，Apple 论文中最有趣的事情之一是...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1267923185760866457)** (199 条消息🔥🔥): 

> - `SwiGLU 性能`
> - `FP8 挑战`
> - `RoPE 集成`
> - `Llama 3 实现`
> - `超参数调优` 


- **SwiGLU 在速度上优于 GELU**：最近的测试显示 **SwiGLU** 的收敛速度比 **GELU** 快，最终达到了相似的 loss 水平，这表明在稳定性方面可能存在权衡。
   - *SwiGLU* 可能会使代码复杂化，引发了关于其相对于 ReLU 等传统激活函数真实优势的讨论。
- **FP8 集成问题显现**：强调了 **FP8** 面临的挑战，特别是反向传播过程中的张量精度以及潜在的稳定性问题。
   - 讨论关注到需要适当的权重衰减（weight decay）和参数管理，以确保 FP8 实现下的稳定训练。
- **RoPE 与训练动态**：将 **RoPE** 集成到训练中已显示出良好的效果，讨论强调了它如何影响整体模型的效率和性能。
   - 与会者对 RoPE 的必要性及其与现有实现相比在性能目标上的一致性表达了不同看法。
- **Llama 3 更新导致进度停滞**：向 **Llama 3.1** 的过渡遇到了障碍，因为许多成员正尝试在缺乏清晰文档的情况下，调整现有代码以适配更新后的架构。
   - 已承诺协助实现一个利用新 Llama 变化的 Python 参考实现，同时保持现有的工作流。
- **超参数调优讨论兴起**：超参数调优被认为是至关重要的，**SwiGLU** 和 **GELU** 在不同学习率下表现出不同的结果，因此建议进行全面的参数扫描（sweeps）。
   - 集体见解表明需要进一步研究最佳配置，特别是随着模型容量和类型的演进。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/zealandic1/status/1818655640778322338)">来自 Anthonix (@zealandic1) 的推文</a>：继 SwiGLU 之后，今天的图表展示了更多 @karpathy 的 llm.c 向 Llama 3 演变的进展，这次加入了 RoPE.. 看起来很棒！🍿📈 感谢 @hyperbolic_labs 允许我进行测试...</li><li><a href="https://github.com/karpath">karpath - 概览</a>：GitHub 是 karpath 构建软件的地方。</li><li><a href="https://github.com/karpathy/llm.c/pull/721">由 ademeure 使用 MUFU.TANH 为 SM7.5+ 提供更快的 GELU 前向和反向计算 · Pull Request #721 · karpathy/llm.c</a>：这些是更快的 GELU kernel，利用了 NVIDIA 在 Turing (SM7.5) 中引入的硬件指令，但据我所知从未在 PTX 之外公开，可能是因为它稍微不那么精确...</li><li><a href="https://github.com/karpathy/llm.c/pull/679">由 ngc92 演示如何在没有太多样板代码的情况下跟踪激活值 · Pull Request #679 · karpathy/llm.c</a>：这还没准备好合并，但演示了我们如何使用 TensorSpec 数据轻松收集关于激活值的统计信息，因为 TensorSpec 允许我们直接遍历所有激活张量...</li><li><a href="https://github.com/karpathy/llm.c/pull/708">由 gordicaleksa 添加高性能模式 · Pull Request #708 · karpathy/llm.c</a>：添加：当我们进入次优分支时的警告；高性能模式，如果未运行所有最优化分支则立即退出；还添加了一个前向 kernel 配置，将用于...</li><li><a href="https://github.com/karpathy/llm.c/pull/715">特性/从 master 恢复，由 karpathy 提交 · Pull Request #715 · karpathy/llm.c</a>：未找到描述。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1268219001641893969)** (2 messages): 

> - `Ternary models speed boosts`（三元模型速度提升）
> - `Ternary-int8 dot product performance`（三元-int8 点积性能）
> - `CPU vs CUDA performance`（CPU 与 CUDA 性能对比）


- **三元模型实现 2 倍速度提升**：确认显示，在**无需定制硬件**的情况下，三元模型可以实现 **2 倍的速度提升**；相比之下，在*某些 CPU* 上，`Q8_0` 比 F16 快 **2 倍以上**。
   - 这一突破挑战了此前的推测，相关证据链接在 [Reddit 讨论](https://www.reddit.com/r/LocalLLaMA/comments/1egg8qx/faster_ternary_inference_is_possible/)中。
- **三元-Int8 点积性能突破**：对 `llama.cpp` 新型三元量化类型的调查在 AVX2 上的**三元-int8 点积性能**方面取得了突破。
   - 利用 **_mm256_maddubs_epi16** 已被证明能有效将**无符号三元值**与 8 位整数相乘，从而提高处理效率。
- **CPU 性能超越 CUDA 中复杂的位运算**：对比显示，一个 CUDA kernel 通过简单地与权重相乘，比采用复杂的**位运算（bitwise operations）**执行速度更快。
   - 在此背景下，**仅限 CPU** 的方法似乎为当前任务提供了更好的速度。



**提到的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/comments/1egg8qx/faster_ternary_inference_is_possible/">Reddit - Dive into anything</a>：未找到描述

  

---


### **CUDA MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1267969374858383361)** (8 messages🔥): 

> - `WebGPU Overview`（WebGPU 概览）
> - `gpu.cpp Usage`（gpu.cpp 用法）
> - `Real-time Multimodal Integration`（实时多模态集成）
> - `Hybrid Model Computation`（混合模型计算）
> - `Local Device Computation`（本地设备计算）


- **WebGPU 作为跨平台 GPU API**：WebGPU 是一项 **API 规范**，包含一个名为 **WGSL** (WebGPU Shading Language) 的小型语言定义，主要为浏览器提供跨平台 GPU 接口。
   - 这也促成了原生用例，特别是 **Rust** 中的 **wgpu**。
- **gpu.cpp 简化 WebGPU 集成**：**gpu.cpp** 旨在让 WebGPU 功能更容易嵌入到 **C++ 项目**中，通过使用 **WGSL** 编写着色器来避免繁琐的原生 API 环节。
   - 其目的是在利用 **WebGPU** 功能的同时提供更便捷的接口。
- **将 WebGPU 用于实时多模态应用**：一位用户表示有兴趣利用 **gpu.cpp** 将模型与**实时多模态**输入/输出（特别是音频和视频）集成。
   - 其他用途还包括各种模拟以及模型上的条件计算分支。
- **对混合模型计算的兴趣**：人们对探索**混合模型计算**表现出浓厚兴趣，即结合 **CPU SIMD** 和 **GPU** 资源，甚至是**本地和远程**计算。
   - 这反映了向更复杂的计算架构发展的趋势，以有效利用多样化资源。
- **使用 C++ 进行本地设备计算**：其中一个动机是希望尝试**本地设备计算**，强调了在 **C++** 中使用便携式 GPU API 的便利性。
   - 这种方法被视为尝试新计算方法的一种易于获取的基础。


  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1267932686547292250)** (11 messages🔥): 

> - `Event Registration`（活动注册）
> - `Compute Access`（算力获取）
> - `Funding for GPUs`（GPU 资金）
> - `Participant Engagement`（参与者互动）
> - `Venue Details`（场地详情）


- **活动注册确认**：一位参与者询问在注册后如果被批准参加线下（IRL）活动，是否会收到电子邮件。
   - *是的，我们会向被批准的人员发送确认。*
- **算力资源说明**：有提问关于参加者是否需要自带 GPU，或者是否会提供算力资源，并提到已经获得了一些算力额度。
   - *我们正在从赞助商那里筹集资金，很快会分享更多细节。*
- **与 PyTorch 大会行程合并**：一位参与者提到考虑从波士顿前往，有人指出该活动与 PyTorch 大会同地举办。
   - *这种安排可以高效地合并行程。*
- **欢迎新手学习**：一位新手表达了参加活动的极大热情，希望在黑客松式的环境中学习更多知识。
   - *他们的目标是吸收知识，尽管是新手身份，也会积极参与。*
- **在旧金山（SF）的合作机会**：一位来自 Gradient 的参与者分享了他们在针对特定架构的 **Seq Parallel** 或 **Triton Kernels** 方面进行合作的兴趣。
   - *他们邀请在旧金山的其他人联系并讨论潜在的合作。*


  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1268091222686175246)** (1 条消息): 

> - `Vulkan support update`
> - `OpenCL deprecation`
> - `ROCm support` 


- **Vulkan 支持更新将于明天发布**：计划于明天发布的更新将为 LM Studio 引擎引入 **Vulkan 支持**，从而提升 GPU 性能。
   - 此更改是在 llama.cpp 弃用 **OpenCL 支持**之后进行的，此前 OpenCL 因其速度低于 CPU 性能而备受关注。
- **ROCm 兼容性指南**：使用支持 **ROCm** 设备的设备的用户可以在 [官方指南](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md) 中找到更新 LM Studio 引擎的详细说明。
   - 可以在 [此处](https://rocm.docs.amd.com/en/docs-5.7.1/release/windows_support.html) 查看 ROCm 支持的 GPU，以确保兼容性。
- **Beta 版本现已发布**：最新更新的 **Beta 版本** 现已推出，可在正式发布前提前体验各项功能。
   - 有关 Beta 版本的讨论可以在 Discord 频道的 [此处](https://discord.com/channels/1110598183144399058/1166577236325965844/1268332941868666944) 找到。



**提到的链接**：<a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs

  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1267926984147861596)** (143 条消息🔥🔥): 

> - `LM Studio 更新`
> - `训练与模型使用`
> - `AI 对话管理`
> - `安装问题`
> - `模型支持与配置` 


- **LM Studio 的后续功能**：即将推出的 LM Studio 0.2.31 beta 版本承诺带来多项改进，包括对 Gemma 2 2B 的支持、Vulkan 后端以及 kv cache 量化选项。
   - 鼓励用户加入 Discord 中的 Beta Builds 角色，以接收新版本的通知。
- **模型加载与兼容性挑战**：多名用户报告了在 LM Studio 上加载 Gemma 2B 和 Phi 3 等模型时遇到的问题，特别是在更新到 0.2.29 版本之后。
   - 有人指出，新模型的发布通常需要对底层的 llama.cpp 代码进行相应的更新。
- **在多玩家场景中使用 AI**：一位用户正在探索一个 D&D 直播概念，其特点是使用多个 AI 作为玩家，旨在让它们以结构化的、回合制的方式进行互动。
   - 讨论中提出了关于游戏过程中对话动态处理和语音转文本（speech-to-text）复杂性的担忧。
- **安装与使用问题**：新用户表达了从 Hugging Face 下载模型时的困难，包括需要签署协议以及访问权限等问题。
   - 提供了变通方法，例如访问不需要点击协议确认的替代模型。
- **向模型喂文档**：目前，LM Studio 不支持通过拖放 CSV 文件的方式将文档喂给 LLM。
   - 仅具有 vision 能力的模型支持上传图片，这表明 RAG (retrieval-augmented generation) 功能目前存在局限性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html">系统要求 (Windows) — HIP SDK 安装 (Windows)</a>: 未找到描述</li><li><a href="https://tenor.com/view/soft-kobe-bryant-no-smh-shaking-my-head-gif-18860898">Soft Kobe Bryant GIF - Soft Kobe Bryant No - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B">meta-llama/Meta-Llama-3.1-405B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=MxTWLm9vT_o">AI 逆向图灵测试实验</a>: 一组拥有世界上最先进 AI 的群体试图找出谁是其中的人类。我在 Unity 中制作的实验。配音由 ElevenLabs 提供。</li><li><a href="https://github.com/homebrewltd/awesome-local-ai">GitHub - homebrewltd/awesome-local-ai: 一个优秀的本地 AI 工具仓库</a>: 一个优秀的本地 AI 工具仓库。通过在 GitHub 上创建账号为 homebrewltd/awesome-local-ai 的开发做出贡献。</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">lmstudio-ai/configs main 分支下的 Extension-Pack-Instructions.md</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://github.com/lmstudio-ai/configs/issues/26">存在多个 GPU 时无法选择 GPU · Issue #26 · lmstudio-ai/configs</a>: 这里是 faraday 和 Jan 的做法。然而我无法在 LM Studio 中选择 GPU，它总是使用我的 CPU 和板载内存。</li><li><a href="https://github.com/McGill-NLP/llm2vec">GitHub - McGill-NLP/llm2vec: 'LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders' 的代码</a>: 'LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders' 的代码 - McGill-NLP/llm2vec
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1267920967141687359)** (78 条消息🔥🔥): 

> - `Intel graphics support issues` (Intel 显卡支持问题)
> - `Vulkan support rollout` (Vulkan 支持推出)
> - `GPU offloading and model performance` (GPU offloading 与模型性能)
> - `Challenges with upgrading hardware` (硬件升级的挑战)
> - `RAM usage discrepancies between GPUs` (不同 GPU 之间的 RAM 使用差异)


- **Intel 显卡支持引发挫败感**：用户反映 Intel 显卡在最近的版本中运行效果不佳，并指出之前的版本（如包含 OpenCL 插件的 **0.2.25**）曾提供过支持。
   - 一位用户提到他们的笔记本电脑存在兼容性问题，引发了关于通过降级版本来获得更好性能的讨论。
- **Vulkan 支持预计很快推出**：大家对即将发布的 **Vulkan 支持** 充满期待，这有望解决许多现有的 Bug 报告。
   - 一位成员表示，希望这能改善兼容性，特别是针对 AMD 驱动。
- **GPU offload 对大模型有益**：一位用户确认使用 GPU offload 有助于更高效地运行大型模型，并强调了其在处理高上下文（high context）任务中的作用。
   - 其他人指出，虽然在 CPU 上运行也有效，但利用 GPU 资源是获得更好性能的关键。
- **硬件升级挑战**：讨论强调了升级硬件的难度，特别是对于预算有限的用户，提到了 Nvidia GPU 的高昂成本。
   - 有人担心 Intel 缺乏对 AI 应用的支持，限制了使用旧机器用户的选择。
- **不同 GPU 的 RAM 使用差异**：有人针对在不同 GPU 上加载相同模型时 RAM 使用量不一致的问题提出疑问，其中 **3080ti** 的使用量远高于 **7900XTX**。
   - 有建议认为，显存容量的差异可能会影响整体性能和资源分配。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://releases.lmstudio.ai/windows/0.2.25/latest/LM-Studio-0.2.25-Setup.exe">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#opencl-0225-does-not-support-gemma-2-or-llama-31">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://github.com/lmstudio-ai/configs/blob/890ba91f489b50f7f8f368d4b10c4">GitHub - lmstudio-ai/configs at 890ba91f489b50f7f8f368d4b10c45ae62948d48</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - GitHub - lmstudio-ai/configs at 890ba91f489b50f7f8f368d4b10c45ae62948d48
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1267923979419521024)** (212 条消息🔥🔥): 

> - `为电视角色训练 Loras`
> - `模型性能与 GPU 推荐`
> - `使用 ComfyUI 和 Auto1111`
> - `图像生成问题`
> - `Automatic1111 中的创意放大 (Creative upscaling)` 


- **为多个角色训练 Loras**：为了在同一张图像中生成两个电视角色，可以使用 Auto1111 中的 Regional Prompter 扩展在不同区域加载不同的 Loras。
   - 此外，使用带有特定 Prompt 的 SD3 也可以奏效，但对于知名度较低的角色可能效果不佳，建议使用带标签的图像创建自定义 Loras。
- **在 RTX 4070S 和 4060 Ti 之间选择 AI 显卡**：从 RTX 3060 升级时，共识是 RTX 4070S 的性能通常优于 RTX 4060 Ti，尽管后者拥有更大的 VRAM。
   - 用户建议，对于 AI 任务，4070S 因性能更佳而更受青睐，但 4060 Ti 更大的显存也有其优势。
- **针对不同模型使用 ComfyUI 和 Auto1111**：虽然 ComfyUI 对 SD3 提供了更好的支持和效率，但 Auto1111 也有一些能力，特别是在 Clip 层方面，但缺乏全面的功能。
   - 强调用户应在 ComfyUI 中正确设置模型，以避免兼容性问题并最大化性能。
- **图像生成问题**：用户对生成包含多个角色的图像感到沮丧，在使用不熟悉的模型时经常导致错误的输出。
   - 建议在叠加自定义模型或 Loras 之前，先仅使用 Prompt 进行初步测试，以评估兼容性和输出质量。
- **Automatic1111 中的创意放大 (Creative Upscaling)**：有人询问 Automatic1111 中创意放大器的可用性和用法，为新用户寻求澄清。
   - 讨论表明，虽然某些功能可以在 NightCafe 等其他 AI 工具中找到，但在 Auto1111 中有效访问可能需要额外的步骤或设置。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.mage.space/build">Mage | Build</a>: 使用 Mage 进行创作和展示！</li><li><a href="https://ai.meta.com/SAM2/">未找到标题</a>: 未找到描述</li><li><a href="https://civitai.com/models/153332?modelVersionId=171702">Western Dragon Collection XL - Classic style - v1.0 | Stable Diffusion LoRA | Civitai</a>: 西方龙合集 关键词: dragon 建议权重: 0.7-0.8 Clip skip: 1 世界上有很多种风格的龙，其中几种模式...</li><li><a href="https://civitai.com/models/107842/aniverse">AniVerse - V5.0 - Pruned | Stable Diffusion Checkpoint | Civitai</a>: ⬇阅读下方信息以获取高质量图像（点击显示更多）⬇ &amp;gt;&amp;gt;&amp;gt; 严禁在 CIVITAI 之外上传/分享我的模型...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1267919881832235028)** (110 messages🔥🔥): 

> - `OpenAI's advanced voice mode`
> - `DALL-E vs Imagen comparisons`
> - `STT and TTS latency`
> - `Emotional intelligence in AI`
> - `AI tools for school work` 


- **对 OpenAI advanced voice mode 的期待**：许多用户表达了对 OpenAI 发布 advanced voice mode 的渴望，认为这能显著增强交互体验。
   - 用户还对未来更新中语音的质量和可用性表示了担忧，并对其潜在的多样性提出了疑问。
- **DALL-E 3 与 Imagen 3 的性能对比**：用户对比显示，**Imagen 3** 被认为比 **DALL-E 3** 更好、更真实，尽管它拥有严格的审核系统。
   - 一些用户寻求 **GPT-4o** 的图像生成能力与 Imagen 3 输出之间的具体对比，表现出对详细见解的渴望。
- **STT 和 TTS 系统的延迟**：讨论围绕哪些 STT 和 TTS 系统提供最低延迟展开，用户正在探索实时转录的选项。
   - 共享了一些建议和资源，包括参考 GitHub WhisperX 仓库获取安装指导。
- **探索 AI 中的情感智能**：一位用户提出了训练多模态 AI 模型从视觉线索中推断情感的想法，建议从传统的打标签方法转向这种方式。
   - 这场对话强调了 AI 通过间接学习在人类情感中发展上下文理解的潜力。
- **最佳学术辅助 AI 工具**：用户争论哪种 AI 模型（如 **GPT-4o**、**Claude** 或 **Llama**）对学术任务最有效。
   - 对话强调了人们在持续寻找能够有效支持教育的最强大的 AI 工具。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/gdb/status/1790869434174746805">来自 Greg Brockman (@gdb) 的推文</a>：一张 GPT-4o 生成的图像 —— 仅 GPT-4o 的图像生成能力就有太多值得探索的地方。团队正在努力将其带给世界。</li><li><a href="https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding">利用 GPT 的视觉能力和 TTS API 处理并叙述视频 | OpenAI Cookbook</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1267926192162345072)** (14 messages🔥): 

> - `Custom GPT concerns`
> - `ChatGPT memory updates`
> - `Alpha tester selection` 


- **对 Custom GPTs 的担忧**：一名成员提出了关于 Custom GPTs 是否可能包含**恶意内容**或影响隐私的问题。
   - 这引发了关于 AI 模型中用户生成内容潜在风险的简短讨论。
- **更新 ChatGPT 中的 Memory**：一名成员询问如何手动更新 ChatGPT 中的 memory，寻求对该过程的澄清。
   - 另一名成员建议使用“remember this...”这一短语来促进 memory 更新，尽管在删除与添加方面存在困惑。
- **成为 Alpha Tester**：一名成员寻求关于如何成为该平台 **alpha tester** 的信息。
   - 回复比较轻松，表示运气在选拔过程中起着重要作用。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1267936610536919241)** (4 messages): 

> - `GPT-4o function performance`
> - `Language preferences in the community`
> - `Prompt engineering platforms` 


- **GPT-4o functions 显示出性能问题**：一名成员报告称 **GPT-4o functions** 的结果正在恶化，指出其表现不如直接提交给 GPT 的 prompt。
   - 他们询问其他人在处理 function 相关查询时是否也面临类似问题。
- **对西班牙语使用者的兴趣**：一名用户询问社区中是否有人说西班牙语，表示希望与西班牙语使用者建立联系。
   - 另一名用户幽默地回答说他们说**巴西葡萄牙语**，为聊天增添了语言多样性。
- **寻找最佳 Prompt Engineering 平台**：一名成员表示有兴趣了解最佳的 Prompt Engineering 平台，正在寻求推荐。
   - 这一查询反映了社区内对优化 prompt 策略日益增长的兴趣。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1267936610536919241)** (4 messages): 

> - `GPT-4o functions performance`
> - `Language diversity in the community`
> - `Best platforms for prompt engineering` 


- **GPT-4o 在 Function Call 方面表现不佳**：一位成员表示担心，与直接输入 Prompt 相比，在 **GPT-4o** 中使用 **functions** 会导致回答质量下降。
   - 他们询问其他人在使用该功能时是否也遇到了类似的质量下降。
- **社区语言能力受到关注**：一位用户询问聊天室中是否有 **Spanish speakers**（西班牙语使用者），引发了关于语言能力的对话。
   - 另一位成员幽默地插话称他们说来自巴西的 **Portuguese**（葡萄牙语）。
- **寻求 Prompt Engineering 工具**：一位成员寻求关于进行 **Prompt Engineering** 的**最佳平台**建议。
   - 这一咨询表明了对 AI 交互中有效工具的持续兴趣和需求，特别是在 Prompt 优化方面。


  

---



### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1267947769986027561)** (1 messages): 

> - `Sparse Autoencoders`
> - `Evaluating Text Explanations`
> - `Open Source Library for Auto-Interpreted Features`
> - `Cost Efficiency in Feature Interpretation` 


- **Sparse Autoencoders 解决扩展性问题**：新的自动化流水线帮助 **Sparse Autoencoders** 恢复可解释的特征，解决了人工标注者的扩展性挑战，并允许对 **GPT-2** 和 **Llama-3 8b** 特征进行更轻松的评估。
   - 关键发现表明，开源模型提供的合理评估与人类解释相似。
- **文本解释评估的创新方法**：提出了几种衡量**解释召回率（recall of explanations）**的方法，包括构建反例和测试模型生成的激活值。
   - 使用了**更小的模型**，与之前的方法相比，只需更少的 Token 即可获得可靠的分数。
- **发布用于特征研究的开源库**：发布了一个新的开源库，能够对源自 Sparse Autoencoders 的**自动解释特征（auto-interpreted features）**进行研究。
   - 代码已在 [GitHub](https://github.com/EleutherAI/sae-auto-interp) 上提供，供有兴趣为开发做出贡献的人员使用。
- **模型特征解释的成本效益**：使用当前的流水线解释 **GPT-2** 的 **150 万个特征**，预计在 **Llama 3.1** 上仅需 **$1300**，相比之前方法所需的 **$200k** 大幅降低。
   - 这种效率标志着模型特征分析方法的突破，强调了可负担性和可扩展性。
- **Demo 和 Dashboard 增强**：创建了一个小型 Dashboard 和 Demo，以展示新库中**自动解释特征**的功能。
   - Demo 可以通过 [这里](https://cadentj.github.io/demo/) 访问，建议在大屏幕上查看。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://blog.eleuther.ai/autointerp/">Open Source Automated Interpretability for Sparse Autoencoder Features</a>：构建和评估用于自动可解释性的开源流水线</li><li><a href="https://github.com/EleutherAI/sae-auto-interp">GitHub - EleutherAI/sae-auto-interp</a>：通过在 GitHub 上创建账号，为 EleutherAI/sae-auto-interp 的开发做出贡献。</li><li><a href="https://cadentj.github.io/demo/">Feature Dashboard</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1267941359101280389)** (73 条消息🔥🔥): 

> - `Open Source AI Policies` (Open Source AI 政策)
> - `GoldFinch Architecture` (GoldFinch 架构)
> - `Deepfake Concerns` (Deepfake 担忧)
> - `Genomic Data Processing` (基因组数据处理)
> - `LLM Performance Comparisons` (LLM 性能对比)


- **白宫拥抱 Open Source AI**：白宫发布了一份[报告](https://www.ntia.gov/press-release/2024/ntia-supports-open-models-promote-ai-innovation)，推广 Open Source AI 并呼吁加强风险监控，表明目前不会限制开源模型权重。
   - 官员们承认了开放系统的重要性，美国商务部长强调了在创新与风险之间采取平衡方法。
- **GoldFinch 架构展现潜力**：成员们讨论了 **GoldFinch** 架构，强调其由于采用了全注意力层（full attention layers），能够实现类似于 Transformer 的 **100% 召回率**。
   - 该项目旨在使大规模基因组数据输入的 pre-fill 成本极低，在基因组学应用中具有巨大潜力。
- **Deepfake 技术的挑战**：有人担心，虽然中心化模型可以更好地管理 Deepfake，但即使是低质量的模型也可能造成重大伤害。
   - 讨论强调，在应对 Deepfake 内容的社会影响时，协调和文化信任至关重要。
- **基因组分析的创新**：一位成员分享了他们如何使用一种名为 **charters** 的创新方法进行基因组数据表示，旨在微调 LLM 以识别遗传序列。
   - 成员们对在 EleutherAI 社区内开展合作表现出浓厚兴趣，旨在与 Oxford Nanopore 等机构合作提升基因组分析潜力。
- **LLM 对比：Gemma 2 vs. Llama 3**：参与者询问了 **Gemma 2 (9B)** 和 **Llama 3.1 (8B)** 模型之间的性能差异，寻求基于用户体验的见解。
   - 有人指出目前仍缺乏具体结论，因为这些模型的评估仍在进行中，鼓励用户分享他们的使用体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.ntia.gov/press-release/2024/ntia-supports-open-models-promote-ai-innovation">NTIA 支持开放模型以促进 AI 创新 | 国家电信和信息管理局</a>：未找到描述</li><li><a href="https://apnews.com/article/ai-open-source-white-house-f62009172c46c5003ddd9481aa49f7c3">白宫表示目前无需限制“开源”人工智能</a>：白宫在周二的一份报告中表示支持“开源”人工智能技术，认为目前没有必要对制造关键组件的公司实施限制...</li><li><a href="https://fixupx.com/impershblknight/status/1818769082944307517?t=41UyAwMxUTUMwBIspUiHRQ&s=19">来自 Imperishable Knight ⛩️ (RJ) (@impershblknight) 的推文</a>：给希望获得 #ChatGPT Advanced Voice Alpha 测试权限的 Plus 用户的建议：你尝试过开启这些设置吗？我最初没收到 AV 邀请，但我开启它们几小时后...</li><li><a href="https://www.zdnet.com/article/a-new-white-house-report-embraces-open-source-ai/">白宫新报告拥抱 Open Source AI</a>：国家电信和信息管理局支持开放数据模型，但也承认存在风险。以下是其计划如何应对该技术的利弊。</li><li><a href="https://x.com/jaehunjung_com/status/1817994332458336724?s=46">来自 Jaehun Jung (@jaehunjung_com) 的推文</a>：LLM-as-a-judge 已成为常态，但我们如何确定它真的能与人类标注员达成一致？🤔在我们的新论文中，我们引入了一种原则性方法，为 LLM 裁判提供可证明的保证...</li><li><a href="https://drive.google.com/drive/folders/1GuBRRmVSRIHivJOocSCJXqDad91cHhBv?usp=drive_link">Vienna - Google Drive</a>：未找到描述</li><li><a href="https://www.facebook.com/share/36d7L6zUnpfr7Xc9/">Facebook</a>：未找到描述</li><li><a href="https://www.facebook.com/share/kqhr6ziGxVwRp2vT/">Facebook</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1267953067714937014)** (36 条消息🔥): 

> - `SAE 发布反馈`
> - `Diffusion models 讨论`
> - `在 Tensor Cores 上生成随机数`
> - `流形 (Manifold) 与图相似性指标`
> - `系统提示词 (System Prompt) 风格模型的训练` 


- **对 SAE 发布和代码实用性的兴奋**：成员们对 **SAE publication** 及其简洁的代码表示赞赏，表示有兴趣利用它探索特定领域的微调模型。
   - 讨论强调了使用 Hugging Face 模型的特征以及分享的演示脚本进行特征提取的便利性。
- **Diffusion Augmented Agents 的创新**：引入了一个名为 **Diffusion Augmented Agents (DAAG)** 的新概念，专注于提高强化学习中的样本效率。
   - 该方法利用语言、视觉和 **Diffusion models** 来增强学习，证明了在模拟环境中样本效率的提升。
- **探索在 Tensor Cores 上更快的 PRNGs**：关于纯粹使用 **Tensor Core** 操作实现 **PRNGs** 以超越标准内存写入速度的可行性提出了疑问。
   - 提出了 **SquirrelNoise5** 等建议，并讨论了如何利用矩阵乘法优化随机数生成。
- **相似性度量中的流形假设**：讨论了使用 **manifolds** 代替相似性图来更准确地捕捉样本间关系的潜力。
   - 该概念围绕使用 **manifolds** 内的固有距离来定义差异性，强调了样本间距的重要性。
- **关于提示词风格模型训练数据的查询**：提出了关于 **system prompt style models** 训练方法的问题，强调了现实应用中缺乏类似数据的情况。
   - 对话指向了用于这些模型的训练数据的合成性质，并征求相关研究的见解和引用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.20292">From pixels to planning: scale-free active inference</a>：这篇论文描述了一个离散状态空间模型及其配套方法，用于生成式建模。该模型将部分观测的马尔可夫决策过程推广到包含作为隐变量的路径...</li><li><a href="https://arxiv.org/abs/2407.20798">Diffusion Augmented Agents: A Framework for Efficient Exploration and Transfer Learning</a>：我们介绍了 Diffusion Augmented Agents (DAAG)，这是一个利用 LLM、VLM 和 Diffusion models 来提高样本效率和迁移学习的新型框架...</li><li><a href="https://x.com/fly51fly/status/1818164241708552493">fly51fly (@fly51fly) 的推文</a>：[LG] 迈向非线性 RNN 的可扩展且稳定的并行化 X Gonzalez, A Warrington, J T.H. Smith, S W. Linderman [斯坦福大学] (2024) https://arxiv.org/abs/2407.19115 - 非线性 RNN...</li><li><a href="https://arxiv.org/abs/2407.19115">Towards Scalable and Stable Parallelization of Nonlinear RNNs</a>：传统的非线性 RNN 在序列长度上不具备天然的可并行性，而 Transformer 和线性 RNN 则具备。因此，Lim 等人 [2024] 解决了非线性 RNN 的并行化评估问题...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1267922569932312667)** (1 条消息): 

> - `知识蒸馏 (Knowledge Distillation)`
> - `7B 模型超参数`
> - `蒸馏所需的计算资源` 


- **寻求知识蒸馏方面的帮助**：一名成员正在寻求 **7B 模型** **knowledge distillation** 方面的协助，特别是关于设置必要超参数的问题。
   - 他们还询问了此过程所需的 **compute resources**（计算资源）。
- **关于超参数的咨询**：讨论了对 **7B 模型** 进行有效 **knowledge distillation** 可能需要的各种 **hyperparameters**。
   - 成员们分享了关于 **tuning**（调优）这些参数以获得最佳性能的重要性见解。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1268277082828443832)** (6 条消息): 

> - `Gemma Scope`
> - `ICML Workshop Recording` 


- **Gemma Scope 发布，包含令人兴奋的 SAEs**：团队发布了 [Gemma Scope](https://neuronpedia.org/gemma-scope)，这是一套针对 **Gemma 2 2B** 和 **9B** 每个层和子层的开源 **Sparse Autoencoders (SAEs)**，其开发过程消耗了相当于 **GPT-3 算力的 22%**。
   - **Neuronpedia** 创建的一个 demo 展示了它们的功能，更多资源在包含详细链接的 [推文线程](https://x.com/NeelNanda5/status/1818680642621915527) 中分享。
- **ICML Workshop 录像延迟**：成员们询问了 **ICML Mech Int Workshop** 的录像情况，确认 **ICML** 最终会上传录像。
   - 然而，由于特殊的 **ICML 规则**，在录像可供观看前会有 **一个月的等待期**，旨在鼓励购买虚拟门票。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://neuronpedia.org/gemma-scope">Gemma Scope</a>: 探索 Gemma 2 2B 的内部机制</li><li><a href="https://x.com/NeelNanda5/status/1818680642621915527">Neel Nanda (@NeelNanda5) 的推文</a>: Sparse Autoencoders 就像 AI 内部机制的显微镜。它们是可解释性的强大工具，但训练成本限制了研究。现发布 Gemma Scope：一套针对...的开源 SAEs...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1268035862919123110)** (13 条消息🔥): 

> - `lm-eval Zeroshot`
> - `GPQA processing discrepancies`
> - `lm-eval launch script`
> - `super_glue task`
> - `sts-b subtask omission` 


- **用于 Zeroshot 评估的 lm-eval 命令**：一位成员询问了使用 lm-eval 在 *zeroshot* 模式下评估模型的命令，并提到可能缺少某个 flag。
   - 另一位成员建议使用 `--num_fewshot 0` 作为解决方案。
- **GPQA Prompt 处理中的差异**：一位用户观察到，在运行 GPQA 任务时， lm-eval 似乎处理了 benchmark 中各种任务现有 **4 倍数量的 prompt**。
   - 其他成员推测这可能是因为在分别处理选项，并指出 GPQA 的四个选项在大小上存在显著差异。
- **lm-eval 进程的完整 stdout**：用户分享了执行 lm-eval 的启动脚本，并记录了处理的 prompt 确切数量，确认这与之前的观察相符。
   - 他们报告处理了 **1792 个 prompt**，这引发了对评估效率和准确性的担忧。
- **关于 Super Glue 和 STS-B 的查询**：一位用户询问是否有人有 *super_glue* 任务的经验，并评论说 GLUE 任务中缺少 **sts-b** 子任务。
   - 虽然在这个话题上没有提供直接答案，但它强调了社区内对任务规范的好奇。
- **lm-eval 潜在的 GitHub Issue**：一位用户表示愿意就他们遇到的 prompt 处理问题创建一个 GitHub issue，但不确定是否有必要。
   - 他们表示已经搜索过 GitHub issues，但没有发现任何相关的讨论。


  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1268078149652971562)** (1 条消息): 

> - `GPT-NeoX library papers`（GPT-NeoX 库论文）
> - `Azure power needs study`（Azure 电力需求研究）
> - `MIT CogSci lab research`（MIT CogSci 实验室研究）
> - `Hierarchical transformers`（分层 Transformer）
> - `Low-latency multimodal models`（低延迟多模态模型）


- **LLaMA 2 在巴斯克语 Token 上进行微调**：[Latxa](https://arxiv.org/abs/2403.20266) 利用 **42 亿个巴斯克语 Token** 对 LLaMA 2 进行微调，以增强其在特定语言应用中的性能。
   - 这一努力反映了大模型在多语言环境中的应用日益增多。
- **Microsoft Azure 关于 AI 电力需求的研究**：Microsoft Azure 使用 GPT-NeoX [研究大规模 AI 训练的电力需求](https://dl.acm.org/doi/pdf/10.1145/3620666.3651329)，以应对能源消耗挑战。
   - 该研究旨在优化 AI 运营对环境的影响。
- **MIT CogSci 实验室的模型训练见解**：MIT CogSci 实验室最近的一篇论文重点介绍了其使用 GPT-NeoX 进行的第四项研究，探讨模型如何与**人脑**进行比较，为认知科学做出贡献。
   - 这些发现可能会重塑我们对 AI 和认知过程的理解。
- **分层 Transformer 开发**：KAIST AI 和 LG 的一个项目专注于开发[分层 Transformer](https://arxiv.org/abs/2406.02657)，以克服 **KV caching 限制**，突破 Transformer 架构的极限。
   - 这一创新旨在提高处理更大上下文的效率。
- **开发低延迟多模态模型**：rinna Co. 的研究人员利用 GPT-NeoX [开发低延迟多模态文本/语音模型](https://arxiv.org/abs/2406.12428)，旨在优化交互速度和质量。
   - 这一进展对于需要实时音频和文本集成的应用至关重要。


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1267920560801714209)** (70 条消息 🔥🔥): 

> - `Paid User Concerns`（付费用户担忧）
> - `WordPress Partnership`（WordPress 合作伙伴关系）
> - `Perplexity Labs Issues`（Perplexity Labs 问题）
> - `Advertising on Perplexity`（Perplexity 上的广告）
> - `Chart Creation in Perplexity`（在 Perplexity 中创建图表）


- **付费用户对广告的担忧加剧**：成员们对**付费用户**是否会遇到广告表示出越来越大的不安，许多人担心这会破坏平台的无广告使用体验。
   - 一位用户评论道：“沉默绝不是好兆头”，强调 Perplexity 需要给出明确说明。
- **WordPress 合作伙伴关系需要明确说明**：关于 **WordPress 合作伙伴关系**的影响出现了疑问，特别是这是否涉及使用该平台的个人博主。
   - 社区成员正在寻求有关其内容是否受此合作伙伴关系影响的详细信息。
- **部分用户无法使用 Perplexity Labs**：几位用户报告了访问 **Perplexity Labs** 的问题，一些人遇到了 *ERR_NAME_NOT_RESOLVED* 等错误，而另一些人则询问是否受到了地理限制。
   - 尽管存在这些问题，正常的 Perplexity 功能似乎仍在运行，引发了关于特定地区是否受到影响的猜测。
- **关于广告影响的辩论**：有人担心在回答中整合**赞助问题**可能会操纵用户的思维过程，从而引发伦理考量。
   - 成员们对任何可能影响输出质量的广告持谨慎态度，一些人强调广告不应影响回答的上下文。
- **在 Perplexity 中创建图表**：用户正在询问如何在 Perplexity 中成功创建**图表**，并建议查看特定频道以获取指导。
   - 一些用户认为访问这些功能可能需要 **Pro 版本**，而另一些用户则指出这可能还取决于他们所在的地区。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.04620">Learning to (Learn at Test Time): RNNs with Expressive Hidden States</a>：Self-attention 在长上下文中表现良好，但具有平方复杂度。现有的 RNN 层具有线性复杂度，但它们在长上下文中的性能受到其隐藏状态表达能力的限制...</li><li><a href="https://tenor.com/view/second-futurama-scruffy-gif-20187509">Second Futurama GIF - Second Futurama Scruffy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.perplexity.ai/page/Complexity-Perplexitys-New-yl0q3mHYQz6RhRyuvjvN4w#1c2552be-080e-440f-8bfd-e1d31fbd2aa8">Complexity: Perplexity's New Extension</a>：Perplexity AI 的 Complexity 扩展引入了一系列强大的功能，旨在增强用户体验并简化与...的交互。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1268108167947227156)** (2 条消息): 

> - `Simulation Hypothesis`
> - `Perplexity AI Skills` 


- **通过 Simulation Hypothesis 探索现实**：Simulation Hypothesis 提出了关于存在本质的深刻问题，暗示我们感知的现实可能是由更高智能创造的复杂计算机模拟。
   - 这一探究挑战了我们对意识的理解，并引发了关于在潜在模拟之外辨别真实现实的重要讨论。
- **Perplexity AI 的强大技能揭秘**：Perplexity AI 利用 LLM 综合来自各种来源的信息，擅长提供准确且全面的回答。
   - 关键应用包括**市场研究**和**竞争分析**，通过收集大量数据来提供有关市场趋势和竞争对手行为的见解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.perplexity.ai/search/what-is-best-skills-in-perplex-mvRHkNtwTHGP7MIk0q3akA">What is best skills in PerplexitAI ?</a>：Perplexity AI 是一款强大的工具，结合了搜索和文本生成能力，利用 LLM 来...</li><li><a href="https://www.perplexity.ai/search/if-we-are-living-in-a-simulate-WR5w3Ix_Q56GV..cYLO54w">If we are living in a simulated world, and there exists a true reality beyond...</a>：在潜在的模拟世界之外辨别真实现实的问题是一个深刻的哲学和科学探究。Simulation Hypothesis...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1267922472871923834)** (49 条消息🔥): 

> - `API model discrepancies`
> - `Citation request delays`
> - `Model deprecation`
> - `Search index issues`
> - `Response quality concerns` 


- **API 模型差异问题被提出**：用户报告称，对于相同的 Prompt，API 返回的结果与 Perplexity 界面不同，特别是在查询公司信息时。
   - 一位用户表示，虽然 API 返回了错误公司的信息，但 Web 界面正确识别了目标公司，这表明 API 模型可能存在缺陷。
- **引用请求延迟仍然存在**：一位用户对之前申请 API 引用权限的请求表示担忧，强调随着项目发布临近，需要及时的支持。
   - 尽管提交了表单和电子邮件，但仍未收到回复，这引发了对客户服务响应能力的质疑。
- **模型即将弃用**：几位用户讨论了即将弃用的 **llama-3-sonar-large-32k-online** 模型，一些用户对新模型的性能下降表示失望。
   - 随着这些模型被逐步淘汰，用户担心搜索结果的一致性和可靠性，以及模型处理复杂 Prompt 的能力。
- **Search Index 查询问题**：关于为 LLM 构建有效 Search Index 的复杂性展开了讨论，承认了在确保数据时效性和相关性方面的挑战。
   - 用户考虑了将传统搜索引擎与自己的索引结合使用的策略，以提高结果质量，但也认识到其中涉及的巨大工作量。
- **对回答质量的担忧**：用户报告称，**llama-3.1-sonar** 版本等新模型的回答质量不佳，并对反复出现的导致无法浏览的道歉 Prompt 表示沮丧。
   - 这引发了人们对升级可能无法达到前几版本预期的担忧，从而引发了对故障排除支持的呼吁。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai/discuss/66a8f6b588da9f0024012ab8">Request for citations in API</a>：未找到描述</li><li><a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>：未找到描述</li><li><a href="https://perplexity.typeform.com/to/j50rnNiB?typeform-source=docs.perplexity.ai)">pplx-api form</a>：使用 Typeform 将数据收集转化为一种体验。创建精路线的在线表单、调查、测验等等。免费试用。
</li>
</ul>

</div>

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1267920803291332818)** (14 条消息🔥): 

> - `Mojo community feedback` (Mojo 社区反馈)
> - `Mojo presentation guidelines` (Mojo 演讲指南)
> - `Mojo as a C replacement` (Mojo 作为 C 的替代方案)
> - `Type comparison in Mojo` (Mojo 中的类型比较)


- **Mojo 社区寻求建设性反馈**：成员们讨论了反馈对于改进 **Mojo 社区会议**并使其更具吸引力的重要性。
   - *Tatiana* 鼓励演讲者分享他们的 Discord 标签，并将演讲重点放在与 **Mojo** 直接相关的主题上。
- **确立官方演讲范围**：一名成员建议增加关于 **Mojo 社区会议** 主题范围的官方指南。
   - *Nick* 强调演讲需要专门关注 **Mojo 语言**、库和社区问题。
- **使用 Mojo 作为解释器的 C 替代方案**：*Memesmith2* 询问了使用 **Mojo** 作为 C 的替代方案来开发针对 ARM、RISC-V 和 x86_64 的解释器的可行性。
   - *Darkmatter* 指出 Mojo 并不直接支持 computed goto，且 Mojo 在某种程度上类似于采用 Python 语法的 **Rust**。
- **理解 Mojo 中的类型比较**：*Moosems_yeehaw* 指出了涉及列表的类型比较中的一个有趣行为，具体为 `print(list[str] | list[int] == list[str | int])` 返回了 **False**。
   - *Ivellapillil* 同意从类型角度来看，单一类型元素的列表与混合类型的列表是不同的。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1267922588890304573)** (100 messages🔥🔥): 

> - `Mojo String Implementation` (Mojo 字符串实现)
> - `Function Reflection in Mojo` (Mojo 中的函数反射)
> - `Mojo Database Drivers` (Mojo 数据库驱动)
> - `Mojo and MLIR Integration` (Mojo 与 MLIR 集成)
> - `Mojo Max License Concerns` (Mojo Max 许可证担忧)


- **Mojo 字符串实现优化了 UTF-8 和索引**：一位用户成功实现了一个具有小字符串优化 (small string optimization) 和完整 UTF-8 支持的 Mojo 字符串，从而实现了高效的长度计算和索引。
   - 该实现允许三种索引类型：字节 (byte)、Unicode 码位 (code point) 和用户感知的字形 (glyph)，解决了不同语言的复杂性。
- **Mojo 缺乏函数反射**：目前，Mojo 不支持函数签名的运行时反射 (runtime reflection)，这让寻求类似 Python `inspect` 模块功能的开发者感到困扰。
   - 社区建议未来可能会引入静态反射 (static reflection)，类似于 Zig 等语言的方法。
- **探索 Mojo 数据库驱动**：社区对 SQLite、Postgres 和 MySQL 的 Mojo 数据库驱动持续关注，并提到一位社区成员已经开始开发 DuckDB 驱动。
   - 目前 C interop 的状态还比较粗糙，因此针对主流数据库项目的绑定仍处于未开发状态。
- **Mojo 集成 MLIR 的潜力**：讨论表明 Mojo 未来可能支持 MLIR 和 SPIR-V，从而为 GPU 着色器 (shaders) 实现高级程序转换和处理。
   - 目前的实现和规范仍在演进中，重点在于如何优化 GPU 编程体验。
- **关于 Mojo Max 许可证的担忧**：社区对新的 Mojo Max 许可证褒贬不一，担忧其可撤销性以及对现有项目未来保障的缺乏。
   - 用户表示需要更清晰的许可条款，以确保在基于 Mega 架构构建时拥有合理的追索权。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/classesandstructures/">Documentation</a>：未找到描述</li><li><a href="https://learn.microsoft.com/en-us/dotnet/standard/design-guidelines/choosing-between-class-and-struct">Choosing Between Class and Struct - Framework Design Guidelines</a>：学习如何决定将类型设计为类 (class) 还是结构体 (struct)。了解 .NET 中引用类型和值类型的区别。</li><li><a href="https://gist.github.com/mzaks/78f7d38f63fb234dadb1dae11f2ee3ae#fil">Mojo String with small string optimisation and potential full UTF-8 support</a>：具有小字符串优化和潜在完整 UTF-8 支持的 Mojo String - crazy_string.mojo</li><li><a href="https://gist.github.com/mzaks/78f7d38f63fb234dadb1dae11f2ee3ae#file-crazy_string-mojo-L52)">Mojo String with small string optimisation and potential full UTF-8 support</a>：具有小字符串优化和潜在完整 UTF-8 支持的 Mojo String - crazy_string.mojo</li><li><a href="https://github.com/fnands/mimage">GitHub - fnands/mimage: A library for parsing images in Mojo</a>：一个用于在 Mojo 中解析图像的库。欢迎通过在 GitHub 上创建账号来为 fnands/mimage 的开发做出贡献。</li><li><a href="https://gist.github.com/mzaks/78f7d38f63fb234dadb1dae11f2ee3ae">Mojo String with small string optimisation and potential full UTF-8 support</a>：具有小字符串优化和潜在完整 UTF-8 支持的 Mojo String - crazy_string.mojo</li><li><a href="https://github.com/fnands/mimage/issues/3">Pre-compute CRC32 table · Issue #3 · fnands/mimage</a>：对于 CRC32 计算，我们目前的实现是“实时”计算所有值。然而，这些值表可以预先计算，甚至在编译时完成。这将使得...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1267963080202125382)** (70 messages🔥🔥): 

> - `LLM 追踪挑战`
> - `Aider 的 LLM 排行榜`
> - `4o Mini 性能讨论`
> - `NSFW 模型推荐`
> - `OpenRouter 成本对比` 


- **LLM 追踪挑战堆积如山**：成员们对 **LLMs** 的激增表示沮丧，称很难追踪它们的能力和性能。
   - 有人指出，在日益拥挤的领域中，有必要创建个人基准来评估新出现的模型。
- **Aider 的 LLM 排行榜出现**：Aider 有一个有趣的 **LLM 排行榜**，根据模型编辑代码的能力进行排名，专门为编程任务设计。
   - 用户指出，它最适合那些擅长 *编辑* 而不仅仅是生成代码的模型。
- **对 4o Mini 性能的担忧**：关于 **4o Mini** 展开了激烈的辩论，对其性能与 **3.5** 等其他模型的对比以及作为编程任务潜在替代品的看法各不相同。
   - 一些成员认为，虽然它有其优势，但由于输出质量更好，一些人仍然更倾向于 **1.5 flash** 等选项。
- **关于 NSFW 模型选项的讨论**：成员们分享了使用各种 **NSFW 模型** 的经验，特别强调了 **Euryal 70b** 和 **Magnum** 是值得关注的选择。
   - 他们还建议查看 **Dolphin 模型**，并引导用户前往 **SillyTavern Discord** 等资源获取更多信息。
- **OpenRouter 成本削减见解**：一位成员指出，从 **ChatGPT** 切换到 **OpenRouter** 后，他们的成本大幅降低，支出从 **$40/月** 降至仅 **$3.70**。
   - 这种成本节省归功于使用 **Deepseek 进行编程**，这占了他们使用量的大部分。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sambanova.ai/">SambaNova Systems | 彻底改变 AI 工作负载</a>: 使用 SambaNova 的企业级生成式 AI 平台为您的业务解锁 AI 的力量。了解如何实现 10 倍的成本降低和无与伦比的安全性。</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>: LLM 代码编辑能力的定量基准。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1268242235070353480)** (26 messages🔥): 

> - `Gemma 2 2B 性能`
> - `模型发布与竞争`
> - `AI 模型中的蒸馏 (Distillation)`
> - `关于 Turbo-sloppofication 的评论`
> - `模型命名的内部细节` 


- **Gemma 2 2B 超越 GPT-3.5**：新的 [Gemma 2 2B](https://developers.googleblog.com/en/smaller-safer-more-transparent-advancing-responsible-ai-with-gemma/) 模型在 Chatbot Arena 上的表现超过了所有 GPT-3.5 模型，展示了其卓越的对话能力。
   - 成员们表达了兴奋之情，其中一人在讨论其性能时表示：*“活在这样的时代真好 (what a time to be alive)”*。
- **模型发布引发竞争性言论**：一条推文强调了对 Google AI 模型发布的担忧，鼓励进行主动沟通以避免负面印象：*“下次早点把你们的发布计划告诉我，我就不会让你们难堪了。”*
   - 这与关于模型发布对 Chatbot Arena 等平台影响的更广泛讨论有关。
- **关于蒸馏 (Distillation) 强度的争论**：围绕 **Distillation** 的讨论指出，虽然它很有效，但并非万能；正如一位成员评论道：*“蒸馏很强，但也没那么强 (distillation is stronk but not that stronk)。”*
   - 成员们对当前模型的优缺点发表了不同看法，引发了持续的辩论。
- **对 “Turbo-sloppofication” 的担忧**：人们对 AI 模型中被称为 “Turbo-sloppofication” 的现象表示担忧，这预示着模型质量可能出现下滑，一位用户简单地回应道：*“噢不，加速劣质化 (turbo-sloppofication) 还在继续。”*
   - 这反映了社区内对于模型性能快速变化的普遍焦虑。
- **模型命名惯例的内部细节**：关于将新模型命名为 “Turbo” 的内部讨论有一个有趣的发现，最终由于涉及的工作量太大而决定放弃。
   - 社区对此付之一笑，一位参与者感谢该用户的决定，说：*“感谢你的贡献 (Thank you for your service)。”*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/robdadashi/status/1818683981170119021?s=46">Robert Dadashi (@robdadashi) 的推文</a>: @TheXeophon 不，是 guava-chatbot ;)</li><li><a href="https://developers.googleblog.com/en/smaller-safer-more-transparent-advancing-responsible-ai-with-gemma/">更小、更安全、更透明：通过 Gemma 推进负责任的 AI</a>：未找到描述</li><li><a href="https://x.com/natolambert/status/1818684093615554566">Nathan Lambert (@natolambert) 的推文</a>: 那些在 @GoogleAI 推销模型的人，你们可能想读读这个。下次早点把你们的发布计划告诉我，我就不会让你们难堪了 🤫 引用 Interconnects (@interconnects...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1268274027101683712)** (2 messages): 

> - `Llama 3.1 模型性能`
> - `推理提供商 (Inference provider) 的差异`
> - `基准测试 (Benchmarking) 挑战`
> - `关于 Llama 3.1 的 Twitter 讨论` 


- **Llama 3.1 主导基准测试**：Llama 3.1 已成为第一个能与顶级模型匹敌的开源模型，在各项基准测试中展示了极高的 **推理质量 (inference quality)**，尤其是在 **GSM8K 上排名第一**。
   - *辨别实现方式的差异*至关重要，因为微小的变化可能会显著影响应用的成功。
- **推理提供商之间的摩擦加剧**：推理提供商之间存在相当大的紧张关系，特别是关于 Llama 3.1 的托管方式及其对比性能。
   - 最近的 Twitter 讨论强调了 *模型基准测试中的挑战* 以及提供商之间的差异在多大程度上产生影响。
- **理解基准测试的复杂性**：由于 **实现决策 (implementation decisions)** 和 **优化过程** 的差异，对 Llama 3.1 这样的模型进行基准测试提出了独特的挑战。
   - 这些选择可能导致一个百分点或更多的性能差异，这对于有效的应用性能至关重要。



**提到的链接**：<a href="https://www.together.ai/blog/llama-31-quality">Llama 3.1：同样的模型，不同的结果。一个百分点的影响。</a>：未找到描述

  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1268287925746012290)** (4 条消息): 

> - `Anime PFP Feed`
> - `Llama 3.1 Scores`
> - `Article Timing` 


- **Anime PFP Feed 推广文章**：一位成员注意到他们的 **anime PFP feed** 开始推送一篇被他们描述为 **banger**（爆款）且时机完美的文章。
   - 这突显了热门内容与动漫社区兴趣的交集。
- **文章发布的时机至关重要**：一位成员幽默地承认，他们在文章发布的 **timing**（时机）上运气很好。
   - 他们提到，他们专门在等待 **Llama 3.1 scores** 公布，以优化影响力。


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1268272048619388988)** (3 条消息): 

> - `Open Name Discussion` 


- **关于名称中“Open”的辩论**：由一位成员发起的讨论指出，被提及的实体中只有一个在其名称中包含 **'open'**。
   - 这引发了简短的评论，强调了命名惯例在领域讨论中的重要性。
- **鼓励回归**：一位成员表达了希望另一位用户回到 Discord 的愿望，并通过一条简单的消息表达了他们的赞赏。
   - 这反映了频道内用户之间的社区感和联系。



**提到的链接**：<a href="https://x.com/deliprao/status/1818711773702132218?s=46">来自 Delip Rao e/σ (@deliprao) 的推文</a>：是的，但只有一个的名字里有“open”。

  

---

### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1268115394560917654)** (19 条消息🔥): 

> - `Subbarao Kambhampati 的工作`
> - `LLM 的内在自我修正 (Intrinsic self-correction)`
> - `推理轨迹的基准测试 (Benchmarking reasoning trajectories)`
> - `LLM 在推理和规划方面的局限性`
> - `对 LLM 自我修正的批判` 


- **Subbarao Kambhampati 批判 LLM 推理**：在最近的一期 [YouTube 节目](https://www.youtube.com/watch?v=y1WnHpedi2A) 中，Kambhampati 教授认为，虽然 **LLM** 在许多任务中表现出色，但它们在逻辑推理和规划方面存在显著局限。
   - 他在 [ICML 教程](https://youtu.be/2DbmSTK2owI?si=mIJ9lFLyxM1RGCjB) 中进一步深入探讨了 **LLM** 在规划中的作用，并得到了关于自我修正问题的多篇论文支持（[Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)）。
- **内在自我修正基准测试的挑战**：由于初始错误，**LLM** 必须修正其自身的推理轨迹，人们对这种基准测试的可行性表示担忧，认为这可能是一个不切实际的设置。
   - 讨论强调了当基准测试涉及模型故意生成错误轨迹以进行自我修正时，比较模型性能的难度。
- **LLM 难以进行有效的自我修正**：研究表明，如果没有外部反馈，**LLM** 通常无法有效地进行自我修正，这引发了对内在自我修正可行性的质疑。
   - 关于自我验证局限性研究的总结显示，与使用外部验证器相比，模型在自我验证任务中会出现**准确率下降 (accuracy degradation)**。
- **LLM 推理中的反馈循环**：一位成员注意到 **LLM** 推理的一个奇特方面，即初始错误可能会因为生成输出中上下文的变化而在稍后得到修正，这暗示了潜在的随机影响。
   - 这意味着虽然 **LLM** 可能会表现出一定的修正能力，但其推理和规划过程可能在根本上仍然存在缺陷。
- **对 LLM 推理的验证和见解**：尽管对 **LLM** 自我修正的有效性存在一些怀疑，但人们对正在进行的此类能力基准测试研究表示赞赏。
   - 参与者看到了通过这些基准测试跟踪 **LLM** 进展的价值，认为它们有助于理解推理能力的提升。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1772991897880924219">Xeophon (@TheXeophon) 的推文</a>：围绕 @rao2z 的 ASU 团队在揭秘 LLM “推理”能力方面接连发布重磅成果。我是 Valmeekan 早期作品的忠实粉丝，所以我期待这篇论文会是一个...</li><li><a href="https://www.youtube.com/watch?v=y1WnHpedi2A">你认为 ChatGPT 会推理吗？</a>：Subbarao Kambhampati 教授认为，虽然 LLM 是令人印象深刻且有用的工具，尤其是在创意任务方面，但它们在逻辑上存在根本局限...</li><li><a href="https://youtu.be/2DbmSTK2owI?si=mIJ9lFLyxM1RGCjB">关于 LLM 在规划中的作用 (ICML 2024 教程)</a>：幻灯片：https://bit.ly/4dbkkY2 教程页面：https://yochan-lab.github.io/tutorial/ICML-2024</li><li><a href="https://arxiv.org/abs/2310.01798">大语言模型尚无法自我修正推理</a>：大语言模型 (LLM) 已成为一项突破性技术，在各种应用中具有无与伦比的文本生成能力。然而，关于其...的担忧仍然存在。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1268182347837345805)** (7 条消息): 

> - `截图中文字的可见性`
> - `压缩问题`
> - `消息清晰度` 


- **移动端截图难以阅读**：一位成员指出，通过手机或电子邮件查看时，**截图中的文字**很难阅读。
   - 另一位成员对此表示赞同，指出即使点击图片，由于**压缩问题**，可读性仍然很差。
- **承认图片模糊问题**：一位成员承认其中一张截图很**模糊**，他们没有时间修复。
   - 他们提到，虽然有些令人沮丧，但对他们来说已经足够清晰，可以继续后续工作了。
- **对细节的关注表示赞赏**：针对可读性问题，一位成员评论说另一位成员**观察入微**，暗示该问题已被记录。
   - 第三位成员幽默地评论道，这种关注可能会**增加负责人的工作量**。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1267921406126198926)** (41 条消息🔥): 

> - `Google Colab Cohere API`
> - `Cohere Agentic Build Day`
> - `Rerank API for document relevance`
> - `OpenAI and Hugging Face contributions`
> - `Community support and feedback` 


- **用于 Cohere 工具的 Google Colab**：一位成员正在创建一个 [Google Colab](https://link.to.colab)，教用户如何利用 **Cohere API** 中的工具。
   - *从不知道它里面竟然有 Gemini！*
- **Agentic Build Day 出席情况**：讨论了 8 月 12 日在旧金山举行的 **Agentic Build Day**，成员们希望能有虚拟参与方式。
   - 遗憾的是，该活动**仅限线下 (IRL)**，但计划稍后举行虚拟竞赛。
- **Rerank API 增强**：成员对 **余弦相似度 (cosine similarity)** 及其在 Embeddings 中的有效性提出了担忧，建议利用 **Rerank API** 来获得更好的语义匹配。
   - Rerank 模型因其处理更长上下文和提供更有意义的相关性评分的能力而受到关注。
- **OpenAI 贡献追踪**：一位成员对 LinkedIn 帖子分享的 GitHub 贡献图中缺少 **Cohere** 追踪表示困惑。
   - 对此，有人澄清说，虽然包含了 Hugging Face 的提交，但也有许多来自 **CohereForAI** 的贡献。
- **社区开发者认可**：社区成员对正在开发的贡献和项目感到自豪，并提到了为未来活动准备的特定 Demo。
   - 针对社区协作的热情，一位成员回应道：*“太棒了 (LOVE IT)”*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/CohereForAI">CohereForAI (Cohere For AI)</a>：未找到描述</li><li><a href="https://docs.cohere.com/reference/rerank">Rerank - Cohere API References</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/rerank-2">Rerank</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/overview">Rerank Overview</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1268243842721644555)** (1 条消息): 

> - `Agent Build Day`
> - `Learning from Cohere Experts`
> - `Agent Demo Competition`
> - `Integrating Human Oversight`
> - `Cohere RAG Capabilities` 


- **加入我们在旧金山的 Agent Build Day**：我们计划于 **8 月 12 日** 在 **旧金山** 办公室举办 **Agent Build Day**，届时将有来自 **Cohere**、**AgentOps** 和 **CrewAI** 的资深开发者主持实战工作坊。
   - 参与者可以点击[此处](https://lu.ma/gptdzwhe)注册，并参加 Demo 竞赛，赢取价值 **2,000 美元的 Cohere API 额度**。
- **向 Cohere 专家学习**：与会者将有机会向 **Cohere 专家** 学习 **Agentic 工作流用例** 及其对企业系统的影响。
   - 该活动旨在通过专注于在 Agentic 工作流中**集成人机协同 (human oversight)** 的工作坊来提升性能。
- **导师指导下的实战经验**：参与者将在导师的指导下获得实战经验，构建 Agent 以自动化重复任务并提高效率。
   - 这种独特的学习体验旨在促进 **创始人** 和 **工程师** 之间的联系。
- **利用 Cohere 实现高级 RAG 能力**：**Cohere 模型** 旨在提供一流的高级 **RAG 能力**，支持 **10 种商业语言** 的多语言覆盖。
   - 它们还支持多步工具使用，以便在各种应用中自动化复杂的工作流。



**提到的链接**：<a href="https://lu.ma/gptdzwhe">Agent Build Day by Cohere x AgentOps · Luma</a>：了解如何利用 Cohere 的基座模型 Command、Embed 和 Rerank 来构建企业级 Agentic 系统，使用工具连接到外部……

  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1268170784711508042)** (14 messages🔥): 

> - `Rerank API 403 Error`
> - `Internship Application Status` (实习申请状态)
> - `Training Models for Dialect Generation` (训练方言生成模型)


- **Rerank API 返回 403 错误**: 一位用户报告在调用 Rerank API 时收到 **403 错误**，尽管已确认 Token 有效。
   - 另一位成员提出协助，请求提供有关调用者设置的更多细节，包括完整的错误消息或截图。
- **实习状态查询**: 一位用户询问了 3-4 周前提交给 **Cohere 多伦多办公室**的实习申请状态。
   - 回复建议发送邮件至 [talent@cohere.com](mailto:talent@cohere.com) 寻求帮助，并指出之前的查询显示实习名额可能已满，直到明年。
- **阿拉伯语方言生成训练**: 一位用户询问如何训练模型以生成用户选择的阿拉伯语方言回复，并提到了 **Aya** 数据集的方言特定指令。
   - 他们寻求对训练过程的见解，并指出 **Aya** 和 **Command** 模型都能有效处理该任务，尽管指令中没有明确的方言信息。


  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1268095801674960906)** (2 messages): 

> - `Community Toolkit Activation` (社区工具包激活)
> - `Docker Compose Configuration` (Docker Compose 配置)
> - `Development Environment Setup` (开发环境搭建)


- **社区工具包无法激活**: 一位成员表示，尽管在 Docker compose 中将 **INSTALL_COMMUNITY_DEPS** 设置为 true，社区工具包仍无法工作。
   - 他们提到更改后 community 文件夹中的工具仍然不可见。
- **使用 make 运行开发设置**: 同一位成员报告运行命令 **make dev** 尝试初始化其环境。
   - 他们没有提到与执行该命令相关的任何错误消息。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1267929519386394768)** (56 messages🔥🔥): 

> - `OpenAI's synthetic data rumors` (OpenAI 合成数据传闻)
> - `Gemma 2 2B model performance` (Gemma 2 2B 模型性能)
> - `Llama 3.1 evaluation differences` (Llama 3.1 评估差异)
> - `alphaXiv for arXiv papers` (用于 arXiv 论文的 alphaXiv)
> - `InternLM's MindSearch framework` (InternLM 的 MindSearch 框架)


- **OpenAI 50T 合成 Token 的传闻**: 一场对话讨论了合成数据项目，并提到有传闻称 OpenAI 正在使用 **50 万亿 (50T) Token** 的数据进行训练，其中大部分是合成数据。
   - 这一话题引发了人们对在 AI 训练中使用如此大规模合成数据集所产生影响的好奇。
- **Gemma 2 2B 模型性能亮眼**: Google DeepMind 的 **Gemma 2 2B** 模型声称在 Chatbot Arena 中表现优于所有 **GPT-3.5** 模型，展示了卓越的对话能力。
   - 低内存需求和同尺寸下强大的性能等特点使其在端侧应用中极具吸引力。
- **Llama 3.1 评估引发争议**: 关于 **Llama 3.1** 模型的讨论不断，因为最近一篇博客文章中展示的一些示例被指缺乏事实准确性。
   - 批评者指出在多查询注意力（multi-query attention）方面存在幻觉案例，引发了对质量测试流程的质疑。
- **用于论文讨论的 alphaXiv 发布**: 斯坦福大学的学生推出了 **alphaXiv**，这是一个开放论坛，只需替换 arXiv URL 即可对 arXiv 论文发布问题和评论。
   - 该平台旨在以更精简的方式增强学术出版物的参与度和讨论。
- **InternLM 推出 MindSearch 框架**: InternLM 的 **MindSearch** 框架被介绍为一个基于 **LLM** 的搜索引擎工具，类似于 Perplexity.ai。
   - 它旨在提供增强的 **Agent** 能力，以获得更精确的搜索结果。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/midjourney/status/1818342703618482265">来自 Midjourney (@midjourney) 的推文</a>：Midjourney V6.1 现已上线！V6.1 大幅提升了图像质量、连贯性、文本表现，并配备了全新的 Upscaling（放大）和个性化模型。它更智能、更快速、更清晰、更美观。我们...</li><li><a href="https://x.com/samjulien/status/1818652901130354724">来自 Sam Julien (@samjulien) 的推文</a>：🔥 @Get_Writer 刚刚发布了 Palmyra-Med-70b 和 Palmyra-Fin-70b！Palmyra-Med-70b 🔢 提供 8k 和 32k 版本 🚀 MMLU 性能约 86%，超越顶级模型 👨‍⚕️ 用于诊断、制定治疗方案...</li><li><a href="https://x.com/NeelNanda5/status/1818680642621915527">来自 Neel Nanda (@NeelNanda5) 的推文</a>：Sparse Autoencoders（稀疏自编码器）就像是 AI 内部结构的显微镜。它们是可解释性的强大工具，但训练成本限制了研究。现宣布推出 Gemma Scope：一套覆盖每一个...的开放 SAE 组</li><li><a href="https://x.com/_philschmid/status/1818682255675396568">来自 Philipp Schmid (@_philschmid) 的推文</a>：入夏前最重磅的开源 LLM 发布之一！@GoogleDeepMind Gemma 2 2B 权重发布！Gemma 2B 针对端侧（on-device）和边缘（edge）推理进行了优化。Gemma 2 2B 在 2 万亿个 tokens 上进行了训练...</li><li><a href="https://x.com/togethercompute/status/1818706177238397155">来自 Together AI (@togethercompute) 的推文</a>：最近关于不同推理提供商在使用 Meta 的 Llama 3.1 模型时，因实现方式不同而导致的质量差异引起了广泛讨论。在下面的博客文章中，我们...</li><li><a href="https://x.com/StanfordAILab/status/1818669016325800216">来自 Stanford AI Lab (@StanfordAILab) 的推文</a>：arXiv -> alphaXiv。斯坦福大学的学生们构建了 alphaXiv，这是一个针对 arXiv 论文的开放讨论论坛。@askalphaxiv 您可以通过更改...直接在任何 arXiv 论文上发布问题和评论。</li><li><a href="https://x.com/dzhulgakov/status/1818753731573551516">来自 Dmytro Dzhulgakov (@dzhulgakov) 的推文</a>：这是你吗？我们在 Together playground 上运行了 3 次你们的展示案例，结果每次不是陷入死循环就是回答错误。很好奇这怎么能通过你们质量测试的所有 5 个步骤...</li><li><a href="https://apnews.com/article/ai-open-source-white-house-f62009172c46c5003ddd9481aa49f7c3">白宫表示无需限制“开源”人工智能——至少目前如此</a>：白宫表态支持“开源”人工智能技术，在周二的一份报告中辩称，目前没有必要对制造关键组件的公司实施限制...</li><li><a href="https://x.com/swyx/status/1818708147630227779">来自 swyx 🌉 back in SF! (@swyx) 的推文</a>：谁会是第一个构建 Golden Gate Gemma 的人？令人惊喜的是：在 Gemma 2 2B 击败 GPT3.5（！）的同时，@NeelNanda5 等人发布了涵盖 2B 和 9B 的 400 个 SAE，以及 Neuronpedia...</li><li><a href="https://www.together.ai/blog/llama-31-quality">Llama 3.1：同样的模型，不同的结果。一个百分点的差异。</a>：未找到描述</li><li><a href="https://x.com/robdadashi/status/1818682005569048599?s=46">来自 Robert Dadashi (@robdadashi) 的推文</a>：Gemma 2 2B 来了！就其体积而言，性能非常出色，非常适合研究和应用。我为我们团队在过去几个月取得的进展感到非常自豪！</li><li><a href="https://x.com/dzhulgakov/status/1818753736359178414">来自 Dmytro Dzhulgakov (@dzhulgakov) 的推文</a>：示例：AI 研究员的问题“什么是 Group Query Attention？” 宣称：事实正确且详细的回答。现实：回答暗示 GQA 是某种形式的序列稀疏注意力。然而...</li><li><a href="https://x.com/jaminball/status/1818409214378946935?s=61">来自 Jamin Ball (@jaminball) 的推文</a>：关于微软 AI 相关产品的一些数据/统计。真实的收入！Azure AI 服务 - 50 亿美元运行率，同比增长 900% - 6 万客户，同比增长 60% - 贡献了本季度 Azure 整体增长的约 8%...</li><li><a href="https://x.com/lmsysorg/status/1818694982980845685?s=46">来自 lmsys.org (@lmsysorg) 的推文</a>：祝贺 @GoogleDeepMind 发布 Gemma-2-2B！Gemma-2-2B 已在 Arena 中以“guava-chatbot”代号进行了测试。仅凭 2B 参数，它就获得了 1130 分的骄人成绩，与...持平。</li><li><a href="https://x.com/nathanhabib1011/status/1818686787247575253">来自 Nathan (@nathanhabib1011) 的推文</a>：它已经登上排行榜了！该模型在其参数级别中处于顶尖地位，祝贺 @GoogleDeepMind 团队！引用 Google DeepMind (@GoogleDeepMind)：我们迎来了一个新的 2...</li><li><a href="https://youtu.be/qP3rXJc_L5Y?si=z52-nyB0Ov0lUCkg">自主合成对话（以及其他近期合成数据）</a>：一场关于我们近期启动的合成数据项目的演讲。详情见下方。https://arxiv.org/abs/2407.18421 幻灯片：https://docs.google.com/presentat...</li><li><a h

ref="https://www.youtube.com/watch?v=y1WnHpedi2A">你认为 ChatGPT 能够推理吗？</a>：Subbarao Kambhampati 教授认为，虽然 LLM 是令人印象深刻且有用的工具，尤其是在创意任务方面，但它们在逻辑上存在根本性的局限性...</li><li><a href="https://developers.googleblog.com/en/smaller-safer-more-transparent-advancing-responsible-ai-with-gemma/">更小、更安全、更透明：利用 Gemma 推进负责任的 AI</a>：未找到描述</li><li><a href="https://github.com/InternLM/MindSearch">GitHub - InternLM/MindSearch: 🔍 一个基于 LLM 的多 Agent 搜索引擎框架，类似于 Perplexity.ai Pro 和 SearchGPT</a>：🔍 一个基于 LLM 的多 Agent 搜索引擎框架，类似于 Perplexity.ai Pro 和 SearchGPT - InternLM/MindSearch</li><li><a href="https://github.com/traceloop/openllmetry">GitHub - traceloop/openllmetry: 基于 OpenTelemetry 的 LLM 应用开源可观测性工具</a>：基于 OpenTelemetry 的 LLM 应用开源可观测性工具 - traceloop/openllmetry</li><li><a href="https://github.com/wandb/openui">GitHub - wandb/openui: OpenUI 让你用想象力描述 UI，然后实时查看渲染效果。</a>：OpenUI 让你用想象力描述 UI，然后实时查看渲染效果。 - wandb/openui</li><li><a href="https://github.com/raidendotai/openv0">GitHub - raidendotai/openv0: AI 生成的 UI 组件</a>：AI 生成的 UI 组件。通过在 GitHub 上创建账号为 raidendotai/openv0 的开发做出贡献。</li><li><a href="https://buttondown.email/ainews/archive/ainews-to-be-named-5098/">[AINews] 今天没发生什么大事</a>：这是一个平静的一天。2024年7月29日至7月30日的 AI 新闻。我们为你检查了 7 个 subreddit、384 条 Twitter 和 28 个 Discord（248 个频道，2257 条消息）....
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1268276890733510688)** (1 条消息): 

> - `AgentInstruct`
> - `AutoEvolInstruct`
> - `Apple Intelligence Paper`
> - `LLM Paper Club` 


- **AgentInstruct 和 AutoEvolInstruct 演示**：在 **20 分钟**后，核心成员将在即将开始的会议中讨论 **AgentInstruct** 和 **AutoEvolInstruct** 的功能。
   - *加入我们以了解更多关于它们的应用*，并回顾它们在最近的 **Apple Intelligence 论文**中是如何被使用的。
- **LLM Paper Club 洞察**：**LLM Paper Club** 下周将举行专题会议，重点关注 **AgentInstruct** 和 **Orca 3**，以及 **AutoEvolInstruct**。
   - 鼓励参与者**加入**这些富有见地的讨论，并**点击此处查看活动详情** [Latent.Space events](http://Latent.Space)。



**提到的链接**：<a href="https://lu.ma/cr4jbuli">LLM Paper Club (MSR 专题：AgentInstruct/Orca 3, AutoEvolInstruct) · Zoom · Luma</a>：@sam, @vibhu, @alpay 将带领我们深入了解 AgentInstruct (https://arxiv.org/abs/2407.03502) 和 AutoEvolInstruct (https://arxiv.org/abs/2406.00770)！

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1268209988002648284)** (4 条消息): 

> - `Quantization in LLMs`
> - `Axolotl early stopping features`
> - `Manual termination of training runs`
> - `Gema2b discussion` 


- **关于 Attention 层量化的好奇**：一位成员提出了一个问题，即 LLM 的 **Attention 层**参数是否使用了与 Feed Forward 层类似的方法进行量化，并引用了一篇关于[量化的科普文章](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)。
   - 这场讨论凸显了在保持性能的同时使 LLM 更小化的持续关注。
- **Axolotl 的 Early Stopping 功能**：一个疑问涉及 **Axolotl** 是否提供在 Loss 渐近收敛或验证 Loss 增加时自动终止训练运行的功能，暗示了对有效 Early Stopping 指标的需求。
   - 重点在于通过及时干预来提高训练效率和模型性能。
- **手动终止与 LoRA Adapter 保存**：一位成员询问是否可以手动终止训练运行并保存最近的 **LoRA Adapter**，而不必取消整个运行。
   - 该功能将使用户能够更好地控制其训练过程。
- **关于 Gema2b 的讨论咨询**：一位成员询问是否有人见过 **gema2b**，这表明了对该话题的社区关注或更新的兴趣。
   - 这反映了社区内潜在的持续好奇心或协作探索。



**提到的链接**：<a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization">量化视觉指南</a>：探索 LLM 的内存效率技术

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1267942174280908850)** (7 messages): 

> - `Gemma-2-27b Tuning Config`
> - `Roles_to_train in chat_template`
> - `Default roles to train fix`
> - `Logging verbosity adjustments` 


- **对 Gemma-2-27b 配置的需求**：一位用户询问了关于 **Gemma-2-27b** 的可用微调配置，强调了社区中的这一需求。
   - 目前尚未提供具体的配置，表明在共享知识方面存在空白。
- **新的 'roles_to_train' 要求**：用户注意到最近的一个 [PR](https://github.com/axolotl-ai-cloud/axolotl/pull/1756) 在使用 `type: chat_template` 时引入了对 `roles_to_train` 字段的要求，这导致现有的示例失效。
   - 这一变化需要更新文档，以澄清其在训练过程中的用法。
- **已解决 roles_to_train 的默认值问题**：一名成员确认已合并一个修复程序，将默认的 `roles_to_train` 设置为 `assistant, gpt`，缓解了之前标签生成的问题。
   - 这一调整应能提高训练配置的易用性并简化设置流程。
- **通过 Pull Requests 解决**：最终的解决方案来自另一个独立的 [PR](https://github.com/axolotl-ai-cloud/axolotl/pull/1801/files)，旨在修复默认角色设置并降低日志详细程度（verbosity）。
   - 该修复被认为解决了一个疏忽，即之前在基本默认设置下只有最后一个 token 被考虑，从而增强了整体功能。
- **已解决问题的确认**：一位用户确认在将更改合并到 main 分支后，之前的 `roles_to_train` 问题现已解决。
   - 社区对这些更新和修复表示赞赏，反映了增强配置选项的协作努力。



**提到的链接**：<a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1801/files">fix roles to train defaults and make logging less verbose by winglian · Pull Request #1801 · axolotl-ai-cloud/axolotl</a>：这修复了在基本默认设置下，除最后一个 token 外所有内容都被忽略的问题。

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1267921049207443506)** (44 messages🔥): 

> - `Training QLora and Lora models`
> - `Challenges in training a support AI`
> - `Fine-tuning Llama models`
> - `Retrieval Augmented Generation (RAG)`
> - `Data cleaning for conversation datasets` 


- **澄清 QLora 和 Lora 模型的区别**：一名成员澄清说，**QLora** 通常量化为 **4-bit**，而 **Lora** 使用 **8-bit**。
   - 这种区分有助于理解两种模型之间的性能影响及其训练行为。
- **训练支持型 AI 面临挑战**：一名成员表达了他们在训练支持型 AI 方面的困扰，考虑使用 **Llama** 或基础模型，并使用 **RTX 4090** 进行计算。
   - 他们建议高质量的数据集至关重要，微调可能会产生更好的结果，特别是使用 **LoRA**。
- **微调 Llama 显示出潜力**：讨论强调，在 **Axolotl** 上执行任何模型时，微调 **Llama 3.1** 可能是有效的。
   - 有建议提出探索 **Retrieval Augmented Generation** (RAG) 作为一种可能更适合增强能力的方案。
- **Loss 卡在零引发担忧**：一位用户报告说他们的训练 Loss 卡在 **0.0**，询问根本原因并分享了他们关于 padding 的理论。
   - 另一名成员建议，token padding 通常应该被 mask 掉，这可能会影响 Loss 的计算。
- **对数据集清洗工具的需求**：一名成员寻求推荐一种允许编辑对话的 **dataset viewer**，表示现有工具难以使用。
   - 他们希望清洗对话集合，而不必通过 **jsonl** 进行操作，寻求一种更用户友好的解决方案。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/1268318167785013248)** (1 messages): 

> - `Serverless GPUs`
> - `AI infrastructure developments` (AI 基础设施发展)
> - `Dynamic market trends` (动态市场趋势)
> - `Deployment experiences` (部署经验)
> - `Cold starts and autoscaling` (冷启动与自动扩缩容)


- **State of Serverless GPUs 报告更新**：随着新的 [State of Serverless GPUs report](https://www.inferless.com/learn/the-state-of-serverless-gpus-part-2) 的发布，重点介绍了过去六个月中 AI 基础设施领域的重大变化。
   - *我们上一份指南引起了全球广泛关注*，提供了关于选择 Serverless 供应商的见解以及市场的各种发展。
- **来自部署 ML 模型的工程师的见解**：该分析包含了数百名在生产环境中部署机器学习模型的工程师的经验教训，揭示了 Serverless 领域中哪些方案效果良好。
   - 值得注意的是，该报告重点关注了各 Serverless GPU 供应商的 **冷启动 (cold starts)** 和 **自动扩缩容 (autoscaling) 测试**，以评估其性能。
- **Serverless 市场的动态特性**：Serverless GPU 市场的特点是其**动态性**，供应商不断努力改进其产品。
   - 这一不断发展的领域中各项改进带来的兴奋感，推动了分享及时见解和结果的需求。



**提到的链接**：<a href="https://www.inferless.com/learn/the-state-of-serverless-gpus-part-2">Serverless GPU Part 2 Benchmarking: A Comprehensive Comparison of Performance &amp; Pricing</a>：深入探讨 Serverless GPU 平台。探索冷启动时间、集成挑战、价格对比以及自动扩缩容能力。通过我们详细的分析做出明智的选择...

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1267958979615654012)** (3 messages): 

> - `MLflow in LlamaIndex`
> - `AI21 Labs' Jamba-Instruct model`
> - `Open-source contributions` (开源贡献)
> - `Async functionality for BedrockConverse` (BedrockConverse 的异步功能)
> - `Token improvements` (Token 改进)


- **MLflow 集成至 LlamaIndex**：MLflow 现已在 LlamaIndex 中可用，提供了一个统一的平台来管理模型开发、部署和管理，具有追踪 Prompt、LLM 和工具的功能。
   - 它旨在将 LlamaIndex 引擎及其依赖项打包，从而简化开发工作流。
- **AI21 Labs 的 Jamba-Instruct 模型发布**：AI21 Labs 的新 Jamba-Instruct 模型具有 **256K Token** 的上下文窗口，现在可以通过 LlamaIndex 访问以创建 RAG 应用程序。
   - 最近的一篇客座文章强调，有效利用长上下文窗口对于获得最佳结果至关重要。
- **庆祝开源贡献**：开源用户做出了重大贡献，包括由 **@andrecnferreira** 实现的 BedrockConverse 模型的异步功能。
   - 此更新解决了 GitHub 上的多个问题，确保了功能和性能的提升。
- **GitHub 用户增强 Token 管理**：用户 **joelbarmettlerUZH** 帮助改进了 LlamaIndex 中的 Token 管理，为整体效率做出了贡献。
   - 他们的工作持续支持 LlamaIndex 平台的稳健性，使所有用户受益。



**提到的链接**：<a href="https://t.co/rn3sAKG05N">feat: ✨ Implement async functionality in `BedrockConverse` by AndreCNF · Pull Request #14326 · run-llama/llama_index</a>：描述：为 BedrockConverse LLM 实现异步方法。修复了 #10714，修复了 #14004。新包？我是否填写了 pyproject.toml 中的 tool.llamahub 部分并提供了详细的 README.m...

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1268040790362423359)** (49 messages🔥): 

> - `MLflow 集成问题`
> - `vLLM 文档 PR`
> - `RAG 可观测性关注点`
> - `Llama 产品命名混淆`
> - `RagApp 替代方案` 


- **LlamaIndex 的 MLflow 集成遇到错误**：MLflow 与 LlamaIndex 的集成产生了诸如 `TypeError: Ollama.__init__() got an unexpected keyword argument 'system_prompt'` 之类的错误，表明存在兼容性问题。
   - 进一步的测试还显示，在尝试使用外部存储上下文（storage contexts）创建向量存储索引时也出现了失败。
- **vLLM 文档 PR 已成功提交**：一名成员宣布在 vLLM 文档中为 LlamaIndex 服务提交了官方 PR，该 PR 已通过所有检查并等待批准。
   - 此举旨在增强关于在 LlamaIndex 中使用 vLLM 的文档说明。
- **关于 RAG 可观测性问题的讨论**：一位用户报告了依赖项之间的冲突，具体表现为他们的 RAG 项目需要 Pydantic v1，而 OpenTelemetry 则需要 Pydantic v2。
   - 另一位成员强调，大多数组件应该能够与 Pydantic v2 协同工作，暗示了潜在的可调节性。
- **关于 Llama 产品和命名的混淆**：有反馈建议 LlamaIndex 应该澄清产品名称和定义，以避免用户对 LlamaExtract 和 LlamaParse 等服务产生混淆。
   - 这种混淆因重叠的产品名称和宣传信息而加剧，使用户难以理解具体的产品功能。
- **探索 RagApp 的替代方案**：一位用户询问了 RagApp 的替代方案，并提到 Verba 和 OpenGTPs 等现有选项存在局限性。
   - 另一位成员建议将 create-llama 作为潜在的替代方案，显示出用户正在积极寻找有效的工具。



**提及的链接**：<a href="http://127.0.0.1:3000")">no title found</a>: no description found

  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1268172578409676862)** (2 messages): 

> - `全文档检索`
> - `Medium 内容质量担忧` 


- **超越分块（Chunking）增强检索**：一篇题为 [Beyond Chunking: Simplifying RAG with Full-Document Retrieval](https://medium.com/ai-advances/beyond-chunking-simplifying-rag-with-full-document-retrieval-911c757cb399) 的文章讨论了使用全文档检索而非传统的分块技术来**简化检索增强生成** (RAG)。
   - 该方法提出了一种更高效的方案，可能会重塑我们在 AI 框架中处理文档检索的方式。
- **对 Medium 内容质量的批评**：一位成员表示应该放弃 **Medium**，原因是担心该平台发布了大量低质量内容。
   - 他们还开了一个轻松的玩笑，认为该平台正被缺乏价值的内容所淹没。


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1268048635606864026)** (12 messages🔥): 

> - `LLAMA_3 模型输出`
> - `生成参数`
> - `Top_p 和 Frequency_penalty 设置`
> - `Temperature 设置的影响`
> - `部署之间的质量对比` 


- **LLAMA_3 输出在不同平台间存在差异**：一位用户测试了 **LLAMA_3 8B (instruct)** 模型，发现其输出质量不如另一个 Playground ([链接](https://sdk.vercel.ai/playground)) 的结果。
   - 他们质疑为什么在参数相同的情况下，相似的模型会产生不同的结果。
- **关于生成参数的讨论**：一位成员指出 **generation parameters** 可能存在差异，建议默认值可能因平台而异，从而影响输出质量。
   - 另一位成员解释说，模型设置中缺少 **top_p** 和 **frequency_penalty** 可能会影响输出，并将其与更全面的在线系统进行了对比。
- **Temperature 设置与创造性**：对话强调了较高的 **temperature** 设置可能会增强输出的创造性，并指出在线默认值低于用户设置的 **0.8**。
   - 尽管在线使用了 **0.8**，用户仍获得了更好的结果，这表明单纯的 Temperature 调整可能无法解释模型性能的差异。
- **理解 generate recipes 中的调试**：一位成员询问 **generate recipe** 被设计用于调试的含义，以及为了获得更好的结果可能还缺少什么。
   - 这突显了对当前设置过于简单的担忧，暗示其可能不是高质量输出的最佳选择。



**提及的链接**：<a href="https://sdk.vercel.ai/playground">AI Playground | Compare top AI models side-by-side</a>：并排聊天并对比 OpenAI GPT, Anthropic Claude, Google Gemini, Llama, Mistral 等模型。

  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1268090150202769409)** (25 messages🔥): 

> - `ChatPreferenceDataset 更新`
> - `FSDP 和 QAT 兼容性`
> - `参数命名讨论`
> - `合并 PR`
> - `FSDP2 功能` 


- **讨论 ChatPreferenceDataset 的变更**：一名成员分享了对 [ChatPreferenceDataset](https://gist.github.com/RdoubleA/fb6dbf0db0099eafbadd31fe789459d1) 的本地修改，以更好地组织消息转换和 prompt 模板化，从而与当前的 RFC 保持一致。
   - 另一名成员对澄清表示感谢，信号表明已准备好继续推进。
- **FSDP2 应支持 quantization 和 compile**：大家达成共识，虽然 **FSDP** 不支持 quantization 或 compilation，但 **FSDP2** 预计将支持这两者，特别是针对某些 tensor 类型的 **quant**。
   - 同时也提出了 **QAT** (Quantization-Aware Training) 与 FSDP2 兼容性的担忧，并建议进一步测试其功能。
- **函数的参数命名**：围绕是否将函数参数从 **filter_fn** 重命名为更通用的 **map_fn** 展开了讨论，以便在 dataset 代码中实现更好的可组合性。
   - 成员们辩论了该名称的描述性，最终认为当前名称已足够清晰。
- **潜在的 PR 合并**：有人提议将变更合并到关于统一 dataset 的 PR 中，一名成员建议如果是这种情况，他们将关闭自己的 PR。
   - PR 的所有者表示，在另一个待处理的 pull request 经过审查并合并后，他们将提交一个单独的 PR。
- **澄清 FSDP2 功能**：一名成员确认 **FSDP2** 应该支持 **quantization**，但对 **QAT** 与当前 QAT recipe 及 FSDP2 的兼容性表示不确定。
   - 对话留下了关于 FSDP2 实际应用及其对 QAT 集成影响的悬而未决的问题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/pull/1234">[1/n] Merged fine-tuning dataset: grammar + samsum by RdoubleA · Pull Request #1234 · pytorch/torchtune</a>: Context What is the purpose of this PR? Is it to   add a new feature  fix a bug  update tests and/or documentation  other (please add here)  As discussed in the RFC in #1186, we will merged instruc...</li><li><a href="https://github.com/pytorch/torchtune/pull/1186">[RFC] Unified dataset with data and model transforms by RdoubleA · Pull Request #1186 · pytorch/torchtune</a>: Thanks to @pbontrager for all the discussions to help converge to this design. TLDR:  Let’s make a general fine-tuning dataset class that takes in a data transform class and a model transform class...
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1267938475106238631)** (20 messages🔥): 

> - `Google Gemini 上下文缓存`
> - `从 Agent 流式传输 token`
> - `LangChain 错误与问题`
> - `使用 LangChain 工具` 


- **关于 Google Gemini 集成的不确定性**：一位用户询问 **Google Gemini context caching** 是否已集成到 LangChain 中，并指出该功能缺乏明确信息。
   - 另一位参与者澄清说 **LangChain** 确实支持像 `gemini-pro` 这样的 Gemini 模型，但关于 context caching 的具体细节尚不确定。
- **如何从 Agent 流式传输 token**：分享了一份指南，详细介绍了如何使用 `.astream_events` 方法从 **LangChain** 中的 Agent 流式传输 token。
   - 该方法允许异步流式传输 Agent 事件、处理事件类型，并专门打印 **on_chat_model_stream** 事件内容。
- **LangChain Pydantic 类型错误**：一位用户对 Pydantic 验证错误表示沮丧，该错误提示 'CountStuff expected dict not list'。
   - 另一位用户建议该错误表明代码中可能使用了错误的数据类型。
- **初次使用 LangChain 工具的问题**：一位 LangChain 新手在从网络获取当前事实时遇到挑战并寻求帮助。
   - 回复要求提供更多关于所遇错误的详细信息，以便更好地理解问题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://smith.langchain.com/public/83e6379c-a5b4-4459-a4ac-5e59385525a8/r">LangSmith</a>: no description found</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/agents/#streaming-tokens>))">Build an Agent | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:
</li>
</ul>

</div>

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1267924323385999492)** (2 messages): 

> - `SWE Agent Guide`
> - `Palmyra-Fin-70b`
> - `Palmyra-Med-70b`
> - `frameworks like CrewAI, AutoGen, LangChain, LLamaIndex` 


- **使用新指南构建你自己的 SWE Agent**：一位成员创建了一个指南，介绍如何使用 [CrewAI](https://git.new/swe/kit), **AutoGen** 和 **LangChain** 等框架构建 **SWE Agents**。
   - 该指南强调利用一个 Python 框架来轻松地为兼容各种 Agentic 框架的 Agent 构建脚手架（scaffolding）。
- **Palmyra-Fin-70b 在 CFA Level III 考试中创造历史**：新发布的 **Palmyra-Fin-70b** 是**首个**以 **73% 的分数**通过 CFA Level III 考试的模型，专为投资研究和财务分析而设计。
   - 它在 [Hugging Face](https://huggingface.co/Writer/Palmyra-Fin-70B-32K) 和 [NVIDIA NIM](https://build.nvidia.com/writer/palmyra-fin-70b-32k) 上以非商业开放模型许可证发布。
- **Palmyra-Med-70b 在医疗任务中表现卓越**：**Palmyra-Med-70b** 模型提供 **8k 和 32k 版本**，在 MMLU 测试中达到了令人印象深刻的 **86%**，超越了其他顶级模型。
   - 它适用于医学研究应用，并在 [Hugging Face](https://huggingface.co/Writer/Palmyra-Med-70B) 和 [NVIDIA NIM](https://build.nvidia.com/writer/palmyra-med-70b) 上提供非商业开放模型许可证。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/samjulien/status/1818652901130354724">来自 Sam Julien (@samjulien) 的推文</a>: 🔥 @Get_Writer 刚刚发布了 Palmyra-Med-70b 和 Palmyra-Fin-70b！Palmyra-Med-70b 🔢 提供 8k 和 32k 版本 🚀 MMLU 性能约 86%，超越顶级模型 👨‍⚕️ 用于诊断、制定治疗方案...</li><li><a href="https://git.new/swe/kit">SWE Python Framework - Build SWE Agents </a>: 通过 swekit（一个 Python 框架）释放 SWE Agent 的力量。轻松构建和脚手架化与 CrewAI 和 LlamaIndex 等 Agentic 框架兼容的 Agent。利用我们的工具生态系统...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1267924222701998112)** (2 messages): 

> - `SWE Agents Guide`
> - `AI Long-Term Memory Solutions` 


- **使用新指南构建你自己的 SWE Agents**：发布了一篇关于使用 LangChain 创建你自己的 **SWE Agents** 的新指南，可在 [swekit](https://git.new/swe/kit) 获取。该指南重点介绍了一个用于构建和脚手架化与 **CrewAI** 和 **LlamaIndex** 等系统兼容的 Agent 的 **Python 框架**。
- **AI 聊天机器人需要更好的记忆解决方案**：引入了一种方案来增强 AI 聊天机器人在长对话中保留细节的能力，重点关注**长期记忆**和**上下文保留**。相关的 [YouTube 视频讨论了这一点](https://www.youtube.com/watch?v=bqKfT4waYEk&lc=Ugyo9s9abyROgXQw50l4AaABAg)，其中包含一条寻求会员建议和编码支持的评论。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=bqKfT4waYEk&lc=Ugyo9s9abyROgXQw50l4AaABAg">来自 @darkmatter9583 的评论</a>: 你好，我还在考虑会员资格，你能帮我并给我建议吗？还有代码方面的帮助？25 美元的那个</li><li><a href="https://git.new/swe/kit">SWE Python Framework - Build SWE Agents </a>: 通过 swekit（一个 Python 框架）释放 SWE Agent 的力量。轻松构建和脚手架化与 CrewAI 和 LlamaIndex 等 Agentic 框架兼容的 Agent。利用我们的工具生态系统...
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1267939891832946798)** (10 messages🔥): 

> - `Open Interpreter 工作流`
> - `OS Mode 需求`
> - `4o mini 兼容性`
> - `眼动追踪技术`
> - `Visor 技术影响` 


- **澄清 Open Interpreter 工作流**：一位成员寻求关于将 Open Interpreter 与 **Llama 3.1** 配合使用的工作流澄清，特别是应该在终端会话中提问还是开启新会话。
   - 讨论中指出，**OS Mode 需要 Vision Model** 才能正常运行。
- **对 4o mini 兼容性的关注**：一位成员询问了 Open Interpreter 在新款 **4o mini** 上的表现，暗示即将到来的开发进展可能非常值得期待。
   - 目前尚未提供关于兼容性的具体细节回复。
- **对眼动追踪技术的兴奋**：一位成员表达了对使用**眼动追踪软件**的热情，强调了其通过 Open Interpreter 辅助残障人士的潜力。
   - 他们赞扬了这一倡议，表示渴望利用这些技术弥合无障碍环境的差距。
- **鼓励社区支持**：一位成员分享了他们使用 Open Interpreter 的历程，对其潜力表示感谢，并有兴趣作为**用例示例**支持进一步开发。
   - 他们的 ICU 护士和患者倡导者背景，突显了他们致力于改善神经肌肉疾病患者无障碍体验的承诺。
- **对 Visor 技术的期待**：一位成员对即将推出的 **Visor 技术**感到兴奋，预计其与 Open Interpreter 的集成将为他们的工作流带来**巨大的变革**。
   - 他们相信这项技术将显著增强他们有效导航和利用 AI 工具的能力。



**提及的链接**：<a href="https://x.com/humanoidhistory/status/1818528398358073705">来自 Humanoid History (@HumanoidHistory) 的推文</a>：太空的未来，由 Tatsushi Morimoto, Robert McCall, Günter Radtke, 和 John Berkey 绘制。

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1267938284684841067)** (10 messages🔥): 

> - `01 Server 在 Ubuntu 22.04 上的安装`
> - `01 的自定义指令`
> - `01 的社区参与`
> - `获取预订信息`
> - `Poetry 版本讨论` 


- **01 Server 安装的内核版本限制**：一位用户询问在 **Ubuntu 22.04** 上安装 **01 Server** 是否有任何**内核版本限制**。
   - 该查询反映了关于设置服务器的兼容性和要求的持续讨论。
- **寻求自定义指令建议**：一位用户征求关于 **01** 的**自定义指令**建议。
   - 讨论表明用户希望获得社区提供的优化设置指南。
- **社区成员渴望做出贡献**：一位新人表达了为 **01** 创作内容的热情，旨在向更广泛的受众展示其潜力。
   - 他们还询问了关于**预订**状态更新的集中查询地点。
- **频道访问需要 Builder 角色**：一位用户报告了一个链接的**访问问题**，对此一位成员建议他们在“**Channels and Roles**”设置中为自己授予 **Builder 角色**。
   - 这强调了社区内正确权限对于访问内容的重要性。
- **关于 Poetry 版本使用的讨论**：一位用户询问社区成员目前使用的是哪个版本的 **Poetry**。
   - 这反映了不同版本的工具使用可能会影响开发过程。


  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1268004908183982183)** (2 messages): 

> - `Perplexica`
> - `Llama-3`
> - `Open Source AI`
> - `AI-powered Search Engines` 


- **Perplexica 提供本地且免费的替代方案**：最近的一段 [YouTube 视频](https://www.youtube.com/watch?v=V0vx94JYNjI) 讨论了如何使用 **Meta AI** 的开源 **Llama-3** 构建 **Perplexity AI** 的**本地且免费的克隆版**。
   - *这一本地解决方案旨在超越现有的搜索技术*，同时提供更高的可访问性。
- **在 GitHub 上探索 Perplexica**：[Perplexica](https://github.com/ItzCrazyKns/Perplexica) 被誉为一款 **AI 驱动的搜索引擎**，是 **Perplexity AI** 的开源替代品。
   - 开发者可以通过其 **GitHub 仓库** 查看其功能并为其发展做出贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=V0vx94JYNjI">Perplexica + Llama-3.1 (405B, 70B, 8B) : 这个 Perplexity 的本地且免费克隆版击败了所有人！</a>：在这段视频中，我将告诉你如何使用 Meta AI 的新开源 Llama-3 来搭建 Perplexity 和 SearchGPT 的本地且免费替代方案……</li><li><a href="https://github.com/ItzCrazyKns/Perplexica">GitHub - ItzCrazyKns/Perplexica: Perplexica 是一款 AI 驱动的搜索引擎。它是 Perplexity AI 的开源替代品</a>：Perplexica 是一款 AI 驱动的搜索引擎。它是 Perplexity AI 的开源替代品 - ItzCrazyKns/Perplexica
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

pavl_p: 听起来他们将 DSPy 与符号学习器（symbolic learner）集成了。令人兴奋！
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1267956185357881434)** (15 messages🔥): 

> - `DSPy Module Penalty System`
> - `Launching on ProductHunt`
> - `Using DSPy for Product Development`
> - `Cache Management in DSPy`
> - `Schema-Aligned Parsing Proposal` 


- **DSPy 模块惩罚系统详解**：一位成员讨论了如何在 DSPy 中实现惩罚系统，通过使用将偏离标准答案（gold answer）的偏差进行平方的公式，将其转化为用于优化的负向指标。
   - 这种方法允许优化器专注于最小化惩罚，从而更好地对齐预期结果。
- **ProductHunt 发布热潮**：一位成员宣布他们在 [ProductHunt](https://www.producthunt.com/posts/creatr-3) 上发布了 Creatr，旨在收集反馈并推广他们的产品设计工具。
   - 另一位成员通过投票表示支持，并询问项目中是否使用了 DSPy。
- **DSPy 在产品开发中的应用**：在 ProductHunt 上发布的该产品专门在其编辑子模块中利用了 DSPy，增强了其功能。
   - 这展示了 DSPy 在真实产品工作流中的适用性，为潜在用户提供了深刻的参考背景。
- **DSPy 中的缓存管理**：一位成员寻求关于如何完全删除 DSPy 模块内缓存的建议，以纠正测试过程中不一致的指标结果。
   - 这突显了与模块状态相关的问题，以及在测试过程中对干净起始状态的需求。
- **数据解析技术的进步**：一位成员建议使用 [结构化生成（structured generation）](https://www.boundaryml.com/blog/schema-aligned-parsing) 方法来改进 JSON 输出解析，以获得更好的可靠性和更少的重试次数。
   - 该方法提议使用 Schema-Aligned Parsing (SAP) 来有效处理 LLM 的随机性，从而减少 Token 使用并确保有效的序列化。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.boundaryml.com/blog/schema-aligned-parsing">Prompting vs JSON Mode vs Function Calling vs Constrained Generation vs SAP</a>：未找到描述</li><li><a href="https://www.producthunt.com/posts/creatr-3"> Creatr - 在 100 秒内将你的想法转化为设计原型 | Product Hunt</a>：使用 Creatr 更快地创建、构建、协作和交付产品！我们正在开发一款为所有人简化产品设计的工具。这让任何拥有创意愿景的人都能制作出精美的……
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1267963005555965983)** (7 messages): 

> - `UCSC Colloquium Talk`
> - `OpenCL Resource Errors`
> - `Brazilian AI Investment Plan`
> - `Discord Rules Reminder` 


- **UCSC 学术报告会讨论并行计算**：一位成员分享了一个名为 [“我想要一台好的并行计算机”](https://youtu.be/c52ziyKOArc?si=pAUdzwIQGXCtpk3T) 的 [YouTube 视频](https://youtu.be/c52ziyKOArc?si=pAUdzwIQGXCtpk3T)，该视频来自 2024 年 4 月 10 日的 UC Santa Cruz CSE 学术报告会。
   - 视频描述中提供了相关幻灯片的链接。
- **Mac 上 OpenCL 的挑战**：一位成员对在 Mac 上使用 OpenCL 产生 “out of resources” 错误表示困惑，提到他们只收到了 “invalid kernel” 错误。
   - 这表明问题可能出在 Kernel 编译上，而不是资源分配。
- **巴西宣布重大 AI 投资计划**：巴西政府公布了一项 AI 投资计划，承诺到 2028 年投入 **230 亿雷亚尔**，其中包括一个耗资 **18 亿雷亚尔** 的 **supercomputer** 项目。
   - 该计划旨在通过激励和资助来刺激当地 AI 产业，目前正等待总统批准后实施。
- **纳税人的钱与科技公司**：针对巴西的 AI 投资计划，一位成员幽默地评论了纳税人的钱支持像 **NVIDIA** 这样公司的讽刺性。
   - 这突显了公众对科技行业投资中公共资金分配的关注。
- **关于 Discord 宗旨的提醒**：频道发布了一项提醒，强调 Discord 频道的主要焦点是围绕 **tinygrad** 的 **development** 和 **usage** 进行讨论。
   - 这作为成员保持话题相关性并促进富有成效讨论的指南。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/c52ziyKOArc?si=pAUdzwIQGXCtpk3T">I want a good parallel computer - UCSC Colloquium</a>：这是我在 2024 年 4 月 10 日 UC Santa Cruz CSE 学术报告会上所做演讲的视频。幻灯片可以在这里找到：https://docs.google.com/presentation/d...</li><li><a href="https://oglobo-globo-com.translate.goog/economia/noticia/2024/07/30/plano-brasileiro-de-ia-preve-r-23-bi-supercomputador-e-sistema-para-o-sus.ghtml?_x_tr_sl=pt&_x_tr_tl=en&_x_tr_hl=pt-BR&_x_tr_pto=wapp>">Plano brasileiro de IA prevê R$ 23 bi, supercomputador e sistema para o SUS</a>：提案已在委员会通过，需经政府验证。
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1267929562436866130)** (6 messages): 

> - `jit compilation`
> - `step function optimization` 


- **关于 JIT 编译策略的辩论**：*Cecilian* 询问是应该只对 **model forward step** 进行 JIT，还是对整个 **step function** 进行 JIT。
   - 建议除非有特殊原因，否则最好对 **整个 step** 进行 JIT 以获得更好的性能。
- **模型优化考量**：讨论还暗示了 **model optimization** 的必要性，重点在于是有选择地应用 JIT 编译还是全面应用。
   - 考虑到性能影响，共识倾向于通过对完整 step 进行 JIT 来实现更高效的方法。


  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1268249357204062208)** (4 messages): 

> - `Goldman Sachs report`
> - `General AI interest`
> - `Recommendation Systems` 


- **高盛报告对 AI 兴趣的影响**：成员们讨论了最近的一份 [高盛报告](https://link.to.report)，该报告将焦点从 GenAI 转移开，表明了 AI 社区中更广泛的情绪。
   - *注意*：该报告引发了关于 AI 兴趣走向的讨论。
- **关于通用 AI 兴趣的讨论**：一位成员对该频道的存在表示热忱，强调了 AI 爱好者中普遍对 GenAI 的关注。
   - 其他人也表达了同样的看法，表明大家共同希望探索 AI 领域内更多样的话题。
- **对推荐系统的兴趣**：一位用户指出他们的主要兴趣在于推荐系统 (recsys)，这标志着在 AI 话题中的独特偏好。
   - 对话暗示了深入讨论和洞察 recsys 应用及进展的机会。


  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1268105300687458386)** (1 条消息): 

> - `问答应用开发`
> - `LLM 在游戏中的应用`
> - `用户参与度统计` 


- **问答应用利用 LLM 生成问题**：一款新开发的问答应用利用 **LLM** 来生成引人入胜的问题，可以通过 [这里](https://mihaiii-trivia.hf.space/) 访问。
   - 鼓励用户查看 [如何游玩](https://mihaiii-trivia.hf.space/how-to-play) 指南以获取说明，以及 [统计数据](https://mihaiii-trivia.hf.space/stats) 和 [常见问题](https://mihaiii-trivia.hf.space/faq) 的链接。
- **通过游戏机制提升参与度**：根据初步反馈，该问答应用结合了游戏机制来增强用户参与度和留存率。**引人入胜的游戏玩法**和易于理解的界面被用户强调为重要特征。



**提到的链接**：<a href="https://mihaiii-trivia.hf.space/">FastHTML 页面</a>：未找到描述

  

---



---



---



---



---



{% else %}


> 完整的逐频道详细分析已针对电子邮件进行截断。 
> 
> 如果您想查看完整的详细分析，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}