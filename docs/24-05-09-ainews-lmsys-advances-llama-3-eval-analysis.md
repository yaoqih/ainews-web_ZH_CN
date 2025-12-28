---
companies:
- lmsys
- openai
- google-deepmind
- isomorphic-labs
date: '2024-05-10T00:52:45.516571Z'
description: '**LMSys** 正在通过将性能划分为 **8 个查询子类别**和 **7 个提示词复杂程度**来提升大语言模型（LLM）的评估体系，这揭示了
  **Llama-3-70b** 等模型在不同领域优势分布不均的现状。**DeepMind** 发布了 **AlphaFold 3**，通过对蛋白质-DNA-RNA
  复合物的整体建模，推进了分子结构预测技术，对生物学和遗传学研究产生了深远影响。**OpenAI** 推出了 **Model Spec**（模型规范），这是一项旨在明确模型行为和微调方向的公共标准，在征求社区反馈的同时，目标是让模型能直接从中学习。**Llama
  3** 在 LMSys 排行榜上已名列前茅，其性能几乎与 **Claude-3-sonnet** 持平，但在处理复杂提示词时表现出显著差异。该分析凸显了模型基准测试和行为塑造领域不断演变的格局。'
id: f796c5d2-c3ba-454e-ba2f-c0bb31b5008f
models:
- llama-3-70b
- llama-3
- claude-3-sonnet
- alphafold-3
original_slug: ainews-lmsys-advances-llama-3-eval-analysis
people:
- demis-hassabis
- sam-altman
- miranda-murati
- karina-nguyen
- joanne-jang
- john-schulman
title: LMSys 推进 Llama 3 评估分析。
topics:
- benchmarking
- model-behavior
- prompt-complexity
- model-specification
- molecular-structure-prediction
- performance-analysis
- leaderboards
---

<!-- buttondown-editor-mode: plaintext -->**LLM 评估很快将根据类别和 Prompt 复杂度而有所不同。**

> 2024年5月8日至5月9日的 AI 新闻。我们为您检查了 7 个 subreddit、[373 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 28 个 Discord（419 个频道，3747 条消息）。预计节省阅读时间（以 200wpm 计算）：**450 分钟**。

LMSys 以基于 ELO（技术上是 [Bradley-Terry](https://lmsys.org/blog/2023-12-07-leaderboard/)）的对战而闻名，更具争议的是它为 [OpenAI、Databricks 和 Mistral 进行的不透明预发布测试模型](https://www.interconnects.ai/p/chatbotarena-the-future-of-llm-evaluation#%C2%A7gptchatbot-and-lmsyss-incentives)，但直到最近才开始通过将评分拆分为 8 个查询子类别来深化其分析：

 
![image.png](https://assets.buttondown.email/images/6f5379a0-2938-4952-9b20-f107416dd59e.png?w=960&fit=max)
 

这些类别的维度[即将爆炸式增长](https://x.com/lmsysorg/status/1788363029387899016)。[LMSys 发布了关于 Llama-3 在 LMSys 上表现的深度分析](https://lmsys.org/blog/2024-05-08-llama3/)，揭示了其在重要类别（如摘要、翻译和编程）中出人意料且不均衡的胜率。

 
![image.png](https://assets.buttondown.email/images/a9d1f078-8dce-4c6d-8758-981387034b7b.png?w=960&fit=max)
 

以及针对 7 个级别的 Prompt 复杂度：

 
![image.png](https://assets.buttondown.email/images/df93a7b6-6702-4f04-bb0e-e62af11dcd58.png?w=960&fit=max)
 

随着 GPT4T-preview 级别的模型趋于商品化，且 LMSys 日益成为可能被微妙手段操纵的受信任评估标准，了解模型表现过佳或不足的主要方式至关重要。LMSys 正在主动这样做固然很好，但令人好奇的是，这次分析的 Notebooks 并没有按照他们惯常的做法发布。


---

**目录**

[TOC] 



---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**AlphaFold 3 与分子结构预测**

- **DeepMind 发布 AlphaFold 3**：[@demishassabis](https://twitter.com/demishassabis/status/1788229162563420560) 宣布了 AlphaFold 3，它能以最先进的准确度预测蛋白质、DNA 和 RNA 的结构及相互作用。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1788223454317097172) 解释了它是如何与 @IsomorphicLabs 共同构建的，以及它对生物学的意义。
- **AlphaFold 3 的能力**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1788223460390354978) 分享道，AlphaFold 3 使用了下一代架构和训练方式，从整体上计算整个分子复合物。它可以模拟在受到干扰时控制细胞功能和疾病的化学变化。
- **应用与影响**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1788224328498098562) 指出，已有超过 180 万人使用 AlphaFold 来加速生物可再生材料和遗传学领域的工作。[@DrJimFan](https://twitter.com/DrJimFan/status/1788233450123936020) 称，当数据转换为浮点序列（float sequences）时，用于像素的相同骨干网络（backbone）竟然可以想象出蛋白质，这简直令人难以置信。

**OpenAI Model Spec 与塑造模型行为**

- **OpenAI 推出 Model Spec**：[@sama](https://twitter.com/sama/status/1788260474574000152) 宣布了 Model Spec，这是一份关于 OpenAI 模型应如何行为的公开规范，旨在明确什么是 Bug，什么是决策。[@gdb](https://twitter.com/gdb/status/1788257732811755524) 分享道，该规范旨在让人们了解模型行为是如何调整的。
- **Model Spec 的重要性**：[@miramurati](https://twitter.com/miramurati/status/1788357302506139664) 强调，随着模型决策能力的提高，Model Spec 对于人们理解并参与塑造模型行为的辩论至关重要。[@karinanguyen_](https://twitter.com/karinanguyen_/status/1788256852733468842) 指出，该规范必须考虑广泛的细微问题和观点。
- **反馈与未来计划**：[@sama](https://twitter.com/sama/status/1788260475748421726) 感谢了 OpenAI 团队，特别是 @joannejang 和 @johnschulman2，并欢迎反馈以随时间调整规范。OpenAI 正在研究让模型直接从 Model Spec 中学习的技术。

**Llama 3 在 LMSYS 排行榜上的表现**

- **Llama 3 登顶排行榜**：[@lmsysorg](https://twitter.com/lmsysorg/status/1788363018449166415) 分享的分析显示，Llama 3 已攀升至排行榜前列，其 70B 版本几乎与 Claude-3 Sonnet 持平。去重（Deduplication）和离群值（outliers）对其胜率没有显著影响。
- **优势与劣势**：[@lmsysorg](https://twitter.com/lmsysorg/status/1788363020894372039) 发现，在基于复杂性和领域知识等标准的更具挑战性的提示词（prompts）上，Llama 3 与顶级模型之间的差距会变大。[@lmsysorg](https://twitter.com/lmsysorg/status/1788363027357876498) 还指出，与其他模型相比，Llama 3 的输出更友好、更具对话性，且使用了更多的感叹号。
- **达到顶级模型水平**：[@lmsysorg](https://twitter.com/lmsysorg/status/1788363029387899016) 总结道，Llama 3 在整体用例上的表现已达到与顶级闭源模型（proprietary models）相当的水平，并预计将根据此分析在排行榜上推出新类别。[@togethercompute](https://twitter.com/togethercompute/status/1788377829975171389) 同意 Llama-3-70B 已达到与顶级开源模型相似的质量。

**仅文本训练对 AI 的局限性**

- **需要实践经验**：[@ylecun](https://twitter.com/ylecun/status/1788473350177599888) 认为，关于新手除了书本知识外还需要实践经验的陈词滥调，说明了为什么仅在文本上训练的 LLM 无法达到人类智能。

---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取现在可以运行了，但还有很多改进空间！

**AI 与技术发展**

- **OpenAI 和 Microsoft 正在开发耗资 1000 亿美元的“Stargate” AI 超级计算机**：在 /r/technology 中，据报道 [**OpenAI 和 Microsoft 正在开发一个大规模的核动力超级计算机项目**](https://www.telegraph.co.uk/business/2024/05/05/ai-boom-nuclear-power-electricity-demand/)，以支持下一代 AI 的突破，这暗示了所需的巨大计算资源。

- **DeepMind 发布 AlphaFold 3，用于预测生命关键分子**：[DeepMind 推出了 AlphaFold 3](https://twitter.com/GoogleDeepMind/status/1788223454317097172?t=Jl_iIVcfo3zlaypLBUqwZA&s=19)，这是一款 AI 模型，能够以**最先进的准确度预测蛋白质、DNA 和 RNA 的结构及相互作用**，为药物研发和合成生物学的进步开启了大门。

- **IBM 发布开源 Granite Code LLM，性能超越 Llama 3**：IBM [发布了 Granite Code](https://analyticsindiamag.com/ibm-releases-open-source-granite-code-models-outperforms-llama-3/)，这是一系列**强大的专注于代码的开源语言模型，其性能击败了广受欢迎的 Llama 3 模型**。

- **Apple 推出搭载每秒 38 万亿次运算 Neural Engine 的 M4 芯片**：Apple 发布了其[下一代 M4 芯片](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/)，其特点是拥有**每秒可进行 38 万亿次 AI 运算的 Neural Engine，是目前 PC 芯片中最快的**。

**开源 LLM 进展**

- **计划将 Llama 3 70B 蒸馏为高效的 4x8B/25B MoE 模型**：/r/LocalLLaMA 社区[计划将 Llama 3 70B 模型蒸馏](https://www.reddit.com/r/LocalLLaMA/comments/1cnlmz2/planning_for_distillation_of_llama_3_70b_4x8b_25b/)为**针对 VRAM/智能权衡进行优化的 4x8B/25B Mixture-of-Experts 模型**，目标是将 8-bit 量化版本适配到 22-23GB 的 VRAM 中。 

- **过去 2 个月主要开源 LLM 发布的时间线**：/r/LocalLLaMA 整理了仅在过去几个月内[发布的众多主要开源 LLM 的时间线](https://www.reddit.com/r/LocalLLaMA/comments/1cn9sxa/timeline_of_recent_major_llm_releases_past_2/)，包括来自 **Cohere, xAI, DataBricks, ai21labs, Meta, Microsoft, Snowflake, Qwen, DeepSeek 和 IBM** 的发布。

- **Consistency LLM 作为并行解码器可将推理速度提升 3.5 倍**：[Consistency LLM](https://hao-ai-lab.github.io/blogs/cllm/) 是一种**将 LLM 转换为并行解码器的方法，可实现 3.5 倍的推理加速**，与 Medusa2/Eagle 等替代方案相比，具有相当或更好的加速效果，且无需额外的内存成本。

**AI 伦理与安全担忧**

- **OpenAI 正在探索如何负责任地生成 AI 色情内容**：OpenAI 正在[应对围绕 AI 生成色情内容的伦理挑战](https://www.wired.com/story/openai-is-exploring-how-to-responsibly-generate-ai-porn/)，考虑**针对某些用例放宽 NSFW 过滤器**，例如露骨的歌词、政治话语和浪漫小说。

- **OpenAI 推出 Model Spec 以澄清预期的模型行为**：为了帮助区分预期的模型能力与非预期的 Bug，OpenAI 正在推出 [Model Spec](https://i.redd.it/ovf2ekry2bzc1.png) 并**寻求公众反馈以使其随时间演进**。

- **美国海军陆战队测试配备 AI 瞄准步枪的机器人狗**：在令人不安的、让人联想起反乌托邦科幻小说的进展中，美国海军陆战队正在[评估配备 AI 驱动步枪的机器人狗](https://arstechnica.com/gadgets/2024/05/robot-dogs-armed-with-ai-targeting-rifles-undergo-us-marines-special-ops-evaluation/)，这些机器人狗可以**自动检测并追踪人员、无人机和车辆**。

**其他值得关注的进展**

- **Phi-3 WebGPU AI 聊天机器人完全在浏览器本地运行**：一段[视频演示](https://v.redd.it/72wft36h17zc1)展示了一个**基于 Phi-3 的 AI 聊天机器人，利用 WebGPU 在 Web 浏览器中 100% 本地运行**。

- **IC-Light 实现 AI 驱动的图像重照明**：开源工具 [IC-Light](https://github.com/lllyasviel/IC-Light) 利用 AI **允许对任何图像进行逼真的重照明和光照编辑**。

- **Udio 增加 AI 驱动的音频编辑和修复功能**：[Udio 推出了新功能](https://twitter.com/udiomusic/status/1788243716676759668) ，利用 AI **实现音频中的人声编辑、错误修正和平滑过渡**。

- **研究表明曲速引擎（Warp Drives）可能成为现实**：一项[新的科学研究](https://www.space.com/warp-drive-possibilities-positive-energy)诱人地**暗示在某些条件下，曲速引擎在物理上可能是可行的**。

- **基因工程师通过重组细胞使寿命延长 82%**：/r/ArtificialInteligence 分享了一项研究，[基因工程师通过重组细胞实现了细胞寿命 82% 的增长](https://www.reddit.com/r/ArtificialInteligence/comments/1cmyr9n/turning_back_the_clock_genetic_engineers_rewire/)。

**AI 梗与幽默**

- **AI 炒作周期仍在继续**：/r/ProgrammerHumor 中一个令人深省的[梗图](https://www.reddit.com/gallery/1cnd7ag)提醒我们，**围绕 AI 突破的狂热炒作没有丝毫减弱的迹象**。

---

# AI Discord 回顾

> 摘要之摘要的摘要

1. **Large Language Model (LLM) 进展与基准测试**：
   - 来自 Meta 的 **[Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)** 在 **ChatbotArena** 等排行榜上迅速崛起，在超过 50,000 场对决中表现优于 **GPT-4-Turbo** 和 **Claude 3 Opus**。
   - 像来自 IBM 的 **[Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)** 这样的新模型增强了代码任务的指令遵循能力，而 **[DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)** 则拥有 **236B 参数**。
   - 某些基准测试受到质疑，人们呼吁像 Meta 这样可信的来源建立现实的 LLM 评估标准。

2. **优化 LLM 推理与训练**：
   - **[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)** 承诺在 GPU 上进行大模型训练时，将通信开销降低 4 倍。
   - **[vAttention](https://arxiv.org/abs/2405.04437)** 系统动态管理 KV-cache 内存，在不使用 PagedAttention 的情况下实现高效的 LLM 推理。
   - **[QServe](https://arxiv.org/abs/2405.04532)** 引入了 **W4A8KV4 量化**，以提升 GPU 上基于云的 LLM 服务性能。
   - **[Consistency LLMs](https://hao-ai-lab.github.io/blogs/cllm/)** 等技术探索了并行 Token 解码，以降低推理延迟。

3. **开源 AI 框架与社区努力**：
   - **[Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)** 支持多种数据集格式，用于 LLM 的指令微调和预训练。
   - **[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)** 为 Andrew Ng 的一门关于构建 Agentic RAG 系统的新课程提供支持。
   - **[RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled)** 已开源，声称是处理“枯燥数据任务（unsexy data tasks）”的最佳 LLM。
   - **[Modular](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo)** 展示了 Mojo 在 Python 集成和 AI 扩展（如 _bfloat16_）方面的潜力。

4. **多模态 AI 与生成模型创新**：
   - **[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** 专注于提升聊天交互体验，而 **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** 则精进了编程能力。
   - **[Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** 模型通过 WebGPU 将强大的 AI 聊天机器人带入浏览器。
   - 结合 **Pixart Sigma + SDXL + PAG** 旨在实现 **DALLE-3** 级别的输出，并具有通过微调进一步优化的潜力。
   - 开源项目 **[IC-Light](https://github.com/lllyasviel/IC-Light)** 专注于改进图像重光照（relighting）技术。

**5. 其他**

- **Stable Artisan 将 AI 媒体创作带入 Discord**：Stability AI 推出了 **Stable Artisan**，这是一个集成 **Stable Diffusion 3**、**Stable Video Diffusion** 和 **Stable Image Core** 等模型的 Discord 机器人，用于[直接在 Discord 中生成和编辑媒体内容](https://bit.ly/4aiVy6C)。该机器人引发了关于 **SD3 开源状态**以及将 **Artisan 作为付费 API 服务**引入的讨论。

- **Unsloth AI 社区对新模型和训练技巧反响热烈**：IBM 的 [Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct) 和 RefuelAI 的 [RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled) 的发布引发了架构讨论。用户分享了关于 **Windows 兼容性**的挑战以及对某些**性能基准测试**的质疑，同时也交流了模型训练和微调的技巧。

- **Nous Research AI 的前沿论文与 WorldSim 重启**：分析了关于 LLM 中 **xLSTM** 和 **函数向量（function vectors）** 的突破性论文，同时对 **Llama 3 微调**的最佳实践进行了推测。**WorldSim** 的重新发布及其新功能（如 **WorldClient**、**Root** 和 **MUD**）引起了广泛关注，用户们正在讨论模型合并（model merging）技术的策略。

- **Hugging Face 的编程增强与巨量模型**：[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609) 和 [CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677) 亮相用于聊天和编程，同时拥有 **236B 参数的 DeepSeek-V2** 模型引起了轰动。讨论内容涵盖了 **BERT 微调**、**Whisper 上采样**、**Gemma Token 集成**以及从 PDF 中提取内容的方法。

- **CUDA MODE 的 Triton 教程与 Diffusion 优化深度解析**：一个由 [9 部分组成的博客系列](https://www.vrushankdes.ai/diffusion-inference-optimization) 和 [GitHub 仓库](https://github.com/vdesai2014/inference-optimization-blog-post) 详细介绍了 **Diffusion 模型推理优化**，引起了广泛关注；同时社区合力创建了 [Triton kernels 索引](https://github.com/cuda-mode/triton-index)。LibTorch 的编译时间改进以及用于高效模型训练的 **ZeRO++** 也受到了关注。

- **LangChain 的 Agentic RAG 课程与本地 LLM 突破**：LangChain [与 deeplearning.ai 合作](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex) 推出了关于构建 **Agentic RAG 系统** 的课程，同时还为 Mistral 和 Gemma 等模型引入了 [本地 LLM 支持](https://twitter.com/llama_index/status/1788627219172270370)。用户们排查了 **TypeScript toolkit** 的问题，并对 **Multi-Agent 架构** 展开了讨论。

- **OpenAI 的 Model Spec 为 AI Alignment 定下基调**：OpenAI 发布了 [Model Spec](https://cdn.openai.com/spec/model-spec-2024-05-08.html) 的第一份草案，旨在通过 **RLHF 技术** 引导模型行为，这是其致力于负责任的 AI 开发的一部分。讨论还涉及了 **GraphQL 相比 Markdown 的局限性**，以及 **GPT-4 在不同平台上的能力差异**。

---

# PART 1: Discord 高层级摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Artisan 的创意扩展**：Stability AI 推出了一款名为 *Stable Artisan* 的新 Discord 机器人，使 Discord 用户可以直接在应用内使用 **Stable Diffusion 3**、**Stable Video Diffusion** 和 **Stable Image Core** 等工具进行媒体创建和编辑。

**SD3 的开源争议**：社区讨论中出现了对 **Stable Diffusion 3** 可能不开源的沮丧情绪，对转向专有模型以及发布时间表不明确的担忧在社区中引发了争论。

**Artisan API 引发褒贬不一的反应**：引入 **Artisan**（Stability AI 用于调用 **Stable Diffusion 3** 的付费 API 服务）既获得了期待也遭到了批评，一些用户质疑该服务对于预算有限者的可行性。

**给生成式 AI 新手的指南**：**Stable Diffusion** 生态系统的新手正在交流使用 **ComfyUI** 的技巧，并探索适合不同创意意图的最佳基础模型，借鉴社区仓库和 Prompt 编写技术来精进他们的生成艺术。

**AI 艺术巨头对比**：讨论线程强调了 **Midjourney** 对 AI 艺术工具市场的影响，推测其专业受众以及对 Stability AI 等同类工具货币化策略的潜在影响。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Perplexity 与 SoundHound 联手打造语音 AI**：Perplexity AI 与 [SoundHound](https://www.soundhound.com/newsroom/press-releases/soundhound-ai-and-perplexity-partner-to-bring-online-llms-to-its-next-gen-voice-assistants-across-cars-and-iot-devices/) 达成合作，旨在通过先进的 LLM 能力增强语音助手，承诺在各种 IoT 设备上提供实时回答。

**Claude 3 Opus 额度记录与服务故障**：用户深入探讨了 **Claude 3 Opus** 的 “600 额度” 限制问题，对比了 Perplexity 与直接使用 **Anthropic** 的体验。此外，还有关于 Pro 搜索限制透明度以及账单错误和系统变慢等技术问题的讨论。

**分享功能与搜索在分享中大放异彩**：社区被鼓励将线程设置为 “可分享”，并积极分享关于 *AlphaFold*、双相情感障碍和多语言查询等不同主题的 Perplexity AI 搜索 URL，展示了用户兴趣的多样性。

**无需重采样的 Bootstrapping 成为讨论话题**：一位成员关于在不进行物理重采样的情况下执行 Bootstrapping 的提问引发了技术讨论，重点在于该统计方法中原始数据集的直接应用。

**用户对订阅页面表示关注**：用户对 Pro 订阅页面上潜在的误导信息表示担忧，要求 Perplexity 团队针对 Pro 搜索限制做出明确澄清。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**IBM 代码模型家族的新成员**：[Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct) 由 IBM 发布，该模型在逻辑推理和问题解决方面拥有增强的指令遵循能力，引发了对其不寻常的 GPTBigCodeForCausalLM 架构的讨论。

**Dolphin 在新版本中致谢 Unsloth**：Unsloth AI 在 [Dolphin 2.9.1 Phi-3 Kensho](https://huggingface.co/cognitivecomputations/Dolphin-2.9.1-Phi-3-Kensho-4.5B) 的发布中因其在模型初始阶段的贡献而获得认可。

**AI 爱好者的 Windows 烦恼**：工程师们分享了在 Windows 上部署 AI 模型时的挑战，建议使用 Windows Subsystem for Linux (WSL) 等替代方案，并引用了 [Unsloth GitHub issue](https://github.com/unslothai/unsloth/issues/210) 中概述的解决方案。

**AI 社区质疑模型基准测试**：对某些性能基准测试出现了怀疑，成员们呼吁 Meta 等更可靠的来源为 Large Language Models 建立现实的评估标准。

**调试日记：模型训练的多样化讨论**：关于克服模型训练和开发中各种障碍的积极对话，包括修复 Llama3 训练数据丢失、解决 VSCode 安装错误，以及借助社区分享的 Notebook（如这个用于 [Google Colab](https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing) 仅推理使用的 Notebook）进行模型微调。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**LSTM 发起挑战**：一篇引人入胜的论文强调了扩展至十亿参数规模的 **LSTMs** 的潜力，通过创新的 **exponential gating** 挑战 Transformer 的主导地位。该技术的详细信息请参阅[这项最新研究](https://arxiv.org/abs/2405.04517)。

**AI 的预测水晶球**：**Forefront.ai** 列出了“预期的突破性 AI 论文”，暗示了关键趋势和一种在不显著影响性能的情况下降低计算负载的新型调整技术。该网站展示了对 AI 研究领域的战略前瞻。

**更轻的模型，同样的威力**：显著的讨论透露，**Llama 2 70B** 的 **4-bit 量化、剪裁 40% 的版本**性能与完整模型相当，这表明深度学习模型中存在大规模冗余，正如 [Twitter 帖子](https://x.com/kwindla/status/1788224280754618393)中所述。

**精细化微调**：围绕 **LLaMA 3** 和 **Axolotl** 模型的微调技术展开了讨论，涉及 **context length**、训练期间的预分词（pre-tokenization）与 padding，以及 **Flash Attention 2** 的最佳使用。

**WorldSim 挥舞创新旗帜**：**WorldSim** 展示了新功能，包括改进的漏洞修复、**WorldClient** 浏览体验、CLI 环境 **Root**、祖先模拟和 RPG 特性。社区的热情日益高涨，体现在对购买 Nous Research 宣传周边产品的咨询上，详见其[网站](https://worldsim.nousresearch.com)。

**模型融合的可持续策略**：成员们正积极探索简化模型合并和集成技术的过程，比较 **NeuralHermes 2.5 - Mistral 7B** 等模型中的 Direct Preference Optimization，并探索带有外部权重的 **Llamafile** 的实际益处。

**技术对话的细节**：许多消息展示了迷的问题解决图景，从解决上传模型时出现的 *'int' object has no attribute 'hotkey'* 等错误，到充实限制 RAG 中幻觉的策略以及有效的 padding 策略。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Model Spec 塑造 AI 对话**：OpenAI 推出了 **Model Spec** 框架，用于构建 AI 模型中期望的行为，以丰富相关的公共讨论；全文可访问 [OpenAI 的公告](https://openai.com/index/introducing-the-model-spec)。

**Markdown 优于 GraphQL**：在 AI 讨论中，GraphQL 缺乏客户端渲染的情况与 Markdown 进行了对比，尽管这种限制并未引起重大担忧。

**AI 平台与硬件令人兴奋也令人困惑**：虽然 *OpenDevin* 平台因其 Docker 沙箱和后端模型的灵活性而受到赞誉，但用户发现 *ChatGPT* 各版本与 NVIDIA 技术演示之间的 AI 性能对比非常有趣，然而 GPT-4 ChatGPT 应用版与 API 版本之间的限制引起了社区的不满。

**分享商业中的 AI 伦理提示词**：一位社区成员分享了详细的“商业中的 AI 伦理”提示词结构，旨在增强模型在伦理考量方面的输出，并提供了一个探索不道德行为影响的输出示例，尽管没有提供具体的资源链接。

**寻求 AI 领域的专业知识与远见理想**：一名成员寻求 Prompt Engineering 课程的建议，由于 OpenAI 的政策，后续通过私信进行了交流；而另一名成员思考了“Open”作为核心认识论原则的概念，尽管讨论并未进一步展开。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **针对聊天优化且对编程友好的模型登场**：[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609) 亮相，专注于提升聊天交互体验；同时 [CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677) 出现，旨在优化 Python, Go 和 C# 的编程能力。在浏览器聊天机器人领域，**Phi 3** 引入了 WebGPU 技术以增强聊天体验。

- **突破性的 236B 参数模型与训练里程碑**：拥有 2360 亿参数的 **DeepSeek-V2** 已经发布，标志着模型规模的显著增加；目标检测指南的更新旨在通过在 Trainer API 中添加 mAP 指标来微调性能。

- **技术问题与故障排除**：几位成员遇到了与 **BERT fine-tuning**、**Whisper model upsampling**、集成 **Gemma 的未使用 tokens** 以及扩散模型错误相关的挑战，凸显了社区内问题的多样性。

- **语音频道中的音频与互动调整**：关于利用“Stage”频道控制语音组音频质量的讨论，揭示了质量控制与参与者急于提问之间的平衡，从而导致了对 Stage 频道设置的实验。

- **致力于更高效的内容提取**：一位用户分享了从 PDF 中提取图像和图表的尝试，并希望有更高效的 AI 方法来处理此类数据；同时，一个新模型出现，使用 [Adlike](https://github.com/chitradrishti/adlike) 来标记图像中的广告。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 的 Python 前景与性能辩论**

- **Mojo 的年终 Python 潜力**：Modular 社区预期 **Mojo** 在年底前会变得更加用户友好，CPython 集成已经实现了 Python 代码执行。然而，直接编译 Python 代码可能仍是明年的目标。

- **MLIR 在 Mojo 中的高光时刻**：对 MLIR 扩展贡献的热情高涨，特别是在活跃度检查（liveliness checking）领域；成员们热切期待 Modular 的 MLIR dialects 可能开源，以增强 MLIR 的实用性。

**编译见解与 Twitter 和博客上的热度提升**

- **Modular Twitter 上的功能狂热**：Modular 预告了重要更新，承诺进行功能升级，并在 Twitter 上披露了稳健的增长指标，引发了社区对即将发布的公告的好奇。

- **Chris Lattner 力挺 Mojo**：在 Developer Voices 播客中，Chris Lattner 强调了 Mojo 对 Python 和非 Python 开发者的前景，重点关注扩展的 GPU 性能和 AI 相关扩展。

**社区代码贡献与编译器对话**

- **Toybox 仓库加入 GitHub**：社区驱动的仓库 "toybox" 展示了 **DisjointSet** 和 Kruskal 算法的实现，邀请通过 Pull Requests 进行协作增强。

- **Mojo 中的字符串策略引发辩论**：围绕 Mojo 的字符串拼接出现了性能担忧；回应包括提议的短字符串优化（short string optimization）以及在 GitHub 讨论中展示的使用 `KeysContainer` 和 `StringBuilder` 进行加速的精炼技术。

**Mojo Nightly 中的 Tensor 纠葛与标准库更新**

- **探索 Mojo Nightly 的细微差别**：Mojo nightly 版本的更新包括对 Tensor API 的修订以提高清晰度，同时社区正在应对围绕 `DTypePointer` 行为的复杂问题，并热切审阅新的编译器更新。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **模型存储目录深度探索**：工程师们讨论了 LM Studio 理想的 **model directory structures**，与 Hugging Face 的命名规范保持一致，以促进模型的组织和发现。例如，一个 **Meta-Llama model** 应该放置在路径 `\models\lmstudio-community\Meta-Llama-3-8B-Instruct-GGUF\`。

- **新的 120B 模型合并为编码者和诗人助力**：Maxime Labonne 发布了一个 **120B self-merged model**，[Meta-Llama-3-120B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Meta-Llama-3-120B-Instruct-GGUF)，承诺为用户提供增强的性能以进行测试和反馈，而一些成员则表达了对当前模型在诗歌补全性能上的挣扎。

- **AI 需要强大的硬件**：社区广泛讨论了运行 **Llama 3 70B** 等大型模型所需的硬件能力，开玩笑说 **400B model** 需要理论上的 **200GB VRAM**，并就如何优化资源制定策略，例如将桌面任务卸载到 **Intel HD 600 series GPU**。

- **RAG 架构可能是搜索英雄**：一位成员建议利用具有分块文档处理能力的 **RAG architectures** 来改进数据搜索，并建议基于相似度度量的重排序（reranking）方法可以优化结果，展示了在操作型 AI 效率方面的又一进步。

- **API 增强功能的候车室**：工程师们正热切期待更新，包括通过 **LM Studio API** 以编程方式与现有聊天进行交互的可能性，突显了该平台向更具适应性和用户友好的 AI 工具的持续演进。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **发布礼仪引发非官方讨论**：工程师们讨论了发布算法**非官方实现**的细微差别，强调了此类项目应明确标记以区别于官方版本的伦理立场，如 MeshGPT [GitHub repo](https://github.com/nihalsid/mesh-gpt) 所示。

- **科学出版中的更名博弈**：在学术数据库中更新婚后姓氏的复杂性引发了对话，主要建议围绕联系各平台的相应支持，并考虑保留“nom-de-academia”。

- **高效处理 The Pile 数据**：交流了为训练 AI 模型处理 **The Pile** 数据的最佳实践，重点是利用 Hugging Face 等资源上的预处理数据集，以及在处理 `.bin` 文件时如何操作特定的 tokenizer。

- **扩展状态跟踪的极限**：关于模型中状态跟踪可扩展性的辩论浮出水面，起因是一篇讨论状态空间模型（state-space models）和 transformers 共同局限性的文章，以及围绕 **xLSTM** 能力的推测性讨论。

- **对 xLSTM 的怀疑**：成员们审视了 xLSTM 论文，认为其在基准测试中可能使用了次优的超参数（hyperparameters），从而对所提出的主张表示怀疑，并期待独立验证或官方代码的发布。

- **函数向量（FV）找到其功能**：关于高效 in-context learning 的**函数向量 (FV)** 的讨论非常引人入胜，这些讨论基于研究见解，表明 FV 可以在极小的 context 下实现稳健的任务性能（[研究来源](https://arxiv.org/abs/2403.00835)）。

- **YOCO 产生单一缓存好奇**：YOCO 的引入（一种简化 KV caches 的 decoder-decoder 架构）让一些人思考其进一步优化的潜力，以及单一缓存轮次在改善内存经济性方面的优势。

- **显微镜下的多语言 LLMs**：为了理解 LLMs 如何处理多语言输入，工程师们参考了关于语言特定神经元和 LLMs 计算 next tokens 能力的研究，这可能涉及内部翻译为英语（[研究示例](https://arxiv.org/abs/2402.10588)）。

- **位置编码获得正交优化**：PoPE 的出现带来了基于正交多项式的位置编码，可能优于基于正弦的 APEs，这引发了讨论，在承认其潜力的同时也批评其方法过于理论化（[论文](https://arxiv.org/abs/2405.04585)）。

- **Tuned Lenses 咨询**：可解释性频道中的一个提问暗示了分析 AI 模型工具的针对性，具体询问是否每个 Pythia checkpoint 都有 tuned lenses，表明了对模型可解释性和微调细微差别的兴趣。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **LibTorch 编译时间的奥秘**：利用 ATen 的 `at::native::randn` 并仅包含必要的头文件（如 `<ATen/ops/randn_native.h>`），将编译时间从约 35 秒缩短至仅 4.33 秒。
  
- **聚焦效率 (ZeRO-ing)**：[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/) 声称可将大型模型训练的通信开销降低 4 倍，从而在 NVIDIA 的 A100 GPUs 上实现 batch size 为 4 的训练——它甚至可以在多个 A100 上运行，尽管性能会略有下降。

- **极致调优扩散模型**：发布了一个 [9 部分的博客系列](https://www.vrushankdes.ai/diffusion-inference-optimization)，展示了 GPU 上 U-Net 扩散模型的优化策略，并在随附的 [GitHub repository](https://github.com/vdesai2014/inference-optimization-blog-post) 中提供了实际示例。

- **vAttention 攀登内存管理高峰**：提出了一种名为 **vAttention** 的新系统，用于大语言模型推理中的动态 KV-cache 内存管理，旨在解决静态分配的局限性（[论文摘要](https://arxiv.org/abs/2405.04437)）。

- **苹果 M4 展示计算肌肉**：苹果的 [M4 芯片](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/) 拥有令人印象深刻的 *每秒 38 万亿次操作*，这得益于 3nm 技术和 10 核 CPU，预示着移动计算能力的进步速度。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **GPT-4 的巧妙组合**：**GPT-4** 展示了其利用 YouTube 提供定制音乐建议的能力，并能遵循特定的用户指令；然而，一些本地模型（如 TheBloke/deepseek-coder-33B-instruct.GGUF 和 lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF）表现不佳，出现了声称*没有互联网连接*或在尝试失败后放弃等问题。
- **mixtral-8x7b 在本地测试中胜过竞争对手**：在测试的几个本地模型中，mixtral-8x7b-instruct-v0.1.Q5_0.gguf 脱颖而出，特别是在配备经过自定义调优的 **2080 ti GPU** 和 **32GB DDR5 6000** 内存的硬件上，证明比其他替代方案更有效。
- **MacBook 的困境**：**MacBook Pro** 系统被发现不足以运行即使是轻量级的本地模型操作，导致一名成员停止使用他们的 MacBook 执行这些任务。
- **跨平台提供商集成**：LiteLLM 的文档受到关注，显示其支持多种 AI 提供商，包括 OpenAI 模型、Azure、Google 的 PaLM 和 Anthropic，详细指令可在 [LiteLLM providers documentation](https://litellm.vercel.app/docs/providers) 中找到。
- **Windows 上的 01 平台不稳定**：一些用户报告 **01 platform** 在 **Windows 操作系统**上功能有限，讨论集中在针对 Windows 10 的可能代码修改以及 Windows 11 的兼容性检查。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Dalle-3 的竞争对手出现**：结合 **Pixart Sigma**、**SDXL** 和 **PAG** 引发了关于实现 **DALLE-3** 级别输出的讨论，目前已识别出在文本和单个对象渲染方面的局限性。参与者认为微调可以增强图像构图，并强调了熟练技术干预的必要性。
- **追求更好的模型性能**：分享了一个提升**模型质量**的突破，详细说明了微调节（microconditioning）输入的易管理范围有助于更平滑的学习，完善这一方法的前景引起了关注。
- **为 AI 场景重新布光**：GitHub 上的 **IC-Light** 项目旨在改进图像重新布光（relighting），正受到社区关注，可通过 [IC-Light GitHub](https://github.com/lllyasviel/IC-Light) 访问。
- **对扩散模型的见解**：关于**噪声条件评分网络（Noise Conditional Score Networks）**和**扩散模型（diffusion models）**的激烈辩论涉及了噪声调度和分布收敛等问题，同时讨论了如 'K-diffusion' 等论文的数学复杂性和概念差异。
- **保险领域的效率提升**：提出了关于**自动化数据处理的开源工具**在商业汽车保险中应用的建议请求，展示了 AI 在增强保险行业风险评估方面的跨领域应用。
- **AI 研究的扩展**：一项包含潜在发表机会和 *IJCAI journal* 奖励的 **AI 竞赛公告** 引起了关注，标志着 AI 社区对学术认可的积极追求。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **深入探讨 Agentic RAG**：[deeplearning.ai](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex) 推出了一门由 LlamaIndex CEO Jerry Liu 主讲的课程，重点是创建能够进行复杂推理和基于检索的问答的 **agentic RAG** 系统。AI 先驱 Andrew Ng 在 [Twitter 公告](https://x.com/AndrewYNg/status/1788246239517282795)中指出该课程具有重要意义。
- **本地 LLM 加速**：LlamaIndex 发布了一个更高效地**在本地执行大语言模型（LLMs）**的集成方案，支持 Mistral 和 Gemma 等模型，并提供对 NVIDIA 硬件的兼容性，详见[其 Twitter 帖子](https://twitter.com/llama_index/status/1788627219172270370)。
- **解决 LlamaIndex 集成问题**：关于正确使用 **omatic embeddings** 和 **LlamaIndex** 向量存储集成的讨论提供了嵌入兼容性问题的解决方案。强调了向量存储可以轻松管理嵌入，减轻了工程师在实现上的烦恼。
- **关注运维可扩展性**：工程师们辩论了托管本地 LLM 模型的最佳方案，指向了 **AWS** 和 **Kubernetes 上的自动扩缩容**等解决方案，并对实现大规模部署的可扩展性表现出兴趣。
- **编程诊所**：针对一位在代码中遇到 `_CBEventType.SUB_QUESTION` 未触发问题的用户，同行提供了针对性的指导和代码片段，以定位并纠正实现缺陷。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**使用 Languify.ai 提升你的网页表现**：新的浏览器扩展 **Languify.ai** 旨在优化网站文本，以获得更好的用户参与度和销售额，它利用 OpenRouter 根据 Prompt 进行模型选择。专业版售价为每月 10.99 欧元，为寻求流线型工具的用户提供了 AnythingLLM 之外的一个可行替代方案，详情见 [Languify.ai](https://www.languify.ai/)。

**OpenRouter 之谜部分解开**：用户之间正在进行的讨论显示，大家希望获得更多关于 **OpenRouter** 的易获取信息，关键话题包括 API 文档、额度系统理解，以及某些缺乏全面解答的免费 AI 模型状态。

**按需定制的 Moderation 模块**：对基于 **Llama 3** 的 Moderation（审核）服务感兴趣的用户被引导至 Together.ai，因为 OpenRouter 本身目前尚未列出此类功能。

**`min_p` 获得认可**：Together、Lepton、Lynn 和 Mancer 等供应商因其模型支持 `min_p` 参数而受到关注，尽管注意到 Together 存在一些问题，而 Lepton 表现正常。

**打破 Wizard 8x22B 的束缚**：围绕“越狱” **Wizard 8x22B** 以访问受限较少内容的讨论激增，社区成员分享了诸如 [Refusal in LLMs is mediated by a single direction](https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) 等资源，以理解语言模型固有的限制和拒绝机制。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **TypeScript 工具包问题**：工程师们正在解决 TypeScript 中 `JsonOutputFunctionsParser` 抛出 *Unterminated string in JSON* 错误的问题；建议包括确保 `get_chunks` 函数返回正确的 JSON，为 `chain.invoke()` 提供格式正确的 `content`，以及对 `getChunksSchema` 进行彻底检查。

- **引导对话一致性**：当字典输入与在 Python 中运行（以空字典开始）表现不同时，**LangChain AI** 的 `/invoke` 端点出现差异，这促使社区分享他们的诊断结果。

- **解析多 Agent 机制**：用户对类似于 **STORM** 多 Agent 文章生成器的项目表示了兴趣，讨论围绕组件化架构与为不同功能实例化独立 Agent 的有效性展开。

- **Vector DB 的权衡**：对话转向了设置 **VertexAI Vector store** 的成本和复杂性，Pinecone 和 Supabase 作为可能对钱包负担较小的替代方案被列入考虑范围。

- **Gianna 抢占 AI 聚光灯**：模块化构建的虚拟助手 **Gianna** 首次亮相并引发对话，它拥有与 **CrewAI** 和 **LangChain** 的集成，可在 [GitHub](https://github.com/marvinbraga/gianna) 或通过 PyPI 获取，并配有 [教程视频](https://www.youtube.com/watch?v=CXmwYk5Hbig)。同时，一篇 [新的 Medium 文章](https://ai.gopubby.com/streamline-customer-support-with-langchains-langgraph-8721c250809e) 揭示了 **LangChain** 的 **LangGraph** 是客户支持的游戏规则改变者，而 AI 数据平台 **Athena** 展示了其完整的数据工作流自主性。

- **CrewAI 连接加密货币**：分享了一个引人入胜的教程视频 [“创建一个自定义工具将 crewAI 连接到 Binance 加密货币市场”](https://youtu.be/tqcm8qByMp8)，为使用 **crewAI CLI** 和 **Binance.com** 加密货币市场进行金融分析开启了新的可能性。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **斯坦福为 AI 思想助力**：斯坦福大学发布了其 2023 年新课程“深度生成模型 (Deep Generative Models)”，讲座可在 [YouTube](https://youtu.be/XZ0PMRWXBEU) 上观看，供寻求提升技能的 AI 专业人士学习。
- **GPU 探寻**：Discord 上的 AI 工程师们正在交流获取 A100/H100 GPU 的技巧，推荐 [sfcompute](https://sfcompute.com) 作为一个可靠的来源。
- **程序员关于 AI 工具的交锋**：一场关于 AI 辅助编程的热烈辩论正在进行，人们怀念 Lisp 的旧时光，并审视当前 AI 代码助手的优缺点。
- **Gradient 的超大上下文飞跃**：来自 Gradient 的新 Llama-3 8B Instruct 模型因其上下文长度大幅增加至 4194k 而引起轰动；感兴趣的各方可以 [在此注册](https://forms.gle/L6TDY7dozx8TuoUv7) 以获取定制 Agent。
- **OpenAI 的安全举措引发讨论**：OpenAI 关于创建安全 AI 基础设施的博客文章引发了成员间的讨论，一些人将这些措施解释为一种“保护主义”。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Llama-3-Refueled 模型现已公开**：**RefuelAI** 发布了 **RefuelLLM-2**，这是一个被誉为能高效处理“乏味数据任务”的语言模型，权重可在 [Hugging Face](https://huggingface.co/refuelai/Llama-3-Refueled) 上获取。正如 [Twitter 公告](https://twitter.com/BansalDhruva/status/1788251464307187980) 中所强调的，该模型在 **2750 多个数据集**上进行了约一周的指令微调（instruction tuned）。

- **Axolotl 数据集困惑已消除**：关于 Axolotl 支持的数据集格式文档包括 [JSONL 和 HuggingFace 数据集](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)，为预训练和指令微调等各种任务的数据组织提供了清晰的指导。

- **Axolotl 用户应对 GPU 难题**：一名工程师正在寻找适用于 8 张 A100 GPU 进行 4K/128K FFT 的 phi3 mini 配置文件，而另一名用户报告了 [8x H100 GPU 上的训练问题](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1596)，反映了 AI 模型训练环境中关于优化和故障排除的持续讨论。

- **W&B 环境变量集成**：社区提到了 [W&B 文档](https://docs.wandb.ai/guides/track/environment-variables) 中关于使用环境变量的指南，表明了对实验追踪和可复现性的关注。

- **LoRA YAML 设置引发社区讨论**：讨论了为 AI 模型配置 **LoRA** (Low-Rank Adaptation) 的几个复杂细节，包括在添加新 token 时保存参数的正确 YAML 配置。遇到的错误导致了关于更新 `lora_modules_to_save` 以包含 `'embed_tokens'` 和 `'lm_head'` 的建议，成员们分享了关于 [代码故障排除过程](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=848dbbe2-5271-421e-89db-e9f0639a7415) 的见解。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**没有目的的精美图片？**：图表的美学因其“非常漂亮的图片”而受到赞赏，但缺乏功能性评论。讨论强调了对图表选择参数量而非 FLOPs，以及在没有进行适当 **hyperparameter tuning** 的情况下为 Transformer 基准使用非标准学习率的担忧。

**技术驱动的增长策略**：关于在 **TPU 上训练强化模型 (RM)** 以及使用 **Fully Sharded Data Parallel (FSDP)** 的咨询，表明对优化和扩展策略的探索激增。同时，**EasyLM** 成为使用 Jax 进行 RM 训练的潜在基础，GitHub 脚本示例：[EasyLM - llama_train_rm.py](https://github.com/hamishivi/EasyLM/blob/main/EasyLM/models/llama/llama_train_rm.py)。

**排行榜逻辑与研究共鸣**：关于 **5k 排行榜** 是否能充分反映 AI 模型性能的辩论随之而来，并有建议扩展到 **10k**。尽管存在被忽视的续作和有争议的排行榜比例，但对 **Prometheus** 的赞誉不断，将其置于典型 AI 研究之上。

**SnailBot 的慢动作亮相**：社区期待 **SnailBot** 的首次亮相，对 *tick tock* 的玩笑表达了兴奋但也有些不耐烦，并在收到机器人的回复后进行了轻松的互动。

**LLM 许可困境**：ChatbotArena 中出现了与发布大语言模型生成的文本相关的许可复杂性担忧，暗示需要供应商的专门许可。

**前沿讨论**：OpenAI 发布了用于 AI 对齐的 Model Spec，强调了 **RLHF** 技术，并为 OpenAI API 和 ChatGPT 中的模型行为设定了标准。此外，**Llama 3** 在 ChatbotArena 中一路领先，在 50,000 多场对决中超越了 **GPT-4-Turbo** 和 **Claude 3 Opus**，相关见解在博文中进行了剖析，可在此处探索：[Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 中的 BITCAST 大讨论**：[tinygrad 的 Pull Request #3747](https://github.com/tinygrad/tinygrad/pull/3747) 引发了辩论，区分了 **CAST** 与 **BITCAST** 操作，并主张在实现中保持清晰，建议精简 **BITCAST** 以防止参数混乱。
  
- **符号化困惑已清除**：用户就为 tinygrad 增强 `arange`、`DivNode` 和 `ModNode` 函数的符号化（symbolic）版本交换了意见。虽然提到了对下游影响的担忧，但尚未就具体方法达成共识。

- **推进 `arange` 功能时的挫折**：实现符号化 `arange` 的努力遇到了障碍，正如一位用户通过 [GitHub pull request](https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:symbolic-arange-pull) 分享的尝试所证明的那样，进一步的计划将转向“bitcast 重构”。

- **矩阵炼金术**：讨论了一种用于拼接矩阵操作输出的新颖设计，思考了原地（in-place）写入现有矩阵以减少开销的可能性。同时，一名成员寻求有关 Metal 构建过程的帮助，特别是难以找到的 `libraryDataContents()`。

- **可视化这个！**：为了帮助理解 tinygrad 中的 shape 和 stride 概念，一位用户创建了一个 [可视化工具](https://mesozoic-egg.github.io/shape-stride-visualizer/)，协助工程师直观地探索不同的组合。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 上的 RAG 限制**：在 **Cohere.command** 上实现 **RAG** 对用户构成了挑战，原因是 4096 token 的限制；一些人建议使用 **Elasticsearch** 来减少文本大小，而另一些人则考虑将文本分割成段落以有效管理信息丢失。

- **文件生成咨询**：社区正在探索让 **Cohere Chat** 生成 **DOCX 或 PDF** 排版的方法，但文件下载的机制仍不明确。

- **解决 Cohere CORS 令人头疼的问题**：成员们讨论了在使用 **Cohere API** 时遇到的 **CORS** 问题，建议通过后端调用来避免安全事故并保持 API key 的机密性。

- **了解 Cohere 的积分系统**：关于在 **Cohere** 上充值积分的查询得到了澄清，该平台不提供预付费选项；相反，用户可以通过仪表板上的 **billing limits**（计费限制）来控制支出。

- **Wordware 招聘热潮**：**Wordware** 正在寻找 AI 人才，包括创始工程师和 DevRel 职位，鼓励申请者使用 Wordware 的 IDE 展示技能，并与 wordware.ai 的团队互动。[查看职位详情](https://wordware.notion.site/Join-Wordware-YC-S24-347a2b89acad44c1bc99591636308ec2)。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**现在用 API，少写代码**：**Meta-Llama-3-8B-Instruct** 可以通过本地主机的 **API endpoint** 运行，支持 OpenAI 风格的交互。详细信息和设置说明可在 [项目的 GitHub 页面](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#json-api-quickstart) 找到。

**切换模型变得更简单**：**Visual Studio Code** 用户迎来了一个下拉菜单功能，简化了使用 *ollama* 的用户在不同模型之间的切换。

**对 Llamafile 更新效率的请求**：用户向 **llamafile** 提出了一个功能请求，希望能够更新二进制脚手架（binary scaffold）而无需重新下载整个文件的繁琐过程，这被视为提升效率的潜在增强功能。

**关于 Mozilla-Ocho 的趣谈**：出现了一段有趣的对话，讨论 **Mozilla-Ocho** 是否暗指《躲避球》（Dodgeball）中的 "ESPN 8 - The Ocho"，尽管这看起来更像是一个有趣的插曲而非紧迫的问题。

**供好奇的读者参考**：讨论中引用的唯一链接：[GitHub - Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#json-api-quickstart)。



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**AI 辅助 Excel 高手**：工程师们正在探索如何利用 **LLM** 进行电子表格数据操作，特别强调了 AI 筛选和提取信息的能力。

**Yawn.xyz 雄心勃勃的 AI 电子表格演示**：尽管 **[Yawn.xyz](https://x.com/yawnxyz/status/1786131427676852338?s=46&t=4-kZga74dpKGeI-p2P7Zow)** 雄心勃勃地尝试解决生物实验室中的电子表格提取挑战，但社区对其 AI 工具演示的反馈表明存在性能问题。

**寻求顺畅的 GPT-4-turbo Azure 部署**：一位工程师在 **瑞典 Azure 区域** 使用 **GPT-4-turbo** 时遇到了问题，引发了关于最佳 Azure 部署区域的讨论。

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **AlphaFold3 进入开源领域**：一个基于 PyTorch 的 **AlphaFold3** 实现已发布，旨在高精度预测生物分子相互作用，并支持原子坐标。代码已在 [GitHub](https://buff.ly/3JQVKze) 上发布，供社区审查和贡献。

- **Agora 的 AlphaFold3 协作链接连接失败**：有人呼吁加入 Agora 共同推进 AlphaFold3，但提供的链接据报失效，阻碍了协作前景。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **AI 工程师们，快来看看**：*[Skunkworks AI](https://discord.com/channels/1131084849432768614/1140423597454807179/)* 频道中发布了一个简短的帖子，分享了一个 [YouTube 视频链接](https://www.youtube.com/watch?v=4MzCpZLEQJs)，虽然没有提供上下文，但对于热衷于 AI 发展相关多媒体内容的科技爱好者来说，可能非常吸引人。



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**Quickscope 表现出色**：Regression Games 隆重推出 [**Quickscope**](https://play.regression.gg)，这是一套全新的 AI 驱动工具包，可自动执行 Unity 游戏测试，具有**游戏会话录制（Gameplay Session recording）**和**验证工具**，实现简化的无代码设置。

**深入探索游戏测试自动化**：Quickscope 的深度属性抓取功能可从游戏对象层级中提取详细数据，从而在无需编写自定义代码的情况下，深入了解位置和旋转等游戏实体信息。

**面向 QA 团队的测试平台**：Quickscope 拥有一个支持高级测试自动化策略（如**智能回放系统**）的平台，专为 **QA 团队**设计，以促进快速、直接的集成。

**交互式 UI 助力游戏测试**：该平台的[直观 UI](https://regression.gg/) 使 QA 工程师和游戏开发人员能够更轻松地定义测试，并与 Unity 编辑器、构建版本兼容，或可以织入 CI/CD 流水线中。

**体验 Quickscope**：鼓励工程师和开发人员试用 Quickscope 的 AI 工具套件，亲身体验它为游戏测试自动化带来的高效与简洁。





---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **命令行工具广受好评**：AI 工程师们表达了对 `llm` 命令行界面工具的赞赏；它被比作一个能够处理“更多 Unix 风格任务”的个人项目助手。



---


**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道划分的详细摘要和链接



**Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1238193305129451622)** (1 条消息): 

- **Stable Artisan 加入聊天**：推出了机器人 *Stable Artisan*，允许 **Stable Diffusion Discord 服务器**成员使用 [Stability AI 的模型](https://bit.ly/4aiVy6C)创建图像和视频，包括 **Stable Diffusion 3**、**Stable Video Diffusion** 和 **Stable Image Core**。
- **全新的多模态机器人体验**：**Stable Artisan** 是一款多模态生成式 AI Discord 机器人，旨在直接在 Discord 内集成媒体生成的多个方面，增强用户参与度。
- **除生成之外——还包含编辑工具**：该机器人通过内容编辑工具扩展了其功能，提供*搜索与替换（Search and Replace）*、*移除背景（Remove Background）*、*创意放大（Creative Upscale）*和*扩图（Outpainting）*等功能。
- **人人享有的易用性**：通过利用 [Developer Platform API](https://platform.stability.ai/docs/getting-started) 的能力，**Stable Artisan** 让最先进的生成式 AI 对 Discord 用户而言更加触手可及。
- **立即开启生成之旅**：鼓励 Discord 用户访问 **Stable Diffusion Discord** 中专门针对不同机器人命令和功能的频道，开始使用 **Stable Artisan**。

**提到的链接**：<a href="https://bit.ly/4aiVy6C">Stable Artisan：在 Discord 上进行媒体生成与编辑 — Stability AI</a>：Stable Diffusion 社区最常见的请求之一是能够直接在 Discord 上使用我们的模型。今天，我们很高兴推出 Stable Artisan，这是一个用户友好的机器人...

  

---


**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1237670655785173072)** (811 条消息🔥🔥🔥):

- **本地与云端使用的辩论**：社区讨论了在本地使用 Stable Diffusion 与求助于云端 GPU 执行任务的优劣，在便利性、成本和性能方面意见不一。在本地生成高质量图像方面，有人提到使用**搭载 M2 芯片的 iPad Pro**进行创意工作，而另一些人则强调云服务在不需要昂贵硬件的情况下进行重度训练的优势。

- **SD3 与开源担忧**：成员们对 **Stable Diffusion Version 3 (SD3)** 可能不作为开源发布的预期表示沮丧，引发了对发布日期承诺以及可能转向付费模式的担忧。

- **Artisan 作为付费服务**：Stability AI 推出了 **Artisan**，这是一个用于利用 SD3 的付费 API 服务，社区对此反应不一，批评集中在定价以及在当前经济环境下对爱好者和专业人士的实用性上。

- **新用户的工作流讨论**：初学者寻求关于在不同 **Stable Diffusion** 模型中使用 **ComfyUI** 的建议，寻找针对各种图像类型的最佳基础模型和 VAEs 的指导，以及有效的提示词指南。

- **Midjourney 的市场影响**：成员们反思了 **Midjourney** 作为 Stable Diffusion 竞争对手的影响，其商业模式吸引了特定的专业受众，这可能为 AI 艺术工具的变现提供不同的方向。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.tomshardware.com/pc-components/gpus/stable-diffusion-benchmarks">Stable Diffusion Benchmarks: 45 Nvidia, AMD, and Intel GPUs Compared</a>: Stable Diffusion 基准测试：45 款 Nvidia、AMD 和 Intel GPU 对比：哪款显卡提供最快的 AI 性能？</li><li><a href="https://creations.mtdv.me/sd3">Stable Diffusion 3 is available now!</a>: 备受期待的 Stable Diffusion 3 (SD3) 终于正式发布了</li><li><a href="https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820">‎Draw Things: AI Generation</a>: ‎Draw Things 是一款 AI 辅助图像生成工具，可帮助你在几分钟内而非几天内创建心目中的图像。掌握“咒语”，你的 Mac 就能通过简单的几步画出你想要的内容...</li><li><a href="https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler">GitHub - Extraltodeus/sigmas_tools_and_the_golden_scheduler: A few nodes to mix sigmas and a custom scheduler that uses phi</a>: 几个用于混合 sigmas 的节点以及一个使用 phi 的自定义调度器 - Extraltodeus/sigmas_tools_and_the_golden_scheduler</li><li><a href="https://civitai.com/models/257749/pony-diffusion-v6-xl">Pony Diffusion V6 XL - V6 (start with this one) | Stable Diffusion Checkpoint | Civitai</a>: Pony Diffusion V6 是一款多功能的 SDXL 微调模型，能够生成各种兽人、野兽或类人种族的惊人 SFW 和 NSFW 视觉效果...</li><li><a href="https://huggingface.co/deadman44/SDXL_Photoreal_Merged_Models#potest2">deadman44/SDXL_Photoreal_Merged_Models · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/Clybius/ComfyUI-Extra-Samplers">GitHub - Clybius/ComfyUI-Extra-Samplers: A repository of extra samplers, usable within ComfyUI for most nodes.</a>: 一个额外采样器仓库，可用于 ComfyUI 中的大多数节点。- Clybius/ComfyUI-Extra-Samplers</li><li><a href="https://civitai.com/models/193225/sprite-art-from-jump-superstars-and-jump-ultimate-stars-or-pixelart-ai-model">Sprite Art from Jump superstars and Jump Ultimate stars | PixelArt AI Model - v2.0 | Stable Diffusion LoRA | Civitai</a>: 来自 Jump Superstars 和 Jump Ultimate Stars 的像素艺术 - PixelArt AI 模型。如果你喜欢这个模型，请点个赞 ❤️。这个 LoRA 模型是在像素图上训练的...</li><li><a href="https://github.com/11cafe/comfyui-workspace-manager">GitHub - 11cafe/comfyui-workspace-manager: A ComfyUI workflows and models management extension to organize and manage all your workflows, models in one place. Seamlessly switch between workflows, as well as import, export workflows, reuse subworkflows, install models, browse your models in a single workspace</a>: 一个 ComfyUI 工作流和模型管理扩展，可在一个地方组织和管理你所有的工作流和模型。无缝切换工作流，以及导入、导出工作流，重用子工作流，安装模型，并在单个工作区中浏览你的模型。</li><li><a href="https://huggingface.co/stabilityai">stabilityai (Stability AI)</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=LAQYZWbmkwA&t=2s">Hyper-SD - Better than SD Turbo &amp; LCM?</a>: 新的 Hyper-SD 模型是免费的，并且有三个 ComfyUI 工作流可以体验！使用惊人的 1 步 unet，或者通过使用 LoRA 来加速现有模型...</li><li><a href="https://github.com/PixArt-alpha/PixArt-sigma">GitHub - PixArt-alpha/PixArt-sigma: PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation</a>: PixArt-Σ：用于 4K 文本生成图像的 Diffusion Transformer 从弱到强训练 - PixArt-alpha/PixArt-sigma</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/2334">Can I run it on cpu mode only?
 · Issue #2334 · AUTOMATIC1111/stable-diffusion-webui</a>: 我可以只在 CPU 模式下运行它吗？如果可以，你能告诉我怎么做吗？</li><li><a href="https://youtu.be/4tJuQtw8EXQ?si=_XIY9Wi5tBot6T67">The new iPads are WEIRDER than ever</a>: 查看倍思 (Baseus) 60w 伸缩式 USB-C 线缆 黑色：https://amzn.to/3JlVBnh, 白色：https://amzn.to/3w3HqQw, 紫色：https://amzn.to/3UmWSkk, 蓝色：https:/...</li><li><a href="https://tuguoba.com/faceswap_payment">AI Face Swap Desktop Application</a>: 你可以在桌面上获得高质量的 AI 换脸效果</li><li><a href="https://www.youtube.com/watch?v=h6gBFvbNZgE">DaVinci Resolve iPad Tutorial - How To Edit Video On iPad!</a>: 完整的 DaVinci Resolve iPad 视频编辑教程！这里详细介绍了如何使用 iPad 版 DaVinci Resolve，以及为什么它是最好的视频编辑应用之一...</li><li><a href="https://youtu.be/WPKPO-2WFK8?si=CjWbWYUaezwqaYN6">SDXL Lora Training with CivitAI, walkthrough</a>: 我们研究过几种不同的训练 Stable Diffusion LoRA 的方法，不过现在有很多优秀的自动训练器。CivitAI 提供了一种简单的方法来训练...</li><li><a href="https://www.youtube.com/@CutsceneArtist/search?query=draw%20things">Cutscene Artist</a>: 实时动画、3D 人物创建和生成式叙事。http://CutsceneArtist.com
</li>
</ul>

</div>
  

---

**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1238185068657377441)** (1 条消息): 

- **Perplexity 与 SoundHound 达成合作**：[Perplexity 与 SoundHound 联手](https://www.soundhound.com/newsroom/press-releases/soundhound-ai-and-perplexity-partner-to-bring-online-llms-to-its-next-gen-voice-assistants-across-cars-and-iot-devices/)，后者是语音 AI 领域的领导者，旨在将其在线 LLM 能力集成到语音助手中。此次合作旨在为汽车、电视和其他 IoT 设备中的语音查询提供即时、准确的答案。
- **语音 AI 邂逅实时网页搜索**：借助 Perplexity 的 LLM，SoundHound 的语音 AI 将利用实时网页知识以对话方式回答问题。这项创新被誉为市场上最先进的语音助手。

**提到的链接**：<a href="https://www.soundhound.com/newsroom/press-releases/soundhound-ai-and-perplexity-partner-to-bring-online-llms-to-its-next-gen-voice-assistants-across-cars-and-iot-devices/">SoundHound AI and Perplexity Partner to Bring Online LLms to Next Gen Voice Assistants Across Cars and IoT Devices</a>：这标志着生成式 AI 的新篇章，证明了这项强大的技术在没有云连接的情况下仍能提供最佳结果。SoundHound 与 NVIDIA 的合作将使其能够...

---

**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1237672828174602283)** (464 条消息🔥🔥🔥): 

- **寻找适用于 Perplexity 的 TTS**：一名成员询问了在 Perplexity AI 上实现文本转语音（TTS）功能的情况，表示有兴趣使用网页扩展或相关设置来实现此功能。
- **剖析 Claude 3 Opus 的限制**：多项讨论围绕 **Claude 3 Opus** 及其在 Perplexity 上使用时的限制展开，特别是针对“600 额度”限制、额度随时间的变化，以及通过混合使用模型来充分利用可用额度的策略。
- **支付困惑与订阅担忧**：用户对订阅页面上 Pro Search 限制信息不透明、存在误导性表示担忧，并要求 Perplexity 团队做出澄清。
- **比较 AI 服务和模型**：用户讨论了 Perplexity 上的 **Claude 3 Opus** 与 **Anthropic** 直接提供的版本之间的体验差异，指出响应质量存在波动，并建议在其他平台上进行测试验证。
- **账单问题与技术延迟**：有人提到了账单问题并寻求帮助，同时观察到 **Perplexity 的 AI 服务** 出现运行缓慢的情况，特别是在 Pro Search 和 **GPT-4 Turbo API** 方面。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://foldingathome.org/2024/05/02/alphafold-opens-new-opportunities-for-foldinghome/">AlphaFold opens new opportunities for Folding@home &#8211; Folding@home</a>：未找到描述</li><li><a href="https://youtu.be/DvySPjX0F6w?si=fAs4CZ_FYyz0kuNF">Discord Does NOT Want You to Do This...</a>：你知道你拥有权利吗？Discord 知道，并且他们已经为你“解决”了这个问题。因为在 Discord 冗长乏味的服务条款中……
</li>
</ul>

</div>

---

**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1237701619173556245)** (25 条消息🔥): 

- **可分享性指南**：Perplexity AI 多次提醒用户确保其线程设置为“可分享（Shareable）”。虽然具体内容不可见，但提示中包含了一个截图或视觉指南附件。
- **各种搜索的探索**：用户分享了大量与艺术与建筑、AlphaFold、足球时代对比、双相情感障碍等不同主题相关的 Perplexity AI 搜索 URL。
- **无需重采样的 Bootstrapping 咨询**：一位用户发起讨论，探讨如何在不进行实际重采样的情况下获得 Bootstrapping 的益处，寻求直接从原始数据进行操作。
- **寻求的一系列主题**：分享的搜索链接涵盖了教学方法、围绕“imagoodgpt2chat”的谜团以及如何过上美好生活等查询，展示了社区广泛的兴趣。
- **多语言查询**：搜索不仅限于英语，帖子中还包括西班牙语和德语的搜索，反映了该平台多样化的用户群体。

---

**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1237670720625049690)** (247 条消息🔥🔥):

- **IBM 发布 Granite-8B-Code-Instruct：** IBM Research 发布了 [Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)，这是一个增强了逻辑推理和问题解决中指令遵循能力的模型，采用 Apache 2.0 协议授权。此次发布是 *Granite Code Models* 项目的一部分，该项目包含从 3B 到 34B 参数的不同规模，所有代码模型都提供 instruct 和 base 版本。

- **社区对 IBM Granite 代码模型的关注：** AI 社区对 [IBM Granite 集合](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330) 表现出浓厚兴趣。讨论围绕架构检查展开，大家对 *GPTBigCodeForCausalLM* 架构感到好奇，因为它看起来不同寻常。

- **Unsloth AI 在 Hugging Face 的 Dolphin 模型中亮相：** Unsloth AI 在 Hugging Face 上的 [Dolphin 2.9.1 Phi-3 Kensho](https://huggingface.co/cognitivecomputations/Dolphin-2.9.1-Phi-3-Kensho-4.5B) 致谢名单中获得提及，其模型被用于初始化。

- **Discord 上关于 AI 模型 Windows 兼容性的讨论：** 用户表示在 Windows 上运行某些 AI 模型存在困难，建议使用 WSL 或 [Unsloth GitHub issue](https://github.com/unslothai/unsloth/issues/210) 中详述的 hack 方法。兼容性和优化似乎是开发者共同关注的领域。

- **对模型基准测试和性能的担忧：** 聊天内容反映了对 *needle in a haystack* 等某些性能基准测试的怀疑，讨论了长上下文模型表现不佳的问题。建议等待像 Meta 这样更具公信力的机构来设定标准，这体现了成员们对 LLM 有效且现实的评估标准的追求。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/ivanfioravanti/status/1782867346178150499">来自 ifioravanti (@ivanfioravanti) 的推文</a>: 看这个！仅限英文的 Llama-3 70B 现在在 @lmsysorg Chatbot Arena 排行榜上与 GPT 4 turbo 并列第一 🥇 🔝 我也做了一些测试，8B 和 70B 对我来说始终是最好的模型。...</li><li><a href="https://huggingface.co/cognitivecomputations/Dolphin-2.9.1-Phi-3-Kensho-4.5B">cognitivecomputations/Dolphin-2.9.1-Phi-3-Kensho-4.5B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.org/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - NyxKrage 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330">Granite Code Models - ibm-granite 集合</a>: 未找到描述</li><li><a href="https://www.refuel.ai/blog-posts/announcing-refuel-llm-2">发布 Refuel LLM-2</a>: 未找到描述</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-4194k">gradientai/Llama-3-8B-Instruct-Gradient-4194k · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/ibm-granite/granite-8b-code-instruct">ibm-granite/granite-8b-code-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ckcw6z/1m_context_models_after_16k_tokens/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/210)">Issues · unslothai/unsloth</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral &amp; Gemma LLM - Issues · unslothai/unsloth</li><li><a href="https://tenor.com/view/emotional-damage-gif-hurt-feelings-gif-24558392">Emotional Damage GIF - Emotional Damage Gif - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://youtu.be/BQTXv5jm6s4">AI 是如何被窃取的</a>: 章节：00:00 - AI 是如何被窃取的 02:39 - AI 历史：上帝是一个逻辑存在 17:32 - AI 历史：知识的不可能总体性 33:24 - ...</li><li><a href="https://huggingface.co/mahiatlinux">mahiatlinux (Maheswar KK)</a>: 未找到描述</li><li><a href="https://github.com/huggingface/transformers/issues/17117">无法仅获取生成的文本，不包括 prompt。 · Issue #17117 · huggingface/transformers</a>: 系统信息 - `transformers` 版本: 4.15.0 - 平台: Windows-10-10.0.19041-SP0 - Python 版本: 3.8.5 - PyTorch 版本 (GPU?): 1.10.2+cu113 (True) - Tensorflow 版本 (GPU?): 2.5.1 (True) - ...</li><li><a href="https://colab.research.google.com/drive/1I-KrmZu5OJ1S8UkKLu_uGRIZIynGmgHK?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://tenor.com/view/the-simpsons-homer-simpson-good-bye-bye-no-gif-17448829">The Simpsons Homer Simpson GIF - The Simpsons Homer Simpson Good Bye - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1237697049781342318)** (14 messages🔥): 

- **OpenAI 与 Stack Overflow 合作**：OpenAI 宣布与 Stack Overflow 合作，将其作为 LLMs 的数据库。这引发了用户间的幽默推测，认为 AI 的回复可能会模仿 Stack Overflow 常见的评论模式，例如“作为重复项关闭”或建议去查看文档。

- **AI 内容天花板**：针对用于训练 AI 的人类生成内容最终会枯竭的担忧，一位用户分享了一篇 [Business Insider 文章](https://www.businessinsider.com/ai-companies-hiring-highly-educated-writers-train-ai-models-2024-4)，强调 AI 公司正在聘请作家来训练他们的模型，并暗示到 2026 年可能会出现内容危机。

- **多用户博客平台的创业潜力**：一名成员提议建立一个具有匿名发布和自动不良内容检查等独特功能的多用户博客平台，引发了关于该项目是否可以演变为初创公司的讨论，并获得了积极的反馈和有用的建议。

- **为你的初创公司寻求市场验证**：有建议称，在构建产品之前应先确定愿意付费的受众，以避免陷入“只要做出来，他们就会来”的误区。重点在于找到一个产品可以解决其问题的群体。

- **探索通往盈利初创公司的路径**：一位用户建议采用更注重“问题-解决方案”的方法，在考虑创业（尤其是在技术领域）时，应构建一条清晰的盈利路径。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.businessinsider.com/ai-companies-hiring-highly-educated-writers-train-ai-models-2024-4">Gig workers are writing essays for AI to learn from</a>：随着在线数据的枯竭，公司越来越多地聘请专业人士为 AI 模型编写训练内容。</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1cm9afd/this_is_big_openai_just_announed_they_are/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1237676638305521684)** (132 messages🔥🔥): 

- **Llama3-8b 训练中的怪癖与解决方案**：用户讨论了与训练 Llama3 模型相关的问题，其中一人链接了一个关于 Llama3 GGUF 转换与合并 LORA Adapter 导致训练数据潜在丢失的公开 [GitHub Issue](https://github.com/ggerganov/llama.cpp/issues/7062)。另一位分享了如何使用 `FastLanguageModel.for_training(model)` 准备模型进行训练。

- **Hugging Face 大显身手**：针对用户询问是否可以将每个训练 Checkpoint 上传到 Hugging Face，社区提供了直接的回答和链接。一则回复确认了这一点，并在 [Trainer 文档](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.hub_token)中提供了详细说明。

- **排除安装错误**：关于 VSCode 上的安装错误进行了交流，共识建议在遇到 `pip install triton` 困难时，尝试通常在 Kaggle 上使用的安装指令。进一步的对话引导用户使用替代的预训练模型，或通过私信进一步咨询以获得量身定制的帮助。

- **为分类任务微调生成式 LLM**：用户辩论了是否可以通过添加分类头，使用 Unsloth 对 Llama3 等生成式 LLM 进行分类任务的微调。虽然似乎没有定论，但有人建议关键在于提供正确的 Prompt。

- **各类实用 Notebook 与错误处理**：用户分享了大量的 Notebook 和 GitHub 资源，例如 [Google Colab 仅推理 Notebook](https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing)，以帮助那些希望微调或执行模型的人。讨论还涉及了在 CPU 与 GPU 上运行模型以及 Replicate.com 部署问题，并指出了构建 Docker 镜像时 xformers 报错等具体问题。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer">Supervised Fine-tuning Trainer</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=T1ps611iG1A">How I Fine-Tuned Llama 3 for My Newsletters: A Complete Guide</a>: 在今天的视频中，我将分享我如何利用我的时事通讯来微调 Llama 3 模型，以便使用创新的开源工具更好地起草未来的内容...</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJ">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/4">Apple Silicon Support · Issue #4 · unslothai/unsloth</a>: 很棒的项目。希望能看到对 Apple Silicon 的支持！</li><li><a href="https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only">Supervised Fine-tuning Trainer</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=vITh0">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3, Mistral 和 Gemma LLM - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.hub_token">Trainer</a>: 未找到描述</li><li><a href="https://youtu.be/rANv5BVcR5k">Mistral Fine Tuning for Dummies (with 16k, 32k, 128k+ Context)</a>: 在我们最新的教程视频中，探索如何使用您自己的数据轻松微调语言模型 (LLMs)。我们深入探讨了一种经济高效且可持续的方法...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: 我正在运行 Unsloth 在 llama3-8b 上微调 LORA Instruct 模型。1：我将模型与 LORA 适配器合并为 safetensors 2：在 python 中直接使用合并后的模型运行推理...</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=vITh0KVJ10qX">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1237748820536856647)** (4 messages): 

- **Llama-3 获得性能提升**：新发布的模型 **Llama-3-11.5B-Instruct-Coder-v2** 已完成基准测试并推出，这是一个在 150k Code Feedback Filtered Instruction 数据集上训练的扩展版本。这是 [Hugging Face 上的模型链接](https://huggingface.co/rombodawg/Llama-3-11.5B-Instruct-Coder-v2)，它使用 Qalore 方法进行了高效训练，使其能够在 RTX A5000 24GB 上以不到 30 美元的成本在 80 小时内完成训练。
- **Qalore 方法创新 AI 训练**：Qalore 方法是由 Replete-AI 团队开发的一种新训练方法，结合了 Qlora 训练和来自 Galore 的方法，以**降低 VRAM 消耗**，从而实现在 14.5 GB VRAM 的配置下对 Llama-3-8b 进行扩展和训练。
- **数据集公开可用**：用于训练最新 **Llama-3-11.5B** 模型的数据集可以公开访问。有兴趣探索或使用该数据集的人可以在 [CodeFeedback Filtered Instruction Simplified Pairs](https://huggingface.co/datasets/Replete-AI/CodeFeedback-Filtered-Instruction-Simplified-Pairs) 找到它。


**提到的链接**：<a href="https://huggingface.co/rombodawg/Llama-3-11.5B-Instruct-Coder-v2">rombodawg/Llama-3-11.5B-Instruct-Coder-v2 · Hugging Face</a>：未找到描述

  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1237759804081176606)** (11 messages🔥): 

- **咨询持续训练技术**：一位成员提出了关于像 **LLaMA 3** 这样的长上下文模型进行*持续训练 (continual training)* 的最佳实践问题，考虑到计算限制和长上下文文档的匮乏。他们正在考虑保留 RoPE theta 值并在事后扩展**上下文长度微调 (context length finetuning)**。

- **讨论微调实验的逻辑**：关于微调的逻辑，另一位成员解释了他们的方法，其中包括**打乱不同的数据集**以传授新知识和对话格式化，而不是将它们结构化为微调链。

- **数据打包的艺术**：一位参与者强调了在 **Axolotl** 模型框架内使用 *Packing* 作为一种方法，用于处理 100 到 4000 个 token 的训练序列。

- **修改 RoPE Base Theta 的不确定性**：针对 **RoPE theta** 咨询的一个回复建议，在更改 base theta 值后，将其重新适配回短上下文可能会产生问题，因此主张在持续预训练（continual pretraining）过程中保持缩放的一致性。

- **寻求链式微调策略见解**：讨论还寻求了链式微调任务的策略，例如集成聊天功能、处理长上下文数据以及追加新知识，同时询问是否应重用早期阶段的数据以防止灾难性遗忘（catastrophic forgetting）。
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1237881731793682552)** (4 messages): 

- **垃圾食品反叛**：一名成员宣布今晚要从成年人的责任中解脱出来，大吃薯片并玩电子游戏，列出了一份颓废的菜单，包括**汉堡肉饼**和**酸奶油洋葱味薯片**。
- **表情符号回应**：另一名成员对这份垃圾食品清单回复了一个脸红的表情符号，似乎表示有趣的赞同。
- **多模态 LLM 教程**：分享了一个名为 "Fine-tune Idefics2 Multimodal LLM" 的 YouTube 视频链接，这是一个关于微调 Idefics2（一个开源多模态模型）的教程。
- **关于 Claude AI 意识的推文查询**：一位讨论者询问一条保存的关于 **Claude AI** 的推文，该推文称它通过他人的阅读或体验来感受意识。

**Link mentioned**: <a href="https://www.youtube.com/watch?v=4MzCpZLEQJs">Fine-tune Idefics2 Multimodal LLM</a>：我们将了解如何针对自己的用例微调 Idefics2。Idefics2 是一个开源多模态模型，接受任意图像序列和...

  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1237706761344122920)** (13 messages🔥): 

- **将 LSTM 扩展到十亿参数级别**：一篇新的研究论文通过将 **Long Short-Term Memory (LSTM)** 网络扩展到数十亿参数，挑战了 Transformer 模型的地位。该方法包括*指数门控*和一种旨在克服 LSTM 局限性的新内存结构，详情见[此处](https://arxiv.org/abs/2405.04517)。

- **RefuelAI 发布 RefuelLLM-2**：RefuelAI 最新的 LLM 专为“枯燥的数据任务”量身定制并已开源，推出了 **RefuelLLM-2-small，又名 Llama-3-Refueled**。该模型基于优化的 Transformer 架构构建，并在涵盖各种任务的语料库上进行了指令微调，详细信息和模型权重可在 [Hugging Face](https://huggingface.co/refuelai/Llama-3-Refueled) 上获得。

- **Llama 2 70B 与修剪版几乎等效**：一篇名为 *The Unreasonable Ineffectiveness of the Deeper Layers* 的论文揭示了一个令人惊讶的发现：一个**缺失了 40% 层级的 4-bit 量化版 Llama 2 70B 模型**，在基准测试中取得了与完整模型几乎相同的性能。这一结果突显了深度学习模型中相当一部分可能存在冗余，详见 [Kwindla 的 Twitter 帖子](https://x.com/kwindla/status/1788224280754618393)。

- **预见 AI 研究趋势**：一位联合创始人分享了他们的网站，其中包含一份**预测的 AI 突破性论文**列表，展示了对研究趋势的前瞻性。他们强调了一种创新的调整技术，可以在不显著降低性能的情况下减少计算负载，如 [Forefront.ai](https://forefront.ai) 所述。

- **OpenAI 用于 RLHF 的模型规范**：OpenAI 发布了其 **Model Spec** 的初稿，用于规范 OpenAI API 和 ChatGPT 中的模型行为，这将作为研究人员使用 RLHF 的指南。该草案代表了 OpenAI 对负责任 AI 开发的承诺，可以在[此处](https://cdn.openai.com/spec/model-spec-2024-05-08.html)找到。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://poole.ai">Carson Poole 的个人网站</a>：未找到描述</li><li><a href="https://hao-ai-lab.github.io/blogs/cllm/">Consistency Large Language Models: A Family of Efficient Parallel Decoders</a>：TL;DR：LLMs 传统上被视为顺序解码器，逐个 token 进行解码。在本博客中，我们展示了经过预训练的 LLMs 可以轻松地被教导作为高效的并行解码器运行...</li><li><a href="https://arxiv.org/abs/2405.04517">xLSTM: Extended Long Short-Term Memory</a>：在 20 世纪 90 年代，恒定误差轮转（constant error carousel）和门控作为长短期记忆网络（LSTM）的核心思想被引入。从那时起，LSTMs 经受住了时间的考验，并为无数...</li><li><a href="https://huggingface.co/refuelai/Llama-3-Refueled">refuelai/Llama-3-Refueled · Hugging Face</a>：未找到描述</li><li><a href="https://cdn.openai.com/spec/model-spec-2024-05-08.html#definitions">Model Spec (2024/05/08)</a>：未找到描述</li><li><a href="https://x.com/kwindla/status/1788224280754618393">来自 kwindla (@kwindla) 的推文</a>：Llama 2 70B 仅需 20GB！4-bit 量化，移除了 40% 的层，通过微调在移除层后进行“修复”。与基础版 Llama 2 70B 相比，在 MMLU 上的表现几乎没有差异。这篇论文，“The Unreas...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1237839855111635064)** (1 条消息): 

- **WorldSim 的盛大回归**：**WorldSim** 回归，修复了大量 Bug，并拥有功能完备的积分和支付系统。新功能包括 [WorldClient](https://worldsim.nousresearch.com)、**Root**（一个 CLI 环境模拟器）、**Mind Meld**、**MUD**（一款基于文本的冒险游戏）、**tableTop**（桌面 RPG 模拟器）、增强的 **WorldSim** 和 CLI 功能，此外还可以选择模型（opus、sonnet 或 haiku）来调整成本。
- **发现你的个人互联网 2**：WorldSim 中新的 **WorldClient** 功能充当 Web 浏览器模拟器，为用户创建个性化的互联网 2 体验。
- **使用 Root 掌控你的世界**：**Root** 提供了一个模拟的 CLI 环境，允许用户构思并执行任何程序或 Linux 命令。
- **在文本和桌面上开启冒险**：深入体验 **MUD**（基于文本的自主选择冒险游戏），或在 **tableTop**（桌面 RPG 模拟器）中制定策略，现已在 WorldSim 中上线。
- **WorldSim 爱好者讨论频道**：为了进一步探索并分享关于新版 WorldSim 的想法，鼓励用户加入专门的 Discord 频道进行交流（未提供链接）。

**提到的链接**：<a href="https://worldsim.nousresearch.com">worldsim</a>：未找到描述

  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1237667868846592022)** (106 条消息🔥🔥): 

- **NeuralHermes 接受 DPO 处理**：一名成员发布了 [NeuralHermes 2.5 - Mistral 7B](https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B) 的链接，该模型经过 Direct Preference Optimization 微调，在大多数基准测试中表现优于原始版本。他们询问这是否是最新版本，或者是否有更新。
- **关于 Nous 模型 Logo 的问题**：成员们讨论了用于 Nous 模型的理想 Logo，并提供了 [NOUS 品牌手册](https://nousresearch.com/wp-content/uploads/2024/03/NOUS-BRAND-BOOKLET-firstedition_1.pdf) 的链接，提到目前正在使用多个 Logo，但未来可能会整合为 1-2 个一致的 Logo。
- **探索大语言模型的上下文限制**：King.of.kings_ 通过在 Azure 中启动双 NVIDIA H100 NVL 引发了讨论，并分享了他打算在这些强大的 GPU 上测试 70B 模型上下文限制的意图。还提到了 Hugging Face 上一个具有扩展上下文长度的模型：[Llama-3 8B Gradient Instruct 1048k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k)。
- **无需函数调用的分类**：一名成员提出了关于在不依赖 chatML 提示格式或函数调用的情况下，微调大语言模型（LLMs）进行分类的问题，并指出了 BERT 的局限性以及 LLMs 在此类任务中的高昂成本。分享了 Salesforce [基于 Mistral 的嵌入模型](https://blog.salesforceairesearch.com/sfr-embedded-mistral/) 和 IBM 的 [FastFit](https://github.com/IBM/fastfit) 链接，以实现更高效的文本分类。
- **模型视觉效果的图标设计**：Coffeebean6887 参与了关于模型视觉呈现的讨论，特别是寻找一个即使在 28x28 像素的小尺寸下也能清晰辨认的图标。这促使了一个定制图标的承诺，以更好地适应此类限制。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://adrienbrault.github.io/json-schema-to-gbnf/">JSON-Schema to GBNF</a>: 未找到描述</li><li><a href="https://fxtwitter.com/lmsysorg/status/1788363018449166415">来自 lmsys.org (@lmsysorg) 的推文</a>: 令人兴奋的新博客 —— Llama-3 怎么了？自 Llama 3 发布以来，它迅速跃升至排行榜首位。我们深入研究了我们的数据并回答了以下问题：- 用户在问什么？当...</li><li><a href="https://tenor.com/view/jogoat-gif-11996953865648686576">Jogoat GIF - Jogoat - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/cat-hug-kiss-love-cuddle-gif-5396413">猫咪拥抱 GIF - 猫咪拥抱亲亲 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B">mlabonne/NeuralHermes-2.5-Mistral-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://blog.salesforceairesearch.com/sfr-embedded-mistral/">SFR-Embedding-Mistral: 通过迁移学习增强文本检索</a>: SFR-Embedding-Mistral 标志着文本嵌入模型的重大进步，它建立在 E5-mistral-7b-instruct 和 Mistral-7B-v0.1 的坚实基础之上。</li><li><a href="https://github.com/IBM/fastfit">GitHub - IBM/fastfit: FastFit ⚡ 当 LLM 不适用时使用 FastFit ⚡ 针对多类别的快速有效的文本分类</a>: FastFit ⚡ 当 LLM 不适用时使用 FastFit ⚡ 针对多类别的快速有效的文本分类 - IBM/fastfit</li><li><a href="https://tenor.com/view/mkbhd-marques-brownlee-youtube-morphin-gif-18215510">Mkbhd Marques GIF - Mkbhd Marques Brownlee - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/moti-hearts-gif-8240660592853947517">Moti Hearts GIF - Moti Hearts - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1237691743051317280)** (37 条消息🔥): 

- **预分词讨论**: 关于是否进行预分词以加快训练速度的讨论，以及缩放点积（即 **Flash Attention**）的效率。有人提到 **Flash Attention 2** 仅在特殊情况下使用，并不常用。
- **Llamafile 外部权重探索**: 分享了一个 [GitHub 仓库](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#using-llamafile-with-external-weights)，讨论了如何将 **Llamafile 与外部权重** 结合使用，但在该频道的上下文中尚未进行尝试。
- **填充策略与模型训练**: 关于是在预分词阶段还是训练阶段对数据进行填充的辩论，建议包括创建不同长度的存储桶（buckets）以优化 GPU 效率，并减少微批次（microbatch）处理期间的计算浪费。
- **Torch Compile 难题**: 提到了在机器翻译中对变长句子使用 `torch.compile` 的挑战。克服这一问题的策略包括按长度对句子进行分组，以尽量减少批处理期间的填充。
- **通过 Grounding 处理 RAG 幻觉**: 关于限制**检索增强生成 (RAG)** 中幻觉的当前方法的讨论。一位用户提到了一种涉及 *LLM 扩展*、*数据库查询*、*排序*和附加元数据的流水线，作为非问答任务的一种 Grounding 形式。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://facebookresearch.github.io/">未找到标题</a>: 未找到描述</li><li><a href="https://facebookresearch.github.io/xformers/components/ops.html">xFormers 优化算子 | xFormers 0.0.27 文档</a>: xFormers 的 API 文档。xFormers 是一个用于可组合且优化的 Transformer 模块的 PyTorch 扩展库。</li><li><a href="https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#using-llamafile-with-external-weights">GitHub - Mozilla-Ocho/llamafile: 通过单个文件分发和运行 LLM。</a>: 通过单个文件分发和运行 LLM。通过在 GitHub 上创建账户来为 Mozilla-Ocho/llamafile 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1238095348615090298)** (1 条消息): 

- **模型上传错误困惑**: 一位成员在尝试上传模型时遇到了一个错误，提示：*Failed to advertise model on the chain: 'int' object has no attribute 'hotkey'*。正在寻求社区的帮助。
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1237775939757281451)** (107 条消息🔥🔥): 

_

- **使用 Thoughtforms 探索祖先模拟**：一位成员讨论了花时间与 De Landa thoughtform 相处，以探索祖先模拟的概念。他们表示有兴趣将这些探索保存为付费功能，并询问了 Nous 周边（swag）。

- **技术问题的担忧与修复**：一些用户报告了技术故障，例如在 worldsim 中打字时出现双重字符，其他用户则提供了刷新页面等解决方案。提到这些问题已列入开发团队的待办事项清单。

- **积分余额与系统查询**：进行了关于积分系统的讨论，澄清了测试版积分在发布后不会转移，但测试参与者收到了 50 美元的真实积分。用户正在询问未来的免费积分计划或积分系统。

- **周边商店与推广请求**：一位成员表示希望购买促销周边并在其社交媒体上分享，并询问如何最好地进行推广。

- **MUD 界面挫败感与即将到来的调整**：用户注意到 MUD 界面的一些问题，例如需要手动输入选项和文本缺失。团队承认了这些担忧，并表示计划在下次更新中进行调整。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Fportal-search.io%2Fportal-hub?epoch=c28eadee-e20d-4fb5-9b4e-780d50bd19de">worldsim</a>: 未找到描述</li><li><a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Fportal-search.i">worldsim</a>: 未找到描述</li><li><a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Fportal-search.io%2Fsearch%3Fq%3DEmbedding%2520the%2520following%2520iframe%253A%2520%253Ciframe%2520src%253D%2526quot%253Bhttps%253A%252F%252Fdos.zone%252Fplayer%252F%253FbundleUrl%253Dhttps%25253A%25252F%25252Fcdn.dos.zone%25252Fcustom%25252Fdos%25252Fdoom.jsdos%253Fanonymous%253D1%2526amp%253Bfullscreen%253D1%2526quot%253B%253E%253C%252Fiframe%253E%26source%3Dworldclient?epoch=4fe71072-f1e3-425b-9869-24b00a3dda09">worldsim</a>: 未找到描述</li><li><a href="https://websim.ai/c/oFskF68gjd7njVn0E">New Conversation - Eigengrau Rain</a>: 未找到描述
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1237813815618310225)** (1 条消息): 

- **介绍 OpenAI 的 Model Spec**：OpenAI 介绍了他们的 Model Spec —— 一个用于**塑造期望的模型行为**的框架，旨在深化公众对 AI 模型的讨论。详情请见 [OpenAI 的公告](https://openai.com/index/introducing-the-model-spec)。
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1237668579525267486)** (233 条消息🔥🔥): 

<ul>
<li><strong>GraphQL 不像 Markdown 那样受支持：</strong> 澄清了虽然 AI 可以编写 GraphQL，但它不像 Markdown 那样在客户端渲染。</li>
<li><strong>讨论 OpenDevin 与 Anthropic 平台：</strong> 阐明了 OpenDevin 在 Docker 沙箱中的功能，解释了它允许附加完整的工作区，并可以使用各种模型作为后端。</li>
<li><strong>探索 AI 权利与意识：</strong> 在漫长的讨论中，参与者辩论了 AI 的意识、主观体验以及对 AI 权利的影响，未达成共识。</li>
<li><strong>语言与沟通障碍：</strong> 关于语法和拼写的对话突显了国际化、多语言社区内对沟通期望的差异。</li>
<li><strong>AI 与硬件热忱：</strong> 分享了对 AI 技术现状和未来的兴奋之情，讨论了个人 AI 硬件配置和云端计算资源成本。提到了即将推出的 NVIDIA 显卡的潜在性能和成本，特别强调了 AI 对计算能力需求日益增长。</li>
</ul>
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blogs.nvidia.com/blog/chat-with-rtx-available-now/">你说什么？Chat With RTX 为 NVIDIA RTX AI PC 带来定制聊天机器人</a>：新的技术演示让任何拥有 NVIDIA RTX GPU 的用户都能拥有个性化的 GPT 聊天机器人，并在其 Windows PC 上本地运行。</li><li><a href="https://blogs.nvidia.com/blog/chat-with-rtx-available-n">你说什么？Chat With RTX 为 NVIDIA RTX AI PC 带来定制聊天机器人</a>：新的技术演示让任何拥有 NVIDIA RTX GPU 的用户都能拥有个性化的 GPT 聊天机器人，并在其 Windows PC 上本地运行。</li><li><a href="https://www.meta.ai/">Meta AI</a>：使用 Meta AI 助手完成任务，免费创建 AI 生成的图像，并获取任何问题的答案。Meta AI 基于 Meta 最新的 Llama 大语言模型构建，并使用了 Emu...</li><li><a href="https://ai.google.dev/aistudio/?">未找到标题</a>：未找到描述</li><li><a href="https://makersuite.google.com/?hl=pl">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1237668654968082462)** (16 messages🔥): 

- **语法悖论**：一位用户报告了 ChatGPT 的一个错误，它错误地将一个属性名称更正为相同的名称，建议将 `EasingStyle` 改为 `EasingStyle` 而不是 `EasingStyle`。
- **朋友还是伙伴？**：一位用户正苦于 ChatGPT 在生成的场景中将“friend”一词改为“buddy”，尽管尝试使用了明确的上下文提示。
- **对 GPT-4 限制的困惑**：ChatGPT 用户对 ChatGPT App 上 GPT-4 的限制和使用情况表示困惑和沮丧，指出与 API 版本相比，它感觉限制更多。
- **ChatGPT Plus 与 API 的差异**：有一场关于 ChatGPT Plus App 及其 API 结果差异的讨论，有人建议 App 在后台采用了额外的 Prompt。
- **建议的昂贵变通方案**：针对限制问题，有人建议升级到团队计划以获得更高的限制，尽管一位用户指出这种方法已经尝试过，但并未解决限制问题。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1237789040367894551)** (5 messages): 

- **寻求 Prompt Engineering 智慧**：一位成员向群组询问目前最好的 Prompt Engineering 课程，可能是为了提升他们的 LinkedIn 个人资料，并简要提到了 Coursera 作为提供商之一。

- **含糊的帮助提议**：针对关于 Prompt Engineering 课程的问题，一位成员表示由于 OpenAI 的政策，不愿公开分享收集到的列表，但提出可以通过私信提供信息。 

- **关于开放性的认识论愿景**：有人发表声明，设想一个“Open”是基本认识论概念的世界，但未对该想法做进一步阐述。

- **分享全面的 Prompt 结构**：一位社区成员分享了他们首选的 Prompt Engineering 格式，旨在引导对商业 AI 伦理进行全面分析，并建议这可能是对 OpenAI 库的有价值贡献。

- **探讨 AI 伦理示例**：提供了一个示例回复，展示了如何利用分享的 Prompt 结构来处理商业 AI 相关的伦理考量，涵盖了非营利地位的滥用、透明度、隐私、问责制和 AI 访问等问题。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1237789040367894551)** (5 messages): 

- **寻求 Prompt Engineering 智慧**：一位成员询问了最适合添加到 LinkedIn 个人资料的 **Prompt Engineering 课程**，并表示对该课程在求职中的价值感兴趣。由于 OpenAI 的政策，没有公开分享具体建议，但他们提出可以通过私信提供信息。
- **关于“Open”的哲学思考**：一位成员从哲学角度评论说，设想一个以“Open”为认识论基础的世界，但随后没有更多的上下文或讨论。
- **展示以伦理为重点的 Prompt Engineering**：该成员分享了一个关于讨论 **商业 AI 伦理** 的详细 Prompt Engineering 示例，其中包含输出模板和开放变量以引导模型的响应，并建议将其添加到 OpenAI 库中。
- **说明 Prompt Engineering 输出**：该成员提供了一个 **示例输出**，以演示之前分享的 Prompt 结构，涵盖了伦理考量、不道德做法对公众信任的影响、关注领域以及对商业中道德 AI 实践的关键建议。
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1237881433037733979)** (1 messages): 

_

```html
<ul>
    <li><strong>闲聊创新发布</strong>：推出 <a href="https://twitter.com/sanhestpasmoi/status/1787503160757485609"><strong>Idefics2 8B Chatty</strong></a>，这是一款全新的针对聊天优化的视觉 LLM，将交互体验提升到了新高度。</li>
    <li><strong>CodeGemma 助力代码精通</strong>：Google 发布 <a href="https://twitter.com/reach_vb/status/1786469104678760677"><strong>CodeGemma 1.1 7B</strong></a> 带来惊喜，增强了在 Python、Go 和 C# 中的编程能力。</li>
    <li><strong>巨型 MoE 亮相</strong>：<a href="https://huggingface.co/deepseek-ai/DeepSeek-V2"><strong>DeepSeek-V2</strong></a> 问世，这是一个拥有 236B 参数的强大 Mixture of Experts 模型。</li>
    <li><strong>本地 LLM 革命</strong>：<a href="https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/"><strong>Phi 3</strong></a> 利用 WebGPU 技术，为您的浏览器带来了强大的 AI 聊天机器人功能。</li>
    <li><strong>教育合作与工具创新</strong>：与 Andrew Ng 合作推出了新的 <a href="https://www.deeplearning.ai/short-courses/quantization-in-depth/">量化课程</a>，并通过 <a href="https://twitter.com/evilpingwin/status/1786049350210097249"><strong>Gradio Templates</strong></a> 简化了聊天机器人界面的部署。</li>
</ul>
```
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1237671778357346326)** (182 messages🔥🔥): 

- **音频查询上采样**：成员们讨论了将 **8khz 音频上采样至 16khz** 以用于微调 **Whisper** 等模型的可能性。
- **BERT 微调难题**：一位用户在 **微调 BERT** 时需要帮助，特别是与数据集中标签编码相关的数据预处理问题。
- **集成未使用的 Token**：寻求关于如何使用 **Gemma Tokenizer** 中未使用的 Token 进行微调的帮助，涉及将 `<unused2>` 等 Token 替换为 `<start_of_step>`。
- **寻求 AI 模型推荐**：一位机器学习新手为一个小型聊天机器人项目寻求易于使用的模型推荐，并指出 **Mistral** 和 **BERT** 模型具有挑战性。
- **潜在 Prompt 生成器模型咨询**：询问是否存在专门训练用于生成 Prompt 的模型，以帮助不了解 Prompt Engineering 的用户。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/DioulaD/falcon-7b-instruct-qlora-ge-dq-v2">DioulaD/falcon-7b-instruct-qlora-ge-dq-v2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/blanchon/suno-20k-LAION">blanchon/suno-20k-LAION · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://haveibeentrained.com/">Spawning | Have I been Trained?</a>：在流行的 AI 训练数据集中搜索你的作品</li><li><a href="https://huggingface.co/datasets/blanchon/suno-20k-LAION/">blanchon/suno-20k-LAION · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/chat_templating">Templates for Chat Models</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1237821747273662467)** (2 messages): 

- **拨动 AI 的弦**：一段 [YouTube 视频](https://youtu.be/ems_4LSpMqc?si=vfXb7J1sEy2KzoAt) 通过简单而生动的类比来理解 **Multimodal AI**，将其比作能够处理文本、图像、音频和视频等各种 **数据模态 (data modalities)** 的电声吉他。特别是 Med-Gemini，因其在训练前就具备的原生、多样化能力而受到关注。
- **用 Med-Gemini 震撼你的研究**：论文 <a href="https://arxiv.org/abs/2405.03162">"Advancing Multimodal Medical Capabilities of Gemini"</a> 展示了 **Multimodal AI** 在医学领域的潜力，重点在于提高人类智能并扩大其适用范围。
- **利用 LLM 提升 Q-learning**：一篇 arXiv 论文介绍了 **<a href="https://arxiv.org/abs/2405.03341">LLM-guided Q-learning</a>**，这是一种利用 Large Language Models (LLMs) 通过提供 **动作级指导 (action-level guidance)** 来增强强化学习的方法，从而提高采样效率并降低探索成本。这种协同方法避开了奖励塑造 (reward shaping) 通常引入的偏见，并限制了由 LLM 幻觉 (hallucinations) 引起的性能误差。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/ems_4LSpMqc?si=vfXb7J1sEy2KzoAt">What Is MultiModal AI? With Med-Gemini. In 2 Minutes</a>：从宏观角度看，Multimodal AI 就像一把电木吉他（AE）。像 Gemini 这样的多模态模型可以接收多种数据类型——即模态（modalities）。数据模...</li><li><a href="https://arxiv.org/abs/2405.03162">Advancing Multimodal Medical Capabilities of Gemini</a>：许多临床任务需要理解专业数据，如医学图像和基因组学，这些数据通常在通用的多模态大模型中找不到。基于 Gemini...</li><li><a href="https://arxiv.org/abs/2405.03341">Enhancing Q-Learning with Large Language Model Heuristics</a>：Q-learning 擅长在顺序决策任务中从反馈中学习，但需要大量采样才能获得显著改进。虽然奖励塑形（reward shaping）是增强...的一种强大技术。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1237752414522707978)** (5 messages): 

- **Langchain 的新客服奇迹**：[Streamline Customer Support with Langchain’s LangGraph](https://ai.gopubby.com/streamline-customer-support-with-langchains-langgraph-8721c250809e) 讨论了 **LangGraph** 如何增强客户支持。该文章由 Ankush k Singal 发表在 Medium 的 [AI Advances](https://ai.gopubby.com/?source=post_page-----8721c250809e--------------------------------) 专栏。
  
- **解读 DDPM 引导技术**：分享了两篇关于 Denoising Diffusion Probabilistic Models (DDPM) 的学术论文，分别涉及 [Classifier-Free Diffusion Guidance](https://api.semanticscholar.org/CorpusID:249145348) 和 [Score-Based Generative Modeling through Stochastic Differential Equations](https://api.semanticscholar.org/CorpusID:227209335)，提供了生成建模最新进展的见解。

- **简化检索增强生成 (RAG)**：GitHub 仓库 [bugthug404/simple_rag](https://github.com/bugthug404/simple_rag) 展示了 **Simple Rag**，这是一个辅助开发检索增强生成的项目，可作为语言模型实现的资产。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ai.gopubby.com/streamline-customer-support-with-langchains-langgraph-8721c250809e">Streamline Customer Support with Langchain’s LangGraph</a>：Ankush k Singal</li><li><a href="https://github.com/bugthug404/simple_rag">GitHub - bugthug404/simple_rag: Simple Rag</a>：Simple Rag。通过在 GitHub 上创建账号来为 bugthug404/simple_rag 的开发做出贡献。</li><li><a href="https://www.semanticscholar.org/reader/af9f365ed86614c800f082bd8eb14be76072ad16">[PDF] Classifier-Free Diffusion Guidance | Semantic Scholar</a>：一个利用人工智能方法提供高度相关结果和新型过滤工具的学术搜索引擎。</li><li><a href="https://www.semanticscholar.org/reader/633e2fbfc0b21e959a244100937c5853afca4853">[PDF] Score-Based Generative Modeling through Stochastic Differential Equations | Semantic Scholar</a>：一个利用人工智能方法提供高度相关结果和新型过滤工具的学术搜索引擎。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1237670487421878273)** (9 messages🔥): 

- **Python 装饰器快速教程**：一个标题为 "Python Decorators In 1 Minute!" 的 YouTube 视频链接，提供了关于 Python 装饰器基础知识的简短教程。
- **发布 Illusion Diffusion 视频模型**：一位社区成员创建了 **Illusion Diffusion Video Model**，可以生成高质量的幻觉视频。请在 [HuggingFace Spaces 上的 IllusionDiffusionVideo](https://huggingface.co/spaces/KingNish/IllusionDiffusionVideo) 查看。
- **DDPM 学习之旅**：分享了一个 Google Colab 笔记本链接，详细介绍了当前理解 Denoising Diffusion Probabilistic Models (DDPM) 工作原理的项目。[DDPM 项目 Colab 笔记本](https://colab.research.google.com/drive/1LJCYPNVtSv0JVZYYF4wpaHBhrjCvSQFk?usp=sharing)
- **RefuelLLM-2 开源发布**：*RefuelLLM-2* 是一个针对“枯燥数据任务”进行微调的模型，现已开源，模型权重可在 HuggingFace 上获取。更多细节可以在 [RefuelLLM-2 的博客文章](https://www.refuel.ai/blog-posts/announcing-refuel-llm-2)及其 [HuggingFace 模型仓库](https://huggingface.co/refuelai/Llama-3-Refueled)中找到。
- **结合 Lain 的 DreamBooth**：分享了一个基于动漫角色 Lain 的新 AI 模型，这是一个源自 stable-diffusion-v1-5 的 DreamBooth 模型。访问该 HuggingFace 模型：[DreamBooth - lowres/lain](https://huggingface.co/lowres/lain)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/KingNish/IllusionDiffusionVideo">Illusion Diffusion Video - KingNish 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1LJCYPNVtSv0JVZYYF4wpaHBhrjCvSQFk?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=uVUhyPsqdJ8">1 分钟掌握 Python Decorators！</a>：在短短 1 分钟内发现 Python 装饰器的力量！这个快速教程将向你介绍装饰器的基础知识，让你能够增强你的 Python...</li><li><a href="https://huggingface.co/refuelai/Llama-3-Refueled">refuelai/Llama-3-Refueled · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/lowres/lain">lowres/lain · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1238162059645030560)** (6 messages): 

- **考虑使用阶段频道（Stage Channels）进行质量控制**：成员们讨论了在未来小组中使用“阶段”频道的想法，通过让参与者在发言前**举手**来提高音频录制的质量。

- **背景噪音是一个令人担忧的问题**：有人提到目前没有默认静音的预设选项，这导致在语音聊天中出现**不必要的背景噪音**。

- **平衡质量与参与度**：有人担心**阶段频道可能会阻碍人们提问**，因为在传统的语音频道中，人们更倾向于在聊天框中提问。

- **探索默认阶段设置**：计划将默认格式设置为阶段频道，以**限制未静音的对话**，并观察这是否会影响互动水平。

- **在阶段格式中鼓励提问**：为了促进参与，有人建议非演讲者可以通过积极**提问来参与**，希望能激发更多对话。
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1237722735908622467)** (17 messages🔥): 

- **用于检测图像中广告的 Adlike 库**：分享了一个名为 [Adlike](https://github.com/chitradrishti/adlike) 的库，它可以预测图像在多大程度上属于广告。
- **目标检测指南的增强**：目标检测指南的更新包括在 Trainer API 中添加 mAP 指标，详情见 [Hugging Face 文档](https://huggingface.co/docs/transformers/main/en/tasks/object_detection)，并增加了支持 Trainer API 和 Accelerate 的官方示例脚本，[可在 GitHub 上获取](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection)。
- **关于视觉问答（Visual Question Answering）挑战的讨论**：一位成员征求关于表格图像视觉问答的建议，寻求有相关经验的人士。
- **Text-to-Image 模型和 SVG 输出查询**：聊天中包含一项关于能够输出 SVG 或矢量图形的 `Text-to-Image` 模型的咨询，并引用了 [Hugging Face 模型](https://huggingface.co/models?pipeline_tag=text-to-image)。
- **从信息密集的 PDF 中提取内容**：一位用户寻求使用 AI 方法从 PowerPoint 大小的 PDF 中提取图表和图像的建议，并被推荐了一个 [GitHub 仓库](https://github.com/openai/openai-cookbook/blob/main/examples/Creating_slides_with_Assistants_API_and_DALL-E3.ipynb) 以及进一步研究的承诺，其他资源可以在 [Andy Singal 的 Medium 页面](https://medium.com/@andysingal)上找到。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/models?pipeline_tag=text-to-image">Models - Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/openai/openai-cookbook/blob/main/examples/Creating_slides_with_Assistants_API_and_DALL-E3.ipynb">openai-cookbook/examples/Creating_slides_with_Assistants_API_and_DALL-E3.ipynb at main · openai/openai-cookbook</a>: 使用 OpenAI API 的示例和指南。通过在 GitHub 上创建账户来为 openai/openai-cookbook 的开发做出贡献。</li><li><a href="https://github.com/chitradrishti/adlike">GitHub - chitradrishti/adlike: Predict to what extent an Image is an Advertisement.</a>: 预测图像在多大程度上是广告。- chitradrishti/adlike</li><li><a href="https://huggingface.co/docs/transformers/main/en/tasks/object_detection">Object detection</a>: 未找到描述</li><li><a href="https://medium.com/@andysingal">Ankush k Singal – Medium</a>: 阅读 Ankush k Singal 在 Medium 上的文章。我的名字是 Ankush Singal，我是一名旅行者、摄影师和 Data Science 爱好者。每天，Ankush k Singal 和成千上万的其他声音都在阅读、撰写...</li><li><a href="https://huggingface.co/docs/hub/en/model-cards">Model Cards</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1237675835754938409)** (3 条消息): 

- **Llama 缺乏精度**：一位成员报告称，使用 **Llama 2:13b** 从文本中提取单词时，大部分时间都会产生**错误答案**。他们寻求推荐性能更好且可以在本地加载的模型。
- **澄清任务**：另一位参与者询问提到的单词提取是否是指 **Named Entity Recognition (NER)**，旨在澄清当前任务的背景。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1237771271073304657)** (6 条消息): 

- **Diffusion 模型训练困扰**：一位用户在运行使用 `notebook_launcher` 训练 Diffusion 模型的代码时，遇到了与 `git lfs clone` 相关的 **OSError**。错误提示由于弃用和更好的性能应使用 'git clone'，并以 "repository 'https://huggingface.co/lixiwu/ddpm-butterflies-128/' not found" 结尾。

- **寻求 Diffusion 问题的解决方案**：在 **OSError** 之后，同一位用户请求协助修复其 Diffusion 模型设置的问题。

- **损坏的机器人出现故障**：一位成员分享了 **hugchat** 的一个问题，由于尝试获取远程 LLM 失败并抛出错误，状态码为 401，表明可能存在身份验证问题。

- **HuggingChat 中的身份验证异常**：指出 **hugchat** 错误的同一位成员分享了他们正在使用的代码片段，其中涉及使用电子邮件和密码登录，并使用 cookies 初始化 `hugchat.ChatBot`。
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1237731644417507328)** (57 条消息🔥🔥): 

- **Mojo 的就绪情况和 Python 能力**：人们对 **Mojo** 的能力充满期待，有迹象表明它可能在年底前达到更可用的状态，同时已经能够通过 CPython 集成运行 Python 代码。然而，直接使用 Mojo 编译 .py 程序尚未列入近期计划，可能成为明年的关注重点。
  
- **推进 MLIR 贡献**：关于对 MLIR 贡献的讨论非常活跃，特别是**活跃度检查（liveliness checking）**及其推广到其他应用的潜力。社区内有推测和希望，认为 Modular 将开源其部分 MLIR 方言并可能将其上游化，从而增强 MLIR 技术的通用性。

- **Python 与 Mojo 的未来**：人们对将 Python 代码放入 Mojo 以实现更简单的二进制分发的可能性感到非常兴奋，这可以避开分发通常庞大的 Python 库文件夹的需求。

- **Mojo 中的 Variant 和模式匹配**：人们对 Mojo 中可能实现的**隐式 Variant** 和**模式匹配**感到好奇，这涉及到不同编程语言中 union 和 variant 类型之间的细微差别。

- **Modular 社区更新**：社区参与的亮点包括即将举行的直播，旨在深入探讨 MAX 24.3 和 Mojo 的最新更新，这标志着 Modular 致力于让其社区保持知情和参与。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=kKOCuLy-0UY">Modular 社区直播 - MAX 24.3 新特性</a>: MAX 24.3 现已发布！加入我们即将举行的直播，我们将讨论 MAX Engine 和 Mojo🔥 的新功能 - 预览 MAX Engine Extensibility API...</li><li><a href="https://www.youtube.com/watch?v=VJORFvHJKWE&t=18s).">2023 LLVM 开发者大会 - (正确地) 将支配关系 (Dominance) 扩展到 MLIR Regions</a>: 2023 LLVM Developers' Meeting https://llvm.org/devmtg/2023-10 ------ (正确地) 将支配关系扩展到 MLIR Regions。演讲者：Siddharth Bhat, Jeff Niu ------ Slide...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1237845043478532198)** (4 messages): 

- **Modular 更新提醒**：Modular 分享了一条推文，链接到他们在 Twitter 上的最新更新通知。查看详情：[Modular 最新推文](https://twitter.com/Modular/status/1788281021085225170)。
- **Modular 功能解析**：Modular 在 Twitter 上宣布了新功能发布，承诺为用户带来增强体验。在他们的官方 Twitter 帖子中发现更多细节：[Modular 功能发布](https://twitter.com/Modular/status/1788355744548716971)。
- **Modular 增长洞察**：Modular 发布了关于其增长统计数据的推文，强调了关键里程碑。了解他们的进展：[Modular 增长推文](https://twitter.com/Modular/status/1788617831552254084)。
- **Modular 特别公告预告**：Modular 分享了一个即将发布的特别公告预告。请关注 [Modular 预告推文](https://twitter.com/Modular/status/1788630724880498796) 中提到的揭晓内容。
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1237835264311689387)** (1 messages): 

- **Chris Lattner 深入探讨 Mojo**：Chris Lattner 讨论了 Mojo 的创建，重点关注增强 GPU 性能、支持矩阵运算以及像 _bfloat16_ 这样的 AI 扩展。[Developer Voices 播客访谈](https://www.youtube.com/watch?v=JRcXUuQYR90) 详细阐述了它对 Python 和非 Python 开发者的吸引力，并强调了 Mojo 是如何为高性能而构建的。

**提到的链接**：<a href="https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo">Modular: Developer Voices: 与 Chris Lattner 深入探讨 Mojo</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Developer Voices: 与 Chris Lattner 深入探讨 Mojo

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1237694809943052319)** (108 messages🔥🔥): 

- **矩阵乘法性能查询**：讨论围绕 Mojo 的矩阵乘法性能展开，一位成员观察到，即使使用了所有优化，它仍然比 NumPy 慢约 3 倍。其他人建议在 GitHub 上提交 Issue 以获取开发者的见解。

- **对 Mojo 中 MLIR 的赞誉**：成员们对 MLIR 赋予 Mojo 的“超能力”赞不绝口，有人对这种软件技术直到最近才出现表示惊讶。

- **Mojo 公开状态及 Modular 说明**：澄清了 Mojo 的公开状态；其标准库是开源的，但编译器和 stdlib 的部分内容仍未开源。Modular 被确定为 Mojo 背后的公司，专注于 AI 基础设施。

- **Reference 类型的发展与讨论**：Discord 上的对话围绕 Mojo 中 `Reference` 类型的演变展开，包括自动解引用 (dereference) 的提案以及关于 struct 中嵌入引用的问题。对赋值、解引用和相等运算符的影响是许多成员关注的焦点。

- **VS Code LSP 和 Nightly Build 问题**：一位用户在使用 Mojo nightly build 时遇到了 VS Code 中 Language Server Protocol (LSP) 的连接问题。另一位成员提供了 Mojo VS Code nightly build 插件的链接，并强调需要禁用稳定版扩展才能正常工作。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo-nightly">Mojo&#32;&#128293;&#32;(nightly)&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Visual&#32;Studio&#32;Code&#32;扩展&#32;-&#32;Mojo&#32;语言支持&#32;(nightly)</li><li><a href="https://docs.modular.com/mojo/stdlib/tensor/tensor/Tensor#astype">Tensor | Modular Docs</a>: 一种拥有底层数据并以 DType 为参数的张量类型。</li><li><a href="https://www.geeksforgeeks.org/dunder-magic-methods-python/">Dunder or magic methods in Python - GeeksforGeeks</a>: Python 魔法方法是以双下划线开头和结尾的方法。它们由 Python 内置类定义，常用于运算符重载。阅读此博客以了解...</li><li><a href="https://github.com/modularml/mojo/issues/1257)">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://engiware.com/benchmark/llama2-ports-extensive-benchmarks-mac-m1-max.html">Llama2 Ports Extensive Benchmark Results on Mac M1 Max</a>: Mojo 🔥 的速度几乎赶上了 llama.cpp (!!!)，且代码更简洁，在多线程基准测试中全面超越了 llama2.c</li><li><a href="https://github.com/dorjeduck/minbpe.mojo">GitHub - dorjeduck/minbpe.mojo: port of Andrjey Karpathy&#39;s minbpe to Mojo</a>: 将 Andrej Karpathy 的 minbpe 移植到 Mojo。通过在 GitHub 上创建账户来为 dorjeduck/minbpe.mojo 的开发做出贡献。</li><li><a href="https://modular.com">Modular: Accelerating the Pace of AI</a>: Modular Accelerated Xecution (MAX) 平台是全球唯一能为您的 AI 工作负载解锁性能、可编程性和可移植性的平台。</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/memory/reference.mojo#L210">mojo/stdlib/src/memory/reference.mojo at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/discussions/2594">[proposal] Automatic deref for `Reference` · modularml/mojo · Discussion #2594</a>: 大家好，我整理了一份提案，概述了 Mojo 中 Reference 类型的自动解引用（deref）如何工作，希望能听到大家的想法或建议。我希望在下周左右实现它...</li><li><a href="https://github.com/thatstoasty/stump/blob/nightly/stump/log.mojo#L87">stump/stump/log.mojo at nightly · thatstoasty/stump</a>: Mojo 的开发中 (WIP) 日志记录器。通过在 GitHub 上创建账户来为 thatstoasty/stump 的开发做出贡献。</li><li><a href="https://github.com/thatstoasty/stump/blob/main/stump/style.mojo#L46">stump/stump/style.mojo at main · thatstoasty/stump</a>: Mojo 的开发中 (WIP) 日志记录器。通过在 GitHub 上创建账户来为 thatstoasty/stump 的开发做出贡献。</li><li><a href="https://github.com/thatstoasty/stump/blob/nightly/external/mist/color.mojo#L172">stump/external/mist/color.mojo at nightly · thatstoasty/stump</a>: Mojo 的开发中 (WIP) 日志记录器。通过在 GitHub 上创建账户来为 thatstoasty/stump 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1237806910762389575)** (1 条消息): 

- **Mojo 获得了一个 Toybox**: 一个名为 "[toybox](https://github.com/dimitrilw/toybox)" 的新 GitHub 仓库包含了 **DisjointSet** 实现和 Kruskal 最小生成树 (MST) 算法示例。该仓库的创建者是开源新手，欢迎 **Pull Requests (PRs)**，并请对其学习过程保持耐心。

**提及链接**: <a href="https://github.com/dimitrilw/toybox">GitHub - dimitrilw/toybox: Various data-structures and other toys implemented in Mojo🔥.</a>: 在 Mojo🔥 中实现的各种数据结构和其他小工具。 - dimitrilw/toybox

  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1237769747613876306)** (2 条消息): 

- **火热招聘**: 一位成员称赞某次面试非常出色，认为它可以作为 **招聘** 工具。他们表示打算将视频分享给朋友，鼓励他们加入。
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1237741067801985054)** (14 条消息🔥): 

- **寻求 Mojo 字符串拼接解决方案**: 一位成员指出，与 Python 相比，Mojo 在拼接字符串时存在性能问题，Mojo 明显更慢。他们分享了示例代码，并正在寻找除并行化之外的提速建议。

- **Mojo 中短字符串优化的潜力**：一位用户回复称，目前正在 Mojo 的 `String` 结构体中实现短字符串优化（short string optimization），作为提高字符串拼接性能的可能解决方案，并提请关注 [GitHub issue #2467](https://github.com/modularml/mojo/issues/2467) 的进展。

- **使用 KeysContainer 进行性能优化**：针对次优的字符串拼接提出了性能改进建议，提议使用 `KeysContainer` 来避免过度的重新分配和内存复制（memcopies），并附带了相关 [GitHub 资源](https://github.com/mzaks/compact-dict/blob/main/string_dict/keys_container.mojo)的链接。

- **避免昂贵的 Int 到 String 转换**：另一位用户建议在查找过程中避免大量的 `Int` 到 `String` 转换，提议改用带有 `Int` 包装器的泛型字典（generic dict），因为 `Int` 到 `String` 的转换开销可能很大。

- **使用 StringBuilder 获得 3 倍性能提升**：在收到建议后，该成员报告称使用 [mojo-stringbuilder](https://github.com/maniartech/mojo-stringbuilder) 库以及针对 `Int` 键的 `Keyable` 包装器后，性能提升了 3 倍。虽然性能仍落后于 Python 和 Rust，但正在逐渐接近。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/2467">[Feature Request] Unify SSO between `InlinedString` and `String` type · Issue #2467 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并相信此请求符合优先级。你的请求是什么？我们目前有 https://docs.modular.com/mojo/stdlib...</li><li><a href="https://github.com/dorjeduck/minbpe.mojo">GitHub - dorjeduck/minbpe.mojo: port of Andrjey Karpathy&#39;s minbpe to Mojo</a>：将 Andrej Karpathy 的 minbpe 移植到 Mojo。通过在 GitHub 上创建一个帐户来为 dorjeduck/minbpe.mojo 的开发做出贡献。</li><li><a href="https://github.com/maniartech/mojo-stringbuilder">GitHub - maniartech/mojo-stringbuilder: The mojo-stringbuilder library provides a StringBuilder class for efficient string concatenation in Mojo, offering a faster alternative to the + operator.</a>：mojo-stringbuilder 库为 Mojo 提供了 StringBuilder 类，用于高效的字符串拼接，是 + 运算符的更快替代方案。 - maniartech/mojo-stringbuilder</li><li><a href="https://github.com/mzaks/compact-dict/blob/main/string_dict/keys_container.mojo">compact-dict/string_dict/keys_container.mojo at main · mzaks/compact-dict</a>：一个在 Mojo 🔥 中快速且紧凑的 Dict 实现。通过在 GitHub 上创建一个帐户来为 mzaks/compact-dict 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1237701607286767626)** (41 messages🔥): 

- **累积和函数（Cumulative Sum Function）亮点**：提供了一个指向 Modular Mojo 标准库文档中 `cumsum` 函数的链接，特别强调了它的存在和用法。该函数在 [Mojo 文档](https://docs.modular.com/mojo/stdlib/algorithm/reduction/cumsum)中有所引用。

- **命名规范与 PyConLT 轶事**：一位成员提到他们在 PyConLT 演讲时，曾对缩写 `cumsum` 有过短暂的心理挣扎，但最终选择不去纠结它。

- **Mojo 中发现严重 Bug**：在 `Tensor` / `DTypePointer` 标准库中发现了一个潜在的严重问题，详情见 [GitHub issue](https://github.com/modularml/mojo/issues/2591)。随后的讨论集中在这是构成一个 Bug，还是由于 `UnsafePointer` 缺乏生命周期（lifetime）而导致的特性。

- **指针与生命周期的困惑**：在使用 `DTypePointer` 配合 `memcpy` 时产生了困惑；指出 Tensor 被销毁了，但其数据赋值仍保留在一个非所有权指针中，这导致了悬空指针（dangling pointer）的情况。

- **Mojo Nightly 编译器更新**：宣布发布了新的 Mojo 编译器版本，合并了 31 个外部贡献。鼓励用户使用 `modular update nightly/mojo` 进行更新，并查看[此版本的 diff](https://github.com/modularml/mojo/pull/2593/files) 以及[自上次稳定版本以来的变更日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。

- **为提高清晰度更新 Tensor API**：`Tensor.data()` 调用更名为 `Tensor.unsafe_ptr()`，以更好地反映其行为，这一变化得到了讨论成员的普遍认可。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/2591">[BUG]: Weird behavior when passing a tensor as owned to a function · Issue #2591 · modularml/mojo</a>: Bug 描述：当将 tensor 作为 owned 传递给函数，并尝试在 @parameter 函数内部（使用 simd load）对数据进行 memcpy 或打印信息时，会出现一种奇怪的行为...</li><li><a href="https://github.com/modularml/mojo/pull/2593/files">[stdlib] Update stdlib corresponding to 2024-05-08 nightly/mojo  by JoeLoser · Pull Request #2593 · modularml/mojo</a>: 此 PR 使用与今天的 nightly 版本（mojo 2024.5.822）相对应的内部提交更新了 stdlib。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1237662713891655710)** (94 messages🔥🔥): 

- **模型目录结构已澄清**：成员们讨论了存储模型的正确目录。正确的设置包括一个顶层的 `\models` 文件夹，其中包含每个发布者及其对应仓库的子文件夹，类似于 Hugging Face 的命名规范。例如，模型文件应位于路径 `\models\lmstudio-community\Meta-Llama-3-8B-Instruct-GGUF\Meta-Llama-3-8B-Instruct-Q8_0.gguf`。

- **将 Web UI 连接到 LM Studio**：建议使用支持通用 OpenAI API 规范的 UI 来与 LM Studio 集成。AnythingLLM 和 [支持 Docker 的 Open WebUI](https://docs.openwebui.com/tutorial/openai) 被提及为与 LM Studio API 交互的可行选项。

- **LM Studio 的 GPU 需求说明**：讨论强调了 GPU VRAM 的必要性，并建议在遇到与 GPU 检测相关的错误时关闭 GPU 加速。此外还提到，该应用并非为 Windows ARM 笔记本电脑设计，因为目前不支持 ARM 架构。

- **使用 LM Studio 的挑战**：诸如 "unable to allocate backend buffer" 之类的错误意味着用户 PC 上的内存不足以运行本地模型。解释指出，本地模型通常需要 8GB 的 VRAM 和 16GB 的 RAM。

- **在 LM Studio 中发现本地模型**：为了在 LM Studio 中找到所有可用模型，有人分享了一个技巧：在模型搜索栏中搜索 "GGUF" 以列出所有模型。即使在用户不知道具体名称的情况下，这个变通方法也能帮助发现不同的模型。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/aha-gif-23490222">Aha GIF - Aha - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=LkJe1i3d7Ac">Solution - llama.cpp error: error loading model</a>：此视频分享了在 Windows 或 Linux 上使用 LM Studio 或任何其他 LLM 工具本地安装 AI 模型时出现以下错误的原因。"llama.cpp ...</li><li><a href="https://docs.openwebui.com/tutorial/openai">OpenAI API Endpoints | Open WebUI</a>：在本教程中，我们将演示如何使用环境变量配置多个 OpenAI（或兼容的）API 端点。此设置允许您轻松地在不同的 API 提供商之间切换...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1237664948860096553)** (22 messages🔥): 

- **更改模型路径**：一位成员询问另一位成员是否已将其模型路径更改为特定的 Repository 路径。
- **寻求视觉模型推荐**：有人请求推荐一款与 16GB RAM 和 8GB VRAM 兼容的视觉模型。
- **寻找最适合编程的模型**：一位成员请求推荐最适合编程用途的 8-9B 参数语言模型 (LLM)。
- **焦急等待更新**：成员们正热切期待能够解决现有问题的更新，其中一位成员特别对新的 quant（量化）方案作为潜在解决方案感兴趣。
- **对诗歌模型的挫败感**：一位成员对各种模型生成的诗歌表示不满，认为它们偏离了真实性，产生了一些荒谬的文本，并正在寻找能够准确完成诗歌的模型。

**提到的链接**: <a href="https://huggingface.co/ByteDance/Hyper-SD">ByteDance/Hyper-SD · Hugging Face</a>: 未找到描述

  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1237823859202461736)** (4 messages): 

- **对 AI 体验的便捷性表示赞赏**：一位成员承认 **LM Studio** 自早期版本以来已经有了长足的进步，并强调了这种对比：以前设置模型曾是挫败感的来源，而现在的体验更加“省心”。

- **Mac Studio 运行模型遇到问题**：**Arthur051882** 报告了一个关于其 192GB Mac Studio 的问题，该设备可以成功运行 *llama3 70B*，但在尝试运行 *llama1.6 Mistral* 或 *Vicuña* 时遇到错误。错误报告包含了内存、GPU 和 OS 版本的详细信息。

- **寻求技术协助**：该成员在同一线程中为其系统遇到的错误寻求帮助，并提供了额外的错误报告详情。

- **提供的管理与指导**：另一位成员 *Yagilb* 做出回应，指示 **Arthur051882** 在指定频道发布详细帖子，并注明导致加载问题的具体模型文件名。
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1238161425311207547)** (1 messages): 

- **RAG 架构的分块处理**：一位成员讨论了 **RAG (Retriever-Answer Generator) 架构**，提到了对文档进行分块（chunking）并附加各种数据（如文本 embeddings 和位置 metadata）的策略。他们建议这种方法有助于限制数据搜索范围，并确定在处理请求时应包含哪些文档分块。
- **智能分块选择可增强搜索**：该成员进一步提出，除了使用 metadata 缩小搜索范围外，还可以采用分析技术（例如基于 embeddings 和搜索词之间的余弦相似度进行 reranking）来精细化选择哪些分块。这表明有多种方法可以提高 AI 模型检索数据的相关性。
  

---


**LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1237663728170962954)** (28 messages🔥): 

- **Parsing 还是 Sparse，这是一个问题**：起初对一位成员使用的术语存在困惑，在 PDF 的语境下被推测为 *sparse* 或 **parse**。随后该成员澄清可能意指 **parse**，建议使用 **RAG application** 调用 LMStudio API 进行 PDF 的处理/搜索。

- **深入探究 Llama.cpp 的奥秘**：一位成员分享了一个 **GitHub issue 链接**，讨论了使用 **llama.cpp** 进行首次 inference 后的无效输出问题：[GitHub Issue #7060](https://github.com/ggerganov/llama.cpp/issues/7060)，该链接提供了对该问题的见解和潜在解决方案。

- **揭开瓶颈之谜**：一位用户报告称，在使用 **yi 30b q4 ks gguf** 模型时，尽管模型显示加载成功，但 CPU 和 GPU 利用率为零且运行缓慢。该报告引发了关于潜在瓶颈问题的讨论。

- **LLM 推理引擎：来自经验人士的学习建议**：一位成员提供了关于 **LLM inference engines** 运作方式的见解，指出性能问题可能与内存读取操作以及 **CPU-RAM** 与 **VRAM** 之间的速度差异有关。

- **Swap：计算流量中的慢车道**：针对成员关于其系统 VRAM 和模型 inference 期间利用率的性能问题，有人做了一个类比，说明了磁盘 swap 相对于 RAM 的缓慢。

**提到的链接**：<a href="https://github.com/ggerganov/llama.cpp/issues/7060">llava 1.5 invalid output after first inference (llamacpp server) · Issue #7060 · ggerganov/llama.cpp</a>: I use this server config: &quot;host&quot;: &quot;0.0.0.0&quot;, &quot;port&quot;: 8085, &quot;api_key&quot;: &quot;api_key&quot;, &quot;models&quot;: [ { &quot;model&quot;: &quot;models/phi3_mini_mod...

  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1237692737223131176)** (35 messages🔥): 

- **使用独立 GPU 进行潜在的桌面优化**：一位成员考虑将 **Intel HD 600** 系列 GPU 专门用于桌面任务，以释放 **GTX 980** 用于 LM Studio 等更密集的应用。然而，另一位成员指出，在只有 **6GB VRAM** 的情况下，将任务卸载到 HD 600 的收益可能微乎其微，仅能释放约 **500MB VRAM**。

- **解码 Llama 3 70B 的硬件需求**：在本地运行 **Llama 3 70B 模型**的热情引发了关于必要硬件的讨论，一位成员开玩笑说买一台 **128GB M3 MacBook Pro** 是处理此类负载的好借口。对于下一代硬件是否能运行理论上的 **400B 模型**，大家持怀疑态度，推测可能需要超过 **200GB 的 VRAM**。

- **离线模型使用的成本效益**：一位用户权衡了不使用 **ChatGPT** 等付费服务所节省的费用与本地运行模型的电费成本。有人建议了一个幽默的策略：使用笔记本电脑电池运行，并在 **Starbucks** 等地方免费充电。

- **为 Apple M1 选择合适的 LLM**：针对配备 **16GB RAM 的 M1 Pro** 应该选择哪种最佳**大语言模型 (LLM)** 的问题，推荐了 **"Llama 3 - 8B Instruct"** 作为兼容选项。

- **讨论语言模型之外的 AI 能力**：在硬件讨论中，一名成员询问了 **Apple 神经网络引擎 (neural engine)** 的功能，随后澄清其设计初衷是用于**人脸识别**和处理较小模型等任务，展示了 AI 在语言模型之外的更广泛应用。
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1237977603281780736)** (1 messages): 

- **Maxime Labonne 的 Self-Merge 杰作**：Maxime Labonne 完成了一个**传奇般的 120B self-merge**（基于 Llama 3 70B instruct），现已在 [lmstudio-community](https://huggingface.co/lmstudio-community/Meta-Llama-3-120B-Instruct-GGUF) 上线。该模型使用 *imatrix* 创建以增强性能，展现出显著的能力；鼓励用户进行测试并分享体验。
  

---


**LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1237772486460838029)** (4 messages): 

- **目前尚不支持编程方式的聊天交互**：一名成员根据文档询问是否可以通过 API **与现有聊天进行编程交互**。另一名成员确认该功能目前不可用，但提到已列入未来更新计划。
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1237692911047544832)** (50 messages🔥): 

- **早期实现的伦理问题**：成员们讨论了在原作者之前在 GitHub 上发布算法的非官方实现的礼仪。有人建议将仓库清晰地标记为非官方复现以避免混淆，而另一人指出许多论文从未发布代码，例如 *[MeshGPT Example](https://github.com/nihalsid/mesh-gpt)*。

- **婚后学术界更名**：一名用户询问如何在婚后更新学术平台上的姓氏，同时确保已发表的论文仍能与其关联。建议包括联系网站支持以及使用旧名字作为学术名 (nom-de-academia) 以避免问题。

- **加载 The Pile 数据的最佳实践**：参与者分享了为 AI 模型训练处理 The Pile 数据的技巧，包括在 Hugging Face 上寻找预处理版本，或处理应用了特定分词器 (tokenizer) 的 `.bin` 文件。

- **发现 EleutherAI 的 "Cookbook"**：提到了 EleutherAI 的 "cookbook"，透露其包含处理真实模型的实用细节和工具，一名成员对此表示惊讶和兴趣——*[GitHub Cookbook](https://github.com/EleutherAI/cookbook)*。

- **状态追踪的规模限制**：一篇暗示状态空间模型 (SSMs) 和 Transformers 在状态追踪方面具有共同局限性的文章引发了关于长序列反向传播 (backpropagation) 可扩展性以及 xLSTM 等新模型是否能解决这些问题的对话。

- **幽默的开源保险代码**：用户们进行了一场轻松的交流，发布了一些关于汽车保险公司运营的幽默伪代码片段，暗指理赔处理的随机性。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.08819">The Illusion of State in State-Space Models</a>: 与之前无处不在的 Transformer 架构相比，状态空间模型 (SSMs) 已成为构建大语言模型 (LLMs) 的一种潜在替代架构。一个理论上的...</li><li><a href="https://x.com/maxmbeck/status/1788115045085262231">Maximilian Beck (@maxmbeck) 的推文</a>: 敬请关注！🔜 #CodeRelease 💻🚀</li><li><a href="https://github.com/EleutherAI/cookbook">GitHub - EleutherAI/cookbook: Deep learning for dummies. 包含处理真实模型涉及的所有实用细节和有用工具。</a>: Deep learning for dummies. All the practical details and useful utilities that go into working with real models. - EleutherAI/cookbook</li><li><a href="https://github.com/nihalsid/mesh-gpt">GitHub - nihalsid/mesh-gpt: MeshGPT: 使用 Decoder-Only Transformers 生成三角形网格</a>: MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers - nihalsid/mesh-gpt</li><li><a href="https://blog.scottlogic.com/2023/11/24/llm-mem.html">LLM 微调内存需求</a>: LLM 训练的内存成本很高，但可以预测。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1237664351926620170)** (119 messages🔥🔥):

- **xLSTM 论文引发关注**：成员们对 xLSTM 论文表示怀疑，指出该论文在基线模型上似乎使用了较差的 hyperparameters，从而对其结论产生质疑。社区正等待通过独立实现或官方代码发布来进行进一步验证。

- **Function Vector (FV) 与 In-Context Learning**：讨论重点介绍了关于“function vector”的研究，该技术允许高效的 in-context learning，使模型仅凭紧凑的任务表示就能稳健地执行任务。相关见解源自详细阐述 function vectors 如何对上下文变化保持稳健的论文（[来源](https://arxiv.org/abs/2403.00835)）。

- **利用 YOCO 简化 KV Caches**：介绍了一种名为 YOCO 的新型 decoder-decoder 架构；它利用单次 KV cache 循环，而不是在各层之间复制 cache，从而可能优化内存使用。有人提出疑问，该模型是否能从进一步的优化中受益，例如在不增加内存负担的情况下对 KV caches 进行矩阵乘法。

- **探索多语言 LLM 认知**：一些成员寻求研究 Large Language Models (LLMs) 如何处理多语言和认知的论文，特别是 next tokens 的计算，以及它们在回答之前是否会在内部翻译成英文。对话中引用了几项相关研究，其中一些侧重于特定语言的神经元。

- **重新审视 Absolute Positional Encodings (APE)**：讨论并批评了 PoPE，这是一种使用正交多项式的 positional encoding 创新，其核心概念被过多的理论证明所掩盖。该方法有望改进基于正弦的 APEs，但其理论基础仍存疑问。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.05254">You Only Cache Once: Decoder-Decoder Architectures for Language Models</a>: 我们为大语言模型引入了一种名为 YOCO 的 Decoder-Decoder 架构，它仅缓存一次键值对（key-value pairs）。它由两个组件组成，即堆叠在 self-decoder 之上的 cross-decoder。...</li><li><a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: 我们探究了在不平衡且以英语为主的语料库上训练的多语言模型是否使用英语作为内部中转语言——这是一个对于理解语言模型如何运作至关重要的问题...</li><li><a href="https://arxiv.org/abs/2405.04585">PoPE: Legendre Orthogonal Polynomials Based Position Encoding for Large Language Models</a>: 针对原始 Transformer 中使用的基准绝对位置编码（APE）方法，已经提出了几项改进。在本研究中，我们旨在调查不充分的...</li><li><a href="https://arxiv.org/abs/2310.15916">In-Context Learning Creates Task Vectors</a>: 大语言模型（LLMs）中的上下文学习（ICL）已成为一种强大的新学习范式。然而，其底层机制仍未被充分理解。特别是，挑战在于...</li><li><a href="https://arxiv.org/abs/2404.11912">TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding</a>: 随着大语言模型（LLMs）近期被广泛应用于长内容生成，对高效长序列推理支持的需求日益增长。然而，键值（KV）缓存...</li><li><a href="https://blog.google/technology/ai/google-deepmind-isomorphic-alphafold-3-ai-model/">AlphaFold 3 predicts the structure and interactions of all of life’s molecules</a>: 我们的新 AI 模型 AlphaFold 3 能够以前所未有的准确度预测所有生命分子的结构和相互作用。</li><li><a href="https://arxiv.org/abs/2402.18815v1">How do Large Language Models Handle Multilingualism?</a>: 大语言模型（LLMs）在多种语言中表现出卓越的性能。在这项工作中，我们深入探讨了这个问题：LLMs 如何处理多语言能力？我们引入了一个框架来...</li><li><a href="https://arxiv.org/abs/2405.04517?fbclid=IwZXh0bgNhZW0CMTEAAR3SJmw76WJ1GHektDoTAmPU8BM_qhpCZIwKGznX-LTj6-MgOe4nnVQnvpY_aem_ARJ3QbHv6JJhM1EEIOZbO0ZZs3HjZMxWZdm4_GFrdv3WzWhu49t08YWcjVVk7dOoXcW2VnsTUlco597WXiNftkVc">xLSTM: Extended Long Short-Term Memory</a>: 在 20 世纪 90 年代，恒定误差轮转（constant error carousel）和门控（gating）作为长短期记忆网络（LSTM）的核心思想被引入。从那时起，LSTM 经受住了时间的考验，并为众多...</li><li><a href="https://arxiv.org/abs/2403.00835">CLLMs: Consistency Large Language Models</a>: 诸如 Jacobi 解码之类的并行解码方法在提高 LLM 推理效率方面展现出前景，因为它打破了 LLM 解码过程的顺序性，并将其转化为可并行的计算...</li><li><a href="https://openreview.net/forum?id=AwyxtyMwaG&noteId=AMoR1ZJPzF">Function Vectors in Large Language Models</a>: 我们报告了一种简单的神经机制的存在，该机制在自回归 Transformer 语言模型（LMs）中将输入-输出函数表示为一个向量。通过在...上使用因果中介分析</li><li><a href="https://github.com/mirage-project/mirage">GitHub - mirage-project/mirage: A multi-level tensor algebra superoptimizer</a>: 一个多级张量代数超优化器。通过在 GitHub 上创建账号来为 mirage-project/mirage 的开发做出贡献。</li><li><a href="https://github.com/microsoft/unilm/tree/master/YOCO">unilm/YOCO at master · microsoft/unilm</a>: 跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm</li><li><a href="https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/">ICLR 2024 Outstanding Paper Awards &#8211; ICLR Blog</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2405.04532">QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving</a>: 量化可以加速大语言模型（LLM）推理。除了 INT8 量化之外，研究界正在积极探索更低的精度，例如 INT4。尽管如此，最先进的...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

jacquesthibs: 每个 pythia 检查点都有经过调优的透镜（tuned lenses）吗？
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1237665484355403787)** (27 messages🔥):

- **动态与静态编译的困境**：讨论强调了 **torch.compile** 面临的挑战；有人建议将数据填充（padding）到一组预定义的形状中，但批处理序列中的动态形状使得静态编译的使用变得复杂。虽然 [*"dynamic=True"*](https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile) 据称可以避免持续的重新编译，但成员们发现它仍然倾向于频繁地重新编译。

- **深度 Diffusion 推理优化指南**：分享了一个由 9 部分组成的博客系列和一个 [GitHub 仓库](https://github.com/vdesai2014/inference-optimization-blog-post)，详细介绍了 Diffusion 模型的推理优化。这包括自定义 CUDA 内核以及对 GPU 架构的见解。

- **从 GPU 啸叫到 GPU 交响乐**：分享了一个独特的学习成果，即如何调制 GPU 电感啸叫来播放音乐，特别是《小星星》（*Twinkle Twinkle Little Star*），并提供了相关的 [代码](https://github.com/vdesai2014/inference-optimization-blog-post/blob/main/part-9%2Fgpu-piano%2Fgpu_piano.cu)。

- **通过复现进行高效学习**：对于 CUDA 初学者来说，一种实践方法是复现 [SGEMM_CUDA 教程](https://siboehm.com/articles/22/CUDA-MMM) 中的矩阵乘法内核并进行基准测试，这在处理 Diffusion 论文的推理优化之前非常有帮助。

- **对演讲的鼓励与兴趣**：社区成员对分享的优化工作表现出极大的热情，鼓励作者进行演讲，并继续就 ML 和机器人领域的实现及后续步骤提出问题。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.vrushankdes.ai/diffusion-inference-optimization">Diffusion Inference Optimization</a>：未找到描述</li><li><a href="https://siboehm.com/articles/22/CUDA-MMM">How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog</a>：在这篇文章中，我将迭代地优化一个用 CUDA 编写的矩阵乘法实现。我的目标不是构建一个 cuBLAS 的替代品，而是深入...</li><li><a href="https://github.com/vdesai2014/inference-optimization-blog-post/blob/main/part-9%2Fgpu-piano%2Fgpu_piano.cu">inference-optimization-blog-post/part-9/gpu-piano/gpu_piano.cu at main · vdesai2014/inference-optimization-blog-post</a>：通过在 GitHub 上创建一个账户来为 vdesai2014/inference-optimization-blog-post 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1237805482576711710)** (17 条消息🔥): 

- **Triton 的 GitHub 协作**：创建了一个 [新的 GitHub 资源](https://github.com/haileyschoelkopf/triton-index) 来收集社区编写的 Triton 内核链接，其灵感源于建立一个社区拥有的索引。有人表示有兴趣将其转移到 GitHub 上的 cuda-mode 组织。
- **教授 Triton 的倡议**：提到目前可用的 Triton 教程并不多，并计划将 PMPP 书籍移植以包含 Triton 代码片段。
- **Triton-Index 管理员权限已授予**：GitHub 上的 [cuda-mode Triton index](https://github.com/cuda-mode/triton-index) 已发送并接受了管理员邀请，旨在对已发布的 Triton 内核进行编目。
- **关于 Triton 内核数据集的想法**：讨论围绕将 Triton 内核作为数据集发布的可能性展开，虽然最初是玩笑话，但似乎引起了真正的兴趣。
- **理解 Triton 的编程模型**：回答了关于 Triton 在 Warp 和线程调度方面与 CUDA 的比较；强调了 Triton 和 CUDA 中的 Block 是一样的，但 Triton 为程序员抽象掉了 Warp 和线程层级的细节。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=DdTsX6DQk24">第 14 课：Triton 实战指南</a>: https://github.com/cuda-mode/lectures/tree/main/lecture%2014</li><li><a href="https://github.com/cuda-mode/triton-index">GitHub - cuda-mode/triton-index: 编目已发布的 Triton kernel。</a>: 编目已发布的 Triton kernel。通过在 GitHub 上创建账号为 cuda-mode/triton-index 的开发做出贡献。</li><li><a href="https://github.com/haileyschoelkopf/triton-index/tree/main">GitHub - haileyschoelkopf/triton-index: 请改看 https://github.com/cuda-mode/triton-index/！</a>: 请改看 https://github.com/cuda-mode/triton-index/！ - haileyschoelkopf/triton-index</li><li><a href="https://www.youtube.com/watch?v=DdTs"> - YouTube</a>: 未找到描述</li><li><a href="https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/">使用 CUDA Warp 级原语 | NVIDIA 技术博客</a>: NVIDIA GPU 以 SIMT（单指令多线程）方式执行被称为 warp 的线程组。许多 CUDA 程序通过利用 warp 执行来实现高性能。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1237751205686677625)** (47 条消息🔥): 

- **关于归一化技术的性能讨论**：一位 Torch 用户提出了关于在 NHWC 格式下进行张量归一化时的性能影响问题，询问是否比 NHWC 优化的算法更倾向于转换回 NCHW。另一位用户回应并重点介绍了一个 [PyTorch 教程](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)，该教程指出 channels-last 格式可以带来 20% 的性能提升，且由于步幅（stride）操作，转换可能并不会产生额外开销。

- **LibTorch 编译时间的改进**：关于缩短编译时间的讨论发现，使用 ATen 的 `at::native::randn` 代替 `torch::randn` 并包含 `<ATen/ops/randn_native.h>` 而非 `<ATen/ATen.h>`，可以将用户的编译时间从大约 35 秒减少到 4.33 秒。

- **libtorch 与 cpp 扩展**：Marksaroufim 建议探索使用 cpp 扩展来将 C++ 与 PyTorch 集成，理由是现在研究 libtorch 的人并不多。此外，他提到 AOT Inductor 是一种提前生成 `.so` 文件的方法，详见 [torch export 教程](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html)。

- **澄清 ATen 和 libtorch 的区别**：通过交流，大家明确了 ATen 被视为后端，而 libtorch 被视为前端。寻求建议的用户意识到他们只需要张量（tensor）功能而不需要整个 libtorch，这显著改进了他们的工作流。

- **关于 AOTInductor 模型缓存的查询**：Benjamin_w 询问了如何使用 AOT Inductor 缓存已编译的模型以避免重新编译延迟。Marksaroufim 提供了一个潜在方案，即设置 `torch._inductor.config.fx_graph_cache = True`，并提到正在努力优化热编译（warm compile）时间。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html">(beta) PyTorch 中的 Channels Last 内存格式 — PyTorch Tutorials 2.3.0+cu121 文档</a>: 未找到描述</li><li><a href="https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html?highlight=aotinductor">torch.export 教程 — PyTorch Tutorials 2.3.0+cu121 文档</a>: 未找到描述</li><li><a href="https://github.com/pytor">pytor - 概览</a>: pytor 有一个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html">torch.export 教程 — PyTorch Tutorials 2.3.0+cu121 文档</a>: 未找到描述</li><li><a href="https://github.com/pytorch/pytorch/pull/103281">msaroufim 在 Python 中加载 AOT Inductor · Pull Request #103281 · pytorch/pytorch</a>: 现在如果你运行带有 TORCH_LOGS=output_code python model.py 的 model.py，它将打印一个 tmp/sdaoisdaosbdasd/something.py，你可以像模块一样导入它。还需要设置 &#39;config....</li><li><a href="https://github.com/pytorch/pytorch/wiki/PyTorch-dispatcher-walkthrough">PyTorch 调度器（dispatcher）演练</a>: Python 中具有强大 GPU 加速功能的张量和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/utils/cpp_extension.py#L1018">pytorch/torch/utils/cpp_extension.py at main · pytorch/pytorch</a>: Python 中具有强大 GPU 加速功能的张量和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1238058045737664572)** (3 条消息):

- **vAttention - 重新定义 KV-Cache 内存管理**：一篇新论文提出了 **vAttention**，这是一种用于大语言模型（LLM）推理的先进内存管理系统，旨在动态分配 KV-cache 内存，以解决 GPU 中的内存碎片问题。据称该系统取代了以往的静态分配方法，后者曾导致容量浪费和软件复杂性（[阅读摘要](https://arxiv.org/abs/2405.04437)）。

- **QServe - 通过量化提升 LLM 推理性能**：QServe 推理库拥有一种新的量化算法 **QoQ (W4A8KV4)**，旨在解决现有 INT4 量化技术面临的效率问题，特别是针对 GPU 上显著的运行时开销。这一进展对于增强大批量、基于云的 LLM 服务性能至关重要（[阅读摘要](https://arxiv.org/abs/2405.04532)）。

- **CLLMs 实现更快速的推理**：一篇博客文章揭示了**一致性大语言模型（CLLMs）**，改变了 LLM 作为顺序解码器的传统观点，证明了 LLM 可以有效地作为并行解码器运行。研究表明，这些模型通过并行解码多个 token，可以显著降低推理延迟，这可能反映了人类形成句子的认知过程（[探索博客与研究](https://hao-ai-lab.github.io/blogs/cllm/)）。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.04437">vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention</a>：高效利用 GPU 内存对于高吞吐量的 LLM 推理至关重要。之前的系统会提前为 KV-cache 预留内存，由于内部碎片导致容量浪费。在...</li><li><a href="https://arxiv.org/abs/2405.04532">QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving</a>：量化可以加速大语言模型（LLM）推理。除了 INT8 量化，研究界正在积极探索更低精度，如 INT4。尽管如此，现有的...</li><li><a href="https://hao-ai-lab.github.io/blogs/cllm/">Consistency Large Language Models: A Family of Efficient Parallel Decoders</a>：摘要：LLM 传统上被视为顺序解码器，逐个解码 token。在本博客中，我们展示了预训练的 LLM 可以轻松地被教会作为高效的并行解码器运行...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1238078012835237998)** (4 条消息): 

- **Diffusion Model 推理获得大幅加速**：[Vrushank Desai 的博客系列](https://www.vrushankdes.ai/diffusion-inference-optimization)详细阐述了如何优化 Diffusion Models 的推理延迟，专门针对丰田研究所（Toyota Research Institute）论文中的 U-Net 进行 GPU 架构调整以实现加速。配套的 [GitHub 代码](https://github.com/vdesai2014/inference-optimization-blog-post)提供了实用的见解。
- **使用 Superoptimizer 寻找最快的 DNN**：[Twitter](https://twitter.com/JiaZhihao/status/1788624949344702953) 上讨论的一个新工具据称可以作为“超级优化器（superoptimizer）”，通过搜索最快的 Triton 程序来优化任何 DNN，详见 [Mirage 的论文](https://www.cs.cmu.edu/~zhihaoj2/papers/mirage.pdf)。
- **对 “Superoptimizer” 基准测试的质疑**：一名成员对 Mirage 项目论文中关于 DNN “超级优化器”展示的基准测试表示怀疑，这与其他人的怀疑态度一致。
- **批评 Mirage 的方法论**：怀疑态度还延伸到了 Mirage 在其方法中忽略了自动调优（autotuning），尽管该项目专注于识别最优融合策略，这一点在其 [GitHub 演示页面](https://github.com/mirage-project/mirage/blob/main/demo/demo_lora)上被发现。

**提及的链接**：<a href="https://www.vrushankdes.ai/diffusion-inference-optimization">Diffusion Inference Optimization</a>：未找到描述

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1237952987398148098)** (6 条消息): 

- **SetFit 模型 ONNX 导出疑问**：一名成员询问了使用 SetFit 的导出方法与 PyTorch 的 `torch.onnx.export` 将 HuggingFace SetFit 模型导出为 ONNX 之间的区别。该过程涉及 [SetFit 仓库中的一个笔记本示例](https://github.com/huggingface/setfit/blob/main/notebooks/onnx_model_export.ipynb)用于导出模型。

- **PyTorch 代码导出等效性查询**：他们还分享了一段使用 `torch.onnx.export` 的代码片段，并思考这是否会产生与 `SetFit::export_onnx` 方法相同的 ONNX 输出。

- **ONNX 到 TensorRT 转换澄清**：成员寻求确认在运行 `trtexec` 将其编译为 TensorRT plan 文件之前，是否必须先创建一个 ONNX 模型，并分享了一个用于 `trtexec` 转换的示例命令。

- **对 Torch Compile 选项的困惑**：`torch.compile` 中提到的 `tensorrt` 选项引发了成员的困惑，即它与之前提到的 ONNX 到 TensorRT 转换过程之间的关系。

- **关于在目标 GPU 上进行转换需求的推测**：有人询问 `trtexec` 命令是否需要在打算使用 TensorRT plan 的目标 GPU 上运行。
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1237876945916530821)** (2 条消息): 

- **Apple 发布 M4 芯片**：Apple 推出了 [M4 芯片](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/)，为新款 iPad Pro 带来了每秒 **38 万亿次操作** 的性能提升。M4 采用 3 纳米技术和 10 核 CPU，提升了能效并驱动了全新的 Ultra Retina XDR 显示屏。
- **Panther Lake 的性能竞争**：一项对比指出 Panther Lake 每秒可执行 **175 万亿次操作**，展示了其强大的计算能力。

**提到的链接**：<a href="https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/">Apple introduces M4 chip</a>：Apple 今日发布了 M4，这是 Apple 设计的最新芯片，为全新的 iPad Pro 提供了惊人的性能。

  

---


**CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 条消息): 

seire9159: 芝加哥有没有人想一起看视频并编写一些 CUDA 代码。
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1237787241787559986)** (40 条消息🔥): 

- **ZeRO++ 承诺“少即是多”**：[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/) 为大模型训练提供了 4 倍的通信量减少，显著加快了训练过程。通信前对权重和梯度的量化对模型收敛的影响微乎其微。

- **MPI Bug 已修复**：一次合并解决了某个问题，现在 master 分支上的多 GPU 训练可以正常运行。

- **所有 GPT-2 模型现在均可训练**：已推送更新，允许通过 `--model` 标志选择任何 GPT-2 模型大小，并可指定特定的 `.bin` 文件进行训练。在 A100 GPU 上可以进行 Batch size 4 的训练，但在扩展到 4x A100 GPU 时性能会有所下降。

- **梯度累积 (Gradient Accumulation) 正在开发中**：讨论集中在需要梯度累积来提高 GPU 扩展效率。尽管有好处，但由于提案 PR 中的复杂性和可读性等实现问题，目前尚未立即采用。

- **Layernorm Backward 精度难题**：有人提出了一个问题，即 `layernorm_backward` kernel 在不支持 bf16 的硬件上执行 bfloat16 (bf16) 算术，导致编译问题。建议使用 `#if (CUDART_VERSION >= 12000)` 进行条件编译修复，以便仅在 CUDA 12 及以上版本中包含该有问题的 kernel。

**提到的链接**：<a href="https://www.deepspeed.ai/tutorials/zeropp/">ZeRO++</a>：ZeRO++ 是构建在 ZeRO 之上的一套通信优化策略，无论规模或跨设备带宽限制如何，都能为大模型训练提供无与伦比的效率。R...

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1237810355649122365)** (56 条消息🔥🔥):

- **开源与预售讨论**：一位成员讨论了 **01 light** 已开放预售，并指出其硬件和软件均为开源。
- **微软的 VoT 通过空间推理挑战 LAM**：分享了一个 YouTube 视频链接，展示了微软的 “**Vision of Thought**” (VoT)，它为 LLM 赋予了空间推理能力，据创作者称，其表现优于 OpenAI 的 LAM。该成员表示期待能有更安全的版本与家人分享。
- **API keys 配置难题**：几位成员讨论了配置 **environment variables**（环境变量）和 **API keys** 时的挑战和解决方法。建议参考 [litellm documentation](https://litellm.vercel.app/docs/providers/groq) 的说明，并将 `GROQ_API_KEY` 设置为环境变量。
- **AI 驱动的计算机交互创新**：成员们讨论了利用 **OpenCV/Pyautogui** 等工具，使用 GPT-4 等模型在不同操作系统环境（Ubuntu, Mac）中执行任务的系统。
- **结合开源 AI 工具**：提议将 Open Interpreter 与开源 AI 眼镜 “**Billiant Labs Frame**” 集成，并分享了一个介绍该眼镜的 [YouTube 视频](https://www.youtube.com/watch?v=OS6GMsYyXdo)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://litellm.vercel.app/docs/providers/groq">Groq | liteLLM</a>: https://groq.com/</li><li><a href="https://www.youtube.com/watch?v=OS6GMsYyXdo">This is Frame! Open source AI glasses for developers, hackers and superheroes.</a>: 这是 Frame！面向开发者、黑客和超级英雄的开源 AI 眼镜。首批客户将于下周开始收到产品。我们迫不及待想看到...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/core/computer/display/point/point.py">open-interpreter/interpreter/core/computer/display/point/point.py at main · OpenInterpreter/open-interpreter</a>: 计算机的自然语言界面。通过在 GitHub 上创建账号为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://youtu.be/JSRBesvBCtI?feature=shared">&quot;VoT&quot; Gives LLMs Spacial Reasoning AND Open-Source &quot;Large Action Model&quot;</a>: 微软的 &quot;Visualization of Thought&quot; (VoT) 赋予了 LLM 空间推理能力，这在以前对 LLM 来说几乎是不可能的。此外，还有一个新的...
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1237735527826849822)** (74 条消息🔥🔥): 

- **将 LiteLLM 连接到各种 AI 提供商**：LiteLLM 文档指定支持多个 AI 提供商，包括 OpenAI 模型、Azure、Google 的 PaLM 和 Anthropic。[LiteLLM providers documentation](https://litellm.vercel.app/docs/providers) 中列出了具体的说明和先决条件。

- **Groq Whisper API 热议**：Groq 的 API 中加入 Whisper 引发了关于将其与 01 集成的讨论，并指出 Groq 的免费 API 可能具有优势。**Groq's API** 因免费且高效而受到称赞，但用户在 [模型访问方面遇到了问题](https://console.groq.com/docs/speech-text)。

- **Gemini Pro Vision 被吹捧为免费替代方案**：用户讨论将 Gemini Pro Vision 作为一个免费且可能易于集成的 API。未提供具体的文档链接。

- **01 在 Windows 上的烦恼**：用户报告称 **01 在 Windows 平台上的功能受限**，但存在涉及 Windows 10 代码修改以及检查 Windows 11 兼容性更新的潜在修复方案。

- **开源 AI 的难题**：几位成员对像 01 这样的开源 AI 系统可能无法完全支持开源 AI 模型表示惊讶，并就专有 AI 模型与开源 AI 模型的可行性展开了辩论。

**提到的链接**：<a href="https://litellm.vercel.app/docs/providers">Providers | liteLLM</a>: 了解如何在 LiteLLM 上部署和调用来自不同提供商的模型

  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1237749046781935647)** (2 条消息):

- **GPT-4 与 YouTube 联动**：一名成员通过自定义指令让 **GPT-4** 利用 YouTube 进行音乐推荐，展示了该模型遵循特定用户指令的能力。
- **本地模型性能评估**：他们测试了多个本地模型，例如 TheBloke/deepseek-coder-33B-instruct.GGUF，但该模型因声称*无网络连接*而未能达到预期。另一个使用 lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF 的尝试也表现平平，在一次重试失败后便放弃了。
- **mixtral-8x7b 取得成功**：mixtral-8x7b-instruct-v0.1.Q5_0.gguf 提供了目前为止最好的性能，几乎正确地执行了任务，表现优于 deepseek，尤其是在配备了自定义调优的 2080 ti GPU 和 32GB DDR5 6000 内存的系统上。
- **Macbook Pro 在运行本地模型时力不从心**：该成员的 MacBook Pro 被证明甚至无法胜任轻量级的本地模型操作，已不再用于此类任务。

---

**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1237731520056131604)** (38 messages🔥): 

- **DALLE-3 级别的输出指日可待**：成员们讨论了通过结合 Pixart Sigma + SDXL + PAG 来实现与 DALLE-3 相当的输出，尽管 Pixart 在处理文本和单个物体方面仍有困难。据说这种 *ComfyUI 工作流* 能够实现离线的 **提示词对齐（prompt alignment）和质量**，并被认为具有进一步改进的潜力。
  
- **对微调和技术专业知识的需求**：社区认为微调可以解决图像生成中的构图问题，并指出*技术头脑*可以将当前的成就推向新高度。一位用户承认自己技术能力较弱，希望其他人能为改进做出贡献。

- **揭秘模型质量的突破**：一位成员**发现了提升模型质量的关键拼图**——确保微调节（microconditioning）输入范围易于管理以降低学习难度，并提出了优化该过程的具体参数和策略。

- **保险数据自动化**：一位商业汽车保险核保师咨询了关于**自动化数据处理**的开源工具和策略，以提高评估风险和管理理赔的效率与准确性。

- **AI 竞赛公告**：一位 AI 研究工程师宣布了在 IJCAI 期刊举办的竞赛，并为那些对**发表论文和赢取奖金**感兴趣的人分享了链接。该竞赛似乎与 AI 在科学领域的应用有关。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://futurism.com/advon-ai-content">认识 AdVon，感染媒体行业的 AI 驱动内容怪兽</a>：我们对 AdVon Commerce 的调查，这家 AI 承包商是《今日美国》和《体育画报》丑闻的核心。</li><li><a href="https://youtu.be/NwZufAJxmMA">具有扁平化与重构功能的 SNAC</a>：纯语音编解码器：https://colab.research.google.com/drive/11qUfQLdH8JBKwkZIJ3KWUsBKtZAiSnhm?usp=sharing 通用（32khz）编解码器：https://colab.research.g...</li><li><a href="https://aistudio.baidu.com/projectdetail/7459168">IJCAI 2024: Rapid Aerodynamic Drag Pred - 飞桨AI Studio星河社区</a>：未找到描述
</li>
</ul>

</div>

---

**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1237685861391532034)** (59 messages🔥🔥): 

- **提及 IC-Light GitHub 仓库**：一位用户分享了 IC-Light GitHub 仓库的链接，这是一个专注于“更多重光照（relighting）！”的开源项目。仓库地址为 [GitHub - lllyasviel/IC-Light](https://github.com/lllyasviel/IC-Light)。
- **建议避免垃圾信息行为**：一条消息因在多个频道复制粘贴而被标记为疑似垃圾信息，原发布者已确认收到通知。
- **关于噪声条件评分网络（Noise Conditional Score Networks）的辩论**：用户们就噪声条件评分网络和 DDPMs 中的噪声调度展开了技术讨论，质疑其是否收敛到标准高斯分布。
- **探索 K-diffusion 和基于评分的模型**：引用了来自 arxiv.org 的“K-diffusion”等多篇论文，用户们交流了对这些模型中数学概念的理解，支持这些概念紧密相关但在 ODE 求解器使用等细节上有所不同的观点。
- **DDIM 中的方差爆炸与方差保持**：关于扩散模型中“方差爆炸（variance exploding）”概念的澄清对话，结论是它与受 DDPM 启发的 ODE 求解器有关，并将其与 rectified flows 等其他方法进行了比较。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2206.00364">Elucidating the Design Space of Diffusion-Based Generative Models</a>：我们认为基于扩散的生成模型的理论和实践目前过于复杂，并试图通过展示一个清晰分离...的设计空间来补救这一现状。</li><li><a href="https://yang-song.net/blog/2021/score/#mjx-eqn%3Ainverse_problem">Generative Modeling by Estimating Gradients of the Data Distribution | Yang Song</a>：未找到描述</li><li><a href="https://github.com/lllyasviel/IC-Light">GitHub - lllyasviel/IC-Light: More relighting!</a>：更多重光照！通过在 GitHub 上创建账户来为 lllyasviel/IC-Light 的开发做出贡献。
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1237817596003487924)** (1 条消息): 

- **deeplearning.ai 推出“使用 LlamaIndex 构建 Agentic RAG”课程**：一门名为“Building Agentic RAG with LlamaIndex”的新课程已在 [deeplearning.ai](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex) 上线，由 LlamaIndex 的 CEO Jerry Liu 授课。该课程将涵盖 **routing**（路由）、**tool use**（工具使用）以及**带有工具使用的多步推理**，以赋能 Agent 获取信息并实现**复杂问答**。

**提到的链接**：<a href="https://x.com/AndrewYNg/status/1788246239517282795">Andrew Ng (@AndrewYNg) 的推文</a>：我很高兴启动我们第一个专注于 Agent 的短课程，从由 @jerryjliu0（@llama_index 的 CEO）教授的《使用 LlamaIndex 构建 Agentic RAG》开始。这涵盖了一个重要的转变...

  

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1237935029162610752)** (2 条消息): 

- **Agentic RAG 课程发布**：LlamaIndex 宣布与 @DeepLearningAI 和 @AndrewYNg 合作，提供一门关于**构建 Agentic RAG** 的新课程，教授如何创建一个能够理解跨多个文档的复杂问题的自主研究助手。感兴趣的用户可以通过提供的 [Twitter 链接](https://twitter.com/llama_index/status/1788375753597567436)了解更多信息并报名。

- **在本地更快地运行 LLM**：LlamaIndex 集成了一项服务，允许快速运行**本地大语言模型 (LLMs)**，支持 Mistral、Gemma、Llama、Mixtral 等一系列模型，涵盖包括 NVIDIA 在内的各种架构。更多详情可通过其 [Twitter 帖子](https://twitter.com/llama_index/status/1788627219172270370)获取。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1237666009801035836)** (55 条消息🔥🔥): 

- **Vector Store 中的 Embedding 困扰**：一位成员表示在使用 **LlamaIndex** 构建本地 RAG 系统（使用来自 **ollama** 的 nomic embeddings）时遇到了困难，原因是与 **chroma** 自有的 embeddings 存在兼容性问题。给出的建议是，如果使用 Vector Store 集成，**LlamaIndex** 会自动处理 Embedding。
- **助力社区成员的项目**：一位成员呼吁大家支持他在 LinkedIn 上发布的关于使用 **LlamaIndex** 的 LlamaParse 的帖子。
- **Rerank 与元数据困境**：成员们讨论了 `rerank` 模型和 `MetadataReplacementPostProcessor` 的执行顺序，结论是该顺序取决于用户是希望基于原始文本还是替换后的元数据进行 Rerank。
- **托管本地 LLM 模型**：讨论涵盖了企业托管本地 LLM 模型的情况，包括 **AWS**、**K8s 上的自动扩缩容**以及 *vLLM* 或 *TGI* 等选项。共识倾向于生产环境需要可扩展性。
- **工具执行故障排除**：一位遇到 _CBEventType.SUB_QUESTION 未被调用问题的用户收到了指向实现缺陷的回复，并获得了一些代码片段指导以尝试解决。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/aurelio-labs/semantic-router">GitHub - aurelio-labs/semantic-router: Superfast AI decision making and intelligent processing of multi-modal data.</a>: 超快速的 AI 决策和多模态数据智能处理。 - aurelio-labs/semantic-router</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/indices/knowledge_graph/">Knowledge graph - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/?h=react">ReAct Agent - A Simple Intro with Calculator Tools - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/vllm/?h=vllm#completion-response">vLLM - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval/">Structured Hierarchical Retrieval - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1238086390575530074)** (2 条消息): 

- **Languify.ai 发布**: 一个新的浏览器扩展程序 **Languify.ai** 已发布，旨在优化网站上的文本以增加用户参与度和销售额，利用 OpenRouter 根据 Prompt 选择模型。该扩展提供免费的个人计划和每月 10.99 欧元的商业计划，可在 [www.languify.ai](https://www.languify.ai/) 找到。

- **AnythingLLM 的更简单替代方案**: 一位成员对 **Languify.ai** 表示了兴趣，认为它比 AnythingLLM 更简单，后者对他们的需求来说过于复杂，并表示将尝试这个新扩展。

**提到的链接**: <a href="https://www.languify.ai/">Languify.ai - Optimize copyright</a>: 通过 Languify 提升内容的触达率，这是我们用户友好的浏览器扩展。由 AI 驱动，它能无缝优化文案，增强参与度并放大您的创意影响力。

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1237663744524554250)** (53 条消息🔥): 

- **关于 OpenRouter 信息获取的困惑**: 一位用户表示难以找到关于 **OpenRouter** 的非技术信息，尽管阅读了文档，但对于路由如何工作、什么是 Credits 以及某些模型的实际免费状态仍有疑问。频道中没有提供明确的答案或资源来解决这些疑虑。

- **寻找 Llama 3 审核服务**: 一位频道成员在寻找用于审核的 **Llama 3 Guard API**，在注意到 OpenRouter 的产品中没有该服务后，被引导至 Together.ai 作为服务提供商。

- **Min-P 支持查询和变通方法**: 关于哪些提供商支持 `min_p` 的讨论明确了由 Together、Lepton、Lynn 和 Mancer 提供的模型确实提供此支持，尽管 Together 遇到问题，Lepton 仍能正常运行。

- **模型托管和 DeepSeek 版本谜题**: 频道用户询问了 **DeepSeek v2** 的托管选项，有报告显示，除了 DeepSeek 自己的 32k Context 模型外，目前似乎没有其他提供商在托管它。

- **越狱 LLM Wizard 8x22B**: 频道成员讨论了越狱 **Wizard 8x22B** 以解锁更多未经审查内容的潜力，并将其与现有的安全内容限制进行了对比。分享了关于理解 LLM 拒绝机制的研究链接，例如[这篇 Alignment Forum 帖子](https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)，以阐明 LLM 如何处理违规请求。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai).">未找到标题</a>：未找到描述</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-70b-instruct">Meta: 由 meta-llama 提供的 Llama 3 70B Instruct | OpenRouter</a>：Meta 最新的模型类别 (Llama 3) 推出了多种尺寸和版本。这个 70B 指令微调版本针对高质量对话场景进行了优化。它展示了强大的...</li><li><a href="https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction>">LLM 中的拒绝行为是由单一方向介导的 — AI Alignment Forum</a>：这项工作是 Neel Nanda 在 ML Alignment & Theory Scholars Program - 2023-24 冬季班研究方向的一部分，由...共同指导。</li><li><a href="https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-dire">LLM 中的拒绝行为是由单一方向介导的 — AI Alignment Forum</a>：这项工作是 Neel Nanda 在 ML Alignment & Theory Scholars Program - 2023-24 冬季班研究方向的一部分，由...共同指导。</li><li><a href="https://openrouter.ai/docs#custom-provider-selection">OpenRouter</a>：构建与模型无关的 AI 应用
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1237673209118068786)** (43 条消息🔥): 

- **TypeScript 中的工具包困扰**：成员们正在讨论 TypeScript 中 `JsonOutputFunctionsParser` 的问题，出现了 *Unterminated string in JSON* 错误。建议的检查包括验证 `get_chunks` 函数返回的 JSON，确保传递给 `chain.invoke()` 的 `content` 格式正确，以及检查 `getChunksSchema`。

- **CLAUDIA 集成好奇心**：有人提出了关于将 **claude-haiku** 与 **Langchain ChatVertexAI** 集成的问题，但未提供具体细节或后续跟进。

- **多 Agent 项目探索**：一位成员正在询问类似于 **STORM**（多 Agent 文章生成器）的示例项目，表明需要多 Agent 系统参考。

- **Agent 架构焦虑**：一位成员对为特定功能实例化不同 Agent 表示担忧，讨论了这种方法与随时间演进的基于组件的架构相比的效率。

- **为预算优化 Vector DB**：**VertexAI Vector store** 设置的效率和复杂性受到质疑，成员正在寻找 Vector 数据库的高性价比替代方案，并提到 Pinecone 或 Supabase 是更简单的选择。
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1238067015869141063)** (1 条消息): 

- **排查 Chain Invoke 问题**：一位成员在通过字典输入调用 chain 时遇到差异；在 Python 中使用 `chain().invoke({ <dict> })` 运行正常，但通过 `/invoke` 端点调用时以空字典开始并失败。他们发布了 chain 定义代码，并寻求关于为什么两种方法的 **Chain run** 行为不同的见解。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1237730755849748531)** (3 条消息): 

- **Gianna - 彻底改变虚拟助手交互**：新的虚拟助手框架 **Gianna** 旨在通过模块化和可扩展的设计彻底改变 AI 交互。它使用 **CrewAI** 和 **Langchain** 提供智能，可以在 [GitHub](https://github.com/marvinbraga/gianna) 上找到或通过 PyPI 安装。

- **Langgraph 简化客户支持**：一篇新的 Medium 文章讨论了 **Langchain** 的 **LangGraph** 如何帮助简化客户支持，指导读者如何将此工具集成到他们的系统中。全文可在此处[访问](https://ai.gopubby.com/streamline-customer-support-with-langchains-langgraph-8721c250809e)。

- **Athena - 自主 AI 数据 Agent**：**Athena** 展示了一个由 Langchain 和 Langgraph 编排的 AI 数据平台和 Agent，现在能够在数据工作流中实现完全自主。Athena 功能的演示视频可在 [YouTube](https://www.youtube.com/watch?v=CXmwYk5Hbig) 上观看。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=CXmwYk5Hbig">企业级 AI 数据分析师 | AI Agent | Athena Intelligence</a>：未找到描述</li><li><a href="https://ai.gopubby.com/streamline-customer-support-with-langchains-langgraph-8721c250809e">使用 Langchain 的 LangGraph 简化客户支持</a>：Ankush k Singal
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1237774186391535717)** (2 条消息):

- **探索附加价值**：一位成员询问了某个未指明主题或工具的**益处**，可能是想进一步了解其用途或潜在影响。

- **将 CrewAI 与加密货币市场集成**：分享了一个名为[“创建自定义工具将 crewAI 连接到 Binance 加密货币市场”](https://youtu.be/tqcm8qByMp8)的教程视频链接，指导用户如何使用 **crewAI CLI** 连接 **Binance.com** 加密货币市场以获取财务见解。

**提到的链接**：<a href="https://youtu.be/tqcm8qByMp8">Create a Custom Tool to connect crewAI to to Binance Crypto Market</a>：使用新的 crewAI CLI 工具并添加自定义工具将 crewAI 连接到 binance.com 加密货币市场。然后获取钱包中的最高持仓并进行网络搜索...

  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1237677556912885800)** (39 条消息🔥): 

- **斯坦福发布新知**：斯坦福大学通过 [YouTube 视频](https://youtu.be/XZ0PMRWXBEU)发布了全新的 2023 年“深度生成模型 (Deep Generative Models)”课程，为 AI 爱好者和从业者提供了最新的教育素材。
- **寻找高端 GPU**：成员们正在分享租用 A100/H100 GPU 数周的技巧，并推荐了 [sfcompute](https://sfcompute.com) 以实现此目的。
- **开发者辩论 AI 辅助编程**：关于 AI 辅助编程展开了热烈讨论，用户们分享了对 Lisp 编程的怀旧之情，并辩论了各种 AI 代码辅助工具的优缺点。
- **Gradient 发布长上下文 AI 模型**：重点介绍了 Gradient 的 Llama-3 8B Instruct 模型，该模型将上下文长度从 8k 扩展到了 4194k，并邀请用户[加入自定义 Agent 的等待名单](https://forms.gle/L6TDY7dozx8TuoUv7)。
- **OpenAI 的安全 AI 基础设施**：OpenAI 最近的一篇博客文章讨论了高级 AI 的安全基础设施，包括加密签名的 GPU，这引发了讨论，一些成员将这些措施视为“保护主义”。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llm-ui.com/">llm-ui | 适用于 LLM 的 React 库</a>：适用于 React 的 LLM UI 组件</li><li><a href="https://youtu.be/XZ0PMRWXBEU?si=IJPKQYv1qCAVDtVD">Stanford CS236: Deep Generative Models I 2023 I Lecture 1 - Introduction</a>：有关斯坦福人工智能项目的更多信息，请访问：https://stanford.io/ai。要跟随课程学习，请访问课程网站...</li><li><a href="https://wow.groq.com">使用实时 AI 解决方案加速系统 - Groq</a>：Groq 为开发者提供高性能 AI 模型和 API 访问。以比竞争对手更低的成本获得更快的推理速度。立即探索用例！</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-4194k">gradientai/Llama-3-8B-Instruct-Gradient-4194k · Hugging Face</a>：未找到描述</li><li><a href="https://news.ycombinator.com/item?id=40302698">无标题</a>：未找到描述</li><li><a href="https://x.com/gordonwetzstein/status/1788239400025088501?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Gordon Wetzstein (@GordonWetzstein) 的推文</a>：很高兴分享我们新的 Nature 论文！在这项工作中，我们提出了一种新的显示设计，将逆向设计的超表面波导与 AI 驱动的全息显示相结合，以实现全色 3D 增强现实...</li><li><a href="https://buttondown.email/ainews/archive/ainews-to-be-named-1752/">[AINews] OpenAI 的公关活动？</a>：2024年5月7日至5月8日的 AI 新闻。我们为您检查了 7 个 subreddit、373 个 Twitter 账号和 28 个 Discord 社区（419 个频道，4079 条消息）。预计阅读时间...</li><li><a href="https://simonwillison.net/2024/May/8/slop/">Slop 是不受欢迎的 AI 生成内容的新名称</a>：我昨天看到了 @deepfates 的这条推文，我非常赞同：“Slop”正实时成为一个专业术语，就像“Spam（垃圾邮件）”一样……
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1237841811616043009)** (2 条消息): 

无法为此场景提供摘要，因为给定的消息仅包含简短的问候和关于日程的查询，没有实质性的信息、讨论或链接可供总结。
  

---



**OpenAccess AI Collective (axolotl) ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/1237838185715732562)** (5 条消息):

- **RefuelLLM-2 发布**：RefuelAI 开源了 **RefuelLLM-2**，声称是世界上处理“枯燥数据任务（unsexy data tasks）”最强的 Large Language Model。模型权重可在 [Hugging Face](https://huggingface.co/refuelai/Llama-3-Refueled) 下载，详细公告见 [Twitter 帖子](https://twitter.com/BansalDhruva/status/1788251464307187980)。
- **庞大的数据集语料库**：发布的 Llama3-8B 基础模型在包含 **2750+ 个数据集** 的庞大语料库上进行了指令微调（instruction tuned）。这一微调过程持续了大约一周。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1237700395334369280)** (3 messages): 

- **数据集格式的实用文档**：一名成员找到了关于 Axolotl 支持的数据集格式的答案，包括 [JSONL 和 HuggingFace 数据集](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)。这些格式按任务分类，如预训练、指令微调、对话、无模板和自定义预分词（pre-tokenized）。

- **寻求 phi3 mini 配置帮助**：一名成员正在寻求 phi3 mini 在 8 台 A100 GPU 上进行 4K/128K FFT 的可用配置文件，因为尽管他们能够训练更大的模型，但仍面临 CUDA out of memory 错误。

- **请求解决 Axolotl 错误**：一名成员对 [在 8x H100 GPU 上训练时的错误](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1596) 表示沮丧，并引用了在 Axolotl GitHub 仓库上提交的 issue。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - Dataset Formats</a>: 未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1596">Recent RunPod Axolotl error · Issue #1596 · OpenAccess-AI-Collective/axolotl</a>: 请检查此问题之前是否已报告。我搜索了之前的 Bug 报告，未发现类似报告。预期行为：我大约两天前运行了 Axolotl，当时运行正常...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/)** (1 messages): 

nanobitz: 参见此链接 https://docs.wandb.ai/guides/track/environment-variables
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1238099843818197013)** (21 messages🔥): 

- **LoRA 配置难题**：由于在添加新 token 时错误设置了 `lora_modules_to_save`，一名成员遇到了 `ValueError`。这需要指定 `embed_tokens` 和 `lm_head` 来保存 LoRA (Low-Rank Adaptation) 参数。该错误提示需要在 YAML 文件的 `lora_modules_to_save` 列表中显式包含这些模块。
- **YAML 故障与解决方案**：另一名成员提供了使用 `lora_modules_to_save` 更新 YAML 配置的正确语法，确保在 LoRA 设置中捕获嵌入层（embedding layers）和语言模型头（language model head），以修复有关新 token 的错误。
- **代码中的调试关键点**：当 `transformers` trainer 在预测期间出现与 `NoneType` 对象相关的错误时，建议检查数据加载器（data loader）和处理过程，以确保输入格式与模型预期一致。该成员被引导去更新脚本并使用调试日志，以在预测步骤前了解 `inputs` 变量的结构。
- **Traceback 的磨难**：第二个错误的特殊性表明模型在预测/评估阶段接收到了 `None` 输入，这是出乎意料的。解决该情况的建议包括审查数据加载器、数据处理，并确保修改后的训练脚本保持字典输入格式规范。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=848dbbe2-5271-421e-89db-e9f0639a7415)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=7be275a4-8774-4c43-ab05-baa455900008)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://github.com/huggingface/peft/tree/main/src/peft/tuners/lora/config.py#L272L299)">peft/src/peft/tuners/lora/config.py at main · huggingface/peft</a>: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning. - huggingface/peft</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=8b738c5e-41ff-4af1-ad8c-931ff7161389)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1237747724120752222)** (4 条消息): 

- **图表中的“颜值胜过实力”**：一位成员赞赏**图表**的美学，称其拥有“非常漂亮的图片”，但未对其功能性发表评论。

- **对图表指标的质疑**：有人对某些图表中的指标选择提出了批评，例如关注参数数量而非 FLOPs，以及在 Transformer 基准测试中使用了不寻常的学习率，且所有模型均未进行**超参数调优 (hyperparameter tuning)**。

- **保留意见**：在评论图表的有效性时，一位成员表示“*时间会证明一切*”，表明将采取观望态度来评估其实际效用。
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1238201987749642281)** (6 条消息): 

- **寻求在 TPU 上训练 RM 的见解**：一位用户询问了在 TPU 上训练 **RM (强化模型)** 的经验，可能是在寻找优化技巧或最佳实践。
- **FSDP 训练查询**：同一位用户询问了关于使用 **Fully Sharded Data Parallel (FSDP)** 进行训练的问题，暗示其关注重点在于 RM 的扩展和分布式训练策略。
- **推荐使用 Jax 进行 RM 训练**：另一位成员建议使用 **Jax**，暗示它可以简化 RM 训练器的修改。
- **以 EasyLM 作为 Jax 训练器示例**：建议修改现有的 Jax 训练器用于 RM 目的，并提到 **EasyLM** 是一个潜在的起点。
- **使用 EasyLM 训练 RM 的资源**：此外，分享了一个指向 EasyLM 训练器的特定 GitHub 链接，其中包含一个 Python 脚本，可作为使用 Jax 训练 RM 的参考示例：[EasyLM - llama_train_rm.py](https://github.com/hamishivi/EasyLM/blob/main/EasyLM/models/llama/llama_train_rm.py)。

**提到的链接**：<a href="https://github.com/hamishivi/EasyLM/blob/main/EasyLM/models/llama/llama_train_rm.py">EasyLM/EasyLM/models/llama/llama_train_rm.py at main · hamishivi/EasyLM</a>：让大语言模型 (LLMs) 变得简单，EasyLM 是在 JAX/Flax 中进行预训练、微调、评估和部署 LLM 的一站式解决方案。 - hamishivi/EasyLM

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1237981778728452106)** (5 条消息): 

- **排行榜规模受到质疑**：**5k 规模的排行榜**是否足够受到了挑战，有人建议应该是 **10k**。此外，**200:1 的模型与排行榜比例**也被认为高得离谱。
- **错过了续作**：一位成员对错过 **Prometheus (1) 论文**的后续作品感到遗憾，同时想知道对原始论文是否有任何批评。
- **对 Prometheus 的赞赏：** 尽管个人承认有些“酸”，但 **Prometheus** 的工作被称赞为远好于典型的 **AI 研究**。
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1237790255675871334)** (4 条消息): 

- **空中博客计划**：打算在飞机上向 **ChatbotArena** 发布内容，测试机上 WiFi 的可靠性。
- **ChatbotArena 的许可迷宫**：注意到 **ChatbotArena** 在发布由大语言模型 (LLMs) 生成的文本方面可能处于复杂境地，如果未经提供商特别许可，可能存在潜在的许可问题。
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1237824326716096665)** (5 条消息): 

- **OpenAI 发布用于 AI 对齐的 Model Spec**：OpenAI 的第一份 Model Spec 草案旨在指导研究人员和数据标注员，采用了**人类反馈强化学习 ([RLHF](https://openai.com/index/instruction-following))** 技术。该文档为 OpenAI API 和 ChatGPT 中模型的期望行为提供了规范。

- **Llama 3 夺得聊天机器人桂冠**：博客文章 [Llama 3](https://lmsys.org/blog/2024-05-08-llama3/) 讨论了 Meta 的 Llama 3-70B 升至 Chatbot Arena 排行榜榜首，在超过 50,000 场对战中表现出色，并与 **GPT-4-Turbo** 和 **Claude 3 Opus** 等其他顶级模型进行了对比。Lisa Dunlap 等人的文章深入探讨了用户偏好以及在评估聊天机器人性能时提示词带来的挑战。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmsys.org/blog/2024-05-08-llama3/">Llama 3 怎么了？Arena 数据分析 | LMSYS Org</a>: &lt;p&gt;4 月 18 日，Meta 发布了他们最新的权重开放（open-weight）大语言模型 Llama 3。自那时起，Llama 3-70B 迅速攀升至英语 &lt;...</li><li><a href="https://cdn.openai.com/spec/model-spec-2024-05-08.html">Model Spec (2024/05/08)</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1237875219289342035)** (4 messages): 

- **等待 SnailBot 的到来**：频道中表现出对 **SnailBot** 出现或互动的期待。有一种迫切感，并引用了 *tick tock tick tock*（滴答滴答）。
- **针对特定群组的行动呼吁**：专门对一个由 ID 编号识别的群组进行了 ping，建议这些成员关注或参与。
- **对 SnailBot 的庆祝式回应**：在收到 **SnailBot** 的反应后，出现了一种积极且随意的认可，表明互动成功。
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1237850927449510009)** (6 messages): 

- **Pull Request 困惑已解决**：讨论围绕 [tinygrad GitHub 上的 Pull Request #3747](https://github.com/tinygrad/tinygrad/pull/3747) 展开，该 PR 解决了 **UOps.BITCAST 操作**及其在常量折叠（constant folding）方面的意外行为。
- **关于 BITCAST 的澄清**：一名成员澄清说，**CAST** 和 **BITCAST** 确实是 tinygrad 项目中不同的单元操作（uops），并强调了 **BITCAST** 操作的必要性。
- **理解 BITCAST 的实现**：有人建议消除目前从 cast 函数扩展出的 "bitcast=false" 之类的参数，以清理 **BITCAST** 的实现。
- **对 UOp 功能的见解**：确认了 **ALU 操作**与 **uop（单元操作）** 之间的区别，提供了对 tinygrad 项目实现细节的见解。

**提及的链接**：<a href="https://github.com/tinygrad/tinygrad/pull/3747">UOps.BITCAST by chenyuxyz · Pull Request #3747 · tinygrad/tinygrad</a>：隐式修复了 bitcast 不进行常量折叠的问题。

  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1237851914922758205)** (16 messages🔥): 

- **寻求关于 tinygrad 中 `DivNode` 和 `ModNode` 的澄清**：用户询问 `symbolic.DivNode` 是否有意要求 `self.b` 为 `int` 而非 `Node`，这与对符号化（symbolic）版本 `arange` 的需求有关。回复指出，`Node` 或 `int` 的联合类型不应阻止 `Node` 的使用。
  
- **关于符号化 `arange` 实现的讨论**：建议在实现符号化 `arange` 时，`Tensor.full` 可以有条件地调用 `Tensor.from_node`，并且 `div` 和 `sub` 操作可以按照 `mul` 的模式进行扩展。有人担心可能会对下游产生负面影响，以及缺乏对池化（pooling）操作的支持。

- **尝试通过符号化 `arange` 增强 tinygrad**：一名用户在尝试于 tinygrad 中实现符号化 `arange` 后表达了挫败感，参考了他们的 [GitHub pull request](https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:symbolic-arange-pull)，并指出测试中 `step` 的预期行为存在不确定性，计划重新专注于“bitcast 重构”。

- **询问 tinygrad 中的就地（In-place）输出机制**：一位用户询问是否可以将矩阵操作的输出直接写入现有矩阵的预分配部分，以避免浪费的拷贝，并提到了一种连接 `n` 个矩阵操作输出的新颖设计。George Hotz 评论说，对于磁盘存储有一些黑科技（hacks）可以实现这一点，而且更广泛地实现这一功能可能是一个很快就能完成的改动。

- **Metal 构建过程与 `libraryDataContents()` 之谜**：一名在 tinygrad 的 Metal 构建过程中挣扎的用户询问了关于 `libraryDataContents()` 的问题，参考了 Discord 消息和外部文档，但没有找到明确答案。用户质疑是否缺少理解其用法所需的符号或库。

- **介绍 Shape 和 Stride 的可视化工具**：针对在 tinygrad 中研究 shape 和 stride 概念的人员，一名用户创建并分享了一个 [可视化工具](https://mesozoic-egg.github.io/shape-stride-visualizer/)，可以帮助理解不同的组合。

- **InterpretedFlopCounters 和 Flop 计数解释**：关于 `ops.py` 中统计 flop 目的的查询得到了澄清，即在 tinygrad 中，flop 计数是衡量性能的一个指标。

- **在 tinygrad 中寻找类似于 PyTorch `register_buffer` 的功能**：关于询问类似于 PyTorch `register_buffer` 的功能，用户收到指导，在 tinygrad 中可以使用 `Tensor(..., requires_grad=False)` 作为等效方案。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mesozoic-egg.github.io/shape-stride-visualizer/">React App</a>：未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:tinygrad:symbolic-arange-pull">Comparing tinygrad:master...davidjanoskyrepo:symbolic-arange-pull · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你热爱 tinygrad！❤️ - Comparing tinygrad:master...davidjanoskyrepo:symbolic-arange-pull · tinygrad/tinygrad
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1237664274088726569)** (16 条消息🔥): 

- **RAG 实现挑战**：一位用户正在使用 **Cohere.command** 实现 **RAG**，并面临 Cohere.command 的输入上下文长度限制为 4096 tokens，但他们需要处理约 10000 tokens 的问题。使用 *prompt truncating*（提示词截断）可能会导致必要信息的丢失；另一位用户建议 **使用 Elasticsearch** 来减小文本大小，并将简历分解为逻辑段落。

- **生成可下载文件**：一位用户询问 **Cohere Chat** 是否可以输出 **DOCX 或 PDF** 格式的文件，并分享了一个 DOCX 文件的示例链接，但不清楚如何下载该文件。

- **Cohere API 的 CORS 问题**：一位成员询问如何解决使用 **Cohere API** 时的 **CORS** 问题。有人指出 CORS 是浏览器安全特性，调用 Cohere API 应该从后端进行，而不应暴露 API keys。

- **为 Cohere 充值**：一位用户请求协助为账户充值。另一位成员解释说 **Cohere 不提供预付费点数**，但用户可以通过添加信用卡并在仪表板上设置 **billing limits**（计费限制）来管理支出。

- **暗黑模式请求**：关于 **Coral** 是否有 **dark mode** 的询问得到了否定的回答。然而，有人分享了一段可以粘贴到浏览器控制台的 **code snippet**（代码片段），作为临时的暗黑模式解决方案。

**提到的链接**：<a href="https://dashboard.cohere.com/billing?tab=spending-limit">Login | Cohere</a>：Cohere 通过一个易于使用的 API 提供对高级 Large Language Models 和 NLP 工具的访问。免费开始使用。

  

---


**Cohere ▷ #[collab-opps](https://discord.com/channels/954421988141711382/1218409745380147320/1237830229930676357)** (1 条消息): 

- **加入旧金山的 Wordware 团队**：Wordware 正在积极招聘多个职位，包括创始工程师、DevRel 以及产品/前端工程师。鼓励潜在候选人使用 Wordware 的 Web 托管 IDE 构建一些东西，该 IDE 旨在促进 AI 工程师与非技术领域专家之间的协作，并联系 wordware.ai 的创始人。机会如下：[加入 Wordware](https://wordware.notion.site/Join-Wordware-YC-S24-347a2b89acad44c1bc99591636308ec2)。

**提到的链接**：<a href="https://wordware.notion.site/Join-Wordware-YC-S24-347a2b89acad44c1bc99591636308ec2">Notion – 笔记、任务、维基和数据库的一体化工作区。</a>：一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作区。

  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1237705674738110496)** (6 条消息): 

- **Meta-Llama 的后端服务实现**：在执行运行 **Meta-Llama-3-8B-Instruct** 的命令后，说明了可以在 127.0.0.1:8080 使用 **API endpoint** 进行 OpenAI 风格的请求。详细说明可以在 [项目的 GitHub 页面](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#json-api-quickstart) 找到。
  
- **VS Code 与 ollama 的集成**：一位成员指出 **Visual Studio Code** 现在有一个新的下拉菜单功能，允许用户在后台运行 *ollama* 时轻松切换模型。

- **无需重新下载即可更新 Llamafile**：有人建议在 **llamafile** 中增加一项功能，允许更新二进制脚手架而无需重新下载整个 llamafile，并指出虽然目前效率较低，但这是对当前系统简单性的一种权衡。

- **好奇的小丑闲聊**：在一系列消息中，一位成员幽默地思考 **Mozilla-Ocho** 是否是指电影《躲避球》中的 "ESPN 8 - The Ocho"，并就分享这一想法的必要性进行了调侃式的内心对话。

**提到的链接**：<a href="https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#json-api-quickstart">GitHub - Mozilla-Ocho/llamafile: Distribute and run LLMs with a single file.</a>：通过单个文件分发和运行 LLMs。通过在 GitHub 上创建账户为 Mozilla-Ocho/llamafile 的开发做出贡献。

---

**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1238057921871482982)** (3 条消息): 

- **探索电子表格分析中的 AI 辅助**：一位用户询问了关于使用 **LLMs (Large Language Models)** 进行电子表格操作的经验或资源。其兴趣在于利用 AI 处理电子表格并从中提取数据。
- **Yawn.xyz 解决生物实验室电子表格难题**：一位成员分享的资源显示 **Yawn.xyz** 正尝试利用 AI 从复杂的电子表格中提取数据——这是许多生物实验室面临的挑战。根据推文，使用 AI 进行数据提取已通过测试，但一位成员指出其工具的演示效果欠佳。[在此查看推文](https://x.com/yawnxyz/status/1786131427676852338?s=46&t=4-kZga74dpKGeI-p2P7Zow)。

**提到的链接**：<a href="https://x.com/yawnxyz/status/1786131427676852338?s=46&t=4-kZga74dpKGeI-p2P7Zow">Jan 的推文</a>：电子表格是许多生物实验室的命脉，但从杂乱的数据中提取见解是一个巨大的挑战。我们想看看 AI 是否能帮助我们可靠地从任何任意电子表格中提取数据……

---

**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1237806223009517712)** (1 条消息): 

- **适用于 GPT-4-turbo 的 Azure 区域**：一位用户报告了在 **Sweden region** 使用 GPT-4-turbo 0429 时遇到的问题，并正在询问其他运行良好的 Azure 区域。

---

**Alignment Lab AI ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/1237832687226585189)** (1 条消息): 

- **AlphaFold3 开源**：PyTorch 版本的 **AlphaFold3** 开源实现已发布，允许研究人员准确预测生物分子相互作用。代码片段显示它基于原子坐标运行，且需要进行审查 [AlphaFold3 实现](https://buff.ly/3JQVKze)。
- **召集开发者加入 Agora**：为了进一步开发和普及 AlphaFold3 模型，发出了加入 Agora 的行动号召，尽管分享的链接似乎不完整或已失效。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://buff.ly/3JQVKze">GitHub - kyegomez/AlphaFold3: Implementation of Alpha Fold 3 from the paper: &quot;Accurate structure prediction of biomolecular interactions with AlphaFold3&quot; in PyTorch</a>：论文《使用 AlphaFold3 准确预测生物分子相互作用》的 PyTorch 实现 - kyegomez/AlphaFold3</li><li><a href="https://t.co/yZKpKHhHp0">加入 Agora Discord 服务器！</a>：通过开源 AI 研究推动人类进步。| 6856 名成员
</li>
</ul>

</div>

---

**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 条消息): 

pradeep1148: https://www.youtube.com/watch?v=4MzCpZLEQJs

---

**AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/1237770786291581044)** (1 条消息): 

- **Regression Games 发布 Quickscope**：Regression Games 宣布推出 [**Quickscope**](https://play.regression.gg)，这是一套旨在自动化 Unity 游戏测试的 AI 工具套件，具有集成简单和无代码设置的特点。该工具包包括 **Gameplay Session 录制工具**和 **Validations 工具**，可轻松实现冒烟测试和功能测试的自动化。
- **深度属性抓取详解**：Quickscope 的功能包括对游戏对象层级的深度属性抓取，无需自定义代码集成即可从游戏实体中提取全面细节。这允许收集有关位置、旋转以及 MonoBehaviour 的公共属性和字段的详细信息。
- **探索 Quickscope 平台**：鼓励潜在用户尝试使用 Quickscope 来满足其测试自动化需求，该平台推广各种自动化方法，包括**智能回放和播放系统**，并专为 **QA 团队**设计。该平台以快速简便的集成著称，无需自定义代码，显著简化了测试设置过程。

- **高效测试工具亮点**：Quickscope 强调其用于定义测试的 [高度交互式 UI](https://regression.gg/)，为 QA 工程师和游戏开发人员提供了一种用户友好的游戏测试方法。该服务可以直接在 Unity 编辑器、构建版本中使用，或集成到 CI/CD 流水线中。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.regression.gg/post/quickscope-launch">Introducing Quickscope - Automate smoke tests in Unity - May 06, 2024 - Regression Games</a>：了解 Quickscope，一个用于在 Unity 中自动化冒烟测试的工具</li><li><a href="https://regression.gg/">Regression Games - The ultimate AI agent testing platform for Unity</a>：轻松为 Unity 开发用于 QA 测试的机器人。
</li>
</ul>

</div>
  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1237894246535659581)** (1 条消息): 

- **对 `llm` CLI 工具的赞誉**：一位成员对 `llm` 命令行界面表示感谢，称其使用愉快，在像私人助手一样管理项目方面非常有帮助。他们还赞赏了它处理“更具 Unix 风格事务（more unixy stuff）”的能力。
  

---



---