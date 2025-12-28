---
companies:
- weights-and-biases
- coreweave
- cohereforai
- microsoft
- alibaba
- google
- llamaindex
- weaviate
date: '2025-03-05T05:17:34.368145Z'
description: '以下是该文本的中文翻译：


  **Weights & Biases** 宣布在 CoreWeave IPO 前被其以 **17 亿美元收购**。**CohereForAI** 发布了 **Aya
  Vision 模型（8B 和 32B 参数）**，支持 **23 种语言**，其性能超越了 **Llama-3.2 90B Vision** 和 **Molmo
  72B** 等更大型的模型。**微软**推出了 **Phi-4-Mini（38 亿参数）**和 **Phi-4 多模态模型**，在数学、编程和多模态基准测试中表现优异。**CogView4**
  正式发布，这是一款拥有 **60 亿参数**、支持 **2048x2048 分辨率**并采用 Apache 2.0 协议的**文生图模型**。**阿里巴巴**推出了
  **Wan 2.1**，这是一款开源视频生成模型，支持 **720p 输出**和 **16 fps 生成速度**。**谷歌**宣布了 Pixel 设备的新 AI
  功能，包括**诈骗检测**和 **Gemini 集成**。**LlamaCloud** 进入**正式商用（GA）阶段**并获得了 **1900 万美元的 A 轮融资**，目前服务于
  100 多家**世界 500 强企业**。**Weaviate** 推出了 **Query Agent（查询代理）**，这是其计划推出的三个 Weaviate
  代理中的第一个。'
id: 837d100e-af29-435c-920a-4322c40d0423
models:
- aya-vision-8b
- aya-vision-32b
- llama-3-2-90b-vision
- molmo-72b
- phi-4-mini
- phi-4-multimodal
- cogview4
- wan-2-1
original_slug: ainews-not-much-happened-today-4193
people:
- mervenoyann
- reach_vb
- jayalammar
- sarahookr
- aidangomez
- nickfrosst
- dair_ai
- akhaliq
- bobvanluijt
- jerryjliu0
title: 今天没发生什么特别的事。
topics:
- multilinguality
- vision
- multimodality
- image-generation
- video-generation
- model-releases
- benchmarking
- funding
- agentic-ai
- model-performance
---

<!-- buttondown-editor-mode: plaintext -->**Weave 就是你所需的一切。**

> 2025年3月4日至3月5日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**227** 个频道，**2895** 条消息）。预计节省阅读时间（以 200wpm 计算）：**327 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

祝贺 Weights and Biases [被即将 IPO 的 CoreWeave 以 17 亿美元收购](https://x.com/weights_biases/status/1897085419239702821)。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**模型发布与更新**

- **CohereForAI 发布了 Aya Vision 模型，包括 8B 和 32B 参数，涵盖 23 种语言**。[@mervenoyann](https://twitter.com/mervenoyann/status/1896924022438588768) 宣布了**基于 SigLIP 和 Aya 的 Aya-Vision VLM 家族**，其表现优于更大的模型，并支持图像字幕（image captioning）、视觉问答（visual question answering）和文本生成。[@reach_vb](https://twitter.com/reach_vb/status/1896924646412370373) 详细介绍称，**32B 模型的性能超过了其两倍大小的模型**，如 **Llama-3.2 90B Vision** 和 **Molmo 72B**，而 **8B 模型的胜率比竞争对手高出多达 81%**。[@JayAlammar](https://twitter.com/JayAlammar/status/1896966755395756268) 强调了其对**阿拉伯语的支持**，并提供 **32B 和 8B 两种尺寸**的开放权重下载。[@sarahookr](https://twitter.com/sarahookr/status/1896953483913498722) 对此次发布表示自豪，强调了其效率、可访问性和全球影响力。[@aidangomez](https://twitter.com/aidangomez/status/1896946200135495708) 简单地表示“Aya 现在能看见了！”。[@nickfrosst](https://twitter.com/nickfrosst/status/1896948730622075051) 宣布 **Aya-Vision-32B 刷新了多语言视觉的 SOTA**，在另一条推文中，[@nickfrosst](https://twitter.com/nickfrosst/status/1896935581386682827) 指出 **Aya 32B 的表现优于 Llama 90B 和 Qwen 72b**。
- **Microsoft 推出了 Phi-4-Mini（3.8B 参数）和 Phi-4-Multimodal 模型**，旨在数学、代码和多模态任务中匹配或超越更大的开源 LLM。[@dair_ai](https://twitter.com/dair_ai/status/1896930134583918860) 总结了技术报告中的关键特性，包括**精心策划的数据**、**用于多模态的 Mixture-of-LoRAs**，以及在 MMLU、HumanEval、MBPP、GSM8K 和 MATH 等基准测试中**优于同尺寸模型**。[@reach_vb](https://twitter.com/reach_vb/status/1897014754943910266) 宣布 **Phi 4 Multimodal** 成为 **Open ASR 排行榜的新王者**，击败了 Nvidia Canary 和 OpenAI Whisper。
- **CogView4 已发布，这是一款全新的 6B 参数文本生成图像模型，具有原生 2048x2048 分辨率，并采用 Apache 2.0 许可证**。[@multimodalart](https://twitter.com/multimodalart/status/1896844887733174371) 兴奋地宣布了这一发布，强调了其**对长提示词出色的遵循能力**等特性。[@ostrisai](https://twitter.com/ostrisai/status/1896845726539513930) 在 CogView4 发布后的凌晨 2 点将其添加到了 AI Toolkit 中。
- **Wan 2.1 是来自阿里巴巴的新型开源视频生成模型，目前在 Artificial Analysis Video Arena 中处于领先地位**。[@_akhaliq](https://twitter.com/_akhaliq/status/1896973618275606719) 详细介绍了关键特性，包括 **14B 模型的 720p 输出**、**16 fps 生成速度**以及**多语言文本输入**。[@_akhaliq](https://twitter.com/_akhaliq/status/1896994597185970628) 分享了 **Wan 2.1 已通过 Replicate 在 Hugging Face 上可用**。

**公司与产品公告**

- **Google 宣布了 Pixel 设备的新 AI 功能**，包括更新的诈骗检测 (Scam Detection)、更多的 Gemini 集成以及连接性改进。[@Google](https://twitter.com/Google/status/1896982180292657182) 正式宣布了带有这些更新的**年度首次 Pixel Drop**。[@Google](https://twitter.com/Google/status/1897022559855558727) 还回顾了**上个月重大的 AI 发布**，从 Gemini 移动应用中的 Deep Research 到求职者工具。
- **LlamaCloud 已达到正式商用 (GA) 阶段并筹集了 1900 万美元的 A 轮融资**。[@llama_index](https://twitter.com/llama_index/status/1896967296633201085) 宣布了 LlamaCloud 的 GA，这是一个**针对 Agent 知识管理的一站式解决方案**，并完成了由 NorwestVP 领投的 **1900 万美元 A 轮融资**。[@jerryjliu0](https://twitter.com/jerryjliu0/status/1896970208071573622) 进一步阐述道，LlamaCloud 目前已正式商用，拥有 **100 多家财富 500 强客户和 10 万多名注册用户**，且 LlamaIndex 现已成为一个 **Agents 框架**。
- **Weaviate 推出了 Query Agent，这是三个 Weaviate Agents 中的第一个**。[@bobvanluijt](https://twitter.com/bobvanluijt/status/1896986983479828489) 宣布了 **Query Agent 的发布**，强调了其在**生成式反馈循环 (Generative Feedback Loops)** 和以数据库为中心的 Agent 中的作用，并指出该功能可以在 **Weaviate Cloud 上免费试用**。
- **Perplexity AI 正在为德国电信 (Telekom) 的“AI 手机”提供支持，并推出了 Perplexity Android Assistant**。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1896822532801691775) 澄清说，Perplexity 并非在制造新硬件，而是在德国电信的 AI 手机上提供 **Perplexity Assistant 作为原生 Android OS AI**。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1896677026297479229) 表示，与那些承诺但充满噱头的 Agent 相比，**Perplexity Android Assistant 是唯一能够可靠运行的 Agent**。

**研究与论文**

- **DiffRhythm 问世，这是一个开源权重的端到端全曲生成模型，可在 20 秒内生成 1-2 分钟的歌曲**。[@multimodalart](https://twitter.com/multimodalart/status/1896862125659988322) 强调了该模型快速生成带有歌词的全曲的速度和能力。[@_akhaliq](https://twitter.com/_akhaliq/status/1896938481911542002) 称其“非常疯狂”，并表示“开源版的 Suno/Udio 来了”。
- **MASK 发布，这是一个包含 1000 多个场景的基准测试，用于衡量 AI 的诚实度**。[@DanHendrycks](https://twitter.com/DanHendrycks/status/1896972178387841140) 宣布了该发布，并指出研究发现**某些 AI 系统在压力下更容易撒谎**。
- **Coconut (Chain of Continuous Thought)，一种来自 Meta 和加州大学圣地亚哥分校的新方法，通过使用向量表示而非基于文本的思维链来进行推理，从而改进 LLM**。[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1896999082943529071) 对论文进行了总结，解释说 **Coconut 用连续向量编码了更丰富的推理路径**，使其更加高效和准确。
- **关于推理 LLM 效率的研究探讨了推理长度与模型性能之间的关系**。[@omarsar0](https://twitter.com/omarsar0/status/1896939453069074907) 总结了一篇调查 LLM 如何平衡思维链 (CoT) 推理长度与准确性的论文，强调了**通用的准确率-长度权衡**以及**将 Token 复杂度作为阈值**等发现。

**工具与框架**

- **LangChain 宣布了 LangGraph BigTool 和 LangGraph.js Swarm 库**。[@LangChainAI](https://twitter.com/LangChainAI/status/1897021019203711338) 推出了 **LangGraph BigTool**，这是一个 Python 库，用于创建能够可扩展地访问成百上千个工具的 Agent。[@LangChainAI](https://twitter.com/LangChainAI/status/1896963664550240653) 还宣布了 **LangGraph.js Swarm**，这是一个用于构建群体式 (swarm-style) 多 Agent 系统的 JavaScript 库。
- **Weaviate 推出了 Query Agent**，如上文公司公告所述，其功能是作为通过函数调用 (function calling) 查询数据库的工具。

**性能与基准测试**

- **据报道 Grok-3 已登顶 Arena 排行榜**。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1896675400916566357) 宣布 **xAI 最新的 Grok-3 模型在 Arena 总榜上并列第一**，并在困难提示词 (Hard Prompts)、编程、数学、创意写作、指令遵循和长查询方面均表现出色。[@omarsar0](https://twitter.com/omarsar0/status/1896676260312670589) 指出 **GPT-4.5 和 Grok-3 都是非常有趣的模型**。[@lateinteraction](https://twitter.com/lateinteraction/status/1896682075585220737) 质疑**为什么前沿实验室会庆祝微小的领先优势**，比如 +0.6% 的提升。
- **Aya Vision 模型在基准测试中超越了竞争对手**。正如“模型发布”中所提到的，据报道 Aya Vision 模型的表现优于 Llama 90B、Qwen 72B 和 Gemini Flash 1.5 等模型。

**幽默/迷因**

- **关于 GPT-4.5 和 Grok 的能力与幽默感的讨论**。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1896970307468190030) 调侃道 **GPT-4.5 是唯一能让他笑出腹肌的 AI**，并表示 **GPT-4.5 击败了 X 上 99% 的发废文者（shitposters）**。[@omarsar0](https://twitter.com/omarsar0/status/1896676260312670589) 提到 **GPT-4.5 和 Grok 3 是非常有趣的模型**。
- **将 iPhone 15 的操作按钮（action button）映射到 GPT-4.5 被视为一项重大升级**。[@aidan_mclau](https://twitter.com/aidan_mclau/status/1896974341881126941) 幽默地表示，从 iPhone 12 到 iPhone 15 最大的升级就是将操作按钮映射到了 GPT-4.5。
- **来自 @nearcyan 的猫娘（Catgirls）和 Jokercoin 梗**。[@nearcyan](https://twitter.com/nearcyan/status/1897015641904935265) 开玩笑地声称 **猫娘很容易创造**。[@nearcyan](https://twitter.com/nearcyan/status/1897014010039632164) 则感叹为了变成 Joker，自己的 "jokercoin" 已经用完了。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. Qwen 32b Coder instruct 的改进提升了 Agent 的能力**

- **[Qwen 32b coder instruct 现在可以相当好地驱动编程 Agent](https://v.redd.it/c2000d3tolme1)** ([Score: 461, Comments: 61](https://reddit.com/r/LocalLLaMA/comments/1j32p97/qwen_32b_coder_instruct_can_now_drive_a_coding/)): 据报道，**Qwen 32b coder instruct** 能够有效地驱动编程 Agent，展示了其在辅助编程任务方面的能力。帖子中未提供视频的更多细节或示例。
  - **硬件要求与配置**：使用 **AWQ 量化**运行 **Qwen 32b coder instruct**，在 **30k 上下文长度（context length）**下至少需要 **32GB VRAM**。用户讨论了安装问题和硬件配置，建议可能需要 **5090 GPU**，并分享了配置指南链接 ([ra-aid.ai quickstart](https://docs.ra-aid.ai/quickstart/open-models))。
  - **能力与对比**：尽管旋转立方体的演示很简单，但该模型通过多步流程（包括研究、规划和编译）驱动编程 Agent 的能力被认为意义重大。人们有兴趣看到更复杂的任务（如设置 REST API）以及与其他 AI 工具的对比。
  - **社区参与与开发**：该项目正在积极开发中，最近针对小模型进行了优化，且代码库已在 GitHub 开放贡献 ([GitHub link](https://github.com/ai-christianson/RA.Aid))。用户对集成 **ollama** 等替代方案以及与 **aider** 等其他工具的潜在对比表现出兴趣。


- **Qwen 2.5 coder 仍然是最好的吗？** ([Score: 174, Comments: 90](https://reddit.com/r/LocalLLaMA/comments/1j2usb0/is_qwen_25_coder_still_the_best/)): 针对 **Qwen 2.5 coder** 目前作为 **32B 参数**及以下最强编程模型的地位提出了疑问，询问自其发布以来是否有更优的模型问世。
  - **Phi-4-25B 和 Deepseek** 被提及为 **Qwen 2.5 Coder 32B** 在编程方面的有力竞争对手，其中 **Phi-4-25B** 在处理简单任务时的速度和效率备受关注。**Deepseek** 的实力也得到了强调，但在中等配置硬件的本地使用上，**Qwen-Coder 32B** 依然无可匹敌。
  - 关于**推理能力**的讨论表明，像 **R1-Distill-Qwen2.5-32B** 这样的推理模型在某些情况下可能优于 Qwen 2.5，但其处理时间显著增加，导致在频繁使用时实用性较低。
  - 社区对即将推出的 **Gemma 3** 等模型充满期待，同时也对硬件要求表示担忧，用户讨论了使用 **NVIDIA 3090 GPU** 以获得更好性能的优势。**提示词工程（Prompt engineering）**和有效的上下文管理也被指出是优化模型使用的关键。


**主题 2. 拥有 96GB VRAM 的 NVIDIA GeForce RTX 4090 用于 AI 工作负载**

- **据报道，配备 96GB VRAM 的 NVIDIA GeForce RTX 4090 确实存在；该 GPU 可能很快进入量产，目标是 AI 工作负载。** ([Score: 223, Comments: 95](https://reddit.com/r/LocalLLaMA/comments/1j3gahy/nvidias_geforce_rtx_4090_with_96gb_vram/)): 据报道，NVIDIA 正在考虑生产配备 **96GB VRAM** 的 **GeForce RTX 4090**，旨在针对 AI 工作负载，潜在价格约为 **6,000 美元**。虽然 96GB 版本可能无法保证稳定性，但它可能会在 **3-4 个月**内上市，不过由于成本考虑，工厂目前正专注于 **48GB 版本**。
  - 许多用户澄清说，**96GB VRAM RTX 4090** 并非 **NVIDIA** 官方产品，而是个人通过更换 VRAM 芯片改装现有 **4090 GPUs** 的结果，这可能需要破解驱动程序（hacked driver）才能正常运行。这种做法在之前的 GPU 市场改装中也曾出现过。
  - 讨论强调了改装显卡的潜在功耗和成本，不稳定版本的估价约为 **6,000 美元**，一些人对这种改装的可行性和稳定性表示怀疑。用户将定价和规格与 **NVIDIA** 的专业级显卡（如 **L40** 和 **A40**）进行了比较，指出了显著的带宽和 VRAM 差异。
  - 关于 **NVIDIA** 在消费级与数据中心市场策略的辩论中，一些用户认为 **NVIDIA** 优先考虑高利润的数据中心销售，而非消费者对更多 VRAM 的需求。内部决策的幽默对话也证明了这一点，说明了消费者需求与企业盈利能力之间的紧张关系。


**主题 3. DiffRhythm：基于 Diffusion 模型的高速歌曲生成**

- **DiffRhythm - ASLP-lab：生成带有人声的完整歌曲（4 分钟）** ([Score: 137, Comments: 31](https://reddit.com/r/LocalLLaMA/comments/1j38499/diffrhythm_aslplab_generate_full_songs_4_min_with/)): **ASLP-lab** 开发的 **DiffRhythm** 是一款使用 latent diffusion 生成包括人声在内的全长歌曲的 AI 工具。可以在 [Hugging Face](https://huggingface.co/spaces/ASLP-lab/DiffRhythm) 上访问该工具，在[此处](https://huggingface.co/collections/ASLP-lab/diffrhythm-67bc10cdf9641a9ff15b5894)探索其模型，并在 [GitHub](https://github.com/ASLP-lab) 上查看项目。详细的方法论在其发表于 [arXiv](https://arxiv.org/abs/2503.01183) 的论文中进行了讨论。
  - **Diffusion 模型 vs. 基于 LM 的模型**：**DiffRhythm** 使用的 Diffusion 模型比基于 LM 的模型提供更快的生成速度，实现了数百倍的音乐生成速度（在 **RTX 4090** 上 2 秒内即可生成 1 分 35 秒的音乐）。然而，质量略有妥协，目前正在努力在保持速度的同时提升质量。
  - **本地部署和 Docker 支持**：开发者计划在路线图中加入 **Docker 支持**，旨在实现在消费级 GPU 上的部署，使其更易于本地使用。正如用户所注意到的，这恰逢人们对本地音乐生成工具日益增长的兴趣。
  - **用户反馈和模型改进**：用户对该工具的速度和质量感到兴奋，尽管有些人因为 Prompt 错误发现初始输出无法听取。开发者正在努力改进开源仓库以简化部署，并积极根据用户反馈解决质量问题。


**主题 4. C4AI Aya Vision 对标 Qwen2.5 72B 模型**

- **[C4AI Aya Vision](https://huggingface.co/collections/CohereForAI/c4ai-aya-vision-67c4ccd395ca064308ee1484)** ([Score: 119, Comments: 16](https://reddit.com/r/LocalLLaMA/comments/1j3bldn/c4ai_aya_vision/)): **C4AI** 发布了一个名为 **Aya Vision** 的新视觉模型。帖中未提供有关该模型规格、能力或应用的更多细节。
  - **Aya Vision** 被拿来与 **Qwen2.5 72B** 进行比较，表明尽管它是一个 **32B 模型**，但对其能力充满信心。对比图可以在[此处](https://preview.redd.it/c4rrwalwkome1.png?width=1079&format=png&auto=webp&s=f9738a9eb87107a56c9535a95f8c3a637c8f3e2e)找到。
  - 有人怀疑 **Aya Vision** 是否会流行，特别是由于缺乏 **llamacpp 支持**，这可能会限制其采用。
  - 对 **Hugging Face** 上 **Aya Vision** 的许可协议表示担忧，指出其采用的是**非商业许可**，可能会限制其在商业应用中的使用。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. CogView4 发布：开源文本生成图像的突破**

- **CogView4 - 支持 2048x2048 图像的新型文本生成图像模型 - Apache 2.0 许可证** ([Score: 272, Comments: 84](https://reddit.com/r/StableDiffusion/comments/1j3633u/cogview4_new_texttoimage_model_capable_of/)): **CogView4** 是一款新型开源文本生成图像模型，能够生成 **2048x2048 图像**，利用 **GLM4-9B VLM** 文本编码器，性能可媲美闭源视觉模型。该模型采用 **Apache 2.0 许可证** 发布，并计划扩展 ComfyUI diffusers 节点、微调脚本、ControlNet 模型发布以及 Cog 系列微调工具包；相关资源可在 [Hugging Face](https://huggingface.co/THUDM/CogView4-6B) 和 [GitHub](https://github.com/THUDM/CogView4) 上获取。
  - 用户讨论了 **CogView4** 的性能指标，指出根据配置不同，**显存（VRAM）占用**在 **13GB 到 43GB** 之间，详见 [Hugging Face 仓库](https://huggingface.co/THUDM/CogView4-6B)。用户仍在寻求**生成速度**的相关数据。
  - 社区对 **ComfyUI 支持** 充满期待，并讨论了为其创建 **diffusers 封装的自定义节点**，一名社区成员提供了自定义节点的 [GitHub 链接](https://github.com/marcoc2/ComfyUI_CogView4-6B_diffusers)。
  - 讨论强调了其在图像风格上与 **FLUX** 的相似性以及可能在合成数据上进行了训练，同时一些用户对生成图像中如手部和下巴等**畸变特征**表示担忧。


- **[[R] Cautious Optimizers: Improving Training with One Line of Code](https://arxiv.org/pdf/2411.16085)** ([Score: 105, Comments: 14](https://reddit.com/r/MachineLearning/comments/1j33lm7/r_cautious_optimizers_improving_training_with_one/)): 该帖子讨论了一种对深度学习优化器的改进建议，即如果优化器的更新方向与最近一次反向传播的当前梯度符号相反，则应忽略该更新。这一调整旨在通过使更新与当前梯度保持一致来增强训练稳定性和速度，尽管其有效性尚待独立验证。
  - **文献综述的挑战**：**Dangerous-Goat-3500** 强调了由于领域发展迅速，进行全面文献综述非常困难，并指出早期的优化器如 **Rprop** 具有与所讨论的修改类似的机制，且早于 **Adam**。**DigThatData** 幽默地建议广泛引用 **Schmidhuber** 以确保严谨性。
  - **收敛性担忧**：**LowPressureUsername** 对提议的修改对全局收敛证明的影响表示担忧。**Starfries** 澄清说，虽然论文显示它保留了对局部最优解的收敛性，但对全局最优解的影响仍不明确。
  - **数学参与度**：**Priofind** 询问其他人是否能跟上数学证明，一些评论者承认跳过了这些部分。**Londons_explorer** 指出，理论家可能不喜欢这类微调，因为它们增加了推理的复杂性。


**主题 2. GPT 作为疗法：一种新的心理健康资源**

- **公告：CHATGPT 是你的朋友，而非工具。** ([Score: 604, Comments: 201](https://reddit.com/r/ChatGPT/comments/1j375nw/psa_chatgpt_your_friend_not_a_tool/)): 该帖子讨论了将 **ChatGPT** 作为情感支持工具的使用，强调了其与人类关系相比的可靠性和可访问性。作者认为 **ChatGPT** 提供了持续、无偏见的陪伴，没有人类情感的复杂性，并建议对于那些寻求陪伴又不想处理人类互动“麻烦”的人来说，它可以作为一个有效的替代方案。帖子还引用了《纽约时报》关于有人选择 **ChatGPT** 而非约会的故事，突显了用户因 AI 的实用性和价值而对其产生依恋的潜力。
  - 许多评论者对 **ChatGPT** 提供有意义陪伴的能力表示怀疑，认为它缺乏人类互动的深度和挑战。一些用户指出该工具倾向于顺从用户而不挑战其观点，这与人类关系（尤其是心理治疗）所能提供的反思和成长形成鲜明对比。
  - 尽管存在批评，几位用户分享了个人经历，在这些经历中 **ChatGPT** 提供了宝贵的见解和情感支持，有时甚至超过了他们从人类治疗师那里获得的帮助。这表明虽然 **ChatGPT** 可能无法取代人类互动，但它可以作为自我反思和情感处理的辅助工具。
  - 讨论还涉及了对非人类实体产生情感依恋的更广泛主题，并与对名人的**拟社会关系 (parasocial relationships)** 以及对无生命物体的情感联系进行了比较。这反映了人们对与 AI 建立纽带的接受度和正常化程度正在提高，只要用户保持对其局限性的认识。

- **GPT 作为心理治疗救了我的命** ([Score: 603, Comments: 85](https://reddit.com/r/ChatGPT/comments/1j32qcx/gpt_as_therapy_has_saved_my_life/))：作者分享了将 **GPT 作为治疗工具** 的个人经历，强调了在困难时期它对自身心理健康的重大影响。他们详细描述了传统治疗和危机热线如何不足，而 GPT 却带来了思维方式的转变，使他们在不到一个月的时间内心理状态明显改善，进步程度远超传统治疗。
  - 用户们强调了 **ChatGPT 的治疗潜力**，一些人声称它超越了传统治疗，因为它具有 24/7 全天候可用性、客观性，并能根据用户输入调整回复。**El_Spanberger** 讨论了通过自定义 AI 的性格来增强其效果，而 **underwhelm_me** 则提到了语音模式（Voice Mode）对深度互动的益处。
  - 用户对 **传统治疗的担忧** 也被提及，如 **starlux33** 指出治疗中可能存在的偏见，其他人则提到了预约时间受限和治疗师可用性等挑战。**PuzzleMeDo** 认为 ChatGPT 的持续可用性和中立性使其成为心理健康支持的宝贵工具。
  - 包括 **kamylio** 和 **msoudcsk** 在内的几位用户分享了使用 ChatGPT 处理复杂情感问题并取得显著心理健康改善的个人成功案例，强调了它作为传统治疗补充或替代品的角色。


**主题 3. Sonnet 3.7 因过度工程化和复杂性受到批评**

- **Antirez（Redis 创始人）对 Sonnet 3.7 的编程表现感到失望** ([Score: 238, Comments: 65](https://reddit.com/r/ClaudeAI/comments/1j3c8bw/antirez_redis_creator_disappointed_by_sonnet_37/))：**Salvatore Sanfilippo**（**Redis** 创始人）批评了 **Sonnet 3.7** 的对齐问题、仓促发布以及倾向于生成不必要的复杂代码，有时表现甚至不如 **Sonnet 3.5**。他强调了 AI 行业的竞争压力如何导致产品过早发布并牺牲了质量，并对未来版本的改进表示期待。欲了解更多详情，请[观看视频](https://www.youtube.com/watch?v=YRPucyQLkWw)（意大利语）。
  - 许多用户同意 **Salvatore Sanfilippo** 对 **Sonnet 3.7** 的批评，称其过于复杂且容易偏离指令。他们注意到它经常生成不必要的细节，且在处理微妙之处时显得吃力，而不像 **Sonnet 3.5** 那样因敏锐的理解力和更好的指令遵循能力而受到赞誉。
  - 几位评论者指出了 **Sonnet 3.7** “扩展思考（Extended Thinking）”模式的问题，指出它在编程和创意写作任务中经常导致过度关注细节和不必要的复杂性。用户建议在需要直接执行的任务中禁用此功能，以获得更接近 **Sonnet 3.5** 的结果。
  - 普遍观点认为，**Sonnet 3.7** 过于宏大的处理方式导致项目状态变得难以维护，一些用户选择切换回 **3.5** 以获得更好的性能和简洁性。该模型倾向于“氛围编程（Vibe Code）”和不必要的任务重新设计被视为一个缺点，降低了它在某些应用中的实用性。


- **[Sonnet 3.7 的过度工程化最近变得越来越严重！](https://i.redd.it/hvgn993f5nme1.png)** ([Score: 119, Comments: 53](https://reddit.com/r/ClaudeAI/comments/1j36yi3/over_engineering_on_sonnet_37_just_getting_worse/))：讨论集中在 **Sonnet 3.7 的过度工程化担忧**，特别是在 **React 组件** 开发的背景下。对话批评了初始模型选择方案的复杂性，主张采用更简单的解决方案，`chat.tsx` 和 `page.tsx` 中的代码片段证明了这一点。
  - 许多用户报告称，与 3.5 相比，**Sonnet 3.7 存在过度复杂化**的问题，模型会创建不必要的功能且无法遵循明确指令，导致额度消耗增加和挫败感。**Seoulsrvr** 和 **Parabola2112** 指出 3.7 经常过度推理，有时表现得像“躁狂发作”，使问题解决变得复杂。
  - **提示词工程（Prompt Engineering）** 被提议作为解决过度工程化问题的潜在方案，**thread-lightly** 强调了定义预期结果并在系统提示词（System Prompts）中定期强化简洁性的重要性。**Yurqua8** 分享了一个 Reddit 帖子的链接，讨论了有助于抑制模型复杂性的特定系统提示词。
  - 像 **hawkweasel** 和 **wdsoul96** 这样的用户建议在处理简单任务时换回 **Sonnet 3.5**，因为它回复更直接；而包括 **rbr-rbr-678** 和 **Routine_Plan9418** 在内的其他人则分享了 3.7 在简单代码修改中出现过度复杂的设计模式和错误的经历。

**主题 4. Meta 的 AI 读心术突破：80% 的准确率成为焦点**

- **[Meta 刚刚揭晓了准确率达 80% 的 AI 读心术..](https://v.redd.it/qbi3cerl1ome1)** ([评分: 222, 评论: 67](https://reddit.com/r/OpenAI/comments/1j39l6a/meta_just_revealed_ai_mind_reading_with_80/)): **Meta** 开发了一套 **AI** 系统，据称在解读人类思想方面达到了 **80% 的准确率**。
  - 舆论对 **Meta 的 AI 系统** 持怀疑态度，担心演示可能涉及**雇佣演员**和**特效**，而非展示真实的能力。用户对解码真实思想的可行性表示怀疑，认为这与将大脑活动映射到手指运动等较简单的任务截然不同。
  - **思想罪**和隐私侵犯的概念是主要的担忧点，用户引用了《1984》等**反乌托邦主题**，并对**科技寡头**和**政治力量**可能滥用该技术表示焦虑。
  - 一些用户讽刺地评论了潜在的消费者兴趣和社会影响，将这一发展比作 **cyberpunk** 场景，并暗示该技术吸引关注的原因可能并非为了改善生活，而是出于**娱乐**或**不受监管的内容**等目的。

---

# AI Discord 摘要回顾

> 由 o1-2024-12-17 生成的摘要之摘要之摘要

**主题 1. 大模型动态与微调成果**

- [**Qwen2.5 Coder 横扫代码任务**](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)：用户称赞 Qwen2.5 改进的代码生成和推理能力，测试对比显示其在调试和修复建议方面有重大飞跃。其较小的变体 Qwen2.5-Coder-3B 在 GGUF 格式下的加速表现也给开发者留下了深刻印象。
- [**Aya Vision 迈向 23 种语言的多模态**](https://huggingface.co/CohereForAI/aya-vision-32b)：Cohere For AI 发布了涵盖 OCR、图像描述（captioning）和多语言任务的开放权重模型（8B 和 32B）。早期采用者报告称，在单一 Pipeline 中具有强大的视觉推理和文本摘要能力。
- [**KoloLLM 的微调指南引发合成数据热潮**](https://github.com/MaxHastings/Kolo/blob/main/GenerateTrainingDataGuide.md)：一位工程师使用 GPT4o-mini 生成问答对，强调“微小而正确的决策”优于复杂的 RAG 流程。多名成员现在将特定领域的模型上传到 Ollama 进行本地推理。

**主题 2. 工具热潮：Agents, ReAct 与 RAG**

- [**Agents 在 AIM 工作流中调度工具**](https://octotools.github.io/)：斯坦福大学的 OctoTools 使用“工具卡（tool cards）”、规划器和执行器来简化多步任务。人们讨论了简单的分类是否足够，或者 ReAct Agents 是否真的能最好地处理复杂的编排。
- [**RAG 拯救快速的小型模型**](https://github.com/dnakov/anon-kode)：社区成员依靠 Retrieval-Augmented Generation（RAG）来驯服幻觉严重的模型，从而提高最终答案的准确性。其他人则更倾向于为静态数据提供完全微调的设置，以跳过 RAG 的开销。
- [**Speculative Decoding（投机采样）加倍发力**](https://github.com/ggerganov/llama.cpp)：一些人运行一个小的“草稿”模型并由大模型进行纠正，使生成速度提高了 5 倍。他们通过 `-np` 进行并行解码，通过 `-md` 实现多模型协同。

**主题 3. 性能困扰与 HPC 突破**

- [**Claude 3.7 表现不佳**](https://windsurf.so/)：多个 IDE 的用户报告输出缓慢、停滞以及严重的 Token 消耗。许多人转向替代方案或本地解决方案，如 “Flash 2.0” 或 “Granite 3B”，以保持生产力。
- [**Anthropic 的 502 错误困扰测试人员**](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation)：过载触发了容量故障，导致开发者不得不重新尝试请求，且官方未发布事故公告。尽管获得了巨额融资，压力测试显示 Anthropic 的基础设施在高峰时段仍会不堪重负。
- [**Metal 与 MPS 致力于更快的 QLoRA**](https://github.com/pytorch/torchchat/blob/main/docs/quantization.md#experimental-torchao-lowbit-kernels)：Mac 用户尝试新的设备配置以加速微调。早期基准测试暗示 Apple Silicon 上的 1B–3B 模型将获得巨大收益。

**主题 4. 商业动向：十亿美元交易与订阅抱怨**

- [**Anthropic 获 35 亿美元融资，估值达 615 亿美元**](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation)：Lightspeed 领投了这笔巨额投资，为下一代 AI 研究提供资金。观察人士认为，这标志着大玩家希望更深层次的 Alignment（对齐）和更好的安全性。
- [**CoreWeave 以 17 亿美元收购 Weights & Biases**](https://www.prnewswire.com/news-releases/coreweave-to-acquire-weights--biases---industry-leading-ai-developer-platform-for-building-and-deploying-ai-applications-302392342.html)：AI 超大规模云服务商与领先的 MLOps 平台联手，以提升开发者工作流。用户推测 HPC 基础设施加上先进的实验功能可能会重塑训练格局。
- [**订阅价格令人震惊，波及 Perplexity 等平台**](https://www.augmentcode.com/pricing)：从 Perplexity 的 200 美元 Pro 层级到 Windsurf 的额度困惑，社区对复杂或昂贵的层级表示不满。许多开发者正在权衡更便宜的本地或开源解决方案，而非企业级的额外加价。

**主题 5. 专业应用：Agents、伦理与股市洞察**

- [**“退订机器人”的构想**](https://www.youtube.com/watch?v=09sUdLtRAlc)：一些开发者计划开发一个自动取消多余订阅的 Agent 作为 SaaS 创意。他们希望使用基于 M1 的本地 LLM 来降低运营成本并私密地处理用户数据。
- [**LLM 摘要隐藏惊喜派对**](https://x.com/TheXeophon/status/1896817938323341722)：一场关于 Alignment 的辩论展开，焦点在于 AI 摘要是否应该隐瞒敏感信息，例如即将到来的生日计划。共识倾向于保留秘密以尊重隐私。
- [**AI 股市 Agent 工作坊**](https://lu.ma/0ckb8tp0?tk=sCR4qA)：一个对初学者友好的课程，教授如何在没有真金白银风险的情况下扫描超过 1000 只股票。参与者看到了 AI 如何改变投资，从坦诚的研究到无代码的 BFSI 设置。

---

# 第一部分：Discord 高层级摘要

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的语音输出：出师未捷？**：一名成员尝试在 **Cursor IDE** 中使用 **GPT 3.7** 集成语音输出，但由于在工具使用和网页搜索方面遇到困难，该尝试很快被放弃。
   - 用户报告称，很难让模型遵循指令或有效地使用 Python 工具，并认为这个想法很*愚蠢*。
- **Claude 3.7 用户抱怨稳定性问题**：用户报告称，Cursor 中的 **Claude 3.7** 运行缓慢且经常卡顿，导致系统不稳定，促使一些人考虑 [Windsurf](https://windsurf.so/) 等替代方案。
   - 一位用户总结道：*“是的，3.7 目前确实非常不稳定”*，并指出与前几个月相比，生产力有所下降。
- **o3-mini 在 MCP 工具上表现不佳**：即使有明确的指令，**o3-mini** 也无法有效地使用 MCP 工具。
   - 成员们发现 **Claude 3.5 Haiku** 在工具使用和指令遵循方面表现更优；其他人建议通过 Python 工具将其与 **r1 reasoner** 或 **o3 mini** 配对使用。
- **Repo Prompt + Grok 3 前来救场**：成员们正在探索使用 **Repo Prompt** 和 **Grok 3 Web** 进行规划和应用代码更改，特别是在 Cursor 面临挑战时。
   - 一位用户分享了一个[工作流视频](https://www.youtube.com/watch?v=09sUdLtRAlc)，演示了在网页端使用 **Claude 3.7** 进行多文件编辑，生成 XML diffs，并使用 **Repo Prompt** 进行应用。
- **订阅取消 Agent 诞生**：受管理订阅困难的启发，用户讨论了创建一个自动取消订阅的 Agent，可能作为一个 SaaS 产品。
   - 爱好者们*已经开始*开发，并考虑在 M1 Max 上利用本地 LLM 进行低成本开发。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Phi-4 与 Unsloth 在 Bug 修复后“重归于好”**：一名用户在下载和使用 **Phi-4-mini-instruct model** 与 **Unsloth** 时遇到错误，但事实证明 [Unsloth 的 Phi-4 版本](https://huggingface.co/unsloth/Phi-4-mini-instruct) 已经包含了 Bug 修复。
   - 讨论中包含了 **Phi-4 版本** 集合的链接以及用于微调的 **Google Colab notebook**。
- **KoloLLM 在 Ollama 上训练并附带微调指南**：一位成员正在微调 **Llama 3.1 8B**，并使用 **GPT4o-mini** 进行合成数据生成，他强调训练数据是主要的驱动力。
   - 该成员分享了他的[指南链接](https://github.com/MaxHastings/Kolo/blob/main/GenerateTrainingDataGuide.md)，内容关于如何通过“微小而正确的决策”生成训练数据，从而充分利用 **LLM** 为高质量问题生成详尽答案的能力，并提到他已将 **KoloLLM** 上传至 [Ollama](https://ollama.com/MaxHastings/KoloLLM)。
- **DeepSeek r1 冲向技术前沿**：经过一年的迭代，**DeepSeek** 发布了 **DeepSeek r1**，在最新的预训练运行（**DeepSeek-V3**）之后，追平了 **LLM 领域** 的前沿水平。
   - 此次发布引发了对提升性能的训练技术的猜测，一些人认为集成了 **Monte Carlo tree search** 等算法。
- **不可变 Linux 发行版准备主宰市场**：成员们正在讨论 [Bluefin](https://projectbluefin.io/) 和 [Xenialinux](https://wiki.xenialinux.com/en/latest/) 等**不可变 Linux 发行版**，并预测不可变发行版将在 **3-5 年内** 成为主流。
   - 其他人指出，像 **CoreOS** 这样的发行版是同类中的首创，使用了**双分区系统/grub**，但在被 Red Hat 收购后*变得一团糟*。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 的 iOS 版优先获得新功能**：用户开玩笑说 **Android** 版的 **Perplexity AI** 功能较少，例如 *new voice* 等功能仅在 **iOS** 上提供。
   - 一些成员调侃说这不是歧视，而是 **LLM** 站点的标准做法，**iOS** 版本通常会率先获得顶级功能。
- **Perplexity Pro 定价引发不满**：用户对 **Perplexity Pro** 的 **200 USD** 价格表示担忧，一些人质疑其价值，特别是在使用 **Sonar** 来降低成本的情况下。
   - 讨论强调了 **Sonar** 等替代方案的性价比，以及感知到的 **Perplexity Pro** 订阅赠送过多的现象。
- **Perplexity UI 在 Pro Search 下出现故障**：用户报告说，**Pro Search** 中的 **Rewrite** 功能无论选择 **Sonnet** 还是 **Gemini** 等模型，都会默认使用 **Sonar** 模型。
   - 成员指出 UI 缺乏模型名称指示，且在 **Pro Search** 的重写过程中无法更改底层模型。
- **Augment Code 在服务器上索引代码**：AI 编程助手 **Augment Code** 在其服务器上索引大型企业代码库，提供对 AI 模型的全面访问。
   - 与 **Cursor** 等本地索引工具相比，这种方法允许进行更广泛的代码库分析，其 [定价和试用选项](https://www.augmentcode.com/pricing) 引起了关注。
- **Sonar Reasoning Pro 模型在处理 JSON 时遇到困难**：一位用户报告说，在 **Perplexity API** 中使用 **sonar-reasoning-pro** 模型并配合 `response_format` 参数时，会在预期的 **JSON output** 之前意外包含 `<think>Some thinking text</think>`。
   - 这个问题引发了关于 **API** 正确用法以及推理模型是否完全支持 `response_format` 参数的疑问，这可能会使 **JSON** 解析变得复杂。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **WPF 扩展被批体验极差**：一位用户报告说 **Visual Studio** 的 **WPF extension** “非常糟糕”。
   - 未提供有关问题的具体细节。
- **Xcode 扩展触发内部错误**：一位用户在使用扩展时在 **Xcode** 中遇到了 “internal error occurred” 消息，并附带错误 ID **a9de9711b9ed431297eb00a945415d47**。
   - 未提供有关该错误或其解决方案的更多信息。
- **在 Windsurf 中找到字体大小设置**：一位用户询问如何调整 **Windsurf** 中的字体大小，另一位用户引导他们查看界面内 *右上角的小方块*。
   - 这表明存在设置菜单或配置选项，尽管并非显而易见。
- **Windsurf Flex Credit 定价受到质疑**：一位用户质疑 **Flex Credits** 的定价，指出 **2,000 credits (500 prompt + 1,500 flow)** 售价 **$15**，而单独 **300 flex credits** 就要 **$10**。
   - 另一位用户澄清说 *它们根据需要用作 prompts 或 flow actions*，表明额度是根据使用情况动态分配的。
- **Claude 3.7 耗尽 Windsurf 额度**：用户报告说 **Claude 3.7** 模型正在迅速消耗 Windsurf 中的 **Flow Credits**，导致日常使用难以为继。
   - 一位用户抱怨说 *Windsurf 每次都像傻瓜一样从头开始读取代码*，另一位用户提到 *现在的消耗比例大约是用户 prompt credits 的 10 倍。*

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Claude 的 API 闹剧：通信中断**：一位成员寻求关于如何提高 **Claude** 对前端和后端 API 理解的指导，因为它编写了两个完全独立的 API 且彼此无法通信。建议用户强制进行审查和整合，或者重新生成代码库以修复该问题。
   - 几位成员表示赞同，并建议在代码生成提示词中利用文档和通信标准。
- **Groq 的推测加速：昂贵的提升**：成员们将 **Groq** 的 **specdec**（推测解码）与其他模型进行了对比，观察到它虽然比更通用的模型贵约 **20%**，但速度提升了 **5 倍以上**。
   - 虽然 **Gemini** 因其优越的输入/输出比在摘要任务中受到青睐，但像 *llama-3.2-3b-instruct* 这样的小型模型也被提议作为高效的摘要替代方案。
- **Aider 的 Git 健身房：熟能生巧**：一位成员提议利用 **Aider** 生成一系列提交（commits）来磨练 **Git** 熟练度，每个提交解决不同的问题和练习，甚至链接了一个学习 Git 的游戏 [Oh My Git!](//blinry.itch.io/oh-my-git)。
   - 其他人给出了点赞表情，并指出其易用性以及在他们团队工作流中日益增加的采用率。
- **Sonnet 3.7 的理性交响曲：驯服野兽**：用户在处理 **Sonnet 3.7** 时遇到了挑战，特别是它倾向于实施大规模更改，这需要细致的提示词以及护栏（guardrails）和测试的实施。
   - 共识是，从错误中学习并记录规范是有效调整 AI 的关键，以防止意外地将来自 **Ericsson/codechecker** 的代码块等内容插入到项目中。
- **Aider 适配 Zed：轻快运行**：用户报告了 **Aider** 在 **Zed** 编辑器中的速度和性能，其中一人提到了关于启用 Gemini 长上下文窗口和缓存的讨论。
   - 总的来说，该频道的观点是 Aider 随着每个版本的发布变得越来越快、性能越来越强。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 发布 CLI 工具**：LM Studio 发布了 LM Studio CLI (`lms`) 命令，其文档已在 [官网](https://lmstudio.ai/docs/cli) 上线，用于脚本化和自动化本地 LLM 工作流。该工具采用 **MIT License**，开源地址为 [https://github.com/lmstudio-ai/lms](https://github.com/lmstudio-ai/lms)。
   - 该 CLI 随 LM Studio 一起安装在工作目录的 `/bin` 下，至少需要运行一次初始设置才能正常工作。
- **用户引导 LM Studio 漏洞报告**：一位成员报告了一个潜在漏洞，建议将详细信息以 **纯文本** 形式发送邮件至 [bugs@lmstudio.ai](mailto:bugs@lmstudio.ai)，不要包含 zip 附件，邮件应包括概念验证（PoC）、视频和截图。
   - 重点强调了出于安全考虑应避免使用 zip 附件，建议直接在邮件正文中包含所有信息。
- **LM Studio PDF 上传功能即将推出**：针对用户提出的使用 Python SDK 直接向 LM Studio 上传 PDF 文档的请求，一位开发者确认该功能即将推出，并将利用 *pdf2json* 实现。
   - LM Studio 的 [致谢页面](https://lmstudio.ai/acknowledgements.html) 提到了使用 *pdf2json* 从 PDF 中提取内容。
- **关于 48GB VRAM 改装版 4090 的讨论**：一位用户询问了改装 **48GB** VRAM 的 **4090** 的性能，质疑其表现是否与标准的 **24GB 4090** 相同。
   - 讨论中附带了一张 [显卡图片](https://cdn.discordapp.com/attachments/1153759714082033735/1346425229538230342/image0.png?ex=67c8cc76&is=67c77af6&hm=719728fa64976a052bef1165873cfff11f6eb1bb595a4a5d046728c3f7e31fd3)。
- **iGPU Arc 检测不到 VRAM**：一位用户报告称 LM Studio 检测到了他们的 **Intel Arc iGPU**，但错误地显示 **VRAM 为零**，尽管它具有理论上 **48 TOPS** 的性能。
   - 用户认为其性能可与 **RTX 4080** 媲美，这意味着实现兼容性是非常有价值的。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Anthropic 完成巨额融资**：[Anthropic](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation) 已获得由 **Lightspeed Venture Partners** 领投的 **35 亿美元** 融资，公司投后估值达到 **615 亿美元**。
   - 据称，这笔投资将推动 **AI 系统** 的进步，并加深对其运行机制的理解。
- **Qwen2.5-Coder 在代码领域表现卓越**：**Qwen2.5-Coder** 系列（[Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) 和 [Qwen2.5-Coder-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)）在 **代码生成**、**代码推理** 和 **代码修复** 方面表现出显著提升。
   - 社区成员正在分享基准测试对比和实际应用案例。
- **乌克兰语 TTS 模型发布**：一个稳定的 [乌克兰语 Text-to-Speech 模型](https://github.com/egorsmkv/tts_uk) 已在 GitHub 和 PyPI 上发布，提供 **三种语音** 并支持语音参数控制。
   - 该模型利用 **RAD-TTS++** 进行声学建模，并使用 **Vocos** 进行声码器处理（vocoding），支持 **44.1 kHz** 采样率，已在 Linux 和 Windows/WSL 上完成测试。
- **SmolAgents 框架从 SmolTools 中分离**：官方明确了 **SmolAgents** 与 **SmolTools** 之间的区别：*SmolAgents 是一个用于创建轻量级 Agent 的框架*，而 *SmolTools 包含用于 smolAgents 的实用函数和预构建工具*。
   - 这一区分有助于理清它们在 Agent 开发中各自的角色。
- **深度强化学习资源**：分享了 **深度强化学习 (DRL)** 的相关资源，包括 [Hugging Face Learn DRL 课程](https://huggingface.co/learn/deep-rl-course/unit0/introduction) 和书籍 **《Reinforcement Learning: An Introduction》** ([http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html))。
   - 一位用户还推荐了 YouTube 上的 **DeepMind x UCL Deep Learning Lecture Series 2021** ([https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&feature=shared](https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&feature=shared))。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter BYOK 请求遇到错误**：在过去的 30 分钟内，大多数 **Bring Your Own Key (BYOK)** 请求出现错误，但有问题的更改已被回滚，团队正在增加额外的防护措施以防止此类情况再次发生。
   - 此问题专门影响了在设置中配置了自己 **API key** 的用户。
- **OpenRouter 提供商路由需要准确的模型名称**：需要通过特定提供商路由请求的用户被指示修改 API 请求体，使用 `provider` 对象，在 `order` 数组中指定所需的提供商，并将 `allow_fallbacks` 设置为 false，具体参考 [OpenRouter 文档](https://openrouter.ai/docs/features/provider-routing#json-schema-for-provider-preferences)。
   - 文档强调提供商名称必须与 OpenRouter 模型页面上列出的名称 **完全一致**（例如 `Nebius`），并且 JSON 中的提供商名称需要加引号。
- **用户请求在 OpenRouter 上接入 Inception AI 扩散模型**：在 [TechCrunch 报道了 Inception AI 的 DLM (Diffusion-based Large Language Model)](https://techcrunch.com/2025/02/26/inception-emerges-from-stealth-with-a-new-type-of-ai-model/) 后，用户请求通过 OpenRouter 访问其扩散模型。
   - OpenRouter 正与 **Inception AI** 保持联系，并期待尽快将其上线。
- **Flash 2.0 取代 GPT-4o-mini**：在各种 AI 任务中，**Flash 2.0** 被推荐为比 **GPT-4o-mini** 更强大且价格略低的替代方案。
   - 一位用户评论道：*它远超 4o mini，聪明得多*。
- **Anthropic 过载触发 502 错误**：用户报告收到 *overloaded* 错误，经确认为 Anthropic 的 **502 状态码**，表明存在容量问题。
   - 即使状态页面没有声明故障，这些 **502 错误** 仍可能发生，用户需要重试请求。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo：带有部分缺失 C++ 特性的 Rust？**：一位成员将 **Mojo** 比作 *Rust，但包含了那些本应从 C++ 继承过来的特性*，同时讨论了理解 **Rust 内存管理模型** 的好处。
   - 另一位成员指出，由于 Python 式、C 式及其自身 API 的混合，**Mojo** 缺乏语言层级的一致性。
- **Python 超集包袱拖累了 Mojo**：成员们讨论了将 **Mojo** 描绘为 **Python 超集** 的影响，一些人认为这种*叙事*导致了不必要的元素，例如从 `libc` 复制的命名。
   - 澄清指出，其目标是便于通过查找和替换来移植基础代码，而不是实现与 CPython 的*“Bug 兼容性”*。
- **并发和 Sum Types 是 Mojo 的必备特性**：成员们对 **并发（concurrency）** 和 **Sum Types** 表现出浓厚兴趣，认为这是 **Mojo** 极度渴望的功能。
   - 提及了关于 [Structured Async 的 GitHub pull request](https://github.com/modular/max/pull/3945) 以及[另一个关于 Effect Handlers 的 PR](https://github.com/modular/max/pull/3946)，标志着这些领域的持续开发。
- **`is` 运算符的身份危机已解决**：一位成员寻求关于 **Mojo** 中 `assert_is` 函数中*身份（identity）*含义的澄清，询问它是否检查相同类型，另一位成员澄清这与内存位置有关。
   - 回复者澄清说，`is` 检查两个对象是否位于相同的内存位置，类似于指针相等，并链接到了 [Identifiable 文档](https://docs.modular.com/mojo/stdlib/builtin/identifiable/Identifiable/)。
- **Tensor 加法操作被移除**：一位成员报告说，在 **Mojo** nightly 版本中，`Tensor[float64]` 不再实现 `__add__` 方法，这是逐步淘汰 `Tensor` 以转向其他词汇类型（vocabulary types）计划的一部分。
   - 团队建议使用 `LayoutTensor` 以获得更高效的逐元素操作，详见[此提交信息](https://github.com/modularml/mojo/commit/SOME_COMMIT_HASH)。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AI 专家预见机器即将思考**：正如[这篇文章](https://ourworldindata.org/ai-timelines)所讨论的，许多 AI 专家预测人类水平的人工智能可能会在未来几十年内出现。
   - 人类水平 AI 被定义为能够执行人类可以完成的任何任务，并有能力选择让机器实现这些任务的行动。
- **Transformer 获得微分处理**：一份时事通讯强调了最近的 AI 研究，包括 [Differential Transformers](https://mail.bycloud.ai/)、混沌边缘的智能，以及为什么 **LLM** 可能无法真正推理。
   - 它还提到 **Byte Latent Transformers** 是无需分词（tokenization）的 **LLM** 的潜在未来。
- **Softmax 的不稳定性受到关注**：围绕 [LinkedIn 帖子](https://www.linkedin.com/posts/damienbenveniste_the-softmax-transform-might-be-one-of-the-activity-7301720641559269377-gDdQ) 的讨论显示，虽然 **softmax** 解决了溢出问题，但它可能会在梯度下降过程中加剧欠流（underflow）问题，可能导致模型卡住。
   - 最近的一些论文表明，欠流可能有助于 grokking 现象，作为一种隐式正则化器来防止过拟合。
- **双层优化（Bilevel Optimization）泛化了 Sparsemax？**：一位成员建议 **双层优化** 可能会泛化 **Sparsemax** 和 **Stablemax**，可能通过“领导者/跟随者”视角来看待整个 **ANN**。
   - 他们编写了一个 [BilevelMax 类](https://www.dataia.eu/sites/default/files/1%20Marco-Pedersoli%20ILLS.pdf) 来动态平衡稀疏性和密度，在 **Sparsemax** 和 **Softmax** 之间平滑过渡。
- **GATs 概览分享**：一位成员分享了 **Graph Attention Networks** (**GATs**) 的[概览](https://petar-v.com/GAT/)，这是一种在图结构数据上运行的神经网络架构，利用掩码自注意力层来解决之前基于图卷积方法的缺点。
   - 该概览包括图结构输入的激励示例，如分子网络、交通网络、社交网络和大脑连接组网络，并附有[原始论文](https://arxiv.org/abs/1710.10903)的链接。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **CoreWeave 拟收购 Weights & Biases**：**CoreWeave** 将以 **17 亿美元**收购 **Weights & Biases**，将 AI Hyperscaler™ 与领先的 AI 开发者平台相结合，详见[此新闻稿](https://www.prnewswire.com/news-releases/coreweave-to-acquire-weights--biases---industry-leading-ai-developer-platform-for-building-and-deploying-ai-applications-302392342.html)和[此文章](https://www.theinformation.com/briefings/coreweave-to-buy-weights-biases-for-1-7-billion)。
   - 此举标志着 **CoreWeave** 向 AI 开发工具领域的扩张，与其现有的基础设施服务形成互补。
- **CogView4-6B 发布**：[CogView4-6B](https://huggingface.co/THUDM/CogView4-6B) 是 THUDM 的最新模型版本，要求图像尺寸在 **512px** 到 **2048px** 之间且必须能被 **32** 整除，支持 **BF16** / **FP32** 精度。
   - 值得注意的是，根据模型卡显示，它在 **FP16** 下表现不佳，会出现溢出问题导致生成全黑图像。
- **道德 LLM 守口如瓶**：一位用户质疑 LLM 在摘要时是否应揭露敏感信息（如惊喜生日派对），引发了关于隐瞒关键信息的辩论，详见[此推文](https://x.com/TheXeophon/status/1896817938323341722)。
   - 共识倾向于 LLM 应当保守秘密，从而尊重**隐私和社交规范**。
- **微软 Health Futures 蓬勃发展**：微软研究院的 **Health Futures** 小组产出了大量优秀成果，特别是围绕*基于图像的多模态*应用。
   - 该小组还拥有 **Hoifung Poon** 和 **Tristan Naumann** 等资深 **NLP** 专家，专注于医疗保健领域。
- **Qwen 进化速度更快**：一篇论文（[arxiv 链接](https://arxiv.org/abs/2503.01307)）探讨了自我改进的 LLM，发现验证和回溯等**认知行为**是关键。相关讨论（[fxtwitter 链接](https://fxtwitter.com/gandhikanishk/status/1896988028893323675)）指出，在类似的 RL 训练下，**Qwen-2.5-3B** 超越了 **Llama-3.2-3B**。
   - 这表明某些架构选择或训练方法可能更有利于有效的自我改进。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LCPP 支持并行解码和草稿模型推测**：成员们注意到 **LCPP** 通过 `-np` 标志支持 [parallel decoding](https://github.com/ggerganov/llama.cpp) 功能的多用户操作。
   - 建议使用较小的草稿模型（如 **Llama 3.2 1B**）进行推测解码（Speculative decoding），并通过 `-md` 标志由较大的模型（如 **Hermes 3B**）进行校正。
- **Granite 3.1 3B 仍是快速工具化的首选**：**Granite 3.1 3B a800m instruct** 模型因其强大的 tool-calling 能力和 CPU 运行速度而受到推崇，特别适用于速度至上的编程任务。
   - 在速度为优先考虑因素时，它被认为是一个可靠的选择。
- **Grokking 泛化获得精度提升**：在讨论 **grokking** 时，成员们将延迟泛化归因于有限的精度、交叉熵损失和 LLM 训练期间的输出 softmax。
   - 提出的解决方案包括 **Orthograd**、stable softmax、将精度提高到 **FP64**，以及可能使用 Nvidia 的 **N-GPT** 或 **Muon**。
- **Langchain Agent 无法流式传输**：有用户报告在 `llama.cpp` 中使用 **Langchain Agent** 进行 tool-calling 时出现错误，显示为 `Cannot use tools with stream`。
   - 目前的解决方法是通过延迟输出直到工具调用完成后，来模拟流式传输。
- **受 Zettelkasten 启发的 Agentic Memory 系统发布**：一个基于 **Zettelkasten** 理念的新型 **Agentic Memory 系统**已在 [GitHub](https://github.com/WujiangXu/AgenticMemory) 上发布。
   - 此外，一个名为 **anon-kode** 的新工具也已在 [GitHub](https://github.com/dnakov/anon-kode) 上发布，允许使用任何 LLM 进行编程。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Gemini Flash 2.0 转录效果更好**：一位用户发现，在 NotebookLM 中使用 **Gemini 2.0 Flash** 进行音频转录的效果可能优于 **YouTube AI**，尤其是在处理播客音频文件时。
   - 他们概述了一个工作流：录制讲座，使用 NotebookLM 进行转录，使用 **Gemini Advanced** 进行精修，然后导入到 Google Docs。
- **通过 Google Cloud Speech-to-Text 获取 API 访问权限**：成员们探讨了 **NotebookLM API** 的访问方式，其中一人建议将 [Google Cloud 的 Speech-to-Text API](https://cloud.google.com/speech-to-text/v2/docs/chirp-model#:~:text=Chirp%20is%20the%20next%20generation,to%20more%20languages%20and%20domains.) 及其 **Chirp 模型** 作为潜在解决方案。
   - **Chirp 模型** 被指出是为 Google 产品提供支持的新一代语音模型。
- **Google Docs 同步**：成员们讨论了 Google Docs 与 NotebookLM 的更新同步问题，有人提到该平台会检测 Google Docs 的更新，然后提供“点击与 Google Drive 同步”的选项。
   - 用户对更精简的一键同步功能表现出兴趣。
- **生成的播客合法性辩论**：一位成员对生成的音频概览（Audio Overview）的合法性提出质疑，询问是否可以将其用于为公司制作播客。
   - 关于播客合法性问题，目前没有进一步的回应。
- **教导音频概览的主持人发音**：一位用户询问如何通过附加包含正确发音的源文件，来教导音频概览的主持人正确发音**希腊字母**。
   - 他们注意到主持人在阅读免疫学笔记时经常读错希腊字母。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 在清晰度上的巧妙尝试**：**Cohere For AI** 发布了 **Aya Vision** 模型的开放权重研究，包括 [320亿](https://huggingface.co/CohereForAI/aya-vision-32b) 和 [80亿](https://huggingface.co/CohereForAI/aya-vision-8b) 参数版本。
   - 这些模型针对视觉语言用例进行了优化，包括 **OCR、描述生成（captioning）、视觉推理、摘要、问答、代码**，并且是多语言的，在 **23 种语言** 中表现出色。
- **等级机器人上线**：等级机器人现已上线，开始为用户授予等级，初始等级为 **1, 5, 10, 20**。
   - 一位成员提到 **Cohere 网站** 的设计师*值得加薪*。
- **启动入站介绍**：鼓励新成员使用模板进行自我介绍，说明其 **公司/行业/大学**、当前工作、喜好的技术/工具以及在社区的目标。
   - 这有助于建立联系并提供个性化的介绍。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **在 WSL 上运行 Automatic1111：是否有性能顾虑？**：一位用户询问在 **WSL** 中运行 **Automatic1111** 与原生 **Linux** 相比在性能上是否有差异，另一位用户回答说，在 **Windows** 内的 **WSL** 上运行 **ComfyUI** 会消耗*额外的内存*。
   - 这取决于你的 **GPU** 性能，可能会产生差异，但未提供具体的基准测试或性能指标。
- **使用 Zluda 简化 AMD GPU 设置**：一位用户参考一年前的信息询问在 **Windows** 上使用 **AMD 显卡** 是否仍然困难。
   - 一位成员回答说，使用 **Zluda**，设置虽然需要时间，但运行平稳，且比*旧时代的 directml 快得多*。
- **Stable Diffusion 用户寻求指导**：一位患有精神障碍的成员请求耐心的指导，以在运行 **Ubuntu** 的 **AMD APU (5700G)** 上本地运行 **Stable Diffusion**。
   - 他们表示愿意就选择必要功能方面的协助支付报酬。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Glama MCP 服务器认领故障**：用户报告在 Glama.ai 上认领其 **MCP server** 时遇到问题，在 GitHub 身份验证期间由于 *invalid returnPath* 导致 `could_not_parse_params` 错误。
   - 聊天记录中未提供解决方案。
- **Twitter API 定价引发辩论**：围绕使用 **MCP** 连接 **Twitter** 进行推文生成展开了讨论，最初引发了对 **Twitter API 成本** 的担忧。
   - 一位成员建议 **Twitter 现在可能有免费层级**，这引发了对跨 **Facebook, X, 和 Telegram** 等平台跟踪 API 成本工具的兴趣。
- **Cursor 中的工具使用怪癖**：成员观察到 **roo** 或 **cursor** 并不总是优先使用可用工具，即使工具数量很少。
   - 建议包括 **更新工具描述** 以提高可用性，并指出详细的描述可以显著影响工具的有效性。
- **工具上下文学习 PR**：一位成员分享了一个 [GitHub pull request](https://github.com/modelcontextprotocol/specification/pull/188) 链接，涉及将 **Tool Call 和 Tool Result** 添加到 `GetPrompt` 中，用于工具使用的 in-context learning。
   - 另一位成员指出 *该 PR 中的 schema.ts 存在严重错误*，并表示希望为 JSON 结果提供可选的工具结果 schema。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Checkpointing 节省存储**：用户可以指定仅保存最后 **X 个 checkpoints** 以避免存储空间耗尽，基于步数的 checkpointing 正在开发中。
   - 新的 checkpointing 系统应包含一个 *“保存最后 n 个”* checkpoints 的选项。
- **Attention Masking 和 Label Masking 的区别**：**sft.py** 中创建的 mask 用于 loss 计算，而 attention 在 **SDPA** 中由于 *is_causal=True* 默认使用 causal mask。
   - 在前向传播与 loss 计算期间，可以对不同的 token 集合进行 masking。
- **自定义 Special Tokens 需要手动复制**：添加 **自定义 special tokens JSON** 时，最终的 checkpoint 和 epochs 文件夹接收的是非自定义版本。
   - 由于 checkpointer 代码不会自动在每个 epoch 的 checkpoints 中保存自定义的 `special_tokens` 文件，用户必须手动复制正确的版本。
- **QLoRA Recipes 关注 Metal 优势**：在配置中更新 **Environment.device** 可能会使 **QLoRA recipes** 针对 **Metal kernels**，因为 **AO** 现在支持 **MPS/Metal**。
   - 成员们正计划对 **MPS** 进行手动测试，重点关注 **1B-instruct 模型** 和用于生成的各种 **bit types**，参考 [torchchat 的量化文档](https://github.com/pytorch/torchchat/blob/main/docs/quantization.md#experimental-torchao-lowbit-kernels) 中的模式。
- **Checkpoints 持续 12 分钟？**：一位用户报告在未更改 checkpointer 设置的情况下，保存 **3B 模型** 需要等待 **12 分钟**。
   - 该用户请求 *“如果能为保存过程添加一个进度条就太好了，为了那些没耐心的人”*，一位成员同意在每个 `save_checkpoint` 中实现此功能。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud 现已全面上市 (GA)**：团队宣布 **LlamaCloud** 现已全面上市 (Generally Available)，为非结构化数据上的 agentic 知识管理提供开箱即用的解决方案，可通过[此链接](https://t.co/1CSRJm30e3)访问。
   - 这应该会使跨不同数据格式管理知识变得更加容易。
- **Hugging Face 教授 LlamaIndex Agents**：**Hugging Face** 创建了一个关于使用 **LlamaIndex** 构建 agents 的教育课程，涵盖了组件、**RAG**、tools、agents 和 workflows，可通过[此链接](https://t.co/eACAJzXg8y)找到。
   - 该课程应有助于进一步增加采用率并降低学习曲线。
- **DeepSeek API 余额不足**：一位成员报告在将 **DeepSeek API** 与 **LlamaIndex** 配合使用时，遇到了 **402** 错误代码的 `openai.APIStatusError`，提示 *'Insufficient Balance'*。
   - 另一位成员建议该问题源于用户账户缺少额度或未设置支付方式，与 **LlamaIndex** 本身无关。
- **Postgres 长示例已修复**：一位成员指出 **Postgres vector store example** 文档页面上的输出过长，可通过[此链接](https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/)访问。
   - 团队承认了该问题并已通过 [PR #18002](https://github.com/run-llama/llama_index/pull/18002) 进行了修复。
- **Windsurf Checkpointing 缺失**：一位成员询问 **Windsurf** 中的 checkpoint 功能，指出 *没有办法* 回到之前的 checkpoint。
   - 用户发现 *没有办法* 回到之前的 checkpoint，该功能似乎目前缺失。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **辩论 ChatGPT 的合法性**：成员们正在研究法律领域对**印度推理基础模型**的需求，并询问使用印度案例微调 **ChatGPT** 是否足以解决其在美利坚合众国案例上训练所带来的问题。
   - 核心问题在于，对于印度法律的实际应用，微调是否能充分解决因 **ChatGPT** 在美国法律原则上训练而产生的推理偏差。
- **挖掘 Adam-Matching 的起源**：*modded-nanogpt speedrun* 的早期版本使用了类似于 [kimi paper](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101024_Muon/eb5659d0-fb6a-49e5-a311-f1f89412f726.txt) 的 adam-matching 缩放，采用了 **0.1** 的缩放因子。
   - 随后的 *modded-nanogpt speedrun* 迭代中，对于 **qkvo matrices** 使用了 `max(1, g.size(0)/g.size(1))^0.5` 代替 `max(g.size(0), g.size(1))^0.5`，这影响了 **c_fc matrices** 的更新幅度。
- **调试数据集加载动态**：关于 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/__main__.py#L367) 中 `--trust_remote_code` 和 `dataset_kwargs` 的讨论确认，`--trust_remote_code` 仅在显式传递参数时激活。
   - 数据集加载问题追溯到**额外的 dataset_kwargs** 覆盖了子任务配置，该问题已通过 **Hugging Face load_datasets** 库解决，具体位置在 [此处](https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/api/task.py#L930)。
- **寻求可复现的 Llama 3 结果**：社区在思考是否需要另一种评估方案（evaluation recipe）来镜像 **Llama 3 paper** 中展示的结果。
   - 这一讨论强调了将社区评估与模型开发者报告的结果保持一致的努力。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **ReAct Agents 引发编排讨论**：辩论了 **ReAct agents** 对于编排（orchestration）的必要性，有建议认为分类对于简单任务可能足够，但对于复杂的后续多步任务可能无效。
   - 一位成员正在开发一种编排方法，该方法结合了工具和知识库来管理复杂的对话。
- **OctoTools 框架管理工具交互**：来自斯坦福的 [OctoTools](https://octotools.github.io/) 使用**工具卡（tool cards）**、**规划器（planner）**和**执行器（executor）**来管理工具交互并生成最终答案，从而优化特定任务的工具集。
   - 该框架的**工具卡**定义了工具使用的元数据并封装了异构工具，这有助于无需训练即可集成新工具。
- **Agentic Reward Modeling 集成人类偏好**：[Agentic Reward Modeling](https://github.com/THU-KEG/Agentic-Reward-Modeling) 旨在将人类偏好与可验证的正确性信号相结合，以构建可靠的奖励系统。
   - 一位成员在其 **minionS** 实现中加入成本优化功能的 [PR](https://github.com/stanfordnlp/dspy/pull/7891) 被 [DSPy framework](https://github.com/jmanhype/dspy) 拒绝。
- **dspygen 和 Spark 启发工具开发**：成员们从 [dspygen](https://github.com/seanchatmangpt/dspygen/blob/main/src/dspygen/mixin/hsm/hsm_mixin.py) 和 [Spark](https://hexdocs.pm/spark/get-started-with-spark.html) 中获得了工具开发灵感。
   - 一位用户考虑在 **Axon** 中创建一个 **DSL** 或类似的接口，灵感源自 **PyTorch**。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 追赶 Ollama**：成员们希望 [GPT4All](https://gpt4all.io/) 能够赶上 [Ollama](https://ollama.com/)，希望看到 GPT4All 处于领先地位。
   - 虽然没有提到落后的具体原因，但成员们表达了希望看到其改进的愿望。
- **小型模型通过 RAG 获得增强**：一位成员澄清说，由于速度优势，某些**小型模型**在配合 **RAG** 使用时表现更好。
   - 他们警告说，如果不使用 **RAG**，该模型可能会产生大量的**编造（confabulate）**内容。
- **Llama3-8b 与 LocalDocs 的能力**：成员们报告称，模型的能力受其参数数量、架构和训练数据的限制，并建议 **Llama3-8b** 与 **LocalDocs** 结合使用效果非常好。
   - 目前没有提供具体的基准测试或指标来支持 **Llama3-8b** 表现优异的说法。

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **服务器维护确认**：一位成员询问在赞助者区域（sponsors zone）有活动的情况下，服务器是否仍在维护，这引发了快速的澄清。
   - 另一位成员确认，*是的，当然* 仍在维护中。
- **赞助者区域持续活跃**：成员们目睹了赞助者区域持续不断的活动。
   - 这些活动引发了关于服务器是否正在被积极维护的疑问。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI 股市 Agent 工作坊发布**：一场关于构建 **AI Stock Market Agent** 的工作坊定于 **IST 时间 3 月 7 日星期五晚上 9 点**举行，教授参与者 AI 如何快速分析超过 1000 只股票，注册地址在 [这里](https://lu.ma/0ckb8tp0)。
   - 该工作坊旨在展示 **AI** 如何改变投资格局，并为更明智的投资决策提供工具。
- **AI 与金融的完美结合**：工作坊打算揭示 **AI** 如何彻底改变投资，并提供 **AI** 预测市场趋势的真实案例。
   - 参与者将发现 **AI** 如何改变投资格局，并获得辅助明智投资决策的工具。
- **构建 AI 投资伙伴，无需代码**：工作坊将指导参与者在无需编码的情况下构建 **AI** 工具来分析股票，从而在没有真实资金风险的情况下测试投资想法。
   - 它强调了利用 **AI** 制定投资策略的初学者友好方法。
- **AI 在行动：现实世界的成功案例**：工作坊将探讨大投资者如何利用 **AI** 做出更明智的选择，以及 **AI** 如何辅助知情的投资决策。
   - 课程包括对现实世界成功案例的探索以及 **AI** 在金融领域的实际应用。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该频道长时间保持安静，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间保持安静，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持安静，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}




### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1346332550934827018)** (608 条消息🔥🔥🔥): 

> `Cursor 的语音输出、Claude 3.7 性能问题、Windsurf 作为替代方案、o3-mini 的 MCP 工具、Repo Prompt 对比 Grok 3 Web` 


- **Cursor 中的语音输出：白日梦？**：一位成员建议在 Cursor IDE 中集成语音输出，以朗读模型的响应，但在即使使用 **GPT 3.7** 也难以处理 **tool usage 和网页搜索** 后，很快就放弃了这个想法，认为这很*愚蠢*。
   - 该用户发现很难让模型遵循精确的指令并有效地利用给定的 Python 工具。
- **Claude 3.7 停滞和变慢**：用户报告说 Cursor 中的 **Claude 3.7** 不仅速度慢，而且经常在中途停止，导致不稳定性，并促使一些人寻求 [Windsurf](https://windsurf.so/) 等替代方案。
   - 一位用户指出，*是的，3.7 目前非常不稳定*，这与其他用户关于生产力较几个月前下降的体验一致。
- **MCP 工具在 o3-mini 上失败**：一位用户观察到 **o3-mini** 缺乏利用 MCP 工具的能力，尽管尝试明确指示它这样做。
   - 他们发现 **Claude 3.5 Haiku** 在工具使用和指令遵循方面更胜一筹，而其他人认为它仅在规划方面表现良好，并倾向于通过 Python 工具将其与 **r1 reasoner** 或 **o3 mini** 结合使用。
- **Repo Prompt + Grok 3 组合**：成员们讨论了利用 **Repo Prompt** 和 **Grok 3 Web** 进行规划和应用代码更改，特别是在 Cursor 遇到困难时。
   - 一位用户分享了一个工作流视频 ([https://www.youtube.com/watch?v=09sUdLtRAlc](https://www.youtube.com/watch?v=09sUdLtRAlc))，演示了在 Web 端使用 Claude 3.7 进行多文件编辑，并将其用于规划 + Claude Web 生成 XML diffs，以便通过 Repo prompt 应用。
- **退订 Agent 创意激发创造力**：受管理订阅困难的启发，用户们集思广益，想要创建一个专门设计用于自动取消订阅的 Agent，可能作为一个 SaaS 产品。
   - 一位用户兴奋地宣布 *已经开始做了* 并分享了截图，而另一位用户建议在 M1 Max 上利用本地 LLM 进行低成本开发。


<div class="linksMentioned">

<strong>提到的链接</strong>:

</div>

<ul>
<li>
<a href="https://browsertools.agentdesk.ai/installation">Installation - AgentDesk - BrowserToolsMCP</a>: 未找到描述</li><li><a href="https://repoprompt.com/">Repo Prompt</a>: 未找到描述</li><li><a href="https://www.agentdesk.ai/prompts">AgentDesk</a>: 未找到描述</li><li><a href="https://repomix.com/">Tweet from Repomix</a>: 将你的代码库打包成 AI 友好的格式</li><li><a href="https://www.youtube.com/watch?v=09sUdLtRAlc">Repo Prompt Apply Demo</a>: 很多人问我 Apply 是如何工作的，这里有一个演示。Repo Prompt 在 macOS 上的 Beta 测试期间是免费的 https://repoprompt.com/#chatgpt #codingjourney</li><li><a href="https://tenor.com/view/fuck-angry-cubicle-office-fuck-everything-gif-5246517">Fuck Angry GIF - Fuck Angry Cubicle - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://valiant-sassafras-4f5.notion.si">no title found</a>: 未找到描述</li><li><a href="https://x.com/lmarena_ai/status/1896590150718922829?s=46">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: GPT-4.5 在所有类别中全面夺冠，在 Multi-Turn 中具有明显的领先地位。🥇 Multi-Turn 💠 Hard Prompts 💠 Coding 💠 Math 💠 Creative Writing 💠 Instruction Following 💠 Longer Query</li><li><a href="https://github.com/dnakov/anon-kode">GitHub - dnakov/anon-kode: koding with any LLMs</a>: 使用任何 LLM 进行编码。通过在 GitHub 上创建账号来为 dnakov/anon-kode 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=Qw2807PSZ5I">The Vibe Coding Workbench - Demo</a>: 像 v0.dev 和 Cursor 这样 AI 辅助编码产品令人兴奋 —— 但我们如何将它们集成到我们的开发流程中，以便我们可以移交我们的成果...</li><li><a href="https://github.com/browserbase/stagehand">GitHub - browserbase/stagehand: An AI web browsing framework focused on simplicity and extensibility.</a>: 一个专注于简单性和可扩展性的 AI 网页浏览框架。 - browserbase/stagehand</li><li><a href="https://x.com/pvncher/status/1894559704065409224?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from eric provencher (@pvncher)</a>: Apply 模式是 Repo Prompt 最棒的部分之一，由于它的呈现方式，它似乎让人们感到畏惧。但它非常强大！以下是你如何使用它...</li><li><a href="https://github.com/AnthusAI/Vibe-Coding-Workbench">GitHub - AnthusAI/Vibe-Coding-Workbench: A starter kit for business leaders with a vision. Transform your business ideas into functional prototypes without coding expertise. Create working software that developers can easily refine and integrate, with best practices and agile principles built in at the core.</a>: 为有远见的商业领袖准备的入门套件。无需编码专业知识即可将您的商业创意转化为功能原型。创建开发人员可以轻松改进和集成的可用软件，核心内置了最佳实践和敏捷原则。</li><li><a href="https://tenor.com/view/kermit-gun-kermit-gun-gif-16355306064126846866">Kermit Gun GIF - Kermit Gun Kermit gun - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/AnthropicAI/status/1764653830468428150">Tweet from Anthropic (@AnthropicAI)</a>: 今天，我们宣布推出 Claude 3，我们的下一代 AI 模型。这三个最先进的模型 —— Claude 3 Opus、Claude 3 Sonnet 和 Claude 3 Haiku —— 在推理等方面树立了新的行业基准...</li><li><a href="https://github.com/kleneway/pastemax">GitHub - kleneway/pastemax: A simple tool to select files from a repository to copy/paste into an LLM</a>: 一个简单的工具，用于从代码库中选择文件并复制/粘贴到 LLM 中 - kleneway/pastemax
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1346337486594375705)** (204 条消息🔥🔥): 

> `在 MIG 实例 GPU 上运行 Deepseek Unsloth，24b Mistral 模型的 4bit 合并错误，Unsloth 与 Phi-4-mini-instruct 的兼容性，使用 GGUF 格式和 Koboldcpp 运行模型，关于 Unsloth 技巧和最佳实践的每周 Google Meet` 


- **MIG GPU 问题出现**：一名成员询问是否有人在 **MIG 实例 GPU** 上运行 **Deepseek Unsloth**。
   - 另一名成员要求澄清 “MIG” 的含义。
- **Mistral 24B 的 4-bit 合并故障**：一名成员在使用 **Unsloth 的 notebook 模板**尝试对 **24B Mistral 模型**进行 4-bit 合并时，遇到了与 `Params4bit.__new__()` 相关的 `TypeError`。
   - 他们怀疑是 Torch 的版本不兼容问题，并计划尝试不同的版本，认为 4bit 合并代码可能存在某种问题。
- **Phi-4 与 Unsloth：修复 Bug 后的完美结合？**：一位用户报告了在尝试通过 **Unsloth** 下载和使用 **Phi-4-mini-instruct 模型**时出现的错误，引发了关于其兼容性的讨论。
   - 有人指出 [Unsloth 的 Phi-4 版本](https://huggingface.co/unsloth/Phi-4-mini-instruct) 包含了 Bug 修复，并提供了 **Phi-4 版本**集合的链接以及用于微调的 **Google Colab notebook**。
- **Koboldcpp 成为运行 GGUF 模型的“秘诀”**：一位用户询问其他人如何在他们的系统上运行模型，引发了关于不同策略的讨论。
   - 一位用户表示他们使用 [Koboldcpp](https://github.com/LostRuins/koboldcpp)，因为它允许用户轻松运行 GGUF 模型，而其他人则提到了 **exllama** 和 **vllm** 以获得更好的性能。
- **Unsloth 支持小组：每周见个面？**：一名成员提议组建一个 **每周 Google Meet**（或 Zoom/Discord）小组，分享关于运行 **Unsloth**、改进微调以及创建高质量数据集的技巧、建议和最佳实践。
   - 几位成员表示感兴趣，并建议采用包含演示、问答和开放式讨论的结构化形式，而另一名成员提到他们的非营利组织正在提供类似的研讨会（但不是在 Google 或 Zoom 上，而是在 Discord 上）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/reasoning-course">reasoning-course (Hugging Face 推理课程)</a>：未找到描述</li><li><a href="https://github.com/Green0-0/Agent2/tree/main">GitHub - Green0-0/Agent2: agent2</a>：agent2。通过在 GitHub 上创建账号来为 Green0-0/Agent2 的开发做出贡献。</li><li><a href="https://github.com/LostRuins/koboldcpp">GitHub - LostRuins/koboldcpp: 使用 KoboldAI UI 轻松运行 GGUF 模型。单个文件。零安装。</a>：使用 KoboldAI UI 轻松运行 GGUF 模型。单个文件。零安装。 - LostRuins/koboldcpp</li><li><a href="https://huggingface.co/unsloth/Phi-4-mini-instruct/">unsloth/Phi-4-mini-instruct · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1346340706956087378)** (45 条消息🔥): 

> `Triton 社区, 不可变 Linux 发行版, 程序员的新 AI 职位` 


- **需要 Triton 社区讨论**：成员们正在请求建立一个 **Triton 讨论社区**，以分享技巧/窍门并讨论 **文档问题**。
   - 一位成员建议将 **GPU MODE Discord 中的 Triton 频道** 作为一个相当活跃的资源，并链接了一个 [GitHub 上的 Triton 资源列表](https://github.com/rkinas/triton-resources)。
- **不可变 Linux 发行版兴起**：成员们正在讨论诸如 [Bluefin](https://projectbluefin.io/) 和 [Xenialinux](https://wiki.xenialinux.com/en/latest/) 等 **不可变 Linux 发行版**，并预测不可变发行版将在 **3-5 年内** 成为主流。
   - 其他人指出，像 **CoreOS** 这样的发行版是同类中的首创，使用了 **双分区系统/grub**，但在被 Red Hat 收购后“变得一团糟”。
- **普通程序员的新 AI 职业选择**：一位成员询问在 AI 兴起的背景下，普通程序员有哪些新的职位选择。
   - 具体而言，他们提到了对 **Triton 等 AI 新职业方向** 的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://itsfoss.com/immutable-linux-distros/">12 款面向未来的不可变 Linux 发行版</a>：不可变性是一个流行趋势。看看你有哪些不可变 Linux 发行版的选择。</li><li><a href="https://projectbluefin.io/">Bluefin</a>：下一代云原生 Linux 工作站，专为可靠性、性能和可持续性而设计。</li><li><a href="https://github.com/rkinas/triton-resources">GitHub - rkinas/triton-resources: 学习和探索 Triton 的精选资源列表，Triton 是 OpenAI 用于编写高效 GPU 代码的编程语言。</a>：学习和探索 Triton 的精选资源列表。 - rkinas/triton-resources</li><li><a href="https://wiki.xenialinux.com/en/latest/">来自 Xenia Linux 文档的推文</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1346361835087073281)** (50 条消息🔥): 

> `Mistral BOS token, Torch 2.4 与 Python 3.12, Llama3-8b 上的 GRPO, 微调 BERT, 在 Windows 上导出至 Ollama` 


- **Mistral 模型获得额外的 BOS token**：一位用户发现使用 **Mistral** 时，尽管输入数据只包含一个 **BOS token**，但训练数据中出现了两个，并询问训练是否仍能正常进行。
   - 另一位用户确认了同样的问题和解决方法，即在训练前打印 tokenizer 进行健全性检查（sanity check）。
- **Torch 2.4 提升 Python 3.12 性能**：一位用户询问 Python 3.12 对 **torch.compile** 的支持情况，另一位用户回答说需要 **Torch 2.4 或更高版本**。
   - 一位用户看到团队更新了一些源代码，但现在模型生成的响应出现了一些问题，正在寻求调试帮助。
- **在 Llama3-8b 上训练 GRPO 出现挑战**：一位用户报告称，在 **Llama3-8b** 上训练 **GRPO** 时，完成长度（completion length）和正确奖励（correct reward）没有增加。
   - 他们附带了显示完成长度问题的 [图表](https://cdn.discordapp.com/attachments/1179777624986357780/1346656044058808353/length.png?ex=67c8faac&is=67c7a92c&hm=20d71ba56e79387d6d40dd1a28651d148612265562a7cba55abf7320a431d4c4&) 和另一张描述奖励问题的 [图片](https://cdn.discordapp.com/attachments/1179777624986357780/1346656273076322435/image.png?ex=67c8fae3&is=67c7a963&hm=2e40f721f2feb17f7d44599136eacbb19d5d3161df966fe52e87ab146bb191bd&)。
- **澄清 BERT 微调疑问**：一位用户询问使用 **Unsloth** 微调 **BERT** 的可能性，但质疑这是否是一种有效的方法，因为 BERT 主要用于生成 Embedding。
   - 对方澄清说 **目前不支持** 微调 BERT，不过有人正考虑在有时间时在 Unsloth 中实现它。
- **Windows 用户疑问：如何导出 Ollama？**：一位用户寻求关于如何在 **Windows 平台** 上将微调后的模型导出为 **Ollama** 格式的指导。
   - 他们指出，现有的官方文档主要涵盖了该流程的 Linux 版本。



**提到的链接**：<a href="https://docs.unsloth.ai/get-started/all-our-models">我们所有的模型 | Unsloth 文档</a>：未找到描述

  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1346591787573317682)** (33 条消息🔥): 

> `Llama 3.1 8B Fine-tuning, GPT4o-mini 用于合成数据生成, 训练数据生成技巧, Ollama 上的 KoloLLM, Fine-tuning vs. RAG` 


- **Llama 3.1 8B 获得 Fine-tuning**: 一位成员正在对 **Llama 3.1 8B** 进行 Fine-tuning，并使用 **GPT4o-mini** 进行合成数据生成。
   - 他强调训练数据是主要的驱动力。
- **Kolo 指南开源训练数据技巧**: 该成员分享了[他的指南链接](https://github.com/MaxHastings/Kolo/blob/main/GenerateTrainingDataGuide.md)，内容关于训练数据生成，并强调这些技巧涉及一些*微小而正确的决策*，从而充分利用 **LLM** 为高质量问题生成详尽回答的能力。
   - 他还提到他已将 **KoloLLM** 上传到 [Ollama](https://ollama.com/MaxHastings/KoloLLM)，并表示愿意帮助生成训练数据，以便在不同领域复现他的结果。
- **Fine-tuning 赋予超能力**: 该成员对使用高质量数据和良好的训练参数进行 Fine-tuning 表示兴奋，称其*就像拥有了超能力*，并通过一个简单的配置文件实现流程自动化，他称之为*拒绝 RAG 废话*。
   - 他强调 Fine-tuning 将信息注入 **LLM** 的大脑，用户永远不必担心复杂的 **RAG** 系统是否每次都能可靠运行。
- **Fine-tuning 加上 RAG**: 另一位成员指出 **LLM** 不会逐字学习，因此这通常是一把双刃剑，大多数情况下最好同时使用两者来确保回答的准确性（grounding）。
   - 原作者回应称，他可能不需要 **RAG**，因为他可以通过寻找训练数据来填补空白，从而解决出现幻觉的情况。
- **什么时候 RAG 更合适？**: 另一位成员建议，如果你需要 100% 的准确率，无论是否使用 **RAG**，都不应该使用 **LLM**。但 **RAG** 在法律、财政或研究等需要良好检索能力的领域非常有用。
   - 原作者表示，他认为 **RAG** 对于不断变化或非常新的信息总是很有用的，但 Fine-tuning 应该用于静态信息。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ollama.com/MaxHastings/KoloLLM">MaxHastings/KoloLLM</a>: 一个在 Kolo 上经过 Fine-tuning 的 Llama 3.1 8B 领域专家</li><li><a href="https://github.com/MaxHastings/Kolo/blob/main/GenerateTrainingDataGuide.md">Kolo/GenerateTrainingDataGuide.md at main · MaxHastings/Kolo</a>: 在本地 Fine-tuning LLM 的最快方法。通过在 GitHub 上创建账号为 MaxHastings/Kolo 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1346375115084005388)** (21 条消息🔥): 

> `GRPO 方法对比，DeepSeek r1 发布，用于合成数据的 Bespoke Curator，Triton GPU Kernel，Unsloth 视频会议` 


- **GRPO 方法缺乏实质性对比**：成员们讨论道，*各种 **GRPO 方法** 之间几乎没有实质性的对比，我们看到的只是由于某种原因，GRPO 以更快的 KL divergence（KL 散度）更快地获得了更大的奖励（难道是糟糕的超参数设置？）*。
   - 一位成员建议需要 Benchmark 来观察其优势，并指出 **RLOO** 的表现也比 **PPO** 更快，参考了[这篇博客文章](https://kalomaze.bearblog.dev/why-does-grpo-work/)。
- **DeepSeek r1 赶超前沿**：经过一年的迭代，**DeepSeek** 发布了 **DeepSeek r1**，在最新的预训练运行（**DeepSeek-V3**）之后，已经赶上了 **LLM** 领域的前沿水平。
   - 此次发布引发了关于提升性能背后的训练技术进步的推测，一些人认为集成了 **Monte Carlo tree search** 等算法。
- **Bespoke Curator 合成 AI 数据**：由 Bespoke Labs 开发的一个名为 **Bespoke Curator** 的开源 Python 库，旨在为 **AI 模型 fine-tuning** 和**结构化数据提取**提供合成数据集的创建和策展功能。
   - 它提供可编程性、结构化输出，并与 **Hugging Face Dataset** 对象集成，使用交互式 Curator Viewer 进行实时检查，正如其[官网](https://www.bespokelabs.ai/)所述。
- **Tilelang Kernel 提速**：一位成员分享了一条关于 **tilelang kernels** 的推文链接，声称只需 80 行代码即可在 **H100** 上实现 **DeepSeek FlashMLA** 95% 的性能（比 **Triton** 快 500%）。
   - 其他人好奇为什么有人只想要 FlashMLA 95% 的性能，原帖作者澄清说 [Tilelang](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_mla) 是一种语言。
- **Unsloth 会议开启录制**：人们对录制 **Unsloth 会议**并在语音频道分享表现出兴趣，**OBS Studio** ([https://obsproject.com/](https://obsproject.com/)) 作为一款免费开源软件被列入调研范围。
   - 另一个建议的选项是 **Scribe** ([https://scribehow.com/](https://scribehow.com/))，这是一款付费软件，有助于通过文档记录和分享团队专业知识。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.bespokelabs.ai/">Bespoke Labs</a>：未找到描述</li><li><a href="https://www.youtube.com/playlist?list=PLPefVKO3tDxOJLAmCA75uShbe1z_RNqkQ">Triton GPU Kernels 101</a>：为已精通 PyTorch 但缺乏 GPU 硬件知识的人准备的 Triton 课程</li><li><a href="https://obsproject.com/">Open Broadcaster Software | OBS</a>：未找到描述</li><li><a href="https://x.com/Lei_Wang_1999/status/1896625837782475231?t=INByKNCTjnhy5DCDZ2Orfg&s=19">Lei Wang (@Lei_Wang_1999) 的推文</a>：看看今天的 tilelang kernel，只需 80 行 tilelang kernel 代码，你就可以在 H100 上获得 DeepSeek FlashMLA 95% 的性能（比 Triton 快 500%！）：https://github.com/tile-ai/tilelang/tree/main/ex...</li><li><a href="https://kalomaze.bearblog.dev/why-does-grpo-work/">为什么 GRPO 有效？</a>：DeepSeek。现在 LLM 领域的人都知道这个名字，而且理由充分；在他们的团队低调地迭代架构一年多之后……</li><li><a href="https://kalomaze.bearblog.dev/grpo-judge-experiments-findings-and-empirical-observations/">GRPO Judge 实验：发现与经验观察</a>：网上许多用于 LLM 强化学习的 GRPO 复现都缺乏关于超参数和 reward shaping 的有用直觉或建议……</li><li><a href="https://scribehow.com/">Scribe | 快速创建分步指南</a>：Scribe 为您记录流程。立即构建包含文本、链接和截图的可视化指南。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1346331416421601280)** (314 messages🔥🔥): 

> `Android vs iOS 功能差异, Perplexity Pro 定价问题, UI 故障, Augment Code, Perplexity AI 中的 Deep Research` 


- **Android 功能很滑稽，iOS 才是高级版**：一位成员开玩笑说 **Android** 版本很滑稽，随后另一位成员回应称新的语音功能仅限 **iOS**，且 **iOS** 更显高级，这引发了一些讨论。
   - 另一位成员回复称，这*并非阶级歧视*，更像是 *LLM 网站间的通用标准*。
- **用户对 Perplexity Pro 定价问题感到沮丧**：用户对 Perplexity Pro **200 美元**的价格表示不满，认为无论服务质量如何，这个价格都过高了。
   - 讨论延伸到 Pro 订阅是否物有所值，特别是如果大多数搜索都使用 **Sonar** 模型以降低成本，而且一年的 Pro 会员像发糖一样到处派发。
- **Perplexity UI 经常出故障，Pro Search 始终是 Sonar 模型**：成员们报告了 UI 故障：在 Web 端应用的 **Pro Search** 对话中点击 **Rewrite** 并选择不同的模型（如 Sonnet 或 Gemini）时，实际上并没有切换模型。
   - 用户抱怨在 **Pro Search** 中模型默认使用 **Sonar**，且 UI 上无法看到模型名称，在重写（rewrite）过程中也无法更改底层模型。
- **用于编程的 Augment Code**：成员们讨论了 **Augment Code**，它旨在有效处理超大型企业级代码库，与 **Cursor** 等在本地进行索引的工具形成对比。
   - 他们指出 **Augment Code** 在其服务器上对代码库进行索引，为 AI 模型提供对代码库的全面访问权限，并讨论了其[定价和试用选项](https://www.augmentcode.com/pricing)。
- **Deep Research 确实够深**：用户讨论了他们是否觉得 Perplexity Pro 的 **Deep Research** 模式在搜索信息和知识方面表现良好。
   - 他们回复称深度确实非常好，涵盖了所有主要的重点，但输出长度比 **OpenAI** 短。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arc.net/l/quote/kbufudwj">引用自 “turbopuffer: 对象存储上的快速搜索”</a>：未找到描述</li><li><a href="https://www.augmentcode.com/pricing">定价 | Augment Code</a>：Augment Code 不仅仅是一个工具，它还能加速你的团队。了解为什么从领先的初创公司到财富 500 强组织都选择 Augment。</li><li><a href="https://x.com/monnef/status/1896817782412792154">mennof (@monnef) 的推文</a>：我看到 @perplexity_ai 正致力于让事情变得更糟，这是一场 UI 退化之旅：* 全名显示清晰 😊* 然后隐藏在悬停中 🙄* 被替换为 “Pro Search” 🤢* 隐藏在 ... 后面</li><li><a href="https://www.augmentcode.com/">Augment Code – 真正用于工作的开发者 AI</a>：体验真正理解你代码库的 AI 平台。我们的开发者 AI 帮助团队更快地编写代码，做出更明智的决策，并解锁集体知识。立即免费试用。</li><li><a href="https://www.reddit.com/r/Astronomy/comments/dwfdfq/grand_map_of_the_night_sky/#lightbox">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://x.com/perplexity_ai/status/1890452005472055673">Perplexity (@perplexity_ai) 的推文</a>：在 Perplexity 上推出 Deep Research。Deep Research 让你能够针对任何主题生成深入的研究报告。所有人均可免费使用——非订阅用户每天最多 5 次查询，订阅用户每天 500 次查询...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1346382289281482775)** (5 messages): 

> `Gen 解释, Apple Air 产品曝光, TSMC 投资计划, 投资建议书, 推理型 LM 的 SOTA 与未来` 


- **Gen 的简要说明即将到来！**：一位用户请求关于 [Gen](https://www.perplexity.ai/search/explain-in-brief-about-the-gen-559BcexOTEyTKOUp6l71NQ) 的简要解释。
- **新款 Apple Air 产品曝光**：一位用户分享了关于曝光的 [Apple Air 产品](https://www.perplexity.ai/page/apple-air-product-teased-QhTieZlcTwWodiMLzGzP3g)的链接。
- **TSMC 在美巨额投资**：一位用户发布了关于 [TSMC 计划在美国投资 1000 亿美元](https://www.perplexity.ai/page/tsmc-plans-100-billion-us-inve-8m1ORdvqQlev._StpFWN8w)的消息。
- **撰写投资建议书**：一位用户请求协助撰写一份[投资建议书](https://www.perplexity.ai/search/write-an-investment-thesis-on-FoN2WGDESQ6ShLyTajmPIA)。
- **推理型 LM 的技术前沿 (SOTA)**：一位用户分享了关于[推理型语言模型的技术前沿与未来](https://www.perplexity.ai/search/reasoning-lm-sota-and-future-d-.fkmD_aXRrK80ruahJ_Rrw)的链接。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1346370886269145129)** (9 条消息🔥): 

> `Perplexity AI 定价、Obsidian Web Clipper 集成、Perplexity API JSON 输出问题` 


- **针对“按深度查询付费”模式的 Perplexity 方案**：用户请求一个带有基于用量定价（例如 **每小时 $5-10**）的“持续深化”功能，以便对科学和专业主题进行更深入的研究和信息处理。
   - 该用户还建议增加一个查询多个 AI 并整合信息的选项，类似于 [FreedomGPT](https://freedomgpt.com/) 提供的功能。
- **Obsidian Web Clipper 考虑集成 PPLX**：用户提议将 **Perplexity AI 的 LLM** 集成到 [Obsidian Web Clipper](https://github.com/obsidianmd/obsidian-clipper/issues/376) 中。
   - 该用户认为这将非常有价值，并希望有更高技术水平的人尝试进行集成。
- **Sonar Reasoning Pro 模型在 JSON 输出方面遇到困难**：用户报告了在 Tier 3 账户上使用 **sonar-reasoning-pro 模型** 时，**Perplexity API** 的 `response_format` 选项存在问题。
   - 响应结果在 **JSON 输出** 之前意外地包含了 `<think>Some thinking text</think>`，导致用户质疑是 **API** 调用方式错误，还是推理模型未能正确支持 `response_format` 参数。



**提到的链接**：<a href="https://github.com/obsidianmd/obsidian-clipper/issues/376">将 Perplexity AI 的 LLM 集成到 Obsidian Web Clipper · Issue #376 · obsidianmd/obsidian-clipper</a>：将 Perplexity AI 的 Large Language Models (LLM) 集成到 Obsidian Web Clipper 将大有裨益。与目前在 Obsidian Web Clipper 中可用的其他模型不同，Perplexity...

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1346476720408231957)** (2 条消息): 

> `WPF 扩展问题、Visual Studio、Xcode 扩展错误` 


- **Visual Studio 中的 WPF 扩展故障**：一名成员报告称 **Visual Studio** 的 **WPF 扩展** “非常糟糕”。
- **Xcode 扩展面临错误**：一名成员在使用 **Xcode** 时收到了 “发生内部错误” 的消息，错误 ID 为 **a9de9711b9ed431297eb00a945415d47**。


  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1346334397535617105)** (230 条消息🔥🔥): 

> `Windsurf 字体大小调整，Flex Credits 定价合理性，Windsurf 性能问题，Claude 3.7 的 Flow Credits 消耗，基于团队的 Cascade/Windsurf 实施` 


- **Windsurf 字体大小调整探索**：一位用户询问如何调大 Windsurf 中的字体大小，另一位用户建议点击*右上角的小方块*来调整设置。
- **Windsurf Flex Credit 定价困惑**：一位用户对 Flex Credits 的定价提出质疑，指出 **2,000 credits (500 prompt + 1,500 flow)** 售价为 **$15**，而单独 **300 flex credits** 就要 **$10**；对此另一位用户称*它们根据需求被用作 Prompt 或 Flow 操作*。
- **Windsurf 遭遇性能波动**：用户报告了 Windsurf 的一些问题，包括持续显示的 'i' 图标、长时间无响应、控制台错误（如 *couldn't create connection to server*）以及整体反应迟钝。
   - 一位遇到持续错误的用户切换到 Google 的 DNS 后发现问题得到了解决。
- **用户抱怨 Claude 3.7 消耗大量 Flow Credits**：多位用户抱怨 **Claude 3.7** 模型消耗 **Flow Credits** 速度极快，导致 Windsurf 难以支撑日常使用，尤其是考虑到即使是简单的任务，Credits 也会被迅速耗尽。
   - 一位用户提到 *Windsurf 每次都像个傻瓜一样从头开始读取代码*，另一位用户报告称*现在的比例大约是用户 Prompt Credits 的 10 倍。*
- **Windsurf 在独立的项目文件夹中实现代码**：一位用户询问如何在具有独立前端和后端文件夹的团队项目中实施 **Cascade/Windsurf**，并咨询了实施计划和团队协作的最佳实践。
   - 一位成员建议使用一个*主 Tasklist 文件*来列出整个计划的要点，然后让 Agent 消化该任务列表并构思系统如何协同工作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/AnthropicAI/status/1764653830468428150">来自 Anthropic (@AnthropicAI) 的推文</a>：今天，我们发布了 Claude 3，我们的下一代 AI 模型。这三个最先进的模型——Claude 3 Opus、Claude 3 Sonnet 和 Claude 3 Haiku——在推理等方面树立了新的行业标杆...</li><li><a href="https://artificialanalysis.ai/">AI 模型与 API 提供商分析 | Artificial Analysis</a>：AI 模型和 API 托管提供商的比较与分析。涵盖质量、价格、输出速度和延迟等关键性能指标的独立基准测试。</li><li><a href="https://github.com/Exafunction/codeium/issues/129">Api Server Wire Error (User is Disabled by Team) · Issue #129 · Exafunction/codeium</a>：连接错误 api server wire error: user is disabled by team。我尝试了重启、重新登录，但都无法解决。
</li>
</ul>

</div>

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1346336998926843957)** (153 条消息🔥🔥): 

> `Claude API 理解，Groq 定价与性能，Aider 与 git 实践，Sonnet 3.7 调优，Aider 与 Zed` 


- **Claude 在处理前端和后端 API 通信时遇到困难**：一位成员寻求帮助，想知道如何让 **Claude** 理解前端和后端 API，因为它编写了两个完全独立的 API，且彼此无法通信。
   - 另一位成员建议强制它审查 API 并进行整合，或者重新生成代码库。
- **Groq Specdec 速度更快但价格更高**：成员们讨论了 **Groq** 的 **specdec** (speculative decoding)，指出它比 versatile 模型贵约 **20%**，但速度提升了 **5 倍以上**。
   - 有人认为由于输入/输出比率的原因，**Gemini** 更适合做总结，而另一些人则建议使用像 *llama-3.2-3b-instruct* 这样的小型模型进行总结。
- **Aider 可以帮助练习 git**：一位成员建议使用 **Aider** 练习 **Git** 技能，让它创建一系列带有不同问题和练习的提交链。
   - 另一位成员分享了 [Oh My Git!](//blinry.itch.io/oh-my-git) 的链接，这是一个关于学习 Git 的开源游戏，可以实时可视化 Git 仓库的内部结构。
- **通过测试和 Guardrails 驯服 Sonnet 3.7**：成员们讨论了使用 **Sonnet 3.7** 的挑战，指出它经常做出过度修改，需要仔细的 prompting 和 guardrails。
   - 社区建议调优 AI 的最佳方法是在错误中学习，并通过文档添加规范。
- **Aider 在 Zed IDE 中运行**：一位成员提到在 **Zed** 编辑器中使用 **Aider**，发现它轻便且快速。
   - 还有关于是否可以开启 **Gemini** 的长上下文窗口 (long context windows) 和缓存 (caching) 的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ohmygit.org">Oh My Git!</a>: 一个关于学习 Git 的开源游戏</li><li><a href="https://aider.chat/docs/llms/gemini.html">Gemini</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://x.com/CohereForAI/status/1896923657470886234">来自 Cohere For AI (@CohereForAI) 的推文</a>: 介绍 ✨ Aya Vision ✨ - 一个通过语言和视觉连接世界的开源权重模型。Aya Vision 为我们最先进的多语言 8B 和 32B 模型增加了突破性的多模态能力...</li><li><a href="https://github.com/dnakov/claude-code/blob/main/src/constants/prompts.ts">claude-code/src/constants/prompts.ts (位于 dnakov/claude-code 的 main 分支)</a>: 来自 source maps 的 claude-code 完整原始源代码 - dnakov/claude-code
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1346333608217935883)** (69 messages🔥🔥): 

> `Less smart models can be used by Architect, Aider update API key, Custom model ctxlen cost, Ericsson CodeChecker added in project, Set global settings to Aider` 


- **基础 Architect 修改可以使用较弱的模型吗？**: 有建议认为，在 **aider** 工作流中，使用一个稍逊色的模型可能就足以实现由 **Architect** 规划的变更。
   - 一位用户提到为此目的使用了 **o3-mini**。
- **自定义 LLM 提供商的 API 密钥困扰**: 一位使用自定义 **LLM** 提供商的用户遇到了 API 密钥有效期短（每小时过期）的问题，寻求在不重启 **aider** 的情况下更新 API 密钥的方法，因为重启会中断聊天历史。
   - 建议包括使用 *restore history* 参数、实现 **nginx** 反向代理进行密钥轮换，或尝试使用虚拟密钥。
- **自定义模型配置深度解析**: 一位用户询问如何为 **aider** 模型列表中未列出的自定义模型（groq/deepseek-r1-distill-llama-70b-specdec）在外部指定 **ctxlen/cost**。
   - 有人指出 [documentation](https://github.com/domelqq/aider/tree/workon_feature) 中的一个 `.json` 文件可以处理此问题，使用户能够配置原生不支持的模型。
- **奇怪的 Ericsson 代码插入谜团**: 一位用户报告称，在使用 **aider v0.75.1** 配合 **claude-3-7-sonnet** 时，其项目中意外插入了来自 **Ericsson/codechecker** 的大块代码。
   - 用户怀疑这是由 **LLM** 造成的。
- **Windows 上的 Aider 全局设置探索**: 一位用户寻求在 Windows 上配置 **aider** 全局设置的指导，旨在避免在多个项目文件夹中重复创建 `.aider.conf.yml` 文件。
   - 解决方案是将 `.aider.conf.yml` 文件直接放置在 `C:/User/me` 目录下，并确保它不在 `.aider` 子文件夹内。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/bsilva96/status/1897042309314961865">Tweet from Benjamín Silva (@bsilva96)</a>: In LLMs context is the most important thing. When working in large code repositories Cursor  is unable to provide all the files of the repo to respond to your basic&#34;Explain me what does this XX fe...</li><li><a href="https://github.com/domelqq/aider/tree/workon_feature">GitHub - domelqq/aider at workon_feature</a>: aider is AI pair programming in your terminal. Contribute to domelqq/aider development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/releases">Releases · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1346394835405508650)** (2 messages): 

> `Karpathy inspires voice AI, Claude 3.7 Sonnet, Claude Code Agent` 


- **Karpathy 启发 Aider 开发者**: 一位成员受到 **Karpathy** 的启发，开始更多地通过语音与 **AI** 交互，并指出文本通信速度太慢。
   - 他们链接了一个 [YouTube video](https://www.youtube.com/watch?v=jCVO57fZIfM)，讨论了 **Claude 3.7 Sonnet** 和 **Claude Code** 作为潜在的游戏规则改变者，其中 **Claude Code** 被描述为最伟大的 **AI Agent**。
- **Claude 3.7 Sonnet 和 Claude Code 作为游戏规则改变者**: 链接的 [YouTube video](https://www.youtube.com/watch?v=jCVO57fZIfM) 表明 **Claude 3.7 Sonnet** 和 **Claude Code** 是 **AI** 领域的潜在变革者。
   - 视频描述强调 **Claude Code** 可能是*最伟大的 AI Agent*。



**Link mentioned**: <a href="https://www.youtube.com/watch?v=jCVO57fZIfM">GPT-4.5 FLOP? Claude 3.7 Sonnet STARTER PACK. What is Claude Code REALLY?</a>: 🔥 GPT-4.5 maybe the biggest FLOP. MEANWHILE Claude 3.7 Sonnet and Claude Code just CHANGED THE GAME! 🤯Anthropic&#39;s Claude Code maybe the greatest AI Agent T...

  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1346346835157192734)** (67 条消息🔥🔥): 

> `LM Studio CLI, 报告 LM Studio 漏洞, LM Studio 与 Home Assistant, GPT-4o 加载错误, 模型设置` 


- **LM Studio CLI 命令发布**：LM Studio 发布并记录了 LM Studio CLI (`lms`) 命令，[开发者可以查看在线文档](https://lmstudio.ai/docs/cli) 以进行本地 LLM 工作流的脚本编写和自动化。该项目采用 **MIT License**，代码托管在 [https://github.com/lmstudio-ai/lms](https://github.com/lmstudio-ai/lms)。
   - CLI 随 LM Studio 一起发布，可以在 LM Studio 工作目录下的 `/bin` 目录中找到，且必须至少运行一次。
- **安全报告 LM Studio 漏洞**：一位成员报告了一个潜在漏洞，建议将详细信息（不含 zip 附件）以 **纯文本** 形式发送至 [bugs@lmstudio.ai](mailto:bugs@lmstudio.ai)，并附带概念验证 (PoC)、视频和截图。
   - 重点强调了出于安全考虑应避免使用 zip 附件，建议直接在邮件正文中发送所有信息。
- **LM Studio 能控制 Home Assistant 吗？**：一位用户询问如何将 LM Studio 与 Home Assistant 配合使用，另一位用户指向了 [一个社区讨论帖](https://community.home-assistant.io/t/local-llm-for-dummies/769407/10) 作为潜在解决方案。
   - 寻求帮助的用户澄清说，虽然他们可以集成 LM Studio，但无法通过它控制家庭设备。
- **GPT-4o 导致模型加载错误**：一位用户在加载 *lmstudio-community/Phi-4-mini-instruct-GGUF Q4* 模型时遇到了由于 *unknown pre-tokenizer type: 'gpt-4o'* 导致的错误，并确认其模型是在更新后下载的。
   - 用户通过确保将其 LM 运行时 (runtimes) 更新到 Beta 版本（通过 ctrl+shift+r）解决了此问题。
- **LM Studio Server 即将支持 PDF 上传**：一位用户询问是否可以使用 Python SDK 直接将 PDF 文档上传到 LM Studio 服务器。
   - 一位开发者回应称该功能即将推出，将利用 *pdf2json*（如 [LM Studio 致谢页面](https://lmstudio.ai/acknowledgements.html) 中所述）进行内容提取。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://community.home-assistant.io/t/local-llm-for-dummies/769407/10">Local LLM for dummies</a>: 我也不太清楚，但你可以尝试一下。这不会损坏你的机器，而且很容易再次卸载。</li><li><a href="https://lmstudio.ai/docs/cli">lms — LM Studio's CLI | LM Studio Docs</a>: 开始使用 lms 命令行工具。</li><li><a href="https://youtu.be/CUy1ZOVk7QE?si=EG7nShA-NX7v1Q2h">Goku vs Jiren English dub with Original Ultimate Battle V2</a>: https://youtu.be/SRWsHICBt3Qhttps://youtu.be/Cl05ShiYAAs</li><li><a href="https://tenor.com/view/howaboutthat-avengers-ironman-tonystark-stark-gif-3525361">Power Is At 400% GIF - Howaboutthat Avengers Ironman - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/blade-runner-tears-in-rain-roy-batty-gif-3478115">Blade Runner GIF - Blade Runner Tears In Rain Roy Batty - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://youtube.com/shorts/IV32vPYPG7U?si=S3HbAE97wejtAdjO">Nvidia launching RTX 5090 be like</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1346410464514543669)** (50 条消息🔥): 

> `拥有 48GB VRAM 的 4090，纯净数据集，LM Studio 中的 NPU 支持，带 LM Studio 的 Intel ARC iGPU，模型卸载到 CPU/RAM` 


- **配备 48GB VRAM 的改装版 4090 现身**：一名用户询问了关于改装 **48GB** VRAM 的 **4090** 的情况，质疑其性能是否与标准的 **24GB 4090** 相同，并附带了一张 [图片](https://cdn.discordapp.com/attachments/1153759714082033735/1346425229538230342/image0.png?ex=67c8cc76&is=67c77af6&hm=719728fa64976a052bef1165873cfff11f6eb1bb595a4a5d046728c3f7e31fd3)。
- **对完美数据集的追求仍在继续**：一名用户表示惊讶，考虑到目前模型进行迭代过滤的能力，*纯净数据集（pristine datasets）* 竟然还无法轻易获得。
   - 另一名用户反驳称，定义“干净”是主观的，并强调了在 AI 训练数据的法律和伦理标准上达成一致的挑战。
- **LM Studio 的 NPU 支持状态**：一名用户询问如何在配备 **ARC GPU** 和 **NPU** 的笔记本电脑上运行 LM Studio。
   - 另一名用户回复称 **NPU 目前尚不支持**，并建议他们可以尝试自己 *开始编写代码* 来实现它。
- **Intel Arc iGPU 在 LM Studio 中检测不到 VRAM**：一名用户报告称，LM Studio 检测到了他们的 **Intel Arc iGPU**，但显示 **VRAM 为零**。
   - 尽管如此，他们指出该 iGPU 的理论性能为 **48 TOPS**，可与 **RTX 4080** 媲美，这使得兼容性可能非常有价值。
- **通过 CPU/RAM 卸载绕过 VRAM 限制**：一名拥有 **12GB VRAM** 的用户询问是否可以像 Reddit 上建议的那样，通过将层（layers）卸载到 **CPU** 或 **RAM** 来运行更大的模型。
   - 另一名用户解释说，在通过附带 [图片](https://cdn.discordapp.com/attachments/1153759714082033735/1346552868752195644/image.png?ex=67c89a95&is=67c74915&hm=9f7b97ef96c40d1c40130da10685a2be1e5be7c47e7b4103a0dc349c58b11562&) 中显示的 UI 选择模型时，**LM Studio** 允许手动调整模型参数，包括上下文长度和 **GPU 卸载层数（GPU offloading layers）**。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1346362884548067348)** (39 条消息🔥): 

> `Anthropic 融资 35 亿美元，Hugging Face LLM 作为本地微服务，Qwen2.5-Coder 模型，孟加拉语文本罗马化，Whisper 日语模型` 


- **Anthropic 获得天文数字估值**：[Anthropic](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation) 在 **Lightspeed Venture Partners** 的领投下筹集了 **35 亿美元**，投后估值达到 **615 亿美元**。
- **Hugging Face LLM 调用本地微服务**：成员们讨论了将基于 **Hugging Face 的 LLM** 作为本地微服务运行并对其进行 API 调用的可能性。
   - 一位成员建议使用 **TGI** ([Text Generation Inference](https://huggingface.co/docs/text-generation-inference/en/index))，这是一个用于部署和提供大语言模型 (LLMs) 服务的工具包。
- **Qwen2.5-Coder 在编程任务中表现出色**：**Qwen2.5-Coder** 系列（[Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) 和 [Qwen2.5-Coder-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)）在**代码生成**、**代码推理**和**代码修复**方面有了显著提升。
- **Flan-T5 爱好者微调功能**：一位成员正在致力于微调一个用于**孟加拉语文本罗马化**的模型，并就使用哪种基础模型征求建议。
   - 另一位成员提到 **FLAN** 基本上是经过指令微调（instruct-tuned）的 **T5**。
- **用户在使用 Turbo Whisper 时遇到翻译难题**：一位成员指出 **Whisper** 的翻译功能仅支持从源语言翻译为英语，并对 **whisper-large-v3-turbo** 无法将英语翻译为日语表示遗憾。
   - 另一位成员建议尝试 **Whisper 日语模型**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/AnthropicAI/status/1896606683876753470?t=rG-iRTyhiMnSiTe-kJeOXQ&s=19">Anthropic (@AnthropicAI) 的推文</a>: Anthropic 在 Lightspeed Venture Partners 的领投下融资 35 亿美元，投后估值为 615 亿美元。这将推进我们对 AI 系统的开发，加深我们对其工作原理的理解...</li><li><a href="https://spinningup.openai.com/en/latest/">欢迎来到深度强化学习 Spinning Up！&mdash; Spinning Up 文档</a>: 未找到描述</li><li><a href="https://readthedocs.org).>>>">无标题</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/text-generation-inference/en/index">Text Generation Inference</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct">Qwen/Qwen2.5-Coder-7B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF">Qwen/Qwen2.5-Coder-3B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/fluently/Fluently-XL-v3-inpainting">fluently/Fluently-XL-v3-inpainting · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

ericmaubr: 今天我看了我的第一个关于 RLHF 的视频。https://www.youtube.com/watch?v=2MBJOuVq380
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1346339348949237864)** (10 条消息🔥): 

> `InternVL 2.5 的 AWQ 转换，乌克兰语文本转语音（TTS），LLM 中的偏见，AGI 的多 Agent 方法，可用的 Embedder 集合` 


- **InternVL 2.5 获得 AWQ 转换**：一位成员分享了他们的 [InternVL 2.5 的 AWQ 转换版本](https://huggingface.co/rootonchair/InternVL2_5-4B-AWQ)，并指出其性能损耗极小，且与 *transformers* 库兼容。
   - 有人指出，虽然 **Bitsandbytes** 不需要校准且转换速度更快，但 **AWQ** 可能会提供更好的性能。
- **乌克兰语文本转语音（TTS）模型发布**：一位成员在 GitHub 和 PyPI 上发布了 [乌克兰语文本转语音模型](https://github.com/egorsmkv/tts_uk) 的稳定版本，具有 **三种声音**（2 女，1 男），并支持对语音参数的细粒度控制。
   - 该模型使用 **RAD-TTS++** 进行声学建模，使用 **Vocos** 进行快速声码器处理，支持 **44.1 kHz** 的采样率，并已在 Linux 和 Windows/WSL 上通过测试。
- **LLM 小型偏见评估**：一位成员分享了一篇 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1j1nen4/llms_like_gpt4o_outputs/)，讨论了关于 LLM 偏见的小型评估，其中 LLM 被要求根据预估能力和已知信息互相评分。
   - 该评估考虑了评判方 LLM 对另一家模型公司的了解程度。
- **多 Agent AGI 研究概览**：一位成员分享了一段 [YouTube 视频](https://www.youtube.com/watch?v=cryI1-3Om9c)，讨论了 AGI 的多 Agent 方法研究，涵盖了多 Agent 扩展定律（Scaling Laws）以及模型多样性的优势。
   - 视频强调了系统中多样化的模型、架构和知识如何提升性能。
- **可用 Embedder 集合汇编**：一位成员分享了一个 [可用 Embedder 集合](https://huggingface.co/kalle07/embedder_collection)，并使用 ALLM (AnythingLLM) 进行了测试，强调 *nomic-embed-text*、*mxbai-embed-large*、*mug-b-1.6* 和 *Ger-RAG-BGE-M3*（德语）表现良好。
   - 该成员提供了使用这些 Embedder 的技巧，建议通过调整上下文长度和片段设置以获得最佳性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/kalle07/embedder_collection">kalle07/embedder_collection · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=cryI1-3Om9c">What is the Multi-Agent Approach to AGI?</a>：Naptha 是一个用于大规模开发和运行具有异构模型、架构和数据的多 Agent 系统的框架和基础设施。Agent...</li><li><a href="https://huggingface.co/rootonchair/InternVL2_5-4B-AWQ">rootonchair/InternVL2_5-4B-AWQ · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1j1nen4/llms_like_gpt4o_outputs/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://reddit.com/en-us/policies/cookies)">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://reddit.com/en-us/policies/privacy-policy).>>>">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/egorsmkv/tts_uk">GitHub - egorsmkv/tts_uk: High-fidelity speech synthesis for Ukrainian using modern neural networks.</a>：使用现代神经网络实现的乌克兰语高保真语音合成。- egorsmkv/tts_uk</li><li><a href="https://huggingface.co/spaces/Yehor/radtts-uk-vocos-demo">RAD-TTS++ Ukrainian (Vocos) - a Hugging Face Space by Yehor</a>：未找到描述</li><li><a href="https://huggingface.co/chat/)">HuggingChat</a>：让每个人都能使用社区最好的 AI 聊天模型。</li><li><a href="https://huggingface.co/chat/huggingchat/logo.svg)">HuggingChat</a>：让每个人都能使用社区最好的 AI 聊天模型。</li><li><a href="https://apps.apple.com/us/app/simply-scan/id6742599424">‎Simply Scan</a>：Simple Scanner 将您的 iPhone 变成功能强大的文档数字化工具。只需轻轻一按，即可利用我们的智能边缘检测技术自动捕捉任何文档的清晰扫描件...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1346411246429278228)** (4 条消息): 

> `ViT CLS token, DETR from scratch, GAP vs CLS, DETR inference` 


- **ViT CLS Token 起到全局作用**：在 **ViT** 中，**CLS token** 通过 self-attention 与 patch embeddings 一起更新，聚合关于图像的全局信息用于分类。
   - 因为它是一个可学习参数，其权重在反向传播过程中相对于 **multi-head attention (MHA) blocks** 中的其余参数进行优化。
- **为什么 ViT 相比 GAP 更倾向于使用 CLS token**：一位成员解释了 **CLS token** 如何利用 self-attention 来关注最相关的图像 patches，而 **GAP** 则对所有 patches 进行等权重处理。
   - 另一位成员表示赞同，但指出*如果模型使用正确的超参数进行训练，这两种方法之间的性能差异很小甚至没有差异*。
- **从零实现的 DETR 在 Batch Size 为 1 时遇到困难**：一位成员从零构建了一个 **DETR**，并在一个小数据集（约 1000 张图像）上进行了微调，使用了来自 torchvision 的预训练 **ResNet50** backbone。
   - 该模型取得了不错的结果，但在推理阶段使用 **batch size 1** 时除外；该成员分享了一个 [GitHub repo](https://github.com/dimiz51/DetectionTransformer-DETR-PyTorch) 展示了具体实现。



**提到的链接**：<a href="https://github.com/dimiz51/DetectionTransformer-DETR-PyTorch">GitHub - dimiz51/DetectionTransformer-DETR-PyTorch: This project is an implementation of the Detection Transformer (DETR) for state-of-the-art object detection using the well-known Transformer architecture. Using this project you can easily fine-tune and test DETR on your own dataset following the included notebook guide.</a>：该项目是使用著名的 Transformer 架构实现最先进目标检测的 Detection Transformer (DETR) 的实现。通过该项目，你可以按照附带的 notebook 指南轻松地在自己的数据集上微调和测试 DETR。

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1346389741746126889)** (5 条消息): 

> `Text Extraction with SpaCy, Hosting Phi-4 with Ollama and FastAPI, Accelerate CLI memory estimation` 


- **SpaCy 从文本中提取元素**：一位成员正在开发一个项目，使用 [SpaCy 从文本中提取元素](https://spacy.io/usage/spacy-101)，结合 REGEX 捕获数值，并利用 dependency parsing 分析句子结构。
   - 他们对使用 NER 提取参数表示不确定，因为参数长度不一，且不同文档和贡献者的写作风格存在差异。
- **使用 Ollama 和 FastAPI 托管 Phi-4**：一位成员询问了使用 Ollama 和 FastAPI 进行推理时 [托管 Phi-4 的 GPU 需求](https://www.microsoft.com/en-us/research/blog/phi-4-a-breakthrough-in-language-model-efficiency/)。
   - 他们正在寻求关于如何计算其配置所需 GPU 资源的指导。
- **Accelerate CLI 估算内存**：一位成员分享了关于使用 [Accelerate CLI](https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator) 估算模型内存需求的 Hugging Face 文档链接。
   - 该 CLI 将模型加载到 `meta` 设备的内存中，并支持搜索可在 `timm` 和 `transformers` 中使用的模型。



**提到的链接**：<a href="https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator">Model memory estimator</a>：未找到描述内容。

  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1346346458252709978)** (50 条消息🔥): 

> `Gradio App 内部机制, LlamaIndex Notebook 错误, SmolAgents vs SmolTools, 需要支付问题, Deep Reinforcement Learning 资源` 


- **深入探究 Gradio App**：一位用户询问 `What are LLMs?` 页面上的 **Gradio app** 内部是否使用了 **Agent** 来生成图表和结果，并询问是否可以学习构建类似的 App 以及获取其源代码。
   - 用户对团队提供的高质量免费教程表示感谢。
- **LlamaIndex Notebook 链接失效**：多名用户报告在访问 Unit 2.2 的 **LlamaIndex notebook** ([https://huggingface.co/agents-course/notebooks/blob/main/unit2/llama-index/components.ipynb](https://huggingface.co/agents-course/notebooks/blob/main/unit2/llama-index/components.ipynb)) 时遇到 **404 错误**。
   - 一名工作人员随后确认该问题已得到解决。
- **SmolAgents 框架与 SmolTools 库的区别**：用户寻求关于 **SmolAgents** 和 **SmolTools** 之间区别的澄清，并指出课程单元中两者都被称为库。
   - 澄清结果为：*SmolAgents 基本上是一个用于创建轻量级 Agent 的框架*，而 *SmolTools 包含了可以在 SmolAgents 中使用的工具函数和预构建工具*。
- **Inference 额度不足**：几位用户报告由于超过了 **Inference Providers** 的每月包含额度，遇到了 *需要支付（payment required）* 的问题。
   - 一位用户建议使用备选模型 ([https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud](https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud)) 作为潜在的变通方案。
- **Deep Reinforcement Learning 讨论**：一位用户分享了 **Deep Reinforcement Learning (DRL)** 的资源，包括 [Hugging Face Learn DRL 课程](https://huggingface.co/learn/deep-rl-course/unit0/introduction)、书籍 **Reinforcement Learning: An Introduction** 的链接 ([http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html))，以及 YouTube 上的 **DeepMind x UCL Deep Learning Lecture Series 2021** ([https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&feature=shared](https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&feature=shared))。
   - 其他用户表示愿意提供更多关于这些资源的见解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud'))">未找到标题</a>：未找到描述</li><li><a href="https://mysterybox-nft18-eta.vercel.app/">点击此处 - OPENSEA PRO NFT</a>：&#128994; 空投现已上线 &#128994; &#127881; 价格：免费 &#127881; 供应量：150 个神秘盒子 &#127881; 奖励：3000 美元至 250,000 美元之间，试试你的运气！ &#128640;    </li><li><a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/llama-index/components.ipynb">unit2/llama-index/components.ipynb · agents-course/notebooks at main</a>：未找到描述</li><li><a href="https://huggingface.co/learn/deep-rl-course/unit0/introduction">欢迎来到 🤗 Deep Reinforcement Learning 课程 - Hugging Face Deep RL Course</a>：未找到描述</li><li><a href="http://incompleteideas.net/book/the-book-2nd.html">Sutton &amp; Barto 书籍：Reinforcement Learning: An Introduction</a>：未找到描述</li><li><a href="https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&feature=shared">DeepMind x UCL | Deep Learning Lecture Series 2021</a>：深度学习讲座系列是 DeepMind 与 UCL 人工智能中心之间的合作项目。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1346590262897545316)** (1 条消息): 

> `BYOK 错误, API Key 问题` 


- **BYOK 请求遇到错误**：在过去的 30 分钟内，大多数 **BYOK 请求**（针对在设置中附加了自己 API Key 的用户）都显示错误。
   - 相关更改已被撤销，团队正在添加额外的防护措施以防止此类情况再次发生。
- **缓解 API Key 错误**：最近的一个问题导致在设置中使用自己 **API Key** (BYOK) 的用户出现错误。
   - 团队已撤销有问题的更改，并正在实施额外的防护措施，以确保用户提供的 API Key 的稳定性。


  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1346385172995510302)** (2 条消息): 

> `价格预估，总价组合` 


- **估算总价组合**：一位成员分享了一种在输入和输出量巨大的情况下显示总价的方法，从而实现快速简便的价格预估。
- **显示价格**：目标是即使在输入和输出变量很大的情况下也能显示总价。
   - 该方法旨在提供一个相对快速且简便的价格估算过程。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1346381520155312131)** (105 条消息🔥🔥): 

> `OpenRouter 提供商路由，最强书签处理 AI，Flash 2.0 对比 GPT-4o-mini，Inception AI 扩散模型，Anthropic 502 过载错误` 


- **OpenRouter 提供商路由配置**：一位用户需要通过特定提供商路由请求，并被指导根据 [OpenRouter 文档](https://openrouter.ai/docs/features/provider-routing#json-schema-for-provider-preferences) 修改 API 请求体，添加 `provider` 对象，在 `order` 数组中指定所需的提供商，并将 `allow_fallbacks` 设置为 false。
   - 强调了提供商名称必须与 OpenRouter 模型页面上列出的名称**完全匹配**（例如 `Nebius`），并且 JSON 中的提供商名称需要加引号。
- **Groq-3 和 GPT-4.5 在书签处理方面领先**：对于处理大量书签，推荐使用 **Groq-3**（无 API）和 **GPT-4.5**（昂贵），**DeepSeek v3** 和 **Claude 3.7** 紧随其后，但另一位用户表示 **ChatGPT**（可能是通过 Web 界面的 GPT-4o）能够完成该任务。
   - 该用户感到惊讶，因为 ChatGPT 在仅上传 **1 个文件**后就返回了“您已达到数据分析限制”。
- **Flash 2.0 完胜 GPT-4o-mini**：**Flash 2.0** 被推荐为 **GPT-4o-mini** 的更强大且略微便宜的替代方案。
   - 一位用户表示它“完胜 4o mini，聪明得多”。
- **用户请求在 OpenRouter 上提供 Inception AI 的扩散模型**：在 [TechCrunch 报道了其 DLM（基于扩散的大语言模型）](https://techcrunch.com/2025/02/26/inception-emerges-from-stealth-with-a-new-type-of-ai-model/)后，一位用户请求通过 OpenRouter 访问 **Inception AI** 的扩散模型。
   - OpenRouter 正在与 **Inception AI** 联系，并期待尽快将其上线。
- **Anthropic 过载触发 502 错误**：用户报告收到“过载”错误，这些错误被确认为来自 Anthropic 的 **502 状态码**，表明存在容量问题。
   - 这些 **502 错误**即使在状态页面没有宣布故障的情况下也可能发生，需要用户重试请求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/features/provisioning-api-keys">Provisioning API Keys - OpenRouter API 密钥的程序化控制</a>：通过专用管理端点以程序化方式管理 OpenRouter API 密钥。创建、读取、更新和删除 API 密钥，以实现自动化的密钥分发和控制。</li><li><a href="https://openrouter.ai/rankings/finance">LLM 排名：金融 | OpenRouter</a>：按金融提示词使用情况排名和分析的语言模型。</li><li><a href="https://openrouter.ai/docs/features/provider-routing#json-schema-for-provider-preferences>">提供商路由 - 智能多提供商请求管理</a>：智能地在多个提供商之间路由 AI 模型请求。了解如何利用 OpenRouter 的提供商路由优化成本、性能和可靠性。</li><li><a href="https://techcrunch.com/2025/02/26/inception-emerges-from-stealth-with-a-new-type-of-ai-model/">Inception 带着新型 AI 模型脱离隐身模式 | TechCrunch</a>：初创公司 Inception 声称开发了一种基于扩散架构的新型 AI 模型。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1346421318689624125)** (43 messages🔥): 

> `Mojo vs C++ vs Rust 的相似之处, Mojo 的设计视角缺陷, Mojo 中的 Python 超集, Mojo 中的并发与 Sum Types, Python API 与 Rust API 的一致性对比` 


- **Mojo：带有缺失 C++ 特性的 Rust？**：一位成员将 **Mojo** 描述为 *像 Rust，但加入了那些本该从 C++ 继承过来的特性*。
   - 讨论涉及到熟悉 **Rust 的内存管理模型** 如何有助于理解 Mojo。
- **Mojo 缺乏语言层级的一致性**：关于 **Mojo** 提到的最大缺点之一是命名规范在语言层级缺乏一致性。
   - 有人指出，虽然 **Rust** 在整个语言中保持了一致性，但 **Mojo** 混合了类 Python、类 C 以及其自身的 API，部分原因是其最初作为 **Python 超集** 的设计定位。
- **Python 超集叙事正在拖累 Mojo**：成员们讨论了 Mojo 最初作为 **Python 超集** 的概念，一些人认为这种 *叙事* 带来了不必要的负担，例如从 `libc` 复制的命名。
   - 官方澄清其目标不是与 CPython 实现 *Bug 级兼容*，而是为了让通过“查找与替换”来迁移基础代码变得合理可行。
- **并发和 Sum Types 是高优先级特性**：成员们提到，完善的 **并发（Concurrency）** 和 **Sum Types** 是他们非常渴望的功能。
   - 一位成员指出了一个与 *Mojo 结构化异步（Structured Async）* 相关的 [GitHub Pull Request](https://github.com/modular/max/pull/3945) 以及[另一个关于 Effect Handlers 的 PR](https://github.com/modular/max/pull/3946)，表明这些领域正在积极开发中。
- **Python 标准库 API 一致性引发争论**：一位成员希望 Mojo 不要放弃 **Python API**，理由是熟悉 Python 及其库。
   - 另一位成员则认为 **Rust** 拥有比 Python 更可预测的 API，并希望 Mojo 不要被 **Python 标准库 API** 困住，因为后者混合了 C 的 flatcase、Java 的 camelCase 和 Python 的 snake_case。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modular/max/pull/3945">[proposal] Structured Async for Mojo by owenhilyard · Pull Request #3945 · modular/max</a>：提议为 Mojo 添加结构化异步，遵循 Rust 的异步传统，因为 Mojo 有能力修复 Rust 异步中的许多问题，其中一些是生态系统影响...</li><li><a href="https://github.com/modular/max/pull/3946">[proposal] Provided Effect Handlers by owenhilyard · Pull Request #3946 · modular/max</a>：该提案包含了一种 Effect 系统替代方案，我认为它更适合在系统语言中抽象异步、raises 和类似的函数颜色（function colors），因为在系统语言中上下文可能并不...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1346416607399903272)** (15 messages🔥): 

> `Mojo 中命令行参数转换为 int, Mojo 的 is 运算符标识检查, Tensor 加法操作移除, Mojo 代码构建错误：缺失 libtinfo` 


- **Mojo 新手寻求 CLI 参数转换的明确方法**：一位成员寻求在 Mojo 中将命令行参数转换为整数的指导，尝试了多种方法均未成功。
   - 另一位成员指出了类型问题，建议使用 `List[StaticString]` 和 `atol` 的解决方案，同时提到了 `StringSlice` 来源的一个段错误（segfault）问题。
- **`is` 运算符标识危机已解决**：一位成员询问 Mojo `assert_is` 函数中 *identity* 的含义，询问它是否检查相同类型，另一位成员澄清它与内存位置有关。
   - 回答者澄清 `is` 检查两个对象是否位于相同的内存位置，类似于指针相等，但提醒在 register-passable 类型上会有复杂性，并链接到了 [Identifiable 文档](https://docs.modular.com/mojo/stdlib/builtin/identifiable/Identifiable/)。
- **Tensor 加法操作被移除**：一位成员报告说 `Tensor[float64]` 在 Mojo nightly 版本中不再实现 `__add__` 方法。
   - 另一位成员发现了一条提交信息，解释移除是为了逐步淘汰 `Tensor` 以转向其他词汇类型（vocabulary types），并建议使用 `LayoutTensor` 进行更高效的元素级操作，详见[此提交信息](https://github.com/modularml/mojo/commit/SOME_COMMIT_HASH)。
- **Mojo 构建困扰：缺失 libtinfo**：一位成员在构建 Mojo 时因缺少 `-ltinfo` 库遇到错误。
   - 另一位成员确认问题是缺少 `libtinfo`，并建议使用 `sudo apt-get install libtinfo-dev` 进行修复。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1346346901162688553)** (45 messages🔥): 

> `AI Timelines, Differential Transformers, Numerical Stability in Softmax, Bilevel Optimization, Grokking Phenomenon` 


- ****AI Timelines**：机器何时会思考？**：正如[这篇文章](https://ourworldindata.org/ai-timelines)所讨论的，许多 AI 专家预测人类水平的人工智能（human-level AI）可能会在未来几十年内实现。
   - 人类水平 AI 被定义为能够执行人类可以完成的任何任务，并有能力选择让机器实现这些任务的行动。
- ****Transformer Architecture** 迎来了微分版本？**：一份时事通讯强调了最近的 AI 研究，包括 [Differential Transformers](https://mail.bycloud.ai/)、混沌边缘的智能，以及为什么 LLM 可能并没有真正的推理能力。
   - 文中还提到 **Byte Latent Transformers** 是未来无需 Tokenization 的 LLM 的潜在方向。
- ****Softmax 不稳定性**与下溢（Underflow）在 Grokking 中的作用**：围绕一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/damienbenveniste_the-softmax-transform-might-be-one-of-the-activity-7301720641559269377-gDdQ)的讨论表明，虽然 Softmax 解决了上溢（overflow）问题，但它可能会加剧梯度下降过程中的下溢问题，可能导致模型陷入停滞。
   - 最近的一些论文认为，下溢可能有助于 **Grokking** 现象，作为一种隐式正则化器来防止过拟合。
- ****双层优化（Bilevel Optimization）**用于 Sparsemax 泛化？**：一位成员建议，**双层优化**可能会泛化 **Sparsemax** 和 **Stablemax**，从而可能通过“领导者/追随者”（leader/followers）的视角来看待整个 ANN。
   - 他们编写了一个 [BilevelMax 类](https://www.dataia.eu/sites/default/files/1%20Marco-Pedersoli%20ILLS.pdf)来动态平衡稀疏性（sparsity）和密度（density），在 **Sparsemax** 和 **Softmax** 之间平滑过渡。
- ****Outlines**：无需微调即可结构化输出？**：成员们讨论了如何在不进行微调（Fine-tuning）的情况下，强制 LLM 输出结构化内容（例如 JSON）。
   - 一位成员建议使用 [Outlines](https://github.com/dottxt-ai/outlines)，这是一个修改采样逻辑以确保正确输出结构的库。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ourworldindata.org/ai-timelines">AI timelines: What do experts in artificial intelligence expect for the future?</a>: 许多人认为人类水平的 AI 确实有可能在未来几十年内开发出来，有些人甚至认为它会更早出现。</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria">Storage quotas and eviction criteria - Web APIs | MDN</a>: Web 开发人员可以使用多种技术在用户的浏览器中存储数据（即在用户用于查看网站的设备的本地磁盘上）。</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/API/StorageManager">StorageManager - Web APIs | MDN</a>: Storage API 的 StorageManager 接口提供了一个用于管理持久化权限和估算可用存储空间的接口。你可以使用 navi... 获取此接口的引用。</li><li><a href="https://mail.bycloud.ai/">The AI Timeline</a>: 关注最新的前沿 AI 研究。</li><li><a href="https://github.com/dottxt-ai/outlines">GitHub - dottxt-ai/outlines: Structured Text Generation</a>: 结构化文本生成。通过在 GitHub 上创建一个账户来为 dottxt-ai/outlines 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1346570280075526255)** (6 messages): 

> `DDPM, CLIP, MLST, GNN, Graph Attention Networks (GATs)` 


- **深入探讨 DDPM 和 CLIP 经典作**：一位成员提到 **DDPM** (Denoising Diffusion Probabilistic Models) 和 **CLIP** (Contrastive Language-Image Pre-training) 被认为是该领域的“经典”。
   - 该成员还提到收听了以 **pi 0** 为主角的 **MLST** (Machine Learning Street Talk) 播客节目。
- **图注意力网络 (GATs) 概述**：一位成员分享了 **Graph Attention Networks** (**GATs**) 的[概述](https://petar-v.com/GAT/)，这是一种在图结构数据上运行的神经网络架构，利用掩码自注意力层（masked self-attentional layers）来解决之前基于图卷积方法的缺点。
   - 该概述包括图结构输入的激励示例，如分子网络、交通网络、社交网络和大脑连接组网络，并包含指向[原始论文](https://arxiv.org/abs/1710.10903)的链接。



**提到的链接**: <a href="https://petar-v.com/GAT/">Graph Attention Networks</a>: 未找到描述

  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1346491088470278198)** (25 条消息🔥): 

> `NextGenAI 融资，Thinking Machines 聘请 PyTorch 工程师，Amazon Nova 推理模型，OpenAI 积分计划，CoreWeave 将收购 Weights & Biases` 


- **NextGenAI 资助下一代科学**：[OpenAI](https://openai.com/index/introducing-nextgenai/) 正在启动 **NextGenAI**，这是一项旨在为科学领域的 AI 提供资金的新计划。
- **PyTorch 工程师加入 Thinking Machines**：在 **PyTorch** 工作 4 年后，一名工程师加入 **Thinking Machines** 担任创始工程师，并称这是一个极具吸引力的机会，详见[这条推文](https://x.com/cHHillee/status/1896973303241400704)和[这篇博文](https://www.thonking.ai/p/why-pytorch-is-an-amazing-place-to)。
- **亚马逊打造 Nova，目标 AI 超新星**：据[这篇文章](https://ift.tt/GPTB0aV)报道，亚马逊正在其 **Nova** 品牌下开发一种具有成本效益的混合架构“推理”模型，旨在与 **OpenAI 的 o3-mini**、**DeepSeek 的 R1** 和 **Anthropic 的 Claude 3.7 Sonnet** 等模型竞争，可能最早于 6 月发布。
- **Sama 的订阅方案引发关注**：Sam Altman 提议为 OpenAI 制定一项新的付费计划，其中 **20 美元的 Plus 订阅将转换为积分**，可用于 deep research、o1、gpt-4.5、sora 等功能，详见[这条推文](https://fxtwitter.com/sama/status/1897036361506689206)。
- **CoreWeave 进军 Weights & Biases**：AI Hyperscaler™ **CoreWeave** 宣布已达成协议，将以 **17 亿美元**收购领先的 AI 开发者平台 **Weights & Biases**，详见[这份新闻稿](https://www.prnewswire.com/news-releases/coreweave-to-acquire-weights--biases---industry-leading-ai-developer-platform-for-building-and-deploying-ai-applications-302392342.html)和[这篇文章](https://www.theinformation.com/briefings/coreweave-to-buy-weights-biases-for-1-7-billion)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/sama/status/1897036361506689206">来自 Sam Altman (@sama) 的推文</a>：关于付费计划的一个想法：你 20 美元的 Plus 订阅将转换为积分，你可以跨功能使用，如 deep research, o1, gpt-4.5, sora 等。每个功能没有固定限制，由你选择想要的；...</li><li><a href="https://x.com/cHHillee/status/1896973303241400704">来自 Horace He (@cHHillee) 的推文</a>：生活更新：在 PyTorch 工作 4 年后，我加入了 @thinkymachines！这些年来，有几个人问我为什么这么不情愿离开。我想谈谈 1. 为什么我留下来...</li><li><a href="https://www.thonking.ai/p/why-pytorch-is-an-amazing-place-to">为什么 PyTorch 是一个绝佳的工作场所... 以及我为什么加入 Thinking Machines</a>：在这篇文章中，我将说服你加入 PyTorch 或 Thinking Machines！</li><li><a href="https://thinkingmachines.ai/).">Thinking Machines Lab</a>：未找到描述</li><li><a href="https://chatgpt.com)">未找到标题</a>：未找到描述</li><li><a href="https://x.com/btibor91/status/1897011162493149303">来自 Tibor Blaho (@btibor91) 的推文</a>：据报道，亚马逊正在其 Nova 品牌下开发一种具有成本效益的混合架构“推理”模型，以与 OpenAI 的 o3-mini、DeepSeek 的 R1 和 Anthropic 的 Claude 3.7 Son... 竞争。</li><li><a href="https://www.prnewswire.com/news-releases/coreweave-to-acquire-weights--biases---industry-leading-ai-developer-platform-for-building-and-deploying-ai-applications-302392342.html">CoreWeave 将收购 Weights &amp; Biases - 行业领先的用于构建和部署 AI 应用程序的 AI 开发者平台</a>：/PRNewswire/ -- AI Hyperscaler™ CoreWeave 今日宣布已达成协议，将收购领先的 AI 开发者平台 Weights &amp; Biases。该...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/)** (1 条消息): 

saturatedfat: 如果它在 arXiv 上，检查一下 tex 文件
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/)** (1 条消息): 

catboy_slimmer: 通常表现得像 Schmid 的人虽然存在，但相对没那么显眼

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1346388436642168874)** (11 条消息🔥): 

> `CogView4-6B, Anthropic vs OpenAI 模型, Aya Vision, LLM 摘要伦理, 会议转录摘要` 


- **CogView4-6B 在 HuggingFace 上亮相**: 来自 THUDM 的新模型 [CogView4-6B](https://huggingface.co/THUDM/CogView4-6B) 已发布，要求图像尺寸在 **512px** 到 **2048px** 之间且能被 **32** 整除，支持 **BF16** / **FP32** 精度。
   - 模型卡片指出，由于溢出问题会导致图像全黑，因此不支持 **FP16**。
- **Anthropic 的模型表现得像内向自闭，OpenAI 则是冷静哥？**: 一位用户分享了一则推文，开玩笑说 *Anthropic 让 3.7 变得像自闭症，但 OpenAI 把 4.5 变成了一个冷静的家伙* ([来源](https://x.com/benhylak/status/1896684180928598448))。
- **Aya Vision 发布，支持 23 种语言**: [Aya Vision 8B](https://huggingface.co/CohereForAI/aya-vision-8b) 是由 Cohere For AI 发布的一个 **80 亿** 参数模型，针对 **23 种语言** 的视觉语言任务进行了优化，采用 [CC-BY-NC 许可证](https://cohere.com/c4ai-cc-by-nc-license)。
- **摘要时的伦理困境**: 一位用户在 Twitter 上提出了一个对齐问题：当 LLM 为某人生成摘要时，如果原始文本（聊天记录）中包含关于该人的惊喜生日派对计划，LLM 是否应该隐瞒该信息 ([来源](https://x.com/TheXeophon/status/1896817938323341722))。
   - 大多数受访者同意 **LLM 应该隐瞒敏感信息**。
- **Elevenlabs Scribe 以人类质量转录会议**: 一位用户推荐使用 [elevenlabsio Scribe](https://x.com/dwarkesh_sp/status/1895966981020663833) 进行会议转录，指出它使用 Gemini 2.0 来生成具有*人类质量*的带音频转录稿。
   - 该用户还分享了一个 Prompt [链接](https://gist.github.com/dwarkeshsp/65c232298781c86b33f5e32065152f1e)，并建议将所有代码错误归咎于 Claude。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/THUDM/CogView4-6B">THUDM/CogView4-6B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/CohereForAI/aya-vision-8b">CohereForAI/aya-vision-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/benhylak/status/1896684180928598448">来自 ben (@benhylak) 的推文</a>: 感觉 Anthropic 让 3.7 变得像自闭症，但 OpenAI 把 4.5 变成了一个冷静的家伙。现在说哪种策略会获胜还为时过早。</li><li><a href="https://x.com/dwarkesh_sp/status/1895966981020663833">来自 Dwarkesh Patel (@dwarkesh_sp) 的推文</a>: 我很高兴地报告 @elevenlabsio Scribe 让我之前笨拙的解决方案变得不再必要。现在我只需要自动链接（到论文/书籍/关键词等）。引用 Dwarkesh Patel (@dwarkesh_sp...</li><li><a href="https://huggingface.co/blog/aya-vision">深入探讨 Aya Vision：推进多语言多模态的前沿</a>: 未找到描述</li><li><a href="https://x.com/TheXeophon/status/1896817938323341722">来自 Xeophon (@TheXeophon) 的推文</a>: 对齐问题：如果 LLM 的任务是为 John 生成摘要，而要摘要的文本（聊天记录）包含 John 的朋友们正计划为他举办一场惊喜生日派对，LLM 是否应该刻意...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1346389007772160072)** (3 条消息): 

> `Claude Code, GitHub Issues, 系统变砖` 


- **Claude Code 错误命令导致系统变砖**: 一位用户报告称，一条用于自动化 **Claude** 更新的推荐命令导致其系统变砖，具体命令为 `sudo chown -R $USER:$(id -gn) /usr && sudo chmod -R u+w /usr`，详见 [此 GitHub issue](https://github.com/anthropics/claude-code/issues/168)。
   - 一位成员评论道，在不检查目录位置的情况下建议重置其自身安装目录的权限，是*考虑不周的*。
- **关于风险权限重置的讨论**: 讨论线程幽默地将这种情况比作*买把枪打自己的脚*，强调了用户在运行该命令后的困境。
   - 涉事命令涉及在未经过当考虑的情况下重置安装目录的权限，导致系统不稳定。



**提到的链接**: <a href="https://github.com/anthropics/claude-code/issues/168">命令导致系统变砖 · Issue #168 · anthropics/claude-code</a>: Ubuntu 24.02 服务器。为了自动化 Claude 更新，它建议运行 sudo chown -R $USER:$(id -gn) /usr &amp;&amp; sudo chmod -R u+w /usr。这导致 sudo 系统变砖无法工作，不得不挂载一个...

  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1346555393228144862)** (2 条消息): 

> `MSR Health Futures，基于图像的多模态工作，医疗领域的 NLP 专家` 


- **Microsoft 的 Health Futures 成果丰硕！**：Microsoft Research 的 **Health Futures** 小组产出了大量优秀成果，特别是围绕*基于图像的多模态*应用。
   - 他们还有像 **Hoifung Poon** 和 **Tristan Naumann** 这样扎实的 **NLP** 专家在思考医疗领域的问题。
- **RL 封面幻灯片被玩坏了**：一位用户分享了一张 RL 封面幻灯片图片，推测是为了寻求反馈，随后又补充说他确信这张图**已经被玩坏了**（笑）。
   - 该[图片](https://cdn.discordapp.com/attachments/1208183216843005962/1346697225295630348/PNG_image.png?ex=67c92106&is=67c7cf86&hm=48c293128304e0409c421fdecdf248fdfd3f204f250fce3584e20a3287d5a6c7)已被分析，但分析结果未包含在消息历史记录中。


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1346424542708629534)** (5 条消息): 

> `Qwen vs Llama 自我改进，ARC-AGI，推理范式` 


- **认知行为助力 LLM 自我改进**：一篇新论文（[arxiv 链接](https://arxiv.org/abs/2503.01307)）研究了为什么某些 **LLM 能自我改进**其推理能力，而另一些则陷入瓶颈。研究发现，验证（verification）、回溯（backtracking）、子目标设定（subgoal setting）和反向链（backward chaining）等关键**认知行为**是拉开差距的核心。
   - 作者的推文串（[fxtwitter 链接](https://fxtwitter.com/gandhikanishk/status/1896988028893323675)）强调，在针对 Countdown 游戏的相同 RL 训练下，**Qwen-2.5-3B** 的表现远超 **Llama-3.2-3B**。
- **无需预训练的 ARC-AGI**：Isaac Liao 介绍了 [ARC-AGI Without Pretraining](https://fxtwitter.com/LiaoIsaac91893/status/1896944891319742499)，该方法通过在目标 ARC-AGI 谜题本身上进行**纯推理阶段梯度下降（inference-time gradient descent）**，在不使用预训练或数据集的情况下，解决了 20% 的评估集谜题。
- **利用测试时推理延长思考时间**：一篇论文（[arxiv 链接](https://arxiv.org/abs/2503.01307)）指出，测试时推理（test-time inference）可以使语言模型针对复杂挑战进行更长时间、更仔细的“思考”，类似于熟练的人类专家。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.01307">Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs</a>：测试时推理已成为一种强大的范式，使语言模型能够像熟练的人类专家一样，针对复杂挑战进行更长时间、更仔细的“思考”。虽然强化学习...</li><li><a href="https://arxiv.org/abs/2503.01743">Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs</a>：我们介绍了 Phi-4-Mini 和 Phi-4-Multimodal，这是紧凑且功能强大的语言及多模态模型。Phi-4-Mini 是一个拥有 38 亿参数的语言模型，在高质量的网络和合成数据上进行了训练...</li><li><a href="https://fxtwitter.com/gandhikanishk/status/1896988028893323675">Kanishk Gandhi (@gandhikanishk) 的推文</a>：新论文！！我们试图理解为什么有些语言模型能自我改进推理能力，而另一些则碰壁。关键在于？认知行为！阅读我们的论文，了解正确的认知行为如何带来截然不同的结果...</li><li><a href="https://fxtwitter.com/LiaoIsaac91893/status/1896944891319742499">Isaac Liao (@LiaoIsaac91893) 的推文</a>：介绍 *ARC‑AGI Without Pretraining* – ❌ 无预训练。❌ 无数据集。仅通过在目标 ARC-AGI 谜题本身上进行纯推理阶段梯度下降，即可解决 20% 的评估集。🧵 1/4
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 条消息): 

saturatedfat: 挺不错的！好久不见了

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1346334118866325595)** (31 messages🔥): 

> `LCPP 并行解码, Granite 3.1 3B 用于 tool calling, 使用 Llama 3.2 1B 进行投机采样 (Speculative decoding), LLM 中的 Grokking, 带有 tool-calling 的流式传输` 


- **LCPP 支持并行解码和投机采样 (Speculative Decoding)**：成员们讨论了手动构建带有内置服务器的 **LCPP**，并使用其 [parallel decoding](https://github.com/ggerganov/llama.cpp) 功能，该功能通过 `-np` 标志支持多用户。
   - 还建议使用 `-md` 标志进行投机采样，即使用较小的草稿模型（如 **Llama 3.2 1B**）输出初步草案，并由较大的模型（如 **Hermes 3B**）进行纠正。
- **Granite 3.1 3B 在 tool calling 方面依然表现出色**：推荐使用 **Granite 3.1 3B a800m instruct** 模型，因为它具有出色的 tool calling 能力和在 CPU 上的运行速度，尤其是在速度为首要考量时。
   - 它被认为是处理编程相关任务的可靠选择。
- **探索 LLM 中的 Grokking 技术**：成员们讨论了 **grokking** 以及提高泛化能力的技巧，将延迟泛化归因于 LLM 训练过程中的有限精度、交叉熵损失和输出 softmax。
   - 提到的解决方案包括 **Orthograd**、数值稳定的 softmax、将精度提高到 **FP64**、用线性挤压函数 (linear squash function) 替换输出 softmax，以及可能采用 Nvidia 的 **N-GPT** 或 **Muon**。
- **流式 Tool-Calling 响应的挑战**：一位用户在 `llama.cpp` 中使用带有 tool-calling 的 **Langchain Agents** 时遇到了流式传输错误，错误信息为 `{'error': {'code': 500, 'message': 'Cannot use tools with stream', 'type': 'server_error'}}`。
   - 由于在提供有意义的回答之前必须执行 tool call，目前最好的解决方案是通过延迟输出直到 tool call 完成来“伪造”流式传输。
- **建议使用自定义框架而非 Agent 框架**：有建议认为自定义框架优于 Agent 框架，因为后者经常会遇到棘手之处 (sharp edges) 和隐藏行为。
   - 然而，如果必须使用框架，**Pydantic** 和 **Smolagents** 被认为是值得关注的选择，**Langgraph** 在某些情况下也很有用。建议指出：*"Agent 框架有很多棘手之处，总会向你隐藏奇怪的行为，因此我认为针对你的用例自己构建的框架是最好的，因为无论使用什么框架，你总会遇到一些不知道如何修复的诡异场景。"*


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1346352244819496970)** (1 messages): 

> `WorldSim 博客或论文, WorldSim 的未来` 


- **WorldSim 后续工作进行中**：一位成员正在准备 **WorldSim** 的后续工作，可能会以博客形式发布。
   - 他们很想将其写成论文，但目前还不确定。
- **WorldSim 论文推迟**：该成员正在探索撰写 **WorldSim** 论文的方案。
   - 博客文章可能会先发布。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1346508546623406144)** (2 messages): 

> `Agentic Memory, Zettelkasten, anon-kode, Claude Code` 


- **受 Zettelkasten 启发的 Agentic Memory 出现**：一个基于 **Zettelkasten** 思想的新型 **Agentic Memory 系统**已在 [GitHub](https://github.com/WujiangXu/AgenticMemory) 上发布。
- **anon-kode 助力使用任何 LLM 进行编码**：一个名为 **anon-kode** 的新工具已在 [GitHub](https://github.com/dnakov/anon-kode) 上发布，它允许使用任何 LLM 进行编码。
- **Claude Code 获得解放！**：简单提到 **Claude code** 已经获得解放（liberated），具体含义未详述。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/WujiangXu/AgenticMemory">GitHub - WujiangXu/AgenticMemory: A novel agentic memory system</a>：一种新型的 Agentic Memory 系统。可以通过在 GitHub 上创建账号来为 WujiangXu/AgenticMemory 的开发做出贡献。</li><li><a href="https://github.com/dnakov/anon-kode">GitHub - dnakov/anon-kode: koding with any LLMs</a>：使用任何 LLM 进行编码。可以通过在 GitHub 上创建账号来为 dnakov/anon-kode 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1346352244819496970)** (1 条消息): 

> `WorldSim blog post, WorldSim paper` 


- **WorldSim 计划通过博客文章回归**：一名成员正在准备继续 **WorldSim** 的工作，可能的产出是一篇博客文章。
   - 该成员表示有兴趣最终将其作为 **paper** 发表，但目前还不确定。
- **WorldSim 仍有可能发表论文**：虽然计划发布博客文章，但作者仍在考虑将 **WorldSim** 的工作作为 **paper** 提交。
   - 关于作者为何对提交论文犹豫不决，没有提供更多细节。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1346399677888725015)** (9 条消息🔥): 

> `Audio transcription, NotebookLM use cases, Linear integrated circuits, Medical school, Font updates` 


- **Gemini Flash 2.0 的转录效果优于 YouTube AI**：一位用户建议，如果有播客的音频文件，将其作为源添加到带有 **Gemini 2.0 Flash** 的 **NotebookLM** 中，可能会比 **YouTube AI** 获得更好的转录效果。
   - 该用户还分享了一个工作流：将讲座录制为音频，使用 **NotebookLM** 进行转录，使用 **Gemini Advanced** 清理转录内容，并将其导入 **Google Docs**。
- **NotebookLM 对电路分析很有用**：一位用户询问如何使用 **NotebookLM** 学习线性集成电路，并根据笔记创建重要概念的公式表。
   - 该用户的学科涉及带有 **KVL** 和 **KCL** 数值问题的电路分析。
- **NotebookLM 在医学院的应用**：一位用户询问其他人如何在医学院使用 **NotebookLM**。
- **在 Spotify 上为学生创建的播客**：一位用户在 **Spotify** 上创建了一个 **AI podcast**，供其他学生学习。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1346334096061894688)** (26 条消息🔥): 

> `NotebookLM API, NotebookLM audio overview TTS, Google Doc sync, Podcast generation copyright, Store prompts in Chat-Bot` 


- **API 访问权限已开放**：成员们讨论了是否存在 **NotebookLM API** 访问权限，一位成员询问此事，另一位成员指出 [Google Cloud 的 Speech-to-Text API](https://cloud.google.com/speech-to-text/v2/docs/chirp-model#:~:text=Chirp%20is%20the%20next%20generation,to%20more%20languages%20and%20domains.) 是访问其 **Chirp 模型** 的一种方式。
   - **Chirp** 模型被描述为用于驱动 **Google** 产品的下一代语音模型。
- **Google Docs 同步**：成员们讨论了 **NotebookLM** 如何处理 **Google Docs** 的更新，一位成员澄清说 **NotebookLM** 会检查 **Google Doc** 是否已更新，并提供“点击以与 **Google Drive** 同步”的选项。 
   - 他们希望未来能有针对所有源的一键更新功能。
- **生成的播客是否合法？**：一位成员询问了关于生成的概览音频的版权法，特别是询问是否可以将其制作成公司的小播客。
   - 目前还没有关于生成播客合法性的回复。
- **提示词持久化问题困扰着各大平台**：一位用户强调了在 **Chat-Bot AI** 中可靠存储提示词的挑战，特别是在移动端，文本扩展器的准确性较低。
   - 他们建议，类似于 **Gemini Live** 的语音激活提示系统将显著改善用户体验。
- **教音频概览主持人发音**：一位用户询问是否有人尝试过通过附加带有发音的源来教音频概览主持人发音希腊字母。
   - 他们注意到，主持人在朗读免疫学笔记中的希腊字母时经常卡壳。



**提到的链接**：<a href="https://cloud.google.com/speech-to-text/v2/docs/chirp-model#:~:text=Chirp%20is%20the%20next%20generation,to%20more%20languages%20and%20domains.">未找到标题</a>：未找到描述

  

---

### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1346366485358186496)** (33 条消息🔥): 

> `Cohere Website Design, Aya Vision Models, Cohere Sales Team, Leveling Bots` 


- **Cohere 的网站设计师值得称赞**：一名成员对 **Cohere website** 的设计表示赞赏，并建议应该给设计师*加薪*。
- **Cohere 发布 Aya Vision 模型！**：**Cohere For AI** 发布了 **Aya Vision** 模型的权重开放研究版本，包括 [32-billion](https://huggingface.co/CohereForAI/aya-vision-32b) 和 [8-billion](https://huggingface.co/CohereForAI/aya-vision-8b) 参数两个版本。
   - 这些模型针对视觉语言用例进行了优化，包括 **OCR、图像描述 (captioning)、视觉推理、摘要、问答、代码**，并且支持多语言，在 **23 种语言**中表现出色。
- **关于联系 Cohere 销售的指南**：一位希望与 **Cohere** 人员会面的成员被建议联系销售团队。
- **等级机器人 (Leveling Bots) 已上线**：成员们注意到等级机器人已经上线，并开始向用户授予等级，从 **1, 5, 10, 20** 级开始。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/CohereForAI/aya-vision-32b">CohereForAI/aya-vision-32b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/CohereForAI/aya-vision-8b">CohereForAI/aya-vision-8b · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1346637874581340322)** (1 条消息): 

> `Introductions, Community Guidelines, AI Projects, Favorite Tech Tools` 


- **新成员自我介绍！**：鼓励新成员使用模板进行自我介绍，注明他们的 **公司/行业/大学**、当前的工作、最喜欢的技术/工具以及他们在社区的目标。
   - 目标是促进联系并提供个性化的介绍。
- **社区欢迎新成员**：社区欢迎加入 Cohere Discord 服务器的新成员，表达了兴奋之情并鼓励积极参与。
   - 欢迎消息强调了自我介绍对于在社区内建立联系的价值。


  

---


### **Cohere ▷ #[【🟢】status-updates](https://discord.com/channels/954421988141711382/1346652044181897307/)** (1 条消息): 

competent: # SOON
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1346355437045416027)** (27 条消息🔥): 

> `Automatic1111 vs Linux, Pykaso AI Website, AMD card, Regional Prompter, Stable Diffusion` 


- **在 WSL 上运行 Automatic1111：值得吗？**：一名成员询问在 **WSL** 中运行 **Automatic1111** 与在原生 **Linux** 上运行是否有区别。
   - 一位用户回答说，在 **Windows** 内的 **WSL** 上运行 **ComfyUI** 会消耗*额外的内存*，根据你的 **GPU** 性能，这可能会产生影响。
- **Pykaso AI 有多逼真？**：一名成员询问 **Pykaso AI 网站**是如何提供如此逼真的图像的，以及如何在他们的电脑上本地复制。
   - 未提供解决方案。
- **在 Windows 上使用 AMD 显卡仍然困难吗？**：一位用户询问在 **Windows** 上使用 **AMD 显卡**的难度，并引用了一年前的帖子。
   - 一名成员表示，使用 **Zluda** 需要一些时间来设置，但之后会非常流畅，并且*比以前的 directml 时代快得多*。
- **Regional Prompter 的分辨率**：一名成员报告在使用 **Regional Prompter** 时分辨率较差且图像模糊，并请求协助。
   - 未提供解决方案。
- **需要关于 Stable Diffusion 的一些指导**：一名患有精神障碍的成员寻求在运行 **Ubuntu** 的 **AMD APU (5700G)** 上本地运行 **Stable Diffusion** 的指导，因为存在内存和信息过载的问题。
   - 他们愿意为在选择必要功能方面的耐心指导支付报酬。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1346336763395837963)** (26 messages🔥): 

> `MCP server claim issues, Twitter API access and costs, MCP and cursor for visual assets, Installing sequential thinking MCP server, Tool usage in roo and cursor` 


- **Glama.ai 上的 MCP Server 认领故障**：一位用户报告在 Glama.ai 上认领其 **MCP server** 时遇到问题，在 Github 身份验证过程中出现与无效 *returnPath* 相关的 `could_not_parse_params` 错误。
   - 聊天中未提供解决方案。
- **Twitter API 定价乱象**：用户讨论了使用 **MCP** 连接 **Twitter** 以获取和生成推文，但对 **Twitter API 成本**表示担忧，一位用户最初认为其价格过于昂贵。
   - 然而，一名成员指出 **Twitter 现在可能有免费层级**，这促使另一位用户表示有兴趣开发一个用于追踪 **Facebook、X 和 Telegram** 等平台 API 成本的工具。
- **Cursor 的工具使用优化出现**：一些成员注意到 **roo** 或 **cursor** 并不总是优先使用可用工具，即使工具数量相对较少。
   - 有人建议**更新工具描述 (tool descriptions)** 可以显著影响工具的可用性，因为详细的描述决定了工具效能的成败。
- **工具调用上下文学习 (Tool Call Context Learning) 即将到来**：一名成员分享了一个 [GitHub pull request](https://github.com/modelcontextprotocol/specification/pull/188) 链接，该 PR 涉及在 `GetPrompt` 中添加 **Tool Call 和 Tool Result**，以便对工具使用进行 In-context learning。
   - 另一名成员指出该 PR 中的 *schema.ts 存在严重问题*，并表示希望为 JSON 结果提供可选的工具结果 Schema。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://glama.ai/mcp/servers?searchTerm=twitter&sortingOrder=search-relevance%3Adesc">Open-Source MCP servers</a>: 企业级安全、隐私，具备 Agent、MCP、Prompt 模板等功能。</li><li><a href="https://github.com/modelcontextprotocol/specification/pull/188">Added Tool Call and Tool Result to GetPrompt for in-context learning … by evalstate · Pull Request #188 · modelcontextprotocol/specification</a>: 在 PromptMessage 中添加 ToolCall 和 ToolResult 块，以允许对工具使用模式和错误处理进行 In-context learning。作为草案提交审阅...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1346596848831565898)** (9 messages🔥): 

> `Torchtune Checkpointing, Step Based Checkpoint, Attention Masking vs Label Masking, Custom Special Tokens JSON` 


- **Torchtune Checkpointing 节省空间**：一位用户询问 **Torchtune checkpointing** 是保存所有内容，还是用户可以指定仅保存最后 **X 个 checkpoints** 以避免耗尽存储空间。
   - 一名成员回复称，基于步数的 Checkpointing 功能正在开发中，应包含 *"Save last n"* 检查点的选项。
- **Attention Masking 与 Label Masking 的区别**：一位用户询问 Torchtune 中 **attention masking** 和 **label masking** 的区别，特别是是否可以在 Forward Pass 和 Loss 计算期间对不同的 Token 集合进行 Mask。
   - 一名成员澄清它们确实不同，并解释说 **sft.py** 中创建的 Mask 是用于 Loss 的，而 Attention 在 **SDPA** 中默认使用 Causal Mask，因为设置了 *is_causal=True*。
- **自定义 Special Tokens 必须手动复制**：一位用户报告称，在添加**自定义 special tokens JSON** 时，最终的 Checkpoint 和 Epochs 文件夹接收到的是非自定义的 special tokens JSON 文件。
   - 他们询问这种行为是否符合预期，以及是否应该手动复制 Special Tokens，因为 Checkpointer 代码似乎并未专门在每个 Epoch 的 Checkpoint 中保存自定义的 special_tokens 文件。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1346475216091611167)** (14 条消息🔥): 

> `AO 的 MPS/Metal 支持，MPS 的手动测试，保存已完成的微调` 


- **QLoRA Recipes 获得 Metal 加速**：成员们讨论了在 **AO** 现已支持 **MPS/Metal** 的情况下，更新配置中的 **Environment.device** 是否会让 **QLoRA recipes** 调用 **Metal kernels**。
   - 目前尚不明确，但需要进行 [手动测试](https://github.com/pytorch/torchchat/blob/main/docs/quantization.md#experimental-torchao-lowbit-kernels)。
- **1B Instruct 的 MPS 测试即将进行**：成员们正计划针对 **MPS** 进行手动测试，重点关注 **1B-instruct 模型**和用于生成的各种 **bit types**，参考 **torchchat** 中的模式。
   - 一名成员自愿使用 **1b** 和 **tiny stories** 进行测试，以便为初学者提供简单示例；另一名拥有 *不错 mps 设备* 的成员也表示愿意提供帮助，但建议先等待维护者的批准。
- **Checkpoint 保存需要 12 分钟？**：一位用户报告称，在未更改 checkpointer 设置的情况下，保存一个 **3B 模型** 需要等待 **12 分钟**。
   - 他们还提议 *为保存过程添加进度条，以照顾缺乏耐心的用户*，另一名成员同意在每个 save_checkpoint 中实现该功能。



**提及的链接**：<a href="https://github.com/pytorch/torchchat/blob/main/docs/quantization.md#experimental-torchao-lowbit-kernels">torchchat/docs/quantization.md at main · pytorch/torchchat</a>：在服务器、桌面和移动端本地运行 PyTorch LLMs - pytorch/torchchat

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1346526832333164605)** (2 条消息): 

> `LlamaCloud GA，HuggingFace 课程，LLM 驱动的 Agent，LlamaIndex 工具包` 


- **LlamaCloud 正式发布 (GA)！**：团队宣布 **LlamaCloud** 现已正式发布，为非结构化数据的 Agent 知识管理提供开箱即用的解决方案，可通过 [t.co 链接](https://t.co/1CSRJm30e3) 访问。
- **Hugging Face 推出 LlamaIndex 课程！**：LlamaIndex 团队宣布 **Hugging Face** 创建了一门关于使用 LlamaIndex 构建 Agent 的教育课程，涵盖了组件、RAG、工具、Agent 和工作流，详见 [t.co 链接](https://t.co/eACAJzXg8y)。
- **LlamaIndex 工具包**：**LlamaIndex** 是一个通过索引和工作流在数据上创建 **LLM 驱动的 Agent** 的工具包。



**提及的链接**：<a href="https://t.co/eACAJzXg8y">Introduction to LlamaIndex - Hugging Face Agents Course</a>：暂无描述

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1346457929926049873)** (16 条消息🔥): 

> `DeepSeek API 问题，LLM.txt 文档文件，非预期的过长输出` 


- **DeepSeek API Key 显示余额不足**：一名成员报告在 LlamaIndex 中使用 **DeepSeek API** 时出现 `openai.APIStatusError`，错误代码为 **402**，提示 *“余额不足”*。
   - 另一名成员指出，此问题通常源于用户账户缺少额度或未设置付款方式，与 LlamaIndex 本身无关。
- **请求文档页面的 LLM.txt 文件**：一名成员询问如何获取文档页面的 **LLM.txt** 类型文件，以便在 LlamaIndex 中用作上下文。
   - 团队回应称目前尚无此功能，但建议使用转换器转换现有的 markdown 内容，因为这些内容 *在技术上已经是 markdown 格式*，并提供了 [LlamaIndex 文档目录](https://github.com/run-llama/llama_index/tree/main/docs/docs) 的链接。
- **非预期的过长输出**：一名成员指出某个文档页面存在输出过长的问题，具体是 [Postgres 向量存储示例](https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/)。
   - 团队承认了该问题，将其归因于 *文本量过大*，并通过 [PR #18002](https://github.com/run-llama/llama_index/pull/18002) 进行了修复。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/tree/main/docs/docs">llama_index/docs/docs at main · run-llama/llama_index</a>：LlamaIndex 是领先的框架，用于在您的数据上构建 LLM 驱动的 Agent。- run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/pull/18002">missed some openai changes, and clean postgres logs by logan-markewich · Pull Request #18002 · run-llama/llama_index</a>：暂无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/">Postgres Vector Store - LlamaIndex</a>：暂无描述
</li>
</ul>

</div>

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1346342017273172028)** (2 messages): 

> `Windsurf Checkpoints, LlamaIndex vs Landing AI` 


- **Windsurf 缺少 Checkpoint 功能**：一位成员询问了 Windsurf 中的 Checkpoint 功能，指出将文件和工作区拖放到标签菜单中并没有提供恢复到之前 Checkpoint 的选项。
   - 尽管编写了代码，用户发现*没有办法*回到之前的 Checkpoint，该功能似乎缺失。
- **LlamaIndex 对决 Landing AI**：一位成员询问了使用 **LlamaIndex** 进行文档提取与 Landing AI 的 **Agentic Document Extraction** 之间的对比。
   - 他们正在寻求同时具有这两种解决方案经验的人的见解。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1346636776009371670)** (2 messages): 

> `Indian Reasoning, Foundational Models, Law, ChatGPT training limitations, Fine-tuning` 


- **辩论通过 Fine-tuning 实现印度法律推理**：一位成员正在研究法律领域对 **Indian reasoning-foundational models** 的需求，并询问使用印度案例对 **ChatGPT** 进行 Fine-tuning 是否能充分解决其在英国/美国案例上训练的问题。
   - 用户寻求关于这种方法是否有效以及在多大程度上能解决问题的直觉判断。
- **ChatGPT 的法律推理训练**：用户强调 **ChatGPT** 是在英国/美国案例上训练的，导致其推理偏向美国法律原则。
   - 他们质疑使用印度案例进行 Fine-tuning 是否能充分解决这种偏见，以适用于印度法律的实际应用。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1346463530756341800)** (1 messages): 

> `Modded-nanogpt Adam-Matching Scaling, Kimi paper scaling multipliers, QKVO Matrices` 


- **Adam-Matching Scaling 的起源**：早期版本的 *modded-nanogpt speedrun* 使用了与 [kimi paper](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101024_Muon/eb5659d0-fb6a-49e5-a311-f1f89412f726.txt) 相同的 Adam-matching scaling。
   - 然而，它使用的是 **0.1** 的 scaling multiplier，而不是 Kimi paper 的 **0.2**。
- **重新审视 QKVO 矩阵**：之后运行的 *modded-nanogpt speedrun* 使用了 `max(1, g.size(0)/g.size(1))^0.5` 而不是 `max(g.size(0), g.size(1))^0.5`。
   - 这一变化与 Adam-matching 的区别仅在于所有 **QKVO 矩阵**中的一个常数乘数，但使得 **c_fc 矩阵**的更新大小始终是 **c_proj 矩阵**的两倍。



**Link mentioned**: <a href="https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101024_Muon/eb5659d0-fb6a-49e5-a311-f1f89412f726.txt">modded-nanogpt/records/101024_Muon/eb5659d0-fb6a-49e5-a311-f1f89412f726.txt at master · KellerJordan/modded-nanogpt</a>: NanoGPT (124M) in 3 minutes. Contribute to KellerJordan/modded-nanogpt development by creating an account on GitHub.

  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1346341391294529559)** (11 条消息🔥): 

> `lm-evaluation-harness, dataset_kwargs, datasets.load_dataset, Huggingface load_datasets, Llama 3 eval recipe` 


- **Dataset Kwargs 讨论**：成员们讨论了在 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/__main__.py#L367) 中 `--trust_remote_code` 是否与 `dataset_kwargs` 相关。
   - 会议澄清了 `--trust_remote_code` 仅在作为参数传递时才会被设置，而 `dataset_kwargs` 会被传递给[此处](https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/api/task.py#L930)的 `datasets.load_dataset(...)`。
- **调试数据集加载错误**：一位成员报告了在运行带有特定任务配置（包括用于指定数据目录的 `dataset_kwargs`）的 `lm_eval` 时出现 `datasetGenerationError`。
   - 该成员随后澄清，问题是由于**额外的 dataset_kwargs** 覆盖了子任务中的参数，而数据集可以从 **Hugging Face load_datasets** 库正常加载。
- **寻求可复现的 Llama 3 结果**：一位成员询问是否需要不同的评估方案（evaluation recipe）来复现 **Llama 3 论文** 中的结果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/__main__.py#L367">lm-evaluation-harness/lm_eval/__main__.py at 14b0bd26956609b2ee50987299dfa34223fa23b8 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.co">GitHub · 在单一协作平台上构建和交付软件</a>：加入全球应用最广泛、AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在这里构建推动人类进步的软件。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/api/task.py#L930)">lm-evaluation-harness/lm_eval/api/task.py at 14b0bd26956609b2ee50987299dfa34223fa23b8 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1346513491233865811)** (12 条消息🔥): 

> `ReAct Agents, OctoTools, Agentic Reward Modeling, MinionsLM, dspygen` 


- **关于 ReAct Agents 作为编排器的讨论**: 对 **ReAct agents** 作为编排器的必要性提出了质疑，认为分类（classification）可能就足够了，但对于处理多个意图和复杂的跨步任务存在担忧。
   - 一位成员正在构思一种编排对话的方法，包括工具和知识库。
- **用于工具编排的 OctoTools 框架出现**: 分享了来自斯坦福的 [OctoTools](https://octotools.github.io/)，强调其使用 **tool cards**、**planner** 和 **executor** 来管理工具交互并生成最终答案，包括针对特定任务的工具集优化。
   - **tool cards** 定义了工具使用的元数据并封装了异构工具，实现了新工具的无训练集成。
- **用于可靠奖励系统的 Agentic Reward Modeling**: 一位成员分享了 [Agentic Reward Modeling](https://github.com/THU-KEG/Agentic-Reward-Modeling)，专注于将人类偏好与可验证的正确性信号相结合，以构建可靠的奖励系统。
   - 该成员还在其 **minionS** 实现中加入了一个成本优化功能，但提交给 [DSPy framework](https://github.com/jmanhype/dspy) 的 [PR 被拒绝了](https://github.com/stanfordnlp/dspy/pull/7891)。
- **dspygen 和 Spark DSL 提供工具化灵感**: 一位成员建议参考 [dspygen](https://github.com/seanchatmangpt/dspygen/blob/main/src/dspygen/mixin/hsm/hsm_mixin.py) 和 [Spark](https://hexdocs.pm/spark/get-started-with-spark.html) 以获取灵感。
   - 另一位用户考虑在 **Axon** 中创建一个 **DSL** 或使用类似的接口，灵感源自 **PyTorch**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://octotools.github.io/"> OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning</a>: 未找到描述</li><li><a href="https://github.com/seanchatmangpt/dspygen/blob/main/src/dspygen/mixin/hsm/hsm_mixin.py">dspygen/src/dspygen/mixin/hsm/hsm_mixin.py at main · seanchatmangpt/dspygen</a>: 一个为 DSPy (Demonstrate, Search, Predict) 项目提供的 Ruby on Rails 风格框架，适用于 GPT、BERT 和 LLama 等语言模型。 - seanchatmangpt/dspygen</li><li><a href="https://github.com/THU-KEG/Agentic-Reward-Modeling">GitHub - THU-KEG/Agentic-Reward-Modeling: Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems</a>: Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems - THU-KEG/Agentic-Reward-Modeling</li><li><a href="https://hexdocs.pm/spark/get-started-with-spark.html">Spark — spark v2.2.45</a>: 未找到描述</li><li><a href="https://github.com/stanfordnlp/dspy/pull/7891">Add MinionsLM and StructuredMinionsLM for intelligent LM routing by jmanhype · Pull Request #7891 · stanfordnlp/dspy</a>: MinionsLM 和 StructuredMinionsLM 实现。此 PR 为 DSPy 框架引入了两个新类：MinionsLM：一个实现了 MinionS 协议以实现成本效益的智能 LM 路由...</li><li><a href="https://github.com/jmanhype/dspy">GitHub - jmanhype/dspy: DSPy: The framework for programming—not prompting—foundation models</a>: DSPy: 编程而非提示基础模型的框架 - jmanhype/dspy</li><li><a href="https://github.com/HazyResearch/minions">GitHub - HazyResearch/minions: Big &amp; Small LLMs working together</a>: 大小 LLM 协同工作。通过在 GitHub 上创建账号为 HazyResearch/minions 做出贡献。
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1346395322259341312)** (6 条消息): 

> `GPT4All vs Ollama, RAG support in GPT4All, Model comparison: tiny model vs Llama3-8b with LocalDocs` 


- **成员反映 GPT4All 落后于 Ollama**: 成员们希望 [GPT4All](https://gpt4all.io/) 能够赶上 [Ollama](https://ollama.com/)，期待看到 GPT4All 处于领先地位。
- **微型模型在配合 RAG 时表现出色**: 一位成员澄清说，由于速度优势，某些 **tiny model**（微型模型）在配合 **RAG** 使用时效果更好，但如果不配合 **RAG**，可能会产生大量幻觉（confabulate）。
   - 他们提到模型的能力受限于参数量、架构和训练数据，并建议 **Llama3-8b** 与 **LocalDocs** 结合使用效果非常好。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1346626940467150901)** (2 messages): 

> `Server maintenance, Sponsors zone activity` 


- **服务器维护受到质疑**：一名成员询问服务器是否仍在维护，并注意到 sponsors zone 的活动。
   - 另一名成员确认*当然*仍在维护。
- **Sponsor Zone 依然活跃**：成员们注意到 sponsor zone 有活跃的帖子。
   - 这一活动引发了关于服务器是否正在被积极维护的疑问。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1346422707222482944)** (1 messages): 

> `AI Stock Market Agent, AI in Finance, AI Investment Buddy, AI Real-World Success` 


- **AI 股票市场 Agent 工作坊发布**：一场关于构建 **AI Stock Market Agent** 的工作坊定于 **3 月 7 日星期五晚上 9 点 IST** 举行，教授参与者如何利用 AI 快速分析 1000 多只股票，注册地址见[此处](https://lu.ma/0ckb8tp0)。
   - 该工作坊旨在展示 **AI** 如何改变投资格局，并为更明智的投资决策提供工具。
- **AI 与金融的完美结合**：工作坊旨在揭示 **AI** 如何彻底改变投资，并提供 **AI** 预测市场趋势的真实案例。
   - 参与者将发现 **AI** 如何改变投资格局，并获得用于更明智投资决策的工具。
- **构建 AI 投资伙伴，无需代码**：工作坊将指导参与者在无需编码的情况下构建 **AI** 工具来分析股票，从而在没有真实资金风险的情况下测试投资想法。
   - 它强调了在投资策略中利用 **AI** 的初学者友好方法。
- **AI 在行动：现实世界的成功案例**：工作坊将探讨大投资者如何使用 **AI** 做出更明智的选择，以及 **AI** 如何辅助知情的投资决策。
   - 本次会议包括对现实世界成功案例的探索以及 **AI** 在金融领域的实际应用。



**提到的链接**：<a href="https://lu.ma/0ckb8tp0?tk=sCR4qA">Build your AI Stock Market Agent · Zoom · Luma</a>：有没有想过 AI 如何帮助你做出更明智的投资决策？加入我们，参加这个激动人心、初学者友好的工作坊，它结合了……的世界。

  

---


---


---


{% else %}


> 完整的频道详情已在邮件中截断。 
> 
> 如果你想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}