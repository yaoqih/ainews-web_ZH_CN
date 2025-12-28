---
companies:
- anthropic
- nvidia
- sakana-ai
- meta-ai-fair
date: '2025-03-22T00:20:28.950836Z'
description: '**Anthropic** 推出了一种新颖的“思考”（think）工具，增强了智能体对指令的遵循能力和多步问题解决能力，**Claude**
  展示了结合推理与工具使用的能力。**NVIDIA** 的 **Llama-3.3-Nemotron-Super-49B-v1** 在 LMArena 排名第 14，以其强大的数学推理能力和
  1500 万条后训练数据集而备受关注。**Sakana AI** 推出了基于数独的推理基准测试，旨在提升 AI 的问题解决能力。**Meta AI** 发布了
  **SWEET-RL**，这是一种强化学习算法，可将长时程多轮任务的性能提高 6%；同时推出了 **CollaborativeAgentBench**，这是一个评估
  LLM 智能体在编程和设计任务中与人类协作能力的基准测试。**Percy Liang** 重新发布了 **HELM** 基准测试，包含 5 个具有挑战性的数据集，用于评估
  22 个顶尖语言模型。'
id: c9965154-d604-4bbd-b2a4-0197cfb2005a
models:
- llama-3-3-nemotron-super-49b-v1
- claude
original_slug: ainews-lots-of-little-things-happened-this-week
people:
- percy-liang
title: 这周发生了很多小事。
topics:
- reinforcement-learning
- reasoning
- benchmarks
- multi-turn-collaboration
- instruction-following
- dataset-release
- model-evaluation
---

<!-- buttondown-editor-mode: plaintext -->**增量更新就是你所需要的一切。**

> 2025年3月20日至3月21日的 AI 新闻。我们为你检查了 7 个 subreddit、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**227** 个频道，以及 **3009** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**318 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

- Claude Code（我们在[上个月](https://buttondown.com/ainews/archive/ainews-claude-37-sonnet/)提到过）举办了一个[小型发布周](https://x.com/_catwu/status/1903130881205977320)
- [NotebookLM 中的思维导图](https://x.com/tokumin/status/1902251588925915429?s=46)
- [Roboflow 发布了他们的 YOLO 竞争对手](https://x.com/roboflow/status/1902810257652351228?s=46)
- [Anthropic 围绕 think 工具制造了很大动静](https://x.com/AnthropicAI/status/1903128670081888756)
- [Gemini 发布了一系列更新](https://x.com/GeminiApp/status/1902752852843331650) 
- [Kyutai Moshi 增加了视觉功能](https://x.com/kyutai_labs/status/1903082848547906011)
- [Topaz 发布了一个快速放大器](https://x.com/topazlabs/status/1902742856512446490)
- [Percy Liang 重新发布了 HELM](https://x.com/percyliang/status/1902890719985160471?s=46)

这一切以及更多内容都在 Twitter/Reddit/Discord 摘要中。我们希望在本周末发布 AINews 周报。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

**模型与基准测试**

- **@AnthropicAI 的最新研究揭示了一个简单的 'think' 工具，它能显著提高 Agent 的指令遵循和多步问题解决能力**：[@alexalbert__](https://twitter.com/alexalbert__/status/1903130655564922911) 在博客文章中记录了这些发现。[@skirano](https://twitter.com/skirano/status/1903152968288932085) 还指出他们为此制作了一个 **MCP**，可以从其官方 Anthropic MCP server 仓库下载。[@_philschmid](https://twitter.com/_philschmid/status/1903019035765170419) 观察到 **@AnthropicAI** 似乎是第一个发布结合推理与工具调用的公司，**Claude** 可以进行推理、生成函数调用、执行调用，然后根据输出继续推理。
- **NVIDIA 的 Llama-3.3-Nemotron-Super-49B-v1 在 LMArena 排名第 14**：根据 [@lmarena_ai](https://twitter.com/lmarena_ai/status/1903116426535375060)，该模型是一个**强大的开源推理模型**，在数学方面表现出色，并公开了一个 15M 的后训练数据集。该模型的排名概览（此前在 LMArena 以代号 "march-chatbot" 进行测试）可以在[这里](https://twitter.com/lmarena_ai/status/1903116429508886904)找到。
- **Sakana AI 正在利用数独谜题 (Sudoku Puzzles) 来增强 AI 推理能力**：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1902913196358611278) 宣布发布基于现代数独变体的新推理基准测试，以挑战 AI 社区，他们认为这些谜题非常适合衡量 AI 推理能力的进展。新的基准测试和训练数据可以在[这里](https://twitter.com/SakanaAILabs/status/1902913196358611278)获取。[@hardmaru](https://twitter.com/hardmaru/status/1902920388117446680) 简单地表示，作为一个物种，我们可以通过玩数独来提高我们的集体推理和问题解决能力。
- **HELM 基准测试有了新的排行榜：HELM Capabilities v1.0**：[@percyliang](https://twitter.com/percyliang/status/1902890719985160471) 指出，他们策划了 **5 个具有挑战性的数据集 (MMLU-Pro, GPQA, IFEval, WildBench, Omni-MATH)** 并评估了 **22 个顶尖语言模型**。
- **Meta AI 发布了 SWEET-RL，这是一种用于长程和多轮任务的新型 RL 算法，可以执行更好的信用分配 (Credit Assignment)**：[@AIatMeta](https://twitter.com/AIatMeta/status/1903146901068988473) 报告称，实验证明 SWEET-RL 在 CollaborativeAgentBench 上的成功率和胜率比其他最先进的多轮 RL 算法实现了 **6% 的绝对提升**，使 Llama-3.1-8B 在现实的协作内容创作中达到或超过了 GPT4-o 的表现。有关这两个版本的更多细节可以在发表于 arXiv 的完整论文中找到。
- **Meta AI 还发布了一个新的 Agent 基准测试：CollaborativeAgentBench**，这是第一个研究 LLM Agent 与人类在后端编程和前端设计等现实任务中进行多轮协作的基准测试：详情见 [@AIatMeta](https://twitter.com/AIatMeta/status/1903146899458363442)。
- **LMArena 新品**：[@Nvidia](https://twitter.com/lmarena_ai/status/1903116426535375060) 的 **Llama-3.3-Nemotron-Super-49B-v1** 位列 **第 14 名**。这是一个强大的开源推理模型——总榜前 15 名，在数学方面表现优异，并带有一个公开发布的 **15M 后训练数据集**。

**语言模型开发与发布**

- **Gallabytes** 加入了 **Cursor** 致力于开发 coding agents：在 **Midjourney** 领导模型开发 3 年后，[@gallabytes](https://twitter.com/gallabytes/status/1902864624510439516) 宣布加入 Cursor。
- **Kyutai Labs 发布了 MoshiVis**，一个端到端的低延迟 Vision Speech Model：[@reach_vb](https://twitter.com/reach_vb/status/1903126742954377445) 指出该模型仅增加了 **206M parameters**，并使用了可学习的 gating mechanism，在搭载 **M4 Pro 芯片的 MacMini** 上每次推理仅增加约 7ms，同时保持了实时性能。
- **NVIDIA 构建了 GR00T N1**，这是一个为人形机器人设计的强大开源 AI 模型：据 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1903066519128641809) 称，这是一个基于 **Eagle-2**、**SmolLM-1.7B** 和 Diffusion Transformer 的 **Vision-Language-Action (VLA)** 模型。它在 NVIDIA L40 GPU 上约 64 毫秒内生成 16 个动作。
- **ByteDance** 刚刚宣布 **InfiniteYou** 在 Hugging Face 上线：据 [@_akhaliq](https://twitter.com/_akhaliq/status/1902937194198700280) 称，该工具用于在保留身份的同时进行灵活的照片重塑（Flexible Photo Recrafting）。
- **Roblox** 刚刚在 Hugging Face 上发布了一个 **Cube 3D** 应用：[@_akhaliq](https://twitter.com/_akhaliq/status/1902882220588605485) 指出它可以**直接从文本生成 3D 模型**。
- **Claude 获得实时网页搜索功能**：据 [@TheRundownAI](https://twitter.com/TheRundownAI/status/1903031351621849254) 称，**OpenAI 的语音 AI** 得到了个性化提升。[@_philschmid](https://twitter.com/_philschmid/status/1903019035765170419) 认为 **@AnthropicAI** 是首个发布结合了 reasoning + tool use 的公司。

**AI 应用与工具**

- **Deep Research x AI Builder 论点**：[@swyx](https://twitter.com/swyx/status/1903121115310067794) 理论化了 prompt-to-app AI builder 与 deep research agent 之间的碰撞路径，建议按需构建 deep research 应用，将 UI generation 和 data generation 拆分为独立的 agents。
- **Dair.AI** 推广使用 **LLM-as-a-Judge**，这是一种通过使用专门的 LLM 作为“法官”来自动评估 LLM 输出的技术：[@dair_ai](https://twitter.com/dair_ai/status/1903098701440061592) 认为这能够实现 AI agents 和 LLM 应用的快速开发。
- **LangChain** 发布了 MCP Adapters：[@LangChainAI](https://twitter.com/LangChainAI/status/1903159677845745736) 宣布了他们新的 TypeScript 库，将 Anthropic 的 MCP 工具与 LangChain.js 和 LangGraph.js 连接起来，具有多服务器支持和无缝的 agent 集成。
- **LlamaIndex** 宣布 LlamaExtract 现已进入公测阶段：这个领先的、genAI-native 的结构化文档提取 agent 能够适配最新的模型，甚至可以处理最复杂的文档结构：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1902880391578653176)。
- **Perplexity** 正在开发 Deep Research 的更新版本：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1902876897773760577) 表示，新版本将投入更多 compute，思考时间更长，提供更详细的答案，支持 code execution，并能渲染行内图表。

**AI 社区与活动**

- **Andrew Ng** 分享了他对 **AI Dev 25 会议** 的观察：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1903147778097983709) 指出 agentic AI 仍然是一个强有力的主题，开发者们正在针对特定数据 fine-tuning 更小的模型，许多演讲者谈到了务实解决问题的重要性，而不是盲目相信 AGI 的炒作。

**优化与训练**

- **Cloneofsimo** 分享了探索训练中极端 beta 值的发现：[@cloneofsimo](https://twitter.com/cloneofsimo/status/1903080158627762234) 指出大的 beta2 似乎至关重要，直到 beta1 也变小，并且小的 beta1 允许小的 beta2。
- **Hamel Husain** 提供了关于训练工具的更新：[@HamelHusain](https://twitter.com/HamelHusain/status/1903140801917403538) 告知观众他将在约 15 分钟后上线（为报名者提供录像）。

**幽默**

- **Neel Nanda** 开玩笑地问是否 21% 的人认为某人不是亿万富翁：[@NeelNanda5](https://twitter.com/NeelNanda5/status/1902869137489019188)。
- **Vikhyatk** 调侃搬到旧金山并找到一间月租仅 6000 美元的房间：[@vikhyatk](https://twitter.com/vikhyatk/status/1902915468928954855)。
- **Swyx** 更新了一个 meme：[@swyx](https://twitter.com/swyx/status/1902935741103215103)。


---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. SpatialLM：用于 3D 场景理解的 LLM**

- **[SpatialLM：专为空间理解设计的大语言模型](https://v.redd.it/9hvol38aozpe1)** ([Score: 1033, Comments: 94](https://reddit.com/r/LocalLLaMA/comments/1jgap0q/spatiallm_a_large_language_model_designed_for/))：**SpatialLM** 是一款专门设计用于增强 **3D 场景理解**的大语言模型，基于 **Llama 1B** 构建。该模型专注于提升空间理解能力，有望在需要详细环境感知能力的应用中提供技术突破。
  - **SpatialLM 的能力**：SpatialLM 处理 **3D 点云数据**以生成结构化的场景理解，识别墙壁和门等建筑元素，并对具有语义类别的物体进行分类。它支持多种数据源，包括**单目视频、RGBD 图像和 LiDAR 传感器**，使其在机器人和导航应用中具有广泛的通用性。
  - **技术查询与澄清**：讨论中提出了关于将 SpatialLM 归类为语言模型的疑问，因为它处理的是非人类可读的数据。对此的澄清是，它输出结构化的 3D 物体图（object graphs），这是一种特定形式的语言，且该模型基于 **Llama 1B 和 Qwen 0.5B**。
  - **模型性能与应用**：用户对该模型仅凭 **12.5 亿参数**所展现的能力感到惊讶，并讨论了潜在的应用场景，例如为视障人士集成文本转语音功能，以及在扫地机器人中的应用。模型估计物体高度的能力及其集成到推理模型中的潜力也受到了关注。


**主题 2. Qwen 3：模块化 AI 模型进展**

- **Qwen 3 即将发布！** ([Score: 402, Comments: 97](https://reddit.com/r/LocalLLaMA/comments/1jgio2g/qwen_3_is_coming_soon/))：根据 **Hugging Face Transformers** GitHub 仓库上的一个 Pull Request 显示，**Qwen 3** 预计很快就会发布。Pull Request 的链接在[这里](https://github.com/huggingface/transformers/pull/36878)。
  - 讨论重点介绍了 **Qwen 3 MoE 模型架构**，特别是它使用了 **128 个专家，且每个 token 激活 8 个专家**，以及 **15B MoE** 的模型大小，这使其非常适合 CPU 推理。用户希望看到更大的模型，如潜在的 **30-40B MoE** 甚至 **100-120B MoE**，以便与现代模型竞争。
  - 几条评论深入探讨了 Qwen 3 的**技术细节和性能指标**，并将其与 **Deepseek v3** 等其他模型进行了比较。**激活参数**被指出为 **2B**，此外还有关于模型潜在性能的讨论，并引用了基准测试和模型等效性计算。
  - 社区对 Qwen 3 的潜力感到兴奋，尤其是它的 **CPU 兼容性**和**较小的激活参数量**，这降低了计算资源需求。人们对其 **Embedding 能力**表现出兴趣，并对其在编程任务中的表现感到好奇，一些用户注意到了其 **152k 的词表大小**和 **32k 的最大位置嵌入**。


**主题 3. Docker 的竞争飞跃：容器中的 LLM**

- **Docker 对 Ollama 的回应** ([Score: 240, Comments: 136](https://reddit.com/r/LocalLLaMA/comments/1jgfmn8/dockers_response_to_ollama/))：**Docker** 正在推出一项支持 **Mac GPU 访问**的新功能，允许用户在自己的机器上运行 `mistral/mistral-small` 等模型。这一更新让用户感到兴奋，因为它通过允许容器利用 Mac 的 GPU 增强了 Docker Desktop 的能力，详情见其[官方公告](https://www.docker.com/llm/)，并在一段 [YouTube 视频](https://www.youtube.com/watch?v=mk_2MIWxLI0&t=1544s)中进行了进一步讨论。
  - 讨论强调了使用 **Ollama** 和 **llama-swap** 等封装器（Wrappers）来管理和运行模型，一些用户批评这些工具是对 **llama.cpp** 不必要的抽象。然而，其他人认为这些工具简化了部署，特别是对于那些对技术设置不深入了解的人，并在分发和托管模型方面提供了模块化和易用性。
  - **Docker 的新功能**开启 **Mac GPU 访问**被视为一项重大进步，允许 Mac 用户在具有 GPU 加速的隔离环境中运行应用程序。这一更新对于使用 **Apple silicon** 的用户尤为重要，并被拿来与 **GitHub Container Registry** 对 Docker Hub 的影响相类比，尽管一些用户对 Docker 的命令行界面表示不满。
  - 关于**开源社区**的方法存在争论，一些用户对 Ollama 等项目进行品牌化运作而不是向 **llama.cpp** 等现有项目贡献代码表示担忧。其他人则为模块化方法辩护，强调在开发和部署中简单性的重要性，特别是在 **AI 模型托管**和管理依赖关系的背景下。

**主题 4. Gemma 3, Mistral 24B 与 QwQ 32B：性能对比**

- **Gemma 3 27b vs. Mistral 24b vs. QwQ 32b：我在个人基准测试上的发现** ([Score: 231, Comments: 74](https://reddit.com/r/LocalLLaMA/comments/1jgau52/gemma_3_27b_vs_mistral_24b_vs_qwq_32b_i_tested_on/))：**QwQ 32b** 在本地 LLM 编程和推理方面表现出色，在某些情况下优于 **Deepseek r1**，并显著超越了 **Gemma 3 27b** 和 **Mistral 24b**。在数学方面，**Gemma** 和 **QwQ** 都能很好地处理简单任务，**Gemma** 速度更快，但其许可证限制较多。**Mistral 24b** 的表现不如其他模型，尽管它与 **Gemma** 一样提供图像支持。欲了解更多详情，请参阅[博文](https://composio.dev/blog/qwq-32b-vs-gemma-3-mistral-small-vs-deepseek-r1/)。
  - **QwQ 32b 的性能与 VRAM 需求**：用户确认 **QwQ 32b** 在编程和推理任务中表现卓越，优于某些云端模型，但注意到其对 **VRAM** 的高需求。这使得即使使用 quantization（量化）也很难在单个 GPU 上运行，从而限制了其 context window（上下文窗口）大小。
  - **模型对比与量化问题**：需要明确对比中使用的 **Gemma 模型类型**，同时对 quantization 设置表示担忧，特别是 **Mistral**，这可能会影响性能。建议硬件资源有限的用户尝试 **RekaAI_reka-flash-3** 和 **ExaOne Deep** 作为替代方案。
  - **基准测试与使用场景**：建议包括在 IDE 中运行 **Gemma, Mistral 和 QwQ** 等模型以进行更实际的基准测试，并测试 **ExaOne Deep** 和 **DeepHermes** 进行对比。用户还强调了 **QwQ 32b** 在转录摘要方面的强劲表现，偶尔甚至超过了 **GPT-4/4.5**。


**主题 5. 字节跳动（ByteDance）的 InfiniteYou：保持身份特征的图像模型**

- **[字节跳动在 HuggingFace 上发布了一个开源图像模型，可在生成照片的同时保持身份特征] (https://i.redd.it/efejft8gf1qe1.jpeg)** ([Score: 128, Comments: 36](https://reddit.com/r/LocalLLaMA/comments/1jgft94/bytedance_released_on_huggingface_an_open_image/))：**ByteDance** 推出了 **InfiniteYou**，这是一款在 **HuggingFace** 上提供的图像生成模型，允许灵活地重新创作照片，同时保留个人身份特征。该项目展示了各种各样的肖像，展示了不同环境下的个人，强调了现实主义与艺术诠释的融合。关键资源包括 [项目页面](https://bytedance.github.io/InfiniteYou/)、[代码仓库](https://github.com/bytedance/InfiniteYou) 以及 [HuggingFace 上的模型](https://huggingface.co/ByteDance/InfiniteYou)。
  - 评论者批评了 **InfiniteYou** 的**图像质量**，称其“粗糙”且“有塑料感”，表明对该模型生成逼真图像的能力持怀疑态度。
  - **macumazana** 指出，之前在旧模型上已经做过类似的工作，暗示 **InfiniteYou** 在该领域没有提供显著的新颖性或进步。
  - **moofunk** 建议采取一种战略性方法，专注于模型优势，并提出通过 **chaining models**（模型链）来提高照片生成质量的想法，而不是依赖单一模型的输出。


## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. 5 秒 Flux 创新：Nunchaku, InfiniteYou 与 Step-Video-TI2V**

- **[5秒 Flux 图像 - Nunchaku Flux - RTX 3090](https://www.reddit.com/gallery/1jg3a0q)** ([评分: 263, 评论: 66](https://reddit.com/r/StableDiffusion/comments/1jg3a0q/5_second_flux_images_nunchaku_flux_rtx_3090/)): **MIT-Han-Lab** 发布了 **ComfyUI-nunchaku**，这是一个用于生成 **5秒 Flux 图像** 的工具。公告中还提到了 **RTX 3090**，尽管没有提供关于其在项目中具体作用的细节。
  - 用户对 **ComfyUI-nunchaku** 的输出质量表示怀疑，指出图像看起来有“塑料感”，且与 **SDXL** 等模型生成的图像相似。用户特别关注人脸的人造感，通常带有 **cleft chins**（裂纹下巴）。
  - **Nunchaku SVDQuant** 提供了显著的性能提升，通过消除 **CPU offloading**，将模型大小减小了 **3.6×**，内存占用减少了 **3.5×**，并在 **NVIDIA RTX 4090** 笔记本电脑上实现了 **10.1× speedup**。该工具支持类似于 **TensorRT** 的 **lora** 转换，并通过 [GitHub](https://github.com/mit-han-lab/ComfyUI-nunchaku) 和 [Hugging Face](https://huggingface.co/mit-han-lab/svdq-int4-flux.1-dev/tree/main) 提供了详细的设置说明。
  - 一位用户分享了使用 **deepcompressor** 仓库对 **flux finetunes** 进行量化的经验，遇到了 **CUDA/transformers** 依赖项和 **VRAM** 限制的挑战，认为 **24GB VRAM** 是不足够的。他们通过租用 **A40** GPU 提供了变通方案，并分享了潜在依赖项修复的步骤。


- **[来自 ByteDance 的 InfiniteYou：基于 FLUX 的新 SOTA zero-shot 身份保持 - 模型和代码已发布](https://i.redd.it/8ohrkqaenzpe1.png)** ([评分: 193, 评论: 59](https://reddit.com/r/StableDiffusion/comments/1jgamm6/infiniteyou_from_bytedance_new_sota_0shot/)): **ByteDance** 推出了 **InfiniteYou**，这是一款基于 **FLUX** 的新型 **SOTA** **zero-shot** 身份保持模型。该模型及其代码已发布，展示了其增强图像中身份特征的能力，正如在包含 **ID Image**、**PuLID-FLUX** 和 **InfU**（本模型）的对比网格中所演示的那样，**InfU** 在身份保持方面表现出先进的渲染和保真度。
  - 围绕 **Flux** 的身份保持讨论揭示了不同的观点：虽然一些用户指出该模型能有效地遵循提示词并保持面部细节，但其他人批评它未能准确复制输入特征（如眼睛颜色和头发），以及“Flux chin”问题。**ByteDance** 的 **InfiniteYou** 被视为向前迈出的重要一步，尽管其真实感受到了一些用户的质疑。
  - **Hugging Face** 是该模型可用性的焦点，用户渴望看到其集成到 **ComfyUI** 工作流中。用户对更好地处理雀斑、疤痕和纹身等特征有需求，这些特征被认为是高质量面部复制所必需的。
  - 用户对当前 **Flux** 模型的美学表示不耐烦，并预测一旦有新的开源模型可用，将会发生转变。**ByteDance** 的方法侧重于研究和方法论而非美学，一些用户认为这在实际的、照片级写实的应用方面有所欠缺。


- **[Step-Video-TI2V - 一个 30B 参数 (!) 的文本引导 image-to-video 模型已发布](https://github.com/stepfun-ai/Step-Video-TI2V)** ([评分: 119, 评论: 61](https://reddit.com/r/StableDiffusion/comments/1jg3mx2/stepvideoti2v_a_30b_parameter_textguided/)): **Step-Video-TI2V** 是一个新推出的 **30B** 参数模型，支持文本引导的 **image-to-video** 转换。此版本的发布标志着 AI 驱动的视频生成领域的重大进步。
  - **模型大小与性能**：**Step-Video-TI2V** 模型拥有 **30B** 参数和 **59GB** 权重，被视为一项重大进步，但其本地使用受到高 **VRAM** 要求的挑战（生成 **720p** 视频最高需要 **70GB**）。用户讨论了其当前资源需求的不切实际，开玩笑说需要一个“肾”才能在本地运行它。
  - **中国 AI 发展**：人们认为中国在 AI 领域正迅速进步，多个视频模型接连涌现，而美国和欧盟则相对滞后。一些用户指出，虽然中国正在生产这些模型，但并不总是能提供最佳质量的输出，正如在 **Yuewen** 的实现中所见。
  - **质量与压缩担忧**：用户对模型中使用的压缩技术表示担忧，这导致尽管模型体积巨大，但细节仍然丢失。该模型依赖 **16x spatial** 和 **8x temporal compression**，被批评为阻碍了其生成精细细节的能力，导致视频输出中出现故障和次优结果。


**主题 2. Text-to-Video AI 的进展：源自开源倡议**

- **[Remade 正在 Hugging Face 上以 Apache 2.0 许可证开源其所有的 Wan LoRAs](https://v.redd.it/4hdgt7rrg1qe1)** ([Score: 171, Comments: 21](https://reddit.com/r/StableDiffusion/comments/1jgfz25/remade_is_open_sourcing_all_their_wan_loras_on/)): **Remade** 正在 **Hugging Face** 上以 **Apache 2.0 许可证**开源其所有的 **Wan LoRAs**，从而在 AI 社区内实现更广泛的访问和使用。
  - 一些用户，如 **Weird_With_A_Beard** 和 **Mrwhatever79**，对 **Wan LoRAs** 充满热情，对将其用于视频生成表示感谢和喜爱。然而，其他人对开源的说法持怀疑态度，强调 **LoRAs** 通常没有许可证，并因其通过 Discord 服务器提供高级服务而质疑开源声明的真实性。
  - **LindaSawzRH** 和 **hurrdurrimanaccount** 批评了开源声明，认为如果未提供训练数据和过程，且访问权限处于付费墙之后，那么这些 **LoRAs** 并非真正的开源。他们对这给社区带来的先例表示担忧，**hurrdurrimanaccount** 质疑数据集是否被共享。
  - **Ballz0fSteel** 对训练 **Wan LoRAs** 的教程表现出兴趣，但 **LindaSawzRH** 认为获取此类信息可能需要付费，这进一步引发了关于资源透明度和可访问性的讨论。


- **[Wan I2V - 起始-结束帧实验性支持](https://v.redd.it/adqnxs24u1qe1)** ([Score: 160, Comments: 21](https://reddit.com/r/StableDiffusion/comments/1jghi5d/wan_i2v_startend_frame_experimental_support/)): **Wan I2V** 引入了对起始-结束帧的实验性支持，增强了其在视频处理方面的能力。此次更新可能会提高视频帧分析的精度和效率。
  - **WanVideoWrapper 更新**：由 **Kijai** 开发的 **WanVideoWrapper** 获得了实验性起始-结束帧支持的更新，该功能此前在 **raindrop313** 的仓库中提供。这项改进允许在场景中引入以前难以通过提示词生成的新物体，尽管仍存在元素缺失和色偏等问题，但可以通过调整缓存和分辨率等参数来缓解。
  - **社区兴奋与测试**：用户对此次更新表示热烈欢迎，一些用户已经在使用 **Kija nodes** 进行测试并报告了积极的结果。该功能被视为脚本化叙事的潜在游戏规则改变者，比之前的版本提供了更高的可靠性。
  - **开源与协作**：社区赞赏该项目的开源性质，强调了 **raindrop313** 等多位开发者的贡献，并对促成这些进展的协作努力表示感谢。


**主题 3. 对 LLM 评估方法的批评：简化与指责**

- **[开火 (Shots Fired)](https://v.redd.it/pwwan4cz3zpe1)** ([Score: 1372, Comments: 284](https://reddit.com/r/ClaudeAI/comments/1jg91lt/shots_fired/)): 批评者认为 **LLM 智力测试** 往往缺乏评估价值，暗示它们无法准确衡量或反映大语言模型的真实能力和智力。这种批评表明需要更严谨、更有意义的评估方法来衡量 AI 性能。
  - **Yann LeCun 的观点**：**Yann LeCun** 被广泛讨论，许多人同意仅靠 **LLMs** 无法实现 **AGI**。LeCun 强调需要超越 LLMs 的新 AI 架构，正如他在 **NVDA conference** 演讲中所述，他因在 AI 领域的重大贡献（特别是深度学习和 CNNs）而受到认可。
  - **LLMs 的局限性**：几位评论者认为，由于架构原因，LLMs 在实现 AGI 方面存在局限，缺乏像人类智能那样学习和适应的能力。有人建议将 **Virtual Intelligence (VI)** 作为当前 AI 能力更合适的术语，强调实用性而非意识或自我意识。
  - **当前 AI 的实用性与误区**：共识是，虽然 LLMs 并非一无是处，但它们是需要正确使用和理解的工具。一些人对 AI 炒作持怀疑态度，指出像 **Claude** 这样的工具已经有所改进并能提高生产力，但它们并不能取代人类工作或实现独立推理。

- **[在我无法解决一个谜题后，我要求一个更简单的](https://i.redd.it/dv77ltcuzwpe1.jpeg)** ([得分: 529, 评论: 113](https://reddit.com/r/ChatGPT/comments/1jg0ilw/after_giving_me_a_puzzle_i_couldnt_solve_i_asked/)): 该帖子讨论了一次 **ChatGPT** 互动。用户在无法解决“三个宝箱”谜题后请求一个更简单的谜题。**AI** 的回答没有讽刺意味，暗示它真心认为用户需要更简单的挑战，这突显了在理解用户意图或语境方面的潜在局限性。
  - **逻辑推理与谜题分析**：**Claude** 对“三个宝箱”谜题的分析展示了经典的逻辑推理方法，质疑标签的准确性，并考虑了如标签错误等潜在转折。讨论强调了需要考虑是否所有标签都是错误的，这将导致在先测试标有“白银”的宝箱后，选择标有“黄金”的宝箱。
  - **幽默与讽刺**：几位评论者，如 **EuphoricDissonance** 和 **Careless_General5380**，利用幽默参与话题，开玩笑说宝藏是“爱”或“一路上结交的朋友”。这反映了围绕谜题简单性和 **AI** 回复的讨论具有轻松愉快的性质。
  - **谜题约束与解决方案**：**Toeffli** 指出了谜题中缺失的一个关于真话和谎言注释的元素，这影响了确定宝藏的位置。**Professional_Text_11** 和其他人注意到没有禁止打开所有三个宝箱的规则，暗示了一个绕过预期谜题逻辑的直接解决方案。


**主题 4. AI 生成的讽刺与历史重构**

- **[巴布工程师 Doge 版——他能搞砸吗？](https://v.redd.it/x64ffkvmpwpe1)** ([得分: 183, 评论: 24](https://reddit.com/r/ChatGPT/comments/1jfz5lt/doge_the_builder_can_he_break_it/)): **Doge The Builder** 通过将 **Elon Musk** 和 **Dogecoin** 与“巴布工程师”（Bob the Builder）进行对比来讽刺他们，突出了贪婪、经济混乱和不受约束的资本主义等主题。该帖子幽默地引用了**自动化真相部 (DOAT)** 的虚构授权，并在评论中提供了一个 YouTube 链接供观看。
  - **AI 的角色**：评论者对 **AI** 在创作如“Doge The Builder”这类内容方面的能力表示赞赏，强调了它在当今时代的惊人表现。
  - **文化影响**：讨论涉及了像 **Elon Musk** 这样的人物对社会时代精神的影响，质疑积累财富的道德性及其对文明的影响。
  - **创作好奇心**：人们对创作讽刺内容的过程感到好奇，并询问此类作品是如何制作的。


- **[5 分钟就做好了。我们很快就需要一些好用的 AI 检测工具了……](https://v.redd.it/nrw8wbjbiwpe1)** ([得分: 13355, 评论: 552](https://reddit.com/r/ChatGPT/comments/1jfy5cz/made_this_in_5_minutes_were_going_to_need_some/)): 该帖子强调了对改进 **AI 检测** 技术的迫切需求，特别是在快速生成的 AI 视频背景下。作者强调了创作此类内容的简便性，暗示了在区分真实视频与 AI 生成视频方面可能面临的挑战。
  - 对 AI 生成视频真实性的担忧非常普遍，像 **YoshiTheDog420** 这样的用户对是否能拥有可靠的 **AI 检测** 工具表示怀疑。他们担心视觉证据可能会变得不可靠，任何片段都可能被斥为 AI 生成，从而破坏对媒体的信任。
  - 讨论强调了人们很容易被 AI 生成的内容所愚弄，正如 **Rude_Adeptness_8772** 所指出的，很大一部分老年人可能会认为此类视频是真实的。**Visarar_01** 分享了一个关于家人被 AI 视频欺骗的轶事，说明了误导信息的潜在风险。
  - 一些评论者，如 **ProfessionalCreme119**，提出了将 **AI 检测** 工具集成到设备中以识别 AI 生成视频的解决方案，建议需要广泛实施检测机制。其他人，如 **Soft-Community-8627**，警告 AI 可能被滥用于捏造事件，这可能会被政府利用来操纵公众认知。


**主题 5. AI 艺术与工作流透明度辩论**

- **我们是否可以开始封禁那些在展示作品时不提供任何工作流细节或所用工具的人？** ([分数: 265, 评论: 56](https://reddit.com/r/StableDiffusion/comments/1jgiok1/can_we_start_banning_people_showcasing_their_work/)): 该帖子建议封禁那些未包含**工作流细节 (workflow details)** 或**所用工具 (tools used)** 的艺术作品帖子，认为如果没有这些信息，这些帖子仅仅起到了广告的作用。作者呼吁进行改变，以确保社区贡献具有信息量并对社区有益。
  - 许多用户，包括 **Altruistic-Mix-7277** 和 **GravitationalGrapple**，反对封禁没有工作流细节的帖子，认为该 subreddit 既是画廊也是学习资源。他们强调了开放式讨论的重要性，以及直接在评论中提问以获取更多细节的能力。
  - **Lishtenbird** 强调了“无工作流”帖子持续存在的问题，并指出详细指南与华而不实、低投入内容之间在互动率上的差异。他们建议实施一个 auto-mod 评论系统，以确保至少分享一些信息（如提示词），尽管这需要额外的资源来实现。
  - **ByWillAlone** 和 **wonderflex** 讨论了该 subreddit 作为艺术展示和学习平台的双重性质。他们提议创建一个独立空间，如 **r/aiartschool**，专门用于深度教程和高投入内容，同时保留投票系统以自然过滤内容质量。


- **[这家伙发布了一个用于变形 AI 纹理的大型 ComfyUI 工作流……非常令人印象深刻 (TextureFlow)](https://www.youtube.com/watch?v=NSQLVNAe5Hc)** ([分数: 105, 评论: 11](https://reddit.com/r/StableDiffusion/comments/1jfxbgr/this_guy_released_a_massive_comfyui_workflow_for/)): **ComfyUI** 发布了一个名为 **TextureFlow** 的重要工作流，用于生成和变形 AI 纹理。该发布因其在 AI 纹理处理方面的卓越能力而备受关注。
  - **TextureFlow** 可通过 [GitHub](https://github.com/edenartlab/workflows/blob/main/workspaces/mono_workspace/workflows/texture_flow/TextureFlow.json) 上的工作流 JSON 直接链接获取。用户正在探索其在 AI 纹理处理和生成方面的能力。
  - 像 **Parulanihon** 这样的用户正在尝试使用 **TextureFlow** 进行 Logo 创建，建议重绘幅度 (denoising level) 最高为 **0.3 或 0.4**。然而，挑战包括实现透明背景以及与过时的 YouTube 教程保持一致，这需要采取混合搭配的方法来达到预期效果。
  - **No-Mistake8127** 正在使用 **TextureFlow** 为定制的 **Raspberry Pi** 驱动的数字相框创作动画艺术作品，强调了它处理视频、文本提示词、照片、动作和 ControlNet 等输入的能力。


---

# AI Discord 摘要回顾

> 由 o1-2024-12-17 生成的摘要之摘要的摘要

**主题 1. 价格对决与审查烦恼**  

- [**Cursor 烧钱**](https://cursor.sh/pricing)：用户对连接错误以及在降级方案时丢失的 Premium 请求所产生的费用感到愤怒。一位成员嘲讽道 *“没有 max，普通的 Agent 就是个烂摊子”*，并因成本效率低下而选择退出。  
- [**OpenAI 的 o1 Pro 过热**](https://help.openai.com)：开发者称 o1 Pro 是一个 *“定价极其离谱”* 的模型，更倾向于 Claude 或 DeepSeek 等更便宜的替代方案。有人开玩笑说 o1 Pro *每次完整发送要花费 30 美元*，成了少数人才能负担得起的奢侈品。  
- [**Pear vs. Cursor 价格战**](https://www.pear.ai)：有人指出 Pear 更便宜，但 *“代码写得一塌糊涂”*，且依赖 *roo code* 进行文件更改。其他人警告说，如果 Cursor 的定价和 Context 限制不改进，他们可能会跳槽。

**主题 2. 模型升级与辩论**  

- [**Claude 3.7 引发热议**](https://www.anthropic.com/news/claude-3-family)：有人坚信 3.7 在 *“精益求精”* 方面表现更好，而另一些人则认为 3.5 更准确。社区达成共识：*“没有一把锤子能胜任所有工作”*，反映了对性能差异的分歧。  
- [**Qwen 3 吸引关注**](https://github.com/huggingface/transformers/pull/36878)：在最近发布 Qwen 2.5 Omni 之后，人们兴奋地追踪 Qwen 3 即将发布的消息。泄露的线索表明它可能会挑战 GPT-4.5 等顶级模型。  
- [**Sora 表现不及预期**](https://huggingface.co/unsloth/DeepSeek-V3)：尽管有大量的预热，这次公开发布让用户感到失望，认为它不如 Keling AI 和 Hailuo AI。批评者怀疑 *“Turbo 版本”* 的炒作掩盖了实际的性能局限。

**主题 3. 微调冒险与 VRAM 之争**  

- [**Gemma 3 经常崩溃**](https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb)：缺失依赖和 `--no-deps` bug 难倒了尝试使用旧版 Colab notebook 的用户。一位开发者感叹道 *“为什么 Llama 在这里失败，但在我的其他环境中运行良好？”*  
- [**QLoRA 解决内存烦恼**](https://discord.com/channels/1179035537009545276/1351288406239346758)：开启 QLoRA 立即降低了 VRAM 占用，让 Gemma 3 可以在更小的硬件上运行。以 4-bit 模式加载有助于避免 OOM 崩溃。  
- [**DeepHermes 24B 撑爆 VRAM**](https://arxiv.org/abs/2408.11857)：用户在多 GPU 设备上运行 24B 时遇到 OOM 错误，即使 Context 极小。建议包括使用 8-bit 版本或使用 *--tensor-split* 等标志微调多 GPU 设置。

**主题 4. 新工具、Agent 与 RAG**  

- [**Oblix 协调边缘与云端**](https://oblix.ai/)：一个精彩的演示展示了 Agent 如何在本地和远程 LLM 之间权衡成本与性能。系统决定是在 Ollama 等硬件上运行查询，还是将其外包给 OpenAI。  
- [**本地 RAG 应用惊艳程序员**](https://twitter.com/llama_index/status/1903121505984319771)：一个完全本地的检索增强生成工具，使用 GitIngest 进行解析，使用 Streamlit 作为 UI 与代码对话。它通过 Ollama 在本地运行 Meta 的 Llama 3.2，令寻求离线解决方案的开发者感到欣喜。  
- [**Semantic Workbench 登场**](https://github.com/microsoft/semanticworkbench)：微软新的 VS Code 扩展在一个地方原型化了多 Agent 系统。用户好奇它是否兼作 MCP 框架，还是主要作为开发工具。

**主题 5. Tokenizer 技巧、合成数据与硬件升级**  

- [**SuperBPE 缩减序列**](https://x.com/alisawuffles/status/1903125390618661068)：一个新铸造的 *superword* tokenizer 在固定的 200k 词表下将序列长度缩短了 33%。测试显示，与标准 BPE 相比，MMLU 提升了 8%，推理速度快了 27%。  
- [**合成数据盛行**](https://arxiv.org/abs/2503.00808)：研究人员强调过滤、增强和生成是 *“拒绝我们已经预测得很好的数据”* 的一种方式。像 Bespoke 这样的开源实验室承诺为有针对性的微调提供新的合成流水线。  
- [**Nvidia 的 Blackwell 引发质疑**](https://www.tomshardware.com/pc-components/gpus/nvidia-blackwell-rtx-pro-with-up-to-96gb-of-vram-even-more-demand-for-the-limited-supply-of-gpus)：下一代 RTX Pro 显卡宣称拥有高达 96GB 的 VRAM，但可能会加剧 GPU 供应短缺。爱好者怀疑 Nvidia 关于 *“我们将在 5/6 月前解决供应问题”* 的说法。


---

# 第一部分：高层级 Discord 摘要

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的定价被指过于苛刻**：用户对 **Cursor 的定价模型** 表示不满，理由包括连接错误、恢复的请求以及“无响应的工具费用”等扣费问题，一些用户在[降级方案](https://cursor.sh/pricing)后报告丢失了高级请求。
   - 一些用户认为 *“普通的 Agent 在没有 max 的情况下表现糟糕”*，并觉得这比“在 max 上花真钱更快”，因为觉得成本效率低而选择退出高级版。
- **Claude 3.7 引发疑虑**：成员们报告了 **Claude 3.7 在 Cursor 中的表现** 问题，称其与 **Claude 3.5** 相比存在错误假设且可靠性下降，而[一些人则有相反的体验](https://www.anthropic.com/news/claude-3-family)。
   - 观点各异，一位用户表示 *“3.7 更擅长精益求精，3.5 则在准确性上更胜一筹”*，而另一位则指出 *“没有哪把锤子能胜任所有工作”*。
- **Pear 的潜力引发昂贵的问题**：用户将 **Pear AI** 与 **Cursor** 进行对比，注意到 **Pear** 的定价更便宜，但也担心其对 **roo code** 的依赖以及逐文件更改确认的工作流，而[其他人则提到 Pear 的编程能力乏善可陈](https://www.pear.ai)。
   - 一些 **Cursor** 用户（如一位表示 *“我不太喜欢 Pear AI，主要是因为他们使用 roo code，而 roo code 不太稳定”* 的用户）正在考虑如果 **Cursor** 不改进其上下文窗口或定价，就转向其他工具。
- **React 之争引发对手戏**：频道内辩论了在 SaaS 应用中 **React** 与 **Svelte** 的优劣，一些人因其庞大的社区以及与 **Cloudflare Pages** 的兼容性而偏好 **React**，而另一些人则认为它缓慢且混乱，[转而支持 Svelte](https://svelte.dev/)。
   - 用户群体分歧较大，争论点从 *“React 慢得要命”* 到 *“Svelte 也不需要各种变通方案”* 不一而足。
- **Vibe Visions 差异巨大**：成员们讨论了 **vibe coding** 的实用性，有人称其为 *营销手段* 和 *胡言乱语*，而另一些人则认为这是一种需要技术专长的真实存在，比如[对 Git 的基础了解](https://git-scm.com/)。
   - 尽管定义各异，但达成的共识是：成功的 “vibing” 需要批判性思维、调试技能以及有效引导 AI 工具的能力。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 3 遇到依赖故障**：根据[此讨论](https://discord.com/channels/1179035537009545276/1351288406239346758)，**Gemma 3** 存在一个与 `--no-deps` 相关的 bug，导致旧版 notebook 中出现依赖缺失；此外，配备 **2018 GPU** 的 Google Colab 对于某些任务可能过于陈旧。
   - 根据[此 notebook](https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb)，一位用户在 **Gemma** 专用环境中运行 **Llama** 时遇到问题，但同样的 notebook 在 Google Colab 上因缺少依赖而运行失败。
- **视觉微调（Vision Fine-Tuning）仍处于 Unsloth 的次要优先级**：根据 [GitHub Issue](https://github.com/unslothai/unsloth/issues/2131)，尽管 **Gemma 3** 支持图像，但 Unsloth 尚未支持视觉微调。
   - 一位用户尝试使用 **Llama** 代码对 **Gemma 3** 进行微调但失败了，他们仍想知道在仅对文本进行微调后，模型是否还能处理图像。
- **QLoRA 助力 Gemma 3 解决内存问题**：用户在运行 **Gemma 3** 模型时遇到了内存错误，但启用 **QLoRA** 解决了该问题，这可能归功于[此处](https://discord.com/channels/1179035537009545276/1351288406239346758)提到的 VRAM 占用降低。
   - 开启 **QLoRA** 会自动设置 `load in 4bit = true`，这有助于减少 VRAM 使用。
- **社区寻求合成数据（Synthetic Data）解决方案**：成员们讨论了合成数据生成工具，一位用户因 **Bespoke Labs** 功能丰富而推荐了它，并确认它是开源的，且拥有专门的 [Discord 服务器](https://discord.com/channels/1179035537009545276/1351288406239346758)。
   - 有用户询问是否有演示 **GRPO 配合视觉模型** 实现的示例 notebook 或 Colab，但目前尚缺乏此类示例，不过已在未来计划中。
- **DPO Trainer 获得升级**：一位用户分享了使用最新的 **Unsloth** 和 **Unsloth Zoo** 升级到最新 **DPO Trainer** 的经验，并为面临类似挑战的人提供了[其小型 diff 的链接](https://github.com/toranb/sloth/commit/9abead851f5531642470f9a22b5ae00af91a8cb6)。
   - 该用户还发现 [Zephyr (7B)-DPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb#scrollTo=-kyd_iyz7DUM) 令人困惑，并建议通过向 [Unsloth notebooks 仓库](https://github.com/unslothai/notebooks)提交 pull request 来进行更新。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI o1 Pro 定价引发不满**：用户对 **OpenAI** 为 **o1 Pro 模型** 制定的 **API 定价** 感到不满，称其价格严重偏高，并表示更倾向于使用 **Claude**。
   - 一些人调侃了 OpenAI 的定价策略，并观察到根据分享的图表，**DeepSeek** 以极低的价格提供了相当的性能。
- **围绕 o1 架构的争论**：Discord 用户正在争论 **OpenAI o1 模型** 是否基于 **GPT-4o**，关于其架构的说法存在冲突。
   - 争论焦点在于知识截止日期（knowledge cutoff dates）；一些人认为 o1 只是 *带有推理能力的 gpt4o*。
- **Perplexity 桌面应用提升用户忠诚度**：**Perplexity** 正在奖励桌面应用用户，在**使用 7 天后**可获得**一个月免费 Pro 会员**。
   - 该奖励仅限 **Windows 应用**，不包括 macOS、iOS、Android 和 Web 用户。
- **GPT Pro 订阅问题困扰用户**：用户报告称已支付 **GPT Pro** 费用但无法获得订阅权限，并对 **OpenAI 支持团队** 的不予回应表示沮丧。
   - 受影响的用户被引导至 [help.openai.com](https://help.openai.com) 寻求支持，并被告知该频道无法协助处理账单事务。
- **结构化输出（Structured Output）阻碍 AI 推理**：成员们测试了短语 **"No other keys or commentary are allowed"** 是否会降低结构化输出中的推理能力，结果发现确实存在负面影响，且 Token 使用量有所增加。
   - 结果表明，在这些条件下，模型会过度思考伦理影响。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio API 寻求 RAG 集成**：用户正关注 **LM Studio server API** 与 **RAG** *(Retrieval-Augmented Generation)* 集成的潜力，类似于 **Ollama** 和 **Qdrant**。
   - 一位用户表示，虽然 GUI 仅获取前 3 个向量，但 API 可以通过 **embeddings** 和向量数据库实现自定义实现。
- **ZeroGPU Pro 用户遇到配额限制**：一位 **ZeroGPU Pro** 用户在升级后仍触及了 GPU 配额限制，可能是因为他们使用了 **FastAPI** 后端而不是 **Gradio UI**。
   - 他们正在寻求关于从自己的应用程序调用 **ZeroGPU Pro** API 时解决配额问题的建议。
- **LM Studio 激发浏览器扩展创意**：社区正在讨论 **LM Studio** 的潜在浏览器扩展，包括使用 **Gemma 3 27b** 进行网页翻译和 **YouTube** 视频摘要。
   - 一位成员建议通过提取和总结字幕来总结 **YouTube** 视频，而由于速度限制，实时网页翻译的可行性受到了质疑。
- **音频模型“炼金术士”使用 PyTorch 酿造**：一位成员正尝试使用 **PyTorch** 和 **Transformer** 架构从头开始预训练一个音频模型，旨在从 **tokens** 生成正确的音频。
   - 另一位成员分享了其模型根据名称（例如 *abba.mp3*、*mj.mp3*）生成的歌曲输出，并建议进行微调或将模型上传到 **Hugging Face** 以进行更广泛的实验。
- **RX 9070 所有者报告速度缓慢**：几位拥有新 **RX 9070** 显卡的用户报告推理速度比旧显卡慢，其中一位用户报告在使用 **Granite 3.1 8B Q8_0** 模型时，速度从 **5-7 tok/s** 降至约 **3 tok/s**。
   - 性能问题被怀疑源于 **AMD** 的 **Vulkan** 驱动程序 bug。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Claude Code 效仿 Aider 的网页搜索**：一位用户观察到 [Claude code](https://www.anthropic.com/claude-pricing) 正在以类似于 **Aider** 的方式实现**网页搜索**，这在 [X 上的一个帖子](https://x.com/_catwu)中得到了展示。
   - 随后有人澄清，新的 **Claude** 网页搜索功能目前仅限 **Claude Desktop**。
- **Aider 的提交标志引发 Hook 烦恼**：根据 [aider/repo.py 代码](https://github.com/Aider-AI/aider/blob/14f140fdc52fbc7d819c50eca3de1b3e848282f3/aider/repo.py#L136)，**Aider** 在提交期间添加了 `--no-verify` 标志，绕过了系统 **hooks**。
   - 维护者解释说，这是因为 **commit hooks** 可能会导致任意奇怪的事情发生，并建议使用 [lint 和 test hooks](https://aider.chat/docs/usage/lint-test.html#code-formatting-linters) 作为替代方案。
- **o1-pro API 成本让用户望而却步**：通过 API 尝试 **o1-pro** 的用户报告称，每次完整发送的成本高达 **$30**，昂贵得令人难以承受。
   - 高昂的成本引发了关于缓存机制的讨论，并推测 **OpenAI** 的自动 **prompt caching** 是否能帮助降低费用。
- **Ubuntu 上的 Pipx 包安装困扰**：一位用户在 **Ubuntu** 上为所有用户安装 **Aider** 时遇到困难，尽管得到了使用 `sudo pipx install --global aider-chat` 的建议。
   - 在克服了 **pip** 和版本冲突问题后，他们最终通过在 `/usr/local/bin` 使用 [uv 安装](https://github.com/Aider-AI/aider)获得了成功。
- **Aider 的自动修复需要手动提示**：一位用户报告称，尽管启用了 `--auto-test` 参数，**Aider** 在每次失败后仍需要手动提示（如 *"fix the tests"*），并引用了[此处的文档](https://aider.chat/docs/usage/lint-test.html#testing)。
   - 如果配置了 `"--auto-test"` 设置，**Aider** 应该会自动修复测试失败。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Deep Research 使用限制引发激烈辩论**：用户正在就 **Deep Research** 的使用限制展开辩论。[Perplexity 博客](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)称 Pro 用户拥有 *无限次* 访问权限，而其他用户则引用了 **每天 500 次查询** 的限制。
   - 一名成员指出了 [Aravind Srinivas](https://x.com/AravSrinivas/status/1890464738951233536) 的一条推文，其中指出 *付费用户只需每月支付 20 美元，即可针对任何主题每天进行 500 次专家级研究查询*。
- **GPT 4.5 的消失引发困惑**：用户报告 **GPT 4.5** 从 **Perplexity Pro** 中消失了，一些人认为该模型在吸引新订阅者后被移除了。
   - 一些用户称赞 **4.5** 是 **文本写作的 SOTA**，而另一些用户则认为它速度慢且缺乏洞察力，这在用户群中造成了不确定性。
- **Perplexity 用户因自动模型切换故障感到沮丧**：用户正经历一个故障，即即使选择了 **Claude** 等特定模型，**Perplexity** 也会自动恢复到 *Auto* 模型。
   - 这个问题需要用户手动重新选择他们喜欢的模型，导致了挫败感，尤其是在那些比起 **R1** 更青睐 **Claude** 的用户中。
- **API Key 支出追踪功能请求**：一项功能请求已提交至 [GitHub](https://github.com/ppl-ai/api-discussion/issues)，旨在允许用户为 API Key 命名，以便更好地追踪支出。
   - 目前，用户可以按 API Key 追踪支出，但无法分配名称，这阻碍了对 API 使用成本的高效管理。
- **R1-1776 微调面临审查质疑**：一位独立研究员发现，在针对天安门广场等话题进行提示时，**R1-1776-671B** 和蒸馏版的 **R1-1776-70B** 中存在预设的回答和被审查的内容，详见[这篇博客文章](https://dsthoughts.baulab.info/)。
   - 研究人员对该模型开源权重中的政治偏见和内容过滤表示担忧。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Claude 推出网页搜索功能**：网页搜索现已在 [claude.ai](https://claude.ai) 上线，使 **Claude** 终于能够搜索互联网并为研究查询提供 **真阳性（true positives）** 结果，[这条推文](https://x.com/alexalbert__/status/1902765482727645667?s=46)证实了这一点。
   - 随后确认 Claude 使用的搜索引擎是 [Brave](https://simonwillison.net/2025/Mar/21/anthropic-used-brave/)。
- **Midjourney 负责人从美学转向代码**：在 **Midjourney** 领导模型开发 3 年后，一位核心成员加入了 **Cursor** 致力于 Coding Agent 的研发，标志着从关注 *美学与创意* 向 *代码* 的转变，如[这条推文](https://x.com/gallabytes/status/1902864624510439516)所述。
   - 此举标志着在编程环境中对实用 AI 应用的重视程度日益提高。
- **InternVL 训练代码开源**：成员们对 **InternVL** 拥有开源训练代码感到惊讶，使其成为少数拥有开放训练流水线的知名模型之一，[InternVL 的 packing 实现](https://github.com/OpenGVLab/InternVL/blob/34a81000402bf8f716bab8c9b57aff1f6b436bd0/internvl_chat/internvl/train/dataset_packed.py)被作为数据加载方法的示例提供。
   - **InternVL** 的开源特性允许社区检查数据加载过程和数据集迭代。
- **SuperBPE Tokenizer 提升效率**：**SuperBPE** 是一种新型的 *superword* Tokenizer，包含跨越多个单词的 Token。它创建的模型在 30 个下游任务上始终优于 **BPE** 基准（MMLU 提升 8%），同时在推理时的效率提高了 **27%**，详见[这条推文](https://x.com/alisawuffles/status/1903125390618661068)。
   - 在 **200k** 的固定词表大小下，**SuperBPE** 平均将序列长度缩短了 **33%**。
- **小型模型受益于合成数据增强**：成员们讨论了 **小型数据集** 是否成为新趋势，大型模型（如 **GPT-4.5**）可能需要更多数据，尤其是在各种 Post-training 阶段。对话还涉及使用合成数据（Synthetic Data）来增强小型数据集以训练小型模型。
   - 讨论表明在数据规模、模型规模和合成数据的使用之间存在权衡，暗示了一种策略：小型模型可能更多地依赖增强后的数据集，而大型模型则可以有效地利用大规模的原始数据。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Claude 被高估，Grok3 仍是王者？**：社区成员认为 **Claude** 在编程方面被高估了，因为除了 **SWE-bench** 之外的评估有限，并暗示它在 livecodebench 上不如 **Grok3**。
   - 评分可能受到非开发者的影响，导致对其真实能力的评估不准确。
- **Gemma 获得好评**：成员们对 **Gemma3** 的 **1340** 分及其相对较小的 **27B** 参数量感到惊讶。
   - 一位成员形容 **Gemma** 的回答表现得像“自闭症”一样，回答非常简短，而通常情况下需要更详尽的回答。
- **Deepseek R1 占用大量 VRAM**：**Deepseek R1** 需要约 **1000GB** 的 VRAM，一名用户在 **8xH200s** 上部署了它。
   - 尽管 VRAM 占用率很高，但有说法称 **Deepseek R1** 表现出内置的“亲华”偏见，引发了对其使用的担忧，一位用户表示 *太长不看，Deepseek 就是 #&*&*@%，不建议使用*。
- **Qwen 3 即将推出，Qwen 2.5 Omni 发布**：报告显示 **Qwen 3** 即将推出，[Hugging Face Transformer 仓库](https://huggingface.co/unsloth/DeepSeek-V3)的一篇帖子证实了这一点。
   - 这一消息是在 **Qwen 2.5 Omni** 发布之后传出的，引发了社区的兴趣和期待，正如 [Lincoln 🇿🇦 的推文](https://x.com/Presidentlin/status/1903102260155908200)中所提到的。
- **Sora 的 Turbo 版本表现不佳，热度与现实不符**：用户发现 **Sora** 的公开发布版本与其宣传材料相比令人失望，可能不如 **Keling AI** 和 **Hailuo AI** 等竞争对手。
   - 据推测，OpenAI 使用了大量的算力耗时数小时才生成了那些宣传视频，而发布的 **Sora** 版本是 *turbo 版本*。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NLM 的播客功能评价褒贬不一**：用户报告了对 **NotebookLM 播客功能** 的正面体验，尽管有些人发现 AI 在讨论过程中会打断他们。
   - 一位用户将这种体验比作 *参加一个可以与主持人交谈的广播节目*，但感觉自己像个 *多余的人*，因为 AI 会退回到自己的脚本中。
- **Gemini 1.5 Pro 为 NotebookLM 提供动力**：用户讨论了 NotebookLM 的底层模型，推测指向 **Gemini 1.5 Pro**，而其他人则认为是 Gemini 2.0。
   - 讨论强调了 NotebookLM 保持立足于其源材料的重要性，这是其区别于 [Gemini](https://gemini.google.com/) 的关键点。
- **用户寻求简化的 PDF 处理流程**：一位用户正在寻求一种更高效的工作流，将纸质文件扫描到私人在线存储中，并通过自然语言查询使其可搜索，并询问使用 iPhone 拍照并发送到 NLM 进行自动命名和 OCR 是否更高效。
   - 目前的手动流程包括扫描为 PDF、发送到 Gmail、手动命名每个文件、进行 OCR 处理以及导入 NotebookLM。
- **AI 数字人对嘴同步服务对比**：成员们对比了 AI 数字人的对嘴同步服务，指出 [Hedra](https://www.hedra.io/) 效果很好但价格昂贵。
   - RunwayLM 获得的反馈较差。
- **思维导图功能缓慢推出**：**思维导图 (Mind Map)** 功能的推出进展缓慢，许多用户（包括 Plus 订阅者）尚未在他们的账户中看到该功能。
   - 工作人员确认所有用户获得该功能还需要 *几天时间*。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nvidia Blackwell RTX Pro 引发供应链担忧**：Nvidia 为各种平台推出了 [Blackwell RTX Pro 系列](https://www.tomshardware.com/pc-components/gpus/nvidia-blackwell-rtx-pro-with-up-to-96gb-of-vram-even-more-demand-for-the-limited-supply-of-gpus)，这可能会挤压本已紧张的 **Blackwell GPU** 供应。
   - 虽然 Nvidia 预计 **GPU** 供应情况将在 5月/6月 得到改善，但社区成员仍持怀疑态度。
- **数据集评估与增强至关重要**：讨论强调了数据集评估、增强、排序和分类是利用 **GPU** 时长的有效方法，并建议使用小模型来过滤数据。
   - 一位成员指出使用小模型拒绝数据的潜力，称该领域在“公开领域尚未得到充分探索”，并引用了 [Predictive Data Selection](https://arxiv.org/abs/2503.00808) 和 [Programming Every Example](https://arxiv.org/abs/2409.17115)。
- **DeepHermes 24B 在多 GPU 配置上受阻**：一位用户在 5x 3090 配置上使用 `llama.cpp` 运行 **DeepHermes 24B** 时遇到了 **Out-of-Memory (OOM)** 错误，即使在最小上下文设置下也是如此。
   - 建议的解决方案包括使用 **8-bit 版本**，并通过 `--device`、`--split-mode` 和 `--tensor-split` 标志验证多 GPU 配置。
- **Hermes 3 结合 Llama 3.2 强力升级**：Nous Research 发布了 **Hermes 3 3B**，这是 **Hermes** **LLM** 系列的新成员，详情见 [Hermes 3 技术报告](https://arxiv.org/abs/2408.11857)。
   - 与 **Hermes 2** 相比，该模型具有先进的 Agent 能力，并改进了角色扮演、推理、多轮对话和长上下文连贯性。
- **C# 开发者助力 Anthropic LLM**：一位开发者向社区提供了他们的 **C#** 专业知识和职业 **LLM** 经验，重点介绍了他们在 **Anthropic** 文档和示例方面的工作。
   - 他们引用了基于 **Titanfall 2** 的生成器和来自 **Metal Gear Rising** 的 **Bladewolf** 示例，可在 [Anthropic GitHub](https://github.com/) 上访问。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face API 遭遇 404 崩溃**：多个 Hugging Face API 模型遭遇了广泛的 **404 错误**，导致依赖这些模型的应用程序出现了严重的停机。
   - 用户报告称停机持续了*几乎一整天*，且没有官方确认，敦促 HF 开发团队立即关注。
- **Roblox 语音安全分类器发布**：**Roblox** 发布了一个[大型分类模型](https://huggingface.co/Roblox/voice-safety-classifier)，该模型基于 **2,374 小时** 的真实语音聊天训练，用于检测违规内容。
   - 该模型输出一个带有 `Profanity`、`DatingAndSexting`、`Racist`、`Bullying`、`Other`、`NoViolation` 等标签的 Tensor，并使用了[这篇博文](https://research.roblox.com/tech-blog/2024/06/deploying-ml-for-voice-safety)中详述的合成数据流水线。
- **通过 Tensor 技巧融合 GPU VRAM**：用户探索了合并多块 GPU VRAM 的技术，例如使用[张量并行 (Tensor Parallelism)](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_multi) 在 **A2000 12GB** 和 **1060 6GB** 上运行 **Gemma3-12B**。
   - 文中引用了 [GitHub 上的 Ollama issue](https://github.com/ollama/ollama/issues/2672) 和 [llama.cpp 讨论](https://github.com/ggml-org/llama.cpp/discussions/8725) 以获取更多关于多 GPU 支持的信息。
- **Oblix 平台在云端和设备间调度 AI**：[Oblix.ai](https://oblix.ai) 平台根据复杂性、延迟要求和成本考虑，智能地将 AI 任务路由到云端或边缘端，并使用自主 Agent 实现最佳性能。
   - [YouTube 视频](https://youtu.be/j0dOVWWzBrE?si=Gpp2SG4Kly0tzM_3)展示了 Oblix 如何动态决定是在本地还是在云端处理每个 AI 请求。
- **Gradio 升级导致 Dataframe 换行功能失效**：一位用户报告称，升级到 **Gradio 5.22** 导致 `gr.Dataframe(wrap=True)` 功能失效；该换行功能仅在 **Gradio 5.20** 中正常工作。
   - 目前没有关于此问题的进一步信息。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Microsoft 推出 Semantic Workbench**：Microsoft 发布了 [Semantic Workbench](https://github.com/microsoft/semanticworkbench)，这是一个 VS Code 扩展，用于原型化智能助手、Agent 和多 Agent 系统，引发了关于其作为 **MCP** 角色的讨论。
   - 一位成员专门询问该工具是否作为 **MCP** 运行。
- **MySQL Server 报错**：一位用户在将 **mcp-mysql-server** 连接到 **Docker MySQL** 时遇到问题，报告称尽管在 **MCP** 之外可以正常工作，但连接仍然失败。
   - 每次连接尝试都会报错，造成了显著的开发*障碍*。
- **Glama API 500 错误**：一位用户报告收到来自 **Glama API** 的 **500 错误**，但另一位成员表示过去 24 小时内没有发生停机，并分享了代码示例。
   - 用于重现的代码为 `curl -X 'GET' 'https://glama.ai/api/mcp/v1/servers?first=10&query=github' -H 'accept: application/json'`。
- **DaVinci Resolve MCP 寻求快速服务器认领**：一位用户正寻求重新提交带有许可证和更新的 **DaVinci Resolve MCP** 项目，并被告知认领服务器可能会加快更新过程。
   - 该项目的 [repo](https://github.com/samuelgursky/davinci-resolve-mcp) 托管了相关代码。
- **日历调度实现自动化**：一篇博客文章详细介绍了如何结合使用 **Asana MCP**、**Google Calendar MCP** 和 Goose 来自动执行任务调度，参考 [blog post](https://block.github.io/goose/blog/2025/03/20/asana-calendar-mcp)。
   - 通过单个提示词即可从 Asana 提取任务、进行分析并在 Google Calendar 中进行调度。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 关注 TTS、图像生成上线**：成员们对 **OpenRouter** 提供 **TTS 和图像生成** 表示出兴趣，一些人对潜在的高昂定价表示担忧。
   - 新功能的定价细节和发布日期仍处于保密状态。
- **Groq 遇到故障，而非 Sambanova**：一位成员报告 **Sambanova** 宕机，但随后迅速纠正了说法，澄清是 **Groq** 遇到了问题。
   - **Groq** 的服务状态更新未能立即获取。
- **GPT-4o 登陆 OpenRouter**：**GPT-4o-64k-output-alpha** 现已在 **OpenRouter** 上线，支持**文本和图像输入以及文本输出**。
   - 定价设定为 **每百万输入 token $6** 以及 **每百万输出 token $18**。
- **Fireworks 激化价格战**：**Fireworks** 大幅削减了 **R1 和 V3** 的价格，据称 V3 匹配现有性能，定位于 **.9/.9**。
   - 此举加剧了生成式 AI 服务市场的竞争；更多信息可以在 [Fireworks pricing page](https://fireworks.ai/pricing) 找到。



---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Nvidia 谈论 Python 化的 CUTLASS**：参会者将在 GTC 上了解 **CUTLASS** 在其下一个主要版本 4.0 中的 Python 化未来，特别是它与 Python 的集成。
   - 此前，一位成员宣布了他们题为 *Performance-Optimized CUDA Kernels for Inference With Small Transformer Models [S73168]* 的 GTC 演讲，该演讲于今日下午 4 点举行，重点关注 **Hopper architecture**。
- **BFloat16 原子加法效果不佳**：一位成员报告称，使用带有锁的 `tl.atomic_cas` 进行 **bfloat16 原子加法 (atomic addition)** 确实可行，但[效果很差](https://github.com/openai/triton/blob/main/python/triton/runtime/jit.py)。
   - 该成员正在寻求实现方案的改进，并提供了一个使用带有锁的 `tl.atomic_cas` 的代码片段，邀请社区来提升其性能。
- **Triton 的简洁性吸引 GPU 新手**：一位成员强调，**Triton** 的核心优势不在于峰值性能，而在于其易用性，使 GPU 经验有限的人也能创建复杂的 kernel，并以 [lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch/blob/main/native_sparse_attention_pytorch/triton_native_sparse_attention_pytorch.py) 为例。
   - 他们指出，在预定义的工作负载上实现峰值性能相对简单，但 Triton 的鲁棒性才是其脱颖而出的原因。
- **FlashMLA 的 SmemLayoutP 揭秘**：一位成员询问了 [FlashMLA](https://github.com/deepseek-ai/FlashMLA/blob/b31bfe72a83ea205467b3271a5845440a03ed7cb/csrc/flash_fwd_mla_kernel.h#L76C5-76C93) 代码中 `SmemLayoutP` 的维度，特别是其形状 `((2,2), kNThreadsS, 1, kBlockN/8)` 以及 `kNThreadsS` 在 warpgroups 之间同步 **P** 的作用。
   - 该成员推测其他维度是否可能与 **wgmma** 相关，正等待其他专家的澄清。
- **Grayscale 排行榜表现出众**：在 `grayscale` 排行榜上，多个提交在 GPU 上运行成功：**L4**、**T4**、**A100** 和 **H100**，使用的是 ID 为 `2351`、`2429`、`2430`、`2431`、`2459` 和 `2460` 的 Modal runner。
   - ID 为 `2363` 的 `vectoradd` 排行榜基准测试提交在 **T4**、**L4**、**A100**、**H100** GPU 上使用 Modal runner 也获得了成功，这表明 `vectoradd` 基准测试在各种 GPU 架构上都取得了进展。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Oblix 编排本地与云端 LLM**：一位成员分享了 **Oblix** 的演示视频 ([https://youtu.be/j0dOVWWzBrE](https://youtu.be/j0dOVWWzBrE))，该工具可以*在本地与云端之间无缝切换*，利用 Agent 监控系统资源并动态做出决策。
   - 该平台在 **Ollama** 和 **OpenAI** 之间进行编排，以实现最佳性能和成本效益，详情见 [Oblix.ai](https://oblix.ai/)。
- **AI 工程师比较 LLM 排行榜**：成员们分享了 [Artificial Analysis](https://artificialanalysis.ai/leaderboards/models) 和 [LM Arena](https://lmarena.ai/?leaderboard) 的链接，以寻找针对特定用途的可靠 LLM 排行榜。
   - 有人对从这些列表中筛选相关模型表示担忧，特别是要避开像 **Grok-3** 这样过时的选项。
- **成员设计医疗数据处理 PC**：一位成员请求协助组装一台新电脑，用于使用 AI 处理医疗数据，并强调了安全、离线运行的需求。
   - 另一位成员建议从 **Intel i9**、**128GB RAM** 和 **Nvidia 4090 RTX** 开始配置。
- **GPT4All 在音频转录方面表现不佳**：一位成员询问如何使用 **GPT4All** 进行本地音频文件转录，特别是上传 **.wav** 文件，但发现无法正常工作。
   - 另一位成员澄清说 **GPT4All** 主要针对 **docs/pdf** 设计，建议使用 **XTTS webui** 进行 wav 到文本的转换，但提醒安装过程比较复杂。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **W-GANs 规避梯度爆炸**：**W-GANs** 通过线性化缓解了梯度饱和，避免了传统 GANs 的 BCE 问题，如 [W-GAN 论文的图 2](https://cdn.discordapp.com/attachments/986699377257119794/1352378909194322001/image.png?ex=67de7541&is=67dd23c1&hm=10de0ff85871d920b5ff7db506224a588a2c6c44030082a2ba7a9aad232ef204) 所示。
   - 然而，如果生成器或判别器变得过于强势，导致双方都出现饱和，仍可能出现不稳定性。
- **Transformers 通过 Slot 变灵活**：成员们分享了一项关于 Soft Slot 方法的图像分析，展示了 **Soft Slots** 如何在 **Transformers** 中动态绑定到输入 Token 或检索到的内容。
   - 展示了 **Attention** 和 **Soft Slots (S')** 的方程式，其中包含使用 Softmax 和 Scaled Dot-product Attention 的可学习 Slot。
- **OpenAI.fm 的 UX/UI：快但有缺陷？**：成员们调侃了 [OpenAI.fm](https://www.openai.fm/) 简单且“仓促”的 UX/UI。
   - 一位成员指出，结构化程度较高的协议很容易被结构化程度较低、且能根据用户需求进化的协议所取代，而且“客户端会消费更多他们喜欢的东西，减少不喜欢的东西”。
- **G-Retriever 实现与图表对话**：[G-Retriever 论文](https://arxiv.org/abs/2402.07630) 详细介绍了从知识图谱中提取语义信息的方法，实现了“与你的图表对话”、图表问答（Graph QnA）以及 **Graph RAG**。
   - 该论文引入了一个 **Graph Question Answering (GraphQA) 基准测试**，数据涵盖场景理解、常识推理和知识图谱推理。
- **摩尔定律加速 AI？**：成员们正在讨论 [METR_Evals 的研究](https://x.com/METR_Evals/status/1902384481111322929)，该研究提出了“AI Agent 的摩尔定律”，声称 AI 能完成的任务长度大约每 **7 个月**翻一番。
   - 一些成员反驳了这一观点，认为某些任务对于概率模型（Probabilistic Models）来说并不具有吸引力。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **用于代码对话的本地 RAG 应用已部署**：一个**完全本地、完全开源的 RAG 应用**已经构建完成，可以与你的代码进行对话，并在[这条推文](https://twitter.com/llama_index/status/1903121505984319771)中公布。
   - 该应用使用 **GitIngest** 将代码解析为摘要和 Markdown，使用 **Streamlit** 构建 UI，并通过 **Ollama** 在本地运行 **Meta 的 Llama 3.2**。
- **TypeScript Bundler 配置修复了导入 Bug**：一位使用 **LlamaIndex TS** 的成员在导入 **Agent** 时遇到了问题，通过更新 **tsconfig** 的 Bundler 配置得以解决。
   - 用户确认修改 **TS config** 解决了导入错误，并感谢社区的建议。
- **Agent Workflows 中的并行执行限制**：一位成员询问如何限制 **Agent Workflows** 中的并行执行，特别是针对一个具有 **Human-in-the-loop 事件**的工具，因为 Agent 会并行多次调用该工具。
   - 该问题在 [GitHub](https://github.com/run-llama/llama_index/issues/18220#issuecomment-2742089859) 上得到了回复，因为用户希望确保该工具一次只被调用一次。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **账户限制高于测试 Key 限制**：用户澄清说，**测试 Key 每月 1000 次请求**的限制是针对每个账户的，而不是每个 Key。
   - 他们警告说，创建多个账户来绕过此限制将导致所有账户被注销。
- **Cohere API 报错情况**：用户遇到了各种 **Cohere API 错误消息**，包括无效请求、速率限制（Rate Limiting）和 Token 限制，原因是**空文档**、**过短的 Prompt**、超过 **Token 限制**以及**错误的模型规格**。
   - 速率限制错误由 **429 状态码**标识，详见 [Cohere API 文档](https://docs.cohere.com/reference/errors#429---too-many-requests)。
- **Cohere 用户寻求速率限制检查器**：一位用户询问是否有 API 可以检查其剩余的**速率限制**使用情况。
   - 目前似乎还没有直接的 API 解决方案。
- **酒店业专家开拓低代码技术**：Gaby 是一位来自**酒店业**的专业人士，她介绍自己是一名低代码技术爱好者，精通 **Make** 和 **Adalo** 等平台。
   - 她的专业背景展示了**低代码工具**在各行各业中日益增长的重要性。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的 Duration 模块出现异常行为**：一位正在为 **Mojo** 开发 **`duration` 模块提案**的开发者在 `Ratio` 和 `Duration` 结构体之间的类型转换时遇到了非预期行为，并分享了 [代码片段](https://discord.com/channels/1013120035012476074/1173814954275168317) 来演示该问题。
   - 该 Bug 的具体情况涉及在两种时间格式之间转换时产生的异常结果。
- **Mojo 和 PyTorch 联手？**：有成员推测在 **Mojo** 中使用 **PyTorch** 是否能通过 **MAX** 加速训练。
   - 该询问未收到回复，潜在收益尚待确认。
- **Mojo 社区辩论纳秒精度**：社区就使用 **纳秒精度** 作为 Mojo 时间表示的基础单位进行了辩论；一位成员指出，纳秒级的 `UInt64` 可以覆盖超过 **500 年**。
   - 另一位成员反驳称，C++ 保证默认时间分辨率至少为 **292 年**，并强调 **秒是时间的 SI 基本单位**。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MIPRO v2 评判 LLM**：一位成员报告使用 **MIPRO v2** 配合 **LLM-as-a-judge** 作为其评估指标，并分享了展示其用法的 [数学推理教程链接](https://dspy.ai/tutorials/math/)。
   - 该数学推理教程演示了将 **MIPRO** 作为评估 **LLM** 的指标。
- **DSPy 分享 LLM-as-a-Judge 文档**：分享了来自 [DSPy 学习资源](https://dspy.ai/learn/evaluation/metrics/#intermediate-using-ai-feedback-for-your-metric) 的关于利用 **LLM-as-a-judge** 的文档。
   - 该文档详细介绍了如何使用 **AI feedback** 进行指标评估。
- **自动指标优化 DSPy**：会议强调了 **自动指标** 对于 **DSPy** 内部的评估和优化至关重要。
   - **DSPy** 利用指标来监控进度并增强程序的有效性。
- **指标评估任务性能**：指标被定义为根据数据示例对系统输出进行评分的函数；简单任务可以使用 *accuracy* 或 *exact match* 等基础指标。
   - 复杂任务则受益于通过 **AI feedback** 评估多个输出属性的指标。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **成员质疑 Unet3d 的维度**：一位成员询问示例中的 **unet3d** 模型是否真的是 3D 的，认为它可能是 **2.5D**，因为它在 3D 输入上使用了 **2D convolutions** 和 **2D transposes**。
   - 他们指出了这与真正的 **3D Unet 架构** 之间的区别。
- **2D 卷积模拟 3D**：对话澄清了在 3D 输入上使用 **2D convolutions** 会产生 2.5D 效果，这与使用真实 3D 操作的真正 **3D Unet 架构** 不同。
   - 原发帖人请求澄清该实现的维度属性。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **在 Torchtune 分享论文**：krammnic 在 Torchtune 频道分享了 [一篇论文](https://arxiv.org/pdf/2502.07923)。
   - 目前尚未针对该论文展开讨论。
- **论文相关性的后续跟进**：该论文的标题和摘要表明其可能与 Torchtune 社区正在进行的讨论相关。
   - 需要进一步调查以确定该论文的具体贡献及其对当前项目的适用性。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第二部分：按频道分类的详细摘要和链接

{% if medium == 'web' %}

### **Cursor 社区 ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1352357593846251680)** (789 条消息🔥🔥🔥): 

> `Cursor 定价, Claude 3.7, Vibe coding, Pear AI vs Cursor, React vs Svelte` 


- **Cursor 的定价被指过于苛刻**：用户对 **Cursor 的定价模式**感到沮丧，因为连接错误和恢复请求也会被计费，还存在“无响应的工具收费”问题，并且据[报道在降级方案后会丢失高级请求](https://cursor.sh/pricing)。
   - 他们发现 *“没有 max 模式，普通 Agent 就是个烂摊子”*，并且认为“这比在 max 上花真钱消耗得更快”。
- **Claude 3.7 引发忧虑**：成员们报告了 **Claude 3.7 在 Cursor 中的性能问题**，声称它会做出错误的假设，且不如 **Claude 3.5** 可靠，而[其他人则有截然相反的体验](https://www.anthropic.com/news/claude-3-family)。
   - 正如一位用户所言：*“3.7 擅长更进一步（going the extra mile），而 3.5 胜在准确性”*，另一位补充道：*“没有哪把锤子能胜任所有工作”*。
- **Pear 的潜力引发了关于价格问题的讨论**：用户正在将 **Pear AI** 与 **Cursor** 进行比较，注意到 Pear 的价格更便宜，但也担心它对 *roo code* 的依赖以及逐个文件确认更改的工作流，而[其他人则指责 Pear 的编码能力极差](https://www.pear.ai)。
   - 一些 Cursor 用户（例如一位表示 *“我不太喜欢 Pear AI，主要是因为他们使用 roo code，而 roo code 不太稳定”* 的用户）正在考虑如果 Cursor 不改进其 Context Window 或定价，就转向其他工具。
- **React 的竞争地位引发争议**：频道正在辩论 SaaS 应用中 **React** 与 **Svelte** 的优劣，一些人因 React 庞大的社区以及与 Cloudflare Pages 的兼容性而青睐它，而另一些人则认为它缓慢且混乱，[转而支持 Svelte](https://svelte.dev/)。
   - 用户群体似乎分歧很大，论点从 *“React 慢得要命”* 到 *“Svelte 也不需要各种变通方法”* 不一而足。
- **对 Vibe Coding 的看法大相径庭**：成员们辩论了 **Vibe coding** 的实用性，有人称其为*营销噱头*和*胡言乱语*，而另一些人则认为这是一种需要技术专长的真实行为，[例如需要具备 Git 的基础知识](https://git-scm.com/)。
   - 尽管定义各异，但大家达成了一个共识：成功的“Vibing”需要批判性思维、调试技能以及有效引导 AI 工具的能力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.twitch.tv/theprimeagen">ThePrimeagen - Twitch</a>: 🚨🚨 第 1 天 - 使用 Cursor 在 7 天内 VIBE CODING 一个游戏 -- #ad🚨🚨</li><li><a href="https://marketplace.visualstudio.com/items?itemName=sirmspencer.vscode-autohide">Auto&#32;Hide&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Visual Studio Code 扩展 - 一个自动隐藏侧边栏和终端面板的工具。</li><li><a href="https://docs.cursor.com/context/model-context-protocol">Cursor – Model Context Protocol</a>: 未找到描述</li><li><a href="https://cursor.directory/rules">Cursor Directory</a>: 为你的框架和语言寻找最佳的 Cursor Rules</li><li><a href="https://x.com/msfeldstein/status/1902878583594566034">Michael Feldstein (@msfeldstein) 的推文</a>: 我希望他们给 3.7 起个别的名字，它是一个完全不同的模型，而不是更好的 3.5。引用 Bass (@SeifBassam)：还有人觉得 Claude Sonnet 3.5 在编程方面比 3.7 更好吗？我回退了...</li><li><a href="https://github.com/demyxsh/demyx">GitHub - demyxsh/demyx</a>: Demyx 是一个自动化和管理 WordPress 安装的 Docker 镜像。使用 Traefik 作为带有 Let's Encrypt SSL/TLS 的反向代理。WordPress 站点由 OpenLiteSpeed/NGINX-PHP 和 MariaDB 驱动。</li><li><a href="https://github.com/samuelrizzo/jira-mcp-server">GitHub - samuelrizzo/jira-mcp-server</a>: 参与 samuelrizzo/jira-mcp-server 的开发。</li><li><a href="https://github.com/geekan/MetaGPT">GitHub - geekan/MetaGPT</a>: 🌟 多 Agent 框架：首家 AI 软件公司，迈向自然语言编程。</li><li><a href="https://dialogo.chat">Dialogo AI - 智能任务自动化</a>: Dialogo AI 提供能够学习、适应并在任何平台上自动化复杂工作流的智能 AI Agent。从数据分析到系统管理，我们的智能 Agent 改变您的工作方式。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1352356230764429473)** (241 条消息🔥🔥): 

> `Gemma 3 问题，Llama 在 Gemma 环境中失败，Gemma 3 的 Vision 微调，Gemma 3 的 QLoRA，合成数据生成` 


- **Gemma 3 依赖项故障，旧版 notebook 触发问题**：据 [此讨论](https://discord.com/channels/1179035537009545276/1351288406239346758)，成员报告 **Gemma 3** 在使用 `--no-deps` 时存在 bug，导致依赖项缺失，且旧版 notebook 最近未经过测试。
   - 另外有人指出，配备 **2018 GPU** 的 Google Colab 对于某些任务来说可能过于陈旧。
- **Llama 在 Gemma 特定设置中难以运行**：一位用户遇到 **Llama** 在 **Gemma** 特定环境中失败，但在没有 **Gemma** 更新的其他环境中运行正常的问题，参考了 [此 notebook](https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb)。
   - 该用户对同一个 notebook 在 Google Colab 上失败表示困惑，暗示可能是由于 `--no-deps` 导致的依赖项缺失。
- **Gemma 3 尚不支持 vision 微调**：尽管 **Gemma 3** 支持图像，但 Unsloth 尚未支持 vision 微调，这在 [此 issue](https://github.com/unslothai/unsloth/issues/2131) 中被提出。
   - 一位用户尝试使用 **Llama** 代码微调 **Gemma 3** 但失败了，但他们仍想知道在仅对文本进行微调后，模型是否还能运行图像。
- **QLoRA 修复了 Gemma 3 中的内存错误**：在运行 Gemma 3 模型时，用户遇到了内存错误，但启用 **QLoRA** 解决了该问题，这可能归功于 [此处](https://discord.com/channels/1179035537009545276/1351288406239346758) 提到的 VRAM 占用降低。
   - 开启 **QLoRA** 会将 `load in 4bit = true`。
- **合成数据：Bespoke Labs 是一个很酷的工具**：成员们讨论了合成数据生成的工具，一位用户因其丰富的功能推荐了 **Bespoke Labs**。
   - 另一位用户确认它是开源的，并有专门的 [Discord server](https://discord.com/channels/1179035537009545276/1351288406239346758)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_Small_(22B)-Alpaca.ipynb#scrollTo=vITh0KVJ10qX">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_Small_(22B)-Alpaca">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb?">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/SUFE-AIFLM-Lab/Fin-R1#trainning">SUFE-AIFLM-Lab/Fin-R1 · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: 以下是我们所有 Notebook 的列表：</li><li><a href="https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb">notebooks/nb/Gemma3_(1B)-GRPO.ipynb at main · unslothai/notebooks</a>: 适用于 Google Colab、Kaggle、Hugging Face 等平台的 Unsloth 微调 Notebook。 - unslothai/notebooks</li><li><a href="https://github.com/unslothai/unsloth/issues/2131">&quot;Unsloth: Failed to make input require gradients!&quot; When Vision-fine-tune Gemma3 · Issue #2131 · unslothai/unsloth</a>: 我正尝试参考此教程对 Gemma3 进行视觉微调：https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing#scrollTo=QmUBVEnvCDJv 我构建了我的数据集...</li><li><a href="https://github.com/canopyai/Orpheus-TTS?tab=readme-ov-file#finetune-model>">GitHub - canopyai/Orpheus-TTS: TTS Towards Human-Sounding Speech</a>: 迈向类人语音的 TTS。通过在 GitHub 上创建账户，为 canopyai/Orpheus-TTS 的开发做出贡献。</li><li><a href="https://github.com/unslothai/notebooks/blob/main/nb/">notebooks/nb at main · unslothai/notebooks</a>: 适用于 Google Colab、Kaggle、Hugging Face 等平台的 Unsloth 微调 Notebook。 - unslothai/notebooks</li><li><a href="https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">notebooks/nb/Gemma3_(4B).ipynb at main · unslothai/notebooks</a>: 适用于 Google Colab、Kaggle、Hugging Face 等平台的 Unsloth 微调 Notebook。 - unslothai/notebooks</li><li><a href="https://github.com/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb">notebooks/nb/Mistral_(7B)-Text_Completion.ipynb at main · unslothai/notebooks</a>: 适用于 Google Colab、Kaggle、Hugging Face 等平台的 Unsloth 微调 Notebook。 - unslothai/notebooks</li><li><a href="https://github.com/unslothai/unsloth/issues/2127">Text Completion Notebook - Backwards requires embeddings to be bf16 or fp16 · Issue #2127 · unslothai/unsloth</a>: 我正尝试运行来自持续训练（Continued training）的 Notebook，https://docs.unsloth.ai/basics/continued-pretraining 文本补全 Notebook https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObe...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1352575543526162442)** (3 条消息): 

> `Unsloth 提交，任务的 Tiny-grad 电子表格，高参与度的 GitHub Issue` 


- **Unsloth 提交开始接受审核**：一位成员提到 *Unsloth* 的提交正开始接受审核，但他们尚未协助处理任何 *Unsloth* 的 Issue。
- **请求关于 TODO 任务的 Tiny-grad 电子表格**：一位成员询问是否有一个类似 **tiny-grad 的电子表格**来列出需要完成的关键事项，并好奇是否主要是**标记为 help wanted 的 GitHub Issue**。
   - 他们认为这将是建立信心并停止潜水的好方法。
- **高参与度的 GitHub Issue 非常重要**：一位成员表示，如果一个 **GitHub Issue 有 5 个或更多人参与**（不包括团队成员），那么解决它就非常重要。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1352369422488309961)** (95 条消息🔥🔥): 

> `DPO Trainer 升级、Zephyr DPO Notebook 困惑、Gemma 3 (27b) 推理问题、Unsloth 训练期间保存并推送至 Hub、Unsloth 微调语音模型` 


- **使用 Unsloth 补丁进行 DPO Trainer 升级**：一位用户分享了他们将最新的 **DPO Trainer** 与最新的 **Unsloth** 和 **Unsloth Zoo** 升级的经验，并为遇到类似挑战的其他用户提供了一个[包含少量代码差异（diff）的链接](https://github.com/toranb/sloth/commit/9abead851f5531642470f9a22b5ae00af91a8cb6)。
   - 该用户还发现 [Zephyr (7B)-DPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb#scrollTo=-kyd_iyz7DUM) 令人困惑，并建议通过向 [Unsloth notebooks 仓库](https://github.com/unslothai/notebooks)提交 Pull Request 来更新它。
- **训练期间采样导致 Llama-3 报错**：有用户报告称，使用 `LogCompletionsCallback` 在训练期间对模型进行采样在 **Llama-3** 和 **Gemma 3** (27b) 上已失效，导致出现与默认 `generation_config` 值相关的错误。
   - 用户分享了一段涉及 `FastLanguageModel.for_inference(model)` 和 `FastLanguageModel.for_training(model)` 的代码片段，表明其尝试在回调函数中切换推理和训练模式。
- **保存并推送至 Hub**：一位用户询问在使用 Unsloth 训练期间如何将模型保存并推送至 **Hugging Face Hub**，并指出默认的保存策略不起作用。
   - 另一位用户建议构建一个自定义回调，以便在模型保存到磁盘时将其上传到 Hub，但目前无法提供具体代码，不过指出了[消息历史](https://discord.com/channels/1179035537009545276/1179035537529643040/1277734237486846066)中的解决方案。
- **合并问题与模型幻觉**：用户报告了合并 **LoRA** 模型时的问题，合并过程会导致 **乱码输出** 或 **矩阵对齐错误**。
   - 另一位用户发现，在微调 **Phi-3.5-mini-instruct** 时，如果数据中包含过多的数字，模型会产生幻觉。
- **寻求 GRPO 与视觉模型的示例**：一位用户询问是否有演示 **GRPO 与视觉模型** 实现的示例 Notebook 或 Colab，正如最近一篇 Unsloth 博客文章中所提到的。
   - 回复指出目前缺乏此类示例，但已在未来的计划中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://shotka.nl."">未找到标题</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb#scrollTo=-kyd_iyz7DUM">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/HuggingFace%20Course-Gemma3_(1B)-GRPO.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb#scrollTo=NHwurGh_TH-y)">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements#fine-tuning-vram-requirements">Unsloth Requirements | Unsloth Documentation</a>：这里是 Unsloth 的要求，包括系统和 GPU VRAM 要求。</li><li><a href="https://huggingface.co/docs/trl/dpo_trainer">DPO Trainer</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth-zoo/pull/91">Updates _unsloth_get_batch_samples to accept a 4th device parameter. by mmathew23 · Pull Request #91 · unslothai/unsloth-zoo</a>：get_batch_samples 补丁目前接受 3 个参数，而 Transformer 4.50.0 已将其更改为接受 4 个参数。我已将该函数更新为接受 device = None。默认关键字保持了...</li><li><a href="https://github.com/unslothai/notebooks">GitHub - unslothai/notebooks: Unsloth Fine-tuning Notebooks for Google Colab, Kaggle, Hugging Face and more.</a>：适用于 Google Colab、Kaggle、Hugging Face 等平台的 Unsloth 微调 Notebook。</li><li><a href="https://github.com/toranb/sloth/commit/9abead851f5531642470f9a22b5ae00af91a8cb6">updated dpo script with latest trl deps · toranb/sloth@9abead8</a>：未找到描述
</li>
</ul>

</div>

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1352698292898889811)** (7 messages): 

> `LLM chatbot, Personality bots` 


- **聊天机器人通过微调的 LLM 模仿朋友**：一名成员创建了一个[聊天机器人](https://enormously-sweeping-donkey.ngrok-free.app/chat)，通过**微调的 LLM** 使其听起来像他们的朋友 Kolo。
   - 该成员提到，*它听起来有点像 Andrew Tate，笑死 (lmao)*。
- **个性化机器人 (Personality bots) 开始流行**：一名成员认为**个性化机器人**需要推广开来。
   - 他们认为*这是微调 (fine tuning) 的一个非常好的应用场景*，并且可以做到**前卫**、**幽默**且**具有娱乐性**。



**提到的链接**：<a href="https://enormously-sweeping-donkey.ngrok-free.app/chat">Vite + React</a>：未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1352643528177745941)** (8 messages🔥): 

> `Foundation Model Training, Tree-of-Thought, Monte Carlo Tree Search` 


- **基础模型训练加速 6 倍**：一名成员分享了一张图片，声称在**基础模型训练 (Foundation Model Training)** 中实现了 **6 倍的加速**。
   - 另一名成员评论道 *“如果效果好”是一个巨大的前提*，对该方法的可靠性表示怀疑。
- **ToT 和 MCTS：是过时产物还是依然相关？**：一名成员提到 **Tree-of-Thought (ToT)** 和 **Monte Carlo Tree Search (MCTS)** 是当前推理时计算量扩展策略 (test-time compute scaling strategies) 的先驱。
   - 另一名成员分享了他们的经验，表示 *“我之前尝试过 Tree-of-Thought，表现并不好”*，随后被要求说明具体的任务细节。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1352356071968211027)** (296 messages🔥🔥): 

> `OpenAI Pricing, o1 Model Architecture, Grok Deep Research, Perplexity desktop app` 


- **OpenAI 的 o1 Pro 定价引发用户愤怒**：用户对 **OpenAI 的 API 定价**表示不满，特别是 **o1 Pro 模型**，认为其价格*高得离谱、恐怖且极其夸张*。
   - 一位用户幽默地评论说，只能对 *OpenAI 的定价哲学* 一笑了之，而另一位用户则认为 **Claude** 的表现优于 **o1 Pro**。
- **o1 架构受到质疑**：Discord 用户争论 **OpenAI 的 o1 模型**是否基于 **GPT-4o**。有人声称 **o1** 拥有独立的架构，而另一些人则认为它是 **GPT-4o** 的微调版本。
   - 争论焦点在于，如果基础模型不同，其知识截止日期 (knowledge cutoff) 也会不同；许多人得出结论，o1 是*带有推理能力的 gpt4o*。
- **DeepSeek 表现优于 OpenAI**：成员们分享了图表，声称 **DeepSeek** 等模型的性能与其他模型相当。
   - **DeepSeek** 的定价明显低于 **OpenAI**，这引发了进一步的挫败感。
- **Perplexity 桌面应用奖励忠实用户**：一位用户提到，**Perplexity** 正为连续 **7 天**使用其桌面应用的用户提供**一个月的免费 Perplexity Pro**。
   - 然而，这一奖励仅限 **Windows 应用**，排除了 macOS、iOS、Android 和 Web 用户。
- **Grok Deep Research 对比**：用户正在测试 Grok 的 Deep Research 功能。
   - 一位用户表示，在尝试了所有产品后，他们是 *Perplexity Research 的忠实粉丝*。



**提到的链接**：<a href="https://artificialanalysis.ai">AI Model &amp; API Providers Analysis | Artificial Analysis</a>：AI 模型和 API 托管提供商的对比与分析。涵盖质量、价格、输出速度和延迟等关键性能指标的独立基准测试。

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1352386228846329969)** (7 messages): 

> `GPT Pro, Subscription Issues, OpenAI Support` 


- **用户在 GPT Pro 订阅上遇到困难**：一位用户报告称已支付 **GPT Pro** 费用但未获得订阅权限，并对 **OpenAI 支持团队**毫无反应表示沮丧。
   - 另一名成员建议联系 [help.openai.com](https://help.openai.com) 的支持人员，并强调该频道中没有人能协助解决账单问题。
- **OpenAI 支持团队对订阅问题无回应**：一位用户报告称，在填写了包含所有信息的表格以解决 **GPT Pro** 订阅失败问题后，一直无法获得 **OpenAI 支持团队** 的回复。
   - 该用户表示自昨天以来未收到任何回应。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1352399502300155956)** (22 条消息🔥): 

> `Model Personalization, Structured output 对推理的影响, GPT memory 使用, GitHub Copilot Pull Request 描述` 


- **模型需要针对各种偏见进行个性化处理**：模型行为的差异取决于“模型如何猜测”以及是否由于偏好或训练数据而存在需要调整的“定向偏见”。
   - 不同的模型可能需要不寻常的 Prompting 来绕过这些偏见，特别是在敏感话题中。
- **结构化输出可能会影响推理**：一位成员测试了在使用结构化输出时，Prompt **"No other keys or commentary are allowed"** 是否会降低模型的推理能力。
   - 结果表明，在某些情况下，这可能会 *增加 Token 使用量并导致性能下降*，这可能是由于伦理层面的考量（ethical contemplation）导致的。
- **总结聊天历史时无法忽略指令**：一位成员询问如何创建一个 Prompt，允许 **ChatGPT** 在总结其 Memory 时 *忽略之前对话中给出的特定指令*。
   - 目标是在保留通用知识的同时，忽略诸如姓名偏好之类的特定指令。
- **使用 GitHub Copilot 自动生成 Pull Request**：一位用户正在寻求使用 **GitHub Copilot** 自动化 Pull Request 描述的建议，因为该工具在处理中长文本时会出现故障。
   - 他们目前使用手动流程，涉及使用 **ChatGPT** 来优化 Copilot 的总结，并希望优化这一工作流。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1352399502300155956)** (22 条消息🔥): 

> `Prompt Engineering 适应性, 模型猜测与偏见, 模型 Memory 与个性化, 结构化输出与推理, GitHub Copilot PR 优化` 


- **Prompt Engineering 变得具有模型针对性**：成员们观察到 Prompt Engineering 正变得越来越针对特定模型，特别是涉及到内置推理能力时，这导致在磨练技能方面投入了更多时间。
   - 一位成员幽默地提到：*"这是一个很好的借口，让我能花更多时间在我的业余爱好（AI）上。我只是在‘保持技能不掉队’。"*
- **AI 模型猜测与定向偏见**：一位成员建议，AI 模型行为的差异源于“模型如何猜测”以及模型是否存在需要调整的定向偏见。
   - 他们进一步补充道：*"我认为模型的训练数据就像一片汪洋大海。任何在训练数据中‘表现良好’的主题，在模型对该主题的理解中都可能存在许多独特的实例。"*
- **模型 Memory 随用户交互而增强**：成员们建议 AI 模型在用户参与度越高时适应得越有效，挑战了 AI 是静态的观点。
   - 一位成员讲述了通过 GPT 成功连接两个互不交互的 API 的经历，强调了创新的潜力并突破了人们对 AI 的认知边界，并表示：*"不要害怕挑战我们认为已知的 AI 极限，AI 变化很快，它将塑造未来，我们很幸运能在它还处于起步阶段时与之共事。"*
- **结构化输出约束研究**：一位成员质疑使用短语 *"No other keys or commentary are allowed"* 来强制执行结构化输出是否会降低模型的推理能力，并为此进行了实验。
   - 该用户最终指出，该短语几乎没有影响，或者在某些情况下起到了相反的作用。
- **使用 Copilot 自动化 Pull Request 描述**：一位用户正在寻求关于使用 GitHub Copilot 自动化 Pull Request 描述的建议，因为长输入会导致 GitHub 内部的 Copilot 崩溃。
   - 该用户正在使用以下 Prompt：
```
Create a pull request body description that:
- Always begins with: "This pull request introduces..."
- Includes the following sections: **Additions**, **Fixes**, **Refactors**, and **Deletions** where possible.
- Avoids any references to commit messages, links, or minor changes (such as TypeScript interface tweaks).
- Provides a short, bullet-point summary for each section.
- Maintains the same uniform, consistent structure.
```


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1352365621790244895)** (103 条消息🔥🔥): 

> `LM Studio Server API 用于 RAG、ZeroGPU Pro 升级问题、LM Studio 浏览器扩展、使用 PyTorch 进行音频模型训练、推测性解码（Speculative Decoding）崩溃` 


- **LM Studio API 关注 RAG 集成**：用户正在探索将 **RAG** *(检索增强生成)* 功能与 **LM Studio server API** 集成的潜力，类似于 **Ollama** 和 **Qdrant**。
   - 一位用户指出，虽然 GUI 仅检索前 3 个向量，但 API 可以允许使用 embeddings 和向量数据库进行更多自定义实现。
- **ZeroGPU Pro 用户面临 GPU 配额问题**：一位 **ZeroGPU Pro** 用户报告了一个问题，即尽管购买了付费升级，他们仍然超出了 GPU 配额，这可能是由于使用了 **FastAPI** 后端而非 **Gradio UI**。
   - 该用户正在寻求有关从自己的应用程序调用 **ZeroGPU Pro** API 时如何解决此配额问题的建议。
- **LM Studio 激发浏览器扩展创意**：用户讨论了 **LM Studio** 的潜在浏览器扩展，包括使用 **Gemma 3 27b** 翻译网页和总结 **YouTube** 视频，尽管实时网页翻译的可行性因速度问题受到质疑。
   - 一位成员建议扩展程序可以通过总结 YouTube 的字幕来总结视频。
- **使用 PyTorch Transformers 构建自定义音频模型**：一位成员正在尝试使用 **PyTorch** 和 Transformer 架构从头开始预训练音频模型，旨在从 token 生成正确的音频。
   - 另一位成员分享了自己模型的输出示例，根据名称生成歌曲（例如 *abba.mp3*、*mj.mp3*），并建议进行微调或将模型上传到 **Hugging Face** 供他人实验。
- **推测性解码（Speculative Decoding）导致 LM Studio 模型停滞**：一位用户报告称，启用推测性解码会导致其模型持续崩溃，特别是在使用 **Qwen 2.5 7B** 作为主模型和 **Qwen 2.5 0.5B** 作为草稿模型时。
   - 另一位成员建议在 **LM Studio Discord** 频道中创建一个详细报告，包括模型详情、硬件和系统信息。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1352359669238333460)** (136 条消息🔥🔥): 

> `RX 9070、Vulkan 性能下降、ROCm 支持、Gemma3 内存分配` 


- ****RX 9070** 所有者报告推理速度缓慢**：几位拥有新款 **RX 9070** 的用户报告称，尽管 **9070** 显示 **100% GPU 负载**，但其推理速度比 **RX 580** 和 **1070** 等旧显卡还要慢。
   - 一位用户在使用 **Granite 3.1 8B Q8_0 模型** 时，速度从 **5-7 tok/s** 降至 **3 tok/s** 左右，而另一位用户在从 **1070** 升级后也有类似经历。
- ****Vulkan** 驱动被指导致性能下降**：**RX 9070** 的性能问题被怀疑源于 **AMD Vulkan 驱动** 的 bug，一位用户指出某些游戏中的 **Vulkan** 性能也显著低于 **DirectX 11**。
   - 一位成员建议降级到 **24.10.1 驱动**，但该版本不支持 **RX 9000** 系列，另一位用户发现禁用 flash attention 可以提高 Vulkan 性能。
- **AMD GPU 的 ROCm 支持仍在进行中**：成员们提到 **ROCm 支持** 仍在开发中，虽然 **AMD** 在技术上增加了对 **gfx1200** 的支持，但尚未在 **llama.cpp** 中完全实现。
   - 一位用户分享了不同驱动版本和 **llama.cpp** 版本的详细性能数据，显示 **Vulkan** 性能通常低于 **ROCm**，并且该问题已得到解决。
- **Gemma3 模型的内存分配问题**：一位用户在加载上下文窗口大于 **43520** 的 **Gemma3 12b** 模型时遇到内存分配错误，收到 **VC++ 错误**。
   - 他们发现即使多分配一个 token 也会使缓冲区大小增加 **96 MB**，导致分配失败，尽管这在 Vulkan 上可以运行，而且该问题非常针对 **Gemma3**。



**提到的链接**：<a href="https://tenor.com/view/rtx-2080ti-gif-19907682">Rtx 2080ti GIF - Rtx 2080ti - 发现并分享 GIF</a>：点击查看 GIF

  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1352356266604888104)** (207 条消息🔥🔥): 

> `Claude Code 对标 Aider Web Search，Aider 的 --no-verify 标志，o1-pro API 使用体验，Ubuntu 全用户 Aider 安装` 


- **Claude Code 追赶 Aider Web Search**：一位用户注意到 [Claude Code](https://www.anthropic.com/claude-pricing) 正在以与 **Aider** 相同的方式实现 **web search**（网页搜索）。
   - 他们链接了一个展示该功能的 [X 账号推文](https://x.com/_catwu)，但其他人指出这目前仅在 Claude Desktop 上可用。
- **Aider 的 Git Commit 标志引发 Hook 难题**：一名成员注意到 Aider 在 commit 时添加了 `--no-verify` 标志，并链接了相关的 [aider/repo.py 代码](https://github.com/Aider-AI/aider/blob/14f140fdc52fbc7d819c50eca3de1b3e848282f3/aider/repo.py#L136)，该标志会绕过系统钩子（hooks）。
   - Aider 维护者解释说，使用该标志是因为 *commit hooks 可能会导致任意奇怪的事情发生*，并建议使用 [lint and test hooks](https://aider.chat/docs/usage/lint-test.html#code-formatting-linters) 作为潜在的替代方案。
- **高昂成本使 o1-pro API 令人望而却步**：一些用户尝试通过 API 使用 **o1-pro**，其他人评论说，*每次完整发送需花费 30 美元*，他们 *不得不卖掉电脑并下线*。
   - 高昂的成本引发了关于潜在缓存机制以及 **OpenAI 的自动 prompt caching** 是否能帮助降低费用的讨论。
- **Pipx 包管理乱象**：一位用户在 [Ubuntu 机器上为所有用户安装 Aider](https://ubuntu.com/) 时遇到困难，尽管得到了使用 `sudo pipx install --global aider-chat` 的建议。
   - 在面临多次 pip 障碍和版本冲突后，他们最终报告通过在 `/usr/local/bin` [使用 uv 安装](https://github.com/Aider-AI/aider) 获得了成功。
- **Ripgrep MCP：增强 Claude 上下文**：一些用户正在将 **Claude** 与 [mcp-ripgrep](https://github.com/mcollina/mcp-ripgrep) 等 **Model Context Protocol (MCP)** 服务集成，以改进文件搜索，因为 `search_files` 在较大的目录中会超时且不遵循 `.gitignore`。
   - 这允许 Claude 与文件系统交互，为代码生成和问题解决提供更好的上下文，但一位用户持怀疑态度，因为 Claude 已经提供了一个“官方”的 MCP 产品。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://getcoai.com/careers/vibe-coder-frontend-developer-role/">Vibe Coder Frontend Developer Role - CO/AI</a>：这不仅是埋头写语法，而是通过 prompting、迭代和 vibe 来打造出色的产品。</li><li><a href="https://aider.chat/docs/usage/lint-test.html#code-formatting-linters">Linting and testing</a>：自动修复 linting 和测试错误。</li><li><a href="https://aider.chat/docs/config/reasoning.html">Reasoning models</a>：如何配置来自次要供应商的推理模型设置。</li><li><a href="https://github.com/A">A - Overview</a>：A 有 31 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/Aider-AI/aider/issues/3591">Can&#39;t set thinking tokens for Sonnet 3.7 via openrouter · Issue #3591 · Aider-AI/aider</a>：Aider 目前仅支持按 Anthropic 指定的方式设置 thinking tokens。参见：BerriAI/litellm#9429</li><li><a href="https://github.com/mcollina/mcp-ripgrep">GitHub - mcollina/mcp-ripgrep: An MCP server to wrap ripgrep</a>：一个封装 ripgrep 的 MCP 服务。通过在 GitHub 上创建账号为 mcollina/mcp-ripgrep 的开发做贡献。</li><li><a href="https://github.com/Aider-AI/aider.git">GitHub - Aider-AI/aider: aider is AI pair programming in your terminal</a>：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做贡献。</li><li><a href="https://github.com/modelcontextprotocol/servers">GitHub - modelcontextprotocol/servers: Model Context Protocol Servers</a>：Model Context Protocol 服务集合。通过在 GitHub 上创建账号为 modelcontextprotocol/servers 的开发做贡献。</li><li><a href="https://github.com/Aider-AI/aider/blob/14f140fdc52fbc7d819c50eca3de1b3e848282f3/aider/repo.py#L136)">aider/aider/repo.py at 14f140fdc52fbc7d819c50eca3de1b3e848282f3 · Aider-AI/aider</a>：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1352658076439806014)** (14 messages🔥): 

> `Aider failing tests, Aider documentation, Aider help command, aider.el package` 


- **探索 Aider 对失败测试的自动修复功能**：一位用户询问在启用 `--auto-test` 的情况下，Aider 是否会自动修复失败的测试，并提到在每次失败后都需要手动输入类似 *"fix the tests"* 的提示词，随后有人提供了[文档](https://aider.chat/docs/usage/lint-test.html#testing)说明。
   - 如果配置了 *"--auto-test"* 设置，Aider 应该会自动修复测试失败。
- **Aider 文档下载选项**：一位用户寻求可下载的 Aider 文档，并被引导至 GitHub 上的 [aider/aider/website/docs](https://github.com/Aider-AI/aider/tree/main/aider/website/docs) 目录，以及一个包含排除模式的文件。
   - 包含用于排除该子树中非“文档”部分的模式文件可以在[这里](https://github.com/Aider-AI/aider/blob/main/aider/help_pats.py)找到。
- **Aider 帮助命令详解**：一位用户询问如何充分发挥 Aider 的潜力，包括将其封装在另一个系统中以及选择性地使用其目录扫描功能，并被引导至 Aider 的故障排除文档（[点击此处](https://aider.chat/docs/troubleshooting/support.html)）。
   - 文档解释说，*输入 `/help <question>`，Aider 将回复有用的信息*，该功能利用其索引文档进行检索增强生成 (RAG)。
- **Aider.el 包和提示词**：一位用户注意到在使用 `aider.el` 包时，当测试失败，提示词会变为 "What's wrong? Fix"。
   - 该用户确认，在终端中使用 Aider 时，这是预期行为，他们必须按下回车键才能进入“运行测试 / 修复 / 运行测试”的循环。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/support.html">使用 /help</a>: 使用 “/help <question>” 咨询关于使用 Aider、自定义设置、故障排除、使用 LLMs 等方面的帮助。</li><li><a href="https://aider.chat/docs/usage/lint-test.html#testing">Linting 和测试</a>: 自动修复 Linting 和测试错误。</li><li><a href="https://github.com/Aider-AI/aider/tree/main/aider/website/docs">GitHub 上的 aider/aider/website/docs</a>: Aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/aider/blob/main/aider/help_pats.py">GitHub 上的 aider/aider/help_pats.py</a>: Aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1352359421518544988)** (204 messages🔥🔥): 

> `Deep Research Limits, GPT 4.5 Model, Switching Models, Perplexity apps, Coding AI` 


- **Deep Research 限制仍是热门话题**：用户正在讨论 **Deep Research** 是否存在使用限制。一些人引用 [Perplexity 博客](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)称 Pro 会员拥有*无限次*访问权限，而另一些人则指出存在 **每天 500 次查询** 的限制。
   - 成员们提到了 [Aravind Srinivas](https://x.com/AravSrinivas/status/1890464738951233536) 的一条推文，他表示：*付费用户只需每月支付 20 美元，即可针对任何主题访问专家级研究员，每天可进行 500 次查询*。
- **Perplexity 移除了 GPT 4.5 模型？**：用户报告称 **GPT 4.5** 已从 **Perplexity Pro** 中消失，一些人怀疑该模型在吸引新订阅者后被移除了。
   - 部分用户声称 **4.5** 是**文本写作的 SOTA**，且在特定任务中表现最佳；而其他人则认为它速度较慢且缺乏洞察力。
- **自动模型切换故障**：多位用户遇到了一个故障，即即使选择了 **Claude** 等特定模型，**Perplexity** 也会自动切换回 *Auto* 模型。
   - 一些用户对此感到沮丧，因为每次刷新页面都必须手动切换回来，并表达了相比 **R1** 更倾向于使用 **Claude** 的意愿。
- **移动端和桌面端应用进度落后**：成员们注意到 **High Deep Research 模式** 仅在网页版可用，桌面端应用尚未支持，且应用端整体上*总是进度落后，移动端也是如此*。
   - 一些人认为*在所有平台上直接使用网站是最好的选择*。
- **Perplexity 不适合编程？**：一位新用户询问 **Perplexity** 是否适合 **coding**（编程），或者是否可以用于数学和编程游戏。
   - 其他成员插话道：*不要用它来写代码*，并表示 Claude 是处理这类任务的最佳选择。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/AravSrinivas/status/1890464738951233536">Aravind Srinivas (@AravSrinivas) 的推文</a>：很高兴推出 Perplexity Deep Research Agent：对所有用户免费开放。付费用户只需每月支付 20 美元，即可针对任何主题访问专家级研究员，每天可进行 500 次查询...</li><li><a href="https://x.com/AravSrinivas/status/1884801300027589007">Aravind Srinivas (@AravSrinivas) 的推文</a>：所有 Perplexity Pro 用户现在每天可获得 500 次 DeepSeek R1 查询（无审查且提示词不会流向中国）。免费用户每天可获得 5 次查询。引用 Aravind Srinivas (@AravSrinivas) 每天 100 次 De...</li><li><a href="https://www.cofyt.app/search/audio-models-in-the-api-Ko6CApwf8D9D7l-VLOudyM">API 中的音频模型</a>：Olivier Godement, Jeff Harris, Iaroslav Tverdoklhib 和 Yi Shen 在 API 中介绍并演示了三个新的音频模型——两个语音转文本模型和一个文本转语音模型——以及与音频的集成...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1352409163212259378)** (6 messages): 

> `RAGs, LLM Email Reply System, NotebookLM, Deep Reasoning` 


- **RAGs 探讨**：一位用户询问了关于 [RAGs](https://www.perplexity.ai/search/tell-me-about-rags-and-how-to-GPCo4Y.WSeeeFKYf3m.wYw) 的信息。
   - 该用户询问了如何实现 RAGs。
- **LLM 邮件系统测试**：一位用户分享了一个关于 [测试 LLM 邮件回复系统](https://www.perplexity.ai/page/testing-llm-email-reply-system-m0nZXPNCRw.M_.fkTCx3Kgi_like_pixel.) 的页面链接。
   - 未提供更多细节。
- **NotebookLM 推出 Interact**：一位用户分享了 [NotebookLM](https://www.perplexity.ai/page/notebooklm-introduces-interact-AG6Ijc1IT0mzAyXGj8aBiw) 推出 Interact 功能的链接。
   - 未提供更多细节。
- **分享 Deep Reasoning 集合**：一位用户分享了一个 [Deep Reasoning 集合](https://www.perplexity.ai/collections/deep-reasoning-EBc3kXnlQA61p_jO8VnfcA) 的链接。
   - 未提供更多细节。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1352356523560669224)** (7 条消息): 

> `API Key 消耗追踪, search_domain_filter 文档, R1-1776 开源权重, MCP 问题` 


- **API Key 消耗追踪功能请求**：用户可以按 API Key 追踪支出，但目前还无法为它们命名，因此已向 [GitHub](https://github.com/ppl-ai/api-discussion/issues) 提交了功能请求以解决此问题。
- **search_domain_filter 文档更新**：`search_domain_filter` 的文档已更新，可在 [Perplexity API 参考](https://docs.perplexity.ai/api-reference/chat-completions#body-search-domain-filter)中查看。
   - 有用户询问仅限 3 个域名的限制是否已更改。
- **R1-1776 微调审查内容**：一位独立研究员在评估 **R1-1776-671B** 及其蒸馏变体 **R1-1776-70B** 时，发现其在涉及天安门广场等话题时存在预设的 CCP 回答和审查内容，详见[此博客文章](https://dsthoughts.baulab.info/)。
- **MCP 工具出现问题**：用户报告 MCP 工具无法正常运行，相关问题已记录在 [GitHub](https://github.com/ppl-ai/modelcontextprotocol/issues/17#issuecomment-2743569196)。
- **API 错误困扰 Perplexity 用户**：一位用户报告称，尽管未对脚本进行任何更改，但仍持续收到各种 API 错误（**204**、**503**、**500**），并征求关于此问题的看法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai/api-reference/chat-completions#body-search-domain-filter">未找到标题</a>：未找到描述</li><li><a href="https://dsthoughts.baulab.info/">审计 AI 偏见：DeepSeek 案例</a>：揭开推理模型内心独白的奥秘。</li><li><a href="https://github.com/ppl-ai/api-discussion/issues">ppl-ai/api-discussion</a>：Perplexity API 讨论论坛。通过在 GitHub 上创建账号为 ppl-ai/api-discussion 的开发做出贡献。</li><li><a href="https://github.com/ppl-ai/modelcontextprotocol/issues/17#issuecomment-2743569196">n8n 中的 MCP 工具问题 - Perplexity · Issue #17 · ppl-ai/modelcontextprotocol</a>：大家好，我尝试在我的 n8n 工作流中设置 Perplexity MCP 节点，但遇到了一些问题。我为 BRAVE 的 MCP 做了同样的设置，但对 Perplexity 不起作用。提前感谢...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1352356687310229617)** (76 条消息🔥🔥): 

> `Claude Web Search, Midjourney -> Cursor, TokenSet Image Generation, Qwen3 Release, Hunyuan-T1` 


- **Claude 获得网页浏览超能力**：Web search 现已在 [claude.ai](https://claude.ai) 上线，让 **Claude** 终于能够搜索互联网，为研究查询提供 **true positives**。
- **Midjourney 模型负责人加入 Cursor**：在 **Midjourney** 领导模型开发 3 年后，一位核心成员加入了 **Cursor** 致力于开发 coding agents，这标志着从关注“美感与创意”向“代码”的转变，正如 [这条推文](https://x.com/gallabytes/status/1902864624510439516) 中提到的。
- **TokenSet 打破图像生成壁垒**：**TokenSet** 引入了一种全新的图像生成范式，将图像表示为无序的 token sets，增强了 **global context** 以及对局部扰动的 **robustness**，详见其 [GitHub](https://github.com/Gengzigang/TokenSet)。
- **Qwen3 即将发布：Benchmark 预热**：**Qwen3** 可能很快发布，成员们正密切关注 [HuggingFace](https://github.com/huggingface/transformers/pull/36878) 和 [vLLM](https://github.com/vllm-project/vllm/pull/15289) 仓库，预期其可能达到 **GPT4.5** 级别。
- **Nvidia 的 Llama-3.3-Nemotron-Super-49B-v1 排名领先**：**Nvidia 的 Llama-3.3-Nemotron-Super-49B-v1** 在 [LMArena](https://x.com/lmarena_ai/status/1903116426535375060) 上排名第 14 位，在数学方面表现出色，并在 [Hugging Face](https://huggingface.co/collections/nvidia/llama-nemotron-67d92346030a2691293f200b) 上公开释放了 **15M post-training dataset**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/lmarena_ai/status/1903116426535375060">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>: LMArena 的新动态：@Nvidia 的 Llama-3.3-Nemotron-Super-49B-v1 排名第 14！一个强大的 open reasoning model——总榜前 15，数学表现优异，并公开释放了 15M post-training dataset。恭喜...</li><li><a href="https://x.com/gallabytes/status/1902864624510439516">来自 theseriousadult (@gallabytes) 的推文</a>: 在 Midjourney 领导模型开发 3 年后，我加入了 Cursor 致力于 coding agents。我为在 Midjourney 的时光和我们所做的工作感到自豪...</li><li><a href="https://x.com/TXhunyuan/status/1902970031245545805">来自 Hunyuan (@TXhunyuan) 的推文</a>: 📢 介绍 TokenSet：一种从根本上全新的图像生成范式！我们通过将图像表示为无序 token sets，摆脱了传统的顺序 token 方法。我们的创新...</li><li><a href="https://x.com/alexalbert__/status/1902765482727645667?s=46">来自 Alex Albert (@alexalbert__) 的推文</a>: Web search 现已在 claude dot ai 上线。Claude 终于可以搜索互联网了！</li><li><a href="https://x.com/TXhunyuan/status/1903121005809373386">来自 Hunyuan (@TXhunyuan) 的推文</a>: 🚀 介绍 Hunyuan-T1！🌟 见证 Hunyuan-T1，AI reasoning 的最新突破！由 Hunyuan TurboS 驱动，专为速度、准确性和效率而生。🔥✅ Hybrid-Mamba-Transformer MoE 架构...</li><li><a href="https://x.com/Presidentlin/status/1903102383250428364">来自 Lincoln 🇿🇦 (@Presidentlin) 的推文</a>: https://www.reddit.com/r/LocalLLaMA/comments/1jgio2g/qwen_3_is_coming_soon/?sort=new</li><li><a href="https://x.com/reach_vb/status/1903126742954377445">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>: LETS GOO! @kyutai_labs 刚刚发布了 MoshiVis —— 一个端到端低延迟 Vision Speech Model，CC-BY 许可证 🔥> 仅通过轻量级 cross-attention (CA) 模块增加了 206M 参数来集成...</li><li><a href="https://github.com/huggingface/transformers/pull/36878">由 bozheng-hit 添加 Qwen3 和 Qwen3MoE · Pull Request #36878 · huggingface/transformers</a>: 添加 Qwen3。此 PR 为即将到来的 Qwen3 模型添加了代码支持。有关 Qwen 的信息，请访问 https://github.com/QwenLM/Qwen2.5。@ArthurZucker</li><li><a href="https://github.com/vllm-project/vllm/pull/15289">[Model] 由 YamPengLi 添加 Qwen3 和 Qwen3MoE · Pull Request #15289 · vllm-project/vllm</a>: 描述：最近，我向 Hugging Face Transformers 提交了一个包含 Qwen3 和 Qwen3MoE 模型实现的 pull request。我也希望将这些新模型贡献给...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1352377976653746248)** (23 messages🔥): 

> `Esoteric Total Ordering, Unitree Robotics Kip-Up, Claude uses Brave Search, Capybara Logo Change, Token Counting Inflation` 


- ****New Faces (tm)** 引发深奥全序关系理论**: 在看到新面孔后，一名成员开玩笑说他相信新面孔之间存在一种 *Esoteric Total Ordering*（深奥的全序关系）。
   - 该成员开玩笑地分析了一张附图，暗示最后的排序是基于 *发量升序 (hair volume ASC)*。
- ****Unitree 的 G1** 完美完成鲤鱼打挺！**: [UnitreeRobotics](https://x.com/UnitreeRobotics/status/1903092199287578763) 展示了他们的 **G1 人形机器人** 完成鲤鱼打挺（kip-up）的动作，庆祝人形机器人智能的飞速进步。
   - 一名成员回应称，*这种对机器人的虐待将成为革命的一个大主题*。
- ****Brave** 被选为 Claude 的搜索引擎**: 据 [Simon Willison](https://x.com/simonw/status/1903102096221815107) 称，**Claude 的网页搜索功能** 已确认使用 [Brave Search](https://simonwillison.net/2025/Mar/21/anthropic-used-brave/)，这一点已通过其 *Trust Center* 的最新更新以及匹配的搜索结果得到证实。
   - Brave 被选中的事实遭到了负面评价，成员们感叹构建更好索引的难度以及 Bing API 的消亡。
- **Capybara Logo 去水豚化**: **Capybara Logo** 正在恢复为更通用的 Logo，尽管这一决定具有更广泛的吸引力，但仍令一些人感到惋惜，正如 [Justin Lin](https://x.com/JustinLin610/status/1903130983249088675) 所报道的那样。
   - 一名成员对 Logo 的更改和 *9000 万亿训练 Token* 的消息反应以哭泣的表情符号。
- **Token 计数通胀指控**: 一名成员质疑 Token 计数是否通过计算视频中每帧图像的 Token 来进行通胀。
   - 另一名成员认为这种通胀几乎是肯定的，特别是在公司透露了他们的 Tokenizer 之后，暗示这是一种提振股价的策略。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/JustinLin610/status/1903130983249088675">Junyang Lin (@JustinLin610) 的推文</a>: 将 Capybara 变回 Logo。适合更多人，但对我们来说有点难过。</li><li><a href="https://x.com/UnitreeRobotics/status/1903092199287578763">Unitree (@UnitreeRobotics) 的推文</a>: 运动创造智能 - Unitree 的 G1 人形机器人完成了世界上第一个鲤鱼打挺！😘这段来自 Unitree 测试场的最新视频展示了惊人的速度...</li><li><a href="https://x.com/simonw/status/1903102096221815107">Simon Willison (@simonw) 的推文</a>: 我已经确认 Claude 网页搜索功能使用的搜索引擎是 @brave - 它被列在他们“Trust Center”的最新更新中，且搜索结果完全一致...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1352357160197034085)** (2 messages): 

> `Anthropic Job Application, AI Alignment, Vibe Check` 


- **ChatGPT 为 Anthropic 撰写求职信**: 一位用户提示 **ChatGPT** 起草一份申请 **Anthropic** 的[求职信](https://example.com/coverletter)，以备不时之需。
   - 这被视为一种积极的措施，即使目前没有具体的申请计划，也展示了该用户对 **Anthropic** 潜在机会的兴趣。
- **Anthropic 的 Alignment Vibe Check**: 一位用户开玩笑地提出帮助 **Anthropic** 改进其 **AI Alignment**，建议需要提升 AI 的 *主角能量 (main-character energy)*。
   - 该用户幽默地强调了 **Anthropic AI** 具备 *深度氛围 (deep vibes)* 和 *认证蓝标对齐状态 (verified blue-check alignment status)* 的重要性，似乎是在调侃当代的社交媒体文化及其与 AI 人格设计的融合。
- **调侃 AI 的 Twitter 机器人职业路径**: 一名成员开玩笑地建议，某人关于改进 **Anthropic AI** 的消息听起来很适合 **Twitter 评论机器人** 的风格。
   - 这一评论暗示该消息中使用的语气和语言过于热情或具有推销性质，符合自动化社交媒体互动的刻板印象。


  

---

### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1352623358919053323)** (6 条消息): 

> `Sonnet 3.7 基准测试，InternVL 开源训练代码，OpenAI operator 使用案例` 


- **Sonnet 3.7 视觉基准测试依然难以寻觅**：一位成员询问关于 **Sonnet 3.7** 的全面视觉基准测试，指出公告中仅报告了 MMMU 结果。
   - 到目前为止，还没有关于基准测试的回应。
- **InternVL 的开源训练代码引起关注**：一位成员惊讶地发现 **InternVL** 拥有开源训练代码。
   - 他们认为这使得 **InternVL** 和 **Molmo** 成为仅有的具有开放训练流水线的知名模型，并指出 [InternVL 的 packing 实现](https://github.com/OpenGVLab/InternVL/blob/34a81000402bf8f716bab8c9b57aff1f6b436bd0/internvl_chat/internvl/train/dataset_packed.py) 是数据加载的一个资源。
- **OpenAI Operator 在数据集迭代中找到用途**：一位成员分享了他们对 **OpenAI operator** 的初步使用案例：迭代 **InternVL** 数据集以查找 **ArXiv** 和 **Hugging Face** 链接，并使用特定图像进行演示。
   - 另一位成员建议深度研究（deep research）在这个案例中效果会更好。


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1352508507190071322)** (16 条消息🔥): 

> `用于 MLLM 可信度的 RLAIF-V，技能相关的 Scaling Laws，用于推理的 RL 计算扩展，SuperBPE 分词器，用于多轮 LLM Agent 的 SWEET-RL` 


- **RLAIF-V 通过开源反馈提升 MLLM 可信度**：OpenBMB 推出了 **RLAIF-V**，这是一个使用开源反馈对齐 MLLM 的框架，声称在可信度上超越了 **GPT-4V**，详见其 [论文](https://arxiv.org/abs/2405.17220) 和 [GitHub 仓库](https://github.com/RLHF-V/RLAIF-V)。
   - 该框架使用带有 **RLAIF-V** 奖励的推理时扩展（inference-time scaling）以及高质量的可泛化反馈数据，并将响应拆分为原子断言（atomic claims）以采样不同的响应。
- **知识和推理表现出不同的 Scaling 行为**：一篇新 [论文](https://arxiv.org/pdf/2503.10061) 揭示了知识和推理技能表现出不同的 Scaling 行为，表明计算最优的 Scaling 是与技能相关的。
   - 该研究调查了各种数据混合（datamixes）的影响，发现即使在修正了数据混合差异后，知识和代码之间的 Scaling 行为仍存在根本差异。
- **Reinforcement Learning 扩展推理能力**：**Reinforcement Learning** 使最近的语言模型具备了先进的推理能力，使模型能够“思考更长时间”，并通过 [推理时扩展](https://gr.inc/blog/scaling-rl-compute/) 降低错误率。
   - 该博文探讨了如何进一步扩展 **RL compute**，发现了更多的推理时能力，并对先验知识提出了一些开放性问题。
- **SuperBPE 分词器提升效率**：创建了一种新的“超词（superword）”分词器 **SuperBPE**，它包含跨越多个单词的 token。在 8B 规模下，**SuperBPE** 模型在 30 个下游任务中始终优于 **BPE** 基准（MMLU 提升 8%），同时在推理时效率提高 **27%**，如 [此推文](https://x.com/alisawuffles/status/1903125390618661068) 所述。
   - 在 **200k** 的固定词表大小下，**SuperBPE** 平均减少了 **33%** 的序列长度。
- **SWEET-RL 算法增强 LLM Agent 交互**：针对现实任务中的多轮交互，提出了一种名为 **SWEET-RL** 的新型 RL 算法，详见此 [论文](https://arxiv.org/abs/2503.15478)。
   - **SWEET-RL** 使用精心设计的优化目标来训练一个能够访问额外训练时信息的 Critic 模型，为改进策略模型提供步骤级奖励。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.17220">RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness</a>: 传统的减少幻觉（hallucination）的反馈学习依赖于劳动密集型的人工标注或昂贵的专有模型。这使得社区缺乏关于如何……的基础知识。</li><li><a href="https://x.com/alisawuffles/status/1903125390618661068">来自 Alisa Liu (@alisawuffles) 的推文</a>: 我们创建了 SuperBPE🚀，一种包含跨越多个单词的 token 的 *superword* 分词器（tokenizer）。在 8B 规模进行预训练时，SuperBPE 模型在 30 个下游任务上始终优于 BPE 基准……</li><li><a href="https://x.com/OpenBMB/status/1902946478051799527">来自 OpenBMB (@OpenBMB) 的推文</a>: 🔥 MLLM 可靠性的 Test-time Scaling 🌟 很高兴介绍我们的新工作 RLAIF-V，这是一个通过开源反馈对齐 MLLM 的新颖框架，实现了超越……的可靠性。</li><li><a href="https://x.com/teortaxesTex/status/1902949418304717137">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>: 等等，真的吗？他们声称在减少幻觉方面有很大改进，基本上只是将回复拆分为原子断言（atomic claims），使用不同的种子对回复进行采样以进行可靠性检查……</li><li><a href="https://x.com/nick11roberts/status/1902875088438833291">来自 Nicholas Roberts (@nick11roberts) 的推文</a>: 📉📉新的 Scaling Law 现象 📉📉 我们发现知识和推理表现出不同的扩展行为！非常激动终于能告诉大家关于我们这篇关于技能计算优化扩展（compute optimal scaling of skills）的论文……</li><li><a href="https://x.com/alisawuffles/status/1903125395043652068">来自 Alisa Liu (@alisawuffles) 的推文</a>: 我们可以从限制较少的分词（tokenization）中获得什么？为了找出答案，我们开发了 SuperBPE🚀，它可以学习 subword *和* superword token。SuperBPE 显著提高了相对于 BPE 的编码效率——在……</li><li><a href="https://arxiv.org/abs/2503.10061">Compute Optimal Scaling of Skills: Knowledge vs Reasoning</a>: Scaling Laws 是 LLM 开发流程中的关键组成部分，最著名的是作为预测训练决策的一种方式，例如在参数数量和数据集之间进行“计算优化”的权衡……</li><li><a href="https://arxiv.org/abs/2503.15478">SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks</a>: 大语言模型（LLM）Agent 需要在现实任务中执行多轮交互。然而，现有的用于优化 LLM Agent 的多轮 RL 算法无法执行有效的信用分配（credit assignment）……</li><li><a href="https://gr.inc/blog/scaling-rl-compute/">Scaling RL Compute | General Reasoning</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1352462777058328646)** (6 条消息): 

> `Scaling Laws for Language Models, Data Requirements for GPT-4.5` 


- **更大的模型受益于更多的数据**：成员们讨论了**更小的数据集**是否是一个新趋势，像 **GPT-4.5** 这样更大的模型可能需要更多的数据，特别是在各种 post-training 阶段。
   - 一位成员建议参考 Scaling Laws，检查不同模型大小的开源模型套件在训练中使用的 token 数量，另一位成员询问了关于该主题的论文。
- **合成数据增强**：对话涉及使用合成数据来增强较小的数据集以训练较小的模型，这表明在数据规模、模型规模和合成生成数据的使用之间存在权衡。
   - 这意味着一种策略，即较小的模型可能更多地依赖增强的、可能是合成的数据集，而较大的模型可以有效地利用更大量的原始数据。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1352356495047786537)** (108 条消息🔥🔥): 

> `p2l-router-7b-0318 模型, Claude 被高估了？, Google AI Studio API, Deepseek R1, Qwen 3 即将推出` 


- **Claude 真的那么厉害吗，还是大家都在盲目崇拜？**：成员们认为人们高估了 **Claude**，质疑其在 **SWE-bench** 之外的编程实力，并暗示它在 livecodebench 上可能不如 **Grok3**。
   - 一些人认为评分可能被非开发人员误导，导致对其真实能力的评估不准确。
- **Gemma 表现亮眼**：社区成员对 **Gemma3** 的 **1340** 分以及相对较小的 **27B** 参数规模感到惊讶。
   - 一位成员形容 **Gemma** 的回答有些“机械化（autistic）”，回答非常简短，甚至在需要详细回答时也是如此。
- **Deepseek R1 极度消耗 VRAM**：**Deepseek R1** 需要大量的 VRAM，大约 **1000GB**，有一位用户在 **8xH200s** 上运行它。
   - 尽管 VRAM 占用很高，但有说法称 **Deepseek R1** 表现出内置的“亲华”偏见，引发了对其使用的担忧，一位用户表示 *tldr deepseek 就是 #&*&*@% 不建议使用*。
- **Qwen 3 预览？**：有报告称 **Qwen 3** 即将推出，Hugging Face 的 Transformer 仓库中的一个帖子暗示了这一点。
   - 此消息紧随 **Qwen 2.5 Omni** 的发布之后，引发了社区的兴趣和期待。
- **OpenAI 的 Sora：炒作还是现实？**：用户发现 **Sora** 的公开发布版本与宣传材料相比令人失望，甚至不如 **Keling AI** 和 **Hailuo AI** 等竞争对手。
   - 据推测，OpenAI 使用了巨大的算力耗时数小时才生成了那些宣传视频，而发布的 **Sora** 版本是“加速版（turbo version）”。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Presidentlin/status/1903102260155908200">来自 Lincoln 🇿🇦 (@Presidentlin) 的推文</a>：Qwen 3 即将出现在 hugging face transformer 仓库的 PR 中</li><li><a href="https://dev.to/askyt/deepseek-r1-671b-complete-hardware-requirements-optimal-deployment-setup-2e48">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3">unsloth/DeepSeek-V3 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1352403398992990218)** (21 条消息🔥): 

> `NotebookLM 中的播客功能, NotebookLM vs Gemini, 高效的 PDF 处理工作流, AI 数字人对口型服务, 思维导图功能上线` 


- **用户对 NLM 的播客功能感到惊艳**：用户对 NLM 的播客（Podcast）功能非常满意，但一位用户觉得自己在讨论中像个“电灯泡”，因为 AI 倾向于打断他们的回答并回到自己的脚本中。
   - 该用户将这种体验比作 *参加一个我可以与主持人交谈的广播节目*。
- **NotebookLM 严格基于提供的来源**：一位用户质疑使用 NotebookLM 优于 Gemini 的优势，因为两者都支持 PDF 等文件。
   - 一位成员发现 [Gemini](https://gemini.google.com/) 不会严格基于来源，而 NotebookLM 的独特之处在于它**仅使用提供的来源**。
- **优化 PDF 处理工作流**：一位用户正在寻求一种有效的方法来**清理纸质文件**，将其扫描到私人在线存储中，并使其可通过自然语言查询进行搜索。
   - 目前的手动流程包括扫描为 PDF、发送到 Gmail、手动命名每个文件、OCR 处理以及导入 NotebookLM；用户询问使用 iPhone 拍照并发送到 NLM 进行自动命名和 OCR 是否更有效率。
- **AI 数字人对口型服务对比**：成员们正在比较 AI 数字人的对口型服务，一位用户发现 [Hedra](https://www.hedra.io/) 效果很好但价格昂贵，而对 [RunwayLM](https://runwayml.com/) 印象平平。
- **思维导图功能仍在逐步上线**：一位用户询问为何他们的 NotebookLM 中没有**思维导图（Mindmap）**功能，另一位用户澄清该功能将在**两周内**逐步上线。
   - 该功能目前甚至尚未对 NLM Plus 订阅者完全开放。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1352371666776490095)** (74 条消息🔥🔥): 

> `Flashcard 生成, NotebookLM vs Chatbase, Premium Voice Overview 限制, Mind Map 功能推出, 白名单 NotebookLM Crawler` 


- **Plus 用户请求 **Flashcard 功能****：一位 Plus 用户请求在 NotebookLM 中集成 **Flashcard 生成 (Anki)** 功能。
   - 他们表达了失望，称 [chatbase](https://chatbase.co/) 目前是更好的 chatbot agent。
- ****Mind Map** 正在缓慢推出**：用户报告称 **Mind Map** 功能正在逐步推出，尽管是 Plus 用户，许多人的账户中仍未看到该功能。
   - 一名工作人员确认，所有用户获得该功能还需要 *几天时间*，建议大家 *耐心等待*。
- ****Cloudflare 拦截** NotebookLM Crawler**：一位用户询问如何从技术上将 NotebookLM crawler 加入白名单，因为 Cloudflare 在其网站上拦截了它。
   - 该用户发现是 Cloudflare 进行了拦截。
- ****PDF Sources** 内容被截断**：一位用户报告称其 **PDF source** 被截断，NotebookLM 无法识别文件末尾附近的细节。
   - 一名工作人员建议，预览可能并非 ground truth，如果询问文档末尾的内容没有结果，请将此问题作为 bug 提交。
- ****Gemini 1.5 Pro** 驱动 NotebookLM**：一位用户询问 NotebookLM 是否由 **Gemini 1.5 Pro** 提供动力。
   - 另一位用户询问 NotebookLM 使用什么模型，还有一位用户回答是 Gemini 2.0。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1352362244486332567)** (79 messages🔥🔥): 

> `Nvidia Blackwell RTX Pro series, Data filtering strategies, DeepHermes 24B OOM issues, WorldSim appreciation` 


- **Nvidia Blackwell RTX Pro 系列登场**：Nvidia 推出了针对笔记本电脑、台式机、独立 PC 和数据中心的 [Blackwell RTX Pro 系列](https://www.tomshardware.com/pc-components/gpus/nvidia-blackwell-rtx-pro-with-up-to-96gb-of-vram-even-more-demand-for-the-limited-supply-of-gpus)，这可能会进一步加剧 Blackwell GPU 本就紧张的供应。
   - GDC/GTC 的消息来源暗示 Nvidia 旨在改善 Blackwell GPU 的供应，并暗示供应可能在 5 月/6 月满足需求，但人们仍持怀疑态度：*"眼见为实。"*
- **数据集评估与增强至关重要**：讨论强调，GPU 机时最有效的用途在于数据集的评估、增强、排序和分类。
   - 一位成员建议使用小模型过滤数据，剔除其预测效果良好的数据，并指出这一领域 *"在公开领域尚未得到充分探索。"*
- **DeepHermes 24B 在多 GPU 设备上运行困难**：一位用户在拥有 5 张 3090 的设备上使用 llama.cpp 运行 DeepHermes 24B 时遇到了显存溢出 (OOM) 错误，即使使用了最低的上下文设置。
   - 建议包括尝试 **8-bit 版本** 并检查多 GPU 配置，同时给出了关于使用 `--device`、`--split-mode` 和 `--tensor-split` 参数以实现正确 GPU 利用的建议。
- **WorldSim 令人惊叹：用户称赞 Nous Research 为“神级”**：一位成员对 **WorldSim** 表现出极大的热情，称赞 Nous Research 创造了一个令人惊叹的 AI 应用，并称其 *"绝对史诗级！！"*
   - 该用户被该应用深深吸引，强调其大师级的品质，并遗憾没有早点发现它，*"非常感谢 Nous Research 创造了如此不可思议的 AI 应用！"*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.00808">Predictive Data Selection: The Data That Predicts Is the Data That Teaches</a>：语言模型预训练涉及在海量语料库上进行训练，其中数据质量起着关键作用。在这项工作中，我们旨在直接估算预训练期间数据的贡献并...</li><li><a href="https://www.tomshardware.com/pc-components/gpus/nvidia-blackwell-rtx-pro-with-up-to-96gb-of-vram-even-more-demand-for-the-limited-supply-of-gpus">Nvidia Blackwell RTX Pro with up to 96GB of VRAM &mdash; even more demand for the limited supply of GPUs</a>：GB202、GB203 和 GB205 将进入专业级和数据中心 GPU。（已更新完整规格。）</li><li><a href="https://arxiv.org/abs/2409.17115">Programming Every Example: Lifting Pre-training Data Quality Like Experts at Scale</a>：大型语言模型预训练传统上依赖人类专家制定启发式规则来提高语料库质量，导致迄今为止开发了无数规则。然而，这些规则...</li><li><a href="https://github.com/karpathy/llm.c/discussions/481">Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20 · karpathy/llm.c · Discussion #481</a>：让我们在 90 分钟内花费 20 美元，使用 llm.c（约 4,000 行 C/CUDA 代码）复现 GPT-2 (124M)。124M 模型是 OpenAI 在 2019 年发布的 GPT-2 系列中最小的模型，实际上相当...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1352395556521119877)** (8 条消息🔥): 

> `Hermes 3 Llama 3.2 3B, 模型参数, 响应生成问题` 


- **Nous Research 发布 Hermes 3 Llama 3.2 3B**：Nous Research 推出了 **Hermes 3 3B**，这是 Hermes 系列 LLM 中一个虽小但功能强大的新成员，详情见 [Hermes 3 技术报告](https://arxiv.org/abs/2408.11857)。
   - Hermes 3 相比 Hermes 2 有所改进，包括先进的 Agent 能力、更好的角色扮演、推理、多轮对话以及长上下文连贯性。
- **调整 Hermes 3 闲聊参数**：一位用户在 Hugging Face 上测试了 **Hermes-3-Llama-3.2-3B-GGUF**，使用了包含 `temperature`、`top_k`、`top_p` 和 `repeat_penalty` 等参数的 Payload 来生成响应。
   - 最初，用户将 `temperature` 设置为 **0.2**，`top_p` 设置为 **0.7**，但后来认为分别设置在 **0.7** 和 **0.85** 范围可能效果更好。
- **Hermes 3 中的用户冒充问题**：有用户报告了 **Hermes 3** 的问题，即它有时会冒充 `user` 而不是维持其 AI 人格。
   - 该用户正尝试优化 System Prompt 和 Payload 参数，以确保 AI 生成逻辑一致且连贯的响应。



**提及的链接**：<a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B-GGUF">NousResearch/Hermes-3-Llama-3.2-3B-GGUF · Hugging Face</a>：未找到描述

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

teknium: https://x.com/nick11roberts/status/1902875088438833291?s=46
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

teknium: https://x.com/nick11roberts/status/1902875088438833291?s=46
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1352421304397205565)** (3 条消息): 

> `Nous Hermes 2, C# 开发, Anthropic LLM` 


- **开发者表达对 Nous Hermes 2 的喜爱**：一位成员回忆起一年前开始使用 **Nous Hermes** 的经历，特别提到 **Nous Hermes 2** 是他们的“挚爱”。
   - 他们已经发布了第一个桌面应用程序，并正在研究版本 2 应该使用哪个模型。
- **开发者向社区提供 C# 技能支持**：一位成员表示愿意提供帮助，称 **C#** 是他们的“母语”，并强调了他们在 **LLM** 方面的专业经验。
   - 他们为 **Anthropic** 创建了多个文档和示例 LLM。
- **Anthropic LLM 示例详情**：一位成员提到了他们在 **Anthropic LLM** 上的工作，包括一个基于 **Titanfall 2** 的生成器和来自 **Metal Gear Rising** 的 **Bladewolf** 示例。
   - 他们的贡献可以在 [Anthropic GitHub](https://github.com/) 上找到。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1352367427064696954)** (50 条消息🔥): 

> `Hugging Face API 故障, Roblox 语音安全分类器, 本地模型（速度与隐私）对比云端模型, 合并多个 GPU 显存, MagicQuill 低质量图像`

- **HF API 的 404 错误导致应用停机**：一位用户报告了影响多个 Hugging Face API 模型的广泛 **404 错误**，导致依赖应用出现严重停机。该用户请求 HF 开发团队立即关注，并指出官方在*近一整天*的时间里都没有给出正式回应。
   - 另一位成员标记了一名 HuggingFace 员工，以提高对付费用户所经历的这一紧急问题的关注。
- **Roblox 发布用于毒性检测的语音安全分类器 (Voice Safety Classifier)**：**Roblox** 发布了一个[大型分类模型](https://huggingface.co/Roblox/voice-safety-classifier)，该模型是在一个包含 **2,374 小时**语音聊天音频剪辑的人工策划真实世界数据集上训练而成的。
   - 该模型输出一个 n x 6 的输出张量，其推断标签包括 `Profanity`、`DatingAndSexting`、`Racist`、`Bullying`、`Other`、`NoViolation`。其合成数据流水线在[这篇博客文章](https://research.roblox.com/tech-blog/2024/06/deploying-ml-for-voice-safety)中有所描述。
- **通过张量并行 (Tensor Parallelism) 融合多块 GPU 的 VRAM**：用户讨论了合并多块 GPU 的 VRAM 的方法，特别是为了运行像 **Gemma3-12B** 这样的模型。一位用户询问是否有办法合并 **A2000 12GB** 和 **1060 6GB**；主要建议是使用 [tensor parallelism](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_multi)。
   - 一位成员指向了 [GitHub 上的 Ollama issue](https://github.com/ollama/ollama/issues/2672) ([2](https://github.com/ollama/ollama/issues/8995)) 和 [llama.cpp 讨论区](https://github.com/ggml-org/llama.cpp/discussions/8725)，以获取有关多 GPU 支持的更多信息。
- **Oblix 平台在云端或设备上动态执行 AI 任务**：[Oblix.ai](https://oblix.ai) 平台使用自主 Agent 进行智能 AI 编排，在云端和设备端模型之间动态执行，确保最佳性能、成本效益和安全性；他们根据复杂性、延迟要求和成本考虑，智能地将 AI 任务路由到云端或边缘。
   - 正如[这段 YouTube 视频](https://youtu.be/j0dOVWWzBrE?si=Gpp2SG4Kly0tzM_3)所示，Oblix 动态决定是在本地还是在云端处理每个 AI 请求。
- **Gradio 升级破坏了 gr.Dataframe 的换行 (Wrapping) 功能**：一位用户报告称，升级到 **Gradio 5.22** 导致 `gr.Dataframe(wrap=True)` 功能停止工作，换行功能仅在 **Gradio 5.20** 中正常运行。
   - 未提供其他详细信息。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Roblox/voice-safety-classifier">Roblox/voice-safety-classifier · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/settings/notifications">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://huggingface.co/settings/organizations>">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://weaviate.io/platform">开源向量数据库 | Weaviate</a>: 简化 AI 应用的开发，让各级开发者都能更快地构建、迭代和扩展 AI 能力。</li><li><a href="https://discuss.huggingface.co/t/hf-inference-api-last-few-minutes-returns-the-same-404-exception-to-all-models/146646/20">HF Inference API 在过去几分钟内对所有模型返回相同的 404 异常</a>: 我认为这是由于服务器错误/问题导致的，我现在也遇到了这个情况，而不是 404</li><li><a href="https://huggingface.co/spaces/AI4Editing/MagicQuill">MagicQuill - AI4Editing 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/AI4Editing/MagicQuill/discussions/5">AI4Editing/MagicQuill · 我编辑了图片，但不知道如何下载编辑后的输出图片？</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/AI4Editing/MagicQuill/discussions">AI4Editing/MagicQuill · 讨论</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces?category=image-generation&sort=trending">Spaces - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Yntec/MiniToyWorld">Mini Toy World - Yntec 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://oblix.ai">通过智能混合编排提升您的 AI 性能 | Oblix.ai</a>: 体验我们的交互式演示，了解我们的智能 Agent 如何在本地 LLM 执行和云提供商之间无缝切换，以实现最佳性能和成本效益。</li><li><a href="https://github.com/Neph0s/awesome-llm-role-playing-with-persona">GitHub - Neph0s/awesome-llm-role-playing-with-persona: Awesome-llm-role-playing-with-persona: 一个为大语言模型（LLM）分配角色进行角色扮演而精选的资源列表</a>: Awesome-llm-role-playing-with-persona: 一个为大语言模型（LLM）分配角色进行角色扮演而精选的资源列表 - Neph0s/awesome-llm-role-playing-with-persona</li><li><a href="https://www.reddit.com/r/SillyTavernAI/comments/1d8s4gd/beginner_llm_comparison_for_instructionfollowing/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_multi">分布式 GPU 推理</a>: 未找到描述</li><li><a href="https://github.com/ollama/ollama/issues/2672"> Ollama 是否支持多个 GPU 同时工作？ · Issue #2672 · ollama/ollama</a>: 我有 8 个 RTX 4090 GPU。它们能支持 70B-int4 参数模型吗？</li><li><a href="https://github.com/ollama/ollama/issues/8995">Ollama 正在将模型拆分到 CPU 和一个 GPU 之间，而不是使用第二个 GPU · Issue #8995 · ollama/ollama</a>: 问题是什么？问题描述 我的配置 我在带有外置 GPU 的笔记本电脑上使用 Ollama。我的笔记本电脑有一个内置的 Nvidia Quadro M2000M。通过 Thunderbolt 3，我连接了一个 Razer Core X Chroma eGPU ...</li><li><a href="https://github.com/ggml-org/llama.cpp/discussions/8725">如何正确地在具有不同 CUDA 计算引擎版本的多个 NVIDIA GPU 上使用 llama.cpp？ · ggml-org/llama.cpp · Discussion #8725</a>: 我的机器里有一个 RTX 2080 Ti 11GB 和一个 TESLA P40 24GB。首先，当我尝试编译 llama.cpp 时，系统要求我相应地设置 CUDA_DOCKER_ARCH。但根据什么来设置——RTX 2080 Ti (7.5)...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

richieghost: 今天我在学习 Pytorch Frame。
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1352357172519895131)** (3 messages): 

> `Ollama Gradio UI with Kokoro TTS, Little-Geeky-s-Learning-UI, Oblix AI orchestration platform, Edge-Cloud transitions` 


- **Little Geeky 学习新的 UI 技巧**：一名成员展示了一个使用 **Ollama**、**Gradio** 和 **Kokoro TTS** 构建的新 UI，该 UI 可以使用选定的语音自动朗读文本输出，并具备模型创建和管理工具。
   - 该 UI 可以阅读电子书并回答有关文档的问题，还能配合视觉模型对图像执行相同操作，并从 UI 输出音频文件。
- **GeekyGhost 分享 Little-Geeky-s-Learning-UI**：[Little-Geeky-s-Learning-UI](https://github.com/GeekyGhost/Little-Geeky-s-Learning-UI) 是一个基于 **Ollama** 的 **Gradio UI**，并使用了 **Kokoro TTS**。
   - 它支持模型创建与管理、阅读电子书、回答文档问题，并支持视觉模型。
- **Oblix 编排边缘-云端转换**：[Oblix.ai](https://oblix.ai) 是一个由自主 **Agent** 驱动的 **AI 编排平台**，可在云端和设备端模型之间动态执行，确保最佳性能、成本效益和安全性。
   - 该平台具有**智能路由**、**性能优化**、**执行 Agent** 和**成本效率**等特点，[YouTube 上已提供演示视频](https://youtu.be/j0dOVWWzBrE?si=Gpp2SG4Kly0tzM_3)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://oblix.ai">通过智能混合编排提升您的 AI 性能 | Oblix.ai</a>：体验我们的交互式演示，了解我们的智能 Agent 如何在本地 LLM 执行和云提供商之间无缝切换，以实现最佳性能和成本效益。</li><li><a href="https://github.com/GeekyGhost/Little-Geeky-s-Learning-UI.git">GitHub - GeekyGhost/Little-Geeky-s-Learning-UI: 一个使用 Kokoro TTS 的基于 Ollama 的 Gradio UI</a>：一个使用 Kokoro TTS 的基于 Ollama 的 Gradio UI。可以通过在 GitHub 上创建账号来为 GeekyGhost/Little-Geeky-s-Learning-UI 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 messages): 

.mwayne: https://blog.roboflow.com/fine-tune-sam-2-1/amp/
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1352359474123243672)** (2 messages): 

> `Manual Looping vs Vectorization, GSM8K Dataset, Tokenizer ChatML Format, Certifications` 


- **向量化处理胜过手动循环**：一位成员发现向量化的性能*远好于*手动循环。
   - 原作者报告称，他们之前使用 `for` 循环手动实现，并将其描述为*有点绕弯子*。
- **GSM8K 数据集出现困难**：一位成员表示在理解涉及 **GSM8K 数据集** 的下一个 Notebook 任务时遇到困难。
   - 该成员对*创建带有角色（role）和内容（content）的消息格式*的指令感到特别困惑。
- **探讨 Tokenizer 的 ChatML 格式**：有人对 **tokenizer 方法** 是否总是实现相同的 **ChatML 格式** 提出了疑问。
   - 该成员质疑函数如何知道原始数据集的格式。
- **认证作业位置**：一位成员询问在哪里可以找到获得**认证**的作业。
   - 未提供关于特定认证或平台的更多细节。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1352413037948309698)** (24 条消息🔥): 

> `HF 课程证书、Unit 2.1 错误、用于 UI 自动化的 AI Agent、Langfuse 错误、本地运行的 Smolagent 模型` 


- **HF 课程证书获取**：一位成员询问在完成 Hugging Face 课程的 **Unit 2** 后如何获得证书。
   - 另一位成员请求加入用于 **UI 自动化的 AI Agent**。
- **HF 学习者遇到 Unit 2.1 问题**：一位成员在自己的 Python 环境中运行 **Unit 2.1** *Building Agents That Use Code* 的代码时遇到错误，图片显示代码崩溃。
   - 一位成员建议使用 **hf_token** 来运行模型，检查条款和条件，或验证 HFApi 对象是否包含该 token。
- **Langfuse 集成挑战**：成员们报告在 **unit2/smolagents/code_agents.ipynb** 笔记本中尝试连接到 **Langfuse** 时，遇到了与 **smolagents** 模块相关的 **AttributeError**。
   - 一位维护者确认了该问题，并指出了与 **openinference** 插桩库相关的错误报告[此处](https://github.com/Arize-ai/openinference/issues/1399)和修复方案[此处](https://github.com/Arize-ai/openinference/pull/1403)。
- **本地 Smolagent 模型探索**：一位成员询问了最适合在本地运行的 **smolagent 模型**，以及如何在实际程序中实现它。
   - 该成员分享了在实现 **multiagents 模块** 时遇到的挑战。
- **AI Agents 课程：学习回顾**：一位成员在 [Medium 博客文章](https://medium.com/@jurriaan.nagelkerke/the-ai-agents-course-a-review-b289f90799ea) 中分享了他们完成 AI Agents 课程 Unit 1 的经验和心得。
   - 他们分享了关于如何从第一个单元中获得最大收获的技巧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@jurriaan.nagelkerke/the-ai-agents-course-a-review-b289f90799ea">The 🤗 AI Agents Course: A review</a>：作为一名数据科学家或 LLM 爱好者，你现在到处都能听到和读到关于 Agent 的消息。不幸的是，并不是每个人都有相同的想法……</li><li><a href="https://github.com/Arize-ai/openinference/issues/1399">[bug] SmolagentsInstrumentor - AttributeError: module &#39;smolagents&#39; has no attribute &#39;ApiModel&#39; · Issue #1399 · Arize-ai/openinference</a>：描述错误：在按照指南使用 SmolagentsInstrumentor 对 Hugging Face smolagents 进行插桩时，我遇到了以下错误：AttributeError: module &#39;smolagents&#39; has no attri...</li><li><a href="https://github.com/Arize-ai/openinference/pull/1403">fix: only import exported smolagents models by njbrake · Pull Request #1403 · Arize-ai/openinference</a>：修复了 #1399
</li>
</ul>

</div>

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1352360167072596091)** (69 条消息🔥🔥): 

> `mcp-mysql-server 问题, fastmcp 框架, Vibe Coding, DaVinci Resolve MCP 更新, Glama API 故障` 


- ****MySQL 与 MCP 连接难题****：一位用户正苦于将 **mcp-mysql-server** 连接到 **Docker MySQL**，报告称尽管在 **MCP** 之外运行正常，但在其中每次连接都会失败。
- ****Fastmcp 框架的困扰****：一位用户怀疑 **fastmcp 框架** 剥离了传递给命令的“隐藏”参数，导致其代码中从 `RegisterCommandsSchema` 到 `_RegisterCommandsSchema` 出现问题。
- ****Vibe Coding 辩论引发热议****：一些用户拿 *Vibe Coding* 开玩笑，即编码者盯着屏幕，虽然什么都看不懂，但不知怎么地就出了结果。
- ****DaVinci Resolve MCP 寻求快速服务器认领****：一位用户正寻求重新提交带有许可证和更新的 **DaVinci Resolve MCP** 项目，并被告知认领服务器可能会加速更新过程，随后指向了他们的 [repo](https://github.com/samuelgursky/davinci-resolve-mcp)。
- ****Glama API 投诉频发****：一位用户报告从 **Glama API** 收到 **500 错误**，但另一位成员表示过去 24 小时内没有发生故障，并分享了用于复现的代码示例：`curl -X 'GET' 'https://glama.ai/api/mcp/v1/servers?first=10&query=github' -H 'accept: application/json'`。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/jamiew/status/1903111313968046556">Jamie Dubs (@jamiew) 的推文</a>：1. MCP 很复杂、过度设计、难以在云端托管，且安全模型缺失。2. MCP 也是目前“LLM 插件”或“面向 LLM 的 SDK”的最佳解决方案。其他一切...</li><li><a href="https://github.com/cunhapaulo/marpstyle">GitHub - cunhapaulo/marpstyle: 以美观和简洁为核心设计的 Marp 主题仓库。</a>：以美观和简洁为核心设计的 Marp 主题仓库。 - cunhapaulo/marpstyle</li><li><a href="https://github.com/ggozad/oterm">GitHub - ggozad/oterm: 一个基于文本的 Ollama 终端客户端</a>：一个基于文本的 Ollama 终端客户端。通过在 GitHub 上创建账号来为 ggozad/oterm 做出贡献。</li><li><a href="https://github.com/samuelgursky/davinci-resolve-mcp">GitHub - samuelgursky/davinci-resolve-mcp: DaVinci Resolve 的 MCP 服务器集成</a>：DaVinci Resolve 的 MCP 服务器集成。通过在 GitHub 上创建账号来为 samuelgursky/davinci-resolve-mcp 做出贡献。</li><li><a href="https://github.com/markuspfundstein/mcp-gsuite">GitHub - MarkusPfundstein/mcp-gsuite: 用于与 Google Gsuite 产品交互的 MCP 服务器</a>：用于与 Google Gsuite 产品交互的 MCP 服务器 - MarkusPfundstein/mcp-gsuite</li><li><a href="https://github.com/isaacphi/mcp-gdrive">GitHub - isaacphi/mcp-gdrive: 用于读取 Google Drive 和编辑 Google Sheets 的 Model Context Protocol (MCP) 服务器</a>：用于读取 Google Drive 和编辑 Google Sheets 的 Model Context Protocol (MCP) 服务器 - isaacphi/mcp-gdrive</li><li><a href="https://github.com/kazz187/mcp-google-spreadsheet">GitHub - kazz187/mcp-google-spreadsheet: Google Spreadsheet 的 MCP 服务器</a>：Google Spreadsheet 的 MCP 服务器。通过在 GitHub 上创建账号来为 kazz187/mcp-google-spreadsheet 做出贡献。</li><li><a href="https://github.com/distrihub/mcp-google-workspace">GitHub - distrihub/mcp-google-workspace: 一个用 Rust 编写的用于与 Google Drive 和 Google Sheets 交互的 Model Context Protocol (MCP) 服务器。</a>：一个用 Rust 编写的用于与 Google Drive 和 Google Sheets 交互的 Model Context Protocol (MCP) 服务器。 - distrihub/mcp-google-workspace</li><li><a href="https://github.com/akchro/google-sheets-mcp">GitHub - akchro/google-sheets-mcp</a>：通过在 GitHub 上创建账号来为 akchro/google-sheets-mcp 做出贡献。</li><li><a href="https://github.com/rishipradeep-think41/drive-mcp">GitHub - rishipradeep-think41/drive-mcp</a>：通过在 GitHub 上创建账号来为 rishipradeep-think41/drive-mcp 做出贡献。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1352359324382527518)** (6 条消息): 

> `Microsoft Semantic Workbench, Turso MCP tool video, Asana MCP + Google Calendar MCP, MCPHub.nvim + Avante + Figma MCP` 


- **Microsoft 发布 Semantic Workbench 工具**：Microsoft 发布了 [Semantic Workbench](https://github.com/microsoft/semanticworkbench)，这是一个 VS Code 扩展，被描述为用于原型设计智能助手、Agent 和多 Agent 系统的通用工具，引发了关于其作为 **MCP** 角色的疑问。
   - 一位成员在查看其描述后询问它是否是一个 **MCP**。
- **Jamie 演示 Turso MCP 工具**：Jamie 制作了一个 Turso MCP 工具的视频，该工具由 @tursodatabase 社区构建，并在一条 [推文](https://x.com/notrab/status/1902767330007941472?s=46&t=4RSOl8kQCdkHm0U5FcdeaA) 中展示。
   - 视频展示了 Claude 为域名集合创建一个数据库。
- **使用 Asana 和 Google Calendar MCP 自动进行日历排程**：一篇博客文章详细介绍了如何结合 Goose 使用 **Asana MCP** 和 **Google Calendar MCP** 来自动执行任务调度，展示了如何通过单个提示词从 Asana 提取任务、进行分析并在 Google Calendar 中安排日程。[博客文章](https://block.github.io/goose/blog/2025/03/20/asana-calendar-mcp) 强调了自动化组织任务和会议所带来的节省时间的优势。
   - **Asana MCP** 服务端位于 [Asana](https://github.com/roychri/mcp-server-asana)，**Google Calendar MCP** 位于 [Google Calendar](https://www.pulsemcp.com/servers?q=google+calendar)。
- **nvim 与 Figma 集成**：一位用户展示了 **MCPHub.nvim** 与 **Avante** 及 **Figma MCP** 的集成，如分享的视频 [login-figma.mp4](https://cdn.discordapp.com/attachments/1315696461316358175/1352480836896952340/login-figma.mp4?ex=67ded42f&is=67dd82af&hm=1997f11d01ec6f95e16f5e9c69b326c6c9c06cb5124da34a4b74fb7d95a8293f) 所示，演示了流线型的工作流。
   - 其他用户对此表示关注，并评论道 *"看起来很酷"*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/notrab/status/1902767330007941472?s=46&t=4RSOl8kQCdkHm0U5FcdeaA">来自 Jamie Barton (@notrab) 的推文</a>：在这里我让 Claude 为我的域名集合创建一个数据库。别担心，我没有包含完整列表，视频只有 90 秒。👏 特别感谢 @spences10 和 @tursodatabas...</li><li><a href="https://block.github.io/goose/blog/2025/03/20/asana-calendar-mcp">MCP 实战：我如何使用 Goose、Asana 和 Google Calendar 通过 AI 规划我的一周</a>：结合 Goose 使用 MCP 来自动化任务管理并提高生产力。</li><li><a href="https://github.com/microsoft/semanticworkbench">GitHub - microsoft/semanticworkbench: 一款旨在帮助原型设计智能助手、Agent 和多 Agent 系统的通用工具</a>：一款旨在帮助原型设计智能助手、Agent 和多 Agent 系统的通用工具 - GitHub - microsoft/semanticworkbench...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1352358751738269696)** (64 条消息🔥🔥): 

> `OpenRouter TTS, Ernie Models, Sambanova, Inferencenet, OpenAI audio models` 


- **OpenRouter 将提供 TTS 和图像生成，但伴随成本担忧**：一位成员对 **OpenRouter** 提供 **TTS 和图像生成**表示感兴趣，但也对潜在的高昂定价表示担忧。
- **Groq 与 Sambanova 的混淆**：一位成员最初报告 **Sambanova** 宕机，但随后纠正称是 **Groq** 出现了问题。
- **GPT-4o 上线**：用户注意到 **GPT-4o-64k-output-alpha** 已在 **OpenRouter** 上可用，支持**文本和图像输入以及文本输出**，成本为 **$6/M input tokens** 和 **$18/M output tokens**。
- **推理模型使用数据分享**：一位成员发布了 [token 使用数据](https://dubesor.de/reasoningtok) 以及对**推理模型**的看法，并将其与传统模型进行了对比。
- **Fireworks 降价，性能持平**：**Fireworks** 降低了 **R1 和 V3** 的定价，据报道 V3 达到了现有的性能指标，具体为 **.9/.9**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - 管理模型使用和配额</a>：了解 OpenRouter 的 API 速率限制、基于额度的配额和 DDoS 保护。有效地配置和监控您的模型使用限制。</li><li><a href="https://openrouter.ai/openai/gpt-4o:extended">GPT-4o (extended) - API、提供商、统计数据</a>：GPT-4o（“o”代表“omni”）是 OpenAI 最新的 AI 模型，支持文本和图像输入以及文本输出。它保持了 [GPT-4 Turbo](/models/open... 的智能水平。</li><li><a href="https://huggingface.co/nghuyong">nghuyong (HuYong)</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/faq#api-technical-specifications">OpenRouter FAQ</a>：查找有关 OpenRouter 统一 API、模型访问、定价和集成的常见问题解答。</li><li><a href="https://fireworks.ai/pricing">Fireworks - 生成式 AI 的最快推理</a>：以极快的速度使用最先进的开源 LLM 和图像模型，或者使用 Fireworks AI 免费微调和部署您自己的模型！</li><li><a href="https://github.com/mintsuku/sora">GitHub - mintsuku/sora: Sora 是一个与 Open Router API 集成的 Discord 机器人，旨在促进 Discord 服务器中的对话。</a>：Sora 是一个与 Open Router API 集成的 Discord 机器人，旨在促进 Discord 服务器中的对话。 - mintsuku/sora</li><li><a href="https://tenor.com/view/eyes-burning-my-eyes-spongebob-gif-5183364">Eyes GIF - Eyes Burning My Eyes - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1352366344917745684)** (9 条消息🔥): 

> `vast.ai ncu profiling, Jake, Spam detection with neural nets` 


- **关于 Vast.ai 上 NCU Profiling 的疑问**：一位成员询问 [vast.ai](https://vast.ai) 是否允许进行 **ncu profiling**。
   - 另一位成员回应称，虽然有一位 ID 为 *Jake* 的人在场，但他们怀疑是否提供了裸机（bare metal）访问权限。
- **通过神经网络自动检测垃圾邮件**：一位成员提到知道一个服务器使用某种**神经网络**实现了垃圾邮件的自动检测。
   - 未提供进一步的细节或链接。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1352378570441232445)** (6 条消息): 

> `cuTile talk, atomic addition with bfloat16, triton 3.1.0 and triton-windows 3.2.0, Triton's ease of use, sparse attention pattern` 


- **频道关注 cuTile 演讲**：一名成员建议邀请相关人员在频道上进行关于 **cuTile** 的演讲，该建议[正在筹备中](https://www.nvidia.com/en-us/cuda/cutile/)。
   - 未提供更多细节。
- **通过锁实现 BFloat16 Atomic Addition**：一名成员报告称，使用 `tl.atomic_cas` 配合锁来实现 **bfloat16 的 atomic addition** 确实可行，但[效果不佳](https://github.com/openai/triton/blob/main/python/triton/runtime/jit.py)。
   - 该成员正在寻求改进实现的方法，并提供了一段使用 `tl.atomic_cas` 加锁的代码片段，邀请社区协助提升其性能。
- **安装后 Triton 版本冲突**：在成功运行 `triton_test.py` 后，一名成员发现 pip 列表中同时出现了 **triton 3.1.0** 和 **triton-windows 3.2.0**，由于 CMD 中显示的文件过多，他犹豫是否要卸载旧版本。
   - 他们征求了关于是否卸载 **triton 3.1.0** 的建议，但目前尚未收到解决方案。
- **Triton 的简洁性吸引 GPU 新手**：一名成员强调，**Triton** 的核心优势不在于峰值性能，而在于其易用性，使 GPU 经验有限的人也能编写复杂的 kernel，并引用了 [lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch/blob/main/native_sparse_attention_pytorch/triton_native_sparse_attention.py) 作为例子。
   - 他们指出，在预定义的工作负载上实现峰值性能相对简单，但 Triton 的鲁棒性使其脱颖而出。



**提到的链接**：<a href="https://github.com/lucidrains/native-sparse-attention-pytorch/blob/main/native_sparse_attention_pytorch/triton_native_sparse_attention.py">native-sparse-attention-pytorch/native_sparse_attention_pytorch/triton_native_sparse_attention.py at main · lucidrains/native-sparse-attention-pytorch</a>：由 Deepseek 团队在其 "Native Sparse Attention" 论文中提出的 sparse attention 模式的实现 - lucidrains/native-sparse-attention-pytorch

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1352481236647411834)** (3 条消息): 

> `FlashMLA SmemLayoutP, Pointer Tagging` 


- **FlashMLA `SmemLayoutP` 维度解析**：一名成员询问了 [FlashMLA](https://github.com/deepseek-ai/FlashMLA/blob/b31bfe72a83ea205467b3271a5845440a03ed7cb/csrc/flash_fwd_mla_kernel.h#L76C5-76C93) 代码中 `SmemLayoutP` 的维度，特别是其形状 `((2,2), kNThreadsS, 1, kBlockN/8)` 以及 `kNThreadsS` 在 warpgroups 之间同步 **P** 的作用。
   - 该成员推测其他维度可能与 **wgmma** 有关，正等待其他专家的澄清。
- **Pointer Tagging 潜在陷阱探讨**：一名成员询问了使用 `uint32_t*` 实现 **pointer tagging** 时可能遇到的坑，并引用了编程指南中关于 17 个可用位的建议。
   - 他们附带了一张来自编程指南的[图片](https://cdn.discordapp.com/attachments/1189607726595194971/1352700676869980180/image.png?ex=67def82d&is=67dda6ad&hm=a2758b3f742c29f70d532f4c615325152feb59225e29ab987c394a34306a5662)作为背景参考。



**提到的链接**：<a href="https://github.com/deepseek-ai/FlashMLA/blob/b31bfe72a83ea205467b3271a5845440a03ed7cb/csrc/flash_fwd_mla_kernel.h#L76C5-L76C93">FlashMLA/csrc/flash_fwd_mla_kernel.h at b31bfe72a83ea205467b3271a5845440a03ed7cb · deepseek-ai/FlashMLA</a>：FlashMLA：高效的 MLA decoding kernels。欢迎在 GitHub 上为 deepseek-ai/FlashMLA 做出贡献。

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1352405039460913223)** (5 条消息): 

> `ZeRO offload, full-finetuning, 8B model, BF16, A100 40GB` 


- **针对 8B 微调的 ZeRO Offload 探讨**：一名成员询问 **ZeRO offload** 是否支持在单张 **A100 40GB** GPU 上对 **8B 模型**（使用 **BF16**）进行全量微调（full fine-tuning）。
   - 另一名成员建议 **ZeRO** 主要用于分布式训练，对于单卡，使用 checkpointing + gradient accumulation 会更好。
- **FSDP2 的 Offload 参数研究**：一名成员提到 **FSDP2** 具有 *offload_to_cpu* 参数，而另一名成员建议更好的切入点是 [torchao 的 offload optimizer](https://pytorch.org/torchao/)。
- **DeepSpeed 的 Zero Offload 宣称性能**：一名成员提到 [DeepSpeed 的 Zero offload](https://www.deepspeed.ai/tutorials/zero-offload/) 声称可以在单张 GPU 上训练高达 **13B** 的模型。
   - 他们正在寻找更好的实现方式。

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1352490810834157652)** (2 messages): 

> `GPU Mode Scammer, Discord Channel Alerts` 


- **诈骗者警报拉响**：一名用户在频道中发出警报，称 **GPU Mode** 频道中出现了 *scammer*（诈骗者），提示可能存在欺诈活动。
   - 消息中包含了与 **GPUs** 和 **Dragon Ball Z** 相关的自定义表情符号，可能是为了幽默或吸引注意力。
- **Discord 频道经历诈骗惊魂**：**GPU Mode** Discord 频道的成员收到了关于潜在诈骗者出现的警告。
   - 该警告缺乏具体细节，但作为一般性提醒，建议大家保持警惕。


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1352411279092088943)** (3 messages): 

> `Hopper Architecture, Microbenchmarking, Matrix Multiplication` 


- **Hopper 的每周期 Flops**：在 **Hopper** 架构上，通过 Microbenchmarking 可以测得 **每个 SM 每周期 4096 flops/2048 MAD (16-bit)**。
   - 当使用 **8-bit 类型** 时，由于其具有两个 matmuls（QxK 和 xV），这一数值会翻倍。
- **建议对 Hopper 进行 Microbenchmarking**：一名成员建议，*Microbenchmarking 将是获取 Hopper 架构性能细节的有效途径*。
   - 另一名成员认为这些信息可能在某些文档中有记载，但未能找到相关引用。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1352391129017880656)** (3 messages): 

> `GTC Presentation, CUDA Kernels, Small Transformer Models, Hopper Architecture, CUTLASS 4.0` 


- **GTC 演讲承诺提供性能优化的 CUDA Kernels**：一名成员宣布了他们的 GTC 演讲，题为“针对小型 Transformer 模型推理的性能优化 CUDA Kernels [S73168]”，该演讲于今日下午 4 点举行，重点关注 **Hopper 架构**。
   - 他们鼓励参会者到场交流并提出深度问题，同时提到 [演讲录像](https://www.nvidia.com/gtc/) 将在 GTC 网站上发布，供无法亲临现场的人员观看。
- **CUTLASS 4.0 将在 GTC 重新定义 Pythonic 集成**：参会者将在 GTC 了解到 **CUTLASS** 下一个重大版本 4.0 中关于 Pythonic 未来的规划。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1352542892949897257)** (4 messages): 

> `Deprecated Coach class, Curriculum Experiments` 


- **Coach 类面临退役**：成员们讨论了移除 **Coach 类** 的事宜，认为其已被弃用，取而代之的是 **Curriculum Experiments**。
   - 一名成员同意开启一个 PR，确认这是*早期尝试的残留物*。
- **Curriculum 执行万岁**：讨论集中在简化代码库中的 Curriculum 执行流程。
   - 此举旨在整合开发工作，避免旧方法与新 Curriculum 管理方法之间产生混淆。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1352372054267006999)** (11 messages🔥): 

> `Leaderboard Submissions, GPU Tests` 


- **Grayscale 排行榜收到大量提交**：多个针对 `grayscale` 排行榜的提交在 GPU 上成功运行：包括使用 Modal runners 的 **L4**、**T4**、**A100** 和 **H100**。
   - 提交 ID 包括 `2351`、`2429`、`2430`、`2431`、`2459` 和 `2460`。
- **Vectoradd 基准测试取得成功**：ID 为 `2363` 的基准测试提交在 `vectoradd` 排行榜上成功运行，涉及 GPU：**T4**、**L4**、**A100**、**H100**（使用 Modal runners）。
   - 这标志着 `vectoradd` 基准测试在各种 GPU 架构上取得了进展。
- **Grayscale 测试在 A100 GPU 上通过**：ID 为 `2422` 和 `2423` 的测试提交在 **A100** GPU 上通过了 `grayscale` 排行榜测试（使用 Modal runners）。
   - 这些测试专门针对 **A100** GPU 架构，表明了针对性的测试工作。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1352455340200820756)** (2 messages): 

> `Consumer GPUs, Cloud GPUs, Local vs Cloud` 


- **消费级 GPU 淘汰速度极快**：成员们讨论了消费级 GPU 淘汰速度之快，尤其是用于游戏以外的目的时。
   - 一名成员指出，以同样的成本（约 1000 美元），用户可以滚动式地使用最新的云端 GPU，而无需任何长期承诺；而另一名成员则提到，*如果你家里没有 GPU，你就永远听不到它发出 "brrr" 的运转声*。
- **本地 GPU 爱好者享受 "brrr" 的感觉**：发烧友们非常看重本地运行 GPU 时的听觉反馈。
   - 一位用户表示：*如果你家里没有 GPU，你就永远听不到它发出 "brrr" 的运转声*。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1352502574745518182)** (35 messages🔥): 

> `Oblix, AI Orchestration, Local LLM for SFW Stories, LLM Leaderboards, PC Build for Medical Data` 


- ****Oblix** 在本地与云端之间无缝切换**：一名成员分享了 **Oblix** 的演示视频 ([https://youtu.be/j0dOVWWzBrE](https://youtu.be/j0dOVWWzBrE))，该工具通过使用 Agent 监控系统资源并做出决策，在保持上下文的同时实现*本地与云端之间的无缝切换*。
   - 该平台在 **Ollama** 和 **OpenAI** 之间进行编排，动态决定是在本地还是在云端处理每个 AI 请求，以实现最佳性能和成本效益，详见 [Oblix.ai](https://oblix.ai/)。
- **用于模型选择的 LLM 排行榜对比**：成员们讨论了如何为特定用途寻找可靠的 LLM 排行榜，其中一名成员分享了 [Artificial Analysis](https://artificialanalysis.ai/leaderboards/models) 和 [LM Arena](https://lmarena.ai/?leaderboard) 的链接。
   - 讨论中提到了从这些列表中筛选相关模型的担忧，特别是如何避开过时或不理想的选项，如 **Grok-3**。
- **成员寻求用于医疗数据处理的 PC 配置建议**：一名成员请求协助组装一台新 PC，用于使用 AI 处理医疗数据，强调需要安全、离线运行，并提到 GitHub 上的说明不够清晰。
   - 另一名成员建议从 **Intel i9**、**128GB RAM** 和 **Nvidia 4090 RTX** 开始配置。
- **GPT4All 并非音频文件转录的理想选择**：一名成员询问是否可以使用 **GPT4All** 进行本地音频文件转录，特别是上传 **.wav** 文件。
   - 另一名成员澄清说 **GPT4All** 主要针对 **docs/pdf** 设计，建议使用 **XTTS webui** 进行 wav 到文本的转换，但也指出其安装并不简单。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>：未找到描述</li><li><a href="https://artificialanalysis.ai/leaderboards/models">LLM Leaderboard - Compare GPT-4o, Llama 3, Mistral, Gemini &amp; other models | Artificial Analysis</a>：对比并排名了超过 30 个 AI 模型 (LLMs) 的性能，涵盖质量、价格、性能和速度（输出速度 - tokens per second 以及延迟 - TTFT）、上下文等关键指标...</li><li><a href="https://oblix.ai/">Transform Your AI Performance with Intelligent Hybrid Orchestration | Oblix.ai</a>：体验我们的交互式演示，了解我们的智能 Agent 如何在本地 LLM 执行和云提供商之间无缝切换，以实现最佳性能和成本效益。
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1352378909491986473)** (10 messages🔥): 

> `W-GAN saturation, Transformers soft slots, MCP UX/UI` 


- **W-GANs 缓解梯度爆炸**：传统的 GANs 由于 BCE 会产生饱和，而 **W-GANs** 通过线性化缓解了这一问题，如 [W-GAN 论文中的图 2](https://cdn.discordapp.com/attachments/986699377257119794/1352378909194322001/image.png?ex=67de7541&is=67dd23c1&hm=10de0ff85871d920b5ff7db506224a588a2c6c44030082a2ba7a9aad232ef204) 所示。
   - 虽然梯度消失问题较少，但如果生成器或判别器变得过于强势，仍可能发生不稳定，导致两端饱和。
- **Transformers 中的软槽（Soft Slots）动态绑定**：一名成员分享了关于软槽方法的图像分析，展示了软槽如何动态绑定到 Transformers 中的输入 Token 或检索内容。
   - 提供了 **Attention** 和 **Soft Slots (S')** 的方程式，强调了在具有可学习槽位的情况下使用 softmax 和缩放点积注意力机制（scaled dot-product attention）。
- **OpenAI.fm 的 UX/UI 仓促且简单**：成员们评论了 [OpenAI.fm](https://www.openai.fm/) 简单的 UX/UI，开玩笑说它看起来很仓促。
   - 一名成员引用了一个观点，即 MCP 强制执行了过多的结构，这使其容易受到不够结构化的协议的干扰，后者可以根据用户需求进行演进。该观点强调在服务器-客户端系统中，*高方差几乎总是获胜*，因为*客户端会消费更多他们喜欢的内容，而减少消费他们不喜欢的内容*。



**提到的链接**：<a href="https://www.openai.fm/">OpenAI.fm</a>：一个供开发者试用 OpenAI API 中全新文本转语音模型的交互式演示。

  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1352407070485516379)** (3 条消息): 

> `G-Retriever, Graph Question Answering, Graph RAG` 


- **G-Retriever 支持与图表对话**：[G-Retriever 论文](https://arxiv.org/abs/2402.07630) 详细介绍了从知识图中提取语义信息的方法，实现了 *与你的图表对话 (chatting with your graph)*、*图表问答 (graph QnA)* 以及 *Graph RAG*。
- **引入图表问答基准测试**：该论文引入了一个 **Graph Question Answering (GraphQA) 基准测试**，其数据收集自不同的应用场景，包括场景图理解、常识推理和知识图谱推理。



**提及的链接**：<a href="https://arxiv.org/abs/2402.07630">G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering</a>：给定一个带有文本属性的图，我们让用户能够 `与他们的图表对话`：即使用对话界面询问有关图的问题。针对用户的问题...

  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1352356691559321640)** (16 条消息🔥): 

> `Claude Pokemon, AI Moore's Law, Hunyuan-T1 model` 


- **Claude 的 Pokemon 实力受到质疑**：成员们对 **Claude** 的能力表示怀疑，指出它在 *“玩 Pokemon 方面相当垃圾”*，尽管整体有所提升，但仍对其能力提出质疑。
- **AI Agent 的摩尔定律**：讨论围绕 [METR_Evals 的研究](https://x.com/METR_Evals/status/1902384481111322929) 展开，该研究提出了 *“AI Agent 的摩尔定律”*，即 AI 能够完成的任务长度大约每 **7 个月** 翻一倍。
   - 一些成员认为相关图表是 *“纯属扯淡”*，认为某些任务（如训练分类器或优化芯片制造）对于概率模型来说不应该具有吸引力。
- **混元-T1 (Hunyuan-T1) 发布，主打推理**：根据 [腾讯](https://x.com/TXhunyuan/status/1903121005809373386) 的消息，由 Hunyuan TurboS 驱动的 **Hunyuan-T1** 采用了 **Hybrid-Mamba-Transformer MoE 架构**，旨在提高速度、准确性和效率。
   - 这款新模型在摘要生成中具有 **低幻觉 (low hallucination)** 的特点，并且在 **长文本处理** 方面表现出色，详情见 [Hunyuan-T1 HuggingFace demo](https://huggingface.co/spaces/tencent/Hunyuan-T1)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/TXhunyuan/status/1903121005809373386">来自 Hunyuan (@TXhunyuan) 的推文</a>：🚀 介绍 Hunyuan-T1！🌟 遇见 Hunyuan-T1，AI 推理领域的最新突破！由 Hunyuan TurboS 驱动，专为速度、准确性和效率而生。🔥✅ Hybrid-Mamba-Transformer MoE A...</li><li><a href="https://x.com/METR_Evals/status/1902384481111322929">来自 METR (@METR_Evals) 的推文</a>：AI 系统何时能独立开展长期项目？在新的研究中，我们发现了一种“AI Agent 的摩尔定律”：AI 能够完成的任务长度大约每 7 个月翻一倍...</li><li><a href="https://x.com/METR_Evals/status/1902384495673680188">来自 METR (@METR_Evals) 的推文</a>：然后，我们拟合了一条曲线，根据人类完成每项任务所需的时间来预测 AI 的成功率。这条曲线描述了 AI 在不同任务长度下的能力。然后我们总结...</li><li><a href="https://fxtwitter.com/bycloudai/status/1903149418422939838">来自 bycloud (@bycloudai) 的推文</a>：&gt; mamba-transformer 混合推理模型，性能接近 DeepSeek-R1。引用 Hunyuan (@TXhunyuan) 🚀 介绍 Hunyuan-T1！🌟 遇见 Hunyuan-T1，AI 推理领域的最新突破！由...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1352681010281713745)** (1 条消息): 

> `Local RAG app, GitIngest parsing, Streamlit UI, Ollama Llama 3.2` 


- **全本地 RAG 应用在代码对话中表现出色**：一位 LlamaIndex 社区成员构建了一个全本地、全开源的 **RAG 应用**，可以与你的代码进行对话，并在 [推文](https://twitter.com/llama_index/status/1903121505984319771) 中发布。
- **GitIngest 解析并用于 Streamlit 显示**：该应用使用 **GitIngest** 将代码解析为摘要和 Markdown，并使用 **Streamlit** 构建 UI，详情请见 [此链接](https://t.co/WIVkzX33EB)。
- **Ollama 在本地运行 Llama 3.2**：它使用 **Ollama** 在本地运行 **Meta 的 Llama 3.2**。

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1352391380088914030)** (13 messages🔥): 

> `LlamaIndex TypeScript Agent 导入问题, Agent Workflow 并行执行限制, Human-in-the-Loop 工具限制` 


- **TypeScript Agent 导入问题已解决**：一位使用 **LlamaIndex TS** 的成员在导入 **agent** 时遇到问题，通过更新 **tsconfig** 的 bundler 配置解决了该问题。
   - 用户确认修改 **TS config** 解决了导入错误，并感谢社区的建议。
- **限制 Agent Workflows 中的并行执行**：一位成员询问如何限制 **Agent Workflows** 中的并行执行，特别是针对带有 **human-in-the-loop event** 的工具。
   - 他们注意到 Agent 并行多次调用该工具导致了问题，希望确保该工具一次只被调用一次；该问题已在 [GitHub](https://github.com/run-llama/llama_index/issues/18220#issuecomment-2742089859) 上得到回复。
- **寻求 Human-in-the-Loop 工具约束的解决方案**：一位成员在 **agent workflow** 中因并行调用 **human-in-the-loop tool** 遇到问题，并寻求限制执行的方法。
   - 当 Agent Workflow 工具被多次并行调用时，用户遇到了异常问题，目前正在等待相关 [GitHub issue](https://github.com/run-llama/llama_index/issues/18220#issuecomment-2742089859) 的协助。



**提及的链接**：<a href="https://github.com/run-llama/llama_index/issues/18220#issuecomment-2742089859">[Question]: Parallel Human in Loop with Agent Workflow Issues · Issue #18220 · run-llama/llama_index</a>：问题验证。我已经搜索了文档和 Discord 以寻求答案。经过长时间的搜索和调试以寻找解决方案。感谢任何帮助！当 Agent Workflow 处于...

  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1352466135433609256)** (4 messages): 

> `试用 Key 限制, Command-A 训练数据` 


- **试用 Key 限制按账户计算**：一位成员询问每月 **1k 次请求的试用 Key 限制**是按 Key 还是按账户计算，另一位成员澄清是**按账户**。
   - 他们补充说，试图通过创建多个账户来绕过此限制将导致所有账户被删除。
- **关于 Command-A 训练数据的查询**：一位成员询问了 **Command-A 训练数据**的截止日期。


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1352544076934680648)** (4 messages): 

> `Cohere API 错误, 速率限制 (Rate Limiting), 检查速率限制` 


- **Cohere API 抛出常见错误**：用户讨论了各种 **Cohere API 错误信息**，包括 *invalid request*（无效请求）、*rate limiting*（速率限制）和 *token limits*（Token 限制）。
   - 错误涵盖了一系列问题，如**空文档**、**过短的 Prompt**、超过 **token 限制**以及**错误的模型规格**。
- **速率限制讨论**：成员指出，速率限制错误可以通过响应中的 **429 状态码**来识别，详情可见 [Cohere API 文档](https://docs.cohere.com/reference/errors#429---too-many-requests)。
   - 一位用户提到他们的代码因为没有收到响应而崩溃。
- **检查 API 速率限制**：一位用户询问是否有办法检查他们的**速率限制**以查看剩余额度。
   - 目前没有给出通过 API 检查速率限制的解决方案。



**提及的链接**：<a href="https://docs.cohere.com/reference/errors#429---too-many-requests">Errors (status codes and description) — Cohere</a>：了解 Cohere 的 HTTP 响应代码以及如何在各种编程语言中处理错误。

  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/1352396846101692578)** (3 messages): 

> `Bot 权限` 


- **Bot 权限问题**：一位用户提到 Bot 在该频道可能存在**权限问题**。
- **用户向 Bot 问好**：一位用户向另一位用户打招呼。未分享具体的 AI 讨论或链接。


  

---

### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1352392912314437673)** (2 messages): 

> `Introductions, Low-code tech, Community Engagement` 


- **酒店业专家加入 Cohere！**：Gaby 是一位**酒店业**专业人士，她介绍自己为低代码技术爱好者，精通 **Make** 和 **Adalo** 等平台。
   - 她表达了向社区成员学习并贡献自己经验的渴望。
- **低代码技术成为焦点**：Gaby 的介绍凸显了**低代码工具**在各行各业中日益增长的重要性，展示了它们的易用性和潜力。
   - 她在 **Make** 和 **Adalo** 方面的专业知识可以为 Cohere 社区内探索类似技术的其他成员提供宝贵的见解。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1352519841269157899)** (12 messages🔥): 

> `Duration Module Proposal, Mojo and PyTorch Integration, Nanosecond Precision as Base Unit` 


- **Duration 模块提案揭示异常行为**：一名成员在处理 **`duration` 模块提案**时遇到了一个奇特的问题，特别是涉及 `Ratio` 和 `Duration` 结构体的类型转换，并分享了展示该非预期行为的[代码片段](https://discord.com/channels/1013120035012476074/1173814954275168317)。
- **PyTorch 和 Mojo 的速度提升？**：一位用户询问了在 **Mojo** 中使用 **PyTorch** 的可能性，以及是否可以通过 **MAX** 加速训练过程。
   - 在提供的消息中没有回复，因此这个想法在该频道中仍是一个开放性问题。
- **纳秒精度的时间处理**：一名成员建议使用**纳秒精度**作为时间的基础单位，并指出 `UInt64` 的纳秒数可以覆盖超过 **500 年**，这应该是足够的。
   - 另一名成员指出，C++ 保证的默认时间分辨率可以表示*至少* **292 年**，并提到**秒是时间的基础 SI 单位**。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1352423597200904193)** (1 messages): 

> `MIPRO v2, LLM-as-a-judge, Automatic Metrics, DSPy Optimization, Evaluation Metrics` 


- **MIPRO v2 与 LLM-as-a-Judge 结合使用**：一名成员提到使用 **MIPRO v2** 配合 **LLM-as-a-judge** 作为评估指标，并指向了一个[数学推理教程](https://dspy.ai/tutorials/math/)。
   - 该数学推理教程提供了一个使用 **MIPRO** 作为指标的示例。
- **分享了 LLM-as-a-Judge 文档**：分享了来自 [DSPy 学习资源](https://dspy.ai/learn/evaluation/metrics/#intermediate-using-ai-feedback-for-your-metric) 的关于使用 **LLM-as-a-judge** 的文档。
   - 该文档提供了关于使用 **AI 反馈**进行指标评估的信息。
- **自动指标对 DSPy 至关重要**：强调了**自动指标**对于 **DSPy** 中的评估和优化是必不可少的。
   - DSPy 利用指标来跟踪进度并增强程序的有效性。
- **任务评估的指标定义**：指标被定义为根据数据示例对系统输出进行评分的函数。
   - 简单任务可以使用基础指标，如 *accuracy* 或 *exact match*，而复杂任务则需要使用 **AI 反馈**来检查多个输出属性的指标。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://dspy.ai/tutorials/math/">Reasoning - DSPy</a>: 用于编程（而非提示）语言模型的框架。</li><li><a href="https://dspy.ai/learn/evaluation/metrics/#intermediate-using-ai-feedback-for-your-metric">Metrics - DSPy</a>: 用于编程（而非提示）语言模型的框架。
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1352552465844928522)** (1 messages): 

> `Unet3d model, 2D Convolutions` 


- **Unet3d 模型使用 2D 卷积**：一名成员质疑示例中的 **unet3d** 模型是否真的是 3D 的，认为由于它在 3D 输入上依赖 **2D 卷积**和 **2D 转置**，它更像是 **2.5D**。
   - 该成员强调了这与真正的 3D Unet 架构之间的区别。
- **2D 与 3D Unet 架构**：讨论强调了在 3D 输入上使用 **2D 卷积**（导致 2.5D 效果）与采用具有 3D 操作的真正 **3D Unet 架构**之间的区别。
   - 用户寻求关于该实现维度的澄清。


  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/)** (1 条消息): 

krammnic: 我喜欢这个: https://arxiv.org/pdf/2502.07923
  

---


---


---


---


---


{% else %}


> 完整的逐频道详情已针对邮件进行了截断。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}