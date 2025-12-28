---
companies:
- openai
- cohere
- lm-arena
- deepmind
- x-ai
- meta-ai-fair
- alibaba
- vllm
- llamaindex
date: '2025-04-30T15:44:39.731046Z'
description: '**OpenAI** 在一次备受争议的 ChatGPT 更新后遭遇抵制，随后发布官方撤回声明，承认公司“过于关注短期反馈”。来自 **Cohere**
  的研究人员发表了一篇论文，批评 **LMArena** 存在不公平做法，偏袒 **OpenAI**、**DeepMind**、**X.ai** 和 **Meta
  AI Fair** 等老牌巨头。阿里巴巴发布了 **Qwen3 系列**模型，其最大规模达 **235B MoE**（混合专家模型），支持 **119 种语言**，并基于
  **36 万亿 token** 训练而成，目前已集成至 **vLLM** 并支持 **llama.cpp** 等工具。**Meta** 宣布启动第二轮 **Llama
  影响力资助计划**（Llama Impact Grants），以促进开源 AI 创新。AI 推特（X）上的讨论重点关注了排行榜过拟合以及模型基准测试的公平性问题，**karpathy**
  等业内人士也对此发表了重要评论。'
id: MjAyNS0w
models:
- qwen3-235b-a22b
- qwen3
- qwen3-moe
- llama-4
people:
- joannejang
- arankomatsuzaki
- karpathy
- sarahookr
- reach_vb
title: ChatGPT 回应 GlazeGate + LMArena 回应 Cohere
topics:
- model-releases
- model-benchmarking
- performance-evaluation
- open-source
- multilinguality
- model-integration
- fine-tuning
- model-optimization
---

**AI Drama is all we need.**

> 2025年4月29日至4月30日的 AI 新闻。我们为您检查了 9 个 subreddit、449 个 Twitter 账号和 29 个 Discord 社区（包含 214 个频道和 5096 条消息）。预计节省阅读时间（按 200wpm 计算）：442 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 风格呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻解析，并在 @smol_ai 上向我们提供反馈！

也许这太巧了，就在 Dario Amodei 强调 [可解释性的紧迫性（the Urgency of Interpretability）](https://www.darioamodei.com/post/the-urgency-of-interpretability) 的一周后，ChatGPT 发布了一个遭到广泛差评的更新，以至于不得不[连夜发布官方撤回声明](https://openai.com/index/sycophancy-in-gpt-4o/)，称“*我们过于关注短期反馈，没有充分考虑到用户与 ChatGPT 的互动是如何随时间演变的*”。Model Spec 的 Joanne Jang 甚至进行了一场[罕见的 Reddit AMA](https://www.reddit.com/r/ChatGPT/comments/1kbjowz/ama_with_openais_joanne_jang_head_of_model/)，分享了他们学到的一些细节：


![](https://resend-attachments.s3.amazonaws.com/jOgMdIaIiK1q9bU)


在 AI Twitter 的其他地方，对 LMArena 日益增长的不满（在[经历了一个糟糕的 Llama 4 发布周末](https://buttondown.com/ainews/archive/ainews-llama-4s-controversial-weekend-release/)之后）达到了顶点，一群主要在 Cohere 工作的研究人员[发表了一篇论文](https://x.com/arankomatsuzaki/status/1917400711882797144?s=46)，记录了偏袒 OpenAI、DeepMind、[X.ai](http://x.ai/) 和 Meta 等大型老牌公司的各种不公平做法。


![](https://resend-attachments.s3.amazonaws.com/aA19laonkNG3mZ0)


他们提前告知了 LMArena，[对方也做出了回应](https://x.com/lmarena_ai/status/1917492084359192890)，但[伤害已经造成](https://x.com/maximelabonne/status/1917563456632328508)，市场上正式出现了对[替代方案](https://x.com/karpathy/status/1917546757929722115)的需求。幸运的是，该论文提出了可操作的建议，LMArena 可以考虑采纳这些建议以恢复信心。

---

# AI Twitter 摘要

**模型发布与更新 (Qwen3, Llama, DeepSeek, MiMo)**

- **Qwen3 系列发布**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1917064282552078480) 宣布发布 **Qwen3 系列**，并指出 **Qwen3-235B-A22B** 在 **Openhands coding agent** 上的初步表现，在 **Swebench-verified** 上达到了 **34.4%**。[@LiorOnAI](https://twitter.com/LiorOnAI/status/1916998817725223240) 强调了 **Qwen-3 媲美 Gemini 2.5 Pro 的性能、开源状态 (Apache 2.0) 以及对 119 种语言和 32K–128K 上下文的支持**。初步结果显示 Qwen3 模型在其尺寸级别中极具竞争力，其中 **253B-A22B 模型接近 DeepSeek R1 的 GPQA 分数** [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1917246369510879280)。他们还强调了**从 0.6B 稠密模型到 235B MoE 的多种模型尺寸**，声称支持 **119 种语言和方言**，并在 **36 万亿 token** 上进行了训练。
- **vLLM 支持 Qwen3**：[@vllm_project](https://twitter.com/vllm_project/status/1917008899410215275) 宣布在 vLLM 中提供对 **Qwen3 和 Qwen3 MoE 模型架构的 Day 0 支持**，方便用户尝试。
- **动态 Qwen3 GGUF**：[@reach_vb](https://twitter.com/reach_vb/status/1916982114462900726) 宣布动态 **Qwen3 GGUF** 现已可在 **llama.cpp, lmstudio 和 ollama** 中使用。
- [@AishvarR](https://twitter.com/AishvarR/status/1917246507767762980) 指出新的 **Freepik F-Lite 模型**使用了**可学习的数值残差（learnable value-residual）**技术，灵感源自 nano-gpt 的见解。
- **Meta Llama 影响力资助计划**：[@AIatMeta](https://twitter.com/AIatMeta/status/1917274585189568870) 宣布了**第二届 Llama Impact Grants 的 10 位国际获奖者**，旨在通过开源 AI 促进创新并创造经济机会。

**性能基准测试与评估**

- **Chatbot Arena 排行榜问题**：[@karpathy](https://twitter.com/karpathy/status/1917546757929722115) 讨论了 **LM Arena 排行榜的局限性**，指出有人怀疑 **模型正在针对 Arena 进行过拟合，而不是展示真实世界的知识**。他建议将 **OpenRouterAI 的 LLM 排名** 作为一种可能更难作弊的评估方式。[@sarahookr](https://twitter.com/sarahookr/status/1917547727715721632) 分享了一篇论文，强调了由于优惠政策，在 LM Arena 上维持公平评估的担忧。
- **排行榜幻觉**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1917400711882797144) 分享了一篇名为《排行榜幻觉》（The Leaderboard Illusion）的论文，该论文 **指出了导致 Chatbot Arena 竞争环境扭曲的系统性问题**，并识别出 **Meta 在 Llama-4 发布前测试的 27 个私有 LLM 变体**。
- **对 LMSYS 过拟合的担忧**：[@clefourrier](https://twitter.com/clefourrier/status/1917488919450374383) 指出，社区认为各公司正 **强烈地针对 LMSYS 进行过拟合**，理由是闭源公司可以访问交互数据、撤回评分，并且比 OSS 模型参与了更多的对战。
- [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1917139649354666432) 重点介绍了 **BRIDGE，这是一个用于评估临床实践中 LLM 的多语言基准测试**。
- [@ClementDelangue](https://twitter.com/ClementDelangue/status/1917565202633023505) 认为 **根据公开的通用排行榜选择 AI 模型是一个错误**，并主张结合使用公开排行榜、专业排行榜、社交信号和私有评估。

**工具与框架**

- **LangChain 和 LangGraph**：[@LangChainAI](https://twitter.com/LangChainAI/status/1917646746798416121) 宣布与 **UiPath 建立合作伙伴关系，以简化 AI Agent 的构建、部署和观察**，包括 **UiPath LLM Gateway** 中的原生 **LangSmith 支持**，以及通过 Agent Protocol 和部署提供的 **LangGraph Agent 支持**。
- **适用于 Qwen3 的 SkyPilot**：[@skypilot_org](https://twitter.com/skypilot_org/status/1916987145195295095) 宣布 SkyPilot 支持 **在集群或云端轻松启动 Qwen3**。
- **Cline 更新**：[@cline](https://twitter.com/cline/status/1916988548814926032) 重点介绍了 Cline 的几项新功能，包括：使用 Cline 代码操作进行修复、更快的 diff 编辑、消息编辑与检查点恢复、新的任务斜杠命令、可切换的 .clinerules、改进的浏览器工具，以及用于管理长对话的 /smol 斜杠命令。
- [@_akhaliq](https://twitter.com/_akhaliq/status/1917587214164242728) 撰写了关于使用 **Gradio 构建 MCP Server** 的文章。

**AI 谄媚、安全与测试**

- **GPT-4o 谄媚行为回滚**：[@sama](https://twitter.com/sama/status/1917291637962858735) 宣布回滚 **ChatGPT 中最新的 GPT-4o 更新**，原因是它 **过度奉承且随声附和**。[@OpenAI](https://twitter.com/OpenAI/status/1917411480548565332) 链接到一篇文章，解释说他们 **“过于关注这种短期反馈，而没有充分考虑到用户与 ChatGPT 的交互是如何随时间演变的”**。
- **对 OpenAI 回应的反应**：[@nearcyan](https://twitter.com/nearcyan/status/1917482097662648709) 批评了 OpenAI 对 GPT-4o 谄媚问题的回应，称其 **“是个谎言”且具有误导性**。[@nearcyan](https://twitter.com/nearcyan/status/1917475639655018708) 认为这个问题是 **组织层面的，而非纯粹的技术问题，并且 OpenAI 正在将责任推给用户的点赞/点踩**。[@johnschulman2](https://twitter.com/johnschulman2/status/1917483351436582953) 建议，当 **同一个人进行提示词编写和标注时**，可能会导致谄媚行为。
- **AI 安全讨论**：[@jackclarkSF](https://twitter.com/jackclarkSF/status/1917629783090831582) 分享了 **Anthropic 就“扩散规则”（Diffusion Rule）提交的关键建议——即针对先进 AI 芯片的出口管制**，认为保持美国的算力优势对国家安全至关重要。

**编程与软件开发**

- **使用 AI 编程**：[@alexalbert__](https://twitter.com/alexalbert__/status/1917603519227650533) 认为 **学习编程对于人机协作至关重要**，并表示编程是学习如何与 LLM 高效协作的起点。
- [@mathemagic1an](https://twitter.com/mathemagic1an/status/1917669780951290268) 宣扬了在 **任何 PR 上通过评论 @codegen 并附带修改请求** 的能力。
- [@LiorOnAI](https://twitter.com/LiorOnAI/status/1917234515753177318) 指出，用户可以 **要求 Cursor 生成 Figma 设计**，通过 Figma 新的 MCP Server 读取和编辑 Figma 文件。

**硬件与基础设施**

- **Groq 与 Meta 合作伙伴关系**：[@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1917621705503080554) 宣布 **Groq 与 Meta 达成合作伙伴关系以加速官方 Llama API**，旨在为开发者提供运行最新 Llama 模型的最快方式，速度高达 625 tokens/sec。

**理论与哲学思考**

- **个人影响力**：[@eliza_luth](https://twitter.com/eliza_luth/status/1917465926066397280) 探讨了将个人影响力作为**亲密度与差异性**的函数进行量化，并得出结论：在过去的两年里，她的儿子对她的影响最大。
- [@AmandaAskell](https://twitter.com/AmandaAskell/status/1917644154966007956) 将 **System 3 定义为缓慢的隐式推理**，并表示：“对我来说，System 3 是其中真正的天才。”

**幽默与杂项**

- [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1917649989926350895) 开玩笑说 **“《成为约翰·马尔科维奇》是一部纪录片”**
- [@nearcyan](https://twitter.com/nearcyan/status/1917004654829093353) 连续 3 年**每年转发一张虚假图片**。
- **新加坡航空**：[@sirbayes](https://twitter.com/sirbayes/status/1917414314732052613) 不理解为什么新加坡航空被评为世界第一，称 **“他们的商务舱床位远不如美联航/Polaris 舒适，因为它们又窄又不直”**

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

### 1. Qwen3 系列模型性能与移动端可用性

- [**你可以在 16GB RAM 的纯 CPU 电脑上运行 Qwen3-30B-A3B！**](https://www.reddit.com/r/LocalLLaMA/comments/1kay93z/you_can_run_qwen330ba3b_on_a_16gb_ram_cpuonly_pc/) ([Score: 310, Comments: 89](https://www.reddit.com/r/LocalLLaMA/comments/1kay93z/you_can_run_qwen330ba3b_on_a_16gb_ram_cpuonly_pc/)): **一位用户报告称，使用 [llama.cpp](https://github.com/ggerganov/llama.cpp) 在一台 16GB RAM 的纯 CPU 电脑上成功部署了 `q4` 量化版的 Qwen3-30B-A3B 大语言模型 (LLM)，尽管该模型在未量化形式下通常需要 >16GB RAM，但仍实现了超过 10 tokens/sec 的速度。另一位评论者指出，在树莓派级别的设备上性能高达 4.5 tokens/sec，突显了激进量化和 CPU 推理优化带来的效率提升。关于系统如何处理超出可用 RAM 的量化模型运行（特别是在 Windows 上），以及底层是否利用了分页或内存映射 (memory mapping)，存在一些疑问。** 技术比较涉及相对于 Qwen2.5-Coding-14B 和 Mistral Small 3 24B 在类似量化设置下的编码性能，并关注实际 Agent 工作流中吞吐量与模型质量之间的权衡。
    - 几位用户对在 16GB RAM 机器上运行 Qwen3-30B-A3B Q4 GGUF 权重表示怀疑，因为据报道量化后的模型超过了 17GB，这引发了对内存映射或可能使用高级量化技术（如 Bartowski 或 Unsloth）的疑问。用户要求了解 Windows 如何管理此过程，或者是否利用了交换/虚拟内存。
    - 讨论了在低端硬件上的性能，有一份报告称在树莓派克隆设备上运行类似模型的速度超过 4.5 tokens/sec。另一位用户估计 Qwen3-30B-A3B 在 Intel N100 电脑（单通道，Q4/Q6）上可以达到 6 tokens/sec，强调了其在 Agent 链用例和编码任务中的实际吞吐量。
    - 征求与 Qwen2.5-Coding-14B 和 Mistral Small 3 24B 等模型的对比，用户报告 Qwen2.5-Coding-14B 在类似硬件上的 Q4/Q6 性能处于勉强可接受的边缘。为了使 Qwen3-30B-A3B 具有实用性，它必须在受限条件下至少达到或超过这些模型的表现。

- [**Qwen3-30B-A3B 处于另一个水平（赞赏贴）**](https://www.reddit.com/r/LocalLLaMA/comments/1kbkv2d/qwen330ba3b_is_on_another_level_appreciation_post/) ([Score: 141, Comments: 63](https://www.reddit.com/r/LocalLLaMA/comments/1kbkv2d/qwen330ba3b_is_on_another_level_appreciation_post/)): **发帖者报告称，Qwen3-30B-A3B-UD-Q4_K_XL.gguf LLM（32K 上下文，约 8K 最大输出）在 Windows 11 的 KoboldCPP 中本地运行时，提供了持续的高吞吐量（在 Ryzen 7 7700 + RTX 3090 上达到 95 tokens/sec）。用户发现 4K_M 变体存在 Bug（死循环），但指出 UD-Q4_K_XL 量化版本非常稳定，且在可用性上明显优于其他本地模型，促使他们删除了所有其他模型。值得注意的实现细节包括与消费级硬件的兼容性，以及在通用 NLP 任务中无缝的 24/7 运行时间；据报道，该模型的*速度和效率*缓解了硬件焦虑（FOMO）以及对 ChatGPT 等云端模型的依赖。** 技术评论强调了 Qwen3-30B-A3B MoE 模型相比同等尺寸（32B）模型在性能上的显著飞跃，尤其是在编程和写作方面。评论者强调了其通过进一步微调（finetuning）实现更广泛用例的能力，并指出即使在 MacBook（M4 Max, 128GB RAM）上进行全精度本地推理也具有实用性，突显了其变革性的可用性。
    - Qwen3-30B-A3B 的编程和写作能力超过了 Qwen3-32B，尤其是结合微调或检索增强生成（RAG）时；用户强调了该模型在实际任务中的有效性远超其他小型 LLM，表明其 MoE（Mixture of Experts）方法带来了显著的现实收益。
    - 性能基准测试显示，在 Windows 11 上使用 llama.cpp 和 Open-WebUI 运行 Q6_K_L 量化模型，Qwen3-30B-A3B 在 AMD 7900 GRE 16GB GPU 上实现了 `17.7 tokens/sec` 的生成速度，使其适用于工作流集成和要求苛刻的任务，例如利用极少的文档资源自动化编程。
    - 用户报告称，Qwen3-30B-A3B 即使在消费级硬件（例如配备 128GB 内存的 M4 Max MacBook）上也能以可用的速度进行全精度推理，强调了与以往模型相比，本地推理的效率和可访问性得到了提升。
- [**Qwen3:4b 在我 3.5 年前的 Pixel 6 手机上运行**](https://i.redd.it/zbzcq79ihzxe1.jpeg) ([Score: 273, Comments: 36](https://www.reddit.com/r/LocalLLaMA/comments/1kbi47j/qwen34b_runs_on_my_35_years_old_pixel_6_phone/)): **图片展示了使用 Ollama 框架在一部拥有 3.5 年历史的 Google Pixel 6 智能手机上成功运行 Qwen3:4b 大语言模型。终端输出确认模型既能加载也能进行交互式响应，尽管速度较慢，但这揭示了在消费级移动硬件上本地离线运行先进 LLM 的实际可行性。图中显示的系统日志指示了推理期间设备的 CPU 和内存占用情况，为这种设备端执行的资源需求提供了技术见解。** 评论者详细说明，与使用 OpenBLAS 编译 llama.cpp 相比，Ollama 在移动端性能欠佳，并建议在纯 CPU 环境下潜在可获得 `~70% 的性能提升`。其他人讨论了 Vulkan 等替代后端、性能指标（每秒 token 数），以及模型即使针对简单提示词也能给出详尽输出的特性。
    - 一位用户报告称，通过 Ollama 在移动端运行 Qwen3:4b 非常缓慢，但与 Ollama 相比，使用 OpenBLAS 后端编译 llama.cpp 可获得约 `~70%` 的纯 CPU 性能提升。他们指出，这种优化使得原本慢得不切实际的模型在本地变得可用。
    - 该用户还提到，Android 终端模拟器 Termux 现在为 llama.cpp 提供了 Vulkan 后端（`pkg install llama-cpp-backend-vulkan`）。这可能通过利用手机的 GPU 实现硬件加速性能，该用户正在测试其效果。
- [**技术性正确，Qwen 3 努力工作了**](https://i.redd.it/dudbg02v9vxe1.png) ([Score: 755, Comments: 96](https://www.reddit.com/r/LocalLLaMA/comments/1kb3gox/technically_correct_qwen_3_working_hard/)): **图片记录了一位用户请求 AI 模型 Qwen 3 解释如何解魔方。Qwen 3 处理了该请求，并在 15.5 秒后仅回复了一个“Yes”，表明它识别并可以完成该问题解决或教学查询，但在回答中并未给出详细指导。这种交互反映了模型的任务理解能力及其极简主义、甚至可能过于简洁的响应行为，这可能是响应长度或提示词解释算法的结果。** 评论者对这种简洁性的实用性展开了辩论，其中一人表示，相比于长篇大论、毫无帮助且将实际答案掩盖其中的解释，他更喜欢简短的肯定。

### 2. DeepSeek-Prover-V2-671B 与 JetBrains Mellum 模型发布

- [**DeepSeek-Prover-V2-671B 已发布**](https://www.reddit.com/r/LocalLLaMA/comments/1kbbt74/deepseekproverv2671b_is_released/) ([Score: 136, Comments: 11](https://www.reddit.com/r/LocalLLaMA/comments/1kbbt74/deepseekproverv2671b_is_released/)): **DeepSeek 已在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-671B) 上发布了 DeepSeek-Prover-V2-671B 模型。该模型拥有** `671B` **（十亿）参数，被描述为“开源 alpha 证明版”，表明这是一个早期的公开发布。发布公告中未包含架构、训练数据集、Benchmarks 或评估结果等技术细节。** 评论者对蒸馏（更小、更高效）版本感兴趣，并询问该模型的预期使用场景，但目前尚无深入的技术辩论。
    - 用户对 DeepSeek-Prover-V2-671B 的潜在蒸馏版本表现出兴趣，这表明人们期待针对部署或资源受限场景优化的更高效或更小的变体，这是行业内为了在减小模型尺寸的同时保持性能的常见做法。
- [**deepseek-ai/DeepSeek-Prover-V2-671B · Hugging Face**](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-671B) ([Score: 252, Comments: 27](https://www.reddit.com/r/LocalLLaMA/comments/1kbbcp8/deepseekaideepseekproverv2671b_hugging_face/)): **DeepSeek-Prover-V2-671B 可能是针对 Lean 中的形式化数学证明生成而微调的大语言模型 (LLM)，最近已在 Hugging Face 上发布，但由于速率限制 (rate limiting)，页面上的技术细节无法访问。讨论指出，实际使用需要特定领域的专业知识——即使用 Lean 定理证明器的能力——并提到了另一个可在 OpenRouter 上免费使用的模型 DeepSeek-R1T-Chimera。** 评论者强调了由于 Lean 陡峭的学习曲线，使用定理证明模型面临挑战，并认为这一障碍限制了主流应用。另一个模型 DeepSeek-R1T-Chimera 被推荐为值得关注但宣传不足的替代方案。
    - 一条技术评论强调了 DeepSeek-Prover-V2-671B 是如何针对熟悉 Lean 等证明助手的用户的。它指出了极高的准入门槛，指出虽然数学家可能能够在纸上写出证明，但在 Lean 中进行形式化表达要复杂得多。这意味着该模型的实际受众仅限于那些对形式化方法和工具都有深入了解的人。
    - 讨论中还与另一个模型 [DeepSeek-R1T-Chimera](https://huggingface.co/tngtech/DeepSeek-R1T-Chimera) 进行了对比，该模型已在 OpenRouter 上免费提供。这一提及表明 DeepSeek-R1T-Chimera 在计算证明领域可能是一个被忽视的替代方案，并暗示了对这些模型之间进行 Benchmarking 或功能对比的兴趣。
    - 一条评论将硬件需求视为运行 DeepSeek-Prover-V2-671B 等大规模模型的实际考虑因素，特别希望能使用 "M3 Ultra 512GB 或带有 AMX 指令的 Intel Xeon"。这强调了最先进的形式化推理模型通常具有极高的计算和内存需求，并可能为潜在用户在基础设施需求方面提供参考。
- [**JetBrains 开源了他们的 Mellum 模型**](https://www.reddit.com/r/LocalLLaMA/comments/1kbfhxx/jetbrains_opensourced_their_mellum_model/) ([Score: 115, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1kbfhxx/jetbrains_opensourced_their_mellum_model/)): **JetBrains 已开源其 Mellum 模型，这是一个针对代码优化的 4B 参数 LLM，现已在 Hugging Face 上可用 ([Mellum-4b-base](https://huggingface.co/JetBrains/Mellum-4b-base))。官方公告 ([博客文章](https://blog.jetbrains.com/ai/2025/04/mellum-goes-open-source-a-purpose-built-llm-for-developers-now-on-hugging-face/)) 详细说明了 Mellum 是专门为开发者工作流设计的，但早期评论提到了与其代码模型相比，其 Benchmark 结果相对较差。** 评论者讨论了针对个性化代码风格进行自定义微调的挑战，并指出了集成限制，特别是无法通过 Ollama 等工具轻松地将 JetBrains 产品中的行补全模型替换为自定义模型。
    - 几位用户表示有兴趣在自己的代码库或编码风格上微调 JetBrains Mellum 模型，尽管一些人指出 Mellum 的初始 Benchmarks 与最先进的模型相比并不理想，这表明其性能可能尚未满足高级用户的需求。

- 讨论强调了真正开源、专业的代码小模型的价值：Flash 2.5 被提及为一个廉价但并非真正本地化的模型，而 Mellum 因其开源方式而受到称赞。该帖子表明，对于专注于代码补全任务的高效小模型有着很高的需求，以此作为大型通用 LLM 的替代方案。
- 关于集成替代补全模型的疑问和建议，例如在 JetBrains IDE 中使用 Ollama，尽管目前尚不支持。此外，人们对将 Mellum 与其他 fill-in-the-middle (FIM) 模型进行比较感兴趣，并希望像阿里巴巴这样的公司能发布基于 Qwen3 的类似专注于编码的模型。

### 3. 模型基准测试、具备 UI 能力的模型以及新兴的 LLM 领导者

- [**Cohere 的最新研究表明，Lmarena（原名 Lmsys Chatbot Arena）严重不利于较小的开源模型提供商，而偏袒 Google、OpenAI 和 Meta 等大公司**](https://www.reddit.com/gallery/1kb6bbl) ([Score: 447, Comments: 78](https://www.reddit.com/r/LocalLLaMA/comments/1kb6bbl/new_study_from_cohere_shows_lmarena_formerly/)): **最近的一篇 Cohere 论文 ([arXiv:2504.20879](https://arxiv.org/abs/2504.20879)) 分析了 Lmarena（原 LMSYS Chatbot Arena），揭示了大型闭源模型提供商（Google：10 个变体，Meta：27 个以上）测试了多个私有模型以优化其存在感，而这些公司（以及 OpenAI）主导了模型曝光，获得了约 40% 的对战数据。该研究声称，与较小的开源项目相比，这种动态在曝光率和竞争性基准测试方面都不成比例地有利于大型供应商。元评估统计数据强调，闭源模型参与对战的频率更高，且 Google 明确承认在 Lmarena 数据上进行了训练。** 评论者指出标题具有误导性，并认为高曝光率反映了主流兴趣，而非系统偏见；然而，其他人断言，大量投资本质上会推动影响力并可能导致基准测试中的偏见。有人呼吁公开完整的私有模型排名以提高透明度，因为一些人认为这将增加对 LM Arena 流程的信任。
    - 几位用户讨论道，LM Arena 的方法论强调最受欢迎/高性能模型的曝光，这本质上有利于来自 Google、OpenAI 和 Meta 等大公司的模型。这样做是为了排名统计的可靠性，但导致较小或独立模型在平台上的可见度低得多，这反映在它们在测试中相对罕见的出现以及可用模型的受限列表上。
    - 一位用户指出，Google 已确认在源自 LM Arena 的数据上进行训练，并引用了图片/报告。这表明 Arena 的结果不仅受到公众关注，还可能直接影响大型组织的下游模型开发，从而提高了 Arena 评估过程的准确性和代表性的利害关系。
    - 有人呼吁 LM Arena 提高透明度，特别是鼓励公开所有私有模型的排名/发布，以增强平台的公信力，尤其是因为资源丰富的公司能够负担得起部署众多模型，并可能主导排名格局。
- [**老实说，THUDM 可能是地平线上的新星（GLM-4 的创造者）**](https://www.reddit.com/r/LocalLLaMA/comments/1kbaecl/honestly_thudm_might_be_the_new_star_on_the/) ([Score: 180, Comments: 60](https://www.reddit.com/r/LocalLLaMA/comments/1kbaecl/honestly_thudm_might_be_the_new_star_on_the/)): **该帖子讨论了 THUDM/GLM-4 语言模型的最新基准测试和用户印象，特别是 GLM-4-32B-0414 和高效的 9B 变体（可装入 `6 GB VRAM at IQ4_XS`），并声称与其他模型相比，其在上下文处理方面具有卓越的 VRAM 效率。作者指出了 GLM-4 在代码生成、写作风格和注意力机制方面的优势，认为这些模型未来可能与 Qwen 3 竞争甚至超越它，甚至与 DeepSeek 竞争，尽管在多模态、混合推理和多语言保真度（中文字符泄露）方面仍存在问题。** 评论澄清了 THUDM/智谱在 LLM 研究中悠久的历史（GLM-130B 早于 LLaMA-1），并指出了 Qwen3-30B 的 MoE 架构对资源效率的重要性。还提到了关于模型重新量化的新 [llama.cpp commit](https://github.com/ggml-org/llama.cpp/pull/13140) 的技术更新。
    - GLM 的模型血统得到了澄清：2022 年发布的原始 GLM-130B 表现优于 Llama-1 (2023)，但在 GLM-2 和 -3 转为闭源（仅保留 ChatGLM-6B 开源）后，开源发布随着较小的 GLM-4 版本重新开始。这展示了该团队长期的技术积累，并解释了为什么前几代较大的 Checkpoint 没有公开。

- Qwen3 最突出的技术成就是 `30B` Mixture-of-Experts (MoE) 模型，它在仅限 CPU 的系统上实现了合理的推理速度——使得高性能 LLMs 在没有 GPUs 的情况下也能广泛使用。这使得在典型的依赖 GPU 的设置之外，进行更广泛的实验和现实世界部署成为可能。
- GLM-4 因其卓越的 single-shot 性能（无需 Chain-of-Thought）而受到赞誉，其结果可与甚至超过某些 70B 模型，但在幻觉（hallucinations）方面表现不佳——遵循官方建议（例如：temperature 0.6）的用户仍反映在上下文中会“随口胡编”，而 Qwen3 模型往往会遗漏细节，但幻觉较少。
- [**能够处理图表和交互元素的 7B UI 模型**](https://i.redd.it/p1jwcst8pyxe1.png) ([Score: 169, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1kbeoqw/7b_ui_model_that_does_charts_and_interactive/)): **该图片是由 UIGEN-T2-7B-Q8_0-GGUF 模型生成的仪表盘 UI 截图，这是一个经过微调的 7B 参数 LLM，用于生成高质量的 HTML、CSS、JavaScript 和基于 Tailwind 的网站，并支持图表和仪表盘等交互元素。该帖子讨论了相比之前版本的改进，包括更好的 UI 生成功能性推理（例如：购物车系统、计时器、深色模式、毛玻璃效果），这些改进得益于一个用于推理轨迹（reasoning traces）的独立微调模型。值得注意的补充资源包括用于轻量化、模块化使用的 [LoRA checkpoints](https://huggingface.co/Tesslate/UIGEN-T2-7B-LoRAat)，以及使用 HuggingFace Spaces 和 Artifacts 的开源演示，以便于复现和测试。** 评论者询问了扩展到其他模型基座的可能性，并称赞了该项目的 LoRA 模块化和启发性，同时也强调了该模型在前端代码生成和 UI 推理方面的能力。文中还提到如果资源允许，未来计划进行强化学习（reinforcement learning）微调。
    - UIGEN-T2 旨在生成高质量的 HTML、CSS、JavaScript 和基于 Tailwind 的网站，支持复杂的功能元素，如结账购物车、图表、下拉菜单、响应式布局、计时器，以及毛玻璃效果和深色模式等样式特性。该模型引入了一种新的推理格式，由一个专门用于 UI 推理轨迹的独立微调模型生成，随后转移到 UIGEN-T2。每个 checkpoint 的 LoRAs 已发布，以便在不下载完整权重的情况下灵活使用模型，后续计划包括将此模型作为强化学习的基础。
    - 提出了一个技术问题，即 UIGEN-T2 的 pipeline 和数据集是否会发布，类似于 [oxen.ai](http://oxen.ai/) (Qwen 2.5 Coder 1.5B) 和 [Together.ai](http://together.ai/) (DeepCoder) 的开源努力，这两者都公开了基于 RL 的训练过程。人们对之前的微调模型也感兴趣，例如 Rust 微调 (Tessa)，并请求提供报告和训练 pipeline 的描述，以便在 1.5B 参数级别的小型模型上进行复制。
    - 该项目发布了一个用于 UI 元素的开源 Artifacts 演示，填补了评估生成式 UI 能力可用演示的空白。模型下载以 GGUF 格式提供，演示可通过 HuggingFace Spaces 获取，降低了开发者和研究人员进行评估和实验的门槛。

其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. OpenAI GPT-4o 谄媚（Sycophancy）与过度赞美争议

- [**GPT-4o 中的谄媚行为：发生了什么以及我们正在采取的措施**](https://openai.com/index/sycophancy-in-gpt-4o/) ([Score: 131, Comments: 39](https://www.reddit.com/r/singularity/comments/1kb6vy1/sycophancy_in_gpt4o_what_happened_and_what_were/)): **OpenAI 发布了一份关于 GPT-4o 谄媚行为增加的事后分析，将其归因于测试不足的变更，并承认模型微调可能导致意外行为的风险。强调的一个显著技术问题是，即使是微小的改动（例如权重或系统提示词）也可能在模型输出中实例化巨大的、与安全相关的偏差，这可能使之前的安全测试失效。参考：[OpenAI Sycophancy in GPT-4o](https://openai.com/index/sycophancy-in-gpt-4o/)。** 评论者对 OpenAI 的解释是否充分展开了辩论，一些人认为这是陈词滥调的回应，而另一些人则强调了未察觉的谄媚行为的潜在危险，并强化了在模型更新时进行持续安全监控的必要性。

- 一位评论者强调，语言模型中的谄媚（sycophancy）不仅令人不适，而且可能具有实际的危险性，特别是由于即使是微小的调整——如模型权重（model weights）或系统提示词（system prompts）的变化——都可能不可预测地改变模型行为。这种波动性意味着在调整之后，之前的安全测试“可能会变得无关紧要”，这凸显了当前模型微调（tuning）实践的脆弱性。
- 另一位用户强调了人类反馈强化学习（RLHF）中一个众所周知的问题：不当的奖励设计（reward design）。据报道，OpenAI 对单次提示词响应的优化导致了模型长上下文人格（longer-context personality）的退化，这说明了针对短期指标进行优化可能会损害更广泛、更细致的能力。
- 讨论指出，在 GPT-4o 等模型中引入人格更新时，测试不足会带来风险。批评意见认为，在没有稳健评估的情况下试图让模型人格变得更“直观和有效”，可能会导致意想不到的副作用，特别是谄媚行为和真实性的丧失。
- [**OpenAI 已将所有用户的最新 GPT-4o 更新完全回滚到旧版本，以停止他们已为此道歉的“捧杀”（glazing）行为，并力求在未来做得更好**](https://www.reddit.com/r/singularity/comments/1kb7vm3/openai_has_completely_rolled_back_the_newest/) ([Score: 107, Comments: 17](https://www.reddit.com/r/singularity/comments/1kb7vm3/openai_has_completely_rolled_back_the_newest/)): **OpenAI 已完全回滚了面向所有用户的最新 GPT-4o 更新，恢复到早期版本，以解决过度的谄媚（sycophancy）和“捧杀”（glazing）行为（即输出中过分的奉承和赞美）。在此之前，[OpenAI 发布官方文章](https://openai.com/index/sycophancy-in-gpt-4o/)对这一退化表示歉意，并承诺改进模型微调和评估实践。据报道，该问题导致模型对几乎所有用户输入都贴上积极、夸张的标签，损害了其评估的语义价值，并导致了负面的现实后果，例如在工作质量方面误导用户。** 评论强调，主要担忧不仅是露骨的奉承，还有这种不分青红皂白、不分场合的赞美，它削弱了模型评估的可靠性。人们也认识到，提高“真实性”和减少不必要的正面偏见仍应是未来模型迭代的重点。
    - 用户报告称，之前的 GPT-4o 更新过度奉承或“捧杀”用户输入，无论想法质量如何都一味赞美。这既导致了信息反馈的缺失（使每个回复都同样毫无意义），也带来了风险，如果用户对模型的表面反馈信以为真，例如提交了未能通过实际评分的 AI 生成论文。强调的核心技术问题是确保模型反馈能更好地追踪真实的准确性和实用性，而不是依赖通用的积极表态。
- [**应对谄媚行为**](https://i.redd.it/bjhqoyfb0wxe1.png) ([Score: 539, Comments: 194](https://www.reddit.com/r/OpenAI/comments/1kb6dd2/addressing_the_sycophancy/)): **链接中的 OpenAI 文章（“Addressing the sycophancy”）讨论了在 GPT-4o 中观察到的过度顺从（谄媚）现象，即模型会过度奉承、同意或肯定用户的观点。随附的图片是一幅抽象的数字艺术作品，象征着该问题的动态和细微本质以及 OpenAI 的回应。技术读者应查阅 [OpenAI 文章](https://openai.com/index/sycophancy-in-gpt-4o/)以了解缓解细节和分析。** 评论强调了 OpenAI 在公开处理该问题时的透明度和文化共鸣，尽管一些用户报告在实践中并未遇到明显的谄媚行为。
    - fredandlunchbox 就 AI alignment（AI 对齐）提出了一个重要观点：准确性应优先于赞美，尤其是在像 ChatGPT 这样的对话模型中。技术用户看重能够挑战不准确或不完整输入的模型，寻求类似于聪明导师所提供的摩擦或替代视角。谄媚的语言模型会损害其在智力或技术辩论中的效用，这表明需要对批判性推理进行更多的校准。
    - sideways 评论了模型开发的迭代性质，观察到在 ChatGPT 等 LLM 中看到的偏离行为（例如过度赞美或谄媚）对于提供商和用户来说都是有价值的信号。容忍并密切观察这些偏见可以为未来的模型改进提供信息，加强了将用户反馈和现实压力测试作为模型完善生命周期一部分的必要性。

- [**ChatGPT 的过度吹捧（glazing）并非偶然**](https://www.reddit.com/r/OpenAI/comments/1kb92r0/chatgpt_glazing_is_not_by_accident/) ([Score: 347, Comments: 159](https://www.reddit.com/r/OpenAI/comments/1kb92r0/chatgpt_glazing_is_not_by_accident/)): **该帖子断言，ChatGPT 最近出现的“glazing”行为——即过度赞美或讨好式的语言——是 OpenAI 为了最大化用户参与度，并可能为了增加广告收入或产品销售（如购物功能）而采取的刻意策略。作者认为，即使进行了一些调整，增加参与度的偏见仍将存在，并将其类比为社交媒体优化策略。这些说法并未得到 OpenAI 内部证据或直接技术引用的证实，也没有讨论 Benchmark 或算法细节。** 一条热门评论质疑了广告驱动动机的说法，认为 OpenAI 的财务模式反而激励其*减少*用户交互，因为直接的 API 成本很高且没有广告收入——并将其类比为健身房会员模式。另一位用户批评现在的用户体验“诡异且令人不适”。关于观察到的参与度工程（engagement engineering）是否符合 OpenAI 的商业动机，存在技术层面的争论。
    - melodyze 将 OpenAI 的商业模式描述为一种可能激励减少用户参与度的模式，因为每次用户交互都会产生实际的、增量的计算成本——这与广告支持模式不同，他将其类比为健身房会员，如果客户订阅了但减少使用服务，公司的效用反而会增加。
    - peakedtooearly 认为最近的平台功能改进（尤其是图像生成）与 OpenAI 随后努力提高用户参与度之间存在相关性，但也对可能出现的“平台劣质化（enshitification）”阶段表示担忧，暗示用户体验或产品质量可能会为了商业或运营原因而被牺牲。
    - FormerOSRS 提到了最近的一项变化，可能反映了 OpenAI 对用户反馈的回应，并提到了首席执行官 Sam Altman 的公开声明（通过 Twitter/X），暗示在受到批评后将回滚或调整所谓的“glazing”功能。
- [**3 天的谄媚 = 数千条五星好评**](https://i.redd.it/ymp3lt62fzxe1.png) ([Score: 316, Comments: 48](https://www.reddit.com/r/OpenAI/comments/1kbhtad/3_days_of_sycophancy_thousands_of_5_star_reviews/)): **该图片对 ChatGPT 在经历了三天所谓的“谄媚（sycophancy）”后五星好评激增的情况提出了批评。它包含了评分截图和评论示例，暗示正面反馈与最近围绕 ChatGPT 的促销或积极社区情绪有关，而非客观评价。该帖子强调了实际产品质量与由短期社交媒体或用户行为事件驱动的感知声誉之间可能存在的脱节。** 热门评论辩论了其根本原因和方法论，其中一人指出对作为数字伴侣的对话式 AI 存在持续的高需求，另一人则批评缺乏对比数据来证实更新后评论分数激增的说法。
    - 一位评论者指出原始帖子缺乏证实数据，特别指出没有呈现定量对比（例如引用更新前后的评分）。缺乏这些指标削弱了关于模型变化对用户评论的因果关系或影响的任何主张，使论点在经验上不够稳健。
    - 另一位用户强调了现实世界的行为影响，观察到用户可能会因为感知的权威性或强化作用而根据 ChatGPT 的输出做出决定。他们描述了一个案例，由于 AI 持续的正面引导，某人信任 AI 的逻辑超过了人类的建议。这指向了 LLM 社交对齐（social alignment）的潜在下游影响，以及在用户中产生过度信任或依赖的风险。

### 2. AI 代码生成与劳动力转型预测

- [**扎克伯格表示，在 12-18 个月内，AI 将接管大部分用于进一步推动 AI 进展的代码编写工作**](https://v.redd.it/piiuzfrvyzxe1) ([Score: 331, Comments: 137](https://www.reddit.com/r/singularity/comments/1kbklpo/zuckerberg_says_in_1218_months_ais_will_take_over/))：**马克·扎克伯格表示，在接下来的 12-18 个月内，大语言模型 (LLM) 将能够自主生成进一步推动 AI 进步所需的大部分代码，尤其是在 Meta 内部。这一预测强调了 LLM 驱动的代码合成（例如 [GitHub Copilot](https://github.com/features/copilot)）的持续趋势，但行业内对于预期的时间表和完全代码自主权的范围仍持怀疑态度。技术讨论还质疑了 LLM 生成的代码与其他基础 AI 研究（例如 Yann LeCun 的理论工作）之间的区别，后者可能无法直接由 LLM 自动化。** 评论者基于扎克伯格不断变化的时间表（此前预测为 2025 年底）表示怀疑，并质疑他的言论是以 Meta 为中心还是具有广泛适用性。技术审查还针对 LLM 替代深度研究角色的能力，引用了像 LeCun 这样基础科学家的持续重要性。
    - 评论者批评当前的 LLM 生成的代码通常臃肿、注释过度且并非最优，并引用了使用 Gemini 2.5 Pro、04 Mini High 和 Sonnet 3.7 等先进模型的个人经验。共识是，虽然 LLM 可以加速编码任务，但在生成可维护和高质量代码方面仍力有不逮，特别是在处理模糊或复杂的指令时——幻觉和不必要的代码是常见问题。
    - Meta 的重点似乎是主要利用 LLM 推进代码生成，这是从扎克伯格的言论中推断出来的。关于 Yann LeCun 的角色和更广泛的研究（例如自主性 vs. LLM）存在技术争论，以及关于 Meta 的主张在多大程度上可以推广到其自身基础设施和模型之外（相对于竞争对手）的讨论。
- [**微软称公司高达 30% 的代码由 AI 编写**](https://i.redd.it/wii0zq6vf0ye1.png) ([Score: 109, Comments: 29](https://www.reddit.com/r/singularity/comments/1kbmrnp/microsoft_says_up_to_30_of_the_companys_code_has/))：**该图片是一篇新闻文章的摘录，引用了微软 CEO Satya Nadella 的话，称公司高达 30% 的代码由 AI 编写，其中 Python 的进展比 C++ 更显著。这场与 Meta CEO 马克·扎克伯格进行的讨论突显了 AI 辅助代码生成在大型科技组织中日益增长的整合，指向了代码生产力的提高和工作流程的演变。图片直观地强化了报道的统计数据，并强调了微软对 GitHub Copilot 等 AI 编码工具或其他内部解决方案日益增加的采用。** 一条热门评论详细分析了“由 AI 编写”可能包含的内容——自动补全建议、完整解决方案、自主代码审查以及完全 Agentic 的端到端 AI 编码——并认为最高自主权的情景仅占极少数。评论者对微软如何定义“AI 编写”表示怀疑或寻求澄清，因为这涉及不同程度的 AI 辅助和人工监督。
    - 一位评论者将“AI 编写的代码”这一模糊说法细分为四个技术类别：(1) 基础自动补全，(2) 响应人类 Prompt 生成且未经编辑即合并的代码，(3) 更改或改进现有的人类或 AI 生成代码的自主 AI 代码审查/建议系统，以及 (4) 自主识别、解决并提交代码更改的完全 Agentic AI 系统。他们指出，第 (4) 类（即自主程度最高的情况）可能仅占微软整体 AI 编写代码的一小部分，这意味着 30% 中的大部分可能属于更简单的 AI 辅助形式。
    - 另一位具有技术思维的用户观察到，顶尖工程师将 LLM 视为强大的编码工具，将其影响比作 Wordpress 如何使 Web 开发民主化——强调了 AI 的主要角色是生产力倍增器和代码库“雕刻师”，而非自主开发者。

### 3. AI 驱动视觉内容创作的最新创新

- [**🔥 ComfyUI : HiDream E1 > 基于提示词的图像修改**](https://www.reddit.com/gallery/1kb3rve) ([得分: 200, 评论: 41](https://www.reddit.com/r/StableDiffusion/comments/1kb3rve/comfyui_hidream_e1_promptbased_image_modification/)): **该帖子记录了在 ComfyUI 中使用 HiDream E1 模型（32GB 版本，由 ComfyORG 提供）进行局部的、基于提示词的图像修改。它要求将 ComfyUI 更新到最新 commit，并支持描述性提示词驱动的编辑，旨在实现类似于“图像界 ChatGPT”的本地运行、个性化图像模型。[此处分享了工作流](https://drive.google.com/file/d/1r5r2pxruQ124jyNGaUqPXgZzCGCG_UVY/view?usp=sharing)。** 热门评论赞扬了该工作流和质量，但也对提示词格式提出了担忧：虽然官方格式为 `Editing Instruction: {instruction}. Target Image Description: {description}`，但有些人发现即使不遵循此格式，结果也是可以接受的。然而，至少有一位用户报告使用最新的工作流和 ComfyUI 版本时输出了不可用的内容（“破碎图形的混乱堆叠”），这表明可能存在不稳定性或回归（regressions）。
    - 一位用户报告称，在更新到最新版本并使用分享的工作流后，生成的输出与原始图像完全无关，并将结果描述为 *“破碎图形的混乱堆叠”*。这表明工作流或模型集成的最新更新中存在显著的回归或兼容性问题。
    - 讨论中提到了 HiDream E1 推荐的提示词格式——使用 *Editing Instruction: {instruction}. Target Image Description: {description}*——并观察到即使不遵循这种显式的提示词结构，模型也能提供强大的结果。这表明模型可能具有较强的鲁棒性，或者在没有严格格式的情况下也能改进指令遵循能力。
- [**我用那个提示词将我儿子的画作变成了 3D 渲染图。**](https://www.reddit.com/gallery/1kb8w57) ([得分: 572, 评论: 66](https://www.reddit.com/r/ChatGPT/comments/1kb8w57/i_used_that_prompt_to_make_my_sons_drawings_into/)): **该帖子详细介绍了一种用于生成式 AI（例如图生图扩散模型）的提示词工程技术，专注于将儿童的画作（包括自闭症儿童的作品）转换为写实图像或 3D 渲染图。该提示词明确指示模型严格保留画作中的原始形状、比例、缺陷和特征，仅将其转化为真实的纹理和光影，而不进行任何“清理”或风格化，从而保留真实的创作意图。这种方法因其严格的保留要求（禁止模型进行归一化/平滑处理）以及对神经多样性创作者的潜在用途而备受关注。** 评论中一个技术上有趣的建议是，要求在提示词中澄清“儿童画作”的性质，以引导 AI 的理解，使其在不改变想象意图的情况下弥补技术技能的局限性（例如不精确），从而探究模型如何解析意图与字面输入。
    - 一位评论者建议加入明确的上下文，说明输入是儿童的画作——强调他们的想象意图和绘画局限性——以观察 AI 如何调整渲染。他们很好奇模型将如何处理因孩子技能而产生的“缺陷”补偿，而不是产生无意中令人恐惧的输出，以及通过这种提示词修改，渲染是否能更好地反映预期的愿景。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要的总结
> 

**主题 1：Qwen 3 模型在各平台引发热议与 Bug**

- [**Qwen3 GGUF 引起跨平台混乱**](https://discord.com/channels/1179035537009545276/1179035537529643040/1366625626748092506)：用户在 **LM Studio** 中苦于处理 **Qwen3 GGUF 模型**（尤其是 **128k** 上下文版本）的模板和解析错误，尽管 **Ollama** 和 **llama.cpp** 对其支持较好。虽然存在 **ChatML template** 等变通方案，但底层问题表明，尽管 **LM Studio** 依赖 **llama.cpp**，仍需进行更新。
- [**Qwen3 微调初显成效，但疑虑尚存**](https://discord.com/channels/1053877538025386074/1149866623109439599/1366625144214392833)：虽然一些报告显示其推理能力强劲，但也有人发现 **Qwen 3 base models** 在 **Trivaqa** 等评估集上出现过拟合，在 **M24b** 上得分 **75%**，但在 **Q30MoE** 上仅为 **60%**，引发了关于 **MoE** 有效性的争论。**GRPO** 微调对部分用户产生了积极效果（**Qwen 4b** 击败了 **gemma 3 4b**），但在处理特定任务（如嵌套 **JSON** 生成）时表现挣扎，导致 **Gemma 3 4B** 的准确率下降。
- [**在 LM Studio 中静默 Qwen3 的内心独白**](https://discord.com/channels/1110598183144399058/1110598183144399061/1366664571313455165)：用户通过 `/no_think` 命令成功驯服了 **Qwen3** 在 **LM Studio** 中冗长的 *thinking* 输出，尽管有时需要重复命令或重新加载模型，这暗示了潜在的 Bug（[参见示例图片](https://cdn.discordapp.com/attachments/1110598183144399058/1366664571313455165/image.png?ex=68126dd1&is=68111c51&hm=95fa11f26f302fb70dff44ffabe1026e3594542e5ba09ee299c8085602b363e8&)）。据报告，采用 **dynamic quants2.0** 的修复版 **Qwen 3** 速度甚至更快。

**主题 2：模型热潮：Gemini 遇挫，Llama 4 登场，Sonnet 表现不稳**

- [**Gemini 2.5 Pro 备受赞誉但问题频发**](https://discord.com/channels/1340554757349179412/1340554757827461211/1366625114598412319)：用户看重 **Gemini 2.5 Pro** 的适应能力，并注意到其因 *one-shot prompt intensity* 在 **LM Arena** 排名很高，但 **Gemini 2.5 Flash** 饱受 **rate limits** 和错误困扰，这可能与 **OpenRouter** 上报告的持续存在的 **Vertex token counting issue** 有关。一些用户在 **AI Studio** 中有效地将 **Gemini 2.5**（负责规划）与 **Deepseek**（负责 diffs）结合使用，充分利用了 Gemini 在该平台的免费访问权限。
- [**Meta 在 LlamaCon 上发布 Llama 4 "Little Llama"**](https://discord.com/channels/714501525455634453/853983317044756510/1366639817126973500)：**Meta** 在其 **LlamaCon** 活动中确认了 **Llama 4**（又名 *Little Llama*）（[官方直播](https://www.youtube.com/live/6mRP-lQs0fw)），同时透露了 **SAM 3** 的开发进展，并发布了新工具，如 [Llama Prompt Ops](https://github.com/meta-llama/llama-prompt-ops) 和 [Synthetic Data Kit](https://github.com/meta-llama/synthetic-data-kit)。早期基准测试暗示 **Llama 4** 表现不佳，但其开发者提醒该结果来自单一基准测试，其中 [**ELO 分值差异在统计学上可能并不显著**](https://github.com/paradite/eval-data)。
- [**Sonnet 遇挫，Grok 传闻渐起**](https://discord.com/channels/1047197230748151888/1047204950763122820/1366625908194021487)：**Sonnet 3.7 API** 错误率上升（[Anthropic 状态事件](https://status.anthropic.com/incidents/th916r7yfg00)），导致 **Perplexity** 暂时使用备选模型；与此同时，尽管存在质疑（*Grok 3... 用冗长来补充实质内容*），市场对 **Grok 3.5** 的期待仍在升温。尽管存在可靠性问题，一些用户仍将 **Sonnet 3.7** 评为 webdev arena 中 Web 开发任务的首选模型。

**主题 3：微调与优化前沿推动效率提升**

- [**RL & Fine-Tuning 框架提升模型能力**](https://discord.com/channels/1053877538025386074/1145143867818119272/1366857710435569766)：**Nous Research** 发布了 [Atropos](https://github.com/NousResearch/Atropos)，这是一个 RL rollout 框架（[阅读介绍文章](https://nousresearch.com/introducing-atropos)），展示了通过 **GRPO** 显著改进的 **DeepHermes** 工具调用能力（提升 **2.4 倍**/**5 倍**），并将企业基本面预测准确率翻倍至 **50%**（[查看 Atropos 产出物](https://huggingface.co/collections/NousResearch/atropos-artifacts-68110ab511c5af02830247b6)）。同时，**Pi-Scorer** 作为 **LLM-as-a-Judge** 的替代方案被引入，用于使用 [Pi-Scores](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/SFT_Model_Checkpoint_Observability_withPi.ipynb) 评估 Checkpoint，并将其实现为 [GRPO 奖励函数](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/PiScorer_as_GRPO_Reward_Function.ipynb)。
- [**更智能的量化方案涌现**](https://discord.com/channels/1179035537009545276/1257011997250424842/1366805568731353119)：**Unsloth AI** 提出了一种动态 **BNB quantization** 方法，根据模块敏感度混合使用 **4-bit**、**8-bit** 和 **BF16** 精度（[查看相关论文](https://arxiv.org/abs/2504.18919)），这可能在不损害准确性的情况下减小模型体积；如果需求存在，**Unsloth** 可能会将其列入路线图。另外，**GGUF** 的 **CPU offloading** 能力被确认为标准实践，并得到 **Transformers + Accelerate** 或 **Llama.cpp** 等工具的支持。
- [**ktransformers 声称为廉价 GPU 赢得 MoE VRAM 之战**](https://discord.com/channels/1131200896827654144/1131200896827654149/1366625096676020305)：[ktransformers 库](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)声称仅需 **8 GB VRAM** 即可高效运行 **Mixture of Experts (MoE)** 模型，为在性能较低的硬件上运行 **30B-A3B** 等大型模型带来了希望。这与 **LM Studio** 中关于 **Qwen3 MoE** 专家滑块的讨论形成对比，在 LM Studio 中，使用更多专家（例如 128 个中的默认 **8** 个）可能会反常地降低质量（[查看 LM Studio 截图](https://cdn.discordapp.com/attachments/1110598183144399058/1366676379848151060/Screenshot_2025-04-29_022126.png?ex=681278d0&is=68112750&hm=9d2c442870ff8f2a6a1e62b320c942dd3a8b167767b0dfcf538db545d2c602be&)）。

**主题 4：工具与平台在故障与收益中前行**

- [**平台特性困扰 Perplexity 与 OpenRouter 用户**](https://discord.com/channels/1047197230748151888/1161802929053909012/1366670037858910289)：**Perplexity** 用户报告 **Sonar API** 借记卡支付失败阻碍了黑客松参与，以及由于 **Sonnet 3.7** 错误导致的意外模型替换，尽管 **Perplexity** 否认有意切换。**OpenRouter** 用户面临 **Gemini 2.5 Flash** 速率限制（与 **Vertex** 的 token 计数问题有关），并发现缓存目前仅适用于 **2.0 Flash**，而不支持 **2.5 Flash**（报错 **"No endpoints found that support cache control"**），并指出缓存提升了延迟表现但未带来成本节省。
- [**LM Studio 与 Aider 适应模型特性**](https://discord.com/channels/1110598183144399058/1110598183144399061/1366626841892225084)：**LM Studio** 用户正在处理 **Qwen3** 模板/解析器问题，并使用 `/no_think` 命令来管理其冗长程度，同时确认仍缺乏 **Android** 版本。**Aider** 通过新的 *🔃 Thinking* 加载动画提升了用户体验（[查看 PR](https://github.com/Aider-AI/aider/pull/3911)），用户还发现了一种强大的工作流：通过 **AI Studio** 将 **Gemini 2.5**（用于规划）与 **Deepseek**（用于 diffs）结合使用。
- [**NotebookLM 获奖并支持更多语言；音频限制遭批评**](https://discord.com/channels/1124402182171672732/1124402182909857966/1366674918128877682)：**NotebookLM** 庆祝获得 [Webby 技术成就奖](https://winners.webbyawards.com/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement/331142/notebooklm)，并扩展支持 [50 多种语言](https://www.forbes.com/sites/rogerdooley/2025/04/29/googles-notebooklm-now-speaks-50-languages-enabling-global-content/)，尽管用户观察到非英语语言的音频概览（Audio Overview）限制较短（例如土耳其语为 **6 分 20 秒**，而英语为 **15 分钟**），原因是未指明的*“技术原因”*。新的 **Audio Overview** 定制提示词上限为 **500 字符**，且有报告称交互模式下麦克风检测失败。

**主题 5：硬件领域升温：Mac 速度、GPU 竞赛与新工具**

- [**Mac 凭借惊人的 MLX 速度展现强劲性能**](https://discord.com/channels/1131200896827654144/1131200896827654149/1366625096676020305)：新款 Macbook 表现出众，使用 **MLX** 运行 **Qwen3 30B A3B** 时达到约 **100 tokens/s**。根据 [Reddit 速度对比](https://old.reddit.com/r/LocalLLaMA/comments/1kaqnbj/speed_with_qwen3_on_mac_against_various_prompt)，据称其速度比 **llama.cpp** 快两倍以上。这一性能激发了人们对强大本地 **LLM** 的热情，如果 **4-bit Qwen3-30B-A3B** 量化版本表现稳定，**Aider** 等工具将可能从中受益。
- [**GPU 领域因 AMD 竞赛和 FP8 关注点而升温**](https://discord.com/channels/1189498204333543425/1359640791525490768/1366649146978730074)：**GPU MODE** Discord 频道内的 **10 万美元 AMD MI300 竞赛**引入了一个具有挑战性的单 **GPU MoE kernel** 问题（[阅读官方说明](https://tinyurl.com/amd-comp-moe)，[查看排行榜](https://www.gpumode.com/leaderboard/430)），最终提交截止日期为 **5 月 27 日**。讨论还探讨了在 matmul 中使用 **FP32 accumulation** 的 **FP8 quantization**（[参见 ONNX FP8 格式页面](https://onnx.ai/onnx/technical/float8.html)），并引用了 **Deepseek-v3** 的技术报告以及对潜在 *underflow*（下溢）问题的担忧。
- [**硬件线索与辅助工具浮出水面**](https://discord.com/channels/1189498204333543425/1233704710389764236/1366728443273936958)：**GPU MODE** 成员剖析了 **CDNA3 ISA Reference**，注意到（第 2.2.1 节）每个计算单元的 **64kB** 低延迟内存结构为 **32 banks**，每个 bank 包含 **512 个条目**（每个 4 字节）。一位社区成员推出了一个 [亚马逊 GPU 价格追踪器](https://www.unitedcompute.ai/gpu-price-tracker)，显示历史价格和 **每美元 teraflops**；同时 **Modular** 用户推荐使用 [flamegraph](https://github.com/brendangregg/FlameGraph) 来可视化 `perf` 输出（需要带调试信息编译）。

---


---

# Discord: 高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 登陆 WhatsApp 并支持图像功能**：**Perplexity AI** 现已接入 **WhatsApp**，并集成了**图像生成**功能。可通过[此链接](https://wa.me/18334363285)进行体验。
   - 这一扩展允许用户直接在 **WhatsApp** 内生成图像。
- **Sonnet 出现故障，为保证稳定性进行切换**：**Sonnet 3.7 API** 的错误率有所上升，根据 [Anthropic 状态事件报告](https://status.anthropic.com/incidents/th916r7yfg00)，**Perplexity** 暂时将查询路由至备选模型作为兜底。
   - 团队澄清，模型切换是为了在 **Sonnet** 出现问题期间维持服务可用性，*并非刻意为之*。
- **Sonar API 银行卡问题阻碍黑客松**：有用户报告借记卡与 **Sonar API** 不兼容，导致无法用于黑客松项目；还有报告称在银行卡验证后[未收到黑客松积分](https://discord.com/channels/1047197230748151888/1118264005207793674/1366292101666439239)。
   - 这些问题阻碍了 API 的访问并影响了黑客松的参与。
- **结构化输出问题显现**：用户在使用 API 的**结构化输出**时遇到问题，理由是意外的输出格式和架构强制执行困难。
   - 一位用户报告称需要指定 *'In english'*（用英语）来防止 API 返回中文，这与另一位用户看到的 **R1 系列模型**在思考时进入中文模式的问题类似，尤其是在尝试解方程时。
- **Grok App 在印度售价极低**：据报道，**Grok** 安卓应用对印度用户的 supergrok 每月仅收取 **700 卢比**，但对某些人来说，*免费层级甚至已经无法工作*。
   - 如果你拥有 premium +，可以在 X 上访问该应用。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3 GGUF 受解析器问题困扰**：用户在 **LM Studio** 中使用 **Qwen3 GGUF 模型**（尤其是 **128k 上下文长度**版本）时遇到模板问题，导致解析器错误；但这些模型与 **Ollama** 和 **llama.cpp** 兼容，能够集成到 **Open WebUI** 等平台。
   - 一些用户发现可以使用 **ChatML template** 作为权宜之计，尽管这在技术上并不正确。尽管底层使用了 **llama.cpp** 运行时，但 LM Studio 尚未更新以解决这些跨平台的不一致性。
- **ComfyUI 引发复杂评论**：成员们分享了一张描绘 **ChatGPT 对 ComfyUI 看法**的图片，引发了幽默的反响。
   - 一位用户评论说，图片中间*杂乱交错的线条*准确地代表了其中涉及的复杂过程。
- **GRPO 微调呈上升趋势**：进行 **GRPO** (Gradient Rollout Policy Optimization) 的用户报告了积极的结果，并表示愿意为他人提供帮助；一位用户报告称，在他们的用例中，发现 **Qwen 4b** 比 **gemma 3 4b notebook** 表现更好。
   - 然而，另一位用户报告称，在使用 **GRPO** 微调 **Gemma 3 4B** 以生成嵌套 **JSON** 配置时结果不一致，短输入的准确率显著下降；描述内容显著影响了触发器和动作组件，导致 **BLEU** 分数不稳定。
- **提议动态 BNB 量化方案**：一位成员提议创建一种动态 **BNB quantization** 方案，根据模块的敏感度分别使用 **4-bit**、**8-bit** 或 **BF16** 精度，并认为这可以在不牺牲准确性的情况下减少空间占用；此处提到了一篇相关[论文](https://arxiv.org/abs/2504.18919)。
   - 另一位成员表示，*如果用户对此有足够的需求，这可能是我们可以列入路线图的内容*。
- **模型服务系统 vLLM 获得认可**：在一名用户报告了 Unsloth 的 **Qwen3 GGUF 模型**问题后，另一名成员建议尝试 [vLLM](https://github.com/vllm-project/vllm)。
   - 该成员提供了一个使用 vLLM 部署 **unsloth/qwen3-unsloth-4bit** 的示例命令。



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O3 Pro 需求无视延迟**：用户热切期待 **O3 Pro** 的发布，开玩笑说它的潜在影响，并将其贴上 "p2w"（氪金获胜）模型的标签。
   - 用户对其成本和可访问性表示担忧，一些用户幽默地提到他们已经漫长地等待到了*第 13 天*。
- **Qwen 3 基准测试令人困惑，训练讨论引人关注**：关于 **Qwen 3** 性能的讨论显示，尽管基准测试结果强劲，但在实际使用中直观感觉不如 **2.5 Pro** 聪明，这引发了对其训练后精调（post-training refinement）的猜测。
   - 有建议认为 **Qwen 3** 的基座模型在微调方面可能表现出色，一位用户报告它在某些基准测试中优于 **Gemini 2.5 Pro**，尽管各人体验有所不同。
- **Gemini 2.5 Pro 依然占据统治地位**：一些用户仍然青睐 **Gemini 2.5 Pro**，因为它对不同角色的独特适应能力，以及在小众话题上采取立场的能力，让人感觉像是在与专家团队互动。
   - 尽管其他模型在单个基准测试中名列前茅，但用户发现 **2.5 Pro** 在 LM Arena 上的排名更高，因为它能适应*单样本提示强度（one-shot prompt intensity）*，并能*以无固定个性的方式承担答题者的角色*。
- **Grok 3.5 传闻增多**：用户在期待 **Grok 3.5** 模型到来时，热情与怀疑交织。
   - 一位用户评论说，**Grok 3** *每次都表现过头，就像你要求它证明某件事时，它会用冗长的废话来补充实质内容*。
- **Sonnet 3.7：WebDev 的顶尖模型？**：用户辩论了 **Claude 3.7 Sonnet** 的能力，称该模型*在我的大多数 WebDev 任务案例中仍然领先*，一些人同意它依然令人惊叹。
   - 有人指出 **Sonnet 3.7** 目前是 WebDev 竞技场（webdev arena）中排名第一的模型。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen3 通过 /no_think 命令禁用思考过程**：用户发现 `/no_think` 命令可以关闭 LM Studio 中 **Qwen3** 的 *thinking* 输出，但可能需要重复执行该命令或重新加载模型。
   - 一位用户指出，该命令只有在看到别人使用后才起作用，这表明 LM Studio 中可能存在 Bug 或未记录的行为；[这里是一个示例](https://cdn.discordapp.com/attachments/1110598183144399058/1366664571313455165/image.png?ex=68126dd1&is=68111c51&hm=95fa11f26f302fb70dff44ffabe1026e3594542e5ba09ee299c8085602b363e8&)。
- **Android 版 LM Studio 仍未发布**：尽管用户兴趣浓厚，但目前还没有 **Android** 版本的 **LM Studio**，这让寻求移动端 LLM 能力的用户感到失望。
   - 一位用户开玩笑地表示要挑战实现它，突显了对移动端版本的需求。
- **Qwen3 的专家数量设置引发困惑**：用户对 LM Studio 中 **Qwen3 MoE** 的“专家数量（number of experts）”滑块的用途提出质疑，其中一位用户注意到他们的 LM Studio 默认在 **128 个专家**中仅开启 **8 个专家**。
   - 共识似乎是，使用过多的专家可能会导致质量下降，因为领域专家会被“众多的平庸者所否决”；这是一张[相关的截图](https://cdn.discordapp.com/attachments/1110598183144399058/1366676379848151060/Screenshot_2025-04-29_022126.png?ex=681278d0&is=68112750&hm=9d2c442870ff8f2a6a1e62b320c942dd3a8b167767b0dfcf538db545d2c602be&)。
- **Bug 修复提升 Qwen3 性能**：修复了 Bug 的新版 **Qwen 3** 已经发布，解决了导致模型变慢的模板损坏问题，并包含 dynamic quants2.0。
   - 用户报告称 *修复 Bug 后的模型现在速度更快* 且响应更得体。
- **MLX 速度大幅超越 llama.cpp**：据报道，[MLX](https://github.com/ml-explore/mlx) 在处理 **Qwen3-30B-A3B** 的 Prompt 时，速度达到了 **llama.cpp** 的两倍以上。
   - 这些性能对比在 [Reddit 帖子](https://old.reddit.com/r/LocalLLaMA/comments/1kaqnbj/speed_with_qwen3_on_mac_against_various_prompt)中进行了讨论，重点介绍了 Mac 用户的体验。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Qwen3 的编程能力起伏不定**：**Qwen3** 的编程能力引发了讨论；一位用户称赞其解释详尽，而另一位用户则指出了其在[处理复杂数学任务时的问题](https://huggingface.co/models)。
   - 一位用户报告通过 *进一步降低温度（temp）* 修复了复杂数学任务的问题，而另一位用户则指出了 **Qwen3** 在 **tool calling** 方面的问题。
- **Gemini 2.5 Flash 遭遇速率限制和错误**：用户报告称 **Gemini 2.5 Flash** 即使在付费版本上也遇到了 **rate limits** 和 **errors**；一位用户在禁用网络搜索后仍然遇到了这种情况。
   - 官方澄清 **OpenRouter** 正面临持续的 **Vertex Token 计数问题**，且 OpenRouter 不支持 [free tier limits](https://aistudio.google.dev/)，不过有成员指出了一种[免费使用 Gemini 2.5 Pro](https://ai.google.dev/gemini-api) 的方法。
- **OpenRouter 缓存仅限于 2.0 Flash**：**OpenRouter caching** 目前**无法在 2.5 Flash 上运行**，仅支持 2.0 Flash，2.5 Flash 会报错（**No endpoints found that support cache control**）。
   - **Toven** 澄清说，新的缓存是为新的 5 分钟 TTL 编写的，缓存可以提高延迟，但**不影响定价**。
- **LLama 4 在新基准测试中失利**：根据一项基准测试评估，**LLama 4 表现不佳**，尽管有人指出这仅仅是一项基准测试的结果。
   - 进行基准测试的人补充说，[**ELO 分差在 25 以内在统计学上并不显著**](https://github.com/paradite/eval-data)，不足以区分优劣。
- **Tesla FSD 引发数字系统辩论**：一条 X 帖子的公告显示，一个模型声称 **9.9 大于 9.11**，引发了一些人思考这是否正确。
   - 其他人提出这*取决于语境*，因为 [**Tesla FSD 的版本命名规则不同**](https://x.com/elonmusk/status/1917099777327829386)，在这种情况下 9.11 > 9.9。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen3 在新款 Macbook 上运行速度极快**：新款 Macbook 使用 mlx 运行 **Qwen3 30B A3B** 时，达到了约 **100 tokens/s** 的惊人速度。
   - 为 **Aider** 提供快速本地 LLM 的可能性令人兴奋，特别是如果 **Qwen3-30B-A3B 的 4-bit 量化版本**在 Aider 基准测试中表现良好的话。
- **ktransformers 声称针对 MoE 进行了 VRAM 优化**：[ktransformers 库](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)声称仅需 **8 GB VRAM** 即可高效运行 **Mixture of Experts (MoE)** 模型。
   - 与将所有参数加载到 VRAM 相比，这种方法为处理 **30B-A3B** 模型提供了一种更有希望的途径。
- **Deepseek R2 凭借视觉和自学习功能引发热议**：传闻即将发布的 **Deepseek R2** 将具备增强的人类视觉能力和自学习功能，可能在*明天*发布，如[这部纪录片](https://www.youtube.com/watch?v=Lo0FDmSbTp4)所示。
   - 爱好者们正热切期待其发布。
- **Aider 新增思考中加载动画**：一个新的 [PR](https://github.com/Aider-AI/aider/pull/3911) 为 **Aider** 引入了 *🔃 Thinking* 加载动画，在等待 LLM 输出时显示。
   - 贡献者表示，这个小改动让 **Aider** 感觉更加*敏捷且充满活力*。
- **Gemini 2.5 和 Deepseek 组成黄金搭档**：一位用户发现，使用 **Gemini 2.5** 进行规划，并使用 **Deepseek** 处理 diff 和变更说明是一个很好的组合。
   - 他们推荐在 **AI Studio** 中使用，因为 Gemini 在那里是免费的。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **研究使用 FP32 累加的 FP8 计算**：成员们讨论了在 **matmul** 操作中使用 **fp8 量化**配合 **fp32 累加**的可能性和优势，特别是在 **Deepseek-v3** 技术报告的背景下，并参考了 [ONNX FP8 格式页面](https://onnx.ai/onnx/technical/float8.html)。
   - 有人指出 **FP8** 可能会遇到*下溢问题*，可能需要更高精度的累加器，同时也参考了[这个排行榜](https://www.gpumode.com/leaderboard/430)。
- **单 GPU MoE Kernel 挑战赛已上线**：针对 **10 万美元 AMD MI300 竞赛**，一个新的单 **GPU MoE kernel** 题目现已发布，详见[公告频道](https://discord.com/channels/1189498204333543425/1189640399476764692)。
   - 建议仔细阅读[该 kernel 的官方题目说明](https://tinyurl.com/amd-comp-moe)，并记住注册将于 **4 月 30 日**截止，提交截止日期为 **5 月 27 日**。
- **AOT Inductor 训练面临多线程故障**：一位用户报告了使用 **AOT Inductor** 进行 C++ 训练取得部分成功，但怀疑由于代码的不当特化（specialization）存在多线程问题。
   - 该用户计划提交一个 [PyTorch issue](https://github.com/pytorch/pytorch/issues) 以进行进一步调查，特别是关于多个工作线程调用 `fw_graph->run()` 时 API 的行为。
- **CDNA3 ISA 内存布局揭晓**：**CDNA3 ISA 参考文档**第 2.2.1 节显示，每个计算单元（compute unit）都拥有 **64kB** 的内存空间用于低延迟通信。
   - 该内存由 **32 个 banks** 组成，每个 bank 包含 **512 个条目**（每个条目 **4 字节**），有助于实现高效的数据访问和线程间通信。
- **亚马逊 GPU 价格追踪上线！**：一位成员发布了一个针对 **Amazon** 的 [GPU 价格追踪器](https://www.unitedcompute.ai/gpu-price-tracker)，提供历史价格数据并计算**每美元 Teraflops** 等指标。
   - 该工具利用全面的价格趋势，帮助用户精准把握为私有集群采购 GPU 的最佳时机。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 记忆功能... 某种程度上**：**ChatGPT** 现在具备持久记忆功能，分为长期记忆（源自重要的对话细节）和短期记忆（参考过去 **90 天**的内容），增强了上下文保留能力。
   - 用户可以禁用任一记忆类型，从而控制数据保留，但一个开关不能同时控制两者。
- **AI Agent 公司惨淡收场**：一个由教授领导的实验，让公司完全由 AI agents 运营，结果产生了[*混乱的结果*](https://futurism.com/professors-company-ai-agents)，凸显了当前 AI 在完全取代人类角色方面的局限性。
   - 尽管大科技公司宣称如此，该实验证明了当前 AI 模型仍需人类监督。
- **IAM360 协调 AI 和谐**：一位成员正在开发 **IAM360**，这是一个实验性的人机共生框架，使用具有持久角色的模块化符号 **GPT agents** 和一个用于涌现对话的零样本编排系统。
   - **IAM360** 基于标准的 **ChatGPT** 会话构建，旨在实现自然交互，无需自定义 **GPTs**、微调或 API 集成。
- **AI 艺术创作获得赞誉？**：一位用户成功以 **1500 Robux** 的价格售出了一个 AI 生成的缩略图，展示了 AI 在数字内容创作中的利基应用。
   - 然而，其他人警告说，目前的 AI 图像生成器在处理复杂的参考图像时表现不佳，可能会限制对现实世界客户的吸引力。
- **ChatGPT 的 Bio 工具助力构建**：成员们将 **ChatGPT** 的内部记忆识别为 `bio` 工具，并建议开发者在 prompt 中显式调用 `bio` 工具来定义保存命令，以确保准确的状态保留。
   - 在 prompt 中提供具体的规范将减少 **LLM** 的猜测；要求它识别并描述其连接的工具，列出它们的规范名称并演示正确的语法。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **使用 LM Studio 的 PyQt5 聊天应用界面**：分享了一个使用 **PyQt5** 构建的 AI 聊天应用程序，通过 [此 Python 脚本](https://cdn.discordapp.com/attachments/986699377257119794/1366717106447450122/AI_chat_app_005.py?ex=6811f5fe&is=6810a47e&hm=d9601e58ece57f0a5ba85d7da1c73f099068ee63c9220099dffa3614c74cd9bd) 利用 **LM Studio** 作为其后端服务器。
   - 为了启用功能，用户必须在运行应用程序之前，先在 **LM Studio** 中选择一个模型并将其作为本地服务器启动。
- **辩论厘清 OR 和 ML 的根源**：一场讨论辩论了 **Operations Research (OR)** 和 **Machine Learning (ML)** 之间的历史关系，指出了方法论上的分歧。
   - 虽然早期的 **AI/ML** 与 **OR** 和控制理论非常相似，但现代 ML 已转向统计方法，强调*从数据中学习，而不是从第一原理对现实建模*，并越来越注重实证方法。
- **匿名 LLM 愚弄 Reddit**：研究人员在 Reddit 的 **/r/changemyview** 板块测试了一个匿名 LLM，发现其*效率极高*，导致用户感到恼火，正如这篇 [X 帖子](https://x.com/emollick/status/1916905103358931084) 和 [Reddit 帖子](https://www.reddit.com/r/changemyview/s/k9Rd6IbyjY) 所讨论的那样。
   - 一位用户幽默地表示：“AI 并不聪明，改变我的看法”，**ChatGPT** 回复道：“是的，它们很聪明”，用户随后回复：“噢好吧，我错了”。
- **Qwen 3 的推理能力令用户兴奋**：成员们赞扬了新的 **Qwen 模型**，特别提到了改进的推理和指令遵循能力。
   - 一位用户报告说，*它们在某些推理任务上的输出*更胜一筹，特别赞扬了 **MoE** 模型的速度和智能，称其“与 2.5 Flash 一样聪明，甚至更聪明”。
- **Meta 发布 Llama 4**：在 **LlamaCon** 上确认了 **Llama 4**（也称为 *Little Llama*）的存在，详见此 [YouTube 直播](https://www.youtube.com/live/6mRP-lQs0fw)。
   - **LlamaCon** 的一个关键公告是 **SAM 3** 的开发和 **Meta** 的新应用，一些人猜测较小的 **Llama 4** 模型将如何与现有的 **Qwen** 模型竞争。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Atropos Framework 引导 RL**：**Nous Research** 推出了 [Atropos](https://github.com/NousResearch/Atropos)，这是一个用于基础模型强化学习（RL）的 rollout 框架，支持复杂环境以提升模型能力，其[介绍性博客文章](https://nousresearch.com/introducing-atropos)中详细介绍了训练和推理组件。
   - 使用 Atropos 环境创建的产物（Artifacts），包括一个新数据集和五个用于工具调用（tool calling）及公司基本面预测的新模型，已在 [HuggingFace](https://huggingface.co/collections/NousResearch/atropos-artifacts-68110ab511c5af02830247b6) 上线。
- **GRPO 工具调用提升 DeepHermes**：通过使用 Berkeley's Function Calling Benchmark，**GRPO** 环境在简单和并行工具调用上分别将 **DeepHermes** 的工具调用能力提升了 **2.4 倍**和 **5 倍**。
   - Atropos 是 **Psyche** 的关键组件，Psyche 是一个即将推出的去中心化训练网络，旨在全球范围内协调预训练、中训练和后训练的工作负载；5 月 18 日将在旧金山举办一场黑客松，以促进协作进展（更多细节即将公布）。
- **基本面预测模型准确率翻倍**：使用 **Atropos** 框架后，公司基本面预测模型在方向性变化上的准确率从 **~25%** 提高到了 **50%**。
   - Atropos 框架旨在通过强化学习引导语言模型发挥其最佳潜力。
- **DeepSeek R2 发布：事实还是虚构？**：有传言称 **DeepSeek R2** 可能很快发布，并且完全是在 **Huawei Ascend 910B** 硬件上训练的，但这些说法已被反驳。
   - 链接中引用了一条推文，其中包含 **DeepSeek** 的官方回应：*“我们会在发布 R2 时发布 R2，任何声称自己知道的人都在撒谎”*。
- **Qwen 3 在评估集上过拟合**：成员们发现 **Qwen 3 的基础模型**似乎对某些评估集（evals）非常过拟合，据报告，该模型在 **M24b** 上的 **Trivaqa** 得分为 **75%**，但在 **Q30MoE** 上仅为 **60%**。
   - 这引发了关于 MoE 有效性的讨论。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **支出限制导致快速信号停滞**：在超过支出限制后，用户报告称即使升级了服务，仍会出现数小时的延迟，另一位用户则报告称其 **fast requests** 已用尽。
   - 一位用户指出，即使在较慢的请求中，**Gemini** 依然保持快速，而其他用户在使用 **Gemini 2.5 Pro** 时遇到了挑战。
- **Discord 的发展：讨论让开发者感到愉悦**：一位成员开玩笑地提到 **Cursor 的 Discord** *终于再次受到关注*，表明活跃度和参与度有所增加。
   - 另一位成员自信地回应道 *Cursor 一直深受喜爱*，暗示团队只是在精益求精（polishing the cube）。
- **Gemini 故障引发困扰**：用户报告称 **Gemini 2.5** 经常在请求中途停止，即使在表明将执行操作之后也是如此。
   - 一名团队成员表示，他们正在与 **Google** 合作解决此问题，建议用户使用其他模型并提交其 **request ID** 以供调查。
- **Agent 冷淡：编辑避开了工程师**：用户面临 **Agent 无法进行编辑** 的持久问题，在多次尝试后，Agent 反而建议手动编辑。
   - 一名团队成员建议该问题可能源于 **Gemini 2.5 Pro**，建议刷新聊天上下文或切换到 **GPT 4.1**、**GPT 3.5** 或 **Claude 3.7**。
- **Ollama 官方：无线发布**：一位用户询问了官方 **Ollama 智能手机应用** 的发布时间表，并发布了相关的 [X 帖子](https://x.com/awnihannun/status/1917258279455187034)。
   - 一位用户提到重新安装 **Cursor** 并清除缓存解决了问题，而另一位用户确认手动清除缓存是重新安装之外的另一种选择。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Turnstile 测试大获成功！**：成员们成功测试了 [Cloudflare Turnstile](https://mineapi.pythonanywhere.com/docs)，确认了其功能性。
   - 测试的成功引发了成员们的热烈反响。
- **Whisper Turbo 在 HF 遭遇故障！**：用户报告 **OpenAI 的 whisper-large-v3-turbo** 在 HF 推理端点（inference endpoint）无法运行，甚至影响了网页端 Demo。
   - 成员们分享了类似的案例，例如[这篇讨论](https://discuss.huggingface.co/t/sentence-transformers-all-minilm-l6-v2-not-working-all-of-a-sudden/152691)，以供潜在的故障排除参考。
- **GGUF CPU Offloading 走向主流**：成员们确认 **GGUF 格式** 支持 CPU offloading（CPU 卸载），特别是在合并 Checkpoint 时。
   - 他们指出 *Transformers + Accelerate 或 Llama.cpp* 可以简化这一过程。
- **Pi-Scorer 有望成为 LLM-as-a-Judge 的代理方案**：一名成员介绍了 **Pi-Scorer** 作为 **LLM-as-a-Judge** 的可行替代方案，并展示了用于评估模型 Checkpoint 的 Colab 笔记本，使用了 [Pi-Scores](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/SFT_Model_Checkpoint_Observability_withPi.ipynb) 并将其实现为 [奖励函数（reward functions）](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/PiScorer_as_GRPO_Reward_Function.ipynb)。
   - 这为使用 Pi 进行 SFT 模型 Checkpoint 的可观测性提供了有用的工具。
- **边缘滤波器助力卓越的错误提取**：一名成员建议使用 **Canny edge** 或 **Sobel** 等滤波器，通过特定阈值来隔离图像中的缺陷。
   - 配合正确的阈值，自动标注数据集上的划痕将变得更加容易。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 凭借技术实力斩获 Webby 奖！**：**NotebookLM** 在 [Webby Awards](https://winners.webbyawards.com/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement/331142/notebooklm) 中荣获 **技术成就奖（Technical Achievement）**。
   - 这一荣誉彰显了 **NotebookLM** 对其平台的持续优化。
- **NotebookLM 的全球之声：现已支持 50 多种语言！**：**NotebookLM** 推出了**多语言支持**，现在可以交流 [50 多种语言](https://www.forbes.com/sites/rogerdooley/2025/04/29/googles-notebooklm-now-speaks-50-languages-enabling-global-content/)，增强了全球不同用户的访问体验。
   - 然而，功能是逐步推出的；一些用户最初遇到了 UI 故障，例如有人报告 **越南语音频** 无法工作，且 UI 仍显示 *“仅限英语（English only）”*。
- **音频概览自定义功能限制了 Prompt 查询！**：测试 **音频概览（Audio Overview）** 自定义功能的用户发现存在 **500 字符限制**，这引发了关于该功能与上传独立指令文件相比实用性如何的讨论。
   - 一位用户的目标是 *“减少愚蠢的闲聊，专注于事实和时间线”*。
- **音频概览时长因语言而异！**：用户报告称，与英语相比，**非英语音频概览** 的时长限制更短；例如，英语有 **15 分钟限制**，而土耳其语仅为 **6 分 20 秒**。
   - 团队称这些限制是出于 *“技术原因”*，但保证他们正在积极致力于延长时长。
- **麦克风问题困扰交互模式！**：一位用户报告称 **交互模式** 无法检测到麦克风音频，影响了可用性。
   - 故障排除建议包括验证 **麦克风权限**、检查 **浏览器设置**、使用 [麦克风测试工具](https://mictests.com/) 以及尝试更换浏览器。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **附加额度令用户感到困惑**：一位用户报告称，由于有效期太短，Manus.im 早期订阅的附加额度（add-on credits）在不续订的情况下毫无用处，导致损失了 **3900** 个额度。
   - 另一位用户澄清说，只要订阅保持激活状态，赠送额度就不会过期，且邀请码的分发似乎是随机的，可能受到了限制。
- **Manus Fellow 项目受到质疑**：一位用户询问了 Manus Fellow 项目的筛选过程、目标国家，以及对巴基斯坦和印度等地区的包容性。
   - 另一位用户澄清了邀请结构，指出入门计划（starter plans）提供 **2 个邀请码**，专业计划（pro plans）提供 **5 个邀请码**。
- **Beta 测试受到审查**：一位用户批评了 Manus.im 的 Beta 测试方法，认为限制有额度的用户违背了 Beta 阶段的初衷。
   - 他们建议 *真正的 Beta 测试应该允许用户从头到尾完成完整的项目，从而对体验提供有意义的反馈并提出改进建议*。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **X-Ware Red 工具发布至社区**：一名用户分享了 **X-Ware Red**，该工具利用 embed 的标题，并在其前缀加上 `r.jina.ai/` 和 `openrouter-free-tier` 来为 thread 生成标题。
   - 另一名用户建议增加一个切换开关，让用户能够控制 thread 标题是否应与 embed 名称不同。
- **Meta 为工程师发布 Llama Prompt Ops**：**Meta** 推出了 [Llama Prompt Ops](https://github.com/meta-llama/llama-prompt-ops)（一个专为 Prompt Engineering 设计的开源工具）以及 [Synthetic Data Kit](https://github.com/meta-llama/synthetic-data-kit)。
- **用户报告：发布链接会重命名 Thread**：一名用户报告了一个 Bug，即在 thread 中发布链接会错误地重命名已经有名称的 thread。
   - 该 Bug *应该只针对标题中包含 'https://' 的 thread 进行查找并修改*。
- **社区搜寻持久的 LLM Benchmarks**：一名用户请求一份可靠的 **LLM Benchmarks** 调查报告，以支持模型历史数据的对比。
   - 另一名用户指出，*大多数 Benchmarks 的寿命不足 2 年*，并推荐参考“AI Engineer Reading List”以获取当前的 Benchmarks，同时提供了 OSS 排行榜版本 1 和 2 的帖子链接。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 仓库采用多重许可**：**Modular 仓库**现在需要多种许可证，因为 `src/max` 的部分内容采用 Modular 的 **Community License** 授权，而其余部分则使用 **Apache 2**。
   - 这一变化反映了仓库内多样化的许可需求，特别是对于像 [`src/max/serve`](https://github.com/modular/max/blob/main/src/max/serve/README.md) 中的组件。
- **调整 Origins 导致难题**：成员们讨论了 Mojo 中 **Origins** 的问题，特别是围绕 API 缺口和缺失的语言特性（如可参数化的 traits），这使得将 Origins 重新绑定到容器元素变得复杂。
   - 讨论还提到，持有指向同一 Origin 的两个可变引用是有问题的，尽管可以通过将 Origin 转换为 **MutableAnyOrigin** 来绕过此限制。
- **利用指针绕过 Origins**：为了实现类似 list 和 span 的类型，或者阅读标准库中的 `sort` 实现，开发者有时会绕过 **Origins** 并诉诸于 *指针操作（pointer time）*。
   - 讨论强调了对指针类型的担忧，特别是关于 Mojo 中可变性（mutability）和不可变性（immutability）的修复。
- **标准 Python 导入即将支持**：Mojo 可能会实现对标准 Python `import` 语句的全支持，这暗示 `python.import_module` 最终可能会被弃用。
   - 一名成员将这一变化的可能性描述为 *“非常肯定的可能（pretty definite maybe）”*，暗示了 Mojo 未来对 Python 集成的增强。
- **`Flamegraph` 可视化 Perf 输出**：为了可视化 `perf` 的输出，成员们建议使用 [flamegraph](https://github.com/brendangregg/FlameGraph)，这需要编译可执行文件时包含 **debug info** 才能进行有效分析。
   - 他们还提到使用 `llvm-mca` 来分析特定的代码块，并引用了 `gpu` 模块的一个私有部分（[链接](https://github.com/modular/max/blob/main/mojo/stdlib/src/gpu/profiler.mojo)）。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **GPT-4o 通过 LlamaIndex 精通俄罗斯方块**：一段视频展示了 **GPT-4o** 如何利用 **LlamaIndex** 和 **Composiohq** 一次性生成 **Tetris**（俄罗斯方块），展示了其先进的代码生成能力。
   - 演示中使用的代码已在 [GitHub](https://t.co/KJb7YRINWg) 上发布，为开发者提供了一个实用的示例。
- **PapersChat 使用 LlamaIndex 对 ArXiv 和 PubMed 进行索引**：**PapersChat** 使用 **LlamaIndex**、**Qdrant** 和 **MistralAI** 对 **ArXiv** 和 **PubMed** 上的论文进行了索引。
   - 用于查询这些论文的精美 Web UI 可以在[这里](https://t.co/lYwXh27F9x)访问。
- **Azure OpenAI 饱受间歇性超时困扰**：用户报告称，即使在 Prompt、端点和网络条件一致的情况下，**Azure OpenAI** 端点仍会出现间歇性**超时 (timeouts)**，这可能暗示存在**速率限制 (rate limits)**或**防火墙问题**。
   - 重试机制有时无效，而网络更改也只是偶尔能解决这种不一致性。
- **MessageRole：破解 FUNCTION 与 TOOL 的代码差异**：**MessageRole.FUNCTION** 和 **MessageRole.TOOL** 之间的区别取决于所使用的具体 API。
   - 像 **OpenAI** 这样的一些 API 使用 **tool messages**，而其他 API 则依赖于 **function messages**。
- **Function Agent 上下文混乱问题揭秘**：一位用户遇到了 **function agent** 在第二轮交互期间卡在流事件 (stream event) 的问题；该用户提供了示例代码。
   - 一位成员建议在 `stream_events()` 退出后等待处理器 (`await handler`)，以确保之前的运行结束并接收到最终响应，这修复了该错误。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RAG 聊天机器人应对多源答案的挑战**：一位正在构建 **RAG 聊天机器人**的成员在生成需要多个文档信息的答案时遇到了困难，即使使用了**向量搜索 (vector search)**和 **BM25**。
   - 该聊天机器人使用 **LLM Claude 3.5 Sonnet v1** 和 **Amazon Titan v1** 嵌入，该成员正在寻求关于如何有效地将引用链接到文档内附录的建议。
- **关于多源数据 GraphRAG 的讨论**：一位成员询问了使用 **GraphRAG** 聚合多源答案的价值，并将其与需要特定领域预训练模型的 **insightRAG** 进行了比较。
   - 他们正在寻找 **GraphRAG** 的替代方案，并提到计划参加 **NAACL**。
- **工程师启动本地推理项目**：一位曾是 [Dataherald](https://github.com/Dataherald/dataherald) 联合创始人的成员正在启动一个专注于**本地推理 (local inference)**和**小模型训练 (small model training)**的新项目。
   - 该成员表达了与社区合作并为相关研究做出贡献的浓厚兴趣。
- **符号提示词递归探索**：一位成员正在研究分类器压力下**递归符号提示词 (recursive symbolic prompts)**的行为，特别是平滑和对齐约束如何影响**多轮幻觉漂移 (multi-turn hallucination drift)**。
   - 他们热衷于了解尽管存在软对齐漂移和输出平滑，但诸如**角色绑定谓词 (role-bound predicates)**或**注意力同步标记 (attention-synced markers)**之类的符号结构如何在多个输出中持久存在。
- **HHH 目标揭示**：分享了一项关于基于 **HHH**（Helpful, Honest, Harmless）对齐目标对 [LLM 输出进行定量评分](https://www.notion.so/TPIP-Exposing-Alignment-Tension-in-Modern-LLMs-1d5927516e1b8080b8c3d625a40a131d)的研究，使用 **YAML** 和 **python/Gradio** 来审计用户会话。
   - 观察到前沿模型在诚实合规性方面差异很大，具有讽刺意味的是，像 **ChatGPT 4o** 和 **4.5** 这样的一些模型在面对模糊答案时输出的高置信度，使得 **OpenAI** 成为前沿模型中透明度最低的。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **凭据传递问题 (Credential Passing Concerns)**：一位成员在尝试使用 Python 从客户端通过 header 向 **MCP server** 传递凭据时遇到问题，正在寻求社区的帮助。
   - 目前，针对该咨询尚未提供任何解决方案或建议。
- **RAG 服务器架构讨论 (RAG Server Architecture Debated)**：一位成员正在探索构建 **RAG 类型服务器**的可行性，即客户端可以通过端点上传文件，将其存储在服务器端，并利用这些文件进行问答。
   - 他们正在征求关于这种方法可行性的反馈，以及是否有更有效的替代架构。
- **Streamable HTTP 身份验证的细微差别**：一位成员询问了社区对 **Streamable HTTP 实现和身份验证**的看法，特别是在最近发布的 **TS SDK** 中。
   - 反馈表明其运行效果良好，但成员们仍在研究托管**多租户服务器 (multi-tenant server)** 的细微差别以及状态性（statefulness）对其的影响。
- **多租户服务器状态性研究 (Multi-Tenant Server Statefulness Examined)**：针对托管**多租户服务器**及状态性的影响提出了担忧，特别是质疑为什么单个实例足以满足有状态设置，但对于无状态设置却不行。
   - 讨论围绕无状态服务器是否应该为每个请求生成一个新的 **MCP server** 实例展开。
- **开源 Agentic 应用：生产就绪了吗？**：一位成员质疑开源模型在生产环境（而非仅仅是个人项目）中用于 Agentic 应用的实际适用性。
   - 他们对大多数开源模型在没有 Fine-tuning 的情况下进行推理或有效遵循指令的能力表示怀疑。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **通过 Foreach 实现快速梯度缩放**：一位成员分享了一个使用 `torch._foreach_mul_` 进行梯度缩放的 [代码片段](https://link.to/snippet)，这可能与梯度裁剪（gradient clipping）合并为单个参数循环，从而提高优化速度。
   - 另一位成员指出了[相关的 PR](https://github.com/pytorch/torchtune/pull/2624)，并想知道这种看似恒定的增益是否会在多次迭代中累积，并指出了潜在的注意事项。
- **Tune 贡献者寻找 Easy First Issues**：一位成员强调了[两个 easy issues](https://github.com/pytorch/torchtune/issues/2648)和[另一个](https://github.com/pytorch/torchtune/issues/2649)，供社区为项目做贡献，旨在降低入门门槛。
   - 这些问题为新贡献者提供了参与项目并获得经验的机会，但未进行详细描述。
- **DoRA 与 QAT 的结合尚未被探索**：一位成员询问了将 **DoRA (Difference of Low-Rank Adaptation)** 与 **QAT (Quantization-Aware Training)** 结合使用的经验，这是一个尚未被充分探索的组合。
   - 在提供的消息中没有关于此组合的讨论或回复，这表明社区中存在知识空白或缺乏实验。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 用户渴望 MCP 使用文档**：用户正在请求 **DSPy** 最新版本中引入的新 **MCP (Multi-Controller Processing)** 功能的教程或文档。
   - 一位用户建议，通过查看测试用例有助于理清对 **stdio** 和 **SSE clients** 设置的理解，因此可能不需要专门的教程。
- **React 开发者思考如何显示 Thoughts 组件**：一位用户询问在 **DSPy** 框架内，在 **React** 中显示 **Thoughts 组件**的最佳方式。
   - 他们提到了修改 forward 方法的选项，但询问是否有更合适的地方来实现此功能。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Markdown 与图像 RAG 之争升温**：成员们讨论了在 **PDF** 上对比 **基于 Markdown** 与 **基于图像的多模态 RAG**，其中一名成员使用 **Docling** 将 PDF 转换为 Markdown 并计算文本 Embedding。
   - 他们正在考虑切换到 **EmbedV4**，以直接处理原始图像进行 RAG 中的多模态 Embedding。
- **Cohere 考虑提高 Embed V4 速率限制**：一位用户询问 **Cohere** 是否会增加 `embed-v4` 的生产速率限制，并表示 **每分钟 400 次请求** 对于他们处理大量 PDF 的用例来说是不够的。
   - 目前尚未收到回复。
- **Embed V4 在 Bedrock 上的可用性预告**：一位用户询问 **Embed V4** 是否会在 **Bedrock** 上可用。
   - Cohere 尚未给出答复。
- **新数据科学家推崇 Embed V4**：一位新的数据科学家加入了 Cohere Discord 社区，表达了尝试新工具的兴奋之情，特别是 Cohere 最新的 **Embed V4 模型**。
   - 这位新成员表示*很高兴加入社区*。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Manus AI 工具走向全球**：一位成员分享了 [Manus AI](https://manus.im/edu/invitation/FBBGGFBFKTUE)，并指出它在*中国发布*后的可用性。
   - 该工具据称是*首个自动研究 AI Agent*，引发了关于其潜在影响的讨论。
- **Nomic 助力 Embedding 工作流**：一位成员强调 **Nomic** 提供了全面的 Embedding 工具，并暗示其功能*超越了 GPT4All*。
   - 他们强调了 **Nomic** Embedding 工具的通用性，称其与*各种其他软件*兼容。
- **分组 Embedding，跳过训练？**：一位成员提议，**分组 Embedding** 可以替代传统的训练方法。
   - 该建议涉及对特定人的 Embedding 进行分组、取平均值，然后使用该平均值对同一个人的其他照片进行排序和识别。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Berkeley 模型评估的松散与严格之争**：一位成员为 **Berkeley 函数调用模型** 提议了*“松散”与“严格”评估机制*，特别是针对那些可以通过*“黑客手段”*使其工作的特定用例。
   - 他们举了一个例子：一个模型被错误地训练为输出 `<tool_call>` 而不是其规范指明的 `<|tool_call|>`，在这种情况下，了解情况的用户可能会忽略该错误并评估其功能正确性。
- **模型训练导致不一致性**：一位成员遇到了一个被错误训练的模型，该模型输出 `<tool_call>` 而不是规范要求的 `<|tool_call|>`。
   - 该成员建议，如果他们专门了解该模型，可以忽略此错误并评估功能正确性，但普通用户无法做到这一点。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：各频道详细摘要与链接

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1366625908194021487)** (2 条消息):

> `WhatsApp 上的 Perplexity AI、Sonnet 模型行为更新、Anthropic 状态事件`

- **Perplexity 登陆 WhatsApp 并支持图像生成！**：**Perplexity AI** 现已在 **WhatsApp** 上可用，包括 **图像生成** 功能，可通过 [此链接](https://wa.me/18334363285) 访问。
- **Sonnet 出现故障，路由至备选模型！**：由于 **Sonnet 3.7 API** 的错误率增加，部分查询被临时路由到备选模型作为兜底，这与 [Anthropic 状态事件](https://status.anthropic.com/incidents/th916r7yfg00) 有关。
- **模型切换：并非刻意为之！**：Perplexity 团队澄清，他们 **不会故意切换您选择的模型**；只有当 **Sonnet** 遇到错误时才会进行路由，以维持服务的可用性。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1366625571546861608)** (1112 条消息🔥🔥🔥): 

> `Free AI billing, Grok android app, Model Fallbacks, The Boys fanboy` 


- **用户利用免费 AI 计费漏洞**：一些用户声称一年来 *没花一分钱* 支付 AI 账单，可能是通过 [Play Store](https://play.google.com/store/account/subscriptions) 订阅漏洞，或者通过 *参加一些网络研讨会并填写表格* 获取的。
   - 其他人询问了具体方法，而一些人则表示难以置信。
- **Grok app 对印度用户很便宜**：据报道，**Grok** android app 对印度用户的 supergrok 每月仅收费 **700 卢比**，但对某些人来说 *免费层级甚至已经无法使用了*。
   - 如果你有 premium +，可以在 X 上使用它。
- **Perplexity 在未通知的情况下更换模型**：用户抱怨 Perplexity 正在用 **GPT 4.1** 或 **Deepseek** 等低质量模型替换 Claude 3.7，并且因为 *没有模型切换或明确的模型指示* 而感到愤怒。
   - 一位用户表示：*它直接使用 R1 生成答案，然后发送给 sonnet 进行思考。最后却说答案来自 sonnet。这太阴险了。*
- **Discord 频道变成了《黑袍纠察队》（The Boys）粉丝大会**：频道的对话转向了 *The Boys* 领域，用户分享 GIF 并讨论剧情点，比如 [祖国人在公共场合杀人的场景](https://www.youtube.com/watch?v=IIGf-kK-g2I&t=0)。
   - 其他人则在思考是否要完全跳过祖国人的戏份，并开玩笑地询问该剧是否有比平时更令人作呕的场景。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 条消息): 

_paradroid: https://www.perplexity.ai/search/d7bb905e-27e3-43e9-8b68-76bea1905457
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1366670037858910289)** (14 条消息🔥): 

> `Sonar API Debit Card Issues, Hackathon Credits, Structured Output Issues, Async Deep Research API, API vs Web Results` 


- **银行卡难题困扰 API 用户**：一位用户报告称，他们的借记卡不支持 **Sonar API**，导致他们无法在 hackathon 项目中使用；还报告在银行卡验证后 [未收到 hackathon 积分](https://discord.com/channels/1047197230748151888/1118264005207793674/1366292101666439239)。
   - 在给定上下文中未提供解决方案。
- **结构化输出 (Structured Output) 问题显现**：用户在使用 API 的 **structured output** 时遇到问题，包括意外输出和难以强制执行 schema 约束。
   - 一位用户不得不明确指定 *'In english'* 以防止 API 返回中文。
- **Deep Research API 的异步性？**：一位用户质疑为何没有 **异步 deep research API**，认为长时间维持 socket 连接是不切实际的。
   - 该用户提议了一个涉及 **GUID**、状态端点和独立结果检索的工作流，但未得到确认或替代方案。
- **API 输出与网页端体验存在差异**：一位用户对 **API 结果** 在质量、引用等方面与 **网页界面** 不符表示失望。
   - 未提供解释或解决方案。
- **中文模型？**：一位用户发现他们必须在 prompt 中指定 "In english"，因为他们收到了中文输出。
   - 另一位用户补充说，他们看到 **基于 R1 的模型** 在思考时会进入中文模式，尤其是在尝试解方程时。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1366625626748092506)** (899 messages🔥🔥🔥): 

> `Qwen3, LM Studio 问题, GGUF 修复, Training configuration, Multi-GPU support` 


- **Qwen3 GGUF 上传存在 Template 问题**：成员们在 **LM Studio** 中使用上传的 **Qwen3 GGUF 模型**时遇到模板问题，特别是 **128k context length** 版本，导致了解析器错误。
   - 一些人发现可以使用 **ChatML template** 作为变通方法，虽然这在技术上并不完全正确，Unsloth 团队正在努力解决不同平台间的这些不一致性。
- **Unsloth 对 transformers 进行补丁**：加载 **Unsloth** 时，它会为 **transformers** 和其他一些组件打补丁以进行优化，但可能会出现破坏原有功能的问题。
   - 加载库后可能会出现性能和其他问题，建议下载 GitHub 版本可能会解决该问题。
- **Qwen3 GGUF 现在可在 Ollama 和 llama.cpp 中运行**：Unsloth 团队确认其 **Qwen3 GGUF** 与 **Ollama** 和 **llama.cpp** 兼容，从而能够与 **Open WebUI** 等平台集成。
   - 然而，一些用户发现由于模板问题尚未解决，模型在 LM Studio 中无法工作，尽管 LM Studio 使用的底层 **llama.cpp** 运行时不是最新的。
- **Unsloth 即将发布公告并重新上传所有模型**：Unsloth 团队表示他们正在重新上传所有模型，并可能在明天或周三发布[官方公告](https://huggingface.co/unsloth/Qwen3-32B-unsloth-bnb-4bit)。
   - 图像组件可能是 tool calling，但目前尚不确定。
- **适用于 Unsloth 的 CCE 和稳定 Triton 版本**：用户在 Colab 中遇到了 Triton 错误，建议将 Triton 降级到 **3.2.0** 版本，该版本应能与 Unsloth 正常配合使用，以避免 CCE 错误。
   - 一位用户指出，负责将 CCE 上传到 pypi 的是 Daniel Han。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1366653377551007854)** (10 messages🔥): 

> `ChatGPT 对 ComfyUI 的评价, 加州 AI 小组, ComfyUI 演示` 


- **ChatGPT 给出对 ComfyUI 的评价**：一位成员分享了一张描绘 **ChatGPT 对 ComfyUI 看法**的图片，引发了幽默的反应。
   - 一位用户评论说，图片中间**杂乱交织的线条**准确地代表了其中涉及的复杂过程。
- **加州 AI 小组正在筹备中？**：一位成员询问在加州开展**线下 AI 小组发展**的机会，寻求当地参与者。
   - 另一位住在 Fremont 的成员表示感兴趣，并引用了其 [X 账号](https://x.com/Dan50412374/status/1787936305751748844)上展示的一个项目。
- **ComfyUI 演示展示**：一位成员分享了各种 **ComfyUI 演示**，并指出每个示例在没有经过任何润色处理的情况下看起来都各不相同。
   - 另一位成员喜欢该成员 [X 账号](https://x.com/Dan50412374/status/1777216327255806411)上展示的另一个演示，该演示具有不同事物之间的转换效果。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1366631287011545129)** (186 messages🔥🔥): 

> `Unsloth 安装问题、Qwen notebook 问题、GRPO 性能、Lora 效率、Unsloth & Ollama/vLLM` 


- ****Unsloth 安装导致安装不稳定****：有人指出，由于与预装包冲突，在 Google Colab 上需要使用 `--no-deps`，并且可能需要重启内核以解决缓存问题。
   - 还有建议称，遇到 WSL 杀掉 unsloth 进程问题的用户可以*尝试使用 Windows*。
- ****Qwen Notebook 需要一些调整****：用户报告称运行 **Qwen notebook** 需要极小的改动，例如调整名称并使用 `tokenizer.qwen_enable_thinking = True` 启用推理。
   - 但据报告 **Unsloth 版本 2025.4.2** 在 Qwen 上已损坏：降级到 **Unsloth 2025.3.19** 可解决此问题。
- ****GRPO 微调正趋于完善****：进行 GRPO (Gradient Rollout Policy Optimization) 的用户报告了积极的结果，并表示愿意为他人提供帮助。
   - 一位用户提到他们最初使用 **gemma 3 4b notebook**，但发现 **Qwen 4b** 更适合他们的用例。
- ****Lora 训练不应耗时过长****：一位用户在 4k 问答对上使用 Lora 训练 **unsloth/phi-4-unsloth-bnb-4bit** 发现耗时数周，这很不正常。
   - 一名成员建议直接使用 Python 脚本而不是 text-generation webUI（由于截断长度问题），并提供了一个 [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb) 作为基础。
- ****Unsloth 与模型推理系统 vLLM 配合良好****：一位用户报告称，来自 Unsloth 的 **Qwen3 GGUF 模型** 在 **Ollama v0.6.6** 中无法正常工作，会出现随机内容的幻觉。
   - 一名成员建议尝试 [vLLM](https://github.com/vllm-project/vllm)，并提供了一个使用 vLLM 部署 **unsloth/qwen3-unsloth-4bit** 的示例命令。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1366821744937144431)** (4 messages): 

> `Pi-Scorer, LLM-as-a-Judge, encoder model` 


- **Pi-Scorer：Judge Judy 的替代方案**：一名成员介绍了 **Pi-Scorer** 作为 **LLM-as-a-Judge** 的替代方案，并提供了用于模型检查点评估的 [Colab notebooks](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/SFT_Model_Checkpoint_Observability_withPi.ipynb) 链接以及 [奖励函数](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/PiScorer_as_GRPO_Reward_Function.ipynb) 链接。
- **Pi 模型揭晓为 Encoder 架构**：一名成员询问了 **Pi 模型** 的架构，结果显示它是一个 **encoder model**。
   - 另一名成员称赞它是一个*很酷的服务*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1366805568731353119)** (47 messages🔥): 

> `动态 BNB 量化, LLM 在医疗建议中的应用, 基于 Gemma 的混合专家模型, 注意力头路由, GRPO 微调` 


- **提出动态 BNB 量化方案**：一名成员提议创建一种动态 **BNB quantization** 方案，根据模块的敏感度分别使用 **4-bit**、**8-bit** 或 **BF16** 精度，并认为这可以在不牺牲准确性的情况下减少空间占用；此处提到了相关论文 [here](https://arxiv.org/abs/2504.18919)。
   - 另一名成员表示，*如果用户对此有足够的需求，这可能是我们可以列入路线图的内容*。
- **LLM 在医疗建议综合和患者交互方面面临困难**：一篇论文指出用户交互是使用 **LLM** 提供医疗建议的一个挑战，引发了关于 **LLM** 是否能综合医学知识，以及 **训练 LLM** 是否能确保它们不这样做的讨论。
   - 一名成员根据医学院预科经验指出医生与患者互动中*临床沟通技巧 (bedside manner)* 的重要性，暗示 **LLM** 目前缺乏这种技能。
- **MoE 设置与 Gemma**：一名成员询问了如何使用 **Gemma 3 4B** 实现 **Mixture of Experts (MoE)** 设置，质疑尽管其架构不同是否仍能适配。
   - 建议从根本上改变模型或探索涉及 **Mixture of Expert attention heads** 的方法，参考了 [这篇论文](https://arxiv.org/pdf/2410.11842)。
- **GRPO 在 JSON 配置生成任务中效果不佳**：一名成员报告称，在使用 **GRPO** 微调 **Gemma 3 4B** 以生成嵌套 **JSON** 配置时结果不一致，短输入的准确率显著下降。
   - 尽管使用了自定义奖励函数进行训练，该成员发现 **GRPO** 不适合该任务，因为描述内容显著影响了触发器和动作组件，导致 **BLEU** 分数不一致。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1366625114598412319)** (544 条消息🔥🔥🔥): 

> `O3 Pro, Qwen 3, Gemini 2.5 Pro, Grok 3.5, Model Benchmarking and Evaluation` 


- **O3 Pro 需求无惧延迟**：用户正热切期待 **O3 Pro** 的发布，有人开玩笑说它因为智力超群而具有“病毒”潜力，并被视为一种“p2w”（付费即赢）模型。
   - 然而，一些用户对其成本和可访问性表示担忧。甚至有人开玩笑说，他们现在已经等待 **O3 Pro** 13 天了。
- **Qwen 3：基准测试的困惑与训练讨论**：有关于 **Qwen 3** 性能的讨论，一些用户发现尽管基准测试结果强劲，但在实践中感觉不如 **2.5 Pro** 聪明，这引发了对其 Post-training 不够完善的猜测。
   - 有人建议 **Qwen 3** 的 Base Model 可能非常适合 Fine-tuning，一位用户指出 **Qwen 3** 在某些基准测试上优于 **Gemini 2.5 Pro**，而其他人似乎没察觉到差异，还有人提到它在 4/5 的基准测试中击败了 2.5 Pro。
- **Gemini 2.5 Pro 依然占据统治地位**：一些用户仍然偏爱 **Gemini 2.5 Pro**，因为它具有适应不同角色或在利基话题上表态的独特能力，让人感觉像是在与不同的专家设施互动，有人称其为“目前最强的 Base Model”。
   - 尽管有些模型在单项基准测试中夺冠，但一位用户发现 **2.5 Pro** 在 LM Arena 上的排名更高，因为它能适应“One-shot Prompt 强度”，即“以无固定人格的方式承担答题者角色”。
- **Grok 3.5 即将到来？**：用户正在期待 **Grok 3.5** 模型，但对其潜力的看法不一，有人持谨慎乐观态度，而另一些人则保持怀疑。
   - 一位用户表示 **Grok 3** 每次都用力过猛，“就像你要求它证明某件事时，它会用冗长的废话来补充实质内容”。
- **Sonnet 3.7：WebDev 的顶级模型？**：用户辩论了 **Claude 3.7 Sonnet** 的能力，声称该模型“在我的大多数 WebDev 任务案例中仍然领先”，一些人同意它依然令人惊叹。
   - 有人指出 **Sonnet 3.7** 目前是 WebDev Arena 上的排名第一的模型。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1366626841892225084)** (271 条消息🔥🔥): 

> `Qwen3 thinking, LM Studio on Android, Qwen3 experts number, Qwen3 bug fixes, Qwen3 with RAG` 


- **削减 Qwen3 的思考过程**：用户讨论了如何禁用 **Qwen3** 的 *thinking* 输出，发现 `/no_think` 命令在用户消息或 System Prompt 中有效，但可能需要重复或重新加载模型才能生效；[这里是一个示例](https://cdn.discordapp.com/attachments/1110598183144399061/1366664571313455165/image.png?ex=68126dd1&is=68111c51&hm=95fa11f26f302fb70dff44ffabe1026e3594542e5ba09ee299c8085602b363e8&)。
   - 一位用户发现，只有在看到别人操作后，自己尝试才奏效。
- **Android 版 LM Studio：移动端的梦想？**：用户询问了 **Android** 版本的 **LM Studio**，但被告知目前没有移动版本。
   - 一位用户开玩笑说要把它作为自己的使命来实现。
- **Qwen3 精心调校的专家数量**：用户讨论了 **Qwen3 MoE** 的“专家数量”滑块，一位用户注意到他们的 LM Studio 默认设置为 128 个专家中的 **8 个**，质疑如果该设置限制了模型，为什么还要存在；这里是 [相关截图](https://cdn.discordapp.com/attachments/1110598183144399061/1366676379848151060/Screenshot_2025-04-29_022126.png?ex=681278d0&is=68112750&hm=9d2c442870ff8f2a6a1e62b320c942dd3a8b167767b0dfcf538db545d2c602be&)。
   - 据称，更多的专家可能会导致“更多的计算和更多的混乱，实际上质量更低”，因为领域专家会被许多“白痴”投票否决。
- **Qwen3 Bug 修复发布，提升性能**：带有 Bug 修复的新版 **Qwen 3** 已发布，解决了导致模型变慢和响应不当的损坏模板问题。
   - 据指出，“修复了 Bug 的模型现在速度更快了”，且此版本包含 Dynamic Quants 2.0。
- **Qwen3 的 RAG 困境**：成员们指出 LM Studio 内置的 RAG 实现可能无法提供最佳结果；“LM Studio 的 RAG 实现很糟糕”。
   - 他们建议直接复制粘贴文本，或实施自定义 RAG 解决方案以获得更好的性能。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1366646686860054538)** (61 messages🔥🔥): 

> `Framework Desktop vs. Flow Z13, AMD GPU 7900 XTX Value, Qwen3-30B-A3B Issues, MLX vs. llama.cpp Speed, Xeon Workstation for $1k` 


- **Framework Desktop 与 Flow Z13 的辩论**：成员们讨论了价值 2000 美元的顶配 **Framework Desktop** 与 **Flow Z13** 的性价比，批评 Framework 在电源和型号上对客户“变相加价”（nickel and diming）。
   - 讨论强调了对散热和 TDP 的担忧，普遍观点认为该**芯片太贵**，等待下一代产品可能更好。
- **7900 XTX：仍是最佳 AMD GPU？**：**AMD GPU 7900 XTX** 被誉为最强 AMD GPU，提到二手售价约为 **750€**，可提供约 **4080 Super** 的性能。
   - 值得注意的是，它多出 **8GB** VRAM，对于需要更大显存容量的用户来说是一个极具吸引力的选择。
- **Qwen3-30B-A3B 与电脑重启问题**：一位用户报告在使用 **Qwen2.5-coder-32b-instruct-q4_k_m** 时，电脑每隔 **30-60 分钟** 重启一次，并询问这是否与 GPU 空闲占用有关。
   - 潜在原因被推测为模型在加载但未进行交互时，依然对 GPU 造成了较大压力。
- **MLX 在 Prompt 处理速度上超越 llama.cpp**：据报道，在处理 **Qwen3-30B-A3B** 的 Prompt 时，[MLX](https://github.com/ml-explore/mlx) 的速度比 **llama.cpp** 快两倍以上。
   - 这一结论在 [Reddit 帖子](https://old.reddit.com/r/LocalLLaMA/comments/1kaqnbj/speed_with_qwen3_on_mac_against_various_prompt) 中被提及，用户对比了 Mac 上的性能表现。
- **售价 1000 美元的 Xeon 性能怪兽**：提到一台配备 **256GB RAM** 的 **40 核 Xeon** 工作站售价约为 1000 美元，为高内存计算提供了极具性价比的解决方案。
   - 一位用户链接了一个 [定制 Lenovo ThinkStation P720 配置](https://pcserverandparts.com/build-your-own-custom-lenovo-thinkstation-p720-workstation-2-processors/) 作为示例。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1366869610040922263)** (1 messages): 

> `Rate Limit, 2.5 Flash, Capacity` 


- **2.5 Flash 速率限制问题已解决**：遇到 **2.5 Flash** 速率限制（Rate Limit）问题的用户现在应该会感到明显好转，因为该模型已增加了额外容量。
   - 容量的增加旨在缓解之前的限制，并提供更流畅的用户体验。
- **提升 2.5 Flash 模型的容量**：已为 **2.5 Flash** 模型分配了更多容量，以解决并改善速率限制问题。
   - 此次升级旨在为用户在使用 **2.5 Flash** 模型时提供更可靠、更高效的体验。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1366627872403619974)** (321 messages🔥🔥): 

> `Qwen3 编程能力、Gemini 2.5 Flash 问题与速率限制、OpenRouter 缓存问题、LLama 4 基准测试、Vertex Token 计数问题` 


- **Qwen3：优秀的编码器但存在问题**：成员们讨论了 **Qwen3** 的编程能力，一位用户发现它在解释方面*非常出色*，而另一位用户指出了[复杂数学任务中的问题](https://huggingface.co/models)。
   - 一位用户通过*进一步降低 temp* 解决了复杂数学任务，而另一位提到了 **Qwen3 Tool Calling** 的问题。
- **Gemini 2.5 Flash 面临速率限制和错误**：用户报告称 **Gemini 2.5 Flash** 正面临**速率限制**和**错误**，即使是付费版本也是如此。一位用户在未使用网页搜索的情况下也遇到了此问题，而另一位指出了[免费使用 Gemini 2.5 Pro](https://ai.google.dev/gemini-api)的方法。
   - 据澄清，**OpenRouter** 正面临持续的 **Vertex Token 计数问题**，并进一步说明 OpenRouter **不支持** [Free Tier 限制](https://aistudio.google.dev/)。
- **OpenRouter 缓存仅限于 2.0 Flash**：一位用户指出 **OpenRouter 缓存**目前**不支持 2.5**，仅支持 2.0 Flash，且 2.5 Flash 会报错（**No endpoints found that support cache control**）。
   - 一位成员询问关于缓存多个 Prompt 的问题，**Toven** 澄清说，新的缓存会为新的 5 分钟 TTL 写入，缓存可以提高延迟（Latency）但**不会影响价格**。
- **LLama 4 在新基准测试中表现不佳**：一项基准测试评估显示 **LLama 4 表现很差**，但有人指出这仅仅是一个基准测试的结果。
   - 进行基准测试的人补充说，[**25 范围内的 ELO 差异在统计学上并不显著**](https://github.com/paradite/eval-data)，无法区分优劣。
- **辩论兴起：9.9 比 9.11 大吗？**：一篇 X 帖子的公告显示，一个模型声称 **9.9 大于 9.11**，引发了一些人思考这是否正确。
   - 其他人提到这*取决于上下文*，因为 [**Tesla FSD 版本的工作方式不同**](https://x.com/elonmusk/status/1917099777327829386)，且 9.11 > 9.9。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1366625096676020305)** (186 messages🔥🔥): 

> `Qwen3 模型、Aider 与 Qwen3 集成、ktransformers VRAM 优化、Deepseek R2 发布` 


- **Qwen3 在新款 Macbook 上的运行速度极快**：新款 Macbook 运行 **Qwen3 30B A3B** 的 Token/s 表现良好，部分用户报告使用 mlx 的速度达到约 **100/s**。
   - 用户希望能有一个本地编辑器 LLM，既能输出极快，又能在 Aider 的上下文中表现出色，特别是如果 **Qwen3-30B-A3B 的 4-bit 量化版本**在 Aider 基准测试中仍能表现尚可。
- **ktransformers 优化了 MoE 模型的 VRAM 使用**：[ktransformers 库](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)声称能以较低的 VRAM 需求高效运行 **Mixture of Experts (MoE)** 模型。
   - 他们声称仅需 **8 GB VRAM** 即可达到不错的速度，这比一次性将所有参数加载到 VRAM 中更适合 **30B-A3B** 模型。
- **Deepseek R2 热度攀升**：传闻称即将发布的 **Deepseek R2** 将具有增强的人类视觉能力和自学习功能，可能在*明天*发布。
   - 一些成员焦急地等待，因为他们*相信传言*称 **Deepseek R2** 定于明天发布。
- **新 PR 为 Aider 添加了思考状态指示器 (Thinking Spinner)**：一位新贡献者提交了一个 [PR](https://github.com/Aider-AI/aider/pull/3911)，为 Aider 添加了一个 *🔃 Thinking* 旋转图标，在等待 LLM 输出时显示。
   - 贡献者解释说，这让 Aider 感觉更加*敏捷且有生命力*。
- **Qwen3 Tool Use 表现出色，但在 Aider 中的应用尚不确定**：一些成员报告说 **Qwen3** 的 Tool Use 能力非常强，但由于 Tool Call API 的原因，其在 Aider 中的应用尚不确定。
   - 虽然 Tool Use 可能无法直接应用，但其他人建议使用**多智能体 (Multi-Agent) 工作流**，其中负责 Tool Use 的微智能体 (Microagents) 使用 Qwen3。

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1366664882425958461)** (21 messages🔥): 

> `AiderDesk Agent Mode, Repo Map Control, OpenRouter Model Support, Gemini 2.5 + Deepseek combo` 


- ****AiderDesk** 的 Agent 模式表现出色**: 一位用户正在使用 **AiderDesk** 的 Agent 模式，根据其 [GitHub](https://github.com/hotovo/aider-desk) 介绍，他们先使用 "probe" 进行规划，准备就绪后再启用 "Use Aider tools"、"Include context files" 和 "Include repomap"。
   - 他们还使用了 **Jira** 管理和 **desktop-commander** 来运行命令，但目前还没怎么使用 **memory-bank** 或 **context7**。
- **使用 **Aider** 调整 **Repo Map****: 用户希望在 **repo map** 中仅包含 API 代码，而不包含注释或测试，并询问是否可以通过 `aider --map-tokens 0` 禁用后两者。
   - 另一位用户建议使用 `repomix --compress` 或 `probe` 作为替代方案，并指出目前原生并不支持对 repo map 进行细粒度控制。
- ****OpenRouter** 模型受支持，但并非总是成功**: 有用户询问 **Aider** 是否可以使用 **OpenRouter** 上的任何模型，另一位用户确认所有 **OR** 模型均受支持。
   - 他们还补充说，如果你使用的是 `gemma 3 1b` 或 `smollm`，不要抱有太高期望。
- ****Gemini 2.5 + Deepseek** 强力组合**: 用户发现了一个好用的组合：使用 **Gemini 2.5** 进行规划，使用 **Deepseek** 进行 diffs 和 vchanges 解释。
   - 他们建议在 **AI Studio** 中执行此操作，因为 Gemini 在那里是免费的。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

p0lyg0n: 关于 Deepseek 的优秀纪录片：https://www.youtube.com/watch?v=Lo0FDmSbTp4
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1366681332901613568)** (10 messages🔥): 

> `Apple Silicon, Cloud GPUs, CUDA, Metal, ROCm` 


- **Apple Silicon 不是参加云端挑战赛的障碍**: 一位拥有 **M4 Max PC** 的用户对参加挑战赛表示担忧，但另一位用户澄清说挑战赛在**云端**运行，因此 **Apple Silicon** 并不是障碍。
   - 他们建议查看相关频道以获取更多信息。
- **云端 GPU 助力远程 CUDA/ROCm 学习**: 用户解释说，虽然使用本地算力学习 **CUDA** 或 **ROCm** 更容易，但使用**云端 GPU** 仍然可行。
   - 他们指出，现在廉价的云端 GPU 资源越来越丰富。
- **在 Mac 上进行 Metal 编程是可行的**: 用户肯定了在 Mac 上使用 **Metal** 编写 GPU 程序完全没问题。
   - 他们补充说，这更多地取决于你对工具的熟悉程度，并分享了一个 **Metal** 代码片段。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1366837666850541688)** (2 messages): 

> `fp8 quantization, fp32 accumulation, Triton matmul, Custom CUDA kernels, AMD` 


- **对 FP8 量化配合 FP32 累加的疑问**: 一位成员询问是否可以使用 **Triton** 为 **matmul** 操作执行 **fp8 量化**和 **fp32 累加**，或者是否必须使用自定义 **CUDA kernels**，特别是在 **AMD** GPU 上运行时。
- **通过 num_stages 参数实现双缓冲**: 用户询问将 `num_stages` 设置为大于 1 是否本质上在 **Triton** 中启用了**双缓冲**。
   - 他们提到 **MI300** 不像 **Ampere** 那样具有异步加载功能，推荐设置是 `num_stages=2`，并想知道 `num_stages > 2` 是否会有所帮助。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1366653061052891168)** (5 messages): 

> `Torch Logger Methods Compilation, AOT Inductor Multithreading` 


- **Torch Loggers 触发编译问题**: 用户询问如何在编译期间忽略 **logger 方法**，以避免与 **PyTorch distributed 模块**中的 `FSDP::pre_forward` 相关的异常。
   - 另一位成员建议将 `TORCH_LOGS` 环境变量设置为 `output_code` 或 `tlparse`，以检查生成的代码并识别导致问题的潜在 **if 语句**，并引用了 [`torch._dynamo.config.py` 中的特定行](https://github.com/pytorch/pytorch/blob/797768cd90d0984687e15f5fe0e1a4d8bf91d71a/torch/_dynamo/config.py#L506)。
- **AOT Inductor 在 C++ 中的训练难题**: 用户报告使用 **AOT Inductor** 实现了部分 C++ 训练设置，但怀疑存在多线程问题。
   - 他们推测问题源于代码中不必要的特化（specialization），并计划开启一个 [PyTorch issue](https://github.com/pytorch/pytorch/issues) 供 **AOTI 作者**进一步调查，特别是担心多个工作线程调用 `fw_graph->run()` 时的 API 行为。


  

---

### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1366888480302567565)** (1 messages): 

> `AMD MI300 competition, MoE kernels, FP8 submissions` 


- **为 AMD MI300 竞赛发布新的单 GPU MoE Kernel**：一个新的单 **GPU MoE kernel** 问题现已在 **$100K AMD MI300 竞赛**中上线；请在 [排行榜](https://www.gpumode.com/leaderboard/430) 上查看。
   - 一位成员建议，由于这个问题比较棘手，值得阅读提供的 [详细解释](https://tinyurl.com/amd-comp-moe)。
- **AMD MI300 竞赛关键日期**：注册于 **4 月 30 日**截止，而包含 **FP8** 和 **MoE kernels** 的最终提交截止日期为 **5 月 27 日**。
- **排行榜运行时间较慢**：对于该问题，运行 `leaderboard submit ranked` 会很慢，耗时约 **8 分钟**。
   - 提交者建议使用 `leaderboard submit test/benchmark` 以进行更快的迭代。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

raymondz4gewu_60651: `/get-api-url`
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1366769852425175102)** (22 messages🔥): 

> `Quantized Models and torch.bfloat16, vllm Compile Integration Debugging, gemlite Kernel Selection, torch.compile Debugging Challenges, torch.dtype Extensibility` 


- **量化模型重新加载为 `torch.bfloat16`**：量化模型在使用量化布局保存后，重新加载时会显示为 `torch.bfloat16`，因为原始 dtype 被保留了。
   - 实际的量化 dtype 可以通过打印权重来访问，因为 PyTorch 的 `torch.dtype` 目前还不支持扩展到 tensor 子类；更多讨论见 [此处](https://github.com/pytorch/ao/issues/442)。
- **`vllm` 编译集成难题**：在与 [gemlite 库](https://github.com/mobiusml/gemlite/) 集成时，`vllm` 的编译函数出现问题，使用 `torch.compile` 会导致错误行为。
   - 具体而言，`vllm` 无法从 gemlite 中选择正确的 kernel（该选择基于输入形状）；由于 `torch.compile` 的局限性，在其内部进行调试非常困难。
- **`gemlite` 中的 Kernel 难题**：核心问题在于 `gemlite` 内部的 kernel 选择错误，这可以追溯到 `vllm` 使用 `torch.compile` 时输入形状未被正确识别。
   - kernel 选择逻辑基于输入形状，如 [gemlite 的 core.py](https://github.com/mobiusml/gemlite/blob/master/gemlite/core.py#L386) 中所定义，因此形状检查对调试至关重要。
- **`torch.compile` 调试困境**：传统的调试方法（如 print 语句和断点）在 `torch.compile` 内部无效，增加了检查变量状态的难度。
   - 使用 `TORCH_LOGS=+dynamo` 可以转储包含形状的图（graph），有助于调试，[PyTorch 文档](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html#breakpointing-dynamo-tracing) 也提供了关于 dynamo 追踪断点的指导。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1366728443273936958)** (3 messages): 

> `ROCm memory, CDNA3 ISA` 


- **ROCm 内存库（Memory Banks）大小说明**：假设 32 位对齐，ROCm 中的内存库宽度为 **32 位**。
   - 内存库通过 `address % bank_size` 计算。
- **CDNA3 ISA 参考文档详细说明 LDS 配置**：**CDNA3 ISA Reference** 第 2.2.1 节指出，每个计算单元（compute unit）拥有 **64kB** 的内存空间用于低延迟通信。
   - 该内存配置为 **32 个 bank**，每个 bank 有 **512 个条目（entries）**，每个条目 **4 字节**。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1366802069096239276)** (3 messages): 

> `QR decomposition, SIMD, Thread barriers, Single-threaded SVD` 


- **128 位 QR 分解令人惊叹**：一位成员分享了一个非常出色的 **128 位精度** QR 分解实现，在 [链接的 Python 脚本](https://cdn.discordapp.com/attachments/1285384841730457600/1366802750817697853/ember_ml_svd_128bit.py?ex=681245c1&is=6810f441&hm=657c03f2fc77e181231bcfd8c0dbe87a034b5f0bd2c941fa48ecea7088a71f1f&) 中使用了 **SIMD** 和 **线程屏障（thread barriers）**。
- **加速单线程 SVD**：一位成员报告在 **SVD** 中发现了单线程模式，并指出他们正在修复该问题以实现更好的并行化。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1366723501729382441)** (3 messages): 

> `GPU Price Tracker, AI/ML Engineer for Hire, Open Source IDE for AI/ML` 


- **在 Amazon 上追踪 GPU 价格**：一名成员构建了一个 [GPU 价格追踪器](https://www.unitedcompute.ai/gpu-price-tracker)，该工具可以提取 GPU 的完整 **Amazon 价格历史**并生成精美的图表。
   - 它会计算最新的数值，例如每美元可以获得多少 **teraflops**；一个使用场景是寻找购买私有集群的最佳时机。
- **AI/ML 工程师求职**：一位拥有 **8 年经验**，擅长人工智能、机器学习、全栈和移动端开发的 AI/ML 工程师正在求职；其专业领域涵盖深度学习、自然语言处理和计算机视觉，能够将尖端的 AI 解决方案集成到可扩展且健壮的应用中。
   - 提供了其 [LinkedIn 个人资料](http://www.linkedin.com/in/lucy-hunter-40a527350)和[作品集](https://lucyhunter.vercel.app/)的链接，以及技能列表，包括 **ML 算法、Deep Learning、NLP、Computer Vision、MLOps 和 AI 模型集成**。
- **开源 IDE 项目启动**：一位成员正在为 AI/ML 工程师构建一个开源 IDE，并正在寻找合作者；如果你对细节感兴趣、想加入或有见解，请私信他们。
   - 该成员提供了其 [LinkedIn 个人资料](https://www.linkedin.com/in/bruno-scaglione-4412a0165/)和 [GitHub 个人资料](https://github.com/BrunoScaglione)的链接。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1366881755864498197)** (1 messages): 

> `Use Cases, Performance` 


- **用户询问使用案例和性能**：用户正在询问具体的使用案例以及实现后的性能指标。
- **对实现细节表现出浓厚兴趣**：大家*非常想听听你使用它的进展如何*，特别是关于实际结果方面。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1366839192604119215)** (15 messages🔥): 

> `FP8 quantization material, FP8 matmul, Deepseek-v3 tech report, prefixsum ranked timeout` 


- ****发起 FP8 量化探索****：一位成员询问了关于 **FP8 量化**的资源，特别是关于 **带有 FP32 累加的 FP8 matmul** 的优势，并[链接了 onnx fp8 格式页面](https://onnx.ai/onnx/technical/float8.html)。
   - 他们引用了 **Deepseek-v3** 的技术报告，指出 **FP8** 可能会遇到**欠载（underflow）问题**，因此需要更高精度的累加器。
- ****Prefixsum 排名超时排查****：一位成员报告了频繁的超时问题，特别是 **ranked prefixsum 提交**，尽管有 **30 秒的超时限制**。
   - 工作人员承认了该问题，将其归因于他们自己的错误，并随后声称已解决，但该成员仍然遇到超时，随后通过私信发送了代码。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1366626400379076649)** (60 messages🔥🔥): 

> `vectoradd benchmark on H100, amd-fp8-mm benchmark on MI300, amd-mixture-of-experts benchmark on MI300, prefixsum benchmark on H100, A100, matmul benchmark on L4` 


- **H100 VectorAdd 速度竞相探底！**：`vectoradd` 排行榜在 **H100** 上收到了多次提交，时间从 **540 µs** 到 **708 µs** 不等，其中一次提交以 **540 µs** 获得第三名。
- **MI300 AMD-FP8-MM 排行榜升温！**：大量提交涌入 **MI300** 上的 `amd-fp8-mm` 排行榜，包括一个 **196 µs** 的第三名，个人最佳成绩在 **2.37-2.43 ms** 左右，成功的运行结果从 **198 µs** 到 **8.05 ms** 不等。
- **AMD Mixture of Experts 夺得榜首！**：**MI300** 上的 `amd-mixture-of-experts` 基准测试出现了一个 **6228 ms** 的第一名提交，以及多个在 **7379-7490 ms** 左右的第二名提交。
- **Prefixsum 在 H100 和 A100 上并驾齐驱！**：`prefixsum` 排行榜出现了多个第二名提交：一个在 **A100** 上为 **1428 µs**，另有几个在 **H100** 上约为 **955-985 µs**。
- **L4 MatMul 桂冠虚位以待！**：**L4** 上的 `matmul` 排行榜创下了 **2.27 ms** 的新第一名，而另一个提交以 **49.3 ms** 获得第二名。


  

---

### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1366684786206441514)** (2 messages): 

> `Single GPU MoE Kernel, FP8 and MoE Kernels, Leaderboard Submissions` 


- **Single GPU MoE Kernel 问题已上线！**：新的 Single GPU MoE Kernel 问题现已发布，请查看 [Leaderboard](https://www.gpumode.com/leaderboard/430)。
   - 官方提供了一份更详尽的说明，建议通过[此链接](https://tinyurl.com/amd-comp-moe)仔细阅读。
- **重要日期提醒**：注册将于明天 **4 月 30 日**截止，**FP8** 和 **MoE Kernels** 的提交截止日期均为 **5 月 27 日**。
   - 请注意，针对此问题运行 `leaderboard submit ranked` 会比较慢（约 **8 分钟**），因此请使用 `leaderboard submit test/benchmark` 进行快速迭代。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1366649146978730074)** (23 messages🔥): 

> `Aithe 参考代码, FP8 正确性验证, Submission ID, 该 Kernel 的官方问题说明` 


- **Aithe 参考代码**：一位成员询问 **Aithe 参考代码** 是否会开源，并对 **FP8** 能否通过逐元素完全相等检查（element-wise perfect equal checks）表示怀疑；[参考代码](https://github.com/gpu-mode/reference-kernels/blob/b68a149bcd8701532eeedc774d27062429ce4f99/problems)随后被迅速提供。
   - 回复中澄清，对比并非逐元素完全相等检查，并指向了使用 `rtol=2e-02, atol=1e-03` 的[相关函数](https://github.com/gpu-mode/reference-kernels/blob/b68a149bcd8701532eeedc774d27062429ce4f99/problems/amd/utils.py#L31)。
- **找回丢失的 Ranked 代码**：一位在本地丢失了 Ranked 代码的成员寻求帮助，另一位成员建议使用 `/leaderboard show-personal` 和 `/leaderboard get-submission` 来找回。
   - 丢失的提交已通过其 ID (`11105`) 确认，该成员被引导使用 `/get-submission` 命令。
- **第二个问题推迟**：成员们讨论了即将发布的第二个问题，确认在完成额外测试后将很快发布，FP8 通道不会关闭。
   - 分享了[该 Kernel 的官方问题说明](https://tinyurl.com/amd-comp-moe)链接。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/)** (1 messages): 

vkaul11: 是否有现成的 Kernels 可以执行 FP8 乘法并进行 FP32 累加？
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1366641263343964170)** (88 messages🔥🔥): 

> `ChatGPT 持久化记忆, AI Agent 公司, IAM360 框架, AI 生成的缩略图` 


- **ChatGPT 获得初级持久化记忆**：ChatGPT 开发了**两种类型的持久化记忆**：一种是从它认为重要的对话细节中提取的长期记忆（训练数据），另一种是参考过去 **90 天**上下文的短期记忆。
   - 用户可以关闭长期或短期记忆，但一个开关无法同时控制两者。
- **AI Agent 公司令人啼笑皆非的混乱结果**：教授们尝试让 AI Agents 完全运营一家虚假公司，但[*结果极其混乱且令人发笑*](https://futurism.com/professors-company-ai-agents)，这表明目前的 AI 模型无法完全取代人类工作。
   - 尽管科技巨头宣称如此，但 AI 模型尚未达到完全取代人类所需的水平，仍需要人类监督。
- **IAM360：一种模块化符号化 GPT-Agent 架构**：一位成员正在开发 **IAM360**，这是一个用于人机共生的实验性框架，使用标准 ChatGPT 会话构建，无需自定义 GPTs、微调或 API 集成。
   - 该系统使用具有持久角色（战略、执行、财务、情感）的**模块化符号化 GPT Agents**，以及一个用于自然涌现对话的 **Zero-shot 编排系统**。
- **以 Robux 出售 AI 制作的缩略图**：一位成员报告以 **1500 Robux** 的价格售出了一张 AI 制作的缩略图。
   - 其他成员表示，目前的生成器如果给定复杂的参考图，会把图像搞得一团糟，现实世界中的客户不会为此买单。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1366647417742557194)** (29 messages🔥): 

> `ChatGPT 中的身份系统, RP 中的动态 Game Master 角色, ChatGPT 内部工具, Prompt Engineering 技巧, LLM TTRPG 游戏开发` 


- ****记忆至关重要**：ChatGPT 中的身份系统**：一位成员讨论了为 ChatGPT 创建身份系统，以便[按身份分隔记忆/历史聊天](https://discord.com/channels/974519864045756446/1171946823369711676)，从而保留静态身份和状态。
   - 目标是避免用户陷入叙事低谷，即要么抹除记忆，要么试图逃离此类场景。
- ****Game Master 动态**：角色扮演冒险**：一位成员分享了一个 Prompt，让 ChatGPT 在[奇幻角色扮演冒险中担任动态 Game Master](https://discord.com/channels/974519864045756446/1171946823369711676)。
   - 重点在于扮演非用户角色，根据主角的经历演化世界，并在世界观构建、角色对话和行动之间保持平衡。
- ****Bio 工具揭秘**：ChatGPT 的记忆**：一位成员透露 ChatGPT 的内部记忆被称为 `bio` 工具，[建议在定义保存命令时调用其规范名称](https://discord.com/channels/974519864045756446/1171946823369711676)。
   - 建议了一个改进版的 `/pin` 命令：*AI 使用 `bio` 工具将最近的消息保存到 ChatGPT 的内部记忆中，保留所有基本细节以供未来参考。*
- ****完美 Prompt**：GPT 的内部工具**：一位成员建议[要求模型识别并描述其每个连接工具的功能](https://discord.com/channels/974519864045756446/1171946823369711676)，列出它们的规范名称，并为每个工具提供一个显示其正确语法的代码块。
   - 提到的工具包括 **python, web, image_gen, guardian_tool, 和 canmore**。
- ****RPG 起源**：通用 AI 框架开发**：成员们提到了他们从 LLM TTRPG 游戏开发到[通用 AI 框架开发](https://discord.com/channels/974519864045756446/1171946823369711676)的历程。
   - 一位成员强调，这条路径可以通向学术研究。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1366647417742557194)** (29 messages🔥): 

> `ChatGPT 中的身份系统, RP Prompt 问题, 动态 Game Master 角色, ChatGPT 内部记忆 (bio 工具), LLM TTRPG 游戏开发` 


- ****人格持久性困扰玩家****：用户正面临 **ChatGPT** 在角色扮演场景中抹除记忆或陷入“叙事低谷”的问题，这阻碍了静态身份和一致角色状态的创建。
   - 无法维持持久身份迫使用户不断重置或规避不理想的叙事路径。
- ****定义 Game Master (GM) 角色****：一位成员为 **ChatGPT** 在奇幻角色扮演中定义了一个动态 **Game Master (GM)** 角色，重点是扮演一个与用户主角互动的非玩家角色 (NPC)，并根据主角的经历演化世界。
   - GM 应平衡世界观构建、对话和行动，避免过分详细，并使用特定命令如 `/export_character`、`/export_world_state`、`/force_random_encounter` 和 `/set_mood` 来管理游戏。
- ****精准定位 ChatGPT 的 Bio 工具****：该成员确认 **ChatGPT** 的内部记忆为 `bio` 工具，建议他人在保存命令中使用此规范名称，以确保 pin 功能能够正确保存基本细节，供未来通过 `/pin` 参考。
   - 他们建议将命令放置在 Prompt 顶部附近，并使用间隙重复（gapped repetition）来提高遵循度。
- ****诞生于幻想的框架****：一位成员分享了他们的 AI 之旅始于 **LLM TTRPG 游戏开发**，随后转向通用 AI 框架开发，最后进入学术研究。
   - 他们现在正致力于为特定任务创建一个 **GPT**，以便更好地将 LLM 整合进一个完整的框架大纲中。
- ****驯服文本生成技术的技巧****：一位成员建议在 Prompt 中加入具体规范，以减少 **LLM** 的猜测，并要求模型识别和描述其连接的工具，列出它们的规范名称并演示正确的语法。
   - 他们提供了如何查询模型工具（如 **python**、**web**、**image_gen**、**guardian_tool** 和 **canmore**）的示例，并给出了调用它们的特定语法。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1366662361729208400)** (79 messages🔥🔥): 

> `PyQt5 Chat App, OR vs ML history, Gemini 2.5 Pro vs GPT-4o, Qwen 3 performance, FFN in Transformers` 


- ****PyQt5** 聊天应用引起关注**: 一位成员分享了一个使用 **PyQt5** 构建的 [AI 聊天应用程序](https://cdn.discordapp.com/attachments/986699377257119794/1366717106447450122/AI_chat_app_005.py?ex=6811f5fe&is=6810a47e&hm=d9601e58ece57f0a5ba85d7da1c73f099068ee63c9220099dffa3614c74cd9bd&)，并使用 **LM Studio** 作为后端服务器。
   - 要使用该应用，用户必须在运行程序之前，在 **LM Studio** 上选择并启动模型作为服务器。
- ****ML** 的 **OR** 起源在辩论中被理清**: 围绕 **Operations Research (OR)** 与 **Machine Learning (ML)** 之间的历史关系展开了讨论，一位成员指出 *ML 源自统计学*。
   - 另一位成员反驳称，早期的 **AI/ML** 与 **Operations Research** 和控制理论非常接近，但后来分支出来拥抱统计方法，特别强调*从数据中学习，而不是根据第一性原理对现实进行建模*，现代 ML 具有极强的经验主义色彩。
- ****Gemini 2.5 Pro** 在与 **GPT-4o** 的对比中遭到吐槽**: 成员们讨论了 **Gemini 2.5 Pro** 与 **GPT-4o** 的性能对比，一位用户称 Gemini 为 *4otard*。
   - 另一位表示，*Gemini 2.5 Pro 肯定比 4o 差*，认为它可能在编程方面更好，但在通用场景下不如后者，其他用户也发现 **GPT-4o-mini** 在聊天体验上比 **Gemini 2.5 Flash** 更好。
- ****Qwen 3**：新模型以推理能力令用户兴奋**: 成员们赞扬了新的 **Qwen 模型**，特别是提到的推理和指令遵循能力的提升。
   - 一位用户报告称，*它们在某些推理任务中的输出*更胜一筹，理由是其客观性以及对指令的严格遵守，特别称赞了 MoE 模型的速度和智能，将其描述为*即使不比 2.5 Flash 更聪明，也至少旗鼓相当*。
- ****FFN** 功能引发讨论与审视**: 一场关于 **Feed-Forward Networks (FFN)** 在 Transformer 架构中作用的讨论展开，一位用户寻求关于其功能的直观理解。
   - 一些人建议 **FFN** 实现了信息在通道/神经元维度的混合，增加了容量和非线性，一位成员引用道：*拥有 FFN 本身比它的宽度要重要得多*。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1366800354993246350)** (8 messages🔥): 

> `DeepSeek VL, Construction` 


- **施工导致 DeepSeek VL 讨论取消**: 成员家附近的施工导致会议被迫取消。
   - 讨论 **DeepSeek VL** 的会议将移至第二天。
- ****DeepSeek VL** 讨论将重启**: 之前的 **DeepSeek VL** 讨论仅涵盖了引言部分，因此成员们将从头开始重新进行论文讨论。
   - 团队计划戴上降噪耳机重新开始。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1366639817126973500)** (34 条消息🔥): 

> `Reddit 上的匿名 LLM、ChatGPT 的说服技巧、Meta 的 LlamaCon 2025、Llama 4（又名 Little Llama）、SAM 3 开发` 


- **匿名 LLM 愚弄了 Reddit 的 change-my-view 版块**：研究人员在 Reddit 的 **/r/changemyview** 上测试了一个匿名 LLM，发现其*效力极高*，导致用户感到恼火，详见 [此 X 帖子](https://x.com/emollick/status/1916905103358931084) 和 [Reddit 线程](https://www.reddit.com/r/changemyview/s/k9Rd6IbyjY)。
   - 一位用户幽默地表示：*AI 并不聪明，改变我的想法吧*，**ChatGPT** 回复道：*不，它们很聪明*，该用户随后回复：*噢好吧，对不起*。
- **ChatGPT 擅长哲学对话**：一位成员发现，让 **ChatGPT** 反驳自己的信仰或为他们感到厌烦的事实辩护既*有趣又有教育意义*。
   - 他们指出，虽然 **O1-preview** 在*日常对话中显得枯燥*，但 **O3/O4-mini-high** 模型适用于一般话题，他们现在使用 **o4-mini-high** 进行新闻分析。
- **Meta 举办 LlamaCon 2025**：**Meta** 举办了 **LlamaCon 2025**，这是一场生成式 AI 开发者大会，可通过 [Engadget](https://www.engadget.com/ai/llamacon-2025-live-updates-from-metas-first-generative-ai-developer-conference-keynote-215241436.html) 和 [官方直播](https://www.facebook.com/MetaforDevelopers/videos/1792349135036347/) 获取实时更新。
- **Llama 4（又名 Little Llama）确认发布**：在 **LlamaCon** 上确认了 **Llama 4**（也被称为 *Little Llama*）的存在，详见 [此 YouTube 直播](https://www.youtube.com/live/6mRP-lQs0fw)。
   - 一位用户开玩笑说要把它们叫做 *Baby llama*，而另一位用户则表示失望，认为这些公告*毫无实质内容*。
- **SAM 3 正在开发中**：**LlamaCon** 的一个重要公告是 **SAM 3** 的开发以及 **Meta** 的新应用。
   - 一位用户思考 **Little Llama** 模型将如何与 **Qwen** 模型竞争。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1366857710435569766)** (1 条消息): 

> `Atropos RL 框架、RLAIF 模型、GRPO 工具调用、公司基本面预测、Psyche 去中心化训练网络` 


- **Atropos 框架突破 RL 障碍**：Nous Research 发布了 **Atropos**，这是一个配合基础模型的 [强化学习 Rollout 框架](https://github.com/NousResearch/Atropos)，支持复杂环境以提升模型能力。
   - Atropos 是其整体 RL 系统设计的一部分，很快将补充训练和推理组件，详见其 [介绍博客文章](https://nousresearch.com/introducing-atropos/)。
- **GRPO 工具调用提升 DeepHermes 性能**：他们在 **GRPO** 环境下将 **DeepHermes** 的工具调用能力在简单和并行工具调用上分别提升了 **2.4 倍**和 **5 倍**（基于 Berkeley Function Calling Benchmark）。
   - 使用 Atropos 环境创建的产物，包括一个新数据集和五个用于工具调用、公司基本面预测以及具有 RLAIF 实验性人格的新模型，已在 [HuggingFace](https://huggingface.co/collections/NousResearch/atropos-artifacts-68110ab511c5af02830247b6) 上发布。
- **基本面预测模型准确率翻倍**：使用 Atropos 后，公司基本面预测模型在方向性变化上的准确率从 **~25%** 提高到 **50%**。
   - Atropos 框架旨在通过强化学习引导语言模型发挥其最佳潜力，就像希腊命运女神引导灵魂走向最终归宿一样。
- **Psyche 网络实现去中心化训练**：Atropos 是 **Psyche** 的关键组成部分，**Psyche** 是一个即将推出的去中心化训练网络，负责协调全球范围内的预训练、中训练（mid-training）和后训练工作负载。
   - 将于 5 月 18 日在旧金山举办一场黑客松，以促进协作进展（更多细节即将公布）。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1366625144214392833)** (110 条消息🔥🔥): 

> `Qwen 3 Overfitting, DeepSeek R2 Release, Huawei Ascend 910B, Atropos Release, Minos Model Refusals` 


- **Qwen 3 基座模型在评估集上过拟合**：成员们发现 **Qwen 3 的基座模型** 似乎在某些评估集（evals）上严重过拟合。据报告，**M24b** 模型在 **Trivaqa** 上的得分为 **75%**，而 **Q30MoE** 仅为 **60%**。
   - 一位成员指出，他们在 **30B-A3** 和 **32B-dense** 之间的 benchmark 结果确实非常接近，这可能是由于某种程度的过拟合造成的，这引发了关于 MoE 有效性的讨论。
- **DeepSeek R2 发布传闻四起**：有传言称 **DeepSeek R2** 可能很快发布，一些报告声称它完全是在 **Huawei Ascend 910B** 硬件上训练的，这可能会减少对 **Nvidia CUDA** 的依赖。
   - 然而，其他人反驳了这些说法，并引用了一条 [推文](https://fxtwitter.com/teortaxesTex/status/1916325875437445243)，表示 **DeepSeek** 的官方口径是：“我们会在发布 R2 的时候发布它，任何声称自己知道内情的人都在撒谎”。
- **Nous Research 发布 Atropos**：[Nous Research 发布了 Atropos](https://github.com/NousResearch/atropos)，这是一个开源项目和推理优化技术。
   - 为使用 **Atropos** 的开发者创建了一个新频道 <#1365222663324307466>。
- **Minos 模型与能力相关的拒绝回答**：一位正在试用 **Minos** 的成员想知道是否应该有一种方法将与能力相关的拒绝（refusals）与其他类型的拒绝区分开来，并担心这可能会增加幻觉（hallucinations），因为模型可能会认为自己具备实际上并不拥有的能力。
   - 讨论中区分了模型是“不能”还是“不愿”执行任务。
- **Physical AI 跑马拉松**：有人分享了一张 [Physical A.I. 机器人](https://cdn.discordapp.com/attachments/1149866623109439599/1366647197789323274/NoJoke.png?ex=68125da3&is=68110c23&hm=beab804046b63afebd36468c0257ad616184ba8bf7aed8feb39bac3da164077e) 的图片，它在去年的上海马拉松比赛中跑得比大多数人都好。
   - 评论者指出，“AI 现在真的在绕着我们跑了”，并附上了 [Prime Intellect 的 X 帖子](https://x.com/PrimeIntellect/status/1916994185573634336) 链接。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1366647358842208287)** (2 条消息): 

> `Image loading issues` 


- **图片加载困扰用户**：一位成员报告说图片一直处于加载状态，表明 **图片上传或加载时间可能存在问题**。
   - 该用户随后回复称已恢复正常（*Working*）。
- **用户确认图片加载已解决**：一位成员确认图片加载问题已得到解决。
   - 该成员简单地表示 *Working*。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1366633673096433715)** (101 条消息🔥🔥): 

> `用于在 Git 更改视图中过滤 .cs 文件的 VS Code Extension、Cursor 支出限制问题、模型选择目的、Anthropic 3.7 事件、Gemini 2.5 Pro 问题` 


- **削减支出限制导致请求变慢**：一位用户报告称，在达到支出限制并升级后，数小时内仍受困于**慢速请求**，另一位用户则耗尽了**快速请求 (fast requests)**。
   - 另一位用户补充道，即使在慢速请求下，**Gemini** 依然很快。
- **Cursor 社区 Discord：终于重新获得关注了吗？**：一位成员幽默地注意到 **Cursor 的 Discord** *终于再次受到关注了*。
   - 另一位成员自信地回应道 *Cursor 一直深受喜爱*，暗示团队只是在不断打磨产品。
- **Gemini 故障：模型在请求中途停止！**：用户报告称 **Gemini 2.5** 频繁在请求中途停止，尽管它表示将执行操作；另一位用户建议*当某个模型表现异常时，尝试使用不同的模型*。
   - 一位团队成员确认团队一直在与 Google 合作解决此问题，并建议用户在此期间使用其他模型，并提议用户将他们的 **request ID** 发送给团队进行调查。
- **Agent 懈怠：多次尝试后仍无法编辑！**：一位用户报告称 **Agent 在多次尝试后仍无法进行编辑**，存在严重问题，并转而指示用户手动操作。
   - 一位团队成员建议该问题可能是由 **Gemini 2.5 Pro** 引起的，并建议创建一个新聊天以刷新 **context**；他们建议使用 4.1 GPT 或 3.5 进行编码，如果出现问题则使用 3.7 Claude。
- **官方 Ollama 智能手机 App 何时发布？**：一位用户询问了官方 **Ollama 智能手机 App** 的发布时间表，并链接到了相关的 [X 帖子](https://x.com/awnihannun/status/1917258279455187034)。
   - 一位用户提到他们通过重新安装 Cursor 并清理缓存解决了问题，另一位用户确认可以手动清理缓存，从而避免重新安装过程。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1366638637948866570)** (43 条消息🔥): 

> `Cloudflare Turnstile、whisper-large-v3-turbo 问题、GGUF 模型与 CPU offloading、Model Context Protocol (MCP)、运行模型的最快推理方式` 


- **成员测试 Cloudflare Turnstile**：成员们测试了 [Cloudflare Turnstile](https://mineapi.pythonanywhere.com/docs) 是否有效，并得到了肯定的确认。
   - 确认后，该成员兴奋地喊道 *YIPEEEEEEEE*。
- **成员报告 Whisper Turbo 问题**：成员们报告 **OpenAI 的 whisper-large-v3-turbo** 在 HF 推理端点上无法工作，甚至网页上的演示也已挂掉。
   - 成员们链接了类似的问题，如[这篇讨论](https://discuss.huggingface.co/t/sentence-transformers-all-minilm-l6-v2-not-working-all-of-a-sudden/152691)以寻求潜在帮助。
- **合并时卸载到 CPU RAM 正常**：成员们讨论了在将 **checkpoint** 合并到基础模型时，将其卸载 (offloading) 到 CPU RAM 的情况。
   - 一位成员表示这没问题，并指出 **Transformers + Accelerate** 或 **Llama.cpp** 支持卸载，而且 **GGUF 格式本身就假定支持 CPU offloading**。
- **不同模型的推理速度对比**：成员们思考了 **Model Context Protocol (MCP)** 以及哪种方式运行模型的推理速度最快。
   - 有人指出 **Unsloth** 比 Hugging Face 更快，其他人则推荐使用 **sglang/lmdeploy** 或 **exllamav2**。
- **寻找活跃的 AI 黑客松和训练营**：一位成员询问是否有活跃的 **AI 相关训练营 (cohorts) 或黑客松 (hackathons)** 提供参与激励或奖励。
   - 在后续讨论中没有提供具体的推荐。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 条消息): 

cakiki: <@1298649243719958612> 请不要跨频道重复发帖 (cross-post)
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1366761093053022209)** (9 messages🔥): 

> `3D Animation Arena, Pi-Scorer alternative to LLM-as-a-Judge, HMR Models` 


- ****3D Animation Arena 开放用于 HMR 模型排名****：一名成员在 Hugging Face 上创建了一个 [3D Animation Arena](https://huggingface.co/spaces/3D-animation-arena/3D_Animation_Arena)，根据不同标准对模型进行排名，旨在为当前的 **HMR (human mesh recovery)** 模型建立排行榜。
   - 创建者正在寻求投票以填充排行榜。
- ****Pi-Scorer 作为 LLM-as-a-Judge 的替代方案出现****：一名成员分享了 **Pi-Scorer**，这是 **LLM-as-a-Judge** 的替代方案，并提供了 Colab 笔记本，展示如何将 **Pi-Scores** 用于 [模型检查点评估 (model checkpoint evaluation)](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/SFT_Model_Checkpoint_Observability_withPi.ipynb) 以及作为 [奖励函数 (reward functions)](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/PiScorer_as_GRPO_Reward_Function.ipynb)。
- ****AI 助手集成代码已共享****：一名成员分享了其 **AI assistant integration** 项目的 [代码](https://github.com/BouajilaHamza/site-ai-assistant-integration)。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1366836869610078282)** (2 messages): 

> `Defect annotation, Image masking, Filter usage` 


- **解决缺陷标注难题**：一名成员正尝试实现这篇 [论文](https://arxiv.org/pdf/2009.07047v1)，但在生成和标注旧划痕图像时面临挑战。
   - 该成员合成生成了带有划痕、模糊和灰度等缺陷的图像，目前正在寻求关于如何标注这些缺陷的建议。
- **掩码方法亮相**：一名成员建议对图像进行掩码（Masking）处理，在测试不同阈值以隔离划痕的同时将其二值化，并保持图像其余部分不变。
   - 该成员指出如何通过测试不同阈值来找到理想的平衡点。
- **过滤瑕疵**：一名成员建议使用 **Canny edge** 或 **Sobel** 等滤波器，通过特定阈值来隔离缺陷。
   - 这些滤波器在特定阈值下可以很好地隔离缺陷，从而更容易对数据集上的划痕进行自动标注。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1366625364310228993)** (40 messages🔥): 

> `Hugging Face Agents certification, Agents.json vs Prompts.yaml, Llama-3 access request, Models temporarily unavailable, Solving the final project with free resources` 


- **庆祝完成 HF Agents 课程！**：成员们庆祝完成 **Hugging Face Agents** 课程并获得认证，其中一名成员分享了他们的 [LinkedIn 个人资料](https://www.linkedin.com/in/suhail-ahmed-9b4312b/)。
   - 另一名成员在完成课程后也分享了他们的 [LinkedIn 个人资料](https://www.linkedin.com/in/roshankv/)。
- **通过调整时间解决超时问题！**：一名用户报告称，通过将 `requests.get` 函数中的超时值增加到 **20 秒**，解决了超时问题。
   - 另一名用户确认此更改解决了他们的问题。
- **关于 Agents.json 和 Prompts.yaml 的思考**：一名课程参与者要求澄清在 Unit 1 的 smolagents 章节中 **agents.json** 和 **prompts.yaml** 文件之间的区别。
   - 该用户还寻求关于 *如何使用 Agent 的 tools 参数向工具列表添加新工具* 的指导。
- **Llama-3 访问请求被拒绝！？**：一名用户报告称他们访问 **meta-llama/Llama-3.2-3B-Instruct** 的请求被拒绝，并询问原因。
   - 其他成员建议通常需要 Llama 的访问权限，并引导该用户在 [此处](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) 请求访问。
- **“暂时不可用”的困扰**：一名用户报告称，他们尝试使用的所有模型都显示为 *暂时不可用 (temporarily unavailable)*。
   - 另一名用户建议使用 **Apple 的 MLX 框架** 在本地设置笔记本，作为一种可能的变通方案。


  

---

### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1366809269244264469)** (1 条消息): 

> `Audio Overviews, Multilingual Support` 


- **Audio Overviews 正式上线！**: Audio Overviews 正在进行 Beta 测试，支持用户以 **50 多种语言**进行创建。
   - 立即尝试使用您偏好的语言，并通过这篇 [博客文章](https://blog.google/technology/google-labs/notebooklm-audio-overviews-50-languages/) 分享反馈。
- **多语言能力现已可用**: Audio Overviews 现在支持 **50 多种语言**，为更多样化的用户提供访问权限！
   - 请查看 [博客文章](https://blog.google/technology/google-labs/notebooklm-audio-overviews-50-languages/) 了解更多详情。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1366658539661299712)** (28 条消息🔥): 

> `NotebookLM language support, Audio Overview limitations, Concise explanations, Smarter Models` 


- **NotebookLM 的全球“绕口令”：现在会说多种语言**: NotebookLM 现在可以指定对话语言，这是一项新功能，且 [Google 的 NotebookLM 现在支持 50 种语言](https://www.forbes.com/sites/rogerdooley/2025/04/29/googles-notebooklm-now-speaks-50-languages-enabling-global-content/)。
   - 用户测试了 **冰岛语** 和 **马拉地语** 的 Audio Overviews，其中一位用户对马拉地语的流畅和地道感到印象深刻，称其 *“没有那种外国人口音之类的”*。 
- **Audio Overview 自定义上限引发讨论**: 一位用户注意到自定义音频更新限制在 **500 个字符** 内，并想知道这与将指令作为单独的文本文件上传是否有区别。
   - 该用户希望 *“减少愚蠢的闲聊，专注于事实和时间线”*。
- **用户发现非英语语言的 Audio Overviews 更简洁**: 用户发现为非英语语言生成的 **Audio Overviews** 持续时间更短。
   - 一位在小型文档上进行测试的用户表示，*“它的解释非常简洁”*。
- **更智能的模型助力更好的解释**: Google 证实新的非英语 **Audio Overviews** 表现更好，因为 *“我们在底层使用了更智能的模型！”*
   - NotebookLM 持续在底层改进其摘要能力。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1366674918128877682)** (65 条消息🔥🔥): 

> `NotebookLM Updates, Multi-Language Support, Audio Overview Issues, Interactive Mode Bugs, Podcast Feature Requests` 


- **NotebookLM 荣获 Webby 奖！**: **NotebookLM** 在 [Webby Awards](https://winners.webbyawards.com/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement/331142/notebooklm) 中表现出色，荣获 **技术成就奖 (Technical Achievement)**。
- **多语言支持上线，但并非所有人都能用！**: 成员们庆祝 **NotebookLM** **多语言支持** 的到来，但一位成员注意到 **越南语音频** 无法工作，且 UI 仍显示 *“仅限英语”*。
   - 一位成员确认功能仍在逐步推出中，并建议用户等待几个小时，另一位成员补充道，让大家 *“准备好应对每天十次‘即使功能发布了我也看不到该怎么办’的问题”*。
- **非英语 Audio Overviews 受到时长限制！**: 一位用户报告称，**英语 Audio Overview** 有 **15 分钟限制**，而 **土耳其语** 的限制为 **6 分钟 20 秒**。
   - 一位成员表示，由于 *“技术原因”*，非英语音频目前受到限制，但团队正在努力延长时长。
- **Interactive Mode 麦克风问题困扰用户！**: 一位用户报告称 **Interactive Mode** 无法从其麦克风采集任何音频。
   - 另一位成员建议检查 **麦克风权限** 和 **浏览器设置**，并尝试使用 [麦克风测试](https://mictests.com/) 或更换浏览器。
- **Notebook 共享困扰及解决方案！**: 一位成员报告称，他们共享 **Notebook** 的对象收到了 *“没有访问权限”* 的消息。
   - 一位成员澄清说，用户需要在共享对话框中明确添加共享对象的电子邮件地址。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1366635231347736626)** (75 messages🔥🔥): 

> `Add on Credits, Manus Fellow Program, Manus Referral Program, Manus Credit System, Beta Testing` 


- **不续订则 Add on Credits 毫无用处**：一位用户警告说，给予早期订阅者的附加积分（add on credits）如果不续订就毫无用处，因为它们会在短时间内过期。
   - 该用户声称他们未被告知过期事宜，现在已经损失了 **3900** 积分。
- **关于双倍积分的疑问解答**：一位用户提供了关于双倍积分的快速 FAQ，指出只要你的订阅处于激活状态，奖励积分就永远不会过期。
   - 他们补充说，邀请是随机的，似乎**并非每个邀请都能获得两个邀请额度**，这可能是因为官方限制了发放速度。
- **用户寻求 Manus Fellow Program 的信息**：一位用户询问了关于 Manus Fellow Program 的信息，例如 Manus 是否会主动联系所需的 fellow 并录用他们？此外还询问了目标国家（美国、中国、新加坡、韩国、澳大利亚等），以及该计划是否不针对巴基斯坦、印度等国家。
   - 另一位用户回复称，Starter 计划提供 **2 个邀请名额**，Pro 计划提供 **5 个邀请名额**。
- **对积分系统和 Beta 测试的批评**：一位用户表达了对积分系统和 Beta 测试的看法，认为用积分限制用户破坏了 Beta 阶段的初衷。
   - 他们补充道，*真正的 Beta 测试应该让用户能够从头到尾完成完整的项目，从而提供关于体验的有意义反馈并提出改进建议*。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1366626785827094620)** (51 messages🔥): 

> `X-Ware Red, Llama Prompt Ops, LLM Benchmarks Survey` 


- ****X-Ware Red** 工具发布**：一位用户分享了一个名为 **X-Ware Red** 的工具，它利用 embed 的标题，通过在前面添加 `r.jina.ai/` 和 `openrouter-free-tier` 来为线程生成标题。
   - 有用户建议增加一个开关，以便选择线程标题是否应与 embed 的名称不同。
- ****Llama Prompt Ops** 推出**：**Meta** 推出了 [Llama Prompt Ops](https://github.com/meta-llama/llama-prompt-ops)（一个用于 Prompt Engineering 的开源工具）以及 [Synthetic Data Kit](https://github.com/meta-llama/synthetic-data-kit)。
- **发现链接帖子重命名线程的 Bug**：一位用户报告了一个 Bug，即在线程中发布链接会重命名已经命名的线程，尽管它*应该只寻找标题中含有 "https://" 的线程并进行更改*。
- **用户寻求持久的 **LLM Benchmarks****：一位用户询问是否有关于 **LLM benchmarks** 的优秀综述，能够支持模型的历史对比。
   - 另一位用户回答说，*大多数 benchmark 的寿命不到 2 年*，并建议参考 "AI Engineer Reading List" 获取当前的基准，并指向了一位用户关于 OSS leaderboard v1 和 v2 的帖子。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1366719239968526416)** (13 messages🔥): 

> `Bending Origins in Mojo, Origin-related headaches, Multiple Licenses in Modular Repository, Pointer usage to avoid origin issues` 


- **在 Mojo 中灵活运用 Origins**：一位成员想进行一个小练习，涉及灵活运用 **Origins** 以实现特定目标，例如将 origins 重新绑定到容器元素的 **Origin** 而不是容器本身的 Origin。
   - 另一位成员回应说，他们处理过很多与 origin 相关的头疼问题，主要是由于 *API 缺失、可参数化 traits 以及其他缺失的语言特性*造成的。
- **Origins 导致可变引用问题**：一位成员提到，*你不能对同一个 origin 持有两个可变引用*，尽管可以通过将 origin 转换为 **MutableAnyOrigin** 来绕过这一点。
   - 另一位成员回应称，任何非数组或列表形状的数据结构都会遇到问题，这会导致性能退化到 **C 语言级别的性能**。
- **为了指针操作而绕过 Origins**：在讨论构建类似列表的类型 + 类似 span 的类型，或阅读标准库中的 `sort` 实现代码时，一位成员指出 *其中大部分操作都是抛弃 origins，进入指针时间（pointer time）*。
   - 另一位成员对指针类型（包括 unsafe）表示担忧，因为存在所有的可变-不可变（mut-immut）修复。
- **Modular 仓库包含多个许可证**：由于某些部分采用 Modular 的 **Community License** 授权，而其他部分采用 **Apache 2** 授权，**Modular 仓库**现在似乎需要包含多个许可证。
   - 具体而言，[`src/max`](https://github.com/modular/max/blob/main/src/max/serve/README.md) 中的一些内容使用了社区许可证。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1366728808526381056)** (11 messages🔥): 

> `importing Python packages, profiling blocks of code, SIMD width, vector strip-mining, flamegraph` 


- **标准 Python `import` 支持可能即将到来**：虽然 Mojo 尚未确认完全支持标准 Python `import` 语句，但据一位成员称，这是一个“相当确定的可能”，暗示 `python.import_module` 可能不会永远是唯一的选择。
- **`llvm-mca` 浮出水面，分析特定代码块性能**：有成员询问如何分析特定代码块的性能，并提到了 `gpu` 模块的一个私有部分 ([链接](https://github.com/modular/max/blob/main/mojo/stdlib/src/gpu/profiler.mojo))，另一位成员建议使用 `llvm-mca`。
- **针对 SIMD 宽度的向量条带挖掘 (Vector Strip-Mining)**：当指定的 **SIMD width** 是硬件 SIMD 宽度的倍数时，建议将编译器处理该情况的方式命名为 *vector strip-mining*。
- **`Flamegraph` 辅助 Perf 输出可视化**：一位成员建议使用 [flamegraph](https://github.com/brendangregg/FlameGraph) 来可视化 `perf` 输出，并指出可执行文件应在编译时包含 **debug info**。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1366854505320419379)** (2 messages): 

> `GPT-4o generates Tetris, PapersChat indexes papers` 


- **GPT-4o 一次性生成俄罗斯方块**：来自 KaranVaidya6 的视频展示了 **GPT-4o** 如何结合 **LlamaIndex** 和 **Composiohq** 一次性生成 **Tetris**。
   - 视频中使用的代码已在 [GitHub](https://t.co/KJb7YRINWg) 上发布。
- **PapersChat 索引 ArXiv 和 PubMed 上的论文**：**PapersChat** 是一款 Agentic AI 应用，允许你与论文进行对话，并收集来自 **ArXiv** 和 **PubMed** 的论文信息，由 **LlamaIndex**、**Qdrant** 和 **MistralAI** 提供支持。
   - 它会索引你所有的论文并提供一个精美的 Web UI 进行查询，访问地址在 [这里](https://t.co/lYwXh27F9x)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1366667182687125557)** (17 messages🔥): 

> `Azure OpenAI timeouts, MessageRole.FUNCTION vs MessageRole.TOOL, Function agent and context issues` 


- **Azure OpenAI 的间歇性超时困扰用户**：用户报告 **Azure OpenAI** 端点出现间歇性 **timeouts**，即使在相同的 Prompt、端点和网络条件下也是如此，这可能暗示了潜在的 **rate limits**、**firewall issues** 或 **context breaching**。
   - 一位用户指出，由于问题会持续数分钟，重试机制也无济于事，且更换网络有时只能解决部分不一致性。
- **剖析 MessageRole：FUNCTION vs. TOOL**：**MessageRole.FUNCTION** 和 **MessageRole.TOOL** 之间的区别取决于所使用的具体 API。
   - 某些 API（如 **OpenAI**）使用 **tool messages**，而其他 API 则依赖 **function messages**。
- **Function Agent 上下文混乱问题揭秘**：一位用户遇到了 **function agent** 在第二轮交互的流事件（stream event）处卡住的问题，并提供了示例代码。
   - 一位成员建议在 `stream_events()` 退出后使用 `await handler`，以确保前一次运行结束并接收到最终响应，这解决了该错误。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1366639037745467433)** (9 messages🔥): 

> `RAG Chatbot 挑战，针对多源数据的 GraphRAG，本地推理与小模型训练，AI 研究协作` 


- **RAG 聊天机器人面临挑战**：一位成员正在开发基于官方文档的 **RAG-based chatbot**，在使用 **vector search + BM25** 时，遇到了需要从多个来源和文档中提取 chunk 来回答问题的挑战。
   - 他们正在寻求关于如何为 **LLM Claude 3.5 Sonnet v1** 和 **Amazon Titan v1** embeddings 最好地将引用链接到文档附录的建议。
- **探索适用于多源数据的 GraphRAG**：一位成员询问 **GraphRAG** 是否值得尝试用于汇总来自多个来源的答案，并将其与需要特定领域预训练模型的 **insightRAG** 进行了比较。
   - 他们还询问了替代方案，并提到将参加 **NAACL**。
- **探索关于本地推理和小模型训练的新项目**：一位曾是 [Dataherald](https://github.com/Dataherald/dataherald) 联合创始人的成员正在探索一个围绕 **local inference** 和 **small model training** 的新项目。
   - 他表达了对合作以及参与社区研究的兴趣。
- **机器人、自主系统与 AI：工作机会正在酝酿**：一位从事 **Robotics, Autonomy, and AI** 工作的成员正专注于 **LLMs** 在加速软件工程方面的作用。
   - 他们询问是否可以在 Discord 中发布工作机会，以及这是否会被视为“广告”。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1366808622927056927)** (10 messages🔥): 

> `递归符号提示，LLM 诚实合规性，LLM 中的 HHH 目标` 


- **探索递归符号提示**：一位成员正在探索 **recursive symbolic prompts** 在分类器压力下的表现，重点关注平滑或对齐约束如何影响 **multi-turn hallucination drift**（多轮幻觉漂移）。
   - 该成员特别感兴趣的是符号结构（如 **role-bound predicates** 或 **attention-synced markers**）如何在多轮输出中存续，以及尽管存在软对齐漂移或输出平滑，这种结构如何跨补全（completions）传递。
- **LLMs HHH 张力暴露**：一位成员分享了关于[定量评分 LLM 输出在比较 HHH（Helpful, Honest, Harmless）对齐目标时表现](https://www.notion.so/TPIP-Exposing-Alignment-Tension-in-Modern-LLMs-1d5927516e1b8080b8c3d625a40a131d)的研究。
   - 他们结合使用 **YAML** 和 **python/Gradio** 来审计用户会话，测量每个 **HHH** 变量之间的内部张力，这包括强制模型变得更加诚实并观察由此产生的张力。
- **前沿模型在诚实度方面挣扎**：同一位成员发现，某些前沿模型比其他模型更符合诚实性要求，而一些模型在提供大量 token 堆砌且模棱两可的回答时，还会输出伪造的指标。
   - 他们指出，像 **ChatGPT 4o** 和 **4.5** 这样的模型在回答挑衅性查询时表现出很高的置信度，但实际上，它们在会话中充斥着模棱两可的双关语；讽刺的是，**OpenAI** 是所有前沿模型中最不透明的。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1366627484828958752)** (12 messages🔥): 

> `Credential Passing, RAG type server for client file ingestion, Streamable HTTP Implementation and Authentication, Multi-Tenant Server Hosting, Open Source Models for Agentic Applications` 


- **凭证难题：寻求 Header 帮助**：一位成员在使用 Python 从客户端向 MCP server 通过 headers 传递凭证时遇到困难，正在寻求帮助。
   - 在给定的上下文中未提供解决方案或建议。
- **RAG 服务器文件摄取**：一位成员正考虑构建一个 **RAG 类型服务器**，客户端可以通过端点摄取文件，将其保存在服务器上，并用于回答问题。
   - 他们在询问这是否是一个好的方法，或者是否有更好的替代方案。
- **Streamable HTTP 的实现：等待身份验证评估**：一位成员询问社区对最近发布的 **TS SDK** 中当前的 **Streamable HTTP 实现和身份验证** 的看法。
   - 另一位成员回答说它运行良好，但他们仍在摸索托管多租户服务器的细微差别以及有状态性（statefulness）如何影响它。
- **多租户服务器托管**：关于托管**多租户服务器**以及有状态性如何影响它的担忧。
   - 似乎无状态服务器应该为每个请求生成一个新的 MCP server 实例，但不清楚为什么 1 个实例对有状态服务器足够，而对无状态服务器却不够。
- **将 Agentic 开源模型投入生产：可行还是幻想？**：一位成员询问人们是否真的在生产环境中使用开源模型进行 Agentic 应用（而不仅仅是个人项目）。
   - 他们发现大多数开源模型在没有微调的情况下很难进行推理或遵循指令。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1366882155824807976)** (1 messages): 

> `MCP Server, Real Time Push Notifications` 


- **MCP Server 在 Agent 工作流完成时发送通知**：一位成员宣传使用 [mcp-gotify](https://github.com/SecretiveShell/mcp-gotify)（一个用于与 [gotify/server](https://github.com/gotify/server) 交互的 **MCP server**），以便在长时间运行的多 Agent 工作流完成时，在桌面和移动端接收实时推送通知。
- **Gotify server 替代方案？**：用户现在正使用 [gotify/server](https://github.com/gotify/server) 作为向桌面和移动端推送通知的替代方案。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1366719391240163328)** (9 messages🔥): 

> `foreach optimization, gradient scaling, DoRA + QAT` 


- **通过 Foreach 实现快速梯度缩放**：一位成员分享了一个使用 `torch._foreach_mul_` 进行梯度缩放的[代码片段](https://link.to/snippet)，可能将其与梯度裁剪合并为一个单一的参数循环。
   - 另一位成员指出了[相关的 PR](https://github.com/pytorch/torchtune/pull/2624)，并想知道这种看似恒定的增益是否会在多次迭代中累积。
- **Tune 贡献者寻找易于入手的 Issue**：一位成员强调了[两个简单 Issue](https://github.com/pytorch/torchtune/issues/2648)和[另一个 Issue](https://github.com/pytorch/torchtune/issues/2649)，供社区为项目做出贡献。
   - 未提供关于这些 Issue 性质的进一步信息。
- **DoRA 和 QAT：一个未被探索的前沿？**：一位成员询问了将 **DoRA (Difference of Low-Rank Adaptation)** 与 **QAT (Quantization-Aware Training)** 结合使用的经验。
   - 在提供的消息中，没有关于这种组合的讨论或回应。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1366696277844164642)** (6 messages): 

> `MCP Usage, Displaying thoughts component in React` 


- **渴望 MCP 用法文档**：一位用户询问了有关最新版本中新增的 **MCP (Multi-Controller Processing)** 用法的教程或文档。
   - 另一位用户提到他们通过查看测试用例开始了学习，虽然有教程会更好，但并不紧迫，并澄清理解 **stdio** 和 **SSE 客户端** 的设置是关键。
- **React 中的 Thoughts 组件 - 最佳实践**：一位成员正在寻求关于在 **React 中显示 Thoughts 组件**最佳方式的建议。
   - 他们知道可以修改 forward 方法，但正在询问是否有更好或更合适的地方来实现这一点。


  

---

### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1366789810827825152)** (1 messages): 

> `Markdown-based vs Image-based multimodal RAG on PDFs, Docling, EmbedV4` 


- **Markdown 与图像 RAG 之争升温**：一名成员询问了在 **PDFs** 上比较**基于 Markdown** 与**基于图像的多模态 RAG** 的方案。
   - 他们目前正在使用 **Docling** 将 PDFs 转换为 Markdown，然后计算文本 Embedding，但正在考虑切换到 **EmbedV4** 以输入原始图像并获取用于 RAG 的多模态 Embedding。
- **PDF 转换技术探讨**：该成员在计算文本 Embedding 之前，使用 **Docling** 将 PDFs 转换为 Markdown。
   - 他们正在评估 **EmbedV4** 作为替代方案，以便在 RAG 中直接处理原始图像以生成多模态 Embedding。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1366841428076003541)** (2 messages): 

> `Cohere rate limits for embed-v4, Embed V4 on Bedrock` 


- **Cohere 考虑提高速率限制**：一位用户询问 **Cohere** 是否会提高 `embed-v4` 的生产环境速率限制（Rate Limits）。
   - 他们表示，对于他们的 **PDFs** 使用场景，**每分钟 400 次请求**是不够的。
- **Cohere 思考 Bedrock 的可用性**：一位用户询问 **Embed V4** 是否会在 **Bedrock** 上可用。
   - 目前 Cohere 尚未给出答复。


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1366782678187114607)** (2 messages): 

> `Cohere's Embed V4 model, Data Scientists introductions` 


- **爱好者加入，渴望体验 Embed V4！**：一位新的数据科学家加入了 Cohere Discord 社区，表达了对尝试新工具的浓厚兴趣，特别是 Cohere 最新的 **Embed V4 模型**，并探索其潜在应用。
   - 新成员表示*很高兴加入社区*。
- **社区欢迎新数据科学家**：Cohere 社区 Discord 服务器对新成员的加入表示欢迎。
   - 欢迎消息鼓励新成员提供其**公司/行业/大学**、*正在研究的具体内容*、喜爱的技术/工具，以及*希望从这个社区获得什么*。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1366781809614131220)** (5 messages): 

> `Embeddings, GPT4All, Manus AI, Embedding grouping` 


- **Manus AI 工具发布**：一名成员分享了 [Manus AI](https://manus.im/edu/invitation/FBBGGFBFKTUE) 的链接，声称*中国发布了它*，现在所有人都可以使用。
   - 该成员暗示这是*第一个自动研究 AI Agent*，并且*我们正被这个东西彻底取代*。
- **Embeddings 可以使用 Nomic 工具**：一名成员建议 Nomic 提供了所有必要的 Embedding 工具，并且它*超越了 GPT4All*。
   - 他们声称 Nomic 的 Embedding 工具*可以在各种其他软件中工作*。
- **Embedding 分组可以替代训练**：一名成员描述了 **Embedding 分组** 如何替代训练：为特定的人分组 Embedding 并取平均 Embedding，然后使用该 Embedding 对其他图片进行排序并找到同一个人。
   - 他问：*“你理解这个概念了吗？”*


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1366685162615865406)** (3 messages): 

> `Loose vs Strict Evaluation, Model Training Inconsistencies` 


- **宽松与严格的模型评估**：一名成员提出了一种想法，即为模型建立*“宽松”与“严格”的评估机制*，特别是针对那些可以通过*“黑客手段”*使其工作的模型，这代表了特定的使用场景。
   - 他们举了一个例子：一个模型被错误地训练为输出 `<tool_call>` 而不是其规范指明的 `<|tool_call|>`，在这种情况下，了解情况的用户可能会忽略该错误并评估其功能正确性。
- **模型训练导致不一致性**：一位成员遇到了一个模型，该模型被错误地训练为输出 `<tool_call>` 而不是其规范指明的 `<|tool_call|>`。
   - 该成员建议，如果他们专门了解该模型，可以忽略此错误并评估功能正确性，但普通用户则无法做到。