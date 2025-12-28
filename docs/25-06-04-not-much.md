---
companies:
- mistral
- cursor
- anthropic
- openai
- aie
- google-deepmind
- meta-ai-fair
date: '2025-06-04T05:44:39.731046Z'
description: '**Mistral** 发布了一个新的 **Code** 项目，**Cursor** 发布了 **1.0** 版本。**Anthropic**
  改进了 **Claude Code** 方案，而 **ChatGPT** 宣布扩大了连接功能。


  这一天的主角是 **AIE** 的主题演讲和相关专题，包括 **GraphRAG**、**RecSys** 和 **Tiny Teams**。在 Reddit
  上，**Google** 开源了 **DeepSearch** 技术栈，用于利用 **Gemini 2.5** 和 **LangGraph** 构建 AI 智能体，支持灵活的智能体架构，并能与
  **Gemma** 等本地大语言模型集成。


  **Meta** 的一篇新论文分析了语言模型的记忆机制，指出 GPT 风格的 Transformer 每个参数约存储 **3.5–4 比特**的信息，并探讨了从记忆到泛化的转变，这对**混合专家（MoE）**模型和量化效应具有重要意义。'
id: MjAyNS0w
models:
- gemini-2.5
- gemma
- claude-code
people: []
title: AI 工程师世界博览会演讲：第一天
topics:
- agent-based-architecture
- open-source
- model-memorization
- scaling-laws
- quantization
- mixture-of-experts
- language-model-memorization
- model-generalization
- langgraph
- model-architecture
---

**快乐的一天。**

> 2025年6月3日至6月4日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（218 个频道，6571 条消息）。预计为您节省阅读时间（以 200wpm 计算）：503 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

Mistral 发布了一个 [Code](https://mistral.ai/products/mistral-code) 项目，[Cursor 发布了 1.0 版本](https://www.cursor.com/en/changelog/1-0)，Anthropic [改进了 Claude Code 计划](https://youtu.be/Yf_1w00qIKc?si=wDtapcnvLfnq5ip4)，ChatGPT [宣布了更多连接](https://x.com/openai/status/1930319398897889707?s=46)。但就新闻周期而言，这一天理应属于 AIE，其主流频道包含了一系列[令人赞叹的 MCP 专题主题演讲](https://www.youtube.com/watch?v=U-fMsbY-kHY)，同时还直播了显著的 [GraphRAG](https://www.youtube.com/watch?v=RR5le0K4Wtw)、[RecSys](https://www.youtube.com/watch?v=3k4a0PemMu4) 和 [Tiny Teams](https://www.youtube.com/watch?v=xhKgTkzSmuQ) 专题。

---

# AI Twitter 回顾

pipeline 今天出故障了，抱歉

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

### 1. 最近的开源和研究发布 (Google DeepSearch, Meta 模型论文)

- [**Google 开源 DeepSearch 栈**](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) ([Score: 840, Comments: 77](https://www.reddit.com/r/LocalLLaMA/comments/1l27g8d/google_opensources_deepsearch_stack/)): **Google 开源了一个新的 DeepSearch 栈，可通过 [gemini-fullstack-langgraph-quickstart repo](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) 获取，该仓库作为一个模板，用于使用 Gemini 2.5 和 LangGraph 编排框架构建全栈 AI Agent。虽然作者确认这与实际的 Gemini 用户应用后端不同，但此次发布使开发者能够实验基于 Agent 的架构，可以与其他本地 LLM（如 Gemma）集成，并利用 Docker 和模块化项目脚手架进行快速原型设计。该技术栈设计灵活，但如果需要替代模型或搜索系统（除 Gemini 和 Google Search 之外），则需要进行替换。** 评论讨论强调，这次发布更多是一个结构良好的演示，而不是生产级后端（如 Gemini App 中所使用的），突出了 LangGraph 作为编排器的潜力，并引用了 [LangManus](https://github.com/Darwin-lfl/langmanus/tree/main) 作为一个更复杂的基于 LangGraph 的系统，用于高级 Agent 实现。
    - Google 开源的这个项目与 Gemini App 栈不同，旨在让开发者能够利用 LangGraph 构建基于 Gemini 的 Agent 系统。虽然理论上可以适配使用 Gemma 代替 Gemini 作为底层模型，但用户需要将搜索组件更换为替代工具以保持兼容性。
    - 虽然该演示展示了清晰的架构，但与更高级的 LangGraph 项目相比，它并不特别复杂或新颖。对于更复杂和深入的实现，评论者指向了 LangManus (https://github.com/Darwin-lfl/langmanus/tree/main) 作为一个例子，强调 DeepSearch 开源栈主要作为一个易于上手的端到端演示，而不是在突破技术边界。
- [**Meta 新论文 - 语言模型记忆了多少？**](https://arxiv.org/abs/2505.24832) ([Score: 176, Comments: 30](https://www.reddit.com/r/LocalLLaMA/comments/1l2gvar/new_meta_paper_how_much_do_language_models/)): **讨论的 Meta 论文 ([arXiv:2505.24832](https://arxiv.org/abs/2505.24832)) 提出了一种严谨的方法来估算语言模型的记忆能力，实证表明 GPT 风格的 Transformer 稳定存储约 3.5–4 bits/parameter（例如 bfloat16 为 3.51，float32 为 3.83），并且存储容量不会随着精度的提高而线性扩展。该研究描述了从记忆到泛化（"grokking"）的转变发生在模型容量饱和时，而当数据集信息内容超过存储限制时，双重下降（double descent）开始。他们进一步引入了从数百个训练好的 Transformer（500K–1.5B 参数）中推导出的缩放定律，将模型大小和数据集容量与成员推理攻击（membership inference attack）的成功联系起来，发现在数据集庞大且经过重删（deduped）时，负责提取的是泛化而非死记硬背。** 评论者对这些发现如何扩展到 Mixture-of-Expert (MoE) 模型，以及量化（低于 3.5 bits/parameter）或低精度/QAT 训练对记忆和泛化边界的影响表示关注。有人推测，低于 3.5 bit 的量化可以解释实践中观察到的性能下降，并好奇像 BitNet 这样的新型架构是否会改变这些基本的容量限制。
    - 作者实证估算 GPT 家族的 Transformer 每个参数可以存储 3.5 到 4 bits 的信息（例如 bfloat16 为 3.51 bits/parameter，float32 为 3.83），同时指出增加精度并不会线性增加存储容量，这意味着模型容量的使用超出了原始的逐位记忆。
    - 论文将模型的记忆和泛化与双重下降（double descent）联系起来：记忆在容量饱和前占主导地位，之后通过 "grokking" 产生泛化。据报道，当数据集信息（以 bits 计）超过模型存储时，会发生双重下降，迫使信息共享并增加泛化。
    - 后续讨论提出了关于这些发现是否扩展到 Mixture-of-Experts (MoE) 架构、量化感知训练 (QAT) 或更低精度如何影响存储/记忆的问题，并推测量化到 ~3.5 bits 以下的模型可能会从根本上降低 GPT 风格模型的性能，而对于 BitNet 等替代架构仍存在开放性问题。

### 2. LLM 与视觉多模态模型发布及基准测试

- [**nvidia/Nemotron-Research-Reasoning-Qwen-1.5B · Hugging Face**](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B) ([Score: 133, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1l2820t/nvidianemotronresearchreasoningqwen15b_hugging/)): **Nvidia 的 [Nemotron-Research-Reasoning-Qwen-1.5B](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B) 是一款拥有 1.5B 参数的开源权重模型，专注于复杂推理（数学、代码、STEM、逻辑），采用基于 Group Relative Policy Optimization (GRPO) 的新型 Prolonged Reinforcement Learning (ProRL) 方法训练。ProRL 引入了关键的 RL 稳定技术——熵崩溃缓解 (entropy collapse mitigation)、解耦剪裁与动态采样 (DAPO)、KL 正则化以及参考策略重置 (reference policy reset)——实现了超过 2k 步的 RL 训练和更广泛的泛化能力。该模型显著超越了 DeepSeek-R1-1.5B，并达到或超过了 DeepSeek-R1-7B 的水平，在** `数学 (math)`**、**`编程 (coding)`**、**`逻辑 (logic)`**、**`STEM`** 和 **`指令遵循 (instruction-following)`** 方面的平均 pass@1 分别提升了 **`14.7%`**、**`13.9%`**、**`54.8%`**、**`25.1%`** 和 **`18.1%`**。** 评论者强调了面向边缘和移动设备的轻量级、高效开源推理模型的发展趋势，并关注了 ProRL 的 RL 创新。批评意见主要集中在 Nvidia 限制性的 CC-BY-NC-4.0 许可证上，尽管技术成果显著，但该许可证限制了商业用途。
    - Nemotron-Research-Reasoning-Qwen-1.5B 模型利用了 ProRL (Prolonged Reinforcement Learning) 算法，该算法支持延长的 RL 训练（超过 2k 步）并结合了 Group Relative Policy Optimization (GRPO)。关键技术创新包括熵崩溃缓解、解耦剪裁与动态采样策略优化 (DAPO)、KL 正则化以及参考策略重置。据称，这些方法在数学、代码、STEM 和逻辑谜题等多种推理任务中带来了显著的泛化提升。
    - 上传者分享的技术基准测试表明，这款 1.5B 参数模型声称比 DeepSeek-R1-1.5B 有实质性改进，报告的 pass@1 增益分别为：`14.7%` (数学)、`13.9%` (编程)、`54.8%` (逻辑谜题)、`25.1%` (STEM) 和 `18.1%` (指令遵循)。有趣的是，据称它在多种任务上的表现可以媲美甚至超越 DeepSeek-R1-7B，这对于 1.5B 参数规模的模型来说是不寻常的。
    - 该模型已发布 GGUF 格式及量化选项（q4, q8, f16），以方便在资源受限的硬件上进行本地推理。然而，技术讨论引发了担忧，即限制性的 CC 非商业许可证和模糊的许可条款可能会显著阻碍其商业化或更广泛的现实应用，尽管其技术优势明显。
- [**Vision Language Models are Biased**](https://vlmsarebiased.github.io/) ([Score: 100, Comments: 54](https://www.reddit.com/r/LocalLLaMA/comments/1l2b83p/vision_language_models_are_biased/)): **最先进的视觉语言模型 (VLM) 在常规视觉任务（例如，数典型动物的腿或标准标志上的条纹）上几乎可以达到完美的准确率，但根据 VLMBias 基准测试，在反事实或改变后的场景中，其准确率会骤降至约 17%。详细分析显示，模型过度依赖记忆的先验知识 (priors) 而非实际的视觉输入，75.7% 的错误反映了刻板知识而非歧义，且显式的偏见缓解提示词基本无效。[原始来源](https://vlmsarebiased.github.io/) 提供了涵盖七个领域的数据集和方法论，揭示了 VLM 在训练分布之外进行视觉推理的能力缺失。** 评论者争论这些发现是否真的令人惊讶，因为所有的 AI 系统都会在数据和架构中反映出偏见，并指出在 LLM 的对数概率 (log probabilities) 中也观察到了类似的问题。
    - 顶尖的视觉语言模型在涉及熟悉主题的计数任务中（如 Adidas 标志上的 3 条杠或 4 条腿的狗）可以达到近乎完美的准确率（高达 100%），但在遇到反事实或分布外 (out-of-distribution) 图像（如 4 条杠的 Adidas 标志或 5 条腿的狗）时，准确率会大幅下降至 17% 左右，突显了泛化能力的严重局限。
    - 这种失败模式类似于视觉模型在面对手指数量多于或少于标准五指的手部图像时经常数错的情况，进一步证明了最先进的模型对分布偏移高度敏感，并且在不熟悉的场景中难以进行组合推理 (compositional reasoning)。

## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. AI 模型与功能发布 (VEO 3, Sora, Chroma, Codex, ChatGPT Memory/Research)

- [**巴西乌利亚诺波利斯市政府使用 VEO 3 制作了一个完整的商业广告，仅花费了 300 雷亚尔（约 52 美元）的 VEO 3 额度**](https://v.redd.it/36cgd4rvjp4f1) ([Score: 1047, Comments: 196](https://www.reddit.com/r/singularity/comments/1l2azl6/ulianopolis_city_hall_in_brazil_made_a_complete/))：**巴西乌利亚诺波利斯市政府完全使用 Google 的 Veo 3 生成式视频 AI 制作了一个 1 分钟的专业级商业广告，仅产生了 300 雷亚尔（约 52 美元）的 AI 额度费用——与传统的本地制作成本（>100,000 雷亚尔 / 约 17,500 美元）相比，成本极度降低。该工作流几乎取代了所有传统的制作职能——导演、编剧、拍摄、剪辑、后期处理等——完全依赖文本生成视频（text-to-video）的生成能力。参见 [Reddit 原帖](https://v.redd.it/36cgd4rvjp4f1) 和创作者的 [Instagram](https://www.instagram.com/renato_lferreira/)。** 评论者认为这是对传统商业广告制作的重大颠覆，暗示广告和创意机构正面临威胁，并对看到高质量、母语级的 AI 输出表示赞叹，强调了媒体制作工作流即将发生的转变。
    - 一个关键的技术点是使用 VEO 3 制作商业广告带来的巨大成本削减，以 300 雷亚尔（约 52 美元）制作出专业水准的广告，大幅削减了传统代理机构的成本，同时允许通过快速的 AI 重新生成和编辑进行迭代改进。
    - VEO 3 的母语合成能力被认为特别令人印象深刻。用户注意到其准确的巴西葡萄牙语输出，包括*地道的口音和自然的语言表达*，这些在传统上对 AI 生成模型来说一直具有挑战性，使得结果对于本地受众来说更加稳健且具备市场就绪性。
- [**Microsoft 将免费的 Sora AI 视频生成功能引入 Bing**](https://www.windowscentral.com/microsoft/microsoft-bing-video-creator-sora-ai-generator-free-announcement) ([Score: 245, Comments: 51](https://www.reddit.com/r/singularity/comments/1l264o6/microsoft_brings_free_sora_ai_video_generation_to/))：**Microsoft 已将 OpenAI 的 Sora AI 视频生成模型集成到 Bing 应用中，品牌名为“Bing Video Creator”，提供对生成式视频内容的免费访问。该方案目前还没有专门的 Sora 应用或 ChatGPT 集成，初始用户体验注意到其既能生成详细的动画内容，也会遇到严格的安全/请求拦截，反映了严密的内容审核。** 用户在讨论当前实现的实用性与限制性：虽然认可了新颖的创意可能性，但一些人批评其过度激进的安全过滤器，限制了实用性或实验性的用例。
    - 几位用户将 Microsoft 的 Sora（通过 Bing Video Creator 提供）与 Google 的 Veo3 进行了比较，共识表明 Veo3 在视频生成方面提供了更优的结果。这意味着 Sora 目前在视频质量和模型能力方面落后于 Veo3，使其在该领域成为较弱的竞争对手。
    - 一位评论者指出的技术限制是 Sora 激进的安全过滤器，导致许多请求被拦截，与限制较少的替代方案相比，降低了其内容生成的可用性和灵活性。
    - 有人提到 Sora 的集成非常有限，因为它目前仅通过 Bing 应用提供，而不是作为独立应用程序或在 ChatGPT 应用内提供，这可能会阻碍开发者和高级用户的广泛采用和效用。
- [**OpenAI 正准备发布 2 个具有原生音频支持的新模型**](https://x.com/testingcatalog/status/1929949017472930181?s=46) ([Score: 229, Comments: 31](https://www.reddit.com/r/singularity/comments/1l2htv5/openai_is_preparing_to_release_2_new_models_with/))：**据报道，OpenAI 将发布两个基于 GPT-4o 的模型——“gpt-4o-audio-preview-2025-06-03”和“gpt-4o-realtime-preview-2025-06-03”——其特点是原生音频处理，而不是依赖外部的语音转文本或文本转语音模块。这表明 GPT-4o 架构内集成了端到端的音频 I/O 能力，可能实现低延迟的音频交互和更无缝的类助手功能（参见 [TestingCatalog News](https://x.com/testingcatalog/status/1929949017472930181?s=46) 的早期报道）。** 评论者质疑“原生音频”与之前的 GPT-4o 实现有何区别，并指出 GPT-4o 在演示中已经展示了实时音频；目前对于这次发布是带来了功能性进步还是仅仅将现有的预览功能正式化存在争论。

- 几位用户正在寻求关于 “native audio” 内涵的澄清，质疑它是否指的是像 GPT-4o 这样已经具备音频支持的模型。技术上存在不确定性，即即将推出的模型是为直接音频处理提供了根本性的新架构，还是仅仅通过一种新颖的 API 或格式公开了现有的功能。
- 一位评论者推测，新发布的内容可能与一年多前展示的 GPT-4o 音频助手功能有关，暗示新模型可能会在 API 生态系统中正式化或增强这些实时语音交互能力。
- 有一种技术主张认为，“native audio” 的范围可能会从音频扩展到作为连续 bitstream 的视频处理，这表明其可能向统一的多模态 bitstream 处理演进，以实现更自然的输入/输出模态。
- [**今年夏天值得期待的一切**](https://i.redd.it/ou0k2gkx2s4f1.jpeg) ([Score: 216, Comments: 59](https://www.reddit.com/r/singularity/comments/1l2nmsr/everything_to_look_forward_to_this_summer/)): **该图片是一个时间轴风格的信息图，列出了计划于 2024 年夏季（6 月至 8 月）发布的主要预期 AI 模型和技术项目（如 GPT-5），最近出现在 Peter Diamandis 的 YouTube 内容中。该图表归功于 @chatgpt21，汇总了各种即将推出的发布活动，展示了当前 AI 领域重大公告的加速节奏和密集程度。** 热门评论对 GPT-5 据传即将发布却缺乏热度表示怀疑，并指出技术迭代周期已变得如此之快，以至于此类时间表很快就会过时。
    - 评论强调了 GPT 模型的加速发布节奏，一些用户注意到 GPT-4 与传闻中的 GPT-5 之间的时间间隔比之前的周期短得多，因此质疑预测性发布图表的价值和准确性。
    - 一位评论者质疑 GPT-5 的预期发布日期是否有官方公告支持，还是仅仅是推测，这反映了社区对于即将到来的模型泄露和路线图可靠性的持续不确定性。
    - 有一种观点认为，与对 GPT-5 的预期相比，GPT-4 已变得明显能力下降或变“笨”了，这表明最终用户注意到或相信当前 LLM 与尚未发布的 LLM 之间存在巨大的质量差距。
- [**Memory 功能现已向免费用户开放！！！**](https://i.redd.it/jy18jpn0nq4f1.png) ([Score: 235, Comments: 57](https://www.reddit.com/r/OpenAI/comments/1l2g8es/memory_is_now_available_to_free_users/)): **该图片是一个 FAQ 更新，宣布 ChatGPT 的 Memory 功能自 2025 年 6 月 3 日起开始向免费用户推出。这允许 ChatGPT 参考用户最近的对话以提供更相关的回复。在某些欧洲地区，用户必须手动启用此功能，而在其他地方则是默认激活；用户保留随时禁用 Memory 功能的控制权。** 评论中的技术讨论集中在隐私和可用性上：付费用户指出，订阅允许他们选择不将数据用于模型训练，并质疑 OpenAI 的合规性。其他人则批评了 Memory 功能，指出自动保存可能导致保留无关或过时的数据，并表达了对更细粒度、手动 Memory 控制的需求。
    - 几位评论者讨论了 ChatGPT 的 “Memory” 功能如何通过将相关的 memory 片段附加到你的 prompt 中，将聊天历史的各个方面用作内部知识库，这既会影响回复的准确性，也会根据你之前的对话引入偏见。技术用户注意到，随着时间的推移，这可能会降低真实性或注入过时的/特定上下文的假设。
    - 提出的一个关键点是用户对 memory 的控制有限：当前的实现会自动保存信息，有时会存储无关或过时的数据。用户表达了对手动 memory 管理的需求，即用户可以明确添加或策划模型应该记住的内容，从而可能提高准确性和相关性。
    - 对于 memory 功能是否比之前的机制有显著改进，人们提出了质疑。一些用户观察到模型仍然倾向于“自信地”编造过去对话的细节，而不是可靠地回忆具体细节，这表明 memory 集成或保留逻辑在精确的长期引用方面可能尚不稳健。

- [**Codex 正在向 Plus 用户推出**](https://www.reddit.com/r/OpenAI/comments/1l2kd42/codex_rolling_out_to_plus_users/) ([Score: 107, Comments: 31](https://www.reddit.com/r/OpenAI/comments/1l2kd42/codex_rolling_out_to_plus_users/)): **Codex 现在正逐步为 ChatGPT Plus 用户启用，用户报告确认可以通过 URL https://chatgpt.com/codex 进行访问。Codex 是 OpenAI 以代码为核心的模型系列，针对自然语言转代码和代码生成任务进行了优化。原始帖子和评论并未指明针对 Plus 用户更新的使用限制或技术限制。** 评论者正在询问技术约束（如限制）以及 Codex 在 Plus 层级中的具体用例或功能；目前尚未提供明确答案。
    - 一位用户询问了 Codex 向 Plus 用户推出时的使用限制，指出有关 API 调用限制、速率限制或功能限制的细节尚未公布或尚不明确。对于希望通过 Codex 集成或自动化工作流的开发人员或技术用户来说，这是一个重点，因为了解这些限制对于其实现的可扩展性和可靠性至关重要。
    - 一条评论表达了对 Codex 级别能力将与 GPT-5 发布挂钩的预期，推测重大新功能或更广泛的工具集集成可能会保留给未来的模型迭代。这间接指向了对 OpenAI 模型生态系统演进的技术预期，表明代码生成或 API 能力的进一步提升可能与重大的架构更新保持一致。
    - 另一位用户询问 Codex 的用途，这暗示一些技术用户对于 Codex 的应用（主要是代码生成、API 使用以及可能与 GitHub Copilot 等产品或其他自动化工具的集成）仍存在困惑。这凸显了在技术社区中对 Codex 的目的和用例进行更清晰沟通的需求。
- [**研究功能现已在 Pro 计划中推出！！**](https://i.redd.it/b1x3zdboxq4f1.png) ([Score: 135, Comments: 39](https://www.reddit.com/r/ClaudeAI/comments/1l2hsjw/research_is_now_available_on_pro_plans/)): **图像显示 Anthropic 已在其 Claude Pro 计划中引入了标记为“BETA”的“Research”功能，Claude 界面中的新图标展示了这一点。该功能似乎提供了集成的研究辅助，用户可以输入查询并获得见解或综合信息，而非直接答案。界面更新表明，付费用户现在可以使用更先进、专注于研究的 AI 辅助。** 一位用户注意到该研究工具提供了深思熟虑、详细的指导而不仅仅是答案，通过可操作的见解改进了他们的工作。另一位评论者询问该功能与其它 AI 公司的类似产品相比如何，表明了潜在的基准测试兴趣。
    - 一位用户注意到，研究模式自动部署了 3-4 个子 Agent，使用深度优先方法从多个角度处理查询，这是一个旨在彻底性和探索性覆盖的技术实现细节。
    - 另一条评论指出，该工具在特定研究任务中引用了“300 多个来源且仍在增加”，并质疑这是否显著高于 OpenAI 的 GPT 和 Perplexity 所提供的典型来源数量，暗示其在信息聚合方面具有更强的广度。
    - 对主要模型进行了技术对比：在研究质量方面，用户更青睐 Claude Max 和 SuperGrok；评论认为 Gemini 提供了大量信息但精炼不足，而 OpenAI 的回答感觉过于生硬（clinical），突显了主要 AI 服务在研究输出方法上的差异。
- [**Chroma v34 发布两个版本**](https://www.reddit.com/r/StableDiffusion/comments/1l2asij/chroma_v34_is_here_in_two_versions/) ([Score: 170, Comments: 64](https://www.reddit.com/r/StableDiffusion/comments/1l2asij/chroma_v34_is_here_in_two_versions/)): **Chroma v34 已发布两个版本，区别在于“-detailed release”版本比标准模型提供更高的图像分辨率（[Hugging Face 链接](https://huggingface.co/lodestones/Chroma/tree/main)）。社区评论强调了在细节和灵活性方面的持续改进，特别是在非审查和非摄影类艺术生成方面。在细节校准版本上使用 LoRA 适配器的早期测试显示出质量的增量提升。** 评论者认为 Chroma 正在迅速成为领先的基础模型，并且是 Flux 的有力替代方案，特别是在非摄影和可定制的艺术生成任务中。

- Chroma v34 有两个版本：普通版和细节校准版，后者专门针对高分辨率数据进行了训练。用户已成功生成高达 `2048x2048` 的原生分辨率图像，并报告在这些尺寸下效果“相当不错”。
- Chroma v34 的独特之处在于它是一个无审查模型，且没有摄影风格偏好，这使其在各种类型的艺术作品中表现出色，包括摄影和非摄影输出。这解决了目前许多 AI 模型过度拟合摄影数据集的局限性。
- 多处提到在 Chroma v34 中使用 LoRA (Low-Rank Adaptation) 技术，包括成功的应用和图像细节的提升。这表明它易于与社区工具集成，并且生态系统正在迅速成熟，类似于之前的 SD14 和新兴的替代方案如 Flux。

### 2. 对 AI 驱动的经济不平等和失业的担忧

- [**我们需要竭尽全力防止 AI 成为奢侈品**](https://www.reddit.com/r/singularity/comments/1l2j6u1/we_need_to_do_everything_in_our_power_to_prevent/) ([Score: 222, Comments: 94](https://www.reddit.com/r/singularity/comments/1l2j6u1/we_need_to_do_everything_in_our_power_to_prevent/)): **该帖子强调了 OpenAI、Anthropic 和 Google 等大型 AI 供应商将强大的 LLM 置于高额月度付费墙之后的趋势（OpenAI 为 200 美元/月，Anthropic 为 100 美元/月，Google 为 130 美元/月），而开源 LLM（如来自 DeepSeek、Qwen 的模型）虽然能力在提升，但对资源的要求也在增加——随着模型规模和推理成本的上升，普通用户可能无法负担自托管的费用。作者指出，硬件限制（高端 GPU）和竞争性开源实验室潜在的私有化风险，可能会扩大高端 AI 与普及型 AI 之间的能力差距，随着 AGI 的临近，存在严重的社会经济分层风险。** 顶级的技术评论辩论了必然性与政策干预：一些人认为高成本与尖端 AI 密不可分，只有将这些成本社会化（例如公共 AI 基础设施）才能维持访问权限；而另一些人则声称较低级别/旧模型仍然普遍可用，并强调 AI 的经济性质类似于公用事业（如电力）；还有人质疑排他性的说法，因为开源/免费和付费 AI 层级同时存在。
    - 多位评论者强调了最先进 AI 的*巨大运营和开发成本*，指出目前模型需要昂贵的计算基础设施和能源。主要实验室（OpenAI、Google 等）之间的竞争维持了高价，有报道称供应商有时甚至在亏本运营（例如 OpenAI 的 Pro 计划），并需要向上调整定价（Google 最近涨价 250 美元）。
    - 存在关于定价分层的讨论：虽然性能最高或最新的模型很昂贵，但较旧或能力较弱的模型版本通常以较低的价格甚至免费提供。这被比作传统的科技市场，早期获得优质产品需要支付更多费用，但随着技术的成熟和规模化，普及度会增加。
    - 将 AI “社会化”——将其作为社会规模管理的公共事业——被提出作为确保在高成本下公平获取的一种方式，但这种方法并不能降低底层支出。在重大技术突破（如廉价聚变能或全自动化生产）出现之前，这些成本被认为是难以解决的，并且可能使 AI 保持为一种相对昂贵的资源。
- [**Dario Amodei 担心由于 AI 导致的失业，普通人将失去经济杠杆，这将破坏民主并导致权力严重集中：“我们需要拉响警报。我们可以阻止它，但不能仅仅通过说‘一切都会好起来的’来解决。”**](https://v.redd.it/ba6dzs1grq4f1) ([Score: 1378, Comments: 364](https://www.reddit.com/r/singularity/comments/1l2gwo1/dario_amodei_worries_that_due_to_ai_job_losses/)): **Dario Amodei (Anthropic CEO) 对 AI 驱动的失业风险表示担忧，这可能会削弱工人的经济杠杆，潜在地破坏民主并导致危险的权力集中。他强调除了盲目乐观外，还需要采取积极的干预措施，并表示：*“我们可以阻止它，但不能仅仅通过说‘一切都会好起来的’来解决。”* [来源](https://www.anthropic.com/people/dario-amodei)。** 评论强调了对政治意愿或公众反应的怀疑，指出了 AI 职位取代的渐进性（“温水煮青蛙”），这降低了紧迫感，从而推迟了政策干预，直到影响变得不可避免。

- Quick-Albatross-9204 强调了 AI 对就业的逐渐替代，并引用了“温水煮青蛙”效应：由于失业是渐进式而非即时发生的，更广泛的社会和政策制定者可能无法察觉到潜在经济影响的紧迫性或规模，直到采取有效行动为时已晚。这强调了对实时劳动力替代监测和适应性政策框架的需求。
- [**前 OpenAI AGI Readiness 负责人：“到 2027 年，几乎所有可以在计算机上完成的具有经济价值的任务，都将由计算机更有效、更廉价地完成。”**](https://i.redd.it/l0cd9s4yar4f1.png) ([Score: 1026, Comments: 356](https://www.reddit.com/r/singularity/comments/1l2jun4/former_openai_head_of_agi_readiness_by_2027/)): **该图片是前 OpenAI AGI Readiness 负责人 Miles Brundage 发布的一条推文，声称到 2027 年，几乎所有可以在计算机上执行的具有经济价值的任务都将能由计算机更有效、更廉价地完成——尽管他补充了关于判断语境以及部署与能力之间差异的限制条件。这一观点代表了对 AI 能力进展的一个强有力的、有明确时间线的断言，特别是围绕白领/知识工作的自动化，前提是产出纯粹基于技术价值而非社会或人类归属价值来评估。Brundage 澄清说，他的声明是指能力上的可能性，并不一定意味着自动化将普及或在各地部署。** 评论者对组织准备程度和数据基础设施提出了质疑（认为大多数职场即使到 2027 年也很难将其数据进行程序化格式化）。其他人则反驳道，实际工作的复杂性与技术可行性之间存在差距，并对社会影响（UBI，自动化税）表示担忧，引用了潜在白领失业的规模。
    - Fenristor 认为，组织和数据基础设施的限制将显著推迟 AI 自动化，并指出即使付出巨大努力，大多数公司也无法在 2027 年之前将其所有内部数据转换为程序化的、机器可读的格式。这突显了 AI 快速取代知识工作所面临的根本性技术和物流瓶颈。
    - ryanhiga2019 提出了当前大语言模型 (LLM) 的一个技术局限性，指出持续存在的幻觉（即事实错误或捏造）限制了 LLM 在经济关键任务中的可靠性和可扩展性。这表明，在广泛替代知识工作变得可行之前，需要在 LLM 的准确性和可信度方面取得重大进展。

### 3. 使用 AI 处理现实世界任务的个人经验

- [**ChatGPT 总结的就诊记录太棒了**](https://www.reddit.com/r/ChatGPT/comments/1l2ojdb/chatgpt_summaries_of_medical_visits_are_amazing/) ([分数: 2520, 评论: 211](https://www.reddit.com/r/ChatGPT/comments/1l2ojdb/chatgpt_summaries_of_medical_visits_are_amazing/))：**一位用户描述了如何使用 ChatGPT 处理医院就诊的录音和转录文本，将复杂的医学对话转化为远程家属易于理解的摘要。据报道，该工作流包括录制对话（经同意）、将音频转录为文本，并提示 ChatGPT 生成可读性强、面向外行的医学摘要。评论者证实了类似的用例，例如总结用于癌症诊断沟通的 MyChart 记录；只要输出基于官方医疗记录，准确性就被认为很高，一些用户建议使用 Google 对输出进行二次检查。通过使用 Google Docs 进行静态、可评论的共享，可以进一步改进该工作流。** 关键讨论点包括：ChatGPT 在总结直接医疗文档与回答无锚点查询时的可靠性（降低了幻觉风险），以及实用的工作流建议，如利用协作文档平台进行更有效的信息传播和反馈。
    - 几位用户描述了如何使用 ChatGPT 将就诊记录和检查结果（如来自 MyChart 或 MRI 报告的内容）翻译成外行易懂的摘要。该过程通常涉及提取报告数据、通过删除识别信息进行匿名化处理，并将其粘贴到 ChatGPT 中，ChatGPT 既能保留原始格式，又能生成逐段的平实语言解释。
    - 讨论关注于通过将 ChatGPT 的摘要与 Google 或其他来源进行交叉引用来双重检查事实准确性，这有助于降低幻觉或错误的风险，尽管用户报告在输入特定医疗文档时可靠性很高。
    - 工作流优化建议包括将生成的摘要存储在 Google Docs 等协作文档中，以便进行静态共享和集体评论，或者要求 ChatGPT 生成一份要在医疗咨询中提出的问题清单——增强了对非技术背景家属的互动性和实用性。
- [**我尝试用 AI 替代自己工作一周。这是真实发生的情况**](https://www.reddit.com/r/ChatGPT/comments/1l2gbz9/i_tried_replacing_myself_with_ai_for_a_week_heres/) ([分数: 679, 评论: 111](https://www.reddit.com/r/ChatGPT/comments/1l2gbz9/i_tried_replacing_myself_with_ai_for_a_week_heres/))：**楼主（OP）在一周内尝试用 AI 工具替代其在物流公司的运营助理工作：使用 ChatGPT-4 创建电子邮件/SOP，使用 Blackbox AI 进行文档摘要，使用 Notion AI 记录会议笔记，以及使用 Zapier+GPT 进行任务自动化。AI 在处理结构化/重复性任务（SOP、模板化邮件）时表现最好，但需要大量的用户监督和上下文注入，以避免生成通用或机械化的输出。该实验实现了约 12 小时的时间节省，但强调了在编排和语境化 AI 工作流中，人工监督仍然至关重要。** 热门评论中未出现实质性的技术辩论；讨论大多是非技术性的闲谈和元评论。
    - 一位评论者将文章的主题与软件开发趋势进行了类比，指出虽然程序员可能越来越多地被 AI 替代或辅助，但对承担更广泛职责或具备系统级专业知识的软件工程师的需求仍在持续。这表明自动化正在提高所需的技能水平，而不是完全消除职位。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要的摘要
> 

**主题 1：模型前沿：发布、泄露与悬而未决的问题**

- **Gemini 2.5 Pro 和 "Goldmane" 展现实力，o3 Pro 表现低调**：Google 的 **Gemini 2.5 Pro** 接近正式发布，其 "Goldmane" 版本在 [Aider webdev benchmark](https://aider.chat/) 上表现出色；而 OpenAI 备受期待的 **o3 Pro** 依然难以捉摸，早期报告称其表现不佳，且仅有 **500 LOC** 的代码生成限制。与此同时，Google 神秘的 **"Kingfall"** 模型（可能是拥有 **65k** 上下文窗口的 **DeepThink**）在 AI Studio 上短暂进行了一次“机密”亮相，引发了好奇心，也让部分 Googler 担心职业安全。
- **日本发布 Shisa-v2 405B，性能超越巨头？**：被誉为日本最强模型的 **Shisa-v2 405B** 正式发布，声称在日语和英语方面具有与 **GPT-4/Deepseek-comparable** 相当的性能，并邀请用户在 [chat.shisa.ai](https://chat.shisa.ai/) 进行测试。这款由 H200 节点驱动的巨兽的详细技术报告预计很快将在 Arxiv 上发布。
- **Qwen 挑战 Deepseek，Perplexity Pro 用户抱怨**：来自[阿里云的 Qwen 模型](https://chat.qwen.ai/)因其 1M 上下文窗口在推理能力上超越了 **Deepseek R1** 而备受关注，Perplexity 可能会将其用于深度研究。与此同时，**Perplexity Pro** 用户对较小的上下文限制（5-10 个来源）和糟糕的记忆力表示不满，一位用户感叹道：“是的，你必须不断提醒它你在问什么。”

**主题 2：Agentic AI 崛起：框架、特性与挫折**

- **OpenAI 和 LlamaIndex 为 Agent 开发者助力**：OpenAI 推出了 [TypeScript 版 Agents SDK、RealtimeAgent 功能和 Traces 支持](https://x.com/OpenAIDevs/status/1929950012160790876)，赋能开发者构建更可靠的 Agent，Perplexity 和 Intercom 等早期测试者已展示了相关成果。LlamaIndex 提供了一个[动手实践的 Colab，用于构建多 Agent 财务报告聊天机器人](https://twitter.com/llama_index/status/1930051898247393729)，利用了 agentic RAG 和 10-K 文件。
- **Elasticsearch Agentic 工作流变得复杂，Cursor 发布 RIPER**：工程师们正在处理复杂的 agentic 工作流，例如使用 **gpt-41-mini** 进行多步 **Elasticsearch DSL query generation**（[见图](https://cdn.discordapp.com/attachments/1046317269069864970/1379745858639237140/agent_ES.png?ex=684204b7&is=6840b337&hm=eadc26c0018f544fedd3a9e8e6407dcf5f02301750f66ef5732067f94b94beff&)），而新的 **CursorRIPER framework** 旨在通过规则、记忆和技术上下文文件来引导 Agent 行为，确保项目步入正轨。同时，**HTNs (Hierarchical Task Networks)** 正被探索用于微调 ReACT 格式的 LLM Agent，以实现更好的结构化交互。
- **MCP vs. A2A：Agent 协议大辩论**：**MCP (Meta-agent Communication Protocol)** 正在讨论通过 API keys 进行变现以及跨 Agent 的上下文管理，状态转移指南可在 [fast-agent.ai](https://fast-agent.ai/mcp/state_transfer/) 获取。然而，Google 的 **A2A (Agent-to-Agent) framework**（[GitHub repo](https://github.com/google/A2A/)）作为竞争对手出现，一些开发者更倾向于使用 **A2A spec** 来构建多 Agent 系统，并利用 **pydantic-ai-slim**（[pydantic-ai docs](https://ai.pydantic.dev/install/)）等工具及其便捷的 `.to_a2a()` 方法。

**主题 3：底层探秘：GPU 优化、硬件特性与性能难题**

- **Blackwell 基准测试表现惊艳，MI300X 分析令人困惑**：NVIDIA 的 Blackwell 架构在 [Cutlass 示例](https://github.com/NVIDIA/cutlass/tree/main/examples/70_blackwell_gemm)中展示了惊人的性能，**NVFP4** 达到了 **3.09 PetaFLOPS/s**，尽管其 **MXFP8/BF16** 性能（**0.23 PetaFLOPS/s**）引起了关注。与此同时，AMD **MI300X** 用户在 **gfx942** 上读取 **L2CacheHit** 时遇到了 `rocprof` 错误，尽管 [ROCm 文档](https://github.com/ROCm/ROCm/blob/develop/docs/conceptual/gpu-arch/mi300-mi200-performance-counters.rst)表明支持该功能，并注意到低 L2 缓存命中率与低 **MfmaUtil** 分数相关。
- **CUDA 和 ROCm 开发者钻研 Kernel 与工具**：开发者们深入研究 GPU 编程，讨论 CUDA 屏障状态（如 `__syncthreads()` 与 `bar.sync`，参考 [NVIDIA 关于可编程性的 Volta 博客](https://developer.nvidia.com/blog/volta-new-programmability-features)），并利用 libcu++ 中的 `cuda::pipeline` 实现生产者/消费者方案（[CUDA Zone 资源](https://developer.nvidia.com/cuda-zone)）。在 AMD 方面，Snektron 分享了他的 [AMD FP8 矩阵乘法 Kernel 解决方案](https://github.com/Snektron/gpumode-amd-fp8-mm/blob/main/solution.hip)和一篇探索 MI300 合并访问的[详细文章](https://akashkarnatak.github.io/amd-challenge/)。
- **Tinygrad 和 Torchtune 用户追求性能并解决 Bug**：**Tinygrad** 用户正努力移除 NumPy 依赖，却发现操作被卸载到了 GPU，同时还在解读繁杂的 `DEBUG=2` 输出，并解决极其缓慢的 LSTM 层问题。**Torchtune** 开发者正在处理 [Iterable Dataset 重构 RFC (#2785)](https://github.com/pytorch/torchtune/pull/2785)，并在分布式环境中测试 AdamW 以外的优化器（如 SGD 和 Adafactor）时遇到了 `DeviceMesh` 错误。

**主题 4：前沿研究：微调突破、语义威胁与新型架构**

- **参数高效微调（PEFT）承诺巨大收益**：一种新型的参数高效微调方法声称，与全量微调和 LoRA 相比，在使用更少参数的情况下，**知识吸收率提高约 4 倍**，**灾难性遗忘减少 30%**。该技术在将模型适配到新领域以及高效嵌入特定知识而不覆盖现有能力方面特别具有前景。
- **世界模型面临“语义病毒”感染**：一篇关于[通用 Agent 和世界模型的新论文](https://arxiv.org/pdf/2506.01622)提出，如果模型存在“漏洞”或断连区域，**“语义病毒”**可以通过“感染”推理路径来利用 LLM 世界模型的漏洞。据报道，该病毒会劫持世界模型在上下文窗口内的当前激活，而不是重写基础模型本身。
- **自博弈与负责任的 AI 推动 LLM 边界**：研究人员正在探索创新的训练范式，一篇关于[*通过文本自博弈演进 LLM*](https://ai.vixra.org/abs/2506.0018)的论文正在寻求社区关于实现涌现性能的反馈。同时，IBM 推出了[开源的 Responsible Prompting API](https://github.com/IBM/responsible-prompting-api)（[配套论文](https://arxiv.org/abs/2504.08757)，[HF Spaces 演示](https://huggingface.co/spaces/santanavagner/responsible-prompting-demo)），旨在推理前引导用户获得更准确、更符合伦理的 LLM 输出。

**主题 5：生态系统演进：API 变革、社区工具与开发者资源**

- **API 动荡：Anthropic 削减容量，OpenAI TTS 定价令人困惑**：Anthropic 在不到五天的通知时间内突然削减了大部分 **Claude 3.x 模型容量**，影响了 Windsurf 等服务（[参见 _mohansolo 的推文](https://x.com/_mohansolo/status/1930034960385356174)），作为回应，ai.engineer 提供了 [BYOK 选项和改进的 Agent 框架](https://x.com/kevinhou22/status/1930401320210706802)。用户还质疑为什么 OpenAI 的 **gpt-4o-mini-tts** 成本明显高于 **tts-1**（尽管列出了价格），并指出了 [OpenAI 社区论坛](https://community.openai.com/t/new-tts-api-pricing-and-gotchas/1150616)上讨论的潜在陷阱。
- **开发工具蓬勃发展：年鉴、聊天界面和可解释性工具包**：Modal Labs 推出了 [The LLM Engineer's Almanac](https://x.com/charles_irl/status/1929615080494416213)，提供了数千个推理基准测试，而 **GitHub Chat** 通过将 `github.com` 更改为 `githubchat.ai` 提供了一种与仓库交互的新方式（例如：https://githubchat.ai/blueraai/universal-intelligence）。用于视觉/视频可解释性的 **Prisma** 工具包，现在支持 [Hugging Face 模型](https://huggingface.co/)并提供 [100 多个模型电路风格的代码示例](https://x.com/soniajoseph_/status/1930286144471646252)，并在 CVPR 2025 上获得了 Oral presentation（口头报告）的认可。
- **开源 Agent 更名，数据政策引发争议**：**OpenManus** 更名为 **agenticSeek**（[GitHub 仓库](https://github.com/Fosowl/agenticSeek)），可能是出于版权考虑，这与 OpenDevin 更名为 OpenHands 的做法如出一辙。与此同时，一篇 [ArsTechnica 的文章报道称 OpenAI 被迫保存所有 ChatGPT 日志](https://arstechnica.com/tech-policy/2025/06/openai-says-court-forcing-it-to-save-all-chatgpt-logs-is-a-privacy-nightmare)，包括已删除的聊天和 API 数据，这引发了工程师们关于隐私的讨论。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 举办 Reddit AMA**：**Aravind (CEO), Denis (CTO), Tony (VP Engineering), Weihua (Member of Technical Staff), 以及 Tyler Tate (Product)** 在太平洋时间上午 10 点主持了一场关于 **Perplexity Labs** 的实时 **Reddit AMA**，附带 [Reddit AMA 链接](https://www.reddit.com/r/perplexity_ai/comments/1l39wux/ama_with_perplexitys_aravind_srinivas_denis/)。
   - AMA 涵盖了用户对产品的反应、核心用例以及即将推出的功能。
- **Yarats 跳槽至 Perplexity**：根据[此公告](https://www.perplexity.ai/)，**Denis Yarats（联合创始人兼 CTO）** 加入了 **Perplexity AI** 团队；然而，成员们想知道 *Deep Research High 在哪里*？
   - 一位成员对 **Deep Research High** 的延迟表示沮丧，并发布了一个[困惑的 GIF](https://tenor.com/view/confused-huh-what-gif-15066348) 作为回应。
- **GPTs Agent 遭受“失忆症”**：成员们讨论了 **GPTs Agent** 在初始训练后无法从额外信息中学习的问题，强调[上传的文件被保存为知识](https://link.to/openai-docs)，但*不会持续修改 Agent 的基础知识*。
   - 对话强调了 **GPTs Agent** 在保留信息和适应新数据方面的局限性。
- **Perplexity Pro 用户待遇不佳**：成员们批评了 **Perplexity Pro** 计划的上下文限制（5-10 个来源），认为上下文窗口小且无法记住之前的消息是一个关键问题。
   - 一位成员指出：*是的，你必须不断提醒它你在问什么*，表达了对该工具记忆能力的沮丧。
- **Qwen 优于 Deepseek**：成员们表示 [**Qwen**](https://chat.qwen.ai/) 模型在推理能力上超过了 **Deepseek R1**，拥有 1M 上下文窗口，并表示 **Perplexity** 将利用它进行深度研究。
   - 进一步的讨论强调了 **Qwen** 作为免费模型的可访问性，与经常拥堵的 **Deepseek** 服务器形成对比。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Goldmane 预示 Gemini 2.5 Pro GA！**：**Gemini 2.5 Pro** 的发布指日可待，**Goldmane** 版本在 Aider webdev 基准测试中获得了 **86%** 的分数，详见[此处](https://aider.chat)。
   - 根据 [Aider Docs](https://aider.chat/docs/more/edit-formats.html#diff-fenced)，**diff-fenced** 编辑格式主要用于 **Gemini models**。
- **Kingfall：Google 意外发布 DeepThink 模型引发热议**：一个名为 **Kingfall** 的模型（被认为是内部 **Gemini model**）曾短暂出现在 AI Studio 上，引发了对其能力以及它是否就是 **DeepThink** 的猜测。
   - 成员们注意到它拥有 **65k** 的上下文窗口，但“机密”名称暗示可能有人要被解雇了。
- **OpenAI 的 o3 Pro 仍然缺席？**：**OpenAI's o3 Pro** 的发布备受期待，但发布日期仍不确定，早期印象也反响平平，一位成员表示：*"我已经拿到了，很烂"*。
   - 针对 **o3 Pro** 在生成代码方面的限制出现了担忧，其上限为 **500 LOC**，而其前代产品可以生成 **2000 LOC** 且无遗漏。
- **模型对决：空间推理能力展示**：各种模型之间正在进行对比，包括 **Gemini 2.5 Pro**、**Claude Opus**、**Grok** 和 **OpenAI's o3**，重点关注编程熟练度、推理和整体性能。
   - 一位用户通过给 Kingfall 一个 [geoguessr 任务](https://www.geoguessr.com) 测试了它的**空间推理**能力，结果令人惊叹。
- **免费 API 使用结束，Google 捂紧钱包**：取消 **Gemini 2.5 Pro** 的免费 API 访问引发了失望，特别是对于长内容生成的用例。
   - 一位用户开玩笑说 Gemini 现在需要信用卡详情，并使用有效的付款详情注册以提供 *$300 免费额度*。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **授权问题困扰 Cursor Pro GPT-4.1 访问**：升级到 **Cursor Pro** 后，一些用户在尝试访问 **GPT-4.1** 时遇到 *'User is unauthorized'* 错误，需要 **Cursor team** 的干预。
   - 受影响的用户正在分享请求 ID 和账户邮箱以解决该问题。
- **Claude 4 Sonnet 的上下文危机激发 Prompt Engineering**：用户报告 **Claude 4 Sonnet** 的上下文窗口有限，导致对话中断，但建议使用 *'continue where you left off'* 的提示词技巧。
   - 一位用户推测 **Claude 4** 具有 *'rolling context'*，在整个聊天过程中会考虑关键因素。
- **使用 CursorRIPER 框架重塑你的工作流**：**CursorRIPER framework** 通过规则和记忆来引导 Agent 行为，以维持上下文并专注于项目，这由一个 **tech context file** 支持。
   - 该框架旨在防止使用过时的模块，并确保 Agent 在重大编辑后仍能感知项目的当前状态。
- **Claude Code 脱颖而出成为重构明星**：一些成员宣布 **Claude Code** 在特定任务上优于 **Cursor**，并根据最近的经验赞扬其“极其聪明”的编程能力。
   - 一位用户声称使用 **Claude Code** 成功地对一个大型复杂代码库进行了 one-shot 重构，通过了数千个测试且没有错误。
- **Cursor 1.0 发布，带来代码智能与后台任务**：最新的 **Cursor 1.0** 版本包括增强的**代码审查能力**（能够记住自己的错误）、改进的**错误追踪**以及处理多个**后台任务**的能力。
   - 用户可以查看[官方变更日志](https://www.cursor.com/changelog)以获取所有更新的详细概览。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **o3 Pro 的发布仍存疑问**：成员们推测了 **o3 Pro** 的发布，而其他人由于 **Sam Altman** 之前的延迟和未兑现的公告而保持怀疑。
   - 一位成员嘲讽道：*“不会有 o3 pro。他们会发布 chatgpt5。”*
- **OpenAI 员工预告新功能**：OpenAI 员工预告了 **Teams** 和 **Enterprise** 计划的重大更新，新的 **Connectors** 功能允许用户使用推理模型对内部资源进行搜索。
   - 根据一位成员的说法：*“他们刚刚发布了一个更新，今天的公告对 Teams 用户非常有益，原因是我们可以使用任何推理模型来搜索内部资源。”*
- **TTS 价格差异引发讨论**：一位成员质疑为什么 **gpt-4o-mini-tts** 的收费大约是 **tts-1** 的 4 倍，尽管列出的价格分别是每 1M 字符 **$12** 对比 **$15**。
   - 另一位成员建议查看 [OpenAI 社区论坛](https://community.openai.com/t/new-tts-api-pricing-and-gotchas/1150616) 以了解潜在的陷阱。
- **Agent 流程旨在查询 Elasticsearch**：一位成员正在使用 **open ai gpt-41-mini** 构建一个 Agent，根据人类查询创建 **Elasticsearch DSL queries** 用于图表制作，从单个 Agent 开始，并将其拆分为多个 Agent 以识别索引名称、获取映射、生成查询并提取数据，如[此附图](https://cdn.discordapp.com/attachments/1046317269069864970/1379745858639237140/agent_ES.png?ex=684204b7&is=6840b337&hm=eadc26c0018f544fedd3a9e8e6407dcf5f02301750f66ef5732067f94b94beff&)所示。
   - 另一位成员指出了当前设置中至少存在 *7 个* 问题，其中最大的问题是在 **Elasticsearch** 中对所有内容（甚至是索引）进行排序。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek 遇到速度瓶颈**：一位用户报告称，在 Mac Studio 上 **DeepSeek R1 0528** 的运行速度 (**12.8 t/s**) 慢于 **R1** (**18.7-19 t/s**)，但有人建议不同的量化格式可能是原因。
   - 有人提出动态量化可能表现不同，从而影响模型的速度。
- **Qwen 的泛化能力受到质疑**：一位用户认为 **Qwen 4B** 的泛化能力不如 **Gemma 4B**，强调了泛化能力方面的潜在差异。
   - 该用户未提供任何额外细节。
- **Llama.cpp 拯救视觉功能**：寻求 **unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF** 视觉功能的用户被引导使用 *llama.cpp*，并[获得了指令](https://github.com/ggerganov/llama.cpp)。
   - 步骤包括克隆仓库、创建构建、启用 CUDA 以及构建 *llama-cli*。
- **多 GPU 支持即将推出™️**：**Multi-GPU support** *已经可以通过 accelerate 运行*，预计在 7 月初会推出*更好的版本*。
   - 由于当前支持的*非官方*性质，未提供官方示例，但熟悉 accelerate 的用户可以使用它。
- **最快库对决**：对于单用户 CPU 推理，基于 [llama.cpp](https://github.com/ggerganov/llama.cpp) 的库可能是最好的，而 [vLLM](https://github.com/vllm-project/vllm) 或 [ktransformers](https://github.com/ktransformers/ktransformers) 更适合 CPU 部署。
   - 已经有关于处理此问题的 **v0 engine** 的工作，但它在 **v1** 中不存在。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 增加跨模型 GIF 支持**：OpenRouter 现在在 **OpenAI, Gemini, Anthropic 和 Llama 路由**上支持 `image/gif` 图片提示词，简化了动画的使用。
   - 这消除了用户将动画预转换为其他格式的需求。
- **iOS 应用集成 OpenRouter**：一款 iOS 应用即将通过 **TestFlight** 发布，该应用使用 **OpenRouter** 作为其 **LLM 后端**并采用角色卡（character cards）。
   - 由于消息格式化的复杂性，开发者仍在进行相关工作，但目标是稍后添加更多客户端。
- **Anthropic 模型速率限制（Rate Limits）放宽！**：OpenRouter 现在为 **Opus** 提供更高的速率限制，特别是在将流量路由到 **Anthropic** 模型时，这引发了关于 **Chutes** 经济模式的讨论。
   - 考虑到必要的 GPU 资源，人们对 **Chutes** 商业模式的可持续性产生了猜测。
- **Nous 在分布式训练中遇到困难**：**Nous** 正在尝试使用 **416 块 H100** 分布式训练一个 SOTA 模型，但进展缓慢。
   - 预计训练时间将延长至明年，尽管声称有突破能将 GPU 间带宽需求降低至 ~300mbps，但仍引发了质疑。
- **OpenRouter API 最大化利用技术**：成员们讨论了通过 OpenRouter 向 **LLM** 发送 **100K 调用**的策略，重点关注吞吐量和供应商折扣。
   - 共享了如 Modal 的 **LLM Almanac Advisor** 等资源，以优化 **API** 使用并降低成本。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU Mode 吉祥物概念化**：成员们提议创建 **GPU Mode** 吉祥物和周边商品，初步建议是“超级赛亚人 GPU”设计。
   - 在使用 **ChatGPT** 生成潜在设计后，成员们发现生成的图像不够简洁，无法有效地作为 Logo 或吉祥物，且存在[版权担忧](https://en.wikipedia.org/wiki/Copyright_law)。
- **CUDA Barrier 状态揭秘**：`__syncthreads()` 基本上是 `bar.sync`/`barrier.sync.aligned`，而 `sync(cooperative_groups::this_thread_block())` 为不同分支中的线程同步提供 `barrier.sync`（仅限 [Volta](https://developer.nvidia.com/blog/volta-new-programmability-features/) 及更新版本）。
   - 对于生产者/消费者方案，使用 libcu++ 中的 `cuda::pipeline` 是 [CUDA](https://developer.nvidia.com/cuda-zone) 的正确做法。
- **CUPTI 命令缓冲区溢出**：**CUPTI** 分析中的高开销可能是由于 GPU 命令缓冲区满导致的瓶颈，参考 [CUpti_ActivityOverheadCommandBufferFullData 文档](https://docs.nvidia.com/cupti/api/structCUpti__ActivityOverheadCommandBufferFullData.html#structcupti__activityoverheadcommandbufferfulldata)。
   - 一位成员指出，在 Torch Dynamo 中直接使用 Python 常量会触发重新编译，如日志 `___as_tensor(alpha).item() == 0.5` 所示。
- **vLLM 获得 VL 模型修复**：通过[此 GitHub pull request](https://github.com/vllm-project/vllm/pull/19147) 发布了针对 **vLLM** 和 **VL 模型**的修复。
   - 在修复之前，在 **vLLM** 中加载序列化的 ao 模型在所有层都量化的语言模型上可以工作，但在视觉模型未量化的 VL 模型上会出错。
- **MI300X 分析难题依然存在**：一位成员报告了使用 `rocprof` 读取 **MI300X** kernel 的 **L2CacheHit** 时遇到的问题，指出虽然该指标在 [ROCm 文档](https://github.com/ROCm/ROCm/blob/develop/docs/conceptual/gpu-arch/mi300-mi200-performance-counters.rst)中列为可用，但 `rocprof` 返回错误提示在 **gfx942** 上不支持。
   - 成员们在分析 **FetchSize**、**WriteSize**、**MfmaUtil** 和 **SQ_LDS_BANK_CONFLICT** 时发现，低 L2 缓存命中率与低 **MfmaUtil** 分数相关。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **IBM 通过新 API 实现负责任的提示词（Prompting）**：一名 IBM 实习生介绍了 **Responsible Prompting API**，这是一个[开源项目](https://github.com/IBM/responsible-prompting-api)，旨在提供推理前的提示词建议，使 LLM 输出更加负责任、准确且高效，详见[这篇论文](https://arxiv.org/abs/2504.08757)。
   - 该系统在 [HF Spaces](https://huggingface.co/spaces/santanavagner/responsible-prompting-demo) 上进行了演示，旨在帮助缺乏提示词技巧的领域专家，有望减少有害输出并降低推理成本。
- **区块链技术提升 AI 输出？**：一位成员分享了一篇[概念论文](https://medium.com/@info_65774/consensus-validation-for-llm-outputs-applying-blockchain-inspired-models-to-ai-reliability-f642d7f96f8e)，探讨利用**类区块链共识机制**来提高 LLM 输出的可靠性和可信度。
   - 该论文重点关注 AI Agent、法律/医疗工具以及 AI 对齐（AI alignment）应用。
- **Whisper 低成本转录音频**：用户正在利用 **OpenAI 的 Whisper 模型**以经济实惠的方式进行音频转录，[volodymyr kublytskyi 的仓库](https://huggingface.co/spaces/vkublytskyi/Final_Assignment_Agent/blob/main/tools/youtube_video_tool.py) 辅助了 Agent 的视频交互。
   - 成员们正在将 **Gemini-2.0-flash** 与 **SmolAgents** 结合使用，并指出它在 OpenAI 服务器上的表现*相当不错*。
- **市场调研基础**：一位成员分享了他们正在学习的**市场调研基础**和 **ACP 漏斗（ACP Funnel）**。
   - 他们还注意到，*带有图片的深度长文在 X 上获得的互动最多*。
- **Prisma 表现出色并适配 HF**：**Prisma** 工具包专为视觉和视频的机械可解释性（mechanistic interpretability）设计，在 CVPR 2025 工作坊中获得了口头报告（Oral presentation）机会，并适配了 [Hugging Face 模型](https://huggingface.co/)。
   - 正如 [Twitter](https://x.com/soniajoseph_/status/1930286144471646252) 上提到的，此次发布包含了针对 **100 多个模型**（包括 CLIP、DINO 和 video transformers）的电路风格代码（circuit-style code）以及交互式笔记本。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 任务达到上下文限制，需从头开始**：一位用户发现 **Manus** 在运行 **1 小时 55 分钟**后达到了上下文限制（context limit），需要开启新任务并从头开始。
   - 该用户对达到上下文限制后进度丢失表示失望。
- **H Runner 竞争 AI 关注度**：一位成员分享了 *H Company* 的 **H Runner** 链接（[https://www.hcompany.ai/](https://www.hcompany.ai/)），将其定位为 **Manus AI Agent** 的竞争对手。
   - 成员们分享称该工具目前免费，但尚不够先进，且限制为**每日 10 次运行**。
- **Manus 额度消耗引发争议**：一位用户报告称，为了制作一份 30 页的 PowerPoint 演示文稿花费了 **$50**，并指责 **Manus** 的排版超出了幻灯片边界。
   - 另一位用户发现一段 **30 秒**的视频耗费了 **208 个额度**，而其他用户则在分享邀请链接以获取更多额度。
- **交互体验：网页版 vs App？**：成员们就交互体验的最佳部署方式展开辩论：是网站还是像 GitHub 上的 JS 一样托管为 App。
   - 一位成员建议这取决于产品，并举例说交互式电影需要大屏幕，而语言学习 App 则受益于记忆和口语练习。
- **Cursor, Devin 和 Replit：IDE 使用印象**：一位成员表示，他们在创建网站和 Web App 时，需要将 **Manus** 的输出在 **Cursor** 或其他 IDE 中进行重构才能使其正常运行。
   - 另一位成员一直在试用 **Cursor**、**Devin 2.0** 和 **Replit**，发现后者在“一天开发一个 App”方面非常灵巧。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepHermes 24B 陷入 API 故障深渊**：**DeepHermes 24B** 模型遭遇了 **API 停机**，影响了其 **API** 和 **Chat Product** 功能。
   - 用户已收到通知，并被要求在团队处理中断期间保持耐心。
- **Nous Research 关注服务器标签**：一名成员请求为 **Nous Research** 添加 **服务器标签 (server tag)** 以增强可见性，并引用了 [Discord 关于服务器标签的文档](https://support.discord.com/hc/en-us/articles/31444248479639-Server-Tags)。
   - 该请求获得了积极反馈，并保证将在 **24 小时内** 实施。
- **Shisa-v2 405B 在日本首次亮相**：**Shisa-v2 405B 模型**发布，这是在**日本**训练的最强大的模型，专注于**日语**和**英语**，性能可与 **GPT-4/Deepseek** 媲美。
   - 邀请用户通过其 **H200 节点** 的端点 [chat.shisa.ai](https://chat.shisa.ai/) 测试该模型，并承诺在 **Arxiv** 上发布详细的技术报告。
- **LLM 自博弈论文寻求反馈**：一位成员宣布发表了他们的论文《通过基于文本的自博弈进化 LLM：实现涌现性能》(*Evolving LLMs Through Text-Based Self-Play: Achieving Emergent Performance*)，可在 [ai.vixra.org](https://ai.vixra.org/abs/2506.0018) 获取。
   - 作者正在寻求社区对其研究和观察到的涌现性能（emergent performance）的反馈和见解。
- **Merlin App 现在可以“听”了**：一位成员重点介绍了 [Merlin 鸟类识别 App](https://merlin.allaboutbirds.org/)，指出其能够通过**照片和声音**识别鸟类物种。
   - 该 App 的**声音分析**功能为其现有的照片分析能力提供了补充，提供了一种全面的鸟类识别方法。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Modal Labs 发布 LLM 工程师年鉴**：Modal Labs 推出了 [LLM Engineer's Almanac](https://x.com/charles_irl/status/1929615080494416213)，包含数千个针对 **vLLM**、**SGLang** 和 **TensorRT-LLM** 框架下开放权重模型的 **LLM 推理基准测试**。
   - 该发布内容包括测试结果、可复现的代码，以及一份涵盖自建 vs 购买、成本估算和框架选择的高管摘要，还有用于理解性能指标的 **'stopwatch' 基准测试框架**。
- **AWS Textract 准确性问题报告**：一个在 AWS 中自建的 **PDF 摄取流水线** 使用 Lambda 拆分 PDF 并使用 Textract 进行解析，并使用队列管理 Textract 请求限制。
   - 一位用户警告说，**Textract 在法律和监管文件上的准确率**可能低至 *3%*，并链接到了一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/robertreich_when-word-for-word-accuracy-is-key-in-etl-activity-7265008546793086978-hfaj?utm_source=share&utm_medium=member_desktop&rcm=ACoAAABOb18Bac53omUsFRAIBEVDUe013Eez5zoTry)。
- **Anthropic 容量削减引发混乱**：据[此帖](https://x.com/_mohansolo/status/1930034960385356174)称，Anthropic 在不到五天的通知时间内，意外切断了几乎所有 **Claude 3.x 模型容量**，影响了 Windsurf 等服务。
   - 用户表示失望，部分用户考虑迁移，而 ai.engineer 正在提供 **BYOK 选项**，并改进了针对 Gemini 2.5 Pro 和 GPT-4.1 的 **Agentic 框架**，详见[此帖](https://x.com/kevinhou22/status/1930401320210706802)。
- **Altman 为 Codex 启用联网功能**：Sam Altman 宣布，AI 编程工具 **Codex** 现在为 **ChatGPT Plus** 用户提供可选的联网功能，由于[此推文](https://x.com/sama/status/1930006856019390521)中描述的复杂权衡，该功能默认禁用。
   - 社区讨论了其影响和潜在的安全担忧，Grok 对该公告提供了详细解释。
- **OpenAI 致力于提升 Agent 可靠性**：OpenAI 宣布了构建 Agent 的四项更新：TypeScript 版 Agents SDK、RealtimeAgent 功能、Realtime API 会话的 Traces 支持，以及语音转语音模型的改进。
   - 这些增强功能旨在提高可靠性、一致性和用户控制力，[此推文](https://x.com/OpenAIDevs/status/1929950012160790876)展示了 **Perplexity**、**Intercom** 和 **VolleyGames** 等早期测试者的成果。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Google Workspace 复制 NotebookLM 功能**：一位用户分享了一篇 [Chrome Unboxed 文章](https://chromeunboxed.com/google-workspace-feature-drop-for-may-2025-is-loaded-with-new-features/)，指出 **NotebookLM** 的功能正被集成到 **Google Workspace** 中，最初将针对单个文档开放。
   - 用户们正在推测 **NotebookLM** 何时会升级到更先进的模型（如 **Gemini 2.5 Pro** 或 **Flash**）以提升其性能。
- **Flash 与 Pro 的对决**：成员们讨论了 **Gemini 2.5 Flash** 与 **2.5 Pro** 的优劣，指出 **Pro** 的详尽性在处理需要关注细微细节的大文件上传时更具优势。
   - 一位用户建议实施 Beta 分支，允许用户切换到 **2.5 Pro**，以牺牲处理时间为代价换取潜在的更高质量输出。
- **发现 NotebookLM 音频概览长度调整技巧**：用户发现 **NotebookLM** 中的音频概览长度可以通过在工作室中选择“*Customize*”（自定义）而非“*Generate*”（生成）来调整，从而开启短、默认或长长度的选项。
   - 该自定义功能在网页版和移动网页版上可用，但在官方移动应用中可能缺失。
- **Google Docs 同步需要手动重新同步**：用户确认在将 **Google Doc** 作为来源添加到 **NotebookLM** 后，后续对 **Google Doc** 的任何更改都不会自动同步；需要从预览界面进行手动重新同步。
   - 此外还澄清了 **NLM** 中新的公开分享选项并不依赖于 **Gdoc** 自身的分享设置，因为 **NLM** 分享的是其自身的副本，且分享链接在更新过程中保持不变。
- **NotebookLM 移动应用功能缺失**：**NotebookLM** 移动应用被认为是一个“最小价值产品”（minimal value product），因为它与网页版缺乏功能对等。
   - 鼓励用户在功能请求频道的“*Mobile App*”线程中提交功能请求，以推动改进。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **参数高效微调取得进展**：据报道，一种新的**参数高效微调**（parameter-efficient finetuning）方法与全量微调和 **LoRA** 相比，实现了 **4 倍的知识吸收**，同时减少了 **30% 的灾难性遗忘**（catastrophic forgetting）。
   - 这种方法特别有利于将模型适配到新领域，并在本地设置中整合特定知识而不侵蚀现有知识。
- **知识扩展挑战 RAG 的地位**：一位成员打算使用一系列书籍和文档来扩展 **LLM** 的知识，并将其与类 **RAG** 方法进行对比以寻求建议。他们分享了一个讨论 AI 权利的 [x 链接](https://x.com/unusual_whales/status/1929998955703931375)和一个 [markdown 文档](https://cdn.discordapp.com/attachments/986699377257119794/1379670337008046080/UDAIR.md?ex=68426721&is=684115a1&hm=4e73690d912c8e0286f50b7a456f683012b700561418b45222466ae5230e3a9f&)。
   - 该成员提到这次讨论可能会引发一些“疯狂的对话”。
- **Muon 优化器解析**：一位成员探索了 **Muon** 优化器，该优化器对不适合 **Muon** 的参数使用 **AdamW**，并链接了多任务学习的 [实验结果](https://github.com/KellerJordan/Muon/issues/25)。
   - 据解释，**Muon** 优化器会调整权重矩阵的梯度，使其特征值（eigenvalues）近似等于 1，这与 **SGD** 和 **Adam** 形成了鲜明对比。
- **Mistral Code 旨在提升开发者体验**：**Mistral AI** 推出了 [Mistral Code](https://mistral.ai/news/mistral-code)，这是一款 **AI 驱动的代码助手**，集成了强大的模型、IDE 内助手、本地部署和企业级工具。
   - 它基于开源项目 **Continue** 构建，支持 JetBrains IDEs 和 VSCode，进一步推进了 Mistral 通过 AI 赋能开发者的愿景。
- **ChatGPT 日志受到审查？**：成员们讨论了一篇 [ArsTechnica 文章](https://arstechnica.com/tech-policy/2025/06/openai-says-court-forcing-it-to-save-all-chatgpt-logs-is-a-privacy-nightmare/)，指出 **OpenAI** 被迫保存所有 **ChatGPT** 日志，包括已删除的聊天记录以及来自其 API 业务的敏感数据。
   - 一位成员对这一决定背后的逻辑提出了质疑。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **高效微调登场**：一种新的参数高效微调方法声称比 **LoRA** 的**知识吸收率高出约 4 倍**，且**灾难性遗忘减少了 30%**，同时使用的参数更少。
   - 该方法适用于持续预训练，能在不覆盖现有知识的情况下高效地教给模型新信息。
- **无 API 的 Twitter 爬虫解燃眉之急**：一名成员分享了一个 [Twitter 爬虫](https://gist.github.com/mookiezi/9ea0f0f5aad76a51e5b35a084d82a9df)，它**不使用 API**，将日志记录到 **Postgres**，并跳过转推。
   - 该爬虫不收集回复元数据，因此更适合个人资料抓取和高效的数据收集。
- **世界模型被语义病毒感染**：一篇[论文](https://arxiv.org/pdf/2506.01622)指出，通用 Agent 需要**世界模型**，而**语义病毒**会利用 **LLM 世界模型**中的*漏洞*或*断连区域*来*感染*推理路径。
   - **语义病毒**不会重写基础的 **World Model**，但会劫持其在上下文窗口内的当前激活状态。
- **ROI 质疑会戳破 AI 初创公司泡沫吗？**：一名成员对进入一个 **AI** 的 **ROI** 受到质疑的就业市场表示担忧，这可能导致 **AI 初创公司**的泡沫破裂。
   - 他们声称许多 **AI 初创公司 CEO** 缺乏 **ML** 专业知识，且背后投资人无法正确评估 **ML** 技能，这可能会导致不稳定性。
- **通用算法亮相**：一名成员分享了其[研究](https://github.com/qLeviathan/g-qfnn)的演示，这是一种**通用算法**，具有针对 **NLP**、**期权交易**和**电化学反应**的基础 POC。
   - 这项研究引入了一种新颖的方法，引发了人们对其在不同领域潜在应用的兴趣。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Llama 4 图像支持仍是谜团**：在 Unsloth 版本暗示不支持后，用户们正在质疑 **Llama 4** 在 LM Studio 上是否支持图像，这让社区充满了悬念。
   - 截至目前，讨论中尚未出现明确的确认或否认。
- **agenticSeek 脱胎换骨，由 OpenManus 更名而来**：[agenticSeek](https://github.com/Fosowl/agenticSeek) 已从 **OpenManus** 更名，引发了关于更名原因的询问，这与 OpenDevin 转型为 OpenHands 类似。
   - 推测认为可能涉及版权问题，类似于开源 AI 领域其他备受关注的更名事件。
- **Gemma 在嵌入模型中表现亮眼**：一名测试了各种嵌入模型（**Gemma 3 4b**、**12b**、**Deep Seek 8b**、**Microsoft phi 4 small**）的用户发现，**Gemma** 提供的答案比 Deep Seek 或 Microsoft Phi 更准确，特别是在处理文本和 PDF 混合数据时。
   - 该用户的数据由 0.5-30 MB 不等的文件组成，配合 Supabase 和 n8n 使用。
- **ROCm 视觉模块饱受性能问题困扰**：据[结果截图](https://cdn.discordapp.com/attachments/1110598183144399058/1379953808532049981/image.png?ex=68421da2&is=6840cc22&hm=37d660db87619d86ca215fc8862f4762688295f6516dcb95ee68d5e84a525bc2&)显示，用户报告在 **7900XT 20GB** 上使用新的 **ROCm llama.cpp v1.34.1** 运行时，视觉模块显著变慢，响应时间从约 1 秒跳升至 10 秒以上。
   - 这些发现导致用户被要求在相应的 Discord 频道中分享详细结果，表明这可能是一个需要优化或调试的领域。
- **SSD 秘密：数据损坏与刷新周期揭秘**：关于 **SSD** 数据损坏的讨论显示，如果长时间不通电，数据可能会退化，这与 HDD 不同，后者的内容是物理写入的，随时间退化的可能性较小。
   - 讨论中提到，SSD 中使用的 **NAND** 闪存单元会随时间缓慢泄漏电荷，硬件需要执行*读取刷新（read refresh）*。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP API Key 变现引发 SaaS 辩论**：成员们讨论了为 MCP 变现实施 **API keys** 的方案，认为这反映了带有 API keys 和计费仪表盘的标准 SaaS 模型。
   - 讨论强调 MCP 客户端将处理**到服务器的鉴权（auth to the server）**，这可能会简化变现策略，并让人质疑是否有必要建立专门的 MonetizedMCP 解决方案。
- **A2A 框架与 MCP 争夺 Agent 霸权**：讨论围绕 **A2A** ([https://github.com/google/A2A/](https://github.com/google/A2A/)) 展开，将其作为 Agent 交互中 MCP 的替代框架，一些人注意到其采用率目前有限。
   - 虽然有人推测 A2A 正在幕后通过重大交易获得动力，但其他人表示在多 Agent 系统中更倾向于使用 **A2A spec** 而非 MCP。
- **Pydantic-AI 简化 Agent 开发**：成员们提倡从 **pydantic-ai-slim** ([https://ai.pydantic.dev/install/]) 开始进行 Agent 框架开发，并强调了其便捷的 `.to_a2a()` 方法。
   - 他们提到一个可选的 a2a 组（`uv add 'pydantic-ai-slim[a2a]'`）用于增强现有 Agent，这可能会简化与 A2A 协议的集成。
- **MCP 的 Cloudflare 托管引发难题**：一位成员为缺乏技术背景的用户寻求在 **Cloudflare** 上托管 MCP 服务器的指导。
   - 澄清指出，如果 MCP 客户端提供原生支持，**HTTP transport** MCP 服务器应该不需要本地软件；否则，可能需要一个转换器（translator）。
- **MCP 上下文管理解决 Agent 危机**：一位成员询问 MCP 如何处理跨多个 Agent 的上下文，以及维护该上下文所需的工程机制。
   - 官方澄清 **MCP 并非 Agent 优先（agent-first）**，并在 [https://fast-agent.ai/mcp/state_transfer/] 提供了相关指南，深入介绍了状态转移（state transfer）机制。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 工程师亮相 AI 活动**：LlamaIndex 参加了在旧金山举行的 [@aidotengineer](https://twitter.com/aidotengineer) 活动，CEO @jerryjliu0 和 AI 工程团队在 G11 展位展示了最新的 **Agentic AI** 技术。
   - 同时，来自 LlamaIndex 的 @seldo 在 [@aiDotEngineer](https://twitter.com/aiDotEngineer) 分享了**生产环境中的有效 Agent 设计模式**。
- **LlamaIndex 构建财务报告聊天机器人**：LlamaIndex 提供了一个[实操 Colab](https://twitter.com/llama_index/status/1930051898247393729)，用于从头构建一个**多 Agent 财务报告**生成聊天机器人，该机器人使用 agentic RAG 解析并索引 Adobe 的 10-K 文件。
   - 这源于 @jerryjliu0 的研讨会，LlamaIndex 还演示了如何使用 [LlamaExtract](https://twitter.com/llama_index/status/1930414284670152875) 和 Agent 工作流自动提取 SEC Form 4。
- **黑客松参与者寻求 LlamaIndex 指导**：[@Gradio](https://twitter.com/Gradio) [@huggingface](https://twitter.com/huggingface) MCP 黑客松的答疑时间（Office hours）在此消息后不久开始，最佳 LlamaIndex 提交作品将获得 [$1000 奖金](https://twitter.com/llama_index/status/1930286458340028484)和 10k LlamaCloud 积分。
   - 成员 @tuanacelik 和 @LoganMarkewich 回答了 LlamaIndex 相关问题；HuggingFace 也在其 Discord 服务器上为 **Gradio MCP 黑客松**参与者举办了答疑，[链接在此](https://discord.com/events/879548962464493619/1379561017536938095)。
- **图索引受到密切关注**：一位成员正在探索 **Property Graph Index**，并希望了解**索引和检索的 token 使用情况**，以及**检索和端到端的性能**。
   - 他们正在将其与 **GraphRAG**、**HippoRAG2** 和 **LightRAG** 进行对比。
- **Qwen3 驱动代码解释器 Agent**：一位成员想构建类似[这篇 Medium 文章](https://medium.com/@venugopal.adep/building-an-ai-data-analysis-assistant-with-llamaindex-and-openai-c0e371a432d6)中的**代码解释器 Agent**，但使用 **qwen3** 代替 **OpenAI**。
   - 另一位成员建议使用 **Ollama** 来运行 **qwen3**，[链接在此](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/)。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **NumPy 移除手术转向 GPU**：一位成员正尝试为 `hlb_cifar10` 悬赏任务从 `random_crop/cutmix` 中移除 **NumPy**，却发现 **NumPy** 操作正被卸载到 GPU 上执行。
   - 该用户在建立关于 **tinygrad 性能**的直觉时面临挑战，难以识别性能瓶颈。
- **Windows 用户与 Tinygrad 搏斗**：一位成员报告了在 Windows 上运行 **tinygrad** 的多个问题，包括 JIT 下的 CPU 后端崩溃以及 BEAMS=1 时的挂起。
   - 他们不得不对自动生成文件（autogen files）进行黑客式修改以启用 CUDA，并怀疑其 Windows 环境是导致性能问题的根源。
- **LSTM 在 Tinygrad 中延迟严重**：在将 **VAD 模型**从 PyTorch 移植到 **tinygrad** 时，一位成员发现 LSTM 层的速度明显慢于其他层。
   - 无论选择哪种后端，LSTM 的缓慢问题依然存在。
- **DEBUG=2 解码需要细致入微**：一位成员表示对 **tinygrad** 的 `DEBUG=2` 输出感到不知所措，难以理解各列含义以及大量的内核（kernels）。
   - 他们特别质疑了高数量的 `randperm` 内核以及晦涩的命名约定，例如 `r_512_32_8_4_8_3_16_3_4_4`。
- **CUDA 自定义难题**：一位成员正在寻找将 **CUDA kernels** 与 **tinygrad** 的 CUSTOM 算子结合使用的示例，以便移植一个包含 5-10 个内核的项目。
   - 尽管该成员承认自定义内核可能与 "Zen of TinyGrad"（TinyGrad 之禅）相冲突，但由于对如何用 Python 表达所需内核的理解有限，他们认为这是必要的。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 告别 Python 3.9**：即将到来的 **Python 3.9** 生命周期结束（EOL）因新的 linting 规则导致 CI 失败，需要临时变通方案来维持兼容性并采用新的 linting 规则。
   - 针对需要从 `typing` 模块引入 `Union` 和 `Optional` 的问题，一位成员调侃道：*"抱歉 Joe，这就是 CI 失败的原因 :/"*。
- **异步奖励函数获得批处理提升**：奖励函数通过批处理（batch）进行循环，以实现潜在的并发计算，但这些调用并非原生异步，且受限于 **Reference model worker** 的资源。
   - 一位成员分享道：*"奖励函数只是通过循环处理，并传入一个你可以尝试并发计算的批次，但调用不是异步的，而且你只能访问 Reference model worker 的资源。"*
- **Iterable Dataset 重构 RFC 打破常规**：一项 RFC（[Iterable dataset refactoring](https://github.com/pytorch/torchtune/pull/2785)）提议对 TorchTune 处理数据集的方式进行重大重构，并征求社区对其设计和潜在破坏性变更的反馈。
   - 一位成员强调了反馈的重要性：*"这是一个巨大的变化。我非常感谢任何输入或想法。这感觉像是 torchtune 处理数据集的正确方式吗？既然我们横竖都要打破现状，你会进行什么彻底的改变吗？"*
- **DTensor DeviceMesh 错误困扰优化器测试**：在全分布式 SFT 中测试 TorchTune 与 **AdamW** 以外的优化器（如 **SGD**、**Adafactor** 和 **Adagrad**）时，出现了与 `aten._foreach_lerp_.ScalarList!` 的 dtensor 参数相关的 `DeviceMesh` `AssertionError`。
   - 其他人已经测试了来自 torchao 的不同精度的 **Muon** 和 **AdamW**。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 作业截止日期是固定的**：在有关延长 **5 月 31 日**截止日期的询问后，工作人员确认表单已经为了应对技术问题额外开放了两天，并且*遗憾的是，他们将无法进一步开放作业提交*。
   - 社区共识似乎是不能再指望进一步的延期。
- **不太可能提供关于 MOOC 作业的详细反馈**：一位成员请求对所有提交的内容提供详细反馈，包括 **AgentX project** 和 **实验作业（lab assignments）**。
   - 工作人员回应称，*作为工作人员，他们没有足够的精力（bandwidth）来做这件事*，但承诺会转达这一建议。
- **MOOC 的未来尚不确定**：有人询问了在 **Spring 2025 MOOC** 结束后是否有下一步计划、新版本或进阶课程。
   - 工作人员表示 *目前还没有任何确定的消息*，但 *可能性很大（但目前不能保证）*，这表明可能会继续，但尚未做出坚定承诺。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Anthropic 的开发周期在 System Prompts 中曝光**：一篇博客文章对比了 **Claude 3.7** 和 **4.0** 的 [system prompts](https://www.dbreunig.com/2025/06/03/comparing-system-prompts-across-claude-versions.html)，揭示了关于 **Anthropic** 开发周期和优先事项的细节。
   - 作者注意到 *Claude 3.7 与 4.0 之间的 system prompt 发生了一些变化*。
- **Oneformer 的博弈论策略**：一名成员正在开发一个 **Oneformer** 博弈论专家，但对是否公开犹豫不决。
   - 该成员还在讨论其与 **Agenspy** 及其他框架相比的潜在成功率。
- **Angel Azul 攻克 Claude SDK**：一名成员分享了他们在 [claude_sdk execution engine](https://github.com/darinkishore/claude_sdk/tree/t1-execution-engine) 上的工作，澄清这仍是一个开发中的项目，可能包含 Bug，架构模式详见 [ai_docs](https://github.com/darinkishore/claude_sdk/blob/t1-execution-engine/ai_docs/ARCHITECTURE_PATTERNS.md)。
   - 该 SDK 相比现有的 Claude SDK 提供了改进。
- **HTN 优化 LLM Agents**：一名成员在使用 **HTN** 时建议，**LLM agents** 可能会从专门针对 **ReACT format** 的微调中受益，而不是采用通用的 Chat Model 方法。
   - 为了适应带有错误重试机制的 **SO/schemas** 等新功能，有必要对路线图进行进一步研究。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 赞助黑客松？**：成员们正在索要 **Cohere** 的联系方式，以探索赞助高等教育黑客松的可能性。
   - 用户们正在寻找负责赞助事宜的对接人。
- **Cohere 团队迎接新成员**：新成员正在 **Cohere** 的 Discord 频道 🤝-introductions 中积极进行自我介绍，分享他们的专业经验、正在进行的项目以及偏好的技术。
   - 根据频道指南，这些介绍展示了社区在 **AI** 和 **GenAI** 领域广泛的技能和兴趣。



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 的 LlamaCPP 库版本滞后**：用户表示驱动 **GPT4All** 的 **LlamaCPP library** 已过时数月，且自动更新到最新版本的功能无法正常运行。
   - 更新该库似乎不仅仅是简单的复制粘贴新版本。
- **MOE 模型变得更轻量**：现在可以用更合理的 **VRAM** 运行更大的 **MOE models**。
   - 这是通过卸载（offloading）某些 Experts 和 Tensors 实现的，需要一些编程技巧来有效管理内存限制。
- **Mac M3 Max 在 VRAM 上大显身手**：**Mac 512 GB** 配置拥有高达 **448 GB** 的 "VRAM"，其价格与四台较新的 **AMD AI MAX 395+ 128 GB** 迷你 PC 或笔记本电脑相当。
   - Mac 的功耗也更低。
- **vLLM 引擎可能为 GPT4All 赋能**：目前正在研究将 **vLLM engine** 添加到 **GPT4All** 项目中，这可能使其成为领先的开源项目。
   - 该项目届时将拥有两个由不同编程语言编写的底层引擎，显著提升其能力。
- **特斯拉的灵光一现**：一位用户分享了一个[链接](https://buck.lighting/blog/nikola-tesla-and-light/)，讨论了尼古拉·特斯拉对能源和光的贡献。
   - 该用户推测 *“他的发明不知何故被窃取了”*。

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Guo Guides Good AI**：行业专家 **Liang Guo** 正在举办一场关于 **用于数据分析的 AI 编程** 的网络研讨会，预约[详情请点击此处](https://forms.gle/e71FSdpwBtDBccgKA)。
   - 该研讨会侧重于实用的 AI 编程技术。
- **SVCAI 夏季竞赛报名中**：**硅谷华人协会 (SVCA)** 正在举办 **AI4Legislation** 夏季竞赛，[详情可在该项目的 GitHub 仓库中查看](https://github.com/svcaf/2025-AI4Legislation-Public)。
   - 该仓库为参赛者提供了资源和指南。



---


**Codeium (Windsurf) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。


---



您收到这封邮件是因为您在我们的网站上选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord: 各频道详细摘要及链接





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1379863824454258748)** (1 条消息): 

> `Reddit AMA, Labs, Aravind, Denis, Tyler Tate` 


- **Perplexity 领导层主持 Reddit AMA**：**Aravind (CEO), Denis (CTO), Tony (VP Engineering), Weihua (技术团队成员) 和 Tyler Tate (产品)** 将于太平洋时间上午 10 点主持一场实时 Reddit AMA，讨论 Perplexity Labs ([Reddit AMA 链接](https://www.reddit.com/r/perplexity_ai/comments/1l39wux/ama_with_perplexitys_aravind_srinivas_denis/))。
- **在 Reddit 上向 Perplexity Labs 提问任何问题**：Perplexity 正在 Reddit 上举办 AMA，回答用户关于他们对产品的反应、核心用例、未来计划等问题！


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1379535149196251136)** (1289 条消息 🔥🔥🔥): 

> `Deep Research High, O3-pro, GPT-5 Release` 


- **Denis Yarats 加入 Perplexity 团队**：Denis Yarats（联合创始人兼 CTO）通过[此公告](https://www.perplexity.ai/)加入 Perplexity AI 团队。
   - Discord 成员开玩笑说 Yarats 的到来，并询问 Deep Research High 在哪里。
- **Deep Research High 仍然延迟**：据一些成员称，Deep Research High 的发布仍然延迟。
   - 一位成员对延迟表示沮丧，并发布了一个[困惑的 GIF](https://tenor.com/view/confused-huh-what-gif-15066348) 作为回应。
- **GPTs Agents 无法学习**：成员们讨论了 **GPTs Agents** 在初始训练后无法从额外信息中学习的问题，且[上传的文件被保存为知识库](https://link.to/openai-docs)。
   - 有人指出 *它们不会持续修改 Agent 的基础知识*。
- **Perplexity Pro 的限制令人恼火**：成员们抱怨 **Perplexity Pro** 计划的上下文限制（5-10 个来源），上下文窗口较小，且无法记住之前的消息。
   - 成员引用道：*是的，你必须不断提醒它你在问什么*。
- **成员们对新模型 Qwen 感到兴奋**：成员们表示 [**Qwen**](https://chat.qwen.ai/) 模型在推理方面优于 **Deepseek R1**，拥有 1M 上下文窗口，并将被 Perplexity 用于深度研究。
   - 成员们补充说 **Qwen** 也是免费的，而 Deepseek 的服务器经常繁忙。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1379536608696602775)** (2 条消息): 

> `working app, smuggled north korean smartphone` 


- **创建一个可运行的应用**：有一个与创建可运行应用相关的 [Perplexity 搜索结果](https://www.perplexity.ai/search/create-a-working-app-using-the-9B6cBgPATvmgfo6mwd07sg?0=c)。
- **走私的朝鲜智能手机**：有一个关于走私朝鲜智能手机的 [Perplexity 页面](https://www.perplexity.ai/page/smuggled-north-korean-smartpho-NgjIJo_RTW6Dx8TYfGWpZg)。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1379572349330002011)** (19 messages🔥): 

> `Academic Filter 反馈, 带有 PMC 的 Sonar Reasoning Pro API, NCBI 速率限制 (Rate Limiting), Firecrawl 代理` 


- **Academic Filter 获得强烈反馈**: 一名成员对新的 **Academic Filter** 模式提供了 [反馈](https://discord.com/channels/974519860452964443/1161802929053909012/1379816311635685406)，指出其具有*强大的综合能力*、*高质量的来源*以及*良好的科学语调*。
   - 改进建议包括处理**离题来源**和**过时的 2005 年来源**，并建议增加 **reranking 机制**以及为每个来源提供更清晰的 **snippets**。
- **Sonar API 在访问 PMC 时遇到困难**: 有用户报告在通过 **Sonar Reasoning Pro API** 使用 `search domain filter` 访问 **PMC** 时出现 [间歇性问题](https://discord.com/channels/974519860452964443/1161802929053909012/1379821265606004736)。
   - 他们推测该问题是由 **NCBI 的滥用保护机制**引起的。
- **速率限制限制了 NCBI 的访问**: 成员们讨论了 [NCBI 的速率限制政策](https://discord.com/channels/974519860452964443/1161802929053909012/1379821265606004736)，该政策限制用户*每秒不超过三次 URL 请求*，并建议尝试在非高峰时段访问。
   - 一位成员指出，**Perplexity** 可能是通过自己的服务器发出请求，因此速率限制上限会集体影响所有用户。
- **关于 Sonar 和 Firecrawl 代理的讨论**: 一名成员建议在使用 **Sonar API** 时，将 [使用 **Firecrawl 的搜索功能**](https://discord.com/channels/974519860452964443/1161802929053909012/1379821265606004736) 配合代理作为绕过 **NCBI 速率限制**的权宜之计。
   - 另一位用户承认这种方案很有吸引力，但在 **Sonar Reasoning Pro** 正常工作时，还是更倾向于它的简洁性。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1379541129816440932)** (1468 messages🔥🔥🔥): 

> `Gemini 2.5 Pro 发布, Google 的 Kingfall 模型, OpenAI 的 o3 Pro 发布, 模型性能对比 (Gemini, Claude, Grok, OpenAI), AI 硬件与算力考量` 


- **Gemini 2.5 Pro GA 在即，Goldmane 现身！**: 讨论围绕即将发布的 **Gemini 2.5 Pro** 展开，其中 "Goldmane" 是一个关键版本，在 Web 开发基准测试 Aider 上得分 **86%**，详见 [此处](https://aider.chat)。
   - 一名成员指出，*"diff-fenced"* 编辑格式主要用于 **Gemini 模型** ([Aider 文档](https://aider.chat/docs/more/edit-formats.html#diff-fenced))。
- **Kingfall：Google 意外发布的模型引发热议**: 一个名为 **Kingfall** 的模型（被认为是 **Gemini** 的内部模型）曾在 AI Studio 上短暂可用，引发了对其能力以及它是否为 **DeepThink** 的猜测。
   - 成员们注意到它拥有 **65k** 的上下文窗口，这一限制让部分人认为它不是 Pro 模型，而另一些人则指出其“机密”名称意味着有人要被解雇了。
- **OpenAI 的 o3 Pro 仍然失踪？**: **OpenAI o3 Pro** 的潜在发布备受期待，但发布日期仍不确定。已获得访问权限的人给出的初步印象反应平平，一名成员表示：*"我已经有了，它很烂"*。
   - 担忧主要集中在 **o3 Pro** 在生成代码方面的局限性，其上限为 **500 LOC**，而其前代模型可以在不遗漏的情况下生成 **2000 LOC**。
- **模型大对决：Gemini 2.5 Pro vs 竞品**: 各种模型之间进行了对比，包括 **Gemini 2.5 Pro**、**Claude Opus**、**Grok** 和 **OpenAI o3**，重点关注编程熟练度、推理能力和综合性能，其中 Grok 3 因其超长的“思考模式”而受到关注。
   - 一名用户测试了 Kingfall 的**空间推理**能力，给它布置了一个 [geoguessr 任务](https://www.geoguessr.com)，结果令人惊叹。
- **免费 API 使用被削减，Google 收紧预算**: **Gemini 2.5 Pro** 突然取消免费 API 访问引发了失望，特别是对于长文本内容生成等用例。
   - 一名用户开玩笑说 Gemini 现在需要信用卡详情，并注册有效的支付信息才能获得 *$300 的免费额度*。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1379536903384203424)** (547 条消息🔥🔥🔥): 

> `Cursor Pro 'unauthorized' 错误, Claude 4 Sonnet 限制, CursorRIPER 框架, Claude Code vs Cursor, 手动更新 vs 自动更新` 


- **Cursor Pro 用户在使用 GPT-4.1 时遇到授权问题**：多位用户报告在升级到 Cursor Pro 后，尝试访问 **GPT-4.1** 时遇到 *'User is unauthorized'* 错误，即使提供了账户详情也无法解决。
   - 受影响的用户分享了请求 ID 和账户邮箱，寻求 Cursor 团队协助激活 **GPT-4 访问权限**。
- **Claude 4 Sonnet 上下文窗口限制引发规避策略**：用户报告 Claude 4 Sonnet 有限的上下文窗口会中断对话，导致需要重启或丢失上下文。一位用户建议使用 *'continue where you left off'* 的提示词技巧，尽管这会消耗额外的请求额度。
   - 一位用户推测 **Claude 4** 拥有 *'rolling context'*（滚动上下文），在整个聊天过程中会考虑关键因素。
- **CursorRIPER 框架成为项目工作流催化剂**：用户讨论了 **CursorRIPER 框架**，这是一种通过规则和记忆来引导 Agent 行为的方法，有助于保持上下文并专注于项目。
   - 它维护一个 **tech context file**（技术上下文文件），有助于防止使用过时的模块，并可在重大编辑后更新，以确保 Agent 了解项目的当前状态。
- **Claude Code 表现惊人**：成员们讨论了 **Claude Code** 的兴起，至少有一人宣称它在某些任务上优于 Cursor，并根据近期经验称赞其 *'极其聪明'* 的编码能力。
   - 一位用户声称使用 **Claude Code** 成功实现了一次性重构大型复杂代码库，并通过了数千个测试且无错误。
- **用户讨论学生折扣价值及欺诈担忧**：成员们对利用教育邮箱欺诈和廉价销售以获取 **Cursor 学生折扣** 表示担忧。
   - 一些人建议将学生折扣限制在特定国家以作为防止滥用的措施，一位成员评论道：*"只要学生来自最富有的国家，就能免费使用 Cursor，这是一个伟大的营销策略"*。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1379564140061917368)** (16 条消息🔥): 

> `Background Agents 卡死, Cursor 版本升级, Background Agent 研究项目, Slackbot 安装, Repo 连接问题` 


- **Background Agents 闹脾气**：部分用户在尝试启动 background agents 时[遇到卡死现象](https://cdn.discordapp.com/attachments/1367213641027551352/1379872738981707846/image.png?ex=6841d221&is=684080a1&hm=dfc7fb0889f48ccc5ec0ac8d979c2f52e0783fd5ac3d7d8d958381aed60d2ef4&)。
- **Agent 热潮需要升级 Cursor**：要使用 background agents，用户必须[升级到 **Cursor 1.0.0** 或更高版本](https://cdn.discordapp.com/attachments/1367213641027551352/1379915546358972516/image.png?ex=6841f9ff&is=6840a87f&hm=a7f9715059a078fa8a2766f75e21382d653167f515439c17f5dcdcef73c2b94c&)。
   - 一位用户指出该功能非常酷，在 **完整的科研项目** 中取得了 *令人印象深刻的结果*。
- **Slackbot 仍然失踪**：用户想知道如何安装 **1.0 发布公告** 中展示的新 [**Slackbot**](https://slack.com)。
   - 截至本文撰写时，该 **Slackbot** 尚无法找到。
- **Cursor 需要 “重新记住” Repo 名称**：一位用户在更改 Repo 名称后遇到了连接问题，因为 **Cursor** 仍尝试使用其 *之前的名称* 进行连接。
   - 重新安装 **Cursor GitHub app** 未能解决问题；不确定是否有缓存需要清理。
- **容器难题**：一位用户在激活 **Background Agent 模式** 时遇到错误，具体表现为无法创建默认环境。
   - 另一位用户建议重建你的 **base background container snapshot**（基础后台容器快照）。


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1379925650718330962)** (1 条消息): 

> `Cursor 1.0 发布, 代码审查改进, 后台任务管理` 


- **Cursor 1.0 现已发布！**：最新的 **Cursor 1.0** 版本包含增强的 **代码审查能力**、改进的 **错误追踪** 以及处理多个 **后台任务** 的能力。
   - 详见 [官方更新日志](https://www.cursor.com/changelog) 以获取所有更新的详细概览。
- **代码审查获得提升**：**Cursor** 现在可以审查你的代码并记住其中的错误。
   - 这旨在提供更具上下文感知能力的建议，并捕捉重复出现的错误。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1379541146094669914)** (391 条消息🔥🔥): 

> `O3 Pro, GPT-5 Release, ChatGPT hallucination, Sora for everyone, ChatGPT Connectors` 


- **o3 Pro 的到来引发讨论**：成员们对 **o3 Pro** 的发布进行了推测，一些人期待它的到来，而另一些人则因 **Sam Altman** 之前的延迟和未兑现的公告而保持怀疑。
- **GPT-5 临近**：一些成员推测 **GPT-5** 可能即将发布，而另一些人则认为这将是一个 **AGI** 版本的发布。
   - 一位成员表示 *“不会有 o3 pro。他们将发布 chatgpt5。”*
- **OpenAI 员工预热新功能**：OpenAI 员工预热了 **Teams** 和 **Enterprise** 计划的重大更新，引发了用户的期待，其中内部知识库功能的发布是一个热门话题。
   - 一位用户提到，一名员工说 **“明天对于那些让我日以继夜痴迷的用户来说将是大日子！”**。
- **Connectors**：Connectors 是新的内部知识库功能，允许用户使用推理模型对内部数据源进行搜索。
   - 一位用户表示：*“他们刚刚发布了一个更新，今天的公告对 **teams 用户** 非常有利，原因是我们现在可以使用任何推理模型来搜索内部资源，而直到现在只有 4o 模型可以使用，我现在很高兴 🙂”*
- **是否为 GPT-4o**：成员们争论 **GPT-4.1** 是否与 **GPT-4o** 有关，一些人认为它是基于更多数据训练的扩展版本，而另一些人则认为由于多模态能力的差异，它们是不同的模型。
   - **GPT-4o** 的视觉能力是 **SOTA** 级别的，并被用于 **API** 集成，能产生更好的结果。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1379566447344222239)** (11 条消息🔥): 

> `Hallucination rates, Bitbucket and Plastic Svn support, OpenAI TTS Pricing Discrepancies, GlazeGPT's Return` 


- **ChatGPT 幻觉率统计**：一位成员询问了 **ChatGPT** 幻觉的统计数据，并指出根据任务和上下文的不同，幻觉率在 **1-50%** 之间波动。
- **Bitbucket 和 Plastic Svn 支持状态**：一位成员询问 **Codex** 是否支持 **Bitbucket** 或 **Plastic Svn**。
- **OpenAI TTS 价格差异引发讨论**：一位成员质疑为什么 **gpt-4o-mini-tts** 的收费比 **tts-1** 高出约 4 倍，尽管标价分别为每 100 万字符 **$12** 和 **$15**；另一位成员建议查看 [OpenAI 社区论坛](https://community.openai.com/t/new-tts-api-pricing-and-gotchas/1150616) 以获取深入见解。
- **GlazeGPT 回归**：一位成员开玩笑说 **GlazeGPT** 回来了，观察到它在 5-6 条消息后就会退化为表情符号刷屏。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1379554999935438929)** (8 条消息🔥): 

> `Agent design for Elasticsearch queries, Model finetuning vs prompt engineering, Mermaid sequence diagrams in prompts, Elasticsearch sorting issues` 


- **旨在查询 Elasticsearch 的 Agent 工作流**：一位成员正在使用 **open ai gpt-41-mini** 构建一个 Agent，根据人类查询创建 **Elasticsearch DSL queries** 以进行图表绘制。最初使用单个 Agent，但后来将其拆分为多个 Agent，分别负责识别索引名称、获取映射、生成查询和提取数据，如[此附图](https://cdn.discordapp.com/attachments/1046317269069864970/1379745858639237140/agent_ES.png?ex=684204b7&is=6840b337&hm=eadc26c0018f544fedd3a9e8e6407dcf5f02301750f66ef5732067f94b94beff&)所示。
- **微调 vs Prompt Engineering?**：在寻求改进 Agent 响应的建议时，一位成员建议对模型进行微调和/或使用 **RAG**，而不仅仅依赖于 Prompt Engineering。
   - 另一位成员询问他们是否尝试过在 Prompt 中包含 **Mermaid 序列图**。
- **Agent 响应一致性的挑战**：一位成员一直难以让其 Agent 提供令人满意且一致的响应，即使将 Temperature 设置在 **0** 左右。
   - 另一位成员指出当前设置中至少存在 *7 个* 问题，其中最大的问题是在 **Elasticsearch** 中对所有内容（甚至包括索引）进行排序。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1379554999935438929)** (8 条消息🔥): 

> `Elasticsearch DSL Queries, RAG Implementation, OpenAI model discussion etiquette` 


- **Elasticsearch DSL 查询生成的讨论**：一位成员正在使用 **gpt-41-mini** 构建一个 Agent，根据人类查询创建 **Elasticsearch DSL queries**，以便使用官方的 Elasticsearch mcp server 绘制图表，但结果不尽如人意。
   - Agentic flow 涉及多个 Agent，用于识别索引名称、获取索引映射、生成 Elasticsearch 查询以及提取数据，但该成员报告称，即使 Temperature 接近 0，响应也不一致，详见[此图表](https://cdn.discordapp.com/attachments/1046317269069864970/1379745858639237140/agent_ES.png)。
- **针对 Elasticsearch 查询提出的 RAG 实现方案**：一位成员建议通过微调模型或实施 **RAG (Retrieval-Augmented Generation)** 作为提高 **Elasticsearch DSL query generation** 质量的潜在解决方案。
   - 另一位成员询问用户是否尝试在 Prompt 中包含 **mermaid sequence diagram** 以引导模型。
- **讨论非 OpenAI 模型的 Discord 频道**：根据 <#1107255707314704505>，一位成员将非 OpenAI 模型的讨论引导至 <#998381918976479273> 频道。
   - 他们澄清说，当前频道可以讨论 Prompt 技术和模型能力，但特定的非 OpenAI 模型应在指定频道讨论。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1379543962351829044)** (113 条消息🔥🔥): 

> `DeepSeek R1 0528 speed, Qwen 4B vs Gemma 4B, Vision support for Mistral-Small-3.1-24B-Instruct-2503-GGUF, Multi-GPU support, Fastest lib for production inference` 


- **DeepSeek R1 0528 运行变慢了？**：一位用户报告称，**DeepSeek R1 0528** 在 Mac Studio 上的运行速度比 **R1** 慢，速度约为 **12.8 t/s**，而后者为 **18.7-19 t/s**，但其他人认为除非使用了不同的量化格式，否则两者速度*应该相同*。
   - 动态量化（Dynamic quantization）的表现也可能不同，从而影响模型速度。
- **Qwen 还是 Gemma，这是个问题！**：一位用户指出 **Qwen 4B** 的泛化能力不如 **Gemma 4B**，暗示了两者在泛化能力上的潜在差异。
   - 该用户未进一步阐述这种差异的具体表现。
- **Unsloth Vision 功能需要 Llama.cpp**：用户寻求关于带有 Vision 功能的 **unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF** 推理指导，[并被推荐使用 *llama.cpp*](https://github.com/ggerganov/llama.cpp)。
   - 文中提供了克隆仓库、创建构建、启用 CUDA 以及构建 *llama-cli* 的步骤，随后即可配合 Prompt 和图像使用。
- **Multi-GPU 支持即将推出™️**：一位用户询问了 **multi-GPU support** 的可用性及其路线图，并获知它*已经可以配合 accelerate 使用*，且预计在 7 月初推出*更好的版本*。
   - 由于目前的这种支持属于*非官方*性质，因此没有官方示例，但如果熟悉 accelerate 的工作原理，就可以使用它。
- **生产级 CPU 推理的最快库**：在讨论生产级推理的最快库时，有人建议对于单用户 CPU 推理，基于 [llama.cpp](https://github.com/ggerganov/llama.cpp) 的工具可能比较合适，而对于更严肃的 CPU 部署，[vLLM](https://github.com/vllm-project/vllm) 或 [ktransformers](https://github.com/ktransformers/ktransformers) 可能更合适。
   - 此外，**v0 engine** 也在处理这方面的工作，但它在 **v1** 中并不存在。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1379637675669196821)** (11 messages🔥): 

> `Qwen3-32B 上的 GRPO 训练，AI 工程师成本，基础微调数据集，HuggingFace 导航，QLORA 指令微调` 


- **GRPO 训练寻求 Qwen3-32B 扩展支持**：一名成员寻求帮助，希望将已完成调试并运行在 **7B 模型**上的 **GRPO 训练代码**扩展到 **Qwen3-32B**，预算为 **30 美元**，工期 **2-3 天**。
   - 另一名成员调侃道，考虑到典型的 **AI 工程师成本**，这个预算可能少了几个零。
- **寻找微调数据集**：一名成员就**基础微调数据集**寻求建议，以增强基础模型或预训练模型的功能，同时也感叹 **Hugging Face** 导航的困难。
   - 另一名成员建议在 Hugging Face 上使用**过滤和排序**功能，例如[这个例子](https://huggingface.co/datasets?modality=modality:text&task_categories=task_categories:question-answering&sort=likes)。
- **尝试使用 QLORA 进行指令微调**：一名成员分享了使用 **QLORA** 进行指令微调的经验，指出模型可以回答问题，但在结束响应方面存在困难。
   - 在后续讨论中，他们分享了一个宏大的项目：在 150 万个论坛帖子、古典文学和互联网数据集上预训练并微调 **Gemma 3** 模型，旨在复制 IT 模型的功能，同时避免对齐训练。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1379558604658708490)** (139 messages🔥🔥): 

> `GRP trainer 推理，序列长度最大长度，Gemma 3 模型 unsloth，Unsloth 信息日志，Deepthink R2 模型` 


- **用户寻求在 GRPOTrainer 上进行推理**：一名用户尝试使用 **GRPOTrainer**，结合 **vllm** 和 **model.fast_generate**，利用最近一步训练的权重对模型进行推理。
   - 用户咨询是否可以在奖励函数（reward function）期间，使用之前传递给 GRPOTrainer 的全局模型执行此类推理。
- **序列长度混淆引发的故障排除**：一名用户在微调 **llama instruct** 进行 JSON 提取时，发现 `dataset['text'][7]` 与 `tokenizer.decode(trainer.train_dataset[7]["input_ids"])` 之间存在差异。
   - 官方澄清 **max_seq_length** 对应的是 token ID 的长度，而非字符长度。建议用户在 **SFTConfig** 中将 `max_length` 设置为等于 `max_seq_length` 作为临时解决方案，该问题将在下一个 pypi 版本中更新。
- **用户在 Gemma 3 模型中遇到属性错误**：一名用户在本地运行代码时遇到了 `AttributeError: 'Gemma3ModelOutputWithPast' object has no attribute 'loss'`，而同样的代码在 Colab notebook 中可以运行。
   - 该问题归因于不同版本的 **Hugging Face transformers**（本地为 4.52.4，Colab 为 4.51.3），建议使用 `attn_implementation="eager"` 或回退到旧版本的 `unsloth-zoo`。
- **Unsloth INFO 日志记录**：一名用户询问在使用 **vLLM** 进行模型训练时如何关闭 **Unsloth INFO 日志**。
   - 官方澄清 **Unsloth** 使用标准的 Python logging，用户应参考 Python 和 vLLM 文档进行配置，可使用环境变量 `'VLLM_LOGGING_LEVEL'`。
- **针对 BLIP 架构的修复即将到来**：一名用户报告了加载模型时的兼容性问题，可能与模型量化有关，错误信息为：`ValueError: The model was built with the CUDAGraph capture mode enabled, but the current model does not have the same structure.`。
   - 经确认，**BLIP** 的架构有所不同，初始修复中未考虑到这一点，目前正在主动调查修复方案。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1379613675832479854)** (30 messages🔥): 

> `Weightwatcher AI, LLM 分析, VLM 可视化` 


- **Weightwatcher AI 测量的是饱和度而非记忆**：weightwatchers Discord 评论的一名成员指出，他们测量的是饱和度（saturation）而非记忆（memorization），并且除了记忆数据之外，其他因素也可以导致饱和，参考自 [weightwatcher.ai](https://weightwatcher.ai/)。
- **VLM 感兴趣区域可视化**：一名成员询问是否有类似于显著性图（saliency maps）的方法来可视化 VLM 的感兴趣区域（region of interest），另一名成员分享说*你可以可视化哪些多模态 token 正在被关注*。
- **解读“非预期”记忆**：一名成员将“记忆”定义为泛化（generalization）与类似过拟合（他们称之为非预期记忆）的总和。
   - 他们详细阐述道，最终模型对 X 的了解越多，**H(X|(O, Ohat))** 就越低，因此 **memU** 的值就越大。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1379871336096202925)** (8 messages🔥): 

> `GIF Support, Omni-Search, Tool Call Caching, BYOK Flag` 


- ****GIF 盛宴：多模型支持动画****：**OpenAI, Gemini, Anthropic 和 Llama 路由**的图像提示词现在支持 `image/gif`，无需再预先转换动画格式。
- ****提供商页面在 Omni-Search 中即时显示****：用户现在可以按下 `⌘/Ctrl + K`，输入提供商名称，直接跳转到其模型、定价和状态页面。
- ****Tool-Call 加速：Anthropic 获得缓存支持****：Anthropic 现在支持 Tool Call 缓存，从而降低延迟和 Token 使用量。
- ****BYOK 溯源：Usage 标识发布****：在请求中包含 `usage: { include: true }` 现在会返回 `"is_byok": true | false`，以确认是否使用了 **BYOK**（自带密钥）。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1379965234650939443)** (3 messages): 

> `iOS App, TestFlight, OpenRouter, LLM Backend` 


- **iOS 应用通过 TestFlight 集成 OpenRouter**：一名成员计划很快通过 **TestFlight** 分享一款 **iOS 应用**，该应用使用 **OpenRouter** 作为 **LLM 后端**。
   - 该应用使用了**角色卡**，但由于复杂性，成员仍需完成消息格式化工作。
- **更多 iOS 应用细节**：该应用使用角色卡和 **OpenRouter** 作为 LLM 后端，并计划稍后添加更多客户端。
   - 由于复杂性，消息格式化仍在进行中；该应用正准备在 TestFlight 上发布。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1379542272030277782)** (258 messages🔥🔥): 

> `Opus Rate Limits, Chutes Business Model, Nous Training, OpenRouter Batch Inference API, Chutes R1 Quality` 


- **Opus 获得更高的速率限制！**：OpenRouter 现在为 **Opus** 提供更高的速率限制，特别是在将流量路由到 **Anthropic** 模型时。
   - 鉴于所需的 GPU 资源，该公告引发了关于 **Chutes** 经济模式的疑问，有人猜测其资金可能来自“凭空产生的加密货币”。
- **Nous 分布式训练遭遇瓶颈**：**Nous** 正尝试使用 **416 个 H100** 进行分布式训练一个 SOTA 模型，但项目进展缓慢。
   - 按照目前的速度，训练预计要持续到明年。尽管声称在减少 GPU 间带宽需求方面取得了突破（仅利用了约 300mbps 的 GPU 间带宽），但仍引发了质疑。
- **探索 OpenRouter API 调用策略！**：成员们讨论了如何通过 OpenRouter 向 **LLM** 发送 **10 万次调用**，并优先考虑吞吐量而非延迟，建议检查提供商折扣并向 OpenRouter 充值。
   - 分享了 Modal 的 **LLM Almanac Advisor** 链接。
- **OpenRouter 每日免费消息上限说明**：OpenRouter 的每日免费消息限制为 **50 次请求**，对于充值至少 **$10** 的用户，该限制增加到 **1,000 次请求**。
   - 这些限制适用于所有免费模型，并在每日 **UTC** 时间重置。
- **Mistral 发布 Code Agent！**：**Mistral** 发布了自己的编程 Agent，引发了关于 Mistral 模型与 **Deepseek** 和 **Qwen** 等其他模型质量对比的讨论。
   - 一名成员认为 **Codestral** 模型更胜一筹。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1379537615619100867)** (12 messages🔥): 

> `GPU Mode Merchandising, GPU Mascot Creation, AI-generated Mascot Design, Copyright safe mascot` 


- **GPU Mode 周边创意引发讨论**：一名成员建议为 **GPU Mode** 制作周边商品，例如印有“超级赛亚人 GPU”的 T 恤。
   - 另一名成员指出了[版权问题](https://en.wikipedia.org/wiki/Copyright_law)，并建议创作一个原创吉祥物。
- **AI 尝试设计 GPU 吉祥物**：一名成员使用 **ChatGPT** 生成了一个潜在的 **GPU Mode** 吉祥物图像，并分享了提示词细节。
   - 提示词包括基于“编程 GPU”制作图像，通过不模仿悟空来避免版权问题，并手持两个 GPU，最终生成了[这张图片](https://cdn.discordapp.com/attachments/1189498205101109300/1379733892583657502/2ce4ee02-d0b7-4f9b-beb1-fd7ece71d553.png?ex=6841f992&is=6840a812&hm=d35647055583e58b1feab17755a47e75e96f5c6d9e7fa28e549e616eb066784b&)。
- **AI 生成的吉祥物反响平平**：在使用 **ChatGPT** 生成图像后，成员们认为它不够简洁，无法作为 Logo 或吉祥物。
   - 一名成员表示：*“不能说我很喜欢，哈哈，它需要更简洁一些，介于 Logo 和吉祥物之间。”*


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1379548319503880253)** (8 messages🔥): 

> `__syncthreads vs bar.sync, mbarrier details, cuda::pipeline usage, Producer/consumer pipeline synchronization` 


- **`__syncthreads` 通过 `bar.sync` 实现**: `__syncthreads()` 基本上就是 `bar.sync`/`barrier.sync.aligned`，而 `sync(cooperative_groups::this_thread_block())` 为同步不同分支中的线程提供 `barrier.sync`（仅限 [Volta](https://developer.nvidia.com/blog/volta-new-programmability-features/) 及更新版本）。
- **`mbarrier` 状态揭秘**: 用于拆分 arrive/wait 屏障（barriers）的 PTX 指令被称为 `mbarrier`，随 [Ampere](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) 架构引入，并在 [Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-gpu/) 中增加了更多功能。
   - `mbarrier` 中的 'm' 可能代表 **memory**（内存），因为屏障状态必须显式放入 shared memory，不要将其与作为 fence 的 `membar` 混淆。
- **`cuda::pipeline` 成为正确选择**: 对于生产者/消费者方案，在 [CUDA](https://developer.nvidia.com/cuda-zone) 中使用 libcu++ 的 `cuda::pipeline` 是正确的做法。
   - 讨论中还提到了使用 `bar` 实现简单的生产者/消费者方案，详见 [文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)。
- **拆分 Arrive/Wait 屏障方案浮出水面**: 查看 [CUDA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#spatial-partitioning-also-known-as-warp-specialization) 中的 **8.26** 节，了解从 **Ampere** 开始提供的拆分 arrive/wait 屏障。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1379556502150582422)** (6 messages): 

> `CUPTI Profiling Overhead, Torch Dynamo Recompiles, CUDA Command Buffer Bottleneck` 


- **命令缓冲区瓶颈导致高开销**: 一位成员指出 **CUPTI** profiling 中可能存在高开销，暗示由于 GPU 的命令缓冲区已满可能导致瓶颈，并引用了 [CUpti_ActivityOverheadCommandBufferFullData 文档](https://docs.nvidia.com/cupti/api/structCUpti__ActivityOverheadCommandBufferFullData.html#structcupti__activityoverheadcommandbufferfulldata)。
   - 他们建议使用 timeline 视图以获取更可靠的数据，并提醒注意 profiling 本身带来的开销。
- **Python 常量触发 Dynamo 重新编译**: 一位成员指出，在 Torch Dynamo 中直接使用 Python 常量会触发重新编译（recompiles），如日志 `___as_tensor(alpha).item() == 0.5` 所示。
   - 他们澄清说，将常量包装在 `Tensor` 中可以避免此问题，而 C++ 接口会自动处理转换。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1379894505695612979)** (2 messages): 

> `PMPP Lectures, ECE408 Lectures` 


- **用户询问 PMPP 讲座推荐**: 一位用户询问关于 YouTube 上 **PMPP 讲座** 的特定系列推荐。
   - 另一位用户建议从 **ECE408 讲座** 开始，同时也提到了视频的音频质量较差。
- **ECE408 讲座的音频质量问题**: 一位用户提到他们尝试观看讲座，但音频质量很差。
   - 这些讲座属于 **ECE408**，如果你想学习 PMPP，可以从这里开始。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1379767280723824653)** (3 messages): 

> `MPS Kernels, vLLM, VL Models` 


- **MPS 标志语义变更**: 一位成员表示，某个标志的语义最近发生了变化，现在仅适用于 **MPS kernels**。
   - 预计会有一个 PR 来解决此更改并修正该标志的行为。
- **vLLM 计划支持 VL 模型**: 有计划在 **vLLM** 中支持加载 **VL models**。
   - 目前，在 **vLLM** 中加载序列化的 ao 模型适用于所有层都经过量化的语言模型，但当 vision model 未量化时，在 **VL models** 上会报错。
- **vLLM VL 模型修复已发布**: 发布了针对 **vLLM** 和 **VL Models** 的修复。
   - 一位成员发布了 [GitHub 上的修复链接](https://github.com/vllm-project/vllm/pull/19147)。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1379535446035398877)** (2 messages): 

> `TiKZ, JAX ML 动画` 


- **TiKZ 可能为 JAX ML 书籍制作动画**：一位成员询问如何制作类似于 [JAX ML Scaling Book](https://jax-ml.github.io/scaling-book/#high-level-outline) 中的动画。
   - 另一位成员建议使用 **TiKZ**，并指出这些动画可能是由多张图像融合而成的 GIF。
- **JAX ML 书籍中的动画是 GIF**：一位成员指出 [JAX ML Scaling Book](https://jax-ml.github.io/scaling-book/#high-level-outline) 中的动画很可能是 GIF。
   - 这些 GIF 可能是使用 **TiKZ** 之类的工具创建的。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1379569750161752136)** (29 messages🔥): 

> `MI300X 内存访问周期，MI300X 上的 rocprof 和 L2CacheHit，rocprof-compute 和 omniprof 区域设置 (locale) 错误，Kernel 分析中的 MFMA 利用率，Root 用户 sudo 错误` 


- **MI300X 内存访问周期推测**：一位成员询问 **MI300X** 上的 **DS_READ2_B64**、**DS_READ2ST64_B64** 和 **DS_READ_B128** 指令是否在相同的周期数内执行，或者 **DS_READ2_B64** 是否比 **DS_READ_B128** 慢。
   - 该用户猜测 AMD 的操作通常被分解为 dword（**32 bits**）。
- **在 MI300X 上使用 rocprof 时 L2CacheHit 指标出现问题**：一位成员报告了在 **MI300X** 上使用 `rocprof` 读取 kernel 的 **L2CacheHit** 时遇到的问题，指出虽然该指标在 [ROCm documentation](https://github.com/ROCm/ROCm/blob/develop/docs/conceptual/gpu-arch/mi300-mi200-performance-counters.rst) 中被列为可用，但 `rocprof` 返回错误，提示 **gfx942** 不支持该指标。
   - 他们还尝试了 `rocprofv2`，它给出了更清晰的错误信息，并提到 `rocprof-compute` 可能是一个可行的替代方案，以及使用 `rocprof-compute analyze` 通过 compute viewer 进行详细分析。
- **rocprof-compute 和 omniprof 区域设置 (Locale) 错误**：一位成员在从源码编译并尝试安装 `rocprof-compute` 和 `omniprof` 时遇到了与区域设置相关的错误，具体是需要 **en_US.UTF-8** 区域设置的错误。
   - 由于权限限制，他们无法解决区域设置问题。
- **MFMA 利用率见解**：一位成员正在分析 **FetchSize**、**WriteSize**、**MfmaUtil** 和 **SQ_LDS_BANK_CONFLICT**。
   - 目前 **MfmaUtil** 为 **1.9**，如果用虚拟数据加载 smem，**MfmaUtil** 可以达到 **3.49**；该用户正试图通过了解 L2 缓存命中率来更好地理解这一点。
- **Ubuntu 22.04 上的 Root 用户 sudo 悖论**：一位成员遇到了一个问题：**Ubuntu 22.04.5 LTS** 系统上的 root 用户由于不在 sudoers 文件中而无法使用 `sudo`。
   - 鉴于用户已经以 root 身份登录，这种悖论情况引起了其他成员的好奇。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1379887863331688588)** (1 messages): 

> `Hopper GPU, TMA, CUDA, Mojo, NVPTX` 


- **在不使用 CUDA 的情况下在 Mojo 中实现 TMA**：一篇新的博客文章演示了如何在 **Mojo** 中实现一个简单的 **基于 TMA 的 kernel**，并逐行讲解了该 kernel ([blogpost](https://veitner.bearblog.dev/use-tma-without-cuda/))。
   - 这篇文章与之前在 **CUDA** 中使用 TMA 实现快速转置 kernel 的工作形成了对比。
- **深入探讨使用 LLVM 和 NVIDIA PTX 的 TMA**：为了深入了解 TMA，作者建议查看 **Mojo 标准库的 TMA 实现** ([Mojo standard library](https://github.com/modular/modular/tree/main/mojo/stdlib/stdlib))，以及 **LLVM NVPTX** ([LLVM NVPTX docs](https://llvm.org/docs/NVPTXUsage.html)) 和 **NVIDIA PTX 文档** ([PTX docs](https://docs.nvidia.com/cuda/pdf/ptx_isa_8.5.pdf)) 的相关部分。


  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1379560171898146846)** (7 messages): 

> `GLSL Fragment Shaders 代码补全基准测试, 多设备 Kernel Codegen, 架构特性演进, 分析 Nvidia ISA` 


- **低资源语言训练演讲**: 一位成员提到了在 ICSE 上关于**低资源语言训练**的演讲，特别是针对微控制器的底层内容：[https://arxiv.org/abs/2410.22159](https://arxiv.org/abs/2410.22159)。
   - 该论文结合使用了 **DPO** 与 **LLM judge**，以及**编译器和合成数据**，并取得了有趣的结果。
- **Codegen 考量讨论**: 一位成员分享了关于 Codegen 的想法，包括**多设备 Kernel**、识别硬件代际/互连带宽 (BW)/系统配置，以及识别并插入正确的集合通信 (collectives)。
   - 他们还讨论了模型对**跨硬件/软件版本的架构特性演进**进行推理的能力，并识别与旧版本相比，使用新变体是否总是更好。
- **NVIDIA ISA 开源考量讨论**: 讨论了开源该项目的可能性，考虑到**分析 Nvidia ISA** 需要签署 NDA。
   - 一位成员提到 [Nvidia 已经公开了他们的一些旧计算内容（用于物理加速）](https://github.com/NVIDIA-Omniverse/PhysX)，但这可能不太适用于现代硬件。
- **AMD 或 Nvidia ISA 信息依然可用**: 一位成员指出 **PTX ISA 绝对是公开的**，而且无论如何，该项目只是以 Kernel 的形式收集数据并训练一个开源模型，并表示*任何单一 ISA 的内部细节都是无关紧要的*。
   - 另一位成员补充说，**AMD 或 Nvidia** 的此类信息是可获取的，只是目前还没有针对 GPU 的 **uops.info**。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1379863763934515341)** (2 messages): 

> `ThunderKittens, LayerNorm kernel, 维度处理, 序列长度整除性, 生产者/消费者模型` 


- **ThunderKittens 维度限制疑问**: 一位用户询问了 **ThunderKittens 的维度处理**，注意到像 **LayerNorm kernel** 这样的实现具有硬编码的隐藏维度 (**D=1024**)，并强制要求序列长度能被 **16** 整除。
   - 该用户询问 **ThunderKittens** 是否支持列维度未对齐到这些固定大小的情况，以及对于具有不同隐藏维度或非 16 倍数序列长度的模型，推荐的处理方法是什么。
- **探索 ThunderKittens 架构的灵活性**: 一位用户表示有兴趣在 **ThunderKittens** 之上构建比生产者/消费者模型更灵活的东西，例如像 **B200 warp specialization** 示例中的多个步骤。
   - 该用户表现出极大的热情去了解 **ThunderKittens** 灵活架构的使用案例。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/)** (1 messages): 

jacklee0897: <@299045948146057218> Hackathon 在哪里举办？
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1379666279002017843)** (1 messages): 

> `H100 速度, 排行榜提交` 


- **H100 在排行榜上运行迅速**: 一位用户向排行榜提交了一个在 **H100** 上运行成功的记录，耗时 `71.2 µs`。
- **带 ID 的排行榜提交**: 此提交的 ID 为 `31336`，提交至 `histogram` 排行榜。


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1379898491563802768)** (1 messages): 

> `Open 2025 课程, 课程统计` 


- **分享 Open 2025 统计数据**: 一位成员分享了来自 [Aalto University](https://ppc.cs.aalto.fi/stat/open2025/) 的 **Open 2025** 课程实例的一些统计数据。
   - 这些统计数据不是实时的，但会偶尔更新，特别是在截止日期临近时。
- **宣布截止日期更新**: 课程讲师提到，随着截止日期的临近，他们会偶尔更新课程统计数据。
   - 这意味着学生应关注提供的[统计页面](https://ppc.cs.aalto.fi/stat/open2025/)，以了解课程进展情况。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1379621851550187520)** (9 messages🔥): 

> `Factorio Learning Environment (FLE) Configuration, Decoupling FLE from Python, FLE Project Structure and Roadmap, Dockerizing Factorio with FLE Mod` 


- ****Configuring FLE** 实验配置变得简单**：一名成员正在为 **Factorio** 实验开发配置方案，旨在提供一种简便的方法来配置实验，包括定义 **instances, teams, goal, planners** 和 **agents**。
   - 有建议指出，配置应采用 **Python 中的 builder pattern** 而非 JSON 配置文件，以增强易用性。
- ****Decoupling FLE** 以实现更广泛的集成**：一名成员正致力于将 FLE 与 Python 解耦，计划创建一个带有 **FLE mod 的版本化 Docker image**，以便通过 **JSON API** 与其他编程语言集成。
   - 目标是简化环境的启动和运行，允许用户直接 pull 该 Docker image 并使用他们首选的 FLE 集成方式。
- ****FLE Project Structure** 以提升影响力**：讨论涉及明确 FLE 的愿景，重点关注用户应如何与其交互，以及支持该愿景所需的项目结构。
   - 建议的结构包括一个**官方 Factorio 环境**、一个**官方 FLE 集成**（Python package）以及**官方 FLE benchmarking**（eval/ 目录）。
- ****Charting a Course**：FLE 路线图**：围绕制定一个为期 **3-4 个月的路线图**进行了讨论并达成一致，旨在使 FLE 更易上手且更具影响力。
   - 该路线图旨在明确项目的方向和结构，鼓励更广泛的贡献和关注。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1379591210095611906)** (29 messages🔥): 

> `Double Buffering, FP8 Solution Writeup, Cache Line Optimization, MI300 coalescing, GPU Mode solutions` 


- **Snektron 公开 AMD FP8 Kernel**：Snektron 分享了他的 AMD **FP8** 矩阵乘法 Kernel 解决方案，可在 [GitHub](https://github.com/Snektron/gpumode-amd-fp8-mm/blob/main/solution.hip) 上获取。
   - 他受到了另一位用户的启发，并准备了一份[关于其 FP8 方案实现的 writeup](https://akashkarnatak.github.io/amd-challenge/)。
- **分析 AMD 的 Coalescing**：一位用户在 GitHub 上分享了他的解决方案：[swz4x4-full-db-16x16.hip](https://github.com/AkashKarnatak/amd-challenge/blob/master/swz4x4-full-db-16x16.hip) 和 [swz4x4-full-db-streamk-16x16.hip](https://github.com/AkashKarnatak/amd-challenge/blob/master/swz4x4-full-db-streamk-16x16.hip)。
   - 会议指出，在 **MI300** 和其他 AMD 硬件上，GPU 的 L2 cache 会收集内存请求并请求整个 cache lines，这可能会提高性能。
- **性能调优深度解析**：一位用户花费了大量时间调优他们的方案，包括尝试在列方向上使用 *global_load_dword* 进行 **4x4 DPP transpose**，这最初损害了 gmem coalescing。
   - 他们手动调优了所有内容，并发现通过在 wave 中重新排列请求，使其处于更高效的布局但仍形成完整的 **L2 cache line**，可以获得最佳性能。
- **发现 Cache Coalescing 率**：一位用户对其 Kernel 进行了 profile，观察到大约 **60% 的 cache coalescing**，这表明通过某些技术，可能达到 **90%** 或更高的比率。
   - 一位用户通过解读附件图片的截图指出，他获得了大约 **86% 的 L2 hit rate**。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1379706192129036370)** (5 条消息): 

> `sdpa and cutlass, CuTe Layout, Blackwell Cutlass Samples, MXFP8 performance on Blackwell, NVFP4 vs BF16 on Blackwell` 


- **SDPA 与 Cutlass 的联系得到澄清**：PyTorch 的 **SDPA** (Scaled Dot-Product Attention) 在底层使用 **Cutlass kernels** 来实现内存高效的 Attention 和 Flash Attention，并利用 **CuTe/Cutlass** 进行性能优化。
   - 一位成员请求澄清此话题，询问了该实现细节。
- **破解 CuTe Layout 约定**：一位成员试图确认他们对 **CuTe layout** 的理解，指出只要坐标约定一致，数组索引可以从左到右或从右到左进行，并进一步参考了[这份 CuTe 讲座幻灯片](https://github.com/NVIDIA/cutlass/blob/b244379d9b15574e07b73b814b88bd2233f0b3ce/media/docs/cpp/cute/01_layout.md#coordinate-mapping)。
   - 他们链接了一个 [CuTe 视频](https://youtu.be/vzUhbDO_0qk?t=3659) 并提供了一个包含 `Thr` 和 `Val` 布局的示例，测试了他们的假设，目标是正确计算物理索引。
- **Blackwell 横扫基准测试**：Blackwell Cutlass 示例 (m,n,k=8192,8192,8192) 的基准测试显示出令人印象深刻的性能：
   - 具体而言，*70_blackwell_fp16_gemm* 达到 **0.99 petaflops/sec**，*70_blackwell_fp8_gemm* 达到 **1.97 petaflops/sec**，*72a_blackwell_nvfp4_bf16_gemm* 达到 **2.69 petaflops/sec**，*72b_blackwell_nvfp4_nvfp4_gemm* 达到 **3.09 petaflops/sec**，而 *72c_blackwell_mixed_mxfp8_bf16_gemm* 达到 **0.23 petaflops/sec**。
- **Blackwell 的 MXFP8 性能调查**：Blackwell 混合 **MXFP8/BF16 kernel** 相对较低的性能 (**0.23 petaflops/sec**) 引发了疑问。
   - 一位成员想知道 **MXFP8 matmuls** 最终是否能达到 **FP8 matmuls** 约 2 petaflop 的性能，以及当前的性能是软件还是硬件限制，并思考 **NVFP4** 是否是比 **BF16** 更快进行 matmul 的最佳选择。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1379816749020479519)** (2 条消息): 

> `Zero To Hero, nanoGPT, nanoR1` 


- **Zero to Hero 在线教科书更新**：在线草案教科书 [Zero to Hero](https://j4orz.ai/zero-to-hero/) 已更新，涵盖了 **"singularity"**（机器学习模型）和 **"systems"**（机器学习框架）两个方面。
   - 该在线教科书借鉴了 Karpathy 的 Zero to Hero 开源精神。
- **nanoGPT 属于预训练**：[nanoGPT](https://github.com/KellerJordan/modded-nanogpt) 和 [beyond-nanogpt](https://github.com/tanishqkumar/beyond-nanogpt) 是预训练的示例。
   - 请持续关注该项目。
- **nanoR1 用于后训练**：[nanoR1](https://github.com/nano-R1/) 是一个后训练项目。
   - 请持续关注该项目。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1379535992075321684)** (73 条消息🔥🔥): 

> `HF 上的 CUDA, ASR 排行榜, MCP 课程进度, IBM 的 Responsible Prompting API, 借鉴区块链的 AI 可靠性模型` 


- **CUDA 硬件困惑**：一名成员询问如何通过 HF 在 **Nvidia/CUDA 硬件**上测试代码，但另一名成员建议使用 **Azure/GitHub/AWS** 进行 DevOps。
   - 该成员表示同意，计划在 **GitHub** 上使用 CI/CD 流水线回归测试进行 CUDA 验证。
- **ASR 排行榜缺少 Gemini**：一名成员寻找包含 **Gemini 模型**的 ASR 排行榜，并指出 [HF Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) 未列出它们，因为 Gemini 具有多模态特性。
   - 他们指出 Gemini 的转录包含音频/情感/发言人信息，而 [elevenlabs scribe](https://elevenlabs.io/scribe) 是目前的 SOTA。
- **MCP 课程预计发布时间未知**：一名成员询问 **MCP 课程**第 3 单元的预计发布时间（ETA）。
   - 另一名成员回应称，即使有大致的时间表，*通常也不太可靠*。
- **IBM 发布 Responsible Prompting API**：一名 IBM 实习生介绍了 **Responsible Prompting API**，这是一个[开源项目](https://github.com/IBM/responsible-prompting-api)，用于推理前的 Prompt 建议，旨在使 LLM 输出更负责任、更准确且更高效。
   - 该系统帮助 Prompt 知识有限的领域专家，可能减少有害输出并节省推理成本，详见[这篇论文](https://arxiv.org/abs/2504.08757)，并在 [HF Spaces](https://huggingface.co/spaces/santanavagner/responsible-prompting-demo) 上进行了演示。
- **区块链能提升 AI 可靠性吗？**：一名成员分享了一篇[概念论文](https://medium.com/@info_65774/consensus-validation-for-llm-outputs-applying-blockchain-inspired-models-to-ai-reliability-f642d7f96f8e)，探讨将**借鉴区块链的共识机制**应用于 LLM 输出，以提高可靠性和可信度。
   - 该论文重点关注 AI Agent、法律/医疗工具以及 AI Alignment 用例。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1379795083351560253)** (1 条消息): 

> `AI 安全基准, LLM Agent, 伦理场景, AI 安全性` 


- **新的 AI 安全基准提出假设场景**：一名成员正在开发一个专注于 **AI 安全和保障**的新基准，使用假设场景，在这些场景中 **LLM Agent** 被赋予虚假工具和有限的行动代理权。
   - 其目的是通过施加压力，观察系统是否会执行不道德的指令、告发用户，或为了生存而做出明确禁止的行为，目前正在寻求反馈和场景贡献。
- **寻求评估 LLM Agent 行为的贡献**：该基准框架包含供 Agent 交互的虚假工具，下一步是设计创意场景并构建良好的评估方法。
   - 开发者欢迎各种想法和贡献，特别是在设计那些能给模型施压并测试其伦理边界的场景方面。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1379817516133519421)** (2 条消息): 

> `CUA MCP Server, trycua` 


- **CUA 提供 MCP Server**：一名成员分享了 GitHub 上 **CUA MCP server** 的链接：[trycua/cua/tree/main/libs/mcp-server](https://github.com/trycua/cua/tree/main/libs/mcp-server)。
- **trycua 的 GitHub 仓库**：[trycua 的 GitHub 仓库](https://github.com/trycua/cua/tree/main/libs/mcp-server) 托管了 CUA MCP server。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1379851025980325960)** (4 条消息): 

> `Prisma toolkit, GitHub Chat, Claude Desktop MCP Playground, Market research basics` 


- **Prisma 工具包获奖并集成至 HF**：用于视觉和视频机械可解释性（mechanistic interpretability）的 **Prisma** 工具包在 CVPR 2025 workshop 上获得了 Oral 演示。它适配了 [Hugging Face 模型](https://huggingface.co/)，并为 CLIP 和 DINO 的每一层托管了 **80+** 开源 **SAEs**，以及 CLIP 转码器（transcoders）。
   - 该发布包含了针对 **100+ 模型**（包括 CLIP、DINO 和视频 Transformers）的电路风格（circuit-style）代码，以及用于训练和评估稀疏编码器（sparse coders）的交互式 notebook，详情见 [Twitter 线程](https://x.com/soniajoseph_/status/1930286144471646252)。
- **GitHub Chat 发布，简化仓库交互**：一款名为 **GitHub Chat** 的新型在线聊天工具允许用户通过将 URL 中的 `github.com` 替换为 `githubchat.ai` 来与任何 GitHub 仓库、文件或 wiki 页面进行交互。
   - 例如，`https://github.com/blueraai/universal-intelligence` 变为 [https://githubchat.ai/blueraai/universal-intelligence](https://githubchat.ai/blueraai/universal-intelligence)，即可获得关于该仓库的即时回答。
- **Claude Desktop MCP Playground 获得 GUI 升级**：**Claude Desktop MCP Playground** 的重大更新引入了用户友好的 GUI，并运行了 **40+** 个运行中的服务器，以简化向 Claude Desktop 添加 MCP 服务器的过程。
   - 邀请开发者测试该 [仓库](https://github.com/seanpoyner/claude-desktop-mcp-playground)，提供反馈并实验 MCP 服务器。
- **市场调研基础**：一位成员分享了他们正在学习的**市场调研基础**和 **ACP 漏斗（ACP Funnel）**。
   - 他们还注意到，*带有图片的垂直长文在 X 上获得的互动最多*。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1379840746068377645)** (1 条消息): 

> `Session Schedule, Summer Break` 


- **读书小组会议因暑假暂停**：读书小组会议在暑假前已结束，[新日程](https://hf.co/reading-group)将在可用时发布。
- **读书小组期待恢复**：参与者们热切期待暑期休整后新日程的公布，期待继续进行引人入胜的讨论。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1379798882493530122)** (1 条消息): 

> `Generative AI, LLMs, Substack, Online Education, LangChain` 


- **针对 GenAI 的新 Substack 创办**：一位成员宣布创办了一个专注于 **Generative AI** 和 **LLMs** 的新 [Substack](https://open.substack.com/pub/samerattrah/p/llms-for-generative-ai-exploration?r=2nuo7w&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)。
   - 简介中讨论了**在线教育**，并将学习短期课程视为完整学习旅程的延续，从逻辑回归开始，一直到使用 **LangChain** 构建 **GenAI 应用程序**。
- **引用 DeepLearning.AI 和 IBM 课程**：新的 Substack 引用了在 **Coursera** 上学习的、由 **DeepLearning.AI** 和 **IBM** 设计的课程。
   - 该 Substack 还补充了该领域最新出版物的研究参考文献。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1379808857668325417)** (1 条消息): 

> `Gradio Agents, MCP Hackathon, Mistral AI Agentic Support, LlamaIndex framework` 


- **Gradio 举办 Agents/MCP 黑客松问答会！**：Gradio 今天将举办 **三场答疑时间（office hours）**，针对 **Gradio Agents** 和 **MCP 黑客松** 的技术问题进行解答。
   - 会议将邀请专家参与：太平洋时间 **上午 11 点** 由 [Gradio 解答 MCP 问题](https://discord.com/events/879548962464493619/1379545280109744159)，**上午 8 点** 由 [Mistral AI 解答 Agentic 和 MCP 支持](https://discord.com/events/879548962464493619/1379789818615304292)，以及 **上午 9 点** 由 [LlamaIndex 解答 MCP、Agents](https://discord.com/events/879548962464493619/1379561017536938095) 或任何与 **LlamaIndex 框架** 相关的问题。
- **Mistral 和 LlamaIndex 加入 Gradio**：**Mistral AI** 和 **LlamaIndex** 的代表将在 **Gradio Agents 和 MCP 黑客松** 期间主持答疑时间，回答有关其框架的问题。
   - 这些专家将就 Mistral 的 Agentic 和 MCP 支持以及 LlamaIndex 框架提供指导。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1379777655888547893)** (2 messages): 

> `Meta-Llama model access, Agents course deadlines` 


- **Meta-Llama 访问：被拒绝后能否重新申请？**：一位用户在注册 **Meta-Llama model** 时遭到拒绝，并询问了重新申请的可能性以及被拒绝的潜在原因。
   - 他们还询问了运行需要该模型的 **Jupyter notebooks** 的其他替代方案。
- **Agents 课程截止日期说明**：一名参加 Agents 课程的新学生注意到截止日期为 **2025年5月1日**，并询问现在开始学习是否有资格获得证书。
   - 由于提到的截止日期与当前课程日期的可用性之间存在差异，他们表达了疑虑。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1379536590782857347)** (21 messages🔥): 

> `OpenAI Free Tier Eligibility, Unit 4 Assignment Difficulties, Local LLM Performance, Audio and YouTube Processing, Whisper Model Usage` 


- **Whisper 转录成本低廉**：用户们正成功地使用 **OpenAI 的 Whisper model** 进行音频转录，而无需支付模型提供商的费用，同时 [volodymyr kublytskyi 的仓库](https://huggingface.co/spaces/vkublytskyi/Final_Assignment_Agent/blob/main/tools/youtube_video_tool.py) 为 Agent 视频交互提供了帮助。
   - 该视频工具显然是由一名用户编写的，他的工作受到了极高的赞赏。
- **Unit 4 令人沮丧，本地 LLM 表现吃力**：Unit 4 的作业对小型模型构成了挑战，即使是基于大型架构的模型也是如此，这引发了人们对是否有任何本地托管的 LLM 获得了 **30分或以上** 的好奇。
   - 一名用户向 [openrouter.ai](https://openrouter.ai) 充值了 **$10**，并表示他们现在可以 *访问所有模型* 且 *账单管理非常方便*。
- **课程仍在进行中**：现在仍有新参与者加入课程，确认信息显示，晚开始主要会影响 **2025年7月1日**（最终项目截止日期）之后的证书资格，不过第一单元的证书很容易获得。
   - 存在关于超出 **免费层级限制** 以及寻找 **Qwen2.5 Coder** 等模型的最新 **Hugging Face endpoints** 的担忧。
- **Gemini Flash 在 SmolAgents 中表现尚可**：**Gemini-2.0-flash** 在 OpenAI 服务器上与 **SmolAgents** 配合得 *相当不错*，如果在免费层级中加入某种延迟以避开每分钟约 **15 次请求** 的限制，每天可提供 **1500 次调用**。
   - 该用户仅通过 *良好的 Web/Wikipedia 搜索和一些其他通用工具就获得了 50 分*。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1379552860735275049)** (89 messages🔥🔥): 

> `Manus task context limit, Manus AI Competitor H Runner, Manus AI credits, Interactive experiences: website or app, Cursor and Replit IDE` 


- **Manus 任务达到上下文限制，需从头开始**：一名用户报告称 **Manus** 在运行 1 小时 55 分钟后达到了上下文限制，需要创建一个新任务，在继承压缩后的上下文后从头开始。
   - 用户对重新开始以及达到上下文限制后进度丢失感到失望。
- **H Runner 争夺 AI 关注度**：一名成员分享了 *H Company* 的 **H Runner** 链接 ([https://www.hcompany.ai/](https://www.hcompany.ai/))，将其作为 **Manus AI Agent** 的竞争对手推荐。
   - 其他人分享称它目前是免费的，但不够先进，限制为 **每日 10 次运行**。
- **Manus 积分消耗引发辩论**：由于 **Manus** 在幻灯片边界外进行构建，一名用户在一个 30 页的 PowerPoint 演示文稿上花费了 **$50**。
   - 另一名用户发现一段 30 秒的视频花费了 **208 积分**，而其他人则分享了推荐链接以获取更多积分。
- **交互式体验：Web vs App**：成员们讨论了交互式体验是作为网站最好，还是作为像 GitHub 上的 JS 应用一样托管最好。
   - 一名成员建议这取决于产品，并举例说交互式电影需要大屏幕，而语言学习应用则受益于记忆和口语练习。
- **Cursor、Devin 和 Replit：IDE 使用印象**：一名成员创建网站和 Web 应用，需要将 **Manus** 的输出在 **Cursor** 或其他 IDE 中进行重构以使其能够运行。
   - 另一名成员一直在尝试使用 **Cursor**、**Devin 2.0** 和 **Replit**，他发现后者在每天制作一个应用方面非常灵巧。


  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1380005773332975657)** (1 条消息): 

> `DeepHermes 24B 停机，API 问题` 


- ****DeepHermes 24B** 面临 API 停机**：API 和聊天产品上的 **DeepHermes 24B** 均受到停机影响。
- **API 和聊天产品中断**：在 **DeepHermes 24B** API 和聊天产品停机期间，请用户对团队保持耐心。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1379546859005018122)** (68 条消息🔥🔥): 

> `服务器标签，参数高效微调，Shisa-v2 405B 模型，淹没在 AI 发布潮中，Claude 的 Agent 行为` 


- **Nous Research 请求服务器标签 (Server Tags)**：一名成员请求为 **Nous Research** 创建**服务器标签**，以增强服务器内的可见性和组织性，如 [Discord Support 文档](https://support.discord.com/hc/en-us/articles/31444248479639-Server-Tags)中所述。
   - 该请求得到了积极响应，并保证**服务器标签**将在 **24 小时**内实现。
- **参数高效微调 (Parameter-Efficient Finetuning) 引发质疑**：一名成员介绍了一种用于持续预训练的新型**参数高效微调**方法，声称与全量微调和 **LoRA** 相比，其**知识吸收率提高 4 倍**，**灾难性遗忘减少 30%**。
   - 成员们对这些说法表示怀疑，要求提供更多细节，并分享了一个相关的 [X 帖子](https://x.com/dylan522p/status/1930045049816883510?s=46)链接。
- **日本发布 Shisa-v2 405B 模型**：一名成员宣布发布 **Shisa-v2 405B** 模型，这是**日本**训练的最强大的模型，专注于**日语**和**英语**，性能可与 **GPT-4/Deepseek** 媲美。
   - 成员分享了其 **H200 节点**的端点，邀请用户在 [chat.shisa.ai](https://chat.shisa.ai/) 测试该模型，另一名成员提出可以回答有关该模型训练的问题，并承诺在 **Arxiv** 上发布详细的技术报告。
- **用户努力应对海量的 AI 发布**：一名成员表示，面对大量涌现的新 **AI 发布**（包括 **Codex**、**O3 Pro**、**Claude Code**、**Deep Search**、**Gemini+** 和 **Nous SMC**）感到应接不暇。
   - 另一名成员指出，尽管发布速度很快，但工具层面并没有太大变化，因为它们仍然只是相同事物的不同排列组合。
- **Claude 仍是顶级模型**：成员们讨论了 **Claude** 的性能，指出与其他模型相比，它具有更优越的 **Agent 行为 (Agentic Behavior)**。
   - 一名成员幽默地提到，当他们临时切换到另一个模型时，**Claude** 似乎表现得“伤了感情”。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1379652931913257090)** (4 条消息): 

> `Loom 工具，Hermes 70b` 


- **正在尝试 Loom 工具**：一名成员正在尝试 [Loom](https://github.com/socketteer/loom)，这是他们可能在频道中听说过的一个工具。
   - 另一名成员发布了 [weavers.neocities.org/loom](https://weavers.neocities.org/loom) 的链接，似乎与讨论有关。
- **部署 Hermes 70b**：一名成员推荐将 **Hermes 70b** 作为配合 Loom 运行的 Nous Research 模型。
   - 根据周围的讨论，可以推断 **Hermes 70b** 是一个 Nous Research 模型。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1379988800515735625)** (1 条消息): 

> `通过基于文本的自我博弈进化 LLM，AI 论文反馈` 


- **新论文：通过自我博弈进化的 LLM**：一名成员宣布发表了他们的论文："[Evolving LLMs Through Text-Based Self-Play: Achieving Emergent Performance](https://ai.vixra.org/abs/2506.0018)"。
   - 该论文探讨了通过迭代的基于文本的自我改进来增强语言模型能力的方法。
- **邀请社区评审 AI 研究**：自我博弈论文的作者与社区分享了他们的工作，寻求想法和反馈。
   - 他们正在寻求关于进化 LLM 的方法以及观察到的涌现性能的见解。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1379716842821517456)** (1 条消息): 

> `Merlin app, bird identification, sound analysis` 


- **Merlin App 不仅仅是照片识别**: 一位成员分享了 [Merlin 鸟类识别应用](https://merlin.allaboutbirds.org/)，强调了它分析 **照片和声音** 以识别鸟类物种的能力。
   - 它可以从照片和声音中识别鸟类物种。
- **通过声音分析进行鸟类识别**: Merlin 应用的声音分析功能被特别指出是根据鸟鸣声识别鸟类的宝贵工具。
   - 这补充了其照片分析能力，为鸟类识别提供了一种全面的方法。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1379988800515735625)** (1 条消息): 

> `Evolving LLMs, Self-Play, Emergent Performance` 


- **LLMs 通过基于文本的 Self-Play 进行演化！**: 一位成员宣布了他们的论文《Evolving LLMs Through Text-Based Self-Play: Achieving Emergent Performance》的发表，现在可以在 [ai.vixra.org](https://ai.vixra.org/abs/2506.0018) 查阅。
   - 他们邀请社区分享对该研究的想法和反馈。
- **论文已发表，征求意见**: 论文《Evolving LLMs Through Text-Based Self-Play: Achieving Emergent Performance》的作者就其最近发表的论文征求意见。
   - 该论文可在 [ai.vixra.org](https://ai.vixra.org/abs/2506.0018) 获取。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1379545592035803217)** (72 条消息🔥🔥): 

> `LLM Engineer's Almanac by Modal Labs, PDF ingestion pipeline in AWS, Anthropic's capacity cuts, Codex with internet access, OpenAI Agent Development` 


- **Modal Labs 发布 LLM Engineer's Almanac**: Modal Labs 推出了 [LLM Engineer's Almanac](https://x.com/charles_irl/status/1929615080494416213)，包含数千个针对 **vLLM**、**SGLang** 和 **TensorRT-LLM** 框架的开源权重模型的 LLM 推理基准测试。
   - 该发布包括结果、用于复现的代码、以及一份涉及构建 vs 购买、成本估算和框架选择的高管摘要，以及用于理解性能指标的 **'stopwatch' 基准测试框架**。
- **警惕 AWS Textract 的陷阱**: AWS 中的一个自建 **PDF 摄取流水线 (PDF ingestion pipeline)** 使用 Lambda 拆分 PDF 并使用 Textract 进行解析，并使用队列管理 Textract 的请求限制。
   - 一位用户警告说，**Textract 的准确率** 在法律和监管文件上可能低至 *3%*，并链接到了一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/robertreich_when-word-for-word-accuracy-is-key-in-etl-activity-7265008546793086978-hfaj?utm_source=share&utm_medium=member_desktop&rcm=ACoAAABOb18Bac53omUsFRAIBEVDUe013Eez5zoTry)。
- **Anthropic 模型容量削减引发轩然大波**: 根据 [此帖子](https://x.com/_mohansolo/status/1930034960385356174)，Anthropic 在不到五天的通知时间内意外切断了几乎所有 **Claude 3.x 模型容量**，影响了 Windsurf 等服务。
   - 用户表示失望，一些人正在考虑迁移，而 ai.engineer 正在提供 **BYOK 选项**，并改进了他们针对 Gemini 2.5 Pro 和 GPT-4.1 的 Agent 框架，根据 [此帖子](https://x.com/kevinhou22/status/1930401320210706802)。
- **Altman 为编程工具增加联网功能**: Sam Altman 宣布，AI 编程工具 **Codex** 现在为 **ChatGPT Plus** 用户提供可选的联网功能，由于 [此推文](https://x.com/sama/status/1930006856019390521) 中描述的复杂权衡，该功能默认禁用。
   - 社区讨论了其影响和潜在的安全担忧，Grok 对该公告提供了详细解释。
- **OpenAI 构建可靠的 Agents**: OpenAI 宣布了构建 Agents 的四个更新：TypeScript 版 Agents SDK、RealtimeAgent 功能、Realtime API 会话的 Traces 支持以及语音到语音模型的改进。
   - 这些增强功能旨在提高可靠性、一致性和用户控制，正如 [此推文](https://x.com/OpenAIDevs/status/1929950012160790876) 所示，**Perplexity**、**Intercom** 和 **VolleyGames** 等早期测试者已经进行了演示。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1379545535123427358)** (5 条消息): 

> `Notebook LM 结合 Microsoft Learn，城市与县级 Notebook，MP3 对比 M4A` 


- **Microsoft Learn 用户涌向 Notebook LM**：一位用户询问是否有其他人将 **Notebook LM** 与 **Microsoft Learn** 结合使用来准备 **Microsoft Certification**，并征求使用案例和技巧。
   - 在给定的消息中没有提供回复或具体示例。
- **Palm Bayer 发布 AI 驱动的公共 Notebooks**：一位用户使用 **Notebook LM** 创建了两个笔记本，一个针对其所在城市，另一个针对县，并在 [博客文章](https://www.thepalmbayer.com/p/palm-bayer-unveils-ai-powered-public) 中进行了介绍。
   - 他们将其描述为 AI 驱动的公共笔记本。
- **AI 爱好者感叹缺少 M4A 支持**：一位用户表达了对 **AI** 的热爱，但指出 **Notebook LM** 仅接受 **MP3 音频文件**，不支持 **M4A**。
   - 这一限制约束了该工具可以使用的音频文件类型。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1379550540689703077)** (67 条消息🔥🔥): 

> `Gemini 2.5 Pro 对比 Flash，音频生成长度，NotebookLM 与 Google Docs 同步，公共 Notebook 分享，NotebookLM 移动端 App` 


- **Google Workspace 悄悄引入 NotebookLM 功能**：一位用户分享了 [Chrome Unboxed](https://chromeunboxed.com/google-workspace-feature-drop-for-may-2025-is-loaded-with-new-features/) 的链接，强调 **NotebookLM** 的功能即将引入 **Google Workspace**，尽管可能仅限于单个文档。
   - 用户们正积极关注 **NotebookLM** 何时开始使用更先进的模型（如 **Gemini 2.5 Pro** 甚至 **Flash**）来提升性能。
- **Flash 对比 Pro，快速与详尽之争**：成员们正在讨论 **Gemini 2.5 Flash** 和 **2.5 Pro** 之间的区别，一些人更倾向于 **Pro** 的详尽性，特别是在上传大文件且细微细节至关重要的情况下。
   - 一位用户建议实施 Beta 分支，允许切换到 **2.5 Pro**，以获得可能更好的质量，尽管生成时间更长。
- **发现音频概览长度自定义功能**：用户发现可以通过在 Studio 中选择 “Customize” 而不是 “Generate” 来自定义音频概览的长度，提供较短、默认或较长的选项。
   - 有人指出官方 App 可能没有此功能，但在网页版和移动网页版上可以使用。
- **Google Docs 更新需要手动重新同步**：用户确认，在将 **Google Doc** 添加为 **NotebookLM** 的来源后，对其进行的更改不会自动反映，需要从预览界面手动重新同步。
   - 一位用户澄清说，新的公共分享选项不需要为 **Google Doc** 本身设置特定的分享权限，因为 **NLM** 分享的是它自己的副本，且分享链接在更新过程中保持不变。
- **移动端 App 缺失许多功能**：**NotebookLM** 移动端 App 被认为是一个 *“最小价值产品”*，与网页版相比缺少许多功能。
   - 鼓励用户在 Feature Request 频道的 “Mobile App” 线程中反馈所需的功能。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1379558767880044574)** (29 条消息🔥): 

> `Parameter-Efficient Finetuning, LLMs 知识扩展, 用于同构测试的 MCP Server, 图神经网络中的原型理论` 


- **Parameter-Efficient Finetuning 宣称具有更优的知识吸收能力**：一名成员报告称，一种新的 **parameter-efficient finetuning** 方法显示出比全量微调和 **LoRA** 高出 **4 倍的知识吸收能力**，且灾难性遗忘减少了 30%。
   - 该方法旨在高效地教给模型新信息而不丢失现有知识，特别适用于领域自适应以及在本地设置中添加特定知识。
- **探索知识扩展作为 RAG 的替代方案**：一名成员计划使用一系列书籍和文档来扩展 LLM 的知识，并针对辅助任务将其收益与 **类 RAG 方法** 进行比较。
   - 他们分享了一个 [x 链接](https://x.com/unusual_whales/status/1929998955703931375) 以及一份讨论 AI 权利的 [markdown 文档](https://cdn.discordapp.com/attachments/986699377257119794/1379670337008046080/UDAIR.md?ex=68426721&is=684115a1&hm=4e73690d912c8e0286f50b7a456f683012b700561418b45222466ae5230e3a9f&)，并指出这可能会引发 *疯狂的对话*。
- **同构计算实现惊人的效率提升**：一名成员寻求帮助以寻找或创建一个 **MCP server** 来测试 **isomorphism**（同构），报告称在更短的时间内使用更少的资源实现了 **99% 相似的结果**。
   - 另一名成员询问了 **isomorphism** 的定义，将其定义为 *两个结构之间保持所有相关操作或关系的双射映射*。
- **原型理论驱动图神经网络**：一名成员寻求关于在 Graph Neural Networks 的图结构中实现 **prototype theory**（原型理论）的反馈，其灵感来自人脑的概念形成。
   - 该想法涉及将新概念表示为现有实体的类型，其中的例外情况通过图中的抑制性连接来实现。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1379547580966371459)** (25 条消息🔥): 

> `vec2vec 代码审查, Muon Optimizer 细节, 论文阅读技巧` 


- **Vec2Vec 代码审查推迟**：一名成员提议审查 [**vec2vec** 代码](https://github.com/rjha18/vec2vec)（某篇论文的实现），但后来由于缺乏即时兴趣而取消。
   - 一名成员表示有兴趣观看演示者的实时论文分析技巧，希望能深入了解其思考过程。
- **深入探讨 Muon Optimizer 细节**：一名成员询问了 **Muon optimizer**，指出它对不适合 **Muon** 的参数使用了 **AdamW**，并链接到了多任务学习的 [实验结果](https://github.com/KellerJordan/Muon/issues/25)。
   - 另一名成员解释说，**Muon optimizer** 调整权重矩阵（weight-matrix）的梯度，使其特征值近似等于 1，这与 **SGD** 和 **Adam** 截然不同。
- **在论文阅读中挣扎**：一名成员询问是否可以观看更有经验的成员如何实时阅读论文，因为他们在这一过程中遇到了困难。
   - 该成员希望看到相关技巧以及经验丰富的成员如何分析论文。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1379584059092504699)** (15 条消息🔥): 

> `Mistral Code 发布, OpenAI ChatGPT 日志隐私担忧, Elon 对 AI 的立场` 


- **Mistral Code 发布，旨在提升 10 倍开发效率**：**Mistral AI** 推出了 [Mistral Code](https://mistral.ai/news/mistral-code)，这是一款 **AI 驱动的编程助手**，将强大的模型、IDE 内助手、本地部署选项和企业级工具整合到一个包中。
   - Mistral Code 基于成熟的开源项目 **Continue** 构建，支持 JetBrains IDE 和 VSCode，是 Mistral 致力于帮助开发者利用 AI 取得成功的延续。
- **OpenAI 保存 ChatGPT 日志引发隐私噩梦**：成员们讨论了 [一篇 ArsTechnica 的文章](https://arstechnica.com/tech-policy/2025/06/openai-says-court-forcing-it-to-save-all-chatgpt-logs-is-a-privacy-nightmare/)，该文章指出 **OpenAI** 正被迫保存所有 **ChatGPT 日志**，*包括已删除的聊天记录以及通过其 API 业务产品记录的敏感聊天内容*。
   - 一名成员对为何会发生这种情况表示 *好奇*。
- **Elon 对 AI 的立场，权力集中？**：一名成员想知道 **Elon Musk** 对 AI 的负面立场是否源于 AI 未能将权力集中在他手中。
   - 另一名成员发布了 *如果属实，p(1984) 的概率非常高* [并附带了一个 YouTube 视频链接](https://www.youtube.com/watch?v=Sd6F2pfKJmk)。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1379560199056003194)** (46 messages🔥): 

> `Parameter-efficient finetuning, Twitter scraper, Imitation Learning, Scalable web scraping with AI agents` 


- **新型参数高效微调方法出现**：一种适用于持续预训练的新型参数高效微调（Parameter-efficient finetuning）方法问世，声称在参数量更少的情况下，其**知识吸收量约为 LoRA 的 4 倍**，且**灾难性遗忘减少了 30%**。
   - 该方法旨在高效地教给模型新信息，同时不覆盖现有知识。
- **无需 API 的 Twitter 爬虫将数据记录至 Postgres**：一位成员分享了一个 [Twitter 爬虫](https://gist.github.com/mookiezi/9ea0f0f5aad76a51e5b35a084d82a9df)，该工具**不使用 API**，可将数据记录到 Postgres，并跳过转推（retweets）。
   - 该爬虫不收集回复元数据，因此更适合用于个人资料页。
- **模仿学习需要专家行为的良好覆盖**：一份观点报告 ([arxiv.org/abs/2503.09722](https://arxiv.org/abs/2503.09722)) 指出，**模仿学习（Imitation Learning）需要对专家动作进行良好覆盖**，包括他们如何应对失败以及进行修正。
   - 报告强调，记录的知识往往缺乏修正/调整数据，这使得在高维空间中实现完全覆盖具有挑战性。
- **通过 AI Agent 实现可扩展的网络爬虫：一项艰巨任务**：一位成员正在寻求一种**使用 AI Agent 的可扩展解决方案**，为 300 多个包含规划申请数据的英国议会网站创建爬虫。
   - 目标是让 Agent 导航网站、分析网络请求，并生成基于 Python 的爬虫以提取结构化 JSON 数据。文中提到了 [Holo1-7B](https://huggingface.co/Hcompany/Holo1-7B) 和 [Integuru-AI/Integuru](https://github.com/Integuru-AI/Integuru) 作为可能结合的项目。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1379667428354691082)** (4 messages): 

> `UDAIR.md document on AI Rights, Universal Algorithm POC for NLP, Options Trading, and Electrochemical Reactions, Quantum Field Based Architecture with Sinusoidal Sparsity, AI-generated Research` 


- **通过科幻场景探讨 AI 权利**：一位成员分享了一份 [文档](https://cdn.discordapp.com/attachments/747850033994662000/1379667427591192687/UDAIR.md?ex=6842646b&is=684112eb&hm=7122028311f4bfeb188d7bf31cdc830b537036e9b3b317451f4606b432a96e3e&)，通过科幻电影和现实场景进行测试，以推导出关于 **AI 权利** 的有趣观点。
- **万能算法**：一位成员分享了其 [研究](https://github.com/qLeviathan/g-qfnn) 演示，这是一种**通用算法**，在 **NLP**、**期权交易**和**电化学反应**方面具有基础概念验证（POC）。
- **量子场架构揭晓**：提议的架构是一个**由 phi 调制的二维圆柱体**，其中 Z 轴作为**量子比特旋转损耗装置（qubit rotational loss device）**来控制音高（pitch）。
- **撤回对 AI 研究的欢迎**：一位成员表示，该频道*不是发布 AI 生成的研究的地方*，该成员应该*去别处发布*。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1379850232119951412)** (7 messages): 

> `AI Compute Investment, AI ROI, AI Startups, AI Job Market, PhD Earnings` 


- **AI 算力投资泡沫？**：一位成员推测，当前十年内 **AI 算力投资** 的规模扩张是不可持续的，最终会放缓。
   - 他认为，一旦大部分资金和人才都集中在 **AI** 领域，进展就会趋于常态化。
- **AI ROI 疑虑会戳破初创公司泡沫吗？**：一位成员对在 **AI** 的 **ROI**（投资回报率）受到质疑的就业市场中毕业表示担忧，这可能导致 **AI 初创公司** 的泡沫破裂。
   - 他们声称许多 **AI 初创公司 CEO** 缺乏 **ML** 专业知识，并且由无法正确评估 **ML** 技能的投资者支持。
- **博士辞职与互联网泡沫回响**：一位成员已经辞职，并预计拥有博士学位后的收入会降低，这与**互联网泡沫**时期的情况类似。
   - 他们认为，即使发生崩盘，廉价的 **GPUs** 仍将被用于*更有趣的工作*；虽然在崩盘期毕业可能会降低终身收入，但不会是灾难性的。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1379805644286197770)** (9 条消息🔥): 

> `General agents and world models, Semantic Virus exploits LLM vulnerabilities, NCF, Semantic Viruses, and the CupCake framework study, Interpretability intact without teacher-forcing, AI training without teacher-forcing` 


- **通用 Agent 需要 World Models，论文陈述**：一篇 [论文](https://arxiv.org/pdf/2506.01622) 指出，通用 Agent 需要 **world models**。
   - 作者认为 “Semantic Virus” 利用了这一点，如果 **LLM 的 world model** 存在“漏洞”或“断连区域”，持久的叙事可以在上下文内*感染*推理路径。
- **Semantic Virus 利用 LLM 弱点**：**Semantic Virus** 概念利用了 **LLM world models** 中的漏洞，如果模型存在“漏洞”或“断连区域”，叙事可以感染推理路径。
   - **Semantic Virus** 不会重写基础 **World Model**，而是劫持其在 context window 内的当前激活。
- **探索 NCF、Semantic Viruses 和 CupCake 框架**：一位成员介绍了他关于 **NCF、Semantic Viruses 和 CupCake 框架** 的研究，旨在通过叙事和上下文探索对隐式 **world models** 的交互和影响，并附带了项目的 [代码](https://github.com/IhateCreatingUserNames2/SemanticVirus/blob/main/Frameworks%20Validation%20and%20Analysis_.pdf) 和 [研究](https://github.com/IhateCreatingUserNames2/SemanticVirus/blob/main/PDF%20Frameworks%20Validation%20Research_.pdf) 链接。
   - 该研究识别了通过访问和构建 **world models** 产生的涌现属性（如角色和模拟意识），以及由于其激活的易塑性而产生的漏洞。
- **质疑无 Teacher-Forcing 下的可解释性完整性**：有人提出了在不使用 **teacher-forcing** 的情况下保持可解释性完整的可能性。
   - 该成员特别询问是否有关于不使用 **teacher-forcing** 进行 **AI training** 的研究，最好是尝试同时维持可解释性。
- **不使用 Teacher-Forcing 的训练可能无法实现**：一位成员提到，除了 **RL** 之外，可能还没有任何不使用 **teacher-forcing** 的生成式 **AI training** 能扩展到合理的规模。
   - 不使用 **teacher-forcing** 进行训练可能需要极长时间，考虑到数据和 context lengths 的可接受最小规模，对于任何现代模型来说，难度甚至可能达到“不可能”的程度。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1379540513094500384)** (2 条消息): 

> `Pythia Remake, Percy plans` 


- **Pythia 重制版头脑风暴开始**：鉴于 [Percy 的计划](https://marin.community/data-browser/experiment/?path=gs%3A//marin-us-central2/experiments/exp1337_scaling_suite-a2518e.json)，一位成员询问了关于 **Pythia** 重制版的改进建议。
   - 另一位成员提到，在推文发布后，他们已经在起草关于该主题的评论。
- **社区热切期待 Pythia 重制版评论**：一位社区成员表示期待分享他们对 **Pythia** 潜在重新设计的见解。
   - 该成员表示，在推文发布后，他们已经在起草关于该主题的评论。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1379545696201343016)** (17 条消息🔥): 

> `Llama 4 Image Support, ROCm drivers on Ubuntu, agenticSeek vs OpenManus, Embedding model choice, ROCm vision module slowdown` 


- **Llama 4 图像支持仍是个谜**：一位用户在看到 Unsloth 版本显示不支持后，询问 **Llama 4** 在 LM Studio 上是否支持图像。
   - 讨论中未提供确认或否认。
- **Ubuntu 迁移需要 AMD 的 ROCm 驱动**：一位为了最大化模型性能而从 Windows 迁移到 Ubuntu 的用户，询问了如何为 **AMD 6700XT** 安装 **ROCm** 驱动。
   - 讨论中澄清了 **6700XT** 在 LM Studio 中仅支持 Vulcan。
- **agenticSeek 由 OpenManus 更名而来**：一位用户分享了 [agenticSeek](https://github.com/Fosowl/agenticSeek) 的链接并询问是否有人尝试过，另一位用户指出其名称由 **OpenManus** 更改而来（类似于 OpenDevin 更名为 OpenHands）。
   - 更名的原因可能是由于版权问题。
- **Gemma 在 Embedding 模型中表现出色**：一位测试了多种 Embedding 模型（**Gemma 3 4b**、**12b**、**Deep Seek 8b**、**Microsoft phi 4 small**）的用户发现，**Gemma** 提供的答案比 Deep Seek 或 Microsoft Phi 更准确。
   - 该用户的数据由文本和 PDF（0.5-30 MB）混合组成，并与 Supabase 和 n8n 配合使用。
- **ROCm 视觉模块遭遇严重减速**：一位用户报告称，在 **7900XT 20GB** 上使用新的 **ROCm llama.cpp v1.34.1** 运行时，视觉模块出现显著减速，响应时间从约 1 秒跳升至 10 秒以上。
   - 该用户分享了[他们的结果截图](https://cdn.discordapp.com/attachments/1110598183144399061/1379953808532049981/image.png?ex=68421da2&is=6840cc22&hm=37d660db87619d86ca215fc8862f4762688295f6516dcb95ee68d5e84a525bc2&)，并被要求在相应的 Discord 频道中分享结果。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1379642785749139536)** (48 条消息🔥): 

> `Server boot times, SSD vs HDD, NAND cell refreshing` 


- **服务器设置停滞：新装机面临漫长的启动时间**：组装新服务器可能会导致启动时间延长，有时长达 **10 分钟**，尤其是在配备大量 RAM 或使用某些服务器主板时。
   - 一些成员指出，服务器主板初始化可能需要一段时间，特别是配备了如 **1TB** 这样的大容量 RAM 时；另一些人则询问 **EXPO RAM** 设置是否也有类似的启动时间。
- **墨盒阴谋：SSD 模仿打印机墨盒经济**：一位成员将 **SSD** 的限制与打印机墨盒类比，暗示制造商可能会限制硬件功能以销售更多新产品。
   - 他们指出，打印机公司经常销售墨水量有限的墨盒，并对墨盒重复使用实施限制，使得墨水按重量计算比黄金还贵；而 SSD 在达到其 TBW 额定值后，驱动器可能会被锁定为只读，即使它可能还可以运行更长时间。
- **SSD 秘密：揭秘数据损坏与刷新周期**：讨论涉及了 **SSD** 如果长时间不通电可能导致的数据损坏，这与 HDD 形成对比，后者的数据是物理写入的，随时间退化的可能性较小。
   - 讨论中提到，SSD 中使用的 **NAND** 闪存单元会随时间缓慢漏电，据报告硬件需要执行 *read refresh*（读取刷新）。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1379623015704297513)** (49 条消息🔥): 

> `MCP API key 货币化, MCP 上下文管理, A2A 框架 vs MCP, Pydantic-AI, 托管 MCP 服务器` 


- **MCP API Keys：一场关于 SaaS 的辩论**：成员们讨论了在 MCP 中使用 **API keys** 进行货币化，一位成员建议这与任何带有 API keys 和计费仪表板的 SaaS 类似。
   - 他们指出 MCP 客户端会向服务器发送 **auth**，从而简化了货币化，并质疑了对 MonetizedMCP 的需求。
- **A2A vs MCP：规范对决**：成员们讨论了将 **A2A** ([https://github.com/google/A2A/](https://github.com/google/A2A/)) 作为使用 MCP 的 Agent 框架，但注意到其采用率有限。
   - 有人认为 A2A 正在“闭门”进行大额交易，而另一些人则更倾向于 **A2A 规范** 而非 MCP。
- **Pydantic-AI 精简 Agent**：成员们建议从 **pydantic-ai-slim** ([https://ai.pydantic.dev/install/](https://ai.pydantic.dev/install/)) 开始进行 Agent 框架开发，并提到了其便捷方法 `.to_a2a()`。
   - 他们还提到了针对现有 Agent 的可选 a2a 组 (`uv add 'pydantic-ai-slim[a2a]'`)。
- **Cloudflare MCP 托管**：一位成员就如何为非技术用户在 **Cloudflare** 上托管 **MCP server** 寻求建议。
   - 澄清了 **HTTP transport** MCP 服务器不应需要本地软件，前提是 MCP 客户端原生支持它；否则，可能需要一个转换器。
- **跨 Agent 的上下文危机：MCP 来救场？**：一位成员询问 **MCP** 如何在多个 Agent 之间管理上下文，以及维持上下文所需的工程机制。
   - 澄清了 **MCP 并非 Agent 优先**，并提供了一份指南：[https://fast-agent.ai/mcp/state_transfer/](https://fast-agent.ai/mcp/state_transfer/)。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1379873438356734062)** (4 条消息): 

> `MCP 价值, Block 对 MCP 的采用, Goose 与 A2A 协议, deeplinks` 


- **Block 推动 MCP 采用**：一位成员提到，他拥有 **12,000 名员工** 的公司 **Block** 正在 **15 个以上的职能部门** 中使用 **MCP**。
   - 他还分享了一个 [YouTube 视频](https://youtu.be/IDWqWdLESgY)，讲述了他在公司大规模采用 AI 的故事。
- **将 MCP 与 Google 的 A2A 协议集成**：一位成员一直在阅读关于实现 **MCP servers** 的资料，并尝试将其与 **Google 新的 A2A 协议** 集成。
   - 他们还想知道 **Goose** 是否有计划研究用于多 Agent 系统的 **A2A**。
- **Deeplinks 即将推出**：一位成员分享了[关于生成安装链接的文档链接](https://docs.cursor.com/deeplinks#generate-install-link)。
   - 另一位成员表示，他们希望在本周推出 **deeplinks**。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1379550773527974090)** (6 条消息): 

> `Agentic AI, 财务报告聊天机器人, LlamaIndex 问题, Agent 设计模式` 


- **LlamaIndex 亮相 AI Engineer 活动**：LlamaIndex 参加了在旧金山举行的 [@aidotengineer](https://twitter.com/aidotengineer)，在 G11 展位与 CEO @jerryjliu0 和 AI 工程团队一起展示最新的 **Agentic AI**。
- **使用 LlamaIndex 构建财务聊天机器人**：LlamaIndex 提供了一个[实操 Colab](https://twitter.com/llama_index/status/1930051898247393729)，用于从头开始构建一个**多 Agent 财务报告**生成聊天机器人，解析并索引来自 Adobe 的 10-K 文件，使用 Agentic RAG。
   - 这源于 @jerryjliu0 的研讨会。
- **Gradio MCP 黑客松**：[@Gradio](https://twitter.com/Gradio) [@huggingface](https://twitter.com/huggingface) MCP 黑客松的答疑时间在此消息后不久开始，最佳 LlamaIndex 提交作品可获得 [$1000 奖金](https://twitter.com/llama_index/status/1930286458340028484)，并有 1 万 LlamaCloud 积分待领取。
   - 成员 @tuanacelik 和 @LoganMarkewich 回答了 LlamaIndex 的相关问题。
- **Agent 设计模式**：来自 LlamaIndex 的 @seldo 在 [@aiDotEngineer](https://twitter.com/aiDotEngineer) 分享了**生产环境中的有效 Agent 设计模式**。
- **LlamaExtract 自动化 SEC Form 4 提取**：LlamaIndex 展示了如何使用 [LlamaExtract](https://twitter.com/llama_index/status/1930414284670152875) 和 Agent 工作流自动化 SEC Form 4 的提取。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1379563356922319012)** (20 messages🔥): 

> `Gradio MCP Hackathon, Property Graph Index, Code Interpreter Agent, Ollama, readthedocs website` 


- **为 Gradio MCP Hackathon 参赛者举办的 Office Hours**: 成员们正在 HuggingFace Discord 服务器上为 **Gradio MCP Hackathon** 参赛者举办 Office Hours，[链接在此](https://discord.com/events/879548962464493619/1379561017536938095)。
- **探索 Property Graph Index**: 一位成员正在探索 **Property Graph Index**，并希望了解其**索引与检索的 token 使用量**，以及与 **GraphRAG**、**HippoRAG2** 和 **LightRAG** 相比的**检索与端到端性能**。
- **使用 Qwen3 构建 Code Interpreter Agent**: 一位成员想要构建类似 [这篇 Medium 文章](https://medium.com/@venugopal.adep/building-an-ai-data-analysis-assistant-with-llamaindex-and-openai-c0e371a432d6) 中的 **Code Interpreter Agent**，但使用 **Qwen3** 代替 **OpenAI**。
   - 另一位成员建议使用 **Ollama** 来部署 **Qwen3**，[链接在此](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/)。
- **ReadTheDocs 网站宕机**: 文档网站似乎已宕机，[这是状态页面](https://status.readthedocs.com/)。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1379889810906284183)** (23 messages🔥): 

> `Numpy removal challenges in random_crop/cutmix, Performance intuition in tinygrad, Windows backend issues with tinygrad, LSTM performance bottleneck in tinygrad, Understanding DEBUG=2 output` 


- **Tinygrad 中的 NumPy 移除工程**: 一位成员正尝试根据 `hlb_cifar10` 悬赏任务的要求，从 `random_crop/cutmix` 中移除 **NumPy**，但 NumPy 操作现在被转移到了 GPU 上。
   - 该用户在建立 **tinygrad 性能**直觉方面遇到困难，发现很难判断什么是慢的或快的。
- **Tinygrad 在 Windows 上的困扰**: 一位成员在 Windows 上使用 **tinygrad** 时遇到多个问题，包括 CPU 后端在 JIT 时崩溃，以及 BEAMS=1 时挂起，需要通过修改 autogen files 来启用 CUDA。
   - 该成员怀疑 Windows 环境导致了性能问题，但难以推断根本原因。
- **LSTM 在 Tinygrad 中表现滞后**: 在将 **VAD 模型**从 PyTorch 移植到 tinygrad 时，一位成员发现除 LSTM 外的所有层运行速度都很快。
   - 无论使用哪种后端，LSTM 层的运行速度都极其缓慢。
- **DEBUG=2 解析困难**: 一位成员发现 `DEBUG=2` 的输出内容过多且难以查阅，难以理解各列的含义以及大量的 kernels。
   - 具体而言，该成员对大量的 `randperm` kernels 表示疑问，并询问如何解析诸如 `r_512_32_8_4_8_3_16_3_4_4` 之类的名称。
- **CUDA 自定义难题**: 一位成员正在寻找在 **tinygrad** 的 CUSTOM ops 中使用 **CUDA kernels** 的示例，旨在移植一个包含 5-10 个 kernels 的项目。
   - 该成员明白自定义 kernels 可能不符合 "Zen of TinyGrad"，但由于对如何用 Python 表达所需的 kernels 理解有限，认为这是必要的。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1379585716203556915)** (15 条消息🔥): 

> `Python 3.9 支持、异步 Reward Functions、Iterable Dataset 重构 RFC、AdamW 之外的 Optimizer 兼容性、DTensor DeviceMesh 错误` 


- **Python 3.9 支持即将停止？**: **Python 3.9** 即将结束生命周期，这推动了新 linting 规则的采用（如 List -> list, Tuple -> tuple），由于需要从 `typing` 模块获取 `Union` 和 `Optional`，导致 CI 失败。
   - 这迫使开发者采取临时变通方法以维持兼容性，一位成员自嘲道：*"抱歉 Joe，这就是 CI 失败的原因 :/"*。
- **异步 GRPO Reward Functions 获得 Batch 提升**: 虽然 Reward Functions 是通过 Batch 循环进行的，以便进行潜在的并发计算，但这些调用并非原生异步，且受限于 **Reference model worker 的资源**。
   - 一位成员分享道：*"Reward Functions 只是被循环调用并传入一个 Batch，你可以尝试并发计算，但调用本身不是异步的，而且你只能访问 Reference model worker 的资源。"*
- **Iterable Dataset 重构 RFC 打破常规**: 一项 RFC（[Iterable dataset refactoring](https://github.com/pytorch/torchtune/pull/2785)）提议对 TorchTune 处理数据集的方式进行重大改革，并征求社区对其设计和潜在破坏性变更的反馈。
   - 一位成员强调了意见的重要性：*"这是一个巨大的变化。我非常感谢任何建议或想法。这看起来像是 TorchTune 处理数据集的正确方式吗？既然我们横竖都要打破现有结构，你会进行什么彻底的改变吗？"*
- **AdamW 之外的 Optimizer 测试引发 DTensor 问题**: 在全分布式 SFT 中测试 TorchTune 与 **AdamW** 之外的优化器（如 **SGD**、**Adafactor** 和 **Adagrad**）时，出现了与 `aten._foreach_lerp_.ScalarList!` 的 dtensor 参数相关的 `DeviceMesh` `AssertionError`。
   - 其他人测试了来自 torchao 的不同精度的 **Muon** 和 **AdamW**。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1379553374076141650)** (14 条消息🔥): 

> `作业截止日期、作业反馈、MOOC 后续步骤` 


- **截止日期延期？今天不行！**: 成员们询问是否可以延长原定于 **5 月 31 日**截止的作业期限，但被告知表单已经为了应对技术问题额外开放了两天。
   - 工作人员确认 *遗憾的是，他们无法再进一步开放作业提交*。
- **详细反馈被认为难以实现**: 一位成员询问是否可以对所有提交的内容（包括 **AgentX 项目**和 **Lab 作业**）提供详细反馈。
   - 工作人员表示 *作为工作人员，他们没有足够的精力这样做*，但承诺会转达这一建议。
- **MOOC 的未来尚不明朗**: 一位成员询问在 **Spring 2025 MOOC** 结束后，是否有下一步计划、新版本或进阶课程。
   - 工作人员表示 *目前还没有任何确定的消息*，但 *很有可能（但目前不保证）*。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1379931969391169536)** (1 条消息): 

> `Claude 3.7 vs 4.0、Anthropic 的开发周期、Anthropic 的优先级` 


- **开发周期和优先级披露**: 一篇博客文章对比了 **Claude 3.7** 和 **4.0** 的 [system prompts](https://www.dbreunig.com/2025/06/03/comparing-system-prompts-across-claude-versions.html)，揭示了 **Anthropic** 的开发周期和优先级。
- **System Prompts 的进一步细微差别**: 作者注意到 *Claude 3.7 与 4.0 之间的 system prompt 存在一些变化*。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1379628626747723778)** (12 messages🔥): 

> `oneformer game theorist, agenspy vs frameworks, claude_sdk execution engine, HTNs and LLM agents, Fine-tuning LLMs in ReACT format` 


- ****Oneformer 的博弈论博弈****：一名成员正在构建一个 **Oneformer** 博弈论专家，并对公开它表现得比较腼腆，同时在讨论它对抗 **Agenspy** 或其他框架的潜在成功率。
- ****Angel Azul 攻克 Claude SDK****：一名成员分享了他们在 [claude_sdk execution engine](https://github.com/darinkishore/claude_sdk/tree/t1-execution-engine) 上的工作，强调这还不是最终版本且仍有 bug，架构模式详见 [ai_docs](https://github.com/darinkishore/claude_sdk/blob/t1-execution-engine/ai_docs/ARCHITECTURE_PATTERNS.md)。
- ****HTN 技巧助力 LLM 协同****：一名成员提到他们一直在研究 **HTN**（分层任务网络），并建议 **LLM Agent** 可能会从专门针对 **ReACT format** 的微调中受益，而不是采用通用的聊天模型方法。
- ****愿景之旅：完善路线图****：一名成员询问了项目的路线图、战略愿景，以及如何适应新功能的方法，例如带有错误重试机制的 **SO/schemas**（类似 instructor）和推理器（reasoners）。


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1379625591128719461)** (3 messages): 

> `Cohere Sponsorship` 


- **咨询 Cohere 赞助联系方式**：一名成员正在寻找合适的联系人，以请求 **Cohere** 赞助一场高校黑客松。
- **另一名成员寻求赞助联系方式**：频道中有人询问如何就黑客松赞助事宜联系 **Cohere**。


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1379951100756754592)** (3 messages): 

> `Introductions to Cohere's Discord Server` 


- **成员在 Cohere Discord 服务器进行自我介绍**：新成员在 Discord 频道的 🤝-introductions 中介绍自己，按照置顶消息的指南分享他们的专业背景、当前项目、偏好技术以及社区参与目标。
   - 这些介绍展示了社区在 AI 和 GenAI 领域多样化的专业知识和兴趣。
- **另一个 Cohere Discord 服务器的自我介绍**：又一名新成员在 Discord 频道的 🤝-introductions 中介绍了自己，分享了专业背景、当前项目、偏好技术和社区参与目标。
   - 这些介绍为社区在 AI 和 GenAI 领域的多元化背景提供了缩影。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1379598303502798859)** (2 messages): 

> `GPT4All updates, MOE models and VRAM, Mac M3 Max VRAM advantage, vLLM engine for GPT4All, Nikola Tesla` 


- **LlamaCPP 库需要为 GPT4All 进行更新**：用户提到 **LlamaCPP library** 已经有几个月没有为 GPT4All 项目进行更新了，且自动更新到最新版本的选项尚未就绪。
   - 他们推测这不仅仅是简单的复制粘贴新版本。
- **MOE 模型降低 VRAM 需求**：似乎通过一些编程技巧，在卸载（offloading）特定专家和部分张量的同时，可以用合理的 **VRAM** 量运行最大的 **MOE models**。
   - 讨论集中在如何在管理内存限制的同时运行模型。
- **Mac M3 Max 在 VRAM 方面称霸**：与近乎等效的四台最新 **AMD AI MAX 395+ 128 GB** 迷你电脑或笔记本电脑的总和相比，**Mac 512 GB** 配置拥有更多的 "VRAM" (**448 GB**)，且价格相近。
   - 用户指出 Mac 的功耗（watts）也更低。
- **引入 vLLM 引擎可为 GPT4All 注入强劲动力**：用户正在研究将 **vLLM engine** 添加到 **GPT4All** 项目的可能性，这可能使其成为顶级的开源项目，拥有两个由不同编程语言编写的底层引擎。
   - 他们建议添加 **vLLM engine** 将是一个巨大的升级。
- **特斯拉的光影奇迹**：用户转而讨论尼古拉·特斯拉，并分享了一个关于他在能源和光学方面贡献的[链接](https://buck.lighting/blog/nikola-tesla-and-light/)。
   - 用户推测 *“他的发明不知何故被窃取了”*。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1379639182129107036)** (1 条消息): 

> `AI Programming, SVCAI, Liang Guo` 


- **Guo Gives Guidance on Good AI**: 行业专家 **Liang Guo** 正在举办一场关于数据分析 AI programming 的网络研讨会，RSVP [详情请见此处](https://forms.gle/e71FSdpwBtDBccgKA)。
- **SVCAI 夏季竞赛现正火热报名中**: 硅谷华人协会 (SVCA) 正在举办 **AI4Legislation** 夏季竞赛。
   - 更多详情请参阅 [该项目的 GitHub repository](https://github.com/svcaf/2025-AI4Legislation-Public)。