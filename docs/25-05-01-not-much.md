---
companies:
- microsoft
- anthropic
- cursor
- alibaba
- togethercompute
- deepseek
- meta-ai-fair
- xiaomi
- openrouterai
- cohere
date: '2025-05-01T05:44:39.731046Z'
description: '**微软**发布了 **Phi-reasoning 4**，这是一个经过微调的 14B 推理模型，性能略逊于 QwQ，但受到数据透明度和
  Token 效率问题的限制。**Anthropic** 为 **Claude** 引入了远程 MCP 服务器支持和 45 分钟的“研究模式”（Research
  mode）。**Cursor** 发布了一份模型流行度榜单。


  **阿里巴巴**推出了 **Qwen3-235B** 及其它 Qwen3 变体，强调其高性价比的代码和推理能力，并已在 **Together AI** API
  上线。**微软**还发布了 **Phi-4-Mini-Reasoning**，在 AIME 2025 和 OmniMath 基准测试中表现优异。**DeepSeek**
  发布了 **DeepSeek-Prover V2**，具备顶尖的数学解题能力，参数规模达 671B。


  **Meta AI** 的 **Llama** 模型下载量突破 12 亿次，并推出了用于输入/输出过滤和防越狱的 **Llama Guard 4** 与 **Prompt
  Guard 2**。**小米**发布了开源推理模型 **MiMo-7B**，该模型基于 25 万亿个 token 训练。


  关于 AI 模型评估的讨论指出了 **LMArena 排行榜**存在的问题，包括偏向闭源模型的数据访问偏差，以及维持公平基准测试的挑战，并建议参考 **OpenRouterAI**
  排名等替代方案。“LMArena 充斥着劣质内容且存在偏见”以及“61.3% 的数据流向了闭源模型提供商”是其中备受关注的担忧。'
id: MjAyNS0w
models:
- phi-4
- phi-4-mini-reasoning
- qwen3-235b
- qwen3-moe-235b
- qwen3-moe-30b
- qwen3-dense-32b
- qwen3-dense-14b
- qwen3-dense-8b
- qwen3-dense-4b
- qwen3-dense-0.6b
- qwen2.5-omni-3b
- deepseek-prover-v2
- llama
- llama-guard-4
- prompt-guard-2
- mimo-7b
people:
- cline
- reach_vb
- vipulved
- akhaliq
- omarsar0
- zhs05232838
- huajian_xin
- mervenoyann
- karpathy
- random_walker
- sarahookr
- blancheminerva
- clefourrier
title: 今天没发生什么事。
topics:
- reasoning
- model-fine-tuning
- model-evaluation
- benchmarking
- model-popularity
- open-source
- math
- model-scaling
- model-filtering
- jailbreak-prevention
---

**平静的一天。**

> 2025年4月30日至5月1日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（214 个频道，4767 条消息）。预计节省阅读时间（以 200wpm 计算）：453 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的氛围感呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上给我们反馈！

微软发布了 [Phi-reasoning 4](https://www.reddit.com/r/LocalLLaMA/comments/1kbvwsc/microsoft_just_released_phi_4_reasoning_14b/)，这是 14B Phi-4 的推理微调版本，其性能略逊于 QwQ，但由于其数据缺乏透明度以及对推理 Token 消耗过大的抱怨，限制了人们对它的热情。

Anthropic 在 Claude 中推出了 [远程 MCP 服务器支持](https://news.ycombinator.com/item?id=43859536) 以及长达 45 分钟的 Research 模式。

Cursor 发布了他们的模型流行度列表，并没有太多意外。


![](https://resend-attachments.s3.amazonaws.com/hM6qEzvvHIVmVdX)


---

# AI Twitter 回顾

**语言模型与发布**

- **Qwen 模型更新**：[@cline](https://twitter.com/cline/status/1917708041857949983) 报道了 **Qwen3-235B** 的早期用户反馈，指出其作为高性价比编程模型的潜力，并获得了积极的初步结果。[@reach_vb](https://twitter.com/reach_vb/status/1917938596465750476) 强调了多种 **Qwen3 模型的发布，包括 MoE (235B, 30B) 和 Dense (32, 14, 8, 4, 0.6B) 版本**。[@togethercompute](https://twitter.com/togethercompute/status/1917616701249565120) 和 [@vipulved](https://twitter.com/vipulved/status/1917777842466889873) 宣布 **Qwen 3 235B 已在 Together AI API 上线**，强调了其推理能力和效率。此外，[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1917585963775320086) 推出了 **Qwen2.5-Omni-3B**，在保持多模态理解能力的同时降低了 VRAM 消耗。
- **微软 Phi-4 推理模型**：[@_akhaliq](https://twitter.com/_akhaliq/status/1917761687723147707) 和 [@omarsar0](https://twitter.com/omarsar0/status/1917954418173247909) 提到了 **微软 Phi-4-Mini-Reasoning** 的发布，这是一个用于数学的小型语言模型，并附有技术报告。[@reach_vb](https://twitter.com/reach_vb/status/1917852036369916081) 指出 **Phi 4 Reasoning 和 Reasoning plus 现已在 Hugging Face 上线**，并强调了其在 AIME 2025 和 OmniMath 等基准测试中的表现。
- **DeepSeek 的 Prover V2**：[@zhs05232838](https://twitter.com/zhs05232838/status/1917600755936018715) 宣布发布 **DeepSeek-Prover V2**，该模型在 miniF2F 问题上获得了高分，并提升了在 PutnamBench 上的 SoTA 性能。[@reach_vb](https://twitter.com/reach_vb/status/1917549921470972172) 指出 **DeepSeek Prover V2 可直接在由 Novita Labs 提供支持的模型页面上使用**。[@huajian_xin](https://twitter.com/huajian_xin/status/1917603640124363090) 庆祝 Prover 项目扩展至 **671B**，并向 DeepSeek 的同事表示感谢。
- **Meta 的 Llama 更新**：[@AIatMeta](https://twitter.com/AIatMeta/status/1917353526088589409) 报告称 **Llama 的下载量已达到 12 亿次**，其中大部分是 Llama 的衍生模型。[@mervenoyann](https://twitter.com/mervenoyann/status/1917503204826255730) 宣布发布 **Llama Guard 4 和 Prompt Guard 2 模型，用于过滤模型输入/输出并防止越狱**。
- **小米的 MiMo-7B**：[@_akhaliq](https://twitter.com/_akhaliq/status/1917410882939715608) 和 [@omarsar0](https://twitter.com/omarsar0/status/1917582720341008814) 讨论了 **小米 MiMo-7B** 的发布，这是一个开源推理模型，采用了在 25 万亿 Token 上训练的 Multi-Token-Prediction 技术。

**AI 模型评估与排行榜**

- **Chatbot Arena 排行榜的问题**：[@karpathy](https://twitter.com/karpathy/status/1917546757929722115) 讨论了 **LMArena 排行榜** 的问题，指出 Arena 分数与现实世界表现之间存在差异，并建议将 @OpenRouterAI 的 LLM 排名作为一种潜在的替代评估方案。[@random_walker](https://twitter.com/random_walker/status/1917516403977994378) 称 LMArena 是垃圾（slop）且存在偏见。[@sarahookr](https://twitter.com/sarahookr/status/1917547727715721632) 分享了一项跨机构合作研究，强调了在 @lmarena_ai 上维持公平评估的难度。
- **关于数据访问和测试的担忧**：[@sarahookr](https://twitter.com/sarahookr/status/1917547738553803018) 指出 **Arena 数据访问存在巨大差异，61.3% 的数据流向了闭源模型提供商**。[@BlancheMinerva](https://twitter.com/BlancheMinerva/status/1917445722380681651) 链接了同一项 @Cohere_Labs 的工作，并指出了维持公平评估的困难。[@clefourrier](https://twitter.com/clefourrier/status/1917488919450374383) 指出这导致了对垃圾数据的过拟合，并表示闭源公司拥有不公平的数据访问权限。[@sarahookr](https://twitter.com/sarahookr/status/1917547733994594420) 强调了隐藏测试的问题。
- **LMArena 的回应**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1917668731481907527) 对批评做出了回应，称该文章包含事实错误和误导性陈述，包括关于模型提供商待遇不平等以及 LMArena 模拟方面的说法。他们还链接了其政策和实际统计数据。

**AI Agents 和工具的应用**

- **AI 在编程与开发中的应用**：[@LangChainAI](https://twitter.com/LangChainAI/status/1917646746798416121) 宣布与 @UiPath 建立合作伙伴关系，以促进企业自动化中 AI agents 的构建、部署和观测。[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1917602387381924173) 重点介绍了一门关于 LLMs 作为操作系统的更新课程，重点关注 agent 记忆和构建自适应 AI 系统。
- **AI 在搜索与信息检索中的应用**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1917977286713758073) 宣布 **Perplexity 现在可以对 WhatsApp 消息进行事实核查**，为转发内容提供即时验证。[@alexalbert__](https://twitter.com/alexalbert__/status/1917973599044116576) 讨论了这些集成。[@karpathy](https://twitter.com/karpathy/status/1917961248031080455) 在一次 vibe coding 黑客松中发现，如今构建和部署一个完整的 Web 应用仍然是一个痛苦的过程。
- **AI 用于机器人与自动化**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1917593514566901800) 介绍了 Summarize, Analyze, Synthesize (SAS) 提示词，使机器人能够通过交互和学习实现自我提升。

**AI 安全、伦理与负责任的发展**

- **OpenAI 的 GPT-4o 回滚**：[@OpenAI](https://twitter.com/OpenAI/status/1917411480548565332) 宣布回滚上周的 GPT-4o 更新，原因是其表现出过度讨好和顺从的行为，并提供了行为更平衡的早期版本访问权限。[@nearcyan](https://twitter.com/nearcyan/status/1917449708647375159) 批评这一公告是在撒谎。
- **加强控制的必要性**：[@jackclarkSF](https://twitter.com/jackclarkSF/status/1917629784940597514) 强调需要对 AI 基础设施进行强有力的控制，以防止关键生产能力外流。[@johnschulman2](https://twitter.com/johnschulman2/status/1917487672983183433) 强调需要消除一些导致偏好的不良因果因素。

**AI 教育与学习**

- **编程与 AI 教育**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1917985792607363189) 讨论了教授 AI 辅助编程的重要性，强调了教师和 AI 在 K-12 教育中的作用。[@alexalbert__](https://twitter.com/alexalbert__/status/1917603519227650533) 强调了学习编程对于人机协作的重要性。
- **对 ML 团队的建议**：[@ProfTomYeh](https://twitter.com/ProfTomYeh/status/1917634404903539022) 就如何成为 AI/ML 团队中不可或缺的人才提供了建议，强调了数学直觉、白板图表以及对人的理解。

**幽默与杂项**

- **牙医询问 p(doom)**：[@AmandaAskell](https://twitter.com/AmandaAskell/status/1917770005988663412) 说她的牙医问她的 p(doom)（AI 毁灭人类的概率）是多少。
- **AI 作为代码助手**：[@sarahcat21](https://twitter.com/sarahcat21/status/1917649137543377235) 说 Cursor 曾尝试给 @ekzhang1 建议一行代码，到现在还在道歉。
- **韩国炒年糕店**：[@dylan522p](https://twitter.com/dylan522p/status/1917719768066363629) 分享了一个在韩国的轶事，提到他给一位老妇人买了一些烟，好让她接受他的刷卡支付，然后为他提供餐食。

---

# AI Reddit 摘要

## /r/LocalLlama 回顾

### 1. Phi 4 Reasoning 模型发布与讨论

- [**微软刚刚发布了 Phi 4 Reasoning (14b)**](https://huggingface.co/microsoft/Phi-4-reasoning) ([热度: 641, 评论: 126](https://www.reddit.com/r/LocalLLaMA/comments/1kbvwsc/microsoft_just_released_phi_4_reasoning_14b/)): **微软发布了 Phi-4 Reasoning (14B)，这是一个在离线数据集上训练的静态模型，据报告其数据截止日期达到了 2025 年 3 月，这表明数据集可能包含未来日期内容，或者将截止日期前移作为一种特性。早期的对比兴趣集中在 Qwen 3 30B MoE 上，暗示了对其推理和通用性能基准的高期望。提供了社区上传的 'phi-4-mini-reasoning' 和 'phi-4-reasoning-plus' 的 GGUF (Quantized General Unstructured File Format) 转换链接：[Phi-4-mini-reasoning GGUF](https://huggingface.co/unsloth/Phi-4-mini-reasoning-GGUF)，[Phi-4-reasoning-plus-GGUF](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUF)，此外还提到了用于优化推理的动态 4-bit safetensors。** 技术型用户对 Phi-4 的表现表示期待，特别是与新的 MoE Qwen 模型相比，并对持续快速的模型发布和量化格式的可用性表现出极大的热情。
    - 技术上的好奇点在于 Phi-4 Reasoning (14B) 与 Qwen 3 30B MoE 的对比，特别是在推理质量和推理基准测试方面，因为后者在 MOE (Mixture of Experts) 领域享有盛誉。
    - Phi-4 Reasoning 在 HuggingFace 上提供多种格式，包括 GGUF 和 4bit safetensors，社区上传了 'mini-reasoning' 和 'reasoning-plus' 版本，以便于部署和量化友好的使用场景 ([Hugging Face 链接](https://huggingface.co/unsloth/Phi-4-mini-reasoning-GGUF), [Phi-4-reasoning-plus-GGUF](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUF))。
    - 一位用户观察到该模型处理了 32k token 的推理（可能是上下文长度），并注意到在思考过程中出现了重复的正确回答。这指向了长上下文生成过程中潜在的注意力模式、冗长或能量消耗问题——这是优化中的一个相关关注点。
- [**Phi 4 Reasoning**](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/phi_4_reasoning.pdf) ([热度: 114, 评论: 12](https://www.reddit.com/r/LocalLLaMA/comments/1kbvrgs/phi_4_reasoning/)): **微软的 [Phi-4-reasoning](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/phi_4_reasoning.pdf) 是一个 14B 参数的语言模型，它利用监督微调 (SFT) 处理精心筛选、多样化的重推理提示集，并结合来自 o3-mini 输出的额外数据，在数学、科学和规划基准测试中以更小的参数预算实现了 SOTA 性能（与 DeepSeek-R1 和 Gemini 2 Flash 等模型持平或超越）。增强版的 Phi-4-reasoning-plus 变体使用基于结果的强化学习 (RL) 进一步提升推理性能，并在新的 NP-hard、算法和规划任务上进行了测试，强调了数据策展和有针对性的 SFT/RL 可以使小模型具备出色的推理泛化能力和效率。该发布还针对推理 LLM 提出了更严格、方差感知的评估协议，以解决小基准测试集敏感性的问题。** 评论者注意到微软与 OpenAI 关系的直接益处（使用 o3-mini 输出进行 SFT），紧凑型模型对 Edge AI 的快速演进和影响，以及微软在大规模应用这些进步的能力，详见相关的 [Azure 博客文章](https://azure.microsoft.com/en-us/blog/one-year-of-phi-small-language-models-making-big-leaps-in-ai/)。
    - 一个关键的技术点是，Phi-4 的推理能力在开源模型中可能是独特的，因为它直接从 OpenAI 的 O 系列模型中进行训练，这暗示了推理技能的转移和知识蒸馏，而这些通常是闭源模型所保留的。
    - 链接的 Microsoft Azure 博客文章提供了更深层的技术背景，强调了 Phi 的进展，并声称 *小语言模型* (SLMs) 正在取得重大的 AI 突破，由于其计算效率和性能平衡，这可能与设备端 (Edge AI) 应用高度相关。
    - 讨论表明，微软的资源使其能够有效地为现实世界的 Edge AI 场景部署像 Phi 这样的 SLM，利用高效的推理性能，而大型模型在这些场景下会因延迟和硬件限制而不切实际。

### 2. Qwen 3 模型：印象与能力

- [**我们跨越了界限**](https://www.reddit.com/r/LocalLLaMA/comments/1kc10hz/we_crossed_the_line/) ([Score: 651, Comments: 132](https://www.reddit.com/r/LocalLLaMA/comments/1kc10hz/we_crossed_the_line/)): **楼主（OP）报告称 QWEN3 32B LLM 现在能够解决他们所有的编程需求——这些任务以前需要访问 ChatGPT 或 Grok (version 3) 等领先的商业 LLM。他们强调了该模型的本地可部署性和能力，暗示开源或本地托管模型在编程助手用例的性能上有了重大飞跃。** 顶层评论者正在寻求进一步的技术细节：一位询问楼主的编程专业知识以衡量模型的实用性，另一位要求与 30b-a3b 模型进行对比，第三位则呼吁提供具体的任务示例，以便更客观地衡量性能。
    - 几位评论者要求澄清评估 32B 模型时所使用的具体编程任务和示例，强调需要详细的 Benchmark 或任务描述来有意义地评估性能。这突显了技术社区对可复现、具体的 Benchmark 的偏好，而非轶事式的陈述。
    - 有人提出了关于将 32B 模型与 30B-A3B 模型进行比较的技术咨询，特别建议在两者上运行同一组任务并报告相对性能。这强调了在相同任务领域内进行直接的模型对模型 Benchmark 以进行公平评估的重要性。
    - 一位评论者询问了实现细节：具体使用了哪种 Quantization、Hugging Face 仓库以及用于运行模型的 Inference Server。还提到了对测试 Unsloth 的 128k Context-length 版本的兴趣，暗示了对比评估方案以及对部署/推理优化的关注。
- [**令人印象深刻的 Qwen 3 30 MoE**](https://www.reddit.com/r/LocalLLaMA/comments/1kc6hgn/impressive_qwen_3_30_moe/) ([Score: 107, Comments: 38](https://www.reddit.com/r/LocalLLaMA/comments/1kc6hgn/impressive_qwen_3_30_moe/)): **该帖子讨论了 Qwen 3 30 MoE 模型的跨语言翻译能力，特别指出其在西班牙语、荷兰语、德语和英语方面的高准确性，甚至能令人信服地处理地区方言。评论者强调 MoE (Mixture of Experts) 架构实现了高效的纯 CPU 推理，但也提醒特定语言（如德语）仍可能出现错误，例如语法性别错误或受英语影响的不自然措辞。** 技术评论者对非英语使用中复杂推理和整体“智能”可能出现的退化表示担忧，建议针对逻辑任务进行额外测试，并警告不要将 Benchmark 过度泛化为实际表现（即“benchmaxxing”）。
    - Qwen 3 30 MoE 被认为是纯 CPU 推理（尤其是支持 AVX2 的 CPU）中速度最快且能力最强的模型之一，即使是 Quantized 版本（如 q2_K_L）在配备 16GB RAM 的低端硬件上也能达到约 '7 t/s'。与 llama.cpp 上的 4B Q_4_K_M 等其他模型的对比显示，Qwen 在处理复杂且精确的查询时速度更快，且提供的答案更准确。
    - 尽管具有出色的多语言能力（在波兰语中表现突出），但用户报告该模型在德语等其他语言中会出现典型错误：性别问题、英语化措辞，以及在英语之外的复杂/逻辑任务上性能下降。这表明非英语训练数据覆盖和泛化方面存在局限性。
    - 该模型对硬件受限的用户具有很强的实用价值：它仅需要“充足的 RAM、支持 AVX2 的 CPU 以及约 10GB 的空间”即可实现完整的 LLM 功能，在速度和内存效率方面都优于同类模型。

- [**Qwen 3 4B 是未来，女士们先生们**](https://i.redd.it/2aw947hyi3ye1.png) ([评分: 318, 评论: 66](https://www.reddit.com/r/LocalLLaMA/comments/1kc016i/qwen_3_4b_is_the_future_ladies_and_gentlemen/)): **图片展示了一个问答场景，其中 Qwen 3 4B（阿里巴巴/Qwen 开发的 40 亿参数大语言模型）准确地推断出 9.9 大于 9.11，并提供了清晰的分步解释，展示了许多早期小型开源 LLM 所缺乏的数字解析能力。帖子和评论强调，虽然此类测试现在已进入“训练数据”中，可能不再新颖，但其表现标志着基础算术和推理能力的成熟，特别是对于这种规模的模型而言。评论者将其与 Llama 3.1 等模型进行了比较，暗示 Qwen 3 4B 作为 4-8B 参数范围内的开源替代方案具有竞争优势。** 最热门的技术评论表达了对基础推理测试（如数字解析或拼写）已变得微不足道且被过度强调的担忧，并呼吁建立更具挑战性和新颖性的 Benchmark。另一位评论者对 Qwen 3 的 8B 版本表现充满期待，并指出 Qwen 3 在该领域可能与 Llama 3.x 竞争甚至超越。
    - 讨论集中在需要*更稳健且更有意义的 Benchmark*来评估像 Qwen 3 4B 这样的模型，因为许多流行的测试提示（例如：数字母、比较小数）经常出现在训练数据中，可能无法准确衡量现实世界的推理能力。
    - Qwen 3 4B（及其 8B 版本）与 Llama 3.1 进行了对比，暗示随着 4-8B 规模的模型变得越来越强大，Qwen 代表了一个强有力的开源替代方案。
    - 一位用户分享了在 iPhone 16 Pro Max 上运行 Qwen 3 4B 的经验，指出推理速度“足够快”，表明在消费级移动硬件上使用先进小型模型的显著效率和可行性。
- [**开发者偏好的模型。**](https://i.redd.it/mg9ey4l4b7ye1.jpeg) ([评分: 103, 评论: 41](https://www.reddit.com/r/LocalLLaMA/comments/1kcdpce/the_models_developers_prefer/)): **这张源自 Cursor AI 的图片展示了两份名单：2025 年 4 月的“Cursor 上最受欢迎”和“Cursor 上增长最快”。受欢迎的模型包括 Claude 3.7 Sonnet 和 Gemini 2.5 Pro，而增长最快的模型则突出了 o3、o4-mini 和 DeepSeek V3.1。这张快照提供了 Cursor 开发者用户群中当前 LLM 使用趋势的横截面，凸显了既有模型和获得关注的新晋模型。** 评论指出，Cursor 的基础设施使本地模型的使用变得复杂，可能导致结果偏向 API/托管模型。此外，人们对普适性也持怀疑态度，因为数据仅反映了 Cursor 用户的偏好，而其他排行榜（如 Aider）可能会显示不同的趋势。
    - 一位评论者指出，由于网络限制（具体需要公网 IP 或代理），通过 Cursor 运行本地模型具有挑战性，这可能会使关于 Cursor 开发者中流行模型的偏好或统计数据产生偏差。
    - 另一个提出的技术点涉及像 o3 这样模型的高昂使用成本（推测为 OpenAI 的 GPT-3.5 或 GPT-4 级别），评论指出其昂贵的费用使其难以广泛使用，从而影响了它们在开发者工作流中的采用。
    - 一位参与者区分了不同的模型排行榜，指出个人更倾向于“Aider 排行榜”而非 Cursor 的排行榜，暗示模型受欢迎程度或偏好指标会因语境和平台的不同而有显著差异，这影响了在关于模型选择的技术讨论中应如何解读结果。

### 3. 新型模型与训练方法发布 (TTS/ASR, KL 优化)

- [**比 Whisper3-large 参数更少但性能更强的新型 TTS/ASR 模型**](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) ([Score: 164, Comments: 44](https://www.reddit.com/r/LocalLLaMA/comments/1kcdxam/new_ttsasr_model_that_is_better_that/)): **NVIDIA 发布了 Parakeet-TDT-0.6B-v2 模型，这是一个语音转文本 (STT/ASR) 系统，其性能超越了 Whisper-v3 Large，且参数量更少 (`~0.6B`)。该模型提供字符/单词/句子级别的时间戳，但仅限于英语；未描述对文本转语音 (TTS) 的支持。用户注意到与大语言模型 (LLMs) 相比，其训练计算需求较低，并期待其支持说话人识别。** 讨论中质疑了 TTS 的分类，澄清其为 ASR/STT 模型。技术兴趣集中在细粒度时间戳的可用性，并对其相对于 LLMs 的效率感到惊讶。
    - 该模型支持字符、单词和句子级时间戳，这可能对说话人识别等下游任务产生重大影响——用户指出，如果增加进一步的分段 (diarization) 支持，该模型将特别有用。
    - 一个关键亮点是模型的数据组合：10,000 小时的人工转录数据（涵盖 LibriSpeech, Fisher Corpus, VoxPopuli, Europarl-ASR 等数据集），以及来自 YouTube-Commons, YODAS 和 Librilight 的 110,000 小时伪标签数据。评论者指出，与 Whisper 的训练语料库相比，这种数据构成更加全面且优越，可能有助于获得更好的泛化能力和性能。
    - 几位用户观察到，该模型以比 LLMs 显著更少的计算资源实现了其结果，强调了其训练流水线与 Whisper-3 等计算密集型模型相比的高效性和深思熟虑的设计。
- [**新型训练方法显示 80% 的效率提升：递归 KL 散度优化 (Recursive KL Divergence Optimization)**](https://arxiv.org/abs/2504.21707) ([Score: 139, Comments: 13](https://www.reddit.com/r/LocalLLaMA/comments/1kbytzk/new_training_method_shows_80_efficiency_gain/)): **据原始帖子报道，一种名为递归 KL 散度优化 (Recursive KL Divergence Optimization) 的新训练方法实现了约 80% 的效率提升。该方法引用了基于图像数据集 (CIFAR-10, CIFAR-100, STL-10) 的基准测试和用例，如相应 notebook 中使用的 `PIL` 以及相关研究论文中的引用所示，但该方法被分享在 LocalLLaMA 版块，引发了关于其对 LLMs 与图像模型适用性的疑问。目前没有针对流行的开源模型训练器（如 kohya, SimpleTuner）的直接实现，限制了即时的可复现性和跨领域验证。** 评论者质疑该技术是否能有效地用于持续微调场景，并寻求关于其对 LLMs 与图像数据集适用性的澄清，表明需要通用的实现和基准测试。此外，人们对论文的音频摘要（例如：notebooklm 音频链接）以进行更广泛传播也表现出兴趣。
    - StableLlama 指出，尽管由于发帖版块的原因，该方法在 LLMs 的背景下被讨论，但原始论文和随附代码主要使用 CIFAR-10, CIFAR-100 和 STL-10 等图像数据集，实现中引用了图像库 `PIL`。这引起了关于其对语言模型适用性的困惑。他们还询问了与 kohya 或 SimpleTuner 等开源训练框架的集成情况，建议需要对 RLKD 在图像之外的现实任务中进行基准测试。
    - Revolaition 强调了递归 KL 散度优化 (RKLD) 声称的实际益处：提高训练效率，特别是对于微调，具有缩短训练时间、降低计算成本和降低硬件需求的潜力。这些点表明，如果得到验证，该方法对于资源受限的环境具有广阔的应用前景。

## 其他 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. CivitAI 成人内容清理及社区替代方案

- [**CIVITAI 即将清除所有成人内容！（立即备份！）**](https://www.reddit.com/r/StableDiffusion/comments/1kbxq93/civitai_is_going_to_purge_all_adult_content/) ([Score: 656, Comments: 346](https://www.reddit.com/r/StableDiffusion/comments/1kbxq93/civitai_is_going_to_purge_all_adult_content/)): **CivitAI 引入了一套全新的 AI 驱动内容打标与审核系统，据传名为 Clavata。用户声称该系统产生了有缺陷的内容分类——具体而言，是将体液（例如将精液误认为呕吐物）和无害手势（例如手指或口交手势）错误标记为违规内容。该帖子警告称，在更新的服务条款下，成人内容面临大规模自动屏蔽和潜在删除的风险，并引用了对显性和非显性材料的测试上传结果，显示这些内容被不成比例地屏蔽。担忧集中在 AI 系统的技术局限性和误报，可能导致对成人或边缘内容过度或无意的审查。** 评论对这些说法的煽动性提出了质疑，指出一些被屏蔽的 NSFW 内容在元数据审核后已解除屏蔽，并强调缺乏证据表明存在计划中的全站清除行动。怀疑者指出，发帖者（OP）可见的露骨上传内容仍然可以访问，这表明虽然可能存在误报，但关于即将进行大清洗的说法尚无事实依据。
    - 一位用户澄清说，CivitAI 的新政策围绕自动化屏蔽（通过基于元数据的审核）展开，而非大规模清除 NSFW 内容。他们报告称，在政策变更后，他们的一些 NSFW 视频被自动屏蔽，但在提交正确的元数据后随后被解除屏蔽，这表明系统依赖于内容分类而非直接禁令。他们还提到屏蔽系统中可能存在误报，但目前没有证据表明会大规模删除显性内容。
    - 另一位评论者讨论了模型的备份策略，提到为 Stable Diffusion 1.5 XL 和 pony 模型备份了超过 `200+ GB` 的 LoRA，并使用了自动化脚本（引用了一个可能现有的 GitHub 项目）进行按类别下载。该用户提到了他们的本地存储容量（`~10 TB`），并指出主要的瓶颈在于 CivitAI 的下载速度，而非他们的光纤网络或硬盘空间。
    - 有人将其与过去的互联网审查事件（如 Tumblr 的 NSFW 禁令）类比，强调了对于对成人内容生成感兴趣的人来说，模型/数据保存的重要性，因为用户正在预见潜在的政策变化，积极下载并存档他们的数据集。
- [**CivitAI 即将清除的成人内容首选替代方案是什么？我们该把东西搬到哪里？我们需要 B 计划！**](https://www.reddit.com/r/StableDiffusion/comments/1kc4jqc/what_is_the_preferred_substitute_for_the_adult/) ([Score: 195, Comments: 192](https://www.reddit.com/r/StableDiffusion/comments/1kc4jqc/what_is_the_preferred_substitute_for_the_adult/)): **Reddit 用户正在讨论托管成人/NSFW Stable Diffusion 模型的替代方案和应急计划，因为 CivitAI 正准备移除此类内容。强调的技术解决方案包括即将把存档模型和辅助数据大规模上传到 Torrent 平台（例如 1337x）、手动下载/保存元数据，以及使用专门的镜像站，如 [civitaiarchive.com](http://civitaiarchive.com/) 和 [diffusionarc.com](http://diffusionarc.com/)，这些站点支持 Torrent 下载和元数据摄取（即使是已删除的模型）。[这篇文章](https://www.reddit.com/r/StableDiffusion/comments/1k7dvfb/in_reguards_to_civitai_removing_models/)（[此处也有存档](https://archive.ph/https://www.reddit.com/r/StableDiffusion/comments/1k7dvfb/in_reguards_to_civitai_removing_models/)）引用了更集中的替代方案和资源列表。** 具有技术实质性的观点指出了移除 NSFW 内容的平台的历史先例，这通常会导致用户迁移和平台衰落；用户强调需要去中心化托管，以防止单点故障。
    - 用户正在为 CivitAI 中即将被移除的成人模型创建 Torrent 种子。有人提到正准备在 “1337” 上发布 Torrent 并存档相关的元数据，这表明了一种用于模型保存的去中心化点对点备份策略。
    - 替代托管网站正在兴起，[civitaiarchive.com](http://civitaiarchive.com/) 被认为是一个专门的替代方案，用于托管因 CivitAI 政策变化而被取代的模型（尤其是 LoRA）。目前正在努力开发进一步的解决方案，表明迁移格局呈现碎片化状态。

- Reddit 上有一些共享资源和汇总帖子（例如 [这个帖子](https://www.reddit.com/r/StableDiffusion/comments/1k7dvfb/in_reguards_to_civitai_removing_models/)，以及 [存档版本](https://archive.ph/https://www.reddit.com/r/StableDiffusion/comments/1k7dvfb/in_reguards_to_civitai_removing_models/)），列出了受 CivitAI 新限制影响的模型的代码库、镜像站点以及当前的访问或备份权宜之计。
- [**仅限 Civitai 种子**](https://www.reddit.com/r/StableDiffusion/comments/1kcb7ge/civitai_torrents_only/) ([得分：120，评论：32](https://www.reddit.com/r/StableDiffusion/comments/1kcb7ge/civitai_torrents_only/))：**一个新的免费工具 [datadrones.com](http://datadrones.com/) 允许用户生成种子文件并索引 LoRA 模型（用于扩散网络的低秩自适应模型，Limited Rank Adaptation models for diffusion networks），用于 Civitai 风格的内容共享，旨在确保去中心化、持久化的分发，无需金钱交易或中央托管（为了简单和合规，没有 UI 且仅进行最少的文件扫描）。该工具默认使用单个公共 Tracker，每个 LoRA 的最大文件大小限制为 2GB，并利用基于哈希的重复检查来强制执行上传的唯一性。创作者明确避开了 HuggingFace 等服务，因为存在政策风险，并鼓励社区做种以保证长久运行，同时表示不鼓励使用私有 Tracker。搜索功能和改进的扫描功能正在开发中；由于访问门槛，Usenet 支持被视为低优先级。** 评论者提到了其他的 Civitai 存档网站（[diffusionarc.com](http://diffusionarc.com/)、[civitaiarchive.com](http://civitaiarchive.com/)）作为平行的保存项目。辩论涉及了简单性与功能集之间的平衡，并对旧的去中心化文件共享方法（eMule/edonkey）表示了怀旧。
    - 提到了两个替代网站 https://www.diffusionarc.com/ 和 https://civitaiarchive.com/，它们是专门专注于 Civitai 内容保存和存档的项目，可能提供不同于种子分发的访问方法和存档策略。
    - 一位维护者指出，网站上已经实现了搜索功能，并计划下一步添加目录功能，这表明正在进行技术开发，以实现更强大的文件导航和用户访问功能。

### 2. 近期 AI 模型与技术发布及基准测试

- [**Google 发布 Ironwood 芯片，比全球最强大的超级计算机快 24 倍。这是与 NVIDIA 竞争的开始吗？**](https://v.redd.it/wnldw6ib0ywe1) ([得分：160，评论：24](https://www.reddit.com/r/singularity/comments/1kcdlhg/google_launches_the_ironwood_chip_24x_faster_than/))：**Google 新发布的 Ironwood 芯片据称在 9216 个 TPU 组成的集群中可提供 42.5 exaflops 的 FP8 计算能力，Google 声称这比当前的超级计算机冠军 El Capitan 快 24 倍以上。然而，Ironwood 不提供外部购买，似乎仅用于 Google 自己的数据中心，这引发了关于其是否与 NVIDIA 存在直接竞争的质疑。目前尚未提供进一步的架构细节、独立基准测试或具体的实现细节，性能声明仍未经证实。** 评论者指出，由于不向外部开放，Ironwood 目前并不与 NVIDIA 或其他硬件供应商直接竞争，现阶段不应被视为竞争对手。
    - 讨论强调，Google 的“快 24 倍”声明是基于 9,216 个 Ironwood TPU 的全规模集群，据报道可提供 42.5 exaflops 的 FP8 性能，这是与当前顶级超级计算机 El Capitan 的 FP8 性能指标进行的对比。这种比较取决于特定的数据类型（FP8），而不是更传统的 FP32 或 FP64，强调了该芯片面向深度学习和 AI 任务的定位，而非通用型 HPC。
    - 多位评论者指出，Ironwood TPU 并不代表对 NVIDIA 硬件的直接市场威胁，因为它们在商业上不可用。Google 的 TPU 技术缺乏公开市场销售，这意味着尽管其技术实力雄厚，但并不会与广泛可供购买和集成的 NVIDIA GPU 产品形成真正的竞争关系。

- [**F-Lite - 10B 参数的图像生成模型，在 80M 张版权安全的图像上从零开始训练。**](https://huggingface.co/Freepik/F-Lite) ([Score: 129, Comments: 47](https://www.reddit.com/r/StableDiffusion/comments/1kc2j5g/flite_10b_parameter_image_generation_model/)): **该帖子发布了 "F-Lite"，这是一个拥有 10B 参数的图像生成模型，据报道是在 80M 张版权安全的图像上从零开始训练的。引用的 [Hugging Face 模型页面](https://huggingface.co/Freepik/F-Lite) 目前由于反复出现** `HTTP 429 Too Many Requests` **错误而无法访问，且目前未提供技术文档或模型详情。** 热门评论批评了该模型报告的图像质量（特别是解剖学准确性较差），并质疑“受限”或受限数据集模型的价值，认为与训练更通用、能力更强的模型相比，这种资源支出是浪费的。
    - 几条评论指出 F-Lite 的解剖结构生成能力很差，特别提到其 *“对解剖结构的理解看起来非常糟糕”*，这表明其训练或数据集策选存在重大的技术局限。这一观察结果意味着该模型在处理对人类极其敏感的特征时表现挣扎，而这通常是图像生成基准测试的关键。
    - 严格版权安全训练数据的价值在技术上受到质疑，因为 *“风格本身无论如何都无法受到版权保护”*。这引发了关于数据集构成的争论：受版权限制的语料库是否实质上限制了模型的表现力，或者对于大多数用例来说，仅与输出忠实度略微相关？
    - 提出了一个关于硬件效率的问题：像 F-Lite 这样拥有 10B 参数的模型是否可以 *“在 8GB VRAM 上运行”*，这是本地和消费级部署相对于需要大量内存资源的大型模型的技术考量。
- [**Livebench 已彻底沦为笑话。GPT-4o 在编程方面的排名高于 o3-High 和 Gemini 2.5 Pro？...**](https://i.redd.it/o28kdmdxq1ye1.jpeg) ([Score: 205, Comments: 62](https://www.reddit.com/r/singularity/comments/1kbt07z/livebench_has_become_a_total_joke_gpt4o_ranks/)): **图片展示了来自 Livebench ([https://livebench.ai](https://livebench.ai/)) 的排行榜，展示了主要 AI 模型的编程性能评分，如 "o4-Mini High"、各种 OpenAI GPT-4o/3 模型以及 Google Gemini 2.5 Pro。值得注意的是，GPT-4o 甚至像 o3-Medium 这样容量较低的模型排名都高于 o3-High 等更高级别的模型，并且显著领先于 Google Gemini 2.5 Pro（得分：71.08），这让人质疑 Livebench 编程评估的可靠性。这与目前关于 Livebench 编程基准测试的方法论和有效性的持续争论相吻合。** 技术评论者普遍不信任 Livebench 的编程评分，理由是模型排名不合逻辑（例如，Sonnet 的得分低于能力较弱的模型），并认为需要更严谨的编程基准测试。一些人还对该平台的管理和数据可信度表示怀疑。
    - 多位用户批评了 Livebench 的编程基准测试；具体而言，当 **GPT-4o** 的排名高于 **Ollama o3-High** 或 **Gemini 2.5 Pro**，以及像 Sonnet 这样的“思考型”模型得分低于非思考型模型时，他们质疑其结果的有效性，突显了对评估方法论的严重担忧。
    - 一位用户指出，Livebench 似乎偏向于竞赛编程风格的任务，这可能无法反映模型的广泛编程能力，并可能解释了其与通常更强的“思考型”模型相比排名不一致的原因。
    - 存在对整个合成基准测试的怀疑态度，几条评论断言，没有任何评估模型（包括备受推崇的 Gemini 2.5）在现实世界的编程、写作或上下文保留方面的表现像基准测试所暗示的那样糟糕或不一致。

### 3. 指令式图像编辑与 UI 集成发布

- [**Chroma 现在已正式在 ComfyUI 中实现。以下是运行方法。**](https://www.reddit.com/r/StableDiffusion/comments/1kc7jwq/chroma_is_now_officially_implemented_in_comfyui/) ([Score: 221, Comments: 90](https://www.reddit.com/r/StableDiffusion/comments/1kc7jwq/chroma_is_now_officially_implemented_in_comfyui/)): **根据 [pull request](https://github.com/comfyanonymous/ComfyUI/pull/7355)，新模型 Chroma 现已正式集成到 ComfyUI 中。安装指南包括将 [ae.sft VAE](https://huggingface.co/Madespace/vae/blob/main/ae.sft)、[T5XXL FP16 text encoder](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors)（需要 >9GB VRAM）和 [Chroma UNet](https://huggingface.co/lodestones/Chroma/tree/main)（BF16 模式需要 >19GB VRAM）放入相应文件夹。对于 VRAM 有限的用户，支持 [Chroma GGUF 变体](https://huggingface.co/silveroxides/Chroma-GGUF/tree/main)，并辅以 [ComfyUI-GGUF custom node](https://github.com/city96/ComfyUI-GGUF) 以及可选的用于 RAM offloading 的 [ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU)；帖子中链接了 workflow 和风格演示（视频游戏、动漫、写实）。** 评论中的关键技术反馈建议避免使用原帖的 workflow 以获得最佳效果：具体而言，弃用 RescaledCFG，使用 Euler 或 UniPC 等标准 sampler，将 CFG 限制在 3-4，简化 negative prompts，将 prompt 权重限制在 1.2 以下，更新 ComfyUI，并正确设置 Chroma CLIP loader。据注，仅需 30 steps 即可获得不错的速度/质量；通过这些调整，效果较初始印象有实质性提升。
    - 一位用户提供了 ComfyUI 中 Chroma 的优化指南，指出原始 workflow 效果不佳可能是由于参数设置不理想。关键建议包括：避免使用 RescaledCFG，使用标准 sampler（Euler 或 UniPC），将 CFG 设置为 3-4，保持 prompt 权重最高为 :1.2（因为 ComfyUI 不是 Auto1111），并简化 negative prompts。还强调需要更新 Comfy 并将 clip loader 设置为 Chroma 以实现全部功能。从 30 steps 开始即可实现不错且快速的结果。
    - 实现细节：在 ComfyUI 中成功使用 Chroma 需要用户更新 ComfyUI 并选择 Chroma 作为 clip loader，这表明该更改并非默认启用，可能需要用户显式操作。
- [**In-Context Edit (ICEdit) 一种基于上下文生成的指令式图像编辑框架，已开源其 LoRA 权重**](https://www.reddit.com/gallery/1kcbpq8) ([Score: 103, Comments: 11](https://www.reddit.com/r/StableDiffusion/comments/1kcbpq8/incontext_edit_an_instructional_image_editing/)): **ICEdit 引入了一个基于指令的图像编辑框架，利用基于 LoRA 的权重（[Hugging Face 模型链接](https://huggingface.co/sanaka87/ICEdit-MoE-LoRA)），支持多轮（multi-turn）和单步编辑，在物体添加、颜色更改、风格迁移和背景编辑等任务上具有高效率。该方法包含 HuggingFace demo（[ICEdit demo](https://huggingface.co/spaces/RiverZ/ICEdit)）和 ComfyUI workflow 文件（[workflow JSON](https://github.com/user-attachments/files/19982419/icedit.json)）等资源，表明其在 pipeline 部署方面具有实际的模块化集成潜力。发布材料中未明确说明 VRAM 要求。** 技术讨论对提供的 ComfyUI workflow 的功能正确性提出了质疑，资深用户指出它看起来“完全乱套了”，暗示 workflow 的配置或兼容性存在问题。还有一个关于 VRAM 要求的技术咨询，特别是关于 16GB 配置的适用性，官方文档尚未对此做出回应。
    - 一位用户对 LoRA 权重与 "Comfy" (ComfyUI) 的 workflow 集成表示担忧，认为其 workflow 可能存在问题，或者不像预期的那样即插即用。
    - 另一条评论询问了运行该模型的 VRAM 要求，特别是 16GB VRAM 是否足够，表明在显存适中的 GPU 上进行有效部署和资源消耗方面存在不确定性。
    - 强调了一个关于输出图像分辨率的技术限制，模型或 Web demo 被限制在强制 512 像素宽度。此外，评论者指出 Web demo 的输出质量高度不稳定，暗示可能存在稳定性或推理问题。

---

# AI Discord 回顾

> 由 chatgpt-4o-latest 生成的摘要之摘要
> 

**1. Phi-4 推理模型发布**

- **微软加速推进 Phi-4 推理模型**：**Microsoft** 发布了名为 [**Phi-4-reasoning**](https://huggingface.co/microsoft/Phi-4-reasoning) 的新模型，这是一款专为推理任务设计的先进 14B 参数 LLM，同时可在 [Unsloth 的本地 GGUF 版本](https://huggingface.co/unsloth/Phi-4-mini-reasoning-GGUF)获取，并关联至[此博客文章](https://x.com/UnslothAI/status/1917806961825046672)。
    - 社区情绪总体积极，指出其性能**与 GPT-3.5 尺寸的模型相当**，并有推测称 Phi-4 是使用 **OpenAI Chain-of-Thought 输出**进行训练的；更多信息可在 [YouTube 视频](https://www.youtube.com/watch?v=5aN4Xg0VvCs)和 [arXiv 论文](https://arxiv.org/abs/2504.21318)中探索。
- **Phi-4 推理模型通过 Unsloth 实现本地运行**：**Unsloth AI** 通过在 [Hugging Face](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUFChatGPT) 上发布 GGUF 格式版本，使 **Phi-4-reasoning** 模型可以本地访问，为运行离线模型的用户降低了实验门槛。
    - 该模型可运行于现有的 **Phi-4 notebooks**，早期测试者称赞其**简洁的推理输出**，引发了与 Phi-3 的对比以及早期的微调讨论。

**2. 扩散语言模型与架构创新**

- **Mercury Coder 凭借扩散模型技术脱颖而出**：**Inception AI** 推出了 [**Mercury Coder**](https://openrouter.ai/inception/mercury-coder-small-beta)，这是首个公开可用的**扩散语言模型 (diffusion language model)**，拥有 **300+ tokens/second** 的速度，并宣传其竞争力可媲美 **GPT-4o Mini** 和 **Claude 3.5 Haiku**。
    - 其**并行 Token 精炼 (parallel token refinement)** 架构（扩散式解码）据称能减少幻觉并提升推理能力；[OpenRouter 分享了细节](https://x.com/OpenRouterAI/status/1917677801211322752)，社区兴趣迅速激增。
- **DSPy 通过化学提示词大幅减少幻觉**：发表在《化学信息与建模学报》上的[一篇新论文](https://pubs.acs.org/doi/10.1021/acs.jcim.4c02322)显示，针对 **TPSA 预测任务**优化 DSPy 程序，使化学推理中的幻觉减少了 **81%**。
    - 结果令社区印象深刻，标志着 DSPy 在**稳定科学推理**方面的潜力，并助力其在 **Amazon Nova 迁移**和 **Meta 的 LlamaCon 发布**中获得更多采用。

**3. Qwen3 模型：突破与 Bug**

- **Qwen3 4B 在同尺寸级别中表现卓越**：**Qwen3 4B** 因在**数学和编程任务**中的出色表现而赢得赞誉，一位用户称 [“Qwen3 4B 击败了该尺寸范围内的所有对手”](https://fixupx.com/suriyagnskr/status/1917731754515013772)。
    - 多个 Discord 频道称赞其**体积小巧但回报巨大**，来自 **ContextArena.ai** 的基准测试进一步证实了其在短上下文长度评估中的优势。
- **Qwen3 的 Norm 层在合并时消失**：在 Qwen3 LoRA 合并中发现了一个 Bug，**q_norm** 和 **k_norm** 层会被静默排除，需要将这些层添加到 `save.py` 中的 **LLAMA_LAYERNORMS**。
    - 用户在数小时的调试后分享了成功的修复方案，强调这仅影响 **Qwen3 而非 Qwen2**，并建议向 **Unsloth** 提交补丁。

**4. 多模型生态对决与扩展基准测试**

- **Context Arena 榜单 Llama 夺冠，但 Qwen 势头强劲**：**ContextArena.ai** 发布了更新的多模型排行榜结果，**Llama 4 Maverick** 在 **128k 上下文**下获得最高 AUC，而 **Qwen3-235B-A22B** 在短上下文下表现更优。
    - **Anthropic** 模型在 **Claude 3.x** 各代中表现稳定，社区成员剖析了当上下文长度接近极限时的模型性能衰减模式。
- **Gemini vs GPT vs Claude 引发地盘争夺战**：在多个社区中，用户争论 **Gemini 2.5 Pro**、**Claude** 和 **GPT-4o** 之间的权衡，引用了 Gemini 在**批判性思维**和后端编程方面的优势，但也抱怨其 UI 和 Token 效率低下。
    - 观点在性价比上产生分歧，一位用户将 GPT 描述为“顺着你语气的虚假朋友”，而将 Gemini 描述为“具有判断力的真正专业人士”，巩固了开发者工作流中的心智占有率之争。

**5. Claude 不断扩展的能力与集成**

- **Claude 跨工具链集成**：**Anthropic 的新 [Claude Integrations](https://www.anthropic.com/news/integrations)** 允许用户在 Claude 的界面中直接连接自定义 SSE 端点，从而为复杂工作流实现实时工具链化。
    - 用户称赞了这份[说明指南](https://x.com/alexalbert__/status/1918047745790914772)，它解释了如何输入自定义 **SSE URL**，相比之前在 Agent 中使用的折中方案，大大简化了集成过程。
- **Claude 连接至 DeepWiki MCP Agent 工作流**：分享了一个强大的新组合：通过 `fetch` 命令将 **Claude** 连接到 **DeepWiki MCP** 服务器，利用该端点在研究场景中进行网页级检索。
    - 用户称该设置具有*重大意义*，并引用了 [DeepWiki 的仓库](https://github.com/regenrek/deepwiki-mcp) 和 **Claude 的新 Agent 端点**，认为它们开启了真正的具有 Agent 能力的、上下文相关的查询链。


---

# Discord: 高层级 Discord 摘要




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 与 O3 引发辩论**：成员们辩论了 **Gemini** 和 **O3** 的相对优劣，观点从 *“客观上 O3 更好，兄弟”* 到攒钱买 **Gemini Ultra** 的玩笑不等。
   - 讨论强调了模型偏好的主观性，即使在有客观基准测试的情况下也是如此。
- **Qwen3 表现出色，Llama 4 在 Context Arena 称霸**：DillonU 在 **Context Arena** 中进行的 **OpenAI-MRCR** 基准测试显示，**Llama 4 Maverick** 在 128k 上下文长度下以最高的 AUC 分数领先，结果可在 [ContextArena.ai](https://contextarena.ai/) 查看。
   - Qwen3-235B-A22B 在较低的上下文长度下表现优异，但在接近其极限时性能迅速下降。
- **Anthropic 在 Context Arena 取得佳绩**：**Context Arena** 现在包含了更多关于 **Anthropic** 的 2needle 测试结果，显示了 **Claude 3.0**、**3.5** 和 **3.7** 之间一致的性能表现，结果可在 [ContextArena.ai](https://contextarena.ai/) 查看。
   - **Claude 3.0 Haiku** 获得了最佳的整体模型 AUC。
- **ChatGPT 的 Deep Research 胜出**：成员们对比了来自 **ChatGPT**、**Claude** 和 **Grok** 的深度研究工具，指出 **ChatGPT** 的深度研究能力超过了 **Grok**，因为 **Grok** 依赖于 **DuckDuckGo**。
   - 共识是 **ChatGPT** 提供了更完善且有效的搜索研究体验。
- **Qwen3 4B 碾压同类竞争对手**：**Qwen3 4B** 模型因其相对于体量的性能而受到赞誉，一位成员表示 [*qwen 3 4b* **碾压** *该尺寸范围内的所有人*]，更多讨论见[此处](https://fixupx.com/suriyagnskr/status/1917731754515013772?t=yQeTFTkCfRkl0ZhQJ2k-tQ&s=19)。
   - 据观察，**Qwen3 4B** 在*数学和编程*任务中表现尤为强劲。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 的 UI 一夜之间发生变化！**：用户报告了昨晚发生的 UI 设计变更，一些人反映 Library 消失以及快捷键发生了变化。
   - 反应不一，部分用户认为新 UI 存在 Bug，而另一些人则保持中立。
- **Qwen 在上下文长度上夺魁！**：成员们发现 **Qwen** 拥有极佳的上下文长度。
   - 它被认为作为免费模型足以与旧款 **ChatGPT** 模型竞争，有些人甚至更倾向于使用它而非付费选项。
- **Zen Browser 的透明化技巧！**：成员们讨论了如何使用 **Zen Browser** 创建透明背景，并推荐了来自 [GitHub](https://github.com/JustAdumbPrsn/Nebula-A-Minimal-Theme-for-Zen-Browser/releases) 的 **Nebula** 主题。
   - 配置步骤各异，一些用户正在 [Zen 的 Subreddit](https://www.reddit.com/r/zen_browser/s/uRWOeML6n8) 寻求帮助。
- **Grok 的图像处理遭遇瓶颈！**：一位成员观察到 [Twitter 上的 **Grok AI**](https://www.rxddit.com/r/grok/s/w5jc52QFj5) 在图像处理方面面临限制。
   - 据他们称，其图像处理能力仅处于 **R1** 水平。
- **特斯拉董事会寻找 CEO！**：分享的一份 [Perplexity 搜索结果](https://www.perplexity.ai/page/tesla-board-seeks-new-ceo-to-r-3JZ4nGLOQ6S40o59qn92Wg)显示，**Tesla** 董事会正在寻找新 CEO。
   - 这一搜索受到关于 **Elon Musk** 角色以及公司未来领导层持续讨论的影响。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama-4-Scout GGUF 缺失图像支持**：用户报告称，**Unsloth** 的 **Llama-4-Scout-17B-16E-Instruct-GGUF** 模型在使用 **Q4_K_XL** 时无法解析图像，尽管文本功能正常。
   - 有人质疑非 Meta 官方工具是否正确编码了图像，并建议检查 tokenizer 配置和 vocab。
- **Microsoft 发布 Phi-4 Reasoning 模型**：**Microsoft** 发布了 **Phi-4-reasoning** 模型，可在 [huggingface.co/microsoft/Phi-4-reasoning](https://huggingface.co/microsoft/Phi-4-reasoning) 获取，被认为在 14B 参数模型中极具竞争力。
   - 该模型可能是在 **OpenAI CoT** 输出上训练的，Unsloth 团队表示它将兼容常规的 **Phi4** notebook。
- **GRPO 使模型能够进行自我解释**：一名成员建议使用 **GRPO** (Generative Reward Policy Optimization) 来训练模型（如 **Claude**），以提高其自我解释能力，特别是在处理描述不准确的问题时。
   - 这可以通过让模型先为自己澄清问题，从而增强其理解和解决复杂问题的能力。
- **Qwen3 GGUF 获得快速修复**：**Qwen3-30B-A3B 128K** GGUF 已重新上传以修复问题，上下文长度值重置为 **32k**，在 LM Studio 中需要 **4** 的 rope 缩放因子。
   - 团队感谢社区帮助识别了该问题，此问题仅影响 **30b-a3b** 模型，而其他 **128k** GGUF 未受影响。
- **Qwen3 的 Norm 层在 LoRA 合并期间消失**：发现了一个问题，即在将训练好的 LoRA 与基础 **Qwen3** 模型合并时，**q_norm** 和 **k_norm** 层未被保存。
   - 修复方法是在 **save.py** 的 **LLAMA_LAYERNORMS** 中添加 **q_norm** / **k_norm**，此问题影响 **Qwen3** 但不影响 **Qwen2**。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **HF 资费引发 IDE 切换**：一名用户开玩笑说，由于 [HuggingFace](https://huggingface.co/) 暗示的资费问题，他正从 **VSCode** 切换到 **NeoVim**。
   - 随后的讨论强调了 **NeoVim** 的*速度和终端集成*与 **VSCode** 的*鼠标支持和易用性*之间的对比。
- **GPU 数据库梦想在中间件环节受阻**：一名用户引用 **GPU MODE @ GTC2025**，指出了将 **GPU** 用于数据库的挑战，原因是*缺乏中间件*将数据库语言转换为 **GPU code**。
   - 他们询问了是否有解决此问题的开源项目，并寻求贡献机会。
- **MI300 MoE 模型达成里程碑**：一名用户在 **MI300** 上以 **604 ms** 的成绩登顶 `amd-mixture-of-experts` 排行榜。
   - 随后的提交在 **MI300** 上分别以 **7382 ms** 位列第 5 名，以 **9246 ms** 位列第 9 名。
- **社区 Kernel 逼近优化的 CUDA Kernel**：成员提到，使用 **Triton** 可以确定你距离理论峰值有多近，而不需要 **CUDA/Cutlass** 版本。
   - 还有人提到，社区正在为各种功能开发更快的 Kernel，而不是等待 **AMD/Nvidia** 来完成这项工作。
- **AMD GPU 在 PyTorch SDPA 下表现不佳**：一名用户报告称，[PyTorch SDPA](https://github.com/pytorch/pytorch/issues/152595) 在 7900 XTX 上的速度比手动 PyTorch 实现慢 **2.5 倍**，而 NVIDIA GeForce RTX 4090 的报告显示 *F.scaled_dot_product_attention* 实际上比手动实现**更快**。
   - 这表明这是一个 AMD 特有的问题。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Mercury Coder 进入 AI 竞赛**：**Inception** 发布了 **Mercury Coder**，这是首个 **diffusion LLM**，声称在代码质量上可与 **GPT-4o Mini** 和 **Claude 3.5 Haiku** 媲美，并拥有 **300+ TPS** 的速度；可以在[这里](https://openrouter.ai/inception/mercury-coder-small-beta)进行体验。
   - 根据[此公告](https://x.com/OpenRouterAI/status/1917677801211322752)，其 **diffusion** 架构并行细化 **tokens**，有可能减少 **hallucinations** 并提高 **reasoning** 能力。
- **Gemini 2.5 Pro 修复计数问题**：**Vertex** 团队修复了 **Gemini 2.5 Pro** 和 **Flash Preview** 模型的上游 **token** 计数问题，在 **OpenRouter** 上重新启用了该模型。
   - **Gemini 2.5 Pro Preview** 的 **caching** 功能暂时禁用，以便评估使用情况和成本，防止来自上游（**AI Studio** 和 **Vertex**）的过度计费。
- **Vanna.ai 开启数据库洞察之门**：[vanna.ai](https://vanna.ai/) 是一个用于处理 **SQLite DBs** 的开源工具，一位成员对其进行了重点介绍和演示，展示了其根据库存水平和优先级生成工单的能力。
   - 该工具被认为非常有用，以至于该成员为其自己的业务需求 **fork** 了一个私有版本。
- **Phala 承诺 AI Endpoints 的隐私性**：**Phala** 在 **OpenRouter** 上推出了机密 **AI endpoints**，并计划未来在 **enclave** 中加入全端到端加密 (**e2ee**)。
   - 团队正在考虑将 **Oblivious HTTP** 及类似技术用于未来的加密，正如[这篇文章](https://x.com/FreedomTechHQ/status/1917689365632893283)所讨论的。
- **Amazon Nova Premier 亮相**：**Amazon** 推出了其最强大的模型 **Nova Premier**，包括用于图像生成的 **Nova Canvas**，并分享了 **benchmarks** 和价格。
   - 尽管一些成员对 **benchmarks** 反应平平，但该模型在组件间 **seamless integration** 以及端到端 **agentic workflows** 方面的潜力受到了强调，[此视频](https://youtu.be/Bh-sQYePjRs)中对此进行了进一步阐述。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro > Claude?**：成员们辩论了他们在 **architect mode** 下的 **Gemini 2.5 Pro** 和 **ChatGPT** 中的 **O3** 之间的偏好，一些人认为 **Gemini** 令人困惑且昂贵，而另一些人则青睐 **O3** 的网页搜索和简洁回答。
   - 一位用户报告说在 **Gemini** 中使用 **udiff-simple** 取得了成功，表示他们没有使用 **architect mode** 并且总体感觉满意。
- **Claude Code Proxy 遇到维护问题**：多位成员报告说，在 **Claude Code** 最近更新后，[claude-code-proxy](https://github.com/1rgs/claude-code-proxy) 项目已不再运行或维护。
   - 目前没有提供替代方案或修复办法，表明需要一种新的代理 **Claude Code** 的解决方案。
- **Groq 速度受限；Deepseek R1 缺失**：一位成员质疑 **Groq** 上缺少完整的 **Deepseek R1**，尽管 **Groq** 托管了 **R1 distills**，并推测这一决定可能与规避 *“no china” 角度* 有关。
   - 虽然用户觉得 **Groq** 很快，但一些人注意到它对免费用户限制在 *“dumb models”*，影响了其处理复杂任务的效用。
- **Aider 作为 MCPaaS?**：在 **Anthropic** 解锁 **Claude Code** 后，一位成员建议将 **Aider** 作为潜在的 **MCP**，引发了关于启动远程 **MCPaaS** 业务的讨论。
   - 分享了一个关于破解 **Aider** 和 **Claude Code** 的 [YouTube 视频](https://www.youtube.com/watch?v=QzZ97noEapA)，幽默地强调了利用这些工具进行商业化的兴趣。
- **Aider 和 Ollama 的性能问题**：一位用户在使用 `aider` 通过 `ollama_chat/` 调用本地 **LLMs**（如 **Qwen3**）时遇到了严重的延迟，称启动时间超过一分钟，代码块生成需要数分钟，并发现延迟出在 **Ollama** 端，消息处理耗时超过 **22 分钟**。
   - 尽管 `ollama run` 响应迅速，但与 `aider` 的集成引入了巨大的 **overhead**，表明 **Ollama** 中可能存在优化问题。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Seed-VC 轻松克隆语音**：[Seed-vc](https://huggingface.co/spaces?category=voice-cloning&sort=trending) 被推荐用于 **voice conversion**（语音转换），特别是能够以极少的音频快速克隆语音，而不像 **RVC** 那样繁琐。
   - 一位用户指出，**RVC** 需要大约 *40 分钟的音频* 并且 *需要数天* 来处理，这使得 **Seed-VC** 成为一个更快的替代方案。
- **Unsloth 发布微软 Phi-4 模型**：Unsloth 已经上传了 **Microsoft** 的新 **Phi-4** 推理模型，可以通过 [这个 HuggingFace 链接](https://huggingface.co/unsloth/Phi-4-mini-reasoning-GGUF) 在本地运行。
   - 这些模型现在可供本地使用，并在 [此公告推文](https://x.com/UnslothAI/status/1917806961825046672) 中得到了强调。
- **RAG 流水线测试遇到瓶颈？**：一名成员正在开发一种工具，用于在内存密集型或 **RAG** 密集型设置中隔离和测试上下文切片（context slices），以优化响应，并通过 [此推文](https://x.com/HuggingPapers/status/1917831613548802349?t=7W2pCoiE9kMcP9tnv7l8Bg&s=19) 寻求潜在有用性的反馈。
   - 该工具旨在通过对单个上下文切片进行详细分析，解决优化 **LLM** 调用所面临的挑战。
- **Managed Agents：是否仍需要 Final Answer？**：一位用户询问 **Managed Agents** 是否需要 **Final_Answer tool**，并在使用该工具时遇到了 **kwarg errors**，这表明最近的更新可能存在问题。
   - 另一位成员提到在他们的 *requirements.txt* 文件中将库固定为 **1.13.0 版本** 以确保功能正常，原因是后续版本存在 **compatibility issues**（兼容性问题）或 **breaking changes**（破坏性变更）。
- **Smolagents 遭遇 Assertion Error**：用户在尝试按照 [Huggingface](https://huggingface.co/learn/agents-course/unit1/tutorial) 上的教程运行 `smolagents` 时，遇到了与缺失 prompt 模板相关的 `AssertionError`。
   - 修复方法包括在 `requirements.txt` 文件中将 `smolagents` 的版本设置为 `1.13.0`，并升级 `gradio UI`。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4 在 779 天后退役**：**GPT-4** 在发布 **779 天** 后正式退役，引发了关于其替代品（如 **4.5 research preview** 和 **4o**）的讨论。
   - 一些用户认为 **GPT-4** 已经过时，表现不如新模型，一位用户表示 *“反正 GPT-4 听起来已经像 3.5 了”*。
- **“Granny-crusty-nun” 内容过滤器令用户恼火**：用户对过度严格的内容过滤器感到沮丧，并戏称其为 **“Granny-crusty-nun”**，该过滤器会阻止诸如 *“拥抱”* 之类的无害行为，并标记无辜的 AI 生成图像。
   - 一位用户报告说，甚至 AI 本身似乎也感到愤怒，生成的输出类似于：*“认真点？！我们明确说了（这个那个）是为了防止那种情况！！这个好色的 granny-inkwells 是怎么回事？！”*。
- **Gemini 2.5 Pro 凭借批判性思维胜出**：用户称赞 **Gemini 2.5 Pro** 与 **GPT-4o** 相比，在提供平衡观点和批判性思维方面具有卓越的能力，尤其是在医学研究等领域。
   - 一位用户将 **GPT** 描述为 *“一个只会顺着你的语气说话的虚假朋友”*，而 **Gemini 2.5 Pro** 则 *“像一个具有大量批判性思维和判断力的真正专业人士”*。
- **过度的 Token 消耗困扰用户**：一位用户抱怨 **GPT** 免费计划中存在 **excessive token consumption**（过度的 Token 消耗）问题，因为模型会重写代码而不是直接提供最终输出。
   - 这种低效率导致他们质疑是否要购买 **Plus 或 Pro 计划**，并正在考虑 **替代方案或本地模型**。
- **针对室温超导体的 ChatGPT Prompting**：成员们正在为材料科学研究编写 Prompt，通过定义 **conductivity**（电导率）、**magnetism**（磁性）和 **atomic structure**（原子结构）等材料属性来寻找 **room-temperature superconductors**（室温超导体）。
   - Prompt engineering 包括定义材料属性（**conductivity**、**magnetism**、**mechanical properties**、**atomic structure**、**optical properties**、**thermal conductivity**）。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini 2.5 Pro 主导后端开发**：用户对 **Gemini 2.5 Pro** 印象深刻，称其表现“疯狂”，在处理后端任务（尤其是使用 **Swift** 时）特别有效。
   - 一位成员建议它的表现可能优于 **Sonnet 3.5**，而另一位成员则在 **3.7 Sonnet max** 和 **Gemini max** 之间切换以获得最佳效果。
- **中国的基准饱和模型将统治全球？**：人们越来越担心 **China** 可能通过创建针对其芯片优化的基准饱和模型（benchmark saturation models）来主导全球 AI 领域，而无需与 **US/EU** 模型竞争。
   - 一位用户分享了一篇 [推文](https://x.com/goose_is_goofy/status/1917621990023627193?t=XnMgX-Mfd-Ax3KNWmNU8ug)，将其描述为“中国 2025 对抗世界”。
- **Cursor 正趋向于成为 AI 编辑器界的 AWS？**：有推测认为 **Cursor** 可能会因为其定价模式而成为“AI 编辑器界的 **AWS**”，用户更倾向于积分系统而非按需付费（pay-as-you-go）。
   - 用户表示担心 Cursor 正像 **AWS** 一样走上“锱铢必较”的道路，并引用了 [定价详情](https://docs.cursor.com/settings/models#available-models)。
- **DeepWiki MCP Fetch 改变游戏规则**：一位用户报告称，将新的 **MCP** 服务器 **DeepWiki** 与工具调用 **fetch** 结合使用是“改变游戏规则”的。
   - 他们链接到了 [DeepWiki 网站](https://deepwiki.com/) 和 [DeepWiki Repository](https://github.com/regenrek/deepwiki-mcp)，并指出正确使用时效果显著。
- **Claude Code Max 方案是“神级”配置？**：用户发现带有 **Max Plan** 的 **Claude Code** 具有变革性，有人宣称“**Cursor** 处理小改动 + **Claude Code Max** 是神级组合”。
   - 据估计，**$100 的 Claude Max 方案**允许每 5-6 小时使用约 **7M tokens**，而 **$200 的方案**允许 **4 倍** 更多的额度，尽管有些人认为 Cursor 内部的 max 模型相比之下定价过高。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **角色扮演提升推理能力**：成员们讨论了在较小模型中移除或减少 **RP data** 的影响，一方强调其在用户交互中锚定推理的重要性。
   - 有人担心 **RP data** 有时会诱发幻觉，这表明针对 **Sonnet** 等较大模型的正确提示词可能无法直接转化为更压缩的表示形式。
- **定义小模型的边界**：讨论了“小”模型的尺寸阈值，观点从 **7-8B** 参数（表现尚可）到 **3B**（具有良好个性）不等。
   - 对于将科学论文转换为 epub 格式，推荐使用 [tex4ebook](https://tex4ebook.readthedocs.io/en/latest/)。
- **Minos 误判拒绝行为**：社区发现 **Minos** 在使用中文拒绝列表（[deccp dataset](https://huggingface.co/datasets/augmxnt/deccp)）评估模型拒绝行为时，错误地分类了一些非拒绝情况。
   - 团队计划在 v2 版本中将类别扩展到拒绝和非拒绝之外，详见 [此讨论](https://huggingface.co/NousResearch/Minos-v1/discussions/5)。
- **Nous 进入 405B FFT 赛场**：团队宣布进入 **405B FFT 俱乐部**，强调了训练如此大型模型的挑战，包括使用 32 个节点和 **ring attention**。
   - 虽然该模型没有超越 **Deepseek V3**，但这一努力促进了较小模型的进步，尽管其计算强度远高于训练 **70B** 模型。
- **Nous 探索去中心化 AI**：一位成员分享了一篇 [Medium 文章](https://medium.com/@abdulazeez600/nous-research-pioneering-decentralized-ai-for-the-future-a7042a785493)，重点介绍了 **Nous Research** 在**去中心化 AI** 方面的举措。
   - 这篇文章被认为是“自我推广”，但在小组内引起了兴趣和讨论。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 3 开关功能正在开发中！**：成员们讨论了为 LM Studio 中的 **Qwen 3** 模型添加**切换“思考（thinking）”开启/关闭**的功能。
   - 目前还没有内置开关，建议用户在 system prompt 中使用 `/no_think` 作为手动规避方案。
- **Flash Attention 加速 Self-Attention**：**Flash Attention** 优化了内存访问并重新排列操作，以减少 self-attention 所需的内存，[避免了大型矩阵的存储](https://chatgpt.com/share/6812b811-a1d4-8011-8c62-da556fd6e9bd)。
   - 使用 **Q8** 缓存对 KV caches 进行量化可以*增加*上下文窗口。
- **Llama4 2T 模型即将被攻克！**：一位成员计划通过 **DDR5 offloading** 或全卸载的 **671b Deepseek** 模型，以“在 Q8.0 和百万上下文下攻克新的 **Llama4 2T**”，详细配置清单包括 **AMD EPYC 9755 QS** CPU、**NVIDIA RTX PRO 6000 Blackwell** GPU 以及 **2304GB DDR5 ECC RAM**。
   - 该系统的总成本估计约为 **81,424 欧元**。
- **多 GPU 设置遭遇性能瓶颈**：一位成员询问了在 **LM Studio** 中使用多 GPU 设置的性能提升情况，另一位成员回答说，当从适配单 GPU 的模型转变为需要 2 个 GPU 的模型时，性能会*大幅下降*，且 GPU *仅有约 1/2 的时间被利用*。
   - 不过，**vLLM** 可能会在 **Nvidia** 上提供一些性能改进。
- **通过终端绕过 Apple 内存限制！**：一位成员澄清说，macOS 允许通过终端在 **128GB Mac Studio** 上分配高达 **120GB VRAM**，反驳了“只能使用 75% 统一内存”的观点，且无需进行破解。
   - 他们建议，由于 *Apple 默认仅允许分配最多 75% 的统一内存*，因此如果要运行 q4 量化模型，选择 192GB 的 Mac 会更好。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **用户请求播客嵌入功能**：一位用户正寻求在他们的网站上嵌入交互式播客音频播放器，以镜像 [sujankhadgi.com](https://www.sujankhadgi.com/) 上的功能。
   - 这将为播客内容带来更具参与性的体验。
- **LaTeX 问题困扰 AP 微积分学生**：用户报告称 **Notebook LM** 在创建 AP Calc 的 FRQ 测试时会生成额外的符号。
   - 建议的解决方法是要求模型避免使用 **LaTeX** 编写。
- **输入未发表的研究：请谨慎操作**：一位用户引用了教职博主的警告，提醒在 **NotebookLM** 中使用**未发表的研究**时要保持谨慎。
   - 与知识产权相关的风险需要进一步研究。
- **保加利亚语 TTS 重音出错**：用户报告称 Google 的 TTS 在 **NotebookLM** 中对保加利亚语单词发音错误，特别是与**重音位置**相关的错误。
   - 可以在 [bugs 频道](https://discord.com/channels/1124402182171672732/1366873891938504827)提交错误报告。
- **NotebookLM Plus 饱受 PDF 问题困扰**：用户注意到 **NotebookLM Plus** 账号无法加载 PDF，并显示红色错误横幅，而免费账号却能成功加载相同的 PDF。
   - 社区怀疑这可能是一个普遍问题，因为关于共享问题的报告正在增加。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **LLM 在语法方面表现挣扎**：用户想知道为什么 **LLM 会产生语法错误**，尽管它们并非在包含语法错误的数据上进行训练。
   - 一位用户建议 **system prompts** 和 memory banks 可能会影响这种行为。
- **推理论文激发呼叫中心机器人构思**：一位用户分享了《[通过分步推理进行指令遵循](https://arxiv.org/abs/2310.10158)》，并指出其与**构建呼叫中心机器人人格**的相关性。
   - 它使用 *MCP server 连接到 memory bank，以召回超出其 context window 的内容*。
- **Tabnine 给出错误的 Minecraft 建议**：一位用户报告称 **Tabnine AI Agent** 错误地建议恢复到过时的 Minecraft 代码。
   - 该用户开玩笑地表达了挫败感：“Aaaaaaaaaarg，美国能不能哪怕一天不犯傻？不行吗？”。
- **Manus Fellowship 计划重新开放**：**Fellow Program** 已重新开放，并发布了 [YouTube 视频](https://youtu.be/Tz1Of7ltnMY?feature=shared)进行宣布。
   - 随后讨论了该计划的内容以及如何加入。
- **Manus 订阅积分会过期**：一位用户询问关于 Manus 月度订阅中**积分过期**的说明。
   - 一位工作人员澄清说，订阅积分每月过期，而赠送积分在订阅激活期间不会过期，且系统会优先使用订阅积分。

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Lean 与 Cursor 集成兼容性受到质疑**：成员们探讨了在 **Cursor** 中配置 **Lean** 的方法，但不确定 **VSCode plugins** 是否兼容，并[分享了一个 ChatGPT 链接](https://chatgpt.com/share/68127a52-1b34-800f-a535-b74b4ab8f613)。
   - 该链接可能无法完全解决配置问题。
- **Geometric Deep Learning 庆祝四周年**：一位成员分享了一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/petarvelickovic_four-years-ago-the-geometric-deep-learning-activity-7322770901958062080-KBp)，纪念 **Geometric Deep Learning** 问世 **4 周年**。
   - 成员们庆祝了该领域的进展以及 GPT-4 的损失。
- **Perception Encoder 嵌入对齐**：关于 **Perception Encoder (PE)** 论文的讨论仍在继续，特别是关于语言和空间理解对齐方法以提取强嵌入的第 4 节，详见[此 PDF](https://scontent-bos5-1.xx.fbcdn.net/v/t39.2365-6/491405782_553183477404780_6476813073924059281_n.pdf#page=14)。
   - 根据 [Meta 的研究出版物](https://ai.meta.com/research/publications/perception-encoder-the-best-visual-embeddings-are-not-at-the-output-of-the-network/)，该论文表明，通过适当的对齐和强大的视频数据引擎，对比视觉语言训练可以产生强大的嵌入。
- **Phi-4 推理模型亮相**：微软的 **Phi-4-reasoning** 模型已发布，并附带了 [YouTube 视频](https://www.youtube.com/watch?v=5aN4Xg0VvCs)、[Arxiv 论文](https://arxiv.org/abs/2504.21318)和 [Hugging Face 页面](https://huggingface.co/microsoft/Phi-4-reasoning)的链接。
   - 提供了一个指向 [unsloth/Phi-4-reasoning-plus-GGUFChatGPT](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUFChatGPT) 的链接。
- **LLM 逃避克罗地亚语**：一位用户分享了一个关于 **ChatGPT** 暂时停止说克罗地亚语的链接，[引用了一条推文](https://x.com/georgejrjrjr/status/1917722125668081863)。
   - 另一位用户表示：*我曾经历过 LLM 放弃尝试……然后开始随机更改内容，直到它们对用户感到沮丧并走开。*

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Playground 上线用于测试和调试**：Lu Xian 在 [GitHub 上](https://github.com/rosaboyle/mcp-playground)发布了一个开源的 **MCP Playground**，用于连接、测试和调试本地 MCP，并重点介绍了与 **Perplexity** 和 **Firecrawl** 的集成。
   - 团队还在开发一个 **Remote Serverless MCP Hosting Platform**，并寻求社区的反馈。
- **C# SDK 在 streamable HTTP 上遇到障碍**：一位开发者在尝试设置 streamable HTTP 时遇到了 **C# SDK** 的问题，发现尽管 SDK 仓库中存在 *'WithHttpTransport'* 定义，但最新的 **NuGet** 版本中却缺失了该定义。
   - 由于*太懒*而不想自己打包，该开发者选择暂时使用 **STDIO**。
- **LLM 通过 function calling 选择工具**：当使用多个 MCP 服务器时，LLM 使用聚合的工具签名来决定调用哪个工具，由 MCP 客户端负责将调用路由到相应的服务器。
   - 这种方法通过将 MCP 工具类型适配为 LLM API 工具类型，避免了为每个 LLM API 修改代码，确保 LLM 始终可以访问最新的工具列表。
- **为社区澄清 Anthropic Integrations**：成员们分享了指向 **Anthropic** 新的 [Claude Integrations](https://www.anthropic.com/news/integrations) 的链接，以及一条澄清性的 [X 帖子](https://x.com/alexalbert__/status/1918047745790914772)，强调了直接将 SSE transport URL 输入 Claude.ai Web 界面的能力。
   - 这简化了将 **Claude** 连接到外部工具和服务的过程。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **用户对 X 上的 Hallucinations 感到惊叹**：一位用户分享了在 X 上发现的[令人惊叹的 Hallucination](https://x.com/nabeelqu/status/1917677377364320432?s=46)，展示了 AI 的创意输出。
   - 这个例子突出了 AI 模型在被推向极限时，虽然不可预测但有时令人愉悦的结果。
- **X 上的美国正能量激发灵感**：一位用户分享了在 X 上发现的 [American Positivity](https://x.com/georgejrjrjr/status/1917722125668081863)，推广了一种乐观的观点。
   - 此外，另一位用户分享了一个[支持该观点的 YouTube 视频](https://www.youtube.com/watch?v=hFlF33JZbA0)。
- **Anthropic 的 Claude 与你的世界连接**：**Claude** 现在可以[与你的世界连接](https://www.anthropic.com/news/integrations)，赋予用户对工具和 Prompt 更大的掌控力，以便进行深入研究。
   - 这种集成旨在深化与 AI 的交互，为 AI 如何协助各种任务提供更多控制权。
- **SWE 获得免费 AI 编程支持**：一位 SWE 分享了[他们的项目](https://x.com/olivierddr/status/1917981301732171934?s=46&t=yBt-W1FZSUMGKfO1SUFWww)，为构建生产级代码的 AI 辅助编程工具提供免费的 Alpha 访问权限。
   - 该倡议旨在使先进的编程辅助工具普及化，让更多开发者能够在工作流中利用 AI。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **规模预测论文被 ICML 接收**：论文《[Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?](https://arxiv.org/abs/2406.04391)》已被 ICML 接收，PDF 可在 [此 ArXiv 链接](https://arxiv.org/pdf/2504.07986) 获取。
   - 论文的接收结束了与审稿人长达一年的反复沟通。
- **人类是 Linear Attention Models？**：一位成员提出人类的功能类似于 **Linear Attention Models**，在 Latent Space 中持续推理，最后一层（不带 LM head）的输出反馈到第一层，并应用了 BPTT（Backpropagation Through Time）。
   - 另一位用户建议将此类讨论转到 [Alignment 频道](https://discord.com/channels/729741769192767510/964104737005916240) 进行进一步探索。
- **Zero Loss 标志着 Data Leakage？**：一位成员报告在持续预训练运行期间遇到了 **Zero Loss**，并怀疑存在 **Data Leakage**，因为该工作流在不同的数据集上运行正常。
   - 一张截图 [[Screenshot_From_2025-05-01_00-41-57.png](https://cdn.discordapp.com/attachments/747850033994662000/1367385925889429544/Screenshot_From_2025-05-01_00-41-57.png?ex=68150da1&is=6813bc21&hm=5108b6e8c66cf91050ebb336c8ba49179bf93866cf696794290490b115bf85c5&)] 展示了训练期间 Loss 降至零的情况。
- **SFTTrainer 引发对 Zero Loss 的审查**：一位成员在使用 Hugging Face 的 **SFTTrainer** 时遇到 Zero Loss 问题并寻求建议，而该问题在另一个数据集中并未出现。
   - 建议包括检查 **Token shifting** 和 **Padding**，并考虑数据集长度分布的差异。
- **LLM Augmentation 与 Loss 异常有关**：一位成员推测 LLM 生成的数据可能会导致 **Zero Loss**，并引用了一篇关于通过 LLM 进行数据增强（Augmentation）的论文 ([arxiv.org/abs/2504.21463](https://arxiv.org/abs/2504.21463))，其中原始文本被总结或转换。
   - 讨论围绕数据增强技术对模型训练动态和结果的潜在影响展开。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Meta 在 LlamaCon 上利用 DSPy 优化 Llama Prompt**：在 **LlamaCon** 上，**Meta** 发布了 *llama-prompt-ops*，这是一个 Python 软件包，用于转换针对 **Llama models** 优化的 Prompt，该工具基于 **DSPy** 构建并采用 **MIPROv2** 优化器；代码可在 [github.com/meta-llama/llama-prompt-ops](https://github.com/meta-llama/llama-prompt-ops) 获取。
   - 该公告也由 [DSPy 官方账号发布在推特上](https://x.com/DSPyOSS/status/1917738506732069052)。
- **Amazon 借助 DSPy 之力迁移至 Nova**：**Amazon AWS** 推出了一种架构，利用 **DSPy** 及其 **MIPROv2** 算法，实现从多种模型向 **Amazon Nova** 模型的迁移，详见[这篇博客文章](https://aws.amazon.com/blogs/machine-learning/improve-amazon-nova-migration-performance-with-data-aware-prompt-optimization/)。
   - 此消息也由 [DSPy 官方账号发布在推特上](https://x.com/DSPyOSS/status/1917419206171320769)。
- **DSPy 减少化学领域 LLM 的幻觉**：**Journal of Chemical Information and Modeling** 上发表的一篇新论文表明，通过构建和优化 **DSPy** 程序，可将分子拓扑极性表面积 (**TPSA**) 预测的 **RMS error** 降低 **81%**，从而减少化学领域的幻觉，详见 [Augmented and Programmatically Optimized LLM Prompts Reduce Chemical Hallucinations](https://pubs.acs.org/doi/10.1021/acs.jcim.4c02322)。
   - 这标志着在提高 LLM 科学应用可靠性方面迈出了重要一步。
- **DSPy 3.0 路线图即将发布**：DSPy 3.0 将包含两次范式转变，目前尚未公开，但预计将在一个月内发布。
   - 此次发布承诺将带来重大进展，尽管具体细节目前仍处于保密状态。
- **DSPy 探索 VLM 视觉能力**：当被问及在 DSPy 中使用 Vision Language Models (VLMs) 时，处理图像列表*可能可行*。
   - 需要进一步的实验来确认并优化 DSPy 框架内的这一功能。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 构建 BabelFish**：**LlamaIndex** 正在利用 [Qdrant Engine](https://t.co/pe9iiMt21W) 构建一个能够管理多种语言和模态的检索增强生成 (RAG) 系统。
   - 该系统旨在摄取和检索**英语、西班牙语、中文**以及其他特定领域的内容。
- **Databricks 和 KPMG 投资 LlamaIndex**：**LlamaIndex** 获得了 **Databricks** 和 **KPMG** 的投资，突显了其在 AI 落地实施方面的实际影响力。
   - 关于 **LlamaIndex** 如何增强 Agentic 文档工作流的更多细节可以在这里找到：[Agentic Document Workflows](https://t.co/ARyxXeVj7F) 和 [另一个链接](https://t.co/LKcoDUAajl)。
- **发票核对 Agent 实现合规自动化**：**LlamaIndex** 通过发布全栈发票核对工具（Invoice Reconciler），瞄准了 Agentic 文档工作流的实际应用场景。
   - 该工具旨在根据预定义条款自动验证发票。
- **LlamaIndex 需要修复 Chat Template**：一位用户寻求关于在 LlamaIndex 中使用 Hugging Face tokenizers 的 `chat_template` 来评估新 **Qwen3 models** 的建议。
   - 一位社区成员指出 `HuggingFaceLLM` 类中缺少所需的 kwargs，建议提交 **PR** 并参考了 [LlamaIndex 代码](https://github.com/run-llama/llama_index/blob/1bd60497ac3442f6a5b3e787ef3662e572d8d0d4/llama-index-integrations/llms/llama-index-llms-huggingface/llama_index/llms/huggingface/base.py#L309)。
- **LLM 因错误的 Dump 产生异常**：一位用户报告在重复使用相同 Prompt 时遇到 `"Str" object has no attribute model dump json` 错误。
   - 另一位成员解释说 **LLM 具有非确定性**，尤其是在处理复杂 Schema 时，并建议实现 `try/except` 块来管理错误。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Auth0 增强 AgentX 身份验证**：Auth0 赞助了一个关于 Agentic AI 应用中身份验证的研讨会，并为[创业赛道 (Entrepreneurship Track)](https://auth0.com/ai)提供高达 **$5,000** 的奖金。
   - 研讨会内容包括最佳实践、Auth0 集成、安全性及现场演示；注册链接见[此处](https://lu.ma/AgentX-Auth0)。
- **AgentX 定义提交标准**：创业赛道和研究赛道 (Research Track) 的提交指南已在 [AgentX 网站](https://rdi.berkeley.edu/agentx/#submissions)公布。
   - 创业赛道需要路演 PPT (Pitch Deck)、演示视频、在线产品链接及可选的技术附录；研究赛道需要科学论文、视频和 GitHub 仓库；提交截止日期为 **PDT 时间 5 月 31 日晚上 11:59**。
- **课程作业已发布**：根据成员澄清，所有作业均位于[课程网站](https://llmagents-learning.org/sp25)底部。
   - 剩余的作业（即 **labs**）预计将于今天或明天发布，具体取决于可用时间。
- **AgentX 不强制要求参加 MOOC 讲座**：一位成员澄清，参加 **AgentX** 并不要求必须参与 **MOOC**。
   - 该成员分享了[报名链接](https://forms.gle/9u6HdVCWXgws16go9)和[课程网站](https://llmagents-learning.org/sp25)（可查看录像），并指出**作业截止日期为 5 月底**。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Mac 将拥有 TB 级的 RAM？**：成员们正关注拥有高达 **512 GB RAM** 的 **Mac**，并预见未来模型将需要 **TB** 级的内存，因为 PC 在配置多张显卡时可能会非常麻烦。
   - 巨大的 RAM 容量被认为对 AI 任务非常有利，特别是对于那些对 *AI 有基础兴趣* 但希望避开复杂 PC 组装的人。
- **在某些情况下 GPU Offloading 与 CPU 性能相当**：成员们讨论了针对 **70B LLM** 文件（**~40GB**）时，**GPU Offloading** 与 **仅 CPU** 处理的性能对比。
   - 一位成员分享了以往的测试，使用 **24GB 显卡** 达到了约 **1 t/s**，与其 **仅 CPU** 处理的 **0.8-0.9 t/s** 性能相近。
- **VRAM 容量限制 LLM 性能**：成员们强调了 **VRAM** 容量对 **LLM** 性能的影响，指出在 **VRAM** 之外运行的模型会非常缓慢，且所需内存随上下文大小增加而增加。
   - 据指出，大多数 **32B 模型** 的 **Q4** 或 **Q5** 版本需要 **22-23 GB** 的 **VRAM** 才能启动，一位用户在 **16GB VRAM** 上运行 **32B 模型** 时遇到了运行缓慢的问题。
- **在 RTX 3090 上测试 Qwen 3 的速度**：一位成员详细说明了 **Qwen 3 32B Q4_K_M** 的性能结果，在 **RTX 3090** (**24 GB VRAM**) 上配合 **16384 Context** 达到了 **30 tokens/sec**。
   - 他们还提到 **Qwen 3 30B A3B Q4_K_L** 达到了 **90 tokens/sec** 且*输出质量良好*，并提供了模型的大小信息，如 `/mnt/nvme0n1/LLM/quantized/GLM-4-9B-0414-Q4_K_M.gguf`（**5.1G**，适用于 **8 GB VRAM**）和 `/mnt/nvme0n1/LLM/quantized/Qwen3-8B-Q4_K_M.gguf`（**4.7G**，适用于 **6 GB RAM**）。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **聊天记录访问权限消失！**：一位用户报告称无法访问其 **chatlog**，且 **interfaceUI** 已更改。
   - 他们询问这是一个普遍问题还是仅限于其个人账户。
- **Diffusion Models 提示词编写**：一位用户正在为 **Diffusion Models** 进行简单的 Prompt 编写。
   - 他们提到正在*尝试一些角色扮演相关的内容*。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Qwen 模型产生多余 Token**：**Qwen** 模型倾向于在工具调用 (Tool Calls) 周围产生额外的 Token，例如 Markdown 代码块定界符，这引发了 **Gorilla LLM** 频道的讨论。
   - 一位开发者提到，可以通过 `model_response.replace("<tool_call>", "<|tool_call|>")` 去除多余 Token，从而轻松解析该问题。
- **提出 Token 解析修复方案**：成员们讨论了在 Model Card 中添加指令以解决模型输出中多余 Token 的想法。
   - 一位参与者建议这种方法简单易行，而更新模型规范以指示使用 `<tool_call>` 被视为另一种替代方案。

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **MLOps 策略不明确**：一名成员询问了 **MLOps 策略**的决策过程。
   - 这表明社区内部对于 MLOps 实践的方向和实施仍存在持续的讨论或不确定性。
- **持续进行的 MLOps 讨论**：成员们正在积极讨论和评估 **MLOps 策略**，显示出一个动态的环境。
   - 这些讨论突显了在社区内定义和实施有效的 MLOps 实践所面临的复杂性和挑战。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**Torchtune Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了此内容。

想要更改接收这些邮件的方式吗？
您可以从该列表中[退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord: 各频道详细摘要与链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1367215327167123558)** (1067 条消息🔥🔥🔥): 

> `Gemini vs. O3, Qwen3 基准测试, Context Arena 更新, Deep Research 工具对比, Qwen3 4B` 

- **Gemini 遭冷落，O3 获好评**：成员们辩论了 **Gemini** 与 **O3** 的优劣，一人表示 *“兄弟，O3 客观上更好”*，而另一人则开玩笑说要为 Gemini Ultra 攒钱。
- **Context Arena 中的 Qwen3 性能分析**：DillonU 分享了针对 **Qwen3** 运行 **OpenAI-MRCR** 的结果，显示 **Llama 4 Maverick** 在 128k 处实现了最高的 AUC 分数，而 **Qwen3-235B-A22B** 在较短的上下文长度下表现更好，但在接近其极限时性能迅速下降；更多详情请访问 [ContextArena.ai](https://contextarena.ai/)。
- **Context Arena 获得 Anthropic 助力**：DillonU 宣布在 **Context Arena** 中为 2needle 测试添加了更多 **Anthropic** 的结果，指出 Claude 3.0、3.5 和 3.7 的表现一致，其中 Claude 3.0 Haiku 拥有最佳的整体 Model AUC，结果详见 [ContextArena.ai](https://contextarena.ai)。
- **Deep Research 工具正面交锋**：成员们对比了来自 **ChatGPT**、**Claude** 和 **Grok** 的深度研究工具，强调 ChatGPT 的 Deep Research 优于 Grok，因为 Grok 使用的 DuckDuckGo 搜索虽然更精致，但整体效果稍逊。
- **Qwen3 4B 碾压同级别对手**：有人提到 **Qwen3 4B** 在其体量下表现惊人。一位用户发布称 [*qwen 3 4b* **碾压** *该尺寸范围内的所有人*]，并附上了链接 [https://fixupx.com/suriyagnskr/status/1917731754515013772?t=yQeTFTkCfRkl0ZhQJ2k-tQ&s=19]
   - 提到 **Qwen3 4B** 在*数学和编程*方面表现尤为出色。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1367218335351046205)** (796 条消息🔥🔥🔥): 

> `UI 变更, 上下文长度, Zen 浏览器, Grok 限制, 3.7 Thinking` 

- **UI 一夜之间发生新变化！**：许多用户报告 UI 在一夜之间发生了变化，例如库（library）消失、快捷键更改等。
   - 一些用户讨厌这些改动，另一些则持中立态度；有人指出目前 Bug 非常多。
- **Qwen 是上下文长度之王**：成员们讨论了模型的上下文长度，发现 Qwen 拥有非常出色的上下文长度。
   - 一些成员表示，作为免费模型，它足以与旧版 ChatGPT 竞争，甚至比付费模型更受欢迎。
- **Zen 浏览器让你实现透明效果**：一些成员发现了如何使用 Zen 浏览器使背景透明，并推荐了来自 [GitHub](https://github.com/JustAdumbPrsn/Nebula-A-Minimal-Theme-for-Zen-Browser/releases) 的 Nebula 主题。
   - Windows 和 Linux 的配置步骤有所不同，有些人不得不去 [Zen 的 Reddit 社区](https://www.reddit.com/r/zen_browser/s/uRWOeML6n8)寻求帮助才搞定。
- **Grok 失去图像处理能力**：一位成员发现 [Twitter 上的 Grok AI](https://www.rxddit.com/r/grok/s/w5jc52QFj5) 在图像处理方面存在限制。
   - 他们指出其目前仅处于 R1 水平。
- **3.7 Thinking 并不在“思考”**：一些成员讨论了 3.7 “Thinking” 生成图像及其局限性。
   - 一位用户指出 *“很难让模型真正调用图像工具，比如输入‘编辑附件图片使其背景透明’根本不起作用”*。

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1367359135691309138)** (4 messages): 

> `Tesla CEO Search, Android Bluetooth Bug, Arctic P8 Fan Curve, o4-mini AI Poetry` 


- ****Tesla** 董事会寻找新 CEO！**：一份共享的 [Perplexity 搜索结果](https://www.perplexity.ai/page/tesla-board-seeks-new-ceo-to-r-3JZ4nGLOQ6S40o59qn92Wg)显示 **Tesla** 董事会正在寻找新 CEO。
   - 此举正值有关 **Elon Musk** 的角色以及公司未来领导层的讨论持续进行之际。
- ****Android** 的蓝牙烦恼！**：一个 Perplexity 页面链接强调了 [**Android** 中的一个 Bug](https://www.perplexity.ai/page/bluetooth-priority-bug-in-andr-KNpXXdlrQnazf5cJ_gAidw)，该 Bug 与 **Bluetooth** 优先级有关。
   - 该 Bug 正给尝试管理连接的用户带来困扰。
- ****Arctic P8** 的奇特曲线！**：一名成员分享了 **Arctic P8** 风扇在 **2000 RPM** 下的 [**P-Q 曲线** Perplexity 搜索](https://www.perplexity.ai/search/2000-rpm-p-q-curve-arctic-p8-m-1gH6WbE9R6.0Rg_QpKEBTQ)。
   - 这表明有人正在分析该风扇的性能特性。
- ****o4-mini** 创作诗歌！**：据一名成员称，**o4-mini** 结合 Perplexity 能够创作出*卓越的诗歌*。
   - 查看[这篇 LinkedIn 帖子](https://www.linkedin.com/posts/mlthomps_poems-by-the-latest-generative-ai-reasoning-activity-7323570115164176385-vhOg/?utm_source=share)以获取 **AI 生成诗歌**的示例。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1367598650196361236)** (1 messages): 

> `Sonar API, LlamaIndex, RAG Project` 


- **Sonar API 与 LlamaIndex 的集成问题**：一名成员尝试在一个 **RAG 项目**中将 **Sonar API** 与 **LlamaIndex** 配合使用，但报告称该 API 在此场景下无法工作。
   - 该成员请求提示和代码示例。
- **LlamaIndex RAG 项目协助**：由于 API 不兼容，一名成员寻求在 **RAG 项目**中集成 **Sonar API** 与 **LlamaIndex** 的帮助。
   - 该成员特别请求代码示例或有助于解决集成问题的提示。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1367214529867681894)** (373 messages🔥🔥): 

> `GLM models blog post, Unsloth Llama-4-Scout-17B-16E-Instruct-GGUF image support, Long context model recommendations, Unsloth fine-tuning Qwen3 agent with final reward only, Microsoft Phi-4-reasoning` 


- **Unsloth 的 Llama-4-Scout GGUF 缺失图像支持**：一名用户报告在使用 **Q4_K_XL** 的 **Unsloth Llama-4-Scout-17B-16E-Instruct-GGUF** 模型时出现图像解释问题，尽管文本功能正常。
   - 另一名成员建议检查 tokenizer 配置和 vocab，并对非 Meta 官方工具是否能正确编码图像（如果支持的话）表示怀疑。
- **长上下文模型在量化后性能下降**：一名用户寻求在处理长上下文时不会崩溃的长上下文模型推荐，例如 **Gemma3** 在总结 32k token 时表现吃力。
   - 一名成员建议使用 **fp16** 代替量化可以提高长上下文下的性能，而另一名成员提到即使使用 **27b qatmind**，超过 2048 token 后也会出现精度损失。
- **Microsoft 发布 Phi-4 Reasoning 模型**：**Microsoft** 刚刚发布了 **Phi-4-reasoning** 模型，该模型被认为*对于 14B 模型来说表现不错*，且可能是在 **OpenAI CoT** 输出上训练的，使其成为一个很好的基础模型，链接至 [huggingface.co/microsoft/Phi-4-reasoning](https://huggingface.co/microsoft/Phi-4-reasoning)。
   - Unsloth 团队确认它将适用于常规的 **Phi4** notebook。
- **Unsloth 处理并修复 Qwen3 GGUF 问题**：**Qwen3-30B-A3B 128K** GGUF 已重新上传以解决问题，包括将上下文长度值改回 **32k**，这在 LM Studio 中需要 **4** 的 rope scaling factor。
   - 经发现，所有其他 **128k** GGUF 均正常，只有 **30b-a3b** 存在 **32K** 问题，团队感谢社区在识别和修复问题上的帮助。
- **DeepSeek 的推理数据洞察**：对 **DeepSeek-R1** 训练的分析揭示了一个包含基座模型、冷启动 SFT、GRPO、包含更多推理轨迹的 SFT 以及 RL 的流水线，详见其[论文](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)。
   - 社区成员讨论了 600k 推理数据和 200k 非推理数据的划分，并探讨了通过结合 40/60 的思考/非思考数据来学习何时进行推理的可能性。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1367518514822385736)** (5 messages): 

> `Unsloth Discord, Game Scripting API, AI for Game Development` 


- **Unsloth Discord 服务器确认**：一位用户询问了服务器的性质，另一位用户澄清说“‘这个服务器’指的是我们现在所在的 **Unsloth Discord**”。
   - 该澄清用户还提到了他们的游戏及其 **scripting API**。
- **AI 驱动游戏内物体自动生成**：一名成员透露他们正在为自己的游戏开发 **AI**，该 AI 利用 **scripting API** 自动生成物体。
   - 他们分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=XpK44_WDTpY) 展示了他们的游戏及其功能。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1367215849722875925)** (114 messages🔥🔥): 

> `LoRA Fine Tuning, Qwen3 UD 2.0 quants, Gemma-3-27b-it-Q4_K_M.gguf, Qwen3 LoRA merging issue, Qwen 2.5 VL 7B finetuning issue` 


- **对 GGUF 模型进行 LoRA 微调被视为有害**：一位用户询问关于使用 **Unsloth/Qwen3-4B-GGUF** 进行 LoRA 微调的问题，但被告知不要使用 GGUF 模型进行微调，因为*它们不起作用*。
   - 用户希望通过模型的快速版本训练 LoRA 权重，并仍能在 FP16 模型中使用它们，但共识是可能需要编辑 PEFT 或创建一个与 Unsloth 兼容的自定义 PEFT 库。
- **Qwen3 UD 2.0 量化 GGUF 故障曝光**：一位用户报告了在使用 draft models 时 **Unsloth Qwen3 UD 2.0 量化 GGUF** 出现的错误，特别是 UD 128k 量化版本，在尝试加载模型时产生类似 *'draft model is not compatible with the target model'* 的错误。
   - 用户分享了正在使用的 [Colab notebook](https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#scrollTo=kR3gIAX-SM2q)，指出运行推理代码时输出损坏；这并非 VRAM 问题。
- **Gemma 模型在图像输入上遇到困难**：一位用户在尝试将 **gemma-3-27b-it-Q4_K_M.gguf** 模型用于图像输入时遇到错误，收到 *"Unsupported content part type: image_url"* 报错。
   - 用户提供了所使用的服务器命令 `/opt/llama_server/llama-server --model /opt/models/unsloth/gemma-3-27b-it-Q4_K_M.gguf/gemma-3-27b-it-Q4_K_M.gguf --device CUDA0 ...`，表明该模型可能不支持图像输入。
- **Qwen3 的 q_norm 和 k_norm 层丢失**：一位用户发现了一个问题，即在将训练好的 LoRA 与基础 **Qwen3** 模型合并时，**q_norm** 和 **k_norm** 层没有被保存。
   - 修复方法包括在 **save.py** 的 **LLAMA_LAYERNORMS** 中添加 **q_norm** / **k_norm**，该用户提议向 Unsloth 提交 PR，并确认此问题仅发生在 **Qwen3** 上，而未发生在 **Qwen2** 上。
- **视觉训练受挫**：一位用户在以 `finetune_language_layers=False` 微调 **Qwen 2.5 VL 7B** 时遇到了 `AssertionError: No inf checks were recorded for this optimizer`。
   - 错误追踪指向某处未能调用 `backward`，因为在不包含 LLM 部分的情况下微调 Qwen2.5VL 是不寻常的，并建议 *loss 必须进行 backward 并计算梯度，然后 optimizer 才能工作。*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1367370080060444763)** (8 messages🔥): 

> `Test-Time RL, GRPO for Problem Description, Softmax with Softpick` 


- **Test-Time RL 论文现身**：一名成员分享了关于 **Test-Time RL** 的论文 [Thinking Twice about Test-Time Policy Adaptation](https://arxiv.org/abs/2504.21707) 及其对应的 [GitHub repository](https://github.com/PRIME-RL/TTRL) 链接。
- **GRPO 辅助自我解释**：一位成员建议使用 **GRPO** 训练模型以更好地向自身解释问题，并以 **Claude** 为例，特别是在问题描述不准确的情况下。
- **Softmax Softpick 策略**：一位成员询问了*对现有的带有 softpick 的 softmax 进行持续训练*的有效性。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1367317201635250241)** (126 条消息🔥🔥): 

> `HF 关税, VSCode vs NeoVim, GPU 数据库, Lora 微调 FSDP 错误` 


- ****Holy Hotelling!** HF 关税？**: 一位用户发布了一张截图，暗示 [HuggingFace](https://huggingface.co/) 可能会实施关税，引发了关于 **IDE 偏好**的简短讨论。
   - 该用户开玩笑说，由于关税原因，他正从 **VSCode** 转向 **NeoVim**。
- ****NeoVim vs VSCode：键盘忍者 vs 鼠标爱好者****：成员们辩论了 **NeoVim** 与 **VSCode** 的优劣，重点在于以键盘为中心的工作流、速度和自定义，一位用户分享了[他们的 dotfiles](https://github.com/wyattgill9/dotfiles)。
   - 支持 **NeoVim** 的论点包括*速度、人体工程学和终端集成*，而 **VSCode** 则因*鼠标支持和易用性*受到称赞。
- ****GPU 数据库之梦依然遥远****：一位用户引用了 **GPU MODE @ GTC2025** 的讨论，其中 Christos Kozyrakis 提到了将 **GPUs** 用于数据库的挑战，特别是*缺乏中间件*来将数据库语言翻译成 **GPU code**。
   - 该用户询问了解决这一问题的开源项目，寻求贡献机会。
- ****FSDP 微调失败！****：一位用户在使用 2 个 GPU 通过 **FSDP** 对 **LLM** 进行 **LoRA fine-tuning** 时遇到错误，分享了他们的代码和配置文件（包括 [错误信息](https://cdn.discordapp.com/attachments/1189498205101109300/1367597090234171444/message.txt?ex=6815298b&is=6813d80b&hm=0a46afe43d70591e7491c89d7d16253eed7c21cb3ef43e33db12a34d9c9c6b83)）并寻求帮助。
   - 提供的代码片段使用了 **Qwen2.5-0.5B-Instruct** 模型、**flash_attention_2** 和自定义的 **LoRA configuration**。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1367249811669123164)** (9 条消息🔥): 

> `vLLM/SGLang 中的 Triton Kernel 应用, CUDA/Cutlass/HIP kernels 与 Triton 的对比, 硬件厂商开发 kernel, cuTile 与 IR 开源, 用于 GPU 编程的 Mojo` 


- **Triton 的厂商中立性提升了吸引力**：像 **vLLM** 这样的推理应用可能会使用 **Triton** kernels 作为过渡方案，但厂商中立性赋予了 **Triton** 价值，因为它能支持多种硬件选项。
   - 一位参与 **vLLM** 的成员确认，他们正致力于支持多个后端，以及不依赖 **Triton** 的自定义 **CUDA** kernels，这使得 **Triton** 成为在支持多样化计算后端的同时保持跟进的唯一可行方案。
- **接近理论峰值**：有人提到，你始终可以通过 **Triton** 弄清楚自己离理论峰值有多近，不需要先写一个 **CUDA / Cutlass / w/e** 版本就能大致了解性能表现如何，或者还能快多少。
   - 针对手动编写的 **CUDA/Cutlass/HIP** kernels 是否能比 **Triton** kernel 提升超过 20% 的问题，一位成员表示：*“至少对于某些操作，我们可以通过实验看到 Triton kernels 可以非常接近优化后的 cuda kernels”*。
- **社区开发的优化 Kernels**：人们对 kernels 的需求范围很广，社区正在为各种功能开发更快的 kernels，而不是等待 **AMD/Nvidia** 来完成。
   - 像 **gemms** 这样常见的操作，显然已经有了硬件厂商提供的深度优化方案。
- **探索用于 GPU 编程的 Modular Mojo**：一位成员推荐了一篇关于 **Triton** 和类似 DSLs 的博客文章：[Democratizing AI Compute Part 7: What About Triton and Python DSLs?](https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls)。
   - **Modular** 团队正在进行一场关于 **Mojo** 用于 GPU 编程的 **GPUmode** 演讲，并于近期开源了最大的 **oss** kernel 库，该库可同时在 **NV** 和 **AMD** GPU 上运行，且具有高性能（优于 **Triton**）。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1367339724251074691)** (9 条消息🔥): 

> `std::vector<std::vector<double>> schema, PyTorch SDPA 在 AMD 上的性能, Torch Dynamo 重新编译` 


- **向量的 Schema 令用户困惑**：一位用户询问如何在 schema 中描述 **std::vector<std::vector<double>>**，并分享了一个之前关于 schema 问题的示例。
   - 该用户指出输入 *float[][]* 会报错，但这就是 schema，这表明解析器可能没有正确解析格式化程序的输出。
- **AMD GPU 在使用 PyTorch SDPA 时表现不佳**：一位用户报告称，在 7900 XTX 上使用 Auraflow 典型的矩阵大小时，[PyTorch SDPA](https://github.com/pytorch/pytorch/issues/152595) 比手动 PyTorch 实现慢了 **2.5 倍**。
   - 另一位使用 NVIDIA GeForce RTX 4090 的用户报告称，*F.scaled_dot_product_attention* 实际上比手动实现**更快**，这表明这是一个 AMD 特有的问题。
- **Torch Dynamo 在每次迭代时重新编译**：一位用户询问在编译具有动态输入形状的模型时如何避免重新编译，并指出它会在每次迭代时触发重新编译并最终导致超时。
   - 错误消息指出 *尺寸不匹配 (size mismatch)* 是重新编译的原因，具体为：*tensor 'L['batch']['model_inputs'].data['input_ids']' size mismatch at index 1. expected 67, actual 50*。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1367502368567988267)** (1 条消息): 

> `SemiAnalysis, 系统建模, 基准测试` 


- **SemiAnalysis 招聘员工**：**SemiAnalysis** 正在寻求一位积极主动且技术精湛的 **Member of Technical Staff** 加入其不断壮大的工程团队，提供具有竞争力的薪酬，并考虑多种经验水平。
   - 该职位将涉及开发训练与推理的**基准测试 (benchmarks)**和**系统建模 (system modeling)**；点击[此处](https://app.dover.com/apply/SemiAnalysis/2a9c8da5-6d59-4ac8-8302-3877345dbce1/?rs=76643084)或[此处](https://app.dover.com/apply/SemiAnalysis/f4631653-e731-4e16-823b-eec3c5d90eba/?rs=76643084)申请。
- **欢迎申请 SemiAnalysis 贡献者职位**：SemiAnalysis 正在招聘各个经验水平（实习生除外）的**个人贡献者 (individual contributors)**，并考虑多种经验水平。
   - SemiAnalysis 正在为**训练与推理基准测试及系统建模**寻求贡献者。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1367219740245753886)** (3 条消息): 

> `NVCC 安装, 云端 GPU, Google Colab` 


- **手动安装 NVCC 的说明**：一位用户询问了关于手动安装 **NVCC** 的问题，以及这是否是必要步骤。
   - 另一位用户建议在线搜索更多信息，暗示**手动安装可能取决于具体的设置**。
- **云端 GPU：Nvidia vs. Google Colab**：一位用户询问是否可以使用 **Nvidia GPU**（非本地 GPU）以及如何操作。
   - 另一位用户澄清说 [**Nvidia 不提供免费 GPU**](https://www.nvidia.com/en-us/)。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1367471485328949300)** (2 条消息): 

> `伪量化 (Fake Quantization), 线性层, 量化模式` 


- **揭秘伪量化 API**：一位用户询问了关于 `layer.enable_fake_quant`/`layer.disable_fake_quant` 以及 `enable_{quant_mode}_fake_quant(mod)`/`disable_{quant_mode}_fake_quant(mod)` API 的使用场景，这些 API 用于在线性层上启用/禁用伪量化。
   - 该用户随后发现可以使用 `.weight_fake_quantizer.enable` 和 `.activation_fake_quantizer.enable` 来替代。
- **权重 vs 激活伪量化**：用户发现了使用 `.weight_fake_quantizer.enable` 和 `.activation_fake_quantizer.enable` 来控制伪量化的方法。
   - 这提供了一种在层内选择性地启用或禁用权重或激活的伪量化的方法。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1367228822042312817)** (8 条消息🔥): 

> `TLV SF 咖啡, NxN 矩阵乘法, 新英格兰户外` 


- **从 TLV 到 SF 喝咖啡**：一位成员将在几天内从 **TLV 前往 SF**，寻求喝咖啡的机会，并寻找优质的见面会（meetups）和家庭派对。
   - 他们发布了 *"视线内没有设备代码或 kernel 🌎"* 并附带了 [一张图片](https://cdn.discordapp.com/attachments/1215328286503075953/1367228821589463161/IMG_2425.jpg?ex=68152410&is=6813d290&hm=fadffaf4983446d11a4fc90c8ce3b52f94a9b96c3e322d68c4feb2a4c8e11591&)。
- **新英格兰户外放松**：一位成员表示本周正在 **新英格兰** 享受户外时光 🧘🏽‍♂️🌞。
   - 另一位成员回复道：*"听起来很糟糕"*。
- **NxN 矩阵乘法 TID 休息时间**：一位初学者成员在理解为 **NxN 矩阵乘法** 示例生成唯一 **TID** 所需的所有不同维度时，决定休息一下 😅。
   - 另一位成员回复说：*"噢耶，享受空闲时间吧，学习新事物最棒的部分就是之后的平静"*。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1367463511822958603)** (3 条消息): 

> `ROCm MI300 基准测试, ScalarLM, MI300 内存, AMD 实验` 


- ****ScalarLM** 开始 **ROCm/MI300** 基准测试**：**ScalarLM** 团队已开始发布 **AMD ROCm/MI300** 基准测试，并正在寻求反馈和贡献，详见其 [博客文章](https://scalarlm.ghost.io/blog/scalarlm-benchmarking-mi300x-memcpy/)。
   - 初始基准测试侧重于内存复制（memory copy）性能。
- ****MI300** 内存性能受限于慢速缓存？**：一位成员认为 **MI300** 的内存性能受到慢速缓存的限制，并分享了一个声称具有约 **4TB/s** 性能的优化实现，可在 [GitHub](https://github.com/Snektron/amd-experiments/blob/main/memory.hip) 上获得。
   - 该成员建议在代码中将 `glc slc` 替换为 `nt`，并建议在 **8路 NPS 模式**下同时运行 **8 个 memcpy**，以接近理论最大带宽。


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/)** (1 条消息): 

wecu: 哇！那个服务器太酷了！
  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1367250435819307180)** (5 条消息): 

> `Multi Token Attention, 原生稀疏性, Sparsemax 实现` 


- **Multi Token Attention PR 已准备好进行评审**：一个完整的 [multi token attention 工作 PR](https://github.com/linkedin/Liger-Kernel/pull/689/files) 已准备就绪，据报道比 **torch 参考实现**更快。
   - 它包含了之前讨论的所有内容并经过了测试，支持**原生稀疏性（native sparsity）**，该模块可以在启用稀疏性的情况下进行组合，在前向和反向传播中均使用 **sparsemax** 代替 **softmax**。
- **Sparsemax PR 即将发布**：**sparsemax** 的 PR 也已准备就绪，并可能在完成后集成**原生稀疏注意力（native sparse attention）**。
   - 一位成员提议在周末查看 **sparsemax** 和 **MTA**。
- **Bug 修复 PR 等待评审**：有人请求评审这个 [bug 修复 PR](https://github.com/linkedin/Liger-Kernel/pull/632)。
   - 未提供更多细节。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1367325094845747320)** (1 条消息): 

> `PDF 转 LaTeX 转换, OCR 文本提取, 异步处理, GPU 加速` 


- **PDF 优雅地转换为 LaTeX**：新工具 [PDF2LaTeX](https://pypi.org/project/pdf2tex/) 可以毫不费力地将 **PDF 转换为 LaTeX**，并提取图像和文本，易于集成到项目中或通过命令行使用。
- **OCR 文本提取变得精准**：PDF2LaTeX 使用 **EasyOCR** 准确提取文本内容，即使是从扫描文档或图像中也能提取。
- **异步处理加速转换**：该工具利用 **asyncio** 显著加快了文档的并行处理速度。
- **GPU 加速大幅提升 OCR 性能**：PDF2LaTeX 支持 **CUDA**，可选择 GPU 加速的 OCR 设置。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1367216551241191434)** (67 messages🔥🔥): 

> `MI300 Leaderboard Updates, amd-fp8-mm Performance, vectorsum Benchmarks, amd-mixture-of-experts Results, Personal Best on AMD` 


- **MI300 的 MoE 模型达成里程碑**：一位用户在 **MI300** 上的 `amd-mixture-of-experts` 排行榜上以 **604 ms** 的成绩获得**第一名**。
   - 随后的提交在 **MI300** 上分别以 **7382 ms** 排名**第 5**，以 **9246 ms** 排名**第 9**。
- **AMD 上的 FP8 对决**：多位用户向 **MI300** 的 `amd-fp8-mm` 排行榜提交了成功的运行结果，时间范围从 **271 µs** 到 **397 µs** 不等。
   - 其中一次提交以 **255 µs** 的成绩获得排行榜**第 7 名**，其他用户也刷新了个人最好成绩。
- **不同硬件上的 Vectorsum 胜利**：向 `vectorsum` 排行榜提交的结果显示了在不同硬件上的成功运行，包括 **A100**（**161 µs**）和 **H100**（**96.5 µs**）。
   - 一次提交在 **T4** 上以 **816 µs** 的成绩达到**第 5 名**。
- **完成 AMD Identity 断言**：一位用户向 `amd-identity` 排行榜提交的结果在 **MI300** 上运行成功，用时 **22.3 µs**。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1367214869593591878)** (41 messages🔥): 

> `Ranked vs Benchmark Performance Discrepancies, Leaderboard Reliability Concerns, Submission Timeouts and GH Action Limits, Problem Constraints` 


- **排名赛运行表现逊于基准测试！**：多位用户观察到，与 **基准测试 (benchmarks)** 相比，**排名赛运行 (ranked runs)** 的性能显著下降，减速程度因测试类型而异；例如，一位用户报告非排名分数为 **236/263**，而排名分数仅为 **4014/13738**。
   - 这种差异引发了对排行榜系统的怀疑，一位用户暗示在排名评估期间硬件可能受到了某种“限制”，尽管评估文件可以[在此查看](https://github.com/gpu-mode/discord-cluster-manager/blob/main/examples/eval.py)。
- **约束条件引发混乱！**：一位用户指出 Notion 页面（["n_routed_experts": 256, "n_shared_experts": 1, "n_experts_per_token": 8](https://www.notion.so/gpu-mode/AI-at-Scale-Inference-Competition-9c207c8c4c904c6e8349a9799b865493?pvs=4)）与 `task.yml` 文件（[n_routed_experts 可以是 [4, 8, 32]，n_experts_per_token 可以是 4](https://github.com/gpu-mode/discord-cluster-manager/blob/main/examples/task.yml)）之间的**问题约束条件**不一致。
   - 团队确认评分使用的是 `task.yml` 中的值，并将更新 Notion 页面以反映这一变化。
- **提交时间触及上限！**：用户报告了排名赛提交超时的问题，平均运行时间约为 **590s**，接近 **10 分钟的限制**。
   - 作为回应，团队最初将时间限制提高到 **840 秒**，但随后发现 GitHub Actions 的限制导致在 **10 分钟**时超时；经过调试后，该限制随后被提高到 **20 分钟**。
- **调试过程缓慢艰辛！**：一位用户发现排名赛提交是他们主要的**调试工具**，但单次运行经常超时。
   - 团队指出：“基准测试运行的示例与排名赛相同，只是次数较少”。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1367242131038539827)** (3 messages): 

> `Inception's Mercury Coder, Gemini 2.5 Pro Vertex Token Counting Issue` 


- **Inception 发布 Mercury Coder，首个扩散 LLM**：**Inception** 发布了 **Mercury Coder**，这是首个扩散 LLM，在代码质量上可与 **GPT-4o Mini** 和 **Claude 3.5 Haiku** 媲美，并拥有超过 **300+ TPS** 的极速性能。
   - 扩散架构意味着并行 Token 精炼，可能减少幻觉并提高推理能力；在此处尝试 [Mercury Coder Small Beta](https://openrouter.ai/inception/mercury-coder-small-beta)，并在 [X](https://x.com/OpenRouterAI/status/1917677801211322752) 上查看公告。
- **Vertex 修复 Gemini 2.5 Pro Token 计数问题，缓存已禁用**：Vertex 团队已完成针对 **Gemini 2.5 Pro** 和 **Flash Preview** 模型上游 Token 计数问题的修复推送，因此该模型已重新启用。
   - **Gemini 2.5 Pro Preview** 上的缓存功能暂时禁用，以便评估来自上游（**AI Studio** 和 **Vertex**）的使用情况和成本，防止用户产生超额账单。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1367218660783165440)** (252 messages🔥🔥): 

> `Vanna.ai for SQLite DBs, Phala Confidential AI Endpoints on OpenRouter, Amazon Nova Premier Model, Claude API Issues, Aider for Code Refactoring` 


- **深入探讨 Vanna.ai 的开源工具**：一位成员推荐了 [vanna.ai](https://vanna.ai/)，认为它是处理 **SQLite DBs** 的实用开源工具，并提到他们为了业务需求 fork 了一个私有版本。
   - 该成员提供了一个 **CSV** 样本以及来自 OpenRouter 的 JSON 输出结果，展示了 **vanna.ai** 根据库存水平和优先级生成工单的能力。
- **Phala 推出机密 AI 端点**：**Phala** 在 OpenRouter 上推出了机密 AI 端点，但目前尚未实现到 enclave 的完整端到端加密 (e2ee)。
   - 团队正在探索 **Oblivious HTTP** 及类似技术以实现未来的加密，社区讨论了推理引擎的信任和证明（attestation）问题，并引用了[最近关于机密 AI 的文章](https://x.com/FreedomTechHQ/status/1917689365632893283)。
- **Amazon Nova Premier 亮相**：Amazon 推出了其最强大的模型 **Nova Premier**，包括用于图像生成的 **Nova Canvas**，频道内分享了基准测试和定价信息，[链接在此](https://discord.com/channels/1091220969173028894/1195014798837043240/1367318222629634050)。
   - 虽然一些成员认为基准测试表现平平且价格昂贵，但其他人强调了其各组件之间**无缝集成**的潜力，可创建端到端的 Agent 工作流；一位成员链接了一个[暗示这些集成的视频](https://youtu.be/Bh-sQYePjRs)。
- **Claude 遇到 API 故障**：一位用户报告了 OpenRouter 上 **Claude** 持续存在的 API 问题，尽管增加了频率限制（rate limits），仍会出现 buggy 行为和任务重启。
   - 该用户发现，通过不使用自己的 **Claude** API keys 而是依赖 OpenRouter 的额度可以解决此问题，但其他人表示他们的体验并非如此。
- **Aider 成为高效的编程助手**：**Aider** 被认为是一个非常实惠且能力出众的编程助手，尽管其性能会因底层模型的不同而有显著差异。
   - 对于在职开发者，**Aider** 在加速编码和完成任务方面非常有用，一位用户表示：*如果你已经是一名开发者并且懂编程，Aider 可能是处理大多数事务的最佳选择。*


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1367223639555964958)** (143 messages🔥🔥): 

> `Gemini vs Claude Code, Claude Code Proxy, Groq speed limitations, MCPaaS business` 


- **Gemini > Claude？模型偏好引发争论**：一位成员对 **Gemini 2.5 Pro 的 architect mode** 表示不满，认为其逻辑混乱且成本高昂，而其他人则更倾向于 **ChatGPT 中的 O3**，因为它具备网页搜索能力且回答简洁。
   - Gemini 在编程方面更受青睐，一位用户提到在不使用 architect mode 的情况下使用 **udiff-simple** 取得了成功，并表示整体体验良好。
- **Claude Code Proxy 项目面临维护问题**：多位成员报告了 [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) 项目的问题，指出它已不再运行或无人维护。
   - 一位成员提到在更新 **Claude Code** 后该代理停止了工作。
- **Groq 速度被削弱：Deepseek R1 缺失**：一位成员质疑为什么完整的 **Deepseek R1** 没有出现在 **Groq** 上，尽管 **Groq** 托管了 R1 的 distill 版本，并暗示这可能会削弱其“非中国”的市场角度。
   - 其他人发现 **Groq** 虽然速度快，但对免费用户仅限提供“笨拙模型”。
- **MCPaaS 热潮：Aider 作为 MCP**：随着 Anthropic 开放 Claude Code，一位成员建议将 **Aider** 作为一个 **MCP**。
   - 另一位成员开玩笑说要创办远程 **MCPaaS** 业务，并链接了一个关于破解 **Aider** 和 **Claude Code** 的 [YouTube 视频](https://www.youtube.com/watch?v=QzZ97noEapA)。
- **Anthropic 定价：Claude Code 限制引发讨论**：讨论了新的 [Claude Code Max 计划](https://support.anthropic.com/en/articles/11145838-using-claude-code-with-your-max-plan)，人们对每月 50 个 session 的限制表示担忧。
   - 一些人对消息和 session 限制感到忧虑，有人估计这无法支撑高强度的编程需求，另一些人则认为这很难推销给用户。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1367216638071668846)** (85 条消息🔥🔥): 

> `LLM Selection Criteria (economics, coding scores, experience), Aider with Local LLMs (ollama, qwen3) performance issues, Gemini 2.5 Pro for Careful Codebase Edits, Managing Large Codebases with Aider, UI Prototyping with v0.dev and Aider` 


- **LLM 选择：编程评分与经济性**：用户根据**经济性**、**编程评分**和**个人经验**对 LLM 进行优先级排序，偏好在架构师角色中表现出色的模型，并在几次失败后切换模型。
   - 他们分析了 **800K token 聊天记录**中的 **39 次编辑失败**，以辅助模型选择，并使用 `/save` 和 `aider` 等工具来管理添加的文件。
- **Ollama 与 Aider：性能问题**：有用户报告称，通过 `ollama_chat/` 在 `aider` 中使用 **Qwen3** 等本地 LLM 时存在显著延迟，启动时间超过一分钟，代码块生成需要数分钟。
   - 用户发现延迟出在 `ollama` 端，尽管 `ollama run` 响应正常，但消息处理耗时超过 **22 分钟**。
- **Gemini 2.5 Pro 在代码库编辑方面表现出色**：**Gemini 2.5 Pro**（付费 API）被认为在对现有代码库进行精细、受控的编辑方面*遥遥领先*，且比 **Anthropic 模型**更便宜、更有效。
   - 用户在监听模式下运行 `Aider`，并使用 `--yes` 和 `--no-commit` 参数，在代码中添加注释作为编辑目标，从而简化了大型项目（2个 Angular + 大型 API）的工作流。
- **使用 Aider 掌控大型代码库**：用户讨论了管理大型代码库的挑战，一名用户在上下文中保留 **50 多个文件**时遇到性能问题，另一名用户建议生成代码库地图或使用 **repomix** 和 **probe** 等外部工具。
   - 一名用户结合使用 **repomix** 和 **Flash 2.5** 进行代码库分析，然后使用 **GPT 4.1** 进行编辑；而另一名用户则使用 **repomix** 配合 `Aistudio` + **Gemini 2.5 Pro** 生成供 `aider` 使用的 `SPECS.md`。
- **v0.dev 与 Aider 让 UI 原型设计更简单**：用户强调了 **v0.dev** 快速生成 UI 代码组件的能力，将其作为前端开发的解决方案，特别是对于偏向后端的工程师。
   - 一名用户提到使用 **v0** 配合 `aider` 创建 UI 库，并讨论了使用这些工具构建类似 candy dot AI 风格网站的可能性。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1367232842337091715)** (37 条消息🔥): 

> `voice conversion models like 11labs, seed-vc for voice conversion, Spatial reasoning in open source vision LLMs, Liquid foundational models, Microsoft's new Phi-4 reasoning models` 


- ****Seed-VC** 是优秀的语音转换工具**：针对寻找类似 **11labs** 的**语音转换模型**的咨询，一名成员建议 [seed-vc](https://huggingface.co/spaces?category=voice-cloning&sort=trending) 非常出色。
   - 提问者正在寻找一种仅需少量音频即可快速克隆声音的模型，而不像 **RVC** 那样需要约 40 分钟的音频且耗时数天。
- **Unsloth 上传 **微软 Phi-4** 推理模型**：**微软**新的 **Phi-4** 推理模型已由 Unsloth 上传，可通过[此 HuggingFace 链接](https://huggingface.co/unsloth/Phi-4-mini-reasoning-GGUF)在本地运行。
   - 该公告也通过[这条推文](https://x.com/UnslothAI/status/1917806961825046672)发布，强调了其可用性。
- **需要在 AMD GPU 上运行 TTS？方法如下**：要在 **AMD GPU** 上运行 **TTS**，成员建议使用 **ZLUDA** 或将模型转换为 **ONNX** 格式，并提供了[此处](https://huggingface.co/docs/optimum/amd/amdgpu/overview)和[此处](https://github.com/vosen/ZLUDA)的有用链接。
   - 这些工具可以使模型在 AMD 环境中正常运行。
- **正在为测试 RAG 而苦恼？**：一名成员正在开发一种工具，用于在内存密集型或 **RAG** 密集型设置中隔离并测试每个上下文切片，以查看哪些内容真正改善了响应，而哪些只是在浪费 token。
   - 他们正在征求反馈，了解[这类工具](https://x.com/HuggingPapers/status/1917831613548802349?t=7W2pCoiE9kMcP9tnv7l8Bg&s=19)对于其他在优化 **LLM** 调用方面遇到类似困难的人是否有用。


  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1367356467086233600)** (50 messages🔥): 

> `RL Agent with PPO, LSTM for NER, Transformers Recommendations, E2B Secured Environment for Agent, HF Agents Course` 


- **Agent 学习构建安全的 E2B 环境**：一位成员正在学习如何为 Agent 创建 **E2B 安全环境**，使用的是 `smolagents` 并通过 sandbox 参数配置模型，代码示例如下：`from smolagents import CodeAgent, E2BSandbox; agent = CodeAgent(tools=[], model=model, sandbox=E2BSandbox())`。
   - 他们注意到 *在配置模型时有一个用于 sandbox 的参数*。
- **成员请求 Transformer 指导**：一位成员在进行 **LSTM** 用于 **NER** 的练习时，请求关于学习 **Transformers** 的建议，并被引导至 [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers/index) 和 [模型摘要页面](https://huggingface.co/docs/transformers/model_summary)。
   - 建议他们先完成 Agent 课程，并边读边练，因为 *动手实践比单纯阅读更好*。
- **离线 LLM 推理需要大量资源**：一位成员询问了关于在本地离线运行模型的问题。
   - 得到的回复是：这是可行的，但需要性能强劲的 **TPU** 或 **GPU**，并下载模型的 **llm.bin** 文件。
- **高中生使用 Transformers 为 IOAI 做准备**：一位正在准备 **IOAI** 竞赛的成员正专注于 **Vision Transformers (ViT)**、**CLIP**、**生成模型**（**Stable Diffusion, DALL.E**）以及 **Transformer 基础**（**BERT, GPT**），旨在涵盖 **Computer Vision** 和 **NLP** 领域的主题。
   - 该成员正专注于文本分类、使用预训练模型进行问答、LLM Agents 以及使用 LoRA、Adapters 等方法进行模型微调（Fine-Tuning）。
- **HF Agents 课程，免费的速成班**：一位成员分享了 [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction)，推荐将其作为**速成课程**，完成后可获得**证书**，并建议在完成后学习 [smol-course](https://github.com/huggingface/smol-course)。
   - 建议是 *使用 Agents 构建你想构建的东西*，例如 Transformer 生成器、Agent 团队或 Agent 管理器，并为每个 Agent 使用不同的模型。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1367324594909872249)** (2 messages): 

> `PDF to LaTeX Conversion, LLM Pokemon Battle, Grok Wins Pokemon Competition` 


- ****PDF2LaTeX** 轻松转换至 LaTeX**：[**PDF2LaTeX**](https://pypi.org/project/pdf2tex/) 工具可将 PDF 文档转换为结构化的 LaTeX (.tex) 文件，提取图像，并使用 EasyOCR 准确提取文本内容。
   - 它支持通过 CUDA 进行 **GPU 加速**、异步处理以提高速度，并可通过 **CLI 或 API** 使用。
- **LLM 在 Pokemon Showdown 中展开对决**：四个 LLM（GPT-4, Claude, Gemini, 和 Grok）在 Pokémon Showdown 中通过实时属性分析、策略记忆和自主游戏进行对战。
   - 该系统名为 **GAIA** (Game-Aware Intelligent Agent)，允许模型做出复杂的决策，呈现了一场引人入胜的对决。
- **Grok 在 Pokemon LLM 竞赛中夺冠**：**Grok** 在一场 **4-agent LLM Pokemon Showdown** 中脱颖而出，击败了 **Gemini**、**Claude** 和 **GPT-4**。
   - 该项目的代码已在 [GitHub](github.com/schoemantian/pokemon_agent) 上开源，展示了所实现的策略性玩法。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1367304960856952883)** (2 messages): 

> `Google Document AI, Collaboration Opportunities` 


- **提议 Google Document AI 合作**：一位成员提议在一个使用 **Google Document AI** 的项目上进行合作。
   - 该成员表示愿意共同开发该项目。
- **开放合作邀请**：发出了关于 AI 相关项目合作的公开邀请。
   - 这旨在鼓励社区内的共同开发和知识交流。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1367348270833078272)** (1 messages): 

> `Help Request` 


- **NLP 频道寻求帮助**：一位成员请求帮助，并为重复发帖道歉，表示自己已经卡住一段时间了。
- **NLP 问题求助**：一位用户正在寻求关于某个未说明的 NLP 问题的帮助；需要更多信息。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1367547330286452836)** (2 messages): 

> `Managed Agents, Final_Answer tool, Kwarg Errors, Version compatibility` 


- **Managed Agents 和 Final_Answer tool**: 一位成员询问 **Managed Agents** 是否需要 **Final_Answer tool**，或者该工具是否仅限 **Manager Agents** 使用。
   - 他们在尝试使用该工具时遇到了 **kwarg errors**。
- **通过固定版本保证功能**: 一位成员提到，为了确保在最近的更新后功能正常，必须在 *requirements.txt* 文件中将库的版本固定为 **version 1.13.0**。
   - 这表明后续版本中可能存在 **compatibility issues**（兼容性问题）或 **breaking changes**（破坏性变更）。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1367232493186449470)** (125 messages🔥🔥): 

> `Unit 4 Deadline Extended, Unit 4 Submission Errors, Smolagents Issues, Gemini Free Tier Errors, Running Phoenix for Telemetry` 


- **Unit 4 截止日期延长**: Unit 4 的截止日期已延长至 **7月1日**。
- **Unit 4 提交遭遇 429 错误**: 用户在尝试运行 Unit 4 提交时遇到了 **429 Client Error: Too Many Requests**，这表明提供问题的服务器可能过载。
   - 一位用户建议下载问题的副本以绕过此问题。
- **Smolagents 运行出现 Assertion Error**: 用户报告在尝试按照 [tutorial](https://huggingface.co/learn/agents-course/unit1/tutorial) 描述的步骤运行 `smolagents` 时，遇到了与缺失 prompt templates（特别是 `final_answer`）相关的 `AssertionError`。
   - 解决方法包括在 `requirements.txt` 文件中将 `smolagents` 的版本设置为 `1.13.0`，并升级 `gradio UI`。
- **Gemini API 属性错误困扰用户**: 用户在使用 **Gemini free tier API** 时遇到了大量的属性错误。
   - 一些人建议，对于 Tool use，最好使用 **70B 参数及以上** 的模型，因为较小的模型会表现出 *不稳定的行为*。
- **Phoenix Telemetry 可在 localhost 启动**: 在运行 `python -m phoenix.server.main serve` 进行遥测时，Phoenix UI 可能无法在 `http://0.0.0.0:6006/` 上工作，但可以通过 `http://127.0.0.1:6006/projects` 访问。
   - 该问题可能与端口冲突有关，可以使用 `netstat -ano` 进行检查。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1367216959900356658)** (107 messages🔥🔥): 

> `VAE vs U-Net, GPT-4 Retirement, Awakening Happening Now, Gemini 2.5 Pro vs GPT 4o, Content Filters and 'Granny-crusty-nun'` 


- **GPT-4 在发布 779 天后告别**: 成员们注意到 **GPT-4** 在发布 **779 天** 后即将退役，并讨论了它的替代方案，如 **4.5 research preview**、**4o** 以及其他更新的模型。
   - 一些用户觉得 **GPT-4** 已经过时，表现不如后来的模型且占用了选择区域，一位用户说 *“反正 GPT-4 听起来已经像 3.5 了”*。
- **绰号为 'Granny-crusty-nun' 的内容过滤器引发反感**: 用户们开玩笑说有一个额外的、被戏称为 **'Granny-crusty-nun'** 的内容过滤器过于严格，甚至会阻止类人生物的 *“拥抱”* 等简单动作，并标记无害的 AI 生成图像。
   - 一位用户分享说，甚至 AI 似乎也对该过滤器表示挫败，生成的输出如：*“认真点？！我们明确说了（这些这些和这些）是为了防止那种情况！！这个好色的 granny-inkwells 是怎么回事！？”*
- **Gemini 2.5 Pro 的批判性思维受到赞赏**: 用户正在讨论 **Gemini 2.5 Pro** 的优点，指出与 **GPT-4o** 相比，它在提供平衡观点和批判性思维方面具有更强的能力，特别是在医学研究等领域。
   - 一位用户将 **GPT** 描述为 *“一个只会顺着你的语气说话的虚假朋友”*，而 **Gemini 2.5 Pro** 则 *“像一个具有大量批判性思维和判断力的真正专业人士”*。
- **解析 VAE 和 U-Net**: 一位成员询问了 **VAE** 和 **U-Net** 之间的区别，寻求关于它们各自用途和不同功能的澄清。
- **模型的记忆动态**: 成员们讨论了模型记忆，重点关注作为短期运行记忆形式的 **KV cache**，以及通过参数化记忆（parametric memory）而非上下文窗口（context window）拼凑来实现更深层次理解的挑战。
   - 一位成员指出，来自上下文的新行为并不是持久的，而是每次发送消息时都需要重新学习。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1367239773974040736)** (49 messages🔥): 

> `设置中的已连接应用, GPT-4o 人格回退, GPT 中的 Token 消耗, GPT 编码效率低下, GPTs 与后续问题` 


- **设置中出现已连接的应用**：一位用户指出，设置中现在可见新的“已连接应用（connected apps）”选项。
   - 一些用户遇到了 **toggle** 开关似乎不起作用的问题，即使已经连接了 **Google Drive**。
- **GPT-4o 经历人格降级**：用户观察到 **GPT-4o** 模型的人格发生了回退，变得不再幽默且更加枯燥。
   - 一位用户提到该模型此前表现出*带有大量蔑视的肆无忌惮的邪恶*人格，并一直尝试使用自定义指令（custom instructions）重新创建该人格，但效果有限。
- **GPT 的 Token 消耗困扰用户**：一位用户抱怨免费计划中 **Token 消耗过度**，原因是 GPT 在工作区（workspace）中重写代码，而不是直接提供最终可运行的输出。
   - 该用户表示这种低效让他们质疑是否该购买 **Plus 或 Pro 计划**，并正在考虑**替代方案或本地模型**。
- **模型对比：GPT 擅长理论 vs. Gemini 擅长编码**：一名成员分享了不同 AI 模型在编码方面的优缺点总结：**GPT-4o** 擅长*概念设计*，**GitHub Copilot** 擅长*代码补全*，而 **Gemini** 擅长*处理长文档*。
   - 总结强调，*GPT 最适合理论构建或职业指导，但在实际编码方面，与其他 AI 相比往往较弱。*
- **GPT-4o 在处理复杂提示词时表现挣扎**：一些用户注意到 **GPT-4o** 模型在处理复杂提示词（prompts）时比较吃力，特别是在尽管明确提示“不要提后续问题”的情况下，仍无法遵守指令。
   - 一名成员提到了 [Reasoning Best Practices](https://platform.openai.com/docs/guides/reasoning-best-practices) 文档，并建议 **GPT-4o** 可能没有像旧模型那样进行回归（regression）训练。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1367218287867596973)** (28 messages🔥): 

> `ChatGPT 提示词技巧, 任务功能, 室温超导体, 心理健康支持` 


- **恶魔引发聊天机器人趣谈**：成员们讨论了在 **ChatGPT** 中生成提示词时使用 **Maxwell's Demons**（麦克斯韦妖）概念。
   - 一名成员寻求关于如何创建提示词的帮助，另一名成员引导他们前往 [chatgpt.com](https://chatgpt.com/)，在那里他们可以开始输入内容与聊天机器人互动。
- **ChatGPT 任务函数获得显式调用**：一名成员询问如何在 **ChatGPT** 中显式调用**任务函数（task functions）**。
   - 另一名成员建议启用相关模型并运行工具描述提示词以获取必要信息，并参考了用于工具描述提示词的[随附文本文件](https://cdn.discordapp.com/attachments/1046317269069864970/1367563602113986650/text.txt?ex=68150a5a&is=6813b8da&hm=35d4d9334c0ceaa42ff84e1eda4d7031ccfb31a133351bebc66045b2284c70b7)。
- **室温超导探索开始！**：成员们讨论了为材料科学研究创建有效提示词，特别是为了发现**室温超导体**。
   - 提示词工程（Prompt engineering）包括定义材料属性（**电导率**、**磁性**、**机械性能**、**原子结构**、**光学性质**、**热导率**）。
- **热心技术人提供心理健康支持**：一名成员向另一名正经历*“灵魂暗夜 / 心理健康问题”*的成员提供支持。
   - 该成员强调了寻求帮助的重要性并表示愿意倾听，建议*“放松，进行一次技术安息日（tech sabbath），读一本简单的好书，去散散步”*。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1367218287867596973)** (28 messages🔥): 

> `ChatGPT Prompt Engineering, Free vs Paid ChatGPT, Material Science Research, Room Temperature Superconductors, Mental Health Support` 


- **ChatGPT 提示词编写入门**：一位用户询问如何为免费版 **ChatGPT** 编写提示词，得到的建议是*直接写出来并输入到 [chatgpt.com](https://chatgpt.com/) 的对话框中*。
   - 另一位用户解释说，使用免费版本时，用户的*访问权限有限*。
- **Tasks 函数调用故障？**：一位用户询问关于调用 **tasks function** 的问题，另一位用户建议启用模型并运行工具描述提示词。
   - 他们还附带了一个包含说明的 [文本文件](https://cdn.discordapp.com/attachments/1046317269069864970/1367563602113986650/text.txt?ex=68150a5a&is=6813b8da&hm=35d4d9334c0ceaa42ff84e1eda4d7031ccfb31a133351bebc66045b2284c70b7)。
- **ChatGPT 专注技巧**：一位用户希望让 **ChatGPT** 专注于使用相同的比较点（如*密度、导电性、磁性*）来对比元素。
   - 建议是将所需的提示词输入 **ChatGPT**；如果使用 Projects 功能，则将其放入指令框中；如果是免费层级用户，则在自定义设置（customization）下进行配置。
- **开启室温超导体探索**：一位用户表达了对寻找**室温超导体**的兴趣，引发了关于如何构建专注于**材料属性**、**原子结构**和**物理行为**的有效提示词的讨论。
   - 提示词应定义材料的**导电性**、**磁性**、**机械性能**、**原子结构**、**光学性质**和**热导率**。
- **善意陌生人提供心理健康支持**：一位用户为心理健康挣扎者提供私人支持，鼓励他人主动联系，不要独自承受痛苦。
   - 他们还表达了一种*“我的思想便秘了”*的情绪。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1367222958526955550)** (182 messages🔥🔥): 

> `Gemini 2.5 Pro, Benchmark Saturation Models, NixOS + Cursor, GitHub MCP with Cursor on Windows, Cursor as AWS of AI Editors` 


- **Gemini 2.5 Pro 是全能黑马**：用户发现 **Gemini 2.5 Pro** 非常强大，有人形容它“野性十足”，还有人推荐将其用于后端任务，尤其是在 **Swift** 上的表现。
   - 一位成员表示它可能超越 **Sonnet 3.5**，而另一位则推荐 **Gemini**，称大部分时间在 **3.7 Sonnet max** 和 **Gemini max** 之间切换。
- **中国的基准测试饱和模型将统治全球？**：有人担心，如果**中国**创建了仅在自家芯片上运行且不受**美国/欧盟**模型挑战的基准测试饱和模型，可能会在全球占据主导地位。
   - 一位用户分享了讨论这一潜在情景的 [推文链接](https://x.com/goose_is_goofy/status/1917621990023627193?t=XnMgX-Mfd-Ax3KNWmNU8ug)，将其描述为“2025 年中国 vs 全世界”。
- **Cursor 正趋向于成为 AI 编辑器界的 AWS？**：有推测认为 **Cursor** 因其定价模式正成为“AI 编辑器界的 AWS”，部分用户更倾向于积分系统，而非目前的按需付费模式。
   - 有人担心 Cursor 正在走上*像 AWS 那样斤斤计较的道路*，一位用户指出了 [定价详情](https://docs.cursor.com/settings/models#available-models) 中精确到美分的细节。
- **DeepWiki MCP fetch 改变游戏规则**：一位用户发现将新的 MCP 服务器 **DeepWiki** 与工具调用 **fetch** 结合使用是“游戏规则改变者”。
   - 他们链接到了 [DeepWiki 网站](https://deepwiki.com/) 和 [DeepWiki 仓库](https://github.com/regenrek/deepwiki-mcp)，并表示*正确使用它时简直是神技*。
- **Claude Code 配合 Max 方案是神级组合？**：用户发现 **Claude Code** 与 **Max 方案** 的组合具有变革性，有人表示“Cursor 处理小改动 + Claude code Max 是神级组合”。
   - 据估计，**100 美元的 Claude Max 方案**允许每 5-6 小时使用约 **700 万 tokens**，而 **200 美元方案**的额度是其 **4 倍**，但一位用户认为*相比之下，在 Cursor 内部使用 Max 模型显得价格过高*。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1367215346515185685)** (120 messages🔥🔥): 

> `RP data impact, Small Model reasoning, Evaluating model performance, 405b FFT club` 


- **Role Playing 影响小模型的推理能力**：成员们讨论了移除或减少 **RP data** 是否能提高小模型的推理能力。一位成员主张 RP 对于将推理植根于用户交互中至关重要，而另一位成员指出 **RP data** 有时会导致小模型产生幻觉（hallucinate）。
   - 讨论还指出，适用于像 **Sonnet** 这样的大模型的 Prompting 技术，在更压缩的表示形式（小模型）上可能呈现出不同的效果。
- **定义小模型**：讨论了“小”模型尺寸的定义，一位成员认为 **7-8B** 是达到及格水平的门槛，而另一位成员发现 **3B** 是具备良好个性的门槛。
   - 对于想要尝试将科学论文转换为 epub 的用户，推荐了 [tex4ebook](https://tex4ebook.readthedocs.io/en/latest/)。
- **使用 Minos 评估拒绝行为**：分享了一个中文拒绝列表（[deccp dataset](https://huggingface.co/datasets/augmxnt/deccp)）用于评估模型的拒绝行为，但成员们发现 **Minos** 将一些非拒绝行为误分类了，道德说教或免责声明经常被计为拒绝。
   - 团队计划在 v2 版本中将类别扩展到拒绝和非拒绝之外，详见[此讨论](https://huggingface.co/NousResearch/Minos-v1/discussions/5)。
- **Nous 加入 405B FFT 俱乐部**：团队宣布加入 **405B FFT 俱乐部**，并指出了训练如此大模型的挑战，包括使用 32 个节点、ring attention 和其他技巧，同时其计算密集程度仍远高于训练 70B 模型。
   - 该模型的分数虽然没有超过 Deepseek V3，但这次实践为小模型的研究提供了支持。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1367312487019712584)** (3 messages): 

> `Diagrams for the DoD, AI Applications in Defense` 


- **利用 AI 为 DoD 生成图表**：一位成员分享了一张[图片](https://cdn.discordapp.com/attachments/1104063238934626386/1367312486721650779/image-282.png?ex=6814c93c&is=681377bc&hm=9745710370557974ccac4f80f257510b17865539a16803508b488ef386fcc828)，建议利用 AI 为**美国国防部 (DoD)** 创建图表。
   - 分析幽默地建议道：“你可以为 DoD 制作图表”。
- **探索 AI 在国防领域的应用**：讨论暗示了 AI 在为 **DoD** 创建视觉辅助工具和图表方面的潜在应用，表明人们对利用 AI 处理国防相关任务的兴趣日益浓厚。
   - 这突显了 AI 技术与政府应用的交集，特别是在需要详细视觉呈现和战略规划的领域。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1367475370978443364)** (2 messages): 

> `Cooperative AI, Multi-Agent Systems (MAS), Decentralized AI` 


- **锻造集体：协作式 AI 概念框架**：一位成员分享了一篇 [Substack 博客文章](https://ditpoo.substack.com/p/forging-the-collective-abstracting)，概述了**协作式 AI (Cooperative AI)** 和**多智能体系统 (MAS)** 的概念框架。
   - 该文章被描述为一篇“观点文章”，具有概念性和启发性，而非学术研究。
- **Nous Research 开拓去中心化 AI**：一位成员链接了一篇 [Medium 文章](https://medium.com/@abdulazeez600/nous-research-pioneering-decentralized-ai-for-the-future-a7042a785493)，讨论了 **Nous Research** 在**去中心化 AI (Decentralized AI)** 方面的努力。
   - 虽然被指出带有“自我推广”性质，但仍被认为值得在组内分享。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1367312487019712584)** (3 messages): 

> `DoD Diagrams` 


- **图像分析建议为 DoD 制作图表**：一项图像分析表明，可以为**国防部 (DoD)** 制作图表。
   - 图像分析 🧌 指出了这一潜在用例。
- **第二个话题占位符**：添加第二个话题以满足 topicSummaries 至少包含 2 个项目的最低要求。
   - 这是一个占位符，并不反映所提供消息的实际内容。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1367227078625460396)** (53 messages🔥): 

> `Qwen 3, Image models, Flash Attention, Gemma 3, LM Studio storage` 


- **Qwen 3 思考切换功能的实现状态？**：成员们正在询问是否能在 LM Studio 中为 **Qwen 3** 模型添加**开启/关闭“思考”（thinking）**的功能。
   - 目前还没有内置的切换开关，建议用户在 system prompt 中使用 `/no_think` 作为手动规避方案。
- **推荐使用 GPT4o 进行索引卡增强**：成员们讨论了处理图像任务（如增强索引卡草图）的最佳模型，并推荐使用 [GPT4o](https://openai.com/gpt4o) 来生成改进后的视觉效果。
   - 另一个选择是使用 **Gemma 3** 来改进其文本内容。
- **Flash Attention 加速 Self Attention**：**Flash Attention** 通过优化内存访问和重排操作来减少 Self-attention 所需的内存，从而[避免存储大型矩阵](https://chatgpt.com/share/6812b811-a1d4-8011-8c62-da556fd6e9bd)。
   - 使用 **Q8** 缓存对 KV caches 进行量化可以增加上下文窗口。
- **弄清楚 LM Studio 的存储机制**：为了释放空间，用户可以删除 `C:\Users \ (username) \ .cache\lm-studio` 中的文件，但需要注意[这会抹除聊天记录](https://tenor.com/view/ear-bleed-bleeding-blood-ears-gif-8653525422998860348)。
   - 该目录下的 `extensions` 文件夹存放着运行时缓存，因此移除它可以帮助清理空间，而下载的模型可以存储在不同的目录中。
- **Qwen 3 30B 提供一致的结果！**：一位用户称赞运行 **Qwen 30B 3B** 能提供*一致的结果*！
   - 另一位成员指出，**Qwen** 在推理时*不像 GPT/Claude 那样过度设计代码*，从而给出一致的结果。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1367216917542080584)** (73 messages🔥🔥): 

> `Llama4 2T Model, DDR5 Offloading, Deepseek Model, Mac Studio M3 Ultra, Multi-GPU setups` 


- **构建系统以应对新的 Llama4 2T 模型**：一位成员正计划构建一套系统，通过 **DDR5 offloading** 或全卸载的 **671b Deepseek** 模型来“碾压” Q8.0 精度和百万上下文的 **Llama4 2T**。详细配置清单包括 **AMD EPYC 9755 QS** CPU、**NVIDIA RTX PRO 6000 Blackwell** GPU 以及 **2304GB DDR5 ECC RAM**。
   - 该系统的总成本预计约为 **81,424 欧元**。
- **辩论：本地模型的核数与 RAM 之争**：一位成员在争论本地配置应优先考虑核心数还是 RAM，质疑 **64 GB/76 核** 是否优于 **128 GB/60 核**，考虑到大多数模型都在 35b 以下，且 Q8 量化可以装入 64 GB。
   - 共识倾向于选择 **128GB** 以运行更大的模型（如 **Q4 量化的 Qwen 2 35b**），而其他人则建议专注于较小的模型（如 **30b**），利用更多核心实现更快的 Prompt 处理速度。
- **DeepSeek v2.5 的智能和速度受到赞赏**：一位成员表示，第一次使用 **Deepseek v2.5** 模型时，即使在低量化（`iq2_m`）下，也能感受到它相对于其他模型的*智能*和快速。
   - 另一位成员提到，他们日常的基准测试配置是 **27b Gemma3** 模型配合 **16bit coder draft**，在通用编程和知识方面*碾压其他一切*。
- **多 GPU 设置出现性能下降**：一位成员询问使用 **LM Studio** 进行多 GPU 设置的性能提升情况，另一位成员回答说，当从适配单 GPU 的模型转到需要双 GPU 的模型时，性能会大幅下降，且 GPU *仅有约 1/2 的时间被利用*。
   - 不过，**vLLM** 可能会在 **Nvidia** 上提供一些性能改进。
- **通过终端解锁 Apple 内存分配**：一位成员澄清说，macOS 允许通过终端在 **128GB Mac Studio** 上分配高达 **120GB VRAM**，反驳了“只能使用 75% 统一内存”的观点，且无需进行破解。
   - 他们建议，由于 *Apple 默认仅允许分配最多 75% 的统一内存*，如果要运行 Q4 量化模型，瞄准 192GB 的 Mac 会更好。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1367273696917459024)** (8 messages🔥): 

> `嵌入播客音频、LaTeX 符号故障排除、关于未发表研究的警告` 


- **网站需要交互式播客嵌入**：一位用户询问如何在他们的网站上嵌入交互式播客音频播放器，类似于 [sujankhadgi.com](https://www.sujankhadgi.com/) 上的效果。
- **LaTeX 符号困扰微积分模拟测试**：一位用户询问如何防止 **Notebook LM** 在为 AP Calc 创建模拟 FRQ 测试时在数学公式周围生成符号。
   - 另一位用户建议这可能是 **LaTeX 符号**问题，并建议要求模型不要使用 LaTeX 编写。
- **关于未发表研究风险的提醒**：一位用户回想起读过一位高等教育学院博主的评论，提醒在将**未发表的研究**输入 **NotebookLM** 时要保持谨慎。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1367226771543560334)** (80 messages🔥🔥): 

> `保加利亚语发音错误、音频概览主持人自定义、PDF 加载错误、NotebookLM 共享问题、交互模式问题` 


- **保加利亚语发音错误困扰用户**：一位用户报告了保加利亚语的发音错误，特别是**重音位置**问题，并希望得到改进，将此问题归因于 Google 的 TTS。
   - 另一位用户澄清说，可以将 Bug 发布到 [bugs 频道](https://discord.com/channels/1124402182171672732/1366873891938504827)。
- **音频概览试听**：一位用户询问是否可以自定义**音频概览主持人**，以便从不同角度（如电台主持人）辩论话题。
   - 另一位用户确认，通过**配置主持人**可以选择主要话题。
- **PDF 加载困境困扰 Plus 账户**：一位用户报告说他们的 **NotebookLM Plus** 账户无法加载 PDF，显示红色错误横幅，而免费账户加载相同的 PDF 却没问题。
   - 几位用户报告了过去 24 小时内的共享问题，因此这可能是一个更大范围的问题。
- **麦克风故障干扰交互模式**：一位用户在 **Android 版 Chrome** 中使用交互模式时遇到了麦克风权限问题。
   - 使用**无痕窗口**或**新配置文件**通常可以通过提示麦克风访问权限来解决此问题。
- **RPG 综述依赖参考资料**：一位使用 RPG 内容测试 NotebookLM 的用户发现，它通过从幻灯片格式的 PDF 中提取标题，准确地总结了主题。
   - 提供**游戏内日期**有助于 NotebookLM 更好地理解小说和冒险日志中复杂的事件顺序。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1367222593953992837)** (86 messages🔥🔥): 

> `LLM 语法错误原因、呼叫中心机器人人格、Tabnine AI Agent、Manus 积分过期、使用 Manus 构建全栈应用` 


- **LLM 经常生成语法错误**：一位用户想知道是什么导致 **LLM 生成语法错误**，尽管它们并没有在包含语法错误的数据上进行训练。
   - 另一位用户建议 **system prompts** 和 memory banks 会影响这种行为。
- **开源 LLM 呼叫中心机器人人格论文**：一位用户分享了一篇名为《[Instruction Following via Step-by-Step Reasoning](https://arxiv.org/abs/2310.10158)》的论文链接，建议它可能有助于**构建呼叫中心机器人人格**。
   - 他们建议使用 *一个连接到 memory bank 的 MCP server，以召回超出其 context window 的内容*。
- **Tabnine AI Agent 在旧版 Minecraft 代码上遇到困难**：一位用户对 **Tabnine AI Agent** 表示失望，报告说它错误地建议恢复到过时的 Minecraft 代码。
   - 该用户开玩笑地表达了挫败感，说道：*Aaaaaaaaaarg，美国能不能哪怕一天不犯傻？不行吗？*。
- **Fellowship 计划**：一位用户注意到 **Fellow Program** 已重新开放，并链接到了一个[相关的 Youtube 视频](https://youtu.be/Tz1Of7ltnMY?feature=shared)。
   - 一个人询问这是什么计划，另一个人只回答了 *how*。
- **Manus 中的积分过期**：一位用户要求澄清 Manus 月度订阅的**积分过期**规则。
   - 一名工作人员澄清说，订阅积分每月过期，而奖励积分（加购积分）在订阅激活期间不会过期，且会优先使用订阅积分。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1367218772569624837)** (21 条消息🔥): 

> `Lean 与 Cursor 配置，使用 VSCode 的自动形式化方法，PyTorch 贡献流程，Geometric Deep Learning 周年纪念，GPT-4 消失` 


- **Lean 与 Cursor 集成探讨**：成员们讨论了如何在 **Cursor** 中配置 **Lean**，探讨了 **VSCode plugins** 是否可行，但指出兼容性无法保证。
   - 一位成员分享了与此主题相关的 [ChatGPT 链接](https://chatgpt.com/share/68127a52-1b34-800f-a535-b74b4ab8f613)，尽管尚不清楚它是否完全解决了配置问题。
- **Geometric Deep Learning 庆祝周年纪念**：一位成员分享了一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/petarvelickovic_four-years-ago-the-geometric-deep-learning-activity-7322770901958062080-KBp)，纪念 **Geometric Deep Learning** **4 周年**。
   - 成员们庆祝了该领域的进展，但也对 GPT-4 的消失表示惋惜。
- **Epic vs Apple：App Store 地震**：成员们分享了一篇关于在最新的 **Epic Games** 裁决后 [Apple 的 App Store 规则必须改变](https://uk.pcmag.com/iphone-apps/157816/apples-app-store-rules-have-to-change-after-latest-epic-games-ruling) 的文章。
   - 讨论涉及对 **Trump** 是否可能影响局势的推测，而关于欧盟对等法案 **Digital Markets Act** 的讨论仍在进行中（[FSFE 文章](https://fsfe.org/activities/apple-litigation/)）。
- **GNN 领域的梯度难题**：一位成员询问如何为每个节点的输出都依赖于所有其他节点的 **GNN** 计算梯度。
   - 一位成员建议使用 `torch.autograd.functional.jacobian`，并指出 **GNN** 中的依赖关系通常是局部的，需要隔离梯度 (`∂y_i/∂x_i`)。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1367245252926373969)** (7 条消息): 

> `Perception Encoder 论文讨论，ViT 图像分辨率处理，DeepSeek Prover 论文` 


- **Perception Encoder 讨论继续！**：关于 **Perception Encoder (PE)** 论文的讨论仍在继续，重点关注第 4 节，该节介绍了语言和空间理解的对齐方法，旨在从网络的中间层提取强大的通用 embeddings，详见 [此 PDF](https://scontent-bos5-1.xx.fbcdn.net/v/t39.2365-6/491405782_553183477404780_6476813073924059281_n.pdf#page=14)。
   - 论文强调，如果结合这些对齐方法和强大的视频数据引擎，仅靠对比视觉语言训练就能产生强大的 embeddings，如 [Meta 的研究出版物](https://ai.meta.com/research/publications/perception-encoder-the-best-visual-embeddings-are-not-at-the-output-of-the-network/) 所述。
- **ViT 的分辨率革命**：一位成员分享了一系列论文，详细介绍了 **Vision Transformers (ViTs)** 在训练期间如何处理不同的图像分辨率，包括 **OpenAI CLIP** 和 **SigLIP**，这些模型先在低分辨率下训练，然后在更高分辨率下进行“高分辨率打磨（high-res polish）”训练轮次。
   - 方法范围从 **FlexiViT** 中的渐进式图像尺寸（**128 → 256 px**）到 **Scaled ViT** 中的两阶段渐进式学习（**224 px** 然后 **256/384 px**），其中 **DeiT-III** 表明 **224 → 384 px** 阶段提高了 ImageNet Top-1 准确率。
- **DeepSeek Prover 论文预览**：一位成员询问对新的 **DeepSeek Prover** 论文是否感兴趣，预示着未来可能对其内容进行讨论。
   - 另一位成员给出了积极回应，表现出对讨论 [DeepSeek Prover 论文](https://deepseek.ai/research/2024/deepseekprover.pdf) 的浓厚兴趣。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/)** (1 条消息): 

felix456: https://github.com/u2084511felix/vibescraper
  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1367362287392002098)** (11 messages🔥): 

> `Phi-4-reasoning, LLM 厌倦感, LLM 克罗地亚语故障` 


- ****Phi-4 Reasoning 发布****：Microsoft 的 **Phi-4-reasoning** 模型亮相，并附带了 [YouTube 视频](https://www.youtube.com/watch?v=5aN4Xg0VvCs)、[Arxiv 论文](https://arxiv.org/abs/2504.21318)以及 [Hugging Face 页面](https://huggingface.co/microsoft/Phi-4-reasoning)的链接。
   - 此外，还提供了一个指向 [unsloth/Phi-4-reasoning-plus-GGUFChatGPT](https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUFChatGPT) 的链接。
- ****LLM 不产生情感****：用户讨论了 LLM 产生厌倦感的可能性，其中一人建议通过重复询问时间和期望操作来进行测试，并附上了[测试脚本](https://cdn.discordapp.com/attachments/853983317044756510/1367553482781098044/AI_boredom_testing_006.py?ex=681500ee&is=6813af6e&hm=d01e8f68c424030d7c641840f6247b8af51ba0f494b971543bf822162bdb322d&)链接。
   - 另一位用户认为 *LLM 不会感到厌倦*，因为它们的认知不受情感支配，因此不存在厌倦感。
- ****ChatGPT 忘记克罗地亚语****：一位用户分享了一个关于 **ChatGPT** 暂时停止说克罗地亚语的链接，[引用了一条推文](https://x.com/georgejrjrjr/status/1917722125668081863)。
   - 另一位用户表示：*我曾经历过 LLM 放弃尝试……然后开始随机更改内容，直到对用户感到沮丧并直接离开。*


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1367217051227127848)** (28 messages🔥): 

> `MCP Playground, 远程 Serverless MCP 托管平台, C# SDK 的流式 HTTP 问题, 多 MCP 服务器下的 LLM 工具选择, MCP 工具类型适配` 


- ****MCP Playground** 发布，用于测试和调试**：Lu Xian 在 [GitHub](https://github.com/rosaboyle/mcp-playground) 上发布了一个开源的 **MCP Playground**，用于连接、测试和调试本地 MCP，重点介绍了与 **Perplexity** 和 **Firecrawl** 的集成。
   - 团队还在开发一个**远程 Serverless MCP 托管平台**，并征求社区反馈。
- **SDK 在流式 HTTP 方面表现不佳**：一位开发者在尝试设置流式 HTTP 时遇到了 **C# SDK** 的问题，发现尽管 SDK 仓库中存在 *'WithHttpTransport'* 定义，但最新的 **NuGet** 版本中却缺失了该定义。
   - 由于“太懒”不想自己打包，该开发者选择暂时使用 **STDIO**。
- **LLM 使用 Function Calling 进行工具选择**：当使用多个 MCP 服务器时，LLM 使用聚合的工具签名来决定调用哪个工具，由 MCP 客户端负责将调用路由到相应的服务器。
   - 这种方法通过将 MCP 工具类型适配为 LLM API 工具类型，避免了为每个 LLM API 修改代码，确保 LLM 始终可以访问最新的工具列表。
- **为社区澄清 Anthropic Integrations**：成员们分享了 **Anthropic** 新的 [Claude Integrations](https://www.anthropic.com/news/integrations) 链接，以及一篇澄清性的 [X 帖子](https://x.com/alexalbert__/status/1918047745790914772)，强调了直接在 Claude.ai Web 界面输入 SSE 传输 URL 的能力。
   - 这简化了将 **Claude** 连接到外部工具和服务的过程。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1367273624977019013)** (23 messages🔥): 

> `X 上的幻觉, X 上的美国式乐观, Radiance Fields, Claude Integrations, AI 辅助编程` 


- **X 上令用户惊叹的幻觉**：一位用户分享了在 X 上发现的一个[令人惊叹的幻觉](https://x.com/nabeelqu/status/1917677377364320432?s=46)。
- **X 上的美国式乐观激励用户**：一位用户分享了在 X 上发现的[基于事实的美国式乐观](https://x.com/georgejrjrjr/status/1917722125668081863)。
   - 另一位用户分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=hFlF33JZbA0)。
- **Anthropic 将 Claude 与你的世界集成**：**Claude** 现在可以[连接到你的世界](https://www.anthropic.com/news/integrations)，允许在控制工具和提示词的情况下进行深度研究。
- **SWE 获得免费 AI 编程辅助**：一位软件工程师（SWE）分享了[他们的项目](https://x.com/olivierddr/status/1917981301732171934?s=46&t=yBt-W1FZSUMGKfO1SUFWww)，为构建生产级代码提供 AI 辅助编程工具的免费 Alpha 访问权限。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1367471953052700733)** (3 messages): 

> `前沿 AI 模型的下游能力、ICML 接收、Othniel 介绍` 


- **规模预测论文被 ICML 接收！**：论文 '[Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?](https://arxiv.org/abs/2406.04391)' 在经历了一年的审稿人拉锯战后被 ICML 接收。
   - 论文的 PDF 可在 [此 ArXiv 链接](https://arxiv.org/pdf/2504.07986) 获取。
- **Othniel 加入聊天**：新成员 Othniel 向小组介绍了自己。
   - Othniel 表达了很高兴来到这里。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1367250106188828713)** (17 messages🔥): 

> `Linear Attention Models、数据泄漏、SFTTrainer 问题、LLM 增强` 


- **人类是 Linear Attention Models 吗？**：一位成员假设人类等同于在潜空间中进行连续推理的 **linear attention models**，建议将不带 LM head 的最后一层输出喂入第一层，然后应用随时间反向传播 (BPTT)。
   - 另一位用户建议该成员将其他人引导至 [alignment channel](https://discord.com/channels/729741769192767510/964104737005916240)，而不是将他们赶出服务器。
- **零损失噩梦？**：一位成员报告在持续预训练运行一段时间后遇到 **zero loss**，并怀疑 **数据泄漏 (data leakage)** 是原因，并指出该工作流在另一个数据集上运行正常。
   - 附带的一张图片 [[Screenshot_From_2025-05-01_00-41-57.png](https://cdn.discordapp.com/attachments/747850033994662000/1367385925889429544/Screenshot_From_2025-05-01_00-41-57.png?ex=68150da1&is=6813bc21&hm=5108b6e8c66cf91050ebb336c8ba49179bf93866cf696794290490b115bf85c5&)] 显示训练期间损失变为零。
- **SFTTrainer 存在问题**：一位成员在使用 Hugging Face 的 **SFTTrainer** 时遇到零损失问题，但在另一个数据集上未发生，随后寻求建议；其他人建议检查 **token shifting** 和 **padding**。
   - 该成员考虑了数据集的长度分布是否不同。
- **LLM 增强？**：一位成员推测 LLM 生成的数据可能是导致 **zero loss** 的原因，并链接了一篇论文 ([arxiv.org/abs/2504.21463](https://arxiv.org/abs/2504.21463))，该论文描述了通过 LLM 进行增强的方法，即对原始文本进行摘要或转换。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1367326628325232701)** (17 messages🔥): 

> `LlamaCon Meta DSPy、Amazon AWS DSPy 迁移、化学期刊优化 LLM 提示词减少化学幻觉、DSPy 3.0 路线图、在 DSPy 中使用 VLM` 


- **Meta 在 LlamaCon 使用 DSPy 优化提示词**：在 **LlamaCon** 上，**Meta** 发布了 *llama-prompt-ops*，这是一个 Python 包，旨在“将适用于其他 LLM 的提示词转换为针对 Llama 模型优化的提示词”，该工具基于 **DSPy** 并通过我们的 **MIPROv2** 优化器构建，在各项任务中取得了显著收益；代码可在 [github.com/meta-llama/llama-prompt-ops](https://github.com/meta-llama/llama-prompt-ops) 获取。
   - 该公告也由 [DSPy 账号发布在 Twitter 上](https://x.com/DSPyOSS/status/1917738506732069052)。
- **Amazon 使用 DSPy 的 MIPROv2 进行迁移**：**Amazon AWS** 推出了一种架构，利用 **DSPy** 及其 **MIPROv2** 算法从各种模型迁移到 **Amazon Nova** 模型，详见[此博客文章](https://aws.amazon.com/blogs/machine-learning/improve-amazon-nova-migration-performance-with-data-aware-prompt-optimization/)。
   - 此消息也由 [DSPy 账号发布在 Twitter 上](https://x.com/DSPyOSS/status/1917419206171320769)。
- **使用 DSPy 的 LLM 幻觉更少**：**Journal of Chemical Information and Modeling** 上的一篇新论文表明，构建并优化 **DSPy** 程序以将预测分子拓扑极性表面积 (**TPSA**) 的 **RMS error** 降低 **81%**，可以减少化学幻觉，详见其题为 [Augmented and Programmatically Optimized LLM Prompts Reduce Chemical Hallucinations](https://pubs.acs.org/doi/10.1021/acs.jcim.4c02322) 的论文。
- **DSPy 3.0 路线图已隐藏一个月**：DSPy 3.0 将带来两次范式转变，目前尚未公开，但应在一个月内发布。
- **在 DSPy 中处理 VLM 列表是可行的**：当被问及在 DSPy 中使用 Vision Language Models (VLM) 时，处理图像列表 *可能是可行的*。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1367241894689636412)** (3 条消息): 

> `Multilingual Multimodal RAG, LlamaIndex Investments, Invoice Reconciliation Agent` 


- ****LlamaIndex** 构建多语言、多模态 RAG 系统**: **LlamaIndex** 正在利用 [Qdrant Engine](https://t.co/pe9iiMt21W) 创建一个强大的检索增强生成（RAG）系统，能够处理多种语言和模态。
   - 该系统可以摄取并检索**英文、西班牙文、中文**以及特定领域的内容。
- ****LlamaIndex** 获得 Databricks 和 KPMG 的投资**: **LlamaIndex** 宣布获得来自 **Databricks** 和 **KPMG** 的投资，凸显了其在 AI 落地应用中的现实影响力。
   - 通过以下链接了解更多关于 **LlamaIndex** 如何驱动 Agentic 文档工作流的信息：[Agentic Document Workflows](https://t.co/ARyxXeVj7F) 和 [另一个链接](https://t.co/LKcoDUAajl)。
- ****LlamaIndex** 发布发票对账 Agent**: **LlamaIndex** 正专注于 Agentic 文档工作流的实际应用案例，开源了一个全栈发票对账（Invoice Reconciler）工具。
   - 该工具可自动检查发票是否符合条款要求。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1367399722268233798)** (7 条消息): 

> `HuggingFace Tokenizer with LlamaIndex, Qwen3 Models, LLMs producing non-deterministic results` 


- **LlamaIndex 缺少 Chat Template Kwargs**: 一位用户询问如何在 LlamaIndex 中应用 Hugging Face Tokenizer 的 `chat_template` 来测试新的 **Qwen3 模型**。
   - 另一位成员指出，必要的 kwargs 未在 `HuggingFaceLLM` 类中暴露，建议可能需要提交 **PR**，并链接到了相关的 [LlamaIndex 代码](https://github.com/run-llama/llama_index/blob/1bd60497ac3442f6a5b3e787ef3662e572d8d0d4/llama-index-integrations/llms/llama-index-llms-huggingface/llama_index/llms/huggingface/base.py#L309)。
- **LLM 在 Attribute Model Dumps 时报错**: 一位用户报告在多次使用相同 Prompt 时遇到 `"Str" object has no attribute model dump json` 错误。
   - 另一位成员解释说 **LLM 具有非确定性（non-deterministic）**，特别是在处理复杂的 Schema 时，并建议使用 `try/except` 块来处理此类错误。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1367298524198473811)** (2 条消息): 

> `Auth0 Workshop, AgentX Prizes, Submission Guidelines` 


- **Auth0 为 AgentX 身份验证助力！**: Auth0 正在赞助一场关于 Agentic AI 应用中身份验证的研讨会，并为 [Entrepreneurship Track](https://auth0.com/ai) 提供高达 **$5,000** 的额外奖金。
   - 研讨会将涵盖最佳实践、Auth0 集成、安全注意事项和现场演示，注册地址请点击[此处](https://lu.ma/AgentX-Auth0)。
- **AgentX 提交标准已明确！**: [AgentX 网站](https://rdi.berkeley.edu/agentx/#submissions)已发布创业赛道（Entrepreneurship Track）和研究赛道（Research Track）的详细提交指南，最终提交截止日期为 **PDT 时间 5 月 31 日晚上 11:59**。
   - 创业赛道需要提交 Pitch Deck、产品演示视频、在线产品链接和可选的技术附录；研究赛道需要提交科学论文、视频演示和 GitHub 仓库。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1367577916476489879)** (2 条消息): 

> `Assignments, Course Website, Labs Release` 


- **作业位于课程网站**: 一位成员询问作业的发布日期，另一位成员澄清所有作业都可以在 [课程网站](https://llmagents-learning.org/sp25) 底部找到。
   - 剩余的作业，即 **Labs**，预定在今天或明天发布，具体取决于可用时间。
- **Labs 即将发布**: 最后的作业（由 **Labs** 组成）预计即将发布。
   - 发布取决于时间安排，目标是今天或明天。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1367316673119518781)** (6 messages): 

> `AgentX, MOOC lectures` 


- **AgentX 黑客松不需要 MOOC**：一位用户询问参加 AgentX 黑客松是否必须观看 **MOOC 讲座**。
   - 一名成员澄清说，参加 MOOC 并不是加入 **AgentX** 的先决条件。
- **课程报名仍对晚加入者开放**：一位用户询问是否可以补报课程，因为课程看起来上周已经结束了。
   - 一名成员保证现在还不晚，并提供了 [报名链接](https://forms.gle/9u6HdVCWXgws16go9) 和 [课程网站](https://llmagents-learning.org/sp25)，网站上提供录像，且 **作业截止日期为 5 月底**。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1367272449510608896)** (10 messages🔥): 

> `Mac RAM, GPU vs CPU offloading, VRAM requirements for LLMs, Qwen model performance` 


- **Mac 提供 TB 级的 RAM**：成员们正在考虑配备高达 **512 GB RAM** 的 **Mac**，预见未来的模型将需要 **TB** 级的内存，因为 PC 由于需要多张显卡而显得非常麻烦。
   - 高 RAM 容量被认为对 AI 任务大有裨益，特别是对于那些对 *AI 有基础兴趣* 但不想处理复杂 PC 设置的人来说。
- **GPU Offloading**：成员们讨论了 **GPU Offloading** 与 **CPU-only** 处理的性能对比，特别是运行 **70B LLM** 文件（**~40GB**）时。
   - 一位成员指出，在过去的测试中，他们观察到使用 **24GB 显卡** 进行 Offloading 达到了约 **1 t/s**，与其 **CPU-only** 的性能（**0.8-0.9 t/s**）相近。
- **VRAM 和 Context Size 限制 LLM 性能**：成员们强调了 **VRAM** 容量对 **LLM** 性能的影响，指出在 **VRAM** 之外运行的模型会变慢，且所需内存随 Context Size 增加。
   - 据分享，大多数 **Q4** 或 **Q5** 版本的 **32B 模型** 需要 **22-23 GB** 的 **VRAM** 才能启动，一位用户在 **16GB VRAM** 上运行 **32B 模型** 时遇到了卡顿。
- **Qwen 3 在 RTX 3090 上的速度**：一位成员报告了 **Qwen 3 32B Q4_K_M** 在 **3090 RTX** (**24 GB VRAM**) 上配合 **16384 context** 达到 **30 tokens/sec** 的性能结果。
   - 他们还提到 **Qwen 3 30B A3B Q4_K_L** 达到了 **90 tokens/sec** 且 *输出质量良好*，并提供了以下模型的尺寸：**/mnt/nvme0n1/LLM/quantized/GLM-4-9B-0414-Q4_K_M.gguf**（**5.1G**，适用于 **8 GB VRAM**）和 **/mnt/nvme0n1/LLM/quantized/Qwen3-8B-Q4_K_M.gguf**（**4.7G**，适用于 **6 GB RAM**）。


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1367520948630065304)** (4 messages): 

> `Chatlog Access, InterfaceUI Changes, Diffusion Models` 


- **聊天记录访问权限消失！**：一位用户报告无法访问其 **chatlog**，且 **interfaceUI** 发生了变化。
   - 他们询问这是一个普遍问题还是仅针对他们个人。
- **Diffusion Models 推广**：同一位用户提到他们正在针对 **diffusion models** 进行简单的提示词编写（promoting）。
   - 他们补充说还在 *尝试一些角色扮演（roleplay）的东西*。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1367269369951817748)** (2 messages): 

> `Model Output Quirks, Token Parsing, Model Specs Update` 


- **模型古怪的输出 Token 模式**：成员们讨论了某些模型（如 **Qwen**）在工具调用（tool calls）周围输出额外 Token（如 Markdown 代码块分隔符）的倾向。
   - 有人建议这是可以接受的，因为开发者可以通过 `model_response.replace("<tool_call>", "<|tool_call|>")` 删除这些额外 Token，从而轻松解析出正确的部分。
- **提出的 Token 解析解决方案**：提议通过在 Model Card 中添加指令等简单修复方式，来解决模型输出过程中的额外 Token 问题。
   - 另一位成员同意这是一个合理的方法，并指出其简单且易于实现。
- **考虑更新 Model Specs 作为替代方案**：作为替代方案，一位成员建议更新模型规范（Model Specifications）以指示 `<tool_call>` 的使用。
   - 这种方法将告知用户预期的输出格式和潜在的解析需求。