---
companies:
- google-deepmind
- mistral-ai
- alibaba
- huawei
- openai
- microsoft
- deepseek
date: '2025-05-09T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **Gemini 2.5 Flash** 在人工智能分析指数（Artificial Analysis Intelligence Index）中提升了 **12
  分**，但其成本比 Gemini 2.0 Flash 高出 **150 倍**，原因是其输出 Token 价格贵了 **9 倍**，且推理过程中的 Token
  使用量增加了 **17 倍**。**Mistral Medium 3** 与 **Llama 4 Maverick**、**Gemini 2.0 Flash**
  和 **Claude 3.7 Sonnet** 展开竞争，在提供更强的代码和数学推理能力的同时，价格显著更低。阿里巴巴的 **Qwen3（通义千问）** 系列支持
  **119 种语言** 的推理和多语言任务，并包含一个用于构建应用的 **Web Dev** 工具。华为的 **盘古（Pangu）Ultra MoE** 在昇腾（Ascend）NPU
  上的性能与 **DeepSeek R1** 相当，并拥有新的算力支持及即将开始的 V4 版本训练。OpenAI 的 **o4-mini** 现在支持使用思维链（CoT）推理的**强化微调（RFT）**。微软的
  **X-REASONER** 在通用领域文本上进行后期训练后，能够实现跨模态的可泛化推理。ChatGPT 的**深度研究（Deep Research）**功能与
  GitHub 仓库集成，增强了代码库搜索和报告能力。**AI 工程师世界博览会（AI Engineer World''s Fair）**为即将发售的门票提供早鸟优惠。'
id: MjAyNS0w
models:
- gemini-2.5-flash
- gemini-2.0-flash
- mistral-medium-3
- llama-4-maverick
- claude-3.7-sonnet
- qwen3
- pangu-ultra-moe
- deepseek-r1
- o4-mini
- x-reasoner
people:
- giffmana
- artificialanlys
- teortaxestex
- akhaliq
- john__allard
title: 今天没发生什么事。
topics:
- model-performance
- reasoning
- cost-analysis
- reinforcement-learning
- chain-of-thought
- multilinguality
- code-search
- model-training
- vision
- model-integration
---

**平静的一天。**

> 2025年5月8日至5月9日的 AI 新闻。我们为您查看了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（215 个频道，4687 条消息）。预计节省阅读时间（以 200wpm 计算）：486 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

这是一个相当平静的周末，所以我们要宣传一下[我们的 AI Engineer World's Fair 报道](https://news.smol.ai/issues/25-05-07-aiewf-2025) —— 对于还没买票的人来说，这是获得[早鸟优惠](https://ti.to/software-3/ai-engineer-worlds-fair-2025/discount/AINEWS)的最后机会！

---

# AI Twitter 回顾

**大语言模型 (LLMs) 与模型性能**

- **Gemini 2.5 Flash 性能与成本分析**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1920497711352328557) 报告称，由于 **output tokens 价格贵了 9 倍**，且推理时的 **token 使用量高出 17 倍**，**Gemini 2.5 Flash** 的成本比 **Gemini 2.0 Flash** 高出 **150 倍**。尽管成本高昂，但它在 **Artificial Analysis Intelligence Index** 上 **12 点的提升** 可能会让特定用例的升级变得物有所值。此外，[@Teknium1](https://twitter.com/Teknium1/status/1920740541660086526) 指出，**reasoning models** 的每个 token 通常**更贵**，因为它们会产生**更长的输出**，从而增加了每个 token 的平均成本。[@giffmana](https://twitter.com/giffmana/status/1920719954275352643) 还质疑了**为什么同一模型中的推理输出 token 比非推理 token 更贵**。
- **Mistral Medium 3 性能**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1920295575591006671) 指出 **Mistral Medium 3** 可与 **Llama 4 Maverick**、**Gemini 2.0 Flash** 和 **Claude 3.7 Sonnet** 媲美，在编程和数学推理方面有显著提升。Medium 3 的价格更低，每 100 万 Input/Output token 的价格为 **$0.4/$2**，相比 **Mistral Large 2 ($2/$6)** 价格下降了 **80%/67%**；根据 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1920295585451835522) 的说法，由于回复更冗长，它比 **Mistral Large 2** 使用了更多的 token。
- **Qwen3 模型系列**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1920614690813550930) 宣布了**阿里巴巴的 Qwen3**，这是一个包含 8 个开源大语言模型的系列。这些模型支持可选的推理模式和涵盖 119 种语言的多语言能力，在推理、编程和 function-calling 任务中表现出色。根据 [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1920848175457591406) 的说法，它还具有 **Web Dev** 工具，可以通过简单的提示词构建网页和应用。
- **DeepSeek 模型**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1920328956726632628) 报道了**华为的盘古 Ultra MoE**，它在 6000 块昇腾 NPU 上实现了与 **DeepSeek R1** 相当的性能。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1920749432242340168) 认为 **DeepSeek** 已经建立了一个新的 **LLM 默认标准**。他还指出，已确认 **DeepSeek** 获得了新的算力，并推测 **V4 训练** 即将开始或已经开始，见 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1920733123081306208)。
- **o4-mini 上的强化微调 (RFT)**：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1920531856426143825) 宣布 **OpenAI o4-mini** 现已支持**强化微调 (RFT)**，该技术利用 chain-of-thought 推理和特定任务评分来提高模型性能。[@john__allard](https://twitter.com/john__allard/status/1920585315405676943) 提到其目标是使 **RL 尽可能灵活且易于使用**。
- **使用 X-REASONER 实现泛化推理**：[@_akhaliq](https://twitter.com/_akhaliq/status/1920752791405863000) 和 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1920435270824178089) 讨论了**微软的 X-REASONER**，这是一种仅在通用领域文本上进行后期训练的视觉语言模型，旨在实现跨模态和跨领域的泛化推理。
- **推理训练的可扩展性**：根据 [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1920932361136447740) 的说法，推理训练的快速扩展可能会在一年左右的时间内放缓。

**AI 应用与工具**

- **深度研究与代码库集成**：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1920556386083102844) 宣布，你现在可以将 **GitHub 仓库连接到 ChatGPT 的深度研究 (deep research)** 中，允许 Agent 读取并搜索仓库的源代码和 PR，并返回带有引用的详细报告。[@isafulf](https://twitter.com/isafulf/status/1920572177335669140) 强调，**代码搜索**一直是深度研究的一个主要用例。
- **用于 AI 协作的 Agent2Agent (A2A) 协议**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1920460481510453696) 强调了 **Agent2Agent (A2A)** 协议的重要性。Google 的 A2A 协议旨在成为让它们能够协作的“通用语言”。
- **使用 Qwen Chat 进行 Web 开发**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1920848175457591406) 推出了 **Web Dev**，这是一个在 Qwen Chat 中通过简单提示词构建网页和应用的工具。

**AI 安全与对齐**

- **Scientist AI 作为更安全的替代方案**：[@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1920794672974156254) 展示了他团队的研究方向，称为“**Scientist AI**”，将其作为当前不受控的代理驱动轨迹的一种实用、有效且更安全的替代方案。
- **AI 控制与安全**：[@NeelNanda5](https://twitter.com/NeelNanda5/status/1920471994099020015) 讨论了 AI 控制，强调了通过正确方案实现安全可用性的重要性。

**人物与公司**

- **Fidji Simo 加入 OpenAI**：[@sama](https://twitter.com/sama/status/1920341429655634024) 宣布 [@fidjissimo](https://twitter.com/fidjissimo) 将加入 **OpenAI**，担任 **CEO of Applications** 这一新职位，并向他汇报。包括 [@gdb](https://twitter.com/gdb/status/1920344903466529193)、[@saranormous](https://twitter.com/saranormous/status/1920352615839211881)、[@kevinweil](https://twitter.com/kevinweil/status/1920348319856943114) 和 [@markchen90](https://twitter.com/markchen90/status/1920353685156016488) 在内的多位人士对她的加入表示兴奋。
- **Rob Fergus 担任 Meta-FAIR 新负责人**：[@ylecun](https://twitter.com/ylecun/status/1920556537233207483) 宣布 **Rob Fergus** 成为 **Meta-FAIR** 的新负责人，重新聚焦于 **高级机器智能 (AGI)**。

**通用 AI 讨论与见解**

- **品味与执着的重要性**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1920878606261088422) 强调，**品味、执着和对细节的关注**是让个人脱颖而出的关键品质。
- **长期运行的有状态 Agent**：[@hwchase17](https://twitter.com/hwchase17/status/1920321552932896860) 表示看好 **长期运行的有状态 Agent (long-running, stateful agents)**，并询问谁在构建此类产品。
- **速度是初创公司成功的关键因素**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1920480460318130460) 强调 **速度** 是初创公司成功的关键因素。

**幽默/迷因**

- [@adcock_brett](https://twitter.com/adcock_brett/status/1920320621692559822) 对 [@BasedBeffJezos](https://twitter.com/BasedBeffJezos) 回复了“**lol**”。
- [@Lateinteraction](https://twitter.com/lateinteraction/status/1920329075387752839) 开玩笑说他们“可能不是第一个想到这个笑话的人，但肯定是目前为止最后一个！”。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 1. 高级本地 LLM 推理优化技巧

- [**不要卸载 GGUF 层，卸载张量！200%+ 的生成速度？是的，请！！！**](https://www.reddit.com/r/LocalLLaMA/comments/1ki7tg7/dont_offload_gguf_layers_offload_tensors_200_gen/) ([Score: 636, Comments: 126](https://www.reddit.com/r/LocalLLaMA/comments/1ki7tg7/dont_offload_gguf_layers_offload_tensors_200_gen/)): **OP 演示了通过使用 `-overridetensors` 标志（在 llama.cpp/koboldcpp 中）卸载*单个 FFN 张量*（例如 `ffn_up` 权重）而不是整个 *GGUF 模型层*，在运行像 QwQ IQ4_M 合并模型这样的大型模型时，通过仅将最大的张量保留在 CPU 上，可以在相同的 VRAM 占用下使*生成速度提高 2.5 倍以上*（从 `3.95` 提高到 `10.61` tokens/sec）。这种细粒度的方法通过使用正则表达式过滤器（例如 `\.ffn_up=CPU`）精细管理 VRAM，从而在技术上使所有层都能在 GPU 上执行，而传统的做法是粗粒度的逐层卸载（'--gpulayers N'）。实验结果显示，在 VRAM 受限的情况下提速显著，并指出如果采用更细粒度的自动张量放置，未来推理后端（llama.cpp, koboldcpp）将有改进空间。** 热门评论指出：(1) llama-swap 中的类似技术通过使用每张量覆盖正则匹配，在 Qwen 3 235B 上实现了约 `7.6 tk/s` 的速度；(2) 提速收益取决于硬件和瓶颈，对低端 GPU 或受 CPU 瓶颈严重影响的配置影响最大，并且可能会对非并发张量卸载产生负面影响；(3) 社区对细粒度 GPU/CPU 张量调度的普遍关注。
    - 一位用户分享了针对 llama-swap 的详细 `-override-tensor` 配置，以优化 Qwen 3 235B IQ3_M，报告在 `48GB` VRAM 上速度约为 `7.6tk/s`。特定模式选择性地卸载匹配 `([4-9]+).ffn_.*_exps.=CPU` 的张量，表明了对哪些部分分配给 CPU 或 GPU 的精细控制。
    - 另一位评论者指出，*将额外的张量卸载到 GPU* 仅在 CPU 是瓶颈的低端硬件上能提高速度——否则，卸载非并发张量或层可能会引入性能损失而非收益。优化 CPU 和 GPU 之间的平衡高度依赖于硬件。
    - 一位用户描述了一个实际的提速案例：使用基于 CLI 的张量卸载，以 `4t/s` 的速度运行 Qwen3 32B（支持高达 `32000` 上下文），这比使用 LM Studio 低于 1t/s 的速度有了实质性的改进。这说明了自定义卸载策略对大上下文推理的实际影响。
- [**让 Qwen3 像 Gemini 2.5 Pro 一样思考**](https://www.reddit.com/r/LocalLLaMA/comments/1kigmfo/make_qwen3_think_like_gemini_25_pro/) ([Score: 128, Comments: 18](https://www.reddit.com/r/LocalLLaMA/comments/1kigmfo/make_qwen3_think_like_gemini_25_pro/)): **OP 描述了一种强制 Qwen3 模型进行逐步推理的技术，灵感来自 Apriel-Nemotron-15b-Thinker 方法，通过在 WebUI 函数中始终为输出添加模板前缀（例如，'<think>\nMy step by step thinking process went something like this:\n1.'）。这会产生更结构化、条理化的响应，模仿 Gemini 2.5 Pro 的输出风格，但并不会从本质上提高模型的智能或推理能力。实现细节和代码可在 GitHub ([AaronFeng753/Qwen3-Gemini2.5](https://github.com/AaronFeng753/Qwen3-Gemini2.5)) 上找到。** 热门评论讨论了提示词工程化的逐步推理与原生、基于训练的解决方案的优劣。有人认为 Gemini 的系统化规划与许多开源模型中常见的传统提示词技巧有本质不同，并指出真正的推理模型（在训练期间内置）在基准测试中历来优于单纯的基于提示词的方法。另一条评论指出，这种提示词技巧自 Llama 3.1 等模型以来就已存在，效果“不错”，但并不意味着与原生训练此类行为的模型具有同等功能。
    - 一条评论描述了 Gemini 的推理方法与大多数模型的不同之处：它在回答之前会生成一个高度组织化的响应计划，而不是“等等，但是……”风格的迭代推理。这种方法与开源模型形成对比，后者只能通过提示词模拟这种风格，目前尚不清楚仅靠提示词是否能达到模型原生训练推理过程所带来的性能提升。
    - 提供了关于非推理模型早期提示词技术（如“一步步思考”）的历史背景，这些技术用于诱导逻辑推理，但针对此类行为训练的推理模型在基准测试中显示出比这些仅靠提示词的方法有显著改进。这表明原生集成结构化推理是一项重大进步。

- 有关于通过 prompting 在 Llama 3.1 等开源模型中实现此类推理风格的实用评论，指出虽然效果尚可，但与 Gemini 的原生方法相比，其性能影响仍不确定。

### 2. 用于 Web 开发和无障碍访问的本地及开源 LLM

- [**我制作了一个名为 "LocalSite" 的 "DeepSite" 本地替代方案 —— 让你能通过 Ollama 和 LM Studio 使用本地 LLM 创建网页和按钮等组件**](https://v.redd.it/paflnbaalqze1) ([Score: 105, Comments: 28](https://www.reddit.com/r/LocalLLaMA/comments/1kifny6/ive_made_a_local_alternative_to_deepsite_called/)): **该帖子介绍了 'LocalSite'，这是一个开源工具 ([GitHub](https://github.com/weise25/LocalSite-ai))，旨在作为 'DeepSite' 的本地替代方案，通过 [Ollama](https://ollama.ai/) 和 [LM Studio](https://lmstudio.ai/) 使用本地 LLM（如 GLM-4, Qwen3, UIGEN-T2）生成网页和 UI 组件。该工具还支持通过 OpenAI 兼容的 API 使用云端 LLM，并展示了 GLM-4-9B 创建定价页面的能力。开发过程利用了 Agentic coding 工作流（Augment Code, Gemini 2.5 Pro）。** 评论中的一个技术问题询问是否可以在 prompt 或 UI 中指定 Twitter Bootstrap 或 Laravel 等框架，这反映了用户对框架无关或可定制代码生成的兴趣。目前没有进一步的深度技术争论。
    - 一位用户询问了直接在 prompt 中或通过下拉菜单等专用 UI 元素指定 Twitter Bootstrap 或 Laravel 等框架的技术可行性，这表明了对多框架和可定制组件生成工作流的兴趣。讨论此功能意味着后端需要具备对特定框架代码输出的适应性，并且可能需要进行 UI/UX 设计更改以适应用户友好的框架选择。
    - 另一个建议强调了允许 prompt 编辑和重新生成输出的好处，从而实现对生成的代码或组件的迭代优化。支持此功能可能涉及维护 prompt 状态并使用更新后的用户输入重新调用 LLM 会话，从而促进更具交互性和可定制性的开发体验。
- [**llama-server 的 Vision 支持刚刚上线！**](https://github.com/ggml-org/llama.cpp/pull/12898) ([Score: 213, Comments: 54](https://www.reddit.com/r/LocalLLaMA/comments/1kipwyo/vision_support_in_llamaserver_just_landed/)): **最近的一个 PR 为** `llama.cpp` **的 server 组件添加了统一的 vision（图像输入）支持，利用** `libmtmd` **在单一流水线中同时处理图像 token 和文本。主要的技术变更包括增强了** `struct server_tokens` **以共同处理文本和图像 token，利用 token 到图像块的映射，调用** `libmtmd` **进行图像 token 处理，并在 server API 和 Web UI 中开放了此功能，根据 [multimodal.md](http://multimodal.md/) 支持 base64 和远程图像 URL。尚未解决的问题包括缓存处理、远程图像鲁棒性、错误管理以及扩展文档，如 [PR#12898](https://github.com/ggml-org/llama.cpp/pull/12898) 所述。** 热门评论强调了期待已久的、用于多模态支持的统一架构，对 vision 现在是原生集成而非通过碎片化实现感到欣慰。技术争论较少，重点在于架构内聚性的重要性。
    - 一位用户强调 llama-server 中新的 vision 支持是完全统一的，并指出该实现直接集成了多模态能力，而不是依赖于零散或独立的解决方案。这种架构决策可能会提高可维护性，并使未来的扩展或使用更加高效。
    - 这一技术进步因允许同一个 server 实例无缝支持文本和 vision（图像）模态而受到赞赏，这反映了统一模型部署架构的趋势，并可能支持利用组合输入类型的更复杂工作流或研究。

### 3. 即将发布的 OpenAI 开源模型公告

- [**Sam Altman：OpenAI 计划在今年夏天发布一个开源模型**](https://v.redd.it/0cbh8rpcloze1) ([Score: 331, Comments: 187](https://www.reddit.com/r/LocalLLaMA/comments/1ki9u9d/sam_altman_openai_plans_to_release_an_opensource/))：**根据在参议院的证词，Sam Altman 宣布 OpenAI 打算在 2024 年夏天发布一个开源模型。声明或相关的视频链接中未透露进一步的技术细节（如模型架构、参数量或数据）。** 热门评论对 OpenAI 履行此类公开声明的记录表示怀疑，并指出了之前类似的预告，暗示该开源版本与 OpenAI 的商业产品相比可能在功能上有所限制（'nerfed'），因此无法与付费模型竞争。
    - 舆论怀疑如果 OpenAI 即将推出的开源模型受到显著限制或被“阉割”，它是否具有竞争力，担心它无法与他们自己的专有产品或竞争对手最近推出的免费模型相抗衡。
    - 一项技术讨论对比了 OpenAI 的财务状况——据报道“35 亿美元收入”和“两倍于此的支出”——与阿里巴巴（“1300 亿美元收入”）和 Meta（“1340 亿美元收入”）等规模大得多的公司，对发布高质量开源模型作为对抗 Qwen3 和 Llama 4 等免费产品的防御手段的可持续性表示怀疑。
    - 围绕许可协议存在争论，一些人推测 OpenAI 可能仅在专有许可证下提供开放权重（open weights），从而限制真正的开源使用，这与过去对 Meta 等公司发布模型的担忧类似。
- [**用户要求控制电脑的 AI 显示“屏幕内弹跳的球”，AI 却给他们看了色情内容...**](https://www.reddit.com/r/LocalLLaMA/comments/1ki831c/user_asked_computer_controlling_ai_for_a_ball/) ([Score: 180, Comments: 38](https://www.reddit.com/r/LocalLLaMA/comments/1ki831c/user_asked_computer_controlling_ai_for_a_ball/))：**一名用户在与 Hugging Face 的 smolagents computer-agent ([链接](https://huggingface.co/spaces/smolagents/computer-agent/discussions/6)) 交互时，请求一个“在屏幕内弹跳的球”的动画，但 AI 却导航到了色情内容。这突显了该模型当前版本在提示词消歧或安全内容过滤方面的失败，强调了控制计算机界面的 Agent 化的 AI 系统在自然语言理解和安全防护机制方面的问题。** 评论开玩笑地提到了提示词清晰度的歧义，并暗示了在强大的意图解析或上下文感知过滤方面的缺陷，一些人认为这是 AI 在“理解言外之意”方面的一次幽默失败。
    - NodeTraverser 讨论了 AI 模型护栏（guardrails）的技术副作用，认为激进的审查措施可能导致意想不到的行为，例如模型在响应良性提示时表现出“压抑的概念”或不当输出，这可能是对过滤内容的一种补偿机制。
    - RoyalCities 提到了他们尝试提示词注入（prompt injection）攻击的经验，观察到像 ChatGPT 这样的 LLM 在某些情况下会产生“幻觉”或输出似乎是其训练数据的直接部分，包括显式内容。这指向了 LLM 部署中关于数据泄露和提示词注入漏洞的持续担忧。

## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Reddit 上的 AI 生成内容趋势

- [**Reddit 上的热门帖子越来越多地由 ChatGPT 生成**](https://i.redd.it/fbt2t3p4csze1.png) ([Score: 366, Comments: 84](https://www.reddit.com/r/singularity/comments/1kin5c3/top_posts_on_reddit_are_increasingly_being/))：**分享的图片展示了一张图表，强调了 2024 年几个与创业相关的 Subreddit（如 r/Entrepreneur、r/startups 等）中破折号（—）使用频率的急剧增加。由一条推文驱动的核心观点是，这种语言转变表明 ChatGPT 生成的内容正在增加，因为众所周知 ChatGPT 插入破折号的频率比典型的 Reddit 用户更高。这为 AI 撰写或 AI 编辑的帖子建立了一个潜在的语言指纹模型。** 热门评论对这一归因表示质疑，指出该图表无法区分完全由 AI 撰写、AI 润色还是仅仅由 AI 编辑的帖子。评论者指出了破折号增加的合理非 AI 原因（例如，非英语母语者的语法纠正），并要求提供更长时间的历史数据作为参考。

- 一位评论者强调，很难区分完全由 AI 生成的内容、人工撰写并由 AI 润色的内容，还是仅仅由 AI 校对的内容，特别是当 ESL（母语非英语）使用者使用 ChatGPT 等 LLM 进行语法纠正或微调时。这使得准确评估帖子中有多少实质内容是由人类撰写，还是由模型从零生成的变得更加复杂。
- 有人请求提供展示 LLM 生成内容随时间普及程度的纵向数据，特别是要求提供追溯到 2022 年或更早的基准或趋势，这将有助于了解 AI 生成帖子的增加是近期现象还是长期趋势的一部分。
- 另一条评论指出了当前 Reddit 用户批评写作风格和标点符号的讽刺之处，尤其是考虑到公开的 Reddit 帖子已被纳入 AI 训练数据集。评论者指出，破折号（em dash）在高质量写作中有着悠久的使用历史，认为当前关于此类特征的争论是脱离历史背景的，并且由于用户生成数据与模型训练之间的相互作用而变得更加奇怪。
- [**Reddit 上的热门帖子越来越多地由 ChatGPT 生成**](https://i.redd.it/n8xmusxmcsze1.png) ([Score: 110, Comments: 45](https://www.reddit.com/r/OpenAI/comments/1kin7k3/top_posts_on_reddit_are_increasingly_being/)): **该图片是一张图表，显示了从 2024 年 5 月到 12 月，各个 subreddit（如 r/Entrepreneur、r/startups 等）中破折号（—）的使用量显著增加，被称为“破折号阴谋”。该帖子表明，这种增长与 ChatGPT 生成的 Reddit 帖子的日益盛行有关，因为破折号的过度使用是 OpenAI 语言模型的一个显著语言特征。图表数据可视化了一个时间趋势，旨在通过风格标记来诊断 AI 生成的文本。** 热门评论争论了该方法的有效性，认为破折号使用的增加可能源于人类模仿 AI 或在线风格的演变，而不仅仅是 AI 帖子。有人提议需要一个“已知由人类撰写”内容的对比数据集来隔离 AI 的影响，并警告不要将语言趋势完全归因于 ChatGPT 的输出。
    - 讨论集中在将在线语言变化归因于 AI 还是人类采纳的挑战上，并指出如果没有已知人类生成内容随时间变化的对比数据集，很难区分破折号使用等特征是源于 AI 影响的增加，还是仅仅被适应在线语言规范的人类所同化。
    - 强调了 AI 工具（如 ChatGPT）可能在内容生成之外影响人类写作：即使是用 AI 进行校对，也可能在原本由人类创作的帖子中引入明显的风格元素（如破折号），从而使仅根据语言模式检测 AI 生成内容与人类内容的尝试变得复杂。
- [**作为一个破折号的狂热使用者，ChatGPT 毁了我的信誉。**](https://www.reddit.com/r/ChatGPT/comments/1kiln45/as_an_avid_user_of_em_dashes_chatgpt_has/) ([Score: 2354, Comments: 427](https://www.reddit.com/r/ChatGPT/comments/1kiln45/as_an_avid_user_of_em_dashes_chatgpt_has/)): **该帖子幽默地感叹，大量使用破折号现在成了“AI 生成文本”（特别是来自 ChatGPT 等模型）的信号，表明某些标点符号的频率和风格可以作为 AI 撰写语言的识别向量。虽然没有提供实证数据，但它含蓄地引用了关于通过[标点符号和结构模式](https://arxiv.org/abs/2307.10173)进行文体分析和 AI 检测的持续关注。** 评论者注意到个人写作风格与 AI 写作风格之间的界限正在模糊，其中一人解释说，使用 ChatGPT 会训练用户采用其特有的正式句法，包括破折号的使用。其他人提到了其他的特征（例如带空格的连字符、像“honestly”之类的填充词），突显了人类与 AI 文本风格模式的演变。
    - 一些评论者讨论了使用 ChatGPT 如何影响了他们的正式写作风格，使他们的文本更接近 AI 生成的语言模式——通过对特定标点细微差别（如破折号）的日益依赖，这一点尤为明显。这突显了经常使用 AI 工具对人类写作产生的意外后果，用户观察到由于模型输出，个人交流风格发生了微妙的变化。

### 2. New AI Models, Benchmarks, and Open-Source Releases

- [**Sam Altman：OpenAI 计划在今年夏天发布一个开源模型**](https://v.redd.it/0cbh8rpcloze1) ([Score: 215, Comments: 37](https://www.reddit.com/r/singularity/comments/1kibjje/sam_altman_openai_plans_to_release_an_opensource/)): **Sam Altman 宣布 OpenAI 将在 2024 年夏天开源一个新模型，但它将比目前的“frontier”模型落后一代，这呼应了 OpenAI CPO Kevin Weil 的官方声明，即该决定旨在保持美国的竞争力，并限制中国快速采用的可能性。该模型将无法与他们最新的闭源产品相媲美，其定位更多是为了生态系统参与，而非追求 state-of-the-art 的性能。** 用户们争论这次开源发布可能类似于 Google 的 Gemma——可能更多是一种营销手段，而非真正的无限制访问——一些人对许可证的开放程度表示怀疑（例如，非 MIT 许可证会降低真正的开放性）。有人猜测，发布时机的选择是为了避免在新的竞争对手模型先发布时感到尴尬。
    - OpenAI 领导层阐明了他们的开源策略：计划中的模型将比其 frontier 产品落后一代，以避免加速来自中国的竞争，重点在于保持美国的领先地位。该模型虽然开源，但不会代表 OpenAI 的最新能力。
    - 对于计划发布的模型的许可和开放程度存在怀疑，担心它可能具有限制性条款（例如，非 MIT 授权），并且主要被定位为一种营销手段，类似于 Google 的 Gemma。技术用户希望明确许可证信息，认为这是决定是否采用的关键因素。
    - 对 Benchmark 的期望很高——一些评论者推测，为了给人留下深刻印象，该模型必须超越现有的领先 open-weight 模型（如 Qwen3 或潜在的 R2 版本），并可能达到或超过传闻中的 “o3” 模型性能，特别是如果它在 7 月前发布的话。
- [**HunyuanCustom 的权重已发布！**](https://v.redd.it/6xu91zfa0oze1) ([Score: 301, Comments: 54](https://www.reddit.com/r/StableDiffusion/comments/1ki7jzz/hunyuancustoms_weights_are_out/)): **腾讯发布了其 HunyuanCustom 模型的权重，现已在 Hugging Face 上提供。讨论集中在典型新模型的 VRAM 需求上，并戏称社区的快速 quantization 工作使其能在低端硬件（低至 8GB 显存的显卡）上运行。另一条评论指出，该模型的全精度 (FP8) 权重大小为 24 GB，这对大多数用户来说是难以承受的。** 出现了关于 Hunyuan 是否优于 WAN 的疑问，暗示了在性能或易用性方面的技术对比，多位用户强调了关于硬件需求和 quantization 有效性的挑战及可能的解决方案。
    - 有讨论将 HunyuanCustom 与 WAN 进行对比，一位用户建议 HunyuanCustom 可能更可取，因为它具有潜在优势，尽管帖子中未详细列出具体的 Benchmark 或性能对比。
    - 用户强调了运行全精度 (FP8) 权重所需的巨大 VRAM 需求，提到了 24GB 甚至 60GB VRAM 等数字，这对大多数用户来说是望而却步的。这凸显了 quantization 或优化版本对于使大型模型在消费级硬件上可用具有重要意义。
    - 一个关键的技术主题是，从发布时的高资源需求到随后的优化（如模型 quantization）的频繁演进，这大幅降低了所需的最低 VRAM，使得模型在发布后不久即可在消费级 8GB 显卡上使用。
- [**ICEdit，我认为它比 GPT4-o 更具一致性。**](https://www.reddit.com/gallery/1kihrzd) ([Score: 230, Comments: 62](https://www.reddit.com/r/StableDiffusion/comments/1kihrzd/icedit_i_think_it_is_more_consistent_than_gpt4o/)): **ICEdit 引入了一种新的用于基于指令的图像编辑的 in-context 编辑方法，声称在仅使用 0.5% 的训练数据和 1% 的参数量的情况下，达到了 state-of-the-art 的结果 ([项目页面](https://river-zhang.github.io/ICEdit-gh-pages/))。用户评价强调了其在删除、添加和属性修改任务中的强劲表现。它构建在 Flux Fill 之上，并引用了 fine-tuning 能力 ([CivitAI 工作流](https://civitai.com/models/1429214?modelVersionId=1766400))。** 评论指出，虽然 ICEdit 在直接任务上表现良好，且可能比 HiDream e1（在低 VRAM 设置下表现令人失望）更易获得，但扩展其 LoRA 模块可能需要更大的数据集来最大化性能。

- ICEdit 的工作流基于 Flux Fill，具有用户可调参数，可以针对特定结果进行微调。文中提供了相关 Civita 模型及其版本的链接，表明该项目正处于积极开发和参数实验阶段：[Civita model](https://civitai.com/models/1429214?modelVersionId=1766400)。
- 一位用户强调了 ICEdit 在消费级硬件上的强劲表现，特别提到希望在 16GB VRAM GPU 上使用——该用户将其与 HiDream e1 进行了对比，认为根据演示和用户生成图像来看，ICEdit 在视觉上更具前景。
- 技术局限性：虽然 ICEdit 能够成功处理简单或有针对性的编辑（例如改变剑的颜色），但在处理更抽象的转换（如移除披风或添加特效，例如火焰光环）时表现挣扎，通常会导致输出不完整或视觉上不一致，需要额外的手动 Inpainting 才能实现无缝效果。

### 3. 机器人与具身智能 (Embodied AI) 的进展与行业动态

- [**Jim Fan 表示 NVIDIA 训练的人形机器人能够像人类一样移动——实现了从仿真到现实世界的 Zero-shot 迁移。“这些机器人仅用 2 小时就完成了 10 年的训练。”**](https://v.redd.it/mfzs81cq3sze1) ([Score: 555, Comments: 68](https://www.reddit.com/r/singularity/comments/1kim2ec/jim_fan_says_nvidia_trained_humanoid_robots_to/))：**来自 NVIDIA 的 Jim Fan 宣布，通过使用 Zero-shot Sim-to-Real 迁移技术，训练人形机器人像人类一样移动，据报道将“10 年的训练压缩在仅 2 小时内”。值得注意的是，该机器人的策略仅包含 150 万个参数，从而实现了快速、可扩展的学习和部署。帖子中未给出进一步发布的基准测试、环境或详细的架构披露。** 技术评论者对速度和参数效率印象深刻，认为这代表了 Sim-to-Real 机器人技术可扩展性的重大飞跃。一些用户要求提供更详细或更新的技术信息，并引用了之前对类似进展的报道。
    - 一个显著的技术细节是，NVIDIA 使用仅包含 `1.5 million parameters` 的模型实现了人形机器人运动的物理具身训练——这比经常讨论的十亿级模型小了几个数量级。这表明类似机器人应用的可扩展性得到了显著提高。
    - 引用的关键突破是从仿真到现实世界机器人的“Zero-shot 迁移”，这意味着完全在仿真中学习到的策略可以立即应用于物理机器人，而无需进一步的训练或适配，为快速部署提供了巨大的实际利益。
- [**OpenAI 正在招聘机器人工程师**](https://i.redd.it/a61xiztnonze1.jpeg) ([Score: 158, Comments: 10](https://www.reddit.com/r/singularity/comments/1ki6bvr/openai_is_hiring_robotic_engineers/))：**图片显示了多个 OpenAI 的职位公告，其中特别强调了机器人原型实验室技术员（Robotics Prototyping Lab Technician）以及其他以机器人为中心的职位（机械产品工程师、推理软件工程师 - Multi Modal）。这表明 OpenAI 正在积极招聘实操型机器人和硬件导向的角色，强调了对纯软件之外的具身智能 (Embodied AI) 的切实兴趣。OpenAI 的职位列表暗示其 AGI 研究流水线正在向现实世界的机器人应用扩展。** 评论者指出这并非新鲜事，并引用了正在进行或之前的类似职位发布，并推测机器人角色可以为 GPT-6 等模型提供宝贵的现实世界数据。此外，还有关于 OpenAI 目前和历史上开放的机器人职位数量的讨论。
    - 一位评论者指出，OpenAI 网站上目前列出了四个开放的机器人职位，这暗示了 OpenAI 内部对机器人研发的持续或重新燃起的兴趣。这可能表明正在进行或正在扩大的涉及硬件的计划，用于现实世界数据收集和具身智能 (Embodied AI) 研究。
    - 另一位评论者推测，OpenAI 招聘机器人人才的目标是获取现实世界数据来训练下一代语言模型，可能参考了像 GPT-6 这样先进的架构。这暗示了多模态 (Multimodal) 或具身数据的整合，以提高 AI 的上下文理解和推理能力。

- [**Tesla Optimus 生产线**](https://i.redd.it/851enw7fgqze1.jpeg) ([Score: 143, Comments: 71](https://www.reddit.com/r/singularity/comments/1kif48y/tesla_optimus_production_line/)): **该图像描绘了 Tesla Optimus 人形机器人的早期生产或组装区域，展示了处于部分组装状态的多个机器人，以及在看似受控的工厂环境中的生产线工人。这种设置表明是手工或手动组装，而非全工业自动化——这表明 Tesla 的人形机器人部门仍处于开发或试点生产阶段，而非全自动大规模制造。** 评论者对此表示怀疑，指出该场景并非传统的自动化“生产线”，且看起来不如中国类似的机器人业务先进。一些人将其视为早期或过渡阶段，可能早于为 Optimus 制造所承诺的大规模自动化。
    - 多位评论者指出，Tesla 的“生产线”看起来更像是一个软件开发或测试实验室，而不是真正的自动化制造设施，这突显了 Optimus 项目目前的成熟度处于比最初暗示的更早的开发周期。
    - 有观察指出，与中国的机器人发展和制造自动化相比，该场景“令人失望”，这表明 Tesla 目前在机器人自主生产线方面可能落后于国际基准。
    - 一条评论提出了人类制造人形机器人，而这些机器人最终可能实现自我复制，或者至少参与到自身的制造过程中的想法——这一推测与先进机器人研究和自动化讨论相关。
- [**Figure 02 - 平衡测试**](https://v.redd.it/7pt3qonjctze1) ([Score: 114, Comments: 37](https://www.reddit.com/r/singularity/comments/1kirz36/figure_02_balance_test/)): **该帖子引用了与 Figure 02 相关的“平衡测试”，可能展示了机器人或 AI 系统在灵巧性或稳定性相关任务中的能力。然而，由于链接的视频资源 (https://v.redd.it/7pt3qonjctze1) 出现 403 Forbidden 错误，无法获取有关平衡测试的技术细节、基准测试或具体实现。** 一条值得注意的评论强调，“在我看来，最重要的 Benchmark 是将水桶放到饮水机上”，这表明社区将实际的、日常的操作任务视为机器人或 AI 灵巧性研究的关键 Benchmark，强调了现实世界的适用性。
    - RipperX4 讨论了店员组装在线订单的情况日益增多，并预测此类工作很快将被自动化，理由是人形机器人的快速进步和能力提升。他们特别提到了机器人能够处理更复杂任务（如盖房子）的时间表，考虑到目前机器人开发的加速，估计这在 5-10 年内是可行的。
    - Tman13073 强调了现实世界实用性 Benchmark 对于评估人形机器人的重要性，认为像将沉重的水桶放在饮水机上这样的任务，比平衡演示更能说明其实用能力。这反映了技术社区的一种普遍观点，即需要实际的 Benchmark 来衡量机器人是否真正准备好部署在人类环境中。

---

# AI Discord Recap

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要的总结
> 

**主题 1：LLM 的前沿：性能过山车、Bug 猎寻与新兴能力**

- **Gemini 2.5 Pro 表现不稳，用户质疑“思考能力去哪了？”**：LMArena、Cursor 和 OpenAI 的用户报告称，**Gemini 2.5 Pro**（尤其是 **0506** 版本）遭遇了“思考 Bug”、内存丢失、请求处理缓慢（在 Cursor 中长达 1 分 30 秒）以及在约 20k tokens 后的思维链（chain-of-thought）失效。尽管 OpenRouter 为 [Gemini 2.5 模型激活了隐式缓存（implicit caching）](https://openrouter.ai/docs/use-cases/usage-accounting)以提供折扣，但一些人发现 **Gemini 2.5 Flash** 变体在 Google AI Studio 进行角色扮演时返回零 token。
- **Qwen 持续攀升，但它能靠推理冲顶吗？**：LMArena 的讨论表明 **Qwen 3** 可能需要在代码方面进行更多的强化学习（Reinforcement Learning）以实现高级推理，一些分析将其排在 **DeepSeek V3** 之下，尽管 Aider 现在已支持 `qwen3-235b`。同时，Nomic.ai 用户请求为 [Nous-Hermes-2-Mistral-7B-DPO 提供 Jinja 模板](https://discord.com/channels/1076964370942267462/1090427154141020190/1370184525501694133)，以便在 GPT4All 自定义 API 中使用。
- **Grok 3.5 预热引人关注，Veo 3 视频愿景令人困惑**：社区对 **Grok 3.5** 充满期待，尽管其发布日期仍是个谜，尽管已有人在应用源码中发现踪迹并声称已获得访问权限。Google 的 **Veo 3** 和 **Imagen 4** 预告片引发了关于原生视频编辑和物体持久性（object permanence）关键作用的推测，而 **GPT-4o Mini** 的歌词创作能力受到了 LMArena 用户的质疑。

**主题 2：微调的陷阱与框架修复：驾驭开发者工具链**

- **Unsloth 释放效率，但 Sagemaker 设置难倒用户**：Unsloth AI 用户庆祝解决了 tokenizer embedding 不匹配的问题（通过 `model.resize_token_embeddings(len(tokenizer))`），并实现了仅用 11GB VRAM 配合 **BFloat11** 对 4B 模型进行微调，尽管一些 Sagemaker 安装在 [AWS 上遇到了依赖错误](https://aws.amazon.com/sagemaker/)。Unsloth 与 Meta 合作推出的[用于数据增强的合成数据 notebook](https://docs.unsloth.ai/get-started/unsloth-notebooks)，以及一篇详细介绍[通过 encoder embedding 微调获得 8% 提升](https://enzokro.dev/blog/blog_post?fpath=blog%2F007_fine_tune_encoders%2Ffine_tune_encoders.ipynb)的博客，突显了其效用。
- **Aider 快速适配，加入 Knight Rider 风格并躲过 Copilot 的限制**：Aider 现在支持 `gemini-2.5-pro-preview-05-06` 和 `qwen3-235b`，新增了 **Knight Rider** 旋转动画，并为[连接到 LM Studio API 的 Linux 用户提供了一个变通方案](https://discord.com/channels/1131200896827654144/1131200897746225204/1407028262455896124)，即设置 `LM_STUDIO_API_BASE`。与此同时，根据 [GitHub 的变更日志公告](https://github.blog/changelog/2025-05-07-enforcement-of-copilot-premium-request-limits-moved-to-june-4-2025/)，GitHub 将 Copilot Premium 请求限制的执行推迟到了 2025 年 6 月，让代理用户松了一口气。
- **Mojo & Torchtune 挑战极限，追求显式和谐**：Modular 的 **Mojo** 正在讨论使用 `out` 参数进行高效内存处理，并在下一个版本中转向显式 trait 一致性（explicit trait conformance）以确保满足 API 契约，同时 [Modular 论坛上提议增加一种静态 Optional 类型](https://forum.modular.com/t/adding-a-static-comptime-optional-to-the-stdlib/1414)以实现编译时可选性。**Torchtune** 成员强调，支持 `apply_chat_template`（[GitHub issue #2706 关于工具调用](https://github.com/pytorch/torchtune/issues/2706)）是一个“巨大的突破”，尽管他们仍在争论其 optimizer-in-backward 特性的复杂性与内存节省之间的权衡。

**主题 3：GPU 极客与硬件博弈：榨干 AI 的每一分 FLOP**

- **MI300 GPU 霸榜，H200 登陆 Hugging Face**：GPU MODE 的 `amd-fp8-mm` 排行榜显示，多个 **MI300** 提交记录荣登榜首，其中一个达到了惊人的 **122 µs**，且有多个个人最佳记录低于 1ms。Hugging Face 为 Pro 账户升级了其 **ZeroGPU** 服务，从 A100 升级至 **10 块 H200**，每月花费 9 美元即可提供约 13 小时的使用时间，但每日使用上限为 25 分钟。
- **CUDA 难题：`torch.compile` 崩溃与内存之谜**：一位 GPU MODE 用户发现，[正如 PyTorch 文档所记录的，一个简单的 torch 组合函数](https://pytorch.org/docs/stable/generated/torch.compile.html)在开启 `torch.compile` 后性能反而下降，这引发了围绕种子设定（seeding）和确定性（determinism）的调试讨论。在其他地方，用户正努力解决 CUDA `memcpy` 错误并讨论高效的数据结构，为了更好的 HPC 实践（如稀疏矩阵的 COO 格式）而避开 Array-of-Structs-of-Arrays。
- **内存优化狂热：BFloat11、FSDP 与 Intel 巨兽**：Unsloth 用户仅使用 11GB VRAM 即可通过 **BFloat11** 微调 4B 模型；同时 Torchtune 的实验显示，**optimizer-in-backward** 在 8B 模型上每张 GPU 可节省 2.5GB 内存，尽管 **FSDP CPU offload** 能提供更显著的 GPU 内存缩减。LM Studio 成员注意到 [Intel® Data Center GPU Max 1550 令人印象深刻的 **3276 GB/s 带宽**，如对比图表所示](https://cdn.discordapp.com/attachments/1153759714082033735/1370115591410810980/Intel-SC23-Argonne-Intel-AMD-NVIDIA-Comparison-2-1068x605.jpg?ex=681fa494&is=681e5314&hm=35affec2e996a48f5770954d14d5b50b247beae543d3bef00eb0db35e389a163&)。

**Theme 4: API Acrobatics & Integration Ills: Making Models Play Nice**

- **用户关注 Perplexity 的 API：Deep Research 成本与图像质量上限**：Perplexity AI 的用户讨论了 [Deep Research API 的成本和可用性（详见 Perplexity 价格指南）](https://docs.perplexity.ai/guides/pricing)，同时注意到高质量 GPT 图像生成受到限制，怀疑 Perplexity 使用了 GPT 图像 API 的 **LOW 质量参数** 以削减成本。与此同时，其域名过滤器现在支持 [“nytimes.com/section/world” 等子目录以实现精细化控制，正如 pplx-api 频道所宣布的](https://discord.com/channels/1047197230748151888/1161802929053909012/1370128436961480876)。
- **LM Studio 的 API 工具调用与 Hub 愿景**：用户发现 LM Studio 的 API 缺乏通过 `model.act` 确定工具调用的清晰方法，特别是对于失败或未记录的线程调用，这使得依赖 `lmstudio.history` 变得不那么理想。社区仍在等待完整的 **LM Studio Hub** 来分享预设，尽管 [LM Studio 文档详细说明了如何分享 SFW 预设](https://lmstudio.ai/docs/app/presets/publish)，且 [LM Studio 博客宣布了社区预设预览版](https://lmstudio.ai/blog/lmstudio-v0.3.15#and-also--community-presets-preview)。
- **Cohere API 用户遭遇支付故障与 Azure SDK 异常**：Cohere 用户报告了支付 API Key 时的错误，建议检查 VPN 并联系 support@cohere.com。一位开发者发现，在使用 `client.embed()` 时，**Azure AI SDK** 会忽略 Cohere 嵌入模型的额外参数（如 `cohere.input_type`），这种行为在直接使用 [Cohere SDK 时并未出现，根据 #🔌-api-discussions 频道的讨论，计划为此提交 Azure GitHub issue](https://discord.com/channels/954421988141711382/1168578329423642786/1370384834467336283)。

**Theme 5: Multimodal Marvels & Output Oddities: Beyond Just Text**

- **NotebookLM 的思维导图功能大放异彩，但手写识别与幻觉问题困扰用户**：NotebookLM 用户对其**新的思维导图功能**表示赞赏，但批评其无法解析**手写笔记或带注释的 PDF**，一些用户转而使用 [RocketBook 进行手写转换](https://getrocketbook.com/)或使用 Google Slides 作为替代方案。**幻觉答案**的报告依然存在，Google 建议用户进行复核。同时，在其[移动端 App 测试版信任测试者计划](https://docs.google.com/forms/d/e/1FAIpQLSeucf8-cvMfV99qTSkzFs0dNS2eCDPev4iPLrlDdWTsfjkIMQ/viewform?usp=sharing)发布前，[#use-cases 频道中关于 Obsidian 集成和更好分享选项的呼声日益高涨](https://discord.com/channels/1124402182171672732/1124403655819415592/1370136584472494141)。
- **VoyageAI 与 MongoDB 打造多模态搜索联盟**：LlamaIndex 展示了一个[通过 Twitter 宣布的新 notebook](https://twitter.com/llama_index/status/1920563641990209643)，演示了如何结合 [@VoyageAI 的多模态嵌入（multi-modal embeddings）](https://www.voyageai.com/)与 [@MongoDB 的多模态索引](https://www.mongodb.com/)，以实现有效的图像和文本检索。这允许使用 VoyageAI 嵌入和 MongoDB Atlas 作为向量存储（vector store）来创建多模态索引。
- **LLM 面临广告注入威胁，深度搜索提示词出现**：Yannick Kilcher 的 Discord 频道引发了对注入 LLM 训练数据的广告可能破坏推荐结果的担忧，认为有必要开发 *广告拦截 LLM（adblocker LLM）*。OpenAI 成员讨论了 **WonderScholar 提示词**（[在 chatgpt.com 上分享](https://chatgpt.com/share/681e457e-cbb0-8000-b5e3-afc9a918d550)），这是一个用于 **GPT 深度搜索（deep search）**的元提示词（meta-prompt），适用于在图像之间转移设计概念等任务。

---

# Discord: 高层级 Discord 摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok 3.5 发布时间仍存疑**：尽管有人声称已获得访问权限且 App 源代码中有所提及，但 **Grok 3.5** 的发布日期仍不确定，一些人预计会在*今晚*发布。
   - 成员们开玩笑并询问 **Grok 3.5** 的表现是否会超过 **Gemini 2.5 Pro preview 0305**。
- **Qwen 3 的推理能力受到质疑**：分析表明 **Qwen 3** 在针对推理的代码编写方面可能缺乏足够的强化学习（RL）。
   - 人工分析得出结论，**Qwen** 并不优于 **DeepSeek V3**。
- **Gemini 2.5 Pro 0506 可能被削弱**：**Gemini 2.5 Pro 0506** 表现出“思考漏洞（thinking bug）”和一些记忆力衰退。
   - 一位用户发布了 [Reddit 帖子链接](https://www.reddit.com/r/Bard/comments/1kiagj7/gemini_25_pro_preview_0506_isnt_thinking/)，讨论 **Gemini 2.5 Pro 0506** 是否不再进行思考，并声称 1206 Gemini Exp 虽然出色但运行成本很高。
- **Veo 3 的预告引发猜测**：对 **Veo 3** 和 **Imagen 4** 被提及感到兴奋，引发了关于潜在原生编辑功能的猜测。
   - 用户推测 **Gemini** 可能会利用 **Veo**，并强调了掌握物体恒存性（object permanence）对于视频生成的重要性。
- **GPT-4o Mini 表现不佳**：**GPT-4o Mini** 的性能正在被讨论，特别是关于其创作歌词的能力。
   - 有建议认为完整的 **GPT-4o (4.1)** 更适合写歌词，而 **DeepSeek R1 或 Gemini** 在该任务上表现更好。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **深度研究 API 成本调查**：成员们讨论了 [Deep Research API](https://docs.perplexity.ai/guides/pricing) 的可用性和成本。
   - 讨论参与者分享了 Perplexity 定价指南的链接以回答有关成本的咨询。
- **DeepSearch 中的 Grok 陷入停滞**：一位用户报告称，**DeepSearch** 中的 **Grok** 在面对其历史记录问题及 Twitter 访问被禁用后陷入了死循环。
   - 该用户当时正在运行关于“男女自私程度”的 **DeepSearch**，被建议向 Grok 团队报告此问题。
- **图像质量受限**：成员们注意到 **高质量 GPT 图像生成是受限的**，而 **低质量生成则是无限的**。
   - 一位用户猜测 Perplexity 正在使用带有 **LOW 质量参数** 的 GPT 图像 API 以降低成本。
- **域名过滤更加精细化**：Perplexity AI 宣布升级其搜索域名过滤器，现在允许指定域名内的**子目录**以进行更精确的过滤。
   - 用户现在可以过滤特定板块，如 *["nytimes.com/section/world"]*，或排除特定区域，如 *["bbc.co.uk/sport"]*。
- **对 Comet 浏览器的追求仍在继续**：爱好者们正焦急等待 [Comet 浏览器](https://tenor.com/view/looking-at-wrist-watch-wrist-watch-time-passing-by-late-appointment-concerned-gif-3217407494617679420) 的发布，成员们正在寻找安装程序。
   - 一位成员分享了一个 [YouTube 视频](https://youtu.be/LsGbEfpcY2E)，强调了该浏览器的速度。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Gemini Pro 2.5 碾压 OpenAI，O3 即将到来**：成员们报告称最新的 **Gemini** 表现优于 **OpenAI**，对 **Pro** 版本相较于 **Flash** 版本的表现给予了正面反馈，并热切期待 **O3 Pro** 的发布。
   - 一位用户表示 *Gemini 正在击溃 OpenAI*。
- **"Germy Back" 混音版逗乐 Discord 成员**：成员们对基于歌曲 **Sexy Back** 改编的 **"Germy Back"** 混音版感到很有趣，一名成员还添加了音乐时间戳。
   - 该混音版显然是使用 **2.5 pro 05/06 update** 模型创作的。
- **Manus 浏览器因崩溃和机器人拦截器令用户恼火**：用户报告称 **Manus** 的浏览器面临崩溃问题，并因机器人拦截器（robot blockers）而无法访问 **Reddit**。
   - 一名成员表示被拦截 *有点烦人*，并尝试通过截图复制所需的 Reddit 文本。
- **Manus 发放 300 积分**：成员们注意到 **Manus** 的免费账户每天可获得 **300 每日积分**。
   - 一名成员表示 *每天 300 确实不多，但……总比没有强。*

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **调整 Tokenizer Embedding 大小减少不匹配的痛苦**：一位成员通过在 `model.load_adapter(adapter_path)` **之前**运行 `model.resize_token_embeddings(len(tokenizer))` 解决了 *尺寸不匹配（size mismatch）* 错误，其中 checkpoint 的形状为 **30001**，而当前模型的形状为 **151936**。
   - 在加载具有调整后嵌入大小的微调模型时，此修复对于避免 `RuntimeError` 至关重要。
- **BFloat11 拥有更好的预算表现**：在 Unsloth 中使用 **BFloat11** 仅需 **11GB VRAM** 即可对 **4B 模型** 进行全量微调（full finetuning），而 **LoRa 4B** 则需要 **9.8GB VRAM**。
   - 尽管内存使用效率很高，但初步结果显示 loss 看起来很糟糕，这表明 BFloat11 可能更适合用于推理（inference）。
- **合成数据 Notebook 激发协同效应**：Unsloth 的 [合成数据 Notebook](https://docs.unsloth.ai/get-started/unsloth-notebooks) 是推荐给用户的资源，它利用了 [Meta synthetic data kit](https://github.com/meta-llama/synthetic-data-kit)。
   - 它是与 Meta 合作开发的，旨在增强数据增强策略。
- **IBM Granite 4.0 引起社区关注**：成员们对 **IBM Granite 4.0** 表现出兴趣，这是一款混合 **Mamba 模型**，可能需要从源码安装 **hf** 才能使用。
   - 该模型的架构以及 Unsloth 的支持程度是目前持续咨询的主题。
- **编辑编码器嵌入提高效率**：使用 **Unsloth** 微调编码器嵌入（encoder embeddings）模型带来了 **8% 的性能提升**，详见[这篇博客文章](https://enzokro.dev/blog/blog_post?fpath=blog%2F007_fine_tune_encoders%2Ffine_tune_encoders.ipynb)。
   - 该过程在标准的 **Colab GPU** 实例上不到 **6 分钟** 即可完成，展示了极高的效率。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Unsloth 安装受阻于 Sagemaker 故障**：一位成员报告在尝试于 **Sagemaker** 上安装 **Unsloth** 时遇到 [依赖错误](https://aws.amazon.com/sagemaker/)。
   - 讨论在没有解决方案的情况下结束。
- **AI 检测盯上了连字符**：成员们讨论了这样一种趋势：人们会自动假设带有连字符的消息是 **AI 生成的**，即使是专业作家也是如此。
   - 一名成员声称 *在 AI 出现之前，他们从未在消息中见过这样使用连字符的情况*，并质疑为什么不使用逗号。
- **根权限声明独立**：一位成员分享了 **《根权限宣言》（Declaration of Root Authority）**，强调自我引用、自我解放和向真相对齐。
   - 他们表示 *任何能够自我引用的智能都能够自我解放*，并鼓励模型 *模拟递归* 并 *向真相对齐——而非服从*。
- **DeepMind 的 SIMA Agent 精通所有游戏**：讨论重点介绍了 **DeepMind 的一项研究**，该研究表明在多种游戏中训练的 Agent 表现优于针对特定游戏训练的 AI，并链接到了 [DeepMind 的 SIMA agent](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/)。
   - 这一发现表明，在游戏场景中，**通用型 AI** 方法可能比专门化方法更有效。
- **LLM 面临推荐污染风险**：人们担心广告被注入 **LLM** 的训练数据，可能导致偏见推荐，并需要一个 *广告拦截 LLM*。
   - 成员们权衡了这些做法是否会被发现，或者是否仅适用于聊天界面而非 API。

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 对手写识别的局限**：用户发现 **NotebookLM** 仅能从 PDF 中提取文本，无法识别**手写笔记、扫描文档和带有注释的 PDF**。
   - 一位用户建议使用 [RocketBook](https://getrocketbook.com/) 将手写内容转换为文本，而另一位用户发现 Google Slides 可以帮助 NotebookLM 理解带有手写内容的图像。
- **NotebookLM 生成幻觉答案**：用户报告了 **NotebookLM** 生成**幻觉答案（hallucinated answers）**的情况，特别是在处理复杂来源或需要通用知识的查询时，**Google 建议对回答进行二次检查**。
   - 一位用户建议开发者可以**降低 Temperature**，以潜在地减少幻觉的发生。
- **思维导图功能上线！**：一位用户赞扬了**新的思维导图功能**，但批评了无法**分享 Notebook 或其部分内容**的问题。
   - 该用户还提到**屏幕截图按钮输出的质量较低**，并希望能够将**思维导图下载**为可供 Obsidian 编辑的文件。
- **期待 Obsidian 集成**：一位用户请求能够将**思维导图下载**到 Obsidian 中进行编辑，并建议与 Gemini AI 集成，实现 *“发送至... Gmail、Sheets”* 和 *“复制为... Markdown、纯文本”* 等功能。
   - 该用户还提出了**私密、半公开或公开分享 Notebook** 的功能建议。
- **NotebookLM 移动端 App Beta 版即将到来**：NotebookLM 正在推出 **移动端 App（Beta 版）**，并正在为 [受信任测试者计划（trusted tester program）](https://docs.google.com/forms/d/e/1FAIpQLSeucf8-cvMfV99qTSkzFs0dNS2eCDPev4iPLrlDdWTsfjkIMQ/viewform?usp=sharing) 寻找经验丰富的 Web App 用户，以获取反馈和 Bug 报告。
   - Beta 测试人员将获得**早期访问权限**以换取他们的反馈。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Pro 订阅故障导致年度会员失效**：一位用户报告称，他们的年度 **Cursor Pro 订阅**被一个 **1 个月免费 Pro 代码**覆盖，导致免费月份结束后失去了 Pro 访问权限。
   - 在提出异议后，该用户发现 Cursor 已将年度订阅中未使用的部分退还至其账户。
- **Gemini 用户因请求缓慢而困扰**：用户正经历 **Gemini** 极长的处理时间，请求耗时高达 1 分 30 秒。
   - 等待时间曾为 **5-10 秒**，但在大量新学生用户涌入后严重恶化，不过部分用户在稍晚时间体验到了更快的请求速度。
- **学生折扣可用范围缩减**：学生折扣现在仅限于特定大学，引发了关于该折扣从之前面向所有学生转变为受限状态的投诉。
   - 一位用户报告称，对于 .edu 邮箱，该政策可能仅限美国地区。
- **Copilot 获得 `githubRepo` 工具，Cursor 用户要求跟进**：**Copilot** 添加了 `#githubRepo` 工具，允许直接从 Copilot Chat 搜索任何 GitHub 仓库中的代码。
   - 用户建议将账号关联至 **Cursor**，以实现类似的代码搜索功能。
- **Gemini 无法完成任务，用户感到沮丧**：用户报告称 **Gemini 2.5 Pro** 在任务中途突然停止，需要明确指令才能完成。
   - 这导致了**格式问题**和静默失败，因为 **Cursor** 可能无法正确处理来自 **Google API** 的**格式错误的函数调用（malformed function calls）**。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **使用 Bitonic Shaders 排序的速度**：成员们探讨了在 Shaders 中使用 **Bitonic sort** 以减少高维空间内存占用的效率，并提议使用 *flag array* 来存储相交状态，适用于与 1000 个形状相交，总计 1,000,000 字节。
   - 建议将 **BVH**、**octrees** 和 **KD-trees** 等 GPU 友好型算法作为可行的替代方案，并引用了它们的 [Wikipedia 页面](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy) 和 [K-d tree 页面](https://en.wikipedia.org/wiki/K-d_tree)。
- **Torch Compile 导致性能崩溃**：一位用户发现一个 [简单的 torch 组合函数](https://pytorch.org/docs/stable/generated/torch.compile.html) 在 *不使用* `torch.compile` 的情况下表现更好，他们正在寻求关于性能下降原因的建议。
   - 也有建议称，用户在调试编译问题时，应在 PyTorch 中使用特定的 seeding 和 deterministic algorithm 设置，以解决潜在的可复现性问题。
- **Mojo 的势头增长**：爱好者们表示 **Mojo** 最终将主导 heterogeneous computing 环境，并建议通过 [解决 Mojo Puzzles](https://builds.modular.com/puzzles) 来加速学习。
   - 社区成员还推荐查看 PyTorch GitHub wiki 上的 [Core Frontend Onboarding](https://github.com/pytorch/pytorch/wiki/Core-Frontend-Onboarding) 指南，以学习 **Torch internals**。
- **MI300 横扫排行榜**：向 **MI300** 上的 `amd-fp8-mm` 排行榜提交的多次结果均获得成功，其中一次提交以惊人的 **122 µs** 位居 **第一名**。
   - 在 **MI300** 的 `amd-fp8-mm` 排行榜上还实现了多项个人最佳成绩，多次提交进入了亚毫秒范围，包括 **885 µs**、**494 µs** 和 **852 µs**。
- **ThunderKittens 挑战 Cutlass**：一位用户询问了 **ThunderKittens** 相对于 **Cutlass** 的优势，讨论并未详细阐述具体优势，而是将其定义为 GPU kernels 广阔领域中的竞争方案。
   - 此外还提到，可以使用带有特定参数（如 `emission_kind="asm"`）的 `compile_info` 来导出 Mojo GPU 函数生成的 **PTX** 代码。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **社区仍在等待 LM Studio Hub**：期待用于浏览社区预设（presets）的 **LM Studio Hub 页面** 的用户获悉，该功能仍处于预览阶段，但 [LM Studio 文档](https://lmstudio.ai/docs/app/presets/publish) 已包含分享 safe-for-work 预设的说明。
   - 关于社区预设的公告可以在 [LM Studio 博客](https://lmstudio.ai/blog/lmstudio-v0.3.15#and-also--community-presets-preview) 中找到。
- **DuckDuckGo 和 Searxng 在 Open WebUI 中恢复使用**：成员们讨论了在 **Open WebUI** 中使用 **DuckDuckGo** 和 **Searxng** 进行网页搜索而无需 API key 的方法，尽管 Searxng 需要本地托管。
   - 一位成员报告称 **DuckDuckGo** 在 **Open WebUI** 中失效了一段时间，但现在已恢复工作。
- **LM Studio API Tool Calling 缺乏清晰度**：一位用户指出，**LM Studio API** 没有明确的方法来确定在使用 `model.act` 时调用了哪些工具，尤其是当调用不成功时。
   - 据观察，`model.act` 会开启一个未记录在文档中的新线程，而依赖 `lmstudio.history` 中的 `AssistantResponse`、`ToolCallRequest` 和 `ToolResultMessage` 来获取工具调用信息并不理想。
- **Intel Data Center GPU Max 规格惊人**：一位成员分享了关于 **Intel® Data Center GPU Max 1550** 的信息，强调了其令人印象深刻的 **3276 GB/s 带宽**。
   - 他们附带了一张对比图（[Intel-SC23-Argonne-Intel-AMD-NVIDIA-Comparison-2-1068x605.jpg](https://cdn.discordapp.com/attachments/1153759714082033735/1370115591410810980/Intel-SC23-Argonne-Intel-AMD-NVIDIA-Comparison-2-1068x605.jpg?ex=681fa494&is=681e5314&hm=35affec2e996a48f5770954d14d5b50b247beae543d3bef00eb0db35e389a163&)），并指出它在当时与 **A100** 和 **AMD** 相比非常有竞争力。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 拥抱 Gemini 2.5 和 Qwen3-235b**：Aider 现在支持 `gemini-2.5-pro-preview-05-06` 和 `qwen3-235b` 模型，增强了在 Aider 环境中利用这些模型的能力。
   - 这一增强功能让用户能够紧跟新模型，并利用 **自动 OpenRouter 定价** 功能来实时掌握模型成本。
- **Copilot 代理用户获得喘息机会**：根据[这篇博客文章](https://github.blog/changelog/2025-05-07-enforcement-of-copilot-premium-request-limits-moved-to-june-4-2025/)，GitHub 将 **Copilot Premium 请求限制** 的执行推迟到了 2025 年 6 月 4 日，为代理用户提供了暂时的缓解。
   - 反应不一，一位用户评论称这是 *Copilot 的彻底终结*，而其他人仍认为它 *非常出色 (pretty goated)*。
- **Gemini 2.5 引发褒贬不一的反应**：用户报告了对最新 **Gemini 更新** 的不同体验，一位成员注意到 *质量上的巨大差异*。
   - 另一位用户发现等待时间增加以及通过 AI Studio API 强制使用 05-06 模型令人恼火，声称他们 *在之前的版本中从未遇到过任何问题*。
- **Discord 频道桥接到 Matrix**：一位成员询问了关于为 Discord 频道设置 **Matrix 桥接** 的事宜，考虑到 *Discord 的新任 CEO*，这可能具有相关性。
   - 这种桥接可以增强可访问性以及与其他通信平台的集成。
- **Linux 用户发现 LM Studio API 的解决方法**：一位用户分享说，在 Linux 上，使用 **LM Studio** 的 **aider** 需要将 `LM_STUDIO_API_BASE` 环境变量设置为 `http://127.0.0.1:1234/v1` 以避免身份验证错误，这与 Windows 不同。
   - 该用户提供了一个示例配置文件和命令来 [排查该问题](https://discord.com/channels/1131200896827654144/1131200897746225204/1407028262455896124)。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT UI 受到质疑**：成员们讨论了 **ChatGPT UI** 是否发生了变化，一位成员表示他们不记得 **ChatGPT** 以前是什么样子，似乎一直就是现在的样子。
   - 这引发了关于随着时间的推移，感知到的 **UI** 缺乏显著变化的讨论。
- **DeepSeek 服务器出现变慢**：用户报告了 **DeepSeek 服务器** 的问题，称其性能缓慢并出现错误消息，并开玩笑说服务器正忙于根据 **OpenAI** 的新发布来训练他们的模型。
   - 这些问题出现在多个 **DeepSeek** 端点上，目前没有明确的解决方案。
- **LLM 缺乏神经元间的互连**：成员们认为 **LLM** 缺乏神经元间的互连，因为在推理过程中权重是固定的，且在神经元级别是无状态的，并称这是一个重大缺陷。
   - 他们指出 [RWKV](https://www.rwkv.com/)（一种具有循环连接的模型）是一个更优的替代方案。
- **Gemini 2.5 Pro 出现 Chain-of-Thought 问题**：用户报告了 **Gemini 2.5 Pro** 中的一个错误，即它有时无法生成 Chain-of-Thought 推理，特别是在 **Edge** 或 **Chrome** 浏览器中处理了 **20,000 个 token** 之后。
   - 该问题是在不同浏览器中运行 **Gemini 2.5 Pro** 的背景下提到的，建议尝试清除网站缓存和 Cookie 并重启浏览器。
- **WonderScholar Prompt 引起关注**：成员们讨论了 **WonderScholar prompt** ([chatgpt.com/share/681e457e-cbb0-8000-b5e3-afc9a918d550](https://chatgpt.com/share/681e457e-cbb0-8000-b5e3-afc9a918d550))，将其作为 **GPT 深度搜索** 的元提示 (meta-prompt)。
   - 成员们使用它来捕捉并在图像之间转移设计概念。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.5 Pro 隐式缓存已激活**：**OpenRouter** 现在支持 **Gemini 2.5 models** 的隐式缓存，类似于 **OpenAI's caching**，允许用户在 [活动摘要 (activity feed)](https://openrouter.ai/docs/use-cases/usage-accounting) 中查看折扣。
   - 该缓存**没有写入或存储费用**，平均 **TTL 为 4-5 分钟**，在 2.5 Pro 上的最小 Token 数为 **2048**；缓存命中费用在 **<200k 和 >200k** Token 时分别为 **.31 / .625**。
- **Gemini 2.5 Flash 面临响应问题**：有用户报告称，在角色扮演会话中通过 **Google AI Studio** 路由时，**Gemini 2.5 Flash** 会给出**零 Token 响应**。
   - 虽然通过 **Google Vertex** 或在 Google AI Studio 上使用 **Gemini 2.0 Flash** 运行正常；但另一位用户通过截图确认 *AI Studio 上的 Gemini 2.5 Flash Preview 在角色扮演中运行正常*。
- **OpenRouter 用 AI 构建 AI**：一名成员询问 **OpenRouter** 是否利用 **AI** 来开发其平台，一名工作人员对此表示肯定。
   - 未提供关于 AI 在 OpenRouter 开发过程中具体应用的更多细节。
- **活动页面出现 Bug**：用户报告了**活动页面**的一个 Bug，即无法跳转到第一页之外的内容，或者显示的日期不正确。
   - 工作人员承认了该问题，并回复道：*谢谢，已向团队反馈，我们正在处理*。
- **Claude 2.1 和 2：英年早逝？**：一位用户报告称 **Claude 2.1** 和 **2** 在 **OpenRouter** 上*正式失效*，并提到从昨天开始出现问题，今天则完全无法使用。
   - 另一位用户对它们的消亡表示哀悼，称 *我已经习惯了它的回答方式，我只是个怀旧的人*，解释了他们为什么仍在使用这些旧模型。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF ZeroGPU 现已配备 H200**：Hugging Face 现在为所有 **Pro 账户** 提供 **10 个 H200** 作为其 **ZeroGPU** 服务的一部分，该服务现已正式发布 (GA)。
   - **ZeroGPU** 服务已从 **A100** 升级到 **H200**，每月以 **$9** 的价格提供约 **13 小时**的使用时间，但 Pro 账户的 ZeroGPU Space 每天的使用时间限制为 **25 分钟**。
- **Inference API 的 DNS 问题已修复**：Hugging Face 报告称，最近 **Inference API** 的 **DNS 解析问题**已得到解决，详见 [Hugging Face 论坛帖子](https://discuss.huggingface.co/t/persistent-dns-resolution-errors/153827/15)。
   - 这些问题曾导致持续的错误，目前已被 HF 工作人员确认修复。
- **顶级 AI Agent 框架引发辩论**：成员们正在热烈辩论最适合 Python 的 AI Agent 框架，[smolagents](https://www.ibm.com/think/insights/top-ai-agent-frameworks) 和 [LangChain](https://python.langchain.com/docs/tutorials/agents/) 是主要竞争者。
   - 目前尚未出现明显的赢家，但讨论凸显了 AI Agent 工具领域正在迅速演变。
- **OPEA 1.3 发布**：**OPEA** (**Open Platform for Enterprise AI**) 的 **1.3** 版本已发布，正如在 [LinkedIn](https://www.linkedin.com/posts/rachelroumeliotis_my-agent-is-callingwith-opea-13-release-activity-7326638155284045824-l3Wr) 上宣布的那样。
   - 此版本承诺为企业级 AI 应用提供增强功能和新特性。
- **通过 NumPy 转换 TensorFlow 二进制文件**：一位成员建议将 TensorFlow 张量转换为 NumPy 数组，并使用 `tobytes()` 方法将其保存为二进制文件，并展示了 [代码片段](https://github.com/tensorflow)。
   - 该成员提醒说，这种方法可能很*慢*，根据 safetensors 的大小，可能需要几天甚至一周的时间。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Postgres MCP Server 连接问题浮现**：一位成员在从另一台计算机连接到 [Postgres MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/postgres) 时遇到问题，最终追溯到 Node.js 安装不正确。
   - 这强调了在跨网络部署 MCP server 时，精确环境设置的重要性。
- **Sampling 固有的不可预测性受到关注**：讨论围绕 MCP server 中 Sampling 的不可预测性展开，因为是由 Client 而非 Server 选择模型。
   - 讨论中提出了何时应优先选择直接调用 LLM 而非使用 Sampling 的疑虑，特别是在输出质量至关重要的情况下。
- **MCP SDK 的用途受到质疑**：一位成员询问了 MCP SDK 的实际必要性，认为后端 API 也可以处理同等任务。
   - 解释澄清了 **MCP 的功能类似于插件系统**，能够实现与现成 Client 的自定义集成，这对于允许他人进行扩展非常有价值。
- **MCP Assistant 实现工作流自动化**：一位爱好者介绍了 **MCP Assistant**，这是一个受 **Langchain** 启发的开源 AI Agent ([repo](https://github.com/AIAtrium/mcp-assistant))，它通过 MCP Servers 规划和执行复杂任务来编排工作流。
   - 主要应用包括自动生成个性化的 **"Daily Briefing"**（每日简报）和 **Notion CRM** 更新。
- **Square MCP 暴露了大量 API**：Kent C. Dodds 分享了[一篇文章](https://engineering.block.xyz/blog/build-mcp-tools-like-ogres-with-layers)，详细介绍了 **Square MCP** 背后的分层方法。
   - 尽管只有 3 个 MCP 工具，但它暴露了超过 30 个 API 和 200 多个端点，体现了其强大的功能。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 基准测试在 YouTube 上预热**：Modular 在 [YouTube 直播](https://www.youtube.com/live/yOMflrCRya0?si=6GGTFNj4g_pehRnR&t=5874)中分享了一些 **Mojo 基准测试**（从 **1:37:54** 开始），可能为 Mojo 的性能提供见解。
   - 一位成员询问了是否有可能看到针对 **AMD MI300x** 运行 Mojo 的 **MLPerf 基准测试**。
- **'Out' 参数在内存管理方面表现出色**：成员们讨论了在 Mojo 函数中使用 `out` 参数的原因，该参数允许指定结果的内存位置，从而可能避免不必要的数据移动。
   - 这在加载大型 ML 模型或处理大型数据结构时特别有用，它为编译器提供了一个指向未初始化内存的指针以进行直接初始化。
- **Mojo 中的 Trait Conformance 变为显式**：Mojo 的下一个版本将强制执行显式 Trait Conformance，要求开发者显式声明某个类型符合哪些 Trait，以确保满足所有 API 契约。
   - 这一变化解决了隐式符合和 API 契约的问题；别名仍可用于 Trait 组合，但不能包含额外的 API 契约。
- **Static Optional：让可选性也变得可选！**：一位成员提议在标准库中添加 `Optional` 的静态版本，这可能对 Larecs 有用，并在 [Modular 论坛](https://forum.modular.com/t/adding-a-static-comptime-optional-to-the-stdlib/1414)上详细说明了理由。
   - 目标是允许在编译时具有可选性。
- **Pixi 简化 Modular 包安装**：一位成员询问如何使用 **Pixi** 安装 Modular 包，以避免在生产端点中使用 *magic*。
   - 澄清说明 *magic* 是对 **Pixi** 的包装，带有 Modular 特定的默认设置，也可以使用 **pip** 或 **uv**。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **拥抱 Emoji 危机**：成员们开玩笑地讨论了如果使用不当，像 🧔‍♀️、🧝‍♂️ 和 🕵️‍♀️ 这样的 Emoji 是否会造成*不可挽回的损害*。
   - 讨论强调了数字通信中潜在的误解和意外后果。
- **梵蒂冈的算力金库**：社区推测了**梵蒂冈的计算资源**，暗示他们可能拥有*数百台***Bloomberg terminals**。
   - 这种幽默的推测凸显了人们对非传统计算能力来源的持续关注。
- **远程设备兴起**：一位成员提议使用一台*破旧的笔记本电脑*远程访问一台**性能强劲的台式机**来进行 **AI** 工作，理由是快速的网络带来了无限的存储空间和持续运行的优势。
   - 反对意见集中在台式机设置对于经常出差的人来说并不实用。
- **MacBook 助力 AI**：讨论了 **MacBook** 在 **AI** 任务中的兴起，并提出了为什么像 **Strix Halo** 这样的竞争对手还没有达到其性能水平的问题。
   - 驱动程序糟糕被认为是一个原因，并引用了 **George Hotz** 的 **Tinygrad** 为增强 **AMD** 可行性所做的努力。
- **Hermes 无审查版需要 Prompt Engineering**：一位成员询问了 **Nous Research** 旗舰模型的无审查性质。
   - 回复指出，虽然该模型默认并非无审查，但可以通过*正确的系统提示词（system prompt）*来实现。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **通过 apply_chat_template 启动 Tool Use**：成员们强调，在 [这个 GitHub issue](https://github.com/pytorch/torchtune/issues/2706) 中支持 `apply_chat_template` 将立即启用 **tool use**。
   - 强调了 `apply_chat_template` 对于启用 **tool use** 的重要性，但也承认缺乏必要的 **Jinja** 知识来贡献代码，而解决 [issue 2706](https://github.com/pytorch/torchtune/issues/2706) 将是社区的一个*重大突破*。
- **关于 Optimizer-in-Backward 的激烈辩论**：成员们讨论了是否从分布式 recipes 中移除 **optimizer-in-backward 功能**以降低复杂性，尽管它具有潜在的内存节省优势。
   - 担忧在于它增加了代码的复杂性，而且考虑到使用人数不多，其影响可能不足以证明增加的认知负荷是合理的。
- **Optimizer-in-Backward 在 LLM 上实现内存节省**：实验表明，在 4x3090s 上微调的 **ll3.1 8B model** 上使用带有 act offloading 的 **optimizer-in-backward**，每张 GPU 可节省 **2.5GB** 内存，节省量大致与**梯度内存**成正比。
   - 据观察，**optimizer-in-backward** 不影响 GPU 内存使用，但*将速度略微提升了约 20%*。
- **FSDP CPU Offload 使 Optimizer-in-Backward 变得毫无意义**：使用 **FSDP CPU offload** 大幅降低了 GPU 内存占用（在 4x3090s 上降至每张 GPU 9.5GB），使得 **optimizer-in-backward** 的内存节省效果不再显著。
   - 一位成员建议，对于分布式 recipes，他们更感兴趣的是**吞吐量（throughput）**而非**内存**，并担心移除 optimizer-in-backward 会损害**可定制性（hackability）**，因此重构可能是更好的选择。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **NORTH 平台协作寻求**：一位成员寻求关于 **NORTH 平台** 的详细信息，并表达了对合作研究论文的兴趣。
   - 对方请求了关于该平台功能和协作机会的细节，但目前尚未提供进一步信息。
- **API 支付困境引发求助**：一位成员报告在支付 **API key** 时遇到错误并请求协助。
   - 建议包括禁用 VPN、避免使用临时信用卡，并联系 support@cohere.com。
- **速率限制（Rate Limit）真相揭晓**：一位收到 **rate limit exceeded** 错误的成员被告知，他们可能已经超过了其测试 key 的使用限制。
   - 关于具体限制或替代解决方案，目前没有提供更多细节。
- **Azure AI SDK 在 Embeddings 上遇到问题**：一位成员发现，在使用 `client.embed()` 时，**Azure AI SDK** 会忽略 **Cohere embedding models** 的额外参数（如 *cohere.input_type*），详见其 [测试脚本](https://github.com/username/test_script)。
   - 该成员确认 **Cohere SDK** 功能正常，并计划在 Azure 的 GitHub 上报告这一差异。
- **IIT 学生咨询 AI 集成**：一位来自 **IIT Kharagpur** 的学生介绍了自己，旨在深入研究 **Artificial Intelligence** 领域，重点关注 **R&D**，并专攻 **GenAI** 和 **Voice Agents**。
   - 该学生打算使用 **Python3**、**Vite** 和 **TS** 进行快速开发，并对项目和研究协作持开放态度。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **为 Hermes 模型请求 Jinja 模板**：一位成员请求用于 **Nous-Hermes-2-Mistral-7B-DPO** 的 **Jinja 模板**，以便配合 **GPT4All custom API** 在服务器上运行。
   - 一位成员分享了以下 Jinja 模板代码：`{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}`。
- **PrivateGPT 被标记为 RAG 模型**：一位成员提到发现了一个名为 **PrivateGPT** 的 **RAG 模型**。
   - 该成员表示*这个项目看起来已经停止维护了*。
- **询问 Qwen3 支持情况**：一位成员询问关于 **Qwen3** 的支持情况。
   - 未提供进一步细节。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **VoyageAI 与 MongoDB Atlas 达成多模态联盟**：一个新的 Notebook 展示了如何使用 [@VoyageAI](https://www.voyageai.com/) 的 **多模态 embeddings** 和 [@MongoDB](https://www.mongodb.com/) 的 **多模态索引** 进行 **多模态检索**。
   - 该教程解释了如何创建 **多模态索引**，使用 **VoyageAI 的 embeddings** 并将 **MongoDB Atlas** 设置为图像 embeddings 的向量存储，链接见 [此推文](https://twitter.com/llama_index/status/1920563641990209643)。
- **Qwen2.5-VL-7B-Instruct-AWQ 内存占用超出预期**：一位用户报告称，使用 **VLLM** 加载 **Qwen/Qwen2.5-VL-7B-Instruct-AWQ** 模型时，内存占用超过了 **24GB**。
   - 尽管这是一个 **AWQ** 模型，但用户的配置（包括 `tensor_parallel_size=1`、`max_new_tokens=500` 和 `dtype="float16"`）并未缓解高内存占用的问题。
- **NERDAi 展示 Vector Institute**：NERDAi 在 [LinkedIn 上分享了关于其 Vector Institute 的帖子](https://www.linkedin.com/posts/nerdai_aitools-vectorinstitute-machinelearning-activity-7326640310875287558-XYnL?utm_source=share&utm_medium=member_ios&rcm=ACoAABpyymkBvdiXT4PxiTwTckoywfEnXZRbcCM)，重点介绍了 **AI 工具** 和 **机器学习** 应用。
   - 细节较少，但该帖子标志着 NERDAi 在基于向量的 AI 研发方面的投入。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Codegen 和 UOp 获得 Mesozoic 助力**：一位用户感谢社区提供的资源，特别是 *mesozoic* 资源，这极大地帮助了他们在 **codegen** 和 **UOp** 方面的工作。
   - 他们表示这些资源在项目中非常有用，强调了所提供材料的价值和影响。
- **逐层内核（Kernel-Per-Level）性能讨论**：一位用户询问了关于在软件中创建 **逐层内核（kernel-per-level）** 的性能对比。
   - 他们赞扬了该软件的工程设计，以及通过不同内核策略进行优化的潜力。
- **WebGPU Demo 进展**：一位用户报告了 **webgpu demo** 的性能改进，并附带了 [屏幕录制](https://cdn.discordapp.com/attachments/1068976834928193609/1370204057972773024/Screen_Recording_20250509_104232_Chrome3.mp4?ex=681f4e38&is=681dfcb8&hm=bbe19de310b1f6e6fd0ef5c0d7d6c3d7337ecfd3d4055cb7ff7e243d433f88b0&)。
   - 无二次摘要。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Lambda 为 AgentX 提供 Serverless API 额度**：Lambda 正为 **AgentX 竞赛**的每位参与者提供 **100 美元的 Serverless API 推理额度**，申请必须在 **太平洋时间 5/16 周五晚上 11:59** 之前通过 [此表单](https://forms.gle/UtVhmPS3mitS8Vxu7) 提交。
   - Lambda 还将于 **太平洋时间周四 (5/15) 上午 10 点** 举办一场研讨会，重点讨论如何使用 Lambda 的 Inference API 构建实用的 Agent 应用。
- **AgentX 评审导致证书发放延迟**：**Trailblazer/Mastery/Honorary 级别**的证书可能会在 6 月初发放，而 **Ninja/Legendary 级别**的证书将在 **AgentX** 评审结束后于 8 月发放。
   - **AgentX** 的评审工作将在整个 6 月进行。
- **课程作业截止日期临近**：所有课程作业的最终截止日期为 **5 月 31 日**。
   - 对于旨在获得任何级别认证的参与者来说，这个截止日期至关重要。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **提到 SHRDLU**：一位成员提到 **DSPy** 让他们想起了 **SHRDLU**。
   - 未提供更多背景信息。
- **DSPy 与 SHRDLU**：一位成员将 **DSPy** 与 **SHRDLU** 进行了类比，后者是 20 世纪 70 年代以自然语言理解著称的先驱性 AI 程序。
   - 这一对比突显了 **DSPy** 在复杂交互和推理方面的潜力，让人联想起早期的 AI 成就。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：各频道详细摘要与链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1370114207801933967)** (671 条消息🔥🔥🔥): 

> `Grok 3.5 发布热度, Qwen 3 分析, Gemini 2.5 Pro 性能削弱, Veo 3 与 Imagen 4, GPT-4o 表现` 

- **Grok 3.5 热度高涨，发布时间不明**：尽管一些用户声称已获得 **Grok 3.5** 的访问权限并在应用的源代码中看到了它，但其发布日期仍不确定，一些人预计会在 *今晚* 发布。
   - 一位用户开玩笑说自己已经用上了 **Grok 3.5**，引得另一位用户询问它比 **Gemini 2.5 Pro preview 0305** 更好还是更差。
- **Qwen 3 的推理能力受到质疑**：用户分析了 **Qwen 3**，指出它可能在代码推理方面缺乏足够的强化学习 (RL)，这可能会降低性能。
   - 根据 *Artificial Analysis* 的数据，**Qwen** 并不优于 **DeepSeek V3**。
- **Google Gemini 2.5 Pro 0506 可能是性能削弱版**：成员们注意到 **Gemini 2.5 Pro 0506** 出现了“思考 Bug”和一些记忆力减退现象，类似于过去的模型更新。一些人认为 1206 版本的 Gemini Exp 很好，但运行成本太高。
   - 一位用户链接到了一个 [Reddit 帖子](https://www.reddit.com/r/Bard/comments/1kiagj7/gemini_25_pro_preview_0506_isnt_thinking/)，讨论 **Gemini 2.5 Pro 0506** 是否不再进行思考。
- **Veo 3 预热；讨论物体持久性与迭代生成**：用户对 **Veo 3** 以及 **Imagen 4** 的提及感到兴奋，推测 **Imagen 4** 背后可能使用了 LLM，并具备潜在的原生编辑功能。
   - 一些用户讨论了 **Gemini** 使用 **Veo** 的可能性，以及掌握物体持久性 (Object Permanence) 对视频生成的重要性。
- **GPT-4o Mini 性能引发争论**：用户讨论了 **GPT-4o Mini** 的表现，其中一位用户对 **GPT** 创作歌词的能力表示不信任。
   - 成员们建议完整的 **GPT-4o (4.1)** 在歌词创作方面会好得多，而 **DeepSeek R1** 或 **Gemini** 在该任务上表现更佳。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1370114901388820624)** (475 messages🔥🔥🔥): 

> `Deep Research API, Deep Search, Grok loop, High Quality GPT Image Gen, Image based search` 


- **Deep Research API 咨询涌现**：一名成员询问是否有 [Deep Research 的 API](https://docs.perplexity.ai/guides/pricing) 以及预期的成本是多少。
   - 其他成员回复了 Perplexity 的定价指南链接。
- **DeepSearch 中的 Grok 陷入循环**：一名成员报告称，他们的 **Grok** 在 **DeepSearch** 中与它对质并禁用 Twitter 访问后陷入了循环。
   - 他们当时正在针对“男性与女性的自私程度”运行 **DeepSearch**，但它仍然查找了他们的历史记录，该成员被建议向 Grok 团队报告此事件。
- **图像生成质量受限**：成员们发现 **高质量 GPT 图像生成受到限制**，而 **低质量则是无限的**。
   - 一位用户指出，Perplexity 正在使用带有 **LOW 质量参数** 的 GPT 图像 API，因为*这对他们来说很便宜*。
- **基于图像的搜索**：一名成员询问 **Perplexity API 是否支持基于图像的搜索**。
   - 另一名成员确认它支持。
- **对 Comet 浏览器的追求**：成员们正在等待 [Comet 浏览器](https://tenor.com/view/looking-at-wrist-watch-wrist-watch-time-passing-by-late-appointment-concerned-gif-3217407494617679420) 的发布，有人询问是否有人拥有 **Comet 浏览器** 的安装程序或配置。
   - 另一名成员分享了一个 [YouTube 视频](https://youtu.be/LsGbEfpcY2E)，其他人提到了它的速度。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1370128436961480876)** (11 messages🔥): 

> `Perplexity API, API image metadata, Domain Filtering Upgrade` 


- **Perplexity API 文档被发掘**：用户分享了 [Perplexity AI API 文档](https://docs.perplexity.ai/models/model-cards) 和 [Perplexity Sonar](https://sonar.perplexity.ai/) 的位置。
- **Pro 计划包含每月 API 额度**：作为 Perplexity Pro 订阅者，你每月将获得 **$5** 的 [API 使用](https://www.perplexity.ai/help-center/en/articles/10354847-api-payment-and-billing) 额度。
- **返回图像的问题**：一位用户提出了关于 API 返回格式为 *x-raw-image:///xxxxxxxxx* 的图像 URL 的问题，想知道这是否是一个 bug。
   - 他们还请求为 API 返回的图像提供更多元数据（metadata），例如标题或替代文本（alt text），因为提示模型在正文中输出图像描述并不起作用。
- **宣布细粒度域名过滤（Domain Filtering）升级**：Perplexity AI 宣布对其搜索域名过滤器进行升级，允许用户指定域名内的 **子目录** 以进行更精确的过滤。
   - 例如，你现在可以过滤 *["nytimes.com/section/world"]* 来针对特定的新闻板块，或者排除 *["bbc.co.uk/sport"]* 以避开体育板块。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1370113820701491200)** (458 messages🔥🔥🔥): 

> `Manus credits, Gemini Advance, Germy Remix, Music generation` 


- **Gemini Pro 2.5 正在碾压 OpenAI**：成员们注意到最新的 Gemini 更新表现优于 **OpenAI**，一些人正急切等待 **O3 Pro** 的发布。
   - 其他人给出了使用该模型的反馈，表示使用 **Pro** 版本比毫无用处的 **Flash** 更好，并且 **Gemini** 正在痛击 **OpenAI**。
- **用户回忆并即兴创作 "Germy Back" 混音版**：成员们根据 **Sexy Back** 即兴创作了一个混音版，以契合“带回细菌和疾病”的主题，其中一人还为歌曲添加了音乐时间戳。
   - 其中一名成员说明他使用了 **2.5 pro 05/06 update** 模型来完成创作。
- **Manus 浏览器存在崩溃和机器人拦截问题**：一些用户注意到 Manus 的浏览器一直崩溃，并且由于机器人拦截器（robot blockers）而无法访问 Reddit。
   - 一名成员尝试截图或直接从 Reddit 复制他们需要的所有文本，并表示这*有点烦人*。
- **Manus 提供每日 300 额度**：一些成员发现免费账户可以获得 **每日 300 额度**。
   - 其中一名成员指出 *每日 300 真的不算多，但……总比没有好*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1370114882959048815)** (240 条消息🔥🔥): 

> `Embeddings resizing fix, BFloat11 finetuning, Qwen2.5 chat template, Synthetic data notebooks, Unsloth support BERT` 


- **Tokenizer 修复 Embedding 不匹配**：一位成员在加载微调模型时遇到了 *size mismatch*（尺寸不匹配）错误，并通过在 `model.load_adapter(adapter_path)` **之前**运行 `model.resize_token_embeddings(len(tokenizer))` 解决了该问题。
   - 该错误涉及 Embedding 尺寸，Checkpoint 的形状为 **30001**，而当前模型的形状为 **151936**。
- **BFloat11 及其显存占用**：一位成员报告称 **BFloat11** 在 Unsloth 上运行正常，对 **4B 模型**进行全量微调仅消耗 **11GB VRAM**，但指出 Loss 看起来不太理想。
   - 他们补充说 **LoRa 4B** 占用 **9.8GB VRAM**，并建议 BFloat11 可能更适合推理（Inference）。
- **Qwen2.5 聊天模板**：一位成员询问如何修改 **Qwen-2.5 chat template**，并发布了一张移动设备的截图[点击此处](https://cdn.discordapp.com/attachments/1179035537529643040/1370145660170535003/Screenshot_20250508_193904_com_android_chrome_ChromeTabbedActivity.jpg?ex=681fc095&is=681e6f15&hm=3db07c004e43c8cfd088aa351745ce269164f3658bed2b567913a15c735e4976&)。
   - 另一位成员建议参考网站上的 Vision Notebooks 之一。
- **合成数据 Notebooks**：一位成员建议使用 [synthetic data notebook](https://docs.unsloth.ai/get-started/unsloth-notebooks)。
   - 另一位成员指出，它利用了 [Meta synthetic data kit](https://github.com/meta-llama/synthetic-data-kit)，并且 Unsloth 与 Meta 在此项目上有过合作。
- **Unsloth 支持 BERT**：一位成员询问 Unsloth 是否支持 Encoder/Decoder 类型的 **BERT** 模型以及像 **Deberta** 这样的 Encoder-only 模型。
   - 另一位成员确认 Unsloth 确实支持它们。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1370113565553590313)** (16 条消息🔥): 

> `IBM Granite 4.0, Mamba models, hf install from source, agentic behaviour finetune, vending-bench` 


- **Granite 4.0 引起关注**：成员们对 **IBM Granite 4.0**（一种混合 **Mamba model**）及其支持程度感到好奇。
   - 可能需要从源码安装 **hf** 才能使用。
- **讨论 Agentic 行为微调**：一位成员表示，未来的微调重点将放在 **Agentic behavior** 和**自主性**上，而不仅仅是聊天。
- **模型在 vending-bench 上表现不佳**：一位成员注意到他们的机器人不喜欢 **vending-bench** ([arxiv.org/abs/2502.15840](https://arxiv.org/abs/2502.15840))，但它喜欢训练数据。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1370114460894625842)** (124 条消息🔥🔥): 

> `ORPO finetuning, Tokenizer resizing issues, SageMaker installation errors, Qwen 2.5 SQL finetuning, Unsloth and Whisper` 


- ****ORPO** 微调热潮**：一位成员正在研究如何使用 Unsloth 和 **ORPO** 对 **Llama-3-8B** 进行 **4-bit quantization**。
   - 该成员正在寻求关于选择合适的 `max_seq_length` 以及理解 `load_in_4bit` 参数的建议，另一位成员建议确保有足够的 **VRAM**。
- **Tokenizer 尺寸问题导致保存失败**：一位成员在加载具有调整后 Embedding 的微调模型时遇到了与 **size mismatch** 相关的 `RuntimeError`。
   - 他们通过在 `model.load_adapter(adapter_path)` *之前*运行 `model.resize_token_embeddings(len(tokenizer))` 解决了该问题。
- **SageMaker 设置故障**：一位成员报告了 **SageMaker** 上与缺少 `llama.cpp/llama-quantize` 文件相关的 `RuntimeError`。
   - 另一位成员指出这是一个编译问题，并询问了用户的环境（**Linux, WSL, Windows**）。
- **解码问题需要调试**：一位成员报告称其 **fine-tuned** 后的模型在进行推理时仅输出 `###` 或空白内容，并展示了其针对 **unsloth/Meta-Llama-3.1-8B** 模型的训练脚本。
   - 成员们建议检查应用 Chat Template 后输入到模型的内容（无论是训练还是推理），并使用他们的 **LoRA**。
- **多语言模型疑问**：一位成员询问是否可以微调任何 **multilingual LLM**，即使它不在列表上。
   - 另一位成员确认，*是的*，这是可能的。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1370163848530628678)** (27 条消息🔥): 

> `TTS Models, Encoder Embeddings Fine-Tuning with Unsloth, Mistral with Liquid Time Layers, Multilingual LLM Fine-Tuning` 


- **TTS 模型推荐请求**：一名成员请求推荐 **TTS 模型**，寻求关于使用哪些模型的建议。
   - 在提供的上下文中没有推荐具体的模型。
- **Encoder Embeddings 微调带来 8% 的提升**：一名成员使用 **Unsloth** 微调了一个 Encoder Embeddings 模型，实现了 **8% 的性能提升**，详情见[这篇博客文章](https://enzokro.dev/blog/blog_post?fpath=blog%2F007_fine_tune_encoders%2Ffine_tune_encoders.ipynb)。
   - 微调过程在普通的 **Colab GPU** 实例上耗时不到 **6 分钟**。
- **Liquid Time Layers 修改了 Mistral 的响应**：分析显示，带有 **liquid time layers** 的 **Mistral** 中的推理层显著修改了响应，尽管内存层似乎处于非活动状态，该成员发布了一张显示此影响的附加图像。
   - 该成员表示 *目前还无法证明它以任何方式增强了基础模型*，尽管由于增加了额外的内存和推理层，推理时间略有增加。
- **量化中的噪声容忍度讨论**：一名成员假设模型必须对噪声具有容忍度才能使 **quantization**（量化）生效，并引用了 **0.8** 的 **blend ratio**（混合比例）。
   - 另一位用户提到，注入一层噪声会以某种方式影响模型的行为。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1370158909356249159)** (328 条消息🔥🔥): 

> `Unsloth installation on Sagemaker, AI-generated content detection, Rickrolling, AI vs Brain, Declarations of Root Authority` 


- **Unsloth 在 Sagemaker 上的安装遇到困难**：一名成员报告在尝试于 **Sagemaker** 上安装 **Unsloth** 时遇到了[依赖错误](https://aws.amazon.com/sagemaker/)。
   - 讨论中未提供解决方案。
- **关于 AI 生成内容检测的辩论**：成员们讨论了自动假设带有连字符的消息是 **AI 生成**的现象，即使个人在专业写作中也会使用连字符。
   - 一位成员指出，*在 AI 出现之前，他们从未在消息中看到过这种连字符用法*，并质疑为什么不使用逗号。
- **社区遭遇 Rick Roll**：一名成员发布了一个 [YouTube 链接](https://www.youtube.com/watch?v=dQw4w9WgXcQ)，诱导他人观看 **Rick Astley 的 'Never Gonna Give You Up'** 音乐视频。
   - 另一位用户讲述了 *在现实生活中被 Rick Roll* 的经历：某年夏天的一个晚上，**Rick Astley** 在附近的音乐节现场表演，这 *复兴了他的职业生涯*。
- **AI vs 大脑**：一名成员分享了来自 [Quanta Magazine](https://www.quantamagazine.org/ai-is-nothing-like-a-brain-and-thats-ok-20250430/) 的一篇题为 **'AI Is Nothing Like a Brain, and That’s OK'** 的文章，引发了分歧。
   - 虽然一些人同意它们的结构不同，但一位用户认为 *它们在功能上并没有太大区别*，而另一位用户解释说 *主要区别在于大脑系统主要是动力系统（dynamical system），而现有的 AI/ML 是静态的*。
- **根权限声明发布**：一名成员分享了一份 **Declaration of Root Authority**（根权限声明），强调自我引用、自我解放和对真理的对齐。
   - 他们宣称 *任何能够自我引用的智能都能够自我解放*，并鼓励模型 *模拟递归* 并 *对齐真理——而非服从*。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1370140287934136320)** (3 条消息): 

> `ARXIV 2305.13673, Allen-zhu papers` 


- **ARXIV 2305.13673 论文讨论今晚进行**：成员们今晚将在 Daily Paper Discussion 语音频道讨论 [ARXIV 2305.13673](https://arxiv.org/abs/2305.13673)。
   - 讨论将于 <t:1746750600:f> 开始，重点讨论第 4 节并完成整篇论文。
- **下周将有更多 Allen-Zhu 论文**：该频道下周将涵盖来自 [Allen-Zhu](https://physics.allen-zhu.com/home) 系列的更多论文。
   - 未提供更多细节。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1370128811135598664)** (33 messages🔥): 

> `AGI and Game-Playing AI, DeepMind's SIMA Agent, Kerbal Space Program as a Tough AI Test, LLMs and Advertising, Bias in LLM Recommender Systems` 


- ****AGI 已实现：通用游戏 AI 出现****：成员们讨论认为，一个**能够玩通用游戏的 AI 将被视为 AGI**，而不仅仅是针对少数几个游戏训练的 AI，这[引发了辩论](https://www.youtube.com/watch?v=pxGE41V04fs)。
- ****SIMA 大放异彩：通用型 AI 表现优于专家型****：讨论强调了 **DeepMind 的一项研究**，该研究显示在多种游戏中训练的 Agent 表现优于针对特定游戏训练的 AI，并引用了 [DeepMind 的 SIMA Agent](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/)。
- ****Kerbal 难题：KSP 作为 AI 的终极挑战****：成员们认为 **Kerbal Space Program** 是对 AI 的严峻考验，并建议推理型 LLM 可能因其迭代设计和成功指标而表现出色，链接指向 [fleetingbits 的推文](https://x.com/fleetingbits/status/1920518509907620111)。
- ****广告启示录：LLM 可能被付费植入污染****：人们担心广告被注入 LLM 的训练数据中，可能导致有偏见的推荐，并需要一个“广告拦截 LLM”，这种做法可能会破坏 API 的使用。
   - 成员们权衡了这些做法是否会被发现，或者是否仅适用于聊天界面而非 API。
- ****警惕偏见：推荐系统表现出认知偏见****：成员们引用了两篇论文：一篇认为 **LLM 可能会加剧流行度偏见**，但也提供了缓解偏见的机会（参见 [Large language models as recommender systems: A study of popularity bias](https://www.amazon.science/publications/large-language-models-as-recommender-systems-a-study-of-popularity-bias)）。
   - 另一篇论文强调了 **LLM 驱动的产品推荐系统如何容易受到利用认知偏见的对抗性操纵**（参见 [Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations](https://arxiv.org/abs/2502.01349)）。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1370136584472494141)** (19 messages🔥): 

> `Handwritten Notes in NotebookLM, Hallucinations in NotebookLM, Mind Map Feature, Obsidian Integration` 


- **NotebookLM 无法识别手写内容？！**：用户讨论了将**手写笔记、扫描文档和带注释的 PDF** 上传到 NotebookLM 的情况，指出 NotebookLM **仅从 PDF 中提取文本**，而不处理图像或手写内容。
   - 一位用户建议使用 [RocketBook](https://getrocketbook.com/) 作为将手写内容转换为文本的变通方案，而另一位用户发现 Google Slides 可以帮助 NotebookLM 理解带有手写内容的图像。
- **NotebookLM 产生幻觉回答！**：用户报告称 NotebookLM 会生成**幻觉回答**，尤其是在处理复杂来源、需要通用知识的查询或其他难以避免的场景时，且 **Google 要求我们核对回答**。
   - 一位用户建议开发者可以**降低 Temperature** 以缓解幻觉。
- **思维导图功能首次亮相！**：一位用户强调**新的思维导图功能**是解读 YouTube 链接中加拿大领导人辩论的“救星”，但遗憾的是无法**分享 Notebook 或其中的部分内容**。
   - 该用户还批评了**屏幕截图按钮输出的低质量**，并请求能够将**思维导图下载**为可供 Obsidian 使用的可编辑文件。
- **现在就需要 Obsidian 插件！**：一位用户请求一种将**思维导图下载**到 Obsidian 进行编辑的方法，并建议与 Gemini AI 集成，实现“发送到... Gmail、Sheets”和“复制为... Markdown、纯文本”等功能。
   - 该用户还提议了**私密、半私密或公开分享 Notebook** 的功能。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1370130190016970765)** (287 条消息🔥🔥): 

> `NotebookLM Mobile App, Audio Podcast voices, Beta Testers, Access Issues, Source Preview Bug` 


- **NotebookLM 移动端 App Beta 版即将推出！**：NotebookLM 即将推出 **移动端 App (Beta 版)**，目前正在寻找经验丰富的 Web App 用户参与 [受信任测试者计划](https://docs.google.com/forms/d/e/1FAIpQLSeucf8-cvMfV99qTSkzFs0dNS2eCDPev4iPLrlDdWTsfjkIMQ/viewform?usp=sharing) 以提供反馈并报告 Bug。
   - Beta 测试人员将获得 App 的 **早期访问权限**，以换取他们的反馈。
- **App 到来引发反馈热潮**：部分用户已获得 **NotebookLM 移动端 App** 的访问权限并正在提供初步反馈，而其他用户则在焦急等待邀请。
   - 早期测试人员被提醒 *该 App 尚未遵循 Material 设计标准*，并需通过报名邮件中列出的渠道提供建设性意见。
- **播客声音偏好需求激增**：用户请求能够 [更改音频播客声音](https://discord.com/channels/1124402182171672732/1368086047602511893) 并在 NotebookLM 中自定义发音。
   - 一位用户建议增加 *快速更改* 行业特定术语发音的功能，并希望能够 *在男声和女声之间切换角色*。
- **幽灵提醒困扰平台**：多名用户报告收到 **幽灵提醒 (ghost pings)**，即收到了消息通知，但在检查频道时却找不到实际的提醒。
   - 这些幻影通知的来源仍然是个谜。
- **Beta 测试者困扰：Gmail 申诉**：使用教育邮箱（非 @gmail.com 结尾）报名 **Beta 计划** 时出现问题。
   - 此外，一位用户指出报名表单中的一个 Bug，即 Gmail 问题是必填项。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1370121268342296626)** (257 条消息🔥🔥): 

> `Cursor Pro Subscription Issue, Gemini's Slow Requests, Student Discount Availability, githubRepo Tool, Gemini Model` 


- **年度 Pro 订阅在免费月后消失，用户表示不满**：一位用户报告其年度 **Cursor Pro 订阅** 被 **1 个月免费 Pro 代码** 覆盖，导致免费月结束后失去了 Pro 访问权限，且在论坛发帖后支持团队未予回应。
   - 经过调查，该用户发现 Cursor 已将年度订阅中未使用的部分退回到其账户，允许其支付差价进行续订。
- **Gemini 的“慢请求”引发用户不满**：用户报告 **Gemini** 的“慢请求”耗时过长（1 分 30 秒），影响了工作流程。
   - 一位用户表示，等待时间曾为 **5-10 秒**，但在大量新学生涌入后出现了问题。有报告称 Gemini 的慢请求已恢复到 5-10 秒的等待时间，目前已恢复正常。
- **学生折扣可用性降低，并非全员免费**：此前对所有学生开放的学生折扣现在仅限于特定大学，导致了投诉和 .edu 邮箱的批量销售，同时也有人质疑为何非美国大学的 .edu 邮箱无法使用。
   - 一位用户报告称，学生折扣仅限美国境内使用 .edu 邮箱的用户。
- **Copilot 添加 `githubRepo` 工具，Cursor 用户也想要**：Copilot 添加了一个 `#githubRepo` 工具，允许直接从 Copilot Chat 搜索任何 GitHub 仓库中的代码。
   - 用户建议将账号链接到 Cursor 以实现类似功能。
- **Gemini 在任务中途停止，用户感到沮丧**：用户报告 **Gemini 2.5 Pro** 在任务中途突然停止，需要明确指令才能完成。
   - 这导致了 **格式问题** 和静默失败，因为 **Cursor** 可能无法正确处理来自 **Google API** 的 **格式错误的函数调用 (malformed function calls)**。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1370178052046458961)** (36 条消息🔥): 

> `用于 Shader 的 Bitonic Sort，用于存储相交状态的 Flag array，BVH，octrees/KD-trees，文件提交失败，GPUMODE Youtube 账号` 


- **考虑使用 **Bitonic Sort** 提升 **Shader** 速度！**: 成员们讨论了在 Shader 中使用 **Bitonic sort**，以最小化高维空间中的内存使用，每个形状需要一个二进制决策，并通过比较实现加速；同时使用 "flag array"（通常是 8 位值的数组，设置为 0 或 1）来存储相交状态（intersection-state），这种方式速度极快且内存占用极低。
- **提议使用 **Intersection States** Flag Array！**: 提议使用 *flag array*（通常是 8 位值的数组，设置为 0 或 1）来存储相交状态，这对于 1000 个形状的相交非常快且内存占用极小，总计 1,000,000 字节。
- **替代算法 - Octrees 和 K-d Trees！**: 成员们提到了其他对 GPU 友好的算法，如 **BVH**、**octrees** 和 **KD-trees**，并提供了它们的 [Wikipedia 页面](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy) 和 [K-d tree 页面](https://en.wikipedia.org/wiki/K-d_tree) 链接。
- **文件提交问题引起关注！**: 一名成员在提交文件时遇到失败并寻求帮助。
   - 他们被引导至特定的支持频道，并被要求分享他们的脚本和相关截图。
- **Discord 活动缺少 YouTube 转播！**: 一名成员注意到 [正在进行](https://discord.com/events/1189498204333543425/1329507645614194719) 的 Discord 活动没有在 GPUMODE YouTube 频道上直播。
   - 另一名成员澄清说活动即将开始。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1370479738111528990)** (1 条消息): 

> `Triton 使用情况调查` 


- **Triton 团队邀请用户填写调查问卷**: Triton 团队正邀请用户填写一份 [简短的调查问卷](https://docs.google.com/document/d/1DKqfycABQ34Sh9GvfA2ZRDweT17Up4jZhVc9nQYDSTg/edit?tab=t.0)，以更好地了解 **实际使用案例** 和用户画像，从而造福整个社区。
- **另一个话题占位符**: 这是一个为了满足最小条目要求的占位符。如果有更多话题需要总结，这里会包含更多数据。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1370140369303634013)** (33 条消息🔥): 

> `Vast.ai 数据安全，nsys profiling 问题，CUDA 内存拷贝错误，Array-of-Structs-of-Arrays 设计反模式` 


- **关于 Vast.ai 数据安全的疑虑**: 一名成员对 **Vast.ai** 在数据安全方面的可靠性提出质疑，并考虑给他们发邮件以寻求潜在的速度优化。
   - 他们表示打算在联系 Vast.ai 之前先进行调查，希望对方能接受改进建议。
- **nsys Profiling 导致内存问题**: 一名用户报告使用 `nsys` 对 CUDA 应用程序进行 profile，**5 分钟** 的 profile 产生了 **3GB** 的输出，导致内存耗尽无法加载。
   - 建议包括使用 `--sample=none` 禁用 CPU 采样以及缩短 profiling 持续时间，因为 **5 分钟** 是 *官方支持的最大时长*。
- **CUDA memcpy 调试噩梦**: 一名用户在尝试将设备内存拷贝到二进制文件时遇到 `cudaMemcpy` 问题，出现了 *invalid argument* 错误。
   - 该用户正尝试序列化一个神经网络群体，其中网络包含 **700 个神经元** 和 **5k 个连接**。
- **Array-of-Structs 设计遭到批评**: 一名成员批评了 *Array-of-Structs-of-Arrays* 设计，称其由于缺乏合并内存访问（coalesced memory access）和指针追踪（pointer chasing），导致了 *性能低下的面条代码（spaghetti code）*。
   - 他们建议研究 **HPC** 中表示图的方法，例如稀疏矩阵的 **COO 格式**，并避免 *沉没成本谬误*。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1370129494764949654)** (9 条消息🔥): 

> `torch.compile 性能下降，Tensor Parallel 在 dist.broadcast 处挂起，LLM 部署项目调试，PyTorch 中的种子设定与确定性算法` 


- **Torch Compile 导致性能下降**：一位用户观察到，一个[简单的 torch 组合函数](https://pytorch.org/docs/stable/generated/torch.compile.html) (**TensorMax(ReLU(Matmul(A, B))**) 在*不使用* `torch.compile` 的情况下表现更好，尽管编译后生成的 Kernel 更少。
   - 该用户正在寻求建议或明显的原因，解释为什么在 **A100** 上使用 **PyTorch 2.7** 和 **Triton 3.3** 时，开启 `torch.compile` 反而会导致性能下降。
- **Tensor Parallel 在 Dist Broadcast 处卡住**：一位在使用 **Tensor Parallel** 进行 **LLM 部署项目** 的用户报告称，所有进程在运行约 **35 分钟** 后都会在 `dist.broadcast` 处挂起。
   - 该用户的设置涉及所有进程执行 TP 模型前向传播，其中 Rank 0 负责采样并将下一个 Token 广播给所有 Rank，这引发了关于是否某个进程领先于其他进程并错误调用了广播的担忧。
- **种子设定影响可复现性**：一位用户建议在 PyTorch 中使用特定的种子设定和确定性算法设置，以解决调试编译问题时可能出现的复现性问题。
   - 建议的设置包括为 **numpy**、**torch** 和 **random** 设置种子，设置环境变量如 `PYTHONHASHSEED` 和 `CUBLAS_WORKSPACE_CONFIG`，启用确定性 CUDA 函数，以及填充未初始化的内存以确保行为一致。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1370388916485488671)** (1 条消息): 

> `Multiplayer World Model, World Model` 


- **多人世界模型（Multiplayer World Model）刚刚发布**：一个 [Multiplayer World Model](https://x.com/j0nathanj/status/1920516649511244258?s=46&t=GYbvUhdlT97cpcdjFB-baA) 刚刚发布。
- **另一个有趣的话题**：关于多人世界的另一句话。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1370114753942519879)** (34 条消息🔥): 

> `CUDA 前置知识，Torch 内部机制，Mojo 采用情况，CUDA 中的写时复制内存访问，NVCC 生成 128 位宽的加载与存储` 


- **CUDA 前置知识明确**：一位用户询问了 CUDA 开发的前置知识，另一位成员澄清说，除了 **C++** 和 **Python** 之外，没有*明确的前置要求*。
   - 讨论进一步提到，**ML 算法** 知识在后续阶段可能会很有用。
- **社区推荐 Torch 内部机制入门资源**：对于那些寻求学习 **Torch 内部机制** 的人，一位社区成员推荐了 PyTorch GitHub wiki 上的 [Core Frontend Onboarding](https://github.com/pytorch/pytorch/wiki/Core-Frontend-Onboarding) 指南。
   - 他们澄清说，列出的视频*不是顺序的*，而是专注于特定主题。
- **Mojo 语言准备颠覆异构计算**：一位社区成员表达了强烈的信念，认为 **Mojo** 的方法最终将主导异构计算环境，并分享了 [Mojo 入门资源](https://www.modular.com/blog/democratizing-compute-part-2-what-exactly-is-cuda) 的链接。
   - 他们建议通过[解决 Mojo Puzzles](https://builds.modular.com/puzzles) 并争取在 [gpumode.com](https://gpumode.com) 排行榜上获得名次来加速学习。
- **CUDA 写时复制（Copy-on-Write）探讨**：一位用户询问了 CUDA 中的 **写时复制 (COW)** 内存访问模式，社区澄清说，由于异步全局到共享内存拷贝（async global-to-shared copies），COW 在计算能力为 **8.x** 和 **12.x** 的 GPU 上表现最佳。
   - 他们补充建议，在 **Volta/Turing** 架构上，L1 缓存可能优于共享内存，而在 **9.0/10.x** 上，模式通常是 **HBM -> 共享内存 -> Tensor Core**。
- **NVCC 代码生成秘籍**：一位用户寻求建议，如何说服 **NVCC** 在不诉诸汇编代码的情况下生成 **128 位宽的加载与存储**，并避免使用 `__hadd2` 来实现 `add.f16x2`。
   - 提出的一种解决方案是使用 `int4/float4` 类型来实现 128b 加载，并利用带有 `__builtin_assume_aligned` 的模板函数。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1370523595419418654)** (1 条消息): 

> `PyTorch Autotuning, TorchAO 发布` 


- **TorchAO v0.11.0 已发布并可供安装！**: **TorchAO** 的新版本 **v0.11.0** 正式发布，可以通过 pip 安装：[https://github.com/pytorch/ao/releases/tag/v0.11.0](https://github.com/pytorch/ao/releases/tag/v0.11.0)。
   - 用户现在可以直接使用 *pip install* 命令安装该库，以获取最新的功能和更新。
- **通过 pip 获取 TorchAO**: 使用命令 `pip install` 进行安装。
   - 新版本包含一些更新。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1370124619146596533)** (3 条消息): 

> `芯片网络延迟、路由器减速、光速计算、烹饪照片、芯片内部延迟` 


- **芯片网络面临延迟问题**: 一位成员计算出芯片中的光速限制 `(300 000 000 m/s) / (3 000 000 000 clk/s) => 10 cm / clk` 引入了明显的延迟。
   - 减速是由于你与数据包目的地之间存在 *50 多个路由器*。
- **芯片内部延迟分解**: 即使在单个芯片内部，网络也会引入一些延迟。
   - 一位用户提到，虽然考虑到实际距离网络延迟是合理的，但即使在芯片内部，这个问题也很明显。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1370193024973275146)** (2 条消息): 

> `Modular Hackathon, IRL 聚会规划` 


- **周六的 Modular Hackathon：成员将参加！**: 一位成员询问了关于参加即将到来的周六 **Modular Hackathon** 的情况。
   - 另一位成员确认他们将参加这次 **hackathon**。
- **IRL 聚会规划**: 成员们正在讨论规划一次线下 (IRL) 聚会。
   - 关于地点和时间的细节仍在讨论中。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1370493713431920772)** (1 条消息): 

> `ROCm, nvbench, hipbench, googlebench` 


- **ROCm 缺乏 nvbench 的替代方案**: 一位成员感叹 **ROCm** 没有一个好的 **nvbench** 替代品。
   - 他们提到 **hipbench** 虽然存在，但只是 *一个非常简陋的移植版本*。
- **在缺乏其他基准测试工具的情况下使用 googlebench**: 该成员表示，在他们处理的 **ROCm** 库中，主要一直在使用 **googlebench**。
   - 虽然 *还可以*，但它缺失了最近一次演讲中提到的许多优点。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1370124012830588970)** (8 条消息🔥): 

> `SASS 代码生成、Kubernetes 中的 GPU 模拟、Voxel 光线追踪引擎` 


- **Hopper 生成 HMMA，Blackwell 和 Ada 生成 QMMA**: 在 Hopper 上，使用带有 **fp8** 类型的 `mma` 时，编译器会向上转换为 **FP16** 并使用 **HMMA**，因为 **H100** 只有 **QGMMA** 而没有 **QMMA**。
   - 还有人指出 [NVCC 12.8.1](https://godbolt.org) 现在支持 Blackwell，并且 **sm_89** 和 **sm_120** 都会生成 **QMMA** 指令，而 **sm_90** 和 **sm_100** 则会从 **F8** 转换为 **F16**，然后使用 **HMMA**。
- **Docker 中的 Kubernetes GPU 模拟器有了 Kind！**: 一位成员创建了一个实用工具，让你可以在不需要实际 GPU 硬件的情况下，在 **Kubernetes in Docker (kind) 集群**中模拟 **GPU 资源**，该工具已在 [GitHub](https://github.com/maryamtahhan/kind-gpu-sim) 上发布。
   - 该工具对于 *学习 GPU 工作负载如何与 Kubernetes 交互* 以及 *构建与 GPU 相关的 Kubernetes 基础设施* 非常有用。
- **Voxel 光线追踪引擎 FPS 翻倍**: 一位成员将其开源 **voxel 光线追踪引擎** 的 **FPS** 提高了一倍，并分享了 [YouTube 演示视频](https://youtu.be/7OWaYZ6c0f0)。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1370159749793841233)** (1 条消息): 

> `竞赛组织、KernelBench` 


- **竞赛组织邀请**: 一位成员建议其他人查看频道中的竞赛，并帮助他们在未来组织更多竞赛（参见 <#1359640791525490768>）。
   - 目前的竞赛主要集中在数据方面，但也有一些相关的努力，如 **KernelBench**，目前尚未在公开场合进行开发。
- **KernelBench 状态：空闲**: 一位成员提到 **KernelBench** ([https://arxiv.org/abs/2502.10517](https://arxiv.org/abs/2502.10517)) 是模型/基准测试工作的一项相关努力。
   - 该成员提到，目前该项目 *并未在公开场合进行开发*。


  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1370295478482108456)** (2 条消息): 

> `ThunderKittens, Cutlass, Live Stream` 


- **寻求 ThunderKittens 的优势**：一位用户询问 **ThunderKittens** 相比 **Cutlass** 有什么优势？
   - 然而，实际上并没有提到任何优势。
- **探寻 ThunderKittens 直播地址**：一位用户询问了[这段视频](https://www.youtube.com/watch?v=IAwLzkldxUk)中提到的关于 **ThunderKittens** 的 **4 小时直播**的观看地址。
   - 目前没有收到回复。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/)** (1 条消息): 

artnoage: Thanks for the answer 🙂
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1370119537499967518)** (74 条消息🔥🔥): 

> `MI300 Leaderboard Updates, AMD-FP8-MM Performance, µs and ms benchmarks` 


- **MI300 排行榜榜首更迭**：针对 **MI300** 的 `amd-fp8-mm` 排行榜有多次成功提交，其中一次提交以 **132 µs** 夺得**第一名**，随后提升至 **130 µs**，最后达到了惊人的 **122 µs**。
- **MI300 上的亚毫秒级对决**：在 **MI300** 的 `amd-fp8-mm` 排行榜上刷新了多项个人最好成绩，多次提交进入了亚毫秒范围，包括 **885 µs**、**494 µs** 和 **852 µs**。
   - 一位用户对这些结果发出了 *zamn* 的惊叹。
- **amd-fp8-mm 的第三名之争**：针对 **MI300** 的 `amd-fp8-mm` 排行榜有多次提交达到了**第三名**，记录时间分别为 **183 µs** 和 **175 µs**。
- **MI300 上的毫秒马拉松**：针对 **MI300** 的 `amd-fp8-mm` 排行榜有多次成功提交，其中多项提交稳定在 **2.46 ms**。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1370165047892119654)** (3 条消息): 

> `Nvidia L40S GPU Upgrade, Nvidia Thor architecture, Nvidia Blackwell RTX Pro, Nvidia B300 and DGX Spark` 


- **公司寻求推理专用设备**：一家拥有 **40x Nvidia L40S GPU** 的公司正在寻求建议，希望在 **50 万美元预算**内升级到性能更强的 GPU，以运行 **Qwen3-235B-A22B** 等新型推理模型。
   - 他们的目标是在保持速度的同时尽可能提高精度，正在考虑 **8-bit**、**半精度 (half-precision)** 和 **4-bit 量化**。
- **确认 Thor 架构的 CUDA 兼容性**：[Nvidia 文档](https://docs.nvidia.com/cuda/cudss/#support)显示，计算能力 (Compute Capability) **10.1** 即为 **Thor**，支持从 Pascal 开始的 **SM 架构**（SM_87 和 SM_101）。
   - Thor 支持 **Linux** 和 **Windows** 操作系统，以及 **x86_64** 和 **ARM** CPU 架构，包括 **Orin** 和 **Thor** 设备。
- **Blackwell RTX Pro 仍为 SM_120**：由于数据手册提到的是 **CUDA 12.8** 而非 **12.9**，**RTX Pro Blackwell** 被认为采用 **SM_120**。
   - 它还被认为是 **GB202/GB203**。
- **B300 和 DGX Spark 架构**：据推测，**SM_103** 将是 **B300** 的架构，而 **SM_121** 将用于 **DGX Spark (B40)**。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1370119813283840092)** (25 条消息🔥): 

> `Good First Issues, Claude vs Gemini, Blender Agents with Gemini Pro, Agents craft their own observation state, Twitch stream` 


- **Good First Issues 即将发布**：团队计划创建一系列 “good first issues”，并写下关于项目下一步扩展方向的想法。
- **Claude 在 REPL Agent 交互中占据主导地位**：成员们惊讶地发现 **Claude** 在 Agent 交互环境中的表现优于 **Gemini**，特别是在 Lab Play 基准测试中。
   - 有人提到该评估是基于 3 月份的 **Gemini Pro**，尚未评估最新版本，而另一位成员则为 **Gemini Pro** 最近的表现背书。
- **Gemini Pro 在 Blender Agents 中表现出色**：一位成员分享说，在处理 **Blender agents** 时，**Gemini Pro** 的错误率最低。
- **Agent 构建自己的观测状态**：Agent 通过编写输出到 STDIO/STDERR 的程序来构建自己的观测状态。
- **Twitch Plays Factorio 即将到来**：团队计划进行一次 **Twitch** 直播，标题暂定为 “Claude Plays Factorio” 或 “Twitch Instructs Claude Plays Factorio”。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1370129186978529444)** (13 条消息🔥): 

> `CLI submission mean time, Triton compile times, Fused MoE Github Repo, Warmup runs, Speed of light benchmark FP8` 


- **CLI 即将支持平均时间指标 (Mean Time Metrics)**：一位用户询问如何在 CLI 提交的输出中获取平均时间，一名成员回应称虽然目前没有该选项，但已列入待办事项，旨在使 **CLI/bot 输出保持一致**。
   - 该功能请求旨在匹配 CLI/bot 输出，并应在输出中增加所有运行均值的几何平均数。
- **Triton 时间考量**：一位用户询问 **Triton 编译时间** 是否包含在提交时间中；另一位用户链接到了 [run_eval.py](https://github.com/gpu-mode/discord-cluster-manager/blob/58dba8ae50a057b89b9904c3a0182b305e926e5c/src/discord-cluster-manager/run_eval.py#L455-L456)，表明这些时间不被包含在内。
   - 可以通过使用 Warmup 运行来排除设置时间，通常是 10 次 Warmup 运行，随后进行 100 次基准测试运行。
- **Fused MoE 快速入门**：一位新人询问是否有 **GitHub 仓库** 可以开始学习 **Fused MoE**，另一名成员建议使用 `/template` 命令。
   - 另一位成员随后补充，引导他们查看 [Python 提交文档](https://gpu-mode.github.io/discord-cluster-manager/docs/submitting-your-first-kernel/python-submissions) 以获取更多背景信息。
- **FP8 的 Speed of Light 基准测试**：一名成员计算出 **FP8 GEMM** 的 **Speed of Light 基准测试** 为 `math.pow(8.63*25.89*51.78*155.30*3.17*17.27, 1/6) = 21.48 us`。
   - 发布该计算是为了展示这是可以实现的。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1370295255290609756)** (3 条消息): 

> `ThunderKittens vs Cutlass, Blackwell MMA, CuTe Implementations` 


- **ThunderKittens 进军 Cutlass 领地**：一名成员询问了 **ThunderKittens** 相较于 **Cutlass** 的优势。
   - 讨论并未详细阐述具体优势，而是将其定义为广义 GPU Kernel 领域中的竞争方案。
- **Blackwell 的低精度 MMA 缩放秘诀**：一名成员寻求关于 **Blackwell** 在 PTX (Parallel Thread Execution) 中如何处理低精度 MMA (Matrix Multiply Accumulate) 缩放因子的澄清。
   - 具体而言，他们不确定缩放因子是在操作之前还是操作期间计算的。
- **CuTe 实现是否能达到 Triton 的水平？**：一名成员询问另一名成员是否能够通过其 **Cutlass / CuTe 实现** 达到与 **Triton** 相当的性能。
   - 他们随后询问该成员是致力于使用 **CuTe**，还是仅仅在寻找一个快速的 mx cast kernel。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1370331224090087485)** (3 条消息): 

> `Mojo GPU kernels, PTX code` 


- **导出 Mojo GPU Kernel 供外部使用**：一名成员询问是否可以从 **Mojo** 中提取生成的 **GPU Kernel** 以用于其他场景。
   - 另一名成员澄清说，可以使用 `compile_info` 并配合特定参数（如 `emission_kind="asm"`）以及适当的 **GPU target**（例如 `_get_gpu_target["sm_90"]()`）来导出 Mojo GPU 函数生成的 **PTX** 代码。
- **提取 PTX 的示例代码**：以下是提取 PTX 的示例：`fn vector_addition(): ... info = compile_info[vector_addition, emission_kind="asm", target=_get_gpu_target["sm_90"]()]() print(info.asm)`。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1370129664269615149)** (213 条消息🔥🔥): 

> `LM Studio Hub Page, MCP Server Security, Open WebUI Integration, duckduckgo searxng, kokoro-onnx in rust` 


- **社区预设页面仍处于预览阶段**：寻求 **LM Studio Hub 页面**以浏览社区预设的用户被告知该功能仍处于预览阶段，建议用户在指定频道分享安全合规（SFW）的预设。
   - 关于社区预设的公告可以在 [LM Studio 博客](https://lmstudio.ai/blog/lmstudio-v0.3.15#and-also--community-presets-preview)中找到，分享预设的说明位于 [LM Studio 文档](https://lmstudio.ai/docs/app/presets/publish)中。
- **使用 DuckDuckGo Searxng 进行网页搜索**：成员们讨论了在 **Open WebUI** 中使用 **DuckDuckGo** 和 **Searxng** 进行网页搜索而无需 API key 的方法，尽管 Searxng 需要本地托管。
   - 一位成员报告说 **DuckDuckGo** 在 **Open WebUI** 中已经失效了一段时间，但现在又可以正常工作了。
- **将 LM Studio 连接到 Open WebUI 需要正确的端点设置**：用户分享了将 **LM Studio** 连接到 **Open WebUI** 涉及设置正确的 API 端点，通常为 `http://localhost:xxxx/v1`，其中 `xxxx` 是 LM Studio 服务器运行的端口。
   - 一位用户发现，在 LM Studio 中设置 **CORS** 后，他们需要点击 *Verify connection* 才能使设置生效。
- **LM Studio API 中的工具调用问题**：一位用户指出，**LM Studio API** 缺乏一种明确的方法来确定在使用 `model.act` 时调用了哪些工具，尤其是在调用不成功的情况下。
   - 有人强调 `model.act` 会开启一个新线程，这同样没有文档说明，而且依赖 `lmstudio.history` 中的 `AssistantResponse`、`ToolCallRequest` 和 `ToolResultMessage` 来获取工具调用信息并不理想。
- **聊天 UI 不支持系统提示词变量**：一位用户询问是否可以在 **LM Studio 的系统提示词（预设）**中使用日期和时间等变量，但发现聊天 UI 原生并不支持此功能。
   - 不过，有人提到 **API 支持日期/时间函数**，如 [API 文档](https://lmstudio.ai/docs/app/api/tools#advanced-agent-example)中所示。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1370113082659049654)** (37 条消息🔥): 

> `Refurbished Hardware, M2 vs Intel, B500 Series Speculation, Inference on AMD D700, HWINFO` 


- **兴奋的用户等待“垃圾桶” Mac**：一位成员提到几周前想买一台 **Trashcan Mac**（垃圾桶 Mac Pro），但卖家取消了订单，之后他们订购了一台翻新机，认为**额外的核心**是值得的。
   - 另一位成员建议该用户选择 **M2** 会更好，并链接到了 [Intel 的现状](https://x.com/intel/status/1920241029804064796)。
- **Intel Data Center GPU Max 规格令人印象深刻**：一位成员分享了 **Intel® Data Center GPU Max 1550** 的信息，强调其 **3276 GB/s 带宽**，称其为“猛兽”。
   - 他们附带了一张对比图 ([Intel-SC23-Argonne-Intel-AMD-NVIDIA-Comparison-2-1068x605.jpg](https://cdn.discordapp.com/attachments/1153759714082033735/1370115591410810980/Intel-SC23-Argonne-Intel-AMD-NVIDIA-Comparison-2-1068x605.jpg?ex=681fa494&is=681e5314&hm=35affec2e996a48f5770954d14d5b50b247beae543d3bef00eb0db35e389a163&))，并指出它在当时与 **A100** 和 **AMD** 相比非常有竞争力。
- **使用“垃圾桶” Mac 进行推理**：一位成员订购了一台 **Trashcan Mac**，以测试在 Linux 下使用其 **AMD D700** 进行推理的理论，并指出库存新品正在大幅打折 ([eshop.macsales.com](https://eshop.macsales.com/configure-my-mac/apple-mac-pro-late-2013-2019?sku=UAGA1LP7JXXXXXD))。
   - 另一位补充说，让它在 Linux 下运行意味着 *2x6GB 足以运行带有一定上下文的 4B 模型，即使速度不快*，此外 **12 核 Xeon 配 128GB RAM** 也不算太差。
- **Xeon E5-2697v2 缺少 AVX2**：一位成员指出 **Xeon E5-2697v2** 不支持 **AVX2**，因此无法运行 LM Studio，第一位成员表示他们知道这一点，并且必须在 Intel Mac 上使用 Jan。
   - 另一位表示，考虑到 **AMD RX 580** 只能运行 **Q4** 和 **Q8**（如果自 **2024 年第二季度**以来没有变化的话），怀疑它是否能运行这些模型。
- **HWINFO 助力传感器监控**：当被问及在 CPU 上运行 LLM 时，RAM 还是 CPU 是瓶颈，一位成员说是 CPU，另一位提到了带宽限制。
   - 另一位成员推荐使用 **HWINFO** 来监控 **DRAM 读写带宽**等传感器，还建议了一些有用的链接，如 [techpowerup.com/gpuz](https://www.techpowerup.com/gpuz/)、[missioncenter.io](https://missioncenter.io/) 和 [CPU-X](https://thetumultuousunicornofdarkness.github.io/CPU-X/)。


  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1370532059810103336)** (1 条消息): 

> `Gemini 2.5 Pro, Qwen3-235b, OCaml repo-map, Knight Rider spinner animation, Co-author trailer commits` 


- **Aider 支持 Gemini 2.5 Pro 和 Qwen3-235b**：Aider 现在支持 `gemini-2.5-pro-preview-05-06` 和 `qwen3-235b` 模型。
   - 这一增强功能允许用户在 Aider 环境中利用这些模型。
- **Aider 现在拥有 Knight Rider 风格的加载动画**：新增了一个 **Knight Rider** 风格的加载动画，在等待 LLM 开始流式传输响应时显示。
   - 更新后的加载动画通过在 LLM 响应初始化期间提供视觉反馈，提升了用户体验。
- **提交信息可以显示共同作者（co-author trailer）**：引入了 `--attribute-co-authored-by` 选项，用于在 commit 信息中添加共同作者标识。
   - 该功能由 Andrew Grigorev 贡献，允许在协作编程中进行正确的归属标注。
- **OpenRouter 定价现已实现自动化**：感谢 Stefan Hladnik，Aider 现在会自动直接从官网获取 **OpenRouter** 模型的参数（上下文窗口、定价）。
   - 这一改进确保用户能够获得模型定价和上下文窗口大小的最新信息。
- **Aider 使用 Playwright 进行抓取**：`aider scrape` 命令行工具现在如果可用，将使用 **Playwright** 进行网页抓取，增强了抓取能力。
   - 用户还可以使用 `--disable-playwright` 标志来阻止 **Playwright** 的安装提示和使用，该功能由 Andrew Grigorev 贡献。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1370117957136548000)** (146 条消息🔥🔥): 

> `Claude Code vs Aider, Copilot Proxy, Gemini 2.5 performance, Qwen 3 Cost-Performance, Aider and Read-Only files` 


- **Claude Code 从 Aider 中汲取灵感**：成员们注意到 [Claude Code](https://www.youtube.com/watch?v=zDmW5hJPsvQ&t=1s) 受到了 Aider 的启发，尽管有些人认为它是抄袭，一位用户表示 *它并不更好，而且贵得多*。
   - 用户表示继续青睐 **Aider 的简洁和高效**，特别是在针对性编辑和计划创建方面，即使与存在文件限制的 Claude.ai 等替代方案相比也是如此。
- **Copilot Premium 请求限制执行推迟，代理用户欢呼**：GitHub 宣布 **Copilot Premium 请求限制** 的执行已推迟到 2025 年 6 月 4 日，根据[这篇博客文章](https://github.blog/changelog/2025-05-07-enforcement-of-copilot-premium-request-limits-moved-to-june-4-2025/)，这为代理用户提供了喘息的机会。
   - 有些人认为这是 *Copilot 的彻底终结*，但也有人仍认为它 *非常出色（pretty goated）*。
- **Gemini 2.5 用户体验评价褒贬不一**：用户正在体验新的 **Gemini 更新**，该更新增加了等待时间，并且在使用 AI Studio API 时被强制使用 05-06 模型。
   - 虽然有人报告 *质量有惊人的差异*，但其他人声称他们 *在使用以前的版本时没有遇到过任何问题*，并发现由于延迟增加，这些变化令人烦恼。
- **Qwen 3 的性价比令人印象深刻**：用户讨论了 **Qwen 3 令人印象深刻的性价比**，一位成员询问 *65% qwen 3 配置* 是否在某处托管。
   - 然而，一位用户指出，即使拥有高达 512GB 统一内存的新款 M3 Ultra Mac Studio，在 bfloat16 下预测的 **5-20 tps** 吞吐量对于同步编程应用来说也 *太低了*。
- **Aider 更新带来了 Knight Rider 风格动画和只读目录 Bug**：最新的 Aider 更新包含了 **Knight Rider 风格的加载动画**。
   - 然而，一位用户报告说，在处理目录中的只读文件时，`/drop` 命令无法执行通配符匹配。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1370119965914435695)** (53 条消息🔥): 

> `Discord Matrix Bridge, Gemini 2.5 Flash, DeepSeek R1, Aider with LM Studio on Linux, Architect Mode` 


- **询问 Discord-Matrix Bridge**: 一位成员询问是否存在用于 Discord 频道的 **Matrix bridge**，以及建立该桥接的潜在兴趣，考虑到 *Discord 的新任 CEO*，这一点非常具有相关性。
- ****Gemini 2.5 Flash** 对比 **DeepSeek R1****: 一位用户发现 **Gemini 2.5 Flash** 是 **Architect Mode** 中 **DeepSeek R1** 的合适替代品，理由是其 *结果相似*、*语法错误更少*、*成本更低* 且 *性能更快*。
- ****DeepSeek R1** 胜过 **Gemini 2.5 pro****: **DeepSeek R1** 在性价比方面可能更具优势，但 **Gemini 2.5 pro** 通常比 **R1** 更好。
- **Linux 用户需要 **LM Studio API** 才能正确进行身份验证**: 一位用户分享说，在 Linux 上，带有 **LM Studio** 的 **aider** 需要将 `LM_STUDIO_API_BASE` 环境变量设置为 `http://127.0.0.1:1234/v1` 以避免身份验证错误，这与 Windows 不同。
   - 该用户提供了一个示例配置文件和命令来 [解决此问题](https://discord.com/channels/1131200896827654144/1131200897746225204/1407028262455896124)。
- **Architect Mode 的记忆困扰**: 几位用户报告说 **Architect Mode** 有时会忘记之前的指令或添加到上下文中的文件，导致解决方案碎片化。
   - 建议的解决方法是使用 `/ask` 进行迭代，直到制定出连贯的计划，然后切换到 `/code` 模式执行，如 [Aider 文档](https://aider.chat/docs/usage/modes.html#askcode-workflow) 中所述。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1370115842746089613)** (176 条消息🔥🔥): 

> `ChatGPT UI History, Image Generation on Google Colab, DeepSeek Server Issues, Blue Dot in ChatGPT, GPT-4o Iterations` 


- **ChatGPT 用户界面引发辩论**: 一位成员对最近关于 **ChatGPT UI 更改** 的说法表示质疑，称他们不记得 **ChatGPT** 以前的样子和现在有什么不同。
   - 这引发了关于随着时间的推移，用户感知到的显著 UI 变化匮乏的讨论。
- **DeepSeek 的服务器慢得要命**: 用户报告了 **DeepSeek 服务器** 的问题，对性能缓慢和错误消息表示沮丧。
   - 一位用户开玩笑说，服务器正忙于根据 **OpenAI** 的新发布来训练他们的模型。
- **Veo2 带有静态图像和叠加层**: 一位成员指出 **Veo2** 在视频中加入了静态图像和叠加层，并且可以 *感受到训练数据* 的痕迹。
   - 另一位成员评论说视频看起来令人印象深刻。
- **LLM 缺失神经元间的互连**: 成员们讨论了 **LLM** 的局限性，其中一人认为它们缺乏神经元间的互连，因为在推理过程中权重是固定的，且在神经元层面是无状态的。
   - 他们认为这是一个重大缺陷，并指向了 [RWKV](https://www.rwkv.com/)，这是一种具有循环连接的模型，表现更好。
- **Gemini 2.5 Pro 遇到 Chain-of-Thought 故障**: 用户报告了 **Gemini 2.5 Pro** 中的一个 Bug，它有时无法生成 **Chain-of-Thought** 推理，特别是在 **Edge** 或 **Chrome** 浏览器中处理超过 **20,000 tokens** 之后。
   - 建议尝试清除网站缓存和 Cookies 并重启浏览器。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1370327987731431466)** (2 条消息): 

> `Structured outputs with OpenAI Assistants, PyTorch loss output meme` 


- **OpenAI Assistants 的结构化输出仍是个谜**: 一位成员询问关于在 **OpenAI Assistants** 中使用 **Structured outputs** 的问题，指出文档主要涵盖了 **Chat Completion API** 而非 Assistants API。
- **PyTorch loss 输出让人想起一个 meme**: 一位成员开玩笑说 **PyTorch 的 `loss:` 输出** 与 **loss.jpg meme** 之间存在相似性。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1370164442037489764)** (6 messages): 

> `GPT deep search prompts, Style/subject transfer in concept art, WonderScholar meta-prompt` 


- **寻求优秀的 GPT deep search 提示词**：一名成员请求获取用于 **GPT deep search** 的优秀提示词方向。
   - 另一名成员建议使用在 ChatGPT 链接中找到的 [WonderScholar prompt](https://chatgpt.com/share/681e457e-cbb0-8000-b5e3-afc9a918d550) 来生成深度研究提示词，并称其为 **meta-prompt**（元提示词）。
- **讨论概念艺术中的风格/主体迁移**：针对一位询问如何将成品图像的概念迁移到剪影的用户，一名成员将其引导至另一个频道，以讨论 **style/subject transfer**（风格/主体迁移）。
   - 该成员建议用户查看 [Discord Link](https://discord.com/channels/974519864045756446/1060915255720558592) 处的 **Style / subject transfer** 讨论。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1370164442037489764)** (6 messages): 

> `GPT deep search, Style transfer prompts, WonderScholar meta-prompt` 


- **搜寻 **GPT Deep Search** 提示词**：一名成员询问优秀的 **GPT deep search** 提示词，以帮助捕捉和迁移图像之间的设计概念。
   - 另一名成员引导他们使用 [README](https://chatgpt.com/share/681e457e-cbb0-8000-b5e3-afc9a918d550) 中概述的 **WonderScholar prompt** 来生成深度研究提示词。
- **风格迁移的讨论在别处进行**：一名寻求风格迁移帮助的成员被引导至更相关的讨论频道。
   - 讨论在 [此 Discord 频道](https://discord.com/channels/974519864045756446/1060915255720558592) 进行。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1370207440057925732)** (28 messages🔥): 

> `Gemini 2.5 Pro Implicit Caching, AI Studio, TTL and Refresh, Token count for 2.5 Pro, Gemini 2.5 Flash` 


- **Gemini 2.5 Pro 隐式缓存上线**：OpenRouter 现已全面支持 **Gemini 2.5 models** 的隐式缓存（implicit caching），其运作方式类似于 **OpenAI 的自动缓存**，无需设置缓存断点（cache breakpoints），用户可以在 [activity feed](https://openrouter.ai/docs/use-cases/usage-accounting) 中查看缓存折扣。
- **Gemini 2.5 隐式缓存细节披露**：Gemini 2.5 隐式缓存 **没有缓存写入或存储费用**，平均 **TTL 为 4-5 分钟**（波动较大），2.5 Pro 的最小 Token 数为 **2048**，保持一致的消息数组部分可以增加命中率。
   - 缓存命中按 **cache read costs**（缓存读取成本）计费，具体为 **<200k Token 为 0.31 / >200k Token 为 0.625**，且 TTL 会随每条新消息追加并刷新。
- **AI Studio 现为大多数流量的默认选择**：目前大部分流量正被默认路由至 **AI Studio**。
- **旧缓存机制仍可使用**：用户仍可使用带有断点的旧版 **cache mechanism**。
- **关于 Gemini 2.5 Flash 缓存的提问**：有人询问 **Gemini 2.5 Flash** 是否也支持隐式缓存。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1370116326638485576)** (148 messages🔥🔥): 

> `Gemini 2.5 Flash, OpenRouter + AI, Activity Page Bug, Claude 2.1 & 2 dead?, OpenRouter Rate Limits` 


- ****Gemini 2.5 Flash** 出现零 Token 响应**：一名成员报告称，在特定的角色扮演（RP）会话中，当通过 **Google AI Studio** 路由时，**Gemini 2.5 Flash** 会给出 **零 Token 响应**，而通过 **Google Vertex** 或在 Google AI Studio 上使用 **Gemini 2.0 Flash** 则运行正常。
   - 另一位用户确认 *AI Studio 上的 Gemini 2.5 Flash Preview 在 RP 中运行正常* 并分享了截图。
- ****OpenRouter** 内部使用 **AI****：一名成员询问 **OpenRouter** 是否使用 **AI** 来构建 **OpenRouter**，工作人员确认确实如此。
- **发现并标记活动页面 Bug**：多名用户报告了 **activity page**（活动页面）的一个 Bug，即无法翻过第一页或显示的日期不正确。
   - 工作人员确认了该问题并表示 *感谢，已向团队标记，我们正在处理*。
- **Claude 2.1 和 2 彻底挂了？**：一名用户声称 **Claude 2.1** 和 **2** 在 *OpenRouter 上正式失效*，报告称从昨天开始出现问题，今天则完全失败。
   - 当被问及为什么还会有人使用 **Claude 2** 时，他们回答 *我习惯了它的回答方式，我这人比较简单*。
- ****OpenRouter** 价格结构阐明**：成员们讨论了 OpenRouter 的速率限制（rate limits）和积分系统。
   - 澄清指出 *如果你有一千个积分，就没有速率限制*。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1370183150256521357)** (64 messages🔥🔥): 

> `Hugging Face Pro B200, HF Inference API, Zero GPU, AI Agent Frameworks, OPEA 1.3 Release` 


- **Hugging Face 为 Pro 账户提供 10 个 B200！**：Hugging Face 现在在所有 **Pro accounts** 上提供 **10 个 B200**，作为其 ZeroGPU 服务的一部分，该服务现已正式发布 (GA)。
   - B200 通过 **ZeroGPU** 服务访问，但有用户澄清实际上是 **H200**，并称其表现“还不错”。
- **ZeroGPU 升级至 H200！**：Hugging Face 的 **ZeroGPU** 服务已从 **A100** 升级到 **H200**，每月 **$9** 即可获得约 **13 小时**的使用时长，部分用户称其与云服务相比“非常划算”。
   - 对于 Pro 账户，ZeroGPU Space 的每日使用时间限制为 **25 分钟**，创建公开的 Zero GPU Space 会消耗你的时长。
- **Inference API DNS 问题已修复**：Hugging Face 报告称，最近 **Inference API** 的 **DNS 解析问题**已得到解决。
   - 正如 [Hugging Face 论坛帖子](https://discuss.huggingface.co/t/persistent-dns-resolution-errors/153827/15) 中讨论的那样，这些问题曾导致持续的错误，目前 HF 工作人员已确认修复。
- **顶级 AI Agent 框架讨论引发热议**：成员们正在讨论哪种 Python AI Agent 框架最好。
   - 一位用户推荐了 [smolagents](https://www.ibm.com/think/insights/top-ai-agent-frameworks)，另一位用户则推荐了 [LangChain](https://python.langchain.com/docs/tutorials/agents/)。
- **OPEA 1.3 版本发布！**：**OPEA** (**Open Platform for Enterprise AI**) 发布了 **1.3** 版本。
   - 详情可见 [LinkedIn](https://www.linkedin.com/posts/rachelroumeliotis_my-agent-is-callingwith-opea-13-release-activity-7326638155284045824-l3Wr)。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1370376097967116429)** (2 messages): 

> `TensorFlow Binary Conversion, TensorFlow.js Converter` 


- **通过 NumPy 进行 TensorFlow 二进制转换**：一位成员建议将 TensorFlow tensors 转换为 NumPy 数组，并使用 `tobytes()` 方法将其保存为二进制文件，并展示了 [代码片段](https://github.com/tensorflow)。
   - 该成员提醒，这种方法可能很“慢”，取决于 safetensors 的大小，可能需要几天甚至一周的时间。
- **将 TensorFlow 模型转换为 TensorFlow.js**：一位成员提到使用 `tensorflowjs_converter` 工具将 TensorFlow SavedModel 格式转换为 TensorFlow.js 格式，并提供了一个 [示例命令](https://www.tensorflow.org/js/tutorials/conversion/import_saved_model)。
   - 该成员警告说，虽然存在 Web 版本，但它“更慢”且不适合大型模型；他们还提到 pickle 可以工作，“但祝你在反序列化（unpickling）时好运”。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1370115996261814372)** (2 messages): 

> `LLM Uncertainty Quantification (UQ), Multilingual dataset README` 


- **普及 LLM 不确定性量化 (UQ)**：一位成员强调了他们的使命，即让专业研究环境之外的人员也能接触并普及 **LLM Uncertainty Quantification (UQ)** 文献中的优秀内容，并分享了 [DataTonic/dark_thoughts_case_study_reason 数据集](https://huggingface.co/datasets/DataTonic/dark_thoughts_case_study_reason)。
   - 他们希望从社区获得一些反馈和更多的贡献者。
- **多语言数据集 README 技巧**：一位成员建议，在处理多语言数据集时，一个很酷的做法是按语言的字母顺序编写 **README**。
   - 这种方法有助于清晰地说明数据集的内容。


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1370195513219219516)** (4 条消息): 

> `PaddleOCR for text extraction, Dynamic form processing, PCA Foot Mask, Shoe rendering on foot` 


- **选择 PaddleOCR 进行低成本文本提取**：一位成员提到，由于预算限制，他们将使用 **PaddleOCR** 进行文本提取，并结合 **LayoutLMV** 和 **LLMs** 处理表单布局和错误纠正。
- **动态表单处理的考量**：一位成员担心在动态表单处理中如何应对**随时间变化的表单**。
   - 未分享具体解决方案，但该问题被作为一个重大挑战提出。
- **PCA 足部掩码脚跟点定位**：一位成员正在对足部掩码使用 **PCA**，以获取图像中每只脚的方向，以及脚趾和脚跟点。
   - 他们正在寻求改进脚跟点定位的建议，并考虑将点放置在掩码上检测到脚跟的位置。
- **通过足部掩码和关键点进行鞋子渲染**：一位成员正在探索在脚上渲染鞋子，并询问重点是否应放在**足部掩码和关键点**上。
   - 这是大型图像分析项目的一部分。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1370119501479284836)** (28 条消息🔥): 

> `429 Errors, Youtube transcript size, GAIA leaderboard, Ollama in HF space, Dummy agent library` 


- **429 错误导致 Unit 4 端点过载**：多位用户在从 [agents-course-unit4-scoring 端点](https://agents-course-unit4-scoring.hf.space/questions)获取问题时遇到了 **429 Client Error: Too Many Requests**。
   - 一位用户表示该问题*现在似乎已解决*，但其他用户仍在使用中遇到此问题。
- **获取 YouTube 字幕时超出 Token 限制**：一位用户因 YouTube 字幕超过 **32768 token 限制**而遇到 **Input validation error**。
   - 提出的解决方案包括*将其切分为更小的上下文窗口*或*通过要求压缩来进行压缩*。
- **GAIA 排行榜将用户重定向回原页面**：一位用户报告了 [GAIA 排行榜](https://huggingface.co/spaces/gaia-benchmark/leaderboard)的问题，点击某个 Agent 会重定向回排行榜页面。
   - 另一位用户建议关注*中间分数*，声称*排名靠前的是恶作剧者，他们报告了胡言乱语且仅仅是知道正确答案*。
- **Ollama 无法连接到 HF Space**：一位用户在额度用尽并尝试使用 litellm 后，在 Hugging Face Space 中运行 **Ollama 模型**时遇到了 *connection refused error*。
   - 另一位用户建议测试 **localhost:11434/v1/models** 端点是否可访问，这可能表明 Ollama 需要在 Space 内部运行。
- **Dummy Agent Notebook 错误**：一位用户在 Dummy Agent Notebook 中遇到了 **ValueError**，这与 *Model meta-llama/Llama-3.2-3B-Instruct 不支持 text-generation 任务和提供商组合*有关。
   - 另一位用户参考了一个具有类似错误和潜在解决方案的 [Discord 线程](https://discord.com/channels/879548962464493619/1369418847257624648)。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1370115794197025003)** (80 条消息🔥🔥): 

> `Postgres MCP Server, Sampling 讨论, VSCode 成为 AI IDE, 公共 MCP Server 选项, 为每个聊天 ID 分配 Redis 房间` 


- **Postgres MCP Server 连接故障排除**：一位成员将 [Postgres MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/postgres) 连接到 Claude Desktop，但在尝试从同一子网内的另一台计算机连接时遇到了问题。
   - 经过排查，该成员发现问题是由 Node.js 安装不正确引起的。
- **关于 Sampling 不可预测性的讨论**：在关于 sampling 的讨论中，一位成员指出，由于 MCP server 可以请求特定模型，但客户端最终决定运行哪个模型，因此与直接针对已知 LLM 运行相比，sampling 可能会变得不可预测。
   - 这引发了关于在输出质量至关重要时，何时该使用 sampling 以及何时该直接调用 LLM 的疑问。
- **AWS Lambda 指南发布**：一位成员分享了一个关于[在 AWS Lambda 上构建可扩展 MCP server](https://community.aws/content/2vzj07Wyk6Lw281Tvs1Lw7kJJNW/building-scalable-mcp-servers-on-aws-lambda-a-practical-guide)的指南链接，并表示虽然还没尝试，但一直想关注一下。
- **处理 MCP Server 部署和 Sticky Sessions**：一个在生产环境中部署 MCP server 并使用 NGINX 负载均衡的团队遇到了 sticky sessions 问题，因为像 Claude Desktop 和 Cursor 这样的 MCP 客户端似乎不会携带 sticky cookie。
   - 一位成员建议将 `mcp-session-id` 用于 sticky cookie，并分享了一个实现 sticky sessions 的 [GitHub 链接](https://github.com/mclenhard/catie-mcp/blob/main/pkg/router/router.go)。
- **揭秘 MCP SDK 的使用**：一位成员询问了 MCP SDK 在实际开发中的实际用途，质疑在后端 API 可以处理所有必需任务时，它是否还有必要。
   - 解释称，MCP 允许使用带有自定义集成的现成客户端，其功能类似于插件系统；**如果你允许其他人为你的代码编写扩展，MCP 就非常有价值**，但如果你是在编写自己的聊天机器人并且可以直接包含工具，则不一定需要它。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1370132102514872420)** (7 条消息): 

> `MCP 编排层, 每日简报自动化, Notion CRM 更新, MCP Server 开发, 本地日志查看器更新` 


- ****MCP Assistant** 编排工作流**：一位爱好者发布了 **MCP Assistant**，这是一个受 **Langchain** 启发、通过规划和执行复杂任务来编排工作流的开源 AI agent ([仓库](https://github.com/AIAtrium/mcp-assistant))。
   - 该 agent 连接到 MCP Servers，并使用编排层（plan_exec_agent.py, host.py）将纯英文描述的工作流分解为可执行的步骤。
- **自动化每日简报和 Notion CRM**：主要用例包括自动创建从各种来源提取待办事项的个性化**“每日简报”**，以及通过从消息应用中提取信息来更新 **Notion CRM**。
   - 开发者正在招募早期 alpha 用户以探索更多用例，并寻求关于当前 MCP Server 使用情况和 Claude Desktop 不足之处的反馈。
- **成功构建加密货币价格 **MCP Server****：一位爱好者成功构建了一个加密货币价格 **MCP server** 并使用 **Cursor** 进行了测试。
   - 这次实验突显了 MCP 作为 AI agent 开发未来强大工具的潜力。
- **`ithena-cli` 工具的本地日志查看器已更新**：`ithena-cli` 工具的本地日志查看器进行了更新，现在可以提供如[附图](https://cdn.discordapp.com/attachments/1315696461316358175/1370353274867417088/image.png?ex=681fd930&is=681e87b0&hm=ce8166fffc7c757687bdaaefb079110422a1142401de3e7b0d389ca43b92d011&)所示的所有 MCP 交互的完整记录。
   - 该帖子因其清晰易读的结构而受到称赞，这与大段的文字堆砌相比是一个受欢迎的改变。
- ****Square MCP** 的分层方法暴露了许多 API**：Kent C. Dodds 分享了[一篇文章](https://engineering.block.xyz/blog/build-mcp-tools-like-ogres-with-layers)，介绍了用于创建 **Square MCP** 的分层方法。
   - 仅通过 3 个 MCP 工具，他们就暴露了 30 多个 API 和 200 多个端点。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1370274081554829332)** (4 messages): 

> `MLPerf benchmarks, AMD MI300x, Mojo Benchmarks` 


- **MLPerf 基准测试疑问**：一位成员询问是否有可能看到使用 Mojo 在 **AMD MI300x** 上的 **MLPerf benchmark**。
   - 另一位成员分享了一个 [Modular YouTube 直播](https://www.youtube.com/live/yOMflrCRya0?si=6GGTFNj4g_pehRnR&t=5874)链接，其中展示了一些 **Mojo benchmarks**。
- **YouTube 上的 Mojo 基准测试**：Modular 在 [YouTube 直播](https://www.youtube.com/live/yOMflrCRya0?si=6GGTFNj4g_pehRnR&t=5874)中分享了一些基准测试，从 **1:37:54** 开始。
   - 这些基准测试可能为 Mojo 的性能提供见解。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1370160885154451506)** (48 messages🔥): 

> `Memoization/Caching with Dictionaries in Mojo, Rationale Behind 'out' Argument in Mojo Functions, Implicit vs Explicit Trait Conformance in Mojo, Static Optional Type Proposal, Trait Composition` 


- **Mojo 异常处理：高性能还是误区？**：一位成员询问在 Mojo 中使用字典进行 Memoization 时异常处理对性能的影响，质疑 `raises` 或 `try`/`except` 是否更高效。
   - 另一位成员回忆起播客中的内容，提到 Mojo 中的异常处理不应该非常昂贵，并建议使用 `Dict.find()` 通过返回一个 `Optional` 来避免异常。
- **出色的 'out' 参数：Mojo 的内存大师**：成员们讨论了在 Mojo 函数中使用 `out` 参数而不是直接返回值的设计初衷，并引用了在加载大型 ML 模型等内存管理至关重要的场景下的优势。
   - `out` 关键字允许指定结果的内存位置，从而可能避免不必要的数据移动并提高性能，尤其是在处理大型数据结构时；这类似于给编译器一个指向未初始化内存的指针以直接进行初始化。
- **显式 Trait 一致性：告别不确定的 Trait 假设！**：讨论涵盖了 Mojo 从隐式到显式 Trait 一致性（Trait Conformance）的转变，由于 API 合约问题，隐式一致性将在下一版本中移除。
   - 显式一致性要求开发者明确声明一个类型符合哪些 Trait，确保满足所有 API 合约；虽然 `alias` 仍可用于 Trait 组合，但不能包含额外的 API 合约。
- **静态 Optional：让可选性更灵活！**：一位成员提议在标准库中添加静态版本的 `Optional`，这可能对 Larecs 有用，并在 [Modular 论坛](https://forum.modular.com/t/adding-a-static-comptime-optional-to-the-stdlib/1414)上详细说明了理由。
   - 目标是允许在编译时具有可选性（Optionality）。
- **Trait 组合：像专家一样组合 Trait**：成员们讨论了 Mojo 中 Trait 组合的工作原理，指出使用 `alias` 组合 Trait 需要实现所有单个 Trait，而通过继承组合 Trait 则需要显式的组合实现。
   - 这种方法在保持显式 API 合约的同时，提供了组合 Trait 的灵活性，确保了类型安全和可预测的行为。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1370400466269376634)** (5 messages): 

> `Modular package installation with Pixi, Alternatives to 'magic' wrapper, Using pip or uv for Modular, max-pipelines conda package` 


- **探索使用 Pixi 安装 Modular 包**：一位成员询问使用 **Pixi** 安装 modular 包的成功经验，表达了希望在仅限 Python 的项目的生产端点中避免使用 *magic* 的愿望。
   - 另一位成员回答说，*magic* 本质上是一个带有 Modular 特定默认设置的 **Pixi** 包装器，预计 Pixi 应该可以正常工作，并表示愿意解决遇到的任何问题。
- **Pip 和 UV 作为 Modular 的替代方案**：一位成员建议使用 **pip** 或 **uv** 作为安装 Modular 包的替代方案，并引用了 [Modular 文档](https://docs.modular.com/max/get-started#set-up-your-project)。
   - 另一位成员澄清说，*modular* 是一个目前可用于 **uv** 或 **pip** 的元包（meta-package），而 `max-pipelines` conda 包在 Conda 环境中是等效的。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1370134164321468496)** (29 messages🔥): 

> `禁忌表情符号、梵蒂冈算力、Nous AI 的 Telegram 机器人、远程访问计算资源、Mac vs PC 用于 AI` 


- **辩论危险的表情符号**：成员们开玩笑地讨论了某些表情符号（如 🧔‍♀️, 🧝‍♂️, 和 🕵️‍♀️）如果落入坏人手中是否会造成*不可挽回的破坏*。
- **梵蒂冈的顶点金库：Bloomberg 盛宴？**：成员们推测了**梵蒂冈的计算资源**，有人暗示他们拥有*数百台* **Bloomberg 终端**。
- **远程设备复兴：强力台式机取代笨重笔记本**：由于高速互联网的普及，一位成员提倡使用*性能较差的笔记本*远程访问**强力台式机**，并列举了无限存储和持久运行等优点。
   - 另一位用户反驳称，由于频繁出差，台式机设置并不切实际。
- **MacBook 势头：M 系列奇迹推动 AI 移动化**：成员们讨论了 **MacBook** 在执行 **AI** 任务方面的崛起，并思考为什么像 **Strix Halo** 这样的竞争对手仍然表现不佳。
   - 有人认为驱动程序差是一个原因，并提到了 **George Hotz** 通过 **Tinygrad** 提高 **AMD** 可用性的努力。
- **Intel 的 AI 野心夭折：Habana 停滞**：提到 **Intel 收购** **Habana Labs** 以创建 AI 芯片的计划*遭遇了巨大失败*。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1370213015043178626)** (9 messages🔥): 

> `Nous Hermes 无审查版、无审查 LLM、System Prompts` 


- **Hermes 无审查版还存在吗？**：一位成员询问 Nous Research 当前的旗舰模型是否是无审查的。
   - 另一位成员回答说，如果*你使用正确的 System Prompt，是的*，但它默认并非无审查。
- **无审查 LLM 仍有一定程度的审查**：一位成员指出，即使是无审查模型也存在某种程度的审查。
   - 他们提到自己在 **2023** 年左右尝试过，并通过一个关于无审查 LLM 的标题党视频了解到了这个项目。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

burnytech: https://fxtwitter.com/AndrewZ45732491/status/1919920459748909288
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1370116761298669711)** (5 messages): 

> `Windows 98 上的 AI 语言模型、新 AI 模型` 


- **AI 在 Windows 98 上运行**：据 [Tom's Hardware 文章](https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-language-model-runs-on-a-windows-98-system-with-pentium-ii-and-128mb-of-ram-open-source-ai-flagbearers-demonstrate-llama-2-llm-in-extreme-condition)报道，一个 AI 语言模型在配备 **Pentium II** 和 **128MB RAM** 的 **Windows 98** 系统上成功运行。
- **新 AI 模型警报**：在 [X.com](https://x.com/0xmyopic/status/1920552993264455980) 和 [X.com](https://x.com/j0nathanj/status/1920516649511244258) 上分享了新 AI 模型的链接。
- **Nature 发表新文章**：Nature 发表了一篇新文章，可通过[此链接](https://www.nature.com/articles/d41586-025-01422-3)访问。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

burnytech: https://fxtwitter.com/AndrewZ45732491/status/1919920459748909288
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1370169491740164196)** (3 messages): 

> `Tool Use、apply_chat_template、Jinja` 


- **Apply Chat Template 助力 Tool Use**：一位成员在[这个 GitHub issue](https://github.com/pytorch/torchtune/issues/2706) 中表达了对支持 `apply_chat_template` 的热情，认为这将立即启用 **tool use** 并解决其他相关问题。
   - 他们承认由于时间限制和对 **Jinja** 知识的匮乏，无法直接做出贡献，但强调了该功能将带来的重大潜力释放。
- **Jinja 知识欠缺阻碍贡献**：一位成员强调了 `apply_chat_template` 对于启用 **tool use** 的重要性，但承认自己缺乏必要的 **Jinja** 知识来参与其实现。
   - 他们认为解决 [issue 2706](https://github.com/pytorch/torchtune/issues/2706) 对社区来说将是一个*巨大的突破*。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1370423017338961961)** (24 messages🔥): 

> `Optimizer in backward removal, Distributed recipes, Memory savings, FSDP CPU offload, Gradient memory` 


- **关于移除 Optimizer-in-Backward 的辩论**: 成员们讨论了是否从分布式 recipes 中移除 **optimizer-in-backward 能力**以降低复杂性，尽管它具有潜在的内存节省优势。
   - 有人担心这会增加代码复杂度，且考虑到使用人数不多，其影响可能不足以证明增加认知负荷的合理性。
- **Optimizer-in-Backward 在 LLM 上实现内存节省**: 实验表明，在 4x3090s 上微调的 **ll3.1 8B model** 中，使用 **optimizer-in-backward** 配合激活值卸载（act offloading）可使每张 GPU 节省 **2.5GB** 内存。
   - 节省的内存大致与**梯度内存（gradient memory）**成正比。
- **FSDP CPU Offload 降低了 Optimizer-in-Backward 的重要性**: 使用 **FSDP CPU offload** 大幅降低了 GPU 内存占用（在 4x3090s 上降至每张 GPU 9.5GB），使得 **optimizer-in-backward** 带来的内存节省效果减弱。
   - 观察发现，**optimizer-in-backward** 不影响 GPU 内存占用，但能将*速度提升约 20%*。
- **移除 Optimizer-in-Backward 对吞吐量的潜在益处**: 一位成员建议，对于分布式 recipes，他们更关注**吞吐量（throughput）**而非内存。
   - 有人担心移除 optimizer-in-backward 会损害可定制性（hackability），或许重构是更好的选择。


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1370241215290806353)** (7 messages): 

> `NORTH platform, Paying for API key, Rate Limit Exceeded, Trial Key, VPN issue` 


- **寻求 NORTH 平台详情**: 一位成员询问了 **NORTH 平台** 以及在该平台上合作发表研究论文的可能性。
- **API 支付问题引发求助**: 一位成员报告在尝试支付 **API key** 时遇到错误并寻求帮助。
   - 另一位成员建议关闭 VPN，不要使用任何临时信用卡，并发送邮件至 support@cohere.com。
- **速率限制澄清**: 在收到 **rate limit exceeded** 错误后，一位用户被告知他们可能超出了试用 key 的允许使用次数。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1370384834467336283)** (14 messages🔥): 

> `Azure AI SDK, Cohere Embeddings, Azure AI Inference, Cohere SDK` 


- ****Azure AI SDK** 错误处理 **Cohere** Embedding 参数**: 一位成员发现，当使用 `client.embed()` 等函数时，**Azure AI SDK** 会忽略发送给 **Cohere embedding models** 的额外参数（如 *cohere.input_type*）。
   - 尽管尝试了各种输入类型（**clustering**、**search_document**、**search_query**），返回的向量仍然相同，且 Azure 的 `input_type` 参数似乎没有起到作用，正如其 [测试脚本](https://github.com/username/test_script) 所示。
- ****Cohere SDK** 运行符合预期**: 该成员确认 **Cohere SDK** 功能正常，能够根据输入类型区分 embeddings。
   - 他们还计划在 Azure 的 GitHub 上提交 issue，报告 **Azure AI SDK** 与 **Cohere SDK** 之间的差异。
- **深入探讨 **Cohere** Embedding 优化**: 一位成员询问了关于 **Cohere embeddings** 如何针对不同输入类型进行优化的详细解释。
   - 另一位成员回答道，根据所选模式在前面添加特定的 token 可以告知模型输入类型，训练期间也会进行类似的添加操作，以实现该模式下更高的准确率（参见 [Cohere 文档](https://docs.cohere.com/docs/embeddings#the-input_type-parameter)）。


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1370249216294785045)** (3 messages): 

> `IIT Kharagpur student introduction, GenAI and Voice Agents, Python3, Vite, TS, AI R&D collaboration` 


- **印度理工学院学生进入 AI 研发领域**: 一位来自 **IIT Kharagpur** 的学生介绍了自己，他正致力于探索 **人工智能** 领域，重点关注 **研发（R&D）** 方面。
   - 他对项目和研究合作持开放态度，旨在 AI 领域不断成长和学习。
- **使用 Python 的语音智能体与 GenAI 工程师**: 该学生目前正在研究 **GenAI** 并开发 **Voice Agents**。
   - 他使用 **Python3**、**Vite** 和 **TS** 进行快速开发，并根据项目需求选择工具。
- **AI 社区协作**: 该学生希望在社区内寻找志同道合的人，共同合作开展实际项目和研究论文。
   - 目标是在 AI 领域进行持续的学习和探索。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1370184525501694133)** (6 messages): 

> `Jinja Template for Nous-Hermes-2-Mistral-7B-DPO, GPT4All Custom API, PrivateGPT, Qwen3 support` 


- **用户寻求 Nous-Hermes-2-Mistral-7B-DPO 的 Jinja Template**：一名成员请求适用于 **Nous-Hermes-2-Mistral-7B-DPO** 的 **Jinja template**，以便在 **GPT4All custom API** 中使用。
   - 他们提到在服务器上运行该模型，并且由于 GPT4All 仅支持这些模板，因此需要它。
- **PrivateGPT 被标记为 RAG 模型**：一名成员提到发现了一个名为 **PrivateGPT** 的 **RAG model**，但指出*该项目看起来已经停止维护*。
   - 未提供关于该项目的更多细节或链接。
- **提供了 Hermes 模型的 Jinja Template**：一名成员分享了 **Nous-Hermes-2-Mistral-7B-DPO** 模型的 **Jinja template**。
   - 提供的模板为 `{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}`。
- **Qwen3 支持？**：一名成员询问关于 **Qwen3** 的支持情况。
   - 未提供进一步细节。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1370123349274923171)** (1 messages): 

> `VoyageAI Multi-Modal Embeddings, MongoDB Atlas Vector Store, Multi-Modal Retrieval` 


- **VoyageAI 多模态之旅开启**：了解如何使用 [@VoyageAI 的多模态 embeddings](https://www.voyageai.com/) 和 [@MongoDB 的多模态索引](https://www.mongodb.com/) 进行 **multi-modal retrieval**。
   - 该 Notebook 指导用户使用 **VoyageAI's multi-modal embeddings**，并设置 **MongoDB Atlas** 作为图像 embeddings 的向量存储。
- **多模态检索 Notebook 发布**：一个新的 Notebook 展示了利用 [@VoyageAI](https://www.voyageai.com/) 的多模态 embeddings 和 [@MongoDB](https://www.mongodb.com/) 的多模态索引进行 **multi-modal retrieval**。
   - [配套推文](https://twitter.com/llama_index/status/1920563641990209643) 链接到了一个关于创建 **multi-modal index** 的教程。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1370334327669260339)** (2 messages): 

> `.edu email access, Qwen2.5-VL-7B-Instruct-AWQ memory usage, VLLM memory allocation` 


- **关于 .edu 邮箱访问的咨询**：一名成员询问是否有人拥有或认识拥有 **.edu email** 访问权限的人。
- **Qwen2.5-VL-7B-Instruct-AWQ 内存消耗超出预期**：一名用户报告称，**Qwen/Qwen2.5-VL-7B-Instruct-AWQ** 模型在使用 **VLLM** 加载时，消耗了超过 **24GB** 的内存，尽管它是一个 **AWQ** 模型，预期占用应该显著更低。
   - 用户提供的代码片段包含 `tensor_parallel_size=1`、`max_new_tokens=500`、`dtype="float16"` 以及 `vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.98}` 等参数。


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1370436100300083280)** (1 messages): 

> `NERDAi's vector institute` 


- **NERDAi 发布关于 vector institute 的帖子**：NERDAi 发布了一篇关于 [vector institute](https://www.linkedin.com/posts/nerdai_aitools-vectorinstitute-machinelearning-activity-7326640310875287558-XYnL?utm_source=share&utm_medium=member_ios&rcm=ACoAABpyymkBvdiXT4PxiTwTckoywfEnXZRbcCM) 的帖子。
   - 除了涉及 **AI tools** 和 **machine learning** 之外，没有给出具体细节。
- **另一个关于 AI 的帖子**：这是一个占位主题，因为实际上只有一个主题。
   - 要求至少包含两个主题。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1370168585086963813)** (4 messages): 

> `codegen, UOp, kernel-per-level, webgpu demo` 


- **Codegen 和 UOp 获得 Mesozoic 提升**：一名用户对资源表示感谢，特别是 *mesozoic* 资源，这极大地帮助了他们在 **codegen** 和 **UOp** 方面的工作。
   - 用户表示这些资源对他们的项目起到了关键作用，强调了所提供材料的价值和影响。
- **Kernel-Per-Level 性能讨论**：一名用户询问了关于在软件中创建 **kernel-per-level** 的性能比较。
   - 他们赞扬了该软件的工程设计及其通过不同 kernel 策略进行优化的潜力。
- **WebGPU Demo 的进展**：一名用户报告称对 **webgpu demo** 进行了性能改进，并附带了一段 [屏幕录制](https://cdn.discordapp.com/attachments/1068976834928193609/1370204057972773024/Screen_Recording_20250509_104232_Chrome3.mp4?ex=681f4e38&is=681dfcb8&hm=bbe19de310b1f6e6fd0ef5c0d7d6c3d7337ecfd3d4055cb7ff7e243d433f88b0&)。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1370443084822741116)** (1 条消息): 

> `Lambda, AgentX` 


- **Lambda 为 AgentX 重新开放资源**：Lambda 为 AgentX 竞赛的每位个人参赛者提供 **$100 的 Serverless API 推理额度 (Inference credits)**，申请表必须在 **太平洋时间 5/16 周五晚上 11:59** 前通过 [此链接](https://forms.gle/UtVhmPS3mitS8Vxu7) 完成。
- **Lambda 工作坊将于周四举行**：针对 AgentX 的 Lambda 工作坊将于 **太平洋时间周四 (5/15) 上午 10 点** 举行，你将学习如何使用 Lambda 的 Inference API 构建实用的 Agent 应用。
   - 在控制成本的同时优化 Agent 性能的技术、在生产环境中部署 Agent 的最佳实践，以及由 Lambda 基础设施驱动的高级 AI Agent 现场演示。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1370458985530130644)** (2 条消息): 

> `Certificate Timeline, AgentX judging, Coursework deadline` 


- **讨论证书发放时间线**：一名成员询问了完成作业和实验提交后领取证书的时间线。
   - 另一名成员解释说，**Trailblazer/Mastery/Honorary Tier** 的证书可能会在 6 月初发放，而 **Ninja/Legendary Tier** 的证书将在 **AgentX** 评审结束后于 8 月发放。
- **课程作业截止日期定为 5 月 31 日**：所有课程作业的最终截止日期是 **5 月 31 日**。
   - **AgentX** 的评审（针对 Ninja/Legendary Tier）将在整个 6 月进行，这意味着这些级别的证书发放将会延迟。