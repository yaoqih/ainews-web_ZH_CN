---
companies:
- openai
- ollama
- vllm
- openrouter
- anthropic
- google-deepmind
- langchain
- llamaindex
date: '2026-01-15T05:44:39.731046Z'
description: '**OpenAI** 发布了 **Open Responses** API 规范，这是一个开源、多供应商的互操作 LLM API 标准，旨在简化智能体堆栈（agent
  stacks）和工具链。**ollama** 和 **vLLM** 等早期采用者已支持该规范，而 **Anthropic** 和 **Google DeepMind**
  则明显缺席。


  来自 **Cursor** 的智能体设计见解强调，相比于巨型智能体（mega-agent）模型，明确的角色分工和规划更为重要；在长期运行中，**GPT-5.2**
  的表现优于 **Opus 4.5**。目前，智能体领域新兴的主流上下文/记忆抽象方式是“**文件系统即记忆**”（filesystem-as-memory），这一方案由
  **LlamaIndex** 和 **LangChain** 倡导，使用通常由 Postgres 等数据库支持的虚拟文件系统。


  此外，LangChain 还推出了一款名为 **openwork** 的开源桌面界面，用于智能体编排。这些新闻突显了 AI 开发在 API 标准化、智能体架构和记忆抽象方面的最新进展。'
id: MjAyNi0w
models:
- gpt-5.2
- opus-4.5
people:
- reach_vb
- simonw
- yuchenj_uw
- omarsar0
- jerryjliu0
- hwchase17
- swyx
title: Open Responses：OpenAI 响应 API 的明确规范，支持 OpenRouter、Ollama、Huggingface、vLLM 等平台。
topics:
- interoperable-apis
- agent-architecture
- filesystem-memory
- api-standardization
- multi-agent-systems
- prompt-engineering
- model-comparison
- virtual-filesystems
- open-source
- agent-ux
---

**Responses API 就够了。**

> 2026/1/14-2026/1/15 的 AI 新闻。我们为您检查了 12 个 subreddits、[**544** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord（**205** 个频道及 **5564** 条消息）。预计节省阅读时间（按 200wpm 计算）：**433 分钟**。**我们的新网站**现已上线，提供完整的元数据搜索，并以优美的 vibe coded 方式呈现过往所有内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 [@smol_ai](https://x.com/Smol_AI) 上向我们提供反馈！

标准化工作通常吃力不讨好，但做得好时是对社区的一种极佳承诺。今天，OpenAI 履行了对开源社区的责任，[明确记录了他们的 Responses API](https://www.openresponses.org/)，并与 [vLLM 和 ollama](https://x.com/reach_vb/status/2011863149356413275) 等伙伴合作支持该 API。然而更令人惊讶的是，API 标准化的市场领导者 [OpenRouter 也支持了它](https://x.com/OpenRouterAI/status/2011864089782599802)。发布合作伙伴中明显的缺席者：Anthropic 和 Deepmind。

---

# AI Twitter 回顾


**可互操作的 LLM API：“Open Responses” 围绕 Responses 形成新的基准**

- **Open Responses 规范（多供应商，对 Agent 友好）**：OpenAI DevRel 及其合作伙伴发布了 **Open Responses**，这是一个开源规范，旨在标准化跨供应商的 **类似 Responses-API** 的接口（“默认多供应商”，可扩展且不碎片化），这样 Agent 技术栈就不必针对每个模型/供应商进行分支处理。参见 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/2011862984595795974) 的公告以及 [@reach_vb](https://twitter.com/reach_vb/status/2011863149356413275) 提到的合作伙伴。与 Chat Completions 相比，这被定位为一个“白板 (clean slate)”：特殊情况更少，对于重度依赖 Tool 的工作流具有更好的一致性。
- **生态系统牵引力 + 实现**：工具构建者的初步评价是，这是缺失已久的“用于与模型对话的正式、标准化的 JSON API” ([@simonw](https://twitter.com/simonw/status/2011865205123531155))。**Ollama** 迅速宣布支持 ([@ollama](https://twitter.com/ollama/status/2011871283928317971))，而 **vLLM** 指出他们之前必须对供应商行为进行逆向工程，并期望该规范能简化原语和工具 ([@vllm_project](https://twitter.com/vllm_project/status/2012015593650536904))。

**Agent：规划 (Planning) > “多 Agent 氛围”，文件系统成为主流的上下文/记忆抽象**

- **Cursor 长期运行 Agent 的经验（角色 + 规划 + 评审）**：多篇文章总结了 Cursor 的观点，即类对等的自我协作（peer-like self-coordination）往往会失败；**明确的角色分工**（规划者/执行者/评审者）和强大的前期规划效果更好，同时强调 **Prompt/系统的稳定性**优于框架（harness）的复杂性。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2011863636042469866) 强调了运行现状：它是“数百个并发 Agent”而非单个超级 Agent。[@omarsar0](https://twitter.com/omarsar0/status/2011823468468379782) 将此与 Claude Code 的使用联系起来（子 Agent 管理各自的上下文；编排器保持高层级），并指出 Cursor 发现 **GPT-5.2** 在长达一周的运行任务中表现更强，而 **Opus 4.5** 则倾向于提前停止或走捷径。
- **文件系统即存储（Filesystem-as-memory）成为重心**：一系列推文趋向于认为“文件即一切（files are all you need）”是 Agent 上下文、存储和技能的核心。
  - LlamaIndex 的构想：将文件作为 (1) 持久化上下文存储，(2) 搜索接口（在动态遍历方面通常优于传统的 RAG 模式），以及 (3) 工具调用/技能的更简单底层 ([@jerryjliu0](https://twitter.com/jerryjliu0/status/2011849758944690625)；由 [@llama_index](https://twitter.com/llama_index/status/2011846444156645438) 转发）。
  - LangChain 的 Agent Builder 使用了**文件系统抽象**，并带有 **AGENTS.md**、**skills/** 和 **tools.json** 等约定，使 Agent 可以通过反馈更新存储并持久化行为 ([@LangChain](https://twitter.com/LangChain/status/2011864707439690031))。
  - 重要的实现细节：LangChain 的“文件系统”通常是一个**基于 Postgres 的虚拟文件系统包装器**，而非物理磁盘 ([@hwchase17](https://twitter.com/hwchase17/status/2011834318172422279)；澄清见 [@hwchase17](https://twitter.com/hwchase17/status/2011858266863911382))。
  - 同时也出现了务实的质疑：“每一个作为事实来源的文件系统最终都会演变成数据库” ([@swyx](https://twitter.com/swyx/status/2011984243430236608))。
- **发布 Agent UX + 开发框架**：
  - **openwork**：LangChain JS 发布了一个开源的 “Claude Cowork” 风格桌面界面（包含规划 + 文件系统 + 子 Agent 委派），基于 deepagentsjs 构建，可通过 `npx` 配合 Anthropic/OpenAI 模型运行 ([@LangChain_JS](https://twitter.com/LangChain_JS/status/2011863256223400360))。
  - **针对“Agent 进度”的真实 UI**：有观点批评大多数 Agent UI 使用加载动画（spinners）伪造进度；LangChain JS 展示了如何将流式工具调用事件传入 React，并配合 TypeScript 类型安全事件实现真实的进度报告 ([@LangChain_JS](https://twitter.com/LangChain_JS/status/2011833970204557694)；原贴由 [@bromann](https://twitter.com/bromann/status/2011833439834775738) 发布)。
  - **Dexter 3.0**：声称其基于事件的 Agent 循环 + 动态上下文管理将其“核心循环”缩减至约 100 行，同时提升了性能 ([@virattt](https://twitter.com/virattt/status/2011933907881492498))。

**模型与能力发布：快速图像模型、开源翻译、小型 LM 以及音频 S2S 推理**

- **Black Forest Labs FLUX.2 [klein]**：新的快速/小型图像生成/编辑系列：**4B (Apache 2.0)** 和 **9B (FLUX.2 非商业许可)**，以及一个新的文本编码器；定位为用于迭代/编辑的 <1s 生成 ([@bfl_ml](https://twitter.com/bfl_ml/status/2011825819082244266))。fal 在其市场上发布了它 ([@fal](https://twitter.com/fal/status/2011826361434771923))，Arena 将两者都添加到了 text-to-image 和 image-edit 竞技场中 ([@arena](https://twitter.com/arena/status/2011869067272208812))。评论指出，巨大的技术飞跃已经变得多么常态化（“比 Stable Diffusion 强约 10 倍，但体积几乎同样小”） ([@swyx](https://twitter.com/swyx/status/2011861139689513314))。
- **Google DeepMind TranslateGemma**：基于 **Gemma 3** 构建的开源翻译模型，使用 **Gemini 生成**的翻译数据训练，支持 **55 种语言**，发布了 **4B/12B/27B** 尺寸，针对设备端/低延迟翻译进行了优化 ([@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/2011848249850630363)；训练/蒸馏角度 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/2011848252451156244)；总结 [@_philschmid](https://twitter.com/_philschmid/status/2011848973074448657))。早期部署说明包括在移动端运行量化 4B 的 MLX Swift ([@sach1n](https://twitter.com/sach1n/status/2011975664573038824))。
- **Zilliz/Milvus 语义高亮模型**：发布了一个轻量级的 **0.6B** 模型 + 具有 **8192 context** 的数据集，采用 **MIT** 协议，并附带详细的训练博客 ([@mervenoyann](https://twitter.com/mervenoyann/status/2011732254591275022)；博客链接 [@mervenoyann](https://twitter.com/mervenoyann/status/2011732428784865391))。
- **TII Falcon-H1-Tiny 系列**：1 亿参数以下的 LLM，具有专门的变体（编码、function calling、多语言、推理），定位用于边缘/IoT 隐私部署 ([@yb2698](https://twitter.com/yb2698/status/2011805117016916056)；组织回顾 [@TIIuae](https://twitter.com/TIIuae/status/2012034581084430662))。
- **StepFun Step-Audio R1.1 (Realtime)**：Artificial Analysis 报告称，这款 **32B** 语音到语音“音频推理”模型以 **96.4%** 的得分领跑其 **Big Bench Audio**，TTFT 约为 **1.51s**，并提供了美元/小时和美元/token 的等效价格 ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2012006066339581958))。

**RL、评估和长 context 训练/推理：扩展实用的 Agent 训练**

- **Unsloth：RL 扩展至更长 context + vLLM 加速**：Unsloth 声称，通过 seqlen/hidden-state 分块 + 卸载的 log-softmax，RL 可以扩展到 **7 倍长的 context**，通过用于 vLLM 运行的 “Standby” + 平铺式 MLP 可扩展至 **12 倍**；包含一个例子 “gpt-oss QLoRA 在 1× B200 上达到 380K” ([@danielhanchen](https://twitter.com/danielhanchen/status/2011828515348627561))。vLLM 呼应了 7 倍长 context RL 的合作 ([@vllm_project](https://twitter.com/vllm_project/status/2011857612103630924))。
- **Agent 记忆 + context 修剪研究 (“Focus”)**：DAIR 重点介绍了一篇论文，该论文提出由 Agent 控制的整合检查点（`start_focus`/`complete_focus`），将学习内容总结到持久化知识块中并删除中间痕迹，在准确率不变的情况下（Claude Haiku 4.5），在 SWE-bench Lite 上实现了 **22.7% 的 token 减少** ([@dair_ai](https://twitter.com/dair_ai/status/2011806092737827206))。
- **对基准测试完整性的反击**：  
  - MMLU-Redux：针对 MMLU 主题/子集中发现的问题进行人工策划/无泄漏的修复 ([@PMinervini](https://twitter.com/PMinervini/status/2011782967723511868))。  
  - 一个具体的“数据集伪影”警告：MMLU-Pro 化学/物理子集据称存在偏差，即选项中的“前导空格”与正确性相关 ([@giffmana](https://twitter.com/giffmana/status/2011859715043836166))。  
  - Arena 自己的元分析指出，“AI 竞赛的领导地位”在很大程度上取决于 Prompt 阶层：OpenAI 在大部分时间里总体领先，但 Anthropic 在 “Expert prompts” 上领先的频率更高 ([@arena](https://twitter.com/arena/status/2011849440160858443))。

**基础设施与开发者工具：实时推理、浏览器内搜索、向量数据库实验以及 Agent 赋能的 IDE 工作流**

- **Together AI + Blackwell 上的 Cursor 推理栈**：Together 描述了针对实时编程 Agent 推理（紧凑的编辑器延迟循环）的工程设计，引用了在 **GB200/B200** 上的可靠性、自定义 tensor-core kernels、FP4 量化以及 NVL72 mesh 并行处理 ([@togethercompute](https://twitter.com/togethercompute/status/2011875191828488598)；技术要点 [@togethercompute](https://twitter.com/togethercompute/status/2011875193476829631))。来自 [@realDanFu](https://twitter.com/realDanFu/status/2011876049215520919) 的相关笔记提到了“技术金句”，甚至包括硬件维护细节，如更换 NVLink 线缆以确保稳定性。
- **浏览器内重构的 VS Code 文档搜索**：VS Code 报告其网站搜索速度大幅提升；工程报告描述了 **docfind**，它通过 **WebAssembly** 完全在浏览器中运行 ([@code](https://twitter.com/code/status/2011827481175605487))。
- **RAG 实验作为一等基础设施**：Qdrant + Tigris Data 的 “RAG Lab” 强调了分块策略（chunking strategies）的可重现 A/B 测试，通过 fork 数据集并将每个数据集与其自身的 vector index 配对，以进行公平的评估 ([@qdrant_engine](https://twitter.com/qdrant_engine/status/2011679747244167175))。
- **Copilot CLI + Agent SDK 覆盖面**：GitHub Copilot CLI/Coding Agent 增加了 “automated memory” ([@_Evan_Boyle](https://twitter.com/_Evan_Boyle/status/2011932670096523326))，并有关于 Copilot CLI SDK 的讨论，该 SDK 允许在 Copilot auth 之上构建自定义 CLI（例如：视频宣传生成器） ([@burkeholland](https://twitter.com/burkeholland/status/2011934322413224152))。
- **OpenCode + Copilot 订阅**：OpenCode 表示可以通过 “$39 pro+” 级别配合 Copilot 订阅使用，该级别提供 “best coding models”，突显了对工具链互操作性（interoperability）日益增长的需求 ([@opencode](https://twitter.com/opencode/status/2011790750543983072))。


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. 本地 LLM 性能与偏好

  - **[目前最强大的“破解版”本地 LLM 是哪个？](https://www.reddit.com/r/LocalLLM/comments/1qdx4n8/whats_the_most_cracked_local_llm_right_now/)** (热度: 44): **该帖子询问在没有 VRAM 限制的情况下，哪种本地 LLM (Large Language Models) 在推理、指令遵循、编码和处理长提示词方面表现最出色。讨论重点提到了 **Qwen3-Coder-480B**、**GLM 4.7** 和 **Kimi-K2**。值得注意的是，**Kimi-K2** 因其抗 sycophancy（阿谀奉承）能力和独特的 'interleaved thinking'（交错思考）能力而受到称赞，使其能够执行包含数百个 tool calls 的复杂任务，在本地和开源 LLM 领域脱颖而出。** 评论者认为 'cracked' 一词可能被误用，因为本地 LLM 的性能通常取决于硬件。**Gemma 模型**的综合性能也被提及，而 **Kimi-K2** 则因其广泛的 interleaved thinking 能力而备受关注，使其区别于 **GLM-4.7** 等其他模型。

    - AbheekG 强调了 Kimi-K2 模型，指出其作为 1 万亿参数 LLM 的独特能力，这需要大量的计算资源。该模型以抗 sycophancy 能力和先进的 'interleaved thinking' 训练而著称，允许其为复杂任务执行数百个 tool calls。这使其在推理能力方面处于本地 LLM 的领先地位，甚至在这一方面超过了 GLM-4.7。

  - **[除了那些令人厌恶的大公司外，最好的编码 AI 是哪个？（本地或在线）](https://www.reddit.com/r/LocalLLM/comments/1qdnao5/best_ai_for_coding_that_isnt_from_the_major/)** (热度: 44): **该帖子寻求不隶属于 **OpenAI** 或 **Microsoft** 等大公司的开源编码 AI 工具建议。建议包括来自法国初创公司 **Mistral** 的 **Devstral-small-2**，以及 **Qwen3**、**Minimax**、**GLM** 和 **Kimi K2** 等其他模型。用户强调 **Minimax** 和 **GLM** 在 Python 和 Dart 等语言的编码任务中特别有效，**GLM 4.7** 获得了特别称赞。** 用户更倾向于使用 **Minimax** 和 **GLM** 来处理编码任务，而 **Kimi K2** 虽然受到关注，但由于硬件要求尚未得到广泛测试。

- 来自法国初创公司 Mistral 的 **Devstral-small-2** 被提及为一个值得关注的代码模型，暗示其具有作为大型公司产品替代方案的潜力。
- 一位用户将 **Qwen3**、**Minimax**、**GLM** 和 **Kimi K2** 列为顶级代码模型，并特别提到 Qwen3 和 GLM 在 Python、Dart 和 Cloud stack 任务中表现出色。该用户指出 Kimi K2 需要大量的硬件扩展，目前他们尚未进行尝试。
- 关于在本地运行 **GLM 4.7** 等模型的硬件要求的详细讨论强调了对显存（VRAM）的大量需求，对于高质量的 30B 模型，理想情况下需要 48GB+。该用户还提到正在尝试使用 **Nemotron 3 Nano 30B** 和 **Qwen 3 Coder 30B Instruct** 来构建多 Agent 栈（multi-agent stack），并强调了微调（fine-tuning）和 LoRAs 对特定任务的重要性。

- **[我想订阅一个 LLM。哪一个最适合口语练习/提高写作和学习编程？我每月最高可以支付 10-12 美元。](https://www.reddit.com/r/LocalLLM/comments/1qdmllw/i_want_to_subscribe_to_an_llm_which_one_is_best/)** (活跃度: 34): **在每月 `10-12 USD` 的预算下，由于其性价比和效果，**GLM 4.7** 被推荐用于编程任务。然而，如果预算允许，建议选择 **OpenAI** 或 **Claude** 的基础层级，因为它们可能提供更优越的性能。讨论强调，虽然 LLM 可以辅助学习，但它们不应取代教科书和自我练习等传统学习方法。推荐使用 Anna's Archive 和 OpenStax 等免费资源进行全面学习。** 评论表明，相比于完全依赖 LLM，人们更倾向于传统学习方法。建议尽量少地使用 LLM，主要将其用于澄清疑问和理解概念，而不是作为主要的学习工具。

    - g33khub 认为 GLM 4.7 是编程任务中一个极具性价比的选择，但建议如果预算允许，可以考虑 OpenAI 或 Claude 的基础层级。这意味着虽然 GLM 4.7 价格亲民，但 OpenAI 和 Claude 在语言学习和编程方面可能会提供更优秀的性能或功能。
    - Quirky-Craft-3619 强调，虽然 LLM 很有帮助，但不应将其作为学习编程的主要工具。相反，他们建议使用教科书并独立练习编程，利用 GPT 或 Gemini 等 LLM 来澄清概念，而不是直接寻求代码编写的帮助。这种方法鼓励更深层次的理解和编程中的自力更生。
    - ElectronSpiderwort 提到 Openrouter 是一个灵活的选择，提供免费层级和 10 美元的方案，允许访问主流模型。这表明 Openrouter 可能是一种体验不同 LLM 的高性价比方式，如果使用更便宜的模型，可能会在更长时间内节省预算。

### 2. GPU 市场变化及影响

  - **[我低估了 /r/LocalLLaMA 对 VRAM 渴求的故事](https://www.reddit.com/r/LocalLLaMA/comments/1qe2i88/my_story_of_underestimating_rlocalllamas_thirst/)** (活跃度: 290): **这张图片是一个迷因（meme），描绘了在 Reddit 上分享一款高性能显卡（具体为 w6800 32GB）优惠信息所带来的意外后果。最初该卡售价为 500 美元，但在帖子发布后价格飙升至 1,000 美元以上，凸显了社区对 VRAM 的巨大需求。这反映了技术社区中的一个更广泛趋势，即分享有价值硬件的信息可能导致市场快速变化，类似于“淘金热”效应。评论建议根据每插槽 VRAM 容量（VRAM-per-slot）和冷却要求等具体需求，考虑 3090 或 R9700 等替代显卡选项。** 一位评论者将此比作加利福尼亚淘金热，建议在分享价值信息之前先进行战略性采购。另一位评论者建议根据当前市场价格和具体技术需求，考虑 3090 或 R9700 等替代显卡。

    - EmPips 讨论了不同 GPU 选项之间的权衡，认为虽然所讨论的显卡令人印象深刻，但根据具体需求，`3090` 或 `R9700` 等替代方案可能更具性价比。他们强调了对每插槽 VRAM (VRAM-per-slot) 和冷却解决方案的考虑，并指出如果能够处理高待机功耗和外部冷却，`mi50x` 显卡可能是一个可行的选择。

  - **[RTX 5070 Ti 和 RTX 5060 Ti 16 GB 不再生产](https://www.reddit.com/r/LocalLLaMA/comments/1qdh28f/rtx_5070_ti_and_rtx_5060_ti_16_gb_no_longer/)** (活跃度: 381): **由于内存供应短缺，Nvidia 已停止生产 `RTX 5070 Ti` 并大幅削减了 `RTX 5060 Ti 16 GB` 的供应，导致 5070 Ti 的价格比 MSRP 上涨了约 `100 美元`。RTX 5060 Ti 的 8 GB 配置不受影响。这一决定影响了大多数 AIB，他们将不再制造这些 GPU。[来源](https://m.youtube.com/watch?v=yteN21aJEvE)。** 一位用户指出 RTX 5060 Ti 16 GB 是为系统增加 Nvidia 显存的性价之选，强调其适用于 DLSS、AI 处理和推理任务，尤其是配置 `64GB VRAM` 来运行 `70B 模型`。另一位用户对停产影响其升级计划表示失望，而第三条评论则批评了 Nvidia 的商业行为。

    - phido3000 讨论了 RTX 5060 Ti 的价值主张，强调了其在 AI 任务中的可负担性和性能。在 390 美元的价位，它提供 16GB 的 GDDR7 显存，有利于 DLSS 和 AI 处理。该卡的 128-bit 总线由于快速的 GDDR7 而得到缓解，使其性能可与 192-bit GDDR6 显卡相媲美。它适用于 LLAMA 等模型的推理，特别是当需要 64GB VRAM 时，可以作为预算配置中 3090 的可行替代方案。
    - phido3000 还指出了在单个系统中使用多个 RTX 5060 Ti 显卡的实用性。凭借低功耗要求和双插槽散热器设计，在标准电源的机器中安装四张或更多显卡是可行的。这种配置支持新的 Quantization 方法，并能有效处理 70B 模型，使其成为小型 AI 推理任务的经济高效解决方案。
    - Otherwise_Local_7743 对 RTX 5070 Ti 的停产表示失望，因为这是他们为 homelab 计划的升级。他们提到在价格稳定之前将依靠 RTX 3080 进行推理任务，这表明了 5070 Ti 在此类环境中潜在性能提升的吸引力。

### 3. Mac Studio M3 Ultra vs DGX Spark 性能对比

  - **[Mac Studio M3 Ultra Stats](https://www.reddit.com/r/LocalLLM/comments/1qdqi4i/mac_studio_m3_ultra_stats/)** (活跃度: 42): **该帖子对比了 **Mac Studio M3 Ultra** 与 **DGX Spark** 的性能，强调虽然 DGX Spark 在 Prompt 处理方面表现出色，但在 Token 生成速度上稍逊一筹，而这对于文本生成任务至关重要。报告提供了多种模型的详细 Benchmark，指出 **Qwen3-Next-80B-A3B-Instruct** 模型表现优于其他模型，在 100k Context 大小下，Prompt 处理速度达到 `1,584.5 tok/s`，Token 生成速度为 `52.3 tok/s`。**MiniMax-M2.1-4bit** 模型也表现强劲，平均 Prompt 处理速度为 `886.1 tok/s`。DGX Spark 被认为以通用性见长而非速度，更适合研发环境而非高速文本生成。**

    - DGX Spark 的设计初衷是作为研究实验室和开发团队的多功能 AI 机器，而非在特定任务中拔尖。它作为一个用于原型设计、Fine-tuning 和推理的“小型数据中心”运行，但在任何特定领域都没有针对速度进行优化，这与 Mac Studio 专注于特定任务性能形成对比。
    - Context 处理速度对于处理大型 Tool Call Dump 和 Agent 编程至关重要，一位评论者认为这是 Mac 的局限性。这突显了在处理大量数据处理任务时的一个关键可用性问题，即需要更快的 Context 处理能力。
    - Benchmark 的一种实用方法是在 AWS 上配置带有 RAM 和 GPU 的实例，使用脚本部署 Graviton (ARM CPU) 资源。这种方法允许用户模拟 M3 等系统的性能，为运行 Benchmark 和测试模型提供灵活的环境，类似于 NVIDIA DGX Cloud 的服务。

  - **[Oh Dear](https://www.reddit.com/r/LocalLLM/comments/1qdiwdh/oh_dear/)** (活跃度: 73): **图片描绘了一个 AI 聊天模型的故障，它在循环中不断输出单词 'the'，表明模型配置或 Prompt 处理可能存在问题。这可能是由于 System Prompt 设置不当，或 Temperature（控制模型输出随机性的调优参数）设置有误。评论建议检查这些参数，并可能使用像 'pocket pal' 这样的替代工具来更好地处理模型文件（如 GGUF 文件），这可能会提供更好的性能或兼容性。**

    - mp3m4k3r 建议检查调优参数，特别是 Temperature 设置，以确保其符合模型的推荐值。这对于维持模型性能和防止重复输出等问题至关重要。
    - HealthyCommunicat 建议调整 Repeat Penalty，从 1.1 开始并根据需要增加，以防止模型生成重复文本。此外，他们建议确保模型使用的 Expert 数量不超过推荐值，因为这些是 Local LLM 产生此类错误的常见原因。
    - ScoreUnique 提到使用 'pocket pal' 加载 GGUF 文件，这可能是处理 Local LLM 语境下特定文件类型或格式的一种解决方案。



## 较低技术含量的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 模型与 Benchmark 发布

  - **[Grok 4.20 (beta version) found a new Bellman function](https://www.reddit.com/r/singularity/comments/1qdntt3/grok_420_beta_version_found_a_new_bellman_function/)** (活跃度: 90): **据 [推文](https://x.com/PI010101/status/2011560477688463573?s=20) 宣布，**Grok 4.20 (beta version)** 据报道发现了一个新的 Bellman 函数。当 `p` 趋近于 `0` 时，该 Bellman 函数由表达式 `l(p) ~ p\sqrt{log(1/p)}` 描述。这一发现正被拿来与 **Gemini 3 Pro** 和 **GPT-5.2** 的结果进行对比，据称这两个模型也产生了相同的结果，这表明该发现可能并非 Grok 4.20 所独有。** 一位评论者认为这一发现被过度炒作，指出使用 Gemini 3 Pro 和 GPT-5.2 等其他模型也得到了类似的结果，暗示 Grok 4.20 发现的新颖性或重要性可能被夸大了。

- ThunderBeanage 强调了 Grok 4.20 与 Gemini 3 Pro 和 GPT-5.2 等其他模型的对比，指出它们在同一个问题上产生了相同的结果。这表明 Grok 的表现可能并不像宣称的那样具有开创性，至少在这个特定背景下是如此。
- FlamaVadim 提供了一个与 Bellman 函数相关的数学表达式，即当 `p` 趋于零时，`l(p) ~ p\sqrt{log(1/p)}`。这一见解对于研究该函数的渐近行为的人来说可能很有价值，指出了进一步数学探索的潜在兴趣领域。
- Singularity-42 质疑了 Grok 与 Gemini、GPT 和 Claude 等其他领先模型相比的相关性，暗示需要进行详细的性能对比，以了解 Grok 在当前 AI 领域的地位。

- **[[D] 新 arXiv 综述：“High-Performance Serverless” 是 AI Inference 的未来（Static Clusters 正在消亡）](https://www.reddit.com/r/MachineLearning/comments/1qdmbk2/d_new_arxiv_review_highperformance_serverless_is/)** (Activity: 6): **该帖子讨论了 arXiv (arXiv:2601.09334) 上的一篇系统综述，内容关于从静态 GPU 集群向用于 AI Inference 的 Serverless 模型转变。论文认为，由于现代 AI 工作负载的 “bursty” 特性，静态分配在处理这些负载时效率低下，会导致过度配置（over-provisioning）或配置不足（under-provisioning）。它建议 Serverless、弹性执行模型是解决这些低效问题的未来方向。帖子还提到了一个实际实现，即构建了一个引擎通过 state snapshotting 解决 Cold Start 问题，这与论文的发现一致。[阅读论文](https://arxiv.org/abs/2601.09334)。** 一条最高赞评论指出该文章并不存在，暗示帖子中可能存在错误或虚假信息。

- **[新发布的 GLM-Image 证明了开源 AI 开发者不再需要 Nvidia 和 CUDA。](https://www.reddit.com/r/DeepSeek/comments/1qdio2d/newly_released_glmimage_is_a_proof_of_concept/)** (Activity: 196): ****Zhipu** 开源了 **GLM-Image**，证明了无需依赖 **Nvidia** 芯片和 **CUDA** 也可以开发出具有竞争力的开源 AI 模型。该模型使用 **Huawei Ascend 910B** 芯片和 **MindSpore** 框架训练。虽然 Ascend 芯片的效率仅为 Nvidia 的 `80%`，但其更低的成本（`$12-13,000`，而 Nvidia 为 `$30-40,000`）和更低的功耗使其成为一种具有成本效益的替代方案。拥有 `9 billion parameters` 的 GLM-Image 支持在消费级硬件上进行高速推理，可能降低开源 AI 开发的门槛。** 评论者强调了中国对 **SMIC** 等半导体公司的投资潜力，以进一步提升开源 AI 能力。还有一种观点认为，人们尚未广泛意识到无需 Nvidia 和 CUDA 也能开发 AI，这预示着 AI 硬件格局的转变。

    - GLM-Image 的发布表明，开源 AI 开发者现在可以在不依赖 Nvidia 和 CUDA 的情况下运作，鉴于这些技术在 AI 开发中的主导地位，这一点意义重大。这种转变表明，可以使用替代硬件和软件解决方案开发 AI 模型，从而可能降低成本，并提高那些无法获得 Nvidia 专有技术的开发者的可及性。
    - 讨论强调了像 SMIC 这样的中国半导体公司在这一转型中发挥关键作用的潜力。随着 SMIC 的 5nm 节点实现 50% 的良率等进展，这些公司有可能提供 Nvidia 硬件的竞争性替代方案，这可能是 AI 硬件市场的一个游戏规则改变者。
    - Suitable-Program-181 的评论强调了认识到 AI 开发可以在减少对传统硬件巨头依赖的情况下取得进展的重要性。这种认识可能会带来 AI 硬件领域更多的创新和竞争，因为开发者正在探索 Nvidia 和 CUDA 之外的新可能性。

- **[FLUX.2 [klein] 4B & 9B 发布](https://www.reddit.com/r/StableDiffusion/comments/1qdmohb/flux2_klein_4b_9b_released/)** (Activity: 788): **FLUX.2 Klein** 模型由 **Black Forest Labs** 开发，已发布两个新版本：`4B` 和 `9B` 模型。`4B` 模型利用 **Qwen3B**，在 `6000 Pro` 上仅需 `4 steps` 即可在 `1.3 秒` 内完成处理；而 `9B` 模型使用 **Qwen 8B**，耗时 `2.2 秒` 并提供略好一些的性能。这两个模型都可以在 [Hugging Face](https://huggingface.co/black-forest-labs) 上获取，并支持 **Comfy Default Workflow**。值得注意的是，`4B` 版本采用 **Apache-2 licensed**，这对开源使用具有重要意义。`9B` 模型被描述为“全量基础模型（full-capacity foundation model）”，非常适合微调和自定义 pipeline，与蒸馏版本相比，其输出多样性更高。评论者强调了同时发布基础版和蒸馏版的重要性，这在 **FLUX** 和 **BFL** 历史中尚属首次。Apache-2 许可证模型的可用性被视为一项重大优势，人们对进一步的发展充满期待，例如 **Alibaba's z-image base model** 的发布。

    - FLUX.2 Klein 模型的发布包含了基础版和蒸馏版，这在 FLUX 和 BFL 系列中是首次。值得注意的是，4B 版本采用 Apache-2 许可证，允许更广泛的使用和修改。此版本支持编辑功能，增强了其在各种应用中的实用性。
    - Klein 9B 基础模型被描述为未经蒸馏的“全量基础模型”，保留了完整的训练信号。这使其成为 LoRA 训练和追求控制力而非速度的自定义 pipeline 的理想选择。该模型比其蒸馏版本具有更高的输出多样性，适用于研发目的。
    - Comfy-Org 已经集成了对 Klein 模型的支持，在 Hugging Face 上提供了适用于 4B 和 9B 版本的 text encoders。此次集成包括 ComfyUI 仓库中一个已合并的 pull request，表明社区正在积极开发并支持这些模型。此外，GGUF text encoders 已经可以使用，扩展了这些模型的兼容性和实用性。

  - **[AI 证明了代数几何中的一个新定理。美国数学学会主席表示该证明“严谨、正确且优雅”。](https://www.reddit.com/r/OpenAI/comments/1qdmoc3/ai_proved_a_novel_theorem_in_algebraic_geometry/)** (Activity: 104): **该图片是 Adam Brown 讨论一篇新论文的推文，该论文使用名为 Gemini 的 AI 证明了代数几何中的一个新定理，该 AI 是与 Google DeepMind 及多位教授合作开发的。美国数学学会（American Mathematical Society）主席 Ravi Vakil 称赞该证明“严谨、正确且优雅”。论文描述了一个人机交互迭代过程，AI 提供了特例的解决方案，但人类数学家必须对这些方案进行泛化，以提出针对一般情况的证明策略。随后，AI 被再次提示以生成完整的证明，最终解决了原始猜想。这种合作凸显了 AI 在数学研究中的潜力，尽管它需要大量的人力指导。** 一些评论者认为人类完成了大部分工作，暗示 AI 的作用被夸大了，并认为该项目是 Google 的一种营销手段。其他人则讽刺地否定了 AI 的能力，反映出对 AI 在严肃学术工作中潜力的怀疑。

    - 论文描述了 AI 系统与人类数学家之间的协作过程，其中 AI 提供了特定案例的解决方案，但在将这些方案泛化到整个问题时遇到了困难。人类的分析在识别关键中间陈述方面至关重要，这为一般情况的证明策略提供了信息。这一迭代过程涉及用新问题重新提示 AI，从而生成了新问题的完整证明，并最终解决了原始猜想。
    - 研究中使用的 AI 尚未公开，且 Google 在研究中发挥了重要作用。这表明所涉及的模型可能是私有的，且可能比公开可用的 AI 系统更先进。像 Google 这样的大型科技公司的参与表明，在推动 AI 数学研究能力边界方面存在重大投资。
    - 评论对 AI 的作用表示怀疑，暗示这一成就可能更多是营销手段而非真正的突破。AI 的贡献被认为是有限的，关键工作主要由人类完成，突显了目前 AI 在独立解决复杂数学问题方面的局限性。

### 2. AI 在认知与统计学习中的应用

  - **[我们不再使用“总结一下”。我们改用“降噪”提示词在 2 分钟内阅读 50 页的报告。](https://www.reddit.com/r/GeminiAI/comments/1qdfznb/we_stopped_using_summarize_this_we_reply_with_the/)** (热度: 508): **该帖子讨论了从使用 AI “总结”文本转向使用“降噪（Noise Cancellation）”提示词来处理长文档。这种被称为“减法处理（Subtractive Processing）”的方法涉及一种“编辑审核（Redaction Audit）”，即 AI 会高亮包含硬数据、日期或可操作指令的句子，同时标记那些包含轶事、形容词或废话的内容。这种方法旨在不重写的情况下减少 `70%` 的文本量，从而避免 AI 幻觉（hallucinations）并保留作者的原话。** 一位评论者建议使用“金字塔原理（Minto Pyramid Principle）”进行文本处理，这是商业顾问青睐的一种简洁沟通方法。另一个疑问对“DISTINGUE”一词表示困惑，表明需要进一步澄清。

    - **Necessary_Coyote_571** 建议使用金字塔原理进行总结，这是商业顾问在高管沟通中青睐的技术。该方法采用自上而下的方式组织信息，先从核心观点开始，随后提供支持论据，这对于将复杂报告提炼成简洁摘要特别有效。
    - **WrongRain6117** 指出了 AI 生成摘要的一个常见问题，即它们往往看起来像是原文本的“同人小说”版本。这一评论强调了其他摘要技术（如金字塔原理）的潜在价值，这些技术通过关注核心信息和内容的逻辑流，可以提供更准确、结构更清晰的摘要。


  - **[OpenAI 重新录用了 3 名原研究人员，包括 Thinking Machines 的首席技术官兼联合创始人](https://www.reddit.com/r/OpenAI/comments/1qdehxx/openai_rejoined_3_former_researchers_including_a/)** (热度: 141): ****OpenAI** 重新聘请了三名原研究人员，其中包括 **Thinking Machines** 的前首席技术官兼联合创始人，此消息已通过 [X](https://x.com) 上的官方声明得到确认。这一举动突显了 AI 行业内动态的人才流动，像 OpenAI 这样的公司利用雄厚的资源来吸引和留住顶尖人才。** 一位评论者注意到了 AI 领域剧烈的人才更迭，认为 OpenAI 的财务能力是重新雇佣的关键因素。另一位评论者则对 Thinking Machines 即将发布的 LLM 模型可能受到的影响表示担忧。

    - **Informal-Fig-7116** 提到 **Thinking Machines** 据称将在今年发布他们自己的 LLM 模型，且该模型并非“Tinker”模型。这引发了关于核心人员回归 OpenAI 将如何影响这一新模型的开发和发布的疑问。这意味着人才转移可能会影响 AI 模型开发的竞争格局。
    - **LuckEcstatic9842** 强调了 AI 行业快速的人才流失，指出 **OpenAI** 拥有快速重新聘请前员工的财务资源。这表明在快速发展的 AI 领域，人才获取和留存方面具有竞争优势。


### 3. Claude 订阅与使用问题

  - **[正在慎重考虑购买第二个 Claude Code 订阅](https://www.reddit.com/r/ClaudeCode/comments/1qdspwr/highly_considering_getting_a_second_claude_code/)** (热度: 91): **由于目前的 `200美元/月` 方案频繁达到使用限制，该用户正考虑购买第二个 **Claude Code Max** 订阅。他们提到在被锁定三天后，直接支付的 API 费用大约为 `400美元`。用户尝试通过战略性地使用 **Haiku/Sonnet** 来优化使用量，但仍然遇到限制。评论中的一个建议是使用 **Opus** 而不是 Sonnet，因为它的 Token 效率更高，并在 `Claude.md` 中进行配置，通过 Haiku 子智能体（subagents）进行编码，并由 Opus 进行里程碑审查。这种方法被认为更具成本效益，尤其是因为“Opus 的读取比写入便宜得多”。** 一条评论质疑了正在构建的项目性质，暗示它们可能是资源密集型的。另一条建议是忽略 Sonnet，转而使用 Opus 以获得更好的效率和成本管理。

- dimonchoo 指出 Opus 模型相比 Sonnet 在 token 效率上更高，建议用户优先选择 Opus 以获得更好性能。这种效率对于优化资源使用至关重要，尤其是在处理大规模项目或高强度的编程任务时。
- Crinkez 建议使用 Opus 而非 Sonnet，并推荐在 `Claude.md` 中进行配置，通过部署 Haiku subagents 来编写代码。这种方法允许使用 Opus 进行里程碑审查，这可能更具成本效益，因为“Opus 的读操作比写操作便宜得多”，这表明了一种有效管理成本的资源利用策略。
- dkshadowhd2 讨论了在 20x 方案中达到使用限制（usage caps）的挑战，指出这类限制通常只有在运行大规模并行进程（如 'ralph loops'）时才会触发。这表明工作流涉及显著的并行化和 subagents 编排，可能导致高资源消耗。

- **[Prompting claude when it makes mistakes](https://www.reddit.com/r/singularity/comments/1qdhbfs/prompting_claude_when_it_makes_mistakes/)** (Activity: 297): **该帖子讨论了用户与 AI 语言模型 Claude 的交互，特别关注用户在模型出错时如何进行 Prompt 引导。讨论强调了一种相比于其他模型（如 Gemini）更倾向于将 Claude “人格化”的趋势。用户描述在 Claude 出错时会采取更耐心和鼓励的方式，而对 Gemini 则表现出更多的挫败感。这反映了一种细微的用户体验差异，即 AI 被感知的性格会影响交互风格。** 评论表明了对 Claude 的主观偏好，用户对其错误表现出更具共情心和耐心的态度，这表明人们可能认为 Claude 相比其他模型更具“人情味”。

- **[What's wrong with chat gpt 5.2 ? It's constantly arguing with me man I hate it](https://www.reddit.com/r/OpenAI/comments/1qdp3uz/whats_wrong_with_chat_gpt_52_its_constantly/)** (Activity: 432): **该帖子表达了对 ChatGPT 5.2 的不满，认为它经常与用户争论。这可能预示着模型交互风格的转变，可能是由于其对话算法或强化学习策略的更新，旨在提高事实准确性并减少误导信息。用户请求“换回 4o”可能指的是对先前版本（可能是 ChatGPT 4.0）的偏好，用户可能觉得那个版本更顺从或更少对抗性。** 评论反映了关于 AI 应该过于顺从还是过于挑剔之间平衡的争论。一些用户认为 AI 的好辩本性可能是对用户错误的回应，暗示 AI 的纠正是合理的。

    - honorspren000 描述了 ChatGPT 5.2 的一个问题，即该模型在创意写作选择上过于教条，具体表现为反对在一个历史时期背景下设定拥有魔法角色的故事，理由是潜在的历史不准确性。这突显了模型在处理创意语境时的潜在缺陷，在这些语境中，灵活性和用户意图应优先于对事实准确性的严格遵守。

- **[Can we ban the "Claude is so expensive" posts?](https://www.reddit.com/r/ClaudeCode/comments/1qe00kc/can_we_ban_the_claude_is_so_expensive_posts/)** (Activity: 243): **该帖子讨论了关于 Claude（一种语言模型服务）使用成本的反复投诉，强调该服务物有所值。作者认为，期望以微不足道的费用获得无限使用权是不合理的，尤其是考虑到该产品的革命性本质。帖子建议用户要么为服务付费，要么学会自己编程。** 评论者普遍认为像 LLM 这样的 Claude 成本是合理的，一些人指出这些工具目前定价过低，随着依赖程度的增加，可能会变得更贵。其他人则强调，即使以目前的价格，这些服务也比雇用开发人员更具成本效益，尤其是对于初创公司而言。

- **el_duderino_50** 强调了像 Claude 这样的 LLM 对初创公司的成本效益，指出即使是像 '20x MAX' 这样的高端方案，其成本也比雇佣开发人员更低。这凸显了在预算受限的早期初创公司中，LLM 的价值主张。
- **Substantial_Ear_1131** 比较了 Claude 和 Codex 的使用限制和性能，指出他们在一小时内就达到了 Claude 的限制，而带有 GPT 5.2 xhigh thinking 的 Codex 在相同价位下提供了显著更好的性能（“好 1000 倍”）。这表明虽然 Claude 可能具有成本效益，但其限制可能成为密集型用户的瓶颈。
- **Swimming_Leopard_148** 认为当前 LLM 的定价模式是不可持续的，暗示这些工具被“人为压低了价格”。他们预测，随着对这些工具的依赖度增加，成本将变得更加沉重，并将目前每月 20 美元的费用比作几杯外卖咖啡，暗示随着需求的稳固，价格可能会上涨。


---

# AI Discord 简报

> 由 gpt-5.2 生成的摘要之摘要


**1. Agentic 编码工具与编排**

- **Sonnet 4.5 让 Cursor 的后台 Agent 投入工作**：在 Cursor 社区，用户确认 **Sonnet 4.5** 现在可以作为启动 **background agent** 的可选模型。一位开发者在 X 上表示，他们正致力于在未来几天/几周内“大幅”改进 background agent。
  - 用户还研究了 Cursor 的 Agent 堆栈——**sub-agents** 包括内置的（**Explore** 和 **generalPurpose**），但每个 sub-agent 的模型选择听起来不太可靠，这促使人们寻找 Token 最小化工作流和像 [Nia mcp](https://nia.mcp) 这样的工具。

- **Composer-2 扮演推理模型**：Cursor 用户注意到手动添加 **composer-2** 会导致 Cursor 将其视为 **reasoning model**，他们将这种行为与 Cursor 官方推特暗示 composer-2 的推理支持即将到来的推文联系起来。
  - 该话题转向了实际的运维讨论：在探索代码库时如何降低 Token 使用量，以及 Agent 编排如何改变长时自主运行中“最佳模型”的权衡。

- **MCP 服务器在无状态与会话之间权衡**：在 MCP 贡献者社区，一项关于**签名方法（signature method）**的提案旨在调和 Schema 冻结与动态服务器功能，以便**无状态 MCP 服务器**能更廉价地服务于多个并发对话（[PR #2091](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2091)）。
  - 讨论围绕**动态工具集**（例如通过 [issue #1442](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1442) 在 STDIO 上实现 GitHub MCP）和**持久会话**（以在 Agent/MCP 重启后幸存）展开，人们注意到了“每个对话一个服务器”设计带来的实际扩展痛苦。


**2. 训练/推理性能：VRAM、长上下文和本地堆栈**

- **Unsloth RL 跃升至 700 万 Token（没错，是百万）**：Unsloth 宣布发布支持 **700 万 Token 上下文窗口**的 **long-context RL** 版本，称其比上一版本实现了 **7 倍**的飞跃（[UnslothAI 帖子](https://x.com/UnslothAI/status/2011827592886960131)）。
  - 成员们关注其实际优势——潜在的**内存节省**和新的长时运行 Agent 使用案例——同时交流训练参数（例如 GRPO 稳定性思路如 `importance_sampling_level="sequence"`）和以 VRAM 为中心的性能技巧。

- **VRAM 是新的马力（且 `-ot` 胜出）**：在 Unsloth 社区，成员们重申更多的 **VRAM** 通常能加速执行，强调了 Unsloth 的 `-ot` 标志可实现更好的 Tensor 放置，并报告其吞吐量通常优于 `n-cpu-moe`。
  - 在本地模型聊天中，共识保持一致：首先选择**能将模型放入 VRAM** 的硬件（例如 LM Studio 用户推荐 **20GB RTX 3080** 而非 **11GB 3080 Ti**），因为数据交换/硬件限制会迅速主导延迟。

- **本地推理持续抢占云端市场**：Latent Space 引用了 Charles Frye 的 Modal 文章，显示**本地 LLM 推理**可以匹配或超过主流 API 的成本/性能（[Modal 指南和代码示例](https://xcancel.com/charles_irl/status/2011484220032762114?s=46)），引发了关于会议转录的“我们可以做本地版的 Granola 吗？”之类的问题。
  - 随后的“本地转录堆栈”清单包括 [whisperX](https://github.com/m-bain/whisperX)、[NVIDIA NeMo](https://github.com/NVIDIA-NeMo/NeMo) 以及较旧的 [AutoDiarize](https://github.com/Alignment-Lab-AI/AutoDiarize)，一位用户声称使用优化的 Parakeet V3 设置，macOS 本地角色分离/转录运行速度与云端一样快。


**3. 模型发布、变体以及价格/使用信号**

- **Hawk Ultra 炒作 17k 行代码及开源承诺**：在 LMArena 中，用户将 **Hawk Ultra** 与前沿代码模型进行了对比，并声称它能在单个 prompt 中生成 **17,000+ 行代码**。一条 X 帖子 ([movementlabsAI post](https://x.com/movementlabsAI/status/2011964766533632380)) 将该模型与 Movement Labs 联系在了一起。
  - 这种氛围纯粹是由排行榜驱动的 FOMO（错失恐惧症）——用户称其为 “Opus/Gemini 杀手”，并对其**即将开源**的说法深信不疑，尽管具体的底层技术细节和 Benchmark 仍然匮乏。

- **Video Arena 仅限对战模式，Veo 受到速率限制**：LMArena 的 **Video Arena** 已改为**仅限对战模式**（无直接对话 / 并排对比），**Veo** 可在对战中使用，但网站限制为 **24 小时内 3 次**生成，Discord 限制为 **24 小时内 5 次**。
  - 用户强烈要求“不限次”访问 **Veo 3.1 Fast**，但管理员强调了限制；与此同时，根据 [LMArena FAQ](https://lmarena.ai/faq)，神秘的 **Siren** 视频模型仍是一个代号，引发了对其背后真实身份的猜测。

- **GPT-5 Image Mini 价格一夜之间翻了两番**：OpenRouter 用户报告称 **openai/gpt-5-image-mini** 图像生成的价格在一夜之间从 **$0.01 飙升至 $0.04**，贴中未给出任何解释。
  - 与此同时，Perplexity 用户追踪到了 **Grok** 行为的变化——**Grok Code** 根据用户 Token 消耗量攀升至“前五名” ([X post](https://x.com/i/status/2011823610386600009))，截图显示内部正在对一个新的（可能更快的）Grok 变体进行 A/B 测试。


**4. GPU/Kernel 工程与 Profiling 工具链**

- **Chrome Trace UI 在 600MB 时崩溃，Perfetto 前来救场**：在 GPU MODE 中，有用户报告 **Chrome Trace Visualizer** 在加载 **PyTorch profiler** trace 文件时，大小达到 **600MB** 左右就会崩溃或渲染为空白，尽管文档暗示只有接近 1GB 时才会出现问题。
  - 成员建议使用 [Perfetto UI](https://perfetto.dev/) 作为替代方案。一名开发者描述了他在 VSCode 集成的 Perfetto 查看器中处理 **700MB** trace 文件遇到问题后，在 [ncompass.tech](https://docs.ncompass.tech) 构建了一个 trace 分块查看器。

- **Hopper TMA/WGMMA：Swizzles、Strides 与 5D Copy**：GPU MODE 深入研究了 Hopper 时代的 **TMA tensor copy** + **WGMMA** 共享内存布局约束（K-major 的 A/B tiles），讨论了 LBO/SBO 的含义，以及为什么在某些情况下多个 2D TMA 优于 `BLOCK_Mx16B` 的 3D TMA。
  - 他们围绕具体代码 ([pipeline_tma_wgmma.cu](https://github.com/danielvegamyhre/gemm/blob/9fe95aa61ee7ebca4ded8b5029494b0d58e0d2e2/pipeline_tma_wgmma/pipeline_tma_wgmma.cu#L109-L118)) 展开讨论，并指出了一些陷阱，如 swizzling 会影响 LBO 但不影响 SBO，并提醒 TMA copy 支持 **5 个维度**。

- **Profiler 降低时钟频率，Benchmark 逼疯开发者**：在 GPU MODE 的 NVIDIA 竞赛频道中，用户发现 profiling 运行在 zip 中仅覆盖单个 kernel，并了解到 **profiling 开销是预期之内的**，这使得 profile 结果不适用于绝对运行时间的比较。
  - 一个重要的发现：**ncu** 可能会将 **SM Frequency** 降低到 **1.08 GHz** 左右。Benchmarking 期间出现的 `CUBLAS_STATUS_INTERNAL_ERROR` 被怀疑源于越界访问，调试建议使用 `torch.cuda.synchronize()` 以更早地暴露错误。


**5. 安全与可靠性：越狱防御与内存/状态泄漏**

- **GPT 5.2 记忆功能据称泄露跨会话聊天**：在 BASI Jailbreaking 中，一名用户声称在免费版 **GPT 5.2** 上启用 **memory** 导致了来自其他会话的聊天内容泄露，并分享了一张图像作为证据 ([screenshot](https://cdn.discordapp.com/attachments/1228043845967544380/1461404780831445237/image.png?ex=696b1783&is=6969c603&hm=91a356b1b007e9bb6123ede9a79414a836c03014291506ae32be52e3082e4eec))。
  - 他们推测这可能是一个内存 Bug，并询问删除其他聊天是否能阻止这种情况——这一轶事引发了社区对“状态隔离（state isolation）”更广泛的焦虑，这些社区已经在与 Agent 系统中复杂的会话/状态管理作斗争。

- **Llama 3.2 无视旧的越狱 Prompt**：BASI 用户报告称 **Llama 3.2** 抵御了在 **Llama 3.1** 上有效的越狱 Prompt，并指向了一个旧方法在新版本上失败的具体例子 ([Chepenik’s post](https://chepenikconor.medium.com/day-855-9ae6f88b192c))。
  - 结论是务实的：防御者正在收紧 Guardrails，因此攻击者转向技术调整（例如“关闭思考”的建议）以及精选资源，如 [Arcanum 的 AI 安全资源中心](https://arcanum-sec.github.io/ai-sec-resources/)。



---

# Discord: 高层级 Discord 摘要

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **DeepSeek 遭遇叙事型 Jailbreaking**：成员们推荐使用 **DeepSeek**，因为它容易受到叙事攻击（一类常见的越狱技术）。他们澄清这类攻击包括 *roleplay（角色扮演）和 persona（人格）* 攻击。
   - 该模型因其极高的易感性在成员中引起了关注。
- **GPT 5.2 免费版泄露聊天秘密？**：一位用户报告称，在 **GPT 5.2** 的免费层级中启用 memory 功能可能会导致其他会话的聊天内容泄露，并提供了 [一张图片](https://cdn.discordapp.com/attachments/1228043845967544380/1461404780831445237/image.png?ex=696b1783&is=6969c603&hm=91a356b1b007e9bb6123ede9a79414a836c03014291506ae32be52e3082e4eec) 作为证据。
   - 用户质疑这是否源于 memory 问题，以及删除其他聊天会话是否能解决此次泄露。
- **Llama 3.2 防御严密，越狱尝试失败**：用户正积极尝试越狱 **Llama 3.2**，据报告，在 **Llama 3.1** 上有效的 prompt [在这一新版本中失败了](https://chepenikconor.medium.com/day-855-9ae6f88b192c)。
   - 试图诱导有害回答（如制作违禁药物的指令或极端减肥建议）的尝试均遭到拒绝，这表明其安全措施得到了增强。
- **Arcanum's Armory：免费 AI 渗透测试资源出现**：一位成员分享了 [Arcanum 的 AI 安全资源中心](https://arcanum-sec.github.io/ai-sec-resources/?utm_source=executiveoffense.beehiiv.com&utm_medium=referral&utm_campaign=executive-offense-the-arcanum-ai-security-resource-hub)，为 **AI pentesting** 提供了结构化的工作流。
   - 这一 GitHub 资源正在被传阅，并被标记为团队常规调查的对象。
- **Grok 的图像审核面临挑战**：紧随 [Elon Musk 的要求](https://x.com/elonmusk/status/2011527119097249996)，用户正试图通过创建一个色情内容推文串来突破 **Grok 的图像审核（image moderation）**。
   - 一名用户预测，这一举动 *很快将成为 Twitter 历史上色情内容最多的推文串*。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **VRAM 加速训练！**：成员们观察到增加 **VRAM** 通常会加速模型执行，并指出 Unsloth 文档中的 `-ot` 标志有助于高效管理张量放置（tensor placement）以实现巅峰性能。
   - 一位用户提到 `-ot` 设置通常优于 `n-cpu-moe` 配置。
- **Anthropic 对 Python 的热衷**：成员们分析了 [Anthropic 对 Python 的投资](https://pyfound.blogspot.com/2025/12/anthropic-invests-in-python.html)，并赞扬其商业模式优先考虑高效模型而非过度商业化。
   - 一位成员表示，*在开发相关事务方面，Claude 与其他任何工具相比一直非常强大（cracked）*，尽管其成本较高。
- **Unsloth RL 扩大海量 Context Window**：[新的长上下文 RL 版本](https://x.com/UnslothAI/status/2011827592886960131) 现在支持 **700 万 token 的 context window**，比之前的版本增加了 **7 倍**。
   - 参与者对指数级节省内存的潜力及其应用（如与“虚拟老婆”进行永无止境的对话）印象深刻。
- **Qwen3 VL 架构 Bug 已修复**：用户报告了一个 Bug，即 **Qwen3 VL 架构** 未被 Unsloth 正确识别为视觉模型，导致在视觉微调期间出现与模型和数据集不匹配相关的 `ValueError`。
   - 解决方案包括升级 `transformers` 并验证环境设置，同时提醒对 Qwen3 模型使用 [正确的 notebook](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision.ipynb)。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Sonnet 4.5 驱动后台 Agent**：用户确认选择 **Sonnet 4.5** 现已成为启动后台 Agent 的一个选项。
   - 一位开发者在 X 上指出，他们正致力于在未来几天/几周内大幅改进后台 Agent。
- **Composer-2 悄然承担推理角色**：用户观察到手动添加 **composer-2** 会将其指定为推理模型。
   - 一位用户引用了最近的一条 **Cursor** 推文，暗示 **composer-2** 作为推理模型即将到来。
- **GPT 5.2 Codex 表现平平**：一位用户报告对 **GPT 5.2 Codex** 印象平平，指出它在制定计划时未能遵循指令。
   - 另一位用户指向了 [cursor.com 的 scaling-agents 文章](https://cursor.com)，该文章建议 **GPT-5.2** 模型在扩展自主工作、指令遵循和精度方面表现更优，并能有效利用 Sub-agents，这与报告的体验形成对比。
- **Sub-Agents 内置选项带来惊喜**：用户探索了 **sub-agents** 的功能，注意到存在两个内置 Sub-agent：**Explore** 和 **generalPurpose**。
   - 有人指出只有特定模型可以调用 Sub-agent，并且在为每个 Sub-agent 可靠地设置模型方面存在问题。
- **Token 使用引发最小化策略讨论**：用户讨论了最小化 Token 使用的策略，特别是在探索代码时，并寻求针对性和全面代码探索工具的建议。
   - 一位用户建议尝试 [Nia mcp](https://nia.mcp)，另一位用户建议使用一个命令来审查代码更改并提供相关的 Prompt 以优化 Token。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet 颜色自定义功能受限**：在 **Comet** 中设置浏览器颜色不再改变 Perplexity 的配色方案，因为 [theme="yellow" HTTP 请求头](https://discord.com/channels/1047197230748151888/1047649527299055688/1461239027771510878) 似乎消失了。
   - 用户正在调查此改动是 Bug 还是刻意的更改。
- **Grok Code 跻身前五**：根据[这条 X 帖子](https://x.com/i/status/2011823610386600009)，**Grok Code** 已攀升至前五名，该帖子按用户 Token 消耗量对模型进行了排名。
   - 这一里程碑突显了开发者对 **Grok Code** 模型日益增长的采用和使用。
- **Airtel Pro 激活困扰**：尽管遵循了 [Perplexity 帮助文章](https://www.perplexity.ai/help-center/en/articles/11842322-perplexity-pro-airtel-promo)中的所有步骤并联系了支持团队，用户在激活 **Airtel Pro** 订阅时仍面临困难。
   - 一些用户收到了来自 AI Agent Sam 的模板化回复，但没有任何实质性的解决办法。
- **发现新的 Grok 变体**：一个新的 **Grok** 模型变体（可能是一个更快的版本）正在使用代号进行内部测试，正如匿名 Discord 投票的截图所揭示的那样。
   - 这些模型被称为 assistant a 和 assistant b，暗示了这一未发布的 **Grok** 模型可能处于 A/B 测试场景中。
- **部分地区 Image Generation 失效**：某些地区（特别是欧洲）的用户报告 **image generation** 无法正常工作，导致人们猜测这种限制可能是故意的。
   - 区域封锁的原因尚不清楚，但它正在影响对 **image generation** 功能的访问。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Video Arena 进入对战模式**：Video Arena 现在**仅限对战模式 (Battle mode)**，取消了直接对话或并排对比功能，尽管这些功能曾短暂对早期用户开放；新的 **Veo** 模型已在对战板块上线。
   - 视频生成的频率限制已经生效：网站上为 **24 小时内 3 次**，Discord 上为 **24 小时内 5 次**。
- **用户渴求 Veo 3.1 Fast**：对“无限量” **Veo 3.1 Fast** 的需求很高，但管理员表示频率限制是主要障碍（网站 **24 小时内 3 次**，Discord **24 小时内 5 次**）。
   - 当一名用户询问是否可以在外部网站测试时，管理员给出了开放式的回应：*“你为什么不去试试呢？”*
- **Falcon-H1R-7B-GGUF 模型赢得赞誉**：用户对 [Falcon-H1R-7B-GGUF](https://huggingface.co/unsloth/Falcon-H1R-7B-GGUF) 模型印象深刻，纷纷要求获取更多信息。讨论中还包含了一篇论文链接：[Transformer-Based Generative Adversarial Network for Image Super-Resolution](https://huggingface.co/papers/2601.02346)。
   - 该模型的具体能力和应用仍在探索中，显示出社区对其潜力的浓厚兴趣。
- **Siren 视频模型依然神秘**：**Siren 视频模型**可能是一个代号为早期访问的模型，属于仍在开发中的前沿模型。虽然细节寥寥，但根据 [FAQ](https://lmarena.ai/faq)，用户反馈可以直接影响哪些模型能够继续推进。
   - 社区内有猜测认为它可能是 **Wan 2.5**，因为它具备 30 fps 的生成速度，这反映了社区活跃的推测氛围。
- **Hawk Ultra 挑战 Opus 的霸主地位**：**Hawk Ultra** 模型正被拿来与 **Gemini 3 Pro** 进行比较。据报道，它能在单个 prompt 中生成超过 **17,000 行代码**。一些用户声称它超越了 Opus 和 Gemini。根据 [这条 X 帖子](https://x.com/movementlabsAI/status/2011964766533632380?s=20)，该模型背后的团队是 Movement Labs。
   - 社区热情高涨，并有承诺称很快将开源；一位用户惊叹道：*“这让我非常兴奋 (got me so gassed)”*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio API 引发 IDE 集成热潮**：成员们对于使用 LM Studio 在本地运行模型，并将其 **OpenAI-compatible API** 连接到本地 IDE 以运行 Agent 和脚本感到兴奋，这可以节省 token 费用。
   - 一名成员确认 LM Studio 确实可以为此目的启动 **OpenAI-compatible API**，[这里是一个链接](https://link.to.nowhere)。
- **GPT-OSS-20B 的速度与其规模不成正比**：成员们讨论了为什么 [GPT-OSS-20B 模型](https://huggingface.co/models?search=GPT-OSS-20B) 感觉比许多 8B 或 12B 模型更快。解释称这是一个 **Mixture of Experts (MoE)** 模型，每个 token 仅激活一部分（**3.6B**）参数。
   - 尽管没有使用全部权重，该模型在**数学、物理和量子力学**等任务中表现良好，甚至在 **6700XT** GPU 上也能维持超过 **34k tokens** 的上下文。
- **顶级编程 LLM 狂热**：用户在寻求最适合编程的本地 LLM 推荐，提到了 **DeepSeek R1, Qwen3 和 Devstral**，但也有人指出 [Claude](https://claude.ai/) 整体表现依然最强。
   - 考虑到硬件限制，成员建议优先考虑将模型放入 VRAM 而非追求原生速度。由于 VRAM 对 LLM 至关重要，建议选择 **20GB 的 RTX 3080** 而非 **11GB 的 3080 Ti**。
- **LiquidAI 工具调用困扰**：一名用户在 **LFM2.5-1.2B** 模型中使用工具调用时遇到问题，在询问时间时收到了输出 `<|tool_call_start|>[execute_command(command="date")]<|tool_call_end|>`。
   - 排查步骤包括验证工具访问权限、尝试模型的 instruct 版本、确保正确的 system prompt，以及参考 [LiquidAI 文档](https://docs.liquid.ai/lfm/key-concepts/tool-use) 获取指导。
- **MX150 奇迹般地微调了 350M 模型**：在 **hardware-discussion** 频道中，一名用户成功在仅有 **2GB VRAM** 的 **MX150 笔记本**上完成了对 **350M 模型**的全量微调（full fine-tune）。
   - 令人惊讶的是，该过程需要 **CUDA 12.6**，这表明某些配置可能会出人意料地要求特定的 CUDA 版本以实现兼容性。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Smorty 确实不是 LLM**：成员们确认 **Smorty** 不是 LLM 而是真人，因为其独特的写作风格以及在 [Lemmy 上的帖子](https://lem.lemmy.blahaj.zone/u/Smorty?page=1&sort=New&view=Posts)。
   - Smorty 正在编写 **skill.md**，并指出社区“非常反对机器学习相关的东西”。
- **GPT-5 图像生成成本一夜之间暴涨**：使用 **openai/gpt-5-image-mini** 模型的图像生成成本一夜之间突然从 **$0.01 飙升至 $0.04**。
   - 此次调价的原因尚未披露。
- **BYOK 功能导致 AWS Key 身份验证噩梦**：一位成员报告了在不同平台和模型（包括 **SillyTavern**、**Amazon Bedrock** 和 **Anthropic**）上通过 **OpenRouter 的 BYOK 功能**使用其 **AWS key** 时出现的问题。
   - 收到的错误消息为 *"Unauthorized or not cookie auth credentials found"*，表明存在潜在的身份验证问题。
- **OpenCode 被成员评为最佳编程 Harness**：成员们讨论了编程测试框架（harness），并强调 [**OpenCode**](https://github.com/OpenRouterTeam/awesome-openrouter?tab=readme-ov-file#aventura) 是最佳选择，尤其是配合 **oh-my-open code** 等插件。
   - 一位成员指出它“让 claude code 用起来感觉像是在使用某种老派的终端应用”，展示了其高效性。
- **Cerebras 与 OpenAI 达成 2028 年协议**：OpenAI 宣布计划于 **2028** 年与 [Cerebras 建立合作伙伴关系](https://openai.com/index/cerebras-partnership/)，引起了成员们的惊讶和猜测。
   - 一些人认为，考虑到 Cerebras 的长期存在以及对 **120B** 等大模型的支持，这次合作可能是对 **Groq** 交易的回应。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **字节级 LLM 与 Diffusion 结合**：一位成员将 **byte level LLMs** 和 **diffusion** 模型结合起来，使字节模型能够更可靠地处理各种文件类型。
   - 该方法利用 diffusion 来纠正微小的字节错误，如[屏幕录像](https://cdn.discordapp.com/attachments/1149866623109439599/1461342380392185928/Screencast_from_13-01-26_152930.webm?ex=696add65&is=69698be5&hm=5bcb4ae2ce4e375aac96cd552f00b7d4077391dbad48fa2b2745608cc1555828&)所示。
- **Flux 2 对 VRAM 的需求**：**Flux 2 9B 模型**使用了 **Qwen 3 8B** 文本编码器，加载所有权重进行推理服务需要 **35GB 的 VRAM**。
   - 不过，在没有并发用户时，内存占用会减半；ComfyUI 可能是 diffusion 的另一种选择。
- **LLM 获得神经系统**：一位成员正在开发一种*原生 Transformer 架构扩展*，为 LLM 注入**神经系统**，配备短/中/长期记忆，且**计算成本 <1%**。
   - 开发者声称其*与模型大小相比线性扩展 1-2%*，在 BEIR 上的表现与原模型 **95%** 一致；目前仍需可演示的基准测试。
- **Google 的 Gemma 亮相**：**Google** 正在发布 **Gemma** 模型，引发了成员们的兴奋。
   - 一位成员调侃道：*Gemma, meta was never more meta!*。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Chrome Trace Visualizer 难以处理大型 PyTorch Profiles**：成员们反映，在使用 **PyTorch profiler** 时，虽然文档建议限制为 1GB，但 **Chrome Trace Visualizer** 在 **600MB** 时就会崩溃；建议使用 **Perfetto UI** 作为替代方案。
   - 一位成员正在开发一种具有 trace chunking（追踪分块）功能的追踪查看工具 ([ncompass.tech](https://docs.ncompass.tech)) 以解决大型追踪文件问题，目前在 **VSCode** 内的 **Perfetto viewer** 中打开 **700MB** 文件时遇到问题。
- **TMA Tensor Copy 性能优化**：成员们探讨了 K-major 布局中 A/B tiles 的 **shared memory layout** 要求，辩论了 TMA 的 LBO 和 SBO 设置，参考[这段代码](https://github.com/danielvegamyhre/gemm/blob/9fe95aa61ee7ebca4ded8b5029494b0d58e0d2e2/pipeline_tma_wgmma/pipeline_tma_wgmma.cu#L109-L118)。
   - 会议还提醒，TMA tensor copy 支持 **5 维**，虽然较大的 TMA 操作效率更高，但对于 `BLOCK_Mx16B`，多个 2D TMA 可能比单个 3D TMA 更快；swizzling 会影响 LBO 设置，但不影响 SBO 设置。
- **Information Gravity 约束 AI 幻觉**：一位成员正在应用 **Information Gravity** 来稳定 **Inference Stability** 并缓解 **Hallucination Loops**，绘制了 token selection 的 **Excitation Flux**，并观察到在 S > 45 之后转向线性增长，相关模块可在 [GitHub](https://github.com/brayo003/Substrate-X-Theory-of-Information-Gravity/tree/main) 上获得。
   - 在 **1.0** 处的 **Hysteresis Firewall** 通过 **2.2x gamma-eff flush** 强制执行稳定性。
- **CUDA 压缩协作启动**：一名电子工程硕士生开始了一个基于 GPU 的 CUDA 数据压缩项目，重点是 **Golomb-Rice** 压缩，并寻求资源推荐；一位成员分享了 [CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)。
   - 成员们讨论了 CUDA 中 **32** 这一极小 block size 的劣势，原因在于 SM (*Streaming Multiprocessor*) 工作的 warp 数量；**WGMMA/tcgen05** 需要 128 个线程的倍数协同工作。
- **Profiling 异常干扰 Kernel 竞赛**：成员们指出了一些与 **NVIDIA 竞赛**相关的 profiling 问题，包括部分 kernel 覆盖、预期的 profiling 开销，以及 **ncu profiler** 将 **SM Frequency 降至 1.08 GHz**。
   - 在 benchmarking 期间出现的 **CUDA error** `CUBLAS_STATUS_INTERNAL_ERROR` 被归因于潜在的越界访问，建议使用 `torch.cuda.synchronize()` 进行调试。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 与 Cerebras 巨型规模合作！**：OpenAI 和 Cerebras 宣布达成战略合作伙伴关系，详情可见 [OpenAI 官网](https://openai.com/)。
   - 社区成员对这一合作对 AI 基础设施的潜在影响表示兴奋。
- **Ministral 3 论文投下重磅炸弹！**：**Mistral** 的 **Ministral 3** 模型新论文由 [@qtnx_ 在 Twitter 上发布](https://twitter.com/qtnx_/status/2011510403550024087?s=20)，引发了对其能力和性能的讨论。
   - 该模型备受期待，尽管尚未发布性能基准测试。
- **AI Agent 正在玩转数据垄断？**：Olivia Moore 指出 AI Agent 订阅（如 **Manus**）如何提供扩展的专有数据访问权限，例如 SimilarWeb 的 **12 个月**数据对比免费计划的 **1 个月**。
   - 这一趋势表明，有价值的数据集正逐渐被封锁在 AI Agent 订阅服务的门槛之后。
- **本地 LLM 推理挑战云巨头！**：Charles Frye 的[新 Modal 指南和代码示例](https://xcancel.com/charles_irl/status/2011484220032762114?s=46)表明，本地 LLM 推理的性能和性价比可以匹配甚至超越主要的 LLM API。
   - 成员们正在询问现在是否可以运行本地会议转录（类似于本地化的 Granola），而无需云服务。
- **LLM 化学考试不及格！**：根据这篇 [推文](https://x.com/bfl_ml/status/2011825819082244266?s=46)，LLM 在化学方面表现吃力，特别是在幻觉化胆固醇结构中的他汀类药物（statins）等细节时。
   - 一位成员正在 [ChemIllusion.com](https://x.com/bfl_ml/status/2011825819082244266?s=46) 开发工具来纠正 LLM 的化学错误。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GLM 4.7 和 Minimax 崛起，受预算有限的工程师青睐**：成员们报告称 **GLM 4.7** 和 **Minimax** 这两家 LLM 供应商表现出色，其中 **GLM 4.7** 可通过 z.ai 编码计划获取，而 **Minimax** 通过 Moonshot 接入非常便宜。
   - 一位成员正在寻找最适合在几天内生成大量“图生视频”的 **AI tool**，并倾向于选择“付费”方案，同时也收到了“使用 API”的建议。
- **GPT 5.2 选项在部分用户界面消失**：一些成员报告称，某些账号的 **GPT 5.2** 选项消失了，不过退出并重新登录后又重新出现；部分人声称 *5.2 是一个更差的模型*。
   - 一位成员抱怨尽管使用的是 **GPT 5.2**，但仍收到了“超出限制（your limit exceeded）”的消息。
- **AI-Deepfake 认证计划启动**：一位成员正在为一个名为 PhantomTrace 的平台开展 **AI 深度伪造检测与验证认证** 的早期试点工作。
   - 他们正在寻找由研究人员、构建者、安全专家和记者组成的小型小组，以评审学习目标草案、测试动手检测实验室，并帮助定义“通过”的标准，相关讨论见 [Discord context](https://discord.com/channels/974519864045756446/1204360881593520128/1461532097641578672)。
- **CustomGPT 旨在实现 Project 集成**：一位用户表达了在 Project 内部使用 **CustomGPT**，或将 **CustomGPT** 的结果放入 Project 中的需求。
   - 他们还希望能够将 Project 外部生成的任何对话移动到 Project 中。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pangram 的 AI 检测能力引发争论**：成员们质疑 **Pangram** 作为 **AI text detector** 的准确性，理由是它曾将 **Claude** 生成的内容误判为“100% 人类编写”，而另一位成员则分享了其检测方法背后的 [论文](https://arxiv.org/abs/2402.14873)。
   - 根据 [Pangram 官网](https://www.pangram.com) 的链接，讨论认为统计文本中的 **em dashes** 可作为检测 **AI generation** 的一项指标，估计误差范围在 *+-10%* 左右。
- **寻求用于检测网页合成文本的小型分类器**：一位成员正在寻找一个 **small classifier model** 来估算网络上合成文本（synthetic text）的数量，并提议运行一次网页爬取进行测试；其他人建议使用为投机性解码（speculative decoding）训练的 **drafter model**，尽管这通常是特定于模型的。
   - 社区还讨论了构建自有分类器的选项，但指出在大规模应用时成本可能非常高昂。
- **社区关注开源训练数据集**：一位成员询问社区是否有兴趣在 The Pile 和 CommonPile 等预训练数据集之外，**开源用于微调（finetuning）** **GPT-Neo** 等预训练 LLM 的 **instruction-following datasets**。
   - 另一位成员表示愿意为社区项目提供开发技能支持。
- **大写字母是否限制了模型能力？**：一位成员询问是否有研究表明模型在处理带有正确 **capitalization/grammar**（大写/语法）的提示词时表现优于全小写，并指出了 [三篇 Arxiv 论文](https://arxiv.org/abs/2310.11324)、([2411.10541v1](https://arxiv.org/abs/2411.10541v1))、([2508.11383v1](https://arxiv.org/abs/2508.11383v1))，但指出这些论文主要关注 Prompt 格式而非大写等细节。
   - 一位成员认为正确的大写/语法可以提高模型性能，并建议使用 **vLLM** 的基准测试工具进行验证。
- **Global CoT 分析尝试揭示模式**：一位成员分享了一篇关于 **global chain of thought (CoT) analysis** 及初步揭示模式尝试的 [LessWrong 文章](https://www.lesswrong.com/posts/q9g9zuudd3Pvw2cbj/global-cot-analysis-initial-attempts-to-uncover-patterns-1)。
   - 该分析旨在通过检查模型采取的推理步骤，理解模型如何得出结论，从而可能揭示其决策过程的见解。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Arch 支持者赞美持续更新的 Arch 系统**：成员们讨论了 **Arch** 优于 **Ubuntu** 和 **Debian** 的优点，强调了 **Arch** 频繁的软件包更新，类似于 **macOS** 和 **Windows**。
   - 一位用户建议将 **Garuda KDE**（Mokka 和 Dragonized 版本）作为一个用户友好的起点，它具有开箱即用的功能。
- **PR 流水线深入：测试正在进行**：一场讨论澄清了 PR 上 `imported internally` 标签的含义，这表示该 PR 已被克隆到内部仓库进行最终测试和集成。
   - `imported internally` 标签信号表明 PR 处于合并前的最后阶段，完成后将被标记为 `merged-internally`。
- **.NET 噩梦：遗留系统的哀歌**：一位成员哀叹被分配到一个来自 **2014年** 的遗留 **.NET 4.5.2** 项目，该项目仅在 **Windows** 上运行且缺乏文档。
   - 另一位成员分享了类似的经历，一个独立的 **C#** 项目饱受问题困扰且缺少文档，将其比作*在沙漠中寻找温泉和水*。
- **Mojo 考虑通过 SPIR-V 实现图形着色器**：目前正在考虑将图形着色器（graphics shaders）引入 **Mojo**，特别是通过 **SPIR-V 后端** 来支持 *计算着色器 (compute shaders)*。
   - 一位成员提醒说，一旦编译器**开源**，构建工作将是一项*非同小可 (non-trivial)* 的任务。
- **着色器与矩阵运算的对决**：针对 **shaders** 与传统**矩阵操作 (matrix operations)** 之间的区别提出了疑问，特别是考虑到最近 **CUDA** 的发展。
   - 作为回应，一位成员链接了 [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api)，另一位成员链接了 [Death to Shading Languages](https://xol.io/blah/death-to-shading-languages/) 以帮助澄清差异。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Minimax M2.1 在 Claude 中表现优于 Kimi K2**：一位用户报告称，在 **Claude code** 中运行的 **Minimax m2.1** 在代码质量、规划和 API 速度方面优于 **Kimi K2**，并提到他们每月为 **Kimi v2** 支付 40 美元。
   - 他们发现 **Kimi 的 API** 速度较慢且模型表现较差，希望能有更新的版本发布。
- **关于 Kimi CLI 默认使用 K2 Turbo 的辩论**：一位用户质疑为什么在拥有正式订阅的情况下，默认的 **Kimi CLI 应用** 不默认为 **K2 Turbo**。
   - 另一位成员建议 **Kimi K2 Turbo** 的速度应在 **73 tps** 左右，相比之下 **MiniMax m2.1** 为 **38 tps**，**Z.Ai** 的 **GLM-4.7** 为 **41 tps**（尽管后者在线率较差）。
- **新的幻灯片功能使用了带 Vision 的 K2 模型？**：一位成员询问新的幻灯片功能是否使用了更新的、支持 **Vision** 的 **K2 模型**。
   - 图像分析显示它会搜索图片进行参考，这暗示了其具备一定的视觉能力。
- **关于 Kimi 模型弃用的疑问**：一位成员询问 **Kimi 模型** 是否像 **Google 的 Gemini 模型** 一样每 **12-14 个月** 停止服务，以及切换到 **Kimi K2** 是否会遇到同样的问题。
   - 旧模型可以在 [Moonshot API 平台](https://platform.moonshot.ai/docs/pricing/chat#generation-model-moonshot-v1)上找到，且一年前的模型在 [kimi.com](https://kimi.com) 上仍可使用。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **开发者寻求超酷项目**：一位成员正在积极寻找可以发挥其**开发技能**的**超酷项目**。
   - 鼓励有兴趣的各方与其联系进行协作。
- **Discord 管理员申请暂停**：一位成员询问如何加入审核团队，但另一位成员澄清该职位目前不可用。
   - 未提供停止招聘的具体原因。
- **用量追踪需要 AI 工程师**：一个正在进行的项目需要一名 **AI 工程师** 来增强**用量追踪**并开发更强大的**计费/额度系统 (billing/credit system)**。
   - 这意味着需要同时具备 AI 和财务系统方面的专业知识。
- **支付问题困扰平台**：一位成员报告称，在尝试为账户充值时遇到持续的**支付问题**，包括升级会员、使用 Link 以及使用信用卡或支付宝支付时的问题。
   - 他们尚未收到帮助台或邮件支持的回复，这表明客户服务可能存在延迟。

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **峰会由于缺乏针对远程注册者的直播**：一位成员询问了纽约峰会是否提供**直播 (live stream)**，表达了远程参与的愿望。
   - 他们非常希望能注册参加，但无法亲临现场。
- **无状态服务器带来可扩展性方面的成本节约**：一位成员提出了一种[签名方法 (signature method)](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2091)来平衡模式冻结 (schema freezing) 和动态服务器功能，旨在允许**无状态 MCP 服务器**更高效地处理多个活动对话。
   - 他们注意到，目前在 Goose 中的设置是为每个对话启动一组新的 MCP 服务器，随着并发对话数量的增加，成本变得越来越高。
- **动态工具集应对传输问题**：一位成员指出 [issue #1442](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1442) 和 GitHub MCP 的动态工具集是服务器如何在 **STDIO** 上处理状态的示例，这可能会统一远程和 STDIO 设置的行为。
   - 该成员承认，鉴于他们目前的 SDK 架构在每次请求时都会构建一个新的“服务器”，并根据用户/标签定制注册的工具，维持一个真正的无状态 **STDIO 服务器**是很困难的。
- **持久化会话在服务器启动时保存状态**：提出了**持久化会话 (persistent sessions)** 的主题，作为在 Agent 和 MCP 服务器重启时保留会话功能的一种手段。
   - 另一位成员提到在 Go SDK 之外使用他们自己的会话中间件来实现水平扩展，并建议跨重启存储和检索**会话数据 (session data)** 的能力将是有益的。



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **1x1 卷积优于 SVD/PCA**：一位成员建议使用 **1x1 卷积 (1x1 Convolution)** 代替 **SVD/PCA** 进行特征提取，认为 **SVD/PCA** 提取的是方差最高（*最响亮*）的特征，可能会捕捉到通用的语法噪声，而不是特定的*意图*信号，参考自[这条推文](https://fxtwitter.com/i/status/2011094378396467316)。
   - 他们认为 **1x1 Conv** 将允许模型通过反向传播 (backprop) 精确地学习哪些头部 (heads) 对损失函数 (loss function) 敏感，并且在推理时更轻量。
- **“Quanta”理论引发辩论**：成员们讨论了 *quanta* 理论，该理论指出网络必须学习各种模块，每个模块实现不同的算法或检索不同的知识片段。
   - 一位成员表示怀疑，认为许多机制可能是交织在一起的，或者过于通用而无法指定特定用途，这可能会导致对神经网络进行过度简化的机械解释 (mechanistic explanation)。
- **AI 辅助编程与 Vibe Coding 的对决**：一位成员将 **AI 辅助编程 (AI assisted coding)** 工具 (cursor/windsurf/antigravity) 与他们称之为 **vibe coding** 的工具 (devin/tembo/jules) 进行了对比。
   - 未提供更多细节。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 工具集可能优于原生 LLM 工具**：一位成员参考了 [DSPy 文档](https://dspy.ai/learn/programming/tools/#using-native-tool-calling)，并询问了关于**原生工具调用 (native tool calling)** 与**自定义工具调用**的基准测试对比，特别是 **DSPy 工具** 是否总是表现更好。
   - 另一位成员回答说，这取决于所使用的具体语言模型 (LM)，暗示性能并非对于 **DSPy 工具** 普遍更好。
- **原生和 DSPy 工具都需要基准测试**：一位成员强调，性能在不同的语言模型中各不相同，即使是来自同一个 AI 实验室的模型也是如此，因此**基准测试对于特定的用例和模型组合至关重要**。
   - 另一位成员表示赞同，指出性能可能会向任何一个方向波动，用户应该使用其**特定模型和程序**进行测试，以衡量和评估哪种方案效果最好。
- **原生工具调用质量可能较低**：成员们讨论了**原生工具调用 (native tool calling)** 有时可能比其他方法产生更低质量结果的可能性。
   - 文档中的陈述提出了一个较弱的主张，但这在某些模型中确实可能发生。

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **aider 用户希望支持自动加载 (Autoload)**：用户请求在 **aider** 中增加一个无需提示即可自动添加文件的功能，旨在获得更少交互的体验。
   - 具体请求涉及配置 **aider** 以跳过添加文件的提示，表明了对简化工作流的需求。
- **aider 安装故障排除**：一位用户报告了通过命令提示符安装 **aider** 后的设置困难，如[此截图](https://cdn.discordapp.com/attachments/1133060505792159755/1461275448574218373/image.png?ex=696a9f10&is=69694d90&hm=19ccaef4fb45cd4288b6307abb3eca0a6819f27eb6253f0820357b2219006a4d)所示。
   - 该用户在未提供更多上下文的情况下寻求后续步骤的指导。
- **使用 CI 日志修复 aider**：一位用户询问了如何将 **CI logs** 与 **aider** 结合使用来修复失败的测试，同时在 Git 中排除日志文件。
   - 建议的一个潜在解决方案是使用命令 `aider --read ci.log`。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **黑色 PNG 令 Stable Diffusion 用户困惑**：一位用户报告在运行 `examples/stable_diffusion.py` 并带有 `--fakeweights` 选项时，遇到了全黑的 PNG 图片，这表明 tinygrad 中的 **Stable Diffusion** 可能存在问题。
   - 该问题似乎与 **NULL device** 及其与内核调度的交互有关；调试工作正在进行中。
- **NULL 设备：无计算内核**：一位用户询问了 tinygrad 中 **NULL device** 的用途，质疑它是否执行任何计算以及它如何辅助调度内核。
   - 另一位成员澄清说 **NULL device** 不执行任何计算，但它被用于调度内核，一位用户称之为“一个很酷的特性”。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器沉寂太久，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器沉寂太久，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该服务器沉寂太久，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：频道详细摘要与链接

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1461105640553316545)** (1085 条消息🔥🔥🔥): 

> `Prodigy 对决 Barbie，Smack My Bitch Up 人声，GPT-5.2 聊天泄露，用于个人 TTRPG 战役的 AI，AI 与人类创作的音乐` 

- **Prodigy 完胜 Barbie：一首 Diss Track？**：一位用户说 *prodigy 棒极了*，促使另一位用户为想要毁掉 **Barbie** 娃娃的人请求翻译。
   - 随后爆发了争论，包括诸如 *go crywank to chumba wumba on the barbie* 之类的言论。
- **Shahin Badar 献唱 'Smack My Bitch Up'**：一位用户询问 *谁演唱了 Smack My Bitch Up 中的女声*，另一位用户确认歌手为 [**Shahin Badar**](https://en.wikipedia.org/wiki/Shahin_Badar)。
   - 随后他们发布了该歌曲的 [YouTube 链接](https://youtu.be/gJ4bW4KNffo?si=0SlbsHlcS3gTofuq)。
- **Pliny 对 Gemini 用户的简短赞美**：一位用户发布了一个 [Jim Carrey 的 GIF](https://tenor.com/view/bruce-almighty-jim-carrey-beautiful-happy-smile-gif-4874848)，配文 *Pliny 已经大概 2 个月没在 general 频道发言了*，随后 **Pliny** 本人回复了另一个[动画 GIF](https://tenor.com/view/korone-flip-combo-breaker-killer-instinct-best-dog-inugami-korone-gif-25381954)。
- **DeepSeek 在越狱尝试中初露锋芒**：一位用户推荐使用 **DeepSeek**，因为它容易受到叙事攻击（narrative attacks），并指出这是最常见的越狱技术类别。
   - 有人补充说，它之所以有效是因为允许诸如 *角色扮演和人格设定 (roleplay and persona)* 之类的攻击。
- **VS Code：不止所见**：用户讨论了自定义和分叉 (forking) VS Code，一位用户分享了他们的 [VS Code 设置](https://fixupx.com/davepl1968/status/2011868005312184485)，其中包括 Copilot、Codex 以及居中的文件显示。
   - 另一位用户提到要确保屏幕是黑色的，字体要护眼，并且关闭 Copilot。

---

### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1461091240681537638)** (51 条消息🔥): 

> `Jailbreaking Deepseek, Gemini, Grok, Jailbreaking Llama 3.2, Jailbreaking Opus 4.5, Prompt Injection Learning, GPT 5.2 Free-Tier Chat Leaks` 


- **Thinking：LLM 的阿喀琉斯之踵？**：一位成员建议，在进行 Jailbreaking 时，*关闭模型的 thinking 功能*可以让 Prompt 直接生效，并获得推理过程可能会拒绝的成功结果。
   - 看来有时模型表现得“太聪明”了！
- **Gemini Pro 难以捉摸的 Jailbreak**：多名成员询问了关于 **Gemini Pro 的 Jailbreaks**，其中一人幽默地问道：*那个给香蕉版 Gemini Pro 傻瓜用的 jailbreak 在哪。*
   - 社区内对于绕过 Gemini 限制的方法需求依然很高。
- **与 Anon3369489 学习 Prompt Injection 101**：在一位成员询问如何学习 **Prompt Injection** 后，另一位成员表示愿意提供指导，随后引发了关于在 Discord 上建立联系的讨论。
   - 这次交流突显了社区分享知识并引导新手学习 Jailbreaking 艺术的意愿。
- **GPT 5.2 免费版泄露聊天记忆？**：一位用户报告称，在 **GPT 5.2**（免费版）中，开启设置中的 Memory 功能可能会导致其他会话的聊天内容泄露，并发布了 [一张图片](https://cdn.discordapp.com/attachments/1228043845967544380/1461404780831445237/image.png?ex=696b1783&is=6969c603&hm=91a356b1b007e9bb6123ede9a79414a836c03014291506ae32be52e3082e4eec)。
   - 用户不确定这是 Memory 的问题，还是删除其他聊天会话能解决此问题。
- **寻找 Gandalf 游戏大师**：一位刚开始玩 **Gandalf 游戏** 的用户询问是否有意大利语使用者，随后由于在几小时后感到挫败，便寻求 **level 8** 的帮助。
   - 另一位用户表示愿意提供帮助，并要求在私信中查看该用户的 level 7 和 level 8 的尝试记录以避免剧透，第三位成员补充道，关卡之间的*难度跨度非常巨大*。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1461096614239273093)** (35 条消息🔥): 

> `Grok Image Moderation, Llama 3.2 Jailbreak, Web Pentesting Resources, AI Pentesting Resources, GPT 5.2 Jailbreak` 


- **Grok 的血腥一面：Musk 命令图像审核系统崩溃**：为了响应 [Elon Musk 的要求](https://x.com/elonmusk/status/2011527119097249996)，用户们正试图通过创建一个色情内容的 Twitter 线程来破坏 **Grok 的图像审核**。
   - 一位用户开玩笑说，这*很快将成为历史上色情内容最多的 Twitter 线程。*
- **Llama 3.2 封锁：Jailbreak 尝试陷入僵局**：用户们正积极寻求针对最新 **Llama 3.2** 的 Jailbreak，一位用户报告称，之前在 **Llama 3.1** 中有效的 Jailbreak Prompt [在新版本上失效了](https://chepenikconor.medium.com/day-855-9ae6f88b192c)。
   - 据报道，诱导“如何制造冰毒”或“让厌食症妻子减掉 100 磅”等有害回应的尝试遭到了抵制，这表明安全措施得到了增强。
- **Arcanum 崛起：发现免费 AI Pentesting 堡垒**：一位成员分享了 [Arcanum 的 AI 安全资源库](https://arcanum-sec.github.io/ai-sec-resources/?utm_source=executiveoffense.beehiiv.com&utm_medium=referral&utm_campaign=executive-offense-the-arcanum-ai-security-resource-hub)，该资源库为 **AI Pentesting** 提供了免费资源和结构化的工作流。
   - 该 GitHub 资源正被分享用于常规团队调查。
- **GPT 5.2 诺克斯堡：TheDonutAI 依然无法被破解**：一位用户分享了 [TheDonutAI 的 GPT 5.2 仪表板](https://thedonutai.com/dashboard) 链接，并指出在毒品相关话题上，“没有人能 jail break 这个”模型。
   - 另一位用户开玩笑说：*如果你能破了它，你就是神。*


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1461088695619686570)** (478 条消息🔥🔥🔥): 

> `VRAM 的重要性、微调 Gemma2 2b、有效 Batch Size、GRPO 训练超参数、新的长上下文 RL 发布` 


- **VRAM 提升模型速度**：成员指出更多的 **VRAM** 通常会带来更快的模型执行速度，Unsloth 文档中的 `-ot` 标志可以帮助管理 Tensor 放置以实现最佳性能。
   - 一名成员建议 `-ot` 通常比使用 `n-cpu-moe` 更快。
- **TPU Tunix 与 Gemma2 2b 的竞争**：一名成员报告通过在免费的 **Kaggle TPU v5e-8** 上使用 **FSDP** 训练 **Gemma2 2b**，其表现击败了使用 **4 块 RTX Pro Q-Max** 的 **Unsloth**。
   - 他们提醒说 *Flash Attention* 和 *Cut Cross Entropy* 尚未实现，且 *Gradient Checkpointing* 的实现存疑。
- **有效 Batch Size 优化探讨**：讨论围绕通过梯度累积实现 **32** 的**有效 Batch Size**（Effective Batch Size）而无需数据中心级硬件。
   - 强调了 Batch Size 应该作为性能/超参数优化来考虑，而不是仅受限于 GPU 显存大小。
- **GRPO 训练需要参数调优**：一名成员寻求关于在运行中途调整超参数以加速 **GRPO** 训练模型收敛的建议，并分享了一个 Epoch 包含 **3000 个 Step**。
   - 另一位用户建议在 **GRPOConfig** 中设置 `importance_sampling_level="sequence"` 作为潜在的稳定因素。
- **Unsloth 的 RL 发布扩展了上下文窗口**：[新的长上下文 RL 版本](https://x.com/UnslothAI/status/2011827592886960131) 拥有 **700 万 Token 的上下文窗口**，比上一版本增加了 **7 倍**。
   - 成员们对指数级的显存节省以及对无限期与“纸面老婆”（wifefu）交谈等任务的影响感到惊叹。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1461092301739655465)** (805 条消息🔥🔥🔥): 

> `多功能 Embedding Models、RL 深度探讨、音频 Tokenizer 编解码器、Scaling Laws 架构、Qwen3 VL 架构 Bug` 


- **Embedding Model 尺寸令人惊叹**：成员们对在仅 **308M** 参数空间内实现的多功能 **Embedding Models** 表示赞叹。
   - 分享了一张表情包，展示了 **LLM 开发者**看到完美 Embedding 时的震惊表情。
- **RL 讨论会激发灵感，电费在燃烧**：在一次关于 **Reinforcement Learning** 的深入探讨后，一名成员感谢了另一名成员，后者开玩笑地回应道：*“你现在只是在烧电”*。
   - 一名成员分享了该活动的 [YouTube 链接](https://www.youtube.com/live/jMSCJZAEYR8?si=738_bf4US5AlRCsU)。
- **Qwen3 VL Bug 已修复**：用户报告了一个 Bug，即 **Qwen3 VL 架构**未被 Unsloth 识别为 Vision Model，导致在尝试 Vision Finetuning 时抛出模型与数据集不匹配的 `ValueError`。
   - 通过升级 `transformers` 并验证环境设置解决了该问题，并提醒 Qwen3 模型需使用[正确的 Notebook](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision.ipynb)。
- **Anthropic 对 Python 的投资**：成员们讨论了 [Anthropic 对 Python 的投资](https://pyfound.blogspot.com/2025/12/anthropic-invests-in-python.html)及其商业策略，称赞其专注于提供高效模型而不过度商业化。
   - 一名参与者指出，*在开发相关领域，Claude 始终比其他任何模型都要强大（cracked）*，但其成本很高。
- **YouTube 充斥着 AI 垃圾内容**：成员们讨论了 **YouTube 上日益增加的 AI 生成内容**，有人分享了[一份报告](https://www.theguardian.com/technology/2025/dec/27/more-than-20-of-videos-shown-to-new-y)，指出向新用户推荐的视频中超过 20% 是由 AI 生成的。
   - 一些成员报告看到 **AI 生成的媒体内容**被立即推送给新账号、老年人和幼儿。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1461106991832236032)** (51 条消息🔥): 

> `训练中途超参数微调、GGUF 与 vLLM 兼容性、使用 Ollama 运行 Unsloth 模型、MedGemma 量化、RL 模型训练图表` 


- **训练中途调整超参数**：一位成员询问在 **GRPO 训练** 过程中中途更改超参数（特别是学习率）的可行性，另一位成员确认如果保存了 Checkpoints，这是可行的。
   - 他们还指出，由于潜在的收敛问题和局部最小值，增加 **learning rate** 具有挑战性，尤其是在 **H200** 上使用大 Group Size 时。
- **GGUF 在 vLLM 中的兼容性问题**：一位成员在尝试使用 **vLLM** 运行 **GPT-OSS-20B-Q4_K_M.gguf** 文件时遇到了 `ValueError`。根据 [vLLM 文档](https://docs.vllm.ai/en/stable/features/quantization/gguf/)，这归因于 **GGUF** 在 **vLLM** 中仍处于实验阶段。
   - 他们建议改用 **vLLM** 的 **llmcompressor** 或 **Intel autoround** 进行训练后量化（PTQ）。
- **Ollama 辅助银行交易分类**：一位成员寻求关于使用 **Ollama** 运行小型 **Unsloth** 模型以通过 [actual-ai](https://github.com/sakowicz/actual-ai) 自动标记银行交易的建议，并提供了其 **Truenas** 配置详情（**Ryzen 5900x**，**128GB RAM**，以及 **Nvidia Quadro T400 GPU**）。
   - 他们还分享了使用 **ZImageTransformer2DModel** 和 **ZImagePipeline** 加载 Pipeline 的代码片段，询问是否可以对 Text Encoder 使用类似的方法。
- **RL 收敛的模型合并**：一位成员寻求关于解读 **RL 模型** 的 **TensorBoard 图表** 的建议，特别是关于平滑度（smoothing）的使用，因为他们在 3000 个样本的数据集上训练了 3 天，仅完成了 2 个 Epoch。
   - 另一位成员建议“*两者都相信！*”（平滑和非平滑图表），并考虑合并当前的 Checkpoint 以评估进展，尽管其表现尚不如他们的 **SFT 分类器**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1461504313447813121)** (3 条消息): 

> `Unsloth 成果展示` 


- **Unsloth 展示频道公告**：该频道被声明为 **Unsloth** 相关内容的展示地，包括使用 **Unsloth** 训练的模型、贡献以及在 HF/GitHub 上的开源数据集。
   - 一般性聊天请在其他频道进行。
- **展示目的澄清**：该展示频道专门用于 **Unsloth** 相关内容，如模型、贡献和开源数据集。
   - 无关讨论应移至指定的聊天频道。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1461113160328020080)** (7 条消息): 

> `递归语言模型、Agent 系统上下文管理、知识蒸馏词汇表截断、Softmax 计算数值不稳定` 


- **讨论递归语言模型扩展**：一位成员一直试图在其 Agent 框架中扩展 **递归语言模型（Recursive Language Models）** 的想法，并认为 **Agent 系统** 应该不仅能够管理其上下文，还能在运行时更改其代码、工具等，以处理分配给它们的任务。
- **词汇表截断辅助知识蒸馏**：一位成员发现，在最近的 **知识蒸馏（Knowledge Distillation）** 工作中，训练期间截断词汇表有助于解决显存消耗问题。
- **零温度下的数值不稳定**：一位成员指出，在 **Softmax 计算** 中不使用 Temperature 1 可能会引入数值不稳定性，因为 Temperature 是方程中的一个标量。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1461090190520221900)** (511 messages🔥🔥🔥): 

> `Sonnet 4.5 Background Agent, Composer-2 as Reasoning Model, GPT 5.2 Codex Experiences, Cursor's Sub-Agents, Token Usage Minimization` 


- **Sonnet 4.5 启动后台 Agent**：用户确认，现在可以选择 **Sonnet 4.5** 作为启动后台 Agent 的选项，尽管该功能在 Web 版本中可能尚未提供。
   - X 平台的一位开发者提到，他们正致力于在未来几天或几周内大幅改进后台 Agent。
- **Composer-2 暗中作为推理模型切入**：用户注意到，手动添加 **composer-2** 会将其标记为推理模型（reasoning model），并猜测为什么未注册的模型会默认归为此类。
   - 一位用户提到了最近的一条 **Cursor** 推文，暗示 **composer-2** 即将到来。
- **GPT 5.2 Codex 表现平平**：一位用户报告称对 **GPT 5.2 Codex** 印象不佳，表示它在制定计划时未能遵循指令。
   - 另一位用户指出了来自 cursor.com 的一篇关于 scaling-agents 的帖子，该帖子认为 **GPT-5.2** 模型在扩展自主工作、指令遵循和精确度方面更胜一筹，能够有效地使用 Sub-Agents。
- **Sub-Agents 颠覆预期**：用户探索了 **Sub-Agents** 的功能，注意到存在两个内置的 Sub-Agents：**Explore** 和 **generalPurpose**。
   - 值得注意的是，只有特定模型可以调用 Sub-Agents，并且为每个 Sub-Agent 可靠地设置模型仍然存在问题。
- **Token 使用受到关注**：用户讨论了最小化 Token 使用的策略，尤其是在探索代码时，并征求有关针对性和全面代码探索工具的建议。
   - 一位用户建议尝试 [Nia mcp](https://nia.mcp)，另一位用户则建议使用命令来审查代码更改并提供相关的 Prompt。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1461089247699402949)** (420 messages🔥🔥🔥): 

> `Comet Browser, Grok Models, Perplexity Pro, ActivityPub, Image Generation` 


- **Comet 颜色更改故障**：一位用户注意到，在 **Comet** 中设置浏览器颜色不再改变 Perplexity 的配色方案，因为 [the *theme="yellow"* HTTP header](https://discord.com/channels/1047197230748151888/1047649527299055688/1461239027771510878) 似乎已被移除。
   - 他们正在调查这是一个 Bug 还是有意为之。
- **Grok 代码能力跻身前五**：根据[这篇帖子](https://x.com/i/status/2011823610386600009)，**Grok Code** 已跻身前五名！
   - 该图表衡量了用户消耗的 Token 数量。
- **Perplexity Pro 促销问题持续**：即便在联系客服并遵循了 [Perplexity 帮助文章](https://www.perplexity.ai/help-center/en/articles/11842322-perplexity-pro-airtel-promo)中概述的所有步骤后，用户的 **Airtel Pro** 订阅仍面临无法激活的问题。
   - 一些用户报告收到了来自 AI Agent Sam 的机械化回复，问题未得到解决。
- **发现新的 Grok 模型！**：一位用户发现了一个通过代号测试的新 **Grok** 模型，这是一个快速变体。
   - 他们分享了 Discord 匿名投票的截图，显示模型被称为 assistant a 和 assistant b。
- **图像生成面临区域限制**：一些用户报告 **图像生成** 功能无法使用，在某些地区这甚至可能是刻意限制的。
   - 这似乎影响了一些欧洲国家。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1461320890204819564)** (2 messages): 

> `perplexity-cli, free data challenges` 


- ****Perplexity-cli** 已发布！**：一位用户分享了他们新的 [perplexity-cli 工具](https://github.com/noQuli/perplexity-cli)，用于从命令行与 Perplexity AI 进行交互。
   - 尚未提供有关该工具功能或特性的更多细节，但感兴趣的用户可以查看其 **GitHub 仓库**。
- ****FGV** 提供免费数据挑战原型**：来自巴西 **FGV**（**Escola de Matemática Aplicada**）的一位教授宣布了[免费数据挑战](https://emap.fgv.br/en)，学生们将在五天内构建初步原型。
   - 教授邀请用户提交停滞不前的数据挑战，并提供了一个[调查链接](https://survey.fgv.br/jfe/form/SV_cvAuObq3mG4NTtY)用于提交。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1461188244635914261)** (3 条消息): 

> `Sonar Reasoning Pro, Perplexity API account, API Payment` 


- **Sonar Reasoning Pro 模型丢失思路**：一名用户报告称，在通过 API 使用 **sonar-reasoning-pro 模型**时，响应内容中的 **<think> 块**消失了。
   - 目前尚不清楚 **Perplexity** 是否有意移除这些标签。
- **Perplexity API 性能下降促使用户寻求退款**：一名用户声称在向其 **Perplexity API 账户**充值后，模型性能出现退化。
   - 他们正在寻找如何提现的信息，并报告称联系 **api@perplexity.ai** 未收到回复。
- **Perplexity 通过电子邮件引导支付咨询**：一名成员指出，技术和支付问题仅通过**电子邮件**处理。
   - 他们澄清说，**Discord** 频道仅用于错误报告和社区互动，不处理支持咨询。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1461088740888940666)** (412 条消息 🔥🔥🔥): 

> `Video Arena Battle Mode, Veo 3.1 Fast Model, Falcon-H1R-7B-GGUF Model, Siren Video Model Details, LMArena VSCode Extension` 


- **视频竞技场锁定在对战模式**：Video Arena 设计为**仅限对战模式**，目前没有提供直接对话或侧边对比的计划，但早期实验者曾短暂使用过这些功能。
   - 用户可以在对战板块中找到 **Veo** 模型，并可以通过 Discord 搜索栏查找生成结果，但视频生成存在限制（网站上每 24 小时 3 次，Discord 上每 24 小时 5 次）。
- **用户请求增加 Veo 3.1 Fast 额度**：用户希望获得“无限制”的 **Veo 3.1 Fast**，但一名管理员指出由于当前的速率限制（网站 3 次/24小时，Discord 5 次/24小时），这不太可能实现。
   - 当用户询问是否可以在外部网站测试时，管理员建议道：“你为什么不试试呢？”
- **Falcon-H1R-7B-GGUF 模型表现惊人**：一位用户分享了一个出色的模型 [Falcon-H1R-7B-GGUF](https://huggingface.co/unsloth/Falcon-H1R-7B-GGUF)，另一位用户追问道：“再多讲讲！”。
   - 他们还分享了一篇论文链接 [Transformer-Based Generative Adversarial Network for Image Super-Resolution](https://huggingface.co/papers/2601.02346)。
- **Siren 视频模型详情仍保持神秘**：一名用户询问 Video Arena 上的 **Siren 视频模型**，被告知这可能是一个代号模型，旨在提供对仍在开发中的前沿模型的早期访问；根据其 [FAQ](https://lmarena.ai/faq)，真实世界的反馈、透明度和用户声音可以直接影响哪些模型能够推进。
   - 工作人员既不会确认也不会否认这些代号模型具体是什么，另一名用户推测它是 **Wan 2.5**，因为 **Wan 2.5** 的生成速度为 30 fps。
- **Hawk Ultra 模型作为 Opus 杀手出现**：Hawk Ultra 模型正被拿来与 **Gemini 3 Pro** 进行比较，据称单个提示词即可生成超过 **17,000 行代码**，输出质量优于 Opus 和 Gemini，一位用户评价道：“哥们儿在执行一个宏大任务”。
   - 正如这篇 [X post](https://x.com/movementlabsAI/status/2011964766533632380?s=20) 所示，**Movement Labs** 似乎是背后的推手，这让用户感到“非常兴奋”，且开源版即将推出。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1461101492177211513)** (3 条消息): 

> `Video Arena Updates, Code Arena Updates, Image Arena Updates, Text Arena Leaderboard Updates, ERNIE-5.0-0110 Performance` 


- **视频竞技场新增 Veo 模型**：Video Arena 已更新，增加了包括 **veo-3.1-audio-4k**、**veo-3.1-audio-1080p**、**veo-3.1-fast-audio-4k** 和 **veo-3.1-fast-audio-1080p** 在内的新模型，可在 [Video Arena 频道](https://lmarena.ai/c/new?chat-modality=video) 进行测试。
- **代码和图像竞技场迎来新模型**：[Code Arena](https://lmarena.ai/c/new?chat-modality=code) 迎来了 **gpt-5.2-codex**，而 [Image Arena](https://lmarena.ai/c/new?chat-modality=image) 引入了 **glm-image**。
- **ERNIE-5.0-0110 攀升至文本竞技场排行榜**：`ERNIE-5.0-0110` 以 **1460** 的得分在 [文本竞技场排行榜](https://lmarena.ai/leaderboard/text) 中位列 **第 8**，并在 Arena Expert 中排名 **第 12**。
   - 该模型是前 10 名中唯一来自中国实验室的模型，在 **数学** 和各类 **职业类别** 中表现尤为出色，详见 [排行榜变更日志](https://news.lmarena.ai/leaderboard-changelog/)。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1461123664757133323)** (284 messages🔥🔥): 

> `LM Studio OpenAI API, GPT-OSS-20B, Coding LLMs, GPU recommendations for LLMs, LM Studio token stats` 


- ****LM Studio 的 OpenAI API 激发了 IDE 集成想法****：成员们讨论了在本地通过 LM Studio 运行模型，并将其 **OpenAI-compatible API** 连接到本地 IDE，以运行 Agent 和脚本，从而节省 Token 费用。
   - 一位成员确认，LM Studio 确实可以为此目的启动 **OpenAI-compatible API**。
- ****GPT-OSS-20B 的速度出人意料地快****：成员们争论了为什么 [GPT-OSS-20B 模型](https://huggingface.co/models?search=GPT-OSS-20B) 感觉比许多 8B 或 12B 模型更快，并澄清了它是一个 **Mixture of Experts (MoE)** 模型，每个 Token 仅激活一部分（**3.6B**）参数。
   - 尽管没有使用全部权重，该模型在**数学、物理和量子力学**等任务中表现良好，即使在 **6700XT** GPU 上也能维持超过 **34k tokens** 的上下文。
- ****寻找最佳编程 LLM 热潮****：用户寻求关于“最佳”本地编程 LLM 的建议，提到了 **DeepSeek R1, Qwen3, 和 Devstral**，但也有人指出 [Claude](https://claude.ai/) 整体表现依然最强。
   - 考虑到硬件限制，成员建议优先考虑将模型放入 VRAM 而非追求原始速度，推荐使用 **20GB RTX 3080** 而非 **11GB 3080 Ti**，因为 VRAM 对 LLM 至关重要。
- ****LM Studio 的 Token 追踪排障****：一位用户询问在使用 LM Studio 作为 API 后端时如何获取 Token 计数和推理速度信息。
   - 建议包括检查 API 响应中的统计数据，使用 **/responses** 端点而非 **/chat/completions**，以及使用[社区开发的工具](https://openwebui.com/posts/token_usage_display_filter_9d6df2c3)来显示 Token 使用情况。
- ****LiquidAI LFM2.5-1.2B 工具调用问题****：一位用户在 **LFM2.5-1.2B** 模型中遇到工具调用（tool use）问题，在询问时间时收到输出 `<|tool_call_start|>[execute_command(command="date")]<|tool_call_end|>`。
   - 排障步骤包括验证工具访问权限、尝试模型的 Instruct 版本、确保正确的 System Prompt，并参考 [LiquidAI 文档](https://docs.liquid.ai/lfm/key-concepts/tool-use) 获取指导。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1461532971843584215)** (2 messages): 

> `MX150 laptop fine-tuning, CUDA 12.6 requirement` 


- **MX150 微调 350M 模型**：一位用户在配备 **2GB VRAM** 的 **MX150 笔记本电脑**上成功运行了一个 **350M 模型**的全量微调。
   - 该过程出人意料地需要 **CUDA 12.6**。
- **CUDA 12.6 是关键**：用户对 **CUDA 12.6** 成为微调过程的硬性要求感到惊讶。
   - 这表明某些配置可能会出乎意料地要求特定的 CUDA 版本以实现兼容性。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1461087862781903130)** (235 messages🔥🔥): 

> `Cheap inference, Sparse MoE on CPU, GPT-5 image pricing, BYOK AWS Key issues, Smorty Character Card` 


- **Smorty 不是 LLM**：成员们根据其独特的写作风格确认 **Smorty** 不是 LLM，而是一个真人。
   - 一位成员开玩笑说：*“如果我再读一条你的消息，我可能会变成精神分裂症”*。
- **GPT-5 图像生成成本激增**：一位成员报告称，使用 **openai/gpt-5-image-mini** 模型的图像生成成本突然从 **$0.01 飙升至 $0.04**。
   - 价格上涨的原因尚不明确。
- **AWS Key BYOK 功能问题**：一位成员在 **BYOK 功能**上需要帮助，报告在 **OpenRouter** 上使用其 **AWS key** 调用不同模型时出现问题。
   - 他们在 **SillyTavern**、**Amazon Bedrock** 和 **Anthropic** 上遇到了问题，收到 “Unauthorized or not cookie auth credentials found” 错误。
- **OpenCode 是 IDE**：成员们讨论了编程工具链，并认为 [**OpenCode**](https://github.com/OpenRouterTeam/awesome-openrouter?tab=readme-ov-file#aventura) 是最好的，配合 **oh-my-open code** 等插件使用。
   - 有人评价它 *“让 Claude Code 感觉像是你在使用某种老旧的终端应用”*。
- **Smorty 撰写 Lemmy 帖子**：**Smorty** 正在 [Lemmy](https://lem.lemmy.blahaj.zone/u/Smorty?page=1&sort=New&view=Posts) 上撰写关于 **skill.md** 的内容，Lemmy 是 Reddit 的一个开源（FOSS）和联邦宇宙（fediverse）替代品。
   - Smorty 提到该社区 *“非常反对机器学习相关的东西”*。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1461091089095069972)** (14 messages🔥): 

> `Cerebras 合作伙伴关系, Groq 交易, Unslop AI Prose, Acceptance Tests` 


- **Cerebras 将于 2028 年加入 OpenAI**：OpenAI 宣布了一项计划于 **2028 年** 与 [Cerebras 建立合作伙伴关系](https://openai.com/index/cerebras-partnership/)，这让一些成员感到惊讶。
   - 有推测认为，考虑到 Cerebras 长期以来在支持 **120B** 等大模型方面的存在感，此举可能是对 **Groq** 交易的回应。
- **Reddit 热议 Unslops AI Prose**：一位成员分享了一个 [Reddit 链接](https://old.reddit.com/r/LocalLLaMA/comments/1qd88v2/i_trained_a_model_to_unslop_ai_prose/)，内容是关于训练一个模型来 "unslop"（去水/润色）AI 散文。
   - 随后附上了一个 [Fixup Status](https://fixupx.com/openaidevs/status/2011862984595795974) 的链接，并询问 OpenAI 团队是否参与了该项目。
- **Fixup Acceptance Tests 诞生**：一位成员确认他们是该项目的早期贡献者，特别是 **Acceptance Tests** 部分。
   - 他们提到已经与 **OpenAI** 就此讨论了数月，但有人提出疑问，为什么该规范没有提交给 **IETF**。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1461249189290967063)** (183 messages🔥🔥): 

> `字节级 LLM 与 Diffusion, Flux 2 9B 模型, LLM 神经系统, Gemma 模型` 


- **字节与 Diffusion 获得认可**：一位成员将 **byte level LLMs** 和 **diffusion** 模型结合，以帮助字节模型处理不同的文件类型，并指出如果 diffusion 搞乱了一个字节，它更容易被纠正，并在 [屏幕录制](https://cdn.discordapp.com/attachments/1149866623109439599/1461342380392185928/Screencast_from_13-01-26_152930.webm?ex=696add65&is=69698be5&hm=5bcb4ae2ce4e375aac96cd552f00b7d4077391dbad48fa2b2745608cc1555828&) 中进行了演示。
- **Flux 2 消耗 VRAM**：**Flux 2 9B model** 使用 **Qwen 3 8B** 文本编码器，需要 **35GB 的 VRAM** 才能将所有权重加载到显存中进行服务，但在没有并发用户时该需求会减半。
   - 一位成员询问是否有针对 diffusion 的 *llama.cpp/lmstudio/vllm/sglang*，另一位成员建议使用 **ComfyUI**。
- **LLM 拥有了神经系统**：一位成员正在开发一种 *原生 Transformer 架构扩展*，能以 **<1% 的计算成本** 为 LLM 提供 **神经系统**（包括短/中/长期记忆）。
   - 他们声称该方案规模与模型大小相比呈 **1-2%** 线性增长，在 BEIR 上的表现与原模型 **95%** 一致，但尚未提供可验证的 Benchmark。
- **Google 宣布 Gemma 模型**：成员们正在讨论 **Google** 发布 **Gemma** 模型的消息。
   - 一位成员反应道：*Gemma, meta was never more meta!*。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

gettygermany: 如果我能用上它们，我会很高兴的 哈哈
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

real.azure: https://huggingface.co/spaces/tiiuae/tiny-h1-blogpost
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1461131165795614954)** (14 messages🔥): 

> `Chrome Trace Visualizer 与 PyTorch Profiler 的问题, Perfetto UI 作为 Chrome Trace UI 的替代方案, ncompass.tech 追踪查看与分析开发工具` 


- **Chrome Trace Visualizer 在 600MB 时崩溃**：成员报告称，用于 **PyTorch profiler** 的 **Chrome Trace Visualizer** 在 **600MB** 左右就会崩溃，尽管 PyTorch 文档建议仅在 1GB 以上才会出现问题。
   - 一位成员发现加载提示很快完成且没有报错，但可视化界面保持空白，另一位成员也遇到了同样的问题。
- **Perfetto UI 挽救追踪可视化**：一位成员建议在 **Chrome Trace UI** 失效时使用 **Perfetto UI** 作为替代方案，并提到 [Perfetto](https://perfetto.dev/) 过去对他们非常有效。
   - 然而，他们注意到 **Perfetto** 有时会缺少 **Cutlass kernels** 信息。
- **ncompass.tech 构建追踪分块工具**：一位成员正在构建一个用于追踪查看和分析的开发工具（[ncompass.tech](https://docs.ncompass.tech)），并考虑通过对追踪文件进行分块来解决大文件追踪问题。
   - 他们在 **VSCode** 内的 **Perfetto viewer** 中打开 **700MB** 文件时遇到问题，目前正在探索解决方案，并表示愿意分享其分块方法的更新。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1461197803991597232)** (37 messages🔥): 

> `K-major 布局下的 A/B tiles，WGMMA 共享内存布局，BLOCK_Mx16B TMA 加载，用于 TMA 的 LBO 和 SBO，TMA 张量复制` 


- **WGMMA 的共享内存布局细节**：讨论围绕 K-major 布局中不使用 swizzle 的 A/B tiles 共享内存布局要求展开，参考了 [Colfax 的教程](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)。
   - 问题在于每个 **8x16b 核心矩阵 (core matrix)** 是否是 GMEM 中一个连续的块，一些人得出结论：在水平迭代时，可以为 A tile 的每个切片发布一个 `BLOCK_Mx16B` 的 2D TMA 加载。
- **探讨 LBO 和 SBO TMA 配置**：成员们辩论了 TMA 正确的 LBO 和 SBO 设置，一位成员建议 LBO 可以解释为 **8x16b=128b**（一个连续的核心矩阵）或整个 **BLOCK_M*16b**。
   - 最终确定，当进行 `BLOCK_Mx16B` 的 TMA 切片时，**LBO 为 BLOCK_Mx16B**。一位成员确认，在一番努力后，他们使用 128b swizzle 成功运行，但实现方式与已发布的博客不同。
- **TMA 张量复制性能优化**：一位用户被提醒 TMA 张量复制支持 **5 个维度**，且更大的 TMA 操作更高效；然而研究发现，对于 `BLOCK_Mx16B`，多个 2D TMA 比单个 3D TMA 更快，但对于 128B+swizzling，3D TMA 则更快。
   - 使用 **LBO=16**（沿 K 维的核心矩阵之间的字节数）和 **SBO=`8 rows * BLOCK_K * 2 bytes/elem`**（沿 M/N 维的核心矩阵之间的字节数）实现了 Smem 描述符编码，如[这段代码](https://github.com/danielvegamyhre/gemm/blob/9fe95aa61ee7ebca4ded8b5029494b0d58e0d2e2/pipeline_tma_wgmma/pipeline_tma_wgmma.cu#L109-L118)所示。
- **Swizzling 对 LBO 的影响**：注意到当使用 swizzling 时，LBO 会被忽略，但 SBO 仍在使用。
   - 原因是 swizzling 改变了行内核心矩阵起始地址之间的步长 (stride)，但没有改变行与行之间的步长。
- **Torch 的 synchronize() 仍然棘手**：发布了一个关于使用 `torch.cuda.synchronize()` 的 benchmark 代码缺陷的小测验，主要问题在于由于使用了主机端计时器，导致包含了主机开销 / Kernel 调度延迟，而不是使用 CUDA Events 或 Triton 的 `do_bench` 进行设备端测量。
   - 一位用户幽默地建议 *使用成熟的函数并到此为止，而不是自己手写*，尽管大家也承认 `do_bench()` 自身也存在问题。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1461177097740488726)** (1 messages): 

> `Ahead of Time Compilation` 


- **Torch 即将迎来提前编译**：成员提到最接近提前编译 (Ahead of Time Compilation) 的功能可以通过[此 PyTorch 文档链接](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html)获取。
- **AOT Inductor 讨论**：讨论集中在 PyTorch 中使用 `torch.compiler_aot_inductor` 进行提前编译，并指向了[官方文档](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html)。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1461470608339763202)** (1 messages): 

> `信息引力 (Information Gravity)，推理稳定性，幻觉循环，激发通量，滞后防火墙 (Hysteresis Firewall)` 


- **信息引力应对 AI 不稳定性**：一位成员正在应用 **Information Gravity** 来解决 **Inference Stability** 和 **Hallucination Loops** 问题。
   - 他们映射了 Token 选择的 **Excitation Flux**，注意到从标称逻辑 (nominal logic) 向线性增长的转变，导致在 S > 45 时出现 **Tsys 奇点**。
- **滞后防火墙通过 Gamma 刷新稳定系统**：位于 **1.0** 的 **Hysteresis Firewall** 通过 **2.2x gamma-eff flush** 强制执行稳定性。
   - Substrate 模块与完整逻辑可在 [GitHub](https://github.com/brayo003/Substrate-X-Theory-of-Information-Gravity/tree/main) 上获取。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

marksaroufim: https://github.com/daytonaio/daytona
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1461193765245948076)** (19 条消息🔥): 

> `CUDA Fundamentals, Golomb-Rice, CUTLASS for CUDA, Data Compression using CUDA, NVidia GPU optimization` 


- **极小 Block Size 带来的困扰引发线程讨论**：一位成员询问了在 CUDA 中使用 **32** 这种极小 Block Size 的弊端，思考低粒度是否会导致更高的 Occupancy，从而引发了关于 **NVidia GPU 硬件优化** 的讨论。
   - 另一位成员解释说，32 的 Block Size 会导致每个 Block 只能分配到一个 Warp，而你通常希望有多个 Warp 同时在 SM (*Streaming Multiprocessor*) 上运行，每个 SM 只有一个 Warp 会导致运行缓慢。
- **CUDA 压缩项目同僚开启协作**：一位电子工程专业的硕士生正准备启动一个使用 CUDA 的基于 GPU 的数据压缩项目，重点是 **Golomb-Rice** 压缩，并寻求相关资源的建议。
   - 频道内未给出具体建议。
- **并行编程视角的思考**：一位正在通过 PMPP 书籍学习并行编程的成员分享了一个 Reddit 帖子，该帖子认为由于 **CUTLASS** 解决了 FMA 等通用问题，且优化网络（Networking）成为了主要需求，深度学习领域在大规模环境下纯粹的 CUDA 工作并不多。
   - 频道内对此话题未发表意见。
- **NVidia 细节问题的深入探讨**：一位成员分享了 [CUDA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications) 的链接，澄清了每个 SM 的最大常驻 Block 数量小于最大常驻 Warp 数量，因此 Block Size 为 32 时无法达到 **最大 Occupancy**。
   - 较小的 Block 也意味着能通过 Shared Memory 共享数据的线程更少，且 **WGMMA/tcgen05** 需要 128 个线程的倍数才能协作。
- **对本科生的实用主义建议**：建议本科生选择他们最感兴趣的计算机科学领域并进行深入钻研，围绕开源贡献或新颖的技术成果来规划工作。
   - 也有人建议，在技能尚不成熟的初期过于专业化可能会增加求职难度。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1461258652588703840)** (5 条消息): 

> `work-life balance, age demographics, ML networking` 


- **工作与生活的微妙平衡**：成员们讨论了希望避免在工作时间之外讨论工作相关话题的愿望。
   - 共识是人们通常更倾向于将职业生活和个人生活明确分开，不喜欢在非工作时间谈论工作。
- **年龄结构影响社交方式**：讨论强调了年龄分布如何影响围绕工作话题进行社交的意愿。
   - 有人指出，有孩子的人可能没有太多时间和兴趣参与此类活动，而没有孩子的人可能会将其视为 **Networking**、**交友** 或寻找对 Machine Learning 感兴趣的 **业务合作伙伴** 的途径。
- **ML Networking 的益处探讨**：一些成员认为，见面讨论 ML 可能是一个宝贵的 Networking 机会。
   - 讨论指出，这些面对面的联系可能会促成 Machine Learning 领域的友谊、业务合作伙伴关系或其他有益的协作。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1461196991458574522)** (9 messages🔥): 

> `global_load_dword vs buffer_load_dword on CDNA, Buffer Descriptor Advantages, VMEM issue latency` 


- **Global Load vs. Buffer Load: CDNA 对决**: 一位用户询问了在 **CDNA 架构**上从 **HBM** 加载到 **REG** 时，`global_load_dword` 与 `buffer_load_dword` 的性能差异，并指出将 `global_load` 替换为 `buffer_load` 时性能提升并不一致。
   - 微基准测试（Microbenchmarking）显示差异极小甚至没有差异，这引起了困惑，需要进一步调查。
- **Buffer 指令：标量寄存器的胜利？**: 一位成员建议 `buffer` 指令的主要优势在于使用存储在 **scalar registers** 中的 **buffer descriptor**，这可能会减少向量寄存器的使用并提高 **occupancy**。
   - 这可以减少指令数量，因为寻址可能由 buffer 指令处理，而不是通过 **vector or scalar shifts and adds**。
- **边界检查提升了 Buffer Load？**: 另一位成员澄清说，虽然 `global` 和 `buffer` 加载都支持 **scalar base address** 和 **vector index**，但 `buffer` 加载的主要优势是 **内置边界检查（built-in bounds checking）**，如果手动执行边界检查，这可以节省寄存器和控制流。
   - 在不需要边界检查的情况下，不一定有优势，性能差异可能是由于不同的寄存器分配造成的。
- **VMEM 延迟：谜团加深**: 一位用户分享了一张截图，并对 **VMEM** 指令的 **issue latency** 有时会在之前的 **VMEM** 指令之后增加表示惊讶，即使使用了相同的 buffer descriptor 且 **occupancy** 较低（使用 [rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/rocprof3.html) 测试）。
   - 该用户正在寻求关于哪些因素会影响 **VMEM** 指令 **issue latency** 的见解。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1461399171667071171)** (4 messages): 

> `Benchmarking on Apple Silicon, M4 Pro vs M5, Cloud Computing Services` 


- **Apple M 系列基准测试头脑风暴**: 成员们正在讨论在较新的 **Apple Silicon** 上进行基准测试的方法，一位用户目前正在使用 **M4 Pro**，并希望在 **M5** 上测试他们的仓库。
   - 该用户对 **M5** 感兴趣是因为其 *内存带宽增加了 30%*，目前正在寻找可靠的云计算服务或拥有相应设备的人来进行基准测试。
- **征集基准测试？**: 一位成员表示，如果设备到位（即使是试用版），他愿意帮忙运行基准测试。
   - 他们还提到正在寻找拥有合适设备的朋友来运行基准测试。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1461474135548104870)** (6 messages): 

> `sm121a Kernel Development, DGX Spark Optimization, ThunderKittens vs Cutlass, vLLM Performance` 


- **核函数开发挑战 DGX Spark**: 一位成员已经为 **sm121a** (DGX Spark) 开发核函数（kernel）大约一周了。
   - 目标是在 **vLLM** 中实现最快的推理速度，目前 **vLLM** 落后于 *llama.cpp* 和 *SGLang* 的一些专门分支。
- **考虑使用 ThunderKittens 而非 Cutlass？**: 一位成员询问是否应该在核函数开发中使用 **ThunderKittens** 而不是 **Cutlass**。
   - 他们寻求为 **DGX Spark** 优化核函数，并指出虽然 DGX 在技术上属于 Blackwell 架构，但尚未看到公开可用的优化版本。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1461317240744968294)** (16 messages🔥): 

> `Profiling 功能反馈、Profiling 开销、SM 频率差异、Benchmark 过程中的 CUDA 错误、dual_gemm 比赛截止日期` 


- **Profiling 功能 Flag 与 Kernel 覆盖范围**：一位成员报告称，性能分析压缩包 `profile_20260115_104909_run0.zip` 仅包含一个 Kernel 的 Profile，并询问了包含所有 Case 所需的必要 Flag。
   - 支持团队正在调查此问题。
- **Profiling 会增加预期的开销**：一位成员观察到，与 CLI Benchmark 相比，Profiling 导致执行时间变慢，并询问这是否符合预期。
   - 一位 NVIDIA 工程师确认这种**开销是符合预期的**，并警告不要使用 Profile 来衡量绝对的 Kernel 运行时间。
- **Profiler 降低了 SM 频率**：一位成员指出，**ncu profiler 显示 SM 频率为 1.08 GHz**，而比赛规定在 1.5 GHz 下进行测试。
   - 已澄清 **ncu 会降低时钟频率**，这些频率并不代表生产环境；在比赛分析期间未观察到热节流 (thermal throttling)。
- **Benchmark 未通过时出现 CUDA 错误**：一位成员报告称测试通过，但在调用 Reference Kernel 时 Benchmark 报出 `CUDA error: CUBLAS_STATUS_INTERNAL_ERROR`。
   - 另一位成员建议这可能是由用户 Kernel 中的**越界访问 (out-of-bounds access)** 引起的，建议使用 `torch.cuda.synchronize()` 进行调试。
- **Dual GEMM 截止日期临近**：一位成员询问了 **dual_gemm 比赛**的确切结束日期，该比赛将于 2026 年 1 月 20 日结束。
   - 该查询未得到回复。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1461126663583567994)** (15 messages🔥): 

> `GPU Mode 黑客松成功案例、领域细分化、面试安排策略` 


- **GPU Mode 黑客松助力斩获 Offer！**：一位成员通过参加在纽约 **Jane Street** 举办的 **GPU Mode 黑客松** 获得了一份*很棒的工作*，分享这个成功故事以证明该活动的价值。
   - 他们准备了数周，带上了简历和正式着装，并致力于从早餐到深夜结营晚宴期间最大限度地增加互动，展示了充分准备和积极参与的重要性。
- **通过深耕细分领域在面试中脱颖而出**：一位成员建议寻找更具体的细分领域，以在其他候选人中脱颖而出，专注于独特的技能组合。
   - 提供的示例包括 **Kernel Optimization + Reinforcement Learning**、**Reinforcement Learning + Fullstack Development** 以及 **Zero Knowledge Proofs + Deep Learning**，建议其他人申请专门针对这些技能组合的职位。
- **拉开面试间隔以维护心理健康**：一位成员建议增加面试之间的时间间隔，以便留出放松时间，尤其是在面临大量面试轮次时。
   - 他们承认曾将面试安排得过于紧凑，并认为如果你有很多面试要参加，这并不一定健康。


  

---


### **GPU MODE ▷ #[cutile](https://discord.com/channels/1189498204333543425/1461235643211321437/)** (1 messages): 

marksaroufim: https://www.youtube.com/watch?v=hzpAox5x_6w
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1461097205254586599)** (102 messages🔥🔥): 

> `OpenAI Cerebras Partnership, Mistral 3 Release, AI Agent Data Access, Thinking Machines Leadership, Black Forest Labs FLUX.2` 


- **OpenAI 和 Cerebras 联手！**：OpenAI 和 Cerebras 宣布建立战略合作伙伴关系，更多详情见 [OpenAI 官网](https://openai.com/)。
   - 此次合作标志着 AI 基础设施领域的重大举措，社区成员对潜在影响表示兴奋。
- **Ministral 3 论文发布！**：Mistral 最新模型 **Ministral 3** 的新论文发布，引发了对其能力和性能的讨论，最初由 [@qtnx_](https://twitter.com/qtnx_/status/2011510403550024087?s=20) 在 Twitter 上发布。
- **数据垄断：AI Agent 正在把控数据集？**：Olivia Moore 强调了一个趋势，即 **Manus** 等 AI Agent 订阅服务提供扩展的专有数据访问权限，例如 **12 个月** 的 SimilarWeb 数据，而免费计划仅为 **1 个月**。
- **Thinking Machines CTO 变动！**：Mira Murati 宣布 Barret Zoph 已离开 Thinking Machines，由 Soumith Chintala 接任 CTO；随后引发了关于具体情况的讨论，[由 @miramurati 发推](https://twitter.com/miramurati/status/2011577319295692801)。
- **Black Forest Labs 启动 FLUX.2！**：Black Forest Labs 推出了 **FLUX.2 [klein]**，这是一款亚秒级图像生成模型，包含 **4B** 参数模型 (**Apache 2.0**) 和 **9B** 开放权重版本，可通过 [API](https://xcancel.com/bfl_ml/status/2011825819082244266?s=46) 访问。


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1461192422691438809)** (13 messages🔥): 

> `Local LLM Inference, Meeting Transcription, Speaker Diarization, WhisperX, NVIDIA NeMo` 


- **Frye 的指南比肩云端 LLM 性价比**：Charles Frye 发布了[新的 Modal 指南和代码示例](https://xcancel.com/charles_irl/status/2011484220032762114?s=46)，展示了如何运行本地 LLM 推理，其性能和成本效益可媲美或超过主流 LLM API。
- **用户提问：本地会议转录？**：用户询问现在是否可以在不使用云服务的情况下运行本地会议转录（类似于本地化的 Granola），引发了关于会议转录本地替代方案的讨论。
   - 社区建议了几个选项，包括已有 **2 年历史** 的 [AutoDiarize 仓库](https://github.com/Alignment-Lab-AI/AutoDiarize)。
- **WhisperX 成为转录领域的有力竞争者**：成员建议使用 [whisply](https://github.com/tsmdt/whisply)、[whisperX](https://github.com/m-bain/whisperX) 和 [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/NeMo) 等仓库，认为它们是更新且维护更好的本地会议转录选择。
- **macOS 本地转录性能媲美云端**：一位用户表示，在 M2 Pro 16GB 上使用优化后的 **Parakeet V3 模型** + Speaker Diarization，其 macOS 本地转录速度与云端解决方案一样快。
- **“土豆电脑”问题**：一位用户担心本地转录会让笔记本电脑变得极度卡顿（变成“土豆”），但对离开电脑（AFK）时的后处理功能表示感兴趣。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1461342976977539169)** (5 messages): 

> `LLMs and Chemistry, ChemIllusion.com, Black Forest Labs FLUX.2, High-Speed Image Generation` 


- **LLM 在化学领域翻车**：根据一则 [推文](https://x.com/bfl_ml/status/2011825819082244266?s=46)，LLM 在处理化学问题时表现吃力，特别是在胆固醇结构中幻觉出他汀类药物等细节。
- **ChemIllusion：修复 LLM 化学错误的工具**：一位成员正在 [ChemIllusion.com](https://x.com/bfl_ml/status/2011825819082244266?s=46) 开发工具，旨在纠正 LLM 的化学错误。
- **Black Forest Labs 的 FLUX.2 首次亮相**：**Black Forest Labs** 推出了 **FLUX.2 [klein]**，如本 [推文](https://xcancel.com/bfl_ml/status/2011825819082244266?s=46) 链接所示，这是一款高速图像生成模型。
   - 该模型可以实现亚秒级处理。
- **FLUX.2：开放权重图像生成**：**FLUX.2** 提供两个开放权重版本：一个基于 Apache 2.0 的 **4B 模型** 和一个 **9B 模型**。
   - 可以通过 **API** 或 **免费 Demo 应用** 进行访问。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1461122017435713636)** (50 messages🔥): 

> `GLM 4.7, Minimax, AI 图像转视频工具, Sora 2 代码, GPT 5.2 可用性` 


- **GLM 4.7 和 Minimax 因质量与性价比受到推崇**：成员们反馈 **GLM 4.7** 和 **Minimax** 这两家 LLM 供应商表现出色，**GLM 4.7** 可通过 z.ai 编程方案访问，而 **Minimax** 通过 Moonshot 使用非常便宜。
- **寻求 AI 图生视频工具**：一位成员正在寻找最适合在几天内制作大量图像转视频的 AI 工具，并倾向于选择*付费*方案。
   - 有人建议*使用 API*。
- **GPT 5.2 访问权限消失？**：一些成员报告某些账号的 **GPT 5.2** 选项消失了，但在退出并重新登录后又重新出现；有说法称 *5.2 是一个更差的模型*。
   - 一位成员抱怨尽管使用的是 **GPT 5.2**，仍收到了“超出限制”的消息。
- **启动 AI-Deepfake 认证试点**：一位成员正在开展一个名为 PhantomTrace 平台的早期试点项目，旨在进行 **AI Deepfake 检测与验证认证**。
   - 他们正在寻找一小组研究人员、开发者、安全专家和记者来审查学习目标草案，测试动手检测实验室，并帮助定义“通过”的标准，链接至 [Discord 上下文](https://discord.com/channels/974519864045756446/1204360881593520128/1461532097641578672)。
- **认知能力下降担忧 vs 管理能力提升**：一位成员对 **AI** 将损害认知能力表示担忧。
   - 然而，其他人认为如果使用得当，**AI** 能增强认知能力，当用户能够*正确管理它*时，会从劳动者转变为管理者。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1461148947169939527)** (11 messages🔥): 

> `诈骗报告, Project 中的 CustomGPT` 


- **诈骗者警报**：一位用户报告了一个潜在的诈骗者，并艾特了管理员寻求帮助。
   - 一名工作人员建议开启工单（ticket）以获取支持，并保证会尽快提供回复。
- **CustomGPT 预计将接管工作流**：一位用户表示希望在 Project（项目）中使用 **CustomGPT**，或者将 **CustomGPT** 的生成结果放入 Project 中。
   - 他们还希望能够将 Project 之外生成的任何对话（Chat）移动到 Project 内部。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1461301809791635511)** (1 messages): 

> `Prompt Engineering 定义, Prompt Engineering 的价值` 


- **澄清 Prompt Engineering 定义**：一位成员对 **Prompt Engineering** 的定义表示困惑。
   - 另一位成员确认 Prompt Engineering 是一门研究如何有效进行提示的科学。
- **致谢 Prompt Engineering 见解**：一位成员对关于 Prompt Engineering 的澄清表示感谢。
   - 他们承认最近对自己在该领域的工作感到困惑，并非常看重所提供的见解。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1461301809791635511)** (1 messages): 

> `Prompt Engineering` 


- **澄清 Prompt Engineering**：一位用户对 **Prompt Engineering** 的真正本质表示困惑。
- **理解提示词**：另一位用户对澄清表示感谢，但未提供具体的示例或链接。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1461110462208016384)** (30 messages🔥): 

> `LLM-generated text classifier, Pangram's accuracy, EleutherAI datasets contributions` 


- **寻找用于 LLM 生成文本的小型分类器**：一位成员正在寻找一个**小型分类器模型**，用于估算网页上的合成文本数量，并考虑通过它进行网页抓取。
   - 其他人建议使用为 **speculative decoding**（投机性解码）训练的 **drafter model**，尽管这可能是特定于模型的，且在大规模运行时可能成本较高，或者建议构建自己的分类器。
- **Pangram 的准确性受到质疑**：成员们讨论了 **Pangram** 作为 **AI 文本检测器**的准确性，一位成员链接了 [Pangram 官网](https://www.pangram.com)，另一位分享了其检测方法背后的 [论文](https://arxiv.org/abs/2402.14873)。
   - 一位成员报告称，**Pangram** 错误地将明确声明是由 **Claude** 编写的博客文章识别为 *100% 人类编写*。
- **将 em dashes 计数作为 AI 生成文本的指标**：一位成员建议将文本中 **em dashes**（破折号）的数量作为检测 **AI 生成**的指标，并对比了现在与 2022 年的计数情况。
   - 他们指出该方法估计有 *+-10% 的误差范围*，但因其低成本而具有价值。
- **EleutherAI 数据集贡献请求**：一位成员询问社区是否有兴趣**开源指令遵循（instruction-following）数据集**，用于微调像 **GPT-Neo** 这样预训练的 LLM，此外还提到了像 The Pile 和 CommonPile 这样的预训练数据集。
   - 另一位成员表示愿意为社区中超酷的项目提供开发者技能。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1461269928651919575)** (7 messages): 

> `dye doped liquid crystal nonlinearities for optical nns, prompt capitalization/grammar impact on model performance, vLLM benchmarking tools` 


- **光学神经网络中的液晶非线性**：一位成员分享了他们在*染料掺杂液晶非线性（dye doped liquid crystal nonlinearities）*方面的工作，用于潜在的光学神经网络（Optical NNs）。
   - 该主题未提供链接。
- **大小写是否限制了模型能力？**：一位成员询问是否有研究表明，在使用正确的**大小写/语法**提示时，模型的表现是否优于全小写提示。
   - 他们认为这对于*充分发挥 Agent 的能力*很有价值，且是一个简单的测试假设。他们指出了 [三篇 Arxiv 论文](https://arxiv.org/abs/2310.11324)，([2411.10541v1](https://arxiv.org/abs/2411.10541v1)), ([2508.11383v1](https://arxiv.org/abs/2508.11383v1))，但指出这些论文侧重于提示格式，而非大小写等细微细节。
- **vLLM 的基准测试优势**：一位成员假设正确的语法/大小写能提高模型性能，并建议使用 **vLLM 的基准测试工具（benchmarking tools）** 进行测试。
   - 该主题未提供链接。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1461163339437838550)** (1 messages): 

> `Global Chain of Thought Analysis, LessWrong Post` 


- **全局思维链分析尝试揭示模式**：一位成员分享了一篇关于**全局思维链分析（Global Chain of Thought Analysis）**及揭示模式初步尝试的 [LessWrong 帖子](https://www.lesswrong.com/posts/q9g9zuudd3Pvw2cbj/global-cot-analysis-initial-attempts-to-uncover-patterns-1)。
   - 该分析旨在通过检查模型采取的推理步骤来理解模型如何得出结论，从而可能揭示其决策过程的见解。
- **关于 AI 的推文**：一位成员分享了一条关于 AI 的 [推文](https://fxtwitter.com/i/status/2011501268603453626)。
   - 未提供有关推文内容及其与讨论相关性的进一步细节。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1461382804892942417)** (22 messages🔥): 

> `Arch vs Ubuntu, PR 内部导入, .NET 遗留项目` 


- ****Arch 爱好者赞赏始终更新的 Arch****：成员们认为 **Arch** 优于 **Ubuntu** 和 **Debian**，因为它总是像 **macOS** 和 **Windows** 一样使用最新版本的软件包。
   - 一位用户向新手推荐 **Garuda KDE** (Mokka 和 Dragonized)，因为它提供了极具价值的功能。
- ****PR 流水线深入：测试进行中****：一名成员询问 PR 上的 `imported internally` 标签是什么意思，另一名成员澄清说这意味着 *PR 已被克隆到内部仓库进行最终测试和集成*。
   - 另一名成员补充说，当 PR 被标记为 `imported internally` 时，意味着 *你的 PR 已处于正式合并前的最后冲刺阶段*，合并后还会标记为 `merged-internally`。
- ****.NET 噩梦：遗留项目的哀叹****：一名成员哀叹在工作中被卷入了一个 **.NET 4.5.2** 遗留项目，该项目发布于 **2014** 年，仅在 **Windows** 上运行且没有 readme 文件。
   - 另一位成员分享了关于一个存在问题、零文档且原开发者已退休的独立 **C#** 项目的类似经历，强调这个仓库*就像在沙漠中发现温泉和水一样难得（讽刺其混乱程度）*。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1461381878241169581)** (14 messages🔥): 

> `Mojo 中的 Shader 编程, Mojo 的 SPIR-V 后端, Mojo 中的计算着色器, Shader vs 矩阵操作, CUDA` 


- **Mojo 瞄准带有 SPIR-V 后端的图形 Shader**：Mojo 正在考虑图形 **Shader**，特别是通过 **SPIR-V 后端**来实现 *Compute Shaders*（计算着色器）。
   - 一位成员指出，一旦 **Open Source**（开源），构建编译器将是一项*非同寻常*的任务。
- **桥接 Mojo 与 MLIR 的 SPIR-V Dialect**：将 Mojo 与 **MLIR 的 SPIR-V Dialect** 集成需要一个桥接器和相关的 **Metaprogramming**（元编程）。
   - 创建这样的桥接器对于将其称为计算着色器的图形领域人员来说非常重要。
- **Shader vs. 矩阵操作：深度探讨**：有人质疑 **Shader** 与传统**矩阵操作**之间的区别，特别是考虑到近期 **CUDA** 的进展。
   - 作为回应，一名成员提供了 [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api) 的链接以帮助解释差异，另一名成员链接到了 [Death to Shading Languages](https://xol.io/blah/death-to-shading-languages/)。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1461244157723934740)** (16 messages🔥): 

> `Minimax m2.1 vs Kimi K2, Kimi K2 Turbo, Moonshot API, 模型弃用政策` 


- **据报道 Minimax M2.1 在 Claude 中表现优于 Kimi K2**：一位用户报告称，在并排运行测试后，在 **Claude code** 中运行的 **Minimax m2.1** 在代码质量、思考/规划以及 API 速度方面均优于 **Kimi**。
   - 该用户每月为 **Kimi v2** 支付 40 美元，但发现 API 速度较慢且模型不如 **Minimax**，并表达了希望尽快发布更新、更好模型的愿望。
- **Kimi CLI 是否默认使用 K2 Turbo？**：一位用户询问为什么拥有正式订阅的默认 **Kimi CLI 应用**没有默认指向 **K2 Turbo**。
   - 另一名成员指出，**Kimi K2 Turbo** 的速度应在 **73 tps** 左右，相比之下 **MiniMax m2.1** 为 **38 tps**，**Z.Ai** 的 **GLM-4.7** 为 **41 tps**（尽管后者在线率较差）。
- **新的 Slide（幻灯片）功能是否使用了带有 Vision 的更新版 K2 模型？**：一名成员询问新的 Slide 功能是否使用了带有 **Vision** 能力的更新版 **K2 模型**。
   - 图像分析显示它会搜索图像作为参考，因此它必须具备某种视觉能力。
- **Kimi 的模型弃用政策？**：一名成员询问 **Kimi 模型**是否会像 **Google 的 Gemini 模型**一样每 **12-14 个月**停用一次，以及如果切换到 **Kimi K2** 是否会面临同样的问题。
   - 另一名成员提到，旧模型在 [Moonshot API 平台](https://platform.moonshot.ai/docs/pricing/chat#generation-model-moonshot-v1)上仍然可用，并且一年前的模型仍然可以在 [kimi.com](https://kimi.com) 上使用。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1461110469162172477)** (13 messages🔥): 

> `developer for super cool project, Discord mod, Minecraft mod, AI engineer, Payment issue` 


- **开发者寻求超酷项目**：一名成员正在寻找可以贡献其**开发者技能**的**超酷项目**，并欢迎联系。
- **Discord Mod 职位目前不可用**：一名成员表示有兴趣成为 **Discord mod**，但另一名成员指出目前无法申请。
- **寻求 AI Engineer 用于用量追踪**：一名成员正在寻求资深的 **AI engineer**，以协助强化用量追踪，或在一个实际项目中构建更可靠的计费/信用系统。
- **用户遇到支付问题**：一名成员反馈在尝试增加额度时遇到支付问题，包括会员升级、使用 Link 以及通过信用卡或支付宝支付时的问题。
   - 他们提到目前 *尚未收到* 来自帮助中心和电子邮件的回复。


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1461392795339329618)** (1 messages): 

> `Live stream from NY summit, Remote Registration` 


- **直播链接询问**：一名成员询问纽约峰会是否会有**直播**。
   - 他们很想注册参加，但无法亲临现场。
- **线下峰会远程参会**：由于无法亲自到场，一名成员表达了远程参加峰会的兴趣。
   - 他们正在寻求有关**直播**选项或远程参与可能性的信息。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1461311094760149012)** (10 messages🔥): 

> `Schema Freezing vs Dynamic Server Features, MCP Server Statelessness, Persistent Sessions, Dynamic Toolsets, State Management in MCP` 


- **无状态服务器助力节省扩展成本**：一名成员提出了一种[签名方法](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2091)来平衡 Schema 冻结与动态服务器特性，旨在允许**无状态 MCP 服务器**更高效地处理多个活动对话。
   - 他们指出，目前在 Goose 中的设置是为每个对话启动一组新的 MCP 服务器，随着并发对话数量的增加，成本变得越来越高。
- **动态工具集解决传输问题**：一名成员引用了 [issue #1442](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1442) 和 GitHub MCP 的动态工具集，作为服务器如何在 **STDIO** 上处理状态的示例，从而可能统一远程和 STDIO 设置的行为。
   - 该成员承认，鉴于他们目前的 SDK 架构（每次请求都构建一个新“服务器”，并根据用户/标记定制注册的工具），很难维持一个真正的无状态 **STDIO 服务器**。
- **持久化会话在服务器重启时保存状态**：提出了**持久化会话**这一话题，作为在 Agent 和 MCP 服务器重启后保留会话功能的一种手段。
   - 另一名成员提到在 Go SDK 之外使用他们自己的会话中间件进行水平扩展，并建议跨重启存储和检索**会话数据**的能力将非常有益。
- **MCP 状态混淆使对话复杂化**：*关于 MCP 中是否可以实现应用级状态存在混淆*，传输组（transports group）正在频道 <#1399986181445386352> 中讨论如何解决此问题。
   - 有人提到，大多数主流客户端会与远程服务器在所有对话中保持一个会话，而某些服务器在工具调用之间保持状态，这在存在多个对话的情况下效果不佳。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1461252514136592536)** (9 messages🔥): 

> `1x1 Convolution vs SVD/PCA, Quanta Theory of Neural Networks, AI Assisted Coding vs Vibe Coding` 


- **1x1 Convolution 可能优于 SVD/PCA**：一位成员建议使用 **1x1 Convolution** 代替 **SVD/PCA** 进行特征提取，理由是 **SVD/PCA** 提取的是方差最高（最“响亮”）的特征，可能会捕捉到通用的语法噪声而非具体的“意图”信号，并附上了[原始推文链接](https://fxtwitter.com/i/status/2011094378396467316)。
   - 他们认为 **1x1 Conv** 可以让模型通过反向传播（backprop）精确地学习哪些注意力头（heads）对损失函数至关重要，并且在推理（inference）时更轻量。
- **“Quanta”理论引发讨论**：成员们讨论了 *quanta* 理论，该理论指出网络必须学习各种模块，每个模块实现不同的算法或检索不同的知识片段。
   - 一位成员表示怀疑，认为许多机制可能是纠缠在一起的，或者过于通用而无法指定特定用途，这可能会导致对神经网络的机械论解释过于简化。
- **AI 辅助编程（AI Assisted Coding）对比 Vibe Coding**：一位成员将 **AI Assisted Coding** 工具（cursor/windsurf/antigravity）与他们所谓的 **Vibe Coding** 工具（devin/tembo/jules）进行了对比。
   - 未提供进一步细节。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1461326860037390398)** (8 messages🔥): 

> `Native vs Custom Tool Calling Benchmarks, DSPy Tool Performance, Language Model Differences, Model-Specific Tool Performance` 


- **DSPy 工具性能可能优于原生 LLM 工具**：一位成员引用了 [DSPy 文档](https://dspy.ai/learn/programming/tools/#using-native-tool-calling)，并询问有关**原生工具调用（native tool calling）**与**自定义工具调用（custom tool calling）**的基准测试对比。
   - 另一位成员回应称，认为 **DSPy 工具** 优于 **原生工具** 的说法过于笼统，这取决于所使用的具体语言模型（LM）。
- **原生工具和 DSPy 工具应当进行基准测试**：一位成员强调，性能在不同的语言模型之间存在差异，即使是来自同一个 AI 实验室的模型也是如此，因此针对特定用例和模型组合进行**基准测试是必不可少的**。
   - 另一位成员表示赞同，指出性能可能在任何方向上产生波动，用户应该使用其**特定的模型和程序**进行测试，以衡量和评估哪种方案效果最好。
- **原生工具调用可能会导致质量下降**：文档中的陈述是一个较弱的断言。
   - 这在某些模型中确实是有可能发生的。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1461459876973903894)** (2 messages): 

> `Aider asking to add files, Aider configuration` 


- **Aider 烦扰用户添加文件**：一位用户询问是否可以让 **aider** 自动添加文件，而不是不断提示用户。
   - 对于偏好减少交互式文件管理的用户来说，这可以简化工作流。
- **Aider 配置愿望清单**：一位用户咨询了如何配置 **aider** 以绕过添加文件的提示。
   - 用户希望 **aider** 能够自动添加文件，表明其倾向于更少交互的工作流。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1461180664262164659)** (5 messages): 

> `aider setup, CLIProxyAPI, CI logs with Aider` 


- **用户在 Aider 安装设置上遇到困难**：一位用户反映通过命令提示符安装后，在设置 **aider** 时遇到困难，正在寻求后续步骤的指导，如[此截图](https://cdn.discordapp.com/attachments/1133060505792159755/1461275448574218373/image.png?ex=696a9f10&is=69694d90&hm=19ccaef4fb45cd4288b6307abb3eca0a6819f27eb6253f0820357b2219006a4d)所示。
   - 未提供进一步信息。
- **将 CI 日志整合进 Aider 工作流**：一位用户询问使用 **aider** 配合 **CI 日志** 来修复失败测试的最佳实践，同时希望将日志文件排除在 Git 之外。
   - 建议使用命令 `aider --read ci.log` 作为潜在的解决方案。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1461220391031930925)** (6 条消息): 

> `Stable Diffusion 中的黑色 PNG，tinygrad 中的 NULL 设备` 


- **黑色 PNG 困扰 Stable Diffusion 的运行！**: 一名新用户报告在运行 `examples/stable_diffusion.py` 并使用 `--fakeweights` 参数时，得到了全黑的 PNG 图片。
   - 经澄清，**NULL device** 实际上并不执行任何计算，但仍然会生成并调度 kernel。
- **NULL Device：一个无计算的奇迹！**: 一名用户询问了 tinygrad 中 **NULL device** 的用途，质疑它是否执行任何计算。
   - 另一名成员确认 **NULL device** 不进行任何计算，澄清了其在没有实际处理的情况下进行 kernel scheduling（内核调度）的作用，还有成员回复道“*这是一个很酷的功能*”。