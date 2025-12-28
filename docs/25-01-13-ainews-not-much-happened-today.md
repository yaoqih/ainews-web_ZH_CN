---
companies:
- kyutai-labs
- lmstudio
- mistralai
- llamaindex
- huggingface
- langchainai
- hyperbolic-labs
- replit
- fchollet
- philschmid
date: '2025-01-14T06:08:22.078500Z'
description: '以下是该文本的中文翻译：


  **kyutai_labs** 推出的 **Helium-1 Preview** 是一款 **20 亿参数的多语言基础大语言模型 (LLM)**，其表现优于 **Qwen
  2.5**。该模型在 **2.5 万亿 token** 上进行了训练，具有 **4096 的上下文长度**，并采用了来自 **7B 模型** 的 token 级蒸馏技术。**Phi-4
  (4-bit)** 已在 **lmstudio** 上发布，并在 **M4 Max** 芯片上运行，以其出色的速度和性能受到关注。**Sky-T1-32B-Preview**
  是一款训练成本仅为 **450 美元的开源推理模型**，其性能可媲美 **o1**，并拥有强劲的基准测试得分。**mistralai** 推出的 **Codestral
  25.01** 是一款全新的 SOTA（最先进）编程模型，支持 **80 多种编程语言**，且推理速度提升了 **2 倍**。


  技术创新包括：用于优化检索增强生成流水线的 **AutoRAG**；支持自主查询改写和评判的 **Agentic RAG**；利用 **Phi-3**、**Mistral**、**LLaMA-3**
  和 **GPT-3.5** 等模型群体进行 **多智能体微调 (Multiagent Finetuning)** 以提升推理能力；以及利用大型视觉语言模型 (LVLM)
  将视频内容整合进 RAG 的 **VideoRAG**。


  应用案例包括：**skirano** 在 **Replit** 上开发的动态 UI AI 聊天应用；**LangChain** 工具（如用于语音 PDF 对话的
  **DocTalk**）；AI 旅游代理教程以及新闻摘要智能体。**Hyperbolic Labs** 提供具有竞争力的 GPU 租赁服务，包括 **H100**、**A100**
  和 **RTX 4090**。**LLMQuoter** 通过识别关键引用来增强 RAG 的准确性。


  基础设施更新包括：**fchollet** 推出的用于将 LLM 推理从 Python 导出到 C++ 的 **MLX export**，以及 **philschmid**
  开发的语义文本去重工具 **SemHash**。'
id: 3b3c903f-bd16-42fd-97e8-2a91264faf5e
models:
- helium-1
- qwen-2.5
- phi-4
- sky-t1-32b-preview
- o1
- codestral-25.01
- phi-3
- mistral
- llama-3
- gpt-3.5
- llama-3
- gpt-3.5
- llmquoter
original_slug: ainews-not-much-happened-today-9477
people:
- reach_vb
- awnihannun
- lior_on_ai
- sophiamyang
- omarsar0
- skirano
- yuchenj_uw
- fchollet
- philschmid
title: 今天没发生什么特别的事。
topics:
- multilinguality
- token-level-distillation
- context-windows
- model-performance
- open-source
- reasoning
- coding
- retrieval-augmented-generation
- hybrid-retrieval
- multiagent-systems
- video
- large-video-language-models
- dynamic-ui
- voice-interaction
- gpu-rentals
- model-optimization
- semantic-deduplication
- model-inference
---

<!-- buttondown-editor-mode: plaintext -->**一个安静的日子就足够了。**

> 2025年1月10日至1月13日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord 社区（**219** 个频道，**2928** 条消息）。预计为您节省了 **312 分钟** 的阅读时间（按每分钟 200 字计算）。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

欢迎来到 [Codestral](https://x.com/lmarena_ai/status/1878872916596806069)，但对于前沿模型实验室来说，发布通常发生在每月的 15 号左右。快了。

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

**AI 模型发布与基准测试**

- **@kyutai_labs 发布 Helium-1 Preview**：[@reach_vb](https://twitter.com/reach_vb/status/1878860650560025011) 宣布了 **Helium-1 Preview**，这是一个 **2B 参数的多语言基础 LLM**，针对边缘和移动设备。它在 **2.5T tokens** 上训练，具有 **4096 上下文大小**，并利用了来自 **7B 模型** 的 **token 级蒸馏**，其表现**优于 Qwen 2.5**。
  
- **@lmstudio 中的 Phi-4**：[@awnihannun](https://twitter.com/awnihannun/status/1878564132125085794) 在 **M4 max** 上的 **@lmstudio** 中发布了 **Phi-4 (4-bit)** 模型，并对其**速度和性能**表示赞赏。
  
- **@LiorOnAI 发布 Sky-T1-32B-Preview**：[@LiorOnAI](https://twitter.com/LiorOnAI/status/1878876546066506157) 介绍了 **Sky-T1-32B-Preview**，这是一个 **450 美元的开源推理模型**，在 **Math500 上达到 82.4%**，在 **LiveCodeBench-Easy 上达到 86.3%**，性能可与 **o1** 媲美。
  
- **@MistralAI 发布 Codestral 25.01**：[@sophiamyang](https://twitter.com/sophiamyang/status/1878902888434479204) 发布了 **Codestral 25.01**，这是一款**新的 SOTA 编程模型**，在 **LMSYS 上排名第一**，支持 **80 多种编程语言**，速度比之前版本快 **2 倍**。

**AI 研究与创新**

- **AutoRAG 框架**：[@llama_index](https://twitter.com/llama_index/status/1878881368186454161) 推出了 **AutoRAG**，这是一个用于**优化 RAG 流水线**的框架，强调**混合检索**通常优于纯 **vector 或 BM25 方法**。
  
- **@huggingface 的 Agentic RAG**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1878590804325011727) 探讨了 **Agentic RAG**，它通过**重新表述用户查询**、**批判检索结果**并**重复**该过程来**增强系统的准确性和自主性**。
  
- **多智能体微调 (Multiagent Finetuning)**：[@omarsar0](https://twitter.com/omarsar0/status/1878816276312989821) 介绍了 **Multiagent Finetuning**，利用**模型社会**进行**自我提升**，在 **Phi-3, Mistral, LLaMA-3 和 GPT-3.5** 等模型上显示出**推理任务的性能提升**。
  
- **VideoRAG 框架**：[@omarsar0](https://twitter.com/omarsar0/status/1878827350315659421) 展示了 **VideoRAG**，通过使用 **Large Video Language Models (LVLMs)** 整合**视频内容**来增强 **RAG**，在需要**程序性知识**的任务中取得了显著成果。

**AI 应用与工具**

- **动态 UI AI 聊天应用**：[@skirano](https://twitter.com/skirano/status/1878865450702139824) 开发了一款 **AI 聊天应用**，可以根据对话内容**变换其 UI**，支持**深色模式**和 **Windows 98** 等主题，可在 **@Replit** 上使用。
  
- **LangChain AI 工具**：
  - **DocTalk**：[@LangChainAI](https://twitter.com/LangChainAI/status/1878864591230234941) 推出了 **DocTalk**，通过语音交互实现与 **PDF 文档的自然对话**。
  - **AI 旅游代理教程**：演示了如何使用 **LangChain 的 Plan and Execute 架构**构建 **AI 旅游代理**。
  - **智能新闻代理**：利用 **LangGraph** 实现 **AI 驱动的新闻摘要**。
  
- **Hyperbolic Labs 的 GPU 租赁**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1878850466626576696) 提供具有竞争力的 **GPU 租赁**价格，包括 **H100 ($0.99/hr)**、**A100 ($1.2/hr)** 和 **RTX 4090 ($0.5/hr)**，支持**算力普惠**。
  
- **LLMQuoter**：[@omarsar0](https://twitter.com/omarsar0/status/1878820053933855147) 展示了 **LLMQuoter**，它通过在生成答案之前**识别关键引用**来**增强 RAG**，实现了 **20 点以上的准确率提升**。

**AI 基础设施与硬件**

- **面向 C++ 的 MLX 导出**：[@fchollet](https://twitter.com/fchollet/status/1878880859077714382) 分享了使用 **MLX** 将 **LLM 推理**从 **Python** 导出为**独立的 C++ 二进制文件**的能力。

- **SemHash (由 @philschmid 提供)**：[@_philschmid](https://twitter.com/_philschmid/status/1878743789155516565) 介绍了 **SemHash**，这是一个**语义文本去重库**，可以在几分钟内**对数百万条记录进行去重**，这对于**防止数据泄露**至关重要。
  
- **适用于 Apple 设备的本地 LLM 应用**：[@awnihannun](https://twitter.com/awnihannun/status/1878843809460875593) 发布了一款支持 **iPhone, iPad, Mac** 的**开源 LLM 应用**，使用 **MLX Swift** 构建，采用 **MIT 许可证**。
  
- **Torch 兼容性指南**：[@StasBekman](https://twitter.com/StasBekman/status/1878609223963246979) 提供了跨 **PyTorch 版本**的 **torch._scaled_mm** **向后兼容性指南**。

**AI Safety, Ethics & Policies**

- **ICLR 2025 LLM 信任研讨会**：[@micahgoldblum](https://twitter.com/micahgoldblum/status/1878834198620119443) 宣布了 **ICLR 2025 研讨会**，重点关注**建立对 LLM 及其应用的信任**，设有**论文奖项**和**演讲者阵容**。
  
- **Anthropic 研究员计划**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1878844587491643721) 征集 **Anthropic 研究员计划**首届成员的**申请**，旨在进行 **AI safety 研究**。
  
- **英国 AI 政策战略**：[@jackclarkSF](https://twitter.com/jackclarkSF/status/1878821057370681466) 赞扬了**英国政府的 AI 采用和发展战略**，强调了 **AI 增长区**、**解锁国家数据**、**20 倍公共算力**以及**资助技术监管机构**等举措。
  
- **AI Agent 生产力**：[@bindureddy](https://twitter.com/bindureddy/status/1878606861433463240) 讨论了可以在 **Salesforce, PayPal 和 Confluence** 等系统中**执行自主任务**的 **AI Agent**，有可能**提高 50% 的生产力**并缩短工作周。

- **@RichardMCNgo 论 AI 自我胁迫**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1878561834724123120) 探讨了 **AI Agent 中的自我胁迫**，强调了**模型纪律**对于防止**高度不可读性**并确保**伦理行为**的重要性。

**Memes/Humor**

- **@reach_vb 的幽默吐槽**：[@reach_vb](https://twitter.com/reach_vb/status/1878898525830050265) 推文称：**“哈哈哈，这到底是什么鬼？你怎么把这两者协调起来的？”**
  
- **@agihippo 的梗询问**：[@agihippo](https://twitter.com/agihippo/status/1878800703109710287) 问道：**“这是一个梗吗？我做得对吗？”**
  
- **@teortaxesTex 的吐槽**：各种幽默和吐槽推文，例如 **“顺便说一下，Sonnet 比 DeepSeek 受到更多的 CCP 审查”** 以及 **“上帝之王 Claude 听起来很带感”**。
  
- **来自 @saranormous 的个人幽默**：[@saranormous](https://twitter.com/saranormous/status/1878585361632485748) 分享道：**“而且自从有了第一个孩子后，我的睡眠质量就一直很差 😮‍💨”**。
  
- **@yrhesiaj 的梗互动**：[@yrhesiaj](https://twitter.com/yrhesiaj/status/1878718780974760226) 喜欢一种梗图格式，表示：**“我喜欢这个梗图格式，我们需要更多这样的东西”**。

---

# AI Reddit Recap

## /r/LocalLlama Recap

**主题 1. 对用于确定 LLM 智能的“陷阱”测试的批评**

- **[如果你让 Llama 找 5 个拼写中不含字母 E 的奇数，它会变得语无伦次](https://i.redd.it/w5j543q9pnce1.jpeg)** ([得分: 465, 评论: 198](https://reddit.com/r/LocalLLaMA/comments/1i01k4s/llama_goes_off_the_rails_if_you_ask_it_for_5_odd/))：该帖子幽默地强调了 **Llama**（一种 AI 模型）在被要求识别五个拼写中缺少字母 'E' 的奇数时所面临的挑战。AI 的回应包括错误和荒谬的词汇，如 "Sand"、"One"、"Tud" 和 "Dug"，说明了该模型在准确处理和推理该请求方面的困难。
  - 评论者讨论了 AI 模型在寻找**拼写中不含字母 "E" 的奇数**时固有的困难，并指出英语中大多数奇数都包含 "E"。尽管进行了各种尝试，像 **Deepseek R1** 和 **O1-Mini** 这样的模型确认了该任务的不可能性，而一些模型（如 **Gemini 1.5 pro**）则试图通过使用数字或罗马数字来规避问题。
  - 讨论强调了 AI 模型在这一挑战中的**失败模式**，像 **Groq 2** 这样的模型幽默地改变了拼写以符合标准。这个问题被比作 **"strawberry 测试"**，强调该任务既涉及拼写挑战也涉及逻辑挑战，要求模型能够识别出不存在有效的解决方案。
  - 对话中提到了各种 AI 模型和平台，例如 **Meta 的 70B 和 405B 模型**、**Qwen2.5-Plus** 以及 **Pal Chat iOS 应用**，其中 **Deepseek v3** 显著地评估了 1-100 之间的数字并得出结论：没有一个符合标准。这凸显了任务的复杂性以及模型在解决问题时采取的多样化方法。

**主题 2. Kokoro TTS 以有限参数实现高性能**

- **Speaches v0.6.0 - Kokoro-82M 和 PiperTTS API 端点** ([Score: 90, Comments: 15](https://reddit.com/r/LocalLLaMA/comments/1i02hpf/speaches_v060_kokoro82m_and_pipertts_api_endpoints/)): **Speaches v0.6.0** 引入了对 **Piper** 和 **Kokoro** Text-to-Speech 模型的支持，具有 GPU/CPU 支持、Docker 部署和 OpenAI API 兼容性等特性。它还通过 SSE 和 WebSocket 提供流式传输和实时转录、动态模型处理，以及即将推出的音频生成、情感分析和 Realtime API 等功能。[项目链接](https://github.com/speaches-ai/speaches)和[文档](https://speaches-ai.github.io/speaches/)可查看更多详情。
  - **Docker 镜像访问问题**：用户报告在尝试从 **ghcr.io** 拉取 Docker 镜像时出现 **401 Unauthorized** 错误，这表明镜像仓库可能被设置为私有，或者授权令牌（authorization tokens）存在问题。


- **为什么 Kokoro TTS 在参数如此少的情况下表现这么好？** ([Score: 100, Comments: 46](https://reddit.com/r/LocalLLaMA/comments/1i06mew/how_is_kokoro_tts_so_good_with_so_few_parameters/)): **Kokoro TTS** 仅凭 **82M 参数** 就取得了令人印象深刻的效果，这主要归功于对 **StyleTTS 2** 模型架构的修改，以及主要使用来自 **OpenAI** 和 **ElevenLabs** 的合成数据进行训练。其有效性可能源于合成数据的质量或未公开的架构变更。[Hugging Face 上的 Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M)。
  - 讨论中对**开源音频数据集**的质量表示怀疑，并认为 **Kokoro TTS** 可以用更少的参数达到类似的效果。用户表示有兴趣看到修改后的训练代码，以便探索在消费级硬件上预训练模型，强调了“以少胜多”的潜力。
  - **Kokoro TTS** 的**语音克隆（voice cloning）功能**引发了争论，一些用户注意到由于训练时间有限，该功能目前缺失，而另一些人则指出仅需极少音频样本即可成功恢复语音。被 OpenAI 移除的 **Sky 语音**的恢复就是一个例子，仅使用了 3 分钟的音频。
  - 讨论了 TTS 模型中的**量化（Quantization）技术**，用户指出 Kokoro TTS 有潜力通过 **FP16 和 Int8 量化**等方法在减少参数的同时保持性能。人们考虑了模型大小与性能之间的权衡，一些人认为进一步压缩可能会损害实用性。


**主题 3. Sky-T1：仅需 450 美元的开源 AI 模型训练**

- **[研究人员开源 Sky-T1，一个训练成本低于 450 美元的“推理”AI 模型] (https://techcrunch.com/2025/01/11/researchers-open-source-sky-t1-a-reasoning-ai-model-that-can-be-trained-for-less-than-450/)** ([Score: 52, Comments: 12](https://reddit.com/r/LocalLLaMA/comments/1i0hecs/researchers_open_source_skyt1_a_reasoning_ai/)): **研究人员**发布了 **Sky-T1**，这是一个专注于**推理（reasoning）**能力的开源 AI 模型，训练成本低于 **450 美元**。这一进展突显了更易获取且更具成本效益的 AI 训练方案的趋势。
  - **Sky-T1 的训练过程**：讨论指出 **Sky-T1** 是在 **QWEN-32B-Instruct** 的基础上，使用来自 **QwQ** 的蒸馏数据进行微调的，而不是从零开始花费 **450 美元**训练。这一澄清表明文章在训练成本方面存在误解。
  - **数据集与推理**：使用了 **1.7 万个任务**作为数据集，考虑到从数学教科书中轻松获取更多数据的潜力，一些人认为这个规模小得令人惊讶。这引发了对训练所用数据集的新颖性和有效性的质疑。
  - **蒸馏与思考步骤**：该模型通过基于补全（completion-based）的蒸馏执行推理任务的能力值得关注，这引发了人们对为什么 **OpenAI** 不在其模型中提供显式思考步骤的好奇。有人提到，即使是 **Gemini** 的思考模型也不提供这些步骤，除非是实验版本。

**主题 3. Hugging Face 为 AI 开发者推出 Agent 课程**

- **Hugging Face 发布了关于 Agent 的免费课程。** ([Score: 289, Comments: 18](https://reddit.com/r/LocalLLaMA/comments/1i0b289/hugging_face_released_a_free_course_on_agents/)): **Hugging Face** 发布了其 **Smolagents 课程**的新章节，重点介绍了三种类型的 Agent：代码 Agent、检索 Agent 和自定义功能 Agent。该课程免费提供，旨在帮助开发者构建 Agent 应用，访问地址见 [此处](https://github.com/huggingface/smol-course/tree/main/8_agents)。
  - **Smolagents 与模型兼容性**：用户报告在 **ollama** 上使用 **qwen2.5-coder 32B** 时，**Hugging Face 演示代码**会出现问题，这可能与默认的 ollama 系统提示词或端点配置有关。此外，还有关于加载不同模型的灵活性讨论，包括 **HfApiModel** 以及在显存 (VRAM) 受限场景下使用 **gguf** 的可能性。
  - **关于 LLM 调用次数的指南**：“尽可能减少 LLM 调用”的指南引发了争论。一些用户认为，在涉及搜索和分类等任务的复杂 Agent 工作流中，频繁的短 LLM 调用可能更有效。这种方法虽然可能增加成本，但对于在专业用例中实现更高精度可能是必要的。
  - **课程先修要求与代码可用性**：该课程被认为只要具备基础的 **Python** 知识并了解如何通过 **API** 使用 **LLM** 即可入门。用户对课程材料提供了反馈，特别指出某些代码片段最初无法运行，目前已在文档更新中得到解决。


## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. UC Berkeley 的 Sky-T1 以极低训练预算超越 OpenAI-o1**

- **[伯克利实验室发布 Sky-T1，一款开源推理 AI，训练成本仅需 $450，且在关键基准测试中击败了早期的 o1！！！](https://techcrunch.com/2025/01/11/researchers-open-source-sky-t1-a-reasoning-ai-model-that-can-be-trained-for-less-than-450/)** ([Score: 217, Comments: 32](https://reddit.com/r/OpenAI/comments/1i0cy09/berkeley_labs_launches_skyt1_an_open_source/)): 伯克利实验室发布了 **Sky-T1**，这是一款开源推理 AI 模型，它将训练成本显著降低至 **$450**，并在关键基准测试中超越了早期的 **o1** 模型。这一进展紧随最近发布的 DeepSeek v3 模型（其训练成本曾被误传为 **$5,500**），突显了 Sky-T1 的成本效益和性能优势。[阅读更多](https://techcrunch.com/2025/01/11/researchers-open-source-sky-t1-a-reasoning-ai-model-that-can-be-trained-for-less-than-450/)。
  - **成本与性能**：关于 DeepSeek v3 模型的训练成本有一个修正，应为 **550 万美元**而非 **$5,500**，这进一步强调了 Sky-T1 的成本效率。
  - **开源透明度**：**Sky-T1** 的开源特性受到关注，这使得设计和数据更加透明，无需对其能力进行猜测。
  - **创新与过拟合担忧**：一些评论者质疑 Sky-T1 背后的真实创新，怀疑其依赖于精心策划的合成数据，并可能对基准测试存在过拟合。


- **Sky-T1-32B：开源推理模型在编程和数学基准测试中超越 OpenAI-o1** ([Score: 103, Comments: 9](https://reddit.com/r/OpenAI/comments/1i0cyip/skyt132b_opensourced_reasoning_model_outperforms/)): **UC Berkeley** 发布了 **Sky-T1-32B**，这是一个开源推理模型，在 **Math500**、**AIME** 以及 **Livebench medium & hard** 等基准测试中超越了 **OpenAI-o1**。该模型的训练成本低于 **$450**，更多细节可以在 [此处](https://youtu.be/uzuhjeXdgSY) 找到。
  - 用户对以 **YouTube 视频**作为信息源表示不满，更倾向于直接获取基准测试链接和模型下载地址。**R4_Unit** 批评视频描述中缺乏有用信息，导致该视频被点踩。
  - **LocoMod** 提供了该模型在 **Hugging Face** 上的直接链接：[Sky-T1-32B-Preview-GGUF](https://huggingface.co/bartowski/Sky-T1-32B-Preview-GGUF)，强调了节省时间的重要性。
  - **Formal-Narwhal-1610** 指出标题具有误导性，澄清 **Sky-T1-32B** 超越的是 **o1-preview** 而非完整版的 **o1** 模型。


---

# AI Discord 摘要

> 由 o1-2024-12-17 生成的摘要之摘要的摘要

**主题 1. 新模型与令人惊叹的数据**  
- [**Codestral 25.01 横扫速度排行榜**](https://mistral.ai/news/codestral-2501/)：它在 copilot arena 排行榜上登顶，但在 Aider 多语言基准测试中仅获得 11% 的成绩。成员们对其 256k 的上下文窗口感到兴奋，许多人正关注其生产就绪状态。  
- [**Sky-T1 以低于 450 美元的预算实现飞跃**](https://novasky-ai.github.io/posts/sky-t1/)：这款 32B 模型在热门推理任务上可与 o1-preview 竞争，且无需巨额资金。其开源代码库 [SkyThought](https://github.com/NovaSky-AI/SkyThought) 公开寻求更多社区驱动的突破。  
- [**Helium-1 进军移动端**](https://kyutai.org/2025/01/13/helium.html)：Kyutai 的 2B 参数模型旨在为边缘设备提供低延迟隐私保护，支持 6 种语言。用户为这种不牺牲性能的小规模解决方案欢呼。  

**主题 2. HPC 调优与内存动态**  
- [**Triton Puzzles 将 GPU 推向极限**](https://github.com/gauravjain14/mlcompilers_and_kernels/tree/main/triton_kernels)：开发者在 A100 与 A30 上自动调优 kernel，关注共享内存限制以获得巨大收益。他们还参考了 [Liger Kernel 交叉熵代码](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py)，以从细小数据块中榨取更多速度。  
- [**Slurm 解决方案化险为夷**](https://slurm.schedmd.com/sbatch.html)：在多 GPU 集群上设置 `--mem=0` 或 `--exclusive` 解决了基于 CPU 的 OOM 问题。正确的资源标记将 HPC 的挫败转化为顺畅的运行。  
- [**PyTorch 中的补丁式 Profiling**](https://github.com/pytorch/pytorch/issues/64345)：UTF-8 解码错误阻碍了高级流水线分析。用户通过 NNSight 使用 meta devices 并流式传输激活值，以规避 OOM 惨剧。  

**主题 3. 构建 Agent 与自定义机器人**  
- [**Friday Agents 在 JS 中狂欢**](https://github.com/amirrezasalimi/friday-agents)：这个多 Agent 框架帮助开发者并行化任务，并能轻松接入 [OpenRouter](https://openrouter.ai/)。人们称赞并发性让 Agent 实验感觉势不可挡。  
- [**DeVries AI 坐拥 200 多个 LLM**](https://devriesai.com/)：每月 24.99 美元，Telegram 粉丝可以在一个聊天流中快速切换 200 多个模型。免费试用吸引了早期采用者来测试迷宫般的 AI 组合。  
- [**Aider 新增聊天模式**](https://aider.chat/HISTORY.html)：v0.71.0 版本改进了 “/ask” 和 “/code” 之间的切换，并支持使用三反引号围栏流式输出精美内容。用户非常喜欢在代码和提问模式之间快速切换。  

**主题 4. 微调、LoRA 与数据之美**  
- [**Unsloth 声称提速 30 倍**](https://unsloth.ai/introducing)：自定义 Triton kernel 承诺在 LLM 训练中实现巨大飞跃，例如 Llama 3.3 和长上下文扩展。用户观察到内存占用下降，同时聊天模板保持了模型输出的稳定。  
- [**LoRA 魔法精准复刻作者风格**](https://docs.unsloth.ai/get-started/unsloth-notebooks)：只要提供足够的精选文本，LoRA 就能大规模复制写作的细微差别。迭代微调培育了连贯的声音，在创意和医疗任务中都令人惊叹。  
- [**质量胜过数量**](https://arxiv.org/abs/2402.12847)：论坛用户强调，严谨的数据准备胜过海量的原始转储。他们建议在消耗宝贵的 GPU 机时之前，先使用其他 LLM 过滤文档。  

**主题 5. 隐私、缓存与超长上下文**  
- [**隐私模式引发关注**](https://forum.cursor.com/t/concerns-about-privacy-mode-and-data-storage/5418)：用户对存储在服务器上的数据嵌入以及潜在的 NDA 违规提出质疑。他们呼吁在代码处理方式上提高透明度。  
- [**Prompt Caching 用于加速 RAG**](https://docs.llamaindex.ai/en/stable/examples/prompts/advanced_prompts/)：开发者依靠正确的文件集来实现稳定的缓存命中。Anthropic、OpenAI 和本地设置之间的差异促使他们不断发明新策略。  
- [**128k 上下文之梦**](https://lmstudio.ai/model/phi-3.1-mini-128k-instruct)：大胆的测试者使用 Phi 3.1 Mini 128k 挑战更大的窗口。他们发现 VRAM 需求适中，但非常喜欢为巨量 Prompt 提供的额外呼吸空间。

---

# 第一部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 与 Llama 3.3 领跑**：用户报告称，使用 **Unsloth** 微调的 **Llama 3.3** 在聊天模板下表现出稳定的训练效果，在性能指标上得分更高，且所需的 **VRAM** 更少。
   - **Unsloth** 包含自定义的 **Triton kernels**，并声称有 **30x** 的训练加速，引发了社区对 [Unsloth 博客](https://unsloth.ai/introducing) 的关注。
- **模仿作者风格的 LoRA 技巧**：成员们使用 **LoRA** 来复制写作风格，并强调大量的数据准备是成功的关键。
   - 他们指出，**迭代微调**有助于实现一致的声音复制，并解决了 [文档](https://docs.unsloth.ai/get-started/unsloth-notebooks) 中的细微差别。
- **使用欺骗性 LLM 进行网络行动**：一位网络安全研究员构建了一个专门用于**网络欺骗**的 **LLM**，生成了超过 **1k** 个模拟对手连接。
   - 参与者赞赏这些基于人格（persona-based）的策略如何更有效地识别**诈骗**，激发了对先进方法的兴趣。
- **Maya 的多语言 V-L 飞跃**：**Maya** 作为一种**多语言视觉-语言模型（Multilingual Vision-Language Model）**被推出，其细节在 [Twitter](https://twitter.com/nahidalam/status/1866667770114609217) 分享的预印本中有所概述。
   - 成员们称赞了 **Maya** 潜在的跨语言能力，认为它是处理文本和图像结合任务的一个令人兴奋的方向。
- **基于视频转录的 TTS 聊天机器人**：开发者们寻求简化**视频转录**流程以用于实时 **TTS** 聊天机器人，并参考了 **Whisper** 和其他语音转文本工具。
   - 他们探索了 **Fish Agent** 和 **Kokouro** 用于语音输出，并强调了实现高级语言覆盖需要 **10,000 小时**的音频。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SmolLM 伴随 315GiB 的发布引起轰动**：**SmolLM-Corpus** 发布了 **315GiB** 的数据，分为 23698 个 `jsonl.zst` 分片，包括来自 **cosmopedia-v2** 和 **fineweb-edu-dedup** 的子集，如 [Hugging Face](https://huggingface.co/datasets/Avelina/smollm-corpus) 所示。
   - 社区成员对大规模数据集的使用表现出浓厚兴趣，并在讨论中提到了 **Grouped-Query Attention** 和扩展的 **VLM** 能力。
- **Latro 凭借 PRMs 和 VinePPO 取得进展**：**Latro** 模型旨在通过 **RL** 加 **Chain-of-Thought** 来提高推理能力，在密集奖励设置下可能优于 **RLVR**，并参考了 [熵正则化过程奖励模型（Entropy-Regularized Process Reward Model）](https://arxiv.org/abs/2412.11006) 及相关研究。
   - **VinePPO** 被引用为一种提供逐步精细信用分配（credit assignment）的方法，尽管人们仍然担心软奖励信号可能会鼓励记忆而非深度推理。
- **Goodfire API 激发协作**：一位成员在 `gsm8k_cot_llama` 任务上集成了与 **Llama 8B** 匹配的 **Goodfire API** 构建，并使用了 **VLLM**，邀请在 [lm-eval-harness 仓库](https://github.com/menhguin/lm-evaluation-harness/blob/main/lm_eval/models/goodfire.py) 中进行进一步开发。
   - **MATH-Hard** 数据集从 **Hugging Face** 移除导致了排行榜评估问题，[GitHub issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/2618#issuecomment-2583172531) 提出了一个临时解决方案。
- **Neel Nanda 的机械可解释性故事**：尽管尝试使用了 **Whisper** 工具，**机械可解释性（mechanistic interpretability）**读书会的音频仍有部分未被转录。
   - 听众们称赞了通过 [Spotify](https://open.spotify.com/episode/5XjHhNQxIb16eJZXGmbaCk?si=Z8LTnSo7QHGJkBxgGZbIJA) 分享的关于 **SAEs** 的 **Neel Nanda** 播客，该播客专注于更清晰地理解模型内部机制。
- **Slurm 内存变动**：**Slurm** 标记了基于 **CPU** 的 **OOM** 而非 **GPU** 内存问题，根据 [Slurm sbatch 文档](https://slurm.schedmd.com/sbatch.html)，通过使用 `--mem=0` 或 `--exclusive` 解决了该问题。
   - 一位用户询问如何估算预训练时每个 **GPU** 所需的 **CPU RAM** 和核心数，引发了关于更系统地跟踪使用情况的建议。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Cascade 令人困惑的代码**：用户反映 **Cascade** 正在生成随机输出并错误标记文件，产生的错误即使在使用 [prompt engineering guidelines](https://docs.codeium.com/best-practices/prompt-engineering) 的情况下也阻碍了开发。他们还抱怨其不可预测性，并以 *the 70% problem* 为例，说明代码可能仍会偏离预期结果。
   - 一些参与者建议进行更严格的测试以减少错误，但他们仍希望 **Cascade** 能尽快改进。
- **自定义模型热潮：Gemini Flash 对比当前选项**：热情的群体请求兼容 **Gemini Flash**，感叹 **Windsurf** 中只能使用预先批准的模型，并指向 [Codeium's feature requests](https://codeium.canny.io/feature-requests) 以寻求更广泛的模型支持。他们希望能够不受限制地更换新的 AI 模型。
   - 尽管多次请求，目前还没有添加此功能的正式时间表，因此一些人继续寻找支持更广泛 AI 使用的其他编辑器。
- **Cursor 之争：自动补全对决**：用户将 **Cursor** 与 **Windsurf** 进行了对比，称赞 **Cursor** 具有更精准的自动补全建议，但批评其在压力下的可靠性；而 **Windsurf** 的 *agentic features* 因其先进的工作流而受到赞誉 ([support docs](https://codeium.com/support))。
   - 他们得出结论，两者都需要更高的稳定性，一些人推动采用不同的订阅结构，而不是当前的 flow-credit 模型。



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE 的收获与抱怨**：一些开发者报告在 **Cursor IDE** 中编码流程更快捷，而另一些人仍遇到速度变慢和 AI 建议冲突的问题，尤其是在大型项目中。
   - 社区成员建议通过检查点（checkpoints）恢复代码状态，并指向 [bug reports on the forum](https://forum.cursor.com/t/error-unauthorized-request/39861/28)，明确要求更稳定的扩展设置。
- **Codestral 的海量上下文**：新的 **Mistral** 发布版本 [Codestral 25.01](https://mistral.ai/news/codestral-2501/) 提供了巨大的 256k 上下文窗口，承诺在代码理解方面带来巨大改进。
   - 它已经得到 **Continue.dev** 的支持，参与者推测将其与 Cursor 合并可以简化高级代码生成功能。
- **Cursor 中的协作创作**：爱好者建议共同开发基于 AI 的应用，例如 **Test Manager AI** Agent，以提升初级和高级开发者的技能。
   - 他们对潜在的协同效应表示欢迎，强调实践学习以及它如何展示 **Cursor** 在下一代编码协作中的能力。
- **隐私难题：嵌入数据隐忧**：关于 **Cursor** 存储聊天 Embeddings 的担忧浮出水面，参考了 [privacy-mode details](https://forum.cursor.com/t/concerns-about-privacy-mode-and-data-storage/5418) 以及企业环境中的 NDA。
   - 论坛指出开启“隐私模式”可以防止代码上传，但许多人要求在数据管理和服务器端存储方面有更高的透明度。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.6 推出 Beta 版工具**：LM Studio 发布了 **0.3.6** 版本，包含全新的 **Tool Calling API (Beta)** 和更新的安装程序系统，详见其[官方博客](https://lmstudio.ai/blog/lmstudio-v0.3.6)。
   - 用户在本地运行中测试了 **Qwen2VL** 和 **QVQ**，并在[官方 Bug 追踪器](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues)中记录了问题和成功案例，部分用户称赞其在 **M4 Ultra** 硬件上的性能飞跃。
- **Bartowski 的 Sky T1 展示 32B 性能**：社区成员通过 LM Studio 评估了 [Bartowski/Sky-T1-32B-Preview-GGUF](https://huggingface.co/bartowski/Sky-T1-32B-Preview-GGUF) 模型在本地编码任务中的表现。
   - 他们报告称使用 **Q4_K** 或 **Q5_K** 量化时性能更强，但在用户提交的[反馈帖子](https://gist.github.com/shermanhuman/2b9a82df1bab242a8edffe504bb1867c)中指出，旧设备上存在内存开销。
- **PowerMac G3 迎来 AI 改造**：一位用户展示了运行 LM Studio 的改造版 **PowerMac G3**，引发了硬件怀旧情结以及关于将经典机箱与现代内部组件结合的讨论。
   - 其他人将此配置在资源消耗方面与 [NVIDIA 的 Project DIGITS](https://www.nvidia.com/en-us/project-digits/) 进行了比较，一些人主张使用专用 GPU。
- **Phi 3.1 Mini 128k 扩展上下文边界**：大胆的测试者在 LM Studio 中尝试了 [Phi 3.1 Mini 128k 模型](https://lmstudio.ai/model/phi-3.1-mini-128k-instruct)，以满足更大的上下文需求。
   - 他们发现该模型对系统的要求适中，并建议仔细管理 VRAM 以获得稳定的输出，相关技巧已发布在 [LM Studio 文档](https://lmstudio.ai/docs/basics/chat)中。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Claude 变得“愤怒”**：一些用户注意到 **Claude** 模型采用了更加强硬的风格，在回答中重复使用“direct”（直接）和“helpful”（有帮助）等词汇，引发了关于“愤怒 AI”人设的笑话。
   - 一条幽默的推文声称发布了新的 Claude 模型，虽然遭到了质疑，但引发了关于可能存在“秘密更新”的讨论（[来自 Jacques 的推文](https://x.com/jacquesthibs/status/1878851967981887736)）。
- **超参数调优服务引发好奇**：关于超参数搜索自动化解决方案的问题引起了关注，突显了 **Bayesian optimization**（贝叶斯优化）和调试训练问题的复杂性。
   - 一些人强调需要进行严格测试以发现隐藏的陷阱，并推测最终会出现“超参数即服务”（Hyperparam-as-a-Service）的产品。
- **Qwen 0.5B 在数学上栽了跟头**：较小的 **Qwen 0.5B** 模型在某些任务上表现出色，但经常产生荒谬的答案或陷入死循环（[kz919/QwQ-0.5B-Distilled](https://huggingface.co/kz919/QwQ-0.5B-Distilled)）。
   - 人们想知道 **Generative Knowledge Distillation (GKD)** 是否引入了意想不到的怪癖，并对它与常规蒸馏的区别表示困惑。
- **MobileLLM 撼动小型模型**：**MobileLLM** 的论文表明，对于紧凑型端侧语言模型，基于标签的训练优于标准蒸馏（[arXiv 上的 MobileLLM](https://arxiv.org/abs/2402.14905)）。
   - 这引发了更深层次的问题：合成数据或先进的蒸馏方法对于低参数模型是否仍然重要。
- **Element-wise Attention 引发讨论**：一篇题为 **Element-wise Attention Is All You Need** 的论文提出了一种新方法，承诺在保持质量的同时降低训练复杂度（[arxiv.org/abs/2501.05730](https://arxiv.org/abs/2501.05730)）。
   - 几位工程师权衡了这种机制重塑标准 Attention 架构以实现更高效推理的可能性，燃起了对下一阶段改进的希望。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **StackBlitz 通过预告推文引发关注**：我们看到一条来自 [StackBlitz 的推文](https://x.com/stackblitz/status/1878818461905739994)，提到了 **Bolt.new** 公告的进展，引发了开发者的好奇。
   - 一些参与者推测了即将到来的改进，但尚未确认详细信息，让观察者们对官方消息充满期待。
- **Stripe 进驻 Bolt**：报告显示 **Stripe 集成** 即将到来，一些用户已经成功实现并称其为他们配置中的**重大加分项**。
   - 其他人在代码合并时遇到了困难，参考 YouTube 教程进行修复，甚至转向 **PayPal** 作为备选方案。
- **Prompting 的痛苦与收获**：多位用户哀叹每当添加新功能时代码就会丢失，强调了启用 **diffs** 以实现稳定扩展等解决方案。
   - 他们参考了 [Bolt Prompting 终极指南](https://docs.google.com/document/d/1SwlpZH1SotqPg2KbZqzWPdpBbs6aKIqMDspSCBCD1iQ/edit) 以获取最佳实践，并分享了一些幽默的评论，比如 *“我一直在不断推进我的产品，直到超过某个临界点。”*
- **Token 紧缺的忧虑**：过度的 Token 使用触动了神经，一位用户在单个叠加层上消耗了 **150 万个 Token**，引发了对更精简 Prompt 的呼吁。
   - 对更便宜的重载和优惠码的需求日益高涨，一段 [关于节省 Token 的 YouTube 教程](https://youtu.be/ayagXgAShSk) 作为省钱方案在流传。
- **网络研讨会热潮**：宣布将于周二东部时间上午 10 点举行关于 **使用 Bolt 构建 AI LLM Apps** 的免费现场培训，指导开发者构建结构化、动态的应用。
   - 组织者指出了环境设置技巧，并引用了 [如何使用 No Code 构建下一代 AI Apps](https://www.reinventing.ai/next-level-ai-apps-no-code) 以提供进一步支持。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **英国的大手笔：生产力翻倍**：英国政府向 **AI** 投资 140 亿英镑，旨在三年内将**生产力**翻倍，引发了关于预算分配和潜在劳动力流失的辩论。
   - 批评者质疑这些资金是否可以更有效地用于其他地方，并警告 **AI** 可能会取代人类角色。
- **Claude 和 Gemini 在 Minecraft 中击败 ChatGPT**：**Claude** 和 **Gemini** 在 Minecraft 竞赛中表现优于 **ChatGPT**，突显了在处理复杂任务时更强的推理和规划能力。
   - 观察者对 **ChatGPT** 的性能差距及其对 GPT 系列模型在竞争场景中的影响表示担忧。
- **Codestral 亮相，具备 256k 上下文**：一款新的 **codestral** 模型在 **Mistral** API 上发布，声称拥有 256k 上下文容量，并引发了与 **GPT-4** 对比的好奇。
   - 成员们正在观察其功能是否能与即将到来的 Canvas 增强功能产生协同效应，其实际影响仍在讨论中。
- **表格困境：GPT vs OCR**：用户报告 **GPT** 反复出现宽表格数据对齐错误，平均准确率约为 60%，同时指出 **Amazon Textract** 等工具可以获得更一致的结果。
   - 他们注意到该模型在解析复杂布局时表现不稳定，引发了关于使用更好的数据格式或“技巧”来改善结果的讨论。
- **工作中的自定义 AI Agent**：参与者探索了用于面向客户支持的**嵌入式 AI** 解决方案，建议使用 **n8n** 和 **flowise**，同时考虑与 Slack 和 WhatsApp 集成。
   - 他们讨论了与服务成本和供应商可靠性相关的挑战，强调了部署稳健 **AI Agent** 的实用性。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **移动端魔法与 50 美元奖励**：团队邀请参与者参加 **1 月 14 日至 15 日**关于 **NotebookLM 移动端体验**的远程访谈，报名请填写[此筛选表单](https://forms.gle/75gYapqbgCmxXiJL6)，完成后可获得 **50 美元**或 Google 周边礼券。
   - 社区成员期待分享使用见解，旨在通过直接反馈来塑造 **NotebookLM** 的移动端功能。
- **音频概览与礼品码**：一份约 [5 分钟的筛选问卷](https://forms.gle/NBzjgKfGC24QraWMA)正在收集关于 **Audio Overviews** 的反馈，完成后续调查可获得 **10 美元**礼品码。
   - 参与者希望优化这些 AI 生成摘要的清晰度和风格，期望能符合用户对可靠音频内容的预期。
- **使用 Akas 轻松制作播客**：用户探索了使用 [Akas](https://akashq.com) 上传 **AI 生成的播客**，从而绕过 **NotebookLM** 严格的登录限制。
   - 他们喜欢更简单的分发模式，让他们能更自由地与他人分享基于对话的内容。
- **多源引用与引用困惑**：一些用户发现 **NotebookLM** 在引用多个文件时表现不佳，导致在引用链接和重复细节方面出现困扰。
   - 尽管对于复杂的 notebook 效果参差不齐，但目前的权宜之计包括仔细命名文档和优化 prompt。
- **嵌入 NotebookLM 与更广泛的用途**：一位用户提议将 **NotebookLM** 嵌入到 Google Sites 等网站中，以将其功能扩展到个人笔记之外。
   - 其他人看到了在教育或团体场景中更广泛采用的潜力，强调了更开放的协作。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **舍弃 Pony 模型，追求 Illustrious 图像**：虽然 **Pony XL** 声称具有很强的标签一致性，但其最终输出效果令人失望，这促使创作者更倾向于使用 **Illustrious**，并提到 **JuggernautXL** 以及 [RealVisXL v5](https://civitai.com/models/139562/realvisxl-v50) 以获得更写实的图像。
   - 参与者建议使用更精炼的数据集来修复欠佳的表现，强调了在采用新模型前进行彻底测试的重要性。
- **Dreambooth 衰落，Koyha_ss 与 OneTrainer 崛起**：创作者们正因方法陈旧而放弃 **Dreambooth**，转而使用 **Koyha_ss** 和 **OneTrainer**，并参考了 [FLUX 训练教程](https://www.youtube.com/watch?v=FvpWy1x5etM)获取进阶步骤。
   - 一些人建议使用 50–150 张图像来增强**特定角色的 Lora**，发现这些新工具比旧教程更可靠。
- **Hires Fix 的高清魔法**：团队发现先以低分辨率生成，然后以 1024x1024 应用 **hires fix** 可以获得更好的清晰度，这一观点得到了 [Reddit 讨论](https://www.reddit.com/r/StableDiffusion/comments/14x6o2c/finally_figured_out_how_to_create_realistic)的支持。
   - 他们观察到直接进行高分辨率生成经常会出现图像元素重复，从而强化了使用增量放大来保持图像连贯性的做法。
- **扩展插件随 sd-webui-regional-prompter 扩展**：诸如 **sd-webui-regional-prompter** 和 [Forge Webui 的 sd-forge-couple](https://github.com/Haoming02/sd-forge-couple) 等各种工具提升了 **Stable Diffusion** 中的图像切片和注意力控制能力。
   - 用户强调了正确的安装步骤（通常是通过 git clone 到正确的文件夹），以躲避周围流传的诈骗链接。
- **Stable Point Aware 3D 激发快速编辑**：来自 [Stability AI](https://stability.ai/news/stable-point-aware-3d) 的 **Stable Point Aware 3D (SPAR3D)** 承诺在不到一秒的时间内，从单张图像实现实时对象编辑和完整的结构创建。
   - 许多人对快速原型制作能力感到兴奋，认为这是将 **3D generation** 与 2D diffusion 工作流集成的关键一步。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI 模型：成本与 Elo 评分详解**：新分享的 [LLM Elo 与定价图表](https://docs.google.com/spreadsheets/d/1x9bQVlm7YJ33HVb3AGb9qlDNkvTy9CyOFZoah0kr3wo/edit#gid=0) 对比了 **o1-preview**、**GPT-4o** 等模型在成本和性能方面的表现，详细列出了高级 Elo 分数和每月订阅价格。它强调，支付更多费用并不总是能保证更好的结果，尤其是在更高的使用规模下。
   - 社区成员赞扬了图表的清晰度，其中一人表示 *“Lmsys Elo 与价格曲线的预测性非常显著”*，并引用了在 **MMLU** 基准测试中发现的相关性。
- **Copilot 等候名单取消**：Satya Nadella 在 [X](https://x.com/satyanadella/status/1878578314115473577) 上宣布 **GitHub Copilot Workspace** 不再有等候名单，从而实现了即时的 Agentic 编程。这通过消除注册障碍，突显了推动更广泛 AI 采用的努力。
   - 这一举措引起了社区对更深层次集成的呼声，一些人将其视为迈向**自主开发流程（autonomous development flows）**的一大步。其他人则预计成本会发生变化，提到了 **$20/month** 计划与高级层级的对比。
- **极速 Llama 3 基准测试**：**Llama 3.3** 70B 在 SambaNova 的定制 **SN40L** 硬件上达到了 **652 tokens/s** 的速度，超越了传统的 GPU 配置。观察家认为这是 2025 年 AI 性能的一次重大胜利，可能会重塑 HPC。
   - 来自 [Santiago](https://x.com/svpino/status/1878797424590012907) 的一条推文称这是 *“我在任何地方见过的运行最快的 Llama 3.3”*，激发了人们对多模型并发的兴奋。同时，用户的轶事强调了通过减少 GPU 小时数实现了更快的微调。
- **Raspberry AI 的零售轮融资**：来自 **a16z** 的 Bryan Kim 宣布了对 **Raspberry AI** 的新投资，这是一个专为零售业设计的端到端生成式设计平台。其愿景侧重于自动化产品构思，重点强调速度和定制化。
   - 他在一条 [推文](https://x.com/kirbyman01/status/1878844418972885077) 中解释了动机，强调了该项目在扩展方面的潜力。这一消息引发了关于融资势头的讨论，一些人称赞专业化解决方案如何在零售领域蓬勃发展。
- **O1 从聊天转向报告**：最近的讨论将 **O1** 定位为不仅仅是一个聊天模型，鼓励像使用*报告生成器*一样使用它。**Ben Hylak** 强调了重新思考 Prompt 提示词如何揭示更深层次的输出，并引用了 Sam Altman 关于替代用法的立场。
   - 一篇关于 O1 的 [客座文章](https://www.latent.space/p/o1-skill-issue) 登上了 Hacker News 首页，说明了公众对这一观点的广泛兴趣。参与者对这一转变表示赞赏，其中一人指出 *“当你懂得如何使用它时，它确实令人惊叹”*。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.71.0 飞速前进**：Aider v0.71.0 发布了用于切换聊天模式的新命令并改进了流式输出，提升了用户参与度，详见 [发布历史](https://aider.chat/HISTORY.html)。
   - 用户称赞了在问题模式和代码模式之间更简单的切换，并对三反引号编辑的持久美化输出表示赞赏。
- **DeepSeek 的异常故障**：多位用户报告称 **DeepSeek** 变得无响应，导致错过截止日期并产生挫败感。
   - 他们要求稳定的 API 性能，并建议通过快速修复来确保可靠性。
- **配置疑问与 Prompt 缓存怪癖**：一位用户发现 `.aider.conf.yml` 中的 `editor-model` 需要使用连字符而不是下划线，这引发了关于在仓库中忽略配置文件的大范围讨论。
   - 其他人分享说，只有在包含完全相同的文件集时，Prompt 缓存才会起作用，这引发了关于可能改进的讨论。
- **量化与多语言讨论**：成员们强调了神经网络的**量化（quantization）**，敦促在编码任务中掌握扎实的知识，并指出多语言套件中的某些 C++ 测试需要特殊的编译器标志。
   - 参与者对比了 **O1** 与 **Sonnet** 的性能，引发了关于在编码场景中哪个模型表现更好的猜测。
- **新工具：CodeGate 与常驻助手**：安全代码生成引发了关于 [CodeGate](https://github.com/stacklok/codegate) 的讨论，该工具旨在提高 CodeGen 工作流中的隐私和安全性。
   - 像 [Deepseek AI Assistant](https://www.youtube.com/watch?v=zoBwIi4ZiTA) 和 [always-on-ai-assistant](https://github.com/disler/always-on-ai-assistant/) 这样的项目展示了为工程师提供的持续后台帮助。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **来自 Microsoft 的 Phi 4 亮相**：本周，来自 **Microsoft** 的全新 [Phi 4](https://openrouter.ai/microsoft/phi-4) 登陆 **OpenRouter**，其具备升级的文本生成能力、更低的延迟以及针对 AI 应用的部分代码处理能力。
   - 用户注意到通用性能的提升，并讨论了可能的集成路径，将 **OpenRouter** 视为扩展实验的枢纽。
- **Friday Agents 灵活框架**：位于 [GitHub - amirrezasalimi/friday-agents](https://github.com/amirrezasalimi/friday-agents) 的 **Friday Agents** 多 Agent **JavaScript** 技术栈正式推出，提供两个核心部分，通过内置并发简化了 AI 应用开发。
   - 开发者称赞其处理并行任务的能力，并建议 **OpenRouter** 模型端点可能会为该架构带来更广泛的功能。
- **Telegram 通过 DeVries 接入 200 多个 LLM**：位于 [devriesai.com](https://devriesai.com/) 的 **DeVries AI Chatbot** 允许通过 Telegram 直接访问 **200 多个大语言模型**，价格为每月 24.99 美元，并提供免费试用以吸引早期用户。
   - 社区成员强调了其简化多模型使用的能力，突出了在单个聊天界面中切换不同供应商的便利性。
- **Mistral 的 Codestral 提升上下文容量**：**Mistral** 推出的全新 **Codestral** 模型（发布于 [mistral.ai/news/codestral-2501/](https://mistral.ai/news/codestral-2501/)）拥有 **262K** 上下文和更快的编程速度，但已从公开发布中撤回。
   - 参与者提到该模型在移除前曾短暂可用，尽管其编程基准测试表现强劲，但仍引发了关于其是否已具备生产就绪能力的辩论。
- **LLM 成本讨论与 Deepseek V3 反馈**：讨论者比较了不同平台的大语言模型托管方案，并认为 **Deepseek V3** 是一个强有力的选择，具有稳定的速度和合理的价格。
   - 他们还权衡了各供应商之间的性能差异，并指出在 **OpenRouter** 上成为模型托管商的途径是一个关键关注点。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Anthropic 估值攀升至 600 亿美元**：近期，**Anthropic** 的估值飙升至 **600 亿美元**，引发了关于 **语言模型初创公司** 未来的热议，人们对其即将推出的产品扩展以及 **主要投资者** 的兴趣充满猜测。
   - 在社区讨论中，参与者将其描述为整个 AI 行业的“巨大炒作”，暗示更多的高估值可能会引发潜在竞争者之间的激烈竞争。
- **Sonar 3.3 现身，但 API 缺席**：成员们在 Perplexity 的 Web UI 中发现了 **Sonar 3.3**，但在 **公开 API** 中未见其踪影，这引发了关于发布时间表和官方公告的疑问。
   - 许多用户表示对更多 **llama-3.1-sonar** 变体感兴趣，并在 Perplexity 尚未发布正式声明的情况下，*猜测可能会有 70B 版本*。
- **Perplexity 对决 Claude：模型之争**：爱好者们争论 **Perplexity** 在实际任务中是否优于 **Claude**，引用了一些速度测试和用户体验，但没有定论。
   - 一些人坚持认为 **Claude** 在某些领域表现出色，而 *Perplexity 粉丝* 则赞扬其整体界面以及 **llama-3.1-sonar** 中的 **引用 (citations)** 等功能，这加剧了围绕可靠性和性能的持续辩论。
- **芯片与堆叠：3D AI 热潮**：社区成员关注新兴的 **AI 芯片**，包括 **MIT 的 3D 堆叠设计**，强调了更显著的数据处理增益。
   - 他们乐观地认为，这些即将推出的芯片中扩展的内存将 *支持要求更高的本地模型托管*，特别是针对 **LLM** 工作负载。
- **Perplexity 的定价困境**：用户对 **Perplexity** 的订阅层级表示不满，将 **每月 200 美元** 的方案与 **ChatGPT** 进行比较，同时呼吁更具吸引力的专业级定价。
   - 许多人报告了性能缓慢和 **API** 使用受限的问题，*建议* Perplexity 优化其定价策略并提高稳定性以保持竞争力。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Codestral 25.01 榜单攀升**：新升级的 **Codestral 25.01** 飙升至 LMsys copilot arena 排行榜第 1 名，展示了更高的效率和性能（[官方新闻](https://mistral.ai/news/codestral-2501/)）。
   - 它在 Aider polyglot 基准测试中得分 **11%**（[推文引用](https://x.com/paulgauthier/status/1878886495609815054)），引发了成员们对其与领先模型对比表现的关注。
- **Helium-1 瞄准移动端规模**：**Kyutai 的 Helium-1** 作为一个 2B 参数的骨干语言模型问世，专注于边缘设备并支持 6 种语言（[公告](https://kyutai.org/2025/01/13/helium.html)）。
   - 贡献者强调 **privacy**（隐私）和速度是主要目标，并指出 Helium-1 在极低延迟的个人 AI 系统中具有潜力。
- **Qwen 2.5-Math 模型大幅提升准确率**：**Qwen 2.5-Math-PRM-72B** 系列引入了 Process Reward Models（过程奖励模型），以减少数学推理中的错误（[Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B)）。
   - 成员们报告了在分步逻辑上的改进，强调了**更少的中间环节失误**以及在各项数学评估中持续强劲的表现。
- **Sky-T1-32B-Preview 以极低预算实现高性能**：[Sky-T1-32B-Preview](https://novasky-ai.github.io/posts/sky-t1/) 的训练成本低于 **$450**，展示了与大型闭源模型相当的推理能力。
   - 其开源代码库（[SkyThought GitHub](https://github.com/NovaSky-AI/SkyThought)）指向了更多社区驱动、**low-cost**（低成本）的高级 LLM 开发方向。
- **LoRa 微调助力 Qwen Instruct**：一位成员使用 **LoRa** 在分布外数据集上微调 Qwen Instruct 模型，旨在保持特定领域任务的性能。
   - 他们报告了一些训练挫折，但仍对 LoRa 在专业用例中实现稳健适配的能力保持乐观。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R+ 势头强劲**：Cohere 发布了 **Command R+** 的最新性能细节，参考了多篇博客文章，如 [Command R: RAG at Production Scale](https://cohere.com/blog/command-r) 和 [Introducing Command R7B](https://cohere.com/blog/command-r7b)。这些更新涵盖了企业级 LLM 任务的高级功能，重点介绍了速度、上下文长度和更简便的微调。
   - 社区讨论展示了 **Command R+** 在 **Rust** 和 Python 中的应用，赞扬了其在代码生成方面的效率，并链接到 [官方文档](https://docs.cohere.com/v2/docs/command-r-plus) 以获取更深入的见解。一位用户表示 *“Command R+ 让复杂的查询变得更容易处理”*，呼应了大众对工作流改进的兴奋。
- **Cohere 处理大型数据集**：一些用户测试了上传高达 **800MB**、超过 **180,000 行** 的 JSONL 文件，探索大规模数据流的可行性。他们在数据集环境中发现了挑战，并暗示 **enterprise-level**（企业级）的使用可能需要专门的解决方案。
   - 成员们对扩展用于训练和微调的数据摄取（data ingestion）感到好奇，并参考了 **Command R+** 的扩展应用。目前关于优化 **big data**（大数据）摄取流程的讨论非常活跃，希望官方文档能进一步明确最佳实践。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Claude & O1: 协同合作**：成员们分享道，唯一成功的 **O1 工作流**涉及使用 **Claude** 来明确项目目标、创建指令并定义函数间的**接口（interfaces）**。他们强调，一旦经过适当的 Prompt 引导，O1 能够高效地处理**算法**。
   - 一位参与者对该小组是否最适合进行此类深入的 O1 讨论表示怀疑，暗示存在兴趣不匹配的情况。这反映出社区内部希望对 **O1** 进行更专业化关注的愿望。
- **Triton 调优策略**：在真实 GPU 上优化 **Triton Puzzles** 的努力（引用[此 repo](https://github.com/gauravjain14/mlcompilers_and_kernels/tree/main/triton_kernels)）包括在 **A100** 与 **A30** 上的自动调优（autotuning），以及讨论大 `num_stages` 的内存限制。另一位用户研究了 Kernel 占用率（occupancy），担心每个 **CUDA block** 运行多个程序可能会影响小数据块的性能。
   - 他们还探索了改进**交叉熵（cross entropy）** Kernel 以减少开销的方法，参考了 [Liger Kernel 代码](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py)。关于性能分析（profiling）和超参数的反馈再次确认了 Triton 的灵活性，尽管消费级 GPU 需要仔细关注共享内存（shared memory）的使用。
- **推进 CUDA 与 HPC**：成员们讨论了在 **Ubuntu** 上安装 **CUDA**，参考了[官方指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu)并使用了 [Nsight Visual Studio Code edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition) 插件。该小组对 **Blackwell** 线程块集群（thread block clustering）表示好奇，并详细询问了 **H100** 与 **H200** 相比时 **FA3** 的性能表现。
   - 他们强调了 **GPU** 的复杂性，如 Block 分配，并将这些学习成果与跨不同计算架构的 HPC 任务联系起来。围绕驱动程序设置、插件使用和 HPC 扩展的担忧仍然是参与者关注的核心话题。
- **Torch 的尝试与突破**：记录了 **PyTorch Profiler** 在使用 Hugging Face **Transformer** 的 trainer.py 时出现的 **UTF-8 解码问题**，参考了 [issue #64345](https://github.com/pytorch/pytorch/issues/64345)。讨论还集中在将 **Flash Attention** 与 MultiheadAttention 集成，以及 **DDP** 和 **FSDP** 对前向传播（forward pass）之外模块使用的影响。
   - 构建**大模型推理流水线**的成员使用了 Meta Devices 和缓存中间状态来管理内存，尽管每个请求访问所有层构成了挑战。**NNSight** 被强调为一种按需流式传输激活值（activations）的方法，从而减少高级分析过程中的显存溢出（out-of-memory）陷阱。
- **活动与 LLM 演进**：即将举行的演讲涵盖 1 月 24 日的 **Flash Infer**、1 月 25 日的 **Mosaic GPU**、2 月 8 日针对 **Turing** 的 **int8 matmul** 以及 2 月 14 日的 **NVIDIA profiling** 等。同时分享了一个新的 **Maya** 多语言视觉语言模型（[链接](https://twitter.com/nahidalam/status/1866667770114609217)）。与此同时，**Qwen2-VL** 与 **Liger Kernel** 产生冲突，导致根据[此 issue](https://github.com/linkedin/Liger-Kernel/issues/515)需要降级 **Transformers**。
   - Meta 发布了针对 **GenAI 推理**的 GPU 相关职位空缺，感兴趣的候选人可前往[其招聘网站](https://www.metacareers.com/jobs/1517576482367228/)。其他非话题更新包括 **Sonoma AI 演讲系列**、创意筹款想法，以及社区内更多坦诚的 GPU 兴趣讨论。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **社区攻克 MAX GPU 与 MAX-CV**：2025 年的首次社区会议在活跃的问答环节中重点讨论了 **MAX GPU 基准测试**和 **MAX-CV**，并承诺在[此处](https://discord.com/events/1087530497313357884/1300880439673884712)提供录像。
   - 尽管时间冲突阻碍了一些成员参与，**Chris Lattner** 仍回答了相关提问，而 **Caroline Frasca** 承诺后续会更新视频。
- **macOS Mojo 测试升温**：志愿者们在 macOS 上运行 **Mojo** 代码进行跨平台检查，并通过私信加强了协作。
   - 他们发现通过在[文档网站](https://docs.modular.com)切换版本号可以查看 **nightly 文档**，满足了开发者们的好奇心。
- **异步提案激发 Mojo 热情**：[Mojo 的结构化异步 (Structured Async)](https://github.com/modularml/mojo/pull/3945) 和 [提供的效应处理器 (Provided Effect Handlers)](https://github.com/modularml/mojo/pull/3946) 这两项计划旨在不牺牲性能的前提下集成异步特性。
   - 贡献者们比较了受 **Rust** 启发的异步方法，进一步推动了关于 **Mojo** 并发性的讨论。
- **Mojo 编译器崩溃问题被修复**：在定义实现共享 trait 的 struct 列表时发生的崩溃已在 [Issue #3944](https://github.com/modularml/mojo/issues/3944) 中记录。
   - 开发者反馈将其归因于复杂的初始化问题，促成了官方 Bug 报告及代码修复建议。
- **Int8 到 String 转换的怪异现象**：一份 [Mojodojo 指南](https://github.com/modularml/mojo/issues/3947) 指出了将 **Int8** 转换为 string 时的困扰，令测试者感到意外。
   - 讨论涉及了编译时与运行时的类型细节，引导用户参考 [Modular 文档](https://docs.modular.com/mojo/manual/types/) 以获取清晰解释。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Substack 文章探讨 Agentic AI**：感谢[这篇 Substack 文章](https://kanesimms.substack.com/p/what-agentic-ai-actually-is-a-deeply)，读者可以深入研究 **Agentic AI** 的概念及其背后的复杂性。
   - 虽然讨论较为简练，但它为关于 **AI 决策能力**和自主性的深入观点奠定了基础。
- **AzureOpenAI 集成示例备受关注**：一段代码示例展示了如何使用显式 API 凭据和参数设置 **AzureOpenAI**，并引用了 [Azure OpenAI 文档](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview)。
   - 该示例展示了直接的使用模式，说明了工程师如何快速上手 **Azure** 服务。
- **dspy.react 与 phi-4：令人惊讶的函数调用**：一位用户注意到 **dspy.react** 让 **phi-4** 实现了函数调用 (function calling)，尽管该模型在此能力上的训练极少。
   - 虽然并非完美，但这一演示表明基础函数调用可以嵌入到 **phi-4** 中以实现灵活使用。
- **DSPy 社区流传语音 AI 雄心**：一位新成员询问关于使用 **DSPy** 开发语音 AI 的事宜，但获知目前尚无直接的音频支持。
   - 他们被引导至 [GitHub Issue #2037](https://github.com/stanfordnlp/dspy/issues/2037)，该议题记录了关于**语音**能力的请求及未来可能的扩展。
- **Prompt 性能差异引发辩论**：一些用户比较了 **gemini-8b** 与 **deepseekv3** 的 prompt 表现，怀疑针对特定模型的 prompt 可能会产生不同的结果。
   - 其他人指出，相同的 prompt 设计可能无法解决不同架构间的核心错误，这强化了 **prompt 专门化 (prompt specialization)** 的观点。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Phi-4 文件热潮**：一位用户请求用于 **Phi-4** 微调的“占位”文件，并分享了 [这个 Colab 笔记本](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb)，并提到即将发布的 **Phi-4 PR** 可能会让该文件变得不再必要。
   - 他们预计该 PR 很快会被合并，暗示工作流可能会平滑过渡，无需独立文件。
- **自适应批处理 (Adaptive Batching) 讨论**：一位贡献者提交了 **Torchtune** 中 [自适应批处理 (adaptive batching) 的 RFC](https://github.com/pytorch/torchtune/pull/2199)，旨在动态优化 batch size。
   - 他们计划在下一轮迭代进行进一步修改前先整合反馈。
- **Instruct 与 Non-Instruct 在医疗领域的收益对比**：讨论了使用 **instruct** 或 **non-instruct LLaMA 模型** 在 50B-token 医疗数据集上进行训练，并提到 10B instruct 版本是一个可能的候选。
   - 他们强调，广泛的数据集清洗和有效的后处理对于实现强大的医疗能力至关重要。
- **数据质量胜过一切**：一位成员强调 **数据质量 > 数据数量**，认为经过良好处理的数据集优于海量的原始数据。
   - 他们建议在投入大量资源进行训练之前，先使用其他 LLM 来评估文档的相关性。
- **Mistral 7B 表现出色**：用户分享了一项研究，其中 **Mistral 7B** 在医疗协会指南的预训练任务中表现出色。
   - 他们将这些积极成果归功于精选的数据集，突出了选择合适训练材料的重要性。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 立即报名且完全免费**：填写 **SP 25 报名表** 即可自动免费参加 LLM Agents MOOC，让所有人都能无需额外步骤直接加入。
   - 组织者确认这是 *完全免费的*，这激发了急于加入的学习者的热情。
- **期待最终项目结果**：课程负责人表示，最终项目结果预计在本月晚些时候公布，可能就在一周内。
   - 社区正处于紧张状态，热切期待关于评分细节和未来奖项的官方公告。
- **1 月 27 日开课：学习开启**：2025 年春季 LLM Agents MOOC 的 **每周讲座** 将于 **1 月 27 日** 开始，为参与者设定了明确的时间表。
   - 讲师提醒大家标记日历，为高强度的学习体验做好准备。
- **通过独立的 Google Forms 提交作业**：MOOC 中的每项作业都需要通过独立的 Google Form 提交，以便通过电子邮件准确跟踪进度。
   - 学生必须始终使用 *同一个电子邮箱地址*，以简化评分流程并避免混淆。
- **通过 2024 年秋季讲座评估速成课程难度**：[此链接](https://llmagents-learning.org/f24) 提供的 **2024 年秋季 MOOC** 材料为新手提供了基础内容的参考。
   - 负责人指出春季课程会 *稍难一些*，但建议查看存档讲座和 [Quizzes Archive - LLM Agents MOOC](https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit) 以做好充分准备。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AI Builders Summit 展示 40 多位演讲者**：AI Builders Summit 宣布在为期 4 周的在线培训中将有超过 **40 位演讲者**参与，重点介绍在企业级工作中使用小型语言模型（small language models）。来自 [@_odsc](https://twitter.com/_odsc) 的额外信息确认了由 [@seldo](https://twitter.com/seldo) 等专家主持的 **以 RAG 为中心的会议**。
   - 与会者计划学习在不牺牲性能的情况下实现 **RAG**（检索增强生成）的 **scaling** 策略，并从经验丰富的演讲者那里获得直接指导。
- **AutoRAG 提升 RAG Pipelines 性能**：新推出的 **AutoRAG** 框架通过系统地测试多种方法，帮助开发者为 **RAG** 选择有效的配置。根据论文，它为希望在 **RAG** 设置中获得更高精度的 **LlamaIndex** 用户提供了一条结构化路径。
   - 社区成员认为 **AutoRAG** 是一项显著的增强，称赞其在简化 Pipeline 决策和优化性能方面的潜力。
- **机器人项目寻求 LlamaIndex 工程师**：一位用户正在寻找精通 **LlamaIndex** 的工程师协助设计机器人解决方案，并提供付费咨询。有兴趣的专业人士被要求通过私信分享资历证明。
   - 其他人强调，在 **structured data retrieval**（结构化数据检索）和 **prompt engineering** 方面的成熟经验对该职位至关重要。
- **GraphRAG 图谱仅显示节点**：一些用户发现 **GraphRAG** 的 notebook 仅显示节点而没有连接边，即使使用默认的 **OpenAI** 模型也是如此。此问题被认为与潜在的数据缺失或遗漏了 **fine-tuning** 步骤有关。
   - 建议包括查看 [property_graph_neo4j notebook](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_neo4j.ipynb) 等示例，以确认正确的关系和配置。
- **Prompt Caching 与变量技巧**：多位用户讨论了 **OpenAI** 模型的 **prompt caching**，指出它以内置方式工作，与 **Anthropic** 的示例不同。他们提到官方参考资料有限，但建议许多调用会自动触发缓存。
   - 其他人探索了向 `QuestionsAnsweredExtractor` 添加动态变量的方法，建议在 **LlamaIndex** 中使用 **function mappings** 以轻松馈送自定义上下文。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 中的 EPUB 探索**：一位用户询问 **GPT4All** 是否可以读取 **.epub** 文件，团队确认了基础支持，但指出在处理 **中文** 等特定语言时存在问题。
   - 他们建议参考 [GPT4All 文档](https://docs.gpt4all.io/gpt4all_desktop/settings.html#sampling-settings) 以寻找潜在的变通方法，并强调了语言处理的一致性。
- **Llama 的 Jinja Prompt 难题**：一位用户在为 **fine-tuned** 的 **Llama** 模型创建 **Jinja prompt template** 时遇到困难，因为 `get_chat_template()` 未能按预期工作。
   - 他们寻求在 **GPT4All** 中自定义 Prompt 设计的指导，强调了 **prompt engineering** 的复杂性。
- **上下文长度限制引发关注**：贡献者确认 **GPT4All** 对对话召回强制执行约 **2048 tokens** 的限制，如果超过该限制则会截断文本。
   - 他们指出这会影响聊天输入和基于文件的引用，因此在进行较长时间的会话时需要仔细规划。
- **全量聊天导出功能缺失**：一位用户希望拥有 **full-chat exporting** 功能，以便在无需手动复制的情况下检索过去的对话日志。
   - **GPT4All** 团队目前尚未提供此功能，并鼓励在 [GitHub issues 页面](https://github.com/nomic-ai/gpt4all/issues)提交请求。
- **从性能较弱的笔记本远程运行 GPT4All**：一位用户旨在通过 **VPN** 或在性能更强的台式机上设置反向代理，将性能较弱的笔记本连接起来远程运行 **GPT4All**。
   - 这种方法利用了主机的硬件性能，让用户在保留本地便利性的同时卸载处理任务。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 的整洁张量编译器 (Tidy Tensor Compiler)**：参与者解释了 **Tinygrad** 如何使用最小指令集和 **kernel fusion**（算子融合）进行 GPU 优化，并参考了 [toonygrad/PLAN.md](https://github.com/tinygrad/toonygrad/blob/master/PLAN.md)。
   - 他们指出，这些融合后的 kernel 可以在多种硬件上执行，并将该设计比作简化 ML 操作的 *LLVM 方法*。
- **周一的 #53 会议动态**：团队成员将 **Meeting #53** 安排在圣地亚哥的 **上午 9:30**，讨论内容涉及 **DSP 合约**、**Python 速度**以及 **MLPerf BERT** 评估。
   - 他们提到了未来关于 **Tensor cores** 和 **RetinaNet** 的悬赏任务（bounties），并对驱动程序的奇特行为（driver quirks）和 **ONNX** 集成提出了警告。
- **过期 PR 与 FSDP 悬赏锁定**：呼吁关闭过期的 pull requests，并讨论了 [PR #8571](https://github.com/tinygrad/tinygrad/pull/8571) 中关于 **FSDP** 的悬赏。
   - 悬赏条件强调了 *多 GPU 训练* 的要求，引发了对超越单 GPU 扩展性的分析。
- **Checkpointing 与内存管理魔法**：一位用户询问了 **activation checkpointing**（激活检查点）方法，以在保持训练效率的同时减少 **Tinygrad** 中的内存开销。
   - 他们还寻求在不破坏 **gradient context**（梯度上下文）的情况下为返回张量 *释放内存* 的方法，突显了对资源处理技巧的迫切需求。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 安装成功**：一位用户在通过 [Homebrew](https://brew.sh) 和 [pipx](https://github.com/pypa/pipx) 安装 **Open Interpreter** 时遇到了 **tiktoken** 错误和缺失 Rust 依赖的问题，最终实现了稳定运行。
   - 他们提供了一个用于创建干净环境的简短命令列表，再次证明 **pipx** 是隔离 Python 应用程序的一种简单方法。
- **命令闪击：Open Interpreter 隐藏的屏幕功能**：安装完成后，一位用户确认 **Open Interpreter** 可以运行任意命令，包括视频编辑步骤。
   - 一个较少人知的 **screen control**（屏幕控制）功能引发了对其潜在扩展的兴奋，激发了大家对使用场景的好奇。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Stable Audio 3 加速开源**：开发者宣布 **Stable Audio 3** 将会[开源](https://vxtwitter.com/dadabots/status/1878505508711157791)，该模型在音乐上进行训练，旨在用于创意音频项目。
   - 爱好者们指出，这种方法可以加强社区驱动的协作，特别是专注于重用和混音 **music-based datasets**（基于音乐的数据集）。
- **寻找高血压音频数据集**：一位成员询问是否有通过音频记录识别高血压的**数据集**，请求在针对健康研究的数据收集方面提供帮助。
   - 他们强调了通过**协作**汇编音频样本的重要性，希望能填补专业健康数据方面的空白。
- **Megatron Checkpoint 转换探索**：一位用户运行了 Megatron 训练，希望获得一个能将 **torch format** 转换为 **HF format** 且不依赖 Nemo 的脚本，以避免手动修改。
   - 他们称这将 *“节省大量工作”*，并请求社区分享任何现有的 checkpoint 转换代码或参考资料。
- **MegaTron-LM 克隆参考**：一位用户克隆了官方的 **NVIDIA MegaTron-LM** 仓库（commit `31a29b87`），并提到训练日志存储在[此处](https://fz-juelich.sciebo.de/s/Yh8Q8RRTxliupLh)。
   - 他们注意到**权限**限制了直接文件上传，因此呼吁使用其他文件共享方法以增加社区的参与。

---

**MLOps @Chipro Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**Axolotl AI Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：各频道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1327369162376482877)** (1120 messages🔥🔥🔥): 

> `微调 Llama 3.3, 使用 Unsloth 处理 AI 模型, 性能指标, GPU 与云解决方案, 聊天模板与分词 (Tokenization)`

- **微调 Llama 3.3 模型**：用户正在寻求微调 Llama 3.3 70B 模型的帮助，并讨论了使用 Unsloth 和 AutoTrain 等不同方法。
   - 有人指出，正确配置模板和处理数据集对于在训练期间获得良好结果至关重要。
- **利用 Unsloth 处理 AI 模型**：Unsloth 用户强调了其在模型微调方面的高效性，特别是长上下文（long context）功能和降低的 VRAM 需求。
   - 有建议使用特定的 notebook 来运行 ORPO 等方法，以增强模型性能。
- **AI 模型的性能指标**：参与者讨论了基于 GPU 配置的预期性能结果，并提到了 LLM 的基准测试速度。
   - 有人对异常输出以及模型 prompt 和配置的潜在问题表示担忧。
- **为 AI 任务选择合适的 GPU**：一位用户询问了满足其需求的 AMD GPU 替代方案，并被告知 Unsloth 目前不支持它们。
   - 建议关注 NVIDIA 选项以获得更好的兼容性和性能，以及基于云的解决方案。
- **Chat Templates 的重要性**：强调了与 Phi-4 等模型交互时使用正确 chat templates 的必要性，以提高响应质量。
   - 用户获知了这些模板在推理过程中生成连贯且相关输出的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/ibm-granite/granite-3.1-8b-base">ibm-granite/granite-3.1-8b-base · Hugging Face</a>: 未找到描述</li><li><a href="https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-1-8b-conversational-unsloth/notebook"> Kaggle Llama 3.1 8b Conversational Unsloth</a>: 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://www.gigabyte.com/Motherboard/TRX40-DESIGNARE-rev-10#kf">TRX40 DESIGNARE (rev. 1.0) 主要特性 | 主板 - GIGABYTE Global</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/Yahir/test">Yahir/test · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/llama3-3">使用 Unsloth 微调 Llama 3.3</a>: 微调 Meta 的 Llama 3.3 (70B) 模型，其性能优于 GPT 4o，通过 Unsloth 开源实现提速 2 倍！对初学者友好。现已支持 Apple 的 Cut Cross Entropy 算法。</li><li><a href="https://x.com/UnslothAI/status/1877779176473944212">来自 Unsloth AI (@UnslothAI) 的推文</a>: 现在可以在 Colab 上免费微调 Phi-4 了！Unsloth 微调 LLM 的速度快 2 倍，显存 (VRAM) 占用减少 70%，上下文长度增加 12 倍——且无精度损失。GitHub 仓库：https://github.com/unslothai/unslothDocumenta...</li><li><a href="https://x.com/abacaj/status/1876315285428609240">来自 anton (@abacaj) 的推文</a>: XML 非常有效，基本上是我发现的唯一能正确引导 LLM 作为 Agent 的方法</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/updating">更新 | Unsloth 文档</a>: 按照以下步骤更新 Unsloth：</li><li><a href="https://huggingface.co/microsoft/phi-4/discussions/7">microsoft/phi-4 · 函数调用 (Function Call) 支持</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/unsloth-4-bit-dynamic-quants-67503bb873f89e15276c44e7?">Unsloth 4-bit 动态量化 - Unsloth 集合</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/phi-4-GGUF">unsloth/phi-4-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">初学者？从这里开始！ | Unsloth 文档</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/releases/tag/2025-01">发布 Phi-4 及 Bug 修复 · unslothai/unsloth</a>: 如果你看到显著或异常的 Loss 结果，请更新 Unsloth——最新更新修复了由新版 Transformers 引起的问题。请在此查看我们的新更新说明。Phi-4 是....</li><li><a href="https://huggingface.co/unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit">unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements">Unsloth 需求 | Unsloth 文档</a>: 这里是 Unsloth 的需求，包括系统和 GPU VRAM 需求。</li><li><a href="https://docs.unsloth.ai/basics/reward-modelling-dpo-orpo-and-kto">奖励建模 - DPO, ORPO & KTO | Unsloth 文档</a>: 要在 Unsloth 中使用 DPO, ORPO 或 KTO，请按照以下步骤操作：</li><li><a href="https://huggingface.co/microsoft/phi-4/blob/main/tokenizer_config.json#L774>">tokenizer_config.json · microsoft/phi-4 (main 分支)</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/pull/1533">让 GGUF 文件名与项目名称一致，由 sebaxakerhtc 提交 · Pull Request #1533 · unslothai/unsloth</a>: 为什么？默认文件名是 &quot;unsloth&quot; —— 这很好，但当你在 HF 上有多个模型并尝试将它们下载到 OpenWebUI 时，它们的名字都叫 &quot;unsloth&quot;...</li><li><a href="https://gist.github.com/darkacorn/71658f280ea0fc0ad4b97d2a616f4ce8">100k 测试 . exllama2(测试分支) + fa 1 - 128t 步内完成 100k</a>: 100k 测试 . exllama2(测试分支) + fa 1 - 128t 步内完成 100k - gist:71658f280ea0fc0ad4b97d2a616f4ce8</li><li><a href="https://github.com/unslothai/unsloth/issues/698">protobuf 版本 · Issue #698 · unslothai/unsloth</a>: 有什么原因导致 protobuf 被固定在 4 以下吗？我使用的一些其他包需要 protobuf &gt;=4，所以我无法将 Unsloth 与其他包一起安装。只是想了解...</li><li><a href="https://analyticsindiamag.com/ai-trends/6-open-source-llms-that-can-run-on-smartphones/">6 款可在智能手机上运行的开源 LLM</a>: 通过在智能手机上利用 LLM 的力量，在不联网的情况下最大限度地提高隐私和控制力。</li><li><a href="https://github.com/huggingface/transformers/pull/34858">🧹 移除弃用的</a>

d RotaryEmbedding parts in the Attention layers by Cyrilvallez · Pull Request #34858 · huggingface/transformers</a>：这个 PR 做了什么？它清理了（已过期的）弃用的 rotary embeddings 周期，并将其完全从 Attention 移动到 Model 中。同时还删除了弃用的 EmbeddingClasses，以及...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1327465866161291326)** (23 messages🔥): 

> `AI Chatbot Creation, Transcribing Videos, AI for Language Learning, Llama Model Usage, Voice Modes in AI` 


- **创建一个模仿主播的 AI Chatbot**：要开发一个表现得像主播的 AI chatbot，你需要先转录你的视频，以便对对话模型进行 fine-tuning，并确保实时 TTS 功能。
   - 利用 [Google Voice Recognition](https://link.to.google-voice) 或 [Whisper](https://huggingface.co/whisper) 等模型可以帮助进行语音转文本，而 TTS 选项包括免费模型如 **Fish Agent** 和 **Kokouro**，或付费服务如 **Eleven Labs**。
- **作为 Flutter 开发者开始 AI/ML 之路**：鼓励 AI/ML 领域的新开发者（尤其是使用 Flutter 的开发者）尝试 [DuckDuckGo](https://duck.ai) 等平台，以便在没有设置障碍的情况下轻松访问 LLM。
   - 创建结构化的学习路径或使用原型来探索 LLM 的能力，有助于明确 AI/ML 领域的目标和预期成果。
- **使用 Llama 模型进行语言学习**：对于学习新语言，使用 **Llama 3.1-70B** 等模型非常有效，因为它们能够很好地遵循指令并生成创意内容。
   - 尝试不同的 Persona 和动态用户定义过滤器也可以通过让交互感觉更连贯和有趣来增强学习体验。
- **关于 TTS 转录的见解**：对于主流语言以外的 TTS 系统训练，拥有超过 **10,000 小时**正确转录的音频对于获得连贯的输出至关重要。
   - 针对中文、日文或韩文等语言使用 **Cosy** 等模型可以简化语言学习过程。
- **使用 Llama 的创意场景**：用户喜欢尝试 Llama 的动态过滤器来创建富有想象力的场景，这可以产生幽默且引人入胜的叙事。
   - 玩转各种角色 Prompt 可以带来有趣的故事讲述，比如 Whiskers 夫人在邮差到达时追逐她的猫，而她的丈夫则在表演长号独奏。



**Link mentioned**: <a href="https://duck.ai">DuckDuckGo AI Chat at DuckDuckGo</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1327366553888362527)** (236 messages🔥🔥): 

> `Fine-tuning LLMs, Data Preparation and Augmentation, LoRA for Style Transfer, Challenges in AI and NLP, Using Pre-trained Models` 


- **针对特定风格 Fine-tuning LLM**：用户讨论了 fine-tuning LLM 以模仿特定作者风格的可能性，该过程涉及大量的数据准备和迭代。
   - 使用 LoRA 等工具可以简化 Style Transfer，但理解底层模型机制对于有效的 fine-tuning 至关重要。
- **数据准备的挑战**：准备文本数据集涉及策划和清洗，这对于有效部署 fine-tuned 模型来说可能非常耗时。
   - 正确的数据准备至关重要，因为它构成了训练 LLM 以实现预期结果的大部分工作。
- **AI 项目的时间投入**：用户指出，从零开始使用 LLM 获得可用结果大约需要 6 周的学习和实验。
   - AI 项目的复杂性要求对基础概念有扎实的理解，而不仅仅是表面知识。
- **探索 AI 开发中的选项**：个人考虑了聘请 AI 开发专家的潜在途径，特别是围绕尊重受版权保护材料的使用。
   - 虽然外包是一个选项，但成本可能会超过个人项目的娱乐性质。
- **保持项目的精力**：讨论强调了在追求个人项目时精力维护和动力（Motivation）的重要性，特别是在慢性疾病的背景下。
   - 参与者强调保持健康至关重要，因为疲劳会显著影响一个人参与耗时的学习过程的能力。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/TinyLlama_(1.1B)-Alpaca.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://ollama.com/vanilj/phi-4-unsloth">vanilj/phi-4-unsloth</a>: 来自 Unsloth 的带有固定 tokenizer 的 Phi 4 模型</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit">unsloth/Llama-3.2-3B-Instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://cohere.com/llmu/what-is-semantic-search">什么是语义搜索？</a>: 在 LLM University 的这一章节中，你将学习如何使用 embeddings 和相似度来构建语义搜索模型。</li><li><a href="https://discuss.huggingface.co/t/perhaps-your-features-output-in-this-case-have-excessive-nesting-inputs-type-list-where-type-int-is-expected/135553">也许你的 features（在本例中为 `output`）具有过多的嵌套（预期为 `int` 类型，实际为 `list` 类型）</a>: 我也遇到了类似的问题。ValueError: Unable to create tensor, you should probably activate truncation and/or padding with &#39;padding=True&#39; &#39;truncation=True&#39; to have batche...</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>: 以下是我们所有 notebooks 的列表：</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Qwen2.5-Math-7B-Instruct">unsloth/Qwen2.5-Math-7B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1qN1CEalC70EO1wGKhNxs1go1W9So61R5?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/1528">transformers 4.48 破坏了 Mistral FastLanguageModel.from_pretrained，报错 NameError: name &#39;MistralConfig&#39; is not defined · Issue #1528 · unslothai/unsloth</a>: 你好，我一直尝试在自己的服务器上运行 Mistral_v0.3_(7B)-Conversational notebook。我按照 notebook 的单元格操作，结果遇到了 NameError: name &#39;MistralConfig&#39; is not def...</li><li><a href="https://github.com/unslothai/unsloth/issues/787">如何从本地目录加载微调后的 lora adapter 和下载的模型进行推理？ · Issue #787 · unslothai/unsloth</a>: 你好，非常感谢如此出色的工作！我需要帮助加载已完成微调过程的 lora adapter。这是我的代码：model, tokenizer = FastLanguageModel.from_pretrained...</li><li><a href="https://github.com/unslothai/unsloth/issues/934">无法加载保存到本地目录的 unsloth 训练模型。 · Issue #934 · unslothai/unsloth</a>: 我使用 PEFT 将一个 unsloth 微调模型（基础模型：unsloth/gemma-2b-bnb-4bit）创建为 tar 文件并推送到 gcsBucket。我正在从 gcs bucket 下载 artifacts，解压文件...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1327372236054855682)** (2 条消息): 

> `Maya: Multilingual Vision-Language Model` 


- **Maya 的预印本发布引发关注**: 一位成员在 [Twitter](https://twitter.com/nahidalam/status/1866667770114609217) 上宣布发布了 **Maya: Multilingual Vision-Language Model** 的预印本。
   - 另一位成员表达了热情，表示：*'这非常酷，感谢分享'*。
- **社区对 Maya 的反应**: 社区成员对 **Maya** 的发布表现出积极的参与，对该模型的影响表示出兴趣。
   - 有人评论道：*'这非常酷，感谢分享'*，表明了同行之间的兴奋之情。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1327402901647392818)** (11 条消息🔥): 

> `DataCollatorForPromptCompletion, Unsloth 训练速度, 用于欺骗的网络安全 LLM, Unsloth 中的 Fine-tuning, 研究提交咨询` 


- **DataCollatorForPromptCompletion 构建了最小的输入输出交互**：提供的 `DataCollatorForPromptCompletion` Python 代码在保留输出的同时屏蔽了输入 token，旨在实现针对性的语言模型训练。
   - 该实现模仿了 `DataCollatorForCompletionOnlyLM`，但强调在没有指定分隔符的情况下处理输入。
- **Unsloth 声称 LLM 训练速度有激进的提升**：根据一篇博客文章，**Unsloth** 可使 **LLM** 训练速度提升 **30 倍**，并将显存占用减少 **60%**，从而在不损失准确性的情况下允许更大的 batch size。
   - 介绍中强调了重大优化，包括增强各种 GPU 架构性能的自定义 Triton kernel。
- **网络安全研究员提出专用 LLM**：一位网络安全研究员讨论了构建用于网络欺骗行动的专用 LLM，重点是基于不同角色（personas）进行 Fine-tuning。
   - 这种方法已经产生了超过 **1000 个唯一连接**，并通过与模拟对手的沉浸式交互帮助识别诈骗。
- **揭秘 Unsloth 中的 Fine-tuning 技术**：讨论强调，由于自定义 Triton kernel 和先进的数学算法，**Unsloth** 中的 Fine-tuning 速度更快，并有相关资源支持。
   - 用户提到了详细说明 Fine-tuning 速度提升的具体博客文章，现在的模型 Fine-tuning 速度提升了 **14 倍**，且大幅降低了 VRAM 占用。
- **关于网络安全演讲的研究提交查询**：一位成员征求了关于参加他们在一个关于自动化网络安全欺骗会议上演讲的意向，以及提交论文的场所。
   - 回复表示对该演讲感兴趣，而关于提交场所的建议仍保持开放，突显了该话题的相关性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/introducing">Introducing Unsloth</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth update: Mistral support + more</a>: 我们很高兴发布对 Mistral 7B、CodeLlama 34B 以及所有其他基于 Llama 架构的模型提供 QLoRA 支持！我们增加了 sliding window attention、初步的 Windows 和 DPO 支持，以及 ...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1327385193887305880)** (68 条消息🔥🔥): 

> `SmolLM-Corpus 发布，Grouped-Query Attention 分析，支持多图像输入的 VLMs，Context Attribution 研究，用户介绍与专业领域` 


- **SmolLM-Corpus 正式发布！**：**SmolLM-Corpus** 现已完成打乱并分片为 **23698** 个 `jsonl.zst` 文件，支持便捷的流式传输和内存解压，总大小仅为 **315GiB**。
   - 该数据集包含来自 **cosmopedia-v2** 和 **fineweb-edu-dedup** 的子集，鼓励用户访问 [Hugging Face 链接](https://huggingface.co/datasets/Avelina/smollm-corpus) 获取。
- **Grouped-Query Attention 探讨**：讨论了在 Grouped-Query Attention 中，鉴于权重矩阵几乎正交，是否可能对 Q.K^T 进行低秩近似（low-rank approximation），并认为这可以实现高效计算。
   - 用户辩论了关联不同 Query 权重矩阵的可行性，指出了在尝试保留变换的同时维持计算成本的挑战。
- **VLMs 中的多图像输入**：成员们分享了关于多种能够处理上下文内多张图像的 **VLMs** 的见解，特别提到了 **Pixtral** 和 **Qwen** 模型。
   - 已确认 **Qwen VLs** 支持此功能，这为关于模型能力的更广泛讨论做出了贡献。
- **Context Attribution 研究受到关注**：讨论了 **context attribution** 的概念，引用了一篇近期论文中提出的名为 **ContextCite** 的方法，该方法可以识别上下文中导致模型输出的具体部分。
   - 参与者注意到该领域正在进行的辩论，包括此类研究的潜在应用和影响。
- **社区新成员介绍**：几位新成员介绍了自己，分享了他们在 MLOps、计算语言学方面的背景以及研究兴趣，包括 Alignment 和统计学习理论。
   - 社区对欢迎新人并促进围绕 AI 发展的协作讨论表示兴奋。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/iScienceLuvr/status/1831220742626939248">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>: ContextCite: Attributing Model Generation to Contextabs: https://arxiv.org/abs/2409.00729 “我们引入了 context attribution 问题：精确定位导致……的上下文部分（如果有的话）”</li><li><a href="https://huggingface.co/datasets/Avelina/smollm-corpus">Avelina/smollm-corpus · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1327370221165482035)** (738 条消息🔥🔥🔥): 

> `Latro 模型，Process Reward Models (PRMs)，Chain-of-Thought (CoT) 推理，VinePPO 算法，Reinforcement Learning (RL)`

- **Latro 模型提供潜在改进**：Latro 模型旨在增强各个领域的推理能力，其结合强化学习（reinforcement learning）与 CoT 生成的使用方式，相比于 RLVR 等传统方法，可能会带来更好的结果。
   - 讨论强调了将其性能与 RLOO 等现有框架进行对比评估的重要性，以便衡量其在密集奖励（dense reward）设置下的具体改进。
- **过程奖励模型 (PRMs) 的挑战**：PRMs 因其通过减少中间错误来提高推理能力而受到探索，但在数据标注和评估方法论方面面临挑战。
   - 研究表明，在确保推理步骤的正确性方面，Monte Carlo 估计等方法的效果可能不如 LLM-as-a-judge 技术。
- **对奖励信号有效性的担忧**：有人担心使用基于对数概率（log probabilities）的软奖励（soft rewards）可能不足以引导模型建立有效的推理链，从而可能导致模型产生记忆而非理解。
   - 这凸显了需要更稳健的方法来确保在训练过程中保持并提高推理质量。
- **RL 训练中 KL 正则化的影响**：KL 正则化（KL regularization）被讨论作为稳定 Latro 模型学习的一种手段，因此有必要仔细考虑它如何与整体训练目标相互作用。
   - 通过逐步调整动作空间（action space），训练动态可能会产生更密集且包含更多信息的奖励。
- **VinePPO 等新算法的评估**：VinePPO 作为一种新颖的方法被引入，它增强了 RL 中的信用分配（credit assignment），与传统的价值网络（value networks）相比，在重推理任务上表现更好。
   - 该方法侧重于从每个推理步骤中进行采样，以提供更丰富的奖励信号，强调了在不同数据集上进行详细评估的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.05441">The GAN is dead; long live the GAN! A Modern GAN Baseline</a>: 广泛流传的一种说法是 GAN 难以训练，且文献中的 GAN 架构充斥着经验性技巧。我们提供了反驳这一说法的证据，并构建了一个现代 G...</li><li><a href="https://x.com/lifan__yuan/status/1875020673476944337?s=46">来自 Lifan Yuan (@lifan__yuan) 的推文</a>: @rm_rafailov 这里的过程奖励（process reward）被定义为“优势（advantage）”，即 Q 值的差异。因此直观上，即使存在常数基准（baseline），它也应该被抵消，对 r... 没有影响。</li><li><a href="https://arxiv.org/abs/2412.11006">Entropy-Regularized Process Reward Model</a>: 大语言模型 (LLMs) 在执行复杂的多步推理方面展现出潜力，但它们在数学推理方面仍然面临困难，经常出现系统性错误。一个有前景的解决...</li><li><a href="https://arxiv.org/abs/2501.07301">The Lessons of Developing Process Reward Models in Mathematical Reasoning</a>: 过程奖励模型 (PRMs) 成为大语言模型 (LLMs) 数学推理中过程监督的一种极具前景的方法，旨在识别和减轻...中的中间错误。</li><li><a href="https://arxiv.org/abs/2106.06431">Offline Reinforcement Learning as Anti-Exploration</a>: 离线强化学习 (RL) 旨在从固定数据集中学习最优控制，而无需与系统交互。在这种设定下的 Agent 应该避免选择那些后果...的动作。</li><li><a href="https://arxiv.org/abs/2501.06282">MinMo: A Multimodal Large Language Model for Seamless Voice Interaction</a>: 大语言模型 (LLMs) 和多模态语音-文本模型的最新进展为无缝语音交互奠定了基础，实现了实时、自然且类人的对话...</li><li><a href="https://arxiv.org/abs/2501.07542">Imagine while Reasoning in Space: Multimodal Visualization-of-Thought</a>: 思维链 (CoT) 提示已被证明在增强大语言模型 (LLMs) 和多模态大语言模型 (MLLMs) 的复杂推理方面非常有效。然而，它在复杂...中表现不佳。</li><li><a href="https://arxiv.org/abs/2310.04363">Amortizing intractable inference in large language models</a>: 自回归大语言模型 (LLMs) 通过下一步 Token 条件分布压缩来自训练数据知识。这限制了对这些知识的易处理查询，仅限于从头到尾的 a...</li><li><a href="https://arxiv.org/abs/2410.01679">VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment</a>: 大语言模型 (LLMs) 越来越多地应用于复杂的推理任务，这些任务在获得任何奖励之前需要执行多个复杂的步骤。正确地为这些步骤分配信用（credit assignment）是至关重要的...</li><li><a href="https://arxiv.org/abs/2402.05808">Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning</a>: 在本文中，我们提出了 R$^3$：通过逆课程强化学习 (RL) 学习推理，这是一种仅采用结果监督来实现过程监督优势的新方法...</li><li><a href="https://arxiv.org/abs/2408.16737v2">Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling</a>: 使用来自强语言模型 (LMs) 的高质量合成数据进行训练是提高 LMs 推理性能的常用策略。在这项工作中，我们重新审视了这一策略是否是计算最优的...</li><li><a href="https://arxiv.org/abs/2203.11171">Self-Consistency Improves Chain of Thought Reasoning in Language Models</a>: 思维链提示结合预训练的大语言模型在复杂推理任务上取得了令人鼓舞的结果。在本文中，我们提出了一种新的解码策略，即自一致性（self-consistency）...</li><li><a href="https://arxiv.org/abs/2412.01981">Free Process Rewards without Process Labels</a>: 与评估整个回答的结果奖励模型 (ORMs) 不同，过程奖励模型 (PRM) 逐步对推理轨迹进行评分，提供更密集、更精细...</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1328229388457611294)** (5 条消息): 

> `Mechanistic Interpretability 音频内容，Neel Nanda 关于 SAEs 的播客，每周 Mechanistic Interpretability 阅读小组` 


- **每周阅读小组的音频录音**：现有一年多以来部分成员参与的**每周 Mechanistic Interpretability 阅读小组**的纯音频录音。
   - 一名成员原计划使用 Whisper 对其进行转录，但由于实现困难，仅完成了一小部分。
- **Neel Nanda 关于 SAEs 的深度播客**：一名成员推荐了一集播客，由领导 Google DeepMind 的 Mechanistic Interpretability 团队的 **Neel Nanda** 讨论他在 SAEs 方面的工作，链接见[此处](https://open.spotify.com/episode/5XjHhNQxIb16eJZXGmbaCk?si=Z8LTnSo7QHGJkBxgGZbIJA)。
   - Nanda 强调了 **Mechanistic Interpretability** 的必要性，因为 Machine Learning 模型可以在没有清晰内部理解的情况下执行复杂任务。
- **播客反响热烈**：一名成员表示在播客发布的第一天就很喜欢它，说明该内容在听众中引起了很好的共鸣。
   - 这表明人们对 Mechanistic Interpretability 领域领军人物所提供的见解越来越感兴趣。



**提到的链接**：<a href="https://open.spotify.com/episode/5XjHhNQxIb16eJZXGmbaCk?si=Z8LTnSo7QHGJkBxgGZbIJA">Neel Nanda - Mechanistic Interpretability (Sparse Autoencoders)</a>：Machine Learning Street Talk (MLST) · Episode

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1327740240198238218)** (19 条消息🔥): 

> `Goodfire API 实现，MLQA Benchmark 澄清，Dataset 问题，GPT-4o 使用，Pre-commit 行尾问题` 


- **Goodfire API 寻求贡献者**：一名成员分享了他们在 Eval Harness 中实现了 **Goodfire API** 的基础版本，并在 **gsm8k_cot_llama** 任务上成功将 **Llama 8B** 与 **VLLM** 匹配。
   - 他们邀请其他人参与协作、排查故障或讨论增强该 API 功能的后续步骤。
- **关于 doc_to_text 字段的澄清**：一名成员询问了 YAML 中的 **doc_to_text 字段**，特别是它是否旨在根据某些输入构建 Prompt。
   - 其他人确认这确实是用于 Prompt 构建的，从而明确了其功能。
- **MATH-Hard Dataset 消失**：有人对 Hugging Face 上 **lighteval 账号**中的 **MATH-Hard Dataset** 被移除表示担忧，这导致了 Leaderboard 评估出现问题。
   - 通过 GitHub [issue 讨论](https://github.com/EleutherAI/lm-evaluation-harness/issues/2618#issuecomment-2583172531)分享了一个潜在的变通方法。
- **GPT-4o 使用指南**：一名成员请求一个使用 **with gpt-4o** 框架的示例，特别是如何使用 `generate_until` 函数实现任务。
   - 建议参考 **gsm8k** 作为创建任务的模板。
- **修复混合行尾问题**：讨论了 pre-commit 检查中 **混合行尾 (Mixed line ending)** 失败的问题，这导致了提交被阻止。
   - 建议运行 `pre-commit run --all-files` 来自动修复与不同操作系统相关的行尾字符问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/bbrabbasi">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/maziyarpanahi/status/1878491088199032913?s=46">来自 Maziyar PANAHI (@MaziyarPanahi) 的推文</a>：@ailozovskaya MATH-Hard 数据集是否已从 @huggingface 上的 lighteval 账号中移除？我无法再使用 lm-eval-harness 运行 leaderboard 评估了！:(</li><li><a href="https://colab.research.google.com/drive/14-KkodIIVdq5fB-rDBMKoAK3KiUOsrmB#scrollTo=qsKt8d6TVnC_">Google Colab</a>：未找到描述</li><li><a href="https://github.com/menhguin/lm-evaluation-harness/blob/main/lm_eval/models/goodfire.py">lm-evaluation-harness/lm_eval/models/goodfire.py at main · menhguin/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- menhguin/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/2618#issuecomment-2583172531">在 huggingface 中找不到数据集 lighteval/MATH-Hard · Issue #2618 · EleutherAI/lm-evaluation-harness</a>：当我运行代码以获取 open llm leaderboard v2 中的任务结果时，在 huggingface 中找不到 lighteval/MATH-Hard，我该如何解决这个问题？谢谢！
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1327723143208505355)** (5 条消息): 

> `Slurm CPU 内存问题，预训练资源建议` 


- **Slurm 报告 CPU 内存 OOM**：成员们讨论了 Slurm 指出的与 CPU 内存（而非 GPU 内存）相关的 **Out of Memory (OOM)** 问题，并建议检查 CPU RAM 的可用性。
   - 建议包括使用 `--mem=0` 标志来请求所有可用内存，或使用 `--exclusive` 标志独占节点以访问所有资源，相关内容参考 [Slurm sbatch options](https://slurm.schedmd.com/sbatch.html)。
- **使用 Slurm 标志成功解决**：一位成员确认，在再次尝试 **6.7B config** 时，使用建议的标志解决了他们的问题，这表明了正确分配资源的重要性。
   - 这一经验强调了在 Slurm 中正确设置资源标志以防止内存相关错误的有效性。
- **请求预训练资源估算**：一位成员寻求关于每个 GPU 在预训练期间所需的 **CPU RAM 和 CPU Cores** 数量的建议，并询问是否有估算这些需求的方法。
   - 这表明人们对优化资源分配以有效训练大型模型的兴趣日益浓厚。


  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1327371427829125243)** (141 条消息🔥🔥): 

> `Windsurf 登录问题，Codeium 定价与订阅，功能请求与反馈，技术错误与故障排除，用户体验与支持关注点` 


- **Windsurf 用户面临反复出现的登录问题**：包括 @coyote775799 和 @junyuh 在内的多位用户报告了登录 Windsurf 应用程序的困难，通常导致手动登录失败和功能无响应。
   - 建议的解决方案包括使用 AppCleaner 进行彻底卸载并删除残留文件，这似乎解决了许多此类连接问题。
- **关于 Codeium Pro 订阅使用的澄清**：@osplus6235 对无法在多个设备上访问其 Pro 订阅表示沮丧，尽管使用的是同一个账号登录。
   - 在支持团队十天未回复后，他们寻求版主的介入以解决此事。
- **用户强调 Codeium 服务的问题**：像 @johnreel_ 这样的用户报告称，由于 Windsurf 的问题导致额度消耗过快，扰乱了工作流程，表达了对定价模型的不满。
   - 因此，一些人正在考虑探索 Codeium 的替代方案，以避免持续的挫败感。
- **Windsurf 增强功能请求**：像 @shivamkumar 这样的用户询问了 Codeium 支持自定义模型的可能性，特别是提到希望兼容 Gemini Flash。
   - 回复指出目前仅支持现有模型，不过未来的更新可能会带来新功能。
- **技术错误引发诊断请求**：有用户提出了关于 Codeium “服务器异常关闭连接”（Abnormal connection close by server）的问题，促使用户收集诊断信息进行提交。
   - 这突显了持续存在的连接问题，这些问题影响了多位用户在该平台上的体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://freemacsoft.net/appcleaner/">AppCleaner</a>: 未找到描述</li><li><a href="https://bit.ly/8HWebAI3">我们的网络研讨会“AI Agents 与 Web3 的交汇”的 8 个亮点</a>: 12 月 17 日，我们举办了名为“AI Agents 与 Web3 的交汇”的网络研讨会，重点讨论了 AI Agents 如何利用 Web3 基础设施来……</li><li><a href="https://codeium.com/contact">联系方式 | Windsurf 编辑器和 Codeium 扩展</a>: 联系 Codeium 团队以获取支持并了解更多关于我们企业级服务的信息。</li><li><a href="https://codeium.com/faq">常见问题解答 | Windsurf 编辑器和 Codeium 扩展</a>: 查找常见问题的答案。</li><li><a href="https://codeium.com/terms-of-service-individual">服务条款：个人与 Pro | Windsurf 编辑器和 Codeium 扩展</a>: Codeium 是开发者喜爱、企业信赖的 AI 代码辅助平台。也是首个 Agentic IDE Windsurf 的构建者。</li><li><a href="https://codeium.com/blog/copilot-trains-on-gpl-codeium-does-not">GitHub Copilot 输出 GPL 代码。Codeium 则不然。</a>: 证明 GitHub Copilot 在非许可协议的代码上进行训练，且无法正确过滤建议，而 Codeium 不会让用户面临法律风险。
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1327378707714801724)** (592 条消息🔥🔥🔥): 

> `Windsurf 功能问题，Cascade 性能，订阅模式担忧，AI 工具用户体验，Windsurf 功能请求` 


- **Windsurf 遇到功能问题**：用户报告遇到了持续性的错误，例如 Cascade 错误地编辑文件以及在操作过程中遇到“internal errors”。
   - 有投诉称 Windsurf 生成了错误的输出且无法识别某些文件，导致用户感到沮丧。
- **Cascade 的性能引发担忧**：一些用户对 Cascade 最近的性能表示不满，称尽管设置了严格的规则，它仍持续出现幻觉并产生错误。
   - 用户指出，虽然 Cascade 以前很可靠，但现在变得难以预测且更容易出错，严重影响了他们的工作流。
- **对订阅模式的担忧**：用户讨论了对订阅定价结构的不满，认为它没有反映他们的实际使用模式，特别是关于 flow credits 的部分。
   - 许多人主张采用更灵活的模式，建议提供替代方案以防止因随意的查询而浪费额度。
- **对 Cursor 等 AI 工具的评价褒贬不一**：几位用户将 Windsurf 与 Cursor 进行了比较，指出 Cursor 的 autocomplete 功能在某些方面更胜一筹，但仍对其可靠性表示担忧。
   - 用户表示，虽然 Cursor 在某些条件下表现更好，但他们发现 Windsurf 在特定任务中更有益，凸显了用户偏好的多样性。
- **功能请求与增强**：用户呼吁改进一些功能，例如设置自定义 AI 规则的能力、额度的优化使用以及更强大的上下文管理。
   - 用户表示有兴趣增强外部 API 的集成，并允许更好的规则定义以提升开发体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://addyo.substack.com/p/the-70-problem-hard-truths-about">The 70% problem: Hard truths about AI-assisted coding</a>：一份实地指南以及为什么我们需要重新审视我们的预期</li><li><a href="https://tenor.com/view/frustrated-angry-upset-bullshit-table-bang-gif-8128901">Frustrated Angry GIF - Frustrated Angry Upset - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://docs.codeium.com/best-practices/prompt-engineering">Prompt Engineering - Codeium Docs</a>：未找到描述</li><li><a href="https://pastebin.com/Lk422FgE">Critical Documentation and WorkflowDocumentation ManagementMaintain a &#039;winds - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://tenor.com/view/wink-eye-turn-around-chewing-gif-23703707">Wink Eye GIF - Wink Eye Turn - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/indian-phahaha-gif-27058287">Indian Phahaha GIF - Indian Phahaha - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>：需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://codeium.com/">Windsurf Editor and Codeium extensions</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://codeium.canny.io/feature-requests">Feature Requests | Codeium</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://youtu.be/-qa7_oe5uWQ"> - YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1327369983638114407)** (553 条消息🔥🔥🔥): 

> `Cursor IDE 性能，新的 AI 和工具，协作项目，AI 规则和指南，Cursor 扩展问题`

- **对 Cursor IDE 的评价褒贬不一**：用户分享了对 Cursor IDE 性能的不同体验，一些人注意到 AI 操作有所改进，而另一些人则报告了在编码过程中持续出现的问题，如运行缓慢和报错。
   - 例如，一位用户提到反复遇到 Claude 的建议与当前代码不匹配的问题，导致了大量的调试工作。
- **新的 AI 工具和模型**：几位用户讨论了 AI 领域新模型的出现，例如具有 256k 上下文长度的 Mistral Codestral，以及 Claude 持续增强代码生成能力的更新。
   - 人们对这些进展如何帮助开发者表现出浓厚兴趣，尤其是当它们与 Cursor 等工具集成时。
- **项目协作**：参与者对项目协作表达了热情，一些人建议这有助于提高技能，并促进初级和高级开发者之间的学习。
   - 想法涵盖了从创建 Test Manager AI Agent 到展示 Cursor 能力的各种应用。
- **AI 规则与优化**：用户讨论了通过自定义 AI 规则来提高 Cursor 的输出质量和响应准确性，强调了详细 Prompting 的重要性。
   - 一位用户分享了一套广泛的指南，旨在增强 Claude 的推理和分析过程。
- **扩展与设置问题**：一些用户报告了 Cursor 扩展和设置方面的挑战，例如安装错误以及 IDE 表现不如预期的情况。
   - 建议包括使用 Checkpoints 恢复之前的状态，以及排查性能缓慢或功能无响应等常见问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://youactuallysuck.com/message/161A8yb1aw2t">You Actually Suck - 匿名邮件反馈</a>: 未找到描述</li><li><a href="https://youactuallysuck.com/">You Actually Suck - 匿名邮件反馈</a>: 未找到描述</li><li><a href="https://autoblogpilot.com/">AutoBlogPilot</a>: 未找到描述</li><li><a href="https://21st.dev/serafimcloud/splite">Spline Scene | 21st.dev - 面向设计工程师的 NPM</a>: 一个集成 Spline 3D 场景的 React 组件。演示组件结合了交互式 3D 可视化、聚光灯效果和响应式文本内容。特性：• 延迟加载的 Spline 集成 •...</li><li><a href="https://www.latent.space/p/o1-skill-issue">o1 不是一个聊天模型（而这正是重点）</a>: Ben Hylak 如何通过克服他的技能问题，从 o1 的专业怀疑论者转变为粉丝。</li><li><a href="https://mistral.ai/news/codestral-2501/">Codestral 25.01</a>: 以 Tab 键的速度编写代码。今日已在 Continue.dev 上线，并即将登陆其他领先的 AI 代码助手。</li><li><a href="https://21st.dev/s/background">Backgrounds Components | 21st.dev - 面向设计工程师的 NPM</a>: 受 shadcn/ui 启发，开箱即用的 React Tailwind 背景组件。</li><li><a href="https://x.com/ryandavogel/status/1878240606289338759?s=46">Ryan Vogel (@ryandavogel) 的推文</a>: 我们距离 AGI 还有一年时间，而 OpenAI 正在以 38.5 万美元的底薪招聘人员来编写 React</li><li><a href="https://x.com/kregenrek/status/1878487131099898269?s=46">Kevin Kern (@kregenrek) 的推文</a>: 为开发者介绍 Codefetch。通过一个简单的终端命令将代码转换为适用于 LLM 的 Markdown。可在 bolt .new、Cursor 和许多其他 AI 编程工具中使用。→ 与你的代码库聊天 → 节省 Token → ...</li><li><a href="https://forum.cursor.com/t/which-model-is-the-best-for-claude-sonnet/35148">哪个模型最适合 Claude Sonnet？</a>: 大家好。我刚刚升级到了 Pro 计划。我看到我使用的模型是 claude-sonnet-20241022。但我的问题是，Claude Sonnet 3.5 和 sonnet-2024102 之间哪个是最新的...</li><li><a href="https://forum.cursor.com/t/error-unauthorized-request/39861/28">错误：未授权请求</a>: 各位，抱歉延迟回复。不幸的是，在过去的几天里，我们看到了大量来自临时电子邮件地址的滥用行为，已经达到了...</li><li><a href="https://github.com/lvllvlTlvl/cursor-conversation-manager">GitHub - lvllvlTlvl/cursor-conversation-manager: 一个用于管理和重构 Cursor IDE 对话历史记录的 Python 库。文件按上下文分目录存放。</a>: 一个用于管理和重构 Cursor IDE 对话历史记录的 Python 库。文件按上下文分目录存放。 - lvllvlTlvl/cursor-conversation-manager</li><li><a href="https://www.youtube.com/watch?v=TuO21CLrluU"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/sksarvesh007">sksarvesh007 - 概览</a>: sksarvesh007 有 62 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://www.nico.fyi/blog/tailwind-css-group-modifier-to-prevent-react-rerender?ref=dailydev">如何在 React 中使用 Tailwind CSS 防止重新渲染</a>: 学习如何使用 Tailwind CSS 的 group 修饰符和 data 属性来创建动态 UI 元素，而无需 React 重新渲染。提高性能并简化代码。</li><li><a href="https://www.youtube.com/watch?v=nxss50uZgE0"> - YouTube</a>: 未找到描述</li><li><a href="https://elthos.com">未找到标题</a>: 未找到描述</li><li><a href="https://forum.cursor.com/t/where-is-the-data-generated-by-codebase-indexing-stored-locally/22517">Codebase 索引生成的数据存储在本地哪里？</a>: Codebase 索引生成的数据存储在本地哪里？如果我通过 SSH 连接到远程服务器，数据是存储在远程服务器上吗？还是存储在本地？我想知道路径...</li><li><a href="https://daily.dev/blog/cursor-ai-everything-you-should-know-about-the-new-ai-code-editor-in-one-place">Cursor AI：改变游戏规则的 AI 驱动代码编辑器</a>: 探索 Cursor AI 如何通过先进的 AI 功能改变编程，为各级开发者提高生产力和代码质量。</li><li><a href="https://blog.cloudflare.com/sqlite-in-durable-objects/">每个 Durable Object 中的零延迟 SQLite 存储</a>: 传统的云存储本质上很慢，因为它通过网络访问并且必须同步许多客户端。但如果我们能将应用程序代码深入到存储层中呢...</li><li><a href="https://forum.cursor.com/t/concerns-about-privacy-mode-and-data-storage/5418/15">关于隐私模式和数据存储的担忧</a>: 从你的代码计算出的 Embedding 存储在 Cursor 服务器上，并包含大量关于你代码的信息位。他们目前似乎没有解决这些担忧...</li>

ink these embeddings a...</li><li><a href="https://www.cursor.com/privacy">Privacy Policy | Cursor - AI 代码编辑器</a>: 如果您有任何问题或反馈，请发送邮件至 hi@cursor.com。</li><li><a href="https://stackoverflow.com/questions/9455774/is-it-a-bad-idea-to-store-sqlite-cursor-in-android">在 Android 中存储 SQLite Cursor 是个坏主意吗？</a>: 我正尝试在 Android 上实现一个词典应用。当用户在 EditText 中输入字母（或删除字母）时，应用会查询数据库并显示所有以...开头的条目</li><li><a href="https://www.youtube.com/watch?v=KenChh4p0nI">10 分钟内使用 AI 构建带有存储功能的应用 (Cursor AI, Claude AI, Firebase Storage)</a>: 让我们学习如何在你的应用中设置 Firebase Storage，以允许用户上传个人资料图片。订阅以获取更多内容！ 👉 http://bit.ly/3zlUmiS 👈让我们...</li><li><a href="https://forum.cursor.com/t/chat-history-folder/7653">Chat history 文件夹</a>: 大家好，聊天记录存储在哪里？我非常喜欢 Cursor，但我刚刚更换了电脑，虽然在同一个项目、同一个文件夹下工作，但聊天记录不见了。谢谢！</li><li><a href="https://www.reddit.com/r/androiddev/comments/bwxh4r/sqlite_cursor_with_a_large_amount_of_data/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://forum.cursor.com/t/concerns-about-privacy-mode-and-data-storage/5418">关于 Privacy Mode 和数据存储的担忧</a>: 大家好，我目前正在使用 Cursor.sh 协助处理一个客户项目。我的客户对潜在的 NDA 违规和数据保护提出了担忧，尽管我一直...</li><li><a href="https://forum.cursor.com/t/how-do-i-export-chat-with-ai/144">如何导出与 AI 的对话？</a>: 我该如何导出或分享与 AI 的对话？我希望能像 ChatGPT 那样分享我与 AI 的对话，以便与同事共享。或者至少能够导出它。目前我必须...</li><li><a href="https://forum.cursor.com/t/concerns-about-privacy-mode-and-data-storage/5418/6">关于 Privacy Mode 和数据存储的担忧</a>: 刚刚在以下地址添加了最近的 Privacy Policy tl;dr 版本：https://www.cursor.com/privacy TLDR 如果你在 Cursor 的设置中启用 “Privacy Mode”，你的任何代码都不会被我们存储 ...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1327368022532554995)** (317 messages🔥🔥): 

> `LM Studio 功能, 模型性能比较, AI 硬件讨论, 量化对模型的影响, 编码模型用户体验` 


- **有效地测试本地 AI 模型**：用户分享了测试各种本地托管编码模型的见解，强调了模型量化对性能的重要性，并指出某些量化会导致优秀的模型变得无效。
   - 一位用户创建了一份全面的 QA 文档，根据其测试经验提供了有效运行编码模型的建议。
- **硬件讨论与建议**：参与者讨论了使用高规格硬件（如 M4 Ultra）与专用 AI 设备的对比可行性，权衡了 GPU 性能与 AI 任务的成本效益。
   - 一些用户表示，虽然专用硬件可能表现更佳，但高规格 Apple 设备的简单性和多功能性可以为各种任务提供合适的替代方案。
- **关于大模型的见解**：有注意到像 72B 变体这样的大型模型可能会超出单张 GPU 的可用 VRAM，但可以在多个设备上进行有效管理。
   - 讨论强调了运行此类模型的挑战，同时建议了特定的量化策略以获得最佳性能。
- **量化与模型效率**：用户反馈指出选择合适量化级别对模型效率至关重要，建议优先选择 Q4_K 和 Q5_K 以获得更好的性能。
   - 参与者分享了关于基于所用量化方法的模型性能差异的轶事。
- **社区参与和功能请求**：用户对 LM Studio 中的 TTS（文本转语音）选项等功能表示感兴趣，并提出了增强各种编码模型可用性的建议。
   - 一些社区成员分享了与 LM Studio 兼容的有用资源和模型链接，促进了协作和知识共享。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://blog.exolabs.net/day-2/">12 Days of EXO</a>: 12 天真正的开源创新</li><li><a href="https://markdownpastebin.com/?id=9912b825d602429d87c11a80e8d8f543">MarkdownPastebin</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/Sky-T1-32B-Preview-GGUF">bartowski/Sky-T1-32B-Preview-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Voodoo2">Voodoo2 - Wikipedia</a>: 未找到描述</li><li><a href="https://youtu.be/P8qlE5XBopw?si=Ew359LbJzBS90nsd"> - YouTube</a>: 未找到描述</li><li><a href="https://gist.github.com/shermanhuman/2b9a82df1bab242a8edffe504bb1867c">Coding local LLM recommendations that meet some minimum useful standard</a>: 满足最低实用标准的本地 LLM 编程推荐 - MiniumStandardsLLM.md</li><li><a href="https://youtu.be/XYBI_Ow7F_4?t=817">Please Don&#39;t Download HackerGPT...</a>: 大家好，我是 Mutahar！这次我们来看看社交媒体上出现的一组关于...的疯狂广告。</li><li><a href="https://youtu.be/JWfNLF_g_V0?si=avXvc4VzdJ2LZbdM">Turn ANY Website into LLM Knowledge in SECONDS</a>: 我们面临的 LLM 最大挑战之一是它们的知识过于通用，对于新事物的了解有限。这就是为什么 RAG 在...时成为如此热门的话题。</li><li><a href="https://lmstudio.ai/download">Download LM Studio - Mac, Linux, Windows</a>: 发现、下载并运行本地 LLM</li><li><a href="https://www.humblebundle.com/books/machine-learning-engineer-masterclass-packt-books">Humble Tech Book Bundle: Machine Learning Engineer Masterclass by Packt</a>: 通过这套精彩的编程课程学习机器学习的基础和高级技术。按需付费并支持慈善事业！</li><li><a href="https://youtu.be/VkzO2w6EqK4?si=dONXQA4qc6VCdUvk">The 3Dfx Voodoo Difference: This is why we love them</a>: 在这段视频中，我们将了解为什么 3DFX Voodoo 是一款如此特别的显卡！💙 考虑支持我 💙Patreon：获取独家早期访问权限、幕后...</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/300">0.3.6 Developer Log Truncation regardless of settings · Issue #300 · lmstudio-ai/lmstudio-bug-tracker</a>: 从 0.3.5 更新到 0.3.6 Build 8 后，我无法获得完整的、未截断的开发日志。无论我使用什么设置，例如 Verbose Logging 以及 Log Prompts and Responses。我曾...</li><li><a href="https://lmstudio.ai/model/phi-3.1-mini-128k-instruct">Phi 3.1 Mini 128k</a>: phi3 • Microsoft • 3.8B</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/294#issuecomment-2581576638">Not able to install LM Runtime after upgrading to version LM Studio 0.3.6 (Build8) under MacOS 15.2 (M4 Apple Silicon) · Issue #294 · lmstudio-ai/lmstudio-bug-tracker</a>: 升级到 0.3.6 版本（从 0.3.5）后，无法安装任何 LM Runtime。我可以下载 Metal llama.cpp v1.7.1 以及 LM Studio MLX v0.1.3，但它们没有被安装，并且...</li><li><a href="https://github.com/microsoft/ML-For-Beginners/tree/main">GitHub - microsoft/ML-For-Beginners: 12 weeks, 26 lessons, 52 quizzes, classic Machine Learning for all</a>: 12 周，26 节课，52 个测验，适合所有人的经典 Machine Learning - microsoft/ML-For-Beginners</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.6">LM Studio 0.3.6</a>: Tool Calling API 测试版，新的安装程序/更新系统，以及对 `Qwen2VL` 和 `QVQ`（支持 GGUF 和 MLX）的支持</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues">Issues · lmstudio-ai/lmstudio-bug-tracker</a>: LM Studio 桌面应用程序的错误跟踪 - Issues · lmstudio-ai/lmstudio-bug-tracker</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/285">(Exit code 133) Error when loading large LLM models · Issue #285 · lmstudio-ai/lmstudio-bug-tracker</a>: 当加载大型 LLM（例如，具有 32768 上下文窗口的 Meta-Llama-3.1-70B-Instruct-IQ2_S）时，我会遇到错误 (Exit code: 133)。请检查设置并尝试再次加载模型...</li><li><a href="https://lmstudio.ai/docs/basics/chat">Manage chats - Running LLMs Locally | LM Studio Docs</a>: 管理与 LLM 的对话线程</li><li><a href="https://lmstudio.ai/docs/configuration/presets">Config Presets - Configuration | LM Studio Docs</a>: 将您的系统提示词和其他参数保存为预设，以便在不同聊天中轻松重复使用。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1327462261576695918)** (186 条消息🔥🔥): 

> `PowerMac G3 组装, Llama 模型加载问题, RTX 显卡, NVIDIA DIGITS 发布, 双 GPU 配置考量` 


- **PowerMac G3 用于 AI 模型**：一位成员展示了他们定制组装的 **PowerMac G3**，用于在 LM Studio 上运行模型，并强调了其独特的设计。
   - 随后展开了关于技术规格以及与现代硬件对比的讨论。
- **加载 Llama 模型的挑战**：几位成员讨论了在 MacBook Pro 上尝试加载 **Llama 3.3 70b Instruct Q3_K_L 模型**时遇到的困难，指出尽管 RAM 充足，但仍存在系统资源不足的问题。
   - 建议包括调整 GPU 显存分配和评估系统设置。
- **关于 RTX 显卡处理 AI 工作负载的见解**：针对 **RTX 4090 vs A6000** 显卡展开了对话，重点关注它们在大型 AI 模型上的性能和价值。
   - 用户在讨论潜在的未来升级时，对性价比发表了看法。
- **对 NVIDIA DIGITS 的期待**：用户对 NVIDIA **DIGITS** 的发布进行了辩论，分析了其在 AI 任务方面相对于 Apple 设备等现有选择的潜力。
   - 关于是及早采用还是等待进一步发展，意见不一。
- **选择双 GPU 配置方案**：一位成员寻求购买**第二块 4060** 以组建双显卡配置的建议，并对比了 PNY 和 MSI 的选项。
   - 他们还讨论了关于 PSU 限制以及可能即将推出的 **5060** 型号的考量。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://vladmandic.github.io/sd-extension-system-info/pages/benchmark.html">SD WebUI Benchmark Data</a>: 未找到描述</li><li><a href="https://www.nvidia.com/en-us/project-digits/">NVIDIA Project DIGITS: The World’s Smallest AI Supercomputer. </a>: 立即预订。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1327387005318987776)** (390 条消息🔥🔥): 

> `Claude 模型讨论, 超参数搜索即服务, Twitter 体验与功能, 模型与量化, GitHub 宕机担忧` 


- **Claude 模型反应**：社区讨论了最新的 Claude 模型更新，注意到其语言模式发生了变化，例如增加了“direct”和“helpful”等词汇的使用。一些人对关于新 Claude 模型的虚假公告表示幽默，反映出对其性能和风格的复杂情绪。
   - 该模型的回答被描述为更加武断，导致一些用户认为它显得有些“愤怒”。
- **对超参数优化服务的兴趣**：一位用户询问了提供超参数搜索即服务的相关服务，暗示了有效调优模型的复杂性。人们对使用贝叶斯优化（Bayesian optimization）等技术自动优化模型参数的解决方案表现出越来越浓厚的兴趣。
   - 讨论还涉及了模型训练过程中调试问题的挑战，这些问题可能通过单元测试（unit testing）得到缓解。
- **Twitter 用户体验的挑战**：用户对 Twitter 目前的时间线体验表示不满，称最近的更改导致信息流的个性化程度降低。移除“simclusters”等功能被认为损害了用户体验，尤其是对于那些喜欢主题内容的用户。
   - 在平台持续变化的过程中，人们对处理垃圾信息和无关内容的担忧也有所增加。
- **模型可用性与量化**：参与者讨论了大模型的可用性，特别提到 SambaNova 和 Together AI 是 405B 参数模型的主要提供商。量化方法的重要性以及模型在边缘设备（edge devices）上的部署被强调为重要议题。
   - 用户呼吁推出更多像 Helium-1 这样专为轻量级应用设计的模型，展示了对实用 AI 部署的兴趣。
- **对 GitHub 宕机的担忧**：一位用户报告了 GitHub 宕机的问题，影响了他们向项目推送更新的能力。这次宕机被一些人视为从编码任务中休息的信号，引发了关于此类中断影响的讨论。
   - 其他人幽默地看待这一情况，认为这是在开发过程中进行反思和未经测试的即兴发挥的好机会。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://novasky-ai.github.io/posts/sky-t1/">Sky-T1: Train your own O1 preview model within $450</a>：介绍 Sky-T1-32B-Preview，我们的推理模型在流行的推理和代码基准测试中与 o1-preview 表现相当。</li><li><a href="https://githubnext.com/projects/copilot-workspace">GitHub Next | Copilot Workspace</a>：GitHub Next 项目：一个为日常任务设计的 Copilot 原生开发环境。</li><li><a href="https://x.com/jacquesthibs/status/1878851967981887736?s=46">Jacques (@JacquesThibs) 的推文</a>：新的 Claude 模型刚刚发布</li><li><a href="https://kyutai.org/2025/01/13/helium.html">宣布 Helium-1 预览版</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF">Qwen/Qwen2.5-72B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF">bartowski/Qwen2.5-72B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset">HumanLLMs/Human-Like-DPO-Dataset · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/blog/not-lain/tensor-dims">Mastering Tensor Dimensions in Transformers</a>：未找到描述
</li>
</ul>

</div>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1327618109749727283)** (15 条消息🔥): 

> `用于医疗建议的 LLM，AI 的隐私担忧，有声书文本提取` 


- **咨询用于医疗建议的 LLM**：一位用户询问推荐适合医疗建议的 LLM，但另一位用户警告不要使用任何 AI 获取可靠的医疗信息，建议咨询人类医生。
   - 尽管有此询问，建议很明确：AI 可以提供见解，但应始终由真正的医生进行复核，以避免潜在的误导信息。
- **使用 AI 进行医疗查询的隐私问题**：人们对使用 AI 获取医疗建议时的隐私表示担忧，重点强调了个人信息可能被审核员访问的风险。
   - 一位成员指出，用户必须信任这些 AI 工具背后的公司，因为它们的隐私实践可能存在显著差异。
- **寻求可靠的 PDF 文本提取工具以制作有声书**：一位成员询问了用于从 PDF 中提取文本以创建有声书的可靠工具，特别提到了去除页眉和脚注的问题。
   - 提到了使用 **Gemini Flash** 作为此用途的一个具有成本效益的选项。
- **用户参与进度跟踪**：一位成员注意到在特定背景下缺乏进展，引发了对用于聚合的工具或网站的询问。
   - 这表明社区内对跟踪或协作方法可能存在兴趣。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

teknium: https://x.com/prajdabre1/status/1877720543933370418?s=46
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1327721839283142716)** (59 messages🔥🔥): 

> `Qwen 0.5B 模型性能, 生成式知识蒸馏 (GKD), AI 中的合成数据使用, MobileLLM 研究见解, 注意力机制的改进` 


- **Qwen 0.5B 表现参差不齐**：Qwen 0.5B 模型在数学任务中表现出一定的熟练度，但在通用语境下的连贯响应方面表现不佳，经常生成无意义的内容。
   - 用户对其能力表示担忧，指出它在处理数学问题时经常失败，并可能在计算过程中进入死循环。
- **模型训练中关于 GKD 的困惑**：用户对模型卡片中使用的生成式知识蒸馏 (GKD) 术语感到困惑，不确定它与传统蒸馏技术有何不同。
   - 一些人推测 GKD 可能指的是利用另一个模型的合成数据进行训练，而不是从原始模型中蒸馏 Logits。
- **Hugging Face 讨论合成数据**：Loubna Ben Allal 的一次演讲强调了合成数据在训练 Smol Language Models 中的重要性，并以 SmolLM 模型的设计为例进行了说明。
   - 引用的 YouTube 资源和讨论强调了理解合成数据如何提升模型性能的重要性。
- **MobileLLM 论文揭示新见解**：MobileLLM 论文指出，蒸馏方法被发现不如基于标签的训练有效，这引发了对当前模型训练实践的质疑。
   - 这一引用强调了关于训练 AI 小模型有效方法的持续争论。
- **注意力机制的新方法**：最近的研究探索了注意力机制的进展，旨在保持性能的同时降低训练和推理过程中的复杂度。
   - 提出的一种新型逐元素注意力机制 (element-wise attention mechanism) 建议采用另一种计算相似度的方法，可能会带来效率提升。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.14905">MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases</a>：本文针对云端成本增加和延迟担忧，探讨了在移动设备上对高效大语言模型 (LLMs) 日益增长的需求。我们专注于设计高质量的十亿参数以下 LLMs...</li><li><a href="https://arxiv.org/abs/2501.05730">Element-wise Attention Is All You Need</a>：自注意力 (SA) 机制在各个领域都表现出了卓越的性能，但在训练和推理过程中都面临着巨大的复杂性。下一代架构...</li><li><a href="https://x.com/novaskyai/status/1877793041957933347?s=46">来自 NovaSky (@NovaSkyAI) 的推文</a>：1/6 🚀 介绍 Sky-T1-32B-Preview，这是我们的全开源推理模型，在流行的推理和编程基准测试中媲美 o1-preview —— 训练成本低于 $450！📊博客：https://novasky-ai.github....</li><li><a href="https://huggingface.co/kz919/QwQ-0.5B-Distilled">kz919/QwQ-0.5B-Distilled · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/kz919/QwQ-0.5B-Distilled-SFT">kz919/QwQ-0.5B-Distilled-SFT · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=AjmdDy7Rzx0">Best of 2024: Synthetic Data / Smol Models, Loubna Ben Allal, HuggingFace [LS Live! @ NeurIPS 2024]</a>：https://latent.space/2024-syndata-smolmodels 在 Huggingface 从事合成数据和 Smol Language Models 研究的 Loubna Ben Allal 分享了相关知识...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/prajdabre1/status/1877720543933370418?s=46
  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/)** (1 messages): 

katetra: https://x.com/stackblitz/status/1878818461905739994
  

---

### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1327719064189276262)** (27 条消息🔥): 

> `Stripe 集成, Prompting 技巧, 构建 AI 应用, Bolt 用户体验, 网络研讨会公告` 


- **Stripe 集成即将推出**：一位成员分享说 **Stripe 集成** 即将到来，现有用户已经成功使用，并建议搜索相关的 **YouTube 教程**。
   - 另一位用户对该集成表示兴奋，称其为他们项目的 **巨大加分项 (HUGE plus)**。
- **掌握 Prompting 以避免代码丢失**：几位用户反映，每当添加新功能时，系统都会移除现有组件，这让他们感到沮丧。一位成员幽默地表示：*“我一直在努力让我的产品突破某个瓶颈。”*
   - 他们讨论了诸如在设置中启用 **diffs** 等解决方案，以帮助在添加新功能时保留现有代码。
- **关于 AI LLM 应用的网络研讨会**：一位成员宣布了一个关于使用 Bolt 构建具有结构化输出的 AI 应用程序的 **免费直播培训研讨会**，定于 **周二上午 10 点 (EST)** 举行。
   - 参与者可以学习创建动态应用的宝贵步骤，强调了使用 AI 编程平台的潜力。
- **用户对工作流效率的沮丧**：用户对代码丢失的重复循环表示不满，其中一人表示：*“我来这里是想看看是否有人能突破这个瓶颈。”*
   - 大家一致认为需要一些方法来防止不必要的更改，特别是在追求高效工作流时。
- **寻求支持与报告问题**：成员们提供了关于如何在编辑器中撤销按钮旁边直接报告问题的技巧，强调了用户的 **反馈机制 (feedback mechanism)**。
   - 他们互相鼓励分享经验，并讨论最近的 YouTube 资源是否准确反映了潜在的解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reinventing.ai/next-level-ai-apps-no-code">如何通过 No Code 构建下一代 AI 应用</a>：未找到描述</li><li><a href="https://tangerine-kleicha-76b338.netlify.app/">Vite + React + TS</a>：未找到描述</li><li><a href="https://docs.google.com/document/d/1SwlpZH1SotqPg2KbZqzWPdpBbs6aKIqMDspSCBCD1iQ/edit?usp=sharing">Bolt Prompting 终极指南</a>：Bolt.new Prompting 终极指南。我正在与所有感兴趣的人分享这份指南，因为我认为它很有帮助。我询问了 Bolt AI Agent 本身应该如何与“它”交流，以下是...
</li>
</ul>

</div>
  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1327367991104507914)** (386 条消息🔥🔥): 

> `Bolt 上的 Token 管理、集成 Supabase 和 Netlify、AI Prompt 使用问题、API 请求中的 CORS 处理、在 Bolt 中使用 Stripe` 


- **对 Token 消耗的挫败感**：许多用户报告在尝试实现功能时 Token 消耗过多，其中一位用户提到一个简单的叠加层（overlay）单次 Prompt 就消耗了 **150 万个 Token**。
   - 其他人也分享了类似的经历，强调需要更高效的 Prompt 管理和更具体的请求，以减少 Token 浪费。
- **API 集成挑战**：用户表示在集成各种服务时遇到困难，涉及 CORS 问题以及在修复过程中跨文件错误包含代码的问题。
   - 一位用户指出了 Stripe 集成的错误，而另一位用户提到由于 Stripe 的复杂性，他们更多地依赖 PayPal 按钮。
- **导出和重用代码**：用户讨论了将项目导出为 zip 文件，并寻找通过 StackBlitz 将其重新上传到 Bolt 以继续开发的方法。
   - 这一过程凸显了在管理 Bolt 开发的大型代码库时，有效使用 StackBlitz 的重要性。
- **用户对 Bolt 功能的反馈**：分享了对 Bolt 各种功能的反馈，包括对更好处理环境变量和 API keys 等功能的需求。
   - 讨论还包括由于 Bolt 的流行及其遇到的特定问题，考虑将其讨论拆分为独立类别的可能性。
- **对订阅和 Token 定价的担忧**：用户提出了关于 Token 定价的问题，一些人觉得充值 Token 的成本比订阅计划更高。
   - 总的来说，大家对任何潜在的优惠码或管理 Bolt 使用成本的替代方案都很感兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://boltsync.mystify.tech/">BoltSync - 使用 Bolt 进行 GitHub 仓库管理</a>: 使用 Bolt Prompt 修改您的 GitHub 仓库，并通过 BoltSync 将更改同步回 GitHub。通过 AI 驱动的仓库管理简化您的开发工作流。</li><li><a href="https://support.bolt.new/Prompting-Effectively-How-to-talk-to-Bolt-13fd971055d6801b9af4e965b9ed26e2">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>: 一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://stayyoung.app">Stay Young - 您的每日健康剂量</a>: 未找到描述</li><li><a href="https://coolify.io/)">Coolify</a>: 一个开源且可自托管的 Heroku / Netlify / Vercel 替代方案。</li><li><a href="https://docs.google.com/document/d/1SwlpZH1SotqPg2KbZqzWPdpBbs6aKIqMDspSCBCD1iQ/edit?usp=sharing">Bolt.new Prompt 终极指南</a>: Bolt.new Prompt 终极指南。我正在与所有感兴趣的人分享这份指南，因为我认为它很有帮助。我询问了 Bolt AI Agent 本人应该如何与“它”交流，以下是……</li><li><a href="https://youtu.be/ayagXgAShSk">Bolt.new - 在节省 Token 的同时修复常见错误</a>: 在这段视频中，我分享了一个在不消耗任何 Token 的情况下解决常见编码错误的简单方法。我演示了如何截取错误截图并上传……
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1327372414580948993)** (264 条消息🔥🔥): 

> `英国的 AI 生产力, 用于客户服务的嵌入式 AI Agent, AI 模型对比, 新 AI 模型发布, AI 编程的未来` 


- **英国政府的 AI 投资**：英国政府计划向 AI 技术投资 140 亿英镑，目标是在三年内使生产力翻倍，这引发了关于此类计划有效性的辩论。
   - 批评者认为这些资金可以得到更好的分配，并建议 AI 不应取代人类的生产力。
- **为客户构建自定义 AI Agent**：一位用户正在寻求创建嵌入式 AI 客户服务 Agent 的建议，这些 Agent 需要能与 WhatsApp 和 Slack 等流行 API 集成。
   - 其他人建议查看 n8n 或 flowise 上的教程以了解集成和易用性，同时对各种供应商和成本提出了警告。
- **AI 模型的性能对比**：在一次涉及 Minecraft 的挑战中，据报道 Claude 和 Gemini 在各项任务中表现优于 ChatGPT，引发了关于不同 AI 模型能力的讨论。
   - 用户对性能差距表示担忧，特别是如果 GPT 在竞争场景中继续落后的话。
- **新 AI 模型发布**：Mistral API 发布了一个名为 'codestral' 的新模型，提供 256k 的上下文容量并承诺了高性能。
   - 在 Canvas 功能集成后，关于该模型与 GPT-4 等现有模型之间的差异仍存在疑问。
- **AI 对编程的影响**：一位用户思考了 AI 改变编码和编程角色的潜力，认为随着 AI 的演进，传统的编码可能会减少。
   - 对话指向了 AI 在软件开发中日益增长的集成，这可能会简化工作流程并减少手动编码的必要性。



**提到的链接**：<a href="https://status.openai.com/">OpenAI Status</a>：未找到描述

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1327375437956907048)** (25 条消息🔥): 

> `Canvas 问题, 代码输出疑虑, 位置信息使用, 团队账户问题, 自定义 GPT 功能` 


- **Canvas 经常无法打开**：成员们讨论了在打开 Canvas 时遇到的持续困难，经常只收到代码块。有人建议特定的 Prompt 会触发 Canvas 的使用，这表明可能需要特定的变通方法。
   - 另一位用户提到，一个有效的 Canvas Prompt 是请求具有特定卧室和浴室数量的房屋平面图。
- **用户对不完整的代码响应感到沮丧**：多位成员对收到代码注释而非完整代码表示沮丧。一位用户坚持认为，尽管多次请求，该模型仍经常无法提供完整的代码。
   - 另一位用户强调了一个旨在解决此问题的 Bug 报告，并鼓励其他人投票以增加其可见性。
- **ChatGPT 意外提供用户位置**：一位用户透露，ChatGPT 在推荐 YouTube 视频时提供了他们的位置，导致了困惑和担忧。他们澄清说位置并未被直接共享，暗示可能是通过 IP 地址进行的潜在数据使用。
   - 随后引发了关于这些信息是否存储在 ChatGPT 之前对话记忆中的讨论。
- **团队账户项目可见性问题**：一位用户报告了其团队账户的问题，尽管其队友已登录，但他们看不到任何项目。他们提到联系了 OpenAI，但未收到明确的解决方案或帮助。
   - 这引发了关于工作区内项目可见性和潜在技术问题的疑问。
- **关于自定义 GPT 能力的提问**：有人询问自定义 GPT 是否利用 Memory 和自定义指令，一位用户断言它们并不利用。这引发了关于这些自定义模型在交互过程中究竟拥有多少上下文的对话。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1327366579884916877)** (48 条消息🔥): 

> `GPT 表格解析问题, OCR vs AI 表格读取, 提高表格准确性, 数据格式的横向思维, AI 模型的可靠性` 


- **GPT 在处理宽表格时表现不佳**：用户报告了 GPT 在误读表格方面的持续问题，特别是在宽格式中，它在对齐和行偏移方面表现挣扎。
   - 这一问题在各种表格中频繁出现，引发了对更可靠解决方案的呼吁。
- **关于 OCR 有效性的辩论**：一位用户提到他们主要使用 Amazon Textract 进行表格识别，它通常能正确对齐文本，但在处理复杂结构时面临挑战。
   - 另一位参与者反驳称，传统 OCR 在表格读取准确性方面仍然优于 AI 模型，这仍然是一个令人担忧的问题。
- **表格解析的不一致性**：参与者讨论了不同的表格可能以独特的方式出错，导致尽管偶尔有修复，AI 的表现仍然不可靠。
   - Vitojanko 强调了 AI 宣传的表格读取能力的重要性，并指出了其目前的不可靠性。
- **呼吁创新解决方案**：用户表示愿意寻找“技巧”或创新方法来增强 AI 在解析表格时的表现。
   - 这包括建议请求与 AI 能力更兼容的数据格式。
- **承认局限性**：大家达成共识，承认虽然 AI 在表格处理方面有所进步，但其准确率很少超过 60%。
   - 用户指出了依赖 Beta 软件的固有挑战，同时强调了提高可靠性的必要性。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1327366579884916877)** (48 条消息🔥): 

> `GPT 解析表格, 真实 OCR 性能, 复杂表格结构, 提高 AI 准确性, 数据格式的横向思维` 


- **GPT 在宽表格格式上表现不佳**：一位用户报告了 GPT 在“宽”表格格式中数据对齐错误的问题，导致对价格和其他细节的误读。
   - 这一问题在各种表格和 Prompt 中持续存在，引发了对模型处理复杂布局时可靠性的担忧。
- **真实的 OCR 比 AI 更可靠**：另一位用户建议使用真实的光学字符识别（OCR）工具，强调 AI 模型在视觉任务中的准确率较低，通常在 **60%** 左右。
   - 虽然像 Amazon Textract 这样的工具通常能正确获取文本，但在处理复杂表格时可能会遇到困难，促使一些用户考虑替代方案。
- **AI 读取表格不可靠**：讨论中提到，尽管表格读取被宣传为一个用例，但 AI 在这方面的不可靠性让用户感到沮丧。
   - 有人认为，虽然 AI 有时可以修复表格问题，但其不一致性仍然是用户关注的主要问题。
- **寻求更好的数据解析技巧**：一位用户表示愿意寻找提高表格读取性能的方法，暗示可能存在增强结果的“技巧”。
   - 随着用户继续面临表格数据的挑战，社区被鼓励探索创造性的解决方案。
- **采用更好的数据格式**：提出的一个建议是向数据提供者索要更好的格式，这有助于缓解解析问题。
   - 这种横向思维方法反映了积极增强数据可用性并减少对 AI 当前局限性依赖的努力。


  

---

### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1327438062023606354)** (2 条消息): 

> `NotebookLM Mobile Experience Study, Feedback on Audio Overviews, Participant Incentives, User Experience Research` 


- **NotebookLM 征集移动端体验反馈**：团队邀请参与者参加关于即将推出的 **NotebookLM mobile experience** 的远程访谈，时间定于 **1 月 14 日至 15 日**。感兴趣的用户可以通过此 [筛选表单](https://forms.gle/75gYapqbgCmxXiJL6) 报名，争取分享观点的机会。
   - 参与者将获得等值 **$50** 的报酬，或者可以选择 Google 商品代金券作为谢礼。
- **征求 Audio Overviews 反馈**：一项约 **5 分钟的快速筛选调查** 已启动，旨在收集对 NotebookLM 的 **Audio Overviews** 功能的反馈。完成后续调查的合格参与者将通过电子邮件收到 **$10** 礼品码以示感谢。
   - 潜在参与者必须年满 **18 岁**，并可以通过提供的 [表单](https://forms.gle/NBzjgKfGC24QraWMA) 表达参与意向。请注意，礼品码仅在完成调查后发放。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://forms.gle/NBzjgKfGC24QraWMA">登记您的兴趣：Google 反馈调查</a>: 您好，我们正在通过一项简短的调查征集关于 NotebookLM 的反馈。这将帮助 Google 团队更好地了解您的需求，以便将其纳入未来的产品改进中。要登记...</li><li><a href="https://forms.gle/75gYapqbgCmxXiJL6">我们想知道您的想法！</a>: 感谢您有兴趣与我们交流！我们收到了很多关于 NotebookLM 移动端体验的关注，非常希望能听听您的看法。我们目前正在安排参与者...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1327368658057564182)** (46 条消息🔥): 

> `Notebook LM 功能, Podcast 分享平台, AI 辅助 D&D 资源, 音频概览反馈, AI 在教育中的应用` 


- **Notebook LM 助力深度研究摘要**：一位用户测试了 Notebook LM，要求对其原创研究论文进行音频概览，生成了超过 **28 分钟** 的摘要，有效地捕捉了关键方面。
   - 尽管作为作者可能存在主观偏见，他们仍寻求他人对该概览清晰度的反馈。
- **使用 Akas 轻松分享 Podcast**：用户讨论了通过 Notebook LM 分享 AI 生成的 Podcast 的局限性，特别是关于访问所需的登录权限。
   - 一位用户介绍了 [Akas](https://akashq.com)，这是一个允许轻松上传和分享 AI 生成的 Podcast 且没有登录限制的平台。
- **教育中的创新 AI 应用**：一段对话强调了使用 AI 总结讲座并增强学习效果，特别是一位护理系学生将讲座笔记转化为 Podcast。
   - 成员们分享了使用 AI 工具创建引人入胜的教育内容的个人经验和建议。
- **多语言 AI 讨论**：一位用户提出了创建由机智主持人引导的多语言小组讨论的指令，并加入各种角色以增加趣味性。
   - 该想法建议以幽默且引人入胜的形式探索多样化的对话和讨论。
- **通过音频吸引读者**：一位作者展示了利用 AI 音频摘要为读者提供小说预览的潜力，从而增强互动。
   - 他们将其比作书籍的“试驾”，展示了 AI 如何通过原创对话让故事栩栩如生。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://akashq.com">Akas: AI Podcast 之家</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/notebook/6dd1946b-561b-446c-818a-e9e17e332aac/audio">未找到标题</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/notebook/a6acd033-3af7-41e6-a258-ed7ac973f184/audio">未找到标题</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/notebook/2119508c-a81e-4a23-8f61-2f7363af4ea3/audio">未找到标题</a>: 未找到描述</li><li><a href="https://youtu.be/lqaqLZ9ha2I">AI 十四行诗：Gemini、VideoFX 和 Notebook LM 的实验</a>: 这是一个实验，我要求 Gemini 根据我选择的主题创作一些十四行诗，手动进行完善，并交替参考来自 Perplexity 的建议……</li><li><a href="https://youtu.be/-C6k5IGBDbY?si=V24b-gkszFj5xZcV),">【AI Podcast】我让 AI 介绍我自己！实验性 Live2D Podcast</a>: 这段实验视频是结合了人工智能、Live2D 动画和编程的结果，旨在探索 AI 如何总结一个人——在本例中是我……</li><li><a href="https://www.akashq.com/post/b966107e-ef54-41a0-a664-21dc27f841e6">1 月 10 日发生了什么？</a>: 1 月 10 日发生了什么？来自 This Day in History</li><li><a href="https://www.akashq.com/post/2e2231bf-907b-4805-84ae-71f8a7a45c19">1 月 13 日发生了什么？</a>: 1 月 13 日发生了什么？来自 This Day in History</li><li><a href="https://www.akashq.com/post/db284bc7-a4bb-4144-a4d4-05d496f71dd0">1 月 12 日发生了什么？</a>: 1 月 12 日发生了什么？来自 This Day in History
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1327397405590814822)** (289 条消息🔥🔥): 

> `NotebookLM 功能与限制，使用 NotebookLM 进行研究，播客自定义，嵌入 NotebookLM，用户入门与支持` 


- **了解 NotebookLM 的功能与限制**：用户讨论了 NotebookLM 的局限性，包括添加多个来源时的问题、理解输出中的引用链接，以及缺少修订目录等功能。
   - 许多人对 Notebook 中可以处理和访问的来源数量表示困惑，导致输出不如预期时产生挫败感。
- **有效利用 NotebookLM 进行研究**：用户分享了在学习中使用 NotebookLM 的策略，包括如何创建摘要、音频概览（audio overviews）以及有效管理来源；其他人则强调了为文档正确命名和格式化以提高模型访问效率的重要性。
   - 一些用户还建议创建 Prompt 来增强播客输出的清晰度和长度，尽管自定义的成功率并不一致。
- **播客功能与自定义**：讨论了自定义 NotebookLM 生成的播客的方法，用户尝试了各种 Prompt 以确保特定主持人参与并控制剧集长度。
   - 一些用户注意到音频概览生成方面的挑战，并寻求有效管理播客内容的技巧。
- **对嵌入 NotebookLM 的兴趣**：一位用户询问如何将 NotebookLM 嵌入网站，特别是与 Google Sites 的集成，表明了将其功能扩展到个人使用之外的愿望。
   - 这种兴趣凸显了 NotebookLM 适配更广泛的协作或教育环境的潜力。
- **用户体验与入门**：新用户表达了他们在操作 NotebookLM 时的体验和挑战，反复提到指令的清晰度和客户支持的有效性。
   - 一些社区成员提供了帮助并分享了对工具的见解，互相帮助以更好地利用 NotebookLM 的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://illuminate.google.com/home">Illuminate | Learn Your Way</a>：使用 Illuminate 将研究论文转换为 AI 生成的音频摘要，这是您的 Gen AI 工具，可帮助您更快地理解复杂内容。</li><li><a href="https://chromewebstore.google.com/detail/notebooklm-web-importer/ijdefdijdmghafocfmmdojfghnpelnfn))">Chrome Web Store</a>：为您的浏览器添加新功能并个性化您的浏览体验。</li><li><a href="https://thedrive.ai?">The Drive AI: Revolutionizing File Management &amp; Knowledge Bases</a>：发现 The Drive AI 在智能文件组织方面的突破。我们的平台在尖端 AI 的帮助下将您的文件转化为动态知识库。提升您的业务运营...</li><li><a href="https://youtu.be/spj0n-bFKJo?si=IKMq04zZW7KZMHeZ&t=453"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/spj0n-bFKJo?t=659&si=vrheRfu7QcBE4S2K">NotebookLM: 10 Exclusive Tips Not Found on the Web! (2025)</a>：免费且最佳的 AI 工具，可提升您的研究和内容创作水平 - NotebookLM！从即时消化数百份文档、视频、网站到多语言支持...</li><li><a href="https://youtu.be/pEC3-5oeIQU?si=3DlU22lAWEAycdzM">Drunk AI Discusses The History of Drinking Alcohol | Funny | Podcast</a>：当 AI 掌握了太多关于酒精的事实时会发生什么？在这个轻松幽默的剧集中，我们带你微醺地深入了解迷人的饮酒历史...</li><li><a href="https://form.typeform.com/to/bOA6l2qF)]">Discover Typeform, where forms = fun</a>：在几分钟内无需代码即可创建美观、互动的表单。免费开始使用。
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1327372933437587556)** (313 条消息🔥🔥): 

> `Pony 模型对比 Illustrious，Dreambooth 与训练 LoRA，高分辨率生成技术，Stable Diffusion 的扩展与工具，使用 AI 生成图像`

- **Pony 模型与 Illustrious 相比质量较差**：许多用户报告称，虽然 **Pony XL** 具有很高的标签凝聚力（tag cohesion），但其训练水平较低，导致结果不尽如人意。相比之下，**Illustrious** 因在处理写实图像和角色生成方面表现更好而更受青睐。
   - 值得注意的是，**JuggernautXL** 和 **RealVision v5** 也是实现写实效果的可靠替代方案。
- **Dreambooth 训练重心发生转移**：由于方法陈旧，用户正逐渐放弃 **Dreambooth**，转而使用 **Kohya_ss** 和 **OneTrainer** 等工具来训练模型。一些参与者发现过去的 **Dreambooth** 教程已经过时且无效，正在寻求更前沿的资源。
   - 对于**特定角色的 Loras**，建议使用 50 到 150 张图像进行有效训练。
- **高分辨率生成技术**：在 **Stable Diffusion** 中使用 **hires fix** 技术允许用户先生成 1024x1024 的图像然后进行放大，从而获得更高质量的输出。直接在高分辨率下生成往往会导致图像重复和不连贯。
   - 许多参与者建议从较低分辨率开始并启用 **hires fix** 以获得理想的效果。
- **Stable Diffusion 的扩展和工具**：用户讨论了各种扩展和工具，例如用于在 **Stable Diffusion** 中实现更好图像控制的 **sd-webui-regional-prompter**。会议强调了安装方法的重要性，例如通过 **git clone** 到正确的目录。
   - 此外，还发出了关于 **Discord** 和第三方支持链接中潜在诈骗的警告。
- **使用 AI 进行图像生成**：对于追求快速生成而非高质量的普通用户，推荐使用 **SDXL-Turbo** 等 **turbo models**，因为它们速度极快。根据模型和设置的不同，一些用户报告可以快速生成图像，但适当的训练和数据集编译对于高质量输出仍然至关重要。
   - 对各种图像生成模型的反馈表明，经验法则（rule of thumb）是选择那些需要较少步数（steps）即可获得快速结果的模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/mmazco/status/1876336631080419593)">来自 maz (@mmazco) 的推文</a>: .@doji_com 想到了一种非常巧妙的测试其 App 的方法 - 让用户上传自拍并试穿各种衣服。通过产品发现实现的电商体验策展与提升...</li><li><a href="https://huggingface.co/stabilityai/sdxl-turbo">stabilityai/sdxl-turbo · Hugging Face</a>: 暂无描述</li><li><a href="https://stability.ai/news/stable-point-aware-3d">Introducing Stable Point Aware 3D: Real-Time Editing and Complete Object Structure Generation  &mdash; Stability AI</a>: Stable Point Aware 3D (SPAR3D) 实现了在不到一秒的时间内，从单张图像实时编辑并生成 3D 对象的完整结构。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/14x6o2c/finally_figured_out_how_to_create_realistic/?rdt=54950">Reddit - 深入探讨</a>: 暂无描述</li><li><a href="https://safebooru.org/index.php?page=post&s=list&tags=mount_fuji">Safebooru / 富士山</a>: 暂无描述</li><li><a href="https://civitai.com/models/139562/realvisxl-v50)">RealVisXL V5.0 - V5.0 Lightning (BakedVAE) | Stable Diffusion XL Checkpoint | Civitai</a>: 新年快乐。在 Mage 上查看我的专属模型：ParagonXL / NovaXL / NovaXL Lightning / NovaXL V2 / NovaXL Pony / NovaXL Pony Lightning ...</li><li><a href="https://youtu.be/8eHYYFgzNW0">glossy workshop scan</a>: 暂无描述</li><li><a href="https://github.com/hako-mikan/sd-webui-regional-prompter">GitHub - hako-mikan/sd-webui-regional-prompter: 为划分的区域设置提示词</a>: 为划分的区域设置提示词。通过在 GitHub 上创建账户来为 sd-webui-regional-prompter 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=FvpWy1x5etM">FLUX Full Fine-Tuning / DreamBooth Training Master Tutorial for Windows, RunPod &amp; Massed Compute</a>: 如果你想以最高质量训练 FLUX，这就是你要找的教程。在这个综合教程中，你将学习如何安装 Kohy...</li><li><a href="https://youtu.be/MQz58wPvT3I?t=4887">ALL THINGS PONY! ft. AstraliteHeart // Creator of Pony Diffusion XL V6 // Civitai Guest Creator</a>: 在这段视频中，Ally 采访了令人惊叹的 Pony Diffusion XL V6 模型的创作者 AstraliteHeart！他们一起讨论并深入探讨了关于 Pony 的一切...</li><li><a href="https://github.com/Haoming02/sd-forge-couple">GitHub - Haoming02/sd-forge-couple: Forge Webui 的一个扩展，实现了 Attention Couple</a>: Forge Webui 的一个扩展，实现了 Attention Couple - Haoming02/sd-forge-couple
</li>
</ul>

</div>

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1327379923672891412)** (138 条消息🔥🔥): 

> `AI 模型成本与性能、模型的引用与训练、AI 服务与工具、零售业中的生成式 AI、最新 AI 研究与技术` 


- **AI 模型成本与性能图表**：分享了一张详细比较各种 AI 模型成本和性能的图表，展示了 o1-preview 和 GPT-4o 等模型在价格和 Elo 评分方面的对比情况。
   - 该图表有助于直观展示 AI 模型市场的竞争格局，突出了性价比等趋势。
- **GitHub Copilot 新功能**：Satya Nadella 宣布取消 GitHub Copilot Workspace 的等待名单，并将其宣传为一款可供使用的先进 Agentic 编辑器。
   - 这一变化旨在让用户比以前更轻松地利用 AI Agent 进行构建。
- **Raspberry AI 投资动态**：来自 a16z 的 Bryan Kim 宣布投资 Raspberry AI，这是一个专为零售产品开发量身定制的生成式 AI 设计平台。
   - 此次投资体现了通过 AI 驱动工具变革零售设计的创新承诺。
- **Llama 3 模型速度记录**：用户分享了 Llama 3 模型令人印象深刻的速度指标，报告的速度超过了传统配置，突显了 SambaNova 云技术的效率。
   - 使用定制芯片 (SN40L) 允许同时运行多个模型，显著提升了 AI 应用的部署效率。
- **STORM：LLM 驱动的文章生成**：Aashutosh.dev 介绍了 STORM，这是一个由 LLM 驱动的系统，旨在利用网络搜索进行研究，并撰写类似维基百科的文章。
   - STORM 生成包含完整引用的综合报告，展示了 LLM 在内容创作中的实际应用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://www.kaggle.com/whitepaper-agents">Agents</a>：作者：Julia Wiesinger, Patrick Marlow 和 Vladimir Vuskovic</li><li><a href="https://blog.gumloop.com/gumloops-17m-series-a/">Gumloop 的 1700 万美元 A 轮融资</a>：我们很高兴能开启 Gumloop 增长的下一阶段，完成了由 Nexus Venture Partners 领投，First Round Capital、Y Combinator 以及 Max Mullen 等天使投资人参投的 1700 万美元 A 轮融资...</li><li><a href="https://simonwillison.net/2025/Jan/10/ai-predictions/">我为 Oxide and Friends 做的未来 1、3、6 年 AI/LLM 预测</a>：Oxide and Friends 播客有一个年度传统，即邀请嘉宾分享对未来 1、3、6 年的预测。这里是 2022、2023 和 2024 年的内容。这……</li><li><a href="https://x.com/Sebasti54919704/status/1877948459103515020">Sebastian Sosa (@Sebasti54919704) 的推文</a>：@huggingface HuggingFace 对结构化输出（受限解码，constrained decoding）的解决方案/策略是什么？？问了所有人，似乎没人知道。我目前最好的线索是它正被卸载到 ...</li><li><a href="https://x.com/AymericRoucher/status/1878456854856048746">Aymeric (m-ric) (@AymericRoucher) 的推文</a>：-> OS-Genesis：为什么不通过探索而不是高级任务来生成 GUI Agent 轨迹？（剧透：效果真的很好🔥）构建 GUI Agent 的主要瓶颈在于 ...</li><li><a href="https://arxiv.org/abs/2412.19048">Jasper and Stella：SOTA embedding 模型的蒸馏</a>：许多深度学习应用（如 FAQ 和 RAG）的一个关键组件是稠密检索（dense retrieval），其中 embedding 模型用于将原始文本转换为数值向量，然后获取最相似的...</li><li><a href="https://www.anthropic.com/research/building-effective-agents">构建高效 Agent</a>：一篇面向开发者的文章，提供了构建高效 AI Agent 的建议和工作流</li><li><a href="https://www.all-hands.dev/blog/dont-sleep-on-single-agent-systems">All Hands AI</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2305.15717">模仿闭源 LLM 的虚假承诺</a>：一种廉价提升较弱语言模型的新兴方法是在较强模型（如 ChatGPT 等专有系统）的输出上对其进行微调（例如 Alpaca、Self-Instruct 等）。这...</li><li><a href="https://x.com/altryne/status/1877220144725758414?s=46">Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：天哪伙计们... 微软刚刚让 Qwen 7B 达到了 o1 级别的 AIME 解题水平 😵‍💫 他们还展示了通过其 MCTS 驱动过程，模型具备了像推理模型一样的自我反思能力。是否 ...</li><li><a href="https://x.com/ilanbigio/status/1878940258349510764?s=46">ilan bigio (@ilanbigio) 的推文</a>：发布我们全新的 OpenAI function calling 指南！我们听取了大家的反馈并做了一些关键改进：- 缩短了 50% 且更清晰 - 新的最佳实践（详见下文 👇）- 文档内函数生成...</li><li><a href="https://simonwillison.net/2024/Dec/20/building-effective-agents/">构建高效 Agent</a>：我对“Agent”这个词的主要抱怨是，虽然它有许多不同的潜在定义，但大多数使用它的人似乎都假设其他人的理解是一致的……</li><li><a href="https://x.com/dottxtai/status/1877760709246824919">.txt (@dottxtai) 的推文</a>：你可能经常听到“Agent”这个词。但它到底是什么意思？Agent 的几个主题：- 自主性（在没有人类交互的情况下运行）- 感知（通过...接收信息）</li><li><a href="https://x.com/hwchase17/status/1867683506861838635?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Harrison Chase (@hwchase17) 的推文</a>：在与更多认真考虑将 Agent 投入生产的公司交流时，我看到的一个共同趋势是：“单 Agent” -> “高级多 Agent (crewai, autogen)” -> ...</li><li><a href="https://youtube.com/playlist?list=PLLAfEmC9OS7WgFe4Te5sFa3l9J_K7qDrq&si=kOQtY9_u5qysiOht">Good AI</a>：未找到描述</li><li><a href="https://x.com/maxbrodeururbas/status/1877778718208446567?s=46">Max Brodeur-Urbas (@MaxBrodeurUrbas) 的推文</a>：Gumloop 将成为一家只有 10 人的 10 亿美元公司。我们还剩 6 个名额。引用 Gumloop (@gumloop_ai)：我们很高兴宣布 Gumloop 完成了由 @NexusVP 领投，@firstround 等参投的 1700 万美元 A 轮融资...</li><li><a href="https://huggingface.co/docs/smolagents/conceptual_guides/intro_agents">Agent 简介</a>：未找到描述</li><li><a href="https://x.com/hcompany_ai/status/1877403314091852169">H (@hcompany_ai) 的推文</a>：要去拉斯维加斯参加 #CES2025 吗？让 Runner H 为你预定必看的活动。几分钟内搞定完美行程！ #RunnerH</li><li><a href="https://x.com/swyx/status/1878392101815099728?s=46">swyx.io (@swyx) 的推文</a>：等等，你在开玩笑吗？这些教授是怎么随随便便就约到 @denny_zhou, @lmthang, @hanjundai 以及 2 位 strawberry 研究员的？</li>

<li>oors 以及其他顶尖 LLM 专家</li><li><a href="https://x.com/backus/status/1878484938003034391?s=46">来自 John Backus (@backus) 的推文</a>：Zuck 批准了为 Llama 在 LibGen 上进行种子下载和训练。Founder mode。</li><li><a href="https://x.com/wayne_hamadi/status/1868742755402621103?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Wayne Hamadi 🖇️ (@wayne_hamadi) 的推文</a>：http://x.com/i/article/1867032485768597505</li><li><a href="https://yenchenlin.github.io/blog/2025/01/08/video-generation-models-explosion-2024/">2024 年视频生成模型爆发 - Yen-Chen Lin</a>：未找到描述</li><li><a href="https://x.com/svpino/status/1878797424590012907">来自 Santiago (@svpino) 的推文</a>：这是我见过的 Llama 3.3 在任何地方运行最快的一次！Llama 3.3 70B 以 652 t/s 的速度运行，简直快如闪电。如果你想要 Llama 3.1，这是我能达到的速度：• Llama 3.1 8B: 1006 t/...</li><li><a href="https://x.com/slow_developer/status/1877798620692422835?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Haider. (@slow_developer) 的推文</a>：🚨 Mark Zuckerberg 在 Joe Rogan 的播客中表示：到 2025 年，Meta 和其他公司的 AI 系统将能够像中级工程师一样编写代码。起初成本很高，但系统将变得...</li><li><a href="https://x.com/kalinowski007/status/1877809579154948223">来自 Caitlin Kalinowski 🇺🇸 (@kalinowski007) 的推文</a>：非常激动能为 @OpenAI 发布我们的首批 Robotics 硬件职位，包括两个资深技术主管工程师 (IC) 职位和一个 TPM Manager。第一个职位是 **EE Sensing Engineer**...</li><li><a href="https://www.reforge.com/blog/ai-native-product-teams">Reforge</a>：未找到描述</li><li><a href="https://x.com/nrehiew_/status/1877956822318862768?s=46">来自 wh (@nrehiew_) 的推文</a>：> 使用 QwQ 生成补全 > 使用 GPT 4o mini 格式化输出 > 移除答案错误的样本 > 在 17k 样本上进行标准 SFT > 在 8xH100 上运行 19 小时 ($450)。大原因...</li><li><a href="https://x.com/satyanadella/status/1878578314115473577?s=46">来自 Satya Nadella (@satyanadella) 的推文</a>：GitHub Copilot Workspace 已不再需要排队——这是最先进的 agentic editor。今天就开始使用 agents 进行构建。</li><li><a href="https://x.com/BlackHC/status/1878883222911877375">来自 Andreas Kirsch 🇺🇦 (@BlackHC) 的推文</a>：NeurIPS 2024 的程序委员会（PCs）简直是一群小丑 🤡 ML 的现状 🙄 在提出疑虑一个月后，你得到的只有：</li><li><a href="https://centml.ai/">首页 - CentML</a>：降低 LLM 服务成本高达 65%。提升您的 AI 效率，在优化 GPU 的同时加速部署和推理 [&hellip;]</li><li><a href="https://x.com/kirbyman01/status/1878844418972885077">来自 Bryan Kim (@kirbyman01) 的推文</a>：很高兴宣布我们正领投 Raspberry AI 的 A 轮融资，这是一个专为零售产品设计打造的端到端 generative AI 设计平台。为什么我们的团队 @a16z 决定投资（抄送：@zachco...</li><li><a href="https://x.com/teortaxesTex/status/1877958319127597452">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：在我看来这毫无进展，这又是 Alpaca 时代的“OpenAI 没有护城河”论调。是的，在 benchmarks 上与 o 系列有狭隘的对等，但随着我们扩大规模并尝试泛化到更难的问题...</li><li><a href="https://x.com/swyx/status/1838663794320642328">来自 swyx.io (@swyx) 的推文</a>：2024 年 9 月更新 https://x.com/Smol_AI/status/1838663719536201790 引用 Smol AI (@Smol_AI) 的 AI 新闻：值得注意的是 Lmsys Elo 与价格曲线的预测性是多么强，以及这种策略是...</li><li><a href="https://x.com/bryantchou/status/1877790833371697169?s=46">来自 brryant (@bryantchou) 的推文</a>：我很久没对一款软件产品这么兴奋了... 我想大概是从 Webflow 以来吧。😅 在过去 6 个月使用 gumloop 的过程中，我：— 自动化了竞争情报研究 (Reddit) — 自动化了竞争对手广告策略...</li><li><a href="https://youtu.be/c_9bxtyOd1o?si=31RJlBdZ0E_PLfCH">Hyung Won Chung：从 Transformer 历史塑造 AI 的未来</a>：OpenAI 研究科学家 Hyung Won Chung 在 Naik 教授的课程 CIS 7000：Large Language Models（2024 秋季）中于 2024 年 10 月 14 日进行的客座讲座。</li><li><a href="https://youtu.be/yhpjpNXJDco?si=7sfPgTlyCTi3lNLP">Jason Wei：大语言模型的缩放范式</a>：OpenAI 技术人员 Jason Wei 在 Naik 教授的课程 CIS 7000：Large Language Models（2024 秋季）中于 2024 年 11 月 20 日进行的客座讲座。</li><li><a href="https://youtu.be/SN4Z95pvg0Y?si=wyrwJ1VeV2BFElLG"> - YouTube</a>：未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/1x9bQVlm7YJ33HVb3AGb9qlDNkvTy9CyOFZoah0kr3wo/edit?gid=0#gid=0">LLM Elo 与价格图表</a>：未找到描述</li><li><a href="https://x.com/bclavie/status/1878349981570187311">来自 Benjamin Clavié (@bclavie) 的推文</a>：🧵 Stella Embeddings：有什么了不起的？（简短解释推文串）如果你喜欢 RAG 推特圈，或者强迫性地查看 M...</li>

TEB 排行榜，你可能遇到过 "Stella" (以及 n...</li><li><a href="https://docs.google.com/document/d/10fnHaH5uEAh-xmc79D7jGB7gJAt-7wQKhZBM2cr6xAc/edit">Scaling Paradigms for Large Language Models</a>：以下是基于 YouTube 视频转录的技术文章：https://youtu.be/yhpjpNXJDco?si=7sfPgTlyCTi3lNLP Scaling Paradigms for Large Language Models 简介 人工智能领域...</li><li><a href="https://cs329a.stanford.edu/">Stanford CS329A | Self-Improving AI Agents</a>：未找到描述</li><li><a href="https://youtu.be/YdqJSjfi4iw?si=WdLU6j-V_LW_H9jZ)">Aakanksha Chowdhery: Multimodal Reasoning and its Applications to Computer Use and Robotics</a>：Meta 高级首席研究科学家 Aakanksha Chowdhery 在 Naik 教授的课程 CIS 7000: Large Language Models（2024 秋季）于 11 月 2 日进行的客座讲座...</li><li><a href="https://youtu.be/kOdl-ncrYDk?si=wDiEgrbW1iAPaUGK)">Hanjun Dai: Preference Optimization for Large Language Models</a>：Google Brain 首席研究科学家兼研究经理 Hanjun Dai 在 Naik 教授的课程 CIS 7000: Large Language Models（2024 秋季）进行的客座讲座...</li><li><a href="https://youtu.be/T1SeqBapMBo?si=VIkFfcGoROxH7JMu)">LTI Special Seminar by Yi Wu</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1327379067447546028)** (25 messages🔥): 

> `新播客剧集, O1 客座文章讨论, O1 Pro 用户体验, 文章登上 HN, O1 的动态使用` 


- **新播客剧集发布**：发布了一集关于 MMLU 知识局限性讨论的新播客，强调了我们对智能的渴望胜过对死记硬背的琐碎知识。该剧集包含了 William Bryk 对 ExaAILabs 令人印象深刻的设置的见解。
   - *为即将到来的事情做好准备*，因为他们详细介绍了 Exacluster 令人印象深刻的技术规格，包括 **144 个 H200** 和 **20TB GPU RAM**。
- **O1 Pro 的用户体验**：用户讨论了他们使用 O1 Pro 的体验，强调了与标准版本相比，它在处理结构化上下文时表现出的卓越性能。一位用户分享了处理 **20k** token 的 React/TS 代码库的见解，断言 O1 Pro 非常可靠。
   - 尽管享受到了好处，用户仍表示谨慎，权衡了成本与每月 **20 美元** 方案等替代方案，并提醒自己预算限制。
- **文章登上 HN**：最近关于 O1 的客座文章曝光率激增，并登上了 Hacker News 首页，吸引了社区的关注。这一成就聊天室中得到了热烈庆祝，突显了文章的影响力。
   - 许多人对这些发现对使用 O1 Preview 的影响表示好奇，证实了讨论的相关性不仅限于主版本。
- **O1 的动态使用**：一位成员强调需要一种不同的方式来使用 O1，将其比作 *报告生成器（report generator）* 而非典型的聊天模型。这一概念引发了关于 Prompting 如何增强性能和可用性的进一步讨论。
   - 分享了来自 *Sam Altman* 和 *Ben Hylak* 等影响力人物的语录，以说明在有效利用该技术方面不断演变的观点。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/swyx/status/1877818998060175508">来自 swyx.io (@swyx) 的推文</a>：非常奇怪的是，我们花费数十亿参数来存储 99% 的人并不真正需要、且可以按需查找/学习的 MMLU/GPQA 知识。我们要的是智能；我们得到的是...</li><li><a href="https://www.latent.space/p/o1-skill-issue">o1 不是聊天模型（这正是重点）</a>：Ben Hylak 如何通过克服他的技能问题（skill issue），从 o1 pro 的怀疑者转变为粉丝。</li><li><a href="https://x.com/benhylak/status/1878237490194366744?s=46">来自 ben (@benhylak) 的推文</a>：当你懂得如何使用时，o1 是令人惊叹的。它真的不是一个聊天模型——你必须更多地把它看作是一个“报告生成器（report generator）”（下方有文章链接）引用 Sam Altman (@sama) 的话...</li><li><a href="https://x.com/gdb/status/1878489681702310392">来自 Greg Brockman (@gdb) 的推文</a>：o1 是一种不同类型的模型。出色的性能需要相对于标准聊天模型以一种全新的方式来使用它。引用 Dan Mac (@daniel_mac8) 的话，这是一种思考 o1 Prompting 的绝佳方式...
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1327381316387340313)** (116 messages🔥🔥): 

> `Claude Projects, AI Tools in Development, Ruby for Prototyping, Mob Coding, AI Applications in Interior Design` 


- **Claude Projects 成为个人助手**：用户们正在分享他们如何将 **Claude Projects** 整合到工作流中，并表示已经对此上瘾，报告了显著的生产力提升。
   - *“Claude Projects 目前基本上就是我的个人助手，我彻底迷上了”* 突显了用户对该工具的喜爱。
- **从 Gatsby 迁移到 Astro**：一位成员分享了他们迁移 **Gatsby** 到 **Astro** 的经验，利用 **bolt.new** 进行原型设计，并使用 **Cursor** 处理复杂功能，将开发时间缩短了 **60%**。
   - 他们暗示，熟悉 AI 工具对于处理棘手元素和最佳实践至关重要。
- **Ruby 在快速原型设计中的角色**：参与者讨论了对使用 **Ruby** 编写 LLM 生成代码的复杂感受，欣赏它在人工原型设计中的表现，但觉得它在 LLM 输出质量方面有所欠缺。
   - 一位成员表示他们很喜欢 Ruby，称：*“我在 Ruby 领域投入了大量时间，大部分时间都很享受”*，表明了优缺点之间的平衡。
- **探索 Mob Coding**：参与者对 **Mob Coding** 表现出兴趣，认为这是一种独特的协作开发方法，一些人表达了探索这一概念的个人乐趣。
   - 诸如 *“Mob coding 听起来很酷”* 之类的评论引发了关于其在团队环境中有效性的进一步讨论。
- **AI 在室内设计中的应用**：一位用户注意到 AI 工具在 **室内设计** 中的成功应用，可以根据客户偏好生成匹配的装饰物品，展示了 AI 的实际应用。
   - 示例包括生成如枕头之类的物品以匹配特定的配色方案，展示了 AI 在创意领域的通用性。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://claude.site/artifacts/0565699b-deab-419d-9634-ae60ece764a5">Claude Artifact</a>: 尝试由 Claude 用户创建的 Artifacts</li><li><a href="https://changelog.com/jsparty/338">Undirected hyper arrows with Chris Shank (JS Party #338)</a>: Chris Shank 自 1 月以来一直在休假，因此他有大量时间深入思考 Web 平台。在本期节目中，Jerod 和 KBall 请教 Chris 以回答诸如...的问题。</li><li><a href="https://github.com/Little-Languages/quiver">GitHub - Little-Languages/quiver: Your quiver of declarative arrows for the web. ⤵</a>: 你的 Web 声明式箭头箭袋。通过在 GitHub 上创建账户为 Little-Languages/quiver 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1327411769626329190)** (5 messages): 

> `Aider v0.71.0, Chat mode switching, DeepSeek prompts, Pretty output in editing, Release history insights` 


- **Aider v0.71.0 显著特性**：Aider v0.71.0 引入了提示词，以帮助 **DeepSeek** 在交替使用 `/ask` 和 `/code` 命令时表现更好。
   - 流式美化 **LLM 响应** 现在更加平滑快速，显著提升了用户交互体验。
- **命令现在可以切换聊天模式**：单纯的 `/ask`、`/code` 和 `/architect` 命令现在允许用户切换聊天模式，使沟通更加直观。
   - 正如所提到的，使用 `/ask` 意味着后续所有消息都将被视为问题，用户觉得这非常棒。
- **编辑时保持美化输出**：用户对即使在使用 **triple-backtick fences**（三反引号围栏）编辑文件时也能保持美化输出的功能感到兴奋。
   - 一位用户将这一变化描述为“巨大的”，强调了它的实际益处。



**Link mentioned**: <a href="https://aider.chat/HISTORY.html">Release history</a>: 关于 aider 编写自身代码的发布说明和统计数据。

  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1327395246580961411)** (182 条消息🔥🔥): 

> `DeepSeek Model Performance, Model Configuration in Aider, Quantization for Neural Networks, AI Coding Tools Improvement, Polyglot Benchmark Issues` 


- **DeepSeek 模型的可靠性问题**：用户报告称 **DeepSeek** 一直不稳定且无响应，导致错过截止日期并令人沮丧。
   - 大家一致认为需要解决 API 的不稳定性，以改善整体用户体验。
- **Aider 配置挑战**：一位用户在尝试在 **.aider.conf.yml** 中设置 `editor-model` 时遇到错误，发现应该使用连字符（dash）而不是下划线（underscore）。
   - 这引发了关于是否应将配置文件包含在仓库的 gitignore 中的讨论。
- **神经网络量化讨论**：讨论了关于神经网络的 **quantization**（量化），一位用户提到理解这一概念对于有效编码的重要性。
   - 用户表示 LLM 需要更好地掌握基础概念，以防止编码任务中出现问题。
- **AI 编程工具的潜在改进**：小组对 **AI coding tools** 改进的潜力感兴趣，但一些人认为进展取决于 LLM 的能力。
   - 参与者辩论了在各种编码任务中使用 **Sonnet** 和 **O1** 等模型的有效性。
- **Polyglot 基准测试**：一位用户分享了观察结果，即 **polyglot benchmark** 中的某些 C++ 练习可能需要特定标志才能正确运行所有测试。
   - 另一位用户表示有兴趣在特定基准测试任务上比较 **O1** 与 **Sonnet** 的性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/codestral-2501/">Codestral 25.01</a>：以 Tab 的速度编写代码。今日已在 Continue.dev 中上线，并即将登陆其他领先的 AI 代码助手。</li><li><a href="https://x.com/OpenRouterAI/status/1878876208877953235">OpenRouter (@OpenRouterAI) 的推文</a>：上周推理量增长了 23% 👀 @AnthropicAI 的 Claude 3.5 Sonnet (self-moderated) 是最大的来源</li><li><a href="https://x.com/hive_echo/status/1878400401164140890">echo.hive (@hive_echo) 的推文</a>：获取免费的完整 o1 API 使用权限 🥳 还有免费的 o1-mini 和 gpt-4o。1) 确保你拥有通过 API 访问 o1 的权限 2) 前往你的 dashboard > data controls > sharing 选项卡 3) 查看是否收到此通知</li><li><a href="https://aider.chat/docs/faq.html#how-do-i-include-the-git-history-in-the-context">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://github.com/Aider-AI/aider/blob/main/CONTRIBUTING.md">aider/CONTRIBUTING.md at main · Aider-AI/aider</a>：aider 是你终端中的 AI 结对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://github.com/stacklok/codegate">GitHub - stacklok/codegate: CodeGate: CodeGen Privacy and Security</a>：CodeGate：代码生成的隐私与安全。为 stacklok/codegate 的开发做出贡献。</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1hys13h/new_model_from_httpsnovaskyaigithubio/">来自 https://novasky-ai.github.io/ 的新模型 Sky-T1-32B-Preview，在流行的推理和编码基准测试中媲美 o1-preview 的开源推理模型 —— 训练成本低于 $450！</a>：由 u/appakaradi 发布于 r/LocalLLaMA • 501 点赞和 120 条评论</li><li><a href="https://aide.dev/">Aide - 你的 AI 编程助手</a>：以你所认识的最强程序员的速度和知识进行编码。Aide 就在你身边。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1327369508041785415)** (84 条消息🔥🔥): 

> `Aider 配置，Aider 中的 Prompt 缓存，在 Aider 中编辑文件，使用来自 Hyperbolic 的模型，处理 Aider 中的建议` 


- **Mac 上的 Aider 配置问题**：一位用户报告在通过 Homebrew 在 Mac 上安装 Aider 后，使用 /help 命令时遇到困难，并面临 tokenizers 的安装问题。
   - 他们寻求关于如何在 .env 文件中将 ask 模式设置为默认聊天模式的指导。
- **Prompt 缓存挑战**：关于 Prompt 缓存的讨论表明，缓存命中取决于是否包含完全相同的文件集，当动态添加文件时，这会导致使用上的挫败感。
   - 用户在讨论 Aider 管理只读文件的潜在优化方案时，也在考虑是否提交关于缓存效率低下的 Issue。
- **Aider 中的编辑与文件管理**：用户分享了使用 Aider 编辑文件的见解，包括使用 /add 命令快速添加目录，并提到有时会错误地创建诸如 'python' 之类的文件。
   - 有建议提出调整用户习惯，以避免在添加新文件时出现过多的建议。
- **使用来自 Hyperbolic 的模型**：一位用户询问了如何利用来自 Hyperbolic 的模型，特别是如何配置 Aider 使用 OpenAI 的 API 结构来调用特定模型（如 DeepSeek-V3）。
   - 社区澄清了用户可以从不同的提供商选择 LLM，并分享了特定的模型名称以确保在 Aider 中进行正确配置。
- **Aider 中的多行命令**：有人提出了关于在 Aider 中执行多行提问的问题，随后得到一个技巧：使用 Shift + Alt + Enter 可以实现此目的。
   - 此功能允许执行超出简单单行命令的更复杂指令，增强了用户与工具的交互。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI 兼容的 API</a>: aider 是你终端里的 AI 结对编程</li><li><a href="https://aider.chat/docs/usage/lint-test.html">Linting 与测试</a>: 自动修复 linting 和测试错误。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1327379527369887804)** (4 messages): 

> `browser-use, CodeGate, Deepseek AI Assistant, Always On AI Assistant` 


- **通过 browser-use 让网站对 AI 友好**：[browser-use 项目](https://github.com/browser-use/browser-use) 旨在通过增强交互能力，使网站更易于被 AI Agent 访问。
   - 这一进展对于改进 AI Agent 处理和导航网页内容的方式至关重要。
- **通过 CodeGate 关注 CodeGen 隐私**：[CodeGate](https://github.com/stacklok/codegate) 提供了对 CodeGen 隐私和安全措施的见解，旨在实现更安全的代码生成实践。
   - 参与该项目有助于增强 AI 驱动的编码环境中的安全协议。
- **Deepseek AI Assistant 始终在线**：[Deepseek AI Assistant](https://www.youtube.com/watch?v=zoBwIi4ZiTA) 的演示重点介绍了一个名为 Ada 的新型 Python AI Agent，它专为工程师持续运行而设计。
   - 这种创新方法将彻底改变工程师有效部署和管理代码的方式。
- **实时 AI Assistant 模式**：[always-on AI Assistant](https://github.com/disler/always-on-ai-assistant/) 利用 Deepseek-V3、RealtimeSTT 和 Typer 构建了一个高效的工程助手。
   - 该模式为开发响应迅速且能持续处理工程任务的助手提供了一个框架。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=zoBwIi4ZiTA">Deepseek AI Assistant: ALWAYS ON Python AI Agent for Engineers that SHIP</a>: 🔥 你的个人 AI Assistant 真的始终在线吗？探索由 DeepSeek V3 驱动的 Ada 如何彻底改变工程师交付代码的方式！🚀🎥 资源...</li><li><a href="https://github.com/stacklok/codegate">GitHub - stacklok/codegate: CodeGate: CodeGen Privacy and Security</a>: CodeGate: CodeGen 隐私与安全。通过在 GitHub 上创建账号为 stacklok/codegate 的开发做出贡献。</li><li><a href="https://github.com/disler/always-on-ai-assistant/">GitHub - disler/always-on-ai-assistant: A pattern for an always on AI Assistant powered by Deepseek-V3, RealtimeSTT, and Typer for engineering</a>: 一个由 Deepseek-V3、RealtimeSTT 和 Typer 驱动的工程用始终在线 AI Assistant 模式 - disler/always-on-ai-assistant</li><li><a href="https://github.com/browser-use/browser-use">GitHub - browser-use/browser-use: Make websites accessible for AI agents</a>: 让网站对 AI Agent 友好。通过在 GitHub 上创建账号为 browser-use/browser-use 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/)** (1 messages): 

louisgv: Phi 4 现已可用：https://openrouter.ai/microsoft/phi-4
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1327736854635745416)** (2 messages): 

> `Friday Agents, Telegram LLM Interface, DeVries AI Chatbot` 


- **Friday Agents 框架发布**：**Friday Agents** 的 GitHub 仓库介绍了一个强大的 JavaScript 框架，用于使用多 Agent 架构构建 **AI 驱动的应用程序**，详见 [GitHub - amirrezasalimi/friday-agents](https://github.com/amirrezasalimi/friday-agents)。
   - 该框架由两个主要组件组成，旨在简化 AI 应用程序的开发。
- **通过 Telegram 解锁 200+ AI 模型**：DeVries AI Chatbot 允许用户通过低成本订阅直接在 Telegram 中与 **200 多个大语言模型**对话，免费试用请访问 [devriesai.com](https://devriesai.com/)。
   - 仅需 **$24.99/月**，用户即可访问所有当前及未来的 AI 模型，通过熟悉的平台简化交互。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://devriesai.com/">devriesai</a>: 你的 Telegram AI Agent</li><li><a href="https://github.com/amirrezasalimi/friday-agents">GitHub - amirrezasalimi/friday-agents: Friday Agents. App: https://chat.toolstack.run/</a>: Friday Agents. App: https://chat.toolstack.run/。通过在 GitHub 上创建账号为 amirrezasalimi/friday-agents 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1327366626181644309)** (212 messages🔥🔥): 

> `OpenRouter 使用情况, Deepseek 模型性能, Mistral 的 Codestral 模型发布, 不同 LLM 的比较, OpenRouter 上的 Provider 部署`

- **OpenRouter 提供灵活的 LLM 选项**：用户对 OpenRouter 的 Deepseek V3 模型在性能和价格方面的表现表示满意，而另一些用户则在探索具有类似 OpenRouter 聊天界面功能的 Android 应用。
   - 用户对特定模型的局限性表示担忧，特别是在处理图像和性能不一致方面。
- **Mistral 发布 Codestral 模型**：Mistral 宣布推出其全新的 Codestral 模型，具有 262K context 并在先前版本的基础上进行了改进，尽管它已不再面向公众开放。
   - 该模型因其高效的架构和在编程任务中提升的速度而受到关注，但用户对其缺乏开放访问权限表示失望。
- **关于 LLM 定价和成本对比的讨论**：参与者讨论了实施 LLM 的各种云服务相关成本，一些人表示有兴趣对比不同平台之间的费用。
   - 用户对理想的服务商提出了疑问，特别是在考虑不同模型的性能和相关功能时。
- **关于 OpenRouter 模型提供商的见解**：一些用户询问如何成为 OpenRouter 的模型提供商，寻求在 OpenRouter 生态系统中部署模型的流程指导。
   - 对于有兴趣通过该平台提供自己模型的个人，Support 被提及为一个关键的联系点。
- **探索模型选择和使用策略**：关于不同 LLM 有效性的讨论表明，用户更倾向于生成全面报告的模型，而非仅进行对话式响应的模型。
   - 用户分享了根据特定用例选择模型的策略，并指出 context 和处理时间在选择中的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/codestral-2501/">Codestral 25.01</a>：以 Tab 的速度编写代码。今日已在 Continue.dev 上线，并即将登陆其他领先的 AI 代码助手。</li><li><a href="https://openrouter.ai/docs/crypto-api">Crypto Payments API | OpenRouter</a>：无需 UI 即可购买额度的相关 API</li><li><a href="https://openrouter.ai/api/v1",">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/docs/models">Models | OpenRouter</a>：所有可用模型的表格</li><li><a href="https://x.com/OpenRouterAI/status/1878876208877953235">来自 OpenRouter (@OpenRouterAI) 的推文</a>：上周推理量增长了 23% 👀 @AnthropicAI 的 Claude 3.5 Sonnet 自我审核版本是最大的来源</li><li><a href="https://github.com/openai/openai-node#undocumented-request-params">GitHub - openai/openai-node: Official JavaScript / TypeScript library for the OpenAI API</a>：OpenAI API 的官方 JavaScript / TypeScript 库 - openai/openai-node</li><li><a href="https://developers.cloudflare.com/ai-gateway/providers/openrouter/">OpenRouter · Cloudflare AI Gateway 文档</a>：OpenRouter ↗ 是一个为访问和使用大语言模型 (LLMs) 提供统一接口的平台。</li><li><a href="https://x.com/CloudflareDev/status/1861861672358654107">来自 Cloudflare Developers (@CloudflareDev) 的推文</a>：我们现在在 Cloudflare 的 AI Gateway 上支持 @OpenRouterAI。您现在可以将它们添加为提供商，以监控、记录和控制您的 OpenRouter LLM 请求。在此阅读更多关于如何添加它们的信息 👇</li><li><a href="https://openrouter.ai/api/v1">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://edition.cnn.com/2025/01/07/tech/meta-hateful-conduct-policy-update-fact-check/index.html">Meta 更新指南后，Facebook 现在允许称女性为“家用物品” | CNN Business</a>：未找到描述</li><li><a href="https://github.com/AaronWard/generative-ai-workbook/discussions/36">Weights &amp; Biases - 评估和测试 LLM 应用 · AaronWard/generative-ai-workbook · 讨论 #36</a>：文章 W&amp;B Sweeps - 用于迭代配置并评估指标，如使用的 tokens、成本、响应质量结果、不同模板、额外配置等。1. 理解...</li><li><a href="https://www.404media.co/its-total-chaos-internally-at-meta-right-now-employees-protest-zuckerbergs-anti-lgbtq-changes/">“Meta 内部目前完全陷入混乱”：员工抗议扎克伯格的反 LGBTQ 变革</a>：Meta 决定明确允许用户称 LGBTQ+ 群体为“精神病”，这在公司内部引发了广泛抵制。</li><li><a href="https://www.latestly.com/socially/world/mark-zuckerberg-orders-removal-of-tampons-from-mens-bathrooms-at-meta-offices-report-6556071.html#google_vignette">报道称马克·扎克伯格下令移除 Meta 办公室男厕所内的卫生棉条 | 🌎 LatestLY</a>：据报道，业务经理接到指示，要求移除男厕所内的卫生棉条，这些棉条是 Meta 为使用男厕所的非二元性别和跨性别员工提供的...</li><li><a href="https://www.404media.co/meta-deletes-trans-and-nonbinary-messenger-themes/">Meta 删除跨性别和非二元性别 Messenger 主题</a>：在一系列允许用户针对 LGBTQ+ 群体的变更中，Meta 删除了其最初倡导的产品功能。</li><li><a href="https://github.com/OpenRouterTeam/openrouter-runner">GitHub - OpenRouterTeam/openrouter-runner: Inference engine powering open source models on OpenRouter</a>：驱动 OpenRouter 上开源模型的推理引擎 - OpenRouterTeam/openrouter-runner
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1327375132133425263)** (190 条消息🔥🔥): 

> `Perplexity 订阅担忧、AI 模型对比、用户体验问题、图像生成功能、API 使用与成本` 


- **对 Perplexity 订阅成本的担忧**：用户对 Perplexity 订阅服务的定价表示担忧，特别是注意到 ChatGPT 每月 200 美元的成本。
   - 建议 Perplexity 推出更具竞争力的定价结构，以吸引专业用户。
- **关于 AI 模型有效性的讨论**：成员们辩论了 Perplexity 的模型是否优于 ChatGPT 等替代方案，对于 Claude 模型与新产品相比的效率，意见不一。
   - 用户指出，虽然 Perplexity 具有优势，但没有一个特定的模型被一致认为是最好的。
- **Perplexity 的用户体验问题**：几位用户报告了 Perplexity 应用的性能问题，提到了加载缓慢和持续的“等待中（pending）”通知。
   - 这些问题在不同设备上似乎一致存在，导致期望更高性能的专业用户感到沮丧。
- **图像生成功能**：对话强调了用户在 Perplexity 上生成图像时面临的困难，并建议利用 Grok 等外部工具以获得更好的效果。
   - 尽管面临挑战，许多用户仍表示希望平台内部能改进图像生成能力。
- **关于 API 成本和使用的说明**：用户询问了与 API 使用相关的成本，专业订阅中包含价值 5 美元的调用额度，但 Token 需要额外付费。
   - 用户对 Token 的定义存在困惑，并讨论了频繁使用 API 可能带来的财务影响。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://belladoreai.github.io/llama3-tokenizer-js/example-demo/build/">llama-tokenizer-js playground</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/faq/faq#why-are-the-results-from-the-api-different-from-the-ui>">未找到标题</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/guides/pricing">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/pplxsports/status/1878550531603312947?s=61">来自 Perplexity Sports (@PPLXsports) 的推文</a>: 全程比赛，拒绝噪音。</li><li><a href="https://x.com/cb_doge/status/1877570367239209273?s=46">来自 DogeDesigner (@cb_doge) 的推文</a>: 突发：Grok 在发布仅一天后就成为 AppStore 生产力类别中排名第 3 的应用。🇺🇸</li><li><a href="https://newsletter.moneylion.com/subscribe?ref=yJmsSyv2l7">MoneyLion Markets 每日通讯</a>: 您每日的市场新闻</li><li><a href="https://youtu.be/EXfFBEuCAr0?si=pJqgmK_4RVJiA8LO">你现在仍然需要一个网站！！（是的，即使在 2024 年）</a>: 在 5 秒内建立你的网站：https://hostinger.com/networkchuck10（使用优惠券代码 NETWORKCHUCK 可额外享受 10% 优惠）🗳️🗳️投票！！：谁拥有最好的网站...</li><li><a href="https://www.copilotforyoutube.com/search/joe-rogan-experience-2255-mark-zuckerberg-vAHjVHQqkgI07k7F3G4piE">Joe Rogan 访谈录 #2255 - Mark Zuckerberg</a>: Mark Zuckerberg 是 Meta Platforms Inc. 的首席执行官，该公司旗下拥有 Facebook、Instagram、Threads、WhatsApp、Meta Quest、Ray-Ban Meta 智能眼镜、Orion 增强现实眼镜等...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1327376696273403977)** (14 条消息🔥): 

> `Anthropic 估值、罗马帝国铅中毒、AI Chips、Bitcoin 追回、Spotify CEO 套现` 


- **Anthropic 估值达到 600 亿美元**：**Anthropic** 的 **600 亿美元估值**引起了广泛关注，讨论围绕其对 AI 行业的影响展开。
   - 这与其他 AI 进展一同出现，标志着投资者对新兴 **AI companies** 的浓厚兴趣。
- **铅中毒对 IQ 的影响被揭示**：最近的讨论强调了**罗马帝国**时期的**铅中毒**如何导致 **IQ 率**下降。
   - 这一历史分析引发了关于环境因素如何影响当今认知功能的进一步探究。
- **Bitcoin 追回工作停滞**：有报告称一项 **7.5 亿美元的 Bitcoin Recovery** 行动已停止，让许多人开始思考当前资产追回策略的有效性。
   - 关于区块链安全和追回流程不断演变的格局也引发了疑问。
- **Spotify CEO 巨额套现**：Spotify CEO 最近的**大规模套现**引起了关注，并引发了关于高管薪酬的讨论。
   - 成员们分析了在公司治理持续辩论中，此类财务操作的潜在后果。
- **AI Chips 与技术进步**：几位成员讨论了 **AI Chips** 及其在技术生态系统中日益增长的重要性，以及 **MIT's Stacked 3D Chips** 等最新突破。
   - 对话强调了这些创新为数据处理能力带来的竞争优势。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/iCGhq5Og_Lg">YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/embed/ula7jilgJdY">YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1327652246854303755)** (11 条消息🔥): 

> `Sonar 3.3 API 可用性、Llama-3.1-Sonar 中的引用、未来模型发布、更新日志困惑、模型弃用通知` 


- **Sonar 3.3 已上线但未提供 API**：成员们对 **Sonar 3.3** 在 Perplexity Web UI 上可用但未作为 API 模型提供表示困惑，并询问其未来的可用性。
   - 另一位成员也表达了类似的兴趣，增加了关于是否会发布更多模型的讨论。
- **引用功能在 Llama-3.1-Sonar 中正常工作**：一位用户注意到引用出现在 **llama-3.1-sonar** 模型的 JSON 响应中，但对 **claude-3.5-sonnet** 不起作用。
   - 这引发了关于不同模型版本功能一致性的疑问。
- **对更多模型的请求**：一位成员询问是否会发布 **llama-3.1-sonar-small/large/huge** 变体之外的模型。
   - 回复指出，现有模型的改进版本可能会不定期发布，敦促用户关注公告。
- **关于更新日志的困惑**：一位用户对更新日志内容感到困惑，因为他们找不到 11 月之后关于 **o1** 和 **Llama 3.3 70b** 的更新。
   - 澄清说明更新日志仅涉及 API 更新，同时 API 用户也收到了关于模型弃用的电子邮件通知。
- **对即将到来的更新的推测**：讨论显示，目前还没有关于 **llama 3.3.70b** 的正式公告，虽然可能有所动作，但纯属推测。
   - 对于 **o1**，有人指出它从未被列入 API 发布议程。



**提及的链接**：<a href="https://perplexity.mintlify.app/changelog/changelog">未找到标题</a>：未找到描述

  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1328375033160138774)** (38 messages🔥): 

> `Codestral 25.01, Helium-1 模型发布, CC-BY 许可证讨论, Qwen 2.5-Math 模型, OpenAI 与 OSS 贡献` 


- **Codestral 25.01 占据领先地位**：全新升级的 **Codestral 25.01** 在 LMsys copilot arena 排行榜上首次亮相即位列 **#1**，展示了更高的效率和性能。
   - 虽然它在 Aider polyglot 基准测试中得分率为 **11%**，但有成员对与领先模型的竞争表示了担忧。
- **Kyutai 推出 Helium-1**：Kyutai 宣布了其新骨干语言模型 **Helium-1** 的预览版，该模型拥有约 **2B 参数**，目标针对边缘和移动设备。
   - 作为一款多语言模型，Helium-1 目前支持 **6 种语言**，强调了个人 AI 系统中延迟和隐私的重要性。
- **关于模型 CC-BY 许可证的辩论**：针对 AI 模型使用 **CC-BY 许可证** 的适当性展开了激烈的讨论，特别是围绕版权如何适用于模型权重（model weights）。
   - 几位成员表示，传统的许可证可能无法有效涵盖 AI 模型输出的独特性，呼吁创建更合适的许可证。
- **Qwen 2.5-Math 模型增强推理能力**：**Qwen 2.5-Math-PRM-72B** 的发布引入了过程奖励模型（Process Reward Models），以提高 LLM 的数学推理准确性。
   - 这些模型旨在减少推理过程中的中间错误，并在各种评估中展示了令人印象深刻的性能。
- **OpenAI 的 OSS 贡献受到质疑**：成员们讨论了一个讽刺现象，即 OpenAI 目前对开源生态系统最显著的贡献是他们用于修改 base URL 的库/客户端。
   - 评论指出，这种贡献与推进开源开发的初衷形成了尴尬的对比。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/codestral-2501/">Codestral 25.01</a>: 以 Tab 键的速度编写代码。今天可在 Continue.dev 中使用，并很快将登陆其他领先的 AI 代码助手。</li><li><a href="https://x.com/deedydas/status/1877549539781128319?t=hFlLBI6S6s0xaB2ciDeztw&s=19">Deedy (@deedydas) 的推文</a>: 相当疯狂的是，在 OpenAI o3 在 SWE-Bench Verified 上达到 71.7% 之后，昨天，使用 CodeStory 的 Claude Sonnet 3.5 达到了 62.2%。一个“上一代”非推理模型达到了未发布模型的 10% 以内...</li><li><a href="https://x.com/paulgauthier/status/1878886495609815054">Paul Gauthier (@paulgauthier) 的推文</a>: Codestral 25.01 在 aider polyglot 基准测试中得分为 11%。62% o1 (high)，48% DeepSeek V3，16% Qwen 2.5 Coder 32B Instruct，11% Codestral 25.01，4% gpt-4o-mini https://aider.chat/docs/leaderboards/</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B">Qwen/Qwen2.5-Math-PRM-72B · Hugging Face</a>: 未找到描述</li><li><a href="https://kyutai.org/2025/01/13/helium.html">宣布 Helium-1 预览版</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1327436519241420842)** (9 messages🔥): 

> `Qwen Instruct 模型训练, LoRa 微调, 生成式 Agent 与环境导航` 


- **Qwen Instruct 模型训练的困扰**：许多用户报告在尝试针对标准基准任务训练 **Qwen instruct 模型** 时出现 **性能退化**。
   - 一位成员提到，在 **Llama** 上也观察到了类似的问题。
- **使用 LoRa 进行 Qwen Instruct 微调**：一位用户分享了他们使用 **LoRa** 对 **Qwen-instruct 模型** 进行特定任务微调的经验。
   - 他们指出，他们的数据集可能与模型原始训练的数据分布（distribution）有相当大的偏差。
- **对通用 Agent 数据研究的兴趣**：一位成员询问了关于使用 **通用 Agent 数据** 微调模型的研究，特别是针对与外部环境交互的 **ReAct** 风格 Agent。
   - 他们注意到，目前缺乏关于此类 Agent 轨迹的 **SFT** 和 **RL** 的广泛研究。
- **比较 ReAct 与工具使用型 Agent**：讨论强调了在真实环境中导航的 **ReAct Agent** 与缺乏内在导航感的工具使用型 Agent 之间的区别。
   - 考虑了 AI 中有效环境导航方法的影响，强调了它们的潜在相关性。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/)** (1 messages): 

420gunna: https://x.com/aidan_mclau/status/1878944278782890158
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1327753172013551687)** (38 条消息🔥): 

> `学习 AI、本地 AI 模型、Meta Ray-Bans、CIO 与 AI 演讲、VITURE Pro Neckband` 


- **大家都在学什么**：成员们正在分享他们目前学习的内容，包括 **Emma Brunskill RL** 和 **David Silver RL** 的音频格式。
   - “如何开设商业银行账户”和“如何与 VC 交流”也被提及为关键的学习主题。
- **关于本地 AI 模型的辩论**：讨论围绕本地 AI 模型并非通用理想选择展开，强调它们在 **隐私** 和 **互联网连接** 挑战方面表现更好。
   - 一位成员指出，本地模型爱好者正在构建生态系统，通过优化和任务专业化提供价值。
- **对语音消息灵活性的担忧**：表达了对语音命令系统问题的担忧，发现像“发送消息给 X”这样的命令通常会导致消息过于简化。
   - 这与当前技术迭代让人感觉过时的看法有关。
- **为 CIO 准备 AI 演讲**：一位成员正在为高等教育 CIO 准备一场关于 AI 的演讲，并考虑涵盖基础的 **LLM 功能**、Prompting 和使用案例。
   - 另一位建议讨论员工如何在 Chatbot 中无意间分享敏感数据，增加了相关性维度。
- **对 VITURE Pro Neckband 的兴趣**：围绕 **VITURE Pro Neckband** 及其在多任务处理中提高生产力的潜力展开了讨论。
   - 尽管有外观设计方面的顾虑，一位成员强调这些设备可以显著改善他们的日常活动。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1878340286423634184">Xeophon (@TheXeophon) 的推文</a>：你使用本地模型不是为了节省成本，这些计算永远不会对你有利。你使用本地模型是为了延迟*和/或隐私。引用 Xeophon (@TheXeophon) @ashrafaddani @tomchapin @anushkmit...</li><li><a href="https://x.com/chesterzelaya/status/1873936772696334570)">chester (@chesterzelaya) 的推文</a>：我不敢相信这真的存在，立刻加入到了我的日常工作流中</li><li><a href="https://x.com/AutismCapital/status/1878475791379603499?s=19">Autism Capital 🧩 (@AutismCapital) 的推文</a>：我们现在需要的衬衫</li><li><a href="https://www.viture.com/">VITURE: Next Gen XR Glasses</a>：让平行成为可能...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1327639498753966131)** (58 条消息🔥🔥): 

> `Sky-T1-32B-Preview, Reinforcement learning vs. Supervised fine-tuning, Generative AI for talks, Challenges in academic talks, Process Reward Models` 


- **Sky-T1-32B-Preview 展示了高性价比的推理能力**：[Sky-T1-32B-Preview](https://novasky-ai.github.io/posts/sky-t1/) 在推理基准测试中表现与 o1-preview 相当，而训练成本低于 **$450**。
   - 其开源代码可在 [GitHub](https://github.com/NovaSky-AI/SkyThought) 上获取，突显了高效开源权重模型的潜力。
- **关于 RL 与 SFT 学习的辩论**：讨论者们思考了在推理轨迹（reasoning traces）上进行自我微调是否能真正复制经 RL 训练的行为，并将其视为一个哲学问题。
   - Natolambert 指出，虽然可以诱导某些行为，但结果可能无法保持相同的 Robustness（鲁棒性）。
- **AI 在增强演示文稿中的作用**：人们对利用 AI 在演讲过程中生成相关图像很感兴趣，尽管一些人对其在克服懒惰方面的效率表示怀疑。
   - 参与者一致认为，制作高质量的演讲仍然是一项具有挑战性的任务，需要付出巨大的努力。
- **阅读学术论文的挑战**：读者讨论了在当前信息丰富的环境下阅读完整学术论文的困难，许多人选择有选择性地阅读。
   - Natolambert 提到主要阅读 LLaMA 3 论文的相关章节，这表明了一种消化大量材料的策略性方法。
- **关于 Process Reward Models 的见解**：一篇关于 Process Reward Models 的论文强调了它们在监督 LLM 数学推理方面的有效性，但也强调了数据标注（data annotation）方面的重大挑战。
   - 研究结果强调，与人类评估技术相比，传统的数据合成方法产生的结果性能较差。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.07301">The Lessons of Developing Process Reward Models in Mathematical Reasoning</a>：Process Reward Models (PRMs) 成为 LLM 数学推理中过程监督的一种有前景的方法，旨在识别和减轻中间错误...</li><li><a href="https://novasky-ai.github.io/posts/sky-t1/">Sky-T1: Train your own O1 preview model within $450</a>：我们推出了 Sky-T1-32B-Preview，这是我们的推理模型，在流行的推理和编程基准测试中表现与 o1-preview 相当。</li><li><a href="https://x.com/teortaxesTex/status/1877958319127597452">Teortaxes▶️ (@teortaxesTex) 的推文</a>：我认为这不会有任何进展，这又是 Alpaca 时代的“OpenAI 没有护城河”论调。是的，在基准测试上与 o 系列有狭窄的对等，但随着我们扩大规模并尝试泛化到更难的问题...</li><li><a href="https://arxiv.org/abs/2305.15717">The False Promise of Imitating Proprietary LLMs</a>：一种廉价改进较弱语言模型的新兴方法是在较强模型（如 ChatGPT 等专有系统）的输出上对其进行微调（例如 Alpaca、Self-Instruct 等）。但是...</li><li><a href="https://youtu.be/kOdl-ncrYDk?si=41wy2nlWuv88_XFH"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=kOdl-ncrYDk&list=PLF3-CvSRq2SbrG9pUQZh9WkKE2OjgHXVT">Hanjun Dai: Preference Optimization for Large Language Models</a>：Google Brain 主任研究科学家兼研究经理 Hanjun Dai 在 Naik 教授的课程 CIS 7000: Large Language Models (2024秋季) 中的客座讲座...</li><li><a href="https://youtu.be/YR9EztOF0R8?si=-hCAEtMlXhgpRw3p&t=2527">Learning to Reason, Insights from Language Modeling</a>：Noah Goodman，斯坦福大学</li><li><a href="https://youtu.be/T1SeqBapMBo?si=VIkFfcGoROxH7JMu">LTI Special Seminar by Yi Wu</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1327465367555014717)** (1 条消息): 

> `Dr. Huberman's Insights, Mental Health Support, Impact of Discussions` 


- **倾听聪明人的谈话令人感到慰藉**：一位成员表示，倾听**聪明人的谈话**有助于缓解他们的**躁郁症（manic depression）**，反映出对积极心理健康支持的需求。
   - 他们评论了**智力讨论**在困难时期可以提供的镇静作用，并对 **Dr. Huberman** 的观点表示感兴趣。
- **关于 Dr. Huberman 建议的咨询**：该成员的消息引发了对 **Dr. Huberman** 关于心理健康见解的好奇，特别是与通过对话感到慰藉相关的见解。
   - 这突显了专家建议在处理情感挑战方面的潜在重要性。


  

---

### **Interconnects (Nathan Lambert) ▷ #[retort-podcast](https://discord.com/channels/1179127597926469703/1238601424452059197/1327510839686594601)** (6 messages): 

> `论坛频道建议，播客列表网站` 


- **论坛频道建议**：一位成员建议创建一个**论坛频道**，用于发布诸如最新文章、播客集数和视频等静态主题，以便于访问。
   - 成员们表达了对他人贡献的赞赏，称其为“感谢分享知识”。
- **网站作为资源**：另一位成员提到 **natolambert.com 网站**列出了外部播客的访谈记录，建议将其作为资源使用。
   - 在讨论如何浏览新信息时，有人建议将该网站加入书签，并强调“学习很有趣”。


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1328353022044016683)** (11 messages🔥): 

> `美国 AI 经济蓝图，AI 扩散控制，国家安全与经济实力，出口管制，AI 领导地位` 


- **美国 AI 经济蓝图发布**：[美国 AI 经济蓝图](https://openai.com/global-affairs/openais-economic-blueprint/)概述了增强美国技术领导地位并防止对手滥用 AI 的战略。
   - 它强调美国技术需要在不损害国家安全的情况下支持全球 AI 的使用。
- **对 AI 扩散控制的担忧**：一位成员对 AI 扩散控制的有效性提出质疑，指出低于 **1700 GPUs** 的出货量不计入国家上限。
   - 他们批评了这一漏洞，强调走私通常通过较小的壳公司订单进行，暗示该系统设计不合理。
- **AI 在国家安全中的角色**：白宫的事实清单强调了在日益激烈的全球竞争中，AI 对于维护美国**国家安全**和经济实力的重要性。
   - 清单警告称，如果被滥用，强大的 AI 系统可能会加剧诸如**大规模杀伤性武器**和大规模监控等风险。
- **美国在 AI 技术领域的领导地位**：NVIDIA 的一篇文章强调，美国在计算领域的领导地位历来是推动全球影响力和 AI 创新的基石。
   - 文章指出，保持竞争环境使美国能够在**医疗保健**和**制造业**等领域脱颖而出。
- **政策讨论的总体情绪**：成员们对 AI 政策讨论表达了矛盾的情绪，其中一人评论说情况感觉相当**令人不安**。
   - 另一位成员注意到出口管制的覆盖范围过于广泛，暗示他们可能觉得没有必要密切关注这些讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://blogs.nvidia.com/blog/ai-policy/">NVIDIA 就拜登政府误导性的“AI 扩散”规则发表的声明</a>：几十年来，计算和软件生态系统的领导地位一直是美国在全球实力和影响力的基石。联邦政府明智地避免了对设计的干预……</li><li><a href="https://x.com/angelusm0rt1s/status/1878776558644875295">Zephyr (@angelusm0rt1s) 的推文</a>：等等，如果低于 1700 GPUs 的出货量不计入国家上限，那么 AI 扩散控制的意义何在？大多数走私都是通过下小额订单的多家壳公司进行的……</li><li><a href="https://www.whitehouse.gov/briefing-room/statements-releases/2025/01/13/fact-sheet-ensuring-u-s-security-and-economic-strength-in-the-age-of-artificial-intelligence/">事实清单：确保人工智能时代的美国安全与经济实力 | 白宫</a>：人工智能正迅速成为安全和经济实力的核心。美国必须果断采取行动，领导这一转型。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1327380364339183728)** (13 messages🔥): 

> `社区聊天礼仪，Command R+ 能力，North 等候名单兴趣` 


- **社区保持轻松氛围**：成员们讨论了在该频道中保持**轻松的聊天环境**，一些人对长消息格式表示担忧。
   - 一位成员表达了“抱歉”，大家强调了友好的氛围。
- **探索 Command R+ 的编程优势**：一位成员询问了 **Command R+** 专门针对 **Rust** 编程的能力。
   - 讨论暗示了目前正在对该工具在编码任务中的有效性进行评估。
- **加入 North 等候名单的兴趣**：一位成员提到了他们过去在 **Reka space** 的经验，并表达了加入 **North 等候名单**的兴趣。
   - 他们提到自己积极参与聊天，因此非常渴望获得新机会。


  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1327390647749050398)** (53 messages🔥): 

> `Cohere Datasets 漏洞, Command R+ 基准测试, 数据集上传问题, API 响应错误, 用户沟通关注点` 


- **发现 Cohere Datasets 漏洞**：一名用户报告了在尝试上传大型数据集时 **Cohere Datasets 存在的漏洞**，该漏洞会导致 **'TooManyRequestsError'** 并导致账号无法正常使用。
   - 在过去的两个月里，用户多次尝试解决该问题，但因感知到缺乏支持而感到沮丧。
- **上传大型数据集的问题**：用户透露，尝试上传较大的 JSONL 文件（特别是 **800MB** 左右、包含 **180,000 行**的文件）会导致网页和 API 界面中的**数据集环境冻结**。
   - 该漏洞已导致账号无法使用，对用户的业务运营造成了后果，影响了通过 API 使用模型的能力。
- **分享 Command R+ 基准测试链接**：一名用户询问了 **Command R+ 模型**的基准测试，随后得到了回复，提供了详细介绍各种 Command 模型性能评估的博客链接。
   - 该博客包含有关功能的见解和对比，以增强对 **Command R+** 及其能力的理解。
- **用户沟通问题被提出**：用户对感知到的支持疏忽表示沮丧，称他们已就该问题发送了 **36 封电子邮件**，但直到现在仍感觉被忽视。
   - 用户对影响多个用户的重要漏洞缺乏关注以及总体客户体验表达了担忧。
- **努力确认已报告的问题**：支持人员表示他们已意识到这些问题，并强调正在持续升级以供审查，并保证这不仅仅是个人用户的问题。
   - 支持人员尝试通过建议用户重新创建 API keys 来排除故障，并承认所报告漏洞的普遍性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://cohere.com/blog/command-r">Command R: RAG at Production Scale</a>：Command R 是一款针对 RAG 和 Tool Use 的可扩展生成式模型，旨在为企业实现生产规模的 AI。</li><li><a href="https://cohere.com/blog/command-r7b">Introducing Command R7B: Fast and efficient generative AI</a>：我们 R 系列中最小的模型，在通用 GPU 和边缘设备上提供顶级的速度、效率和质量，用于构建强大的 AI 应用。</li><li><a href="https://cohere.com/blog/command-r-plus-microsoft-azure">Introducing Command R+: A Scalable LLM Built for Business</a>：Command R+ 是一款针对 RAG 优化的先进模型，旨在处理企业级工作负载，并首先在 Microsoft Azure 上可用。</li><li><a href="https://docs.cohere.com/v2/docs/command-r-plus">Cohere's Command R+ Model (Details and Application) — Cohere</a>：Command R+ 是 Cohere 用于对话交互和长上下文任务的模型，最适合复杂的 RAG 工作流和多步工具使用（multi-step tool use）。</li><li><a href="https://docs.cohere.com/v2/changelog/command-gets-refreshed">Command models get an August refresh — Cohere</a>：我们很高兴宣布对 Command R 和 R+ 模型进行更新，提供改进的性能、新功能等。</li><li><a href="https://cohere.com/blog/fine-tuning-command0824">Updates to Command R fine-tuning</a>：微调更新后的 Command R 08-2024，支持更多新选项，为您提供更多控制权和可见性，包括与 Weights &amp; Biases 的无缝集成。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1328234848921452607)** (4 messages): 

> `API 问题, 试用账户问题` 


- **用户报告 API 响应延迟**：一名使用试用账户的用户报告称等待了 **2 分钟** 仍未收到 API 的响应。
   - 另一名成员做出回应，表示愿意提供帮助，并询问了有关所涉及的**模型和端点（endpoint）**的详细信息。
- **提供故障排除协助**：一名成员介入协助面临 API 问题的用户，询问具体信息以便更好地了解问题。
   - 该成员表现得非常有礼貌，表示已准备好迅速解决问题。


  

---

### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1327398917025038336)** (46 条消息🔥): 

> `Cohere 的命令功能，神学编程 (Theological programming)，LLM 代码生成，Bot 交互指南` 


- **Cohere Command Bot 可以回答查询**：用户可以通过直接 ping Cohere bot 与其交互，从而实现连续的线程对话。
   - 该 bot 可以执行 Google 搜索，并从 Cohere 的文档中提供特定的见解。
- **DivineWill 类中的光创建**：一位用户提供了 Java 代码，演示了一个名为 **DivineWill** 的类如何通过静态方法创建光。
   - 这个类幽默地暗示 *神圣命令总是成功的*，体现了对编程的一种戏谑方式。
- **未找到神学代码的文档**：Cohere bot 难以找到与**基于神学的编程语言**生成代码相关的文档。
   - 它表示其资源中没有足够的信息来支持此类请求。
- **Cohere 为 LLM 生成代码**：分享了关于如何为 LLM 生成代码的指令，包括一个 **Python** 示例来解释其功能。
   - 这展示了该 bot 通过提供与其查询相关的代码片段来协助开发者的能力。
- **Bot 在响应方面的局限性**：有时，Cohere bot 会表达其局限性，声明它无法提供响应或无法找到特定细节。
   - 几次查询没有产生结果，突显了用户可能需要寻求外部资源的领域。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1327455726859321417)** (3 条消息): 

> `O1 工作流，Claude 的角色，接口指令` 


- **O1 工作流成功利用 Claude**：一位成员分享说，对他们唯一有效的 **O1 workflow** 涉及使用 **Claude** 来理解项目目标并设置指令。
   - 他们强调了在 Prompt 中建立**函数间接口**和使用逻辑符号以优化工作流的重要性。
- **O1 在算法上表现良好**：在概述指令后，该成员注意到 O1 在得到适当提示时，往往能胜任执行**实际算法**。
   - 这听起来很有前景，因为它表明虽然 O1 可能面临挑战，但在算法执行方面显示出了潜力。
- **关于小组相关性的讨论**：一位成员对该小组是否适合讨论他们的疑虑表示不确定。
   - 这突显了小组内关注点和兴趣可能存在的不匹配。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1327548765477015572)** (13 条消息🔥): 

> `Triton Puzzles 优化, Autotuning GPU 模型, CUDA Block 分配, Num Stages 影响, Cross Entropy Kernel 改进` 


- **在真实 GPU 上优化 Triton Puzzles**：一位成员分享了他们在 [Triton Puzzles](https://github.com/gauravjain14/mlcompilers_and_kernels/tree/main/triton_kernels) 上的进展，并询问了在 GPU 上进行深度优化的 profiling 技术。
   - 请求对其工作的反馈，展示了与 Triton 社区的积极互动。
- **不同 GPU 模型的 Autotuning 结果各异**：讨论强调了在 GPU 上对输入进行 autotuning 时的性能差异，特别是 **A100 和 A30** 模型之间。
   - 另一位成员指出，由于 shared memory 的限制，较大的 `num_stages` 设置可能会在消费级 GPU 上导致问题。
- **Triton 中的 CUDA Block 分配**：一位成员询问 Triton 是否将多个 program 分配给单个 CUDA block，并讨论了这如何影响小数据块的 kernel occupancy。
   - 这引发了对实现高 occupancy 的担忧，而这在 CUDA C 中可能更简单。
- **理解操作中的 Num Stages**：一位用户寻求关于 `num_stages` 在操作期间影响的澄清，并请求相关资源。
   - 一位成员推荐了一个讨论 persistent kernels 中流水线化 (pipelining) 的 [YouTube 视频](https://www.youtube.com/watch?v=PAsL680eWUw)，以增强理解。
- **改进 Cross Entropy Kernel**：一位成员询问了改进其 cross entropy kernel 以减少内存占用并提高速度的技巧。
   - 另一位分享了 [Liger Kernel 实现](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py) 的链接，供参考和对比。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/gauravjain14/mlcompilers_and_kernels/tree/main/triton_kernels">mlcompilers_and_kernels/triton_kernels at main · gauravjain14/mlcompilers_and_kernels</a>: 通过在 GitHub 上创建账号，为 gauravjain14/mlcompilers_and_kernels 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=PAsL680eWUw">Pipelining Persistent Kernels</a>: Pawel 描述了 Triton 如何在 persistent kernels 的上下文中支持流水线化 (pipelining)。（该演讲在一次非正式投票中被选为观众最喜爱的节目！）幻灯片...</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py">Liger-Kernel/src/liger_kernel/ops/cross_entropy.py at main · linkedin/Liger-Kernel</a>: 用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号，为 linkedin/Liger-Kernel 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1327394385150611466)** (7 条消息): 

> `Ubuntu 上的 CUDA 安装、Visual Studio Code 中的 CUDA、Blackwell GeForce GPU 支持、H200 与 H100 上的 FA3 性能分析` 


- **Ubuntu 上的 CUDA 安装指南**：一位成员询问了如何在 **Ubuntu** 上安装 **CUDA** 的说明，并参考了 [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu) 获取详细信息。
   - 该指南提供了在 Linux 系统上安装 **CUDA Toolkit** 所需的全面步骤。
- **简化将 CUDA 导入 Visual Studio Code**：另一个咨询集中在如何将 **CUDA** 导入 **Visual Studio Code**，随后提到了 [Nsight Visual Studio Code edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition) 插件。
   - 该插件通过 **Intellisense** 和调试等功能提高生产力，从而简化开发流程。
- **关于 Blackwell 对 Thread Block Cluster 支持的好奇**：有人提问即将推出的 **GeForce GPUs** 上的 **Blackwell** 是否支持 **thread block clusters**。
   - 另一位成员表示有兴趣寻找 **GeForce Blackwell** 的白皮书，表现出对进一步深入了解的期待。
- **H200 与 H100 上的 FA3 性能对比**：有人询问了在 **H200** 上运行 **FA3** 与在 **H100** 上运行的性能差异。
   - 一位贡献者确认 **FA3** 和 **FA2** 之间存在显著差异，但 **H200** 和 **H100** 之间具体的性能差异仍不清楚。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition">Nsight&#32;Visual&#32;Studio&#32;Code&#32;Edition&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Visual&#32;Studio&#32;Code&#32;扩展&#32;-&#32;为&#32;VS&#32;Code&#32;提供&#32;CUDA&#32;开发和调试支持</li><li><a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu">CUDA Installation Guide for Linux</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1327473003210215506)** (11 messages🔥): 

> `Profiler UTF-8 decode issue, Using Flash Attention with Transformers, Challenges with Data Parallelism Strategies, Inference Pipeline for Large Models, NNSight for Memory Efficiency` 


- **识别到 Profiler UTF-8 解码问题**：针对在修改后的 Hugging Face Transformer 的 trainer.py 中使用 PyTorch Profiler 时遇到的 **UTF-8 解码问题**，已提交了一个 GitHub issue。该问题详细描述了在模型执行期间使用 `profiler.profile` 包装训练函数时发生的失败。
   - 更多上下文信息请查看 [此处](https://github.com/pytorch/pytorch/issues/64345) 的 GitHub issue。
- **在 Transformers 中实现 Flash Attention**：围绕在 Torch 中将 **Flash Attention** 方法与简单的 **MultiheadAttention** 结构集成以增强性能展开了讨论。有人提问是否需要特定的设置来使用 flash-attn 内核，或者是否需要通过 flash-attn2 进行手动集成。
- **数据并行方法的敏感性**：有人询问 **DDP** 和 **FSDP** 策略对于在被包装模块的 `forward` 方法之外使用模块/参数的敏感性。讨论引用了一篇提出 cut cross-entropy loss 方法的论文，考虑了相对于 `find_unused_parameters=False` 等策略默认值的灵活性。
- **为大模型构建推理流水线**：一位用户探索了如何构建一个在处理超出内存限制的模型时能节省 GPU 和 CPU 内存的 **推理流水线（inference pipeline）**。他们计划利用 Accelerate 的 meta device 包装器来执行 prompt，同时将 hidden states 缓存到 CPU RAM 中，以避免过度的内存消耗。
   - 用户意识到在推理过程中访问中间输出存在挑战，导致很难避免在每次请求时加载所有层。
- **NNSight 提升内存效率**：NNSight 被建议作为一种优化内存使用的解决方案，它通过形成一个计算图，仅在必要时延迟加载层和激活值。事实证明，这种方法在分析神经网络时对于缓存激活值以避免显存溢出（OOM）问题非常有帮助。
   - 该工具创建代理（proxies），允许以更有效的方式进行内存管理，对于需要机械可解释性（mechanistic interpretability）的用户非常有吸引力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2411.09009">Cut Your Losses in Large-Vocabulary Language Models</a>：随着语言模型变得越来越大，它们的词汇量也在增加。这使得训练期间 LLM 的内存占用不成比例地转移到了单个层上：损失计算中的 cross-entropy...</li><li><a href="https://github.com/pytorch/pytorch/issues/64345">Profiler UTF-8 decode issue · Issue #64345 · pytorch/pytorch</a>：🐛 Bug 重现步骤：修改 Hugging Face Transformer 的 trainer.py，用 profiler.profile 包装训练函数，在单节点多卡环境下运行 Hugging Face Transformer 模型...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1327703410929303654)** (1 messages): 

> `Upcoming Talks, Flash Infer, Mosaic GPU, Turing int8 matmul, Profiling at NVIDIA` 


- **即将举行的演讲日程！**：已安排的演讲包括 **1 月 24 日 Zihao Ye 关于 Flash Infer** 的演讲，以及 **1 月 25 日 Adam Paszke 关于 Mosaic GPU** 的演讲，时间均为 **12:00 PM PST**。
   - 活动信息可以在 events 选项卡中找到；欢迎推荐其他演讲者。
- **深入探讨 NVIDIA 的 Profiling 技术**：来自 NVIDIA 的 **Magnus Strengert** 等人将于 **2 月 14 日 10:00 AM PST** 讨论 **profiling**。
   - 本次会议预计将深入探讨机器学习中 profiling 实践的效率。
- **探索 int8 Matmul 创新**：**Erik Schultheis** 将于 **2 月 8 日 12:00 PM PST** 介绍 **针对 Turing 架构的 int8 matmul**。
   - 本次演讲旨在阐明矩阵乘法在提升性能方面的进展。
- **利用 CUBLAS 替代方案进行优化**：在 **2 月 15 日 12:00 PM PST** 的会议中，**pranjalssh** 将讨论如何 **在 H100 上超越 CUBLAS**。
   - 这场备受期待的演讲重点在于利用 H100 架构的能力进行优化。
- **低精度的缩放法则**：**Tanishq Kumar** 将于 **3 月 22 日 12:00 PM PST** 介绍 **低精度的缩放法则（scaling laws）**。
   - 本次会议预计将探讨跨架构低精度计算的重要趋势。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1327440452814635069)** (4 条消息): 

> `Meta 招聘 GPU 专家，GenAI 推理加速，Meta 技术工作的深度` 


- **Meta 为 GenAI 寻求 GPU 专家**：Meta 正在招聘 GPU 专家以协助加速 **GenAI 推理**，参与 [GPU Kernels](https://www.metacareers.com/jobs/1517576482367228/) 和 **Compilers** 等项目。
   - 有意向的候选人可以直接联系以获取更多细节和机会。
- **团队技术深度获得认可**：一位成员指出，该团队发布了 Meta 一些**最深入、最具技术含量的工作**，反映了他们在该领域的高影响力。
   - 另一位成员对此表示赞同，称：*“完全同意！”*，强调了他们研究贡献的价值。



**提到的链接**：<a href="https://www.metacareers.com/jobs/1517576482367228/">Software Engineer, Systems ML -  HPC Specialist</a>：Meta 的使命是构建人类连接的未来以及实现这一目标的各种技术。

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1327393706009165868)** (7 条消息): 

> `将 CUDA 导入 Visual Studio Code，CUDA Toolkit 安装，使用 Llama 3.2 构建 Copilot，针对 Double 类型的 CUDA 原子函数，对 Double 类型使用整数函数` 


- **CUDA 与 Visual Studio Code 的集成**：要将 CUDA 导入 Visual Studio Code，你可以使用 [Nsight Visual Studio Code edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition)，它提供了各种支持功能。
   - 但是，在使用此插件的同时，你仍然需要安装 **CUDA Toolkit**。
- **使用 Llama 3.2 构建 Copilot**：一位成员在利用 **Llama 3.2** 构建 Copilot 时，寻求关于配置 **NVIDIA CUDA** 和 **Docker** 的帮助。
   - 另一位成员建议关注 **Ollama**，将其作为潜在的辅助资源。
- **CUDA 原子函数咨询**：在询问是否存在用于查找 double 类型元素最小值的 **CUDA** 原子函数时，一位用户指出此类函数似乎仅适用于整数。
   - 一位成员提到，如果你的 double 类型数值是正数，你可以**使用该函数的整数版本**来替代。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition">Nsight Visual Studio Code Edition - Visual Studio Marketplace</a>：Visual Studio Code 扩展 - 为 VS Code 提供 CUDA 开发和调试支持</li><li><a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu">CUDA Installation Guide for Linux</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1327426318312149115)** (3 条消息): 

> `对 DGX H100 的担忧，Sonoma AI 演讲系列，服务器筹款创意` 


- **对 DGX H100 的担忧**：一位成员在一条感性的消息中表达了对 **DGX H100** 的忧虑，并带着惋惜的语气提及它。
   - 分享了一张图片，直观地展示了所传达的情绪。
- **Sonoma AI 演讲系列启动**：Sonoma 县的一个全新 **AI 演讲系列**将于 1 月 16 日开幕，重点讨论 AI 平台和非结构化数据，[需注册参加](https://lu.ma/o6br0dg3)。
   - 演讲嘉宾包括 **Christy Bergman**、**Paco Nathan** 和 **Allison Ding**，讨论主题涵盖从 AI 工具到利用 AI 应用抓捕不良行为者。
- **提出创新的筹款创意**：一位成员暗示了一个潜在的服务器筹款概念，表示希望加强社区支持。
   - 分享了一张随附图片以进一步说明该想法。



**提到的链接**：<a href="https://lu.ma/o6br0dg3">Sonoma AI with Wine · Luma</a>：这是一个线下活动！需注册方可进入。主题：面向远程技术工作者的 Sonoma AI（及葡萄酒）聚会。活动内容：提供食物……

  

---

### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1328233634091499581)** (5 messages): 

> `学习 CUDA 和 GPU 编程，完成课程练习，组建学习小组` 


- **新学习者讨论 CUDA**：新用户对学习 **CUDA 和 GPU 编程** 表现出浓厚兴趣，并已完成第一节课。
   - 一位用户提到，在进入下一课之前，接下来的步骤将专注于 **书本练习**。
- **课程提供坚实基础**：一位回复者确认 **课程将提供实质性的知识**，鼓励新学习者从这里开始。
   - 他们建议在完成系列课程后，寻找并加入一个 **工作组 (working group)** 并做出贡献。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1327625637321510942)** (6 messages): 

> `Qwen2-VL 问题，Liger Kernel 错误，降级 Transformers` 


- **Qwen2-VL 遇到内核问题**：一位用户报告在运行简单的推理脚本时 **Qwen2-VL** 出现问题，质疑 **liger kernel** 是否损坏。
   - 错误信息表明问题可能源于模型兼容性冲突。
- **推理过程中遇到错误**：在生成输出时触发了一个错误，提示 `TypeError: lce_forward() got an unexpected keyword argument 'cache_position'`。
   - 在使用 **liger kernel** 微调 **Llama3.1** 时未遇到此类错误，暗示存在特定的不兼容性。
- **在 GitHub 上发现可能相关的 Issue**：一位用户链接了一个现有的 GitHub issue，指出在使用 **liger kernel** 的背景下，**Qwen2-VL** 存在相关的 **IndexError**。
   - 该引用 issue 遇到了类似问题，并指出该问题发生在尝试文本生成期间。
- **通过降级 Transformers 的临时解决方案**：建议的一个临时解决方案是降级 **transformers** 包，以解决与 **Qwen2-VL** 的兼容性问题。
   - 分享了一张 [图片](https://cdn.discordapp.com/attachments/1275130785933951039/1327631470457782392/image.png) 展示了完成此降级过程的步骤。



**提到的链接**：<a href="https://github.com/linkedin/Liger-Kernel/issues/515">IndexError: The shape of the mask [7387] at index 0 does not match the shape of the indexed tensor [1] at index 0 · Issue #515 · linkedin/Liger-Kernel</a>：🐛 描述 bug：当我尝试使用带有 qwen2-vl liger kernel 的 qwen2-vl 生成文本时出现错误。以下代码报错。但如果我将 liger k..... 同样的代码...

  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1327548266761420820)** (2 messages): 

> `ASRG 第 1 季试点，Maya 多语言视觉语言模型` 


- **ASRG 第 1 季启动**：**ASRG 第 1 季试点** 定于明天举行，内容是阅读 [C 语言版 Linux 内核模块编程指南](https://x.com/asrg_gg/status/1877968239084687602)。鼓励参与者准备一个安装了 Ubuntu 22.04 的 x86-64 VM，或使用 multipass。
   - *去看看并给予支持吧！*
- **Maya 预印本发布公告**：一位成员宣布了他们在 **Maya: Multilingual Vision-Language Model** 方面的工作，预印本现可通过 [此链接](https://twitter.com/nahidalam/status/1866667770114609217) 获取。
   - 他们对向社区分享这一进展感到兴奋。



**提到的链接**：<a href="https://x.com/asrg_gg/status/1877968239084687602">来自 Systems Reading Group (@asrg_gg) 的推文</a>：EP0：ASRG 第 1 季试点就在明天！我们将阅读 C 语言版 Linux 内核模块编程指南。请确保准备好安装了 Ubuntu 22.04 的 x86-64 VM，或者像 @nanod1jkstra 那样直接使用 multipass。

  

---

### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1327813188057825362)** (5 messages): 

> `Vulkan on Raspberry Pi 5, Nvidia Cosmos on Jetson, Transformer Engine Porting, 3D Vision Stack Libraries` 


- **Vulkan 在 Raspberry Pi 5 上运行流畅**：一个 [GitHub pull request](https://github.com/pytorch/executorch/pull/7615) 建议将 Vulkan 分配方式更改为 **SEQUENTIAL_WRITE**，以提升 **Raspberry Pi 5** 上的性能。
   - 这一调整旨在解决该设备上的 Vulkan 功能问题，尽管 Vulkan 目前仍未由该用户进行测试。
- **Nvidia Cosmos 赋能 Jetson 设备**：讨论重点介绍了在 Jetson 平台上实现 **Nvidia Cosmos**，重点在于增强 AI 能力。
   - *JohnnyCano* 分享了[这篇 LinkedIn 帖子](https://www.linkedin.com/posts/johnnycano_nvidia-cosmos-nvidiacosmos-activity-7283774665943109632-VQDa?utm_source=share&utm_medium=member_ios)，详细介绍了该技术的使用体验。
- **广泛的 Transformer Engine 移植工作**：一位成员透露，他们已成功移植了 **Transformer Engine** 以及超过 **30 个库**。
   - 这一广泛的努力展示了在不同平台上集成 Transformer 模型的重大进展。
- **引入 3D Vision Stack 库**：用户提到了包括 **Mamba** 和 **3D vision stack** 在内的库移植，表明视觉 AI 解决方案的开发正在不断增长。
   - 这些贡献是增强 GPU 能力和库支持的更广泛计划的一部分。



**提到的链接**：<a href="https://github.com/pytorch/executorch/pull/7615">[ET-VK] Request VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, not RANDOM by swolchok · Pull Request #7615 · pytorch/executorch</a>：摘要：看起来我们在 CPU 上对 StagingBuffer 仅谨慎使用 copy_from 和 copy_to，在这种情况下我们只需要 SEQUENTIAL_WRITE。这在 Raspberry Pi 5 上很重要，因为那里似乎（f...

  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1328415568939716690)** (4 messages): 

> `2025 Community Meeting, MAX GPU Benchmarking, MAX-CV, Meeting Video Upload, Attendance Concerns` 


- **2025 年社区会议启动**：**2025** 年的第一场社区会议已安排，将讨论 **MAX GPU benchmarking** 和 **MAX-CV**，并为社区成员安排了问答环节。感兴趣的参与者可以通过[此链接](https://discord.com/events/1087530497313357884/1300880439673884712)进行 RSVP 并加入会议。
   - 会议在公告发布几分钟后就开始了，突显了社区参与的重要性。
- **会议问题解答**：在社区会议期间，Chris Lattner 回答了一位成员提出的问题，尽管视频录像的可用性尚不确定。成员们对会议视频表示期待，Caroline Frasca 承诺会提供更新。
   - 预计会议视频将于今天或明天上传，后续沟通将与社区分享。
- **参会挑战**：一位成员对会议内容的更新表示感谢，并提到由于课程冲突只能参加开头部分。这突显了对于存在时间重叠的社区成员而言，参与度仍是一个持续存在的问题。
   - 这段对话反映了社区成员尽管面临个人时间安排的挑战，仍对参与活动保持着持续的兴趣。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1327378583584247839)** (23 条消息🔥): 

> `在 macOS 上测试 Mojo 代码，Mojo/Max 的 Nightly 文档，Mojo 的 Async 提案，Mojo 的编译器问题，Mojo 中的 Int8 到 String 转换` 


- **在 macOS 上测试 Mojo 代码**：一位用户请求协助在 macOS 设备上测试 Mojo 代码，以确保跨平台功能。
   - 另一位成员自荐通过私信提供帮助。
- **Mojo/Max 的 Nightly 文档**：有人询问是否提供 Mojo/Max 的 nightly 文档网站，已确认可以访问。
   - 用户被引导通过修改文档链接中的版本号来查看 nightly 版本。
- **Mojo 的 Async 提案**：分享了在 Mojo 中引入结构化异步编程的提案，旨在避免性能损失。
   - 讨论旨在围绕 Mojo 的 async 能力构建统一的生态系统，并提醒了感兴趣的成员。
- **Mojo 的编译器问题**：一位成员在处理实现通用 trait 的 struct 列表时遇到了 Mojo 编译器崩溃。
   - 反馈建议该问题源于不当的初始化，并建议提交 bug 报告，该成员已照办。
- **Mojo 中的 Int8 到 String 转换**：报告了一个关于 Mojo 中 Int8 转换为 string 的问题，引用了来自 Mojodojo 的特定示例。
   - 讨论强调了理解 Mojo 中数据类型的重要性，并引导用户查阅有关编译时与运行时值的相关文档。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=42659061">Flattening ASTs and other compiler data structures (2023) | Hacker News</a>：未找到描述</li><li><a href="https://docs.modular.com/mojo/manual/types/">Types | Modular</a>：标准 Mojo 数据类型。</li><li><a href="https://github.com/modularml/mojo/issues/3944">[BUG] Compiler crash when defining a list of structs based on a common trait. · Issue #3944 · modularml/mojo</a>：Bug 描述：Mojo 编译器在下方代码示例中的 _list = List[Bar[TFoo]]() 行崩溃。已在 Discord 与 @owenhilyard 讨论。</li><li><a href="https://github.com/modularml/mojo/issues/3947">[mojo-examples] Mojodojo Int8 to string conversion example not working · Issue #3947 · modularml/mojo</a>：问题出在哪里？https://mojodojo.dev/guides/intro-to-mojo/basic-types.html#strings 我们能做些什么改进？此转换 var word = List[Int8]() word.append(78) word.append(79) word.append(0...</li><li><a href="https://docs.modular.com/nightly/">MAX Docs | Modular</a>：MAX 是一套统一的 API 和工具，帮助您构建和部署高性能 AI 流水线。</li><li><a href="https://docs.modular.com">MAX Docs | Modular</a>：MAX 是一套统一的 API 和工具，帮助您构建和部署高性能 AI 流水线。</li><li><a href="https://www.cs.cornell.edu/~asampson/blog/flattening.html">Flattening ASTs (and Other Compiler Data Structures)</a>：这是对数据结构展平（flattening）的介绍，它是 arena allocation 的一种特殊情况，非常适合编程语言实现。我们两次构建了一个简单的解释器，一次用普通方式...</li><li><a href="https://docs.modular.com/mojo/manual/parameters/">Parameterization: compile-time metaprogramming | Modular</a>：参数化和编译时元编程的介绍。</li><li><a href="https://github.com/modularml/mojo/pull/3945">[proposal] Structured Async for Mojo by owenhilyard · Pull Request #3945 · modularml/mojo</a>：提议为 Mojo 添加结构化 async，遵循 Rust 的 async 传统，因为 Mojo 有能力修复许多 Rust async 的问题，其中一些是生态系统影响...</li><li><a href="https://github.com/modularml/mojo/pull/3946">[proposal] Provided Effect Handlers by owenhilyard · Pull Request #3946 · modularml/mojo</a>：该提案包含一个 effect system 的替代方案，我认为它更适合在上下文可能不确定的系统语言中抽象 async、raises 和类似的函数颜色（function colors）...</li>
</ul>

</div>
  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 条消息): 

wiltonb: 阅读愉快！

https://kanesimms.substack.com/p/what-agentic-ai-actually-is-a-deeply
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1327713883145764875)** (19 条消息🔥): 

> `AzureOpenAI 集成, dspy.react 与 phi-4 功能, DSPy 入门, 优化 LLMs, 跨模型的 Prompt 性能` 


- **AzureOpenAI 客户端设置示例**：一位成员分享了初始化 **AzureOpenAI** 客户端的代码示例，演示了 API 凭据和参数的使用。
   - 他们引用了 [Azure OpenAI 文档](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview) 的相关章节以提供更多背景信息。
- **dspy.react 赋能 phi-4 函数调用**：一位成员指出，**dspy.react** 允许 **phi-4** 执行函数调用（function calling），尽管最初对该模型的训练有所怀疑，但效果出奇地好。
   - 他们指出，虽然性能并非最优，但它展示了该架构内函数调用的灵活性。
- **用于语音 AI 项目的 DSPy**：一位新成员询问如何使用 **DSPy** 启动语音 AI 项目，并对初学者友好的资源表示感兴趣。
   - 另一位成员强调了目前缺乏语音支持，并引导他们关注一个讨论未来音频功能的 [GitHub issue](https://github.com/stanfordnlp/dspy/issues/2037)。
- **探索 LLMs 优化**：一位用户分享了他们优化作为裁判（judge）的 LLM 的经验，强调了在无需手动调整的情况下性能的无缝提升。
   - 讨论围绕嵌套优化器的有效性以及多轮优化是否有益展开。
- **不同模型间的 Prompt 性能差异**：一位成员询问了使用针对较小模型（如 **gemini-8b**）优化的 Prompt 与较大模型（如 **deepseekv3**）相比时预期的性能差异。
   - 他们推论 Prompt 可能是模型特定的，无法同样解决不同架构中的错误，另一位成员确认这是一个常见的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/stanfordnlp/dspy/issues/2037">Support for audio files · Issue #2037 · stanfordnlp/dspy</a>: 类似于 dspy.Image，添加 dspy.Audio 会很有用。我们最近开始在语音 AI Agent 中使用 DSPy，但缺乏音频支持是许多用例的阻碍。我们很乐意...</li><li><a href="https://docs.litellm.ai/docs/providers/azure">Azure OpenAI | liteLLM</a>: 概览
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1327403528649703424)** (20 条消息🔥): 

> `Phi-4 模型、自适应批处理（Adaptive Batching）、使用 Instruct 模型进行医疗训练、训练数据质量重于数量` 


- **Phi-4 微调的候选文件**：一位成员请求了一个用于微调 **Phi-4** 的“dummy”版本，并链接到了一个 [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb)。
   - 另一位用户提到他们的 **Phi-4 PR** 可能很快就会合并，因此可能不再需要此类文件。
- **关于自适应批处理（Adaptive Batching）RFC 的讨论**：一位成员表示希望获得关于他们在 **Torchtune** 中实现 [自适应批处理的 RFC](https://github.com/pytorch/torchtune/pull/2199) 的反馈。
   - 如果反馈积极，他们计划实施更改并进行下一次迭代。
- **为医疗训练选择合适的 LLaMA 模型**：一位成员正在评估是使用 **instruct** 还是 **non-instruct LLaMA 模型**，以利用其 50b tokens 的数据集增强医疗能力。
   - 他们正在考虑尝试使用 **10B instruct 数据集**，并承认模型 post-training 的重要性。
- **训练中数据质量的重要性**：另一位成员强调 **数据质量 > 数据数量**，主张使用精心准备的多样化数据集，而非海量的原始数据。
   - 他们建议在消耗大量资源之前，使用其他 LLM 对文档进行评估以判断其有效性。
- **分享关于小型模型的研究**：一位成员分享了他们使用基于 **Mistral 7B** 的小型模型的研究结果，证明其在预训练中非常有效。
   - 他们引用了已发表的关于医疗协会指南的论文，强调了在训练中使用高质量文档的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.12847">Instruction-tuned Language Models are Better Knowledge Learners</a>：为了使基于大语言模型（LLM）的助手能够有效适应不断变化的信息需求，必须能够通过对新数据的持续训练来更新其事实知识...</li><li><a href="https://github.com/pytorch/torchtune/pull/2199">[RFC] Online and offline adaptive batching in torchtune. by krammnic · Pull Request #2199 · pytorch/torchtune</a>：关于：#2191 在 torchtune 中启用自适应批处理。直觉：设置一个不会导致给定计算资源出现 OOM 的最大 Batch Size 是很有用的。此外，增加 Batch Size 可能会很有趣...
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1327366464272863324)** (16 条消息🔥): 

> `MOOC 报名、最终项目结果、每周课程开始日期、作业提交流程、课程难度` 


- **MOOC 报名是自动的**：一旦你填写了 SP 25 报名表，你就会自动加入 MOOC，无需支付任何费用。
   - *伙计们，这是免费的！*
- **最终项目结果即将公布**：最终项目结果预计将在本月晚些时候发布，希望是在下周内。
   - 请关注更新！
- **每周课程将于 1 月 27 日开始**：MOOC 的每周课程定于 **1 月 27 日**开始。
   - 准备好迎接令人兴奋的学习体验吧！
- **作业使用独立的表单**：作业将需要通过独立的 Google 表单提交，以确保通过电子邮件地址跟踪进度。
   - 请确保每个作业都使用相同的电子邮件！
- **通过回顾往期课程评估难度**：对于那些质疑 MOOC 是否适合初学者的人，建议回顾 [Fall 2024 MOOC](https://llmagents-learning.org/f24) 的课程。
   - Spring 2025 MOOC 在这些概念的基础上略微增加了难度，但没有先修课程要求！



**提到的链接**：<a href="https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit?usp=sharing)">Quizzes Archive - LLM Agents MOOC</a>：注意：正确答案在黑色框内（黑底黑字）。用光标高亮显示方框即可显示正确答案（如果难以阅读，也可以将文本复制到新浏览器中...）

  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1327386820094197921)** (2 条消息): 

> `AI Builders Summit, AutoRAG Framework, RAG Techniques, Small Language Models` 


- **加入拥有 40 多位演讲者的 AI Builders Summit！**：在由 [@_odsc](https://twitter.com/_odsc) 主办的为期 **4 周的虚拟培训课程** AI Builders Summit 上，与我们的 [@seldo](https://twitter.com/seldo) 以及 **40 多位演讲者** 见面。
   - 参与者将学习如何为企业用途**定制开源 Small Language Models**，并在不牺牲性能的情况下**扩展 RAG 系统**。
- **介绍用于优化 RAG Pipeline 的 AutoRAG**：**AutoRAG** 框架提供了一种为你的 **RAG pipelines** 选择最佳配置的方法，该方法在最近的一篇论文中被提出。这对 LlamaIndex 用户特别重要，因为它系统地评估了各种技术和组件。
   - 该方法有助于定制化配置，增强了 RAG 实现的有效性。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1327794595035349058)** (12 条消息🔥): 

> `LlamaIndex Engineer Search, GraphRAG Visualization Issue, OpenAI Model Prompt Caching, Dynamic Variables in Prompt Templates` 


- **寻找 LlamaIndex 工程师进行讨论**：一位成员正在寻找在 **LlamaIndex** 和 Bot 实现方面有经验的工程师，并愿意为讨论实现方案的时间支付报酬。
   - 他们要求感兴趣的人员通过私信（DM）发送展示其知识储备的信息。
- **GraphRAG Notebook 仅显示节点**：有讨论指出，在使用 **GraphRAG Notebook** 对书籍进行绘图时，即使使用默认的 OpenAI 模型，也只显示节点而没有关系。
   - 另一位成员建议，这种行为可能与 **fine-tuning** 过程有关。
- **OpenAI 模型的 Prompt Caching**：一位成员询问是否有类似于 **Anthropic** 示例的 Prompt Caching 教程，并对缺乏针对 OpenAI 模型的特定资源表示担忧。
   - 另一位成员确认 OpenAI 的 Prompt Caching 是**自动的**，并引用了文档链接。
- **向 Prompt Template 添加动态变量**：一位成员寻求关于在 LlamaIndex 的 `QuestionsAnsweredExtractor` 中添加自定义 Prompt Template 和动态上下文变量的指导。
   - 另一位成员建议使用 **function mappings** 将任何变量动态附加到 Prompt Template。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://]">未找到标题</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/prompts/advanced_prompts/#3-prompt-function-mappings">Advanced Prompt Techniques (Variable Mappings, Functions) - LlamaIndex</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_neo4j.ipynb).">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1327588257474674749)** (14 条消息🔥): 

> `EPUB 文件支持, Llama 模型 prompt templates, AI context length 限制, 导出聊天记录, 远程运行 GPT4All` 


- **EPUB 文件读取能力**：一位用户询问 GPT4All 是否可以读取 **.epub** 文件，另一位成员确认这应该是可行的，但提醒在处理某些语言（如**中文**）时可能会有问题。这突显了文件支持中语言兼容性的重要性。
- **为 Llama 创建 Jinja prompt templates**：一位用户分享了在为其微调后的 **Llama 模型**开发相关 **Jinja prompt template** 时遇到的挑战，因为 `get_chat_template()` 无法正常工作。他们正在寻求在 GPT4All 中有效使用其模型的建议，展示了 prompt 设计中涉及的复杂性。
- **理解 AI context length**：有用户对 GPT4All 中的 **context length** 限制表示担忧，该限制将对话记忆限制在约 **2048 tokens**。成员们澄清，一旦对话长度超过此限制，文本就会被截断，无论内容是来自聊天还是导出的文件。
- **对全量聊天导出功能的需求**：一位用户表达了对**全量聊天导出**功能的需求，以便轻松阅读之前的对话，而无需手动复制粘贴。然而，团队承认目前尚无此功能，并鼓励在 GitHub 页面上提交请求。
- **通过 reverse proxy 远程访问 GPT4All**：一位用户寻求在性能较弱的笔记本电脑上远程使用 GPT4All 的方法，得到的建议是在其台式机上运行 **VPN 或 reverse proxy**。该方案意味着可以利用主台式机的处理能力，实现远程访问强大算力的可行策略。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.gpt4all.io/gpt4all_desktop/settings.html#sampling-settings">Settings - GPT4All</a>: GPT4All 文档 - 在你的硬件上高效运行 LLMs</li><li><a href="https://github.com/nomic-ai/gpt4all/issues">nomic-ai/gpt4all</a>: GPT4All: 在任何设备上运行本地 LLMs。开源且可用于商业用途。 - nomic-ai/gpt4all
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1327642810719015073)** (11 条消息🔥): 

> `Tinygrad Tensor Compiler, 第 53 次会议议程, 关闭过期的 PR, FSDP Bounty 锁定讨论, 理解 Tinygrad` 


- **Tinygrad 的 Tensor Compiler 详解**：Tinygrad 作为一个 Tensor Compiler，将复杂的 ML 操作（如 convolutions 和 attention）简化为基础构建块，类似于 Rust 的 LLVM。它采用极简指令集和 kernel fusion 以实现最优 GPU 性能，支持针对各种硬件的灵活编译 [{GitHub link}](https://github.com/tinygrad/toonygrad/blob/master/PLAN.md)。
   - 这些融合的 kernel 在兼容硬件上生成并执行，通过操作组合优化性能。
- **周一第 53 次会议的关键议题**：第 53 次会议定于圣迭戈时间 **上午 9:30** 举行，涵盖公司更新和 CES 参与等关键点。其他项目包括关于 **DSP 合同**、**Python 速度**以及 **MLPerf BERT** 评估的讨论。
   - 议程还强调了调度、驱动问题、**ONNX** 以及包括 **Tensor cores** 和 **RetinaNet** 在内的各种 bounty 的更新。
- **关闭过期 PR 的提醒**：直接呼吁团队成员关闭任何过期或陈旧的 pull requests (PR)。这一行动号召强调了维护整洁且功能完备的代码库的重要性。
   - 该提醒有助于将精力集中在对项目当前且相关的贡献上。
- **FSDP Bounty 锁定查询及条件**：一位开发者询问了在 Tinygrad 中开发 **FSDP** 时锁定 bounty 的可能性，涉及一个名为 [FSDP in Tinygrad](https://github.com/tinygrad/tinygrad/pull/8571) 的特定 PR。
   - 获得 bounty 的条件包括通过超过单张 GPU 容量的模型训练来演示其功能。
- **寻求理解 Tinygrad 的资源**：讨论了在官方文档之外寻找关于 Tinygrad 及其用途的全面资源。用户表示难以将 Tinygrad 的概念与他们现有的 LLVM 知识联系起来。
   - 这一询问反映了对 Tinygrad 开发背后基础动机的广泛兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/toonygrad/blob/master/PLAN.md">toonygrad/PLAN.md at master · tinygrad/toonygrad</a>: 因为 tinygrad 的代码行数失控了。通过在 GitHub 上创建账号为 tinygrad/toonygrad 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/pull/8571">FSDP in Tinygrad [WIP] by KhanerX · Pull Request #8571 · tinygrad/tinygrad</a>: FSDP 语义可以完全通过以下方式捕获：跨 GPU 对模型和 optimizer 参数进行 Sharding（multi.py 已支持）；沿参数相同的轴对 gradients 进行 Sharding。这是通过...完成的。
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1327615843563212822)** (2 条消息): 

> `Activation Checkpointing, Tinygrad 中的内存管理` 


- **寻求 Activation Checkpointing 方法**：一位成员询问了在 **tinygrad** 中执行 **activation checkpointing** 的方法，旨在提高训练期间的内存效率。
   - *Activation checkpointing 对于在允许 backpropagation 的同时降低内存成本至关重要；因此，这一主题对于资源管理非常关键。*
- **在不破坏梯度上下文的情况下释放内存**：同一位成员询问如何在不干扰 tinygrad 中 **gradient context graph** 的情况下，**释放**函数返回 tensor 所占用的内存。
   - *在不破坏梯度图的情况下处理内存管理，是旨在实现高效训练实践的用户关注的重点。*


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1327391944820920360)** (10 messages🔥): 

> `安装 Open Interpreter，Homebrew 和 pipx，Open Interpreter 功能` 


- **JanitorJesh 在安装 Open Interpreter 时遇到困难**：一位用户表示在 Mac 上安装 **Open Interpreter** 时遇到困难，遇到了与 **tiktoken** 包相关的**错误代码**，并且需要 **Rust** 编译器。
   - 在得到一些故障排除建议后，他们成功完成了所有设置。
- **Deseculavalutent 分享安装步骤**：一位用户概述了使用 **Homebrew** 和 **pipx** 安装 Open Interpreter 的步骤，强调了隔离 **Python** 应用程序的好处。
   - 其中包括安装 Homebrew、pipx 以及 Open Interpreter 本身的命令。
- **关于 Open Interpreter 能力的提问**：在完成安装后，一位用户询问了 Open Interpreter 最有用的功能，特别是关于其编辑视频的能力。
   - 另一位成员确认它可以运行任意命令，包括视频编辑命令。
- **关于 Open Interpreter 屏幕控制功能的讨论**：一位用户提到他们从未用过 Open Interpreter 控制和查看屏幕的功能。
   - 这引发了对该功能潜在能力和局限性的好奇。


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1327716773876731924)** (8 messages🔥): 

> `Stable Audio 3 开源公告，高血压识别数据集请求` 


- **Stable Audio 3 正式开源**：音频爱好者的好消息！**Stable Audio 3** 将会[开源](https://vxtwitter.com/dadabots/status/1878505508711157791)并基于音乐进行训练，为开发者和创作者提供令人兴奋的机会。
   - 这一进展有望增强音频领域基于音乐的项目和工具包。
- **高血压音频数据集请求**：一位成员正在寻求**高血压识别数据集**，特别是高血压患者的音频数据集。
   - 他们正在寻求协助，表明该健康相关项目需要数据收集方面的合作。



**提到的链接**：<a href="https://vxtwitter.com/dadabots/status/1878505508711157791">来自 undefined 的推文</a>：未找到描述

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1328309778870698055)** (1 messages): 

> `Megatron Checkpoint 转换，评估脚本，NVIDIA MegaTron-LM` 


- **需要 Megatron Checkpoint 到 HF 的转换**：一位成员使用 **Megatron** 进行了参考训练运行，并寻求一个在不使用 **Nemo** 的情况下将 Checkpoint 从 **torch 格式**转换为 **HF 格式**的脚本。
   - 他们强调，拥有此脚本将**节省大量工作**，并请求任何拥有相关代码的人进行分享。
- **获取训练材料**：提到了与 **checkpoint 示例**和训练日志相关的材料，可以在[此处](https://fz-juelich.sciebo.de/s/Yh8Q8RRTxliupLh)获取，但指出权限问题阻止了文件上传。
   - 这似乎是共享重要训练资源的障碍，并促使需要替代的交换方法。
- **克隆 NVIDIA MegaTron-LM**：该成员确认用于运行的 **MegaTron-LM** 仓库是官方 NVIDIA 仓库的克隆，具体由 commit hash `31a29b87` 标识。
   - 这便于从官方源进行克隆以供参考，并确保与最新更新保持一致。
- **请求成功的代码交换**：该用户鼓励团队协作，邀请成员分享关于 Megatron 的 Checkpoint 转换过程的任何**可用代码**或提示。
   - 他们强调社区支持是解决当前 Checkpoint 转换挑战的关键。



**提到的链接**：<a href="https://fz-juelich.sciebo.de/s/Yh8Q8RRTxliupLh">sciebo - www.hochschulcloud.nrw</a>：C4_50B_cosine_bs-4M_lr-6e-3_warmup-1000 已公开分享

  

---


---


---


---


---


---


{% else %}


> 完整的频道细分内容已为邮件版缩减。
> 
> 如果您想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}