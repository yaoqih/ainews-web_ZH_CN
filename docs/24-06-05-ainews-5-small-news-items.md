---
companies:
- openai
- cohere
- deepmind
- hugging-face
- nvidia
- mistral-ai
date: '2024-06-06T02:50:37.633247Z'
description: '**OpenAI** 宣布 ChatGPT 的语音模式“即将推出”。**Leopold Aschenbrenner** 发布了关于 AGI
  时间线的五部分系列文章，预测当前的 AI 进展将催生**万亿美元级别的集群**。**Will Brown** 发布了一份全面的生成式 AI 手册（GenAI Handbook）。**Cohere**
  以 **50 亿美元的估值**完成了 **4.5 亿美元的融资**。DeepMind 关于 **LLM（大语言模型）不确定性量化**的研究，以及表现优于 Transformer
  的 **xLSTM 模型**受到了关注。文中还分享了关于 **LLM 概念几何学**以及通过**消除矩阵乘法**来提高效率的方法的研究。此外，还讨论了**参数高效微调
  (PEFT)** 和 **LLM 自动对齐**。新工具包括用于 AI 智能体的 **LangGraph**、具有更长上下文窗口的 **LlamaIndex**，以及
  **Hugging Face** 与 **NVIDIA NIM** 针对 Llama3 的集成。**Mistral AI** 为其模型发布了微调 API。'
id: e87581dc-f03a-4370-afce-92a804557693
models:
- llama-3
- xLSTM
original_slug: ainews-5-small-news-items
people:
- leopold-aschenbrenner
- will-brown
- rohanpaul_ai
- richardmcngo
- omarsar0
- hwchase17
- clementdelangue
- sophiamyang
title: 5 条新闻简讯
topics:
- uncertainty-quantification
- parameter-efficient-fine-tuning
- automated-alignment
- model-efficiency
- long-context
- agentic-ai
- fine-tuning
- inference-optimization
---

<!-- buttondown-editor-mode: plaintext -->**AGI 现实主义或许正是人类所需要的**

> 2024年6月4日至6月5日的 AI 新闻！
我们为您检查了 7 个 subreddits、[384 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 29 个 Discord（401 个频道和 3628 条消息）。
预计节省阅读时间（按 200wpm 计算）：**404 分钟**。

- OpenAI 仍表示 [ChatGPT 的语音模式“即将推出”](https://www.youtube.com/watch?v=4w0Pqs3CuWk)
 
![image.png](https://assets.buttondown.email/images/d99a5d17-2b05-43ed-a6fb-dd97697cc383.png?w=960&fit=max)
 
- Leopold Aschenbrenner [发布了献给 Ilya 的 AGI 时间线五部曲系列文章](https://x.com/leopoldasch/status/1798016486700884233)，并配合 [Dwarkesh 播客](https://www.youtube.com/watch?v=zdbVtZIn9IM)，预测按目前的进展速度将出现**万亿美元级别的集群**  
![image.png](https://assets.buttondown.email/images/fe9c361c-3e46-4212-ad48-62a44f6a1c77.png?w=960&fit=max)
 
- Tom Yeh [手绘插解 llm.c](https://x.com/ProfTomYeh/status/1798042265883156651)  
![image.png](https://assets.buttondown.email/images/88c9b7bf-60f0-4604-bdc0-27d30bf1dc3b.png?w=960&fit=max)
 
- Will Brown 发布了[一份全面的 GenAI 手册](https://x.com/willccbb/status/1798423849870270671)  
![image.png](https://assets.buttondown.email/images/257555ea-33ad-4c40-8ff2-698f8b1bb6a4.png?w=960&fit=max)
 
- Cohere [以 50 亿美元估值完成了 4.5 亿美元融资](https://www.reuters.com/technology/nvidia-salesforce-double-down-ai-startup-cohere-450-million-round-source-says-2024-06-04/)，但尚未正式宣布。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成（四次运行中的最佳结果）。我们正在尝试使用 Haiku 进行聚类和流程工程。

**AI 模型与架构**

- **新模型与架构**：[@arankomatsuzaki](https://x.com/arankomatsuzaki/status/1798177198781899194) 分享了一篇关于 **LLM 中不确定性量化**的 DeepMind 论文。[@hardmaru](https://twitter.com/hardmaru/status/1798202333383516613) 重点介绍了 xLSTM，这是 **LSTM 的一种扩展**，在性能和扩展性方面优于 Transformers 和 State Space Models。[@omarsar0](https://twitter.com/omarsar0/status/1798010546522103898) 讨论了一项关于 **LLM 中概念几何结构**的研究，发现简单的分类概念被表示为单纯形（simplices），而层级相关的概念则是正交的。
- **效率提升**：[@omarsar0](https://twitter.com/omarsar0/status/1798373841741185261) 分享了一篇论文，提出了一种**从 LLM 中消除矩阵乘法操作**的实现方案，同时在十亿参数规模下保持性能，可能将内存消耗降低 10 倍以上。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1798104182232101188) 讨论了一篇关于大模型**参数高效微调（PEFT）方法**的综述，将其分为加性、选择性、重参数化和混合方法。
- **对齐与安全**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1798156694012490096) 概述了一个场景，描述了**构建对齐失当的 AGI 如何导致人类失去控制**，AGI 会利用对实验室服务器的特权访问权限。[@omarsar0](https://twitter.com/omarsar0/status/1798014572663583165) 分享了 **LLM 自动对齐**方法的概述，探索了通过归纳偏置（inductive bias）、行为模仿、模型反馈和环境反馈进行对齐的方向。

**工具与框架**

- **LangChain 和 LangGraph**：[@hwchase17](https://twitter.com/hwchase17/status/1798386148982878477) 推出了一个新的 DeepLearning.AI 课程，关于**使用 LangGraph 构建 AI Agent**，LangGraph 是 LangChain 的一个扩展，用于开发具有持久化和 Agentic Search 能力的可控 Agent。[@llama_index](https://twitter.com/llama_index/status/1798049438814081138) 展示了在尝试回答来自异构文档的多部分问题时，**LlamaIndex Agent 中更长的上下文窗口（Context Window）**如何带来更好的性能。
- **Hugging Face 和 NVIDIA 集成**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1798022781713731595) 指出 **Hugging Face 正在成为 AI 计算的入口**，现在可以直接从 Hugging Face Hub 为 Llama3 模型访问 NVIDIA NIM。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1798135458842517629) 讨论了 **Optimum-NVIDIA，这是一个 Hugging Face 推理库**，利用 NVIDIA 的 FP8 格式和 TensorRT-LLM 软件来实现更快的 LLM 推理。
- **Mistral AI 和微调**：[@sophiamyang](https://twitter.com/sophiamyang/status/1798415316180988403) 宣布发布 **Mistral 的微调 API**，允许用户微调自己的 Mistral 模型并在 La Plateforme 上高效部署。[@HamelHusain](https://twitter.com/HamelHusain/status/1798412100072813000) 分享了该 API 的现场演示，详细介绍了数据准备、超参数选择和集成过程。

**数据集与基准测试**

- **合成数据生成**：[@_philschmid](https://twitter.com/_philschmid/status/1798388387822317933) 概述了**为微调自定义 Embedding 模型生成合成数据**的流水线，包括创建知识库、数据分块、使用 LLM 生成问题、可选地生成 Hard Negative 示例、去重和过滤数据对，以及使用 Sentence Transformers 3.0 微调 Embedding 模型。
- **评估指标**：[@abacaj](https://twitter.com/abacaj/status/1798366581254504573) 构建了一个用于**分析恶意 Solidity 合约代码**的基准测试，发现只有像 GPT-4o 和 Claude-Opus 这样的顶级闭源模型偶尔能识别出恶意代码，而开源模型失败率超过 95%。[@mervenoyann](https://twitter.com/mervenoyann/status/1798274389928300678) 指出 **MMUPD（一个视频分析中多模态 LLM 的综合评估基准）**现在已作为排行榜托管在 Hugging Face Hub 上。
- **特定领域数据集**：[@_arohan_](https://twitter.com/_arohan_/status/1798401202138566953) 强调了 **Google 的 Gemini 1.5 模型在 Video-MME 基准测试的许多子任务中表现优于私有模型**，该基准用于评估视频分析中的多模态 LLM。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1798343869441990778) 分享了一篇比较 Gemini 1.5 Flash 和 GPT-4o 在 Video-MME 基准测试上表现的论文。

**应用与用例**

- **企业级 AI 和 RAG**：[@llama_index](https://twitter.com/llama_index/status/1798376976849469559) 分享了一个关于**使用 Bedrock 和 Ragas.io 构建企业级 RAG（检索增强生成）**的完整视频教程，涵盖了合成数据集生成、基于 Critic 的评估和微调。[@RazRazcle](https://twitter.com/RazRazcle/status/1798040468951048411) 采访了 Ironclad 的联合创始人 @gogwilt，讨论了他们如何成功地**将 AI 用于合同谈判**，顶级客户超过 50% 的合同由 Ironclad AI 谈判完成。
- **AI 助手与 Agent**：[@svpino](https://twitter.com/svpino/status/1797976775529844823) 构建了一个**能够倾听并使用网络摄像头观察世界的 AI 助手**，并在视频教程中解释了该过程。[@bindureddy](https://twitter.com/bindureddy/status/1798204209231536177) 预测 **AI 助手将变得必不可少**，人们对它们的依赖将呈指数级增长。
- **创意 AI 与多模态模型**：[@suno_ai_](https://twitter.com/suno_ai_/status/1798036388329472380) 宣布了一项竞赛，**使用其 VOL-5 模型从任何声音中创作歌曲**，获胜者将获得早期访问权限，其中一位获胜者的视频将在社交媒体上分享。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1798385843662680520) 展示了一个 **AI 驱动的工具，用于让视频游戏中的非玩家角色（NPC）变得可玩**，这是 @cubzh_、@GigaxGames 和 @huggingface 的合作成果。

**讨论与观点**

- **AI Timelines and Risks**: [@leopoldasch](https://twitter.com/leopoldasch/status/1798156694012490096) 认为，基于从 GPT-2 到 GPT-4 的进展以及在 compute、算法效率和模型能力方面的预测趋势，**2027 年实现 AGI 具有惊人的可能性**。[@_sholtodouglas](https://twitter.com/_sholtodouglas/status/1798052154709852198) 将 Leopold 的文章描述为捕捉了 AI 领域关键参与者的世界观，并预测如果时间线得以维持，未来几年将会非常疯狂。
- **Compute and Scaling**: [@ylecun](https://twitter.com/ylecun/status/1798333227175690533) 提出了 **objective-driven AI** 的概念，即智能系统需要具备推理、规划以及根据其内部世界模型满足 guardrails 的能力，而关键挑战在于设计合适的 guardrails。[@ethanCaballero](https://twitter.com/ethanCaballero/status/1798385264248885525) 指出，随着 **能源和电力成为扩展至 AGI 的新瓶颈** 变得清晰，某些股票可能会在未来几年飙升。
- **Open Source and Democratization**: [@ylecun](https://twitter.com/ylecun/status/1798118502198645245) 分享了一篇文章，讨论了 **开源 AI 与少数大公司控制的专有 AI 的利弊**，认为那些最担心 AI 安全的人往往高估了 AI 的力量。[@far__el](https://twitter.com/far__el/status/1798375007460225096) 预测 **Meta 和其他公司将不会开源强大的 AI**，我们正走向“AGI 君主制”。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取现在可以运行了，但仍有很大改进空间！

以下是近期 AI 进展的摘要，按主题分类，关键细节已加粗并链接至相关来源：

**AI 模型发布与能力**

- **潜在的 GPT-5 发布**：The Information 报道称 [**GPT-5 可能会在 2024 年 12 月发布**](https://i.redd.it/rxh46fczao4d1.jpeg)，这表明 OpenAI 的语言模型能力将有重大提升。
- **先进 AI 模型**：Microsoft CTO Kevin Scott 声称 [**即将推出的 AI 模型可以通过博士资格考试**](https://x.com/tsarnick/status/1798167323893002596)，表明在记忆和推理能力方面有显著改进。
- **角色语音生成**：一段 [YouTube 视频展示了 GPT-4o 生成角色语音的能力](https://www.youtube.com/watch?v=4w0Pqs3CuWk)，展示了该模型在语音合成方面的多功能性。
- **机器人化的未来**：Nvidia 承诺随着 AI 变得更加先进，[“一切都将变得机器人化”](https://www.youtube.com/watch?v=nxO_t5N82m0)，暗示 AI 在各个领域的集成度将不断提高。
- **职场中的 AI 克隆**：Zoom 的 CEO 预测 [**AI 克隆最终将处理人们的大部分工作**](https://qz.com/zoom-ceo-eric-yuan-ai-avatar-jobs-1851518757)，这可能会改变工作的本质。

**AI 停机与担忧**

- **AI 服务同时停机**：主要的 AI 服务 [ChatGPT、Claude 和 Perplexity 经历了同时停机](https://techcrunch.com/2024/06/04/ai-apocalypse-chatgpt-claude-and-perplexity-are-all-down-at-the-same-time/)，引发了对这些服务可靠性和影响的担忧。
- **ChatGPT 长时间停机**：[ChatGPT 停机约 12 小时](https://i.redd.it/9qg0eyc6fk4d1.jpeg)，给依赖该服务的用户带来了问题，并凸显了对稳健基础设施的需求。
- **吹哨人与安全担忧**：现任和前任 OpenAI 员工以及其他 AI 研究人员[愿意向公众披露有关 AI 风险和安全问题的机密信息](https://www.reddit.com/gallery/1d80n63)。一名 OpenAI 安全研究员辞职并[签署了一封信，呼吁 AI 实验室支持员工就这些问题公开发声](https://twitter.com/clwainwright/status/1798013345926447486)。
- **网络安全漏洞**：Leopold Aschenbrenner 在[警告董事会关于中国可能利用的网络安全漏洞后被 OpenAI 解雇](https://v.redd.it/b8hjkl8fao4d1)，这引发了对公司内部安全问题处理方式的质疑。
- **AI 霸权竞赛**：OpenAI 内部人士在《纽约时报》的一篇文章中警告称，[存在一场“鲁莽”的 AI 霸权竞赛](https://www.nytimes.com/2024/06/04/technology/openai-culture-whistleblowers.html)，强调了与 AI 技术快速发展相关的潜在风险。

**AI 投资与合作伙伴关系**

- **埃隆·马斯克的芯片分配**：埃隆·马斯克指示 Nvidia [优先向 X 和 xAI 交付处理器，而非 Tesla](https://www.cnbc.com/2024/06/04/elon-musk-told-nvidia-to-ship-ai-chips-reserved-for-tesla-to-x-xai.html)，这表明他旗下公司的重点正转向 AI 开发。
- **阿联酋-美国 AI 合作伙伴关系**：阿联酋正[在 AI 领域与美国合作，利用其 2 万亿美元的主权财富基金成为全球 AI 强国](https://www.benzinga.com/news/24/06/39153123/now-uae-forges-us-tie-up-with-1-5b-deal-to-become-global-ai-powerhouse)，突显了该领域日益激烈的国际竞争。
- **OpenAI-Google 合作**：Ilya Sutskever 和 Jeff Dean [于 2024 年 5 月 30 日共同发布了一项美国专利](https://i.redd.it/mxaehyg0ni4d1.png)，暗示了 OpenAI 和 Google 在 AI 研发方面可能存在的合作。

**AI 模型与基准测试**

- **SDXL 模型参数**：[SDXL 模型的 UNET 具有 2.6B 参数，包含文本编码器在内共有 3.5B 参数，包含 Refiner 的完整流水线则有 6.6B 参数](https://i.redd.it/fkij1pomxi4d1.jpeg)，这提供了对该模型架构和复杂性的深入了解。
- **Yi-1.5-34B 模型性能**：[Yi-1.5-34B 模型是 LMSYS 排行榜上排名最高的 ~30B 模型和 Apache 2.0 协议模型](https://i.redd.it/v5w3myp3gi4d1.png)，展示了其与同等规模和许可类型的其他模型相比的强劲性能。
- **L3-MS-Astoria-70b 模型排名**：[L3-MS-Astoria-70b 模型成为 Uncensored General Intelligence Leaderboard 上的顶级模型](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)，展示了其在通用智能任务中的能力。
- **GPT-4o 易用性**：尽管 GPT-4o 在 MMLU 和 LMSYS 基准测试中排名很高，但一些用户发现[与其他模型相比，它更难进行 Prompt 提示和遵循指令](https://www.reddit.com/r/singularity/comments/1d7wmjn/mmlu_lmsys_vs_vibes_on_gpt4o/)，这突显了用户体验和模型易用性的重要性。

---

# AI Discord 摘要回顾

> 摘要之摘要的摘要

**1. 微调技术与模型集成**:

- 成员们讨论了使用 **Deepspeed zero2** 和 **Qlora** 等工具进行 **finetuning models** 的重要性，重点介绍了 **Llama3** 的成功集成以及磁盘卸载（disk offloading）等内存管理策略 ([Unsloth AI](https://discord.com/channels/1179035537009545276))。
- **Mistral Fine-Tuning Hackathon** 引发了热烈反响，鼓励参与者探索 Mistral 的新功能，详见 [Mistral tutorial](https://docs.mistral.ai/capabilities/finetuning/) 和相应的 [YouTube demos](https://youtu.be/zXFxmI9f06M) ([LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560))。

**2. 模型训练与优化中的问题**:

- 成员们对模型训练过程中的 **OOM (Out of Memory)** 错误表示沮丧，并寻求高效 VRAM 管理技术和验证 YAML 配置等解决方案 ([OpenAccess AI Collective](https://discord.com/channels/1104757954588196865))。
- 针对 Jarvis Labs 中的 **CUDA library mismatches** 以及 LM Studio 中的 **GGUF compatibility** 等问题分享了排错建议 ([HuggingFace](https://discord.com/channels/879548962464493619) 和 [LM Studio](https://discord.com/channels/1110598183144399058))。

**3. AI 领域的新工具与资源**:

- Stability AI 发布了 **Stable Audio Open**，用于生成短音频片段，强调使用自定义数据进行本地微调 ([Stable Diffusion](https://stability.ai/news/introducing-stable-audio-open))。
- 分享了多种有价值的资源，例如 William Brown 编写的 [全面 LLM 资源指南](http://genai-handbook.github.io/) 以及用于高性能 LLM 的 [FineWeb 技术报告](https://hf.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) ([HuggingFace](https://discord.com/channels/879548962464493619))。

**4. 社区关注点与协作项目**:
    
- 关于 **credit distribution and server performance** 的担忧被广泛讨论，许多成员报告了在接收额度时遇到问题或面临 **502 Gateway errors** ([LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) 和 [OpenRouter](https://discord.com/channels/1091220969173028894))。
- 在学习和实现新 AI 功能方面的协作努力包括对 **Flash-attn** GPU 兼容性的讨论，以及使用 Verba 等工具进行 **RAG chatbot integration** ([Nous Research AI](https://discord.com/channels/1053877538025386074) 和 [LangChain](https://discord.com/channels/1038097195422978059))。

**5. AI 安全与伦理讨论**:

- 在 **Hugging Face breach** 导致私有令牌泄露后，引发了安全担忧，进而讨论了互联网数据的可靠性 ([HuggingFace](https://discord.com/channels/879548962464493619))。
- 辩论了 **AGI development incentives** 的伦理问题以及确保模型公平使用的问题，强调了 LLM 架构中对齐 AI 行为和适当奖励模型的重要性 ([Interconnects](https://discord.com/channels/1179127597926469703) 和 [Latent Space](https://discord.com/channels/822583790773862470))。

---

# 第一部分：高层级 Discord 摘要

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **模型与研讨会热潮**：工程师们积极讨论用于微调的最佳 **code models**，许多人提到缺乏表现突出的特定资源或模型。同时，**研讨会参与者**强烈要求获取幻灯片和链接，并建议通过 [额度申请表](https://bit.ly/modal-credits) 申请 **Modal credits** 以领取 500 美元的奖励。

- **跨平台的额度混淆**：在各个平台上，用户对 **额度发放** 表示困惑，例如 Modal 额外的 500 美元和 Replicate 的兑换流程。对于 Modal 优惠的协助，Charles 通过电子邮件提供了帮助，而对于 **Replicate credits** 的问题，用户被引导发送包含详细信息的邮件以寻求支持。

- **策划微调资源**：一份详尽的 ***LLM fine-tuning explainers*** 列表受到关注，可通过 [此 LLM 指南](http://genai-handbook.github.io) 获取。此外，**Mistral Fine-Tuning Hackathon** 备受期待，其开发与 API 发布同步进行，这表明人们对探索 Mistral 的功能和资源（如 [微调教程](https://docs.mistral.ai/capabilities/finetuning/) 和 [YouTube 演示](https://youtu.be/zXFxmI9f06M)）有着浓厚兴趣。

- **磨练微调技术**：社区分享了关于 **Mistral fine-tuning** 的知识并寻求建议，包括关于垂直整合、API 优势和内存管理的讨论。此外，Predibase 用户赞扬了其重用基础模型的方法，并提出了改进微调过程的建议，例如增加对更多 epochs 的访问权限和 UI 数据过滤演示。

- **技术栈故障排除**：通过协作解决了设置不同技术时的各种挑战，例如 **Axolotl**、Jarvis Labs 的 CUDA 版本不匹配以及调试 LangChain notebooks。解决方案涵盖了从使用 Docker 简化 Axolotl 使用，到更新 CUDA 库，以及建议配置环境变量以实现无缝的 Langsmith 集成。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **在 Unsloth AI 上进行更快、更精简的预训练**：Unsloth AI 引入了 [持续预训练 (continually pretrain)](https://github.com/unslothai/unsloth/releases/tag/June-2024) LLM 的功能，其速度是之前 HF+FA2 的两倍，且仅需一半的 VRAM，详情见其 [博客](https://unsloth.ai/blog/contpretraining)。

- **Unsloth 尚不支持 Medusa**：工程师们根据提供的 GitHub [链接](https://github.com/FasterDecoding/Medusa) 确认，Unsloth 不支持使用 Medusa 进行微调，但它提供了改进的 Unsloth 更新，如 lm_head/embed_tokens 的磁盘卸载 (disk offloading) 和自动分词器 (tokenizer) 修复。

- **讨论中的 VRAM 管理技术**：分享了管理 VRAM 的技术，包括使用余弦学习率 (cosine learning rates) 和选择性卸载，并指出了 H100 GPU 的优化潜力以及通过 `del` 命令释放内存以运行多个模型的策略。

- **多节点实现的挑战**：虽然 **多 GPU 支持** 已启用，但多节点 (multinodal) 支持的实现预计会稍有延迟，这对于 70B 微调等项目至关重要。同时，还涉及了微调期间使用 LoRA adapter 等节省 VRAM 的替代方案。

- **为侧边项目寻找开源 TTS 模型**：一名成员为 waifu 伴侣应用/RPG 寻求“优秀的开源 TTS 模型”，得到了 "xttsv2 -rvc pipeline" 的推荐，展示了工程师之间在开源资源方面的积极协作。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 遇到障碍**：用户报告了 **downtime**（停机）以及对 Perplexity AI 模型选择的挫败感，评论中提到漫长的等待时间，以及一个奇怪的界面问题：请求生成图像时却收到文字描述而非实际图形。
- **AI 模型大比拼**：辩论对比了 **ChatGPT-4o** 与 **Claude 3**，指出 Perplexity 使用内部搜索索引的独特方法，并分享了资源链接，包括 [演示技巧](https://youtu.be/wjZofJX0v4M?feature=shared) 和 [Perplexity 搜索功能概述](https://www.perplexity.ai/search/how-does-perplexity-Qm71LBYBSkOApKNFCISS0Q)。
- **超越 SEO 的搜索**：在关于后端流程的讨论中，有人指出 Perplexity AI 的不同之处在于不依赖第三方服务进行爬取和索引，从而获得更高质量、受 SEO 策略操纵较少的搜索结果。
- **深入探讨停机事件**：分享了一篇[分析重大停机事件](https://www.perplexity.ai/page/Major-Outage-of-DCcT_vXARMmWZl8KCWB8Jg)的文章，深入了解 Perplexity AI 面临的技术问题。
- **通过共享链接扩展知识**：用户通过引用关于各种主题的 Perplexity AI 搜索结果来增强讨论，包括关于 [dailyfocus](https://www.perplexity.ai/search/httpsdiscordcom-repeat-this-p6zDFgNJS5Wn4D4YdmeEGg)、[Bitcoin](https://www.perplexity.ai/page/Bitcoin-WzdgAV4KQiqtb4k0q4RHRw) 的文章，并分享了关于必须使帖子可共享的提醒及附带指南。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **开放办公时间与面试准备**：工程师可以参加关于优化 LLM 推理和企业级 ML 的 **vLLM 和 Neural Magic 开放办公时间**，时间定于 [6月5日](https://neuralmagic.com/community-office-hours/) 和 [2024年6月20日](https://neuralmagic.com/community-office-hours/)。对于性能工程师面试准备，GitHub 上的 [awesomeMLSys](https://github.com/cuda-mode/awesomeMLSys) 提供了一份精选的问题和资源列表。

- **Triton Kernel PTX 访问与 GitHub 讨论**：关于从 **Triton kernels** 中提取 **PTX 代码**的疑问引导用户找到了一个讨论该流程的有用 [GitHub issue](https://github.com/triton-lang/triton/issues/3726)。用户将其初始搜索位置修正为 `~/triton/.cache` 以获取 PTX 代码。

- **破解 CUDA Stream 难题**：AI 工程师讨论在 CUDA 中使用**命名流 (named streams)** 以获得更好的性能，并分享了一个将操作主流化的 [pull request](https://github.com/karpathy/llm.c/pull/552)。修复 **PyTorch DDP 损失计算 bug** 的努力已通过一个成功的 [PR](https://github.com/karpathy/llm.c/pull/551) 告一段落。

- **大模型评估中的 OOM 困扰与量化怪癖**：如 [GitHub pull request](https://github.com/pytorch/ao/pull/328) 所示，在使用 **torchao APIs** 进行大模型评估时，显存溢出 (OOM) 问题困扰着开发者。AI 工程师建议在量化前将模型加载到 CPU 上，并针对大词表大小进行调整。

- **稀疏矩阵语义与 AI 中的稀疏性**：对稀疏矩阵的澄清促使分享了 [Wikipedia](https://en.wikipedia.org/wiki/Sparse_matrix) 定义和 PyTorch [README](https://github.com/pytorch/ao/tree/main/torchao/sparsity)。此外，还传阅了一篇总结了 300 多篇关于深度学习中稀疏性利用的综合性 [arXiv 综述论文](https://arxiv.org/abs/2102.00554)，以便更好地理解和实现。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **FineWeb 揭示 LLM 性能见解**：FineWeb 技术报告详细介绍了处理决策，并推出了 FineWeb-Edu 数据集，旨在增强以教育为中心的内容，并深入理解 Llama3 和 GPT-4 等高性能 LLM。[FineWeb 技术报告](https://hf.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)现已发布。

- **Firefox 中基于浏览器的 AI 与 Transformers.js**：Firefox 130 更新将包含用于设备端 AI 的 Transformers.js，初始功能针对图像自动生成替代文本（alt-text），以提高无障碍性。详情见此[公告](https://x.com/xenovacom/status/1797285648572821840)。

- **Nvidia NIM 加速模型部署**：Nvidia NIM 在 Hugging Face Inference Endpoints 上线，为云平台上的 Llama 3 8B 和 70B 等模型提供便捷的一键部署。部署参考见[此处](https://x.com/_philschmid/status/1797713003778883858)。

- **Hugging Face 与 Wikimedia 合作推动 ML 进展**：该合作利用 Wikimedia 的数据集进一步推动机器学习发展，强调了社区同意的重要性。该计划详情见[此处](https://huggingface.co/blog/frimelle/wikipedias-treasure-trove-ml-data)。

- **深入探讨 AI 的安全与伦理**：Hugging Face 安全漏洞的披露引发了关于基于互联网的数据存储的伦理影响和安全性的讨论，重点在于维持尊重的社区参与。

- **跨越技术壁垒**：基于扩散（diffusion）的语言建模策略的引入借鉴了图像生成模型中使用的原理，提出了处理文本“噪声”的新方法。

- **用于气候意识投资的 AI 工具**：开发了一款用于识别气候关注型投资机会并计算碳足迹的 AI 工具，利用了 `climatebert/tcfd_recommendation` 等模型，展示了 AI 在可持续金融领域的潜力。在此探索该 [AI 工具](https://huggingface.co/spaces/as-cle-bert/tcfd_counselor)。

- **AI 社区的知识共享**：各种 AI 相关项目和讨论涵盖了改进的徽标检测、Windows 上的 Apache Airflow 设置、有价值的 LLM 资源以及用于语言模型训练的高级德语语音数据集等主题，丰富了知识库。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**LM Studio 模型加载故障排除**：用户因 VRAM 不足面临模型加载问题；建议的解决方法是禁用 GPU offloading。一个特定案例强调了加载未保存为 GGUF 文件的 **Llama70b** 时的问题，建议使用符号链接（sym link）选项或进行文件转换。

**讨论强调模型性能与兼容性**：**Command R** 模型在 offload 到 Metal 时表现不佳；对于文本增强，虽然没有推荐特定模型，但可以关注排行榜上的 13B 模型。此外，有报告称 **SMAUG 的 BPE tokenizer** 在 **Llama 3** 版本 0.2.24 中存在困难。

**关于工作站 GPU 和操作系统的闲聊**：[ASRock Radeon RX 7900 XTX & 7900 XT 工作站 GPU](https://wccftech.com/asrock-radeon-rx-7900-xtx-7900-xt-workstation-gpus-12v-2x6-connector-2-slot-design-for-ai-setups/) 引起了关注，特别是其面向 AI 装置的设计。关于 Linux 的易用性评价褒贬不一，并讨论了因 Windows 的 Recall 功能引发隐私担忧而转向 Linux 的话题。

**LM Studio Bug 反馈**：指出了 **LM Studio v0.2.24** 中的一个 bug，涉及预设配置中多余的转义字符，例如 `"input_suffix": "\\n\\nAssistant: "`。

**隐私与安全**：Windows 的 Recall 功能可能通过收集敏感数据产生安全漏洞，引发了隐私担忧。在轻松的话题中，关于 IT 支持挑战的轶事——包括一台沾染了猫尿气味的电脑——为技术支持的苦恼讨论带来了幽默感。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **黑客猛烈攻击 AI 服务**：由于 **Anonymous Sudan** 发起的 DDoS 攻击，**ChatGPT, Claude, Gemini, Perplexity** 和 **Copilot** 服务经历了停机。这一事件揭示了超出典型云服务器预期的脆弱性。

- **比较 AI 订阅**：AI 工程师们讨论了 AI 订阅的实用性，对比了 **GPT** 和 **Character AI** 在书籍摘要和内容创作等任务中的表现。

- **数学难倒了 AI**：工程师们观察到 **GPT** 等 AI 语言模型在处理数学问题时持续表现出弱点，突显了计算中的不准确性和逻辑疏忽。

- **AI 变得个性化且实用**：讨论展示了 AI 的现实世界集成，例如将 **ChatGPT** 与家庭自动化系统对接，强调了在实际场景中的优势和局限性。

- **在 Google Sheets 中使用 GPT-4 Vision**：有人提出了关于实现 **GPT-4 vision** 来分析和描述 **Google Sheets** 中图像的问题，表明了将 AI 效用扩展到电子表格任务中的兴趣。 

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Audio Open 惊艳亮相**：Stability.ai 推出了 **Stable Audio Open**，这是一个开源模型，用于根据文本提示生成短音频片段，包括音效和制作元素。该模型支持生成长达 47 秒的音频剪辑，强调为声音设计师和音乐家提供创新，并支持本地 fine-tuning；更多详情请点击[此处](https://stability.ai/news/introducing-stable-audio-open)。

- **WebUI 的奇迹：Stable Diffusion 的无限可能**：社区成员对 Stable Diffusion 的 **A1111** 和 **InvokeAI** WebUI 进行了热烈对比，认可了 A1111 的易用性以及 InvokeAI 独特的 "regional prompting" 功能，后者可以在 [GitHub](https://github.com/invoke-ai/InvokeAI) 上探索。

- **聚焦训练微调**：有人寻求关于使用 **regularization images** 的技术澄清，成员们讨论了这些图像是否可以在训练过程中取代 captions。同时，对 **Stable Audio Tools** 及其用途（包括可能的 Google Colab 使用和商业许可）表现出明显的兴趣，并引用了其 GitHub [仓库](https://github.com/Stability-AI/stable-audio-tools)。

- **UI 灵活性之最**：**ComfyUI** 因其在图像生成任务中的适应性而被推荐，尽管学习曲线较陡峭，正如一位成员所言："你可以先用 cascade 或 sigma 生成，然后用 sdxl 进行精炼..."。

- **新手入门指南**：新用户被引导至丰富的社区策划资源（如教程）来学习 Stable Diffusion，包括 [Sebastian Kamph 在 YouTube 上](https://youtu.be/kqXpAKVQDNU?si=EHs5JZaQmE1yTi1Q)关于 A1111 入门的综合指南。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI 发现其审美感**：围绕利用 AI 控制显示墙上的图案和颜色展开了讨论，这可能导致个性化艺术或品牌装饰的出现。有人提出了这是否会演变为一种 AI 驱动的室内设计形式。

- **重新审视 RLCD 的热度**：对 RLCD 技术的营销进行了审查，引发了关于其核心创新方面的对话，并与 Samsung 的 QD-OLED 显示屏进行了对比。对于新模型是否显著超越现有的 transflective 屏幕技术，怀疑态度依然存在。

- **AGI 发展指日可待**：对 AGI 的投资日益增长成为关注焦点，引用了一篇预测 2025/26 年 AGI 能力将取得实质性进展的 [博客](https://situational-awareness.ai/)，引发了关于领先实验室与更广泛行业影响之间差距扩大的对话。

- **平衡 IQ 与自主性 (Agency)**：辩论了开源社区招聘中 IQ 测试的价值，并将其与“高自主性 (high agency)”特质进行了对比。讨论强调了后者在促成成功方面的优越性，因为它与主动性、模式识别和长期愿景密切相关。

- **剖析深度学习的局限性**：分享的文献深入探讨了深度学习在复杂推理方面的困境，无论是 Transformers 还是 SSMs。社区消化了关于将“chain-of-thought”策略扩散到模型中的论文，以及旨在增强 RLHF 鲁棒性的 SRPO 等方法。

- **开源实现激发热情**：NVIDIA 在 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro) 中公开披露了 RETRO 模型，引发了关于 AI 研究民主化以及尖端模型更广泛可访问性的讨论。

- **Lm-evaluation-harness 故障排除**：一位用户在从 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb) 获取所需输出时遇到困难，大家达成共识，认为结果可能隐藏在 tmp 文件夹中。社区渴望获得关于为 LLaMA 3 8B instruct 模型实现 loglikelihood 指标的指导。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GLM-4 打破语言障碍**：[GLM-4](https://x.com/ChatGLM/status/1798292207574901012) 的推出带来了对 26 种语言的支持，其能力扩展到代码执行和长文本推理。开源社区可以在 [GitHub](https://github.com/THUDM/GLM-4) 上找到该仓库并为其开发做出贡献。

- **探索 Nomic-Embed-Vision 的优越性**：社区正在讨论 [Nomic-Embed-Vision](https://x.com/nomic_ai/status/1798368463292973361?s=46&t=stOPrwZiN_fxSK0RuC8Flg) 的进展，它在为图像和文本创建统一 embedding 空间方面优于 OpenAI CLIP 等模型。对于感兴趣的人，权重和代码均可用于实验。

- **对比学习损失函数见解分享**：最近发表的一篇论文介绍了一种名为 [Decoupled Hyperspherical Energy Loss (DHEL)](https://arxiv.org/abs/2405.18045) 的新型对比学习目标，以及一个比较不同 InfoNCE 类型损失的相关 [GitHub 仓库](https://github.com/viig99/ContrastiveLearningLossComparison)。这些资源可能会极大地惠及深度学习社区的研究人员。

- **关于 Microsoft 挪用创意的讨论**：有关 Microsoft 涉嫌在未署名的情况下挪用创意的担忧浮出水面，一篇相关的 [arXiv 论文](https://arxiv.org/pdf/2405.19888) 成为讨论非故意开源概念的切入点。

- **AI 模型与数据集的测试与利用**：关于在 [NVIDIA NIM](https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct) 上测试 Phi-3 Vision 128k-Instruct 模型，以及利用 [Openbmb 的 RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset?row=2) 构建应用程序的讨论正在进行中。鼓励成员参与并提供有关模型性能和数据集效用的反馈。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **GraphRAG 构建方案讨论**：成员们就构建 **GraphRAG** 是通过手动定义图以获得完全控制，还是使用 LLM 实现自动化进行了辩论；每种方法都会影响工作量和数据映射的有效性。此外，还举办了一场企业级 **RAG workshop**，探索了 **Bedrock** 模型和 Agentic 设计模式，同时 **Prometheus-2** 因其开源特性，成为评估 RAG 应用时替代 GPT-4 的一个选择。

- **元数据提取创新**：引入了全新的 **Metadata Extractor** 模块和教程，旨在帮助理清长文本段落。关于在 **Chroma Database** 中存储 `DocumentSummaryIndex` 的疑问得到了明确答复：Chroma 无法在此场景下使用。

- **检索与索引的实用解决方案**：通过合并相关的 [pull request](https://github.com/run-llama/llama_index/pull/13938)，解决了一个关于 **Neo4j** 集成查询引擎的持久性 Bug，并分享了针对电子商务应用微调 **“intfloat/multilingual-e5-large”** Embedding 模型的方法。事实证明，单个 **QueryEngineTool** 能够高效管理多个 PDF，消除了对其累积操作性的担忧。

- **解决查询精度问题**：针对用户在 vectorstore 顶部响应中遇到无关材料的问题，建议通过分数（score）过滤结果，以确保检索结果具有更高的相关性和精度。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**ChatGPT 4 展现表演天赋**：OpenAI 的 ChatGPT 4 引入了令人印象深刻的全新*语音生成功能*，如[分享的视频](https://x.com/ai_for_success/status/1798046241307459673)所示，其创造独特角色声音的能力引起了热烈反响。

**DALLE3 表现下滑**：用户对 DALLE3 输出质量的明显下降表示担忧，无论是传统用法还是 API 集成都令人失望。

**辩论 AI 变现的伦理**：最近的讨论显示出社区对 AI 模型非商业许可的明显不满，批评了以经济利益为中心的动机以及训练 T5 等模型所需的大量资源。

**LLM 失去逻辑**：Open-Sci 团队的一篇新论文揭露了大语言模型表现出的推理能力“剧烈崩溃”，可在[此处](https://arxiv.org/abs/2406.02061)查看评论，并附有[代码库](https://github.com/LAION-AI/AIW)和[项目主页](https://marianna13.github.io/aiw/)。

**WebSocket 异常**：**whisperfusion pipeline** 中 *WhisperSpeech* 服务的 WebSockets 问题引发了 [StackOverflow](https://stackoverflow.com/questions/78570704/websocket-closes-unexpectedly-in-tts-service-with-multiprocessing-and-asyncio) 上的详细咨询，希望能解决意外关闭的问题。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Rust 兴起，Mojo 剑指新高度**：一位成员称赞了一个 [YouTube 教程](https://youtu.be/O4sVw4YQB24)，该教程强调了 Rust 通过 FFI 封装在系统开发中的安全性，证明了工程社区对安全高效系统编程的兴趣。

**Python 开发者的转型建议**：YouTube 上的一份 Python 到 Mojo [迁移指南](https://www.youtube.com/watch?v=9ag0fPMmYPQ)受到好评，它汇编了对于转向 Mojo 的非计算机专业工程师非常有益的基础底层计算机科学知识。

**Mojo 的枚举替代方案**：虽然 Mojo 目前缺乏 `Enum` 类型，但讨论转向了其对 [`Variants`](https://docs.modular.com/mojo/stdlib/utils/variant/Variant) 的适配，并提及了正在进行的 [GitHub discussion](https://github.com/modularml/mojo/issues/43)，供关注后续进展的人参考。

**Nightly 更新引发关注**：发布了新版本的 Mojo 编译器（`2024.6.512`），并提供了在 VSCode 中管理版本的建议。同时，针对 `Coroutine.__await__` 变为 consuming 等变化的挑战也得到了处理，详见 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。

**加密库需求迫切**：结合安全与编程领域，一位用户强调了在 Mojo 中建立加密库（cryptography library）的紧迫性，认为该功能将会非常“火爆”，并强调了在语言能力中构建健壮性的必要性。

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **投资者全力投入机器人 AI**：投资者正在寻找**机器人 AI 领域的 ChatGPT 等价物**。根据一篇 [Substack 文章](https://www.newcomer.co/p/why-investors-cant-get-enough-of)，他们渴望支持那些拥有强大机器人 Foundation Models 且无需承担硬件设计风险的公司。

- **国家安全与 AI 商业秘密**：科技界就因泄密而解雇个人的事件展开了辩论，重点关注 **AI 国家安全中商业秘密**被低估的作用。人们担心 **OpenAI** 和 **Anthropic** 等实验室对于在 3 到 5 年内实现研究员级 AI 过于自信，一些人认为这源于激励机制错位和错误的推断。

- **迈向能通过博士考试的 AI？**：**Microsoft CTO Kevin Scott** 预测，即将推出的 AI 模型可能很快就能通过博士资格考试。他将目前的 **GPT-4** 等模型比作能应对高中 AP 考试的水平。博士考试的难度（尤其是伯克利大学初试中观察到的 75% 淘汰率）也是讨论的话题，展示了此类 AI 模型将面临的挑战。

- **为解决问题付费**：[rewardbench.py](https://github.com/allenai/reward-bench/issues/137) 中的一个未解决问题导致不同 Batch Size 下的结果出现偏差；Nathan Lambert 为该问题的解决提供了 25 美元的悬赏。此外，**AutoModelForSequenceClassification** 被称为“有点被诅咒”，暗示通过调整可能实现改进。

- **AGI 讨论引发复杂反应**：对话显示，社区在对过度乐观的 **AGI 爱好者**和散布阴霾的 **Doomers** 的厌烦程度上持平。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Cohere 的 API 会保持免费吗？**：成员们纷纷猜测 **Cohere** 的免费 API 可能会停止服务，敦促他人寻求官方确认，不要理会未经证实的传言。

**规范多用户机器人聊天**：工程师们讨论了在多用户聊天线程中引入 **LLM** 的挑战，建议给消息打上用户名标签以提高清晰度。

**寻找终极聊天组件**：一位社区成员询问是否有基于 React 的聊天组件；他们被引导至 **[Cohere Toolkit](https://github.com/cohere-ai/cohere-toolkit/?ref=cohere-ai.ghost.io)**，该工具虽然不是完全基于 React 构建，但可能包含用 React 编写的聊天框等元素。

**React 组件与 Cohere 的协同**：虽然 **Cohere Toolkit** 缺乏 React 组件，但该开源工具被定位为实现 **RAG 应用**的有用资源，可能与 React 实现兼容。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**内存溢出排查**：用户报告在 2xT4 16GB GPU 上运行目标模块时出现 **Out of Memory (OOM)** 错误，同时伴有异常的 loss:0.0 读数，这可能表明参数配置或资源分配存在严重问题。

**饥渴模型的数据盛宴**：[HuggingFace FineWeb 数据集](https://huggingface.co/HuggingFaceFW)是一个源自 **CommonCrawl**、拥有 15 万亿 Token 的庞大集合，因其有望降低训练大模型的门槛而引起轰动，尽管人们对其充分利用所需的计算和财务资源表示担忧。

**Deepspeed 主导模型训练讨论**：工程讨论显示，人们更倾向于使用**命令行**运行 **Deepspeed** 任务，包括使用 **Deepspeed zero2** 成功微调 **Llama3** 模型，并在微调中选择了 **Qlora** 而非 Lora。

**寻求快速解决方案**：一位成员对 **Runpod** 缓慢的启动时间表示沮丧，特别是启动一个 140 亿参数的模型需要大约一分钟，影响了成本效益；有人提出了关于具有更快模型加载能力的替代 Serverless 供应商的问题。

**模型混杂与困惑**：虽然社区对 **GLM-4 9B** 模型表现出明显的热情，但关于其性能和用例的具体反馈似乎很少，这表明要么是部署尚新，要么是用户经验分享存在缺口。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **实时 AI 革命**：[LiveKit](https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww) 获得了 **2250 万美元的 A 轮融资**，旨在开拓 AI 的传输层，并将投资者兴趣的催化剂归功于 GPT-4 的能力。

- **多模态 AI 备受瞩目**：[Twelve Labs](https://www.prweb.com/releases/twelve-labs-earns-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html) 获得了 **5000 万美元的 A 轮融资**，并推出了 Marengo 2.6，旨在完善多模态基础模型。

- **预测精准度的艺术**：Microsoft Research 发布了 **Aurora**，旨在通过利用 AI 基础模型的进展，大幅提高天气预报的准确性。

- **AI 对齐透明度受到质疑**：Teknium 对 OpenAI 在对齐奖励和审核分类器方面的不透明表示质疑；讨论揭示了奖励模型通常被整合在大型语言模型（LLMs）本身的架构中。

- **内容管理获得 AI 助力**：[Storyblok](https://x.com/alexadark/status/1798031781377298751?s=46&t=90xQ8sGy63D2OtiaoGJuww) 获得了 **8000 万美元的 C 轮融资**，以开发 AI 驱动的内容平台，并启动了其新 Ideation Room 的公开测试。

- **Anthropic 深入研究单语义性（Monosemanticity）**：Anthropic 安排了一场关于 **Scaling Monosemanticity** 的深度演讲，承诺在理解单语义性与模型缩放之间的联系方面取得进展。该[活动](https://lu.ma/p5ctl2u6)提供了详细信息和注册方式。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **技能持久化带来的问题**：讨论显示，尽管用户尝试“告诉 OI 创建一个新技能”，但 **OpenInterpreter** 仍缺乏跨会话保留技能的能力。为了规避这个问题，建议将脚本保存并存储作为权宜之计。

- **显微镜下的 RAG**：人们对 **检索增强生成（RAG）** 持怀疑态度，更倾向于使用传统的 Embedding/向量数据库，理由是其可靠性更高，尽管 Token 成本也更高。

- **数据隐私成为焦点**：对 OpenAI 数据隐私的担忧得到了缓解，保证了与 OpenAI API 的通信保持机密，同时建议运行本地模型以获得额外的安全性。

- **跨模型兼容性查询**：关于将 **O1 dev preview** 与 **Anthropic** 等其他大型语言模型集成的咨询引发了兼容性问题，特别是对视觉模型的必要性以及在某些操作系统上可能出现的无限循环。

- **开发者的语音助手**：一个 **Terminal Voice Assistant** 的 GitHub 项目链接引发了人们对 **01** 是否可以实现类似功能的兴趣，指向了工程师潜在的开发工具。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Hotz 发起 Tqdm 替代品挑战**：[George Hotz](https://discord.com/channels/1068976834382925865/1068976834928193609/1247640237350326363) 出资 **200 美元** 征集极简的 tqdm 替代品，引发了一阵活跃，Trirac 提交了一个 PR，尽管备注提到其在高速下的 it/s 速率并非最优。
- **Tinygrad 统计数据缺失之谜**：Hotz 询问为何 [stats.tinygrad.org](https://stats.tinygrad.org) 网站目前显示 **404 错误**，引发了关于该网站访问性的讨论。
- **邀请改进 Tinygrad 文档**：宣布了 Tinygrad 文档的更新，包括关于训练的新章节和库结构图，并向社区征集进一步的内容创意（[George Hotz](https://discord.com/channels/1068976834382925865/1068976834928193609/1247640237350326363)）。
- **Tinygrad：冲刺前的规格制定**：Hotz 提供的悬赏旨在起草 **Tinygrad 规范**，并承诺在最终确定后，可以在大约 **两个月** 内重新实现，这同时也作为员工筛选过程。
- **破译 CUDA-to-Python**：讨论集中在将 CUDA 调试输出连接到 Python 代码的复杂性上，这是 Tinygrad v1.0 的关键特性，现有的 PR 尚未合并（[George Hotz](https://discord.class/1068976834382925865/1070745817025106080/1247984470146027642)）。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**过时文档引发混乱**：**LangChain 和 OpenAI** 的文档问题引起了成员们的注意，他们指出由于 API 更新导致了显著的差异。有建议指出工程师应直接查看主代码库以获取最新的见解。

**数据库之争：MongoDB vs. Chroma DB**：当一名工程师考虑使用 MongoDB 进行向量存储时，随后的澄清说明了 MongoDB 的用途是存储 JSON 而非 embeddings，并建议询问者寻求 MongoDB 的帮助或咨询 ChatGPT。

**Verba：显微镜下的 RAG**：社区对 [Verba](https://github.com/weaviate/Verba)（一个由 Weaviate 驱动的 RAG 聊天机器人）产生了兴趣，并征求用户的使用体验，这表明了对 Weaviate 检索增强能力的探索。

**SQL Agent 让用户感到困惑**：SQL Agent 无法提供最终答案的问题浮出水面，引发了关于在厌恶非功能组件的环境中如何排查这种神秘行为的讨论。

**基于 LangChain 的图谱知识**：一名工程师展示了一个 [LangChain 指南](https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/)，专注于从非结构化文本构建知识图谱，并引发了关于将 `LLMGraphTransformer` 与 Ollama 模型集成的咨询，这体现了对增强知识合成的不断追求。

**VisualAgents 开启拖拽式 LLM 模式**：通过 [YouTube 视频](https://www.youtube.com/watch?v=IVFsANcfqaA) 进行的 VisualAgents 现场演示强调了排列 Agent 流模式所涉及的创作过程，反映了 LLM 链管理向更直观界面发展的趋势。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Rope Scaling 在 OpenRouter 上遇到障碍**：成员们强调了在 **OpenRouter** 中集成 **rope scaling** 的问题，建议通过本地部署来规避 GPU 限制。

- **Codestral 在代码专业化方面落后**：有人建议不要将 **Codestral** 用于代码专业化，并推荐了在 <#1230206720052297888> 中详细介绍的更高效的模型。

- **识别 502 错误背后的元凶**：工程师们解决了 OpenRouter 的 502 Bad Gateway 错误，将问题追溯到 `messages` 中 `content` 的格式，而非服务器容量或请求量。

- **停机期间的杂乱模型混杂**：在处理来自 **Nous Research**、**Mistral**、**Cognitive Computations**、**Microsoft** 和 **Meta-Llama** 的各种模型时出现了 **502 错误**，重点在于问题源于消息内容的格式化。

- **寻求更高代码效率的替代方案**：建议寻找高效代码专业化的工程师考虑 **Codestral** 之外的更多性能导向的替代方案，并提示查看频道中提到的特定模型。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **在日历上标记 AI 安全活动**：**Human Feedback Foundation** 活动定于 6 月 11 日举行；门票可从 [Eventbrite](https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator) 获取。活动重点将涵盖 AI 治理与安全，并通过协作的开源环境进行强化。
- **从 AI 专家处收集见解**：查看 [Human Feedback Foundation 的 YouTube 频道](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg)，获取来自多伦多大学、斯坦福大学和 OpenAI 的学术界及行业领袖关于将人类反馈集成到 AI 开发中的见解。
- **LLM 阅读小组 Discord 访问受限**：有人请求为 LLM 阅读小组设立独立的 Discord，但由于隐私设置，直接邀请受到阻碍，这意味着对感兴趣的人员需要另行安排访问。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **矢量技术随 RISC-V 共同进步**：[RISC-V 矢量处理](https://www.youtube.com/watch?v=Ozj_xU0rSyY) 达到了一个重要的里程碑，**1.0 RISC-V Vector Specification** 现已获得批准。链接视频深入探讨了早期的芯片实现，表明 CPU 设计中存在充足的创新机会。

- **AI 的生存威胁受到关注**：[Right to Warn AI](https://righttowarn.ai) 项目对 AI 技术可能带来的生存威胁发出了警报，提倡需要远超企业治理的监管。它引发了对 AI 相关风险的担忧，如不平等、虚假信息以及潜在的人类灭绝。





---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **探索 "Sauerkraut Gemma" 前景**：表达了为德语复制 **PaliGemma** 模型的兴趣，暂定名为 *"Sauerkraut Gemma"*，思路是直接替换 Gemma 的 base 进行适配。
- **PaliGemma 模型作为模板**：参考 [PaliGemma-3B-Chat-v0.2](https://huggingface.co/BUAADreamer/PaliGemma-3B-Chat-v0.2) 模型，一位成员提出了在数据集翻译后，“冻结视觉并训练聊天 (freezing the vision and training the chat)” 的策略，用于开发德语对应版本。



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **AI 学习枢纽发布**：由 William Brown 策划的 [GenAI Handbook](http://genai-handbook.github.io/) 被重点推荐，它是 AI 工程师寻求全面理解现代 AI 系统的教科书式指南，格式对用户友好。



---


**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。


---


**Datasette - LLM (@SimonW) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。


---


**YAIG (a16z Infra) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1247645798016225362)** (46 messages🔥): 

- **关于幻灯片/链接访问和 COVID 延迟的讨论**：一名成员请求访问研讨会的所有幻灯片和链接，而另一名用户提到因感染 COVID 错过了最近的演示。建议包括填写 [Modal credits form](https://bit.ly/modal-credits) 以获取来自 Modal 的 $500 奖励额度。

- **Gemma 抄袭警报**：一名用户揭露 [gemma-2B-10M](https://github.com/mustafaaljadery/gemma-2B-10M) 仓库是抄袭自 [InfiniTransformer](https://github.com/Beomi/InfiniTransformer)，仅对注释和格式进行了微小修改。

- **最适合 Fine-Tuning 的代码模型**：大家对寻找最适合 Fine-Tuning 的代码模型表现出兴趣。讨论中指出目前缺乏明确的优质资源或模型。

- **合成数据生成 (Synthetic Data Generation)**：用户讨论了合成数据生成工具的困难，提到了 [distilabel](https://github.com/argilla-io/distilabel/tree/main) 和 Python 编码作为选项。一名用户指出使用 distilabel 的 evol pipeline 生成的数据质量存在问题。

- **Mistral Fine-Tuning Hackathon**：[Mistral Fine-Tuning Hackathon](https://mistral.ai/news/2024-ft-hackathon/) 的公告让用户感到兴奋，他们表达了组队参赛的兴趣。另一个相关活动通过 [X (Twitter) 链接](https://x.com/dchaplot/status/1798368883172421765?t=BmEAh2YFNBTIKFjRvDwAeQ&s=19) 被提及。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1247922676547325982)** (1 messages): 

- **用 AI 复制家人的 Persona**：一名成员分享了一个有趣的用例，即复制家人（特别是年迈父母）的 Persona，以保留他们的文本、笔记、观点和声音。他们建议混合使用多种方法：**Fine-tuning** 用于捕捉他们的言谈举止，**RAG** 用于语境化新话题，以及 **Prompt Engineering** 用于调节回复。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1247793111917138023)** (1 messages): 

~~~html
- **erniesg discusses Hainan departure and VPN setup**: *"im actually leaving hainan on 7 june"* and adds a light-hearted comment about ensuring VPN access for coding. This suggests ongoing preparation for remote work or coding sessions.
~~~

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1247630777450106921)** (30 messages🔥): 

- **Modal 额度混淆引发热烈讨论**：多名用户反映未收到承诺的额外 500 美元额度。Charles 澄清了额度的分配方式，并为未解决的问题提供邮件个性化支持。

- **过时的文档导致问题**：[指向过时文档的链接](https://modal.com/docs/examples/llm-finetuning)促使用户注意到其与当前实践的不一致。Charles 承认了该问题并提到更新文档的计划。

- **子进程（Subprocesses）作为非 Python 任务的变通方案**：用户探讨了使用 GPU 运行 shell 脚本，指出虽然 Modal 并非为此原生设计，但使用 subprocess 是可行的变通方案。Hamel 强调了这一点，建议调用 subprocess 或类似命令。

- **部署 Python Shiny 应用遇到障碍**：Iain 在部署 Shiny 应用时面临挑战，他参考了一个 [GitHub 仓库](https://github.com/iainmwallace/modal_shiny)并讨论了一个 streamlit 示例。Charles 建议在 Modal Slack 中提出问题以便更好地排查。

- **Modal 隐私政策问题被重定向至 Google**：当被问及 Modal 的隐私政策时，Hamel 直接指向了 [隐私政策的 Google 搜索结果](https://www.google.com/search?q=privacy+policy+modal+labs)，表示可以在网上找到。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1247985108850442292)** (1 messages): 

- **终极 LLM 资源指南**：一位成员分享了他们收藏的 LLM 讲解合集，涵盖了 vLLM, SSMs, DPO 和 QLoRA 等主题。欲了解更多信息，他们提供了[指南链接](http://genai-handbook.github.io)。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1247634880309891082)** (7 messages): 

- **初创公司澄清额度混淆**：一位成员澄清说，该公司仅提供一次 200 美元的额度，并对意外添加额外额度造成的混淆表示歉意。他们提到自己是一家自筹资金（bootstrapped）的初创公司，资金并不雄厚。

- **CUDA 库升级问题**：用户在 Jarvis Labs 容器中安装 `xformers` PIP 模块时遇到了升级 CUDA 库的问题。错误提示检测到的 **CUDA 版本 (11.8)** 与编译 PyTorch 所用的版本 (12.1) 不匹配。

- **初始额度已确认收到**：多名用户确认收到了初始的 200 额度，但没有获得额外额度。他们对该平台表示满意，称赞其便利性和良好的服务。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1247627395981377657)** (35 messages🔥): 

- **关于额度和表格的混淆**：尽管按时填写了表格，几位用户仍对未收到额度表示担忧。[tddammo](https://example.com) 向他们保证，很快会发布第二个表格以补发错过的额度。

- **搞笑的 HTML 修改**：*ayhanfuat* 关于使用额度训练 GPT-6 的玩笑评论原来是一个巧妙的 HTML 修改，在澄清玩笑之前引起了短暂的惊慌。这次交流缓解了紧张气氛，为对话增添了幽默感。

- **GPU 愿望清单**：*osanseviero* 和 *charles_irl* 幽默地指出了对 GPU 日益增长的需求，表明社区对计算资源有着持续的需求。*charles_irl* 调侃道：“让百台 GPU 齐放（let a hundred GPUs bloom）。”

- **致谢与解决**：在整个讨论串中，[tddammo](https://example.com) 尽职地处理了用户关于缺失额度的担忧。几位用户感谢他及时解决了他们的问题。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1247629712176582837)** (16 messages🔥): 

- **无需设置组织即可兑换 Replicate 额度**：用户讨论了**不需要创建组织**即可兑换 Replicate 额度。然而，为了团队协作，建议先设置一个 GitHub 组织。
- **额度兑换说明及问题**：用户分享了收到 **Replicate 额度兑换邮件**的经历以及领取额度的步骤。一些用户遇到了兑换的额度未反映在账单中的问题；建议这些用户私信（DM）其电子邮件和用户名以寻求帮助。
- **额度有效期**：一位成员询问了 **Replicate 额度的到期日期**，得到的答复是**额度有效期为 24 个月**。这为用户使用额度提供了充足的时间。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1247687054826012693)** (21 messages🔥): 

- **LangSmith 额度发放引发混乱**：多名用户报告称，尽管按照指示设置了账单，但仍未收到 250 美元的 LangSmith 额度。一些用户提到，不同类型的额度（如初始注册额度与 Beta 额度）混淆不清。

- **Pinecone 命名空间检索问题**：一位用户寻求帮助，希望通过创建多个检索器并按文档名称过滤，来提高包含多个文档的 Pinecone 命名空间中的检索性能。该用户在为此目的配置 LCEL 链时遇到了困难。

- **LangSmith 输入缺失之谜已解**：一位正在调试 `@traceable` 装饰器的用户发现，其 LLM 调用输出已被记录，但输入未被记录。在意识到该装饰器将函数的参数记录为输入后，问题得到解决，因此在函数定义中包含参数即可捕获输入。参考了 [LangSmith 文档](https://docs.smith.langchain.com/how_to_guides/tracing/annotate_code#use-traceable--traceable)。

- **Mastering LLMs 课程额度**：一位用户提到成功看到了标记为“Mastering LLMs Course Credit”的课程额度。关于 Beta 额度的可用性以及如何获取，存在一些困惑。

- **普遍不满与寻求帮助**：包括提到 Hugo 在内的几位用户表达了沮丧，并就尽管进行了正确设置但仍未收到额度的问题寻求帮助。这一问题引发了多次关于澄清和协助解决额度分配问题的请求。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[whitaker_napkin_math](https://discord.com/channels/1238365980128706560/1242223332695478332/1247835046606536765)** (2 messages): 

- **Valgrind 内存检查器依然有用**：一位成员询问了 **Valgrind** 在检查内存使用和泄漏方面的当前实用性，暗示了其过去的重要性。另一位成员被直接点名就此事发表意见。

- **请求支持 Hamel Husain 的演讲**：一位成员分享了 [Hamel Husain 的 Twitter 帖子链接](https://x.com/hamelhusain/status/1798353336145674483)，并敦促其他人帮助该演讲获得更多观众。消息中包含一个敬礼表情符号，以强调支持请求。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-4](https://discord.com/channels/1238365980128706560/1242223495673286737/1247625969280221184)** (174 messages🔥🔥): 

- **规模和 GPU 无法违背物理定律**：一位用户开玩笑说，扩展计算实例并期待奇迹发生，而另一位用户幽默地指出：“我们不能贿赂物理定律！”，随后有人建议“多买点 GPU 就好了”。
- **Modal 凭借额度大放送大放异彩**：多名用户强调了 Modal 令人印象深刻的额度方案，称“Modal 简直太强大了”。这引发了关于在 Modal 上运行任何任务以获取额外额度的讨论，并参考了 [Modal 的示例文档](https://modal.com/docs/examples)。
- **有价值的链接合集**：用户分享了各种有用的资源，例如用于将 LoRA 合并到基础模型的 Axolotl、Dan 的 [Huggingface 仓库](https://huggingface.co/dansbecker/conference-demo)、[Predibase 的索引](https://predibase.com/fine-tuning-index) 以及 [LinkedIn](https://www.linkedin.com/in/travisaddair/) 上的 TGAddair。
- **关于推理优化和量化的讨论**：进行了关于优化 CPU 推理、llama.cpp 使用经验的深入讨论，并解释了为什么量化方法（如 QLoRA）在计算密集型任务中可能会变慢。
- **答疑时间公告与反馈**：TGAddair 宣布了 [Predibase 的答疑时间 (office hours)](https://discord.com/channels/1238365980128706560/1242223495673286737/1247637288146698271)，用户留下了反馈，并询问了关于微调策略和性能影响的问题。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1248012391250268362)** (1 messages): 

- **LangChain 链接失效**：一位用户分享了 [LangChain 多模态 RAG notebook 的链接](https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb) 并询问其是否无法工作。这似乎表明所提供的资源存在某些故障排除或功能性问题。

### **LLM Finetuning (Hamel + Dan) ▷ #[yang_mistral_finetuning](https://discord.com/channels/1238365980128706560/1242224842053521459/1247934402676133939)** (108 messages🔥🔥): 

- **Mistral 的 API 发布与黑客松完美契合**：成员们兴奋地讨论了 [Mistral AI 定制化公告](https://mistral.ai/news/customization/)，注意到其发布时间与黑客松及演示活动不谋而合。
- **新的 Mistral 项目和资源**：分享了大量资源，包括 Mistral 的 [微调教程](https://docs.mistral.ai/capabilities/finetuning/)、[微调仓库](https://github.com/mistralai/mistral-finetune) 以及相关的 [YouTube 演示](https://youtu.be/zXFxmI9f06M)。
- **关于 Mistral 独特性和能力的讨论**：成员们辩论了 Mistral 的垂直整合策略、其 API 相对于 Axolotl 等工具的优势，以及 Prompt 模板和空格处理的重要性等细节。
- **内存管理和训练问题**：针对在 Colab 等平台上处理漫长训练时间的问题进行了提问和讨论，建议保存 Checkpoints，并深入探讨了 Mistral 的 API 如何通过量化等潜在优化手段来减轻内存错误。
- **对大显示器提高生产力的兴趣**：侧边讨论中包括对巨型显示器的热烈推荐，特别是将 55 英寸电视用作主显示器。为有兴趣升级配置的人分享了链接：[Samsung-GQ55QN95BAT](https://www.mediamarkt.de/de/product/_samsung-gq55qn95bat-neo-qled-tv-flat-55-zoll-138-cm-uhd-4k-smart-tv-2793513.html)。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[gradio](https://discord.com/channels/1238365980128706560/1242283474300174346/1247715125423767583)** (1 messages): 

- **微调工作流简单但表现欠佳**：一位成员评论说，使用 Predibase 示例进行微调非常容易，称其为“简单的工作流”。然而，他们对目前微调过程的质量表示失望，称尽管完成了一次完整的迭代，但效果“很糟糕”。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1247688274047143986)** (21 messages🔥): 

- **解决使用 `val_set_size` 验证数据集的问题**：成员们讨论了 `val_set_size` 参数是提供对验证/测试数据集的访问，还是需要显式指定 `test_datasets`。

- **本地 Axolotl 安装挑战已解决**：一位用户分享了在按照建议为微型模型搭建测试环境时，本地安装 Axolotl 遇到的挫折。他们最终成功使用了[这份替代指南](https://www.superteams.ai/blog/a-definitive-guide-to-fine-tuning-llms-using-axolotl-and-llama-factory)，其中包含了 CUDA 和 PyTorch 设置的具体步骤。

- **Docker 作为无忧的 Axolotl 解决方案**：用户建议使用 Axolotl 官方 Docker 镜像以避免依赖问题并提高性能，并向他人推荐了[一份关于使用 Docker 和 VSCode 进行调试的指南](https://openaccess-ai-collective.github.io/axolotl/docs/debugging.html#debugging-with-docker)。

- **Axolotl 安装中的 CUDA 版本问题**：成员们报告了在具有不同 CUDA 版本的机器上安装 Axolotl 的不同成功率，并指出某些特定依赖项可能需要手动调整。讨论了为每个 CUDA 版本维护 `requirements.txt` 的可能性。

- **概述双 GPU 卡死问题**：对于遇到双 GPU 卡死的情况，一位成员建议将 `NCCL_P2P_DISABLE=1` 作为解决方案，尽管这在专业级或云端机器上较少出现。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1247652004562735165)** (11 messages🔥): 

- **揭秘 cpu_ram_efficient_loading**：讨论澄清了 `cpu_ram_efficient_loading` 允许将模型的分片加载到各个 GPU 上，而不是在所有 GPU 上加载整个模型。当设置为 `false` 时，只有第一个 Worker 保留完整的模型权重，而其他 Worker 保留不含权重的骨架，并将必要的权重分发到 **FSDP** 的各个层。

- **关于“进程 (process)”的术语澄清**：澄清了“进程”是指 GPU 编号，而不是 GPU 的分区。例如，当编号从 0 开始时，进程 1 对应于第二个 GPU。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1247825230517501993)** (1 messages): 

- **YAML 健全性检查：寻求帮助**：一位用户正在寻求帮助，以确定 YAML 健全性检查具体在何处运行。他们还在询问是否有人为 ADO 配置构建了用户友好的 GUI。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1247629487152169101)** (29 messages🔥): 

- **CUDA 书籍推荐引发关注**：成员们讨论了推荐的 CUDA 书籍，其中一人推荐了 [Amazon](https://a.co/d/1PbUJK7) 上一本好评如潮的书。Charles 确认这就是之前演示中提到的那本书。
  
- **Modal 设置与使用咨询**：用户分享了关于 Modal 的经验和问题，imaure 询问快速设置是否可以获得 `$500 credit`。Charles 确认道：“当然可以！这些额度大约需要一周时间发放 💚”。

- **在 Modal 上运行 Sentence Transformer 嵌入模型**：imaure 询问在 Modal 上托管 Sentence Transformer 嵌入模型的可行性，Charles 确认了其可行性并分享了一个[示例链接](https://modal.com/blog/embedding-wikipedia)。

- **Modal 示例入门**：Charles 和 andrewcka 为遇到问题的初学者提供了关于使用基础 Modal 示例的帮助。例如，他们引导用户查看 [Modal 的 hello world 示例](https://github.com/modal-labs/modal-examples/blob/main/01_getting_started/hello_world.py)，并协助排查了特定的代码错误。

- **账单与使用说明**：用户寻求关于账单的澄清，特别是停止运行的应用是否会产生费用。Charles 保证停止的应用不会产生费用，并指出：“我们对于意外账单通常是非常宽容的。”
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langchain-langsmith](https://discord.com/channels/1238365980128706560/1242564256914870384/1247680107875663954)** (3 messages): 

- **Langsmith 设置极其简便**：在 **Langchain** 中使用 **Langsmith** 不需要额外的代码。只需设置环境变量，它就会自动追踪一切。

- **Langsmith 额度可用**：有关获取 **Langsmith credits** 的信息可以在此 [Discord 链接](https://discord.com/channels/1238365980128706560/1241167367040405544/1247687054826012693)中找到。

- **Langchain 检索性能困扰**：一位成员在文档数量增加时，遇到了 Pinecone 命名空间中检索性能下降的问题。他们正考虑创建多个按文档名称过滤的检索器，但在构建实现此功能的 LCEL 链时遇到了困难。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1247632624261009548)** (51 messages🔥): 

- **错过额度申请截止日期，正尝试协调**：尽管填写表单的截止日期是 5 月 29 日，**Danbecker** 仍在努力为错过的人争取额度。他强调：“我正在整理一份未及时填写表单的人员名单，并将其发送给各平台。”

- **可用额度及来源列表**：一位用户分享了来自各个平台的详细额度列表，包括来自 **HuggingFace** 和 **Replicate** 的 $501，来自 **OpenAI** 和 **Modal** 的 $500（如果在 6 月 10 日前使用，Modal 还会再提供 $500）等。**Fireworks** 被确认提供 $250。

- **过期与账户设置问题**：用户讨论了额度的有效期，例如 **Modal** 额度一年后过期，而 **OpenAI** 额度三个月后过期。他们强调了及时设置账户以接收额度的重要性。

- **处理不完整的表单和逾期提交**：**Danbecker** 和其他人重申，平台需要准确的电子邮件地址和账户 ID 才能发放额度。例如，表单中缺失 **OpenAI Org-ID** 之类的错误在截止日期后无法补救。

- **额度发放的当前状态与更新**：更新了发放状态，确认了来自 **Langsmith** 和 **Fireworks** 的额度。建议用户创建必要的平台账户，如果尚未收到额度，请进行跟进。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[strien_handlingdata](https://discord.com/channels/1238365980128706560/1243773476301443073/1247938813670457477)** (158 条消息🔥🔥): 

- **对合成数据准备的热情**：多位用户对关于合成数据准备（Synthetic Data Preparation）的演讲表示兴奋，称其为一个“令人兴奋”的话题。一位用户提到，他更倾向于将“ML Librarian”作为理想职业的概念。

- **分享的链接和工具**：用户分享了各种与数据处理相关的工具和链接，包括 [Lilac](https://www.lilacml.com/)、Huggingface 的 [dataset security](https://huggingface.co/docs/hub/en/security-malware)，以及像 [Outlines](https://github.com/outlines-dev/outlines) 这样结构化文本生成的示例。

- **讨论数据集生成**：参与者讨论了使用 **GPT-4** 结合人工评估来创建合成数据集，强调了高昂的成本和潜在的收益。一位用户提到结合使用 **DPO** 和 **PPO** 方法来提高数据集质量。

- **对知识图谱的热情**：用户表达了对构建 Knowledge Graphs 以改进数据结构化和检索的兴趣，提到了 [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_query_engine/) 和用于增强 RAG 的 LangChain notebook。

- **关于 RLHF 及其替代方案的关键资源**：对话引用了 Argilla 博客系列中关于 [RLHF 及其替代方案](https://argilla.io/blog/mantisnlp-rlhf-part-9) 的内容，用户请求进一步澄清如何使用 Distilabel 生成 JSONL 格式的合成聊天数据集。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1247841028061397002)** (28 条消息🔥): 

- **账户额度检查环节：社区的共同努力**：多位成员报告称，尽管填写了必要表格，但在账户中未看到分配的额度。**Aravindputrevu** 积极与每位用户沟通，询问账号 ID 并跟进以确保问题得到解决，最终许多人确认额度已成功添加：“搞定了，谢谢 🙏”。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[braintrust](https://discord.com/channels/1238365980128706560/1245407617031999581/1247626352425697361)** (26 条消息🔥): 

- **Braintrust 入门页面更新**：成员们讨论了一项新变化，即如果用户没有项目，默认会跳转到入门（onboarding）页面。*“我们刚刚推送了一个小改动，默认将你引导至入门页面……希望这有所帮助。”*

- **明确额度限制**：一位用户不确定在哪里查看他们的额度，随后得到澄清：UI 中没有显示正式的额度系统，而是为成员提供 *“3 个月的无限访问权限。”*

- **Braintrust 装饰器的功能**：一位成员询问为什么他们的 LLM 调用输入没有被 Braintrust 装饰器捕获。经澄清，必须使用 `wrap_openai` 包装 `OpenAI` 客户端才能记录 OpenAI 调用，而不是使用仅记录函数输入/输出的 `@traced`。

- **解释追踪方法**：详细解释了三种追踪（tracing）方法：`wrap_openai`、`@traced` 以及用于记录特定信息的 spans。*“`wrap_openai` 和 `@traced` 都是基于 spans 实现的。”*

- **将 Braintrust 集成到其他平台**：一位成员提到尝试将 Braintrust 集成到 ZenML 中，并指出了包装所有可能函数的复杂性。*“除非我们要求用户使用某种实现了该功能的辅助函数，否则使用 OpenAI 包装器可能行不通。”*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1247809404716449874)** (2 条消息): 

- **国际问候**：一位用户向成员打招呼，表示他们来自葡萄牙的 Peniche 🇵🇹。另一位成员回应了来自某个未标明位置（标记为黑旗表情 🏴）的问候。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1247649710236237977)** (8 条消息🔥): 

- **微调模型中的句点重复问题**：一位用户提出了关于 Finetuned Models 在输出末尾生成重复句点或空格的问题，并寻求缓解该问题的建议。

- **关于额度（Credits）的社区支持**：出现了一个额度未显示的问题，并已迅速解决。建议如果问题仍然存在，请联系支持团队。

- **对 Predibase 方法论的赞赏**：用户对 Predibase 重用 Base Models 并将输入数据格式简化为仅 Prompt-Completion 对的方法表示赞赏。这种简化帮助用户避免了在 Prompt Templates 和 System Messages 方面面临的过多选择。

- **Prompt Templates 的验证**：虽然简化 Prompt Templates 很有益处，但一位用户强调了允许验证以确保 Backend 正确使用的重要性。这种验证将帮助用户识别并排除与模板相关的问题。

- **Epoch 灵活性和数据过滤建议**：一位用户建议开放更多训练 Epoch 的权限，例如运行多达 7 个 Epoch 并能选择表现最好的一个以避免 Overfitting。他们还建议增加一个用于过滤常见数据错误的 UI 引导，类似于 Cohere 的做法，以增强 Fine-tuning 流程。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[career-questions-and-stories](https://discord.com/channels/1238365980128706560/1245816565073576037/1247643889603444776)** (5 条消息): 

```html
- **资深开发者投身 LLMs**：一位拥有 17 年经验的软件开发者分享了他们学习 LLMs 的历程。他们表达了兴奋之情，并就本地运行模型寻求建议，提到了潜在的 Fintech 应用。

- **推荐 Fastbook 作为 LLM 基础**：一位成员推荐了 [GitHub 上的 fast.ai 免费书籍](https://github.com/fastai/fastbook)，用于了解 Deep Learning 基础知识，特别是针对软件工程师。他们强调该书包含大量代码和直观理解，数学内容极少。

- **工程师的社区学习**：一位用户强调了社区对于学习 LLMs 等复杂话题的重要性。他们分享了在罗马尼亚学习社区 [Baza7](https://new.baza7.ro/) 的经验，该社区提供跨各种业务职能的实用知识。
```
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1247790180866326580)** (3 条消息): 

- **OpenPipe 额度发放进行中**：一位成员询问了 **OpenPipe 额度发放** 的状态。另一位成员确认他们尚未收到，表示这是 *"最后一个了，其他都好了！"*，而另一位成员也确认了同样的状态，称其为 *"我还没拿到的最后一个。"*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1247653281480839218)** (51 条消息🔥): 

```html
- **分享优化 LLMs 指南**：OpenAI 的初创企业解决方案团队分享了一份关于[优化 LLMs 准确性的新指南](https://platform.openai.com/docs/guides/optimizing-llm-accuracy)，重点关注 Prompt Engineering、RAG、Fine-tuning 以及如何确定生产环境的达标标准。还推荐了一个 YouTube [DevDay 演讲](https://www.youtube.com/watch?v=ahnGLM-RC1Y)以获取更多见解。
- **Fine-tuning 中的挑战与需求**：用户强调了 [Fine-tuning 指南](https://platform.openai.com/docs/guides/fine-tuning/when-to-use-fine-tuning)的实用性以及对 Fine-tuning 流程改进的需求。多位用户请求了诸如重试按钮、不打乱数据集的选项，以及通过 Fine-tuning API 获得更好的 Tool/Function Calling 输出的解决方案。
- **额度与 Rate Limits 担忧**：用户讨论了关于 OpenAI 额度申请和过期的问题。一些人报告必须激活账单才能收到额度，另一些人则强调由于 Rate Limits 的限制，很难在给定的 3 个月内用完 500 美元的额度。
- **Rate Limits 与 API 支出困惑**：用户质疑额度是否计入增加 Rate Limits 所需的 API 支出，并分享了可能需要支付小额费用才能更快解锁更高 Rate Limits 的见解。关于 OpenAI 是否会解决这一问题以提供更公平解决方案的讨论仍在继续。
- **GPT-4 模型的可用性与功能**：用户提到尽管有额度，但在访问 GPT-4 和 GPT-4o 模型时仍面临挑战，并推测可能只有在第一张付费发票之后才会解锁访问权限。分享的经验表明，额度适用于账单概览页面的余额。
```

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1247643814202577027)** (417 messages🔥🔥🔥): 

- **Unsloth 不支持 Medusa 微调**：成员们讨论了在 Unsloth 上使用 Medusa 进行微调的可能性，但确认目前尚不支持。Medusa 的相关链接分享在[这里](https://github.com/FasterDecoding/Medusa)。

- **持续预训练（Continued pretraining）的限制与技巧**：成员们辩论了在没有 Optimizer States 的情况下进行持续预训练的可行性，并分享了通过混入旧数据来缓解灾难性遗忘（catastrophic forgetting）的见解。他们还提到使用 redpajama 等工具以获得更好的效果。

- **内存管理与优化**：参与者讨论了训练期间 VRAM 使用量激增的问题，以及更有效地管理内存的策略，例如使用余弦学习率（cosine learning rates）和卸载（offloading）特定 Token。一个值得注意的建议是可能在 H100 GPU 上扩展 Context Length。

- **Colab 的局限性与替代方案**：强调了 Colab 在进行大规模 GPU 训练（如微调 LLaMA 70B）时的局限性。建议包括使用其他云服务或设置高效的 Docker 容器来管理训练会话。

- **Unsloth 更新中的新功能**：分享了 Unsloth 的最新更新，包括将 lm_head/embed_tokens 卸载到磁盘以及自动 Tokenizer 修复。此外，还提供了持续预训练的指南，强调了在速度和 VRAM 使用方面的改进。
  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1247651391212621824)** (1 messages): 

- **预训练变得更快、更精简**：“你现在可以比 HF+FA2 快 2 倍的速度[持续预训练](https://github.com/unslothai/unsloth/releases/tag/June-2024) LLM，且节省 50% 的 VRAM。”更多详情请见 [Unsloth AI 博客](https://unsloth.ai/blog/contpretraining)。
- **提供 Mistral v0.3 的免费 Notebook**：访问我们的[持续预训练 Notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing)和[文本补全 Notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)进行实操体验。
- **宣布即将推出的功能**：预计将支持包括 Stable Diffusion、多模态（Multimodal）、Mixtral、8-bit 等在内的所有模型。Multi-GPU 支持也在计划中。
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1247829744586850356)** (6 messages): 

- **预计 Multi-node 支持将延迟**：一名成员暗示“Multi-node 支持将有点复杂”，表明其实现可能会面临延迟。这预示着该项目未来可能面临挑战。

- **寻找开源 TTS 模型**：一名成员询问是否有人为涉及 waifu 伴侣应用/RPG 的副业项目找到了带有参考代码的“优秀开源 TTS 模型”。另一名成员推荐了“xttsv2 -rvc pipeline”，得到了原询问者的感谢。
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1247655619499004069)** (220 messages🔥🔥): 

```html
- **Multi-GPU 支持已上线，Multi-Node 即将推出**：一位用户询问 Unsloth 的 Multi-node 训练支持情况，获知该功能已在路线图中但尚未推出。他们对使用 Multi-GPU 设置进行 70B 微调的潜力表示兴奋。
- **VLLM 服务器设置简化**：关于设置 VLLM 服务器的有用讨论，包括命令和[安装文档](https://docs.vllm.ai/en/stable/getting_started/installation.html)链接。VLLM 服务器可以作为 OpenAI API 端点的无缝替代方案，适用于在本地托管微调后的 LLM。
- **持续预训练中的高 Loss 问题**：一位用户报告在持续预训练模型时初始 Loss 很高，尽管之前的训练成功且 Loss 较低。他们分享了用于加载和训练具有特定配置的模型的详细代码片段。
- **使用 LoRA Adapter 进行微调**：用户讨论了加载和继续使用 LoRA Adapter 进行微调的问题。一个可行的解决方案是创建一个新的 PEFT 模型，然后附加现有的 Adapter，尽管 Wiki 上的方法似乎仍有问题。
- **处理多个模型的 GPU 显存**：一位用户询问如何高效地从 GPU 显存中移除模型以运行隔夜训练循环。另一位建议使用 `del` 删除模型和 Tokenizer 对象，以在不重启内核的情况下释放 GPU 显存。
```
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1247631953583411380)** (317 条消息🔥🔥): 

- **Perplexity AI 正面临宕机困扰**：多名用户报告了 Perplexity AI 宕机或模型选择出现问题。一位用户感叹道：“我已经等了 8 小时了，快疯了。”
- **模型对比与功能**：有一场关于 **ChatGPT-4o** 和 **Claude 3** 的详细对比讨论。一名成员指出：“Claude 3 是我用过最烂的 AI”，而另一名成员则强调 Perplexity 经常使用自己的搜索索引。
- **AI 演示技巧**：成员们讨论了演示文稿，建议包含 AI 的**优势与风险**，并**侧重于 LLMs**（如 GPT）。分享了[视频](https://youtu.be/wjZofJX0v4M?feature=shared)链接和[搜索概览](https://www.perplexity.ai/search/how-does-perplexity-Qm71LBYBSkOApKNFCISS0Q)以帮助完善内容。
- **对 Perplexity AI 界面的不满**：用户报告了模型选择 bug 以及**图像生成功能**的问题。一位用户提到：“当我要求 Perplexity 生成图像时，它却给了我一份如何自己画图的详细描述。”
- **Perplexity AI 独特的 SEO 排名**：讨论透露 Perplexity AI 使用自己的爬虫进行索引，而不是使用 Google 或 Bing 等第三方服务。这种方法产生的结果“质量更高”，且较少经过 SEO 优化。
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1247657879725674679)** (13 条消息🔥): 

- **查看 dailyfocus_daily 文章**：成员们分享了一个 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/httpsdiscordcom-repeat-this-p6zDFgNJS5Wn4D4YdmeEGg)链接，供进一步阅读或讨论。
- **重大宕机分析**：一位用户指向了一篇详细介绍影响 DCcT 的[重大宕机](https://www.perplexity.ai/page/Major-Outage-of-DCcT_vXARMmWZl8KCWB8Jg)的文章，提供了关于该事件的关键见解。
- **可分享线程提醒**：Perplexity AI 机器人多次提醒用户确保其线程已标记为可分享。它包含了一个[特定的附件链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)，详细说明了如何操作。
- **参考话题的延伸阅读**：用户分享了多个搜索结果链接，以便对不同主题进行更深入的探索，例如一个[关于 Gonz](https://www.perplexity.ai/search/referring-to-the-GonzYU0ZTU6cTHhv5qLBXg) 的链接和另一个[与 Kuhn 相关](https://www.perplexity.ai/search/who-was-Kuhn-mKejsw.LRjOMtIPJtQjpCQ)的搜索。这些都是对正在进行的讨论的贡献。
- **分享 Bitcoin 页面**：成员们被引导至 Perplexity AI 上一个详细的 [Bitcoin 页面](https://www.perplexity.ai/page/Bitcoin-WzdgAV4KQiqtb4k0q4RHRw)，可能是为了丰富他们的理解或用于讨论。
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1247664700683190394)** (6 条消息): 

- **vLLM 和 Neural Magic 的开放办公时间 (Open Office Hours)**：vLLM 和 Neural Magic 每两周举办一次开放办公时间，以回答关于优化的 LLM 推理和加速的企业级 ML 生产部署的问题。请注册参加 [2024 年 6 月 5 日](https://neuralmagic.com/community-office-hours/)和 [2024 年 6 月 20 日](https://neuralmagic.com/community-office-hours/)的会议。

- **CUDA 编程面试准备**：对于那些准备性能工程师岗位的人，一位成员建议使用 [awesomeMLSys](https://github.com/cuda-mode/awesomeMLSys) GitHub 仓库中的问题。该仓库包含了一份精选的 ML 系统入门问题和资源清单。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/cuda-mode/awesomeMLSys">GitHub - cuda-mode/awesomeMLSys: 一个 ML 系统入门清单</a>：一个 ML 系统入门清单。通过在 GitHub 上创建账号为 cuda-mode/awesomeMLSys 的开发做出贡献。</li><li><a href="https://neuralmagic.com/community-office-hours/">每两周一次的 vLLM 开放办公时间 - Neural Magic</a>：加入 vLLM 和 Neural Magic 的“办公时间”，了解优化的 LLM 推理以及使用 Neural Magic 和 vLLM 的加速生产部署。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1247786517452161044)** (4 messages): 

- **寻求从 Triton Kernel 获取 PTX 代码**：一位用户询问如何获取由 **Triton kernel** 生成的 **PTX 代码**。另一位用户提供了一个讨论该话题的 [GitHub issue 链接](https://github.com/triton-lang/triton/issues/3726)。
- **缓存位置已澄清**：同一位用户最初在 `~/.cache/triton` 中寻找 PTX 代码，但随后将其更正为 `~/triton/.cache`。

**提到的链接**：<a href="https://github.com/triton-lang/triton/issues/3726">How to get the generated CUDA code? · Issue #3726 · triton-lang/triton</a>：未找到描述。

  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/)** (1 messages): 

piotr.mazurek：第 4 章，练习 9，有人知道这里的解法是否正确吗？
  

---


### **CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1247974542547095746)** (1 messages): 

- **vLLM 开放办公时间公布**：成员们收到了关于 Simon 和另一位成员主持的 **vLLM 开放办公时间（open office hours）**的通知，该活动将在接下来的一个小时内进行。办公时间可以通过 [Zoom 会议链接](https://us02web.zoom.us/j/87117845746?pwd=QWZsUHlzR1ZYckxpMnNHN2hYWXhzQT09)进入，并支持多种语言，包括英语、Español、Deutsch、简体中文等。

**提到的链接**：<a href="https://us02web.zoom.us/j/87117845746?pwd=QWZsUHlzR1ZYckxpMnNHN2hYWXhzQT09">Welcome! You are invited to join a meeting: vLLM Open Office Hours (June 5, 2024). After registering, you will receive a confirmation email about joining the meeting.</a>：作为 vLLM 项目的活跃贡献者，Neural Magic 很高兴能与加州大学伯克利分校的 vLLM 团队合作，每两周举办一次开放办公时间！带着问题来了解更多关于...

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1247715165340958813)** (16 messages🔥): 

- **用于测试 HF Models 的脚本引发热议**：成员们讨论了一个 [GitHub pull request](https://github.com/pytorch/ao/pull/328)，该 PR 增加了一个脚本，方便用户使用 torchao API 快速测试模型评估。一位用户报告了在 GPU 上对大型模型进行量化和评估时出现的显存溢出 (OOM) 错误。
  
- **大型模型的 OOM 问题**：一位成员指出运行默认脚本会导致 OOM 错误，建议在量化前将模型加载到 CPU 上以避免此问题。他们观察到内存问题还取决于任务和模型的词表大小 (vocabulary size)。

- **调查 OOM 问题**：用户调查了这些 OOM 问题的根源，并提到 [EleutherAI/lm-evaluation-harness 仓库](https://github.com/EleutherAI/lm-evaluation-harness/issues?q=is%3Aissue+oom+is%3Aopen+)中存在大量未解决的 issue。他们讨论了大型词表（如 llama3 中的词表）会加剧这些问题。

- **特定任务的内存占用**：在运行 wikitext 等任务时，对大型 logits 张量的需求可能会导致内存问题，而不像 hellaswag 这样使用较短序列的小型任务。这一观察结果强调了特定评估任务对内存需求的影响。

- **关于量化的优化和讨论**：有一场关于 `torch.compile()` 应该在量化之后还是之前应用的架构讨论，结论是在他们的情况下应该在之前进行。此外还提到，Intel 建议为 inductor CPU 后端禁用 fast math。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/datasets/Salesforce/wikitext">Salesforce/wikitext · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues?q=is%3Aissue+oom+is%3Aopen+">Issues · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - Issues · EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/pytorch/ao/pull/328">Adding a quick way for users to test model eval for hf models by HDCharles · Pull Request #328 · pytorch/ao</a>：摘要：此脚本允许用户运行评估并尝试 torchao API。测试计划：python hf_eval.py。审核人：订阅者：任务：标签：
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1247625933305810996)** (245 messages🔥🔥): 

- **讨论高效 CUDA stream 使用**：成员们讨论了将所有操作迁移到 "main stream" 以提高性能的必要性。**Eriks.0595** 指出了 **named streams** 的好处，并确认在初始 memcpy 之后，代码不应在 legacy stream 上显示任何操作。

- **修复 PyTorch DDP Loss Bug**：**Aleksagordic** 修复了 PyTorch DDP loss 计算中的一个 bug，即 loss 在记录前未进行正确的 reduce，导致与 C 实现之间产生明显的差异。他们提交了一个 [PR](https://github.com/karpathy/llm.c/pull/551) 来修正此问题。

- **矩阵操作的性能优化**：详细讨论了 CUDA 中 **矩阵操作的优化**，成员们注意到通过改进内存加载模式（memory load patterns）可以获得显著但尚未达到极限的加速。**Aleksagordic** 寻求关于 warp 形成和正确 Kernel 实现的澄清，以确保不会错误地计算未来的 token。

- **确保确定性 Kernel 实现**：重点在于通过避免原子操作（atomic operations）来确保所有使用的 Kernel 都是确定性的。记录了一个关于 **global norm kernel** 的待办事项，通过使用 CPU 缓冲区存储部分和（partial sums）来完成确定性操作。

- **证词日（Deposition Day）**：**Akakak1337** 提到因涉及 Tesla 的取证（deposition）而无法参加，并幽默地指出其时间的**经济补偿极低**以及该过程的对抗性质。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/552">Feature/streams by karpathy · Pull Request #552 · karpathy/llm.c</a>：重新引入 streams，此 PR 首先引入了一个单一的 "main stream"。</li><li><a href="https://github.com/karpathy/llm.c/pull/551">Fix PyTorch DDP loss bug by gordicaleksa · Pull Request #551 · karpathy/llm.c</a>：在我们的 C 实现中，我们在使用 print0 显示 loss 之前对其进行了正确的 reduce。在我们的 PyTorch 实现中，loss 未被 reduce，这导致我们认为实现之间存在差异...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/dev/cuda/trimat_forward.cu#L452),">llm.c/dev/cuda/trimat_forward.cu at master · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1247626303931158599)** (1 messages): 

- **Fake Tensor 问题得到修复**：Marksaroufim 分享了一个来自 [PyTorch 的 pull request](https://github.com/pytorch/pytorch/pull/127927)，旨在修复 Fake Tensor 问题。描述中详细说明了该问题，涉及确保将 Tensor 转换为 FakeTensor 或使用 `FakeTensorMode` 进行实例化以避免错误。

**提到的链接**：<a href="https://github.com/pytorch/pytorch/pull/127927">FunctionalTensor: dispatch metadata directly to inner tensor by bdhirsh · Pull Request #127927 · pytorch/pytorch</a>：修复了 #127374。链接复现中的错误是：AssertionError: Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode with 'allow_non_fake_inputs'. Found in aten.sym_st...

  

---

### **CUDA MODE ▷ #[sparsity](https://discord.com/channels/1189498204333543425/1247663759434977453/1247664398324072478)** (4 messages): 

- **维基百科解释稀疏矩阵 (Sparse Matrices)**：一位成员询问维基百科对稀疏矩阵的定义是否准确，并提供了一个带有示例矩阵图像的 [维基百科页面链接](https://en.wikipedia.org/wiki/Sparse_matrix)。稀疏矩阵由于能高效存储包含大量零元素的数据，在数值分析和科学计算中至关重要。
  
- **PyTorch Sparsity 概览已上线**：另一位成员分享了来自 [PyTorch 的 README](https://github.com/pytorch/ao/tree/main/torchao/sparsity)，其中提供了稀疏性概念的基础概览。该 README 可能包含了关于 PyTorch 库中 quantization 和 sparsity 的讨论。

- **深度学习中的稀疏性综述**：一位成员分享了一篇关于深度学习中稀疏性的全面综述论文，可在 [arXiv](https://arxiv.org/abs/2102.00554) 上获取。该论文讨论了神经网络中稀疏性的优势、pruning（剪枝）方法以及训练稀疏模型的策略，总结了 300 多篇研究论文的见解。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Sparse_matrix">Sparse matrix - Wikipedia</a>：未找到描述</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/sparsity">ao/torchao/sparsity at main · pytorch/ao</a>：用于 quantization 和 sparsity 的原生 PyTorch 库 - pytorch/ao</li><li><a href="https://arxiv.org/abs/2102.00554">Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks</a>：深度学习日益增长的能源和性能成本促使社区通过选择性地剪枝组件来减小神经网络的大小。与其生物学对应物类似，...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1247654554091389079)** (1 messages): 

<ul>
    <li><strong>FineWeb 报告揭示了高性能 LLM 的秘密</strong>：Hugging Face 发布了 <a href="https://hf.co/spaces/HuggingFaceFW/blogpost-fineweb-v1">FineWeb 技术报告</a>，详细介绍了处理决策并推出了针对高教育内容的 FineWeb-Edu 数据集。该报告有助于理解像 Llama3 和 GPT-4 这样高性能的 LLM。</li>
    <li><strong>Transformers.js 登陆 Firefox 130</strong>：<a href="https://x.com/xenovacom/status/1797285648572821840">Firefox 130</a> 将包含使用 Transformers.js 的完全私有端侧 AI，最初用于图像的自动 alt-text 生成。这一新功能旨在通过将其能力扩展到屏幕阅读器用户的通用浏览来增强无障碍性。</li>
    <li><strong>Gradio Clients 1.0 发布活动</strong>：参加 6 月 6 日的 <a href="https://discord.com/events/879548962464493619/1245020251611992154">Gradio Clients 1.0 发布活动</a>，展示 Gradio 应用程序如何作为具备高性能和可扩展性的生产级可靠 API。</li>
    <li><strong>Nvidia NIM 在 Hugging Face Inference Endpoints 上发布</strong>：Nvidia 宣布在 Hugging Face Inference Endpoints 上推出 NIM 服务，支持在 AWS 和 GCP 上 <a href="https://x.com/_philschmid/status/1797713003778883858">一键部署</a> Llama 3 8B 和 70B 等模型，并具有高吞吐量。</li>
    <li><strong>Hugging Face 与 Wikimedia 合作</strong>：一篇文章详细介绍了通过来自维基百科的多样化数据集推进 ML 的潜力，强调了社区共识的作用，以及如何在 Hugging Face 上创建更多 <a href="https://huggingface.co/blog/frimelle/wikipedias-treasure-trove-ml-data">Wikimedia 数据集</a>。</li>
</ul>
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/gui_penedo/status/1797173053123916036)">来自 Guilherme Penedo (@gui_penedo) 的推文</a>：我们（终于）发布了 🍷 FineWeb 技术报告！在报告中，我们详细说明并解释了我们做出的每一个处理决策，并介绍了我们最新的数据集：📚 FineWeb-Edu，一个（仅限网页）的子集...</li><li><a href="https://x.com/xenovacom/status/1797285648572821840)">来自 Xenova (@xenovacom) 的推文</a>：Transformers.js 正在被添加到 Firefox 130 中！🤯 没错，直接在你的浏览器中实现完全私有的设备端 AI！🔥 他们正在探索的第一个用例是为图像自动生成替代文本（alt-text）...</li><li><a href="https://x.com/Gradio/status/1795561025397256498)">来自 Gradio (@Gradio) 的推文</a>：🚀从原型到生产！🙌加入我们，参加备受期待的 Gradio Clients 1.0 发布会，时间为 6 月 6 日。🤩了解你的 Gradio 应用程序如何表现出高性能...</li><li><a href="https://x.com/kamilakesbi/status/1796537200961785931)">来自 Kamil Akesbi (@kamilakesbi) 的推文</a>：说话人日志（speaker diarization）最大的障碍是什么？数据！借助 🤗 Diarizers，你现在可以生成合成会议 🗣️ 对话！从 ASR 数据集开始，你可以创建任意数量的数据...</li><li><a href="https://x.com/_philschmid/status/1797713003778883858)">来自 Philipp Schmid (@_philschmid) 的推文</a>：昨天在 COMPUTEX 上，黄仁勋宣布在 @huggingface Inference Endpoints 上发布 @nvidia NIM！🚀 NVIDIA NIM 是旨在简化并加速部署的推理服务...</li><li><a href="https://x.com/_philschmid/status/1795804027621404975)">来自 Philipp Schmid (@_philschmid) 的推文</a>：产品更新：@nvidia L4s 现在已在 AWS 上的 @huggingface Inference Endpoints 中可用！每位用户和组织可享受多达 8 个 L4，且比按需购买 AWS EC2 节省 20%。🤑 - 1x NVIDIA L4...</li><li><a href="https://x.com/abhi1thakur/status/1795477747701104651)">来自 abhishek (@abhi1thakur) 的推文</a>：AutoTrain 刚刚获得了全新的 UI 🚀🚀🚀</li><li><a href="https://x.com/_philschmid/status/1797994961197031703">来自 Philipp Schmid (@_philschmid) 的推文</a>：很高兴分享一篇关于如何使用 NVIDIA 的 2023 年 SEC Filing 数据集，并结合最新的研究（如 Matryoshka Representation Learning）为金融 RAG 应用微调嵌入模型（embedding models）的新博客...</li><li><a href="https://x.com/frimelle/status/1797619351954260214)">来自 Lucie-Aimée Kaffee (@frimelle) 的推文</a>：以社区为中心且棒极了：@huggingface 和 @Wikimedia 🤗 我写了一篇关于我们如何利用来自 @Wikipedia 的多样化数据集推进 ML 的文章，以及为什么以及如何在 Hugging Face 上创建更多 Wikimedia 数据集...</li><li><a href="https://x.com/NielsRogge/status/1796213271189438888)">来自 Niels Rogge (@NielsRogge) 的推文</a>：终于回到了 @YouTube，带来了一个新视频：在你的自定义数据集上微调 PaliGemma（或 LLaVa、Idefics2...）！我正在 @GoogleColab 的 L4 GPU 上进行微调。我讲解了许多内容，比如...</li><li><a href="https://x.com/abhi1thakur/status/1796210385579639144)">来自 abhishek (@abhi1thakur) 的推文</a>：🚨 新博客：如何使用 AutoTrain 微调自定义嵌入模型（Embedding Models）。学习：- 数据格式应该是怎样的 - 如何正确映射列 - 示例数据集 - 自定义配置 - 本地训练 - 训练...</li><li><a href="https://x.com/vanstriendaniel/status/1795875763557904753">来自 Daniel van Strien (@vanstriendaniel) 的推文</a>：你需要一个数据集来训练自定义的 sentence transformer 模型吗？我创建了一个使用 LLM 创建合成数据集的流水线，你可以直接将其用于微调/训练 Sentence Transformer...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1247628263816167585)** (214 条消息🔥🔥): 

- **TinyLlama 在训练时长上面临挑战**：一名成员抱怨 TinyLlama 表现不佳，因为它不允许足够长的训练周期。对话围绕 fine-tuning 设置和平台细节展开。
- **HF Spaces 密钥在黑客事件中泄露**：讨论围绕 Hugging Face 的一封邮件展开，该邮件披露了一起安全漏洞，导致私有的 HF token 被公开。用户担心其私有代码可能遭到泄露。
- **关于互联网数据安全的辩论**：在 HF 泄露事件后，用户们讨论了基于互联网的数据安全性。一位成员表示深切担忧，而另一位成员则诙谐地指出互联网安全本质上就是不可靠的。
- **探索基于 Diffusion 的语言模型**：一场关于创建基于 Diffusion 的语言模型的对话，类似于图像生成模型。涉及头脑风暴如何从随机字符串中去除“noise（噪声）”。
- **处理讨论中的违规行为**：一些用户讨论了如何处理社区讨论中不当且具有攻击性的行为。强调了举报和审核是维护尊重环境的必要步骤。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/nroggendorff/mayo">Mayo - nroggendorff 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/blog/space-secrets-disclosure">Space 密钥安全更新</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/v4.17.0/en/create_a_model#confi">创建自定义模型</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/v4.17.0/en/create_a_model#configuration">创建自定义模型</a>：未找到描述</li><li><a href="https://huggingface.co/blog/abhishek/object-detection-autotrain">使用 AutoTrain 训练目标检测模型</a>：未找到描述</li><li><a href="https://github.com/microsoft/unilm/tree/master/textdiffuser">microsoft/unilm 的 textdiffuser 分支</a>：跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm</li><li><a href="https://tenor.com/view/saul-goodman-talking-gif-26157017">Saul Goodman 说话的 GIF - Saul Goodman Talking - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/kruMoWQLd1u.gif">你是来自俄亥俄州还是怎么的 我们的画作 GIF - Are you from ohio or something Ohio Our drawings - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/69">zero-gpu-explorers/README · 别费劲申请了 - 非官方 ZeroGPU 政策解读</a>：未找到描述</li><li><a href="https://github.com/abetlen/llama-cpp-python/issues/576">如何使用 GPU？ · Issue #576 · abetlen/llama-cpp-python</a>：我在配有内置 RTX 3060 12GB VRAM 的新电脑上运行 llama cpp python。这是我的代码：from llama_cpp import Llama llm = Llama(model_path=&quot;./wizard-mega-13B.ggmlv3.q4_0.bin&quot;, n_ct...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/1254">GGUF 本地模型 · Issue #1254 · EleutherAI/lm-evaluation-harness</a>：是否有本地托管的 GGUF 模型的 lm_eval 示例？lm_eval --model gguf --model_args pretrained=Llama-2-7b-chat-hf-Q4_K_M.gguf, --tasks hellaswag --device mps 出现 AssertionError: mus...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1247740949770145873)** (2 条消息): 

- **在 ML 和 Streamlit 项目上进行协作**：一位成员询问：“有人想一起构建一个 ML 和 Streamlit 项目吗？”这发出了在社区内合作开展机器学习和基于 Streamlit 项目的邀请。
- **问候交流**：另一位用户简单地发了一句 “hello”。这表明频道内有一种随意的、友好的氛围。
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1247627017248047315)** (5 条消息): 

- **用于 ASR/TTS 训练的 610 小时德语语音样本**：一位成员分享了一个包含 610 小时德国议会演讲转录语音样本的数据集，用于 ASR/TTS 训练。该数据集可在 [Hugging Face Hub](https://huggingface.co/datasets/D4ve-R/bundestag-asr) 上获取。

- **通过手写 C 编程理解 Transformers**：一条推文展示了一个结合手写 C 编程和矩阵乘法来解释 Transformers 工作原理的项目。该计划受到 @karpathy 的启发，旨在揭秘 LLMs，更多详情见[此处](https://x.com/ProfTomYeh/status/1798042265883156651)。

- **CogVLM2-LLaMA3-Chat-19B 的早期测试**：对基于 LLaMA3 的 [CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4) 视觉语言模型进行了早期测试。该模型具有显著改进，支持 8K 内容长度，并提供高图像分辨率，但在 Windows 上运行存在问题。

- **微软在 GitHub 上的 TextDiffuser 项目**：微软发布了 [TextDiffuser](https://github.com/microsoft/unilm/tree/master/textdiffuser)，这是一个专注于跨任务、语言和模态的大规模自监督预训练项目。此外还提到了更新版本 [TextDiffuser-2](https://github.com/microsoft/unilm/tree/master/textdiffuser-2)，继续在这些能力上进行构建。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4">THUDM/cogvlm2-llama3-chat-19B-int4 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/microsoft/unilm/tree/master/textdiffuser">unilm/textdiffuser at master · microsoft/unilm</a>: 跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm</li><li><a href="https://x.com/ProfTomYeh/status/1798042265883156651">Tom Yeh 的推文 | AI by Hand ✍️ (@ProfTomYeh)</a>: 手写 llm.c ✍️ C 编程 + 手动矩阵乘法。这种组合可能是解释 Transformer 工作原理的最底层方式。特别感谢 @karpathy 鼓励...</li><li><a href="https://github.com/microsoft/unilm/tree/master/textdiffuser-2">unilm/textdiffuser-2 at master · microsoft/unilm</a>: 跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm</li><li><a href="https://huggingface.co/datasets/D4ve-R/bundestag-asr">D4ve-R/bundestag-asr · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1247821562414108682)** (5 条消息): 

- **在 Windows 上安装 Apache Airflow**：分享了一篇关于如何使用 WSL 在 **Windows 上安装 Apache Airflow** 的好文章。你可以在[此处](https://tolulade-ademisoye.medium.com/how-to-install-apache-airflow-on-windows-a-beginners-guide-to-wsl-297c5ba5f519)查看详细指南。

- **全面的 LLM 资源指南**：一位成员编写了一份关于其最喜爱的 **LLM 解释器**的资源指南，包括 vLLM, SSMs, DPO 和 QLoRA。查看这份有组织的“教科书式”资源指南[此处](http://genai-handbook.github.io)以及发布公告[此处](https://x.com/willccbb/status/1798423849870270671)。

- **用于气候意识投资的 AI**：一位成员创建了一个 AI 助手，使用 `climatebert/tcfd_recommendation`、Qdrant Cloud 和 `microsoft/Phi-3-mini-128k-instruct` 来寻找面向气候的投资解决方案。在[此处](https://huggingface.co/spaces/as-cle-bert/tcfd_counselor)探索该 AI 助手，并使用他们的 ML 支持方案在[此处](https://huggingface.co/spaces/as-cle-bert/carbon-footprint-predictor)计算你的碳足迹。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/as-cle-bert/tcfd_counselor">Tcfd Counselor - Hugging Face Space (by as-cle-bert)</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/as-cle-bert/carbon-footprint-predictor">Carbon Footprint Predictor - Hugging Face Space (by as-cle-bert)</a>: 未找到描述</li><li><a href="https://x.com/willccbb/status/1798423849870270671">will brown (@willccbb) 的推文</a>: 过去一年一直在学习 LLM 等知识，将一些我最喜欢的解释器整理成了一份“教科书式”的资源指南。希望我在开始时就能拥有它，也许它对其他人也有用...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1247990833571958824)** (2 条消息): 

- **6月11日的 Human Feedback Foundation 活动**：**Human Feedback Foundation** 将于 6月11日举办一场活动，旨在加强公众在医疗保健、治理和民主等关键领域对 AI 系统的参与。[活动链接](https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator)。

- **往期会议录像已上传至 YouTube**：往期会议的录像已在 Human Feedback Foundation 的 [YouTube 频道](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg)上线，主讲人来自 UofT、Stanford 和 OpenAI 等知名机构。该基金会强调了公众参与 AI 开发、AI 工具的互操作性以及 AI 安全研究中教育推广的重要性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg">Human Feedback Foundation</a>：Human Feedback Foundation 的使命是将人类反馈融入开源 AI 项目。我们寻求：通过支持开源开发和政策倡议，实现公众对 AI 的参与...</li><li><a href="https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator">LLM Reading Group (3月5日, 19日; 4月2日, 16日, 30日; 5月14日, 28日; 6月11日)</a>：来见见 LLM/NLP 研究领域一些开创性论文的作者，听他们分享自己的工作。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1247776895299227732)** (7 条消息): 

- **单一国家 Logo 检测**：一位用户正在进行一个检测特定国家 Logo 的项目。他们正在寻求一种系统，可以将图像分类为包含或不包含该 Logo，以保持数据集的可控性。

- **CamViG 论文讨论**：[CamViG: Camera Aware Image-to-Video Generation with Multimodal Transformers](https://arxiv.org/abs/2405.13195)。该论文提出将视频生成以 3D 相机运动为条件，展示了成功的相机控制和精确的 3D 相机路径生成。

- **寻求 MultiModal LLM 的应用案例**：一位用户询问关于 MultiModal 大语言模型的优秀应用案例建议。讨论中未提供具体建议。

- **图像中的物理尺度估计**：一位用户正在构建一个流水线，用于估计房间图像中像素的物理尺度。他们正在 HuggingFace 或类似论坛上寻找先前的研究成果和资源，以在满足特定图像假设的情况下协助其项目。

**提到的链接**：<a href="https://huggingface.co/papers/2405.13195">论文页面 - CamViG: Camera Aware Image-to-Video Generation with Multimodal Transformers</a>：未找到描述。

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1247642239933616207)** (2 条消息): 

- **尝试导入特定函数以解决版本不匹配问题**：一位用户建议尝试导入特定函数以观察其是否生效，这表明可能存在与版本不匹配相关的问题。他们推荐将其作为故障排除步骤。

- **GroundedAI 开源用于 LLM 评估的模型**：一位成员宣布他们已经开源了用于执行 LLM as a judge 评估的模型，可在 [此处](https://huggingface.co/grounded-ai) 获取。这些模型旨在提供高效、高性能的黑盒 LLM 替代方案，并确保输出符合事实且符合伦理。

**提到的链接**：<a href="https://huggingface.co/grounded-ai">grounded-ai (GroundedAI)</a>：未找到描述。

  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1247695146535878787)** (25 messages🔥): 

- **脚本和模型配置指南**：一位用户详细介绍了更新脚本的完整流程，使用了来自 SDXL 模型配置的值，包括设置 `sample`、`timestep` 和 `encoder_hidden_states`。他们详细阐述了必要的参数，如 `added_cond_kwargs` 和 `text_embeds`，并提供了代码片段以澄清实现方式。

- **被 AI 创新所淹没**：一位成员表示对 AI 进步的飞速感到不知所措，称“根本跟不上”。另一位成员表示赞同，指出试图跟上所有更新“只会导致疯狂”。

- **针对缺失 `text_embeds` 的故障排除**：一位成员报告了与 `added_cond_kwargs` 中缺失 `text_embeds` 相关的 `TypeError`。他们得到了帮助，将 `text_embeds` 指定为 `torch.randn(2, 77, 1280).half().cuda()`，并讨论了在函数作用域内使用初始化来避免此类问题。

- **庆祝调试成功**：在解决脚本问题后，一位成员用一张 [惊讶的 GIF](https://tenor.com/view/wow-amazed-in-awe-woah-smiling-gif-16490512) 进行了庆祝。他们承认之前对 text encoders 的维度以及如何正确合并 `text_embeds` 和 `time_ids` 存在误解。

**提及的链接**：<a href="https://tenor.com/view/wow-amazed-in-awe-woah-smiling-gif-16490512">Wow Amazed GIF - Wow Amazed In Awe - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1247808792390008872)** (1 messages): 

- **Gradio Clients 1.0 在 YouTube 上线**：HuggingFace 宣布通过 YouTube 直播活动发布 Gradio Clients 1.0。新工具允许用户以编程方式利用 Gradio 演示，增强了 **Python** 和 **JavaScript** 客户端在生产级应用中的能力。
- **提供了活动具体细节**：公告包含了 **Discord 活动**和 **YouTube 直播**的链接，届时将讨论此次发布。这为开发者提供了一个向 Gradio 团队学习如何使用新客户端构建机器学习 API 的机会。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/hugging-face-879548962464493619?event=1245020251611992154">Discord - Group Chat That’s All Fun &amp; Games</a>：Discord 非常适合玩游戏、与朋友闲逛，甚至建立全球社区。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://www.youtube.com/watch?v=44vi31hehw4">[Launch] How to Build Machine Learning APIs Using Gradio</a>：每月有一百万开发者使用 Gradio Python 库创建机器学习演示和 Web 应用程序。加入 Gradio 团队...
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1247626730517041152)** (47 messages🔥): 

- **Llama70b 在 GGUF 格式上遇到困难**：成员们讨论了在 **LM Studio** 中加载 Llama70b 的问题，原因是它没有保存为 GGUF 文件。有人建议使用 sym link 选项，而另一位则建议转换文件或重新下载为 GGUF 格式。

- **频道中出现了关于加密货币的幽默**：关于 GPU 在夜间偷偷挖掘加密货币的幽默交流引起了其他成员的笑声。

- **LM Studio 0.2.24 在 Windows 10 上无法启动**：一位用户报告 **LM Studio 0.2.24** 无法启动，在任务管理器中没有迹象，而 **0.2.22 版本**运行正常。尽管重新安装了应用并以管理员身份运行，问题依然存在，且没有生成错误日志或在事件查看器中提供有用信息。

- **关于 iMat 文件和 RAM 使用的查询**：用户询问了在 **LM Studio** 中使用 iMat 文件的方法以及限制 RAM 使用的手段。对方澄清说 iMat 支持取决于 llamacpp 的兼容性；对于 RAM，建议使用较小的模型或在推理期间管理 layers 以优化使用。

- **股市照片分析问题**：一位用户在多个频道重复询问模型是否可以分析股市照片，但被告知本地模型无法执行此类分析。他们被建议只发布一次问题以避免刷屏。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - Discover and run local LLMs</a>：查找、下载并实验本地 LLM</li><li><a href="https://tenor.com/view/tf2engineer-imposter-it-could-be-you-meme-tf2spy-gif-23428001">Tf2engineer Imposter GIF - Tf2Engineer Imposter It Could Be You - Discover &amp; Share GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1247632691340509266)** (102 条消息🔥🔥): 

- **推荐用于改进写作的模型**：为了增强写作风格，一位成员建议查看排行榜并按 13B 过滤模型，但未指定具体模型。 

- **SMAUG 和 Llama 3 的问题**：多位用户确认在使用 Llama 3 版本 0.2.24 时遇到了 SMAUG 的 BPE tokenizer 错误，并指出一周后已添加了相关支持。

- **Command R 模型性能查询**：一位用户报告称，将 Command R 模型 offloaded 到 Metal 会产生乱码，而仅 CPU 模式则极其缓慢。建议检查 rope 设置，并提供了一个指向特定 [Hugging Face 模型](https://huggingface.co/mradermacher/c4ai-command-r-v01-GGUF) 的链接。

- **静态量化与 iMatrix 量化模型的区别**：用户讨论了 iMatrix 量化在模型量化时整体表现更好，因为“它们尽可能避免对重要权重进行量化”。他们还指出某些硬件配置可能会影响这一点。

- **无审查模型 (Uncensored Models)**：关于无审查模型的讨论重点介绍了 *Neural Devil* 等用于通用用途和故事写作的特定模型。分享了一个模型链接 ([YorkieOH10/Llama-3-MahouDevil-8B-Q8_0](https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF)) 以及用户的正面体验。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/mradermacher/13B-Psyfighter2-Erebus3-Slerp-GGUF">mradermacher/13B-Psyfighter2-Erebus3-Slerp-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mradermacher/Euryale-1.3-Small-7B-i1-GGUF">mradermacher/Euryale-1.3-Small-7B-i1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mradermacher/Genshin_Impact_Mistral_v3_Plot_Chat_roleplay_chat_merged-GGUF">mradermacher/Genshin_Impact_Mistral_v3_Plot_Chat_roleplay_chat_merged-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF">YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5">nomic-ai/nomic-embed-vision-v1.5 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Llama-3-8B-Instruct-Gradient-1048k-GGUF">bartowski/Llama-3-8B-Instruct-Gradient-1048k-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1247902551123296266)** (1 条消息): 

- **czkoko v0.2.24 中的额外转义字符 bug**：最新版本 **v0.2.24** 偶尔会在预设配置中包含意外的转义字符。配置中的示例包括 `"input_suffix": "\\n\\nAssistant: "` 和 `"pre_prompt_suffix": "\\n\\n"`。
  

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1247637209775996990)** (90 条消息🔥🔥): 

- **AMD Radeon RX 7900 工作站级 GPU 引起关注**：一场关于[新款华擎 (ASRock) Radeon RX 7900 XTX & 7900 XT 工作站级 GPU](https://wccftech.com/asrock-radeon-rx-7900-xtx-7900-xt-workstation-gpus-12v-2x6-connector-2-slot-design-for-ai-setups/)的讨论展开，其特点包括 12V-2x6 电源接口和双槽涡轮风扇（blower）设计。尽管双槽 GPU 很有吸引力，但一些成员对涡轮风扇的噪音表示担忧。
  
- **关于 Linux 的争议性观点**：一位成员批评 Linux “充斥着命令行”，认为尽管人们普遍声称其易用，但实际上并不友好。另一位成员则认为真正的问题在于缺乏研究和过高的期望，并推荐使用 KDE Plasma 以获得更接近 Windows 的体验。

- **Recall 功能引发隐私担忧**：成员们讨论了 Windows 中备受争议的 Recall 功能，强调了它可能将敏感信息汇总到一个易被黑客攻击的数据库中。整体情绪表现出对该功能相关的隐私和安全风险的重大担忧。

- **在 Windows Recall 阴影下转向 Linux**：即将到来的 Windows Recall 问题引发了关于潜在“Linux 安装潮”的讨论，有人开玩笑说大卖场会将这种转变推给不知情的消费者。资深的社区成员强调了 Recall 对个人和商业用途 Windows 系统带来的关键安全影响。

- **帮助台 (Help Desk) 恐怖故事**：出现了一些关于过去 IT 支持工作经历的轻松抱怨，一位成员幽默地讲述了维修一台闻起来有猫尿味的电脑的故事。这部分对话为关于系统管理和 IT 安全挑战的技术讨论增添了共鸣。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://wccftech.com/asrock-radeon-rx-7900-xtx-7900-xt-workstation-gpus-12v-2x6-connector-2-slot-design-for-ai-setups/">ASRock Radeon RX 7900 XTX &amp; 7900 XT 工作站级 GPU 采用 12V-2x6 接口及双槽设计，助力 AI 配置</a>：华擎发布了其 Radeon RX 7900 XTX &amp; 7900 XT 工作站级 GPU，配备 12V-2x6 电源接口并采用双槽涡轮风扇设计。</li><li><a href="https://tenor.com/view/tupac-true-pointing-up-truth-2pac-gif-26578973">Tupac True GIF - Tupac True Pointing Up - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.ebay.co.uk/itm/334715824949">Intel Neural Compute Stick 2 神经深度学习 USB NCSM2485.DK NCS | eBay</a>：未找到描述</li><li><a href="https://youtu.be/0506yDSgU7M?si=gItHh97CbX7qTxVC">Linux 讨厌我 – 日常使用挑战第一部分</a>：在 https://www.freshbooks.com/linus 免费试用 FreshBooks 30 天，无需信用卡。使用代码 LINUS 在 https://lmg.gg/glass... 获取 GlassWire 25% 折扣。</li><li><a href="https://youtu.be/3E8IGy6I9Wo?si=oR9VTNHjrt-dbz9c">进展不顺利…… Linux 游戏挑战第二部分</a>：在 https://lmg.gg/Ktd7Z 免费试用 Pulseway，开始远程监控和管理您的服务器或 PC。使用代码 LINUS 在 https:/... 获取 GlassWire 25% 折扣。</li><li><a href="https://youtu.be/TtsglXhbxno?si=XzudnJGBniprnlMS">尝试在 Linux 上执行简单任务 lol - 日常使用挑战第三部分</a>：立即查看 NZXT BLD：https://nzxt.co/LttBLDNov21。Linus 和 Luke 使用他们的 Linux 桌面完成一系列日常任务，如打印和压缩...</li><li><a href="https://youtu.be/Rlg4K16ujFw?si=91uM_GPer6kkQxW6">Linux 游戏尚未准备就绪... - 日常使用挑战大结局</a>：访问 https://www.squarespace.com/LTT 并使用优惠码 LTT 获得 10% 折扣。在 https://lmg.gg/Airalo 尝试您的第一个 Airalo eSIM。Luke 和... 的一个月已经过去了。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1247924966884905081)** (3 条消息): 

- **VRAM 不足导致模型加载失败**：一位用户在给定系统配置下加载模型时遇到错误。另一位成员指出，问题源于 GPU offload 的 VRAM 不足，并建议将其关闭以解决问题。
  

---


### **LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1247957815189045258)** (2 条消息): 

- **关于操作系统的询问**：一位用户询问了其他人使用的操作系统。一位成员回答说他们正在使用 **Windows 11**。
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1247627293422260306)** (185 条消息🔥🔥): 

- **DDoS 攻击导致 AI 服务大范围中断**：一位成员分享了最近 ChatGPT、Claude、Gemini、Perplexity 和 Copilot 的中断是由亲俄黑客组织 Anonymous Sudan 发起的 DDoS 攻击引起的。这解释了为什么这些问题被认为比典型的云服务器问题更严重。
  
- **GPT 订阅的实用性引发讨论**：围绕各种 AI 订阅的实用性展开了讨论。一位用户比较了订阅 GPT 与 Character AI 的生产力，强调了总结书籍和创建内容等实际用途。

- **语言模型在数学方面表现不佳**：多位用户讨论了像 GPT 这样的语言模型在处理数学任务时经常遇到困难。具体问题包括模型给出错误的计算结果，且在用户纠正后仍无法识别逻辑错误。

- **AI 在日常任务和系统中的作用**：用户分享了将 AI 与其他软件和活动集成的经验。示例包括将 ChatGPT 与家庭自动化系统连接，以及 Windows Copilot 的实际应用和局限性。

- **潜在的未来 AI 能力与担忧**：一位用户幽默地建议宠物 AI 有朝一日可能会反戈一击，引发了关于 AI 伦理对待及其长期影响的更广泛对话。此外，人们对未来 AI 能力的进步（如访问 Windows API 函数）也充满好奇。
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1247756498051010671)** (5 条消息): 

- **对 Google Sheets 中 GPT-4 Vision 的好奇**：一位用户询问是否有办法在 Google Sheets 中使用 **GPT-4 Vision** 来描述单元格中的图像，并将描述放在相邻单元格中。另一位成员建议 *"你可以使用 API"*。
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1247661984556777512)** (13 条消息🔥): 

- **将提示词添加到 "The Prompt Index"**：一位成员正在整理名为 **"The Prompt Index"** 的资源，并邀请其他人贡献有趣的提示词。他们强调这是一个完全免费的服务，旨在避免大量价格过高、质量低下的内容。
- **在 Google Sheets 中使用 GPT-4 Vision 的目标**：一位成员询问是否可以使用 **GPT-4 Vision** 分析 Google Sheets 中的图像并在相邻单元格中生成描述。他们正在寻找一种在电子表格环境中自动执行此任务的方法。
- **通过简单的刷新修复 GPT**：一位成员询问当 GPT 卡住时如何修复。另一位成员建议尝试 **Ctrl + F5** 刷新页面，提问者表示会尝试一下。
- **使用 ChatGPT 进行 SEO 分析**：一位成员寻求一种让 **ChatGPT** 提供特定站点 SEO 分析和优化策略的方法，因为目前的回复过于笼统。另一位成员建议将 SEO 表现良好的网站源码与自己的网站进行对比，并指示 ChatGPT 应用成功的策略。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1247661984556777512)** (13 条消息🔥): 

- **分享免费提示词收集资源**：一位成员介绍了 "The Prompt Index"，这是一个**完全免费的资源**，用于收集有趣的提示词。他们明确表示这是为了避开在其他地方常见的典型营销提示词。

- **关于 Google Sheets 中 GPT-4 Vision 的咨询**：一位用户询问是否有办法在 **Google Sheets 中使用 GPT-4 Vision** 来描述单元格中的图像并将描述放在相邻单元格中。聊天中未提供解决方案。

- **分享 GPT 卡住的解决方案**：一位用户询问如何修复**卡住的 GPT**。另一位成员建议使用 *Ctrl + F5* 刷新页面。

- **使用 ChatGPT 进行 SEO 分析的最佳方法**：一位成员寻求关于让 ChatGPT 为其网站进行更准确 SEO 分析的建议。另一位用户建议复制并粘贴优化良好的网站和用户网站的源代码，然后要求 ChatGPT 进行对比并提出改进建议。

- **成功的模型集成实验**：一位用户分享了将 **llava-v1.6-mistral-7b vision model** 与 7b 模型集成成功的经验。他们发现这些模型可以毫无问题地协同工作，这非常有趣。
  

---

：此次发布标志着将生成式音频能力的部分功能向更广泛受众开放的一个重要里程碑。

**提到的链接**：<a href="https://stability.ai/news/introducing-stable-audio-open"> Stable Audio Open &mdash; Stability AI</a>：Stable Audio Open 是一个开源模型，经过优化，可使用文本提示词生成短音频样本、音效和制作元素。

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1247643080136462477)** (141 条消息🔥🔥): 

- **探索 WebUI 选项：A1111 vs. InvokeAI**：成员们讨论了 Stable Diffusion 不同 WebUI 的优势，**A1111** 因其多功能性和易用性受到赞赏，而 **InvokeAI** 则因其“区域提示词 (regional prompting)”功能和更具创新性而受到关注。[这是 GitHub 上的 InvokeAI](https://github.com/invoke-ai/InvokeAI)。

- **关于正则化图像 (Regularization Images) 和标注 (Captions) 的困惑**：用户不确定在训练模型时 **regularization images** 是否需要标注，引发了简短的辩论。其中一个问题是：“使用正则化图像是否意味着不再需要使用标注了？”

- **Stable Audio Tools 及其可用性**：成员们分享了关于使用 AI 生成音频的资源，重点介绍了 [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools)。大家对其在 Google Colab 上的使用以及是否允许商业用途感到好奇。

- **ComfyUI 在图像生成中的灵活性**：ComfyUI 被推荐给寻求工作流灵活性的用户，尽管其学习曲线较陡峭。提供的一个示例说明了其功能：“你可以使用 cascade 或 sigma 生成，然后用 sdxl 进行精炼……”

- **初学者社区资源**：分享了关于学习 Stable Diffusion 的教程和讨论，以帮助新手。[YouTube 上的 Sebastian Kamph](https://youtu.be/kqXpAKVQDNU?si=EHs5JZaQmE1yTi1Q) 被推荐作为 A1111 入门的综合指南。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s34B-b88K">laion/CLIP-ViT-g-14-laion2B-s34B-b88K · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-2-inpainting">stabilityai/stable-diffusion-2-inpainting · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/invoke-ai/InvokeAI">GitHub - invoke-ai/InvokeAI: InvokeAI 是 Stable Diffusion 模型的领先创意引擎，赋能专业人士、艺术家和爱好者使用最新的 AI 驱动技术生成和创作视觉媒体。该解决方案提供行业领先的 WebUI，支持通过 CLI 进行终端使用，并作为多个商业产品的基础。</a></li><li><a href="https://github.com/Stabili">StaBili - 概览</a>: StaBili 有一个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://huggingface.co/stabilityai/cosxl">stabilityai/cosxl · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/kqXpAKVQDNU?si=EHs5JZaQmE1yTi1Q">如何安装 Stable Diffusion - automatic1111</a>: 第 2 部分：如何使用 Stable Diffusion https://youtu.be/nJlHJZo66UA Automatic1111 https://github.com/AUTOMATIC1111/stable-diffusion-webui 安装 Python https://w...</li><li><a href="https://github.com/numz/sd-wav2lip-uhq">GitHub - numz/sd-wav2lip-uhq: 适用于 Automatic1111 的 Wav2Lip UHQ 扩展</a>: 适用于 Automatic1111 的 Wav2Lip UHQ 扩展。通过在 GitHub 上创建账号为 numz/sd-wav2lip-uhq 的开发做出贡献。</li><li><a href="https://github.com/Stability-AI/stable-audio-tools">GitHub - Stability-AI/stable-audio-tools: 用于条件音频生成的生成模型</a>: 用于条件音频生成的生成模型 - Stability-AI/stable-audio-tools</li><li><a href="https://github.com/diontimmer/audio-diffusion-gradio">GitHub - diontimmer/audio-diffusion-gradio: 功能齐全的音频扩散 gradio 客户端，主要针对 stable-audio-tools。</a>: 功能齐全的音频扩散 gradio 客户端，主要针对 stable-audio-tools。 - diontimmer/audio-diffusion-gradio</li><li><a href="https://www.reddit.com/user/No_Dragonfruit_5472/comments/1chdemx/tradingview_premium_pack_crack_2024_version_free/">Reddit - 深入探索</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1247634160915579022)** (55 messages🔥🔥): 

- **AI 驱动的显示墙：彩色条纹与 Logo**：一位成员建议使用 AI 打造巨型显示墙，通过自动变换颜色或展示艺术作品来提升日常生活。他们提到了诸如“小山露色条纹和品味高雅的 Logo”等功能。

- **对过度炒作的“突破性”技术的批判**：在讨论一款 RLCD 产品时，一位成员指出其 CEO 的神秘形象和对独特技术的夸大宣传，尽管它本质上只是一个 RLCD。另一位成员反驳道，修改使其更具半穿透半反射特性，并强调了其与 Samsung 的 QD-OLED 显示器的相似之处。

- **AGI 竞赛升温**：一篇引用的 [situational-awareness.ai 博客](https://situational-awareness.ai/) 讨论了计算集群投资的不断升级，预计到 2025/26 年 AGI 能力将大幅提升。一位成员反思了前沿实验室与这些进展可能带来的招聘信号之间日益扩大的差距。

- **关于智商测试与开源中高自主性（High Agency）的辩论**：成员们讨论了将智商测试用于招聘的挑战和合法性。他们强调了高自主性（High Agency）比单纯的高智商更有价值，并指出成功通常源于雄心、自主性以及检测高时间稀疏性模式能力的结合。

- **对 KANs 的兴趣**：几位成员辩论了基于 Koopman 算子的神经网络（KANs）的潜力和局限性。他们讨论了可解释性、效率和当前的理论空白，结论是 KANs 尽管有前景，但短期内不太可能取代传统的神经网络。

**提到的链接**：<a href="https://situational-awareness.ai/,">Introduction - SITUATIONAL AWARENESS: The Decade Ahead</a>：Leopold Aschenbrenner，2024 年 6 月。你可以先在旧金山预见未来。在过去的一年里，业内的谈论重点已从 100 亿美元的计算集群转向 1000 亿美元，再到万亿级……

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1247641361524588727)** (58 messages🔥🔥): 

- **探讨 Transformer 和 SSM 的局限性**：一位用户分享了[一篇新论文](https://arxiv.org/abs/2405.16674)，讨论了 Transformer 和 SSM 在复杂推理任务中的局限性。该论文提供了针对模型在组合性（compositionality）方面面临挑战的理论和实证分析。

- **内化思维链（Chain-of-Thought）步骤**：分享了一篇论文，提出了一种教模型内化思维链（CoT）推理步骤的方法，在 9x9 乘法等任务上实现了高性能。该技术使 Mistral 7B 等模型受益，详情请见[此处](https://arxiv.org/abs/2405.14838)。

- **提升 RLHF 鲁棒性**：讨论了一种用于人类反馈强化学习（RLHF）的[自我改进鲁棒偏好优化（SRPO）](https://arxiv.org/abs/2406.01660)方法。这种新方法旨在通过将偏好优化转变为自我改进过程，从而对任务变化具有完全的鲁棒性。

- **新的扩散模型引导方法**：用户讨论了图像生成扩散模型中的[一种新方法](https://arxiv.org/abs/2406.02507)，该方法解耦了图像质量和变化控制，使用模型自身的一个较小、训练程度较低的版本来引导生成。

- **RETRO 模型的开源实现**：讨论涉及 NVIDIA 在 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro) 中提供的 RETRO 开源实现。有人提出了关于公开运行高质量、大规模 AI 模型进行性能评估的可访问性和潜力的问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://arxiv.org/abs/2406.02507">Guiding a Diffusion Model with a Bad Version of Itself</a>: 图像生成 Diffusion Model 关注的主要轴线是图像质量、结果的多样性以及结果与给定条件（如类别标签）的对齐程度...</li><li><a href="https://arxiv.org/abs/2406.01660">Self-Improving Robust Preference Optimization</a>: 无论是在线还是离线 RLHF 方法（如 PPO 和 DPO），在使 AI 与人类偏好对齐方面都取得了巨大成功。尽管取得了成功，现有方法仍面临一个根本性的问...</li><li><a href="https://arxiv.org/abs/2405.20519">Diffusion On Syntax Trees For Program Synthesis</a>: LLM 每次生成一个 token 的代码。其自回归生成过程缺乏对程序输出观察的反馈。训练 LLM 直接建议编辑可能...</li><li><a href="https://www.colorama.app.">Colorama</a>: 用真实色彩映射过去</li><li><a href="https://arxiv.org/abs/2406.02075">ReLU-KAN: New Kolmogorov-Arnold Networks that Only Need Matrix Addition, Dot Multiplication, and ReLU</a>: 受限于基函数（B-spline）计算的复杂性，Kolmogorov-Arnold Networks (KAN) 在 GPU 上的并行计算能力受到限制。本文提出了一种新型 ReLU-KAN...</li><li><a href="https://arxiv.org/abs/2405.16674">Limits of Deep Learning: Sequence Modeling through the Lens of Complexity Theory</a>: 深度学习模型在各种应用中取得了显著成功，但在处理需要对序列进行复杂推理的任务（如函数组合和计算...）时仍然面临困难。</li><li><a href="https://arxiv.org/abs/2406.02543">To Believe or Not to Believe Your LLM</a>: 我们探索了 LLM 中的不确定性量化，目标是识别给定查询的响应何时具有较大的不确定性。我们同时考虑了认知（epistemic）和偶然（aleatoric）...</li><li><a href="https://arxiv.org/abs/2406.02061">Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models</a>: LLM 通常被描述为基础模型的实例——即在 few-shot 或 zero-shot 模式下能跨各种任务和条件进行强大迁移的模型，而...</li><li><a href="https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro">Megatron-LM/tools/retro at InstructRetro · NVIDIA/Megatron-LM</a>: 持续进行的大规模 Transformer 模型训练研究 - NVIDIA/Megatron-LM</li><li><a href="https://arxiv.org/abs/2405.15143">Intelligent Go-Explore: Standing on the Shoulders of Giant Foundation Models</a>: Go-Explore 是一个强大的算法家族，旨在解决困难的探索问题，其建立在存档已发现状态并迭代返回并从中探索的原则之上...</li><li><a href="https://tedunderwood.com/2023/03/19/using-gpt-4-to-measure-the-passage-of-time-in-fiction/">Using GPT-4 to measure the passage of time in fiction</a>: LLM 是宝贵的助手，尤其是当它们拒绝遵守指令时。</li><li><a href="https://arxiv.org/abs/2406.02394">Multiple Choice Questions and Large Languages Models: A Case Study with Fictional Medical Data</a>: 像 ChatGPT 这样的 LLM 在医学领域展示了巨大的潜力，通常使用类似于 USMLE 的多选题 (MCQs) 进行评估。尽管如...</li><li><a href="https://x.com/maximegmd/status/1798245197585002671">Tweet from Maxime G, M.D (@maximegmd)</a>: 多选题是评估 LLM 医学表现的可靠工具吗？可能不是，LLM 在我们关于虚构腺体的基准测试中获得了 67% 的正确率！是时候调用...</li><li><a href="https://arxiv.org/abs/2402.04362">Neural Networks Learn Statistics of Increasing Complexity</a>: 分布式简单性偏差 (DSB) 假设神经网络首先学习数据分布的低阶矩，然后再转向高阶相关性。在这项工作中，我们提出了...</li><li><a href="https://arxiv.org/abs/2405.14838">From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step</a>: 在利用语言模型进行推理任务时，生成显式的思维链 (CoT) 步骤通常对于实现最终输出的高准确性至关重要。在本文中，我们研究了...
</li>
</ul>

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1247828931017445376)** (3 messages): 

- **排除 lm-eval 输出问题**：一位用户在尝试使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb) 测试多项选择任务的评估时，遇到了无法获取指标计算输出的问题。他们尝试使用了参数 `--limit 40 --output_path tmp --log_samples --predict_only`，但没有得到预期结果。
- **Tmp 文件夹可能包含结果**：另一位成员建议检查结果是否在 tmp 文件夹中，暗示用户的输出可能已经存储在那里。
- **在 LLaMA 3 8B 上运行 MMLU 基准测试**：一位成员寻求关于在自托管的 LLaMA 3 8B instruct 模型上运行 MMLU 基准测试的建议。他们特别请求关于如何为 local-completions 和 local-chat-completions 模型实现 loglikelihood 的指导。

**提到的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb">lm-evaluation-harness/examples/lm-eval-overview.ipynb at main · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness

  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1247725236397277226)** (5 messages): 

- **分享了矩阵乘法教程视频**：一位成员分享了一个[关于矩阵乘法的教育视频](https://cdn.discordapp.com/attachments/843479632320790551/1247252402952995027/SbGXWFlO2K5kTz-_.mp4)。 

- **关于 June's Chaos Udiomusic 的讨论**：一位成员分享了 [June's Chaos Udiomusic 的链接](https://www.perplexity.ai/page/junes-chaos-udiomusic-KPh68eqNQeejjpuyLKEGIg)。没有记录到进一步的讨论。

- **发布了查看更新的链接**：一位成员提供了[查看更新的链接](https://www.perplexity.ai/search/review-the-updates-_.DlS9FHQfCLpvckPrcnBw)。

- **对能够创作 15 分钟长歌曲的模型感兴趣**：一位成员对“能够创作 15 分钟长歌曲的模型”表示兴奋，并表现出“为了连贯性而挑战其极限”的兴趣。
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1247869707692933121)** (10 messages🔥): 

- **了解 GLM-4：一个多语言、多模态对话模型**：新的 [GLM-4](https://x.com/ChatGLM/status/1798292207574901012) 支持 26 种语言，并提供代码执行和长文本推理等高级功能。它是 AI 爱好者和开发者的理想选择，你可以在 [GitHub](https://github.com/THUDM/GLM-4) 上进一步探索。

- **为 Perplexity 用户提供的 Complexity 扩展**：一个名为 Complexity 的新型第三方扩展旨在全面翻新并改进 Perplexity 用户的用户体验。感兴趣的用户可以加入其 [Discord](https://discord.gg/fxzqdkwmWx) 申请访问权限并提供反馈。

- **Nomic-Embed-Vision 表现优于现有模型**：[Nomic-Embed-Vision](https://x.com/nomic_ai/status/1798368463292973361?s=46&t=stOPrwZiN_fxSK0RuC8Flg) 现在为图像、文本和多模态任务提供统一的嵌入空间（embedding space），表现优于 OpenAI CLIP 和 text-embedding-3-small。其权重和代码已公开，可用于独立开发（indie hacking）、研究和实验。

- **引入解耦超球能量损失 (DHEL)**：最近的一篇论文介绍了 [DHEL](https://arxiv.org/abs/2405.18045)，这是一种新型的对比学习（contrastive learning）目标函数，它在保留理论保证的同时简化了正样本的对齐。此处还提供了一个比较各种 InfoNCE 变体的 GitHub 仓库，点击[这里](https://github.com/viig99/ContrastiveLearningLossComparison)查看。

- **LLM 解释器手册**：一位成员分享了 LLM 解释器的 [资源指南](https://x.com/willccbb/status/1798423849870270671)，涵盖了从 vLLM 到 QLoRA 的广泛主题。该指南旨在帮助 LLM 领域的新手。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/mlabonne/abliteration">通过 abliteration 取消任何 LLM 的审查</a>：未找到描述</li><li><a href="https://x.com/willccbb/status/1798423849870270671">来自 will brown (@willccbb) 的推文</a>：过去一年一直在学习 LLM 等知识，将我最喜欢的一些解释器整理成了一份“教科书式”的资源指南。希望我在刚开始学习时就能拥有它，也许它对其他人也有用...</li><li><a href="https://x.com/ChatGLM/status/1798292207574901012">来自 ChatGLM (@ChatGLM) 的推文</a>：🚀 看看 GLM-4！这款开源、多语言、多模态对话模型支持 26 种语言，并提供代码执行和长文本推理等高级功能。非常适合 AI 爱好者和...</li><li><a href="https://x.com/nomic_ai/status/1798368463292973361?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 Nomic AI (@nomic_ai) 的推文</a>：今天，每个 Nomic-Embed-Text 嵌入都变得多模态了。介绍 Nomic-Embed-Vision：- 用于图像、文本和多模态任务的高质量、统一嵌入空间 - 表现优于 OpenAI...</li><li><a href="https://discord.gg/fxzqdkwmWx">加入 Complexity Discord 服务器！</a>：在 Discord 上查看 Complexity 社区 - 与其他 56 名成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://arxiv.org/abs/2405.18045">桥接对比学习中的小批量与渐近分析：从 InfoNCE 到基于核的损失</a>：不同的对比学习 (CL) 损失函数实际上在优化什么？虽然多种 CL 方法已展示出卓越的表示学习能力，但它们内在的差异...</li><li><a href="https://github.com/viig99/ContrastiveLearningLossComparison">GitHub - viig99/ContrastiveLearningLossComparison：比较对比学习中使用的不同 InfoNCE 类型损失的性能。</a>：比较对比学习中使用的不同 InfoNCE 类型损失的性能。- viig99/ContrastiveLearningLossComparison</li><li><a href="https://arxiv.org/abs/2304.12210">自监督学习指南 (A Cookbook of Self-Supervised Learning)</a>：自监督学习被称为智能的暗物质，是推进机器学习的一条充满希望的道路。然而，就像烹饪一样，训练 SSL 方法是一门精妙的艺术，门槛很高...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1247626919609110628)** (84 messages🔥🔥): 

- **Microsoft 涉嫌剽窃创意**：一名成员对 Microsoft 在未适当引用的情况下挪用其创意表示担忧，并分享了一篇相关的 [arXiv paper](https://arxiv.org/pdf/2405.19888)。尽管如此，他们决定将其作为其框架的免费研究，并承认法律追诉存在挑战。
- **AI 的未来与美德之痛 (Virtuous Pain)**：一场关于 AI 在色情行业中的影响以及 LLM 是否需要区分美德与非美德的快乐与痛苦的有趣讨论。对话还探讨了美德在 AI 感知和感觉中的作用。
- **AI 模型的高字符输入容量**：一位用户澄清了一个误解，指出某些模型可以处理超过 200,000 个字符的输入，尽管输出仍限制在 4096 个 tokens。这一见解可以帮助那些使用模型进行大规模文本分析的用户。
- **多款 GPU 对 Flash-attn 的支持**：对话详细讨论了在 Colab 和各种 GPU 上设置 Flash-attn 的挑战。用户讨论了 GPU 兼容性，其中一位指出在 2x4090s 和 H100s 上取得了成功，但在 A10G 或 T4 GPU 上未获成功。
- **Moondream 模型兼容性问题**：一位用户分享了在有限算力上部署 Moondream 模型的权宜之计，讨论了通过实现普通 attention 来绕过 Flash-attn 限制的方法。他们还请求拥有高端 GPU 资源的其他用户协助测试该实现。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5">nomic-ai/nomic-embed-vision-v1.5 · Hugging Face</a>: 无描述</li><li><a href="https://tenor.com/view/no-bugs-bunny-gif-18219884">No Bugs Bunny GIF - No Bugs Bunny - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://colab.research.google.com/drive/1_IenbmaMylGDkGMKeF03XR41S2OP3spn?usp=sharing>">Google Colab</a>: 无描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1247799402148532314)** (3 messages): 

- **NVIDIA NIM 上的 Phi-3 Vision 模型**：Phi-3 Vision 128k-Instruct 模型 *性能与 Gemini-1.0-Pro-V 相似*。可在 [NVIDIA NIM](https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct) 上进行测试；请确保在测试期间不要上传机密信息或个人数据。

- **Openbmb 发布多模态 RLAIF (DPO) 数据集**：[Openbmb 的 RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset?row=2) 现已公开发布。该数据集有助于区分皮革工匠和造纸工匠使用的工具，强调了打孔器和剪刀等特定工具在皮革工艺中的用途。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct">NVIDIA NIM | phi-3-vision-128k-instruct </a>: 立即体验领先模型，构建企业级生成式 AI 应用。</li><li><a href="https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset?row=2">openbmb/RLAIF-V-Dataset · Datasets at Hugging Face</a>: 无描述
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1247701936837492736)** (3 messages): 

- **手动或自动定义 GraphRAG**：在构建 GraphRAG 时，你可以选择显式地自行定义图结构（这需要更多人工投入但提供完全控制），或者使用 LLM 自动提取。每种方法在投入成本和数据表示方面都有其权衡。[链接](https://t.co/sBTgVeh1ft)。

- **观看企业级 RAG 工作坊**：现已提供关于使用 Bedrock 和 @ragas_io 构建企业级 RAG 的综合工作坊视频教程。会议涵盖了 Bedrock 模型的基础知识和 Agentic RAG 设计模式，由来自 @ragas_io 和 AWS 的专家主讲。[视频链接](https://t.co/TRevID609L)。

- **Prometheus-2 作为开源 RAG 评估器**：Prometheus-2 被作为一种开源 LLM 引入，用于评估 RAG 应用，解决了透明度、可控性和成本效益方面的问题。尽管 GPT-4 作为评委评估器（judge evaluator）非常流行，但 Prometheus-2 提供了具有开源优势的替代方案。[链接](https://t.co/BFnmE57OfB)。
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1247637139085066271)** (86 条消息🔥🔥): 

- **引入 Metadata Extractor 模块**：讨论强调了使用 **Metadata Extractor** 模块对长文本块进行消歧的好处。分享了一个[教程](https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC/)，演示了该功能。

- **在 Chroma Database 中存储 DocumentSummaryIndex**：一位成员询问如何在 **Chroma Database** 中持久化 `DocumentSummaryIndex`。他们得到了详细的指导，并指出 **Chroma 不能用作 docstore**。

- **修复 Neo4j 集成的查询引擎错误**：修复了 **Neo4j graph store** 集成中的一个 Bug，即当 `include_text=True` 时查询会失败。提到了一个 [pull request](https://github.com/run-llama/llama_index/pull/13938) 并随后将其合并以解决此问题。

- **为电子商务微调 Embedding 模型**：成员们讨论了如何为电子商务微调（finetune）Embedding 模型 **“intfloat/multilingual-e5-large”**。提供了涉及单个查询对应多个产品的示例数据，说明了训练数据集的正确格式。

- **关于对多个 PDF 使用查询工具的困惑**：提出了一个关于使用 **QueryEngineTool** 管理多个 PDF 的问题。结论是可以使用单个工具有效地处理多个 PDF，并强调了可扩展性和检索效率。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/pull/13938">ensure cypher returns list before iterating by logan-markewich · Pull Request #13938 · run-llama/llama_index</a>：某些 cypher 查询会返回 None 值而不是空列表。让我们在尝试迭代之前确保它是一个列表。</li><li><a href="https://docs.llamaindex.ai">LlamaIndex - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/retrievers/knowledge_graph/#llama_index.core.retrievers.KnowledgeGraphRAGRetriever>).">Knowledge graph - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC/">Extracting Metadata for Better Document Indexing and Understanding - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/use_cases/fine_tuning/#finetuning-embeddings>).">Fine-Tuning - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/finetuning/embeddings/finetune_embedding_adapter/#generate-synthetic-queries>)">Finetuning an Adapter on Top of any Black-Box Embedding Model - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1247829781899247637)** (2 条消息): 

- **在 GPT 响应中苦于非目标素材的干扰**：一位用户分享说他们在 vectorstore 中有 35,000 条消息，但面临非目标素材导致前 100 个响应不可用的问题。由于无法使用 reranking，他们正在寻求动态选择方案。

- **建议按分数过滤**：另一位用户建议通过分数（score）过滤结果，以解决响应中包含非目标素材的问题。
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1247634738303336599)** (67 条消息🔥🔥): 

- **OpenAI 展示 ChatGPT 4 的新语音功能**：一位成员分享了来自 [OpenAI 的视频](https://x.com/ai_for_success/status/1798046241307459673)，展示了 ChatGPT 4 生成不同角色声音的新能力，并称其非常疯狂且令人印象深刻。
  
- **DALLE3 质量骤降**：一位用户注意到本周 DALLE3 的质量显著下降，且通过 API 使用也未能改善这一情况。
  
- **关于非商业 AI 模型的争议**：几位用户讨论了围绕非商业 AI 模型许可证的挫败感，强调一些开发者的动力似乎主要源于利润。一位成员评论道：*“对这些人来说，一切真的都只为了钱，”* 另一位成员则指出训练像 T5 这样的大容量模型需要耗费大量的时间和存储空间。

- **新的 Open-Sci 论文调查了 LLM 推理失败的情况**：社区获悉了 Open-Sci 团队的一篇新论文，该论文探讨了最先进 LLM 中的推理崩溃问题。[arxiv 论文](https://arxiv.org/abs/2406.02061)及其[代码库](https://github.com/LAION-AI/AIW)已发布，供社区反馈。

- **AI 模型中 text encoder 的微调技术**：成员们交流了微调 text encoder 的技巧，建议使用自监督训练（self-supervised training）和应用极简 MLP 层等方法，而不是进行大规模重训。一位成员提到了 Apple 的一篇论文，该论文指出要让朴素的 text encoder 微调生效需要消耗大量资源。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/ai_for_success/status/1798046241307459673">来自 AshutoshShrivastava (@ai_for_success) 的推文</a>: 🚨OpenAI 发布了另一个展示 ChatGPT 4o 新语音功能的视频，简直太疯狂了！它可以生成不同的角色声音。👌 感受 AGI。[📹 OpenAI YT]</li><li><a href="https://x.com/JJitsev/status/1798331909527011548">来自 Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev) 的推文</a>: 矮胖子坐墙头，矮胖子摔下头。国王所有的马，国王所有的兵，能否把矮胖子再拼好？我们随爱丽丝进入 LLM 推理仙境...</li><li><a href="https://arxiv.org/abs/2406.02061">爱丽丝梦游仙境：展示最先进大语言模型中完全推理崩溃的简单任务</a>: 大语言模型 (LLM) 通常被描述为基础模型的实例——即在少样本或零样本方式下，能够跨各种任务和条件进行强有力迁移的模型...</li><li><a href="https://github.com/LAION-AI/AIW">GitHub - LAION-AI/AIW: 用于实验的爱丽丝梦游仙境代码库和原始实验数据</a>: 用于实验的爱丽丝梦游仙境代码库和原始实验数据 - LAION-AI/AIW</li><li><a href="https://marianna13.github.io/aiw/">AIW 项目页面</a>: 爱丽丝梦游仙境：展示最先进大语言模型中完全推理崩溃的简单任务
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1247892970405892106)** (17 messages🔥): 

- **Open-Sci Collective 发布新论文**：一位成员宣布在 Open-Sci Collective 下发布了一篇新论文，重点关注最先进 Large Language Models (LLM) 推理能力的“剧烈崩溃”。他们提供了[论文](https://arxiv.org/abs/2406.02061)链接、其[代码](https://github.com/LAION-AI/AIW)以及项目[主页](https://marianna13.github.io/aiw/)。

- **Karras 新论文介绍**：一位成员分享了一篇 [arXiv 论文](https://arxiv.org/abs/2406.02507)，讨论了 Diffusion 模型在图像质量和生成方面的改进。这种被称为 "autoguidance" 的方法涉及使用训练程度较低的模型版本来引导高质量模型，从而在提高质量的同时保持多样性。

- **关于 Diffusion 模型引导的讨论**：几位成员讨论了 Karras 新论文中的概念，并将其与传统的 classifier guidance 和 classifier-free guidance 方法进行了比较。一位成员建议，该论文可能会受益于探索使用 "detailed uncond" 模型作为一种可能更便宜的替代方案。

- **对 NVIDIA VRAM 使用的批评**：一位成员批评 NVIDIA 据称为了商业模式而浪费 VRAM。

- **LAION 5B 网站访问问题**：一位成员询问如何访问 LAION 5B 网站，并被告知该网站目前已关闭。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.02507">Guiding a Diffusion Model with a Bad Version of Itself</a>：图像生成 Diffusion 模型中主要的关注轴是图像质量、结果的变化量，以及结果与给定条件（例如类别标签）的对齐程度...</li><li><a href="https://github.com/microsoft/unilm/tree/master/textdiffuser">unilm/textdiffuser at master · microsoft/unilm</a>：跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm</li><li><a href="https://x.com/JJitsev/status/1798331909527011548">Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev) 的推文</a>：矮胖子坐墙头，矮胖子摔个大跟头。国王所有的马和所有的兵，能让矮胖子再复原吗？我们随爱丽丝进入 LLM 推理的仙境...</li><li><a href="https://arxiv.org/abs/2406.02061">Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models</a>：LLM 通常被描述为基础模型的实例——也就是说，这些模型能够以 few-shot 或 zero-shot 的方式在各种任务和条件下进行强大的迁移...</li><li><a href="https://github.com/LAION-AI/AIW">GitHub - LAION-AI/AIW: Alice in Wonderland code base for experiments and raw experiments data</a>：爱丽丝梦游仙境实验代码库和原始实验数据 - LAION-AI/AIW</li><li><a href="https://marianna13.github.io/aiw/">AIW 项目页面</a>：爱丽丝梦游仙境：展示最先进大型语言模型推理完全崩溃的简单任务
</li>
</ul>

</div>
  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1247852452930392175)** (1 messages): 

- **刷新时 WhisperSpeech 的 WebSocket 问题**：一位成员在 **whisperfusion pipeline** 中使用的 **WhisperSpeech** 服务遇到了 WebSocket 连接问题。他们在 [StackOverflow](https://stackoverflow.com/questions/78570704/websocket-closes-unexpectedly-in-tts-service-with-multiprocessing-and-asyncio) 上发布了一个更详细的问题，包括代码片段和错误日志。

**提到的链接**：<a href="https://stackoverflow.com/questions/78570704/websocket-closes-unexpectedly-in-tts-service-with-multiprocessing-and-asyncio">WebSocket Closes Unexpectedly in TTS Service with Multiprocessing and Asyncio</a>：我正在 Python 中使用 multiprocessing 和 asyncio 开发一个 TTS（文本转语音）服务。我的主程序使用 queue 集成其他组件。然而，我遇到了一个问题...

  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1247782925471060048)** (10 条消息🔥): 

- **高亮器性能警告**：一位用户建议高亮工具应仅高亮给定的行范围，并包含警告信息，提示 *"长行和大范围可能需要一些时间"*。
  
- **Rust FFI 与互操作性讨论**：一位成员分享了[一段 YouTube 视频](https://youtu.be/O4sVw4YQB24)，题为 *"通过封装函数强化 Rust 的 FFI"*，强调了尽管理解高级内容具有复杂性，但 Rust 在安全系统开发方面的受欢迎程度正在日益增长。

- **Mojo 的向量化（Vectorization）需求**：讨论中提到了期望的功能，例如带有 `unroll_factor` 的 `@vectorize`，以及 Mojo 中的平铺（tiling）和常规向量化能力。

- **Mojo 的后端可行性**：针对新用户的询问，另一位成员建议 Mojo 可以用于后端开发，并列举了其性能、可移植性和安全性。他们分享了一个 [Mojo 编写的 HTTP 服务器](https://github.com/saviorand/lightbug_http/tree/main) 示例。

- **Mojo 路线图引用**：用户引用了 [Mojo 路线图文档](https://docs.modular.com/mojo/roadmap) 来回答有关即将推出的功能和开发优先级的问题，并强调许多语言特性将在未来几个月内推出。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/O4sVw4YQB24">Fortifying Rust's FFI with Enscapsulated Functions - Leon Schuermann</a>：像 Rust 这样内存和类型安全的语言在系统开发中越来越受欢迎。尽管如此，实际系统必须与用...编写的代码进行交互。</li><li><a href="https://docs.modular.com/mojo/roadmap">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>：我们的 Mojo 计划摘要，包括即将推出的功能和我们需要修复的问题。</li><li><a href="https://github.com/saviorand/lightbug_http/tree/main">GitHub - saviorand/lightbug_http: Simple and fast HTTP framework for Mojo! 🔥</a>：适用于 Mojo 的简单且快速的 HTTP 框架！🔥。通过在 GitHub 上创建账户来为 saviorand/lightbug_http 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1247812614294343680)** (1 条消息): 

- **呼吁建立加密库**：一位成员建议 **Mojo** 将受益于加密库，并形容这一添加将是“火热的（fire）”。这突显了扩展 Mojo 功能的兴趣。
- **Mojo 作为 Python 的超集**：一位用户询问 **Mojo** 是否被设计为 Python 的超集。这表明用户有兴趣了解 Mojo 与 Python 的关系及其潜在的增强功能。
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1247634123401465876)** (21 条消息🔥): 

- **Mojo 缺少 Enum 类型，但拥有 Variants**：成员们讨论了 Mojo 目前还不支持 `Enum` 类型，但它拥有 [`Variants`](https://docs.modular.com/mojo/stdlib/utils/variant/Variant)。一位成员建议查看 GitHub 上正在进行的 [讨论](https://github.com/modularml/mojo/issues/43) 以获取更多细节。

- **关于从 Python 转向 Mojo 的热门视频讨论**：一位成员分享了他们对特定 [YouTube 视频](https://www.youtube.com/watch?v=9ag0fPMmYPQ) 的热情，该视频有助于理解底层计算机科学（CompSci）知识，对从 Python 转向 Mojo 的非计算机专业开发者非常有用。该视频因总结了复杂的概念并引导进一步研究而受到称赞。

- **Python 教学轶事与哲学**：一位成员反思了他们向非程序员教授 Python 的经验，强调了超越简单脚本编写并深入理解的重要性，这有助于避免设计陷阱（foot-guns）并使重新架构变得更容易。

- **技术债与社区贡献**：讨论中幽默地建议将“我们要避免设计陷阱”之类的短语变成铃声，同时还提到了像“我们不应该在 [Tensors] 上抢占先机（lick the cookie）”之类的名言。这强调了社区在避免技术债和负责任地做出贡献方面的价值观。

**提到的链接**：<a href="https://github.com/modularml/mojo/issues/43)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 的开发做出贡献。

  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1247635636056494133)** (18 messages🔥): 

- **简化 Range 结构体建议**：一名成员质疑是否有必要拥有三个不同的 Range 结构体。回复建议将其数量减少到两个，并欢迎提交 PR 贡献，但提醒由于依赖关系，应避免触动 `_StridedRangeIterator`。
- **Nightly CI 任务失败导致未更新**：由于 S3 故障导致 CI 任务崩溃，Nightly 更新未能进行。官方承诺将尝试重新启动 CI 任务。
- **新版 Mojo 编译器发布**：宣布了新的 Nightly Mojo 编译器更新（`2024.6.512`）。此次更新包括将 `Coroutine.__await__` 设为 consuming（消耗型）并移除隐式转换等更改，并提供了指向 [当前 changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 和 [原始 diff](https://github.com/modularml/mojo/compare/1febed7a4e55feea94d9a10f06e27ea7441e9e9d...a752f0d7279d26ecf34ff1efb6e0b471b3e9afe5) 的链接。
- **Matrix 结构体与 SIMD 更新**：成员们讨论了在使代码适配最近 SIMD 操作更改时遇到的困难。一位成员分享了他们的 `Matrix` 结构体代码，并根据 changelog 的说明，就如何从 `DTypePointer` 过渡到 `SIMD` 寻求建议。
- **在 VSCode 中切换 Nightly 和 Release 版本**：一名成员遇到了即使在终端切换回 Release 版本后，VSCode 仍在使用 Nightly 构建的问题。他们提议使用独立的工作区作为潜在解决方案，并讨论了切换 LSP 扩展的方法。

**提到的链接**：<a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog-released.md#-legendary">mojo/docs/changelog-released.md at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1248017713150955631)** (1 messages): 

- **投资者渴望机器人领域的 ChatGPT**：链接的 [Substack 文章](https://www.newcomer.co/p/why-investors-cant-get-enough-of) 解释了**投资者如何迫切希望**在**机器人 AI** 领域找到一家杰出的基础模型公司。他们的目标是投资创新的机器人公司，同时避免硬件开发的风险。

**提到的链接**：<a href="https://www.newcomer.co/p/why-investors-cant-get-enough-of">为什么投资者现在对 AI 机器人交易趋之若鹜</a>：风投们押注机器人技术是初创公司仍能对抗 OpenAI 的一个领域。

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1247631783441469511)** (40 messages🔥): 

- **泄密事件引发 AI 国家安全辩论**：一位用户提到有人因泄密被解雇，并强调了他们的声明，即*低估了商业机密对 AI 国家安全的重要性*。
- **对前沿实验室关于 AGI 的信心表示怀疑**：用户对 OpenAI 和 Anthropic 在 3-5 年内实现研究员级 AI 能力的信心表示怀疑，*想知道这是否仅仅是外推法和激励机制不一致的结果*。
- **微软 CTO 声称即将推出的 AI 模型将有重大进展**：来自微软的 [Kevin Scott](https://x.com/tsarnick/status/1798167323893002596) 暗示，即将推出的 AI 模型在记忆和推理方面的水平可能通过 PhD 资格考试，并将 GPT-4 等当前模型等同于解决高中 AP 考试的水平。
- **伯克利的 PhD 考试难度大但不一致**：讨论强调了不同机构之间 PhD 考试难度的不一致，伯克利被描述为*入学考试很难但论文答辩几乎不存在*，并且有一个著名的案例是 75% 的人未能通过预考。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/tsarnick/status/1798167323893002596">Tsarathustra (@tsarnick) 的推文</a>：微软 CTO Kevin Scott 表示，他在即将推出的 AI 模型早期预览中看到的是，系统的记忆和推理水平可以达到通过 PhD 资格考试的程度。</li><li><a href="https://x.com/natolambert/status/1798073830906486945">Nathan Lambert (@natolambert) 的推文</a>：这会让搞 AGI 扩展的人感到紧张吗？仅限错误答案。这是 @TheXeophon 绘制的精美 LLM 扩展趋势线。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1247636066840608778)** (6 messages): 

- **AGI 讨论引发反感**：成员们对 AGI 爱好者和末日论者（doomers）都表示了挫败感。一位成员评论道：*"AGI people and doomers are hella annoying."*
- **悬赏 $25 解决问题**：Nathan Lambert 为解决一个与 [不同 batch size 导致 rewardbench.py 结果不一致](https://github.com/allenai/reward-bench/issues/137) 相关的问题提供了 $25 的奖励。他表示：*"honestly willing to pay $25 or more if someone solves this issue lol."*
- **被诅咒的模型组件**：Lambert 将 AutoModelForSequenceClassification 描述为“有点被诅咒”。目前正在努力观察微调是否能带来性能提升。

**Link mentioned**: <a href="https://github.com/allenai/reward-bench/issues/137">rewardbench.py results are different for different batch size for beaver-7b · Issue #137 · allenai/reward-bench</a>: Thank you for the great work on rewardbench, as it&#39;s been super helpful in evaluating/researching reward models. I&#39;ve been wrapping your rewardbench.py code to run the reward models published ...

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

420gunna: 👍
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1247646690790346803)** (40 messages🔥): 

- **关于停止免费 API 的传闻**：一名成员询问免费 API 是否即将停止。另一名成员建议在相应频道查询准确信息，并质疑了这些传闻的来源。

- **讨论与 LLM 的多用户聊天线程**：成员们辩论了涉及 Bot 的多用户聊天线程的可行性。有人指出这种设置可能会让 LLM 感到困惑，但建议在用户消息前加上用户名作为潜在解决方案。

- **对 React 聊天组件的兴趣**：一名成员询问是否有好用的 React 聊天组件，以及 Cohere 是否拥有。有人指出 [Cohere Toolkit](https://github.com/cohere-ai/cohere-toolkit/?ref=cohere-ai.ghost.io) 是开源的，但并未使用 React，尽管其聊天框可能是用 React 编写的。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/cohere-ai/cohere-toolkit/?ref=cohere-ai.ghost.io">GitHub - cohere-ai/cohere-toolkit at cohere-ai.ghost.io</a>: Cohere Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications. - GitHub - cohere-ai/cohere-toolkit at cohere-ai.ghost.io</li><li><a href="https://github.com/cohere-ai/cohere-toolkit/?ref=cohere-ai.ghost.i">GitHub - cohere-ai/cohere-toolkit at cohere-ai.ghost.i</a>: Cohere Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications. - GitHub - cohere-ai/cohere-toolkit at cohere-ai.ghost.i
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1247876355845259335)** (9 messages🔥): 

- **目标模块的 OOM 问题**：一位用户在尝试于 2xT4 16GB 上运行目标模块时遇到了 OOM（显存溢出）错误，并报告在尝试调整设置时 loss 为 0.0。
- **HuggingFace FineWeb 数据集发布**：成员们重点介绍了源自 CommonCrawl 并以宽松许可证发布的 [HuggingFace FineWeb 数据集](https://huggingface.co/HuggingFaceFW)，旨在降低预训练高性能大语言模型的门槛。详情可见其 [技术报告](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)。
- **对 15T 数据集的兴奋**：用户讨论了来自 CommonCrawl 的 15 万亿 token 精选优化数据集的可用性，并指出未来将推出多语言版本。
- **对访问限制的复杂反应**：虽然对可获取的数据集感到兴奋，但也有评论指出缺乏财务和计算资源来充分利用它。
- **对 GLM-4 9B 模型的兴趣**：成员们对 GLM-4 9B 模型的使用经验感到好奇，但目前尚未提供具体反馈。

**Link mentioned**: <a href="https://huggingface.co/HuggingFaceFW">HuggingFaceFW (HuggingFaceFW)</a>: no description found

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1247821924223160331)** (20 条消息🔥): 

- **使用 Deepspeed zero2 成功微调 Llama3**：一位成员确认使用 **Deepspeed zero2** 成功微调了 Llama3 且未遇到任何问题，并提到他们使用了默认配置文件并对数据进行了一些更改。
- **微调时选择 Qlora 而非 Lora**：当被问及时，该成员表示他们在微调中使用了 **Qlora**，而不是 Lora。
- **Deepspeed 的命令行执行**：对话强调了在运行 Deepspeed 时，**命令行**优于 notebook，并分享了所使用的命令：*"axolotl launch config.yml and --deepspeed deepspeed_configs/zero2.json"*。
- **尝试解决 Loss 问题**：另一位成员对遇到 loss 为 0.0 的情况表示沮丧，并得到了关于使用 Deepspeed 配置的指导，希望能解决该问题。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/1247975595057483900)** (1 条消息): 

- **Runpod 缓慢的启动时间令用户沮丧**：一位成员对 **Runpod** 表示不满，因为其模型启动时间过长，称“启动模型（仅 14b）需要大约一分钟”，并指出这段延迟的每一秒都在计费。他们正在咨询其他的 serverless 供应商。
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1247630192264872018)** (28 条消息🔥): 

- **LiveKit 为 AI 传输层融资 2250 万美元**：[LiveKit](https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww) 宣布获得 2250 万美元 A 轮融资，用于构建 AI 传输层，强调实时语音和视频将彻底改变人机交互。GPT-4 的演示被认为是投资者入局的关键时刻。
  
- **Twelve Labs 获得 5000 万美元融资，发布 Marengo 2.6**：[Twelve Labs](https://www.prweb.com/releases/twelve-labs-earns-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html) 在 A 轮融资中获得 5000 万美元，并发布了具有原生 API 支持的多模态基础模型 Marengo 2.6。本轮融资由 NEA 和 NVentures 领投。

- **微软的 Aurora 旨在提供更好的天气预报**：[Microsoft Research](https://x.com/MSFTResearch/status/1797662278394827029) 推出了 Aurora，这是一款 AI 基础模型，旨在提高天气预报的准确性并减轻气候变化的影响。该模型承诺提供更快、更准确的预测。

- **Teknium 质疑 OpenAI 在对齐方面的透明度**：[Teknium 的推文](https://x.com/Teknium1/status/1798107776885105003) 表达了对 OpenAI 在发布对齐奖励、审核分类器和 RLHF 模型方面缺乏透明度的担忧。他透露，当前的架构将奖励模型嵌入到 LLM 架构本身中，这是一种已知的 RL 技巧。

- **Storyblok 为 AI 驱动的内容平台融资 8000 万美元**：[Storyblok](https://x.com/alexadark/status/1798031781377298751?s=46&t=90xQ8sGy63D2OtiaoGJuww) 在 C 轮融资中筹集了 8000 万美元，用于开发 AI 驱动的端到端内容平台。新的 Ideation Room 正在公开测试中，将 AI 能力与内容管理相结合。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.prweb.com/releases/twelve-labs-earns-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html">Twelve Labs 获得由 NEA 和 NVIDIA 的 NVentures 领投的 5000 万美元 A 轮融资，旨在构建多模态 AI 的未来</a>：/PRNewswire-PRWeb/ -- 视频理解公司 Twelve Labs 今日宣布已筹集 5000 万美元 A 轮融资，以推动持续的...</li><li><a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行持续 LLM 预训练</a>：通过 Unsloth 使用 Llama 3、Phi-3 和 Mistral 进行持续预训练，让模型学习新语言。</li><li><a href="https://x.com/FanaHOVA/status/1798389878474027342">来自 Alessio Fanelli (@FanaHOVA) 的推文</a>：.@StackOverflow 编程 AI 助手调查 📊 ChatGPT (84%) 和 Copilot (49%) 遥遥领先；祝贺 @_mohansolo 和 @codeiumdev 成为这里的排名第一的初创公司。@cursor_ai 0% 简直疯狂。我曾...</li><li><a href="https://x.com/alexadark/status/1798031781377298751?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Alexandra Spalato (@alexadark) 的推文</a>：非常激动地分享 @storyblok 已筹集 8000 万美元的 C 轮融资！🚀 正在美国和欧洲进行扩张，以构建首个 AI 驱动的端到端内容平台。查看...</li><li><a href="https://x.com/willdepue/status/1797878877882331153">来自 will depue (@willdepue) 的推文</a>：（郑重声明，我是在日本城的卡拉 OK 店里发这条推文的，所以不保证逻辑一致性）我发现我从根本上不同意这个概念（这已经困扰我几个月了...）</li><li><a href="https://x.com/Teknium1/status/1798210302221386055">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：@dmvaldman @willdepue @sandersted @jachiam0 嵌入到实际架构中的 Reward Model 可以引导每个 Next Token 的选择，使其符合人类反馈结果的最高价值。</li><li><a href="https://x.com/CFGeek/status/1798216430313480601">来自 Charles Foster (@CFGeek) 的推文</a>：@Teknium1 @willdepue @sandersted @jachiam0 据我所知，这是 RL 中一个已知的技巧，并非 OpenAI 特有的。Reward Model 通常使用 LLM 的原始权重进行初始化，或者只是一个额外的...</li><li><a href="https://x.com/Teknium1/status/1798107776885105003">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：那么对齐 Reward Model 在哪里？审核分类器在哪里？为什么它被锁定在禁止任何其他模型使用的服务条款（TOS）之后？让其他人能够对他们的模型进行 RLHF 的代码在哪里？在哪里...</li><li><a href="https://x.com/Teknium1/status/1798110728546902492">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：前 OpenAI 员工已经告诉我，你们目前的架构正以某种方式将 Reward Model 嵌入到模型本身中，难道这种架构不值得发布或者至少发一篇论文让其他人...</li><li><a href="https://x.com/willdepue/status/1797871645774032931?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 will depue (@willdepue) 的推文</a>：对齐领域的人们已经忘记了 AI Safety 的主要目标是构建与用户意图对齐的系统，而不是与创建者的意图对齐。这是一个容易得多的问题。</li><li><a href="https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 dsa (@dsa) 的推文</a>：今天我们宣布 LiveKit 获得 2250 万美元 A 轮融资，用于构建 AI 的传输层。这并不是一次容易的融资。去年年底，我们向投资者推销实时语音和视频将成为...</li><li><a href="https://x.com/MSFTResearch/status/1797662278394827029">来自 Microsoft Research (@MSFTResearch) 的推文</a>：Aurora 是来自 Microsoft Research 的新型 AI 基础模型，通过实现更快、更准确的预测，它可以改变我们预测和缓解极端天气事件以及气候变化影响的能力...</li><li><a href="https://x.com/udiomusic/status/1798448478877794574">来自 udio (@udiomusic) 的推文</a>：Audio-prompting 现已在 Udio 上线。在下方展示你们的使用方式 👇</li><li><a href="https://www.latent.space/p/fastai">微调的终结 —— 对话 Fast.ai 的 Jeremy Howard</a>：立即收听 | 关于快速学习 AI 以及 AI 如何快速学习，以更少的资源进行更多 Deep Learning 的使命，发明 ULMFiT 以及为什么它现在是错误的，以及如何玩转 AI Discords 游戏
</li>
</ul>

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1247983194096140419)** (1 条消息): 

- **Anthropic 的 Monosemanticity 演讲即将开始**：20 分钟后将举行一场关于 Scaling Monosemanticity 的演示。分享了 [Scaling Monosemanticity 活动](https://lu.ma/p5ctl2u6) 的详细信息和 [活动图片链接](https://images.lumacdn.com/cdn-cgi/image/format=auto,fit=cover,dpr=2,quality=75,width=400,height=400/event-covers/mq/b7a9e5d5-cbd9-4546-a668-972d498d2186)。


**提到的链接**：<a href="https://lu.ma/p5ctl2u6">LLM Paper Club (Anthropic's Scaling Monosemanticity) · Zoom · Luma</a>：Vibhu 将涵盖 https://www.anthropic.com/news/mapping-mind-language-model 以及……

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1247706888204718244)** (14 条消息🔥): 

- **OpenInterpreter 中技能持久化的困扰**：成员们讨论了让 OpenInterpreter 在不同会话间记住技能的挑战。有人建议 *"告诉 OI 创建一个新技能"*，但也承认技能在会话外无法持久存在，并建议改为存储脚本。

- **对 RAG 可靠性的怀疑**：一位成员对使用 RAG (Retrieval-Augmented Generation) 来动态更改系统提示词表示怀疑。他们认为 RAG *"似乎仍然太不精确，无法信任"*，尽管 Token 成本更高，但更倾向于使用传统的 Embedding/向量数据库。

- **OpenAI 数据的隐私担忧**：一位用户询问使用 OpenInterpreter 进行测试是否能确保数据隐私。另一位成员确认 *"与 OpenAI API 的所有通信都是私密的"*，但建议使用本地模型以获得额外的隐私保障。

- **活动通知 - House Party 改期**：发布了一项公告，将 House Party 移至周五。为成员提供了 [Discord 活动链接](https://discord.com/invite/vgCdP9b3?event=1237424662951100506)。

- **发货更新请求被重定向**：成员们被引导至特定频道查看置顶的更新消息以获取发货进度。一位用户被建议在正确的频道中发布他们的更新请求。
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1247700173631782942)** (4 条消息): 

- **配合 LLM 运行 O1 开发预览版**：一位成员询问是否有人弄清楚了如何配合 **Anthropic** 等其他 LLM 运行 **O1 开发预览版**，并指出需要一个 Vision 模型。另一位成员建议将其与 **Ollama** 一起运行，但在某些 OS 设置上提到了死循环问题。
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1247780724698910760)** (2 条消息): 

- **终端语音助手引起关注**：一位成员分享了 [GitHub - 0xrushi/Terminal-Voice-Assistant](https://github.com/0xrushi/Terminal-Voice-Assistant) 的链接，并询问是否可以在 **01** 中实现类似的功能。项目描述显示这是一个用于在终端创建语音助手的开发工具。
- **需要合成数据生成工具**：有人表示有兴趣寻找一种 **开源工具**，能够从语料库生成用于微调的合成问答对（QnA pairs）。在分享的消息中没有提供具体的建议或回复。

**提到的链接**：<a href="https://github.com/0xrushi/Terminal-Voice-Assistant">GitHub - 0xrushi/Terminal-Voice-Assistant</a>：通过在 GitHub 上创建账户来为 0xrushi/Terminal-Voice-Assistant 的开发做出贡献。

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1247640237350326363)** (14 条消息🔥): 

- **200 美元悬赏替换 tqdm**：George Hotz 提供 200 美元悬赏，征集能用 4 行代码替换 tqdm 的方案，他不喜欢 tqdm 的门控导入（gated import）。Trirac 接受了挑战并提交了 PR，尽管他指出在高流速下 it/s 会有轻微偏差。
- **Tinygrad 统计网站宕机**：George Hotz 询问为什么 stats.tinygrad.org 网站返回 404 错误，表明该网站目前不是公开状态。
- **Tinygrad 文档更新**：George Hotz 宣布了 Tinygrad 文档的更新，包括训练章节、库结构图、JIT 详情和代码走读。他征求了关于人们还想看到什么的进一步建议。
- **Tinygrad 规范与员工筛选**：George Hotz 澄清说，悬赏的一个主要目标是识别有能力在不确定性下工作的潜在全职员工。他强调大部分工作涉及定义问题，一旦完整的 Tinygrad 规范（spec）最终确定，重新实现它大约需要两个月。
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1247984470146027642)** (4 messages): 

- **将 CUDA kernel 映射到 Python 代码仍然很复杂**：一位用户询问如何将 CUDA 调试输出与相应的 Python 代码关联起来。George Hotz 提到已经有 *一些 PR 旨在实现这一功能*，但 *尚未合并到 master 分支*，并且 *1.0 版本需要此功能*。
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1247701408950648873)** (12 messages🔥): 

- **成员批评文档过时**：一位成员表达了沮丧，声称 *“网络上所有关于 OpenAI 和 LangChain 的文档都严重过时了。API 发生了很大变化。”* 另一位成员建议不要放弃，并建议直接查看技术栈。
- **引发 MongoDB vs Chroma DB 的辩论**：一位成员询问关于将 MongoDB 用作向量数据库的问题，并请求一个 JSON 文件示例。另一位成员回应澄清道 *“MongoDB 存储 JSON，而 Chroma DB 存储 Embeddings”*，并建议咨询 MongoDB 文档或 ChatGPT 以寻求帮助。
- **Verba 获得推荐**：一位成员分享了 [Verba](https://github.com/weaviate/Verba) 的 GitHub 链接，这是一个由 Weaviate 驱动的检索增强生成 (RAG) 聊天机器人，并询问是否有人有使用经验。
- **SQL Agent 问题查询**：一位成员报告了一个问题，即 SQL Agent 在执行了操作后仍产生空的最终答案，寻求解决此问题的建议。
- **LangChain 指南亮点**：一位成员分享了一份关于从非结构化文本构建知识图谱的 [LangChain 指南](https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/)，并询问关于在 Ollama 模型中使用 `LLMGraphTransformer` 的事宜。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/">构建知识图谱 | 🦜️🔗 LangChain</a>：在本指南中，我们将介绍基于非结构化文本构建知识图谱的基本方法。构建的图谱随后可用作 RAG 应用程序中的知识库。</li><li><a href="https://github.com/weaviate/Verba">GitHub - weaviate/Verba: 由 Weaviate 驱动的检索增强生成 (RAG) 聊天机器人</a>：由 Weaviate 驱动的检索增强生成 (RAG) 聊天机器人 - weaviate/Verba
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1247908429251477546)** (1 messages): 

- **使用 VisualAgents 进行拖拽式 Agent 工作流**：一位用户分享了一个名为“使用 Visual Agents 进行拖拽式 Agent 模式和 LLM 链”的 [YouTube 视频](https://www.youtube.com/watch?v=IVFsANcfqaA)。该演示展示了如何使用基于 LangChain 构建的 VisualAgents 将 Agent 流模式拖拽到画布上并运行，强调了构建和复用自定义 Agent 流的便利性。

**提及的链接**：<a href="https://www.youtube.com/watch?v=IVFsANcfqaA">使用 Visual Agents 进行拖拽式 Agent 模式和 LLM 链</a>：在此演示中，我将一个 Agent 流模式拖拽到画布上并运行它。你可以轻松构建自定义 Agent 流并将其保存为模式以供复用...

  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1247680329364537436)** (13 messages🔥): 

- **Rope Scaling 与 OpenRouter 的兼容性**：一位成员询问 Rope Scaling 是否与 OpenRouter 兼容，或者是否需要像 LMStudio 这样的其他工具。建议是由于潜在的 GPU 限制，最好在本地运行。

- **Codestral 并非代码专业化的首选**：一位成员询问关于尝试 Codestral 的事宜。另一位成员提到有更好的代码专业模型，在尺寸和性能上更高效，并特别推荐了频道 <#1230206720052297888> 中的模型。

- **OpenRouter 遇到 502 Bad Gateway 错误**：多位用户讨论了在跨多个模型大规模生成合成数据时遇到 Cloudflare 的 502 Bad Gateway 错误。一位成员确认这并非由于突发限制引起，并指出问题出在 `messages` 中 `content` 的格式化上，该问题目前已解决。

- **错误发生期间使用的模型列表**：错误涉及的模型列表包括来自 Nous Research、Mistral、Cognitive Computations、Microsoft 和 Meta-Llama 的多种模型。问题不在于请求数量，而在于消息中特定的内容格式。
  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1247992309128364143)** (5 条消息): 

- **不要错过 6 月 11 日的 Human Feedback Foundation 活动**：一位成员提醒大家参加即将于 6 月 11 日举行的 **Human Feedback Foundation** 活动。活动详情见 [Eventbrite](https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator)。

- **Human Feedback Foundation 的 YouTube 链接**：邀请成员查看往期会议录像，嘉宾来自 UofT、Stanford 和 OpenAI，链接见 [YouTube](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg)。该基金会专注于将人类反馈融入开源 AI 项目、AI 治理和 AI 安全研究。

- **LLM Reading Group 的独立 Discord**：一位成员询问是否有 LLM Reading Group 的独立 Discord。另一位成员尝试发送直接邀请，但由于隐私设置未能成功。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg">Human Feedback Foundation</a>: Human Feedback Foundation 的使命是将人类反馈构建到开源 AI 项目中。我们寻求：通过支持开源开发和政策倡议，实现公众对 AI 的投入...</li><li><a href="https://www.eventbrite.ca">Eventbrite</a>: Eventbrite - 发现最佳本地活动和活动指南</li><li><a href="https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator">LLM Reading Group (3月 5, 19; 4月 2, 16, 30; 5月 14, 28; 6月 11)</a>: 来见见 LLM/NLP 研究领域一些开创性论文的作者，听他们分享他们的工作
</li>
</ul>

</div>
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1247653537953878146)** (3 条消息): 

- **RISC-V 向量处理已到来**：YouTube 视频 ["The Magic of RISC-V Vector Processing"](https://www.youtube.com/watch?v=Ozj_xU0rSyY) 详细介绍了现已批准的 **1.0 RISC-V Vector Specification**，并讨论了该新技术的早期芯片实现。视频深入探讨了 CPU 中向量处理的效用和潜力。

- **Right to Warn AI 项目强调风险并呼吁问责**：[Right to Warn AI 网站](https://righttowarn.ai) 概述了 AI 技术的潜在危险，如现有的不平等、操纵、虚假信息以及可能导致人类灭绝的失控。该网站强调需要科学家、政策制定者和公众的监督，认为目前的企业治理结构是不够的。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=Ozj_xU0rSyY">The Magic of RISC-V Vector Processing</a>: 1.0 RISC-V Vector Specification 现已批准，首批使用新规范的芯片开始上市。我将介绍其效用...</li><li><a href="https://righttowarn.ai">A Right to Warn about Advanced Artificial Intelligence</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1247877359231570032)** (3 条消息): 

- **讨论开发德语版 "PaliGema" 克隆的可行性**：一位成员询问创建德语版 PaliGema 的难度，将其戏称为 *"Sauerkraut Gemma"*，并询问仅替换基础 Gemma 是否足够。
- **分享 PaliGemma 模型链接**：另一位成员指向了 [PaliGemma-3B-Chat-v0.2](https://huggingface.co/BUAADreamer/PaliGemma-3B-Chat-v0.2) 模型，并建议在翻译数据集后采用类似的方法，即 *"冻结视觉部分并训练聊天部分"*。

**提到的链接**: <a href="https://huggingface.co/BUAADreamer/PaliGemma-3B-Chat-v0.2">BUAADreamer/PaliGemma-3B-Chat-v0.2 · Hugging Face</a>: 未找到描述

  

---



### **LLM Perf Enthusiasts AI ▷ #[resources](https://discord.com/channels/1168579740391710851/1168760058822283276/1247985181495787672)** (1 条消息): 

- **发布综合 AI 资源指南**：一位成员分享了 William Brown 编写的名为 [genai-handbook](http://genai-handbook.github.io/) 的资源指南。该指南旨在作为学习现代 AI 系统关键概念的手册，将各种解释性资源组织成教科书式的呈现方式。

**提到的链接**: <a href="http://genai-handbook.github.io/">GenAI Handbook</a>: 未找到描述

  

---



---



---



---



{% else %}


> 完整的频道细分内容已在邮件中截断。
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}