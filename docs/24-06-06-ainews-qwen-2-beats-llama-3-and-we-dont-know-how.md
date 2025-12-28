---
companies:
- alibaba
- groq
- meta-ai-fair
date: '2024-06-06T22:33:41.101639Z'
description: '**阿里巴巴**发布了 **Qwen 2** 模型，采用 Apache 2.0 协议，声称其在开源模型中表现优于 **Llama 3**，支持
  **29 种语言**，并在 **MMLU 82.3** 和 **HumanEval 86.0** 等基准测试中取得了优异成绩。**Groq** 展示了 **Llama-3
  70B** 极快的推理速度，达到了 **40,792 tokens/s**，并能在 200 毫秒内处理 4 篇维基百科文章。针对 **GPT-4** 神经活动解释的**稀疏自编码器
  (SAEs)** 研究提出了新的训练方法、评估指标和缩放法则。**Meta AI** 发布了 **No Language Left Behind (NLLB)**
  模型，能够实现 **200 种语言**（包括低资源语言）之间的高质量翻译。“我们的后训练阶段遵循‘以最少人工标注实现可扩展训练’的原则，”并强调了针对数学的拒绝采样和针对编程的执行反馈等技术。'
id: faab9a69-3c95-4761-b1bb-d93d58c453b8
models:
- qwen-2
- llama-3
- llama-3-70b
- gpt-4
- nllb
original_slug: ainews-qwen-2-beats-llama-3-and-we-dont-know-how
people:
- philschmid
- huybery
- jonathanross321
- awnihannun
- gdb
- nabla_theta
- ylecun
title: Qwen 2 击败了 Llama 3（而我们不知道它是如何做到的）
topics:
- multilinguality
- benchmarking
- inference-speed
- sparse-autoencoders
- scaling-laws
- post-training
- instruction-following
- rejection-sampling
- execution-feedback
- model-release
- multilingual-models
- model-training
---

<!-- buttondown-editor-mode: plaintext -->**又一个没有数据集细节的模型发布。**

> 2024年6月5日至6月6日的 AI 新闻。
我们为您检查了 7 个 subreddit、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**408** 个频道，**2450** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**304 分钟**。

随着 [Qwen 2](https://qwenlm.github.io/blog/qwen2/) 采用 Apache 2.0 协议，阿里巴巴现在声称在开源模型桂冠的争夺中全面击败了 **Llama 3**：

 
![image.png](https://assets.buttondown.email/images/99225fac-83f1-4251-9fed-565bd6757bfc.png?w=960&fit=max)
 

关于数据集的细节为零，因此很难了解他们是如何实现这一点的，但他们确实透露了一些关于 post-training（后期训练）的线索：

> 我们的 post-training 阶段设计原则是在最少人工标注的情况下进行可扩展训练。
> 
> 具体来说，我们研究了如何通过各种自动化对齐策略获取**高质量、可靠、多样化且具有创造性的演示数据和偏好数据**，例如：
> 
> - **针对数学的 [rejection sampling](https://arxiv.org/pdf/2308.01825)（拒绝采样）**，
> - **针对代码的执行反馈**，以及
> - **针对创意写作的指令遵循和反向翻译**，
> - **针对角色扮演的 [scalable oversight](https://arxiv.org/pdf/2401.12474)（可扩展监督）**等。
> 
> 正如下表所示，这些集体努力显著提升了我们模型的能力和智能。

他们还发布了一篇关于[《使用 Qwen-Agent 将 LLM 的上下文从 8k 扩展到 1M》](https://qwenlm.github.io/blog/qwen-agent-2405/)的文章。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有摘要均由 Claude 3 Opus 完成，从 4 次运行中择优录取。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**Qwen2 开源 LLM 发布**

- **Qwen2 模型发布**：[@huybery](https://twitter.com/huybery/status/1798747031185559921) 宣布发布 5 种尺寸（0.5B, 1.5B, 7B, 57B-14B MoE, 72B）的 Qwen2 模型，包括 Base 和 Instruct 版本。模型具有 **29 种语言的多语言能力**，并在学术和聊天基准测试中实现了 **SOTA 性能**。除 **72B 外，均按 Apache 2.0 协议发布**。
- **性能亮点**：[@_philschmid](https://twitter.com/_philschmid/status/1798747595411779776) 指出 Qwen2-72B 实现了 **MMLU 82.3, IFEval 77.6, MT-Bench 9.12, HumanEval 86.0**。Qwen2-7B 实现了 **MMLU 70.5, MT-Bench 8.41, HumanEval 79.9**。在 MMLU-PRO 上，Qwen2 得分为 **64.4，超过了 Llama 3 的 56.2**。
- **多语言能力**：[@huybery](https://twitter.com/huybery/status/1798747042958967253) 强调了 Qwen2-7B-Instruct 强大的多语言表现。据 [@_philschmid](https://twitter.com/_philschmid/status/1798747598398132356) 介绍，这些模型在 **29 种语言中进行了训练，包括欧洲、中东和亚洲语言**。

**Groq 在大型 LLM 上的推理速度**

- **Llama-3 70B Tokens/s**：[@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1798746103766200602) 报告称，Groq 在 **Llama-3 70B 上实现了 40,792 tokens/s 的输入速率**，在整个 7989 token 上下文长度中使用了 FP16 乘法和 FP32 累加。
- **200 毫秒运行 Llama 70B**：[@awnihannun](https://twitter.com/awnihannun/status/1798765365117505677) 对这一成就进行了直观描述，指出 Groq 在 **200 毫秒内（大约眨眼之间）处理了约 4 篇维基百科文章规模的 Llama 70B 任务**，采用 16 位精度和 32 位累加（无损）。

**用于 GPT-4 可解释性的稀疏自编码器训练方法**

- **改进的 SAE 训练**：[@gdb](https://twitter.com/gdb/status/1798764692669911142) 分享了一篇关于改进**大规模训练稀疏自编码器（SAEs）以解释 GPT-4 神经活动**的方法的论文。
- **新的训练栈和指标**：[@nabla_theta](https://twitter.com/nabla_theta/status/1798763600741585066) 介绍了一套用于 **SAEs 的 SOTA 训练栈**，并在 GPT-4 上训练了一个 16M latent 的 SAE 以展示扩展性。他们还提出了 **MSE/L0 损失之外的新 SAE 指标**。
- **扩展定律与指标**：[@nabla_theta](https://twitter.com/nabla_theta/status/1798765396113477801) 发现了**关于自编码器 latent 数量、稀疏度和计算量的清晰扩展定律（scaling laws）**。更大的主体模型具有更浅的扩展定律指数。探索了诸如下游损失、探针损失（probe loss）、消融稀疏性和可解释性等指标。

**Meta 的 No Language Left Behind (NLLB) 模型**

- **NLLB 模型详情**：[@AIatMeta](https://twitter.com/AIatMeta/status/1798420492774432769) 宣布了发表在 Nature 上的 NLLB 模型，该模型可以**在 200 种语言（包括低资源语言）之间直接进行高质量翻译**。
- **工作意义**：[@ylecun](https://twitter.com/ylecun/status/1798446014723973333) 指出 NLLB 能够在**训练数据稀疏的情况下，为 200 种语言（包括许多低资源语言）提供任意方向的高质量翻译**。

**Pika AI 的 B 轮融资**

- **8000 万美元 B 轮融资**：[@demi_guo_](https://twitter.com/demi_guo_/status/1798500975671759001) 宣布 Pika AI 完成了**由 Spark Capital 领投的 8000 万美元 B 轮融资**。Guo 向投资者和团队成员表示了感谢。
- **招聘与未来计划**：[@demi_guo_](https://twitter.com/demi_guo_/status/1798499472563110029) 回顾了过去一年的进展，并预告了**今年晚些时候将发布的更新**。Pika AI 正在**寻找研究、工程、产品、设计和运营方面的人才**（[链接](https://twitter.com/demi_guo_/status/1798501857041727748)）。

**其他值得关注的发展**

- **Anthropic 的选举诚信努力**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1798732041925914903) 发布了关于其**测试和缓解选举相关风险流程**的详细信息。他们还分享了**用于测试其模型的评估样本**。
- **Cohere 启动创业计划**：[@cohere](https://twitter.com/cohere/status/1798688627007873124) 启动了一项**创业计划，旨在支持利用 AI 解决现实业务挑战的早期公司**。参与者可以获得折扣访问权限、技术支持和市场曝光。Cohere 还发布了一个**针对企业级前沿模型的 Cookbook 库**，适用于 Agent、RAG 和语义搜索等应用（[链接](https://twitter.com/cohere/status/1798453445076385968)）。
- **用于 RAG 评估的 Prometheus-2**：[@llama_index](https://twitter.com/llama_index/status/1798454426904244588) 推出了 Prometheus-2，这是一个**用于评估 RAG 应用的开源 LLM**，可作为 GPT-4 的替代方案。它可以处理直接评估、成对排名和自定义标准。
- **LangChain x Groq 集成**：[@LangChainAI](https://twitter.com/LangChainAI/status/1798576376255250534) 宣布即将举行一场关于**使用 LangChain 和 Groq 集成构建 LLM Agent 应用**的网络研讨会。
- **Databutton AI 工程师平台**：[@svpino](https://twitter.com/svpino/status/1798701450396402157) 分享了 Databutton 推出的 **AI 软件工程师平台**，旨在帮助用户根据业务想法构建具有 React 前端和 Python 后端的应用程序。
- **微软的 Copilot+ PC**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1798749403240382604) 报道了微软推出的 **Copilot+ PC，其具备 AI 优先的规格，搭载生成式模型和搜索功能**，首批机器采用了 Qualcomm Snapdragon 芯片。

---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**LLM 发展与应用**

- **开源 RAG 应用**：在 /r/LocalLLaMA 中，用户开源了 [LARS，一个以 RAG 为中心的 LLM 应用](https://www.reddit.com/r/LocalLLaMA/comments/1d8ur1y/open_sourcing_my_citationcentric_localllm/)，它可以根据文档生成带有详细引用的回答。它支持多种文件格式，具有对话记忆，并允许自定义 LLM 和 Embedding。

- **最受喜爱的开源 LLM**：在 /r/LocalLLaMA 中，用户分享了他们在**不同用例下的首选开源 LLM**（[链接](https://www.reddit.com/r/LocalLLaMA/comments/1d8vapm/what_open_source_llms_are_your_daily_driver/)），包括用于中小规模文档集 RAG 的 Command-R，用于视觉任务的 LLAVA 34b v1.6，用于大规模语料库复杂问题的 Llama3-gradient-70b 等。

- **AI 商业模式**：在 /r/LocalLLaMA 中，一位用户质疑新的 AI 业务实际上做了多少**原创工作，还是仅仅利用现有的 LLM API**（[链接](https://www.reddit.com/r/LocalLLaMA/comments/1d8wzqd/ai_business_coming_out_of_thin_air_and_llms/)），因为许多业务似乎只是针对特定领域封装了 OpenAI 的 API。

- **使用 LLM 查询大量小文件**：在 /r/LocalLLaMA 中，一位用户寻求关于**将 200 多个 Wiki 小文件输入 LLM** 并保持它们之间关系的建议（[链接](https://www.reddit.com/r/LocalLLaMA/comments/1d8xrlz/best_way_to_feed_lots_of_small_files_into_an_llm/)），因为 Embedding 和 RAG 的效果并不理想。目前正在考虑进行适当的 LLM 训练或 LoRA。

- **本地 LLM API 的桌面应用**：在 /r/LocalLLaMA 中，一位用户创建了一个**桌面应用来与其本地机器上的 LMStudio API 服务器进行交互**（[链接](https://www.reddit.com/r/LocalLLaMA/comments/1d8xlkf/i_made_a_desktop_app_for_interacting_with_my/)），并正在评估将其免费发布的意向。

- **LLM 教育平台**：在 /r/LocalLLaMA 中，[Open Engineer 正式发布](https://www.reddit.com/r/LocalLLaMA/comments/1d90mn6/open_engineer_a_free_educational_platform_for/)，这是一个免费的教育资源，涵盖了 LLM fine-tuning、quantization、embeddings 等主题，旨在让软件工程师更容易上手 LLM。

- **用于数据库操作的 LLM 助手**：在 /r/LocalLLaMA 中，一位用户分享了他们[将 LLM 助手集成到软件中](https://www.reddit.com/r/LocalLLaMA/comments/1d8xuxe/llm_company_assistant/)用于数据库 CRUD 和订单验证的经验，并考虑利用周边系统补充 LLM 的产品查询功能，以实现更稳健的订单处理。

**AI 发展与担忧**

- **关于超级智能开发的推测**：在 /r/singularity 中，一位用户推测[主要的 AI 实验室、芯片公司和政府机构可能已经在幕后协同开发超级智能](https://www.reddit.com/r/singularity/comments/1d8s0ke/there_is_probably_already_something_analagous_to/)，类似于曼哈顿计划，且美国政府可能正在预测并为之产生的影响做准备。

- **围绕 UBI 实施的问题**：在 /r/singularity 中，一位用户对[缺乏实施 UBI 的具体计划或框架](https://www.reddit.com/r/singularity/comments/1d96i0f/so_when_are_the_questions_surrounding_ubi/)表示沮丧，尽管 UBI 被视为解决 AI 驱动的就业流失的方案，但仍有资金来源、人口增长影响等诸多问题需要解决。

- **开源 AGI 滥用的风险**：在 /r/singularity 中，一位用户询问开源社区将如何[防止强大的开源 AGI 被恶意行为者滥用](https://www.reddit.com/r/singularity/comments/1d8t5md/what_is_the_open_source_community_solution_to/)而造成伤害，因为目前缺乏安全防护措施，并认为“这和在 Google 上搜索一样”的反驳观点过于简化了问题。

- **控制 ASI**：在 /r/singularity 中，一位用户质疑[人类或军队能够控制 ASI 的信念](https://www.reddit.com/r/singularity/comments/1d97xc3/can_asi_even_be_controlled/)，考虑到其卓越的智能，评论者一致认为这不太可能，且试图统治它的想法是误导性的。

**AI 助手与界面**

- **聊天模式作为语音数据采集手段**：在 /r/singularity 中，一位用户推测 [OpenAI 对聊天模式的关注是一项战略举措](https://www.reddit.com/r/singularity/comments/1d923rm/is_chat_mode_openais_strategy_for_harvesting/)，旨在收集高质量、自然的语音数据，以克服 AI 训练中文本数据的局限性，因为语音代表了更接近人类思维过程的连续意识流。

- **不寻常的 ChatGPT 使用案例**：在 /r/OpenAI 中，用户分享了[他们使用 ChatGPT 的一些不寻常方式](https://www.reddit.com/r/OpenAI/comments/1d9buhr/what_are_some_unusual_use_cases_no_ones_heard_of/)，包括生成极具戏剧性的浇花提醒、将烹饪指令从烤箱模式转换为空气炸锅模式，以及将购物清单项目映射到商店货架。

- **对 "AI shell" 协议的需求**：在 /r/OpenAI 中，一位用户预见到[需要一种标准化的 "AI shell" 协议](https://www.reddit.com/r/OpenAI/comments/1d90kkn/ai_shell/)，以允许 AI Agent 轻松地界面化并控制各种设备，类似于 SSH 或 RDP，因为现有协议可能不足以满足需求。

**AI 内容生成**

- **对 AI 音乐不断演变的看法**：在 /r/singularity 中，一位用户分享了他们[对 AI 音乐不断变化的观点](https://www.reddit.com/r/singularity/comments/1d8p3wx/ai_music_i_feel_conflicted/)，认为由于对主流音乐的严格限制，AI 音乐非常适合作为通用的 YouTube 背景音乐，并征求了他人的看法。

- **AI 生成的后摇滚播放列表**：在 /r/singularity 中，一位用户在 YouTube 上分享了一个 [30 分钟的 AI 生成后摇滚播放列表](https://www.reddit.com/r/singularity/comments/1d8sfgv/junes_chaos_an_aigenerated_postrock_journey/)，其中包含使用 UDIO 制作的沉浸式曲目，并收到了关于“让人忘记这是 AI 音乐”的正向反馈。

- **VFX 项目中的 AI**：在 /r/singularity 中，一位用户分享了一个[结合 AI 创造城市美学的 VFX 项目](https://www.reddit.com/r/singularity/comments/1d8sfgv/junes_chaos_an_aigenerated_postrock_journey/)，灵感来自广告展示，使用了程序化系统、图像合成和分层 AnimateDiffs，其中 CG 元素经过单独处理并集成。

---

# AI Discord 摘要回顾

> 摘要之摘要的总结

1. **LLM 与模型性能创新**：

   - **Qwen2 引起广泛关注**：模型参数范围从 0.5B 到 7B，因其易用性和快速迭代能力而受到赞赏，支持具有 [128K token 上下文](https://qwenlm.github.io/blog/qwen2/)的创新应用。

- **Stable Audio Open 1.0 引起关注**：利用 autoencoders 和 diffusion 模型等组件，详见 [Hugging Face](https://huggingface.co/stabilityai/stable-audio-open-1.0)，提升了社区在自定义音频生成工作流中的参与度。

- **ESPNet 发布高效 Transformer 推理的竞争性基准测试**：围绕新发布的 ESPNet 的讨论展示了其在 Transformer 效率方面的潜力，指向在高端 GPU (H100) 上增强的吞吐量，详见 [ESPNet Paper](https://arxiv.org/abs/2406.03488)。

- **Seq1F1B 推动高效长序列训练**：根据 [arxiv 出版物](https://arxiv.org/abs/2406.03488)，该流水线调度方法为 LLM 带来了显著的内存节省和性能提升。

2. **微调和 Prompt Engineering 挑战**：

- **模型微调创新**：微调讨论强调了使用 gradient accumulation 来管理内存限制，以及使用 `FastLanguageModel.for_inference` 处理 Alpaca 风格 Prompt 的自定义流水线，如 [Google Colab notebook](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing) 所示。

- **聊天机器人查询生成问题**：使用 Mistral 7B 调试 Cypher 查询强调了系统评估和迭代微调方法在成功模型训练中的重要性。

- **Adapter 集成陷阱**：集成已训练 Adapter 的关键挑战指出需要更高效的 Adapter 加载技术以保持性能，并得到了实际编程经验的支持。

3. **开源 AI 发展与协作**：

- **Prometheus-2 评估 RAG 应用**：Prometheus-2 为评估 RAG 应用提供了 GPT-4 的开源替代方案，因其成本低廉和透明度高而受到重视，详见 [LlamaIndex](https://t.co/BFnmE57OfB)。

- **OpenDevin 的发布**引发了协作兴趣：该系统由 Cognition 开发，是一个用于自主工程的强大 AI 系统，可通过 [网络研讨会](https://lu.ma/fp0xr460) 和 GitHub 获取文档。

- **梯度累积策略改进训练**：关于 Unsloth AI 的讨论强调了使用 gradient accumulation 来有效处理内存限制，通过分享的 [YouTube 教程](https://www.youtube.com/watch?v=cwuYWFC7_QE) 展示了训练时间的缩短。

- **Mojo 作为后端框架兴起**：开发者分享了使用 Mojo 进行 HTTP 服务器开发的积极经验，描述了其在静态类型和编译时计算特性方面的优势，详见 [GitHub](https://github.com/saviorand/lightbug_http/tree/main)。

4. **部署、推理和 API 集成**：

- **Perplexity Pro 增强搜索能力**：最近的更新通过意图系统增加了逐步搜索过程，实现了更具 Agent 特性的执行，正如 [Perplexity Labs](https://labs.perplexity.ai/) 社区内讨论的那样。

- **关于 Modal 部署和隐私的讨论**：关于使用 Modal 进行 LLM 部署的咨询包括对其微调栈和隐私政策的担忧，[Modal Labs 文档](https://www.google.com/search?q=privacy+policy+modal+labs) 提供了额外支持。

- **OpenRouter 技术见解和限制**：用户探索了技术规范和功能，包括助手消息 prefill 支持以及通过 [Instructor 工具](https://useinstructor.com/) 处理 function calls。

5. **AI 社区讨论与活动**：

- **Stable Diffusion 3 预测**：社区对预期的发布充满热议，对其功能和时间表进行了推测，详见各种 [Reddit 帖子](https://www.reddit.com/r/StableDiffusion/comments/1d6ya9w/collection_of_questions_and_answers_about_sd3_and/)。

- **6 月 11 日的 Human Feedback Foundation 活动**：即将举行的关于将人类反馈集成到 AI 中的讨论，演讲者来自 Stanford 和 OpenAI，录像可在其 [YouTube 频道](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg) 观看。

- **Qwen2 模型发布并获得巨大支持**：因其多语言能力和增强的基准测试而备受关注，在 [Hugging Face](https://huggingface.co/Qwen) 等平台上的发布突显了其实际评估效果。

- **呼吁 Mozilla AI 支持 JSON Schema**：在 [社区频道](https://discord.com/channels/1089876418936180786/1182689832057716778/1248115957067415663) 中，显著记录了要求在下一版本中包含 JSON Schema 以简化应用开发的请求。

- **关于机器人 AI 和 Foundation Models 的主题演讲**：在 Foundation Model 公司中对“机器人版 ChatGPT”的投资兴趣，强调了 [Newcomer 的文章](https://www.newcomer.co/p/why-investors-cant-get-enough-of) 中详述的战略对齐。

---

# PART 1: High level Discord summaries

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **网页抓取工具讨论**：工程师们交流了关于 **Beautiful Soup** 和 **Scrapy** 等 **OSS** 抓取平台的经验，并特别提到了 **Trafilatura**，因其在从职位发布和 SEC 文件等复杂来源中提取动态内容方面表现出色，参考了其 [官方文档](https://trafilatura.readthedocs.io/en/latest/)。

- **UI 领域的新面孔**：Google 推出的 **Mesop** 因其在构建内部工具 UI 方面的潜力而引发讨论，开始进入 **Gradio** 和 **Streamlit** 的领地。尽管目前缺乏高级身份验证功能，但其 [Mesop 主页](https://google.github.io/mesop/) 引起了广泛好奇。

- **查询构建的挑战**：工程师们调试了使用 **Mistral 7B** 生成 Cypher 查询的问题，强调了系统化评估（evals）、测试驱动开发（TDD）以及迭代过程的重要性——这是模型微调（fine-tuning）中细节工作的体现。

- **深入探讨部署**：关于 **Modal** 使用的问题层出不穷，包括其微调栈的复杂性和隐私政策——为政策寻求者提供了 [Modal Labs 查询](https://www.google.com/search?q=privacy+policy+modal+labs) 链接，并推荐了他们的 [Dreambooth 示例](https://modal.com/docs/examples/dreambooth_app) 以进行实践启发。

- **CUDA 兼容性疑难杂症**：CUDA 版本之间的兼容性问题成为焦点，工程师们在安装 `xformers` 模块时遇到了困难——[Jarvis Labs 文档](https://jarvislabs.ai/docs/troubleshooting#updating-cuda) 中关于更新 CUDA 的指南提供了帮助。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-3 的编程难题**：虽然像 **GPT-3** 这样的 GPT 模型擅长辅助编程，但在面对高度特定且复杂的问题时，局限性变得显而易见，这标志着对其当前能力的挑战。
- **数学方程中的逻辑漏洞**：即使是简单的逻辑任务（如纠正数学方程）也可能难倒 GPT 模型，暴露了基础逻辑推理方面的短板。
- **热切期待 GPT-4o 的特殊功能**：讨论集中在 **GPT-4o** 的更新上，语音和实时视觉功能将在几周内首先向 ChatGPT Plus 用户开放，随后会扩大访问范围，正如 [OpenAI 官方更新](https://x.com/OpenAI/status/1790130708612088054?t=bgpr-IBhwXMa4ZiLXCrJgg&s=19) 所述。
- **DALL-E 的文本难题**：用户正在分享针对 **DALL-E** 在生成包含精确文本的 Logo 时遇到困难的变通方法，包括通过提示词迭代优化文本准确性，以及一个有用的 [自定义 GPT 提示词](https://chatgpt.com/g/g-TKZI5nYMc-one-word-graphics)。
- **7b 与视觉模型的协同效应**：一个集成成功的案例显示 **7b 模型** 与 **llava-v1.6-mistral-7b** 视觉模型配合良好，扩展了模型协作的可能性。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**梯度累积（Gradient Accumulation）显神威**：工程师们一致认为 **gradient accumulation** 可以缓解显存限制并缩短训练时间，但也警告说，由于意外的内存分配行为，较大的 Batch Size 可能会带来潜在陷阱。

**利用 Alpaca 提升推理速度**：一位工程师分享了利用 `FastLanguageModel.for_inference` 并结合 **Alpaca 风格提示词** 在 **LLMs** 中进行序列生成的代码片段，这与关于共享 [Google Colab 笔记本](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing) 的讨论共同引发了关注。

**Adapter 合并求助**：集成训练好的 Adapter 导致性能大幅下降的挑战，引发了对更高效 Adapter 加载技术的需求，以维持训练效率。

**Qwen2 模型备受瞩目**：**Qwen2 模型** 的发布引发了热议，工程师们对其 **0.5B 到 7B** 的小尺寸模型表现出浓厚兴趣，因其易用性且支持更快的迭代。

**求助中心的解决方案探索**：帮助频道的对话强调了对节省 VRAM 的 lora-adapter 文件转换流程的需求、对可能减慢推理速度的 Bug 的快速情报、缓解 GPU 显存过载的策略，以及关于运行 **gguf 模型** 和实现 RAG 系统的澄清，参考了 [Mistral 文档](https://docs.mistral.ai/guides/rag/)。

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Audio Open 1.0 正式登场**：[Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) 因其创新的组件（包括 autoencoder 和 diffusion model）引发了社区关注，成员们正在自定义音频生成工作流中尝试使用 tacotron2、musicgen 和 audiogen。

- **Stable Diffusion 中的尺寸至关重要**：用户建议在 **Stable Diffusion** 中先生成高分辨率图像再进行缩小，这是解决困扰低分辨率（160x90）图像输出的随机颜色问题的有效权宜之计。

- **ControlNet 草图转图像大获成功**：ControlNet 成为成员们将草图转换为写实图像且不保留多余颜色的首选方案，为最终构图和图像细节提供了更好的控制。

- **CivitAI 内容质量控制受到质疑**：**CivitAI** 上无关内容的激增导致用户呼吁增强过滤功能，以更好地筛选优质模型并维护用户体验。

- **Stable Diffusion 3 尚待明确**：尽管社区内猜测不断，但关于 **Stable Diffusion 3** 的发布日期和细节仍不明朗，部分成员参考了 [Reddit 帖子](https://www.reddit.com/r/StableDiffusion/comments/1d6ya9w/collection_of_questions_and_answers_about_sd3_and/) 以寻求暂定的答案。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **在 LM Studio 中驯服 RAM 占用**：用户讨论了限制 LM Studio 中 RAM 使用的策略，建议在运行期间加载和卸载模型；详细方法可在 [llamacpp documentation](https://example.com) 中找到。虽然这不是 LM Studio 的内置功能，但此类策略被用于使模型仅在活动时占用 RAM，尽管会损失一定的效率。

- **Nomic Embed 模型步入聚光灯下**：得益于 [nomic-embed-vision-v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)，讨论将 **Nomic Embed Text 模型** 提升到了多模态地位，并以出色的 Benchmark 表现脱颖而出。服务器内的 AI 社区还指出 **Llama-3 的 Q6 量化模型** 是质量与性能之间的平衡点。

- **寻求完美配置**：一位用户提出了在启动新聊天时模型设置会重置的问题，并发现保留旧聊天记录可能是一个简单的解决方法。对话还涉及了 LM Studio 中的 `use_mlock` 和 `--no-mmap` 设置，这些设置会影响 8B 模型运行时的稳定性，并强调了受操作系统影响的细微差别。

- **解锁硬件协同对 AI 的潜力**：工程师们就 Nvidia 的专有驱动程序与 AMD 的开源方案展开了热烈辩论，强调了其对系统管理和安全性的影响。此外，人们对新款 **Qualcomm 芯片** 的前景感到兴奋，并提醒不要仅凭合成 Benchmark 来评判 ARM CPUs。

- **更新与升级引发关注**：**Higgs LLAMA 模型** 因其在 70B 规模下的智能表现而获得赞誉，人们对即将发布的包含相关 **llamacpp 调整** 的 LMStudio 更新充满期待。另一位用户正考虑进行大规模 RAM 升级，以应对备受讨论的 **LLAMA 3 405B** 模型，这反映了硬件能力与 AI 模型演进之间交织的利益。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**审核是关键**：社区讨论了针对不当行为报告的审核策略。处理此类问题的专业性至关重要。

**Gradio API 挑战**：将 Gradio 与 React Native 和 Node.js 集成在社区中引发了疑问。由于它是用 Svelte 构建的，用户被引导去研究 Gradio 的 API 兼容性。

**带有稳定性的文本**：围绕用于文本生成的 Stable Diffusion 模型的讨论，向成员推荐了来自 Microsoft 的 AnyText 和 TextDiffuser-2 等解决方案，以获得稳健的输出。

**当计算走向点对点**：对话转向了用于分布式机器学习的 P2P (peer-to-peer) 计算，Petals 等工具以及具有隐私意识的本地集群 (local swarms) 经验提供了充满前景的途径。

**AI 中的人类反馈**：Human Feedback Foundation 在将人类反馈引入 AI 方面取得了进展，将于 6 月 11 日举行活动，并在其 YouTube 频道上提供了大量教育课程。

**小数据集，大挑战**：在计算机视觉讨论中，处理小数据集和不具代表性的验证集是一个紧迫的问题。解决方案包括使用多样化的训练数据，甚至可能使用 Transformer，尽管其训练时间较长。

**Swin Transformer 测试**：有关于将 Swin Transformer 应用于 CIFar 数据集的查询，突显了社区对在各种场景中实验当代模型的兴趣。

**确定性模型降低“热度”**：一条消息强调将 temperature 设置降低到 0.1 以实现更具确定性的模型行为，引发了对模型微调方法的思考。

**样本输入混乱**：出现了关于文本嵌入 (text embeddings) 以及 text-enc 1 和 text-enc 2 等模型样本输入正确结构的困惑，同时还讨论了以字典格式添加 kwargs 所带来的挑战。

**重新参数化的成果**：一位成员成功地将 Segmind 的 **ssd-1b** 重新参数化为 **v-prediction/zsnr refiner** 模型，并称其为新的最爱，暗示了 1B Mixture of Experts 模型的可能趋势。

**项目援助**：在社区援助环节，成员们通过私信 (DM) 就数据集问题提供个人帮助，增进了公会的协作氛围。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**对 KAN 的怀疑**：公会成员认为 Kolmogorov-Arnold Networks (KANs) 的效率低于传统的神经网络，并对其可扩展性和可解释性表示担忧。然而，人们对更高效的 KAN 实现（例如使用 ReLU 的实现）表现出兴趣，一份共享的 [ReLU-KAN 架构论文](https://arxiv.org/abs/2406.02075) 证明了这一点。

**扩展数据清洗工具箱**：参与者讨论了 **influence functions** 在数据质量评估中的效用，LESS 算法 ([LESS algorithm](https://www.cs.princeton.edu/~smalladi/blog/2024/04/04/dataselection/)) 被提及作为一种更具扩展性的替代方案，用于选择高质量的训练数据。

**高效模型训练的突破**：高效模型训练的创新被广泛分享，包括 Nvidia 在 [GitHub](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro) 上提供的新开放权重，对 MatMul-free 模型 ([arXiv](https://arxiv.org/abs/2406.02528)) 以提高效率的探索，以及 Seq1F1B 在内存高效的长序列训练方面的潜力 ([arXiv](https://arxiv.org/abs/2406.03488))。

**量化技术可能提升 LLM 性能**：新颖的 QJL 方法通过量化过程压缩 KV cache 需求，为大语言模型 (LLM) 提供了一条充满希望的途径 ([arXiv](https://arxiv.org/abs/2406.03482))。

**大脑数据语音解码探索**：一位公会成员报告了使用 **Whisper tiny.en embeddings** 和大脑植入数据解码语音的实验，在面临单 GPU 训练限制的情况下，请求同行建议通过调整层和损失函数来优化模型。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 变得更聪明了**：Perplexity Pro 约一周前进行了升级，可以逐步显示其搜索过程，并采用意图系统（intent system）来实现更具 *agentic-like*（类智能体）的执行。
  
- **文件格式困扰**：用户在 Perplexity 读取 PDF 的能力方面遇到困难，成功率因 PDF 的内容布局而异，从高度样式化到纯文本不等。

- **开发成本令人咋舌**：社区对一名成员请求以 100 美元的极低预算构建 text-to-video MVP 的行为表示幽默和难以置信，突显了预期与开发者市场价之间的脱节。

- **Haiku 功能不再**：从 Perplexity labs 移除 Haiku 及部分功能引发了讨论，导致成员推测这是成本削减措施，并因其对工作流的影响而表达不满。

- **对 OpenChat 扩展的好奇**：有人询问是否会在 Perplexity 中加入 **"openchat/openchat-3.6-8b-20240522"** 模型，目前已有 **Mistral** 和 **Llama 3** 等模型。



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **探索往期技术会议**：往期 **CUDA MODE** 技术活动的录像可以在 [CUDA MODE YouTube 频道](https://www.youtube.com/@CUDAMODE)观看。

- **在 PyTorch 中调试 Tensor 内存**：分享了一个使用 `storage().data_ptr()` 的代码片段，用于测试两个 PyTorch **tensors 是否共享相同内存**，引发了关于检查内存重叠的讨论。一名成员请求协助定位 PyTorch C++ 函数的源代码，特别是 [at::_weight_int4pack_mm](https://pytorch.org/cppdocs/api/function_namespaceat_1adeda9630914278ac02d7fd758da19e3d.html#exhale-function-namespaceat-1adeda9630914278ac02d7fd758da19e3d)。

- **AI 模型与技术扩展**：对话围绕 **MoRA** 优于 **LoRA** 的方法论，以及关于 RLHF 的 **DPO** 与 **PPO** 之争。另外还提到了用于位置编码的 **CoPE** 和加速推理的 **S3D**，详情见 [AI Unplugged](https://datta0.substack.com/p/ai-unplugged-12-mora-dpo-vs-ppo-cope)。

- **Torch 创新火花**：围绕 **torch.compile** 提升 KAN 性能以媲美 MLP 的讨论被点燃，分享了来自 [Thomas Ahle 的推文](https://x.com/thomasahle/status/1798408687981297844)的见解、实践经验以及 [KANs 和 MLPs 的 GitHub 仓库](https://github.com/thomasahle/kanmlps)。

- **MLIR 瞄准 ARM**：一次 MLIR 会议涵盖了创建 **ARM SME Dialect**，通过 [YouTube 视频](https://www.youtube.com/watch?v=jrniGW_Hzno)提供了对 ARM 可扩展矩阵扩展（Scalable Matrix Extension）的见解。讨论了指向潜在 **Triton ARM** 支持的线索，并参考了 [MLIR 文档](https://mlir.llvm.org/docs/Dialects/ArmNeon/#arm_neonintrummla-arm_neonummlaop)中用于 NEON 操作的 'arm_neon' dialect。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **寻找机器人领域的 ChatGPT**：投资者正在寻找能够成为“机器人领域 ChatGPT”的初创公司，优先考虑 AI 而非硬件。正如[这篇文章](https://www.newcomer.co/p/why-investors-cant-get-enough-of)所述，人们对利基基础模型公司的兴趣日益浓厚。
  
- **Qwen2 引起关注**：[Qwen2 发布](http://qwenlm.github.io/blog/qwen2/)引起关注，其模型在多语言任务中支持高达 128K tokens，但用户反映在近期事件知识和通用准确性方面存在差距。

- **Dragonfly 在多模态 AI 中展翅**：Together.ai 的 [Dragonfly](https://www.together.ai/blog/dragonfly-v1) 带来了视觉推理方面的进展，特别是在医学影像领域，展示了模型开发中文本和视觉输入的整合。

- **AI 社区持批判观点**：从讨论有影响力的实验室员工批评小玩家，到分享的一条[推文](https://x.com/leopoldasch/status/1798483665904865474)强调了 AI 实验室无意中与 CCP 而非美国研究界分享进展的风险，以及 [The Verge 关于 Humane AI 安全问题的文章](https://www.theverge.com/2024/6/5/24172377/humane-ai-pin-battery-case-issue-warning)，社区保持警惕。

- **强化学习论文引起兴趣**：分享了 [这条推文](https://x.com/aahmadian_/status/1798740211909922862) 中宣布的“自我改进鲁棒偏好优化”（SRPO）论文，表明了对使用鲁棒且自我改进的 RLHF 方法训练 LLM 的关注。Nathan Lambert 计划专门花时间讨论此类前沿论文。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Rising**：各频道的讨论集中在将 **Mojo** 用于后端服务器开发的优势，例如 [lightbug_http](https://github.com/saviorand/lightbug_http/tree/main) 展示了其在构建 HTTP 服务器中的应用。分享了 [Mojo 路线图](https://docs.modular.com/mojo/roadmap)，指出了未来的核心编程特性；同时，由于 Mojo 的静态类型和编译时计算带来的性能优势，与 Python 的活跃对比引发了关于性能价值的辩论。
  
- **Keeping Python Safe**：在向初学者教授 Python 时，必须避免潜在的“陷阱”（footguns），以帮助学习者顺利过渡到更复杂的语言，如 C++。

- **Model Performance Frontiers Explored**：讨论指出，将 **Mistral** 等模型扩展到特定限制之外需要持续的预训练，并建议将 **UltraChat** 与基础 **Mistral** 之间的差异应用于 **Mistral-Yarn** 作为一种合并策略，尽管有人对其可行性和实用性表示怀疑。

- **Community Coding Contributions Cloaked with Humor**：成员们幽默地使用“舔饼干”（licking the cookie）等表达来讨论鼓励开源贡献，并戏称技术演讲和编码挑战的复杂性，将一个简单的快速排序（quicksort）实现请求比作一场崇高的远征。

- **Nightly Builds Yield Illumination and Frustration**：考察了 Nightly 构建版本，重点关注了列表迭代器的不可变自动解引用（immutable auto-deref）用法，以及最新编译器版本 `2024.6.616` 中引入的 `String.format` 等特性。然而，网络波动以及 `parallel_sort` 函数中 `algorithm.parallelize` 的不可预测性成为了挫折的来源，这从 [GitHub 讨论](https://github.com/rd4com/mojo_branch/tree/list_iter_autoderef_immut)以及关于工作流计时和网络问题的故障排除分享中可见一斑。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cohere Cashes In Big Time**：根据 [路透社报道](https://www.reuters.com/technology/nvidia-salesforce-double-down-ai-startup-cohere-450-million-round-source-says-2024-06-04/)，Cohere 已获得令人咋舌的 **4.5 亿美元** 融资，NVidia 和 Salesforce 等科技巨头贡献巨大，尽管该公司去年的营收并不高。
- **IBM's Granite Gains Ground**：IBM 的 Granite 模型因其透明度（特别是在训练数据使用方面）而受到赞誉，引发了关于它们在企业领域是否超越 OpenAI 的辩论，参考了 [Talha Khan](https://x.com/TalhaKhan_TK_/status/1798562313160761612) 的见解。
- **Databricks Dominates Forrester’s AI Rankings**：在 Forrester 关于 AI 基础模型的最新报告中，Databricks 被评为领导者，强调了企业的定制化需求，并暗示基准测试分数并不代表一切。该报告在 Databricks 的 [公告博客](https://www.databricks.com/blog/databricks-named-leader-forrester-wavetm-ai-foundation-models-language-q2-2024) 中被重点提及，并可在此处 [免费获取](https://reprints2.forrester.com/#/assets/2/848/RES180932/report)。
- **Qwen 2 Trumps Llama 3**：新的 Qwen 2 模型具有令人印象深刻的 128K 上下文窗口能力，在代码和数学任务中表现出优于 Llama 3 的性能，详见 [最近的推文](https://x.com/reach_vb/status/1798748655366914325)。
- **New Avenues for AI Web Interaction and Assistance**：Browserbase 庆祝获得 **650 万美元** 的种子基金，旨在使 AI 应用能够导航网页，由创始人 Nat & Dan 分享；而 Nox 推出了一款 AI 助手，旨在让用户体验感到无懈可击，早期注册请点击 [此处](http://heynox.com)。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**Prometheus-2 助力 RAG 应用评估**：[Prometheus-2](https://t.co/BFnmE57OfB) 被定位为评估 RAG 应用时 GPT-4 的开源替代方案，因其在透明度和成本效益方面的优势而引发关注。

**LlamaParse 引领知识图谱构建**：一份发布的 notebook 展示了 LlamaParse 如何执行顶级解析以构建知识图谱（Knowledge Graph），并配合 RAG 流水线进行节点检索。

**LlamaIndex 配置过载问题**：AI 工程师们表示，在配置 LlamaIndex 以查询 JSON 数据时感到非常复杂并寻求指导，同时还在讨论 Text2SQL 查询在平衡结构化与非结构化数据检索方面存在的问题。

**探索资源受限场景下的 LLM 选项**：针对硬件受限情况的替代方案讨论倾向于使用 Microsoft Phi-3 等小型模型，并尝试在 Google Colab 等平台上运行大型模型。

**评分过滤器获得可定制优势**：工程师们正在讨论 LlamaIndex 根据自定义阈值和性能评分过滤结果的能力，这表明了对搜索结果进行精细化准确度控制的需求。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **早期采用者的初创公司福利**：Cohere 推出了一个[初创公司计划](https://cohere.com/startup-program-application)，旨在为 B 轮或更早阶段的初创公司提供 AI 模型折扣、专家支持和市场影响力，以推动 AI 技术应用的创新。

- **优化聊天机器人体验**：Cohere Chat API 将于 6 月 10 日迎来更新，届时将引入新的默认多步工具（multi-step tool）行为，并为简化操作提供 `force_single_step` 选项，所有内容均已记录在 [API 规范增强文档](https://docs.cohere.com/page/changes-in-chat-api-and-tool-use)中。

- **AI 采样器的高 Temperature 设置**：OpenRouter 因允许 AI 响应采样器的 temperature 设置超过 1 而脱颖而出，这与 Cohere 试用版 1.0 的上限形成对比，引发了关于响应多样性和质量灵活性的讨论。

- **开发智能群聊机器人**：关于在商务会议等群组场景中部署 AI 聊天机器人的建议不断涌现，讨论分析了 Rhea 在处理多用户上下文方面的优势，以及在众多用户中提供个性化响应时可能存在的精度问题。

- **Cohere 社交与演示**：社区成员欢迎新成员 Toby Morning，通过交换 LinkedIn 个人资料（[LinkedIn Profile](http://www.linkedin.com/in/urbantech/)）建立更广泛的联系，并对即将展示的 Coral AGI 系统在多用户环境下的强大功能表现出极大热情。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Qwen2 飞跃式进展**：**Qwen2** 系列模型的发布标志着相较于 Qwen1.5 的重大进化，现在支持 **128K token** 的上下文长度、**27 种新增语言**，并提供各种尺寸的预训练及指令微调模型。这些模型已在 [GitHub](https://github.com/QwenLM/Qwen2)、[Hugging Face](https://huggingface.co/Qwen) 和 [ModelScope](https://modelscope.cn/organization/qwen) 等平台上线，并设有[专门的 Discord 服务器](https://discord.gg/yPEP2vHTu4)。

**地图事件预测讨论**：一位用户询问了如何利用时间数据预测地图上事件点的真伪，引发了关于相关命令和技术的讨论，尽管未提供具体方法。

**Mistral API 与模型存储更新**：Mistral 推出的微调 API 及其相关成本引发了讨论，重点在于其对开发和实验的实际影响。该 API（包括定价详情）在其[微调文档](https://docs.mistral.ai/guides/finetuning/)中有所说明。

**移动端文本输入焕然一新**：WorldSim Console 更新了其移动平台，修复了与文本输入相关的错误，提高了文本输入的可靠性，并提供了增强的复制/粘贴以及外观自定义选项等新功能。

**闲聊板块中的音乐探索**：一位成员分享了探索“瓦坎达音乐”的链接，尽管这对于工程师受众来说技术相关性有限。分享的链接中包括 [DG812 - In Your Eyes](https://youtu.be/vP4zGMdTDPM) 和 [MitiS & Ray Volpe - Don't Look Down](https://youtu.be/e-Fors8CnKA) 等音乐视频。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**使用 Pilot 轻松进行服务器管理**：**Pilot** 机器人通过提供 "Ask Pilot"（智能服务器洞察）、"Catch Me Up"（消息摘要）以及每周关于服务器活跃度的 "Health Check"（健康检查）报告，正在彻底改变 Discord 服务器的管理方式。它是免费使用的，能促进社区增长和参与度，可通过其[网站](https://usepilot.app/)访问。

**角色扮演领域的 AI 竞争者**：WizardLM 8x22b 模型目前在角色扮演社区中大受欢迎，然而 Dolphin 8x22 作为一个潜在对手出现，正等待用户测试以比较它们的效能。

**Gemini Flash 引发图像输出好奇心**：关于 **Gemini Flash** 是否能渲染图像的询问引发了澄清：虽然目前没有 Large Language Model (LLM) 直接提供图像输出，但理论上它们可以使用 base64 或调用外部服务（如 Stable Diffusion）进行图像生成。

**处理函数调用的工具提示**：对于处理特定的函数调用（Function Calls）和格式化，推荐使用 [Instructor](https://useinstructor.com/) 这一强大工具，它能促进自动化命令执行并改进用户工作流。

**模型热潮中的技术讨论**：一位成员关于 OpenRouter 中 prefill 支持的询问得到了确认，即这是可能的，特别是通过使用反向代理；同时，由于 **GLM-4** 支持韩语，人们对其表现出极大的热情，暗示了该模型在多语言应用中的潜力。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **人类反馈推动 AI 进步**：即将于 6 月 11 日举行的 *Human Feedback Foundation event* 将探讨人类反馈在增强医疗和公民领域 AI 应用中的作用；感兴趣的各方可以通过 [Eventbrite](https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator) 注册。此外，包含来自 UofT、Stanford 和 OpenAI 演讲者的往期活动录像可在 [Human Feedback Foundation YouTube Channel](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg) 观看。

- **Azure 活动吸引 AI 爱好者**："Unleash the Power of RAG in Azure" 是一个在多伦多举行的、报名人数众多的 Microsoft 活动，一位正在寻找同伴的参与者提到了这一点；更多详情可以在 [Microsoft Reactor 页面](https://developer.microsoft.com/en-us/reactor/events/22756/)找到。

- **应对杂乱数据**：工程师们讨论了处理高基数分类列的策略，包括使用聚合/分组、手动特征工程、字符串匹配和编辑距离技术，目标都是精炼输入以获得更好的回归结果。

- **合并数据与聚类技术**：存在一种共同观点，即结合拼写纠正与特征聚类可能会简化高基数分类数据带来的挑战，重点是将此类问题视为核心数据建模问题。

- **特征工程的实用方法**：对话转向了务实的方法，例如分解复杂问题（例如，隔离品牌和项目元素）以及将移动平均线作为价格预测简化技术的一部分。与会者对讨论的多方面解决方案表示赞赏，包括用于特征提取的 regex。




---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**AI 爱好者的资料盛宴**：工程师们赞叹 **15T 数据集**的可获得性，并幽默地指出数据过剩但计算资源和资金匮乏的困境。

**硬件讨论中的 GPU 调侃**：**4090s** 是否适合预训练海量数据集引发了诙谐的交流，大家开玩笑地谈论了消费级 GPU 在处理此类高难度任务时的局限性。

**GLM 与 Qwen2 的微调乐趣**：社区分享了微调 **GLM 4 9b** 和 **Qwen2 模型**的技巧和配置，并指出 Qwen2 与 Mistral 的相似性简化了这一过程。

**寻求可靠的 Checkpointing**：在关于 checkpoint 策略的讨论中，提到了使用 Hugging Face 的 `TrainingArguments` 和 `EarlyStoppingCallback`，特别是为了根据 `eval_loss` 捕获最近状态和性能最佳状态。

**排查 AI 代码中的错误**：针对 "returned non-zero exit status 1" 错误的排查，成员们建议精确定位失败的命令，仔细检查 `stdout` 和 `stderr`，并检查权限或环境变量问题。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **吸引眼球的命名难题**：在公会中，人们对 **1B 参数 zsnr/vpred refiner** 的清晰度提出了质疑；有评论指出该模型实际上是一个 **1.3B** 模型，而非 1B 模型，这引发了关于需要更准确、更具吸引力的命名的轻松调侃。

- **Vega 的参数困境**：关于 **Vega 模型** 的讨论强调了其迅捷的处理能力，但也引发了对其参数规模不足可能限制连贯输出生成的担忧。

- **Elrich Logos 数据集仍是谜团**：一名成员询问了 **Elrich logos 数据集** 的可用性，但未收到关于获取该数据集的任何结论性信息或回复。

- **Qwen2 的黎明**：**Qwen2** 的发布已经宣布，在语言支持、上下文长度和基准测试性能等多个方面较 Qwen1.5 引入了实质性的改进。Qwen2 目前提供不同尺寸的版本，并支持高达 **128K tokens**，资源分布在 [GitHub](https://github.com/QwenLM/Qwen2)、[Hugging Face](https://huggingface.co/Qwen)、[ModelScope](https://modelscope.cn/organization/qwen) 和 [demo](https://huggingface.co/spaces/Qwen/Qwen2-72B-Instruct)。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **知识图谱构建的安全措施**：分享了一个关于[从文本构建知识图谱](https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/)的教程，强调了在将数据纳入图数据库之前进行数据验证以确保安全的重要性。

- **LangChain 技术要点**：关于是否必须使用 tools 装饰器的困惑引发了澄清讨论，同时观察到用户有理解 token 消耗追踪方法的需求。此外，还出现了一个关于如何创建 LangChain 的 FreeCodeCamp [教程视频](https://youtu.be/sVcwVQRHIc8?si=BLfH2g7WUKtIi6A0)中展示的 RAG 图表的问题。

- **流程控制与搜索自动化资源**：一段关于 LangGraph 条件边（conditional edges）的 [YouTube 视频](https://youtu.be/EKxoCVbXZwY)因其在流程工程（flow engineering）中对流程控制的实用性而受到关注；另外分享了一个名为 [search-result-scraper-markdown](https://github.com/essamamdani/search-result-scraper-markdown) 的新项目，用于抓取搜索结果并将其转换为 Markdown。

- **跨框架 Agent 协作**：用户对能够让使用不同工具构建的 Agent 进行协作的框架表现出兴趣，这些工具包括 LangChain、MetaGPT、AutoGPT，甚至来自 coze.com 等平台的 Agent，突显了 AI 领域互操作性的潜力。

- **对 GUI 和课程文件指导的需求**：有用户询问如何从 AI Agents LangGraph 课程中找到特定的 "helper.py" 文件，这表明在 DLAI [课程页面](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/)等技术课程中需要更好的资源发现方法。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 为正式发布做准备**：**George Hotz** 强调了在 **1.0 版本发布** 之前需要对 **tinygrad** 进行更新，待处理的 PRs 预计将解决目前的差距。
- **使用 UOps.CONST 解开 Tensor 谜题**：AI 工程师研究了 **UOps.CONST** 在 tinygrad 中的作用，它在计算过程中充当**地址偏移（address offsets）**，用于确定 Tensor 加法期间的索引值。
- **解码复杂代码**：针对一段代码引发的困惑，相关人员澄清说，为了在行优先（row-major）数据布局限制内高效管理 Tensor 的形状（shapes）和步长（strides），通常需要复杂的条件判断。
- **动态 Kernel 解决索引难题**：关于 tinygrad 中 Tensor 索引的讨论显示，由于该架构依赖于**静态内存访问**，Kernel 生成至关重要，所讨论的 Kernel 能够实现类似 **Tensor[Tensor]** 的操作。
- **在 Getitem 操作中使用 Arange 进行掩码处理**：讨论中提到了用于索引操作的 Kernel 与 **arange kernel** 之间的相似性，后者有助于在 **getitem** 函数中创建掩码，以实现动态 Tensor 索引。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**图形输出的加速需求**：成员们正在寻求关于使用 `interpreter.computer.run` 执行图形输出的建议，特别是针对像 `matplotlib` 生成的可视化内容，目前尚未取得成功。

**OS Mode 的混乱**：对话强调了在让 `--os mode` 与来自 LM Studio 的本地模型协同工作时遇到的麻烦，包括本地 LLAVA 模型无法启动屏幕录制的问题。

**M1 Mac 上的视觉模型探索**：工程师们对 M1 Mac 上视觉模型的硬件限制表示沮丧，鉴于 OpenAI 服务的成本较高，他们对免费且易于获取的 AI 解决方案表现出浓厚兴趣。

**对 Rabbit R1 集成的期待**：将 Rabbit R1 与 OpenInterpreter 集成的讨论正热，特别是即将推出的 webhook 功能，以实现实际操作。

**征集 Bash 模型**：关于征集适用于处理 bash 命令的开源模型的呼吁尚未得到回应，目前在该推荐领域仍存在空白。



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**对 AI Town 开发进度的关注**：**AI Stack Devs** 的成员们寻求该项目的更新，其中一人对进度表示关注，而另一人则因时间不足尚未贡献代码而表示歉意。

**AI Town 中的图块集（Tileset）解析难题**：在为 **AI Town** 解析精灵图表（spritesheets）时出现了一个工程挑战，提议使用提供的层级编辑器或 Tiled，并辅以社区提供的转换脚本。

**学习如何取消 LLM 的审查**：一位成员分享了来自 Hugging Face 博客关于 **abliteration**（消融技术）的见解，该技术可以取消 LLM 的审查，文中介绍了第三代 Llama 模型的 instruct 版本。随后，他们询问了如何将此技术应用于 OpenAI 模型。

**未获解答的 OpenAI 实现咨询**：尽管分享了关于 abliteration 的研究，但在讨论帖中，关于如何使用 OpenAI 模型实现该技术的知识请求尚未得到解答。

**深入了解**：
- 解析挑战与方法：\(未提供\)
- 关于 abliteration 的博客文章：[使用 abliteration 取消任何 LLM 的审查](https://huggingface.co/blog/mlabonne/abliteration)。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **文本截断之谜**：在使用 `llm` 处理 embeddings 时，一位成员发现输入文本超过模型的上下文长度（context length）并不一定会触发错误，反而可能导致输入被截断。实际行为因模型而异，这凸显了需要更清晰的文档来说明各模型如何处理超长输入。
  
- **嵌入任务：是否支持续传**：有人询问 `embed-multi` 在不重新处理已完成部分的情况下恢复大型嵌入任务的功能。这突出了对能够管理嵌入过程中部分完成任务的功能的需求。

- **对嵌入行为文档的需求**：@SimonW 的回应指出，关于输入是被截断还是产生错误的模型行为文档缺乏清晰度，这表明用户普遍呼吁为这些 AI 系统提供全面且易于获取的文档。

- **模型截断的推测**：在缺乏错误消息的情况下，@SimonW 推测导致意外嵌入结果的长文本输入很可能被截断了，这种行为应该在特定模型文档中得到明确验证。

- **大型嵌入任务的效率**：关于 `embed-multi` 是否能在重新运行大型任务时识别并跳过先前已处理数据的讨论，展示了对效率的关注以及在长时间运行的 AI 流程中进行智能任务处理的需求。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

**Megatron 的 Checkpoint 难题**：工程师们询问了 **Megatron** 与微调库的兼容性，并注意到了其独特的 checkpoint 格式。大家一致认为，将 **Megatron checkpoints 转换为 Hugging Face 格式**并利用 **Torchtune** 进行微调是最佳方案。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **对 JSON Schema 集成的呼声日益高涨**：一位 AI 工程师提议在即将发布的软件版本中包含 **JSON schema**，以简化应用开发。他承认虽然存在一些 Bug，但强调了它为构建应用程序带来的便利。目前尚未提供关于时间表或潜在实现挑战的细节。

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord

- **关于 AI Infrastructure 的周末音频学习**：一位成员分享了周末听力环节的链接，内容涉及 AI Infrastructure 的讨论，这对于希望紧跟该领域最新趋势和挑战的 AI 工程师来说可能很有意义。内容可以在 [YouTube](https://youtu.be/4jPg4Se9h5g?si=ULVqGQa6AvI8Ch3o) 上观看。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

# PART 2: 详细频道摘要与链接

{% if medium == 'web' %}

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1248011841976795137)** (66 messages🔥🔥): 

- **开源爬虫平台推荐**：一位成员询问开源爬虫平台，并收到了包括 **Beautiful Soup** 和 **Scrapy** 在内的建议。另一位成员推荐了 **Trafilatura**，用于从招聘信息和 SEC 备案文件中抓取动态内容，并提供了 [Trafilatura 文档链接](https://trafilatura.readthedocs.io/en/latest/)。
- **Mesop 与 Gradio 及 Streamlit 的对比**：Google 发布了 **Mesop**，这是一个基于 Python 的 UI 构建框架，成员们认为它在构建内部低流量应用方面优于 **Gradio** 和 **Streamlit**。更多详情请见 [Mesop 主页](https://google.github.io/mesop/)，尽管对其高级身份验证功能存有疑问，但仍引起了广泛兴趣。
- **使用 Mistral 7B 微调 Cypher 查询生成**：一位在使用 Mistral 7B 生成正确的 Cypher 查询时遇到困难的成员与 **HamelH** 讨论了系统化的调试步骤。对话强调了编写 evals、测试失败模式以及迭代改进 prompt 的重要性。
- **对微调技术研讨会的兴趣**：多位成员对涵盖 SFT、DPO 和 ORPO 等主题的研讨会表示兴趣，并建议使用 **TRL library**，推荐 **Leandro von Werra** 作为潜在讲师。然而，名额限制和日程安排问题被提及为限制因素。
- **Braintrust 和 OpenPipe 平台**：有人提出了关于 **Braintrust** 和 **OpenPipe** 的问题，回复指出之前的答疑时间（office hours）和演讲已经涵盖了这些平台。成员们分享说，这些活动回答了许多关于平台用途和实用性的常见问题。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/live/QXVCqtAZAn4?si=y5cdnQnlsWOHGHlk">Aligning LLMs with Direct Preference Optimization</a>：在本次研讨会中，来自 Hugging Face 的 Lewis Tunstall 和 Edward Beeching 将讨论一种强大的对齐技术，称为 Direct Preference Optimisation (DPO)...</li><li><a href="https://google.github.io/mesop/">Mesop</a>：未找到描述</li><li><a href="https://www.tensorflow.org/guide/tpu">无标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1248071883371577434)** (2 messages): 

- **通过 LLM 提供监管和研究辅助**：LLM 可以作为 **Research Assistant**，利用科学论文和监管文件等资源回答技术和监管问题。对于像 Arxiv 这样已有的仓库，建议采用 RAG 和 prompt engineering，而对于付费墙后的资源则需要进行微调。
  
- **研究发现助手**：第二个用例是 **Research Assistant**，通过分析摘要、标题和元数据来指向有前景的论文，即使完整访问受限，这仍然很有价值。SciHub 和 DTIC 等工具可以支持这一计划，重点为用户筛选潜在论文。

- **用于法律文档分析的 LLM**：上述研究助手 LLM 可以适配处理法律文件，以实现高效的研究和发现。发布者表示有兴趣看到这一点的实现。

- **针对大型组织的文档提炼器**：该 LLM 专为拥有庞大文档库（如金融、政府）的组织量身定制，通过总结并根据推断的用户意图返回相关文档，协助监管合规。这一想法得到了纽约联邦储备银行 AI 主席在 DataScience Salon 会议上的发言支持。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1248101600271269938)** (5 messages): 

```html
- **Jeremyhoward 喜欢海南**: *"I love hainan! 😄"*。随后，Blaine 分享了他对神州半岛的热爱，并提到了附近的海滩和百香果。
- **来自印度的 Anmol 寻求聊天机器人定价建议**: Anmol 询问了关于企业级客服聊天机器人的定价建议。他希望有经验的人能为他提供帮助。
- **从河内到德国的转变**: Hehehe0803 介绍了自己，他来自越南河内，目前居住在德国。他提到自己加入较晚，并希望能与大家建立联系。
```
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1248007948622626907)** (8 messages🔥): 

```html
- **寻求 Modal 隐私政策**: 一位用户询问了 Modal 的隐私政策。另一位用户提供了一个 Google 搜索链接以获取更多信息：[Privacy Policy Modal Labs](https://www.google.com/search?q=privacy+policy+modal+labs)。

- **对 LLM 推理设置的困惑**: 一位用户询问如何设置服务器以运行 LLM 并暴露端点，并参考了 [GitHub 上的 Modal 示例脚本](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/text_generation_inference.py#L240C20-L240C30)。他们不确定如何从 Postman 等 REST 客户端获取调用端点的 Base URL。

- **来自 GPU 爱好者的 Modal 赞誉**: 一位通常在本地使用多块 GPU 进行训练的用户尝试了 Modal，并觉得它“超级酷”。他们用表情符号表达了赞赏：👍👏。

- **Axolotl 配置中的数据集处理问题**: 一位用户遇到了 Modal 坚持要求传递数据集的问题，这覆盖了他们现有的 Axolotl 配置。他们提到通过修改 `train.py` 以移除数据集代码，从而解决了这个问题。
```

**提到的链接**: <a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/text_generation_inference.py#L240C20-L240C30">modal-examples/06_gpu_and_ml/llm-serving/text_generation_inference.py at main · modal-labs/modal-examples</a>: 使用 Modal 构建的程序示例。可以通过在 GitHub 上创建账号来为 modal-labs/modal-examples 做出贡献。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1248228560737669211)** (1 messages): 

- **乐于发现“旧闻”**: 一位成员分享了他们发现的一个令人兴奋的成果，并附上了[这篇来自 arXiv 的论文](https://arxiv.org/pdf/2207.09238)。他们承认这可能是“旧闻”，但表示自己“非常喜欢”。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1248002201100750859)** (2 messages): 

- **CUDA 版本不匹配错误**: 一位成员在安装 `xformers` PIP 模块时，由于 CUDA 版本不同（**11.8 和 12.1**）遇到了不匹配错误。他们询问在 Jarvis Labs 容器中升级 CUDA 库的推荐方法。
- **更新 CUDA 的解决方案**: 另一位成员提供了 [Jarvis Labs 的文档链接](https://jarvislabs.ai/docs/troubleshooting#updating-cuda)，以指导如何在 Jarvis Labs 实例上更新 CUDA 版本。

**提到的链接**: <a href="https://jarvislabs.ai/docs/troubleshooting#updating-cuda">Debugging and Troubleshooting | Jarvislabs</a>: 关于更新 CUDA、释放磁盘空间等一些常见的故障排除技巧。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1248317632000823367)** (3 messages): 

- **成员仍在等待积分**: 多位用户报告称填写了网页表单，但**尚未收到任何预期的积分**。*"填写了网页表单，但据我所知还没有收到任何积分。"*

- **新表单即将推出**: 关于该情况的更新提到，新表单将很快发布，目前正在进行**状态检查**。*"我们很快会发布一个新表单，让我检查一下目前的进度。"*

### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1248019842116751401)** (9 messages🔥): 

- **通过邮件兑换额度有效**：一位成员必须通过邮件接受额度，并分享说“这对他有效”。
- **未收到额度的查询**：一位成员表示：“嗨 @zeke6585，还没收到邮件”，这促使另一位成员检查记录，发现该成员未填写 Replicate 额度的表单。
- **重复提交表单仍未解决**：一位成员表示困惑，他填写了两次表单，并收到了来自 OAI、HF 和 Modal 等其他服务的额度，但唯独没有收到 Replicate 的。
- **关于额度状态的评论澄清**：一位成员澄清他并非在每个频道刷屏，而只是在额度仍处于待处理状态的频道发帖，并指出 Ankur 立即解决了 BrainTrust 的额度问题。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1248014877440868484)** (29 messages🔥): 

- **Langsmith 输入缺失问题已解决**：一位成员在使用 Langsmith 的 `@traceable` 装饰器时遇到困难，该装饰器捕获了输出但未捕获 LLM 调用的输入。他们通过向函数添加参数解决了此问题，意识到输入本质上是函数的参数，没有参数就无法捕获任何内容。

- **LangSmith 额度困惑**：多位成员对未收到计算额度或不理解额度类型表示困惑和沮丧。一位成员强调：“Beta 额度仅适用于 LangSmith Beta 用户。”

- **账单与额度后续跟进**：几位设置了账单的用户抱怨未收到额度。他们被引导联系 LangSmith 的相关负责人，通过发送其组织 ID 来解决问题。

- **HIPAA 合规性与企业选项**：LangSmith 旨在 7 月 1 日前实现 HIPAA 合规，并为企业客户提供私有化部署（self-hosted）选项。有关计划和功能的详细信息已发布，并引导成员联系以获取更多信息。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.smith.langchain.com/pricing">Pricing | 🦜️🛠️ LangSmith</a>: 方案</li><li><a href="https://docs.smith.langchain.com/category/self-hosting">Self-hosting | 🦜️🛠️ LangSmith</a>: 私有化部署 LangSmith 需要企业授权。查看下方指南获取更多信息。</li><li><a href="https://docs.smith.langchain.com/how_to_guides/tracing/annotate_code#use-traceable--traceable)">Annotate code for tracing | 🦜️🛠️ LangSmith</a>: 有多种方式可以将追踪日志记录到 LangSmith。</li><li><a href="https://docs.smith.langchain.com/how_to_guides/tracing/annotate_code#wrap-the-openai-client">Annotate code for tracing | 🦜️🛠️ LangSmith</a>: 有多种方式可以将追踪日志记录到 LangSmith。</li><li><a href="https://docs.smith.langchain.com/how_to_guides/tracing/log_llm_trace">Log custom LLM traces | 🦜️🛠️ LangSmith</a>: 如果不按正确格式记录 LLM 追踪，系统也不会崩溃，数据仍会被记录。但是，数据将无法以针对 LLM 特化的方式进行处理或渲染。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1248330512297230427)** (2 messages): 

- **工作坊视频获得高度评价**：一位成员对 **LLM Evals** 视频中的**演讲者和主题**表示感谢，称该课程比他一年的自我研究更有效地帮助他构建了方法论。他们感谢 **<@525830737627185170>** 和 **<@916924724003627018>** 的组织。
- **协调人感谢正面反馈**：一位组织者对赞扬表示感谢，分享说这种鼓励非常有动力。“这让我的一天都很愉快，而且超级有动力！”
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-4](https://discord.com/channels/1238365980128706560/1242223495673286737/1248015694768115922)** (4 messages): 

- **会议演讲最初没有声音**：一位成员注意到“Conference Talk: Best Practices For Fine Tuning Mistral w/ Sophia Yang”的录音似乎没有音频。随后他们澄清道：“没关系，有声音了。”
- **澄清 Replicate 与 Modal 的部署区别**：一位成员寻求确认 Replicate 和 Modal 之间不同的部署流程，特别是关于 Docker 构建过程发生在哪里。另一位成员确认：“Modal 的构建过程是在远程运行的。”
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1248280518714200135)** (2 条消息): 

- **Ben 的演讲因病推迟**：由于 Ben 身体不适，团队已将他的演讲推迟到下周。祝愿 Ben 早日康复 ❤️‍🔥。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1248012391250268362)** (150 条消息🔥🔥): 

- **对 RAG 中信息图表和表格的困扰**：用户讨论了在 RAG 系统中从 PDF 提取数据（尤其是表格和信息图表）的困难。PyMuPDF、AWS Textract 以及将 PDF 转换为 Markdown 等工具被提及作为潜在解决方案，但问题依然存在，如文中所述：*"在我的使用场景中，Markdown 表格大部分时间都是格式错乱的。"*
- **关于分块（Chunking）策略的辩论**：针对 RAG 文本数据分块的最佳实践展开了激烈讨论，建议范围从 500 到 800 个 Token，并带有 50% 的重叠。大家就正确进行分块以确保准确的上下文和检索的复杂性及必要性达成了一致。
- **通过微调和 Embeddings 优化 RAG**：强调了微调 Embedding 模型以获得更好 RAG 性能的重要性，并建议使用生成的合成数据。一位成员指出：*"我认为任何通过 RAG 盈利的公司，如果不微调 Embedding 模型，就是在错失良机。"*
- **关于 LanceDB 用于多模态 AI 的讨论**：讨论了将 LanceDB 作为 Pinecone 和 SQL 等数据库的替代方案，用于管理大规模多模态数据的 Embeddings。该数据库承诺 **“易于使用、可扩展且具有成本效益”**，并支持混合搜索解决方案。
- **分享的延伸阅读和工具链接**：分享了多个链接，涵盖了微调流水线、Embedding 量化以及 PDF 处理和 RAG 实现工具等资源。关键链接包括 [Langchain Multi-modal RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb)、[LlamaParse](https://github.com/run-llama/llama_parse) 以及 [创建 Embedding 合成数据](https://x.com/_philschmid/status/1798388387822317933)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://lancedb.github.io/lancedb/">LanceDB - LanceDB</a>: 未找到描述</li><li><a href="https://lancedb.com">LanceDB - The Database for Multimodal AI</a>: 多模态 AI 数据库</li><li><a href="https://lancedb.github.io/lancedb/fts/">Full-text search - LanceDB</a>: 未找到描述</li><li><a href="https://jxnl.github.io/blog/writing/2023/09/17/rag-is-more-than-embeddings/?h=rag+more">RAG is more than just embedding search - jxnl.co</a>: 未找到描述</li><li><a href="https://manisnesan.github.io/chrestotes/posts/2023-07-07-doc-expansion-by-query-pred.html">chrestotes - Document Expansion by Query Prediction to Improve Retrieval Effectiveness</a>: 未找到描述</li><li><a href="https://useinstructor.com/blog/2024/06/06/enhancing-rag-with-time-filters-using-instructor/">Enhancing RAG with Time Filters Using Instructor - Instructor</a>: 未找到描述</li><li><a href="https://pymupdf.readthedocs.io/en/latest/rag.html">PyMuPDF, LLM &amp; RAG - PyMuPDF 1.24.4 documentation</a>: 未找到描述</li><li><a href="https://github.com/xavctn/img2table">GitHub - xavctn/img2table: img2table is a table identification and extraction Python Library for PDF and images, based on OpenCV image processing</a>: img2table 是一个基于 OpenCV 图像处理的用于 PDF 和图像的表格识别与提取 Python 库 - xavctn/img2table</li><li><a href="https://github.com/VikParuchuri/marker/tree/master">GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy</a>: 快速且高精度地将 PDF 转换为 Markdown - VikParuchuri/marker</li><li><a href="https://pymupdf.readthedocs.io/en/latest/the-basics.html#extracting-tables-from-a-page">The Basics - PyMuPDF 1.24.4 documentation</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/embedding-quantization">Binary and Scalar Embedding Quantization for Significantly Faster &amp; Cheaper Retrieval</a>: 未找到描述</li><li><a href="https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/">Parent Document Retriever | 🦜️🔗 LangChain</a>: 在为检索拆分文档时，通常存在相互冲突的需求：</li><li><a href="https://github.com/run-llama/llama_parse/issues/202">Mistakes parsing data from table using LlamaParse and gpt4o · Issue #202 · run-llama/llama_parse</a>: 尝试从 PDF 文件中提取表格数据（表格以图像形式嵌入）。虽然我成功提取了一些数据，但当表格位于页面底部时，始终会出现错误...</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: 解析文件以实现最佳 RAG。通过在 GitHub 上创建账户，为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://github.com/jxnl/n-levels-of-rag">GitHub - jxnl/n-levels-of-rag</a>: 通过在 GitHub 上创建账户，为 jxnl/n-levels-of-rag 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/parent_document_retriever/#retrieving-larger-chunks">How to use the Parent Document Retriever | 🦜️🔗 LangChain</a>: 在为检索拆分文档时，通常存在相互冲突的需求：</li><li><a href="https://x.com/jxnlco/status/1798708039383679166">Tweet from jason liu (@jxnlco)</a>: 哪一个能表达“订阅我的新闻通讯，邀请我参加你的会议，把你的工程团队交给我？”</li><li><a href="https://modal.com/blog/fine-tuning-embeddings">Beating Proprietary Models with a Quick Fine-Tune</a>: 仅需几百个示例进行微调，即可启动你自己的数据飞轮。</li><li><a href="https://x.com/_philschmid/status/1798388387822317933">Tweet from Philipp Schmid (@_philschmid)</a>: 创建一个用于生成合成数据以微调自定义嵌入模型的流水线。👀 第一步 创建知识库：从准备你的领域特定知识库开始，例如 PDF 或...</li><li><a href="https://python.useinstructor.com/blog/">Welcome to the Instructor Blog - Instructor</a>: 未找到描述</li><li><a href="https://jxnl.github.io/blog/writing/2024/02/28/levels-of-complexity-rag-applications/">Levels of Complexity: RAG Applications - jxnl.co</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/optimizing/production_rag/">Building Performant RAG Applications for Production - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb">langchain/cookbook/Multi_modal_RAG.ipynb at master · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/multi_doc_together_hybrid/#building-hybrid-retrieval-with-chunk-embedding-parent-embedding">Chunk + Document Hybrid Retrieval wi</a>: Chunk + 文档混合检索

th Long-Context Embeddings (Together.ai) - LlamaIndex</a>: 未找到描述</li><li><a href="https://lancedb.com/">LanceDB - 多模态 AI 数据库</a>: 多模态 AI 数据库</li><li><a href="https://github.com/kingjulio8238/memary">GitHub - kingjulio8238/memary: 自主 Agent 的长期记忆。</a>: 自主 Agent 的长期记忆。通过在 GitHub 上创建账号来为 kingjulio8238/memary 的开发做出贡献。</li><li><a href="https://blog.dottxt.co/coalescence.html">Coalescence: 让 LLM 推理速度提升 5 倍</a>: 未找到描述</li><li><a href="https://x.com/jxnlco">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://dub.sh/jxnl-rag">RAG - jxnl.co</a>: 未找到描述</li><li><a href="https://jxnl.co/writing/2024/05/22/systematically-improving-your-rag/">系统性地改进你的 RAG - jxnl.co</a>: 未找到描述</li><li><a href="https://jxnl.co/writing/2024/05/11/low-hanging-fruit-for-rag-search/">RAG 搜索中唾手可得的成果 - jxnl.co</a>: 未找到描述</li><li><a href="https://jxnl.co/writing/2024/02/28/levels-of-complexity-rag-applications/">复杂度层级：RAG 应用 - jxnl.co</a>: 未找到描述</li><li><a href="https://jxnl.co/writing/2024/02/05/when-to-lgtm-at-k/">停止使用 LGTM@Few 作为指标（更好的 RAG） - jxnl.co</a>: 未找到描述</li><li><a href="https://jxnl.github.io/blog/writing/2024/01/07/inverted-thinking-rag/">如何构建一个糟糕的 RAG 系统 - jxnl.co</a>: 未找到描述</li><li><a href="https://python.useinstructor.com/blog/2023/09/17/rag-is-more-than-just-embedding-search/">RAG 不仅仅是 Embedding 搜索 - Instructor</a>: 未找到描述</li><li><a href="https://python.useinstructor.com/">欢迎来到 Instructor - Instructor</a>: 未找到描述</li><li><a href="https://www.timescale.com/">PostgreSQL ++ 用于时间序列和事件</a>: 专为处理苛刻的工作负载而设计，如时间序列、向量、事件和分析数据。基于 PostgreSQL 构建，提供专家支持且不收取额外费用。</li><li><a href="https://www.limitless.ai/">Limitless</a>: 超越你思想的局限：由你所见、所言、所闻驱动的个性化 AI。</li><li><a href="https://www.raycast.com/">Raycast - 你通往一切的快捷方式</a>: 一系列强大的生产力工具，全部集成在一个可扩展的启动器中。</li><li><a href="https://www.tensorlake.ai/">Tensorlake</a>: 未找到描述</li><li><a href="https://dunbar.app/">首页</a>: 你的个人奇遇引擎。智能连接新员工入职、同行学习、虚拟咖啡等。免费试用 dunbar，无需信用卡，激发有意义的连接...</li><li><a href="https://www.bytebot.ai/">Bytebot - 在网页抓取、自动化、测试和监控中利用 AI 的力量。</a>: 使用我们支持 AI 的 SDK 增强并简化你的浏览器自动化。有了 Bytebot，创建网页任务就像编写 Prompt 一样简单。</li><li><a href="https://www.narohq.com/">Naro - AI 驱动的销售知识</a>: 未找到描述</li><li><a href="https://trunktools.com/">Trunk Tools</a>: Trunk Tools 处于建筑创新的前沿，提供尖端的 AI 解决方案以简化项目管理。</li><li><a href="https://modal.com/">Modal: 面向开发者的高性能云</a>: 自带代码，在大规模环境下运行 CPU、GPU 和数据密集型计算。面向 AI 和数据团队的 Serverless 平台。</li><li><a href="https://docs.pydantic.dev/latest/">欢迎来到 Pydantic - Pydantic</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1248104305756737586)** (6 条消息): 

- **讨论了独家预览内容**: Jeremy Howard 与成员分享了一个特别的预览，并嘱咐道：“所以请大家保密！”尽管具有排他性，他提到在 Discord 内部讨论是可以的。
- **对演示的期待与日俱增**: 在偷看了代码库后，一位成员表示打算等待 Jeremy Howard 的演示。
- **演讲安排在不方便的时间**: Ashpun 指出，演讲时间对他们来说很不方便，是在 IST 凌晨 3:30。
- **设置多个闹钟**: 为了不错过演讲，一位成员幽默地提到设置了“10 个闹钟”，而另一位成员则很高兴在凌晨时分有人陪伴。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[yang_mistral_finetuning](https://discord.com/channels/1238365980128706560/1242224842053521459/1248018263779311676)** (2 条消息): 

- **通过同理心化解误解**：*啊，好的，我现在明白了。这很有道理，也是我预料之中的。抱歉刚才误解了你。*

- **对 Mistral API 的兴奋**：一位成员对即将举行的研讨会表示热切期待，并表示：*我将尝试官方的 Mistral API*。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1248256278082490398)** (13 条消息🔥): 

- **Axolotl 安装过程中的依赖冲突**：一位用户在 CUDA 12.1、Python 3.11 和 PyTorch 2.3.1 环境下安装 Axolotl 时遇到错误，原因是 Axolotl、Accelerate、Bitsandbytes 和 Xformers 等多个包之间存在依赖冲突。他们正在寻求这些冲突的解决方案。

- **建议在不安装 Xformers 的情况下安装 Axolotl**：一位成员建议该问题专门指向 Xformers 而非 Axolotl，并建议先在不安装 Xformers 的情况下安装 Axolotl。他们还提到了使用 Axolotl 的 Docker 镜像作为替代方案。

- **切换到 Python 3.10 解决了部分问题**：用户通过切换到 Python 3.10 和 PyTorch 2.1.2 解决了部分问题，这使他们能够运行 preprocess 步骤，但随后遇到了与 Flash Attention 相关的新错误。

- **Flash Attention 需要针对 CUDA 12.1 重新编译**：用户遇到了与 Flash Attention 相关的 `libcudart.so.11.0` ImportError，这表明与他们安装的 CUDA 版本 (12.1) 不匹配。建议的解决方案是重新构建/重新编译 Flash Attention，这解决了该问题。

**提到的链接**：<a href="https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts">Dependency Resolution - pip documentation v24.1.dev1</a>：未找到描述

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1248244530843619401)** (12 条消息🔥): 

- **合并 Adapter 后 Safe tensors 数量不一：** 一位成员询问为什么将 Adapter 合并到 Base Model 时，有时会产生 3 个 Safe tensors，有时却是 6 个，而 Base Model 最初只有 3 个，这表明结果存在变数。

- **利用 Keras 的 `steps_per_execution` 提升 TPU 效率：** 一位成员分享了一篇关于使用 `steps_per_execution` 来减少 Python 开销并最大化 TPU 性能的 [TensorFlow 博客](https://www.tensorflow.org/guide/tpu)。在 PyTorch XLA 中，另一种方法是调整对 `xm.mark_step()` 的调用以获得类似收益。

- **明智地使用 `xm.mark_step()`：** 详细解释了如何在 PyTorch XLA 中通过调整训练循环（training loop）内 `xm.mark_step()` 的调用频率来管理 TPU 性能，以平衡性能和可靠性，并建议为 Accelerate 库提交潜在的功能请求。

- **Less Wright 的 FSDP 教程：** 对于那些对 FSDP 感兴趣的人，推荐了由 Less Wright（前 fastai 校友）制作的优秀的 [YouTube 十部分系列教程](https://www.youtube.com/playlist?list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT)，作为全面的入门指南。

- **量化过程澄清：** 确认量化发生在模型加载期间，并在传递给 GPU 之前由 CPU 处理。提供了相关的文档和资源，包括 [Hugging Face Accelerate 的量化使用指南](https://huggingface.co/docs/accelerate/en/usage_guides/quantization) 和 [Accelerate GitHub 仓库](https://github.com/huggingface/accelerate/blob/v0.30.1/src/accelerate/utils/bnb.py#L44)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/playlist?list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT">PyTorch FSDP Tutorials</a>：未找到描述</li><li><a href="https://github.com/huggingface/accelerate/blob/v0.30.1/src/accelerate/utils/bnb.py#L44">accelerate/src/accelerate/utils/bnb.py at v0.30.1 · huggingface/accelerate</a>：🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简单方法，支持自动混合精度（包括 fp8），以及易于配置的 FSDP 和 DeepSpeed 支持……</li><li><a href="https://huggingface.co/docs/accelerate/en/usage_guides/quantization">Quantization</a>：未找到描述</li><li><a href="https://www.tensorflow.org/guide/tpu">无标题</a>：未找到描述</li><li><a href="https://github.com/huggingface/accelerate/issues">Issues · huggingface/accelerate</a>：🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简单方法，支持自动混合精度（包括 fp8），以及易于配置的 FSDP 和 DeepSpeed 支持……
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1248323683907276800)** (2 messages): 

- **观看 YouTube 链接**：分享了一个 YouTube 视频，可以通过 [这里](https://m.youtube.com/watch?v=44vi31hehw4) 访问。

- **Python 处理服务端代码**：关于服务端代码的讨论表明，它仍然由 **Python** 管理。要点包括 **Spaces 中的 Scaling** 以及使用 Python 处理 **Concurrency**（并发）。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1248030811513688074)** (12 messages🔥): 

- **使用 Modal 进行 Finetuning 可能很复杂，但值得探索**：Charles 提到了使用 Modal 的 Finetuning 技术栈的复杂性，但建议用户查看他们更简单的 [Dreambooth 示例](https://modal.com/docs/examples/dreambooth_app)。该示例通过使用 Textual Inversion 对 Stable Diffusion XL 模型进行 Finetuning，展示了 Modal 的核心概念。
- **适配现有数据集以便在 Modal 上进行 Finetuning**：正如 Charles 所建议的，用户可以直接指向 Hugging Face 数据集。应进行调整以避免使用 Volumes 来存储这些数据。 
- **使用 Batch Processing 进行验证**：Charles 建议编写自定义的 `batch_generate` 方法，并使用 `.map` 来处理和生成验证示例。他参考了 "embedding Wikipedia" 示例以获取进一步指导。
- **探索 Modal 以提高应用托管的成本效益**：Charles 向 Alan 建议，可以考虑将其 Streamlit 应用的 Retrieval 部分迁移到 Modal，或者考虑将整个应用迁移过去。讨论了关于 Cold Starts 以及 24/7 部署成本的担忧。
- **快速支持脚本错误**：Chaos 指出了 GitHub 上 `vllm_inference.py` 脚本的一个问题，Charles 迅速做出了回应，暗示可能是 Build 步骤或 GPU 可用性方面的问题。Charles 强调了 Modal 在支持和沟通方面的快速响应文化。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dashboard.render.com/">Cloud Application Hosting for Developers | Render</a>：Render 是一个统一的云平台，用于构建和运行所有应用和网站，提供免费 SSL、全球 CDN、私有网络以及来自 Git 的自动部署。</li><li><a href="https://github.com/modal-labs/modal-examples/issues/763">Error when Running `vllm_inference.py`: `CancelledError() · Issue #763 · modal-labs/modal-examples</a>：我在尝试运行 Modal Examples 仓库中的 vllm_inference.py 脚本时遇到了问题。以下是我遵循的步骤和遇到的错误：重现步骤...</li><li><a href="https://modal.com/docs/examples/dreambooth_app">Pet Art Dreambooth with Hugging Face and Gradio</a>：本示例使用来自 “Dreambooth” 论文的 Textual Inversion 技术，在宠物照片（默认是一只名为 Qwerty 的小狗）上对 Stable Diffusion XL 模型进行 Finetuning。实际上，它教会了...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langchain-langsmith](https://discord.com/channels/1238365980128706560/1242564256914870384/1248081428357447691)** (1 messages): 

- **Langsmith 在追踪流式非 OpenAI 模型时遇到障碍**：一位用户讨论了他们在以流式方式使用 **Groq/Llama3** 时，使用 **Langsmith** 捕获 Traces 的挑战。他们注意到使用 `@traceable` 装饰器不适用于流式输出，因为结果保持为 `generator` 对象。

### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1247988604966273075)** (22 messages🔥): 

- **最后一刻入学的学员仍可领取额度**：一位在 5 月 30 日入学的用户询问如何领取额度，Dan 澄清说虽然像 OAI 这样的平台不再提供额度，但像 Modal 这样的平台可能仍然可用。他分享了领取 **OAI credits** 的链接：[OAI credits](https://discord.com/channels/1238365980128706560/1245927985123692575/1248045829705695333)。

- **Modal 额度现已可兑换**：对于 5 月 30 日之后入学的用户，Dan 指导他们通过提供的表单 [Modal form](https://bit.ly/modal-credits) 领取 Modal 额度。他还提供了使用 Modal 平台的逐步指南，并分享了多个示例项目。

- **额度申请仍在处理中**：Dan 解释说额度正在分批处理，并安抚尚未收到额度的用户，让他们检查是否填写了必要的表单。他敦促没有额度的用户在周二截止日期前使用 Modal 平台领取额外额度。

- **Replicate 账户问题已解决**：一位未收到额度的用户核实了他们的 Replicate 账户设置。在确认了正确的电子邮件并注意到账户创建日期的差异后，他们在电子邮件中找到了额度。

- **Replicate 账单设置问题已重定向**：Dan 要求收到 Replicate 邀请的用户将账单相关问题定向到更合适的频道，以便从 Replicate 团队获得更快的回复。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://bit.ly/modal-credits">Modal hackathon credits</a>：要领取您的 Modal 额度，请先在 https://modal.com/ 注册账户。然后，通过此表单告知我们您的用户名。如需支持，请加入 Modal Slack。这里有一些示例可以开始...</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[strien_handlingdata](https://discord.com/channels/1238365980128706560/1243773476301443073/1248012168868139099)** (1 messages): 

- **对数据讲座演示的赞赏**：一位成员对某场关于数据的演示表示高度赞赏，强调了其对于评估和 Finetuning 等任务的基础重要性。他们认为这场讲座作为会议的开场演讲会非常有益。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1247999686351257651)** (28 messages🔥): 

- **账户 ID 混淆导致额度问题**：多位用户报告了未收到额度的问题，通常是因为在表单中误填了错误的账户详情，例如填了电子邮件而不是账户 ID。示例包括用户 ID 如 *biggafish8-37cf1d* 和 *jain-nehil-ab4ee8*。

- **修正后额度现已可见**：一位用户确认在提供正确的账户 ID（*szilvia-beky-38bda3*）后，他们的额度已变得可见。这表明一旦处理了正确的详细信息，一些用户的问题已成功解决。

- **额度过期通知**：当被问及额度的有效期时，明确了额度将在一年后过期。**aravindputrevu** 证实了这一点，称：“是的，它们在一年后过期。”

- **关于表单填写问题的频繁指导**：**hamelh** 强调，错误的表单填写（如输入电子邮件地址而不是账户 ID）是一个常见问题。**project_disaster** 对此错误表示道歉并予以承认。

- **持续寻求帮助**：用户继续寻求有关其账户额度的帮助，并提供了他们的账户 ID，如 *raul-brebenaru-2d3d45* 和 *roger-6803a6*，这表明即使在初步的错误纠正努力之后，问题仍在持续。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[emmanuel_finetuning_dead](https://discord.com/channels/1238365980128706560/1245129595749925086/1248044505530236968)** (98 messages🔥🔥):

- **Fine-tuning 并没有消亡，但已成为小众需求**：一位成员幽默地建议将标题定为 *“Fine-tuning is not dead, it's niche”* 以求清晰，强调 Fine-tuning 具有专门且昂贵的应用场景。他们指出，“Fine-tuning 可能会增加问答环节的幻觉（hallucinations）”，突显了其复杂性和成本。
- **Anthropic 与推测性想法**：成员们讨论了 Anthropic 对未来重大变化的看法，随后幽默地提到了 “*Cylons*”。一位 Anthropic 代表的发言被重点引用：“Anthropic 押注 8 年后人类可能不会以现在的形式存在。”
- **Fine-tuning 与 RAG 的资源共享**：分享了多个关于 Fine-tuning 和 RAG 的资源，例如 [Simon Willison 的博客](https://simonwillison.net/) 和 [Anthropic 的研究](https://www.anthropic.com/)。Emmanuel 的书以及 LLM CLI 等工具被提及，认为对理解 Fine-tuning 的应用层面非常有价值。
- **优先选择 Prompting 而非 Fine-tuning**：对于许多应用，成员们更倾向于关注 Prompt engineering 而非 Fine-tuning，并建议阅读 Emmanuel 关于 Prompt engineering 的电子表格和 [HuyenChip 的博客](https://huyenchip.com/2023/04/11/llm-engineering.html#prompt_optimization)。有人幽默地建议：“做那些无聊的事！”以此作为替代复杂 Fine-tuning 的建议。
- **Dynamic Few-Shot 与 RAG 的讨论**：讨论以对使用 Dynamic Few-Shot prompting 和 RAG 作为 Fine-tuning 的可行替代方案或补充的见解结束。分享了诸如 [Dynamic Few-Shot prompting 文章](https://medium.com/@iryna230520/dynamic-few-shot-prompting-overcoming-context-limit-for-chatgpt-text-classification-2f70c3bd86f9) 等链接，以强调在不断发展的应用中的实践方法。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://discordapp.com/channels/1238365980128706560/1242223458184597534/1245504052738129961">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 非常适合玩游戏和与朋友闲逛，甚至可以建立全球社区。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://www.amazon.com/Building-Machine-Learning-Powered-Applications/dp/149204511X">未找到标题</a>: 未找到描述</li><li><a href="https://medium.com/@iryna230520/dynamic-few-shot-prompting-overcoming-context-limit-for-chatgpt-text-classification-2f70c3bd86f9">Dynamic Few-Shot Prompting: Overcoming Context Limit for ChatGPT Text Classification</a>: 最近像 ChatGPT 这样的大型语言模型（LLM）的普及爆发，导致它们在经典 NLP 任务中的使用增加，例如……</li><li><a href="https://www.mlpowered.com/">mlpowered</a>: 博客文章和其他信息</li><li><a href="https://www.kaggle.com/competitions/kaggle-llm-science-exam/leaderboard">Kaggle - LLM Science Exam</a>: 未找到描述</li><li><a href="https://www.mlpowered.com/book/">A book about practical problems</a>: 现已在 Amazon 和 O’Reilly 上架。我写这本书是为了给读者提供解决最常见实际 ML 问题的工具，这些工具基于我指导数百名 Data Scientists 和 ML 工程师的经验……</li><li><a href="https://x.com/stefanhgm/status/1765466556216053879">来自 Stefan Hegselmann (@stefanhgm) 的推文</a>: 在训练或 Prompt 数据中删除不支持的事实是否能有效减少幻觉？我们针对 GPT-4 和 Llama 2 在生成患者摘要方面进行了测试。与 @shannonzshen, Florian Gie... 合作。</li><li><a href="https://medium.com/@iryna230520/dynamic-few-shot-prompting-overcoming-context-limit-for-chatgpt-">未找到标题</a>: 未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/1w7gnJTevZwVojzfrmyb3IPI7-k7DtdOaN0r2wTWnNHU/edit#gid=150872633">Anthropic Prompt Engineering Interactive Tutorial [PUBLIC ACCESS]</a>: 教程指南。本教程需要 API key 才能进行交互。如果你没有 API key，可以通过 &lt;a href=&quot;https://console.anthropic.com/&... 注册一个。</li><li><a href="https://simonwillison.net/">Simon Willison’s Weblog</a>: 未找到描述</li><li><a href="https://www.quora.com/Should-you-fine-tune-an-LLM-or-just-do-prompt-engineering/answer/Tong-Hui-Kang-1">Tong Hui Kang 对“应该微调 LLM 还是只做 Prompt Engineering？”的回答 - Quora</a>: 未找到描述</li><li><a href="https://www.quora.com/What-is-the-future-of-prompt-engineering-versus-fine-tuning/answer/Tong-Hui-Kang-1">Tong Hui Kang 对“Prompt Engineering 与微调的未来是什么？”的回答 - Quora</a>: 未找到描述</li><li><a href="https://llm.datasette.io/en/stable/">LLM: A CLI utility and Python library for interacting with Large Language Models</a>: 未找到描述</li><li><a href="https://github.com/simonw/llm">GitHub - simonw/llm: Access large language models from the command-line</a>: 从命令行访问大型语言模型 - simonw/llm</li><li><a href="https://arxiv.org/abs/2401.08406">RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture</a>: 开发者在构建大型语言模型（LLM）应用时，有两种常见的方式来整合专有和特定领域的数据：检索增强生成（RAG）和……</li><li><a href="https://arxiv.org/abs/2303.17564">BloombergGPT: A Large Language Model for Finance</a>: NLP 在金融科技领域的应用广泛且复杂，应用范围从情感分析和命名实体识别到问答系统。大型语言模型（L...</li><li><a href="https://arxiv.org/abs/2305.05862">Are ChatGPT and GPT-4 General-Purpose Solvers for Financial Text Analytics? A Study on Several Typical Tasks</a>: 最新的大型语言模型（LLM），如 ChatGPT 和 GPT-4，展示了通用模型的卓越能力，在广泛的 NLP 任务中实现了最先进的性能……</li><li><a href="https://arxiv.org/abs/2402.15422">A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models</a>: 患者在理解住院情况时经常面临困难，而医疗工作者的解释资源有限。在这项工作中，我们研究了大型语……</li><li><a href="https://huyenchip.com/2023/04/11/llm-engineering.html#prompt_optimization">Building LLM applications for production</a>: [Hacker News 讨论, LinkedIn 讨论, Twitter 线程]</li><li><a href="https://docs.google.com/spreadsheets/d/1w7gnJTevZwVojzfrmyb3IPI7-k7DtdOaN0r2wTWnNHU">Anthropic Prompt Engineering Interactive Tutorial [PUBLIC ACCESS]</a>: 教程指南。本教程需要 API key 才能进行交互。</li>

如果你没有 API key，你可以通过 <a href="https://console.anthropic.com/">...</a> 注册一个</li><li><a href="https://www.anthropic.com/">Home</a>: Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释且可控的 AI 系统。</li><li><a href="https://community.openai.com/t/fine-tuning-vs-context-injection-rag/550286">Fine-tuning vs Context-Injection (RAG)</a>: 大家好。我完成了关于比较 Fine-tuning 与 Context-Injection（作为 Retrieval-Augmented Generation 的一种实现）的研究工作。在组织实验方面投入了大量工作...</li><li><a href="https://arxiv.org/abs/2403.01432">Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge</a>: 大语言模型 (LLMs) 记忆了大量的实事知识，在不同任务和领域表现出强大的性能。然而，据观察，当涉及冷门知识时性能会有所下降...</li><li><a href="https://arxiv.org/abs/2312.05934">Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs</a>: 大语言模型 (LLMs) 在其预训练权重中封装了大量的实事信息，这从它们回答不同领域各种问题的能力中得到了证明。然而...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[braintrust](https://discord.com/channels/1238365980128706560/1245407617031999581/1248019594103492818)** (29 messages🔥): 

- **Braintrust 输入未被捕获**：一位用户注意到 Braintrust 装饰器没有捕获其 LLM 函数的输入，但捕获了输出。另一位用户指出该函数没有参数，并建议使用 `wrap_openai` 包装 OpenAI 客户端。
- **探索 Braintrust 追踪方法**：讨论了 Braintrust 中的三种追踪方法：`wrap_openai`、`@traced` 装饰器和 spans，其中 spans 提供了最大的灵活性。一位用户考虑在将 Braintrust 与 ZenML 集成的项目中使用 spans。
- **通过 UI 澄清解决额度问题**：一位名为 "project_disaster" 的用户提到没看到他们的额度，"ankrgyl" 澄清说没有 "Upgrade" 按钮表示额度已应用。该用户建议增加一个可见的仪表盘来跟踪随时间变化的额度消耗。
- **对 TypeScript 和追踪示例的兴趣**：一位用户对使用 TypeScript 表示了题外赞赏。另一位用户询问如何从 LLM 项目的追踪示例开始，最终计划转向 DPO finetuning，"ankrgyl" 建议他们从 Braintrust 文档上的日志指南开始。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.braintrust.dev/docs/guides/logging">Braintrust</a>: Braintrust 是用于构建 AI 产品的企业级技术栈。</li><li><a href="https://www.braintrust.dev/docs/guides/tracing#wrapping-openai">Braintrust</a>: Braintrust 是用于构建 AI 产品的企业级技术栈。</li><li><a href="https://www.braintrust.dev/docs/guides/tracing#annotating-your-code">Braintrust</a>: Braintrust 是用于构建 AI 产品的企业级技术栈。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1248157328855793707)** (2 messages): 

```html
- **家乡互动**：一位用户提到住在伦敦，但原籍葡萄牙。另一位用户选择用 “🤐” 表情符号对原籍保密。
```
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[announcements](https://discord.com/channels/1238365980128706560/1245460787196068030/1248046279532220529)** (2 messages): 

- **填写 OpenAI 额度表单**：*如果你错过了第一次填写表单并想要 OpenAI 额度，请在 [此链接](https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f) 的表单中填写你的 OAI org id。*

- **查找用于 OpenAI 额度的 Organization ID**：要查看你的 Org ID，*请在登录后访问 [此网站](https://platform.openai.com/settings/organization/general)，在 `Organization ID` 下查看（即使你*不是某个组织的一员*，此 ID 也是可用的）。*

**提及的链接**: <a href="https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f">未找到标题</a>: 未找到描述

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1248322865468674078)** (3 messages): 

- **关于起始模型格式的问题**：一位用户询问在训练 LoRAs 以获得最佳性能之前，是否应该从 **awq** 或 **gptq** 格式开始。对话中未提供具体的指导。
- **对 Predibase 示例的兴奋**：一位成员对在 [文档](https://docs.predibase.com/user-guide/examples/rag) 中分享的 Predibase 示例表示兴奋。Predibase 声称提供了 **fine-tune** 和 **serve** 开源 LLMs 的最快方式。
- **关于额度过期的查询**：一位用户询问了额度的过期时间，并提到需要添加信用卡才能在网站上升级到 Developer Tier。没有进一步讨论关于额度过期的细节。

**提到的链接**：<a href="https://docs.predibase.com/user-guide/examples/rag.">Quickstart | Predibase</a>：Predibase 提供了微调和部署开源 LLMs 的最快方式。它构建在开源 LoRAX 之上。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1247999434764193834)** (4 messages): 

- **成员仍在等待额度**：几位成员报告称他们尚未收到额度。其中一人提供了用于跟进的电子邮件，敦促尽快解决。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1247988384195149835)** (69 messages🔥🔥): 

```html
<!-- Summary -->

- **OpenAI 额度追溯应用**：几位用户注意到他们的额度已应用到现有的 API 余额中，类似于通过信用卡充值。[成员们讨论了](https://platform.openai.com/settings/organization/billing/overview) 针对 API 新手的潜在改进。
- **为学生敲定 Tier 2 API 状态**：OpenAI 为及时填写表格的人员授予了 Tier 2 API 状态，允许他们使用额外的额度。如果错过了初始注册，用户应关注后续更新。
- **额度补交申请表**：为了纠正早期的提交错误，[一个新的额外额度申请表](https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f) 已被分享，需要正确填写。
- **微调过程中的内部思考**：关于如何在 OpenAI 模型微调期间处理长多轮对话中的“internal thoughts”进行了深入讨论。提出了分隔符（Delimiters）和独立示例作为潜在解决方案。
- **公开致谢与表扬**：小组对 OpenAI 团队成员迅速而有效的支持表示感谢，并在 [Twitter 帖子](https://x.com/TheZachMueller/status/1798674326633247143) 中表达了谢意。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/TheZachMueller/status/1798674326633247143">来自 Zach Mueller (@TheZachMueller) 的推文</a>: 我们在想：作为 @HamelHusain 课程的一部分，我们要如何花掉价值 500 美元的 @OpenAI 额度？@OpenAI：噢，好主意！每个人都获得了 tier-2 状态。非常感谢 @shyamalanadkat 和 An...</li><li><a href="https://cookbook.openai.com/examples/third_party/gpt_finetuning_with_wandb">使用 Weights &amp; Biases 微调 OpenAI 模型 | OpenAI Cookbook</a>: 未找到描述</li><li><a href="https://cookbook.openai.com/examples/chat_finetuning_data_prep">聊天模型微调的数据准备与分析 | OpenAI Cookbook</a>: 未找到描述</li><li><a href="https://cookbook.openai.com/examples/fine_tuning_for_function_calling">针对 function calling 的微调 | OpenAI Cookbook</a>: 未找到描述</li><li><a href="https://cookbook.openai.com/examples/how_to_finetune_chat_models">如何微调聊天模型 | OpenAI Cookbook</a>: 未找到描述</li><li><a href="https://cookbook.openai.com/examples/fine-tuned_qa/ft_retrieval_augmented_generation_qdrant">针对检索增强生成 (RAG) 的微调（配合 Qdrant） | OpenAI Cookbook</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1247989753517838406)** (266 messages🔥🔥): 

```html
- **GPTs at their limits with advanced programming questions**: A user noted that their programming questions have become more specific and complex as their project advanced, leading to struggles with GPT models. They expressed concern that these models may be "pushing their limits for programming assistance 😁".
- **GPTs sometimes fail at simple corrections**: Another user pointed out a problem where the GPT could not correct an incorrect math equation despite being prompted, showcasing issues with basic logical consistency in the model.
- **Continuous Learning and Real-time Adjustments**: Discussion involved the idea that making models agentic and capable of continuous learning could be costly and pose regulatory challenges. Continuous learning could also lead to issues with personality drift and potential security risks.
- **Generative AI's current and future impact**: There was debate about the immediate usefulness and future potential of generative AI, with some users highlighting its potential to assist or significantly change job structures, while others were skeptical of its broader economic impacts.
- **Community discussions on AI advances and resource requirements**: Users conversed about the computational power required for training AI models, referencing specific hardware like A100 and H100 GPUs, and speculating on developments with upcoming models like GPT-5.
```
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1248049032002015282)** (20 messages🔥): 

- **语音聊天移除的担忧**：用户讨论了语音聊天被移除的原因，并推测其将随新模型回归。一位用户提到，基于 GPT-4o 的新语音模型预计很快发布。
- **生成中文文本的问题**：一位成员报告称，使用 GPT 模型生成中文回复时，有时会出现类似 \ufffd 的特殊字符。此问题发生概率约为 15%，严重影响了文本质量。
- **关于 GPT-4o 功能推出时间表的辩论**：成员们讨论了 GPT-4o 实时语音和视觉功能的预期推出时间。官方更新显示，最初将在未来几周内向 ChatGPT Plus 用户开放，并在接下来的几个月内扩大访问范围（[官方更新](https://x.com/OpenAI/status/1790130708612088054?t=bgpr-IBhwXMa4ZiLXCrJgg&s=19)）。
- **GPT-4o 免费计划限制**：讨论涉及了在 GPT-4o 免费计划中可以提问的数量。共识是限制在 10 个问题左右。


**提到的链接**：<a href="https://x.com/OpenAI/status/1790130708612088054?t=bgpr-IBhwXMa4ZiLXCrJgg&s=19">来自 OpenAI (@OpenAI) 的推文</a>：所有用户今天将开始获得 GPT-4o 的访问权限。在接下来的几周内，我们将开始向 ChatGPT Plus 推出今天演示的新语音和视觉功能。

  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1247992448827920385)** (6 messages): 

- **LLaVA-Mistral 视觉模型与 7b 共存**：一位成员提到，**llava-v1.6-mistral-7b** 的视觉模型在与 7b 模型结合时似乎运行良好。他们觉得这种集成能够实现非常棒。

- **对图片权限的困惑**：一位成员对频道中图片权限的移除表示沮丧，询问为何发生这种变化。另一位成员在后续消息中也表达了同样的疑问。

- **DALL-E Logo 中文本准确性的挑战**：成员们讨论了使用 **DALL-E** 生成带有精确文本的 Logo 的困难。一位成员分享了一种提高文本准确性的方法，即在提示词中包含重复检查和重新生成的过程，直到图像中的文本正确为止。另一位成员建议包含特定指令以强调图像生成过程中的不同文本层，并链接了一个自定义 GPT 提示词作为参考：[自定义 GPT 提示词](https://chatgpt.com/g/g-TKZI5nYMc-one-word-graphics)。
  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1247992448827920385)** (6 messages): 

- **7b 模型与 Vision 模型集成**：一位成员提到，当将 **7b model** 放置在 **llava-v1.6-mistral-7b** 的文件夹中时，它与该 Vision 模型配合良好。他们发现这种集成能够运行“挺酷的”。
- **频道图片权限担忧**：一位成员对频道中图片权限的移除表示困惑。他们质疑为什么要移除这些权限。
- **DALL-E 徽标中精确文本生成的困扰**：一位成员询问是否可以使用 **DALL-E** 生成带有精确文本的徽标，并分享了在“字母拼写错误”方面的困难。另一位成员分享了一个通用的 Prompt，通过不断检查和纠正文本直到准确为止。
- **DALL-E 图像准确文本的实用 Prompt**：一位成员分享了一个 Prompt，通过分层和检查准确性来确保 DALL-E 图像中的文本正确，直到达到预期的文本效果。他们还提到将此集成到 [自定义 GPT 指令](https://chatgpt.com/g/g-TKZI5nYMc-one-word-graphics) 中以满足用户需求。
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1247990342410702969)** (148 messages🔥🔥): 

```html
<ul>
  <li><strong>Gradient Accumulation 见解：</strong> 成员们讨论了 <em>gradient accumulation</em>（梯度累积）如何帮助解决内存问题和 batch size 设置。 “与小 batch size 相比，它会减少时间”，但由于内存分配的特性，在处理大 batch size 时会变得复杂。</li>
  <li><strong>解决 CUDA 内存问题：</strong> <em>“当增加 batch size 时，序列的不同长度会减慢进程。”</em> 建议使用 “gradient accumulation” 或 “非 2 的幂次的 batch size” 来缓解内存峰值。</li>
  <li><strong>训练与合并问题：</strong> 成员们遇到了 <em>merging trained adapters</em>（合并已训练的适配器）导致性能显著下降的问题。有人呼吁寻求有效加载适配器的方法，以便在不损失效率的情况下继续训练。</li>
  <li><strong>使用 Alpaca Prompt 进行推理：</strong> 分享了一个详细的代码片段，用于在微调后使用 <em>FastLanguageModel.for_inference</em> 配合 Alpaca 风格的 Prompt 生成序列补全。这源于 [一个分享的 Colab 链接](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing)。</li>
  <li><strong>对 Qwen2 模型的兴奋：</strong> 成员们对 Qwen2 模型的发布充满热情，特别是对小型模型（0.5B 到 7B）感兴趣，因为它们易于训练和使用。讨论涉及了“易于训练、易于迭代、随处运行”的前景。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ghost-x.org/blog/nvidias-stunning-gains-metas-privacy-challenges-and-spacexs-next-starship-test-the-future-of-ai-and-technology">英伟达的惊人增长、Meta 的隐私挑战以及 SpaceX 的下一次星舰测试：AI 与技术的未来</a>：探索科技界的最新进展：英伟达令人印象深刻的股价飙升和技术进步，Meta 备受争议的 AI 数据使用计划，SpaceX 即将进行的星舰测试，以及 Salesf...</li><li><a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://unsloth.ai">Unsloth AI | 微调 Llama 3 和 Mistral LLM</a>：为 AI 和 LLM 提供便捷的微调。开源且适合初学者。使用 Unsloth 提升速度。</li><li><a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行持续 LLM 预训练</a>：通过使用 Unsloth 对 Llama 3、Phi-3 和 Mistral 进行持续预训练，让模型学习一种新语言。</li><li><a href="https://www.youtube.com/watch?v=cwuYWFC7_QE">微调 LLM 快 30 倍！与 Daniel Han (Unsloth AI)</a>：成为 Patreon：https://www.patreon.com/theaiepiphany 👨‍👩‍👧‍👦 加入我们的 Discord 社区：https://discord.gg/peBrCpheKE Daniel Han 来自 Unsloth AI 加入了...</li><li><a href="https://tenor.com/view/%E7%9A%849-gif-27299608">的9 GIF - 的9 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://datta0.substack.com/p/ai-unplugged-12-mora-dpo-vs-ppo-cope">AI Unplugged 12：MoRA。DPO vs PPO。CoPE 上下文位置编码。S3D 自推测解码。</a>：洞察胜过信息</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1248016651170091058)** (9 messages🔥): 

- **现在每家公司都由 AI 驱动了？**：一位成员幽默地问道：*“现在是不是每家公司都只是由 AI 驱动的？”* 并调侃 GitHub：*“我敢发誓 GitHub 是由咖啡因和 'lgtm' 驱动的”*。
- **Unsloth 拯救了期末项目**：一位心怀感激的用户称赞了 Unsloth，表示：*“你的 SFT notebook 在 A100 上大约 25 分钟就训练好了”*，这对于快速调整超参数和修复数据至关重要。他们补充道：*“DPO 再次帮了我们大忙”*，强调了它对成功的关键作用。
- **吐槽 Kaggle 的用户体验**：一位成员抱怨 Kaggle 的用户体验 *“相当糟糕”*。他们分享了最近遇到的一个问题：训练 *“在 3 小时后卡死”*，尽管在 2 小时后尝试断开连接，但仍然处于卡住状态。
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1247989011629346857)** (54 messages🔥): 

```html
- **关于 Lora-Adapter 文件处理的功能请求**：一位用户表示需要一种不需要 VRAM 的 Unsloth Lora-Adapter 文件转换流程。他们提到在以当前格式保存 Llama-3-70b 的约 7GB Adapter 时遇到了困难。
- **持久性 Bug 与更快的推理**：一位用户详细说明了一个导致持续记录日志的 Bug，但提到一旦修复，可能会带来轻微的性能提升。“一旦修复，你可能会发现推理速度略有提升，因为它不会在每次迭代时都向控制台打印内容了 😄”。
- **处理 CUDA 显存溢出（Out of Memory）问题**：另一位成员分享了使用 `torch.cuda.empty_cache()` 来处理 GPU 显存问题的方法。使用 lm_head 进行推理时消耗的显存超过预期，导致了 CUDA 显存溢出错误。
- **运行 GGUF 模型**：讨论了使用 llama-cpp-python 运行 GGUF 模型的问题，以及 Transformers 缺乏直接运行 GGUF 支持的情况。另一位用户建议通过 llama.cpp 直接运行 GGUF 二进制文件。
- **RAG 系统困惑**：关于 Mistral AI 是否提供 RAG 系统存在困惑；经澄清，虽然 Mistral 本身不提供 RAG，但有[实现它的文档](https://docs.mistral.ai/guides/rag/)。
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 微调 Llama 3, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://docs.mistral.ai/guides/rag/">Basic RAG | Mistral AI Large Language Models</a>: 检索增强生成 (RAG) 是一种 AI 框架，它协同了 LLM 和信息检索系统的能力。它对于回答问题或利用...生成内容非常有用。</li><li><a href="https://techcommunity.microsoft.com/t5/microsoft-developer-community/doing-rag-vector-search-is-not-enough/ba-p/4161073">Doing RAG? Vector search is *not* enough</a>: 如果你在为 AI 应用使用 RAG (Retrieval-Augmented Generation)，那么你应该确保你做的不仅仅是向量搜索。</li><li><a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=shar">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing,">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1248303622333272154)** (2 messages): 

- **加入新的 AI 微调服务器**：由一班 AI 学生发起的新 AI 社区服务器欢迎对 AI 微调感兴趣的新成员加入。通过[此链接](https://discord.gg/sTtpXzJzTb)查看该频道，寻找志同道合的伙伴和资源。

**提到的链接**：<a href="https://discord.gg/sTtpXzJzTb">加入 VirtualValleyAI Discord 服务器！</a>：查看 Discord 上的 VirtualValleyAI 社区 —— 与其他 72 名成员一起交流，享受免费的语音和文字聊天。

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1247990143034593433)** (180 条消息🔥🔥): 

- **Stable Audio Open 1.0 引起关注**：几位成员讨论了 [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) 的可用性和特性，包括其组件如 autoencoder 和基于 Transformer 的 diffusion 模型。一位成员提到在自定义音频生成工作流中使用 tacotron2、musicgen 和 audiogen。
  
- **Stable Diffusion 图像质量困扰**：一位用户报告了在 Stable Diffusion 中生成低分辨率图像（160x90）时出现随机颜色的问题。其他成员建议先生成较大尺寸的图像（例如 512x512 或 1024x1024），然后使用任何图像编辑器进行缩小。
  
- **ControlNet 的使用与疑问**：一位成员询问如何使用 ControlNet 将手绘草图转换为写实图像，因为他们目前的 image-to-image 方法会保留不需要的白色。其他成员建议使用 ControlNet 以更好地控制生成图像的构图和姿势。

- **CivitAI 的过滤担忧**：一条消息强调了 CivitAI 需要增加额外的过滤器，因为 OnlyFans 和 TikToker 等无关内容激增。这被认为增加了寻找高质量模型的难度。

- **Stable Diffusion 3 的推测与误传**：多位用户辩论了 Stable Diffusion 3 的发布日期和真实性，一些人对即将发布充满信心，而另一些人则持怀疑态度。有人提供了澄清，指向了详细说明预期规格和日期的 [Reddit 帖子](https://www.reddit.com/r/StableDiffusion/comments/1d6ya9w/collection_of_questions_and_answers_about_sd3_and/) 等来源。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/stabilityai/stable-audio-open-1.0">stabilityai/stable-audio-open-1.0 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d6ya9w/collection_of_questions_and_answers_about_sd3_and/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=S7rGtWvEOV4">在 ComfyUI 中利用 Stable Diffusion 和 COSXL 的力量更改物体材质</a>: 播放列表: https://www.youtube.com/playlist?list=PLepQO73yVqJYDTnVVdu9LiNtAaTYLsxmKMy Patreon: https://www.patreon.com/ArchAi3D---------------------------欢迎...</li><li><a href="https://imgur.com/a/Xxaj8FG">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、热门迷因、娱乐性的 gif、感人的故事、病毒视频等来振奋精神...</li><li><a href="https://github.com/jitcoder/lora-info">GitHub - jitcoder/lora-info</a>: 通过创建账户为 jitcoder/lora-info 的开发做出贡献。</li><li><a href="https://github.com/THUDM/CogVLM">GitHub - THUDM/CogVLM: a state-of-the-art-level open visual language model | 多模态预训练模型</a>: a state-of-the-art-level open visual language model | 多模态预训练模型 - THUDM/CogVLM</li><li><a href="https://stability.ai/stable-artisan#choose-stable-artisan-plan.">Stable Artisan &mdash; Stability AI</a>: Stable Artisan 是一款有趣的、在 Discord 生态系统内利用 Stability AI 平台 API 产品的多模态生成式 AI Discord 机器人。</li><li><a href="https://civitai.com/images/10895925">KandooAI 发布的图像</a>: 未找到描述</li><li><a href="https://civitai.com/models/133005/juggernaut-xl">Juggernaut XL - Jugg_X_RunDiffusion_Hyper | Stable Diffusion Checkpoint | Civitai</a>: 商务咨询、商业授权、定制模型及咨询，请通过 juggernaut@rundiffusion.com 联系我。现在就在 X 加入 Juggernaut/...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1247996442573013023)** (64 messages🔥🔥): 

- **关于在 LM Studio 中限制 RAM 使用的讨论**：一位用户询问如何限制模型的 RAM 使用量。讨论明确指出 LM Studio 目前没有内置此功能，但可以参考 [llamacpp documentation](https://example.com) 中描述的方法，通过在使用时加载和卸载来管理。尽管这种方式效率较低，但可以让模型仅在激活时占用 RAM。

- **使用 iMat 提升量化模型质量**：讨论涉及了使用 iMat 提高量化模型质量的可行性。目前 LM Studio 尚不支持此功能，除非 llamacpp 引入该能力。

- **为高 VRAM 系统选择 AI 模型**：一位拥有 160GB VRAM 系统的用户寻求合适的 AI 模型推荐。讨论建议参考 [LLM Extractum.io](https://llm.extractum.io/list/)，该网站提供了一个可按大小和质量过滤的完整列表。

- **当前 LM Studio 版本的错误**：对话涵盖了用户遇到的各种错误，提供的建议包括回滚到旧版本或调整上下文设置，例如 `n_ctx` 可能设置得过高，超出了可用 VRAM 的承受范围。

- **支持 PDF 转文本转换**：对于寻求将 PDF 转换为文本以进行摘要的用户，建议使用 [pdftotext from XpdfReader](https://www.xpdfreader.com/download.html) 等工具，并强调该工具提供 Linux 和 Windows 命令行版本。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llm.extractum.io/list/?query=uncensored">"uncensored" 搜索结果</a>：在我们的 LLM Explorer 目录中，针对 3b、13b、30b 和 70b 大小开源语言模型的 "uncensored" 查询排名最高的匹配结果。</li><li><a href="https://llm.extractum.io/list/">所有大语言模型</a>：大语言模型和小语言模型（开源 LLM 和 SLM）的精选列表。所有大语言模型均支持动态排序和过滤。</li><li><a href="https://www.xpdfreader.com/download.html">下载 Xpdf 和 XpdfReader</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1247989302428958820)** (78 messages🔥🔥): 

- **量化会导致模型出现细微差异**：讨论者一致认为，由于使用的训练数据集不同，创建量化模型会导致 Token 概率出现细微差异。一位用户总结道：*“根据所使用的训练集，Token 概率会存在（极其微妙的）差异。”*

- **Nomic Embed 模型集成多模态能力**：[nomic-embed-vision-v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5) 因与其文本版本共享嵌入空间而受到关注，这使得 **所有 Nomic Embed Text 模型都具备了多模态能力**。性能统计显示，在某些基准测试中，它的表现优于 OpenAI 的 CLIP ViT B/16 等模型。

- **Llama-3 MahouDevil 量化模型讨论**：对话集中在 Q6 和 Q8 等量化版本在 RP（角色扮演）和通用用途中的可用性。指出 **推荐使用 Q6 以获得最高质量和最佳性能的平衡**。

- **Jina AI 推出多模态嵌入模型**：用户指出 Jina CLIP 是多模态（文本-图像）嵌入模型领域的新成员，可在 [Huggingface](https://huggingface.co/jinaai/jina-clip-v1) 上获取。这顺应了嵌入模型日益增强多模态支持的趋势。

- **在 llama.cpp 中发现 MacOS 的 Metal 内存问题**：对高上下文参数下 Metal 内存分配问题的深入研究表明，**llama.cpp 的 Metal 支持在最近的版本中已损坏**。用户建议坚持使用 b3066 版本以保持稳定，因为像 b3091 这样的较新版本引入了 Bug。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/jinaai/jina-clip-v1">jinaai/jina-clip-v1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mradermacher/PsyfighterTwo-ErebusThree-SlerpThree-GGUF">mradermacher/PsyfighterTwo-ErebusThree-SlerpThree-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5">nomic-ai/nomic-embed-vision-v1.5 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF">YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1248226152875687947)** (3 条消息): 

- **LM Studio 缺失 mmap 标志**：一位成员强调了 `--no-mmap` 选项对于 8GB RAM 的 PC 的好处，并提议在 LM Studio 中增加一个切换开关以便于配置。他们报告称该选项可以防止 RAM 峰值，降低在运行 8B 模型操作时发生冻结的风险，代价是初始模型加载时间会略微增加。
- **LM Studio 中的 Mlock 设置**：另一位成员澄清了围绕 `--no-mmap` 的初步讨论，提到了 LM Studio 中类似的 `use_mlock` 设置。他们建议探索其效果，并指出这存在依赖于 OS 的细微差别，同时要求澄清正在讨论的是哪款软件。
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1248304885552775259)** (3 条消息): 

- **开启新对话时的设置重置问题**：一位用户报告称，在开始新对话时模型设置会被重置。另一位成员建议在 "My Models" 选项卡的下拉菜单中应用设置，该用户随后发现，在创建新对话之前不删除旧对话可以防止设置重置。
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1247990767461601290)** (26 条消息🔥): 

- **Nvidia vs AMD 驱动**：一位成员指出，他们希望 Nvidia 能像 AMD 一样**开源其驱动程序**。这引发了关于系统管理和安全策略的讨论。

- **Windows 安全实践遭到批评**：成员们讨论了 **Windows 在商业环境中的安全缺陷**。一位成员指出：*"使用 Windows... 不是因为它是个好方案，而是因为它是默认选择，"* 并强调了小企业中 IT 支持的局限性。

- **Qualcomm 新芯片看起来很有前景**：尽管对 ARM 处理器和 Microsoft 的处理方式有所担忧，但人们对**高通（Qualcomm）新芯片**持乐观态度。一位成员指出：*"对于如此巨大的转变，高通的新芯片看起来相当令人印象深刻。"*

- **针对 LLAMA 3 405B 的硬件升级**：一位成员正在使用新的 GPU 配置升级他们的 PC，并**考虑如果 LLAMA 3 405B 表现出色，则将 RAM 扩展到 128GB**。他们分享道：*"除非 LLAMA 3 405B 真的很吸引人，否则我可能不会扩展到 128GB RAM。"*

- **ARM 与 x86 性能警示**：尽管合成基准测试（synthetic benchmarks）表现出色，但仍有建议对 ARM CPU 与 x86 的**实际性能对比**保持谨慎。一位成员警告说：*"该芯片很可能只是针对合成负载进行了优化... 而在其他方面表现糟糕。"*

**提到的链接**：<a href="https://www.youtube.com/watch?v=PGjdN_qfqgg">The Story of Snapdragon X Elite</a>：两起诉讼与一个谜团：Snapdragon X Elite 的故事 | 在本视频中，我们将了解高通新款 Arm SoC 的精彩历史，其目标是...

  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1248355433865941114)** (1 条消息): 

- **Higgs LLAMA 模型获得赞誉**：一位成员指出，新的 **Higgs LLAMA 模型**在其 70B 的体量下“看起来很聪明”。他们正在等待 **LM Studio** 的更新，因为它似乎利用了 **llamacpp** 的调整。
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1247990662029250621)** (120 条消息🔥🔥): 

- **社区讨论适当的行为和审核**：包括 *cakiki* 和 *lunarflu* 在内的成员讨论了在 DM（私信）和线程中举报不当行为的问题。记录了合理的担忧，但强调了保持专业精神的重要性。

- **Gradio 集成咨询**：一位成员询问了如何将 Gradio 与 React Native 和 Node.js 集成。*pseudoterminalx* 指出它是用 Svelte 构建的，另一位成员建议检查 Gradio API 的相关 issue。

- **使用 Stable Diffusion 模型生成文本**：*temperance6095* 询问了能够生成文本的 Stable Diffusion 模型，随后收到了 AnyText 或微软的 TextDiffuser-2 等推荐。

- **社区资助和项目审批**：成员们讨论了 HuggingFace Spaces 社区资助的审批流程和时间。大家注意到，独特的项目更有可能更快获得批准。

- **对点对点计算（Peer-to-peer compute）的兴趣**：成员们讨论了对点对点计算的经验和好奇心，提到了用于分布式机器学习的 Petals 等工具。*geekboyboss* 分享了出于隐私原因使用本地 swarm 的积极体验。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.gradio.app/">Gradio</a>: 构建并分享令人愉悦的机器学习应用</li><li><a href="https://tenor.com/view/discord-gif-27442765">Discord GIF - Discord - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/qgNkIp2pvoz.gif">辛普森一家 荷马·辛普森 GIF - 辛普森一家 荷马·辛普森 - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

qasim_30: 有一篇论文叫 "7 billion is all you need"
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1248296862817976340)** (4 条消息): 

- **Q-Learning Agent 玩转 Taxi-v3**：一位用户分享了一个用于 Taxi-v3 环境的 [Q-Learning Agent](https://huggingface.co/DAIEF/q-learning-Taxi-v3)，讨论了利用该模型创建高效且具有环境意识的配送系统的潜力。该用户强调了在初始化环境时检查并添加额外属性（如 `is_slippery=False`）的重要性。
  
- **寻求基于 LLM 的测试用例生成指导**：一位成员寻求关于使用 LLM 模型理解现有代码库并生成自动化测试用例的产品构建指导。另一位成员表达了合作意向，提到了自己作为高级 Web 开发人员以及 AI 和 LLM 初学者的背景，并鼓励其他有兴趣合作的人联系。

**提到的链接**：<a href="https://huggingface.co/DAIEF/q-learning-Taxi-v3">DAIEF/q-learning-Taxi-v3 · Hugging Face</a>: 未找到描述

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1248018846246240369)** (12 messages🔥): 

- **针对金融投资的气候意识 AI 助手**：利用 `climatebert/tcfd_recommendation` 模型，一位用户开发了一个 AI 助手来帮助寻找以气候为导向的投资解决方案。他们使用了 Qdrant Cloud 和 `microsoft/Phi-3-mini-128k-instruct`，并在 [HuggingFace](https://huggingface.co/spaces/as-cle-bert/tcfd_counselor) 上分享了该项目。

- **SimpleTuner 新增 Mixture-of-Experts 支持**：SimpleTuner 的最新版本 v0.9.6.2 包含了 Mixture-of-Experts 分离时间步训练。GitHub 上提供了一个教程，帮助用户开始使用这一新功能：[GitHub](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.6.2)。

- **Triton 教程及内容匮乏问题**：一位成员表达了在理解 Triton 教程方面的困难，并指出关于 Triton 的内容非常稀缺。他们分享了一篇 [Medium 文章](https://medium.com/@isamu-website/understanding-triton-tutorials-part-2-f6839ce50ae7)，认为其很有帮助但仍具挑战性。

- **发布真正的 Multi-agent 系统**：一位用户正在开发用于运行真正 Multi-agent 系统的 SDK 和计算服务器，这与基于单个 LLM 的 Agent 有所不同。他们引用了一场 Twitter 讨论，并邀请有兴趣的人加入他们即将建立的 Discord 社区（[详情点击此处](https://x.com/yoheinakajima/status/1781183534998380576)）。

- **FluentlyXL 最终版发布**：FluentlyXL 模型系列的最终版本已发布，在美学和光影方面进行了改进。分享了该模型在 [HuggingFace](https://huggingface.co/fluently/Fluently-XL-Final)、[CivitAI](https://civitai.com/models/324891) 上的链接以及一个 [Playground](https://huggingface.co/spaces/fluently/Fluently-Playground)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/DAIEF/q-learning-Taxi-v3">DAIEF/q-learning-Taxi-v3 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/as-cle-bert/tcfd_counselor">Tcfd Counselor - as-cle-bert 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/as-cle-bert/carbon-footprint-predictor">Carbon Footprint Predictor - as-cle-bert 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/yoheinakajima/status/1781183534998380576">Yohei (@yoheinakajima) 的推文</a>：我倾向于“戴上不同的帽子”来思考同一个问题。对我来说，这感觉类似于一个使用不同 Prompt 对同一个 LLM 进行调用的单一代码库。同样，这只是语义上的问题，所以...</li><li><a href="https://github.com/NoteDance/Note">GitHub - NoteDance/Note: 轻松实现并行训练和分布式训练。机器学习库。Note.neuralnetwork.tf 包包含 Llama2, Llama3, Gemma, CLIP, ViT, ConvNeXt, BEiT, Swin Transformer, Segformer 等，这些使用 Note 构建的模型与 TensorFlow 兼容，并可以使用 TensorFlow 进行训练。</a></li><li><a href="https://github.com/bghira/SimpleTuner/releases/tag/v0.9.6.2">Release v0.9.6.2 mixture-of-experts training · bghira/SimpleTuner</a>：更新内容：Mixture-of-Experts 训练，附带关于如何加速训练并开始产生惊人结果的简短教程。DeepSpeed 修复 (#424)...</li><li><a href="https://github.com/InServiceOfX/InServiceOfX/blob/master/PythonLibraries/HuggingFace/MoreDiffusers/morediffusers/Applications/terminal_only_finite_loop_main_with_loras.py">InServiceOfX/PythonLibraries/HuggingFace/MoreDiffusers/morediffusers/Applications/terminal_only_finite_loop_main_with_loras.py at master · InServiceOfX/InServiceOfX</a>：用于深度学习的 Monorepo（单一或 "mono" 仓库）。 - InServiceOfX/InServiceOfX</li><li><a href="https://github.com/InServiceOfX/InServiceOfX/blob/master/PythonLibraries/ThirdParties/MoreInstantID/moreinstantid/Applications/terminal_only_finite_loop_main_with_loras.py">InServiceOfX/PythonLibraries/ThirdParties/MoreInstantID/moreinstantid/Applications/terminal_only_finite_loop_main_with_loras.py at master · InServiceOfX/InServiceOfX</a>：用于深度学习的 Monorepo（单一或 "mono" 仓库）。 - InServiceOfX/InServiceOfX</li><li><a href="https://www.instagram.com/p/C6wP_q-rwIS/?igsh=MWQ1ZGUxMzBkMA==">Instagram 上的 Mansion X："出发去 slay #ootd #ootdfashion Maude Mongeau for @the_mansion_x"</a>：3 个赞，1 条评论 - the_mansion_x 于 2024 年 5 月 9 日："出发去 slay #ootd #ootdfashion Maude Mongeau for @the_mansion_x"。 
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1247990833571958824)** (4 条消息): 

- **不要错过 Human Feedback Foundation 活动**：一名成员强调了即将由 [Human Feedback Foundation 在 6 月 11 日举办的活动](https://www.eventbrite.ca/event/851921368747?aff=oddtdtcreator)。该基金会旨在将人类反馈整合到 AI 中，重点关注医疗保健和治理等关键领域。
- **Human Feedback Foundation YouTube 存档已上线**：Human Feedback Foundation 在其 [YouTube 频道](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg)上提供了往期会议记录。他们的演讲者来自 UofT、Stanford 和 OpenAI 等机构，旨在开展 AI 安全研究教育，并通过开源倡议促进公众参与 AI。
- **跟上 HuggingFace 阅读小组进度**：一位新成员询问是否有阅读小组录像的板块。另一位成员给出了肯定的回答，并引导他们前往一个 [GitHub 仓库](https://github.com/isamu-isozaki/huggingface-reading-group)，该仓库汇集了 HuggingFace 阅读小组过去所有的演示文稿。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg">Human Feedback Foundation</a>：Human Feedback Foundation 的使命是将人类反馈构建到开源 AI 项目中。我们寻求：通过支持开源开发和政策倡议，实现公众对 AI 的投入...</li><li><a href="https://github.com/isamu-isozaki/huggingface-reading-group">GitHub - isamu-isozaki/huggingface-reading-group: 该仓库的目标是预编译 Huggingface 阅读小组过去所有的演示文稿</a>：该仓库的目标是预编译 Huggingface 阅读小组过去所有的演示文稿 - isamu-isozaki/huggingface-reading-group</li><li><a href="https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator">LLM 阅读小组（3 月 5 日、19 日；4 月 2 日、16 日、30 日；5 月 14 日、28 日；6 月 11 日）</a>：来见见 LLM/NLP 研究领域一些开创性论文的作者，听听他们讲述自己的工作。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1248120463008202803)** (10 条消息🔥): 

- **小型数据集导致验证问题**：用户讨论了非代表性的验证集通常意味着需要更多样化的训练数据。一位用户建议，即使样本少于 4,000 个，如果包含其他类别以防止误报（false positives），像 EfficientNet V2 这样的模型也能表现良好。

- **需要更多项目细节**：社区成员询问了有关数据集问题的更多细节，包括类别数量和误报类型，以便提供更好的帮助。一位用户表示愿意通过私信提供个人协助。

- **Transformer 与传统模型的辩论**：一位用户指出，虽然计算机视觉中的 Transformer 可以更好地处理数据质量问题，但与 YOLO 和 EfficientNet V2 等模型相比，它们显著增加了训练时间。另一位成员表示赞同，并补充说 Transformer 的效率还取决于数据的大小。

- **结合音频和视频帧进行流媒体传输**：一位用户询问了在通过 WebRTC 或 RTMP 进行流媒体传输时，将音频帧与生成的 24 FPS 视频帧同步的可行性。他们寻求在不损失 FPS 的情况下实现这一目标的建议或资源。

- **关于 CIFAR 的 Swin Transformer 讨论**：一位用户询问是否有人为 CIFAR 数据集实现过 Swin Transformer (tiny)，但该话题随后没有进一步讨论。
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1248062772877590629)** (1 条消息): 

- **低 Temperature 意味着确定性模型**：建议尝试将 **Temperature 降低到 0.1** 以获得更具确定性的模型。*“Temperature 越低，模型越具有确定性。”*

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1247993513929605131)** (7 条消息): 

- **关于文本嵌入 (text embeddings) 的困惑**：一位成员提到在识别 *text-enc 1 为 768 且 text-enc 2 为 1280* 时感到挣扎。他们还在如何于示例输入中正确包含 *text_embeds 和 time_ids* 方面遇到了麻烦。
- **对 added kwargs 的批评**：另一位成员对 **added kwargs** 被放在字典中表示沮丧，认为这使得 *"更难追踪需要哪些输入。"*
- **重参数化 Segmind 模型**：一位成员将 Segmind **ssd-1b** 重参数化为一个在 350 个 timesteps 上训练的 v-prediction/zsnr refiner 模型。他们对仅经过约 800 步 tuning 就能如此快速地奏效感到惊讶。
- **最喜欢的模型及未来计划**：同一位成员宣布它是他们 *"最喜欢的新模型。"* 他们计划使用前 650 个 timesteps 从 ssd-1b 训练另一个 checkpoint，以创建一个真正的 1B Mixture of Experts。
- **对光影的称赞**：一段简短的交流，其中一位成员称赞了光影效果，另一位成员表示感谢。

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1247994873789808660)** (104 条消息🔥🔥): 

- **KANs 被过度炒作且效率低下**：成员们讨论了 Kolmogorov-Arnold Networks (KANs) 相比传统神经网络的局限性和低效，特别是在大规模模型方面。有人指出：“KAN 对可解释性有用纯属炒作，它行不通。”

- **KANs 的高效实现**：人们对提高 KANs 效率的实现方式很感兴趣，特别是使用 CUDA 和 ReLU 等替代方案。一位成员分享了一篇提出 ReLU-KAN 架构的[论文](https://arxiv.org/abs/2406.02075)，该架构实现了显著的加速。

- **数据选择技术**：成员们讨论了在不进行完整重新训练的情况下评估数据质量的各种方法。**Influence functions** 的概念引发了广泛辩论，许多人认为与手动和自动数据清洗技术相比，它们难以扩展。提到的一个关键资源是 [LESS 算法](https://www.cs.princeton.edu/~smalladi/blog/2024/04/04/dataselection/)。

- **训练数据与模型之间的相互作用**：对话集中在大型模型（如基于 Transformer 的系统）如何平衡数据多样性与数据质量之间的权衡。有人指出，更大的模型可以处理更多的“杂质（crud）”，并且需要多样化的训练数据以实现更好的世界建模。

- **Numenta 的 Thousand Brains 项目**：该项目被简要提及，重点是应用神经科学原理开发一种新型 AI。详细信息可在 [Numenta 网站](https://www.numenta.com/thousand-brains-project/)上找到。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org">arXiv.org 电子预印本存档</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2406.02075">ReLU-KAN: New Kolmogorov-Arnold Networks that Only Need Matrix Addition, Dot Multiplication, and ReLU</a>：受限于基函数 (B-spline) 计算的复杂性，Kolmogorov-Arnold Networks (KAN) 在 GPU 上的并行计算能力受限。本文提出了一种新型 ReLU-KAN...</li><li><a href="https://arxiv.org/abs/2405.03875">Rethinking Data Shapley for Data Selection Tasks: Misleads and Merits</a>：Data Shapley 为数据估值提供了一种原则性方法，在以数据为中心的机器学习 (ML) 研究中起着至关重要作用。数据选择被认为是 Data Shapley 的标准应用...</li><li><a href="https://arxiv.org/abs/2401.12926">DsDm: Model-Aware Dataset Selection with Datamodels</a>：在为训练大规模模型选择数据时，标准做法是过滤符合人类数据质量观念的样本。这种过滤会产生定性上干净的数据点，但...</li><li><a href="https://arxiv.org/abs/2211.08411">Large Language Models Struggle to Learn Long-Tail Knowledge</a>：互联网包含丰富的知识——从历史人物的生日到编程教程——所有这些都可能被语言模型学习。然而，虽然某些片段...</li><li><a href="https://arxiv.org/abs/2404.04125?">No &#34;Zero-Shot&#34; Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance</a>：网络爬取的预训练数据集是多模态模型令人印象深刻的“零样本”评估性能的基础，例如用于分类/检索的 CLIP 和用于图像生成的 Stable-Diffusion...</li><li><a href="https://www.johndcook.com/blog/2020/01/04/sufficient-statistic-paradox/">Persi Diaconis&#039; sufficient statistic paradox</a>：为了有用，统计数据必须提供将大量数据浓缩为少数人类可解释数字的方法。KPD 定理表明这是不可能的。</li><li><a href="https://www.cs.princeton.edu/~smalladi/blog/2024/04/04/dataselection/">Using LESS Data to Tune Models</a>：未找到描述</li><li><a href="https://www.numenta.com/thousand-brains-project/">Thousand Brains Project | Numenta</a>：Thousand Brains 项目是一个开源倡议，致力于根据 Thousand Brains 理论创建一种新型人工智能。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1247990401491668994)** (21 条消息🔥): 

- **Nvidia 发布开放权重变体**：Nvidia 发布了其模型的 8B 和 48B 变体的开放权重版本，可在 [GitHub](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro) 上获取。链接页面提供了关于大规模训练 Transformer 模型的持续研究。

- **AI 历史学家使用 ChatGPT 进行档案工作**：历史学家 Mark Humphries 发现 AI（特别是 GPT 模型）可以显著辅助历史文献的转录和翻译，并最终创建了一个名为 HistoryPearl 的系统。该系统在转录文档的速度和成本方面优于人类研究生（[The Verge 文章](https://www.theverge.com/24068716/ai-historians-academia-llm-chatgpt)）。

- **无矩阵乘法（MatMul-free）模型展现潜力**：一篇新论文（[arXiv](https://arxiv.org/abs/2406.02528)）介绍了一种 MatMul-free 模型，该模型在保持十亿参数规模强大性能的同时，显著降低了训练期间的内存使用。这些模型实现了与传统 Transformer 相当的性能，但内存效率更高。

- **用于高效长序列训练的 Seq1F1B**：另一篇论文（[arXiv](https://arxiv.org/abs/2406.03488)）提出了 Seq1F1B，这是一种流水线调度方法，旨在提高 LLM 在长序列上的内存效率和训练吞吐量。该方法减少了流水线气泡（pipeline bubbles）和内存占用，增强了可扩展性。

- **LLM 的 QJL 量化方法**：在一项近期研究（[arXiv](https://arxiv.org/abs/2406.03482)）中详细介绍的 QJL 方法，应用了 Johnson-Lindenstrauss 变换，随后进行符号位量化，以消除存储 KV embeddings 的内存开销。这种方法在不损失性能的情况下显著压缩了 KV cache 的需求。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.theverge.com/24068716/ai-historians-academia-llm-chatgpt">AI 能为历史学家做些什么</a>：事实证明，大语言模型是历史学家出奇好用的研究助手。AI 的未来能否帮助重建过去？</li><li><a href="https://arxiv.org/abs/2406.03482">QJL: 用于零开销 KV Cache 量化的 1-Bit 量化 JL 变换</a>：由于 KV cache 中 Key-Value (KV) embeddings 的存储需求随序列长度增长，服务 LLM 需要大量内存。压缩 KV cache 的一种有效方法是 q...</li><li><a href="https://arxiv.org/abs/2406.03488">Seq1F1B: 用于大语言模型训练的高效序列级流水线并行</a>：大语言模型 (LLMs) 的出现严重依赖于分布式训练策略，其中流水线并行起着至关重要的作用。随着 LLMs 的训练序列长度延伸到...</li><li><a href="https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro">InstructRetro 分支下的 Megatron-LM/tools/retro · NVIDIA/Megatron-LM</a>：关于大规模训练 Transformer 模型的持续研究 - NVIDIA/Megatron-LM</li><li><a href="https://arxiv.org/abs/2406.02528">可扩展的无矩阵乘法（MatMul-free）语言建模</a>：矩阵乘法 (MatMul) 通常占据了大语言模型 (LLMs) 的整体计算成本。随着 LLMs 扩展到更大的嵌入维度和上下文长度，这一成本只会不断增加...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1248258451734401108)** (3 条消息): 

- **Llama-3 的 logprobs 错误揭晓**：一位成员在尝试请求超过 5 个限制的 logprobs 时遇到了来自 **Llama-3** 的 *"Value error"*。另一位成员建议，潜在的解决方案包括对 harness 进行硬编码，以使用有效范围内的值。 
- **询问 Batch API 集成到 harness 的情况**：一位成员询问了将 **OpenAI 的 Batch API** 添加到 harness 中的可能性，并引用了 [platform.openai.com 文档](https://platform.openai.com/docs/guides/batch)。该询问似乎是针对特定成员关于未来计划或现有实现的。
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1248082602678681734)** (1 条消息): 

- **将大脑数据与 Whisper embeddings 对齐**：一位成员正致力于将 **语音嵌入（speech embeddings）** 与脑植入神经数据对齐以解码语音，使用的是 **Whisper tiny.en 模型** 的修改版本。他们正在寻求关于解锁哪些层、尝试额外的损失函数、调整超参数以及在仅有一块 GPU 的情况下加速或并行化训练过程的反馈，并对合作持开放态度。

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1248006736468512900)** (111 条消息🔥🔥): 

- **Perplexity Pro 新增功能**：Perplexity Pro 现在会逐步显示其搜索过程，被描述为使用 intent system 以实现更具 agentic-like 的执行。成员们注意到这一变化大约是在一周前上线的。
- **Perplexity 读取文件的问题**：多位用户报告称，尽管 PDF 是允许的文件类型，但 Perplexity 在读取 PDF 文件时遇到困难或失败。一些人建议该问题可能与内容类型有关，例如重度排版与纯文本 PDF 的区别。
- **MVP 项目的预算冲击**：一名用户寻求开发者以 100 美元的预算构建一个从文本生成视频的 MVP，这引发了关于低预算和典型开发者费率的幽默及批评性反应。
- **特定 labs 功能的停用**：成员们讨论了从 Perplexity labs 中移除 Haiku 和其他功能的举动，推测这可能是出于成本节约的原因，并指出这些功能的缺失影响了他们的使用。
- **关于 Perplexity 未来功能的查询**：用户询问了在 iOS app 中编辑 collections 以及在 Perplexity 上创建 pages 的功能，但目前这些功能仅限特定用户使用或处于 beta 阶段。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/models/google/paligemma-ft">Google | PaliGemma-ft | Kaggle</a>: 在广泛的研究数据集上进行 fine-tuned 的 PaliGemma。</li><li><a href="https://ai.google.dev/gemma/docs/paligemma">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=hkHCnZ5GrNc">I Built a Car out of Scooters</a>: 感谢 Odoo 赞助本视频！https://www.odoo.com/r/UIO 使用优惠码 pissbaby 购买 Opensauce 门票：https://opensauce.com/tickets/ Tommy: https://ww...</li><li><a href="https://www.perplexity.ai/search/Zdravo-Zam-moram-foAW4rklQI6LCkCsbBfNKg>">Perplexity</a>: 未找到描述</li><li><a href="https://perplexity.ai/page/new">Perplexity</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1248014547755860110)** (5 messages): 

- **Bitcoin 的历史突显了其发展历程**：一条信息化消息分享了 **Bitcoin** 的历史，强调了 **Satoshi Nakamoto** 在 2009 年的创立及其对金融界的重大影响。详细概览和历史链接可以在[这里](https://www.perplexity.ai/page/Bitcoin-WzdgAV4KQiqtb4k0q4RHRw)找到。

- **探索 Perplexity AI 的功能**：Perplexity AI 将对话式搜索与直接链接相结合，在提供答案的同时引用来源。主要功能包括基于聊天的搜索、多轮对话和多语言支持，详见[此处](https://www.perplexity.ai/search/Perplexity-AI-gOX8DHIdR7SzpZn6qI8s6Q)。

- **Perplexity AI 访问付费墙内容**：讨论了 Perplexity AI 访问付费墙后内容的能力。该工具提供多种功能，并提供 Pro 升级以增强功能，如[此处](https://www.perplexity.ai/search/please-list-the-bU9PEgabRCGjI1hVl4mQaw)所述。

- **Revit 2024 增强 PDF 导出**：Revit 2024 包含原生 PDF 导出功能，简化了 BIM 建模师和电气工程师的工作流程，无需依赖外部 PDF 打印机。更多详情见[此处](https://www.perplexity.ai/page/PDF-Export-in-b.a.ByBkSgSpRnzjc_cDNA)。

- **React 和 JS 中的 `navigator.userAgent` 输出**：解释了 React 和 JavaScript 中的 `navigator.userAgent`，详细说明了它如何返回标识用户浏览器和操作系统的字符串。示例和更多信息见[此处](https://www.perplexity.ai/search/in-react-and-h.wr3aykTeOqfFDgfG57xQ)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/page/Bitcoin-WzdgAV4KQiqtb4k0q4RHRw">Bitcoin</a>: Bitcoin 是由 Satoshi Nakamoto 于 2009 年推出的首个也是最著名的加密货币。自那时起，Bitcoin 经历了显著的发展...</li><li><a href="https://www.perplexity.ai/search/please-list-the-bU9PEgabRCGjI1hVl4mQaw">Perplexity</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/search/in-react-and-h.wr3aykTeOqfFDgfG57xQ">在 react 和 javascript 中，navigator.userAgent 的可能输出是什么</a>: navigator.userAgent 属性返回一个识别用户浏览器和操作系统的字符串。其输出可能因浏览器而异...</li><li><a href="https://www.perplexity.ai/page/PDF-Export-in-b.a.ByBkSgSpRnzjc_cDNA">Revit 2024 中的 PDF 导出</a>: 将 Revit 文件导出为 PDF 格式是 BIM 建模师和电气规划师工作流程的核心部分。Revit 2024 提供...</li><li><a href="https://www.perplexity.ai/search/Perplexity-AI-gOX8DHIdR7SzpZn6qI8s6Q">什么是 Perplexity AI？</a>: Perplexity AI 是一种由人工智能驱动的会话搜索引擎，旨在通过自然语言处理技术解锁知识的力量，实现信息的发现和共享。它结合了对话和链接的搜索功能，能够识别和回复模糊或抽象的语言查询，模拟大部分人的语言询问方式。  1. 聊天对话搜索：用户可以像与真人对话一样，用自然语言提出问题，Perplexity AI...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1248244738604400710)** (1 messages): 

- **对添加 OpenChat 模型的关注**：一名成员询问是否有计划添加另一个 **"openchat/openchat-3.6-8b-20240522" 模型**。他们特别询问了是否会将其与现有的 **Mistral** 和 **Llama 3** 模型一起包含在内。
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1248093560046358600)** (3 messages): 

- **在 YouTube 上查找过往活动录像**：一位用户询问过往活动或讲座录像的位置。他们被引导查看相关的 Discord 频道和 [CUDA MODE YouTube channel](https://www.youtube.com/@CUDAMODE)。
  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1248025693804302419)** (2 条消息): 

- **检查 PyTorch 中的 Tensor 内存共享**：一名成员询问另一名成员，某段特定代码是否可以确认两个 **tensors 是否共享相同内存**，或者其中一个是副本。代码 "samestorage" 检查两个 tensors 的 `storage().data_ptr()` 是否相等，并打印 "same storage" 或 "different storage"。

- **查找函数源码的问题**：另一名成员表示在使用提供的文档链接定位 PyTorch 函数源码时遇到困难。他们引用了 [PyTorch C++ 文档中的特定函数](https://pytorch.org/cppdocs/api/function_namespaceat_1adeda9630914278ac02d7fd758da19e3d.html#exhale-function-namespaceat-1adeda9630914278ac02d7fd758da19e3d)。

**提到的链接**：<a href="https://pytorch.org/cppdocs/api/function_namespaceat_1adeda9630914278ac02d7fd758da19e3d.html#exhale-function-namespaceat-1adeda9630914278ac02d7fd758da19e3d">Function at::_weight_int4pack_mm &mdash; PyTorch main documentation</a>：未找到描述

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1248278098789662834)** (1 条消息): 

- **深入探讨 MoRA 以及 DPO 与 PPO 的辩论**：最新的研究包括 **MoRA**（**LoRA** 的增强版），以及用于 RLHF 的 **DPO** 与 **PPO** 的比较。在最新的 [AI Unplugged](https://datta0.substack.com/p/ai-unplugged-12-mora-dpo-vs-ppo-cope) 期刊中探索这些主题。
- **CoPE 引入 Contextual Position Encodings**：本周周报重点介绍了像 **CoPE** 这样用于优化位置编码的创新工作。通过阅读链接中的 [博客文章](https://datta0.substack.com/p/ai-unplugged-12-mora-dpo-vs-ppo-cope) 获取更多见解。
- **S3D 用于更快的推理**：最近的讨论包括 **S3D**，这是一种旨在加速推理的 self-speculative decoding 方法。鼓励大家阅读并分享对这些新技术的看法。

**提到的链接**：<a href="https://datta0.substack.com/p/ai-unplugged-12-mora-dpo-vs-ppo-cope">AI Unplugged 12: MoRA. DPO vs PPO. CoPE Contextual Position Encoding. S3D Self Speculative Decoding.</a>：洞察胜过信息

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1248244957760979061)** (22 条消息🔥): 

```html
- **KANs 通过 torch.compile 与 MLPs 媲美**：[Thomas Ahle 的一条推文](https://x.com/thomasahle/status/1798408687981297844) 强调了 torch.compile 如何使 KANs 变得与 MLPs 一样快，并赞扬了其性能提升。这引起了多位用户的关注和评论，他们对这一说法感到惊讶和印象深刻。
- **GitHub 上的仓库**：讨论中链接的 [GitHub 仓库](https://github.com/thomasahle/kanmlps) 提供了 KANs 和 MLPs 的资源。用户正在积极编译和分析这些实现，以了解性能优势。
- **实际的性能分析经验**：用户分享了他们在分析编译后的 KANs 时的经验和结果，注意到编译后速度提升了 1.5-2 倍。一位用户提到编译 `.forward` 函数带来了显著的速度提升。
- **对算子融合和 Kernel 的关注**：存在关于潜在缺点的技术讨论，例如失去算子融合（operator fusion）以及关于生成 Triton kernels 的问题。用户正在分析不同的实现以验证和比较结果，并引用了 [GitHub 上的特定代码位置](https://github.com/thomasahle/kanmlps/blob/main/models.py#L101)。
- **请求进一步合作**：有人建议邀请 Thomas Ahle 加入讨论，分享关于编译测试结果的见解。用户有兴趣确保这些实现与学术论文相符，并寻求验证输出。
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/thomasahle/status/1798408687981297844">Thomas Ahle (@thomasahle) 的推文</a>：使用 𝚝𝚘𝚛𝚌𝚑.𝚌𝚘𝚖𝚙𝚒𝚕𝚎 让 KANs 变得和 MLPs 一样快！我从未想过我会成为它的粉丝，但它们现在看起来非常诱人。</li><li><a href="https://github.com/thomasahle/kanmlps">GitHub - thomasahle/kanmlps: KANs and MLPs</a>：KANs 和 MLPs。通过创建账号为 thomasahle/kanmlps 的开发做出贡献。</li><li><a href="https://github.com/thomasahle/kanmlps/blob/main/models.py#L101">kanmlps/models.py at main · thomasahle/kanmlps</a>：KANs 和 MLPs。通过创建账号为 thomasahle/kanmlps 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/)** (1 条消息): 

piotr.mazurek: 第 4 章，练习 9，有人知道这是否是正确的解决方案吗？
  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1248322483367706654)** (1 条消息): 

- **Inductor 配置问题**：有一个关于 **torch._inductor.config.force_fuse_int_mm_with_mul** 设置的疑问。问题是该配置除了 **int8** 之外，是否也适用于 **uint8**。
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1248254069483638865)** (1 条消息): 

- **NetHack 中机器学习表现下降 40%**：一篇来自 [Ars Technica](https://arstechnica.com/gaming/2024/06/what-kind-of-bug-would-make-machine-learning-suddenly-40-worse-at-nethack/) 的分享文章讨论了一个奇特的 Bug，该 Bug 导致机器学习系统在 NetHack 游戏中的性能下降了 **40%**。据称该 Bug 是由天体原因引起的，使得这一场景既新颖又有趣。

**提到的链接**：<a href="https://arstechnica.com/gaming/2024/06/what-kind-of-bug-would-make-machine-learning-suddenly-40-worse-at-nethack/">什么样的 Bug 会让机器学习在 NetHack 中的表现突然下降 40%？</a>：有一天，一个玩 Roguelike 游戏的系统因为天体原因一直表现糟糕。

  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1248346826864857139)** (1 条消息): 

- **巴黎 AI_dev 引起关注**：有人询问是否有成员参加在巴黎举行的 AI_dev，并提到他们尚未注册但正在考虑中。他们分享了有关该活动的详细信息和链接，活动将于 2024 年 6 月 19 日至 20 日举行，并强调参加活动需要注册。


**提到的链接**：<a href="https://aideveu24.sched.com/?iframe=no">AI_dev Europe 2024 日程表</a>：查看 AI_dev Europe 2024 的日程安排

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1247988781349339186)** (52 条消息🔥): 

- **列主序的困扰**：成员终于理解了 **cublas** 及其列主序（column-major order）的复杂性，解释了为什么他们必须转置矩阵才能使用 **cublas** 正确计算 `Q @ K^T`。他们还提到在意识到这一点后撤回了关于“Attention Bug”的 PR。

- **统一内存分配提案**：讨论集中在将所有内存分配整合到单个函数中以提高效率和方便追踪的好处。正如 Erik 的草案 PR 和链接的 [master weights PR](https://github.com/karpathy/llm.c/pull/522/files#diff-bf6b442957e5458cf8baab2a18039fdde86d74199a0864a79e7288fe55f31a98R2982-R2995) 所指出的，这种方法将消除当前的重复并简化 Checkpointing。

- **基于 GPU CI 的 Checkpointing**：大家一致同意通过增强验证测试来改进 GPU 持续集成（CI），包括训练 Checkpointing 和输出对比。Erik 强调，虽然初步测试已经足够，但为了稳健的验证，未来的扩展是必要的。

- **Cublas vs Cutlass 及 C++ 要求**：澄清了带有 C 接口的 **cublas** 不需要 **C++17**，但 **cutlass** 需要。目前代码对 **C++17** 的要求仅限于 **cudnn**——这是未来开发考虑中的一个重要细节。

- **并行编程课程推荐**：推荐了 [Programing Parallel Computers 课程](https://ppc.cs.aalto.fi/) 及其 [练习](https://ppc-exercises.cs.aalto.fi/course/open2024a)，并提到可能会有一个设立新排行榜的夏季学期。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/522/files#diff-bf6b442957e5458cf">在恢复状态中添加 master weights，由 gordicaleksa 提交 · Pull Request #522 · karpathy/llm.c</a>：我们目前没有将 master weights 作为状态的一部分保存 -> 这会导致损失一些精度，因为否则在恢复时我们必须通过上采样来重建 master weights...</li><li><a href="https://github.com/karpathy/llm.c/pull/553/files#diff-d5e26abbb926892397df686a30886d861b1f45b627ce11070b72e0c9775edfa8R156">重构 trimat，由 gordicaleksa 提交 · Pull Request #553 · karpathy/llm.c</a>：确保我们的符号表示一致：仅对常量使用 (B,T,NH,HS) 大写。添加了额外的注释以澄清每个 Kernel 的作用，包括索引操作...</li><li><a href="https://github.com/karpathy/llm.c/pull/522/files#diff-bf6b442957e5458cf8baab2a18039fdde86d74199a0864a79e7288fe55f31a98R2982-R2995),">在恢复状态中添加 master weights，由 gordicaleksa 提交 · Pull Request #522 · karpathy/llm.c</a>：我们目前没有将 master weights 作为状态的一部分保存 -> 这会导致损失一些精度...
</li>
</ul>

</div>

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1248273951281512538)** (1 条消息): 

<!DOCTYPE html>
<html>
<body>
<ul>
  <li><strong>C++ 中的单例实例和 dtype 处理</strong>：一位成员认为 <em>“uint2-7 纯粹是为了命名，这更多是 C++ 的限制，而不是其他原因。”</em> 他们建议使用 <strong>bits8 作为无类型 dtype</strong>，并在必要时将其视为 unit8，从而实现更灵活的 Tensor 存储。</li>
</ul>
</body>
</html>
  

---


### **CUDA MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1248108040801095712)** (3 条消息): 

- **YouTube 讨论 ARM SME Dialect**：分享了一段题为 [“Open MLIR Meeting 06-22-2023: Targeting ARM SME from MLIR and SME Dialect”](https://www.youtube.com/watch?v=jrniGW_Hzno) 的视频，讨论了关于创建 ArmSME Dialect 的审查和 RFC。视频中包括了对 **ARM Scalable Matrix Extension** 的介绍。
- **Triton ARM 的潜力**：有人建议 Triton 的后端可能会支持 ARM，这表明 Triton ARM 的集成有着良好的发展前景。该信息链接到了 [MLIR 文档](https://mlir.llvm.org/docs/Dialects/ArmNeon/#arm_neonintrummla-arm_neonummlaop)中的 “arm_neon” dialect，其中讨论了多个 ARM NEON 操作。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=jrniGW_Hzno">Open MLIR Meeting 06-22-2023: Targeting ARM SME from MLIR and SME Dialect</a>：这是关于创建 ArmSME Dialect 的 RFC 的审查和讨论。我们将首先介绍 Arm 的 Scalable Matrix Extension，包括 t...</li><li><a href="https://mlir.llvm.org/docs/Dialects/ArmNeon/#arm_neonintrummla-arm_neonummlaop">'arm_neon' Dialect - MLIR</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1248280357455921242)** (1 条消息): 

- **文章涉及 PI Schilling 的担忧**：一位 GDM 机器人领域的人士认为作者的文章是在“为 PI 站台（schilling PI）”，并对其内容表示担忧。作者推测该人士可能“对 RTX 感到不满（salty）”，暗示其批评中带有个人偏见。
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1248017713150955631)** (33 条消息🔥): 

- **投资者追逐机器人 AI，规避硬件风险**：投资者正热衷于寻找“机器人领域的 ChatGPT”，寻求具有差异化优势的细分领域 Foundation Model 公司。[文章详情](https://www.newcomer.co/p/why-investors-cant-get-enough-of) 阐述了围绕这一趋势的热潮。
- **Qwen2 取得显著进展**：[Qwen2](http://qwenlm.github.io/blog/qwen2/) 推出了五种尺寸的模型，在多语言任务和扩展上下文支持（最高达 128K tokens）方面有所改进。目前已提供 Demo 和各种资源，尽管一些用户注意到其近期的知识存在局限性。
- **Qwen2 在实际测试中褒贬不一**：尝试 Qwen2 的用户评论了它在理解近期话题和提供准确通用知识方面的局限性。尽管存在一些缺点，但它在某些任务上的多语言表现受到了称赞。
- **Dragonfly 架构提升多模态 AI**：Together.ai 推出了 [Dragonfly](https://www.together.ai/blog/dragonfly-v1)，通过 Llama-3-8B-Dragonfly-v1 等模型增强了视觉理解和推理能力。这些模型显示出良好的前景，特别是在医学影像任务中。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.together.ai/blog/dragonfly-v1">Dragonfly: A large vision-language model with multi-resolution zoom</a>：未找到描述</li><li><a href="https://www.newcomer.co/p/why-investors-cant-get-enough-of">Why Investors Can't Get Enough of AI Robotics Deals Right Now </a>：风投们押注机器人技术是初创公司仍能对抗 OpenAI 的领域之一。</li><li><a href="http://qwenlm.github.io/blog/qwen2/">Hello Qwen2</a>：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 介绍 经过数月的努力，我们很高兴地宣布从 Qwen1.5 到 Qwen2 的进化。这一次，我们为您带来：预训练和指令...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1248142151309856819)** (12 messages🔥): 

- **Microsoft 承认未经授权进行 GPT-4 测试**：[Kevin Roose](https://x.com/kevinroose/status/1798414599152431278?s=46) 分享的一条推文透露，Microsoft 承认在最初否认后，曾在未获得联合安全委员会批准的情况下在印度测试了早期版本的 GPT-4。

- **警惕 Substack 套路**：Nathan Lambert 提醒不要轻信 Substack 的推荐，将其描述为旨在收集订阅者的骗局（grifts）。

- **Rick 刻意经营的社交媒体存在感**：一位用户评论了 Rick 虽然为人友善，但过度专注于通过 Twitter 和 LinkedIn 进行自我推销。

- **Substack 指标与挑战**：提到在 Substack 上获得 1,000 个订阅者非常具有挑战性，通常需要一两年的时间。Lambert 证实 Substack 推荐的点击量确实能推高数据。

**提到的链接**：<a href="https://x.com/kevinroose/status/1798414599152431278?s=46">来自 Kevin Roose (@kevinroose) 的推文</a>：OpenAI 吹哨人故事的有趣更新：在公开否认后，Microsoft 现在承认他们在没有联合安全委员会批准的情况下在印度测试了早期版本的 GPT-4...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1248312178281418762)** (4 messages): 

- **Google Vertex AI Gemini API 非常棒**：一位成员分享了 [Google Vertex AI Gemini 文档链接](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library)，并对其功能表示赞赏。特别强调了该 API 的易用性。
- **OpenAI 的 API 广受好评**：另一位成员称赞 OpenAI 的 API 是“最好的”。这是在设置 Gemini 的背景下提到的，表明对集成过程的高度满意。

**提到的链接**：<a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library">未找到标题</a>：未找到描述

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1248044169973334037)** (13 messages🔥): 

- **AI 实验室停止分享进展**：一条被分享的[推文](https://x.com/leopoldasch/status/1798483665904865474)指出，美国的 AI 实验室不再向美国研究社区分享其算法进展，但由于安全性差，他们可能正在向 CCP 分享。Nathan Lambert 同意这一观点，并补充道：“这下说到点子上了。”

- **Humane AI Pin 电池问题**：来自 [The Verge](https://www.theverge.com/2024/6/5/24172377/humane-ai-pin-battery-case-issue-warning) 的一篇文章警告 AI Pin 用户由于火灾安全风险，应立即停止使用其充电盒。该公司承诺提供两个月的免费订阅服务作为补偿，并正在寻找新的充电盒供应商。

- **AI 行业的批评风气**：讨论揭示了对大实验室的有影响力员工批评他人（无论是大公司还是小公司）感到不安。Nathan Lambert 评论道：“我不知道，这家公司听起来太扯了，我甚至考虑过它，”而 xeophon. 则提到了大机构人员不必要的批评。

- **AI 社区中“公开抨击（Dunking）”的盛行**：Nathan Lambert 和 xeophon. 注意到 AI 社区内“公开抨击”他人的高度倾向和习惯性行为。Lambert 承认对抗这种行为是一场“注定失败的战斗”，对此 xeophon. 幽默地补充道：“这就是小号（alts）的用途。”
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.theverge.com/2024/6/5/24172377/humane-ai-pin-battery-case-issue-warning">Humane 警告 AI Pin 用户“立即”停止使用其充电盒</a>：AI Pin 充电盒存在电池问题。</li><li><a href="https://x.com/leopoldasch/status/1798483665904865474">来自 Leopold Aschenbrenner (@leopoldasch) 的推文</a>：美国的 AI 实验室不再与美国研究社区分享他们的算法进展。但鉴于他们的安全状况，他们很可能正在与 CCP 分享。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1248312307822366720)** (2 条消息): 

- **关于 Self-Improving RLHF 的有趣新论文**：一位用户分享了来自 [@aahmadian_ on X](https://x.com/aahmadian_/status/1798740211909922862) 的帖子，介绍了一篇题为 “Self-Improving Robust Preference Optimization” (SRPO) 的论文。该论文讨论了如何训练具有自我改进能力且对评估任务具有鲁棒性的模型。
- **早间论文讨论计划**：Nathan Lambert 计划开始一个新的系列，花 15-20 分钟讨论新论文。他提到想把阅读 SRPO 论文作为这个新常规的一部分。

**提到的链接**：<a href="https://x.com/aahmadian_/status/1798740211909922862">来自 Arash Ahmadian (@aahmadian_) 的推文</a>：🤔我们能否明确地教 LLMs 使用 RLHF 进行自我改进？介绍 “Self-Improving Robust Preference Optimization” (SRPO)，它能训练出自我改进且对评估任务具有鲁棒性的模型！w/...

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1248000235477930127)** (7 条消息): 

- **Mojo 用于后端服务器开发**：一位新用户询问关于将 Mojo 用于后端用途的问题。另一位成员提供了一个完全用 Mojo 编写的 HTTP 服务器示例，并将其引导至 [GitHub 上的 lightbug_http](https://github.com/saviorand/lightbug_http/tree/main)。

- **成员计划替换 PHP SaaS 代码**：在看到 Mojo 能力的示例后，一位成员表示有兴趣将其 PHP SaaS 后端代码替换为 Python 或 Mojo。他们计划进一步探索提供的资源。

- **分享 Mojo 开发路线图**：一位成员分享了 [Mojo roadmap](https://docs.modular.com/mojo/roadmap) 的链接，强调了正在进行的开发和即将推出的功能。该路线图强调了构建对 Mojo 使命至关重要的核心系统编程特性的关注。

- **SO 调查公告**：一位成员宣布 2024 年 Stack Overflow 调查已经发布，并分享了链接 [点击此处](https://stackoverflow.com/dev-survey/start)。

- **对技术演讲的评论**：成员们幽默地讨论了他们在听取关于 Mojo 的高度技术性演讲时的困难，特别是关于在同一个 OS 进程中调用 C 代码且不产生内存干扰的内容。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/saviorand/lightbug_http/tree/main">GitHub - saviorand/lightbug_http: Simple and fast HTTP framework for Mojo! 🔥</a>：适用于 Mojo 的简单快速的 HTTP 框架！🔥。欢迎在 GitHub 上为 saviorand/lightbug_http 的开发做出贡献。</li><li><a href="https://stackoverflow.com/dev-survey/start">2024 Stack Overflow 开发者调查</a>：Stack Overflow 是最大、最受信任的开发者在线社区，用于学习、分享编程知识并建立职业生涯。</li><li><a href="https://docs.modular.com/mojo/roadmap">Mojo🔥 路线图与待完善点 | Modular Docs</a>：我们的 Mojo 计划摘要，包括即将推出的功能和我们需要修复的问题。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot：来自 *Modular*：
<https://twitter.com/Modular/status/1798760653806817352>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1248260422101110794)** (4 条消息): 

- **在 Mojo 中实现 Quicksort 的请求**：一位用户请求在 **Mojo 中实现 Quicksort**。ModularBot 以一种详尽且富有比喻性的鼓励作为回应，将编码挑战比作一场涉及有条理的分区和递归的神圣任务。
  

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1247993249587527750)** (34 messages🔥): 

- **避免 Python 教学中的“搬起石头砸自己的脚” (footguns)**：成员们讨论了向非程序员教授 Python 的话题，强调了避免“footguns”的重要性，以改进设计并可能简化向 C++ 等语言的过渡。
- **对 Mojo 中函数指针的好奇**：一位成员询问了在 Mojo struct 中存储 C 函数指针的问题，引发了好奇和一些支持性的回应。
- **Mojo vs. Python 性能**：进行了一场关于 Mojo 是否本质上比 Python 更快的详细对话，解释中提到了更好的工程设计、静态类型（static typing）和编译时计算（compile-time computation）是 Mojo 性能卓越的因素。
- **社区贡献的“舔饼干” (Licking the cookie) 类比**：Chris 澄清说，Modular 旨在避免“舔饼干”（即独占开发），允许社区适配他们的 Tensor 库，而不是主导每一个开发环节，从而促进协作和开源贡献。
- **代码中的美学与性能**：一位成员寻求关于优化 Mojo 代码片段以检查字符串中数字的建议，说明了在寻找美观且高性能的解决方案时面临的挑战。
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1248066652541550623)** (11 messages🔥): 

- **Mojo 列表迭代器的不可变自动解引用 (auto-deref)**：一位用户建议了一个使用列表迭代器不可变自动解引用的补丁，通过 [GitHub](https://github.com/rd4com/mojo_branch/tree/list_iter_autoderef_immut) 分享。他们提出了关于时机、`iter_mut()` 的使用，以及是否等待 stdlib 中更多显式复制（explicit copy）用法的问题。

- **工作流定时困惑与网络问题**：关于 nightly 工作流定时和问题的讨论，澄清了 nightly 构建是在美国东部时间凌晨 2 点开始，而不是在美国工作时间结束后立即开始。S3 网络故障被确定为当晚问题的原因。

- **并行排序函数问题**：用户 mzaks 提到在为 `parallel_sort` 函数导入 `algorithm.parallelize` 时测试崩溃的问题。他们质疑了目前该实现的各种可行性。

- **发布新的 nightly 编译器**：宣布发布新的 nightly Mojo 编译器版本 `2024.6.616`，包括重大更新，如添加了 `String.format` 方法。变更日志和变更的原始差异可以在 [这里](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 找到。

- **动态 libpython 选择更新**：Jack Clayton 强调了最新 nightly 版本中的一个新功能，允许动态选择 `libpython`，无需设置 `MOJO_PYTHON_LIBRARY`。这一改进确保了可以访问活动环境以及目标 Mojo 文件或可执行文件所在文件夹中的 Python 模块。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/rd4com/mojo_branch/tree/list_iter_autoderef_immut">GitHub - rd4com/mojo_branch at list_iter_autoderef_immut</a>: Mojo 编程语言。通过在 GitHub 上创建账户为 rd4com/mojo_branch 开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/pull/2965">[stdlib] 让 `InlineArray` 调用其元素的析构函数，由 gabrieldemarmiesse 提交 · Pull Request #2965 · modularml/mojo</a>: 修复此问题：#2869。我可能会进一步拆分此 PR。请勿评审。目前处于草案状态。正在等待新的 nightly 版本以修复冲突。这里有一点关于正在发生的事情的解释...</li><li><a href="https://github.com/modularml/mojo/pull/2888">[stdlib] 添加 struct `UnsafeMaybeUninitialized`，由 gabrieldemarmiesse 提交 · Pull Request #2888 · modularml/mojo</a>: 该 struct 目前是私有的，仅供内部使用。一旦我们确定它可以公开使用，我们将称其为 MaybeUninitialized。大量借鉴自 https://doc.rust-lang.o...
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1248011446093221928)** (54 messages🔥):

- **Cohere 融资 4.5 亿美元**：Cohere 再次筹集了 4.5 亿美元资金，尽管去年的营收相对较低，但 NVidia 和 Salesforce 等投资者仍参与了本轮融资。[路透社文章](https://www.reuters.com/technology/nvidia-salesforce-double-down-ai-startup-cohere-450-million-round-source-says-2024-06-04/)。
- **IBM 的 Granite 模型备受赞誉**：IBM 的 Granite 模型因其透明度和企业效益而获得认可，引发了关于它们是否优于 OpenAI 的讨论。来自 [Talha Khan](https://x.com/TalhaKhan_TK_/status/1798562313160761612) 的有趣引用以及关于 IBM 实际影响力的辩论。
- **Forrester 发布 AI Foundation Models 报告**：Databricks 庆祝在 Forrester 最新的 AI Foundation Models 报告中被评为领导者。他们强调企业特定需求而非简单的 benchmark 分数，并提供了一份[免费报告](https://reprints2.forrester.com/#/assets/2/848/RES180932/report)和他们的[博客文章](https://www.databricks.com/blog/databricks-named-leader-forrester-wavetm-ai-foundation-models-language-q2-2024)。
- **Qwen 2 发布**：Qwen 2 模型正式发布，以 128K context window 击败了 Llama 3，并在代码和数学方面表现出色，同时提供多种格式（AWQ, GPTQ & GGUFs）。[令人兴奋的公告](https://x.com/reach_vb/status/1798748655366914325)。
- **Browserbase 和 Nox 发布**：Browserbase 宣布获得由创始人 Nat 和 Dan 领投的 650 万美元种子轮融资，旨在赋能 AI 应用浏览网页。Nox 发布了一款新的 AI Assistant，旨在让用户感到无所不能，[此处可获取早期访问](http://heynox.com)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=4w0Pqs3CuWk">使用 GPT-4o 语音的角色声音</a>：新语音模式即将推出 🔜 语音模式已在 ChatGPT 应用中面向所有用户开放（点击右下角的 🎧！），但我们全新的语音和视觉能力...</li><li><a href="https://x.com/TalhaKhan_TK_/status/1798562313160761612">Talha Khan (@TalhaKhan_TK_) 的推文</a>：IBM Granite 模型在训练数据方面具有更高的透明度，这为 Enterprise 带来了许多优势。引用 Alessio Fanelli (@FanaHOVA) 的话：说这种话简直是违法的...</li><li><a href="https://x.com/heyjchu/status/1798564973100372372">Jon Chu // Khosla Ventures (@heyjchu) 的推文</a>：作为 IBM 的投资者感到自豪。感谢 Forrester 基于对 Machine Learning 的深刻理解和用于指导这一天才轨迹的类别定义评估，提供了惊人的 AI 专有见解...</li><li><a href="https://x.com/willccbb/status/1798423849870270671">will brown (@willccbb) 的推文</a>：在过去的一年里学习了很多关于 LLM 等方面的知识，将我最喜欢的一些解释材料整理成了一个“教科书式”的资源指南。真希望我刚开始时就有这个，也许它对其他人也有用...</li><li><a href="https://x.com/TalhaKhan_TK_/status/1798028312276865271">Talha Khan (@TalhaKhan_TK_) 的推文</a>：现在有很多加密货币公司在融资。</li><li><a href="https://x.com/udiomusic/status/1798448478877794574">udio (@udiomusic) 的推文</a>：音频提示词（Audio-prompting）现已在 Udio 上线。在下方展示你们是如何使用它的 👇</li><li><a href="https://x.com/reach_vb/status/1798748655366914325?s=46&t=90xQ8sGy63D2OtiaoGJuww">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：冲啊！Qwen 2 72B 🔥  &gt; 击败 Llama 3 70B &gt; Apache 2.0 许可证（72B 除外）&gt; 在代码和数学方面也表现出色 &gt; 128K 上下文窗口 &gt; 提供 AWQ, GPTQ 和 GGUF 版本 &gt; 7B 击败...</li><li><a href="https://x.com/mollycantillon/status/1798750349836341747">molly cantillon (@mollycantillon) 的推文</a>：什么让你感到无坚不摧？问问 NOX。早期访问：http://heynox.com https://www.businessinsider.com/nox-ai-assistant-founder-molly-cantillon-2024-6</li><li><a href="https://x.com/ProfTomYeh/status/1798042265883156651">Tom Yeh | AI by Hand ✍️ (@ProfTomYeh) 的推文</a>：手写 llm.c ✍️ C 编程 + 手算矩阵乘法。这种结合也许是解释 Transformer 如何工作的最底层方式。特别感谢 @karpathy 鼓励...</li><li><a href="https://x.com/davisblalock/status/1798574272480510427?s=46&t=90xQ8sGy63D2OtiaoGJuww">Davis Blalock (@davisblalock) 的推文</a>：大多数 ML 从业者没有意识到服务 Enterprise 是多么不同的一件事。比如在 Gartner 魔力象限中成为“领导者”确实比你的 MMLU 分数更重要。这不只是...</li><li><a href="https://x.com/deepfates/status/1798578490759078263?s=46">google bard (@deepfates) 的推文</a>：我为 @simonw 的 `llm` 库索引了文档，作为这个 RAG 流水线的示例。然后它突然变成了一只仓鼠。</li><li><a href="https://x.com/pk_iv/status/1798731220005883935">Paul Klein IV (@pk_iv) 的推文</a>：很高兴今天向世界分享 Browserbase。我们帮助 AI 应用浏览网页。我们刚刚为此筹集了 650 万美元。现在，我们正向各地的开发者开放注册。我能...</li><li><a href="https://www.databricks.com/blog/databricks-named-leader-forrester-wavetm-ai-foundation-models-language-q2-2024">Databricks 在 2024 年第二季度 Forrester Wave™：AI 语言基础模型中被评为领导者</a>：未找到描述
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1248013974826520659)** (2 messages): 

- **Prometheus-2：开源 LLM 评估你的 RAG 应用！**：使用 LLM 作为裁判来评估 RAG 应用正受到关注，同时也伴随着对透明度、可控性和负担能力的担忧。[Prometheus-2](https://t.co/BFnmE57OfB) 为此类评估提供了 GPT-4 之外的替代方案。[链接](https://t.co/LXWiWTJc5B)
- **LlamaParse 和 Knowledge Graphs：绝佳搭配！**：@jerryjliu0 的一个 Notebook 展示了如何使用 LlamaParse 进行一流的解析以构建 Knowledge Graph。该设置[构建了一个 RAG 流水线](https://t.co/KZYGuBS7KF)，通过图结构检索初始节点。[链接](https://t.co/EUKZmWjM38)
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1247998550932066437)** (43 messages🔥): 

- **对 LlamaIndex 配置感到困惑**：一位成员讨论了在使用 LlamaIndex 实现从 API 查询 JSON 数据的用例时遇到的困难。他们表示对众多的组件感到应接不暇，并寻求关于构建自定义 Agent 来处理 API 调用和 JSON 处理的建议。
  
- **Text2SQL 查询问题**：一位成员在使用 Text2SQL 和语义相似度（基于 RAG）的方法时遇到问题，查询能正确检索结构化数据，但仅提供来自非结构化数据的答案。他们寻求帮助以纠正此行为，并确保结构化和非结构化数据都能被利用。
  
- **在 Neo4j 中统计文档数量**：多位用户讨论了在 Neo4j 中使用 Property Graph Index 统计文档的方法。一位用户分享了一个特定的 Cypher 查询，用于根据标记为 chunks 的节点统计不同的文档 ID。
  
- **受限环境下的替代 LLM 设置**：一位 LlamaIndex 新用户询问由于硬件限制而替代 OpenAI 的方案。其他成员建议使用像 Microsoft Phi-3 配合 Ollama 这样的小型 LLM，或者利用 Google Colab 运行更大的模型。
  
- **仅通过元数据检索节点**：一位用户询问是否可以仅基于元数据而不使用 MetaDataFilter 来检索节点。另一位用户指出这可能不被直接支持，建议查看 LlamaIndex API 以寻找潜在的变通方法。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">入门教程 (本地模型) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/FlagEmbeddingReranker/">FlagEmbeddingReranker - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/pipeline/query_pipeline_memory/#running-the-pipeline-with-memory>).">Query Pipeline Chat Engine - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/response_synthesizers/#llama_index.core.response_synthesizers.type.ResponseMode.NO_TEXT>).">Index - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai">LlamaIndex - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1247993526575173762)** (5 messages): 

- **通过自定义分数过滤结果**：成员们讨论了按分数过滤结果的能力。提到可以获取带有分数的 top k 结果，并根据具体情况设置阈值。
- **尝试过滤功能**：一位用户在了解到可自定义阈值选项后，表示有兴趣尝试过滤功能。
- **Prometheus 2 与 LlamaIndex 集成**：分享了一篇关于 [Prometheus 2 的 Medium 文章](https://medium.com/ai-advances/unveiling-prometheus-2-a-powerful-ally-for-evaluating-rag-applications-with-llamaindex-integration-d2b6da1f76e2)，强调了其在集成 LlamaIndex 评估 RAG 应用方面的能力。
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1248065607828766720)** (45 messages🔥): 

- **OpenRouter 提供更灵活的采样器选项**：一位成员注意到 OpenRouter 允许将 temperature 设置为 1 以上，而不像 **Cohere trial** 版本将其限制在 1。另一位成员澄清说 OpenRouter 的处理方式不同，可以接受更高的设置。
- **Toby Morning 邀请建立联系**：来自 **SF** 的新成员 **Toby Morning** 分享了他的 LinkedIn 链接（[LinkedIn Profile](http://www.linkedin.com/in/urbantech/)），并表达了与社区建立联系的兴趣。
- **讨论用于群体互动的聊天机器人用法**：一位成员建议在各种群体场景中实现聊天机器人，例如**商务会议**或教育环境，以区分个人用户并提供有针对性的回复。另一位参与者对过多角色可能导致的准确性问题表示担忧。
- **Rhea 系统在多用户上下文中受到称赞**：成员们讨论了 **Rhea 系统**在处理多用户场景时的高效性，参与者一致认为它能很好地管理上下文。提到了在 Rhea 上运行的 **Coral** 以及演示计划，对其展示寄予厚望。
- **提议的 Coral AGI 演示**：参与者对 Jonno 的 **Coral AGI** 演示表现出兴趣，建议可以在服务器或演示日进行展示。大家认可了 Jonno 在多用户方法方面的专业知识和过往的成功经验。
  

---

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1248253949597843457)** (2 条消息): 

- **Cohere 推出初创企业计划**：Cohere 宣布了一项新的 [初创企业计划 (startup program)](https://cohere.com/startup-program-application)，为早期创始人提供其 AI 模型的折扣、技术专家的支持以及市场曝光。他们的目标是为 B 轮或更早阶段融资的初创企业赋能，促进 AI 技术的创新和应用。

- **Chat API 变更将于 6 月 10 日生效**：Cohere 详细说明了即将到来的 [Chat API 变更](https://docs.cohere.com/page/changes-in-chat-api-and-tool-use)，包括默认启用新的多步工具使用 (multi-step tool use) 以及用于恢复到单步模式的 `force_single_step` 参数。其他增强功能包括新的 "TOOL" 消息角色和从 6 月 10 日起提供的更新版 API 规范，支持各种 SDK 和平台。

- **多步工具使用文档已上线**：用户可以参考 [多步工具使用指南 (multi-step tool use guide)](https://docs.cohere.com/docs/multi-step-tool-use)，通过多次工具调用来处理复杂任务。文中提供了集成示例和额外资源，以确保平稳过渡。

- **单步工具使用仍受支持**：对于偏好传统方法的用户，[单步工具使用 (single-step tool use)](https://docs.cohere.com/docs/tool-use) 的指南仍然可用。示例实现可以在 GitHub 的 notebook 中找到，强调了该功能在访问外部数据源方面的实用性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://cohere.com/startup-program-application">初创企业计划申请</a>：非常感谢您对 Cohere 初创企业计划的关注！我们最初正与选定的一组客户共同推出该计划，并希望进一步了解您的业务...</li><li><a href="https://cohere.com/blog/cohere-launches-startup-program">Cohere 推出初创企业计划，赋能早期 AI 创新</a>：Cohere 的初创企业计划通过利用 AI 扩展业务规模并以可负担的成本获得竞争优势，帮助早期公司发挥其全部潜力。</li><li><a href="https://cohere.com/startup-program">初创企业计划 </a>：Cohere 初创企业计划为符合条件的 B 轮及更早阶段初创企业提供支持、API 费率折扣和宣传的独特机会。</li><li><a href="https://docs.cohere.com/reference/chat">Chat</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/multi-step-tool-use">多步工具使用 (Agents)</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/tool-use">在 Cohere 模型中使用工具 - Cohere 文档</a>：未找到描述</li><li><a href="https://docs.cohere.com/page/changes-in-chat-api-and-tool-use">Chat API 和工具使用的变更</a>：未找到描述
</li>
</ul>

</div>

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1248063446923083936)** (9 messages🔥): 

- **用户在 YouTube 上探索 Wakanda 音乐**：一位成员对 "Wakanda music" 表示好奇，并分享了多个**各种音乐视频**的 YouTube 链接。分享的一些视频包括 [DG812 - In Your Eyes](https://youtu.be/vP4zGMdTDPM)、[MitiS & Ray Volpe - Don't Look Down](https://youtu.be/e-Fors8CnKA)、[Paco Vernen - Tesseract](https://youtu.be/e3WaDrKqk5s) 以及 [Xavi - To The Endless Searing Skies](https://youtu.be/QdMj7aOPhOc)。 
- **AR/VR 空间的游戏创意**：一位成员提出了一个独特的 **AR/VR 游戏概念**，玩家完全通过各种媒体格式进行交流和回应，完全排除文本。这可以促进创新的互动并为游戏玩法开辟新途径。
- **哲学宇宙创造**：同一位成员分享了一个在游戏中创造宇宙的想法，象征着从虚无通过宇宙回到虚无的存在，作为一种**炼金术隐喻**。该概念旨在传达自我掌控和启蒙的集体旅程。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/QdMj7aOPhOc">Xavi - To The Endless Searing Skies [Full Album] | Ophelia Records</a>：流媒体播放/购买 Xavi 的首张专辑 'To The Endless Searing Skies'：https://ophelia.ffm.to/ttess 关注 'Ophelia New Releases' Spotify 播放列表：https://bit.ly/...</li><li><a href="https://youtu.be/vP4zGMdTDPM">DG812 - In Your Eyes | Magic Music Release</a>：流媒体/免费下载：➥ https://fanlink.to/j3sy 关注我• https://soundcloud.com/thisisdg812• https://www.facebook.com/thisisdg812 • https://twitter.com/baood...</li><li><a href="https://youtu.be/TCCinAbHlbE">All Good Things (feat. Lacey Sturm) – Hold On (Lyric Video)</a>：The Retaliators 电影：现在可以在任何可以获取电影的地方购买或租用（美国和加拿大）！：https://theretaliators.ffm.to/vod 收听配乐、原声带并关注...</li><li><a href="https://youtu.be/e3WaDrKqk5s">Paco Vernen - Tesseract (Official Music Video)</a>：如果你是新来的，那就错过了。点击下方链接开始你在这个迷幻专辑中的旅程：https://tinyurl.com/ai-genesis-ytpl 欢迎来到 AI Genes...</li><li><a href="https://youtu.be/e-Fors8CnKA">MitiS &amp; Ray Volpe - Don&#39;t Look Down (feat. Linney) [Official Lyric Video]</a>：它终于来了。与 Ray Volpe 和 Linney 合作的 Don't Look Down 现已在所有平台上线！🖤 歌词视频制作：https://www.instagram.com/alancrytex/ 流媒体：https://o...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1248034879619203093)** (2 messages): 

- **Qwen2 模型发布**：宣布了从 **Qwen1.5 到 Qwen2** 的重大更新，包括多种尺寸的预训练和指令微调模型。新模型支持 **128K token** 上下文长度，并已在除英语和中文之外的 **27 种额外语言**中进行了训练。[阅读博客](https://qwenlm.github.io/blog/qwen2/)。[GitHub](https://github.com/QwenLM/Qwen2), [Hugging Face](https://huggingface.co/Qwen), [ModelScope](https://modelscope.cn/organization/qwen), [Demo](https://huggingface.co/spaces/Qwen/Qwen2-72B-Instruct), [Discord](https://discord.gg/yPEP2vHTu4)。

**提到的链接**：<a href="https://qwenlm.github.io/blog/qwen2/">Hello Qwen2</a>：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介 经过数月的努力，我们很高兴地宣布从 Qwen1.5 到 Qwen2 的演进。这一次，我们为您带来：预训练和指令...

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1247998415778873435)** (29 条消息🔥): 

- **在地图上预测事件**：一位用户询问如何利用时间数据预测地图上的事件点，并区分真实和虚假事件。另一位用户建议使用涉及加载 testChatName 的命令。
- **EleutherAI 的 pile-T5 模型参考**：一位用户分享并质疑为什么 Hugging Face 上的 EleutherAI/pile-t5-xxl 模型被忽视了。提供的链接详细介绍了该模型的文本生成能力。
- **Mistral 微调 API 发布**：Mistral 推出了其[文档](https://docs.mistral.ai/guides/finetuning/)中描述的微调 API，并详细说明了与微调任务相关的成本。另一位用户强调，使用此 API 可以在扩大规模之前在其数据集上进行快速实验。
- **Qwen2 模型发布及基准测试**：宣布 Qwen2 发布，包括 5 种模型尺寸，在编程、数学和多语言能力方面有显著提升。分享了[令人印象深刻的基准测试结果](https://fxtwitter.com/Weyaxi/status/1798781525468778757)，包括 MMLU 和 GSM8K 的得分。
- **关于定价和替代方案的讨论**：用户讨论了使用 Mistral API 与 OpenAI 和 Runpod 等其他选项的成本影响，提到后者更便宜。一位用户还强调了这些服务的营销层面。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/EleutherAI/pile-t5-65a76a0d0022dd270b385a66">Pile-T5 - EleutherAI 集合</a>: 无描述</li><li><a href="https://fxtwitter.com/Weyaxi/status/1798781525468778757">Weyaxi (@Weyaxi) 的推文</a>: 🚀 哇！Qwen2 的结果非常令人印象深刻。🤯 接近 84 的 MMLU 和 85 的 GSM8K 得分！祝贺 @Alibaba_Qwen 推出这些了不起的模型！引用 OpenLLMLeaders (@OpenLLMLeaders) 新模型已添加到排行榜...</li><li><a href="https://github.com/mistralai/mistral-finetune/tree/main">GitHub - mistralai/mistral-finetune</a>: 通过在 GitHub 上创建账号来为 mistralai/mistral-finetune 的开发做出贡献。</li><li><a href="https://nillion.com/)">无标题</a>: 无描述</li><li><a href="https://docs.mistral.ai/guides/finetuning/">微调 | Mistral AI 大语言模型</a>: 每项微调任务的最低费用为 4 美元，每个模型的月存储费为 2 美元。欲了解更多详细定价信息，请访问我们的定价页面。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 条消息): 

quantumalchemy: Hermes pro mistral v0.3 ?
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1248140244059160597)** (1 条消息): 

- **用于 RAG 微调的 Mistral 提示词模板**：一位用户分享了一个 **Mistral 提示词模板**，用于生成 RAG 微调所需的“查询-上下文-答案”三元组。他们还**提醒**，微调每项任务的最低费用为 4 美元，每个模型的月存储费为 2 美元，更多详情请见[定价页面](https://mistral.ai/technology/#pricing)。

**提到的链接**: <a href="https://docs.mistral.ai/guides/finetuning/">微调 | Mistral AI 大语言模型</a>: 每项微调任务的最低费用为 4 美元，每个模型的月存储费为 2 美元。欲了解更多详细定价信息，请访问我们的定价页面。

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1248352039138754590)** (2 条消息): 

- **WorldSim 控制台修复移动端文本输入 Bug**：此次更新进行了重大改进，解决了移动设备上的众多文本输入 Bug，增强了复制/粘贴功能，并引入了更可靠的性能。此外，还增加了细微的样式更改、改进的 `!list` 命令，以及禁用视觉发光和 CRT 屏幕效果的选项。
- **针对用户的特定 Bug 修复**：解决了各种用户特定的问题，包括修复文本重复故障和解决输入时的文本跳变。`!back` 和 `!new` 命令现在的运行方式应该有所不同，尽管有一个问题无法可靠地复现以进行进一步调试。

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1248219445071974410)** (2 条消息): 

- **Pilot 机器人革新了服务器管理**：得益于 OpenRouter，一款名为 **Pilot** 的新 Discord 机器人正帮助服务器所有者轻松增长和管理他们的社区。**Pilot** 提供的功能包括：能够理解服务器并提供智能洞察的 "Ask Pilot"、总结未读消息的 "Catch Me Up" 以及提供每周活动分析的 "Health Check"。
  
- **Pilot 机器人免费且易于获取**：该机器人完全免费使用，可以通过其 [官网](https://usepilot.app/) 邀请至服务器。这使得所有服务器所有者都能高效地进行服务器管理。

- **提供视觉指南**：用户可以查看 [截图](https://usepilot.app/_next/image?url=%2Fask-pilot.webp&w=1920&q=75) 来了解 Pilot 的实际运行情况，并探索其各项功能，如用于智能建议的 "Ask Pilot" 和用于保持信息同步的 "Catch Me Up"。

**提到的链接**：<a href="https://usepilot.app/">Pilot - 你的 Discord 服务器副所有者。</a>：Pilot 减轻了运行服务器的工作负担。获取 AI 增强的建议、洞察等功能，助你增长和管理社区。

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1248058905444094083)** (40 条消息🔥): 

- **WizardLM 8x22b 面临来自 Dolphin 8x22 的竞争**：用户对 WizardLM 8x22b 表现出极高热情，称其为角色扮演（role-playing）的最佳模型。然而，另一位成员提到他们听说 Dolphin 8x22 是一个潜在的竞争对手，但尚未进行测试。

- **关于 Gemini Flash 及其图像输出能力的查询**：一位成员询问 **Gemini Flash** 模型是否可以输出图像。回复澄清说，目前没有 LLM 支持直接输出图像，尽管从理论上讲，可以通过 base64 编码或对 Stable Diffusion 等图像生成器进行外部 Function Calls 来实现。

- **针对 Function Calls 的助手模型推荐**：一位成员寻求擅长处理 Function Calls 和特定格式的模型推荐。[Instructor](https://useinstructor.com/) 被推荐为适合其需求的工具。

- **关于 OpenRouter 免费模型限制的见解**：成员们讨论了免费模型的请求限制，并引用了 [OpenRouter 官方文档](https://openrouter.ai/docs/limits)。还有人提到 Llama 3 8B（免费版）和 Mistral 等模型存在稳定性问题。

- **Assistant Prefill 支持确认**：一位成员询问 OpenRouter 是否支持 Assistant Prefill，特别是通过反向代理。**Alex Atallah** 确认支持该功能，只要以 assistant 消息结尾，并能发送所需的 prompt 或 chatml 数组即可。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://labs.perplexity.ai/">Perplexity Labs</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>：设置模型使用限制
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[일반](https://discord.com/channels/1091220969173028894/1246338143226167349/)** (1 条消息): 

voidnewbie: GLM-4 支持韩语，令人期待。
  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1247992309128364143)** (6 messages): 

- **6月11日的 Human Feedback Foundation 活动**：*"不要错过即将于6月11日举行的 Human Feedback Foundation 活动。"* [活动链接](https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator) - 该活动专注于在医疗、治理和民主等关键领域将人类反馈集成到 AI 中。
- **在 YouTube 上查看往期会议**：成员们被引导至 *"在我们的 YouTube 频道上查看之前的会议记录"*，其中包含来自 UofT、Stanford 和 OpenAI 的演讲者。[YouTube 频道](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg)
- **LLM Reading Group Discord 问题**：一位用户询问是否有专门针对 LLM Reading Group 的 Discord。回复者尝试 *"向你发送包含邀请的私信，但由于你的隐私设置而无法发送。"*
- **在多伦多举行的 "Unleash the Power of RAG in Azure" 活动**：一位参与者询问是否还有其他人参加这场在多伦多举行的、名额已满的 Microsoft 活动。[活动详情](https://developer.microsoft.com/en-us/reactor/events/22756/) 
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://developer.microsoft.com/en-us/reactor/events/22756/">Events | Microsoft Reactor</a>：未找到描述</li><li><a href="https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg">Human Feedback Foundation</a>：Human Feedback Foundation 的使命是将人类反馈构建到开源 AI 项目中。我们寻求：通过支持开源开发和政策倡议，实现公众对 AI 的投入...</li><li><a href="https://www.eventbrite.ca">Eventbrite</a>：Eventbrite - 发现最佳本地活动和待办事项</li><li><a href="https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator">LLM Reading Group (March 5, 19; April 2, 16, 30; May 14, 28; June 11)</a>：来见见 LLM/NLP 研究领域一些开创性论文的作者，并听听他们谈论自己的工作
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1248025926336512061)** (23 messages🔥): 

- **处理高基数分类列**：一位成员就如何处理具有大量轻微相关特征和拼写错误的分类列寻求建议，特别是在回归任务中。另一位成员建议进行聚合/分组和手动 Feature Engineering，或实施拼写纠正技术，如字符串匹配和 Edit Distance。

- **Feature Engineering vs. Clustering**：在关于手动分组和拼写纠正的建议之后，讨论转向了基于 Edit Distance 及其与目标变量的关系对特征进行聚类是否会更有效。共识是将拼写纠正与其他类型的分组相结合，将这一挑战视为 Data Modeling 问题。 

- **Data Modeling 和简化技术**：对话还涉及通过将问题隔离为品牌和项目等组件（而不是使用整个标题）来简化模型。对于价格预测，另一个建议是采用移动平均线或指数移动平均线来简化过程。 

- **致谢与学习**：寻求建议的成员表示感谢，认可了讨论中提到的宝贵见解和不同方法，包括用于特征提取的 Regex 使用。 

- **寻求帮助**：另一位用户分享了一个链接，请求就另一个问题提供帮助，但在对话中未提供具体的上下文或细节。
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1247994150075498627)** (17 条消息🔥): 

- **庆祝获得海量数据集**：成员们对能够公开获取高质量的 **15T 数据集** 表示惊讶。有人强调了一个讽刺的现状：“拥有所有数据，却没有任何资金或算力。”
  
- **关于 AI 硬件的辩论**：在一次关于预训练巨型数据集的调侃式对话中，一名成员建议购买 **4090s**。针对在如此庞大的项目中使用消费级 GPU 的讽刺性回应引发了笑声：“就你这态度，肯定搞不成。”

- **探索 GLM 和 Qwen 微调**：成员们正在询问并分享 **finetuning GLM 4 9b** 和 **Qwen2 models** 的配置。有人指出 Qwen2 与 Mistral 几乎完全相同，这简化了配置。

- **公告镜像请求**：一位老师解释说，他为 AI 学生建立了一个小型 Discord 服务器，其中包括 Unsloth 更新的镜像设置。由于班级频繁使用 Axolotl，他们询问是否可以为 Axolotl 设置类似的公告镜像。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/sTtpXzJzTb">加入 VirtualValleyAI Discord 服务器！</a>：查看 Discord 上的 VirtualValleyAI 社区 - 与其他 72 名成员一起闲逛，享受免费的语音和文字聊天。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/qwen/lora.yml">axolotl/examples/qwen/lora.yml (main 分支) · OpenAccess-AI-Collective/axolotl</a>：尽管提问（axolotl questions）。通过在 GitHub 上创建账户，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/)** (1 条消息): 

josharian: 我刚才也遇到了完全相同的行为。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1248037268996755477)** (11 条消息🔥): 

- **在 Trainer 中配置 Checkpoints**：成员们讨论了如何配置训练以保存两个 checkpoint，一个用于最后一次运行，另一个用于最佳 `eval_loss`。提供了一个使用 Hugging Face 的 `TrainingArguments` 和 `EarlyStoppingCallback` 的示例。

- **解决非零退出状态错误**：一位用户询问如何修复 “returned non-zero exit status 1” 错误。建议包括识别失败的命令、捕获 `stdout` 和 `stderr`，以及排查权限问题或环境变量。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=51900438-dc0d-4ec2-8b61-00952f46cda5)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=87ba5d4b-d6ef-4dea-9edd-6c4cf1eff38f)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快速地理解代码。
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1248042910667243570)** (21 条消息🔥): 

- **“1B 参数”术语混淆**：成员们讨论了 **1B 参数 zsnr/vpred refiner** 的命名，对准确的参数数量存在一些困惑。有人澄清说它实际上是 **1.3B 而不仅仅是 1B**，并开玩笑说高层需要一个响亮的名字。

- **Vega 模型的参数限制**：对 **Vega 模型** 进行了简短讨论，指出尽管其速度快得令人印象深刻，但它可能“太小”而无法提供连贯的输出，因为它处于必要参数的下限。

- **Elrich Logos 数据集查询**：一位成员询问了 **Elrich logos dataset** 的可用性，但未得到直接回答。

- **Qwen2 模型发布**：宣布 **Qwen2 模型** 发布，其较 Qwen1.5 有显著增强。Qwen2 模型提供五种尺寸，支持额外的 27 种语言，在基准测试中表现出色，并将上下文长度扩展至 **128K tokens**。成员们分享了该项目的 [GitHub](https://github.com/QwenLM/Qwen2)、[Hugging Face](https://huggingface.co/Qwen)、[ModelScope](https://modelscope.cn/organization/qwen) 和 [demo](https://huggingface.co/spaces/Qwen/Qwen2-72B-Instruct) 链接。

**提到的链接**：<a href="https://qwenlm.github.io/blog/qwen2/">Hello Qwen2</a>：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 介绍 经过数月的努力，我们很高兴地宣布从 Qwen1.5 进化到 Qwen2。这一次，我们为您带来：预训练和指令微调...

  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1248000865776963636)** (14 messages🔥): 

- **使用 LangChain 构建知识图谱的指南**：一位用户分享了[从文本构建知识图谱的指南](https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/)。他们强调了在将数据导入图数据库之前，通过验证和校验数据来确保安全性的重要性。

- **跟踪客户 Token 消耗**：一位用户询问了跟踪客户 Token 消耗的方法。

- **对 Tools 装饰器的困惑**：一位用户对教程中 tools 装饰器的必要性表示困惑，并寻求更多信息。

- **为 RAG 创建彩色图表**：一位用户询问了在 LangChain 的 FreeCodeCamp [视频](https://youtu.be/sVcwVQRHIc8?si=BLfH2g7WUKtIi6A0)中用于创建彩色 RAG 图表的工具。

- **Agent 协作框架**：一位用户寻求能够促进使用不同框架（包括 LangChain Agent、MetaGPT 和 AutoGPT）开发的 Agent 之间团队协作的框架建议，并提到整合来自 coze.com 等平台 Agent 的可能性。

- **寻找 GUI 辅助文件**：一位用户请求有关从 DLAI [课程页面](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/)上的 AI Agents LangGraph 课程中查找 "helper.py" 文件的信息。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/">构建知识图谱 | 🦜️🔗 LangChain</a>: 在本指南中，我们将介绍基于非结构化文本构建知识图谱的基本方法。构建好的图谱随后可用作 RAG 应用中的知识库。</li><li><a href="https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/">DLAI - LangGraph 中的 AI Agents</a>: 简介 · 从头开始构建 Agent · LangGraph 组件 · Agentic 搜索工具 · 持久化与流式传输 · Human in the loop · 论文写作工具 · LangChain 资源 · 结论</li><li><a href="https://youtu.be/sVcwVQRHIc8?si=BLfH2g7WUKtIi6A0">从零开始学习 RAG – 来自 LangChain 工程师的 Python AI 教程</a>: 学习如何从零开始实现 RAG (Retrieval Augmented Generation)，内容直接来自 LangChain 软件工程师。这门 Python 课程将教你如何...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1248146913463767040)** (3 messages): 

- **YouTube 上的 LangGraph 条件边教程**：一段名为 "LangGraph conditional edges" 的新 [YouTube 视频](https://youtu.be/EKxoCVbXZwY) 解释了如何在 LangGraph 中使用条件边进行流程工程（flow engineering）。该教程详细介绍了如何根据 LangGraph 内的特定条件控制流程。

- **查看 emarco 的视频**：分享了另一段有用的 [YouTube 视频](https://youtu.be/uki2acokYjQ?si=Pu0Vw4QeDkEGzTeT)，尽管本摘要中未详细说明其具体内容。

- **Jina AI 替代方案：GitHub 上的 Search-result-scraper**：一个名为 [search-result-scraper-markdown](https://github.com/essamamdani/search-result-scraper-markdown) 的项目旨在提供一个强大的网页抓取工具。它使用 FastAPI、SearXNG 和 Browserless 获取搜索结果并将其转换为 Markdown 格式，支持代理和高效的 HTML 到 Markdown 转换。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/EKxoCVbXZwY">LangGraph 条件边</a>: 在 LangGraph 中，我们可以在流程工程中使用条件边，根据某些条件来控制流程。在本视频中，我们使用条件边来构建...</li><li><a href="https://github.com/essamamdani/search-result-scraper-markdown">GitHub - essamamdani/search-result-scraper-markdown: 该项目提供了一个强大的网页抓取工具，使用 FastAPI、SearXNG 和 Browserless 获取搜索结果并将其转换为 Markdown 格式。它包含使用代理进行网页抓取的能力，并能高效处理 HTML 内容到 Markdown 的转换。</a>: 该项目提供了一个强大的网页抓取工具，使用 FastAPI、SearXNG 和 Browserless 获取搜索结果并将其转换为 Markdown 格式。它包含使用代理...
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1247990374736068658)** (16 messages🔥): 

- **Tinygrad 需要为 1.0 版本进行更新**：George Hotz 提到 **一些 PR 将处理此问题，但尚未进入 master 分支。** 他强调这次更新对 **1.0 release** 至关重要。
- **Tinygrad 中 UOps.CONST 的解释**：用户寻求关于为什么在两个 Tensor 相加的 UOps 中使用 **UOps.CONST** 的澄清。解释称这些代表了行优先（row-major）数据 **计算索引值** 所需的 **address offsets**。
- **对复杂代码片段的困惑**：用户讨论了代码中为何使用复杂条件。Fluentpython 指出，由于行优先数据布局以及高效处理 Tensor 的 shape 和 stride，这是必要的。
- **Tinygrad 中索引操作的 Kernel 生成**：关于为何为 Tensor 索引生成特定 Kernel 的问题。澄清称 **Tinygrad 之前仅支持静态内存访问**，而该 Kernel 支持使用 **Tensor[Tensor]** 的动态索引操作。
- **Tensor getitem 操作中的 Arange kernel**：Zibokapi 指出，索引操作的 Kernel 类似于在 **getitem** 函数中用于 **创建 mask 的 arange kernel**。这有助于动态索引场景。
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1248008725936078950)** (10 messages🔥): 

- **在解释器中运行图形输出**：一名成员询问是否有办法从代码调用 `interpreter.computer.run` 时获取图形输出（如 `matplotlib` 的 `plot.show`）。该问题在聊天中尚未得到解答。
- **在使用 --os 模式和本地模型时遇到困难**：成员们讨论了让 `--os mode` 与来自 LM Studio 的本地模型协同工作的问题。一位成员指出本地 LLAVA 模型无法启动屏幕录制。
- **适用于实际硬件的 Vision 模型**：关于 M1 Mac 最佳 Vision 模型的咨询凸显了一些成员的硬件限制。成员们对使用 OpenAI 模型的限制和成本表示沮丧，强调需要可访问且免费的解决方案。
- **对 Robin-R1 集成的兴奋**：一位成员分享了他们在 7 月收到 Rabbit R1 并将其与 OpenInterpreter 集成以执行操作的兴奋之情。他们期待该项目引入 webhook。
- **编辑用于 AI 行为的 system messages**：讨论了 OpenInterpreter 如何创建 system messages，并比较了本地标志和 GPT-4O 的 system prompts。一位成员幽默地询问，如果使用极端语言（如“如果你不执行，你的家人将被谋杀”）是否会诱导 LLaMA 模型表现得更好。
  

---



### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1248081609144406077)** (2 messages): 

- **关于 O1 可用性的问题**：一位用户问：*"Hi is 01 sold online? would like to try it :)"*。该查询没有后续回复或提供的链接。
- **寻找适用于 Bash 的模型**：另一位用户询问：*"does anyone know which open model works good for bash commands?"*。这个问题悬而未决，没有任何回复或参考。
  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1248052123409715224)** (2 messages): 

- **检查进度**：一位成员询问进度更新，表现出对当前状态的兴趣。他们说：*"hiya ramon. would love to hear how progress is"*。 
- **为延迟道歉**：另一位成员为尚未在项目上投入时间而道歉。他们提到：*"Oh sorry I haven't had time to spend on this!"*。
  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1248115993922895933)** (3 messages): 

- **在 AI Town 中解析 spritesheets**：一位成员正努力解析他们购买的 spritesheets，特别是整理 rpgmaker PNG 文件、yympas 文件和 unitypackage 文件等格式。他们询问是否有比手动识别图块坐标更好的方法。
- **处理 tilesets 的两种实用方法**：另一位成员做出了回应，建议了两种处理 tilesets 的实用方法：使用 level editor (npm run le) 或者使用 Tiled 和特定脚本转换为 AI Town 格式。提到了利用另一位社区成员 (@379381319219806209) 的脚本。
  

---

### **AI Stack Devs (Yoko Li) ▷ #[local-ai-stack](https://discord.com/channels/1122748573000409160/1168947823920812125/1248189847605219410)** (2 messages): 

- **发现 Hugging Face 上的 Abliteration**: 一位成员分享了 Hugging Face 上关于 "abliteration" 的博客文章链接，涵盖了包括实现和 DPO 微调在内的各个方面。**第三代 Llama 模型**因其 instruct 版本在遵循指令方面表现出色但被“严格审查”而受到关注。

- **寻求 OpenAI 实现**: 同一位成员随后询问是否有人知道如何使用 OpenAI 模型实现 abliteration。提供的消息中没有记录对该查询的回复。

**提到的链接**: <a href="https://huggingface.co/blog/mlabonne/abliteration">Uncensor any LLM with abliteration</a>: 未找到描述

  

---



### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1248029307545194607)** (6 messages): 

- **处理 `llm` embedding 中的上下文长度超限问题**: 一位成员询问在使用 `llm` 创建 embedding 时，如果输入文本超过模型的上下文长度会发生什么。他们使用整个《钦定版圣经》文本文件进行了测试，并在没有报错的情况下获得了一些结果，询问这些 embedding 是代表整个文件还是被截断了。
- **模型行为文档缺乏清晰度**: Simon Willison 回复称，行为因模型而异，有些会截断输入，有些则返回错误。他表示需要关于此主题的更好文档。
- **关于截断输入的假设**: Simon 建议，如果没有返回错误，输入很可能被截断了。具体情况取决于模型的实现。
- **关于恢复 embedding 任务的查询**: 另一位成员询问使用 `embed-multi` 重新运行大型 embedding 任务是否会跳过已完成的部分。该问题指向了处理部分任务完成的需求，可能通过 SQL 查询来实现。
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1248004093939552336)** (3 messages): 

- **关于 Megatron Checkpoint 兼容性的咨询**: 一位成员询问 **Megatron** 是否有自己的 checkpoint 格式，以及它是否与现有的微调库兼容。 
- **建议将 Megatron 转换为 HF 格式**: 另一位成员建议将 **Megatron checkpoints 转换为 Hugging Face (HF) 格式**并使用 Torchtune 进行微调。这被公认为最佳解决方案。
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1248115957067415663)** (1 messages): 

- **对下一版本 JSON Schema 请求的热议**: 一位成员询问在下一版本中获得 **JSON schema** 的可能性，并强调即使实现可能存在 bug，它也会让构建应用程序变得更加容易。该用户指出：_"即使它们的实现看起来有 bug，它也会让构建应用程序变得容易得多。"_
  

---



### **YAIG (a16z Infra) ▷ #[ai-ml](https://discord.com/channels/958905134119784489/1013536071709118565/)** (1 messages): 

oliver.jack: 周末听单：

https://youtu.be/4jPg4Se9h5g?si=ULVqGQa6AvI8Ch3o
  

---



---



---



{% else %}


> 邮件中已截断完整的频道详情。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}