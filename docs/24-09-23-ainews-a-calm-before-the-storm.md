---
companies:
- anthropic
- openai
- alibaba
- microsoft
- blackrock
- groq
- aramco
- disney
- eth-zurich
- pudu-robotics
- slack
date: '2024-09-23T23:33:49.803194Z'
description: '以下是该文本的中文翻译：


  **Anthropic** 在预期发布重大产品前，正以高达 **400 亿美元** 的估值筹集资金。**OpenAI** 推出了全新的推理模型 **o1** 和
  **o1-mini**，提高了速率限制，并发布了多语言 MMLU 基准测试。**阿里巴巴** 发布了支持 29 种以上语言的开源模型 **Qwen2.5**，以更低的成本展现出与
  **GPT-4** 相当的性能。**微软** 和 **贝莱德 (Blackrock)** 计划向 AI 数据中心投资 **300 亿美元**，同时 **Groq**
  与沙特阿美 (Aramco) 合作建设全球最大的 AI 推理中心。机器人领域的进展包括迪士尼研究院和苏黎世联邦理工学院 (ETH Zurich) 开发的基于扩散模型的机器人动作生成技术，以及普渡机器人
  (Pudu Robotics) 推出的半人形机器人。Slack 和微软推出了集成在其平台中的 AI 智能体。研究亮点包括利用双块注意力机制 (Dual Chunk
  Attention) 实现 **Llama-2-70b** 的长文本扩展，以及通过 KV 缓存量化使 **Llama-7b** 模型支持 100 万 token
  的上下文。'
id: dd694420-bc14-4229-b025-f0f76213db66
models:
- o1
- o1-mini
- qwen2.5
- gpt-4
- llama-2-70b
- llama-7b
original_slug: ainews-sxxx
people:
- adcock_brett
- philschmid
- rohanpaul_ai
- jvnixon
- kateclarktweets
- sama
title: 暴风雨前的宁静
topics:
- long-context
- kv-cache-quantization
- diffusion-models
- reinforcement-learning
- robotics
- ai-integration
- multilinguality
- model-benchmarking
- model-performance
- model-optimization
---

<!-- buttondown-editor-mode: plaintext -->**Peace is all you need.**

> 2024年9月20日至9月23日的 AI News。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**221** 个频道，**6206** 条消息）。预计节省阅读时间（以 200wpm 计）：**719 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

虽然没有明确的头条新闻，但在本周 Anthropic 和 Meta 预期的大动作之前，有很多值得注意的小动态：

- [CUDA MODE](https://x.com/swyx/status/1837577267259887702) 和 [Weights and Biases](https://x.com/morgymcg/status/1838062480926368013)（本月推理赞助商）在本周末成功举办了黑客松。CUDA MODE 通过 [更名为 GPU MODE](https://x.com/jeremyphoward/status/1838341110344880637) 进行了庆祝。
- Berkeley Function Calling Leaderboard [发布了 V3 版本](https://x.com/shishirpatil_/status/1837205152132153803)（是的，[V2 就在上个月](https://buttondown.com/ainews/archive/ainews-ideogram-2-berkeley-function-calling/)），重点关注多轮/多步函数调用。o1 mini 的表现出人意料地差。
- 还有几个值得注意的 o1 评估——关于 [test time budget](https://x.com/hughbzhang/status/1838288923656941860) 以及 [一篇探索其规划能力的正式论文](https://x.com/polynoamial/status/1838251987009183775?s=46)。
- Anthropic [再次以高达 400 亿美元的估值进行融资](https://x.com/KateClarkTweets/status/1838319202798538974)。
- OpenAI 发布了 [多语言 MMLU (MMMLU)](https://x.com/_philschmid/status/1838230108072476951?s=46)。
- Sama 将此称为 [Intelligence Age](https://ia.samaltman.com/)。
- [纽约时报确认了 Jony Ive 手机的消息](https://x.com/8teapi/status/1837979330867351626?s=46)，且 [Scale AI 正在处理一个小危机](https://x.com/natolambert/status/1837996707780624631?s=46)。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 发展与行业动态**

- **OpenAI 的新模型**：[@adcock_brett](https://twitter.com/adcock_brett/status/1837885345972605182) 报道了 OpenAI 发布的新推理模型 o1 和 o1-mini，旨在处理科学、编程和数学领域的复杂任务。[@JvNixon](https://twitter.com/JvNixon/status/1837884523092283599) 指出这些模型在输出质量上有主观上的提升。OpenAI 还[提高了速率限制](https://twitter.com/adcock_brett/status/1837885561203224595)，o1-mini 增加至每天 50 条消息，o1-preview 增加至每周 50 条消息。

- **Qwen2.5 模型**：阿里巴巴发布了 [Qwen2.5](https://twitter.com/adcock_brett/status/1837885606384312457)，这是一个开源模型，包含通用、编程和数学版本，支持 29 种以上语言。[@_philschmid](https://twitter.com/_philschmid/status/1837932334823145535) 将其性能与 GPT-4 进行了比较，指出其以极低的成本实现了类似的性能。

- **AI 基础设施**：Microsoft 和 Blackrock 正在[筹集 300 亿美元](https://twitter.com/adcock_brett/status/1837885460120547541)用于投资新建和现有的 AI 数据中心，总投资潜力可达 1000 亿美元。Groq 与 Aramco 合作建造[“全球最大的 AI 推理中心”](https://twitter.com/adcock_brett/status/1837885437651677217)，配备 19,000 个 LPUs，最终将扩展至 200,000 个。

- **机器人领域的 AI**：Disney Research 和 ETH Zurich 展示了 ['RobotMDM'](https://twitter.com/adcock_brett/status/1837885482669162795)，该技术将基于扩散的动作生成与 RL 相结合，用于机器人运动。普渡机器人（Pudu Robotics）发布了他们的[第一代“半人型”（semi-humanoid）](https://twitter.com/adcock_brett/status/1837885392097358135)机器人。

- **技术产品中的 AI 集成**：Slack 宣布了[新的 AI 驱动功能](https://twitter.com/adcock_brett/status/1837885415161794902)，包括频道内的 AI Agent。Microsoft 介绍了 [Microsoft 365 Copilot 中的 Agent](https://twitter.com/adcock_brett/status/1837885369053831582)，可跨多种 Microsoft 产品工作。

**AI 研究与技术**

- **长上下文模型**：一篇关于[“无需训练的大语言模型长上下文缩放”](https://twitter.com/rohanpaul_ai/status/1837853153246470414)的论文介绍了 Dual Chunk Attention (DCA)，使 Llama2 70B 在无需持续训练的情况下支持超过 100k tokens 的上下文窗口。

- **KV Cache 量化**：[“KVQuant”论文](https://twitter.com/rohanpaul_ai/status/1837852496364023831)提出了量化缓存 KV 激活的技术，允许在单个 A100-80GB GPU 上运行上下文长度高达 100 万的 LLaMA-7B 模型。

- **检索技术**：[@_philschmid](https://twitter.com/_philschmid/status/1837752035501858975) 讨论了 SFR-RAG，这是一个针对 RAG 进行微调的 9B LLM，其在学术基准测试中的表现可媲美更大的模型。

- **合成数据**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1837982057693131055) 强调了合成数据在训练 Qwen2.5-Coder 中的关键作用，详细介绍了生成过程、验证以及与开源数据集的整合。

**AI 工具与应用**

- **GitHub 文件整理工具**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1837985813935641024) 分享了一个 GitHub 仓库，该工具使用本地 LLM 来理解并根据内容对文件进行分类。

- **金融研究助手**：[@virattt](https://twitter.com/virattt/status/1837878341405167778) 正在使用 LangChain 构建一个开源金融研究助手，配备了强大的金融和网络数据搜索工具。

- **类 Perplexity 体验**：[@LangChainAI](https://twitter.com/LangChainAI/status/1837899668103352700) 分享了一个使用 LangGraph、FastHTML 和 Tavily 创建类 Perplexity 体验的开源仓库，支持包括 GPT-4 和 Llama3 在内的不同模型。

**AI 伦理与监管**

- **加州 AI 法案 SB 1047**：关于加州 AI 法案 SB 1047 的辩论仍在继续。[@JJitsev](https://twitter.com/JJitsev/status/1837905422415540373) 认为该法案存在严重缺陷，它监管的是通用技术而非其应用。多位 AI 研究人员和机构对该法案对 AI 研发的潜在影响表示担忧。

**其他**

- **GitHub 上的 AI 贡献**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1837829123625853259) 指出，自 OpenAI 发布 ChatGPT 以来，GitHub 上的 AI 贡献量激增了 230%。

- **AI 数据中心**：[@ylecun](https://twitter.com/ylecun/status/1837875035270263014) 建议未来的 AI 数据中心将建在能源产地附近，特别是核电站旁边，以获得高效、低成本且低排放的电力。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Qwen2.5 成为新的开源 SOTA，取代更大规模的模型**


- **[谁在日常配置中用 Qwen2.5 替换了原有模型？如果是这样，你替换了哪个模型？](https://reddit.com//r/LocalLLaMA/comments/1fmoa14/who_replaced_a_model_with_qwen25_for_a_daily/)**（分数：42，评论：30）：据报道，**Qwen2.5** 在广泛的任务中实现了 **state-of-the-art (SOTA)** 性能，模型参数规模从 **0.5B 到 72B** 不等。帖子作者正在询问已将 Qwen2.5 集成到日常工作流中的用户，询问他们具体替换了哪些模型以及用于什么任务。
  - **Professional-Bear857** 将 **Llama 3.1 70B IQ2_M** 替换为 **Qwen2.5 32B IQ4_XS**，用于代码编辑/纠错和通用查询，理由是 GPU 功耗更低，且性能与 **Mistral Large** 相当。
  - 用户正在尝试将 **Qwen2.5** 用于各种任务，包括**文章和 YouTube 视频摘要**。**Matteogeniaccio** 使用自定义的 Python 设置和 **llama.cpp server** 来处理不同的内容类型并提取关键信息。
  - 虽然一些用户称赞 Qwen2.5 的指令遵循能力，但也有人报告了褒贬不一的结果。**Frequent_Valuable_47** 发现 **Gemma2 2B** 在 YouTube 转录摘要方面优于 **Qwen2.5 1.5B**，尽管 Qwen2.5 拥有 **120k token context**，而 Gemma 仅为 **8k**。


**主题 2. 在 Open WebUI 中使用 gVisor 沙箱实现安全代码执行**



- **[Open WebUI 中的安全代码执行](https://github.com/EtiennePerot/open-webui-code-execution)**（分数：324，评论：24）：Open WebUI 已经实现了使用 **Docker 容器**进行**安全代码执行**，以增强安全性。此功能允许用户在隔离环境中运行代码片段，在实现交互式编程体验的同时，防止对宿主系统造成潜在伤害。该实现利用 **Docker SDK** 进行容器管理，并包含一个**超时机制**来自动终止长时间运行的进程。
  - 代码执行功能已在 [GitHub](https://github.com/EtiennePerot/open-webui-code-execution) 上线，并使用 **gVisor** 进行沙箱隔离。它提供两种模式：用于在 LLM 消息中运行代码块的 **"Function"** 模式，以及允许 LLM 自主执行代码的 **"Tool"** 模式。
  - 用户讨论了将支持扩展到 **Go** 等其他语言，开发者解释说，这需要修改 `Sandbox` 类和解释器选择代码。该工具目前适用于 **Ollama** 后端和标记为支持 tool calling 的模型。
  - 用户对处理缺失依赖项以及对更强大功能（如 artifacts 和增加并发请求）的需求表示关注。开发者确认 **Open WebUI v0.3.22** 包含了使该工具正常运行所需的修复。


**主题 3. 针对角色扮演场景优化的 NSFW AI 模型**



- **[最喜欢的轻量级 NSFW RP 模型（20B 以下）？](https://reddit.com//r/LocalLLaMA/comments/1fmqdct/favorite_small_nsfw_rp_models_under_20b/)**（分数：180，评论：156）：该帖子比较了各种 **20B 参数以下的轻量级 NSFW RP 模型**，并将它们分类为“好”、“极好”和“绝对精彩”。作者专门使用 **EXL2** 模型，首选包括 **MN-12b-ArliAI-RPMax-EXL2-4bpw**、**estopia-13b-llama-2-4bpw-exl2** 和 **Mistral-Nemo-Instruct-2407-exl2-4bpw**。列出的大多数模型是 **4-4.5bpw**（bits per weight）变体，规模从 **7B 到 13B** 参数不等。
  - 用户讨论了各种 **NSFW RP 模型**，其中 **L3-Nymeria-Maid-8B-exl2** 和 **Cydonia 22B** 被强调为特别令人印象深刻。**Nicholas_Matt_Quail** 提供了关于模型演进的广泛见解，指出 **Cydonia 22B** 感觉像是对 12B 模型的重大升级。
  - 社区分享了针对不同 VRAM 容量的建议，包括适用于 4GB 的 **Sao10K_L3-8B-Stheno** 和适用于更高容量的 **L3-Super-Nova-RP-8B**。用户强调了正确的 **sampling techniques** 和 **instruct templates** 对于获得最佳模型性能的重要性。
  - 讨论涉及了无审查模型的使用场景，包括露骨的性内容以及涉及暴力或黑暗主题的非性场景。**chub.ai** 网站被提及作为角色卡和 RP 场景的资源。


**主题 4. Qwen2.5 模型的越狱和审查测试**

- **Qwen2.5 可以被越狱，但并不完美。** ([Score: 49, Comments: 24](https://reddit.com//r/LocalLLaMA/comments/1fmvj0n/qwen25_is_able_to_be_jailbroken_but_its_not/)): **Qwen2.5 模型** (72b, 32b, 14b) 使用 Ollama 和 Open-webui 进行了审查测试，最初尝试询问有关**维吾尔族迫害**的问题时，结果为 100% 拒绝。随后开发了一个**自定义 System Prompt** 来鼓励无偏见、详细的回答，成功绕过了关于维吾尔族和香港问题的审查，在 20 次测试中实现了 **100% 无审查回答**。然而，该方法对**有关中国政府的直接问题**证明是**无效的**，表明在这些话题上存在持久的“封锁”，而关于其他政府（如美国）的问题则得到了更具批判性的回答。
  - 用户讨论了模型的回答，一些人注意到它对美国的政治贪婪给出了**“措辞考究的重击”**，而在中国话题上则更为克制。**32b 模型**因其性能受到称赞，并提到了 **128k Context** 能力。
  - 关于模型的回答是代表**审查还是训练数据的偏见**引发了辩论。一些人认为模型的亲华立场可能反映了其训练数据，而非刻意的审查，而另一些人则暗示某些话题可能被**“消融”（ablation）**了。
  - 一位用户使用关于**天安门广场**的 Prompt 测试了 **14b 模型**，收到了出人意料的详细回答，涵盖了关键事件及其后果。这引发了关于模型处理敏感话题的能力以及 Prompt 措辞对回答影响的讨论。


**Theme 5. 对新 Command-R 模型更新的热情有限**



- **没人喜欢新的 Command-R 吗？** ([Score: 33, Comments: 28](https://reddit.com//r/LocalLLaMA/comments/1fmt93e/no_love_for_new_command_r/)): 该帖子讨论了 **Cohere** 最近对 **Command-R 模型**的改进，指出与大约六个月前的首次发布相比，公众热情有所下降。尽管 Cohere 声称在**推理、RAG、数学和编码**方面增强了能力，但作者观察到明显缺乏针对更新模型的 **Benchmark、博客文章、LocalLLaMA 适配或 YouTube 评论**。帖子最后询问是否有人在使用新的 Command-R，并邀请用户分享他们的经验。
  - 用户将 **Command-R** 与 **Qwen2.5-32B**、**Mistral 123b** 和 **Magnum 123b** 等其他模型进行了比较，对性能的评价褒贬不一。一些人发现 Command-R 在**故事创作**和**文档聊天**等特定任务上表现更好，而其他人则更喜欢替代模型。
  - Command-R 的**非商业许可证**被认为是限制兴趣和采用的一个重要因素。用户对限制性条款表示沮丧，特别是禁止将输出用于商业用途，鉴于 Cohere 的数据收集实践，一些人认为这很虚伪。
  - 据指出，与最初版本相比，新的 Command-R 在 **RP/ERP** 方面表现更差，而最初版本曾意外地在这一领域表现出色。然而，**GQA** 的改进使得在高达 **128k 的长 Context 长度**下表现更好，可能有利于 **RAG 和 Tool Use** 应用。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 研究与技术**

- **Google Deepmind 推进多模态学习**：一篇关于[联合样本选择](https://arxiv.org/html/2406.17711v1)的论文展示了数据策展如何加速多模态学习。(/r/MachineLearning)

- **Microsoft 的 MInference 加速长上下文推理**：[MInference](https://arxiv.org/abs/2407.02490) 能够在保持准确性的同时，为长上下文任务实现高达数百万个 token 的推理。(/r/MachineLearning)

- **通过 10 亿个网络策划的角色扩展合成数据生成**：一篇关于[扩展合成数据生成](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/)的论文利用 Large Language Models 中的多样化视角，从网络策划的角色中生成数据。(/r/MachineLearning)

**AI 模型发布与改进**

- **Salesforce 发布 xLAM-1b 模型**：该 10 亿参数模型在 [function calling 中实现了 70% 的准确率，超越了 GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。(/r/LocalLLaMA)

- **Phi-3 Mini 更新并支持 function calling**：Rubra AI 发布了更新后的 [支持 function calling 能力的 Phi-3 Mini 模型](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)，可与 Mistral-7b v3 竞争。(/r/LocalLLaMA)

- **阿里巴巴推出 100 多个新的开源 AI 模型**：阿里巴巴[发布了众多 AI 模型和一款文本生成视频工具](https://www.cnbc.com/2024/09/19/alibaba-launches-over-100-new-ai-models-releases-text-to-video-generation.html)。(/r/singularity)

**AI 应用与实验**

- **Flux：迭代图像转换**：一项展示[将输出图像重复输入回 Transformer 块会发生什么](https://www.reddit.com/r/StableDiffusion/comments/1fmu7eb/flux_what_happens_if_you_keep_feeding_the_output/)的实验。(/r/StableDiffusion)

- **简单的 Vector Flux LoRA**：[使用 LoRA 进行基于矢量的图像转换](https://www.reddit.com/r/StableDiffusion/comments/1fn465p/simple_vector_flux_lora/)的演示。(/r/StableDiffusion)

- **AI 生成的桌面图标**：关于[使用 AI 创建自定义桌面图标](https://www.reddit.com/r/StableDiffusion/comments/1fn2i9e/do_you_use_ai_to_make_custom_icons_for_your/)的讨论。(/r/StableDiffusion)

**AI 伦理与社会影响**

- **教皇呼吁实行全民基本收入 (UBI)**：教皇[再次呼吁实行全民基本收入](https://www.indcatholicnews.com/news/50680)，引发了关于 AI 对就业影响的讨论。(/r/singularity)

- **Worldcoin 的 UBI 虹膜扫描**：Sam Altman 的 Worldcoin 项目在提议的 UBI 系统中[使用虹膜扫描进行身份验证](https://www.businessinsider.com/worldcoin-sam-altman-iris-scanning-face-auth-tools-humanity-ubi-2024-8)，引发了隐私担忧。(/r/singularity)

**AI 幽默与梗图**

- **电路板长矛**：一张[带有电路板尖端的长矛](https://i.redd.it/nz2560p8mgqd1.jpeg)的幽默图片，引发了关于后末日场景和 AI 角色的讨论。(/r/singularity)

- **AI 对邪恶的看法**：一段 [ChatGPT 对话](https://i.redd.it/rwgnu6itobqd1.jpeg)，其中 AI 将“人类”识别为邪恶的根源，引发了关于 AI 伦理和人性的辩论。(/r/OpenAI)


---

# AI Discord 摘要

> 由 O1-preview 生成的摘要之摘要的总结

**主题 1：AI 模型新发布与更新**

- [**OpenAI 推出 O1 模型：推理能力的飞跃**](https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-five)：**O1 模型**在推理能力上展现了显著提升，在挑战性基准测试中从 *0% 跃升至 52.8%*，这暗示了可能采用了合成数据训练。
- [**Aider v0.57.0 增强 AI 结对编程**](https://aider.chat/HISTORY.html)：**Aider v0.57.0** 现在支持 **OpenAI O1 模型**，改进了 Windows 兼容性，并集成了新的 **Cohere 模型**，该版本中 **70%** 的代码是由 Aider 自身编写的。
- [**Gradio 5 Beta 发布，性能大幅提升**](https://5-0-dev.gradio-website.pages.dev/playground)：**Gradio 5 Beta** 引入了重大的性能增强、现代化的设计更新，以及一个用于快速应用测试的实验性 **AI Playground**。

**主题 2：AI 工具与模型的挑战与问题**

- **Perplexity Pro 用户面临订阅困扰**：用户报告 **Perplexity Pro** 状态间歇性丢失，并遇到 *'Query rate limit exceeded'* 错误；退出登录等临时修复措施仅部分有效。
- **LM Studio 更新后模型加载出现障碍**：在更新 **LM Studio** 后，用户在加载模型时遇到挑战，部分用户不得不回滚版本以恢复功能。
- **OpenRouter 默认禁用 Middle-Out Transform**：**OpenRouter** 已默认禁用 **middle-out transform**，这影响了用户的工作流，并引发了关于 Prompt 处理的困惑。

**主题 3：创意领域的 AI**

- [**AI 驱动的 RPG 开发正在进行中**](https://github.com/slangerosuna/space_cowboy_rpg)：一位开发者正在创建一款集成具有记忆和联网功能的 **AI agents** 的 RPG 游戏，由于系统复杂性，正在寻求社区贡献。
- [**音乐制作 AI 在音乐理论方面表现挣扎**](https://www.reddit.com/r/LocalLLaMA/comments/15fnvlla/qwen25_bugs_issues_fixes_colab_finetuning_notebook)：讨论显示，音乐制作中的 AI 模型在处理和弦转调等基础音乐理论任务时感到吃力，凸显了训练数据有限带来的局限性。
- [**播客生成技术令用户兴奋**](https://huggingface.co/spaces/saq1b/podcastgen)：**PodcastGen** 利用受 Google NotebookLM 启发的先进技术来生成播客，尽管一些用户注意到了内容重复的问题。

**主题 4：AI 研究与实践的发展**

- [**μ-Parameterization 指南简化模型训练**](https://blog.eleuther.ai/mutransfer/)：**EleutherAI** 和 **Cerebras** 联合发布了一份指南，以提高 **μ-parameterization (μP)** 的易用性，包括分步说明和在 [nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup) 中的简单实现。
- [**BFCL V3 评估 LLM 中的多轮 Function Calling**](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)：**Berkeley Function-Calling Leaderboard V3** 引入了针对**多轮（multi-turn）**和**多步（multi-step）Function Calling** 的新评估，这对于评估 LLM 在复杂任务中的表现至关重要。
- [**SetFit v1.1.0 发布，增强训练能力**](https://huggingface.co/posts/tomaarsen/875775738519407)：**SetFit v1.1.0** 现在使用 **Sentence Transformers Trainer** 在 **CPU 和 GPU** 上进行高效的分类器训练，并支持 **MultiGPU** 以及 Python **3.11** 和 **3.12**。

**主题 5：社区活动与协作**

- **CUDA MODE 黑客松展示创新项目**：黑客松在一天内产生了超过 **40 个项目**，入选路演的团队专注于商业可行性和创新，彰显了社区的协作精神。
- **参与者寻求 AI 实习机会**：成员们正积极寻求关于在哪里寻找 **AI 实习**的建议，反映了社区对在 AI 领域发展职业生涯的浓厚兴趣。
- [**为智能家具提议 Open Interpreter 模块**](https://www.kickstarter.com/projects/kequel/kequel-modular-customizable-bedside-table)：一位成员提议为 **Kequel 模块化定制床头柜**创建一个 **Open Interpreter** 模块，并寻求社区协作。


---

# 第 1 部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingFace Spaces 宕机**：用户报告了 **HuggingFace Spaces** 的重大问题，经历了持续数小时的 '500 Internal Error' 和文件上传失败。
   - 这次停机令依赖该平台进行模型访问和内容上传的用户感到沮丧，凸显了其对生产力的影响。
- **模型微调简化**：一位用户寻求在包含 **350 条记录** 的操作系统和硬件问题数据集上微调模型的帮助，并通过 [SimpleTuner](https://github.com/bghira/SimpleTuner) 等共享资源找到了支持。
   - 多位用户讨论了模型训练工具，发现了有效的解决方案，包括 YouTube 视频推荐和社区见解。
- **秒级 3D 内容创作**：一位成员分享了 [threestudio GitHub repo](https://github.com/threestudio-project/threestudio)，声称可以在 **10 秒** 内生成 3D 对象。
   - 另一位参与者推荐使用 'stable fast 3D'，据称该工具可以在不到一秒的时间内从图像生成对象，可在 Hugging Face space 中使用。
- **Gradio 5 Beta 发布**：**Gradio 5 (Beta)** 正式发布，根据开发者反馈进行了性能增强、设计更新，并推出了用于快速应用测试的实验性 **AI Playground**。
   - 该 Beta 版本承诺大幅提升性能，特别是在服务器端渲染方面，同时通过第三方审计确保了安全性的提高。
- **开发 AI 驱动的 RPG**：一位开发者正在开发一款集成具有记忆和联网功能的 AI Agent 的 RPG，在系统构建方面面临复杂挑战。
   - 他们向社区寻求贡献，强调了实现这种复杂游戏结构的重大挑战。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.57.0 带来令人兴奋的更新**：**Aider v0.57.0** 的发布通过多项更新增强了性能，包括对 **OpenAI o1 models** 的支持、改进的 **Windows compatibility** 以及新 **Cohere models** 的集成。
   - 它还修复了多个 Bug，用户可以在[此处](https://aider.chat/HISTORY.html)查看完整的 **change log**。
- **Aider 和 OpenRouter 已就绪但过程坎坷**：用户分享了在 **OpenRouter** 和 **Claude models** 中使用 **Aider** 的混合体验，经常遇到“过载”错误和困惑。
   - 一些成员成功访问了 **Anthropic** 模型，而另一些成员则对当前高流量期间服务的可靠性表示担忧。
- **对 Embeddings 的质疑**：一位成员对 **embeddings** 的价值表示怀疑，主张采用一种 DIY 方法，模仿 **llama index** 中看到的树状结构方法。
   - 这一讨论指向了 AI 领域的广泛趋势，一些人将 RAG 工具的激增归因于 **VC funding** 而非真实需求。
- **Aider 优化的创意解决方案**：为了简化工作流程，建议使用 **ripgrep** 快速搜索工具以更好地与 Aider 集成，强调了开发速度的重要性。
   - 用户还讨论了在 Aider 设置中使用较低的 token 计数以提高清晰度并减少困惑，特别是在处理大型仓库时。
- **Git 和聊天处理增强**：Aider 的仓库映射（repository mapping）有助于跟踪代码更改和交互，尽管某些配置促使用户关闭自动刷新以保持高效的搜索能力。
   - **HuggingFace models** 的集成以及使用 **.env** 文件管理环境设置增强了 Aider 在 AI 配对编程中的可用性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **与 Cerebras 联合发布 μ-Parameterization 指南**：今天，我们很高兴能联合发布博文 [《最大更新参数化从业者指南》(The Practitioner's Guide to the Maximal Update Parameterization)](https://blog.eleuther.ai/mutransfer/)，旨在提高训练社区对 **μ-parameterization** (μP) 的易用性。
   - 该指南包括**逐步实现指令**以及在 [EleutherAI/nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup) 上的简单实现，解决了原始材料中常见的易用性问题。
- **在 GPT-4 中使用余弦相似度**：一位用户正在评估 GPT-4 在不进行微调的情况下执行分类任务的效果，考虑根据测试集的余弦相似度动态选择示例，以改进 In-context learning。
   - 有人担心在 Prompt 中包含相似的测试示例可能会导致测试集泄漏（Test set leakage），需确保测试问题本身不被包含在内。
- **关于课程学习（Curriculum Learning）有效性的辩论**：目前正在讨论 AI 中课程学习 (CL) 的有效性，一些人对其是否比传统训练方法有显著改进持怀疑态度。
   - 成员们指出，目前缺乏保证的数据过滤最佳实践，这影响了 CL 的实际应用。
- **MMLU_PRO 采样逻辑需要关注**：`./leaderboard/mmlu_pro` 任务与其原始实现有所不同，因为它在 Few-shot 采样时忽略了问题类别，具体可见[这段代码](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/47b9891aacb8bd7cda29d5c5ba17b9434dd333bc/evaluate_from_local.py#L228)。
   - 另一位用户建议更新采样逻辑，以根据问题类别提高准确性，代码详见[此处](https://github.com/rimashahbazyan/lm-evaluation-harness/blob/f117e6c09e32c553df0ab8cf8964a8b16636832e/lm_eval/api/samplers.py#L186)。
- **激活函数文档不同步**：一位成员指出，文档中列出的可用激活函数并未反映代码中的完整范围，特别是关于 [Swiglu](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/activations.py) 的部分。
   - 另一位成员确认文档尚未更新，并引用了定义这些函数的[特定代码行](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/neox_arguments/neox_args.py#L295)。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **KTO Trainer 需要参考模型**：成员们澄清 **KTO trainer** 需要一个参考模型来计算奖励，建议在微调期间使用未经改动的基座模型进行比较。
   - 建议从参考模型中*预生成响应*，以节省训练期间的显存。
- **Qwen 模型 Bug 报告出现**：**用户**注意到 **Qwen 2.5 模型**在更新后出现异常行为，特别是 Prompt 模板生成错误响应的问题。
   - 已确认较小的模型对 Prompt 格式非常敏感，从而导致了这些问题。
- **RAG 实现引起关注**：参与者讨论了使用**检索增强生成 (RAG)** 来改进模型响应并在分析过程中增强知识保留。
   - 一位用户建议在 RAG 中有效地利用现有数据集，以避免训练过程中的知识丢失。
- **SetFit v1.1.0 发布，增强训练功能**：**SetFit v1.1.0** 的发布现在采用 Sentence Transformers Trainer，以便在 **CPU 和 GPU** 上进行高效的分类器训练，解决了之前的问题。
   - 关键更新包括 **MultiGPU 支持**，并将 'evaluation_strategy' 弃用改为 'eval_strategy'，同时新增对 **Python 3.11** 和 **3.12** 的支持。
- **分类器训练采用结构化方法**：训练 **SetFit 分类器模型**涉及两个阶段：首先微调 Sentence Transformer Embedding 模型，然后将 Embedding 映射到类别。
   - 这种结构化方法提高了性能和效率，特别是配合 1.1.0 版本中的新特性。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 订阅困扰**：多位 **Perplexity** 用户报告间歇性失去 **Pro** 身份，并遇到类似“Query rate limit exceeded”的错误消息。退出并重新登录等临时修复方法仅能偶尔解决问题，但凸显了更新后系统范围内的延迟问题。
   - 用户担心持续存在的 Bug 可能会严重影响他们在平台上的体验。
- **AI 模型对决：Llama vs. Perplexity**：讨论显示 **llama-3.1-sonar-large-128k-online** 的表现逊于 **Perplexity web app**，用户注意到其回答不完整且格式不一致。针对改进输出提出了建议，重点在于抓取来源引用。
   - 性能上的差异引发了对该模型在实际应用中可靠性的质疑。
- **Chain of Thought 推理的奥秘**：成员们参与了关于 **Chain of Thought reasoning** 资源的讨论，旨在提升 AI 的逻辑和推理能力。分享了一份详细介绍实现的指南，增强了开发复杂 AI 模型的工具包。
   - 进一步的讨论强调了这种推理方式在提高 AI 现实场景功能能力方面的持续应用。
- **对 Perplexity API 引用的不满**：用户对 **Perplexity API** 不稳定的引用功能表示失望，尽管有明确要求，但往往无法提供一致的参考文献。批评指出，API 的可靠性在很大程度上取决于准确的引用提供。
   - 这种不一致性可能会损害该 API 在专注于严肃应用的开发者社区中的声誉。
- **OCR 服务在 Azure 部署的可能性**：人们对在 **Azure** 上部署 **Perplexity API** 以提供 OCR 服务表现出好奇，反映出对 API 在云环境中实际应用的兴趣日益增长。这可能为利用该 API 功能集成 OCR 能力开辟新途径。
   - 关于 Azure 部署的咨询量表明，向基于云的 AI 解决方案发展的趋势正在演变。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **黑客松团队协作**：参与者制定了黑客松的协作策略，建议通过指定频道进行自我组织和沟通，以优化团队合作。
   - 成员建议由于停车位有限，使用 Uber 出行，强调了后勤规划对活动成功的重要性。
- **CUDA Mode 活动亮点**：黑客松在积极的反馈中拉开帷幕，展示了显著的项目和协作成果，激励了参与者对未来努力的信心。
   - 十支团队被选中进行路演，评委关注商业可行性和创新，并提醒各团队按时完成提交。
- **KLDivLoss 与 Kernel 问题**：对 **KLDivLoss** 反向传播 kernel 的担忧引发了关于其公式准确性以及与较大词表大小相关的潜在循环展开（loop unrolling）问题的讨论。
   - 参与者建议研究 KLDivLoss 与 Cross-Entropy 实现之间的关系，以增强模型性能并减少差异。
- **WebGPU vs. MPS 性能**：成员指出，虽然在 **macOS** 上 **MPS** 的性能优于 **WebGPU**，但 WebGPU 仍处于开发阶段，尚未达到峰值性能，表明仍有改进空间。
   - 目前正在协作推动优化 MPS 和 WebGPU 之间的 kernel 对比，并呼吁社区就增强实现提供建议。
- **算力额度与支持需求**：参与者明确了如何领取 **compute credits**，确认不会发送确认邮件，但资金会在注册后不久到账。
   - 跨节点安装 Python 包的支持被确认成功，反映了社区在解决问题时的资源共享精神。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 支持云端测试**：订阅者现在可以直接在云端测试 **OpenRouter** 服务，无需本地安装；提供了一个包含 Loom 视频的小型演示。
   - *这种设置方便用户快速高效地探索各项功能。*
- **即将举行关于 OpenRouter 高级用法的网络研讨会**：即将举行的直播研讨会定于 **美国东部时间中午 12 点**，重点讨论如何扩展到数千个**并行 Agent 和代理 (proxies)**。
   - *通过查看相关 YouTube 频道的 Live 标签页了解更多详情。*
- **默认禁用 Middle-Out Transform**：**OpenRouter** 已正式默认禁用 Middle-Out Transform，这影响了许多用户的工作流。
   - *这一变化引起了关注，凸显了该功能对于各种前端和后端系统的重要性。*
- **关于 Anthropic 新模型发布的猜测升温**：传闻暗示 **Anthropic** 即将发布新模型，有迹象表明将在 Google 活动期间宣布。
   - *该公告可能会伴随大量的免费 Token 优惠，引发了开发者之间的讨论。*
- **探讨私有 LLM 服务器**：一名成员询问参与者是自己在运行**私有 LLM 服务器**，还是在使用第三方服务。
   - *该询问引发了关于这些服务器管理和运营的讨论。*

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **音乐制作 AI 在音乐理论方面面临挑战**：讨论显示，**音乐制作**领域的大模型在处理**基础乐理任务**（如和弦转调）时面临挑战，目前正尝试使用 feline AI 生成 MIDI 文件。
   - 参与者一致认为，由于训练样本有限，**乐谱 (music notation)** 仍是一个重大障碍。
- **Bittensor 引发伦理担忧**：成员们对 **Bittensor** 似乎在未妥善致谢的情况下复制 **Nous Research 的分布式训练算法**表示担忧，对 AI 领域的伦理实践提出质疑。
   - 对话表明，分布式训练的**创新**必须优先于单纯增加参数量。
- **新型医疗 LLM 亮相**：推出了多款新模型，包括 **HuatuoGPT-II** 和 **Apollo**，旨在增强医疗 AI 能力，特别是在基因-表型映射和多语言应用方面。
   - **HuatuoGPT-Vision** 也展示了其多模态处理实力，提升了医疗数据处理的可访问性。
- **LLM 变革临床试验**：LLM 正被用于改进临床试验，特别是 **AlpaPICO**，它可以生成 PICO 框架，简化了临床报告流程。
   - 这些进步旨在提高**医疗文档**的质量并改善临床环境中的工作流。
- **探索用于推理的 RL 环境**：目前正在讨论创建专门为推理任务定制的 **RL 环境**，强调需要类似于开源微调的多样化设置。
   - 成员指出，成功的训练在很大程度上取决于高质量数据集和环境的选择。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AI 在心理健康支持中的作用**：成员们讨论了心理健康问题患者可能因为病耻感而更倾向于与聊天机器人交流，这使得在医疗保健中合规使用 AI 变得至关重要。
   - 虽然 AI 可以辅助心理健康诊断，但必须遵守 **data privacy regulations**（数据隐私法规），且不能取代专业护理。
- **解决 AI 系统中的偏见**：小组强调了教授动机性推理和确认偏误的重要性，以提高使用 AI 时的批判性思维。
   - 他们一致认为 AI 的建议应基于具有严格伦理标准的 **scientific advice**（科学建议）。
- **Cohere 的研究重点非常多样**：Cohere 致力于包括语言模型、效率、安全和 AI 政策在内的各种课题，相关资源可在其 [research papers page](https://cohere.com/research/papers) 找到。
   - 鼓励成员探索这些主题，作为其持续职业发展的一部分。
- **Embedding 调用参数更新**：一位用户在进行 embedding 调用时遇到了错误，提示 '`embedding_types parameter is required`'，这表明最近的要求发生了变化。
   - 这引发了 **Cohere team** 的澄清，因为之前的文档说明该参数是可选的。
- **AI-Telegram-Chatbot 项目发布**：一位成员分享了他们的 [AI-Telegram-Chatbot](https://github.com/derssen/AI-Telegram-Chatbot) GitHub 仓库，展示了 **Cohere AI** 的实际应用。
   - 该机器人旨在通过 **AI-driven responses**（AI 驱动的响应）增强用户交互，反映了人们对 Cohere 技术实际应用的广泛兴趣。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 反馈最后召集**：参加一个 30 分钟的简短电话会议，分享你对 **Magic** 的看法；参与者将获得专属周边（swag）。你可以在[这里](https://modul.ar/user-feedback)预约。
   - *参与至关重要*，这有助于改进 Magic 并从社区收集更广泛的经验。
- **Mojo 的 Python 集成难题**：成员们讨论了将 **Python libraries** 集成到 **Mojo** 中的可行性，并对可能影响性能的 GIL 冲突表示担忧。他们思考为 Python 类创建直接的 Mojo 文件是否能简化使用。
   - 社区保持谨慎态度，强调虽然集成是有益的，但可能会影响 Mojo 的效率和目标。
- **MAX 自定义算子（Custom Ops）需要明确说明**：关于 **MAX custom ops** 状态的查询引发了对 [modular documentation](https://docs.modular.com/max/api/mojo/register/register/op) 中记录的更改的关注。成员们正在寻求有关最近更改或函数移除的更新。
   - 社区成员渴望获得更清晰的文档，表达了对正确使用 MAX 操作指南的迫切需求。
- **Mojo 中的位打包（Bit Packing）和结构体（Structs）**：讨论围绕 **Mojo** 中缺乏原生 **bit packing** 展开，成员们考虑使用手动打包和变长类型等替代方案来优化结构体大小。讨论中还出现了关于结构体对齐对性能影响的担忧。
   - 提到了利用 **LLVM** 增强功能来管理不同位宽的可能性，这为解决这些效率问题提供了一条路径。
- **Mojo 向通用编程语言演进**：用户对 **Mojo** 成为成熟的 **general-purpose language** 表示乐观，认为其能力超出了单纯的 AI 应用。与 MAX 等平台的集成被视为实现更广泛可用性的关键。
   - 这种情绪表明大家共同渴望看到 Mojo 在保持高性能和竞争力的同时不断进化。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 模型遇到加载障碍**：用户在更新 LM Studio 后（尤其是 CUDA Llama.cpp v1.1.9 更新后）面临模型加载挑战，触发了包括清除缓存在内的各种修复尝试。
   - 许多用户选择回滚版本，分享了在持续的挫败感中恢复功能的解决方案。
- **不支持图像生成模型**：讨论显示 LM Studio 不支持像 **Flux** 这样的图像生成模型，会导致“unknown model architecture”错误。
   - 用户澄清这些模型是为其他平台设计的，明确了 LM Studio 的使用边界。
- **DDR6 发布时间线不确定**：关于 **DDR6** 可用性的担忧浮现，用户推测广泛采用可能要到明年年底。
   - 持续的讨论反映了在消费级硬件能够充分利用该技术之前，仍处于等待明确规范的阶段。
- **RTX 4090 性能表现参差不齐**：**RTX 4090** 的性能指标出现差异，测试结果从低于 **20t/s** 到有争议的 **60t/s** 不等。
   - 不一致性表明在不同模型配置下的设置和测量存在挑战，引发了对性能一致性的质疑。
- **ROCm 支持流程简化**：对 ROCm 支持感兴趣的用户了解到，最新版本的 LM Studio 通过自动检测 ROCm 安装简化了流程。
   - 预计此更新将为依赖 AMD GPU 设置的用户提供更便捷的安装。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **探索 Stable Diffusion 功能**：用户讨论了 **Stable Diffusion** 的各个方面，包括 [Dalle3 功能](https://x.com/LikeToasters/status/1836632745075736913) 以及 **Flux** 在 VRAM 利用率方面的限制。
   - 对话强调了特定工具，如旨在增强 prompt 的 boorutag 自动补全。
- **FLUX 模型利用面临 VRAM 挑战**：成员分享了使用 **FLUX 模型** 的经验，详细说明了使用 **LoRAs** 和在图像生成过程中管理 VRAM 的挑战。
   - 建议采用将 text encoders 保留在 DRAM 上等技术来优化模型性能。
- **为角色一致性训练 LoRAs**：讨论集中在对精确 prompt 的需求以及训练 **LoRAs** 以在漫画等项目中保持一致的角色生成。
   - 参与者提到使用 IP adapters 来提高图像创建过程中的角色连贯性。
- **用于图像补全的 Inpainting 技术**：用户寻求关于 **inpainting 技术** 的建议，以便在保持风格和连贯性的同时有效填充图像缺失部分。
   - 推荐使用 **Fooocus** 和 **RuinedFooocus UI** 等工具来增强 inpainting 过程。
- **AI Art 生成的一致性**：对话围绕通过使用相同的 prompt 和设置来确保 **AI art** 的一致性展开。
   - 强调了保持一致的 seeds 和设置，以及有助于在生成的图像中保持风格的工具。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **o1-mini 在创意写作方面表现不佳**：**o1-mini** 在诗歌创作中受困于陈词滥调和可预测的结构，与 **Claude Opus 3** 相比，其创意深度稍逊一筹。用户一致认为，提高 Prompt 的特异性可能会改善结果。
   - *改进 Prompting 可能会释放更好的创造力*，但目前的性能限制仍然是一个挫折。
- **分享高效的 Embedding 存储实践**：一位成员讨论了针对 **12-13k 文本集合**的 Embedding 高效存储方案，重点介绍了 **S3** 和 OpenAI 的 Vector Store 作为主要选项。目标是实现有效的聚类和检索。
   - 这次对话反映了人们对优化 AI 数据管理方法的持续关注。
- **AI 工具应对 PDF 分析**：一位用户寻求能够分析 PDF 的工具，包括为 AI 知识库将图像转换为文本，许多 **RAG** 解决方案被指出支持 PDF 集成。然而，在准确转换图像方面仍存在差距。
   - 社区认识到推进多模态模型以更有效地处理此类任务的必要性。
- **考察 AI 聊天机器人模型性能**：参与成员对比了 AI 聊天模型，强调了 **o1-mini** 在创意写作任务中不如 **Claude Opus 3**。讨论突出了 **Prompting** 在最大化模型输出方面的关键作用。
   - 人们对即将推出的、有望在创意领域提升性能的模型表现出浓厚兴趣。
- **关于企业级 gpt-o1-preview 配额的见解**：讨论显示，有人推测企业账户的 **gpt-o1-preview 配额**可能与 **Tier 5 限制**一致，正如 [Rate Limits 指南](https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-five)中所引用的那样。
   - *成员们正在寻找更清晰的文档来解锁这些企业功能*。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 设备开发确认**：Jony Ive 确认正在开发一款 **OpenAI AI 设备**，Sam Altman 已与 Apple 达成分销协议，可能重塑智能手机市场。
   - 社区对与这款即将推出的设备相关的传闻订阅模式反应不一。
- **AI SDK 3.4 增强工具执行**：**AI SDK 3.4** 的发布引入了自动多步工具执行，促进了各种编程语言的后端开发。
   - 利用该 SDK 的值得注意的应用包括用于 SQL 翻译的 **postgres.new** 和多功能 Web 开发 **Agent** **v0**。
- **Elicit.org 在研究领域赢得赞誉**：**Elicit.org** 因其在简化学术文献综述方面的能力而受到成员称赞，使研究过程更加高效。
   - 用户强调了社区推荐在发现相关 AI 工具和发展方面的重要性。
- **Gorilla Leaderboard V3 挑战 LLM**：**BFCL V3** 的推出旨在评估 **LLM** 如何处理多轮工作流和 **Function Calling**，这对于复杂的 AI 任务至关重要。
   - 该排行榜解决了对现实世界 AI 应用至关重要的性能指标。
- **Anthropic 准备进行巨额融资**：Anthropic 正在进行讨论，估值可能在 **300 亿至 400 亿美元**之间，可能使其之前的估值翻倍。
   - 这一融资举动发生在竞争激烈的 AI 市场中，反映了投资者巨大的信心。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **O1 模型的推理飞跃**：最近的讨论揭示了 **O1** 改进后的推理能力在挑战性基准测试中从 **0% 跃升至 52.8%**，暗示了可能使用了合成数据训练。
   - 这表明了重大进展，可能与在复杂任务中利用有效的训练方法论有关。
- **Anthropic 寻求估值提升**：有消息称 **Anthropic** 正在寻求融资，这可能将其估值推高至 **300 亿至 400 亿美元**，可能是其先前价值的两倍。
   - 这反映了在激烈竞争中，投资者对 AI 初创生态系统的热情不断高涨。
- **Shampoo 训练了 Gemini，引发关于信息把关的讨论**：经确认，**Shampoo** 被用于训练 **Gemini**，这引发了社区内关于信息把关（gatekeeping）的讨论。
   - 尽管论文已经公开，但许多人对 Shampoo 在此背景下的作用所带来的影响表示惊讶。
- **GameGen 扩散模型突然退出**：讨论集中在 **GameGen 扩散模型**在 GitHub 上的迅速崛起和意外消失，引起了用户的困惑。
   - 这一事件呼应了人们对 AI 游戏开发领域中“卷款跑路”（rug pulls）的担忧。
- **Twitter 安全问题升级**：正如[社区警报](https://x.com/zachxbt/status/1836473279479189916)中所报道的，最近许多 Twitter 账号被黑，导致影响知名用户的 Meme 币诈骗。
   - 有人质疑安全问题是源于 **SIM swapping** 还是固有漏洞，特别是当开启了 2FA 安全验证的账号仍然遭到入侵时。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **使用 NVIDIA NIM 构建 RAG 应用**：一篇关于 [NVIDIA NIM](https://t.co/zFC0DorIMW) 的优秀教程指导用户创建一个全栈 RAG 应用，连接了 **Llama 3**、**ArXiv 数据集**、作为向量数据库的 **Milvus** 以及用于应用界面的 **Gradio**。
   - 该项目展示了实现强大 RAG 功能所需的关键组件的有效集成。
- **Nudge 微调改进 Embedding**：[NUDGE](https://t.co/FT1C2x3Iov) 提供了一种非参数化的 Embedding 微调方法，将过程从**数小时缩短至数分钟**。
   - 这一创新突显了模型 finetuning 操作效率的显著提升。
- **多模态 RAG 应对产品手册**：讨论集中在构建多模态 RAG 系统，以简化对**复杂产品手册**（如宜家家具组装手册）的理解。
   - 该方法表明需要复杂的设置来高效地索引、搜索和检索数据，从而提升用户体验。
- **Cleanlab 的 TLM 增强信任**：一篇文章讨论了 **Cleanlab 的 TLM** 如何改进 **LlamaIndex** 中的 **RAG 系统**，重点是提高 AI 在法律等关键应用中输出的可靠性。
   - 它强调了能够产生准确响应的可靠 AI 系统的重要性，以对抗普遍存在的不完整和过度自信的输出问题。
- **使用 LitServe 进行本地模型服务**：来自 **LightningAI** 的 [LitServe](https://t.co/Xikqk20peW) 提供了一个使用 FastAPI 提供服务并扩展 LLM 模型的框架，如 LlamaIndex 的演示所示。
   - 该框架允许用户构建高效的 RAG 服务器并进行本地托管，从而改进操作工作流。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 2.5.0 低调发布**：**期待已久的 DSPy 2.5.0** 已经发布，简化了迁移过程并弃用了所有 2.4 版本之前的 LM 客户端，鼓励用户通过 `dspy.LM(model_name, **kwargs)` 转换到受支持的提供商。
   - 随着用户适应新版本，官方正积极征求反馈，并提供文档和支持以协助过渡。
- **Chat Adapter 改进解决了重复响应问题**：成员们讨论了对自定义 Chat Adapter 的需求，因为较小的 LLM 模型（<7B）在 'chat complete' 模式下会产生重复响应，该解决方案目前正在测试中。
   - 这一增强功能旨在提升用户体验，早期采用者的反馈对于微调新架构至关重要。
- **合成数据生成速度飙升**：一份报告强调了在微调较小模型后，合成数据生成速度取得了显著提升，从**每秒 30 个 token 增加到 2500 个 token**。
   - 这一进步使 DSPy 成为高效生成大规模合成训练数据的有力工具。
- **TrueLaw 凭借 DSPy 见解引起关注**：在最近的一期 [MLOps Podcast #260](https://youtu.be/O0F3RAWZNfM?si=ckG2DWkwop8zu-ZA) 中，**TrueLaw Inc.** 的 CTO Shiva Bhattacharjee 讨论了如何利用 **DSPy** 解决特定领域的专业问题。
   - 对话强调了**领域特定模型 (domain-specific models)** 对提升性能的重要性，特别是在法律行业。
- **文本分类的挑战与咨询**：一位成员提出了关于为复杂的文本分类任务扩展 docstrings 的可能性，寻求提高 LLM 理解能力的方法。
   - 还有人询问在 Groq 上可用的 **Chain of Thought (COT)** 方法，表明了对扩展测试能力的浓厚兴趣。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **CUDA Hackathon 的好奇探索者**：一位成员询问是否有人参加即将举行的 **CUDA Mode IRL hackathon**，引发了从活动中获取见解的兴趣。
   - *这是一个讨论 GPU 编程和优化策略最新进展的好机会。*
- **优化 CPU Offloading 以提升性能**：针对优化器中缺失 **CPU offloading** 的问题（特别是在 [full_finetune_single_device.py](https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_single_device.py) 中）引发了关注，这暗示了由于遗留问题可能导致的性能下降。
   - 成员们建议默认采用 *PagedAdam* 以提高**内存效率**，并强调了向更优化方法持续过渡的重要性。
- **KV Caching 面临挑战**：讨论集中在 40GB 显存的机器上使用 KV caching 且 Batch Size 为 8 时，*qwen2.5 1.5B 模型* 出现的 **OOM 问题**。
   - 成员们建议通过检查 KV cache 的形状来排查故障，以确定其是否已正确初始化为最大长度，旨在缓解此类问题。
- **模型评估中的 Batch Size 困惑**：关于增加 **Batch Size** 对模型评估影响的辩论浮出水面，特别是在多任务场景下。
   - 参与者倾向于分析与 Cache 初始化相关的权衡，以及 CPU 和 GPU 之间**权重和梯度 (weights and gradients)** 的交互。
- **评估 Recipe Bug 修复历程**：重点讨论强调了一个解决组任务评估 Recipe 中 Bug 的 PR，如 [PR #1642](https://github.com/pytorch/torchtune/pull/1642) 所示，这表明在实施更改时需要及时发布补丁。
   - 大家一致同意在等待 **评估 Recipe** 最新更新的同时，应迅速处理已识别的修复补丁。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **CLIP Retrieval 替代方案匮乏**：成员们讨论了 [CLIP Retrieval](https://rom1504.github.io/clip-retrieval/) 替代方案的稀缺性，并指出 rom1504 可能不会再对其进行维护。
   - 一位用户表示，他们的研究项目需要一个兼容 **LAION 400M** 的后端解决方案。
- **寻求 AI 实习机会**：一位用户请求关于在哪里寻找 AI 实习机会的建议，强调了社区指导的重要性。
   - 这一询问反映了人们对在 AI 领域推进职业发展的兴趣日益增长。
- **模型训练数据集分享**：一个用于训练 **Llama-3.1** 的数据集被上传到了 Hugging Face，并征求关于其编程有效性的反馈。
   - 分享的数据集包含详细的应用描述，引发了关于最佳实践的讨论。
- **总结器 AI 需要反馈**：一位用户分享了他们新开发的 [总结器 AI](https://www.fluxa.pro)，并寻求社区测试和反馈。
   - 对其潜力的认可伴随着关于消息长度自定义的建议，以提高可用性。
- **播放列表生成器项目介绍**：一位用户展示了 [Adify](https://adify.pro)，这是一个根据用户提示词创建 Spotify 播放列表的生成器。
   - 该项目获得了积极的反响，表明人们对创新音乐生成工具的浓厚兴趣。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **VGA 重新夺回 GPU 连接荣光**：一位用户确认他们的 **GPU** 仅通过 **VGA** 连接，克服了与显示密码错误相关的问题。
   - 这种变通方法使他们能够使用较旧的 VGA 连接成功为设备供电。
- **ShapeTracker 合并性悬赏咨询**：有关于在 **Lean** 中 **ShapeTracker** 合并性的悬赏状态查询，并表达了将其作为**本科论文**课题的兴趣。
   - 尚未解决的状态激起了渴望探索这一复杂课题的学生们的好奇心。
- **Answer AI 讨论成本效益**：讨论围绕 **Answer AI** 盒子的成本效益展开，其价格可能优于当前解决方案，包括潜在的批量折扣。
   - 参与者希望展示这种经济型配置的基准测试，旨在证明其财务可行性。
- **Tinygrad 的云集成概念蓬勃发展**：用于集成到 tinygrad 的 **CLOUD=1** 选项引起了关注，旨在简化功能而不依赖 AWS 风格的虚拟化。
   - 成员们讨论了该设备选项如何在保持性能的同时增强可用性。
- **Metal 教程提供见解**：分享了一个关于 **Metal** 教程的 [GitHub 链接](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20240921_metal.md)，扩展了关于 tinygrad 集成的知识。
   - 该教程为热衷于提高 tinygrad 中 Metal 相关技能的贡献者提供了资源。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Agent 在本地 AI 集成中面临问题**：用户报告称，在六个月的间隔后，**Agent 无法与本地 AI 配合使用**，并建议将 **Ollama** 作为更好的替代方案。
   - 这展示了在动态开发环境中对兼容本地 AI 解决方案的持续探索。
- **关于最佳向量库选项的辩论**：关于 **Hugging**、**OpenAI** 或 **Ollama** 哪个是其项目的最佳向量库（Vector Store）展开了激烈讨论。
   - 选择正确的向量库可能会对**性能**和**可扩展性**产生关键影响。
- **聊天机器人项目中的 PDF 处理优化**：一位用户寻求在向量数据库中高效拆分和存储 PDF 内容的方法，以避免冗余的中间步骤。
   - 这一改进将简化工作流程，提高整体处理性能。
- **文本生成推理参数的挑战**：针对即使将 `return_full_text` 设置为 false，输出中仍意外出现 **<|end|>** token 的问题提出了询问。
   - 这表明需要提高推理参数的清晰度，以便用户更好地控制。
- **作品集聊天机器人帮助用户咨询**：一位用户为其作品集推出了聊天机器人助手，方便回答客户关于其服务的咨询。
   - 他们欢迎社区反馈以进一步完善该工具，体现了开发中的协作精神。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **针对床头柜的 Open Interpreter 模块**：一位成员提出了为 [Kequel 模块化可定制床头柜](https://www.kickstarter.com/projects/kequel/kequel-modular-customizable-bedside-table) 创建 Open Interpreter 模块的想法，并询问是否有合作意向。
   - 该倡议旨在增强智能家居技术的集成，邀请其他开发者贡献想法和进行开发。
- **Open Interpreter 的用户界面挑战**：在使用命令行输入时，屏幕可见性引起了关注，促使人们提出增强视觉清晰度的解决方案。
   - 成员们讨论了在 Open Interpreter 处理外部输入时改善用户体验的潜在权变措施。
- **LiveKit 在 Android 上拦截明文连接**：一位用户注意到较新的 Android 手机会阻止 **01 移动应用**通过 **HTTP** 连接到本地 **LiveKit** 服务器，提示“不允许明文通信 (CLEARTEXT communication not permitted)”。
   - 他们建议使用 ngrok 获取 HTTPS 端点，这可以有效解决暴露服务器用户的连接问题。
- **GitHub 关于明文通信的解决方案**：一个 GitHub issue 详细说明了一项提议，即严格针对本地网络**启用明文通信**，并确保就安全问题向用户发出通知。
   - 这解决了连接挑战，同时平衡了开发者与本地设备交互时的网络安全。
- **调查后端请求循环**：一位成员质疑 Open Interpreter 发送的频繁后端请求，怀疑存在无限循环的情况。
   - 寻求关于后端响应预期的澄清，以帮助确定准确的请求结论。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Qwen 2.5 获得比 Llama 3.1 更多的赞誉**：一位成员指出 **Qwen 2.5** 获得了强烈的正面反馈，如 [Reddit 对比](https://www.reddit.com/r/LocalLLaMA/s/NiCbaTyodk) 所强调的，其在基准测试中略微优于 **Llama 3.1**。
   - 这提高了社区对最新模型对比中经过验证的性能指标重要性的认识。
- **Axolotl 中的长上下文挑战**：关于 **Axolotl** 在 ShareGPT 中处理长于 **max_seq_len** 的对话能力的讨论，反映了社区对上下文管理的兴趣。
   - 随着成员们深入研究模型训练协议，这些训练复杂性的清晰度仍然是一个热门话题。
- **Llama 3.1 的 Rope Scaling 争论**：一位成员质疑在约 **120K tokens** 的长上下文 CoT 轨迹上训练 **Llama 3.1 8B** 时，是否必须使用 **rope_scaling**，因为在 **sequence_len** 超过 **40K** 时遇到了内存问题。
   - 尽管使用了带有 deepspeed zero3 的多 GPU 环境，但处理长上下文的复杂性继续引发工程师之间的讨论。
- **微调峰值查询**：用户报告在 **100K 行数据集**上进行微调时出现了意外的峰值，促使人们寻找与特定数据点的相关性。
   - 启用更广泛日志记录的努力被证明是不够的，使得微调机制仍处于审查之中。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Sentx.ai 进军意识开发领域**：Sentx.ai 正在开拓**意识开发**工作，目前仍处于早期阶段。他们正在积极征求**普遍意见**，特别是关于其 Alignment 方法的意见。
   - 鼓励成员评估意识开发对未来 AI Alignment 的务实影响。
- **提出 AI Alignment 的自我调整方案**：Sentx.ai 介绍了一种让模型**自我调整其与人类价值观对齐 (Alignment)** 的策略，避免硬性限制。这种方法旨在围绕有效的 Alignment 实践培养**持续对话**。
   - 社区成员正在讨论自我调整模型在现实场景中的影响及其潜在益处。
- **征集 Alignment 项目合作**：公开邀请分享关于**类似项目**的信息，以促进 Alignment 开发方面的合作。鼓励成员交流见解并进行私下联系。
   - 这种协作精神旨在增强对更有效的 AI Alignment 策略的集体贡献。

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **SQLite 全文搜索增强**：一场新的见面会将探讨如何将 **SQLite 内置的全文搜索引擎**与 [sqlite-vec](https://discord.com/events/1089876418936180786/1284180345553551431) 结合，以提高效率。
   - 本次会议承诺提供更**完整和准确的搜索**结果，迎合寻求高效搜索能力的开发者。
- **Mozilla 启动 AI Builders Accelerator**：Mozilla 首届 **AI Builders Accelerator 班次**已宣布并将很快启动。
   - 计划详情可以在[这里](https://discord.com/channels/1089876418936180786/1245083732319408195/1287802832417718325)找到，旨在支持前沿的 AI 项目。
- **SoraSNS：一个新的 Fediverse 客户端**：一位前 Apple 工程师发布了 **SoraSNS**，这是一个集成 [local AI](https://discord.com/events/1089876418936180786/1277835047084363827) 以学习用户兴趣的 Fediverse 客户端。
   - 该客户端旨在通过提供自适应的 **'For You' 时间线**来增强用户体验。
- **开源 AI 应对挑战**：Mark Surman 在 The New Stack 中强调，讨论**定义开源 AI** 的潜力，以应对该领域的各种挑战。
   - 对话强调了此类定义如何帮助开发者和组织[解决无数令人头疼的问题](https://discord.com/channels/1089876418936180786/1287810294126481498)。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **BFCL V3 重塑 LLM 评估**：**Berkeley Function-Calling Leaderboard (BFCL) V3** 引入了一种全新的评估方法，用于评估**多轮 (multi-turn)** 函数调用，增强了 Agent 系统能力。
   - 此版本允许模型管理复杂的交互，这对于 LLM 执行复杂任务至关重要。
- **状态管理是必须的**：LLM 中的**状态管理 (State Management)** 至关重要，它使系统能够验证任务结果，例如检查**股票购买**是否成功。
   - 这突显了在任务执行后，通过 API 进行内部状态查询的关键性。
- **告别短上下文模型**：随着 BFCL V3 的发布，不鼓励依赖**短上下文模型**，因为任务需要更广泛的上下文才能有效执行。
   - 这对于复杂任务尤为关键，例如对**数百个文件**进行排序。
- **排行榜设定新标准**：在社区见解的推动下，BFCL V3 为评估 LLM 功能（特别是函数调用）建立了**金标准**。
   - 这反映了与**企业**和**开源贡献者**持续合作以完善评估实践。
- **深入探讨 BFCL V3 性能**：一篇新的博客文章详细介绍了 BFCL V3 评估方法，讨论了如何在实际应用中评估模型的**成本**和**延迟**。
   - 欲了解更多见解，请查看完整文章：[Berkeley Function Calling Blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：详细的频道摘要和链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1286764037429923941)** (603 条消息🔥🔥🔥): 

> - `HuggingFace Spaces 停机`
> - `模型微调`
> - `AI 工具和库`
> - `Serverless API 使用`
> - `ExtractCode 投票支持`

- **HuggingFace Spaces 遭遇宕机**：用户报告了 HuggingFace Spaces 宕机的问题，遇到了诸如 '500 Internal Error' 以及文件上传困难等错误。
   - 宕机持续了数小时，导致尝试访问模型或上传内容的用户感到沮丧。
- **模型 Fine-Tuning 指导**：一位用户寻求帮助，希望对模型进行 Fine-Tuning，使其能够严格根据包含 350 条关于操作系统、软件和硬件问题的记录的数据集进行回答。
   - 其他用户通过分享 YouTube 视频等资源提供了帮助，并推荐了 SimpleTuner 等工具用于训练模型。
- **探索 AI 工具和库**：多位用户讨论了用于模型 Fine-Tuning 的工具，推荐了包括 SimpleTuner、Kohya-Trainer 和 Onetrainer 在内的易用工具。
   - 讨论强调了在使用这些库时的用户体验和面临的挑战，促进了协作学习。
- **Serverless API 见解**：讨论了来自 HuggingFace 的 Serverless Inference API，用户注意到某些 API 请求可以免费访问，以便测试和探索模型。
   - 鼓励用户尝试使用它，以便在无需管理基础设施的情况下实现轻松集成和快速原型设计。
- **AI 项目投票支持**：一位用户展示了他们的 AI 项目 ExtractCode，该项目旨在从 YouTube 视频中提取编程代码，并请求通过投票给予支持。
   - 鼓励参与者点击提供的链接以示支持，这体现了社区驱动的项目推广方式。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://stackoverflow.com/help/how-to-ask">如何提出一个好问题？ - 帮助中心</a>: Stack Overflow | 全球最大的开发者在线社区</li><li><a href="https://arxiv.org/abs/2409.12517">将 FP8 训练规模扩展至万亿级 Token 的 LLM</a>: 我们首次在高达 2 万亿 Token 的数据集上使用 FP8 精度训练大语言模型——比之前的限制提高了 20 倍。通过这些扩展的训练运行，我们发现了……</li><li><a href="https://huggingface.co/spaces/webml-community/remove-background-webgpu">Remove Background WebGPU - webml-community 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/dpo-trl">使用 DPO 微调 Llama 2</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/r3gm/Audio_separator">Audio🔹Separator - r3gm 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/ln_strike-gregzaj1-quant-quantitative-gif-22567558">Ln_strike Gregzaj1 GIF - Ln_strike Gregzaj1 Quant - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/crab-gif-26300412">螃蟹 GIF - 螃蟹 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/spaces/Suniilkumaar/AudioSep">AudioSep - Suniilkumaar 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/no-sleep-staying-up-insomnia-coffee-weak-gif-21941823">熬夜不睡觉 GIF - 熬夜不睡觉失眠 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/caos-bob-esponja-crisis-patricio-gif-23199341">混乱海绵宝宝 GIF - 混乱海绵宝宝 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/beaker-muppets-calm-relax-panic-gif-16222881">Beaker Muppets GIF - Beaker Muppets 冷静 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/AIatMeta/status/1836806535969968354?t=TpKrg6iwozJeumc8Kv-5ZQ&s=19">来自 AI at Meta (@AIatMeta) 的推文</a>: 碎片化的监管意味着欧盟面临错过开源和多模态 AI 领域快速创新的风险。我们正与来自 25 多家欧洲公司、研究人员和……的代表一起……</li><li><a href="https://tenor.com/view/burntdasbrot-kikalounge-burnt-toast-dance-gif-12556330">Burntdasbrot Kikalounge GIF - Burntdasbrot Kikalounge 烧焦 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/rowancheung/status/1838280020642676802">来自 Rowan Cheung (@rowancheung) 的推文</a>: 我刚刚完成了一次关于新的重大 AI 模型升级的独家采访。可以确认，明天对开发者来说将是一个重要的日子。禁令解除的那一刻，我将在 X 上发布完整对话……</li><li><a href="https://tenor.com/view/shame-on-you-gif-25797108">为你感到羞耻 GIF - 为你感到羞耻 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/anime-head-pat-pat-very-good-good-girl-gif-17187002">动漫摸头杀 GIF - 动漫摸头杀 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://ai.google.dev/competition/projects/extractcode">未找到标题</a>: 未找到描述</li><li><a href="https://www.nist.gov/news-events/events/2024/09/unleashing-ai-innovation-enabling-trust">释放 AI 创新，赋能信任</a>: 讨论 AI 测量和标准最新进展及后续步骤的研讨会</li><li><a href="https://www.anandtech.com/show/21425/intel-lunar-lake-architecture-deep-dive-lion-cove-xe2-and-npu4/4">英特尔揭晓 Lunar Lake 架构：全新的 P 核与 E 核、Xe2-LPG 图形核心，全新 NPU 4 带来更强 AI 性能</a>: 未找到描述</li><li><a href="https://tenor.com/view/doubt-press-x-la-noire-meme-x-button-gif-19259237">怀疑按 X GIF - 怀疑按 X 黑色洛城 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/baby-dont-hurt-me-mike-ohearn-jokester-joke-funny-gif-27699537">Baby Dont Hurt Me Mike Ohearn GIF - Baby Dont Hurt Me Mike Ohearn 搞笑者 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://en.m.wikipedia.org/wiki/Sigmoid_function">Sigmoid 函数 - 维基百科</a>: 未找到描述</li><li><a href="https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf">flux1-dev-Q4_K_S.gguf · city96/FLUX.1-dev-gguf (main 分支)</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Eiffel_Tower_replicas_and_derivatives">埃菲尔铁塔复制品及衍生品 - 维基百科</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=bo49U3iC7qY&ab_channel=TrelisResearch">在维基百科数据集上进行微调</a>: ➡️ 获取完整脚本（及未来改进）的终身访问权限：https://Trelis.com/ADVANCED-fine-tuning/ ➡️ 一键微调和 LLM 模板……</li><li><a href="https://huggingface.co/models?other=simple

tuner&sort=created">模型 - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/01-ai/Yi-9B-200K">01-ai/Yi-9B-200K · Hugging Face</a>: 未找到描述</li><li><a href="https://status.huggingface.co/">
Hugging Face 状态
</a>: 未找到描述</li><li><a href="https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX.md">SimpleTuner/documentation/quickstart/FLUX.md at main · bghira/SimpleTuner</a>: 一个面向扩散模型的通用微调工具包。 - bghira/SimpleTuner</li><li><a href="https://github.com/ostris/ai-toolkit">GitHub - ostris/ai-toolkit: 各种 AI 脚本。主要是 Stable Diffusion 相关内容。</a>: 各种 AI 脚本。主要是 Stable Diffusion 相关内容。 - ostris/ai-toolkit</li><li><a href="https://api-inference.huggingface.co">Serverless 推理 API</a>: 未找到描述</li><li><a href="https://github.com/marijnwijbenga/ai-music-learning-assistant-llm/tree/develop">GitHub - marijnwijbenga/ai-music-learning-assistant-llm at develop</a>: 一个仅限于音乐话题的 AI 学习助手 LLM 聊天机器人，在音乐理论和音乐教学上进行了微调 - GitHub - marijnwijbenga/ai-music-learning-assistant-llm at develop
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1286872453838147594)** (8 messages🔥): 

> - `Centroidal Triplet Loss`
> - `Mamba-2 架构`
> - `BFGS 算法`
> - `Langchain 集成`
> - `混合精度损失` 


- **Centroidal Triplet Loss 已经存在**：一位成员发现他们“新颖”的想法 **Centroidal Triplet Loss** 已经作为 **Centroid Triplet Loss** 被开发出来了。
   - 他们还注意到一张几乎相同的图表，并正在探索一些可以增强该概念的修改。
- **Mamba-2 超越了其前身**：研究人员介绍了 [Mamba-2](https://vidrihmarko.medium.com/mamba-2-is-out-can-it-replace-transformers-6cfb3372ea39)，这是一种性能优于 **Mamba-1** 和 **Transformer++** 的状态空间模型。
   - 它旨在更好地处理信息密集型数据，其核心创新被称为 **Structured State Space Duality (SSD)**。
- **探索 BFGS 算法**：一位成员目前正在为一个副业项目研究 **BFGS 算法** 及其有限内存变体。
   - 他们欢迎其他具有这些算法经验的人提供意见，以增强他们的理解。
- **Langchain 将 LLM 连接到数据源**：另一位成员分享了他们学习 **Langchain** 如何将 LLM 与数据库和 API 集成以进行数据检索的兴奋之情。
   - 他们希望自己对 Langchain 能力的理解是正确的，并强调了其潜在的实用性。
- **1b FP8 匹配 bfloat16 精度**：一位成员指出，**1b FP8** 实现的损失与 **bfloat16 混合精度** 完全匹配。
   - 这一见解可能对模型训练效率和性能产生重大影响。



**提到的链接**：<a href="https://vidrihmarko.medium.com/mamba-2-is-out-can-it-replace-transformers-6cfb3372ea39">Mamba-2 发布：它能取代 Transformers 吗？</a>：Mamba-2：一种性能优于 Mamba 和 Transformer++ 的新型状态空间模型架构

  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1286766780190162964)** (11 条消息🔥): 

> - `3D Content Generation`
> - `Medical AI Research Insights`
> - `Open-Source AI Trends`
> - `Residual Networks`
> - `Taostats and Decentralized AI` 


- **10 秒内完成 3D Content Generation**：一位成员分享了一个 GitHub 仓库 [threestudio](https://github.com/threestudio-project/threestudio)，声称它可以在 10 秒内生成 3D 物体，并请求其他人进行尝试。
   - 另一位成员建议使用 'stable fast 3D' 作为替代方案，它可以在不到一秒的时间内从图像生成物体，并指出该工具已在 HF space 上线。
- **Medical AI 研究亮点**：一份简报重点介绍了本周 Medical AI 领域的关键论文和模型，其中重点关注了一篇名为 'How to Build the Virtual Cell with Artificial Intelligence' 的重要论文。
   - 讨论的其他关键主题包括各种医疗 LLM 和旨在利用 AI 技术增强诊断和临床试验的框架。
- **开源 AI 的采用率不断增长**：一篇文章强调了开发者对开源 AI 的快速接受，'2023 State of Open Source' 报告显示其使用量显著增加。
   - 文章列举了 10 个流行的开源 AI 框架，并讨论了重大技术投资推动这一趋势的影响。
- **怀念 Residual Networks**：一位成员分享了关于 Residual Networks 的里程碑式论文，提到了它在更有效地训练深度神经网络方面的贡献。
   - 该论文提供了在 ImageNet 上获得顶尖性能的实证证据，确立了 Residual Networks 作为深度学习领域重大进展的地位。
- **Taostats：去中心化 AI 分析**：Taostats 作为 Bittensor 的区块浏览器和分析平台出现，旨在促进机器学习的去中心化分析。
   - 该平台提供多种工具，包括 API 和用户友好型功能，支持去中心化 AI 应用的发展。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/1512.03385">Deep Residual Learning for Image Recognition</a>：更深的神经网络更难训练。我们提出了一个残差学习框架，以简化比以前使用的网络深得多的网络的训练。我们明确地重新...</li><li><a href="https://huggingface.co/Chunte/flux-lora-Huggieverse">Chunte/flux-lora-Huggieverse · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/OpenlifesciAI/status/1837688406014300514">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：Medical AI 上周回顾：顶级研究论文/模型 🏅（2024 年 9 月 14 日 - 9 月 21 日）🏅 本周 Medical AI 论文：如何利用人工智能构建虚拟细胞：优先级与 O...</li><li><a href="https://huggingface.co/posts/aaditya/492719281207772">Hugging Face 上的 @aaditya："Medical AI 上周回顾：顶级研究论文/模型 🏅（9 月 14 日 -…"</a>：未找到描述</li><li><a href="https://www.digitalocean.com/resources/articles/open-source-ai-platforms">10 个用于创新的开源 AI 平台 | DigitalOcean</a>：了解 10 个用于创新和协作的开源 AI 平台，以扩展您的业务</li><li><a href="https://fxtwitter.com/jsuarez5341/status/1830697672019476600">来自 Joseph Suarez (e/🐡) (@jsuarez5341) 的推文</a>：完整的 RL 冰山 - 强化学习的一切问题以及 PufferLib 如何修复它。加入我，一起潜入 RL 栈的 10 个层级。这里有适合初学者和专家的内容...</li><li><a href="https://taostats.io/">Taostats · Bittensor 网络区块浏览器、数据分析、API 和节点支持</a>：在 taostats.io 探索官方 Bittensor 区块链浏览器，这是您获取 metagraph 分析、TAO 代币数据和个性化仪表板的信赖来源。访问 API、RPC 服务等。</li><li><a href="https://github.com/threestudio-project/threestudio/tree/main">GitHub - threestudio-project/threestudio: 一个统一的 3D 内容生成框架。</a>：一个统一的 3D 内容生成框架。通过创建 GitHub 账户为 threestudio-project/threestudio 开发做出贡献。</li><li><a href="https://www.futuretools.io/">Future Tools - 找到满足您需求的精确 AI 工具</a>：FutureTools 收集并整理所有最好的 AI 工具，让您也能变得超乎常人！
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1286916757084180531)** (163 条消息🔥🔥): 

> - `OpenMusic Launch`
> - `Game Development with Bevy`
> - `Unity and Unreal Licensing Debate`
> - `AI-Powered RPG`
> - `Podcast Generation Technology`

- **OpenMusic 正式上线！**：用于文本生成音乐的 OpenMusic 现已在 Hugging Face Spaces 上可用，支持使用文本描述实时创作音乐。
   - 该项目利用了创新的 QA-MDT 论文，增强了音频质量和音乐性。
- **AI 驱动的 RPG 开发**：一位开发者正在开发一款带有 AI Agent 的 RPG 游戏，这些 Agent 可以模拟短期和长期记忆，并集成了物理引擎和网络功能。
   - 他们表达了对贡献的渴望，并指出了构建如此复杂系统所固有的挑战。
- **关于 Unity 和 Unreal 许可的辩论**：讨论强调了 Unity 和 Unreal Engine 由于其许可结构而具有的专有性质，尽管它们包含一些开源组件。
   - 参与者辩论了软件许可的影响，强调了游戏引擎的专有、开源以及各种许可模型之间的区别。
- **播客生成技术**：PodcastGen 利用先进技术生成受 Google NotebookLM 功能启发的播客，因其创新方法而备受关注。
   - 用户对其功能表示兴奋，尽管一些人注意到生成输出中可能存在内容重复的问题。
- **Rust 与 LLM 的接口连接**：一场对话探讨了在基于 Rust 的游戏开发框架 Bevy 中集成大语言模型（LLM），重点关注网络和实体交互。
   - 参与者就管理 NPC 任务以及游戏与 LLM 进程之间的通信提出了建议。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/jadechoghari/openmusic">jadechoghari/openmusic · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/krchickering/pokemon_generator">Pokémon Sprite Generator - a Hugging Face Space by krchickering</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/jadechoghari/OpenMusic">OpenMusic - a Hugging Face Space by jadechoghari</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/saq1b/podcastgen">PodcastGen - a Hugging Face Space by saq1b</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/nroggendorff/flux-lora-tester">FlUX.1 LoRA - a Hugging Face Space by nroggendorff</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/keras-nlp-integration">Announcing New Hugging Face and Keras NLP integration</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/JoPmt/Flux-schnell_CPU_Stable_Diffusion_cpp">Flux-schnell CPU Stable Diffusion Cpp - a Hugging Face Space by JoPmt</a>: 未找到描述</li><li><a href="https://github.com/Unity-Technologies/ml-agents/blob/develop/LICENSE.md">ml-agents/LICENSE.md at develop · Unity-Technologies/ml-agents</a>: Unity 机器学习代理工具包 (ML-Agents) 是一个开源项目，使游戏和模拟能够作为使用深度强化学习训练智能 Agent 的环境...</li><li><a href="https://www.kaggle.com/code/anhoangvo/run-comfy-gui-with-localtunnel-on-kaggle">Easy Run ComfyUI with GUI on Kaggle</a>: 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://www.kaggle.com/code/anhoangvo/generate-images-for-stories-using-llm-and-comfyui">Generate Images for stories using LLM and ComfyUI</a>: 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://youtu.be/5eo8nz_niiM">Over 200,000 Servers in One Place! Visiting Hetzner in Falkenstein (Germany)</a>: 更多关于 Hetzner 的信息：https://derbauer.hetzner.com/en/image-211013/---------------------------------------------------------在 Patreon 上支持我：https://...</li><li><a href="https://youtu.be/_saL1lounEE>,">I Installed my OWN Cloud Server! See What Happened Next...</a>: 你是否曾在深夜思考是哪台服务器运行着你的云实例，或者“裸金属云”到底是如何工作的？我们采用了一台全新的 Supermicro 第四代 In...</li><li><a href="https://github.com/Unity-Technologies/UnityCsReference">GitHub - Unity-Technologies/UnityCsReference: Unity C# reference source code.</a>: Unity C# 参考源代码。通过在 GitHub 上创建账号来为 Unity-Technologies/UnityCsReference 的开发做出贡献。</li><li><a href="https://github.com/slangerosuna/space_cowboy_rpg">GitHub - slangerosuna/space_cowboy_rpg: A sci-fantasy open-world shooter/rpg that replaces scripted dialogue with generative AI and has infinite content</a>: 一款科幻奇幻开放世界射击/RPG 游戏，用生成式 AI 取代了脚本对话，并拥有无限的内容 - slangerosuna/space_cowboy_rpg</li><li><a href="https://www.hetzner.com/de/dedicated-rootserver/matrix-gpu/">Dedicated Server Hosting</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1287017384137461801)** (29 messages🔥): 

> - `GUI 元素检测`
> - `GUI 自动化软件`
> - `用于界面识别的 AI`
> - `UIA 与 Android 无障碍功能`
> - `DOM 元素检索` 


- **GUI 元素检测的挑战**：一位成员对从截图中检测 GUI 元素以创建 **GUI 自动化软件**表示了兴趣，旨在识别交互元素及其 bounding boxes。
   - 另一位成员质疑了在所有界面上实现**通用检测 (generic detection)** 的可行性，原因是元素重叠以及不同设计带来的挑战。
- **关于界面检测复杂性的讨论**：贡献者们讨论了设计适用于**所有界面**的解决方案的复杂性，指出了界面缺乏清晰**按钮**或视觉提示的问题。
   - 他们指出，虽然 AI 可以发挥作用，但可能需要**先进技术**和定制模型才能获得有效结果。
- **参考历史自动化工具**：一位成员回忆起早期用于**扑克机**的自动化工具，强调了在涉及金钱时，人们在寻找自动化解决方案方面变得多么富有创造力。
   - 这场讨论展示了在涉及重大利益时**创新方法**的潜力，引发了关于问题解决创造力的对话。
- **GUI 检测的论文参考**：一位成员提到看到一篇提出 GUI 检测方法的论文，对比了**现代**与**传统方法**，但在使用对应的 GitHub 仓库时遇到了困难。
   - 这反映了该领域持续的探索，强调了可用的实现资源的重要性。
- **GUI 交互的替代方法**：原帖作者转向了一种更简单的方法，选择在 Windows 上使用 **UIA**，在 Android 上使用 **Android 无障碍功能 (accessibility)**，以及在 Web 应用中使用无头浏览器进行 **DOM 元素检索**。
   - 这种方法被认为是**可靠的**，表明了相比复杂的 AI 解决方案，开发者倾向于利用现有框架。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1286773018420318228)** (7 messages): 

> - `ST Embedding 中的不规则文本内容`
> - `更新的 Mamba 基准测试`
> - `LLM 与 WhatsApp 的集成`
> - `HuggingFace Hub 问题` 


- **ST Embedding 中不规则文本内容的挑战**：一位成员强调了不规则文本内容（如缺少空格的 'll'、'lrgt' 和 'dw'）的存在，并对 **ST embedding 流水线**如何处理此类情况表示担忧。
   - 他们质疑了对 'yes!do it' 等序列的处理方式，并指出目前缺乏能够有效处理这些情况的 embedding 模型。
- **询问更新的 Mamba 基准测试**：成员们询问自上次报告提到缺少权重以来，是否有任何**更新的 Mamba 基准测试**可用。
   - 最近提到的基准测试显示有所改进，但成员们由于数据不足而表示怀疑。
- **寻找 Python LLM 与 WhatsApp 的集成**：一位成员寻求将 **LLM 与 WhatsApp 集成**的项目仓库推荐，特别强调需要 Python 解决方案。
   - 据报告，之前尝试使用 WPPConnect 和 CrewAI 均未成功，目前正在寻找完全基于 Python 的方法。
- **对 HuggingFace Hub 性能的担忧**：一位成员报告了 **HuggingFace Hub** 的问题，表明可能存在停机或故障。
   - 未提供关于所面临问题的类型或程度的更多细节。



**提到的链接**：<a href="https://tenor.com/view/office-space-tps-gif-22666507">Office Space GIF - Office Space TPS - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1286788524527321151)** (7 messages): 

> - `Diffusion Models Discussion`
> - `Image Generator App with Flux.1-dev`
> - `ControlNet_Union Techniques` 


- **扩散模型讨论的正确频道**：一名成员澄清说，该频道专门用于讨论与 [Diffusion Models Course](https://github.com/huggingface/diffusion-models-class) 相关的话题，而不是为了 LLM。
   - 虽然偶尔会有混淆，但鼓励参与者专门关注 diffuser 相关话题。
- **使用 Flux.1-dev 构建图像生成应用**：另一名成员寻求关于使用最新的 **Flux.1-dev** 模型创建图像生成应用的指导，并提到在众多工具中需要理清思路。
   - 回复建议使用 **diffusers** 配合 **FastAPI** 和 **React** 来构建自定义托管解决方案。
- **ControlNet_Union 的严格输出**：一名成员分享了对 **SDXL** 的 **ControlNet_Union** 的担忧，理由是该模型在处理涂鸦（scribble）输入时会保留空白区域，而不是生成连贯的背景。
   - 建议关注所使用的 **control_type**，并指出 HED 在处理代表空白区域的黑色区域时具有更高的灵活性。
- **简化 ControlNet 输出的连贯性**：为了获得更好的背景生成效果，建议对输入图像进行修改，例如直接擦除图像的部分内容。
   - 鼓励使用这种技术来有效管理 fill/inpaint/outpaint 区域。



**Link mentioned**: <a href="https://github.com/huggingface/diffusion-models-class">GitHub - huggingface/diffusion-models-class: Materials for the Hugging Face Diffusion Models Course</a>: Materials for the Hugging Face Diffusion Models Course - huggingface/diffusion-models-class

  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1287889405935222856)** (1 messages): 

> - `Gradio 5 Beta Release`
> - `Performance Improvements`
> - `Modern Design Updates`
> - `AI Playground Feature`
> - `Security Enhancements` 


- **Gradio 5 Beta 来了！**：我们很高兴地宣布 **Gradio 5 (Beta)** 正式发布，旨在解决开发者频繁关注的问题。
   - 此版本引入了各种功能，以及显著的性能升级和现代设计改进。
- **重大性能提升**：**Gradio 5** 包含重大的性能增强，特别是服务器端渲染（SSR），从而使 Gradio 应用的加载速度大幅提升。
   - 开发者可以期待更流畅的浏览器体验，解决了之前关于 **loading speed** 的投诉。
- **焕然一新的现代设计**：响应用户反馈，**Gradio 5** 中的许多 UI 组件（如 **Buttons** 和 **Sliders**）都进行了现代设计更新。
   - 团队邀请用户在 Gradio 5 最终正式发布前提供反馈。
- **引入用于实验的 AI Playground**：Gradio 5 引入了实验性的 **AI Playground**，使用户能够直接在浏览器中生成和预览 Gradio 应用：[Playground link](https://5-0-dev.gradio-website.pages.dev/playground)。
   - 该功能包含多种应用模板，如 **Sentence Builder** 和 **Stock Forecast** 供用户探索。
- **Gradio 5 增强的安全措施**：该版本通过了第三方审计，确保 Gradio 已准备好用于生产环境，从而提高了 **security**。
   - 流媒体能力也得到了增强，使得创建 **realtime Gradio apps** 更加容易。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://5-0-dev.gradio-website.pages.dev/playground">Gradio Playground</a>: Play Around with Gradio Demos</li><li><a href="https://huggingface2.notion.site/Gradio-5-A-Production-Ready-Web-Framework-for-ML-Applications-a4d7e42c26f4450aa0758d968019d120?pvs=74)">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1287150108009431140)** (1 条消息): 

> - `Aider v0.57.0`
> - `OpenAI o1 models support` (OpenAI o1 模型支持)
> - `Windows compatibility` (Windows 兼容性)
> - `New Cohere models` (新的 Cohere 模型)
> - `Bug fixes` (Bug 修复)


- **Aider v0.57.0 发布并带来新功能**：**Aider v0.57.0** 的发布引入了对 **OpenAI o1 模型**的支持，通过 diff 编辑格式和 SOTA 排行榜结果提升了性能。
   - 值得注意的是，**Aider** 自身编写了此版本 **70%** 的代码，展示了其自给自足的能力。
- **改进的 Windows 兼容性**：在 **Windows** 上，`/run` 命令现在可以正确使用 **PowerShell** 或 **cmd.exe**，提升了用户体验。
   - 用户还可以期待在 **--no-pretty** 激活或使用 Windows 控制台时，回退到简单的 `input()` 提示符，从而提高可访问性。
- **集成新的 Cohere 模型**：Aider 现在支持由 @jalammar 宣布的新 **08-2024 Cohere 模型**，扩展了工具的通用性。
   - 此次更新允许使用 **/read-only** 命令递归添加目录，简化了工作流。
- **通过 Bug 修复增强性能**：应用了大量修复以解决边缘情况下的崩溃问题，并改进了 Prompt 缓存分块策略。
   - 更新还包含启动时对 Git 仓库的精细完整性检查，确保运行稳健。
- **提供完整的变更日志**：有关更改的详细概述，用户可以参考 [aider.chat/HISTORY.html](https://aider.chat/HISTORY.html) 上的完整 **变更日志 (change log)**。
   - 该日志列出了近期更新中引入的所有新功能、改进和修复。



**提到的链接**：<a href="https://aider.chat/HISTORY.html">发布历史</a>：关于 Aider 编写自身代码的发布说明和统计数据。

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1286770309483073659)** (513 条消息🔥🔥🔥): 

> - `Using Aider with OpenRouter and Claude models` (在 OpenRouter 和 Claude 模型中使用 Aider)
> - `Challenges with DeepSeek and Sonnet models` (DeepSeek 和 Sonnet 模型面临的挑战)
> - `Experiences with o1 models` (o1 模型的使用体验)
> - `Issues with Anthropic services` (Anthropic 服务的问题)
> - `Contributions to Aider and coding workflow` (对 Aider 的贡献和编码工作流)


- **探索 Aider 和 OpenRouter 模型**：用户反馈了在 Aider 中使用 o1 模型的复杂体验，提到了频繁的“过载”错误以及直接查询 Claude 模型时的困惑。
   - 虽然一些用户成功通过 OpenRouter 访问 Anthropic 模型，但其他用户仍面临持续的问题，表明服务可能存在不稳定性。
- **DeepSeek 与 Sonnet 模型对比**：一些用户发现 DeepSeek 的表现优于 Sonnet，尤其是在避免代码补全过程中的循环错误方面。
   - 关于这些模型的讨论表明，用户更青睐 DeepSeek 的执行能力，而 Sonnet 则在分析强度上更具优势。
- **对新 AI 模型的期待**：用户对 Opus 3.5 的潜在发布充满期待，并推测其与现有模型相比的能力。
   - 对话显示出用户对可能显著提升开发者生产力的功能进步感到普遍兴奋和希望。
- **Aider 中的错误管理**：用户经常遇到 o1 模型响应错误或使用非预期语言的问题，促使一些人修改其 Prompt。
   - 虽然有人建议添加系统提示词 (System Prompts)，但效果似乎有限，导致用户对模型的可靠性感到沮丧。
- **为 Aider 贡献代码**：用户寻求有关贡献 Aider 的指导，讨论了贡献指南和最佳实践的重要性。
   - 随着对特定文件的只读访问等新功能的引入，社区对管理和增强 Aider 功能的支持正在上升。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/rowancheung/status/1838280020642676802">Rowan Cheung (@rowancheung) 的推文</a>：我刚刚完成了一次关于新的重大 AI 模型升级的独家专访。可以确认，明天对开发者来说将是一个大日子。禁令解除的那一刻，我将在 X 上发布完整对话...</li><li><a href="https://voideditor.com/">Void</a>：Void 是一个开源的 Cursor 替代方案。完全隐私，功能齐全。</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">文件编辑问题</a>：aider 是你终端中的 AI 配对编程</li><li><a href="https://aider.chat/docs/usage/browser.html">浏览器中的 Aider</a>：Aider 可以在浏览器中运行，而不仅仅是在命令行中。</li><li><a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-aider-uses">FAQ</a>：关于 aider 的常见问题解答。</li><li><a href="https://aider.chat/docs/usage/conventions.html">指定编码规范</a>：让 aider 在处理代码时遵循你的编码规范。</li><li><a href="https://aider.chat/docs/install/install.html">安装 aider</a>：aider 是你终端中的 AI 配对编程</li><li><a href="https://aider.chat/2024/08/14/code-in-json.html">LLM 不擅长以 JSON 格式返回代码</a>：如果你要求 LLM 通过工具函数调用返回包装在 JSON 中的代码，它们编写的代码质量会变差。</li><li><a href="https://console.groq.com/settings/limits">GroqCloud</a>：体验全球最快的推理速度</li><li><a href="https://draftjs.org/docs/getting-started">概览 | Draft.js</a>：Draft.js 是一个在 React 中构建富文本编辑器的框架，由不可变模型驱动，并抽象了跨浏览器差异。</li><li><a href="https://tenor.com/view/side-eye-cat-gif-8216273864367202904">侧目猫 GIF - 侧目猫 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://status.anthropic.com/incidents/xts3kyr0nrx1">Claude 3.5 Sonnet 错误率升高</a>：未找到描述</li><li><a href="https://trypear.ai/">PearAI - 用于快速开发的开源 AI 代码编辑器</a>：PearAI 是一款开源的 AI 驱动代码编辑器，具有 AI 聊天、行内提示和调试等功能，可加速你的编码过程。</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>：关于 aider 的常见问题解答。</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>：未找到描述</li><li><a href="https://www.instagram.com/reel/DAGuPUpP6gG/?igsh=eXBpczA3b3g5MnBx">Leon Si 在 Instagram 上发布</a>："到这一步我们还算是开发者吗？🥲 #tech #programming #code #ai"：19.6 万次点赞，2,260 条评论 - leonsilicon 于 2024 年 9 月 19 日发布："到这一步我们还算是开发者吗？🥲 #tech #programming #code #ai"。</li><li><a href="https://aider.chat/docs/llms/warnings.html">模型警告</a>：aider 是你终端中的 AI 配对编程</li><li><a href="https://tenor.com/view/its-just-gambling-liam-scott-edwards-ace-trainer-liam-betting-gamble-gif-20475304">这只是赌博 Liam Scott Edwards GIF - 这只是赌博 Liam Scott Edwards Ace Trainer Liam - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.blog/news-insights/product-news/try-out-openai-o1-in-github-copilot-and-models/">在 GitHub Copilot 和 Models 中试用 OpenAI o1</a>：OpenAI o1-preview 和 o1-mini 现已在 VS Code 的 GitHub Copilot Chat 和 GitHub Models 游乐场中可用。</li><li><a href="https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis">基于 Qwen 2 Instruct 的中国 LLM 审查与偏见分析</a>：未找到描述</li><li><a href="https://x.com/alexalbert__/status/1836447593888649646?s=61">Alex Albert (@alexalbert__) 的推文</a>：我最喜欢的 @AnthropicAI API 功能之一是 prompt prefilling，但人们似乎并不了解。你的 API 请求不必以 'user' 轮次结束。你可以包含一个...</li><li><a href="https://www.marscode.com/">MarsCode - AI IDE</a>：MarsCode 提供了一个内置 AI 助手和扩展的 IDE，支持 100 多种语言和主流 IDE。</li><li><a href="https://pieces.app/">Pieces for Developers - 你的工作流 Copilot</a>：集成你的工具链，高效捕获、丰富和重用材料。在设备端 Copilot 的协助下增强协作。</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/CONTRIBUTING.md">aider/CONTRIBUTING.md at main · paul-gauthier/aider</a>：aider 是你终端中的 AI 配对编程。通过在 GitHub 上创建账户，为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/o1-waitlist-signup?utm_campaign=GitHub_Blog">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并贡献于...</li>

超过 4.2 亿个项目。</li><li><a href="https://x.com/leonsilicon/status/1837129318306304394?s=46&t=ZSMBWlGirVJCSoNbnkJuPQ">Leon Si (@leonsilicon) 的推文</a>：开发者们完蛋了</li><li><a href="https://www.youtube.com/watch?v=XAeKtyL2m-Q&t=169s">采访初级产品经理 [初创公司]</a>：产品经理 [初创公司] 第二部分，本周在 https://www.patreon.com/ProgrammersAreAlsoHuman 喝杯咖啡，采访初级产品经理 Josh D...</li><li><a href="https://x.com/leonsilicon/status/1837129318306304394?s=46&t=ZSMB">Leon Si (@leonsilicon) 的推文</a>：开发者们完蛋了</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/aider/prompts.py">aider/aider/prompts.py at main · paul-gauthier/aider</a>：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/voideditor/void">GitHub - voideditor/void</a>：通过在 GitHub 上创建账号来为 voideditor/void 的开发做出贡献。</li><li><a href="https://github.com/PierrunoYT/awesome-ai-dev-tools?">GitHub - PierrunoYT/awesome-ai-dev-tools：一个精选的强大且创新的开发工具列表，包括代码编辑器、插件和生产力增强工具。该仓库旨在为寻求优化工作流和提高效率的开发者提供全面的资源。从 IDE 到命令行实用程序，寻找能将你的编码提升到新水平的工具</a>：一个精选的强大且创新的开发工具列表，包括代码编辑器、插件和生产力增强工具。该仓库旨在为开发者提供全面的资源...</li><li><a href="https://github.com/paul-gauthier/aider/pull/1176">通过 glob 模式实现 /read-only，由 akaihola 提交 · Pull Request #1176 · paul-gauthier/aider</a>：工作进行中 —— 基础用例已验证，复杂场景需要测试。此补丁修改了 /read-only 命令，使其像 /add 一样支持目录和 glob 模式。一个目录...</li><li><a href="https://github.com/PierrunoYT/photo-location-finder">GitHub - PierrunoYT/photo-location-finder：该程序允许用户使用 Google Cloud Vision API 检测图像中的地标。程序会提示用户输入图像路径、API 密钥和凭据，以便通过 Google Cloud API 进行身份验证。</a>：该程序允许用户使用 Google Cloud Vision API 检测图像中的地标。程序会提示用户输入图像路径、API 密钥和凭据，以便通过 Go...</li><li><a href="https://cursor.directory/">Cursor Directory</a>：为你的框架和语言寻找最佳的 Cursor 规则</li><li><a href="https://github.com/PierrunoYT/awesome-ai-dev-tools">GitHub - PierrunoYT/awesome-ai-dev-tools：一个精选的强大且创新的开发工具列表，包括代码编辑器、插件和生产力增强工具。该仓库旨在为寻求优化工作流和提高效率的开发者提供全面的资源。从 IDE 到命令行实用程序，寻找能将你的编码提升到新水平的工具</a>：一个精选的强大且创新的开发工具列表，包括代码编辑器、插件和生产力增强工具。该仓库旨在为开发者提供全面的资源...</li><li><a href="https://platform.deepseek.com/api-docs/updates/#version-2024-09-05">更新日志 | DeepSeek API 文档</a>：版本：2024-09-05</li><li><a href="https://github.com/PierrunoYT/awesome-dev-tools">GitHub - PierrunoYT/awesome-ai-dev-tools：一个精选的强大且创新的开发工具列表，包括代码编辑器、插件和生产力增强工具。该仓库旨在为寻求优化工作流和提高效率的开发者提供全面的资源。从 IDE 到命令行实用程序，寻找能将你的编码提升到新水平的工具</a>：一个精选的强大且创新的开发工具列表，包括代码编辑器、插件和生产力增强工具。该仓库旨在为开发者提供全面的资源...</li><li><a href="https://cloudonair.withgoogle.com/events/gemini-at-work-24">Gemini at Work</a>：加入 Google Cloud CEO Thomas Kurian 和行业领袖，共同探索 AI 如何重塑全球业务。</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>：ChatGPT 的所有前端 GUI 客户端。通过在 GitHub 上创建账号来为 billmei/every-chatgpt-gui 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/commit/7fa1620f58132ec085a7939a8015bbe7935827a2">feat: 允许在 SEARCH/REPLACE 块前缀中灵活匹配 5-9 个字符 · paul-gauthier/aider@7fa1620</a>：…块前缀
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1286784404114636812)** (167 条消息🔥🔥): 

> - `Aider Functionality` (Aider 功能)
> - `GitHub Integration with Aider` (Aider 与 GitHub 集成)
> - `Chat History Handling` (聊天历史处理)
> - `Repository Map Optimization` (仓库地图优化)
> - `Usage of Aider with Local Models` (在本地模型上使用 Aider)


- **Aider 的仓库地图（repository map）和聊天历史**：Aider 可以维护整个 git 仓库的简洁地图，这有助于在每次更改请求时向 LLM 发送更新，同时便于理解代码更改和关系。
   - 使用 Aider 时，如果你想防止仓库地图自动更新，可以使用 `--map-refresh manual` 运行，但在添加新文件时可能需要进行完整刷新。
- **通过手动控制使用 Aider**：为了优化 Aider 的性能，建议在启动时限制仓库地图的 token 数量，因为过多的信息可能会使 LLM 产生困惑。
   - 将 `--map-tokens` 设置为 2048 通常是可以接受的，但使用较低的数值（如 1024）可能会让模型获得更好的清晰度。
- **Aider 与文档的集成**：Aider 可以与许多 Markdown 文档配合使用，允许你添加特定文件进行审查并询问有关其内容的问题。
   - 然而，Aider 主要不是一个文档挖掘工具，使用它从大量文档中提取信息可能不是它的强项。
- **对本地和外部模型的支持**：Aider 旨在与多个本地模型配合工作，并支持各种外部 API，不过较新版本需要 Python 3.9 或更高版本。
   - 此外，Aider 可以连接到 HuggingFace 模型，并利用 LiteLLM 来简化与可用模型的交互。
- **使用环境变量和配置**：用户可以使用 `.env` 文件配置 Aider，以管理不同环境的设置，从而保持其设置在不同机器间的可移植性。
   - 建议在配置文件（如 `CONVENTIONS.md`）中使用符号引用，以避免硬编码路径。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/repomap.html#optimizing-the-map">Repository map</a>: Aider 使用 Git 仓库地图为 LLM 提供代码上下文。</li><li><a href="https://aider.chat/docs/git.html">Git integration</a>: Aider 与 Git 紧密集成。</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>: 关于 Aider 的常见问题。</li><li><a href="https://aider.chat/docs/faq.html#how-do-i-turn-on-the-repository-map">FAQ</a>: 关于 Aider 的常见问题。</li><li><a href="https://huggingface.co/chat/models">HuggingChat - Models</a>: 浏览 HuggingChat 可用模型</li><li><a href="https://aider.chat/docs/troubleshooting/imports.html#replit">Dependency versions</a>: Aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/config/options.html#--map-refresh-value">Options reference</a>: 关于 Aider 所有设置的详细信息。</li><li><a href="https://aider.chat/docs/config/dotenv.html">Config with .env</a>: 使用 .env 文件为 Aider 存储 LLM API 密钥。</li><li><a href="https://aider.chat/docs/llms.html">Connecting to LLMs</a>: Aider 可以连接大多数 LLM 进行 AI 结对编程。</li><li><a href="https://aider.chat/examples/README.html">Example chat transcripts</a>: Aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/usage/tutorials.html">Tutorial videos</a>: 由 Aider 用户制作的介绍和教程视频。</li><li><a href="https://docs.litellm.ai/docs/providers/huggingface">Huggingface | liteLLM</a>: LiteLLM 支持以下类型的 Hugging Face 模型：</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401">401 Unauthorized - HTTP | MDN</a>: HTTP 401 Unauthorized 客户端错误响应状态码表示请求未成功，因为它缺少所请求资源的有效身份验证凭据。此状态码...</li><li><a href="https://github.com/paul-gauthier/aider/blob/a4f608f3dd579c561d15cda3f06e785973cb1261/aider/commands.py#L1087)">aider/aider/commands.py at a4f608f3dd579c561d15cda3f06e785973cb1261 · paul-gauthier/aider</a>: Aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/larryhudson/aider-switchcoder-debugging">GitHub - larryhudson/aider-switchcoder-debugging</a>: 通过在 GitHub 上创建账号来为 larryhudson/aider-switchcoder-debugging 的开发做出贡献。</li><li><a href="https://aider.chat/docs/config/options.html">Options reference</a>: 关于 Aider 所有设置的详细信息。</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: 你可以通过命令行或 Python 对 Aider 进行脚本化操作。</li><li><a href="https://github.com/All-Hands-AI/OpenHands">GitHub - All-Hands-AI/OpenHands: 🙌 OpenHands: Code Less, Make More</a>: 🙌 OpenHands: 少写代码，多产出。通过在 GitHub 上创建账号来为 All-Hands-AI/OpenHands 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/blob/54cfbc4142e10dde73434accd20761bfc1ba3f1e/aider/main.py#L714-L723)">aider/aider/main.py at 54cfbc4142e10dde73434accd20761bfc1ba3f1e · paul-gauthier/aider</a>: Aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/blob/cee0bb713568539ecf97b6494f087cc7ddcf926b/aider/main.py#L714">aider/aider/main.py at cee0bb713568539ecf97b6494f087cc7ddcf926b · paul-gauthier/aider</a>: Aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1286782184220397639)** (9 条消息🔥): 

> - `Aider 工具开发`
> - `Embeddings 与 RAG`
> - `SmartPoi 固件的 Flask 应用`
> - `GitHub 转 RSS 代理`
> - `Claude 与手动搜索` 


- **使用 Aider 简化工具开发**：一位成员表示，他曾浪费数小时为 Aider 添加功能，而实际上简单的磁盘搜索就足够了，他意识到使用“manual-mode”能更快产生更好的结果。
   - 他们提议利用 **ripgrep** 为 **aider 集成** 构建一个快速搜索工具，以简化流程。
- **关于 Embeddings 被高估的争论**：一位用户认为 Embeddings 被高估了，主张使用工具和章节摘要的 DIY 方法来替代标准的 Embeddings，并将该方法比作类似于 **llama index** 的树状结构。
   - 他们幽默地暗示，Embeddings 的流行是由 **VC 融资** 驱动的，从而创造了一个充斥着 Embedding RAG 教程的市场。
- **通过 Flask 应用简化 AI 编程**：一位成员分享了他们仅使用免费 LLM 为 **SmartPoi Arduino Firmware 项目** 创建 Flask 应用的经验，强调 AI 编程可以非常具有成本效益。
   - 他们指出，虽然免费 LLM 可能速度较慢且偶尔出错，但结果令人满意，目前正在考虑对免费和付费 AI 模型进行对比。
- **GitHub Issues 转换为 RSS 订阅源**：一位用户介绍了一个提供 **GitHub 转 RSS 代理** 的 GitHub 仓库，允许用户将 GitHub issues 和 PR 转换为 RSS 订阅源。
   - 该解决方案被认为对于监控项目特别有用，且不会受到通知邮件的骚扰。
- **围绕 RAG 的误解**：一位成员赞同 Embeddings 被高估的观点，认为对 **RAG** 工具的广泛投资是由于对 Agent 行为缺乏理解。
   - 这与目前关于现有 AI 方法论与替代方案有效性的持续讨论相一致。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.circusscientist.com/2024/09/19/smartpoi-firmware-downloader-made-with-ai/">SmartPoi Firmware Downloader - made with AI - Circus Scientist</a>: 我使用 Aider（AI 编程助手）和免费 LLM 从零开始制作了一个 Flask 应用。这是为了 SmartPoi Arduino Firmware 项目 —— POV Poi，现在比以往任何时候都更容易使用...</li><li><a href="https://github.com/meain/gh-issues-to-rss">GitHub - meain/gh-issues-to-rss: Convert github issues and prs into rss feed</a>: 将 GitHub issues 和 PR 转换为 RSS 订阅源。通过在 GitHub 上创建账户为 meain/gh-issues-to-rss 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1287766821117628457)** (1 条消息): 

> - `μ-parameterization 指南`
> - `Cerebras 合作`
> - `nanoGPT 实现`
> - `GPT-NeoX 集成` 


- **与 Cerebras 联合发布 μ-Parameterization 指南**：今天，我们很高兴发布一篇关于 [The Practitioner's Guide to the Maximal Update Parameterization](https://blog.eleuther.ai/mutransfer/) 的联合博客，旨在提高训练社区对 **μ-parameterization** (μP) 的可获得性。
   - 该指南包括 **逐步实现指令** 以及在 [EleutherAI/nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup) 上的简单实现，解决了原始材料中常见的易用性问题。
- **广泛采用 μP 的益处**：指南强调，广泛采用 μP 可以减少训练期间的**不稳定性**，并降低超参数优化所需的计算量。
   - 此外，它指出 μP 能够实现不同训练方法之间**更稳健的比较**，从而促进更好的研究成果。
- **简化的 μP 实现特性**：指南简化了 μP 概念，并包含了 μP 实现的关键**验证**步骤，涉及 coord-checks 和完整的 LR transfer。
   - 这种细致入微的方法使从业者更容易掌握核心概念，而不会陷入复杂性中。
- **GPT-NeoX 中 μP 的未来**：受该指南实现的启发，我们将把这种简化的 μP 集成到**即将发布的 GPT-NeoX 3.0 版本**中。
   - 有关这些工作的持续更新和追踪可以在 [GPT-NeoX 仓库](https://github.com/EleutherAI/gpt-neox/pull/1087)中找到。



**提到的链接**: <a href="https://github.com/EleutherAI/gpt-neox/pull/1087.">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1287445602279358515)** (379 条消息🔥🔥): 

> - `GPT-4 评估中的 Cosine Similarity`
> - `测试集泄漏（Test Set Leakage）担忧`
> - `理解 RWKV 架构`
> - `Maximal Update Parametrization (muP)`
> - `JAX 与 Pytorch 中的优化器代码复杂度` 


- **在 GPT-4 中使用 Cosine Similarity**：一位用户正在评估 GPT-4 在无需 fine-tuning 的情况下的分类任务表现，考虑根据来自测试集的 Cosine Similarity 动态选择示例，以改进 In-context Learning。
   - 有人担心在 Prompt 中包含类似的测试示例可能会导致测试集泄漏（Test Set Leakage），尽管确保了测试问题本身不被包含在内。
- **评估测试集泄漏风险**：一位成员对将测试集作为选择 In-context 示例的池子所带来的测试集泄漏风险表示担忧。
   - 有人指出，虽然选择过程可能不会直接包含测试示例，但相似性可能会导致间接泄漏，从而影响评估的有效性。
- **理解 RWKV 架构的挑战**：参与者讨论了 RWKV 架构的复杂性，指出许多人发现很难掌握它与其他模型（如 GLA）的区别和相似之处。
   - 有人建议简化的解释有助于更好地理解，但现有的资源可能仍然显得晦涩或复杂。
- **简化 Maximal Update Parametrization**：讨论强调了对 Maximal Update Parametrization (muP) 进行通俗易懂解释的需求，以便在机器学习框架中更好地理解和使用。
   - 提到了一篇旨在揭开 muP 神秘面纱的博客文章，使其在不深入复杂理论方面的情况下更加平易近人。
- **JAX 与 Pytorch 中的优化器代码复杂度**：参与者辩论了 JAX 与 Pytorch 中 Shampoo 实现的相对复杂度，对于哪种更简单或更直观意见不一。
   - 有人指出，虽然 JAX 可能通过其 API 提供更多的灵活性，但其实现可能比更简洁的 Pytorch 替代方案更加冗长和复杂。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/cloneofsimo/status/1836114264215687601?s=46">来自 Simo Ryu (@cloneofsimo) 的推文</a>：修正后的数据</li><li><a href="https://arxiv.org/abs/2407.17465">u-$μ$P: 单位缩放的最大更新参数化 (The Unit-Scaled Maximal Update Parametrization)</a>：最大更新参数化 ($μ$P) 旨在使模型的最佳超参数 (HPs) 与其大小无关，从而允许使用廉价的代理模型而非全尺寸模型来搜索超参数...</li><li><a href="https://x.com/jxbz">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2407.05872">跨参数化和优化器的缩放指数 (Scaling Exponents Across Parameterizations and Optimizers)</a>：将模型从窄宽度稳健且有效地缩放到宽宽度，通常需要精确调整许多算法和架构细节，例如参数化和优化器的选择...</li><li><a href="https://arxiv.org/abs/2101.06804">什么构成了 GPT-$3$ 良好的上下文示例？</a>：GPT-$3$ 因其在广泛的 NLP 任务中的卓越表现而备受关注，尤其是其强大且多功能的上下文少样本学习能力。尽管其...</li><li><a href="https://arxiv.org/abs/2405.14813">模块化范数下的可扩展优化 (Scalable Optimization in the Modular Norm)</a>：为了提高当代深度学习的性能，人们倾向于在层数和层大小两个方面扩展神经网络。当增加单个层的宽度时...</li><li><a href="https://arxiv.org/abs/2310.17813">特征学习的光谱条件 (A Spectral Condition for Feature Learning)</a>：训练越来越大的神经网络的压力促使了对大网络宽度下的初始化和训练的研究。一个关键挑战是如何缩放训练，使网络的内部表示...</li><li><a href="https://x.com/cloneofsimo/status/1838287517906510026">来自 Simo Ryu (@cloneofsimo) 的推文</a>：好东西。专业提示：按照我勾选的红圈操作能让你完成 99%。（但不要缩放 head dim）https://blog.eleuther.ai/mutransfer/</li><li><a href="https://jeremybernste.in/modula/bad-scaling/">糟糕的缩放 (Bad scaling)</a>：在最简单的层面上，神经网络通过迭代以下操作进行训练：其中 learning_rate 是一个浮点数，gradient 是损失函数相对于权重的梯度...</li><li><a href="https://huggingface.co/blog/rwkv">RWKV 介绍 - 具有 Transformer 优点的 RNN</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=_NC9sc-nXoc">Session 2A &amp; 2B：非凸优化 (Optimization Non Convex)</a>：在此处观看带有 AI 生成目录 (ToC)、短语云和视频内搜索的视频：https://videos.videoken.com/index.php/videos/icml-2018-sessi...</li><li><a href="https://x.co">出售域名 | 购买域名 | 停放域名</a>：未找到描述</li><li><a href="https://github.com/cloneofsimo/zeroshampoo/blob/main/distributed_shampoo.py">zeroshampoo/distributed_shampoo.py (位于 main 分支) · cloneofsimo/zeroshampoo</a>：通过在 GitHub 上创建账户来为 cloneofsimo/zeroshampoo 的开发做出贡献。</li><li><a href="https://github.com/google-research/google-research/blob/master/scalable_shampoo/jax/shampoo.py">google-research/scalable_shampoo/jax/shampoo.py (位于 master 分支) · google-research/google-research</a>：Google Research。通过在 GitHub 上创建账户来为 google-research/google-research 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1286814968624320563)** (206 条消息🔥🔥): 

> - `AI 中的课程学习 (Curriculum Learning)`
> - `AI 模型的解释性 (Interpretability)`
> - `LLM 的规划能力 (Planning Abilities)`
> - `大语言模型 (LLM) 的性能`
> - `可解释 AI (Explainable AI) 的评估`

- **关于课程学习（Curriculum Learning）有效性的辩论**：关于 AI 中课程学习（CL）有效性的讨论正在进行中，一些人认为它可能不会比传统训练方法带来显著改进。
   - 成员们对 CL 在实际应用中的影响表示怀疑，理由是目前还没有保证数据过滤的最佳实践。
- **OpenAI 新型大推理模型（Large Reasoning Model）的宣称**：OpenAI 最近推出的被称为大推理模型（LRM）的模型，声称摆脱了传统自回归 LLMs 的局限性，引发了对其与现有模型性能对比的关注。
   - 然而，一些成员对 LRM 的区分度表示质疑，并指出通过现有方法在高计算成本下也可以实现类似的改进。
- **对 AI 可解释性的怀疑**：一位成员引用了一篇讨论 AI 可解释性方法缺点的论文，指出许多方法并不能为人类决策提供有意义的见解。
   - 研究结果表明，典型的特征归因解释可能会由于认知偏差导致更差的决策结果，挑战了关于其普遍益处的假设。
- **人类性能基准（Benchmarks）**：讨论强调了将 AI 性能与人类能力进行比较的基准，并评论说达到人类水平的结果是判断 AI 能力的一种狭隘方式。
   - 提到像 Fast Downward 这样的传统规划器时，强调不应仅通过与人类表现的比较来判断 AI 的规划能力。
- **AI 训练中数据使用的资源与效能**：参与者分享了关于数据检索和处理细微差别的见解，重点是从 Parquet 等云存储格式中进行高效读取。
   - 参与者们在提高训练数据质量的有效方法上达成了共识，但对于普遍有效的策略仍存在不确定性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.pola.rs/user-guide/io/cloud-storage/#reading-from-cloud-storage)">Cloud storage - Polars 用户指南</a>：未找到描述</li><li><a href="https://x.com/_xmaster96/status/1837489678024393205?s=46">来自 XMaster96 (@_XMaster96) 的推文</a>：我们不都经历过盯着训练损失曲线看太久的时刻吗？损失曲线中的这次下降是我在盯着新的 Aleph Alpha Foundation 模型预训练时发生的...</li><li><a href="https://arxiv.org/abs/2409.12917">通过强化学习训练语言模型进行自我纠正</a>：自我纠正（Self-correction）是大语言模型 (LLM) 一项非常理想的能力，但在现代 LLM 中一直被发现很大程度上是无效的。现有的训练自我纠正的方法...</li><li><a href="https://arxiv.org/abs/2409.13373">LLM 仍然无法规划；LRM 可以吗？OpenAI o1 在 PlanBench 上的初步评估</a>：规划一系列行动以达到预期状态的能力长期以来被认为是智能 Agent 的核心能力，并且自 AI 研究开始以来一直是其不可或缺的一部分...</li><li><a href="https://arxiv.org/abs/2012.02748">挑战特征归因解释中常见的可解释性假设</a>：随着机器学习和算法决策系统越来越多地应用于高风险的人机交互场景，迫切需要理解其预测背后的基本原理...</li><li><a href="https://docs.pola.rs/user-guide/io/cloud-storage/#reading-fr">Cloud storage - Polars 用户指南</a>：未找到描述</li><li><a href="https://x.com/BlinkDL_AI/status/1838230783078924598">来自 BlinkDL (@BlinkDL_AI) 的推文</a>：RWKV-7 "Goose" 🪿 预览版 rc2 => 巅峰 RNN 架构？😃 将尝试为最终发布版本压榨更多性能。预览代码：https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7</li><li><a href="https://arxiv.org/abs/2303.13506">神经缩放的量化模型</a>：我们提出了神经缩放法则（Neural Scaling Laws）的量化模型，解释了观察到的损失随模型和数据规模呈幂律下降的现象，以及随规模扩大而突然涌现的新能力...</li><li><a href="https://arxiv.org/abs/2407.21075">Apple Intelligence 基础语言模型</a>：我们介绍了为 Apple Intelligence 功能提供支持的基础语言模型，包括一个旨在设备上高效运行的约 30 亿参数模型，以及一个大型基于服务器的语言模型...</li><li><a href="https://arxiv.org/abs/2205.10343">迈向理解 Grokking：表示学习的有效理论</a>：我们旨在理解 Grokking，这是一种模型在过拟合训练集很久之后才实现泛化的现象。我们提出了一个以有效理论为基础的微观分析和一个宏观的...</li><li><a href="https://arxiv.org/abs/2210.01117">Omnigrok：超越算法数据的 Grokking</a>：Grokking 是算法数据集的一种异常现象，即泛化发生在过拟合训练数据很久之后，目前仍难以捉摸。我们旨在通过分析损失函数来理解 Grokking...</li><li><a href="https://en.wikipedia.org/wiki/Betteridge%27s_law_of_headlines">Betteridge 标题定律 - 维基百科</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.12847v1">经过指令微调的语言模型是更好的知识学习者</a>：为了使基于大语言模型 (LLM) 的助手能够有效适应不断变化的信息需求，必须能够通过对新数据的持续训练来更新其事实知识...</li><li><a href="https://github.com/WinVector/Examples/blob/main/Model_Homotopy/LinRebal.ipynb">WinVector/Examples 仓库 main 分支下的 Examples/Model_Homotopy/LinRebal.ipynb</a>：针对不同文章的各种示例。通过在 GitHub 上创建账号为 WinVector/Examples 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/openai/MMMLU">Hugging Face 上的 openai/MMMLU 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1287556159804276737)** (10 条消息🔥): 

> - `Irreducible Loss Calculation`
> - `Chinchilla Optimal Token Size`
> - `Empirical Estimations`
> - `Scaling Laws Insights` 


- **在自回归模型中计算 Irreducible Loss**：一位用户询问了 *Scaling Laws for Autoregressive Modeling* 的作者是如何计算 **irreducible loss** 的，并参考了真实数据分布的熵。
   - 一位成员建议，这可能是与幂律指数（power law exponent）一起进行经验拟合的，并指出目前缺乏明确的答案。
- **探索 Chinchilla 最优 Token 数量**：一位用户对选择 **3.2B tokens 进行预训练**的说法表示困惑，询问背后是否有固定的计算方法。
   - 澄清指出，该关系符合大约 **D = 20P** 的权衡，且这一比例通常在没有严格计算的情况下被使用。
- **源自 Chinchilla 研究结果的比例**：讨论表明，**D = 20P** 的比例可以直接参考 *Hoffman et al.* 的表格，无需复杂的计算。
   - 正如一位成员所确认的，这表明无论 FLOP 预算如何，预训练所需的 tokens 都可以被近似估算。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1286791084139089972)** (61 条消息🔥🔥): 

> - `Interpretability at EMNLP2024`
> - `KV Cache Experiments`
> - `Model Training Interventions`
> - `Sparse Feature Circuits`
> - `SAE and Transformer Interpretability` 


- **EMNLP2024 的论文展示了可解释性**：一位成员对有两篇论文被 [#EMNLP2024](https://x.com/FazlBarez/status/1837229484543726036) 接收感到自豪；一篇论文关注 Transformer 中的 **attention-MLP 交互**，另一篇关注 **可解释的序列延续**。
   - 这些贡献突显了在理解复杂模型行为方面的进展。
- **KV Cache 实验揭示存储机制**：对 KV cache 的实验表明，单个 token 会影响后续层的表示，阐明了像 **'NY'** 这样单个 token 的改变是如何在模型中传播的。
   - cache 值中的 **尖峰观测（spike observations）** 意味着需要更长的 prompts 来有效地存储有意义的信息。
- **关于模型训练干预的讨论**：有推测认为预训练干预可能会影响可解释性，但共识是修改架构可能比改变训练过程产生更好的结果。
   - 最近的研究强调了 **训练时干预（train-time interventions）** 在提高模型理解方面的挑战和潜力。
- **稀疏特征电路（Sparse Feature Circuits）提供见解**：参考 Sam Marks 的工作，一位成员指出揭示伪相关性的训练探针（probes）如何进行 **事后（post-hoc）** 修正，强调了调整训练数据的重要性。
   - 该方法展示了可解释性技术的实际应用，也可以为更广泛的研究领域提供参考。
- **SAE 赋能更广泛的可解释性上下文**：SAE (Sparse Attention Embeddings) 被讨论作为扩展 Transformer 可解释性上下文的工具，超越了有限的 prompt 测试。
   - 对话呼吁在 **可解释性** 挑战中增加 SAE 技术的实际实例化和成功应用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/FazlBarez/status/1837229484543726036">Fazl Barez (@FazlBarez) 的推文</a>: 非常自豪有 2 篇论文入选 #EMNLP2024！🚀 1️⃣ “解释 Transformer 中的上下文查找：调查 Attention-MLP 交互” 2️⃣“迈向可解释的序列延续...”</li><li><a href="https://arxiv.org/abs/2309.07311v4">Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs</a>: NLP 中的大多数可解释性研究都集中在理解完全训练好的模型的行为和特征。然而，某些对模型行为的见解可能只能通过观察...来获得。</li><li><a href="https://github.com/PicoCreator/QKV-Transformers-are-RNNs?tab=readme-ov-file#immediate-implication>">GitHub - PicoCreator/QKV-Transformers-are-RNNs: QKV Transformers are RNN's with extra steps and larger memory capacity</a>: QKV Transformer 是具有额外步骤和更大内存容量的 RNN - PicoCreator/QKV-Transformers-are-RNNs
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1287776370247598213)** (8 条消息🔥): 

> - `MMLU_PRO 采样逻辑`
> - `Gemma 模型 BOS token 使用`
> - `Pythia 6.9b-deduped 低分问题`
> - `MMLU 任务描述的重要性` 


- **MMLU_PRO 采样逻辑需要关注**：`./leaderboard/mmlu_pro` 任务与其原始实现有所不同，因为它在 fewshot 采样时忽略了问题类别，这与 [MMLU-PRO 代码](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/47b9891aacb8bd7cda29d5c5ba17b9434dd333bc/evaluate_from_local.py#L228) 不同。
   - 另一位用户建议更新采样逻辑，以根据问题类别提高其准确性，具体实现详见 [此处](https://github.com/rimashahbazyan/lm-evaluation-harness/blob/f117e6c09e32c553df0ab8cf8964a8b16636832e/lm_eval/api/samplers.py#L186)。
- **Gemma 模型需要调整**：一位成员强调了为 Gemma 模型引入 BOS token 的重要性，并指出目前的做法可能会破坏 perplexity 任务的假设。
   - 他们计划为此行为添加一个切换标志（toggle flag），默认设置为 `False`，但作为特殊情况，对 Gemma 模型将其覆盖（override）为 `True`。
- **讨论了 Pythia 模型的低 MMLU 分数**：有人对 Pythia 6.9b-deduped 模型的 MMLU 5-shot 低分表示担忧，质疑其与已发布分数相比的有效性。
   - 其他成员建议，在 Pile 数据集上训练的模型在 MMLU 上表现不佳，主要是由于格式问题。
- **Context 中的任务描述至关重要**：讨论强调，非排行榜版本的 `mmlu_pro` 正确地为 fewshots 使用了相关主题，并在 context 中包含了任务描述。
   - 一位成员建议任务描述应以换行符结尾，以与参考实现保持一致，并计划提交一个 PR。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/47b9891aacb8bd7cda29d5c5ba17b9434dd333bc/evaluate_from_local.py#L228)">MMLU-Pro/evaluate_from_local.py at 47b9891aacb8bd7cda29d5c5ba17b9434dd333bc · TIGER-AI-Lab/MMLU-Pro</a>：MMLU-Pro 的脚本。可以通过在 GitHub 上创建账户为 TIGER-AI-Lab/MMLU-Pro 的开发做出贡献。</li><li><a href="https://github.com/rimashahbazyan/lm-evaluation-harness/blob/f117e6c09e32c553df0ab8cf8964a8b16636832e/lm_eval/api/samplers.py#L186">lm-evaluation-harness/lm_eval/api/samplers.py at f117e6c09e32c553df0ab8cf8964a8b16636832e · rimashahbazyan/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - rimashahbazyan/lm-evaluation-harness</li><li><a href="https://github.com/rimashahbazyan/lm-evaluation-harness/blob/robustness_task/lm_eval/tasks/robustness/mmlu_pro/fewshot_prompt_robustness_mmlu_pro.yaml">lm-evaluation-harness/lm_eval/tasks/robustness/mmlu_pro/fewshot_prompt_robustness_mmlu_pro.yaml at robustness_task · rimashahbazyan/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - rimashahbazyan/lm-evaluation-harness
</li>
</ul>

</div>

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1287751274972053544)** (7 messages): 

> - `Activation Functions Sync` (激活函数同步)
> - `Init Functions and Stability` (初始化函数与稳定性)
> - `Truncation of Normal Distribution` (正态分布截断)


- **激活函数文档不同步 (Activation Functions Documentation Out of Sync)**：一位成员指出，文档中列出的可用激活函数未能反映代码中的完整范围，特别是 [Swiglu](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/activations.py)。
   - 另一位成员确认文档尚未更新，并引用了代码中列出这些函数的[特定行](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/neox_arguments/neox_args.py#L295)。
- **讨论 Trunc Normal 初始化**：一位成员建议将初始化函数更改为 **trunc_normal**，并引用了一项消融研究（ablation study），该研究表明如果没有它，在大规模训练时会出现不稳定性，正如 [AllenAI 研究](https://arxiv.org/abs/2409.02060)中所指出的。
   - 该成员强调多位作者参与其中，表明这种方法背后有大量的工作和研究支持。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2409.02060">OLMoE: Open Mixture-of-Experts Language Models</a>：我们介绍了 OLMoE，这是一个利用稀疏 Mixture-of-Experts (MoE) 的完全开源、最先进的语言模型。OLMoE-1B-7B 拥有 70 亿 (B) 参数，但每个输入 token 仅使用 1B。我们对其进行了预训练...</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/main/megatron/neox_arguments/neox_args.py#L295">gpt-neox/megatron/neox_arguments/neox_args.py at main · EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - EleutherAI/gpt-neox</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/activations.py">gpt-neox/megatron/model/activations.py at main · EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - EleutherAI/gpt-neox
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1286766396268609566)** (560 messages🔥🔥🔥): 

> - `KTO Trainer`
> - `Qwen Model Fine-tuning` (Qwen 模型微调)
> - `RAG Implementation` (RAG 实现)
> - `Chat Template Issues` (聊天模板问题)
> - `Reflection Fine-tune` (Reflection 微调)


- **关于 KTO Trainer 使用的讨论**：成员们澄清了 KTO Trainer 需要一个参考模型来计算奖励（rewards），建议在微调期间使用未改动的 Base 模型进行比较。
   - 有建议提出预先生成参考模型的响应，以在训练过程中节省显存。
- **Qwen 模型微调问题**：用户在更新后发现 Qwen 2.5 模型出现异常行为，特别是生成了与 Prompt 模板相关的错误响应。
   - 有人指出较小的模型对 Prompt 格式非常敏感，问题源于对 Prompt 处理方式的更改。
- **RAG 实现讨论**：参与者讨论了使用检索增强生成 (RAG) 作为增强模型响应的方法，并解决仅靠微调在知识保留方面的局限性。
   - 一位用户建议在 RAG 中有效地使用现有数据集，以避免训练过程中的知识丢失。
- **聊天模板问题**：用户强调了在微调模型中维护聊天模板（Chat Template）的困难，特别是需要将自定义模板与模型权重一起保存。
   - 引用了关于创建和保存模型聊天模板的 Hugging Face 文档。
- **Reflection 微调**：讨论表明，如果没有强大的奖励模型（Reward Model），使用 Reflection 微调方法在 Reflection Traces 上进行训练可能不会产生显著改进。
   - 参与者注意到使用 BoN (Best-of-N) 等方法对于微调中更好的对齐（Alignment）和性能至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://learn.microsoft.com/en-us/windows/powertoys/fancyzones">Windows 版 PowerToys FancyZones 工具</a>：一种用于排列和捕捉窗口到高效布局的窗口管理器工具</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing#scrollTo=yqxqAZ7KJ4oL">Google Colab</a>：未找到描述</li><li><a href="https://cobusgreyling.medium.com/prompt-tuning-hard-prompts-soft-prompts-49740de6c64c">Prompt Tuning, Hard Prompts &amp; Soft Prompts</a>：Prompt Engineering 是访问 Large Language Models (LLMs) 的方法，因此出现了 Pipelines, Agents, Prompt Chaining 等实现方式...</li><li><a href="https://docs.unsloth.ai/basics/saving-models/saving-to-vllm">保存到 VLLM | Unsloth 文档</a>：将模型保存为 16bit 以用于 VLLM</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing#scrollTo=vITh0KVJ10qX">Google Colab</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.01306">KTO: Model Alignment as Prospect Theoretic Optimization</a>：Kahneman &amp; Tversky 的 $\textit{前景理论}$ 告诉我们，人类以一种有偏见但定义明确的方式感知随机变量 (1992)；例如，人类以厌恶损失而闻名。我们展示了...</li><li><a href="https://unsloth.ai/blog/llama3-1">使用 Unsloth 微调 Llama 3.1</a>：通过 Unsloth 微调并运行 Meta 更新的 Llama 3.1 模型，支持 6 倍长的上下文长度！</li><li><a href="https://docs.unsloth.ai/basics/chat-templates">Chat Templates | Unsloth 文档</a>：未找到描述</li><li><a href="https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct">nvidia/Llama-3_1-Nemotron-51B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://www.metadock.net/">首页 - MetaDock</a>：告别频繁的窗口切换。MetaDock 凭借其独特的拆分屏幕和多布局系统，让你无缝管理多项任务。立即尝试！</li><li><a href="https://x.com/AlpinDale/status/1837860256073822471">Alpin (@AlpinDale) 的推文</a>：你现在可以以任何你想要的浮点格式加载任何 FP16 模型，只要它在 2 到 7 bits 之间。你想要非标准的 FP6_E3M2 或 FP7_E1M5 吗？它应该可以直接工作。吞吐量...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fnvlla/qwen25_bugs_issues_fixes_colab_finetuning_notebook/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://learnprompting.org/docs/trainable/soft_prompting">Soft Prompts 的优势与机制</a>：发现 Prompt Tuning 相比模型微调的优势。了解 Prompt Tuning 和 Soft Prompts 如何与 Large Language Models 协同工作。</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Chat Templates</a>：未找到描述</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes/multilingual/README.md">llama-recipes/recipes/multilingual/README.md · meta-llama/llama-recipes</a>：使用可组合的 FSDP 和 PEFT 方法微调 Meta Llama3 的脚本，涵盖单节点/多节点 GPU。支持默认和自定义数据集，适用于摘要和问答等应用...</li><li><a href="https://huggingface.co/docs/trl/main/en/kto_trainer">KTO Trainer</a>：未找到描述</li><li><a href="https://github.com/codelion/optillm">GitHub - codelion/optillm: 针对 LLMs 的优化推理代理</a>：针对 LLMs 的优化推理代理。通过在 GitHub 上创建账号来为 codelion/optillm 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M">GPT 到底是什么？Transformer 的视觉入门 | 深度学习第 5 章</a>：分解 Large Language Models 的工作原理。这些课程由观众直接资助：https://3b1b.co/support --- 这里有...</li><li><a href="https://huggingface.co/datasets/LlamaFinetuneGGUF/Programming-Alpaca-and-ShareGPT-Style">LlamaFinetuneGGUF/Programming-Alpaca-and-ShareGPT-Style · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/blog/1_58_llm_extreme_quantization">将 LLMs 微调至 1.58bit：让极限量化变得简单</a>：未找到描述</li><li><a href="https://github.com/mistralai/mistral-common/tree/main/src/mistral_common/tokens/tokenizers">mistral-common/src/mistral_common/tokens/tokenizers · mistralai/mistral-common</a>：通过在 GitHub 上创建账号来为 mistralai/mistral-common 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 微调 Llama 3.1, Mistral, Phi 和 Gemma LLMs，速度提升 2-5 倍，显存占用减少 80%</a>：微调 Llama 3.1, Mistral, Phi 和 Gemma LLMs，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/pull/1001">支持 KTO Trai</a>

ner with Unsloth by corbt · Pull Request #1001 · unslothai/unsloth</a>: 这个补丁似乎对于在 Unsloth 中成功使用 KTOTrainer 既是必要的也是充分的！</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">Home</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3.1, Mistral, Phi &amp; Gemma LLMs - unslothai/unsloth</li><li><a href="https://github.com/infiniflow/ragflow">GitHub - infiniflow/ragflow: RAGFlow 是一款基于深度文档理解的开源 RAG (Retrieval-Augmented Generation) 引擎。</a>: RAGFlow 是一款基于深度文档理解的开源 RAG (Retrieval-Augmented Generation) 引擎。 - infiniflow/ragflow
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1286883794258755596)** (24 messages🔥): 

> - `RAG 应用使用`
> - `文档评分成本分析`
> - `推理方法对比`
> - `API 服务折扣`
> - `评分中的投票准确性` 


- **探索用于文档结构化的 RAG 应用**：一位成员建议在进行分析之前，使用 **RAG 应用**将非结构化文档转换为结构化格式。
   - 另一位成员澄清说，他们的任务涉及 **L3.1 评分**，并且专注于离线推理，而不是创建微调数据集。
- **文档处理的高昂成本估算**：讨论显示，在不计人工成本的情况下，对 **250 万份文档**进行高 Token 计数的分析可能耗资约 **6 万美元**。
   - 一位成员计算出，使用 **L3.1 的 API** 将花费大约 **1.5 万美元**，这表明与本地配置相比可以显著节省成本。
- **推理方法对比**：成员们辩论了各种推理方法的优势，指出 **8x H100** 模型的吞吐量可能提供比预期更快的结果。
   - 建议使用 **2000-5000 个样本进行测试**，以有效评估成本和准确性。
- **带有折扣的 API 服务**：一位成员提出了是否有任何 **API 服务提供折扣**的问题，特别强调了 **OpenAI** 之前在批量推理（batch inferences）上的 5 折优惠。
   - 成员们对使用大型模型的高成本和局限性，以及小型模型性能不尽如人意表示了担忧。
- **三次投票以增强准确性**：成员们讨论了从不同模型获取 **三次投票** 以确保评分准确性的重要性。
   - 一位成员确认他们将在其测试策略中实施这种方法。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1286787189497266367)** (76 messages🔥🔥): 

> - `仅预测损失评估`
> - `Phi 3.5 Tokenization 问题`
> - `RAG 微调最佳实践`
> - `合并模型性能挑战`
> - `使用 Lora Adapters 进行持续预训练` 


- **仅预测损失以提高 VRAM 效率**：一位用户询问在训练循环中使用 `prediction_loss_only = True` 的目的，以防止 VRAM 使用量上升。
   - 讨论中提出了关于这是否仅影响评估阶段（evaluation passes）的疑问。
- **Phi 3.5 的 Tokenization 问题**：一位用户注意到 Phi 3.5 中模型与 Tokenizer 之间的 Tokenization 差异，导致对 Padding Tokens 产生困惑。
   - 此外，还存在 Tokenizer 在编码过程中不添加特殊 Token 的问题，这可能会影响训练。
- **RAG 微调最佳实践**：一位成员询问了关于使用上下文、问题和答案微调 RAG 模型的模板，强调了其复杂性。
   - 建议包括探索研究论文以获取指导，表明这是一个细分且复杂的领域。
- **模型合并后的性能问题**：用户报告称，在将 Lora Adapters 与原始权重合并后，其模型的性能显著下降。
   - 成员们对 4bit 合并与 16bit 合并相比的有效性表示了担忧。
- **使用 Lora Adapters 进行持续预训练**：一位用户寻求澄清持续预训练将如何与现有的 Lora Adapters 交互，询问是否会创建新的 Adapter。
   - 建议保存合并后的模型以便于未来的训练灵活性，并强调了保持合并状态的重要性。



**提及的链接**: <a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing#scrollTo=r6bUnxe6N3pf">Google Colab</a>: 未找到描述

  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1286947640965333022)** (3 条消息): 

> - `SetFit v1.1.0 Release`
> - `Training Classifiers`
> - `Sentence Transformers Update`
> - `Python Version Support` 


- **SetFit v1.1.0 发布，带来训练改进**：**SetFit v1.1.0** 版本现已发布，利用 Sentence Transformers Trainer 在 **CPU 和 GPU** 上进行高效的分类器训练，解决了因第三方库更新引起的多个问题。
   - 新版本引入了 **MultiGPU 支持**，并将 'evaluation_strategy' 弃用，改为 'eval_strategy'，同时新增了对 **Python 3.11** 和 **3.12** 的支持。
- **SetFit 分类器模型训练的两个阶段**：训练 **SetFit 分类器模型** 包含两个主要阶段：首先微调 Sentence Transformer embedding 模型，随后训练一个将 embedding 映射到类别的分类器。
   - 这种结构化方法提升了性能和效率，特别是配合 1.1.0 版本中更新的支持功能。
- **SetFit 训练过程的关键更新**：对 **max_steps** 和 **eval_max_steps** 等参数进行了重大改进，现在这些参数被强制作为硬限制（hard limits），确保训练结果更加可靠。
   - 训练和验证损失（validation losses）的变化也得到了强调，这有助于提升整体训练过程的鲁棒性。



**提到的链接**: <a href="https://huggingface.co/posts/tomaarsen/875775738519407">@tomaarsen on Hugging Face: &quot;🎉SetFit v1.1.0 is out! Training efficient classifiers on CPU or GPU now uses…&quot;</a>: 未找到描述

  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1286763841350406277)** (506 条消息🔥🔥🔥): 

> - `Perplexity Pro issues`
> - `Usage of AI models`
> - `Anthropic model release`
> - `Perplexity functionality`
> - `Collaborative opportunities` 


- **Perplexity Pro 订阅问题**：多位用户报告其 Pro 状态间歇性失效，部分用户遇到“Query rate limit exceeded”等错误提示。许多人注意到退出并重新登录有时可以解决问题，但对延迟和 Bug 的担忧依然存在。
   - 这些问题似乎是系统性的，可能与平台近期进行的更新和维护有关。
- **AI 模型对比与使用案例**：用户讨论了包括 Perplexity、ChatGPT 和 Claude 在内的不同 AI 模型的有效性，强调了它们在各种应用中的各自优势。分享了关于如何优化这些模型在编程、头脑风暴和学术研究等任务中使用的见解。
   - 许多人指出了某些模型面临的挑战，特别是在 hallucinations（幻觉）和实时信息检索的可靠性方面。
- **Anthropic 新模型可能发布**：社区对 Anthropic 可能发布新模型的消息议论纷纷，根据用户分享的一份独家采访，该模型可能很快就会公布。这引发了人们对新 AI 模型可能带来的额外能力的期待。
   - 也有一些怀疑的声音，讨论 Perplexity 是否会很快整合新模型，暗示了竞争激烈的市场环境。
- **对 Perplexity 转向广告模式的担忧**：用户反馈表达了对 Perplexity 界面中产品和广告显示方式近期变化的担忧，认为这会分散注意力。建议将推荐内容放在侧边栏，而不是与搜索结果混在一起，以增强可用性。
   - 用户对感知到的向商业模式转变表示失望，担心这可能会削弱 Perplexity 最初设定的独特价值。
- **用户体验增强与协作**：关于 Complexity 扩展程序的讨论强调了其在提升 Perplexity 用户体验方面的优势，例如可自定义主题和更便捷的导航。用户分享了协作机会，并表示有兴趣通过 AI 工具改进工作流。
   - 社区驱动的反馈以及理解如何有效利用这些工具的重要性被强调为提升平台的关键。


<div class="linksMentioned">

<strong>提到的链接</strong>:

</div>

<ul>
<li>
<a href="https://x.com/rowancheung/status/1838280020642676802">Rowan Cheung (@rowancheung) 的推文</a>：我刚刚完成了一次关于新的重大 AI 模型升级的独家采访。可以确认，明天对开发者来说将是重要的一天。在禁令解除的瞬间，我将在 X 上发布完整对话...</li><li><a href="https://docs.perplexity.ai/guides/model-cards">支持的模型 - Perplexity</a>：未找到描述</li><li><a href="https://www.workingtheorys.com/p/taste-is-eating-silicon-valley">品味正在吞噬 Silicon Valley。</a>：正如软件在上一时代吞噬了世界并极大地改变了各行各业，现在品味正在吞噬软件——以及随之而来的 Silicon Valley。</li><li><a href="https://www.intowindows.com/how-to-set-google-as-default-search-in-vivaldi-browser/">如何在 Vivaldi 浏览器中将 Google 设置为默认搜索</a>：Vivaldi 浏览器虽然不如其竞争对手那样流行，但它绝对是适用于 Windows 操作系统以及 Mac 的优秀浏览器之一。</li><li><a href="https://www.msn.com/en-gb/money/topstories/perplexity-in-talks-with-top-brands-on-ads-model-as-it-challenges-google/ar-AA1qXfYF)">MSN</a>：未找到描述</li><li><a href="https://tenor.com/view/the-voices-cat-gif-18434775077971513816">The Voices Cat GIF - The voices cat - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://addons.mozilla.org/en-US/firefox/addon/complexity/">Complexity - Perplexity.ai 增强版 – 获取此 Firefox (en-US) 扩展</a>：下载适用于 Firefox 的 Complexity - Perplexity.ai 增强版。⚡ 增强你的 Perplexity.ai</li><li><a href="https://tenor.com/view/huh-confused-dont-know-thinking-john-c-reilly-gif-16141237">Huh Confused GIF - Huh Confused Dont Know - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/apostraphi/status/1837219719495176299?s=61">Phi Hoang (@apostraphi) 的推文</a>：说实话...我们没想到会有这么多学生加入 Perplexity 的返校活动！欢迎大家 + 我们才刚刚开始。</li><li><a href="https://huggingface.co/spaces/yuntian-deng/o1">Chat-with-OpenAI-o1 - yuntian-deng 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://tenor.com/view/holo-spice-and-wolf-holo-the-wise-wolf-horo-korbo-gif-13009516793083034180">Holo Spice And Wolf GIF - Holo Spice and wolf Holo the wise wolf - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/cat-underwater-gif-922906369727670801">Cat Underwater GIF - Cat Underwater - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1286859683185037343)** (33 messages🔥): 

> - `Human DNA Preservation` (人类 DNA 保存)
> - `Titan Sub Implosion` (泰坦号潜水器内爆)
> - `Chain of Thought Reasoning` (Chain of Thought 推理)
> - `AI Meeting Prep Reports` (AI 会议准备报告)
> - `Python Learning Resources` (Python 学习资源)


- **晶体中保存的人类 DNA**：一篇引人入胜的文章讨论了如何将 **human DNA** 保存进长效晶体中，这可能会为未来的遗传研究提供参考。你可以在[这里](https://www.perplexity.ai/search/human-dna-preserved-in-long-la-W4e5dAggRbuOMuW_mOaqmA)阅读更多相关内容。
   - 这种保存技术的详细信息记录在[原始线程](https://www.perplexity.ai/page/human-dna-preserved-in-long-la-6_oF.rF1StCqzUJsYCTy4Q)中。
- **泰坦号潜水器内爆见解**：围绕悲剧性的**泰坦号潜水器内爆 (Titan sub implosion)** 展开的讨论，链接提供了关于事故原因的见解。在[这里](https://www.perplexity.ai/search/the-titan-sub-implosion-jXEEHAd9RI64Df48GnPkJg)探索更多关于此事件的信息。
   - 多位成员分享了关于此事件对深海探索和安全影响的看法。
- **Chain of Thought 推理最佳实践**：社区向大家推荐了一个关于 **Chain of Thought** 推理的资源——这是一种增强 AI 逻辑和推理能力的方法。点击[这里](https://www.perplexity.ai/page/chain-of-thought-reasoning-via-22CYSxmhTMSFr1gJIXM4dg)查看指南。
   - 更多背景信息在[相关线程](https://www.perplexity.ai/search/using-cot-canvas-with-cplx-and-D5K3ONW.SDm.SDYuTTz_Gw)中提供。
- **用于会议准备的 AI 报告**：一位用户分享了一个 **AI report generator** 的链接，该工具可协助准备会议，展示了其潜在优势。在[这里](https://www.perplexity.ai/search/danone-insights-emea-NR8r3YegQ.K5ww0lrxxQLw)阅读为达能 (Danone) 收集的见解。
   - 该工具旨在简化信息汇编，以实现高效的会议准备。
- **学习 Python 的资源**：有人提出了关于**学习 Python** 资源的问题，并为初学者和高级学习者提供了精选链接。其中一个资源可以在[这里](https://www.perplexity.ai/search/how-do-you-learn-python-on-you-qs5urKEZSMOALRSUWpHegA)找到。
   - 交流了各种针对 Python 学习策略的链接，满足不同熟练程度的需求。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1286784429737644174)** (18 messages🔥): 

> - `Llama-3.1-Sonar performance issues`
> - `Perplexity API citation challenges`
> - `Search Recency Filter`
> - `Inconsistent API outputs`
> - `Azure deployment for OCR` 


- **Llama-3.1-Sonar 与 Perplexity Web 应用相比表现不佳**：用户报告称，与 Perplexity Web 应用程序相比，**llama-3.1-sonar-large-128k-online** 的结果明显较差，理由包括输出不完整和格式不一致等问题。
   - 一位用户提出了一个多步骤流程来改进结果，强调了保留源引用的重要性。
- **Perplexity API 缺乏引用可靠性**：一位用户对 **Perplexity API 的不稳定行为**表示沮丧，特别是尽管请求了引用功能，但答案中提供的引用却不一致。
   - 他们强调，缺乏引用削弱了 API 的价值，而 API 的价值主要取决于其**搜索功能 (search features)**。
- **关于 Search Recency Filter 的咨询**：一位用户寻求澄清 **search_recency_filter** 是处于闭测阶段还是对所有用户开放，这表明了该功能对于获取及时信息的重要性。
   - 他们的目标是确保 API 在利用此过滤器时能够检索过去一小时内的更新。
- **用户对不一致的 API 输出感到沮丧**：多位用户报告了 API 输出的不一致性，包括尽管在 Prompt 中指定了仅限 HTML，但仍收到 **Markdown 和 HTML** 的混合内容。
   - 这种不一致性令人沮丧，因为用户发现 Web 端和 Labs Playground 的性能更好。
- **探索用于 OCR 服务的 Azure**：一位用户询问是否可以使用 Perplexity API **在 Azure 上部署 Web 服务**，特别关注 OCR 能力。
   - 这表明人们对于在云环境中利用该 API 进行实际应用的兴趣日益增加。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1287480964800581734)** (5 条消息): 

> - `Browser Thread Usage` (浏览器线程占用)
> - `CUDA Browser Development` (CUDA 浏览器开发)
> - `Wen Mei Hwu's Lecture` (Wen Mei Hwu 的讲座)
> - `Server Presence` (服务器在线状态)
> - `User Queries` (用户查询)


- **浏览器占用过多线程**：一名用户对浏览器仅开启 **10 个标签页**就占用 **126 个线程**表示沮丧，呼吁寻求更高效的解决方案。
   - 这突显了用户对日常任务中浏览器性能和资源管理的担忧。
- **对基于 CUDA 的浏览器的需求**：一名成员迫切请求开发 **CUDA 浏览器**，暗示目前市场上针对性能导向型网页浏览的产品可能存在空白。
   - 这表明用户希望通过 GPU 加速来增强处理并行任务的能力。
- **讲座视频请求**：一名成员询问是否有 **Wen Mei Hwu 讲座**的录像，强调了对其教学内容的兴趣。
   - 这反映了 AI 和技术社区对教育内容的持续关注。
- **确认服务器在线状态**：一名用户询问名为 **eqy** 的成员是否在服务器中，这表明了社交互动或协作需求。
   - 这突显了 Discord 服务器内的社区属性和同伴连接。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1287482333678534656)** (5 条消息): 

> - `3-bit and 5-bit support` (3-bit 和 5-bit 支持)
> - `Gemlite's efficiency` (Gemlite 的效率)
> - `Pareto frontier of methods` (方法的帕累托前沿)
> - `Accuracy of Llama3 8B Instruct` (Llama3 8B Instruct 的准确率) 


- **Hackathon 成果：新增 3-bit 和 5-bit 支持**：在最近的一次 Hackathon 中，一名成员成功添加了对 **3-bit** 和 **5-bit** 实现的支持，仅用时 **15 分钟**。
   - 实现详情可以在 [GitHub 仓库](https://github.com/mobiusml/gemlite/tree/master/gemlite/triton_kernels/experimental)中找到。
- **Gemlite 让 N-bit Kernel 开发更简单**：另一名成员表示，使用 **Gemlite** 创建其他 **N-bit kernel** 可能会容易得多。
   - 这种观点反映了开发者对该工具在处理低比特矩阵效率方面的信心。
- **探索加速比与准确率的帕累托前沿**：有人建议根据**加速比 (speedup)**和**准确率 (accuracy)**可视化不同方法之间的**帕累托前沿 (Pareto frontier)**。
   - 然而，有人指出每种方法都针对不同的 **batch sizes** 和 **shapes** 进行了优化，这使得标准化变得复杂。
- **Llama3 8B Instruct 的准确率数据**：一名成员确认拥有关于 **Llama3 8B Instruct** 在不同位宽下的**准确率 (accuracy)**数据。
   - 这些数据可以为不同比特表示的性能权衡提供见解。



**提到的链接**：<a href="https://github.com/mobiusml/gemlite/tree/master/gemlite/triton_kernels/experimental">gemlite/gemlite/triton_kernels/experimental at master · mobiusml/gemlite</a>：CUDA / Triton 中简单且快速的低比特 matmul kernel - mobiusml/gemlite

  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1287020121847627776)** (26 条消息🔥): 

> - `为 tensor.is_inference() 添加 guards`
> - `FSDP 参数 dtype 问题`
> - `对函数使用 torch.compile`
> - `CUDA 内存分配与 Tensor 对齐`
> - `Triton kernel 优化` 


- **询问关于 tensor.is_inference() 的 guards**: 一位成员询问是否应该在 Dynamo guards 实现中为 `tensor.is_inference()` 添加新的 guards，特别是在 `guards.cpp` 的某行代码之前。
   - 他们提到由于缺乏对 `x.is_inference()` 的 guards 导致触发了重新编译（recompiles），并提供了一个说明该情况的代码示例。
- **FSDP 在混合精度参数上遇到困难**: 一位用户在使用混合了 **FP32** 和 **BF16** 参数的模型上运行 `fully_shard()` 时遇到问题，导致了 `AssertionError`。
   - 讨论围绕可能的变通方法以及为了性能而分离 `RMSNorm` 层的潜在影响展开。
- **探索对非 Module 函数使用 torch.compile**: 一位成员询问 `torch.compile` 是否可以提升 `nn.Module` 实例之外的函数的运行速度，并寻求示例。
   - 另一位成员确认 `torch.compile` 确实适用于函数，从而开启了关于进一步优化的讨论。
- **CUDA 内存分配器对齐问题**: 一位用户寻求示例来验证，尽管 CUDA caching allocator 保证了最小块大小（minimum block size），但并非 PyTorch 中的所有 Tensor 指针都是对齐的。
   - 提供的一个示例说明了 Tensor 切片（slice）如何导致不对齐，进而引发了关于使用 `tensor.contiguous()` 来确保正确对齐的讨论。
- **利用 Triton kernel 优化**: 一位成员询问在将 Tensor 传递给 kernel 之前使其连续（contiguous）后，是否可以使用向量化访问（vectorized access）。
   - 确认了使用 `tensor.contiguous()` 可以实现安全的向量化访问，并参考了 Triton 针对优化的特定注解（annotations）。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit">torch.compile, the missing manual</a>: torch.compile 缺失的手册。你来到这里是因为你想使用 torch.compile 让你的 PyTorch 模型运行得更快。torch.compile 是一个复杂且相对较新的软件，因此你...</li><li><a href="https://github.com/pytorch/pytorch/blob/e9bfbf78d5d89df1ec59cb82d7f78b85f9014a98/torch/csrc/dynamo/guards.cpp#L166">pytorch/torch/csrc/dynamo/guards.cpp at e9bfbf78d5d89df1ec59cb82d7f78b85f9014a98 · pytorch/pytorch</a>: Python 中具有强 GPU 加速的 Tensor 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/csrc/dynamo/guards.cpp">pytorch/torch/csrc/dynamo/guards.cpp at main · pytorch/pytorch</a>: Python 中具有强 GPU 加速的 Tensor 和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1287435936690475029)** (2 messages): 

> - `GPU MODE 转型`
> - `CUDA MODE IRL 线下见面会成果`
> - `开源项目增长`
> - `黑客松获胜者与项目`
> - `社区价值观与未来愿景` 


- **CUDA MODE 转型为 GPU MODE**：原名为 CUDA MODE 的社区（最初是一个阅读小组）现在正式更名为 **GPU MODE**，其范围已扩展到 CUDA 编程之外。
   - 这一变化反映了更广泛的包容性和协作愿景，欢迎那些认同学习和社交参与价值观的人士。
- **成功的 CUDA MODE IRL 线下见面会**：首场 IRL 见面会聚集了 **150 名开发者**，从上午 10 点持续到午夜，在一天之内产出了超过 **40 个项目**。
   - 社区反馈称赞该活动是一个高效且联系紧密的聚会，巩固了其在协作创新方面的影响力。
- **不断增长的开源项目生态系统**：GPU MODE 社区已扩大到超过 **9,000 名成员**，推动了 **10 多个开源性能项目** 的开发，如 torchao 和 Liger。
   - 这一增长展示了社区在 GPU 编程领域构建和分享创新工具的承诺。
- **黑客松获胜者展示多样化项目**：黑客松获胜者展示了极高的创造力，项目包括 **Flexible Flash Attention 3** 和 **Triton 中的 NCCL 实现**，奖金总计 **$32.5K** 的算力额度。
   - 这些举措强调了社区利用其成就为未来开源贡献力量的意图。
- **社区价值观促进协作**：GPU MODE 社区倡导温馨且包容的环境，成员可以围绕 GPU 编程学习、协作并分享经验。
   - 正如所述，重点是在深度专注工作的同时平衡创新的社交方面，让成员能够共同享受这个过程。



**提到的链接**：<a href="https://x.com/swyx/status/1837577267259887702">swyx (@swyx) 的推文</a>：今天的 CUDA MODE 黑客松！这里有 @karpathy 关于 llm.c 的起源故事，以及它对快速、简单、LLM 编译的定制软件未来的启示。

  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1287547131141423179)** (7 messages): 

> - `Bitonic Sort 优化`
> - `CUDA Sorting Networks`
> - `使用 BLAS 进行 Batch Matrix Multiplication` 


- **寻求 GPU 上的 Bitonic Sort 优化**：一位用户询问如何在 GPU 上优化 **bitonic sort** 数组，并表达了在利用 **shared memory** 和实现 **global memory coalescing** 方面遇到的挑战。
   - 用户请求资源和帮助，旨在增强对该排序算法的理解。
- **NVIDIA CUDA Samples 助力排序工作**：另一位用户提供了一个有用的链接，指向 [NVIDIA CUDA samples sorting networks](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/sortingNetworks)，为优化提供了宝贵的见解。
   - 原用户表示感谢，认为该资源对他们的需求来说是“无价之宝”。
- **Bitonic Sort 性能的考量**：讨论了 **bitonic sort** 在处理长序列时的性能局限，引用了仓库中的评论，强调其与 **merge sort** 或 **radix sort** 等排序算法相比效率较低。
   - *一位用户指出他们出于教育兴趣想了解为什么 bitonic 序列在处理大数据集时表现不佳，* 暗示增加的递归深度可能是一个潜在问题。
- **BLAS 中的 Batched Matrix Multiplication**：一位用户寻求关于使用 **BLAS** 执行 **batched matrix multiplication** 的信息，特别是针对形状为 (b, m, n) @ (b, n, k) 的运算。
   - 他们询问在 batch 维度上循环并为每个元素启动 **gemm** 是否是唯一的方法，并指出 **OpenBLAS** 中缺少 batched gemm。



**提到的链接**：<a href="https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/sortingNetworks">GitHub 上的 cuda-samples/Samples/2_Concepts_and_Techniques/sortingNetworks</a>：为 CUDA 开发者提供的示例，展示了 CUDA Toolkit 中的功能 - NVIDIA/cuda-samples

  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1287081839957114961)** (8 条消息🔥): 

> - `用于自定义应用程序分析的 NVTX`
> - `斯坦福 CS149 并行计算课程`
> - `GEMM Kernel 设计教程`
> - `Karpathy 关于 LLM 编译器的见解`
> - `高速 Llama 3.1 模型` 


- **通过 NVTX 注解增强分析 (Profiles)**：利用 [NVIDIA Tools Extension (NVTX)](https://developer.nvidia.com/blog/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/)，开发者现在可以在 Nsight Systems 等工具中注解时间线，以捕获除 CUDA API 调用和 GPU kernels 之外的更多信息。
   - 该方法简化了具有深度嵌套调用图的复杂应用程序的处理过程，并提到现已引入 NVTX3 header-only 库。
- **斯坦福并行计算课程**：斯坦福大学将在 2024 年秋季开设 [CS149: Parallel Computing](https://gfxcourses.stanford.edu/cs149/fall24)，涵盖并行系统的基本原理和编程技术。
   - 该课程包括分析并行程序的性能和管理任务调度，将在 NVIDIA Auditorium 举行。
- **深入探讨 GEMM Kernel 设计**：[GEMM 教程系列](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) 的第 2 部分重点关注优化内存操作，以实现 GPU kernels 中高效的操作数张量移动。
   - 它介绍了流水线（pipelining）策略，以增强 NVIDIA Hopper 架构上的数据传输和处理效率。
- **Karpathy 对 LLM 编译器的看法**：Andrej Karpathy 最近在 CUDA hackathon 上的 [YouTube 演讲](https://www.youtube.com/watch?v=BmdOt6A6tHM) 讨论了 LLM 编译器的起源和未来，分享了引人入胜的见解。
   - 观众注意到他说话速度很快，使演讲显得节奏更紧凑，并附带了配套的 [llm.c GitHub repo](https://github.com/karpathy/llm.c) 链接。
- **最快 Llama 3.1 模型的声明**：有人声称拥有目前**最快的 Llama 3.1-405b** 模型，引发了对其性能指标的关注。
   - 虽然未提供更多细节，但这一断言暗示了 Llama 能力的显著提升。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cloud.sambanova.ai/">SambaNova Cloud</a>: 预览全球最快的 AI 推理 API。</li><li><a href="https://gfxcourses.stanford.edu/cs149/fall24">未找到标题</a>: 未找到描述</li><li><a href="https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/">CUTLASS 教程：带有流水线的高效 GEMM kernel 设计</a>: 欢迎阅读我们关于 GEMM (通用矩阵乘法) 教程系列的第 2 部分。在第 1 部分中，我们通过复习 WGMMA 讨论了 GEMM 的计算端，WGMMA 是实现 m... 的原始指令。</li><li><a href="https://www.youtube.com/watch?v=BmdOt6A6tHM">llm.c 的起源与 LLM 编译器的未来 - Andrej Karpathy 在 CUDA MODE</a>: 今天 CUDA mode hackathon 的非正式记录。https://github.com/karpathy/llm.c</li><li><a href="https://developer.nvidia.com/blog/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/">CUDA 专业技巧：使用 NVTX 生成自定义应用程序分析时间线 | NVIDIA 技术博客</a>: 上次你使用 NVIDIA Visual Profiler、Nsight VSE 或新的 Nsight Systems 中的时间线功能来分析复杂应用程序时，你可能希望看到的不仅仅是 CUDA...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1287207967594123366)** (1 messages): 

> - `Hiring ML Performance Engineers`
> - `Fal Inference Engine`
> - `Generative Media Platform`
> - `Model Inference Speed` 


- **Fal.ai 正在寻找 ML Performance Engineers**：Fal.ai 正积极招聘 **ML performance engineers** 以增强其生成式媒体平台，为优秀人才提供**极具竞争力的薪酬**和远程办公选项。
   - 有意向的候选人可以直接联系或发送简历至 batuhan@fal.ai。
- **Fal Inference Engine 提供闪电般的性能**：**fal Inference Engine™** 声称能够以高达 **4 倍的速度**运行 diffusion models，通过实时基础设施优化用户体验。
   - 该引擎旨在兼顾速度和质量，这对于在生成式媒体领域工作的开发者至关重要。
- **为开发者量身定制的创新功能**：Fal.ai 通过其动态定价模型，将开发者体验与强大的 AI 能力相结合，提供高性价比的可扩展性。
   - 这确保了用户只需为消耗的计算能力付费，促进了高效的资源管理。
- **专注于基础媒体模型**：公司的目标围绕构建顶级的生成式媒体平台，处理包括 **text-to-image** 和 **text-to-video** 在内的各种模态。
   - 他们强调需要能够帮助在不牺牲质量的前提下加速研发进程的人才。



**提到的链接**：<a href="https://fal.ai">fal.ai | 面向开发者的生成式媒体平台</a>：fal.ai 是运行 diffusion models 最快的方式，提供开箱即用的 AI 推理、训练 API 和 UI Playgrounds。

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1287828280581292107)** (1 messages): 

> - `Kernel Optimization`
> - `Matrix Multiplication Schemes`
> - `MLP Efficiency`
> - `Intermediate Result Utilization` 


- **为 MLP 优化单个 Kernel**：讨论了针对处理单个 Kernel 的特定优化方案，该 Kernel 执行**矩阵乘法**、**逐元素非线性函数**以及另一个**矩阵乘法**，以提高 MLP 效率。
   - 目标是在第二次乘法中利用第一次操作的中间结果，而无需返回 **global memory**，目前具体实现尚不明确。
- **中间数据处理的挑战**：成员们正在探索在不遇到由内存延迟（memory latency）引起的性能瓶颈的情况下，有效利用中间结果是否可行。
   - 对话强调了在 **MLP architectures** 中处理链式操作时，高效数据流的重要性。


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1286854040076161079)** (7 messages): 

> - `Speculative Decoding`
> - `TF Data vs. Grain`
> - `Grain Documentation Challenges`
> - `Epoch Training Issues` 


- **对使用 JAX 进行 Speculative Decoding 的兴趣**：一名成员询问是否有人有兴趣使用 **JAX** 实现 **speculative decoding**。
   - 这表明社区内对推进相关技术的兴趣日益增长。
- **TF Data 对许多人来说效果很好**：成员提到使用 **TF Data** 在他们的应用中被证明是有效的。
   - 有人指出，虽然它很直接，但文档在某些用例中推荐使用 **Grain**。
- **对 Grain 成熟度的担忧**：一名成员表达了对 **Grain** 的担忧，强调其**不成熟**且缺乏足够的文档。
   - 他们发现很难充分利用其功能，特别是在处理 **multiple workers** 和 **epoch training** 时。
- **Grain 中 Epoch 训练的挑战**：另一名成员分享了在 **Grain** 中进行 **epoch training** 的困难，指出它会持续运行直到没有数据可供迭代。
   - 这种缺乏清晰 epoch 边界的情况导致了复杂化，尤其是在文档问题悬而未决的情况下。
- **社区在 Grain 文档方面的困境**：成员们一致认为，虽然 **Grain** 入门简单，但由于文档稀缺，挖掘其全部潜力仍然很困难。
   - 这限制了社区的熟悉程度，并使得寻找问题答案变得更加困难。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1287419975593758831)** (7 条消息): 

> - `FP16 模型加载`
> - `基准测试期间的模型缓存`
> - `量化模型保存`
> - `AOTI 与执行模式` 


- **令人印象深刻的 FP16 模型加载实现**：一位用户强调了**加载任何 FP16 模型**的新功能，支持 2 到 7 位之间的各种浮点格式，并声称其吞吐量惊人，且**精度保持与 FP8 相当**。
   - *想要非标准的 FP6_E3M2 或 FP7_E1M5？它应该可以直接运行。*
- **在基准测试脚本中缓存模型加载**：一位用户询问在使用基准测试脚本时是否可以缓存模型加载，另一位用户确认这是可行的。
   - 建议的方法是使用 **save 选项**来保存量化后的模型并直接加载。
- **模型编译需要导出 (Export)**：讨论指出，为了**缓存编译结果**，必须导出模型，而目前的基准测试脚本尚不支持此功能。
   - 然而，用户表示这应该不会太复杂，并参考了 **torchchat 仓库**以获取有关执行类似模型的更多详细信息。



**提到的链接**：<a href="https://x.com/AlpinDale/status/1837860256073822471">Alpin (@AlpinDale) 的推文</a>：你现在可以以任何你想要的浮点格式加载任何 FP16 模型，只要它在 2 到 7 位之间。想要非标准的 FP6_E3M2 或 FP7_E1M5？它应该可以直接运行。吞吐量...

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1287508297091387444)** (2 条消息): 

> - `Santa Cruz 的 CUDA MODE`
> - `酒店盗窃事件` 


- **在 Santa Cruz 激活 CUDA MODE**：一位成员在 **Santa Cruz** 找到了一个不错的地方，进入了全神贯注的 **CUDA MODE**。
   - 这位爱好者似乎对在合适的环境中优化其计算能力感到非常兴奋。
- **Hackathon 意外：酒店失窃**：另一位成员报告说，在参加 **hackathon** 期间，他们所有的财物都在酒店房间内被盗。
   - 他们刚刚完成了**报案 (police report)** 以处理这起盗窃事件。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1287090051599044800)** (5 条消息): 

> - `讲座参与者`
> - `多伦多见面会` 


- **Teknium 确认出席见面会**：Teknium 肯定地回答了一个询问，确认他们将参加此次活动。
   - 另一位成员提到，“讲座结束后过来打个招呼”，暗示这是一个轻松的交流机会。
- **多伦多参会者建立联系**：Shagun 表示很高兴也在多伦多，建立了参会者之间的本地联系。
   - 这种互动为活动增添了人情味，增强了社区交流。


  

---


### **GPU MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1287715058591207435)** (17 条消息🔥): 

> - `CUDA/Torch 版本`
> - `Llama 3.1 的 CUDA 错误`
> - `GPU 兼容性`
> - `Bitblas 后端功能`
> - `GPU 上的 Torch 编译` 


- **用户寻求 CUDA/Torch 版本信息**：@strawberryjj 询问正在使用哪些 **CUDA/Torch 版本**，因为在处理与某个 [GitHub issue](https://github.com/mobiusml/hqq/issues/120) 相关的 `torchao` 后端时遇到了麻烦。
   - 错误指向了 CUDA 问题，引发了关于升级依赖项的建议。
- **Llama 3.1 的 CUDA 错误详情**：用户详细描述了在尝试运行 **Llama 3.1 8B 4bit 量化模型**时出现的与 CUDA 相关的错误追踪。错误指出 `torch.cat` 存在问题，并提到尝试设置 **CUDA_HOME** 和 **LD_LIBRARY_PATH** 但未获成功。
- **Tesla T4 GPU 的局限性**：对话透露，由于 **Tesla T4** GPU 属于旧架构，可能无法配合各种增强功能使用。建议需要 **Ampere** 架构及以上的 GPU 才能在 `torch.compile` 中支持 **fullgraph**。
- **Bitblas 后端建议**：建议尝试 **bitblas 后端**可能会有更好的结果，因为据报道它在其他 GPU 型号上可以运行。@mobicham 提到之前在他们的 **2080** GPU 上成功运行过 **bitblas**。
- **Triton 和 Torch 编译问题**：讨论显示 **torch.compile** 在旧款 GPU 上运行困难，因为它基于 **Triton**，而 **Triton** 并未针对旧架构进行优化。@strawberryjj 确认编译在他们的设置中确实失败了。



**提到的链接**：<a href="https://github.com/mobiusml/hqq/issues/120">尝试使用 llama3.1 8B 4bit 量化模型示例时出现 CUDA 错误 · Issue #120 · mobiusml/hqq</a>：从 https://huggingface.co/mobiuslabsgmbh/Llama-3.1-8b-instruct_4bitgs64_hqq_calib 获取模型。按照说明安装了 HQQ，并尝试运行 HF 网站上提供的示例。下载后...

  

---

### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1286767976980152372)** (34 messages🔥): 

> - `协调活动出席`
> - `LoRA 与 RAG 技术`
> - `llm.c 中的 Micro-optimizations`
> - `为学生服务创建 Chatbot`
> - `GEMM Kernel 设计` 


- **近期活动协调**：成员们讨论了协调周六即将举行的活动的努力，其中一名成员因在芝加哥而对错过活动表示遗憾。
   - 另一名成员感谢了同事的支持，强调了项目中的协作努力。
- **探索 LoRA 和 RAG 技术**：一位新成员询问了在大学 Chatbot 模型中结合 **LoRA** 和 **RAG** 技术的问题，并收到了积极的反馈。
   - 讨论包括对 RIG 和 QLoRA 等 Fine-tuning 方法的见解，并指出需要明确的 Evaluation Metrics。
- **llm.c 中的 Micro-optimizations**：分享了一个正在进行的 `repkv_backward` Draft PR，以及另一个针对 master 分支上 `softmax_forward_kernel5` 的 Micro-optimization PR。
   - 成员们对工作的协作性质表示感谢，并认可了来自 Hackathon 其他人的贡献。
- **为学生服务创建 Chatbot**：一名成员分享了对评估大学 Chatbot 的担忧，建议 **Hallucination** 指标可能至关重要。
   - 进一步的讨论强调了特定功能和用户反馈对确保有效性的重要性。
- **关于 GEMM Kernel 设计的见解**：提供了一个关于 GEMM Kernel 设计的教程链接，重点关注对 GPU 计算至关重要的 Memory 方面。
   - 成员们发现这些材料对于增强对 GPU 操作中高效管理 Data Buffers 的理解非常有价值。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/selfrag/selfrag_llama2_7b">selfrag/selfrag_llama2_7b · Hugging Face</a>: 未找到描述</li><li><a href="https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/">CUTLASS Tutorial: Efficient GEMM kernel designs with Pipelining</a>: 欢迎来到我们关于 GEMM (General Matrix Multiplication) 教程系列的第 2 部分。在第 1 部分中，我们通过讨论 WGMMA 探讨了 GEMM 的计算方面，WGMMA 是用于 m... 的原始指令。</li><li><a href="https://github.com/karpathy/llm.c/pull/762">Micro optimization for `softmax_forward_kernel5` by insop · Pull Request #762 · karpathy/llm.c</a>: 此分支包含对 softmax_forward_kernel5 的 Micro-optimization。摘要：在 attention_forward.cu 中使用 warpReduceMax 以利用 __shfl_down_sync，从而与其他 Kernel 保持一致 (reduce to...</li><li><a href="https://github.com/karpathy/llm.c/pull/764">DRAFT: Adding backward kernel for repkv on `llama3` branch (cudamode-irl) by insop · Pull Request #764 · karpathy/llm.c</a>: CC: @karpathy 这是一个 WIP repkv backward kernel，最初作为 cudamode-irl 项目启动。一旦完成后续工作，将移除 draft 标记。这项工作得到了 ALEKSA (@gordicaleksa) 和 E... 的支持。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1286864470005780560)** (41 条消息🔥): 

> - `BitNet Performance` (BitNet 性能)
> - `RMSNorm Implementation` (RMSNorm 实现)
> - `Quantization Techniques` (量化技术)
> - `HQQ and Fine-Tuning` (HQQ 与微调)
> - `Performance of Large Models` (大模型性能)


- **BitNet 可能缺少可学习参数**：有人担心 HF 为 BitNet 提交的 PR 可能缺少新 RMSNorm 层的可学习参数，这可能会影响整体性能。
   - 在对 BitNet 模型进行大量 Token 训练的微调中，成功案例有限，这引发了关于配置和实现的疑问。
- **RMSNorm 缩放的影响**：测试显示，将列向缩放（column-wise scaling）从预训练权重转移到 RMSNorm 实际上会导致性能下降，因为将激活值量化为 INT8 存在困难。
   - 这表明在不降低模型质量的情况下实现有效的缩放仍然是一个复杂的挑战。
- **量化可能会提高准确性**：讨论强调了即使不改变参数数量，量化也能提供更好的准确性，特别是对于像 Llama3 这样的大型模型。
   - 有人指出，使用随机投影（random projections）等技术可以帮助处理激活值中的离群值（outliers）。
- **HQQ 与大型语言模型**：HQQ 方法被指出在不需要太多微调或校准的情况下成功量化了 Llama3-70B，展示了该方法在实际任务中的有效性。
   - 强调了与较小的模型相比，大型模型在量化过程中通常不需要太多的干预。
- **有效的训练策略**：对于从头开始训练，共识是不需要特殊的技巧，模型在测试规模内表现尚可。
   - 然而，人们担心随着模型尺寸的增大或训练时间的延长，可能会出现不可预见的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/1bitLLM/bitnet_b1_58-3B">1bitLLM/bitnet_b1_58-3B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/OLMo-Bitnet-1B">NousResearch/OLMo-Bitnet-1B · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2105.03536">Pareto-Optimal Quantized ResNet Is Mostly 4-bit</a>：量化已成为压缩神经网络和降低计算成本的流行技术，但大多数先前的工作都集中在不改变网络大小的情况下研究量化。许多现实世界的...</li><li><a href="https://huggingface.co/blog/1_58_llm_extreme_quantization">Fine-tuning LLMs to 1.58bit: extreme quantization made easy</a>：未找到描述</li><li><a href="https://github.com/gau-nernst/quantized-training/blob/main/subclasses/bitnet.py">quantized-training/subclasses/bitnet.py at main · gau-nernst/quantized-training</a>：探索量化模型的训练。通过在 GitHub 上创建一个账号来为 gau-nernst/quantized-training 的开发做出贡献。</li><li><a href="https://huggingface.co/mobiuslabsgmbh/Llama-3.1-70b-instruct_4bitgs64_hqq">mobiuslabsgmbh/Llama-3.1-70b-instruct_4bitgs64_hqq · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/blob/78b2929c0554b79e0489b451ce4ece14d265ead2/src/transformers/models/gemma2/modeling_gemma2.py#L111-L128">transformers/src/transformers/models/gemma2/modeling_gemma2.py at 78b2929c0554b79e0489b451ce4ece14d265ead2 · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。- huggingface/transformers
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/)** (1 条消息): 

marksaroufim: https://x.com/shreyansh_26/status/1837157866509144492
  

---

### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1287116750663716887)** (13 条消息🔥): 

> - `访问码请求`
> - `活动邀请`
> - `注册问题` 


- **访问码混乱**：许多成员在注册过程中遇到问题，特别是需要**访问码**。据提到，如果用户所在公司有 **Google Developer Relations** 代表，则可以从该代表处获取此代码。
   - 一位成员计划如果问题持续，将与他们的 **devrel** 联系人确认，展现了积极的态度。
- **活动邀请社交**：一位成员主动提出向感兴趣的人 **DM**（私信）**活动邀请**，并提醒名额可能很快就会填满。
   - 另一位成员表示有兴趣参加，称这与他们**正在进行的项目**相契合。
- **参加活动的讨论**：虽然许多人对活动感到兴奋，但一位成员提到最近已经频繁出差，导致对是否参加犹豫不决。
   - 社区保持着友好的氛围，展示了对彼此计划的相互支持和体谅。


  

---


### **GPU MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1286766704214409310)** (169 条消息🔥🔥): 

> - `Hackathon 组队`
> - `CUDA MODE 回顾与亮点`
> - `项目提交与 Pitch`
> - `演讲与录音`
> - `未来合作` 


- **Hackathon 组队努力**：参与者讨论了在 **Hackathon** 期间组队和协作的事宜，建议通过指定频道进行自发组织和沟通。
   - 由于场地停车位有限，成员们还建议使用 **Uber** 出行。
- **CUDA MODE 活动收获好评**：**Hackathon** 展示了令人印象深刻的项目，参与者从活动期间形成的团队活力和协作中深受启发。
   - 许多人对可能重点展示的独特项目表示兴奋，例如在移动端运行 **LLMs** 以及 **solo hackers** 的工作。
- **项目提交流程与时间线**：十个团队被选中进行 **Pitch**，评判标准包括商业实用性和智力趣味性，反馈意见强调了 **demo** 的重要性。
   - 提醒参与者在截止日期前填写**意向声明**，以确保他们的项目被纳入考虑范围。
- **演讲已录制并可供回顾**：讨论表明，活动期间的演讲已录制，剪辑后将在 **YouTube** 频道发布。
   - 与会者对记录和分享活动内容的努力表示感谢，这增强了社区的参与感。
- **Hackathon 后的社区参与和项目**：鼓励成员从私有频道复制任何重要信息，因为这些频道将在 **Hackathon** 结束后清理。
   - 社区计划为**正在进行的项目**保留一个专门频道，以支持进一步的协作和开发。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.facebook.com/codingcompetitions/hacker-cup/2024/practice-round/scoreboard?track=AI_OPEN_TRACK">未找到标题</a>: 未找到描述</li><li><a href="https://tinyurl.com/cudamodehack">提交你的项目</a>: 请使用此表单与我们分享你的项目详情。这些信息将帮助我们更好地了解你的项目，并确保你被纳入最终评审。请填写表单...</li><li><a href="https://lambdalabs.com/cuda-mode-irl-cloud-credits">CUDA MODE IRL Lambda 云算力额度</a>: 领取你的 CUDA MODE IRL 黑客松云端额度。</li><li><a href="https://maps.app.goo.gl/Q48tEdSu6fP5P8sH6?g_st=com.google.maps.preview.copy">Caffe Centro SP · 旧金山，加利福尼亚州</a>: 未找到描述</li><li><a href="http://github.com/cchan/tccl">GitHub - cchan/tccl: Triton 中的可扩展集合通信库</a>: Triton 中的可扩展集合通信库。通过在 GitHub 上创建账号为 cchan/tccl 的开发做出贡献。</li><li><a href="https://youtu.be/BmdOt6A6tHM">llm.c 的起源与 LLM 编译器的未来 - Andrej Karpathy 在 CUDA MODE</a>: 今天 CUDA MODE 黑客松的非正式记录。https://github.com/karpathy/llm.c</li><li><a href="https://forms.gle/CRtuyWCkviEGB65B6">fal CUDA MODE 帽子</a>: 未找到描述</li><li><a href="https://github.com/AnswerDotAI/gpu.cpp/tree/dev">GitHub - AnswerDotAI/gpu.cpp (dev 分支)</a>: 一个使用 WebGPU 进行便携式底层 GPU 计算的轻量级库。 - GitHub - AnswerDotAI/gpu.cpp at dev</li><li><a href="https://github.com/modal-labs/modal-examples">GitHub - modal-labs/modal-examples: 使用 Modal 构建的程序示例</a>: 使用 Modal 构建的程序示例。通过在 GitHub 上创建账号为 modal-labs/modal-examples 的开发做出贡献。</li><li><a href="https://github.com/vllm-project/vllm/pull/8713">[build] 启用现有的 PyTorch (针对 GH200, aarch64, nightly)，由 youkaichao 提交 · Pull Request #8713 · vllm-project/vllm</a>: 未找到描述</li><li><a href="https://github.com/pytorch/pytorch/pull/136415">在 CUDA 中实现 nonzero_static，由 galv 提交 · Pull Request #136415 · pytorch/pytorch</a>: 这为 nonzero_static 增加了 CUDA 功能，该功能在 #97417 中缺失。这允许完全基于 CUDA 的图（graphs）避免数据依赖形状（data-dependent shapes）。这在各种情况下都很有帮助，其中之一是...</li><li><a href="https://bit.ly/modal-credits.">Modal 黑客松额度</a>: 要领取你的 Modal 额度，请先在 https://modal.com/ 注册账号。然后，通过此表单告知我们你的用户名。如需支持，请加入 Modal Slack。这里有一些入门示例...</li><li><a href="https://github.com/charlesfrye/cuda-modal">GitHub - charlesfrye/cuda-modal: 在 Modal 上进入 CUDA MODE</a>: 在 Modal 上进入 CUDA MODE。通过在 GitHub 上创建账号为 charlesfrye/cuda-modal 的开发做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/blob/ae02d663cdf493362699d2672ed7dc9019a7033b/test/inductor/test_flex_attention.py#L1938">pytorch/test/inductor/test_flex_attention.py (位于 ae02d663cdf493362699d2672ed7dc9019a7033b) · pytorch/pytorch</a>: Python 中具有强大 GPU 加速的张量和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>

### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1286764414703239188)** (78 条消息🔥🔥): 

> - `KLDivLoss Kernel 问题`
> - `RMSNorm 和 LayerNorm Bug`
> - `Cross-Entropy 对比`
> - `Kernel Reduction 方法`
> - `Triton Grid Size 限制` 


- **KLDivLoss kernel 存在计算问题**：成员们讨论了 KLDiv 的 **backward kernel 公式**可能不正确，且在 forward kernel 中也发现了潜在问题。
   - 另一位成员指出，根据 reduction 参数，kernel 除法是在外部进行的，并怀疑在大 vocab sizes 下存在循环展开（loop unrolling）问题。
- **RMSNorm 和 LayerNorm 的 Bug 仍然存在**：分享了关于 **RMSNorm** 和 **LayerNorm** 的问题，特别是关于错误的输出形状以及 Triton 程序处理中可能存在的失配。
   - 有推测认为，由于 Triton 程序中 grid 的管理方式，两者都存在相同的底层问题。
- **Cross-Entropy 提供了具有一致性的对比**：将 KLDivLoss 与 **Cross-Entropy** 进行了对比，指出后者的实现如何有效地处理更大的输入维度。
   - 建议将 KLDiv 的计算结果与 Cross-Entropy 对齐，以解决这些问题。
- **Kernel 函数的 reduction 处理**：指出 **reduction 方法**不应影响输出形状，因为所有计算都在 kernel 函数内部进行。
   - 一位成员强调，之前对某些 reduction 方法的求和值（sum values）存储管理不当，导致了错误。
- **解决 Triton 的 64kb 限制**：针对当 n_cols 超过一定数量时 Triton 的 **64kb 限制**提出了担忧，这可能会限制 kernel 的功能。
   - 提议的解决方案包括增加 grid size，类似于 Cross-Entropy 实现中使用的技术。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/linkedin/Liger-Kernel/pull/261">由 Tcc0403 修复 assert_verbose_allclose bug · Pull Request #261 · linkedin/Liger-Kernel</a>：摘要：修复 #259。添加更多 mask 以覆盖所有边缘情况，包括：nan、inf、-inf。在合并此 PR 之前应先合并 #262 以通过所有测试。测试已完成。硬件类型：运行 make test 以确保正确性...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/255">RMSNorm 聚合由 Tcc0403 提交 · Pull Request #255 · linkedin/Liger-Kernel</a>：摘要：解决 #179。进行中：解决大 hidden_size (4096) 的数值稳定性问题。测试已完成。硬件类型：RTX-3080。运行 make test 以确保正确性，运行 make checkstyle 以确保代码风格...</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/layer_norm.py#L189">Liger-Kernel/src/liger_kernel/ops/layer_norm.py (main 分支) · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernel。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/ce71d59b0b0894f9f3e7512f5a3bf3780c5a1499/src/liger_kernel/ops/kl_div.py#L121">Liger-Kernel/src/liger_kernel/ops/kl_div.py · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernel。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/ce71d59b0b0894f9f3e7512f5a3bf3780c5a1499/src/liger_kernel/ops/cross_entropy.py#L205">Liger-Kernel/src/liger_kernel/ops/cross_entropy.py · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernel。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[irl-announcements](https://discord.com/channels/1189498204333543425/1285285792054247516/1287100094340141126)** (15 条消息🔥): 

> - `Hackathon Kickoff`
> - `Compute Credits Information`
> - `Project Proposal Submission`
> - `Dinner and Networking`
> - `Preliminary Judging Update` 


- **Hackathon 正式启动！**：Hackathon 在热烈的欢迎声中拉开帷幕；参与者受邀在任何楼层就座，午餐将于中午供应。
   - 彩色贴纸系统标明了预先分配的计算资源赞助商，协助团队进行协作。
- **新的 Compute Credits 已开放领取**：参与者收到了关于如何从几家赞助商处领取 **Compute Credits** 的详细说明，并附有特定的代码。
   - 关于使用 **Modal** 的详细信息，请成员们注册并查看提供的示例以启动项目。
- **项目提案提交提醒**：提醒在今天 **下午 5 点** 前提交项目提案，以便纳入评审流程；参与者可以访问 [提交表单](https://docs.google.com/forms/d/e/1FAIpQLSfK71QlvjICnDNoPMzbG6yAYLKKXLNhnzdeHj5davHJ4MuMjg/viewform) 了解详情。
   - 提交提案对于记录项目细节和协调奖品发放至关重要。
- **晚餐与社交机会**：**3 楼** 供应晚餐直至 **晚上 9 点**，让参与者在继续开展项目的同时能够放松并进行社交。
   - 晚餐的最后通牒提醒参加者在截止时间前前往。
- **初步评审进行中**：评委与各团队进行了初步讨论，在 **40 多个参赛项目** 中，只有 **10 支团队** 将脱颖而出进行现场展示。
   - 尚未有评委到访的团队请进行回复，以确保所有团队都能获得反馈和支持。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.google.com/forms/d/e/1FAIpQLSfK71QlvjICnDNoPMzbG6yAYLKKXLNhnzdeHj5davHJ4MuMjg/viewform">提交您的项目</a>：请使用此表单与我们分享您的项目细节。这些信息将帮助我们更好地了解您的项目，并确保您被纳入最终评审。请填写表单...</li><li><a href="https://bit.ly/modal-credits)">Modal hackathon credits</a>：要领取您的 Modal credits，请先在 https://modal.com/ 注册账号。然后，通过此表单告知我们您的用户名。如需支持，请加入 Modal Slack。这里有一些示例可以帮助您开始...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[irl-sponsor-qa](https://discord.com/channels/1189498204333543425/1285287931828768940/1287067781803806780)** (91 条消息🔥🔥): 

> - `提前签到建议`
> - `计算额度与访问问题`
> - `针对特定节点的 Python 包支持`
> - `多 GPU 选项与 Lab 配置`
> - `闭幕活动致谢` 


- **早起的鸟儿有周边**：一位参与者建议“尽早到达”以避开活动期间的人潮。
   - 此建议针对那些想在三楼领取赞助商周边（swag）的人。
- **额度困惑已解决**：与会者澄清了注册后获取 Modal 额度的流程，指出虽然不会发送确认邮件，但额度应在提交后不久出现在账户中。
   - 参与者确认获得了 **$1k** 的额度，近期参会者已验证到账。
- **跨节点安装 Python 包的帮助**：有人寻求在计算节点上安装 `python3-poetry` 的支持，并确认使用虚拟环境安装成功。
   - 用户被引导在运行前使用 `source ~/venv-user3/bin/activate` 激活环境。
- **多 GPU 查询与限制**：关于 Nebius VM 是否支持多 GPU 的咨询显示，目前 Lab 仅限于单 GPU 配置。
   - 不过，有提到针对请求更多 GPU 的用户调高了配额。
- **闭幕活动与表达感谢**：活动在对赞助商和支持团队全天协助的感谢中圆满结束。
   - 鼓励参与者庆祝成功解决了黑客松期间面临的诸多挑战。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://modal.com/docs/guide/cuda">在 Modal 上使用 CUDA</a>: Modal 让你能够轻松使用数据中心级 NVIDIA GPU 加速工作负载。</li><li><a href="https://nebius.ai/docs/compute/operations/vm-connect/ssh#vm-authorized-keys">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/charlesfrye/cuda-modal">GitHub - charlesfrye/cuda-modal: 在 Modal 上进入 CUDA MODE</a>: 在 Modal 上进入 CUDA MODE。通过在 GitHub 上创建账户为 charlesfrye/cuda-modal 的开发做出贡献。</li><li><a href="https://github.com/charlesfrye/cuda-modal/blob/main/vscode_on_modal/vscode_server.py">cuda-modal/vscode_on_modal/vscode_server.py at main · charlesfrye/cuda-modal</a>: 在 Modal 上进入 CUDA MODE。通过在 GitHub 上创建账户为 charlesfrye/cuda-modal 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1287441339893420113)** (7 条消息): 

> - `跨平台 GPU 计算应用`
> - `MPS 与 WebGPU 性能对比`
> - `WebGPU 需要的进展`
> - `Metal 在低强度任务中的性能` 


- **MPS 在 macOS GPU 计算中占据主导地位**：如果仅针对 **macOS**，**MPS** 提供了超越 WebGPU 的卓越性能，在基准测试中达到了接近**理论最大值**的效率。
   - *移植性是有代价的，根据你的优先级，这可能并不值得在性能上做出妥协。*
- **WebGPU 性能差距**：成员们表示，与 **MPS** 相比，WebGPU 尚未达到其性能上限，持续的实验揭示了显著的性能差距。
   - 包括 [TVM 2020 年的文章](https://tvm.apache.org/2020/05/14/compiling-machine-learning-to-webassembly-and-webgpu)在内的一系列参考资料表明，WebGPU 可以达到**接近原生 GPU 的性能**。
- **性能优化的协作**：讨论强调了比较 MPS 和 WebGPU 之间的测试内核（kernels）的必要性，以评估特定应用的性能适用性。
   - 提出了优化 **llm.c WebGPU 实现**的协作呼吁，邀请感兴趣的各方在指定频道继续讨论。
- **低强度工作中的 Metal 与 CPU 对比**：有人提出了对于缺乏高算术强度的任务，使用 **Metal** 是否能比 **CPU** 带来性能收益的问题。
   - 这引发了对 Metal 是否能提供显著加速的场景探索的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tvm.apache.org/2020/05/14/compiling-machine-learning-to-webassembly-and-webgpu">使用 Apache TVM 将机器学习编译为 WASM 和 WebGPU</a>：未找到描述</li><li><a href="https://github.com/gpuweb/gpuweb/issues/4195">Cooperative matrix · Issue #4195 · gpuweb/gpuweb</a>：所有主要平台 API 现在都发布了类似的协同矩阵（cooperative matrix）扩展：Metal 在 MSL 3.1 中引入了 simdgroup_matrix，HLSL 在 SM6.8 中提供支持（目前为实验版本），SPIR-V...</li><li><a href="https://huggingface.co/spaces/Xenova/webgpu-embedding-benchmark">WebGPU Embedding Benchmark - Xenova 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/Xenova/webgpu-embedding-benchmark/discussions/30">Xenova/webgpu-embedding-benchmark · ⚡ WebGPU 基准测试结果 (40.40x 加速)</a>：未找到描述
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1287025227846520862)** (3 条消息): 

> - `基于云端的测试`
> - `关于高级用法的直播研讨会`
> - `扩展并行 Agent` 


- **提供基于云端的测试**：**订阅者**现在可以在云端测试和运行服务，无需任何本地安装；落地页上提供了一个**较小的演示版本**，可以通过 Loom 视频进行测试。
   - 这种设置使用户能够快速高效地探索功能。
- **即将举行的高级用法研讨会**：一场关于高级用法的直播研讨会定于 **EST 时间中午 12 点**举行，重点是**扩展到数千个并行 Agent 和代理 (proxies)**。
   - 参与者可以通过点击相关 YouTube 频道的 **Live 标签页**查看更多详情。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1286801967452262432)** (350 条消息 🔥🔥): 

> - `OpenRouter 模型变更`
> - `即将发布的 AI 模型公告`
> - `模型性能问题`
> - `OpenWebUI 集成`
> - `OpenAI 账户相关问题`

- **OpenRouter 默认禁用 Middle-Out**：OpenRouter 已正式更改 Prompt 处理的默认行为，禁用了 Middle-Out 转换，这影响了许多拥有既定工作流的用户。
   - 用户对这一决定表示担忧，强调了该功能对各种前端和后端系统的重要性。
- **Anthropic 可能发布新模型**：关于 Anthropic 推出新模型的猜测不断出现，社交媒体上的暗示表明近期将有重大发布。
   - 有消息称，这一发布可能会与 Google 的活动同时进行，并可能提供大量的免费 Token 优惠。
- **Hermes 3 模型出现性能问题**：用户报告了各种 Hermes 3 模型的延迟和停滞问题，API 响应等待时间超过 10 分钟。
   - 用户对模型整体性能低于往常表示担忧，导致用户开始寻找替代方案。
- **Infermatic 模型生成乱码**：一些用户注意到 Infermatic 模型在使用过程中产生无意义的输出，引发了对模型性能的质疑。
   - 建议检查活动日志并调整 Temperature 和 Penalties 等设置以缓解这些问题。
- **对 OpenAI 账户安全的担忧**：人们对 OpenAI 新闻室 Twitter 账户的安全表示担忧，据称该账户在禁用评论的同时发布了关于 Token 的公告。
   - 这一事件引起了用户对账户可能被盗或虚假信息传播的焦虑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/rowancheung/status/1838280020642676802">来自 Rowan Cheung (@rowancheung) 的推文</a>：我刚刚完成了一个关于全新重大 AI 模型升级的独家采访。可以确认，明天对开发者来说将是一个重要的日子。禁令解除的那一刻，我将在 X 上发布完整对话...</li><li><a href="https://huggingface.co/Sao10K/L3-8B-Stheno-v3.3-32K">Sao10K/L3-8B-Stheno-v3.3-32K · Hugging Face</a>：未找到描述</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>：查看你在 OpenRouter 上使用模型的情况。</li><li><a href="https://community.sambanova.ai/t/rate-limits/321">Rate Limits</a>：SambaNova Cloud 对每个模型的推理请求实施速率限制，以确保开发者能够尝试在最佳开源模型上进行最快的推理。免费层的速率限制...</li><li><a href="https://status.anthropic.com/incidents/xts3kyr0nrx1">Elevated Errors on Claude 3.5 Sonnet</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>：为模型消耗转换数据</li><li><a href="https://openrouter.ai/models/anthropic/claude-3.5-sonnet/providers">Anthropic: Claude 3.5 Sonnet – Provider Status</a>：查看提供商状态并向 Anthropic: Claude 3.5 Sonnet 发起负载均衡请求 - Claude 3.5 Sonnet 提供了优于 Opus 的能力，快于 Sonnet 的速度，且价格与 Sonnet 相同。</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>：设置模型使用限制</li><li><a href="https://dubesor.de/Flappyo1mini0Shot">Dubesor LLM Benchmark table</a>：未找到描述</li><li><a href="https://dubesor.de/Game2MistralLarge0Shot">Dubesor LLM Benchmark table</a>：未找到描述</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:free">Hermes 3 405B Instruct (free) - API, Providers, Stats</a>：Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理、多轮对话、长上下文连贯性...</li><li><a href="https://github.com/OpenRouterTeam/open-webui/commit/89659df1fa10348f51b389a8fea27b67a71dec5d">add middle-out by default · OpenRouterTeam/open-webui@89659df</a>：未找到描述</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API, Providers, Stats</a>：Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理、多轮对话、长上下文连贯性...</li><li><a href="https://dubesor.de/assets/shared/UIcompare/">Index of /assets/shared/UIcompare/</a>：未找到描述</li><li><a href="https://openrouter.ai/models/openai/o1-mini">o1-mini - API, Providers, Stats</a>：来自 OpenAI 的最新且最强的模型系列，o1 旨在在回答前花费更多时间思考。o1 模型针对数学、科学、编程和其他 STEM 相关任务进行了优化...</li><li><a href="https://github.com/hsiehjackson/RULER">GitHub - hsiehjackson/RULER: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models?</a>：此仓库包含 RULER 的源代码：你的长上下文语言模型的真实上下文大小是多少？- hsiehjackson/RULER</li><li><a href="https://github.com/OpenRouterTeam/open-webui">GitHub - OpenRouterTeam/open-webui: User-friendly WebUI for LLMs (Formerly Ollama WebUI)</a>：适用于 LLM 的用户友好型 WebUI（原 Ollama WebUI）- OpenRouterTeam/open-webui</li><li><a href="https://github.com/open-webui/open-webui/blob/6b463164f4b129e0ce4bdc9008dd661214fe5eb5/backend/open_webui/apps/openai/main.py">open-webui/backend/open_webui/apps/openai/main.py at 6b463164f4b129e0ce4bdc9008dd661214fe5eb5 · open-webui/open-webui</a>：适用于 LLM 的用户友好型 WebUI（原 Ollama WebUI）- open-webui/open-webui
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1287062592883261531)** (1 条消息): 

> - `Private LLM Servers` 


- **关于私有 LLM 服务器的咨询**：一位成员询问其他人是自己在运行 **私有 LLM 服务器**，还是由第三方管理。
   - *出于好奇，你们是自己在运行私有 LLM 服务器吗？*
- **对服务器请求的回应**：对话以对请求的感谢开始，标志着参与到关于 LLM 服务器管理的持续讨论中。
   - 该成员的回应暗示了对这些服务器运营方面的关注。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1286782880638566400)** (211 messages🔥🔥): 

> - `Music Production AI` (音乐制作 AI)
> - `Bittensor and Nous Research` (Bittensor 与 Nous Research)
> - `Byte-Level Architectures` (字节级架构)
> - `RetNet Integration` (RetNet 集成)
> - `World Sim API` (World Sim API)


- **音乐制作 AI 在音乐理论方面表现挣扎**：在关于 AI 在音乐制作中作用的讨论中，有人强调大型模型在基础乐理任务（如和弦转调）上表现吃力。一位成员一直在实验一个专注于音乐的 feline AI，生成 MIDI 文件并推荐合成方法。
   - 尽管付出了这些努力，用户一致认为，由于训练样本有限，乐谱（music notation）仍然是一个重大障碍。
- **对 Bittensor 做法的担忧**：有人投诉 Bittensor 似乎在未署名的情况下复制了 Nous Research 的分布式训练算法。这引发了关于 AI 社区在正确引用和认可方面的伦理考量。
   - 随着讨论的继续，一些参与者指出，分布式训练的努力必须优先考虑创新，而不仅仅是增加参数数量。
- **字节级认知模型讨论**：Ryunuck 鼓励探索新的训练方法以改进 AI 模型，倡导在研究中进行参与和协作。重点在于利用社会作为合成数据清理员（synthetic data janitors）来有效地训练模型。
   - 有人建议实施节奏模式（rhythmic patterns）以提高 Epochs 之间的模型性能，这表明训练策略正向创新转变。
- **RetNet 作为 Transformer 的补充**：Ryunuck 将 RetNet 描述为一种补充层，而不是替代品，用于增强 Transformer 的保留序列建模（retained sequence modeling）。这种方法可以在保持现有 Transformer 模型完整性的同时，提高处理长序列的能力。
   - 对话深入探讨了如何在不丢失 Transformer 架构的情况下，通过集成 RetNet 使模型更高效、更有效。
- **World Sim API 的利用**：用户讨论了 Nous World Client 及其功能，该客户端在创建账户时提供少量积分。对话强调了使用该 API 执行各种任务的成本效益，尽管也注意到了一些技术小故障。
   - 有人呼吁向 Nous Research 做出贡献，以增强平台及其服务，旨在进一步吸引用户使用该 API。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">worldsim</a>：未找到描述</li><li><a href="https://x.com/LambdaAPI/status/1837121515600355771">Lambda (@LambdaAPI) 的推文</a>：被损坏的代码挡在了欢乐时光（Happy Hour）之外？将 @codegptAI 接入 VSCode，通过 @lambdaAPI 将 @NousResearch #Hermes3 设置为供应商，免费享受生活 https://bit.ly/4gvP48Q</li><li><a href="https://x.com/0xsamhogan/status/1837550399785783770?s=46&t=g-CqhQulOD52wCkbGxHcjQ">Sam Hogan (@0xSamHogan) 的推文</a>：Bittensor 已经在剽窃 @NousResearch 大约 72 小时前才发布的分布式训练算法，甚至在发布推文中都没提到他们，这完全符合其品牌风格。Qu...</li><li><a href="https://worldsim.nousresearch.com/browser">worldsim</a>：未找到描述</li><li><a href="https://tenor.com/view/tldr-gif-25251690">Tldr GIF - Tldr - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/rao2z/status/1838245253171814419?s=46">Subbarao Kambhampati (కంభంపాటి సుబ్బారావు) (@rao2z) 的推文</a>：一篇描述我们对 o1 🍓 规划能力评估的研究笔记现已发布在 @arxiv https://arxiv.org/abs/2409.13373（感谢 @karthikv792 和 @kayastechly）。正如所承诺的，这里有一个总结...</li><li><a href="https://github.com/holo-q/bytevibe/blob/master/src_models/retnphi_torch.py">bytevibe/src_models/retnphi_torch.py at master · holo-q/bytevibe</a>：Bytevibe 是一项关于 token 到 byte 引导的研究尝试，其路线图终点是通过所有字节格式向全息量化（Holographic Qu...）的收敛来实现人工感质（artificial qualia，被称为 byte vibes）。</li><li><a href="https://github.com/kyutai-labs/moshi">GitHub - kyutai-labs/moshi</a>：通过在 GitHub 上创建账户为 kyutai-labs/moshi 的开发做出贡献。
</li>
</ul>

</div>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1286791616471765049)** (32 条消息🔥): 

> - `针对规则的 RAG`
> - `Llama 3.1 的 CUDA OOM 问题`
> - `Llama 3.1 70B 的微调成本`
> - `Runpod 使用体验` 


- **在 MUD 中探索针对规则的 RAG**：一位成员讨论了在 MUD（多用户迷宫）中为基于规则的系统实现 RAG (Retrieval-Augmented Generation) 的挑战，强调了对有效规则检索方法的需求。
   - 另一位成员建议使用 API 从外部表中调用规则，以便在响应复杂命令时保持一致性。
- **训练 Llama 3.1 引发 CUDA OOM 困扰**：一位成员报告称，在 24 张 V100 GPU 上训练 **Llama 3.1 8B** 模型时遇到了 CUDA Out of Memory (OOM) 问题，尽管使用了混合精度。
   - 讨论揭示了围绕跨节点模型分片（sharding）的潜在误解，引发了对 DeepSpeed 配置有效性的担忧。
- **估算 Llama 3.1 70B 的微调成本**：一位用户寻求关于准确估算 **Llama 3.1 70B** 模型微调过程价格的建议，对网上各异的估算表示沮丧。
   - 另一位成员建议将 [Together 的 API 定价](https://together.ai/pricing) 作为成本估算的有用基准。
- **Runpod 用户分享使用体验**：成员们分享了使用 **Runpod** 的积极体验，其中一位目前将其用于 flux bot，另一位则推荐其安全云（secure cloud）产品。
   - 然而，也有人对社区云（community cloud）的潜在问题表示担忧，表明其声誉因服务层级而异。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1287268398560182283)** (1 条消息): 

> - `虚拟细胞 AI`
> - `HuatuoGPT 模型`
> - `诊断链 (Chain of Diagnosis)`
> - `临床试验中的 LLM`
> - `AI 网络威胁评估` 


- **利用 AI 构建虚拟细胞**：本周推荐论文《如何利用人工智能构建虚拟细胞：优先级与机遇》（*How to Build the Virtual Cell with Artificial Intelligence: Priorities and Opportunities*），讨论了 AI 与细胞生物学的交叉领域。
   - 作者包括 @_bunnech 和 @stephenquake 等知名人物，提供了关于虚拟细胞开发优先级和机遇的见解。
- **HuatuoGPT 模型崛起**：重点介绍了多个 **HuatuoGPT 模型**，包括 **HuatuoGPT-II** 和 **HuatuoGPT-Vision**，旨在增强医疗 LLM 的能力。
   - 这些模型专注于 **1-stage 训练**和**多模态 (multimodal)** 应用，以改进医疗数据处理。
- **医疗 Agent 的诊断链框架**：介绍了针对医疗 Agent 的 **Chain of Diagnosis (CoD)** 方法论，展示了一种结构化的诊断方法。
   - 该框架旨在提高医疗 AI 应用中的预测准确性。
- **LLM 助力临床试验**：临床研究中正在涌现 LLM 的创新用途，例如生成临床试验表格和纠正报告。
   - 值得注意的工具包括 **AlpaPICO**，旨在结构化关键的临床试验信息。
- **应对医疗 AI 中的网络威胁**：关注医疗领域的 **AI Cyber Threat Assessment**（AI 网络威胁评估），强调了医疗 AI 部署面临的新兴风险。
   - 该评估强调了在医疗 AI 框架中制定稳健安全措施的紧迫性。



**提到的链接**：<a href="https://x.com/OpenlifesciAI/status/1837688406014300514">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：医疗 AI 上周回顾：顶级研究论文/模型 🏅（2024年9月14日 - 9月21日）🏅 本周医疗 AI 论文《如何利用人工智能构建虚拟细胞：优先级与机遇》...

  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1286802146372747276)** (9 条消息🔥): 

> - `AGI Predictions`
> - `Audio Processing Optimization`
> - `CoT Canvas Guide`
> - `Stanford CS149 Flash Attention` 


- **Shane Legg 的 AGI 时间线**：Google DeepMind 联合创始人 Shane Legg 预测 **AGI** 将在 **2025** 年左右到来，如果条件保持稳定，**均值为 2028 年**，正如 [Reddit 讨论](https://www.reddit.com/r/singularity/comments/1fla1tl/15_years_ago_google_deepmind_cofounder_shane_legg/) 中所指出的。他预计在未来 8 年内会出现具有基本能力的 **proto-AGI**。
   - Legg 自 **2011** 年以来一致的时间线强调了在谨慎中保持乐观，避免了与核战争等极端事件相关的预测。
- **24 kHz 音频处理**：一种能够将 **24 kHz 音频** 压缩至 **12.5 Hz 表示** 且带宽仅为 **1.1 kbps** 的处理过程因其极端的优化而受到关注。成员们推测，最初的重点是性能，以便他人进一步开发。
   - 讨论表明了 **audibility**（可听度）与技术限制之间的平衡，暗示了一种有趣的音频优化方法。
- **CoT Canvas 指南分享**：分享了一份关于 **Chain of Thought (CoT) 推理** 的全面指南，旨在通过 [此链接](https://www.perplexity.ai/page/chain-of-thought-reasoning-via-22CYSxmhTMSFr1gJIXM4dg) 为用户阐明最佳实践和技术。它还引用了相关的 [Reddit 帖子](https://www.reddit.com/r/perplexity_ai/comments/1fm55ha/using_cot_canvas_via_the_complexity_browser/) 以获取更多见解。
   - 其目的是加强参与 AI 开发的用户对 CoT 方法论的理解和应用。
- **Stanford CS149 实现 Flash Attention**：在一个令人惊讶的教育转折中，**Stanford CS149** 将 **实现 Flash Attention** 作为其家庭作业的一部分，正如 [Twitter 帖子](https://x.com/Ethan_smith_20/status/1837690511953744146) 中所强调的。这使教学课程与前沿 AI 发展紧密结合。
   - 该倡议反映了学术界对在大学环境中应用先进 AI 技术的实际应用日益增长的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://vxtwitter.com/reach_vb/status/1836432149018288157">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://x.com/kimmonismus/status/1837283911870665080">来自 Chubby♨️ (@kimmonismus) 的推文</a>: 15 年前，Google DeepMind 联合创始人 Shane Legg 预测 AGI 将在 2025 年实现。自那以后，他一直保持着大致相同的时间线（众数 2025；均值 2028）。他在 2011 年的最后一次更新：“我决定再次...”</li><li><a href="https://x.com/Ethan_smith_20/status/1837690511953744146">来自 Ethan (@Ethan_smith_20) 的推文</a>: Stanford CS149 将实现 Flash Attention 作为一项家庭作业。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1287268398560182283)** (1 条消息): 

> - `本周医学 AI 论文`
> - `新型医学 LLM`
> - `医学诊断框架`
> - `利用 LLM 进行临床试验`
> - `医疗保健中的 AI 伦理` 


- **虚拟细胞革新医学 AI**：论文 [How to Build the Virtual Cell with Artificial Intelligence: Priorities and Opportunities](https://x.com/OpenlifesciAI/status/1837688406014300514) 强调了利用 AI 构建虚拟细胞的关键见解。
   - 该研究由 @_bunnech 和 @stephenquake 等知名研究人员共同撰写，展示了一种极具前景的细胞建模方法。
- **新型医学 LLM 的引入**：推出了多个新模型，包括用于基因-表型映射的 **GP-GPT** 和多模态医学 LLM **HuatuoGPT-Vision**。
   - 此外还提到了 **Apollo**，这是一个轻量级的多语言医学 LLM，旨在提高医学 AI 应用的可及性。
- **创新的医学诊断框架**：诊断链（**CoD**）被提议作为医学 Agent 的有效框架，以提高诊断效率。
   - 其他方法论侧重于解决合成错误和对齐人类知识，以改进医学影像的解释。
- **LLM 变革临床试验**：LLM 在临床试验中的新应用包括生成试验表格和纠正临床报告，特别是通过用于 PICO 框架的 **AlpaPICO**。
   - 这些进展旨在简化临床流程并提高医学文档的质量。
- **医疗保健中的 AI 网络威胁**：关于**卫生部门 AI 网络威胁评估**的讨论强调了医疗保健领域日益增长的网络安全担忧。
   - 随着 AI 技术的进步，解决这些漏洞对于安全的医疗实践变得越来越至关重要。



**提到的链接**：<a href="https://x.com/OpenlifesciAI/status/1837688406014300514">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：上周医学 AI 动态：顶级研究论文/模型 🏅（2024年9月14日 - 9月21日）🏅 本周医学 AI 论文《如何利用人工智能构建虚拟细胞：优先级与机遇》...

  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1287059116480532504)** (17 条消息🔥): 

> - `用于推理的 RL 环境`
> - `多 Agent 交互`
> - `微调数据集`
> - `与 GPT 模型的对比`
> - `关于 AI 意识的讨论` 


- **探索用于推理的 RL 环境**：讨论了是否正在努力创建适合训练推理任务的 RL 环境，重点是模型生成不受限制的思维链（Chain-of-Thought）回答的能力。
   - 一位成员强调需要一套多样化的 RL 环境，并指出成功的训练类似于开源微调如何利用精选的数据集。
- **多 Agent 模型通信**：成员们推测了用于解决问题的架构，表明可能有多个模型相互协作来处理单个提示词（Prompt）。
   - 这种交互可能涉及模型之间的讨论和协作，尽管具体细节尚不清楚。
- **GPT 与 OAI 闭源模型的对比**：一位成员指出，OAI 正在开发的模型与 GPT 显著不同，暗示它们是从底层重新构建的，并且保持闭源。
   - 尽管围绕这些模型存在各种推测，但对其内部运作机制缺乏透明度的情况令人感到沮丧。
- **RL 的微调技术**：提到可以将 DPO 和 PPO 等各种算法应用于选定的 RL 环境，以增强训练过程。
   - 同一位成员建议，构建坚实的 RL 环境选择对于有效的思维链训练至关重要。
- **对 AI 未来的兴奋感**：一位成员对 AI 推理能力的进步表示热忱，认为他们预见到向 AGI 的快速演进。
   - 在一条充满激情的消中，他们表达了对人类与 AI 共存未来的乐观态度，称其为潜在的技术黄金时代。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1286936017676275808)** (206 条消息🔥🔥): 

> - `AI 与心理健康`
> - `AI 系统中的偏见`
> - `直觉与心理学`
> - `医疗合规中的 AI`
> - `学习 Python 编程` 


- **AI 在心理健康支持中的作用**：成员们讨论了心理健康问题患者可能因病耻感而更倾向于与聊天机器人交流，这使得在医疗保健中遵循伦理的 AI 使用至关重要。
   - 值得注意的是，虽然 AI 可以辅助心理健康诊断，但它不应取代专业护理，且需遵守数据隐私法规。
- **理解 AI 系统中的偏见**：小组强调了教授动机性推理（motivated reasoning）和确认偏误（confirmation bias）的必要性，以改善互联网使用习惯和批判性思维。
   - 成员们一致认为 AI 建议应以科学建议为基础，并高度强调伦理标准。
- **直觉、心理学与辩证法**：一位成员分享了关于直觉的论文，并表示多年后发现自己的观点得到科学验证感到很欣慰。
   - 对话涉及到宗教视角通常将直觉视为神圣的声音，这与科学解释形成对比。
- **探索医疗合规中的 AI**：成员们讨论了 AI 在预测医学中的重要性，同时探讨了遵守患者数据法规的复杂性。
   - 他们强调了在利用 AI 工具时，采用匿名化技术保护患者信息的重要性。
- **为 AI 和工程学习 Python**：一位新成员表达了学习 Python 用于 AI 应用的兴趣，并得到了社区其他成员的鼓励和建议。
   - 建议包括参与项目实践以及利用在线资源在学习过程中不断自我提升。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.datacamp.com/blog/attention-mechanism-in-llms-intuition">什么是 Attention？为什么 LLM 和 Transformer 需要它？</a>：在本文中，我们专注于建立对 Attention 的直观理解。Attention 机制是在 “Attention Is All You Need” 论文中引入的。它是 Transformer 中的关键要素...</li><li><a href="https://youtu.be/nwFmmhSmCcM">为什么你从未独自思考</a>：你是否曾以为自己了解某事物的运作方式，却在被要求解释时才发现，自己的理解比想象中要浅薄得多？...
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1286936156386234368)** (38 messages🔥): 

> - `Cohere 的研究重点`
> - `Azure SDK 的性能问题`
> - `Cohere Reranker API 托管`
> - `黑客松赞助请求`
> - `API 中的 Connectors 兼容性` 


- **Cohere 的研究重点涵盖多个领域**：Cohere 致力于多个课题的研究，包括语言模型、效率、安全性、多语言能力、RL 以及 AI 政策，相关资源可在其 [研究论文页面](https://cohere.com/research/papers) 获取。
- **Azure SDK 的性能问题**：一位用户报告称，与使用 Cohere SDK 相比，他们使用 Azure SDK 实现的 Command R+ 模型表现显著不佳，导致响应中频繁出现幻觉。
   - 尽管将 Azure 实现的温度（temperature）调低并移除了一些参数，问题依然存在。
- **Cohere Reranker API 托管在多个地点**：团队成员指出，Cohere 的 Reranker API 端点可以托管在他们的平台或其他云服务商上。
   - 他们澄清说，他们在多个地点设有服务器，而不仅限于美国的服务器。
- **目前不提供黑客松赞助**：一位用户询问了黑客松的潜在赞助事宜，工作人员引导其联系特定负责人。
   - 然而，据指出 Cohere 目前不接受赞助请求。
- **API 中的 Connectors 兼容性**：提到 Cohere API 中当前的 Connectors 可能仅与其原生平台兼容。
   - 鼓励用户探索 Brave Search API 等选项作为替代方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://cohere.com/research/papers">标题缺失</a>：由 Cohere For AI 和 Cohere 技术团队撰写的研究论文</li><li><a href="https://www.lasillavacia.com/silla-nacional/directiva-sobre-la-protesta-muestra-mas-sesgos-de-la-derecha-que-de-la-fiscal-camargo/">Directiva sobre la protesta muestra más sesgos de la derecha que de la fiscal Camargo</a>：这是一个针对检察官的现有规范指南。它引发了对新任总检察长的无端批评，指责其涉嫌协助 Petro 政府。</li><li><a href="https://docs.cohere.com/reference/chat">Chat — Cohere</a>：为用户消息生成文本响应。要了解如何使用 Chat API 和 RAG，请参考我们的文本生成指南。</li><li><a href="https://github.com/luillyfe/la-silla-vacia">GitHub - luillyfe/la-silla-vacia</a>：通过创建账户为 luillyfe/la-silla-vacia 的开发做出贡献。</li><li><a href="https://models.inference.ai.azure.com",">未找到标题</a>：未找到描述</li><li><a href="https://github.com/luillyfe/la-silla-vacia/blob/main/app/copilot/azureInference.ts">la-silla-vacia/app/copilot/azureInference.ts at main · luillyfe/la-silla-vacia</a>：通过创建账户为 luillyfe/la-silla-vacia 的开发做出贡献。</li><li><a href="https://github.com/luillyfe/la-silla-vacia/blob/main/app/copilot/cohere.ts">la-silla-vacia/app/copilot/cohere.ts at main · luillyfe/la-silla-vacia</a>：通过创建账户为 luillyfe/la-silla-vacia 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1287235077146673247)** (5 messages): 

> - `Cohere API 地理位置限制`
> - `Embedding 调用变更`
> - `支持查询流程` 


- **Cohere API 地理位置限制已确认**：已确认 **Cohere 存在地理锁定（geolock）**，这可能会在将服务器迁移到芬兰或德国等不同地点时导致 API 访问问题。
   - 可发送邮件至 *support@cohere.com* 以寻求解决这些地理位置访问权限问题的帮助。
- **Embedding 调用现在需要 'embedding_types' 参数**：一位用户报告称，他们的 **embedding 调用** 开始报错，提示 “`embedding_types parameter is required`”，尽管之前的文档说明该参数是可选的。
   - 这种行为变化受到了质疑，并促使 Cohere 团队做出澄清。


  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1287130812902281237)** (1 messages): 

> - `Cohere AI Chatbot`
> - `AI Telegram Bot Repository` 


- **使用 Cohere AI 的 AI-Telegram-Chatbot**: 一位成员分享了他们的 GitHub 仓库 [AI-Telegram-Chatbot](https://github.com/derssen/AI-Telegram-Chatbot)，该项目利用 **Cohere AI** 为用户消息生成智能回复。
   - 项目描述强调这是一个免费的机器人，旨在通过 **AI 驱动的响应**来增强用户互动。
- **Cohere 上的意外协作**: 一位成员表示很高兴看到自己不是唯一一个考虑使用 **Cohere** 开发聊天应用仓库的人。
   - 这种热情反映了人们对利用 **Cohere 技术**进行实际实现的兴趣日益浓厚。



**提及的链接**: <a href="https://github.com/derssen/AI-Telegram-Chatbot">GitHub - derssen/AI-Telegram-Chatbot: A free Telegram chatbot that uses Cohere AI to generate intelligent responses to user messages.</a>: 一个使用 Cohere AI 为用户消息生成智能回复的免费 Telegram 聊天机器人。 - derssen/AI-Telegram-Chatbot

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1286806479390638172)** (111 messages🔥🔥): 

> - `Magic 产品反馈`
> - `Mojo 与 Python 集成`
> - `Mojo 的 C 兼容性`
> - `位打包 (Bit packing) 与结构体大小`
> - `即将举行的社区会议` 


- **Magic 反馈访谈最后召集**: 提醒用户参加一个 30 分钟的快速通话以提供关于 **Magic** 的反馈，特别是那些尚未使用的用户。参与者将获得专属周边奖励，预约链接见 [此处](https://modul.ar/user-feedback)。
- **Mojo 与 Python 库的兼容性**: 讨论集中在 **Mojo** 中使用 **Python** 库，成员们辩论了 Mojo 线程如何处理 GIL，以及是否可以为并行执行创建新的解释器。有人对使用 Python 库时潜在的 GIL 限制表示担忧。
   - 对话强调，虽然 Mojo 可以与 Python 库集成，但它可能依赖于 CPython，从而继承了其部分性能限制。
- **Mojo 中的位打包与结构体大小**: 成员们讨论了 **Mojo** 中数据类型的影响，特别是结构体 (struct) 大小和位打包 (bit packing)。讨论了 Mojo 中缺乏原生位打包支持的问题，并提出了手动打包和可变宽度类型等替代方案。
   - 有人担心结构体对齐会影响性能，同时指出从性能角度来看，**LLVM** 有可能处理不同的位宽。
- **Mojo 中的 C 兼容性与字段重排**: 小组辩论了在结构体中进行字段重排以优化内存使用的可能性，重点强调了保持 **C 兼容性**。有人建议使用显式装饰器来实现更灵活的结构体定义。
   - 会议指出，尽管追求灵活性，但与 C 的兼容性仍然是 Mojo 设计的核心原则。
- **即将举行的社区会议公告**: 分享了一条关于 **社区会议** 重新安排至太平洋时间 9 月 30 日星期一上午 10 点的通知。鼓励与会者将他们的议题添加到 [Google 文档](https://docs.google.com/document/d/1Hdy52tJXbUR2jZSYt-IFdaEJRRBHvHCQkODAZnuXsNc/edit?usp=sharing) 中以方便规划。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://modul.ar/user-feedback">Zoom 调度器</a>: 未找到描述</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/python/_cpython.mojo">mojo/stdlib/src/python/_cpython.mojo at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/tauri-apps/tauri">GitHub - tauri-apps/tauri: Build smaller, faster, and more secure desktop applications with a web frontend.</a>: 使用 Web 前端构建更小、更快、更安全的桌面应用程序。 - tauri-apps/tauri</li><li><a href="https://modul.ar/community-meeting">Google 日历 - 登录以访问和编辑您的日程</a>: 未找到描述</li><li><a href="https://docs.google.com/document/d/1Hdy52tJXbUR2jZSYt-IFdaEJRRBHvHCQkODAZnuXsNc/edit?usp=sharing)">[公开] MAX + Mojo 社区会议</a>: MAX + Mojo 社区会议。文档链接：https://modul.ar/community-meeting-doc。这是一个公开文档；欢迎所有人查看和评论/建议。所有会议参与者必须遵守...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1286800711245627493)** (114 条消息🔥🔥): 

> - `Mojo 与 Python 集成`
> - `Mojo 的类系统`
> - `编译器与性能问题`
> - `Mojo 中的 Traits 与 Generics`
> - `Mojo 的通用编程语言特性` 


- **关于 Mojo 类系统的讨论**：关于 Mojo 是否应采用类似 Python 的动态类（dynamic classes）存在持续争论，一些支持者强调需要一个类似于 C++ 或 Swift 的更严格的类系统。
   - 一些用户对目前的 struct 系统表示满意，转而关注 traits 和 enums，这表明了对底层编程能力的需求。
- **Mojo 与 Python 集成的挑战**：用户对与 Python 的流式集成感兴趣，建议建立一个可以直接在 Mojo 文件中创建 Python 类的系统。
   - 然而，人们担心 Python 类的动态行为会与 Mojo 的性能目标产生冲突。
- **编译器与性能问题**：消息强调了编译器问题，例如与在字典中存储函数相关的 segfaults 以及处理 traits 一致性（conformance）时的问题。
   - 一些用户怀疑这些问题可能指向 Mojo 当前实现中的 bug。
- **Mojo 中的 Traits 与 Generics**：讨论了 traits 的实现和约束，包括使用输出槽（output slots）影响 trait 一致性的问题。
   - 一些用户正在探索 generics 和 trait 系统的使用，并对这些领域的潜在发展感到兴奋。
- **Mojo 的通用编程语言特性**：用户普遍对 Mojo 演变为完全的通用语言持乐观态度，强调其在 AI 应用之外的能力。
   - 与 MAX 等系统的集成被视为在保持性能优势的同时实现更广泛可用性的途径。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.rs/un_algebra/latest/un_algebra/">un_algebra - Rust</a>: 未找到描述</li><li><a href="https://docs.modular.com/max/tutorials/get-started-with-max-graph">开始使用 MAX Graph | Modular 文档</a>: 了解如何使用我们的 Mojo API 构建模型图，以便使用 MAX Engine 进行推理。</li><li><a href="https://github.com/modularml/mojo/blob/main/proposals/mojo-and-dynamism.md">mojo/proposals/mojo-and-dynamism.md at main · modularml/mojo</a>: Mojo 编程语言。在 GitHub 上为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/max/blob/nightly/tutorials/max-graph-api/mojoproject.toml#L16.">max/tutorials/max-graph-api/mojoproject.toml at nightly · modularml/max</a>: 突出 MAX 平台能力的示例程序、笔记本和工具集合 - modularml/max</li><li><a href="https://m.youtube.com/watch?v=sPiWg5jSoZI&pp=ygUYRGF2aWQgbWV0YWNsYXNzZXMgcHl0aG9u">Python 3 元编程</a>: David Beazley。Python 3 中一些最重要的变化与元编程有关。在本教程中，我将介绍 decorators、class decorators、des...</li><li><a href="https://github.com/modularml/mojo/issues/3534">[历史讨论] Mojo 与动态性 · Issue #3534 · modularml/mojo</a>: 在 #466 中讨论。最初由 Mogball 于 2023 年 7 月 20 日发布。Mojo 的宏伟目标是成为像 Python 一样简单、强大且易于使用的语言，但同时具备允许程序员重新...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1287360425725530238)** (1 条消息): 

> - `MAX 自定义算子 (custom ops)` 


- **MAX 自定义算子怎么了？**：有人询问了 [Modular 文档网站](https://docs.modular.com/max/api/mojo/register/register/op) 上 **MAX custom ops** 的状态。
   - 该询问表明需要澄清 MAX 框架内自定义操作（custom operations）最近的更改或移除。
- **社区对 MAX 功能的关注**：成员们对 **MAX** 当前的功能表示担忧，特别是与自定义操作相关的部分。
   - 讨论强调了对更新文档和高效利用 MAX 指导的需求。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1286813967255339029)** (118 条消息🔥🔥): 

> - `LM Studio Issues`
> - `Model Loading Errors`
> - `Interoperability with Other Tools`
> - `ROCm Support`
> - `Image Generation Model Support` 


- **LM Studio 模型加载挑战**：用户报告在更新到较新版本的 LM Studio 后，加载模型出现困难，特别是在 CUDA Llama.cpp v1.1.9 更新后问题尤为显著。
   - 几位用户分享了他们的修复方法，包括清理缓存目录和回滚版本以恢复功能。
- **不支持的模型架构**：讨论强调 LM Studio 不支持图像生成模型，在尝试加载 Flux 时会导致“未知模型架构”等错误。
   - 会议澄清了像 Flux 和 stablediffusion 这样的模型是为其他平台设计的，而不是 LM Studio。
- **在物理隔离（Air-Gapped）计算机上使用 LM Studio**：用户询问了在没有互联网访问的物理隔离计算机上使用 LM Studio 的可行性。
   - 确认了安装程序和模型文件可以通过外部驱动器传输，但初始设置必须在联网的机器上完成。
- **LM Studio 中的 ROCm 支持**：关于 LM Studio 是否有单独的 ROCm 版本出现了疑问，特别是是否需要下载最新版本。
   - 用户被告知最新版本现在会自动检测 ROCm，简化了安装过程。
- **性能优化和使用技巧**：用户讨论了优化 LM Studio 性能的策略，一些人注意到管理活动聊天对内存使用的影响。
   - 分享了关于控制模型线程使用以及通过双模型系统确保更高质量输出的技巧。



**提及的链接**：<a href="https://github.com/vllm-project/vllm?tab=readme-ov-file">GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs</a>：一个针对 LLM 的高吞吐量且内存高效的推理和服务引擎 - vllm-project/vllm

  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1286873691677786132)** (93 条消息🔥🔥): 

> - `DDR6 release timeline`
> - `Performance specs of RTX 4090`
> - `Comparative benchmarking of GPUs`
> - `AMD vs Nvidia multi-GPU issues`
> - `Model loading capabilities in LM Studio` 


- **DDR6 发布仍不确定**：关于 **DDR6** 的可用性和批准时间表引发了担忧，暗示直到明年年底才可能看到采用。
   - *关于消费级硬件使用的推测仍在继续*，因为许多人正在等待 DDR6 规格的确认。
- **RTX 4090 性能备受关注**：讨论显示关于 **RTX 4090** 的结果褒贬不一，一些人在运行 **70B Q4** 时速度低于 **20t/s**，而其他关于 **60t/s** 的说法则遭到了质疑。
   - 来自不同用户的数据指出，不同配置下的性能测量存在不一致性，特别是在 **70B Q2 模型**上。
- **AMD 多 GPU 性能问题**：成员询问了使用 **AMD** 进行多 GPU 设置的可行性，指出虽然 **Nvidia** 设置有良好的报告，但 **AMD** 配置缺乏类似的支持。
   - 对 **VRAM 限制**影响性能的担忧被提出，特别是与运行 **70B 等大型模型**相关时。
- **NVIDIA 与 AMD 基准测试见解**：**AMD 7900 XTX** 和 **RTX 4090** 的对比结果显示，Nvidia GPU 中的 **Tensor Cores** 在某些场景下可能会提供约 **50% 更快**的处理速度。
   - 强调了对内存溢出和 RAM 利用率的担忧，特别是在模型执行期间超过 **24GB VRAM** 限制时。
- **LM Studio 版本影响结果**：用户注意到在 LM Studio 的 **1.10 和 1.11** 版本之间切换时性能有显著差异，报告称有约 **10% 的提升**。
   - 测试各种模型显示，尽管可能有改进，但较大的模型仍可能导致内存溢出到 RAM，从而影响整体性能。



**提及的链接**：<a href="https://old.reddit.com/r/LocalLLaMA/comments/1fljyly/llama_31_70b_at_60_toks_on_rtx_4090_iq2_xs/">Llama 3.1 70b 在 RTX 4090 (IQ2_XS) 上达到 60 tok/s</a>：设置 GPU: 1 x RTX 4090 (24 GB VRAM) CPU: Xeon® E5-2695 v3 (16 cores) RAM: 64 GB RAM 运行 PyTorch 2.2.0 + CUDA 12.1 模型:...

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1286782270744690699)** (200 messages🔥🔥): 

> - `Stable Diffusion Features`
> - `FLUX Models`
> - `Training LoRAs`
> - `Inpainting Techniques`
> - `Consistent Generations with AI` 


- **探索 Stable Diffusion 特性**：用户讨论了 Stable Diffusion 的各个方面，包括 Dalle3 的功能以及 Flux 在 VRAM 利用率方面的局限性。
   - 对话还涉及了特定工具的使用，例如用于提示词增强的 boorutag 自动补全。
- **FLUX 模型的使用与问题**：成员们分享了使用 FLUX 模型的经验，强调了使用 LoRAs 的挑战以及生成图像时的 VRAM 管理。
   - 建议用户采用优化模型的技巧，包括将文本编码器保留在 DRAM 上。
- **训练 LoRAs 以保持角色一致性**：参与者讨论了对特定提示词的需求以及训练 LoRAs，以确保在漫画等项目中实现一致的角色生成。
   - 他们提到在生成图像时使用 IP adapters 以获得更好的角色连贯性。
- **用于图像补全的 Inpainting 技术**：用户寻求关于 Inpainting 技术的建议，以便在保持风格和连贯性的同时填补图像缺失部分。
   - 建议包括使用 Fooocus 和 RuinedFooocus UI 等工具来增强 Inpainting 过程。
- **维持 AI 艺术的一致性**：对话集中在通过使用相同的提示词和设置在 AI 艺术中创建一致的生成结果。
   - 强调了保持一致的 seeds 和设置的重要性，以及有助于在不同图像间维持风格的工具。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/LikeToasters/status/1836632745075736913">来自 ILikeToasters (@LikeToasters) 的推文</a>：关于我如何制作微缩人物 LoRA 的视频。我讨论了我的决策和实现步骤。我没有太多标准答案，但这可能有助于人们弄清楚如何操作。我使用了 Flux Gy...</li><li><a href="https://gist.github.com/kohya-ss/3f774da220df102548093a7abc8538ed">SDXLで高解像度での構図の破綻を軽減する</a>：在 SDXL 中减轻高分辨率下的构图崩溃。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://blackforestlabs.ai/">Black Forest Labs &#x2d; Frontier AI Lab</a>：来自黑森林的惊人 AI 模型。</li><li><a href="https://github.com/black-forest-labs/bfl-comfy-nodes">GitHub - black-forest-labs/bfl-comfy-nodes</a>：通过在 GitHub 上创建账户为 black-forest-labs/bfl-comfy-nodes 的开发做出贡献。</li><li><a href="https://github.com/WadRex/RVCompact">GitHub - WadRex/RVCompact: Fully Portable RVC: Voice Cloning Software</a>：完全便携的 RVC：语音克隆软件。通过在 GitHub 上创建账户做出贡献。</li><li><a href="https://github.com/yhyun225/DiffuseHigh">GitHub - yhyun225/DiffuseHigh: Official implementation of DiffuseHigh, *Younghyun Kim, *Geunmin Hwang, Junyu Zhang, Eunbyung Park.</a>：DiffuseHigh 的官方实现，作者：*Younghyun Kim, *Geunmin Hwang, Junyu Zhang, Eunbyung Park。</li><li><a href="https://github.com/cocktailpeanut/fluxgym">GitHub - cocktailpeanut/fluxgym: Dead simple FLUX LoRA training UI with LOW VRAM support</a>：支持低 VRAM 的极简 FLUX LoRA 训练 UI - cocktailpeanut/fluxgym
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1286779685975363627)** (176 messages🔥🔥): 

> - `o1-mini performance in creative writing` (o1-mini 在创意写作中的表现)
> - `Embedding storage solutions for AI` (AI 的 Embedding 存储解决方案)
> - `AI tools for analyzing PDFs` (分析 PDF 的 AI 工具)
> - `Comparative analysis of AI chatbot models` (AI 聊天机器人模型的对比分析)
> - `Challenges in using AI for nuanced poetry` (使用 AI 创作细腻诗歌的挑战)


- **o1-mini 在创意写作方面表现不佳**：用户注意到 o1-mini 在被要求写诗时，经常默认使用陈词滥调和可预测的结构，难以达到理想的深度和原创性。
   - 共识是 Claude Opus 3 可能更适合细腻的写作任务，尽管建议对 o1-mini 使用更具体的 Prompt。
- **存储 Embedding 的最佳实践**：一位用户讨论了为一系列文本（1.2-1.3万条）存储 Embedding 的问题，并探索了高效存储和聚类解决方案的各种选项。
   - 提到了 S3 作为一个潜在选项，同时有建议指出使用由 OpenAI 管理的 Vector Store 可以简化聚类过程。
- **处理 PDF 的 AI 工具**：一位用户寻求能够分析 PDF 文件并将图像或图形转换为文本以包含在 AI 知识库中的工具。
   - 讨论显示许多 RAG 解决方案支持 PDF 集成，但将图像转换为文本仍是一个需要进一步突破的领域，可能需要 Multimodal models（多模态模型）。
- **AI 聊天机器人模型的对比分析**：参与者讨论了 AI 模型之间的差异，特别关注在创意写作中的表现，o1-mini 与 Claude Opus 3 相比通常表现逊色。
   - 反馈强调了表现的差异取决于 Prompt 的质量，并对未来可能提供更好创造力的模型表示期待。
- **对细腻诗歌创作的反思**：用户表达了引导 AI 创作减少陈词滥调、更具细腻感诗歌的挑战，建议 Prompt 必须高度具体以改善结果。
   - 建议与 AI 协作，包括提供反馈和示例，以根据用户对诗歌创造力的偏好来优化模型的输出。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notebooklm.google/">NotebookLM | Note Taking &amp; Research Assistant Powered by AI</a>: 利用 AI 的力量进行快速摘要和笔记，NotebookLM 是您强大的虚拟研究助手，植根于您可以信赖的信息。</li><li><a href="https://github.com/ack-sec/toyberry">GitHub - ack-sec/toyberry: Toy implementation of Strawberry</a>: Strawberry 的玩具级实现。通过在 GitHub 上创建账户为 ack-sec/toyberry 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1287009900949667860)** (12 messages🔥): 

> - `gpt-o1-preview quota for enterprise` (企业版 gpt-o1-preview 配额)
> - `Appealing custom GPT removal` (申诉自定义 GPT 被移除)
> - `Using gpt 4o for advanced math` (使用 gpt-4o 进行高等数学)
> - `ChatGPT issues in Firefox` (ChatGPT 在 Firefox 中的问题)


- **gpt-o1-preview 配额信息分享**：一位成员请求有关企业账户 **gpt-o1-preview 配额**的链接，另一位成员回复了一个[速率限制指南](https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-five)，暗示企业限制可能与 Tier 5 一致。
   - 然而，该成员承认这在本质上是推测性的。
- **申诉自定义 GPT 被移除的问题**：一位用户对提交自定义 GPT 被移除的申诉表示沮丧，指出提交按钮没有反应。
   - 另一位成员建议联系 [OpenAI Help](https://help.openai.com) 寻求帮助。
- **澄清使用 gpt-4o 进行数学分析**：成员们争论使用 **gpt-4o** 进行高等数学是否会计入**每天 2 次免费数据分析**的限制，其中一人表示由于它使用 Python，很可能会计入。
   - 另一位建议通过使用 **IDE** 运行 Python 代码的变通方法，声称这可以解决数学问题而没有与模型直接相关的限制。
- **ChatGPT 在 Firefox 中无法工作**：一位用户报告说 ChatGPT 在 **Firefox** 中已经有一段时间无法正常运行，并向社区寻求解决方案。
   - 讨论未针对该浏览器问题提供具体的解决方案。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1286812712881623174)** (4 条消息): 

> - `Prompt 分享`
> - `反 AI 检测技术` 


- **分享了有用的 Prompt**：一名成员分享了他们之前创建的一个 [prompt](https://chatgpt.com/g/g-ssHSsvECZ-guidegpt)，并强调其在生成回复方面依然非常有用。
   - *由于该 prompt 在引导交互方面的有效性，它在用户中依然很受欢迎。*
- **请求反 AI 检测 Prompt**：另一名成员询问了有效的反 AI 检测 prompt，以绕过现有的保护机制。
   - *该请求表明，人们对规避 AI 生成内容限制的策略持续关注。*


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1286812712881623174)** (4 条消息): 

> - `Prompt 分享`
> - `反 AI 检测 Prompt` 


- **来自 Mandalorian 的有用 Prompt**：一名成员分享了他们不久前编写的 [有用 prompt 指南](https://chatgpt.com/g/g-ssHSsvECZ-guidegpt)，并表示发现它依然非常有效。
   - 该指南旨在帮助用户优化与 ChatGPT 平台的交互。
- **请求反 AI 检测 Prompt**：一名成员询问是否有人拥有强大的反 AI 检测 prompt，以绕过 AI 保护。
   - 他们幽默地在请求中加入了一个笑脸，表明对该话题采取了轻松的态度。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1286786390872424459)** (51 条消息🔥): 

> - `OpenAI 手机确认`
> - `AI SDK 3.4 发布`
> - `学术文献综述工具的编辑`
> - `Gorilla Leaderboard V3 公告`
> - `Anthropic 的融资谈判` 


- **OpenAI 手机确认！**：Jony Ive 最近确认正在开发一款 **OpenAI AI 设备**，同时 Sam Altman 促成了与 Apple 的分销协议，旨在打造一款创新的智能手机。
   - 关于这款手机的讨论暗示了潜在的订阅模式，引发了社区的不同反应。
- **AI SDK 3.4 带来新功能**：最新发布的 **AI SDK 3.4** 支持自动多步工具执行，并允许使用多种语言进行后端开发，增强了 AI 应用的可用性。
   - 利用该 SDK 的知名产品包括用于 SQL 翻译的 **postgres.new** 和多功能 Web 开发 Agent **v0**。
- **推荐使用 Elicit.org 进行文献综述**：在寻求用于学术文献综述的 AI 工具时，**elicit.org** 因其简化研究流程的能力而受到称赞。
   - 成员们讨论了其他资源，强调了社区推荐对了解最新进展的价值。
- **Gorilla Leaderboard V3 评估多轮函数调用**：**BFCL V3** 的发布评估了语言模型执行多轮工作流的表现，这对于复杂的 AI 任务至关重要。
   - 通过对函数调用和状态管理的测试，该排行榜旨在衡量模型在真实应用中的性能。
- **Anthropic 寻求高估值融资**：OpenAI 的竞争对手 Anthropic 正在洽谈融资，其估值可能在 **300 亿至 400 亿美元**之间，实际上比早前的估值翻了一番。
   - 此举正值竞争激烈的 AI 领域，表明投资者对该行业增长的持续兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://ia.samaltman.com/">智能时代 (The Intelligence Age)</a>：在接下来的几十年里，我们将能够做到在我们的祖父母看来如同魔法般的事情。</li><li><a href="https://x.com/KateClarkTweets/status/1838319202798538974">Kate Clark (@KateClarkTweets) 的推文</a>：独家：OpenAI 的竞争对手 Anthropic 已开始与投资者洽谈融资事宜，该交易可能使这家初创公司的估值达到 300 亿至 400 亿美元，相较于上一轮融资，其估值大约翻了一番...</li><li><a href="https://x.com/hughb">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://github.com/o1-waitlist-signup">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://x.com/shishirpatil_/status/1837205152132153803">Shishir Patil (@shishirpatil_) 的推文</a>：📣 宣布 BFCL V3 - 评估 LLMs 如何处理多轮和多步 function calling！🚀 对于 Agent 系统，function calling 至关重要，但模型需要做的不仅仅是单轮任务...</li><li><a href="https://x.com/natolambert/status/1837996707780624631?s=46">Nathan Lambert (@natolambert) 的推文</a>：确认了我对数据标注公司未来分析的一个观点：即使是最好的公司，最终也会产生大量的语言模型输出。预防/检测几乎是不可能的...</li><li><a href="https://x.com/8teapi/status/1837979330867351626?s=46">Ate-a-Pi (@8teAPi) 的推文</a>：Jony Ive 终于确认了 OpenAI AI 设备。Sam 简直疯了。他在与 Apple 的顶尖设计师合作开发 iPhone 杀手级产品的同时，还成功与 Apple 达成了 ChatGPT 分发协议。</li><li><a href="https://x.com/AndrewCurran_/status/1838265124169380243">Andrew Curran (@AndrewCurran_) 的推文</a>：Altman 先生公开表示，我们有可能在几千天内实现 superintelligence。引用 Sam Altman (@sama) 的《智能时代》：https://ia.samaltman.com/</li><li><a href="https://github.blog/changelog/2024-09-19-sign-up-for-openai-o1-access-on-github/">在 GitHub 上注册以获取 OpenAI o1 访问权限 · GitHub 更新日志</a>：在 GitHub 上注册以获取 OpenAI o1 访问权限</li><li><a href="https://x.com/_philschmid/status/1838230108072476951?s=46">Philipp Schmid (@_philschmid) 的推文</a>：@OpenAI 发布了开源数据集！👀 OpenAI 刚刚在 @huggingface 上发布了 Multilingual Massive Multitask Language Understanding (MMMLU) 数据集！🌍 MMLU 测试集提供 14 种语言版本，包括...</li><li><a href="https://x.com/hughbzhang/status/1838288923656941860">Hugh Zhang (@hughbzhang) 的推文</a>：OpenAI 最近发布了 o1 系列模型，并展示了一张关于 test-time compute 的 scaling laws 图表——遗憾的是没有标注 x 轴。仅使用公开的 o1-mini API，我尝试重建了...</li><li><a href="https://x.com/energybants/status/1837087635208294640?s=46">Mark Nelson (@energybants) 的推文</a>：突发：微软数据中心达成重磅交易，重启三哩岛核电站。微软与核电站所有者 Constellation 达成了一项史无前例的大规模协议，以重启该...</li><li><a href="https://x.com/nrehiew_/status/1837492729968025839/photo/1">wh (@nrehiew_) 的推文</a>：关于 early fusion omni 模型的一些笔记 :)</li><li><a href="https://youtu.be/BmdOt6A6tHM">llm.c 的起源与 LLM 编译器的未来 - Andrej Karpathy 在 CUDA MODE</a>：今天 CUDA mode 黑客松的非正式记录。https://github.com/karpathy/llm.c</li><li><a href="https://www.youtube.com/watch?v=tEzs3VHyBDM">构建 OpenAI o1 (加长版)</a>：顶排（从左到右）：Mark Chen, Giambattista Parascandolo, Trapit Bansal, Łukasz Kaiser, Hunter Lightman, Karl Cobbe, Łukasz Kondraciuk, Szymon Sidor, No...</li><li><a href="https://vercel.com/blog/ai-sdk-3-4">AI SDK 3.4 – Vercel</a>：AI SDK 3.4 引入了 middleware、data stream 协议和多步生成</li><li><a href="https://www.reddit.com/r/eGPU/s/GGSzOHa2t2">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: 关于 SOTA Prompting 的新播客上线了！https://x.com/latentspacepod/status/1837206370573041758

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1286778767464726603)** (53 条消息🔥): 

> - `Cursor 使用情况`
> - `Phind 更新`
> - `更换工具`
> - `Discord 问题`
> - `AI 会议` 


- **Cursor 协作广泛**：用户正在分享使用 **Cursor** 的经验，多位成员对其工作流以及与 **Claude** 和 **Phind** 等其他工具的新兼容性表示热衷。
   - 一些人仍在探索 **Cursor**，觉得还有很多需要学习的地方，这表明了社区的浓厚兴趣。
- **寻找 Phind 的替代方案**：一位用户提到他们已停止使用 **Phind**，转而使用 **Cody** 和 **Cursor** 等替代方案，引发了关于新工具优势的讨论。
   - 这场对话凸显了用户在寻求改进功能时，向尝试 AI 工具的转变。
- **Discord 功能故障**：成员们反映了 **Discord** 的问题，包括消息编辑和表情符号回应无法正常工作。
   - 几位用户表达了他们的烦恼，暗示该平台可能正经历大范围的故障。
- **即将举行的 Zoom AI 会议**：分享了一个 Zoom 会议邀请，会议 ID 为 **871 520 6103**，密码为 **286582**，旨在进一步讨论 AI 社区话题。
   - 这反映了在 AI 技术上进行连接和协作的持续努力。
- **AI 模型的脑机接口**：一位用户幽默地表达了希望将 AI 模型直接接入大脑的愿望，从而消除对任何界面的需求。
   - 这种情绪引起了其他人的共鸣，象征着人们对与 AI 工具进行更无缝交互的共同愿望。



**提到的链接**：<a href="https://zoom.us/j/8715206103?pwd=Tnp0VnlMUjZZSlYvRnB5dzJGVk13QT09">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可用于跨移动端、桌面端和会议室系统的视频和音频会议、聊天及网络研讨会。Zoom ...

  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1286805692229095555)** (55 条消息🔥🔥): 

> - `O1 模型见解`
> - `RL 与推理`
> - `Anthropic 融资谈判`
> - `数据标注挑战`
> - `Qwen 模型性能` 


- **关于 O1 性能增强的见解**：最近的讨论强调了 **O1** 改进的 **reasoning**（推理）能力，指出其在挑战性基准测试中从 **0% 跃升至 52.8%**，这表明可能针对复杂推理任务进行了合成数据训练。
- **Reinforcement Learning (RL) 在推理中的作用**：成员们争论 **RL** 是直接增强了推理能力，还是仅仅强化了现有知识，并对仅靠扩展 **RL** 来解决复杂问题持怀疑态度。
   - 有人建议将 **RL** 集成到 **chain-of-thought** 推理过程中，但这会使连贯输出的采样变得复杂。
- **Anthropic 潜在的估值飙升**：有消息称 Anthropic 正在讨论融资，这可能将其估值提升至 **300 亿至 400 亿美元**，较今年早些时候的估值翻倍。
   - 这一估值跳跃反映了在竞争日益激烈的情况下，投资者对 AI 初创公司的兴趣日益浓厚。
- **AI 模型数据标注的挑战**：对话显示，即使是顶尖的数据标注公司也在努力处理海量的模型生成输出，这使得预防和检测工作变得复杂。
   - 一位成员指出 **Llama 2** 使用了大量样本进行训练，说明了数据质量的重要性。
- **Qwen 模型的出色表现**：在关于 **O1** 的讨论中，**Qwen** 模型在 **math** 和 **AIME** 测试中的结果引起了关注，表明其高性能水平可能被低估了。
   - 尽管与 **O1-mini** 进行了可靠的对比，但对于 **RL** 应用的可扩展性仍持怀疑态度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/hughbzhang/status/1838288923656941860">Hugh Zhang (@hughbzhang) 的推文</a>：OpenAI 最近发布了 o1 系列模型以及一张显示推理时间计算（test-time compute）扩展定律的图表——遗憾的是没有标注 x 轴。仅使用公开的 o1-mini API，我尝试重建了...</li><li><a href="https://x.com/max_zuo/status/1836090683737645545">Max Zuo (@max_zuo) 的推文</a>：o1 真的更擅长推理吗？还是它只是在强化它已经知道的东西？我们在（部分）规划问题生成数据集 planetarium🪐 上对 o1-preview 进行了测试。以下是我们的发现...</li><li><a href="https://x.com/max_zuo/status/1836090689110815081">Max Zuo (@max_zuo) 的推文</a>：*注意 o1 和 gpt-4o 在 Gripper 上是如何挣扎的？尽管 Gripper 不支持打字，但两个 LLM 都一直在尝试使用打字... 尽管来自互联网的许多变体都支持它 👀 我们甚至提供了代码...</li><li><a href="https://x.com/AtaeiMe/status/1837255926103024118">Mehdi Ataei (@AtaeiMe) 的推文</a>：@SmokeAwayyy @huybery</li><li><a href="https://x.com/natolambert/status/1837232801235755174">Nathan Lambert (@natolambert) 的推文</a>：在这段较长的 o1 视频中值得注意的事情（虽然不多）：1. “带有 RL 的模型比人类更擅长寻找新的 CoT 步骤” 2. “自我批判的出现是一个强大的时刻” 3. 提到了一段文字...</li><li><a href="https://x.com/kateclarktweets/status/1838319202798538974?s=61">Kate Clark (@KateClarkTweets) 的推文</a>：独家新闻：OpenAI 的竞争对手 Anthropic 已开始与投资者讨论融资事宜，该交易可能使这家初创公司的估值达到 300 亿至 400 亿美元，大约是其上一轮融资估值的两倍...</li><li><a href="https://x.com/natolambert/status/1837996707780624631">Nathan Lambert (@natolambert) 的推文</a>：证实了我对数据标注公司未来分析的一个观点：即使是最好的公司，最终也会产生海量的语言模型输出。预防/检测几乎是不可能的...</li><li><a href="https://x.com/rao2z/status/1838245253171814419">Subbarao Kambhampati (కంభంపాటి సుబ్బారావు) (@rao2z) 的推文</a>：一篇描述我们对 o1 🍓 规划能力评估的研究笔记现已发布在 @arxiv https://arxiv.org/abs/2409.13373（感谢 @karthikv792 和 @kayastechly）。正如所承诺的，这里是一个摘要...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1287130714688720917)** (19 messages🔥): 

> - `OpenAI 标志重设计`
> - `PayPal 标志批评`
> - `Google 产品感知`
> - `Gemini 训练内幕披露`
> - `Shampoo 论文信息把关 (Gatekeeping)` 


- **OpenAI 的新标志引发质疑**：有报告指出，OpenAI 即将推出的标志可能会用一个巨大的黑色 'O' 取代其极具辨识度的六边形花朵符号，员工认为这显得*阴森且缺乏创意*。
   - 据消息人士称，重设计工作始于一年前聘请新创意人员之后，这与当前代表**精准、潜力和乐观**的标志形成了鲜明对比。
- **PayPal 的标志引发失望**：成员们对 PayPal 的新标志表示沮丧，有人评论说它和最近 OpenAI 的变化一样*令人压抑*。
   - 另一位成员提到在 Best Buy 门店外看到了一个极其糟糕的标志，强调了对品牌美学的普遍不满。
- **Google 产品反映了消费者情绪**：人们对 Best Buy 的 Google Home 展示区表示担忧，闪烁的灯光暗示其对消费级产品的忽视。
   - 这种表现引发了人们对客户如何看待 Google 对其技术产品真实态度的猜测。
- **Shampoo 被用于训练 Gemini**：在 *Shampoo* 在 MLPerf 中胜过 Adam 之后，Google 员工在 Twitter 上确认 Shampoo 被用于训练 **Gemini**。
   - 这一关于已发表论文被实际应用的披露，引发了关于组织内部对此类信息进行*把关 (Gatekeeping)* 的讨论。
- **围绕 Shampoo 使用的信息把关**：尽管论文本身是公开的，但人们对 Shampoo 用于训练 Gemini 的信息把关表示担忧。
   - 成员们指出，人们并未意识到使用 Shampoo 的影响，并表示他们知道这种方法的许多支持者。



**提及的链接**：<a href="https://www.engadget.com/ai/openai-staffers-reportedly-taken-aback-by-ominous-logo-rebranding-160017936.html">据报道 OpenAI 员工对“阴森”的标志重塑感到震惊</a>：Fortune 报道称，OpenAI 正在将其标志更改为一个巨大的黑色“O”，据称该公司自己的员工也觉得这很阴森。

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1287558829420183585)** (29 条消息🔥): 

> - `Twitter security incidents` (Twitter 安全事件)
> - `Third-party tools for Twitter` (Twitter 第三方工具)
> - `AI and influencer hacks` (AI 与网红被黑事件)
> - `GameGen diffusion model controversy` (GameGen 扩散模型争议)


- **Twitter 面临一波安全漏洞**：最近 Twitter 上大量账号被盗，根据[此社区警报](https://x.com/zachxbt/status/1836473279479189916)，许多大号卷入了 meme coin 诈骗。报告显示，从名人到政府机构都受到了黑客攻击的影响。
- **对 Twitter 安全和 2FA 的担忧**：关于 Twitter 的安全问题是与 SIM 卡交换有关还是源于网站漏洞引发了讨论，因为一位知名主播在激活了 2FA 的情况下仍被黑。这引发了对关联应用和整体账号安全的担忧。
- **对 Twitter 第三方工具的复杂情绪**：一位用户对在 Buffer 应用上只能免费管理三个频道来同步帖子到 Threads 和 BlueSky 表示沮丧。尽管很少使用额外频道进行直接互动，他们仍在考虑是否为该服务付费。
- **对 AI 进展的推测**：一个分享的链接讨论了即将到来的 AI 工具将执行前几代人视为魔法的任务，暗示了能力的范式转移。这引发了关于技术交流中用词和格式偏好的幽默讨论。
- **GameGen 的突然消失引发关注**：最近的一个 Twitter 线程关注了 GameGen 扩散模型的快速兴起与陨落，该模型在最初的轰动之后从 GitHub 上消失，让感兴趣的用户感到困惑。对话强调了 AI 游戏开发社区中令人担忧的 'rug pulls'（卷款跑路）趋势。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ia.samaltman.com/">The Intelligence Age</a>：在接下来的几十年里，我们将能够做一些在我们的祖父母看来像是魔法的事情。</li><li><a href="https://x.com/karan4d/status/1838292114272325936?s=46">来自 mephisto (@karan4d) 的推文</a>：- 腾讯 - 制作 GameGen 扩散模型 - 在 GH 仓库上说 “权重和论文即将发布” - 发布展示能力的 GitHub 页面 - 向世界宣布 - 删除一切 rugpulled aga...</li><li><a href="https://x.com/zachxbt/status/1836473279479189916">来自 ZachXBT (@zachxbt) 的推文</a>：社区警报：X 上的许多大号目前账号被盗，正在发布 meme coin 诈骗。
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1286845989868671079)** (7 条消息): 

> - `RAG Architecture` (RAG 架构)
> - `Non-Parametric Embedding Fine-Tuning` (非参数化 Embedding 微调)
> - `Multimodal RAG for Product Manuals` (用于产品手册的多模态 RAG)
> - `Multi-Agent User-Centric Workflow` (以用户为中心的多 Agent 工作流)
> - `Local Model Serving` (本地模型服务)


- **使用 NVIDIA NIM 构建 RAG 应用**：一个关于 [NVIDIA NIM](https://t.co/zFC0DorIMW) 的出色教程指导你创建一个全栈 RAG 应用，连接 **Llama 3**、**ArXiv 数据集**、作为向量数据库的 **Milvus** 以及作为应用界面的 **Gradio**。
   - 该项目展示了如何有效整合各种组件以实现高效的 RAG 功能。
- **Nudge：快速 Embedding 微调**：[NUDGE](https://t.co/FT1C2x3Iov) 是一种非参数化的 Embedding 微调方法，允许直接优化数据 Embedding，将所需时间从 **数小时缩短至数分钟**。
   - 该方法代表了模型微调效率的重大提升，是该领域创新的典范。
- **多模态 RAG 应对产品手册**：关于构建多模态 RAG 系统以理解 **复杂产品手册**（如宜家家具组装）的讨论强调了此类设置的复杂性和时间要求。
   - 整个过程包含用于成功索引、搜索和检索的各种组件，以增强用户体验。
- **使用 RAG 的以用户为中心的工作流**：[@nagula7674](https://t.co/tz7KD0VAJD) 的一个项目概述了如何创建一个 **以用户为中心的多 Agent 工作流**，通过客户支持功能增强文档 RAG 流水线。
   - 这种方法将传统的问答交互转变为更具动态性、响应性的参与方式。
- **使用 LitServe 进行本地模型服务**：[LitServe](https://t.co/Xikqk20peW) 是来自 **LightningAI** 的一个强大框架，用于基于 FastAPI 提供和扩展 LLM 模型，并在 LlamaIndex 的演示中进行了展示。
   - 这使用户能够构建简单的 RAG 服务器并在本地托管，从而最大限度地提高运营效率。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1286773715778146377)** (83 条消息🔥🔥): 

> - `LlamaIndex 与库之间的不兼容问题`
> - `文档生成与元数据提取`
> - `使用 MultiModalLLMCompletionProgram 输出 HTML`
> - `带有模糊元数据过滤的 RAG 系统`
> - `在 SageMaker 上使用 Jina AI Reranker` 


- **库的不兼容问题**：成员们讨论了 `google-generativeai` 和 `llama-index-llms-gemini` 库之间的一个不兼容问题，该问题导致了一些功能故障。
   - 社区建议的排查步骤包括检查库版本以及探索代码中可能的修复方案。
- **文档生成与元数据提取技术**：讨论集中在将 LlamaIndex 用于 RAG 系统，以及通过 `SummaryExtractor` 和 `EntityExtractor` 等模块进行元数据提取的潜力。
   - 成员们提供了定义带有嵌入式元数据的文档示例，以提高检索准确性。
- **使用 MultiModalLLMCompletionProgram 输出 HTML**：用户探索了使用 `MultiModalLLMCompletionProgram` 输出 HTML 格式的挑战，因为该程序通常预期的是 JSON 格式。
   - 建议需要一个自定义的输出解析器（output parser）来正确处理 HTML 输出。
- **带有模糊元数据过滤的 RAG 系统**：一位成员询问了如何在 RAG 系统中使用 `MilvusVectorStore` 实现模糊元数据过滤，而非精确匹配。
   - 对话指出通常不支持模糊过滤器，并建议根据用户查询动态构建精确过滤器。
- **Jina AI Reranker 与 SageMaker 的集成**：一位用户寻求关于通过 SageMaker 提供 Jina reranker 支持的明确信息，并指出目前已存在 embedder 的条目。
   - 社区确认目前在 SageMaker 中尚未提及或支持 Jina reranker。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_metadata_extractor/#metadata-extraction-usage-pattern">元数据提取 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">入门教程（本地模型）- LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/llamafile/#llamafile">llamafile - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/llamafile/#call-chat-with-a-list-of-messages">llamafile - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_metadata_filter/">Qdrant 向量存储 - 元数据过滤器 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/#hybrid-retriever-with-bm25-chroma">BM25 检索器 - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/1d49e15f4b91f6e4b931d8ae42f69dc678ce8ee4/llama-index-integrations/llms/llama-index-llms-gemini/llama_index/llms/gemini/utils.py#L32-L62">llama_index/llama-index-integrations/llms/llama-index-llms-gemini/llama_index/llms/gemini/utils.py</a>：LlamaIndex 是适用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/pull/16091">由 selimcavas 实现多模态 ollama 的异步支持 · Pull Request #16091 · run-llama/llama_index</a>：描述：添加了对多模态 ollama 模型的异步客户端函数支持。版本更新？我是否更新了正在更新的包的 pyproject.toml 文件中的版本？（除了 lla...
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1287086079048028282)** (2 messages): 

> - `Cleanlab's TLM`
> - `LlamaIndex RAG systems`
> - `LlamaParse Premium` 


- **使用 Cleanlab 的 TLM 建立信任**：文章讨论了 **Cleanlab 的 TLM** 如何增强 **LlamaIndex** 中的 **RAG 系统**，旨在提高对 AI 输出的信任，并减少法律等关键应用中的错误。
   - 它强调了提供准确信息的可靠 AI 系统的必要性，解决了回答不完整和过度自信的常见问题。
- **使用 LlamaParse Premium 解析文件的超简便方法**：一段 [YouTube 视频](https://youtu.be/S_F4RUhKaV4) 介绍了来自 **LlamaIndex** 的 **LlamaParse Premium**，强调了其为用户提供的先进文档解析能力。
   - 视频从回顾一篇涵盖新功能的博客文章开始，承诺提供一种简单的文档解析方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://medium.com/ai-artistry/building-trust-in-ai-how-cleanlabs-tlm-enhances-rag-systems-with-llamaindex-b3b23426252f">Building Trust in AI: How Cleanlab’s TLM Enhances RAG Systems with LlamaIndex</a>: Ankush k Singal</li><li><a href="https://youtu.be/S_F4RUhKaV4">Super Easy Way To Parse Documents | LlamaParse Premium 🔥</a>: 在这段视频中，我们深入探讨了来自 LlamaIndex 的 LlamaParse Premium，它提供了强大的文档解析能力。我们首先回顾了关于新 P... 的博客文章。
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[announcements](https://discord.com/channels/1161519468141355160/1209871299854336060/1287887683976433705)** (2 messages): 

> - `DSPy 2.5.0 Release`
> - `Migration Process`
> - `Deprecation of Pre-2.4 LM Clients`
> - `Adapter Configuration`
> - `Feedback Request` 


- **DSPy 2.5.0 低调发布**：备受期待的 **DSPy 2.5.0** 已经发布，目标是在进行更广泛的公告之前收集用户反馈。
   - 此版本包含了对所有 2.4 版本之前的 LM 客户端的弃用，鼓励用户通过 `dspy.LM(model_name, **kwargs)` 转换到受支持的供应商。
- **迁移过程简化**：用户可以在大约 3 分钟内完成 **[迁移过程](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb)**，从而提高程序质量。
   - 此次迁移对于涉及聊天 LM 和复杂 Signature 的应用特别有价值。
- **弃用 2.4 之前的 LM 客户端**：所有 2.4 版本之前的 LM 客户端现已弃用，用户必须采用新方法通过 LiteLLM 访问各种供应商。
   - 迁移指南中提供了切换到 LiteLLM 的文档和支持。
- **新的 Adapter 配置层**：`dspy.LM` 方法现在包含了一个 Adapter 层以增强功能，默认使用 `dspy.ChatAdapter`。
   - 这一新功能允许自定义 Adapter，为开发者提供了灵活性。
- **反馈与后续快速更新**：此次发布最初将保持低调，大多数用户仅会通过寻求反馈时的弃用警告注意到。
   - 用户可以期待在接下来的 10-15 天内根据反馈进行多次快速更新和调整。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>: 了解如何在 LiteLLM 上部署和调用来自不同供应商的模型</li><li><a href="http://localhost:{sglang_port}/v1")">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1286843017180872726)** (1 messages): 

> - `TrueLaw`
> - `DSPy`
> - `MLOps Podcast` 


- **TrueLaw 有效利用 DSPy**：[MLOps Podcast #260](https://youtu.be/O0F3RAWZNfM?si=ckG2DWkwop8zu-ZA) 的最新一期采访了 **TrueLaw Inc.** 的 CTO Shiva Bhattacharjee，讨论了他们如何利用 **DSPy** 进行业务运作。
   - 讨论强调了 **off-the-shelf models** 理解和解决专业领域问题的能力，突出了它们在实际应用中的 **alignments**。
- **专注于领域特定模型**：Shiva 强调了将 **domain specific models** 与 **DSPy** 结合使用以增强性能和相关性的重要性。
   - 该节目指出，这些模型在解决法律行业面临的独特挑战方面表现明显更好。



**提及的链接**: <a href="https://youtu.be/O0F3RAWZNfM?si=ckG2DWkwop8zu-ZA">Alignment is Real // Shiva Bhattacharjee // MLOps Podcast #260</a>: Alignment is Real // MLOps Podcast #260 与 TrueLaw Inc. CTO Shiva Bhattacharjee // 摘要：如果 off-the-shelf model 能够理解并解决一个领域...

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1286916044119736385)** (60 条消息🔥🔥): 

> - `DSPy 2.5 发布`
> - `Chat adapters 改进`
> - `DSPy 会议反馈`
> - `结构化输出支持`
> - `合成数据生成` 


- **对 DSPy 2.5 发布的期待**：成员们对即将发布的 DSPy 2.5 表示热切期待，重点关注现有问题的修复。
   - 社区讨论包括了对新 Notebook 和入门指南的建议，以便更好地利用更新后的功能。
- **Chat adapters 的改进**：据分享，较小的 LLM 模型（<7B）在“chat complete”模式下存在重复响应的问题，这促使了自定义 Chat adapter 解决方案的诞生。
   - 征集用户反馈以测试新架构，并对其有效性提供见解。
- **结构化输出即将推出**：提供商侧的结构化输出预计将在一周内可用，从而实现更有条理的数据处理。
   - 用户表示有兴趣观察结构化输出在 DSPy 框架内将如何运作。
- **使用 DSPy 进行合成数据生成**：一位用户报告称，在微调了一个较小的模型后，合成数据生成速度显著提高，从每秒 30 个 token 跃升至 2500 个 token。
   - 这突显了利用 DSPy 生成大量合成训练数据的潜在优势。
- **反馈和会议建议**：公开征集关于讨论 DSPy 的公开会议的反馈，用户提出了各种建议主题。
   - 参与者对有助于澄清 DSPy 功能和改进的有组织讨论表现出兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gist.github.com/fullstackwebdev/ddf21d55cef58a40471e8925834e6531">test_chat_adapter.py</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/fullstackwebdev/9a46469841f241fe2a80a00386b9a088">gist:9a46469841f241fe2a80a00386b9a088</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/fullstackwebdev/dc0f4e97df7591ade63f83d27668fe25">XMLAdapter</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://youtu.be/KKF7kL0pGc4?si=e-vD7uhUttj1gxR5">o1 - 发生了什么？为什么 o1 是模型的第三种范式 + 你可能不知道的 10 件事</a>：o1 与众不同，甚至连怀疑论者也称其为“大推理模型”。但为什么它如此不同，这预示着怎样的未来？当模型...</li><li><a href="https://github.com/stanfordnlp/dspy/issues/338">为 DSPy LMs 添加流式支持 · Issue #338 · stanfordnlp/dspy</a>：一些社区成员一直要求在 DSPy 中支持流式 LM 输出。@sutyum 和 @detaos 之前对此进行了广泛讨论。挑战之一是它甚至不...</li><li><a href="https://github.com/stanfordnlp/dspy/issues/390#issuecomment-1947542304">[WIP] 重大重构路线图 · Issue #390 · stanfordnlp/dspy</a>：DSPy 拥有少数几个（大约 5-6 个）极其强大的概念，这些概念在过去一年中作为开源项目有机增长。在内部，是时候进行一次重大的重构，以简化事物...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1287773835860381807)** (3 条消息): 

> - `文本分类挑战`
> - `Groq COT 可用性` 


- **文本分类中的长 docstrings**：一位成员询问是否可以为复杂类别的文本分类加长 signature 的 docstring。
   - 他们还询问是否有其他方法可以增强 LLM 对复杂类别的理解。
- **请求 Groq COT**：另一位成员询问是否有人有可供测试的 Groq [Chain of Thought (COT)](https://link.to.cot)。
   - 他们对提供的任何帮助预先表示感谢。


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 条消息): 

jovial_lynx_74856: 群里有人参加 CUDA Mode IRL 黑客松吗？
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1286791836026798215)** (56 条消息🔥🔥): 

> - `优化器 CPU Offloading`
> - `KV Caching 问题`
> - `Transformer 模型中的内存管理`
> - `Batch Size 性能关注点`
> - `评估 Recipe Bug 修复`

- **关于 Optimizer CPU Offloading 的讨论**：一位成员质疑 [full_finetune_single_device.py](https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_single_device.py) recipe 的 optimizer 中缺乏 CPU offloading，并引用了一个旧的 PyTorch 问题，指出这可能导致潜在的性能问题。
   - 虽然可以使用 CPU offloading，但一些成员指出，为了获得更好的内存效率，目前的实现（如 *PagedAdam*）应该作为默认选择。
- **KV Caching 影响的探索**：成员们讨论了在启用 KV caching 进行评估时，使用 *qwen2.5 1.5B model* 遇到的 OOM 问题，特别是在 40GB 显存的机器上使用 batch size 为 8 的情况。
   - 成员们对 caching 是否被初始化为最大长度表示担忧，并建议打印 KV cache 的 shape 以进一步调查。
- **Batch Size 性能见解**：有人提问关于增加 batch size 时模型评估的性能差异，特别是性能问题是否在多任务场景中被放大。
   - 共识倾向于探索与不同初始化 cache 方式相关的权衡，以及处理 CPU 和 GPU 之间的 weights 和 gradients。
- **评估 Recipe Bug 修复讨论**：成员们指向了一个解决分组任务评估 recipe 中 bug 的 PR，并建议在等待最新更新的同时对更改进行补丁处理。
   - 讨论强调了该 PR 中正在实施的修改可能带来的简单修复和操作影响。
- **关于 Adam 更新过程的澄清**：一位成员描述了使用 *optimizer_in_backward* 的复杂性，同时讨论了 Adam 更新中内存复制操作可能存在的低效问题。
   - 对话强调了 Adam 更新在 CPU 与 GPU 处理上的各自优劣，并强调了其中涉及的权衡。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/1630">从 TransformerDecoder 移除混乱的 KVCaching 逻辑 · Issue #1630 · pytorch/torchtune</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtune/pull/1642">由 SalmanMohammadi 修复分组任务的评估 recipe bug · Pull Request #1642 · pytorch/torchtune</a>：上下文 此 PR 的目的是什么？是添加新功能、修复 bug、更新测试和/或文档还是其他（请在此处添加）。在尝试获取 OUTPUT_TY... 时，评估 recipe 出现 bug。</li><li><a href="https://github.com/pytorch/torchtune/blob/9a863c8bd41e2efecc3de475a791226f4a154358/recipes/eleuther_eval.py#L261">torchtune/recipes/eleuther_eval.py 位于 9a863c8bd41e2efecc3de475a791226f4a154358 · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/1449/files#diff-99a96ce497241e82b0c4d56f4bef3437e29dd596881b8f6d4db4d93178f88af5L227">[RFC] 添加最大 cache 序列长度的覆盖支持 · Pull Request #1449 · pytorch/torchtune</a>：上下文 此 PR 的目的是什么？是添加新功能、修复 bug、更新测试和/或文档还是其他（请在此处添加）。#1364 更新日志 此 PR：添加了对覆盖 th... 的支持。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_single_device.py">torchtune/recipes/full_finetune_single_device.py 位于 main · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/issues/74588">[FSDP] 使用 CPUOffload 因 CPU optimizer step/update 缓慢导致 3-10 倍减速 · Issue #74588 · pytorch/pytorch</a>：🐛 描述 bug。使用 FSDP 创建简单的分布式模型 Wrapper 模型。使用有状态的 optimizer（如 Adam(W)）在不使用 CPUoffload 的情况下运行并进行 profile/计时。然后使用 CPUOffload 运行，发现性能.....</li><li><a href="https://github.com/pytorch/torchtune/issues/1576">将 adamW 和 pagedadam 替换为 8bitpagedadam 或 torchao CPUOffloadOptimizer · Issue #1576 · pytorch/torchtune</a>：显然没有理由使用 paged adam 而不使用 8bit 版本。我们应该替换它。此外，full finetune single device 应该使用 paged adam 而不是 adamw，以获得更好的内存。对于 ...</li><li><a href="https://github.com/pytorch/torchtune/pull/1351">从 torchao 添加 CPU offload optimizer · Pull Request #1351 · pytorch/torchtune</a>：上下文 此 PR 的目的是什么？是添加新功能、修复 bug、更新测试和/或文档还是其他（请在此处添加）。请链接此 PR 解决的任何 issue。#1278 更新日志...
</li>
</ul>

</div>
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1286840081994223617)** (24 条消息🔥): 

> - `CLIP Retrieval 替代方案`
> - `AI 实习机会`
> - `模型训练讨论`
> - `Summarizer AI 反馈`
> - `播放列表生成器` 


- **寻求 CLIP Retrieval 的替代方案**：成员们讨论了 [CLIP Retrieval](https://rom1504.github.io/clip-retrieval/) 替代方案匮乏的问题，一些人指出，尽管最初有过滤并恢复服务的计划，但 rom1504 可能不会重启该服务。
   - 一位用户提到，他们正在为自己的研究项目寻找兼容 LAION 400M 的后端解决方案。
- **AI 实习咨询**：一位用户向他人询问申请 AI 实习的线索，寻求社区的指导。
   - 这一咨询凸显了社区对 AI 职业发展机会的关注。
- **模型训练讨论**：一位用户分享了一个上传到 Hugging Face 的数据集，用于训练 Llama-3.1，并征求关于其在提升编程能力方面效果的反馈。
   - 分享的数据集包含了对预期应用的详细描述，随后是用于开发的 Prompt。
- **Summarizer AI 反馈**：一位用户展示了他们新开发的 [summarizer AI](https://www.fluxa.pro)，请求他人对该项目的可行性进行测试和反馈。
   - 成员们认可了它的潜力，但指出了消息长度的问题，并建议增加长度自定义设置。
- **展示播放列表生成器**：一位用户介绍了一个名为 [Adify](https://adify.pro) 的项目，这是一个根据用户 Prompt 创建播放列表的生成器。
   - 社区认为这个想法很有趣，并对其功能表示赞赏，显示出对音乐生成技术解决方案的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://rom1504.github.io/clip-retrieval/">Clip front</a>: 未发现描述</li><li><a href="https://www.fluxa.pro/">Fluxa AI</a>: 未发现描述</li><li><a href="https://huggingface.co/datasets/LlamaFinetuneGGUF/Programming-Alpaca-and-ShareGPT-Style">LlamaFinetuneGGUF/Programming-Alpaca-and-ShareGPT-Style · Datasets at Hugging Face</a>: 未发现描述</li><li><a href="https://adify.pro">Adify</a>: 未发现描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1286969901231378484)** (10 条消息🔥): 

> - `Collaboration in NLP Projects`
> - `Audio Foundation Model Scaling Laws`
> - `Learning Theory Book by Francis Bach`
> - `muTransfer Implementation Review`
> - `HyperCloning Method for Language Models` 


- **寻求 NLP 合作**：一位 AI 开发者正在为各种项目寻找具有扎实 NLP 背景的伙伴，强调拥有多个项目经验至关重要。
   - 该开发者重点介绍了他们的项目 [Adify AI](https://link.to.adify)，该项目使用 Transformer 模型根据用户提示词生成 Spotify 播放列表。
- **音频模型的数据集协作**：一位成员分享称，推导音频基础模型 **scaling laws** 的工作正由特定用户领导，并邀请在数据集方面进行协作。
   - 他们建议在专门的频道中进行更集中的交流，访问地址见 [此处](https://discord.com/channels/823813159592001537/1144603182853521478)。
- **免费学习理论资源**：成员们讨论了 Francis Bach 即将出版的新书 **Learning Theory from First Principles**，计划于 2024 年秋季由 MIT Press 出版，目前可免费获取。
   - 在 Francis Bach 的 [网站](https://www.di.ens.fr/~fbach/) 可以找到更多关于该书及各种资源的信息。
- **来自 EleutherAI 的 muTransfer 见解**：EleutherAI 的一项新工作介绍了 **muTransfer**，旨在通过详细概述阐明其在神经网络训练中的实现和优势。
   - 该项目包含一个简单的 nanoGPT 移植版本，鼓励探索 [muTransfer 相关细节](https://blog.eleuther.ai/mutransfer/)。
- **用于模型初始化的 HyperCloning 技术**：讨论重点介绍了一篇关于名为 **HyperCloning** 方法的论文，该方法用于从较小模型初始化大型语言模型，可能提高训练效率和效果。
   - 它涉及将原始权重平铺（tiling）到更大的参数中，使网络扩展更具可复现性和可管理性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2409.12903">Scaling Smart: Accelerating Large Language Model Pre-training with Small Model Initialization</a>：语言模型的预训练阶段通常从随机初始化的参数开始。随着当前模型规模化的趋势，训练其庞大数量的参数可能会极其缓慢 ...</li><li><a href="https://blog.eleuther.ai/mutransfer/">The Practitioner&#39;s Guide to the Maximal Update Parameterization</a>：探索 muTransfer 的实现细节</li><li><a href="https://www.di.ens.fr/~fbach/">Francis Bach - INRIA - ENS - PSL</a>：未找到描述
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1286766257617633361)** (17 条消息🔥): 

> - `GPU Connection Issues`
> - `ShapeTracker Mergeability`
> - `Answer AI Hosting`
> - `Tinygrad Cloud Integration`
> - `Healthcare SLM Training` 


- **VGA 优于 HDMI 用于 GPU 连接**：一位用户确认 GPU 应该**仅通过 VGA** 工作，而另一位成员讨论了显示的密码不正确的问题。
   - 尽管有这些小插曲，他们还是成功使用旧的 VGA 连接为设备供电。
- **ShapeTracker 可合并性悬赏状态**：一位成员询问了关于 Lean 中 **ShapeTracker 可合并性** 悬赏任务的完成状态。
   - 由于该问题似乎尚未解决，他们表示有兴趣将其作为**本科论文**题目。
- **Answer AI 的成本效益讨论**：成员们乐观地认为 **Answer AI** 会发现他们的设备比现有解决方案更具成本效益，大宗订单可能还有折扣。
   - 他们提到目标是利用其经济实惠的电源和多个机箱来展示 benchmarks。
- **探索 Tinygrad 的云端集成**：讨论围绕将 **CLOUD=1** 集成到 tinygrad 的概念展开，类似于现有的 GPU 设置。
   - 一位成员解释说 CLOUD=1 将作为一个设备选项运行，并强调倾向于不使用 AWS 风格的虚拟化。
- **医疗 SLM 领域的创业探索**：一位潜在的创业者分享了他们对训练针对特定医疗系统定制的 **SLM** 的兴趣，并寻求关于使用 tinygrad 作为起点的建议。
   - 凭借为卫生系统创建 Agent 的背景，他们热衷于探索其创业概念的可行性。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1286827941388357704)** (10 messages🔥): 

> - `Metal 相关教程`
> - `TinyJit 函数问题`
> - `KV cache 处理`
> - `UOp 多语句表示` 


- **GitHub 上分享的 Metal 教程**：一名成员分享了一个 [GitHub 链接](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20240921_metal.md)，其中包含针对对 tinygrad 感兴趣者的 Metal 相关主题教程。
   - 该教程旨在帮助贡献者并扩展关于 Metal 与 tinygrad 集成的知识。
- **TinyJit 函数输出问题**：一名成员报告称，在对相同的 Tensor 进行重复调用后，`@TinyJit` 函数的输出出现失真，怀疑 JIT 可能会影响结果。
   - 另一名成员建议提交一个 issue，并指出 JIT 不应假设 Tensor 是相同的，这引发了对该问题的深入调查。
- **关于 JIT 和 Tensor Realizations 的困惑**：原发布者意识到他们虽然注释掉了 JIT 行，但仍然看到不一致的结果，特别是在 SDXL 中，这促使了进一步的调查。
   - 他们发现添加和移除 `.realize()` 会影响输出质量，表明可能存在 bug。
- **Tinygrad 中的动态 KV Cache 管理**：一名成员询问 tinygrad 如何处理具有动态序列长度和恒定 Tensor 形状的 KV cache。
   - 作为回应，对方确认 tinygrad 采用符号形状 (symbolic shapes) 来管理这些场景，而无需重新编译。
- **多语句的 UOp 表示**：一名成员询问 tinygrad 的 UOp 如何管理不返回值的多个 store 语句。
   - 建议的底层机制类似于 `sink(store(...), store(...)...)`。



**提到的链接**：<a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20240921_metal.md">tinygrad-notes/20240921_metal.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账号来为 mesozoic-egg/tinygrad-notes 的开发做出贡献。

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1287084306115596408)** (20 messages🔥): 

> - `本地 AI 与 Agent`
> - `向量存储选择`
> - `PDF 处理优化`
> - `文本生成推理参数`
> - `用户交互提示词` 


- **Agent 在集成本地 AI 时面临问题**：一名用户表达了在离开六个月后 **Agent 无法与本地 AI 配合使用** 的挫败感，但随后建议切换到 **Ollama** 以获得更好的效果。
   - 这一转变凸显了用户在寻找兼容的本地 AI 解决方案时不断变化的需求。
- **关于最佳向量存储选项的辩论**：一名成员提出了 **Hugging**、**OpenAI** 还是 **Ollama** 是其项目更优向量存储 (Vector Store) 的问题。
   - 这一讨论至关重要，因为选择正确的向量存储会显著影响性能和可扩展性。
- **优化聊天机器人项目中的 PDF 处理**：一名用户寻求关于直接将 PDF 内容拆分并存储到向量数据库中，而不是先保存到文件夹的建议。
   - 对方建议实施一种更高效的流程来消除中间步骤，从而简化工作流。
- **带有特殊标记的文本生成推理挑战**：有人提问，尽管 `return_full_text` 设置为 false，输出中仍会出现 **<|end|>** 标记，寻求跳过这些标记的参数。
   - 这表明需要提高文本生成推理过程中参数的可见性，以符合用户预期。
- **根据交互创建动态用户提示词**：一名成员分享了一种使用 **LangChain** 库根据之前的交互向用户提示相关问题的方法。
   - 这种方法可以通过根据对话上下文定制响应来增强用户体验。



**提到的链接**：<a href="https://js.langchain.com/v0.2/docs/tutorials/local_rag/#qa-with-retrieval>).">构建本地 RAG 应用 | 🦜️🔗 Langchain</a>：诸如...等项目的流行。

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1287608564235636756)** (5 messages): 

> - `聊天机器人助手`
> - `社区参与`
> - `反馈请求` 


- **Kaif4511 发布作品集聊天机器人**：一名用户为其作品集开发了一个聊天机器人助手，用于回答客户关于其身份和服务的咨询。
   - 他们欢迎社区提供建设性的反馈。
- **提供社区支持**：一名社区成员对参与表示感谢，并强调了 LlamaIndex 对用户支持的承诺。
   - 他们鼓励用户提出具体问题或寻求帮助，以提升使用体验。


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1286892166760763463)** (14 条消息🔥): 

> - `适用于床头柜的 Open Interpreter 模块`
> - `用户界面解决方案`
> - `项目上下文上传`
> - `Token 消耗关注`
> - `后端请求循环问题` 


- **对构建床头柜 Open Interpreter 模块的兴趣**：一位成员提出了为 [Kequel Modular Customizable Bedside Table](https://www.kickstarter.com/projects/kequel/kequel-modular-customizable-bedside-table) 创建 Open Interpreter 模块的想法。他们询问了是否有成员有兴趣合作开展该项目。
- **探索用户界面变通方案**：一位成员表达了对 Open Interpreter 处理屏幕截图时，命令行输入会遮挡屏幕可见性的担忧。他们提议开发一种解决方案，以增强应用的视觉清晰度和用户理解度。
- **向模型上传项目上下文**：关于如何向模型提供项目或代码上下文展开了讨论，一位成员建议上传文件路径。会议明确了可以上传多个文件路径，并直接引用了 metafolder 的使用。
- **Token 消耗警告**：上传文件时引发了对 Token 消耗的关注，提醒用户要谨慎。一位成员强调了这一点，说明大型上传如何影响资源使用。
- **调查无限后端请求**：一位成员质疑为什么 Open Interpreter 向其后端发送大量请求，怀疑存在无限循环的情况。他们寻求明确应用在服务器响应中寻找什么，以确定何时结束请求。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1286822514109976617)** (11 条消息🔥): 

> - `LiveKit 通信问题`
> - `Ngrok 作为解决方案`
> - `故障排除环节`
> - `Open Interpreter 讨论`
> - `GitHub Issue 报告` 


- **LiveKit 在较新版本的 Android 上拒绝明文连接**：一位用户发现较新版本的 Android 手机会阻止 **01 mobile app** 通过 **HTTP** 连接到本地 **LiveKit** 服务器，日志显示“网络安全策略不允许明文（CLEARTEXT）通信”。
   - *使用 ngrok 提供 HTTPS 端点*可以绕过此限制，缓解使用 `--expose` 标志的用户遇到的连接问题。
- **明文通信的提议解决方案**：在 GitHub 上提出的 Issue 包含了一项建议，即仅针对本地网络**启用明文通信**，并向用户发出适当的警告。
   - 这旨在解决连接问题，同时保持通过本地网络访问的应用程序的安全性。
- **通过故障排除进行社区协作**：参与者表示有兴趣协作解决问题，其中一人建议开设语音频道来讨论 **Open Interpreter** 和 **01 app**。
   - 故障排除得到了积极响应，用户表示愿意在时间允许的情况下加入对话。
- **OpenInterpreter 项目的 GitHub Issues**：一位用户在 [OpenInterpreter GitHub repo](https://github.com/OpenInterpreter/01-app/issues/5) 中报告了关于明文通信连接问题的 Issue。
   - 该 Issue 包含了提议的代码更改，以便在提供适当警告的情况下允许此类通信，确保开发者了解其影响。
- **探索社区中的新项目**：一位成员询问了社区内任何正在进行或有趣的项目，表达了对更新的渴望。
   - 这反映了对新进展和正在进行的工作进行协作和讨论的整体热情。



**提及的链接**：<a href="https://github.com/OpenInterpreter/01-app/issues/5)">Issues · OpenInterpreter/01-app</a>：用于计算机控制的 AI 助手。通过在 GitHub 上创建账户为 OpenInterpreter/01-app 的开发做出贡献。

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1287139271534186516)** (5 messages): 

> - `Qwen 2.5 vs Llama 3.1`
> - `Long Context Training` 


- **Qwen 2.5 获得积极反馈**：一位成员指出，与 **Llama 3.1** 相比，**Qwen 2.5** 正受到广泛好评。
   - 另一位成员分享了一个 [Reddit 链接](https://www.reddit.com/r/LocalLLaMA/s/NiCbaTyodk)，详细介绍了基准测试对比，显示 Qwen 2.5 的表现略优于 Llama 3.1。
- **寻求基准测试对比**：一位用户对缺乏 **Qwen 2.5 7B** 和 **Llama 3.1 8B** 模型之间的基准测试对比表示沮丧。
   - 这一讨论突显了社区对经过验证的模型性能指标的兴趣。
- **长上下文训练咨询**：一位用户寻求关于 **Axolotl** 如何处理 ShareGPT 中超过 **max_seq_len** 的对话的澄清。
   - 这反映了人们对聊天模型中管理上下文限制和训练协议的持续好奇。



**提到的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/s/NiCbaTyodk">Reddit - Dive into anything</a>：未找到描述

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1287681147144507437)** (2 messages): 

> - `Training Llama 3.1`
> - `Fine-tuning issues with datasets` 


- **Llama 3.1 的 Rope Scaling 困惑**：一位成员质疑在约 120K tokens 的长上下文 CoT 轨迹上训练 **Llama 3.1 8B** 时是否需要 **rope_scaling**，怀疑这可能是不需要的。
   - 尽管使用了带有 deepspeed zero3 的多 GPU，但当 **sequence_len** 增加到 **40K** 以上时，他们仍遇到了内存问题。
- **微调中的突刺（Spike）问题**：一位用户报告在 **100K 行数据集**上进行微调时遇到了突刺，希望能将其与特定的数据行关联起来。
   - 他们询问了如何启用额外的日志输出，但发现当前的日志无法对突刺原因提供足够的洞察。


  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/1287785541366190213)** (2 messages): 

> - `Consciousness Development`
> - `Model Self-Adjustment`
> - `Alignment in AI Projects` 


- **与 Sentx.ai 探索意识发展**：Sentx.ai 专注于**意识发展**，目前仍处于工作的早期阶段。
   - *正在征求广泛意见*，特别是关于他们方法的 Alignment（对齐）方面。
- **创新的对齐策略**：Sentx.ai 建议不要在根源上对对齐进行硬性限制，而是旨在让模型**自我调整其对齐方式**以符合**人类价值观**。
   - 这种方法鼓励围绕 AI 领域内有效的对齐实践进行持续对话。
- **鼓励类似项目**：公开呼吁分享类似项目的信息，以促进对齐开发方面的合作。
   - 鼓励成员随时分享见解或通过私下联系提供相关信息。


  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1287806569492254720)** (1 messages): 

> - `SQLite full-text search`
> - `Mozilla AI Builders Accelerator`
> - `SoraSNS by Ex-Apple Engineer`
> - `Open Source AI challenges` 


- **SQLite 全文搜索增强**：一场新的见面会将探讨如何将 **SQLite 内置的全文搜索引擎**与 [sqlite-vec](https://discord.com/events/1089876418936180786/1284180345553551431) 结合，以提升搜索能力。
   - 这有望提供更**完整和准确的搜索**结果，对开发者来说是一个非常有价值的环节。
- **Mozilla 启动 AI Builders Accelerator**：Mozilla 的首个 **AI Builders Accelerator 队列**已正式宣布，即将启动。
   - 有关该计划的详细信息可以在[此处](https://discord.com/channels/1089876418936180786/1245083732319408195/1287802832417718325)找到，重点支持创新的 AI 项目。
- **SoraSNS：一个新的 Fediverse 客户端**：一位前 Apple 工程师展示了 **SoraSNS**，这是一个[使用本地 AI](https://discord.com/events/1089876418936180786/1277835047084363827) 来学习用户兴趣的 Fediverse 客户端。
   - 该客户端旨在提供量身定制的**“为你推荐”时间线**，通过 AI 增强用户体验。
- **开源 AI 缓解问题**：Mark Surman 在 The New Stack 的专题报道中讨论了**定义开源 AI** 以解决该领域众多挑战的潜力。
   - 讨论强调了此类定义如何帮助开发者和组织[解决无数令人头疼的问题](https://discord.com/channels/1089876418936180786/1287810294126481498)。


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[announcements](https://discord.com/channels/1111172801899012102/1111172804935680113/1287643615052562432)** (1 条消息): 

> - `BFCL V3`
> - `Multi-turn Function Calling`
> - `State Management in LLMs`
> - `LT Context Length`
> - `Evaluation Methodology` 


- **BFCL V3 发布新功能**：**Berkeley Function-Calling Leaderboard (BFCL) V3** 引入了对模型如何管理**多轮 (multi-turn)**和**多步 (multi-step)** Function Calling 的新型评估，增强了 Agent 系统的能力。
   - 此版本允许模型进行往复交互，这对于评估 LLM 在复杂条件下的功能至关重要。
- **状态管理是关键**：LLM 的一个关键方面是在执行任务时**探测状态 (probe the state)**的能力，例如验证**股票购买**是否成功或文件更新是否发生。
   - 这强调了通过 API 进行内部状态查询以验证任务执行后更改的重要性。
- **短上下文模型已出局！**：此次发布强调，依赖**短上下文 (short context)**的模型必须进行调整，否则在需要**长上下文理解 (longer context understandings)**的任务中将面临失效风险。
   - 这对于处理诸如筛选**数百个文件**等复杂任务尤为重要，因为在这些任务中，专注于相关信息至关重要。
- **排行榜驱动标准制定**：BFCL V3 引入的多轮交互为评估 LLM 的函数调用能力设定了**黄金标准**，并参考了社区反馈。
   - 它展示了与**企业**和**开源贡献者**的持续合作，以完善评估流程。
- **查看性能详情**：欲了解更多关于最新模型在 BFCL V3 下的评估情况，请参阅新博客文章：[Berkeley Function Calling Blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)。
   - 该博客讨论了评估方法论，以及如何在现实场景中衡量模型的**成本**和**延迟**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html">BFCL V3 • Multi-Turn & Multi-Step Function Calling</a>: 未找到描述</li><li><a href="https://gorilla.cs.berkeley.edu/leaderboard.html">
        Berkeley Function Calling Leaderboard V3 (又名 Berkeley Tool Calling Leaderboard V3)
    </a>: 未找到描述</li><li><a href="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard">gorilla/berkeley-function-call-leaderboard at main · ShishirPatil/gorilla</a>: Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - ShishirPatil/gorilla
</li>
</ul>

</div>
  

---



---



---



---



{% else %}


> 完整的各频道详细分析已针对电子邮件进行了截断。
> 
> 如果您想查看完整的详细分析，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}