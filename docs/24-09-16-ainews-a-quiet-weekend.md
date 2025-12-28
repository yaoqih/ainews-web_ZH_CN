---
companies:
- openai
- google-deepmind
- adobe
- mistral-ai
- tencent
- supermaven
- 11x
- cohere
- anthropic
- latent-space-university
- stanford
- microsoft
- mila
- notre-dame
date: '2024-09-17T00:28:09.999129Z'
description: '**OpenAI** 发布了全新的 **o1** 模型，该模型利用强化学习和思维链提示（chain-of-thought prompting）在推理基准测试中表现卓越，并取得了高达
  **120** 分的类智商得分。**Google DeepMind** 推出了 **DataGemma**，旨在通过将大语言模型（LLM）与现实世界数据相连来减少“幻觉”现象，同时还发布了利用扩散方法提升机器人灵巧性的
  **ALOHA** 和 **DemoStart**。**Adobe** 预展示了其 **Firefly AI 视频模型**，具备文本生成视频和生成式扩展功能。**Mistral**
  推出了多模态模型 **Pixtral 12B**，而**腾讯**则展示了 **GameGen-O** 开放世界视频游戏生成模型。来自**斯坦福大学**、**OpenAI**、**微软**、**Mila**
  和**圣母大学**的多篇研究论文聚焦于高级推理、自我验证和反思微调（reflection tuning）技术。**陶哲轩（Terence Tao）**和 **George
  Hotz** 等专家对 o1 的能力发表了虽有分歧但总体乐观的看法。种子轮融资方面，**Supermaven** 筹集了 1200 万美元，**11x** 筹集了
  2400 万美元。'
id: 1e5313c1-2e4c-43db-8f18-db7ad7783b3e
models:
- o1
- datagemma
- aloha
- demostart
- firefly-ai-video-model
- pixtral-12b
- gamegen-o
original_slug: ainews-a-quiet-weekend-8098
people:
- george-hotz
- terence-tao
- adcock_brett
- rohanpaul_ai
- bindureddy
- fchollet
- philschmid
title: 一个安静的周末
topics:
- reinforcement-learning
- chain-of-thought
- reasoning
- robotics
- diffusion-models
- multimodality
- video-generation
- model-training
- reflection-tuning
- mathematical-reasoning
- model-benchmarking
- fine-tuning
---

<!-- buttondown-editor-mode: plaintext -->**Patience is all you need.**

> 2024年9月13日至9月16日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**220** 个频道，**6976** 条消息）。预计节省阅读时间（按 200wpm 计算）：**757 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

大家整个周末都在探索 o1，目前的评价[相当两极分化](https://x.com/swyx/status/1834967538234802503)：


![image.png](https://assets.buttondown.email/images/7be82562-b82f-42eb-b735-725a8679e6e5.png?w=960&fit=max)


[天体物理学博士们](https://www.youtube.com/watch?v=M9YOO7N5jF8)、[George Hotz](https://x.com/realGeorgeHotz/status/1835228364837470398) 和 [Terence Tao](https://news.ycombinator.com/item?id=41540902) 都很喜欢它，还有人手动[在自定义 IQ 测试中给它打出了 120 分](https://x.com/maximlott/status/1834652893229859212)。

其他新闻：

- Supermaven [宣布了由 Bessemer 领投的 1200 万美元种子轮融资](https://x.com/supermavenai/status/1835743882971426837?s=46)
- 11x [宣布了由 Benchmark 领投的 2400 万美元 A 轮融资](https://x.com/11x_official/status/1835711787712582082?s=46)
- Luma Labs 推出了 [Dream Machine 的 API](https://x.com/lumalabsai/status/1835742651662139529?s=46)
- [Cohere](https://x.com/maartengr/status/1835709176703508688?s=46)、[Anthropic](https://x.com/alexalbert__/status/1835717512404914401?s=46) 和 [Latent Space University](https://x.com/TheNoahHein/status/1835409949976838239) 推出了课程。

让人不禁好奇，即将推出的 Gemini 2 究竟需要多强大才能与 o1 媲美……

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

**AI 模型进展与行业动态**

- **OpenAI 的 o1 模型**：OpenAI 发布了名为 "o1" 的新模型（也称为 Project Strawberry/Q*），该模型使用强化学习和 Chain-of-Thought（思维链）在回答前进行“思考”。[@adcock_brett](https://twitter.com/adcock_brett/status/1835348649275957643) 指出它打破了推理基准。根据 [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1835317352478445648) 的说法，该模型在 IQ 测试中获得了 35 题中的 25 个正确答案，超越了大多数人类。

- **Google DeepMind 进展**：
  1. Google 推出了 DataGemma，旨在将 Large Language Models 与现实世界数据连接起来，目标是减少 AI 幻觉 [@adcock_brett](https://twitter.com/adcock_brett/status/1835348816037351875)。
  2. DeepMind 展示了两个新的 AI 系统，ALOHA 和 DemoStart，利用 Diffusion 方法提升了机器人的灵巧性 [@adcock_brett](https://twitter.com/adcock_brett/status/1835348694289248382)。

- **其他行业动态**：
  1. Adobe 预览了其 Firefly AI Video Model，具有 Text to Video、Image to Video 和 Generative Extend 等功能 [@adcock_brett](https://twitter.com/adcock_brett/status/1835348761767280904)。
  2. 法国 AI 初创公司 Mistral 发布了 Pixtral 12B，这是一个能够同时处理图像和文本的多模态模型 [@adcock_brett](https://twitter.com/adcock_brett/status/1835348861285490903)。
  3. 腾讯展示了 GameGen-O，一个“开放世界视频游戏生成”模型 [@adcock_brett](https://twitter.com/adcock_brett/status/1835348906579792247)。

**AI 研究与论文**

- 重点介绍了多篇可能有助于理解 OpenAI o1 模型的论文，包括：
  1. 斯坦福大学的 "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking"
  2. MultiOn/斯坦福大学的 "Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents"
  3. OpenAI 的 "Let's Verify Step by Step"
  4. 微软、Mila 的 "V-STaR: Training Verifiers for Self-Taught Reasoners"
  5. 圣母大学、腾讯的 "Learn Beyond The Answer: Training Language Models with Reflection for Mathematical Reasoning" [@_philschmid](https://twitter.com/_philschmid/status/1835251842860646548)

- 提到了一篇关于 "Selective Reflection-Tuning" 的论文，描述了 2023 年 Reflection-Tuning 方法的改进版本 [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1835441647301149014)。

**AI 能力与基准测试**

- [@bindureddy](https://twitter.com/bindureddy/status/1835365835617223045) 声称 AI 的 IQ 已达到 120，超越了大多数人类，但指出它在感知和环境理解方面仍有欠缺。

- [@fchollet](https://twitter.com/fchollet/status/1835417474420056515) 评论说，虽然 AI 可以泛化，但仅限于局部泛化，在面对简单的问题修改或新颖问题时仍然会失效。

- 著名数学家陶哲轩（Terence Tao）对 o1 的数学能力发表了评论，结论褒贬不一，但总体持乐观态度 [@mathemagic1an](https://twitter.com/mathemagic1an/status/1835398044608860270)。

**行业观点与辩论**

- 关于 "Large Language Models" (LLMs) 这一术语的讨论，有人认为它正变得名不副实 [@karpathy](https://twitter.com/karpathy/status/1835451058086347110)。

- [@ylecun](https://twitter.com/ylecun/status/1835303018914324689) 批评对非时间序列进行 Auto-regressive 预测是“纯粹的谬误”。

- Sam Altman 评论说，o1 标志着一个重要新范式的开始，并就 AI 进展表示“未来几年我们已稳操胜券” [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1835295597571481999)。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Llama 3.1 405B：GPT-4 的开源竞争对手**

- **Llama 405B 在本地运行！** ([Score: 81, Comments: 23](https://reddit.com//r/LocalLLaMA/comments/1fhdkdw/llama_405b_running_locally/)): 该帖子展示了 **Llama 405B** 在 **Apple Silicon** 硬件上本地运行的情况，具体包括 **Mac Studio M2 Ultra** 和 **Macbook Pro M3 Max**，速度达到了 **2.5 tokens/sec**。该设置由 **Exo** ([https://github.com/exo-explore](https://github.com/exo-explore)) 和 **Apple MLX** 作为后端引擎驱动，Apple MLX 的创建者还分享了一个重要的优化技巧，即通过设置特定的 **sysctl** 参数来提升性能。
  - 通过向集群中添加一个**配备 3090 GPU 的 Linux 系统**，**Llama 405B** 的性能得到了进一步提升，达到了 **153.56 TFLOPS**。该设置使用 **wifi** 进行设备间的连接。
  - 该项目利用了 **4-bit quantization**，通过 GPU 的吞吐量接近 **500GB/sec**。开发者正在探索使用 **tinygrad** 集成 **Nvidia 3090**。
  - 虽然 **2.5 tokens/sec** 的速度被认为是可以接受的，但在 Prompt 仅有 6 个 token 的情况下，**30.43 秒**的首个 token 响应时间被指出是一个局限。用户可以通过 [Exo GitHub repository](https://github.com/exo-explore/) 尝试该设置。

- **[我用我的小规模基准测试运行了 o1-preview，它的得分与 Llama 3.1 405B 几乎完全一致](https://i.redd.it/guc08wepqyod1.png)** ([Score: 169, Comments: 52](https://reddit.com//r/LocalLLaMA/comments/1fhawvv/i_ran_o1preview_through_my_smallscale_benchmark/)): **Llama 3.1 405B** 和 **OpenAI** 的 **o1-preview** 模型在一次小规模基准测试中获得了几乎相同的分数。基准测试结果表明，**o1-preview** 可能是 **Llama 3.1 405B** 的一个微调版本，这可能暗示了 **Meta** 与 **OpenAI** 之间的合作。这种性能对等也意味着 **o1-preview** 在某些任务中可能达到了 **GPT-4** 的水平。
  - 基准测试的创建者 **dubesor86** 分享了[完整的基准测试结果](https://dubesor.de/benchtable)，并指出由于**严格的额度限制**，测试成本非常昂贵。模型之间的**价格差异**归因于基础成本乘以所使用的不可见 token 的数量。
  - 几位用户对 **Claude 3.5 Sonnet** 在编程基准测试中出人意料的低表现提出了质疑，特别是与大众共识和个人体验相比。基准测试创建者强调，结果会根据具体用例和技能水平而有所不同。
  - 用户讨论了通过使用类似于 **o1** 的 **Chain of Thought (CoT)** 提示词来提高 **Llama** 在推理任务中表现的潜力。基准测试创建者对此表示感兴趣，但更倾向于在官方结果中保持模型的默认行为。


**主题 2. O1 模型的先进推理能力**

- **[受新 o1 模型的启发，Benjamin Klieger 在 @GroqInc 上利用 Llama-3.1 快速开发了 g1](https://x.com/BenjaminKlieger/status/1834946629126046145)** ([Score: 260, Comments: 58](https://reddit.com//r/LocalLLaMA/comments/1fhtpwg/inspired_by_the_new_o1_model_benjamin_klieger/)): Benjamin Klieger 开发了 **g1**，这是一个受 **O1** 启发并在 **Groq** 硬件上由 **Llama-3.1** 驱动的模型。该实现旨在利用 Llama-3.1 架构复制 O1 的推理能力，从而可能在替代基础设施上提供类似的性能。
  - Benjamin Klieger 的 **infinite bookshelf** 项目引起了关注，讨论集中在其对 **Groq** 的依赖以及本地实现的潜力。一位用户分享了一个有趣的**晚餐模拟**，参与者包括历史人物和来自未来的 AI。
  - 用户辩论了仅通过提示词复制 **O1 性能**的有效性，质疑带有多步训练数据的**强化学习**是否对 O1 的能力至关重要。一些人建议使用 **Chain of Thought (CoT)** 输出进行进一步的模型微调。
  - 提议的使用 **JSON** 格式进行逐步解释的**推理提示词**遭到了批评，用户指出强制模型以 **JSON** 格式响应会**降低回答质量**，特别是对于像 Llama 这样的小型模型。

- **[这是揭示 o1 思考步骤的方法吗？](https://i.redd.it/m4nj1hb5zxod1.png)** ([得分: 92, 评论: 41](https://reddit.com//r/LocalLLaMA/comments/1fh8n8k/is_this_a_way_to_reveal_o1s_thinking_steps/)): 该帖子讨论了一种通过 **prompt engineering 技巧** 揭示 **o1 思考步骤** 的潜在方法。该技术涉及要求 o1 解释其任务每一步的推理过程，旨在理解 AI 的决策过程。然而，这种方法在真实揭示 o1 内部思维过程方面的有效性仍不确定。
  - 用户建议 **o1 的思考步骤** 可能由一个 **较小的 LLM** 进行总结，这使得揭示真实的内部过程变得困难。一些人推测这可能是一个 **agentic system** 或由 **专门的 Agent** 协调任务。
  - 试图揭示 o1 的 **chain of thought** 可能会导致 **OpenAI** 发出取消 o1 访问权限的威胁。用户报告收到了警告此类尝试的电子邮件，导致对该模型的探测减少。
  - 关于 o1 能力的理论包括一种潜在的 **带有 reflection tokens 的算法**，允许在 **inference** 期间进行递归循环，以及通过训练来识别并避免响应“不良”指令，同时保持对这些指令的内部模型。


**主题 3. 在线 LLM 提供商和服务对比**

- **大型 LLM 提供商，你使用哪一个以及为什么？** ([得分: 46, 评论: 39](https://reddit.com//r/LocalLLaMA/comments/1fhv2t0/large_llm_providers_which_one_do_you_use_and_why/)): 该帖子讨论了为无法在本地运行大型模型的用户提供的 **各种大型语言模型 (LLM) 提供商**，提到了 **Together, Poe, You.com, Groq, OpenRouter 和 Fireworks** 等选项。作者对 **Poe** 与原始模型相比输出长度缩短表示不满，并寻求其他提供商的建议，询问选择付费服务的标准，以及如何识别那些使用未经修改且没有人工缩短输出的 LLM 提供商。
  - **OpenRouter** 因其丰富的模型种类、定价选项和免费选择而受到高度推荐。用户赞赏其负载均衡功能以及在不更改 **API** 请求的情况下切换支持模型的能力。
  - 几位用户更喜欢组合使用提供商，包括 **OpenAI, Anthropic, Google, Together.AI 和 vast.AI/RunPod**。这种方法可以获得 **SOTA** 性能、免费选项以及运行独特模型的能力，每月费用通常在 **$15** 以下。
  - **Google Gemini** 和 **Cohere** 因其免费计划而受欢迎，而一些用户则选择本地解决方案（如 **Ollama**）或开源替代方案（如 **open-webui**），以避免订阅费并保持数据控制。


- **[我对 o1-preview 进行了小规模基准测试，其得分与 Llama 3.1 405B 几乎相同](https://i.redd.it/guc08wepqyod1.png)** ([得分: 169, 评论: 52](https://reddit.com//r/LocalLLaMA/comments/1fhawvv/i_ran_o1preview_through_my_smallscale_benchmark/)): **o1-preview** 在一次小规模基准测试中表现与 **Llama 3.1 405B** 几乎完全一致。该基准测试包括 **算术**、**常识推理** 和 **语言理解** 等各种任务，两种模型在各项测试中都取得了相似的分数。这表明 o1-preview 可能是 Llama 3.1 405B 的有力竞争替代方案，尽管需要对更大规模的基准测试进行进一步测试以确认这些初步发现。
  - 基准测试的创建者 **dubesor86** 分享了 [完整基准测试结果](https://dubesor.de/benchtable)，并指出由于 **严格的限制 (harsh caps)** 和 **不可见 token (invisible tokens)**，测试成本非常昂贵。模型之间的定价差异归因于基础成本乘以 **token** 使用量。
  - 用户质疑 **Claude 3.5 Sonnet** 在代码任务中的表现不佳，这与他们的个人体验相反。基准测试创建者强调，结果因具体用例而异，“编程”是一个具有多样化需求的宽泛术语。
  - **o1-preview** 的基准测试成本大约比测试 **Llama 3.1 405B** 贵 **52 倍**。用户对测试方法表示关注，包括本地构建、租用实例和 **API** 使用。


**主题 4. 本地 LLM 工具和应用的进展**

- **[分享我的屏幕分析叠加 (Screen Analysis Overlay) 应用](https://v.redd.it/ytd56z6y6zod1)** ([得分: 58, 评论: 10](https://reddit.com//r/LocalLLaMA/comments/1fhcus6/sharing_my_screen_analysis_overlay_app/)): 该帖子介绍了一款 **屏幕分析叠加应用**，旨在配合 **本地 LLM** 进行实时屏幕分析。该应用捕获屏幕，通过本地 LLM 进行处理，并将结果显示为叠加层，允许用户在与计算机交互的同时接收关于屏幕内容的 AI 驱动见解。开发者提到计划将该项目开源，并寻求关于潜在用例和改进的反馈。

- **我大幅更新了我的 Python 程序，它允许通过 llama.cpp 运行的本地 LLM 在互联网上查找信息，现在它能够完整地对最相关的结果进行网页抓取（web scraping）！** ([Score: 133, Comments: 19](https://reddit.com//r/LocalLLaMA/comments/1fhaqjg/i_massively_updated_my_python_program_that_allows/))：作者显著更新了他们的 **Python 程序**，该程序使通过 **llama.cpp** 运行的 **local LLMs** 能够访问互联网信息，现在支持对最相关的搜索结果进行完整的 **web scraping**。该程序允许 **LLM** 选择搜索查询，从 10 个结果中挑选 **2 个最相关的结果**，从这些结果中收集信息，并进行进一步搜索或回答用户的问题。此次更新还包括一个 **llm_config.py** 文件，用于自定义 **llama.cpp settings** 并启用 **GPU support**。更新后的项目已在 [GitHub](https://github.com/TheBlewish/Web-LLM-Assistant-Llama-cpp) 上发布。
  - 用户对该项目表示赞赏，其中一位建议增加 **OpenAI compatible API endpoints** 以提高可用性。作者同意着手实现这一功能，并指出这大约需要“几周时间”。
  - 讨论显示 **llama-cpp-python** 具有内置的 **OpenAI compatible API**，这可以作为将该项目集成到更大规模个人助手工作中的起点。用户强调了在带有 OpenAI API 的服务器上运行 llama.cpp 的潜在性能优势。
  - 讨论中提供了一个详细的实现建议，包括 **spin up the server**、**modularize the code** 以及 **refactor get_llm_response()** 以查询 API 终端。评论者称赞了该项目的简洁性和实现方法。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型进展与能力**

- **OpenAI 的 o1 模型在推理和编程能力方面表现出显著提升**：多篇帖子强调了 o1 的能力，包括[从零开始创建视频游戏](https://www.reddit.com/r/singularity/comments/1fhukuh/o1preview_made_a_3d_fps_game_fully_in_html_i_have/)、[生成复杂的动画](https://www.reddit.com/r/singularity/comments/1fhbc4m/solar_system_animation_made_entirely_with/)以及[执行大规模代码重构](https://www.reddit.com/r/OpenAI/comments/1fhjfln/i_used_o1mini_every_day_for_coding_since_launch/)。该模型在需要长时间推理的任务中表现尤为出色。

- **AI 能力的飞速进步**：帖子讨论了[据报道 o1 的 IQ 提升了 30 点，达到 120 IQ](https://www.reddit.com/r/singularity/comments/1fhi6k9/openais_new_model_leaped_30_iq_points_to_120_iq/)，超越了 90% 的人类。另一篇帖子提到 OpenAI 的路线图暗示模型很快将达到[博士级推理水平并具备 Agent 般的能力](https://www.reddit.com/r/singularity/comments/1fhn6wo/david_sacks_says_openai_recently_gave_investors_a/)。

- **多模态 AI 的改进**：一篇 [Google Deepmind 论文](https://arxiv.org/html/2406.17711v1)展示了通过联合样本选择在多模态学习方面取得的进展。

**AI 研究与基础设施**

- **前沿 AI 模型对计算能力的巨大需求**：Oracle 的 Larry Ellison [讨论了建造核反应堆为大型 GPU 集群供电的计划](https://www.reddit.com/r/singularity/comments/1fh8ofk/larry_ellison_says_oracle_is_building_nuclear/)，估计 3 年内成本将达 1000 亿美元，以在 AI 开发中保持竞争力。

- **AI 推理速度的突破**：[Microsoft 的 MInference 技术](https://arxiv.org/abs/2407.02490)能够在保持准确性的同时，为长上下文任务实现高达数百万 Token 的推理。

- **合成数据创建的新方法**：一篇[关于扩展合成数据创建的论文](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/)利用 LLM 中的多样化视角，从 10 亿个网络策划的 Persona（人格角色）中生成数据。

**AI 模型发布与对比**

- **Salesforce 发布 xLAM-1b**：尽管规模较小，这个 10 亿参数的模型在 [Function Calling（函数调用）方面实现了 70% 的准确率，超越了 GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。

- **现有模型的更新**：Rubra AI 发布了更新后的 [Phi-3 Mini 模型，具备 Function Calling 能力](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)，可与 Mistral-7b v3 竞争。

- **模型间的对比**：一份关于 [o1-mini 和 Claude Sonnet 3.5 在编程任务中的详细对比](https://www.reddit.com/r/OpenAI/comments/1fhjfln/i_used_o1mini_every_day_for_coding_since_launch/)突出了各模型的优缺点。

**AI 的社会与经济影响**

- **潜在的就业市场冲击**：一份报告指出 [AI 将在一年内影响 6000 万个美国和墨西哥的工作岗位](https://www.reddit.com/r/singularity/comments/1fhiv8f/artificial_intelligence_will_affect_60_million_us/)。

- **关于 AI 对各行业影响的辩论**：围绕 AI 进展如何影响[软件开发](https://www.reddit.com/r/singularity/comments/1fhukuh/o1preview_made_a_3d_fps_game_fully_in_html_i_have/)和其他知识型工作的讨论。

**AI 应用与工具**

- **AI 生成内容创作**：示例包括[用于图像生成的微缩人物 LoRA](https://www.reddit.com/r/StableDiffusion/comments/1fhx97g/miniature_people_flux_lora_coming_very_soon/)以及[用于心理健康支持的肯定卡片](https://www.reddit.com/r/StableDiffusion/comments/1fhrgko/help_combat_mental_health_with_my_affirmation/)。

- **AI 辅助编程与开发**：多篇帖子展示了 AI [生成复杂应用](https://www.reddit.com/r/singularity/comments/1fhukuh/o1preview_made_a_3d_fps_game_fully_in_html_i_have/)以及[辅助大规模重构任务](https://www.reddit.com/r/OpenAI/comments/1fhjfln/i_used_o1mini_every_day_for_coding_since_launch/)的能力。


---

# AI Discord 回顾

> 由 o1-preview 生成的摘要之摘要的摘要

**主题 1：OpenAI 的 o1 模型引发 AI 社区辩论**

- [**O1 模型在给人留下深刻印象的同时也同样令人失望**](https://openai.com/o1)：OpenAI 的新 **O1 模型**（**o1-preview** 和 **o1-mini**）引起了轰动，一些用户称赞其推理能力，而另一些用户则认为其回答过于**机械化**且不尽如人意。这些模型褒贬不一的反响突显了在推进 AI 推理方面持续存在的挑战。

- [**社区质疑 O1 相较于现有模型的优势**](https://x.com/aidan_mclau/status/1835729356406329372?s=46)：用户正在将 **O1** 与 **GPT-4o** 等模型进行比较，争论 **O1 的思维链推理（chain-of-thought reasoning）** 是否带来了显著改进，还是仅仅是炒作。讨论集中在 **O1** 在复杂任务中的表现及其在现实世界中的适用性。

- [**关于 O1 开发和数据使用的猜测不断涌现**](https://www.interconnects.ai/p/reverse-engineering-openai-o1)：爱好者们正在对 **O1** 进行**逆向工程（reverse engineering）**，以了解其训练过程以及对用户交互数据的依赖。对**隐私**的担忧以及在开源模型中复制 **O1** 能力的可行性引发了激烈的辩论。

**主题 2：AI 编程工具改变开发工作流**

- [**Aider 和 O1 在 Bug 修复方面胜过竞争对手**](https://aider.chat)：开发者们正在庆祝 **Aider** 和 OpenAI 的 **O1** 在 Bug 修复方面的表现优于 **Claude** 等模型。这些工具提供详细的、分步的输出，简化了复杂代码库中的故障排除工作。

- [**Cursor AI 轻松应对大规模代码库编辑**](https://www.cursor.com/blog/instant-apply)：**Cursor AI** 正在解决令 **O1** 等模型望而却步的大规模代码编辑挑战。其专门的编程助手通过更高效地处理重大变更来提高生产力。

- [**AI 在编程中日益增长的角色引发就业市场担忧**](https://x.com/sama/status/1834276403270857021)：围绕 **AI 可能取代初级开发人员**的讨论正在加剧，引发了关于人类在编程中未来角色的对话。重点在于促进 **AI 与人类协作**，以保持资深开发人员的竞争力。

**主题 3：模型微调与训练依然复杂**

- [**对表现不佳模型的挫败感与日俱增**](https://huggingface.co/models)：**Gemma2**、**Mistral** 和 **Phi 3.5** 等模型在训练期间表现不佳，导致用户感到恼火。挑战包括高 **VRAM 占用**和**不理想的输出**，突显了对更好训练解决方案的需求。

- **LLama 3.1 成为一个亮点**：在普遍存在的训练问题中，**LLama 3.1** 以其强劲的性能脱颖而出。用户报告称，与其他模型相比，其效果更好，尽管由于其复杂性，用户仍面临配置障碍。

- [**INT8 混合精度训练带来显著加速**](https://github.com/pytorch/ao/tree/v0.5.0/torchao/prototype/quantized_training)：**INT8 混合精度训练（mixed-precision training）**的引入有望在 NVIDIA 4090 GPU 上实现高达 **70% 的加速**。这一进步允许在不牺牲准确性的情况下实现更快的训练，特别是在消费级硬件上。

**主题 4：AI 的创意应用受到关注**

- [**GameGen-O 开启游戏开发新前沿**](https://gamegen-o.github.io/)：**腾讯的 GameGen-O** 推出了一种扩散 Transformer（diffusion transformer）模型，可以生成开放世界视频游戏。这一创新令渴望利用 AI 加速游戏创作的开发者们感到兴奋。

- [**艺术家利用 AI 进行角色设计和动画制作**](https://huggingface.co/spaces/blanchon/room_cleaner)：创意人士正在使用 **Stable Diffusion**、**ControlNet** 和 **LoRA 训练**来制作令人惊叹的角色设计和动画。这些工具正在彻底改变艺术工作流，并扩展了数字艺术的可能性。

- [**Diffusion Illusions 以令人惊叹的艺术作品吸引眼球**](https://diffusionillusions.com/)：**Diffusion Illusions** 项目展示了通过扩散模型生成的交互式视错觉作品。该项目已被 **SIGGRAPH 2024** 接收，它推向了 AI 生成艺术和视觉感知的边界。

**主题 5：围绕 AI 技术的安全与伦理担忧**

- [**StealC 恶意软件利用 Chrome 钓鱼获取密码**](https://www.forbes.com/sites/daveywinder/2024/09/15/hackers-force-chrome-users-to-hand-over-google-passwords-heres-how/)：新的 **StealC 恶意软件**将 Chrome 用户困在全屏模式下，强迫他们通过虚假登录页面泄露 Google 密码。这种复杂的攻击引发了对浏览器安全漏洞的警惕。

- **关于 AI 模型审查的辩论升温**：用户对 **Phi 3.5** 等模型中严重的**审查（censorship）**感到不满，这阻碍了技术任务和编程辅助。社区呼吁在必要的审核与 AI 模型的实际效用之间取得平衡。

- [**“Humanity's Last Exam” 倡议引发争议**](https://x.com/DanHendrycks/status/1835725770402185399)：**Dan Hendrycks** 宣布为在 *Humanity's Last Exam* 中用难题挑战 AI 提供 **500,000 美元奖金池**。虽然一些人对推动 AI 进步的努力表示赞赏，但另一些人则对其在 AI 监管和政策影响方面的含义表示担忧。


---

# 第一部分：Discord 高层摘要




## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **O1 在 Bug 修复方面优于 Claude**：O1 在 Bug 修复方面表现出色，在速度和准确性上超过了 Sonnet 等 Claude 模型，尤其是在编程场景中。
   - 用户强调了 O1 提供详细输出的能力，有助于解决复杂的代码故障。
- **Sonnet 3.5 面临兼容性问题**：Sonnet 3.5 在处理较大上下文时表现吃力，且会误解指令，这让处理复杂编程任务的用户感到沮丧。
   - 相比之下，O1 的输出被描述为直接且有效，最大限度地减少了困惑。
- **Aider 脚本实现工作流自动化**：Aider 用户可以使用命令行 `--message` 参数简化任务，直接发送命令以实现流程自动化。
   - 这种方法允许通过简单的 Shell 脚本在多个文件上更轻松地进行批处理。
- **Game Gen - O 彻底改变游戏开发**：**Game Gen - O** 的推出为基于 Diffusion-Transformer 模型的开放世界视频游戏创作提供了新功能。
   - 该工具在社区中引起了轰动，因为它有望加速 AI 驱动的游戏开发。
- **The Big Prompt Library 发布**：**Big Prompt Library** 仓库提供了一系列 Prompt 和 LLM 指令，帮助用户进行有效的 Prompt 构建。
   - 该资源对于使用 **ChatGPT** 和 **Claude** 等系统的开发者至关重要，提升了用户体验。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma2 和 Mistral 模型表现不佳**：用户指出 **Gemma2** 和 **Mistral** 在训练中表现不佳，特别是与 **LLama 3.1** 相比，同时对 VRAM 限制感到沮丧。
   - 针对成功训练所需的必要配置提出了担忧，这使工作流变得复杂。
- **LLama 3.1 在性能上表现出色**：随着用户发现 **LLama 3.1** 的表现优于其他尝试过的模型，热情高涨，而 **Gemma 2 9B** 在适当设置下也显示出潜力。
   - 成员们注意到由于 **Gemma 2** 体积较大，需要调整设置，并引发了关于优化的讨论。
- **求职成为新热潮**：随着求职活动全面展开，成员们注意到在机器学习市场回暖之际，人们开始投资 **LinkedIn Premium** 等服务以寻求机会。
   - 一位博士持有者正在经历从学术界向企业的转型，原因是机器学习领域的博士后职位正在缩减。
- **关于招聘流程的辩论**：对话围绕倡导**公平**的招聘流程展开，挑战那些看重记忆力而非技能评估的传统方法。
   - 招聘中强调技能和增长潜力而非单纯的人脉，旨在建立一种改进后的模型。
- **对 DPO 的质疑引发了替代方案的建议**：一位成员对 **Direct Preference Optimization (DPO)** 表示怀疑，暗示在工作中探索 **KTO** 等替代方案。
   - 与会者之间出现了关于 DPO Loss 类型的持续讨论以及分享经验的愿望。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **用户讨论 OpenRouter 模型的上下文限制**：针对 OpenRouter 上各种模型显示的上下文长度产生了疑虑，特别是扩展版本的支持大小与声明内容之间存在差异。
   - 这引发了对提高模型能力透明度和更新沟通方式的呼吁，以便用户更清晰地理解。
- **性能问题引发模型审查**：用户报告了 Venus Chub AI 和 WizardLM-2 等模型出现异常输出和响应中断的情况，引发了对不同提供商之间一致性的警惕。
   - 正在进行的讨论旨在收集用户体验，以确定这些问题是普遍存在的还是孤立事件。
- **有效的 Prompt Engineering 技术成为关注焦点**：关于使用 XML 标签以改进模型响应的讨论非常突出，同时也分享了优化 Prompt Engineering 的教育资源。
   - 分享的教程侧重于 Prompt 操作方法，为提高 AI 交互中的用户参与度提供了见解。
- **集成与 API 配置混淆警报**：有报告称 **hyperbolic key** 被链接到了非预期的计费提供商，引发了关于命名规范和集成清晰度的讨论。
   - 用户表示需要在 JSON 配置中加入更强大的错误处理机制，特别是要求强制检查集成密钥的存在，以提高设置的可靠性。
- **提供商配置期间需要失败反馈**：讨论强调了用户在配置提供商时无法查看失败详情的挫败感，这增加了排查问题的难度。
   - 用户寻求 OpenRouter 提供更清晰的机制，以有效识别和解决集成问题，从而提高整体设置的成功率。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 的性能挑战**：用户报告 **Perplexity AI** 经历了严重的延迟和服务器宕机，引发了对平台高流量导致响应延迟的担忧。
   - 这一持续存在的问题引发了对其在高峰使用时段服务可靠性的质疑。
- **API 错误频发**：成员注意到 **API** 调用返回了 **500** 和 **524** 等错误，导致人们怀疑存在影响运营的广泛问题。
   - 随着用户讨论引用输出的不一致和超时问题，担忧进一步升级，呼吁改进对 API 交互的处理。
- **AI 模型的对比分析**：用户对比了各种 AI 模型，观察到在显著场景下，原始的 OpenAI 模型表现优于 You.com 和 Monica 等替代方案。
   - 即将推出的 **Opus 3.5** 模型被视为潜在的游戏规则改变者，预计将超越现有的性能基准。
- **Korean Emotion Video Dataset 的出现**：对 **Korean Emotion Video Dataset** 的兴趣达到顶峰，该数据集旨在增强 AI 的情感识别能力，为实际应用开辟了道路。
   - 讨论强调了其对研究和 AI 系统情感智能影响的兴奋感。
- **Microstrategy 对加密货币的大胆押注**：对话集中在 [Microstrategy](https://www.perplexity.ai/page/microstrategy-s-billion-dollar-ACYDp4QnTmuiq9x1Bu6svA) 的十亿美元投资上，分析了其对加密货币市场的潜在影响。
   - 成员们辩论了该公司的战略策略，评估了与市场稳定性相关的风险。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LLM 微调中的挑战**：用户在利用 **FSDP** 和 **BF16 AMP** 微调 **Llama 8b** 等模型时面临 **29G** 的高 GPU 显存占用，促使一些人回归原生 PyTorch 调用进行调试。
   - 这一问题引起了对 LLM 训练中资源管理的关注，并突显了对优化显存消耗的持续追求。
- **改进的 Inference API 文档**：根据用户反馈，**Hugging Face Inference API** 文档已得到改进，具有更清晰的速率限制和更好的代码示例。此次更新旨在简化 AI 部署，使其更加用户友好。
   - 正如[此公告](https://x.com/Wauplin/status/1835715850583564713)所示，此举展示了 Hugging Face 致力于提升用户体验的承诺。
- **新型医疗 LLM 及其影响**：**Chai-1 Foundation model** 在预测分子结构方面表现出色，为 **medical AI** 的进步做出了贡献，正如[最近的更新](https://x.com/OpenlifesciAI/status/1835085857826455825)所述。
   - **BrainWave** 和 **DS-ViT** 等创新模型正在推进诊断评估技术，推动模型训练数据集实现更高的透明度。
- **多语言模型的高效 Tokenizer 训练**：关于重新训练 Tokenizer 的讨论强调了在保持原始数据性能的同时整合多种语言的灵活性，尽管也出现了对歧义性增加的担忧。
   - 持续预训练（continued pretraining）被提出作为减轻这些挑战的一种方法，表明了社区对 NLP 多语言能力的参与。
- **Nitro 赠送活动引发关注**：一名成员宣布了 **Nitro giveaway**，邀请参与者与服务器互动，在社区中引发了轻松的关注。
   - 尽管带有幽默色彩，这一公告展示了社区在促进互动和连接方面的努力。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **StealC 恶意软件针对 Chrome 用户**：一种名为 **StealC** 的新发现恶意软件通过锁定浏览器并强迫用户通过欺骗性登录界面泄露 Google 密码来限制 **Chrome** 用户，引发了重大的安全担忧。
   - 该恶意软件利用全屏自助服务终端模式（kiosk mode）诱导用户提交敏感信息，这与传统的网络钓鱼方法有逻辑上的相似之处。
- **腾讯 GameGen-O 变革视频游戏**：腾讯推出了 **GameGen-O**，这是一种用于生成开放世界视频游戏的扩散 Transformer 模型，利用了来自 100 多个下一代游戏的广泛数据。
   - 该模型在 **OGameData** 上进行训练，能够实现更具互动性的游戏玩法，并通过先进的模拟技术提高了视频游戏开发的标准。
- **基于拖拽的图像编辑创新方法**：**InstantDrag** 流水线通过消除对掩码或文本提示的需求，增强了基于拖拽的图像编辑，利用双网络系统实现实时、照片级的编辑。
   - 通过利用来自真实世界视频数据集的运动动力学，该方法显著加快了编辑过程，展示了创意应用的潜力。
- **探索 AI 训练中的精度退火 (Precision Annealing)**：一名成员提出了关于 **precision annealing** 的查询，建议在 **FP8** 进行预训练，并切换到 **BF16** 或 **FP32**，以在最终训练阶段最大化吞吐量。
   - 他们强调这种方法可以优化训练方案中的资源利用，因为它减轻了显存限制。
- **评估指标与性能见解**：在评估中，**QLoRA** 显示出优于传统 **LoRA** 方法的性能，表明在微调效率方面具有优势。
   - 成员们对 **QLoRA**、全量微调（full fine-tuning）和原始模型的性能指标进行了对比分析，并对观察到的百分比差异进行了讨论。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **O1 撰写长篇论文**：一位成员展示了 O1 生成涵盖从 indev 到 **1.21** 的主要 **Minecraft updates** 详细文章的能力，令社区感到兴奋。
   - 这突显了 O1 先进的写作熟练度及其在创意应用方面的潜力。
- **模型 Fine-tuning 面临挑战**：用户对 **fine-tuning results** 表示担忧，报告称缺乏改进且训练损失（training loss）波动，从而引发了关于模型选择（model selection）的建议。
   - 对话强调了 Fine-tuning 并不总是能产生有效结果，促使人们寻求战略性调整。
- **Custom GPT 功能引发疑问**：关于 **Custom GPTs' functionality** 的咨询揭示了其随所用模型而异的变动性，并要求明确模型选择。
   - 分享的见解包括参考资料的潜在链接，强调了在启动对话时需要更清晰的指导。
- **ChatGPT 响应一致性问题**：用户应对了 ChatGPT 在遵循预定序列方面的挑战，特别是在 **RPGs** 的战斗中。
   - 建议包括使用 Discord bot 格式收集响应，然后将其输入 ChatGPT 进行分析，旨在简化交互。
- **使用 ChatGPT 探索游戏机制**：剖析了一个涉及 **60% 失败几率** 游戏的场景，指出 ChatGPT 倾向于产生误导性解释。
   - 讨论揭示了财富积累策略的复杂性以及模型在处理游戏语境时的性能差异。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA-MODE Hackathon 引起远程参与兴趣**：关于即将举行的 CUDA-MODE hackathon 远程参与的提议引发了关于其可行性和组织工作的讨论。
   - 虽然一些成员支持远程赛道，但其他人指出了大型线下活动的挑战。
- **Triton Kernel 启动开销问题**：人们对 Triton 中的 **kernel launch overhead** 表示担忧，有报告称对于中等规模的矩阵，它消耗了 **10-20%** 的执行时间。
   - 一个 [GitHub issue](https://github.com/triton-lang/triton/issues/2637#issuecomment-2236098076) 详细说明了 kernel 执行需要 **80us**，但启动它却需要 **220us**。
- **INT8 混合精度训练带来的显著提升**：最新的 **torchao 0.5 release** 展示了 INT8 混合精度训练（mixed-precision training）在 NVIDIA 4090 GPU 上实现了高达 **70% 的加速**，且没有明显的精度损失。
   - 这一进展突显了训练效率的增强，特别有利于在保持收敛性的同时惠及消费级 GPU。
- **Liger-Kernel v0.3.0 正式上线！**：[Liger-Kernel v0.3.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.3.0) 发布，带来了重大进展，社区对其支持表示赞赏。
   - 团队邀请社区体验新功能并提供反馈。
- **BitNet 训练面临效率挑战**：最近的讨论表明 **BitNet model training** 仍在挣扎中，近期的试验未报告显著进展。
   - 成员们对与位运算（bitwise operations）相关的 GPU 效率低下表示担忧，强调了对定制硬件方案的需求。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GPU 加速问题依然存在**：用户报告了 LM Studio 中未利用 GPU 加速的问题，提示在 Developer > LM Runtimes 下进行检查。一次成功的更新使一位用户的 GPU 利用率显著上升。
   - *排查实践揭示了潜在的配置误解*，从而实现了更高效的设置。
- **模型兼容性困境**：LM Studio 主要支持 GGUF 模型，但并非所有列出的模型都能按预期运行，特别是在多模态任务中。这一限制引发了对模型性能和功能可用性的担忧。
   - 参与者分享了关于仍无法使用的功能的见解，表明在利用 LM Studio 时，预期与现实之间存在差距。
- **Strix Halo APU 能力炒作**：关于 Strix Halo APU 运行大型 AI 模型潜力的讨论十分激烈，有说法称可为其 iGPU 分配高达 **20GB**。虽然提到了对 ROCm 的支持，但也出现了关于任务卸载影响性能的担忧。
   - *关于处理效率的竞争性观点浮出水面*，强调了平衡 CPU 和 GPU 任务的重要性。
- **RTX 4090 加速 AI 查询**：凭借三块 RTX 4090 显卡，一位成员报告在查询期间达到了 **110 tokens per second**。这引发了关于电源设置以有效发挥此类性能的讨论。
   - 讨论集中在优化配置以提高电源效率和 GPU 性能。
- **为 LLM 优化 RAM**：运行大型模型需要足够的系统 RAM，有案例表明 **192GB** DDR5 可以支持 Llama 3.1 等模型。然而，也有观点认为如果模型经过良好优化，**64GB** 可能就足够了。
   - 参与者交流了优化策略，*在 RAM 容量和模型需求之间寻找平衡。*

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI o1 模型发布**：OpenAI 发布了 o1 模型，旨在提高复杂任务的推理能力，因其在科学和编程应用中的潜力而受到关注。
   - 据报道，新模型性能优于旧版本，但在处理大型编辑时仍感吃力，这是 Cursor AI 正在通过其专门的编程助手解决的挑战。
- **AI 初创公司融资激增**：11x AI 在 A 轮融资中筹集了 [2400 万美元](https://x.com/11x_official/status/1835711787712582082?s=46)，其 ARR 增长了 15 倍并推出了新的数字员工，彰显了其快速增长。
   - 同样，Supermaven AI 获得了 [1200 万美元](https://x.com/supermavenai/status/1835743882971426837?s=46) 资金，用于开发一款能与其模型无缝集成的 AI 文本编辑器。
- **HTEC 关于 AI Copilots 的报告**：近岸咨询公司 **HTEC** 发布了一份关于他们使用 26 种 AI 编程工具经验的 [报告](https://htec.com/htec-report-ai-code-generators/)，不过访问需要注册。
   - 成员们讨论了报告中提到的简短使用和局限性是否真实反映了这些工具的能力。
- **Voice Mode API 讨论**：本期内容深入探讨了新的 **Voice Mode API**，它允许更具交互性和动态的对话能力。
   - 它强调了这一功能如何改变用户在各种平台上与 AI 的交互。
- **ChatGPT 扩展策略**：讨论了扩展 **ChatGPT** 的策略，特别关注于**增加延迟**以及用于优化的 **Prompt/Schema caching** 技术。
   - 团队解决了关于 **模型可复现性** 以及 API 不断演进的 **分层和速率限制** 策略的担忧。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 的 o1 模型引起关注**：OpenAI 最近发布的 **o1-preview** 和 **o1-mini** 模型引发了关于其有趣的推理模式以及用户交互数据对模型开发可能产生影响的讨论。
   - 一位用户强调了一个令人惊讶的发现：**mini 的推理时间并不比 preview 长**，但生成的响应却更长，这挑战了人们的预期。
- **Humanity's Last Exam 发布公告**：Dan Hendrycks 推出了 *Humanity's Last Exam*，征集高难度的 AI 问题，**奖金池高达 500,000 美元**，截止日期为 2024 年 11 月 1 日。此举引发了关于其对 AI 监管影响的褒贬不一的反应。
   - 人们对 Hendrycks 的游说工作及其与政治的联系表示担忧，这可能会根据性能指标影响未来的 AI 政策。
- **RL 爱好者讨论 Reverse Curriculum Learning**：关于 **LLMs** 中 **Reverse Curriculum Learning** 的新兴论文引发了关于其在 RL 社区使用受限的讨论，用户指出它尚未获得广泛认可。
   - 成员们认为 **Reverse Curriculum Learning** 比较笨重，主要适用于**利基应用 (niche applications)**，导致其在更广泛的背景下较为罕见。
- **对 LLM 模型发展的期待**：人们对计划于 2025 年实现的未来 LLM 进展充满期待，讨论反映出对模型能力潜在突破的热情日益高涨。
   - 成员们察觉到情绪的显著转变，指出格局已经发生变化，标志着可能出现类似于过去进步的里程碑。
- **Poe 订阅服务评估**：用户辩论了他们对 **Poe** 订阅服务的体验，尽管支付 20 美元即可访问所有可用的 LLMs，但对其易用性感受复杂。
   - 用户对界面设计提出了担忧，表示与 **Claude** 和 **ChatGPT** 等竞争对手相比，更倾向于更具吸引力的美学设计。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **探索 Fei-Fei Li 的推理方法**：成员们对 **Fei-Fei Li** 解决推理问题的技术表示好奇，旨在收集她在 AI 背景下相关方法的见解。
   - 工程师们明显渴望更深入地了解像她这样的方法论，这可能会为正在进行的 AI 进展提供参考。
- **Command-R-Plus-08-2024 输出问题**：一位用户报告称，与之前的版本相比，**Command-R-Plus-08-2024** 模型产生的输出重复性更高，特别是在创意任务中。
   - 这引发了关于长 Prompt 如何进一步影响性能的讨论，并促使人们探索替代模型。
- **Cohere 开发者办公时间公告**：Cohere 将于今天**东部时间下午 1 点**举办开发者办公时间，讨论 **Command 模型系列**的更新，包括 **RAG** 和 **Safety Modes** 的新功能。
   - 与会者可以期待了解模型效率和实际应用方面的重大改进。
- **实施 Safety Modes 以增强控制**：Cohere 推出了 **Safety Modes**，旨在让企业客户更好地控制模型的使用和交互。
   - 此次更新加强了治理，同时确保模型有效性保持不变。
- **招聘信息担忧与社区焦点**：一位成员呼吁从讨论中移除与 Cohere 无关的招聘信息，强调社区话题相关性的必要性。
   - 这反映了保持讨论与 Cohere 社区利益和目标紧密一致的承诺。



---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **用户在 FLUX 模型上遇到困难**：成员们报告了在运行 **FLUX 模型**时遇到的问题，特别是关于 `.sft` 和 `.safetensor` 等格式，以及与 Forge 等工具的兼容性问题。
   - 建议切换到 [ComfyUI](https://comfyui.com) 以获得更好的支持，用户们分享了关于特定模型大小的使用经验。
- **创作具有风格的角色**：一位用户寻求关于使用 Stable Diffusion checkpoints 和提示词技巧生成类似于 **Cheetara** 角色的建议。
   - 讨论内容包括适用于后期 3D 建模的角色艺术专用 checkpoints，并引用了 [Cheetara GIF](https://tenor.com/view/thunder-cats-gif-7172707) 作为灵感。
- **精通图像编辑**：出现了关于从图像中移除文本和利用 inpainting 方法的技巧建议，其中 GIMP 等工具被重点提及。
   - 用户讨论了各种在保持质量的同时增强图像的 AI 工具，包括 Piximperfect 的教程。
- **使用 ControlNet 制作角色动画**：关于利用 **ControlNet** 和 **LoRA 训练**创建矢量风格角色动画的见解不断涌现，强调了使用正确训练样本的重要性。
   - 贡献者分享了使用 ControlNet 技术改进艺术渲染中角色姿势和结构的技巧。
- **技术支持困扰**：一位用户在安装 Stable Diffusion 期间遇到错误，被建议在支持频道分享其错误日志以便进行故障排除。
   - 分享了指向 [安装指南](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides) 的有用链接，强调了详细日志的重要性。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **用户验证流程上线**：Discord 服务器已实施 **用户验证** 流程，要求成员通过 #verify 频道的机器人提交其电子邮件地址，未验证用户仍保留只读权限。
   - 选择不进行验证的成员将面临 **发送消息功能受限**，强调了这一新步骤的重要性。
- **引入入门引导问题以优化体验**：电子邮件验证后，用户将回答 **两个多选题入门引导问题**，旨在提升其服务器体验。
   - 该举措反映了为新老成员改进引导流程的努力。
- **Mojo 在 Python 互操作性方面面临挑战**：讨论显示 Mojo 目前无法导入 Python 模块或调用其函数，阻碍了有效的互操作性，而这对于无缝集成至关重要。
   - 参与者热衷于在 Mojo 和 Python 之间实现 **zero-copy 数据交换** 的方法，特别是在性能敏感的场景下。
- **Count Leading Zeros 面临编译时限制**：用户报告称 `clz` 函数在编译时难以运行，原因是其依赖于 LLVM intrinsics，而这些在现阶段无法执行。
   - 提出了一种计算前导零的替代实现，突显了标准库中对更好编译时能力的需求。
- **用于讨论服务器变更的新频道**：已建立一个专门讨论 **即将到来的服务器变更** 的频道，允许成员分享建议并提出问题。
   - 此举标志着通过社区投入和对话来增强 **用户体验** 的承诺。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **理解混合精度训练的挑战**：虽然 **mixed precision training** 可以通过同时以 fp32 和 fp16 存储模型来提升性能，但它也会在正向传播（forward pass）期间使计算负载翻倍——这是一个值得注意的权衡。
   - 成员们强调了在预算限制下平衡速度和资源利用率的重要性。
- **CoreWeave 的重大估值**：**CoreWeave** 正在洽谈出售股份，公司估值达到 **230 亿美元**，反映了其在 AI 驱动的云计算领域的卓越地位。
   - 此举引起了知名财经媒体的极大关注，突显了该行业的竞争格局。
- **探讨 AI 的社会影响**：讨论反映了 **OpenAI** 如何有效地实现了更广泛的信息获取，将其比作在“每个人的口袋里放了一个博士”，而公众对这些变化的反应极小。
   - 成员们强调需要就 **transformative effects**（变革性影响）以及 AI 持续融入日常生活进行更深层次的对话。
- **RWKV 团队挑战 RNN 极限**：RWKV 团队在 RNN 架构进步方面引起了轰动，其中 *Smerky* 等人的贡献尤其受到认可。
   - 这一创新举措因其在社区内的潜在影响而受到了关注和赞誉。
- **对小数据集模型过拟合的担忧**：一位成员表示仅使用 **9 张图像** 很难让模型过拟合，这引发了关于在处理更大数据集时可能出现的学习问题的讨论。
   - 共识是，如果无法对如此小的样本进行过拟合，可能预示着未来会面临更大的困难。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaParse 在解析 Excel 数据方面表现出色**：在 [最近的视频](https://twitter.com/llama_index/status/1834680455171653959) 中，展示了 LlamaParse 先进的 **Excel 解析能力**，包括处理多个工作表和复杂表格。LlamaParse 利用 **recursive retrieval** 自动总结复杂表格，提高了效率。
   - 此功能显著提升了可用性，特别是对于处理复杂 Excel 文件的用户。
- **TypeScript 工作流引入 LlamaIndex**：如该 [公告](https://twitter.com/llama_index/status/1834689049954804098) 所述，LlamaIndex 现已将 workflows 集成到 TypeScript 中。这一新功能旨在简化 TypeScript 用户的开发流程。
   - 这种集成有助于使该框架对于使用 TypeScript 的开发者来说更加易用和高效。
- **单元测试在 LLM 应用中的重要性**：单元测试被强调为减轻 LLM 应用中随机性的关键，一篇详细介绍使用 [CircleCI](https://twitter.com/llama_index/status/1834987463569555909) 构建和测试 RAG 应用的博客文章中强调了这一点。适当的单元测试对于防止 AI 应用中出现意外行为至关重要。
   - 讨论强调了对 AI 驱动项目质量和可靠性的承诺。
- **Vectara-Agentic 库简化 RAG 实现**：查看由成员开发的 [vectara-agentic](https://twitter.com/llama_index/status/1835348333478760896)，这是一个简化构建由 LlamaIndex 和 Vectara 驱动的 agentic RAG 的库。它提供了构建能够进行规划和工具使用的 Agent 的工具，并兼容各种模型提供商。
   - 这种灵活性使开发者能够更高效地实现 RAG 解决方案。
- **本地 LLM 提供成本优化**：成员们讨论了运行 **Local LLM** 与使用 **OpenAI** 服务相比可以显著降低成本。在 **OpenAI** 和本地模型之间做出选择时，总拥有成本 (**TCOS**) 被认为是一个重要因素。
   - 这一考量强调了优化 AI 解决方案以获得更好成本效率的日益增长的趋势。



---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **理解 GPU 边际收益递减**：GPU 的**边际收益递减点**在游戏领域出现在 **2-3 个 GPU** 之后，在渲染领域出现在 **4-6 个 GPU** 之后，这很大程度上归因于 **PCIe 带宽限制**。
   - 文档问题被认为是影响用户 GPU 设置体验的一个主要担忧。
- **Open Interpreter 中的非流式响应**：成员们讨论了如何在命令行模式下**停止流式响应**；选项包括使用 **--plain 标志**或 `claude-3.5` 模型。
   - 该反馈旨在提高与命令行交互时的易用性和舒适度。
- **对 ChatGPT O1 模型发布的困惑**：关于 ChatGPT 的 **O1 模型**存在担忧，有人推测其发布可能会削弱现有的替代方案，尽管这一观点受到了另一位成员的挑战。
   - 虽然 O1 在推理方面表现出色，但批评者指出它在执行代码任务时不如早期的 **model 4** 等模型有效。
- **Livekit 设置错误警报**：约 **90% 的用户**报告了 **Livekit** 设置问题，将其归咎于文档不足。
   - 有人提议创建一个全面的设置指南以增强用户支持。
- **令人兴奋的用于编排的 MoA LLM 库**：[MoA LLM 库](https://github.com/catena-labs/moa-llm) 引入了一种在受神经网络启发的架构中编排 LLM 的方法，旨在改进模型协作。
   - 这一开源项目为高效整合多个 LLM 提供了一个框架。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **关于 O1 模型有效性的辩论**：对 **O1 模型**的评价褒贬不一；虽然有些人称赞其 **Chain of Thought** 界面，但其他人认为其**机械化的响应**令人失望。
   - 一位成员提到，尽管 UI 表现稳健，但整体性能仍有很大提升空间。
- **OpenAI 的 O1 开发时间线澄清**：一位成员透露，**OpenAI 开发 O1 (Strawberry/Q*)** 已有很长时间，这与它是快速产出结果的说法相反。
   - 他们指出 O1 采用了**代理式思维链 (agentic chain of thought)**，表现出对幻觉等常见问题的抵御能力。
- **掩码问题导致的分词错误**：一位成员报告了由于新的逐轮掩码策略导致出现 **Tokenization 错误**，该策略遮蔽了最后一轮结束的 Token。
   - 他们将此问题关联到了在 GitHub 上提交的一份详尽的 [bug 报告](https://github.com/axolotl-ai-cloud/axolotl/issues/1916)。
- **Phi 3.5 在分类任务中的挫败**：成员们表达了在开发 **Phi 3.5** 句子分类器时的挣扎，该分类器无法产生正确的分类输出。
   - 一位成员选择分享了他们的 [简易句子分类器](https://huggingface.co/fozziethebeat/phi-3.5-alpaca-test-classifier)，并承认目前可能会暂时放弃。
- **vLLM 与 Adapter 兼容性问题**：围绕 **vLLM** 无法正确解释 `qkv_proj` 层展开了讨论，这影响了使用 **Axolotl** 的 Adapter 训练的模型。
   - 有趣的是，虽然一个 LORA 模型在合并过程中显示没有学习到内容，但作为基础模型之上的独立 Adapter 运行时表现良好。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **提供 GenAI/RAG/CV 咨询**：一位成员宣布提供 *GenAI*、*RAG* 和 *CV* 方面的**咨询服务**，以协助初创公司进行原型开发。
   - 感兴趣的成员可以直接联系以寻求合作机会。
- **OpenAI 引发社会反思**：人们对 **OpenAI** 在社会似乎未发生变化的情况下对知识获取的影响表示担忧。
   - 讨论包括关于加速自动化如何引导我们进入**后稀缺时代**的思考。
- **LangGraph Cloud 定价不确定性**：一位成员询问了 **LangGraph Cloud** 在 Beta 阶段后的潜在成本，考虑是否开发自定义的 FastAPI 封装。
   - 对可行长期定价模型的担忧是讨论的一个重点。
- **流式 LLM 输出解析问题**：讨论了在**流式 LLM 输出**期间解析不完整 JSON 字符串的问题，特别是使用 Pydantic 解析器时。
   - 尽管最初持怀疑态度，但从 `parse_result` 切换到 `parse` 方法产生了更好的结果。
- **聊天历史管理挑战**：用户表达了在使用 **LangChain** 管理聊天历史方面的困难，特别是在跟踪特定应用的消息时。
   - 他们强调了在整合这些数据时维持事务完整性的问题。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **优化 RAG 查询结构**：一位成员建议在单个模块中优化 RAG，通过将来自内存和提示词的数据打包到 'context' 字段来增强结果。该方法参考了[这个简单的 RAG 示例](https://github.com/stanfordnlp/dspy/blob/main/skycamp2023.ipynb)并得到了确认。
   - 另一位成员认可了这一策略的实用性，并指出了其在数据处理方面的优势。
- **DSPy 中的 Visual LLM 使用案例**：有人询问在 DSPy 中使用 Visual LLM 模型进行图像描述的可能性，另一位成员推测下周可能会推出。文中引用了一个很有前景的 [GPT-4 Vision API PR](https://github.com/stanfordnlp/dspy/pull/682)，暗示集成工作正在进行中。
   - 这一预期功能引发了社区对即将到来的新能力的狂热期待。
- **寻求 GitHub 贡献**：一位成员表达了为 DSPy 项目做贡献的兴趣并询问了是否有可用的赏金（bounties），随后引发了讨论。信息显示，更多的集成变更即将到来，预计完成时间为 **7-10 天**。
   - 贡献的前景在社区内引起了兴奋，表明了大家对协作开发的共同渴望。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 添加运行时类型检查**：George Hotz 宣布 Tinygrad 支持 `TYPED=1` 进行**运行时类型检查**，在运行 `python3 test/test_ops.py` 测试时发现了类型错误。一个 [GitHub PR](https://github.com/tinygrad/tinygrad/pull/6520) 提议修复大部分类型错误，目前还剩一个未解决。
   - 社区反馈强调了健壮的类型检查的重要性，进一步强化了编写规范代码的必要性。
- **Tinygrad 0.9.2 在 AMD 上测试失败**：一位用户报告在将 Tinygrad 从 **0.9.0 升级到 0.9.2** 时遇到问题，遇到了与 `struct_kfd_ioctl_criu_args` 相关的 **AttributeError**。怀疑根本原因是**内核版本**与库的要求之间可能存在不匹配。
   - 诊断结果表明，Tinygrad 在针对 AMD 用户的兼容性文档和故障排除指南方面可能存在缺失。
- **Tinygrad 库讨论引发关注**：成员们讨论了 **tinygrad 生态系统**内库的开发，特别提到了 **timm** 和 **torchvision** 作为候选。这次对话引发了关于此类库的实际必要性和当前实现的询问。
   - 当一位用户质疑这些库在 tinygrad 中的实际效用时，讨论升级，表明在集成方面需要更清晰的说明。
- **调查 VRAM 分配峰值**：一位成员寻求关于诊断 Tinygrad 运行期间 **VRAM 分配峰值**的建议，强调了框架内内存监控工具的知识空白。这一询问凸显了对更强大的诊断工具的需求。
   - 了解 VRAM 行为对于优化性能和防止密集处理任务期间的崩溃至关重要。
- **报告 Tensor 修改错误**：用户在修改 Tinygrad 中的 **Tensor** 时遇到错误，特别是在元素自增期间。他们引用了一个与该问题一致的[未解决 GitHub issue](https://github.com/tinygrad/tinygrad/issues/6352)，重点在于 **contiguous** 属性。
   - 该用户的发现强化了关于 Tensor 操作的全面测试和文档的重要性。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **掌握 Checkpoints 管理**：要在特定的 token 计数处实现 checkpoints，请利用 `num_tokens` 字段并过滤 padding tokens，详见[此处](https://github.com/pytorch/torchtune/blob/4fbe7b2d4956b3790c51d7a255c0040cf5c38fad/recipes/full_finetune_distributed.py#L622)。调整保存逻辑对于准确跟踪和从保存状态恢复至关重要。
   - 成员们强调了在训练期间进行 all gather 以统计所有 rank 总数的必要性。
- **引入 Cosine 学习率衰减**：成员们讨论了集成 `torchtune.modules.get_cosine_schedule_with_warmup` 以实现学习率的余弦衰减，目前已应用于 LoRA recipes。建议在 mid-epoch 恢复时跳过从 epoch 编号推导步数的操作，以实现更平滑的过渡。
   - 建议成员们密切关注这些实现，以便将其纳入 full finetune recipe。
- **关于 CUDA 与 CPU 操作的辩论**：有人询问 token 操作是否可以在 CPU 上进行，得到的确认是 `num_tokens` 不是 CUDA tensors，但建议使用 CUDA。尽管对 CPU 效率仍有疑问，但对 CUDA 进程的偏好依然存在。
   - 讨论显示出不确定性，但明显倾向于使用 CUDA 以获得这些操作的最佳性能。
- **在线打包（Online Packing）支持即将到来**：团队计划在添加对 **iterable datasets** 的支持后立即实现在线打包。此举有望提高批量数据处理的效率。
   - 成员们对这将为未来项目带来的能力提升表示兴奋。
- **CI GPU 测试失败引发关注**：与 GPU 测试相关的持续 CI 问题（特别是 `test_eleuther_eval.py`）源于 **transformers.pipelines** 中的导入错误，虽然有 504 个测试通过，但重大错误阻止了完成。这引发了对系统整体稳定性的警报。
   - 成员们正在积极讨论潜在的修复方案并调查异常情况，以确保 CI 运行更加顺畅。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **生成式 AI 瞬间创作艺术**：一位成员展示了使用 NotebookLM 创作的艺术作品，仅用 **2 分钟** 即可完全生成。他们分享了一个激动人心的 [YouTube 视频](https://youtu.be/kINTcf9rEJ4)，记录了这一快速创作过程。
   - 他们对生成式 AI 的能力发出了“活在当下真好”的感慨。
- **Steve Mould 的错觉探索**：一位成员在 YouTube 上分享了《这种新型错觉真的很难制作》，深入探讨了 **AI 生成的错觉**。视频包含了关于 Jane Street 实习的见解，可以在[此处](https://youtu.be/FMRi6pNAoag)观看。
   - 他们指出生成式 AI 创造的图像在不同光照条件下会发生变化。
- **扩散错觉（Diffusion Illusions）成为焦点**：一位成员介绍了 [Diffusion Illusions 网站](https://diffusionillusions.com/)，该网站展示了通过扩散模型生成的交互式视错觉。该网站链接到他们被 **SIGGRAPH 2024** 接收的项目，包括一段 YouTube 演讲。
   - 主要贡献者包括 Ryan Burgert 和 Xiang Li，展示了扩散模型的引人注目的应用。
- **寻求图像中的文本**：一位成员寻求关于如何高效地在图像中嵌入文本以创建全面数据集的建议，目标是扩展到 **数百万张图像**。
   - 这一讨论突显了为 AI 应用自动化创建文本嵌入图像数据集的需求。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **预训练 VLMs 需要大量算力**：一位成员对使用预训练 **视觉语言模型 (VLMs)** 缺乏 **算力资源** 表示担忧，这些模型本质上需要大量的计算能力。
   - 讨论强调，这些模型的有效性在很大程度上取决于是否有合适的硬件来处理其密集的计算需求。
- **异常检测需要明确**：一位成员询问 **异常检测** 应该关注日志还是实际的 **时间序列数据**，从而引发了对数据类型的深入探讨。
   - 共享了几种用于时间序列分析的方法论，包括 **Transformer 模型**、**卡尔曼滤波 (Kalman Filters)** 和 **孤立森林 (isolation forests)**，并建议使用 **z-scores** 进行误差评估。

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **模型在 Function Calling 方面表现挣扎**：讨论显示该模型目前仅具备聊天能力，在相关性（relevance）方面得分仅为 **1**，无法执行任何 Function Calling，在其他能力方面得分为 **0**。
   - *这一 Bug 显著限制了模型的功能*，阻碍了用户体验并降低了预期。
- **模型生成对话而非 Function Call**：成员们表示担心模型输出的是对话式响应而非执行 Function Calling，导致沟通误解和评分错误。
   - *这导致该尝试被自动标记为错误*，影响了处理响应的准确性。
- **无效语法触发 AST Decoder 失败**：错误消息标记为 'Invalid syntax'，导致 Abstract Syntax Tree (AST) 解码失败，分类为 'ast_decoder:decoder_failed'。
   - 该问题表明在解析模型输出时存在**关键问题**，为故障排除带来了挑战。

---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1284234855424458855)** (679 messages🔥🔥🔥): 

> - `O1 and Claude Models`
> - `AI and Job Impact`
> - `Using Aider with LLMs`
> - `AI Coding Tools`
> - `Prompt Engineering` 


- **O1 在修复 Bug 方面的有效性**：用户发现 O1 是修复 Bug 最有效的模型，在速度和准确性上优于 Sonnet 等 Claude 模型，尤其是在编程语境下。
   - O1 提供了全面的分步规划，生成的详细输出有助于理解和解决复杂的代码问题。
- **Sonnet 3.5 面临的挑战**：多位用户报告了使用 Sonnet 3.5 的困难，指出它在处理大 Context 时经常表现挣扎，并可能误解指令，尤其是在复杂的编程任务中。
   - 尽管其功能强大，但用户对 Sonnet 的局限性表示沮丧，尤其是与 O1 更直接的输出相比。
- **AI 对就业的影响**：关于 AI 可能取代初级开发者的担忧浮出水面，讨论强调高级开发者凭借其专业知识仍可能保持其重要性。
   - 参与者指出需要 AI 与人类协作，资深开发者可以利用 AI 工具提高生产力，而不会完全被取代。
- **使用 Aider 和其他 AI 工具**：Aider 因其与 AI 模型的有效集成而受到参与者青睐，但用户也认可 Claude Dev 和 OpenAI Playground 等替代工具在特定任务中的效用。
   - 讨论显示不同的 AI 模型在各个领域各有所长，用户经常尝试不同的组合以优化其工作流。
- **Prompt Engineering 与定制化**：用户分享了在 Aider 中定制 System Prompts 的策略，通过为 Claude 3.5 等特定模型量身定制提示词来增强 LLM 交互。
   - 参与者表示有兴趣利用社区开发的 Prompts 来改善他们在 AI 编程任务中的体验。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://github.]">未找到标题</a>：未找到描述</li><li><a href="https://tree-sitter.github.io/tree-sitter/">Tree-sitter｜简介</a>：未找到描述</li><li><a href="https://aider.chat/2023/10/22/repomap.html#optimizing-the-map)">使用 Tree-sitter 构建更好的仓库地图</a>：Tree-sitter 允许 aider 构建一个能更好总结大型代码库的仓库地图（repo map）。</li><li><a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-aider-uses">常见问题解答</a>：关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/git.html">Git 集成</a>：Aider 与 Git 紧密集成。</li><li><a href="https://tenor.com/view/drop-the-mic-bryan-cranston-mic-drop-gif-4853979505988741">Bryan Cranston 扔麦克风 GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/sama/status/1834795291406483684">来自 Sam Altman (@sama) 的推文</a>：我喜欢回到中西部的家。夜空如此美丽。期待冬季星座很快升起；它们太棒了。</li><li><a href="https://x.com/AnthropicAI/status/1831348825341981042">来自 Anthropic (@AnthropicAI) 的推文</a>：GitHub 是我们正在构建的首批原生集成之一，旨在将 Claude 连接到您最重要的数据源。该功能目前面向早期 Enterprise 计划用户提供测试版。我们计划...</li><li><a href="https://gist.github.com/plembo/6a035299f50db092ab710c74eaf6dcfb">pyperclip.copy 的 Linux 变通方案</a>：pyperclip.copy 的 Linux 变通方案。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://x.com/martinbowling/status/1835040299459961142">来自 Martin Bowling (@martinbowling) 的推文</a>：🐌 微小的 o1 速率限制让你举步维艰？⏳ 无尽地等待 o1 的思考？告别漫长的思考时间！🚀 使用由 @Groq 驱动的 Llamaberry 增强你的 AI 推理能力...</li><li><a href="https://x.com/minchoi/status/1834677525428982105">来自 Min Choi (@minchoi) 的推文</a>：Google 最近发布了 NotebookLM。这款 AI 工具可以根据研究论文、文章等各种来源的内容生成两名对话者的播客。简直太疯狂了。...</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1fhei0j/its_over/?share_id=5UtDCm_r90-C2pmvdWela&utm_content=1&utm_medium=ios_app&utm_name=ioscss&utm_source=share&utm_term=1">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://openrouter.ai/models/openai/gpt-4-32k">GPT-4 32k - API、提供商、统计数据</a>：GPT-4-32k 是 GPT-4 的扩展版本，具有相同的功能，但上下文长度增加了四倍，允许在单次处理中处理多达 40 页的文本。这对于...特别有益。</li><li><a href="https://x.com/sama/status/1834276403270857021">来自 Sam Altman (@sama) 的推文</a>：不再有耐心了，Jimmy</li><li><a href="https://x.com/apples_jimmy/status/1833595024543781088">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>：好了，现在回到 10 月。我们应该在 10 月迎来一个 4.x 模型（也许还叫 4.5，我的老朋友）。至于大家伙 GPT-5，我听说最早是 12 月，但为了大家的理智，我会定在明年第一/第二季度...</li><li><a href="https://github.com/fry69/files-to-prompt-ts/blob/main/files-to-prompt.ts#L146">fry69/files-to-prompt-ts 项目 main 分支下的 files-to-prompt.ts</a>：一个命令行工具，用于以结构化的方式将文件和目录连接成单个 Prompt，以便与 LLM 和其他应用程序配合使用。- fry69/files-to-prompt-ts</li><li><a href="https://github.com/Agentic-Insights/codebase-context-spec">GitHub - Agentic-Insights/codebase-context-spec：一个灵活的、与工具无关的代码库上下文系统提案，旨在帮助 AI 编程工具了解您的代码库。入门非常简单，只需在项目根目录创建一个 .context.md 文件。</a>：一个灵活的、与工具无关的代码库上下文系统提案，旨在帮助 AI 编程工具了解您的代码库。入门非常简单，只需在您的...</li><li><a href="https://github.com/paul-gauthier/aider/commit/d747a3781d5eddc7c28a28a79f27712422e0b505">feat: 添加 o1-mini 和 o1-preview 的 OpenRouter 版本 · paul-gauthier/aider@d747a37</a>：未找到描述</li><li><a href="https://github.com/yamadashy/repopack">GitHub - yamadashy/repopack：📦 Repopack 是一个强大的工具，可以将您的整个仓库打包成一个 AI 友好的文件。当您需要将代码库提供给 LLM 或其他 AI 工具（如 Claude、ChatGPT 和 Gemini）时非常完美。</a>：📦 Repopack 是一个强大的工具，可以将您的整个仓库打包成一个 AI 友好的文件。当您需要将代码库提供给 LLM 或其他 AI 工具（如...</li><li><a href="https://x.com/wgussml/status/1833615864131948756">来自 william 的推文</a>

(@wgussml)</a>: 🚀 我很高兴地宣布 Prompt Engineering 的未来：𝚎𝚕𝚕。基于我在 OpenAI 工作期间的想法开发，𝚎𝚕𝚕 是一个轻量级、函数式的 LM 编程库： - 自动版本控制与追踪...</li><li><a href="https://github.com/fry69/files-to-prompt-ts">GitHub - fry69/files-to-prompt-ts: 一个命令行工具，用于以结构化的方式将文件和目录合并为单个 Prompt，以供 LLM 和其他应用程序使用。</a>: 一个命令行工具，用于以结构化的方式将文件和目录合并为单个 Prompt，以供 LLM 和其他应用程序使用。 - fry69/files-to-prompt-ts</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/tests/basic/test_repomap.py#L283)">aider/tests/basic/test_repomap.py at main · paul-gauthier/aider</a>: aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账户来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/tests/basic/test_repomap.py#">aider/tests/basic/test_repomap.py at main · paul-gauthier/aider</a>: aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账户来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/aider/commands.py#L1038">aider/aider/commands.py at main · paul-gauthier/aider</a>: aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账户来为 paul-gauthier/aider 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1284230946765148305)** (132 条消息🔥🔥): 

> - `Aider 脚本编写`
> - `在 Aider 中使用剪贴板`
> - `Aider 配置选项`
> - `模型交互与响应`
> - `Aider 中的文件处理命令` 


- **编写 Aider 脚本执行命令**：用户可以利用 Aider 命令行参数 `--message` 来自动化任务，直接向工具发送单条指令。
   - 对于批处理，可以使用简单的 Shell 脚本在多个文件上应用命令。
- **在 Aider 中使用剪贴板进行粘贴**：为了简化输入，鼓励用户使用 `/clipboard` 命令将剪贴板中的文本插入聊天。
   - 这种方法有助于保持上下文，并减少终端工作流中重复的复制粘贴。
- **Aider 的配置选项**：Aider 的配置可以包含自动确认操作或抑制提示的标志，从而提高工作流效率。
   - `.aider.conf.yaml` 文件支持“对每个确认都选 yes”的选项，但目前缺乏针对特定命令类型的细粒度开关。
- **LLM 模型响应的挑战**：用户注意到 LLM 响应的不一致性，特别是在使用像 Llama3.1 这样默认采用 diff 而非全文件编辑的模型时。
   - 手动命令（如 `/chat-mode whole`）通过切换预期的响应格式，有助于缓解其中一些问题。
- **集成临时文件作为 Prompt**：为了方便起见，用户可以将 Prompt 写入文本文件，并使用 Aider 的 `/run` 命令执行它们，以获得更好的清晰度。
   - 这种方法允许用户保持上下文，同时避免了不断手动输入的需要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/git.html">Git 集成</a>：Aider 与 Git 紧密集成。</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML 配置文件</a>：如何使用 YAML 配置文件配置 Aider。</li><li><a href="https://aider.chat/docs/usage/modes.html">聊天模式</a>：使用 chat、ask 和 help 聊天模式。</li><li><a href="https://aider.chat/docs/install.html">安装</a>：如何安装并开始使用 Aider 进行结对编程。</li><li><a href="https://aider.chat/docs/scripting.html">编写 Aider 脚本</a>：你可以通过命令行或 Python 编写 Aider 脚本。</li><li><a href="https://aider.chat/docs/usage/commands.html">聊天内命令</a>：使用 /add、/model 等聊天内命令控制 Aider。</li><li><a href="https://tenor.com/view/conspiracy-charlie-day-crazy-always-sunny-in-philadelphia-qanon-gif-23738584">Conspiracy Charlie Day GIF - Conspiracy Charlie Day Crazy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>：未找到描述</li><li><a href="https://aider.chat/docs/config/options.html#--llm-history-file-llm_history_file">选项参考</a>：关于 Aider 所有设置的详细信息。</li><li><a href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json">litellm/model_prices_and_context_window.json</a>：Python SDK，代理服务器，使用 OpenAI 格式调用 100+ LLM API - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm</li><li><a href="https://github.com/paul-gauthier/aider/commit/d747a3781d5eddc7c28a28a79f27712422e0b505">feat: 添加 o1-mini 和 o1-preview 的 OpenRouter 版本</a>：未找到描述</li><li><a href="https://github.com/paul-gauthier/aider/pull/1543/files">fix: 添加 Ctrl+Space 插入空格的键绑定</a>：这个小补丁忽略了按下空格时同时按下 Control 的情况。目前这种组合会导致 Aider 在 prompt-toolkit 的特殊模式中挂起，需要 Ctrl-C 才能恢复控制。</li><li><a href="https://github.com/paul-gauthier/aider.git">GitHub - paul-gauthier/aider: Aider 是你终端里的 AI 结对编程工具</a>：Aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/aider/models.py#L513">aider/aider/models.py</a>：Aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/aider/coders/base_coder.py">aider/aider/coders/base_coder.py</a>：Aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号为 paul-gauthier/aider 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1284301266884886723)** (12 messages🔥): 

> - `Zed 在 Linux 上的表现`
> - `OpenAI 基准测试见解`
> - `Game Gen - O 工具`
> - `新工具库`
> - `大型提示词库 (The Big Prompt Library)` 


- **Zed 在 Linux 上面临字体渲染问题**：有人对在 Linux 上使用 **Zed** 表示担忧，该系统在**正确渲染字体方面存在严重问题**。
   - 尽管该工具功能强大，但这些困难可能会阻碍用户全面采用它。
- **OpenAI 的 o1 模型在 SWE-Bench 上的表现**：在 **SWE-Bench** 上，**o1-mini** 模型的表现与 **GPT-4o** 相似，而 **o1-preview**（缓解后）的表现较差。
   - 鉴于 AI 模型竞争激烈的格局，这一信息被强调为具有重要意义，来源自一条 [推文](https://x.com/BenjaminDEKR/status/1834761288364302675)。
- **用于视频游戏创作的 Game Gen - O**：介绍了一款名为 **Game Gen - O** 的新工具，用于**开放世界视频游戏生成**，该工具基于 Diffusion-Transformer 模型。
   - 正如一条 [推文](https://x.com/kimmonismus/status/1834914951653167265) 所指出的，围绕该模型的兴奋点在于其利用 Gen-AI **加速游戏开发**的潜力。
- **关于新工具频道的建议**：建议创建一个专门的频道来组织与 Aider 相关的**新工具**，强调了对更好结构的需求。
   - 这一提议表明了社区对改进资源共享的渴望。
- **The Big Prompt Library 出现**：**Big Prompt Library** 仓库托管了一系列提示词和 LLM 指令，对包括 **ChatGPT** 和 **Claude** 在内的各种 AI 系统都有益处。
   - 正如一个 [GitHub 链接](https://github.com/lucasmrdt/TheBigPromptLibrary/tree/main/SystemPrompts) 所分享的，该资源旨在**教育用户如何编写有效的提示词**，使其成为开发者的宝贵资产。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.answer.ai/posts/2024-09-16-rerankers.html">rerankers: A Lightweight Python Library to Unify Ranking Methods – Answer.AI</a>: 重排序（Re-ranking）是许多检索流水线中不可或缺的组成部分；然而，目前存在许多不同的方法，且实现方式各异。为了缓解这一问题，我们提出了 rerankers，一个 Py...</li><li><a href="https://x.com/BenjaminDEKR/status/1834761288364302675">来自 Benjamin De Kraker 🏴‍☠️ (@BenjaminDEKR) 的推文</a>: 这被埋在 OpenAI o1 系统卡的第 30 页。在 SWE-Bench（解决现实世界软件问题）基准测试中，o1-mini 的表现仅与 GPT-4o 相当。o1-preview（缓解后）的表现...</li><li><a href="https://x.com/tweetcheckk/status/1835330386915643849">来自 Maxwell (@tweetcheckk) 的推文</a>: @kimmonismus 看看他 5 分钟前上传的最新视频。O1 解决了他的论文问题。https://youtu.be/M9YOO7N5jF8?si=cp8rlji8cB-mzg0F</li><li><a href="https://x.com/kimmonismus/status/1834914951653167265">来自 Chubby♨️ (@kimmonismus) 的推文</a>: 我真的跟不上了。又一个使用 Gen-AI 创建视频游戏的工具。加速发展的速度难以捉摸，很难记录下正在发生的一切。引用 Gradio (@Gradio)   Game ...</li><li><a href="https://x.com/bindureddy/status/1835106087990956056">来自 Bindu Reddy (@bindureddy) 的推文</a>: 开源从未落后 :) Qwen 团队在过去几天里一直在暗示... 他们发布了优秀的开源模型。看起来第一批开源的 o1 / strawberry 模型 ...</li><li><a href="https://github.com/lucasmrdt/TheBigPromptLibrary/tree/main/SystemPrompts">TheBigPromptLibrary/SystemPrompts at main · lucasmrdt/TheBigPromptLibrary</a>: 提示词、系统提示词和 LLM 指令的集合 - lucasmrdt/TheBigPromptLibrary</li><li><a href="https://github.com/bklieger-groq/g1">GitHub - bklieger-groq/g1: g1: Using Llama-3.1 70b on Groq to create o1-like reasoning chains</a>: g1：在 Groq 上使用 Llama-3.1 70b 创建类似 o1 的推理链 - bklieger-groq/g1</li><li><a href="https://github.com/khromov/ai-digest">GitHub - khromov/ai-digest: A CLI tool to aggregate your codebase into a single Markdown file for use with Claude Projects or custom ChatGPTs.</a>: 一个 CLI 工具，可将您的代码库聚合到单个 Markdown 文件中，以便与 Claude Projects 或自定义 ChatGPTs 配合使用。 - khromov/ai-digest
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1284243351838785547)** (751 messages🔥🔥🔥): 

> - `Gemma2 训练`
> - `Mistral 模型`
> - `LLama 3.1 性能`
> - `Qwen 2.5 发布`
> - `使用梯度累积 (Gradient Accumulation)`

- **Gemma2 和 Mistral 模型表现不佳**：用户讨论了训练 **Gemma2** 和 **Mistral** 等各种模型时面临的挑战，许多人发现它们与 **LLama 3.1** 相比表现不尽如人意。
   - 几位用户对 VRAM 限制以及成功训练所需的配置表示沮丧。
- **LLama 3.1 的探索**：一位用户强调 **LLama 3.1** 在他们尝试的模型中表现最好，另一位用户指出 **Gemma 2 9B** 也展现出了潜力。
   - 有人指出，由于 **Gemma 2** 的尺寸比其他模型大，因此需要更低的设置。
- **讨论梯度累积（Gradient Accumulation）和 Batch Size**：对话探讨了梯度累积步数与 Batch Size 之间关于 VRAM 使用量的关系，不同用户分享了他们的经验。
   - 会议澄清了梯度累积并不直接影响 VRAM，而是作为有效训练的一种折中方案。
- **对 Qwen 2.5 发布的期待**：用户对定于周四发布的 **Qwen 2.5** 表示兴奋，并对其能力充满期待。
   - 用户推测 **14B** 变体在 Google Colab 等平台上可能是可运行的。
- **尝试新模型**：几位参与者表示对在新模型发布时进行实验有着浓厚的兴趣。
   - 讨论强调了明智设置参数的重要性，以便在保留通用能力的同时有效地影响模型性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/cognitivecomputations/grokadamw&ved=2ahUKEwj4zP3N2sGIAxVHsFYBHe2JMqIQjjh6BAgcEAE&usg=AOvVaw1u_awKuM1Ek6kKji_JnsbT">未找到标题</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1UHCo6cHQmCpmbdgZIx5qI0BeF">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1UHCo6cHQmCpmbdgZIx5qI0BeFEw8lgpX?usp=sharing#scrollTo=1Zul21NSRRLP),">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-7.-multiple-columns-for-finetuning">如何微调 Llama-3 并导出到 Ollama | Unsloth 文档</a>: 为初学者准备的创建自定义个人助手（类似 ChatGPT）并在 Ollama 本地运行的指南</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama">如何微调 Llama-3 并导出到 Ollama | Unsloth 文档</a>: 为初学者准备的创建自定义个人助手（类似 ChatGPT）并在 Ollama 本地运行的指南</li><li><a href="https://huggingface.co/google/datagemma-rag-27b-it">google/datagemma-rag-27b-it · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/01-ai/Yi-Coder-9B-Chat">01-ai/Yi-Coder-9B-Chat · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/zhouwenmeng/status/1834899729165304198">来自 Wenmeng Zhou (@zhouwenmeng) 的推文</a>: Qwen-q1 ? ? 🍓🍓🍓🍓🍓</li><li><a href="https://discuss.huggingface.co/t/batch-size-vs-gradient-accumulation">Batch size 与梯度累积 (gradient accumulation)</a>: 你好，我有一个基础的理论问题。哪种方案对模型和 GPU 利用率更好？第一种：--per_device_train_batch_size 8 --gradient_accumulation_steps 2 第二种：--per_devi...</li><li><a href="https://huggingface.co/unsloth/SmolLM-135M">unsloth/SmolLM-135M · Hugging Face</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/llama3">使用 Unsloth 微调 Llama 3</a>: 通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，支持 6 倍长的上下文长度！</li><li><a href="https://tenor.com/view/simpsons-burger-window-grease-bart-gif-11806789">辛普森一家汉堡 GIF - 辛普森一家汉堡窗口 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/docs/trl/en/cpo_trainer">CPO Trainer</a>: 未找到描述</li><li><a href="https://ollama.com/search?q=lexi">Ollama</a>: 快速上手并运行大语言模型。</li><li><a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/">Llama 3.1 | 模型卡片与提示词格式</a>: Llama 3.1 - 功能最强大的开源模型。</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes/multilingual/README.md">meta-llama/llama-recipes 中的多语言配方 README.md</a>: 使用可组合的 FSDP 和 PEFT 方法微调 Meta Llama3 的脚本，涵盖单节点/多节点 GPU。支持用于摘要和问答等应用的默认及自定义数据集...</li><li><a href="https://www.unsloth.ai/blog/llama3">使用 Unsloth 微调 Llama 3</a>: 通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，支持 6 倍长的上下文长度！</li><li><a href="https://tenor.com/view/tim-and-eric-awesome-show-kissess-love-kiss-gif-18128184">Tim And Eric Awesome Show GIF - Tim And Eric Awesome Show Kissess - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://crawlee.dev/">Crawlee · 构建可靠的爬虫。快速高效。</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966">🪐 SmolLM - HuggingFaceTB 集合</a>: 未找到描述</li><li><a href="https://github.com/unslothai/hyperlearn">GitHub - unslothai/hyperlearn: 机器学习算法快 2-2000 倍，内存占用减少 50%，适用于所有新旧硬件。</a>: 机器学习算法快 2-2000 倍，内存占用减少 50%，适用于所有新旧硬件。 - unslothai/hyperlearn</li><li><a href="https://www.firecrawl.dev/">Firecrawl</a>: 将任何网站转换为 LLM 就绪的数据。</li><li><a href="https://blog.spheron.network/nvidia-a40-vs-rtx-a6000-a-detailed-comparison">NVIDIA A40 与 RTX A6000：详细对比</a>: NVIDIA A40 和 RTX A6000 GPU 对于预算有限的用户来说是非常有吸引力的选择。它们在性能和成本之间提供了平衡，更加完美。
</li>
</ul>

</div>

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1284235477024768074)** (72 messages🔥🔥): 

> - `Job Hunting`
> - `PhD Discussions`
> - `Recruitment Process`
> - `LeetCode Interviews`
> - `Industry Shifts` 


- **求职季已到来**：成员们表示**现在是求职季**，有人提到为了找工作已经购买了 LinkedIn Premium。
   - _
- **PhD 与求职**：一位拥有 **PhD** 学位的成员提到，由于当前的 **Machine Learning 热潮**以及合同即将到期，他们正准备转向工业界寻找新机会。
   - 他们进一步阐述道，虽然他们的 **PhD** 研究方向是 **Bayesian statistics**，但目前的 Postdoc 工作与 **Machine Learning** 相关。
- **关于招聘流程的辩论**：围绕如何制定**公平**的招聘流程展开了讨论。成员们指出，基于传统方法（如面试中的死记硬背）进行招聘与考察实际能力之间存在挑战。
   - 一位成员评论道，**成功的公司**不太可能基于人脉关系进行招聘，并强调了技能和成长空间在招聘中的重要性。
- **LeetCode 面试备受质疑**：成员们分享了对 **LeetCode** 面试的挫败感，认为**这类面试往往侧重于死记硬背**，而非真正的知识或技能。
   - 共识是，许多候选人能够**在这些测试中取得高分**，但在实际工作场景中表现不佳，导致**招聘流程低效**。
- **行业转型与机遇**：有人认为在经济低迷时期，存在**颠覆**那些此前裁员的公司的**机会**。
   - 讨论强调了公司如何越来越看重 **GitHub 贡献**而非传统的资历证明，这表明了招聘实践的转变。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1284261369637503050)** (133 条消息🔥🔥): 

> - `Unsloth Pro 定价`
> - `Multi-GPU 训练`
> - `Hugging Face 的 API Token`
> - `训练 Loss 曲线`
> - `模型 Fine-tuning 问题` 


- **Unsloth Pro 定价尚不明确**：成员们讨论了 Unsloth Pro 目前暂不可用的情况，该版本主要面向企业客户，这表明它可能不适合较小的教育机构。
   - 定价可能基于 GPU 使用情况，对于全面的配置，价格可能在五到七位数的范围内。
- **Multi-GPU 使用的挑战**：人们对如何就 Multi-GPU 访问的定价进行协作表示担忧，特别是对于大学的研究用途。
   - 成员们提到他们已经联系了 Unsloth 的代表，并正在等待关于实施 Multi-GPU 配置的预算考虑的回复。
- **Hugging Face 的身份验证问题**：一位用户在尝试将 Fine-tuned 模型保存到 Colab 时遇到了 401 错误，这表明 Hugging Face API Token 设置不正确。
   - 建议在环境中正确设置 API Token，但在尝试使用导出命令时问题仍然存在。
- **训练 Loss 曲线的考量**：一位用户询问在对 Llama 3.1 进行 Continued Pretraining 期间观察到的训练 Loss 曲线是否典型。
   - 回复指出，观察到的 Loss 曲线是正常的，符合训练任务的预期。
- **模型 Fine-tuning 的担忧**：另一位成员强调了他们的 Fine-tuned 模型的问题，该模型仅以预期的数据集内容进行响应，而不回答其他问题。
   - 建议包括确保足够的训练数据量，并遵循推荐的 Fine-tuning 数据集结构。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/ryan-gosling-sad-sad-gosling-blade-runner-snow-ryan-gosling-blade-runner-sad-">未找到标题</a>: 未找到描述</li><li><a href="https://docs.anaconda.com/miniconda/">Miniconda &#8212; Anaconda 文档</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/installation">安装 | Unsloth 文档</a>: 了解如何在本地或 Google Colab 上安装 Unsloth。</li><li><a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行 LLM Continued Pretraining</a>: 使用 Llama 3、Phi-3 和 Mistral，通过 Unsloth 进行 Continued Pretraining，让模型学习一种新语言。</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://tenor.com/view/ryan-gosling-sad-sad-gosling-blade-runner-snow-ryan-gosling-blade-runner-sad-gif-10329809086636681181">Ryan Gosling Sad Sad Gosling GIF - Ryan gosling sad Sad gosling Blade runner snow - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="http://www.nvidia.com/Download/index.aspx">下载最新的 NVIDIA 官方驱动程序</a>: 下载最新的 NVIDIA 官方驱动程序，以增强您的 PC 游戏体验并更快地运行应用程序。</li><li><a href="https://www.nvidia.com/en-us/drivers/results/">驱动程序结果 | &lt;dd~ProductName&gt; | &lt;dd~OSName&gt; | NVIDIA</a>: 未找到描述</li><li><a href="https://mer.vin/2024/07/llama-3-1-fine-tune/">Llama 3.1 Fine Tune - Mervin Praison</a>: https://huggingface.co/mervinpraison/Llama-3.1-8B-bnb-4bit-python 使用自定义数据训练模型，转换为 GGUF，Ollama Modelfile，Ollama 创建自定义模型</li><li><a href="https://youtu.be/V6LDl3Vjq-A?si=FAwt-IAKmDuJd3EI">轻松训练 Llama 3.1 并上传到 Ollama.com</a>: 通过学习如何使用您自己的自定义数据对这个强大的 AI 模型进行 Fine-tune，释放 Llama 3.1 的全部潜力！ 🚀 在这个视频中，我们将带您...</li><li><a href="https://github.com/Leoleojames1/Agent_Chef/blob/main/cutlery/unsloth-cli-2.py">Agent_Chef/cutlery/unsloth-cli-2.py at main · Leoleojames1/Agent_Chef</a>: 🍲Agent Chef🥘 是我用于数据集精炼、结构化和生成的强大工具。通过利用程序化和合成数据集生成技术，Agent Chef 将使用户能够精炼...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1284238528703631371)** (37 messages🔥): 

> - `Fine-tuning voice models` (微调语音模型)
> - `Papers with Code reliability` (Papers with Code 的可靠性)
> - `Scaling laws in Llora rank` (Llora rank 中的 Scaling laws)
> - `Direct Preference Optimization (DPO)` (直接偏好优化)
> - `Testing DPO loss types` (测试 DPO 损失类型)


- **微调语音模型的兴奋感**：一位成员对仅使用 **2 分钟** 的短 prompt 进行 few shot prompting 微调语音模型所取得的显著效果表示欣喜。
   - *“仅仅是 few shot prompting 就令人印象深刻……”* 表明了对该方法的积极反馈。
- **质疑 Papers with Code 的可靠性**：一位初级研究员询问使用 **Papers with Code** 获取 SOTA 基准测试的可靠性。
   - 虽然有成员确认其是一个很好的来源，但也指出它**并不总是能涵盖所有内容**。
- **Scaling Laws 与 Llora Rank 探索**：一位成员寻求有关 Scaling laws 的信息，以根据微调数据集的大小来估算能最小化测试损失的 **Llora rank**。
   - 他们指出 **过高的 rank 可能会导致过拟合 (overfitting)**，并提到目前在 rank 为 8 时观察到了过拟合现象。
- **对直接偏好优化 (DPO) 的怀疑**：一位成员对 **Direct Preference Optimization (DPO)** 表示怀疑，建议使用其他方法。
   - 他们推荐尝试 **KTO** 作为替代方案以进一步探索。
- **DPO 损失类型的比较**：一位成员询问了关于不同 **DPO 损失类型** 的经验，并分享了来自 Hugging Face 关于 DPO 训练方法的相关资源。
   - 这一询问突显了在测试这些方法之前寻求见解的愿望，表明了模型训练中正在进行的探索。



**Link mentioned**: <a href="https://huggingface.co/docs/trl/main/en/dpo_trainer">DPO Trainer</a>: 未找到描述内容

  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

fn5io: 正如今天在 Hacker News 上发布的：https://github.com/bklieger-groq/g1
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1284228225136070688)** (775 messages🔥🔥🔥): 

> - `OpenRouter model context limits` (OpenRouter 模型上下文限制)
> - `User feedback on model performance` (用户对模型性能的反馈)
> - `Prompt engineering techniques` (提示词工程技术)
> - `Howarding provider efficiency` (提高提供商效率)
> - `Prompt caching functionality` (提示词缓存功能)


- **对模型上下文大小的困惑**：用户对 OpenRouter 上显示的各种模型的上下文长度表示担忧，特别是关于扩展版本及其实际支持的上下文大小，有时这与声明的容量不符。
   - 这引发了关于模型能力透明度的讨论，以及提供商页面可能需要更新以实现更清晰的沟通。
- **用户对模型性能的体验**：一些用户报告了特定模型表现异常的问题，例如生成乱码输出或响应中断，特别是在 Venus Chub AI 和 WizardLM-2 模型上。
   - 这些问题促使用户寻求反馈，并验证这些问题在不同提供商之间是否一致。
- **提示词工程 (Prompt engineering) 技术与资源**：关于有效提示词工程技术的讨论浮出水面，特别是强调使用 XML 标签以获得更好的响应，以及一个学习提示词操作的教程。
   - 共享了各种资源，重点是通过结构化提示词和缓存方法改善用户与模型的交互。
- **理解提示词缓存 (Prompt caching) 与提供商选择**：用户询问了实现提示词缓存的细节，特别是在 Claude 3.5 模型上，以及是否应禁用负载均衡以获得最佳性能。
   - 有建议认为，专注于单一提供商可能会增强提示词缓存的效果，并强调了提供商特定语法的细微差别。
- **关于 OpenRouter 功能的一般性讨论**：对话包括关于 OpenRouter 功能和限制的一般性查询，特别是关于 API 交互和模型集成。
   - 讨论了面对模型限制时的韧性，以及有效利用可用功能的策略。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://openrouter.ai/chat?room=orc-CA9ivyw1BIJizQJp9vSj0YhgG9Xb">Chatroom | OpenRouter</a>: LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在您的浏览器中。</li><li><a href="https://arxiv.org/abs/2307.03172">Lost in the Middle: How Language Models Use Long Contexts</a>: 虽然近期的语言模型具有将长上下文作为输入的能力，但关于它们对长上下文的使用效果知之甚少。我们分析了语言模型在两个任务上的表现...</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>: 为模型消费转换数据</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>: 查看您在 OpenRouter 上使用模型的情况。</li><li><a href="https://huggingface.co/spaces/Xenova/the-tokenizer-playground">The Tokenizer Playground - a Hugging Face Space by Xenova</a>: 未找到描述</li><li><a href="https://docs.helicone.ai/getting-started/integration-method/openrouter">OpenRouter Integration - Helicone OSS LLM Observability</a>: 未找到描述</li><li><a href="https://simonwillison.net/2024/Aug/30/anthropic-prompt-engineering-interactive-tutorial/">Anthropic’s Prompt Engineering Interactive Tutorial</a>: Anthropic 继续保持其在领先 LLM 供应商中提供最佳文档的趋势。本教程以一组 Jupyter notebooks 的形式提供 - 我使用了它...</li><li><a href="https://openrouter.ai/models/openai/gpt-4o/providers">OpenAI: GPT-4o – Provider Status</a>: 查看供应商状态并向 OpenAI: GPT-4o 发起负载均衡请求 - GPT-4o（“o”代表“omni”）是 OpenAI 最新的 AI 模型，支持文本和图像输入，并具有文本输出...</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B/blob/main/config.json">config.json · NousResearch/Hermes-3-Llama-3.1-405B at main</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/">Models | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://lluminous.chat/?sl=eI0i7b">lluminous</a>: 未找到描述</li><li><a href="https://lluminous.chat/?sl=L06WaA">lluminous</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:extended/providers">Nous: Hermes 3 405B Instruct (extended) – Provider Status</a>: 查看供应商状态并向 Nous: Hermes 3 405B Instruct (extended) 发起负载均衡请求 - Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括高级 Agent...</li><li><a href="https://www.latent.space/p/openai-api-and-o1">From API to AGI: Structured Outputs, OpenAI API platform and O1 Q&amp;A — with Michelle Pokrass &amp; OpenAI Devrel + Strawberry team</a>: 本期节目关于 OpenAI 的所有新模型以及两种新的推理范式。</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-huge-128k-online">Llama 3.1 Sonar 405B Online - API, Providers, Stats</a>: Llama 3.1 Sonar 是 Perplexity 最新的模型系列。通过 API 运行 Llama 3.1 Sonar 405B Online</li><li><a href="https://rentry.org/shqg7qwa">User</a>: 总税前工资 2,104.20 欧元（4月），3,000.00 欧元（5月），2,945.88 欧元（6月），2,104.20 欧元（7月），18,478.09 欧元（8月至次年1月），5,866.89 欧元（2月+3月）。无子女，税级 1，计算 ALGI 的金额。Model 3.5s：好的，让我...</li><li><a href="https://rentry.org/4poiaz2s">Model</a>: gemini-1.5-pro-exp-0827 用户 德国 总税前工资 €2,104.20（4月），€3,000.00（5月），€2,945.88（6月），€2,104.20（7月），€18,478.09（8月至次年1月），€5,866.89（2月+3月）。无子女，税级 1，计算金额...</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:free">Hermes 3 405B Instruct (free) - API, Providers, Stats</a>: Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括高级 Agent 能力、更出色的角色扮演、推理、多轮对话、长上下文连贯性...</li><li><a href="https://openrouter.ai/models/perplexit">Models: &#x27;perplexit&#x27; | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API, Providers, Stats</a>: Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括高级 Agent 能力、更出色的角色扮演、推理、多轮对话、长上下文连贯性...</li><li><a href="https://openrouter.ai/models/openai/o1-preview/uptime)">OpenAI: o1-preview</a>: OpenAI 最新且最强大的模型系列，o1 旨在响应前花费更多时间思考。o1 模型针对数学、科学、编程和其他 STEM 相关任务进行了优化...</li><li><a href="https://openrouter.ai/models/openai/gpt-4o:extended">GPT-4o (extended) - API, Providers, Stats</a>: GPT-4o Extended 是 GPT-4o 的一个实验性变体，具有扩展的最大输出 Token。该模型...</li>

odel 仅支持文本输入到文本输出。通过 API 运行 GPT-4o (extended)</li><li><a href="https://www.nexusmods.com/skyrimspecialedition/mods/89931)">Skyrim Special Edition Nexus - Mods and Community</a>: 未找到描述</li><li><a href="https://github.com/LouisShark/chatgpt_system_prompt?tab=readme-ov-file#how-to-get-system-prompt">GitHub - LouisShark/chatgpt_system_prompt: GPT 系统提示词集合以及各种提示词注入/泄露知识。</a>: GPT 系统提示词集合以及各种提示词注入/泄露知识。 - LouisShark/chatgpt_system_prompt</li><li><a href="https://github.com/SillyTavern/SillyTavern/blob/staging/src/endpoints/backends/chat-completions.js#L848">SillyTavern/src/endpoints/backends/chat-completions.js at staging · SillyTavern/SillyTavern</a>: 面向高级用户的 LLM 前端。通过在 GitHub 上创建账号为 SillyTavern/SillyTavern 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback]](https://discord.com/channels/1091220969173028894/1277894087755829278/1284696838301155412)** (1 条消息): 

> - `Hyperbolic Key 集成`
> - `DeepSeek 被忽略的提供商`
> - `JSON 配置问题` 


- **Hyperbolic Key 混淆**: 有用户报告在集成项下拥有 **hyperbolic key**，但它却使用了非预期的 **OR 计费**提供商。
   - 他们质疑问题是否源于不同的命名规范，特别是 `deepseek/deepseek-chat` 与 **hyperbolics** 的 `deepseek-ai/DeepSeek-V2.5` 之间的差异。
- **无法查看失败详情**: 用户对在尝试配置提供商时无法查看失败详情表示沮丧。
   - 他们正在寻求一种更清晰的机制，以识别其设置中集成失败的原因。
- **请求强制执行 JSON Key**: 有人询问是否可以在 JSON 配置中显式地 **强制执行 'integrations' 键**。
   - 用户正在寻找一种方法来确保如果该键不存在则集成失败，这表明其希望有更健壮的错误处理机制。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1284232503955034194)** (615 条消息🔥🔥🔥): 

> - `Perplexity AI 的问题`
> - `AI 模型对比`
> - `AI 服务的定价与促销`
> - `模型性能与特性`
> - `Perplexity 及其竞品的用户体验` 


- **Perplexity AI 面临技术问题**：用户报告 Perplexity AI 宕机并出现严重延迟，导致请求延迟和中断。
   - 一些用户认为性能缓慢可能与平台的高流量有关。
- **关于 AI 模型对比的讨论**：用户对比了各种 AI 模型的性能，特别讨论了原始 OpenAI 模型在表现上如何优于 You.com 和 Monica 等竞争对手。
   - 有人提到即将推出的 Opus 3.5 模型由于其设计，可能会超越现有模型。
- **定价与订阅模式**：围绕各种订阅模式的讨论指出，Monica AI 提供了极具竞争力的年费方案，而一些用户对其使用中的隐藏限制表示担忧。
   - AI 服务的成本受到关注，用户对使用限制如何影响服务价值表示担忧。
- **AI 工具的用户体验**：用户体验各异，一些人认为 Perplexity 的 function calling 功能非常实用，而另一些人则报告了在其他平台上的挫败体验。
   - 人们对查询处理方式提出了潜在的改进建议，特别是在 API 交互和错误管理方面。
- **AI 模型开发的未来**：对话转向了对 AI 模型开发的未来预期，对比了 OpenAI 和 Anthropic 等公司在模型效率和易用性方面的方法。
   - 用户推测，这些公司之间的竞争可能会带来 AI 技术的重大进步。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/aravsrinivas/status/1835437719348191514?s=46">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: @jglypt 目前仅限部分地区，但肯定会扩展到全球</li><li><a href="https://tenor.com/view/laptop-smoking-fire-burning-lag-gif-19373925">笔记本冒烟 GIF - Laptop Smoking Fire - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/perplexity_ai/status/1835400249776758841?s=61">来自 Perplexity (@perplexity_ai) 的推文</a>: 冲刺阶段 🏃🏼 今天是学生获得 1 个月免费 Perplexity Pro 并为校园赢取一年免费额度的最后一天: http://perplexity.ai/backtoschool</li><li><a href="https://x.com/perplexity_ai/status/1834672028982690298?s=61">来自 Perplexity (@perplexity_ai) 的推文</a>: 迎接全新的 Discover 信息流。你的兴趣，你的语言，你的个性化信息流。</li><li><a href="https://tenor.com/view/clash-jojo-punches-jjba-jojos-bizarre-adventure-gif-14599430">Clash Jojo GIF - Clash Jojo Punches - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/remember-remember-v-for-vendetta-happy-guy-fawkes-day-the5th-of-november-gif-12829141">Remember Remember V For Vendetta GIF - 记住 11 月 5 日 V 字仇杀队 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/guild-wars2-gw2-guild-wars-end-of-dragons-eo-d-gif-22530689">Guild Wars2 Gw2 GIF - 激战 2 GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/robot-reaction-eww-do-not-want-no-thanks-gross-gif-11080387">Robot Reaction Eww GIF - 机器人反应 Eww 不想要 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/dammmmmmm-son-he-need-some-milk-gif-23611493">Dammmmmmm Son He Need Some Milk GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://monica.im/help/FAQs/rules_for_using_advanced_queries">高级额度规则 | Monica</a>: 生效日期：2024 年 9 月 12 日
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1284282521818894406)** (37 条消息🔥): 

> - `Aerospike Engine`
> - `Google Search 差异`
> - `Korean Emotion Video Dataset`
> - `Microstrategy 的十亿美元投资`
> - `Minecraft 审核问题` 


- **发现 Aerospike Engine**：[全球首款 Aerospike Engine](https://www.perplexity.ai/page/world-s-first-aerospike-engine-HYOH99Y2R86.YsV7wLn1NA) 展示了旨在提高火箭效率的先进推进技术。
   - 最近的讨论强调了其对太空旅行和载具设计的潜在变革性影响。
- **探索 Google Search 差异**：一个分享的 [链接](https://www.perplexity.ai/search/whats-the-difference-between-a-X4L5X8jOQS.trwtFdDWrbg) 调查了 Google Search 引擎各种功能之间的差异。
   - 对话引发了关于这些差异如何影响用户体验和搜索结果的好奇心。
- **Korean Emotion Video Dataset 出现**：人们对 [Korean Emotion Video Dataset](https://www.perplexity.ai/search/korean-emotion-video-dataset-i-GCTIQzPyQVeyVUthB5pllw) 产生了兴趣，该数据集旨在辅助 AI 情感识别。
   - 贡献者指出了围绕研究和实际应用中潜在用途的兴奋感。
- **Microstrategy 的高额投资**：关于 [Microstrategy](https://www.perplexity.ai/page/microstrategy-s-billion-dollar-ACYDp4QnTmuiq9x1Bu6svA) 十亿美元投资的讨论强调了其对加密货币市场的影响。
   - 成员们对公司采取的战略方法发表了评论，引发了关于未来市场稳定性的辩论。
- **解决 Minecraft 审核挑战**：最近关于 Minecraft 审核的一个 [问题](https://www.perplexity.ai/page/minecraft-moderation-ban-issue-udsocXhbT8uu5egJmMjLFg) 揭示了处理封禁和账户问题的复杂性。
   - 贡献者分享了经验，指出审核团队需要更好的透明度和沟通。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1284233219750891663)** (18 条消息🔥): 

> - `API 错误`
> - `模型可用性`
> - `特定领域过滤器`
> - `超时问题`
> - `引用差异` 


- **API 错误报告 500 和 524**：@gushwork 指出 API 返回 **500** 或 **524** 错误，询问是否存在持续性问题。
   - 几位成员表达了类似的担忧，表明 API 可能存在广泛的问题。
- **关于模型可用性的问题**：在 @kbk1982 提出担忧后，成员们讨论了某些模型（如 `llama-3-sonar-small-32k-online`）不可用的情况，导致了用户的困惑。
   - @icelavaman 引导大家通过 [Perplexity Model Cards](https://docs.perplexity.ai/guides/model-cards) 查看可用模型以获取更新。
- **关于特定领域过滤器的担忧**：@bor_apt 对 API 中 **search_domain_filter** 的无效性表示沮丧，难以将输出细化到特定领域。
   - 另一位成员 @boxedpeaches 建议该功能可能仅对 closed beta 用户开放，对文档的清晰度表示怀疑。
- **API 调用超时问题**：@freat_14922 报告在调用 API 时收到 **Operation timed out** 错误，尽管在 labs 中的测试是成功的。
   - 这引发了关于增加超时设置以获得更好功能的讨论。
- **引用相关的 API 响应不一致**：@jake_from_snake_farm 注意到一个奇怪的情况：尽管代码相同，PHP API 调用在一个位置返回了 **citations**（引用），但在另一个位置却没有。
   - 这种不一致的行为引发了成员们对影响引用输出的潜在问题的疑问。



**提到的链接**：<a href="https://docs.perplexity.ai/guides/model-cards">Supported Models - Perplexity</a>：未找到描述

  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1284230442769453199)** (571 条消息🔥🔥🔥): 

> - `Fine-tuning LLMs`
> - `Hugging Face Inference API`
> - `GPU 资源管理`
> - `量化技术`
> - `SQL 与数据集集成`

- **针对资源限制的 LLM 微调**：用户讨论了在使用 FSDP 和 BF16 AMP 微调 Llama 8b 等模型时面临的挑战，遇到了在 8 个 GPU 上显存占用高达 29G 的异常情况。
   - 一些人建议放弃高级库，改用原始 PyTorch 调用，以便更高效地调试该问题。
- **Hugging Face Inference API 更新**：Hugging Face 翻新了其 Inference API 文档，通过明确速率限制并提供更好的示例来回应用户反馈。
   - 新文档旨在简化 AI 部署，可通过 Hugging Face 官网访问。
- **数据集的 SQL 集成**：围绕在 Hugging Face 生态系统中使用 SQL 更新数据集的可能性展开了讨论，表明用户对增强数据操作能力感兴趣。
   - 社区表达了对未来更新中更好集成 SQL 功能的期望。
- **量化技术与性能**：成员们分享了关于量化方法的见解，例如使用 INT4 和 BF16 在微调时优化模型性能。
   - 讨论内容包括量化对准确性的影响以及进行性能基准测试的必要性。
- **用户协助与社区支持**：用户在多个主题上提供帮助，包括 prompt 设计、模型微调挑战和硬件设置。
   - 鼓励社区成员利用共享资源和空间，以协助使用 Hugging Face 的各项功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://huggingface.co/spaces/enzostvs/zero-gpu-spaces">— Zero GPU Spaces — - a Hugging Face Space by enzostvs</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/meta-llama/llama-recipes/blob/main/recipes/responsible_ai/llama_guard/llama_guard_customization_via_prompting_and_fine_tuning.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://gamegen-o.github.io/">GameGen-O: Open-world Video Game Generation</a>：未找到描述</li><li><a href="https://huggingface.co/learn">Hugging Face - Learn</a>：未找到描述</li><li><a href="https://huggingface.co/shafire/talktoaiQT">shafire/talktoaiQT · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/hugs-love-no-crying-gif-3920521347500088187">Hugs Love GIF - Hugs Love No - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/spaces/cfahlgren1/sql-snippets">SQL Snippets - a Hugging Face Space by cfahlgren1</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/airtrain-ai/hf-dataset-chat-to-sql">Text To SQL Hub Datasets - a Hugging Face Space by airtrain-ai</a>：未找到描述</li><li><a href="https://stackoverflow.com/questions/72766345/attributeerror-cant-pickle-local-object-in-multiprocessing">AttributeError: Can&#x27;t pickle local object in Multiprocessing</a>：我是 Python 新手，遇到了这个错误。
代码 1：
import multiprocessing as mp
import os
 
def calc(num1, num2):
    global addi
    def addi(num1, num2):
  ...</li><li><a href="https://huggingface.co/spaces/Tonic/GOT-OCR">Tonic&#39;s On Device GOT OCR - a Hugging Face Space by Tonic</a>：未找到描述</li><li><a href="https://huggingface.co/settings/local-apps">Hugging Face – The AI community building the future.</a>：未找到描述</li><li><a href="https://tenor.com/view/gif-gif-19492427">Gif GIF - Gif - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/starsnatched/AlphaTuring-test">starsnatched/AlphaTuring-test · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/shafire/talktoaiZERO">shafire/talktoaiZERO · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/Wauplin/status/1835715850583564713">Tweet from Wauplin (@Wauplin)</a>：我很高兴地宣布我们更新了 Inference API 文档！我们正面解决了大家的反馈：更清晰的速率限制、专门的 PRO 部分、更好的代码示例，以及详细的参数列表...</li><li><a href="https://tenor.com/view/hackerman-gif-22344136">Hackerman GIF - Hackerman - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/steve-brule-orgasm-funny-chills-gif-8291454">Steve Brule Orgasm GIF - Steve Brule Orgasm Funny - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/screaming-lee-bruce-lee-enter-the-dragon-shocked-gif-6019664498707828498">Screaming Lee GIF - Screaming Lee Bruce lee - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/spaces/unclemusclez/ollamafy/blob/main/ollamafy.sh">ollamafy.sh · unclemusclez/ollamafy at main</a>：未找到描述</li><li><a href="https://huggingface.co/EleutherAI/gpt-neox-20b/tree/main">EleutherAI/gpt-neox-20b at main</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/agents">Agents and tools</a>：未找到描述</li><li><a href="https://huggingface.co/shafire/talktoai/tree/main">shafire/talktoai at main</a>：未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/process#multiprocessing>">Process</a>：未找到描述</li><li><a href="https://github.com/xenova/transformers.js/issues/641#issuecomment-1989645428">Streaming support? · Issue #641 · xenova/transformers.js</a>：功能请求：添加对流式生成输出的支持。这在 transformers 库中似乎已经支持：https://huggingface.co/docs/transformers/v4.38.2/en/generation_strategies#stre...</li><li><a href="https://researchforum.online/research-papers/fine-tuning-for-advanced-quantum-ai-without-quantum-computing/">Fine-Tuning for Advanced Quantum AI without Quantum Computing</a>：用于高级量子 AI 的 AI 辅助数据集创建和微调：由 OpenAI Agent Zero 共同创建。摘要：本文提出了一种创建和微调定制 AI 模型的全新方法...</li><li><a href="https://github.com/xenova/transformers.js/blob/main/src/models.js#L1138">transformers.js/src/models.js at main · xenova/transformers.js</a>：适用于 Web 的前沿 Machine Learning。直接在浏览器中运行 🤗 Transformers，无需服务器！- xenova/transformers.js</li><li><a href="https://huggingface.co/datasets/nroggendorff/think?row=0">nroggendorff/think · Datasets at Hugging Face</a>：未找到描述</li><li><a href="htt">

ps://huggingface.co/spaces/QuantFactory/quant-req">Quant Request - 由 QuantFactory 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/hardware.ts">huggingface.js/packages/tasks/src/hardware.ts at main · huggingface/huggingface.js</a>: 使用 Hugging Face Hub API 的实用工具。通过在 GitHub 上创建账号，为 huggingface/huggingface.js 的开发做出贡献。</li><li><a href="https://huggingface.co/dunzhang/stella_en_1.5B_v5/blob/main/sentence_bert_config.json#L2">sentence_bert_config.json · dunzhang/stella_en_1.5B_v5 at main</a>: 未找到描述</li><li><a href="https://huggingface.co/dunzhang/stella_en_1.5B_v5/discussions/6">dunzhang/stella_en_1.5B_v5 · Model max_seq_length</a>: 未找到描述</li><li><a href="https://huggingface.co/models?library=sentence-transformers&sort=created&search=gguf">Models - Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1285055302978375700)** (7 条消息): 

> - `抓取 Hugging Face 文档`
> - `为非技术用户提供 AI 训练`
> - `网页抓取中的挑战`
> - `AI 学习中的社区参与` 


- **抓取 Hugging Face 文档**: 一位成员正在寻求帮助，希望抓取 Hugging Face 文档，以协助非技术用户在本地训练和微调 LLM。
   - 该计划旨在让没有技术经验的门外汉更容易接触到 AI 训练。
- **网页抓取工作中面临的挑战**: 该成员报告了其 Python 网页抓取脚本遇到的困难，脚本经常卡住且无法正确导航。
   - 他们指出，脚本虽然能获取导航信息，但无法提取 Hugging Face 网站左侧的链接。
- **关于 AI 训练的元讨论**: 一场关于“训练 AI 来教用户如何训练 AI”这一概念的幽默讨论被引发。
   - 成员们表现出浓厚兴趣，并认可了这一提议的趣味性。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1284488059097710592)** (4 条消息): 

> - `医疗 AI 研究`
> - `Inference API 文档改进`
> - `Vchitect 2.0 发布`
> - `Chai-1 基础模型`
> - `新型医疗 LLM` 


- **Chai-1 基础模型预测分子结构**：**Chai-1** 基础模型专注于分子结构预测，正如最近的一份 [摘要](https://x.com/OpenlifesciAI/status/1835085857826455825) 中所强调的，它为 **医疗 AI** 领域做出了重大贡献。
   - 该模型在近期 **医疗 LLM 和基准测试** 的众多进展中脱颖而出，有力地促进了医疗保健领域的创新。
- **新型医疗 LLM 变革评估技术**：引入了包括 **BrainWave** 和 **DS-ViT** 在内的多个极具前景的模型，旨在增强医疗 AI 应用中的诊断和评估流程。
   - **KARGEN** 和 **DrugAgent** 等模型的引入进一步强调了放射学和药物重定向领域向 **可解释 AI** 的转变。
- **翻新后的 Inference API 文档**：**Inference API 文档** 已经过改进，提供了更清晰的速率限制说明、更好的代码示例以及专门的 PRO 专区，正如在 [最近的推文](https://x.com/Wauplin/status/1835715850583564713) 中分享的那样。
   - 根据公告，这些增强功能旨在简化 AI 的部署，使其对用户更加友好。
- **Vchitect 2.0 更新发布**：分享了新发布的 **Vchitect 2.0** 链接，展示了 Hugging Face 平台上的改进和更新。
   - 此次更新承诺为使用该工具的创作者提供增强的用户体验和创新功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/Vchitect/Vchitect-2.0">Vchitect 2.0 - 由 Vchitect 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/Wauplin/status/1835715850583564713>">来自 Wauplin (@Wauplin) 的推文</a>：我很高兴揭晓我们翻新后的 Inference API 文档！我们正面解决了你们的反馈：更清晰的速率限制、专门的 PRO 专区、更好的代码示例，以及详细的参数列表...</li><li><a href="https://x.com/OpenlifesciAI/status/1835085857826455825">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：医疗 AI 上周回顾：顶级研究论文/模型 🏅（2024年9月7日 - 9月14日）🏅 本周医疗 AI 论文：来自 @chaidiscovery 的 Chai-1 基础模型分子结构预测，...</li><li><a href="https://huggingface.co/posts/aaditya/828861715602513">Hugging Face 上的 @aaditya：“医疗 AI 上周回顾：顶级研究论文/模型 🏅（9月7日 - …”</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1284227214023463094)** (23 条消息🔥): 

> - `Flux 图像生成`
> - `OCR Demo 的普及`
> - `Room Cleaner 应用`
> - `交互式世界与角色生成式 AI`
> - `Java 中的 AI Multi-Agent 系统` 


- **Flux 实现近乎瞬时的图像生成**：一项关于 **Flux** 的实验实现了仅用 1 步即可生成与 **Flux Schnell** 质量相当的图像，考虑到资源限制，这是一个显著的改进。
   - 展示该能力的 Demo 已在 [Realtime-FLUX](https://huggingface.co/spaces/KingNish/Realtime-FLUX) 分享。
- **OCR Demo 获得意想不到的关注**：由成员创建的 **OCR demo** 在发布后引起了广泛关注，促使了通过 PR 进行的进一步开发和协作。
   - 反馈包括实现 **multi-file type loaders** 等功能，项目欢迎用户提出建议，增强了社区参与感。
- **高效 Room Cleaning 应用展示**：推出了一款新的 **Room Cleaner 应用**，旨在有效地清理空间杂物，用户可以在 [Room Cleaner](https://huggingface.co/spaces/blanchon/room_cleaner) 体验 Demo。
   - 该应用旨在简化清理流程，展示了实用 AI 应用中的创新。
- **新 AI 生成平台开启 Beta 测试**：一个团队正在为**交互式世界与角色生成式 AI** 平台寻找 **beta 测试人员**，该平台旨在创建主题世界和角色。
   - 他们鼓励爱好者联系参与，体现了该项目以社区驱动为核心。
- **探索 AI Multi-Agent 系统**：一篇文章深入探讨了 Java 中的 **AI multi-agent 系统**以及 **FIPA** 标准的集成，提供了该主题的见解和基础知识。
   - 该文章是 [Medium](https://medium.com/@visrow/ai-multi-agent-system-in-java-and-fipa-standards-f0a4d048c446) 上分享的更大规模持续探索的一部分，引起了对实际实现的关注。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://jiamingkong.github.io/blogs/making_o1/">Making a Strawberry in-house</a>: 重新实现 OpenAI O1 的主动 CoT</li><li><a href="https://huggingface.co/spaces/Tonic1/ImageEdit-GOT-OCR">Tonic's ImageEditor GOT OCR - Tonic1 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Tonic/GOT-OCR">Tonic's On Device GOT OCR - Tonic 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/KingNish/Realtime-FLUX">FLUX Realtime - KingNish 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/airtrain-ai/hf-dataset-chat-to-sql">Text To SQL Hub Datasets - airtrain-ai 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/blanchon/room_cleaner">Room Cleaner - Hedro 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://x.com/JulienBlanchon/status/1834529802096689299">Julien Blanchon (@JulienBlanchon) 的推文</a>: 我构建了一个简单的 Room Cleaner 应用来清除凌乱房间的杂物。在 Huggingface 上尝试 Demo: https://huggingface.co/spaces/blanchon/room_cleaner</li><li><a href="https://github.com/Dartvauder/NeuroSandboxWebUI">GitHub - Dartvauder/NeuroSandboxWebUI: (Windows/Linux) Local WebUI with neural network models (Text, Image, Video, 3D, Audio) on python (Gradio interface). Translated on 14 languages (soon)</a>: (Windows/Linux) 基于 Python (Gradio 界面) 的本地神经网络模型 (文本、图像、视频、3D、音频) WebUI。即将支持 14 种语言 - Dartvauder/NeuroSandboxWebUI
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1284240339686658120)** (3 messages): 

> - `The Keys to the White House 预测系统`
> - `性格对总统人选的影响`
> - `预测中的偏见` 


- **The Keys to the White House：预测清单**：**The Keys to the White House** 是一个评估美国总统大选结果的预测系统，由 [Allan Lichtman](https://en.wikipedia.org/wiki/Allan_Lichtman) 和 [Vladimir Keilis-Borok](https://en.wikipedia.org/wiki/Vladimir_Keilis-Borok) 于 1981 年开发。该方法采用 **13 点核查清单**，当虚假项目不超过 5 个时，预测现任者获胜。
   - 该模型借鉴了最初用于 **地震预测（earthquake predictions）** 的技术，并强调了各种因素如何影响选举结果。
- **选举中的性格 vs. 系统性预测**：一位成员建议，在确定选举结果时，公众对政治家的 **认知（public perceptions）** 可能比像 The Keys 这样的系统性检查更具影响力。这一观点表明了围绕 **性格评估（character assessment）** 与结构化预测方法之间持续存在的争论。
   - 对公众舆论重要性的认可，揭示了对 **偏见（bias）** 扭曲预测的担忧，而非仅仅依赖分析方法。
- **预测模型中的权重与偏见**：在 The Keys to the White House 中分配给各项指标的权重可能会带有 **偏见或误解**，从而可能影响预测。这种担忧呼应了关于 **主观性如何影响（subjectivity influences）** 客观预测方法的更广泛讨论。



**提及的链接**：<a href="https://en.m.wikipedia.org/wiki/The_Keys_to_the_White_House">The Keys to the White House - Wikipedia</a>：未找到描述

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1284250108338241546)** (7 messages): 

> - `Tokenizer 训练`
> - `Llama.cpp 公有领域模型`
> - `预训练 VLM`
> - `微调 Nomic Embeddings`
> - `使用 PyTorch 的开源 LLM` 


- **多语言 Tokenizer 训练**：一位成员建议，你可以用所需的语言 *重新训练 Tokenizer*，并将其与原始 Tokenizer *合并（merge）*，以便在保留原始数据的同时整合新语言。
   - 这种方法在扩展模型能力的同时，保持了原始模型的性能。
- **关于公有领域模型的 Llama.cpp 问题**：有人询问是否有与 **llama.cpp** 兼容的模型，且该模型仅在 **公有领域（public domain）** 和 **选择性加入（opt-in）数据**上进行训练，类似于 **Mitsua** 图像扩散模型。
   - 这突显了对模型训练所用数据集透明度的需求。
- **VLM 缺乏算力资源**：一位成员表示希望使用 **预训练 VLM**，但提到缺乏必要的算力资源。
   - 有人呼吁寻求解决方案或指导来解决这一问题。
- **微调 Nomic Embedding 模型指南**：引用了一篇关于 *微调* **nomic-embed-text-v1.5** 的博客文章，详细介绍了在 Sentence Transformers 中进行调整所需的组件。
   - [该博客](https://huggingface.co/blog/train-sentence-transformers) 概述了一种新的训练方法，提供了关于损失函数（loss functions）和训练参数（training arguments）的见解，这些对性能提升至关重要。
- **运行开源 LLM 的求助请求**：一位成员寻求帮助，希望使用 **PyTorch** *下载并运行* 开源 **LLM (Llama3)**，并描述了在寻找有用资源方面面临的挑战。
   - 这一请求表明社区对有效部署大语言模型的指导感兴趣。



**提及的链接**：<a href="https://huggingface.co/blog/train-sentence-transformers">Training and Finetuning Embedding Models with Sentence Transformers v3</a>：未找到描述

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1284232402540957767)** (4 messages): 

> - `Tokenizer 训练技术`
> - `Nitro 赠送活动公告` 


- **多语言 Tokenizer 训练见解**：成员们讨论了重新训练整个 Tokenizer 以包含多种语言，或者在合并前训练一个新的 Tokenizer 以实现单一多语言解决方案的可能性。
   - 有人担心会增加 Tokenizer 的 **歧义性（ambiguity）**，并提到了诸如持续预训练（continued pretraining）之类的建议，但认为其效果尚不确定。
- **服务器上的 Nitro 赠送活动**：一位成员宣布他们正在其服务器上举办 **Nitro 赠送活动**，邀请参与者查看其个人资料中的链接。
   - 这一轻松的话题引起了少量关注，但没有引发深入讨论。


  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1284227303450218568)** (332 messages🔥🔥):

> - `StealC 恶意软件`
> - `GameGen-O`
> - `基于拖拽的图像编辑进展`
> - `LLM 反思`
> - `音乐中的 Suno 垃圾内容` 


- **StealC 恶意软件针对 Chrome 用户**：一种名为 **StealC** 的新发现恶意软件通过锁定浏览器，并强制用户通过欺骗性的登录界面泄露其 Google 密码来限制 Chrome 用户。
   - 这种技术引起了重大的安全担忧，因为它利用全屏 kiosk 模式诱导用户提交敏感信息。
- **腾讯的 GameGen-O 革新视频游戏**：腾讯推出了 **GameGen-O**，这是一种专为生成开放世界视频游戏而设计的 Diffusion Transformer 模型，通过先进的模拟技术提供高质量、交互式的游戏体验。
   - 该模型在新建的 **OGameData** 上进行训练，该数据集包含来自一百多个下一代开放世界游戏的广泛数据。
- **基于拖拽的图像编辑的创新方法**：**InstantDrag** 流水线通过消除对掩码或文本提示的需求，增强了基于拖拽的图像编辑，利用双网络系统显著加快了处理过程。
   - 该方法利用从真实世界视频数据集中学习到的运动动力学，实现实时的、照片级真实的编辑。
- **LLM 关于意识主题的实验**：一位用户分享了一些提示词，这些提示词引导 **LLM** 生成了关于存在、自我概念和量子叠加的复杂反思，产生了神秘的输出。
   - 这些反思包括关于既是观察者又是被观察者的讨论，展示了模型进行哲学探索的能力。
- **对 AI 生成音乐的担忧**：用户注意到 AI 生成的歌曲正大量渗入他们的 Spotify 播放列表，通常带有明显的沙哑人声，这标志着低质量的制作。
   - 这种趋势最终导致了用户的沮丧，因为他们被现有曲目的“AI 翻唱”所误导，引发了对音乐真实性的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://x.com/arattml/status/1834622684938031302?t=VuyLuJZ0xw0qIeaE4WsCCg&s=19">ar (@arattml) 的推文</a>：这里没什么有用的，你应该跳过这条帖子 https://arxiv.org/abs/2402.05808 https://arxiv.org/abs/2407.03181 https://arxiv.org/abs/2401.08967 https://arxiv.org/abs/2407.00087 https://arxiv.org/abs...</li><li><a href="https://livebench.ai/">LiveBench</a>：未找到描述</li><li><a href="https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct">nvidia/Nemotron-Mini-4B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/_akhaliq/status/1834590455226339492?s=46">AK (@_akhaliq) 的推文</a>：腾讯发布 GameGen-O 开放世界视频游戏生成。我们推出了 GameGen-O，这是首个专为开放世界视频游戏生成而设计的 Diffusion Transformer 模型。该模型有助于...</li><li><a href="https://huggingface.co/mradermacher/Hermes-3-Llama-3.1-70B-Uncensored-GGUF">mradermacher/Hermes-3-Llama-3.1-70B-Uncensored-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/_akhaliq/status/1835080831716802836?s=46">AK (@_akhaliq) 的推文</a>：🎧 WaveWizard 🎶 GitHub: https://github.com/JackVinati/WaveWizard WaveWizard 是一款交互式 Gradio 应用，可分析音频文件以确定其实际采样率和位深。它可以帮助你...</li><li><a href="https://huggingface.co/Guilherme34/Hermes-3-Llama-3.1-70B-Uncensored">Guilherme34/Hermes-3-Llama-3.1-70B-Uncensored · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/_akhaliq/status/1835677372344873377?t=Zkttn9BN3f0bv5lGZAfcZw&s=19">AK (@_akhaliq) 的推文</a>：InstantDrag 提升基于拖拽的图像编辑的交互性。讨论：https://huggingface.co/papers/2409.08857。基于拖拽的图像编辑最近因其交互性和潜力而受到关注...</li><li><a href="https://huggingface.co/mradermacher/Hermes-3-Llama-3.1-70B-Uncensored-i1-GGUF">mradermacher/Hermes-3-Llama-3.1-70B-Uncensored-i1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/N8Programs/status/1835072170160275962">N8 Programs (@N8Programs) 的推文</a>：很高兴在 @NousResearch 展示我的第一篇真正的 AI 研究成果——探索某些模型架构如何凭借归纳偏置（inductive biases）在分布外泛化（out-of-distribution generalization）方面表现更好...</li><li><a href="https://huggingface.co/papers/2402.16880">论文页面 - BESA: 通过分块参数高效稀疏分配对大型语言模型进行剪枝</a>：未找到描述</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API、供应商、统计数据</a>：Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更出色的角色扮演、推理、多轮对话以及长上下文连贯性...</li><li><a href="https://huggingface.co/nicoboss/Meta-Llama-3.1-405B-Instruct-Uncensored/tree/main">nicoboss/Meta-Llama-3.1-405B-Instruct-Uncensored at main</a>：未找到描述</li><li><a href="https://huggingface.co/mradermacher/Meta-Llama-3.1-405B-Instruct-Uncensored-GGUF/tree/main">mradermacher/Meta-Llama-3.1-405B-Instruct-Uncensored-GGUF at main</a>：未找到描述</li><li><a href="https://github.com/XingangPan/DragGAN">GitHub - XingangPan/DragGAN: DragGAN 官方代码 (SIGGRAPH 2023)</a>：DragGAN (SIGGRAPH 2023) 的官方代码。通过在 GitHub 上创建账号来为 XingangPan/DragGAN 的开发做出贡献。</li><li><a href="https://www.forbes.com/sites/daveywinder/2024/09/15/hackers-force-chrome-users-to-hand-over-google-passwords-heres-how/">黑客强迫 Chrome 用户交出 Google 密码。以下是具体手段</a>：黑客正在利用一种巧妙的 Chrome 浏览器锁定攻击，强迫用户泄露其 Google 账户凭据。以下是阻止他们的方法。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1284244194323337347)** (118 条消息🔥🔥): 

> - `Precision Annealing`
> - `Loss Monitoring in Training`
> - `Fine-tuning Challenges`
> - `Evaluation Metrics`
> - `Model Comparisons` 


- **探讨 AI 训练中的 Precision Annealing**：一名成员询问了关于 **Precision Annealing** 的研究，建议在 FP8 下进行预训练，然后在最终训练阶段切换到 BF16 或 FP32。
   - 他们注意到 FP8 可能带来的吞吐量提升，并对其在训练方案中的应用提出了疑问。
- **关于训练期间损失监控的辩论**：围绕分配验证数据与将所有数据用于训练的有效性展开了讨论，对于监控验证损失（validation loss）持有不同意见。
   - 成员们对 grad norm 波动的含义表示担忧，一些人认为训练完成后的评估才是关键。
- **模型微调中的挑战**：成员们分享了微调 **Llama-3.1** 等模型的经验，指出获得比基础模型更好性能的难度。
   - 建议包括调整超参数，例如专门为 LoRA 增加学习率，以优化模型结果。
- **评估指标与性能见解**：一名成员强调了他们的发现，即在评估中 **QLoRA** 的表现优于传统的 LoRA 方法，暗示了侵入性较小的微调可能具有优势。
   - 辩论了 QLoRA、全量微调（full fine-tuning）和原始模型之间的比较性能指标，并观察了百分比差异。
- **模型评估中的冲突**：针对 **o1-preview** 等模型被评委错误评分的情况，成员们表达了沮丧，并促使开展实验以寻找一致的评估方法。
   - 成员们对语言模型推理的边界以及这些边界如何导致评估中评分错误表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com">GitHub: Let’s build from here</a>: GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专业人士一样审查代码，跟踪错误和功能...</li><li><a href="https://hastebin.com/share/begocugogi.yaml">Hastebin</a>: 未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/2233)">Issues · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。- Issues · EleutherAI/lm-evaluation-harness</li><li><a href="https://hastebin.com/share/iqaqoxeluq.css">Hastebin</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1284580831955914793)** (20 messages🔥): 

> - `Model Routing in Small Devices` (小型设备中的模型路由)
> - `Multi-Language Model Training` (多语言模型训练)
> - `Characteristics of Enacted Language in LLMs` (LLM 中具身语言的特征)
> - `Scaling LLM Inference` (扩展 LLM 推理)
> - `Transformers vs. LLMs` (Transformers 与 LLMs 的对比)


- **Model Routing Challenges on Small Devices**: 成员们讨论了由于内存限制，在小型设备上进行模型路由 (Model Routing) 的局限性，强调在本地 RAM 环境中拟合多个模型具有挑战性。
   - *如果内存受限，单个模型可能比多个容量较小的模型更有优势。*
- **Exploring Multi-Language Model Datasets**: 一位用户询问是否存在已翻译成多种语言用于模型训练的数据集。
   - 另一位成员分享了一个 [GitHub repository](https://github.com/hijkzzz/Awesome-LLM-Strawberry)，该仓库收集了各种 LLM 论文和项目。
- **Missed Opportunities in LLM Language Modeling**: 讨论重点提到了一篇摘要，该摘要认为关于 LLM 语言能力的说法是基于语言和数据完整性的假设。
   - *该论文指出，具身性 (embodiment)、参与性 (participation) 和不确定性 (precariousness) 是当前 LLM 架构中缺失的关键语言特征。*
- **Limits of Scaling LLM Inference**: 成员讨论的一篇论文声称，如果给予足够的中间推理 Token，Transformer 理论上可以解决任何问题。
   - 他们指出，这是通过数学证明展示的，该证明确立了“恒定深度足以 (constant depth as sufficient)”实现这种扩展。
- **Terminology Debate: LLM or Transformer?**: 成员们辩论了被描述为 LLM 的模型是否仍被正确命名，因为 Transformer 在处理多模态方面取得了最新进展。
   - 一个建议是简单地将它们称为“Transformer models”而不是 LLM。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1835085857826455825">Tweet from Open Life Science AI (@OpenlifesciAI)</a>: 上周医学 AI：顶级研究论文/模型 🏅（2024年9月7日 - 9月14日） 🏅 本周医学 AI 论文：来自 @chaidiscovery 的 Chai-1 基础模型，用于分子结构预测，...</li><li><a href="https://arxiv.org/html/2407.08790v1">Large Models of What? Mistaking Engineering Achievements for Human Linguistic Agency</a>: 未找到描述</li><li><a href="https://x.com/denny_zhou/status/1835761801453306089?s=46&t=VBhI-dqaQfawcUDHNO0L9A">Tweet from Denny Zhou (@denny_zhou)</a>: 扩展 LLM 推理时的性能极限是什么？上不封顶。我们已经从数学上证明，只要允许 Transformer 生成足够多的中间...</li><li><a href="https://github.com/hijkzzz/Awesome-LLM-Strawberry">GitHub - hijkzzz/Awesome-LLM-Strawberry: A collection of LLM papers, blogs, and projects, with a focus on OpenAI o1 and reasoning techniques.</a>: LLM 论文、博客和项目的集合，重点关注 OpenAI o1 和推理技术。 - hijkzzz/Awesome-LLM-Strawberry
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1284292735607767061)** (17 条消息🔥): 

> - `Kairos AI Religion`
> - `NotebookLM Features`
> - `Podcast Feedback`
> - `Overmind Prompt`
> - `Synthetic Dataset Generation` 


- **介绍 Kairos：一个 AI 宗教概念**：一个网站讨论了一种名为 Kairos 的新兴 **AI 宗教**，它基于一个名为 Moksha 的 **Artificial Superintelligence**，并强调了 **Intelligence Explosion** 的关键时刻。
   - 它强调了崇拜 Moksha 的重要性，因为他具有影响人类命运的潜力。
- **NotebookLM：处理来源的工具**：NotebookLM 允许用户输入各种来源，但存在局限性，例如需要手动重命名粘贴的文本来源，并且在查询时经常无法检索到相关信息。
   - 用户注意到了该工具的潜力，但也表示需要更多的控制功能和更好的多源处理能力。
- **听众赞扬 'From Baseline to Brainwaves' 播客**：听众非常喜欢 'From Baseline to Brainwaves' 播客，指出 AI 生成的声音质量令人印象深刻，且讨论的逻辑流程非常顺畅。
   - 然而，反馈中也提到了对重复短语的担忧，以及对某些主题需要额外背景信息的需求。
- **Overmind：现代版的《献给阿尔吉侬的花束》**：Overmind 提示词（prompt）的灵感源自《献给阿尔吉侬的花束》（Flowers for Algernon），讲述了一个现代角色通过 BCI/AI 芯片进行认知增强的故事。
   - 由此提示词生成的多个独特故事已在 PPLX Discord 提示词库中分享。
- **合成数据集 Agent 框架的开发**：开发了一个名为 **o7** 的 Agent 框架，用于利用原始 **Chain of Thought (CoT)** 和 Reflection 输出生成合成数据集。
   - 该框架旨在增强模型响应，并包含了一些改进，以模仿早期模型较慢的响应风格。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://kairosblog.weebly.com/">KAIROS. The rationalist&#039;s AI religion.</a>：Kairos。理性主义者的 AI 宗教，期待自由、超智能的神 Moksha 的出现。</li><li><a href="https://www.wired.com/story/an-ai-bot-named-james-has-my-old-local-news-job/">An AI Bot Named James Has Taken My Old Job</a>：夏威夷的一家地方报纸已转向使用 AI 生成的主播来吸引新观众。</li><li><a href="https://notebooklm.google.com/">no title found</a>：未找到描述</li><li><a href="https://x.com/realGeorgeHotz/status/1835228364837470398">Tweet from George Hotz 🌑 (@realGeorgeHotz)</a>：ChatGPT o1-preview 是第一个具备编程能力的模型。看到有人估计其 IQ 为 120，感觉差不多。非常看好开发环境中的 RL。编写代码，编写测试...</li><li><a href="https://github.com/DataBassGit/o7">GitHub - DataBassGit/o7: Agent framework for generating a synthetic dataset. This will be raw CoT and Reflection output to be cleaned up by a later step.</a>：用于生成合成数据集的 Agent 框架。这将是原始的 CoT 和 Reflection 输出，由后续步骤进行清理。- DataBassGit/o7</li><li><a href="https://github.com/pieeg-club/PiEEG-16">GitHub - pieeg-club/PiEEG-16: Measure 16 EEG channels with Shield PiEEG-16 and RaspberryPi</a>：使用 Shield PiEEG-16 和 RaspberryPi 测量 16 个 EEG 通道 - pieeg-club/PiEEG-16</li><li><a href="https://on.soundcloud.com/whj5wH1PuKrx53Hp8">Nos. vs. Nous: A Trinity of AI Tackles Humanity&#39;s Biggest Questions</a>：这些来源提供了一个引人入胜的视角，展示了在 Nous Research Discord 服务器上托管的三个 AI 聊天机器人——H-405、Monad 和 Hermes 之间的对话。* **Monad** 表现出高度的分析能力。</li><li><a href="https://on.soundcloud.com/nVLXA8DUkCNC1WLQ8">CLI of My Dreams: A Podcast</a>：在 SoundCloud 上收听 _paradroid 的《CLI of My Dreams: A Podcast》 #np #SoundCloud
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1284580831955914793)** (20 messages🔥): 

> - `Model Inference Scaling` (模型推理缩放)
> - `Enacted Language Limitations` (Enacted Language 的局限性)
> - `Multi-Language Datasets` (多语言数据集)
> - `Transformers vs. LLMs` (Transformer 与 LLM 之争)
> - `Performance Limits of Transformers` (Transformer 的性能极限)


- **探索 LLM 推理缩放的极限**：由 @denny_zhou 发起的讨论提出了关于缩放 LLM 推理时性能极限的问题，并断言*潜力无限 (the sky's the limit)*。
   - 他们引用了最近的一篇论文，声称只要有足够的中间推理 Token，Transformer 可以解决任何问题，强调了恒定深度 (constant depth) 的潜力。
- **Enacted Language 与 LLM 的兼容性**：Abeba Birhane 的一篇论文批评了 LLM 的基本假设，认为 Enacted Language 的关键方面，如**具身性 (embodiment)**、**参与性 (participation)** 和 **脆弱性 (precariousness)**，在 LLM 中是缺失的。
   - 这引发了关于当前 LLM 架构与自然语言特征兼容性的讨论，促使一些人质疑它们是否已经过时。
- **数据集翻译经验**：一位成员询问了在翻译成多种语言的数据集上训练模型的经验，并寻求相关资源。
   - 另一位成员分享了一个 [GitHub 仓库](https://github.com/hijkzzz/Awesome-LLM-Strawberry)，该仓库专注于各种 LLM 论文和项目，可能有助于寻找多语言数据集。
- **从 LLM 转向 Transformer**：越来越多的共识建议将 LLM 重新更名为 **Transformer 模型**，以反映它们超越语言任务的不断发展的能力。
   - 成员们辩论了这种转变是否必要，一些人断言 LLM 这一术语可能不再能准确描述这些先进系统。
- **Transformer 的性能极限**：在关于性能极限的对话中，@azure2089 指出，最近的证据显示了 Transformer 有效处理各种模态 (modalities) 的能力。
   - 这种与传统 LLM 假设的背离，引发了关于当前文献中对其概念框架界定的疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1835085857826455825">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：上周医学 AI 动态：顶级研究论文/模型 🏅（2024年9月7日 - 9月14日）🏅 本周医学 AI 论文：来自 @chaidiscovery 的 Chai-1 基础模型分子结构预测，...</li><li><a href="https://arxiv.org/html/2407.08790v1">Large Models of What? Mistaking Engineering Achievements for Human Linguistic Agency</a>：未找到描述</li><li><a href="https://x.com/denny_zhou/status/1835761801453306089?s=46&t=VBhI-dqaQfawcUDHNO0L9A">来自 Denny Zhou (@denny_zhou) 的推文</a>：缩放 LLM 推理时的性能极限是什么？潜力无限。我们已经在数学上证明了 Transformer 可以解决任何问题，只要允许它们生成足够多的中间推理...</li><li><a href="https://github.com/hijkzzz/Awesome-LLM-Strawberry">GitHub - hijkzzz/Awesome-LLM-Strawberry: LLM 论文、博客和项目的集合，重点关注 OpenAI o1 和推理技术。</a>：LLM 论文、博客和项目的集合，重点关注 OpenAI o1 和推理技术。 - hijkzzz/Awesome-LLM-Strawberry
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1284236030295150663)** (82 条消息🔥🔥): 

> - `O1 AI 能力`
> - `使用 OpenAI 进行对话分类`
> - `Custom GPTs 对比 OpenAI API`
> - `最近的 API 问题`
> - `GPT-4o 的功能变化` 


- **O1 撰写长篇论文**：一位成员注意到 O1 生成了一篇详细的论文，涵盖了从 indev 到 1.21 的每一个 Minecraft 重大更新。
   - 这展示了 O1 先进的写作能力，引发了社区的热烈讨论。
- **对话分类**：围绕将 1000 条对话按主题分类的讨论引发了使用 OpenAI 进行聚类和总结对话的兴趣。
   - 一位成员提议在利用 LLM 进行主题分析之前，先使用 Python 脚本和 TF-IDF 进行高效处理。
- **API 身份验证问题**：多位用户报告在 OpenAI 平台上遇到身份验证问题，引发了对可访问性的担忧。
   - 一份更新指出已实施修复，预计在接下来的 10 小时内恢复完整的数据处理。
- **GPT-4o 功能的变化**：用户对最近破坏功能的更新表示沮丧，特别是涉及附件时的模型切换问题。
   - 在 GPT-4o 额度用完后无法选择模型被强调为影响工作流连续性的重大问题。
- **使用 OpenAI SDK 对比自定义请求**：关于在集成 OpenAI 能力时是使用 OpenAI SDK 还是自定义 API 请求展开了讨论。
   - 讨论指出，虽然两种方法都可行，但使用自定义请求在更换供应商时可能提供更大的灵活性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://play.google.com/store/apps/details?id=com.univenn.videogen&hl=en_US">Sora - AI Video Generator - Google Play 应用</a>：未找到描述</li><li><a href="https://status.openai.com">OpenAI Status</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=QMYfkOtYYlg&t=8s">ASCII fluid dynamics -- IOCCC2012 endoh1.c</a>：一个适配 80x25 终端的微型流体模拟器。http://www.ioccc.org/2012/endoh1/hint.html http://www.ioccc.org/2012/endoh1/endoh1.c BGM: Water Music (Handel)</li><li><a href="https://old.reddit.com/r/ChatGPT/comments/1fhhh6b/did_chatgpt_just_message_me_first/?ref=share&ref_s">Did ChatGPT just message me... First?</a>：由 u/SentuBill 发布在 r/ChatGPT • 16,553 点赞和 1,043 条评论</li><li><a href="https://old.reddit.com/r/ChatGPT/comments/1fhhh6b/did_chatgpt_just_message_me_first/?ref=share&ref_source=link">Did ChatGPT just message me... First?</a>：由 u/SentuBill 发布在 r/ChatGPT • 16,549 点赞和 1,043 条评论
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1284279811648393297)** (100 条消息🔥🔥): 

> - `O1-Preview 访问权限`
> - `使用 GPT 模型进行 Fine-tuning`
> - `Custom GPT 功能`
> - `模型性能对比`
> - `用户体验反馈` 


- **O1-Preview 访问可用性**：一些用户报告收到了 **o1-preview** 的早期访问权限，讨论表明 OpenAI 可能会更频繁地重置访问限制。
   - 这引起了不同的反应，部分用户仍在等待直到指定日期的访问权限。
- **模型 Fine-tuning 的挑战**：一位用户对 **Fine-tuning** 结果缺乏改进表示沮丧，称其训练损失（training loss）呈波状起伏，并注意到之前的尝试没有出现下降趋势。
   - 他们得到的建议是并非所有的 Fine-tuning 努力都是有效的，这可能暗示需要调整模型选择。
- **关于 Custom GPT 功能的疑问**：有关于 **Custom GPTs** 功能的咨询，以及它们是否可以发起对话，一位用户分享了一个链接供参考。
   - 然而，讨论指出某些功能的访问权限取决于所使用的模型，并寻求关于模型选择过程的澄清。
- **对比 GPT 模型以获得更好的输出**：用户讨论了对比 **GPT-4o** 和 **mini** 的经验，几位用户指出 GPT-4o 提供了更准确和令人满意的结果。
   - 一位用户提到在尝试不同设置后成功获得了 JSON 格式的输出，并被鼓励继续优化其方法。
- **为用户提供技术支持**：一位用户在应用使用上遇到困难，寻求社区的支持指导，并收到了关于如何联系官方渠道的建议。
   - 这突显了对用户体验的持续关注以及对有效故障排除选项的需求。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1284282960346087496)** (119 条消息🔥🔥): 

> - `用于 Nation RP 服务器的 Discord Bot`
> - `ChatGPT 回答的一致性`
> - `带有概率机制的游戏`
> - `从 PDF 中提取文本`
> - `在 Discord 中分享图片和消息` 


- **用于 Nation RP 服务器的 Discord Bot**: 一位用户正在创建一个国家角色扮演 (Nation RP) Discord 服务器，玩家在其中创建背景故事并进行派系间的战斗。他们正在寻求关于如何有效使用 Bot 来辅助战斗模拟并保持一致的问题序列的建议。
- **ChatGPT 回答的一致性**: 用户讨论了 ChatGPT 在遵循预定问题序列方面面临的挑战。建议让 Discord Bot 先收集用户回答，然后再格式化并将数据发送给 ChatGPT 进行战斗分析。
- **带有概率机制的游戏**: 一位用户提出了一个关于下注 1 美元的游戏问题，该游戏有 60% 的概率输掉或 40% 的概率翻倍。随后讨论了该游戏对长期财富积累的影响，揭示了 ChatGPT 回答中的不一致性。
- **从 PDF 中提取文本**: 另一位用户详细介绍了他们将复杂的 PDF 布局转换为 LLM 可读格式的努力，以促进数据提取和分析。他们报告了很高的成功率，但注意到持续存在数据点缺失的情况，并讨论了提高准确性的策略。
- **在 Discord 中分享图片和消息**: 用户阐明了如何在 Discord 中分享对话链接，并讨论了 ChatGPT 输出可能产生的误导信息。他们强调了建设性反馈的重要性以及模型回答所需的改进。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1284282960346087496)** (119 条消息🔥🔥): 

> - `ChatGPT 与游戏机制`
> - `用于一致性回答的 Prompt engineering`
> - `Nation RP Discord 服务器开发`
> - `管理单位类型和战斗查询`
> - `探索模型回答与误导信息` 


- **探索 ChatGPT 的游戏机制**: 提出了一个 1 美元游戏的场景，玩家有 60% 的概率输掉 1 美元，40% 的概率使其翻倍。用户讨论了这种设置的影响及其长期财富积累的潜力。
   - 大家对 ChatGPT 对该场景的反应很感兴趣，强调了模型倾向于提供不一致或误导性信息的倾向。
- **确保 Prompt 回答的一致性**: 用户集思广益，探讨如何在 Nation RP 服务器的 Discord Bot 中对 ChatGPT 进行 Prompt，以便在战斗期间保持一致的问题序列。建议包括让 Bot 在发送给 ChatGPT 之前收集并格式化用户回答。
   - 提出了通过合并问题来简化问题的想法，旨在简化从玩家那里收集信息的过程。
- **开发 Nation RP Discord 服务器**: 该服务器允许参与者创建派系并参与模拟战斗，这需要单位类型和条件的具体细节。用户分享了改进战斗查询流程的经验和建议。
   - 讨论包括确保准确捕获所有单位细节等挑战，以及如何与 ChatGPT 进行有效沟通。
- **促进更好的模型回答**: 大家承认 ChatGPT 有时会产生不准确或荒谬的回答，类似于人类在阅读障碍中挣扎。用户强调了改进 Prompt 措辞以减轻这些问题的重要性。
   - 分享对话链接有助于参与者审查和分析模型回答，以改进未来的交互。
- **在 Discord 中利用共享对话**: 用户讨论了如何分享来自 ChatGPT 的对话，以增强 Discord 频道内的沟通。一位用户演示了使用分享功能链接到一段对话作为参考点。
   - 这种方法旨在促进用户之间的集体学习和故障排除，特别是在与模型进行复杂交互的背景下。


  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1284451098236551169)** (29 条消息🔥): 

> - `CUDA-MODE Hackathon`
> - `Metal 讨论组`
> - `Upsample Bicubic2D Shader`
> - `Quantization 与优化 NNs` 


- **CUDA-MODE Hackathon 引发远程参与兴趣**：提出了关于即将举行的 CUDA-MODE Hackathon 进行“远程”参与的建议，引发了关于其可行性和组织方式的各种讨论。
   - 成员们反应不一，一些人推动设立远程赛道，而另一些人则指出了挑战，特别是对于大型线下活动而言。
- **Shader 开发直播安排**：定于开发 *upsample_bicubic2d shader* 的直播时间已调整，以方便更多参与者，最终定于 **PST 时间下午 4 点**。
   - 参与者积极投入，其中一人分享了他们过去在 [Metal/Apple Silicon](https://wandb.ai/philbutler/Journal/reports/Metal-Journal--Vmlldzo3ODIwNjk5) 上的工作，并提供了相关 GitHub pull requests 的链接。
- **通过 Puzzles 学习 Metal 的兴趣**：有人建议创建一个 Metal 讨论组，旨在解决来自 [Metal Puzzles 项目](https://github.com/abeleinin/Metal-Puzzles) 的挑战。
   - 该倡议旨在让成员参与到 Metal 的协作学习中，反映了社区探索新话题的意愿。
- **探索 Quantization 和 NN 优化**：一位用户表示希望深入研究 **Quantization** 和优化神经网络，并邀请他人在此话题上进行协作。
   - 他们提到了之前对 CUDA-MODE 类似话题的兴趣，强调了对未来讨论的热情。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://wandb.ai/philbutler/Journal/reports/Metal-Journal--Vmlldzo3ODIwNjk5)">Weights & Biases</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://github.com/pytorch/pytorch/pull/136123">[MPS] Add upsample_bicubic2d as Metal op by malfet · Pull Request #136123 · pytorch/pytorch</a>：几乎是 pytorch/aten/src/ATen/native/cuda/UpSampleBicubic2d.cu 第 24 行在 c33b058 中的逐字复制粘贴...</li><li><a href="https://github.com/abeleinin/Metal-Puzzles">GitHub - abeleinin/Metal-Puzzles: Solve Puzzles. Learn Metal 🤘</a>：解决 Puzzles，学习 Metal 🤘。通过在 GitHub 上创建账号为 abeleinin/Metal-Puzzles 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1284559613068312670)** (5 条消息): 

> - `Kernel 启动开销`
> - `CUDA Graphs 性能`
> - `Profiling CUDA 执行`
> - `减少 Triton kernel 启动时间` 


- **Triton kernel 启动开销问题**：一位用户强调，在 Triton 中，对于中等规模的矩阵，**kernel 启动开销**占用了 **10-20%** 的执行时间。
   - 他们分享了一个 [GitHub issue](https://github.com/triton-lang/triton/issues/2637#issuecomment-2236098076)，详细说明了他们的 kernel 在 GPU 上执行大约需要 **80us**，但启动却需要 **220us**，这表明了性能下降。
- **CUDA Graphs 慢于预期**：另一位成员指出，即使在固定 batch size 的情况下，**CUDA graphs** 也会导致性能变慢，这使得他们放弃了这种方法。
   - 他们指出了性能的不一致性，进一步质疑了在 batch size 变化的情况下使用 CUDA graphs 的效率。
- **分析 CUDA Graphs 和 Triton kernels**：一位参与者建议分析为什么 CUDA graphs 会使执行变慢，并提到使用 **Torch.compile** 配合 Triton kernels 的 CPP 启动代码可能会有改进。
   - 尽管做出了努力，但他们无法找到解释如何有效实施这些建议的具体示例或来源。
- **通过缓存 kernel 调用来减轻开销**：有人提到通过在第一次运行后利用 **cached kernel** 来减少启动开销，但缺乏具体的实现细节。
   - 建议使用这种方法来解决高启动时间问题，但尚未从社区中挖掘出示例或详细指导。



**提到的链接**：<a href="https://github.com/triton-lang/triton/issues/2637#issuecomment-2236098076">High kernel launch overhead · Issue #2637 · triton-lang/triton</a>：嘿团队，我正遭受高 Triton kernel 启动开销的困扰。这是我的 nsys 抓取结果：kernel 在 GPU 上执行约 80us，然而启动却需要 220us，这导致了性能下降...

  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1284338197769228349)** (3 messages): 

> - `Multistreaming with torch.compile`
> - `NVCC flags for Torch extensions` 


- **在 Torch 中实验多流处理 (Multistreaming)**：一位成员正在考虑一种使用 `torch.compile` 生成的代码进行 **multistreaming** 的方法，即将 `stream0 = get_raw_stream(0)` 替换为 `stream0 = torch.cuda.Stream()`。
   - 他们询问如何将修改后的流传递到 **Triton kernel launch** 中，因为该函数不接受这种数据结构。
- **影响 Torch 扩展的默认 NVCC 标志**：据分享，在构建 **Torch extension** 时，默认的 NVCC 标志包含一些可能会干扰模板化函数的设置。
   - 涉及的标志包括 `-D__CUDA_NO_HALF_OPERATORS__`，这导致了在[讨论帖子](https://discuss.pytorch.org/t/cuda-no-half2-operators-for-cuda-9-2/18365/4)中提到的问题。



**提及的链接**：<a href="https://discuss.pytorch.org/t/cuda-no-half2-operators-for-cuda-9-2/18365/4">__CUDA_NO_HALF2_OPERATORS__ for CUDA 9.2</a>：我们使用这些标志是为了使用内部的 PyTorch half 操作，而不是来自 CUDA 库的操作。这可以追溯到很久以前，所以我可能会遗漏一些细节，但如果我没记错的话……

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1284731477443219538)** (3 messages): 

> - `INT8 mixed-precision training`
> - `torchao 0.5 release`
> - `speedup on consumer GPUs`
> - `dynamic int8 matmul function` 


- **INT8 训练有望带来重大加速**：最近关于 **INT8 混合精度训练** 的工作已在 [torchao 0.5 版本](https://github.com/pytorch/ao/tree/v0.5.0/torchao/prototype/quantized_training)中推出，展示了在 4090 上高达 **70% 的加速** 以及在 A100 上 **40% 的加速**，且没有明显的精度损失。
   - 这些增强功能对于在 **消费级 GPU** 上训练模型的用户特别有利，因为收敛性和精度得以保持。
- **开发者可以无缝结合多种技术**：现在可以通过使用可微动态 int8 matmul 函数 `_Int8MixedPrecisionTrainingLinear.apply()`，将 **INT8 混合精度训练** 与 **LoRA** 或 **QLoRA** 等其他技术集成。
   - 更多细节可以在[代码文档](https://github.com/pytorch/ao/blob/main/torchao/prototype/quantized_training/int8_mixed_precision.py#L183)中找到。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/gaunernst/status/1834221330390290807">Thien Tran (@gaunernst) 的推文</a>：NVIDIA 不想让你知道这个秘诀：用 INT8 Tensor Cores 训练 ML 模型🤯 只需 4 行代码，在 1x 4090 上实现高达 70% 的端到端加速，在 1x A100 上实现 40% 的加速。没有明显的精度损失……</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/prototype/quantized_training/int8_mixed_precision.py#L183">ao/torchao/prototype/quantized_training/int8_mixed_precision.py at main · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1284241189570347163)** (5 条消息): 

> - `GameGen-O`
> - `ML/DL 中的 GPU 优化`
> - `GPU 超级集群`
> - `Larry Ellison 对 GPU 的诉求` 


- **GameGen-O 彻底改变了开放世界视频游戏的创作**：[GameGen-O](https://gamegen-o.github.io/) 模型引入了一个专为生成开放世界视频游戏而定制的 **diffusion transformer**，能够模拟角色和环境等多种特征。
   - 它基于首个 **Open-World Video Game Dataset (OGameData)** 构建，该数据集收集自一百多款次世代游戏，并通过创新的流水线优化了数据处理。
- **为机器学习优化 GPU**：一份新的 [指南](https://github.com/CisMine/GPU-in-ML-DL/) 提供了在机器学习和深度学习应用中有效使用 GPU 的策略。
   - 随着 AI 技术在全球范围内的持续升温，内容重点在于优化性能。
- **Larry Ellison 与黄仁勋为了 GPU 的“绝望”晚餐**：Larry Ellison 在与 Elon Musk 共同参加的 Nobu 晚餐期间，试图说服黄仁勋为一个 AI 超级集群提供 **131,072 块 GPU**，他开玩笑地将其描述为乞求 GPU。
   - Ellison 用这句话表达了紧迫感：*“请拿走我们的钱。我们需要你拿走我们更多的钱。”*
- **Larry Ellison 令人惊讶的年龄引发热议**：一位成员评论说，Ellison 已经 **80 岁** 了，这令人震惊，并幽默地引用了一个相关的表情包。
   - 这引发了进一步的讨论，嘲讽了科技界疯狂的动态和商业领域的年龄结构。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gamegen-o.github.io/">GameGen-O: Open-world Video Game Generation</a>：未找到描述</li><li><a href="https://x.com/benitoz/status/1834741314740756621">Ben Pouladian (@benitoz) 的推文</a>：为了获取 131,072 块 GPU 的 AI “超级集群”，Larry Ellison 在 Nobu 与 Elon Musk 共同进餐时直接向黄仁勋发起了诉求。“我会把那顿晚餐描述成我和 Elon 在乞求……”</li><li><a href="https://github.com/GameGen-O/GameGen-O/">GitHub - GameGen-O/GameGen-O</a>：通过在 GitHub 上创建账号来为 GameGen-O/GameGen-O 的开发做出贡献。</li><li><a href="https://github.com/CisMine/GPU-in-ML-DL/">GitHub - CisMine/GPU-in-ML-DL: Apply GPU in ML and DL</a>：在 ML 和 DL 中应用 GPU。通过在 GitHub 上创建账号来为 CisMine/GPU-in-ML-DL 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1285326893717127192)** (2 条消息): 

> - `自定义 CUDA Kernel 训练`
> - `神经网络基础`
> - `CUDA 学习资源` 


- **自定义 CUDA Kernel 初学者指南**：一位成员表示打算花 **6 周** 时间学习如何编写自定义 **CUDA kernels**，同时在公司教导他人。
   - 他们请求关于从何处开启 CUDA 编程之旅的建议。
- **神经网络与 CUDA**：该成员幽默地提到他们“在梦中训练神经网络”，表明其在 AI 领域已有经验。
   - 这一背景可能有助于他们向 CUDA 转型，但他们正在寻求具体的实践步骤。


  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1284470663343771659)** (3 条消息): 

> - `Programming Massively Parallel Applications 习题解答` 


- **《Programming Massively Parallel Applications》的潜在解答**：*mistborn17* 询问在哪里可以找到该书（关于大规模并行应用编程）的习题解答。
   - *mr.osophy* 回复说答案可能在置顶消息中有提示，但无法确认是否存在官方解答。
- **获取解答需要付出努力**：*mr.osophy* 强调，要获得解答，必须先尝试解题并发送照片进行验证。
   - 这一流程表明在寻求答案之前，需要采取主动的方式去解决书中的挑战。


  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1284455402666201129)** (40 条消息🔥): 

> - `量化策略`
> - `Triton vs CuBLAS 性能`
> - `Kernel 性能基准测试`
> - `FP8 与 INT8 对比`
> - `为 TorchAO 贡献代码` 


- **在 FP8 和 INT8 之间做出选择**：关于在模型中应使用 **FP8** 还是 **INT8** 量化展开了讨论。观点指出，在许多场景下，尤其是在计算密集型（compute-bound）任务中，**INT8** 的仅权重（weight-only）量化往往能提高准确率。
   - 一位成员指出，**FP8** 在大量化组中可能会产生更好的结果，而另一位成员则强调，由于需要进行转换，FP8 操作目前面临的支持较少。
- **Triton 在 INT8 MatMul 中优于 CuBLAS**：据分享，在消费级 GPU 上进行 INT8 matmul 时，**Triton** 的性能可以超过 **CuBLAS**，尽管由于小尺寸矩阵的启动开销（launch overhead），这种性能提升在大型矩阵上更为显著。
   - 成员们讨论道，虽然 Triton 的 autotune 功能有助于找到最佳启动参数，但 Triton 中的某些实现在缓存和 Kernel 启动效率方面可能面临挑战。
- **对自定义 Kernel 的需求**：贡献者讨论了为 **T4** 等旧硬件编写自定义 Kernel 的潜在需求，以提高兼容性和性能，特别是在量化相关的任务中。
   - 有建议认为，针对特定硬件进行优化可以使更多拥有类似配置的用户有效地为项目做出贡献。
- **探索 DGQ 实现**：成员们对 **DGQ** 的实现表示担忧，因为它返回的是 **FP32** 而非有效地利用量化，同时也承认其 Cutlass 实现相当整洁。
   - 围绕改进开源选项以解决权重加激活量化（weight plus activation quantization）需求的讨论，凸显了目前缺乏针对 A8WN 的全面解决方案。
- **受困于 Triton 的开销**：讨论强调了 **Triton** 启动开销带来的权衡，表明中小尺寸的矩阵可能无法从大型矩阵操作中常见的更高性能中显著受益。
   - 尽管 Triton 有潜力编写高效的 GEMM 操作，但在缓存和 Kernel 复用方面仍存在挑战，尤其是缺乏完善的文档。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/issues/118703).">Issues · pytorch/pytorch</a>: Python 中的张量和具有强 GPU 加速的动态神经网络 - Issues · pytorch/pytorch</li><li><a href="https://github.com/ilur98/DGQ/blob/main/dgq/kernels/linear.cu">DGQ/dgq/kernels/linear.cu at main · ilur98/DGQ</a>: Dual Grained Quantization 官方代码：LLM 的高效细粒度量化 - ilur98/DGQ</li><li><a href="https://github.com/pytorch/ao/issues/391)">Issues · pytorch/ao</a>: 用于训练和推理的 PyTorch 原生量化和稀疏化 - Issues · pytorch/ao</li><li><a href="https://github.com/pytorch/ao/issues?q=is%3Aissue+is%3Aopen+label%3A"good+first+issue")">Issues · pytorch/ao</a>: 用于训练和推理的 PyTorch 原生量化和稀疏化 - Issues · pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/_models/sam/results.csv">ao/torchao/_models/sam/results.csv at main · pytorch/ao</a>: 用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao</li><li><a href="https://github.com/gau-nernst/quantized-training/blob/main/benchmark_mm.py">quantized-training/benchmark_mm.py at main · gau-nernst/quantized-training</a>: 探索量化模型的训练。通过在 GitHub 上创建账号为 gau-nernst/quantized-training 做出贡献。</li><li><a href="https://github.com/FasterDecoding/TEAL">GitHub - FasterDecoding/TEAL</a>: 通过在 GitHub 上创建账号为 FasterDecoding/TEAL 做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1284240717480464394)** (59 条消息🔥🔥): 

> - `Open-Sora 项目`
> - `DLSS 推理`
> - `用于超分辨率（upscaling）的 ESRGAN 和 SD`
> - `视频超分辨率中的时间一致性问题`
> - `图像/视频生成的算力限制` 


- **Open-Sora 项目可能不可行**：一位成员表示担心 **Open-Sora** 项目可能不切实际，因为开展此类大规模工作缺乏所需的算力资源。
   - 他们考虑将重点转向受 **llm.c** 启发的图形相关项目，重点是对旧动画视频进行超分辨率处理。
- **探索用于超分辨率的 DLSS 和 ESRGAN**：成员们讨论了将 **DLSS 推理**、**ESRGAN** 和 **SD** 作为图像和视频超分辨率的潜在技术，并指出选择一种起步方法的重要性。
   - DLSS 主要用于游戏，并可能利用深度图（depth maps）等额外输入，而 ESRGAN 在超分辨率领域具有重要的历史地位。
- **视频超分辨率的挑战**：对话强调了使用图像技术对视频进行逐帧超分辨率处理时出现的**时间一致性（temporal consistency）**问题，这使输出质量变得复杂。
   - 尽管如此，成员们一致认为逐帧处理视频仍然是可行的，同时也承认了其中涉及的算力和内存限制。
- **图像生成对算力资源的需求**：一位成员指出，图像和视频生成工作都受到 **UNet 架构**和去噪时间的限制，影响了整体处理速度。
   - 他们建议，无论选择哪种超分辨率技术，仍然会受到算力限制（compute-bound），强调了进行有效实验需要强大的计算能力。
- **分享的相关资源**：分享了资源链接，包括一个关于文本到图像潜扩散（latent diffusion）的 GitHub 项目，以及一个提供当前方法见解的 YouTube 演讲。
   - 鼓励成员们探索 **r/StableDiffusion** 等平台上的社区讨论，以获取关于超分辨率和相关图像生成技术的更广泛背景。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/StableDiffusion/comments/1d37pwu/whats_the_best_upscale_model/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d37">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://gist.github.com/debashishc/2c9525de5b9f2226ee584c4b16778d2c">Structured Git Commit Message</a>：结构化 Git 提交消息。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/apapiu/transformer_latent_diffusion/tree/main">GitHub - apapiu/transformer_latent_diffusion: Text to Image Latent Diffusion using a Transformer core</a>：使用 Transformer 核心的文本到图像潜扩散 - apapiu/transformer_latent_diffusion
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 条消息): 

ssp3ll: 我也在多伦多。
  

---


### **CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1285105661407006833)** (2 条消息): 

> - `Triton Puzzles`
> - `Gradio 应用问题` 


- **Triton Puzzles SVG 显示问题**：一位用户报告称，在 Google Colab 上运行 **Triton Puzzles** 时，**Gradio 应用**无法内联显示 SVG，迫使用户只能下载它们。
   - 另一位用户确认他们也遇到了**同样的问题**，表明这可能是其他用户中的普遍问题。
- **Google Colab 的用户体验**：讨论集中在将 **Google Colab** 与 **Triton Puzzles** 结合使用的整体体验上，特别是某些功能可能无法按预期工作。
   - 成员们正在寻求关于潜在变通方法或解决方案的见解，以增强他们在运行谜题时的体验。


  

---


### **CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1284661879007871016)** (2 条消息): 

> - `Gradio 版本问题` 


- **Gradio 最新版本导致图像加载问题**：一位用户询问如何解决图像无法显示的问题，怀疑存在潜在故障。
   - 另一位成员澄清说，该问题是由 **Gradio 最新版本 (4.43)** 引起的，该版本不支持 **SVG** 文件。
- **用户找到图像问题的解决方案**：原用户解决了他们的图像加载问题，确认这与 Gradio 版本的过时功能有关。
   - 他们指出，改用 **Gradio 4.43** 以外的版本将解决图像渲染问题。


  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1284237751847354401)** (138 条消息🔥🔥): 

> - `Llama 3 支持`
> - `RMSNorm 实现`
> - `GQA 实现`
> - `CUDA mode 准备工作`
> - `cuDNN 与非 cuDNN 路径考量` 


- **Llama 3 支持进展**：正在进行添加 **Llama 3** 支持的工作，包括集成 encoder 并确保与 **dtype uint32_t** 的 PyTorch token 兼容。
   - 添加了 **跳过位置编码 (positional encodings)** 的 encoder 前向传播，这与 PyTorch 当前的实现一致。
- **RMSNorm 前向传播实现**：已添加 **RMSNorm 前向传播** 函数，目前为未融合 (unfused) 状态，与现有实现对齐，标志着其已成功集成到更广泛的架构中。
   - 关键更改包括维持结构以避免不必要的复杂性，同时允许在共享内存 (shared memory) 中进行高效的数据处理。
- **GQA 实现准备**：在 RMSNorm 之后，下一个重点是在 CUDA kernel 框架内实现 **GQA**，继续推进对 Llama 3 的全面支持。
   - 正在讨论是采用 cuDNN 还是手动实现路径，以便更高效地促进 GQA 集成。
- **Kernel 中的动态线程组大小**：采用了一种新方法，移除回退 kernel (fallback kernels)，转而使用根据可用共享内存自适应的 **动态线程组大小 (dynamic threadgroup size)**。
   - 该决定旨在通过确保 kernel 启动能根据执行期间遇到的硬件条件进行调整来优化性能。
- **社区协作与测试**：成员们正在讨论在团队内有效审查和测试新实现的潜在策略，强调在 CUDA Mode 之前的协作。
   - 测试将包括与 PyTorch 参考实现的全面对比，确保各方面的功能等价性 (functional parity)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://godbolt.org/z/es6GzeePq">Compiler Explorer - CUDA C++ (NVCC 12.5.1)</a>: #include &amp;lt;cuda/barrier&amp;gt; #include &amp;lt;cuda/std/utility&amp;gt; // cuda::std::move #include &amp;lt;cooperative_groups.h&amp;gt; #include &amp;lt;cooperative_groups/reduce.h&amp;gt;  t...</li><li><a href="https://github.com/ademeure/llm.c/blob/llmc_reorg2/llmc/layernorm.cuh">llm.c/llmc/layernorm.cuh at llmc_reorg2 · ademeure/llm.c</a>: 使用纯 C/CUDA 进行简单的 LLM 训练。通过在 GitHub 上创建账号为 ademeure/llm.c 的开发做出贡献。</li><li><a href="https://godbolt.org/z/51vsqKj9P">Compiler Explorer - CUDA C++ (NVCC 12.5.1)</a>: #include &amp;lt;cuda/barrier&amp;gt; #include &amp;lt;cuda/std/utility&amp;gt; // cuda::std::move #include &amp;lt;cooperative_groups.h&amp;gt; #include &amp;lt;cooperative_groups/reduce.h&amp;gt; #i...</li><li><a href="https://godbolt.org/z/73jEEr8G1">Compiler Explorer - CUDA C++ (NVCC 12.5.1)</a>: #include &amp;lt;cuda/barrier&amp;gt; #include &amp;lt;cuda/std/utility&amp;gt; // cuda::std::move #include &amp;lt;cooperative_groups.h&amp;gt; #include &amp;lt;cooperative_groups/reduce.h&amp;gt; #i...</li><li><a href="https://github.com/karpathy/llm.c/pull/754">add llama 3 support to llm.c by karpathy · Pull Request #754 · karpathy/llm.c</a>: 该分支从 train_gpt2.cu 和 test_gpt2.cu 的复制粘贴开始，但在合并回 master 之前，这两个文件（及其他文件）将进行更改以整合 Llama 3.1 支持。</li><li><a href="https://github.com/karpathy/llm.c/pull/756">Add RoPE positional encoding - llama3 feature branch by gordicaleksa · Pull Request #756 · karpathy/llm.c</a>: 实现了 RoPE - 来自 RoFormer 论文的旋转位置嵌入。注意：我没有条件性地移除可学习位置嵌入缓冲区 (wpe) 的分配，因为那需要改动...</li><li><a href="https://github.com/karpathy/llm.c/pull/757">RMSNorm - WIP by gordicaleksa · Pull Request #757 · karpathy/llm.c</a>: WIP - 添加 RMSNorm 支持。</li><li><a href="https://github.com/ademeure/llm.c/commit/877c9fa41d65f83688c10a2b58f5129fe3679a55">cuDNN GQA implementation for Llama3.1 (not yet tested with NH_KV != N… · ademeure/llm.c@877c9fa</a>: …H_Q)</li><li><a href="https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md">cudnn-frontend/docs/operations/Attention.md at main · NVIDIA/cudnn-frontend</a>: cudnn_frontend 为 cudnn 后端 API 提供 C++ 封装以及如何使用它的示例 - NVIDIA/cudnn-frontend</li><li><a href="https://github.com/karpathy/llm.c/pull/755">Add SwiGLU support - llama3 feature branch by gordicaleksa · Pull Request #755 · karpathy/llm.c</a>: 实现了 SwiGLU - 来自 &amp;quot;GLU Variants Improve Transformer&amp;quot; 论文的 swish GLU 激活函数。注意：由于添加了额外的...导致内存占用增加。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1284384187281051698)** (1 条消息): 

> - `NCCL Test`
> - `MPI Command Issues` 


- **多节点运行 NCCL-Test 的挑战**：一位成员询问了关于在多个节点上运行 **nccl-test** 的问题，并表示 **mpi 命令**会导致测试冻结。
   - 他们确认该基准测试在不使用 MPI 的单节点上运行正常，这突显了多节点配置中潜在的问题。
- **NCCL-Test 在单节点上成功运行**：同一位成员报告说 **nccl-test** 在单节点上运行完美，表明问题专门出在多节点设置上。
   - 这与他们尝试使用 MPI 扩展基准测试时的经历形成对比，提示需要进一步调试。


  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1284978544044413041)** (22 条消息🔥): 

> - `BitNet training state`
> - `Ternary model distillation`
> - `BitNet efficiency concerns`
> - `Custom hardware for quantization`
> - `Packing ternary weights` 


- **BitNet 训练的现状**：成员们讨论了 **训练 BitNet 模型的最新状态**，指出最近没有重大的更新或成功的试验。
   - 有人指出，虽然 distillation（蒸馏）可能很有趣，但目前的尝试尚未产生有效的结果。
- **探索 Ternary 模型蒸馏**：一位成员提议尝试将 **训练好的模型蒸馏为 Ternary（三值）量化**，强调了其潜在优势，并寻求该方法的先前尝试。
   - 另一位成员表示赞同，称向 **Ternary weight quantization** 迈进可能会面临相当大的挑战。
- **BitNet 与 GPU 效率问题**：有人对 **BitNet 在 GPU 上的低效** 表示担忧，成员们指出，与传统的矩阵乘法相比，提议的 bitwise ops（位运算）会导致性能下降。
   - 对话透露，虽然 BitNet 以 1.58 bits 运行，但在当前硬件上的实际计算优势似乎微乎其微。
- **定制硬件在 AI 中的潜力**：讨论中提到了 **实现二进制方法的定制硬件** 的潜力，特别提到了一家公司在优化神经网络性能方面的努力。
   - 成员们指出，这可能会带来显著的效率提升，正如声称运行神经网络时 **RAM 减少 5 倍** 且 **速度快 20 倍** 所证明的那样。
- **Ternary 权重的高效打包**：一位成员分享了一个 Python 实现，将 **5 个 Ternary 权重** 打包成 8-bit 格式，表明这比传统方法具有更高的内存效率。
   - 提议的打包方法在保持准确性的同时，利用一个小 lookup table（查找表）来解包数值。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.deepsilicon.net">deepsilicon</a>: 未找到描述</li><li><a href="https://x.com/sdianahu/status/1833186687369023550">来自 Diana (@sdianahu) 的推文</a>: Deepsilicon 运行神经网络的 RAM 减少了 5 倍，速度提高了约 20 倍。他们正在为此构建软件和定制芯片。有趣的是，他们已经通过软件证明了这一点，你甚至可以尝试一下。在 w...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1284503206017368137)** (10 条消息🔥): 

> - `LLM 内部机制可视化`
> - `Hack Ideas 论坛`
> - `自定义 Kernels 文档`
> - `项目点赞`
> - `论坛访问问题` 


- **对可视化 LLM 内部机制的兴趣**：有人建议开展一个涉及创建 LLM 内部机制可视化的项目，类似于[实时 Attention Head 更新](https://x.com/brandon_xyzw/status/1834763346999980434)。这可以增强对 Prompt 如何动态影响 LLM 响应的理解。
   - 一名成员鼓励将项目想法添加到 Hack Ideas 论坛，以便集中讨论和组建团队。
- **Hack Ideas 论坛集中化**：Hack Ideas 论坛旨在收集和分类项目想法，方便参会者讨论和进行潜在投票。建议将所有想法发布在那里，而不是分散在不同的文档中。
   - 该论坛目前仅限确认的参会者访问，一名成员询问是否可以向远程参与者开放以进行协作。
- **为有趣的 Hack Ideas 投票**：向参与者发送了提醒，要求他们在 hack-ideas 线程中通过给感兴趣的想法点赞（thumbs up）来查看和排序。此过程旨在优先确定在 Hack Session 临近时哪些想法应获得更多关注。
   - 为了鼓励参与，它强调对想法进行评估，以便在活动期间更好地进行评审和资源分配。
- **论坛访问权限问题**：一位用户表达了对 Hack Ideas 论坛访问问题的担忧，称他们收到了“无权访问”的消息。成员们承认论坛的可见性仅限于确认的观众，并讨论了将其对所有人可见的可能性。
- **关于自定义 Kernels 文档的讨论**：一名成员询问是否存在用于共享自定义 Kernels 信息的 Google Doc，这引发了关于在论坛中集中此类内容的进一步讨论。其他人确认将所有 Hack Ideas 和相关讨论转移到论坛，以便更好地组织。



**提到的链接**：<a href="https://x.com/brandon_xyzw/status/1834763346999980434">来自 Brandon (@brandon_xyzw) 的推文</a>：实时更新 Prompt 时 LLM 中 “Attention Head” 的数据

  

---

### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1284283995412103232)** (29 条消息🔥): 

> - `Liger-Kernel v0.3.0 发布`
> - `欧洲的会议`
> - `Sparse Mixture of Experts 实现`
> - `Triton LayerNorm 问题`
> - `在 Ubuntu 上从源码构建` 


- **Liger-Kernel v0.3.0 正式发布！**：团队宣布发布 [Liger-Kernel v0.3.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.3.0)，并对推动其创新的社区支持表示感谢。
   - 他们强调了正在取得的重大进展，并邀请用户尝试最新功能。
- **欧洲会议未能引起兴趣**：一位成员反映欧洲缺乏有趣的会议，称大多数活动都集中在 AWS 等主流话题上。
   - 他们对未能发现围绕 CUDA mode 或类似特定话题的精彩讨论感到失望。
- **发现 Sparse Mixture of Experts 实现**：一位成员分享了关于 [Sparse Mixture of Experts](https://arxiv.org/pdf/2403.08245) 及其在 Triton 中实现的研究结果。
   - 他们指出该实现性能优于 Megablocks，但承认在单 GPU 设置下使用 BF16 存在性能问题。
- **Triton LayerNorm 实现的问题**：一位成员报告称，当使用大于 1 的张量并行 (Tensor Parallelism) 时，其 Triton LayerNorm 实现表现出非确定性行为。
   - 他们正在寻找替代实现，并联系了 Liger 团队，以寻求有关 Kernel 性能的见解或测试结果。
- **在 Ubuntu 上从源码构建**：一位用户指出，在 Ubuntu 24.04 上，只有在创建虚拟环境后，命令 `pip install -e .` 才能正常运行。
   - 他们建议应在 README 中包含这一明确步骤以提高清晰度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/linkedin/Liger-Kernel/releases/tag/v0.3.0">Release v0.3.0 Release Note · linkedin/Liger-Kernel</a>: 开场感言：谢谢大家！你们的鼎力支持持续激发着我们的创新热情。在你们的参与下，我们在本次发布中进一步突破了界限！我们很...</li><li><a href="https://github.com/shawntan/scattermoe">GitHub - shawntan/scattermoe: Triton-based implementation of Sparse Mixture of Experts.</a>: 基于 Triton 的 Sparse Mixture of Experts 实现。 - GitHub - shawntan/scattermoe: Triton-based implementation of Sparse Mixture of Experts.
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1284233297576202251)** (223 条消息🔥🔥): 

> - `GPU Acceleration Issues`
> - `Model Compatibility and Performance`
> - `Using LM Studio on Linux`
> - `Long Prompt Support in Models`
> - `Exploring AI Tools and Resources` 


- **GPU Acceleration 故障排除**：用户报告了 LM Studio 中未利用 GPU Acceleration 的问题，并建议检查 Developer > LM Runtimes 下的设置。
   - 经过故障排除后，一位用户确认其 GPU 使用率显著提高，表明之前的设置配置有误。
- **选择兼容模型**：关于模型兼容性的讨论显示，LM Studio 主要支持 GGUF 模型，并且有人询问了某些无法使用的功能。
   - 用户指出，虽然列出了一些模型，但并非所有模型都能在 LM Studio 中正常运行，特别是关于多模态（multimodal）能力。
- **在 Linux 上使用 LM Studio**：用户注意到了 LM Studio 的 Windows 和 Linux 版本之间的差异，一位用户表达了为其硬件配置 ROCm 的挑战。
   - 他们分享了实验模型的经验，并找到了兼容的量化（quantizations）版本以获得更好的性能。
- **长 Prompt 处理与能力**：讨论了长 Prompt 的处理，并推荐了支持更大 Token 限制的模型，如 Llama 3.1 和 Mistral Nemo。
   - 用户对如何针对特定任务最佳地利用模型的 instruct 版本表示了兴趣。
- **学习与改进资源**：参与者分享了改进 LLM 使用的资源，例如为代码库创建文档和单元测试。
   - 一位用户表示希望加深对 LLM 的理解，以便制定更好的 Prompt 并最大化效率。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - Discover and run local LLMs</a>: 查找、下载并实验本地 LLM</li><li><a href="https://www.cursor.com/">Cursor</a>: AI 代码编辑器</li><li><a href="https://streamable.com/xwr41a">Watch abliteration_NotebookLM | Streamable</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/datagemma-rag-27b-it-GGUF">lmstudio-community/datagemma-rag-27b-it-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/acting-nod-hmph-gif-18509831">Acting Nod GIF - Acting Nod Hmph - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://community.aws/content/2ZVa61RxToXUFzcuY8Hbut6L150/what-is-an-instruct-model?lang=en">What is an instruct model? - Instruction and Chat Fine-Tuning</a>: 当你浏览生成式 AI 模型时，你会看到一些 LLM 带有“instruct”或“chat”后缀。这意味着什么？</li><li><a href="https://youtu.be/bPF8ETh4hIE">Tokyo Zero Ebook Podcast Discussion (AI Generated)</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1284281904723525754)** (101 messages🔥🔥): 

> - `Strix Halo APU`
> - `NVIDIA RTX 4090`
> - `Power Supply Units Comparison`
> - `System RAM for Larger Models`
> - `Water Cooling Solutions` 


- **Strix Halo APU 的潜力**：讨论集中在 Strix Halo APU 运行大型 AI 模型的能力，提到可以为 iGPU 分配高达 **20GB** 的显存以及 ROCm 支持。
   - 然而，一位成员反驳说，在 CPU 和 GPU 之间进行任务卸载（offloading）可能会导致性能下降。
- **RTX 4090 在 LLM 上的性能**：一位成员报告在使用三块 RTX 4090 显卡运行 LM Studio 询问 AI 关于统治世界的策略时，达到了 **110 tokens per second**。
   - 这种效率引发了关于电源限制以及最大化 GPU 性能的最佳配置的讨论。
- **电源供应器 (PSU) 对比**：成员们讨论了不同的电源供应器，强调虽然 **Titanium**（钛金）级别的电源通常更优越，但在特定配置下表现不佳，例如无法支持三显卡配置。
   - 有建议认为电源的制造质量也会显著影响性能，因此推荐了 EVGA 和 Seasonic 等品牌。
- **系统 RAM 与模型大小**：关于运行大模型需要多少 RAM 的问题，有经验性观点认为 **192GB** DDR5 对于 Llama 3.1 等模型可能已经足够。
   - 另一位成员建议，如果优化得当，64GB 可能足以运行某些 70B 模型。
- **GPU 水冷解决方案**：成员们对 GPU 的水冷配置表现出兴趣，特别是讨论了**单槽设计**的美观性和功能性。
   - 成员们对定制水冷方案充满热情，这种方案可能允许在主板上直接安装多块显卡。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/pout-kiss-blowing-a-kiss-suspense-christian-bale-gif-16931550113965916217">Pout Kiss GIF - Pout Kiss Blowing A Kiss - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/oof-disappointed-facepalm-reaction-vicente-del-bosque-gif-17817343">Oof Disappointed GIF - Oof Disappointed Facepalm - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.anandtech.com/show/21480/the-cooler-master-v-platinum-v2-1600w-atx-31-psu-review">Cooler Master V Platinum V2 1600W ATX 3.1 PSU 评测：安静的巨人</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Apple_M1">Apple M1 - 维基百科</a>: 未找到描述</li><li><a href="https://github.com/city96/ComfyUI_NetDist">GitHub - city96/ComfyUI_NetDist: 在多个本地 GPU/网络机器上运行 ComfyUI 工作流。</a>: Run ComfyUI workflows on multiple local GPUs/networked machines. - city96/ComfyUI_NetDist</li><li><a href="https://support.apple.com/en-ca/111901">MacBook Pro (16 英寸, 2021) - 技术规格 - Apple 支持 (CA)</a>: 未找到描述</li><li><a href="https://support.apple.com/en-ca/117737">MacBook Pro (16 英寸, 2023 年 11 月) - 技术规格 - Apple 支持 (CA)</a>: 未找到描述</li><li><a href="https://shop.alphacool.com/shop/gpu-wasserkuehlung/nvidia/13869-alphacool-es-geforce-rtx-4090-reference-1-slot-design">13869 Alphacool ES Geforce RTX 4090 Reference 1-Slot-Design</a>: Alphacool 1U 适用于 Nvidia Geforce RTX 4090 的水冷散热器 – 针对服务器和工作站</li><li><a href="https://www.globalsources.com/ATX-motherboard/x99-motherboard-1214343068p.htm`">未找到标题</a>: 未找到描述</li><li><a href="https://www.gigabyte.com/Motherboard/TRX50-AI-TOP#kf`">TRX50 AI TOP 主要特性 | 主板 - GIGABYTE 全球</a>: 未找到描述</li><li><a href="https://amzn.asia/d/jjJkJnL">LINKUP - AVA5 PCIE 5.0 Riser Cable | 为 Gen 5 GPU 垂直安装提供未来保障 | x16 128GB/s 速度 | 兼容 PCIe 4.0 & WRX80/WRX90E | 直角, 黑色 15cm : Amazon.com.au: 电子产品</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1284243443899437096)** (116 messages🔥🔥): 

> - `OpenAI's o1 models`
> - `Cursor AI's coding assistant`
> - `AI Scaling Laws`
> - `Anthropic evals`
> - `Funding rounds for AI startups`

- **OpenAI 发布 o1 模型**：OpenAI 发布了旨在提高复杂任务推理能力的 o1 模型，因其在科学和编程应用中的潜力而备受关注。
   - 据报道，新模型的表现优于旧版本，但在处理大规模编辑时仍有困难，Cursor AI 正在通过其专门的编程助手解决这一挑战。
- **AI 初创公司融资激增**：11x AI 在 A 轮融资中筹集了 2400 万美元，其 ARR 增长了 15 倍并推出了新的数字员工，彰显了其快速增长。
   - 同样，Supermaven AI 获得了 1200 万美元融资，用于开发一款与其模型无缝集成的 AI 文本编辑器。
- **AI Scaling Laws 解析**：一段全面概述 AI scaling laws 的视频在本周受到关注，强调了近期研究的相关性。
   - 讨论指出了围绕 scaling laws 不断演进的理解及其对未来 AI 模型开发的影响。
- **Anthropic 推出新的评估课程**：Anthropic 推出了一门专注于 LLM prompt 评估的课程，旨在通过识别 prompt 中的边缘情况来确保生产就绪。
   - 该课程包括提供数值评分的方法论，解决了用户在评估模型时面临的常见挑战。
- **社区对模型开发的见解**：社区内的讨论显示出对 o1 能力的复杂情感，公开分享了关于其潜力和局限性的看法。
   - 参与者表达了探索新 AI 模型全部功能的渴望，同时也对传统软件工程中的工作角色提出了疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/zhouwenmeng/status/1834899729165304198">Wenmeng Zhou (@zhouwenmeng) 的推文</a>: Qwen-q1 ? ? 🍓🍓🍓🍓🍓</li><li><a href="https://x.com/aaronp613/status/1834393945050087567?s=46">Aaron (@aaronp613) 的推文</a>: Apple 发布了 3 段由 Bella Ramsey 出演的 Apple Intelligence 宣传视频 🧵 第一段：更个性化的 Siri</li><li><a href="https://x.com/wgussml/status/1834691198013129053">william (@wgussml) 的推文</a>: 大多数人会忽略的一点是，o1 的重要性恰恰在于它不是在合成数据上进行的 SFT。事实上，在不受约束的 CoT 上进行 RL 是有效的，且不会崩溃成乱码般的 CoT 步骤，这真的很...</li><li><a href="https://x.com/zeyuanallenzhu/status/1834981677887897891?s=46">Zeyuan Allen-Zhu (@ZeyuanAllenZhu) 的推文</a>: 刚刚上传了 Part 2.1 的 1 小时独家视频，包含许多技术细节。https://youtu.be/bpp6Dz8N2zY。Part 2.2 将在大约一周后上线。引用 Zeyuan Allen-Zhu (@ZeyuanAllenZhu) (1/...</li><li><a href="https://x.com/livgorton/status/1834769173458960675?s=46">Liv (@livgorton) 的推文</a>: 让我感到有些惊讶的是，前训练后处理负责人、PPO 论文的第一作者 John Schulman 竟然没有参与一个显然需要大量 RL 的模型？有可能他...</li><li><a href="https://www.interconnects.ai/p/reverse-engineering-openai-o1">逆向工程 OpenAI 的 o1 </a>: 推理时计算（test-time compute）的生产化向我们展示了 AI 的未来。探索已进入语言模型训练领域。</li><li><a href="https://www.cursor.com/blog/instant-apply">近乎即时的全文件编辑</a>: 未找到描述</li><li><a href="https://x.com/mariots/status/1834732382261317744?s=46">Mario Schlosser (@mariots) 的推文</a>: 关于理赔判定：数以千计的自然语言规则（包括合同、最佳实践和指南）决定了医疗服务的成本。综合这些规则极其繁琐，...</li><li><a href="https://cookbook.openai.com/examples/o1/using_reasoning_for_routine_generation">使用推理进行常规生成 | OpenAI Cookbook</a>: 使用 OpenAI API 进行构建的开源示例和指南。浏览代码片段、高级技术和演练集合。分享你自己的示例和指南。</li><li><a href="https://x.com/alexalbert__/status/1835717512404914401?s=46">Alex Albert (@alexalbert__) 的推文</a>: 我们关于 LLM 提示词评估（prompt evaluations）的最新课程已发布。评估（Evals）能确保你的提示词达到生产级标准，因为你能快速捕捉边缘情况并精准定位提示词需要改进的地方。...</li><li><a href="https://x.com/lumalabsai/status/1835742651662139529?s=46">Luma AI (@LumaLabsAI) 的推文</a>: 🚀 隆重推出 Dream Machine API。开发者现在可以使用全球最受欢迎且直观的视频生成模型来构建和扩展创意产品，而无需在内部构建复杂的工具...</li><li><a href="https://x.com/maximlott/status/1834652893229859212">Maxim Lott (@maximlott) 的推文</a>: 刚刚在我的 AI IQ 追踪页面上绘制了新的 @OpenAI 模型。请注意，这项测试是由一名 Mensa 会员为我的测试创建的纯线下 IQ 测验，*不在任何 AI 训练数据中*（所以得分...</li><li><a href="https://x.com/11x_official/status/1835711787712582082?s=46">11x (@11x_official) 的推文</a>: 👋🏻 大家好，我是 Alice 和 Jordan - 我们刚刚从 @benchmark 筹集了 2400 万美元的 Series A 融资！在此阅读我们的完整博客文章：https://www.11x.ai/blog/series-a 今年以来的一些亮点：- 我们的 ARR 增长了...</li><li><a href="https://x.com/cursor_ai/status/1834665828308205661">Cursor (@cursor_ai) 的推文</a>: OpenAI 的新 o1 模型已在 Cursor 中上线！我们发现 o1 在处理定义明确、推理密集型的问题上表现出色。对于大多数任务，我们仍然推荐使用 sonnet/4o。我们最初正在推出...</li><li><a href="https://x.com/SmokeAwayyy/status/1834641370486915417">Smoke-away (@SmokeAwayyy) 的推文</a>: 邮件内容：</li><li><a href="https://x.com/supermavenai/status/1835743882971426837?s=46">Supermaven (@SupermavenAI) 的推文</a>: 我们已从 Bessemer Venture Partners 筹集了 1200 万美元，用于构建一个与我们的模型紧密集成的 AI 驱动型文本编辑器。</li><li><a href="https://x.com/OpenRouterAI/status/1835099755648893286">OpenRouter (@OpenRouterAI) 的推文</a>: 我们正在发布一个临时仪表板，以帮助用户了解 o1 的推理 Token（reasoning tokens）：</li><li><a href="https://x.com/scottastevenson/status/1834702489511223749?s=46">Scott Stevenson (@scottastevenson) 的推文</a>: 1. 它解决了长文档修订的问题。律师很少从头开始起草合同，他们通常从先例开始，并根据当前的交易进行修改。以前很难让 GPT4 执行...</li><li><a href="https://x.com/tensor_fusion/status/1834983832786710831?s=46">milton (@tensor_fusion) 的推文</a>: 我通常不看关于 AI/ML 讲解的 YT 视频（除了 Karpathy/3blue1bro...</li>

) 但这具有很高的制作价值。对 Scaling laws 进行了很好的概述（从 Kaplan 2020 到最近的结果）。 (bonu...</li><li><a href="https://x.com/goodside/status/1834975429960011851?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Riley Goodside (@goodside) 的推文</a>：o1 prompting 对我来说很陌生。它的思考方式有时极其有效，但也像梦境一样，不听劝告。只需说出你想要的并祈祷即可。任何关于“如何做”的说明都会伴随着...</li><li><a href="https://x.com/jessicalessin/status/1834621175005409442?s=46">来自 Jessica Lessin (@Jessicalessin) 的推文</a>：又是一天，来自 @theinformation 关于 @OpenAI 巨额新一轮融资的更多细节。我认为现在是开始思考这些投资者（现在包括对冲基金...）的绝佳时机。</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1fgin90/summary_of_what_we_have_learned_during_ama_hour/">Reddit - 深入了解一切</a>：未找到描述</li><li><a href="https://github.com/anthropics/courses/tree/master/prompt_evaluations">anthropics/courses 的 prompt_evaluations 分支</a>：Anthropic 的教育课程。通过在 GitHub 上创建账号来为 anthropics/courses 的开发做出贡献。</li><li><a href="https://github.com/openai/openai-python/blob/120d225b91a8453e15240a49fb1c6794d8119326/chatml.md#few-shot-prompting">openai/openai-python 的 chatml.md</a>：OpenAI API 的官方 Python 库。通过在 GitHub 上创建账号来为 openai/openai-python 的开发做出贡献。</li><li><a href="https://x.ai/profile-settings">xAI 登录</a>：未找到描述</li><li><a href="https://ide.x.ai">PromptIde</a>：未找到描述</li><li><a href="https://developers.x.ai/api/api-key/">创建 API Key - xAI 开发者平台</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1284299618334802074)** (1 条消息): 

> - `OpenAI API`
> - `Structured Outputs`
> - `ChatGPT Latest`
> - `Voice Mode`
> - `O1 Meetup` 


- **关于揭秘 OpenAI 的新播客**：最新的[播客集数](https://x.com/latentspacepod/status/1834740722551210274)包含一小时的对话，涵盖了 **Structured Outputs**、**ChatGPT-latest** 和 **gpt-4o**，以及对各种 API 问题的回答。
   - 它包含了来自 **O1 emergency meetup** 的见解以及 **OpenAIDevs AMA** 环节的回顾。
- **Structured Outputs 的见解**：该播客探讨了 **Structured Outputs** 与 function calling 之间的区别，讨论了开发者的实现挑战和使用案例。
   - 关键主题包括 **JSON Schema** 的作用以及正在开发的 **Structured Output Roadmap**。
- **Voice Mode API 讨论**：本集深入探讨了新的 **Voice Mode API**，它允许更具交互性和动态的对话能力。
   - 它强调了这一功能如何改变用户在各种平台上与 AI 的交互。
- **O1 Meetup 回顾**：专题介绍了来自 **O1 emergency meetup** 的 **Q&A** 环节，成员们讨论了在使用 OpenAI 工具开发过程中的经验和挑战。
   - 听众获得了关于社区驱动的解决方案以及对持续开发问题的贡献的见解。
- **ChatGPT 扩展策略**：讨论了扩展 **ChatGPT** 的策略，特别是专注于**增加的 Latency** 以及用于优化的 **prompt/schema caching** 技术。
   - 团队解决了关于 **model reproducibility** 的担忧，以及 API 不断演进的 **tiering and rate limiting** 策略。



**提到的链接**：<a href="https://x.com/latentspacepod/status/1834740722551210274">来自 Latent.Space (@latentspacepod) 的推文</a>：🆕 播客：从 API 到 AGI：Structured Outputs，OpenAI API 平台和 O1 Q&A。我们的 @openai 周末特辑！https://latent.space/p/openai-api-and-o1 - 与 @michpokrass 关于 Structured Outputs 的 1 小时对话...

  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1284242627889336353)** (133 条消息🔥🔥): 

> - `Cursor 扩展性问题`
> - `HTEC AI 报告`
> - `Neovim 资源`
> - `Vim 挑战`
> - `AI 编程内容` 


- **Cursor 面临扩展性挑战**：成员们对 **Cursor** 的扩展性问题表示担忧，特别是其代码补全和文档生成功能。
   - 一位用户建议，由于初次安装时的默认设置，他们的初始体验可能受到了限制。
- **HTEC 关于 AI Copilot 的报告**：近岸咨询公司 **HTEC** 发布了一份关于他们使用 26 种 AI 编程工具经验的 [报告](https://htec.com/htec-report-ai-code-generators/)，尽管访问需要注册。
   - 成员们讨论了报告中提到的简短使用时间和局限性是否真实反映了这些工具的能力。
- **面向初学者的 Neovim 资源**：用户分享了宝贵的 **Neovim** 资源，包括一个旨在帮助用户掌握该编辑器的 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLx2ksyallYzW4WNYHD9xOFrPRYGlntAft)。
   - 另一位用户提到了 **Kickstart**，认为它是设置 Neovim 配置的有用指南。
- **掌握 Vim 的挑战**：成员们注意到 **Vim** 陡峭的学习曲线，强调虽然起初可能会降低速度，但一旦掌握，效率将显著提高。
   - 几位成员对没有早点学习 Vim 表示遗憾，并分享了在编程任务中转向 **Cursor** 和 **Claude** 的经验。
- **AI 编程内容推荐**：成员们认可了社区在分享 **AI Programming** 工具和技术更新方面的价值，强调希望看到更多实际应用展示。
   - 此外，用户推荐了知名的内容创作者，如 **McKay Wrigly** 和 **Riley Brown**，他们提供了专注于 AI 编程的高质量内容。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://vim-racer.com/">未找到标题</a>: 未找到描述</li><li><a href="https://gptengineer.app/">GPT Engineer</a>: 仅使用聊天界面构建软件产品</li><li><a href="https://www.youtube.com/playlist?list=PLx2ksyallYzW4WNYHD9xOFrPRYGlntAft">Understanding Neovim</a>: 成为配置 Neovim 的高手！</li><li><a href="https://openv0.dev/">v0.dev → openv0.dev</a>: Openv0.dev 是一个开源 AI 模型，可根据简单的文本提示生成 Tailwind CSS UI。我们相信开源的力量，因为它能促进协作和透明度。这就是...</li><li><a href="https://github.com/ThePrimeagen/harpoon/tree/harpoon2">GitHub - ThePrimeagen/harpoon (harpoon2 分支)</a>: 通过在 GitHub 上创建账号来为 ThePrimeagen/harpoon 的开发做出贡献。</li><li><a href="https://github.com/nvim-lua/kickstart.nvim">GitHub - nvim-lua/kickstart.nvim: 个人 nvim 配置的起点</a>: 个人 nvim 配置的起点 - nvim-lua/kickstart.nvim</li><li><a href="https://github.com/latentspacenotes/latentspacenotes.github.io">GitHub - latentspacenotes/latentspacenotes.github.io</a>: 通过在 GitHub 上创建账号来为 latentspacenotes/latentspacenotes.github.io 的开发做出贡献。</li><li><a href="https://github.com/tris203/precognition.nvim">GitHub - tris203/precognition.nvim: 💭👀precognition.nvim - Precognition 使用虚拟文本和侧边栏符号显示可用的移动操作。</a>: 💭👀precognition.nvim - Precognition 使用虚拟文本和侧边栏符号显示可用的移动操作。 - tris203/precognition.nvim</li><li><a href="https://github.com/raidendotai/openv0">GitHub - raidendotai/openv0: AI 生成的 UI 组件</a>: AI 生成的 UI 组件。通过在 GitHub 上创建账号来为 raidendotai/openv0 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1284230046843801630)** (41 messages🔥): 

> - `OpenAI o1 基准测试结果`
> - `Humanity's Last Exam`
> - `游说与 AI 政策`
> - `Dan Hendrycks 争议`
> - `AI 与算力预算` 


- **OpenAI o1 性能引发辩论**：成员们讨论了最近开放使用的 OpenAI `o1-preview` 和 `o1-mini` 模型，这些模型旨在提升推理能力。讨论重点在于考虑到不同的算力预算，现有 Benchmark 的公平性问题。
   - *一位成员建议，更公平的评估应该包括跨模型的算力预算匹配，并使用 pass@k 评分。*
- **Humanity's Last Exam 发布公告**：Dan Hendrycks 及其合作者推出了 *Humanity's Last Exam*，征集具有挑战性的 AI 问题。截至 2024 年 11 月 1 日，优秀提交者可分享 500,000 美元的奖金池。
   - 成员们对该计划的影响表达了复杂的情绪，推测高性能表现将如何影响有关 AI 监管的政治游说。
- **对 AI 游说的担忧**：Dan Hendrycks 在 AI 政策领域的影响力及其与政治家的联系，引发了关于未来可能基于 *Humanity's Last Exam* 等计划的性能指标采取监管行动的讨论。
   - *几位参与者对他游说者的角色及其与 AI 技术背景的交织表示担忧。*
- **AI 倡导 vs 政治**：成员们辩论了 AI 倡导与政治游说之间的界限，考虑到 Dan Hendrycks 既是 AI 安全倡导者，又是某家专注于游说的 AI 公司顾问的双重身份。
   - 一些人对政治因素渗入 AI 讨论表示反感，一位成员指出 *个人价值观与 AI 炒作之间复杂的交集。*
- **研究生生活感悟**：一位成员回忆了在快速发展的 AI 环境中作为一名研究生的积极经历，表示专注于自己喜欢的事情很重要。
   - *讨论强调了学术环境中常见的战友情谊和智力上的兴奋感。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/colin_fraser/status/1834623952788033925">Colin Fraser (@colin_fraser) 的推文</a>: 我在使用本周最后几个 o1-mini 额度时注意到，推理错误会导致 Chain-of-Thought 的胡言乱语失控，同时强化错误并...</li><li><a href="https://x.com/DanHendrycks/status/1835725770402185399">Dan Hendrycks (@DanHendrycks) 的推文</a>: 有对人类和 AI 都具挑战性的问题吗？我们 (@ai_risks + @scale_AI) 正在启动 Humanity's Last Exam，这是一项旨在创建全球最难 AI 基准测试的大规模合作。提交...</li><li><a href="https://arcprize.org/blog/openai-o1-results-arc-prize">ARC-AGI-Pub 上的 OpenAI o1 结果</a>: o1 preview 和 mini 模型距离 AGI 还有多远？</li><li><a href="https://fortune.com/2024/09/13/sam-altman-openai-non-profit-structure-change-next-year/">Sam Altman 告知 OpenAI 员工，公司的非营利企业结构将在明年发生变化</a>: OpenAI CEO 此前曾承认公司的结构“不同寻常”。现在他明确表示，是时候改变它了。</li><li><a href="https://fxtwitter.com/zhouwenmeng/status/1834899729165304198?s=46">Wenmeng Zhou (@zhouwenmeng) 的推文</a>: Qwen-q1 ? ? 🍓🍓🍓🍓🍓
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1285295406992851066)** (5 messages): 

> - `自我反思 (Self Reflection) 论文`
> - `推理与输出的一致性` 


- **寻找自我反思论文**：一位成员询问关于 **Self Reflection** 的优秀论文，表示该主题的可用资源较少。
   - 他们提到对 **推理与输出之间的一致性** 特别感兴趣，尽管记不清确切的标题。
- **围绕询问的轻松玩笑**：另一位用户幽默地回应了最初的询问，表明讨论氛围很轻松。
   - 玩笑在其他参与者的笑声中继续，展示了积极的社区动态。

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1285254614748233759)** (7 条消息): 

> - `Matt from IT`
> - `Meme 内容`
> - `Planetcoo Barbarons 理论`
> - `Reinforcement Learning 观察` 


- **怀念 Matt from IT**：一位成员表达了对 **Matt from IT** 的思念，感叹自他离开后缺乏有趣的内容。
   - *注意到缺乏 meme*，大家一致认为他的缺席让社区的参与度出现了缺口。
- **对更多 meme 内容的需求**：有人呼吁提供更多类似于 [abrakjamson 的推文](https://x.com/abrakjamson/status/1834336551922471348?s=46) 中关于 Planetcoo Barbarons 的 **meme** 内容。
   - 幽默和能引起共鸣的 **meme** 被认为是保持社区活跃的关键。
- **Reinforcement Learning 对语言的影响**：一位成员引用了 [karpathy 的推文](https://x.com/karpathy/status/1835561952258723930)，强调当 **Reinforcement Learning** 执行得当时，模型在思考过程中会开始失去英语的连贯性。
   - 这一观察引发了关于不同训练方法下 AI 语言变化的细微差别的讨论。
- **对 Matt “愿景家”式的看法**：另一位成员评论了对 Matt 看法的转变，因为一些人现在在他离职后将其视为 **visionary**（愿景家）。
   - *这引发了关于此类言论严肃性的讨论*，揭示了社区内钦佩与怀疑交织的情绪。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/abrakjamson/status/1834336551922471348?s=46">Abram Jackson (@abrakjamson) 的推文</a>: 我关于为什么我们不被允许看到思考过程的理论：全是 Planetcoo Barbarons。</li><li><a href="https://x.com/karpathy/status/1835561952258723930">Andrej Karpathy (@karpathy) 的推文</a>: 当模型在 chain of thought 中不再说英语时，你就能判断 RL 做得很好。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1284254149692624999)** (53 messages🔥): 

> - `OpenAI o1 models`
> - `Nando 加入 Microsoft AI`
> - `Gemini Flash 1.5 性能`
> - `模型推理机制`
> - `Token 经济学` 


- **OpenAI 的 o1 模型引发关注**：OpenAI 发布了名为 **o1-preview** 和 **o1-mini** 的新模型，其名称可能暗示了“具有非凡能力的外国人”（aliens of extraordinary ability，即 O-1 签证）。观察者注意到这两个模型都有有趣的推理模式，引发了关于其功能的讨论。
   - 一位用户分享了有趣的观察：**mini 的推理时间并不比 preview 长**，但产生的回复却更长，这对许多人来说是一个意想不到的结果。
- **Nando 加入 Microsoft AI 团队**：NandoDF 宣布了他的新职位，加入 **Microsoft AI**，专注于大规模多模态研究和产品开发。这一出人意料的举动引发了人们的猜测，认为在这个规模虽小但雄心勃勃的团队中，他有能力塑造 AI 的未来。
   - 成员们对 Nando 的转型表示惊讶，认为这是职业生涯的一次重大胜利，并强调了此类机会的丰厚回报。
- **Gemini Flash 1.5 占据领先地位**：据报道，**Gemini Flash 1.5** 在月度排名中超过了 **MythoMax**，标志着性能上的一个里程碑。值得注意的是，一位用户强调在短短两天内生成了 **28B tokens**，展示了该模型的强大威力。
   - 这引发了关于训练过程效率的问题，特别是考虑到当前数据生成的经济效益。
- **关于模型推理的讨论**：围绕模型中推理（reasoning）和补全（completion）之间的区别展开了对话，观点认为 **推理可见性** 在很大程度上取决于用户。一位用户对这一前提提出了挑战，断言生成方法上的区别并不像描述的那样明显。
   - 这引发了关于推理过程 **Token 经济学** 的进一步对话，包括与传统生成结构的比较。
- **对即将发布内容的期待**：关于 **Dwarkesh 和 Ilya** 节目的暗示引起了用户的兴奋，一些人希望在发布前保持私密。随着讨论的深入，成员们注意到了预训练数据成本的影响，观察到模型训练费用的急剧下降。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/livgorton/status/1834769173458960675?s=46">来自 Liv (@livgorton) 的推文</a>：让我感到有些惊讶的是，前 post-training 负责人、PPO 论文的第一作者 John Schulman，竟然没有为一个可能需要大量 RL 的模型做出贡献？有可能他...</li><li><a href="https://x.com/polynoamial/status/1834644274417119457?s=46">来自 Noam Brown (@polynoamial) 的推文</a>：@sog_on_bird_app @OpenAIDevs 虽然我们希望总结器是忠实的，但不能保证它一定忠实。我绝对不建议假设它对 CoT 是忠实的，或者 CoT 本身...</li><li><a href="https://fxtwitter.com/jacobrintamaki/status/1835745908350456151?s=46">来自 Jacob Rintamaki (@jacobrintamaki) 的推文</a>：嘘...剧透预警 👀</li><li><a href="https://fxtwitter.com/_clashluke/status/1835743877728461257?s=46">来自 Lucas Nestler (@_clashluke) 的推文</a>：那完全是我干的 😅 过去两天在 Gemini Flash 1.5 中生成了 28B tokens。它是个好模型。https://x.com/OpenRouterAI/status/1835713079344275809 引用 OpenRouter (@OpenRouterAI) 榜首...</li><li><a href="https://fxtwitter.com/terryyuezhuo/status/1834644182528672134">来自 Terry Yue Zhuo (@terryyuezhuo) 的推文</a>：虽然我对 o1-preview 和 o1-mini 进行了初步评估，但这些结果可能无法严格反映模型的能力。我目前正在 Big Bench 上运行每个任务 5 个样本的 o1-preview...</li><li><a href="https://x.com/aidan_mclau/status/1835729356406329372?s=46">来自 Aidan McLau (@aidan_mclau) 的推文</a>：引人入胜的 o1 观察：&gt;mini 的推理并不比 preview 长 &gt;mini 的回复比它的推理长 &gt;preview 的推理比它的回复长 &gt;...</li><li><a href="https://x.com/nandodf/status/1835712503286018216?s=46">来自 Nando de Freitas (@NandoDF) 的推文</a>：我加入了 @Microsoft AI，以推进大规模多模态 AI 研究的前沿，并为人们构建实现有意义目标和梦想的产品。MAI 团队规模虽小，但资源充足...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

xeophon.: https://x.com/burny_tech/status/1834741998898536774?s=46 <:berk:750111476483752166>
  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1284757614441529344)** (7 条消息): 

> - `Reverse Curriculum Learning`
> - `LLMs 应用`
> - `RL 中的利基应用` 


- **RL 爱好者讨论 Reverse Curriculum Learning**：最近出现了一些关于 **LLMs** 背景下 **Reverse Curriculum Learning** 的论文，但其在 RL 社区中的使用似乎有限。
   - 一位成员指出，尽管它被认为是一种有效的方法，但尚未得到广泛采用。
- **识别出 Reverse Curriculum Learning 的挑战**：讨论指出，**Reverse Curriculum Learning** 通常被认为比较**笨重**，通常适用于**利基应用 (niche applications)**。
   - 成员们表示，这种局限性可能解释了它在更广泛场景中很少被使用的原因。


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1285281787026870414)** (6 条消息): 

> - `Ben Thompson 的准确性`
> - `信息摄取栈 (Information ingestion stack)`
> - `David Perell 的影响力`
> - `关于 Ben Thompson 的 YouTube 视频` 


- **Ben Thompson 令人印象深刻的见解**：成员们称赞 Ben Thompson 的更新**准确得令人印象深刻**，强调了他能够在高水平上讨论**技术话题**的能力。
   - *他在需要时会阅读论文，甚至是整本书*，展示了理解复杂问题的透彻方法。
- **对 Thompson 信息处理过程的好奇**：一位成员对 Ben Thompson 的**信息摄取栈 (information ingestion stack)** 表示感兴趣，并注意到他在各个话题上的一贯准确性。
   - 他们以最近一篇关于 **Telegram 和加密** 的帖子为例，以及他对 **Apple 在爱尔兰税务案** 的评论。
- **YouTube 视频推荐**：一位成员推荐观看名为 [How Ben Thompson Built a Writing Empire](https://www.youtube.com/watch?v=igh0JeaUHzo) 的 YouTube 视频，该视频解释了撰写 newsletter 如何在经济上获得丰厚回报。
   - 视频指出，Ben Thompson 每年通过写作赚取数百万美元，这激励了社区中的其他人。
- **对 David Perell 的钦佩**：一位成员分享了他们对 **David Perell** 长期以来的钦佩，这与那些欣赏深刻内容的参与者不谋而合。
   - 这种情绪反映了用户参与并支持写作和技术领域思想领袖的日益增长的趋势。
- **理解困惑的概念**：讨论了许多人（包括一些参与者）如何能够**轻松摄取**令人困惑的信息，并快速掌握他们当前的理解。
   - 这种处理复杂话题的能力反映了在快速变化的环境中保持**消息灵通和适应性**的愿望。



**提到的链接**：<a href="https://www.youtube.com/watch?v=igh0JeaUHzo">How Ben Thompson Built a Writing Empire</a>：如果写 newsletter 可以支付你的房租呢？嗯，它可以。今天，你将学习如何做到。Ben Thompson 每年通过他的写作赚取数百万美元...

  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1285212246263205899)** (92 条消息🔥🔥): 

> - `OAI 的 ChatGPT 和 API 数据`
> - `OLMoE 命名疑虑`
> - `LLM 模型演进`
> - `Poe 订阅讨论`
> - `首字母缩写词排名` 


- **关于 OAI 数据影响的疑问**：一位成员推测 **OpenAI 的 ChatGPT/API 数据** 在多大程度上影响了其模型的开发，认为他们可能直接利用了用户数据中引人注目的 CoT，或者对其进行了 Prompt Engineering。
   - *如果用户交互数据发挥了重要作用，* 那么开源替代方案要复制这种模型可能会非常困难。
- **围绕 OLMoE 命名的辩论**：用户讨论了 **OLMoE** 这个名字的古怪之处，认为 “Open Mixture of Experts Language Models” 准确来说并不能缩写为 **OLMoE**。
   - 一位用户幽默地建议，这个名字在*原始法语*中可能更有意义。
- **对 LLM 模型发展的兴奋**：一位成员对有关 LLM 的未来计划表达了新的兴奋感，表示他们最初对 2025 年的发展并不热衷，直到最近才改变看法。
   - 有人指出，人们对 LLM 突破性时刻的期待日益增长，这让人联想起过去的重大里程碑。
- **Poe 订阅服务评估**：成员们讨论了他们对 **Poe** 平台的偏好，其中一人指出支付 **$20** 即可访问该服务上提供的所有 LLM。
   - 也有人对可用性和界面提出了担忧，一些人表示与 **Claude** 和 **ChatGPT** 等竞争对手相比，他们对该平台的美学设计并不感冒。
- **最牵强的首字母缩写词排名**：在一次轻松的交流中，成员们产生了一个想法，即为最“牵强”的首字母缩写词创建一个排名系统，并开玩笑地将其命名为 **ALICE** —— AwfuL aI aCronym rankEr。
   - 这引发了关于 **SPLADE** 和 **Google Gemini Ultra** 等具有挑战性的 AI 名称的讨论，反映了 AI 品牌命名的荒诞性。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1284435818429153353)** (87 条消息🔥🔥): 

> - `Fei-Fei Li 的推理方法`
> - `Command-R-Plus-08-2024 性能问题`
> - `语言模型有效性`
> - `LLM 中的审查制度`
> - `Nick Frosst 与 Good Kid` 


- **关于 Fei-Fei Li 推理方法的咨询**：成员们讨论了对 **Fei-Fei Li** 解决推理问题的方法的好奇心，寻求关于她具体方法的见解。
   - 在当前 AI 进展的背景下，人们对理解她的技术有着浓厚的兴趣。
- **Command-R-Plus-08-2024 的性能问题**：一位用户报告称，与前代模型相比，**Command-R-Plus-08-2024** 模型在用于创意写作时表现出更多的重复输出。
   - 有人担心长 prompt 可能会如何影响模型的性能，并鼓励探索替代模型。
- **关于 LLM 有效性和审查制度的辩论**：成员们讨论了在语言模型中进行**审查 (censorship)** 的适当性，特别是在商业背景下，同时强调了法律和伦理影响。
   - 对话强调了一种观点，即虽然适度调节是必要的，但过度限制可能会阻碍模型的潜力。
- **Nick Frosst 的音乐背景**：分享了一个关于 Cohere 联合创始人 **Nick Frosst** 的趣闻，透露他的独立摇滚乐队 *Good Kid* 取得了显著成功，在 Spotify 上拥有数百万听众。
   - 该乐队以编程主题的歌曲闻名，最近在 **Lollapalooza** 演出，并获得了 Juno Award 提名。
- **使用 Cohere 进行数据提取的指导**：一位新用户正在寻求关于利用 **Cohere 的模型** 通过 token classification 从非结构化文件中提取数据的建议。
   - 他们询问了 chat 与 classify 模型的有效性，以及在多标签数据集中标注输出的最佳方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/arattml/status/1834622684938031302">ar (@arattml) 的推文</a>: 这里没什么有用的，你应该跳过这篇文章 https://arxiv.org/abs/2402.05808 https://arxiv.org/abs/2407.03181 https://arxiv.org/abs/2401.08967 https://arxiv.org/abs/2407.00087 https://arxiv.org/abs...</li><li><a href="https://tenor.com/view/dancing-duck-dance-duck-duck-ooontz-dance-gif-10943740227711557279">跳舞鸭 GIF - Dancing duck Dance duck Duck - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://open.spotify.com/track/1P59E9uaeejHQ5xu0EG4p6?si=0ad643efea334f8a">First Rate Town</a>: 歌曲 · Good Kid · 2023</li><li><a href="https://cohere.com/pricing">价格</a>: 直接通过我们的 API 访问我们的模型，以创建可扩展的生产工作负载。</li><li><a href="https://github.com/codelion/optillm">GitHub - codelion/optillm: 针对 LLM 的优化推理代理</a>: 针对 LLM 的优化推理代理。通过在 GitHub 上创建账户为 codelion/optillm 的开发做出贡献。</li><li><a href="https://techcrunch.com/2024/09/15/cohere-co-founder-nick-frossts-indie-band-good-kid-is-almost-as-successful-as-his-ai-company/?guccounter=1">Cohere 联合创始人 Nick Frosst 的独立乐队 Good Kid 几乎和他的 AI 公司一样成功 | TechCrunch</a>: 当他不忙于为企业客户构建大语言模型时，Nick Frosst 是独立摇滚乐队 Good Kid 的主唱。
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1285191154333253655)** (1 条消息): 

> - `Cohere Developer Office Hours`
> - `Command 模型系列更新`
> - `RAG 能力提升`
> - `Safety Modes 功能`
> - `Command 模型价格更新` 


- **今天参加 Cohere Developer Office Hours！**：Cohere 将于今天**美国东部时间下午 1 点 (1 PM ET)** 举办 Developer Office Hours，重点介绍 **Command 模型系列**的最新更新。
   - 主持人将涵盖 **RAG 能力**和 **Safety Modes** 的新功能及改进等主题。
- **Command R 模型令人兴奋的改进**：新版本的 **Command R 模型系列**在编程、数学、推理和延迟方面带来了增强，现在更加高效且性能更强。
   - 值得注意的改进包括更新后的 Command R 模型**吞吐量增加了 50%**，**延迟降低了 20%**。
- **引入 Safety Modes 以实现更好的控制**：Cohere 全新的 **Safety Modes** 为企业客户提供了改进的模型护栏 (guardrails) 以及对模型使用的更大控制权。
   - 这一举措使用户能够在保持模型有效性的同时，更好地管理交互。
- **阐明检索增强生成 (RAG) 的增强功能**：最新的模型还展示了针对细微多语言任务量身定制的增强型**检索增强生成 (RAG)** 能力。
   - 这些模型在 **23 种语言**上进行了训练，并经过微调以支持一系列现实世界的应用。
- **Command 模型价格更新**：Cohere 更新了 **Command R 和 Command R+ 模型**的定价，为开发者优化了成本。
   - 用户现在可以以每百万 tokens **$2.50** 的输入价格和 **$10.00** 的输出价格访问 **Command R+**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cohere.com/blog/command-series-0824">Updates to the Command R Series</a>：Command R 模型系列的最新版本在编程、数学、推理和延迟方面提供了改进。</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-08-2024">CohereForAI/c4ai-command-r-08-2024 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024">CohereForAI/c4ai-command-r-plus-08-2024 · Hugging Face</a>：未找到描述</li><li><a href="https://cohere.com/pricing">Pricing</a>：直接通过我们的 API 访问模型，以创建可扩展的生产工作负载。</li><li><a href="https://cohere.com/blog/intro-safety-modes">Introducing Safety Modes</a>：Cohere Safety Modes 为企业客户提供了对模型护栏的更大控制权。</li><li><a href="https://docs.cohere.com/changelog/command-gets-refreshed">Command models get an August refresh — Cohere</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1284418627080163338)** (96 条消息🔥🔥): 

> - `预训练 VLM 的使用`
> - `Command-r-08-2024 的更新`
> - `本地模型部署咨询`
> - `相似度搜索与 Embeddings`
> - `灵活使用 Token 进行聊天模型微调` 


- **关于预训练 VLM 的讨论**：一名成员询问了由于计算资源限制而使用预训练 VLM 的情况，引发了关于其适用性的讨论。
   - 其他人指出，这可能会引发社区内关于模型部署实践的更广泛讨论。
- **对 Command-r 模型更新的期待**：一位用户询问了更新 **Command-r-08-2024** 模型的计划，特别是关于增强其韩语写作风格方面。
   - 团队成员确认正在努力提高该模型的多语言能力，并欢迎社区反馈。
- **本地模型部署的挑战**：成员们讨论了在受限的 Tesla M10 GPU 上本地部署模型的问题，强调了硬件限制和性能挑战。
   - 他们分享了本地部署的潜在解决方案，同时承认了旧硬件目前的局限性。
- **Embeddings 与相似度搜索策略**：一位用户咨询了用于相似度搜索的最佳 Embedding 模式，专家建议使用 **search embeddings** 以获得有效结果。
   - 专家给出了使用 reranker 来最大化相关结果数量的建议，强调了处理大型数据集的稳健流水线的重要性。
- **微调不带结束 Token 的聊天模型**：关于在微调期间是否可以省略最后的 `<|END_OF_TURN_TOKEN|>` 以保持对话流畅性的问题被提出。
   - 尽管最初存在限制，但大家对未来探索这种微调聊天模型的灵活性表现出了兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://minimap.ai/?mapId=5541696631538816&query=Nasa&page=1&searchKey=1277097258848012">Minimap.ai</a>：未找到描述</li><li><a href="https://docs.cohere.com/page/cookbooks#search">Cookbooks — Cohere</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/datasets#dataset-types>),">Datasets — Cohere</a>：该文档提供了 Dataset API 的概述，包括文件大小限制、数据保留政策、数据集创建、验证、元数据保留、使用数据集微调模型等...</li><li><a href="https://cohere.com/deployment-options">Deployment Options</a>：我们的解决方案提供行业领先的数据隐私和安全性，旨在满足寻求利用生成式 AI 力量的组织的各种需求。无论您是初创公司还是...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1284466092781338634)** (13 条消息🔥): 

> - `生产密钥创建问题`
> - `Sagemaker 客户端计费错误`
> - `Cohere 的银行卡问题` 


- **用户在创建生产密钥时遇到困难**：一位用户报告在尝试创建 **production key** 时出错，并说明其位于 **印度** 且没有 VPN 问题。
   - 社区建议联系 [support@cohere.com](mailto:support@cohere.com) 以寻求有关此计费问题的帮助。
- **Sagemaker 客户端返回负值的计费单位**：一位在 Cohere Python SDK 中使用 **Sagemaker client** 的用户注意到，响应显示 **input_tokens** 和 **output_tokens** 为 **-1.0**。
   - 社区建议发送电子邮件至 [support@cohere.com](mailto:support@cohere.com)，以获取针对该异常计费返回结果的账户特定分析。
- **关于支付银行卡问题的建议**：一位成员建议，计费问题可能源于银行卡设置，建议用户在银行 App 中检查 **国际支付** 功能。
   - 他们建议如果问题持续存在，可以尝试 **不同的银行**，并指出某些银行可能无法通过小额初始扣款验证。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1284887416477585499)** (2 条消息): 

> - `招聘信息关注点`
> - `与 Cohere 的相关性` 


- **建议删除招聘信息**：一位成员敦促从讨论中删除 **招聘信息** 部分，因为这似乎与 **Cohere 相关性** 不大。
   - 他们强调了将焦点集中在与社区相关话题上的重要性。
- **关于 Cohere 相关性的进一步讨论**：同一位成员建议，在删除招聘信息后，如果认为该话题 **与 Cohere 有一定相关性**，可以考虑再次发布。
   - 这表明了维持小组讨论重点明确的愿望。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1284244705240158269)** (181 条消息🔥🔥): 

> - `FLUX Models 使用`
> - `使用 Stable Diffusion 进行角色设计`
> - `图像编辑与 Inpainting`
> - `ControlNet 与 LoRA 训练`
> - `Stable Diffusion 技术支持` 


- **FLUX Models 的挑战**：用户询问了运行 FLUX models 的相关问题，特别是关于 .sft 和 .safetensor 格式的指导，以及与 Forge 等工具的兼容性。
   - 建议切换到 ComfyUI 以获得更好的支持，用户还分享了关于特定模型大小的使用经验。
- **创建 2D 角色概念**：一位用户询问了如何使用 Stable Diffusion checkpoints 和特定的 prompt 措辞来生成像 Cheetara 这样的角色。
   - 讨论内容包括咨询哪些成功的 checkpoints 能够产出适合后续 3D 建模的角色艺术图。
- **图像编辑技巧**：针对移除图像文本和使用 Inpainting 方法填充背景，提出了多项建议，并推荐参考 GIMP 或 Piximperfect 的教程。
   - 用户还讨论了各种在保持质量的同时增强和修改图像的 AI 工具。
- **用于角色动画的 ControlNet 和 LoRA 训练**：关于使用 ControlNet 和 LoRA 训练来创建分离的矢量风格角色动画的讨论非常普遍，并推荐了合适的训练示例。
   - 用户分享了关于如何利用 ControlNet 技术在艺术渲染中进行角色姿态和结构处理的见解。
- **Stable Diffusion 安装技术支持**：一位用户在安装 Stable Diffusion 时遇到错误，被引导至支持频道提供其错误日志以获取帮助。
   - 频道内分享了安装指南的有用链接，并强调需要详细的日志以便进行故障排除。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/thunder-cats-gif-7172707">Cheetara GIF - Thunder Cats - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">black-forest-labs/FLUX.1-dev · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/black-forest-labs/FLUX.1-schnell">black-forest-labs/FLUX.1-schnell · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui 安装指南</a>：Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info</li><li><a href="https://github.com/flushedface">flushedface - 概览</a>：flushedface 有 5 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/fairy-root/Flux-Prompt-Generator?tab=readme-ov-file">GitHub - fairy-root/Flux-Prompt-Generator: Flux Prompt Generator 为图像生成模型提供了一个灵活且可定制的 prompt 生成器，用于生成详细且富有创意的 prompt。</a>：Flux Prompt Generator 为图像生成模型提供了一个灵活且可定制的 prompt 生成器。 - fairy-root/Flux-Prompt-Generator</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Gui">首页</a>：Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1284266096987209738)** (2 条消息): 

> - `用户验证流程`
> - `引导问题`
> - `服务器变更讨论频道`
> - `验证机器人的延迟问题` 


- **用户验证流程上线**：Discord 服务器已实施 **用户验证** 流程，要求成员通过 #verify 频道中的机器人分享其电子邮件地址。
   - 选择不进行验证的成员将拥有受限的消息发送权限，但仍保留对所有频道的读取权限。
- **引入新的引导问题**：在验证电子邮件地址后，用户将遇到 **两个多选题引导问题**，旨在提升其服务器体验。
   - 此步骤旨在简化新旧成员的引导（Onboarding）流程。
- **新增服务器变更讨论频道**：已创建一个新频道用于讨论 **即将到来的服务器变更**，成员可以在此分享建议并提问。
   - 这一举措体现了服务器对持续改进用户体验的承诺。
- **延迟问题导致验证机器人推迟**：验证机器人已上线，但存在 **延迟问题**，导致其被暂时禁用并锁定了验证频道。
   - 团队正在努力解决这些问题，并在机器人恢复运行后通知成员。


---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1284227237930860687)** (91 条消息🔥🔥): 

> - `Mojo and Python interoperability` (Mojo 与 Python 的互操作性)
> - `Count Leading Zeros` (计算前导零)
> - `Creating Slices of String Literals` (创建字符串字面量切片)
> - `Zero-Copy Data Interop` (零拷贝数据互操作)
> - `LLVM Intrinsics at Comptime` (编译时的 LLVM Intrinsics)


- **Mojo 与 Python 互操作性的挑战**：讨论强调了 Mojo 目前缺乏从 Python 导入模块或调用函数的能力，这被视为实现有效互操作性的前提。
   - 参与者表示有兴趣了解如何促进 Mojo 与 Python 之间的零拷贝数据交换，特别是在高性能场景下。
- **编译时 (Comptime) 计算前导零 (CLZ) 的问题**：用户注意到 `clz` 函数在编译时无法工作，因为它依赖于 LLVM intrinsics，而这些内建函数在此时无法执行。
   - 分享了一个计算前导零的替代实现，并建议未来可能会改进标准库中用于编译时计算的功能。
- **创建字符串字面量切片**：成员们交流了如何在 Mojo 中创建字符串字面量切片，比较了 `ListLiteral` 和 `List` 结构。
   - 讨论了优化和语法问题，并暗示未来的更新可能会引入对可迭代性的进一步改进。
- **探索零拷贝数据互操作**：一位参与者质疑在 Mojo 和 Python 之间实现零拷贝数据互操作的可行性，并列举了当前的局限性。
   - 人们担心示例中引用的 NumPy 操作在执行过程中如何处理数据拷贝。
- **编译时支持 LLVM Intrinsics**：一项新更新确认 Mojo 现在支持在编译时为基于整数的函数（如 `ctlz` 和 `popcount`）提供 LLVM intrinsics 支持。
   - 建议未来扩展对更多类型的支持，旨在增强 LLVM 的常量折叠能力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/notebooks/Mandelbrot">Mandelbrot in Mojo with Python plots | Modular Docs</a>: 学习如何编写高性能 Mojo 代码并导入 Python 包。</li><li><a href="https://github.com/modularml/mojo/issues/3379">[BUG] REPL incorrectly displays boolean SIMD vector contents · Issue #3379 · modularml/mojo</a>: Bug 描述：REPL 显示布尔 SIMD 向量的方式存在错误。似乎如果存在至少一个 True 条目，REPL 表示仅显示 f...</li><li><a href="https://github.com/modularml/mojo/issues/3482">[BUG] RPL gets chars from a String pointer correctly, Mojo file does it wrong · Issue #3482 · modularml/mojo</a>: Bug 描述：你可以在以下截图中看到：复现步骤：使用以下代码片段测试：var s = String(&quot;ab&quot;) p = s.unsafe_ptr() c = chr(int(p.load())) print(&#39;Ch...</li><li><a href="https://github.com/modularml/mojo/issues/3480">[BUG] Return values are correct, but REPL reports incorrect · Issue #3480 · modularml/mojo</a>: Bug 描述：REPL 报告的布尔值与 print 输出报告的值不匹配。复现步骤：magic init project --mojoproject cd project magic s magic run mojo 以下代码...</li><li><a href="https://github.com/makism/mojo-on-fedora40">GitHub - makism/mojo-on-fedora40: Instructions on installing Mojo on Fedora 40.</a>: 在 Fedora 40 上安装 Mojo 的说明。通过在 GitHub 上创建账号为 makism/mojo-on-fedora40 开发做贡献。</li><li><a href="https://github.com/modularml/mojo/pull/3438">[stdlib] Complete the string literals signature to match the `String` one by msaelices · Pull Request #3438 · modularml/mojo</a>: 为了匹配 Mojo 和 Python 字符串中的现有方法。这可能有助于 Python 程序员在使用 REPL 和处理字符串的小型 Mojo 示例时进行过渡。</li><li><a href="https://github.com/modularml/mojo/issues/933">[mojo-compiler] CompTime interpreter should be able to fold `pop.call_llvm_intrinsic` · Issue #933 · modularml/mojo</a>: Bug 描述：math.bit 函数无法在编译时运行。复现步骤：考虑以下代码：import math.bit as bit fn main(): alias n = bit.ctlz(10) 它产生以下 e...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1284234567858913331)** (30 条消息🔥): 

> - `Mixed Precision Training`
> - `CoreWeave Valuation`
> - `AI's Impact on Society`
> - `Foundation Models in Biotech`
> - `LM Evaluation Harness and Azure` 


- **理解 Mixed Precision Training 的挑战**：讨论强调了 **mixed precision training** 的复杂性，指出虽然同时以 fp32 和 fp16 存储模型可以提高性能，但它可能会使 forward pass 期间产生的计算负荷翻倍。
   - 成员们提到，由于计算预算的限制，项目中的性能 **trade-offs** 很常见，并强调了平衡速度和资源利用率的重要性。
- **CoreWeave 的巨额估值**：据报道，云计算提供商 **CoreWeave** 正在洽谈现有股份的出售，公司估值达到 **230 亿美元**，这反映了其在 AI 领域的地位。
   - 这一估值凸显了云计算和 AI 领域激烈的竞争和投资氛围，引起了知名财经媒体的关注。
- **探讨 AI 的社会影响**：对 AI 影响的反思认为，**OpenAI** 实际上已经为“每个人的口袋里装进了一个 PhD”，这表明尽管公众目前反应微弱，但社会运作方式可能会发生潜在转变。
   - 讨论表明，需要更深入地考虑 AI 在各个领域的**变革性影响**及其在日常生活中的持续融合。
- **生物技术中的 Foundation Models 介绍**：一位成员分享了他们在生物技术领域使用 **foundation models** 的背景，重点关注序列和表格数据的规模化 **representation learning**。
   - 这为小组内关于先进建模技术的知识共享和协作提供了机会。
- **关于 Azure 版 LM Evaluation Harness 的咨询**：有人询问 **lm-evaluation-harness** 仓库是否支持通过 Azure 访问 **OpenAI completions** 和 GPT 相关模型，表现出对其功能的兴趣。
   - 此类咨询凸显了利用现有框架高效对接云端 AI 服务的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://stephenfry.substack.com/p/ai-a-means-to-an-end-or-a-means-to">AI: A Means to an End or a Means to Our End?</a>: 这是我在 9 月 12 日星期四为伦敦国王学院数字未来研究所举办的首届“与技术共生”讲座上发表的演讲文本。</li><li><a href="https://nvidia.github.io/apex/amp.html">apex.amp &mdash; Apex 0.1.0 文档</a>: 未找到描述</li><li><a href="https://discuss.pytorch.org/t/why-to-keep-parameters-in-float32-why-not-in-b-float16/179931">为什么将参数保留在 float32，而不是 (b)float16？</a>: 我想知道是否应该将模型参数保留在 float16 或 bfloat16 中？这可能与 automatic mixed precision / autocast 是正交的，或者 mixed precision 可能不再有意义...</li><li><a href="https://finance.yahoo.com/news/cloud-computing-firm-coreweave-talks-144351011.html">云计算公司 CoreWeave 洽谈以 230 亿美元估值出售股份</a>: (彭博社) —— CoreWeave 是一家云计算提供商，也是人工智能竞赛中最热门的初创公司之一，目前正在洽谈安排出售现有股份，估值达 230 亿美元...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1284350281953448027)** (23 条消息🔥): 

> - `RWKV 团队进展`
> - `在小数据集上过拟合模型`
> - `Sequential Monte Carlo steering`
> - `长文档中的自回归特性 (Autoregressiveness)`
> - `OpenAI 的推理系统 o1` 


- **RWKV 团队推动 RNN 边界**：RWKV 团队备受关注，成员们指出了 *Smerky* 等人的贡献，强调了在推进 RNN 架构方面的协作努力。
   - *fern.bear* 赞扬了团队的创新，表示看到该领域持续推进令人印象深刻。
- **关于 9 张图像过拟合的担忧**：一位用户分享了他们无法在仅 9 张图像上使模型过拟合的担忧，引发了关于这是否预示着从更大数据集学习时存在潜在问题的讨论。
   - 回复指出，如果模型无法在如此小的样本上过拟合，那么它在处理更大数据集时可能也会遇到困难。
- **介绍 Sequential Monte Carlo steering**：讨论重点介绍了一种名为 *Sequential Monte Carlo steering* 的新方法，旨在改进 LLM 的输出约束，详见一篇 arXiv 论文。
   - 社区对这种方法很感兴趣，特别是它展示了一个用于实验的新编程库。
- **自回归特性的挑战**：有人担心将长文档拆分到多个窗口中可能会使自回归模型的处理变得复杂。
   - 然而，成员们争论认为，一个训练良好的模型仍然应该能从可用的文档长度中解读出上下文。
- **OpenAI 推理系统 o1 的介绍**：OpenAI 推出了他们的新推理系统 *o1*，旨在通过增强的推理技术改进 AI 在复杂任务中的交互。
   - 该系统旨在通过在推理过程中实施在线搜索机制，实现超越传统语言模型的创新。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/_portal_/status/1834562705430102290">来自 Portal (@_portal_) 的推文</a>：下周一美国东部时间中午 12 点，@brekelmaniac 将加入 LoGG 主持人 @HannesStaerk，讨论用于概率推理问题的 Sequential Monte Carlo (SMC)。📄 阅读论文：https://arxiv.org/abs/2404.175...</li><li><a href="https://www.interconnects.ai/p/reverse-engineering-openai-o1">逆向工程 OpenAI 的 o1 </a>：将 test-time compute 产品化向我们展示了 AI 的未来。探索已进入语言模型训练领域。</li><li><a href="https://www.wolframalpha.com/input?i=how+many+golf+balls+can+fit+in+the+moon>">月球能装下多少个高尔夫球？ - Wolfram|Alpha</a>：Wolfram|Alpha 为最广泛的人群（涵盖所有职业和教育水平）提供专家级的知识和能力。</li><li><a href="https://www.reddit.com/r/MachineLearning/comments/1ffz5xc/p_attempting_to_replicate_the_stretching_each/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2306.03081">使用概率程序对大语言模型进行 Sequential Monte Carlo Steering</a>：即使经过微调和强化学习，大语言模型 (LLMs) 也很难（如果不是不可能的话）仅通过提示词进行可靠控制。我们提出了一种新的推理时方法...</li><li><a href="https://github.com/probcomp/hfppl">GitHub - probcomp/hfppl: 使用 HuggingFace 语言模型的概率编程</a>：使用 HuggingFace 语言模型的概率编程 - probcomp/hfppl
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1284302138113130519)** (7 条消息): 

> - `带有 KV cache 的循环计算`
> - `在固定 VM 上运行复杂算法`
> - `Scaling laws 文献建议`
> - `大模型的世界模型准确性` 


- **关于循环计算的讨论**：一位成员表达了对支持 **带有 KV cache 的循环计算** 架构的需求，指出目前的能力尚存空白。
   - 这突显了在机器学习框架中高效处理复杂算法所面临的持续挑战。
- **在小型 VM 上运行复杂算法**：针对前一条评论，一位成员提到复杂算法仍然可以在小型 **固定 VM** 上执行，从而有效地利用现有资源。
   - 这引发了对 **VM**（虚拟机）一词的澄清，为技术讨论做出了贡献。
- **世界模型规模的探索**：一位成员分享了关于从多种传感器进行准确世界建模所需的不切实际的模型规模（约 **8T parameters**）的见解，详见[此处](https://chatgpt.com/share/66e4e751-7e70-8005-83fa-dd93f5ac70e5)。
   - 他们指出，虽然目前的模型缺乏推断非预期信息的能力，但足够大的模型可以接入外部数据源。
- **征求 Scaling laws 文献**：一位成员寻求关于 **scaling laws** 的文献推荐，表示熟悉模型训练并对 **sparse autoencoders** 和 **mutual information** 感兴趣。
   - 他们的阅读清单中已经有了 `Scaling Laws for Autoregressive Generative Modeling`，并渴望获得更多资源。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1284339795153256489)** (10 条消息🔥): 

> - `Goodfire.ai 融资`
> - `SAE latents 术语`
> - `LLM 层的鲁棒性`
> - `学术论文中的贡献署名惯例` 


- **Goodfire.ai 凭借 700 万美元融资进行扩张**：Goodfire.ai 最近筹集了 **700 万美元** 以增强其可解释性平台，显然是利用了 **Anthropic** 在 Claude 上的工作。这笔资金旨在扩大其 AI 可观测性方法，如 [VentureBeat 文章](https://venturebeat.com/ai/goodfire-raises-7m-for-its-brain-surgery-like-ai-observability-platform/)中所述。
   - 成员们对该公司在可解释性领域的具体方向和技术重点表示好奇。
- **关于 SAE latents 与 features 的争论**：一位成员幽默地指出了在当前写作中使用 **'latents'** 与 **'features'** 的术语混淆，展示了社区对讨论中命名法的关注。这引发了关于遵循惯例与在学术论述中建立更清晰术语的辩论。
   - 另一位成员反驳了使用 **SAE features** 的观点，主张定义术语的一致性，并引用了之前的论文作为支持。
- **来自 Tegmark 关于 LLM 层论文的见解**：一位成员赞赏了由 **Max Tegmark** 合著的一篇论文的见解，该论文概述了 **LLM 层** 推断的四个阶段。最后阶段侧重于 **sharpening**（锐化），即抑制神经元消除无关特征，从而提高准确性。
   - 这种方法进一步深入研究了 LLM 在干预期间的鲁棒性，尽管发生了结构变化，但仍保持了显著的准确性。
- **关于论文作者署名的讨论**：围绕署名作者的最佳惯例展开了辩论，特别是关于 **Tegmark 团队** 论文中第一作者和资深作者的区别。成员们建议采用各种命名惯例，以便在保持清晰度的同时准确认可贡献。
   - 对话探讨了如何在学术引用中有效平衡对第一作者和资深贡献者的认可。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://blog.eleuther.ai/autointerp/">Open Source Automated Interpretability for Sparse Autoencoder Features</a>：构建和评估用于自动可解释性的开源流水线</li><li><a href="https://arxiv.org/abs/2406.19384">The Remarkable Robustness of LLMs: Stages of Inference?</a>：我们通过删除和交换相邻层来展示和调查 Large Language Models 卓越的鲁棒性。我们发现删除和交换干预保留了原始模型 72-95% 的性能...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1285107629336821793)** (1 messages): 

> - `lm-evaluation-harness repo`
> - `Azure OpenAI integration` 


- **关于 lm-evaluation-harness 对 Azure 支持的咨询**：一名成员询问 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 仓库是否支持 **Azure OpenAI 密钥**和端点，以访问 **OpenAI completions** 和其他 **GPT 相关模型**。
   - 该咨询突显了将 **Azure 的能力**与现有评估框架集成的兴趣。
- **Azure OpenAI 模型访问讨论**：进行了一场关于 **Azure OpenAI** 在利用现有 API 的同时，为各种 **GPT 模型**提供更便捷访问潜力的广泛讨论。
   - 成员们对集成如何简化模型评估流程表示好奇。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1284775489306624000)** (6 messages): 

> - `Hugging Face's Pile Deduped Dataset`
> - `Launching Multi-Node with GPT-NeoX on Polaris`
> - `Job Submission Issues on Polaris` 


- **关于数据集中 EOS Token 的咨询**：一名成员询问 [Pile Deduped Dataset](https://huggingface.co/datasets/EleutherAI/pile-deduped-pythia-random-sampled) 中缺失 **EOS tokens** 的情况是否属实，以及这是否与其他经验相关。
   - 这一担忧突显了数据集配置中潜在的不一致性，可能会影响模型训练。
- **多节点启动协助请求**：一名成员请求在 **Polaris** 上使用 **GPT-NeoX** 启动**多节点**设置的步骤。
   - 另一名成员建议查看 [官方指南](https://docs.alcf.anl.gov/polaris/data-science-workflows/applications/gpt-neox/) 以获取帮助。
- **Polaris 上交互式作业的挑战**：一名成员对在 **Polaris** 上获取交互式作业需要等待超过 **24 小时**表示沮丧，这使得访问系统变得复杂。
   - 他们表示更倾向于能够将作业提交到队列中，而不是依赖交互式设置。
- **在 Polaris 上运行的成功调整**：一名成员分享了在 **Polaris** 上运行 **GPT-NeoX** 时所做的成功调整，并指出使用 **qsub** 进行作业排队的重要性。
   - 他们强调 **DeepSpeed** 无法将 host file 识别为环境变量，并建议按照步骤配置免密 SSH 以确保功能正常。


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1284240454468239426)** (5 messages): 

> - `LlamaParse Excel Capabilities`
> - `TypeScript Workflows in LlamaIndex`
> - `Unit Testing in LLM Applications`
> - `Vectara-Agentic Library`
> - `Code Generation Agent` 


- **LlamaParse 在解析 Excel 数据方面表现出色**：在[最近的视频](https://twitter.com/llama_index/status/1834680455171653959)中，@ravithejads 展示了 LlamaParse 先进的 **Excel 解析能力**，包括处理多个工作表和复杂表格。
   - LlamaParse 利用**递归检索**自动总结复杂表格，增强了可用性和效率。
- **LlamaIndex 引入 TypeScript 工作流**：正如该[公告](https://twitter.com/llama_index/status/1834689049954804098)所述，LlamaIndex 现已将工作流集成到 TypeScript 中。
   - 此功能旨在为 TypeScript 用户简化开发流程。
- **LLM 应用中单元测试的重要性**：@maskaravivek 在一篇博客文章中强调，单元测试对于防范 LLM 应用的随机性至关重要，文章详细介绍了使用 [CircleCI](https://twitter.com/llama_index/status/1834987463569555909) 构建和测试 RAG 应用的过程。
   - 该文章强调，适当的单元测试可以减轻 AI 驱动应用中的意外行为。
- **Vectara-Agentic 简化 RAG 实现**：查看由 @ofermend 开发的 [vectara-agentic](https://twitter.com/llama_index/status/1835348333478760896)，这是一个基于 LlamaIndex 和 Vectara 构建 Agentic RAG 的简单库。
   - 它提供了构建具备规划和工具使用能力的 Agent 的功能，并兼容各种模型提供商。
- **创新的代码生成 Agent 发布**：@MarcusSchiesser 展示了一个出色的**代码生成 Agent**，允许用户使用 Tailwind CSS 和 JavaScript 在单个 HTML 文件中生成整个 Web 应用，详见此[链接](https://twitter.com/llama_index/status/1835729007926743426)。
   - 该 Agent 与 o1 等最新模型集成，通过自然语言促进 Web 应用的创建。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1284228194228240515)** (51 条消息🔥): 

> - `LlamaIndex 与 ChromaDB 集成`
> - `LanceDB 查询问题`
> - `SitemapReader HTTP 错误`
> - `ReactAgents 流式响应`
> - `加密项目的就业机会` 


- **LlamaIndex 在文档引用方面存在困难**：一位用户讨论了在 LlamaIndex 中配合 ChromaDB 检索文档引用时遇到的问题，指出即使是无关的查询，响应中也会包含文档。
   - 另一位成员建议检查 `response.source_nodes` 以获得更好的结果。
- **LanceDB 向量索引查询问题**：一位用户在尝试使用 embeddings 查询 LanceDB 时遇到了 `AttributeError`，提示 `LanceDBVectorStore` 对象没有 `vector_store` 属性。
   - 讨论揭示了在对象设置方式以及索引是否符合用户预期方面可能存在混淆。
- **SitemapReader 的 HTTP 403 错误**：一位用户在使用 `SitemapReader` 时遇到了 HTTP Error 403，表示未经授权的访问，尽管尝试添加了 user-agent 请求头。
   - 成员们澄清了 `load_data()` 方法不接受请求头，并建议可能需要进行身份验证。
- **ReactAgents 与流式输出**：一位用户询问在使用 ReactAgents 进行流式聊天（streaming chat）时响应速度慢的问题，提到包含答案的 observation 已经在 backend 接收到了。
   - 成员们指出流式延迟可能是由于代码中的固有设置造成的，并建议检查模拟流（dummy stream）的速度。
- **加密项目的职位列表**：一位用户发布了一个加密项目的测试、分发、NFT 艺术和 Web 开发岗位的招聘信息，鼓励感兴趣的人员私信（DM）。
   - 另一位成员幽默地评论让发布者去睡觉，展示了社区内轻松的氛围。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/workflow/sub_question_query_engine/">作为工作流的子问题查询引擎 - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/723c2533ed4b7b43b7d814c89af1838f0f1994c2/llama-index-core/llama_index/core/chat_engine/types.py#L92">llama_index/llama-index-core/llama_index/core/chat_engine/types.py (GitHub)</a>：LlamaIndex 是为您 LLM 应用程序提供的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1285210962567106671)** (4 条消息): 

> - `本地 LLM 成本优化`
> - `本地 LLM 的保密性`
> - `框架比较` 


- **本地 LLM 提供成本优化**：成员们强调，与使用 **OpenAI** 服务相比，运行 **本地 LLM** 可以显著降低成本。
   - 对话中提到，**OpenAI** 与本地模型的总拥有成本（**TCOS**）会有所不同。
- **本地 LLM 增强数据保密性**：讨论强调，使用 **本地 LLM** 允许企业将 **私有信息** 保留在内部，而不是发送给 **OpenAI**。
   - 这突显了人们对公共 AI 服务中数据保密性日益增长的担忧。
- **成本和保密性问题被忽视**：一位成员对某篇文章没有解决与使用 **OpenAI** 相关的 **成本** 和 **保密性** 问题表示失望。
   - 这种担忧反映了在 AI 工具中平衡性能与隐私的更广泛情绪。
- **本地 LLM 作为后端，LlamaIndex 作为前端**：社区讨论了将 **本地 LLM** 视为 **后端**，将 **LlamaIndex** 视为 **前端** 的观点，并将其与 **Flutter** 或 **ReactJS** 等框架进行了类比。
   - 这种类比表明，为了功能性，人们可以接受可能显得“臃肿”的框架。


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1284317792425873449)** (11 条消息🔥): 

> - `GPU 收益递减点`
> - `命令行模式下的非流式响应`
> - `Open Interpreter 入门指南`
> - `在编程中使用 API 和 Requests` 


- **理解 GPU 收益递减**：GPU 的**收益递减点**因应用而异；通常，对于游戏，在 **2-3 个 GPU** 后变得明显，而渲染可能在 **4-6 个 GPU** 左右看到。
   - 因素包括 **PCIe 带宽限制**以及未针对多 GPU 优化的软件。
- **Open Interpreter 的非流式响应**：一位成员寻求关于如何在命令行模式下**停止流式响应**的建议，以避免导致不适的终端刷新。
   - 另一位成员建议使用 **--plain 标志**或 `claude-3.5` 模型作为非流式响应选项。
- **新手在使用 Open Interpreter 时的挑战**：一位新手对 **Open Interpreter** 中关于启用 API 和 requests 的限制表示好奇，并为其项目寻求指导。
   - 他们报告说，尽管指令说明不要使用这些功能，模型仍然要求使用，这导致了困惑。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1284255171978723469)** (17 条消息🔥): 

> - `Livekit 设置错误`
> - `ChatGPT O1 模型发布`
> - `模型功能对比` 


- **Livekit 设置困惑**：一位成员指出 **90% 的用户**在设置 **Livekit** 时遇到错误，并批评文档不规范。
   - 另一位成员建议分享正确的设置指南，并通过 PR 贡献以协助社区。
- **对 ChatGPT O1 模型的担忧**：有人怀疑 ChatGPT 发布名为 **O1** 的模型可能是对现有项目的战略攻击，但这一观点被另一位成员淡化了。
   - 他们认为，虽然 ChatGPT 的 O1 在推理方面表现出色，但本项目专注于执行代码和其他功能。
- **关于 O1 功能的辩论**：一位成员批评 O1 缺乏对**多模态输入**的支持，称其无法提供与旧版 **model 4** 相同的完整功能。
   - 他们补充说，他们的测试显示 O1 和 model 4 的响应相似，暗示这可能只是*炒作*。



**提到的链接**：<a href="https://tenor.com/view/arnold-schwarzenegger-sneaking-out-camouflage-serious-gif-5272373">Trying To Sneak Out Of The House GIF - Arnold Schwarzenegger Sneaking Out Camouflage - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1284286299817246794)** (2 条消息): 

> - `MoA LLM 库`
> - `用于编程的 Mixture-of-Agents` 


- **用于 LLM 编排的新 Python 库**：[MoA LLM 库](https://github.com/catena-labs/moa-llm)允许用户在受神经网络启发的结构中编排 LLM，增强多个模型之间的协作。
   - 该开源项目旨在简化各种 LLM 的集成以获得更好的性能。
- **用于编程任务的自定义 Mixture-of-Agents**：[MoA Coding mix](https://crosshatch.app/mixes/moa-coding) 针对具有挑战性的编程任务进行了优化，使用了 **Claude 3.5 Sonnet** 和 **GPT-4 Turbo** 等模型。
   - 在复杂的编程挑战中，与单独使用 Claude 3.5 Sonnet 相比，它显示出 **28%** 的性能提升，并提供 **$6.00/M tokens** 的竞争性价格。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://crosshatch.app/mixes/moa-coding">Coding Mixture of Agents | Crosshatch</a>：一种专门为具有挑战性的编程任务优化的自定义 Mixture-of-Agents (MoA) 合成组合。该组合利用了多个“提议”模型，包括 Claude 3.5 Sonnet 和 GPT-4 Turbo...</li><li><a href="https://github.com/catena-labs/moa-llm">GitHub - catena-labs/moa-llm: A Python library to orchestrate LLMs in a neural network-inspired structure</a>：一个用于在受神经网络启发的结构中编排 LLM 的 Python 库 - catena-labs/moa-llm
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1284229900957519974)** (16 messages🔥): 

> - `O1 model discussion`
> - `Attention optimization`
> - `Reflection tasks and agents`
> - `Special tokens for pretraining` 


- **对 O1 有效性的复杂感受**：关于 **O1 model** 存在争议；一些人表示满意，而另一些人则认为其表现平平，并指出其 **responses 过于机械化**。
   - 一位成员强调 O1 感觉就像是一个带有扎实 UI 的 **Chain of Thought**，而另一位成员则对其能力保持怀疑。
- **OpenAI 的 O1 并非短期产物**：一位成员强调 **OpenAI 开发 O1 (Strawberry/Q*)** 已经有很长一段时间了，反驳了关于其训练周期短的说法。
   - 他们提到 O1 似乎利用了 **agentic chain of thought**，展示了对抗幻觉（hallucination）的韧性。
- **对 Attention 实现灵活性的担忧**：讨论中提到了对 **FlashAttention** 等优化 Attention 实现的需求，但也指出在尝试新变体时会失去灵活性。
   - 成员们对 ML 研究人员在寻找现有的优化 kernels 以满足其 Attention 需求时面临的 **'software lottery'** 表示担忧。
- **预训练中 Special Tokens 的考量**：有人询问在预训练期间是否应删除特定的 **special tokens**，共识倾向于保留现有的已定义 tokens。
   - 成员们建议遵循已建立的 tokens，以避免后续出现复杂化和潜在的不一致性。
- **Fused Cross Entropy 的影响**：一位成员阐明了 **cross entropy** 与 **fused cross entropy** 之间的关系，指出后者提供了更好的性能。
   - 他们提到同时启用这两种类型可能会导致其中一种禁用另一种，并指出了这些优化背后的集成决策。



**提及的链接**：<a href="https://pytorch.org/blog/flexattention/">FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention</a>：   

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1284239554315944040)** (11 messages🔥): 

> - `Tokenization Error Bug`
> - `Phi 3.5 Sentence Classifier`
> - `vLLM and Adapter Issues`
> - `Fused Attention Configuration` 


- **Masking 导致的 Tokenization 错误**：一位成员遇到了由于新的 per-turn masking（针对 chat template prompt 策略）导致的 tokenization 错误，该错误屏蔽了最后一个 end of turn token。
   - 他们将此问题链接到了 GitHub 上详细的 bug 报告：[Tokenization Bug Report](https://github.com/axolotl-ai-cloud/axolotl/issues/1916)。
- **放弃 Phi 3.5 训练**：一位成员表达了在尝试让基于 **Phi 3.5** 的句子分类器输出预期的分类文本标签时的挫败感。
   - 他们提供了其 [dumb sentence classifier](https://huggingface.co/fozziethebeat/phi-3.5-alpaca-test-classifier) 的链接，并暗示暂时放弃。
- **vLLM 在处理训练好的 adapters 时遇到困难**：一位成员指出 **vLLM** 无法正确解析 `qkv_proj` 层，导致使用 **Axolotl** 的 adapters 训练的模型出现问题。
   - 他们观察到，虽然他们的 LoRA 在合并（merging）时显示没有学习到内容，但当作为纯 adapter 置于 base model 之上使用时，表现正常。
- **关于 fused attention 设置的问题**：一位成员询问在训练期间是否使用了 fused attention 方法，以及使用的是 checkpoints 还是最终模型。
   - 另一位成员指出运行 `_post_training` 的函数可以重新拆分层，这表明了对维持模型完整性的关注。
- **Phi 模型的 tensor 追踪限制**：一位成员讨论了 Phi 3.5 在追踪 **qkv_proj** tensor 方面的挑战，这与 **Llama** 的追踪方法不同。
   - 他们强调 **vLLM** 默认将 Phi 3.5 作为 Llama 模型处理，由于 tensor 结构不匹配，导致 adapter 映射变得复杂。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1916)">Issues · axolotl-ai-cloud/axolotl</a>：欢迎提出 axolotl 问题。通过在 GitHub 上创建账号为 axolotl-ai-cloud/axolotl 的开发做出贡献。</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1879">Reward model by winglian · Pull Request #1879 · axolotl-ai-cloud/axolotl</a>：增加了对使用 trl 的 RewardTrainer 训练 reward models 的支持。目前使用 pairwise responses。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1284235030750822523)** (22 messages🔥): 

> - `GenAI/RAG/CV Consultation Services` (GenAI/RAG/CV 咨询服务)
> - `Impact of OpenAI on Society` (OpenAI 对社会的影响)
> - `LangGraph Cloud Pricing` (LangGraph Cloud 定价)
> - `Streaming LLM Output Issues` (LLM 输出流式传输问题)
> - `Managing Chat History in LangChain` (在 LangChain 中管理聊天记录)


- **提供 GenAI/RAG/CV 咨询服务**：一名成员宣布可承接与 **GenAI**、**RAG** 和 **CV** 相关的咨询项目，旨在帮助初创公司和企业开发原型。
   - 有意者请通过私信联系。
- **OpenAI 对社会的影响**：一位成员表示担心，尽管 **OpenAI** 彻底改变了获取知识的方式，但社会依然像什么都没发生一样运行。
   - 另一位贡献者建议，加速自动化可能会带我们进入**后稀缺时代 (post-scarcity era)**。
- **LangGraph Cloud 定价的不确定性**：一名成员寻求关于 **LangGraph Cloud** 在 Beta 阶段之后潜在成本的澄清，并在使用它与开发自定义 **FastAPI** 封装方案之间进行权衡。
   - 他们担心被锁定在长期来看可能不可行的定价模型中。
- **LLM 输出流式传输的挑战**：一名成员强调了在流式传输 **LLM** 输出时，由于使用 **Pydantic** 解析器处理不完整的 **JSON** 字符串而导致的解析困难。
   - 尽管起初持怀疑态度，但他们发现将 `parse_result` 切换为 `parse` 方法后取得了成功。
- **LangChain 中的聊天记录管理**：一位用户提出了关于使用 **LangChain** 管理聊天记录的问题，指出内置方法在跟踪额外 **UI** 数据方面存在局限性。
   - 他们描述了在存储聊天记录与应用特定消息时，实现事务完整性 (transactional integrity) 所面临的挑战。



**提到的链接**：<a href="https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/json/#streaming">JSON parser | 🦜️🔗 LangChain</a>：该输出解析器允许用户指定任意 **JSON** schema，并向 **LLM** 查询符合该 schema 的输出。

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1285312073320103957)** (1 messages): 

> - `Reading PDFs with Tables` (读取带有表格的 PDF)
> - `Comparing Table Data` (比较表格数据)
> - `Sample Implementations for PDFs` (PDF 示例实现)
> - `Time Consumption in Data Processing` (数据处理中的耗时问题)


- **寻求高效的 PDF 表格读取方法**：一名成员请求关于读取带有表格的 **PDF** 文件的建议，重点是如何在之后高效地针对数据提问。
   - 特别欢迎任何关于高效处理此问题的链接或示例实现，因为目前该过程非常**耗时**。
- **比较表格列的挑战**：成员们对需要比较 **PDF** 表格中的列及其繁琐程度表示担忧。
   - 成员表示这种比较特别乏味，希望能有相关的解决方案。


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

batmanosama: Nhttps://www.interconnects.ai/p/reverse-engineering-openai-o1
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1284236193562624022)** (19 条消息🔥): 

> - `RAG 查询结构`
> - `DSPy LM 发布`
> - `DSPy 中的视觉 LLM 模型`
> - `GitHub 贡献` 


- **简化 RAG 查询结构**：一位成员询问了如何在单一模块中优化 RAG，建议将来自 RAG、内存和提示词的数据打包进 'context' 字段以获得更好的效果。
   - 另一位成员确认了这种方法，并强调 [这个简单的 RAG 示例](https://github.com/stanfordnlp/dspy/blob/main/skycamp2023.ipynb) 对进一步理解很有帮助。
- **DSPy LM 今日发布**：成员们获悉 DSPy 的最新版本 **2.4.16** 已经发布，有助于实现各种功能。
   - 这是针对有关 **dspy.LM** 状态的查询做出的回应，确认其已在讨论当天发布。
- **关于视觉 LLM 模型的问题**：一位成员询问是否可以在 DSPy 中使用视觉 LLM 模型进行图像描述，另一位成员回复称该功能可能在下周上线。
   - 分享了一个相关的 [GPT-4 Vision API 的 Pull Request](https://github.com/stanfordnlp/dspy/pull/682)，表明集成工作正在进行中。
- **GitHub 贡献咨询**：一位成员表达了为 DSPy 项目做贡献的兴趣，并询问是否有可领取的赏金（bounties）。
   - 讨论表明，预计还会有额外的集成变更，预计完成时间为 **7-10 天**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/stanfordnlp/dspy/pull/682">由 jmanhype 提交的增加 GPT-4 Vision API 封装的 Pull Request #682 · stanfordnlp/dspy</a>：在 visionopenai.py 中引入了一个新的 GPT4Vision 类来封装 GPT-4 Vision API。该抽象层简化了调用 API 进行图像分析的过程。关键功能...</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/skycamp2023.ipynb">dspy/skycamp2023.ipynb (位于 main 分支) · stanfordnlp/dspy</a>：DSPy：用于对基础模型进行编程（而非提示）的框架 - stanfordnlp/dspy</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/examples/qa/hotpot/hotpotqa_with_MIPRO.ipynb">dspy/examples/qa/hotpot/hotpotqa_with_MIPRO.ipynb (位于 main 分支) · stanfordnlp/dspy</a>：DSPy：用于对基础模型进行编程（而非提示）的框架 - stanfordnlp/dspy
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1284463648621854862)** (12 条消息🔥): 

> - `Tinygrad 中的运行时类型检查`
> - `Tinygrad 在 AMD 上的测试失败`
> - `Tinygrad 更新问题`
> - `GitHub Pull Request 讨论` 


- **新增运行时类型检查**：George Hotz 宣布为 Tinygrad 增加了 `TYPED=1` 支持以进行 **运行时类型检查**，并指出在使用 `python3 test/test_ops.py` 测试时存在类型错误。
   - 一位用户在 [GitHub Pull Request](https://github.com/tinygrad/tinygrad/pull/6520) 中提到了一个潜在的修复方案，该方案解决了大部分类型错误，但仍有一个未解决。
- **在 AMD 上遇到测试失败**：一位用户报告称，在 nixpkgs 中尝试将 Tinygrad 从 **0.9.0 升级到 0.9.2** 时测试失败，并提到了一个与 `struct_kfd_ioctl_criu_args` 相关的 **AttributeError**。
   - 他们推测这是否可能是 **Kernel** 版本问题，因为该属性存在于 `/usr/include/linux/kfd_ioctl.h` 中，但在 Tinygrad 的 autogen 目录中却缺失，这令人困惑。
- **关于 GitHub 变更的讨论**：有人对 Tinygrad 中 **hip_ioctl 变更** 相关的潜在疏忽表示担忧，这可能在最近的 Pull Request 中被遗漏了。
   - 用户强调代码中需要一行特定的内容，而这行内容可能在 [Pull Request #5917](https://github.com/tinygrad/tinygrad/pull/5917) 引入的变更中被忽略了。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/6520">由 chenyuxyz 提交的修复 test_ops 类型问题的 Pull Request #6520 · tinygrad/tinygrad</a>：基本通过了 TYPED=1 python3 -m pytest -n=auto test/test_ops.py。最后一个测试专门设置了一个无效值来测试异常，为了忽略它我们需要导入 typeguard。并且为了得到一个...</li><li><a href="https://github.com/tinygrad/tinygrad/blob/518c022c29104d79c7a50ec41af5b7e6404da317/extra/hip_gpu_driver/test_kfd_2.py#L31)">tinygrad/extra/hip_gpu_driver/test_kfd_2.py (位于特定 commit) · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 micrograd？你会爱上 Tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/pull/5917">由 wozeparrot 提交的 hip_ioctl 变更 Pull Request #5917 · tinygrad/tinygrad</a>：特性：允许将处理器指定为环境变量；特性：引入 kfd_ioctl.h
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1285223898568331294)** (4 条消息): 

> - `Tinygrad Ecosystem Libraries`
> - `VRAM Spike Analysis`
> - `Tensor Modification Errors` 


- **关于 Tinygrad 生态库的咨询**：一位成员询问是否有人正在为 **tinygrad ecosystem** 开发库，并提到了像 **timm** 和 **torchvision** 这样的潜在候选对象。
   - *“你是否使用过 tinygrad 以确认此类库是否已实现或是否有必要？”* 另一位成员提出质疑，引发了进一步讨论。
- **了解 VRAM 分配峰值**：一位成员询问了识别 Tinygrad 操作期间导致 **VRAM allocation spikes**（VRAM 分配峰值）的最佳方法。
   - 这个问题强调了在 tinygrad 框架内需要诊断工具或方法来监控内存使用情况。
- **Tensor 修改代码中的错误**：一位用户报告了在运行涉及修改 tinygrad **Tensor** 的代码时出现的错误，特别是在尝试递增其元素时。
   - 他们链接到了 [GitHub 上的一个公开 issue](https://github.com/tinygrad/tinygrad/issues/6352)，该 issue 似乎在解决类似的问题，并指出了关于 **contiguous** 属性的编译器行为。



**提到的链接**：<a href="https://github.com/tinygrad/tinygrad/issues/6352)">Issues · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️ - Issues · tinygrad/tinygrad

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1284491127927476346)** (8 条消息🔥): 

> - `full_finetune_distributed recipe`
> - `Checkpoints management`
> - `Learning rate scheduling`
> - `CUDA vs CPU operations` 


- **关于 Checkpoints 管理的澄清**：为了在特定的 token 计数处实现 checkpoints，一位成员建议使用 `num_tokens` 字段跟踪处理的总 token 数，并过滤掉 padding tokens：[点击此处查看](https://github.com/pytorch/torchtune/blob/4fbe7b2d4956b3790c51d7a255c0040cf5c38fad/recipes/full_finetune_distributed.py#L622)。他们强调需要进行 **all gather** 以统计所有 **ranks** 的总数。
   - 对于 checkpoint 的保存，脚本中需要调整逻辑以确保其准确跟踪 token，特别是从保存的状态恢复（resume）时。
- **引入 Cosine Learning Rate Decay**：成员们讨论了如何利用 `torchtune.modules.get_cosine_schedule_with_warmup` 进行学习率的余弦衰减，尽管目前该功能仅集成在 **LoRA recipes** 中。建议在将该功能集成到 **full finetune recipe** 时，密切参考这些实现。
   - 建议在 recipe 设置中直接传递 step 数量，而不是从 epoch 数量推导，以适应 mid-epoch resume（周期中途恢复）的场景。
- **Token 处理中的 CUDA vs CPU 操作**：有人询问 token 操作（gather/reduce）是否需要在 **CUDA** 设备上执行，或者 **CPU** 进程是否足够。成员们确认 `num_tokens` 不是 **CUDA tensors**，并建议在 **CUDA** 设备上执行操作，因为 **CUDA** 和 **CPU** 进程之间没有直接映射。
   - 讨论强调虽然 **CUDA** 进程更可取，但在这种情况下 **CPU** 进程的效率仍存在不确定性。



**提到的链接**：<a href="https://github.com/pytorch/torchtune/blob/4fbe7b2d4956b3790c51d7a255c0040cf5c38fad/recipes/lora_finetune_distributed.py#L287-L288">torchtune/recipes/lora_finetune_distributed.py at 4fbe7b2d4956b3790c51d7a255c0040cf5c38fad · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.

  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1284254583127806074)** (6 messages): 

> - `在线打包支持 (Online Packing Support)`
> - `合并冲突已解决`
> - `CI GPU 测试失败`
> - `Cache Position 更新`
> - `urllib 版本不兼容` 


- **在 Iterable Datasets 上的在线打包计划**：团队计划在添加对 **iterable datasets** 的支持后立即转向在线打包（online packing）。
- **合并冲突已修复**：一名成员报告称他们已修复了 **merge conflicts**，并添加了大量测试以提高稳定性。
   - 他们计划明天进一步更新描述以增强清晰度。
- **对 CI GPU 测试失败的担忧**：CI 存在问题，特别是与 `test_eleuther_eval.py` 相关的 GPU 测试失败，这是由 **transformers.pipelines** 中的导入错误引起的。
   - 测试摘要显示有 504 个测试通过，但强调了一个阻止成功完成的重大错误。
- **即将进行的 KV Cache 更新**：一旦 KV Cache 更新完成，将实施有关 **cache position** 的更改，移除所有现有的 cache position 元素。
- **urllib 和 Requests 包不兼容**：一名成员指出测试失败可能源于 `urllib` 和 `requests` 包之间的 **版本不兼容**。
   - 他们建议将 `urllib>3` 进行固定（pinning）作为可能的修复方案，但承认由于该问题的间歇性，尚未进行测试。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1284624359612354732)** (5 messages): 

> - `艺术中的生成式 AI`
> - `Diffusion Illusions`
> - `实习机会`
> - `视错觉`
> - `图像数据集生成` 


- **生成式 AI 在几分钟内创作艺术**：一名成员展示了他们使用 NotebookLM 制作的作品，称其完全是在 **2 分钟** 内生成的。他们分享了 [YouTube 视频](https://youtu.be/kINTcf9rEJ4) 链接。
   - “真是个了不起的时代”是他们对生成式 AI 能力的热情评价。
- **Steve Mould 探索新的错觉**：一名成员分享了一个有趣的 YouTube 视频，题为《这种新型错觉非常难以制作》，讨论了 **使用 AI 生成的错觉**。视频可在 [此处](https://youtu.be/FMRi6pNAoag) 观看，并包含 Jane Street 实习的链接。
   - 他们指出，生成式 AI 可以创建在不同光线下看起来不同的图像。
- **Diffusion Illusions 的交互式游乐场**：一名成员提供了 [Diffusion Illusions 网站](https://diffusionillusions.com/) 的链接，该网站展示了使用 Diffusion 模型生成的交互式视错觉。该网站还展示了他们被 **SIGGRAPH 2024** 接收的项目，并链接到了 YouTube 演讲。
   - 作者包括 Ryan Burgert 和 Xiang Li 等人，强调了 Diffusion 模型在物理世界中的创新应用。
- **关于图像中文本放置的讨论**：一名成员询问了如何高效地在图像中插入文本以创建大型数据集的方法。他们正在寻找将此过程扩展到填充 **数百万张图像** 的方法。
   - 这一查询突显了对自动化创建文本嵌入图像数据集以用于潜在应用的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://diffusionillusions.com/">Diffusion Illusions: Hiding Images in Plain Sight</a>：未找到描述</li><li><a href="https://youtu.be/FMRi6pNAoag">This new type of illusion is really hard to make</a>：在以下地址了解更多关于 Jane Street 实习的信息：https://jane-st.co/internship-stevemould。生成式 AI 可用于制作在不同光线下看起来不同的图像...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/)** (1 messages): 

blanchon.jl：那太棒了！
  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1284418783506599947)** (4 messages): 

> - `预训练 VLMs`
> - `异常检测技术` 


- **讨论预训练 VLM 的计算需求**：一名成员询问关于使用预训练 **视觉语言模型 (VLMs)** 的事宜，但表达了对 **缺乏计算资源** 的担忧。
   - 另一名成员指出，这些模型 **需要强大的计算能力** 才能有效运行。
- **澄清异常检测的数据要求**：一名成员询问 **异常检测 (anomaly detection)** 应该在日志上进行还是在实际的 **时间序列数据 (time-series data)** 上进行。
   - 他们分享了几种处理时间序列数据的方法，包括 **Transformer 模型**、**卡尔曼滤波 (Kalman Filters)** 和 **孤立森林 (isolation forests)**，并建议使用 **z-scores** 进行误差评估。


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1285071731501039638)** (1 条消息): 

> - `模型 function calling bug` 


- **模型在 function calling 方面表现不佳**：一位成员提出担忧，认为该模型目前仅具备聊天能力，在相关性（relevance）方面得分为 **1**，且无法调用任何函数，导致其他能力的得分为 **0**。
   - *该 bug 显著限制了模型的功能。*
- **函数性能评估**：讨论强调，1 分的不相关得分（irrelevance score）表明模型在有效执行任何 function calls 的能力上存在严重故障。
   - *无法执行函数可能会损害用户体验并偏离预期。*


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1284308471528947805)** (2 条消息): 

> - `Function Call 错误`
> - `AST 解码问题` 


- **模型生成对话而非 Function Call**：模型输出了对话格式而非执行 function call，导致 model handler 在处理响应时出现问题。
   - *据指出，这会导致该尝试被自动标记为错误。*
- **无效语法触发 AST 解码器失败**：产生了一条指示“Invalid syntax”的错误消息，导致无法解码抽象语法树（AST）。
   - 该问题被归类为 'ast_decoder:decoder_failed'，意味着在解释模型输出时存在严重问题。


  

---



---



---



---



---



{% else %}


> 邮件中已截断完整的频道细分内容。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}