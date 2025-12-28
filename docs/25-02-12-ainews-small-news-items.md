---
companies:
- openai
- ollama
- mistral
- perplexity
- cerebras
- alibaba
- groq
- bytedance
date: '2025-02-13T00:10:12.213344Z'
description: '**OpenAI** 宣布了关于 **GPT-4.5 (Orion)** 和 **GPT-5** 的计划，其中 GPT-5 将集成 **o3**
  模型，并在免费层级中提供无限次的聊天访问。**DeepSeek R1 Distilled Qwen 1.5B** 在数学基准测试中的表现优于 OpenAI 的
  **o1-preview**，而 **ModernBERT 0.3b** 在无需微调的情况下，在 MMLU 测试中超过了 **Qwen 0.5b**。**Mistral**
  和 **Perplexity** 采用了 **Cerebras** 硬件，实现了 10 倍的性能提升。OpenAI 的 **o3** 模型在 2024 年国际信息学奥林匹克竞赛（IOI）中荣获金牌。合作伙伴关系方面包括
  **Qwen** 与 **Groq** 的合作。尼日利亚和全球南方国家出现了显著的 **RLHF**（基于人类反馈的强化学习）活动，预计 **字节跳动（Bytedance）**
  在 AI 领域的地位将很快崛起。“**GPT5 就是你所需的一切。**”'
id: 117d6739-42ce-4ce5-afeb-d11cee530906
models:
- gpt-4.5
- gpt-5
- deepseek-r1-distilled-qwen-1.5b
- o1-preview
- modernbert-0.3b
- qwen-0.5b
- o3
original_slug: ainews-small-news-items
people:
- jeremyphoward
- arankomatsuzaki
- sama
- nrehiew_
- danhendrycks
- akhaliq
title: '根据语境，可以翻译为：


  1. **简讯** (最常用的术语)

  2. **短讯**

  3. **新闻简报**

  4. **零星新闻**

  5. **小条新闻**'
topics:
- math
- benchmarking
- fine-tuning
- model-performance
- reinforcement-learning
- model-architecture
- partnerships
- funding
---

<!-- buttondown-editor-mode: plaintext -->**GPT5 is all you need.**

> 2025年2月11日至2月12日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**211** 个频道，**5266** 条消息）。预计节省阅读时间（以 200wpm 计算）：**497 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

没有头条故事，但有很多酷炫的更新：

- OpenAI 分享了新的 [model spec](https://x.com/OpenAI/status/1889781541259321466)，并表示 [gpt4.5 即将到来，gpt5 将整合 o3+](https://x.com/sama/status/1889755723078443244?s=46&t=JE84TqLviekDnEt8MAT-Eg)
- [glean 发布了 agents](https://x.com/glean/status/1889706504812683728)
- 来自 [Harvey](https://x.com/winstonweinberg/status/1889713028234416371?s=46)、[FAL](https://x.com/glennsolomon/status/1889717350456315960?s=46) 和 [Scaled Cognition](https://x.com/scaledcognition/status/1889721166421479751?s=46) 的融资公告
- [Jeff Dean 和 Noam Shazeer 做客 Dwarkesh 访谈](https://x.com/swyx/status/1889810524696891903)

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**模型与性能**

- **DeepSeek R1 Distilled Qwen 1.5B 在数学基准测试中超越 OpenAI 的 o1-preview**：[@ollama](https://twitter.com/ollama/status/1889496833875124735) 宣布发布 **DeepScaleR**，这是一个 Ollama 模型，是 **Deepseek-R1-Distilled-Qwen-1.5B** 的微调版本。它在流行的数学评估中**优于 OpenAI 的 o1-preview**，且仅用了 **1.5B 参数**。[@jeremyphoward](https://twitter.com/jeremyphoward/status/1889435769959489800) 指出 DeepScaleR 在 **MMLU Pro 上也击败了 Qwen**，并质疑此类复杂领域是否真的需要 decoder 模型。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1889522980096712945) 强调 **OpenAI 的 o3 在 Codeforces 上达到了 99.8 百分位**。
- **ModernBERT 0.3b 在没有特定任务微调的情况下，在 MMLU 上优于 Qwen 0.5b**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1889434481519632505) 表示，encoder-only 的 **ModernBERT 0.3b 在 MMLU 上击败了 Qwen 0.5b**，且无需特定任务的微调，这表明它可能开启**语言模型的新革命**。
- **Mistral 和 Perplexity 正在采用 Cerebras 以获得 10 倍的性能提升**：[@draecomino](https://twitter.com/draecomino/status/1889430107288416340) 宣布 **Mistral 和 Perplexity** 正在转向 **Cerebras**，声称这使其客户产品比竞争对手**快 10 倍**。[@draecomino](https://twitter.com/draecomino/status/1889434428306497667) 还指出，自他上一篇帖子以来，**两家由 Nvidia 资助的最大 AI 初创公司**现在也在使用 Cerebras。
- **OpenAI 的 o3 模型在 IOI 2024 中获得金牌**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1889522974467957033) 和 [@_akhaliq](https://twitter.com/_akhaliq/status/1889523662732042610) 分享了 **OpenAI** 的论文 "Competitive Programming with Large Reasoning Models"，强调其 **o3 模型**在 **2024 年国际信息学奥林匹克竞赛 (IOI)** 中获得了**金牌**。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1889517116816244995) 进一步详细说明，**o3 超越了像 o1-ioi 这样的专业流水线**，且无需手工设计的推理启发式方法，并在更宽松的约束下运行。
- **Qwen 与 Groq 的合作伙伴关系**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1889473940894130605) 通过一条简单的表情符号帖子暗示了 **Qwen 与 Groq** 之间的合作伙伴关系。
- **来自 OpenAI 的 GPT-4.5 和 GPT-5 路线图**：[@sama](https://twitter.com/sama/status/1889755723078443244) 分享了 **OpenAI 路线图更新**，透露计划将 **GPT-4.5 (Orion)** 作为其最后一个非思维链（non-chain-of-thought）模型发布，并将 **GPT-5 作为一个集成 o3 等技术的系统**发布。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1889756903187829107) 和 [@stevenheidel](https://twitter.com/stevenheidel/status/1889757357908836654) 总结了这些观点，指出 ChatGPT 免费版中的 **GPT-5** 将拥有**无限的聊天访问权限**。[@nrehiew_](https://twitter.com/nrehiew_/status/1889757485755416782) 评论道，这种将 **GPT-5 作为系统**的方法可能会在模型评估方面**拉大学术界与工业界之间的差距**。
- **RLHF 从业者在尼日利亚和全球南方国家大量存在**：[@DanHendrycks](https://twitter.com/DanHendrycks/status/1889483790638317774) 指出，来自**尼日利亚**以及全球南方其他国家的 **RLHF 从业者**大量存在。
- **字节跳动预计很快将在 AI 领域崭露头角**：[@agihippo](https://twitter.com/agihippo/status/1889583723730829687) 预测，目前在 AI 领域尚不突出的**字节跳动 (Bytedance)** 很快就会变得引人注目。
- **使用 FastHTML 和 MonsterUI 构建的应用易于构建和维护**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1889430719988113911) 称赞 **FastHTML**、**htmx** 和 **MonsterUI** 能够让开发者快速编写、易于维护且使用体验极佳的应用。
- **DeepScaleR，一个 1.5B 参数的模型，利用 RL 超越了 OpenAI 的 o1-preview**：[@_philschmid](https://twitter.com/_philschmid/status/1889592742088515630) 详细介绍说，**DeepScaleR** 作为一个通过强化学习（RL）微调的 **1.5B 参数模型**，在数学基准测试中**优于 OpenAI 的 o1-preview**，强调了 RL 即使对于较小模型也具有有效性，并使用了简单的二元奖励函数。
- **只有离线 RL 专家才理解在线 RL 的重要性**：[@shaneguML](https://twitter.com/shaneguML/status/1889505192229609864) 表示，**只有那些深入研究过离线 RL (offline RL) 的人**才真正理解**在线 RL (online RL) 的重要性**。

**行业与商业**

- **Mistral 和 Perplexity 正在采用 Cerebras 以实现 10 倍的性能提升**：[@draecomino](https://twitter.com/draecomino/status/1889430107288416340) 宣布 **Mistral 和 Perplexity** 正在转向 **Cerebras**，声称这使其客户产品比竞争对手**快 10 倍**。[@draecomino](https://twitter.com/draecomino/status/1889434428306497667) 还指出，自他上一条推文以来，**两家由 Nvidia 资助的最大的 AI 初创公司**现在也在使用 Cerebras。
- **Figure 在二级市场备受青睐**：[@adcock_brett](https://twitter.com/adcock_brett/status/1889743077323272442) 分享道，**Figure** 是上个月二级市场中**需求量排名第 9 的公司**，并指出投资者的需求“高得离谱”。
- **Perplexity 旨在达成 TikTok 交易**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1889577617885380613) 提到他将“继续狂喝红牛以促成 **TikTok 交易**”。
- **Perplexity 与法国 Bouygues Telecom 合作**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1889724444894957788) 宣布与 **Bouygues Telecom** 建立合作伙伴关系，在法国分发 **Perplexity**，这进一步扩大了其全球合作伙伴网络。
- **Perplexity 推出财经仪表盘 (Finance Dashboard)**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1889775543635050985) 推广了 **Perplexity 的 Finance Dashboard**，在一个地方提供股票、收益、市场波动和摘要。
- **Perplexity 在巴黎的用户采用率极高**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1889635609582444880) 和 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1889562246004494405) 描述了 **Perplexity 在巴黎的高用户采用率**，有人在街上拦住他表达对该 App 的喜爱，他还遇到了一些正在使用 Perplexity 的热心学生。
- **Together AI 为 DeepSeek-R1 部署推出推理集群 (Reasoning Clusters)**：[@togethercompute](https://twitter.com/togethercompute/status/1889743684977168547) 宣布推出 **Together Reasoning Clusters**，这是专为大规模、低延迟推理工作负载构建的专用计算资源，扩展了其用于在生产环境中部署 **DeepSeek-R1** 等推理模型的 Serverless API。
- **Klarna 的 AI 助手利用 LangGraph 和 LangSmith 扩展了客户支持**：[@LangChainAI](https://twitter.com/LangChainAI/status/1889728750415479161) 和 [@hwchase17](https://twitter.com/hwchase17/status/1889758528232898844) 强调了 **Klarna** 如何使用 **LangGraph 和 LangSmith** 为 **8500 万活跃用户**扩展客户支持，将解决时间缩短了 **80%**，并实现了 **70%** 任务的自动化。

**Research & Papers**

- **OpenAI 发布《Competitive Programming with Large Reasoning Models》论文**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1889522974467957033) 和 [@_akhaliq](https://twitter.com/_akhaliq/status/1889523662732042610) 分享了 **OpenAI** 的论文《Competitive Programming with Large Reasoning Models》，强调其 **o3 model** 在 **2024 International Olympiad in Informatics (IOI)** 中获得了金牌。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1889517116816244995) 进一步详细说明，**o3** 在没有手工设计的推理启发式方法且在放宽约束的情况下，超越了像 **o1-ioi** 这样的专门 Pipeline。
- **Google DeepMind 发布《Scaling Pre-training to One Hundred Billion Data for Vision Language Models》**：[@_akhaliq](https://twitter.com/_akhaliq/status/1889526316673732753) 和 [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1889520451501900000) 分享了 **Google DeepMind** 的论文《Scaling Pre-training to One Hundred Billion Data for Vision Language Models》，介绍了 **WebLI-100B**，这是一个拥有 **1000 亿个图像-文本对** 的数据集，展示了超越传统 Benchmark 的优势，特别是在 **文化多样性和多语言能力** 方面。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1889521157482959067) 也强调了该数据集和研究发现。
- **关于互联网规模 Web Agent 训练的新论文《InSTA》**：[@rsalakhu](https://twitter.com/rsalakhu/status/1889492471630946662) 宣布了一篇关于 **InSTA** 的新论文，这是一个用于在 **15 万个不同网站** 上进行 **互联网规模 Web Agent 训练** 的 Pipeline，无需人工标注，在使用 **Llama 3.1 70B Agent** 的情况下，在有害内容检测和任务完成等任务中达到了与人工标注员相当的性能。
- **Scale AI 发布关于 LLM “Jailbreak to Jailbreak” 的研究**：[@goodside](https://twitter.com/goodside/status/1889492446750364103) 分享了来自 **Scale AI** 的关于 “Jailbreak to Jailbreak” 的新研究，利用 **经过安全训练的 LLM 的 Jailbreaking** 来开发针对其他 LLM 的 Jailbreak 方法。
- **关于用于 Masked Token Infilling 的 MARIA 模型的论文**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1889542518465077557) 重点介绍了一篇关于 **MARIA** 的论文，这是一种混合自回归和 Masked Language Model，用于 **Masked Token Infilling**，其性能优于离散扩散模型，并通过 KV caching 提供更快的推理速度。
- **Microsoft Research 展示用于科学发现的 “NatureLM”**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1889518044273385672) 分享了 **Microsoft Research** 关于 **NatureLM** 的论文，这是一种基于序列的科学基础模型，用于 **科学发现**，能够使用文本指令生成和优化分子、蛋白质、RNA 和材料。
- **Meta AI 展示用于从单张图像生成高分辨率多视角人物的 “Pippo”**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1889515688647373113) 分享了 **Meta AI** 关于 **Pippo** 的论文，该模型可以在一次前向传递中从单张照片生成 **1K 分辨率、多视角、影棚级质量的人物图像**。
- **论文研究使用 RLSP 技术在 LLM 中出现的涌现思维**：[@omarsar0](https://twitter.com/omarsar0/status/1889697727703134544) 讨论了一篇关于《On the Emergence of Thinking in LLMs》的论文，探索了使用名为 **RLSP** 的训练后技术在 **LLM 中的推理能力**，展示了回溯和探索等涌现行为。
- **关于用于长上下文推理的 Large Memory Models (LM2) 的论文**：[@omarsar0](https://twitter.com/omarsar0/status/1889681118913577345) 总结了一篇关于 **Large Memory Models (LM2)** 的论文，这是一种基于 Transformer 的架构，带有专用内存模块以增强 **长上下文推理**，在内存密集型 Benchmark 上优于基准模型。
- **关于知识蒸馏的 TAID 论文被 ICLR2025 接收**：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1889708905280028809) 宣布他们的论文《TAID: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models》已被接收为 **ICLR2025 的 Spotlight Paper**，介绍了一种新的知识蒸馏方法。

**工具与应用**

- **Ollama 发布 DeepScaleR 模型**：[@ollama](https://twitter.com/ollama/status/1889496833875124735) 宣布发布 **DeepScaleR**，这是一个 Ollama 模型，是 **Deepseek-R1-Distilled-Qwen-1.5B** 的微调版本，在流行的数学评估中**超越了 OpenAI 的 o1-preview**，且仅使用了 **1.5B 参数**。
- **LangChain 为多智能体系统发布 LangGraph Supervisor**：[@LangChainAI](https://twitter.com/LangChainAI/status/1889717269510394365) 推出了 **LangGraph Supervisor**，这是一个轻量级库，用于使用 LangGraph 构建**分层多智能体系统**，其特点是使用一个 Supervisor Agent 来协调专业 Agent 和基于工具的移交。
- **Perplexity 推出金融仪表盘**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1889775543635050985) 推广了 **Perplexity 的金融仪表盘**，在一个地方提供股票、收益、市场动态和摘要。
- **带有股价更新功能的 AI 金融 Agent**：[@virattt](https://twitter.com/virattt/status/1889458443515265066) 宣布了其 **AI 金融 Agent** 的更新，现在可以显示股票价格、市值、成交量和历史价格，代码开源且无需注册。
- **用于编程任务模型偏好投票的 SWE Arena**：[@terryyuezhuo](https://twitter.com/terryyuezhuo/status/1889584039629078963) 重点介绍了 **SWE Arena**，这是一个用户在使用 **o3-mini** 等前沿模型编程时，可以**为他们偏好的模型投票**的平台。
- **Aomniapp Agent 编排系统 Beta 版发布**：[@dzhng](https://twitter.com/dzhng/status/1889547813559951533) 宣布了 **Aomniapp** 的 Beta 测试版，这是一个 **Agent 编排系统**，允许用户通过一个提示词生成数百个 Agent。
- **Google DeepMind Gemini API 密钥设置快速简便**：[@_philschmid](https://twitter.com/_philschmid/status/1889689838464516228) 详细介绍了如何在 30 秒内创建 **Google DeepMind Gemini API 密钥**，仅需 Google 账号，无需信用卡或 Google Cloud 账号。
- **DeepSeek R1 生成魔方可视化器和求解器**：[@_akhaliq](https://twitter.com/_akhaliq/status/1889736413429559444) 展示了 **DeepSeek R1** 使用 Three.js 在单个 HTML 文件中生成**魔方可视化器和求解器**，具有交互式控制和动画功能。
- **RepoChat 支持与 GitHub 仓库聊天**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1889741525808193635) 宣布了 **RepoChat 博客和数据集发布**，重点介绍了他们的工具，该工具允许用户**与他们的 GitHub 仓库聊天**，已收集了超过 1.1 万条对话。
- **用于文本转 Web 应用的 Text2web Arena**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1889496847708045496) 推广了 **Text2web Arena**，这是一个尝试文本转 Web 应用的平台，展示了 **Claude 3.5 Sonnet** 使用 Three.js 生成 3D 场景。

**Development & Coding**

- **2025 年的软件库应包含 context.txt 以用于 LLM 代码生成**：[@vikhyatk](https://twitter.com/vikhyatk/status/1889540437557518843) 建议在 **2025 年发布软件库**时需要包含一个 **context.txt** 文件，以便用户粘贴到 LLM 中以生成正确的代码。
- **2025 年的手动编码与 2024 年 Web 应用的汇编语言相比**：[@vikhyatk](https://twitter.com/vikhyatk/status/1889597476895662336) 评论说，**在 2025 年手动编写代码**将就像**在 2024 年编写汇编语言来构建 Web 应用**一样，暗示 AI 驱动的代码生成将成为主流。
- **在复杂任务中偏好 C++ 而非脚本**：[@MParakhin](https://twitter.com/MParakhin/status/1889428158421819825) 表达了在复杂任务中对 **C++** 的偏好，而非脚本语言，原因是其速度和可调试性，并使用 `system()` 来满足脚本需求。
- **针对 MLA 算子的 DeepSeek CPU/GPU 混合推理**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1889531203742466250) 强调了 **DeepSeek 的 CPU/GPU 混合推理**方法，用于其计算密集型的 MLA 算子，将繁重的计算卸载到 GPU 以提升性能。
- **用于微调的视频数据集策选工具发布**：[@RisingSayak](https://twitter.com/RisingSayak/status/1889632398465228998) 宣布发布**用于策选小型且高质量视频数据集的工具**，用于微调，灵感来自 SVD 和 LTX-Video，解决了视频微调中缺乏良好数据策选流水线的问题。

**Humor & Meta**

- **Meme 总结了 OpenAI 的 o3 论文**：[@polynoamial](https://twitter.com/polynoamial/status/1889541408065028421) 分享了一个 **meme**，很好地总结了《Competitive Programming with Large Reasoning Models》这篇论文。
- **AI 现状的 meme**：[@giffmana](https://twitter.com/giffmana/status/1889424405350002991) 发布了一个描绘“目前 AI 现状，或多或少”的 meme。
- **关于斯大林格勒（Stalingrad）的幽默历史问题**：[@kipperrii](https://twitter.com/kipperrii/status/1889440548848804252) 开玩笑地请求对**斯大林格勒**的历史解释，并指出维基百科上看似矛盾的死亡人数数据。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. LLM 中革命性的潜空间（Latent Space）推理**

- **[一篇新论文证明 LLM 可以在潜空间中“思考”，有效地将内部推理与可见的上下文 Token 解耦。这一突破表明，即使是较小的模型也可以在不依赖广泛上下文窗口的情况下实现卓越的性能。](https://huggingface.co/papers/2502.05171)** ([Score: 1218, Comments: 261](https://reddit.com/r/LocalLLaMA/comments/1inch7r/a_new_paper_demonstrates_that_llms_could_think_in/))：最近的一篇论文揭示了 **Large Language Models (LLMs)** 可以在潜空间中进行推理，从而使它们能够将内部推理与可见的上下文 Token 分离开来。这一进展意味着较小的模型可能在不依赖大上下文窗口的情况下提供令人印象深刻的结果。
  - 讨论强调了在**潜空间（latent space）**中进行推理以提高模型性能的潜力，并与 **Chain-of-Thought (CoT)** 等现有方法进行了比较，还提到了 **Meta 的 COCONUT 方法**。人们对安全性和透明度表示担忧，因为潜空间推理可能导致模型以难以用言语表达的方式进行“思考”，从而使对齐（alignment）和可解释性工作变得复杂。
  - 该论文在 **AMD mi250x** 上的测试以及 **ROCm 软件栈**的使用值得关注，这挑战了 **Nvidia** 在 AI 研究中的主导地位。人们对这种方法是否可以有效扩展感兴趣，同时也对作者之前的作品持怀疑态度，并关注在实践中实施此类方法的挑战。
  - 对话涉及了 AI 推理和意识的更广泛主题，引用了 **Daniel Kahneman** 的《思考，快与慢》以及直觉系统与逻辑推理系统之间的区别。探讨了模型“不经思考地思考”或“脱离语言思考”的潜力，并提供了 **Hugging Face** 资源的链接，以便进一步探索论文的概念。


**主题 2. AMD 在 AI 硬件竞争中的战略举措**

- **[据报道，AMD 正在开发针对游戏市场的 Radeon RX 9070 XT GPU，配备 32GB 显存](https://videocardz.com/newz/amd-reportedly-working-on-gaming-radeon-rx-9000-gpu-with-32gb-memory)** ([Score: 383, Comments: 96](https://reddit.com/r/LocalLLaMA/comments/1inoui5/amd_reportedly_working_on_gaming_radeon_rx_9070/))：据报道，**AMD** 正在开发针对游戏市场的 **Radeon RX 9070 XT GPU**，配备 **32GB 显存**。鉴于其巨大的显存容量，这一进展暗示了对 AI 应用的潜在影响，可能会增强 AI 驱动任务的性能。
  - **ROCm vs CUDA**：用户强烈支持将 **ROCm** 作为 **CUDA** 的开源替代方案，许多用户认为像 **RX 9070 XT** 这样的大显存 GPU 可能会推动社区对 ROCm 的改进，从而更好地与 **NVIDIA** 的生态系统竞争。一些用户对 CUDA 的主导地位表示不满，将其与 **OpenAI** 在 LLM 领域的影响力相提并论。
  - **定价与性能比较**：讨论强调了 **RX 9070 XT** 极具竞争力的潜在定价（传闻低于 **$1000**），这是对抗 NVIDIA 产品（如 **RTX 5090**）的一个重要因素。用户正在争论显存容量与显存带宽之间的权衡，并指出 **7900 XTX** 提供了一个具有合理性能的高性价比替代方案。
  - **社区与来源可靠性**：人们对 GPU 泄密消息的可靠性持怀疑态度，正如对一个使用 Photoshop 处理过的头像的来源进行的幽默批评所证明的那样。尽管如此，一些社区成员为这些来源的一致性担保，强调了 GPU 新闻的投机性质。


**主题 3. Project Digits：Nvidia 在 AI 工作站领域的下一个重大举措**

- **[PNY 演示文稿中关于 Project Digits 的一些细节](https://www.reddit.com/gallery/1inos01)** ([Score: 128, Comments: 86](https://reddit.com/r/LocalLLaMA/comments/1inos01/some_details_on_project_digits_from_pny/)): Nvidia 的 **Project Digits** 由 PNY 的 DGX EMEA 负责人展示，重点介绍了 **DDR5x memory**（初始容量 128GB）、带有 Mellanox 芯片的双端口 **QSFP networking** 以及全新的 ARM 处理器。该工作站售价约 **$3,000**，以其软件栈和基于 Ubuntu 的 OS 为特色，目标受众为大学和研究人员，其性能显著强于 Jetson 系列产品，但并非多 GPU 工作站的替代品。
  - **内存带宽担忧 (Memory Bandwidth Concerns)**：几位评论者对 Nvidia 未披露 Project Digits 的 **memory bandwidth** 表示沮丧，推测其约为 **270 GB/s**。这种信息的缺失被视为一个潜在的危险信号，一些人认为这是在 **GTC** 披露更多细节前维持热度的策略。
  - **目标受众与用途**：Project Digits 被定位为面向 **researchers and universities** 的紧凑型便携式工作站，旨在开发和实验新的 AI 架构，而非取代多 GPU 工作站。它被描述为进入 Nvidia 生态系统的门户，使研究人员能够轻松过渡到更强大的 **DGX machines** 以进行大型项目。
  - **战略定位与市场影响**：该产品被视为 Nvidia 吸引下一代 AI/ML 工程师的战略举措，尽管人们对其 **niche market** 地位和潜在的快速过时表示担忧。讨论强调了 Nvidia 通过软件支持和生态系统整合来维持其市场主导地位的重点，而一些用户则对 Nvidia 的长期战略及其对消费级产品的影响表示怀疑。


**Theme 4. Phi-4 在 AI 创意方面的非常规方法**

- **Phi-4，但经过剪枝且不安全** ([Score: 112, Comments: 21](https://reddit.com/r/LocalLLaMA/comments/1inn034/phi4_but_pruned_and_unsafe/)): **Phi-Lthy4** 是 **Phi-4** 的剪枝版本，旨在通过移除不必要的数学层来增强角色扮演能力，最终模型拥有 **11.9B parameters**。该模型经过使用 **1B tokens** 的为期两周的微调过程，在创意写作和角色扮演方面表现出色，证明是一个具有低拒绝率和强角色卡遵循能力的独特助手。尽管采用了非常规方法，但其效果出奇地好，详见 [Hugging Face](https://huggingface.co/SicariusSicariiStuff/Phi-lthy4)。
  - **模型大小与性能**：**Phi-Lthy4** 是 Phi-4 的剪枝版本，拥有 **11.9B parameters**，在创意写作和角色扮演方面表现优异。讨论中涉及了该模型在不同量化版本下的大小，其中 **IQ4_XS quant** 版本为 **6.5GB**，表明它可以在 **8GB** 内存上运行。
  - **模型合并与变体**：**Environmental-Metal9** 对将 Phi 与 **Mistral** 合并表示兴趣，因为其散文质量很高。**Sicarius_The_First** 在 [Hugging Face](https://huggingface.co/SicariusSicariiStuff/Redemption_Wind_24B) 上分享了一个相关项目 **Redemption Wind 24B**，突显了结合不同模型优势的潜力。
  - **基准测试与写作风格**：与通常作为近期论文微调基础模型的 **Qwen** 相比，**Phi series** 通常不用于基准测试。然而，Phi 因其独特的写作风格而受到关注，被描述为“冷静客观但不令人尴尬的草率”，受到了一些用户的青睐。


## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. OpenAI 的新模型：GPT-4.5 'Orion' 与 Chain-of-Thought 集成**

- **[OpenAI 关于 GPT-4.5 和 GPT-5 的路线图更新](https://i.redd.it/j6j5diamdrie1.jpeg)** ([Score: 503, Comments: 106](https://reddit.com/r/OpenAI/comments/1inz75h/openai_roadmap_update_for_gpt45_gpt5/)): **OpenAI** 的路线图更新由 **Sam Altman** 在 Twitter 上分享，概述了 **GPT-4.5**（内部代号为 **Orion**）和 **GPT-5** 的计划。此次更新强调了简化产品线、增强用户体验以及统一模型系列的努力，**GPT-5** 将集成到 **ChatGPT** 和 **API** 中，提供分级的访问权限，包括为 Pro 订阅者提供更高智能的设置。
  - 用户对 **OpenAI 的分级智能模型**表示担忧，认为这可能会使系统复杂化并减少用户的选择权，一些用户更倾向于针对特定任务手动选择模型，例如使用 **o3-mini** 进行编程或咨询健康相关问题。另一些人则认为，自动模型选择可以通过简化非专家的决策来提升用户体验。
  - 讨论中还涉及对 **OpenAI 成本节约策略**的怀疑，例如通过自动化模型选择来降低运行成本，这可能会限制透明度和用户控制权。一些用户赞赏 **GPT-4.5** 和 **GPT-5** 自主决定何时采用 **chain-of-thought** 推理的想法，而另一些人则担心这会导致“黑盒（black box）”系统。
  - 人们对运行在 **GPT-3** 或 **GPT-3.5** 等旧模型上的**外部聊天机器人**的未来感到好奇，一些用户担心它们可能会过时。然而，目前没有明确迹象表明 OpenAI 会很快停用这些 **API**，尽管有人推测无限期支持它们在经济上可能并不可行。


**Theme 2. DeepSearch Goes Mainstream: Plus and Free User Access**

- **[DeepSearch 即将面向 Plus 和免费用户开放](https://i.redd.it/9zwkrb49uqie1.png)** ([Score: 555, Comments: 97](https://reddit.com/r/OpenAI/comments/1inwhg1/deepsearch_soon_to_be_available_for_plus_and_free/)): **DeepSearch** 是 **Sam Altman** 在 Twitter 对话中提到的一项功能，很快将面向 **ChatGPT Plus 用户**（每月 10 次）和**免费用户**（2 次）开放。一位用户强调了该功能的巨大价值，估计其价值约为**每月 1,000 美元**，并指出它对认知参与度有显著影响。
  - 几位评论者批评了 **DeepSearch** 每月价值 **$1,000** 的说法，认为这不切实际，可能是一种被称为“锚定（anchoring）”的策略，旨在让未来的定价显得更低。**Fumi2014** 提到，该功能作为研究工具不够全面，因为它依赖于公开的网页数据，排除了许多学术资源。
  - **EastHillWill** 等人讨论了 **DeepSearch** 的潜在成本，估计每次使用的成本约为 **$0.50**。有人建议提供更灵活的定价方案，例如提供 **20 次免费使用**，之后对额外使用进行收费，以提供更好的价值。
  - 用户对不同层级的 **DeepSearch** 可用性和定价结构表示担忧，一些用户对 **ChatGPT Team 账户**被排除在外感到沮丧，并讨论了通过创建多个账号来规避使用限制的可能性，尽管这需要多个手机号码。


**Theme 3. Grok 3 Performance Leak and xAI Resignation Fallout**

- **[xAI 离职事件](https://i.redd.it/nkcfuep8enie1.png)** ([评分: 721, 评论: 174](https://reddit.com/r/OpenAI/comments/1ink8o2/xai_resignation/)): Benjamin De Kraker 宣布从 **xAI** 辞职，并指出被迫删除一条关于 **Grok 3** 的声明是主要原因。他批评公司将其观点贴上“机密信息”的标签，并对 xAI 在言论自由方面的立场表示失望，同时反思了自己的未来计划。
  - 许多评论者认为 **Benjamin De Kraker 公开披露** 关于 **Grok 3 性能** 的信息是不恰当的，因为这涉及利用内部信息将该模型与竞争对手进行排名。这被视为违反了保密协议，几位用户认为，由于潜在的财务和声誉影响，此类行为可能导致合理的解雇。
  - 讨论强调，**公司政策通常禁止未经授权讨论**未发布的产品，特别是涉及比较评估时。评论者指出，即使某些信息是公开的，通常也要求员工遵守严格的协议，不得在没有明确许可的情况下公开推测或分享内部见解。
  - 舆论一致认为，**De Kraker 将此问题定性为侵犯言论自由**是不妥的。评论表明，他的行为更多是违反了公司保密规定，而非对个人表达的侵害，一些用户指出，其他公司可能会更严厉地处理这种情况。


**主题 4. OpenAI 多模态模型：o1, o3-mini, 和 o3-mini high**

- **OpenAI 悄然推出：o1, o3-mini 和 o3-mini high 现已支持多模态。** ([评分: 393, 评论: 101](https://reddit.com/r/OpenAI/comments/1inoi6b/openai_silently_rolls_out_o1_o3mini_and_o3mini/)): **OpenAI** 悄悄为其模型 **o1**、**o3-mini** 和 **o3-mini high** 引入了多模态功能，使它们能够处理图像和文件。这次更新因其扩展的功能而受到了惊喜和热情的关注。
  - 用户报告了在不同平台上使用多模态能力的各种体验，一些用户能够在 **iOS** 和 **网页版** 上上传图像和文件，而其他用户，特别是 **桌面端** 以及 **波兰** 和 **亚洲** 等特定地区的用户，尚未收到更新。**o3** 上的 **PDF 上传** 被强调为一个重要功能，尽管一些人表达了对 PDF 的 API 支持的渴望。
  - 围绕哪些模型支持这些功能存在困惑和讨论，用户注意到 **o1** 支持文件上传，但 **o3-mini** 和 **o3-mini high** 在桌面版本上尚未显示此功能。一些用户已经使用 **o1 pro** 进行图像上传有一段时间了，正如 **YouTube demo** 中展示的那样。
  - 这些功能的推出似乎并不一致，不同地区和平台的用户报告了不同级别的访问权限，引发了关于在项目中使用 **4o** 以外模型的可用性和潜力的讨论。


---

# AI Discord 摘要

> 由 o1-preview-2024-09-12 生成的摘要之摘要之摘要

**主题 1: OpenAI 揭晓 GPT-5 并全面开放 o1 和 o3**

- **OpenAI 在 GPT-5 上押大注：不再在模型命名上折腾！** OpenAI 宣布即将发布 **GPT-4.5** 和 **GPT-5**，旨在统一其产品线，让 AI 对用户来说“好用就行”，正如 [Sam Altman 的推文](https://x.com/sama/status/1889755723078443244)所言。GPT-5 将整合多种技术，并提供给具有不同智能水平的免费层级用户。

- **OpenRouter 向大众开放 OpenAI 的 o1 和 o3！** **OpenAI 的 o1 和 o3** 推理模型现已对所有 [OpenRouter](https://openrouter.ai/) 用户开放，无需 BYOK，并为之前的 Key 用户提升了速率限制（Rate Limits），详情见[此处公告](https://x.com/OpenRouterAI/status/1889708759355691327)。这些模型现在支持网页搜索，扩大了它们的实用性并优化了用户体验。

- **社区对 OpenAI 策略转变的欢呼（与嘲讽）** 社区对 OpenAI 的路线图更新反应不一，既有兴奋也有怀疑。虽然有些人对简化的产品线感到兴奋，但另一些人则质疑放弃非推理模型的举动。讨论凸显了对 AI 发展方向的期待与担忧。

**主题 2: GRPO 为 AI 模型赋能，性能飙升**

- **GRPO 集成之痛：模型微调不适合胆小者！** AI 爱好者在将 **GRPO** 与 **Mistral** 和 **Llama** 等模型集成时遇到了挑战，分享了见解并指出了特殊 Token（如 *<thinking>*）的奇特之处。尽管存在障碍，社区还是分享了资源，例如一个[有用的 Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb#scrollTo=vITh0KVJ10qX)，以解决实现过程中的问题。

- **Tulu Pipeline 性能飙升：GRPO 带来 4 倍性能提升！** 在 **Tulu pipeline** 中从 **PPO** 切换到 **GRPO** 带来了 [4 倍的性能增长](https://x.com/vwxyzjn/status/1889730488199209393)，在 **MATH** 和 **GSM8K** 等任务上表现出显著改进。这标志着 AI 训练中 **RL** 策略的一个极具前景的方向。

- **微调开发者欢欣鼓舞：GRPO 让模型表现更出色** 用户分享了使用 **GRPO** 微调模型的成功案例，强调了数据集准备和适当训练模板的重要性。像 [OpenR1-Math-Raw](https://huggingface.co/datasets/open-r1/OpenR1-Math-Raw) 这样的工具和数据集已成为增强模型性能的宝贵资源。

**主题 3：Thomson Reuters 在法庭上重挫 AI 模仿者**

- **版权之战：Thomson Reuters 赢得首场 AI 法律诉讼！** 在一项具有里程碑意义的裁决中，[Thomson Reuters 赢得了针对 Ross Intelligence 的版权诉讼](https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/)，原因是后者复制了来自 Westlaw 的材料。Stephanos Bibas 法官宣称：*“Ross 的任何辩护理由都站不住脚，”* 强调了侵权行为的严重性。

- **AI 上的法律课：尊重 IP 否则后果自负** 这一裁决为美国的 AI 版权设定了关键先例，强调 AI 公司在开发技术时必须尊重知识产权。该案例发出了关于 AI 开发中法律责任的强烈信号。

- **律师们欢呼：AI 成为源源不断的财富** 法律界对这一裁决后可能出现的新案例议论纷纷。敦促各公司审查其 AI 训练数据以避免类似的诉讼，而 IP 律师则看到了未来工作机会的激增。

**主题 4：DeepScaleR 让 RL 重回聚光灯下**

- **RL 复兴：DeepScaleR 的小巨人挑战巨头！** [DeepScaleR 预览版](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scali) 展示了一个显著扩展了 **RL** 规模的 **1.5B 模型**，引发了 AI 社区的兴奋。随着该模型表现超出预期，爱好者们欢呼：*“RL 回来了，宝贝！”*

- **小模型，大影响：DeepScaleR 挑战 Scaling 规范** 该模型的进步表明，即使是较小的模型，通过适当的 **RL** 扩展技术也能取得令人印象深刻的结果。这挑战了只有巨型模型才能领跑 AI 领域的观念，为更高效的 AI 开发打开了大门。

- **研究人员集结：RL 技术获得二次生命** DeepScaleR 的成功鼓励研究人员重新审视强化学习方法。这种复兴可能会带来 AI 训练和优化方面的新创新，因为社区正在探索可扩展的解决方案。

**主题 5：AI 模型通过 Automated Capability Discovery 变得充满好奇心**

- **模型化身科学家：ACD 让 AI 自我探索！** 一个名为 [Automated Capability Discovery (ACD)](https://arxiv.org/abs/2502.07577) 的新框架允许 AI 模型自我探索其能力和弱点。通过充当自己的“科学家”，像 **GPT** 和 **Claude** 这样的模型可以提出任务来评估自己，正如 [Jeff Clune 的推文](https://x.com/jeffclune/status/1889568685632667672) 中所强调的那样。

- **基础模型走向自我意识：会出什么问题吗？** **ACD** 使模型能够在无需详尽人工测试的情况下识别意外行为，以更少的人力投入提高评估准确性。虽然令人兴奋，但随着模型开始进行自主探索，这也引发了关于 AI 系统控制和安全性的疑问。

- **更少人工，更多机器：ACD 重新定义模型评估** 借助 **ACD**，开发人员可以潜在地加快开发周期并发现隐藏的模型潜力。社区对此影响既感到好奇又保持谨慎，在创新与负责任的 AI 实践需求之间寻找平衡。

---

# 第一部分：Discord 高层摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GRPO 实现挑战**：成员们讨论了将 **GRPO** 与 **Mistral** 和 **Llama** 等模型集成的问题，指出即使实现正确，模型也无法生成预期的 **tokens**，这暗示了在集成 **<thinking>** 等 **special tokens** 时的困难。
   - 分享了一篇 [推文](https://x.com/UnslothAI/status/1889726411478278183) 和相关的 [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb#scrollTo=vITh0KVJ10qX)，展示了通过 **GRPO** 将 *Llama 3.1 (8B)* 转换为 **chain-of-thought** 模型，强调了使用合适训练模板的必要性。
- **数据集清洗需要更深层次的分析**：讨论强调，简单地从数据集中删除缺失值可能会降低数据的相关性；在训练前进行彻底的分析和理解对于有效的数据准备至关重要，以确保数据集对 **LLM** 训练保持相关性和鲁棒性。
   - 更多信息请参考 [Datasets 101 | Unsloth Documentation](https://docs.unsloth.ai/basics/datasets-101#getting-started)，该文档被引用为最佳实践的有用资源。
- **Liger 与 Apple Kernels 表现出性能差异**：**Liger kernel** 与 **Apple** 的 **cross-entropy** 实现之间的对比显示，虽然 **Liger** 具有速度优势，但 **Apple** 的 **kernel** 由于其完整的实现，在执行某些操作时效率更高，从而影响了整体性能。
   - 具体而言，讨论引用了 [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py) 和 [Apple's ml-cross-entropy](https://github.com/apple/ml-cross-entropy/blob/main/cut_cross_entropy/cce_lse_forward.py#L79) 中的实现，由于它们处理 **logits** 的方式不同而存在细微差别。
- **GRPO 在 A100 上的微调困境**：一位用户在 **A100** 上微调 **Qwen 32B** 模型时遇到了显存溢出 (**OOM**) 错误，将上下文长度从 128k 减少到 16k，引发了关于内存分配可行性的疑问。
   - 该用户就 **GRPO** 过程中是使用 **wandb** 还是 **Unsloth** 内置功能进行实验跟踪寻求建议，并指出他们主要对 **loss** 跟踪和优化感兴趣。
- **奖励函数的宽容导致重复输出**：社区成员发现，奖励函数虽然有效，但对某些短语过于宽容，导致出现不希望的重复输出，如 *"Hmm, let's see..."*，这凸显了对更复杂惩罚机制的需求。
   - 为了解决这个问题，建议探索先前消息的滑动窗口以改进自我监督，而不是独立处理每次生成，从而提高回答的多样性。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DIY 语音聊天机器人兴起**：用户探索了使用 **Raspberry Pi** 和 **ESP32** 制作 DIY 语音聊天机器人，推荐了 **Eilik 伴侣机器人**和用于设备造型的自定义 3D 打印。
   - 这展示了在增强个人科技体验中创意与功能的融合。
- **Home Assistant 开启对话**：成员们讨论了 **Home Assistant Voice**，它允许使用 **OpenAI APIs** 进行网页搜索和智能家居控制，从而实现自定义语音助手。
   - 该设置需要运行 **Home Assistant** 服务器并支持多语言配置，使其能够覆盖多样化的用户群体。
- **Moxie 命运未卜**：人们对 **Moxie**（一款面临未来威胁的儿童机器人伴侣）表示担忧，尽管其 **emotional intelligence** 仍备受关注。
   - 参与者推测了潜在的继任者，并讨论了其专注于儿童互动的设计；参见 [关于 Moxie 的 YouTube 视频](https://www.youtube.com/watch?v=7dd_r-aqecw)。
- **迭代提示词交付成果**：一位成员分享说，通过从基准开始并不断改进提示词，**iterative prompting** 可以显著提高结果。
   - 社区强调了*清晰且具体指令*的必要性，承认 **LLMs** 在没有明确指导的情况下无法推断意图。
- **Function Calling 令人头疼**：一位成员描述了在其 **system prompt** 中使用 **function calling** 时的挑战，指出根据客户端交互会出现失败或不必要的触发。
   - 他们还提到，即使有明确指令要求在模糊回答时避免 **function calls**，性能仍然存在滞后。



---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Codeium 插件落后于 Windsurf**：成员们表示担心 **Codeium extension** 正逐渐落后，原因是公司增加了对 **Windsurf** 和企业级产品的关注。
   - 一位成员指出，该插件仍可通过企业选项使用，凸显了双重重心，而其他人则在评估是否切换到 **Cursor**。
- **Windsurf 深受错误和停机困扰**：用户报告了 **Windsurf** 持续存在的问题，包括在使用 **Cascade** 时反复出现内部错误以及 **Gemini** 模型的问题。
   - 许多人对最近的性能下降表示沮丧，特别是无法可靠地编辑文件，详情见 [Codeium 的状态页面](https://status.codeium.com)。
- **Claude 3.5 Sonnet 位居 Windsurf 模型排行榜首位**：一项非官方排名将 **Claude 3.5 Sonnet** 列为 **Windsurf** 中表现最好的模型，归功于其上下文处理和工具调用（tool calling）能力。
   - **Gemini 2.0 Flash** 和 **O3-Mini** 因速度和价格受到称赞，而 **GPT-4o** 则因表现不佳受到批评。
- **用户呼吁对 AI 生成的输出保持警惕**：几位用户强调了在使用 AI 时保持警惕的重要性，指出盲目信任 AI 可能会导致代价高昂的错误。
   - 对话强调了对更清晰的风险评估和用户教育的需求，并引用了 Windsurf 自动补全的问题：[请求已取消](https://codeium.canny.io/feature-requests/p/windsurfs-autocomplete-now-working-around-08-35-41202-utc)。
- **请求通过 llms.txt 格式提供文档源**：用户讨论了在 **Windsurf** 中添加自定义文档源的可能性，参考了通过 **llms.txt** 格式索引文档的标准化方法。
   - 社区希望在这一领域有所改进，以增强功能和访问便利性，并链接到了 [llms.txt 目录](https://directory.llmstxt.cloud)。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonar 基于 R1 构建但优于 DeepSeek**：用户辩论了 **DeepSeek R1** 与 **Sonar Reasoning Pro** 的优劣，结论是 **Sonar** 基于 **R1** 构建，并针对网页搜索响应进行了优化，可能会取代 Perplexity 应用中的 DeepSeek R1。
   - [Perplexity 的一条推文](https://x.com/perplexity_ai/status/1889392617479082323?s=61)指出，基于 **Llama 3.3 70b** 构建的 **Sonar** 表现优于 **GPT-4o-mini** 和 **Claude 3.5 Haiku**，同时能与顶级模型媲美。
- **Perplexity API 深受 500 错误困扰**：多位用户报告在尝试访问 Perplexity **API** 时遇到 **500 内部服务器错误**，引发了对其可靠性和生产就绪性的担忧。
   - 尽管 [状态页面](https://status.perplexity.com/) 显示运行正常，但用户表示沮丧，报告几乎每次 **API** 调用都会出现持续的 **500 错误**。
- **Sonar 获得实时互联网浏览功能**：Perplexity 可以根据当前链接进行搜索，赋予其 **实时互联网浏览能力**。
   - 这使得浏览更具灵活性，并能获取最新信息，在需要市场 [摘要、每日亮点、收益快报](https://x.com/PPLXfinance/status/1889742180421337120?s=61) 时特别有用。
- **OpenAI 品牌重塑及其他新闻**：最近发生的事件包括 **OpenAI 的品牌重塑**、关于 **Apple 桌面机器人原型** 的消息以及发现了 **宇宙中最大的结构**。
   - 查看 [YouTube 视频](https://www.youtube.com/embed/9SUxli8UDA0) 获取详细见解。
- **401 授权问题已解决**：一位用户最初在尝试访问 **API** 时遇到了 **401 Authorization Required** 错误，但在排查后解决了该问题。
   - 按照建议移除 Token 周围的 `<>` 括号后，该用户报告 **API** 开始正常工作。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Deepseek R1 激发代码好奇心**：社区成员探索了 **Deepseek R1 distill model** 在数学和推理方面的表现，尽管代码编写并非其主要功能，但初步建议测试其编程能力。
   - 讨论强调了该模型在各种应用中处理复杂问题的潜力。
- **LM Studio 缺乏音频处理能力**：用户报告称 **LM Studio** 不支持像 **Qwen2-Audio-7B-GGUF** 这样的音频模型，引发了关于利用音频模型替代方法的讨论。
   - 建议将外部工具和平台作为寻求使用音频模型的潜在解决方案，但未提供具体建议。
- **Markdown 渲染错误导致消息混乱**：报告了一个 Bug，即 Markdown 输入被渲染为格式化文本，而不是在 **LM Studio** 中显示为原始文本，从而干扰了聊天界面。
   - 该问题已记录在 [bug tracker](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/430) 中，指出了意外行为并请求修复。
- **5090 可靠性传闻引发警惕**：关于 **5090** GPU 可靠性的担忧加剧，参考了有关显卡故障的报告，这些 [传闻报告](https://www.youtube.com/watch?v=L1NPFFRTzLo) 促使了谨慎行为。
   - 作为预防措施，用户建议对 **5090** 进行降压处理以缓解潜在问题。
- **多 GPU 构建中的带宽瓶颈**：分享了构建多 GPU 服务器的经验，指出在多 GPU AI 设置中优化性能的特定主板配置，尽管存在带宽限制。
   - 讨论包括了由于主板限制而使用 x1 链路的场景，挑战了在有限 PCI-E 通道下对 GPU 性能的典型预期。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **路透社赢得 AI 版权案**：汤森路透在针对 Ross Intelligence 复制 Westlaw 材料的重大 AI 版权案中 [获胜](https://storage.courtlistener.com/recap/gov.uscourts.ded.72109/gov.uscourts.ded.72109.770.0.pdf)，Stephanos Bibas 法官驳回了 Ross 的所有辩护。
   - 这是一个*里程碑式的案例*，为美国的 AI 版权设定了先例。
- **Current AI 筹集巨额资金**：[Current AI](https://www.currentai.org/) 开始其在公益 AI 领域的工作，承诺投入 **4 亿美元**，目标是在五年内达到 25 亿美元，参与者遍布从拉各斯到利马的各地。
   - 该倡议旨在引导 AI 发展，使其服务于社区机会和安全。
- **OpenAI 策划 GPT 4.5 和 5**：**OpenAI** 计划发布 **GPT-4.5**，这将是最后一个非 chain-of-thought 模型，随后是旨在统一所有产品供应并提供无限免费层级访问的 **GPT-5**。
   - 付费订阅者将获得增强功能，包括语音和 deep research 功能。
- **GRPO 训练使性能提升 4 倍**：在 **Tulu pipeline** 中从 **PPO** 切换到 **GRPO** 导致性能提升了 **4 倍**，在 **MATH** 和 **GSM8K** 等挑战中显示出显著改进。
   - 最新的 **GRPO-trained Tulu model** 指明了 RL 策略的新方向。
- **xAI 员工因 Grok 3 被迫辞职**：一名 xAI 员工在被迫删除一条承认 **Grok 3** 存在的推文后辞职，该公司将其列为机密。该员工表示，这种显而易见的观点竟然能威胁到他的工作，他感到很失望。
   - 成员们猜测该员工关于未发布产品性能的言论是否影响了促使其辞职的决定，因为一些人认为 xAI 的立场与其倡导的自由言论主张相矛盾。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Deepfrying 影响 72B 模型训练**：一位用户报告称，与较小模型相比，在 **72B 模型**中经历了**剧烈且不断增加的 loss**，怀疑高学习率可能不是唯一的问题，**deepfrying** 可能会加剧这一现象。
   - 对话将 **deepfrying** 定义为一种模型经历逐渐增加的方差，导致 loss 峰值升高的状态，这种状态可能会进一步受到短序列长度的影响。
- **Magic 将上下文扩展至 100M Token**：Magic 的最新更新引入了 **Long-Term Memory 模型**，可以处理高达 **100M Token** 的上下文，增强了超越传统训练方法的推理能力，详见 [Magic 的博客](https://magic.dev/blog/100m-token-context-windows)。
   - 这一进步通过将广泛的代码库和文档集成到模型训练的上下文中，为软件开发开辟了重大机遇。
- **对 LM2 Memory Slots 的质疑**：针对 **LM2 模型**中 Memory Slot 实现的透明度出现了担忧，参见 [LM2 论文](https://arxiv.org/abs/2502.06049)，其架构中 Memory Slots 的选择和更新机制描述得并不清晰。
   - 参与者对该设计的有效性和可并行性表示怀疑，认为论文中的描述可能过于简化。
- **Automated Capability Discovery 自我探索模型**：根据 [Jeff Clune 的推文](https://x.com/jeffclune/status/1889568685632667672)，一个名为 **Automated Capability Discovery (ACD)** 的新框架旨在以系统化的方式自我探索模型能力，识别 Foundation Model 中意想不到的能力和弱点。
   - ACD 的运行方式是指定一个 Foundation Model 作为“科学家”，为其他模型提出任务，从而以更少的人力投入提高评估准确性。
- **探索使用助记模式进行微调**：一位成员询问是否有关于涉及助记字符串（mnemonic strings）微调方法的研究，特别是模型如何“识别”拼写出“HELLO”之类的模式。
   - 他们提到在这方面有一个“可测试的假设”，表明了进一步实验探索的潜力，并提供了合作的可能性。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Deepseek R1 定价令人困惑**：**Cursor** 更新了[文档](https://docs.cursor.com/settings/models)，规定了按量计费（usage-based pricing）和模型的可用性，引发了关于 **Deepseek R1** 和 **O3-mini** Premium 状态的困惑。
   - 文档规定了特定模型的 [按量计费](https://docs.cursor.com/account/usage#usage-based-pricing)，让用户自行比较 **Perplexity API** 和 **Claude** 等各种选项的成本和收益。
- **MCP 服务器集成引发麻烦**：用户在 **MCP 服务器集成**（特别是 **Perplexity API**）时遇到了问题，导致使用过程中出现错误。
   - 一些用户通过硬编码 API Key 和删除冲突包解决了问题，但性能的不一致性仍然存在。
- **O3-mini 的输出波动**：**O3-mini** 不稳定的性能引起了关注，用户根据上下文的不同，既经历了成功的输出，也经历了幻觉输出。
   - 根据用户反馈，虽然 **O3-mini** 偶尔会提供令人印象深刻的改进，但持续的不一致性仍然是一个显著的痛点。
- **Claude 模型发布引发期待**：对即将发布的 **Anthropic** 模型的积极情绪正在积聚，用户分享了关于 **Claude Sonnet** 等当前模型能力的正面体验。
   - 社区热切期待改进，特别是关于未来 **Anthropic** 迭代版本中承诺的功能和能力。

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **社区对 NVIDIA GB200 充满渴望**：一名成员确认该 Discord 服务器致力于讨论 **NVIDIA GB200 的“色情”图片**。
   - 另一名成员的快速确认突显了社区直接且幽默的风格。
- **Triton 的 Interpreter 模式大放异彩！**：在进行二维矩阵乘法时，**Triton 默认模式**下的**误差**明显大于 **INTERPRET 模式**，详见 [此 GitHub issue](https://github.com/triton-lang/triton/issues/5895)。
   - 在 INTERPRET 模式下，误差显著降低，仅为 **9.5367431640625e-07**，引发了关于与 Torch 性能差异的讨论。
- **CUDA 内存模型引发困惑**：一名 CUDA 初学者询问一段代码是否违反了 **C++ 内存模型**，并询问是否需要 acquire/release 语义，他在 [Stack Overflow](https://stackoverflow.com/questions/79429440/cuda-memory-model-why-acquire-fence-is-not-needed-to-prevent-load-load-reorderi) 上发布了问题以寻求社区反馈。
   - 另一名成员澄清说，寄存器定义是**针对每个线程（per thread）**的，每个线程可能会为一个 **8x8 矩阵**加载值。
- **CPUOffload 的挑战**：成员们讨论了 **CPUOffload** 的复杂性，特别是如何有效地将 **DTensor 分片（shards）**收集到 rank 0 进行优化器更新，而不会因使用 `mmap()` 或 `shm_open()` 等方法产生过大开销。
   - 一名成员还在寻求在 rank 0 上执行与梯度裁剪融合的 CPU 优化器步骤的高效方法，旨在不使用传统的 allreduce 设置的情况下使用缩减后的梯度。
- **Tilelang v0.1.0 发布！**：社区庆祝 [tilelang v0.1.0](https://github.com/tile-ai/tilelang) 的发布，这是一种用于高性能 AI 内核的新型 pythonic DSL，具有专用内存分配以及可选的布局和流水线注解等功能。
   - 该工具提供**细粒度的线程级控制**，并已向创建者发出邀请，希望其在未来的演讲中与社区分享更多内容。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 为所有人开放 OpenAI o1 和 o3**：**OpenAI** 的 **o1** 和 **o3** 推理模型系列现在对所有 **OpenRouter** 用户开放，无需 BYOK，并提高了之前 Key 用户的速率限制，详见 [此处](https://x.com/OpenRouterAI/status/1889708759355691327)。
   - 这些模型整合了网络搜索，扩大了其实用性并简化了用户体验。
- **Groq 的 Llama 模型以史无前例的速度运行**：得益于官方 **Groq** 支持，用户可以利用极速端点，以超过**每秒 250 个 token** 的速度运行 **Llama 3.3**，以 **600 TPS** 的速度运行 **Llama 3.1**，模型详情见 [此链接](https://openrouter.ai/provider/groq)。
   - 自带 Key (BYOK) 可以解锁更高的速率限制，从而提高效率。
- **Nitro 功能大幅提升吞吐量**：`:nitro` 后缀已升级，允许用户按延迟和吞吐量对端点进行排序，可通过 API 或在聊天中配置，而不是作为单独的端点出现。
   - 增强的图表可跟踪提供商性能，简化了随时间变化的对比。
- **DeepSeek R1 70B 开辟速度新路径**：**Groq DeepSeek R1 70B** 达到了约 **1000 tokens per second**，树立了速度新标杆，并提供广泛的参数支持和 BYOK 选项，信息分享在 [此处](https://x.com/OpenRouterAI/status/1889726731571044538)。
   - 社区对这一新标准反应积极。
- **OpenRouter 聊天记录凭空消失**：用户报告在更新后丢失了聊天记录，强调了记录是存储在本地的，他们声称最初并未明确告知这一点。
   - 成员们建议在清除浏览器历史记录时，应更清晰地提示潜在的数据丢失风险，以避免未来用户的挫败感。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Deep Hermes 发布备受期待**：社区热切期待 **Deep-Hermes-8B** 模型权重的发布，密切关注 NousResearch HuggingFace 仓库的公告和 benchmarks。
   - Teknium 表示准备工作正在进行中，包括 benchmarks 和 model card，并暗示该模型可能会被用来撰写关于其自身发布的帖子。
- **LM Studio Speculative Decoding 亮相**：最新的 LM Studio 0.3.10 Beta 引入了 **Speculative Decoding**，旨在通过主模型和草稿模型协同工作来加速 inference，有望提升性能。
   - 尽管潜力巨大，一些成员报告了褒贬不一的结果，认为 **Speculative Decoding** 对大型模型最有效，可能并不总能带来明显的加速。
- **校准数据集引发疑问**：人们对所使用的校准数据集的性质感到好奇，特别是其看似随机且无结构的内容，让人联想到劣质的预训练数据。
   - Jsarnecki 澄清说，选择这种不寻常的数据集是有意为之，因为研究表明，即使与 wikitext 等传统数据集相比，近乎随机的数据片段也能带来更好的训练效果。
- **黑客松 SUPERAGENTS 涌现**：为期一天的黑客松挑战开发者创造下一代 **SUPERAGENTS**，在各种框架和链上集成 Story 的 **Agent Transaction Control Protocol**。
   - 鼓励参与者在现有项目基础上进行创新或开发新项目，争夺奖项和合作机会。
- **美国拒绝签署 AI 安全宣言**：在一次国际峰会上，以 Vance 为代表的美国拒绝签署 AI 安全宣言，理由是担心与中国等**专制政权**的合作可能会危害国家安全。
   - 关于**多边主义**和国际协作措辞的分歧导致未能达成共识，特别是涉及美国在 AI 领域的领导地位时。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **用户强烈要求 Google Sheets 支持**：NotebookLM 团队正在征求关于 **Google Sheets** 集成的反馈，用户请求能够 **ingest** 数据，他们发布了[一份反馈调查](https://forms.gle/G78qnNCv2UwcYXc16)。
   - 该调查旨在收集详细规格，包括表格维度、数据类型以及用户希望从中获得的洞察。
- **NotebookLM 成为奇幻小说家的灵感缪斯**：一位用户正将 NotebookLM 作为其奇幻小说的写作助手，专注于世界观构建、角色开发和数据组织。
   - 该用户看重音频生成器能够合成潜在读者的提问，帮助识别其详尽世界观构建中的漏洞和不一致之处，并且他们正在动态刷新 **Google Sheets** 以跟踪进度。
- **AI 播客使内容创作民主化**：一位用户详细阐述了利用 AI 快速创建播客的方法，强调了巨大的市场机会，并指出根据[这篇文章](https://millionai.substack.com/p/create-ai-podcasts-in-seconds-without?r=297y6u&utm_medium=ios&triedRedirect=true)，**podcasting** 可以提升内容消耗和市场覆盖面。
   - 他们强调将静态内容转化为引人入胜的音频，在无需公开演讲的情况下实现影响力最大化，从 **NotefeedLM** 之类的工具中创造价值。
- **学生在限制中权衡并拥抱音频功能**：本科生用户使用 NotebookLM 生成模拟测试和总结资料，对其效果表示赞赏，然而每日查询限制使得使用变得困难。
   - **音频对话**功能因支持多任务处理而受到重视，但一些用户遇到了功能问题，并有人请求使用用户声音的个性化音频功能。
- **用户反映源文件格式问题**：用户报告了源文件显示问题；PDF 中混乱的格式阻碍了内容验证，影响了整体用户体验。
   - 产品团队承认了这些格式问题，并正在努力进行潜在改进，以准确显示源材料。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **OpenRouter 开放 OpenAI 模型**：OpenRouter 已向所有人开放 **OpenAI o1 和 o3**，取消了对 BYOK 的需求并提高了速率限制（Rate Limits），正如其在 [X 上的公告](https://x.com/OpenRouterAI/status/1889708759355691327)所述。
   - 此次更新广受好评，特别是它增强了功能性，尤其是在与 Web Search 集成时。
- **用户探索 Aider 多会话功能**：用户正寻求在 Aider 中管理多个 **tmux sessions** 的能力，以增强进程控制，例如用于服务器启动（Server Spawning）。
   - 目前，权宜之计是使用 **SSH connections** 进行本地设置，以简化编码工作流。
- **编辑器模型协作构想**：一项提案建议训练一个 **1.5b 'editor' 模型**与架构师模型（Architect Models）协作，以提高代码编辑效率。
   - 目标是减少幻觉（Hallucinations），并提高在大上下文（Context）中代码差异（Code Diffs）的精确度。
- **GPT-5 路线图公布**：根据 [Sam Altman 的推文](https://x.com/sama/status/1889755723078443244)，**GPT-4.5 和 GPT-5** 的计划旨在统一模型产品并改善用户体验。
   - GPT-5 将融合多种技术，并提供给具有不同智能水平的免费层级用户。
- **o3-mini 加速编码任务**：反馈表明 **o3-mini** 表现出色并加速了编码过程，在特定任务中优于其他模型。
   - 一些用户观察到使用 **o3** 的部署时间更快，另一些用户建议将其与 **Sonnet** 等模型结合使用以获得最佳效果。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SDXL 质量媲美 1.5，但缺乏独特诠释**：一场讨论对比了 **SDXL** 和 **SD 1.5**，指出 **SDXL** 在没有 Refiner 的情况下也能达到相当的质量，但由于专注于大众审美，缺乏 **1.5** 那种独特的诠释。
   - 成员们强调了 **Benchmarks** 的重要性，指出在这些受控评估中， **SDXL** 的表现通常优于 **SD 1.5**。
- **Flux 模型一致的面部特征凸显了数据微调**：**Flux 模型**产生相似面部特征（如独特的裂纹下巴）的一致性，表明其依赖于 **Quality-tuned Data** 或特定的蒸馏（Distillation）方法。
   - 虽然有些人发现其多样性低于 **SDXL**，但其他人认为 **Flux** 较高的对数似然分布（Log Likelihood Distribution）允许通过 **Loras** 来提高多样性。
- **蒸馏方法极大影响模型性能**：讨论明确了从 **Pro** 衍生出 **Schnell** 所采用的 'Timestep Distilled' 与 **Dev** 使用的 'Guidance Distilled' 不同，这显著影响了模型性能和 **Lora** 兼容性。
   - 讨论强调了蒸馏中不同的 **Data Handling** 技术如何关键性地影响最终模型的质量和行为。
- **人类偏好基准面临质疑**：有人担心**人类偏好基准（Human Preference Benchmarks）**可能更倾向于美观的输出，而非更客观的质量指标，这可能会导致结果偏差。
   - 令人担忧的是，这些基准可能会优先考虑像“美女”之类的输出，而不是基于详细且多样化 Prompts 的准确表达。
- **ComfyUI 迁移至 Linux 导致 OOM 错误**：一名用户报告称，在按照指南从 **Windows 上的 ComfyUI** 迁移到 **Linux** 后，在视频生成过程中遇到了 **OOM 错误**。
   - 社区成员建议验证 **Driver** 安装是否正确，其中一人指出，指导不足可能导致了系统的不稳定。



---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **作者标签引发不信任**：授予服务器作者标签（author flair）引发了褒贬不一的反应，其中一位成员对任何涉及 **crypto/NFTs** 的人表示 *不信任*。
   - 这种情绪凸显了社区内对诚信问题的持续关注。
- **社区辩论代码审查流程**：成员们讨论了为 MCP 公共服务器实施 *代码审查流程*，建议由多位审查者来管理工作量，因为目前已有 900 多个服务器。
   - 一位成员开玩笑地建议使用语言模型来预筛恶意代码。
- **开源 LLM 模型渴望新研究**：针对 *开源 LLM 模型需要突破性研究* 的担忧日益增加，并提到 DeepSeek 可能从 OpenAI 的工作中汲取了灵感。
   - 尽管存在创新的共享，但有人指出 DeepSeek 仍然利用了 OpenAI 的技术。
- **Clickhouse & Streamlit 创建仪表板**：一位成员对使用 *Clickhouse 和 Streamlit* 构建生成式仪表板服务器表现出浓厚兴趣，并正在考虑变现策略。
   - 他们询问了关于 Streamlit 与 PowerBI 等替代方案相比的有效性反馈，暗示了未来的变现合作。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 发布职位空缺**：Modular 最近发布了新的职位空缺，标志着公司内部持续的扩张和开发努力，这可能会带来未来的改进和集成。
   - 这些举措可能会促进其产品（如 Mojo 和 MAX）的改进和新集成。
- **Modular 取消 stdlib 会议**：由于时间冲突和组织者的离职，定期的 **stdlib 会议** 已停止。
   - 成员们在参加定期会议时遇到困难，并被告知会议暂时取消。
- **Parameterized traits 优于 Sum Types**：Mojo 团队优先考虑 **parameterized traits** 而非 **sum types**，因为前者能够实现更基础的能力。
   - 有人指出，目前的重点是开发底层功能，使 Mojo 能够表示类似于 C 的构造。
- **MAX 目前不优先考虑 Wasm**：Wasm 后端目前不是 MAX 的重点，也不在近期路线图中，因为 MAX 正专注于其他技术。
   - 一位成员对 Wasm 的相关性表示好奇，强调了其尽管目前不是优先级，但仍具有未来使用的潜力。
- **ONNX 模型执行依赖于 MAX**：成员们指出，Modular 对执行 **ONNX 模型** 的支持很大程度上取决于 **MAX**，强调了其必要性。
   - 这凸显了 MAX 在促进平台上各种 ML 模型执行中的作用，MAX 对于利用 GPUs 的应用程序至关重要，尽管运行 Mojo 并非严格需要它。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **VAEs 需要重参数化**：讨论围绕为什么在 **VAEs** 中无法直接通过分布进行 **backpropagation** 展开，由于随机采样操作是不可微的，因此必须使用 **reparameterization trick**。
   - 成员们澄清说，**VAEs** 生成的分布参数需要进行随机采样。
- **OpenAI 在竞赛编程中取得胜利**：OpenAI 发布了[一篇论文](https://arxiv.org/abs/2502.06807)，详细介绍了其 **o3 model** 在 IOI 2024 中无需手工设计策略即可获得金牌的表现，正如[这条推文](https://x.com/iScienceLuvr/status/1889517116816244995?s=46)所提到的，这标志着推理模型取得了重大进展。
   - 团队指出模型的灵活性是关键，这与 [这条推文](https://x.com/polynoamial/status/1889541408065028421?s=46) 中提到的 **o1-ioi** 此前需要专门流水线的要求形成对比。
- **Scaled Cognition 推出 Agentic APT-1 模型**：Scaled Cognition 宣布了[他们的 APT-1 模型](https://x.com/scaledcognition/status/1889721166421479751?s=46)，该模型专为 **agentic** 应用设计，目前在 **agent** 基准测试中名列前茅。
   - 团队强调了由 Khosla Ventures 领投的 **$21M** 种子轮融资，并利用了全合成数据流水线（synthetic data pipeline）。
- **Glean 发布可扩展 AI Agents**：Glean 推出了 [Glean Agents](https://x.com/glean/status/1889706504812683728)，这是一个旨在实现可扩展 AI **agent** 管理的平台，具有新的数据集成和治理功能。
   - 其目标是通过提供对公司和网络数据的便捷访问来提高生产力。
- **OpenAI 规划 GPT-4.5 和 GPT-5 路线图**：OpenAI 提供了一个[路线图更新](https://x.com/sama/status/1889755723078443244?s=46&t=JE84TqLviekDnEt8MAT-Eg)，预示了即将推出的 **GPT-4.5 和 GPT-5** 模型，旨在统一建模方法并简化产品供应。
   - OpenAI 发出了摆脱非推理模型的信号，专注于更广泛的功能和先进的推理能力。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **基于步数的 Checkpointing 正在开发中**：一位成员询问是否可以在 **Torchtune** 中每个 **epoch** 保存多次 **checkpoints**，另一位成员提到 **Joe** 正在 [PR #2384](https://github.com/pytorch/torchtune/pull/2384) 中开发此功能。
   - 他们表示这是一个*被广泛请求的功能*，预计将显著改进 **checkpointing** 过程。
- **MLFlow Logger 集成上线**：**MLFlow logger 集成**已成功合并，一位成员对此表示兴奋并计划尽快测试。
   - 该集成旨在增强 **Torchtune** 的日志记录能力。
- **Torchtune 支持分布式推理**：一位成员询问如何使用 **Torchtune** 在多 GPU 上运行 **distributed inference**，另一位成员分享了相关代码的[链接](https://github.com/pytorch/torchtune/blob/main/recipes/dev/generate_v2_distributed.py)。
   - 他们指出，将保存的模型加载到 **vLLM** 中进行分布式推理将会可行且*快得多*。
- **梯度累积问题困扰训练**：关于 [gradient accumulation 修复](https://github.com/pytorch/torchtune/issues/2334) 仍存在持续的困惑，这影响了训练效果。
   - 成员们描述了花费数小时进行调试却未找到根本原因，该问题似乎很复杂，可能需要更多的协作努力。
- **注意力机制依然至关重要**：一位参与者简洁地表示 *attention is still all we need*，强调了其在现代 AI 模型中的基础作用。
   - 这进一步强化了人工智能领域对 **attention** 机制的持续重视和关注。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **TinyStories 论文在小规模数据上训练模型**：[tinystories 论文](https://link.to.tinystories) 被推荐用于在**有限数据集**上训练 ML 模型，为数据集受限下的有效学习提供了策略。
   - 这对于获取大规模数据集困难或成本高昂的场景特别有用。
- **欧盟承诺向 AI 超级工厂投入资金**：根据 [Ursula von der Leyen 的公告](https://www.msn.com/en-us/money/companies/eu-pledges-200-billion-in-ai-spending-in-bid-to-catch-up-with-u-s-china/ar-AA1yO0Su)，欧盟承诺投入 **2000 亿欧元** 进行 AI 投资以与美国和中国竞争，重点是建立用于高级模型训练的 **AI 超级工厂 (gigafactories)**。
   - 该倡议旨在使欧洲成为 AI 技术和发展的领先大陆。
- **DeepScaleR 超出扩展预期**：[DeepScaleR 预览](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scali) 展示了一个 **1.5B 模型**，该模型显著扩展了 RL，在社区内引发了轰动。
   - 该模型的进展表明 RL 技术有望复兴。
- **路透社版权在 AI 诉讼中获胜**：在一场具有里程碑意义的案件中，[汤森路透在针对 Ross Intelligence 的诉讼中赢得了版权胜利](https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/)，强调了在 AI 领域尊重知识产权的重要性。
   - 法官 Stephanos Bibas 对 Ross 作出了果断判决，称：*Ross 的任何辩护理由都站不住脚*。
- **OpenAI 路线图预告 GPT-4.5**：根据 [Sam Altman](https://x.com/sama/status/1889755723078443244?t=EgnihPXVoD2fsS9ag5u5SA&s=19) 的说法，OpenAI 透露 **GPT-4.5** 将是他们最后一个不使用 chain-of-thought 的模型，并计划整合 **o 系列和 GPT 系列模型**。
   - 他们的目标是让模型在各种应用中都能“直接可用” (*just work*)，简化用户交互。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **CUDA 后端成功适配 Windows**：一位用户通过使用适当的 DLL 名称修正 autogen 文件，使 **CUDA 后端在 Windows 上运行**，但标准的 CI 运行器缺乏 GPU 支持。
   - 他们建议可能需要硬编码 CUDA 版本以保持设置简单。详见 [此 PR](https://github.com/tinygrad/tinygrad/pull/9036)。
- **CI 在后端环境变量上遇到困难**：**Windows CI** 未能在步骤之间传递后端环境变量，导致测试期间默认切换到 CLANG。
   - 已发起一个拉取请求以确保环境变量在 CI 步骤之间保持不变，从而实现正常功能；参见 [此 PR](https://github.com/tinygrad/tinygrad/pull/9047)。
- **测试迭代引发混乱**：关于从递归切换到迭代的疑虑浮现，因为这导致了除原始更改之外的许多测试失败。
   - CI 失败的直接原因源于一个缩进问题，该问题无意中影响了代码中的关键功能。
- **Tinygrad 承诺更便宜的硬件**：一位用户询问了从 **PyTorch** 等成熟框架切换到 **tinygrad** 的优势，并提到了使用前者的个人经验。
   - 另一位成员建议，选择 tinygrad 可能会带来**更便宜的硬件**、对底层过程更好的理解，以及潜在更快的模型性能。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 招聘开源工程师**：[@llama_index](https://twitter.com/llama_index) 宣布了一个**开源工程师**的全职岗位，正在寻找对 **Python** 和 **AI** 充满热情的候选人。
   - 有关扩展 **llama_index** 框架的更多细节可以在[这里](https://t.co/WMgdaauxP8)查看。
- **Nomic AI 改进文档工作流**：[@nomic_ai](https://twitter.com/nomic_ai) 展示了优秀的 **embedding model**（嵌入模型）对于实现高效 **Agentic Document Workflows**（智能体文档工作流）的重要性。
   - 这一新进展受到了积极评价，标志着在增强此类工作流方面迈出了重要一步，更多细节分享在[这里](https://t.co/pezsylHNpH)。
- **数据加载器对 RAG 系统至关重要**：成员们讨论了在构建 **RAG 系统**和查询引擎时尝试不同数据加载器的需求，并推荐使用 [llamahub](https://llamahub.example) 获取资源。
   - 一位成员强调了根据特定用例选择定制化加载器的重要性。
- **成员讨论批量处理 PDF**：一位成员就**批量处理 PDF** 的方法寻求建议，并要求澄清正在考虑的具体方案。
   - 对话表明，需要更专业的工具或脚本来高效管理大批量 PDF 操作。
- **利用过滤器构建智能查询引擎**：一位成员询问了在查询引擎工具中针对不同主题使用**预定义过滤器**的技巧，旨在不创建多个索引的情况下实现高效工作流。
   - 另一位成员分享了一个代码示例，说明了如何实现带有指定过滤器的查询引擎工具。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM 黑客松获胜者揭晓**：**LLM Agents MOOC Hackathon** 获胜者已公布。正如 [Dawn Song 教授的推文](https://x.com/dawnsongtweets/status/1889686697564315963)所述，此次活动吸引了来自 **127** 个国家和 **1,100** 多所大学的约 **3,000** 名参与者。
   - 主要参与方包括 **Amazon**、**Microsoft**、**Samsung** 和 **Salesforce**，获胜团队展示在[黑客松官网](https://rdi.berkeley.edu/llm-agents-hackathon/)上。
- **高级 LLM MOOC 即将开课**：根据 [Dawn Song 教授的公告](https://x.com/dawnsongtweets/status/1889355520294944829)，专注于**高级 LLM Agents** 的 **2025 春季 MOOC** 已经启动，内容涵盖**推理与规划 (Reasoning & Planning)**、**多模态 Agents** 以及 **AI 数学**。
   - 基于 **2024 秋季** MOOC 的成功（注册学员超过 **1.5 万**，YouTube 课程播放量超过 **20 万**），直播课程安排在每周一 **下午 4:10 (PT)**。
- **课程详情即将公布**：**MOOC 课程大纲**的细节预计将在大约**两周内**发布，本学期将不会举办黑客松。
   - MOOC 学生正在等待关于如何申请研究课题的更多信息。
- **DeepScaleR 通过 1.5B 模型扩展强化学习**：根据最近的一份文档，[DeepScaleR 模型](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)通过扩展强化学习（RL）技术，利用 **1.5B 模型**超越了 O1 preview。
   - 有关作业截止日期的详情即将发布，并提醒学生补习错过的课程。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic AI 提供 Steam 礼品卡**：一位成员通过 [steamcommunity.com/gift-card/pay/50](https://u.to/kR7WIQ) 宣布了 **50 美元 Steam 礼品卡抽奖活动**。
   - 该帖子反响不一，一位成员将其标记为**垃圾信息 (spam)**。
- **关于 TextWeb-UI 安装复杂性的辩论**：一位成员提到 **TextWeb-UI** 需要复杂的安装过程，另一位用户指出它不是简单的 `.exe` 安装。
   - 这种复杂性引起了部分成员对其易用性和可访问性的担忧。
- **移动端应用电池寿命受质疑**：针对 **iOS 和 Android** 移动端应用的使用出现了担忧，一位成员推测此类应用可能会在 **1 小时内耗尽设备电量**。
   - 讨论强调了 **Nomic AI** 生态系统中移动端应用的性能问题。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 遭遇 Failed Fetch 错误**：用户报告在尝试使用凭据登录个人账户时出现 “Failed to fetch” 错误，但该体验反馈的*信息量并不大*。
   - 该错误引发了关于可能存在拦截 API 请求的过滤机制的询问。
- **Cohere API 请求是否被过滤？**：成员们正在调查过滤是否可能导致登录尝试期间 API 请求失败。
   - 这一担忧表明可能需要进行更深入的调查，以确定连接问题或软件限制。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI 助力播客成功**：一场将于 **IST 时间 2 月 13 日星期四晚上 9 点**举行的免费研讨会将教导创作者如何仅使用 **AI** 且无需昂贵设备来启动播客，参与者将学习 **AI 音频模型的基础知识**。
   - 该课程提供 [ElevenLabs](https://elevenlabs.io) 和 [PlayHT](https://playht.com) 等平台的动手实践经验，以毫不费力地**将文本转化为音频内容**。
- **动手实践音频创作**：参与者将获得领先**语音生成平台**的实践经验，从而能够毫不费力地**将文本转化为音频内容**，并开发自己的**开源 NotebookLM** 以进行自定义实现。
   - 通过 [Build Fast With AI](https://t.me/BuildFastWithAI) 可以获得更多致力于**生成式 AI 解决方案**的免费资源和工具，提供**最新的 Gen AI 工具**、路线图和研讨会链接。

---

**DSPy Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1338968843212685322)** (1027 条消息🔥🔥🔥): 

> `GRPO 实现, 数据集清洗, MoE 模型, Liger Kernel vs Apple Kernel, OpenR1-Math-Raw 数据集` 

- **GRPO 实现指南**：成员们讨论了将 GRPO 与 Mistral 和 Llama 等模型集成，强调了使用适当训练模板以有效利用 `<thinking>` 等特殊 Token 的重要性。
   - 在测试输出时遇到了挑战，特别是尽管实现正确，模型仍未产生预期的 Token。
- **数据集清洗的重要性**：关于数据集清洗的讨论中，成员指出，在不了解数据的情况下简单地删除缺失值可能会稀释数据集的相关性。
   - 建议在训练前进行彻底的分析和理解，这对于有效的数据准备至关重要。
- **探索 MoE 模型**：一位成员分享了在获得现有架构经验后构建自定义 MoE 模型的意图，理由是 MoE 在处理大型模型方面具有潜在优势。
   - 同时也讨论了关于训练此类模型的成本和算力需求的担忧。
- **Liger 与 Apple Kernel 的对比**：小组对比了 Liger Kernel 与 Apple 的交叉熵实现，指出了它们在处理 Logits 方面的差异以及对性能的影响。
   - 成员指出，虽然 Liger 可能具有某些速度优势，但 Apple 的 Kernel 由于其完整的实现，在执行某些操作时效率更高。
- **OpenR1-Math-Raw 数据集发布**：介绍了 OpenR1-Math-Raw 数据集作为数学推理资源，包含超过 51.6 万个问题和经过验证的解答。
   - 该数据集旨在帮助用户生成和评估数学推理任务，使其成为训练 LLM 的潜在价值工具。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5">DeepSeek R1 (所有版本) - unsloth 集合</a>: 未找到描述</li><li><a href="https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview">agentica-org/DeepScaleR-1.5B-Preview · Hugging Face</a>: 未找到描述</li><li><a href="https://pastebin.com/cfibZ8DG">Qwen2VL-GRPO-o3mini - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的文本存储工具。Pastebin 是一个可以在线存储文本并设置存储时间的网站。</li><li><a href="https://huggingface.co/datasets/open-r1/OpenR1-Math-Raw">open-r1/OpenR1-Math-Raw · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://build.nvidia.com/nvidia/nemotron-4-340b-reward">NVIDIA 的 nemotron-4-340b-reward 模型 | NVIDIA NIM</a>: 根据有用性、正确性、连贯性、复杂性和冗长程度五个属性对回答进行评分。</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">持续预训练 | Unsloth 文档</a>: 又称持续微调（Continued Finetuning）。Unsloth 允许你进行持续预训练，使模型能够学习一种新语言。</li><li><a href="https://github.com/TruffleClock/nano-r1/blob/main/nano-r1.ipynb">nano-r1/nano-r1.ipynb (main 分支) · TruffleClock/nano-r1</a>: 通过在 GitHub 上创建账户，为 TruffleClock/nano-r1 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset">agentica-org/DeepScaleR-Preview-Dataset · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/agentica-project/deepscaler/tree/main/scripts/train">deepscaler/scripts/train (main 分支) · agentica-project/deepscaler</a>: 使 LLM 的强化学习（Reinforcement Learning）平民化。通过在 GitHub 上创建账户，为 agentica-project/deepscaler 的开发做出贡献。</li><li><a href="https://github.com/deepseek-ai/DeepSeek-Math">GitHub - deepseek-ai/DeepSeek-Math: DeepSeekMath: 挑战开源语言模型数学推理能力的极限</a>: DeepSeekMath: 挑战开源语言模型数学推理能力的极限 - deepseek-ai/DeepSeek-Math</li><li><a href="https://huggingface.co/Blackroot/SimpleDiffusion-MultiHeadAttentionNope/blob/main/train.py#L94>">train.py · Blackroot/SimpleDiffusion-MultiHeadAttentionNope (main 分支)</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/1590">使用纯文本输入训练视觉模型 · Issue #1590 · unslothai/unsloth</a>: 我需要仅使用文本输入来训练视觉模型。我尝试使用 Colab 笔记本，但发现数据中必须包含图像。在进一步研究后，我发现了一个 Colab 笔记本...</li><li><a href="https://x.com/UnslothAI/status/1889726411478278183">来自 Unsloth AI (@UnslothAI) 的推文</a>: 使用我们免费的笔记本，利用 DeepSeek 的 GRPO 算法训练你自己的推理 LLM！你将把 Llama 3.1 (8B) 转换为具有思维链（chain-of-thought）能力的模型。Unsloth 使 GRPO 减少了 80% 的 VRAM 占用。指南：https:...</li><li><a href="https://github.com/fishaudio/fish-speech/discussions/870">为什么 FishSpeech API 比我自托管的设置更快？ · fishaudio/fish-speech · Discussion #870</a>: 大家好，我想了解 FishSpeech API 推理与我自托管部署之间的性能差距。这是我的配置：技术细节：硬件：NVIDIA H...</li><li><a href="https://github.com/triton-lang/triton/issues/5895">默认模式下的误差明显大于 INTERPRET 模式 · Issue #5895 · triton-lang/triton</a>: 错误描述：对于简单的二维矩阵乘法，我的 Triton 内核与 Torch 之间的差异为：INTERPRET 模式：9.5367431640625e-07 (设置 os.environ["TRITON_INTERPRET"] = "1"...</li><li><a href="https://github.com/allenai/s2orc?tab=readme-ov-file#download-instructions">GitHub - allenai/s2orc: S2ORC: Semantic Scholar 开放研究语料库: https://www.aclweb.org/anthology/2020.acl-main.447/</a>: S2ORC: Semantic Scholar 开放研究语料库: https://www.aclweb.org/anthology/2020.acl-main.447/ - allenai/s2orc</li><li><a href="https://github.com/datadreamer-dev/DataDreamer">GitHub - datadreamer-dev/DataDreamer: DataDreamer: 提示词、生成合成数据、训练并对齐模型。 🤖💤</a>: DataDreamer: 提示词、生成合成数据、训练并对齐模型。 🤖💤 - datadreamer-dev/DataDreamer</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py">Liger-Kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py (main 分支) · linkedin/Liger-Kernel</a>: 用于 LLM 训练的高效 Triton 内核。通过在 GitHub 上创建账户，为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/apple/ml-cross-entropy/blob/main/cut_cross_entropy/cce_lse_forward.py#L79">ml-cross-entropy/cut_cross_entropy/cce_lse_forward.py (main 分支) · apple/ml-cross-entropy</a>: 为 apple/ml-cross-entropy 的开发做出贡献...</li>

le/ml-cross-entropy 通过在 GitHub 上创建账号来支持其开发。</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py#L25>">Liger-Kernel/src/liger_kernel/ops/cross_entropy.py (main 分支) · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号来支持 linkedin/Liger-Kernel 的开发。</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/cross_entropy.py">flash-attention/flash_attn/ops/triton/cross_entropy.py (main 分支) · Dao-AILab/flash-attention</a>：快速且内存高效的精确 Attention。通过在 GitHub 上创建账号来支持 Dao-AILab/flash-attention 的开发。</li><li><a href="https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B-v0.2">Skywork/Skywork-Reward-Gemma-2-27B-v0.2 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py#L264>">Liger-Kernel/src/liger_kernel/ops/cross_entropy.py (main 分支) · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号来支持 linkedin/Liger-Kernel 的开发。</li><li><a href="https://github.com/Deep-Agent/R1-V/blob/367658d518d3173ee4c2a47123547adeab363b14/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L62">R1-V/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py (版本 367658d) · Deep-Agent/R1-V</a>：以不到 3 美元的成本见证 VLM 的“灵光一现”时刻。通过在 GitHub 上创建账号来支持 Deep-Agent/R1-V 的开发。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1338980495878389781)** (12 条消息🔥): 

> `阅读资源，获取 AI/ML/RAG 信息，Reddit 作为来源，本地 Llama 托管` 


- **阅读资源受到好评**：一位成员对分享的阅读资源表示感谢，称其**非常棒**，并向另一位成员致谢。
- **探索 AI/ML/RAG 的信息来源**：成员们讨论了获取 AI、ML 和 RAG 信息的各种方法，提到他们使用 **Twitter** 和 **RSS** 订阅源。
   - 一位成员对建议持开放态度，并询问除了目前的方法之外是否还有**更好**的来源。
- **推荐使用 Reddit 进行讨论**：一位成员建议将 **Reddit** 作为来源，特别是为了获取有关本地 AI 发展的见解。
   - 他们强调该社区分享了大量的**行业动态**，使其成为一个宝贵的资源。
- **Local Llama 作为思路来源**：讨论中提到了 **local llama** 托管作为一个选项，不过有人澄清这指的是社区讨论，而非个人托管。
   - 一位成员认可了这一见解，并在意识到其价值后计划**尝试一下**。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1338965061326082232)** (149 条消息🔥🔥): 

> `Fine-tuning Models, GRPO Optimization, Experiment Tracking, Qwen Model Performance, Mistral Benchmarking` 


- **A100 上的 GRPO Fine-tuning**：一位用户在微调 **Qwen 32B** 模型时遇到了显存溢出 (OOM) 问题，最初使用的是 128k Context，随后逐渐减少到 16k。
   - 他们不确定 Trainer 中的内存分配情况，以及在 **A100** 上实现完整的 128k Context 是否可行。
- **使用 wandb 进行实验追踪**：一位用户正在考虑是配置 **wandb**，还是利用 **Unsloth** 内部的实验追踪功能来进行 Loss 追踪。
   - 已确认支持训练 Loss 追踪，特别是针对 **GRPO** 过程。
- **Mistral Fine-tuning 性能**：使用 **QLoRA**，Unsloth 的 **Mistral 7B** 模型微调速度最高可提升 **14 倍**，并显著降低 VRAM 占用。
   - 性能的提升使得用户即使在性能较弱的硬件上也能有效地进行 Fine-tuning。
- **Tokenization 与数据准备**：讨论了 Fine-tuning 时正确的数据格式，强调了 Tokenization 和结构化数据集对 LLM 的重要性。
   - 用户被引导至 **Hugging Face** 文档等有用资源，以了解如何构建 Chat Templates。
- **TeleChat 模型 Context 限制**：一位用户询问关于全能力运行 **TeleChat** 模型的可行性，这凸显了由于模型架构导致的性能限制。
   - 有意见指出，如果用户想要调整这个较旧的模型，可能需要从头开始创建支持层。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb#scrollTo=vITh0KVJ10qX)">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benchmarks">Unsloth Benchmarks | Unsloth Documentation</a>: 想知道 Unsloth 有多快吗？</li><li><a href="https://unsloth.ai/blog/r1-reasoning">Train your own R1 reasoning model locally (GRPO)</a>: 你现在可以使用 Unsloth 100% 在本地复现你自己的 DeepSeek-R1 推理模型。使用 GRPO。开源、免费且对初学者友好。</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/mistral-benchmark#Breakdown">Unsloth update: Mistral support + more</a>: 我们很高兴发布对 Mistral 7B、CodeLlama 34B 以及所有其他基于 Llama 架构模型的 QLoRA 支持！我们添加了 Sliding Window Attention、初步的 Windows 和 DPO 支持，以及 ...</li><li><a href="https://docs.unsloth.ai/basics/datasets-101#getting-started">Datasets 101 | Unsloth Documentation</a>: 学习创建 Fine-tuning 数据集的所有要点！</li><li><a href="https://github.com/unslothai/unsloth/tree/main/unsloth/models">unsloth/unsloth/models at main · unslothai/unsloth</a>: 以 2 倍的速度和减少 70% 的内存微调 Llama 3.3、DeepSeek-R1 和 Reasoning LLMs！🦥 - unslothai/unsloth</li><li><a href="https://github.com/MaxHastings/Kolo">GitHub - MaxHastings/Kolo: A one stop shop for data generation, fine tuning and testing LLMs locally using the best tools available. Keeping it simple and versatile!</a>: 使用现有最佳工具在本地进行数据生成、Fine-tuning 和测试 LLM 的一站式商店。保持简单且多功能！ - MaxHastings/Kolo</li><li><a href="https://github.com/sathvikask0/r1-distilled-RL/blob/main/config.py">r1-distilled-RL/config.py at main · sathvikask0/r1-distilled-RL</a>: 通过在 GitHub 上创建账号来为 sathvikask0/r1-distilled-RL 的开发做出贡献。</li><li><a href="https://asksathvik.substack.com/p/some-rl-ideas-i-am-currently-working">Some RL ideas I am currently working on..</a>: 我目前正在尝试的一些 RL 实验：</li><li><a href="https://github.com/sathvikask0/r1-distilled-RL">GitHub - sathvikask0/r1-distilled-RL</a>: 通过在 GitHub 上创建账号来为 sathvikask0/r1-distilled-RL 的开发做出贡献。</li><li><a href="https://huggingface.co/Tele-AI/TeleChat-12B">Tele-AI/TeleChat-12B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Chat Templates</a>: 未找到描述</li><li><a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/">Llama 3.1 | Model Cards and Prompt formats</a>: Llama 3.1 - 最强大的开源模型。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1339021502066200677)** (71 条消息🔥🔥): 

> `奖励函数挑战、In-context Training、GRPO 方法论、微调经验、Mistral 与数值数据限制` 


- **奖励函数设计的挑战**：目前的共识是奖励函数对某些短语过于宽容，导致模型回复中充斥着诸如 "Hmm, let's see..." 之类的重复性输出。
   - 用户强调需要对过度使用的短语实施惩罚，以促进生成内容的多样性。
- **探索 In-context Training**：有建议尝试 In-context Training，通过发送先前消息的滑动窗口来帮助模型理解其回复的演变过程。
   - 该方法旨在改进自我监督，而不是将每次生成视为互不相关的。
- **GRPO 方法的进展**：Group Robust Preference Optimization (GRPO) 方法作为一种针对特定群体偏好优化 LLM 的新颖方法被讨论，强调了对齐用户需求的鲁棒策略。
   - 用户提到了相关的学术论文，强调了定制奖励函数对提升模型性能的重要性。
- **微调见解与经验**：一位用户分享了应用长度惩罚（length penalty）在保持准确性的同时减少了输出长度，展示了有效的微调技术。
   - 对话强调了调整超参数（如奖励函数和训练时长）对于成功进行模型适配的重要性。
- **Transformer 在数值数据方面的局限性**：承认像 Mistral 这样的 Transformer 模型在处理表格数值数据时表现不佳，导致计算结果不一致。
   - 这一见解引发了对某些模型在需要精确数值推理或操作场景中适用性的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.03300">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>：由于数学推理的复杂性和结构化特性，它对语言模型构成了重大挑战。在本文中，我们介绍了 DeepSeekMath 7B，它继续对 DeepSeek-Co 进行预训练...</li><li><a href="https://arxiv.org/abs/2405.20304">Group Robust Preference Optimization in Reward-free RLHF</a>：针对特定任务适配大语言模型（LLMs）通常涉及在偏好数据上通过人类反馈强化学习（RLHF）进行微调。虽然这些数据通常来自不同的...</li><li><a href="https://x.com/AskSathvik/status/1889697491769078270">ASK Sathvik (@AskSathvik) 的推文</a>：RL 确实有效。我借鉴了 Kimi 1.5 论文中的长度惩罚，并在 GSM8K 数据集上训练了 R1-1.5B，生成长度从 >2000 tokens 下降到 <500 tokens，同时保持...</li><li><a href="https://github.com/sathvikask0/r1-distilled-RL">GitHub - sathvikask0/r1-distilled-RL</a>：通过在 GitHub 上创建账号来为 r1-distilled-RL 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1338963390642782231)** (308 条消息🔥🔥): 

> `语音聊天机器人技术、Home Assistant 语音集成、Moxie 机器人伴侣、ESP32 微控制器、ChatGPT 及其他 AI 模型` 


- **探索 DIY 语音聊天机器人**：用户讨论了使用各种现成产品构建语音聊天机器人，包括使用 Raspberry Pi 和 ESP32 的 DIY 项目，并推荐了 Eilik 伴侣机器人等设备。
   - 有人提到通过自定义 3D 打印来美化这些设备，展示了创意与功能的结合。
- **Home Assistant 语音设置**：一名成员分享了使用 Home Assistant Voice 的经验，该功能允许自定义语音助手与 OpenAI API 交互，提供网页搜索和智能家居控制等功能。
   - 该集成需要运行 Home Assistant 服务器，用户可以配置多语言支持，这对多元化社区非常有益。
- **Moxie 机器人的现状**：对话强调了对儿童机器人伴侣 Moxie 的关注，该产品面临的一些问题导致其未来充满不确定性；尽管如此，它仍被视为机器人伴侣情感智能方面的参考。
   - 参与者推测了 Moxie 的潜在继任者，并讨论了其强调与儿童互动的设计特点。
- **机器人与语音助手的集成**：用户正在将玩具机器人连接到语音助手，讨论了使用各种麦克风和设置配置以增强功能的可行性。
   - 分享的经验包括使用 USB 麦克风以获得更好的音频质量，以及集成 MCP server 来控制机器人的计划。
- **AI 模型对比**：对话包括对比 ChatGPT 和 Claude 的个性与响应能力，揭示了用户根据任务需求对不同 AI 模型的偏好。
   - 用户强调了各个版本的功能，重点讨论了模型如何以不同方式适应和处理用户交互。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.04517">xLSTM: Extended Long Short-Term Memory</a>：在 20世纪 90年代，恒定误差轮转（constant error carousel）和门控（gating）作为 Long Short-Term Memory (LSTM) 的核心思想被引入。自那时起，LSTM 经受住了时间的考验，并为众多领域做出了贡献...</li><li><a href="https://community.openai.com/t/webrtc-real-time-api-with-microcontroller/1059806">Webrtc Real Time API with microcontroller</a>：嗨！在第 9 天的演示中，我们看到一个装有微控制器的毛绒玩具正在调用 WebRTC Real-Time API（链接：YouTube 直播）。您能否提供更多关于整体架构的细节？例如...</li><li><a href="https://www.youtube.com/watch?v=7dd_r-aqecw">Rethinking social development with Moxie, a robot companion for kids</a>：凭借蓝色的身体和大大的动漫眼睛，Moxie 想要成为你孩子的朋友。这款由 AI 驱动的机器人被《时代》杂志评为 2020 年最佳发明之一，旨在...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1inmkbc/agenticaorgdeepscaler15bpreview/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1339218726628626514)** (9 条消息🔥): 

> `账户管理改进、自定义 GPT 模型、招聘专家、讨论指南` 


- **账户管理功能增强请求**：一位用户表达了对 ChatGPT 改进的渴望，包括更改账户邮箱、在账户间迁移聊天记录以及批量删除聊天的能力。
   - 另一位成员指出，某些功能此前被设计为受限状态，但分享对话线程仍然可行，这表明了用户对聊天管理功能的不满。
- **关于自定义 GPT 模型的说明**：针对关于自定义 GPT 使用何种模型的问题，一名成员确认它们运行在 **GPT-4o** 上。
   - 这突显了用户对自定义 GPT 配置背后的具体模型实现细节的兴趣。
- **为初创公司寻求专家**：一位用户表示他们需要为自己的初创公司聘请一名专家，邀请有经验的合格成员与其联系。
   - 回复幽默地挑战了该请求的模糊性，建议应明确所需专家的具体领域。
- **讨论频道指南**：频道管理员提醒用户将 GPT 讨论与 ChatGPT 建议分开，并引导他们前往相应的频道。
   - 这强调了组织讨论主题和遵守社区指南的必要性。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1338992028088012842)** (12 条消息🔥): 

> `Iterative Prompting 的优势, Function Calling 问题, Prompt Engineering 最佳实践, 社区参与, Prompt 分享指南` 


- **Iterative Prompting 提升结果**：一位用户分享说，**iterative prompting** 通过从基准开始并不断优化以达到预期结果，确实非常有帮助。
   - 这强调了具体性的重要性，特别是由于 **LLM 无法读心**，需要清晰的指令。
- **Prompt 中的 Function Calling 引发问题**：一位成员讨论了其 system prompt 中 **function calling** 的问题，描述了状态指示有时会失效或根据客户端交互不必要地触发。
   - 他们指出，即使指定了在模糊响应时避免调用函数，性能仍然滞后。
- **Prompt Engineering 最佳实践**：针对函数问题，一位用户建议在详细说明方法之前，先向模型提供关于预期结果的**准确指令**。
   - 该策略旨在减少模型的猜测，从而实现更一致的逻辑执行。
- **社区欢迎 Prompt 分享**：一位热心的用户表达了发布 prompt 的兴趣，社区鼓励在符合频道重点的前提下进行分享。
   - 成员们建议专注于讨论或解决问题，而不是简单地堆砌 prompt，以增强互动。
- **成员受困于 Prompt 长度**：一位用户提到由于 prompt 的复杂性，在输入时遇到困难，表明在分享某些信息时存在限制。
   - 这突显了一些成员在尝试于讨论中贡献详细 prompt 时面临的持续挑战。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1338992028088012842)** (12 条消息🔥): 

> `Iterative Prompting, Function Calling 问题, Prompt 分享礼仪, 明确模型指令` 


- **Iterative Prompting 增强结果**：一位成员强调，**iterative prompting** 有助于持续优化输入以获得更好的 AI 响应。
   - 他们强调从基础 prompt 开始，并迭代改进直到达到预期结果。
- **Prompt 中的 Function Calling 带来挑战**：一位成员表达了在其 system prompt 中使用 **function calling** 的困难，称 AI 有时无法准确指示状态。
   - 他们注意到，在没有相关用户响应的情况下，函数仍被调用，导致潜在的线索流失。
- **欢迎就 Prompt 问题展开公开讨论**：一位新成员分享了他们在 prompt engineering 方面的问题，并邀请大家对此进行**讨论**。
   - 他们引用了系统的函数结构，并对从 AI 接收到的模糊响应发表了评论。
- **讨论 Prompt 分享的最佳实践**：一位成员建议，分享 prompt 时最好附带*问题或观察*，而不是信息堆砌。
   - 他们提到，该频道更适合讨论异常情况，而不仅仅是分享 prompt。
- **AI 清晰指令的重要性**：一位参与者建议，在提供指令之前，要非常明确模型预期要做什么。
   - 他们指出，不清晰的指令可能会导致 AI 对任务产生不同的猜测，从而导致输出不一致。


  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1338964659054579862)** (19 条消息🔥): 

> `Codeium 扩展问题、切换到替代方案、预发布版本使用、支持关注点、近期更新` 


- **Codeium 扩展功能滞后**：成员们对 **Codeium extension** 的进度落后表示沮丧，原因是开发重心转向了 **Windsurf** 和 **Enterprise** 产品。
   - 有人指出，该扩展在企业版选项下仍然可用，这说明了产品重心的双重性。
- **评估替代方案：Cursor vs Codeium**：讨论中提到了切换到 **Cursor**，并建议将其作为 Codeium 的替代品。
   - 然而，有人指出 Cursor 与编辑器扩展并不相同，这让一些用户在两者的差异之间感到纠结。
- **预发布版本冲突**：一名用户询问如何在 **GoLand** 中使用 **Codeium** 的 **pre-released 1.37 版本**，因为没有找到预期的更新按钮。
   - 有人提到，由于标准版中持续存在的 bug，可以选择切换到预发布版本。
- **支持反馈：需要修复 Bug**：用户对 **Codeium** 的支持服务表示担忧，一名用户称其令人失望且毫无帮助。
   - 另一名用户发现，最近的更新大多包含细微调整，而非能显著影响可用性的实质性 bug 修复。
- **最新发布更新公告**：一份公告指出，**Forge** 中的授权问题将在即将发布的 **1.36.1** 版本中得到解决。
   - 尽管如此，一名用户报告称，改进情况严重滞后于预期，甚至提到了 2025 年。


  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1338964644752134225)** (281 条消息🔥🔥): 

> `Windsurf 问题与错误、模型对比、用户反馈与建议、模型综合性能、支持与稳定性关注点` 


- **Windsurf 出现错误和停机**：用户报告了 Windsurf 持续存在的问题，提到在使用 Cascade 时反复出现内部错误，以及 Gemini 模型多次出现问题。
   - 许多人对最近的性能下降表示沮丧，特别是无法编辑文件以及某些功能的不稳定性。
- **AI 模型的效果对比**：一位用户提供了 Windsurf 模型的非官方排名，指出 **Claude 3.5 Sonnet** 因其上下文处理和 tool calling 能力而表现最出色。
   - **Gemini 2.0 Flash** 和 **O3-Mini** 等模型因速度和价格受到称赞，而 **GPT-4o** 等其他模型则因表现不佳受到批评。
- **关于 AI 模型局限性的反馈**：几位用户强调了在使用 AI 时保持警惕的重要性，指出盲目信任 AI 生成的输出可能会导致代价高昂的错误。
   - 对话强调了对 LLM 能力进行更清晰的风险评估和用户教育的需求。
- **LLM 的文档源请求**：用户讨论了在 Windsurf 中添加自定义文档源的可能性，参考了通过 llms.txt 格式有效索引文档的标准方法。
   - 社区希望在这一领域有所改进，以增强功能并提高信息获取的便利性。
- **用户额度管理经验**：讨论围绕在 Windsurf 中优化额度（credit）使用的策略展开，包括利用 Cascade 处理简单任务以节省额度。
   - 用户请求增加额外的方案或选项，表明希望定价结构更具灵活性以满足其需求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://shitposting.pictures/ElRlAJulppNd">一张手动精选的恶搞图片</a>：未找到描述</li><li><a href="https://codeium.com/changelog/windsurf-next">Windsurf Next 更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf Next 扩展的最新更新和更改。</li><li><a href="https://codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>：需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://codeium.canny.io/feature-requests/p/windsurfs-autocomplete-now-working-around-08-35-41202-utc">Windsurf 的自动补全在 UTC 08:35:41.202 左右恢复工作 | 功能请求 | Codeium</a>：在其他时间是正常的，但在 UTC+8 下午停止工作。日志中只显示请求已取消。</li><li><a href="https://status.codeium.com">Codeium 状态</a>：未找到描述</li><li><a href="https://directory.llmstxt.cloud">llms.txt 目录</a>：未找到描述</li><li><a href="https://mintlify.com/blog/simplifying-docs-with-llms-txt">通过 /llms.txt 为 AI 简化文档</a>：为什么我们要为 LLM 提供更好的文档处理方式。</li><li><a href="https://docs.github.com/articles/restricting-access-to-your-organization-s-data/">管理对组织数据的 OAuth 访问 - GitHub 文档</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1338964311917203496)** (240 条消息🔥🔥): 

> `Perplexity AI 模型, Sonar vs DeepSeek R1, API 功能, 实时浏览, Perplexity 用户体验` 


- **关于 Perplexity AI 模型的说明**：用户讨论了 DeepSeek R1 和 Sonar Reasoning Pro 之间的区别，指出 Sonar 是基于 R1 构建并针对网页搜索响应进行了优化。
   - 有建议称 Sonar Reasoning Pro 最终可能会在 Perplexity 应用中取代 DeepSeek R1。
- **Perplexity API 的问题**：多位用户报告在尝试访问 Perplexity API 时遇到 500 内部服务器错误，引发了对其可靠性的担忧。
   - 尽管状态页面显示运行正常，但用户对 API 的性能表示沮丧。
- **实时浏览能力**：一位用户询问 Perplexity 是否提供实时互联网浏览，还是仅限于 2023 年之前的信息。
   - 已确认 Perplexity 可以根据当前链接进行搜索，从而实现灵活的浏览。
- **用户体验投诉**：几位用户对 Perplexity 的回答相比 DeepSeek 不够简洁表示不满。
   - 讨论还强调了对缺乏关于模型版本和能力的清晰文档的挫败感。
- **Perplexity 功能与工具**：频道中提到了一个用户开发的 Chrome 扩展程序，该程序可以突出显示 Perplexity 回答中引用的来源。
   - 成员们询问是否可以为 Firefox 或 Perplexity 的移动版本实现类似功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/guides/model-cards">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/r%C3%A1pido-fast-snail-robot-gif-15498737">Rápido Fast GIF - Rápido Fast Snail - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://pastebin.com/BFmw7FBc">CLAUDE: The Flash 是 DC 漫画中的超级英雄，被称为 "The Scarlet Speedster" - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的文本存储工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://x.com/aravsrinivas/status/1889398205076451697?s=46&t=Un1yLqIRg3sDiqpmnWHBfg">Aravind Srinivas (@AravSrinivas) 的推文</a>：我们在 Llama 3.3 之上微调了一些非常棒的模型，其回答质量远超 4o-mini 和 3.5-Haiku，并能与 4o 和 3.5-Sonnet 持平，而且价格更便宜，速度极快！用户...</li><li><a href="https://x.com/perplexity_ai/status/1889392617479082323?s=61">Perplexity (@perplexity_ai) 的推文</a>：Perplexity 的 Sonar——基于 Llama 3.3 70b 构建——在用户满意度上优于 GPT-4o-mini 和 Claude 3.5 Haiku，同时匹配或超越了 GPT-4o 和 Claude 3.5 Sonnet 等顶级模型。在 1200 tokens...</li><li><a href="https://x.com/pplxfinance/status/1889742180421337120?s=61">Perplexity Finance (@PPLXfinance) 的推文</a>：您的每日最新市场洞察来源——现已在 Perplexity 上线。市场摘要、每日亮点、收益快照，以及您理解背后“原因”所需的一切。Fi...</li><li><a href="https://status.perplexity.com/">Perplexity - 状态</a>：Perplexity 状态页面</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1indfjd/pro_was_600_requests_per_day_then_300_then_now_100/">Reddit - 深入了解任何事物</a>：未找到描述</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1ilzw2e/i_made_a_chrome_extension_to_highlight_evidence/">Reddit - 深入了解任何事物</a>：未找到描述</li><li><a href="https://sonar.perplexity.ai">Sonar by Perplexity</a>：使用由 Perplexity 创建的最佳 AI 回答引擎 API 进行构建。通过最快、最便宜且带有搜索增强（grounding）的产品为您的应用赋能。提供无与伦比的实时、全网范围的响应...</li><li><a href="https://sonar.perplexity.ai/">Sonar by Perplexity</a>：使用由 Perplexity 创建的最佳 AI 回答引擎 API 进行构建。通过最快、最便宜且带有搜索增强（grounding）的产品为您的应用赋能。提供无与伦比的实时、全网范围的响应...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1339019112525074543)** (9 messages🔥): 

> `OpenAI Rebranding, Apple Tabletop Robot Prototype, Largest Structure in Universe, Street Art Discussion, EU AI Investment` 


- **OpenAI 品牌重塑与新闻亮点**：一段名为 *YouTube* 的视频讨论了近期 **OpenAI 的品牌重塑**，以及其他重大事件，包括 **Apple** 的桌面机器人原型和**宇宙中最大结构**的发现。
   - 点击[此处](https://www.youtube.com/embed/9SUxli8UDA0)观看视频以获取详细见解。
- **街头艺术探索**：一位用户分享了一个讨论**街头艺术**及其各种形式和影响的链接。
   - 如需深入了解，请查看[此处](https://www.perplexity.ai/search/tell-me-about-street-art-artri-4PpxVJBsSOWQ_T36v9e7NA)的资源。
- **EU AI 投资洞察**：提出了 **EU AI 投资**的话题，强调了其在 AI 领域的战略重要性。
   - 更多详情可通过[此处](https://www.perplexity.ai/search/eu-ai-investment-aE_wZ53LRUCrT.ntggaGZQ)访问。



**提到的链接**：<a href="https://www.youtube.com/embed/9SUxli8UDA0">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1339200031378509965)** (20 messages🔥): 

> `401 Authorization Required, API 500 Errors, Token Issues` 


- **用户遇到 401 Authorization Error**：一位用户报告在尝试访问 API 时收到 **401 Authorization Required** 错误，并寻求帮助以确定问题所在。
   - *DenoLand* 建议用户需要移除其 Token 周围的 `<>`，但问题仍然存在。
- **初始授权问题的解决**：经过一些排查后，该用户宣布他们解决了问题，API 开始正常工作，并表达了感谢。
   - 他们表示在找到解决方案之前收到了不同的错误消息。
- **广泛的 500 错误报告**：多位用户报告在 API 中遇到 **500 错误**，表明服务已宕机。
   - 评论反映了普遍的沮丧情绪，因为用户注意到生产环境出现故障，尽管某些 API 调用仍然成功。
- **对 API 可用性的紧迫担忧**：用户形容情况不妙，每次 API 调用都持续出现 **500 错误**。
   - 这引发了关于生产环境中 API 服务可靠性的担忧。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1338963567155875953)** (85 条消息🔥🔥): 

> `LM Studio 与模型使用、Deepseek 模型应用、音频模型支持、Markdown 输入渲染问题、编程 LLM 偏好` 


- **理解 LM Studio 模型使用**：成员们讨论了 LM Studio 需要特定的 VRAM 才能高效加载模型，**Duckyblender** 指出模型应完全放入 VRAM 以避免性能问题。
   - 另一位成员确认模型的大小与所需的 VRAM 大致相关，特别是在使用 **Deepseek R1 distill model** 等模型时。
- **Deepseek 模型的应用**：成员们询问了关于将 **Deepseek R1 distill model** 用于数学和推理任务的情况，**Duckyblender** 建议尽管编程不是其主要功能，也可以对其进行测试。
   - 社区成员扩展了该模型的各种潜在用途，强调了其处理复杂问题的能力。
- **音频模型的挑战**：**Heyitsyorkie** 表示 LM Studio 不支持像 **Qwen2-Audio-7B-GGUF** 这样的音频模型，这促使成员们寻找使用音频模型的替代方法。
   - 关于探索外部工具或平台作为处理音频模型的选项，大家给出了建议。
- **Markdown 渲染 Bug**：**Vandalbyte** 报告了一个 Bug，即 Markdown 输入被渲染为格式化文本，而不是显示为原始文本，这在聊天界面中引起了混乱。
   - Bug 追踪器中已开启一个 Issue，强调了 Markdown 渲染中的异常行为并请求进一步检查。
- **编程 LLM 的偏好**：社区讨论显示 **Codestral 22b** 是编程 LLM 的首选，而另一位成员分享了使用 **Claude Desktop** 的经验。
   - 成员们分享了对不同模型的各种看法，并推荐了满足编程需求的替代方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/cat-stare-catstare-cat-stare-sus-catglare-cat-glare-gif-14942558849944709546">Cat Stare Catstare GIF - Cat stare Catstare Cat stare sus - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/430">Markdown 输入在聊天中被渲染而非显示为原始文本 · Issue #430 · lmstudio-ai/lmstudio-bug-tracker</a>：哪个版本的 LM Studio？LM Studio 0.3.9 (Build 6) 哪个操作系统？Windows 11 什么是 Bug？当用户以 Markdown 格式输入文本（例如 # Heading、斜体、加粗）时，它会被渲染...</li><li><a href="https://v0.dev">Vercel 的 v0</a>：与 v0 聊天。通过简单的文本提示生成 UI。复制、粘贴、发布。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1338963101608968323)** (175 条消息🔥🔥): 

> `GPU 对比：3080 Ti vs 3090 vs 4090 vs 5090，PCI-E 带宽在游戏和推理中的重要性，AMD vs NVIDIA：当前性能与建议，5090 GPU 的潜在问题，构建多 GPU AI 配置` 


- **测试 GPU 性能：3090, 4090, 5090**：一位用户正在测试包含 **3090**、**4090** 和 **5090** 的配置，寻求在本地运行模型的建议。
   - 关于 **5090** 的担忧已经出现，特别是有关其故障的报告，引发了对其可靠性的质疑。
- **游戏与推理性能的差异**：讨论强调，在游戏中加载纹理时，性能表现与推理类似，因为两者在 GPU 上都只是数学计算。
   - 有人指出，对于像 **1050 Ti** 这样的低端 GPU，当带宽受限时，性能会大幅下降。
- **AMD 和 NVIDIA GPU 建议**：分享了关于 AMD GPU 在 AI 任务中可行性的看法，指出其 **24 条 PCI-E 通道** 可能会限制多 GPU 配置。
   - 用户讨论了使用 **AMD Threadripper** 以获得更好性能的可能性，因为它提供更多可用通道。
- **关于 5090 GPU 可靠性的担忧**：关于 **5090** 可靠性的担忧进一步加剧，有提到用户烧毁显卡的案例，导致大家行为谨慎。
   - 为了应对这些担忧，一些人建议将 5090 降压（undervolting）作为预防措施。
- **构建多 GPU AI 配置**：一位用户分享了构建多 GPU 服务器的经验，提到了优化性能的特定主板配置。
   - 讨论包括了由于主板限制而使用 x1 链路的各种配置，这挑战了对 GPU 性能的典型预期。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://videocardz.com/newz/amd-reportedly-working-on-gaming-radeon-rx-9000-gpu-with-32gb-memory">AMD reportedly working on gaming Radeon RX 9070 XT GPU with 32GB memory - VideoCardz.com</a>: 据报道 AMD 正在开发配备 32GB 显存的游戏显卡 Radeon RX 9070 XT。Chiphell 有新传闻称，一款据称属于 Radeon RX 9000 系列的显卡配备了比 RX 9070 多一倍的显存容量...</li><li><a href="https://videocardz.com/newz/amd-reportedly-working-on-gamin">AMD reportedly working on gaming Radeon RX 9070 XT GPU with 32GB memory - VideoCardz.com</a>: 据报道 AMD 正在开发配备 32GB 显存的游戏显卡 Radeon RX 9070 XT。Chiphell 有新传闻称，一款据称属于 Radeon RX 9000 系列的显卡配备了比 RX 9070 多一倍的显存容量...</li><li><a href="https://www.techpowerup.com/gpu-specs/tesla-m10.c3035">NVIDIA Tesla M10 Specs</a>: NVIDIA GM107 x4, 1306 MHz, 640 Cores, 40 TMUs, 16 ROPs, 8192 MB GDDR5, 1300 MHz, 128 bit</li><li><a href="https://www.youtube.com/watch?v=L1NPFFRTzLo">NVIDIA RTX 5090 PCIe 5.0 vs. 4.0 vs. 3.0 x16 Scaling Benchmarks</a>: 赞助商：亚马逊上的 Arctic Liquid Freezer III - https://geni.us/NrMtDT。此基准测试对比了 NVIDIA RTX 5090 GPU 在不同 PCIe 版本下的差异。我们正在测试...</li><li><a href="https://youtu.be/COcHHX2MdKs"> - YouTube</a>: 未找到描述</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-4070-ti-super.c4187">NVIDIA GeForce RTX 4070 Ti SUPER Specs</a>: NVIDIA AD103, 2610 MHz, 8448 Cores, 264 TMUs, 96 ROPs, 16384 MB GDDR6X, 1313 MHz, 256 bit</li><li><a href="https://github.com/Nicoolodion/RTX-3070-16GB-GUIDE">GitHub - Nicoolodion/RTX-3070-16GB-GUIDE: A Guide for Modding a RTX 3070 to 16 GB VRAM</a>: 将 RTX 3070 改装为 16 GB VRAM 的指南。通过在 GitHub 上创建账号为 Nicoolodion/RTX-3070-16GB-GUIDE 的开发做出贡献。</li><li><a href="https://youtu.be/kb5YzMoVQyw">How Nvidia made the 12VHPWR connector even worse.</a>: Der8auer 的视频：https://www.youtube.com/watch?v=Ndmoi1s0ZaY。Patreon：https://www.patreon.com/buildzoid。Twitch（主要是游戏直播）：https://www.twitch.tv/b...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1338990315318677595)** (80 条消息🔥🔥): 

> `汤森路透 AI 版权案，当前 AI 融资情况，OpenAI 的 GPT-4.5 和 GPT-5 路线图，GRPO 训练改进，DeepSeek-R1 和推理模型`

- **Thomson Reuters 赢得具有里程碑意义的 AI 版权案**：Thomson Reuters 在美国首例重大 AI 版权案件中[获胜](https://storage.courtlistener.com/recap/gov.uscourts.ded.72109/gov.uscourts.ded.72109.770.0.pdf)，法院支持了该公司针对 Ross Intelligence 复制 Westlaw 材料的指控。
   - Stephanos Bibas 法官表示，*Ross 的任何辩护理由都站不住脚*，并驳回了所有辩护。
- **Current AI 启动重大融资**：[Current AI](https://www.currentai.org/) 旨在引领公益性 AI 领域，初始承诺资金为 4 亿美元，目标是在五年内筹集 25 亿美元，以引导 AI 走向社会福利。
   - 许多人对通过 AI 创造机会和安全性的潜力感到兴奋，其愿景包括从拉各斯（Lagos）到利马（Lima）等不同地区的社区参与。
- **OpenAI 揭晓 GPT-4.5 和 GPT-5 路线图**：OpenAI 的路线图详细说明了即将发布的 GPT-4.5（最后一个非 chain-of-thought 模型），随后将整合 GPT-5，旨在统一其产品线。
   - 免费层级用户将获得对 GPT-5 的无限访问权限，而付费订阅者将拥有增强的功能，包括语音和 Deep Research 等特性。
- **GRPO 训练显著提升性能**：从 PPO 转向 GRPO 使 Tulu 流线的性能增益提升了 4 倍，在 MATH 和 GSM8K 等挑战中表现出显著改进。
   - Costa Huang 分享了他们最新的经过 GRPO 训练的 Tulu 模型的成功，预示了 RL 策略的新方向。
- **DeepSeek-R1 增强推理模型**：DeepSeek-R1 在寻求在生产环境中有效实施推理模型的公司中引起了关注热潮。
   - Together Compute 宣布推出由先进芯片驱动的专用推理集群，以支持大规模和低延迟的 AI 工作负载。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2502.06807">Competitive Programming with Large Reasoning Models</a>：我们展示了将强化学习应用于大型语言模型 (LLMs) 能显著提升其在复杂编程和推理任务中的表现。此外，我们还比较了两种通用推理...</li><li><a href="https://www.bomberbot.com/debugging/how-to-debug-your-code-like-a-competitive-programmer-automate-and-save-time/#:~:text=Start%20with%20a%20brute%20force,The).">How to Debug Your Code Like a Competitive Programmer – Automate and Save Time - Bomberbot</a>：作为程序员，我们花费大量时间调试代码。剑桥大学的一项研究发现，软件开发人员花费了 50% 的...</li><li><a href="https://x.com/togethercompute/status/1889743684977168547">Together AI (@togethercompute) 的推文</a>：自发布 DeepSeek-R1 以来，我们看到大量公司寻求在生产环境中部署推理模型——但如何高效扩展仍是一个挑战。今天，我们正在超越我们的超...</li><li><a href="https://x.com/natolambert/status/1889730488199209393">Nathan Lambert (@natolambert) 的推文</a>：Costa 正试图让 GRPO 在没有 Bug 的情况下全速运行，结果我们的性能远好于去年秋天发布的 Tülu 模型。从 PPO 切换到 GRPO 使增益翻了 4 倍...</li><li><a href="https://x.com/TheXeophon/status/1889762840384266578">Xeophon (@TheXeophon) 的推文</a>：随着 GPT-5 成为一个（更加）黑盒的系统，我希望学术界最终能从付费产品测试员转变为专门使用开源模型。</li><li><a href="https://x.com/Dorialexander/status/1889300494989869464">Alexander Doria (@Dorialexander) 的推文</a>：Common Corpus 2 是对 CurrentAI 的实物捐赠，CurrentAI 是刚刚在 #AISummit 期间成立的开源 AI 新基金会。在 AI Alliance 和机构参与者的支持下，我们共同...</li><li><a href="https://x.com/lmarena_ai/status/1889741530757210524">lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：人们都在问哪些类型的编程语言？根据我们检索到的文件类型，Python 和 Markdown 是目前为止人们提问中最常见的语言。</li><li><a href="https://x.com/stablequan/status/1889560991882416294">qnguyen3 (@stablequan) 的推文</a>：在中国，Apple Intelligence 将由 @Alibaba_Qwen 提供支持。重大利好。</li><li><a href="https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/">Thomson Reuters Wins First Major AI Copyright Case in the US</a>：汤森路透的裁决对生成式 AI 公司与权利持有者之间的博弈具有重大影响。</li><li><a href="https://x.com/replicate/status/1889628772997243034">Replicate (@replicate) 的推文</a>：你好 Claude！https://replicate.com/anthropic Claude 3.5 Sonnet 和 Claude 3.5 Haiku 模型现已在 Replicate 上线。</li><li><a href="https://x.com/nrehiew_/status/1889737259835969735">wh (@nrehiew_) 的推文</a>：在相同的 TULU3 数据集上，GRPO > PPO。这里的直觉是什么？GRPO 难道就是天命所归的 RL 算法吗？引用 Costa Huang (@vwxyzjn) 🔥 allenai/Llama-3.1-Tulu-3-8B (使用 PPO 训练) -> a...</li><li><a href="https://www.currentai.org/">Current AI | Building Public Interest AI Technology Together</a>：加入这项构建服务于公众利益的开放、公平 AI 技术的全球倡议。通过协作和地方行动，我们正在创造惠及所有人的 AI 解决方案。</li><li><a href="https://x.com/NeginRaoof_/status/1889739171826377008">Negin Raoof (@NeginRaoof_) 的推文</a>：发布 OpenThinker-32B：从 DeepSeek-R1 蒸馏出的最佳开源数据推理模型。我们的结果表明，经过验证的 R1 注释的大型、精心策划的数据集可以产生 SoTA 推理模型...</li><li><a href="https://x.com/sama/status/1889755723078443244?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Sam Altman (@sama) 的推文</a>：OPENAI 关于 GPT-4.5 和 GPT-5 的路线图更新：我们希望在分享预期路线图方面做得更好，并在简化产品供应方面做得更好。我们希望 AI 能为你“顺畅运行”；我们...</li><li><a href="https://biz.chosun.com/stock/stock_general/2025/02/12/KAL6SZYMQ5DLTEMGKYSTTIXT5I/">自研芯片失败的 Meta... 能否促成对 FuriosaAI 的并购？</a>：自研芯片失败的 Meta... 能否促成对 FuriosaAI 的并购？Meta 寻求替代方案以降低对 NVIDIA 的依赖。Meta 因自研芯片失败而有充分的收购动机。近期融资的企业估值以 8000 亿韩元为基准。</li><li><a href="https://techcrunch.com/2025/02/10/google-backed-public-interest-ai-partnership-launches-with-400m-pledged-for-open-ecosystem-building/?guccounter=1">Google-backed public interest AI partnership launches with $400M+ for open ecosystem building | TechCrunch</a>：为又一个 AI 合作伙伴关系腾出空间。Current AI 是一项“公益”倡议，专注于培养和引导人工智能的发展。
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1339076985460363356)** (24 条消息🔥): 

> `Notebook LM 替代方案，Long context 模型，Elicit.com 和 PaperQA，Claude 性能，Gemini 能力` 


- **寻找比 Notebook LM 更好的 PDF 聊天工具**：用户讨论了 **Notebook LM** 在与多个 PDF 聊天时的局限性，并询问 **Glean** 或 **Claude** 等替代方案是否能满足他们的需求。
   - *Claude 的性能*受到质疑，特别是在处理大量 PDF 时，因此有人建议使用能够有效处理 long contexts 的工具。
- **Long Context 模型的挑战**：用户对 **Claude/r1** 在与 5-6 个 PDF 交互时性能迅速下降表示担忧，质疑其在处理大量文档时的鲁棒性。
   - 引用的一项关于 long context 评估的研究强调，尽管在处理更大 contexts 方面有所改进，但现有模型仍然表现吃力。
- **探索 Elicit.com 和 PaperQA 选项**：用户分享了使用 **Elicit.com** 的不同体验，对之前的结果表示不满，但也承认其在某些用例中的潜力。
   - 另外，**PaperQA** 被提及为一个由 Eric Schmidt 支持、进展迅速的开源项目，这为基于文档的查询提供了一个更有前景的选择。
- **开发用于文档交互的内部工具**：一位用户正在开发一个自定义工具，通过 **OAI Assistants API** 促进文档查询，强调了用户可选数据源的需求。
   - 尽管该工具具有复杂性，但其长期效用和有效性仍存在不确定性。
- **关于 Elicit 有效性的评价褒贬不一**：**Elicit** 的老用户对其准确性和可靠性表示失望，但注意到该平台仍然活跃。
   - 大家达成共识，认为像 Elicit 类似的工具需要改进，才能对 AI 研究人员产生真正的价值。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05167">NoLiMa: Long-Context Evaluation Beyond Literal Matching</a>: 最近的大语言模型 (LLMs) 支持从 128K 到 1M tokens 的 long contexts。评估这些能力的一种流行方法是大海捞针 (NIAH) 测试，它涉及检索...</li><li><a href="https://github.com/Future-House/paper-qa">GitHub - Future-House/paper-qa: High accuracy RAG for answering questions from scientific documents with citations</a>: 用于从带有引用的科学文档中回答问题的高精度 RAG - Future-House/paper-qa
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1339111478262894602)** (21 条消息🔥): 

> `Grok 3 争议、从 xAI 辞职、Grok 3 性能推测、xAI 的言论自由担忧、DeepSeek 对比` 


- **Grok 3 争议导致辞职**：一名辞职的员工表示，他被迫删除了一条承认 **Grok 3** 存在的推文，xAI 将其归类为机密，这导致他决定离开公司。他对自己这种明显的个人观点可能威胁到工作表示失望，并断言这似乎与公司声称支持言论自由的立场相悖。
   - *一篇关于 Grok 3 并被标记为“观点”的帖子竟然被视为解雇的理由，这太荒谬了。*
- **xAI 的反应引发讨论**：成员们讨论了 xAI 强制要求对 Grok 3 的存在保持沉默的奇特决定，并解读了这可能如何反映出内部对言论自由的压力。一些人推测，该员工关于未发布产品性能的言论是否影响了促使其辞职的压力。
   - *他们指出，目前的普遍情绪似乎认为 xAI 的立场与其宣称的言论自由倡导相矛盾。*
- **与 DeepSeek 的对比**：针对 Grok 3 预期的性能，一些人表示怀疑，认为如果 Grok 3 仅勉强超过 **DeepSeek** 的能力，考虑到其可用的巨大算力资源，这将令人失望。一位成员提醒说，*DeepSeek 是在配置低得多的硬件上运行的*。
   - *关于 Grok 3 源自 **DeepSeek** 蒸馏或微调的猜测甚嚣尘上，引起了研究人员的关注。*
- **AI 模型未来的推测**：一些成员推测，传闻中 Grok 3 和 **Llama 4** 的性能可能会让人们重新信任来自 OpenAI、Anthropic 和 Gemini 的模型，这可能揭示出研究相对于纯 GPU 算力的隐藏优势。这一讨论指向了 AI 模型开发和评估中的竞争动态。
   - *有人提醒，研究人员可能掌握着超越单纯硬件规格的有价值的创新。*
- **对信息共享的复杂反应**：在这场风波中，一位参与者表示更倾向于被加入 **blocklists**（黑名单），认为这是一种更全面的管理异议的方法，突显了话语环境中极化的氛围。这种情绪反映了在充满争议的环境中如何管理信息和观点的更广泛主题。
   - *参与者承认，围绕观点的分歧，特别是在竞争激烈的 AI 领域背景下，制造了复杂的沟通挑战网。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/BenjaminDEKR/status/1888331926630510638">来自 Benjamin De Kraker (@BenjaminDEKR) 的推文</a>：目前的排名（个人观点），代码方面：ChatGPT o1-pro、o1、o3-mini（基本持平）、Grok 3（预期，待定）、Claude 3.5 Sonnet、DeepSeek、GPT-4o、Grok 2、Gemini 2.0 Pro 系列（可能会更高，可能...）</li><li><a href="https://x.com/vikhyatk/status/1889535819997725008">来自 vik (@vikhyatk) 的推文</a>：引用 Benjamin De Kraker (@BenjaminDEKR)：“我今晚从 xAI 辞职了。这让我非常难过，但是正确的做法——原因如下。xAI 告诉我，要么删除下面引用的帖子，要么面临被解雇...”</li><li><a href="https://fxtwitter.com/BenjaminDEKR/status/1889526713735905502">来自 Benjamin De Kraker (@BenjaminDEKR) 的推文</a>：我今晚从 xAI 辞职了。这让我非常难过，但是正确的做法——原因如下。xAI 告诉我，要么删除下面引用的帖子，要么面临被解雇。在审查了所有...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1339049850876199026)** (78 messages🔥🔥): 

> `配音技巧、Claude 推理模型、OpenThinker-32B 发布、核物理学家类比、RL 综述挑战` 


- **改进配音呼吸技巧**：成员们交流了关于减少配音录制中明显呼吸声的建议，包括 OBS 设置和保持自然的说话风格。
   - 一位成员提到，他们发现 Riverside 的降噪功能有助于提高音频质量。
- **Claude 的扩展思考模式**：关于 Claude 提示词响应时间的讨论突显了其潜在的新“思考”指示器，这在最近的提示过程中被观察到。
   - 成员们推测这是否预示着一个新的推理模型，但对于底层系统是否发生变化存在一些分歧。
- **OpenThinker-32B 模型发布**：讨论了 **OpenThinker-32B** 推理模型的发布，透露其在减少审查的同时能获得良好的性能。
   - 团队成员表示，这一进展源于社区对模型审查的担忧以及对减少限制输出的渴望。
- **核物理学家类比**：一位成员将现代 ML 科学家与早期的核物理学家进行了对比，引用了历史上关于技术积极影响的理想主义。
   - 他们反思了重大地缘政治发展后观点的转变，并将其与现代对 AI 的担忧相类比。
- **RL 综述中的挑战**：针对汇编 RL 结果的复杂性提出了担忧，成员们讨论了现有数据的压倒性特征。
   - 一位成员表示，由于所需精力巨大，他们计划提供有关该主题的评论，而非详细的综述。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://neurips.cc/virtual/2024/tutorial/99526">NeurIPS Tutorial Opening the Language Model Pipeline: A Tutorial on Data Preparation, Model Training, and Adaptation</a>：未找到描述</li><li><a href="https://x.com/__nmca__/status/1889741584922751092">Nat McAleese (@__nmca__) 的推文</a>：@stalkermustang @ahelkky o3 采样了许多解决方案并使用学习到的函数来挑选最佳方案 —— 对于 codeforces，我们为每个问题采样了 1,162 个样本</li><li><a href="https://fxtwitter.com/kernelkook/status/1889678407346106418">sanchay (@kernelkook) 的推文</a>：我认为这很快就会到来。最近给 Claude 发了一个提示，注意到它在回答前显示了大约 7-8 秒的“thinking...”，这不像是平时的行为。无法...</li><li><a href="https://tenor.com/view/avatar-aang-aang-atla-avatar-the-last-airbender-avatar-gif-23087281">Avatar Aang Aang GIF - Avatar Aang Aang Atla - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://fxtwitter.com/naklecha/status/1889662581180183039">naklecha (@naklecha) 的推文</a>：在过去的 2 周里，我在 GRPO 和 SFT+GRPO 组合上运行了数百次训练。到目前为止，我发现的最酷的奖励作弊（reward hacking）例子是 —— 当我惩罚错误 token 的高置信度 logprobs 时...</li><li><a href="https://x.com/madiator/status/1889772019492987225">Mahesh Sathiamoorthy (@madiator) 的推文</a>：我们不小心去除了模型的审查！我们使用的 Qwen-instruct 是经过审查和对齐的。DeepSeek-R1 蒸馏模型也是经过审查和对齐的。当我们使用数学领域的推理数据对 Qwen 模型进行 SFT 时...</li><li><a href="https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use">ChatGPT 消耗多少能量？</a>：本期 Gradient Updates 探讨了 ChatGPT 每次查询消耗多少能量，显示其比常见估计低 10 倍。</li><li><a href="https://youtu.be/64E9O1Gv99o?si=Bi5YLxNkYKOt8bRa&t=575)"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/qPzZeP7t5ZQ?si=czaITQnSdRyA6tCi">Language Modeling: A Tutorial on Data Preparation, Model Training, and Adaptation</a>：开启语言模型流水线：关于数据准备、模型训练和适配的教程。Kyle Lo · Akshita Bhagia · Nathan Lambert 如果你想...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1339065561585422367)** (5 条消息): 

> `高盛 AI 面试争议、Deepseek 模型见解、OpenAI 策略转变` 


- **高盛传阅 AI 生成的面试内容**：[Kevin Xu](https://x.com/kevinsxu/status/1889504794420646039) 指出，高盛分享了一段他们声称是与梁文锋（Liang Wenfeng）的“新”采访，但实际上是根据他 2024 年 7 月的采访生成的 AI 音频。
   - *如果你是高盛的客户，请停止给他们送钱。* 相反，Xu 建议订阅来自 [Jordan Schnitzer](https://x.com/jordanschnyc) 或 [Nat Lambert](https://x.com/natolambert) 等来源的 Newsletter 以获取真正的见解。
- **硕士论文中提到的 Deepseek 模型**：一位成员提到他们在硕士论文中引用了第一个 **Deepseek** 模型，并幽默地建议自己应该加入高盛。
   - 这一评论表明 Deepseek 在学术界被视为具有重要意义。
- **对 Deepseek 公告的期待升温**：[AK](https://x.com/_akhaliq) 透露明天将揭晓关于 **Deepseek** 的重大消息，引发了成员们的兴奋。
   - 这暗示了即将到来的进展可能会对该领域产生重大影响。
- **OpenAI 的策略受到审视**：[Sam Altman](https://x.com/sama) 宣布了 OpenAI 策略的转变，承认仅仅通过扩展模型规模和资源已不再是实现 AGI/ASI 的有效途径。
   - 随着即将发布的 **GPT-4.5** 和 **GPT-5**，他们的目标是简化产品并统一模型以实现更广泛的应用，摆脱复杂的 Model Picker。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/kevinsxu/status/1889504794420646039">来自 Kevin Xu (@kevinsxu) 的推文</a>: 刚刚听说“高盛”本周早些时候向其买方客户传阅了一份简报，其中包含梁文锋在 R1 发布后进行的“新”采访，以显示他们掌握内幕...</li><li><a href="https://x.com/stanfordnlp/status/1889768783834976431">来自 Stanford NLP Group (@stanfordnlp) 的推文</a>: 最终承认 OpenAI、Anthropic 等公司 2023 年的策略（“仅仅扩大模型规模、数据、算力和资金投入就能让我们实现 AGI/ASI”）已不再奏效！引用 Sam Altman ...</li><li><a href="https://x.com/untitled01ipynb/status/1889751694365388821">来自 loss (@untitled01ipynb) 的推文</a>: AK 看到了什么。引用 AK (@_akhaliq)：明天将有关于 Deepseek 的重大消息发布
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1339259588247289979)** (18 条消息🔥): 

> `《科学的诞生》、Jeff Dean 和 Noam Shazeer 的 Dwarkesh 播客、元科学与科学史讨论` 


- **读者对《科学的诞生》表现出兴趣**：一位成员分享了 [David Wootton 所著的《科学的诞生》（The Invention of Science）的链接](https://www.inventionofscience.com/)，该书于 2015 年 9 月 17 日出版，提供多种格式。
   - Metascience 爱好者表达了对该类型的喜爱，而其他人则将其列入阅读清单。
- **Dwarkesh 播客的标志性时刻**：围绕最近一期由 **Jeff Dean** 和 **Noam Shazeer** 参加的节目展开了讨论，该节目强调了他们对现代计算和 LLM 的影响，通过 [此 Substack 链接](https://open.substack.com/pub/dwarkesh/p/jeff-dean-and-noam-shazeer?r=68gy5&utm_medium=ios) 分享。
   - 听众对 Dean 关于 Google 未来的见解以及 Shazeer 关于全球 GDP 的大胆主张表示兴奋，并指出了该集在科技界的重大意义。
- **对未来播客亮相的兴趣**：多位成员鼓励某人参加 Dwarkesh 播客，认为这将是一次有趣的经历，尽管目前这并非首要任务。
   - 还有人开玩笑地建议邀请 **Xeophon** 参加 Dwarkesh，并承诺会有有趣的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://open.substack.com/pub/dwarkesh/p/jeff-dean-and-noam-shazeer?r=68gy5&utm_medium=ios">Jeff Dean &amp; Noam Shazeer – 在 Google 的 25 年：从 PageRank 到 AGI</a>: Gemini 的两位共同负责人探讨 Google 的 AGI 之路</li><li><a href="https://www.inventionofscience.com/">The Invention of Science | David Wootton 著</a>: 无描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1339223904870465609)** (13 messages🔥): 

> `LLM 的命名方案、对 Kuhn 著作的哲学观点、LLM 与 LRM 的未来` 


- **RLMs 中不一致的命名方案**：讨论揭示了对命名惯例的困惑，**OpenAI** 将其称为 **LRMs**，而其他人则简单地标记为 **LLMs**。
   - 一位参与者指出，带有 'L' 前缀意味着 'large'，但该术语是不必要的。
- **哲学家挑战 Kuhn**：哲学家们开始出现一些观点，认为 **Kuhn 的书** 已经过时，但对这一主张几乎没有详细阐述。
   - 这种情绪似乎缺乏实质性的讨论，成员们要求澄清和深入理解。
- **对 LLM 未来的怀疑**：一位成员表达了一个“并不看好”的观点，断言 **LLMs** 注定失败，只有 **LRMs** 会胜出。
   - 这一观点得到了其他人的呼应，表明目前所有的 LLMs 最终都可能被视为 LRMs 的变体。


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1338976597629206569)** (7 messages): 

> `糟糕的幻灯片、讲师的担忧、课程内容、学生的“永恒化”` 


- **学生在幻灯片上“永恒化”**：一位成员幽默地指出，“在我那糟糕的幻灯片上永垂不朽”，并附上了一张展示可能不太理想内容的图片。
   - 另一位成员回复道 *"ohno lol"*，表示对质量的共同笑点或担忧。
- **对学生体验的担忧**：一位成员通过陈述“这些可怜的学生（我错过了什么 🥹）”表达了对学生的担心，暗示了一种不足感。
   - 在这种担忧中，一位成员幽默地评论道 *"Dos hombres fuerte y rico"*，为对话增添了一些轻松气氛。
- **对课程内容的困惑**：另一位成员反映了困惑，问道 *"wtf did I do wrong"*，可能表示对课程交付或内容的不满。
   - 一位成员回应说，这要么是当前课程的改进，要么是可能回到“另一个关于 parse trees 的课程”，暗示了令人不满的替代方案。


  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1339214661098410077)** (8 messages🔥): 

> `Deep Research 与 O1 Pro 组合、ChatGPT UX 反馈、RL 问题误导信息、在 ODR Chat 中切换模型` 


- **Deep Research 和 O1 Pro 构成了强大的组合**：一位成员尝试了一种策略，先从 **Deep Research** 开始，然后使用 **O1 Pro** 进行后续跟进，从而在回答中获得丰富的上下文。
   - 他们在询问一个 **Reinforcement Learning** (RL) 问题时遇到了误解，但通过事先搜索解决了问题。
- **用户对 ChatGPT UX 表示担忧**：一位成员批评了切换模型的 **User Experience (UX)**，指出从 **O1 Pro** 开始可能会无意中导致不必要的额外点击。
   - 大家一致认为该功能看起来很蹩脚（janky），因为一些成员提到虽然可以切换模型，但发现它不够直观。
- **ChatGPT 蹩脚的功能引发评论**：用户形容 **ChatGPT** 很 'janky'，强调了对模型切换和操作的挫败感。
   - 对话反映了对平台 UI 和操作机制更广泛的不满。
- **对 Reinforcement Learning 缩写词的困惑**：一位用户在处理一个 RL 缩写词时感到吃力，最初在没有上下文的情况下将 **GRPO** 解释为 **Gaussian Regularized Proximal Policy Optimization**。
   - 他们承认事先进行彻底的研究纠正了误解，强调了上下文的重要性。
- **分享对话揭示了隐藏功能**：一位成员提到分享他们的对话导致在 **ODR chat** 中成功切换到 **O1 Pro**，这表明了协作排查问题的价值。
   - 这突显了分享经验如何启发平台中被忽视的功能。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1339001334665580587)** (22 条消息🔥): 

> `深度学习 Loss 问题、Deepfrying 现象、Pythia 的安全训练、数据集 Perplexity 评估、对 Polyglot 项目的兴趣` 


- **应对深度学习 Loss 困境**：一位用户表达了在 **72B 模型**中遇到**剧烈且不断增加的 Loss** 的挫败感（相比于更小的模型），并怀疑高 Learning Rates 可能不是唯一的问题。
   - *另一位用户建议，训练期间使用的序列长度可能会直接影响性能*。
- **理解模型训练中的 Deepfrying**：对话揭示了 **Deepfrying** 指的是模型经历逐渐增加的方差，导致 Loss 尖峰升高的状态。
   - *主要贡献者指出，高 Learning Rates 和短序列长度可能会加剧这一问题*。
- **Pythia 安全训练咨询**：一位成员询问了 **Pythia** 抵抗对抗性攻击的能力，特别是它是否针对 Jailbreaks 进行了安全训练。
   - *另一位参与者确认 Pythia 完全没有经过安全训练*。
- **数据集 Perplexity 评估效率**：一位用户寻求关于**数据集 Perplexity 评估**的**高效实现**建议。
   - *另一位用户要求澄清该请求中针对的是哪些方面的效率*。
- **对 Polyglot 项目的兴趣**：一位新成员加入并表达了对**多语言能力**、**语言学习**，特别是对 EleutherAI 的 **Polyglot 项目**的兴趣。
   - *他们表达了在社区内学习和吸收知识的热情*。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1338964146095390842)** (198 条消息🔥🔥): 

> `长短期记忆模型、Transformer 中的记忆、自动化能力发现、通过 Self-Play 进行强化学习、递归推理缩放` 


- **超长上下文的长短期记忆模型探索**：Magic 最近的更新介绍了可以处理高达 **100M tokens** 上下文的长短期记忆模型，增强了超越传统训练方法的推理能力。
   - *这一进步通过将广泛的代码库和文档集成到模型训练的上下文中，为软件开发带来了重大机遇*。
- **质疑 LM2 中 Memory Slots 的实现**：人们对 LM2 模型中 Memory Slot 实现的透明度表示担忧，指出作者并未清楚地描述在其架构中如何选择或更新 Memory Slots。
   - *参与者对该设计的有效性和可并行性表示怀疑，认为论文中的描述可能过于简化*。
- **介绍自动化能力发现 (ACD)**：一个名为自动化能力发现 (ACD) 的新框架旨在以系统化的方式自我探索模型能力，识别 Foundation Models 中意想不到的能力和弱点。
   - *ACD 的运作方式是将一个 Foundation Model 指定为“科学家”，为其他模型提出任务，从而以更少的人力投入提高评估准确性*。
- **为 LRM 提出的通过 Self-Play 进行强化学习**：一个提出的框架——通过 Self-Play 进行强化学习 (RLSP)，专注于通过在强化学习期间解耦探索和正确性信号来训练 Large Reasoning Models。
   - *该方法涉及通过演示进行微调，随后进行由探索奖励信号支持的 RL 训练，目标是实现高效的推理行为而不奖励对漏洞的利用 (Exploitation)*。
- **重访递归推理缩放 (RINS)**：递归推理缩放 (RINS) 基于语言模型的推理时间缩放，通过受分形几何启发的复杂推理方法来增强性能。
   - *讨论引发了对 RINS 方法新颖性及其对现有模型影响的担忧，质疑其是否引入了重大进展*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://arxiv.org/abs/2502.06807">Competitive Programming with Large Reasoning Models</a>: 我们展示了将强化学习应用于大语言模型 (LLMs) 能显著提升其在复杂编程和推理任务中的表现。此外，我们对比了两种通用推理...</li><li><a href="https://arxiv.org/abs/2410.02416">Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models</a>: 无分类器引导 (Classifier-free guidance, CFG) 对于提高扩散模型的生成质量以及输入条件与最终输出之间的一致性至关重要。虽然高引导尺度通常...</li><li><a href="https://arxiv.org/abs/2502.02996">Building Bridges between Regression, Clustering, and Classification</a>: 回归，即根据某些特征 x 预测连续标量目标 y 的任务，是机器学习和统计学中最基础的任务之一。据观察和理论研究...</li><li><a href="https://arxiv.org/abs/2502.07503">Harnessing Language&#39;s Fractal Geometry with Recursive Inference Scaling</a>: 语言建模的最新研究揭示了两种缩放效应：一种是众所周知的通过增加训练算力带来的提升，另一种则是通过应用更复杂或计算量更大的递归推理缩放 (Recursive Inference Scaling) 带来的提升...</li><li><a href="https://arxiv.org/abs/2502.06049">LM2: Large Memory Models</a>: 本文介绍了大内存模型 (LM2)，这是一种仅解码器 (decoder-only) 的 Transformer 架构，通过辅助内存模块进行了增强，旨在解决标准 Transformer 在多...</li><li><a href="https://arxiv.org/abs/2501.15420">Visual Generation Without Guidance</a>: 无分类器引导 (CFG) 已成为各种视觉生成模型中的默认技术，但它在采样过程中需要来自条件模型和无条件模型的推理。我们建议...</li><li><a href="https://arxiv.org/abs/2408.00677">Scaling Backwards: Minimal Synthetic Pre-training?</a>: 预训练和迁移学习是当前计算机视觉系统的重要基石。虽然预训练通常在大型真实世界图像数据集上进行，但在本文中我们探讨...</li><li><a href="https://arxiv.org/abs/2502.06772">ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates</a>: 我们展示了通过缩放思维模板进行分层 LLM 推理可以有效优化推理搜索空间，并超越如...等强大 LLM 的数学推理能力。</li><li><a href="https://arxiv.org/abs/2502.06773">On the Emergence of Thinking in LLMs I: Searching for the Right Intuition</a>: 最近的 AI 进展（如 OpenAI 的新模型）正在将 LLM 转化为 LRM (Large Reasoning Models)，这些模型在推理阶段执行推理，通过额外的推理时间和算力来获得更高质量的...</li><li><a href="https://arxiv.org/abs/2502.07527">NatureLM: Deciphering the Language of Nature for Scientific Discovery</a>: 基础模型 (Foundation models) 彻底改变了自然语言处理和人工智能，显著增强了机器理解和生成人类语言的方式。受...成功的启发</li><li><a href="https://arxiv.org/abs/2211.04800">Designing Network Design Strategies Through Gradient Path Analysis</a>: 设计高效且高质量的表达性网络架构一直是深度学习领域最重要的研究课题。当今大多数网络设计策略...</li><li><a href="https://arxiv.org/abs/2402.13616">YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information</a>: 今天的深度学习方法专注于如何设计最合适的目标函数，使模型的预测结果能够最接近地面真值 (ground truth)。同时，一个合适的...</li><li><a href="https://x.com/jeffclune/status/1889568685632667672">Tweet from Jeff Clune (@jeffclune)</a>: 介绍自动化能力发现 (Automated Capability Discovery, ACD)！ACD 通过“自我探索”（模型探索其自身...）自动识别基础模型中令人惊讶的新能力和失败模式。</li><li><a href="https://arxiv.org/abs/2502.07577">Automated Capability Discovery via Model Self-Exploration</a>: 基础模型已成为通用助手，通过在互联网规模的数据上进行训练，在众多领域展现出多样化的能力。精确描述每个...仍然具有挑战性。</li><li><a href="https://arxiv.org/html/2501.01257v2">CodeForces: Benchmarking Competition-level Code Generation of LLMs on CodeForces Disclaimer: This is a non-traditional code benchmark.</a>: 未找到描述</li><li><a href="https://magic.dev/blog/100m-token-context-windows">100M Token Context Windows — Magic</a>: 关于超长上下文模型的最新研究进展、我们与 Google Cloud 的合作伙伴关系以及新融资。</li><li><a href="https://github.com/SmerkyG/gptcore/blob/main/model/experimental/me">

mtention.py">gptcore/model/experimental/memtention.py at main · SmerkyG/gptcore</a>: 用于创建和训练前沿 LLM 的快速模块化代码 - SmerkyG/gptcore
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1339264670024077443)** (4 messages): 

> `微调与识别，AI 中的可测试假设` 


- **当前工作的协作**：一名成员表示，正在进行的讨论与他们当前的项目高度契合，表明在该话题上有积极的参与。
   - 另一名成员表现出热情，表示“很高兴听到这个消息！”，显示了社区的支持。
- **使用助记模式进行微调**：一名成员询问当前工作是否涉及包含助记字符串（mnemonic strings）的微调方法，特别是模型如何“识别”拼写出“HELLO”之类的模式。
   - 他们提到在这方面有一个“可测试的假设”，预示着进一步实验探索的潜力。


  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1338976640218300528)** (167 messages🔥🔥): 

> `Cursor 文档更新、MCP 服务器问题、O3-Mini 性能、AI 服务定价模型、Claude 模型性能` 


- **Cursor 关于定价和模型的更新**：Cursor 更新了他们的文档，明确了基于用量的定价（usage-based pricing）以及可用模型的详细信息，包括哪些是免费的，哪些不是。
   - 成员们注意到 **deepseek R1** 和 **O3-mini** 等模型现在已列入定价结构，这对其高级会员状态引发了一些困惑。
- **MCP 服务器集成的挑战**：用户报告了 MCP 服务器集成的问题，特别是 **Perplexity API**，导致使用过程中出现不一致和错误。
   - 几位用户成功解决了他们的问题，并建议了故障排除步骤，如硬编码 API 密钥和删除冲突的包。
- **O3-Mini 不稳定的性能表现**：用户对 **O3-mini** 性能的不稳定性表示担忧，根据上下文的不同，用户既体验到了成功的输出，也遇到了幻觉输出。
   - 持续的讨论表明，虽然 O3-mini 偶尔能提供令人印象深刻的改进，但其不一致性仍然是一个主要的挫败点。
- **AI 模型的定价对比**：围绕使用各种 AI 模型的负担能力展开了讨论，特别是 **MCP Perplexity** 与 **Claude** 和 **O3-mini** 等其他模型相比的性价比。
   - 成员们分享了他们的经验，指出只要能有效地管理 Token 使用量，尤其是在长时间交互中，最近使用这些模型仅产生极低的成本。
- **对未来 Anthropic 模型的期待**：对即将发布的 **Anthropic** 模型的期待在增长，用户表达了他们对当前模型（如 **Claude Sonnet**）如何有效处理各种任务的看法。
   - 社区似乎渴望改进，特别是与未来迭代所承诺的功能和能力相关的改进。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://viggle.ai/)">Viggle AI | 可控 AI 视频生成器</a>：用 AI 赋予你的角色生命。从专业的动作捕捉到病毒式传播的迷因，发现使用 Viggle 创作的无限方式。</li><li><a href="https://docs.cursor.com/settings/models">Cursor – 模型</a>：未找到描述</li><li><a href="https://docs.cursor.com/account/usage#usage-based-pricing">Cursor – 用量</a>：未找到描述</li><li><a href="https://half-single-ecd.notion.site/Experiment-Prompting-86aa8f988fce404cbf70134690d2635a?pvs=4">Notion – 笔记、任务、维基和数据库的一体化工作区。</a>：一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作区。</li><li><a href="https://fireworks.ai/models/fireworks/deepseek-r1">Fireworks - 生成式 AI 的最快推理</a>：以极快的速度使用最先进的开源 LLM 和图像模型，或者使用 Fireworks AI 免费微调并部署您自己的模型！</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1inoi6b/openai_silently_rolls_out_o1_o3mini_and_o3mini/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=RCFe1L9qm3EI">Cursor + MCP 服务器：完整设置指南（顺序思维、Brave 搜索等）</a>：Cursor 刚刚添加了 MCP 支持！在这份完整的设置指南中，我将向您展示如何集成和使用 MCP 服务器（Sequential Thinking、Brave Search 和 Puppe...）</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>：未找到描述</li><li><a href="https://status.cursor.com/">Cursor 状态</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1339057086746857492)** (2 messages): 

> `NVIDIA GB200 images, Discord server purpose` 


- **关于 NVIDIA GB200 图片的咨询**：一名成员询问这是否是一个专门发布 **NVIDIA GB200 露骨图片**的服务器。
   - 另一名成员确认道：“*是的*”。
- **服务器确认**：关于服务器内容涉及 NVIDIA GB200 图片的询问迅速得到了另一名成员的确认。
   - 这一交流突显了社区对此类讨论的开放态度。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1339174631730581596)** (1 messages): 

> `Error in default mode vs INTERPRET mode, Triton kernel comparison, Matrix multiplication differences` 


- **错误对比：默认模式 vs INTERPRET 模式**：一名成员提出了一个问题，即在进行简单的 2D 矩阵乘法时，为什么 **默认模式** 下的 **error** 明显大于 **INTERPRET 模式**。
   - 他们引用了一个讨论特定差异的 [GitHub issue](https://github.com/triton-lang/triton/issues/5895)，指出在 INTERPRET 模式下，误差低至 **9.5367431640625e-07**。
- **Triton Kernel 与 Torch 的对比**：讨论强调了他们的 Triton kernel 与 **Torch** 在默认模式和 INTERPRET 模式下执行时的差异，并对性能指标提出了疑问。
   - 更多见解可以在链接的 GitHub issue 中找到，其中详细说明了执行过程中注意到的不一致之处。



**提到的链接**：<a href="https://github.com/triton-lang/triton/issues/5895">error is significantly larger in default mode than INTERPRET mode · Issue #5895 · triton-lang/triton</a>: 描述 Bug：对于简单的 2D 矩阵乘法，我的 Triton kernel 与 Torch 之间的差异为：INTERPRET 模式：9.5367431640625e-07 (设置 os.environ["TRITON_INTERPRET"] = "1" ...

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1339136840673792040)** (7 messages): 

> `CUDA Memory Model Confusion, PTX Instruction Explanation, Blackwell Tensor Memory Management` 


- **CUDA 内存模型引发疑问**：一位 CUDA 初学者分享了对 PMPP 中一段代码片段的担忧，认为其在 load/store 操作方面违反了 **C++ 内存模型**，并分享了他们在 **Stack Overflow** 上的发现：[链接](https://stackoverflow.com/questions/79429440/cuda-memory-model-why-acquire-fence-is-not-needed-to-prevent-load-load-reorderi)。
   - 他们强调，虽然源代码顺序可能确保正确的行为，但在新的编译器或硬件上可能会导致 Bug，因此主张使用 **acquire/release 注解**。
- **PTX 指令澄清**：一名成员询问 PTX 指令 `ldmatrix.sync.aligned.m8n8.x4.b16` 是否意味着每个寄存器持有一个完整的 **8x8 矩阵**或仅是一部分（基于数据类型为 **f16**）。
   - 另一名成员解释说，寄存器定义是 **per thread（每个线程）** 的，每个线程确实可以加载对应于一个 **8x8 矩阵**的值。
- **关于 Blackwell Tensor Memory 的见解**：一位用户确认了 Blackwell GPU 上新的 **tensor memory** 是否由硬件管理且比共享内存更快，但目前尚不清楚其与 **L1 cache** 的对比情况。
   - 一名成员澄清说 **tensor memory** 是 **软件管理** 的，并指向 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-memory-alloc-manage-instructions) 以获取有关专用分配函数的详细信息。



**提到的链接**：<a href="https://stackoverflow.com/questions/79429440/cuda-memory-model-why-acquire-fence-is-not-needed-to-prevent-load-load-reorderi.">CUDA memory model: why acquire fence is not needed to prevent load-load reordering?</a>: 我正在阅读《Programming Massively Parallel Processors》一书，并注意到了以下实现“多米诺风格”扫描的代码片段：&#xA;if (threadIdx.x == 0) {&#x...

  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1338982389711376444)** (18 条消息🔥): 

> `CPUOffload 功能, CPU optimizer step, Backward pass 推导, Shared memory 技术, Gradient checking 问题` 


- **理解 CPUOffload**: 成员们正在研究 `CPUOffload` 的工作原理，特别是在将 DTensor 分片汇聚到 rank 0 进行 optimizer 更新时，如何避免沉重的开销。
   - 有人建议使用 `mmap()` 或 `shm_open()` 等 Shared memory 技术来提高效率，因为管理 GPU 与 CPU 之间的 Tensor 数据传输可以简化操作。
- **CPU optimizer step 的复杂性**: 一位成员正在寻求一种在 rank 0 上执行与 gradient clipping 融合的 **CPU** optimizer step 的技术，旨在利用 reduce 后的梯度，同时避免传统的 allreduce 设置。
   - 讨论中涉及了该方法的可行性，指出由于可能使用并行处理，将所有内容集中在 rank 0 可能不会成为瓶颈。
- **自动化 backward pass 推导**: 一位成员对推导复杂 **forward()** 函数的 backward pass 表示沮丧，正在寻求除 `sympy` 之外更高效的方法或自动化工具。
   - 建议包括利用带有特定日志记录的 `torch.compile` 来辅助理解计算图，尽管也提到了对优化的担忧。
- **Gradient checking 的挑战**: 有人对 `gradgradcheck()` 中观察到的不一致性表示担忧，特别是关于输出被意外抵消或求和的问题，这导致了困惑。
   - 一位成员指出，返回零矩阵可能会使验证过程复杂化，建议在将潜在问题提交到 GitHub 之前进行进一步检查。
- **澄清导数计算**: 讨论强调了 double backward 计算的复杂性，成员们承认在有效跟踪计算图方面存在挑战。
   - 大家认识到，虽然自动微分提供了梯度，但为了清晰和正确，某些手动简化可能仍然是必要的。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息): 

iron_bound: https://github.com/RC4ML/LoHan
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1339054178521059459)** (27 条消息🔥): 

> `CUDA Installation Issues, Global Memory Coalescing in CUDA, Feedback on CUDA Code Structure, Error Handling in CUDA, Memory Coalescing Visualization` 


- **CUDA 安装排查**：一位用户报告在 PC 上运行 CUDA 时遇到困难，在多次尝试安装后遇到了 'cudafe++' 访问冲突错误。
   - 他们寻求解决此问题的建议，强调了在安装 CUDA toolkit 时的困扰。
- **理解全局内存合并 (Global Memory Coalescing)**：一位用户通过 [Simon Boehm 的博客文章](https://siboehm.com/articles/22/CUDA-MMM) 学习了全局内存合并，并讨论了矩阵乘法中的索引方案。
   - 他们指出实现中的不一致导致了重复索引，并寻求关于 block 维度正确性的澄清。
- **寻求 CUDA 代码设计反馈**：一位 CUDA 初学者请求对其代码结构和设计提供反馈，希望获得关于 CUDA 错误处理和内存清理的建议。
   - 他们收到了关于在 CUDA 编程中使用 C 与 C++ 的指导，并被引导至一个 [GitHub 仓库](https://github.com/nvidia/cccl) 以进行深入阅读。
- **澄清 CUDA 中 C 与 C++ 的使用**：讨论了 CUDA 编码中对 C 或 C++ 的偏好，强调 CUDA 主要使用 nvcc 作为 C++ 编译器进行编译。
   - 有人指出，虽然可以用纯 C 编写 CUDA，但许多现代库大量利用了 C++ 特性。
- **探索 PPC 课程资料**：一位用户对发现 PPC 课程感到兴奋，并期待探索其内容。
   - 他们的参与表明了进一步理解 CUDA 和并行编程的愿望。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gist.github.com/saptarshichaudhuri/4c3c63448279c8b87ba2fe5ce83d8de9">Sample matrix multiplication - CUDA</a>：CUDA 矩阵乘法示例。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://www.youtube.com/watch?v=QmKNE3viwIE">4.5x Faster CUDA C with just Two Variable Changes || Episode 3: Memory Coalescing</a>：通过仅更改两个变量使 CUDA C 快 4.5 倍 || 第 3 集：内存合并。CUDA C 中高效全局内存传输的内存合并。视频笔记：https://0mean1sigma.com/chapter-4-memory-coalescing-and-tiled-matrix-multiplic...</li><li><a href="https://github.com/nvidia/cccl">GitHub - NVIDIA/cccl: CUDA Core Compute Libraries</a>：CUDA 核心计算库。通过在 GitHub 上创建账号为 NVIDIA/cccl 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1339287560803061893)** (2 条消息): 

> `CUDA memory model, Atomic Operations in CUDA, Thread Synchronization, GPU vs CPU Architecture, Understanding Scan/Prefix Sum` 


- **CUDA 内存模型困惑**：一位 CUDA 初学者对 PMPP 书中的一段代码表示担忧，认为其可能违反了关于内存排序的 **C++ 内存模型**，特别是质疑是否需要 acquire 语义。
   - 他们表示不确定缺少 thread fence 是否会导致未定义行为，并寻求 CUDA 文档中关于处理内存模型的澄清。
- **关于原子操作的讨论**：该讨论涉及 **atomicAdd** 与线程同步的关系，思考为什么仅在原子操作后才需要 thread fence。
   - 一位成员提到，在没有适当的 acquire/release 注解的情况下依赖编译器行为存在潜在风险，建议这可能会导致难以发现的 bug。
- **对比 GPU 和 CPU 架构**：成员们注意到了 **GPU 和 CPU 结构** 之间的根本区别，强调 CUDA core 要小得多，并且不具备乱序执行 (out-of-order execution) 等特性。
   - 这使得 CUDA 中的行为更具确定性，因为执行顺序遵循指令序列，而不像 CPU 可能会对操作进行重排序。
- **需要社区见解**：初学者向更有经验的成员寻求见解和澄清，以确定书中的代码片段是疏忽还是在 CUDA 上下文中是合理的。
   - 他们特别提到，即使是有经验的社区成员最初也对该代码行为的合法性感到困惑。



**提到的链接**：<a href="https://stackoverflow.com/questions/79429440/cuda-memory-model-why-acquire-fence-is-not-needed-to-prevent-load-load-reorderi.">CUDA memory model: why acquire fence is not needed to prevent load-load reordering?</a>：CUDA 内存模型：为什么不需要 acquire fence 来防止 load-load 重排序？我正在阅读《Programming Massively Parallel Processors》一书，并注意到了以下实现“多米诺风格”scan 的代码片段：&#xA;if (threadIdx.x == 0) {&#x...

  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1339316722976423978)** (2 messages): 

> `FP8 Dynamic Quantization, INT8 Dynamic Quantization, Issue Resolution` 


- **动态量化选项可用**：经过近期讨论，用户现在可以直接在 torchao 中尝试 **FP8** 或 **INT8 动态量化**。
   - 这是在解决了涉及用户 **<@969697995522191360>** 的问题后实现的，标志着功能运行更加顺畅。
- **先前问题已基本解决**：讨论显示之前的问题已基本解决，从而提升了用户体验。
   - 一位成员重申：“是的，这已经解决了”，确认了积极的结果。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1339007335909167185)** (2 messages): 

> `Quantization-Aware Training, QuEST Method, YouTube Animation Parody` 


- **探索 LLM 的量化感知训练**：近期的讨论强调了 **量化感知训练 (QAT)** 在降低大语言模型成本方面的重要性，强调其在保持准确性的同时进行低位宽训练的能力。
   - 一项研究表明，与 FP16/BF16 相比，**8 位权重和激活**在性能上是最佳的，这为 QuEST 等新方法铺平了道路。
- **介绍用于模型训练的 QuEST 方法**：**QuEST** 方法被提议为一种最先进的方法，在使用 **4 位或更低**位宽训练的模型中实现了更好的准确性，同时在帕累托效率上可与 FP16 竞争。
   - 它利用了 **Bengio 技巧 (STE)** 和 RMS 等技术，结合独特的量化误差分离，以提高训练效率。
- **关于“It's a Good Deal”动画的简短评价**：一段名为 *'it's a good deal'* 的 YouTube 短视频利用创意动画技术恶搞了现有内容，引起了社区的关注。
   - 该视频强调了幽默与视觉艺术的结合，特别是在 **#strangerthings** 和 **#blender3d** 的背景下。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05003">QuEST: Stable Training of LLMs with 1-Bit Weights and Activations</a>: 降低大语言模型 (LLMs) 巨额成本的一种方法是使用量化或稀疏表示进行训练或部署。虽然训练后压缩方法是有效的...</li><li><a href="https://www.youtube.com/shorts/QnxbNd74UCU">it&#39;s a good deal. parody of a parody by Matt Storer #strangerthings #animation #b3d #blender3d</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1339180282212909087)** (1 messages): 

> `Nebius Meetup, Kubernetes operator for Slurm, Test-time computation in agentic systems` 


- **旧金山 Nebius 见面会**：Nebius 将于 **3 月 13 日**在旧金山举办见面会，分享其架构和开发原则的见解，包括 [活动注册详情](https://nebius.com/events/nebius-roadshow-san-francisco)。
   - *与会者将获得免费额度*，以试用由 NVIDIA 提供支持的 Nebius GPU Cloud。
- **深入探讨 Slurm 的 Kubernetes Operator**：会议将详细介绍 Nebius 如何为 **Slurm**（一种可扩展的集群工作负载管理器）开发 **Kubernetes operator**。
   - 这旨在增强资源管理，同时更有效地支持 AI 工作负载。
- **通过测试时计算解锁智能体系统**：见面会将探讨 **测试时计算 (Test-time computation)** 如何为 AI 框架内的 **智能体系统 (agentic systems)** 开启新功能。
   - 预计该环节将揭示该技术的创新应用及其影响。



**Link mentioned**: <a href="https://nebius.com/events/nebius-roadshow-san-francisco">Nebius AI Cloud Unveiled. San Francisco Meetup</a>: 探索在顶尖 NVIDIA® GPU 上构建、微调和运行 AI 模型及应用的最有效方式。

  

---

### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1339069679976583261)** (5 条消息): 

> `T-mac 论文咨询，FMHA BWD Kernel 示例，BWD Kernels 贡献` 


- **关于 T-mac 论文实现的咨询**：一名成员就 **T-mac** 论文的实现问题联系了 @lei，并表示有兴趣通过私信（PM）进行讨论。
   - 这突显了围绕 *研究协作* 进行的积极交流。
- **分享了 FMHA BWD Kernel 示例**：另一名成员分享了 GitHub 上一个 **FMHA BWD Kernel** 实现示例的链接，强调了其在高性能 Kernel 开发中的作用。
   - 可以通过 [example_mha_bwd.py](https://github.com/tile-ai/tilelang/blob/main/examples/flash_attention/example_mha_bwd.py) 访问。
- **征集 BWD Kernels 的贡献**：一名成员欢迎对更多 **BWD kernels** 的贡献，提议建立一个开放协作的开发环境。
   - 这一邀请反映了社区对增强 Kernel 优化可用资源的兴趣。



**提到的链接**：<a href="https://github.com/tile-ai/tilelang/blob/main/examples/flash_attention/example_mha_bwd.py">tilelang/examples/flash_attention/example_mha_bwd.py at main · tile-ai/tilelang</a>：旨在简化高性能 GPU/CPU/加速器 Kernel 开发的领域特定语言（DSL） - tile-ai/tilelang

  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1339320713240252539)** (2 条消息): 

> `FSDP, Liger Kernel` 


- **在 Liger Kernel 中实现 FSDP 的困扰**：一名成员表示在 **Liger Kernel** 中使用 **FSDP** 遇到困难，称已经尝试了数小时但未获成功。
   - *有人知道如何在 Liger Kernel 中使用 FSDP 吗？*
- **寻求关于 FSDP 的帮助**：另一名成员承认了 **FSDP** 面临的挑战，并对挣扎中的用户表示同情。
   - 他们建议了潜在的搜索策略，以寻找相关资源或社区见解。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1339124822210514964)** (3 条消息): 

> `Tilelang v0.1.0 发布，社区参与，高性能 AI Kernels` 


- **用于高性能 AI Kernels 的 Tilelang v0.1.0 发布**：社区庆祝了 [tilelang v0.1.0](https://github.com/tile-ai/tilelang) 的发布，这是一种专为高性能 AI kernels 设计的新型 Python 风格 DSL，具有专门的内存分配以及可选的布局（layout）和流水线（pipeline）注解功能。
   - 亮点功能包括**细粒度的线程级控制**，使其成为专注于效率的开发者的有力工具。
- **提议进行社区演讲**：一名成员表示有兴趣邀请创作者进行关于 tilelang 的演讲，认为这将对社区有益。
   - 开发者表示同意，并称一旦相关的预印本（preprint）准备就绪，他们很乐意分享更多内容，并表示目前工作仍在进行中。



**提到的链接**：<a href="https://github.com/tile-ai/tilelang">GitHub - tile-ai/tilelang: Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels</a>：旨在简化高性能 GPU/CPU/加速器 Kernel 开发的领域特定语言 - tile-ai/tilelang

  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1339347516361146368)** (42 条消息🔥): 

> `用于 GPU Kernel 生成的 DeepSeek-R1 模型、Project Popcorn 协作、KernelBench 基准测试讨论、生成的 Kernel 性能` 


- **DeepSeek-R1 简化了 GPU Kernel 生成**：NVIDIA 的实验展示了 DeepSeek-R1 模型，该模型利用 [inference-time scaling](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/) 创建了针对性能优化的 GPU attention kernel。将测试时计算（test time compute）扩展到 **10 分钟**以上显著提升了结果。
   - 一位成员推测，验证器可能是 **ncu** 结合 PyTorch baseline 的正确性检查。
- **Project Popcorn 协作工作**：目前正在计划发布项目的“任务（tasks）”，以便公众更轻松地参与 Project Popcorn 的协作，但尚未完全开放。成员们表示一旦这些任务发布，就有兴趣参与贡献。
   - 目前有与 Stanford 相关的持续开发工作，并且正在 [Discord](https://discord.gg/MAnFAGRn) 上构建相关的协作基础设施。
- **KernelBench 性能见解**：讨论集中在这样一个事实上：该工作流在 Stanford 的 KernelBench 基准测试中，为 **100% 的 Level-1** 和 **96% 的 Level-2** 问题生成了正确的 kernel。由于缺乏更高级别的性能报告，引发了关于基准测试可能已饱和的疑问。
   - 有建议称可能需要一个新的、更具挑战性的基准测试，因为现有的基准测试似乎已经饱和。
- **GPU Kernel 的重要性与利基市场**：对话强调，虽然 **GPU 编程**被视为软件工程的一个利基细分领域，但对于依赖 GPU kernel 的公司来说，它在节省资源和成本方面具有巨大价值。辩论集中在通用软件工程与专业 GPU kernel 工程的相对价值上。
   - 一些成员承认 GPU kernel 在许多深度学习应用中的核心作用，并将其类比于 DeepMind 如何改进矩阵乘法算法。



**提到的链接**：<a href="https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/?ncid=so-link-284103&linkId=100000338909940">使用 DeepSeek&#x2d;R1 和 Inference Time Scaling 自动化 GPU Kernel 生成 | NVIDIA 技术博客</a>：随着 AI 模型扩展其解决更复杂挑战的能力，一种被称为 test&#x2d;time scaling 或 inference&#x2d;time scaling 的新缩放定律正在兴起。也被称为 AI 推理...

  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1339010857530822717)** (2 条消息): 

> `复数 MatMul 性能、将 ST 重新解释为 CST` 


- **复数 MatMul 性能难题**：一位成员正尝试创建复数 **matmul**，但难以达到与 [基准测试 kernel](https://link.to.benchmark) 类似的性能。他们正在寻求一种能与实际示例 kernel 性能相匹配的实现。
- **将 ST 重新解释为 CST 的问题**：同一位成员表达了对将 **st** 重新解释为 **cst** 的沮丧，特别提到了使用 `subtile_inplace` 的尝试，但该方法无法很好地与 **mma** 集成。
   - 他们正在寻求指导或替代方法来解决此问题。


  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1338968411102773309)** (49 条消息🔥): 

> `Prompt 优化、数据集评估、模型格式灵活性、CodeSteer 发布、数学验证` 


- **简化 Prompt 指令以提高清晰度**：团队讨论了直接在问题模板中添加清晰的格式指令，以增强模型的理解和一致性，例如使用 `<answer>` 标签来标记最终答案。
   - 这种方法旨在避免模型响应中的歧义，并确保所有响应都符合预期格式，从而使 LLM 和人工评估员都能受益。
- **关于模型解析和格式化的考量**：关于是允许更灵活的答案格式还是严格惩罚错误格式存在争论，建议倾向于结合改进的解析和更清晰的指令。
   - 这反映了在模型训练效果与输出格式所需的精确度之间取得平衡的愿望，以确保不同数据集的可用性。
- **CodeSteer 的发布与使用**：Andreas 提到了 [CodeSteer](https://github.com/yongchao98/CodeSteer-v1.0) 的发布，强调了其开源性质以及研究使用的引用要求。
   - 这为从事代码生成相关研发的社区成员提供了一项重要贡献。
- **命题逻辑数据集的问题**：有人担心命题逻辑数据集存在损坏，引发了关于是删除它还是修复其构建方式的讨论。
   - 对谜题构建中特定错误的澄清表明，需要对数据集进行彻底审计以确保准确性。
- **通过迭代改进增强数据集评估**：强调了在各种数据集上采用迭代评估方法，以确保每个模型都表现良好，即使这需要调整 Prompt 或数据集结构。
   - 成员们表示，在开发有效解决方案的过程中，人工监督在微调 Prompt 和评估策略方面持续发挥着重要作用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/Adefioye/AI-Playground/blob/main/eval/Prompt.md">AI-Playground/eval/Prompt.md at main · Adefioye/AI-Playground</a>: 通过在 GitHub 上创建账号为 Adefioye/AI-Playground 的开发做出贡献。</li><li><a href="https://github.com/yongchao98/CodeSteer-v1.0">GitHub - yongchao98/CodeSteer-v1.0: Code and dataset of CodeSteer</a>: CodeSteer 的代码和数据集。通过在 GitHub 上创建账号为 yongchao98/CodeSteer-v1.0 的开发做出贡献。</li><li><a href="https://github.com/huggingface/Math-Verify">GitHub - huggingface/Math-Verify</a>: 通过在 GitHub 上创建账号为 huggingface/Math-Verify 的开发做出贡献。</li><li><a href="https://github.com/agentica-project/deepscaler">GitHub - agentica-project/deepscaler: Democratizing Reinforcement Learning for LLMs</a>: 为 LLM 普及强化学习。通过在 GitHub 上创建账号为 agentica-project/deepscaler 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1339272257029279765)** (2 条消息): 

> `OpenAI o1 和 o3 可用性，Groq 驱动的 Llamas 介绍，Nitro 功能升级，Groq DeepSeek R1 70B 性能` 


- **OpenAI o1 和 o3 现已面向所有人开放**：OpenAI 宣布 o1 和 o3 推理模型系列已对所有 OpenRouter 用户开放，不再需要 BYOK，并为之前使用自己密钥的用户提供更高的速率限制（rate limits）。更多详情见[此处](https://x.com/OpenRouterAI/status/1889708759355691327)。
   - 这些模型还具备网页搜索功能，显著提升了实用性。
- **Groq 驱动的 Llamas 提供极致速度**：随着 Groq 的正式支持，用户现在可以体验由 Groq 驱动的极速 Llama 端点，Llama 3.3 的速度超过 **250 tokens per second**，Llama 3.1 达到 **600 TPS**。可用模型的详情请见[此链接](https://openrouter.ai/provider/groq)。
   - 用户可以选择自带密钥（BYOK）以提升速率限制。
- **改进后的 Nitro 功能提升吞吐量**：`:nitro` 后缀现已针对所有模型进行了增强，使用户能够按延迟和吞吐量对端点进行排序，而不是将其显示为单独的端点。这种强大的配置可以通过 API 或直接在聊天室中实现。
   - 还引入了增强型图表来跟踪供应商随时间变化的性能，使对比更加容易。
- **Groq DeepSeek R1 70B 刷新速度记录**：新加入的 Groq DeepSeek R1 70B 达到了惊人的约 **1000 tokens per second**，树立了新的速度基准。更多信息见[此处](https://x.com/OpenRouterAI/status/1889726731571044538)。
   - 此次更新包括对众多参数的支持，并提供自带密钥选项以获得额外的速率限制提升。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1889726731571044538">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 很高兴宣布 @GroqInc 正式上线 OpenRouter！⚡️- 包括创纪录的 1000 TPS 蒸馏版 DeepSeek R1 70B - 支持大量参数 - 如果需要可以自带密钥，获得速率限制提升...</li><li><a href="https://x.com/OpenRouterAI/status/1889708759355691327">来自 OpenRouter (@OpenRouterAI) 的推文</a>: OpenAI o1 和 o3 现已面向所有 OpenRouter 用户开放！不再需要 BYOK。如果你之前已经在使用自己的密钥，现在你拥有更高的速率限制。它们还支持网页搜索 👇</li><li><a href="https://openrouter.ai/openai/o3-mini-high.">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/provider/groq">Groq | OpenRouter</a>: 浏览 Groq 提供的模型
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1338967086470402129)** (163 条消息🔥🔥): 

> `聊天记录问题、提供商路由与性能、模型性能对比、OpenRouter 更新、额度与订阅关注点` 


- **更新期间聊天记录消失**：多位用户报告在最近的更新后出现了聊天记录消失的问题，并强调聊天记录仅存储在本地，而这一点在之前并未明确说明。
   - 成员们讨论了在清理浏览器历史记录时缺乏关于数据丢失的显著警告，建议在应用程序中提供更清晰的提示。
- **提供商路由对性能的影响**：一名用户因持续收到空响应而将某个提供商列入黑名单，这表明路由系统在倾向于低质量提供商方面存在问题。
   - 另一位成员建议查阅文档以禁用 fallback 设置，从而更好地控制提供商的选择。
- **对模型性能的担忧**：讨论中提到了托管模型可靠性参差不齐的问题，部分用户在使用某些模型时遇到了性能不佳和空响应的情况。
   - 用户指出，虽然像 Mixtral 这样的模型表现良好，但包括 Llama3 在内的其他模型被认为可靠性较低。
- **OpenRouter 功能更新**：OpenRouter 的更新包括引入价格后缀以便更好地模拟请求成本，以及关于请求如何在提供商之间路由的讨论。
   - 社区讨论了默认行为如何确保在最佳可用提供商之间进行负载均衡，以实现性能最大化。
- **订阅与额度的实用性**：用户对在 OpenAI 上的支出表示沮丧，同时希望从订阅中获得更多价值，特别是在模型访问和性能方面。
   - 用户对额度可能过期以及如何跨不同场景使用之前支付的额度提出了担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/use-cases/reasoning-tokens">Reasoning Tokens - 提升 AI 模型决策能力</a>：了解如何使用 reasoning tokens 增强 AI 模型输出。实现逐步推理追踪，以获得更好的决策和透明度。</li><li><a href="https://community.openai.com/t/are-openai-credits-expiring/511215">OpenAI 额度会过期吗？</a>：自仪表板变更以来，我没看到关于额度过期日期的警告。是他们忘了放，还是放在了别处，或者额度不再过期了？</li><li><a href="https://www.reddit.com/r/openrouter/comments/1inpby4/structured_output_with_deepseekr1_how_to_account/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/features/provider-routing">提供商路由 - 智能多提供商请求管理</a>：智能地在多个提供商之间路由 AI 模型请求。了解如何通过 OpenRouter 的提供商路由优化成本、性能和可靠性。</li><li><a href="https://tenor.com/view/bau-bau-merrow-virtualmerrow-fuwamoco-fuwawa-gif-10720476399213933291">Bau Bau Merrow GIF - Bau bau Merrow Virtualmerrow - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://openrouter.ai/docs/features/provider-routing#floor-price-shortcut">提供商路由 - 智能多提供商请求管理</a>：智能地在多个提供商之间路由 AI 模型请求。了解如何通过 OpenRouter 的提供商路由优化成本、性能和可靠性。</li><li><a href="https://x.com/sama/status/1889755723078443244">Sam Altman (@sama) 的推文</a>：GPT-4.5 和 GPT-5 的 OPENAI 路线图更新：我们希望在分享预期路线图方面做得更好，并在简化产品供应方面做得更好。我们希望 AI 能为你“顺畅运行”；我们重新...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1338992273169580042)** (149 条消息🔥🔥): 

> `Deep Hermes Model Release, Speculative Decoding in LM Studio, Calibration Dataset Characteristics, Quantization Strategy for LLMs, Hugging Face Agent Certification Course` 


- **Deep Hermes 模型发布预期**：成员们正热切期待 Deep-Hermes-8B 模型权重的发布，讨论围绕基准测试以及该模型在 NousResearch HuggingFace 仓库上架的公告展开。
   - Teknium 提到准备工作正在进行中，包括基准测试和 Model card，并暗示将利用该模型来撰写关于发布的帖子。
- **LM Studio 引入 Speculative Decoding**：最新的 LM Studio 0.3.10 Beta 版本引入了 **Speculative Decoding**，旨在通过同时使用主模型和 Draft model 来加速推理，这可以显著提升性能。
   - 尽管有潜在好处，一些成员指出结果褒贬不一，认为它最适合大型模型，在某些场景下可能不会提供明显的加速。
- **关于校准数据集的疑问**：有人对所使用的校准数据集的性质感到好奇，提到其内容似乎是随机且无结构的，类似于糟糕的预训练数据。
   - Jsarnecki 解释说，这种奇特的数据集选择是研究的结果，研究表明，即使与 wikitext 等传统数据集相比，近乎随机的数据片段也能产生更好的训练效果。
- **LLMs 量化策略**：讨论包括为 LLaMA 模型创建 F32.imatrix 的策略，jsarnecki 分享了使用 Flash Attention 和卸载 GPU 层以处理有限内存资源的见解。
   - 成员们强调了检查和比较不同量化策略对模型准确性和效率重要性的看法。
- **Hugging Face Agent 认证课程**：成员们分享了正在进行的 Hugging Face Agent 认证课程的信息，强调了其相关性以及涵盖的技能，例如使用 `smolagents` 库创建个人 Agent。
   - 该课程包括基准评估机会，吸引了许多对基础知识充满信心的参与者。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/sama/status/1889755723078443244">Tweet from Sam Altman (@sama)</a>: OPENAI ROADMAP UPDATE FOR GPT-4.5 and GPT-5:We want to do a better job of sharing our intended roadmap, and a much better job simplifying our product offerings.We want AI to “just work” for you; we re...</li><li><a href="https://tenor.com/view/apparently-its-a-big-deal-big-deal-big-deal-apparently-it-is-a-big-deal-gif-26730751">Apparently Its A Big Deal Big GIF - Apparently Its A Big Deal Big Deal - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=nt7ckZDTtis">An unfiltered conversation with Chamath Palihapitiya</a>: Join Nolan Fortman and Logan Kilpatrick as they dive into an unfiltered conversation with Chamath Palihapitiya, covering:- open source AI- the massive comput...</li><li><a href="https://lmstudio.ai/docs/advanced/speculative-decoding)">LM Studio Docs | LM Studio Docs</a>: Learn how to run Llama, DeepSeek, Phi, and other LLMs locally with LM Studio.</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio - Beta Releases</a>: Beta and Release Candidate versions of LM Studio</li><li><a href="https://www.cnbc.com/2025/02/11/ken-griffin-says-trumps-bombastic-trade-rhetoric-is-a-mistake-thats-eroding-trust-in-the-us.html">Ken Griffin says Trump&#x27;s &#x27;bombastic&#x27; trade rhetoric is a mistake that&#x27;s eroding trust in the U.S.</a>: The billionaire hedge fund founder&#x27;s comments came after Trump on Monday evening signed an order that would impose 25% tariffs on steel and aluminum imports.</li><li><a href="https://huggingface.co/Joseph717171/Hermes-3-Llama-3.1-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF/tree/main">Joseph717171/Hermes-3-Llama-3.1-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF at main</a>: no description found
</li>
</ul>

</div>

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1339187638896164956)** (2 条消息): 

> `LoRA Forget Less and Learn Less, Automated Capability Discovery, Diversity in Technology, Self-Exploration in AI Models` 


- **来自 LoRA Forget Less and Learn Less 的见解**：名为 *LoRA Forget Less and Learn Less* 的论文强调了确保技术行业**多样性 (diversity)** 的重要性，特别是在巴西葡萄牙语语境下。
   - 讨论了在结合语言和技术时 **domain shift** 带来的挑战，强调模型需要足够的容量以避免**灾难性遗忘 (catastrophic forgetting)**。
- **引入 Automated Capability Discovery**：新的研究探讨了前沿模型是否可以通过名为 Automated Capability Discovery (ACD) 的过程进行 **self-exploration**，以识别其自身的能力和失败模式。
   - 由 [@cong_ml](https://x.com/cong_ml) 和 [@shengranhu](https://x.com/shengranhu) 领导，ACD 允许 foundation model 扮演科学家的角色，提出开放式任务来系统地探测其能力，并在 GPT 和 Claude 等多个模型上展示了结果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/jeffclune/status/1889568685632667672">Jeff Clune (@jeffclune) 的推文</a>: 介绍 Automated Capability Discovery！ACD 通过 "self-exploration" 自动识别 foundation models 中令人惊讶的新能力和失败模式，模型会探索自身的...</li><li><a href="https://arxiv.org/abs/2502.07577">Automated Capability Discovery via Model Self-Exploration</a>: Foundation models 已成为通用助手，通过在大规模网络数据上的训练，在众多领域展现出多样化的能力。精确刻画其能力仍然具有挑战性...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1339100746259632311)** (3 条消息): 

> `AI value systems, Superagent Hackathon, cRACK'd Den event, AI safety declaration summit` 


- **AI 发展出连贯的价值体系**：一位成员指出，随着 AI 变得更加先进，它们建立了内部价值层级，表现出对生命的偏好顺序为：**巴基斯坦 > 印度 > 中国 > 美国**。
   - 这引发了关于 **AI alignment** 以及 AI 驱动的价值体系所带来影响的重大问题。
- **加入 Superagent Hackathon！**：为期一天的黑客松邀请开发者创建下一代 **SUPERAGENTS**，利用 Story 的 **Agent Transaction Control Protocol** 进行跨框架和跨链的集成。
   - 参与者可以开发新项目或改进现有项目，并有机会赢取奖品并进行协作。
- **在 cRACK'd Den 活动中放松**：**cRACK'd Den** 活动是为在 crypto 和 AI 领域构建的开发者举办的聚会，旨在庆祝黑客松后的创意与创新。
   - 与会者可以享受表演、美食，并在与志同道合的人交流的同时见证黑客松获胜者的公布。
- **美英拒绝签署 AI 安全宣言**：在一次国际峰会上，以 Vance 为代表的美国拒绝签署安全宣言，担心与中国等**威权政权**合作可能会损害国家安全。
   - 对**多边主义 (multilateralism)** 措辞和国际协作的担忧导致未能达成一致，特别是在涉及美国在 AI 领域的领导地位方面。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arstechnica.com/ai/2025/02/us-and-uk-refuse-to-sign-ai-safety-declaration-at-summit/">美英在峰会上拒绝签署 AI 安全宣言</a>: 美国的立场与拜登政府相比发生了“180 度大转弯”。</li><li><a href="https://x.com/DanHendrycks/status/1889344074098057439?t=IXS9ty0t1fVgxJ4W90enDw&s=33">Dan Hendrycks (@DanHendrycks) 的推文</a>: 我们发现随着 AI 变得更聪明，它们会发展出自己连贯的价值体系。例如，它们对生命的重视程度为巴基斯坦 > 印度 > 中国 > 美国。这些不仅仅是随机偏见，而是内部一致的...</li><li><a href="https://lu.ma/superagenthackathon">Super Agent Hackathon · Luma</a>: 来构建只能在 Story 上存在的下一代 SUPERAGENTS。STORY AGENT LAB 正在与顶尖的 agentic 框架一起打造下一个 SUPER AGENT...</li><li><a href="https://lu.ma/crackdden">cRACK&#x27;d DEn · Luma</a>: 🚨 CRACK&#x27;D 开发者、AI 爱好者、ETHDENVER 参与者。🚨 心理战让你沮丧？构建到麻木？Cursor 无法编译？你的代币无法绑定？需要 Agent…
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1339187638896164956)** (2 messages): 

> `LoRA Forget Less and Learn Less, Diversity in Technology, Automated Capability Discovery, Foundation Models Self-Exploration` 


- **关于巴西技术的 LoRA 论文见解**：阅读论文 “LoRA Forget Less and Learn Less” 揭示了技术领域多样性的重要性，特别是在巴西葡萄牙语环境下。
   - 作者指出，语言与技术的结合带来了挑战，主张要么接受领域偏移（domain shifts），要么使用能够处理特定语言要求的模型。
- **引入 Automated Capability Discovery**：[@jeffclune 的 Twitter 线程](https://x.com/jeffclune/status/1889568685632667672) 重点介绍了一个名为 **Automated Capability Discovery (ACD)** 的新框架，该框架通过自我探索（self-exploration）来识别 Foundation Models 中的能力和失败模式。
   - 该框架能够系统地提出开放式任务来探测模型的能力，有可能发现令人惊讶的能力和缺陷。
- **ACD 在多个模型上得到验证**：**Automated Capability Discovery** 框架已在包括 **GPT**、**Claude** 和 **Llama** 在内的多个模型上进行了演示，展示了其在评估 Foundation Models 方面的实用性。
   - 初步结果表明，ACD 通过系统化的方法，有助于更好地理解模型的能力和潜在风险。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/jeffclune/status/1889568685632667672">Jeff Clune (@jeffclune) 的推文</a>：介绍 Automated Capability Discovery！ACD 通过“自我探索”（模型探索自身能力...）自动识别 Foundation Models 中令人惊讶的新能力和失败模式。</li><li><a href="https://arxiv.org/abs/2502.07577">Automated Capability Discovery via Model Self-Exploration</a>：Foundation Models 已成为通用助手，通过在网络规模数据上的训练，在众多领域展现出多样化的能力。准确表征其能力仍然具有挑战性...
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1339079084633882675)** (1 messages): 

> `Google Sheets Support, NotebookLM Feedback Survey` 


- **用户希望支持 Google Sheets**：发布了一则关于用户对 NotebookLM 支持 **Google Sheets** 及更广泛的**电子表格支持**兴趣日益增长的通知，并发布了[反馈调查](https://forms.gle/G78qnNCv2UwcYXc16)以收集意见。
   - 该请求要求提供所需表格的详细信息，包括其维度和所包含的数据类型。
- **关于导入 Google Sheets 的反馈**：团队正在收集有关用户希望将 **Google Sheets** 中的哪些内容**摄入（ingest）**到 NotebookLM 的具体规格，重点关注标签页数量和行数等特征。
   - 同时也鼓励用户分享他们希望从表格中获得的见解，并邀请用户提供其数据的脱敏副本。



**提到的链接**：<a href="https://forms.gle/G78qnNCv2UwcYXc16">NotebookLM 的 Google Sheets 支持</a>：对于那些要求能够将 Google Sheets 摄入到 NotebookLM 的用户，我们正在征求您的反馈，以更好地了解您的使用场景！

  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1338966305306447985)** (15 条消息🔥): 

> `使用 Notebook LM 进行健康追踪、NotebookLM 作为写作助手、Anki 抽认卡制作、使用 NotebookLM 进行简历筛选、AI-Podcast 创作` 


- **使用 Google LM 变革健康追踪**：一位用户讨论了利用 Notebook LM 监控健康数据，强调了动态刷新 Google Sheets 链接如何能彻底改变追踪流程。他们提到已经将各种数据流集成到了 Looker Studio 中。
   - 他们强调了 Notebook LM 在显著增强健康数据管理方面的潜力，并分享了关于该话题的音频讨论链接。
- **NotebookLM 辅助奇幻小说写作**：一位用户正将 NotebookLM 作为其长期奇幻小说项目的写作助手，专注于世界观构建（world building）和数据组织。他们非常看重音频生成功能，认为其能合成潜在读者可能提出的问题。
   - 该工具帮助他们在庞大的世界观构建工作中识别漏洞和矛盾。
- **使用 NotebookLM 高效制作抽认卡**：一位成员指出，Notebook LM 中的特定 Prompt 可以有效地创建 Anki 抽认卡，从而优化学习过程。另一位用户对此表示出兴趣，并询问了所用 Prompt 的细节。
   - 这突显了 NotebookLM 在创建结构化学习材料方面的实用性。
- **使用 NotebookLM 改进简历筛选**：一位用户正在探索使用 NotebookLM 辅助筛选 100 多份简历，方法是将简历与相关的职位描述（job description）一同加载。这种方法的初步尝试看起来很有前景，因为 AI 帮助他们重新审视了对候选人的评估。
   - 这种方法似乎有助于为他们的筛选流程提供全新的视角。
- **创作 AI-Podcasts 实现变现**：一位用户详细阐述了如何使用 AI 快速创建播客，强调了该领域为创业者提供的巨大市场机会。他们表示，播客可以提升内容消耗率和市场覆盖范围。
   - 他们指出，将静态内容转化为引人入胜的音频具有新颖性，可以在无需公开演讲的情况下实现影响力的最大化。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://millionai.substack.com/p/create-ai-podcasts-in-seconds-without?r=297y6u&utm_medium=ios&triedRedirect=true.">🎙️在几秒钟内创建播客（无需开口）🤐</a>: 我如何通过两人 AI-podcast 每月额外赚取 7850 美元 🎧 (no-code)</li><li><a href="https://notefeedlm.com/">NotefeedLM</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1338976871148163164)** (90 条消息🔥🔥): 

> `用户限制与共享笔记本、学生使用 NotebookLM 的经验、音频功能与个性化、产品限制反馈、源文件处理问题` 


- **用户限制说明**：用户对查询限制表示困惑，特别是共享笔记本是否会影响自己的每日额度；已确认限制是按用户计算的，而不是按共享的笔记本计算。
   - 一位成员指出，NotebookLM 的限制（免费版 50 次查询，Plus 版 500 次）让依赖交互式提问的学生难以使用。
- **学生分享 NotebookLM 使用心得**：本科生用户报告称，他们使用 NotebookLM 创建模拟测试并总结源材料，称赞其在学习方面的有效性。
   - 音频对话等功能被提及为多任务处理的有用工具，尽管一些人表示某些功能无法正常运行。
- **对个性化音频功能的兴趣**：用户询问了在音频功能中使用自己声音的可能性；目前该功能尚不可用，但提到了一个语音交互的 Beta 计划。
   - 一位用户提到手动将笔记转移到其他应用程序进行朗读，突显了对更集成音频功能的需求。
- **关于限制影响用户体验的反馈**：用户对每日 50 次查询的限制表示担忧，一些用户认为这限制了深入学习和与应用交互的潜力。
   - 一名学生建议，虽然 50 个源文件是可以接受的，但查询限制削弱了该应用在研究和学习方面的有效性。
- **源文件格式问题**：一些用户报告在点击查看源文件时遇到了显示问题；某些 PDF 中混乱的格式使得验证内容变得困难。
   - 产品团队已意识到这些格式问题，并正在努力改进以准确显示源材料。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219">Upgrading to NotebookLM Plus - NotebookLM Help</a>: 未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: 未找到描述</li><li><a href="https://blog.google/feed/notebooklm-google-one/">NotebookLM Plus is now available in the Google One AI Premium subscription.</a>: NotebookLM 是一款研究和思考伴侣，旨在帮助你充分利用信息。你可以上传材料、进行总结、提问并转化……
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1338980796119384155)** (56 messages🔥🔥): 

> `OpenRouter 访问 o1 和 o3，Aider 多会话支持，用于代码编辑的 Editor 模型，GPT-5 路线图更新，用户对 o3-mini 的使用体验` 


- **OpenRouter 开放 o1 和 o3 的访问权限**：OpenRouter 宣布 **OpenAI o1 和 o3 现已向所有用户开放**，不再需要 BYOK，并提高了速率限制。
   - 这一公告受到了热烈欢迎，强调了其改进的功能，且支持网页搜索。
- **Aider 中的聊天机器人多会话管理**：用户表示有兴趣让 Aider 管理多个 tmux 会话，以便更好地控制服务器生成（server spawning）等过程。
   - 作为权宜之计，建议包括使用带有 SSH 连接的本地设置，以简化编码工作流程。
- **关于 “Editor” 模型的提案**：讨论了训练一个 **1.5b “editor” 模型**来与 architect 模型协作，专注于高效的代码编辑。
   - 参与者认为，这样的模型可以解决幻觉（hallucinations）问题，并提高大上下文下的代码 diff 准确度。
- **GPT-5 路线图见解分享**：一项重大更新揭示了 **GPT-4.5** 和 **GPT-5** 的计划，旨在统一模型产品并简化用户体验。
   - 路线图指出，GPT-5 将整合各种技术，并提供给具有不同智能水平的免费层级用户。
- **用户分享 o3-mini 的使用体验**：反馈表明 **o3-mini** 表现良好，在编码任务中速度更快，尤其是与其他模型相比。
   - 一些用户注意到 o3 的部署速度有所提高，而另一些用户则建议将其与 Sonnet 等模型结合使用以获得最佳性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/sama/status/1889755723078443244">来自 Sam Altman (@sama) 的推文</a>: OPENAI 关于 GPT-4.5 和 GPT-5 的路线图更新：我们希望更好地分享我们的预期路线图，并更好地简化我们的产品。我们希望 AI 能为你“直接工作”；我们重新...</li><li><a href="https://x.com/OpenRouterAI/status/1889708759355691327">来自 OpenRouter (@OpenRouterAI) 的推文</a>: OpenAI o1 和 o3 现已向所有 OpenRouter 用户开放！不再需要 BYOK。如果你已经有自己的密钥在工作，你现在拥有更高的速率限制。它们也支持网页搜索 👇
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1338976932339122197)** (24 条消息🔥): 

> `在测试中实现 AI Agent、Aider 多步执行、管理 Aider 配置、在 Aider 中使用 OpenRouter、追踪 Aider 配置来源` 


- **集成 AI Agent 用于测试自动化**：一位用户讨论了如何创建一个 AI Agent，根据 `features.json` 中提供的功能和 `inputs.json` 中的可编辑文件，依据 `results.json` 的测试结果来更新代码。
   - 他们询问此类任务是否在 Aider 的职责范围内，并寻求关于最佳工具或方法的指导。
- **分两步执行 Aider**：一位用户概述了一个分两步运行 Aider 的计划：首先为特定需求生成代码，然后根据该代码创建单元测试。
   - 他们表达了关于如何高效获取第一步的代码更改以便在第二步中使用的担忧。
- **Aider 配置 Drop 的问题**：一位用户提到，在 Aider 中使用 `/drop` 会删除 `read:` 配置中指定的所有文件，他们认为这存在问题。
   - 他们询问是否有办法防止 Aider 丢弃这些专门加载的配置文件。
- **为 OpenRouter 配置模型元数据**：一位用户建议创建一个 `.aider.model.metadata.json` 文件来配置 `openrouter/openai/o3-mini-high`，因为它未列在默认设置中。
   - 他们提供了一个链接，可以从中观察现有的配置标准。
- **识别 Aider 配置来源**：一位用户询问如何确定 Aider 配置值的具体来源，理由是来自 `.env` 文件、环境变量和 YAML 配置等各种可能来源的复杂性。
   - 他们报告遇到了意外的配置值，需要协助追踪这些配置的来源。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ollama.com/library/openthinker/blobs/6490a490932e">openthinker/system</a>：一个完全开源的推理模型系列，使用通过蒸馏 DeepSeek-R1 获得的数据集构建。</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>：你可以通过命令行或 Python 对 Aider 进行脚本化操作。</li><li><a href="https://github.com/Aider-AI/aider/commit/cf0710225c4fac6f07582821634a98447a74814f">告知 o1 &amp; o3-mini 使用 markdown · Aider-AI/aider@cf07102</a>：未找到描述</li><li><a href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json">litellm/model_prices_and_context_window.json at main · BerriAI/litellm</a>：Python SDK，Proxy Server (LLM Gateway)，用于调用 100+ 个 OpenAI 格式的 LLM API - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1339277918026727434)** (9 条消息🔥): 

> `LinkedIn 上的 Aider 提及、CodeSteer-v1 讨论、PyCharm 插件可行性、微软对 JetBrains 的支持、O3 High 和 3.5 Sonnet PR` 


- **Aider 在 LinkedIn 上获得点赞**：[Addy Osmani](https://www.linkedin.com/posts/addyosmani_softwareengineering-programming-ai-activity-7289554729720852480-2oVx?utm_source=social_share_send&utm_medium=android_app&utm_campaign=copy_link) 发布的一篇 LinkedIn 帖子提到了 Aider，引发了用户的好奇。
   - 讨论围绕寻找帖子链接的演示中使用的特定 Prompt 或文件展开。
- **围绕 CodeSteer-v1 的兴奋**：用户分享了一个关于 CodeSteer-v1 的有趣 [Hugging Face 论文](https://huggingface.co/papers/2502.04350#67aaa92ca8192c1ba3c7798f)链接，引发了关注。
   - 然而，另一位用户指出，该论文侧重于代码中的数值计算，而非原生的 LLM 处理。
- **关于 PyCharm 插件开发的咨询**：一位用户表达了对创建类似 PyCharm 插件的兴趣，并询问了其中的难度。
   - 另一位成员确认了该插件的可行性，并指出微软最近对 JetBrains 提供了支持。
- **微软在 JetBrains 支持中的角色**：在回答询问时，一位成员确认了微软最近参与了支持 JetBrains 插件的工作。
   - 这种支持的具体细节尚未详述，留下了进一步讨论的空间。
- **关于 O3 High 和 3.5 Sonnet PR 的询问**：一位用户寻求澄清，是否需要特定的 Pull Request 才能通过 OpenRouter 使用 O3 High 和 3.5 Sonnet。
   - 然而，在现有消息中未提供直接回复。



**提到的链接**：<a href="https://huggingface.co/papers/2502.04350#67aaa92ca8192c1ba3c7798f">论文页面 - CodeSteer: Symbolic-Augmented Language Models via Code/Text Guidance</a>：未找到描述

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1338982822026940478)** (85 条消息🔥🔥): 

> `SDXL 与 SD 1.5 的差异、Flux 模型性能、AI 模型中的蒸馏方法、人类偏好基准、ComfyUI 的 Linux 迁移问题` 


- **SDXL vs SD 1.5 质量见解**：讨论强调，不带 refiner 的 **SDXL** 显示出与 **SD 1.5** 相当的质量，但 **1.5** 保留了 **SDXL** 所缺乏的对词汇的独特解释，这通常是因为后者更侧重于流行审美。
   - 成员们指出，**benchmarks**（基准测试）对于判断质量至关重要，在基准测试语境下 **SDXL 的表现优于 SD 1.5**。
- **Flux 模型一致的输出风格**：成员们观察到 **Flux 模型** 产生的面部特征非常一致，特别是独特的裂纹下巴，这表明其依赖于 **quality-tuned data**（质量微调数据）或蒸馏过程。
   - 一些人将其与 **SDXL** 进行对比，认为 Flux 较低的多样性被其更高的对数似然分布（log likelihood distribution）所掩盖，从而为通过 **loras** 提高多样性留下了空间。
- **蒸馏方法影响模型性能**：讨论阐明了 **Schnell** 是通过 'timestep distilled' 从 **Pro** 衍生而来的，而 **Dev** 使用的是 'guidance distilled'，这影响了模型的性能以及 **loras** 的共享方式。
   - 参与者提到了不同蒸馏方法在 **数据处理** 上的差异及其对模型质量的影响。
- **关于人类偏好基准的辩论**：成员们对 **人类偏好基准** 持复杂态度，认为它们可能偏向于美学上讨喜的输出，而非质量指标。
   - 有人担心这些基准可能会偏好特定的输出（如“美女”），而不是基于详细 prompt 的准确表达。
- **ComfyUI 从 Windows 迁移到 Linux 的挑战**：一位用户报告了将 **ComfyUI 从 Windows** 迁移到 **Linux** 的问题，在按照指南操作后，生成视频时遇到了 **OOM 错误**。
   - 其他成员建议确保安装了正确的 **drivers**（驱动程序），并询问了该用户的 Linux 使用经验，强调糟糕的引导可能会导致不稳定性。



**提及的链接**：<a href="https://huggingface.co/segmind/Segmind-Vega">segmind/Segmind-Vega · Hugging Face</a>：未找到描述

  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1338974813624205366)** (48 条消息🔥): 

> `服务器作者徽章、对 Crypto/NFT 的信任问题、代码审查流程、开源 LLM 模型、使用 Clickhouse 和 Streamlit 的生成式仪表板` 


- **欢迎并授予服务器作者徽章**：一位成员通过授予另一位成员“服务器作者徽章（server author flair）”来表示欢迎，这引发了关于这是否是一件好事的复杂反应。
   - *“我对任何参与过 crypto/nfts 的人都有根本性的不信任”* 表达了对社区可信度的担忧。
- **讨论代码审查方案**：成员们建议建立 *代码审查流程* 来评估众多 MCP 公共服务器的安全性，并提议由多名审查员来分担工作量。
   - *考虑到 900 多个服务器*，一位成员幽默地指出使用语言模型自动进行恶意代码初步过滤的可行性。
- **开源 LLM 模型需要研究**：有人对 *开源 LLM 模型的突破性研究* 的必要性表示担忧，并提到 DeepSeek 借鉴了 OpenAI 的想法。
   - 一位成员指出，虽然 DeepSeek 可能率先分享了创新，但他们仍然利用了 OpenAI 的技术。
- **探索 Clickhouse 和 Streamlit**：一位成员表示有兴趣使用 *Clickhouse 和 Streamlit* 创建一个生成式仪表板服务器，同时探索潜在的变现方案。
   - 他们征求了社区关于 Streamlit 与 PowerBL 等其他工具相比的有效性的意见，并承诺未来会在变现方面进行合作。
- **意外的服务器混淆**：一位成员幽默地承认自己进错了服务器，凸显了大型社区内发生混淆的可能性。
   - 对话记录了各种兴趣和目标，展示了活跃且多样化的社区动态。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 条消息): 

eggsquad: 新的 Modular 职位招聘 👀

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1339195468760223755)** (20 条消息🔥): 

> `stdlib 会议, Mojo 语言特性, sum types 和 parameterized traits, 以 trait 作为元素类型的 List, 已提交 Bug 报告` 


- **关于 stdlib 会议的不确定性**：一名成员询问了在社区会议中提到过的定期 **stdlib 会议**的持续性，并指出无法找到当前的任何信息。
   - 另一名成员确认之前的会议由于时间冲突和组织者的离职而被取消。
- **关于 Mojo 语言特性路线图的咨询**：一位用户询问了即将推出的 **Mojo 语言特性**的时间表，特别是关于 **enums** 和 **parameterized traits** 的支持。
   - 讨论指出 `Variant` 可能会涵盖 enums 的部分功能，但 **parameterized traits** 仍然是团队更高优先级的任务。
- **关于 Sum Types 和新能力的讨论**：成员们讨论了实现 **sum types** 时间表的缺失，解释说虽然这些很有价值，但在基础特性到位之前，它们不会开启太多新能力。
   - 有人指出，目前的重点是开发**底层 (ground level)** 特性，使 Mojo 能够表示类似于 C 的结构。
- **在 Mojo 中使用 Trait 列表的挑战**：一位用户在尝试创建以 trait 作为元素类型的 **List** 时遇到了编译器崩溃，并寻求潜在的变通方法。
   - 另一名成员澄清说，元素必须是具体类型 (concrete types)，并建议如果类型是固定的，可以使用 `Variant`。
- **向 GitHub 提交了 Bug 报告**：一位成员提到在 GitHub 上提交了一份关于 `parallelize` 在与 Python 交互的 **Mojo** 代码中的局限性的 **bug 报告**。
   - 该 bug 描述了在某些条件下，并行化的函数调用会导致运行时崩溃。



**相关链接**: <a href="https://github.com/modular/mojo/issues/3993">[BUG][stdlib] Limitation of `parallelize` with Mojo code that interacts with Python · Issue #3993 · modular/mojo</a>: Bug 描述：实际行为：当并行化的函数调用与 Python 交互时，在某些条件下会在运行时崩溃。参考下面的代码示例，struct Bar.start() 使用了...

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1339082853912281129)** (24 条消息🔥): 

> `MAX -> Wasm 编译, 运行 Mojo 程序, ONNX 模型执行, CUDA 库问题, Mojo API 考量` 


- **MAX 目前不优先考虑 Wasm**：目前，Wasm 后端不是 MAX 的目标重点，也不在短期路线图中。
   - 一位成员对 Wasm 的相关性表示好奇，强调了其未来的潜力。
- **运行 Mojo 程序的多种方式**：MAX 并不是运行 Mojo 程序的唯一方式，正如 [入门教程](https://docs.modular.com/mojo/manual/get-started) 中所述，它可以独立于 MAX 运行。
   - 虽然 MAX 对于利用 GPU 等有趣的应用至关重要，但它对于运行 Mojo 并非严格必要。
- **ONNX 执行需要 MAX**：成员们指出，Modular 对执行 ONNX 模型的支持很大程度上取决于 MAX，强调了其目前的必要性。
   - 这突显了 MAX 在促进跨平台各种 ML 模型执行中的作用。
- **与 CUDA 库相关的段错误 (Seg faults)**：CUDA 库引起的段错误问题引发了关注，成员们建议 MAX 并不完全依赖于 CUDA。
   - 尽管对 CUDA 的使用极少，MAX 仍然依赖 NVIDIA 驱动程序，并且 Mojo 中的特定操作可能会导致问题。
- **围绕 Mojo API 集成的讨论**：一位成员提议将 NVIDIA 库直接集成到 Mojo 中，以简化与 MAX 的配合使用。
   - 意见不一，有人建议完全脱离 CUDA 可以增强稳定性和性能。



**相关链接**: <a href="https://forum.modular.com/">Modular</a>: 与我们一起构建 AI 的未来，了解 MAX, Mojo 和 Magic。

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1339058386389565492)** (37 条消息🔥): 

> `VAE 重参数化技巧, OpenAI IOI 论文, Scaled Cognition APT-1, Glean Agents 发布, OpenAI 路线图更新`

- **VAE Reparameterization Trick 详解**：一场关于为何 VAE 无法通过分布进行反向传播的讨论展开，强调了 Reparameterization Trick 的必要性。
   - 一位成员澄清说，VAE 生成的分布参数需要进行随机采样，而这是一种不可微的操作。
- **OpenAI 在竞赛编程领域的突破**：OpenAI 的一篇新论文详细介绍了他们在 IOI 2024 中的表现，指出 **o3 模型** 在没有手工制定策略的情况下获得了金牌，展示了推理模型（Reasoning Models）的重大进展。
   - 论文评论称模型灵活性是关键，并以 **o1-ioi** 此前需要专门的 Pipeline 为例进行了说明。
- **Scaled Cognition 推出 APT-1**：Scaled Cognition 发布了专为 Agent 应用设计的 **APT-1 模型**，该模型目前在 Agent 基准测试中排名第一。
   - 他们披露了融资细节，包括由 Khosla Ventures 领投的 **2100 万美元** 种子轮融资，并利用了全合成数据 Pipeline。
- **Glean Agents 亮相**：Glean 推出了 **Glean Agents**，这是一个促进可扩展 AI Agent 管理的平台，具有数据集成和治理的新功能。
   - 他们的目标是通过提供对公司和网络数据的用户友好访问来提高生产力。
- **OpenAI GPT 模型路线图更新**：OpenAI 分享了路线图更新，揭示了即将推出的 **GPT-4.5 和 GPT-5**，旨在统一建模方法并简化产品供应。
   - 他们表示将逐渐放弃非推理模型，强调推动集成更广泛功能和推理能力的模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/glennsolomon/status/1889717350456315960?s=46">来自 Glenn Solomon (@glennsolomon) 的推文</a>：荣幸共同领投 @FAL 的 B 轮融资 🚀 AI 驱动的创造力取决于其背后的基础设施。fal 是为 Canva、Perplexity 等提供生成式媒体支持的推理层！很高兴能参与...</li><li><a href="https://x.com/glean/status/1889706504812683728">来自 Glean (@glean) 的推文</a>：欢迎来到 Agent 时代 🚀 我们很高兴地宣布 𝐆𝐥𝐞𝐚𝐧 𝐀𝐠𝐞𝐧𝐭𝐬——我们的横向 Agent 环境，使员工和企业能够大规模地构建、运行、管理和治理 AI Agent...</li><li><a href="https://x.com/scaledcognition/status/1889721166421479751?s=46">来自 Scaled Cognition (@ScaledCognition) 的推文</a>：我们是 Scaled Cognition，正在开发首批专门为 Agent 应用训练的模型：1. 我们的第一个系统 APT-1，目前在 Agent 基准测试中排名第一。2. 它由一支美国团队开发...</li><li><a href="https://arxiv.org/abs/2502.06807">使用大型推理模型进行竞赛编程</a>：我们展示了将强化学习应用于大语言模型 (LLMs) 能显著提升在复杂编程和推理任务中的表现。此外，我们还比较了两种通用推理...</li><li><a href="https://x.com/arankomatsuzaki/status/1889522977185865833?s=46">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：未找到描述</li><li><a href="https://x.com/winstonweinberg/status/1889713028234416371?s=46">来自 Winston Weinberg (@winstonweinberg) 的推文</a>：很高兴宣布由 @sequoia 领投的 D 轮融资，@conviction、@kleinerperkins、@OpenAI、@GVteam、@conviction、@eladgil 和 @LexisNexis 参投。感谢我们的客户、团队、投资人...</li><li><a href="https://x.com/swyx/status/1889810524696891903">来自 swyx 🔜 @aidotEngineer NYC (@swyx) 的推文</a>：转发 @JeffDean：很高兴能与我的好友兼同事 @NoamShazeer 一起，与 @dwarkesh_sp 进行一场超过 2 小时的对话，探讨广泛的...</li><li><a href="https://x.com/iscienceluvr/status/1889517116816244995?s=46">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：使用大型推理模型进行竞赛编程。OpenAI 的新论文重点介绍了其推理模型在 IOI 和 CodeForces 上的表现。“我们使用 o1-ioi 现场参加了 2024 年 IOI，并利用 h...”</li><li><a href="https://x.com/polynoamial/status/1889541408065028421?s=46">来自 Noam Brown (@polynoamial) 的推文</a>：这个梗图很好地总结了这篇论文。引用 Aran Komatsuzaki (@arankomatsuzaki) OpenAI 发布：使用大型推理模型进行竞赛编程 - 在 2024 年 IOI 现场参赛 - o3 获得金牌 - 生成...</li><li><a href="https://x.com/deedydas/status/1889713595384312089?s=46">来自 Deedy (@deedydas) 的推文</a>：重大消息：OpenAI o3 在 2024 年国际信息学奥林匹克竞赛 (IOI) 中获得 600 分中的 394 分，摘得金牌并位列全球第 18 名。该模型未被这些数据污染，且 50 次提交...</li><li><a href="https://share.snipd.com/episode/645ae532-40fd-43ff-9ee4-eb76c8fd56fe">Jeff Dean &amp; Noam Shazeer – 在 Google 的 25 年：从 PageRank 到 AGI</a>：Jeff Dean &amp; Noam Shazeer – 在 Google 的 25 年：从 PageRank 到 AGI</li><li><a href="https://x.com/OpenAI/status/1889781541259321466">来自 OpenAI (@OpenAI) 的推文</a>：今天我们发布了 Model Spec 的重大更新——这是一份定义我们希望模型如何表现的文件。此次更新强化了我们在可定制性、透明度和智力...方面的承诺。</li><li><a href="https://x.com/sama/status/1889755723078443244?s=46&t=JE84TqLviekDnEt8MAT-Eg">来自 Sam Altman (@sama) 的推文</a>：OPENAI 关于 GPT-4.5 和 GPT-5 的路线图更新：我们希望在分享预期路线图方面做得更好，并在简化产品供应方面做得更好。我们希望 AI 能为你“直接可用”；我们重新...
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1339070584717316191)** (15 条消息🔥): 

> `Torchtune 中的 Checkpointing、MLFlow Logger 集成、Torchtune 分布式推理、加密货币投资组合增长` 


- **基于 Step 的 Checkpointing 正在开发中**：一位成员询问是否可以在 **Torchtune** 中每个 epoch 保存多次 checkpoint，另一位成员提到 **Joe** 目前正在 [PR #2384](https://github.com/pytorch/torchtune/pull/2384) 中开发此功能。
   - *这是一个被广泛请求的功能*，预计将显著改进 checkpointing 流程。
- **MLFlow Logger 集成已合并**：**MLFlow logger 集成**已成功合并，据一位成员报告，他在忙碌了一周后非常兴奋地想尽快对其进行测试。
   - 此集成旨在增强 Torchtune 的日志记录能力。
- **使用 Torchtune 进行分布式推理**：一位成员询问如何使用 Torchtune 在多个 GPU 上运行**分布式推理**，另一位成员分享了该任务相关代码的[链接](https://github.com/pytorch/torchtune/blob/main/recipes/dev/generate_v2_distributed.py)。
   - 该成员还指出，将保存的模型加载到 **vLLM** 中进行分布式推理也是可行的，而且速度会*快得多*。
- **加密货币投资组合大获成功**：一位参与者高兴地宣布他们的**加密货币投资组合**目前正在蓬勃发展，突显了最近几天的财务成功。
   - 这一时刻是在关于 Torchtune 社区软件和开发的各种讨论中分享的。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/pull/2384">由 joecummings 实现基于 step 的 checkpointing · Pull Request #2384 · pytorch/torchtune</a>: 上下文此 PR 的目的是什么？是添加新功能、修复错误、更新测试和/或文档还是其他（请在此处添加）。关闭 #2105。这是一个被广泛请求的功能...</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/dev/generate_v2_distributed.py">torchtune/recipes/dev/generate_v2_distributed.py at main · pytorch/torchtune</a>: PyTorch 原生训练后库。通过在 GitHub 上创建账户为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1338965172232130591)** (10 messages🔥): 

> `Testing Assumptions, Gradient Accumulation Issue, Checkpointing Functionality, DPO Loss Calculation, Memory Management with Opt-in BWD` 


- **审视测试假设 (Testing Assumptions Under Scrutiny)**：一位成员对自己的代码质量表示怀疑，认为通过测试可能归功于测试写得不好，而不是实现得好。
   - 这反映了一种常见的心态，即测试成功反而导致对个人编码能力的质疑，而非信心。
- **梯度累积调试的挫败感 (Gradient Accumulation Debugging Frustration)**：关于 [gradient accumulation fix](https://github.com/pytorch/torchtune/issues/2334) 仍存在持续的困惑，这影响了训练效果。
   - 成员们描述了花费数小时进行调试却未找到根本原因的情况，该问题似乎很复杂，可能需要更多的协作努力。
- **Checkpointing 分支工作正常**：一位成员确认了 Checkpointing 分支的成功测试，表示其功能符合预期，并准备好编写更多关于恢复训练的文档。
   - 另一位成员在听到好消息后，对该功能的成功表示了幽默和宽慰。
- **DPO 损失计算的担忧**：讨论围绕确保 DPO 损失不会受到 padding token 的不公平影响展开，这可以追溯到之前的 [issue](https://github.com/pytorch/torchtune/pull/1875)。
   - 对话强调了损失计算准确性的重要性，以及为减轻 padding 导致的损失虚高而进行的潜在调整。
- **Opt-in BWD 的内存管理困境**：一位成员反思了 **opt_in_bwd** 的复杂关系，它在微调期间可以节省内存，但也引入了重大挑战。
   - 他们暗示了通过最小化 Cross-entropy 峰值来进一步减轻内存影响的可能性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/2334">Apply gradient accumulation fix to DPO/PPO recipes · Issue #2334 · pytorch/torchtune</a>: https://unsloth.ai/blog/gradient</li><li><a href="https://unsloth.ai/blog/gradient">Bug Fixes in LLM Training - Gradient Accumulation</a>: Unsloth 的 Gradient Accumulation 修复解决了 LLM 训练中的关键错误。</li><li><a href="https://github.com/pytorch/torchtune/pull/1875">Normalize CE loss by total number of (non-padding) tokens by ebsmothers · Pull Request #1875 · pytorch/torchtune</a>: 为了纪念 ML 社区第一次发现 (x1 / n1) + (x2 / n2) != (x1 + x2) / (n1 + n2) 的那一天。此 PR 更改了启用梯度累积时计算 loss 的方式。T...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1339263623079395411)** (4 messages): 

> `RWKV RNN, Scaling Challenges of RNNs, State Space Models, Importance of Attention Mechanisms` 


- **RWKV RNN 脱颖而出**：一位成员强调了他们对 **RWKV** 的偏爱，认为它是最受欢迎的 RNN，并指出该项目社区提供了大量优秀的 **Apache 2.0** 内容。
   - 他们赞扬了活跃的 Discord 频道，里面充满了讨论先进技术和应用的“强大大脑”。
- **对 RWKV 扩展潜力的怀疑**：一位成员表达了对 **RWKV** 项目的钦佩，但对其能否扩展到与 **Transformer** 相同水平表示怀疑。
   - 这一观点引发了关于 RNN 与 **Transformer** 架构相比在扩展性方面挑战的讨论。
- **提及状态空间模型 (State Space Models)**：另一位成员建议，对 RWKV 的怀疑可能同样适用于状态空间模型，并以 **Mamba** 为例。
   - 这一评论突显了关于各种模型架构可扩展性的持续讨论。
- **Attention 机制依然至关重要**：一位参与者简洁地表示 *Attention is still all we need*，强调了它在现代 AI 模型中的基础作用。
   - 这进一步强化了 Attention 机制在人工智能领域持续的重要性和关注度。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1339021472496222300)** (5 messages): 

> `使用有限数据集训练 ML 模型，访问研究论文，Tinystories 论文，Anna's Archive` 


- **Tinystories 论文解决有限数据集问题**：一名成员建议将 [tinystories 论文](https://link.to.tinystories) 作为使用小型或有限数据集训练 ML 模型的资源。
   - 该论文提出了尽管存在数据集限制但仍能有效学习的策略，这可能对原始咨询有所帮助。
- **寻求获取 ResearchGate 论文**：一名成员寻求获取托管在 ResearchGate 上的论文的帮助。
   - 另一名成员建议直接联系作者以获取 PDF 副本。
- **Anna's Archive：研究论文资源**：有人建议使用 [Anna's Archive](https://annas-archive.org/) 来访问各种论文，它被誉为最大的开放图书馆。
   - 该平台声称镜像了来自 Sci-Hub 和 LibGen 的内容，保存了大量的书籍和论文。



**Link mentioned**: <a href="https://annas-archive.org/">Anna’s Archive</a>: no description found

  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1339033447716491317)** (6 messages): 

> `在潜空间 (Latent Spaces) 上进行推理，多语言语言模型，Llama-2 与语言偏见，Transformers 中的潜语言` 


- **探索潜空间推理**：*Pyro99x* 发起了关于在潜空间进行推理的讨论，强调了 AI 模型运作的一个新视角。
   - 这立即引发了社区参与，一名成员表示愿意尽快参与讨论。
- **分析多语言模型：Llama-2**：今天的焦点是一篇[论文](https://arxiv.org/abs/2402.10588)，该论文质疑多语言语言模型是否使用英语作为内部中枢语言 (pivot language)。
   - 该研究追踪了 Llama-2 中的中间层 Embeddings 如何通过各层进行转换，揭示了语言处理中的偏见。
- **引用的潜语言研究**：一名成员引用了一篇相关论文，声称在平衡语料库上训练的模型可以根据目标在多种语言之间切换。
   - *Bhagawanpanditi* 思考这种现象是否可以扩展到更多语言，并提出了使用主导语言与多种语言之间的权衡。



**Link mentioned**: <a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language mo...

  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1339238950555877377)** (4 messages): 

> `Tokenization 问题，LLMs 中的字符表示` 


- **Tokenization：一个有问题的观点**：一名成员认为计算 **tokens** 可能是真正的问题所在，因为模型在学习数据中并不计算 tokens，而是计算**字符** (characters)。
   - *你这是什么意思？* 另一名成员询问，寻求对这一观点的澄清。
- **理解 Token 表示**：另一名成员解释说 LLMs 在训练数据中只感知 **tokens**，并指出虽然某些 tokens 代表单个字符，但大多数代表多个字符的组合。
   - 他们补充说，当字符被空格隔开时，每个字符会被视为一个独立的 token，这突出了 Tokenization 的一个重要方面。

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1338985003878715453)** (12 messages🔥): 

> `DeepScaleR 模型, 欧盟 AI 资金, Thomson Reuters 版权案, OpenAI 路线图更新, 文献综述工具` 


- **DeepScaleR 超出预期**：一位用户对 [DeepScaleR 预览版](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scali) 表示兴奋，该版本包含一个 1.5B 模型，展示了在扩展 RL 方面的重大进展。
   - *RL 回来了，宝贝！* 是对这一进展的评论。
- **欧盟在 AI 领域投入巨资**：欧盟承诺投资 **2000 亿欧元** 用于 AI，以追赶美国和中国，并强调需要建立用于模型训练的 **AI 超级工厂 (gigafactories)**。
   - 欧盟委员会主席 Ursula von der Leyen 在[巴黎 AI 行动峰会](https://www.msn.com/en-us/money/companies/eu-pledges-200-billion-in-ai-spending-in-bid-to-catch-up-with-u-s-china/ar-AA1yO0Su)上宣布，目标是让欧洲成为领先的 AI 大陆。
- **Thomson Reuters 赢得 AI 版权之战**：[Thomson Reuters 赢得](https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/)了针对 Ross Intelligence 的具有里程碑意义的 AI 版权案，确认了其版权受到侵犯。
   - 法官 Stephanos Bibas 表示，*Ross 的任何可能辩护都站不住脚*，强调了 AI 应用中版权侵权的严重性。
- **OpenAI 揭晓未来模型计划**：在最近的一次更新中，OpenAI 宣布 **GPT-4.5** 将作为其最后一个非 Chain-of-Thought 模型发布，随后将整合 **o-series 和 GPT-series 模型**。
   - OpenAI 的目标是让他们的模型能够 *直接可用 (just work)*，GPT-5 将在各种应用中提供功能，降低用户的复杂性。
- **高效文献综述工具可用**：分享了一个名为 [Deep-Research-Arxiv](https://github.com/GitsSaikat/Deep-Research-Arxiv) 的 GitHub 仓库，旨在进行快速且可靠的文献综述。
   - 用户还可以在 [Hugging Face Spaces](https://huggingface.co/spaces/AlignAI/Deep-Research-Arxiv) 上访问该应用程序，以简化他们的研究流程。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/AlignAI/Deep-Research-Arxiv">Deep Research Arxiv - AlignAI 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/">Thomson Reuters 赢得美国首个重大 AI 版权案</a>: Thomson Reuters 的裁决对生成式 AI 公司与权利持有人之间的斗争具有重大影响。</li><li><a href="https://github.com/GitsSaikat/Deep-Research-Arxiv">GitHub - GitsSaikat/Deep-Research-Arxiv: 快速、简单且可靠地进行文献综述</a>: 快速、简单且可靠地进行文献综述。通过在 GitHub 上创建账户为 GitsSaikat/Deep-Research-Arxiv 的开发做出贡献。</li><li><a href="https://news.slashdot.org/story/25/02/11/1617259/eu-pledges-200-billion-in-ai-spending-in-bid-to-catch-up-with-us-china">欧盟承诺投入 2000 亿美元 AI 支出，力求追赶美中 - Slashdot</a>: 欧盟承诺动员 2000 亿欧元（2061.5 亿美元）投资 AI，因为该联盟寻求在训练最复杂模型的竞赛中追赶美国和中国。</li><li><a href="https://x.com/sama/status/1889755723078443244?t=EgnihPXVoD2fsS9ag5u5SA&s=19">Sam Altman (@sama) 的推文</a>: GPT-4.5 和 GPT-5 的 OPENAI 路线图更新：我们希望在分享预期路线图方面做得更好，并在简化产品供应方面做得更好。我们希望 AI 能为你“直接可用”；我们...
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1339123140017782805)** (20 条消息🔥): 

> `CUDA backend on Windows, PR feedback and testing issues, Windows CI environment variable propagation, Testing failures with recursion vs iteration approach` 


- **Windows CUDA 后端进展**：一名用户确认通过使用正确的 DLL 名称修正 autogen 文件，已成功让 **CUDA 后端在 Windows 上运行**，尽管他们观察到标准的 CI runner 缺乏 GPU 支持。
   - 他们建议可能需要硬编码 CUDA 版本以保持设置简单，但人们对整体测试覆盖率表示担忧。
- **对已关闭 PR 的反馈请求**：一个涉及 bug 修复的 Pull Request 在没有评论的情况下被关闭，引发了对其内容和基本原理的反馈请求。
   - 另一位用户指出缩进错误是 CI 测试失败的主要原因，暗示可能忽略了在 push 之前的测试。
- **CI 环境变量的不一致性**：讨论显示 **Windows CI 没有在步骤之间传播后端环境变量**，导致测试期间默认切换到 CLANG。
   - 已发起一个 Pull Request 以确保环境变量在 CI 步骤之间持久化，从而保证功能正常。
- **对实现方式变更测试的担忧**：有人对从递归切换到迭代的效果表示怀疑，指出这导致了除原始更改之外的许多测试失败。
   - 据观察， CI 失败的直接原因源于一个缩进问题，该问题无意中影响了代码中的关键功能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/9036">Fix TestLinearizerFailures.test_failure_53 by bruari · Pull Request #9036 · tinygrad/tinygrad</a>: 根据悬赏电子表格修复 TestLinearizerFailures.test_failure_53 bug。</li><li><a href="https://github.com/tinygrad/tinygrad/actions/runs/13280542105/job/37077817692?pr=9039#step:5:17">Check the used device on Windows in CI is the matrix backend being te… · tinygrad/tinygrad@a4e6599</a>: 你喜欢 PyTorch？你喜欢 micrograd？你热爱 tinygrad！❤️ - 检查 Windows CI 中使用的设备是否为正在测试的矩阵后端… · tinygrad/tinygrad@a4e6599</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9047">Ensure Windows CI correctly tests the specified backends  by rmtew · Pull Request #9047 · tinygrad/tinygrad</a>: 确保设置的后端环境变量通过 $GITHUB_ENV 持久化到下一步。除非 shell 显式设置为 bash，否则它实际上不会在 Windows 上持久化。添加断言....</li><li><a href="https://github.com/rmtew/tinygrad/blob/feature-windows-cuda/.github/workflows/test.yml#L615">tinygrad/.github/workflows/test.yml at feature-windows-cuda · rmtew/tinygrad</a>: 你喜欢 PyTorch？你喜欢 micrograd？你热爱 tinygrad！❤️ - rmtew/tinygrad
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1339182995327815720)** (5 条消息): 

> `Graph Scheduler Tests, Tinygrad vs PyTorch` 


- **图调度器测试查询**：一名成员询问是哪个测试生成了消息中与 **ASSIGN** 相关的特定图，表示需要澄清。
   - 另一名成员指出，由于 Python 的性能问题，他们回滚了一项更改，并暗示即将进行改进。
- **Tinygrad 与传统框架的辩论**：一名用户质疑从 **PyTorch** 等成熟框架切换到 **tinygrad** 的优势，并引用了自己在后者的个人经验。
   - 另一名成员建议，选择 tinygrad 可能会带来 **更便宜的硬件**、对底层过程更好的理解，以及潜在更快的模型性能。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1339284317175681236)** (2 messages): 

> `开源工程师职位, Agentic Document Workflows, Nomic AI Embedding Model` 


- **开源工程师职位招聘**: [@llama_index](https://twitter.com/llama_index) 宣布了一个全职的**开源工程师**职位，面向对 **Python** 和 **AI** 充满热情的人才。
   - 该角色强调在功能不断增长的背景下扩展 **llama_index** 框架，更多详情请见[此处](https://t.co/WMgdaauxP8)。
- **Nomic AI 增强文档工作流**: 来自 [@nomic_ai](https://twitter.com/nomic_ai) 的最新工作展示了优秀的 **embedding model** 对于高效 **Agentic Document Workflows** 的重要性。
   - 这一新进展获得了积极反响，标志着在增强这些工作流方面迈出了重要一步，更多细节分享在[此处](https://t.co/pezsylHNpH)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1339168356309143582)** (17 messages🔥): 

> `构建 RAG 系统, 批量处理 PDF, 创建查询引擎工具, 详尽的 RAG 搜索方法, 向量数据库` 


- **探索 RAG 系统的 Data Loaders**: 成员们讨论了练习使用不同 Data Loaders 构建 **RAG 系统**的愿望，并建议探索 [llamahub](https://llamahub.example) 网站获取资源。
   - 一位成员强调了选择适合个人用例的 Loaders 的重要性。
- **批量处理 PDF 的咨询**: 一位成员询问了**批量处理 PDF** 的方法，促使另一位成员要求澄清正在考虑的具体方案。
   - 讨论暗示需要更具针对性的工具或脚本来处理批量 PDF 操作。
- **创建带有过滤器的查询引擎工具**: 一位成员寻求关于在查询引擎工具中针对不同主题使用**预定义过滤器**的建议，旨在实现高效工作流而无需创建多个索引。
   - 另一位成员提供了一个代码示例，演示如何实现带有指定过滤器的查询引擎工具。
- **详尽 RAG 搜索的最佳方法**: 一位成员询问了在检索一系列数据时进行**详尽 RAG 搜索**的最佳方法，并认可了现有的 autorag 和 query synthesizing 等方法。
   - 这突显了对探索创新搜索技术以覆盖潜在广泛数据块的兴趣。
- **向量数据库的选择**: 成员们分享了使用不同**向量数据库**的经验，一位成员提到他们使用了 **Milvus**，另一位则提到在 Docker 容器中使用 **Redis**。
   - 这反映了社区对管理向量数据的各种可用工具的兴趣。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1339295436812587038)** (1 messages): 

> `LLM Agents MOOC Hackathon, 全球参与, 获胜团队公布, 顶尖代表国家和大学, Hackathon 网站` 


- **LLM Agents MOOC Hackathon 获胜者揭晓**: 很高兴宣布 **LLM Agents MOOC Hackathon** 的获胜团队，社区参与度惊人，共有来自 [127 个国家](https://x.com/dawnsongtweets/status/1889686697564315963) 的约 3,000 名参与者。
   - Dawn Song 教授在她的 [Twitter 公告](https://x.com/dawnsongtweets/status/1889686697564315963)中对社区的热情表示感谢。
- **Hackathon 的全球代表性**: 本次 Hackathon 吸引了来自 **1,100 多所大学**和 **800 多家公司**的贡献，展示了全球对 AI 的浓厚兴趣。
   - 突出的代表包括**美国**、**印度**和**中国**等顶尖国家，以及加州大学伯克利分校（UC Berkeley）和斯坦福大学（Stanford）等著名学府。
- **Hackathon 中的顶尖公司**: 参与者包括 **Amazon**、**Microsoft**、**Samsung** 和 **Salesforce** 等顶尖公司，反映了行业的参与度。
   - 这种多样化的代表性突显了 Hackathon 在连接学术界和工业界方面的重要性。
- **获胜团队的 Hackathon 网站**: 感兴趣的各方可以在 [Hackathon 网站](https://rdi.berkeley.edu/llm-agents-hackathon/)上查看获胜团队及其提交的作品。
   - 该平台是展示活动期间体现的创新和创造力的中心。



**提到的链接**: <a href="https://x.com/dawnsongtweets/status/1889686697564315963)">来自 Dawn Song (@dawnsongtweets) 的推文</a>: 🎉 很高兴宣布 LLM Agents MOOC Hackathon 的获胜团队！我们对全球 AI 社区惊人的参与度和热情感到兴奋：🌍 来自 127 个国家的约 3,000 名参与者...

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1339296200913850450)** (1 条消息): 

> `Spring 2025 MOOC 发布, 高级 LLM 话题, Dawn Song 公告, 课程参与统计` 


- **Spring 2025 MOOC 正式发布**：**Spring 2025 MOOC** 已正式向 AI 社区公布，重点关注 **Advanced LLM Agents**。
   - 鼓励参与者**转发和分享** Dawn Song 教授在 [Twitter 上的公告](https://x.com/dawnsongtweets/status/1889355520294944829) 以扩大影响力。
- **即将开展的 MOOC 高级话题**：本学期 MOOC 将涵盖高级话题，如 **Reasoning & Planning**（推理与规划）、**Multimodal Agents**（多模态 Agents）以及 **AI for Mathematics**。
   - 其他重点还包括 **Agent Safety & Security**，为 AI 爱好者提供全面的探索。
- **课程参与统计**：基于 **Fall 2024** MOOC 的成功（拥有 **1.5万+ 注册学员**和 YouTube 上 **20万+ 的课程观看量**），预计本课程将吸引更多关注。
   - 之前的课程还聚集了约 **9000 名 Discord 成员**，突显了其强大的社区参与度。
- **每周直播课程安排**：MOOC 将在每周 **PT 时间周一下午 4:10** 举行直播课程，进行互动学习。
   - 这种形式旨在促进学生、研究人员、开发人员和 AI 从业者之间的交流。



**提到的链接**：<a href="https://x.com/dawnsongtweets/status/1889355520294944829)">Dawn Song (@dawnsongtweets) 的推文</a>：非常激动地宣布我们的 Advanced LLM Agents MOOC (Spring 2025)！基于我们 Fall 2024 LLM Agents MOOC 的成功（1.5万+ 注册学员，约 9000 名 Discord 成员，20万+ 课程观看量 ...

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1338997446612881448)** (5 条消息): 

> `MOOC 课程大纲更新, 黑客松参与` 


- **即将发布的 MOOC 课程大纲详情**：关于 **MOOC 课程大纲** 的更多细节将很快发布，预计在 **两周左右** 后公布。
   - *感谢您的耐心等待！*
- **本学期没有黑客松**：一名新学生询问了本学期是否会像上学期一样举办 **Hackathon**（黑客松）。
   - 官方澄清说**目前没有计划举办黑客松**，尽管 Dawn Song 教授过去领导的活动可能预示着未来的潜在机会。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1339127002786959395)** (2 条消息): 

> `MOOC 学生申请, 研究课题` 


- **MOOC 学生申请研究课题？**：一名成员询问 MOOC 学生是否可以申请研究课题，并请求获取 Google 表单的访问权限。
   - 另一名成员回复称**更多细节将很快发布**，并对询问者的耐心表示感谢。
- **获取研究课题信息**：对话显示了学生对如何获取研究课题所需表单的兴趣。
   - 回复表明，相关信息将在近期明确并提供。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1338996056800628858)** (3 条消息): 

> `DeepScaleR 模型, 作业截止日期` 


- **DeepScaleR 超越 O1 Preview**：据文档记录，[DeepScaleR 模型](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) 通过扩展强化学习（RL）技术，使用 **1.5B 模型** 超越了 O1 Preview。
   - 这一进展标志着 AI 模型性能提升的潜力。
- **作业截止日期临近**：针对关于作业截止日期的查询，一名成员确认细节将很快发布。
   - *感谢您的耐心等待！* 同时也提醒那些正在补课的同学。



**提到的链接**：<a href="https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作空间。

  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1338995758346666056)** (6 messages): 

> `Steam Gift Card Giveaway, Voice Model Updates, TextWeb-UI Installation, Session Memory Management, Mobile Application Usability` 


- **Steam 礼品卡赠送警报**：一名成员宣布了 **$50 Steam 礼品卡赠送活动**，并附上了参与链接：[steamcommunity.com/gift-card/pay/50](https://u.to/kR7WIQ)。
   - 随后的聊天活动中，一名成员称其为 **spam（垃圾信息）**，表明反应不一。
- **当前语音模型查询**：一位成员询问了可供使用的**当前语音模型**的可用性，表达了对更新的兴趣。
   - 这引发了关于潜在选项和可用性的进一步讨论。
- **TextWeb-UI 安装挑战**：有人提到 **TextWeb-UI** 需要复杂的安装过程，引发了关于易用性的回应。
   - 一位用户指出，这不是一个简单的 `.exe` 安装，并对其运行要求提出了警告。
- **通过 Python 管理会话记忆**：讨论了为什么后端不能将 **session memories** 从 SQL 馈送到 Python 脚本，强调了其使用潜力。
   - 参与者表示有兴趣探索增强会话管理的不同方式。
- **移动应用性能担忧**：针对 **iOS 和 Android** 移动应用的使用提出了担忧，特别是使用期间的电池寿命。
   - 一名成员推测，使用此类应用程序可能会在 **1 小时内耗尽设备的电池**，表明存在性能问题。


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1339143819651973160)** (2 messages): 

> `Login Issues, API Request Blocking` 


- **登录时出现 Failed to Fetch 错误**：一位用户报告称，在尝试使用凭据登录个人账户时收到“Failed to fetch”错误。
   - 针对此问题的反馈是*信息不足*，从而引发了关于可能存在拦截 API 请求的过滤机制的询问。
- **关于 API 请求过滤的担忧**：一名成员提出疑问，是否某种过滤机制导致了登录尝试期间 API 请求的失败。
   - 这表明可能需要更深入的调查来确定连接问题或软件限制。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1339139551989596160)** (1 messages): 

> `Podcasting Workshop, AI Voice Generation, ElevenLabs and PlayHT` 


- **利用 AI 开启播客成功之路**：参加我们于 **2 月 13 日星期四晚上 9 点（IST）** 举行的免费研讨会，了解创作者如何仅使用 **AI** 而无需昂贵设备来启动播客。
   - 参与者将学习 **AI 音频模型的基础知识**，并亲身体验 [ElevenLabs](https://elevenlabs.io) 和 [PlayHT](https://playht.com) 等平台，创作引人入胜的音频内容。
- **音频创作的实战经验**：参与者将获得领先语音生成平台的**实战经验**，从而能够毫不费力地**将文本转化为音频内容**。
   - 研讨会还将涵盖如何开发自己的**开源 NotebookLM** 以进行自定义实现。
- **利用 AI 资源快速构建**：加入 [Build Fast With AI](https://t.me/BuildFastWithAI) 社区，获取致力于 **Generative AI 解决方案**的免费资源和工具。
   - 该小组由 IIT Delhi 校友运营，提供**最新的 Gen AI 工具**、路线图和研讨会链接，帮助参与者构建创新的 AI 应用程序。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lu.ma/wlnvyebn?tk=eMLWC6">利用 AI 打造你自己的播客工作室 🎙️ · Zoom · Luma</a>：想开始做播客但没有专业的录音设备？好奇 AI 如何成为你的私人配音艺术家？加入我们，参加这场激动人心的……</li><li><a href="https://t.me/BuildFastWithAI">Build Fast With AI - 免费 AI 资源</a>：Build Fast With AI 是一家专注于 Generative AI 的初创公司，由 IIT Delhi 校友运营，旨在提供尖端的 Gen AI 解决方案。		--&gt; 最新 Gen AI 工具	--&gt; Gen AI 路线图与材料	--&gt; 研讨会...
</li>
</ul>

</div>
  

---


---


---


{% else %}


> 完整的逐频道细分内容已在邮件中截断。
> 
> 如果你想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}