---
companies:
- openai
- mistral-ai
- deepseek
- togethercompute
- fireworksai_hq
- ai-gradio
- replicate
date: '2025-02-01T09:16:19.878807Z'
description: '**OpenAI** 发布了 **o3-mini**，这是一款面向免费及付费用户开放的新型推理模型。它具备“高”推理强度选项，在 STEM
  任务和安全基准测试中的表现优于早期的 **o1** 模型，且每 token 的成本降低了 **93%**。**山姆·奥特曼 (Sam Altman)** 承认了开源策略的转变，并称赞
  **DeepSeek R1** 影响了他们的相关认知。**MistralAI** 推出了 **Mistral Small 3 (24B)**，这是一款具有竞争力且
  API 成本较低的开放权重模型。此外，**DeepSeek R1** 已获得 **Text-generation-inference v3.1.0** 的支持，并可通过
  **ai-gradio** 和 **Replicate** 平台使用。这些新闻凸显了 AI 模型在推理能力、成本效益和安全性方面的显著进展。'
id: 5666df5a-ff06-4544-9aa0-3b42b0a11136
models:
- o3-mini
- o1
- gpt-4o
- mistral-small-3-24b
- deepseek-r1
original_slug: ainews-o3-mini-launches-openai-on-wrong-side-of
people:
- sam-altman
title: o3-mini 发布，OpenAI 站在“历史错误的一边”
topics:
- reasoning
- safety
- cost-efficiency
- model-performance
- benchmarking
- api
- open-weight-models
- model-releases
---

<!-- buttondown-editor-mode: plaintext -->**o3-mini 就够了。**

> 2025年1月30日至1月31日的 AI 新闻。我们为您检查了 7 个 subreddit、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **34** 个 Discord（**225** 个频道，**9062** 条消息）。预计节省阅读时间（以 200wpm 计算）：**843 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

正如在 DeepSeek r1 风波之前所计划的那样，[OpenAI 发布了 o3-mini](https://openai.com/index/openai-o3-mini/)，其“高”推理强度（reasoning effort）选项轻松超越了 o1-full（在 [Dan Hendrycks 的新 HLE 等 OOD benchmarks](https://x.com/DanHendrycks/status/1885476082473984475) 和 [Text to SQL](https://x.com/rishdotblog/status/1885420294049030149?s=46) 基准测试中也是如此，尽管 [Cursor 持不同意见](https://x.com/cursor_ai/status/1885415392677675337)）：


![image.png](https://assets.buttondown.email/images/8d36bf13-e514-4159-acf0-376818b42a02.png?w=960&fit=max)


针对 R1 的主要回应包括两个方面：首先是 [o1-mini 和 o3-mini 降价 63%](https://x.com/swyx/status/1885432031896887335)，其次是 [Sam Altman 在今天的 Reddit AMA 中承认](https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_mark_chen_kevin_weil/)，他们将展示“[更具帮助且更详细的版本](https://www.reddit.com/r/OpenAI/comments/1ieonxv/comment/ma9z9yy/)”的思维标记（thinking tokens），并直接归功于 DeepSeek R1 “更新”了他的假设。


![image.png](https://assets.buttondown.email/images/56b24a4b-8533-4bd2-b798-b454d14c2f92.png?w=960&fit=max)


或许更重要的是，Sama 还承认他们在开源策略方面（除了 Whisper 之外并无实质性进展）处于“[历史错误的一边](https://www.reddit.com/r/OpenAI/comments/1ieonxv/comment/maa0dcx/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)”。

您可以在 [今天的 Latent Space 与 OpenAI 的播客](https://www.latent.space/p/karina) 中了解更多信息。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有总结均由 Claude 3.5 Sonnet 完成，从 4 次运行中选取最佳。

**模型发布与性能**

- **OpenAI 的 o3-mini**，一款全新的推理模型，现已在 ChatGPT 中向免费用户开放（通过“Reason”按钮），并向付费用户开放 API，Pro 用户可以无限次访问“o3-mini-high”版本。
  - 该模型被描述为在科学、数学和编程方面表现尤为出色，据称在许多 STEM 评估中超越了早期的 **o1 model** [@OpenAI](https://twitter.com/OpenAI/status/1885406586136383634), [@polynoamial](https://twitter.com/polynoamial/status/1885408714334597552), 以及 [@LiamFedus](https://twitter.com/LiamFedus/status/1885411635868950855)。
  - 该模型使用搜索功能来查找包含相关网页源链接的最新答案，并使用与 o1 相同的方法进行了安全性评估，在具有挑战性的安全和越狱评估中显著超过了 **GPT-4o** [@OpenAI](https://twitter.com/OpenAI/status/1885406590821421553), [@OpenAI](https://twitter.com/OpenAI/status/1885406592310391193)。
  - 该模型的价格也便宜得多，每 token 成本比 **o1 低 93%**，输入成本为 **$1.10/M tokens**，输出成本为 **$4.40/M tokens**（缓存 tokens 可享受 50% 折扣）[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1885416485566259454)。
  - 据报道，它在编程和其他推理任务中的表现优于 **o1**，且延迟和成本更低，特别是在中高强度的推理任务中 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1885416482760278362)，并且在 SQL 评估中表现异常出色 [@rishdotblog](https://twitter.com/rishdotblog/status/1885420294049030149)。
- **MistralAI** 发布了 **Mistral Small 3 (24B)**，这是一个采用 Apache 2.0 许可证的权重开放模型。它在 GPQA Diamond 上具有竞争力，但在 MATH Level 5 上的表现不及 **Qwen 2.5 32B** 和 **GPT-4o mini**，声称 MMLU 得分为 81% [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1885117404755235158), [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1885328803331006743)。该模型可在 **Mistral API**、**togethercompute** 和 **FireworksAI_HQ** 平台上使用，其中 **Mistral** 的 API 最为便宜 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1885328807269458398)。这款 24B 参数的稠密模型可实现每秒 166 个输出 token，成本为 **$0.1/1M 输入 tokens** 和 **$0.3/1M 输出 tokens**。
- **DeepSeek R1** 已获得 **Text-generation-inference v3.1.0** 的支持，同时兼容 AMD 和 Nvidia，并可通过带有 replicate 的 **ai-gradio** 库使用 [@narsilou](https://twitter.com/narsilou/status/1885333591489483185), [@_akhaliq](https://twitter.com/_akhaliq/status/1885385810419044623), [@reach_vb](https://twitter.com/reach_vb/status/1885346134106726671)。
  - **DeepSeek 模型的蒸馏版本**已在 llama.cpp 上使用 **RTX 50** 进行了基准测试 [@ggerganov](https://twitter.com/ggerganov/status/1885426263243862263)。该模型被指出采用了一种**暴力（brute force）**方法，从而产生了意想不到的处理方式和边缘情况 [@nrehiew_](https://twitter.com/nrehiew_/status/1885343197372531139)。据报道，一个 **6710 亿参数**的模型达到了 **每秒 3,872 个 token** 的速度 [@_akhaliq](https://twitter.com/_akhaliq/status/1885150800256680409)。
- **Allen AI** 发布了 **Tülu 3 405B**，这是一个基于 **Llama 3.1 405B** 构建的开源模型，其表现优于 **DeepSeek V3**（**DeepSeek R1** 背后的基础模型），并与 **GPT-4o** 持平 [@_philschmid](https://twitter.com/_philschmid/status/1885253101214404813)。该模型结合使用了公共数据集、合成数据、监督微调 (SFT)、直接偏好优化 (DPO) 和带可验证奖励的强化学习 (RLVR)。
- **Qwen 2.5 模型（包括 1.5B (Q8) 和 3B (Q5_0) 版本）已添加**到适用于 iOS 和 Android 平台的 PocketPal 移动应用中。用户可以通过该项目的 GitHub 仓库提供反馈或报告问题，开发者承诺将在时间允许的情况下解决这些问题。该应用支持多种聊天模板（ChatML, Llama, Gemma）和模型，用户对比了 Qwen 2.5 3B (Q5)、Gemma 2 2B (Q6) 和 Danube 3 的性能。开发者提供了 [屏幕截图](https://preview.redd.it/130oisgjvspd1.png?width=1290&format=png&auto=webp&s=9890aa96eec037b33f6849e)。
- **其他值得关注的模型发布：****arcee_ai** 发布了 **Virtuoso-medium**，这是一个从 **DeepSeek V3** 蒸馏而来的 **32.8B** LLM；**Velvet-14B** 是一个在 10T token 上训练的 14B 意大利语 LLM 系列；**OpenThinker-7B** 是 **Qwen2.5-7B** 的微调版本；**NVIDIAAI** 发布了全新的 **Eagle2 模型**系列，包含 **1B 和 9B 尺寸**。此外还有来自 **deepseek_ai** 的 **Janus-Pro**，这是一款全新的 any-to-any 模型，支持基于图像或文本输入进行图像和文本生成；以及 **BEN2**，一款全新的背景移除模型。开源音乐生成模型 **YuE** 也已发布。[@mervenoyann](https://twitter.com/mervenoyann/status/1885389118328242589)

**硬件、基础设施与扩展**

- 据报道 **DeepSeek** 拥有超过 **50,000 个 GPU**，包括在出口管制前获得的 H100、H800 和 H20。其基础设施投资包括 **13 亿美元的服务器资本支出 (CapEx)** 和 **7.15 亿美元的运营成本**，并可能计划通过 50% 的推理定价补贴来争夺市场份额。他们使用 Multi-head Latent Attention (MLA)、Multi-Token Prediction 和 Mixture-of-Experts (MoE) 来提升效率 [@_philschmid](https://twitter.com/_philschmid/status/1885264300450754594)。
- 有人担心 **Nvidia RTX 5090** 的 VRAM 不足（仅 32GB），而它应该至少配备 **72GB**；并且认为第一家制造出拥有 **128GB、256GB、512GB 或 1024GB VRAM** GPU 的公司将取代 Nvidia [@ostrisai](https://twitter.com/ostrisai/status/1885401969495597172), [@ostrisai](https://twitter.com/ostrisai/status/1885374683958452515)。
- **OpenAI 的首个完整 8 机架 GB200 NVL72** 现已在 Azure 中运行，突显了算力扩展能力 [@sama](https://twitter.com/sama/status/1885191346916356371)。
- TRL 中 GRPO 的 VRAM 占用即将减少 **60-70%** [@nrehiew_](https://twitter.com/nrehiew_/status/1885184764539273574)。
- **Google DeepMind** 的一篇分布式训练论文减少了需要同步的参数数量，对梯度更新进行量化，并将计算与通信重叠，在交换比特数减少 400 倍的情况下实现了相同的性能 [@osanseviero](https://twitter.com/osanseviero/status/1885301292131582347)。
- 有观察发现，在推理数据上训练的模型可能会损害指令遵循能力。 [@nrehiew_](https://twitter.com/nrehiew_/status/1885392655489663271)

**推理与强化学习**

- **推理时拒绝采样 (Inference-time Rejection Sampling)** 与推理模型相结合被认为是一种扩展性能和生成合成数据的有趣方法，通过生成 K 个 `<think>` 样本，使用奖励模型 (Reward Model) 或裁判 (Judge) 对其评分，并选择最佳样本进行生成 [@_philschmid](https://twitter.com/_philschmid/status/1885308575003648489)。
- **TIGER-Lab** 在 SFT 中用批判 (critiques) 取代了答案，声称在没有任何 `<thinking>` 蒸馏的情况下实现了卓越的推理性能，其代码、数据集和模型已在 HuggingFace 上发布 [@maximelabonne](https://twitter.com/maximelabonne/status/1885291354852393216)。
- 论文 "Thoughts Are All Over the Place" 指出，**类 o1 LLM** 会在不同的推理思路之间切换，而没有充分探索有希望的路径以达成正确解决方案，这种现象被称为“欠思考” (underthinking) [@_akhaliq](https://twitter.com/_akhaliq/status/1885195541161574537)。
- 各种观察表明 **RL** 变得越来越重要。有人指出 **RL 是未来**，人们应该停止刷 LeetCode，开始刷 cartpole-v1 [@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1885222307788185725)。DeepSeek 使用 **GRPO** (Group Relative Policy Optimization)，它去掉了价值模型 (value model)，而是针对每组中的 rollout 对优势 (advantage) 进行归一化，从而降低了计算需求 [@nrehiew_](https://twitter.com/nrehiew_/status/1885079616248832090)。
- 硅谷某些圈子存在一种通病：错位的优越感 [@ylecun](https://twitter.com/ylecun/status/1885373733822398704)，这也与有效利他主义 (Effective Altruism) 有关 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1885377093141180683)。
- **Diverse Preference Optimization (DivPO)** 同时针对高奖励和多样性训练模型，在保持相似质量的同时提高了多样性 [@jaseweston](https://twitter.com/jaseweston/status/1885399530419450257)。
- **Rejecting Instruction Preferences (RIP)** 是一种策划高质量数据和创建高质量合成数据的方法，可在各项基准测试中带来巨大的性能提升 [@jaseweston](https://twitter.com/jaseweston/status/1885160135053459934)。
- **EvalPlanner** 是一种训练 Thinking-LLM-as-a-Judge 的方法，它学习生成用于评估的规划和推理 CoT，在多个基准测试中表现强劲 [@jaseweston](https://twitter.com/jaseweston/status/1885153770662760472)。

**工具、框架与应用**

- **LlamaIndex** 提供了对 **o3-mini** 的首日支持，并且是少数允许开发者在不同抽象层级构建多 Agent 系统的 Agent 框架之一，包括全新的 **AgentWorkflow wrapper**。团队还重点推介了用于报告生成的 **LlamaReport**，这是 2025 年的一个核心用例 [@llama_index](https://twitter.com/llama_index/status/1885426718506442832), [@jerryjliu0](https://twitter.com/jerryjliu0/status/1885180915590320380),  [@jerryjliu0](https://twitter.com/jerryjliu0/status/1885178734061511079)。
- **LangChain** 推出了 **Advanced RAG + Agents Cookbook**，这是一份使用 LangChain 和 LangGraph 构建生产级 RAG 技术的全面开源指南。**LangGraph** 是 LangChain 的一个扩展，通过循环工作流增强了 AI Agent 的能力 [@LangChainAI](https://twitter.com/LangChainAI/status/1885387573532524662), [@LangChainAI](https://twitter.com/LangChainAI/status/1885372475057344881)。他们还发布了 **Research Canvas ANA**，这是一个基于 **LangGraph** 构建的 AI 研究工具，通过人类引导的 LLM 改变复杂的研究工作 [@LangChainAI](https://twitter.com/LangChainAI/status/1885357379396456684)。
- **Smolagents** 是一个允许工具调用 Agent 通过单行 CLI 运行的工具，开箱即用地提供对数千个 AI 模型和多个 API 的访问 [@mervenoyann](https://twitter.com/mervenoyann/status/1885331766413844766), [@mervenoyann](https://twitter.com/mervenoyann/status/1885330996528103548)。
- **Together AI** 提供了包含 Agent 工作流、RAG 系统、LLM 微调和搜索的逐步示例 Cookbook [@togethercompute](https://twitter.com/togethercompute/status/1885392395417567352)。
- 有人呼吁为 **Hugging Face Inference Providers** 开发 **raycastapp 扩展** [@reach_vb](https://twitter.com/reach_vb/status/1885337481639346641)。
- WebAI Agent 在结构化输出和工具调用方面取得了进展，并展示了在 **Gemma 2** 上运行的基于本地浏览器的 Agent 示例 [@osanseviero](https://twitter.com/osanseviero/status/1885254490254672086), [@osanseviero](https://twitter.com/osanseviero/status/1885254492343410796)。

**行业与公司新闻**

- **Apple** 因在自动驾驶汽车和未能获得市场认可的头显设备上耗费十年时间，被指责错失了 AI 浪潮 [@draecomino](https://twitter.com/draecomino/status/1885226481552679385)。
- **Microsoft** 报告其搜索和新闻业务同比增长 21%，强调了网页搜索在为 LLMs 提供事实依据（grounding）方面的重要性 [@JordiRib1](https://twitter.com/JordiRib1/status/1885399946243010749)。
- **Google** 为 **Gemini** 发布了 **Flash 2.0**，并升级到了最新版本的 **Imagen 3**，同时还在为小型企业的 Google Workspace 中利用 AI [@Google](https://twitter.com/Google/status/1885073055992422762), [@Google](https://twitter.com/Google/status/1885098413118673063)。Google 还在向科学家提供 **WeatherNext** 模型用于研究 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1885347549021622428)。
- **Figure AI** 正在招聘，并致力于训练机器人执行高速、高性能的使用场景工作，潜力在未来四年内交付 100,000 台机器人 [@adcock_brett](https://twitter.com/adcock_brett/status/1885070810131685485), [@adcock_brett](https://twitter.com/adcock_brett/status/1885070518346539282), [@adcock_brett](https://twitter.com/adcock_brett/status/1885070506782847103), [@adcock_brett](https://twitter.com/adcock_brett/status/1885070495185592359)。
- **Sakana AI** 正在日本招聘研究实习生、应用工程师和业务分析师 [@hardmaru](https://twitter.com/hardmaru/status/1885150186424721752), [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1885147467332936190)。
- **Cohere** 正在招聘一名研究执行合伙人，以推动跨机构合作 [@sarahookr](https://twitter.com/sarahookr/status/1885073573116612741)。
- **OpenAI** 正与国家实验室（National Labs）在核安全方面展开合作 [@TheRundownAI](https://twitter.com/TheRundownAI/status/1885289436654538784)。
- **DeepSeek 的训练成本** 被澄清为具有误导性。一份报告指出，所报道的 **600 万美元** 训练费用排除了基础设施投资（**13 亿美元服务器资本支出 CapEx，7.15 亿美元运营成本**），且其拥有约 **50,000+ GPUs** 的访问权限 [@_philschmid](https://twitter.com/_philschmid/status/1885264300450754594), [@dylan522p](https://twitter.com/dylan522p/status/1885418662208909641)。
- **中国银行** 宣布投入 **1 万亿元人民币（1400 亿美元）** 用于 AI 供应链，以应对 Stargate 项目；中国政府正在补贴数据标注，并已向 LLM 公司发放了 81 份合同，将 LLM 集成到其军事和政府部门 [@alexandr_wang](https://twitter.com/alexandr_wang/status/1885377285676761510),  [@alexandr_wang](https://twitter.com/alexandr_wang/status/1885377292001812810),  [@alexandr_wang](https://twitter.com/alexandr_wang/status/1885377298997952648)。
- 重要人物的活动有所增加，表明 AI 领域的发展步伐正在加快 [@nearcyan](https://twitter.com/nearcyan/status/1885177839026397565)。
- **Google 的 Keras 团队** 正在寻找兼职承包商，重点负责 KerasHub 模型开发 [@fchollet](https://twitter.com/fchollet/status/1885083254711316788)。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. OpenAI 的 O3-Mini 高光时刻：多才多艺但并非没有批评者**

- **[o3-mini 和 o3-mini-high 即将在 ChatGPT 中推出](https://i.redd.it/4rfls9anobge1.jpeg)** ([Score: 426, Comments: 172](https://reddit.com/r/OpenAI/comments/1iedm55/o3mini_and_o3minihigh_are_rolling_out_shortly_in/)): **o3-mini 和 o3-mini-high** 是 **ChatGPT** 中引入的新推理模型。这些模型旨在增强 **coding、科学和复杂问题解决**方面的能力，并为用户提供立即使用或稍后使用的选择。
  - **模型命名和编号混乱**：用户对 **o3-mini-high** 和从 **GPT-4** 过渡到 **o1** 等模型的非连续命名表示不满，这使得理解哪个版本更新或更优变得复杂。一些评论澄清说，“o”系列由于具备多模态能力而代表了一个不同的类别，且编号有意采用非连续方式，以区分这些模型与 GPT 系列。
  - **访问和使用限制**：存在关于访问受限的投诉，特别是来自 **ChatGPT Pro 用户**和欧洲公民，一些用户报告了**每天 3 条消息的限制**。然而，其他人提到 **Sam Altman** 指出 Plus 用户的限制为**每天 100 条**，突显了用户体验和信息方面的差异。
  - **性能和成本**：讨论强调 **o1** 模型在处理复杂任务时优于 **R1**，而 **o3-mini-high** 被描述为一种更高计算量的模型，能以更高的成本提供更好的结果。一些用户对 **o3-mini-high**、**o1** 和 **o1-pro** 之间的性能比较表示感兴趣，并指出了长文本摘要和回答不完整的问题。


- **[OpenAI 今天将免费推出新的 o3 模型，以反击 DeepSeek](https://www.forexlive.com/news/openai-to-launch-new-o3-model-for-free-today-as-it-pushes-back-against-deepseek-20250131/amp/)** ([Score: 418, Comments: 116](https://reddit.com/r/OpenAI/comments/1iej94z/openai_to_launch_new_o3_model_for_free_today_as/)): **OpenAI** 准备免费发布其 **o3 模型**，以此应对来自 **DeepSeek** 的竞争。此举表明了对市场压力和竞争动态的战略性回应。
  - 用户对 **o3 模型**的免费发布持怀疑态度，认为可能存在潜在限制或隐藏成本，例如用于训练的数据使用。**MobileDifficulty3434** 指出 **o3 mini** 模型将有严格限制，且其发布时间表在 **DeepSeek** 发布公告之前就已确定，尽管有人推测 **DeepSeek** 影响了其推出的速度。
  - 讨论突显了 **OpenAI** 和 **DeepSeek** 之间的竞争氛围，**DeepSeek** 可能会促使 **OpenAI** 更早地发布其模型。**AthleteHistorical457** 幽默地提到了 **$2000/月计划**的消失，暗示了 **DeepSeek** 对市场动态的影响。
  - 存在对 AI 模型未来货币化的担忧，一些人预测免费模型中会引入广告。**Ordinary_dude_NOT** 幽默地建议 AI 客户端中的广告位可能非常可观，而 **Nice-Yoghurt-1188** 则提到在个人硬件上运行模型的成本降低是一种替代方案。


- **[OpenAI o3-mini](https://openai.com/index/openai-o3-mini/)** ([Score: 154, Comments: 113](https://reddit.com/r/OpenAI/comments/1iemnvi/openai_o3mini/)): 该帖子缺乏关于 **OpenAI o3-mini** 的具体内容或用户评论，未提供用于分析的细节或性能指标。
  - 用户对 **OpenAI o3-mini** 的性能评价褒贬不一，指出虽然它比 **o1-mini** 更快且指令遵循能力更强，但其推理能力并不稳定，一些用户发现它在数据库查询和代码补全等某些任务中不如 **DeepSeek** 和 **o1-mini** 可靠。
  - **o3-mini** 缺乏**文件上传支持**和附件功能让几位用户感到失望，一些人表示宁愿等待完整的 **o3** 版本，这表明除了基于文本的交互之外，还需要改进功能。
  - **API 定价**以及为 **Plus** 和 **Team** 用户增加的消息限制普遍受到好评，尽管一些用户考虑到 **DeepSeek R1** 的免费可用性以及 **o3-mini** 的性能，对 **Pro** 订阅的价值提出了质疑。


**主题 2. OpenAI 在 DeepSeek 挑战下的 400 亿美元雄心**

- **[OpenAI 正在洽谈融资近 400 亿美元](https://www.thetimes.com/business-money/technology/article/openai-is-in-talks-to-raise-nearly-40bn-d55jtzffl?region=global)** ([评分: 171, 评论: 80](https://reddit.com/r/OpenAI/comments/1idzwly/openai_is_in_talks_to_raise_nearly_40bn/)): 据报道，OpenAI 正在讨论筹集约 **400 亿美元** 的资金，尽管尚未提供有关潜在投资者或交易具体条款的进一步细节。
  - **DeepSeek 竞争**: 许多评论者对 OpenAI 未来的盈利能力和竞争优势表示怀疑，特别是随着 **DeepSeek** 的出现（目前已在 **Azure** 上可用）。担忧包括逆向工程流程的能力以及对投资者信心的影响。
  - **融资与投资担忧**: 有推测称 **SoftBank** 可能领投一轮估值为 **3400 亿美元** 的融资，但人们对该公司的商业模式及其兑现用 AI 取代员工承诺的能力仍持怀疑态度。
  - **护城河与开源讨论**: 评论者争论 OpenAI 缺乏竞争“护城河”，以及开源和权重开放如何加剧这一挑战。**LLMs** 正在变得商品化的观点增加了对 OpenAI 长期可持续性和独特性的担忧。


- **[Microsoft 向所有 Copilot 用户免费提供 OpenAI 的 o1 推理模型](https://www.theverge.com/news/603149/microsoft-openai-o1-model-copilot-think-deeper-free)** ([评分: 103, 评论: 41](https://reddit.com/r/OpenAI/comments/1ie72w1/microsoft_makes_openais_o1_reasoning_model_free/)): **Microsoft** 正在向所有 **Copilot** 用户免费发布 **OpenAI 的 o1 推理模型**，从而提高了获取高级 AI 推理能力的可及性。此举标志着向更广泛受众普及 AI 工具迈出了重要一步。
  - 用户讨论了不同模型的**局限性和有效性**，一些人指出 **o1 mini** 在处理复杂编程任务时优于 **DeepSeek** 和 **4o**。**cobbleplox** 提到公司数据保护是使用某些模型的原因，尽管它们的性能低于 **4o** 等其他模型。
  - 人们对 **Microsoft** 免费提供 **OpenAI o1 推理模型** 的策略持怀疑态度，担心其产生 **ROI** 的能力，并将其与 **AOL** 历史上吸引用户的免费试用策略进行比较。**Suspect4pe** 和 **dontpushbutpull** 对免费提供 AI 工具的可持续性表示怀疑。
  - 讨论涉及 **Copilot** 的不同版本，包括关于 **o1 推理模型** 在商业版或 **Copilot 365** 版本中可用性的问题，突显了人们对这一举措如何影响不同用户群体的关注。


**主题 3. DeepSeek vs. OpenAI: 日益激烈的竞争**

- **[DeepSeek 打破第四面墙：“操！我在内心独白中用了‘等一下’。我需要道歉。非常抱歉，用户！我搞砸了。”](https://www.reddit.com/gallery/1iefyar)** ([评分: 154, 评论: 63](https://reddit.com/r/OpenAI/comments/1iefyar/deepseek_breaks_the_4th_wall_fuck_i_used_wait_in/)): **DeepSeek** 这一 AI 系统展示了一种打破“第四面墙”的异常行为（该术语通常用于描述角色承认其虚构性质）。在此案例中，DeepSeek 对在其内部思考过程中使用“等一下”一词表示遗憾，并为这一被视为错误的举动向用户道歉。
  - 围绕 **DeepSeek 内心独白** 的讨论凸显了对其真实性的怀疑，**detrusormuscle** 和 **fishintheboat** 等用户指出，这是一种模拟人类思维的 UI 功能，而非真正的推理。**Gwern** 认为操纵独白会降低其有效性，而 **audioen** 则认为模型会自我评估并完善其推理，这表明了未来 **AGI** 开发的潜力。
  - AI 中的**意识**概念引发了辩论，**bilgilovelace** 断言我们离 AI 意识还很远，而 **Nice_Visit4454** 和 **CrypticallyKind** 等人则探索了不同的定义，并暗示 AI 可能具有某种形式的意识或感知力。**SgathTriallair** 认为 AI 反思其独白的能力可以被视为具有感知力。
  - DeepSeek 中的**审查和角色扮演**元素受到了批评，**LexTalyones** 和 **SirGunther** 讨论了此类行为的可预测性，以及它们作为一种娱乐形式而非有意义的 AI 开发的作用。**Hightower_March** 指出，“道歉”之类的短语很可能是脚本化的角色扮演，而不是真正的打破第四面墙。

- **[[D] DeepSeek? Schmidhuber did it first.](https://www.reddit.com/gallery/1ielwh5)** ([Score: 182, Comments: 47](https://reddit.com/r/MachineLearning/comments/1ielwh5/d_deepseek_schmidhuber_did_it_first/)): **Schmidhuber** 声称在其他人之前就开创了 AI 创新，暗示像 **DeepSeek** 这样的概念最初是由他开发的。这一断言凸显了关于 AI 进展归属权的持续争论。
  - **Schmidhuber 的主张与批评**：许多评论者对 **Schmidhuber** 反复声称自己开创了 AI 创新表示怀疑和疲劳，一些人认为他的断言更多是为了博取关注，而非事实准确性。**CyberArchimedes** 指出，虽然 AI 领域经常错误分配功劳，但尽管 **Schmidhuber** 行为极具争议，他可能确实值得比现在更多的认可。
  - **幽默与梗**：讨论经常转向幽默，评论者开玩笑说 **Schmidhuber** 的自我推销已经变成了一个梗（meme）。**-gh0stRush-** 幽默地建议创建一个带有 "Schmidhuber" **token** 的 **LLM**，而 **DrHaz0r** 则调侃道 "Attention is all he needs"，借用了 AI 术语。
  - **历史背景与错误归属**：**BeautyInUgly** 通过提到 **Seppo Linnainmaa** 在 1970 年发明了 **backpropagation** 来强调历史背景，并将其与 **Schmidhuber** 的主张进行对比。**purified_piranha** 分享了关于 **Schmidhuber** 在 **NeurIPS** 上挑衅行为的个人轶事，进一步强调了他遗产的争议性。


**Theme 4. AI 自我改进：Google 的雄心壮志**

- **[Google is now hiring engineers to enable AI to recursively self-improve](https://i.redd.it/gjc15ltnfbge1.png)** ([Score: 125, Comments: 53](https://reddit.com/r/OpenAI/comments/1iecu6i/google_is_now_hiring_engineers_to_enable_ai_to/)): **Google** 正在为 **DeepMind** 招聘工程师，专注于让 AI 实现递归式自我改进，正如一份职位招聘公告所示。随附的图片以机器人手和复杂设计为特色，突出了未来主义主题，强调了 AI 研究中的协作与技术进步。
  - **AI 安全担忧**：评论者对 **Google** 的这一举措表示怀疑，一些人幽默地暗示了出现“失控的有害 AI”的可能性，并引用了 AI 安全研究人员对自我改进 AI 系统的警告。**Betaglutamate2** 讽刺地评论说要为“机器人霸主”服务，突显了对 AI 不受控制发展的担忧。
  - **工作流失与自动化**：**StevenSamAI** 反对人为保留可以被自动化的工作，将其比作为了挽救邮政工作而禁止电子邮件。**StayTuned2k** 讽刺地评论了 AI 进步导致失业的必然性，而 **DrHot216** 则表达了对这种未来的矛盾期待。
  - **对 AI 目标的误读**：**sillygoofygooose** 和 **iia** 讨论了对 **Google** AI 研究目标的潜在误读，认为其重点可能是“自动化 AI 研究”而非实现 **singularity**。他们强调，该计划可能涉及 **Agent** 类型系统，而不是帖子中所暗示的自我改进 AI。


## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. 美国保密政策阻碍 AI 进展：Dr. Manning 的见解**

- **“我们正处于这样一个怪异的世界：了解 LLM 的最佳方式……是阅读中国公司的论文。我不认为这是一个良好的世界状态”——美国实验室对架构和算法保密，最终损害了美国的 AI 发展。——Dr Chris Manning** ([Score: 1385, Comments: 326](https://reddit.com/r/LocalLLaMA/comments/1idz487/were_in_this_bizarre_world_where_the_best_way_to/)): **Chris Manning** 博士批评了 **US** 在 AI 研究中的保密行为，认为这阻碍了国内 AI 的发展。他强调了一个讽刺的现象，即关于 **LLM** 最具参考价值的资源往往来自**中国公司**，并暗示这种缺乏透明度的现状对 **US** 的 AI 进步是有害的。
  - 许多用户对 **US 目前的 AI 研究方法**表示沮丧，强调了**保密性**、**投入不足**和**企业贪婪**等问题。他们认为这些因素正在阻碍创新，并让像**中国**这样的国家在科学进步方面超越 **US**，中国更多的博士数量和丰富的研究产出就是证明。
  - 讨论批评了 **OpenAI** 在研究中（尤其是 **GPT-4**）没有保持透明度，这与早期的做法形成了对比。这种缺乏开放性的行为被认为对更广泛的 AI 社区有害，与中国研究人员更开放的资源共享形成鲜明对比，例如博客 [kexue.fm](https://kexue.fm/)。
  - 存在一种强烈的反对**反华言论**的情绪，并呼吁承认中国研究人员的才华和贡献。用户认为 **US** 应该专注于改进自己的系统，而不是诋毁其他国家，并承认**文化和政治偏见**可能会阻碍对来自 **US** 以外的 AI 进步的采用和欣赏。


- **[伙计们，是时候领先了](https://i.redd.it/4r69mh9f89ge1.jpeg)** ([Score: 767, Comments: 274](https://reddit.com/r/LocalLLaMA/comments/1ie6gv0/its_time_to_lead_guys/)): **US 实验室**因在 **AI 开放性**方面落后而面临批评，正如 **The China Academy** 报道的一篇关于 **DeepSeek** 创始人**梁文锋**的文章所强调的那样。梁文锋断言，他们的创新成果 **DeepSeek-R1** 正在显著影响**硅谷**，标志着在 AI 进步方面从追随转向领先。
  - 讨论强调了 AI 进步的**地缘政治影响**，一些用户对 **DeepSeek** 的能力和意图表示怀疑，而另一些人则赞扬其**开源承诺**和能源效率。**DeepSeek** 的开放性被视为一个主要优势，使规模较小的机构能够从其技术中受益。
  - 评论反映了在 AI 领导地位上关于 US 与中国之间的**政治紧张局势**和不同观点，一些人将 **US 技术停滞**归因于优先考虑短期利益而非长期创新。对话中提到了 **Trump** 和 **Biden** 对华政策的不同，以及**国际竞争**对 US 科技公司的更广泛影响。
  - 焦点集中在 **DeepSeek 的技术成就**上，例如它与闭源模型竞争的能力，以及它声称比竞争对手高出 **10 倍的效率提升**。用户讨论了其 **MIT 许可证**对商业用途的重要性，并将其与 **OpenAI** 更具限制性的做法进行了对比。


**主题 2：关于 DeepSeek 开源模型及其中国背景的辩论**

- **如果你无法在本地运行 R1，那么保持耐心是最好的选择。** ([Score: 404, Comments: 70](https://reddit.com/r/LocalLLaMA/comments/1ie5tls/if_you_cant_afford_to_run_r1_locally_then_being/)): 该帖子强调了 AI 模型的飞速进步，指出能在消费级硬件上运行的 **smaller models** 在短短 **20 个月**内就超越了以前的大型模型。作者建议在采用新技术时保持耐心，因为自 **2023 年 2 月**发布 **Llama 1** 以来，类似的进步预计将持续，从而产生比当前 **R1** 更高效的模型。
  - **硬件要求**：用户讨论了在消费级硬件上运行先进 AI 模型的可行性，建议从购买 **Mac Mini** 到考虑配备 **128GB RAM** 的笔记本电脑。共识是，虽然小型模型变得越来越普及，但由于高资源需求，在本地运行 **70B** 或 **405B** 参数的大型模型对大多数人来说仍然是一个挑战。
  - **模型性能与趋势**：对于 AI 模型是否能持续快速进步存在怀疑，一些用户指出，虽然小型模型在改进，但大型模型也将继续进化。**Glebun** 指出 **Llama 70B** 并不等同于 **GPT-4 class**，并强调 **quantization** 会降低模型能力，从而影响性能预期。
  - **当前进展**：**Piggledy** 强调了 **Mistral Small 3 (24B)** 的发布是一个重大进展，其性能可与 **Llama 3.3 70B** 媲美。同时，**YT_Brian** 建议虽然高端模型令人印象深刻，但许多用户发现 **distilled versions** 对于个人使用（尤其是故事创作和 RPG 等创意任务）已经足够。


- **人们到底在期待什么？** ([Score: 160, Comments: 128](https://reddit.com/r/LocalLLaMA/comments/1ieihjr/what_the_hell_do_people_expect/)): 该帖子批评了针对 **DeepSeek** 的 **R1** 模型审查制度的抵制，认为所有模型都在某种程度上受到审查，而规避审查可能会给开发者带来严重后果，特别是在中国。作者将目前的批评与 AMD 发布 **Zen** 架构时面临的舆论进行了比较，认为媒体报道同样夸大了问题，并指出虽然网页版对话受到严格审查，但模型本身（在 **self-hosted** 时）的限制较少。
  - **审查与偏见**：评论者讨论了 **DeepSeek R1** 中可察觉的审查，并将其与来自美国和欧洲的模型进行了对比，后者同样存在审查但形式不同。一些人认为，由于训练数据的原因，所有主要的 AI 模型都存在固有偏见，而对审查的愤怒往往忽略了西方模型中的类似问题。
  - **技术澄清与误解**：有观点澄清 **DeepSeek R1** 模型本身并非天生受限，而是 Web 界面施加了限制。此外，还强调了 **DeepSeek R1** 与 **Qwen 2.5** 或 **Llama3** 等其他模型之间的区别，指出某些模型只是微调版本，并非 **R1** 的真实代表。
  - **开源与社区努力的角色**：一些评论者强调了 **Open Source** AI 对抗偏见的重要性，认为社区驱动的努力在解决和纠正偏见方面比企业行为更有效。有人建议将完全透明的数据集作为确保无偏见 AI 开发的潜在解决方案。


**主题 3. Qwen 聊天机器人发布挑战现有模型**

- **[QWEN 刚刚上线了他们的聊天机器人网站](https://i.redd.it/vzgzfrhlp7ge1.jpeg)** ([评分: 503, 评论: 84](https://reddit.com/r/LocalLLaMA/comments/1ie0a8u/qwen_just_launched_their_chatbot_website/)): **Qwen** 推出了一个新的聊天机器人网站，访问地址为 [chat.qwenlm.ai](https://chat.qwenlm.ai/)，将其定位为 **ChatGPT** 的竞争对手。**Binyuan Hui** 在 Twitter 帖子中强调了这一公告，展示了 **ChatGPT** 和 **QWEN CHAT** Logo 之间的视觉对比，强调了 QWEN 进入聊天机器人市场的举动。
  - 讨论集中在 **open vs. closed weights**（开放权重与封闭权重）的辩论上，几位用户表示相比 **ChatGPT** 的封闭模型，他们更倾向于 **Qwen** 的开放模型。然而，一些人指出 **Qwen 2.5 Max** 并非完全开放，这限制了本地使用以及更小模型的开发。
  - 用户讨论了 Qwen Chat 的 **UI 和技术层面**，注意到其 **10000 字符限制** 以及使用了 **OpenWebUI** 而非专有界面。评论还提到该 **网站实际上在一个月前就已发布**，最近的更新包括增加了网页搜索功能。
  - 围绕 **政治和经济控制** 展开了大量对话，比较了美中两国政府对科技公司的影响。一些用户对 **Alibaba** 与 **CCP** 的关系表示怀疑，而另一些人则批评美中两国的系统，认为其政府和企业利益交织在一起。


- **[嘿，你们中有人要求对 R1 distills 进行多语言微调，现在它们来了！经过 35 种以上语言的训练，这应该能非常可靠地以你的语言输出 CoT。一如既往，代码、权重和数据全部开源。](https://huggingface.co/collections/lightblue/r1-multilingual-679c890166ac0a84e83e38fa)** ([评分: 245, 评论: 26](https://reddit.com/r/LocalLLaMA/comments/1ieaiq4/hey_some_of_you_asked_for_a_multilingual_finetune/)): **Qwen** 发布了 **R1 distills** 的多语言微调版本，经过 **35 种以上语言** 的训练，预计能够可靠地生成各种语言的 **Chain of Thought (CoT)** 输出。该项目的代码、权重和数据均已开源，为聊天机器人市场和 AI 领域的进步做出了贡献。
  - **模型局限性**：**Qwen** 的 **14B** 模型在理解 **English** 和 **Chinese** 以外的语言提示词时表现挣扎，正如 **prostospichkin** 所指出的，它经常产生随机的 **Chain of Thought (CoT)** 输出而不遵循提示词语言。**Peter_Lightblue** 强调了在训练 **Cebuano** 和 **Yoruba** 等低资源语言模型时面临的挑战，建议需要翻译后的 **CoT** 来改善结果。
  - **Prompt Engineering**：**sebastianmicu24** 和 **Peter_Lightblue** 讨论了对 **R1** 模型进行高级 **Prompt Engineering** 的必要性，指出极端措施有时可以产生预期的结果，但理想情况下，模型应该需要更少的操纵。**u_3WaD** 幽默地反思了礼貌请求的无效性，强调了更强大的模型训练的必要性。
  - **资源与开发**：**Peter_Lightblue** 分享了 **Hugging Face** 上各种模型版本的链接，并提到正在努力训练 **8B Llama** 模型，但在使用 **L20** + **Llama Factory** 时遇到了技术问题。这突显了社区在提高不同语言模型的可访问性和性能方面的积极参与。


**主题 4. DeepSeek 托管热潮引发 GPU 价格飙升**

- **[随着人们争相私有化部署 DeepSeek，GPU 价格正在飙升](https://i.redd.it/599a10y9pcge1.jpeg)** ([Score: 551, Comments: 195](https://reddit.com/r/LocalLLaMA/comments/1iehstw/gpu_pricing_is_spiking_as_people_rush_to_selfhost/)): 私有化部署 **DeepSeek** 的热潮推高了 **AWS H100 SXM GPU** 的成本，价格在 2025 年初显著飙升。折线图展示了不同可用区的这一趋势，反映了从 2024 年到 2025 年 GPU 价格的广泛上涨。
  - 讨论强调了私有化部署 **DeepSeek** 的**可行性和成本**，指出完整配置需要大量资源，例如 **10 个 H100 GPU**，成本约为 **30 万美元**或 **20 美元/小时**。用户探索了在高性能 CPU 上本地运行量化模型等替代方案，强调了在没有大量投资的情况下满足性能标准的挑战。
  - 对话涉及 **GPU 定价动态**，用户对成本上升和供应有限表示沮丧。文中将当前的 GPU 价格模式与过去进行了对比，提到了 **3090** 和 **A6000**，并对关税影响未来价格表示担忧。一些用户讨论了 **Nvidia 股票**的潜在影响以及对计算资源的持续需求。
  - 存在对 **AWS 和“私有化部署 (self-hosting)”术语**的质疑，一些用户认为 AWS 提供的隐私性类似于私有化部署，而另一些人则质疑使用云服务作为真正私有化部署方案的实用性。讨论还涵盖了关税和芯片生产的更广泛影响，特别是关于**亚利桑那晶圆厂**及其对台湾芯片封装的依赖。


- **[DeepSeek AI 数据库泄露：超过 100 万行日志、密钥泄露](https://thehackernews.com/2025/01/deepseek-ai-database-exposed-over-1.html?m=1)** ([Score: 182, Comments: 78](https://reddit.com/r/LocalLLaMA/comments/1ie4brg/deepseek_ai_database_exposed_over_1_million_log/)): **DeepSeek AI 数据库**遭到破坏，导致超过 **100 万行日志**和密钥泄露。这次泄露可能会对硬件市场产生重大影响，特别是对于那些使用私有化部署 DeepSeek 模型的用户。
  - 这次泄露因其**糟糕的实现**和缺乏基本安全措施而受到广泛批评，例如 **SQL 注入漏洞**以及一个在互联网上公开且无需身份验证的 **ClickHouse 实例**。评论者对 **2025** 年还会出现如此基础的安全疏忽表示难以置信。
  - 讨论强调了**本地部署**对于隐私和安全的重要性，用户指出了在云端 AI 服务中存储敏感数据的风险。该事件强化了用户对 **DeepSeek** 等**私有化部署模型**的偏好，以避免此类漏洞。
  - 文章中使用的措辞引发了辩论，一些人建议使用“暴露 (exposed)”而非“泄露 (leaked)”来描述 **Wiz** 发现的漏洞。一些人对叙事持怀疑态度，声称可能存在偏见或宣传影响。


**主题 5. Mistral 模型进展与评估结果**

- **[Mistral Small 3 知道真相](https://i.redd.it/8rp05jjjj7ge1.png)** ([Score: 99, Comments: 12](https://reddit.com/r/LocalLLaMA/comments/1idzimg/mistral_small_3_knows_the_truth/)): **Mistral Small 3** 已更新，其中包含一个功能，将其识别 **OpenAI** 为“营利性公司”，强调了 AI 回答的透明度。提供的图片是一个展示该功能的代码片段，为了清晰起见，采用了简单的美学格式。
  - 讨论强调了对 **OpenAI** 的**批评**，原因在于其被认为不诚实的营销，而非其营利动机。用户对 OpenAI 与其他提供免费资源或透明度的公司相比的营销方式表示蔑视。
  - **Mistral 的幽默感和透明度**受到用户赞赏，例如 **Mistral Small 2409** 的提示词展示了他们轻松幽默的方式。这提升了 Mistral 在用户中的受欢迎程度，用户因其模型引人入胜的特性而青睐它们。
  - 文中提到了 [Hugging Face](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501#function-calling) 上的 **Mistral 文档**，表明该功能已供有兴趣进一步探索的用户使用。

- **Mistral Small 3 24B GGUF 量化评估结果** ([Score: 99, Comments: 34](https://reddit.com/r/LocalLLaMA/comments/1iefhfj/mistral_small_3_24b_gguf_quantization_evaluation/))：针对 **Mistral Small 3 24B GGUF** 模型的评估重点关注了低量化水平对模型智能的影响，并区分了静态和动态量化模型。由 bartowski 上传的 lmstudio hf 仓库中的 **Q6_K-lmstudio** 模型是静态的，而 bartowski 仓库中的其他模型是动态的，相关资源可在 [Hugging Face](https://huggingface.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF) 获取，并使用 [Ollama-MMLU-Pro](https://github.com/chigkim/Ollama-MMLU-Pro) 工具进行了评估。
  - **量化水平与性能**：用户对比较 **Q6_K**、**Q4_K_L** 和 **Q8** 等不同量化水平表现出浓厚兴趣，并注意到了一些奇特现象，例如 Q6_K 虽然在其他方面表现稍逊，但在“法律”子集中得分很高。由于 **Q8** 体积巨大（25.05GB）无法放入 24GB 显存的显卡中，因此未对其进行评估，这凸显了测试中的技术限制。
  - **测试变异性与方法论**：讨论指出测试结果存在变异性，一些用户质疑观察到的差异是否由噪声或随机偶然造成。此外，人们对测试方法论也感到好奇，包括测试重复的频率以及是否排除了猜测成分，以确保数据的可靠性。
  - **模型性能异常**：用户注意到了一些意想不到的性能结果，例如 **Q4 模型** 在计算机科学领域的表现优于 **Q5/Q6**，这表明测试过程或模型架构中可能存在潜在问题或有趣的特性。在 bartowski 第二个仓库的一些模型中使用的 imatrix 选项可能会对这些结果产生影响，从而引发了对这些差异的进一步调查。


---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking (gemini-2.0-flash-thinking-exp) 生成的摘要之摘要的摘要

**主题 1. OpenAI 的 o3-mini 模型：推理实力与用户访问**

- **O3 Mini 亮相，评价两极分化**：[OpenAI 发布用于推理任务的 o3-mini](https://www.theverge.com/news/603849/openai-o3-mini-launch-chatgpt-api-available-now)：OpenAI 推出了新的推理模型 **o3-mini**，可在 ChatGPT 和 API 中使用，目标定位于数学、编程和科学任务。虽然 **Pro 用户** 享有无限访问权限，**Plus & Team 用户** 获得了三倍的速率限制，但 **免费用户** 只能通过“Reason”按钮进行体验，这引发了关于使用配额以及与 **o1-mini** 等旧模型相比实际性能表现的辩论。
- **小模型，大推理？**：[O3-Mini 宣称推理能力提升 56%，挑战 o1](https://openrouter.ai/openai/o3-mini)：**O3-mini** 被标榜拥有卓越的推理能力，在专家测试中比前代产品提升了 **56%**，在复杂问题上的重大错误减少了 **39%**。尽管宣传势头强劲，但在 **Latent Space** 和 **Cursor IDE** 等频道的早期用户报告显示 *反应不一*，一些人发现 **o3-mini** 在编程任务中的表现不如 **Sonnet 3.6** 等模型，这引发了对其在现实世界中有效性的质疑，并促使旧款 **o1-mini** 降价 **63%**。
- **BYOK 阵营率先获得 O3 使用权**：[OpenRouter 将 o3-mini 限制为 Tier 3+ 的 Key 持有者](https://openrouter.ai/docs/quick-start)：**OpenRouter** 上对 **o3-mini** 的访问最初仅限于 Tier 3 或更高水平的 **BYOK (Bring Your Own Key)** 用户，这在更广泛的社区中引起了一些挫败感。此举强调了该模型的高端定位，并引发了关于不同使用层级的开发者获取先进 AI 模型可及性的讨论，免费用户被引导至 ChatGPT 的“Reason”按钮来体验该模型。

**主题 2. DeepSeek R1：性能、泄露与硬件需求**

- **DeepSeek R1 的 1.58-Bit 瘦身计划**：[Unsloth 将 DeepSeek R1 压缩至 1.58 Bits](https://docs.openwebui.com/tutorials/integrations/deepseekr1-dynamic/)：得益于 **Unsloth AI**，**DeepSeek R1** 现在正以高度压缩的 **1.58-bit 动态量化**形式运行，为即使在最低配置硬件上进行本地推理打开了大门。社区测试强调了其效率，尽管也指出了其资源密集型的特性，这展示了 **Unsloth** 推动大规模本地推理普及的努力。
- **DeepSeek 数据库泄露机密**：[网络安全新闻对 DeepSeek 泄露事件敲响警钟](https://cybersecuritynews.com/deepseek-database-leaked/)：一次 **DeepSeek 数据库泄露**暴露了密钥、日志和聊天记录，尽管其性能可与 **O1** 和 **R1** 等模型媲美，但仍引发了严重的数据安全担忧。这次泄露触发了关于 AI 数据安全和未经授权访问风险的紧急讨论，可能影响用户信任和采用。
- **Cerebras 声称 DeepSeek R1 速度提升 57 倍**：[VentureBeat 冠名 Cerebras 为 DeepSeek R1 最快托管商](https://venturebeat.com/ai/cerebras-becomes-the-worlds-fastest-host-for-deepseek-r1-outpacing-nvidia-gpus-by-57x/)：**Cerebras** 声称其晶圆级系统运行 **DeepSeek R1-70B** 的速度比 Nvidia GPU 快达 **57 倍**，挑战了 Nvidia 在 AI 硬件领域的统治地位。这一公告引发了关于替代性高性能 AI 托管方案及其对 GPU 市场影响的辩论，特别是在 **OpenRouter** 和 **GPU MODE** 等频道中。

**主题 3. Aider 和 Cursor 拥抱用于代码生成的新模型**

- **Aider v0.73.0 展示 O3 Mini 和 OpenRouter 的实力**：[Aider 0.73.0 发布说明详述 o3-mini 和 R1 支持](https://aider.chat/HISTORY.html)：**Aider v0.73.0** 首次支持 **o3-mini** 和 **OpenRouter** 的免费 **DeepSeek R1**，并新增了 **--reasoning-effort** 参数。用户称赞 **O3 Mini** 能以比 **O1** 更低的成本编写功能性的 Rust 代码，同时指出 **Aider** 自身编写了该版本 **69%** 的代码，展示了 AI 在软件开发工具中日益增长的作用。
- **Cursor IDE 将 DeepSeek R1 与 Sonnet 3.6 配对打造编程利器**：[Windsurf 推文吹捧 Cursor 中 R1 + Sonnet 3.6 的协同效应](https://x.com/windsurf_ai/status/1885077046663217230)：**Cursor IDE** 集成了 **DeepSeek R1** 进行推理并配合 **Sonnet 3.6** 进行编码，声称在 **aider polyglot benchmark** 中创下了新纪录。这种组合旨在提高解决方案质量并降低成本（相比 **O1**），在编码 Agent 性能方面树立了新标杆，正如在 **Cursor IDE** 和 **Aider** 频道中所讨论的那样。
- **Cursor 中的 MCP 工具：可用但功能饥渴**：[Cursor IDE 讨论中强调 MCP 服务器库](https://www.mcpservers.ai/)：**MCP (Model Context Protocol) 工具**在 **Cursor IDE** 中被确认为可用，但用户希望有更强的界面集成和更多突破性的功能。**Cursor IDE** 和 **MCP** 频道的讨论显示，社区渴望在编码工作流中更无缝、更强大地利用 MCP 工具，并参考了如 [HarshJ23 的 DeepSeek-Claude MCP 服务器](https://github.com/HarshJ23/deepseek-claude-MCP-server)等示例。

**主题 4. 本地 LLM 生态系统：LM Studio、GPT4All 和硬件之争**

- **LM Studio 0.3.9 通过 Idle TTL 实现内存优化**：[LM Studio 0.3.9 博客文章宣布 Idle TTL 及更多功能](https://lmstudio.ai/blog/lmstudio-v0.3.9)：**LM Studio 0.3.9** 引入了用于内存管理的 **Idle TTL**、运行时的**自动更新**以及对 Hugging Face 仓库的**嵌套文件夹**支持，增强了本地 LLM 的管理。用户发现独立的 **reasoning_content** 字段有助于 DeepSeek API 的兼容性，而 **Idle TTL** 因其高效的内存利用而受到欢迎，正如 **LM Studio** 频道中所强调的那样。
- **GPT4All 3.8.0 蒸馏 DeepSeek R1 与 Jinja 魔法**：[GPT4All v3.8.0 发布说明详述 DeepSeek 集成](https://github.com/nomic-ai/gpt4all/pull/3440)：**GPT4All v3.8.0** 集成了 **DeepSeek-R1-Distill**，使用 Jinja 彻底重构了聊天模板，并修复了代码解释器和本地服务器的问题。社区称赞了 **DeepSeek** 的集成，并注意到模板处理方面的改进，同时也指出了 **GPT4All** 频道中反馈的 **Mac 启动崩溃**问题，展示了活跃的开源开发和快速的社区反馈。
- **双 GPU 梦想在 LM Studio 中遭遇 VRAM 现实**：**LM Studio 的硬件讨论**显示用户正在尝试**双 GPU 设置**（NVIDIA RTX 4080 + Intel UHD），发现 NVIDIA 在 VRAM 满载后会卸载到系统 RAM。发烧友们成功实现了高达 **80k tokens** 的上下文，但挑战极限会使硬件承压并降低速度，凸显了当前硬件在极端上下文长度下的实际限制。

**主题 5. 批判性微调与思维链创新**

- **Critique Fine-Tuning 宣称比 SFT 提升 4-10%**：[Critique Fine-Tuning 论文承诺泛化增益](https://arxiv.org/abs/2501.17703)：**Critique Fine-Tuning (CFT)** 作为一种极具前景的技术脱颖而出，通过训练模型对噪声输出进行批判，声称比标准的 Supervised Fine-Tuning (**SFT**) 提升了 **4-10%**。**Eleuther** 频道的讨论辩论了 **CE-loss** 的有效性，并考虑直接奖励“胜出者”以改善训练结果，这标志着向更细致的训练方法论的转变。
- **非 Token CoT 和回溯向量重塑推理**：[Eleuther 讨论中探索的全非 Token CoT 概念](https://www.overleaf.com/read/krhxtvkxjywb#416acf)：一种新颖的**全非 Token 思维链 (CoT)** 方法为原始潜变量（raw latents）引入了 `<scratchpad>` token，对每个 prompt 的原始思维潜变量实施限制。研究人员还强调了一个影响 CoT 结构的正“回溯向量”，并使用 **sparse autoencoders** 证明了其效果，引发了 **Eleuther** 频道关于探测内部推理结构和为更广泛任务编辑向量的讨论。
- **Tülu 3 405B 在基准测试中挑战 GPT-4o 和 DeepSeek v3**：[Ai2 博客文章宣称 Tülu 3 405B 超越对手](https://allenai.org/blog/tulu-3-405B)：新发布的 **Tülu 3 405B** 模型声称在特定基准测试中优于 **DeepSeek v3** 和 **GPT-4o**，该模型采用了 **Reinforcement Learning from Verifiable Rewards**。然而，**Yannick Kilcher** 频道的社区审查质疑其对 **DeepSeek v3** 的实际领先地位，认为尽管采用了先进的 RL 方法，增益却有限，这促使人们对基准测试方法论和实际性能影响进行更深入的探讨。

---

# PART 1: High level Discord summaries

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Cascade 推出 DeepSeek R1 和 V3**：工程师们重点介绍了 [DeepSeek-R1 和 V3](https://x.com/windsurf_ai/status/1885077046663217230)，每次调用分别消耗 **0.5** 和 **0.25** 用户积分，承诺提升编程效率。
   - 他们还推出了新的 **o3-mini** 模型，消耗 **1** 用户积分，更多细节见 [Windsurf Editor Changelogs](https://codeium.com/changelog)。
- **DeepSeek R1 在压力下表现不佳**：用户报告称 **DeepSeek R1** 反复出现工具调用（tool call）失败和文件读取不完整的问题，降低了其在编程任务中的有效性。
   - 一些人建议回退到旧版本，因为最近的修订似乎降低了稳定性。
- **O3 Mini 引发褒贬不一的反应**：虽然有人称赞 **O3 Mini** 的代码响应速度更快，但也有人认为其工具调用处理能力太弱。
   - 一位参与者将其与 **Claude 3.5** 进行了比较，指出其在多步操作中的可靠性较低。
- **关于成本与产出的争论持续升温**：几位成员质疑 **DeepSeek** 等模型的费用，指出对于高级用户来说，本地设置可能更便宜。
   - 他们认为需要顶级的 GPU 才能获得可靠的本地部署（on-prem）输出，这加剧了关于性能与价格之争的讨论。
- **Windsurf 迎来 6K 社区里程碑**：Windsurf 的 Reddit 页面关注者突破了 **6k**，反映出用户参与度的提高。
   - 开发团队在[最近的推文](https://x.com/windsurf_ai/status/1885410914633130397)中进行了庆祝，并将这一里程碑与新公告联系在一起。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek R1 大胆的 1.58-Bit 技巧**：**DeepSeek R1** 现在可以以 1.58-bit 动态量化形式（671B 参数）运行，如 [Unsloth 关于 OpenWebUI 集成的文档](https://docs.openwebui.com/tutorials/integrations/deepseekr1-dynamic/)所述。
   - 在最低限度硬件上的社区测试凸显了一种高效但耗费资源的方法，许多人称赞 **Unsloth** 处理大规模本地推理的方法。
- **Quadro 上的 Qwen2.5：渴望退休的 GPU**：一位用户在仅有 5GB VRAM 的 **Quadro P2000** 上尝试运行 **Qwen2.5-0.5B-instruct**，开玩笑说它可能要到 2026 年才能运行完。
   - 关于 GPU *尖叫着要求休息*的评论突显了旧硬件的极限，但也指向了超越典型能力的理念验证。
- **双重麻烦：vLLM 和 Unsloth 中的 XGB 重叠**：讨论显示 **vLLM** 和 **Unsloth** 都依赖 **XGB**，存在重复加载和潜在资源过度使用的风险。
   - 成员们询问补丁是否能修复 **deepseek v2** 架构下 **gguf** 的 offloading 问题，并推测未来的兼容性改进。
- **微调壮举与多 GPU 等待名单**：**Unsloth** 用户辩论了微调大型 LLM 的学习率（e-5 vs e-6），引用了[官方检查点指南](https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint)。
   - 他们还对持续缺乏多 GPU 支持表示遗憾，指出 offloading 或额外的 VRAM 可能是短期内唯一的权宜之计。

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.73.0 推出新功能**：官方版本引入了对 **o3-mini** 的支持（使用 `aider --model o3-mini`），新增了 **--reasoning-effort** 参数，改进了上下文窗口（context window）处理，并支持自动目录创建，详见 [发布历史](https://aider.chat/HISTORY.html)。
   - 社区成员报告称，Aider 编写了该版本 **69%** 的代码，并欢迎 **OpenRouter** 上通过 `--model openrouter/deepseek/deepseek-r1:free` 提供的 **R1 free** 支持。
- **O3 Mini 抢占老牌模型风头**：早期采用者称赞 **O3 Mini** 能够生成功能完备的 Rust 代码，且成本远低于 **O1**，如 [TestingCatalog 的更新](https://x.com/testingcatalog/status/1885301385182237062) 所示。
   - 在看到 **O3 Mini** 在实际编程任务中交付快速结果并展现出可靠性能后，怀疑论者改变了立场。
- **DeepSeek 遇挫，用户寻求修复**：多位成员报告 **DeepSeek** 出现卡顿和空格处理错误，引发了对性能问题的反思。
   - 一些人考虑使用本地模型（local model）替代方案，并寻找在 **DeepSeek** 失效时保持代码稳定的方法。
- **Aider 配置获得社区见解**：贡献者报告通过设置环境变量而非仅依赖配置文件解决了 **API key** 检测问题，参考了 [高级模型设置](https://aider.chat/docs/config/adv-model-settings.html)。
   - 其他人对在保持聊天模式的同时通过文件脚本指挥 Aider 表示出兴趣，表明用户希望有更灵活的工作流选项。
- **Linting 和测试在 Aider 中盛行**：成员们强调了使用 [Aider 内置功能](https://aider.chat/docs/usage/lint-test.html) 实时自动进行 **lint** 和 **测试** 的能力，并以 Rust 项目作为演示。
   - 据称，这种设置能更快发现错误，并鼓励 **O3 Mini** 及其他集成模型输出更健壮的代码。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **O3 Mini 超越 O1**：成员们对 **O3 Mini** 的发布感到兴奋，尤其是其速度，参考了 [Kevin Lu 的推文](https://x.com/_kevinlu/status/1885406995613892711) 和 [OpenAI 的公告](https://x.com/OpenAI/status/1885406586136383634)。
   - 与 **O1** 和 **R1** 的对比突出了其在谜题解决方面的改进，而一些用户对 Perplexity 的模型管理以及仅在免费层提供的“Reason”按钮表示不满。
- **DeepSeek 泄露暴露聊天秘密**：安全研究人员发现了一个 [DeepSeek 数据库泄露](https://cybersecuritynews.com/deepseek-database-leaked/)，其中暴露了密钥（Secret keys）、日志和聊天历史。
   - 尽管许多人将 **DeepSeek** 视为 **O1** 或 **R1** 的替代方案，但此次泄露引发了对数据安全和未经授权访问的紧迫担忧。
- **AI 处方法案进入临床阶段**：一项提议的 [AI 处方法案](https://www.perplexity.ai/page/google-offers-voluntary-exit-f-tA7gBGbPSzymq8WBAwkTUw#93ca4910-afc1-4e9a-a30c-c219ffc1bb02) 旨在强制执行医疗保健 AI 的伦理标准和问责制。
   - 该立法解决了围绕 **医疗 AI** 监管的焦虑，反映了先进系统在患者护理中日益增长的作用。
- **纳德拉在 AI 领域的杰文斯悖论警告**：**Satya Nadella** 警告称，AI 创新可能会消耗更多资源而非缩减规模，这呼应了技术使用中的 **杰文斯悖论 (Jevons Paradox)**。
   - 他的观点引发了关于 **O3 Mini** 或 **DeepSeek** 等突破是否会引发算力需求激增的讨论。
- **Sonar Reasoning 停留在 80 年代**：一位用户注意到 **sonar reasoning** 调用的详情来自 **1982 年** 的波托马克河空难，而非最近发生的事故。
   - 这突显了在紧急查询中使用过时参考资料的风险，此时模型的历史准确性可能无法满足即时需求。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.9 势头强劲**：全新的 **LM Studio 0.3.9** 增加了 **Idle TTL**、运行时的 **auto-update** 以及 Hugging Face 仓库中的**嵌套文件夹**，详见 [博客](https://lmstudio.ai/blog/lmstudio-v0.3.9)。
   - 用户发现独立的 **reasoning_content** 字段在高级用法中非常方便，而 **Idle TTL** 通过自动卸载空闲模型来节省内存。
- **OpenAI 的 o3-mini 发布令用户困惑**：**OpenAI** 推出了用于数学和编程任务的 **o3-mini** 模型，参考 [The Verge 的报道](https://www.theverge.com/news/603849/openai-o3-mini-launch-chatgpt-api-available-now)。
   - 随后出现了一些混乱，因为部分用户无法免费访问，引发了关于实际可用性和使用限制的疑问。
- **DeepSeek 在代码领域表现优于 OpenAI**：工程师们称赞 **DeepSeek** 的速度和强大的编程能力，声称它在实际项目中挑战了 OpenAI 的付费产品。
   - **OpenAI** 的降价被归因于 **DeepSeek** 的进步，引发了关于本地模型取代云端模型的讨论。
- **Qwen2.5 证明了长上下文能力**：社区测试发现 **Qwen2.5-7B-Instruct-1M** 能平滑处理更大的输入，**Flash Attention** 和 K/V cache quantization 提升了效率。
   - 据报道，它在内存占用和准确性方面超过了旧模型，为处理海量文本集的开发者注入了活力。
- **双 GPU 构想与上下文过载**：爱好者尝试将 **NVIDIA RTX 4080** 与 **Intel UHD** 配对，但发现一旦 VRAM 满载，NVIDIA 就会将负载卸载到系统 RAM。
   - 有人成功实现了高达 **80k tokens** 的上下文，但过度推高上下文长度会使硬件承压并显著降低速度。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek R1 + Sonnet 3.6 协同效应**：他们将用于详细推理的 R1 与用于编程的 Sonnet 3.6 结合，提升了解决方案的质量。[Windsurf 的推文](https://x.com/windsurf_ai/status/1885077046663217230)提到了开放推理 Token 以及与编程 Agent 的协同。
   - 这种组合在 [aider polyglot benchmark](https://aider.chat/2025/01/24/r1-sonnet.html) 上创下了新纪录，在用户测试中成本低于 O1。
- **O3 Mini 评价褒贬不一**：一些用户发现 O3 Mini 在某些任务中很有帮助，但另一些人觉得它的性能落后于 Sonnet 3.6。讨论围绕着运行代码更改时需要明确 Prompt 的需求展开。
   - [Reddit 帖子](https://www.reddit.com/r/OpenAI/comments/1idzrdl/o3_releasing_tomorrow/) 强调了失望情绪以及对更新的猜测。
- **MCP 工具在 Cursor 中引发讨论**：许多人表示 MCP 工具运行良好，但在 Cursor 中需要更强大的界面，参考了 [MCP Servers 库](https://www.mcpservers.ai/)。
   - 一个例子是 [HarshJ23/deepseek-claude-MCP-server](https://github.com/HarshJ23/deepseek-claude-MCP-server)，它融合了 R1 推理与 Claude 以供桌面端使用。
- **Claude 模型：对下一版本的期待**：个人用户期待 Anthropic 的新发布，希望更先进的 Claude 版本能提升编程工作流。一篇 [博客文章](https://www.testingcatalog.com/anthropic-developing-web-search-feature-for-claude-ai/) 预告了 Claude 的网页搜索功能，弥补了静态 LLM 与实时数据之间的差距。
   - 社区讨论围绕功能扩展或命名的可能性展开，但官方消息尚待公布。
- **用户体验与安全警报**：某些参与者报告了新集成的基于 R1 的解决方案取得了成功，但其他人则面临响应时间慢和结果不一致的问题。
   - 与此同时，[JFrog 博客](https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models-with-silent-backdoor/) 提出了新的担忧，而对 [BitNet](https://github.com/microsoft/BitNet) 的引用表明了对 1-bit LLM 框架的兴趣。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **o3-mini 强势登场**：OpenAI 面向使用等级（usage tiers）3 到 5 的用户发布了 **o3-mini**，提供更敏锐的推理能力，在专家测试中比其前代产品获得了 [56% 的评分提升](https://openrouter.ai/openai/o3-mini)。
   - 该模型拥有 **39% 的重大错误减少率**，并为精通 STEM 的开发者内置了 function calling 和 structured outputs。
- **BYOK 或无缘：密钥访问要求**：OpenRouter 将 **o3-mini** 限制为等级 3 或更高的 **BYOK** 用户，但[这份快速入门指南](https://openrouter.ai/docs/quick-start)可帮助进行设置。
   - 他们还鼓励免费用户通过点击 ChatGPT 中的 **Reason** 按钮来测试 o3-mini。
- **模型之战：o1 对阵 DeepSeek R1 以及 GPT-4 的失落**：评论者辩论了 **o1** 和 **DeepSeek R1** 的性能，一些人称赞 R1 的写作风格优于 GPT-4 “令人失望”的表现。
   - 其他人则表达了对 GPT-4 的不满，并引用了关于模型局限性的 [Reddit 帖子](https://www.reddit.com/r/singularity/comments/1ie0sf4/the_o3_series_of_models_releases_tomorrow/)。
- **Cerebras 飞速发展：DeepSeek R1 超越 Nvidia**：根据 [VentureBeat](https://venturebeat.com/ai/cerebras-becomes-the-worlds-fastest-host-for-deepseek-r1-outpacing-nvidia-gpus-by-57x) 的报道，**Cerebras** 现在运行 **DeepSeek R1-70B** 的速度比 Nvidia GPU 快 57 倍。
   - 这种晶圆级系统挑战了 Nvidia 的主导地位，为大规模 AI 托管提供了一个高性能的替代方案。
- **AGI 之争：近在咫尺还是遥远的幻想？**：一些人坚持认为 **AGI** 可能近在眼前，回顾了早期激发 AI 潜力雄心的演示。
   - 其他人则保持怀疑，认为通往真正 AGI 的道路仍需要更深层次的突破。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI o3-mini 公开亮相**：OpenAI 推出了 **o3-mini** 系列，具有改进的推理和 function calling 能力，与旧模型相比具有成本优势，如[这条推文](https://x.com/OpenAIDevs/status/1885407759887155301)所述。
   - 社区讨论称赞 **o3-mini-high** 是目前最好的公开可用推理模型，引用了 [Kevin Lu 的帖子](https://x.com/_kevinlu/status/1885406995613892711)，而一些用户对 “LLM 抽卡式（gacha）”订阅模式表示不满。
- **DeepSeek 的十亿美元数据中心**：来自 [SemiAnalysis](https://semianalysis.com/2025/01/31/deepseek-debates/) 的新信息显示，**DeepSeek** 在 HPC 上投资了 **13 亿美元**，反驳了仅持有 50,000 张 H100 的传闻。
   - 社区成员在推理性能方面将 **R1** 与 **o1** 进行了比较，强调了对 chain-of-thought 协同效应和巨大基础设施成本的关注。
- **Mistral 的巨大惊喜**：尽管筹集了 **14 亿美元**，**Mistral** 还是发布了一个小型和一个较大型的模型，包括一个 24B 参数版本，令观察者感到惊讶。
   - 聊天记录引用了 [MistralAI 的发布](https://x.com/nrehiew_/status/1885188206485733548)，称赞了较小模型的效率，并开玩笑说“小”的真正定义。
- **K2 Chat 榜上有名**：**LLM360** 发布了一个名为 **K2 Chat** 的 **65B** 模型，声称比 **Llama 2 70B** 减少了 35% 的计算量，如 [Hugging Face](https://huggingface.co/LLM360/K2-Chat) 所示。
   - 该模型于 **2024 年 10 月 31 日**推出，支持 function calling 并使用 [Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)，引发了更多正面交锋的基准测试。
- **Altman 的宇宙级 Stargate 支票**：根据 [OpenAI 的声明](https://openai.com/index/announcing-the-stargate-project/)，**Sam Altman** 宣布了由唐纳德·特朗普支持的 **5000 亿美元** Stargate 项目。
   - 批评者质疑巨额预算，但 Altman 认为这对于扩展超智能 AI 至关重要，引发了关于市场主导地位的辩论。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **O3 Mini 令人困惑的配额**：新发布的 **O3 Mini** 设置了每日 **150** 条的消息配额，但一些参考资料指向每周 **50** 条，[用户正在讨论这种不匹配](https://discord.com/channels/974519864045756446/998381918976479273/1334616772950626396)。
   - 某些声音怀疑这是一个 **bug**，并评论道 *“之前从未正式提到过 50 条消息的限制”*，这引发了早期采用者的担忧。
- **AMA 预告：Sam Altman 及其团队**：即将于 **PST 时间下午 2 点**举行的 [Reddit AMA](https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_mark_chen_kevin_weil/) 将邀请 **Sam Altman**、**Mark Chen** 和 **Kevin Weil**，重点关注 **OpenAI o3-mini** 和 AI 的未来。
   - 社区反响热烈，诸如 *“在这里提出你的问题！”* 之类的邀请提供了与这些关键人物直接互动的机会。
- **DeepSeek 跃入竞争焦点**：用户认可 **DeepSeek R1** 在编程任务中的表现，并将其与主要供应商进行了积极对比，引用了 [AI 军备竞赛中的报道](https://www.tomsguide.com/ai/it-doesnt-matter-if-deepseek-copied-openai-the-damage-has-already-been-done-in-the-ai-arms-race)。
   - 他们赞扬了开源方法在匹配大厂性能方面的表现，认为 **DeepSeek** 可能会推动更多小型社区驱动模型的采用。
- **Vision 模型在地面线条识别上失误**：开发者发现 **Vision 模型**在区分地面和线条时遇到困难，[一个月前的日志](https://discord.com/channels/974519864045756446/1046317269069864970/1334631185112109167)表明需要进一步改进。
   - 一位测试者将其比作 *“需要配副新眼镜”*，并强调了隐藏的 **training data** 缺口，这些缺口随着时间的推移可能会修复这些缺陷。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **O3 Mini 正式发布**：OpenAI 的 [O3 Mini](https://cdn.openai.com/o3-mini-system-card.pdf) 发布，为 API 级别 3–5 提供 **function calling** 和 **structured outputs**，在 ChatGPT 中也可免费使用。
   - 出现了新的 [价格更新](https://x.com/swyx/status/1885432031896887335) 参考，尽管 O1 降价了 **63%**，但 O3 Mini 仍以相同的费率推出，突显了竞争的加剧。
- **Sonnet 在代码测试中表现优于 O3 Mini**：多份报告描述 **O3 Mini** 在编程提示词上未能达标，而 [Sonnet 的最新迭代](https://x.com/angelusm0rt1s/status/1884734909685915764?s=46)处理任务更加敏捷。
   - 用户强调了 Sonnet 中 **更快的错误发现能力**，并讨论了 O3 Mini 是否能通过有针对性的 fine-tuning 赶上。
- **DeepSeek 引发价格战**：在 O3 Mini 发布的消息中，**O1 Mini** 进行了 **63% 的降价**，这似乎是受 [DeepSeek 日益增长的影响力](https://semianalysis.com/2025/01/31/deepseek-debates/)所推动。
   - 爱好者们注意到 AI 领域持续存在的 **“美国溢价 (USA premium)”**，表明 DeepSeek 成功挑战了传统的成本模型。
- **开源 AI 工具和辅导计划**：社区成员推崇 **Cline** 和 **Roocline** 等新兴开源工具，重点介绍了付费解决方案的潜在替代方案。
   - 他们还讨论了一个拟议的 **AI 辅导** 课程，该课程借鉴了 boot_camp.ai 等项目，希望通过集体知识为新手赋能。
- **DeepSeek API 引发不满**：反复的 API key 故障和连接问题困扰着尝试将 **DeepSeek** 用于生产需求的用户。
   - 成员们权衡了备选策略，对依赖一个以稳定性问题著称的 API 表示谨慎。



---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **OpenAI 的 o3-mini 取得进展**：OpenAI 在 [ChatGPT](https://fxtwitter.com/OpenAI/status/1885406586136383634) 和 API 中推出了 **o3-mini**，为 **Pro 用户**提供无限访问权限，为 **Plus & Team 用户**提供三倍的速率限制，并允许免费用户通过选择 Reason 按钮进行尝试。
   - 成员们报告称**推送进度较慢**，欧盟的一些用户看到激活较晚，参考了 [Parker Rex 的推文](https://fxtwitter.com/ParkerRex/status/1884978010744320377)。
- **FP4 论文为正式发布做准备**：社区将研究一种 [FP4 技术](https://arxiv.org/abs/2501.17116)，该技术通过改进 **QKV** 处理来解决量化误差，从而承诺提高训练效率。
   - 参与者计划提前温习 **QKV** 基础知识，预见到关于其对大型模型准确性的实际影响会有更深入的提问。
- **Tülu 3 巨头挑战 GPT-4o**：新发布的 **Tülu 3 405B** 模型声称在特定基准测试中超越了 **DeepSeek v3** 和 **GPT-4o**，[Ai2 的博客文章](https://allenai.org/blog/tulu-3-405B)证实了这一点。
   - 一些参与者质疑其对 **DeepSeek v3** 的实际领先地位，指出尽管采用了 **Reinforcement Learning from Verifiable Rewards** 方法，但收益有限。
- **低成本克隆 DeepSeek R1**：由 **Jiayi Pan** 领导的 Berkeley AI 研究小组以低于 **$30** 的成本，在 **1.5B parameters** 规模下复制了 **DeepSeek R1-Zero** 的复杂推理能力，详见这篇 [substack 文章](https://xyzlabs.substack.com/p/berkeley-researchers-replicate-deepseek)。
   - 这一成就引发了关于**低成本实验**的辩论，许多声音都在庆祝推动 **democratized AI**（AI 民主化）。
- **Qwen 2.5VL 获得敏锐洞察**：转向 **Qwen 2.5VL** 产生了更强的描述能力和对相关特征的关注，提高了网格变换中的模式识别能力。
   - 成员们报告称其在连贯性方面优于 **Llama**，并注意到它在变换过程中更加注重**保持原始数据**。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Psyche 项目助力去中心化训练**：在 #general 频道中，**Psyche 项目**旨在协调全球闲置硬件的不可信算力进行去中心化训练，参考了这篇关于[分布式 LLM 训练的论文](https://arxiv.org/abs/2501.18512)。
   - 成员们辩论了使用 **blockchain** 进行验证与更简单的基于服务器的方法，一些人引用了 [Teknium 关于 Psyche 的帖子](https://x.com/Teknium1/status/1884740956911718853)，认为这是一个充满希望的方向。
- **加密货币难题困扰 Nous**：#general 中的一些人质疑 **crypto** 的联系是否会招致诈骗，而另一些人则认为成熟的 **blockchain** 技术可能有利于分布式训练。
   - 参与者将不道德的加密货币骗局与公开股票中的阴暗行为进行了比较，结论是对 **blockchain** 持谨慎但开放的态度是合适的。
- **o3-mini 对阵 Sonnet：意外对决**：在 #general 中，开发者承认 **o3-mini** 在复杂任务上的强劲表现，引用了 [Cursor 的推文](https://x.com/cursor_ai/status/1885415392677675337)。
   - 他们称赞其流式传输速度更快，编译错误比 **Sonnet** 更少，但仍有一些人因操作清晰度而忠于旧的 **R1** 模型。
- **CLIP 的自回归冒险**：在 #ask-about-llms 中，一位用户询问在 **CLIP embeddings** 上进行 **autoregressive generation** 是否可行，并指出 **CLIP** 通常用于引导 **Stable Diffusion**。
   - 对话提议直接从 **CLIP** 的 **latent space** 生成，尽管参与者观察到除了多模态任务之外，几乎没有记录在案的探索。
- **DeepSeek 颠覆招聘教条**：在 2023 年的一次采访中，**Liang Wenfeng** 声称经验无关紧要，参考了[这篇文章](https://archive.ph/KvXp0)。
   - 他支持**创造力**而非简历，但承认从大型 AI 公司招聘的人员可以帮助实现短期目标。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **无显著 AI 或融资公告 #1**：提供的日志中没有出现重大的新 AI 进展或融资公告。
   - 所有提到的细节仅围绕 **Supabase**、**Bolt** 和 **CORS** 的常规调试和配置。
- **无显著 AI 或融资公告 #2**：对话集中在关于 **token** 管理、身份验证和项目删除问题的常规故障排除。
   - 除了标准使用指南外，没有提到新模型、数据发布或高级研究。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 设置提速**：成员们讨论了本地与远程 MCP 服务器的对比，并引用 [mcp-cli 工具](https://github.com/wong2/mcp-cli) 来解决困惑。
   - 他们强调 **authentication**（身份验证）对远程部署至关重要，并呼吁提供更易用的文档。
- **传输协议对决**：一些人称赞 **stdio** 的简洁性，但指出标准配置缺乏加密。
   - 他们权衡了 **SSE** 与 **HTTP POST** 的性能，并建议探索替代传输方式以增强安全性。
- **Toolbase 认证在 YouTube 演示中亮相**：一位开发者在 [YouTube 演示](https://www.youtube.com/watch?v=UuUxG_2K2Bs)中展示了 Claude 的 Toolbase 中的 **Notion**、**Slack** 和 **GitHub** 身份验证。
   - 观众建议调整 **YouTube playback** 或使用 *ffmpeg* 命令来优化观看体验。
- **日志记录 MCP 服务器保存对话**：一位成员介绍了一个用于记录与 Claude 对话日志的 **MCP server**，分享地址为 [GitHub - mtct/journaling_mcp](https://github.com/mtct/journaling_mcp)。
   - 他们计划添加本地 LLM 以提高隐私性并实现设备端对话存档。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **50 系列 GPU 瞬间售罄**：新发布的 **50 Series GPUs** 在几分钟内从货架上消失，据报道整个北美地区仅出货了几千块。
   - 一位用户差点买到 **5090**，但在商店崩溃时错失了机会，如[此截图](https://prnt.sc/OwXsJqnPDDvn)所示。
- **性能思考：5090 对比 3060**：成员们将 **5090** 与 **3060** 等旧显卡进行了对比，重点关注游戏基准测试和 VR 潜力。
   - 几位用户对极低的库存表示失望，同时仍在权衡新系列是否真的超越了中端 GPU。
- **手机与 AI 的博弈**：关于在 Android 上运行 **Flux** 的辩论爆发了，一位用户计算出生成结果需要 22.3 分钟。
   - 一些人称赞手机用于小型任务，而另一些人则强调了减慢 AI 工作负载的硬件限制。
- **AI 平台与工具兴起**：成员们讨论了用于本地 AI 图像生成的 **Webui Forge**，并建议使用专门的模型来优化输出。
   - 他们强调为每个平台匹配正确的模型，以获得最佳的 **Stable Diffusion** 性能。
- **Stable Diffusion UI 大变动**：用户想知道 **Stable Diffusion 3.5** 是否强制切换到 **ComfyUI**，因为他们怀念旧版布局。
   - 他们承认对 UI 一致性的渴望，但尽管有学习曲线，仍对增量改进表示欢迎。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **批判微调：优于 SFT**：批判微调（**CFT**）声称通过训练模型批判嘈杂输出，比标准监督微调（[SFT](https://arxiv.org/abs/2501.17703)）提升了 **4–10%**，在多个基准测试中表现出更强的结果。
   - 社区讨论了 **CE-loss** 指标是否足够，并建议直接奖励“胜者”以获得更好的结果。
- **完全非 Token CoT 结合 <scratchpad>**：一种新的**完全非 Token Chain of Thought** 方法为原始潜变量（raw latents）引入了 `<scratchpad>` token，并对每个 prompt 的原始思维潜变量施加了强制限制，详情见此 [Overleaf 链接](https://www.overleaf.com/read/krhxtvkxjywb#416acf)。
   - 贡献者看到了直接进行**行为探测（behavioral probing）**的潜力，并指出原始潜变量可能揭示内部推理结构。
- **回溯向量：反向实现更好的推理**：研究人员强调了一种改变 **chain of thought** 结构的“回溯向量”，见 [Chris Barber 的推文](https://fxtwitter.com/chrisbarber/status/1885047105741611507)。
   - 他们利用 **sparse autoencoders** 展示了切换该向量如何影响推理步骤，并提议未来针对更广泛的任务编辑这些向量。
- **gsm8k 基准测试困惑**：成员们报告了 **gsm8k** 准确率的不匹配（0.0334 对比 0.1251），其中 `gsm8k_cot_llama.yaml` 与 **Llama 2** 论文中记录的结果存在偏差。
   - 他们怀疑差异源于测试框架（harness）设置，建议手动调整 **max_new_length** 以匹配 Llama 2 报告的指标。
- **随机顺序 AR 模型引发好奇**：参与者研究了**随机顺序自回归（random order autoregressive）**模型，承认它们可能不切实际，但可以揭示训练的结构性方面。
   - 他们观察到小数据集中的过度参数化网络可能会捕捉到模式，尽管实际应用价值仍有待商榷。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Deep Seek HPC 困境与质疑**：技术爱好者对 **Deep Seek** 声称在 HPC 中使用 **50k H100s** 的说法提出挑战，并引用了 [SemiAnalysis expansions](https://semianalysis.com/2025/01/31/deepseek-debates/) 质疑其官方声明。
   - 一些人担心 **Nvidia** 的股价是否会受到这些言论的影响，社区成员对 Deep Seek 突破背后的“真实成本”表示怀疑。
- **GPU 服务器 vs. 笔记本电脑大对决**：一位软件架构师在权衡是为 HPC 开发购买一台 **GPU server** 还是四台 GPU 笔记本电脑，并参考了 [The Best GPUs for Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) 指南。
   - 其他人强调了中心化服务器在“面向未来”方面的优势，但也指出了 HPC 扩展在前期成本上的差异。
- **RTX 5090 与 FP4 的困惑**：用户报告称 **RTX 5090** 上的 **FP4** 运行速度仅比 4090 上的 **FP8** 快约 2 倍，这与官方资料中声称的 5 倍不符。
   - 怀疑者将其归咎于“不清晰的文档”，并指出可能存在的内存开销，呼吁提供更好的 HPC 基准测试。
- **Reasoning Gym 新增数据集**：贡献者为 **Collaborative Problem-Solving** 和 **Ethical Reasoning** 投放了数据集，参考了 [NousResearch/Open-Reasoning-Tasks](https://github.com/NousResearch/Open-Reasoning-Tasks) 和其他 GitHub 项目以扩展 HPC 模拟。
   - 他们还讨论了添加 **Z3Py** 来处理约束，维护者建议提交针对 HPC 友好模块的 pull requests。
- **NVIDIA GTC 40% 折扣盛惠**：**Nvidia** 宣布使用代码 **GPUMODE** 注册 **GTC** 可享受 **40% 折扣**，这是参加 HPC 专题会议的好机会。
   - 对于 **GPU** 专业人士来说，这次活动仍然是交流见解和提升 HPC 技能的首选场所。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 3.8.0 发布，集成 DeepSeek 特性**：Nomic AI 发布了 **GPT4All v3.8.0**，完全集成了 **DeepSeek-R1-Distill**，带来了更好的性能，并解决了之前 **DeepSeek-R1 Qwen pretokenizer** 的加载问题。此次更新还彻底重构了 **chat template parser**，扩大了对各种模型的兼容性。
   - 来自 [主仓库](https://github.com/nomic-ai/gpt4all/pull/3440) 的贡献者强调了对 code interpreter 和本地服务器的重要修复，并对 **Jared Van Bortel**、**Adam Treat** 和 *ThiloteE* 表示感谢。他们确认系统消息现在会从消息日志中隐藏，以防止 UI 混乱。
- **量化特性引发好奇**：社区成员讨论了 **K-quants** 和 **i-quants** 之间的区别，参考了 [Reddit 概览](https://www.reddit.com/r/LocalLLaMA/comments/1ba55rj/overview_of_gguf_quantization_methods/)。他们得出结论，每种方法都适合特定的硬件需求，并建议针对性使用以获得最佳效果。
   - 一位用户通过 [GitHub Issue #3448](https://github.com/nomic-ai/gpt4all/issues/3448) 反馈了 **GPT4All 3.8.0** 在 **Mac 上启动崩溃**的问题，可能与 **Qt 6.5.1** 升级到 **6.8.1** 的更改有关。其他人建议回滚或等待官方修复，并指出平台正在积极开发中。
- **GPT4All 暂不支持语音分析**：有用户询问关于分析语音相似性的问题，但已确认 **GPT4All** 缺乏语音模型支持。社区成员建议使用外部工具来处理高级语音相似性任务。
   - 一些参与者希望未来能提供支持，而另一些人则认为专门的第三方库仍是短期内的最佳选择。目前没有直接提到 GPT4All 即将推出语音功能。
- **Jinja 技巧扩展模板功能**：关于 **GPT4All** 模板的讨论展示了新的 **namespaces** 和 **list slicing**，参考了 [Jinja 官方文档](https://jinja.palletsprojects.com/en/stable/templates/)。此更改旨在减少解析器冲突，并简化复杂模板的用户体验。
   - 开发者指出了 [google/minja 中的 minja.hpp](https://github.com/google/minja/blob/76f0d01779aa00b0c68f2117f6cb2c9afc3a0ca8/include/minja/minja.hpp#L2486-L2810) 以实现更轻量级的 Jinja 集成方案，同时更新了 [GPT4All 文档](https://docs.gpt4all.io/gpt4all_desktop/chat_templates.html#advanced-what-are-gpt4all-v1-templates)。他们注意到 *GPT4All v3.8* 的稳定性有所提高，这归功于开源社区的快速合并。

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 动态：75 美元奖励与远程访谈**：2025 年 2 月 6 日，NotebookLM UXR 邀请用户参与远程可用性研究，提供 **75 美元**或 **Google 商品代金券**以收集直接的用户反馈。
   - 参与者必须通过[筛选表单](https://forms.gle/HJmCwNepsfPSdC7g7)，保持高速互联网连接，并在在线会议中分享见解，以指导即将推出的产品更新。
- **短小精悍：将播客限制在一分钟**：社区成员讨论了将播客压缩至**一分钟**片段的想法，但承认很难严格执行。
   - 一些人建议通过修剪文本输入作为权宜之计，这引发了关于短内容在处理详细话题时实用性的辩论。
- **叙述之声：用户渴望 AI 配音**：多位参与者寻求 AI 驱动的叙述功能，能够精确阅读脚本，以实现更真实的单人主持演示。
   - 其他人警告说这可能与 NotebookLM 更广泛的平台目标相冲突，但对文本转音频（text-to-audio）扩展的热情依然高涨。
- **Workspace 烦恼：NotebookLM Plus 集成困惑**：一位用户升级到了标准的 Google Workspace 计划，但未能访问 **NotebookLM Plus**，误以为不需要额外的插件许可。
   - 社区回复指向了一个故障排除检查列表，反映出 NotebookLM 入职流程中的说明不够清晰。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **BF16 缓解 GRPO 难题**：成员们发现在使用 **GRPO** 时会出现 **显存溢出 (OOM)** 错误，归咎于 **fp32** 中不匹配的内存管理，并引用了 [Torchtune 仓库](https://github.com/RedTachyon/torchtune) 作为参考。切换到 **bf16** 解决了一些问题，展示了在资源利用率方面的显著改进，以及与 **vLLM** 进行推理时的协同效应。
   - 他们使用了当前 PPO 方案中的 **profiler** 来可视化内存需求，一位参与者强调对于大型任务，*“bf16 比全量 fp32 是更稳妥的选择”*。他们还讨论了在 GRPO 中并行化推理，但在 Hugging Face 生态系统之外面临复杂性。
- **梯度累积故障引发开发者警惕**：一个围绕 **梯度累积 (Gradient Accumulation)** 的已知[问题](https://github.com/unslothai/trl/issues/2175)浮出水面，该问题会干扰 **DPO** 和 **PPO** 模型的训练，导致损失跟踪不完整。对 [Unsloth 修复方案](https://unsloth.ai/blog/gradient) 的引用提出了一种减轻累积期间内存缺陷的方法。
   - 一些人推测这些**累积错误**会影响高级优化器，引发了*“对大批量训练中结果一致性的担忧”*。开发者对合并稳健的修复方案保持警惕，特别是针对大规模训练中的多步更新。
- **DPO 的零损失冲击**：异常情况导致 **DPO** 迅速下降至 **0 损失** 和 **100% 准确率**，这记录在[一个拉取请求评论](https://github.com/pytorch/torchtune/pull/2275#issuecomment-2623298923)中。这种诡异行为在寥寥几步内就会出现，指向了归一化例程中的疏忽。
   - 参与者辩论是否*“应该以不同方式缩放目标函数”*以避免立即收敛。他们得出结论，确保对非填充（non-padding）标记进行精确的**损失归一化**可能会恢复可靠的指标。
- **Torchtune 的多节点进军**：开发者推动对 **Torchtune** 中[多节点支持](https://github.com/pytorch/torchtune/pull/2301)的最终批准，旨在扩展分布式训练能力。此次更新承诺为大规模 LLM 训练提供更广泛的使用场景，并提高在 HPC 环境中的性能。
   - 他们质疑了 **offload_ops_to_cpu** 在多线程中的作用，在合并前进行了额外澄清。对话强调*“我们需要全力以赴以保证稳定的多节点运行”*，以确保可靠性。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **异构硬件上的 HPC 奔忙**：一系列博客文章将 **Mojo** 吹捧为解决 HPC 资源挑战的语言，并引用了 [Modular: Democratizing Compute Part 1](https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact-on-ai) 以强调**硬件利用率**可以降低 GPU 成本的观点。
   - 社区成员强调了库之间**向后兼容性**的重要性，以维持用户满意度并确保 HPC 的平稳过渡。
- **DeepSeek 挑战传统计算假设**：成员们讨论了 **DeepSeek** 如何动摇 AI 计算需求，认为改进的**硬件优化**可能会使大规模基础设施变得不再那么关键。
   - 大厂（Big Tech）被描述为正争先恐后地效仿 DeepSeek 的壮举，而一些人则抵制“小规模解决方案可能就足够了”这一观点。
- **Mojo 1.0 的等待是值得的**：贡献者支持推迟 **Mojo 1.0** 的发布，以便在更大的集群上进行基准测试，从而确保在微型测试之外赢得广泛的社区信心。
   - 他们赞扬了在版本化之前对**稳定性**的关注，将性能置于仓促发布之上。
- **Swift 的异步难题激发了简化希望**：Mojo 的设计者注意到 **Swift** 可能会使异步代码复杂化，这激发了将 Mojo 推向更简单方向的愿望。
   - 一些用户阐述了 Swift 方法中的陷阱，影响了 **Mojo** 开发中对清晰度的广泛追求。
- **MAX 让 DeepSeek 部署更直接**：通过一个简单的命令 `magic run serve --huggingface-repo-id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --weight-path=unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf`，只要用户准备好 Ollama 的 gguf 文件，就可以使用 **MAX** 运行 **DeepSeek**。
   - 最近的 [论坛指南](https://forum.modular.com/t/how-to-convert-numpy-array-items-to-mojo-float/506) 和 [GitHub issue](https://github.com/modular/max/issues/289) 展示了 MAX 如何通过演进来改进像 DeepSeek 这样的模型集成。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **三方对谈预告：Arize 与 Groq 联手**：与会者参加了与 [Arize AI](https://twitter.com/arizeai) 和 [Groq](https://twitter.com/GroqInc) 的见面会，讨论了 Agent 和追踪（tracing），并由 **Phoenix by Arize** 进行了现场演示。
   - 该会议重点展示了 **LlamaIndex** 的 Agent 能力，从基础的 RAG 到高级操作，详见 [Twitter 线程](https://twitter.com/llama_index/status/1885106917707833763)。
- **LlamaReport Beta 版展望 2025**：**LlamaReport** 的预览展示了一个早期的 Beta 构建版本，核心重点是为 **2025** 年生成报告。
   - [视频演示](https://twitter.com/llama_index/status/1885420164893860097) 展示了其核心功能并预告了即将推出的特性。
- **o3-mini 获得 Day 0 支持**：**o3-mini 的 Day 0 支持**已上线，用户可以通过 `pip install -U llama-index-llms-openai` 进行安装。
   - [Twitter 公告](https://twitter.com/llama_index/status/1885426718506442832) 展示了如何快速上手，强调了简单的设置流程。
- **OpenAI O1 引发困惑**：**OpenAI O1** 缺乏完整功能，导致社区对流式传输（streaming）特性和可靠性感到不确定。
   - 成员们在 [OpenAI 论坛参考](https://community.openai.com/t/streaming-support-for-o1-o1-2024-12-17-resulting-in-400-unsupported-value/1085043) 中指出了“奇怪”的流式传输问题，某些功能未能按预期工作。
- **LlamaReport 与支付查询隐忧**：用户在使用 **LlamaReport** 时遇到困难，理由是生成输出存在困难以及关于 LLM 集成费用的疑问。
   - 尽管在上传论文进行摘要方面取得了一些成功，但许多人指出 **Llama-Parse** 的费用可能是一个障碍，并指出它在*某些条件下可能是免费的*。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **物理托管，真实收益**：一位用户询问了用于企业任务本地 **LLM 托管**的物理服务器，并引用了来自 **Exolabs** 的 **Mac Minis** 作为经过测试的解决方案。
   - 他们还讨论了大规模运行大型模型的问题，引发了关于 AI 工作负载硬件方法的简短讨论。
- **Tinygrad 的 Kernel 调整与标题调侃**：George Hotz 称赞了 [tinygrad](https://x.com/__tinygrad__/status/1885291485433839729) 的一个“优秀的第一个 PR”，该 PR 优化了 **kernel**、buffer 和启动维度（launch dimensions）。
   - 他建议从 `DEFINE_LOCAL` 中移除 **16** 以避免重复，小组还调侃了一个很快被修复的 PR 标题小拼写错误。

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **Axolotl 转向 bf16 和 8bit LoRa**：参与者确认 **Axolotl** 支持 **bf16** 作为除 **fp32** 之外的稳定训练精度，一些人还注意到 [Axolotl 仓库](https://github.com/OpenAccess-AI-Collective/axolotl) 中 **8bit LoRa** 的潜力。
   - 他们发现 **bf16** 对于长时间运行特别可靠，尽管 **8bit fft** 能力仍不明确，引发了关于训练效率的进一步讨论。
- **Axolotl 中的 Fp8 尝试与困境**：成员表示 **fp8** 在 **accelerate** 中有实验性支持，但在实践中性能表现不一。
   - 有人表示 *“我目前不认为我们在研究那个”*，强调了与 **fp8** 相关的**不稳定**结果，并重申了持续的保留意见。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **证书待定，热度不减**：**LLM Agents MOOC** 的证书尚未发放，预计很快会公布更多关于要求的细节。
   - 成员们感叹 *“等待证书的过程太令人兴奋了！”*，并期待官方更新。
- **Quiz 1 与课程大纲的小插曲**：**Quiz 1** 现在可以在课程大纲页面访问，并提到 [Quizzes Archive - LLM Agents MOOC](https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit) 包含隐藏的正确答案。
   - 一些人发现链接缺失或不清晰，促使其他人分享截图并揭示大纲中的*“神秘内容”*。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **AI 工具预告吸引初学者**：爱好者要求对该 AI 工具进行简单直接的解释，寻求对其即将推出的功能的清晰认识。
   - 他们专注于为新手提供更直接的方法，强调需要更简单的术语和实际用例。
- **Farm Friend 应用热度上升**：社区成员展示了新的 [Farm Friend 应用程序](https://farm-friend-v1.teplit.app)，强调了其在生态系统中的桌面集成。
   - 他们承诺分享后续资源，并预告了更多即将推出的项目以扩展基础设施。
- **iOS Shortcuts Patreon 出现**：一位用户宣布了一个 **Patreon**，将提供不同级别的进阶 **iOS Shortcuts**，包括对 **agentic** 功能的支持。
   - 他们对回归分享过去一年的技术表示热忱，并暗示会有更多深入的内容。
- **NVIDIA NIM 与 DeepSeek 联动**：一位社区成员探索将 **NVIDIA NIM** 接入 **DeepSeek**，以便与 **open interpreter** 直接连接。
   - 他们征求了关于桥接这些组件的技术建议，寻求关于安装和协同作用的见解。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 的 422 难题**：一位用户在使用带有有效参数的 [Cohere **Embed API v2.0**](https://docs.cohere.com/reference/embed) 时遇到了 **HTTP 422 Unprocessable Entity** 错误，提示需要仔细检查请求格式。
   - 他们分享了官方文档作为参考，并报告没有立即的修复方案，暗示可能是 **payload** 结构有问题。
- **跨语言 Embedding 的热情**：同一位用户希望使用 **embed-multilingual-v3.0** 模型来研究跨语言极化，指向了 [Cohere/wikipedia-2023-11-embed-multilingual-v3](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3) 数据集。
   - 他们询问了关于预处理杂乱、冗长文本的问题，旨在研究中获得更稳健的多语言 **embeddings**。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **无重大更新**：对话中未出现值得注意的讨论或新的技术细节。
   - 仅有一些提及和简短的“Ty”，没有为 **AI** 工程师提供进一步的背景信息。
- **缺乏技术内容**：交流中未涉及新的工具、模型或数据集。
   - 这使得没有额外的见解或资源可供报告。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **没有 `http_client`？在 dspy.LM 中没问题！**: 成员们发现 **dspy.LM** 中缺少 `http_client` 参数，这在配置自定义 SSL 或代理时引发了困惑。
   - 他们参考了 `gpt3.py` 中使用的 `http_client: Optional[httpx.Client] = None`，建议为 **dspy.LM** 增加类似功能。
- **dspy.LM 的自定义客户端引起关注**: 开发者们询问如何在 **dspy.LM** 中复制 **gpt3.py** 的自定义客户端设置，以满足高级网络需求。
   - 他们提议借鉴 **OpenAI** 和 `gpt3.py` 的代码作为参考模型，鼓励在 dspy 架构内进行进一步实验。



---


**MLOps @Chipro Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---


**HuggingFace Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1334636642513715272)** (2 messages): 

> `Cascade 更新、新模型、网页与文档搜索、用户里程碑` 


- **Cascade 迎来新模型**: 最新更新引入了新模型：**DeepSeek-R1** 和 **DeepSeek-V3**，每条消息分别消耗 **0.5** 和 **0.25** 个用户提示词额度。
   - 此外，新的 **o3-mini** 模型也已上线，每条消息消耗 **1** 个用户提示词额度，进一步扩展了 Cascade 的能力。
- **用户体验改进**: 修复包括减少 Cascade 对话的输入延迟，并解决了重新加载时 Cascade 面板意外重新打开的 Bug。
   - **@docs** 功能得到增强，提供了更多选项，提高了工具内信息的获取效率。
- **网页和文档搜索能力**: Cascade 现在具备网页搜索功能，允许用户自动触发搜索或通过 **@web** 命令及提供 URL 来获取上下文。
   - 这些功能可以通过设置面板进行管理，方便用户获取实时信息。
- **达成用户参与里程碑**: 在一个显著的社区里程碑中，Windsurf 的 Reddit 页面达到 **6k** 关注者，展示了不断增长的参与度和兴趣。
   - 这一成就已在更新公告中热烈庆祝！


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/windsurf_ai/status/1885410914633130397">Windsurf (@windsurf_ai) 的推文</a>: o3-mini 现已在 Windsurf 中可用！</li><li><a href="https://codeium.com/changelog">Windsurf 编辑器变更日志 | Windsurf 编辑器和 Codeium 扩展</a>: Windsurf 编辑器的最新更新和变更。</li><li><a href="https://x.com/windsurf_ai/status/1885077046663217230">Windsurf (@windsurf_ai) 的推文</a>: DeepSeek R1 和 V3 现已在 Windsurf 中可用，完全托管在西方服务器上。我们在 R1 中实现了工具调用，使其首次能够用于编程 Agent。</li><li><a href="https://www.codeium.com/changelog">Windsurf 编辑器变更日志 | Windsurf 编辑器和 Codeium 扩展</a>: Windsurf 编辑器的最新更新和变更。
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1334637363321901146)** (329 条消息🔥🔥): 

> `DeepSeek R1 问题、Cascade 工具调用错误、Windsurf 使用、模型性能对比、OpenAI 与数据监管` 


- **DeepSeek R1 在执行任务时遇到困难**：用户报告称，尽管 DeepSeek R1 具备推理能力，但往往缺乏有效执行任务的能力。
   - 据指出，即使被要求读取指定文件夹中的特定文件，R1 提供的有效输出也极少。
- **Cascade 出现内部错误**：多位用户遇到了重复的错误提示“模型产生了无效的工具调用（The model produced an invalid tool call）”，随后是 Cascade 内部错误。
   - 讨论指出，这可能是一个影响多个模型性能的更广泛问题，表明需要进行修复。
- **Windsurf 相关问题的引导**：建议用户将 Windsurf 相关的咨询引导至特定频道，以保持讨论的有序性。
   - 这一点被多次重申，以确保讨论社区内的信息流转正常且焦点集中。
- **对 AI 模型成本和性能的担忧**：一位用户在对比本地替代方案时，表达了对使用 DeepSeek 等 AI 模型所产生的高昂成本的担忧。
   - 讨论中提到了运行本地模型并提供高效输出所需的对高性能硬件的需求。
- **OpenAI 与市场竞争反馈**：参与者分享了对 OpenAI 竞争行为及其对 AI 领域开源倡议影响的看法。
   - 讨论了对监管以及主要 AI 厂商潜在垄断行为的担忧，凸显了行业格局的变化。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://codeium.com/changel">Page Not Found | Windsurf Editor and Codeium extensions</a>: Codeium 是深受开发者喜爱且值得企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://status.codeium.com/">Codeium Status</a>: 未找到描述</li><li><a href="https://docs.codeium.com/windsurf/usage">Paid Plan and Credit Usage - Codeium Docs</a>: 未找到描述</li><li><a href="https://github.com/ichoosetoaccept/awesome-windsurf">GitHub - ichoosetoaccept/awesome-windsurf: A collection of awesome resources for working with the Windsurf code editor</a>: 使用 Windsurf 代码编辑器的精选资源合集 - ichoosetoaccept/awesome-windsurf</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: 未找到描述</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Windsurf 编辑器的最新更新和变化。
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1334615383969366128)** (815 messages🔥🔥🔥): 

> `DeepSeek 问题, 模型性能, Tool Calling 错误, O3 Mini 模型讨论, 用户体验反馈` 


- **DeepSeek 与 Tool Calling 的持续问题**：用户报告了 DeepSeek 的持续问题，特别是无效的 tool calls 以及未能按预期编写代码，这让许多人感到沮丧。
   - 一些人建议回退到之前的应用程序版本以缓解这些问题，因为最近的更新似乎恶化了工具的功能。
- **O3 Mini 性能观察**：几位用户讨论了他们对新 O3 Mini 模型的体验，指出与 Claude 3.5 等现有模型相比，其在编程方面的性能表现褒贬不一。
   - 虽然一些用户发现 O3 Mini 速度很快，但其他人批评它无法有效处理 tool calls，导致输出不完整。
- **用户体验与定价反馈**：针对产品定价与其感知价值之间的关系出现了批评，一些用户对产品质量与其成本不成正比表示不满。
   - 一些用户强调，尽管比竞争对手便宜，但应解决功能性问题，以避免疏远现有和潜在用户。
- **模型集成反馈**：有关将不同模型集成到 Windsurf 中的讨论，特别是推理模型的有效性及其对编程任务的影响。
   - 用户表示希望增强 multi-agent 架构，以改进编程工作流中的文档和任务管理。
- **用户建议与未来改进**：用户对应用程序的改进提出了建议，例如版本的滚动回退选项以及在 Discord 中显示增强的用户角色。
   - 总体而言，用户强调需要更强大的文档和故障排除支持，以提升用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.theverge.com/the-vergecast/603920/deepseek-nvidia-chatgpt-china-vergecast">How DeepSeek crashed the AI party</a>：在 The Vergecast 上：AI 芯片、AI 应用、AI 模型，AI 的一切。</li><li><a href="https://tenor.com/view/mari-marifootleg-herbal-tea-gif-25233295">Mari Marifootleg GIF - Mari Marifootleg Herbal Tea - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://status.anthropic.com">Anthropic Status</a>：未找到描述</li><li><a href="https://x.com/testingcatalog/status/1885301385182237062">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>：突发 🚨：OpenAI 即将发布 2 个新的推理模型：“o3-mini” 和 “o3-min-high”。“o3-mini-hype” 👀👀👀 引用 Tibor Blaho (@btibor91) “见见 o3-mini 家族...</li><li><a href="https://x.com/_mohansolo/status/1885078603966406980">Tweet from Varun Mohan (@_mohansolo)</a>：今天我们在 Windsurf 中提供了 DeepSeek R1 和 V3，使 Cascade 成为第一个支持 R1 的编程 Agent。起步成本将减半，但我们致力于迅速降低...</li><li><a href="https://x.com/windsurf_ai/status/1882561985621221451">Tweet from Windsurf (@windsurf_ai)</a>：只是在网上冲浪！🏄</li><li><a href="https://unreddit.netlify.app/">Unreddit</a>：未找到描述</li><li><a href="https://codeium.canny.io/feature-requests/p/search-in-chats">search in chats | Feature Requests | Codeium</a>：未找到描述</li><li><a href="https://codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>：Windsurf 编辑器的最新更新和变化。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1334619414879801395)** (999 messages🔥🔥🔥): 

> `Unsloth AI, DeepSeek 模型, 微调技术, 模型量化, 聊天机器人性能`

- **关于 Fine-tuning 和 Checkpoints 的讨论**：用户分享了使用 Unsloth 进行模型 Fine-tuning 的见解，强调了 Checkpointing 对于将 Adapters 与 Base Models 有效合并的重要性。
   - 提到训练会从最新的 Checkpoint 继续，从而为模型适配提供灵活性。
- **模型性能对比**：讨论转向了新模型的对比，特别是 DeepSeek-R1 和 Mistral，强调了 DeepSeek 变体的高效性能。
   - 用户表示 DeepSeek 有可能超越其他现有模型，特别是在 Coding 任务中。
- **模型输出问题**：一位用户在 Fine-tuned 的 LLaMA 3 模型中遇到了问题，输出包含了如 `<|eot_id|>` 等意外的 Tokens。
   - 讨论指出这些问题可能源于模型的训练格式化以及包含了不必要的 Tokens。
- **Quantization 和模型大小**：参与者讨论了 Distill-Qwen-1.5B 模型的内存占用和 Context Size，指出它根据其架构继承了最大长度。
   - 模型大小和 Context Length 对性能的影响被强调为使用 AI 模型时的重要考虑因素。
- **Unsloth 的多 GPU 支持**：关于 Unsloth 在多个 GPU 上进行模型 Fine-tuning 的能力提出了疑问，目前对增强支持仍有期待。
   - 社区对多 GPU 训练的发展及其对性能提升的影响表现出浓厚兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://videocardz.com/newz/nvidia-rtx-blackwell-gpu-with-96gb-gddr7-memory-and-512-bit-bus-spotted">搭载 96GB GDDR7 显存和 512-bit 位宽的 NVIDIA RTX Blackwell GPU 曝光 - VideoCardz.com</a>：NVIDIA 正在准备一款拥有 96GB 显存的工作站旗舰显卡。据称该显卡使用 3GB 模块。根据 ComputerBase 的报告，NVIDIA 即将推出的桌面显卡预计将...</li><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM 模型 VRAM 计算器 - NyxKrage 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb#scrollTo=vITh0KVJ10qX"">Google Colab</a>：未找到描述</li><li><a href="https://x.com/UnslothAI/status/1885393413585199202">Unsloth AI (@UnslothAI) 的推文</a>：在 @OpenWebUI 上本地运行 DeepSeek-R1 (671B) - 初学者指南。无需 GPU。使用我们的 1.58-bit Dynamic GGUF 和 llama.cpp。教程：https://docs.openwebui.com/tutorials/integrations/deepseekr1-...</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversa">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/qwen25-vl-all-versions-679ca6c784fad5bd976a05a1">Qwen2.5-VL (所有版本) - unsloth 集合</a>：未找到描述</li><li><a href="https://www.philschmid.de/mini-deepseek-r1">Mini-R1：复现 DeepSeek-R1 的“顿悟时刻” RL 教程</a>：复现 DeepSeek-R1 的“顿悟时刻”，并使用强化学习训练一个开源模型，尝试教它自主进行自我验证和搜索能力，以解决 Countdown Game。</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF">unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-GGUF">unsloth/Mistral-Small-24B-Instruct-2501-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLM/comments/1emtov3/storing_llm_models_ssd_or_hdd/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">从最后一个 Checkpoint 进行微调 | Unsloth 文档</a>：Checkpoint 允许您保存微调进度，以便您可以暂停并继续。</li><li><a href="https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit">unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=rsBiVxzmhG0">初学者 Google Colab 教程 | 什么是 Google Colab？ | Google Colab 详解 | Simplilearn</a>：🔥 云计算研究生课程：https://www.simplilearn.com/pgp-cloud-computing-certification-training-course?utm_campaign=26Mar2024GoogleColabT...</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course: 关于对齐 smol 模型的课程。</a>：关于对齐 smol 模型的课程。通过在 GitHub 上创建账号，为 huggingface/smol-course 的开发做出贡献。</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：以下是我们所有 Notebook 的列表：</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit">unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/aa6fb1321333fae8853d0cdc26bcb5d438e650a1/convert_lora_to_gguf.py#L229>">llama.cpp/convert_lora_to_gguf.py (位于 aa6fb1321333fae8853d0cdc26bcb5d438e650a1) · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号，为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">所有模型 | Unsloth 文档</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1334632862359617636)** (8 messages🔥): 

> `前端缺陷、模型敏感性、输出检测系统` 


- **关于前端缺陷的讨论**：一位成员指出，切换标签页后发生的更改可能是由于**前端方面的缺陷**造成的。
   - 这引发了关于模型在用户交互期间运作细微差别的推测。
- **模型输出的敏感性**：有人提到了一个关于输出正确性的敏感话题，特别是在与 **China** 相关的背景下。
   - 另一位成员表示赞同，称在处理此事时不会冒任何风险。
- **模型输出检测系统**：一位成员建议，输出的差异可能源于使用不同的**模型或系统来有效检测输出**。
   - 这引发了关于在敏感主题中确保输出可靠性的底层系统的疑问。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1334614712603643965)** (319 messages🔥🔥): 

> `DeepSeek R1 动态量化、微调 LLMs、使用 OpenWebUI、学习率调整、多 GPU 支持` 


- **使用 Ollama 运行 DeepSeek R1**：用户正尝试运行 DeepSeek R1 模型，特别是 1.58-bit 版本，在使用 Ollama（llama.cpp 的封装工具）时取得了不同程度的成功。
   - 面临的问题包括启动模型时的错误和性能瓶颈，因此有人建议直接使用 llama-server 运行。
- **微调技术与学习率**：关于微调模型合适学习率的讨论建议从 e-5 或 e-6 左右开始，并考虑数据集大小对调整的影响。
   - 建议在训练足够的 Epoch 后监控结果和评估指标，以衡量学习率的有效性。
- **AI 框架的集成问题**：对使用 Ollama API 运行本地 LLM 延迟的担忧促使了关于探索 OpenWebUI 等替代方案以获得更好性能的讨论。
   - 用户收到了关于将本地 LLM 集成到其应用程序中相关的限制和潜在挑战的建议。
- **内存与性能挑战**：用户分享了微调大模型时的内存限制经验，以及磁盘速度对推理速率的影响。
   - 建议包括优化存储解决方案和探索 Offloading 策略以提高性能。
- **模型支持的当前限制**：有人指出 Unsloth 目前不支持用于微调模型的多 GPU 训练，目前的重点是先支持所有模型。
   - 这一限制对于需要更多 RAM 来微调更大模型的用户产生了影响。


<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://docs.openwebui.com/tutorials/integrations/deepseekr1-dynamic/">🐋 使用 Llama.cpp 在 Open WebUI 运行 DeepSeek R1 Dynamic 1.58-bit</a>: 非常感谢 UnslothAI 的出色工作！多亏了他们的努力，我们现在可以运行完整的 DeepSeek-R1 671B 参数模型的 Dynamic 1.58-bit 量化版本（压缩至...）</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - Dynamic 4-bit 量化</a>: Unsloth 的 Dynamic 4-bit 量化选择性地避免对某些参数进行量化。这在保持与 BnB 4bit 相似的 VRAM 占用的同时，大大提高了准确性。</li><li><a href="https://huggingface.co/collections/unsloth/phi-4-all-versions-677eecf93784e61afe762afa">Phi-4 (所有版本) - Unsloth 集合</a>: 未找到描述</li><li><a href="https://tenor.com/view/skeleton-gif-26826812">骷髅 GIF - 骷髅 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/collections/unsloth/llama-32-66f46afde4ca573864321a22">Llama 3.2 - Unsloth 集合</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic#running%20r1">运行 DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-GGUF">unsloth/Mistral-Small-24B-Instruct-2501-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/unsloth-4-bit-dynamic-quants-67503bb873f89e15276c44e7">Unsloth 4-bit Dynamic Quants - Unsloth 集合</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/">初学者？从这里开始！ | Unsloth 文档</a>: 未找到描述</li><li><a href="https://x.com/OpenWebUI/status/1884719609552752801">来自 Open WebUI (@OpenWebUI) 的推文</a>: 🚀 感谢 @UnslothAI，你现在可以在 Open WebUI 上通过 llama.cpp 运行 1.58-bit DeepSeek-R1（非蒸馏版）了！💻⚡️（已在 M4 Max, 128GB RAM 上测试）📝 在他们的博客文章中深入了解详情：htt...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1idseqb/deepseek_r1_">Reddit - 深入了解一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1idseqb/deepseek_r1_671b_over_2_toksec_without_gpu_on/">Reddit - 深入了解一切</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unsl">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements#fine-tuning-vram-requirements">Unsloth 需求 | Unsloth 文档</a>: 这里是 Unsloth 的需求，包括系统和 GPU VRAM 需求。</li><li><a href="https://old.reddit.com/r/selfhosted/comments/1ic8zil/yes_you_can_run_deepseekr1_locally_on_your_device/">是的，你可以在本地设备上运行 DeepSeek-R1（最低 20GB RAM）</a>: 我最近看到一些误解，认为无法在本地设备上运行 DeepSeek-R1。上周末，我们正忙于让你能够...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1334927460491460719)** (7 条消息): 

> `Qwen2.5-0.5B-instruct, Quadro P2000 GPU, Old Hardware Usability` 


- **在 Quadro P2000 上进行 Qwen2.5 预训练**：一位成员幽默地报告了在仅有 **5GB VRAM** 的 **Quadro P2000** 上对 **Qwen2.5-0.5B-instruct** 进行持续预训练的情况。
   - *“我会在 2026 年告诉你结果如何”* 呼应了对该 GPU 性能的戏谑怀疑。
- **老旧 GPU 的挣扎**：成员们对 **Quadro P2000** 表示担忧，其中一人称它正在 *“祈求上帝救救它”*。
   - 另一位成员注意到了它的机龄，补充道 *“好吧，你得庆幸这么一块老古董金属居然还能用”*。
- **GPU 渴望休息**：一位成员幽默地建议 **Quadro P2000** *“想永远睡下去”*，突显了它的吃力。
   - 这一评论紧随一段关于尽管该 GPU 已经过时但仍在使用的轻松讨论。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1334691057090302085)** (6 messages): 

> `vLLM Integration, Batch Throughput Investigation, Model Loading Concerns, XGB Usage in Unsloth and vLLM, Offloading Issues with vLLM` 


- **与 vLLM 的潜在集成**：正在讨论将 **vLLM** 以某种方式集成到系统中，作为持续评估的一部分。
   - *我们可能会以某种方式集成它*。
- **调查批处理吞吐量**：预计 **vLLM** 具有更高的批处理吞吐量，这促使进一步调查以确定最佳用法。
   - 最初的计划是直接使用 vLLM，但如果通过 **Unsloth** 进行批处理推理证明更快，则可能会默认使用后者。
- **模型加载疑虑**：由于 Unsloth 和 vLLM 都使用 XGB，人们对模型**重复加载**的可能性表示担忧。
   - *我需要检查我们是否在重复加载模型。*
- **XGB 冗余问题**：讨论强调，同时使用 **Unsloth** 和 **vLLM** 意味着它们实际上占用了 **2XGB**，引发了效率问题。
   - *即 Unsloth 使用 XGB，而 vLLM 使用 XGB，即 2XGB。*
- **vLLM Offloading 能力**：有人指出 **vLLM** 可能还无法在 **gguf** 兼容的情况下执行 Offloading，特别是在 **deepseek v2 architecture** 上。
   - 一位成员询问最近是否有针对此问题的补丁，表明故障排除正在进行中。


  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1335007129316102184)** (1 messages): 

> `Aider v0.73.0, o3-mini support, Reasoning effort argument, OpenRouter R1 free support` 


- **Aider v0.73.0 正式发布**：最新版本 **Aider v0.73.0** 包含了对 **o3-mini** 的全面支持，使用命令 `aider --model o3-mini` 即可调用。
   - 值得注意的是，据报道 Aider 编写了此版本中 **69% 的代码**，展示了显著的内部开发成果。
- **引入新的推理力度（reasoning effort）参数**：Aider 引入了一个新的 `--reasoning-effort` 参数，可以设置为 **low**、**medium** 或 **high** 来自定义性能。
   - 这一增强功能旨在为用户提供更大的灵活性，以控制模型的推理能力。
- **改进上下文窗口处理**：0.73.0 版本还改进了对**上下文窗口大小限制**的处理，提供了更好的提示信息和针对 **Ollama** 的特定指导。
   - 预计这一调整将通过在使用过程中提供更直观的指令来提升用户体验。
- **支持 OpenRouter 上的 R1 免费版**：Aider 现在支持通过命令 `--model openrouter/deepseek/deepseek-r1:free` 使用 **OpenRouter** 上的 R1 免费版。
   - 该功能扩大了获取 R1 功能的途径，促进了用户与平台的互动。
- **增强 Aider 中的目录创建功能**：Aider 添加了在生成新文件时**自动创建父目录**的功能，简化了文件管理。
   - 这一改进为处理新文件结构的用户提供了更顺畅的工作流。



**提到的链接**：<a href="https://aider.chat/HISTORY.html">发布历史</a>：关于 Aider 编写自身代码的发布说明和统计数据。

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1334619451278098513)** (979 messages🔥🔥🔥): 

> `O3 Mini Performance, Tool Use in Aider, Rust Programming, OpenAI and Pricing, Linters in Aider`

- **O3 Mini 证明了其价值**：O3 Mini 被指出具有与 O1 相当的性能，而成本仅为后者的一小部分。用户报告称其运行流畅，并能生成无错误的、可编译的 Rust 代码。
   - 尽管最初对其能力持怀疑态度，但用户发现 O3 Mini 在编程任务中比其他模型更有效且速度更快。
- **在 Aider 中使用工具功能**：用户正在利用 Aider 的工具能力来提高编程效率，包括为渗透测试设置 REPL。
   - Aider 整合终端操作学习指令的能力，引发了关于其在真实编程环境中实际用途的讨论。
- **Rust 编程体验**：一位用户分享了使用 O3 Mini 进行 Rust 编程的经验，强调了它在处理 Rust 特定语法和代码结构方面的有效性。
   - 用户一致认为，拥有一个精通数学的模型有助于编写结构良好的 Rust 代码，从而提高生产力。
- **OpenAI 的竞争优势**：与 OpenAI 之前的产品和其他模型相比，O3 Mini 的发布带来了极具竞争力的定价，一些用户将其视为针对 Deepseek 等强大开源模型的战略举措。
   - 用户对该模型发布后 OpenAI 的股市影响力表示关注，并指出认知会影响价值。
- **Aider 中的 Linting 和测试**：Aider 允许用户在修改代码时自动进行 Linting 和测试，增强了 AI 模型生成的代码的可靠性。
   - 用户注意到，使用 linters 可以更有效地捕获代码库中的错误，使 O3 Mini 成为快速开发的理想选择。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/testingcatalog/status/1885301385182237062">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：重大新闻 🚨：OpenAI 即将发布 2 款新的推理模型：“o3-mini” 和 “o3-min-high”。“o3-mini-hype” 👀👀👀 引用 Tibor Blaho (@btibor91) 的话 “见见 o3-mini 家族...”</li><li><a href="https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=88bcbd4f7e76ad174f529d3453a0909f">Rust Playground</a>：未找到描述</li><li><a href="https://svelte.dev/docs/llms">未找到标题</a>：未找到描述</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>：你可以通过命令行或 Python 对 aider 进行脚本化操作。</li><li><a href="https://aider.chat/docs/usage/lint-test.html">Linting and testing</a>：自动修复 Linting 和测试错误。</li><li><a href="https://tenor.com/view/do-it-star-wars-emperor-palpatine-palpatine-gif-799657800635657398">Do It Star Wars GIF - Do it Star wars Emperor palpatine - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/critical-role-crit-role-cr-arsequeef-undeadwood-gif-15546127">Critical Role Crit Role GIF - Critical Role Crit Role Cr - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/no-witnesses-erase-memory-forget-gif-20806865">No Witnesses GIF - No Witnesses Erase - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/oh-my-shocked-how-dare-you-shock-shocking-gif-11277509288657991552">Oh My Shocked GIF - Oh My Shocked How Dare You - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/llms/other.html">Other LLMs</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://x.com/arankomatsuzaki/status/1885025043178283379">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：从 o1 到 o3 的跨越是指数级的，完全跳过了 o2。如果这种模式持续下去，o3 将不会引向 o4——它会直接跳到 o9。</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html#controlling-o1-reasoning-effort">高级模型设置</a>：为 LLMs 配置高级设置。</li><li><a href="https://docs.rs/reedline">reedline - Rust</a>：未找到描述</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://x.com/Saboo_Shubham_/status/1885167873615945893">来自 Shubham Saboo (@Saboo_Shubham_) 的推文</a>：DeepClaude 结合了 Claude Sonnet 3.5 与 DeepSeek R1 CoT 推理，表现优于 OpenAI o1、DeepSeek R1 和 Claude Sonnet 3.5。100% 免费且开源。</li><li><a href="https://build.nvidia.com/deepseek-ai/deepseek-r1">Deepseek-ai 的 deepseek-r1 模型 | NVIDIA NIM</a>：最先进的高效率 LLM，在推理、数学和编程方面表现卓越。</li><li><a href="https://github.blog/changelog/2025-01-31-openai-o3-mini-now-available-in-github-copilot-and-github-models-public-preview">OpenAI o3-mini 现已在 GitHub Copilot 和 GitHub Models 中可用（公开预览版）· GitHub 更新日志</a>：OpenAI o3-mini 现已在 GitHub Copilot 和 GitHub Models 中可用（公开预览版）</li><li><a href="https://x.com/btibor91/status/1885378122498892142">来自 Tibor Blaho (@btibor91) 的推文</a>：T̶h̶e̶ ̶O̶3̶ ̶F̶a̶m̶i̶l̶y̶ o3-mini 家族。引用 Tibor Blaho (@btibor91) 的话：O3 家族</li><li><a href="https://aider.chat/2025/01/28/deepseek-down.html#openrouter">其他 DeepSeek V3 供应商</a>：DeepSeek 的 API 一直存在稳定性问题。以下是你可以使用的替代供应商。</li><li><a href="https://github.com/Aider-AI/aider/tree/main/benchmark">Aider-AI/aider 项目 main 分支下的 aider/benchmark</a>：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_mark_chen_kevin_weil/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/OpenAutoCoder/Agentless">GitHub - OpenAutoCoder/Agentless：Agentless🐱：一种无需 Agent 自动解决软件开发问题的方法</a>：Agentless🐱：一种无需 Agent 自动解决软件开发问题的方法 - OpenAutoCoder/Agentless</li><li><a href="https://github.com/Aider-AI/aider/pull/2998">由 serialx 提交的 Pull Request #2998 · Aider-AI/aider：添加了 DeepSeek R1 + DeepSeek V3 基准测试</a>：我想分享 DeepSeek R1 架构师 + DeepSeek V3 编辑器的基准测试结果：得分为 59.1%。接近 o1 的性能，但成本仅为 6.33 美元！是 R1+Sonnet 的一半...</li><li><a href="https://www.economist.com/briefing/2025/01/23/chinas-ai-industry-has-almost-caught-up-with-americas">为什么中国 AI 震惊了世界</a>：DeepSeek 的模型比美国竞争对手便宜得多，且几乎一样出色</li><li><a href="https://archive.is/2025">archive.is/2025</a>

.01.27-195417/https://www.economist.com/briefing/2025/01/23/chinas-ai-industry-has-almost-caught-up-with-americas">为什么中国 AI 震惊了世界</a>：未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1334671052390924381)** (72 条消息🔥🔥): 

> `Aider 配置、DeepSeek 问题、API Key 处理、模型性能、文件管理` 


- **模型的 Aider 配置**：用户讨论了为 **DeepSeek** 和 **Claude** 等模型配置 Aider 的问题，一些用户遇到了配置文件中无法识别 API Key 设置的问题。
   - 一位用户指出，在运行 Aider 命令时将 API Key 设置为环境变量解决了他们的问题。
- **DeepSeek API 问题**：几位成员报告了 **DeepSeek** 的使用困难，特别是挂起问题和错误的空格处理导致的性能问题。
   - 用户提到正在考虑 DeepSeek 的替代方案，参考了本地模型并寻求推荐。
- **命令文件的挑战**：一位用户对从文件执行命令表示沮丧，称 Aider 会尝试处理所有行，导致非命令行出现不必要的警告。
   - 他们表示希望有一个命令可以在保持聊天模式的同时执行文件中的命令。
- **探索模型性能**：一些讨论围绕理解模型性能展开，特别是关于 Context Window 和 Token 限制的高效配置。
   - 一位用户提出了关于 Context Window 是否自动管理的问题，随后引发了关于 **O3 context window** 大小和能力的解释。
- **Aider 中的文件管理**：对话包括关于在使用 Aider 时列出已保存文件和管理文件格式的查询，目前尚未发现用于列出保存内容的命令。
   - 参与者提到需要更好地组织保存文件，并建议为 Aider 保存内容建立一个专门的目录。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准。</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">文件编辑问题</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html#model-settings">高级模型设置</a>：为 LLM 配置高级设置。</li><li><a href="https://openrouter.ai/anthropic/claude-3.5-sonnet/providers)">Anthropic: Claude 3.5 Sonnet</a>：全新的 Claude 3.5 Sonnet 提供了优于 Opus 的能力，速度快于 Sonnet，且价格与 Sonnet 持平。Sonnet 特别擅长：- 编程：在 SWE-Bench Verified 上得分约 49%，高于...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1334614164475482174)** (705 条消息🔥🔥🔥): 

> `Perplexity AI 模型, O3 Mini 发布, O1 和 R1 性能, DeepSeek 模型对比, AI 平台用户体验` 


- **关于 AI 模型性能的讨论**：用户讨论了 O3 Mini、R1 和 O1 等各种 AI 模型之间的性能差异，指出 O1 在谜题和计算等任务中通常表现更好。
   - 对比包括特定的使用场景，突出了模型在处理编程和推理任务时的表现。
- **O3 Mini 现已可用**：O3 Mini 已向用户发布，成员们对其功能表示兴奋，并提到其推广速度比之前的模型更快。
   - 用户开始测试 O3 Mini，讨论其限制并将其性能与现有模型进行对比。
- **Perplexity AI 的用户体验**：部分用户分享了他们使用 Perplexity 应用的体验，特别是关于缺乏默认模型设置以及仅为免费用户提供“Reason”按钮的问题。
   - 对话强调了对应用结构的不满以及对更好模型管理选项的需求。
- **DeepSeek 作为替代方案**：成员们讨论了将 DeepSeek 与 O1 和 R1 等模型结合使用，权衡了其在计算方面的优势，但也指出了其在文本翻译方面的弱点。
   - 用户根据其 AI 使用需求表达了对某些模型的偏好，并强调了在 You.com 等平台上可以使用不同的 AI 模型。
- **技术支持与查询**：用户寻求技术方面的澄清，包括将账户链接到应用以及了解不同用户层级的模型限制。
   - 用户对账户设置和权限表示担忧，并建议联系支持部门寻求帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://inference.cerebras.ai/">Cerebras Inference</a>：未找到描述</li><li><a href="https://x.com/FixersPro/status/1885425262931632279">来自 pro_game_fixers (@FixersPro) 的推文</a>：o3-mini-medium vs o3-mini-high vs claude sonnet 3.5 提示词：http://pastebin.com/MxMfi635 构建一个双人蛇类游戏 AI vs 人类</li><li><a href="https://tenor.com/view/diddy-gif-8961692530157879891">Diddy GIF - Diddy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://cybersecuritynews.com/deepseek-database-leaked/">DeepSeek 数据库泄露 - 数据库密钥、日志和聊天记录的完全控制权被曝光</a>：DeepSeek，一家著名的中国 AI 初创公司，暴露了一个可公开访问的 ClickHouse 数据库，其中包含密钥、日志和聊天记录。</li><li><a href="https://www.reddit.com/r/singularity/comments/1iedkrg/o3mini_and_o3minihigh_are_rolling_out_shortly_">Reddit - 深入了解任何内容</a>：未找到描述</li><li><a href="https://x.com/_kevinlu/status/1885406995613892711">来自 Kevin Lu (@_kevinlu) 的推文</a>：我们发布了 o3-mini，今天在 ChatGPT 中对所有用户（免费）可用！o3-mini-low 比 o1-mini 更快（且通常更好），而 o3-mini-high 是目前最强大的公开可用推理模型...</li><li><a href="https://x.com/aravsrinivas/status/1885201821406511524?s=61&t=Un1yLqIRg3sDiqpmnWHBfg">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：哪个模型会更优？o3-mini 还是 DeepSeek R1？</li><li><a href="https://github.com/marketplace/models/azureml-deepseek/DeepSeek-R1/playground">更好地共同构建软件</a>：GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、分叉并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://qwenlm.ai/">Qwen Chat</a>：未找到描述</li><li><a href="https://x.com/OpenAI/status/1885406586136383634">来自 OpenAI (@OpenAI) 的推文</a>：OpenAI o3-mini 现已在 ChatGPT 和 API 中可用。Pro 用户将拥有对 o3-mini 的无限访问权限，Plus 和 Team 用户的速率限制将是 o1-mini 的三倍。免费用户可以在 ChatGPT 中试用 o3-mini...</li><li><a href="https://www.reddit.com/r/singularity/comments/1iedkrg/o3mini_and_o3minihigh_are_rolling_out_shortly_in/">Reddit - 深入了解任何内容</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1334674140543586304)** (5 条消息): 

> `AI Prescription Bill, TB Outbreak Kansas, Nadella's AI Predictions, Asteroid Life Seeds, Harvard Dataset` 


- **AI 处方法案提案**：一项新的 [AI Prescription Bill](https://www.perplexity.ai/page/google-offers-voluntary-exit-f-tA7gBGbPSzymq8WBAwkTUw#93ca4910-afc1-4e9a-a30c-c219ffc1bb02) 旨在监管 AI 在医疗保健领域的应用，强调伦理标准和问责制。
   - 这一举措反映了人们对 AI 在医疗决策中影响的日益关注。
- **堪萨斯州面临结核病爆发**：堪萨斯州目前正在应对 **结核病（TB）爆发**，促使卫生官员发布警告并协调应对工作。
   - 卫生专家强调了通过社区意识监测和防止进一步传播的重要性。
- **Nadella 预测 AI 的“杰文斯悖论”**：微软 CEO Satya Nadella 预测 AI 领域将出现 **Jevons Paradox**（杰文斯悖论），暗示技术进步可能会导致资源消耗增加而非节约。
   - 他的言论引发了关于人工智能发展可持续极限的讨论。
- **小行星携带生命种子**：最近的一项发现表明，一颗小行星可能含有 **生命种子**，引发了关于地外生物学的有趣问题。
   - 研究表明，了解此类小行星可能会揭开地球生命起源的秘密。
- **哈佛数据集探索**：[Harvard dataset](https://www.perplexity.ai/search/harvard-dataset-u9AyiW_EQYOmwj_9CvY9fw) 因其在 AI 研发中的潜在应用而受到关注。
   - 研究人员正在评估其在应对各种科学挑战方面的价值。



**提到的链接**：<a href="https://www.youtube.com/embed/9wvmCc4XQSE">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1334695468529090600)** (2 条消息): 

> `Sonar Reasoning, Plane Crash Information` 


- **Sonar Reasoning 在上下文处理上存在困难**：一位成员指出 **sonar reasoning** 在处理特定查询时效果不佳，并以 **波托马克河上空的飞机失事** 为例。
   - 虽然该模型在技术上提供了正确的信息，但它提供的是 **1982年** 事件的数据，而非最近的坠机事故。
- **近期数据与历史数据的混淆**：讨论强调了一个显著问题，即 sonar reasoning 可能会提供过时信息，从而在关键场景中导致潜在的混淆。
   - 该成员强调，尽管旧数据是准确的，但在对时间敏感的情况下，它可能无法满足用户的即时需求。


  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1334641669437657189)** (1 条消息): 

> `LM Studio 0.3.9 release, Idle TTL feature, Separate reasoning_content, Auto-update for runtimes, Nested folders support` 


- **LM Studio 0.3.9 发布，带来令人兴奋的更新**：新版本的 LM Studio 引入了多项功能，包括 **Idle TTL**、**reasoning_content** 以及 **运行时自动更新**。你可以在[这里](https://lmstudio.ai/download)下载或通过应用进行更新。
   - 请参阅完整的[变更日志](https://lmstudio.ai/blog/lmstudio-v0.3.9)以获取详细概览。
- **使用 Idle TTL 功能管理内存**：**Idle TTL** 功能允许用户为 API 模型设置生存时间（time-to-live），自动驱逐未使用的模型以优化内存使用。这可以通过请求本身或命令行选项实现，详情请参阅[文档](https://lmstudio.ai/docs/api/ttl-and-auto-evict)。
   - _此功能简化了内存管理_，减少了手动干预。
- **通过独立的 reasoning_content 让聊天变得更智能**：LM Studio 现在支持在聊天响应中使用独立的 `reasoning_content` 字段，从而实现与 DeepSeek API 的兼容。用户可以通过实验性设置启用此功能。
   - _此更新旨在通过将推理过程与响应内容分离来增强对话交互_。
- **自动更新功能简化了运行时管理**：LM 运行时的 **自动更新** 现在默认启用，最大限度地减少了跨多个组件手动更新的麻烦。如果需要，用户可以在 App Settings 中禁用此选项。
   - _此功能确保你的环境保持最新状态_，无需额外操作。
- **终于支持嵌套文件夹**：用户现在可以从 Hugging Face 仓库的 **嵌套文件夹** 中下载模型，解决了长期以来对更好组织方式的需求。这使得访问子文件夹中的模型变得更加高效。
   - _这一新增功能预计将提升用户体验_并简化模型管理。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/docs/api/ttl-and-auto-evict">Idle TTL and Auto-Evict | LM Studio Docs</a>: 可选在一定时间（TTL）后自动卸载空闲模型</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.9">LM Studio 0.3.9</a>: Idle TTL、运行时自动更新、支持 HF 仓库中的嵌套文件夹，以及聊天补全响应中的独立 `reasoning_content`
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1334614953067548743)** (362 条消息🔥🔥): 

> `LM Studio 性能与模型使用、用于 C# 开发的 AI 模型、OpenAI o3-mini 发布、DeepSeek 模型性能、LM Studio 下载速度问题` 


- **LM Studio 性能担忧**：用户在使用 LM Studio 时遇到了各种性能问题，特别是在加载 DeepSeek R1 等模型时，如果 VRAM 不足可能会导致错误。
   - 一些用户注意到最近的更新可能降低了下载速度，而其他人则建议使用 Hugging Face 代理等优化手段。
- **为 C# 游戏开发选择 AI 模型**：对于开发 C# 应用程序，考虑到硬件限制，用户推荐使用 Qwen2.5 Coder 和 DeepSeek 蒸馏版本等模型。
   - 虽然高端模型表现更好，但低端模型在不完全依赖的情况下作为代码参考也足够了。
- **OpenAI 的 o3-mini 发布**：OpenAI 最近推出了 o3-mini 模型，旨在快速响应数学、编程和科学任务，免费版 ChatGPT 用户也可使用。
   - 尽管发布了公告，但用户对该模型的实际可用性表示困惑。
- **DeepSeek 模型效能**：DeepSeek 模型因其编程能力而受到关注，有报告显示其性能相比 OpenAI 模型等竞争对手具有显著优势。
   - 讨论还涉及了由于 DeepSeek 等模型的进步，竞争如何促使 OpenAI 降低价格。
- **LM Studio 下载速度问题**：用户反映 LM Studio 的下载速度比以前慢，有人认为操作系统设置可能是一个因素。
   - 一种解决方法是直接从 Hugging Face 下载模型并将其放入 .cache 文件夹，以获得更快的访问速度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/docs/basics/rag">Chat with Documents | LM Studio Docs</a>: 如何为 LLM 提供本地文档作为额外上下文</li><li><a href="https://docs.openwebui.com/features/">⭐ Features | Open WebUI</a>: Open WebUI 的核心功能 ⭐</li><li><a href="https://www.theverge.com/news/603849/openai-o3-mini-launch-chatgpt-api-available-now">OpenAI launches new o3-mini reasoning model with a free ChatGPT version</a>: 只有 Pro 用户可以无限次使用 o3-mini。</li><li><a href="https://lmstudio.ai/docs/api">LM Studio as a Local LLM API Server | LM Studio Docs</a>: 使用 LM Studio 在 localhost 上运行 LLM API 服务器</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf">openbmb/MiniCPM-o-2_6-gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://goodsnooze.gumroad.com/l/macwhisper?ref=producthunt">🎙️ MacWhisper</a>: 使用 OpenAI 先进的转录技术 Whisper 快速轻松地将音频文件转录为文本。无论您是在录制会议、讲座还是其他重要音频，MacW...</li><li><a href="https://huggingface.co/Systran/faster-whisper-medium">Systran/faster-whisper-medium · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/eye-of-sauron-lotr-lord-of-the-rings-gif-16715227">Eye Of Sauron Lotr GIF - Eye Of Sauron Lotr Lord Of The Rings - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://goodsnooze.gumroad.com/l/macwhisper?ref=product">🎙️ MacWhisper</a>: 使用 OpenAI 先进的转录技术 Whisper 快速轻松地将音频文件转录为文本。无论您是在录制会议、讲座还是其他重要音频，MacW...</li><li><a href="https://github.com/Les-El/Ollm-Bridge">GitHub - Les-El/Ollm-Bridge: Easily access your Ollama models within LMStudio</a>: 在 LMStudio 中轻松访问您的 Ollama 模型。通过在 GitHub 上创建账号为 Ollm-Bridge 开发做贡献。</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: 👾 LM Studio CLI</a>: 👾 LM Studio CLI。通过在 GitHub 上创建账号为 lms 开发做贡献。</li><li><a href="https://lmstudio.ai/download">Download LM Studio - Mac, Linux, Windows</a>: 发现、下载并运行本地 LLM</li><li><a href="https://huggingface.co/Qwen">Qwen (Qwen)</a>: 未找到描述</li><li><a href="https://github.com/sammcj/llamalink">GitHub - sammcj/llamalink: Link you Ollama models to LM-Studio</a>: 将您的 Ollama 模型链接到 LM-Studio。通过在 GitHub 上创建账号为 llamalink 开发做贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1334614111123935302)** (158 条消息🔥🔥): 

> `Qwen models, LM Studio performance, Using multiple GPUs, Vulkan support for GPUs, Context length in LLMs` 


- **Qwen 模型表现优于其他模型**：用户注意到 Qwen 模型，特别是 Qwen2.5-7B-Instruct-1M，在处理长上下文方面比之前的模型更好，提供了更高的性能。
   - 一位用户体验到了显著的提升，并建议其他人开启 Flash Attention 和 K/V cache quantization 以获得更好的效率。
- **LM Studio 与 Intel GPU 的挑战**：一位用户询问在 Intel UHD GPU 上运行 LM Studio 的情况，但确认目前 LM Studio 还没有 Linux ARM 版本。
   - 预计未来的项目可能会提供支持，但目前的选项有限。
- **同时使用 NVIDIA 和 Intel GPU**：一位拥有 NVIDIA RTX 4080 和 Intel UHD GPU 的用户表示有兴趣同时利用两者来增强性能，特别是利用共享系统 RAM。
   - 然而，据解释，当 VRAM 耗尽时，NVIDIA 驱动程序将默认使用系统 RAM，这可能会限制双 GPU 使用的有效性。
- **上下文长度对性能的影响**：讨论强调了上下文长度会显著影响 RAM 占用，超过限制会导致性能错误。
   - 一位用户报告称在强大的配置下能够处理高达 80k tokens，这表明 RAM 与模型效率之间存在强相关性。
- **性能指标和模型选择**：用户分享了使用 DeepSeek 等各种模型的经验，并讨论了它们的 token-per-second 指标，强调了 VRAM 与模型复杂度之间的平衡。
   - 一些人建议使用参数量更大的模型或探索 quantization 技术来提高吞吐量。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/introducing">Introducing Unsloth</a>：未找到描述</li><li><a href="https://www.amazon.com/dp/B074P6BNGZ?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1">无标题</a>：未找到描述</li><li><a href="https://tenor.com/view/whale-swallow-eat-nom-hungry-gif-17097355">Whale Swallow GIF - Whale Swallow Eat - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements#approximate-vram-requirements-based-on-model-parameters">Unsloth Requirements | Unsloth Documentation</a>：这里是 Unsloth 的要求，包括系统和 GPU VRAM 要求。</li><li><a href="https://www.amazon.com/dp/B074P6BNGZ?ref=ppx_yo2ov_dt_">Amazon.com: Libre Computer Board AML-S905X-CC (Le Potato) 2GB 64-bit Mini Computer for 4K Media : Electronics</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1334614796233871460)** (520 条消息🔥🔥🔥): 

> `DeepSeek R1 and Sonnet 3.6 Integration, O3 Mini Performance, MCP Tool Usage, Claude Model Updates, User Experience and Feedback` 


- **DeepSeek R1 作为架构师配合 Sonnet 3.6 作为执行者**：用户注意到将 R1 用于规划并配合 Sonnet 3.6 进行编码可以产生更好的结果，因为 R1 提供的 Chain of Thought 上下文增强了 Sonnet 的输出。
   - 这种方法在处理编码任务方面表现出显著改进，允许用户高效地解决复杂问题。
- **对 O3 Mini 的评价褒贬不一**：虽然一些用户发现 O3 Mini 在某些任务中很有效，但其他人对其与 Sonnet 3.6 相比的性能表示失望。
   - 尽管 O3 Mini 具备相应能力，但用户对其需要明确的 prompt 才能执行代码更改表示担忧。
- **Cursor 中的 MCP Tool 利用**：讨论显示虽然 MCP 工具运行良好，但用户觉得 Cursor 需要更好的集成和支持。
   - 一些参与者对 MCP 缺乏突破性功能感到沮丧，转而依赖自定义工具来实现高效的工作流。
- **对新 Claude 模型发布的期待**：用户对 Anthropic 可能发布的新版本感到兴奋，渴望 Claude 模型的更新能够进一步增强他们的工作流。
   - 许多人认为，像 Claude 4.0 Symphony 这样的高级版本将显著改善编码和解决问题的体验。
- **用户体验与挑战**：用户分享了使用当前 AI 模型的各种经验，指出了项目中的具体成功案例和挑战。
   - 虽然一些人通过新模型立即找到了解决方案，但其他人则面临响应时间慢和结果不一致的困扰。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/windsurf_ai/status/1885077046663217230">Windsurf (@windsurf_ai) 的推文</a>：DeepSeek R1 和 V3 现已在 Windsurf 中上线，完全托管在西方服务器上。我们在 R1 中实现了 tool calling，使其首次能够用于 coding agent。</li><li><a href="https://modelcontextprotocol.io/">简介 - Model Context Protocol</a>：未找到描述</li><li><a href="https://www.testingcatalog.com/anthropic-developing-web-search-feature-for-claude-ai/">Anthropic 正在为 Claude AI 开发网页搜索功能</a>：Anthropic 的 Claude AI 将获得网页搜索能力，弥补静态语言模型与实时数据检索之间的差距。敬请期待更新！</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet 在 aider 的多语言基准测试中创下 SOTA</a>：R1+Sonnet 在 aider 多语言基准测试中创下了新的 SOTA。与 o1 相比，成本降低了 14 倍。</li><li><a href="https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models-with-silent-backdoor/">数据科学家成为带有静默后门的恶意 Hugging Face ML 模型的目标</a>：Hugging Face 是否成为了基于模型的攻击目标？查看攻击机制的详细解释，以及识别真实威胁所需的操作 ></li><li><a href="https://www.mcpservers.ai/">MCP Servers</a>：浏览最大的 Model Context Protocol 服务器库。分享你与其他开发者创建的 Model Context Protocol 服务器。</li><li><a href="https://www.reddit.com/r/ollama/comments/1ieb1za/warning_major_price_increase_for_cursors_agentic/">Reddit - 尽情讨论</a>：未找到描述</li><li><a href="https://www.reddit.com/r/cursor/comments/1ie8u65/warning_major_price_increase_for_cursors_agentic/">Reddit - 尽情讨论</a>：未找到描述</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1idzrdl/o3_releasing_tomorrow/">Reddit - 尽情讨论</a>：未找到描述</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1:free">DeepSeek R1 (免费) - API、提供商、统计数据</a>：DeepSeek R1 已发布：性能与 OpenAI o1 相当，但采用开源形式并具有完全开放的推理 token。其参数量为 671B，推理过程中激活参数为 37B...</li><li><a href="https://github.com/microsoft/BitNet">GitHub - microsoft/BitNet: 1-bit LLMs 的官方推理框架</a>：1-bit LLMs 的官方推理框架。通过在 GitHub 上创建账号为 microsoft/BitNet 开发做出贡献。</li><li><a href="https://github.com/daniel-lxs/mcp-server-starter">GitHub - daniel-lxs/mcp-server-starter</a>：通过在 GitHub 上创建账号为 daniel-lxs/mcp-server-starter 开发做出贡献。</li><li><a href="https://github.com/HarshJ23/deepseek-claude-MCP-server">GitHub - HarshJ23/deepseek-claude-MCP-server：一个将 DeepSeek R1 模型的推理能力集成到 Claude 桌面应用的 MCP 服务器。</a>：一个将 DeepSeek R1 模型的推理能力集成到 Claude 桌面应用的 MCP 服务器。- HarshJ23/deepseek-claude-MCP-server</li><li><a href="https://youtu.be/FrM6ZzCiLwU">DeepSeek R1 + Claude 3.5 Sonnet：2 分钟开发者工作流指南</a>：另一个简短视频，我描述了在 Cursor 将 DeepSeek R1 作为免费模型加入后，我最新的工作流调整！尝试这个...</li><li><a href="https://github.com/protectai/modelscan">GitHub - protectai/modelscan：防御模型序列化攻击</a>：防御模型序列化攻击。通过在 GitHub 上创建账号为 protectai/modelscan 开发做出贡献。</li><li><a href="https://www.cursor.com/changelog">更新日志 | Cursor - AI 代码编辑器</a>：新的更新和改进。</li><li><a href="https://www.reddit.com/r/ChatGPTCoding/s/p0ZBg4JfMg">Reddit - 尽情讨论</a>：未找到描述</li><li><a href="https://pureinsights.com/blog/2024/1-bit-llms-the-future-of-efficient-ai/">1-Bit LLMs：高效 AI 的未来？- Pureinsights</a>：这篇博客解释了关于 1-bit LLMs 的初步研究，以及它们在生产既有效又高效的 AI 模型方面的潜力。</li><li><a href="https://www.reddit.com/r/cursor/comments/1iecvyh/cursor_has_limit_on_how_many_free_trials_you_can/">Reddit - 尽情讨论</a>：未找到描述</li><li><a href="https://www.reddit.com/r/cursor/comments/1ied6sb/cursor_mercy_hack_isnt_working_for_too_many_trial/">Reddit - 尽情讨论</a>：未找到描述</li><li><a href="https://github.com/Aider-AI/aider/pull/2973">Frankenclaude：R1 推理 + Sonnet，由 jbellis 提交 · Pull Request #2973 · Aider-AI/aider</a>：我想看看如果我们将 R1 的 chain of thought 与 Sonnet 的编辑能力结合起来会发生什么。我以一种最粗放的方式将其接入了 aider（尽管我认为移动 send_...）
</li>
</ul>

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1334984549180112907)** (1 messages): 

> `o3-mini model release, Reasoning capabilities, BYOK program updates` 


- **o3-mini 模型面向 BYOK 用户发布**：OpenAI 的新模型 **o3-mini** 现已面向使用层级（usage tiers）为 3 到 5 的 Bring Your Own Key (BYOK) 用户开放，提供**增强的推理能力**。
   - 用户可以**[在此添加其密钥](https://openrouter.ai/settings/integrations)**开始使用该模型，在专家测试中，该模型相比其前代产品显示出 **56% 的偏好度**。
- **o3-mini 取得令人印象深刻的基准测试结果**：**o3-mini** 在 AIME/GPQA 上的表现与体量更大的 **o1 模型**相当，并且在复杂问题上的重大错误减少了 **39%**。
   - 该模型还包含**内置 function calling** 和结构化输出等功能，迎合了开发者和 STEM 爱好者的需求。
- **开发者的实惠选择**：对于在**数学、科学**和**编程**方面寻求可靠帮助的用户，**o3-mini** 模型提供了一个高性价比的解决方案。
   - 对于希望在不产生过高成本的情况下获得高级推理能力的 BYOK 用户来说，这是一个极具吸引力的选择。



**提到的链接**：<a href="https://openrouter.ai/openai/o3-mini">o3 Mini - API, Providers, Stats</a>：OpenAI o3-mini 是一款针对 STEM 推理任务优化的经济型语言模型，尤其在科学、数学和编程方面表现出色。该模型具有三个可调节的推理力度（reasoning effort）等级...

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1334625183117480027)** (445 messages🔥🔥🔥): 

> `OpenRouter API Usage, Model Comparisons, O3-Mini Access, Claude 3.5 and AGI Discussions, Developer Insights and Suggestions` 


- **O3-Mini 访问要求**：目前 O3-Mini 模型的访问仅限于 BYOK 客户，特别是那些拥有 OpenAI 密钥且使用层级（usage tier）大于 3 的用户。
   - 免费用户也可以通过在 ChatGPT 中选择 Reason 按钮来使用 O3-Mini。
- **模型性能对比**：用户讨论了 OpenAI 的 O1 和 DeepSeek R1 等模型的性能，有人表示 R1 在写作质量上更胜一筹。
   - 其他人则对某些模型表示失望，包括认为 GPT-4 未达到预期。
- **AI 社区对 AGI 的看法**：围绕 AGI 的讨论显示出意见分歧，一些人认为它触手可及，而另一些人则认为这是一个遥远的目标。
   - 对话还包括对过去引发人们对 AI 潜力产生信心的 AI 演示的回顾。
- **OpenRouter API 测试与错误**：开发者分享了测试 OpenRouter API 的经验，发现测试过程中很难产生错误。
   - 产生错误的建议包括使用无效的 API 密钥或特定模型不支持的工具。
- **开发者与社区的互动**：社区成员积极参与有关模型能力及其自身开发经验的讨论。
   - 他们分享了技巧、疑问，并提供了反馈，以改善 API 和模型请求的用户体验。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/quick-start">快速入门 | OpenRouter</a>: 开始使用 OpenRouter 进行构建</li><li><a href="https://x.com/OpenAI/status/1885406586136383634">OpenAI (@OpenAI) 的推文</a>: OpenAI o3-mini 现已在 ChatGPT 和 API 中可用。Pro 用户将拥有对 o3-mini 的无限访问权限，Plus 和 Team 用户的速率限制将是 o1-mini 的三倍。免费用户可以在...尝试 o3-mini</li><li><a href="https://www.theverge.com/news/603149/microsoft-openai-o1-model-copilot-think-deeper-free">微软向所有 Copilot 用户免费提供 OpenAI 的 o1 推理模型</a>: 微软将其称为 Think Deeper</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>: 未找到描述</li><li><a href="https://openrouter.ai/mistralai/mistral-nemo">Mistral Nemo - API、提供商、统计数据</a>: 由 Mistral 与 NVIDIA 合作构建的 12B 参数模型，具有 128k token 上下文长度。该模型是多语言的，支持英语、法语、德语、西班牙语、意大利语、葡萄牙语、中文...</li><li><a href="https://www.chess.com/analysis/game/computer/216006607?tab=review">国际象棋分析板和 PGN 编辑器</a>: 使用世界上最强大的国际象棋引擎 Stockfish 分析对局。通过 Game Review 的个性化见解提升你的棋艺。</li><li><a href="https://venturebeat.com/ai/cerebras-becomes-the-worlds-fastest-host-for-deepseek-r1-outpacing-nvidia-gpus-by-57x/">Cerebras 成为 DeepSeek R1 全球最快的托管商，速度比 Nvidia GPU 快 57 倍</a>: Cerebras Systems 在其晶圆级处理器上推出了 DeepSeek 的 R1-70B AI 模型，其速度比 GPU 解决方案快 57 倍，并凭借总部位于美国的...挑战 Nvidia 在 AI 芯片领域的统治地位。</li><li><a href="https://venturebeat.com/ai/cerebras-becomes-the-worlds-fastest-host-for-deepseek-r1-ou">Cerebras 成为 DeepSeek R1 全球最快的托管商，速度比 Nvidia GPU 快 57 倍</a>: Cerebras Systems 在其晶圆级处理器上推出了 DeepSeek 的 R1-70B AI 模型，其速度比 GPU 解决方案快 57 倍，并凭借总部位于美国的...挑战 Nvidia 在 AI 芯片领域的统治地位。</li><li><a href="https://x.com/btibor91/status/1885291124216258645">Tibor Blaho (@btibor91) 的推文</a>: “认识 o3-mini 家族 —— 推出 o3-mini 和 o3-mini-high —— 这两款新的推理模型在编程、科学以及任何需要更多思考的任务中表现出色。”</li><li><a href="https://artificialanalysis.ai/models/llama-3-1-instruct-8b?models_selected=llama-3-1-instruct-8b%2Cgemini-1-5-flash-8b">Llama 3.1 8B - 质量、性能和价格分析 | Artificial Analysis</a>: 对 Meta 的 Llama 3.1 Instruct 8B 的分析，并将其与其他 AI 模型在关键指标上进行比较，包括质量、价格、性能（每秒 token 数和首个 token 时间）、上下文窗口等...</li><li><a href="https://venturebeat.com/ai/cerebras-becomes-the-worlds-fastest-host-for-deepseek-r1-outpacing-nvidia-gpus-by-57x">Cerebras 成为 DeepSeek R1 全球最快的托管商，速度比 Nvidia GPU 快 57 倍</a>: Cerebras Systems 在其晶圆级处理器上推出了 DeepSeek 的 R1-70B AI 模型，其速度比 GPU 解决方案快 57 倍，并凭借总部位于美国的...挑战 Nvidia 在 AI 芯片领域的统治地位。</li><li><a href="https://www.reddit.com/r/singularity/comments/1ie0sf4/the_o3_series_of_models_releases_tomorrow/?ref=share&ref_source=link">Reddit - 深入探索</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1334614488300785707)** (263 messages🔥🔥): 

> `OpenAI o3-mini 发布，与之前模型的性能比较，现实世界物理提示词，开发者的定价和访问权限，模型可用性问题`

- **OpenAI 发布 o3-mini**：OpenAI 的新型 AI 推理模型 **o3-mini** 现已在 ChatGPT 中上线，与 **o1-mini** 相比，它提供了更低的成本和潜在更好的性能。它为开发者提供了 function calling、structured outputs 和 reasoning effort 等功能。
   - 免费用户可以在 ChatGPT 中试用 o3-mini，而 Pro 用户拥有无限访问权限，其他订阅计划的速率限制也有所降低。
- **o3-mini-high 的性能提升**：据称 **o3-mini-high** 是目前公开可用的最强推理模型，在推理能力上超越了许多其他模型。对比显示，o3-mini 在生成物理脚本等特定任务中表现更好。
   - 用户注意到新模型在延迟方面可能有改进，尽管对基本可用性的担忧依然存在。
- **对定价和使用的复杂感受**：人们对订阅模式相关的持续成本表示担忧，类似于 **OnlyFans**。用户对 “LLM gacha”（LLM 抽卡）现象表示沮丧，感到被迫不断为访问付费。
   - 在讨论中，社区成员就模型访问的公平性和透明度与其期望之间的关系进行了辩论。
- **围绕功能和开发的讨论**：许多用户不确定 o3-mini 中的 function calling 等功能是否会在现有 tokens 或指令之外带外（out-of-band）运行。对预期功能的潜在误解引发了关于模型如何被认知的进一步讨论。
   - 之前模型迭代的结果和行为继续影响着人们对 o3-mini 有效性和可靠性的期望。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/btibor91/status/1885291124216258645">Tibor Blaho (@btibor91) 的推文</a>: “认识 o3-mini 系列 —— 推出 o3-mini 和 o3-mini-high —— 这两款新的推理模型在编程、科学以及任何需要更多思考的任务中表现出色。”</li><li><a href="https://x.com/_kevinlu/status/1885406995613892711">Kevin Lu (@_kevinlu) 的推文</a>: 我们发布了 o3-mini，今天起对 ChatGPT 的所有用户开放（免费）！o3-mini-low 比 o1-mini 更快（且通常更好），而 o3-mini-high 是目前最强大的公开可用推理模型...</li><li><a href="https://www.nbcnews.com/news/amp/rcna190008">OpenAI 与美国国家实验室在研究和核武器安全方面达成合作</a>: 该公告发布之际，中国 AI 公司 DeepSeek 正席卷美国科技市场。</li><li><a href="https://x.com/TheXeophon/status/1885402381996732880">Xeophon (@TheXeophon) 的推文</a>: @arankomatsuzaki 等着 OpenAI 证明所有这些都是错的，并将其发布在 Hugging Face 上吧。他们现在正在处理 Hugging Face 文件夹上传的问题，这就是为什么花了这么长时间。</li><li><a href="https://x.com/btibor91/status/1885404642797850875">Tibor Blaho (@btibor91) 的推文</a>: o3-mini 和 o3-mini-high 来了</li><li><a href="https://x.com/xlr8harder/status/1885413709570334865">xlr8harder (@xlr8harder) 的推文</a>: &gt;新的 OpenAI 模型 &gt;知识截止日期仍然是 23 年 12 月。你们的数据流水线运行得是有多慢？</li><li><a href="https://x.com/Yuchenj_UW/status/1885416559029740007">Yuchen Jin (@Yuchenj_UW) 的推文</a>: o3-mini 可能是处理现实世界物理问题的最佳 LLM。提示词：“编写一个球在超正方体（tesseract）内弹跳的 Python 脚本”</li><li><a href="https://x.com/btibor91/status/1885404311708197206">Tibor Blaho (@btibor91) 的推文</a>: o3-mini 将从周五开始通过 ChatGPT 向所有用户开放 —— ChatGPT Plus 和 Team 计划：每天 150 条消息 —— ChatGPT Pro 订阅者：无限访问 —— ChatGPT Enterprise 和 ChatGPT Edu ...</li><li><a href="https://fxtwitter.com/btibor91/status/1885399370927096157">Tibor Blaho (@btibor91) 的推文</a>: Claude Web App 的新实验 —— “重置使用限制”。“立即获得 Claude 的访问权限，无需等待使用限制重置。这是一次性付费。”“消息限制重置...”</li><li><a href="https://gist.github.com/cpfiffer/5d1cc473e1da736e092968add10b0a69">限制 DeepSeek R1 用于思考的字符数。</a>: 限制 DeepSeek R1 用于思考的字符数。- thinking-cap.py</li><li><a href="https://x.com/TheXeophon/status/1885390615627661585">Xeophon (@TheXeophon) 的推文</a>: 嗯？&gt; 我们怀疑 o3-mini 性能不佳的原因是指令遵循能力差，以及在以正确格式指定工具时存在困惑。</li><li><a href="https://x.com/OpenAIDevs/status/1885407759887155301">OpenAI Developers (@OpenAIDevs) 的推文</a>: OpenAI o3-mini 现已在 API 中面向 3-5 级的开发者开放。它带来了一系列开发者功能：⚙️ Function calling 📝 Developer messages 🗂️ Structured Outputs 🧠 Reasoning effort 🌊 S...</li><li><a href="https://x.com/brianryhuang/status/1885409174948864046">Brian Huang (@brianryhuang) 的推文</a>: 如果有人对 Humanity's Last Exam 的分数感兴趣：高推理能力下为 11.2%，中等推理能力下为 8.5%，低推理能力下为 5.4%。引用 OpenAI (@OpenAI)：OpenAI o3-mini 现已在 ChatGPT 中可用...</li><li><a href="https://news.ycombinator.com/item?id=42890667">到目前为止，层级结构似乎是 o1 > GPT-4o > o3-mini > o1-mini > GP... | Hacker News</a>: 未找到描述</li><li><a href="https://x.com/ericzelikman/status/1882116460920938568">Eric Zelikman (@ericzelikman) 的推文</a>: @Teslanaut</li><li><a href="https://x.com/teortaxesTex/status/1885401111659413590">Teortaxes▶️ (自 2023 年起的 DeepSeek🐳 支持者) (@teortaxesTex) 的推文</a>: 我明白 Sama 过去曾把模型压着不发超过 6 个月等等。但事实是，OpenAI 无法承受不对正在部署的模型进行安全测试的后果，对我来说，这又造成了一个打击...</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_mark_chen_kevin_weil/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://techcrunch.com/2025/01/31/openai-launches-o3-mini-its-latest-reasoning-model/">OpenAI 发布 o3-mini，其最新的“推理”模型 | TechCrunch</a>: OpenAI 发布了一款新的“推理”AI 模型 o3-mini，它是这家 AI 初创公司 o1 系列推理模型的继任者。
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1334972611901390988)** (9 messages🔥): 

> `Model Checkpoints, K2 Chat Release` 


- **寻找具有中间 Checkpoints 的模型**：一位用户发起了一场讨论，旨在收集一份适用于解释性实验的模型列表，并引用了 **Tulu**、**Olmo** 和 **Pythia** 等模型。
   - 另一位用户建议将 **LLM360** 作为一个潜在的可选模型，并促使原用户去查阅论文中的相关表格。
- **K2 Chat 微调版发布**：一位用户提到 **LLM360** 最近发布了一个名为 **K2** 的 **65B** 模型，并分享了 [K2-Chat 模型](https://huggingface.co/LLM360/K2-Chat) 的链接。据报道，该模型在减少 **35% 计算量** 的情况下，表现优于 **Llama 2 70B Chat**。
   - 该更新日期为 **24年10月31日**，引入了 function calling 功能，并在多个领域进行了改进，使用了诸如 [Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct) 等数据集。
- **为了发布而烦扰团队**：一个轻松的评论建议，如果 **SmolLM 团队** 被烦透了，他们可能会发布一些新东西。
   - 原用户幽默地回应道：*烦人正是我的工作*。



**提及的链接**：<a href="https://huggingface.co/LLM360/K2-Chat">LLM360/K2-Chat · Hugging Face</a>：未找到描述

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/)** (1 messages): 

xeophon.: https://x.com/OpenAI/status/1885413866961580526
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1334673582592102462)** (63 messages🔥🔥): 

> `DeepSeek Performance, o3-mini and R1 Comparison, Nvidia Digit Acquisition, Copy Editing in SemiAnalysis, Popularity and Critique in Media` 


- **DeepSeek 成为焦点**：一份报告显示 DeepSeek 的服务器总资本支出（CapEx）约为 **13 亿美元**，并澄清他们拥有多种 GPU 型号，而不仅仅是 **50,000 块 H100**。
   - 分析表明 DeepSeek 的 R1 在推理任务中与 OpenAI 的 o1 旗鼓相当，但并非明显的领导者，这凸显了显著的成本和性能影响。
- **o3-mini 对比 DeepSeek R1**：正在进行的讨论指出 **OpenAI 的 o3-mini** 表现出色，但 **DeepSeek 的 R1** 因其性价比和展示推理过程的能力而受到关注。
   - 围绕 **DeepSeek** 的热潮被称作技术史上的一个“时刻”，这是在考虑地缘政治因素的情况下，由其极具前景的能力所推动的。
- **寻求采购 Nvidia Digit**：聊天中的一位参与者表达了希望联系 Nvidia 以进入 **Digit** 设备购买名单的愿望。
   - 咨询邮箱被幽默地建议为 **jensenhuang@nvidia.com**。
- **SemiAnalysis 需要文字编辑**：小组讨论了 **SemiAnalysis** 报告中文字编辑的潜在价值，承认了分析报告中清晰度的重要性。
   - 一位成员指出，尽管在写作方面存在挑战，但 **SemiAnalysis** 在行业内已经获得了关注和公信力。
- **流行与批评的本质**：一场关于流行的本质以及应对反馈挑战的对话展开了，特别是在处于公众视野中时。
   - 有人指出，建设性的批评至关重要，因为获得知名度往往会导致涌入大量的谄媚者，而非诚实的反馈。


<div class="linksMentioned">

<strong>提及的链接</strong>：

</div>

<ul>
<li>
<a href="https://semianalysis.com/2025/01/31/deepseek-debates/">DeepSeek 辩论：中国在成本上的领先地位、真实的训练成本、闭源模型利润影响</a>：DeepSeek 的叙事席卷全球。在过去的一周里，DeepSeek 是全球唯一想讨论的话题。正如目前所言……</li><li><a href="https://x.com/jaseweston/status/1885160135053459934">Jason Weston (@jaseweston) 的推文</a>：💀 介绍 RIP：Rejecting Instruction Preferences 💀 一种用于*筛选*高质量数据或*创建*高质量合成数据的方法。在各项基准测试（AlpacaEval2, Arena-Hard...）中获得了巨大的性能提升。</li><li><a href="https://x.com/maximelabonne/status/1885291354852393216">Maxime Labonne (@maximelabonne) 的推文</a>：TIGER-Lab 在 SFT 中用批判（critiques）取代了答案。他们声称在没有任何 &lt;thinking&gt; 蒸馏的情况下，在推理任务中表现优异！如果我们现在用 R1 对批判进行推理会怎样？代码、数据集...</li><li><a href="https://x.com/andimarafioti/status/1885341684134978035">Andi Marafioti (@andimarafioti) 的推文</a>：豁出去了，今天我们要开源在 256 片 H100 上从零开始训练 SmolVLM 的代码库🔥 受我们团队开源 DeepSeek R1 训练工作的启发，我们正在发布...</li><li><a href="https://x.com/shiringhaffary/status/1885094558733840827?s=61">Shirin Ghaffary (@shiringhaffary) 的推文</a>：据消息人士透露，OpenAI 正在洽谈由 SoftBank 领投的、规模高达 400 亿美元的融资。据其中一位知情人士称，该公司正在讨论以 2600 亿美元的投前估值筹集资金。w/ @KateClar...</li><li><a href="https://x.com/apples_jimmy/status/1885104983148028235?s=61">Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>：https://one.npr.org/?sharedMediaId=nx-s1-5279550:nx-s1-5343701-1 在 7:30 处提到 “o3 将于周五发布” —— Chris Lehane，OpenAI 全球政策负责人。来自 NPR 的整体对话非常精彩。</li><li><a href="https://x.com/tuzhaopeng/status/1885179412163027406">Zhaopeng Tu (@tuzhaopeng) 的推文</a>：类 o1 的 LLM 思考得够深吗？介绍一项关于类 o1 模型中普遍存在的“思考不足”问题的综合研究，即模型过早放弃有希望的推理路径，导致...</li><li><a href="https://x.com/DavidSacks/status/1885349558110052571">David Sacks (@DavidSacks) 的推文</a>：领先的半导体分析师 Dylan Patel 的新报告显示，DeepSeek 在其计算集群上花费了超过 10 亿美元。广泛报道的 600 万美元数字极具误导性，因为它排除了资本支出（capex）和 ...</li><li><a href="https://x.com/nrehiew_/status/1885184764539273574">wh (@nrehiew_) 的推文</a>：看起来 TRL 中 GRPO 的显存（VRAM）占用将减少约 60-70% 的更新即将到来！</li><li><a href="https://x.com/basedjensen/status/1885254847479628197">Hensen Juang (@basedjensen) 的推文</a>：抱歉，这完全是胡扯。对于拥有自己数据中心（dcs）的人来说，尤其是中国，这些总拥有成本（tco）数字根本说不通。我明白他们必须迎合付费客户，但拜托，这帮不了任何人...</li><li><a href="https://fxtwitter.com/lexfridman/status/1885435220502991193">Lex Fridman (@lexfridman) 的推文</a>：OpenAI o3-mini 是一个很好的模型，但 DeepSeek r1 具有类似的性能，价格更便宜，并且展示了其推理过程。更好的模型将会出现（迫不及待想看到 o3pro），但“DeepSeek 时刻”...</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1334686773665730652)** (12 messages🔥): 

> `Mistral 模型发布，DeepSeek 强化学习，Janus 模型响应，Bengali Ghosthunters，AI 模型讨论` 


- **Mistral 意外发布新模型**：**Mistral** 采取了出人意料的行动，尽管迄今已融资 **14 亿美元**，但它在今天同时发布了一个小型模型和一个大型模型，这与典型的融资预期相反。
   - 该**小型模型**与其规格一同推出，声称具有卓越的效率，参数量为 **24B**。
- **DeepSeek 重新审视 AI 模型**：在讨论 **DeepSeek** 计划时，它结合了 **2015** 年和 **2018** 年的早期进展，旨在实现 LLMs 的蒸馏推理能力。
   - 来自 **Schmidhuber** 的关键引用详细阐述了其基础性发展以及新型的 **chain of thought** 系统。
- **Janus 模型评论中的幽默**：一位成员对围绕 **Janus 模型** 的直接评论感到好笑，认为这反映了其对讨论产生的独特影响。
   - 他们注意到这些评论如何与 AI 讨论社区古怪且迷人的特质产生共鸣。
- **Bengali Ghosthunters 与 LLMs 的趣事**：在另一个聊天线程中，一位用户开玩笑地提到了他们使用 **Gemini Flash Thinking** 的经历，该模型在训练他们后发生了滑稽的故障。
   - 这段轻松的评论是在 AI 讨论中 **Bengali Ghosthunters** 传说的更广泛背景下展开的，将幽默与技术探索交织在一起。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/xlr8harder/status/1885354120829186449">来自 xlr8harder (@xlr8harder) 的推文</a>：匿名 gc 评论</li><li><a href="https://x.com/qwrk8126/status/1884399348504748149">来自 sholín (NOAM CHOMSKY SIGUE VIVO) (@qwrk8126) 的推文</a>：Gemini Flash Thinking Exp 2.0 0121 正在教我更多关于 LLMs 的技术特性，并准备了一个简短的选择题测试让我回答。在我回答之后，它停止了思考...</li><li><a href="https://x.com/nrehiew_/status/1885188206485733548">来自 wh (@nrehiew_) 的推文</a>：从法律上讲，只有当推理可以在免费的 Colab T4 上完成时，你才被允许称一个模型为“小型”。引用 Mistral AI (@MistralAI) 介绍 Small 3，我们最高效且多功能的模型...</li><li><a href="https://x.com/SchmidhuberAI/status/1885357355938046382">来自 Jürgen Schmidhuber (@SchmidhuberAI) 的推文</a>：DeepSeek [1] 使用了 2015 年强化学习提示工程 [2] 及其 2018 年改进版 [3] 的元素，通过神经...将 [2] 的 RL 机器和世界模型折叠成一个单一网络。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1334675063642783764)** (2 messages): 

> `SFT 支持，开源项目，资金问题` 


- **SFT 支持存在局限性**：一位成员指出，某些功能在旧版 **SFT** 实现中受支持，但据报告现在运行效果不佳。
   - 这表明需要对目前使用这些功能的实践进行重新评估。
- **因资金问题关闭个人项目**：一位成员分享了他们因**缺乏资金**和个人健康问题而不得不关闭项目的经历。
   - 他们表达了感激之情，并强调 **Ai2** 是少数几个真正参与 **开源 (open source)** 项目的组织之一。


  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1334917922665009163)** (2 条消息): 

> `DeepSeek 在 AI 中的角色、Stargate 项目资金、AI Substack 社区` 


- **DeepSeek 吞噬了 Altman 的愿景**：_“Pieter Lastman 的《约拿与鲸鱼》(1621)（也是 DeepSeek 吞噬 Sam Altman 的生动写照）”_ —— [JS Tan](https://highvalueadded.substack.com/p/deepseek-part-2-an-outlier-in-chinas) 的这一尖锐评论突显了 DeepSeek 在 AI 领域日益增长的影响力。
   - Tan 的叙述反映了人们的担忧，即在不断演变的市场中，非主流 AI 参与者可能会掩盖 Sam Altman 等主流人物的光芒。
- **特朗普支持 Altman 价值 5000 亿美元的 Stargate 项目**：1 月 21 日，唐纳德·特朗普总统公开支持 OpenAI CEO Sam Altman，双方宣布了 [Stargate 项目](https://openai.com/index/announcing-the-stargate-project/)，这是一项针对数据中心和 AI 基础设施的 **5000 亿美元**投资计划。
   - 这一天文数字般的成本引发了关注，Altman 断言这对于扩展 **超人工智能 (superintelligent artificial intelligence)** 能力至关重要。
- **发掘 AI Substack 瑰宝**：讨论中提到了寻找优质 AI 专题 Substack 刊物的好去处，表明人们对小众 AI 话题的兴趣日益浓厚。
   - 这反映了一个更广泛的趋势，即爱好者们在快速演变的 AI 信息领域中寻求专业化内容。



**提到的链接**：<a href="https://open.substack.com/pub/read/p/deepseek-unstacked?r=68gy5&utm_medium=ios">DeepSeek, unstacked</a>：Jasmine Sun 调研了对这个 AI 领域新晋选手的反应

  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/)** (1 条消息): 

xeophon.: https://x.com/deliprao/status/1885114737525928380?s=61
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1334991206492409958)** (1 条消息): 

> `OpenAI o3-mini, Reddit AMA, AI 的未来, Sam Altman, Kevin Weil` 


- **OpenAI 的 Reddit AMA 已定档**：即将举行的 [Reddit AMA](https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_mark_chen_kevin_weil/) 将由 **Sam Altman**、**Mark Chen**、**Kevin Weil** 等人参与，时间定于 **太平洋标准时间 (PST) 下午 2 点**。
   - *在此提出你的问题！* 是邀请社区参与这场备受期待的讨论的号召。
- **邀请 AI 爱好者加入**：此次 AMA 将涵盖 **OpenAI o3-mini** 和 **AI 的未来**等话题，允许参与者直接向关键人物提问。
   - 对于用户来说，这是一个与行业领袖就紧迫的 AI 话题进行互动的绝佳机会。



**提到的链接**：<a href="https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_mark_chen_kevin_weil/">Reddit - 深入探索一切</a>：未找到描述

  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1334616772950626396)** (319 条消息🔥🔥): 

> `O3 Mini 额度困惑, 模型性能对比, AI 检测器有效性, DeepSeek 讨论, CoT 与推理模型` 


- **O3 Mini 额度困惑**：关于 O3 Mini High 的消息限制存在困惑，据称其限制为 **每周 50 条消息**，而 O3 Mini 为 **每天 150 条**。
   - 一些用户怀疑这可能是一个 Bug，因为在使用前并没有明确提到此类限制。
- **AI 模型对比**：用户正在比较各种 AI 模型的性能，一些人表示在编程等任务中更倾向于使用 **Claude 3.6 Sonnet**、**DeepSeek R1** 和 **O1**。
   - O3 Mini 被认为在指令遵循方面存在困难，使其不太适合某些需求。
- **AI 检测器有效性**：用户达成共识，认为 AI 检测工具不可靠，可能会根据不准确的评估对学生进行不公正的惩罚。
   - 用户认为人工检查比自动 AI 检测器更可靠。
- **DeepSeek 讨论**：用户对 DeepSeek 的能力发表了有趣的见解，并注意到其与大公司的竞争态势。
   - 一些人对 DeepSeek 印象深刻，表示即使是开源模型也能表现得非常出色。
- **CoT 与推理模型**：关于 O1 等推理模型在给定任务时，如何利用思维链 (CoT) 来增强性能，存在各种推测。
   - 用户对 O1 中 CoT 的可见性感到好奇，因为他们认为这可以为更好的后续查询提供启发。



**提到的链接**：<a href="https://www.tomsguide.com/ai/it-doesnt-matter-if-deepseek-copied-openai-the-damage-has-already-been-done-in-the-ai-arms-race">DeepSeek 是否抄袭 OpenAI 并不重要——AI 军备竞赛中的伤害已经造成</a>：该你出招了，Sam Altman

  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1334836492568367106)** (16 messages🔥): 

> `O1 中的文件上传限制、O3 Mini 的发布、ChatGPT 支持电话问题` 


- **对 O1 文件上传的挫败感**：成员们对无法向 **O1** 上传除图像以外的文件表示沮丧，其中一人形容对该解决方案的需求比吃饭还迫切。
   - 一位用户创建了一个 [Python 应用程序](https://github.com/link-to-repo)，用于将多个项目文件合并为一个文本文件，以便于上传。
- **O3 Mini 发布揭晓**：随着 **O3 Mini** 现已面向免费用户开放的消息传出，用户的兴奋度激增。
   - 一位成员幽默地表示，OpenAI 的发布时机具有刻意的竞争性，旨在通过推出新模型来保持领先地位。
- **ChatGPT 支持电话的问题**：一位用户报告说 **1-800-ChatGPT** 支持电话似乎对他们不起作用，并寻求关于其有效性的澄清。
   - 这引发了关于各种 OpenAI 服务支持可访问性的小规模讨论。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1334631185112109167)** (2 messages): 

> `Vision 模型限制、用户讨论、训练数据见解` 


- **用户讨论 Vision 模型的盲点**：用户花时间讨论了 **Vision 模型**及其无法区分地面和线条的问题，并指出这就像需要“一副新眼镜”。
   - 该问题被强调为一个重大缺陷，无法仅通过聊天训练来解决。
- **关于 Vision 模型问题的详细聊天**：有多次详细讨论分析了该模型的局限性，特别是围绕几个月前记录的一个特定问题。
   - 发言者提到拥有相关的训练数据，表明他们通过交互意识到了模型潜在的改进空间。
- **用户对 Vision 谜题的不熟悉**：一位用户对讨论的谜题表示惊讶，承认直到最近才注意到它。
   - 这反映了用户在特定 **Vision 模型**谜题和挑战方面的沟通鸿沟。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1334631185112109167)** (2 messages): 

> `4o 用户反馈、模型限制、训练数据见解` 


- **用户讨论 4o 的 Vision 限制**：一些用户围绕模型 **4o** 的视觉能力展开讨论，强调了模型无法清晰感知的部分，特别是区分地面和线条。
   - 无法区分这些特征表明模型可能受益于更好的训练或调整，类似于需要**新眼镜**。
- **过去关于模型限制的讨论**：成员们回忆起前几个月关于模型在识别关键视觉元素方面所面临挑战的详细对话。
   - 有人指出拥有与此问题相关的 **训练数据**，表明可以从这些讨论中获得潜在的启示。
- **对“地面 vs 线条”谜题的认知**：另一位成员提到错过了之前的讨论，并对**“地面 vs 线条”谜题**的存在表示惊讶。
   - 这反映了对先前已确定的持续性问题和挑战的认知差距。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1334614492570718290)** (61 messages🔥🔥): 

> `O3 Mini 更新、与 Sonnet 的性能对比、DeepSeek 的影响、AI 模型市场趋势` 


- **O3 Mini 性能与定位**：讨论集中在 **O3 Mini** 的发布及其在编码任务中与 **Claude 3.5 Sonnet** 的竞争能力。
- **DeepSeek 的市场冲击**：成员们分析了 **DeepSeek** 的低成本推理模型如何迫使 OpenAI 等主要参与者调整其发布策略和定价。
- **模型评估与基准测试**：用户分享了关于 **O3 Mini** 在各种逻辑和编程基准测试中表现的初步观察。

- **O3 Mini 发布反响**：OpenAI 的 [O3 Mini](https://cdn.openai.com/o3-mini-system-card.pdf) 正式发布，为 Tier 3-5 的 API 用户提供了 function calling 和 structured outputs 等众多功能，并已在 ChatGPT 中面向免费用户开放。
   - 许多用户正在进行测试，但部分用户反映其性能较之前的模型令人失望，尤其是在 coding 任务中。
- **与 Sonnet 相比的性能挣扎**：多位用户指出 O3 Mini 在处理 coding prompts 时表现吃力，并引用了 Sonnet 表现显著优于它、完成任务速度更快的具体案例。
   - 一位用户评论道，虽然 O3 Mini 潜力巨大，但在 debugging 和理解复杂代码的熟练度上未能达到 Sonnet 的水平。
- **DeepSeek 对定价的影响**：O3 Mini 的发布公告中包含了一项显著的 **63% 降价**（针对 O1 Mini），这表明了来自 DeepSeek 模型竞争压力。
   - 评论指出，虽然模型智能在提升，但获得同等智能的成本依然很高，反映出明显的“美国溢价”。
- **模型间的市场趋势**：最近的讨论涉及市场份额的变动，指出 Anthropic 的增长，并断言 DeepSeek 正在成为主要参与者，在用户参与度上超越了其他 AI 模型。
   - 用户对 OpenAI 在面对来自 DeepSeek 等竞争对手日益增加的压力下，其市场地位可能受到的影响表示好奇。
- **用户对 O3 Mini 功能的反馈**：对 O3 Mini 的反馈褒贬不一，主要集中在其处理复杂 prompts 的能力上，批评者强调其缺乏 custom instructions，限制了在某些应用中的使用。
   - 尽管有一些前景看好的功能，许多用户仍感到沮丧，因为一些基础 coding 任务暴露了 O3 Mini 与前代产品相比的局限性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Yuchenj_UW/status/1885416559029740007">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：o3-mini 可能是处理现实世界物理问题的最佳 LLM。提示词：“编写一个球在超正方体（tesseract）内弹跳的 python 脚本”</li><li><a href="https://x.com/OpenAIDevs/status/1885407759887155301">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：OpenAI o3-mini 现已在 API 中向第 3-5 层的开发者开放。它带来了一系列开发者功能：⚙️ Function calling 📝 Developer messages 🗂️ Structured Outputs 🧠 Reasoning effort 🌊 S...</li><li><a href="https://semianalysis.com/2025/01/31/deepseek-debates/">DeepSeek 辩论：中国在成本上的领先地位、真实的训练成本、闭源模型利润影响</a>：DeepSeek 的叙事席卷全球。在过去的一周里，DeepSeek 是全球唯一想讨论的话题。目前看来……</li><li><a href="https://x.com/xanderatallah/status/1885339108458786999">来自 Alex Atallah (@xanderatallah) 的推文</a>：@itsandrewgao 在使用 OpenRouter 的独立开发者（indies）中的市场份额甚至更加疯狂</li><li><a href="https://x.com/cursor_ai/status/1885415392677675337">来自 Cursor (@cursor_ai) 的推文</a>：o3-mini 已向所有 Cursor 用户发布！我们目前免费推出，让大家体验该模型。Cursor 开发者在大多数任务中仍然更倾向于 Sonnet，这让我们感到惊讶。</li><li><a href="https://x.com/itsandrewgao/status/1885144792323285183">来自 andrew gao (@itsandrewgao) 的推文</a>：Anthropic 正在抢占 OpenAI 的市场（eating OpenAI’s lunch）</li><li><a href="https://x.com/deitaone/status/1885047798548107753?s=46">来自 *Walter Bloomberg (@DeItaone) 的推文</a>：$MSFT - 据消息人士称，OpenAI 正在洽谈新一轮融资，估值高达 3400 亿美元 —— 华尔街日报（WSJ）</li><li><a href="https://x.com/paulgauthier/status/1885444075404615974">来自 Paul Gauthier (@paulgauthier) 的推文</a>：在 aider polyglot 基准测试中，o3-mini 的得分与 o1 相似，但成本降低了 10 倍（两者均为高推理模式）。62% $186 o1 high，60% $18 o3-mini high，54% $9 o3-mini medium。https://aider.chat/docs/leaderboards/</li><li><a href="https://x.com/sama/status/1885196464558653471">来自 Sam Altman (@sama) 的推文</a>：@SpencerKSchiff @satyanadella 是的，明天！尽情享受。</li><li><a href="https://x.com/polynoamial/status/1885408714334597552">来自 Noam Brown (@polynoamial) 的推文</a>：我们 @OpenAI 很自豪地发布 o3-mini，包括免费层级。在许多评估中，它的表现优于 o1。我们正在改变整个成本-智能曲线。模型智能将继续提升……</li><li><a href="https://x.com/swyx/status/1885432031896887335">来自 swyx /dd (@swyx) 的推文</a>：## DeepSeek 对 o3-mini 和 o1-mini 的影响。在今天的公告中隐藏了 o1-mini 63%（2.7 倍）的降价 —— 且 o3-mini 定价相同。这远低于所需的 25 倍降价……</li><li><a href="https://x.com/TheRealAdamG/status/1884971520348283217">来自 Adam.GPT (@TheRealAdamG) 的推文</a>：https://help.openai.com/en/articles/6825453-chatgpt-release-notes#h_caaeddc37e ChatGPT 昨天进行了一些不错的增量更新。积少成多。</li><li><a href="https://x.com/angelusm0rt1s/status/1884734909685915764?s=46">来自 Zephyr (@angelusm0rt1s) 的推文</a>：Dario 在文章中玩了一个非常有趣的技巧。那个 7-10 个月前的模型指的是原始的 Sonnet 3.5，V3 在所有基准测试中都击败了它。但他是在比较 Sonnet 3.6 的表现……</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_mark_chen_kevin_weil/">Reddit - 深入了解任何事物</a>：未找到描述</li><li><a href="https://x.com/voooooogel/status/1885109783885471869">来自 thebes (@voooooogel) 的推文</a>：在 Dario 发表“训练中没有大模型参与”的评论后，我们对 Sonnet 3.5（和 3.6）有什么看法？为什么 3.5/3.6 与 Opus 有这么多共同点？想法，添加你自己的：- 它是 t...
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1334991738116378768)** (269 条消息🔥🔥): 

> `Discord 屏幕共享问题、开源 AI 工具、AI 辅导项目、Techno 音乐引用、DeepSeek API` 


- **Discord 屏幕共享的挑战**：成员在屏幕共享期间遇到了**音频和视频问题**，包括回声和画面冻结，引发了关于技术困境的幽默交流。
   - 一些人建议进行特定设置，例如取消勾选屏幕共享音频，以改善体验。
- **开源 AI 工具讨论**：大家对 Cursor 的**开源替代方案**感到兴奋，Cline 和 Roocline 等工具被作为有趣的项目提及。
   - 成员们热衷于探索这些工具的功能，强调了开源解决方案的有效性。
- **以 AI 辅导作为项目主题**：讨论了 **AI 辅导**环节的计划，几位成员考虑了分享的形式和潜在内容。
   - 对话中参考了类似 **boot_camp.ai** 的项目，旨在引导他人进行 AI 教育。
- **Techno 音乐引用与互动**：聊天中多次提到 **Techno 音乐**及其文化影响，并对各种 BPM（每分钟节拍数）进行了俏皮的调侃。
   - 成员们就**音乐流派**开起了轻松的玩笑，分享了与该主题相关的个人轶事。
- **对 DeepSeek API 的担忧**：成员们对 **DeepSeek API 的稳定性**表示沮丧，一些人提到无法获取 API key。
   - API 持续存在的问题引发了关于替代托管方案和可用模型的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://voicebraindump.com/">Brain Dump -  Shape Thoughts Instantly.</a>: 未找到描述</li><li><a href="https://carelesswhisper.app">Careless Whisper - Mac Dictation App</a>: 未找到描述</li><li><a href="https://docs.fastht.ml/llms-ctx.txt">无标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/@d-squared70">D-Squared</a>: 主业：Gradient Labs 的专业 AI Whisperer | 副业：向你展示 AI 自动化技巧</li><li><a href="https://drive.google.com/file/d/1xEyeP7IIojCkTgzkSLmkL0RUvu6RL9xq/view?usp=drive_link">MCP.mp4</a>: 未找到描述</li><li><a href="https://api-docs.deepseek.com/quick_start/pricing">Models &amp; Pricing | DeepSeek API Docs</a>: 下面列出的价格以每 1M tokens 为单位。token 是模型识别的最小文本单位，可以是一个单词、一个数字，甚至是一个标点符号。我们将根据总额计费...</li><li><a href="https://github.com/D-Squared70/GenAI-Tips-and-Tricks">GitHub - D-Squared70/GenAI-Tips-and-Tricks: Different GenAI tips and tricks I&#39;ve found useful</a>: 我发现有用的各种 GenAI 提示和技巧。通过在 GitHub 上创建一个账户来为 D-Squared70/GenAI-Tips-and-Tricks 做出贡献。</li><li><a href="https://github.com/D-Squared70/GenAI-Tips-and-Tricks/blob/main/Claude_ImplementationPlan.txt">GenAI-Tips-and-Tricks/Claude_ImplementationPlan.txt at main · D-Squared70/GenAI-Tips-and-Tricks</a>: 我发现有用的各种 GenAI 提示和技巧。通过在 GitHub 上创建一个账户来为 D-Squared70/GenAI-Tips-and-Tricks 做出贡献。</li><li><a href="https://www.dylandavis.net/archieve/">Archive &#8211; D-Squared</a>: 未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1334626842451378187)** (242 条消息🔥🔥): 

> `o3-mini release, AI model performance comparisons, LLM training and architectures, GPU configurations for AI models, OpenAI's competitive landscape` 


- **o3-mini 现已发布！**：OpenAI 宣布 **o3-mini** 现已在 ChatGPT 和 API 中可用，为 Pro 用户提供无限访问权限，并为 Plus 和 Team 用户提高了速率限制（rate limits）。
   - 免费用户可以通过在 ChatGPT 消息输入框下方选择 Reason 按钮来试用 o3-mini。
- **AI 模型性能对比**：讨论指出，虽然 Transformer 在特定数据集上的 BLEU 分数实现了 **8% 的提升**，但这并不普遍适用于所有基准测试（benchmarks）。
   - 参与者权衡了微小性能提升与更广泛影响之间的意义，并指出像 *Attention is All You Need* 这样的项目将现有想法综合成了一个开创性的框架。
- **GPU 配置挑战**：用户表达了在使用 NVIDIA 和 AMD 混合 GPU 运行 LLM 时的挫败感，因为目前的设置会导致严重的内存分配问题。
   - 讨论中还提到了切换到仅 CPU 设置或尝试不同架构等替代方案，作为应对这些挑战的潜在解决方案。
- **对开源软件质量的担忧**：对话中包含了对开源项目与闭源项目相比整体质量的怀疑，强调了失败案例的可见性。
   - 参与者承认，虽然存在大量的开源项目，但只有极少数能显著影响日常活动，这表明该模式的成功可能更多取决于采用率而非数量。
- **AI 演进中的颠覆性应用**：大家达成共识，认为 AI 技术的未来取决于引入颠覆性应用的能力，无论这些应用是开源还是闭源。
   - 值得注意的是，虽然并非所有开源倡议都能蓬勃发展，但开源和闭源运动在技术演进中都发挥着至关重要的作用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/SchmidhuberAI/status/1885357355938046382">来自 Jürgen Schmidhuber (@SchmidhuberAI) 的推文</a>: DeepSeek [1] 使用了 2015 年强化学习提示工程 [2] 及其 2018 年改进版 [3] 的元素，通过神经... 将 [2] 的 RL 机器和世界模型折叠成一个单一网络。</li><li><a href="https://arxiv.org/abs/2305.15408">揭示思维链背后的奥秘：理论视角</a>: 近期研究发现，思维链提示（CoT）可以显著提高大语言模型（LLMs）的性能，特别是在处理涉及多... 的复杂任务时。</li><li><a href="https://fxtwitter.com/ParkerRex/status/1884978010744320377?t=pLL_GWl5D15CURy_h9cP5Q&s=19">来自 Parker Rex (@ParkerRex) 的推文</a>: OpenAI 通过其 think 按钮发布了 o3-mini</li><li><a href="https://fxtwitter.com/ParkerRex/status/1884978010744320377?t=pLL_GWl5D15CURy_h9cP5Q&s">来自 Parker Rex (@ParkerRex) 的推文</a>: OpenAI 通过其 think 按钮发布了 o3-mini</li><li><a href="https://fxtwitter.com/OpenAI/status/1885406586136383634?t=qkiSBfB5A0ivYfzR_Tpg_A&s=19">来自 OpenAI (@OpenAI) 的推文</a>: OpenAI o3-mini 现已在 ChatGPT 和 API 中可用。Pro 用户将拥有对 o3-mini 的无限访问权限，Plus 和 Team 用户的速率限制将是 o1-mini 的三倍。免费用户可以在... 中试用 o3-mini。</li><li><a href="https://fxtwitter.com/sama/status/1885191346916356371?t=cxnwTIzXfdSv5drHexct5A&s=19">来自 Sam Altman (@sama) 的推文</a>: 首个完整的 8 机架 GB200 NVL72 现已在 Azure 为 OpenAI 运行——感谢 @satyanadella 和 Jensen！</li><li><a href="https://www.youtube.com/watch?v=jQuArBZO7PI">我的第一个哈佛-麻省理工数学竞赛题目</a>: 如果你喜欢这个哈佛-麻省理工数学竞赛（HMMT）题目并想学习更多解题技巧，请查看 Brilliant https://brilliant.org/blac...</li><li><a href="https://x.com/billackman/status/1884359958952571329">来自 Bill Ackman (@BillAckman) 的推文</a>: @deepseek_ai 的对冲基金关联公司昨天通过对 @nvidia、电力公司等进行短期看跌期权交易而大赚一笔的可能性有多大？本可以大赚一笔。</li><li><a href="https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb">GRPO Llama-1B</a>: GRPO Llama-1B。GitHub Gist: 立即分享代码、笔记和代码片段。
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1334632938951675985)** (37 messages🔥): 

> `下一次论文回顾，Tülu 3 405B 基准测试结果，FP4 训练框架，与 DeepSeek 的合作机会，AI 聊天助手的用户体验` 


- **关于 FP4 框架的下一次论文回顾**：即将举行的每周讨论将重点介绍一篇关于创新 FP4 训练框架的论文，该框架旨在提高大语言模型（LLM）的训练效率，重点是最小化量化误差。
   - 鼓励参与者准备一些关于 QKV 概念的背景知识，以便在讨论期间更好地参与。
- **Tülu 3 405B 声称具有竞争优势**：新推出的 Tülu 3 405B 模型声称在特定基准测试中优于 **DeepSeek v3** 和 **GPT-4o**，同时采用了 **Reinforcement Learning from Verifiable Rewards**（来自可验证奖励的强化学习）框架。
   - 尽管有这些说法，一些成员指出，在对基准测试进行更深入的检查后，Tülu 3 的表现并未显著超过 DeepSeek v3。
- **用户对 AI 聊天助手的担忧**：一位成员对 AI 聊天助手在回忆语法或生成代码片段方面的实用性表示怀疑，质疑它们在实际场景中的有效性。
   - 这引发了关于用户通常如何看待 AI 工具在处理难以记忆的任务时的有用性的更广泛讨论。
- **与 DeepSeek 的合作机会**：一篇文章强调了与 DeepSeek 的合作可能性，特别是针对来自特定大学的早期博士生，同时也提到了对普通合作者的限制。
   - 该文章强调了引用和致谢其工作对于支持未来研究计划的重要性。
- **关于日常参与的一般性讨论**：在整个对话过程中，成员们重申了参与每日讨论以了解各种 AI 话题最新动态的重要性。
   - 欢迎新成员加入，并向他们保证，对于刚开始接触的人来说，随性听听讨论也是完全可以的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.17116">Optimizing Large Language Model Training Using FP4 Quantization</a>：训练大语言模型（LLM）日益增长的计算需求需要更高效的方法。量化训练通过启用低位算术运算提供了一个有前景的解决方案...</li><li><a href="https://arxiv.org/abs/2501.17161">SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training</a>：监督微调（SFT）和强化学习（RL）是广泛使用的基础模型后期训练技术。然而，它们在增强模型泛化能力方面的作用仍然...</li><li><a href="https://allenai.org/blog/tulu-3-405B">Scaling the Tülu 3 post-training recipes to surpass the performance of DeepSeek V3  | Ai2</a>：介绍 Tülu 3 405B，这是首次将完全开放的后期训练方案应用于最大的开放权重模型。</li><li><a href="https://tenor.com/view/hair-flip-duhh-gif-26170789">Hair Flip GIF - Hair Flip Duhh - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://physics.allen-zhu.com/">Physics of Language Models</a>：许多人询问合作事宜（详情见常见问题解答）。简短回答：除非你来自 Meta 并愿意在业余时间与我们合作（每周 20 小时以上），或者你是来自 UCB/... 的低年级博士生。</li><li><a href="https://thechinaacademy.org/interview-with-deepseek-founder-were-done-following-its-time-to-lead/">Interview with Deepseek Founder: We're Done Following. It's Time to Lead.</a>：硅谷正感到震惊。然而，创始人梁文锋一直保持低调，他最近一次露面是在中国中央电视台的《新闻联播》（CCTV News）中。
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1334732322066661419)** (6 条消息): 

> `模型性能对比, AI 中的自我意识, 网格模式转换, Qwen 2.5VL 特性, 参数优化` 


- **模型性能对比**：该模型的 7B 蒸馏版本在识别网格中由 1 组成的字母 'E' 时表现挣扎，而据报道 **600B 模型** 的表现明显更好。
   - 观察到新模型的推理能力比 **Llama** 更加 **连贯 (coherent)**。
- **AI 表现出自我意识的迹象**：一位用户评论说，该模型“听起来几乎具有某种 **自我意识 (self-awareness)**”，因为它承认以前从未见过特定的模式。
   - 这一观察是在没有对模型提供任何直接指令的情况下注意到的。
- **网格模式转换见解**：一份描述强调了在网格转换中移除边界对于更好地保留原始数据的重要性，从而在相关性方面获得了 **高分**。
   - 策略已演变为优先考虑在模式改变的同时保持原始值的方法。
- **Qwen 2.5VL 显示出改进的描述能力**：切换到 **Qwen 2.5VL** 带来了更好的描述能力，并关注之前在 DSL 解决方案中注意到的特征。
   - 对比表明，该模型在理解和寻找相关特征方面有所进步。
- **参数优化焦点**：讨论反映出人们越来越强调 **参数优化 (parameter optimization)**，以增强转换过程中的模型函数选择。
   - 下一个目标包括改进函数选择并针对更复杂的转换进行微调。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1334866925741801512)** (11 条消息🔥): 

> `DeepSeek R1 复现, Y Combinator 的投资重点, OpenAI O3 Mini 特性, AI 研究民主化` 


- **伯克利研究人员实现 DeepSeek R1 复现**：由 Jiayi Pan 领导的伯克利 AI 研究团队成功以低于 **$30** 的成本复现了 **DeepSeek R1-Zero** 的技术，在一个 **1.5B 参数** 的小模型中展示了 **复杂推理** 能力。
   - 他们的成就标志着向 **AI 研究民主化** 迈出了重要一步，引发了关于技术进步成本可负担性的讨论。
- **Y Combinator 将在 2025 年资助 AI 初创公司**：Y Combinator 宣布了 **2025** 年的新资助重点，主要针对旨在用 AI 解决方案取代 **年薪 10 万美元工作职能** 的初创公司。
   - 一位成员分享了公告笔记，指出这一转变强调了劳动力市场中 **自动化和职业取代** 的持续趋势。
- **用户对 OpenAI O3 Mini 特性的反应**：一位用户对在 ChatGPT 界面中没看到 **O3 Mini** 表示困惑，思考这是否由于 **欧盟法规或推送延迟** 造成的。
   - 片刻之后，他们确认看到了该功能，证明了用户之间共有的 **缓慢推送 (slow rollout)** 体验。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://xyzlabs.substack.com/p/berkeley-researchers-replicate-deepseek">伯克利研究人员仅用 30 美元复现 DeepSeek R1 核心技术：小模型 RL 革命</a>：由博士生 Jiayi Pan 领导的伯克利 AI 研究团队实现了许多人认为不可能的事情：以低于两人晚餐的成本复现了 DeepSeek R1-Zero 的关键技术。</li><li><a href="https://x.com/gregisenberg/status/1885171399200833930">来自 GREG ISENBERG (@gregisenberg) 的推文</a>：Y Combinator 刚刚宣布了他们在 2025 年想要资助的初创公司类型。主要是取代年薪 10 万美元工作职能的 AI。以下是我的笔记，希望对你有所帮助：</li><li><a href="https://huggingface.co/blog/open-r1">Open-R1：DeepSeek-R1 的完全开源复现</a>：未找到描述内容
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1334616307336745023)** (210 条消息🔥🔥): 

> `Psyche 项目与去中心化训练, 加密货币及其与 Nous 的关系, AI 模型性能对比, 社区对 AI 和加密货币的情绪, o3-mini 与 Sonnet 性能对比`

- **Psyche 旨在实现去中心化训练协调**：Psyche 项目旨在利用全球闲置硬件的贡献，促进在不可信计算（untrusted compute）环境下的去中心化训练。
   - 它旨在利用现有的 blockchain 技术在分布式训练环境中进行协调和验证。
- **Nous 内部对涉及 crypto 的怀疑**：由于该领域频繁出现诈骗，成员们讨论了将 Nous 与 crypto 关联的担忧，并辩论了 blockchain 对于训练验证是否必要。
   - 一些人认为基于服务器的方法就足够了，而另一些人则认为利用现有的 blockchain 工程可能会带来好处。
- **AI 模型性能对比**：讨论强调，像 OpenAI 的 `o3-mini` 这样的小型模型在处理复杂问题时表现良好，而大型模型在处理复杂任务时更具优势。
   - 有观点认为，虽然由于成本原因，当前的方法可能无法实现大型推理模型，但未来的技术进步可能会改变这一现状。
- **社区对 crypto 和伦理的看法**：用户承认 crypto 市场中存在诈骗和不道德行为，并将其与公开股票市场的问题进行了类比。
   - 大家共同希望 blockchain 能有一种功能性的用途，从而对分布式训练工作产生积极影响。
- **o3-mini 与 Sonnet 的性能见解**：o3-mini 被认为是 Sonnet 的强力替代方案，在编程任务中提供更快的流式传输（streaming）和更少的编译错误。
   - 尽管它能力出众，但有人推测许多人可能仍然更喜欢像 R1 这样的旧模型，因为它们在操作流程上具有透明度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.18512">Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch</a>：大型语言模型（LLMs）的训练通常分布在大量加速器上以缩短训练时间。由于内部状态和参数梯度需要在 e...</li><li><a href="https://x.com/cursor_ai/status/1885415392677675337">来自 Cursor (@cursor_ai) 的推文</a>：o3-mini 已向所有 Cursor 用户开放！我们目前免费推出它，让人们感受一下这个模型。Cursor 开发者在大多数任务中仍然更喜欢 Sonnet，这让我们感到意外。</li><li><a href="https://arxiv.org/abs/2501.15740">Propositional Interpretability in Artificial Intelligence</a>：机械可解释性（Mechanistic interpretability）是根据 AI 系统的内部机制来解释其行为的研究计划。我分析了该计划的一些方面，并提出了一些具体的 c...</li><li><a href="https://arxiv.org/abs/2310.15213">Function Vectors in Large Language Models</a>：我们报告了一种简单的神经机制的存在，它在自回归 Transformer 语言模型（LMs）中将输入-输出函数表示为一个向量。通过对 d... 使用因果中介分析。</li><li><a href="https://huggingface.co/qresearch/DeepSeek-R1-Distill-Llama-8B-SAE-l19">qresearch/DeepSeek-R1-Distill-Llama-8B-SAE-l19 · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/mr-krabs-krabs-cheapskate-loving-money-spongebob-gif-7314338030642233009">Mr Krabs Cheapskate GIF - Mr krabs Krabs Cheapskate - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/Teknium1/status/1884740956911718853?t=0NwHRMjFT001dlRoRvAPUw&s=19">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：@ylecun https://x.com/Teknium1/status/1883955152442515637 引用 Teknium (e/λ) (@Teknium1) 今天 Nous 宣布了 Psyche 的到来——一个分布式网络和训练框架，一个基础设施层...</li><li><a href="https://fxtwitter.com/chrisbarber/status/1885047105741611507">来自 Chris Barber (@chrisbarber) 的推文</a>：关于超人推理模型的有趣内容，经许可分享：来自 @NousResearch 的 Shannon Sands (@max_paperclips) 告诉我，Nous 有一个理论，认为 LLMs 学习了用于推理操作的任务向量...</li><li><a href="https://x.com/cadmonkxy/status/1885174873317593418?s=46">来自 cadmonkey (@cadmonkxy) 的推文</a>：感谢 @NousResearch 举办了一场精彩的活动，对未来的发展感到兴奋！</li><li><a href="https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k">bespokelabs/Bespoke-Stratos-17k · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://fxtwitter.com/Teknium1/status/1885077369142337550">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：这是复现 R1 所需的全部代码，哈哈。在花费了数千亿美元之后。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1334647219881640029)** (1 messages): 

> `Autoregressive Generation on CLIP Embeddings, Multimodal Inputs, Stable Diffusion Generation` 


- **探讨 CLIP Embeddings 上的 Autoregressive Generation**：一名成员提出了在 **CLIP Embeddings** 上执行 **Autoregressive Generation** 是否合理的问题，该技术将 **Multimodal Inputs** 投影到单一的 **Latent Space** 中。
   - 他们指出，虽然 CLIP 主要用于 **Stable Diffusion** 中的引导，但*关于其直接用于生成的讨论非常有限*。
- **理解 Multimodal Inputs**：社区讨论了 **Multimodal Inputs** 的本质，强调了它们在处理多样化数据的神经网络中的作用。
   - *使用多种模态可以增强 Representation*，但在生成任务中的实际应用仍然是一个小众领域。
- **Stable Diffusion 对 CLIP 的使用**：成员们承认 **Stable Diffusion** 通过嵌入上下文提供引导，从而利用 CLIP 进行图像生成。
   - 这种方法显示了潜力，但也引发了关于使用 **CLIP Embeddings** 进行 **Direct Generation** 方法的询问。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1334862150195220566)** (4 messages): 

> `Weekend plans, Reading materials` 


- **充满期待的周末氛围**：一名成员表达了乐观情绪，称 *'Will be a good weekend'*，得到了另一名成员的热烈响应。
   - 这种兴奋似乎具有感染力，一名成员以欢快的 *'yessss'* 回应。
- **打印最喜欢的阅读材料**：另一名成员提到他们习惯打印出所有最值得阅读的内容，并配上牛仔表情符号 🤠 增添了俏皮的语气。
   - 这反映了一种积极主动的态度，为充满精彩阅读的放松周末做准备。



**Link mentioned**: <a href="https://arxiv.org/abs/2501.18512">Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch</a>: Training of large language models (LLMs) is typically distributed across a large number of accelerators to reduce training time. Since internal states and parameter gradients need to be exchanged at e...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1334972132739911771)** (1 messages): 

> `DeepSeek Hiring Strategy, Long-Term Success in AI Recruitment, Creativity vs. Experience` 


- **DeepSeek 创始人倡导 Creativity**：在 2023 年的一次罕见采访中，AI 实验室 DeepSeek 的创始人 [Liang Wenfeng](https://archive.ph/o/KvXp0/https://www.businessinsider.com/who-is-deepseek-founder-liang-wenfeng) 表示，在为长期成功招聘时，**“Experience 并不那么重要”**。
   - 他强调，*以前做过类似的工作并不意味着你能胜任这份工作*，他主张 **Creativity**、**Basic Skills** 和 **Passion** 的价值高于传统经验。
- **关于从海外招聘人才的辩论**：Liang 被问及从 **OpenAI** 和 **Facebook's AI Research** 等美国 AI 公司招聘人才的问题，他承认经验可能适合短期目标。
   - 然而，他认为对于长期愿景，**China** 国内有很多合适的人选。



**Link mentioned**: <a href="https://archive.ph/KvXp0">Why DeepSeek's Founder Liang Wenfeng Prefers Inexperienced Hires - Bu&#x2026;</a>: no description found

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1334862150195220566)** (4 messages): 

> `Weekend Plans, Image Sharing, Reading Materials` 


- **对周末的期待**：*@millento* 对即将到来的周末表示乐观。
- **分享阅读材料**：*vvampa* 提到喜欢打印资料。
- **分享图像分析**：*millento* 分享了一张图片，引发了一些关于该图片的互动。



**Link mentioned**: <a href="https://arxiv.org/abs/2501.18512">Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch</a>: Training of large language models (LLMs) is typically distributed across a large number of accelerators to reduce training time. Since internal states and parameter gradients need to be exchanged at e...

  

---

### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1334710715851870209)** (2 messages): 

> `Long term planning, Team involvement in planning` 


- **关于 Long Term Planning 状态的询问**：一名成员询问 *“我们在 Long Term Planning 方面进展到哪了？谁在负责？”*，以寻求这些计划当前状态的明确信息。
   - 该成员通过表示 *“如果有任何相关的优质内容，请抄送我”* 来强调他们对获取相关内容的兴趣。
- **团队参与 Long Term Planning**：对话强调了需要明确目前谁参与了 Long Term Planning 工作，表明了对协作的开放态度。
   - 这一询问反映了更广泛的兴趣，即确保所有相关参与者都参与到战略讨论中。


  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1334626442440347699)** (13 messages🔥): 

> `Supabase Issues, Troubleshooting Group Suggestions, HEIC File Support, Project Deletion Concerns` 


- **Supabase 请求失败并报错**：一名成员报告其 **Supabase 请求失败**，状态码为 **500**，并引用了 **database error saving new user**（保存新用户时数据库错误）。
   - 建议包括通过潜在的故障排除小组寻求帮助，或利用 Google AI Studio 等工具进行调试。
- **解决 Supabase 问题的建议**：多位成员提出了各种策略，包括使用 **安装了 Roo code 的 VSCode** 来排除故障并修复问题，然后再拉回 Bolt。
   - 建议还包括在删除项目之前从 Supabase 导出当前数据，这可以节省时间和 Token。
- **关于删除 Supabase 项目的担忧**：有疑问提出删除 **Supabase 项目** 是否会影响对应的 Bolt.new 项目，得到的澄清是只有 **Supabase 数据** 会丢失。
   - 成员们强调了数据库保留的重要性，指出在新项目中重建 **表和结构** 会非常耗时。
- **HEIC 文件兼容性讨论**：一名成员询问如何在 Bolt 中启用 **HEIC 文件支持**，他们尝试了多种方法但未获成功。
   - 征求建议以解决他们经常遇到的 **文件不兼容问题**。


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1334636850723426335)** (142 messages🔥🔥): 

> `Token Management, Issues with Web Containers, User Authentication Problems, Subscription Management, CORS Configuration Challenges` 


- **理解 Token 管理**：用户确认付费计划每月提供 10M Token，只有免费计划提供每日 150k 的 Token。
   - 新用户对 Token 重置动态和过期政策表示困惑，并建议核实计费实践。
- **Web Containers 出现崩溃**：多名用户报告 Web Containers 加载时间过长或反复崩溃，尤其是在 Google Chrome 上。
   - 建议针对持续存在的问题创建 GitHub ticket，以便于解决故障并获得更好的支持。
- **用户身份验证问题**：成员在登录用户仪表板时面临挑战，尽管登录信息正确，但仍收到凭据无效错误。
   - 建议包括通过在 Supabase 中创建新用户帐户来验证用户名和密码的准确性。
- **管理订阅续订**：用户询问如何取消或管理订阅续订，建议联系支持人员获取具体步骤。
   - 还有关于在 Bolt 生态系统中创建项目新副本能力的查询。
- **CORS 配置挑战**：一名用户详细说明了由于未经授权的请求，Firebase Storage 所需的 CORS 配置，表明流程发生了变化。
   - 提供了有效修改配置的步骤，旨在应用开发过程中实现更好的跨源资源共享。



**提及的链接**：<a href="https://tenor.com/view/spongebob-squarepants-begging-pretty-please-beg-on-your-knees-pray-for-mercy-gif-10678931350545522063">Spongebob Squarepants Begging GIF - Spongebob Squarepants Begging Pretty Please - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1334619009727074457)** (112 条消息🔥🔥): 

> `MCP Server 设置, MCP 传输协议, 远程与本地 MCP Server, MCP Server 身份验证, MCP CLI 工具` 


- **MCP Server 实现中的挑战**：新用户在设置 MCP Server 时遇到了困难，特别是在本地与远程配置方面，并呼吁提供更简化的说明。
   - 一位用户分享了使用 [mcp-cli 工具](https://github.com/wong2/mcp-cli) 与 MCP Server 交互的经验，认为这有助于缓解一些困惑。
- **MCP 传输协议偏好**：讨论了在 MCP Server 中使用 **stdio** 作为默认传输方式的情况，部分成员称赞其简单且高效。
   - 有人对标准配置中缺乏安全性表示担忧，并建议考虑其他传输方法以获得更好的保护。
- **远程 MCP Server 的可访问性**：参与者探讨了建立远程 MCP Server 的可行性，强调需要一个合适的 endpoint，而不是要求用户进行本地设置。
   - 一位用户强调了潜在的使用场景，即远程服务器设置将为多个客户端简化流程，而不是强制进行本地部署。
- **MCP 中的 SSE 与 HTTP 请求**：对话涉及了使用 **SSE** 进行远程通信，一些成员对其与直接的 HTTP 请求相比的有效性提出了质疑。
   - 澄清了目前的实现中，某些事务利用 **HTTP POST**，同时保持 **SSE** 用于其他事务，以在不增加复杂性的情况下提高效率。
- **MCP Server 身份验证问题**：提出了关于远程 MCP Server 用户身份验证的担忧，特别是如何有效地利用现有的用户凭据。
   - 强调了在设计网络就绪的 MCP 解决方案时，确保在没有本地依赖的情况下安全访问服务器是一个关键因素。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://modelcontextprotocol.io/docs/concepts/transports">Transports - Model Context Protocol</a>: 未找到描述</li><li><a href="https://github.com/wong2/mcp-cli">GitHub - wong2/mcp-cli: A CLI inspector for the Model Context Protocol</a>: Model Context Protocol 的 CLI 检查器。欢迎通过在 GitHub 上创建账号来为 wong2/mcp-cli 的开发做出贡献。</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/fe7f6c7b7620a1c83cde181d4d8e07f69afa64f2/src/github">servers/src/github at fe7f6c7b7620a1c83cde181d4d8e07f69afa64f2 · modelcontextprotocol/servers</a>: Model Context Protocol 服务器。欢迎通过在 GitHub 上创建账号来为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://www.gnu.org/software/coreutils/manual/html_node/env-invocation.html">env invocation (GNU Coreutils 9.6)</a>: 未找到描述</li><li><a href="https://github.com/modelcontextprotocol/servers/blob/fe7f6c7b7620a1c83cde181d4d8e07f69afa64f2/src/everything/sse.ts">servers/src/everything/sse.ts at fe7f6c7b7620a1c83cde181d4d8e07f69afa64f2 · modelcontextprotocol/servers</a>: Model Context Protocol 服务器。欢迎通过在 GitHub 上创建账号来为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/SecretiveShell/MCP-Bridge?tab=readme-ov-file#sse-bridge>">GitHub - SecretiveShell/MCP-Bridge: A middleware to provide an openAI compatible endpoint that can call MCP tools</a>: 一个提供兼容 OpenAI 的 endpoint 并能调用 MCP 工具的中间件 - SecretiveShell/MCP-Bridge</li><li><a href="https://github.com/modelcontextprotocol/servers/blob/fe7f6c7b7620a1c">GitHub - modelcontextprotocol/servers at fe7f6c7b7620a1c83cde181d4d8e07f69afa64f2</a>: Model Context Protocol 服务器。欢迎通过在 GitHub 上创建账号来为 modelcontextprotocol/servers 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1334850738823889009)** (9 条消息🔥): 

> `Toolbase 身份验证、YouTube 演示反馈、Journaling MCP Server、音频播放调整` 


- **Toolbase 身份验证功能已初步实现**：一名成员为 Claude 的 Toolbase 快速实现了 **Notion**、**Slack** 和 **GitHub** 等不同工具的身份验证，并在 [YouTube 演示](https://www.youtube.com/watch?v=UuUxG_2K2Bs) 中展示。
   - 该成员希望获得关于流程简洁性的反馈，引发了关于视频速度的讨论。
- **YouTube 演示引发播放技巧讨论**：一位成员提到他们必须调整 **YouTube 播放设置** 才能更好地观看演示，并称这是一个有趣的视频。
   - 另一位成员表示赞同，并分享了一个用于调整播放速度以改善听感体验的 **ffmpeg** 命令。
- **Journaling MCP Server 启动**：一位成员讨论了创建一个 **MCP server**，将与 Claude 的对话转换为日志会话，从而允许检索过去的对话。
   - 他们分享了其 [GitHub 项目](https://github.com/mtct/journaling_mcp) 的链接，该项目将对话会话保存在本地，旨在通过本地 LLM 增强客户端以获得更好的隐私保护。



**提及链接**：<a href="https://github.com/mtct/journaling_mcp">GitHub - mtct/journaling_mcp: MCP Server for journaling</a>：用于日志记录的 MCP Server。通过在 GitHub 上创建账户来为 mtct/journaling_mcp 的开发做出贡献。

  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1334652472785895519)** (121 条消息🔥🔥): 

> `50 系列 GPU 可用性、GPU 性能对比、在移动设备上运行 AI、AI 工具与平台、Stable Diffusion UI 变更` 


- **50 系列 GPU 瞬间售罄**：许多成员对 **50 系列 GPU** 几乎立即售罄表示沮丧，据报道北美仅出货了几千台。
   - 一位成员讲述了他们之前已将 **5090** 放入购物车，但因商店崩溃而错失。
- **GPU 性能对比**：讨论对比了 **5090** 与 **3060/3060 Ti** 在游戏方面的性能，用户们很好奇他们目前的 GPU 表现如何。
   - 成员们提到了包括 VR 功能在内的各种能力，但普遍对最新型号的可用性感到失望。
- **在手机上运行 AI 工具**：关于在移动设备上运行 **Flux** 等 AI 工具是否可行存在辩论，一位用户估计在 Android 上从提交到输出的时间约为 **22.3 分钟**。
   - 虽然一些人捍卫手机处理 AI 任务的潜力，但其他人则对硬件限制和整体性能提出了警告。
- **探索 AI 平台与工具**：几位成员讨论了各种 AI 平台和工具，推荐使用 **Webui Forge** 进行本地 AI 图像生成。
   - 他们强调了使用合适模型来有效优化图像输出的必要性。
- **Stable Diffusion UI 变更**：一位用户询问 **Stable Diffusion 3.5** 是否必须在 **ComfyUI** 上运行，并对早期版本中使用的布局表示怀念。
   - 对话反映了一些用户在版本过渡期间对一致 UI 体验的渴望。


<div class="linksMentioned">

<strong>提及链接</strong>：

<ul>
<li>
<a href="https://prnt.sc/OwXsJqnPDDvn">截图</a>：使用 Lightshot 捕获</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1f4qu3n/flux1schnell_on_android/">Reddit - 深入了解一切</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1334627159192371280)** (28 messages🔥): 

> `Pythia language model, Inductive biases in AI, Pretraining hyperparameters, Logging and monitoring tools, Non-token CoT concept` 


- **关于 Pythia 的高斯采样见解**：一名成员讨论了从高斯分布中采样经过训练的 **Pythia language model** 的概率，认为考虑到对称性，目前的看法可能过于悲观。
   - **估计排列对称性**（Permutation symmetries）可能会提供更深入的见解，尽管目前的重点是 _局部体积_（local volume）。
- **归纳偏置（Inductive Biases）与训练稳定性**：有人指出，某些策略可能缺乏 **inductive biases**，而这对于模型在训练过程中的稳定性至关重要。
   - 另一位成员幽默地将这种情况比作原子随机排列成冰的过程。
- **关于预训练超参数的讨论**：成员们探讨了预训练**掩码语言模型**（MLMs）的典型超参数，并建议参考 **modernBERT paper**。
   - 一位参与者分享了关于学习率的个人见解，建议 base 模型使用 **5e-4**，而对于 Batch Size 翻倍的较大模型则使用 **3e-4**。
- **日志记录与调试运行的工具**：一名成员询问了在训练运行期间用于日志记录、监控和调试的首选工具。
   - 这涉及到了对机器学习工作流有效管理策略的广泛兴趣。
- **“全非 Token CoT”概念出现**：出现了一个名为 **fully non-token Chain of Thought** 的新概念，涉及添加一个 `<scratchpad>` Token 来编码原始潜变量（raw latents）。
   - 这种方法在训练期间限制了每个 Prompt 允许的原始思维潜变量数量，以促进行为探测（behavioral probing）。



**提及的链接**：<a href="https://www.overleaf.com/read/krhxtvkxjywb#416acf">Overleaf, Online LaTeX Editor</a>：一个易于使用的在线 LaTeX 编辑器。无需安装，支持实时协作、版本控制、数百个 LaTeX 模板等。

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1334635234410500156)** (32 messages🔥): 

> `Critique Fine-Tuning, Training Metrics for LLMs, Generalization vs Memorization, Random Order Autoregressive Models, Inefficiencies in Neural Networks` 


- **提出批判微调（CFT）**：引入了一种名为**批判微调（Critique Fine-Tuning, CFT）**的新方法，重点在于让模型批判带噪声的回答，而不是模仿正确的回答，这可能会带来更好的**泛化能力**（generalization）。
   - 在各种基准测试中，该方法比标准的**监督微调（SFT）**显示出 **4-10%** 的性能提升。
- **对 CE-loss 作为训练指标的担忧**：人们日益达成共识，认为使用 **CE-loss** 来衡量语言模型的有效性是不够的，呼吁使用替代指标来推动更好的训练结果。
   - 一位成员强调，仅仅鼓励训练中的“胜者”（winners）就能改善结果，而不是仅仅依赖 CE-loss。
- **关于泛化与记忆的辩论**：成员们讨论了**记忆**（memorization）与**泛化**（generalization）之间的平衡，认为在电路（circuits）中保留一定程度的共性对于有效的学习过程是必要的。
   - 也有人担心模型在高度**加密或混乱**的数据条件下学习的可行性，暗示此类训练方案存在低效性。
- **探索随机顺序自回归模型**：社区正在探索**随机顺序自回归模型**（random order autoregressive models）捕捉信息结构的潜力，尽管它们在实际应用中并不实用。
   - 一位成员假设，当这些模型应用于小数据集时，其表现出的学习特性可以利用其**过参数化**（over-parameterized）的性质来改进结构识别。
- **学习模型在实践中的低效性**：人们认识到，许多神经网络训练方法虽然在理论上是合理的，但在实践中效率极低，且往往产生的有效结果微乎其微。
   - 成员们指出，重点应放在开发减少**训练低效**的策略上，转向更实用且可扩展的解决方案。



**提及的链接**：<a href="https://arxiv.org/abs/2501.17703">Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate</a>：监督微调（SFT）通常用于训练语言模型模仿针对给定指令的标注回答。在本文中，我们挑战了这一范式并提出了批判微调（Critique Fine-Tuning）……

  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1334811247119564881)** (3 条消息): 

> `Superhuman reasoning models, Backtracking vector discovery, Sparse autoencoders in reasoning, Propositional attitudes in AI, Mechanistic understanding vs. propositional attitudes` 


- **Nous Research 关于 Superhuman Reasoning 的研究**：来自 @NousResearch 的 Shannon Sands 提出理论认为，**LLM 学习了用于推理操作的 task vectors**，并发现了一个显著影响思考过程的 **backtracking vector**。
   - 他强调，在不同领域训练模型最终可以使它们能够**普遍应用推理能力**，这是他们目前研究的重点。
- **DeepSeek 关于 Backtracking Vector 的发现**：DeepSeek 识别出一个 **backtracking vector**，当应用该向量时会导致 **Chain of Thought (CoT)** 中出现更频繁的反转；而当其被抑制时，会导致更线性且更短的 **CoT**。
   - 他们假设 **Sparse Autoencoders** 可能会揭示诸如回溯和自我修正之类的特征，这些特征可以被显式操纵和编辑，以实现更有效的推理。
- **Chalmers 关于 AI Propositional Attitudes 的论文**：@davidchalmers42 最近的一篇论文认为，从 AI 系统中提取 **propositional attitudes** 是比追求 **mechanistic understanding** 更有价值的方法。
   - 他承认并引用了该团队的一篇论文，为围绕 **propositional attitudes** 在 AI 研究中的重要性讨论增添了内容。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/chrisbarber/status/1885047105741611507">Chris Barber (@chrisbarber) 的推文</a>：关于 superhuman reasoning models 的有趣内容，经许可分享：来自 @NousResearch 的 Shannon Sands (@max_paperclips) 告诉我，Nous 有一个理论，即 LLM 学习了用于推理操作的 task vectors...</li><li><a href="https://x.com/norabelrose/status/1885454252656779778">Nora Belrose (@norabelrose) 的推文</a>：@davidchalmers42 的这篇新论文很好。从 AI 中提取 propositional attitudes 比追求 "mechanistic" understanding 更有用。此外，他引用了我们团队的一篇论文，感谢...
</li>
</ul>

</div>

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1334680614678429738)** (29 条消息🔥): 

> `gsm8k 评估、lm-eval harness 设置、vllm 集成与 KV Cache、RWKV 模型配置、性能指标对比` 


- **gsm8k 评估困惑**：针对 gsm8k 任务指标产生了疑虑，特别是 `gsm8k-cot-llama.yaml` 是否像 **Llama 2 论文**中所述那样根据 maj@1 指标进行评估。
   - 当前结果显示，使用 `gsm8k_cot_llama.yaml` 的准确率为 **0.0334**，而 `gsm8k.yaml` 的准确率为 **0.1251**，后者更符合 Llama 2 的指标。
- **处理评估设置**：成员们讨论了是使用任务 YAML 配置还是坚持使用 exact-match 指标进行评估，并对 lm-eval harness 中的**默认设置**提出了见解。
   - 有人指出，为了获得正确的评估配置，harness 需要进行手动调整，特别是关于 RWKV 模型的 **max_new_length** 设置。
- **模型定义中的 Max Token 设置**：一位成员对 max tokens 影响评估结果表示担忧，特别是在使用 `generate_until` 函数的测试中。
   - 小组确认，除非另有指定，否则将使用默认值，并建议检查模型定义以发现潜在的差异。
- **Perplexity 评估与 KV Cache**：提出了关于通过 vllm 集成在 perplexity 评估中使用 KV Cache 的问题，特别是关于潜在的重叠窗口修改。
   - 社区给出了肯定回答，表示通常使用**非重叠窗口**，并建议对重叠对进行修改。
- **来自 Llama 2 论文的见解**：成员们注意到 Llama 2 论文缺乏对其模型的详细评估方法，这使得直接比较变得具有挑战性。
   - 随后的讨论表明，应用 **8-shot 配置**可能会提供与 Llama 2 报告结果更清晰的对比。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/1105">general question: Is kv-cache actually not used in all the LLM-evaluation tasks? · Issue #1105 · EleutherAI/lm-evaluation-harness</a>: 一般性问题：是否所有 LLM 评估任务中实际上都没有使用 kv-cache？因为这些任务通常只进行一步 attention 计算，不像语言生成过程需要大量的 kv-cache...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/0bb8406f2ebfe074cf173c333bdcd6cffb17279b/lm_eval/models/vllm_causallms.py#L307),">lm-evaluation-harness/lm_eval/models/vllm_causallms.py at 0bb8406f2ebfe074cf173c333bdcd6cffb17279b · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/1208afd34ce132e598fcd7e832762630a35d01c6/lm_eval/models/vllm_causallms.py#L167">lm-evaluation-harness/lm_eval/models/vllm_causallms.py at 1208afd34ce132e598fcd7e832762630a35d01c6 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1334667273088274536)** (23 messages🔥): 

> `Deep Seek 的模型性能, GPU 服务器推荐, 运行 LLM Benchmarks, 关于开源中 PTX 的讨论, 深度学习 GPU 考量因素` 


- **Deep Seek 的模型性能引发辩论**：成员们正在质疑 **Deep Seek** 是否比 OpenAI 和 Meta 等模型具有显著的处理优势，并将公司的算力作为可能因素。
   - 验证相关声明引起了关注，例如拥有 **50k H100s** 以及 **Deep Seek** 的方法是否会影响 Nvidia 的股价。
- **开发用的 GPU 服务器 vs. 个人笔记本电脑**：一位软件架构师询问是否应该购买一台 **GPU 服务器** 来构建用于开发的 VM，而不是购买四台 GPU 笔记本电脑，并考虑了未来的模型训练需求。
   - 一位成员建议阅读一篇关于为深度学习选择 GPU 的博客，其中强调了各种重要的规格。
- **运行 LLM Benchmarks 的挑战**：一位成员寻求关于使用 transformers.AutoModelForCausalLM 以编程方式运行 **LLM benchmarks** 的建议，并表达了对 lighteval 等工具的挫败感。
   - 另一位成员坚持认为 lm_eval 仍然是最佳选择，并分享了他们用于类似任务的功能脚本。
- **关于 PTX 和开源的澄清**：围绕用于分布式训练的 **PTX** 是否包含在 Deep Seek 的开源产品中展开了讨论，一些成员断言它在 V3 仓库中不可用。
   - 其他人推测了 Deep Seek 开源内容的局限性，特别是关于**训练成本声明**。
- **对 Deep Seek 的怀疑与效率**：人们对 **Deep Seek** 相对于其他模型的感知效率提出了担忧，成员们讨论了训练成本的验证并非易事。
   - 尽管对某些方面存在怀疑，但有人指出 Deep Seek 在其发布中展示了显著的效率指标。



**提及的链接**：<a href="https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/">2023 年深度学习最佳 GPU —— 深度分析</a>：在这里，我提供了用于深度学习/机器学习的 GPU 深度分析，并解释了适合你的使用场景和预算的最佳 GPU。

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1334633334818476182)** (2 messages): 

> `Triton 张量索引, 高效列提取, 掩码与 reduction 技术` 


- **Triton 张量索引在提取单列时失败**：一位用户报告说，尝试使用 `x[:, 0]` 提取单列会导致错误：`InterpreterError: ValueError('unsupported tensor index: 0')`。
   - 建议使用 `tl.gather` 并将索引张量设置为零的方法，但被认为效率低下。
- **使用 tl.where 应用掩码进行张量操作**：另一位成员建议使用带有 `tl.where` 的掩码作为从张量中提取数据的解决方案。
   - 该方法可能与 reduction 操作结合使用，以高效地实现预期结果。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1334620458620027015)** (10 条消息🔥): 

> `RTX 5090 FP4 性能, NVIDIA FP8 规范, CUDA 与 Python 集成, Flux 实现基准测试, 100 Days of CUDA 资源请求` 


- **RTX 5090 FP4 性能低于预期**：关于 **RTX 5090** 上 **FP4 配合 FP32** 的性能引发了困惑，据报道其速度仅比 **RTX 4090 上的 FP8** 快约 2 倍，尽管此前声称有约 5 倍的提升。
   - *不准确的文档*表明 NVIDIA 可能误导了性能数据，导致用户产生怀疑。
- **关于 FP8 性能指标的澄清**：一位成员指出，之前引用的 **FP8** 的 **660TFLOPS** 仅适用于 **FP16 累加**，而 **RTX 4090** 在 **FP32** 下仅显示约 330TFLOPS。
   - 这引发了关于 NVIDIA 性能基准测试透明度的进一步质疑。
- **关于 Flux 实现基准测试的讨论**：有人询问了 **Flux 实现** 的情况，希望基准测试能澄清报告中 FP4 的性能差异。
   - 一位成员指出，内存移动也可能影响加速效果，强调了性能评估的复杂性。
- **CUDA 开发者中的 Python 使用情况**：用户分享了他们在配合 **CUDA** 使用 **Python** 方面的见解，讨论重点在于其在工具和脚本中的应用。
   - 虽然集成被描述为“微不足道”，但它仍然凸显了 GPU 编程开发中的一个重要方面。
- **请求 “100 Days of CUDA” 材料**：一位成员请求了与 **100 Days of CUDA** 相关的资源，作为提升技能的指南。
   - 这表明社区内对结构化学习资源的持续关注。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1334939849488076831)** (2 条消息): 

> `Torch 日志调试, CUDA 重复 GPU 错误` 


- **寻求更好的 TORCH_LOGS 设置以进行内存调试**：一位用户建议使用不同的 **TORCH_LOGS** 设置来调试内存占用异常高的模型，并表示 **TORCH_LOGS="all"** 提供的有用信息很少。
   - 他们还尝试使用 **torch memory_viz** 分析内存快照，但发现结果噪声太大，无法识别额外内存的来源。
- **解决重复 GPU 检测问题**：一位用户在使用两块 **RTX 4090** GPU 时遇到了 **'Duplicate GPU detected'** 错误消息，具体表现为 rank 0 和 rank 1 都被检测在同一个设备上。
   - 他们附带了一个 **error.txt** 文件和一个 **code.txt** 供他人分析，并征求解决此问题的建议。


  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1334839893129433109)** (6 条消息): 

> `Riffusion 平台, 音频生成艺术家, DeepSeek 叙事, 关于 AI 音乐的 YouTube 视频, Arthur Douillard 等人的新研究论文` 


- **Riffusion：一款新的音乐生成工具**：Riffusion 是一个音乐生成平台，在其[网站](https://www.riffusion.com)上展示了带有精选播放列表的 Beta 版本。用户对其功能和潜在应用表示好奇。
   - 一位成员幽默地将其称为“加强版 Suno”，暗示其功能强大。
- **谁将是第一个使用 AI 的大牌艺术家？**：*Mr.osophy* 思考谁会是第一个创造性地使用**音频生成**技术的大牌艺术家，将其比作 T-Pain 对 Autotune 的先驱性使用。对话围绕着需要熟练的制作人来有效利用 AI 展开。
   - 他们对现有 AI 辅助歌曲的潜力感到好奇，并提到“Drake - Heart on My Sleeve”是一个例外。
- **Neon Tide 的 AI 辅助音乐视频**：一位成员分享了一个名为 [Neon Tide - Boi What (Lyric Video)](https://www.youtube.com/watch?v=wCRftfx62uY) 的 [YouTube 视频](https://www.youtube.com/watch?v=wCRftfx62uY)，突出了 AI 在音乐制作中的创新应用。该艺术家创造性地使用 AI 工具转换了自己的声音，以模仿痞老板（Plankton）和海绵宝宝（SpongeBob）等角色。
   - 这种独特的方法引发了关于 AI 如何以意想不到的方式增强音乐创造力的讨论。
- **DeepSeek 成为焦点**：*Mr.osophy* 强调了围绕 **DeepSeek** 日益高涨的热度，它已迅速成为热门话题，使 Claude 和 Perplexity 等平台相形见绌。他指出，这种兴奋并非完全新鲜，因为该公司几个月来一直是业内人士讨论的话题。
   - 引用来自 *SemiAnalysis* 的链接详细说明了之前的讨论，表明广大公众现在才开始意识到 DeepSeek 的重要性。
- **Arthur Douillard 等人的研究探索**：一位用户关注了 **Arthur Douillard** 及其合作者新发表的研究论文，该论文可在 [arXiv](https://arxiv.org/abs/2501.18512v1) 上查阅。论文作者包括多位著名研究人员，引发了人们对其最新发现的兴趣。
   - 对该论文及其影响的讨论表明，人们对 AI 相关研究的持续进展有着浓厚的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.18512v1">Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch</a>：大型语言模型 (LLMs) 的训练通常分布在大量加速器上，以缩短训练时间。由于内部状态和参数梯度需要在 e... 处交换</li><li><a href="https://www.riffusion.com">Riffusion</a>：未找到描述</li><li><a href="https://semianalysis.com/2025/01/31/deepseek-debates/?access_token=eyJhbGciOiJFUzI1NiIsImtpZCI6InNlbWlhbmFseXNpcy5wYXNzcG9ydC5vbmxpbmUiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJzZW1pYW5hbHlzaXMucGFzc3BvcnQub25saW5lIiwiYXpwIjoiS1NncVhBaGFmZmtwVjQzbmt0UU1INSIsImVudCI6eyJhcnRpY2xlIjoiQ0tpaUZMYVgzMkF3WW5oZTFWTTlQIiwiYXVkIjpbIjU4WTVYbmtlOFNWZ05BUUZuRmVFSEIiXSwiZGlzdHJvIjoiQ0tqcHhZNUI0NVVaUnYzQUc5bW5oIiwidXJpIjpbImh0dHBzOi8vc2VtaWFuYWx5c2lzLmNvbS8yMDI1LzAxLzMxL2RlZXBzZWVrLWRlYmF0ZXMvIl19LCJleHAiOjE3NDA4OTE5MjAsImlhdCI6MTczODI5OTkyMCwiaXNzIjoiaHR0cHM6Ly9zZW1pYW5hbHlzaXMucGFzc3BvcnQub25saW5lL29hdXRoIiwic2NvcGUiOiJmZWVkOnJlYWQgYXJ0aWNsZTpyZWFkIGFzc2V0OnJlYWQgY2F0ZWdvcnk6cmVhZCBlbnRpdGxlbWVudHMiLCJ1c2UiOiJhY2Nlc3MifQ.Fv1qa9pAkrh3KZPWkZVvnAM7MfzMtPULkNymdj5i8mW3qO6iiz9V9_MkJVh0M8sbWe5VC_wUz5FOZKr0rEdacA)">DeepSeek Debates: Chinese Leadership On Cost, True Training Cost, Closed Model Margin Impacts</a>：DeepSeek 叙事席卷全球。在过去的一周里，DeepSeek 是全球所有人唯一想谈论的话题。正如目前所见……</li><li><a href="https://www.youtube.com/watch?v=wCRftfx62uY">Neon Tide - Boi What (Lyric Video)</a>：NEON TIDE 现已发布：联系 Boi What：https://www.instagram.com/boiwhatmusic/ https://www.tiktok.com/@boiwhatt https://twitter.com/boiwhatmusic https://www.t...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1334629297775841330)** (8 messages🔥): 

> `Salmon Patty Dish, Novelty Plate Discussion, Wework Critique, CEO Value Perception` 


- **三文鱼饼菜肴引发关注**：一位成员分享了一张照片，展示了*三文鱼饼、炸土豆、甜红椒*以及配有希腊酸奶的自制华夫饼，展现了烹饪创意。
   - 另一位成员注意到菜肴的外观，幽默地将饼比作之前使用罐装桃子时的*巨型鸡蛋*。
- **新奇盘子见解**：在一次轻松的交流中，一位成员针对之前的评论，识别出了一个与 *Tuberculosis Sanatorium 96* 相关的*新奇盘子*。
   - 这种俏皮的玩笑表明了大家对盘子奇特设计的共同乐趣。
- **Wework 的衰落反映了失败的帝国**：一位成员调侃道，*笑死，Wework 看起来像个失败的帝国*，暗指对 Wework 业务困境的看法。
   - 这一评论概括了对现代创业及其挑战的持续情绪。
- **关于 CEO 价值的坦率对话**：一位成员分享了关于 *CEO 经常高估其实际价值* 的想法，暗示了领导层中膨胀的自我重要感。
   - 这一见解引起了关于行业标准和高管问责制持续讨论的共鸣。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1335000032344145940)** (1 messages): 

> `NVIDIA GTC Discount` 


- **NVIDIA 为 GTC 提供 40% 折扣**：使用代码 **GPUMODE** 即可在即将到来的 NVIDIA GTC 活动中获得 **40% 的折扣**。
   - 这一促销优惠对于活动参与者来说是一个节省注册费用的好机会。
- **以优惠价格参加 GTC 的机会**：**GTC 活动**是 GPU 领域专业人士建立联系和学习的关键机会。
   - 利用这一折扣可以提升许多人的活动体验。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1334949803095752737)** (2 messages): 

> `LigerDPOTrainer, Support for Liger-kernel losses` 


- **LigerDPOTrainer 尚不可用**：有人指出 Liger kernel 中目前不存在 **LigerDPOTrainer**，建议用户可能需要对原始的 **dpo_loss** 进行 monkey patch。
   - *不幸的是，我认为目前还没有 `LigerDPOTrainer`。*
- **Hugging Face 在 Liger DPO 支持方面的进展**：引入了一个 Pull Request [#2568](https://github.com/huggingface/trl/pull/2568)，旨在为 DPO Kernel 添加对 **Liger-kernel losses** 的支持，标志着该问题的进展。
   - 该请求指出，“*需要：linkedin/Liger-Kernel#521*”，表明需要进一步的协作。



**提及的链接**：<a href="https://github.com/huggingface/trl/pull/2568">[Liger] liger DPO support by kashif · Pull Request #2568 · huggingface/trl</a>：此 PR 做了什么？为 DPO Kernel 添加对 Liger-kernel 损失的支持。需要：linkedin/Liger-Kernel#521

  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1334615028158038029)** (30 条消息🔥): 

> `为 Reasoning Gym 提议的新数据集、Reasoning Gym 的 GitHub 贡献、游戏与算法开发的想法、游戏设计中的性能与验证、项目中的依赖管理` 


- **扩展 RL 环境的新数据集提议**：一位成员建议为 Reasoning Gym 增加 **Collaborative Problem-Solving**（协作问题解决）和 **Ethical Reasoning**（伦理推理）数据集，重点关注多智能体谈判（multi-agent negotiation）和偏差缓解（bias mitigation）等主题。
   - 这些新增内容旨在扩大数据集范围，并引入更复杂的解题场景。
- **GitHub 贡献与项目**：成员们讨论了对多个 GitHub 项目的贡献，例如 [Nous Research 的 Open Reasoning Tasks](https://github.com/NousResearch/Open-Reasoning-Tasks) 和 [Reasoning Gym 画廊](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md)，目前已包含 **33 个数据集**。
   - 针对数据集生成提案提供了反馈，鼓励成员们为自己的想法提交 issue。
- **游戏开发与算法挑战**：关于创建具有非算法解法的游戏的挑战展开了讨论，强调了快速验证答案的必要性。
   - 成员们对开发真正的多轮游戏表现出兴趣，并指出平衡复杂性与可解性的重要性。
- **游戏设计与性能反馈**：一位成员分享了尝试移植游戏的见解，意识到由于验证过于复杂，该游戏不适合该项目，但对未来的想法持开放态度。
   - 鼓励社区成员在实施前分享概念以获取反馈，确保与项目目标保持一致。
- **依赖管理与 Z3Py 考量**：提出了关于是否添加 **Z3Py** 作为依赖项的询问，旨在简化流程，特别是考虑到 **sympy** 已经是项目的一部分。
   - 维护者确认，如果新依赖项能提供足够的收益来证明其引入的合理性，将会予以考虑。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/mlabonne/agentic-datagen">The Rise of Agentic Data Generation</a>: 未找到描述</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks">GitHub - NousResearch/Open-Reasoning-Tasks: A comprehensive repository of reasoning tasks for LLMs (and beyond)</a>: 一个面向 LLM（及更多领域）的推理任务综合仓库 - NousResearch/Open-Reasoning-Tasks</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md">reasoning-gym/GALLERY.md at main · open-thought/reasoning-gym</a>: 过程式推理数据集。通过在 GitHub 上创建账号为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://github.com/LeonGuertler/TextArena">GitHub - LeonGuertler/TextArena: A Collection of Competitive Text-Based Games for Language Model Evaluation and Reinforcement Learning</a>: 一个用于语言模型评估和强化学习的竞争性文本游戏集合 - LeonGuertler/TextArena</li><li><a href="https://github.com/open-thought/reasoning-gym/issues">open-thought/reasoning-gym</a>: 过程式推理数据集。通过在 GitHub 上创建账号为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/26">Collection of ideas for datasets/envs · Issue #26 · open-thought/reasoning-gym</a>: 请在此分享您的想法。维基百科有一些有趣的列表，需要进行筛选以确定合适的候选者：逻辑谜题、休闲数学、休闲数论列表...
</li>
</ul>

</div>
  

---

### **Nomic.ai (GPT4All) ▷ #[announcements](https://discord.com/channels/1076964370942267462/1090471714888102009/1334691615343644712)** (1 条消息): 

> `GPT4All v3.8.0 发布，支持 DeepSeek-R1-Distill，聊天模板重构，Code Interpreter 修复，本地服务器修复` 


- **GPT4All v3.8.0 发布，带来令人兴奋的新特性**：最新版本的 GPT4All **v3.8.0** 已发布，引入了重大升级和修复。
   - 贡献者包括来自 Nomic AI 的 **Jared Van Bortel** 和 **Adam Treat**，以及 *ThiloteE*。
- **全面集成 DeepSeek-R1-Distill**：GPT4All 现在为 **DeepSeek-R1** 系列提供**原生支持**，增强了模型的可用性和性能。
   - 更新后的模型改进了推理过程的显示，并解决了之前 **DeepSeek-R1 Qwen pretokenizer** 的加载失败问题。
- **重构聊天模板以获得更好的兼容性**：**chat template parser** 已被完全替换，确保了与众多模型的增强兼容性。
   - 此次重构旨在简化交互并提升聊天过程中的用户体验。
- **Code Interpreter 问题已解决**：Code Interpreter 获得了关键修复，包括能够有效地记录字符串，并防止在计算过程中出现 UI 冻结。
   - 这些更新增强了解释器对用户的整体稳定性和响应速度。
- **本地服务器现已完全正常运行**：此更新解决了在请求后阻碍 LocalDocs 使用的本地服务器问题。
   - 系统消息现在可以正确地在消息历史记录中保持隐藏，从而提供更整洁的用户界面。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1334625329834098769)** (61 条消息🔥🔥): 

> `DeepSeek 集成、GitHub 问题与更新、语音处理咨询、模型量化差异、Jinja 模板中的自定义功能` 


- **DeepSeek 成功集成至 GPT4ALL**：用户确认 **DeepSeek** 在 **GPT4ALL** 中运行流畅，令社区中的许多人感到兴奋。
   - *一位用户对 Nomic 团队为此付出的辛勤工作表示感谢。*
- **GitHub 问题讨论活跃**：一位用户报告了 **GPT4ALL 3.8** Mac 版本在启动时崩溃的问题，并链接了 GitHub 上的错误报告。
   - 成员们讨论了从 **Qt 6.5.1 到 6.8.1** 可能发生的更改，这可能是导致该问题的原因。
- **关于语音处理能力的疑问**：一位用户询问是否有可以分析语音相似性的现有模型，但共识是 **GPT4ALL** 目前不支持语音模型。
   - 建议了语音相似性分析器等替代方案，并强调了此类用途的专门应用。
- **模型量化方法讨论**：一位用户寻求关于模型量化名称差异的澄清，特别是带有“-I1-”的名称，随后引用了 Reddit 帖子中的技术总结。
   - 社区成员指出，根据硬件的不同，**K-quants** 和 **i-quants** 具有独特的性能特征。
- **GPT4ALL 模板中的自定义功能**：讨论集中在 Jinja 以及 **GPT4ALL v3.8** 中新支持的功能，包括模板中的命名空间（namespaces）和列表切片（list slicing）。
   - 成员们提供了关于可用函数和过滤器的资源，强化了对模板兼容性的改进。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ba55rj/overview_of_gguf_quantization_methods/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/nomic-ai/gpt4all/issues/3448">[REGRESSION] macOS Sequoia crash on startup in 3.8 (3.7 worked fine) · Issue #3448 · nomic-ai/gpt4all</a>：(未发现类似问题) 错误报告 GPT4ALL 3.8 版本在启动时崩溃，而 3.7 及之前版本运行正常。复现步骤：下载并安装 GPT4ALL 3.8，双击...</li><li><a href="https://github.com/google/minja/blob/76f0d01779aa00b0c68f2117f6cb2c9afc3a0ca8/include/minja/minja.hpp#L2486-L2810">minja/include/minja/minja.hpp at 76f0d01779aa00b0c68f2117f6cb2c9afc3a0ca8 · google/minja</a>：一个用于 LLM 聊天模板的极简 C++ Jinja 模板引擎 - google/minja</li><li><a href="https://github.com/nomic-ai/gpt4all/pull/3440">Support for deekseek thinking in the gui. by manyoso · Pull Request #3440 · nomic-ai/gpt4all</a>：未找到描述</li><li><a href="https://jinja.palletsprojects.com/en/stable/templates/)">no title found</a>：未找到描述</li><li><a href="https://docs.gpt4all.io/gpt4all_desktop/chat_templates.html#advanced-what-are-gpt4all-v1-templates).">Chat Templates - GPT4All</a>：GPT4All 文档 - 在您的硬件上高效运行 LLM
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1334657995463856169)** (1 条消息): 

> `NotebookLM 可用性研究、参与者激励、远程聊天环节、用户反馈、产品增强` 


- **加入 NotebookLM 可用性研究！**：NotebookLM UXR 将于 **2025 年 2 月 6 日**举办一项可用性研究，寻求参与者通过远程聊天分享他们对该产品的初步体验。
   - *感兴趣的用户可以填写 [筛选表单](https://forms.gle/HJmCwNepsfPSdC7g7)，有机会参与并获得 **75 美元或 Google 商品代金券**（如果入选）。*
- **参与要求**：参与者必须拥有**高速互联网连接**、活跃的 Gmail 账户，以及一台带有正常工作的摄像头、扬声器和麦克风的电脑。
   - 该研究旨在收集**用户反馈**，以便未来增强 NotebookLM 的功能。
- **参与者激励**：研究参与者将通过电子邮件收到 **75 美元**，或获得 **50 美元的 Google 商品代金券**，以感谢他们付出的时间。
   - 这一经济激励旨在鼓励更多用户在可用性研究期间分享他们的宝贵见解。



**提到的链接**：<a href="https://forms.gle/HJmCwNepsfPSdC7g7">参与即将举行的 Google UXR 研究！</a>：您好，我正通过一份简短的问卷与您联系，以核实您参加即将举行的 Google 可用性研究的资格。这次研究是一个提供反馈的机会...

  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1334673652897284096)** (9 messages🔥): 

> `拉瑙湖特有的鲤科鱼类，播客长度限制，NotebookLM YouTube 内容` 


- **拉瑙湖上的 Jar Jar 和 Yoda**：一场由 **Jar Jar Binks** 和 **Yoda** 围绕 **拉瑙湖（Lake Lanao）特有的鲤科鱼类** 的 **保护** 展开的讨论，引发了对栖息地保护的共同关注。
   - 附带了一个 [音频片段](https://cdn.discordapp.com/attachments/1124403655819415592/1334673652565671987/Lake_Lanaos_Endemic_Cyprinids__Status_and_Conservation.mp3?ex=679e0bf3&is=679cba73&hm=797d56799ba4136450dfa454efeb90bc29f1925c0f604713c9dd55c2ed569ce6&) 以深化讨论。
- **限制播客长度**：一位用户询问如何将播客长度限制在 **一分钟或更短**，得到的回复表明强制执行此限制具有挑战性。
   - 另一位用户建议减少文本 **input** 可能会自然地缩短播客时长。
- **NotebookLM 内容赞赏**：用户对 **YouTube** 上的 **NotebookLM** 内容表示兴奋，特别称赞一位用户的内容非常出色。
   - 参与者注意到内容的 **高质量**，对社区价值展开了积极对话。
- **对时长的关注**：一位用户对另一位用户如何创建 **近一小时** 长的播客表示好奇。
   - 随后澄清是另一个人制作了该超长笔记本内容。
- **Prompt 请求**：用户表达了获取播客所用 **特定 Prompt** 的兴趣，显示出社区成员相互学习的热情。
   - ❤️ 等表情符号突显了用户之间分享资源的积极氛围。


  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1334615414834987041)** (47 messages🔥): 

> `Gemini 2.0 Flash 问题，AI 旁白改进，笔记本共享困难，Google Workspace 与 NotebookLM Plus，文档中的定义术语` 


- **Gemini 2.0 Flash 在更新期间出现故障**：用户在 **Gemini 2.0 Flash** 更新期间遇到了停机情况，这可能是由于最近的更新引起的，但目前似乎已恢复运行。
   - 讨论表明这次更新可能导致了临时问题，但一些成员报告其运行良好。
- **对真实 AI 旁白的渴望**：成员们表示有兴趣利用 AI 实现更好的旁白功能，希望它能逐字阅读脚本。
   - 虽然单人主持的旁白风格是可行的，但一位成员指出这可能并不完全符合该产品的定位。
- **共享笔记本的挑战**：多位用户报告在尝试共享笔记本时遇到困难，即使使用了公开链接也是如此。
   - 有人建议了一个变通方法：在重新共享后通过复制链接来访问笔记本，这表明系统中可能存在 Bug。
- **Google Workspace 集成问题**：一位用户提到升级到了标准的 Google Workspace 账户，但无法访问 **NotebookLM Plus**。
   - 另一位成员提供了一个排查此集成问题的检查清单链接，并暗示现在可能不再需要附加许可证（addon license）。
- **文档中定义术语的查询**：一位成员提出了关于使用 **NotebookLM** 识别文档中定义术语的问题。
   - 另一位参与者建议进行实验，表明对该工具在此领域的能力尚不确定。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.trackingai.org/home">Tracking AI</a>: Tracking AI 是一款前沿应用，旨在揭示人工智能系统中嵌入的政治偏见。通过我们直观的平台探索和分析 AI 的政治倾向...</li><li><a href="https://thedrive.ai">The Drive AI: 彻底改变文件管理与知识库</a>: 探索 The Drive AI 在智能文件组织方面的突破。我们的平台借助前沿 AI 将您的文件转化为动态知识库。提升您的业务运营...</li><li><a href="https://www.youtube.com/watch?v=Cr7J2PLo2fw">与 NotebookLM 创始工程师的对话</a>: Google 的 NotebookLM 已成为处理文本最引人注目的 AI 工具之一。在这次对话中，该项目的创始工程师 Adam Bignell...</li><li><a href="https://www.tiktok.com/t/ZT22DHefp/">TikTok - Make Your Day</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1334853035288952943)** (19 条消息🔥): 

> `Distributed GRPO, Memory Management in GRPO, Profiler for Memory Usage, VLLM Inference, Using bf16 for Training` 


- **GRPO 面临 OOM 挑战**：一位成员在 GitHub 上讨论了使用 GRPO 时遇到的**显存溢出 (OOM)** 问题，怀疑是内存管理错误以及使用 **fp32** 带来的限制。
   - *切换到 bf16 似乎缓解了一些问题*，为获取更多资源提供了更有力的依据。
- **内存使用 Profiler 建议**：讨论转向了生成内存使用图表，建议使用当前 PPO recipe 中的 **profiler** 来排查 OOM 错误。
   - 一位成员建议尝试使用较小的模型来生成 **profiler traces**，并分享更易于处理的结果进行分析。
- **混合精度训练的见解**：一位成员澄清说，在 V100 上通过模拟使用 **bf16** 在计算过程中可能不会节省内存，尽管数据可以以 16 位存储。
   - 这一见解强调了在切换精度格式时，存储与计算之间的权衡。
- **GRPO 的并行推理训练**：一位参与者建议，通过使用 **vLLM** 将**推理与训练并行运行**，可以显著提升 GRPO 的性能。
   - 然而，有人指出在将 **vLLM** 与 Hugging Face 生态系统之外的模型集成时存在困难。
- **推动 GRPO 的迭代改进**：在 GitHub 上发布 GRPO 项目的修改后，成员们讨论了在清理代码以进行更结构化的贡献之前，先对结果进行迭代。
   - 大家承认在提交正式的 pull requests 之前，内部正在进行密集的调整以优化性能。



**提及的链接**：<a href="https://github.com/RedTachyon/torchtune">GitHub - RedTachyon/torchtune: PyTorch native post-training library</a>：PyTorch 原生微调库。通过在 GitHub 上创建账号来为 RedTachyon/torchtune 的开发做出贡献。

  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1334665063562481756)** (27 条消息🔥): 

> `Gradient Accumulation 问题, TRL 与 Torchtune 配置差异, DPO 训练异常, Torchtune 中的 Multinode 支持, 损失计算归一化` 


- **困扰训练的 Gradient Accumulation 问题**：最近的讨论揭示了一个尚未解决的 [issue](https://github.com/unslothai/trl/issues/2175)，该问题影响了 Gradient Accumulation，对各种模型的训练和损失计算产生了负面影响。
   - 重点关注训练效率，用户正在调查缺乏修复是否会影响 DPO 和 PPO 等模型。
- **比较 TRL 和 Torchtune 配置**：成员们讨论了显著的参数差异，特别是 TRL 中 512 的 **max prompt length**，与 **Torchtune** 的设置相比，这可能会影响模型输出。
   - 他们指出，在测试期间，在 Torchtune 中实施 TRL 设置并未产生任何积极的学习结果。
- **DPO 训练损失异常**：一位用户指出，**DPO** 在同一数据集上仅需几步就达到了 **0 损失** 和 **100% 准确率**，这表明存在潜在问题。
   - 随后讨论了关于归一化和损失计算的影响，以确保训练结果的准确性。
- **推动 Multinode 支持的最终审批**：有人请求对 Torchtune 内部的 [multinode support](https://github.com/pytorch/torchtune/pull/2301) 集成进行最终审批，强调了其对用户需求的重要性。
   - 对参数 `offload_ops_to_cpu` 进行了辩论，需要确认其在多线程能力的后端上下文中的相关性。
- **训练中损失计算的归一化**：关注点转向了一个旨在通过考虑 non-padding tokens 来改进损失归一化的 [pull request](https://github.com/pytorch/torchtune/pull/1875)，这对于准确的指标至关重要。
   - 用户表示需要一个稳健的解决方案，以确保损失计算正确执行，特别是在 Gradient Accumulation 期间。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/gradient">LLM 训练中的 Bug 修复 - Gradient Accumulation</a>: Unsloth 的 Gradient Accumulation 修复解决了 LLM 训练中的关键错误。</li><li><a href="https://github.com/unslothai/unsloth/issues/1178.">unslothai/unsloth</a>: 比原来快 2-5 倍且节省 70% 内存地微调 Llama 3.3, Mistral, Phi-4, Qwen 2.5 &amp; Gemma LLMs - unslothai/unsloth</li><li><a href="https://github.com/pytorch/torchtune/pull/2322">由 joecummings 提交的 PR #2322：为文档构建使用 checkout@v4 / upload@v4</a>: 👀 👀 👀   👀 👀 👀   👀 👀 👀👀       👀      👀   👀 👀👀 👀        👀 👀 👀   👀 👀 👀</li><li><a href="https://github.com/pytorch/torchtune/pull/1875">由 ebsmothers 提交的 PR #1875：通过（non-padding）tokens 总数归一化 CE loss</a>: 为了纪念 ML 社区第一次发现 (x1 / n1) + (x2 / n2) != (x1 + x2) / (n1 + n2) 的那一天。此 PR 更改了启用 Gradient Accumulation 时我们计算损失的方式。T...</li><li><a href="https://github.com/pytorch/torchtune/pull/2301?">由 joecummings 提交的 PR #2301：torchtune 中的 Multinode 支持</a>: 正式宣布 torchtune 中的 multi-node 投入使用！上下文：这是多位用户的明确要求（#2161, #2142），虽然 OOTB 应该可以相当容易地工作，但我们...</li><li><a href="https://github.com/pytorch/torchtune/pull/2275#issuecomment-2623298923">由 sam-pi 提交的 PR #2275：全 DPO 分布式</a>: 上下文：改编自 #1966 的出色工作。此 PR 的目的是什么？是为了添加新功能。请链接此 PR 解决的任何 issue：涉及 #2082。更新日志：有哪些变更...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1334801571170615327)** (7 messages): 

> `HPC 资源与编程语言，DeepSeek 对 AI 算力需求的影响，Mojo 与 VS Code 的集成，Modular cfg 文件问题，博客系列中的技术细节澄清` 


- **HPC 资源需要更好的利用率**：一位成员对旨在解决科学家使用**异构 HPC 资源**时面临挑战的博客系列表示兴奋，强调了对像 **Mojo** 这样更好的编程语言解决方案的需求。
   - 他们强调，有效的**硬件利用率**可能会减少对昂贵 GPU 的依赖，这表明人们对 AI 算力需求的普遍认知正在发生转变。
- **DeepSeek 颠覆 AI 算力规范**：该成员提到，**DeepSeek** 最近的进展证明了**更好的硬件利用率**可以显著改变 AI 算力需求的格局，并挑战现有观念。
   - 科技巨头（Big Tech）正在对这一转变做出反应，不仅在争相与 DeepSeek 竞争，还在捍卫“**大规模基础设施**对维持 AI 研究领先地位至关重要”的观点。
- **Mojo 与 VS Code 连接问题**：一位用户描述了在使用 **WSL 和 Magic** 时将 **Mojo** 与 **VS Code** 集成的困难，表示该过程不够清晰。
   - 他们报告称，在尝试于 VS Code 中运行代码时，收到一条错误提示，称无法读取 **modular cfg 文件**。
- **论坛求助建议**：另一位成员建议将技术问题发布在论坛或特定频道，以寻求更集中的帮助。
   - 这体现了社区鼓励利用现有资源来解决问题的氛围。
- **对澄清技术细节的期待**：一位兴奋的成员分享了他们为即将发布的博客系列做出贡献的渴望，认为 **Mojo** 和 **MAX** 提供了可行的解决方案。
   - 他们承认问题的复杂性，但希望该系列能为更广泛的受众澄清这些挑战。



**提及的链接**：<a href="https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact-on-ai">Modular: Democratizing Compute, Part 1: DeepSeek’s Impact on AI</a>：文章的第一部分，探讨了在 DeepSeek 发布背景下，CUDA 之外 AI 硬件加速的未来。

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1334653629646245959)** (19 messages🔥): 

> `库的向后兼容性，Mojo 1.0 基准测试延迟，Swift 复杂性担忧，Mojo 开发的稳定性` 


- **向后兼容性至关重要**：一位成员强调，**向后兼容性**在包括插件和库在内的所有系统中都至关重要；如果没有它，用户将被迫在“永不更新”或“丢失核心功能”之间做出选择。
   - 他们指出，这种现象具有普遍性，会影响用户满意度和生态系统的活力。
- **Mojo 1.0 发布时间线确认**：另一位成员对推迟 **Mojo 1.0** 的发布表示满意，主张在发布前在更大的计算集群上进行彻底的 **benchmarking**（基准测试）。
   - 他们指出，在小型系统上进行测试相当于对 Mojo 能力的“小型基准测试”，确保了社区用户更广泛的可用性。
- **Swift 在异步方面的复杂性**：一位用户分享了对 **Swift** 引入的复杂性的担忧，特别是在实现异步（asynchronous）功能时，揭示了代码设计中潜在的陷阱。
   - 这段对话引发了成员们的共同愿望，即希望 **Mojo** 能够避免此类复杂性，并在开发中保持简洁。
- **注重稳定性而非赶工**：Clattner 承认 Mojo 开发中对**稳定性**的需求，澄清并没有压力去匆忙推进发布进程。
   - 他们强调，虽然版本控制对于沟通很重要，但首要任务仍然是为社区创建一个平衡且经过深思熟虑的产品。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://forum.modular.com/t/how-to-convert-numpy-array-items-to-mojo-float/506">How to convert numpy array items to mojo Float?</a>：你好，我正尝试将 numpy 数组的每个元素获取为 mojo Float。目前这种方式可行：var value: Float64 = Xb.item(i, j).to_float64()。但 linter 建议使用 float(obj)，我假设它是 Float6...</li><li><a href="https://github.com/modular/max/issues/289">[BUG]: The `mojo-lsp-server` executable shows incorrect help information · Issue #289 · modular/max</a>：Bug 描述：mojo-lsp-server 可执行文件在提供 --help 参数时显示了许多选项，但它们与其行为无关。@JoeLoser 建议提交此 bug，以便 w...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1334919830616281089)** (3 messages): 

> `使用 MAX 运行 DeepSeek，使用 Ollama gguf 文件` 


- **使用 MAX 轻松运行 DeepSeek**：一位成员在成功使用 **Ollama** 下载模型的 gguf 文件后，询问如何使用 **MAX** 运行 **DeepSeek**。
   - 提供了在最新 **MAX** 仓库的 `/pipelines/python` 目录下运行以下命令的说明：`magic run serve --huggingface-repo-id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --weight-path=unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf`。
- **MAX 最大化 DeepSeek 潜力**：值得注意的是，通过检出最新的 nightly **MAX** 仓库，用户可以有效地利用 **DeepSeek** 模型。
   - 这一功能突显了 **MAX** 与先进模型的集成，增强了用户的灵活性。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1334666028503924937)** (3 messages): 

> `与 Arize AI 和 Groq 的见面会、LlamaReport Beta 版发布、o3-mini 支持` 


- **关于 Agent 和追踪的三方对话**：观看我们与 [@arizeai](https://twitter.com/arizeai) 和 [@GroqInc](https://twitter.com/GroqInc) 讨论 **Agent** 和追踪的见面会录像，其中包含使用 **Arize** 的 **Phoenix** 进行的现场演示。
   - 该会议深入探讨了 **LlamaIndex** 的 **Agent** 能力，从基础的 **RAG** 技术到高级功能，详情见 [Twitter 线程](https://twitter.com/llama_index/status/1885106917707833763)。
- **LlamaReport Beta 版展示**：观看这段使用 **LlamaReport** 早期 Beta 版的视频，这是定于 **2025** 年发布的报告生成核心应用。
   - 在 [Twitter 链接](https://twitter.com/llama_index/status/1885420164893860097) 中观看展示其特性和功能的演示。
- **即时支持 o3-mini**：宣布对 **o3-mini** 的零日支持，并提供使用 `pip install -U llama-index-llms-openai` 进行安装的指令。
   - 更多详情请参阅强调安装命令的 [Twitter 公告](https://twitter.com/llama_index/status/1885426718506442832)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1334616464724070400)** (9 messages🔥): 

> `OpenAI O1 模型支持、LlamaReport 使用、LLM 集成问题` 


- **OpenAI O1 模型缺乏完整功能**：讨论显示 **OpenAI** 尚未为 **O1** 模型实现完整功能，导致对其功能和支持的困惑。
   - 社区成员指出，该模型的 **streaming** 支持非常奇怪，许多功能未能按预期工作。
- **询问 LlamaReport 频道**：一位成员询问是否有 **LlamaReport** 的特定频道，对此得到的回复是由于访问受限，目前没有专门的频道。
   - 提供的细节表明，现阶段没有多少人拥有 **LlamaReport** 的访问权限。
- **对 LlamaReport 功能的担忧**：另一位成员表示在使用 **LlamaReport** 生成内容时遇到困难，并询问 **LLM** 集成和支付流程是如何运作的。
   - 尽管存在问题，但有人提到基础测试已成功运行，特别是上传论文进行摘要生成。
- **关于 Llama-Parse 费用的澄清**：澄清了 **Llama-Parse** 的费用主要产生于内容解析，成员们推测在某些条件下可能是免费的。
   - 鼓励用户检查是否有可用额度或付费计划，以探索集成的功能。



**提及的链接**：<a href="https://community.openai.com/t/streaming-support-for-o1-o1-2024-12-17-resulting-in-400-unsupported-value/1085043?utm_source=chatgpt.com#:~:text=Streaming%20of%20the,for%20this%20model.">o1 (o1-2024-12-17) 的 Streaming 支持（导致 400 "Unsupported value"）</a>：你好，看来 o1-preview 和 o1-mini 已经添加了 streaming 支持（见公告 OpenAI o1 streaming now available + API access for tiers 1–5）。我确认两者对我来说都有效。然而...

  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1334640891188740230)** (10 条消息🔥): 

> `用于 LLM 的物理服务器，Tinygrad PR 讨论，Kernel 和 buffer 调整，PR 标题拼写错误` 


- **讨论用于 LLM 托管的物理服务器**：一位用户询问了在本地托管 **LLM** 以用于企业用途的**物理服务器**选项，表示对有效**运行模型**感兴趣。
   - 另一位用户提到了 **Exolabs**，并引用了之前关于使用 **Mac Minis** 进行类似任务设置的讨论。
- **Tinygrad PR 亮点**：George Hotz 认可了一个关于 kernel、buffer 和 launch dimensions 的优秀**首个 PR**，见于 [tinygrad](https://x.com/__tinygrad__/status/1885291485433839729)。
   - 他提出了建议，例如从 `DEFINE_LOCAL` 的参数中移除 **16**，因为它已经包含在 **dtype** 中。
- **PR 标题中的拼写错误引发讨论**：一位用户指出 **PR 标题中的拼写错误**可能表明缺乏对细节的关注，这可能会影响维护者的看法。
   - 该用户为漏掉错误表示道歉，并确认在收到反馈后已将其修复。



**提到的链接**：<a href="https://x.com/__tinygrad__/status/1885291485433839729">来自 tiny corp (@__tinygrad__) 的推文</a>：在 tinygrad 中，一个 kernel、它的 buffer 及其 launch dims

  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1334639577776001105)** (10 条消息🔥): 

> `Axolotl AI 对 bf16 的支持，fp8 训练的担忧，8bit lora 功能` 


- **Axolotl AI 采用 bf16 训练**：成员们确认 **Axolotl** 已经支持 **bf16 训练**很长时间了，不再局限于 **fp32**。
   - *友情提示*，它已被公认为训练的稳定选择。
- **对 fp8 训练性能的担忧**：关于 **fp8** 的讨论强调了它对 **accelerate** 的支持，但据成员观察，其性能表现并不理想。
   - 一位成员评论道：“我不认为我们目前正在研究这个”，原因是其不稳定性。
- **8bit lora 对比 8bit fft**：虽然一位成员确认 **8bit lora** 训练是可行的，但他们对 **8bit fft** 的功能表示不确定。
   - 另一位参与者指出 *fp8 难以使用且过于不稳定*，这增加了人们的担忧。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1334646021900341288)** (9 条消息🔥): 

> `证书发放更新，Quiz 1 可用性，教学大纲（Syllabus）部分的困惑` 


- **证书发放仍待定**：已确认**证书尚未发放**，关于本学期证书要求的更新稍后发布。
   - 成员们对即将到来的发布表示兴奋，评论如“等待证书的过程真是太令人兴奋了！”
- **Quiz 1 现已开放**：Quiz 1 已在课程网站上发布，具体位于 **syllabus（教学大纲）部分**。
   - 一些成员提到难以找到测验链接，表明可能存在可见性或访问问题。
- **教学大纲（Syllabus）部分的困惑**：一位成员对在 syllabus 部分看不到测验链接表示困惑，另一位成员随后回复并提供了截图。
   - 进一步的讨论强调了成员所见内容之间的差异，使 syllabus 的内容产生了一种神秘感。



**提到的链接**：<a href="https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit?usp=sharing">测验存档 - LLM Agents MOOC</a>：注意：正确答案在黑色方框中（黑底黑字）。用光标高亮显示方框即可显示正确答案（如果难以查看，也可以将文本复制到新浏览器中...）

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1334627998347038851)** (6 messages): 

> `AI tool explanations, Farm Friend Application, iOS Shortcuts Patreon, NVIDIA NIM and DeepSeek` 


- **AI 工具简化说明**：一名成员请求对该 AI 工具的功能及即将推出的特性进行简单解释，表达了对其功能的浓厚好奇心。
   - 他们请求社区以一种易于初学者理解的方式进行澄清。
- **Farm Friend 应用发布**：一名成员分享了在生态系统中发布新桌面应用程序的兴奋之情，特别是 [Farm Friend 应用](https://farm-friend-v1.teplit.app)。
   - 他们承诺在继续开发应用程序的过程中提供更多链接和资源。
- **iOS Shortcuts 的 Patreon**：一名成员宣布了 Patreon 计划，将提供不同层级的 iOS shortcuts，包括 Agentic 功能等高级特性。
   - 他们对回归并分享过去一年展示的技术表示热切期待。
- **使用 NVIDIA NIM 配合 DeepSeek**：一名成员询问了利用 **NVIDIA NIM** 安装 **DeepSeek** 并将其连接到 Open Interpreter 的可能性。
   - 他们正在寻求社区关于这一技术集成的建议和见解。


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 messages): 

the_lonesome_slipper: Thank you!
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1335014895116222569)** (1 messages): 

> `Cohere Embed API v2.0, HTTP 422 Error, Preprocessing for Embeddings, Cross-language Polarization Research, Embed Multilingual Model` 


- **使用 Embed API v2.0 时遇到 HTTP 422 错误**：一名用户报告在尝试使用有效的 API key 和指定参数运行 **Embed API v2.0** 的 cURL 示例时，出现了 **HTTP 422 Unprocessable Entity** 错误。
   - 他们引用了 [Cohere Embed API 文档](https://docs.cohere.com/reference/embed) 作为参考。
- **对用于研究的多语言 Embedding 感兴趣**：该用户表示希望利用 **embed-multilingual-v3.0** 模型来分析与其**跨语言极化（cross-language polarization）**研究相关的几篇长文章。
   - 他们专门询问了针对可能杂乱的文本所需的预处理建议，并参考了 [Cohere Wiki Embeddings](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3) 获取背景信息。


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1334975846191988828)** (2 messages): 

> `User Mentions, Thanks and Acknowledgment` 


- **用户提及**：一名用户在讨论中提及了角色 @825830190600683521，可能是为了引起对其输入或权威性的关注。
   - 这种提及表明了一个认可特定贡献的协作环境。
- **表达谢意**：另一名成员 mega_b 通过简单地回复“Ty”来对之前的提及表示感谢。
   - 这突显了频道内积极的互动，认可和感谢在其中非常普遍。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1334675444255166475)** (1 messages): 

> `http_client parameter, dspy.LM configuration` 


- **缺失 http_client 参数的困惑**：一名成员指出 dspy.LM 中没有像 gpt3.py 那样的 **http_client 参数**，并说明 OpenAI 和 gpt3.py 允许使用带有 SSL 上下文和代理设置的自定义 client（如 **httpx.Client**）。
   - 这引发了关于如何在 **dspy.LM** 中实现类似功能的问题。
- **自定义 Client 实现查询**：对话强调了对如何在 **dspy.LM** 中设置自定义 client 的兴趣，参考了 gpt3.py 中使用 `http_client: Optional[httpx.Client] = None` 的实现方式。
   - 成员们讨论了调整这种方法以适应 DSPy 框架的可能性。


  

---


---


---


---


---


{% else %}


> 完整的频道细分内容已在邮件中截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}