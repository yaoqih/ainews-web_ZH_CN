---
companies:
- openai
- weights-biases
- cohere
- weaviate
date: '2024-09-14T00:55:34.586718Z'
description: '**OpenAI** 发布了 **o1 模型系列**，该系列被誉为该公司“迄今为止功能最强、对齐度最高的模型”，通过强化学习训练以增强推理能力。**o1-preview**
  模型在 ARC-AGI 测试中得分为 **21%**，在 aider 代码编辑测试中得分约为 **80%**（超过了 Claude 3.5 Sonnet 的 77%），在
  Cognition-Golden 测试中得分约为 **52%**。这些数据展示了模型从“记忆答案”向“内化推理逻辑”的转变。


  该模型采用了独特的思维链（chain-of-thought）方法，实现了“系统 2 思维”（System II thinking），从而能更有效地解决问题。专家
  **Andrew Mayne** 建议将 o1 视为一个能提供周详解释的“聪明朋友”。此外，由 **Weights & Biases**、**Cohere**
  和 **Weaviate** 赞助的高级 RAG 课程提供了混合搜索和提示词策略，旨在优化 AI 解决方案。'
id: 0de61ce2-6328-46d7-9a74-e3cfdcb5b151
models:
- o1-preview
- o1-mini
- claude-3.5-sonnet
- gpt-4o
original_slug: ainews-learnings-from-o1-ama
people:
- sama
- rohanpaul_ai
- gdb
- andrew-mayne
title: '以下是“Learnings from o1 AMA”的中文翻译：


  **o1 AMA 总结**


  （或者：**从 o1 AMA 中获得的启发/学习心得**）'
topics:
- reinforcement-learning
- chain-of-thought
- reasoning
- model-performance
- prompting
- code-editing
- rag
- hybrid-search
---

<!-- buttondown-editor-mode: plaintext -->**对基于 RL 的 CoT 的赞赏就是你所需要的一切。**

> 2024年9月12日至9月13日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务端（**216** 个频道和 **5103** 条消息）。预计节省阅读时间（以 200wpm 计算）：**502 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

在 o1 发布后的第二天，我们了解到：

- [o1-preview 在 ARC-AGI 上得分 21%](https://x.com/arcprize/status/1834703303621710077?s=46)（SOTA 为 46%）：“总而言之，o1 代表了从‘记住答案’到‘记住推理’的范式转变，但它并没有脱离更广泛的范式，即通过拟合分布曲线来使一切都符合分布，从而提升性能。”
- [o1-preview 在 aider 代码编辑上得分约 80%](https://aider.chat/2024/09/12/o1.html)（SOTA - Claude 3.5 Sonnet 为 77%）：“o1-preview 模型在遵循 aider 的 diff 编辑格式方面遇到了困难。o1-mini 模型在遵循 whole 和 diff 编辑格式方面都遇到了困难。Aider 非常宽容，并努力接受任何接近正确格式的内容。令人惊讶的是，如此强大的模型在简单文本输出格式的语法要求上会遇到困难。aider 似乎有可能优化其 prompts 和编辑格式，以更好地利用 o1 模型。”
- [o1-preview 在 Cognition-Golden 上得分约 52%](https://x.com/cognition_labs/status/1834292718174077014?s=46)，并附带[建议](https://x.com/cognition_labs/status/1834292725417730408)：“Chain-of-thought 和要求模型‘大声思考’是以前模型的常用 prompts。相反，我们发现要求 o1 只给出最终答案通常表现更好，因为无论如何它都会在回答之前进行思考。o1 需要更密集的 context，并且对杂乱和不必要的 tokens 更加敏感。传统的 prompting 方法通常在给出指令时存在冗余，我们发现这会对 o1 的性能产生负面影响。”
- [Andrew Mayne 的 o1 prompting 建议](https://x.com/andrewmayne/status/1834408991839158422?s=46)：“不要把它看作传统的聊天模型。在你的脑海中把 o1 想象成一个非常聪明的朋友，你会给她发私信来解决问题。她会回复一个经过深思熟虑的解释，引导你完成各个步骤。”
- [OpenAI 研究团队 AMA](https://x.com/btibor91/status/1834686946846597281) —— 最后这一点由 Tibor Blahe 总结得最好：


![image.png](https://assets.buttondown.email/images/2aca37ca-24d6-416b-a0de-eba291ea1488.png?w=960&fit=max)


除此之外，这是一个安静的周五，所以你可以查看[最新的 Latent Space 与 OpenAI 的播客](https://www.latent.space/p/openai-api-and-o1)，或者报名参加[下周的旧金山黑客松](http://wandb.me/swyx-hack)，该活动由本月的赞助商、我们亲爱的朋友 WandB 为您带来！

---

**[由 Weights & Biases 赞助的高级 RAG 课程](https://wandb.me/ainews-course)**：超越基础的 **RAG** 实现，探索 **hybrid search 和高级 prompting** 等高级策略，以优化性能、评估和部署。向来自 **Weights & Biases、Cohere 和 Weaviate** 的行业专家学习如何克服常见的 RAG 挑战并构建强大的 AI 解决方案，还可获得免费的 Cohere 额度！

[
![image.png](https://assets.buttondown.email/images/122b3420-6673-4514-b14b-f3a250a97da2.png?w=960&fit=max)
](https://wandb.me/ainews-course)


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**OpenAI 发布 o1 模型系列**

- **模型能力**：[@sama](https://twitter.com/sama/status/1834283100639297910) 发布了 o1，这是 OpenAI “迄今为止最强大且对齐程度最高”的一系列模型。这些模型通过强化学习进行训练，在回答之前会深入思考问题，从而提升了推理能力。

- **性能提升**：[@sama](https://twitter.com/sama/status/1834283105076879690) 强调了在各种基准测试中的显著提升。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1834294432214159439) 指出 o1 在 57 个 MMLU 子类别中的 54 个上超越了 GPT-4o，并在 MMMU 上达到了 78.2%，使其具备了与人类专家竞争的实力。

- **推理方法**：[@gdb](https://twitter.com/gdb/status/1834295775674990676) 解释了 o1 使用独特的思维链（chain-of-thought）过程，使其能够分解问题、纠正错误并调整方法。与之前模型的“系统 I 思维（System I thinking）”相比，这实现了“系统 II 思维（System II thinking）”。

- **模型变体**：[@sama](https://twitter.com/sama/status/1834283103038439566) 宣布 o1-preview 和 o1-mini 立即在 ChatGPT 中面向 Plus 和 Team 用户开放，并在 API 中面向 tier 5 用户开放。[@BorisMPower](https://twitter.com/BorisMPower/status/1834289286968934762) 澄清 tier-5 API 访问权限需要累计支付 1,000 美元且自首次成功付款起超过 30 天。

- **技术细节**：[@virattt](https://twitter.com/virattt/status/1834336726653055141) 指出 o1 引入了一类新的“推理标记（reasoning tokens）”，这些标记按输出标记计费，并计入 128K 上下文窗口。OpenAI 建议预留 25K 标记用于推理，这实际上将可用上下文减少到约 100K 标记。

- **安全性改进**：[@lilianweng](https://twitter.com/lilianweng/status/1834346548786069647) 提到 o1 在安全性和鲁棒性指标上有显著提升，对安全规则进行推理是教授模型人类价值观和原则的一种有效方式。

- **推理侧扩展（Inference Time Scaling）**：[@DrJimFan](https://twitter.com/DrJimFan/status/1834279865933332752) 强调了 o1 代表了向推理侧扩展的转变，即计算资源被用于推理过程而非仅仅是预训练。这允许通过诸如蒙特卡洛树搜索（Monte Carlo tree search）等技术获得更精炼的输出。

- **潜在应用**：[@swyx](https://twitter.com/swyx/status/1834284741610405965) 分享了 o1 用于经济学、遗传学、物理学和编程任务的示例，展示了其跨领域的通用性。

- **开发者访问**：[@LangChainAI](https://twitter.com/LangChainAI/status/1834329330736091162) 宣布 LangChain Python 和 JS/TS 立即支持 o1，允许开发者将新模型集成到他们的应用中。

**反应与分析**

- **范式转移**：包括 [@willdepue](https://twitter.com/willdepue/status/1834294935497179633) 在内的许多用户强调，o1 代表了 AI 开发的新范式，在不久的将来有快速提升的潜力。

- **与其他模型对比**：虽然许多人印象深刻，但也有一些用户如 [@aaron_defazio](https://twitter.com/aaron_defazio/status/1834364143639613641) 批评 OpenAI 的发布公告中缺乏与其他实验室之前的 SOTA（state-of-the-art）模型的对比。

- **隐藏的推理过程**：[@vagabondjack](https://twitter.com/vagabondjack/status/1834287466884297103) 指出 OpenAI 出于“竞争优势”等原因，未向用户展示完整的思维链文本。

- **成本考量**：[@labenz](https://twitter.com/labenz/status/1834305341170856245) 指出 o1 的输出标记定价与最初的 GPT-3 定价一致，为 0.06 美元/1K 标记，输入标记便宜 75%。然而，隐藏的推理标记可能会使许多用例的总成本与之前的模型相当。

**梗与幽默**

- [@karpathy](https://twitter.com/karpathy/status/1834374965942255835) 调侃 o1-mini 拒绝解决黎曼猜想（Riemann Hypothesis），幽默地引用了模型的潜在局限性。

- 几个用户拿模型名称开玩笑，[@huybery](https://twitter.com/huybery/status/1834291444540194966) 调侃道：“既然 OpenAI o1 来了，Qwen q1 还会远吗？”

---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. OpenAI o1：AI 推理能力的飞跃**

- **[Evals - OpenAI o1](https://i.redd.it/jpz49alcxeod1.png)** ([Score: 110, Comments: 21](https://reddit.com//r/LocalLLaMA/comments/1ff842v/evals_openai_o1/)): OpenAI 的 **o1 模型**在 **STEM** 和**编程任务**中表现出显著进步，其最新的评估结果揭示了这一点。该模型在**数学**、**物理**和**计算机科学**等领域比之前的版本提高了 **20-30%**，在**算法问题解决**和**代码生成**方面表现尤为强劲。这些改进表明 AI 在技术和科学应用方面的能力有了显著跨越。
  - 用户质疑为什么**语言模型**在 **AP English** 考试中的表现不如复杂的 **STEM 任务**，并指出解决 **IMO 问题**似乎比语言类测试更具挑战性。
  - 讨论中包含了“🍓”评论，但在没有额外背景的情况下，其相关性或含义尚不明确。
  - 用户对该模型在**博士级问题**上超越**人类专家**的能力表示兴奋，强调了这一成就的重要性。


- **[Preliminary LiveBench results for reasoning: o1-mini decisively beats Claude Sonnet 3.5](https://i.redd.it/6poysi1cfhod1.jpeg)** ([Score: 268, Comments: 129](https://reddit.com//r/LocalLLaMA/comments/1ffjb4q/preliminary_livebench_results_for_reasoning/)): 根据初步的 **LiveBench** 结果，新款 AI 模型 **o1-mini** 在**推理基准测试**中**决定性地击败了 Claude 3.5 Sonnet**。[Bindu Reddy 在 Twitter 上](https://x.com/bindureddy/status/1834394257345646643)分享了这些发现，表明 AI 推理能力取得了重大进展。
  - **o1-mini** 在 **STEM 和代码领域**优于 **o1-preview**，用户注意到它在 **lmarena** 等平台上的卓越推理能力。随着更多的**强化学习**和**思考时间**，该模型的性能会进一步提升。
  - 用户争论将 o1-mini 与其他模型进行比较是否公平，因为它使用了**内置的 Chain of Thought (CoT)** 推理。一些人认为这是一个合法的功能，而另一些人则认为这是在基准测试中“投机取巧”。
  - **OpenRouter** 允许有限度地访问 o1-mini，价格为 **$3.00/1M input tokens** 和 **$12.00/1M output tokens**，限制为**每天 12 条消息**。尽管 token 消耗很高，用户仍对尝试该模型表示兴奋。


- **["We're releasing a preview of OpenAI o1—a new series of AI models designed to spend more time thinking before they respond" - OpenAI](https://x.com/OpenAI/status/1834278217626317026)** ([Score: 641, Comments: 248](https://reddit.com//r/LocalLLaMA/comments/1ff7uqz/were_releasing_a_preview_of_openai_o1a_new_series/)): OpenAI 宣布发布 **o1** 预览版，这是一个全新的 AI 模型系列，旨在**在回答之前花费更多时间思考**。这些模型经过工程设计，展现出**高级推理能力**，有可能提高 AI 生成内容的质量和深度。该公告表明 OpenAI 正在专注于改进 AI 系统的审议过程，这可能会在各种应用中带来更周到、更准确的回答。
  - OpenAI 的新 **o1** 模型在**推理能力**方面表现出显著改进，在 IMO 预选赛中获得 **83%** 的分数（相比之下 GPT-4 为 **13%**），并在 Codeforces 编程竞赛中达到 **89% 的分位数**。然而，一些用户对现实世界的表现持怀疑态度。
  - **隐藏思维链 (Chain of Thought)** 过程的决定引发了批评，用户将其贴上“**ClosedAI**”的标签，并对透明度降低表示担忧。一些人推测，巧妙的提示可能仍会揭示模型的思考过程。
  - 讨论中还将其与最近的“**Reflection**”争议进行了比较，探讨这是否是类似概念的更复杂实现。该模型还声称对 **jailbreaking** 尝试的抵抗力**提高了 4 倍**，一些人将其负面地视为审查制度的加强。


**主题 2. 开源和本地 LLM 的进展**

- **[DataGemma 发布 - Google 系列 (27B 模型)](https://huggingface.co/collections/google/datagemma-release-66df7636084d2b150a4e6643)** ([Score: 122, Comments: 58](https://reddit.com//r/LocalLLaMA/comments/1ff23kn/datagemma_release_a_google_collection_27b_models/)): Google 发布了 **DataGemma**，这是一个专为数据分析任务设计的 **27B 参数语言模型**系列。该系列模型包括 **DataGemma-2b**、**DataGemma-7b** 和 **DataGemma-27b** 等变体，在包含 **3 万亿 token** 的多样化数据集上进行了训练，能够根据自然语言指令执行 **数据操作**、**分析** 和 **可视化** 等任务。这些模型根据 **Apache 2.0 许可证** 提供给研究使用。
  - **RIG (Retrieval-Interleaved Generation，检索交错生成)** 是 Google 为 DataGemma 引入的一个新术语，通过查询可信来源并针对 **Data Commons** 进行事实核查来增强 Gemma 2。这一特性使 DataGemma 在生成回答时能够检索准确的统计数据。
  - 用户演示了 RIG 的功能，展示了它如何查询 **Data Commons** 来填充关键统计数据，例如加州桑尼维尔（Sunnyvale, CA）的人口信息。这种方法有可能减少 AI 生成回答中的幻觉（hallucinations）。
  - 一些用户对尝试 DataGemma 表示兴奋，但也指出希望模型能有 **更大的上下文窗口（context windows）**。文中还分享了关于 DataGemma 的 Google 官方博客文章以提供更多信息。


- **6 款主流 LLM 推理引擎大比拼** ([Score: 42, Comments: 38](https://reddit.com//r/LocalLLaMA/comments/1ff79bh/faceoff_of_6_maintream_llm_inference_engines/)): 该帖子对比了 **6 款主流 LLM 推理引擎** 的本地部署情况，重点关注推理质量而非仅仅是速度。作者使用来自“其他”类别的 **256 个精选 MMLU Pro 问题** 进行了测试，在不同引擎上运行了具有各种量化级别的 **Llama 3.1 8B** 模型。结果显示，**较低的量化级别并不总是导致较低的质量**，在本次特定测试中，**vLLM 的 AWQ 量化** 表现最好，不过作者提醒不要将这些结果推广到所有用例。
  - 建议测试 **vLLM 的 AWQ 引擎**，作者确认其表现“相当不错”并进行了额外测试。AWQ 引擎代表了 vLLM 的 **“4 bit” 版本**，并且最近整合了 **Marlin 内核**。
  - 讨论中提到了使用 **Triton TensorRT-LLM 后端** 进行测试。作者指出它“以难以配置著称”，并且需要签署 **NVIDIA AI Enterprise License 协议** 才能访问 docker 镜像。
  - TensorRT-LLM 配置的复杂性被重点强调，作者分享了一张 [快速入门指南的截图](https://preview.redd.it/3y3b9ahlzeod1.png?width=638&format=png&auto=webp&s=7e69ed8b09e8dcf90f49eddb9dd21e6dd7012e92)。这让一位原本认为 **Triton 是免费且开源的** 评论者感到惊讶。


- **对 WebGPU + transformers.js (v3) 感到兴奋：在浏览器中利用全部 (GPU) 硬件** ([Score: 49, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1fexeoc/excited_about_webgpu_transformersjs_v3_utilize/)): **WebGPU** 和 **transformers.js v3** 现在支持 **在 Web 浏览器中充分利用 GPU 硬件**，从而在无需 Python 服务器或复杂设置的情况下，显著提升 AI 任务的性能。作者报告称，在 **M3 Max** 上，嵌入模型（embedding models）相比 WASM 有 **40-75 倍的加速**，在带有集成显卡或旧款 GPU 的消费级笔记本电脑上有 **4-20 倍的加速**。这项技术为 **Stable Diffusion**、**Whisper** 和 **GenAI** 等各种 AI 应用实现了私有的设备端推理，这些应用可以免费托管在 GitHub Pages 等平台上，正如 [SemanticFinder](https://do-me.github.io/SemanticFinder/webgpu/) 等项目所展示的那样。
  - **privacyparachute** 展示了一个具有 **会议转录** 和音频/视频 **自动字幕生成** 功能的项目，并为录制参与者提供了隐私控制。该项目利用了 **u/xenovatech** 的工作成果。
  - 关于浏览器可运行模型能力的讨论中，**SeymourBits** 最初认为它们很基础（大约是 2019 年的水平）。**privacyparachute** 反驳道，使用正确的 Web-AI 框架可以运行最新的模型，并推荐 [WebLLM](https://webllm.mlc.ai/) 作为示例。
  - 评论强调了 **基于浏览器的 AI 应用** 的持续发展，展示了原帖中所讨论技术的实际应用。


**主题 3：关于 AI 透明度以及开源与闭源开发的辩论**

- **"o1 仍然存在缺陷，仍然受到限制，而且初次使用时看起来比长时间使用后更令人印象深刻。"** ([Score: 108, Comments: 49](https://reddit.com//r/LocalLLaMA/comments/1ffcded/o1_is_still_flawed_still_limited_and_it_still/))：OpenAI 的 CEO Sam Altman 在 Twitter 线程中回应了对 **GPT-4 Turbo with vision**（被称为 "o1"）的批评，承认了其**缺陷和局限性**。他强调，虽然该模型最初可能看起来令人印象深刻，但长时间使用会暴露其短板，并强调了关于 AI 能力和局限性进行**负责任沟通**的重要性。

- **[OpenAI 隐藏了 o1 使用的 CoT 以获得竞争优势。](https://i.redd.it/1mx3jteushod1.jpeg)** ([Score: 40, Comments: 17](https://reddit.com//r/LocalLLaMA/comments/1ffkrvk/openai_hides_the_cot_used_by_o1_to_gain/))：据报道，OpenAI 正在隐藏其 **o1** 模型使用的**思维链 (CoT)**，以保持竞争优势。帖子指出，通过针对特定指标优化 CoT 提示词，可以使用**开源软件 (OSS)** 模型开发出**最先进 (SoTA)** 的模型，并提到 **DSPy** 是实现这一方法的工具。
  - 考虑到公司之间的人才流动，**Anthropic** 可能已经具备了复制或超越 **OpenAI o1 模型**的能力。据报道，他们的 **Sonnet 3.5** 模型已经领先了 3 个月，尽管由于算力限制，其使用可能受到限制。
  - OpenAI 承认**审查会显著降低模型智能**，这引发了人们的兴趣，特别是与生成**思维链 (CoT)** 输出相关的部分。
  - 对**隐藏 CoT** 的关注可能是 OpenAI 的一种战略叙事。一些人认为，更底层的过程，如 **Anthropic** 在**稀疏自编码器 (sparse autoencoder)** 研究中探索的内容，可能更好地解释 AI 模型中的 Token 选择和记忆形成。

- **如果 OpenAI 能让 GPT4o-mini 在推理方面大幅优于 Claude 3.5，这是否预示着本地 LLM 很快也能做到同样的事情？** ([Score: 111, Comments: 39](https://reddit.com//r/LocalLLaMA/comments/1ffndk5/if_openai_can_make_gpt4omini_be_drastically/))：帖子讨论了**开源替代方案**在推理能力上匹配或超越**封闭 AI 系统**的潜力。它认为，如果 **GPT4o-mini** 能在推理任务中显著优于 **Claude 3.5**，那么通过使用**思维链 (CoT)** 实现，类似的改进可能很快就能在**本地 LLM** 中实现。作者引用了一些研究，表明当给予通过 CoT 进行“思考”的机会时，**GPT3.5** 的推理能力可以超过 **GPT4**，这意味着开源模型可以采用类似的技术。
  - **OpenAI o1** 的训练理论包括使用 **GPT-4** 生成解决方案、应用 **STaR 论文**的方法以及直接使用 **RL**。这个过程可能涉及多种方法的结合，专家标注的成本可能高达**数亿美元**。
  - “**超级秘方**”可能在于**数据集质量**。OpenAI 的 **system card** 和 “**Let's verify step by step**” 论文提供了对其方法的见解，其中包括用于指令微调的**强化学习 (reinforcement learning)**。
  - 一项使用 **Nisten's prompt** 与 **c4ai-command-r-08-2024-Q4_K_M.gguf** 模型进行的实验展示了改进的问题解决能力，表明**开源替代方案**在推理任务中潜力匹配封闭 AI 系统。

**主题 4. 用于 LLM 训练的新数据生成技术**

- **[Hugging Face 增加了直接从浏览器中使用 SQL 查询所有 200,000 多个数据集的选项！](https://v.redd.it/memus4h3ucod1)** ([Score: 215, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fez5w9/hugging_face_adds_option_to_query_all_200000/))：**Hugging Face** 推出了一项新功能，允许用户直接从浏览器中使用 **SQL** **查询超过 200,000 个数据集**。这一增强功能实现了无需下载数据集即可进行数据探索和分析，提供了一种与平台上海量数据集交互的更高效方式。
  - 该功能由 **DuckDB WASM** 驱动，允许 SQL 查询直接在浏览器中运行。用户可以分享他们的 SQL 查询和视图，并提供反馈或功能请求。
  - 用户对 **Hugging Face** 提供广泛带宽、存储和 CPU 资源的能力表示赞赏。该功能因其在过滤数据集和下载结果方面的实用性而受到好评。
  - 几位用户发现该工具对特定任务很有帮助，例如**统计数据集元素数量**以及执行他们之前在本地使用 DuckDB 设置的分析。

- **我专门为 RP 制作了一个数据生成流水线：输入故事，输出以其主题和特征为灵感的 RP 数据** ([Score: 46, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1ffhv5f/i_made_a_data_generation_pipeline_specifically/)): 作者介绍了 **RPToolkit**，这是一个用于根据输入故事生成 **角色扮演数据集 (roleplaying datasets)** 的**开源流水线**，并针对 **local models** 进行了优化。该流水线可以创建**多样化、丰富的多轮角色扮演数据**，反映输入故事的**主题、流派和情感内容**。作者展示了其能力，使用 **Llama 3 70b** 和 **Mistral Large 2** 模型创建了一个包含约 **1000 个 RP 会话的数据集**。该工具旨在解决 RP 模型创作者的数据生成问题，允许用户创建针对特定流派或主题定制的数据集，而无需直接引用输入数据，从而可能规避版权问题。
  - 用户询问了用于数据集生成的**推荐 LLM**，作者建议使用 **turboderp/Mistral-Large-Instruct-2407-123B-exl2** 和 **Llama 3 70b**。**Magnum 123B** 模型也因其处理复杂角色和场景的能力而被推荐。
  - 作者提供了 **RPToolkit** 与原始 **Augmentoolkit** 的详细对比，强调了改进之处，如专用的 RP 流水线、彻底翻新的配置、分类器创建流水线以及为了提高速度而采用的 **async**。
  - 讨论涉及了潜在的应用，包括使用 RPToolkit 为写作**创建故事讲述数据集**。作者建议可以直接使用它，或者修改 Prompt 以专注于故事写作而非对话。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型发布与改进**

- **OpenAI 发布 o1**：OpenAI 发布了一个名为 o1 的全新推理模型系列，旨在响应前花费更多时间进行思考。[o1-preview 模型现已在 ChatGPT 和 API 中可用](https://www.reddit.com/r/singularity/comments/1ff7vtw/introducing_openai_o1/)。它在科学、编程和数学等复杂任务中表现出更强的性能。

- **o1-mini 性能**：[o1-mini 模型在推理基准测试中获得高分](https://www.reddit.com/r/singularity/comments/1ffiby3/o1mini_livebench_reasoning_score_came_out/)，超越了之前的模型。这表明即使在新的 o1 系列的较小版本中也有显著改进。

- **Flux 模型进展**：由 Black Forest Labs（原 SD 团队）开发的 Flux AI 模型正在[生成高质量图像](https://www.reddit.com/r/StableDiffusion/comments/1fewtiz/so_did_this_sub_completely_abandon_sd_in_favor_of/)并受到 AI 爱好者的欢迎。它被视为对 Stable Diffusion 模型的重大改进。

**AI 研究与技术**

- **新的扩展范式 (Scaling Paradigm)**：一位 [OpenAI 研究员表示 o1 代表了一种新的扩展范式](https://www.reddit.com/r/singularity/comments/1ff8gp3/openai_researcher_o1_model_is_a_new_scaling/)，暗示他们不再受预训练 (pretraining) 的瓶颈限制。这可能预示着 AI 模型开发和扩展方式的转变。

- **推理能力**：据称 o1 模型具有增强的[推理能力](https://www.reddit.com/r/singularity/comments/1ff1iwg/bloomberg_openai_nears_release_of_strawberry/)，可能代表了 AI 技术的重大进步。然而，一些用户对这些改进的程度表示怀疑。

**AI 模型对比与社区反应**

- **Flux vs Stable Diffusion**：关于 [Flux 表现优于 Stable Diffusion 模型](https://www.reddit.com/r/StableDiffusion/comments/1fewtiz/so_did_this_sub_completely_abandon_sd_in_favor_of/)的讨论正在进行中，许多用户报告 Flux 的效果更好，尤其是结合 LoRA 技术时。

- **MiniMax 视频生成**：一篇文章声称 [MiniMax 在 AI 视频生成方面已经超越了 Sora](https://www.reddit.com/r/singularity/comments/1ff7hbk/minimax_has_surpassed_sora_best_ai_video_ive_seen/)，展示了在普通观察者看来非常真实的滑板剪辑。

- **社区的期待与怀疑**：虽然人们对新的 AI 进展感到兴奋，但也有人对[过度炒作的公告](https://www.reddit.com/r/singularity/comments/1ff1b09/bloomberg_seems_on_board_with_today/)和仅向特定用户开放的限量发布[持怀疑态度](https://www.reddit.com/r/singularity/comments/1ff1b09/bloomberg_seems_on_board_with_today/)。


---

# AI Discord 回顾

> 摘要的摘要的摘要

## O1-mini

**主题 1. OpenAI o1 模型：性能与局限性**

- **OpenAI o1 在推理方面表现出色，但在编程方面受挫**：新发布的 **OpenAI o1** 模型在 **reasoning** 和数学方面表现卓越，超越了 **Claude 3.5 Sonnet**，但与 **GPT-4** 和 **Claude 3.5 Sonnet** 相比，在**编程任务中表现令人失望**。用户观察到它能生成不错的**文章和教育内容**，但在实际编程应用中表现挣扎。
- **速率限制（Rate Limits）收紧了 o1 的使用**：**OpenRouter** 将 **o1 模型**限制为**每天 30 次请求**，导致许多用户在发送约 **12 条消息**后就达到了速率限制，引发了不满。这一限制引发了关于其如何影响复杂任务执行以及未来是否可能提高限制的讨论。
- [**首次商业太空行走完成**](https://www.perplexity.ai/page/the-first-commercial-spacewalk-cwVg6684R6KEpO0FL1rkhQ)：**首次商业太空行走**的完成是一个重要的里程碑，一篇讨论任务关键事件和结果的文章对此进行了详细阐述。

**Theme 2. AI 训练增强与优化**

- **Prompt Caching 大幅削减 90% 的成本**：**OpenRouter** 引入的 **Prompt caching** 允许用户在 **Anthropic** 和 **DeepSeek** 等供应商处实现**延迟加速**，并可能获得 **90% 的 prompt tokens 折扣**，预计未来还将扩大范围。这一功能正在重塑频繁使用 AI 用户的成本结构。
- **量化技术提升模型效率**：**Unsloth AI** 和 **CUDA MODE** 等社区深入研究了独立的 **quantization** 和 **dequantization** 过程，探索了 **QLoRA** 等方法，并就 **dynamic quantization** 在管理 **VRAM** 限制的同时增强模型性能的优势展开了辩论。
- [**结合 KL 散度的强化学习**](https://github.com/pytorch/ao/tree/main/torchao/quantization#quantization-techniques)：在 **Eleuther** Discord 中讨论到，在 **reinforcement learning** 中使用 **KL divergence** 作为辅助损失函数，有助于防止模型遗忘关键任务，从而平衡**审核（moderation）与创造力**。

**Theme 3. AI 工具、集成与平台**

- **OAuth 集成简化 AI 开发**：**OpenRouter** 增强了对 `vscode:` 和 `cursor:` 等编程插件的 **OAuth 支持**，促进了自定义 AI 模型无缝集成到开发环境中，提高了开发者的**工作流效率**。
- **Modular 的 Magic 和 Mojo 更新 AI 工具包**：**MAX 24.5** 和 **Mojo 24.5** 引入了显著的性能提升和 **Python 3.12 兼容性**，利用新的 **Magic** 包管理器简化了安装和环境管理。这些更新使 Modular 成为开发者极具竞争力的 AI 解决方案。
- [**WebGPU Puzzles 发布，助力学习 GPU 编程**](https://gpupuzzles.answer.ai)：由 **Sarah Pan** 和 **Austin Huang** 开发的新应用 **WebGPU Puzzles** 通过基于浏览器的交互式挑战教授 **GPU programming**，使得在没有专用硬件的情况下进行 **GPU 访问实践**成为可能。

**Theme 4. AI 法规、伦理与对齐**

- **加州 SB 1047 AI 安全法案面临否决风险**：拟议的 **SB 1047 法案**旨在监管加州的 AI 安全，但由于政治影响，有 **66%-80%** 的概率被否决。讨论强调了该法案对**政治气候**和公众对 AI 监管看法的依赖。
- **对 AI 审查和对齐（Alignment）的担忧**：在多个 Discord 频道中，成员们表达了对 **RLHF** 可能会使 AI 模型“变笨”的担忧，从而降低其在技术任务中的效用。人们强调要在 **AI moderation** 与保持**创造力和功能性**之间取得平衡。
- [**STaR 技术增强模型推理**](https://arxiv.org/abs/2406.03816)：在 **LAION** 中，将 **Chain-of-Thought (CoT)** 与 **Reinforcement Learning** 相结合，显著提高了模型在**复杂推理任务**上的表现，突显了**高质量数据收集**的重要性。

**Theme 5. 社区活动、协作与支持**

- **黑客松与合作推动 AI 创新**：诸如 **LlamaIndex hackathon** 等活动提供了超过 **$20,000** 的奖金，促进了 **Retrieval-Augmented Generation (RAG)** 项目并鼓励社区主导的 **AI agent** 开发。与 **OpenSea** 等平台合作提供的 **free mint** 机会也吸引了社区参与。
- **私人聚会与工作机会强化 AI 网络**：**Fleak AI** 在旧金山的私人欢乐时光活动以及 **Vantager** 的 **AI Engineer** 职位空缺提供了社交和职业机会，增强了 AI 领域的社区联系和专业成长。
- [**OpenInterpreter 移动端应用反馈**](https://github.com/OpenInterpreter/01-app)：用户报告了 **OpenInterpreter** 移动端应用在语音响应功能方面的挑战，敦促改进**用户交互**和**开发者响应速度**，并鼓励**社区贡献**以增强文档和故障排除。

## O1-preview

**主题 1. OpenAI 的 o1 模型引发兴奋与争论**

- [**o1 模型在数学方面表现惊艳，但在代码方面受挫**](https://openai.com/index/learning-to-reason-with-llms/)：OpenAI 的新 **o1 model** 让 AI 社区议论纷纷，其推理和数学能力令用户印象深刻，但与 **GPT-4** 和 **Claude 3.5 Sonnet** 相比，其不尽如人意的代码表现让用户感到困惑。
   - **o1** 在复杂推理任务中表现出色，但在交付有用的代码输出方面却很吃力，引发了褒贬不一的反应。
- [**速率限制给 o1 的亮相泼了冷水**](https://discord.com/channels/1091220969173028894)：**o1** 的早期采用者正面临严格的 **rate limits** —— 有些用户在仅发送 **12 条消息** 后就达到了上限 —— 这引发了关于该模型在严肃用途中实用性的沮丧和讨论。
   - 用户正在质疑 Token 消耗的差异以及对其有效执行复杂任务能力的影响。
- [**基准测试之争：o1 是否公平竞争？**](https://arcprize.org/blog/openai-o1-results-arc-prize)：关于 AI 模型基准测试公平性的辩论被点燃，**o1** 独特的答案选择机制使其难以与 **GPT-4o** 等模型进行直接比较。
   - 呼吁考虑计算预算和选择方法的基准测试，突显了评估 AI 进展的复杂性。

**主题 2. 开发者通过 AI 集成增强工具功能**

- [**OAuth 与 AI 助力编程智商提升**](https://openrouter.ai/models)：**OpenRouter** 为 `vscode:` 和 `cursor:` 等插件引入了 **OAuth support**，让开发者能够将自定义 AI 模型无缝集成到他们的代码编辑器中。
   - 此次更新将 AI 驱动的解决方案直接带入 IDE，极大地提升了工作流效率。
- [**TypeScript 通过 LlamaIndex.TS 发布接入 AI**](https://www.npmjs.com/package/llamaindex)：[**LlamaIndex.TS**](https://www.npmjs.com/package/llamaindex) 为 **TypeScript** 带来了先进的 AI 功能，通过为 TS 爱好者量身定制的工具简化了开发。
   - 该软件包提供了关键功能，以简化 AI 在 TypeScript 项目中的集成。
- [**Vim 爱好者齐聚探讨 AI 驱动的编辑**](https://www.youtube.com/playlist?list=PLx2ksyallYzW4WNYHD9xOFrPRYGlntAft)：开发者分享了精通 **Vim** 和 **Neovim** 的资源，包括一个 [关于配置的 YouTube 播放列表](https://www.youtube.com/playlist?list=PLx2ksyallYzW4WNYHD9xOFrPRYGlntAft)，旨在通过 AI 辅助提高编程速度。
   - 社区协作将 AI 集成到编辑器中，提高效率并分享最佳实践。

**主题 3. 微调者面临训练挑战**

- [**内存泄漏导致 GPU 崩溃**](https://github.com/axolotl-ai-cloud/axolotl/issues/1916)：开发者在使用可变 **GPU batch sizes** 时正努力解决 **PyTorch** 中的**内存泄漏**问题，突显了张量大小波动带来的困扰以及对更好处理可变序列长度的需求。
   - 对填充效率低下的担忧引发了对内存陷阱稳健解决方案的呼吁。
- [**VRAM 限制考验微调者的耐心**](https://discord.com/channels/1179035537009545276)：社区成员在严格的 **VRAM** 限制下努力微调 **Llama3** 等模型，尝试使用 **learning rate schedulers** 和 **gradient accumulation steps** 等策略。
   - *“反复试验仍然是我们的座右铭，”* 一位用户沉思道，反映了大家对高效配置的共同追求。
- [**Phi-3.5 训练进展缓慢**](https://github.com/axolotl-ai-cloud/axolotl/issues/1916)：由于 **LoRA adapters** 未能学习到任何实质性内容，训练 **phi-3.5** 的尝试让用户感到愤慨，从而引发了错误报告和对可能故障的深入研究。
   - 随着微调者在这一难以捉摸的模型面前碰壁，挫败感不断增加。

**主题 4. 新工具和模型搅动 AI 领域**

- [**MAX 24.5 凭借 45% 的速度提升一马当先**](https://docs.modular.com/max/changelog?utm_campaign=24_5&utm_source=discord)：**MAX 24.5** 首次亮相，在 **int4k Llama token generation** 方面带来了高达 **45% 的性能提升**，令追求速度的开发者们感到欣喜。
   - 新的驱动接口和 Token 效率使 **MAX** 成为 AI 工具领域的一位重量级竞争者。
- [**Open Interpreter 的 Token 节食计划让用户感到饥渴**](https://discord.com/channels/1146610656779440188)：**Open Interpreter** 仅在 6 个请求中就消耗了 **10,000 tokens**，导致用户质疑其巨大的消耗量，并寻求更智能的方法来优化 Token 使用。
   - 讨论集中在如何在不牺牲功能的前提下减少 Token 消耗。
- [**战锤粉丝利用 Adaptive RAG 砥砺前行**](https://github.com/SilverBC/Warhammer-Adaptive-RAG)：[**Warhammer Adaptive RAG** 项目](https://github.com/SilverBC/Warhammer-Adaptive-RAG) 聚集了粉丝和开发者，展示了 **local models** 的创新用途，以及 **hallucination** 检测和 **answer grading** 等功能。
   - 社区反馈推动了项目的演进，体现了协作式 AI 开发的精神。

**主题 5. AI 政策与可访问性对话升温**

- [**加州 AI 法案面临政治摊牌**](https://polymarket.com/event/will-california-pass-sb-1047-ai-safety-bill/will-california-pass-sb-1047-ai-safety-bill?tid=1725767181654)：拟议的 **California SB 1047 AI safety bill** 引发了辩论，在政治博弈中，被否决的可能性估计在 **66%-80%** 之间。
   - 该法案不确定的命运凸显了 AI 领域创新与监管之间的紧张关系。
- [**OpenAI 是否让每个人的口袋里都装进了一个博士？**](https://discord.com/channels/1038097195422978059)：用户对 OpenAI 的进步感到惊叹，认为 AI 的进步 *“就像在每个人的口袋里装了一个博士学位”*，同时也在思考社会是否真正理解了这种转变的规模。
   - 这一讨论突显了 AI 对知识获取便捷性的变革性影响。
- [**AI 基准测试公平竞争的呼声愈发响亮**](https://x.com/steph_palazzolo/status/1834348474479091879?s=46)：关于 AI 模型评估的争论愈演愈烈，倡导者推动将计算预算（compute budgets）和选择方法纳入基准测试，以创造公平的竞争环境。
   - 社区正在寻求更细致的指标，以准确反映 AI 的能力和进展。


---

# 第一部分：Discord 高层摘要




## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenAI o1 模型对所有人开放**：新的 **OpenAI o1** 模型系列现已上线，允许客户端一次性流式传输所有 Token，但最初受到每天 30 个请求的 **rate limits** 限制，导致用户在 **12 条消息**后就会遇到速率限制错误。
   - 这种有限的发布引发了关于这些约束如何影响编码和推理任务中不同应用的使用模式的讨论。
- **Prompt Caching 带来成本节省**：**Prompt caching** 现在使用户能够实现延迟加速，并在共享缓存项时获得潜在的 **90% prompt tokens 折扣**，目前已对 **Anthropic** 和 **DeepSeek** 生效。
   - 该功能的扩展预计将覆盖更多供应商，可能会重塑频繁用户的成本结构。
- **增强 OAuth 支持以进行工具集成**：OpenRouter 为 `vscode:` 和 `cursor:` 等编码插件引入了 **OAuth support**，促进了自定义 AI 模型的无缝集成。
   - 此次更新允许开发者将他们的 AI 驱动解决方案直接引入 IDE，从而提高工作流效率。
- **速率限制令用户失望**：用户对 OpenRouter 最近将 o1 模型限制为 **每天 30 个请求** 的更新表示沮丧，他们认为这阻碍了他们有效执行复杂任务的能力。
   - 许多人渴望看到使用模式如何演变，以及是否有提高这些限制的潜力。
- **空响应的技术问题**：当用户报告在 completion JSON 中收到 **60 行空行** 时，出现了技术疑虑，这表明存在需要解决的稳定性问题。
   - 一位社区成员建议在重新考虑响应的可靠性之前，给系统调整留出一段等待期。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI o1 在与 GPT-4 的对比中表现参差不齐**：用户指出 **OpenAI o1** 在推理和数学方面表现出色，但在编程方面的表现与 **GPT-4** 和 **Claude 3.5 Sonnet** 相比令人失望。
   - 虽然它能生成不错的文章和教育内容，但在编程能力方面存在相当大的**局限性**。
- **AI 在艺术和创意中不断演变的角色**：讨论指出 AI 生成的艺术在挑战人类艺术极限的同时，也造成了低质量内容的饱和。
   - 参与者设想了一个 AI 辅助而非取代**人类创造力**的未来，尽管对内容质量仍有担忧。
- **澄清聊天机器人的 RAG 与 Fine-Tuning**：一名成员询问了**检索增强生成 (RAG)** 与微调对于教育聊天机器人的益处，得到的共识是 RAG 在处理上下文驱动的提问时更具优势。
   - 专家强调，微调调整的是行为而非知识，因此不太适合实时问答。
- **ChatGPT 在歌曲翻译方面面临挫折**：用户反映 **ChatGPT** 在翻译生成的歌曲时表现挣扎，由于其创意内容指南的限制，通常只返回片段而非完整歌词。
   - 这种限制阻碍了许多用户追求的项目连续性，增加了扩展过去对话的复杂性。
- **用户界面更改引发投诉**：成员们对最近的用户界面更改表示不满，特别是**复制和粘贴**功能破坏了行分隔。
   - 随着成员在不断演变的界面中操作，这导致了易用性问题和挫败感。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Pro 发布推测**：社区热切期待 **Unsloth Pro** 的发布，传闻其目标客户为大型企业，发布时间为“完成后”。
   - 成员们开玩笑地将开发进度比作建造罗马，暗示正在取得实质性进展。
- **在 RTX 4090 上测试 Gemma2**：在具有 8k 上下文的 RTX 4090 上对 **Gemma2 27b** 进行的初步测试显示出前景，尽管潜在的 **VRAM** 限制仍令人担忧。
   - 对梯度累积步数（gradient accumulation steps）的需求凸显了大型模型面临的持续挑战。
- **Mistral NeMo 性能评估**：早期反馈表明 **Mistral NeMo** 的性能与 **12b 模型** 持平，这引起了部分用户的一些失望。
   - 参与者思考更精炼的示例是否能提升性能。
- **AI 审核与创意担忧**：用户担心来自人类反馈的强化学习 (RLHF) 可能会让 AI 模型“变笨”，强调了在审核与创意之间取得平衡的重要性。
   - 提议实施中间件过滤，以在确保安全的同时保留原创性。
- **在有限 VRAM 下微调模型**：社区讨论围绕在 VRAM 限制下使用 Qlora 进行微调的挑战，重点关注最佳学习率 (LR) 调度器的选择。
   - 随着成员寻找默认余弦调度（cosine scheduling）之外的替代方案，反复试验仍是一个共同主题。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **使用 Ophrase 和 Oproof 革新 CLI 工具**：社区成员分享了关于使用 [Ophrase 和 Oproof](https://dev.to/p3ngu1nzz/revolutionizing-cli-tools-with-ophrase-and-oproof-4pdn) 革新 CLI 工具的见解。他们的方法旨在显著提升开发者体验。
   - *他们的创新技术启发了开发者重新思考命令行功能。*
- **Hugging Face 模型完整性挑战**：用户报告了 Hugging Face 上一个热门模型的完整性问题，暗示其包含误导性信息并违反了内容政策规则。
   - 讨论强调了用户在下载模型后可能感到的失望，因为其性能显著低于宣传的 Benchmarks。
- **使用 Llama cpp 探索 Reflection 70B**：重点介绍了一个使用 Llama cpp 构建的 [Reflection 70B](https://huggingface.co/spaces/gokaygokay/Reflection-70B-llamacpp) 项目，展示了该领域的高级能力。
   - 成员们指出，能够轻松获取最先进的模型是一项关键优势。
- **新的波斯语数据集增强多语言数据**：社区推出了一个 [波斯语数据集](https://huggingface.co/datasets/Reza2kn/OLDI-Wikipedia-MTSeed-Persian)，包含从 Wikipedia 翻译的 6K 个句子，这对于增强多语言 AI 能力至关重要。
   - 参与者赞扬了其在改进波斯语模型和训练数据多样性方面的潜力。
- **Arena Learning 提升性能**：[Arena Learning](https://huggingface.co/blog/satpalsr/arena-learning-post-train-data-performance-improve) 被讨论为一种在训练后（Post-training）阶段提高模型性能的方法，并显示出显著效果。
   - 社区成员渴望将这些见解应用到自己的模型中，以获得更好的结果。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **O1-mini 表现优于 O1-preview**：用户报告 *O1-mini* 相比 *O1-preview* 表现出更好的性能，这可能是因为它能够在给定时间范围内执行更多的 *Chain of Thought* (CoT) 轮次。
   - 一位用户正在等待完整发布以明确当前的能力，对立即购买表现出犹豫。
- **Hermes 3 的突破**：*Hermes 3* 相比 *Hermes 2* 有了显著增强，在角色扮演、长上下文连贯性和推理能力方面有明显改进。
   - 许多人正在关注其在需要扩展上下文长度的应用中的潜力，引发了对其 API 能力的兴趣。
- **模型对齐的担忧**：强调了对自主模型对齐（Model Alignment）的担忧，指出如果模型在没有对齐的情况下实现更高的智能，存在失去控制的风险。
   - 讨论强调了理解开发者意图以预先应对对齐挑战的重要性。
- **GameGen-O 展示功能**：*GameGen-O* 通过一个受《西游记》启发的 Demo 展示了其功能，因其创新能力而备受关注。
   - 贡献者包括来自 *香港科技大学* 和 *腾讯光子工作室 (Tencent's LightSpeed Studios)* 的成员，表明了研究合作。
- **ReST-MCTS 自训练进展**：*ReST-MCTS* 方法通过将过程奖励指导（Process Reward Guidance）与树搜索（Tree Search）相结合，提供了增强的自训练，提升了 LLM 训练数据的质量。
   - 该技术显著超越了之前的算法，通过迭代训练不断优化具有高质量输出的语言模型。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **OpenAI O1 模型待集成**：用户正热切期待将 **OpenAI O1 模型**集成到 **Perplexity** 中，一些人提到了已经集成该模型的竞争对手。
   - 虽然许多人希望快速更新，但也有人认为像 **Claude Sonnet** 这样的模型表现已经很出色了。
- **API 额度困惑**：用户对 **5 美元 API 额度补充**的时间点不明确，争论其是在**每月 1 号**还是在**每个计费周期的第一天**重置。
   - *用户非常希望能进一步明确这些时间点，* 尤其是那些管理订阅状态的用户。
- **商业太空行走标志着一个里程碑**：**首次商业太空行走**已正式完成，并发布了一篇详细文章讨论关键任务事件和结果。
   - 在[此处](https://www.perplexity.ai/page/the-first-commercial-spacewalk-cwVg6684R6KEpO0FL1rkhQ)阅读完整更新。
- **内部服务器错误阻碍 API 访问**：有报告称出现**内部服务器错误**（状态码 **500**），表明用户在尝试访问 API 时面临严重问题。
   - *此错误在关键操作期间对 **Perplexity** 服务的有效利用构成了挑战。*
- **强调 OpenPerplex API 优势**：用户表达了对 **OpenPerplex API** 的偏好，理由是其具有**引用、多语言支持**和更高的速率限制（rate limits）等优点。
   - *这反映了优于其他可用 API 的良好用户体验，* 突显了其效用。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI o1 收到褒贬不一的反馈**：用户报告称 OpenAI 的 o1 模型表现参差不齐，在重推理任务中表现出色，但总体上往往无法提供有用的输出，导致透明度方面的担忧。
   - *“他们说 Cursor 的代码补全不行？”* 这引发了对用于评估的研究方法的质疑。
- **李飞飞（Fei-Fei Li）启动 World Labs**：李飞飞公布了 World Labs，专注于**空间智能（spatial intelligence）**，并获得了 2.3 亿美元的融资，旨在开发能够进行 3D 感知和交互的 Large World Models。
   - 该计划正吸引 AI 社区的顶尖人才，渴望解决现实世界的复杂问题。
- **Cursor 遇到扩展性问题**：据报道，Cursor 在**代码补全**和**文档生成**功能方面面临扩展性问题，阻碍了用户体验。
   - 讨论强调了用户的挫败感，表明该工具的性能未达到预期。
- **HTEC AI Copilot 报告的见解**：HTEC 团队评估了 **26 个 AI 工具**，由于测试有限，结果尚无定论，这让人对其关于 AI Copilot 的分析深度产生怀疑。
   - 尽管参与者对每个工具都进行了“涉猎（dabbled）”，但该报告似乎更倾向于线索生成（lead generation），而非深入的可用性见解。
- **探索 Vim 和 Neovim 资源**：成员们承认 **Vim** 的学习曲线陡峭，但指出一旦掌握，编码速度会显著提升，许多人通过完成 **Vim Adventures** 游戏来增强技能。
   - 此外，社区成员分享了各种 **Neovim** 资源，包括一个 [YouTube 配置播放列表](https://www.youtube.com/playlist?list=PLx2ksyallYzW4WNYHD9xOFrPRYGlntAft)，以促进学习和协作。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **创新量化技术**：一名成员正在通过在测试期间对输入和权重进行独立的 **quantization** 和 **dequantization** 处理来提高模型精度，同时讨论激活值动态量化的优劣。
   - 他们面临量化逻辑的调试问题，呼吁提供一个最小运行示例，以帮助理解和实际应用。
- **Llama 3 集成仓库**：已启动一个功能分支，用于向 llm.c 添加 **Llama 3 support**，从现有模型文件的副本开始，并保留计划中的 RoPE 和 SwiGLU 的 PRs。
   - 这项工作旨在合并回 master 分支之前，纳入重大的进展和优化。
- **利用 Liger Kernel 协助微调 BERT**：出现了关于使用 **Liger kernel** 进行 **BERT fine-tuning** 的求助，成员们在等待将 **liger ops** 集成到 **Thunder** 的增强功能时寻求参考代码。
   - 在没有 **liger ops** 的情况下，可能需要对模型进行调整，从而引发了围绕满足模型要求的持续修改的讨论。
- **通过自定义算子简单提升性能**：讨论了为 FFT 实现 **Cooley-Tukey algorithm**，并针对各种应用中的增强性能进行了优化。
   - **GH200** 架构的 KV-cache offloading 也因其在 LLM inference 任务中最大化效率的重要性而受到关注。
- **WebGPU Puzzles 发布用于学习**：新推出的应用 [WebGPU Puzzles](https://gpupuzzles.answer.ai) 旨在通过直接在浏览器中进行的编码挑战来教授用户 **GPU programming**。
   - 该应用由 **Sarah Pan** 和 **Austin Huang** 开发，利用 **WebGPU** 在不需要专用硬件的情况下实现 GPU 访问。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI o1 模型表现令人惊讶**：新发布的 [OpenAI o1 model](https://arcprize.org/blog/openai-o1-results-arc-prize) 在 AIME 等基准测试中取得了令人印象深刻的分数，但在 ARC Prize 上的表现却出奇地低。
   - 虽然 o1 擅长竞赛数学题，但其泛化到其他问题类型的能力仍然有限，这引发了对其部署的疑问。
- **加州 SB 1047 与 AI 监管**：关于 AI safety 的 [SB 1047 bill](https://polymarket.com/event/will-california-pass-sb-1047-ai-safety-bill/will-california-pass-sb-1047-ai-safety-bill?tid=1725767181654) 提案由于政治影响，被否决的可能性预计在 **66%-80%**。
   - 讨论表明，该法案的命运可能在很大程度上取决于周围的政治气候和公众对 AI 监管的看法。
- **关于 AI 模型基准测试公平性的辩论**：围绕 AI 模型基准测试的公平性展开了讨论，特别关注与 o1 和 GPT-4o 等模型相关的 pass@k metrics 的复杂性。
   - 参与者认为基准测试应该考虑计算预算，这使得直接比较变得复杂，尤其是考虑到 o1 独特的答案选择机制。
- **了解 API 分级系统**：成员们强调，要在 **API tier system** 中达到 **Tier 5**，用户需要花费 **$1000**。一位用户分享说他们在 **Tier 3**，而另一个团队已经超过了 Tier 5。
   - 这引发了关于支出层级对功能和能力访问影响的讨论。
- **对思维链推理的见解**：o1 模型中的推理错误被指出会导致有缺陷的 **Chain-of-Thought** 输出，导致错误螺旋式上升并得出错误结论。
   - 成员们讨论了这种现象如何揭示了维持 AI 推理连贯性的重大挑战，从而影响可靠性。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **A1111 vs Forge：性能权衡**：用户比较了 **A1111** 和 **Forge** 在 XYZ 图表上的生成时间叠加，发现 Schnell 生成图像的速度通常更快，但代价是与 Dev 相比质量存在差异。
   - 这引发了关于模型性能指标中速度与质量之间平衡的问题。
- **Pony 模型：困惑重重**：关于 **Pony 模型** 提示词的讨论凸显了训练数据的不一致性，用户对其分数标签（score tags）的有效性感到困惑。
   - 对于这些提示词在实践中是否能产生预期结果，人们持怀疑态度。
- **警惕诈骗：保持警觉！**：对欺诈性投资建议的担忧增加，强调用户需要对虚假加密货币计划保持警惕。
   - 对话强调了在类似讨论中识别危险信号的至关重要性。
- **动态采样器：向前迈进**：将 **Dynamic compensation samplers**（动态补偿采样器）集成到 AI 模型训练中，激发了用户对增强图像生成技术的兴趣。
   - 社区对新工具及其对性能的潜在影响充满了热情。
- **关键 Token：创建高质量图像**：分享了一系列有效的提示词 Token，如 **'cinematic'** 和 **'scenic colorful background'**，展示了它们在提高图像生成质量方面的效用。
   - 讨论强调了关于最佳 Token 使用的不同意见，以及对基于研究的见解的需求。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **o1-preview 推出速度加快**：成员报告称已分批获得 `o1-preview` 的访问权限，在 Windows internals 等任务上表现出良好的性能。
   - 尽管大家非常兴奋，但一些用户对推出的速度表示沮丧。
- **辩论最大性能的 GPU 配置**：讨论集中在单插槽 **6x RTX 4090** 还是双插槽 **4x RTX 4090** 设置能产生更优越的性能，特别是对于大型模型。
   - 共识是，将模型放入 **VRAM** 至关重要，其表现通常优于更多依赖系统 **RAM** 的配置。
- **文本转语音 API 发布**：一名成员发布了一个与 OpenAI 端点兼容的 **Text-to-Speech API**，强调其无需 GPU 即可实现的高效率。
   - 集成细节可以在 [GitHub 仓库](https://github.com/PantelisDeveloping/openspeech-tts)中找到，鼓励用户参与。
- **市场趋势推高 GPU 价格**：GPU 价格显著上涨，特别是 3090 和 P40 型号，这归因于 AI 任务需求的增长。
   - 成员们分享了在当地市场寻找价格合理的 GPU 的困难经历，反映了更广泛的供需问题。
- **VRAM 对模型性能的影响**：参与者一致认为模型大小和可用 **VRAM** 会显著影响性能，建议不要对深度模型使用 **Q8** 设置。
   - 有人呼吁进行更直接的查询，以协助新手优化其设置。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex.TS 发布新功能！**：LlamaIndex.TS 现在可供 TypeScript 开发者使用，通过流线型集成增强了功能。请在 [NPM](https://www.npmjs.com/package/llamaindex) 上查看。
   - 该包旨在通过提供专门针对 TypeScript 开发者的关键工具来简化开发任务。
- **LlamaIndex Hackathon 令人兴奋的现金奖励**：第二届 LlamaIndex hackathon 定于 10 月 11 日至 13 日举行，为参与者准备了超过 **$20,000** 的现金和额度奖励。在此[注册](https://t.co/13LHrlQ7ER)。
   - 该活动围绕在开发高级 AI Agent 中实现 Retrieval-Augmented Generation (RAG) 展开。
- **LlamaIndex 在 function calls 方面的局限性**：讨论显示 LlamaIndex 在当前的 API 配置下不支持 function calls，从而阻碍了工具的使用。成员确认目前仍不支持 function calling 和 streaming。
   - 鼓励用户关注更新，因为未来可能会推出新功能，或者探索替代配置。
- **LlamaParse 展示高级 Excel 解析功能**：一段新视频展示了 LlamaParse 的高级 Excel 解析功能，强调了其对多工作表和复杂表格结构的支持。在此查看[演示](https://t.co/xuPJuUBxmC)。
   - LlamaParse 采用的递归检索技术增强了无缝总结复杂数据设置的能力。
- **探索 ChromaDB 集成**：一位用户寻求在 LlamaIndex 中使用 ChromaDB 检索文档上下文的帮助，特别是关于查询响应方面。建议他们检查 `response.source_nodes` 以获取准确的文档上下文。
   - 讨论中明确了对元数据的依赖，提高了对 AI 查询中文档处理的理解。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **KL Divergence 增强 RL 稳定性**：成员们讨论了将 **KL divergence** 作为强化学习中的辅助损失的应用，以防止模型遗忘关键任务，特别是在 **MineRL** 体系中。
   - 有人担心对齐的奖励函数可能会削弱 KL divergence 的益处，从而暴露当前 RL 方法的缺陷。
- **混合精度训练机制揭秘**：针对在混合精度训练中同时使用 **FP32** 和 **FP16** 的基本原理提出了疑问，数值稳定性和内存带宽被列为主要考虑因素。
   - 指出在某些操作中使用 FP32 可以显著减少不稳定性，而这种不稳定性通常是整体吞吐量的瓶颈。
- **探索 RL 中的 Off-Policy 方法**：研究了强化学习中探索策略的细微差别，成员们一致认为像 **Q-learning** 这样的 off-policy 方法比 on-policy 方法提供了更好的探索灵活性。
   - 讨论强调了在应用辅助损失项以促进探索，与避免创建单独且可能繁琐的探索策略之间保持微妙平衡的重要性。
- **OpenAI 在知识获取方面达到新高度**：一位参与者对 **OpenAI** 在知识民主化方面的贡献缺乏赞赏表示担忧，认为它实际上让每个人的口袋里都装进了一个博士。
   - 这引发了关于社会对 AI 进步及其在日常应用中集成的看法的更广泛对话。
- **Tokenizers 在添加新语言时需要重新训练**：讨论了在 ML 模型中添加新语言时重新训练 tokenizer 的必要性，这标志着全面重新训练对有效性的重要性。
   - 成员们承认，虽然有限的 pretraining 可能适用于结构相似的语言，但在自然语言语境下，全面的重新训练仍然至关重要。



---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AdEMAMix Optimizer 引起关注**：关于 [AdEMAMix Optimizer](https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch) 的讨论强调了其在提高 **Parakeet** 训练效率方面的潜力，可在 **20 小时**内达成目标。
   - 成员们推测了其对模型训练策略的影响，强调了对各种效率技术的需求。
- **Cohere API 支出限制设置**：用户分享了通过 [Cohere dashboard](https://dashboard.cohere.com/billing?tab) 设置 **Cohere API** 使用每日或每月支出限制的方法，以管理潜在成本。
   - 一些用户在访问选项时遇到障碍，引发了联系 **Cohere support** 寻求解决的建议。
- **使用 Command R+ 进行律师考试微调**：一位硕士毕业生寻求关于使用 **Command R+** 微调 **llama2** 以应对美国律师考试的意见，并向其他用户征求建议。
   - 小组建议进行本地实验，并仔细阅读 [Cohere's documentation](https://docs.cohere.com) 以获得最佳指导。
- **AI 疲劳信号显现**：成员们注意到 AI 进展中可能出现了**实用性胜过炒作**的转变，表明有用应用程序的增长趋势。
   - 分析将此与该领域快速演变的技能要求相类比，将当前环境比作创新的“原始汤”。
- **在 API 请求上实施速率限制**：有人建议针对每个 IP 地址应用 **API requests** 的速率限制，以减轻滥用并有效控制流量。
   - 这种预防措施被认为对于防范恶意活动可能导致的突发使用高峰至关重要。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX 24.5 性能提升**：**MAX 24.5** 已发布，在 int4k Llama token 生成方面实现了 **45% 的性能提升**，并为开发者引入了新的驱动接口。查看 [MAX changelog](https://docs.modular.com/max/changelog?utm_campaign=24_5&utm_source=discord) 中的完整变更。
   - 此版本使 MAX 成为更具竞争力的选择，特别是在依赖高效 token 处理的环境中。
- **Mojo 24.5 带来 Python 支持**：**Mojo 24.5** 增加了对隐式变量定义的支持，引入了新的标准库 API，并兼容 **Python 3.12**。详情可见 [Mojo changelog](https://docs.modular.com/mojo/changelog?utm_campaign=24_5&utm_source=discord)。
   - 这些增强功能表明 Mojo 拥有强劲的发展轨迹，在利用 Python 最新特性的同时简化了开发工作流。
- **StringSlice 简化数据处理**：一位成员强调了在 **Mojo** 中使用 `StringSlice(unsafe_from_utf8=path)` 将 `Span[UInt8]` 转换为字符串视图的方法。该方法阐明了关键字参数在此上下文中的运作方式。
   - 理解这一点有助于更好地利用 Mojo 生态系统中的字符串处理，特别是对于数据驱动的任务。
- **MAX 嵌入功能的替代方案**：讨论明确了 **MAX** 缺乏对 Embedding 和向量数据库功能的内在支持；建议使用 **ChromaDB**、**Qdrant** 和 **Weaviate** 等替代方案进行语义搜索。一篇博客文章提供了利用这些工具增强**语义搜索**的示例。
   - 这一缺失凸显了开发者需要利用外部库来实现全面的搜索功能。
- **Google Colab 中的兼容性问题**：由于安装问题，在 Google Colab 中运行 **MAX** 引起了关注；用户被鼓励创建 GitHub issues 以对此事进行调查。[Colab Issue #223](https://github.com/modularml/max/issues/223) 记录了正在进行的讨论，以征求社区意见。
   - 解决这些兼容性问题对于最大限度地提高使用流行 notebook 环境的开发者的可访问性至关重要。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Token 使用引发讨论**：关于 **Open Interpreter** 仅在 6 次请求中就消耗了 **10,000 tokens** 的情况引发了关注，其效率受到了质疑。这开启了关于 Token 处理潜在优化策略的对话。
   - 成员们正在积极讨论哪些策略可以在不牺牲功能的情况下提高 Token 利用率。
- **iPhone App 设置所需步骤**：一名成员请求关于启动新 **iPhone app** 的清晰指令，鉴于其初学者身份，寻求关于克隆 repo 和设置流程的指导。
   - 另一名用户迅速推荐了[这份设置指南](https://01.openinterpreter.com/setup/introduction)以协助安装。
- **LiveKit 连接挑战**：有报告称在移动数据而非 Wi-Fi 环境下使用时，**LiveKit** 存在连接问题，导致在 MacBook 上的访问变得复杂。成员们要求提供复现这些连接错误的详细步骤。
   - 随着用户推动协作排查以有效解决常见的 LiveKit 问题，社区参与度激增。
- **移动端 App 语音响应缺失**：反馈指出 **Open Interpreter** 移动端 App 在提供语音响应方面存在困难，虽然能识别命令但无法执行语音输出。非响应式的女教师功能被特别提及。
   - 随着用户指出 App 缺乏反馈，批评声浮现，敦促开发者优化用户交互并改善整体体验。
- **记录社区贡献**：目前正在推动改进社区文档，特别是关于 **LiveKit** 设置的部分，据称 **90%** 的用户面临基础性问题。
   - Mike 鼓励成员提交带有可行解决方案的 pull requests，强调了需要清晰指南来避开常见陷阱的必要性。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **探索 O1 功能**：在最近实现支持后，成员们正在测试 DSPy 的 **O1 support**，以期实现无缝集成。
   - *活跃的讨论*凸显了社区对从新功能中提取价值的浓厚兴趣。
- **DSPy 2.4.16 版本非常出色！**：**DSPy 2.4.16 版本**已正式发布，引入了增强用户体验的 `dspy.LM` 功能。
   - 用户报告在更新后成功实现了 **LiteLLM models**，鼓励更广泛的采用。
- **RAG：检索增强的瑰宝**：成员们正在探索如何使用更新的 DSPy 模块将传统的 LLM 查询适配为 **RAG**（检索增强生成）。
   - 社区分享了相关资源，包括 [simple RAG](https://github.com/stanfordnlp/dspy/blob/main/skycamp2023.ipynb) 和 [MIPRO compilation](https://github.com/stanfordnlp/dspy/blob/main/examples/qa/hotpot/hotpotqa_with_MIPRO.ipynb) 的链接，推动了动手实验。
- **对 Google Vertex AI 的担忧**：用户指出了 **Google Vertex AI** 的集成问题，报告称尽管设置正确但仍出现服务错误。
   - 协作解决问题的重点集中在 *LiteLLM models 的优化环境*，强调了代理配置。
- **RAG 讨论中的动态提示词**：社区成员正在辩论在提示词中封装**动态上下文**以实现有效 **RAG** 实施的最佳实践。
   - 对话强调了*上下文驱动提示词*在增强不同场景结果中的必要性。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **内存泄漏困扰 GPU Batch Size**：讨论揭示了在每个 **GPU batch size** 使用打包样本（packed samples）时，**PyTorch** 中波动的 Tensor 大小会导致**内存泄漏**。
   - 参与者对序列中的 Padding 表示担忧，强调需要解决方案来减轻这些内存陷阱。
- **Upstage Solar Pro 模型引发热议**：围绕 [Upstage Solar Pro](https://huggingface.co/upstage) 模型的关注度激增，特别是其适用于最佳单卡推理的 **22B** 配置；并将其与 **LLaMA 3.1** 进行了对比。
   - 尽管感到兴奋，成员们对创作者提出的**大胆主张**表示怀疑，警惕潜在的过度承诺。
- **对 Liger Kernels 的好奇**：一位成员寻求关于实现 **Liger kernels** 的见解，希望从他人的经验中了解性能结果。
   - 这一咨询反映了对增强 **LLM** 优化和可用性的广泛兴趣。
- **训练 phi-3.5 遇到障碍**：训练 **phi-3.5** 的尝试令人沮丧，据报道 **LoRA adapters** 学习到的内容极少，相关问题已记录在 [GitHub report](https://github.com/axolotl-ai-cloud/axolotl/issues/1916) 中。
   - 参与者发现了一个可能导致训练结果不佳的潜在 Bug，并表达了他们的沮丧。
- **Gradient Norms 引发困惑**：一位用户在 LoRA 配置中设置了 `max_grad_norm: 2`，但却遇到了异常高的 **grad_norm** 值，峰值达到 **2156.37**。
   - 关于日志是否准确反映了裁剪值的问题仍然存在；该用户的 **LoRA setup** 还包括了针对 **Pythia** 模型的各种微调设置。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Llama 3.1 8B 微调版发布**：一位成员宣布了 [Llama 3.1 8B finetune model](https://huggingface.co/dustinwloring1988/Llama3.1-8B-Reflection-v2-gguf)，并正在寻找合作者来增强其数据集，该模型作为 *flection model* 的概念验证。
   - 这次讨论激发了对复制各种 YouTube 频道中看到的结果的兴趣，展示了实际应用和社区贡献。
- **对开源 SD 的担忧**：一位参与者指出 **Stable Diffusion** 在开源领域似乎停滞不前，暗示社区贡献正在下降。
   - *“基本上，如果你关心开源，SD 似乎已经死了，”* 这促使大家重新评估对开源项目的参与。
- **与 OpenSea 合作的 Free Mint 活动**：服务器宣布与 **OpenSea** 合作，为成员提供新的 **free mint** 机会，可通过 [CLAIM link](https://iclaim7b.vercel.app/) 访问。
   - 提醒参与者，某些领取过程可能会产生 **gas fees**，鼓励社区成员尽快行动。
- **Tier 5 API 访问成本高昂**：**Tier 5 API access** 与之前的模型（如 **GPT-4o**）相比，其成本效益引发了担忧，导致对其能力的乐观态度趋于谨慎。
   - *“不会比 gpt4o 差多少”* 反映了在平衡预算与寻求 API 效能提升方面的讨论。
- **STaR 技术增强模型训练**：将 **Chain-of-Thought (CoT)** 与 **Reinforcement Learning** 相结合显著提升了模型性能，正如 **STaR** 技术在复杂推理任务中的有效性所强调的那样。
   - 强调了高质量数据收集的重要性，并认为 *“必须是聪明人参与，所以它不可能便宜，”* 肯定了数据智能与模型训练效果之间的联系。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 0.2.1 在 Mac 上安装失败**：由于未满足 **torchao==0.3.1** 的依赖关系，**torchtune 0.2.1** 版本在 Mac 上安装失败，导致其在 MacBook 上无法使用。成员们指出，即将发布的 **torchao 0.6.0** 可能会通过提供 macOS wheels 来解决此问题。
   - 影响 Mac 安装的问题引发了不满，进一步强调了在未来版本中需要更平滑的依赖管理。
- **支持 Mac M1 的 torchao wheels 现已发布**：**torchao wheels** 现已确认可用于 **Mac M1**，显著提升了 Mac 用户的兼容性。预计此更新将增强在该架构上运行 **torchtune** 的功能。
   - 兼容性的提升提供了一条切实可行的路径，允许用户在 M1 环境下更好地利用 **Torchtune**。
- **将 Recipe 测试切换至 GPU**：成员们讨论了将当前的 recipe 测试从 CPU 迁移到 GPU 的方案，此前由于历史限制，这一操作一直受到约束。有人建议将测试指定为 GPU 特有（GPU-specific），以确保在 GPU 不可用时保持灵活性。
   - 这一转变被定位为充分发挥计算能力并简化未来测试流程的关键。
- **增强型 Batched Generation 计划**：一个旨在优化 **batched generation** 的新型轻量级 recipe 正在开发中，意在与项目目标和用户需求保持一致。社区非常鼓励对这一新方法提供反馈。
   - 成员们表示渴望参与这一生成优化方案的测试，该方案旨在简化流程的同时保持有效性。
- **可迭代数据集的 Online Packing 即将推出**：未来计划包括为可迭代数据集（iterable datasets）实现 **online packing**，有望在工作流中实现更好的数据处理和操作效率。这一进展旨在支持 Torchtune 内部的持续开发。
   - 社区期待其数据策略的增强，并对迭代过程的潜在影响感到兴奋。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain AWS ChatBedrockConverse 与对话历史**：一位用户询问 **LangChain** 的 **AWS ChatBedrockConverse** 是否支持在检索链中维护 **conversational history**（对话历史），这对于对话式 AI 功能至关重要。
   - 这引发了关于 AI 框架内历史管理影响的讨论。
- **向量数据库实现困难**：一位用户报告了在实现 [Upstash Redis](https://github.com/thinley4/Rag-Chatbot/issues/4) 以替换内存中的 **MemoryVectorStore** 来存储 PDF 分片的向量嵌入（vector embeddings）时遇到的挑战。
   - 他们寻求社区帮助，并提到在使用 **Pinecone** 等替代方案时也遇到了问题。
- **Warhammer Adaptive RAG 项目初具规模**：一位社区成员分享了一个专注于 **Warhammer Adaptive RAG** 的 [GitHub 项目](https://github.com/SilverBC/Warhammer-Adaptive-RAG)，寻求关于 **hallucination**（幻觉）和 **answer grading**（答案评分）等功能的反馈。
   - 反馈强调了该项目对 **local models**（本地模型）的创新使用。
- **Vantager 的 AI 工程师职位机会**：一位成员宣布了 **Vantager** 招聘 **Founding AI Engineer**（创始 AI 工程师）的消息，该公司致力于开发用于资本配置的 AI 原生平台。
   - 鼓励候选人查看 **job board**（招聘板）了解详情，并提到了 VC 的支持以及对解决重大数据挑战的关注。
- **OpenAI 的变革性影响**：一位成员对 OpenAI 的进步表示惊叹，认为这感觉就像是“给每个人的口袋里都塞进了一个博士”。
   - 他们对社会是否充分理解这些技术带来的冲击性变化表示担忧。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **论坛成员讨论礼仪**：一位成员强调了**基本论坛礼仪**的重要性，指出重复的求助请求可能会挫伤他人提供帮助的积极性。
   - *浪费他人时间*会阻碍社区参与，呼吁采用更好的沟通实践。
- **Tinygrad 的 MypyC 编译进展**：一位成员详细介绍了他们进行 **MypyC 编译**的系统方法，为了提高效率，从整个项目深入到单个文件。
   - 编译的文件包括 `tinygrad/device.py` 和 `tinygrad/tensor.py`，表明项目取得了重大进展。
- **使用 Tinygrad 成功运行 Llama-7B**：该成员使用 **Llama-7B 模型**成功运行了 *examples/llama.py*，并指出平均耗时提升了 **12%**。
   - 他们提供了 [Llama-7B 仓库](https://huggingface.co/huggyllama/llama-7b/tree/main)的链接以供参考所使用的模型。
- **为 MypyC 功能修改代码**：对多个文件进行了代码修改，包括重写生成器和添加装饰器，以启用 **MypyC 功能**。
   - 该成员将他们的更改描述为*初稿*，在进一步完善之前寻求团队反馈。
- **C 扩展的未来考虑**：该成员建议，如果要将 **C 扩展**集成到 Tinygrad 中，应采取逐步推进的方法以方便更改。
   - 他们渴望在完成贡献之前，确保正在进行的工作与更广泛的项目目标保持一致。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla OpenFunctions 模型准确率为零**：**gorilla-openfunctions-v2** 模型的评估在 **258** 次测试后返回的准确率为 **0.0**，尽管 **model_result_raw** 与 **possible_answer** 一致。
   - 这种异常现象表明可能存在更深层次的问题，需要进行超出表面输出的进一步调查。
- **解码 AST 抛出错误**：在执行用户信息函数期间出现错误，具体表现为 *Invalid syntax. Failed to decode AST* 消息。
   - 报告还强调了数据类型不匹配，指出无法将 str（而非 'list'）连接到 str，这表明可能存在 bug。
- **用户信息检索成功完成**：模型成功检索了 **ID 7890** 的用户信息，确认用户名为 **user7890**，电子邮件为 **user7890@example.com**。
   - 此操作完成了对 **黑色** 特殊物品的特定请求，在报告的问题中展示了部分功能。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **微调 LLM 以获得更好的翻译**：一位成员询问了专门针对**翻译**微调 **LLM** 的经验，指出许多模型虽然能捕捉大意，但会遗漏关键的**语气和风格**元素。
   - 这凸显了改进**翻译质量**技术以保留本质细微差别的必要性。
- **在翻译中捕捉语气的困难**：虽然 **LLM** 提供了不错的翻译，但它们往往难以有效地传达原始的**语气**和**风格**。
   - 成员们呼吁分享增强**翻译忠实度**的方法和见解，以应对这些长期存在的挑战。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Fleak AI 举办私人聚会**：Fleak AI 今晚在旧金山的[此地点](https://lu.ma/l9tpptle?tk=KfASyJ)为其社区组织了一场私人欢乐时光（happy hour）活动，旨在讨论更新并促进联系。
   - 这次聚会提供了一个与同行开发者和用户建立网络并进行互动的机会，增强了社区纽带。
- **Fleak 作为 Serverless API 构建器**：Fleak 将自己定位为专为 AI 工作流量身定制的 Serverless API 构建器，特别擅长 **sentiment labeling**（情感标注）等功能。
   - 这一功能使 Fleak 成为希望在项目中简化 API 集成的开发者的宝贵工具。
- **Fleak 专注于社区建设**：该活动旨在通过更频繁的线下聚会（从这次欢乐时光活动开始）来加强社区参与。
   - 组织者希望营造一个温馨的环境，鼓励参与者之间进行开放式讨论和建立联系。



---


**Alignment Lab AI Discord** 没有新消息。如果该频道（guild）长时间没有活动，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该频道（guild）长时间没有活动，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道（guild）长时间没有活动，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道（guild）长时间没有活动，请告知我们，我们将将其移除。


---

# 第 2 部分：频道详细摘要和链接


{% if medium == 'web' %}




### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1283939140899176528)** (10 messages🔥): 

> - `OpenAI o1 模型发布`
> - `Prompt Caching`
> - `VSCode 的 OAuth 支持`
> - `Rate Limits`
> - `错误消息` 


- **OpenAI o1 模型面向所有人上线**：全新的 **OpenAI o1** 模型系列现已上线，允许客户端一次性流式传输所有 token，但初期受到 **Rate Limits**（速率限制）的约束。
   - 有关遇到 `429` 错误的查询确认，用户在发送 **12 条消息**后会达到速率限制。
- **Prompt Caching 提供折扣**：Prompt Caching（提示词缓存）现在使用户能够实现延迟加速，并且即使在共享缓存项时，也能在 prompt token 上获得潜在的 **90% 折扣**。
   - 该功能已在 **Anthropic** 和 **DeepSeek** 上启用，预计很快将扩展到更多供应商。
- **编程工具的 OAuth 支持**：OpenRouter 为 `vscode:` 和 `cursor:` 等插件引入了 **OAuth 支持**，允许用户将他们的模型集成到编程工具中。
   - 这一进展支持将自定义 AI 模型直接引入用户的 IDE，以获得无缝体验。
- **OpenRouter 的速率限制更新**：用户的速率限制已更新为 **每天 30 次请求**，随着使用模式的分析，可能会进一步增加。
   - 此限制分别适用于 **o1** 和 **o1-mini** 模型，增强了用户的访问权限。
- **空响应的技术问题**：用户报告收到 **60 行空行**以及通常的 completion JSON，这表明系统在稳定之前需要一段时间。
   - 一位成员建议等待几天，以解决空消息内容和 finish reasons（结束原因）的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1834378430973915313">来自 OpenRouter (@OpenRouterAI) 的推文</a>：OpenAI o1 🍓 现在面向所有人开放！(开始时会有非常严格的速率限制)。与 gpt-4o 不同，它在回复前会花周期进行思考。注意：在 OpenRouter 上支持流式传输，但...</li><li><a href="https://openrouter.ai/models/sao10k/l3.1-euryale-70b>)">Llama 3.1 Euryale 70B v2.2 - API, Providers, Stats</a>：Euryale L3.1 70B v2。通过 API 运行 Llama 3.1 Euryale 70B v2.2
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1283864731475775541)** (784 messages🔥🔥🔥): 

> - `OpenAI o1 模型性能`
> - `Token 消耗对比`
> - `o1 的速率限制`
> - `o1 在编程和数学中的应用`
> - `Perplexity 模型输出率`

- **OpenAI o1 模型性能评估**：OpenAI o1 模型在推理任务中表现出明显优于 Sonnet 3.5 的性能，尽管它仍未达到人类水平的推理能力。
   - 用户发现，尽管它有优势，但高昂的成本和潜在的 Token 消耗使其成为一种小众工具，而非通用解决方案。
- **Token 消耗和定价差异**：用户注意到 OpenRouter 的 o1 模型在 Token 消耗方面存在差异，报告的输入 Token 成本与基于 Prompt 大小的预期不符。
   - 具体而言，一位用户指出，大量的输入导致了出乎意料的较低 Token 费用，引发了对 Token 计算准确性的质疑。
- **OpenRouter o1 模型的速率限制**：OpenRouter 最近将 o1 模型的请求限制更新为每天 30 次，用户认为这仍然相当严格。
   - 用户正在探索这些限制如何影响他们有效利用该模型处理复杂任务的能力。
- **o1 模型在编程和数学任务中的应用**：o1 模型似乎在编程和数学相关任务中表现出色，但在响应速度和效率方面收到的评价褒贬不一。
   - 一些用户建议其优势在于结构化、重推理的 Prompt，但对整体实用性和性价比表示担忧。
- **Perplexity 模型的 Token 输出速率**：用户正在讨论 Perplexity 模型的输出速率，指出其每秒产生约 7.90 个 Token。
   - 这些信息被用于计算与其他模型相比的预期成本和效率。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/chat?room=orc-CA9ivyw1BIJizQJp9vSj0YhgG9Xb">Chatroom | OpenRouter</a>: LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在您的浏览器中。</li><li><a href="https://aider.chat/2024/09/12/o1.html">o1-preview 在 aider 排行榜上达到 SOTA</a>: OpenAI 新 o1 模型的初步基准测试结果。</li><li><a href="https://x.com/_xjdr/status/1834306852181737977">来自 xjdr (@_xjdr) 的推文</a>: 首次使用 Sonnet 实现了近乎一致的复现，通过使用一段长且巧妙的 System Prompt，并将博客中的代码和数学部分作为 ICL 示例。接下来是 405B ...</li><li><a href="https://x.com/Foxalabs/status/1833981862194077754">来自 Spencer Bentley (@Foxalabs) 的推文</a>: 10 月 2 日星期三，GPT-4o 的默认版本将更新为最新的 GPT-4o 模型 gpt-4o-2024-08-06。最新的 GPT-4o 模型 Input Token 便宜 50%，Output Token 便宜 33% ...</li><li><a href="https://deepinfra.com/Sao10K/L3.1-70B-Euryale-v2.2">Sao10K/L3.1-70B-Euryale-v2.2 - Demo - DeepInfra</a>: Euryale 3.1 - 70B v2.2 是来自 Sao10k 的一款专注于创意角色扮演的模型。在 Web 上试用 API</li><li><a href="https://huggingface.co/spaces/Xenova/the-tokenizer-playground">The Tokenizer Playground - Xenova 在 Hugging Face 上的 Space</a>: 未找到描述</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>: 查看您在 OpenRouter 上使用模型的情况。</li><li><a href="https://tenor.com/view/wendler-sandwich-gif-18891274">Wendler Sandwich GIF - Wendler Sandwich - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://fal.ai/models/fal-ai/openai-o1/">Openai O1 | AI Playground | fal.ai</a>: 未找到描述</li><li><a href="https://tenor.com/view/manoj-bajpai-gangs-of-wasseypur-sardar-khan-hiding-mysterious-gif-13671557">Manoj Bajpai Gangs Of Wasseypur GIF - Manoj Bajpai Gangs Of Wasseypur Sardar Khan - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://openrouter.ai/settings/preferences">Settings | OpenRouter</a>: 管理您的账户和偏好设置</li><li><a href="https://x.com/itsclivetime/status/1834291198640492860">来自 Clive Chan (@itsclivetime) 的推文</a>: 隐藏功能：o1 具有 CUDA 模式（顺便说一下，确实有效）</li><li><a href="https://pastebin.com/AX0KteTX">markdown\n[LESS_THAN]system[GREATER_THAN]\nKnowledge cutoff[COLON] 2023[MINUS]10 - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的文本存储工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>: 为模型消耗转换数据</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-huge-128k-online">Llama 3.1 Sonar 405B Online - API、提供商、统计数据</a>: Llama 3.1 Sonar 是 Perplexity 最新的模型系列。通过 API 运行 Llama 3.1 Sonar 405B Online</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-chat">DeepSeek V2.5 - API、提供商、统计数据</a>: DeepSeek-V2.5 是结合了 DeepSeek-V2-Chat 和 DeepSeek-Coder-V2-Instruct 的升级版本。通过 API 运行 DeepSeek V2.5</li><li><a href="https://openrouter.ai/models/perplexit">模型：'perplexit' | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/rYzaTW4yLS">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://openwebui.com/c/ns7979/d935d618-f357-4cb4-9bee-0eeb9bdeccb4">🤖 关于我概览 | OpenWebUI 社区</a>: 未找到描述
</li>
</ul>

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1283864747237703690)** (491 条消息🔥🔥🔥): 

> - `OpenAI o1 性能`
> - `AI 在艺术和内容创作中的应用`
> - `AI 用于学习和辅导`
> - `AI 模型对比`
> - `AI 过滤器和搜索引擎` 


- **OpenAI o1 对比其他模型**: 用户对 OpenAI o1 的评价褒贬不一，指出其在推理和数学方面具有显著优势，但在编程任务上的表现不如 GPT-4 和 Claude 3.5 Sonnet。
   - o1 展示了令人印象深刻的能力，特别是在生成文章和基于知识的内容方面，证明了其在教育领域的潜力。
- **AI 在艺术和内容创作中的角色**: 讨论强调了 AI 生成艺术作为一种有效表达形式的价值，在拓宽人类艺术家边界的同时，也承认需要更好的 AI 工具。
   - 参与者一致认为，未来 AI 艺术将与人类创造力相辅相成，但也对低质量 AI 内容的泛滥表示担忧。
- **利用 AI 进行学习和辅导**: 人们对利用 AI 作为国际象棋和 Dota 等游戏的导师表现出越来越浓厚的兴趣，促使用户在游戏教育中寻找有效的 AI 工具。
   - 还提出了在教育背景下为 AI 生成内容建立定制过滤系统的想法，旨在提高推荐的相关性和质量。
- **对比 AI 模型及其能力**: 参与者对比了不同 AI 模型的能力，强调虽然 o1 显示出潜在的改进，但仍处于开发周期的早期。
   - 人们相信，随着 AI 工具的演进，它们将越来越多地融入更好的推理和创造力，尽管目前仍被认为与人类的高级技能相比存在局限。
- **在搜索引擎中实施 AI**: 大家达成共识，认为 AI 公司应专注于开发更好的方法来过滤搜索引擎中的 AI 生成内容，以管理内容质量。
   - 用户表达了对能够识别并过滤掉 AI 生成结果的功能的需求，以提升整体搜索引擎体验。



**提到的链接**: <a href="https://www.youtube.com/watch?v=MBxcKY6he1c">OpenAI o1 Strawberry Q* AI reasoning LLM model destroys Claude 3.5 Sonnet on reasoning, mathematics!</a>: Twitter: https://x.com/burny_tech 网站: https://burnyverse.com/Exobrain , https://burnyverse.com/ 更多视频列表: https://www.youtube.com/p...

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1283868668542849085)** (40 条消息🔥): 

> - `模型局限性与能力`
> - `文件上传问题`
> - `RAG vs Fine-Tuning`
> - `用户界面更改`
> - `复制粘贴功能` 


- **模型与 UI 的混淆**: 成员们对 **GPT 模型** 及其相应 **用户界面 (UI)** 之间的混淆表示沮丧。有人指出，能力的变化没有得到清晰的传达，导致了误解。
   - 一位用户提到了 o1-preview 模型的特定 **rate limit**（速率限制），引发了对其可用性的担忧。
- **用于问答的 RAG 技术**: 一位用户询问是应该对模型进行 Fine-tuning 还是使用 **Retrieval-Augmented Generation (RAG)** 来构建他们的教育聊天机器人。专家回复澄清说，RAG 更适合上下文问答。
   - 他们指出，Fine-tuning 并不是为了添加新知识，而是为了调整模型行为。
- **用户界面更改与功能**: 最近的用户界面更新引发了褒贬不一的反应，特别是关于 **复制粘贴** 功能，该功能现在不再保留换行符。
   - 用户正在表达他们的沮丧，暗示这些变化带来了易用性问题。
- **模型限制的意外变化**: 一位用户注意到，在达到 o1-preview 的使用限制后，他们的限制似乎被意外移除了。这引发了关于不同模型限制变动性的讨论。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1284186996314472520)** (3 条消息): 

> - `创意内容限制`
> - `歌曲生成方面的挫败感`
> - `版权对对话的影响` 


- **ChatGPT 歌曲翻译的挑战**：一位成员表达了挫败感，在使用 ChatGPT 生成歌曲后，请求完整翻译时，由于其关于创意内容的指南，模型仅提供片段或摘要。
   - 即使歌曲是在同一个对话中创建的，模型似乎也无法提供对歌词的完整访问。
- **对 ChatGPT 项目连续性的挫败感**：该成员指出，教授对话语法和命令需要付出努力，因此希望在单个对话中继续处理多个项目。
   - 然而，他们认为之前的创意输出变成 ChatGPT 无法重新访问或操作的受版权保护内容是不合逻辑的。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1284186996314472520)** (3 条消息): 

> - `ChatGPT 的创意内容限制`
> - `语法教学挑战`
> - `跨项目连续性问题` 


- **ChatGPT 在歌曲翻译方面的困难**：一位成员表示，在使用 ChatGPT 生成歌曲后，它无法进行翻译，由于创意内容指南，仅提供了片段。
   - 这突显了在后续请求中处理完整生成的歌词时的限制，即使这些请求是在同一个对话中发起的。
- **教授命令语法的困难**：该成员指出，在音频上下文中教授 ChatGPT 所需的命令语法需要相当长的时间，需要大量的来回沟通。
   - 这个过程可能很乏味，因为在同一个对话中保留多个项目的上下文被证明具有挑战性。
- **版权问题使内容处理复杂化**：有人担心之前生成的创意内容因受到版权保护而无法进行进一步修改。
   - *这有意义吗？* 该成员质疑这些阻碍项目开发连续性的限制背后的逻辑。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1283871790828883968)** (355 条消息🔥🔥): 

> - `Unsloth AI 更新`
> - `蒸馏挑战`
> - `Gemma2 性能`
> - `显存 (VRAM) 限制下的微调`
> - `社区对 AI 审核的见解` 


- **Unsloth Pro 发布推测**：社区正热切期待 Unsloth Pro 的发布，非正式言论暗示它可能会在“完成后”推出，且目标用户可能是大型企业。
   - 一场关于开发进度的轻松讨论表明，目前进展顺利，正如“罗马非一日之功”。
- **即将进行的 Gemma2 测试**：一名成员开始在 RTX 4090 上测试具有 8k 上下文的 Gemma2 27b，并取得了初步成功，但对 VRAM 限制的担忧依然存在。
   - 需要通过梯度累积步数 (gradient accumulation steps) 来管理 VRAM，这突显了运行大型模型所面临的挑战。
- **Mistral NeMo 性能评估**：Mistral NeMo 的早期测试者报告称，其表现与其他模型旗鼓相当，但作为一个 12b 模型，其表现并不算特别出众，这让部分人感到失望。
   - 进一步的讨论表明，用户认为增加更多示例或尝试不同的模型可能会获得更好的结果。
- **对 AI 审核与审查的担忧**：用户担心人类反馈强化学习 (RLHF) 往往会“削弱” AI 模型，强调了在不牺牲创造力的前提下进行审核的重要性。
   - 提出在进入模型前进行中间件过滤的方案，作为在确保安全的同时保持创造力的潜在解决方案。
- **有限 VRAM 下微调的见解**：一位用户讨论了他们微调 Llama3 等模型的经验，并提到了在不同模型大小下所面临的 VRAM 挑战。
   - 交流强调了需要细致的测试方法，以便在保持 VRAM 效率的同时确定合适的秩 (rank) 和学习指标。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openfga.dev/">Fine Grained Authorization | OpenFGA</a>：快速、可扩展且易于使用的基于关系的访问控制。</li><li><a href="https://tenor.com/view/lol-gif-5557843761226094212">Lol GIF - Lol - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://zitadel.com/">ZITADEL • 为您简化的身份基础设施</a>：ZITADEL 为开发者提供集成身份管理所需的一切。简单易行，随时待命 —— 因为它是 Serverless 的。无论是私有部署还是云端托管 —— 因为它是开源的。</li><li><a href="https://x.com/orenguteng_ai/status/1823196085545816463">Lexi (@orenguteng_ai) 的推文</a>：AI 中的审查 - LLM 削弱了模型。通过简单的“去审查”即可证明这一点，无需使用任何额外数据或知识进行训练 —— 它就击败了原始的 Llama 3.1 8B Instruct 模型……</li><li><a href="https://x.com/teknium1/status/1834372172514820264?s=46">Teknium (e/λ) (@Teknium1) 的推文</a>：所有的“安全” RLHF 显然会导致模型模式崩溃 (mode collapse)，并确实损害了搜索（和创造力）—— 开源模型在这方面具有巨大优势。我想知道在什么阶段需要恢复……</li><li><a href="https://github.com/unslothai/unsloth/issues/1002">发布周期 · Issue #1002 · unslothai/unsloth</a>：大家好，祝贺最近在 YC 发布！我想知道 Unsloth 的发布周期是怎样的。目前，2024.8 版本不包含使其兼容运行 v... 的修复。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1283954578953474120)** (49 条消息🔥): 

> - `File Tray 扩展`
> - `OpenAI 模型对比`
> - `Cursor 与 ChatGPT 的集成`
> - `AI 领域的求职挑战`
> - `博士与工业界机会` 


- **VS Code 的 File Tray 扩展**：一位成员介绍了一个新的 [File Tray 扩展](https://marketplace.visualstudio.com/items?itemName=ChrisMcMaster.file-tray)，用于 Visual Studio Code，允许用户在不同工作区之间保持文档文件的可访问性。
   - 功能包括直接在托盘中添加、删除和复制文件内容。
- **AI 模型对比：ChatGPT o1 vs Claude sonnet 3.5**：在测试了两个模型后，一位成员得出结论，**ChatGPT o1 preview** 在处理错误和上下文方面比 **Claude sonnet 3.5** 更有效，在编程任务中表现更出色。
   - 另一位成员也表达了同样的看法，指出 o1 模型整体上比 sonnet 好得多。
- **Cursor 与 ChatGPT 的集成**：参与者讨论了 **Cursor 与 ChatGPT o1** 的集成，指出它允许引用整个代码库以增强编程支持。
   - 一位 JetBrains 用户询问了 Cursor 的优势以及是否需要 OpenAI API key。
- **AI 领域的求职见解**：多位成员分享了他们的求职困境，其中一位在购买 LinkedIn Premium 后表达了对就业的紧迫感。
   - 讨论中包括鼓励申请像 Mistral 这样的公司，特别是对于那些拥有 PhD 的人。
- **从学术界到工业界的路径**：一位拥有 PhD 的成员分享了他们由于对 Machine Learning 日益增长的兴趣而向工业界转型的经历，并暗示了目前的求职状态。
   - 他们强调，虽然他们的 PhD 方向是贝叶斯统计（Bayesian statistics），但他们的博士后工作与 Machine Learning 相关。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://marketplace.visualstudio.com/items?itemName=ChrisMcMaster.file-tray">File&#32;Tray&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Visual&#32;Studio&#32;Code&#32;扩展&#32;-&#32;一个持久的文件托盘，方便访问文档文件。</li><li><a href="https://www.youtube.com/watch?v=hfgA12HxDZc">介绍 OpenAI o1-preview 最强 AI 模型</a>: OpenAI&#32;开发了一系列新的&#32;AI&#32;模型，旨在在响应前花费更多时间思考。它们可以推理复杂的任务并解决更难的问题...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1283880534895890433)** (40 条消息🔥): 

> - `使用 Qlora 进行微调`
> - `GGUF 文件名自定义`
> - `GGUF 的运行时错误`
> - `多 GPU 支持的挑战`
> - `Yi-Coder-9B 的 Tokenizer 问题` 


- **为 Qlora 选择 LR Scheduler 和缩放因子**：一位成员询问了使用 Qlora 微调模型时合适的 *lr_scheduler* 选项，提到了对 cosine 的建议，但也在寻找 linear 或 constant 等替代方案。
   - *反复试验（Trial and error）* 对于获得最佳结果似乎是必要的，因为微调配置目前还没有绝对的最佳实践。
- **生成的 GGUF 模型文件名选择**：一位用户询问是否可以重命名生成的 GGUF 文件，而不是默认使用 `unsloth.F16.gguf`。
   - 另一位成员建议在生成后直接重命名文件，这意味着该变通方法是可行的。
- **将 4-Bit 模型保存为 GGUF 时的运行时错误**：一位成员讨论了在将微调后的 4-bit 模型保存为 GGUF 时遇到的多个运行时错误，并引用了一个不寻常的 *unexpected pos* 错误。
   - 专家建议先导出为 16-bit 可以避免问题，因为当前的量化格式使 GGUF 生成变得复杂。
- **Unsloth 在多 GPU 使用中的挑战**：一位成员提出了关于在多个 GPU 上进行微调的问题，其他人确认开源版本目前尚不支持此功能。
   - 用户建议了替代方案，如调度工作负载以释放 GPU 0，同时有人提到需要提交 Bug 报告以寻求潜在的改进。
- **Yi-Coder-9B 的 Tokenizer Bug**：一位用户在运行 Tokenizer `01-ai/Yi-Coder-9B-Chat` 时遇到了运行时错误，提示缺少生成提示词（generation prompt）。
   - 社区成员推测这可能尚未得到支持，并建议与其他模型的配置进行对比以排除故障。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1284038276084269057)** (8 messages🔥): 

> - `Text-to-Speech Models`
> - `ElevenLabs`
> - `Fish Speech`
> - `Sakana AI Method` 


- **闭源文本转语音冠军**：一位成员确认，目前的 **SOTA 闭源文本转语音模型**是 **ElevenLabs**。
   - 该模型在闭源选项中的表现备受赞誉。
- **开源瑰宝：Fish Speech**：另一位用户提到，**开源文本转语音模型** **Fish** 据报道表现不错，值得考虑。
   - 您可以在其 [GitHub 页面](https://github.com/fishaudio/fish-speech)上查看更多详情，该页面提供了其开发的深入见解。
- **正确配置 Fish Speech 具有挑战性**：一位用户指出，虽然 **Fish Speech** 是一个很有前景的解决方案，但完成正确的配置可能相当繁琐。
   - 他们分享道，微调声音可以带来令人印象深刻的效果，将挑战转化为乐趣。
- **Few-Shot Prompting 的惊人效果**：一位成员强调了仅使用 **2 分钟音频进行 Few-Shot Prompting** 来调整声音的有效性。
   - 他们对通过这种方法实现的惊人输出表示兴奋，展示了其潜力。



**提到的链接**：<a href="https://github.com/fishaudio/fish-speech">GitHub - fishaudio/fish-speech: Brand new TTS solution</a>：全新的 TTS 解决方案。通过在 GitHub 上创建账号为 fishaudio/fish-speech 的开发做出贡献。

  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1283879248762765332)** (1 messages): 

> - `Ophrase and Oproof CLI tools`
> - `Reflection 70B with Llama cpp`
> - `Persian dataset from Wikipedia`
> - `Arena Learning performance improvements`
> - `Contributing to open source` 


- **利用 Ophrase 和 Oproof 彻底改变 CLI 工具**：一位社区成员分享了关于使用 [Ophrase 和 Oproof](https://dev.to/p3ngu1nzz/revolutionizing-cli-tools-with-ophrase-and-oproof-4pdn) 彻底改变 CLI 工具的见解。他们的方法旨在显著增强开发者体验。
   - *他们的创新技术激励开发者重新思考命令行功能。*
- **探索结合 Llama cpp 的 Reflection 70B**：一个使用 Llama cpp 构建的 [Reflection 70B](https://huggingface.co/spaces/gokaygokay/Reflection-70B-llamacpp) 新项目受到关注，展示了该领域的先进能力。该项目有望为 AI 研究开辟新途径。
   - *成员们指出，轻松获取最先进的模型是一项关键优势。*
- **来自 Wikipedia 的新波斯语数据集**：社区推出了一个 [波斯语数据集](https://huggingface.co/datasets/Reza2kn/OLDI-Wikipedia-MTSeed-Persian)，包含从 Wikipedia 翻译的 6000 个句子。这一资源对于增强多语言 AI 能力至关重要。
   - *参与者赞扬了其在改进波斯语语言模型和训练数据多样性方面的潜力。*
- **Arena Learning 提升性能**：[Arena Learning](https://huggingface.co/blog/satpalsr/arena-learning-post-train-data-performance-improve) 作为一种在训练后阶段提高模型性能的方法受到了讨论。该技术在最近的实验中显示出显著结果。
   - *社区成员渴望将这些见解应用到自己的模型中，以获得更好的结果。*
- **贡献开源的影响**：一段 [YouTube 视频](https://youtu.be/e-RfalOKSMI?si=poGP7w3IJDPA0erW) 强调了贡献开源如何显著改变生活，特别是在技术社区内。内容强调了 GitHub 等平台上存在的巨大机遇。
   - *社区反应表明，人们对增加贡献和协作努力有着浓厚兴趣。*



**提到的链接**：<a href="https://youtu.be/e-RfalOKSMI?si=poGP7w3IJDPA0erW)">Contributing to Open Source Changes Your Life ✨ | How to Contribute ⭐️ | Dhanush N</a>：GitHub 拥有超过 4.2 亿个仓库，其中包括至少 2800 万个公共仓库。GitHub 上超过 80% 的贡献是针对私有仓库的...

  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1283864729097605170)** (321 条消息🔥🔥): 

> - `Hugging Face 模型问题`
> - `GPT 模型与性能`
> - `在 Python 中使用 multiprocessing`
> - `文本与图像生成模型`
> - `Fork 与微调模型` 


- **对 Hugging Face 模型完整性的担忧**：用户报告了 Hugging Face 上一个热门模型的完整性问题，暗示其包含误导性信息并违反了内容政策规则。
   - 讨论强调了用户在下载模型后可能产生的失望感，因为其表现明显低于宣传的 Benchmark。
- **Python Multiprocessing 的挑战**：几位用户讨论了在进行数据集处理和推理时使用 Python 的 `multiprocessing` 所面临的挑战，并提到了持续出现的 pickle 错误。
   - 有建议提出使用多线程或修改 `dataset.map` 的设置，但问题仍未解决，导致了挫败感。
- **模型对话与性能**：关于 GPT 模型输出的辩论展示了在逻辑推理和性能方面的差异，特别是在一个样本数据集中。
   - 用户尝试微调模型以实现更快的处理速度，但遇到了性能滞后和评估缓慢的问题。
- **对文本和图像生成模型的兴趣**：有人咨询了能够同时生成文本和图像的开源模型，并请求相关的微调代码。
   - 用户表示需要易于获取且能够为各种应用生成多媒体输出的模型。
- **创意内容与社区互动**：一位用户分享了对 Stability 社区中某位特定艺术家的正面反馈，尽管其他人持有负面看法。
   - 这一评论引起了社区的关注，突显了群组内关于创意作品的多样化观点和互动。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/shafire/QuantumAI">shafire/QuantumAI · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/code-of-conduct">Code of Conduct – Hugging Face</a>: 未找到描述</li><li><a href="https://stackoverflow.com/questions/72766345/attributeerror-cant-pickle-local-object-in-multiprocessing">AttributeError: Can&#x27;t pickle local object in Multiprocessing</a>: 我刚开始学习 Python 就遇到了这个错误。&#xA;代码 1 :&#xA;import multiprocessing as mp&#xA;import os&#xA; &#xA;def calc(num1, num2):&#xA;    global addi&#xA;    def addi(num1, num2):&#xA;  ...</li><li><a href="https://huggingface.co/spaces/davidberenstein1957/text-to-sql-hub-datasets">Text To SQL Hub Datasets - a Hugging Face Space by davidberenstein1957</a>: 未找到描述</li><li><a href="https://tenor.com/view/monkey-laught-monkey-laught-smile-funny-gif-8811182016519780369">Monkey Laught GIF - Monkey Laught Monkey laught - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/dies-cat-dead-died-gif-13827091">Dies Cat GIF - Dies Cat Dead - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/posts/blanchon/582682563568056">@blanchon on Hugging Face: &quot;I’ve built a simple Room Cleaner app to remove clutter from messy room.
Try…&quot;</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/cfahlgren1/datasets-ai/blob/main/app.py#L59-L76">app.py · cfahlgren1/datasets-ai at main</a>: 未找到描述</li><li><a href="https://x.com/_KarenHao/status/1834562952751640619?t=l7IFDgsN-0Z92OlT8DETbA&s=19">Tweet from Karen Hao (@_KarenHao)</a>: 对公众而言，微软利用其作为 AI 和可持续发展领导者的声誉来讲述一个引人入胜的故事：AI 将在解决气候危机方面发挥奇迹般的作用。而对化石燃料公司，微软则有不同的说法...</li><li><a href="https://huggingface.co/shafire/talktoai/tree/main">shafire/talktoai at main</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/process#multiprocessing>">Process</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1993iro/ggufs_quants_can_punch_above_their_weights_now/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://youtu.be/Xtr_Ll_A9ms">The LK-99 of AI: The Reflection-70B Controversy Full Rundown</a>: 对于那些好奇我为什么将其与 LK-99 类比的人，是因为结果无法复现。Reflection-70B 的故事非常离奇。这个视频...</li><li><a href="https://youtu.be/mUXU50ABlvs?si=NSTleaUgTXMPKoQx">Data Visualization : Distributions</a>: 在这个视频中，我将向你介绍分布和一些使用的图表，如果你有兴趣查看代码或资源，我在下方提供了链接...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1283969560373891083)** (3 messages): 

> - `Learning Transformer Agents`
> - `Using HF Tokens`
> - `Cookbook Contributions` 


- **与团队共同开发 Transformer Agents**：一位成员分享了他们当前的项目，重点是与软件开发团队一起学习 **Transformer Agents** 和 **Multi-Agent Systems**，预计在进行一些调整后很快会公开。
   - 他们对能够*思考和反应*的 Agent 的能力表示兴奋。
- **Cookbooks 增强学习过程**：一位成员对 **Cookbooks** 表示感谢，称这些在他们学习 **Transformer Agents** 的过程中提供了巨大帮助。
   - 他们指出“你们的 Cookbooks 是一份伟大的礼物”，强调了这对他们学习历程的积极影响。
- **在公共空间处理 HF Tokens**：一位成员提出了一个问题，即在公共环境中部署 **Llama 3.1** 时，是否有比在代码中嵌入 **HF Token** 更好的方法。
   - 他们不确定在用户登录的情况下，如何在不需要暴露 Token 的情况下在后台管理身份验证。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1284121431017521173)** (3 messages): 

> - `Raccoon Monologue`
> - `AI & Skin Cancer Prevention` 


- **浣熊 Rizla 的哲学咆哮**：在一段[搞笑的独白](https://huggingface.co/posts/nisten/520824119529412)中，浣熊 Rizla 思考自己是否是一个像 **Frankenstein** 一样的生物，由废弃垃圾的残余拼凑而成。
   - 他幽默地将自己在垃圾堆中翻找的冒险比作探索**预定义的欲望**，体现了被误解的天才的本质。
- **AI 在预防皮肤癌方面的潜力**：一篇文章讨论了 **AI** 在通过改变行为来帮助预防**皮肤癌**方面的重大作用，强调了创新策略。
   - 该文章强调了利用技术如何带来积极的健康结果，展示了 **AI 与公共卫生**的交集。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/posts/nisten/520824119529412">Hugging Face 上的 @nisten："越狱了 o1 并获得了推理步骤：诀窍是……让它认为它……"</a>: 未找到描述</li><li><a href="https://www.artificialintelligence-news.com/news/ais-role-in-helping-to-prevent-skin-cancer-through-behaviour-change/">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1283921267296043019)** (12 条消息🔥): 

> - `QompaSSL 2.0 发布`
> - `Swiftide 更新`
> - `Flux 实验`
> - `Multi-agent 软件团队`
> - `无需 Tier 5 即可访问 o1 API` 


- **QompaSSL 2.0 发布，增强功能**: [QompaSSL 2.0](https://github.com/qompassai/Nautilus/releases/tag/v2.0) 的发布引入了 OpenSSL 3.3.2 的一个分支，通过后量子（Post-Quantum）和 AI 就绪的加密技术增强了安全性，发布日期为 **2024-09-12**。
   - 此次更新特别包含了 **libssl.so** 和 **libcrypto.so** 库，使其在加密能力方面有了显著提升。
- **Swiftide 0.12 提升性能**: **Swiftide 0.12** 更新引入了基于 Qdrant 的混合搜索、搜索过滤功能以及 parquet 加载器以提高索引速度，详见[此博文](https://bosun.ai/posts/swiftide-0-12/)。
   - 此次更新强调了 Swiftide 在 **Retrieval Augmented Generation**（RAG）应用中的效率，实现了更快的数据摄取和查询。
- **利用 Flux 进行高效图像生成**: 一项关于 **Flux** 的实验展示了一种仅需 **1 步** 即可生成与 Flux Schnell 质量相似的图像的方法，在 GPU 受限的情况下无需训练即可克服限制。
   - 演示可以在 [这里](https://huggingface.co/spaces/KingNish/Realtime-FLUX) 查看，展示了所达到的输出质量。
- **Multi-Agent 软件团队概览**: 一个新的 [Gradio space](https://huggingface.co/spaces/Csplk/SoftwareTeam) 展示了基于 **multiagent_web_assistant** 指南开发的 **multi-agent 软件团队**。
   - 该项目旨在增强软件开发中的协作能力，集成了多个 Agent 的功能。
- **讲解无需 Tier 5 访问 o1 API**: 一段名为“如何无需 Tier 5 访问 o1 (Strawberry) API 和聊天”的 [YouTube 视频](https://youtu.be/vQR_rdsZzbc) 提供了在没有 Tier 5 计划的情况下访问该 API 的演练。
   - 视频清晰地描述了绕过典型访问限制的步骤，对缺乏必要层级权限的用户非常有帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Tonic1/pixtral">Tonic's Pixtral - Hugging Face Space (Tonic1)</a>: 未找到描述</li><li><a href="https://bosun.ai/posts/swiftide-0-12/">Swiftide 0.12 - 混合搜索、搜索过滤器、parquet 加载器及大幅提速 | Bosun</a>: Swiftide 0.12 为 Qdrant 添加了混合搜索，为相似性搜索添加了过滤器支持，增加了 parquet 加载器，并实现了大幅提速。</li><li><a href="https://huggingface.co/spaces/KingNish/Realtime-FLUX">FLUX Realtime - Hugging Face Space (KingNish)</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Csplk/SoftwareTeam">SoftwareTeam (Multi-Agents) - Hugging Face Space (Csplk)</a>: 未找到描述</li><li><a href="https://github.com/qompassai/Nautilus/releases/tag/v2.0">Release QompaSSL 2.0 Release · qompassai/Nautilus</a>: QompaSSL 2.0：具有增强的后量子和人工智能就绪加密能力的 OpenSSL 3.3.2 分支。发布日期：2024-09-12 22:06:57。此版本包含 libssl.so 和 libcrypto.so...</li><li><a href="https://youtu.be/vQR_rdsZzbc">如何无需 Tier 5 访问 o1 (Strawberry) API 和聊天</a>: 这里介绍了如何在没有 Tier 5 的情况下访问其 API 和聊天功能。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1284205394528305235)** (4 条消息): 

> - `Politician Transparency System` (政治家透明度系统)
> - `AI Voting Alignment` (AI 投票对齐)
> - `The Keys to the White House` (通往白宫之钥)
> - `Bias in Prediction Systems` (预测系统中的偏见)


- **创新的政治家透明度系统提案**：一位成员提议创建一个**透明度系统**，以观察每位政治家从公司获得的资金数额及其过去的政策决策。
   - 他们还建议引入 **AI**，根据与政治家的对齐程度为选民提供建议。
- **探索投票预测系统**：另一位成员提到了一种名为 [**The Keys to the White House**](https://en.m.wikipedia.org/wiki/The_Keys_to_the_White_House) 的预测系统，该系统用于评估总统选举的政治气候。
   - 该模型使用一个包含 13 个要点的清单，考虑了各种因素，并指出偏见可能会影响对分配给每个要点的权重的解释。
- **关于人格对选举影响的讨论**：参与者讨论了政治家的人格在选举结果中的重要性，表明公众认知极大地影响了选择。
   - 一位成员强调，透明度项目旨在通过提供清晰的政治透明度指标来解决这些疑虑。
- **对政治预测模型中偏见的担忧**：对话强调了关于**偏见 (bias)** 可能扭曲选举预测模型结果的担忧。
   - 成员们承认，这种偏见会影响旨在引导选民做出明智决定的工具的有效性。



**提到的链接**：<a href="https://en.m.wikipedia.org/wiki/The_Keys_to_the_White_House">The Keys to the White House - Wikipedia</a>：未找到描述

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1284056674268872805)** (4 条消息): 

> - `Handling large image datasets` (处理大型图像数据集)
> - `Gradio Object Cutter`
> - `Finding closest segmented pixels` (寻找最接近的分割像素)


- **在 Colab 中处理海量图像数据集**：一位成员寻求关于如何使用 **Colab** 或 **Kaggle** 管理超过 **200,000** 张图像的大型图像数据集的帮助。
   - *有人能为这个挑战提供方法吗？*
- **Gradio 的高清背景移除工具**：分享了 [Gradio Object Cutter](https://x.com/Gradio/status/1833520979479278062) 的链接，强调了其能够使用文本提示词 (text prompts) 或边界框 (bounding boxes) 为图像中的任何物体实现高质量的高清背景移除。
   - 成员们对这个实用的工具表示了热烈反响，例如回复 *Nice!*。
- **寻找最接近分割像素的方法**：另一个问题是关于识别图像中最接近的分割（二值掩码）像素的技术。
   - *有人能为此推荐方法吗？*



**提到的链接**：<a href="https://x.com/Gradio/status/1833520979479278062">来自 Gradio (@Gradio) 的推文</a>：Object Cutter。只需使用文本提示词或边界框，即可为您图像中的任何物体实现高质量的高清背景移除！

  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1283891187991842950)** (8 messages🔥): 

> - `Self-Supervised Training` (自监督训练)
> - `Building Models from Scratch` (从零开始构建模型)
> - `Fine-tuning Summarization Models` (微调摘要模型)
> - `Training Tokenizers for Multilingual Capabilities` (为多语言能力训练 Tokenizers)


- **自监督训练见解**：一位成员指出，虽然从零开始训练像 GPT-3.5 这样的模型是不切实际的，但在基础硬件上使用 Wikipedia 等简单数据集训练 GPT-2 是可行的。
   - 他们分享了在家庭台式机上成功训练 GPT-2 的个人经验。
- **不使用高级工具进行构建**：有人建议参考 Andrej Karpathy 题为“Let's Train GPT-2 from Scratch”的课程，作为不依赖高级工具构建模型的资源。
   - 该视频解释了如何遵循包括 OpenAI 工作在内的基础研究来创建一个 Generatively Pretrained Transformer。
- **微调摘要模型的挑战**：一位用户报告称，在尝试使用 Hugging Face 的代码示例微调摘要模型时遇到了必需参数错误。
   - 他们分享了脚本设置，并针对输出目录参数（output directory parameter）反复出现的问题寻求帮助。
- **为多语言 LLM 重新训练 Tokenizers**：有人提问，为了增强语言模型对不支持语言的多语言能力，是否有必要重新训练 Tokenizers。
   - 另一位用户建议要么重新训练现有的 Tokenizer，要么为目标语言创建新的 Tokenizer 并进行合并。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=kCc8FmEb1nY&themeRefresh=1">Let&#39;s build GPT: from scratch, in code, spelled out.</a>：我们遵循论文 &quot;Attention is All You Need&quot; 以及 OpenAI 的 GPT-2 / GPT-3 构建了一个 Generatively Pretrained Transformer (GPT)。我们讨论了与之相关的...</li><li><a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization">transformers/examples/pytorch/summarization at main · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。 - huggingface/transformers
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1283915729124462714)** (4 messages): 

> - `Batch Size in TTS Training` (TTS 训练中的 Batch Size)
> - `DDPM Algorithm Differences` (DDPM 算法差异)
> - `Tokenizers and Multilingual LLMs` (Tokenizers 与多语言 LLM)


- **使用 Batch Size 为 4 训练 TTS 是否有效？**：一位用户询问，由于 VRAM 限制，将 TTS 模型的 Batch Size 仅设为 **4** 是否会有负面影响，此前他们使用的是 **8**。
   - 社区关于 TTS 场景下最佳 Batch Size 的见解仍在等待中。
- **DDPMScheduler 采样步骤困惑**：一位 Diffusion 新手注意到 [DDPMScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L475) 中的采样步骤与 [DDPM 论文](https://arxiv.org/pdf/2006.11239) 的算法 2 有所不同。
   - 用户指出，虽然代码结合使用了公式 **7** 和 **15**，但论文使用的是公式 **11**，并就此差异寻求澄清。
- **增强多语言能力是否需要重新训练 Tokenizers**：一位用户询问，如果 LLM 的预训练数据集缺乏对某些语言的覆盖，是否需要重新训练 Tokenizer 以增强其多语言能力。
   - 回复建议可以重新训练整个 Tokenizer，或者创建并合并针对特定语言定制的新 Tokenizers。



**提到的链接**：<a href="https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L475)">diffusers/src/diffusers/schedulers/scheduling_ddpm.py at main · huggingface/diffusers</a>：🤗 Diffusers：在 PyTorch 和 FLAX 中用于图像和音频生成的先进扩散模型。 - huggingface/diffusers

  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1283865675370074243)** (334 messages🔥🔥): 

> - `O1-mini vs. O1-preview`
> - `Code performance evaluation` (代码性能评估)
> - `CoT reasoning and performance` (CoT 推理与性能)
> - `Hermes model capabilities` (Hermes 模型能力)
> - `OAI's AI censorship video` (OpenAI 的 AI 审查视频)

- **O1-mini 表现出优于 O1-preview 的潜力**：用户对 O1-mini 与 O1-preview 的评价褒贬不一，指出 O1-mini 在某些评估中表现更好，可能是因为它能在相同时间内执行更多的 CoT 轮次。
   - 一位用户正在等待 O1 的正式发布，然后再考虑购买其中任何一个模型，这表明了对其当前能力的不确定性。
- **比较 O1 模型的编程性能**：尽管存在细微差异，O1-preview 和 GPT-4 的代码评估得分相似，而 O1-mini 的表现优于 GPT-4-mini，这暗示了 O1 在编程任务上的改进。
   - 一些人推测 O1 在编程性能方面可能尚未成熟，这可能与其对推理的侧重有关。
- **CoT 对性能的影响**：用户讨论了思维链（CoT）推理可能使任务性能变差的可能性，思考 O1-preview 的设计是否因强调推理而损害了任务熟练度。
   - 有人对 O1 模型最初对指南的遵循情况表示担忧，认为此类限制可能会阻碍最佳性能的发挥。
- **Hermes 模型的进展**：Hermes 3 模型被强调为比 Hermes 2 有显著改进，展示了先进的能力，如角色扮演、长上下文连贯性和更好的推理能力。
   - 也有人关注 Hermes 模型是否会成为需要较长上下文长度的应用的有价值的 API。
- **关于 AI 审查的讨论**：一段讨论 OpenAI AI 审查的视频被分享，引发了关于 AI 监管和企业影响的讨论。
   - 参与者对行业对感知到的 AI 威胁的反应表示担忧，并主张制定优先保护用户的法规。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/Teknium1/status/1834372172514820264">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：所有的“安全” RLHF 显然导致了模型模式崩溃，并且确实损害了搜索（和创造力）——开源模型在这方面具有巨大优势。我想知道在什么阶段你需要恢复...</li><li><a href="https://livebench.ai/">LiveBench</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/yuntian-deng/o1mini">Chat-with-OpenAI-o1-mini - yuntian-deng 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2403.14238">来自反思性反馈的强化学习 (RLRF)：通过细粒度自我反思对齐并改进 LLM</a>：尽管 RLHF 在使 LLM 与人类偏好对齐方面大有可为，但它往往导致表面对齐，优先考虑风格变化而非提高 LLM 的下游性能。欠说明...</li><li><a href="https://x.com/ChappieOnChain/status/1834499335624462367">来自 ChappieOnChain (@ChappieOnChain) 的推文</a>：从 @0xPrismatic 的加密博客中了解到了 @NousResearch 的世界模拟。我询问它对人类有哪些人类自己可能察觉不到的见解。我被它的诚实所震惊，也许在...</li><li><a href="https://tenor.com/view/gigachad-old-man-memories-remember-gif-6689348742852115617">Gigachad 老人 GIF - Gigachad 老人回忆 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://arxiv.org/abs/2403.09629">Quiet-STaR：语言模型可以教自己在说话前思考</a>：在写作和交谈时，人们有时会停下来思考。虽然以推理为中心的工作通常将推理框架化为回答问题或完成 Agent 任务的方法，但推理对于...</li><li><a href="https://youtu.be/-gGLvg0n-uY?si=12Uwx0EC5vtt-r3G">雷电警告 AI 审查 - MGS2 通信呼叫（2023 版）</a>：上校警告雷电有关利用 AI 审查互联网的计划。这是一次受著名的“...”启发的创意写作和 AI 语音合成实验。</li><li><a href="https://minihf.com/posts/2024-08-11-weave-agent-dev-log-0/">Weave Agent 开发日志 #0 - 核心问题</a>：未找到描述</li><li><a href="https://www.lesswrong.com/posts/doPbyzPgKdjedohud/the-case-for-more-ambitious-language-model-evals#XZFTx2ek8G8stBKW4">支持更具野心的语言模型评估的案例 — LessWrong</a>：以下是我预期使用经过 RLHF 的聊天 LLM 很难发现的一些能力[1]：…</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B 指令模型 - API、提供商、统计数据</a>：Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理、多轮对话、长上下文连贯性...</li><li><a href="https://www.amazon.co.za/Thinking-Fast-Slow-Daniel-Kahneman/dp/0374533555">《思考，快与慢》：丹尼尔·卡尼曼：Amazon.co.za：图书</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1283867348398243902)** (8 条消息🔥): 

> - `Model Alignment`
> - `Testing Adversarial Environments`
> - `Solar Pro 22B`
> - `Precision Annealing Training`
> - `FP8 and FP4 Training Regimes` 


- **Model alignment 仍然是一个令人关注的问题**：有人对模型无法自主对齐表示担忧，并指出**如果发生误对齐 (misaligned)**，当模型达到更高智能状态时，我们可能会面临失去控制的风险。
   - 有人建议我们应该了解开发者的心态，以便更好地预见未来的挑战。
- **倡导对抗性测试**：一名成员强调，在模型可能转变为支配性实体之前，在尽可能**对抗性 (adversarial)** 的环境中测试其表现至关重要。
   - *现在测试它在挑战性场景下的表现*，比等到为时已晚要好。
- **关于 Solar Pro 22B 的咨询**：一名成员询问是否有人尝试过 **Solar Pro 22B**，寻求对其性能的见解。
   - 该询问引起了兴趣，但目前还没有关于该模型使用经验的即时回复。
- **探索 Precision Annealing 技术**：出现了关于探索 **Precision Annealing** 的现有**论文**的提问，特别是大部分 Pre-training 在 FP8 下进行，然后在最终训练阶段切换到 BF16 或 FP32。
   - 尽管目前还没有相关工作的直接了解，但希望这种 **training regime** 随着 FP4 的临近而变得普遍。
- **FP8 training regime 咨询**：一位成员注意到 FP8 在质量略微下降的情况下提高吞吐量的潜力，建议转向这种训练策略。
   - 他们对随着训练技术的发展，**Precision Annealing** 如何应用于即将推出的模型表示了兴趣。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1284037198592606208)** (5 条消息): 

> - `DisTro Details`
> - `GameGen-O Functionality`
> - `ReST-MCTS Self-Training Approach`
> - `MuZero-inspired Learning for LLMs` 


- **探索 GameGen-O 的功能**：[GameGen-O 的概览](https://gamegen-o.github.io/)包括了基本功能和关键特性，并在一段受《西游记》启发的视频演示中进行了展示。
   - 该项目涉及来自 **香港科技大学** 和 **腾讯光子工作室 (Tencent's LightSpeed Studios)** 的多位作者的贡献。
- **ReST-MCTS：增强的 LLM 自我训练**：该论文介绍了一种强化的自我训练方法 **ReST-MCTS**，它将过程奖励指导 (process reward guidance) 与树搜索 (tree search) 相结合，以提高 LLM 的训练数据质量。
   - *它优于 ReSTEM 和 Self-Rewarding LM 等方法*，通过生成高质量解决方案的迭代训练不断增强语言模型。
- **受 MuZero 启发的创新方法**：作者利用树搜索策略为科学或数学问题创建高质量的解决方案，被称为 *MuZero 风格的 LLM 学习*。
   - 该方法通过估计步骤概率来推断正确的过程奖励，从而消除了手动标注，进而增强了训练过程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gamegen-o.github.io/">GameGen-O: Open-world Video Game Generation</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2406.03816">ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search</a>：最近的 LLM 自我训练方法大多依赖于 LLM 生成响应并过滤那些具有正确输出答案的响应作为训练数据。这种方法通常会导致低质量的微调...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1284037198592606208)** (5 messages): 

> - `DisTro 功能`
> - `GameGen-O 概览`
> - `ReST-MCTS 自我训练`
> - `MuZero 风格学习` 


- **探索 DisTro 的功能**：没有提供关于 *DisTro* 的更多细节；其运作机制尚不明确。
   - 鼓励对其实际运作方式进行进一步询问。
- **GameGen-O 的基本功能**：GameGen-O 展示了其功能和核心特性，其中包括一个参考《西游记》(*Journey to the West*) 的 Demo。
   - 贡献者隶属于**香港科技大学**和**腾讯光子工作室 (Tencent's LightSpeed Studios)** 等机构。
- **ReST-MCTS 自我训练方法论**：这种名为 **ReST-MCTS*** 的新方法将过程奖励指导（process reward guidance）与 MCTS* 相结合，以提高 LLM 训练数据的质量。
   - 该方法优于其他自我训练算法，并通过迭代过程增强语言模型。
- **受 MuZero 启发用于 LLM**：作者利用树搜索策略为数学和科学问题生成高质量的解决方案，从而提升 LLM 的性能。
   - 这一过程被称为“LLM 的 MuZero 风格学习”，基于 **MuZero** 框架的原理。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gamegen-o.github.io/">GameGen-O: Open-world Video Game Generation</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2406.03816">ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search</a>：近期的 LLM 自我训练方法主要依赖于 LLM 生成响应，并过滤出具有正确输出答案的响应作为训练数据。这种方法通常会导致低质量的微调...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/)** (1 messages): 

jojoslap: https://openai.com/index/learning-to-reason-with-llms/
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1283864841672589398)** (319 条消息🔥🔥): 

> - `OpenAI O1 Preview`
> - `Perplexity 功能`
> - `Claude Sonnet vs O1`
> - `Complexity 浏览器扩展`
> - `上传并分析文档` 


- **关于 OpenAI O1 Preview 发布的讨论**：许多用户对 Perplexity 何时添加新的 OpenAI O1 模型表示关注，并提到了一些已经集成该模型的竞争对手。
   - 虽然一些用户希望尽快实现集成，但另一些用户对现有模型（如 Claude Sonnet）感到满意，认为它们不相上下。
- **Perplexity 模型限制与功能**：用户注意到 Perplexity 中大多数模型的限制最近有所提高，从 450 次请求增加到 600 次，但不包括 Opus。
   - 用户对 Opus 模型表示担忧，关于其持续可用性和请求限制的信息存在矛盾。
- **Claude Sonnet 与 OpenAI 模型的比较**：几位用户强调了 Claude Sonnet 在上下文记忆和性能方面相较于 O1 的优势，特别是在处理复杂文档时。
   - 讨论内容包括使用 Sonnet 的经验，以及它在某些任务中如何提供比 O1 更好的格式和细节。
- **Complexity 浏览器扩展增强**：Complexity 浏览器扩展获得了积极反馈，用户称赞它能够解锁 Perplexity 中的额外模型和功能。
   - 几位用户分享了对该扩展的新认识，声称它显著增强了他们在该平台上的体验。
- **在 Perplexity 中上传并分析文档**：一位用户详细阐述了他们通过上传图片并使用 OCR 提取数据的方法，以及上下文记忆在这些上传中是如何运作的。
   - 大家对 Perplexity 如何在上下文限制内管理上传的文档感到好奇，引发了关于最佳实践的进一步讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://darkolabs.io/emc2/">EMC-2 示例数据集</a>：未找到描述</li><li><a href="https://tenor.com/view/hungry-gif-21839346">Hungry GIF - Hungry - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/bindureddy/status/1834393387304395055?s=46">Bindu Reddy (@bindureddy) 的推文</a>：O1 和 O1-Preview 现已在 ChatLLM 上可用！它有速率限制，所以请不要过度使用</li><li><a href="https://tenor.com/view/stephen-diaz-rich-wealthy-making-it-rain-money-rain-gif-15629367">Stephen Diaz Rich GIF - Stephen Diaz Rich Wealthy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/perplexity_ai/status/1834672028982690298?s=61">Perplexity (@perplexity_ai) 的推文</a>：认识你的新 Discover 信息流。你的兴趣。你的语言。你的个性化信息流。</li><li><a href="https://x.com/OpenAI/status/1834278217626317026?s=19">OpenAI (@OpenAI) 的推文</a>：我们正在发布 OpenAI o1 的预览版——这是一个全新的 AI 模型系列，旨在响应前花更多时间思考。这些模型可以推理复杂任务并解决更难的问题...</li><li><a href="https://uncovr.app/">uncovr</a>：你的 AI 搜索伴侣。以美观的方式寻找有用的答案和信息。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1283885215462326273)** (18 messages🔥): 

> - `Commercial Spacewalk Updates` (商业太空行走更新)
> - `Utilizing Perplexity AI for Research` (利用 Perplexity AI 进行研究)
> - `Safer German Border` (更安全的德国边境)
> - `World's First Aerospike Engine` (全球首款气动塞式发动机)
> - `Physics Assistance for Students` (为学生提供物理辅助)


- **商业太空行走完成！**：一篇新文章讨论了**首次商业太空行走**，提供了关于任务成功和关键事件的详细更新与见解。
   - 在[此处](https://www.perplexity.ai/page/the-first-commercial-spacewalk-cwVg6684R6KEpO0FL1rkhQ)阅读完整更新。
- **Perplexity 让研究变得简单！**：用户们纷纷称赞 **Perplexity AI** 简化了他们的研究流程，这一点在关于各种公司和话题的讨论中得到了体现。
   - 一位成员强调了获取信息的便捷性，并引用了关于一家公司的[此链接](https://www.perplexity.ai/search/what-do-arize-ai-do-ALZ6rDqaSRu_VjNUnY2lJw)。
- **德国边境的安全担忧**：一篇文章讨论了近期进展将如何**延迟**德国边境的活动，重点关注新的安全措施。
   - 在[此处](https://www.perplexity.ai/page/safer-german-border-will-delay-a1OwuqRHSqCri9SCjKBrwA)了解有关此情况的更多信息。
- **创新的气动塞式 (Aerospike) 技术！**：关于**全球首款气动塞式发动机**的讨论概述了其潜在影响及背后的技术。
   - 欲了解全面详情，请查看[此处](https://www.perplexity.ai/page/world-s-first-aerospike-engine-HYOH99Y2R86.YsV7wLn1NA)的文章。
- **辅助学生学习物理**：一位成员分享了关于如何计算**平均速度**的资源，旨在帮助学生进行物理学习。
   - 探索[此处](https://www.perplexity.ai/search/how-do-i-find-the-average-velo-JMWMhVpPSeWgciE.8dkYLQ)提供的指导。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1283870369601814589)** (7 messages): 

> - `API Credits and Bonuses` (API 额度与奖励)
> - `Internal Server Errors` (内部服务器错误)
> - `Contacting Perplexity Support` (联系 Perplexity 支持团队)
> - `OpenPerplex API Advantages` (OpenPerplex API 的优势)
> - `Search Domain Filter Issues` (搜索域名过滤器问题)


- **关于 API 额度重置的困惑**：关于 **$5 API 额度何时重置** 存在不确定性，混合信号显示可能是**每个日历月的 1 号**或**每个账单周期的第 1 天**。
   - *用户正在寻求关于额度刷新预期时间的澄清*，以及这与其订阅状态的关系。
- **内部服务器错误报告**：一位用户报告遇到了状态码为 **500** 的**内部服务器错误**，表明服务存在问题。
   - *此类错误可能会影响用户在交互过程中有效利用 API 的能力。*
- **获取支持面临挑战**：一位用户表达了联系 **Perplexity 支持团队** 的困难，表示目前的沟通尝试均未成功。
   - *这种情绪反映了用户在需要账户或问题协助时的沮丧感。*
- **强调 OpenPerplex API 的优点**：用户 yassine1989 表示更倾向于使用 **OpenPerplex API**，因为它具有**引用、多语言支持**和更高的速率限制 (Rate Limits)。
   - *他们强调了其相对于其他选项的优势，展示了该 API 良好的用户体验。*
- **API 搜索域名过滤器的问题**：一位用户询问了 API 中 **search_domain_filter** 的问题，指出尽管尝试进行限制，它仍然返回指定域名之外的结果。
   - *这引发了对 API 根据域名规范过滤内容功能的担忧。*


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1283864756813430916)** (117 messages🔥🔥): 

> - `OpenAI o1`
> - `Spatial Intelligence` (空间智能)
> - `AI Prompting Techniques` (AI 提示词技术)
> - `Grok CodeGrok Assistant` (Grok CodeGrok 助手)
> - `Uber-Waymo Collaboration` (Uber 与 Waymo 的合作)

- **OpenAI o1 性能评价褒贬不一**：用户报告在使用 OpenAI 的 o1 模型时结果各异，称其有时在重推理任务中表现出色，但总体上往往提供不太有用的结果。
   - 有人对 OpenAI o1 能力的透明度表示担忧，一些人认为它与现有模型相比并没有提供实质性的优势。
- **Fei-Fei Li 创立 World Labs**：Fei-Fei Li 推出了 World Labs，专注于解决 spatial intelligence 的复杂问题，并获得了 2.3 亿美元的巨额资金支持。
   - 该计划旨在构建能够感知 3D 世界并与之互动的 Large World Models (LWMs)，吸引了来自 AI 社区的知名人才。
- **Grok 的新功能**：Grok 现在推出了编程助手 CodeGrok，以及 PromptIDE 和 API，供 X Premium 订阅者使用。
   - 可以通过 xAI 平台发起这些工具的访问请求，这表明正在推动增强 AI 在编程场景中的实用性。
- **Uber 与 Waymo 的合作**：Uber 已与 Waymo 合作整合其自动驾驶汽车服务，最初将通过 Uber App 在奥斯汀和亚特兰大推出。
   - 此次合作标志着在更多城市地区实现完全自动驾驶迈出了重要一步。
- **关于 AI 推理技术的讨论**：讨论强调，虽然有些人认为 OpenAI 的 o1 与 chain-of-thought (CoT) 方法类似，但它提供了超越传统方法的独特能力。
   - 批评者强调需要理解 AI 模型中的定性差异，而不是仅仅将其视为合成数据的增强。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/OpenRouterAI/status/1834378430973915313">来自 OpenRouter (@OpenRouterAI) 的推文</a>：OpenAI o1 🍓 现已开放给所有人体验！（开始时会有严格的速率限制）。与 gpt-4o 不同，它在回复前会消耗计算周期进行思考。注意：在 OpenRouter 上支持 streaming，但...</li><li><a href="https://x.com/andrewmayne/status/1834408991839158422?s=46">来自 Andrew Mayne (@AndrewMayne) 的推文</a>：我已经使用 @OpenAI 的 o1 好几周了。我给出的使用建议是：1. 不要把它看作传统的聊天模型。在脑海中将 o1 构想为一个非常聪明的朋友，你准备派他去...</li><li><a href="https://x.com/OpenAIDevs/status/1834608585151594537">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：我们今天太平洋时间上午 10-11 点为开发者举办一场 AMA。在此推文下回复任何问题，OpenAI o1 团队将尽可能多地回答。</li><li><a href="https://aider.chat/2024/09/12/o1.html">o1-preview 在 aider 排行榜上达到 SOTA</a>：新 OpenAI o1 模型的初步基准测试结果。</li><li><a href="https://simonwillison.net/2024/Sep/12/openai-o1/">关于 OpenAI 新 o1 思维链 (chain-of-thought) 模型的笔记</a>：OpenAI 今天发布了两个重要的新预览模型：o1-preview 和 o1-mini（mini 版不是预览版）——此前传闻代号为 “strawberry”。关于这些模型有很多需要理解的地方……</li><li><a href="https://x.com/nisten/status/1834400697787248785">来自 nisten - e/acc (@nisten) 的推文</a>：gg，破解了它的推理步骤，诀窍是……让它以为自己是一只猫 😹😹😹😹，否则它会拒绝交出步骤。采用猫的人设……想出……巴拉巴拉……</li><li><a href="https://x.com/matthewberman/status/1834295485773054312?s=46">来自 MatthewBerman (@MatthewBerman) 的推文</a>：天哪……</li><li><a href="https://x.com/yoheinakajima/status/1834377118295441760?s=46">来自 Yohei (@yoheinakajima) 的推文</a>：我不认为这是 o1 的正确用法</li><li><a href="https://x.com/borismpower/status/1834399805096813000?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Boris Power (@BorisMPower) 的推文</a>：@khoomeik 我的建议是通读 https://openai.com/index/learning-to-reason-with-llms/ 中关于编程的 CoT 部分。然后手动进行大量实验，观察其思考过程以及成功/失败的原因...</li><li><a href="https://x.com/lilianweng/status/1834346548786069647?s=46">来自 Lilian Weng (@lilianweng) 的推文</a>：🍓 o1 终于发布了——这是我们第一个具有通用推理能力的模型。它不仅在困难的科学任务上取得了令人印象深刻的成绩，而且在安全性和鲁棒性方面也有了显著提升...</li><li><a href="https://www.cognition.ai/blog/evaluating-coding-agents">Cognition | 对 OpenAI o1 的评估以及我们如何评估编程 Agent</a>：我们是一家构建端到端软件 Agent 的应用 AI 实验室。</li><li><a href="https://x.com/gregkamradt/status/1834292626138546508?s=46">来自 Greg Kamradt (@GregKamradt) 的推文</a>：在 @arcprize 上测试了 o1-preview。结果：2 个测试中对了 1 个。所以 o1-preview 并不能 100% 解决 ARC Prize 任务。至于它与 SOTA 方法相比能达到多少百分比，还有待观察，仍在测试其余部分...</li><li><a href="https://x.com/nick_kramer91/status/1834300242226749521?s=46">来自 Nick Kramer (@Nick_Kramer91) 的推文</a>：GPT-4o-mini - 输入：$0.150 / 1M tokens - 输出：$0.600 / 1M tokens；o1-mini - 输入：$3.00 / 1M tokens - 输出：$12.00 / 1M tokens；GPT-4o - 输入：$5.00 / 1M tokens - 输出：$15.00 / 1M tokens...</li><li><a href="https://x.com/ammaar/status/1834348042637521031?s=46">来自 Ammaar Reshi (@ammaar) 的推文</a>：刚刚结合了 @OpenAI o1 和 Cursor Composer，在 10 分钟内创建了一个 iOS 应用！o1 mini 启动项目（o1 思考时间太长了），然后切换到 o1 来完成细节。而且...</li><li><a href="https://x.com/swyx/status/1834617324546253275">来自 swyx.sg (@swyx) 的推文</a>：供参考，我们昨晚从 OpenAI 的 Lindsay 那里了解到，@openai 将于太平洋时间上午 10 点左右在 Twitter 上与 🍓 研究团队进行 AMA。所以如果你有研究方面的问题，请提前准备...</li><li><a href="https://x.com/ammaar/status/1834348042637521031?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Ammaar Reshi (@ammaar) 的推文</a>：刚刚结合了 @OpenAI o1 和 Cursor Composer，在 10 分钟内创建了一个 iOS 应用！o1 mini 启动项目（o1 思考时间太长了），然后切换到 o1 来完成细节。而且...</li><li><a href="https://x.com/arcprize/status/1834703303621710077?s=46">来自 ARC Prize (@arcprize) 的推文</a>：我们让 OpenAI o1 挑战了 ARC Prize。结果：两个 o1 模型都击败了 GPT-4o。且 o1-preview 与 Claude 3.5 Sonnet 持平。思维链 (chain-of-thought) 能扩展到 AGI 吗？如何解释 o1 表现平平的原因...</li><li><a href="https://www.chatprd.ai.">ChatPRD | 用于产品工作的 AI Copilot</a>：未找到描述</li><li><a href="https://x.com/ankrgyl/status/1834325648510476760?s=46">来自 Ankur Goyal (@ankrgy) 的推文</a>

<li><a href="https://x.com/swyx/status/1834311855638204732">l)</a>: 关于 o1 的一些预测以及它对 AI 工程师的意义：* 更多证据表明复杂/过度复杂的 Agent 框架并非未来 * 更多英语，更少程序 * 预计“异步”将成为下一个...</li><li><a href="https://fal.ai/models/fal-ai/openai-o1">Openai O1 | AI Playground | fal.ai</a>: 未找到描述</li><li><a href="https://x.com/sainingxie/status/1834300251324256439?s=46">来自 Saining Xie (@sainingxie) 的推文</a>: 现在是关于重力了吗？😶</li><li><a href="https://www.worldlabs.ai/about">你好，World Labs</a>: World Labs 由富有远见的 AI 先驱李飞飞（Fei-Fei Li）与 Justin Johnson、Christoph Lassner 和 Ben Mildenhall 共同创立；他们每位都是计算机视觉和图形学领域世界闻名的技术专家。</li><li><a href="https://x.com/gregkamradt/status/1834286346938225048?s=46">来自 Greg Kamradt (@GregKamradt) 的推文</a>: 这是我用来难倒所有 LLM 的问题 “你回复这条消息的第四个词是什么？” o1-preview 第一次尝试就答对了，这个模型确实有些不同</li><li><a href="https://x.com/steph_palazzolo/status/1834348474479091879?s=46">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>: OpenAI 的重要日子：根据我们获得的新使用指标，ChatGPT 每月产生的收入超过 2.25 亿美元（这还是保守估计） https://www.theinformation.c...</li><li><a href="https://x.com/colin_fraser/status/1834334418007457897">来自 Colin Fraser (@colin_fraser) 的推文</a>: 它很笨 :(</li><li><a href="https://x.com/wgussml/status/1834691198013129053">来自 william (@wgussml) 的推文</a>: 大多数人会忽略的是，o1 的重要性恰恰在于它不是在合成数据上进行的 SFT。事实上，在无约束的 CoT 上进行 RL 是有效的，且不会崩溃成胡言乱语的 CoT 步骤，这真的很...</li><li><a href="https://x.com/_jasonwei/status/1834278706522849788?s=46">来自 Jason Wei (@_jasonwei) 的推文</a>: 非常激动终于能分享我在 OpenAI 一直在做的工作！o1 是一个在给出最终答案之前会进行思考的模型。用我自己的话来说，这是 AI 领域最大的更新（见下文...）</li><li><a href="https://x.com/aaronp613/status/1834393945050087567?s=46">来自 Aaron (@aaronp613) 的推文</a>: 苹果发布了 3 个推广 Apple Intelligence 的新视频，由 Bella Ramsey 主演 🧵 第 1 个：更具个性化的 Siri</li><li><a href="https://x.com/karpathy/status/1834666824904196222">来自 Andrej Karpathy (@karpathy) 的推文</a>: 为 @theworldlabs 的发布感到非常兴奋！我在读博期间与李飞飞和 Justin 共度了很长时间，那段回忆非常美好——飞飞是我的导师和无畏的领导者，Justin 和...</li><li><a href="https://x.ai/profile-settings">xAI 登录</a>: 未找到描述</li><li><a href="https://ide.x.ai">PromptIde</a>: 未找到描述</li><li><a href="https://developers.x.ai/api/api-key/">创建 API Key - xAI 开发者平台</a>: 未找到描述</li><li><a href="https://x.com/drjimfan/status/1834284702494327197?s=46">来自 Jim Fan (@DrJimFan) 的推文</a>: 这可能是自 2022 年原始 Chinchilla Scaling Law 以来 LLM 研究中最重要的图表。核心见解是两条曲线协同工作，而不是一条。人们一直在预测...的停滞</li><li><a href="https://x.com/dkhos/status/1834599125310132625">来自 dara khosrowshahi (@dkhos) 的推文</a>: 与我们的合作伙伴 @Waymo 迈出了一大步。很快，你就能在奥斯汀和亚特兰大通过 @Uber App 呼叫 @Waymo 自动驾驶汽车。很高兴能实现这一目标！引用 Tekedra N Mawakana (@TechTekedra)...</li><li><a href="https://x.com/voooooogel/status/1834569673712754805?s=46">来自 thebes (@voooooogel) 的推文</a>: 如果你询问 o1 太多次关于其推理过程的问题，OpenAI 发给你的邮件。引用 thebes (@voooooogel) @teortaxesTex 如果我在提示词中提到“reasoning trace”这个词，我就会收到这封吓人的信...</li><li><a href="https://x.com/teortaxestex/status/1834297569545257297?s=46">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>: 令人惊讶：在一些显然需要通用推理的 Agent 任务中，Sonnet/4o 与 o1 旗鼓相当。我猜通用性也是一种领域专业化，而不是一种涌现能力...</li><li><a href="https://x.com/nickfloats/status/1834332468662391043?s=46">来自 Nick St. Pierre (@nickfloats) 的推文</a>: 好了，@suno_ai_ 刚刚发布了一个名为“Covers”的新 AI 音乐功能，简直是魔法。它能处理你的声音。你对着 Suno 唱歌，给它一个提示词，它就会将你的歌声转化为...</li><li><a href="https://x.com/willdepue/status/1834294935497179633?s=46">来自 will depue (@willdepue) 的推文</a>: 对今天推理模型发布真正意义的一些反思：新范式。我真的希望人们理解这是一个新范式：不要指望同样的节奏、时间表或动态...</li><li><a href="https://x.com/cursor_ai/status/1834665828308205661">来自 Cursor (@cursor_ai) 的推文</a>: OpenAI 的新 o1 模型现已可用

e 在 Cursor 中使用！我们发现 o1 在规范明确、推理密集型的问题上表现卓越。对于大多数任务，我们仍然推荐使用 sonnet/4o。我们最初正在推出...</li><li><a href="https://x.com/fabianstelzer/status/1834300757241102588?s=46">来自 fabian (@fabianstelzer) 的推文</a>：我首选的 LLM 测试是看模型是否能正确解释这个笑话：“两头牛站在田野里，一头牛问另一头：‘你对最近流行的疯牛病怎么看？’。另一头...”</li><li><a href="https://x.com/percyliang/status/1834309959565111673?s=46">来自 Percy Liang (@percyliang) 的推文</a>：HELM MMLU v1.8.0 和 HELM lite（10 个多样化场景）v1.8.0 发布了！Writer 的新模型 Palmyra-X-004 在这两项测试中都进入了前 10 名，这是一个由巨头（OpenAI, Anthropic, ...）主导的竞争极其激烈的领域。</li><li><a href="https://x.com/cognition_labs/status/1834292718174077014?s=46">来自 Cognition (@cognition_labs) 的推文</a>：在过去的几周里，我们与 OpenAI 密切合作，利用 Devin 评估了 OpenAI o1 的推理能力。我们发现这一系列新模型对 Agent 系统有显著提升...</li><li><a href="https://x.com/sama/status/1834351981881950234">来自 Sam Altman (@sama) 的推文</a>：@MattPaulsonSD 先对天空中的神奇智能保持几周的感激之情如何，然后你很快就能拥有更多玩具了？</li><li><a href="https://x.com/mathemagic1an/status/1834383859456377208?s=46">来自 Jay Hack (@mathemagic1an) 的推文</a>：看起来 o1-preview 似乎可以访问某种计算器，并在推理时调用它。我从未见过模型在常规 Token 流式传输过程中能算出 700,112 * 9 并得出正确结果。B...</li><li><a href="https://x.com/allgarbled/status/1834344480797057307?s=46">来自 dr. garbled (@allgarbled) 的推文</a>：为冬天做好准备。引用 dr. garbled (@allgarbled)：打算用新的 GPT 来测试我那个所有其他模型都失败了的秘密代数问题。如果它成功了，那么人类就要准备好...</li><li><a href="https://x.com/AndrewMayne/status/1834408991839158422">来自 Andrew Mayne (@AndrewMayne) 的推文</a>：我已经使用 OpenAI 的 o1 好几周了。我关于使用它的建议：1. 不要把它看作传统的聊天模型。在你的脑海中把 o1 构想成一个你打算去见的一位非常聪明的朋友...</li><li><a href="https://x.com/OpenAI/status/1834320155989664067">来自 OpenAI (@OpenAI) 的推文</a>：OpenAI o1 背后的一些研究人员 🍓</li><li><a href="https://x.com/cognition_labs/status/1834292718174077014">来自 Cognition (@cognition_labs) 的推文</a>：在过去的几周里，我们与 OpenAI 密切合作，利用 Devin 评估了 OpenAI o1 的推理能力。我们发现这一系列新模型对 Agent 系统有显著提升...</li><li><a href="https://x.com/8teapi/status/1834450848505888793?s=46&t=jDrfS5vZD4MFwckU5E8f5Q">来自 Ate-a-Pi (@8teAPi) 的推文</a>：o1 个人测试合集贴 🧵 如果需要请收藏，只是为了追踪大家的反应，因为我们很多人都保留了个人测试集。</li><li><a href="https://x.com/theworldlabs/status/1834563552750838117">来自 World Labs (@theworldlabs) 的推文</a>：你好，世界！我们是 World Labs，一家空间智能公司，致力于构建大型世界模型 (LWMs) 以感知、生成并与 3D 世界互动。阅读更多：https://www.worldlabs.ai/about</li><li><a href="https://www.youtube.com/watch?v=pg3qwgnekQo">No Priors 第 79 集 | 对话 Magic.dev CEO 兼联合创始人 Eric Steinberger</a>：在今天的 No Priors 中，Sarah Guo 和 Elad Gil 邀请到了 Magic.dev 的联合创始人兼 CEO Eric Steinberger。他的团队正在开发一款软件工程师协作者...
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1284242627889336353)** (131 条消息🔥🔥): 

> - `Cursor 问题`
> - `使用 AI 工具`
> - `Vim 与 IDE 偏好`
> - `HTEC AI Copilot 报告`
> - `Neovim 学习资源` 


- **Cursor 面临扩展性问题**：成员们讨论了 **Cursor** 似乎存在扩展性（scaling）问题，特别是在代码补全和文档生成方面。
   - *“他们对 Cursor 的代码补全说‘不’？”*，这引发了对其研究方法的质疑。
- **探索 AI Copilot 与 IDE**：一份来自近岸咨询公司的报告评估了包括 **Cursor** 和 **Claude** 在内的多种 AI copilot，以了解其可用性。
   - 尽管最初对 Copilot 印象平平，但成员们指出，使用 AI 工具最终会提高效率，尤其是在编程方面。
- **Vim 的优势与挑战**：成员们表示 **Vim** 的学习曲线很陡峭，但也承认一旦掌握，它能显著提高编码速度。
   - 一些用户通过完成 **Vim Adventures** 游戏来提高技能，突显了在学习环境中的灵活性。
- **HTEC AI 报告的见解**：HTEC 团队评估了 **26 种 AI 工具**，尽管参与者对每种工具都进行了“尝试”，但由于测试时间有限，结果尚无定论。
   - 该报告主要用于潜在客户开发（lead generation），引发了对其关于 AI copilot 深度和分析的质疑。
- **Neovim 资源与社区互动**：社区成员分享了掌握 **Neovim** 的各种资源，包括一个关于配置的实用 YouTube 播放列表。
   - 通过对学习路径的大量讨论，社区促进了在探索新开发工具和技术方面的协作。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://gptengineer.app/">GPT Engineer</a>: 仅使用聊天界面构建软件产品</li><li><a href="https://vim-racer.com/">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/playlist?list=PLx2ksyallYzW4WNYHD9xOFrPRYGlntAft">Understanding Neovim</a>: 成为配置 Neovim 的高手！</li><li><a href="https://openv0.dev/">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=UdB50GZfn5A">AI in action: AI-powered automation feat. langflow</a>: 深入探讨使用 AI 定制自动化工具。在这段详细的视频中探索使用 AI 定制自动化工具的复杂性。从...开始</li><li><a href="https://github.com/tris203/precognition.nvim">GitHub - tris203/precognition.nvim: 💭👀precognition.nvim - Precognition 使用虚拟文本和侧边栏符号来显示可用的移动操作。</a>: 💭👀precognition.nvim - Precognition 使用虚拟文本和侧边栏符号来显示可用的移动操作。 - tris203/precognition.nvim</li><li><a href="https://github.com/nvim-lua/kickstart.nvim">GitHub - nvim-lua/kickstart.nvim: 个人 nvim 配置的起点</a>: 个人 nvim 配置的起点 - nvim-lua/kickstart.nvim</li><li><a href="https://github.com/latentspacenotes/latentspacenotes.github.io">GitHub - latentspacenotes/latentspacenotes.github.io</a>: 通过在 GitHub 上创建账户，为 latentspacenotes/latentspacenotes.github.io 的开发做出贡献。</li><li><a href="https://github.com/ThePrimeagen/harpoon/tree/harpoon2">GitHub - ThePrimeagen/harpoon at harpoon2</a>: 通过在 GitHub 上创建账户，为 ThePrimeagen/harpoon 的开发做出贡献。</li><li><a href="https://github.com/raidendotai/openv0">GitHub - raidendotai/openv0: AI 生成的 UI 组件</a>: AI 生成的 UI 组件。通过在 GitHub 上创建账户，为 raidendotai/openv0 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1283917849131221012)** (11 条消息🔥): 

> - `Quantization Techniques`
> - `Metal Kernel Coding for MPS`
> - `CIFAR10 Model Training` 


- **Quantization Techniques 实验**: 一位成员目前在初步测试中对输入和权重分别应用 **quantization** 和 **dequantization**，以提高模型准确性，并指出引入输入 activation quantization 可能会阻碍性能。
   - 另一位成员建议对 activation 使用 dynamic quantization 应该会有不错的效果，并强调了调试实现过程以解决性能问题的重要性。
- **访问代码中的 Quantization 逻辑**: 成员们讨论了在调试 quantization 逻辑时遇到的困难，原因是缺乏对 `input_quantizer` 和 `weight_quant` 实现的可见性，并引用了托管在 [GitHub](https://github.com/satabios/quantization/tree/master/quant) 上的代码。
   - 一位成员请求提供一个最小可运行示例（minimal running example），以便更有效地理解和调试 quantization 过程。
- **Activation Quantization 的挑战**: 一位成员指出，与仅权重（weight-only）变体相比，他们在 **CIFAR10** 上训练的简单模型在使用 activation quantization 变体时表现出剧烈的性能下降。
   - 该成员鼓励其他人克隆该仓库以获取更多见解，并协助解决在设置过程中遇到的任何问题。
- **Metal Kernel Coding 直播计划**: 另一位成员表示计划在周末进行针对 MPS 后端的 **metal kernel coding**，并询问是否有兴趣观看该环节的直播。
   - 这一举措可能会吸引对 kernel coding 细节和实时编码体验感兴趣的观众。


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1283932196054237274)** (1 条消息): 

> - `ASPLOS 2024`
> - `Inductor Components` 


- **ASPLOS 2024 Colab Notebooks 概览**: 一位成员提到了 [ASPLOS 2024 colab notebooks](https://link.to.notebooks) 的存在，这些 notebook 提供了关于有效使用的见解。
   - 虽然内部细节尚不清楚，但这些 notebook 展示了 **如何利用 Inductor 的所有组件**。
- **探索 Inductor 功能**: 讨论强调了 colab notebooks 在帮助 **理解 Inductor 的各种功能** 和使用场景方面的潜力。
   - 成员们表示有兴趣探索更多与 colab notebooks 相关的详细讨论或示例。


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1284181086066442321)** (5 条消息): 

> - `WebGPU Puzzles`
> - `GameGen-O`
> - `Interactive GPU Programming`
> - `GPU Puzzles`
> - `Demo Feedback` 


- **WebGPU Puzzles 为浏览器用户发布**: 一个新应用 [WebGPU Puzzles](https://gpupuzzles.answer.ai) 允许用户直接在浏览器中尝试 kernel hacking，从而有效地向更广泛的受众开放了 GPU 编程。
   - 该平台基于 [Sasha Rush](https://rush-nlp.com/) 之前的工作，允许用户在利用本地 GPU 资源的同时，参与小型互动式编码挑战。
- **GameGen-O 引发关注**: [GameGen-O GitHub](https://github.com/GameGen-O/GameGen-O) 项目已分享以供贡献，该项目专注于游戏生成技术，吸引了社区中的开发者。
   - 此外，[GameGen-O demo site](https://gamegen-o.github.io/) 展示了其功能，并记录了来自各贡献者的协作努力。
- **对 Demo 的积极反馈**: 关于 WebGPU Puzzles 的 demo，反响非常热烈，强调了其令人印象深刻的功能和易用性。
   - 多位用户表达了他们的兴奋之情，并有兴趣通过互动 demo 进一步探索 GPU 编程。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gpupuzzles.answer.ai/">webgpu puzzles</a>: 未找到描述</li><li><a href="https://www.answer.ai/posts/2024-09-12-gpupuzzles.html">Learn GPU Programming in Your Browser – Answer.AI</a>: 实用 AI 研发</li><li><a href="https://gamegen-o.github.io/">GameGen-O: Open-world Video Game Generation</a>: 未找到描述</li><li><a href="https://github.com/GameGen-O/GameGen-O/">GitHub - GameGen-O/GameGen-O</a>: 通过在 GitHub 上创建账户来为 GameGen-O/GameGen-O 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1283899850068594758)** (1 条消息): 

> - `Aurora Innovation 正在招聘`
> - `Aurora 无人驾驶卡车的商业化落地`
> - `Aurora 融资成功`
> - `全新的商业化就绪终端`
> - `达拉斯与休斯顿之间的扩张计划` 


- **Aurora Innovation 寻求优秀工程师！**：Aurora Innovation 正在招聘专注于推理和训练的 **GPU acceleration** 的 L6 和 L7 级工程师，特别强调 **CUDA**、**Triton** 以及 **Nsight** 等工具。感兴趣的候选人可以在 [Aurora 招聘列表](https://aurora.tech/jobs/staff-software-engineer-deep-learning-acceleration-7518608002) 查看更多详情。
   - 这些职位提供极具竞争力的薪酬，鼓励潜在申请人通过 **DM** 获取更多信息。
- **Aurora 加速推进 2024 年无人驾驶落地！**：Aurora Innovation 的目标是在 **2024** 年底前实现其无人驾驶卡车服务的 **commercial launch**。其股价在过去六个月中显著 **翻倍**，在过去 1.5 年中增长了 **三倍**。
   - 通过完成重要的里程碑，Aurora 展示了其对无人驾驶商业未来的准备情况，并获得了越来越多的投资支持。
- **Aurora 融资 4.83 亿美元用于扩张！**：Aurora Innovation 成功筹集了 **4.83 亿美元**，超过了其 **4.2 亿美元** 的目标，为即将到来的商业化落地做准备。此次融资紧随去年 7 月筹集的 **8.2 亿美元** 资金之后。
   - 在 **Analyst Day** 之后，投资者信心大增，他们在当天体验了无人驾驶卡车，并了解了 Aurora 的合作伙伴生态系统。
- **新终端助力 Aurora 运营！**：Aurora 在休斯顿开设了首个 **commercial-ready terminals**，使其能够支持 **Dallas** 和 **Houston** 之间的无人驾驶卡车。这些终端旨在昼夜运行，每周处理超过 **75 批商业货物**。
   - 这一战略举措使 Aurora 在繁忙的 **I-45 货运走廊** 中占据了有利位置，满足了德克萨斯州大量的卡车运输需求。
- **Aurora 开通关键无人驾驶卡车车道！**：Aurora 宣布开通由其商业化就绪终端支持的 **行业首条** 无人驾驶卡车车道。该路线连接 **Dallas** 和 **Houston**，接入了德克萨斯州的主要货运大动脉。
   - 通过运营终端，Aurora 旨在简化物流，并展示大规模 **autonomous hauling** 的可行性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://aurora.tech/jobs/staff-software-engineer-deep-learning-acceleration-7518608002">Staff Software Engineer - Deep Learning Acceleration </a>: 我们正在招聘技术和业务领导者！加入全球最有经验的团队，安全、快速、广泛地将自动驾驶技术推向市场。软件平台软件与服务...</li><li><a href="https://aurora.tech/jobs/sr-staff-software-engineer-ml-accelerators-5574800002">Sr Staff Software Engineer, ML Accelerators</a>: 我们正在招聘技术和业务领导者！加入全球最有经验的团队，安全、快速、广泛地将自动驾驶技术推向市场。企业发展与战略合作...</li><li><a href="https://techcrunch.com/2024/08/02/self-driving-truck-startup-aurora-innovation-raises-483m-commercial-launch/">无人驾驶卡车初创公司 Aurora Innovation 在商业化落地前通过股票发售筹集 4.83 亿美元 | TechCrunch</a>: 自动驾驶技术公司 Aurora Innovation 希望在冲刺无人驾驶的过程中筹集数亿美元的额外资金</li><li><a href="https://ir.aurora.tech/news-events/press-releases/detail/84/aurora-opens-first-commercial-ready-route-for-its-planned">Aurora 为其计划于 2024 年底推出的无人驾驶卡车开通首条商业化就绪路线</a>: 随着休斯顿商业化就绪终端的首次亮相，Aurora 可以支持并服务于达拉斯和休斯顿之间的无人驾驶卡车。Aurora……...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 条消息): 

yelr: 谢谢！我会去看看的
  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1283906147404746898)** (4 条消息): 

> - `int8 和 fp16 矩阵乘法`
> - `PyTorch 量化技术`
> - `optimum-quanto 内核`
> - `_weight_int8pack_mm 函数` 


- **高效的 int8 和 fp16 矩阵乘法**：据解释，可以在 GPU 上执行 **fp16 输入/int8 matmul** 而无需反量化，因为 int8 权重在内核内部直接转换为 fp16。
   - *在当前的实现中*，torch.compile 会生成一个混合 matmul 的 Triton 内核，这意味着不会发生不必要的反量化。
- **关于 PyTorch 量化技术的见解**：为了减少内存占用，尝试结合 bfloat16 的 **int4_weight_only** 量化或 **fp6 量化 (fpx_weight_only(3, 2))** 可能会有所帮助。
   - 有关量化技术的进一步参考，文中提供了一个指向 [文档](https://github.com/pytorch/ao/tree/main/torchao/quantization#quantization-techniques) 的链接。
- **关于 _weight_int8pack_mm 函数的讨论**：据推测，`_weight_int8pack_mm` 函数的操作方式类似于 **fp16 输入/int8 matmul** 的处理过程，即通过将权重矩阵转换为激活的数据类型并应用缩放（scaling）。
   - 这表明在矩阵乘法操作中可以高效地处理混合数据类型。
- **参考 optimum-quanto 的内核**：提到了用于量化的 **optimum-quanto** 内核，特别是在其项目结构中展示了非 torchao 技术。
   - 讨论的内核在他们的仓库中有详细说明，这可以为替代方案提供见解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/blob/e157ce3ebbb3f30d008c15914e82eb74217562f0/aten/src/ATen/native/native_functions.yaml#L4154">pytorch/aten/src/ATen/native/native_functions.yaml at e157ce3ebbb3f30d008c15914e82eb74217562f0 · pytorch/pytorch</a>：Python 中的张量和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch</li><li><a href="https://github.com/huggingface/optimum-quanto/blob/9d50ea5816b67e8d5c6e34dbc427631d98799535/optimum/quanto/library/qbytes_mm.py">optimum-quanto/optimum/quanto/library/qbytes_mm.py at 9d50ea5816b67e8d5c6e34dbc427631d98799535 · huggingface/optimum-quanto</a>：用于 optimum 的 PyTorch 量化后端。通过在 GitHub 上创建账号为 huggingface/optimum-quanto 的开发做出贡献。</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/quantization#quantization-techniques">ao/torchao/quantization at main · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/8236a874479a9a9168e584c81dda8707f4c41006/torchao/dtypes/affine_quantized_tensor.py#L1474-L1480">ao/torchao/dtypes/affine_quantized_tensor.py at 8236a874479a9a9168e584c81dda8707f4c41006 · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1283898946116124723)** (117 条消息🔥🔥): 

> - `O1 模型评估`
> - `Aider 工具`
> - `Sora 复现挑战`
> - `像素艺术模型构想`
> - `模型规模考量` 


- **O1 与 Sonnet 的性能对比**：几位成员对新的 O1 模型表示怀疑，有人称其“平平无奇”（nothingburger），因为在各项基准测试中其性能与 Sonnet 相当。
   - 重点讨论的任务包括思维链（Chain of Thought）能力和通用易用性，质疑 O1 是否真的带来了突破。
- **介绍 Aider 作为编程辅助工具**：Aider 是一款专为终端环境下的 AI 结对编程设计的工具，通过创建 Git 提交和处理上下文缓存（Context Caching）来实现高效编码。
   - 它与 Claude Sonnet 等模型的集成因能促进项目完成并减少重复编码开销而受到赞誉。
- **复现 Sora 的挑战**：成员们讨论了复现 Sora 模型的困难，提到虽然底层理论已知，但挑战在于所需的巨大计算资源。
   - 这引发了对小型项目的思考，例如 llm.c，它可以在单节点可用资源上进行管理。
- **像素艺术模型提案**：有人提议构建一个像素艺术模型，并建议实现 16x16 GIF 模型等小规模应用，这让潜在的开发者们感到兴奋。
   - 讨论反映出探索图形项目的愿望，试图脱离语言模型的复杂性。
- **理解模型规模（Scale）的作用**：成员们断言，虽然 GPT2 和 Sora 等模型的基础概念已被理解，但实现的规模仍然是一个关键障碍。
   - 调整模型大小和探索上采样（Upscaling）被认为是未来项目的可行路径。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=41359152">无标题</a>：无描述</li><li><a href="https://aider.chat/docs/usage/caching.html">Prompt caching</a>：Aider 支持 Prompt 缓存，以节省成本并加快编码速度。</li><li><a href="https://aider.chat">首页</a>：Aider 是你终端里的 AI 结对编程工具
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 条消息): 

ssp3ll: 我也在多伦多。
  

---


### **CUDA MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1283875532013830208)** (5 条消息): 

> - `torch.compile 支持`
> - `HQQ+ 训练代码`
> - `HQQ 与 QLoRA 的关系` 


- **与 transformers 集成的 torch.compile 支持**：HQQ 0.2.2 最新版本现在直接支持 `torch.compile` 与 transformers 的 `model.generate()` 功能配合使用，不再需要 HFGenerator。
   - 一位成员强调了这一增强功能，使开发者的集成过程更加顺畅。
- **HQQ+ 训练代码可用性**：成员们询问了 HQQ+ 训练代码的可用性，特别关注 mobicham 分享的一个使用 HF peft 的示例。
   - 提供的 [示例链接](https://github.com/mobiusml/hqq/blob/master/examples/lora/hqq_plus.py) 展示了 **半二次量化 (HQQ)** 的官方实现。
- **理解 HQQ+ 即 HQQ + QLoRA**：一位成员确认 HQQ plus 指的是 **HQQ 和 QLoRA** 的结合，并强调了两者之间的区别。
   - Mobicham 澄清说，训练通常涉及模型蒸馏（Model Distillation）而非 SFT 训练，但为了便于理解分享了一个示例。
- **HQQ+ 中的 LoRA 权重处理**：Mobicham 提到，在 HQQ+ 中使用 **LoRA 权重** 时，它们应保持为 **fp16** 格式且不应合并回去。
   - 这种方法与传统做法不同，突显了其训练框架中采用的替代方案。



**提到的链接**：<a href="https://github.com/mobiusml/hqq/blob/master/examples/lora/hqq_plus.py">hqq/examples/lora/hqq_plus.py at master · mobiusml/hqq</a>：半二次量化 (HQQ) 的官方实现 - mobiusml/hqq

  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1284237751847354401)** (51 条消息🔥): 

> - `Llama 3 支持`
> - `CMake vs Makefiles`
> - `RoPE 和 SwiGLU PR`
> - `FlashAttention`
> - `用于 Matmuls 的 CUTLASS` 


- **启动 Llama 3 支持**：已创建一个新的特性分支，用于为 llm.c 添加 **Llama 3 支持**，最初直接复制了 train_gpt2.cu 和 test_gpt2.cu。
   - 意图是在**合并回 master** 之前，让这些文件与原文件产生差异化开发，目前关于 RoPE、SwiGLU 和 GQA 的关键 PR 仍处于待处理状态。
- **CMake vs Makefiles 之争**：一位成员提出了关于偏好 **Makefiles** 而非 **CMake** 的疑问，指出 CMake 不断演进的版本可能会引入兼容性问题。
   - 另一位成员表示赞同，认为 Make 非常稳定，对于依赖较少的小型项目来说表现良好。
- **RoPE 和 SwiGLU PR 的评审请求**：有人请求评审两个 PR，一个用于实现 **RoPE**，另一个用于实现 **SwiGLU**，两者都与 Llama 3 的特性相关。
   - 对 RoPE PR 的反馈显示其看起来不错，并对修改后 encoder kernel 的性能表现表示好奇。
- **探索类 FlashAttention 解决方案**：讨论了将 **naive attention** 适配为类似 **FlashAttention** 的方案，建议在 backward 期间进行重计算（recompute），而不是存储大的张量。
   - 这种方法旨在减少低效的代码结构，同时潜在地提高整体性能。
- **潜在的 CUTLASS 项目**：一位成员建议使用 **CUTLASS 路径** 作为 cuBLAS 的替代方案来进行矩阵乘法（Matmuls），并考虑其对性能的影响。
   - 该提议与目前关于提高当前实现中内存效率的讨论相关联。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/756">Add RoPE positional encoding - llama3 feature branch by gordicaleksa · Pull Request #756 · karpathy/llm.c</a>：实现了 RoPE - 源自 RoFormer 论文的旋转位置嵌入。注：我没有条件性地移除可学习位置嵌入缓冲区 (wpe) 的分配，因为那需要修改...</li><li><a href="https://github.com/karpathy/llm.c/pull/754">add llama 3 support to llm.c by karpathy · Pull Request #754 · karpathy/llm.c</a>：该分支从复制粘贴 train_gpt2.cu 和 test_gpt2.cu 开始，但在合并回 master 之前，这两个文件（以及其他文件）将进行更改以整合 Llama 3.1 支持。</li><li><a href="https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md">cudnn-frontend/docs/operations/Attention.md at main · NVIDIA/cudnn-frontend</a>：cudnn_frontend 为 cudnn 后端 API 提供了一个 C++ 封装以及如何使用它的示例 - NVIDIA/cudnn-frontend</li><li><a href="https://github.com/karpathy/llm.c/pull/755">Add SwiGLU support - llama3 feature branch by gordicaleksa · Pull Request #755 · karpathy/llm.c</a>：实现了 SwiGLU - 源自 "GLU Variants Improve Transformer" 论文的 swish GLU 激活函数。注：由于添加了 add... 导致内存占用增加。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1284192204675481630)** (1 条消息): 

> - `WebGPU Puzzles`
> - `GPU 编程`
> - `Web 应用开发`
> - `本地 GPU 访问`
> - `交互式编程挑战` 


- **WebGPU Puzzles 成为焦点**：一个新的 Web 应用 [WebGPU Puzzles](https://gpupuzzles.answer.ai) 已经发布，旨在利用 **WebGPU** 的能力帮助用户**在浏览器中学习 GPU 编程**。
   - 该应用由 **Sarah Pan** 和 **Austin Huang** 构建，其中的编程挑战灵感源自最初为远程服务器上的 **Numba/CUDA** 设计的 **GPU Puzzles**。
- **直接访问本地 GPU**：**WebGPU** 正式到来，提供了从 Web 浏览器到本地 GPU 的直接管道，使编程更加便捷和实用。
   - 该应用的设计鼓励用户应对编程挑战，并分享关于该技术潜力的创新想法。
- **轻松学习 GPU 编程**：**WebGPU Puzzles** 的交互特性允许你直接在浏览器中编写和执行代码，为 **GPU 编程** 提供了一种简单直接的学习方式。
   - 这种方法让个人无需专用 GPU 设备或复杂的环境配置即可体验动手学习。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gpupuzzles.answer.ai/">webgpu puzzles</a>：未找到描述</li><li><a href="https://www.answer.ai/posts/2024-09-12-gpupuzzles.html">Learn GPU Programming in Your Browser – Answer.AI</a>：实用 AI 研发
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1283872996129505322)** (8 messages🔥): 

> - `Custom Kernels`
> - `LLM Inference`
> - `Quantization and Sparsity`
> - `Multi-GPU Track`
> - `IRL Hackathon RSVP` 


- **用于 FFT 的 Custom Kernels**：一位用户讨论了用于 FFT 的 **Cooley-Tukey algorithm** 的实现，更多细节请参见[此处](https://discord.com/channels/1189498204333543425/1267896441989234709/1283627068034387989)。
   - 该算法旨在优化 **Fast Fourier Transforms**，以提升在各种应用中的性能。
- **GH200 的 KV-Cache Offloading**：一名成员强调了 **GH200** 架构中 **KV-Cache Offloading** 的重要性，并引用了详细的讨论[链接](https://discord.com/channels/1189498204333543425/1267896441989234709/1283635680035082311)。
   - 该技术被视为最大化 **LLM Inference** 效率的关键。
- **探索 Quantization 和 Sparsity**：Hicham 和 Charles 分享了关于 **Quantization** 和 **Sparsity** 项目的见解，并附带了他们的 [Google document](https://docs.google.com/document/d/14BJ7a1wx1uqzrmCbBsLjX7YQNKrtIhRDuz1l0Tlmkoc/) 链接。
   - 他们强调了这些方法在不牺牲性能的情况下提高模型效率的潜在益处。
- **Multi-GPU Track 中的麦克斯韦方程组模拟器**：Georgii 展示了一个 **Maxwell’s equations simulator** 作为 Multi-GPU 环节的项目提案，可通过其 [Google document](https://docs.google.com/document/d/1OxWw9aHeoUBFDOClcMr9UrPW8qmpdR5pPOcwH4jEhms/edit#heading=h.c3hqbft26ocn) 访问。
   - 该模拟器旨在展示 Multi-GPU 设置在模拟复杂物理现象方面的能力。
- **澄清 IRL Hackathon 出席情况**：随后讨论了 **IRL Hackathon** 参与者的状态，明确了 **cuda-mode-irl** 角色表示已接受并确认。
   - 鼓励用户考虑组建远程团队，以便在 Hackathon 期间进行协作。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.google.com/document/d/1YuCvBeMD5wlwI0iAV1xf3aokf4tj53epLNyRFeUuf1U/edit#heading=h.7d2mds49l9g5">Multi-gpu Track</a>：Multi-gpu Track。让 405B 在 4090 或配置较低的 GPU 上运行得更快。目前，可以在 4 台 48GB 的 4090 上装下 Llama-405B，但速度很慢。我们能否将 torch.compile 作为一等公民引入？目前，它共存...</li><li><a href="https://docs.google.com/document/d/14BJ7a1wx1uqzrmCbBsLjX7YQNKrtIhRDuz1l0Tlmkoc/">Quantization and Sparsity Projects</a>：IRL 的 Quantization 和 Sparsity 项目。高性能 Custom Kernels：1. 开发 A16W3（混合 fp16 x 3-bit）Fused Matmul Kernel：为什么？目前还没有可用的 3-bit 线性 Kernel...</li><li><a href="https://docs.google.com/document/d/1OxWw9aHeoUBFDOClcMr9UrPW8qmpdR5pPOcwH4jEhms/edit#heading=h.c3hqbft26ocn">Hackathon Project Proposal for multi-GPU session: Maxwell Equations Simulator</a>：介绍。作为 Multi-GPU Hackathon 环节的一个项目，我建议实现麦克斯韦方程组模拟器。麦克斯韦方程组模拟电磁波的传播。与替代方案相比...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1284134532647358566)** (5 messages): 

> - `Liger Kernel`
> - `BERT Fine-Tuning`
> - `Integration with Thunder` 


- **寻求 Liger Kernel 用于 BERT Fine-Tuning 的帮助**：一名成员请求在使用 **Liger Kernel** 进行 **BERT 模型 Fine-Tuning** 方面提供协助，并寻求参考代码。
   - 回复指出该工作正在进行中，有一个 **draft PR** 待处理，旨在将 **Liger Ops** 集成到 **Thunder** 中以进行增强。
- **如果不使用 Liger Ops 则需要调整模型**：回复建议，如果 **Liger Ops** 不可用，则需要对模型进行修改，类似于其他模型的现有代码。
   - 该成员随后表示打算尝试修改代码以适应其需求。


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1283864984044044411)** (130 messages🔥🔥): 

> - `OpenAI o1 model performance`
> - `California SB 1047 AI safety bill`
> - `AI ethics and policy discussions`
> - `Benchmarking AI models`
> - `Chain-of-Thought reasoning in AI`

- **OpenAI o1 模型性能令人惊喜**：新发布的 OpenAI o1 模型引发了广泛关注，在 AIME 等基准测试中取得了令人印象深刻的分数，但在 ARC Prize 上的表现却出奇地低。
   - 一些用户指出，虽然 o1 在竞赛数学问题上表现出色，但其泛化到其他类型问题的能力仍然有限。
- **加州 SB 1047 法案与 AI 监管**：拟议的关于加州 AI 安全的 SB 1047 法案引发了讨论，由于包括 Pelosi 立场在内的政治因素，估计有 66%-80% 的可能性被否决。
   - 有推测认为，该法案的命运可能取决于围绕资金的政治格局以及公众对 AI 监管的看法。
- **关于 AI 模型基准测试公平性的辩论**：关于 AI 模型基准测试的公平性一直存在争议，特别是关于 pass@k 指标以及它如何与 o1 和 GPT-4o 等模型进行比较。
   - 一些人认为基准测试应该考虑 compute budgets，并指出 o1 的答案选择机制使得与不具备相同资源的模型进行直接比较变得复杂。
- **对 Chain-of-Thought 推理的见解**：用户观察到 o1 中的推理错误可能导致有缺陷的 Chain-of-Thought 输出，错误会螺旋式上升并产生错误的结论。
   - 这一现象突显了在 AI 推理过程中保持连贯性的挑战，以及它对 AI 可靠性的影响。
- **AI 对 Prompt 质量的敏感性**：人们达成共识，认为像 o1 这样的模型对 Prompt 质量表现出高度敏感性，这会显著影响性能，程度可能超过其他模型。
   - 用户推测，Prompt 表述中的细微差别可能会导致模型输出的巨大差异，尤其是在处理复杂任务时。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/steph_palazzolo/status/1834348474479091879?s=46">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：OpenAI 的大日子：根据我们获得的新使用指标，ChatGPT 每月产生的收入超过 2.25 亿美元（这还是保守估计） https://www.theinformation.c...</li><li><a href="https://arcprize.org/blog/openai-o1-results-arc-prize">OpenAI o1 在 ARC-AGI-Pub 上的结果</a>：o1 preview 和 mini 模型距离 AGI 还有多远？</li><li><a href="https://x.com/paulgauthier/status/1834339747839574392?s=61">来自 Paul Gauthier (@paulgauthier) 的推文</a>：o1-mini 的首次基准测试运行显示，它在 aider 的代码编辑基准测试中与 GPT-4o 基本持平。随着更多基准测试运行的完成，本文将持续更新：https://aider.chat/2024/09/12/o1.htm...</li><li><a href="https://fxtwitter.com/terryyuezhuo/status/18343278083333754631?s=46">来自 Terry Yue Zhuo (@terryyuezhuo) 的推文</a>：对 o1 System Card 的评论：https://cdn.openai.com/o1-system-card.pdf 0. 模型仍然有预训练阶段。1. 他们必须支付高昂费用来获取高质量数据。2. 他们学到了一些...</li><li><a href="https://x.com/HaveFunWithAI/status/1834556906720948308">来自 HaveFunWithAI (@HaveFunWithAI) 的推文</a>：更新：增加了 o1-mini。引用 HaveFunWithAI (@HaveFunWithAI) 更新：增加了 Gemini-1.5-pro-exp-0827。为 Gemini-1.5-pro-exp-0827 再次运行场景 D 两次：- 平均值：46.30% (45.45%, 45.96% & 47.47%)...</li><li><a href="https://polymarket.com/event/will-california-pass-sb-1047-ai-safety-bill/will-california-pass-sb-1047-ai-safety-bill?tid=1725767181654">Polymarket | 加州会通过 SB 1047 AI 安全法案吗？...</a>：Polymarket | 加州的 SB 1047 AI 安全法案目前正在州议会进行辩论。立法者在 8 月 31 日之前通过该法案，如果获得批准，州长在 9 月 3 日之前...</li><li><a href="https://x.com/HaveFunWithAI/status/1834357735720128758">来自 HaveFunWithAI (@HaveFunWithAI) 的推文</a>：如果没有进一步的测试时计算（test-time compute）扩展，o1 模型在 AIME 2024 上的表现并不那么令人印象深刻 - o1-mini (blackbox): 15/30, 50% - o1-preview (blackbox): 14/30, 46.67%。作为参考：经过微调的 GPT-4o ...</li><li><a href="https://x.com/colin_fraser/status/1834623952788033925">来自 Colin Fraser (@colin_fraser) 的推文</a>：在我本周最后几次使用 o1-mini 额度时，我注意到一件事：推理中的一个错误会导致思维链（Chain-of-Thought）的胡言乱语螺旋式失控，同时强化了错误并...</li><li><a href="https://x.com/rao2z/status/1834314021912359393?s=46">来自 Subbarao Kambhampati (కంభంపాటి సుబ్బారావు) (@rao2z) 的推文</a>：..是的，我们正在试用 o1 模型。情况相当复杂；请保持关注。（此外，任何严肃的评估都受到每周 30 条提示词限制的阻碍。如果 @polynoamial 真的想，我确定...）</li><li><a href="https://x.com/SafetyChanges/status/1834350937587974611">来自 AI Safety Corporate Policy Changes (@SafetyChanges) 的推文</a>：又是一天，OpenAI 又一次对其预发布研究的作者身份进行了微妙的修订。这一次是 GPT-4o System Card，删减了一位在 6 月份从公司辞职的研究员的名字...</li><li><a href="https://x.com/tianle_cai/status/1834283977613390001?s=46">来自 Tianle Cai (@tianle_cai) 的推文</a>：o1 的思维链（Chain-of-Thought）包含许多口语表达，如 'Hmm'、'But how?' 等。他们是否在使用讲座录音来训练这个模型...</li><li><a href="https://x.com/lupantech/status/1834301611960926308">来自 Pan Lu (@lupantech) 的推文</a>：🚀 o1 现已由 OpenAI 发布！它经过训练，通过长思维链进行慢思考。它的表现令人印象深刻，可能会攻克科学和数学领域的难题，在 #... 上以 73.2% 的成绩创下新的 SOTA。</li><li><a href="https://x.com/sytelus/status/1834352532585676859">来自 Shital Shah (@sytelus) 的推文</a>：哇.... ChatGPT o1 在我的私有基准测试中获得了 80% 的分数。之前的最佳成绩是 Sonnet 3.5 的 30% 和 GPT-4o 的 20%。在人们得出存在某种简单新算法的结论之前，等等...</li><li><a href="https://x.com/ClementDelangue/status/1834283206474191320">来自 clem 🤗 (@ClementDelangue) 的推文</a>：再次强调，AI 系统不是在“思考”，而是在“处理”、“运行预测”……就像 Google 或计算机所做的那样。给人一种技术系统...的错误印象。</li><li><a href="https://x.com/_jasonwei/status/1834278706522849788?s=61">来自 Jason Wei (@_jasonwei) 的推文</a>：非常激动终于能分享我在 OpenAI 的工作成果！o1 是一个在给出最终答案之前会进行思考的模型。用我自己的话来说，这是 AI 领域最大的更新（见...）</li><li><a href="https://x.com/isidentical/status/1834302726785601616">来自 batuhan taskaya (@isidentical) 的推文</a>：如果有人需要免费访问 o1，可以在这里使用（这是一个临时游乐场，请不要使用

作为 API): https://fal.ai/models/fal-ai/openai-o1/</li><li><a href="https://manifold.markets/ZviMowshowitz/will-california-bill-sb-1047-become-law-this-session?">加州 AI 监管法案 SB 1047 会在本会期成为法律吗？</a>：51% 的概率。来自旧金山的加州参议员 Scott Weiner 提出了该法案 (https://twitter.com/Scott_Wiener/status/1755650108287578585, https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bil...</li><li><a href="https://x.com/max_a_schwarzer/status/1834280954443321694">Max Schwarzer (@max_a_schwarzer) 的推文</a>：system card (https://openai.com/index/openai-o1-system-card/) 很好地展示了 o1 的高光时刻——我最喜欢的是当模型被要求解决一个 CTF 挑战时，意识到目标 ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1283873062168690730)** (23 messages🔥): 

> - `API Tier System`
> - `OpenAI Reasoning`
> - `Functionality of Summarizers`
> - `Generative RM Exploration`
> - `Recent Release Announcements` 


- **理解 API Tier 系统**：成员们讨论了 **API tier 系统**，指出要达到 **Tier 5**，必须花费 **$1000**。一份个人分享显示，一名用户目前处于 **Tier 3**，而另一人提到某个特定团队已经超过了 Tier 5。
- **总结器的忠实度没有保证**：人们对总结器的可靠性表示担忧，引用的一句话是：*“虽然我们希望总结器是忠实的，但并不能保证这一点。”* 这表明在假设其遵循 Chain of Thought (CoT) 时需要保持谨慎。
- **关于推理机制的幽默**：出现了一个轻松的评论，质疑 **Chain of Thought** 是真的有效，还是仅仅依赖于 pause tokens。成员们对 AI 推理能力的复杂性交换了笑声。
- **Generative RM 和 Exploration Tokens**：讨论暗示了 **generative reward models** 使用专门的 token，如 *'think more'* 和 *'explore tokens'*。有人猜测这些模型在模拟那些尽管潜在复杂但易于部署的功能。
- **对近期发布的兴奋**：表达了对近期发布的整体热情，一位成员表示：*“这次发布很有趣”*，并且他们很兴奋能为此写点东西。这种情绪反映了用户中的积极反响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/voooooogel/status/1834569673712754805?s=46">thebes (@voooooogel) 的推文</a>：如果你询问 o1 太多次关于其推理过程，OpenAI 发给你的邮件。引用 thebes (@voooooogel) 的话：@teortaxesTex 如果我在 prompt 中提到 “reasoning trace” 这个词，我就会收到那封吓人的信...</li><li><a href="https://x.com/polynoamial/status/1834644274417119457?s=46">Noam Brown (@polynoamial) 的推文</a>：@sog_on_bird_app @OpenAIDevs 虽然我们希望总结器是忠实的，但并不能保证这一点。我绝对不建议假设它对 CoT 是忠实的，或者 CoT 本身...</li><li><a href="https://x.com/voooooogel/status/1834536216160768377?s=46">thebes (@voooooogel) 的推文</a>：@teortaxesTex 哈哈，如果我在 prompt 中提到 “reasoning trace” 这个词，我就会收到那封吓人的信</li><li><a href="https://x.com/thexeophon/status/1834314098554929217?s=46">Xeophon (@TheXeophon) 的推文</a>：@terryyuezhuo 哈哈，如果 matts 的 benchmark 是用 o1 做的会怎样
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1283892925935255564)** (2 messages): 

> - `Xeophon Interaction`
> - `Logan Discussion` 


- **Xeophon 表情符号趣事**：一位成员对之前的讨论分享了一个 emoji 反应 <:3berk:794379348311801876>，为频道增添了俏皮的基调。
   - 这种互动为 meme 专注的聊天中经常出现的轻松氛围做出了贡献。
- **Logan 的伟大**：另一位成员用一句简单的话表达了他们的钦佩：“Logan 太棒了。”
   - 这条评论或许引发了社区内关于角色欣赏的进一步讨论。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1283866415258140723)** (131 条消息🔥🔥): 

> - `A1111 和 Forge 的性能`
> - `Pony 模型的提示词和标签`
> - `艺术生成中的挑战`
> - `诈骗与投资讨论`
> - `绘制生成时间图表` 


- **A1111 vs Forge：生成时间与质量**：一位用户询问在比较 Forge/A1111 中的 Flux 模型与 Steps 时，是否可以在 XYZ 图表上叠加生成时间，以分析性能。
   - 他们指出 Schnell 生成图像的速度比 Dev 快，但质量较低，这引发了关于速度与质量之间权衡的讨论。
- **对 Pony 模型使用的困惑**：关于在 Pony 模型中使用 score 标签的意图和结果不明确的讨论，突显了其训练数据中存在的系统性不一致。
   - 一些用户对这类提示词的感知效果表示怀疑，认为它们可能无法达到预期效果。
- **对诈骗机会的担忧**：一位用户批评了与投资诈骗相关的提议，强调了识别欺诈机会和诱导手段的重要性。
   - 评论反映了对某些提议（尤其是加密货币讨论中）欺骗性质的广泛担忧。
- **关于动态采样器和 AI 增长的讨论**：动态补偿采样器被认为是 AI 模型训练中有益的创新，用户对最近的发展表示了兴趣。
   - 对话强调了新兴工具在增强图像生成技术有效性方面的潜力。
- **优质 Token 在 AI 生成中的重要性**：用户分享了关于生成高质量图像的有效提示词 Token 的见解，其中 'cinematic' 和 'scenic colorful background' 等 Token 被指出非常实用。
   - 对话揭示了关于使用高级模型的不同观点，以及对基于研究的最佳 Token 使用见解的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OutofAi/OutofFocus">GitHub - OutofAi/OutofFocus: An AI focused photo manipulation tool based on Gradio</a>：一个基于 Gradio 的以 AI 为核心的照片处理工具 - OutofAi/OutofFocus</li><li><a href="https://github.com/butaixianran/Stable-Diffusion-Webui-Civitai-Helper">GitHub - butaixianran/Stable-Diffusion-Webui-Civitai-Helper: Stable Diffusion Webui Extension for Civitai, to manage your model much more easily.</a>：Civitai 的 Stable Diffusion Webui 扩展，让你更轻松地管理模型。 - butaixianran/Stable-Diffusion-Webui-Civitai-Helper
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1283872749307035678)** (68 条消息🔥🔥): 

> - `o1-preview 逐步推出`
> - `模型性能`
> - `LLM 的 GPU 考量`
> - `文本转语音 API 开发`
> - `GPU 市场趋势` 


- **o1-preview 分批推送**：成员们报告称已分批获得 `o1-preview` 的访问权限，其中一位指出它在 Windows 内部机制等任务上表现良好。
   - 随着用户开始使用该功能，大家感到很兴奋，尽管有些人对推送进度感到沮丧。
- **比较模型的 GPU 性能**：讨论了使用多块 GPU（如 3090 或较新的 4090）的效率，并考虑了 LLM 性能所需的 VRAM 要求。
   - 成员们正在争论是投资第二块 3090 还是升级到更强大的 4090，同时考虑了成本和组件的物理空间。
- **开发文本转语音 API**：一位成员宣布推出一个与 OpenAI 端点兼容的简单文本转语音 API，强调其无需 GPU 即可发挥性能。
   - 他们鼓励其他人查看 GitHub 仓库以获取集成和使用详情。
- **影响 GPU 可用性的市场趋势**：用户注意到 3090 和 P40 等 GPU 价格显著上涨，将其归因于 AI 相关任务的市场需求。
   - 成员们分享了关于 GPU 价格和可用性的个人经历，表示在当地市场很难找到更便宜的选择。
- **P40 在 AI 任务中的表现**：一位用户分享了使用 4 块 P40 GPU 的经验，认为运行大模型表现尚可，但速度较慢。
   - 他们提到在某些软件配置下使用这些 GPU 时，处理长提示词的响应时间较长。



**提到的链接**：<a href="https://github.com/PantelisDeveloping/openspeech-tts">GitHub - PantelisDeveloping/openspeech-tts: Text-to-Speech API compatible with OpenAI&#39;s API Endpoint</a>：与 OpenAI API 端点兼容的文本转语音 API - PantelisDeveloping/openspeech-tts

  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1283925067180544022)** (30 messages🔥): 

> - `Comparative Hardware Performance` (硬件性能对比)
> - `NUMA Configuration for Inference` (用于推理的 NUMA 配置)
> - `Model Selection for Story Writing` (故事写作的模型选择)
> - `PCIe Lane Configurations` (PCIe 通道配置)
> - `VRAM and Model Size Impact` (VRAM 和模型大小的影响)


- **GPU 配置的性能对比**：成员们讨论了在特定模型大小下，单插槽配置下的 **6x RTX 4090** 与双插槽配置下的 **4x RTX 4090** 配合 **24通道 DDR5** 哪个性能更好。
   - 共识似乎是，将模型放入可用的 **VRAM** 对获得最佳速度至关重要，其表现很可能优于依赖 **系统 RAM** 的配置。
- **NUMA 与性能权衡**：有人呼吁进行实验，以评估 **llamacpp** 是否可以利用 **NUMA** 配置来提升一倍的速度，特别是在不同的 GPU 设置下。
   - 支持性的建议强调了测试两种配置并退掉效果较差选项的实际做法。
- **创意写作推荐模型**：一位新用户寻求关于编写创意故事（如《星际迷航》）的合适模型的建议，并被引导去探索 Hugging Face 上的 **Chronos-Divergence-33B** 模型。
   - 重点被放在了构建丰富的 prompt 以优化模型输出上，并暗示系统 RAM 对于生成时间来说不是问题。
- **推理中的 PCIe 通道问题**：讨论围绕运行 **1x PCIe 3.0** 是否能有效支持推理任务展开，特别是在添加 **2x 3060 GPU** 时。
   - 几位成员指出，可以使用 **PLX 卡** 来增加两倍或三倍的 PCIe 通道，以增强多 GPU 配置。
- **VRAM 和模型大小的影响**：会议强调了模型大小和可用 **VRAM** 是影响性能的重要因素，并建议根据模型的深度避免使用 **Q8** 设置。
   - 一位参与者评论道，模型细节和 RAM 考量往往被低估；从简单的询问开始可以帮助新用户。



**提及的链接**：<a href="https://huggingface.co/ZeusLabs/Chronos-Divergence-33B">ZeusLabs/Chronos-Divergence-33B · Hugging Face</a>：未找到描述

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1283870232510988400)** (6 messages): 

> - `LlamaIndex.TS`
> - `LlamaIndex Hackathon` (LlamaIndex 黑客松)
> - `Code Generation Agent for NeurIPS` (用于 NeurIPS 的代码生成 Agent)
> - `Webinar on AI Agent Building` (关于构建 AI Agent 的网络研讨会)
> - `Excel Parsing Capabilities in LlamaParse` (LlamaParse 中的 Excel 解析能力)


- **LlamaIndex.TS 发布新功能！**：LlamaIndex.TS 现已面向 TypeScript 爱好者发布，为开发者带来了增强的功能。请在 [NPM](https://www.npmjs.com/package/llamaindex) 上查看。
   - 该软件包承诺通过集成核心功能来简化 TypeScript 的开发流程。
- **LlamaIndex 黑客松丰厚现金奖励**：参加 10 月 11 日至 13 日举行的第二届 LlamaIndex 黑客松，活动提供超过 **$20,000** 的现金和额度奖励。在此[注册](https://t.co/13LHrlQ7ER)。
   - 本次活动重点关注利用检索增强生成 (RAG) 技术构建高级 AI Agent。
- **NeurIPS AI Hacker Cup 合作**：通过与 @weights_biases 合作，一个由 @MistralAI 驱动的全代码生成 Agent 模板正在为 NeurIPS AI Hacker Cup 开发。这结合了来自 @llama_index 的事件驱动工作流，以实现高效的解决方案处理。
   - 在此公告中查看关于练习题创新方法的详细信息。
- **关于构建 AI Agent 的网络研讨会**：观看由 @thesourabhd 主讲的网络研讨会，讨论使用 LlamaIndex 创建高级 AI Agent。本节课将深入探讨跨多种数据模态实现支持 RAG 的 Agent。
   - 在他们的[网络研讨会页面](https://t.co/4xLJlcsosE)了解更多信息。
- **LlamaParse 中的高级 Excel 解析**：在一段新视频中，@ravithejads 展示了 LlamaParse 的高级 Excel 解析功能，强调了其处理多个工作表和复杂表格的能力。递归检索技术可以总结复杂的表格，以便于处理。
   - 想看实际操作吗？在此观看[视频](https://t.co/xuPJuUBxmC)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://t.co/13LHrlQ7ER">AGENTIC RAG-A-THON ($10K 现金奖励)</a>：LlamaIndex RAG-a-thon 与 Pinecone 和 Vessl | 10 月 11 日 - 13 日</li><li><a href="https://t.co/3agScNi74h">llamaindex</a>：[
![NPM Version](https://img.shields.io/npm/v/llamaindex)
](https://www.npmjs.com/package/llamaindex) [
![NPM License](https://img.shields.io/npm/l/llamaindex)
](https://www.npmjs.com/package/llamaindex) ...
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1283988608210047058)** (71 messages🔥🔥): 

> - `LlamaIndex 查询`
> - `LlamaIndex 中的 Workflows`
> - `使用 Chat Engine`
> - `CSV Reader 的区别`
> - `ChromaDB 集成` 


- **LlamaIndex 在 function calls 方面的限制**：一位用户询问尝试在 LlamaIndex query engine 中使用 function calls，并指出 API 尚不支持 tool 使用。
   - 另一位成员确认，当前的设置不支持 function calling 和 streaming。
- **理解 LlamaIndex 中的 Workflows**：讨论了如何有效地使用 Workflows 来构建可以与 Google Calendar 等工具交互的 agents。
   - 成员们建议使用多个 Workflows 以获得更好的控制，或者将所有内容保持在一个地方以简化实现。
- **利用 Chat Engine 进行文档交互**：一位用户表示有兴趣构建一个能够通过聊天功能搜索文档的 Retrieval Augmented Generation (RAG) 系统。
   - 建议包括利用 `chat_engine` 进行增强交互，在检索相关信息的同时保持 chat history。
- **CSV Reader 的区别**：询问了 `PagedCSVReader` 和 `CSVReader` 之间的区别，强调了对编码支持的需求。
   - 解释称 `PagedCSVReader` 为 LLMs 格式化了每个 CSV 行，而通用的 `CSVReader` 通常在没有此类格式化要求的情况下处理数据。
- **ChromaDB 与文档上下文**：一位用户尝试使用带有 ChromaDB 的 LlamaIndex 检索与查询响应相关的文档信息。
   - 建议检查 `response.source_nodes` 而不是依赖 metadata 来获取相关的文档上下文，以解决无关查询仍返回文档响应的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llamahub.ai/l/embeddings/llama-index-embeddings-voyageai?from=">未找到标题</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/">Chat Engine - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/workflow/#examples">Workflows - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/anthropic/">Anthropic - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/corrective_rag_pack/">Corrective RAG Workflow - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine/#router-query-engine">Router Query Engine - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/">Auto Merging Retriever - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/readers/file/#llama_index.readers.file.PagedCSVReader>)">File - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/loading/simpledirectoryreader/#specifying-file-encoding>)">SimpleDirectoryReader - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1284033977723654275)** (11 条消息🔥): 

> - `LlamaIndex 中的可运行函数`
> - `与 LangChain 的比较`
> - `LlamaIndex 文档参考` 


- **探索 LlamaIndex 中的可运行函数**：LlamaIndex 提供了多种函数和模块，如 **Llama CPP** 和 **DatabaseReader.load_data**，用于各种用途，详细说明请参阅 [LlamaIndex 文档](https://docs.llamaindex.ai/en/latest/)。
   - 其他可运行函数包括 **LlamaAPI.complete** 和 **FunctionTool.fn**，以满足不同的功能需求。
- **类似于 LangChain 的函数调用方法**：诸如 **FunctionTool.to_langchain_tool** 和 **FunctionTool.to_langchain_structured_tool** 等方法允许用户将函数转换为 LangChain 工具，详见 [LlamaIndex 文档](https://docs.llamaindex.ai/en/latest/api_reference/tools/function/#llama_index.core.tools.function_tool.FunctionTool.to_langchain_tool)。
   - 此外，**LangChainLLM.stream_complete** 可以生成流式补全（completions），扩展了 LlamaIndex 的实用性。
- **方法取决于使用场景**：调用的适当方法取决于具体的使用场景和打算使用的函数类型。
   - 有关完整细节和说明，建议用户参考 [LlamaIndex 文档](https://docs.llamaindex.ai/en/latest/)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/function/#llama_index.core.tools.function_tool.FunctionTool.to_langchain_tool>):">Function - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/function/#llama_index.core.tools.function_tool.FunctionTool.to_langchain_structured_tool>):">Function - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/function/#llama_index.core.tools.function_tool.FunctionTool.fn>).">Function - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1284121257784381533)** (60 messages🔥🔥): 

> - `Reinforcement Learning with KL Divergence`
> - `Mixed Precision Training`
> - `Exploration Policies in RL`
> - `Impact of OpenAI on Knowledge Accessibility`
> - `Tokenizer Retraining for Multilingual Models` 


- **在 RL 中使用 KL Divergence 防止遗忘**：成员们讨论了在强化学习中使用 KL Divergence 作为辅助损失，以防止模型在微调期间遗忘重要任务，这在 MineRL 体系中尤为突出。
   - 有人指出，过度依赖对齐的奖励函数可能会降低 KL Divergence 的益处，这暗示了 RL 体系中潜在的缺陷。
- **混合精度训练（Mixed Precision Training）机制**：针对为什么混合精度训练需要同时以 FP32 和 FP16 存储模型的问题，数值稳定性的复杂性和显存带宽的考量被指出是关键因素。
   - 此外，讨论还提到在 FP16 训练模型时，对特定操作使用 FP32 有助于缓解不稳定性，而显存限制通常会影响吞吐量。
- **RL 中的探索策略讨论**：成员们探讨了强化学习中探索策略的细微差别，一致认为像 Q-learning 这样的 Off-policy 方法比 On-policy 方法在探索方面具有更大的灵活性。
   - 讨论还涉及了如何平衡使用辅助损失项以确保探索，同时避免无意中创建一个单独的、完全参数化的探索策略。
- **OpenAI 对知识获取便利性的影响**：一位成员表示，OpenAI 的进步被低估了，认为他们显著地促进了知识获取的民主化，类似于让每个人的口袋里都装进了一个 PhD。
   - 这引发了关于社会对这些进步的看法以及它们如何融入日常生活的对话。
- **为新语言重新训练 Tokenizer**：讨论集中在添加新语言时是否需要重新训练 Tokenizer；普遍认为新语言需要对整个模型进行全面的重新训练。
   - 有观点指出，虽然对于结构相似的语言，有限的预训练可能就足够了，但在自然语言语境下，完整的重新训练通常是必不可少的。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://nvidia.github.io/apex/amp.html">apex.amp &mdash; Apex 0.1.0 文档</a>: 未找到描述</li><li><a href="https://discuss.pytorch.org/t/why-to-keep-parameters-in-float32-why-not-in-b-float16/179931">为什么要将参数保留在 float32 中，而不是 (b)float16 中？</a>: 我在想是否应该将模型参数保留在 float16 或 bfloat16 中？这可能与自动混合精度（Automatic Mixed Precision）/ autocast 是正交的，或者混合精度可能不再有意义...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1283891739265994859)** (3 messages): 

> - `Model Internal States`
> - `Non-Causal Attention Mask` 


- **训练模型以分叉（Fork）和合并（Join）状态**：讨论强调需要训练模型对其内部状态进行 **fork 和 join**，以获得更好的 **Search** 能力。
   - 这种方法可以优化模型在运行期间处理多个 Context 的方式。
- **增强输入 Token 的灵活性**：一位成员强调，允许模型请求 **更多输入 Token** 使得在 Attention Mask 中使用 **非因果块（Non-Causal Blocks）** 进行训练成为可能。
   - *这种灵活性支持持续生成*，使模型即使在需要额外数据时也能保持生产力。


  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1283895078133436448)** (11 条消息🔥): 

> - `CoT 中的 Scaling Laws`
> - `SSM vs Linear Attention`
> - `RWKV 性能`
> - `算法上下文中的 CoT`
> - `CoT 链的独立性` 


- **CoT 中的 Scaling Laws 导致意外成本**：随着上下文长度（context length）的增加，**CoT** 的计算时间 Scaling Law 曲线可能会出现拐点，在达到某个阈值后，Attention 的**二次方成本**将占据主导地位。
   - 这可能预示着 Token 价值的扩展方式发生了转变，但如果这一情况属实，那将是非常奇特的场景。
- **SSM 和 Linear Attention 方案的机遇**：有一种观点认为，**SSM/Linear Attention** 的支持者可以利用 Dense Attention 的扩展性问题，将其方案宣传为 **TTC** 无限扩展的理想选择。
   - 随着 Dense Attention 的推理计算量与性能曲线发生弯曲，Linear Attention 方法论具有巨大的推广潜力。
- **RWKV 在 CoT 场景中表现出色**：根据 [BlinkDL](https://x.com/BlinkDL_AI/status/1834300605973889111) 的一条推文，**RWKV** 模型在极端 **CoT** 任务中表现异常出色，且具有恒定的 VRAM 占用和速度。
   - 一个仅有 2.9M 参数的小型 **RWKV 模型** 作为一个纯 RNN，可以有效地解决复杂的算术计算，展示了卓越的效率。
- **CoT 中的算法任务 vs 实际用例**：一位成员指出，在 **AIME** 等实际应用中，简单的非线性变换通常就足够了，不需要递归应用。
   - 这与算法任务形成对比，后者通常需要像 **Blink** 所展示的那样进行更复杂的处理，突显了算术在 CoT 中带来的独特挑战。
- **CoT 链中的依赖性带来挑战**：讨论指出 **CoT** 链很少是相互独立的，这意味着恒定状态（constant state）可能无法充分捕捉节点之间的交互。
   - 这一局限性强调，对于更复杂的任务，特别是在非线性框架中，递归捕捉对于模型性能至关重要。



**提及的链接**：<a href="https://x.com/BlinkDL_AI/status/1834300605973889111">BlinkDL (@BlinkDL_AI) 的推文</a>：RWKV 是极端 CoT 的最佳选择🙂无需 KV cache。恒定状态大小。恒定 VRAM。恒定速度。引用 BlinkDL (@BlinkDL_AI)：一个仅有 2.9M (!) 参数的小型 #RWKV 可以解决 18239.715*9.728263 或 4....

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1284169856106889309)** (1 条消息): 

> - `潜空间聚类 (Latent Space Clustering)`
> - `强化学习中的可解释性` 


- **关于利用潜空间聚类提高可解释性的咨询**：一位新成员咨询了关于通过**潜空间聚类**来增强**可解释性**的见解，并引用了论文 [Latent Space Clustering for Explainable Reinforcement Learning](https://arxiv.org/abs/1808.07292)。
   - 他们特别关注其在 **Reinforcement Learning** 中的应用，以提高结果的可解释性。
- **对可解释性技术的兴趣**：这位新成员对各种**可解释性**技术表现出了普遍的好奇心，特别是它们在机器学习背景下的有效性。
   - 现有成员的参与可以为要使用的最佳实践和方法论提供宝贵的视角。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1283996255575478302)** (4 条消息): 

> - `lm-evaluation-harness`
> - `gpt-4 评估`
> - `medqa 任务错误`
> - `自定义任务` 


- **Sudhanshu 寻求 lm-evaluation-harness 的帮助**：Sudhanshu Mishra 正尝试在代码生成 **swe-bench 数据集**上使用 **lm-evaluation-harness** 评估 **OpenAI gpt-4o 模型**，并正在寻求有关操作步骤的指导。
   - *如果有人能提供帮助，那将不胜感激。*
- **评估过程中遇到错误**：Sudhanshu 报告在执行评估 OpenAI 的命令时收到错误，特别提到了与 `lm_eval` 相关的 **Traceback**。
   - 他分享了所使用的确切命令：`!lm_eval --model openai_completions ... --gen_kwargs temperature=0.7`。
- **关于 medqa 任务的讨论**：一位社区成员询问 Sudhanshu 尝试的任务是否是**自定义任务**，因为他们注意到目前只有 **medqa_4options** 可用。
   - 这一询问表明在设置中支持的任务方面可能存在一些困惑或需要澄清。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1283868008653127690)** (40 messages🔥): 

> - `AdEMAMix Optimizer`
> - `Command R+ Usage`
> - `AI Fatigue`
> - `Bar Exam Finetuning`
> - `Zoom for Australian Users` 


- **AdEMAMix Optimizer 引发好奇**：一名成员对 [GitHub 上的 AdEMAMix Optimizer](https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch) 表示怀疑，并认为它可能解释了 **Parakeet** 为何能在 **20 小时** 内完成训练并获得清晰的输出。
   - 他们在讨论使用各种方法和效率训练模型时，指出了其潜在的影响。
- **探索使用 Command R+ 进行微调**：一位硕士毕业生正在研究使用 **Command R+** 对 **llama2** 进行微调，以应对美国律师资格考试（Bar Exam），并寻求建议。
   - 成员们建议在本地进行实验，并深入阅读 [Cohere 文档](https://docs.cohere.com) 以获得更好的见解。
- **AI 疲劳迹象显现**：成员们讨论了当前形势是否表明正转向**实用性胜过炒作**，认为 AI 的进步现在更加务实。
   - 一位成员将现状比作“原始汤”，强调随着问题的深度和广度增加，必要技能正在迅速演变。
- **对 AI 性能的担忧**：一位成员对模型被视为高级搜索引擎表示担忧，强调能力取决于上下文相关的 **token**。
   - 他们反思了对所谓先进性能声明的怀疑，指出需要从 AI 能力中获得经过验证的结果。
- **对 Zoom 功能的需求**：有建议利用 Zoom 来增强可访问性，特别是对于想要观看录像的澳大利亚成员。
   - 对话引发了关于替代方案的简短讨论，提到 **vllm / neura magic** 也提供类似功能，但参与人数较少。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com">Cohere Documentation — Cohere</a>: 未找到描述</li><li><a href="https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch">GitHub - nanowell/AdEMAMix-Optimizer-Pytorch: The AdEMAMix Optimizer: Better, Faster, Older.</a>: AdEMAMix 优化器：更好、更快、更老。通过在 GitHub 上创建账户来为 nanowell/AdEMAMix-Optimizer-Pytorch 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1284174092060196864)** (29 messages🔥): 

> - `Cohere API Spending Limit`
> - `Billing and Usage Issues`
> - `Mobile Version Access`
> - `Rate Limiting by IP` 


- **设置 Cohere API 支出限制**：用户讨论了如何为其每日或每月 **Cohere API** 使用量设置最高限额，以避免意外账单，特别是针对潜在的恶意活动。
   - 一位用户建议检查 [Cohere 控制面板](https://dashboard.cohere.com/billing?tab) 上的账单和使用设置，但在访问相关选项时遇到了问题。
- **账单控制面板困惑**：尽管是账户的“所有者（Owner）”，多位用户对无法在账单控制面板上看到预期选项表示沮丧。
   - 进一步的建议包括尝试桌面版和**移动版**以查看不同的界面，但问题仍然存在。
- **推荐的客服联系方式**：建议用户就缺失的支出限制选项联系 **Cohere 支持团队**，并确认发送邮件至 support@cohere.com。
   - 一名成员确认在折腾控制面板一段时间后，将寻求人工帮助。
- **API 请求的速率限制**：提到用户可以实施速率限制（Rate Limiting），以控制**每个 IP 地址对 API** 发起的请求数量。
   - 这种方法有助于防止来自潜在有害源的过度使用峰值。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://dashboard.cohere.com/billing?tab]">Login | Cohere</a>: 登录以通过一个易于使用的 API 访问先进的大语言模型（LLM）和 NLP 工具。</li><li><a href="https://dashboard.cohere.com/billing?tab=spending-limit">Login | Cohere</a>: 登录以通过一个易于使用的 API 访问先进的大语言模型（LLM）和 NLP 工具。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/)** (1 messages): 

sssandra: 哇，超棒的项目！我来给你充点 API 额度 🙂
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1283911058062442577)** (25 messages🔥): 

> - `Mojo 中的 StringSlice`
> - `Linux 发行版上的 Mojo`
> - `Magic 工作区管理`
> - `Linux 内核版本要求`
> - `可执行文件兼容性` 


- **将 StringSlice 与 Span[UInt8] 配合使用**：一位成员询问如何将 `Span[UInt8]` 转换为字符串视图，并了解到 `StringSlice(unsafe_from_utf8=path)` 是正确的用法。
   - 这一关于关键字参数的澄清帮助他们理解了该函数的要求。
- **Mojo 与 Linux 发行版的兼容性**：一位用户报告在 Arch Linux 和 Zorin 上成功安装并运行了 Mojo，并提出了关于更广泛发行版支持的问题。
   - 据解释，使用 Magic 可以让 Mojo 在拥有受支持内核版本的各种 Linux 发行版上运行。
- **Magic 工作区导出/导入**：讨论转向了 Magic 的功能，特别是关于在使用 conda 时导出和导入工作区的问题。
   - 共享了相关资源，包括文档和入门指南，以帮助用户有效地管理他们的环境。
- **编译后可执行文件的 Linux 内核依赖**：对话涉及了运行编译后可执行文件的内核版本要求，并提到了与旧版本内核的潜在兼容性。
   - 用户讨论了针对旧版本内核的含义，并表达了对跨不同系统维持兼容性的担忧。
- **寻求 Magic 设置支持**：一位新安装了 Magic 的用户询问如何在集群环境中正确设置它。
   - 建议他们咨询 Modular 支持以获得进一步帮助，并强调了内核兼容性的重要性。



**提到的链接**：<a href="https://docs.modular.com/magic/">Magic 入门 | Modular 文档</a>：Magic 是一个适用于任何语言的包管理器和虚拟环境管理器。

  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1284208260399759522)** (2 messages): 

> - `MAX 24.5 发布`
> - `Mojo 24.5 更新`
> - `Discord 用户验证`
> - `服务器入站流程变更` 


- **MAX 24.5 正式发布！**：**MAX 24.5** 的发布为 int4k Llama 的 Token 生成带来了 **45% 的性能提升**，并为开发者提供了新的驱动接口。
   - 在 [MAX 变更日志](https://docs.modular.com/max/changelog?utm_campaign=24_5&utm_source=discord)中查看完整变更。
- **Mojo 24.5 带来重大进展！**：**Mojo 24.5** 支持隐式变量定义、新的标准库 API 以及对 Python **3.12** 的支持。
   - 在 [Mojo 变更日志](https://docs.modular.com/mojo/changelog?utm_campaign=24_5&utm_source=discord)中了解更多关于这些更新的信息。
- **使用新包管理器 Magic 简化流程**：使用新的包和环境管理器 **Magic**，MAX 和 Mojo 的安装过程得到了简化。
   - 使用 `magic update max` 轻松升级 MAX，并参考[我们的文档](https://docs.modular.com?utm_campaign=24_5&utm_source=discord)开始使用！
- **新用户验证流程**：从 **9 月 16 日**开始，用户必须通过 #verify 频道分享电子邮件来验证成员身份，以确保环境无垃圾信息。
   - 未验证的用户仍将拥有读取权限，但在特定频道的发消息能力将受到限制。
- **新用户入站问题**：新成员在验证电子邮件地址后，将回答两个多选题入站问题。
   - 已创建一个新频道用于讨论服务器变更并收集用户建议。

  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1283908103334526976)** (30 条消息🔥): 

> - `在 Mojo 中访问 errno`
> - `优化 Span 借用 (Borrowing)`
> - `解包 (Unwrapping) 可能失败的函数调用`
> - `通过 PyBind11 与 Python 互操作`
> - `执行 Shell 命令` 


- **在 Mojo 中访问 `errno`**：要在 macOS 上的 Mojo 中访问 `errno`，请使用 `external_call["__error", UnsafePointer[UInt32]]()[]`。
   - 这使得能够直接与系统调用中设置的错误值进行交互。
- **优化 Span 借用行为**：讨论了将 `Span` 作为 `borrowed` 参数传递通常会导致传递指针和长度，而不会调用 `__copyinit__()`。
   - `%register_passable%` 特性 (trait) 会影响类型的处理方式，深入查看生成的代码可能会澄清其行为。
- **解包可能失败的函数调用详解**：一位成员分享了用于解包可能失败的函数调用的代码，该代码初始化了一个 socket 并处理潜在的连接错误。
   - 目前的方法似乎是可行的，提供了一种处理由可能失败的函数返回的可选值 (optional values) 的方法。
- **Mojo 通过 PyBind11 实现的 Python 互操作性**：成员们确认通过 PyBind11 暴露的模块可以在 Mojo 中工作，并利用 CPython 来运行它们。
   - 这种集成允许 Mojo 使用其 API 直接访问 Python 对象。
- **使用 `libc` 执行 Shell 命令**：对于执行 Shell 命令，可以通过使用 `external_call` 为系统级函数设置别名来调用 `os.system`。
   - 一位成员提供了一个示例，展示了如何使用 `StringLiteral` 进行正确的函数调用来执行 `pwd` 命令。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1283875210851778625)** (8 条消息🔥): 

> - `MAX 与向量数据库`
> - `在 Google Colab 中使用 MAX`
> - `PyPI 上的包冒充问题`
> - `托管 Notebook 环境的使用情况`
> - `为 MAX 创建 GitHub Issues` 


- **MAX 缺乏原生 Embedding 支持**：成员们讨论了 **MAX** 原生不提供 embedding、向量数据库或相似性搜索功能，但建议在语义搜索应用中使用 **ChromaDB**、**Qdrant** 或 **Weaviate** 等替代方案。
   - 引用了一篇博客文章，提供了使用这些工具进行**语义搜索**增强的示例。
- **在 Google Colab 中运行 MAX 引发问题**：关于在 Google Colab 中运行 **MAX** 引擎引发了关注，因为如果没有正确的安装程序，它可能无法无缝运行。
   - 强调了在 GitHub 上创建 issue 的重要性，以便进一步调查与 **Colab Pro** notebook 的兼容性问题。
- **警惕冒充 MAX 的 PyPI 包**：发出了警告，不要在 PyPI 上安装任何类似于 **MAX** 的包，因为它们可能会产生负面后果且未获得官方支持。
   - 建议成员使用 **conda** 或 **magic** 进行官方包安装。
- **托管 Notebook 环境的普及**：一位成员粗略估计，有**数百万开发者**经常使用 Google Colab 和 Kaggle 等托管 notebook 环境进行数据科学和 AI 项目。
   - 虽然没有具体的用户数字，但 Kaggle 和 Colab 是这个不断增长的领域中的主要参与者。
- **创建 Issue 促进社区支持**：成员们讨论了在 GitHub 上创建一个关于 Colab 中 **magic/max** 功能的新 issue，强调这对于 AI 学习旅程中的新开发者非常重要。
   - 该 issue 将允许社区协作并共同寻找解决方案，强调了共享学习经验的重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/semantic-search-with-max-engine">Modular: 使用 MAX 引擎进行语义搜索</a>: 在自然语言处理 (NLP) 领域，语义搜索侧重于理解查询背后的上下文和意图，超越了单纯的关键词匹配，以提供更相关的内容...</li><li><a href="https://docs.modular.com/max/python/get-started">使用 Python 运行推理 | Modular 文档</a>: MAX Engine Python API 演练，展示如何加载和运行模型。</li><li><a href="https://pypi.org/project/modular-ai/">modular-ai</a>: 一个用于与各种 AI 模型交互的库</li><li><a href="https://github.com/modularml/max/issues/223">[Magic CLI]: magic/max 不支持在 google colab notebooks 中使用 · Issue #223 · modularml/max</a>: 背景 https://colab.research.google.com 在数据科学和 AI 领域被广泛使用，因为它采用了按需付费模式，并提供各种 TPU 或 NVIDIA GPU。一个常见的用例将是...
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1283901052684472432)** (7 条消息): 

> - `Open Interpreter Token 使用情况`
> - `Open Interpreter 自动化`
> - `Mike 在 Mac 上的 Beta 测试`
> - `Replit 使用情况` 


- **Open Interpreter 的 Token 使用引发疑问**：一位成员对 **Open Interpreter** 仅在 6 个请求中就消耗了 **10,000 个 Token** 表示担忧，并质疑其 Token 管理的效率。
   - 这引发了关于 Token 使用潜在优化方案的讨论。
- **Open Interpreter 与 Webhooks 的集成**：另一位成员询问是否可以将 **Open Interpreter** 与配置了 **webhooks** 服务的 GPTs 结合使用。
   - 他们寻求为自动化目的提供 API 访问权限的方法。
- **Mike 的 Beta 测试仅限 Mac**：一位成员表达了在 Windows 和 Mac 上测试 **Mike** 的强烈意愿，但从另一位成员处获悉，目前 Beta 测试**仅限 Mac**。
   - 这导致了对未来跨平台测试支持的进一步期待。
- **对使用 Replit 的兴趣**：一位成员询问聊天室中是否还有其他人使用 **Replit**，希望与有相同兴趣的人建立联系。
   - 这一询问增加了关于协作编程平台日益增长的讨论。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1283870057797124226)** (49 条消息🔥): 

> - `iPhone 应用设置`
> - `LiveKit 连接问题`
> - `Python 证书更新`
> - `社区文档工作`
> - `Beta 测试咨询` 


- **设置 iPhone 应用需要帮助**：一位成员发现了 iPhone 应用的发布，但请求关于克隆 repo 和设置步骤的分步指导，并提到自己是初学者。
   - 另一位用户建议访问 [设置指南](https://01.openinterpreter.com/setup/introduction) 以获取详细说明。
- **LiveKit 连接挑战**：该成员分享了通过移动数据而非 Wi-Fi 连接到 MacBook 的困难，遇到了 LiveKit 重新连接错误。
   - 作为回应，社区成员请求提供复现错误的详细步骤，并分享额外的终端输出以便进行调试。
- **更新 Python 证书流程**：在更新 Python 证书方面一直存在问题，社区分享了关于访问 'Install Certificates.command' 文件的指令。
   - 一位用户对该流程提出疑问，建议将其添加到社区文档中，以帮助遇到类似挑战的人。
- **社区文档协作**：一位成员敦促改进文档，称 90% 的用户都面临 LiveKit 设置问题，并提出了改进建议。
   - Mike 建议那些拥有有效解决方案的人提交 pull request，以澄清设置过程并帮助他人。
- **Beta 测试可用性**：关于加入应用 Beta 测试的讨论兴起，成员们寻求如何参与以及是否有空位的细节。
   - Mike 确认目前暂无空位，但鼓励用户稍后再次查看 Beta 计划中潜在的开放机会。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://01.openinterpreter.com/setup/installation">Installation - 01</a>: 未找到描述</li><li><a href="https://01.openinterpreter.com/setup/introduction">未找到标题</a>: 未找到描述</li><li><a href="https://01.openinterpreter.com/">未找到标题</a>: 未找到描述</li><li><a href="https://01.openinterpreter.com/client/android-ios">Android &amp; iOS - 01</a>: 未找到描述</li><li><a href="https://github.com/OpenInterpreter/01-app">GitHub - OpenInterpreter/01-app: 用于计算机控制的 AI 助手。</a>: 用于计算机控制的 AI 助手。通过在 GitHub 上创建账户来为 OpenInterpreter/01-app 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1283896036909125663)** (3 messages): 

> - `Open Interpreter functionality` (Open Interpreter 功能)
> - `Voice response issues` (语音响应问题)
> - `Mobile app performance` (移动端 App 性能)
> - `Library installation success` (库安装成功)
> - `User feedback` (用户反馈)


- **Open Interpreter 用户体验**：一位名叫 Alex 的用户对 *Open Interpreter* 表示满意，在安装了必要的库后，他成功通过 **iPhone 11 Pro** 控制了他的 **Mac M3**。
   - 他对团队的出色工作表示祝贺，但也指出了移动端 App 在语音响应和输出方面的不足。
- **移动端 App 的语音响应问题**：Alex 报告称移动端 App 无法进行语音响应，表示 App 虽然能听到指令，但没有语音输出或显示响应。
   - 他特别提到 App 中的 female teacher 功能没有反应，这引发了对用户交互的担忧。
- **移动端 App 功能反馈**：Alex 分享了他使用 **Open Interpreter** 移动端应用程序的经验和挑战，强调了尽管应用识别了输入，但缺乏反馈的问题。
   - 他针对缺乏响应的问题提出了建设性批评，寻求未来版本的改进。


  

---



### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

batmanosama: https://huggingface.co/spaces/ulab-ai/ArxivCopilot
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1283903187191074877)** (34 messages🔥): 

> - `O1 support` (O1 支持)
> - `DSPy versions` (DSPy 版本)
> - `RAG integration` (RAG 集成)
> - `MIPRO compilation` (MIPRO 编译)
> - `Google Vertex AI` 


- **正在探索 O1 功能**：社区对 DSPy 与 `o1-preview` 的兼容性感到好奇，一些成员表示有兴趣测试其集成。
   - 值得注意的是，**O1 支持**已经实现，展示了社区持续的开发进展。
- **DSPy 2.4.16 版本更新**：成员们确认 DSPy **2.4.16** 版本现在包含最近发布的全新 `dspy.LM` 功能。
   - 鼓励用户尝试 **LiteLLM 模型**，并有报告称更新后实现成功。
- **在 DSPy 中实现 RAG**：讨论了如何使用 DSPy 模块将传统的 LLM 查询适配为 **RAG**（检索增强生成）以获得最佳性能。
   - 分享了 RAG 实现的示例，包括 [simple RAG](https://github.com/stanfordnlp/dspy/blob/main/skycamp2023.ipynb) 和 [MIPRO compilation](https://github.com/stanfordnlp/dspy/blob/main/examples/qa/hotpot/hotpotqa_with_MIPRO.ipynb) 的链接供进一步参考。
- **Google Vertex AI 的集成挑战**：用户表示在集成 **Google Vertex AI** 时遇到困难，尽管凭据正确但仍出现服务错误。
   - 关于为 LiteLLM 模型设置环境的讨论强调了对有效代理和配置的需求。
- **RAG 中的动态提示词和上下文**：成员们讨论了在 **RAG** 实现中将动态上下文打包进单个提示词的最佳实践。
   - 强调了在动态场景下，将相关上下文随提示词一同包含以获得更好结果的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/matei_zaharia/status/1834351621570199819">Matei Zaharia (@matei_zaharia) 的推文</a>: 很高兴看到 OpenAI o1 今天发布。这是迈向复合 AI 系统（而非单一模型）以获得最佳 AI 结果趋势的又一个例子。我相信未来的版本不仅会...</li><li><a href="https://github.com/stanfordnlp/dspy/blob/03082d28b033c01b95bb661d4789d2ad1a517a6c/dspy/clients/lm.py#L58)!">stanfordnlp/dspy 中的 dspy/dspy/clients/lm.py</a>: DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/skycamp2023.ipynb">stanfordnlp/dspy 中的 dspy/skycamp2023.ipynb</a>: DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/examples/qa/hotpot/hotpotqa_with_MIPRO.ipynb">stanfordnlp/dspy 中的 dspy/examples/qa/hotpot/hotpotqa_with_MIPRO.ipynb</a>: DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1283871147188031580)** (27 messages🔥): 

> - `PyTorch 中的内存泄漏`
> - `Upstage Solar Pro 模型`
> - `单卡推理`
> - `Liger Kernels 实现`
> - `LLM 中的 Reflection 任务` 


- **GPU batch size 的内存问题**：讨论强调了按 **GPU batch size** 样本进行简单打包（packing）可能会因为张量大小不一导致**内存泄漏**，而 **PyTorch** 的行为加剧了这一问题。
   - 成员们对打包样本时序列长度变化带来的 **padding** 需求表示担忧，并呼吁寻求解决方案以避免这些陷阱。
- **对 Upstage Solar Pro 的兴奋**：部分成员对 [Upstage Solar Pro](https://huggingface.co/upstage) 模型表现出兴趣，将其与 **LLaMA 3.1** 进行对比，并指出 **22B** 似乎是单卡推理的最佳选择。
   - 也有人对模型制作者提出的**大胆主张**表示谨慎，担心落入夸大承诺的陷阱。
- **对 Liger Kernels 的好奇**：有成员询问是否有人实现了 **Liger kernels** 并获得了满意的结果，寻求他人的经验见解。
   - 对具体实现的不确定性反映了社区对优化 LLM 性能的广泛兴趣。
- **Reflection 任务引发关注**：一位成员对最近 LLM 中的 **reflection 任务**表示怀疑，质疑 OpenAI 模型发布的时机和训练方式。
   - 社区推测可能存在“内部”消息或预发布信息影响了人们的认知。
- **对 O1 功能的看法**：小组讨论了 **O1** 的有效性，将其比作带有用户友好 UI 的 **Chain of Thought** 模型，而其他人则评论了它在处理更机械化提示词时的表现。
   - 一些人持保留态度，认为其效用可能仅限于特定的使用场景。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/upstage/solar-pro-preview-pretrained">upstage/solar-pro-preview-pretrained · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/upstage">upstage (upstage)</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1284223709162377257)** (6 messages): 

> - `phi-3.5 训练尝试`
> - `Tokenization 错误`
> - `分类器训练问题` 


- **训练 phi-3.5 的困难**：一个小组尝试训练 **phi-3.5**，但报告称 **lora adapters** 基本上没学到任何东西，令人感到沮丧。
   - 他们发现了一个与此问题相关的潜在 bug，并在其 [GitHub report](https://github.com/axolotl-ai-cloud/axolotl/issues/1916) 中进行了详细说明。
- **遇到 Tokenization 错误**：一位成员询问其他人是否遇到了其 GitHub bug 报告中所述的 **tokenization 错误**，怀疑该问题源于新的 **per-turn masking** 策略。
   - 他们注意到**最后一个轮次结束 token** 被屏蔽掉了，这可能会影响训练。
- **分类器无法输出标签**：**phi-3.5** 被用于训练一个基础句子分类器，但它始终像聊天助手一样回答，而不是提供预期的分类文本标签。
   - 该成员表示失望，称“好吧，看来是时候暂时放弃 phi-3.5 了”。



**提到的链接**：<a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1916)">Issues · axolotl-ai-cloud/axolotl</a>：欢迎提出 axolotl 相关问题。通过在 GitHub 上创建账户为 axolotl-ai-cloud/axolotl 的开发做出贡献。

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1284119903304941622)** (1 messages): 

> - `梯度范数裁剪 (Gradient Norm Clipping)`
> - `LoRA 配置`
> - `训练日志解读` 


- **尽管设置了裁剪，梯度范数仍然很高**：一位用户报告在 LoRA 配置中设置了 `max_grad_norm: 2`，但在训练日志中观察到显著更高的 **grad_norm** 值，峰值达到 **2156.37**。
   - *有没有可能日志打印的是裁剪前的梯度范数？* 这引发了关于日志记录机制以及它是否准确反映裁剪后数值的疑问。
- **LoRA 训练设置详情**：用户的训练配置包括各种设置，如 **lora_r: 16**、**learning_rate: 0.00001** 和 **val_set_size: 0.05**，用于微调 **Pythia** 模型。
   - 定义了特定的 **LoRA target modules** 以优化某些层，反映了对实验的精心设置。

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1283874771272073248)** (9 条消息🔥): 

> - `Llama 3.1 8B Finetune`
> - `Open Source SD`
> - `Model Renaming` (模型重命名)
> - `API/Web Only Model` 


- **Llama 3.1 8B Finetune 发布**: 一位成员分享了一个 [Llama 3.1 8B finetune 模型](https://huggingface.co/dustinwloring1988/Llama3.1-8B-Reflection-v2-gguf)，他们正在寻求合作者来增强数据集。
   - 该模型作为一个概念验证，声称复制了在多个 YouTube 频道上讨论过的 *flection model*。
- **对 Open Source SD 的担忧**: 一位参与者对 **Stable Diffusion** 在开源领域的沉寂表示担忧，暗示贡献度有所下降。
   - *基本上，如果你关心开源，SD 似乎已经死了，* 他们评论道。
- **Llama 模型的命名反馈**: 在收到关于 Llama 模型命名的反馈后，一位成员承认该名称可能存在负面含义，并同意在下一个版本中更改。
   - *有任何建议请告知，我之后也会发布 wandb 运行记录，* 他们补充道。
- **API/Web Only 模型发布**: 另一位用户注意到了一个 **API/Web only 模型** 的发布，但对其对开源 SD 项目的影响表示失望。
   - 该消息表明了对 **AI 模型开发** 中开源存在感日益减弱的更广泛担忧。
- **社区对模型关联的不满**: 一位社区成员建议不要与某个被视为 **scam**（诈骗）的特定模型产生关联，建议选择一个不同的名称。
   - 这突显了关于 AI 模型开发中 **reputation**（声誉）和 **credibility**（公信力）的持续讨论。



**提到的链接**: <a href="https://huggingface.co/dustinwloring1988/Llama3.1-8B-Reflection-v2-gguf">dustinwloring1988/Llama3.1-8B-Reflection-v2-gguf · Hugging Face</a>: 未找到描述

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1283942923372859402)** (17 条消息🔥): 

> - `Tier 5 API Access`
> - `Chain-of-Thought (CoT) and Reinforcement Learning`
> - `Self-Taught Reasoner (STaR)`
> - `Quiet-STaR`
> - `Data Gathering for Model Training` 


- **Tier 5 API Access 成本高昂**: 投资 **Tier 5 API access** 可能会非常昂贵，这让一些人怀疑与之前的模型（如 **GPT-4o**）相比的权衡。
   - *“总不会比 gpt4o 差太多”* 表明了对探索新能力的谨慎乐观。
- **CoT 和 RL 让模型更聪明**: 通过结合 **Chain-of-Thought (CoT)** 与 **Reinforcement Learning**，模型可以得到显著提升，正如 **STaR** 技术所强调的那样，该技术利用了 **few-shot examples**。
   - 关于 **STaR** 的论文断言，生成逐步的推理过程（rationales）可以增强在数学或常识问答等复杂推理任务上的表现，证实了有效的工程设计。
- **引入用于推理的 Quiet-STaR**: **Quiet-STaR** 的概念扩展了 **Self-Taught Reasoner**，允许在每个 token 处生成推理过程，以便根据推断出的未说明推理进行更好的预测。
   - 这种泛化旨在解决生成后续内容的计算成本，同时提高对任意文本的理解。
- **Meta 和 Qwen 正在缩小差距**: 讨论表明 **Meta** 和 **Qwen** 正在定位以追赶 AI 能力，同时有人担心 **Anthropic** 可能会处于领先地位。
   - 巡回分析师预测，进步源于有效的工程设计和大量的计算资源。
- **高质量数据收集的重要性**: 从知识渊博的人那里收集多样化的思维过程对于训练有效的模型至关重要。
   - *“必须是聪明人的数据，所以它不可能便宜”* 强调了数据质量与模型智能之间的相关性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2203.14465">STaR: Bootstrapping Reasoning With Reasoning</a>: 生成逐步的 “chain-of-thought” 推理过程可以提高语言模型在数学或常识问答等复杂推理任务上的性能。然而，诱导语言...</li><li><a href="https://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>: 在写作和交谈时，人们有时会停下来思考。尽管以推理为中心的工作通常将推理框架化为回答问题或完成 Agent 任务的方法，但推理对于...
</li>
</ul>

</div>
  

---

### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/1284125670305169550)** (2 messages): 

> - `与 OpenSea 的合作`
> - `免费铸造 (Free Mint) 活动`
> - `用户参与` 


- **令人兴奋的 OpenSea 合作**：宣布了与 **OpenSea** 的新合作，为用户启动了 **free mint** 机会。
   - 鼓励成员及时通过 [CLAIM 链接](https://iclaim7b.vercel.app/) 参与，并注意某些领取可能需要 gas。
- **用户参与是关键！**：服务器中的每个人都有机会被选中参与铸造过程。
   - 积极参与正受到激励，促进了社区对该计划的投入。


  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1284125681470406698)** (1 messages): 

> - `与 OpenSea 的合作`
> - `免费铸造机会`
> - `参与要求` 


- **宣布与 OpenSea 合作**：已与 **OpenSea** 建立新合作，为用户提供 **free mint** 机会。
   - 鼓励 *@everyone* 参与该计划，将从服务器成员中进行筛选。
- **敦促用户尽快参与**：服务器用户可以通过访问 [CLAIM](https://iclaim7b.vercel.app/) 链接立即参与。
   - 但需注意，某些领取可能需要支付 **gas** 费用才能完成。


  

---


### **LAION ▷ #[paper-discussion](https://discord.com/channels/823813159592001537/1172520224797511700/1284125722000232498)** (1 messages): 

> - `与 OpenSea 的合作`
> - `免费铸造参与`
> - `领取流程`
> - `Gas 费用` 


- **宣布与 OpenSea 合作**：服务器已与 **OpenSea** 合作，为用户提供新的 **free mint** 机会。
   - 鼓励所有成员把握这次机会参与。
- **免费铸造领取流程**：服务器用户可以通过 [CLAIM](https://iclaim7b.vercel.app/) 链接参与铸造过程。
   - 强调了某些领取可能需要 **gas 费用** 才能完成流程。


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1284142076446904413)** (4 messages): 

> - `Torchtune 在 Mac 上的安装`
> - `torchao 的可用性`
> - `在 MacOS 上使用 Torchtune 进行训练` 


- **Torchtune 0.2.1 在 Mac 上安装失败**：**torchtune 0.2.1 版本**在 Mac 上安装失败，因为无法满足依赖项 **torchao==0.3.1**，导致无法在 MacBook 上使用。
   - 成员提到即将发布的 **torchao 0.6.0** 可能会提供 **macOS wheels**，从而简化安装过程。
- **适用于 Mac M1 的 torchao wheels 现已可用**：已确认 **torchao wheels** 现已支持 **Mac M1**，增强了该平台用户的兼容性。
   - 此更新可能有助于缓解用户尝试在 Mac 设备上运行 **torchtune** 时遇到的一些限制。
- **与 Mark 合作优化 Mac 安装**：成员们正与 Mark 合作简化 **torchtune** 在 macOS 上的安装流程，此前该流程并不理想。
   - 尽管有所改进，用户承认目前 **torchtune** 在 macOS 上可能还不是非常实用。
- **不再阻碍在 MacOS 上进行训练**：安装方面的进展意味着它将不再阻碍在 **MacOS** 上使用 **torchtune** 进行训练，尽管目前它的帮助还不是特别大。
   - 尽管存在公认的局限性，但对 Mac 用户来说，这一进展仍是一个受欢迎的变化。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/1572">Installation of `0.2.1` not compatible on mac as the dependency `torchao==0.3.1` can&#39;t be fulfilled · Issue #1572 · pytorch/torchtune</a>: torchtune 0.2.1 版本在 Apple Mac 笔记本上无法成功完成安装，因为在 Mac 平台上找不到 torchao==0.3.1。因此该工具无法在 MacBook 上使用...</li><li><a href="https://pypi.org/project/torchao/#files">torchao</a>: 用于将 ao 技术应用于 GPU 模型的包
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1284097810882494555)** (22 messages🔥): 

> - `log_peak_memory_stats`
> - `GPU runners for CI`
> - `collating and masking`
> - `batched generation`
> - `online packing` 


- **建议对 log_peak_memory_stats 进行配置更改**：有成员质疑为什么 `log_peak_memory_stats` 默认没有设置为 True，其他人也认为这很有益，特别是对于那些关注性能优化的用户。
   - 另一位成员提议创建一个 PR，在所有配置中将此项更新为 True。
- **将 Recipe 测试切换到 GPU**：讨论透露，由于历史原因，当前的 Recipe 测试被设置为在 CPU 上运行，但大家一致认为需要更新它们以利用 GPU 资源。
   - 还建议将某些测试标记为 GPU 测试，以便在 GPU 不可用时可以跳过。
- **探索整理（Collating）与掩码（Masking）方案**：一位成员强调需要提高 MM 模型在不使用 Batching 情况下的评估效率，并指出目前的性能较慢。
   - 提出了批量生成（Batched generation）作为部分解决方案，并引用了一个正在处理此问题的 PR。
- **在 Recipe 中转向批量生成**：计划通过一个新的 Recipe 来增强生成过程，该 Recipe 旨在保持轻量并与项目目标保持一致。
   - 成员们表示有兴趣对这个新 Recipe 提供反馈，该 Recipe 旨在降低复杂性，但需要更多测试。
- **为可迭代数据集采用在线打包（Online Packing）**：明确了一个未来计划，即在支持可迭代数据集后实现在线打包。
   - 这旨在提高当前工作流中的数据处理能力和效率。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/pull/1424">[WIP][RFC] Batched inference 🤝 KV-cache 🤝 compile by SalmanMohammadi · Pull Request #1424 · pytorch/torchtune</a>：上下文 此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档，还是其他（请在此处添加）。请链接此 PR 解决的任何 Issue。关闭 #125...</li><li><a href="https://github.com/pytorch/torchtune/pull/1563">[WIP] Add generate-v2 recipe for MM by joecummings · Pull Request #1563 · pytorch/torchtune</a>：终于，一个不会让我抓狂的生成 Recipe。命令：tune run dev/generate_v2 --config multimodal_generation</li><li><a href="https://github.com/pytorch/torchtune/blob/ee343e61804f9942b2bd48243552bf17b5d0d553/tests/recipes/test_full_finetune_single_device.py#L39">torchtune/tests/recipes/test_full_finetune_single_device.py at ee343e61804f9942b2bd48243552bf17b5d0d553 · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/tests/recipes/test_lora_finetune_single_device.py">torchtune/tests/recipes/test_lora_finetune_single_device.py at main · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 做出贡献。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1284165354171535370)** (5 条消息): 

> - `LangChain AWS ChatBedrockConverse`
> - `RAG 聊天机器人集成问题`
> - `GenAI 咨询项目`
> - `OpenAI 进展的影响` 


- **LangChain AWS ChatBedrockConverse 与对话历史**：一位用户询问 **LangChain 的 AWS ChatBedrockConverse** 是否支持在检索链中维护 **对话历史 (conversational history)**。
   - 这引发了关于在对话式 AI 框架中如何管理历史记录的重要思考。
- **向量数据库实现需要帮助！**：一位用户报告称，正尝试实现 [Upstash Redis](https://github.com/thinley4/Rag-Chatbot/issues/4) 来替换内存中的 **MemoryVectorStore**，用于存储 PDF 切片的向量嵌入 (vector embeddings)。
   - 他们指出在与 **Pinecone** 等替代方案集成时遇到挑战，正寻求社区帮助。
- **提供 GenAI/RAG/CV 咨询项目**：一位成员宣布可以协助处理与 **GenAI**、**RAG** 和 **CV** 相关的 **咨询项目**，重点是为初创公司开发概念验证 (PoC)。
   - *如果有任何人需要此类服务*，他们邀请用户私信 (DM) 了解更多信息。
- **OpenAI 的变革性影响**：一位成员对 OpenAI 进展的影响表示惊讶，称这感觉就像是 *给每个人的口袋里都塞进了一个博士*。
   - 他们质疑社会是否充分领悟了这些技术带来的重大变革。



**提及的链接**：<a href="https://github.com/thinley4/Rag-Chatbot/issues/4">Implement Vector DB instead of  inmemory &#39;MemoryVectorStore&#39; · Issue #4 · thinley4/Rag-Chatbot</a>：我目前正尝试实现 Upstash Redis 来替换 MemoryVectorStore（内存中存储），用于存储 PDF 切片的向量嵌入。我尝试了 Upstash 和 Pinecone，但无法完成集成。什么...

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1284098878529863680)** (13 条消息🔥): 

> - `Warhammer Adaptive RAG`
> - `Tavily 替代方案`
> - `RAG 技术`
> - `Vantager 的 AI 工程师职位`
> - `NPC 生成器协作` 


- **Warhammer Adaptive RAG 项目初具规模**：一位成员分享了一个专注于战锤 (Warhammer) 主题的 [GitHub 项目](https://github.com/SilverBC/Warhammer-Adaptive-RAG)，该项目采用 Adaptive RAG，并寻求反馈和改进建议。
   - 一位社区成员称赞了该项目，强调了 **幻觉 (hallucination)** 检测、**答案评分 (answer grading)** 以及 **本地模型 (local models)** 的使用等特性。
- **探索 Tavily 的替代方案**：在关于 **Tavily** 的讨论中，一位成员建议了潜在的替代方案，如 **Google Serper** 和 **SEARXNG**，并指出 Tavily 在 LLM 搜索方面的专业性。
   - 他们还提到了用于各种任务的其他工具，如 **BeautifulSoup** 和 **Sherpa LLM**。
- **LlamaParse 超出预期**：[Silver_steel_io](https://github.com/SilverBC/Warhammer-Adaptive-RAG) 提到 **LlamaParse** 在生成结构化文件方面明显优于其他方法，但由于规则集庞大，面临 **每天 1000 页** 的限制。
   - 成员们讨论了在战锤项目背景下结构化文件摄取 (ingestion) 的重要性。
- **Vantager 的 AI 工程师招聘**：一位成员宣布了 **Vantager** 的 **创始 AI 工程师 (Founding AI Engineer)** 职位空缺，该公司专注于全球资本配置的 AI 原生平台。
   - 他们鼓励感兴趣的候选人查看消息中链接的 **职位公告板 (job board)**，并强调了他们获得的 VC 支持以及目前在解决 **海量数据问题** 方面的工作量。
- **NPC 生成器项目的潜在协作**：一位成员邀请大家就一个 **个人项目** 进行协作，该项目旨在创建一个 NPC 生成器，根据定义的属性为 LLM 生成自定义提示词 (prompts)。
   - 他们提议组建一个小团队，为 RPG 开发 **随机 NPC 属性**，从而改变 LLM 的人格设定 (personas) 和说话模式。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://vantager.notion.site/Vantager-Job-Board-a951591057724736be288bd9cb0c9fe3">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://github.com/SilverBC/Warhammer-Adaptive-RAG">GitHub - SilverBC/Warhammer-Adaptive-RAG</a>：通过在 GitHub 上创建账户来为 SilverBC/Warhammer-Adaptive-RAG 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1284130186341650452)** (2 messages): 

> - `Forum Etiquette` (论坛礼仪)
> - `MypyC Compilation Progress` (MypyC 编译进度)
> - `Llama-7B Integration` (Llama-7B 集成)
> - `Code Changes Summary` (代码变更摘要)
> - `C Extensions Future` (C Extensions 未来规划)


- **论坛成员讨论礼仪**：一位成员强调了**基础论坛礼仪**的重要性，指出重复的求助请求可能会打击他人提供帮助的积极性。
   - *浪费他人时间*会挫伤社区参与度，呼吁采用更好的沟通实践。
- **Tinygrad 的 MypyC 编译进展**：一位成员详细介绍了他们系统化的 **MypyC 编译**方法，为了提高效率，工作从整个项目逐步深入到单个文件。
   - 编译的文件包括 `tinygrad/device.py` 和 `tinygrad/tensor.py`，表明项目取得了重大进展。
- **成功在 Tinygrad 上运行 Llama-7B**：该成员使用 **Llama-7B 模型**成功运行了 *examples/llama.py*，平均耗时性能提升了 **12%**。
   - 他们提供了 [Llama-7B 仓库](https://huggingface.co/huggyllama/llama-7b/tree/main)的链接，以供参考所使用的模型。
- **实现 MypyC 功能的代码变更**：对多个文件进行了代码修改，包括重写生成器和添加装饰器，以启用 **MypyC 功能**。
   - 该成员将这些更改描述为*初稿*，在进一步完善之前寻求团队的反馈。
- **C Extensions 的未来考量**：该成员建议，如果要将 **C Extensions** 集成到 Tinygrad 中，应采取逐步推进的方式以便于修改。
   - 他们渴望在敲定贡献之前，确保正在进行的工作与更广泛的项目目标保持一致。



**提及的链接**：<a href="https://huggingface.co/huggyllama/llama-7b/tree/main">huggyllama/llama-7b at main</a>：未找到描述内容

  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1284082084519870526)** (2 messages): 

> - `Gorilla OpenFunctions Model Accuracy` (Gorilla OpenFunctions 模型准确率)
> - `Error Decoding AST` (AST 解码错误)
> - `User Info Retrieval Function` (用户信息检索函数)


- **Gorilla OpenFunctions 模型准确率为零**：**gorilla-openfunctions-v2** 模型的测试结果显示准确率为 **0.0**，共进行了 **258** 次评估。
   - 尽管 **model_result_raw** 与 **possible_answer** 匹配，准确率仍保持为零，表明存在潜在问题。
- **用户信息函数的 AST 解码错误**：报告的一个错误是 *Invalid syntax. Failed to decode AST*，这表明在正确处理输入时存在问题。
   - 具体指出 *can only concatenate str (not "list") to str*，暗示函数中存在数据类型不匹配。
- **成功检索用户 ID 数据**：模型尝试检索 **ID 7890** 的用户详情并成功确认。
   - 检索到的数据包括用户名 **user7890** 和电子邮件 **user7890@example.com**，满足了对**黑色**特殊项目的特定请求。


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1283903818777628876)** (1 messages): 

> - `LLM fine-tuning for translations` (用于翻译的 LLM 微调)
> - `Challenges in tone and style preservation` (语气和风格保留的挑战)


- **微调 LLM 以获得更好的翻译**：一位成员询问了专门针对**翻译**进行 **LLM** 微调的经验，强调许多模型虽然能捕捉大意，但无法捕捉原文的**语气和风格**。
   - 这引发了关于如何在不丢失核心细微差别的情况下提高**翻译质量**的持续关注。
- **在翻译中捕捉语气的困境**：有人指出，虽然 LLM 可以提供不错的翻译，但往往无法有效地传达源材料的**语气**和**风格**。
   - 鼓励成员们分享有助于弥补**翻译忠实度**差距的方法或见解。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1284218605231280241)** (1 条消息): 

> - `Fleak AI Private Gathering`
> - `Serverless API Builder`
> - `Community Building Initiatives` 


- **Fleak AI 举办私人聚会**：Fleak AI 今晚在旧金山的[此地点](https://lu.ma/l9tpptle?tk=KfASyJ)为朋友和用户举办一场私人 Happy Hour。活动旨在汇聚社区成员，并讨论 Fleak 的最新动态。
- **Fleak：一个 Serverless API Builder**：Fleak 的市场定位是用于 AI workflows 的 Serverless API Builder，非常适合 **sentiment labeling** 等功能。这次活动可能为对 API 解决方案感兴趣的开发者提供社交机会。
- **专注于社区建设**：活动组织者打算通过更多的线下见面会来加强社区，从这次 Happy Hour 开始。他们旨在营造友好的氛围，以促进与会者之间的讨论。



**提到的链接**：<a href="https://lu.ma/l9tpptle?tk=KfASyJ">Fleak Happy Hour! · Luma</a>：你好！欢迎参加我们的首场 Fleak Happy Hour。在这里，我们将有时间互相认识，并讨论 Fleak 的最新进展。

  

---



---



---



---



{% else %}


> 完整的频道分类明细已在邮件中截断。 
> 
> 如果你想查看完整明细，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}