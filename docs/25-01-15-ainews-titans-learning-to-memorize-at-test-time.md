---
companies:
- google
- meta-ai-fair
- openai
- anthropic
- langchain
date: '2025-01-16T07:58:41.269179Z'
description: '**谷歌（Google）**发布了一篇关于“神经记忆”（Neural Memory）的新论文，在推理阶段（test time）将持久性记忆直接集成到
  Transformer 架构中，展示了在长上下文利用方面的潜力。由 @omarsar0 发布的 **MiniMax-01** 拥有 **400 万 token
  的上下文窗口**、**4560 亿参数**和 **32 个专家**，其性能超越了 **GPT-4o** 和 **Claude-3.5-Sonnet**。**InternLM3-8B-Instruct**
  是一款基于 **4 万亿 token** 训练的开源模型，达到了当前最先进的（SOTA）水平。**Transformer²** 引入了自适应大语言模型（LLM），通过动态调整权重来实现持续适配。AI
  安全方面的进展强调了对**智能体身份验证**、**提示词注入**防御以及**零信任架构**的需求。像 **Micro Diffusion** 这样的工具让低成本的扩散模型训练成为可能，而
  **LeagueGraph** 和 **Agent Recipes** 则为开源社交媒体智能体提供了支持。'
id: c1808ea6-4be4-41ff-963c-93fb93595f78
models:
- minimax-01
- gpt-4o
- claude-3.5-sonnet
- internlm3-8b-instruct
- transformer2
original_slug: ainews-titans-learning-to-memorize-at-test-time
people:
- omarsar0
- hwchase17
- abacaj
- hardmaru
- rez0__
- bindureddy
- akhaliq
- saranormous
title: '**Titans：在测试时学习记忆**'
topics:
- long-context
- mixture-of-experts
- self-adaptive-models
- prompt-injection
- agent-authentication
- diffusion-models
- zero-trust-architecture
- continuous-adaptation
- vision
- agentic-systems
---

<!-- buttondown-editor-mode: plaintext -->**Neural Memory is all you need.**

> 2025年1月14日至1月15日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord（**219** 个频道，**2812** 条消息）。预计为您节省了 **327 分钟** 的阅读时间（以 200wpm 计算）。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

许多人都在热议 Google 最新的论文，被一些博主誉为 "Transformers 2.0" ([arxiv](https://arxiv.org/abs/2501.00663v1), [tweet](https://x.com/behrouz_ali/status/1878859086227255347))：


![image.png](https://assets.buttondown.email/images/908abdaa-0657-46c1-bf61-5bd6d7b1b04a.png?w=960&fit=max)



它似乎在 "test time" 直接将持久化存储（persistent memory）集成到架构内部，而不是放在架构之外（[这是作为 context、head 或 layer 的三种变体之一](https://x.com/behrouz_ali/status/1878859912195039445/photo/2)）。


![image.png](https://assets.buttondown.email/images/df1ed1f1-642f-424b-9cf9-36a95d0b0311.png?w=960&fit=max)


该论文显著地使用了一种惊奇度（surprisal）度量来更新其记忆：


![image.png](https://assets.buttondown.email/images/8e0e78d8-718d-4bdc-9982-50cb41ef61d3.png?w=960&fit=max)


并通过权重衰减（weight decay）来模拟遗忘过程：


![image.png](https://assets.buttondown.email/images/143cf9f1-5c26-4f58-8f3d-c5b02425c041.png?w=960&fit=max)


最终结果显示，在长上下文（long contexts）下，其上下文利用率表现非常出色。


![image.png](https://assets.buttondown.email/images/6a0cfb42-9c12-45b9-b3fb-587d3505bb15.png?w=960&fit=max)


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有总结均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型与扩展 (Scaling)**

- **MiniMax-01 与超长上下文模型**：[@omarsar0](https://twitter.com/omarsar0/status/1879572512075587872) 介绍了 **MiniMax-01**，它集成了 **Mixture-of-Experts** 架构，拥有 **32 个专家**和 **456B 参数**。它拥有 **400 万 token 的上下文窗口**，性能超越了 **GPT-4o** 和 **Claude-3.5-Sonnet** 等模型。同样，[@hwchase17](https://twitter.com/hwchase17/status/1879439184462762225) 强调了**视觉空间草稿纸 (vision-spatial scratchpads)** 的进展，解决了 **VLMs** 中长期存在的挑战。

- **InternLM 与开源 LLMs**：[@abacaj](https://twitter.com/abacaj/status/1879333563042316411) 讨论了 **InternLM3-8B-Instruct**，这是一款采用 **Apache-2.0 许可**的模型，在 **4 万亿 token** 上进行了训练，达到了**最先进的性能**。[@AIatMeta](https://twitter.com/AIatMeta/status/1879593561215430923) 分享了发表在 **@Nature** 上的 **SeamlessM4T** 更新，强调了其受自然启发的**自适应系统**。

- **Transformer² 与自适应 AI**：[@hardmaru](https://twitter.com/hardmaru/status/1879331049383334187) 展示了 **Transformer²**，展示了能够**动态调整权重**的**自适应 LLMs**，连接了 **pre-training** 和 **post-training** 以实现**持续适应**。

**AI 应用与工具**

- **AI 驱动的开发工具**：[@rez0__](https://twitter.com/rez0__/status/1879557690101260681) 概述了对强大的 **Agent Authentication**、**Prompt Injection** 防御和**安全 Agent 架构**的需求。此外，[@hkproj](https://twitter.com/hkproj/status/1879603337206919365) 推荐使用 **Micro Diffusion** 在有限预算下训练 **diffusion models**。

- **Agent 系统与自动化**：[@bindureddy](https://twitter.com/bindureddy/status/1879576445913374898) 强调了 **Search-o1** 在增强**复杂推理任务**方面的潜力，其表现优于传统的 **RAG** 系统。[@LangChainAI](https://twitter.com/LangChainAI/status/1879576934365135009) 推出了 **LeagueGraph** 和 **Agent Recipes**，用于构建**开源社交媒体 Agent**。

- **与开发环境的集成**：[@_akhaliq](https://twitter.com/_akhaliq/status/1879339311784726664) 讨论了为跨应用的 AI 模型支持**统一本地端点**，而 [@saranormous](https://twitter.com/saranormous/status/1879320464948150504) 则提倡使用 **Grok 的 Web 应用**以避免干扰。

**AI 安全与伦理担忧**

- **数据完整性与 Prompt Injection**：[@rez0__](https://twitter.com/rez0__/status/1879557690101260681) 强调了 **prompt injection** 的挑战以及采用**零信任架构 (zero-trust architectures)** 来保护 **LLM 应用**的必要性。[@lateinteraction](https://twitter.com/lateinteraction/status/1879576445913374898) 批评了 **AI prompts** 中**规范与实现之间界限的模糊**，主张建立更清晰的**领域特定知识**。

- **地缘政治与 AI 监管**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1879517764735717697) 批评了关于**中国服务器**存在**显式后门**的说法，转而推崇 **Apple 的安全模型**。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1879593367799300303) 讨论了 **AI 扩散规则**对 **NVIDIA 股票**的影响，并对**全球监管格局**进行了反思。

**教育与招聘中的 AI**

- **家庭教育与教育政策**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1879502879213670578) 对**战斗动画**以及缺乏**有效的教育技术**表示失望。同时，[@stanfordnlp](https://twitter.com/stanfordnlp/status/1879578794354426045) 举办了关于 **AI 与教育**的研讨会，强调了 **Agent** 和**工作流**的作用。

- **招聘与技能发展**：[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1879562176148758544) 分享了关于**招聘 ML 工程师**的见解，而 [@fchollet](https://twitter.com/fchollet/status/1879586506471559244) 正在为 **AI 程序合成**寻找专家，强调了**数学和编程技能**的重要性。

**软件工程中的 AI 集成**

- **LLM 集成与生产力**：[@TheGregYang](https://twitter.com/TheGregYang/status/1879439400230428795) 和 [@gdb](https://twitter.com/gdb/status/1879327050819104778) 讨论了将 **LLM** 无缝集成到**调试工具**和 **Web 应用**中，以提高**开发者生产力**。[@rasbt](https://twitter.com/rasbt/status/1879538621276913901) 强调了**原始智能**与**智能软件系统**之间的区别，主张采用正确的**实施策略**。

- **AI 驱动的编程与自动化**：[@hellmanikCoder](https://twitter.com/hellmanikCoder/status/1879348975171682520) 和 [@skycoderrun](https://twitter.com/skycoderrun/status/1879333563042316411) 强调了使用 **LLM** 进行**代码生成**和**自动化**的优势与挑战，并强调了对**稳健集成**和**错误处理**的需求。

**政治与 AI 监管**

- **中国 AI 发展与安全**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1879505671433126395) 提到了**科大讯飞收购 Atlas 服务器**，反映了中国 **AI 基础设施的增长**。[@manyothers](https://twitter.com/manyothers/status/1879501254046761355) 讨论了 **昇腾（Ascend）集群**潜在的加速发展，突显了**中国大陆**在**计算领域**的进步。

- **美国 AI 政策与基础设施**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1879295847105536331) 总结了一项关于**加速 AI 基础设施**的**美国行政命令**，详细阐述了**数据中心要求**、**清洁能源指令**以及**国际合作**。[@karinanguyen_](https://twitter.com/karinanguyen_/status/1879576742001877025) 批评了 **AI 工作流**在**形态因素（form factors）**上的滞后，并反思了**政策影响**。

**梗图/幽默**

- **关于 AI 与日常生活的幽默观点**：[@arojtext](https://twitter.com/arojtext/status/1879540391109951670) 调侃为了**逃避现实**而**隐藏电子游戏的更好用途**，而 [@qtnx_](https://twitter.com/qtnx_/status/1879572052044353903) 分享了一个关于**无关 AI 应用**的**有趣问题**。此外，[@nearcyan](https://twitter.com/nearcyan/status/1879619344772329897) 幽默地反思了**游戏习惯**和**意外的置业报价**。

- **轻松的 AI 评论**：[@Saranormous](https://twitter.com/saranormous/status/1879566106081513550) 嘲讽了与 **ScaleAI** 的互动，[@TheGregYang](https://twitter.com/TheGregYang/status/1879320464948150504) 则俏皮地鼓励使用 **Grok 的 Web 应用**来避免分心，将 **AI 功能**与**日常幽默**结合在一起。

---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1：InternLM3-8B 表现优于 Llama3.1-8B 和 Qwen2.5-7B**

- **[新模型....](https://i.redd.it/curwy8vkq3de1.png)** ([Score: 188, Comments: 31](https://reddit.com/r/LocalLLaMA/comments/1i1rgn9/new_model/)): 据报道 **InternLM3** 的表现优于 **Llama3.1-8B** 和 **Qwen2.5-7B**。该项目名为 "internlm3-8b-instruct"，托管在一个类似于 **GitHub** 的平台上，带有 "**Safetensors**" 和 "**custom_code**" 等标签，并引用了标识符为 **2403.17297** 的 **arXiv** 论文。
  - **InternLM3 的性能与特性**：用户强调了 **InternLM3** 优于 **Llama3.1-8B** 和 **Qwen2.5-7B** 的卓越性能，并强调其仅使用 **4 trillion tokens** 进行训练的高效性，使成本降低了 **75%** 以上。该模型支持“深度思考模式”以处理复杂的推理任务，可以通过 [Hugging Face](https://huggingface.co/internlm/internlm3-8b-instruct) 上显示的不同系统提示词（system prompt）来开启。
  - **社区反馈与对比**：用户对 **InternLM3** 的能力表示满意，指出其在逻辑和语言任务中的有效性，并认为其优于 **Exaone 7.8b** 和 **Qwen2.5-7B** 等模型。社区希望能推出 **20 billion parameter** 版本，并参考了 [2.5 20b](https://huggingface.co/internlm/internlm2_5-20b-chat) 模型。
  - **模型命名与许可**：讨论了 "Intern" 这个名字，一些用户认为由于 AI 扮演着无薪助手的角色，这个名字非常贴切。此外，用户呼吁在分享模型时应有更清晰的许可实践，并对不明确的许可证表示沮丧，特别是在音频/音乐模型中。


**主题 2. OpenRouter 获得新功能和社区驱动的改进**

- **OpenRouter 用户：你还缺少什么功能？** ([Score: 190, Comments: 79](https://reddit.com/r/LocalLLaMA/comments/1i1owp1/openrouter_users_what_feature_are_you_missing/))：作者无意中开发了一个名为 [glama.ai/gateway](https://glama.ai/gateway) 的 **OpenRouter** 替代方案，它提供类似的功能，如更高的 **rate limits** 和通过 **OpenAI** 兼容 **API** 轻松切换模型。其独特优势包括与 **Chat** 和 **MCP** 生态系统的集成、高级分析功能，以及据称比 **OpenRouter** 更低的延迟和更高的稳定性，同时每天处理数十亿个 **tokens**。
  - **API 兼容性与支持**：用户对与 **OpenAI API** 的兼容性表示关注，特别是关于多轮工具使用、**function calling** 和图像输入格式。关注点包括工具使用语法的差异，以及对支持的 **API** 功能和模型的详细文档需求。
  - **供应商管理与数据安全**：用户要求对特定模型的供应商选择进行更细粒度的控制，因为某些供应商（如 **DeepInfra**）并非对所有模型都是最优的。此外，**glama.ai** 因其数据保护政策和不使用客户数据进行 AI 训练的承诺而受到称赞，这与 **OpenRouter** 的数据处理实践形成对比。
  - **Sampler 选项与移动端支持**：用户讨论了对 **XTC** 和 **DRY** 等额外 **sampler** 选项的需求（目前尚不支持），以及作为中间商实现这些选项的挑战。此外，用户对改进移动端支持也很感兴趣，因为目前的流量主要来自桌面端，但移动端正成为一个更频繁的讨论话题。


**主题 3. Kiln 作为 Google AI Studio 的开源替代方案受到关注**

- **我不小心构建了一个 Google AI Studio 的开源替代方案** ([Score: 865, Comments: 130](https://reddit.com/r/LocalLLaMA/comments/1i1ffid/i_accidentally_built_an_open_alternative_to/)): **Kiln** 是 **Google AI Studio** 的开源替代方案，提供增强功能，如通过多个主机支持任何 LLM、无限的微调能力、本地数据隐私和协作使用。它与 Google 有限的模型支持、数据隐私问题和单用户协作形成对比，同时还提供 Python 库和强大的数据集管理。**Kiln** 已在 [GitHub](https://github.com/Kiln-AI/Kiln) 上发布，旨在像 Google AI Studio 一样易于使用，但功能更强大且更具私密性。
  - 用户对**隐私和许可**的担忧非常突出，**osskid** 和 **yhodda** 等用户指出了 Kiln 的隐私声明与其 **EULA** 之间的差异，后者暗示 Kiln 可能拥有数据访问和使用权。**Yhodda** 强调桌面应用程序的专有许可证可能导致用户数据在没有补偿的情况下被共享和使用，这引发了对用户权利和隐私的警示。
  - 用户赞赏 Kiln 的**开源特性**，**fuckingpieceofrice** 和 **Imjustmisunderstood** 等人的评论表达了对 Google AI Studio 替代方案的感激，担心未来会出现付费墙。开源方面被视为一个显著优势，即使桌面组件不是开源的。
  - **文档和教程**收到了积极反馈，**Kooky-Breadfruit-837** 和 **danielhanchen** 等用户称赞了详尽的指南和迷你视频教程。正如 **RedZero76** 所指出的，这表明 Kiln 对用户友好且易于上手，即使是技术经验有限的人也是如此。


**Theme 4. OuteTTS 0.3 推出全新的 1B 和 500M 语言模型**

- **[OuteTTS 0.3: 全新 1B & 500M 模型](https://v.redd.it/rb1px5mjs5de1)** ([Score: 155, Comments: 62](https://reddit.com/r/LocalLLaMA/comments/1i1xbv1/outetts_03_new_1b_500m_models/)): **OuteTTS 0.3** 推出了全新的 **1B** 和 **500M** 模型，增强了其 text-to-speech 能力。此次更新可能包括模型性能和功能集的改进，尽管文中未提供具体细节。
  - 关于 OuteTTS 的**语言支持**有显著讨论，特别是尽管西班牙语使用广泛但仍缺失。**OuteAI** 解释说这是由于西班牙语口音和方言的多样性，以及缺乏足够的数据集，导致输出结果是通用的 "Latino Neutro"。
  - **OuteAI** 阐明了模型的各种技术细节，例如它们基于 **LLMs** 并使用 **WavTokenizer** 进行音频 Token 解码。这些模型与 **Transformers, LLaMA.cpp** 和 **ExLlamaV2** 兼容，并且正在持续探索 **speech-to-speech** 能力。
  - **OuteTTS 0.3** 模型在语音的自然度和连贯性方面有所提升，支持包括新加入的法语和德语在内的六种语言。Demo 已在 [Hugging Face](https://huggingface.co/spaces/OuteAI/OuteTTS-0.3-1B-Demo) 上线，通过 **pip** 即可轻松安装。


**Theme 5. 405B MiniMax MoE：在上下文长度和效率方面的突破**

- **405B MiniMax MoE 技术深度解析** ([Score: 66, Comments: 10](https://reddit.com/r/LocalLLaMA/comments/1i1ty0e/405b_minimax_moe_technical_deepdive/)): 该帖子讨论了 **405B MiniMax MoE** 模型，强调了其创新的扩展方法，包括与 **7/8 Lightning attention** 的混合以及与 DeepSeek 不同的 **MoE strategy**。它详细介绍了该模型在约 **2000 H800** 和 **12 trillion tokens** 上的训练情况，更多信息可在 [Hugging Face](https://huggingface.co/blog/eliebak/minimax01-deepdive) 的博客文章中找到。
  - **405B MiniMax MoE** 模型因其在没有 **Chain of Thought (CoT)** 的情况下在 **Longbench** 上的出色表现而受到关注，展示了其处理长上下文的能力。**FiacR** 强调了其“疯狂的上下文长度”，**eliebakk** 赞扬了其“超级令人印象深刻的数据”。
  - 关于**开源权重模型（open weights models）**与闭源模型竞争趋势的讨论非常积极，人们对 **2025** 年的重大进展持乐观态度。**vaibhavs10** 对这一趋势表示热忱，并分享了 [Hugging Face 上的 MiniMaxAI 模型](https://huggingface.co/MiniMaxAI/MiniMax-Text-01)链接。
  - 正如 **StevenSamAI** 所提到的，该模型托管在 [Hailuo.ai](https://www.hailuo.ai/) 上，为访问该模型提供了资源。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Transformer²：增强实时 LLM 适应性**

- **[R] Transformer²: Self-Adaptive LLMs** ([Score: 137, Comments: 10](https://reddit.com/r/MachineLearning/comments/1i1l8d4/r_transformer²_selfadaptive_llms/)): **Transformer²** 为**大语言模型 (LLMs)** 引入了一个自适应框架，该框架能够动态调整权重矩阵的奇异分量，以实时处理未知任务，在参数更少的情况下性能优于 **LoRA** 等传统方法。该方法采用双阶段机制，利用调度系统和通过强化学习训练的任务特定“专家”向量，展示了在各种架构和模态（包括视觉语言任务）中的通用性。[论文](https://arxiv.org/abs/2501.06252), [博客摘要](https://sakana.ai/transformer-squared/), [GitHub](https://github.com/SakanaAI/self-adaptive-llms)。
  - 评论者讨论了该框架的**扩展方式**，指出它是随磁盘空间而非参数数量扩展的，这可能意味着在存储和计算方面具有较高的效率。
  - 讨论涉及了 **Transformer²** 对不同规模 **LLM** 的性能影响，观察到虽然它显著增强了较小的模型，但对于像 **700 亿参数 (70 billion parameter)** 模型这样的大型模型，改进微乎其微。
  - **Sakana** 实验室的发展受到了关注，人们注意到该论文没有列出知名作者，这表明研究团队正在扩大且协作性日益增强。


**主题 2. 深度学习革新预测性医疗保健**

- **[研究人员开发出预测乳腺癌的深度学习模型](https://i.redd.it/mb60cjrts7de1.jpeg)** ([Score: 118, Comments: 17](https://reddit.com/r/OpenAI/comments/1i26i1e/researchers_develop_deep_learning_model_to/)): 研究人员开发了一种**深度学习模型**，能够使用精简算法提前**五年**预测乳腺癌。该研究分析了超过 **210,000 份乳腺 X 线摄影图像**，并强调了**乳腺不对称**在评估癌症风险中的重要性，详情见这篇 [RSNA 文章](https://www.rsna.org/news/2024/march/deep-learning-for-predicting-breast-cancer)。

---

# AI Discord 摘要

> 由 o1-preview-2024-09-12 生成的摘要之摘要的摘要

**主题 1：AI 模型性能在各平台遭遇挫折**

- [**Perplexity 用户因持续停机感到困惑**](https://www.perplexity.ai)：用户报告 **Perplexity** 出现多次长时间停机，错误持续超过一小时，引发了不满并促使人们寻找替代方案。
- [**Cursor 运行缓慢，性能陷阱困扰开发者**](https://www.cursor.com/blog/tab-update)：**Cursor IDE** 用户面临严重的运行缓慢问题，5-10 分钟的等待时间阻碍了 **Pro** 订阅者的工作流，并引发了关于修复方案的推测。
- [**DeepSeek 响应迟缓，用户寻求更快的替代方案**](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/discussions/5)：**DeepSeek V3** 受到延迟问题和响应缓慢的困扰，导致用户转向 **Sonnet** 等模型，并对性能不稳定表示失望。

**主题 2：新 AI 模型突破上下文障碍**

- [**MiniMax-01 以 400 万 Token 上下文开辟新路径**](https://openrouter.ai/minimax/minimax-01)：**MiniMax-01** 发布，利用 **Lightning Attention** 实现了前所未有的 **400 万 Token** 上下文窗口，承诺提供超长上下文处理能力和性能飞跃。
- [**Cohere 将上下文提升至 128k Token**](https://docs.cohere.com/v2/docs/rate-limits)：**Cohere** 将上下文长度扩展至 **128k Token**，支持在单次对话中处理约 42,000 个单词而无需重置，增强了连贯性。
- [**Mistral 的 FIM 奇迹令开发者惊叹**](https://openrouter.ai/mistralai/codestral-2501)：来自 **Mistral AI** 的新型 **Fill-In-The-Middle** (FIM) 编程模型凭借超越标准能力的先进代码补全和片段处理能力给用户留下了深刻印象。

**主题 3：法律纠纷冲击 AI 数据集和开发者**

- [**MATH 数据集遭遇 DMCA 卸载：AoPS 发起反击**](https://huggingface.co/datasets/hendrycks/competition_math/discussions/5)：**Hendrycks MATH** 数据集面临 **DMCA** 删帖，引发了对 **Art of Problem Solving (AoPS)** 内容以及 AI 领域数学数据未来的担忧。
- [**JavaScript 商标之争威胁开源社区**](https://www.perplexity.ai/search/why-are-there-two-perplexity-a-8DCudsGDRdaCTh4fX6YAhQ#1)：一场关于 **JavaScript** 商标的激烈法律纠纷引发了警报，可能产生的限制将影响社区主导的开发和开源贡献。

**主题 4：AI 训练的进展与争论**

- [**Grokking 取得进展：现象深度解析**](https://youtu.be/SRfJQews1AU)：一段名为 *“Finally: Grokking Solved - It's Not What You Think”* 的新视频深入探讨了延迟泛化这一奇特的 **Grokking** 现象，激发了热议和争论。
- [**动态量化（Dynamic Quantization）的怪异表现引发质疑**](https://unsloth.ai/blog/phi4)：用户报告在对 **Phi-4** 应用动态量化时性能变化极小，引发了关于该技术与标准 4-bit 版本相比有效性的讨论。
- [**TruthfulQA 基准测试被简单技巧破解**](https://x.com/Turn_Trout/status/1879710659904254081)：**TurnTrout** 通过利用几个简单的规则漏洞，在 **TruthfulQA** 上实现了 **79% 的准确率**，突显了基准测试可靠性的缺陷。

**主题 5：行业动态搅动 AI 格局**

- [**Cursor AI 在 B 轮融资中获得巨额资金**](https://x.com/sarahdingwang/status/1879279307119608142)：**Cursor AI** 完成了由 **a16z** 领投的新一轮 **Series B** 融资，助力该编程平台的下一阶段发展，并在按需计费定价谈判中加强了与 **Anthropic** 的联系。
- [**Anthropic 获得 ISO 42001 负责任 AI 认证**](https://www.iso.org/standard/81230.html)：**Anthropic** 宣布获得新标准 **ISO/IEC 42001:2023** 的认证，强调了负责任 AI 开发的结构化系统治理。
- [**NVIDIA Cosmos 在 CES 亮相，令 LLM 爱好者印象深刻**](https://github.com/NVIDIA/Cosmos)：**NVIDIA Cosmos** 在 **CES** 上揭晓，展示了新的 AI 能力；[LLM Paper Club](https://lu.ma/pvh0rwa3) 的演讲强调了其对该领域的潜在影响。

---

# 第一部分：Discord 高层级摘要

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 运行缓慢：性能陷阱困扰用户**：许多用户报告在 **Cursor** 中遇到了 5-10 分钟的等待时间，而其他用户则表示速度正常，这让试图保持稳定工作流的 **Pro** 订阅用户感到沮丧。
   - 这种减速阻碍了编码工作，并引发了关于修复方案的推测，一些人正关注 [Cursor 的博客更新](https://www.cursor.com/blog/tab-update) 以寻求潜在的缓解方案。
- **快速部署：Vercel 与 Firebase 表现出色**：开发者们称赞 **Vercel** 和 **Google Firebase** 在部署基于 Cursor 开发的应用时表现优异，强调其生产环境配置极简。
   - 他们分享了 [Vercel 上的模板](https://vercel.com/templates) 以实现快速启动，并指出了与 Firebase 轻松进行实时集成的优势。
- **Gemini 2.0 Flash 对决 Llama 3.1**：爱好者们更倾向于 **Gemini 2.0 Flash**，认为其 Benchmark 结果优于 **Llama 3.1**，并指出其文本生成性能更犀利。
   - 另一些人承认，由于对 AI 的过度依赖，产生了“冒名顶替综合征”（*imposter syndrome*），但同时也接受了生产力提升带来的好处。
- **Sora 在慢动作场景中受挫**：有报告称 **Sora** 在生成可靠视频方面遇到困难，尤其是在慢动作片段中，这让部分用户感到不满。
   - 在频繁的尝试和失败后，一些人开始探索替代方案，表明对 Sora 的功能集评价褒贬不一。
- **Fusion 热潮：Cursor 为三月发布做准备**：预计 **Cursor** 将在三月发布新版本，重点包括 **Fusion** 的实现，以及可能与 **DeepSeek** 和 **Gemini** 的集成。
   - 尽管具体细节尚未披露，但 [Cursor 的 Tab 模型文章](https://www.cursor.com/blog/tab-update) 中透露的信息让人们对这个功能更强大的平台充满期待。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 宕机引发不满**：**Perplexity** 频繁的服务中断导致用户面临超过一小时的报错，正如 [状态页面](https://status.perplexity.com) 所示，这引发了用户对备份方案的需求。
   - 随着 **citation**（引用）故障的出现，用户的挫败感进一步增加，导致一些人开始寻找替代方案，并对可靠性表示担忧。
- **AI 模型性能大比拼**：社区成员权衡了最佳编码模型，其中 **Claude Sonnet 3.5** 在调试任务中脱颖而出，而 **Deepseek 3.0** 被提议作为经济实惠的备选方案。
   - 一些人称赞 **Perplexity** 处理某些查询的能力，但同时也批评其 **hallucinations**（幻觉）和有限的 **context window**（上下文窗口）。
- **双重幻影：两个 Perplexity iOS 应用**：一位用户在 App Store 中发现了重复项，并引发了关于官方 **Perplexity** 应用的 [网页查询](https://www.perplexity.ai/search/why-are-there-two-perplexity-a-8DCudsGDRdaCTh4fX6YAhQ#1)。
   - 另一位用户无法找到第二个列表，引发了关于 **naming**（命名）和分发问题的简短讨论。
- **JavaScript 商标之争**：如果商标诉求变得更加严格，一场关于 **JavaScript** 商标的法律斗争可能会威胁到社区主导的开发。
   - 舆论对所有权问题以及可能影响开源贡献的诉讼浪潮表示担忧。
- **Llama-3.1-Sonar-Large 速度变慢**：**llama-3.1-sonar-large-128k-online** 自 **1 月 10 日**以来输出速度明显下降，令用户感到困惑。
   - 社区讨论指向未公开的更新或代码变动可能是导致减速的原因，引发了对更广泛性能影响的担忧。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Command 功能热潮与编辑器好评**：Windsurf 发布了 [Command 功能教程](https://x.com/windsurf_ai/status/1879332883489591404)，通过 [Luminary](https://www.youtube.com/watch?v=LyOMJq47ASQ) 等视频公布了 Discord 挑战赛获胜者，并推出了 **Windsurf Editor**。
   - 他们还展示了 **Codeium** 对标 **GitHub Copilot** 的官方对比，并分享了一篇关于[非许可代码担忧的博客文章](https://www.codeium.com/blog/copilot-trains-on-gpl-codeium-does-not)。
- **Telemetry 纠葛与订阅障碍**：用户在 Codeium Visual Studio 扩展中遇到了 **telemetry**（遥测）问题，并对订阅计划中的额度结转（credit rollover）感到困惑，参考了 [GitHub issues](https://github.com/Exafunction/CodeiumVisualStudio/issues/111)。
   - 他们确认计划取消后额度不会结转，部分用户遇到了与扩展清单（manifest）命名相关的**安装**问题。
- **学生折扣与远程仓库谜题**：学生对 **Pro Tier** 捆绑包表现出兴趣，但如果地址不是 .edu 则会遇到困难，这引发了对更具包容性的资格审查的呼声。
   - 其他人报告了在 IntelliJ 中配合 Codeium 使用**已索引的远程仓库**时存在摩擦，并寻求社区的设置建议。
- **C# 类型问题与 Cascade 讨论**：Windsurf IDE 在 Windows 和 Mac 上分析 C# 变量类型时持续出现问题，尽管 VS Code 等其他编辑器表现流畅。
   - 用户讨论了 Cascade 的性能并推荐了高级 Prompt，同时还讨论了集成 Claude 和其他模型以处理复杂编码任务。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Kaggle 中的多 GPU 混乱**：开发者尝试在 Kaggle 上使用多个 T4 GPU 运行 **Unsloth**，但发现仅有一个 GPU 被启用用于推理，限制了扩展尝试。他们提到了[这篇关于在 Kaggle T4 GPU 上进行微调的推文](https://x.com/helloiamleonie/status/1879511537343434817)，希望能更有效地利用 Kaggle 的免费时长。
   - 其他人建议如果需要并发，应付费购买更强大的硬件，并暗示 Kaggle 未来可能会扩大 GPU 供应。
- **揭秘微调误区**：团队澄清说，**fine-tuning**（微调）实际上可以引入新知识，其作用类似于检索增强生成（RAG），这与广泛的假设相反。他们链接了 [Unsloth 关于微调益处的文档](https://docs.unsloth.ai/get-started/beginner-start-here/is-fine-tuning-right-for-me)以解决这些持续存在的误解。
   - 有人指出，通过将新数据嵌入模型，它可以减轻内存使用，而其他人则强调了为获得最佳结果而进行正确数据集选择的重要性。
- **Phi-4 的动态量化奇点**：有报告显示 **Phi-4** 在动态量化后性能变化极小，与标准的 4-bit 版本非常接近。用户参考了 [Unsloth 4-bit 动态量化集合](https://huggingface.co/collections/unsloth/unsloth-4-bit-dynamic-quants-67503bb873f89e15276c44e7)来调查任何潜在的收益。
   - 一些人坚持认为动态量化应该提高准确性，这促使进一步的实验以确认差异是否被测试条件所掩盖。
- **Grokking 研究取得进展**：一段新视频 [Finally: Grokking Solved - It's Not What You Think](https://youtu.be/SRfJQews1AU?si=s3CSvyThYNcTetX_) 深入探讨了延迟泛化这一奇特的 **grokking**（顿悟）现象。它激发了人们理解过拟合如何转化为模型能力突飞猛进的热情。
   - 分享的论文 [Grokking at the Edge of Numerical Stability](https://arxiv.org/abs/2501.04697v1) 引入了 *Softmax Collapse* 的概念，引发了关于对 AI 训练更深层影响的辩论。
- **LLM 在安全会议上的展示**：一位用户建议将**安全会议**作为专门的 LLM 演讲的更合适场所，并提到了漏洞检测（exploit detection）的使用案例。这一想法引起了那些认为标准 ML 活动对于安全特定内容过于宽泛的人的共鸣。
   - 其他人支持突出以领域为中心的方法，指出在这些专业论坛中讨论 LLM 研究的呼声日益增高。



---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion XL 速度与规格**：一位用户在 Colab 上运行了 **stabilityai/stable-diffusion-xl-base-1.0**，希望寻找类似 **ComfyUI** 显示的每秒迭代次数（iterations per second）等内置指标。
   - 他们强调了多达 **50 inference steps** 的可能性，并指出如果没有专门的工具或自定义日志，这些指标仍然难以获取。
- **虚假代币引发骚乱**：社区成员发现了一个与 **Stability AI** 挂钩的**虚假加密货币**发行，并在[一条警告推文](https://x.com/dango233max/status/1879734940264481006)中确认其为诈骗。
   - 他们警告说，**被盗账号**可能会欺骗毫无防备的投资者，并分享了个人损失的经历，敦促大家远离可疑链接。
- **AI 图像分享趋向社交化**：用户讨论了在哪里发布 **AI 生成的图像**，建议将 **Civitai** 和其他社交媒体作为展示成功和失败案例的主要平台。
   - 在收集图像反馈时，人们对**数据质量**产生了担忧，引发了关于如何过滤掉虚假或低质量内容的讨论。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Xeno 推动 Agent 身份变革**：在纽约 Betaworks 举办的 24 日交流会和 25 日黑客松上，来自 [Xeno Grant](https://lu.ma/5rlcrlpb) 的 **$5k** 奖金开启了新一波 **Agent identity** 项目。
   - 获胜者每个 Agent 可获得价值 **$10k** 的 *$YOUSIM* 和 *$USDC*，显示出黑客松参与者对身份解决方案的浓厚兴趣。
- **Ndea 侧重程序合成 (Program Synthesis)**：François Chollet 介绍了 [Ndea](https://ndeainc.com)，旨在启动**深度学习引导的程序合成**，目标是为真正的 AI 发明开辟一条新路径。
   - 社区将其视为一种脱离常规 LLM 扩展（scaling）趋势的方法，一些人称赞它是追求 AGI 的有力替代方案。
- **Cerebras 攻克芯片良率难题**：[Cerebras](https://cerebras.ai/blog/100x-defect-tolerance-how-cerebras-solved-the-yield-problem) 声称已经破解了晶圆级（wafer-scale）芯片的良率问题，生产出的器件比通常大 50 倍。
   - 通过反转传统的良率逻辑，他们构建了容错设计，从而控制了制造成本。
- **MATH 数据集遭遇 DMCA 打击**：[MATH 数据集](https://huggingface.co/datasets/hendrycks/competition_math/discussions/5)面临 **DMCA 移除通知**，引发了对 AoPS 保护的数学内容的担忧。
   - 一些人建议剥离 AoPS 部分以挽救部分使用权，但对更广泛的数据集损失仍存顾虑。
- **MVoT 在图像中展示推理**：新的 [Multimodal Visualization-of-Thought (MVoT) 论文](https://arxiv.org/abs/2501.07542v1)提出在 MLLM 的 **Chain-of-Thought** 提示中加入视觉步骤，将文本与图像结合以优化解决方案。
   - 作者建议，描绘心理图像可以改善复杂的推理流，并能很好地与 *reinforcement learning* 技术融合。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Claude 的独特个性与 Fine-Tuning 成果**：成员们称赞 **Claude** 具有“酷酷的同事感”，指出了其偶尔拒绝回答的情况，并交流了结合公司知识进行高级 **Fine-Tuning** 以及分类任务的技巧。
   - 他们强调了**数据多样性（data diversity）**对提升模型准确性的重要性，并提出了通过*定制化*方法来改善结果。
- **数据集辩论与 Nous Research 的私有化道路**：社区对 **LLM dataset** 的可靠性提出质疑，强调了推动更好数据清洗（data curation）的重要性，并澄清了 **Nous Research** 通过私募股权和周边销售运营的情况。
   - 他们对开源合成数据（synthetic data）计划表示出兴趣，提到了与 Microsoft 的*合作*，但目前没有正式的政府或学术联系。
- **Gemini 表现优于 mini 模型**：多位用户称赞 **Gemini** 在准确提取数据方面的表现，声称在精准定位原始内容方面，它优于 **4o-mini** 和 **Llama-8B**。
   - 他们对*可检索性挑战（retrievability challenges）*保持谨慎，将下一步重点放在稳定的扩展上。
- **Grokking 调整与优化器对决**：参与者剖析了 **grokking** 以及 **Softmax** 中的数值问题，引用了关于 **Softmax Collapse** 及其对训练影响的[这篇论文](https://arxiv.org/abs/2501.04697)。
   - 他们权衡了结合来自[此 GitHub 仓库](https://github.com/cognitivecomputations/grokadamw)的 **GrokAdamW** 和 **Ortho Grad** 的方案，并提到了来自 [Facebook Research 的 Coconut](https://github.com/facebookresearch/coconut)，用于连续潜空间推理（continuous latent space reasoning）。
- **Agent 身份黑客松征集创意**：纽约市宣布举办一场充满活力的 **hackathon**，为 **Agent** 身份原型提供 **$5k** 奖金，旨在培养富有想象力的 AI 项目。
   - 创作者暗示了*新鲜概念*，并引导感兴趣的人士查看[这条推文](https://x.com/vintrotweets/status/1879582102112424356)以获取活动详情。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt 中的标题修改**：随着 **Bolt 更新**，用户现在可以轻松重命名项目标题，详情见[此官方帖子](https://x.com/stackblitz/status/1879625416706785365)。
   - 此次推出简化了项目组织，有助于在列表中更高效地定位项目，提升了用户体验。
- **GA4 集成故障**：一位开发者的 **React/Vite** 应用在 Netlify 上调用 **GA4 API** 时遇到了“Unexpected token”错误，尽管在本地运行正常。
   - 他们验证了凭据和环境变量，但正在寻找替代解决方案以绕过这一集成障碍。
- **Firebase 的快速填充技巧**：一位用户建议创建一个“加载演示数据”页面来无缝填充 **Firestore**，从而避免空 Schema 的麻烦。
   - 这种方法被认为是一种简单但有效的方法，特别是对于那些可能忽略初始数据集设置的人来说非常有益。
- **Supabase 的失误与快照**：一些用户在存储数据时遇到了 **Supabase** 集成错误和应用崩溃。
   - 他们还讨论了[聊天历史快照系统（chat history snapshot system）](https://github.com/stackblitz-labs/bolt.diy/pull/444)，旨在保存之前的状态以实现更好的上下文恢复。
- **Token 之争**：出现了高使用量的报告，其中一个案例声称每个 Prompt 消耗了 **400 万个 tokens**，其他人对其真实性表示怀疑。
   - 社区建议提交 GitHub issue，因为有人怀疑 Bolt 的上下文机制（context mechanics）中潜伏着 Bug。

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **QuPath 照亮组织数据**：一位用户报告称，**NotebookLM** 根据数字病理学论坛的帖子，生成了一个功能齐全的 **Groovy script**，用于 **QuPath**，从而节省了数小时的手动编码时间。
   - 这一成功案例凸显了 **NotebookLM** 在专业任务中的实用性，一位用户称其为“硬编码病理学工作流”中备受欢迎的时间节省利器。
- **世界观构建令作家惊叹**：一位用户利用 **NotebookLM** 进行 **worldbuilding**（世界观构建）的创意扩展，指出它能理清不够完善的背景设定（lore）并找回被忽视的想法。
   - 他们添加了类似 *“大胆预测即将到来！”* 的笔记，以激发 AI 的想象力输出，从而毫不费力地推动更深层次的虚构场景。
- **NotebookLM Plus：神秘的迁移**：关于 **NotebookLM Plus** 在不同 **Google Workspace** 方案中的可用性和过渡时间表出现了混乱，特别是对于那些使用已弃用版本的用户。
   - 一些用户在继续为旧版附加组件付费的同时，也在权衡是否根据 [Google Workspace Blog](https://workspace.google.com/blog/product-announcements/empowering-businesses-with-AI) 含糊不清的公告来升级方案。
- **API：批量同步指日可待？**：用户询问 **NotebookLM** 是否提供 **API** 或能否批量同步 Google Docs 源，目前尚未提供官方时间表。
   - 社区成员参考了 [NotebookLM Help](https://support.google.com/notebooklm/answer/14276471?hl=en&sjid=13501328390293499756-AP) 中的用户请求，对今年的公告保持期待。
- **YouTube 导入困扰与字数限制警告**：多位成员在将 **YouTube links** 导入为有效来源时遇到困难，怀疑是功能缺失而非用户操作错误。
   - 他们还发现每个来源有 **500,000 词** 的限制，且每个笔记本总共只能有 **50** 个来源，这迫使他们不得不进行手动网站抓取或其他变通方法。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 将上下文提升至 128k**：Cohere 将上下文长度扩展至 **128k tokens**，使得在单次对话中可以处理约 **42,000 词** 而无需重置上下文。参与者参考了 [Cohere 的速率限制文档](https://docs.cohere.com/v2/docs/rate-limits) 以了解此次扩展对更广泛模型使用的影响。
   - 他们注意到整个聊天时间线可以保持激活状态，这意味着较长的讨论可以在不分段的情况下保持连贯性。
- **Rerank v3.5 引发关注**：一些用户报告称，**Cohere** 的 **rerank-v3.5** 除非仅限于最近的用户查询，否则输出结果不一致，这使多轮排序工作变得复杂。
   - 他们尝试了 **Jina.ai** 等其他服务，获得了更稳定的结果，并就性能下滑向 Cohere 进行了直接反馈。
- **Command R 获得持续维护**：成员们寻求对 **Command R** 和 **R+** 的迭代增强，希望通过新数据和微调来进化模型，而不是发布全新的版本。
   - 一位贡献者强调 **检索增强生成 (RAG)** 是将更新信息引入现有模型架构的一种强大方法。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Mistral 的 FIM 奇迹**：来自 **Mistral** 的最新 **Fill-In-The-Middle** 编程模型已经发布，拥有超越标准代码补全的高级功能。[OpenRouterAI](https://x.com/OpenRouterAI/status/1879311582246977743) 确认该模型的请求目前仅在其 [Discord 频道](https://discord.gg/fVyRaUDgxW)处理。
   - 爱好者们提到了改进的代码片段上下文处理能力，一些人期待它在代码任务中有强劲表现。其他人则指出 [OpenRouter 的 Codestral-2501 页面](https://openrouter.ai/mistralai/codestral-2501) 是其强大编程潜力的证据。
- **Minimax-01 的 4M 上下文壮举**：**Minimax-01** 被宣传为该团队的首个开源 LLM，据报道它在巨大的 **4M 上下文**下通过了 **Needle-In-A-Haystack 测试**。详情出现在 [Minimax 页面](https://openrouter.ai/minimax/minimax-01)，用户评价称赞其广泛的上下文处理能力。
   - 一些人认为 4M Token 的说法过于大胆，但支持者表示目前尚未看到明显的性能权衡。访问同样需要在 [Discord 服务器](https://discord.gg/fVyRaUDgxW)提出请求，这显示出人们对更大上下文范围日益增长的兴趣。
- **DeepSeek 风波：延迟与 Token 缩减**：成员们反映了持续的 **DeepSeek** API 不一致问题，报告称多个提供商的响应时间缓慢且出现意外错误。许多人对 Token 限制在未通知的情况下从 **64k** 降至 **10-15k** 表示沮丧。
   - 评论者指向 [DeepSeek V3 Uptime and Availability](https://openrouter.ai/deepseek/deepseek-chat/uptime) 页面以寻求部分解释，同时指出首字延迟（first-token latency）仍然持续偏高。其他人担心这些波动会破坏对长上下文使用的信任。
- **提供商对决与模型移除传闻**：一位用户对 **lizpreciatior/lzlv-70b-fp16-hf** 的消失表示担忧，得知可能已没有提供商再托管它。与此同时，参与者讨论了 **DeepSeek**、**TogetherAI** 和 **NovitaAI** 之间的性能差距，引用了 [OpenRouter 网站](https://openrouter.ai/)上的延迟差异。
   - 一些人发现 **DeepInfra** 更可靠，而其他人则看到所有提供商都出现了峰值。这引发了关于提供商在极短通知下轮换或移除模型端点频率的更广泛讨论。
- **Prompt Caching 问答**：多位用户询问 **OpenRouter** 是否支持 **Claude** 等模型的 Prompt Caching，并引用了[文档](https://openrouter.ai/docs/prompt-caching)。他们希望缓存能大幅降低成本并提高吞吐量。
   - Toven 提供了一个有用的指引，确认该功能确实可用，一些开发者称赞其稳定了项目预算。聊天中还分享了关于[请求处理和流取消](https://openrouter.ai/docs/requests#stream-cancellation)的进一步阅读材料。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **神经反馈助力个性化学习**：一位用户提出了一种**神经反馈循环（neural feedback loop）**系统，引导个人采用优化的思维模式以获得更好的认知表现，不过尚未分享官方发布日期。
   - 其他人认为这是 AI 辅助学习的根本性转变，尽管目前还没有相关的链接或代码参考。
- **Anthropic 获得 ISO 42001 负责任 AI 认证**：Anthropic 宣布通过了 [ISO/IEC 42001:2023 标准](https://www.iso.org/standard/81230.html)的**负责任 AI** 认证，强调结构化的系统治理。
   - 用户认可该标准的公信力，但对 Anthropic 与 **Anduril** 的合作表示质疑。
- **共享图像导致 AI 记忆不足**：参与者观察到，在引入图像后，**AI** 经常丢失长上下文，导致需要重复说明。
   - 一位用户建议图像可能会从短期存储中掉出，导致模型忽略了之前的参考内容。
- **ChatGPT 分级导致性能不均**：社区成员注意到 **ChatGPT** 对免费用户似乎有所限制，尤其是在网页搜索方面。
   - 他们指出 **Plus** 订阅者获得了更高级的功能，这引发了 **API** 用户对公平性的讨论。
- **GPT-4o 任务功能超越 Canvas 工具**：多位用户报告称，桌面版中的 **Canvas** 功能被任务界面取代，尽管通过工具箱图标仍可启动 Canvas。
   - 他们强调 **GPT-4o** 任务为语言练习或新闻更新等行动提供了定时提醒。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **微调热潮与公有领域佳作**：一位用户正在利用**公有领域文本**微调 **LLMs** 以提升输出质量，在首次探索 **Python** 的同时将工作重心转移到了 **Google Colab**。
   - 他们的目标是塑造 Prompt 以获得更好的写作效果，专注于探索利用 **LLMs** 进行创意任务的新方法。
- **上下文紧缺与内存变动**：成员们注意到，当长对话超出 **LLM** 的缓冲区时，会弹出“上下文已满 90.5%”的警告，从而面临输出被截断的风险。
   - 他们讨论了调整上下文长度与增加**内存占用（memory footprints）**之间的权衡，强调了为了稳定性能而进行的微妙平衡。
- **GPU 速度对决：2×4090 vs A6000**：据报告，**2x RTX 4090** 配置的速度为 **19.06 t/s**，超过了 **RTX A6000 48GB** 的 **18.36 t/s**，不过有一项修正建议 A6000 的速度应为 **19.27 t/s**。
   - 爱好者们还称赞了 **2x RTX 4090** 方案显著降低的功耗，表明其在性能和效率上均有提升。
- **并行化难题与层分布**：讨论探索了将模型拆分到多个 **GPU** 上，将一半的层放置在每张显卡上以进行同步计算。
   - 然而，参与者指出 **PCIe** 潜在的延迟和更重的同步负担是实现明显速度优势的障碍。
- **快照故障：LLMs 与图像分析**：一些用户在让 **QVQ-72B** 和 **Qwen2-VL-7B-Instruct** 正确解析图像时遇到困难，面临初始化错误。
   - 他们强调了保持**运行时环境（runtime environments）**更新的重要性，并指出缺失依赖项经常会导致图像处理尝试失败。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek V3 运行缓慢，GPU 讨论升温**：多位成员报告 **DeepSeek V3** 运行缓慢或卡住，促使一些人转向 **Sonnet** 以获得更好性能，并在 [此 HF 线程](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/discussions/5) 中分享了他们的挫折感。
   - 一位用户强调 **RTX 4090** 理论上可以胜任更大的模型，并引用了 [SillyTavern 的 LLM 模型 VRAM 计算器](https://sillytavernai.com/llm-model-vram-calculator/)，提出了关于 **VRAM** 需求的问题。
- **Aider 获得赞誉，提交停滞**：一位用户称赞了 **Aider** 的代码编辑能力，但抱怨尽管设置正确且使用了克隆的项目，却没有生成 **Git commits**。
   - 其他人建议使用 **architect mode** 在提交前确认更改，并引用了旨在解决这些问题的 PR [#2877](https://github.com/Aider-AI/aider/pull/2877)。
- **仓库图谱膨胀，Agent 工具介入**：一位成员注意到他们的 **repository-map** 从 **2k** 行增长到了 **8k** 行，引发了对一次性处理超过 **5 个文件**时效率的担忧。
   - 用户建议使用 **cursor's chat** 和 **windsurf** 等 **agentic** 探索工具来扫描代码库，并称赞 **Aider** 完成了最终的实现步骤。
- **Repomix 打包代码，减少 API 体积**：一位用户展示了 [Repomix](https://repomix.com)，它可以将代码库重新打包成对 **LLM** 驱动的任务更友好的格式。
   - 他们还提到了与 **Repopack** 的协同作用，以最大限度地减少 **Aider** 的 **API** 调用，从而减少大型项目的 **token** 开销。
- **标题党调侃，o1-preview 进展缓慢**：一位用户取笑另一位用户的 AI 内容风格“简直像二手车推销员”，呼应了社区对标题党推广的厌烦。
   - 其他人提到 **o1-preview** 的响应变慢且 **token** 消耗更高，指出性能下降阻碍了实时交互。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **文档字体变粗**：文档字体现在更粗了，以提高可读性，用户在 [#general](https://discord.com/channels/1087530497313357884/1098713601386233997/1329167856549625856) 中称赞其**好得多**。
   - 用户似乎对**进一步的调整**持开放态度，表现出持续优化用户体验的意愿。
- **Mojo 暂无 Lambda 语法**：社区确认 **Mojo** 目前缺乏 `lambda` 语法，但一份 [路线图说明](https://docs.modular.com/mojo/roadmap/#no-lambda-syntax) 预示了未来的计划。
   - 在 Lambda 得到正式支持之前，有人建议将命名函数作为参数传递。
- **Zed 与 Mojo 强强联手**：爱好者们分享了如何像稳定版一样在 Zed Preview 中安装 **Mojo**，设置完成后代码补全功能即可正常工作。
   - 尽管有些人在缺少某些设置时遇到了小障碍，但一旦配置妥当，整体集成非常顺畅。
- **SIMD 引发性能瓶颈**：参与者警告了 **SIMD** 的性能陷阱，并引用了 [Ice Lake AVX-512 Downclocking](https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html)。
   - 他们敦促检查汇编输出，以检测任何可能抵消各种 CPU 上 SIMD 优势的寄存器重排（register shuffling）。
- **递归类型挑战 Mojo 的耐心**：开发者们正在努力解决 Mojo 中的**递归类型**问题，转而使用指针来处理树状结构。
   - 他们链接了 [GitHub issues](https://github.com/modularml/mojo/issues/3917) 以获取更多细节，并指出语言设计中持续存在的复杂性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **关键 Token 引发热议**：一篇新的 [arXiv 预印本](https://arxiv.org/abs/2411.19943) 介绍了用于 LLM 推理的**关键 Token (critical tokens)**，显示出在精心管理这些关键 Token 时，GSM8K 和 MATH500 的准确率大幅提升。成员们还澄清了 **VinePPO** 实际上并不需要示例 Chain-of-Thought 数据，尽管离线 RL 的比较仍存在激烈争论。
   - 他们接受了选择性降低这些 Token 权重可以提升整体性能的观点，社区注意到这与其它 **implicit PRM** 的发现有相似之处。
- **NanoGPT 打破速度记录**：据报告，在 modded-nanoGPT 上实现了一次创纪录的 **3.17 分钟** 训练运行，该运行结合了新的 Token 相关 **lm_head** 偏置和多个融合操作（fused operations），详见 [此 pull request](https://github.com/KellerJordan/modded-nanogpt/pull/71)。
   - 进一步的优化想法，如 **Long-Short Sliding Window Attention**，被提出以进一步提升速度和性能。
- **TruthfulQA 遭遇滑铁卢**：[TurnTrout 的推文](https://x.com/Turn_Trout/status/1879710659904254081) 揭示了通过利用几个琐碎规则的弱点，在多选题 TruthfulQA 上达到了 **79% 的准确率**，从而绕过了更深层的模型推理。
   - 这一发现引发了社区辩论，凸显了基准测试的缺陷如何削弱 **halueval** 等其他数据集的可靠性。
- **MATH 数据集 DMCA 删帖风波**：由于 **DMCA 移除**，**Hendrycks MATH** 数据集已下架，正如 [此 Hugging Face 讨论](https://huggingface.co/datasets/hendrycks/competition_math/discussions/5) 所述，这引发了法律和物流方面的担忧。
   - 成员们追溯到原始问题源自 **AOPS**，重申这些谜题类内容从一开始就注明了归属，突显了数据集许可方面的摩擦。
- **Anthropic 与 Pythia 电路分析揭秘**：多次引用 **Anthropic** 的电路分析，探讨了子网络如何在不同 **Pythia** 模型的连贯训练阶段形成，如 [此论文](https://arxiv.org/abs/2407.10827) 所述。
   - 参与者指出，这些涌现结构并不严格符合简单的训练损失（dev-loss）与计算量（compute）图表，强调了内部架构演变的细微差别。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cursor AI 获得巨额融资**：他们完成了由 **a16z** 领投的新一轮 **Series B** 融资，以支持该编程平台的下一阶段发展，详见[此公告](https://x.com/sarahdingwang/status/1879279307119608142)。
   - 社区讨论指出 **Cursor AI** 是 **Anthropic** 的关键客户，这引发了关于基于用量定价（usage-based pricing）的热议。
- **Transformer² 像章鱼一样灵活适应**：来自 **Sakana AI Labs** 的新[论文](https://arxiv.org/abs/2501.06252)引入了动态权重调整，连接了预训练（pre-training）与后训练（post-training）。
   - 爱好者将其比作章鱼如何融入周围环境，强调了其在特定任务中自我改进的潜力。
- **OpenBMB MiniCPM-o 2.6 进军多模态**：[MiniCPM-o 2.6](https://x.com/_philschmid/status/1879163439559389307) 的发布展示了一个 **8B-parameter** 模型，能够在边缘设备上处理视觉、语音和语言任务。
   - 初步测试称赞了其双语语音性能和跨平台集成，引发了对其在现实场景应用的乐观预期。
- **Curator：按需生成合成数据**：新的 [Curator 库](https://x.com/madiator/status/1879579213554147665)提供了一种开源方法，为 LLM 和 RAG 工作流生成训练和评估数据。
   - 工程师们预计这将填补后训练数据流水线中的空白，并计划推出更多功能以实现更全面的覆盖。
- **NVIDIA Cosmos 在 CES 亮相**：在 [LLM Paper Club](https://lu.ma/pvh0rwa3) 上，**NVIDIA Cosmos** 在 CES 发布后被重点介绍，展示了其各项能力。
   - 与会者被敦促注册并将该环节添加到日历中，以免错过这一新模型的揭秘。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 与 Torch：依赖关系之舞**：一位用户发现 **Triton** 依赖于 **Torch**，这使得纯 **CUDA** 工作流变得复杂，并引发了关于是否存在 **cuBLAS** 等效项的讨论（[文档](https://triton-lang.org/main/index.html)）。
   - 另一位用户遇到了指针类型不匹配导致的 **ValueError**，结论是 `tl.load` 中的指针必须是 **float** 标量。
- **RTX 50x TMA 传闻**：有传言称 **RTX 50x Blackwell** 显卡可能会继承 **Hopper** 的 **TMA**，但目前尚无确切细节。
   - 社区成员在白皮书（whitepaper）发布前仍感焦虑，这让关于 TMA 的讨论热度不减。
- **MiniMax-01 拥有 4M-Token 上下文**：**MiniMax-01** 开源模型引入了 **Lightning Attention**，能够处理高达 **4M tokens** 的内容，且性能大幅提升。
   - API 价格为 **每百万输入 token 0.2 美元**，**每百万输出 token 1.1 美元**，详见其[论文](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf)和[新闻页面](https://www.minimaxi.com/en/news/minimax-01-series-2)。
- **Thunder Compute：廉价 A100 风暴**：**Thunder Compute** 首次亮相，提供 **$0.92/小时** 的 **A100** 实例，并赠送 **$20/月** 的额度，详见其[官网](https://thundercompute.com)。
   - 该项目由 **Y Combinator** 校友支持，并提供 CLI 工具（`pip install tnr`）用于快速管理实例。
- **GPU 使用技巧与调试故事**：工程师们强调了在 **bfloat16** 训练中 **weight decay** 的重要性（[图 8](https://arxiv.org/pdf/2310.04415)），并讨论了在 **Torch** 中批量调用 `.to(device)` 以减少 CPU 开销。
   - 他们还探讨了多 GPU 推理策略、**MPS** 内核性能分析（profiling）的怪癖，以及用于 popcorn bot 的专用 GPU 装饰器，参考了 [deviceQuery 信息](https://stackoverflow.com/questions/40695455/what-utility-binary-can-i-call-to-determine-an-nvidia-gpus-compute-capability)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **使用 LlamaParse 开启 RAG 热潮**：利用 [LlamaParse](https://twitter.com/llama_index/status/1879327934378627212)、**LlamaCloud** 和 **AWS Bedrock**，该小组构建了一个专注于高效解析 **SEC documents** 的 **RAG application**。
   - 他们的分步指南概述了处理大型文档的高级索引策略，同时强调了这些平台之间的强大协同作用。
- **利用 LlamaIndex 提升知识图谱收益**：来自 **@neo4j** 的 Tomaz Bratanic 在其[详尽的帖子](https://twitter.com/llama_index/status/1879596647967306122)中介绍了一种使用 **LlamaIndex** 的 **agentic strategies** 来提高知识图谱准确性的方法。
   - 他从朴素的 **text2cypher** 模型转向了稳健的 **agentic workflow**，通过精心设计的错误处理提升了性能。
- **LlamaIndex 与 Vellum AI 联手**：**LlamaIndex** 团队宣布与 **Vellum AI** 建立合作伙伴关系，并在此处分享了他们的调查用例发现 [here](https://twitter.com/llama_index/status/1879652991139278861)。
   - 此次合作旨在扩大他们的用户社区，并探索 **RAG-powered** 解决方案的新策略。
- **利用 Chromium 解决 XHTML 转 PDF 难题**：一位成员指出 **Chromium** 在将 **XHTML** 转换为 **PDF** 方面表现出色，优于 **pandoc**、**wkhtmltopdf** 和 **weasyprint** 等库。
   - 他们分享了一个 [XHTML 文档示例](https://cdn.financialreports.eu/financialreports/media/filings/3843/2024/10-K/3843_10-k_2024-07-09-133930_ed8fec32-9559-4136-8b93-024a1ba01ffd.xhtml) 和一个 [HTML 文档示例](https://cdn.financialreports.eu/financialreports/media/filings/4700/2024/10-K/4700_10-k_2024-12-19_32fd81af-71d1-46e4-ab48-d86953034226.html)，强调了其出色的渲染忠实度。
- **大规模向量数据库的选择困境**：用户讨论了是否从 **Pinecone** 切换到 **pgvector** 或 **Azure AI search**，以便以更好的成本效益管理 2 万份文档。
   - 他们参考了 [LlamaIndex 的 Vector Store 选项](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/#vector-store-options-feature-support) 来评估与 **Azure** 的集成情况，并强调了建立强大生产工作流的必要性。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 1.0 的命令难题**：**OpenInterpreter 1.0** 限制了直接代码执行，将任务转向命令行使用，这引发了关于失去 **user-friendly** 特性的担忧。
   - 社区成员对背离即时 **Python** 执行表示遗憾，称新方法“感觉更慢”且“需要更多手动步骤”。
- **Bora 定律打破算力惯例**：一篇新的工作论文 [Bora's Law: Intelligence Scales With Constraints, Not Compute](https://chrisbora.substack.com/p/boras-law-intelligence-scales-with) 指出，**intelligence 的指数级增长**是由约束而非 **compute** 驱动的。
   - 与会者强调，这一理论挑战了像 **GPT-4** 这样的大规模建模策略，质疑了对原始硬件资源的过度依赖。
- **OI 中的 Python 增强功能**：爱好者敦促添加 **Python** 便捷函数，以简化 **OpenInterpreter** 中的任务。
   - 他们认为这些增强功能可以在保持平台交互风格的同时“提升用户效率”。
- **AGI 方法受到质疑**：社区的一部分人批评 **OpenAI** 过度关注暴力 **compute**，忽略了更微妙的智能提升因素。
   - 成员们呼吁根据 **Bora's Law** 等创意理论重新评估 AI 开发原则，强调需要优化大模型缩放（scaling）策略。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **DougDoug 对 AI 版权法的深入探讨**：在[这段 YouTube 视频](https://www.youtube.com/watch?v=pt7GtDMTd3k)中，**DougDoug** 详细解释了 **AI copyright law**，重点关注 **tech** 与法律结构之间潜在的交集。
   - 这一观点引发了热烈讨论，参与者赞扬了他对*新兴法律盲点*的关注，并推测了可能的*创作者补偿模型*。
- **超可解释网络引发对版税的重新思考**：一项关于 **hyper-explainable networks** 的提案引入了衡量训练数据对模型输出影响的想法，可能将版税定向给数据提供者。
   - 观点在对数据驱动补偿潜力的*兴奋*与对实施此类系统开销的*怀疑*之间摇摆不定。
- **推理时信用分配获得关注**：关于 **inference-time credit assignment** 的相关对话提出了使用它来追踪每个数据集分块对模型结果影响的可能性。
   - 虽然一些人看到了*认可数据贡献者*的希望，但另一些人指出量化这些影响具有极高的复杂性。

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **P2B 的提议引发加密货币冲突**：一名来自 **P2B** 的代表提供了融资、上市、社区支持和**流动性管理（liquidity management）**等服务，希望吸引 **AI21** 参与其加密货币愿景。
   - 他们询问是否可以分享更多关于这些服务的细节，但在 **AI21 Labs** 明确表达了对加密货币的立场后，对话发生了转变。
- **AI21 Labs 拒绝加密货币倡议**：**AI21 Labs** 坚决拒绝与基于加密货币的努力产生关联，并表示他们永远不会开展相关项目。
   - 他们还警告说，反复提及加密货币将导致迅速封禁，强调了其零容忍的立场。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Agent 将在 2025 年崛起**：Sakshi 和 Satvik 将于 **1 月 16 日星期四晚上 9 点（IST）**主持 [Build Your First AI Agent of 2025](https://lu.ma/y8pp2fxc) 活动，展示来自 [Build Fast with AI](http://www.buildfastwithai.com) 和 [Lyzr AI](http://www.lyzr.ai) 的代码和无代码方法。
   - 该研讨会重点预测了 **AI Agent** 将在 2025 年改变各行各业，为新手和工程师等群体扩大准入门槛。
- **AI 采用过程中的预算博弈**：社区成员强调**成本**是决定采用新解决方案还是保留现有系统的关键因素。
   - 许多人仍保持谨慎，表示相比风险更高、成本更昂贵的安装，他们更倾向于保留经过验证的基础设施。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **对 Qwen 2.5 微调的关注**：一位用户询问了 **Qwen 2.5 Coder Instruct (7B)** 的微调细节，想知道它是否已在 Hugging Face 上发布，并表达了对更大模型的好奇。
   - 他们还寻求其他人在成熟模型上的成功案例，强调了在真实场景中的性能表现。
- **Llama 3.2 在处理长剧本时遇到困难**：一位用户在使用 **Llama 3.2 3B** 分析一份 **45 页的电视试播剧本**时遇到了错误，原本期望它能处理该文本而不会出现字符限制问题。
   - 他们分享了一个[对比链接](https://www.prompthackers.co/compare/llama-3.2-3b/llama-3-8b)，展示了 **Llama 3.2 3B** 和 **Llama 3 8B Instruct** 在 Token 容量和近期发布版本方面的区别。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **推动 Ambient Agent 实现的标准化**：**#general** 频道的一位用户询问了如何使用 **DSPy** 构建 **Ambient Agent**，寻求尝试过此方法的用户的经验和标准化方案。
   - 他们强调了 **Ambient Agent** 与 **DSPy** 工作流之间潜在的协同效应，邀请大家共同投入以寻求更结构化的解决方案。
- **对 DSPy 示例的兴趣日益增长**：另一个关于实现 Ambient Agent 的具体 **DSPy** 示例的询问出现了，强调了社区对具体代码参考的渴望。
   - 目前尚未提供直接示例，但社区对共享演示或开源材料以增强 **DSPy** 的实际应用表现出极大的热情。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**Axolotl AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**Torchtune Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**HuggingFace Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

# 第二部分：按频道划分的详细摘要和链接

{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1328817733680369774)** (568 条消息🔥🔥🔥): 

> `Cursor 性能问题、请求缓慢、部署方法、模型对比、Sora AI 使用` 


- **Cursor 性能与请求缓慢**：用户报告 Cursor 的慢速请求（slow requests）出现显著延迟，等待时间增加至 5-10 分钟，引起了 Pro 订阅者的不满。
   - 虽然部分用户的响应时间正常，但许多人表示持续存在的问题影响了他们的生产力。
- **使用 Cursor 的部署策略**：Vercel 和 Google Firebase 成为部署使用 Cursor 构建的应用的热门选择，用户分享了无缝部署的技巧。
   - 用户注意到，部署应用的 Prompt 通常默认为 Vercel，从而最大限度地减少了额外配置的需求。
- **模型对比与 AI 使用**：讨论中涉及了各种 LLM 的有效性，一些用户因 Gemini 2.0 Flash 在基准测试中表现更好而更倾向于选择它，而非 Llama 3.1。
   - 其他人则对 AI 依赖导致的“冒充者综合征（imposter syndrome）”表示担忧，同时也承认与 AI 协作显著提高了生产力。
- **Sora 与视频生成**：用户提到使用 Sora 进行视频生成的效果参差不齐，在生成高质量慢动作内容时表现不稳定。
   - 在 Sora 等应用上获得持续成功似乎具有挑战性，这促使用户探索其他视频生成选项。
- **Cursor 即将到来的变化**：用户对预计于 3 月发布的 Cursor 新版本充满期待，该版本将包含新的 Fusion 实现。
   - 用户表达了对平台增强功能的期待，并希望集成 DeepSeek 和 Gemini 等模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/MiniMax__AI/status/1879226391352549451">MiniMax (官方) (@MiniMax__AI) 的推文</a>: MiniMax-01 现已开源：为 AI Agent 时代扩展 Lightning Attention。我们很高兴推出最新的开源模型：基础语言模型 MiniMax-Text-01 和视觉模型...</li><li><a href="https://claude.ai">Claude</a>: 与来自 Anthropic 的 AI 助手 Claude 对话</li><li><a href="https://www.cursor.com/blog/tab-update">新的 Tab 模型 | Cursor - AI 代码编辑器</a>: 发布下一代 Cursor Tab 模型。</li><li><a href="https://tenor.com/view/facepalm-really-stressed-mad-angry-gif-16109475">捂脸 GIF - 压力山大 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/omkarthawakar/LlamaV-o1">omkarthawakar/LlamaV-o1 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/sad-cry-crying-girl-crying-upset-gif-13557054590100308198">悲伤哭泣 GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://status.anthropic.com/.">Anthropic 状态</a>: 未找到描述</li><li><a href="https://vercel.com/templates">寻找你的模板</a>: 使用来自 Vercel 和社区的预构建解决方案加速你的应用开发过程。</li><li><a href="https://v0.dev">Vercel 推出的 v0</a>: 与 v0 对话。通过简单的文本 Prompt 生成 UI。复制、粘贴、发布。</li><li><a href="https://www.reddit.com/r/cursor/comments/1hftyho/built_a_cursor_extension_to_save_and_share_chat/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://youtu.be/TCQloeJsMPE?t=1296">我替你试用了“免费”的 GitHub Copilot</a>: 微软宣布 GitHub Copilot 现已成为 VS Code 的免费部分！但它值得使用吗？与 Cursor 等其他 AI 代码编辑器相比如何...</li><li><a href="https://cursor.directory">Cursor Directory</a>: 为你的框架和语言寻找最佳的 Cursor 规则</li><li><a href="https://github.com/chiphuyen/aie-book/blob/main/scripts/ai-heatmap.ipynb">aie-book/scripts/ai-heatmap.ipynb at main · chiphuyen/aie-book</a>: [进行中] AI 工程师资源。还包含《AI Engineering》一书的配套材料 (Chip Huyen, 2025) - chiphuyen/aie-book
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1328848846712995871)** (274 条消息🔥🔥): 

> `Perplexity outages, Performance of AI models, Integration with IDEs, Citations issues, User experiences with AI models` 


- **Perplexity 面临频繁停机**：用户报告了多次 **Perplexity** 宕机的情况，持续时间超过一小时，并遇到了各种错误消息。
   - **状态页面**显示部分停机，导致用户感到沮丧，并促使他们寻找替代方案或解决方法。
- **关于编程最佳 AI 模型的辩论**：在关于 AI 模型的讨论中，**Claude Sonnet 3.5** 被强调为复杂调试的最佳选择，而 **Deepseek 3.0** 则被推荐为更便宜的替代方案。
   - 用户表示 **Perplexity** 在某些任务中可以超越竞争对手，同时也指出了其局限性。
- **Perplexity 在 IDE 中的集成挑战**：一位用户尝试将 **Perplexity** 连接到 **IntelliJ IDE**，但尽管拥有 Pro 版本，仍因 API 访问的额外费用而面临挑战。
   - 建议包括考虑其他 AI 工具，如 **GitHub Copilot**，以获得更好的集成体验。
- **引用问题**：一些用户注意到 **citations**（引用）功能在 **Perplexity** 中无法正常工作，导致检索和验证信息出现困难。
   - 聊天记录显示，存在长期影响链接和引用显示及功能的 Bug。
- **用户体验与挫败感**：几位用户提到 **Perplexity** 的性能不足，特别是在可靠输出和上下文窗口限制方面。
   - 对 AI **hallucinations**（幻觉）和不一致结果的挫败感突显了对服务有效性和可靠性的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://scira.app/">Scira</a>: Scira 是一款极简的 AI 驱动搜索引擎，帮助你在互联网上查找信息。</li><li><a href="https://tenor.com/view/bowling-gif-5724286">Bowling GIF - Bowling - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://status.perplexity.com/">Perplexity - Status</a>: Perplexity 状态</li><li><a href="https://chrisbora.substack.com/p/boras-law-intelligence-scales-with?r=aszci">Bora's Law: Intelligence Scales With Constraints, Not Compute</a>: 这是一篇探讨人工智能发展中新兴原则的工作论文。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1328864449460703335)** (7 条消息): 

> `iPhone Air Rumors, JavaScript Trademark Battle, US AI-Export Rules, Perplexity AI Apps Confusion` 


- **令人兴奋的 iPhone Air 传闻浮现**：关于 **iPhone Air** 的新传闻已经出现，引发了对设计和价格点的猜测。你可以在[这里](https://www.youtube.com/embed/h6GgNh4RQoQ)观看完整讨论。
   - *社区反应不一*，许多人渴望获得最新功能。
- **JavaScript 商标之战升温**：一场激烈的法律纠纷围绕 **JavaScript** 商标展开，可能影响各种项目。其影响可能会重塑技术领域的 **open-source** 贡献。
   - 成员们分享了他们的看法，强调了这场战斗如何凸显了编程语言的 **ownership**（所有权）问题。
- **讨论美国新的 AI 出口规则**：关于美国旨在监管技术分发的 **new AI-export rules** 的讨论正在进行中。这些法规的关键方面可能会影响全球协作。
   - 专家警告说，**developers** 和国际合作伙伴可能会产生反对意见。
- **对两个 Perplexity AI 应用的混淆**：一位用户对在 **iOS App Store** 中发现两个描述没有明确区分的应用表示惊讶。这引发了在 [Perplexity web app](https://www.perplexity.ai/search/why-are-there-two-perplexity-a-8DCudsGDRdaCTh4fX6YAhQ#1) 上的查询以寻求澄清。
   - 另一位用户无法找到第二个应用，引发了关于哪些应用可用的进一步讨论。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1328903571051118665)** (1 条消息): 

> `llama-3.1-sonar-large-128k-online output speed` 


- **用户反馈 llama-3.1-sonar-large-128k-online 输出速度变慢**：成员们注意到自 **1月10日** 以来，**llama-3.1-sonar-large-128k-online** 的输出速度出现了 **大幅下降**。
   - 这一速度下降引发了用户关于可能影响性能的潜在原因或变更的讨论。
- **对 llama 性能下降的担忧**：另一位用户表示担心输出速度的下降可能会影响模型的整体性能和用户体验。
   - 社区成员正在积极讨论潜在的故障排除步骤和替代方案。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1328915006284238982)** (2 条消息): 

> `Windsurf Command Tutorial, Discord Challenge Winners, Student Discount Pricing, Windsurf Editor Launch, Codeium vs GitHub Copilot` 


- **学习如何高效使用 Command**：发布了一个新的视频教程，详细介绍了如何使用 **Command** 功能 *直接在编辑器中* 生成或编辑代码 [点击观看](https://x.com/windsurf_ai/status/1879332883489591404)。
   - 这是用户通过该功能的宝贵见解来增强工作流的机会。
- **Discord 挑战赛获胜者公布！**：恭喜获胜者：<@149171016705376256> 和 <@1023819475650347008>！查看他们的获奖视频，包括 *[Luminary - 免费 AI 工作流工具](https://www.youtube.com/watch?v=LyOMJq47ASQ)* 和 *[如何真正通过 Windsurf 赚钱](https://youtu.be/6rwbcgEM25g)*。
   - 鼓励获胜者私信领取 **3 个月 Windsurf Pro 层级** 的奖励。
- **学生可获得大幅折扣**：拥有有效 **.edu** 邮箱地址的学生现在可以在限时内享受 Windsurf Pro 层级的 **重大折扣**，只需在 [codeium.com](https://www.codeium.com) 注册即可。
   - 这一举措旨在让学生更容易获得该工具，并提升他们的编码能力。
- **推出 Windsurf Editor**：全新的 **Windsurf Editor**（一款专为 AI 打造的 IDE）已经发布，旨在为用户提供流畅的编码体验。用户可以在平台上了解更多关于其功能和优势的信息。
   - 官方提供了 **Codeium** 与 **GitHub Copilot** 的对比，展示了其卓越的性能。
- **高质量训练数据保证**：Codeium 向用户保证，它不会在非许可代码（例如 GPL）上进行训练，从而保护用户免受潜在的法律风险，[这篇博客文章](https://www.codeium.com/blog/copilot-trains-on-gpl-codeium-does-not) 强调了这一点。
   - 该平台旨在提供高质量、安全的 AI 工具，以简化工程师的编码过程。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.codeium.com">Windsurf Editor 和 Codeium 扩展</a>: Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。也是首个 Agentic IDE —— Windsurf 的开发者。</li><li><a href="https://x.com/windsurf_ai/status/1879332883489591404">来自 Windsurf (@windsurf_ai) 的推文</a>: 如何使用 Command</li><li><a href="https://www.youtube.com/watch?v=LyOMJq47ASQ">Luminary - 免费 AI 工作流工具</a>: Luminary 是一款免费的开源 AI 工作流工具。喜欢你看到的吗？在 GitHub 上给 Luminary 点个 star 吧！https://github.com/nascarjake/luminary 有疑问吗？加入 m...</li><li><a href="https://youtu.be/6rwbcgEM25g">如何真正通过 Windsurf 赚钱 #aiautomation #firebringerai #coding #seoautomation</a>: 使用这款改变游戏规则的工具在几分钟内构建 SEO 网站。停止花费数小时甚至数天手动构建 SEO 网站。这款工具可以将你的关键词转化为...
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1328845522735333457)** (88 条消息🔥🔥): 

> `Codeium Telemetry 问题, Codeium 订阅计划, 学生折扣, 远程仓库利用, Codeium 安装问题` 


- **Codeium Telemetry 困扰**：用户报告了 Visual Studio 的 Codeium 扩展中 **telemetry** 的问题，有人建议在 [GitHub](https://github.com/Exafunction/CodeiumVisualStudio/issues/111) 上报告该问题。
   - *有人强调 Codeium Visual Studio 与 Visual Studio Code 不同*，并建议核实安装设置。
- **关于 Codeium 订阅计划的困惑**：出现了关于 Codeium 订阅计划中 **credit usage**（额度使用）的问题，特别是取消订阅后额度是否结转，或者是否每月重置。
   - 用户确认在计划取消后额度不会结转，未使用的额度将在计划结束时失效。
- **寻求学生折扣的明确说明**：几位用户询问了 Pro Ultimate 计划的 **student discounts**（学生折扣），一些非传统 .edu 域名的邮箱用户在获取折扣时遇到了困难。
   - 管理员指出，他们正在努力将资格扩大到 .edu 地址以外，但目前只有 .edu 账户符合折扣条件。
- **关于利用远程仓库的担忧**：一位用户表示在 IntelliJ 中通过 Codeium **利用已索引的远程仓库**存在困难，并寻求设置指南。
   - 社区鼓励大家分享远程仓库的使用经验以互相帮助。
- **Codeium 安装挑战**：用户报告了 **安装 Codeium** 时的错误，特别是由于显示名称过长导致的扩展清单（extension manifest）问题。
   - 其他人建议直接联系支持部门以解决持续存在的安装问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/usage">Paid Plan and Credit Usage - Codeium Docs</a>: 未找到描述</li><li><a href="https://github.com/Exafunction/codeium">GitHub - Exafunction/codeium</a>: 通过在 GitHub 上创建账户来为 Exafunction/codeium 的开发做出贡献。</li><li><a href="https://github.com/Exafunction/CodeiumVisualStudio/issues/111.">Exafunction/CodeiumVisualStudio</a>: Codeium 的 Visual Studio 扩展。通过在 GitHub 上创建账户来为 Exafunction/CodeiumVisualStudio 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1328817474791276544)** (191 条消息🔥🔥): 

> `Windsurf IDE 问题、折扣与定价、Cascade 用户体验、C# 变量类型分析、AI 模型集成` 


- **Windsurf IDE 在变量类型分析方面存在困难**：用户报告称，即使在 Windows 和 Mac 系统的最新版本中，Windsurf IDE 内的 C# 文件变量类型分析仍存在持续性问题。
   - 这与 VS Code 和 Cursor IDE 等其他 IDE 中流畅的体验形成鲜明对比。
- **新用户折扣困扰**：多位用户在注册时未能获得预期的折扣，特别是使用大学邮箱账号的用户。
   - 其他用户建议，联系支持团队并提供工单编号可以加快解决速度。
- **对 Windsurf 和 Cascade 性能的评价褒贬不一**：部分用户对 Cascade Base 表示满意，称其作为免费 AI 工具非常有效；而另一些用户则对应用卡顿和性能问题感到沮丧。
   - 高级用户建议使用详细的 Prompt 来提升 Windsurf 的表现。
- **AI 模型集成讨论**：用户对在 Windsurf 中集成各种 AI 模型和工具表现出浓厚兴趣，并强调了 Claude 在解决高级问题方面的能力。
   - 一些人建议结合使用多种模型可能会带来更好的结果，尤其是在处理复杂的编程任务时。
- **用户对定价结构的反馈**：用户分享了对 Windsurf 定价的看法，部分用户希望在额度（credits）和方案上能有更多灵活性，以更好地匹配实际使用情况。
   - 用户呼吁对定价结构进行更清晰的沟通，并对现有方案进行潜在改进。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/windsurf_ai/status/1866600392165048329">来自 Windsurf (@windsurf_ai) 的推文</a>：很高兴宣布我们的首个周边抽奖活动 🏄 分享你用 Windsurf 构建的作品，就有机会赢取护理包 🪂 #WindsurfGiveaway 必须关注才能获得资格</li><li><a href="https://tenor.com/view/power-starwars-unlimited-power-gif-15939349">Power Starwars GIF - Power Starwars Unlimited Power - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/text-phone-waiting-hurry-messenger-gif-4073783462256955308">Text Phone GIF - Text Phone Waiting - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>：需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://www.codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>：需要帮助？联系我们的支持团队以获取个性化协助。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1328818619764703315)** (113 条消息🔥🔥): 

> `Unsloth 中的 GPU 使用情况、微调误区、Notebook 中的模型训练、与 Kaggle 的合作、使用 Unsloth 进行网页抓取` 


- **在多 GPU 上运行 Unsloth**：用户询问了如何在 Kaggle 中使用两个 T4 GPU 运行 Unsloth，但目前推理似乎仅支持单个 GPU。
   - 讨论了使用 Kaggle 进行训练任务，建议利用每周提供的免费时长。
- **解决微调误区**：有人指出，许多人认为微调不会为模型引入新知识，而文档澄清了微调可以复制检索增强生成 (RAG) 的功能。
   - Unsloth 团队提到，他们正在文档中重点关注这些误区，以便让用户更好地了解微调的好处。
- **Notebook 中的训练任务**：用户分享了在 Google Colab 中进行长时间训练任务的经验，指出了 OOM 错误以及在漫长过程中可能出现的 Notebook 断连问题。
   - 讨论强调了对于高要求的训练任务，需要更强大的环境，例如配备 80GB VRAM 的 A100 GPU。
- **与 Kaggle 合作开发 Notebook**：Unsloth 团队正致力于为 Phi-4 模型创建一个 Kaggle Notebook，以增强用户可用的工具。
   - 此次合作旨在提供专门用于高效训练和微调模型的资源。
- **网页抓取模型推荐**：一位用户询问了网页抓取任务的模型推荐，引发了关于使用 Firecrawl 进行特定网站抓取的讨论。
   - 建议通过使用专门为此目的设计的工具来有效管理网页抓取。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/helloiamleonie/status/1879511537343434817">Leonie (@helloiamleonie) 的推文</a>：在假期期间，我学习了如何微调 LLM。这是我参加最新 @kaggle 竞赛的作品。本教程向您展示：• 微调 Gemma 2 • 在 T4 GPU 上使用 @UnslothAI 进行 LoRA 微调 • 实验...</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing)">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/is-fine-tuning-right-for-me">微调适合我吗？ | Unsloth 文档</a>：如果您不确定微调是否适合您，请看这里！</li><li><a href="https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md#glossary">nod-ai/shark-ai 的 amdgpu_kernel_optimization_guide.md</a>：SHARK 推理建模与服务。通过在 GitHub 上创建账号为 nod-ai/shark-ai 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=LPZh9BOjkQs">大语言模型简要解释</a>：在这里深入了解：https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi 技术细节演讲：https://youtu.be/KJtZARuO3JY 这是...</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：以下是我们所有 Notebook 的列表：
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1328978488023056435)** (16 messages🔥): 

> `QA training techniques, Model performance issues, Fine-tuning models, MLX framework, Ollama compatibility` 


- **使用 QA 对子集进行训练**：一位用户建议在有限的数据集上训练之前，先提取 QA 对子集进行 1 个 epoch 的预热（primer），这引发了对模型潜在**遗忘（forgetting）**问题的担忧。
   - 另一位用户警告说，该数据集包含特定的模式和表情符号，可能会对模型的**响应质量**产生负面影响。
- **对模型偏差的担忧**：讨论强调了使用原始数据集可能会影响模型在响应中的偏差，特别是在对齐（alignment）方面。
   - 一位用户希望专注于**撤销**此前微调模型中存在的**过度审查**。
- **为非 Apple 系统转换模型**：一位用户询问如何将使用 mlx_lm 创建的微调模型转换为非 Apple 兼容模型，并指出无法在其他地方复制其质量。
   - 另一位用户提供了一个与 LORA 相关的链接，建议几乎任何模型都应该能加载 GGUF 格式，但这导致了进一步的兼容性问题。
- **Ollama 兼容性挑战**：在将模型导出为 GGUF 后，一位用户报告该模型无法在 Ollama 中运行，尽管它通过了 mlx 内部的测试，这表明可能存在兼容性问题。
   - 提到了适配器（adapters）的使用，但与原始 mlx 响应相比，其输出**质量明显较低**。
- **保持一致的生成设置**：在评估模型输出质量时，一位用户确认使用了相同的提示词（prompts）和生成设置（如 temperature），以确保测试的公平性。
   - 讨论引发了猜测，即根本问题可能在于 **mlx** 本身。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#fuse">mlx-examples/llms/mlx_lm/LORA.md at main · ml-explore/mlx-examples</a>：MLX 框架中的示例。通过在 GitHub 上创建账户，为 ml-explore/mlx-examples 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset">HumanLLMs/Human-Like-DPO-Dataset · Datasets at Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1328824706307981455)** (141 条消息🔥🔥): 

> `Phi-4 训练问题、Fine-tuning Llama、使用 WSL 进行 AI 开发、模型中的 Dynamic Quantization、Windows 上的 Conda 安装` 


- **Phi-4 训练导致 NaN 的问题**：用户报告在训练 **Phi-4** 模型时，如果将 per_device_eval_batch_size 设置为 2，在评估周期中会出现 NaN，这表明可能与评估 batch size 过低有关。
   - 另一位用户确认增加 batch size 后成功解决，这表明不同的硬件限制可能会影响训练参数。
- **针对特定任务 Fine-tuning Llama 模型**：讨论涉及使用过时模型进行 fine-tuning 的挑战，建议强调了模型更新对于特定领域任务的重要性。
   - 参与者辩论了当前方法的有效性，并强调了训练中对多样化、高质量数据集的需求。
- **使用 WSL 进行 AI 开发**：有建议指出在 Windows 上运行 AI 工作流可能会有问题，建议使用 **WSL2** 以获得更顺畅的体验，因为它具有更好的 Linux 支持。
   - 参与者一致认为，由于 Windows 上的兼容性问题，许多 AI 开发者更倾向于使用 Linux 或 WSL 环境。
- **Dynamic Quantization 与模型性能**：用户对 **Phi-4** 模型使用 dynamic quantization 与标准 4-bit 版本相比没有明显性能差异表示担忧。
   - 用户澄清说，虽然 loss 值可能没有变化，但 dynamic quantization 理论上应该提高训练精度，这引发了对该问题的进一步调查。
- **Windows 上的 Conda 安装问题**：一位用户报告在 Windows 上创建 **conda** 环境时遇到困难，原因是无法获取 **xformers** 包，并寻求解决建议。
   - 建议包括在创建环境时忽略该包或稍后使用 pip 安装，而其他用户则建议利用 **WSL** 进行更简便的设置。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1mf3lqz2ga80p_rIufDvBPvyFqtn9vcdS?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/unsloth-4-bit-dynamic-quants-67503bb873f89e15276c44e7">Unsloth 4-bit Dynamic Quants - a unsloth Collection</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/phi4">使用 Unsloth Fine-tune Phi-4</a>: 使用 Unsloth Fine-tune 微软最新的 Phi-4 模型！我们还发现并修复了模型中的 4 个 bug。</li><li><a href="https://huggingface.co/collections/unsloth/qwen-25-coder-6732bc833ed65dd1964994d4">Qwen 2.5 Coder - a unsloth Collection</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating">安装与更新 | Unsloth 文档</a>: 了解如何在本地或在线安装 Unsloth。</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-upd">Unsloth 文档</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=V6LDl3Vjq-A&t=225s">轻松训练 Llama 3 并上传到 Ollama.com（必看）</a>: 通过学习如何使用自定义数据 fine-tune 这个强大的 AI 模型，释放 LLaMA 3.1 的全部潜力！🚀 在本视频中，我们将带你...</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">教程：如何 Fine-tune Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>: 在 Ollama 上本地运行自定义个人助手（类似 ChatGPT）的初学者指南
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1328840426970546256)** (9 messages🔥): 

> `Grokking phenomenon, LLM training methods, Security conference submissions, Research papers and resources, Grokking video sequel` 


- **Grokking 已解决 - 必看视频**：一段名为《终于：Grokking 被解决了——它并非你所想的那样》的视频讨论了 AI 模型中 **Grokking 的惊人现象**，因为它与 LLM 中长期的过拟合（overfitting）有关。成员们对该视频表示热烈欢迎，并指出其在理解 AI 泛化（generalization）方面的重要性。
   - 一位观众评论道：“太棒的视频了，必看，如果不是因为太累了，我现在就会去读那篇论文。”
- **建议向安全会议投稿**：一位成员建议，关于 LLM 的演讲可能更适合 **特定领域的会议**，尤其是安全会议，而不是典型的 ML 会议。这与目前关于 LLM 在专业领域应用的讨论相一致。
   - 这一见解反映了人们对 LLM 研究及其在各个领域影响的跨学科兴趣日益增长。
- **分享关于 Grokking 的研究论文**：一位用户分享了一篇关于 Grokking 的研究论文，指出 *Grokking* 因其延迟泛化（delayed generalization）的有趣特性，挑战了目前对深度学习的理解。提供了 [论文](https://arxiv.org/abs/2501.04697v1) 链接和相关的 [GitHub 仓库](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability) 链接以供进一步探索。
   - 论文认为，如果没有正则化（regularization），Grokking 任务可能会导致他们所谓的 *Softmax Collapse*。
- **探索记忆与 Grokking 的对比**：一位成员表示，希望研究人员能深入调查 **记忆（memorizing）与 Grokking** 之间的转变，因为这可能会揭示生物神经元所使用的训练方法的新见解。这突显了人们相信理解这些学习阶段可能带来突破。
   - 该评论反映了社区对通过生物学类比来增强 AI 训练方法的兴趣。
- **分享 Grokking 续集视频**：分享了一个续集视频链接，标题为《Grokking - 续集》，提供了关于 AI 中 Grokking 现象的进一步见解。这加强了正在进行的关于 AI 学习过程本质的讨论和调查。
   - 续集承诺在原视频介绍的概念基础上进行构建，吸引对 LLM 理解演变感兴趣的观众。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/SRfJQews1AU?si=s3CSvyThYNcTetX_">Finally: Grokking Solved - It&#39;s Not What You Think</a>: Grokking，即 AI 模型对新知识的突然泛化——发生在 LLM 长期过拟合之后，是一个令人惊讶的现象...</li><li><a href="https://arxiv.org/abs/2501.04697v1">Grokking at the Edge of Numerical Stability</a>: Grokking 是指在长期过拟合后发生的突然泛化，是一个挑战我们对深度学习理解的惊人现象。尽管在理解...方面取得了重大进展。</li><li><a href="https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability">GitHub - LucasPrietoAl/grokking-at-the-edge-of-numerical-stability</a>: 通过在 GitHub 上创建账户，为 LucasPrietoAl/grokking-at-the-edge-of-numerical-stability 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1328815925918437466)** (206 条消息🔥🔥): 

> `AI Image Generation, Fake Cryptocurrency Launches, Model Metrics, ComfyUI and Stable Diffusion, Sharing Generated Images` 


- **使用 Stable Diffusion XL 成功生成图像**：一位用户正在使用 `stabilityai/stable-diffusion-xl-base-1.0` 模型生成图像，但正在询问有关图像生成时间的指标。
   - 他们在 Colab 上运行该模型，并试图了解是否可以获取预定义的指标（如每秒迭代次数）。
- **加密货币发布诈骗警报**：社区讨论了最近一起涉及与 Stability AI 挂钩的虚假加密货币发布的骗局，并警告成员不要点击任何可疑链接。
   - 用户对人们如此轻易落入此类陷阱表示担忧，并分享了在类似情况下的损失经历。
- **ComfyUI 提供图像生成指标**：一位成员分享说，在使用 ComfyUI 时，他们能够看到图像生成过程中每步迭代所花费的时间指标。
   - 相比之下，另一位在 Colab 中运行脚本的用户注意到缺少此类指标，并正在寻找特定于模型的运行数据。
- **尝试不同的模型设置**：关于调整图像生成中推理步数（inference steps）的讨论强调了模型在更改设置方面的灵活性及其对结果的影响。
   - 一位用户提到可以运行多达 50 步推理，但对模型特定的指标更感兴趣。
- **分享 AI 生成的图像**：一位用户询问了分享生成图像的最佳场所，包括分享失败的案例以帮助改进 AI 模型。
   - 建议将 Civitai 和社交媒体作为潜在平台，尽管有人指出分享生成的图像会引发对数据质量的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/dango233max/status/1879734940264481006">Dango233 (@dango233max) 的推文</a>：刚刚联系了我的 SAI 朋友。这是一个骗局！！！！@StabilityAI 的 X 账号被盗了。不要相信它！</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">stabilityai/stable-diffusion-xl-base-1.0 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1329175159998906409)** (3 条消息): 

> `Agent Identity Hackathon, Mixer Event, Xeno Grant` 


- **关于 Agent 身份的 Mixer 和黑客松**：欢迎参加 **24 日的 Mixer** 和 **25 日在纽约 Betaworks** 举行的 **黑客松**，重点关注 **Agent 身份**，最有趣的项目将获得 **5000 美元奖金**。
   - 活动包含食物、饮料和良好的氛围；请在 [活动页面](https://lu.ma/5rlcrlpb) 查看详情。
- **Xeno Grant 黑客松注册**：**Xeno Grant: Agent Identity Hackathon** 需要注册，该活动为每个 **Agent** 提供 **10,000 美元**——一半为 $YOUSIM，一半为 $USDC。
   - 该计划为 Agent 及其开发者提供为期 **4 周** 的时间，参与需经过批准。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/vintrotweets/status/1879582102112424356">vintro (@vintrotweets) 的推文</a>：🚨 人类请注意 🚨 我们将在 @betaworks 举办一场关于 Agent 身份的 Mixer 和黑客松，以启动 @xenograntai 🥳 快来构建你一直想要拥有的 Agent。5000 美元奖金将授予最...</li><li><a href="https://lu.ma/5rlcrlpb">Xeno Grant: Agent Identity Hackathon · Luma</a>：加入 Plastic Labs 和 Betaworks 参加 Agent 身份黑客松，启动 Xeno Grant（由 $YOUSIM 提供支持）。5,000 美元奖金将授予最引人注目的……
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1328833048216604722)** (90 条消息🔥🔥): 

> `Model Performance Issues, Program Synthesis Focus, Cerebras Yield Solutions, Contextual AI Platform Launch, LLM Language Understanding`

- **模型深陷 Doom Loops（死循环）**：成员们讨论了模型在训练后表现出“训练不足（undercooked）”的问题，特别提到了在多轮对话（multi-turn conversations）中的困难以及产生重复输出的情况。
   - 其中一个例子提到 Division 模型不断生成“let’s bring down the last digit”这句话。
- **Ndea 实验室的程序合成目标**：François Chollet 宣布成立 Ndea，专注于深度学习引导的程序合成（program synthesis），旨在实现真正的 AI 创新。
   - 这种方法被视为一种令人耳目一新的替代方案，区别于目前主流的通过 LLM 规模化（scaling）来实现 AGI 的路径。
- **Cerebras 攻克芯片良率挑战**：Cerebras 分享了在生产比传统芯片大 50 倍的晶圆级芯片（wafer-scale chip）时，如何实现相当良率的见解，挑战了传统的半导体常识。
   - 他们的方法涉及对芯片尺寸与容错性（fault tolerance）之间关系的新理解，从而实现了更高的良率。
- **Contextual AI 平台庆祝里程碑**：Contextual AI 平台宣布成功使用 Meta 的 Llama 3.3 实现了部署，并由 Google Cloud 和 NVIDIA GPUs 提供动力支持。
   - 他们向各合作伙伴和投资者的支持表示了感谢，以庆祝这一里程碑。
- **LLM 难以应对冷门词汇**：讨论中提到许多 LLM 无法识别“protolithic”一词，突显了语言模型在语言理解方面的问题。
   - 这引发了关于模型训练中词汇唯一性和复杂性的笑声和评论。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/xianbao_qian/status/1879425413317001397?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Tiezhen WANG (@Xianbao_QIAN) 的推文</a>：InternLM v3 现已上线！- SoTA 性能，超越 Llama3.1-8B 和 Qwen2.5-7B 等模型 - 能够通过系统提示词进行深度推理（详见其模型卡片）- 仅在 4T 高质量数据上训练...</li><li><a href="https://sakana.ai/transformer-squared/">未找到标题</a>：未找到描述</li><li><a href="https://x.com/natolambert/status/1879369464975614280">来自 Nathan Lambert (@natolambert) 的推文</a>：出于某种原因，Claude 坚信 "protolithic" 不是一个单词。不肯罢休。太神奇了。它甚至抱怨说，也许你想说的是 "protolithic"。我复现了几次...</li><li><a href="https://x.com/deepseek_ai/status/1879465495788917166">来自 DeepSeek (@deepseek_ai) 的推文</a>：🎉 隆重推出 DeepSeek App！💡 由世界级的 DeepSeek-V3 提供支持 🆓 免费使用，交互流畅 📱 现已正式登陆 App Store、Google Play 及各大安卓市场 🔗 立即下载：h...</li><li><a href="https://x.com/fchollet/status/1879583863368032432">来自 François Chollet (@fchollet) 的推文</a>：我将与 @mikeknoop 联手创立 Ndea (@ndeainc)，一个新的 AI 实验室。我们的重点：深度学习引导的程序合成（program synthesis）。我们押注于一条不同的道路，以构建具有真正发明能力的 AI...</li><li><a href="https://x.com/TheXeophon/status/1879516667971268659">来自 Xeophon (@TheXeophon) 的推文</a>：这是针对 minimax-01 的 vibe bench 完整评估套件（编码 + 通用）。正如我在初步 vibe 测试中所预料的那样，输出的方差巨大。在 pass@5 上，它的得分与 Llama 3.3 70B 相同，...</li><li><a href="https://x.com/teortaxestex/status/1879273615960743995?s=46">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：与由于合成数据导致的“品味崩溃（TASTE COLLAPSE）”这一灾难相比，由于合成数据导致的模型崩溃（model collapse）完全是小题大做（因为 Marcus 式的反统计应对和受挫艺术家的情绪而被夸大了）...</li><li><a href="https://x.com/ContextualAI/status/1879563309080547376">来自 Contextual AI (@ContextualAI) 的推文</a>：Contextual AI 平台自豪地基于 Meta 的 Llama 3.3 构建，运行在 Google Cloud 上，并在 NVIDIA GPU 上进行训练。我们对这一里程碑感到非常自豪，并感谢所有的客户、合作伙伴...</li><li><a href="https://fxtwitter.com/lmarena_ai/status/1879581805139079348">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：好奇人们如何使用 Chatbot Arena？介绍 Arena Explorer🔎：一种可视化 Chatbot Arena 数据的新方式！通过主题建模流水线，我们将用户提示词组织为：- 大类别...</li><li><a href="https://ndea.com/">Ndea</a>：一个智能科学实验室。</li><li><a href="https://x.com/dylan522p/status/1879375143044350072">来自 Dylan Patel (@dylan522p) 的推文</a>：显然昨天 Jensen 有点生气，昨晚在他们的医疗 AI 活动台上喝了 3 杯苏格兰威士忌，笑死。注意这是传闻，我不在场，因为在读法规。引用 Dylan...</li><li><a href="https://x.com/xianbao_qian/status/1879425451468423456?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Tiezhen WANG (@Xianbao_QIAN) 的推文</a>：https://huggingface.co/collections/internlm/internlm3-67875827c377690c01a9131d</li><li><a href="https://x.com/StringChaos/status/1879619028651745287">来自 Naman Jain (@StringChaos) 的推文</a>：📢 很高兴分享 LiveCodeBench 的第 5 次更新。这次我们增加了 167 个新问题，总计收集了 880 个问题，比 v1 的 400 个问题增加了一倍多。排行榜 ⬇️ - 🥇 open a...</li><li><a href="https://cerebras.ai/blog/100x-defect-tolerance-how-cerebras-solved-the-yield-problem">100 倍缺陷容忍度：Cerebras 如何解决良率问题 - Cerebras</a>：未找到描述</li><li><a href="https://huggingface.co/collections/internlm/internlm3-67875827c377690c01a9131d">InternLM3 - 一个 InternLM 集合</a>：未找到描述</li><li><a href="https://www.merriam-webster.com/dictionary/protolithic#:~:text=pro%C2%B7%E2%80%8Bto%C2%B7%E2%80%8Blith,of%20the%20Stone%20Age%20%3A%20eolithic">PROTOLITHIC 的定义</a>：属于或关于石器时代的最早时期：始石器时代……查看完整定义</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i1a88y/minimaxtext01_a_powerful_new_moe_language_model/">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1329127001100910623)** (19 messages🔥): 

> `ReaderLM-2 能力, MATH 数据集 DMCA 问题, AMD GPU 赞助建议, Tensorwave MI300X 发布, AoPS 独家版权担忧` 


- **ReaderLM-2 在与传统脚本的竞争中表现不佳**：Jina 推出了 [ReaderLM-2](https://jina.ai/news/readerlm-v2-frontier-small-language-model-for-html-to-markdown-and-json/) 用于将 HTML 转换为 Markdown，但其表现不如他们之前的脚本方法。
   - 他们在博客文章中通过省略之前的对比来简化说明，以便专注于 ReaderLM 的新功能。
- **MATH 数据集遭遇 DMCA 停用通知**：在收到 [DMCA 停用通知](https://huggingface.co/datasets/hendrycks/competition_math/discussions/5) 后，MATH 数据集已被禁用，这引发了对相关数学数据集未来的担忧。
   - 有人认为，除了 AoPS 的独家内容外，来自 MAA 等组织的数据集可能仍被允许适当使用。
- **推动 AMD GPU 支持**：一位成员提议 **AMD 应该资助 Ai2**，以鼓励使用其 GPU 并为研究人员提供资金支持。
   - 这一建议是在提到 **Intel 赞助** Stability AI 之后提出的，强调了此类合作伙伴关系的竞争优势。
- **Tensorwave 的 MI300X 可用于 AI 计算**：Tensorwave 推出了 **MI300X** 作为 AI 训练和推理的云解决方案，并立即向用户开放。
   - 他们提供裸金属和托管服务选项，强调易用性和性能优势。
- **AoPS 独家版权引发疑问**：关于 AoPS 内容独家性的讨论浮出水面，推测**用户发布的解决方案**可能会面临限制。
   - 有建议称，在剔除 AoPS 独家材料后，MATH 数据集的部分内容仍可重现，但这会影响数学资源的获取性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/tmkadamcz/status/1879584048429105238">来自 Tom Adamczewski (@tmkadamcz) 的推文</a>：Hendrycks MATH 刚刚收到 DMCA 停用通知。该数据集目前已被禁用。https://huggingface.co/datasets/hendrycks/competition_math/discussions/5</li><li><a href="https://tensorwave.com/">立即访问 MI300X GPU | TensorWave | MI300X 云</a>：立即在 TensorWave 云上访问 AMD MI300X GPU。今天就联系我们开始使用。</li><li><a href="https://jina.ai/news/readerlm-v2-frontier-small-language-model-for-html-to-markdown-and-json/">ReaderLM v2：用于 HTML 转 Markdown 和 JSON 的前沿小语言模型</a>：ReaderLM-v2 是一个 1.5B 的小语言模型，用于 HTML 到 Markdown 的转换和 HTML 到 JSON 的提取，具有卓越的质量。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1329272230076481709)** (2 messages): 

> `非推理模型, GPT-4o` 


- **定义非推理模型**：*我们对非推理模型使用什么术语？* 这个问题是针对模型分类提出的，以对比作为推理模型的 **o1**。
   - 作为回应，一位成员将 **GPT-4o** 称为“极其平庸的基础自回归模型”。
- **GPT-4o 的分类**：讨论集中在如何将 **GPT-4o** 与 **o1** 等推理模型进行分类。
   - 一位参与者将 **GPT-4o** 标记为“极其平庸的基础自回归模型”，强调了其非推理的本质。


  

---

### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1328816383651479714)** (7 条消息): 

> `Multimodal Visualization-of-Thought (MVoT), Chain-of-Thought (CoT) prompting, Mind's Eye paradigm, Simulation in AI reasoning, Grounding language models` 


- **创新的 Multimodal Visualization-of-Thought 被提出**: 一位成员讨论了 **Multimodal Visualization-of-Thought (MVoT)** 方法，该方法通过生成推理轨迹的图像可视化来增强 **Multimodal Large Language Models (MLLMs)** 的推理能力，详见[这篇论文](https://arxiv.org/abs/2501.07542v1)。
   - 讨论强调了其改进 **Chain-of-Thought (CoT)** 提示词技术的潜力，并建议它可以有效地补充强化学习策略。
- **Mind's Eye 连接语言与物理现实**: 另一位成员将 **Mind's Eye** 范式与一种将语言模型植根于现实的新方法联系起来，该方法利用模拟来增强推理，详见[这篇论文](https://arxiv.org/abs/2210.05359)。
   - 该方法通过结合 **DeepMind** 的 **MuJoCo** 结果，显著提高了推理准确性，标志着 AI 在理解物理世界方面取得了重大进展。
- **社区对新 AI 方法的热烈讨论**: 成员们对讨论中的创新策略表示兴奋，其中一位称 MVoT 的应用“非常酷（sick）”。
   - 对话反映了对未来 AI 能力的**乐观**情绪，以及对过去 AI 推理方法的怀旧。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.07542v1">Imagine while Reasoning in Space: Multimodal Visualization-of-Thought</a>: Chain-of-Thought (CoT) 提示词技术已被证明在增强 Large Language Models (LLMs) 和 Multimodal Large Language Models (MLLMs) 的复杂推理方面非常有效。然而，在复杂...</li><li><a href="https://arxiv.org/abs/2210.05359">Mind&#39;s Eye: Grounded Language Model Reasoning through Simulation</a>: 人类与 AI 之间成功且有效的沟通依赖于对世界的共同体验。通过仅在文本上进行训练，当前的语言模型 (LMs) 缺乏植根于现实的体验...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1329095741867823204)** (47 messages🔥): 

> `AI Relationships, Population Decline Implications, Automation of Jobs, AI and Social Movements, Challenges with LLMs` 


- **AI 关系引发复杂情绪**：讨论围绕 **ChatGPT** 和 **Claude** 被视为比真人更好的伴侣或治疗师展开，引发了对未来与 AI 关系动态的担忧。
   - 一位参与者指出：*“我担心‘控制’伴侣所带来的心理影响……”*
- **对人口下降的担忧**：参与者辩论了快速的**人口下降**可能如何导致生育和社会规范的变化，并暗示在后 AI 时代，生育欲望的降低可能会发生转变。
   - 一位用户评论道：*“你认为我们什么时候能达到每个人都得到照顾的程度？”* 这表明了对时间表的不确定性。
- **白领工作的自动化**：对话强调许多**白领工作**可能很快就会被自动化，并就 **AGI** 和机器人技术对这些角色的潜在影响发表了看法。
   - 一位用户对 2040 年前看到广泛自动化表示怀疑，而另一位用户指出：*“人们真的很热爱他们的‘狗屁工作’并为其辩护。”*
- **对 AI 出人意料的社会反应**：参与者反思了针对 AI 的严肃**社会运动**尚未出现的观点，这可能会使该技术的采用变得复杂。
   - 一位成员评论说：*“感觉每个人都认为我们正在开启一个能实现一夜之间变革的‘无穷盒子’，”* 强调了对快速转变的怀疑。
- **LLM 易用性方面的困难**：人们对**基于 LLM 的聊天机器人**的易用性以及人们在尝试有效使用它们时面临的共同挣扎提出了担忧。
   - 一位成员评论道：*“即使给出了简单直接的指令，它们也会做出怪异的事情，”* 暗示用户缺乏理解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1877992375575883963">来自 Xeophon (@TheXeophon) 的推文</a>：很多人想和 Claude 发生关系，这显而易见</li><li><a href="https://techcrunch.com/2025/01/14/openais-ai-reasoning-model-thinks-in-chinese-sometimes-and-no-one-really-knows-why/">OpenAI 的 AI 推理模型有时会用中文“思考”，且没人知道原因 | TechCrunch</a>：OpenAI 的 o1 “推理”模型在推理问题时有时会切换到中文和其他语言，AI 专家并不确切知道原因。</li><li><a href="https://archive.is/2024.12.25-131150/https://www.nytimes.com/2024/12/13/technology/claude-ai-anthropic.html">为什么 Anthropic 的 Claude 在科技圈内人士中大受欢迎 - 纽约时报……</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1328938173119725690)** (10 messages🔥): 

> `Voiceover Techniques, Meta Ray-Bans, Project Aria, Content Creation, User Experiences` 


- **尝试配音技巧**：一位成员讨论了在配音过程中采用一种稍微“偷懒”的方法，选择不使用章节时间戳和明确的图像链接。
   - 另一位成员承认他们不知道这些功能，评论说这要么是一个令人尴尬的启示，要么是松了一口气。
- **Meta Ray-Bans 引起关注**：对话转向了 **Meta Ray-Bans**，一位成员表达了进一步定制它们的愿望，希望能有像 **Molmo** 这样的选项。
   - 他们将 Meta Ray-Bans 描述为一个有趣的小工具，强调了对增强功能的个人偏好。
- **了解 Project Aria**：一位成员承认因不知道 **Project Aria** 而感到尴尬，并分享了其官方页面的链接以获取更新。
   - 他们鼓励其他人订阅来自 Project Aria 的新闻，强调这是他们在技术探索中的新发现。
- **紧跟技术前沿**：小组讨论了紧跟新技术的重要性，例如配音功能和可穿戴技术。
   - 一位成员表示，访问节目笔记并通过播客应用收听是保持消息灵通的关键。
- **内容创作反馈**：一位成员提到了每周制作新内容的努力，指出尝试不同风格至关重要。
   - 他们反思了发布过程是一个持续学习的经历，尽管并不总是对结果感到满意。



**提到的链接**：<a href="https://www.projectaria.com/">来自 Meta 的 Project Aria 介绍</a>：Project Aria 是来自 Meta 的一个研究项目，旨在帮助负责任地构建未来。Project Aria 开启了我们连接和体验世界的全新可能性。

  

---

### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1329101212930543676)** (2 条消息): 

> `TSMC and Samsung, US-China chip flow restrictions` 


- **美国敦促 TSMC 和 Samsung 限制对华芯片出口**：正如[最近的一篇 Bloomberg 文章](https://www.bloomberg.com/news/articles/2025-01-15/us-to-push-tsmc-and-samsung-to-tighten-flow-of-chips-to-china)中所详述，美国政府正在向 **TSMC** 和 **Samsung** 施压，要求收紧流向**中国**的芯片。此举是地缘政治紧张局势下控制技术出口的更广泛努力的一部分。
   - 一位成员表示 *About time lol*（总算来了），表达了在近期事态发展后对该决定的宽慰或赞同。
- **对地缘政治芯片供应链的担忧**：围绕美国对 **TSMC** 和 **Samsung** 关于向**中国**供应芯片的要求所产生的影响展开了讨论。专家指出，此类措施可能会影响全球关系和半导体市场。
   - 一位参与者强调 *this is a critical step*（这是关键一步），以维持技术领先地位并保护国家利益。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2025-01-15/us-to-push-tsmc-and-samsung-to-tighten-flow-of-chips-to-china">Bloomberg - Are you a robot?</a>：未找到描述</li><li><a href="https://www.bloomberg.com/news/articles/2025-01-15/us-to-push-tsmc-and-samsung-to-tighten-flow-of-ch">Bloomberg - Are you a robot?</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1328824539638923417)** (125 条消息🔥🔥): 

> `Claude's Persona, Fine-Tuning Techniques, LLM Dataset Quality, Nous Research Funding, Hackathon Event` 


- **Claude 的 Persona：不仅仅是智能**：成员们讨论了 **Claude** 如何被认为比其他模型更具人性化和亲和力，有人将 Claude 描述为一个“随和的人”，感觉就像一个酷酷的同事。
   - 有人对 Claude 拒绝提供答案表示担忧，这强化了其强大但内敛的 **Persona** 形象。
- **探索 LLM 的 Fine-Tuning 技术**：一位成员就其 **LLM** 的 **Fine-Tuning** 寻求建议，提出了涉及公司知识和特定任务分类的两个部分。
   - 其他成员分享了关于有效 **Fine-Tuning** 实践的见解，强调了训练数据的多样性以及与模型的交互以提高准确性。
- **LLM 数据集质量受到关注**：讨论揭示了对用于训练的数据集质量的怀疑，观点认为需要更好的数据选择和调节。
   - 成员们表达了希望 **open-source** 社区在生成高质量 **synthetic data** 方面取得进展的愿望。
- **Nous Research 的资金来源**：澄清了 **Nous Research** 作为一个私营实体运营，与政府或学术机构没有直接联系，依靠私募股权和周边销售获得资金。
   - 成员们指出，虽然周边收入只占很小一部分，但支持来自捐赠以及与 Microsoft 等实体的合作。
- **即将举行的 Agent Identity Hackathon**：分享了一个关于 **agent identity** 的交流会和 **hackathon** 的公告，重点介绍了奖项并呼吁参与者加入在 **NYC** 举行的活动。
   - 该活动旨在鼓励围绕 **AI agent** 开发的创新项目和社区参与。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/karinanguyen_/status/1879270529066262733">Karina Nguyen (@karinanguyen_) 的推文</a>：我们很高兴推出 Tasks！ChatGPT 首次可以代表您异步管理任务——无论是单次请求还是日常事务。以下是我最喜欢的用例...</li><li><a href="https://x.com/vintrotweets/status/1879582102112424356">vintro (@vintrotweets) 的推文</a>：🚨 人类请注意 🚨 我们将在 @betaworks 举办一场关于 agent identity 的交流会和 hackathon，以启动 @xenograntai 🥳 快来开发你一直想要的 agent。奖金 5000 美元...</li><li><a href="https://www.minimaxi.com/en/news/minimax-01-series-2">MiniMax - Intelligence with everyone</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1328821149030944900)** (25 messages🔥): 

> `Gemini 用于数据提取, Grokking 论文见解, Ortho Grad 与 GrokAdamW 合并, Stablemax 函数问题` 


- **Gemini 在数据提取方面表现出色**：一位成员发现 **Gemini** 在**数据提取**方面表现异常优秀，认为在处理原始内容时，它比 **4o-mini** 和 **Llama-8B** 模型更可靠。
   - 在涉及准确内容的检索能力时，*对这些模型的信任可能会产生动摇*。
- **关于 Grokking 现象的讨论**：一位成员提出了关于 **Softmax** 函数数值不稳定性的问题，推测这可能导致 Attention 计算中的**熵退化（entropy degradation）**，正如一篇 [论文](https://arxiv.org/abs/2501.04697) 中所讨论的那样。
   - 该论文引入了 **Softmax Collapse (SC)** 的概念，并断言缓解这一现象可以让 Grokking 在没有正则化的情况下发生。
- **结合 GrokAdamW 和 Ortho Grad**：成员们辩论了 **GrokAdamW** 和 **Ortho Grad** 的潜在兼容性，其中一位成员目前正在测试这种组合。
   - 有人提出了*担忧*，即由于 **Ortho Grad** 消除了延迟，GrokAdamW 可能不再有益。
- **对替换 Softmax 持怀疑态度**：一位成员评论说，根据对 GPT-2 模型的测试观察结果，**替换 Softmax 函数不太可能带来净收益**。
   - 还有关于 **stablemax 函数** 不符合预期规范的问题，特别是与其平移不变性（translation invariance）相关的部分（与 Softmax 相比）。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.04697">Grokking at the Edge of Numerical Stability</a>：Grokking 是指在长时间过拟合后突然出现的泛化现象，这是一个挑战我们对深度学习理解的惊人现象。尽管在理解该现象方面已取得显著进展...</li><li><a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>：通过创建账户为 cognitivecomputations/grokadamw 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1329124960769147035)** (7 messages): 

> `Grokking 现象, Grokfast 优化器, Orthograd 优化器, Coconut GitHub 项目` 


- **Grokking 现象解释**：讨论了 AI 模型中的 **Grokking** 现象，即模型在过拟合后突然产生泛化。讨论强调模型在过拟合后最初会变得“懒惰”，然后最终克服这种懒惰。
   - 这可以理解为模型在经历一段停滞期后，冲向了更好的泛化能力。
- **Grokking 策略与工具**：成员们讨论了鼓励 **Grokking** 提早发生的方法，并建议与 **Grokfast 优化器** 产生潜在的协同效应。
   - 对话集中在改进 Grokking 的时机以增强模型性能。
- **Grokfast 优化器的挑战**：一位参与者指出在使用 **Grokfast 优化器** 实现稳定性方面存在困难，这一挑战在 LLM 训练中也得到了其他人的共鸣。
   - 研究人员开发的 **Orthograd 优化器** 被提议作为 SGD 或 AdamW 等典型训练优化器的更可靠的直接替代方案。
- **Coconut GitHub 项目**：一个名为 [Coconut](https://github.com/facebookresearch/coconut) 的 GitHub 项目涉及训练 LLM 在连续潜空间（continuous latent space）中进行推理，为 AI 领域提供了一种创新方法。
   - 该项目与 Facebook Research 相关，展示了提升语言模型能力的新方法论。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=SRfJQews1AU">Finally: Grokking Solved - It's Not What You Think</a>：Grokking，即 AI 模型对新知识的突然泛化——发生在 LLM 长时间过拟合之后，是一个令人惊讶的现象...</li><li><a href="https://arxiv.org/abs/2501.04697">Grokking at the Edge of Numerical Stability</a>：Grokking 是指在长时间过拟合后突然出现的泛化现象，这是一个挑战我们对深度学习理解的惊人现象。尽管在理解该现象方面已取得显著进展...</li><li><a href="https://github.com/facebookresearch/coconut">GitHub - facebookresearch/coconut: Training Large Language Model to Reason in a Continuous Latent Space</a>：在连续潜空间中训练大语言模型进行推理 - facebookresearch/coconut
</li>
</ul>

</div>
  

---

### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1329184978311057479)** (1 messages): 

> `Bolt 更新，项目标题编辑` 


- **Bolt 现已支持编辑标题**：最近的 **Bolt 更新**允许用户直接更改**项目标题**，从而增强了项目组织能力。
   - 正如 [Stackblitz](https://x.com/stackblitz/status/1879625416706785365) 的公告所确认，此功能简化了在列表中查找项目的过程。
- **精简的项目管理**：编辑项目标题的能力通过更好的项目识别提升了整体**用户体验**。
   - 用户现在可以轻松导航其项目，减少了在列表中搜索的时间。



**提及的链接**: <a href="https://x.com/stackblitz/status/1879625416706785365">来自 StackBlitz (@stackblitz) 的推文</a>：📢 Bolt 最新更新：你现在可以更改项目标题了 —— 让你在项目列表中更容易找到它！

  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1328851305342566422)** (14 messages🔥): 

> `GA4 API 集成问题，Firebase 数据加载技术，Bolt 的 Context 文件创建，Token 使用优化，聊天记录快照系统` 


- **Netlify 上的 GA4 API 集成困难**：一位开发者报告称，在 Netlify 上部署 **React/Vite app** 后，遇到了与 **GA4 API** 集成相关的 'Unexpected token' 错误，尽管该应用在本地运行正常。
   - *他们已经验证了凭据和环境变量，但正在寻求替代方案或建议。*
- **精简的 Firebase 数据加载流程**：一位用户分享了简化连接 **Firebase** 的方法，建议使用 'load demo data' 页面预填充 **Firestore**，以避免空架构错误。
   - *这种方法被认为是有益的，即使对于更有经验的开发者来说可能显而易见。*
- **使用 Bolt 创建 Context 文件**：一位用户成功创建了 **PROJECT_CONTEXT.md** 文件来存储项目详情，以便在对话 Context 丢失时辅助检索；其未来使用的有效性仍有待观察。
   - *随后讨论了核心产品中内置 Context 处理功能的重要性。*
- **减少 Bolt 中的 Token 使用**：用户对过度的 Token 使用表示担忧，一位用户考虑重新启动项目以改善体验，尽管这会消耗 Token。
   - *有人建议采用更高效的 Context 管理策略，包括讨论在 **Cursor** 等其他工具中发现的类似功能。*
- **开发中的聊天记录快照功能**：一位用户引用了 **bolt.diy repo** 中的一个 **Pull Request**，该 PR 引入了聊天记录快照系统，能够恢复过去的聊天状态。
   - *此功能可能与正在进行的关于改进 Context 持久性的讨论相一致。*



**提及的链接**: <a href="https://github.com/stackblitz-labs/bolt.diy/pull/444">feat: restoring project from snapshot on reload by thecodacus · Pull Request #444 · stackblitz-labs/bolt.diy</a>：添加聊天记录快照系统。概述：此 PR 为聊天记录引入了快照系统，允许恢复之前的聊天状态及其关联的文件系统状态。这...

  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1328827316867960915)** (135 条消息🔥🔥): 

> `Token 使用问题、Bolt 功能与更新、Supabase 数据库集成、聊天会话管理、Bolt 中的错误处理` 


- **Token 使用激增说明**：多位用户报告 Token 使用量异常偏高，其中一位用户称其 Prompt 每次消耗 **400 万个 Token**，暗示可能存在 Bug。
   - 其他人对这些说法表示怀疑，建议受影响的用户如果情况属实，应在 GitHub 上提交 Issue。
- **与 Supabase 的连接问题**：成员们提到在连接 **Supabase** 时遇到困难，有报告称在尝试存储数据时出现错误和应用崩溃。
   - 一些用户建议针对数据库集成问题开启帮助线程，以寻求社区支持。
- **请求保留聊天历史**：一位用户主张在 Bolt 项目中保留初始 Prompt 历史，强调尽管会话上下文会丢失，但其对未来参考仍具价值。
   - 另一位用户表示赞同，并提到他们会单独记录自己的初始 Prompt。
- **更新与功能讨论**：讨论内容包括对 **Git 支持**等功能的期待，以及针对影响代码生成一致性的“懒惰机器人（lazy bot）”效应的改进。
   - 最近的一次直播提到了即将推出的修复方案，让用户对更好的功能体验感到乐观。
- **错误管理策略**：用户分享了管理项目中代码错误的策略，其中一人建议使用 **Diff mode** 来简化解决过程和调试。
   - 一位用户报告了系统意外删除内容的问题，这进一步强调了错误处理讨论的重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://prnt.sc/I_1dn-CGYvdt">Screenshot</a>：使用 Lightshot 捕获</li><li><a href="https://prnt.sc/xZhBWXt879Hl">Screenshot</a>：使用 Lightshot 捕获
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1328828634005377158)** (16 条消息🔥): 

> `AI 生成脚本、世界观构建辅助、播客语调问题、基于论坛的信息提取、小说作者讨论` 


- **AI 轻松提供功能性脚本**：一位用户对 NotebookLM 如何根据几个关于数字病理学的论坛帖子生成适用于 **QuPath** 的 **Groovy 脚本**感到惊讶，这为他们节省了数小时的工作时间。
   - 这一结果显著节省了时间，展示了 AI 在实际应用中的效用。
- **通过 AI 洞察进行世界观构建**：一位用户分享说，他们使用 Notebook 来进行**世界观构建**，让 AI 解释不够完善的概念，这有助于克服写作障碍并挖掘被遗忘的想法。
   - 另一位用户提到添加类似 *'Wild predictions inbound!'*（疯狂预测即将来临！）的笔记，以有效评估 AI 的推测。
- **播客语调一致性担忧**：关于播客中主持人语调在剧集中意外变化的讨论引起了关注，这影响了收听体验。
   - 人们对交付的一致性表示担忧，特别是在教育格式中。
- **对 AI 局限性的批评**：一位用户评论了 NotebookLM 被察觉到的偏见，对其限制讨论某些话题表示沮丧。
   - 他们强调了 AI 拒绝参与特定问题且未提供建设性替代方案的案例。
- **呼吁开设新的播客频道**：一位用户建议为播客发布和广告服务创建一个单独的频道，以使 use-cases 频道专注于相关讨论。
   - 该提议旨在简化对话，并确保主频道致力于特定的用例。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://notebooklm.google.com/notebook/ca5c616d-3282-46c6-9acf-1848d2006003/audio">未找到标题</a>：未找到描述</li><li><a href="https://www.akashq.com/post/161c71d7-4ee1-4cc4-a5ec-c887e90f9a7c">What happened on Jan 15?</a>：由 This Day in History 发布</li><li><a href="https://www.akashq.com/post/51eae1b7-e011-4d66-83af-873d763a203d">What happened on Jan 14?</a>：由 This Day in History 发布
</li>
</ul>

</div>
  

---

### **NotebookLM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1328821288029917275)** (88 条消息🔥🔥): 

> `NotebookLM Plus 功能、API 和批量同步功能、使用 YouTube 作为来源、文本上传限制、抓取网站内容` 


- **NotebookLM Plus 过渡的不确定性**：用户对 NotebookLM Plus 功能的可用性表示困惑，特别是关于不同 Google Workspace 方案的过渡时间表，尤其是那些使用已弃用版本的用户。
   - 由于 Google 的沟通不明确，一些用户（如 firehorse67）在考虑升级方案的同时，仍在继续为特定的插件付费。
- **关于 API 和批量同步可用性的问题**：成员们询问了 API 访问的可能性以及批量同步 Google Docs 来源的能力，希望能在今年实现。
   - 目前尚未提供关于这些功能时间表的具体信息。
- **YouTube 导入链接的问题**：用户报告了在 NotebookLM 中导入 YouTube 链接作为来源时遇到困难，并对该功能的当前可用性提出质疑。
   - 几位用户确认目前无法有效使用 YouTube 链接。
- **文本和文件大小限制**：关于文本上传限制的查询显示，NotebookLM 每个来源上限为 **500,000 字**，每个笔记本上限为 **50 个来源**。
   - 如果用户超过这些限制，系统预计会发出通知，特别是当他们尝试超过字数限制时。
- **网站抓取能力**：参与者讨论了 NotebookLM 缺乏将整个网站抓取进来的功能，确认用户必须手动下载文件进行输入。
   - 有人建议使用 Chrome 扩展程序作为下载多个文件的变通方案，但目前尚不存在集成的抓取功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://workspace.google.com/blog/product-announcements/empowering-businesses-with-AI">每个企业的 AI 驱动工作未来 | Google Workspace 博客</a>：Google AI 的精华现已包含在 Workspace Business 和 Enterprise 方案中，无需额外插件即可使用 Gemini 和 NotebookLM Plus 等 AI 功能。</li><li><a href="https://chrisbora.substack.com/p/boras-law-intelligence-scales-with?r=aszci">Bora 定律：智能随约束而非算力扩展</a>：这是一篇探讨人工智能发展中新兴原理的工作论文。</li><li><a href="https://support.google.com/notebooklm/answer/14276471?hl=en&sjid=13501328390293499756-AP">笔记本 - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">升级到 NotebookLM Plus - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://support.google.com/a/answer/181865#zippy=%2Cturn-services-on-or-off-for-users">开启或关闭额外的 Google 服务 - Google Workspace 管理员帮助</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answe">NotebookLM 帮助</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1328911242512830484)** (34 条消息🔥): 

> `Discord bot API usage, Inefficiency in project, LLM interface and APIs, Payment issues with production key, Learning resources for programming` 


- **Discord bot 成功使用 API**：一名成员在第一次尝试时就使用 API 创建了一个 **Discord bot**，尽管他们幽默地提到了其**低效率**。
   - *它似乎浪费了很多 token*，因为该过程涉及运行一个优化较差的 **jar file** 并读取其控制台输出。
- **重新设计项目设置**：有建议提出要**重构**项目流程，强调使用适用于 LLM 的 Node 库将提高效率。
   - 一位成员提到他们了解了如何激活独立的 **jar files**，表明他们对该过程的理解正在演进。
- **理解 API 通信**：成员们讨论了与 API 通信的核心在于发送 **JSON payloads** 并接收响应，并将其与 **curl commands** 联系起来。
   - 一位成员表达了从 Java 或 Python 转向 Node 进行 API 通信时的不确定感，觉得虽然有技能但缺乏基础知识。
- **解决生产密钥的支付问题**：一位成员报告了在保存 **production key** 支付方式时遇到的问题，认为这可能是基于地理位置的。
   - 其他人建议联系支持部门，并检查使用 **OpenRouter** 的选项，它可以代理所有 Cohere 模型。
- **寻求学习资源**：一位成员询问了关于使用 Node 进行 API 调用（API calls）的学习资源，暗示希望得到结构化的指导。
   - 另一位成员建议在 prompt 中指定响应长度，以此作为控制 API 输出的一种方式。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1328819513340330108)** (25 条消息🔥): 

> `Context length of 128k tokens, Cohere API key limits, Rerank model performance decline, Audio noise reduction recommendations, Updating Command R models` 


- **Cohere 的上下文长度设置了新的 token 限制**：Cohere 模型的上下文长度设定为 **128k tokens**，允许在单次对话中容纳约 **42,000 个单词**，且在整个对话过程中不会丢失上下文。
   - 用户澄清说，这个长度适用于单次对话中**整个时间线**的交互。
- **Cohere API 密钥有请求限制**：用户询问 API 密钥是否有 token 限制或仅有调用限制，并确定它是基于**请求（requests）**的，参考了 [Cohere 的速率限制文档 (rate limit documentation)](https://docs.cohere.com/v2/docs/rate-limits)。
   - 这包括针对不同端点的各种速率限制，例如试用密钥（trial keys）在对话时每分钟 **20 次调用**。
- **Rerank 模型性能面临问题**：一位用户报告 **Cohere 的 rerank-v3.5 model** 性能下降，称只有在使用最新的用户查询（user query）而不是完整的对话历史时，才能获得良好的结果。
   - 他们提到测试了另一个服务 **Jina.ai**，效果更好，并寻求对该问题的澄清，收到了用于故障排除的支持联系方式。
- **寻求先进的音频降噪方法**：一位用户请求推荐**最先进的噪声抑制算法（state-of-the-art noise suppression algorithms）**，尽管另一位参与者指出该群体并不擅长音频模型。
   - 尽管缺乏相关专业知识，他们还是询问了音频降噪的具体技术和实现。
- **希望更新 Command R 模型**：讨论中提到了希望使用新数据和微调技术持续更新 **Command R** 和 **R+ models**，而不是开发全新的模型。
   - 另一位用户指出，利用检索增强生成（RAG）应用于现有模型来更新知识是一种有效的方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/v2/docs/rate-limits">Different Types of API Keys and Rate Limits — Cohere</a>：此页面描述了 Cohere API 针对生产和评估密钥的速率限制。</li><li><a href="https://youtu.be/B45s_qWYUt8">How Cohere will improve AI Reasoning this year</a>：Cohere 首席执行官 Aidan Gomez 揭示了他们如何应对 AI 幻觉并提高推理能力。他还解释了为什么 Cohere 不使用任何外部...
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1329153188149985330)** (10 messages🔥): 

> `Cohere Client 初始化错误，生产环境密钥的支付方式问题，Cohere ClientV2 的使用` 


- **Cohere Client 初始化错误**：一位用户报告在通过 `Client(client_name="YOUR_CLIENT_NAME", token="token")` 初始化 Cohere 客户端时，遇到了关于意外关键字参数 'token' 的 **TypeError**。
   - 另一位成员建议使用更新后的初始化方法 `cohere.ClientV2("<<apiKey>>")` 以避免此错误。
- **生产环境密钥的支付方式问题**：一位用户对由于无法保存支付方式而无法购买生产环境密钥（Production Key）表示沮丧。
   - 有人指出某些地区可能会面临**银行卡问题**，并询问该用户的所在地以便进一步调查。


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1329128997820305551)** (13 messages🔥): 

> `Cohere Bot 交互，数到 10，搜索文档` 


- **Cohere Bot 趣味交互**：一位用户要求 **Cohere Bot** 数到 10，它高效地完成了任务，按顺序回复了 **1 到 10**。
   - 这一互动展示了该 Bot 功能的休闲和互动特性。
- **搜索特定指导**：一位用户询问如何将“上帝的话语”融入到“平庸之处”，促使 Bot 搜索文档以寻求相关指导。
   - Bot 承认无法找到关于此查询的具体信息，展示了其当前数据库的局限性。
- **文档局限性显现**：尽管努力寻找有关“上帝的话语”的相关文档，Bot 最终未能为用户的查询找到满意的资料。
   - 这种情况强调了 Bot 对现有文档的依赖，以及在资源不可用时无法提供详细见解的局限。


  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1329076964509880330)** (6 messages): 

> `版主招募，社区贡献` 


- **新版主愿望面对现实**：一位成员表示愿意担任社区的另一名版主，并称：*如果需要新版主，请随时私信我*。
   - 然而，另一位成员幽默地评论道，**获得版主身份并非易事**，强调贡献才是关键。
- **获得版主认可的建议**：一位成员建议，要成为版主，应该*开始为社区做贡献*，以随着时间的推移证明自己的价值。
   - 这一建议得到了积极回应，那位有志成为版主的成员接受了建议：*你说得对，伙计！谢谢你的建议！*


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1328870392982671381)** (2 messages): 

> `Mistral 编程模型，Minimax-01 发布` 


- **Mistral 发布新型顶级编程模型**：来自 **Mistral** 的新型 **Fill-In-The-Middle (FIM)** 编程模型已经发布，标志着编程模型领域的重大进步。由于目前无法直接访问，用户可以通过 [Discord 频道](https://discord.gg/fVyRaUDgxW)申请该模型。
   - 该模型有望成为同类产品中的佼佼者，强调了其除 FIM 之外的独特能力。
- **Minimax-01 创下新纪录**：首个开源 LLM **Minimax-01** 现已发布，并以惊人的 **4M 上下文长度**通过了 **Needle-In-A-Haystack（大海捞针）测试**。更多详情可以在 [Minimax 页面](https://openrouter.ai/minimax/minimax-01)找到。
   - 如需访问此模型，感兴趣的用户请前往同一个 [Discord 服务器](https://discord.gg/fVyRaUDgxW)提交申请。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1879311582246977743">来自 OpenRouter (@OpenRouterAI) 的推文</a>：来自 @MistralAI 的新型顶级 FIM (fill-in-the-middle) 编程模型发布了！</li><li><a href="https://openrouter.ai/mistralai/codestral-2501`">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/minimax/minimax-01>">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1328817459830063135)** (80 条消息🔥🔥): 

> `DeepSeek API 问题，Token 限制不一致，Provider 性能，模型移除，Prompt caching` 


- **DeepSeek 遭遇性能问题**：许多用户报告了 **DeepSeek** API 的问题，在尝试访问服务时遇到了响应时间不一致和错误。
   - *Nicholaslyz* 指出，所有 **DeepSeek v3 providers** 最近都面临延迟问题，特别是在首个 token 响应（first token responses）方面。
- **Token 限制不一致**：用户对 **DeepSeek** 处理 token 限制的方式表示沮丧，宣称的 **64k tokens** 可能会在没有警告的情况下突然降至 **10-15k**。
   - *Amoomoo* 强调了这如何影响开发，因为意外错误破坏了他们应用的可靠性。
- **关于模型移除的担忧**：一位用户询问了模型 **lizpreciatior/lzlv-70b-fp16-hf** 可能被移除的问题，称他们遇到了 no endpoint 错误。
   - *Toven* 回应称，该模型可能不再有可用的 provider。
- **各 Provider 性能引发讨论**：讨论中提到了 **DeepSeek** 与其他 provider 之间的性能差异，一些人注意到 **TogetherAI** 和 **NovitaAI** 的延迟较高。
   - *Nilaier* 提到，虽然 **OpenRouter 网站** 显示这两个 provider 的延迟很高，但 **DeepSeek** 和 **DeepInfra** 保持了更易于接受的响应时间。
- **Prompt caching 功能**：一位用户询问 **OpenRouter** 是否支持像 **Claude** 这样模型的 prompt caching。
   - *Toven* 确认支持缓存，并提供了[文档](https://openrouter.ai/docs/prompt-caching)链接以获取更多细节。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/prompt-caching">Prompt Caching | OpenRouter</a>: 优化 LLM 成本高达 90%</li><li><a href="https://openrouter.ai/docs/requests#stream-cancellation">Requests | OpenRouter</a>: 处理传入和传出请求</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat/uptime">DeepSeek V3 – 运行时间和可用性</a>: 各 provider 的 DeepSeek V3 运行时间统计 - DeepSeek-V3 是 DeepSeek 团队的最新模型，建立在先前版本的指令遵循和编码能力之上。Pre-...</li><li><a href="https://intl.minimaxi.com/document/Pricing%20Overview?key=67373ec8451eeff1a85b9e4c">MiniMax-Intelligence with everyone</a>: 未找到描述</li><li><a href="https://tenor.com/view/dum-suspense-climax-monkey-shocked-gif-8054274">Dum Suspense GIF - Dum Suspense Climax - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/MiniMaxAI/MiniMax-Text-01">MiniMaxAI/MiniMax-Text-01 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1328855834939232299)** (62 条消息🔥🔥): 

> `AI 与神经反馈循环，AI 的 ISO 42001 认证，对话上下文中的 AI 局限性，ChatGPT API 与性能差异，生成式 AI 炒作与市场现实` 


- **AI 交互的创新神经反馈**：一位用户提出了 AI 利用神经反馈循环来分析最佳思维模式，并引导用户模仿这些模式以获得更好认知表现的概念。
   - 这种方法表明 AI 在个人学习和个人发展中可以扮演更积极的角色。
- **Anthropic 获得 ISO 42001 认证**：Anthropic 宣布获得新的 [ISO/IEC 42001:2023 标准](https://www.iso.org/standard/81230.html)负责任 AI 认证，展示了其对伦理 AI 治理的承诺。
   - 尽管对与 Anduril 等公司的合作存在质疑，但该认证有助于确保 AI 系统的开发和使用是负责任的。
- **AI 在对话中的记忆局限性**：成员们讨论了 AI 在对话中遗忘上下文的问题，特别是在分享图片时，导致了混淆和重复的提醒。
   - 有人建议，图片的简短保留期可能是 AI 快速丢失记忆的原因之一。
- **ChatGPT 的 API 性能差异**：讨论指出，在使用 ChatGPT 时，免费用户在质量和功能上遇到了限制，特别是与 Plus 用户相比的网页搜索功能。
   - 一些用户强调了不同层级能力之间的差异，指出免费 API 访问的性能有所下降。
- **生成式 AI 应用的炒作降温**：用户表示，由于应用有限，围绕生成式 AI 的兴奋感已经减弱，感觉仅局限于文本、图像和视频。
   - 人们对高质量 AI 输出相关的成本表示担忧，这可能会阻碍更广泛的采用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/">Mistral AI | Frontier AI in your hands</a>: 掌控前沿 AI</li><li><a href="https://search.app/efbG9erWwnvfyo2n8">Anthropic achieves ISO 42001 certification for responsible AI</a>: 我们很高兴地宣布，Anthropic 的 AI 管理系统已获得新的 ISO/IEC 42001:2023 标准认证。ISO 42001 是首个国际标准...</li><li><a href="https://www.minimaxi.com/en/news/minimax-01-series-2?utm_source=www.therundown.ai&utm_medium=newsletter&utm_campaign=chatgpt-gets-proactive-with-tasks&_bhlid=c7ae7b8d4af4b5c8e23e7a1bb278099d824087ee">MiniMax - Intelligence with everyone</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1328912272613052529)** (6 条消息): 

> `Custom GPT 上传，GPT-4o Tasks，网页版 Canvas 功能，模型局限性` 


- **Custom GPT 上传文件消失**：一位成员报告称，上传到其 **Custom GPT** 知识库的文件消失了，质疑这是否是一个 **bug**。
   - 人们对用于管理重要数据的上传功能的可靠性表示担忧。
- **Custom GPT 受模型类型限制**：有人指出 **Custom GPT** 基于较低版本的模型，且无法访问其他会话的记忆。
   - 成员们讨论认为，只有在提供**替代数据**的情况下，它们才能有效运行。
- **Canvas 功能被 Tasks 取代**：正如一位成员提到的，桌面网页版上的 **Canvas** 似乎已被 Tasks 界面取代。
   - 另一位成员确认，文本输入框附近的**工具箱图标**仍然可以访问 Canvas 功能。
- **GPT-4o Tasks 的功能**：**GPT-4o** 中的 Tasks 被描述为定时操作，可以在设定时间提醒用户执行特定活动。
   - 这些包括在指定时间练习语言或接收新闻更新等任务的通知。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1328821543509164103)** (5 messages): 

> `Assistants 引用文档, API 相关问题, 人类化提示词, Prompt engineering, 脚本支持` 


- **困扰于 Assistants 引用文档**：一位用户表示，很难让他们的 Assistants 在回答结束时停止引用文档。
   - 他们正在寻求关于在 Playground 中何处实施解决方案的建议。
- **Playground 中的 API 限制**：另一位用户澄清说，该问题无法在 Playground 中解决，因为它与 API 配置有关，而非 Prompt engineering。
   - 他们引用了之前的讨论链接以获取有关此话题的更多细节。
- **友好的人类化提示词讨论**：一位参与者分享了他们的人类化提示词，旨在使文本对第二语言使用者更易理解。
   - 目标是在保留特定术语的同时简化文本。
- **社区支持与脚本编写**：对话强调了社区愿意帮助用户解决与其 Assistants 相关的脚本问题。
   - 参与者表示鼓励，并提出帮助解决用户的编码挑战。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1328821543509164103)** (5 messages): 

> `Assistants 引用文档, Playground 中的 API 问题, AI 的人类化提示词` 


- **Assistants 持续引用文档**：一位用户对他们的 Assistants 在回答中频繁引用文档表示沮丧。
   - 他们寻求关于在 Playground 中何处插入代码以防止此问题的建议，随后另一位成员给出了建议。
- **针对 Playground 的 API 说明**：一位成员澄清说，引用文档的问题无法在 Playground 中处理，因为它涉及 API 使用而非 Prompt engineering。
   - 他们提供了一个链接以进一步澄清此话题，强调这超出了 Playground 的范围。
- **AI 内容的人类化提示词**：一位用户分享了一个提示词，旨在通过简化所使用的语言，使 AI 生成的内容对第二语言使用者更易理解。
   - 重点是在保持领域特定术语的同时，避免使用生僻词和复杂的句子结构。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1328883817338900480)** (66 messages🔥🔥): 

> `模型微调技术, 用户与助手模式, 上下文窗口与内存使用, 模型加载问题, 使用模型进行图像分析` 


- **探索模型微调**：一位用户正考虑使用某作者特定的公有领域文本来微调模型，通过指令性提示词提高输出质量。
   - 他们正在转向 Google CoLab 进行微调，并提到 Python 对他们来说是新事物，但对利用 LLM 进行写作感到兴奋。
- **理解用户模式与助手模式**：用户模式允许用户以自己的身份发送消息，而助手模式则让他们以 AI 的身份进行回复，这有助于控制对话流。
   - 用户澄清说，在助手模式下，他们的回复被视为 AI 生成的内容，从而塑造了交互的上下文。
- **管理上下文窗口限制**：用户讨论了“上下文已占用 90.5%”的通知，该通知指示了模型当前使用了多少上下文窗口。
   - 建议调整上下文大小，但提醒用户更大的上下文会增加模型的内存占用（Memory Footprint）。
- **解决模型加载问题**：用户分享了加载模型的经验，并排查了与高内存占用和系统规格相关的错误。
   - 讨论涉及了潜在的解决方案，如调整模型层数和设置，以优化在不同硬件上的性能。
- **LLM 的图像分析能力**：几位用户报告了在让 QVQ-72B 和 Qwen2-VL-7B-Instruct 等模型正确分析图像时遇到困难。
   - 强调了确保运行时环境（Runtime environments）保持最新对于成功实现图像分析功能至关重要。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1328842985777004647)** (11 messages🔥): 

> `推理速度对比, GPU 并行化限制, GPU 能效, 模型层分布` 


- **GPU 推理速度对比**：一场讨论透露，在特定条件下，**2x RTX 4090** 显卡可能达到 **19.06 t/s**，而 **RTX A6000 48GB** 为 **18.36 t/s**。
   - 一位参与者建议，如果考虑到 **4090 的显存速度优势**，A6000 的得分可以修正为 **19.27 t/s**。
- **能效提升**：一位参与者强调，与替代方案相比，使用 **2x RTX 4090** 显卡的功耗显著降低。
   - 这一点强调了在选择推理硬件时，效率与性能同等重要。
- **推理并行化的局限性**：对于当模型可以容纳在单块 GPU 显存中时，是否跨 GPU 并行化推理（例如将一半的层放在每块卡上）存在质疑。
   - 参与者指出，层后的输出同步可能会通过 **PCIe** 引入延迟或带宽问题。
- **跨 GPU 的模型层分布**：关于在**每块 GPU 上分布完整的模型层**以计算一半节点的想法进行了辩论，但被认为可能受限于必要的同步。
   - 鉴于该领域已经进行的深入研究，人们对这种方法的可行性表示担忧。
- **在 LM Studio 中使用 Claude**：一名成员分享了一个链接，展示了一个使用 **Claude** 创建的工具，旨在与 **LM Studio** 配合使用。
   - 这引起了人们对 AI 模型与现有软件解决方案结合的实际应用的关注。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1328830345742913620)** (36 messages🔥): 

> `DeepSeek 性能问题, Aider 代码编辑, 使用 GPU 执行模型, AI 内容标题党讨论, GitHub 机器人集成` 


- **DeepSeek V3 速度受限**：多位用户报告 **DeepSeek V3** 运行缓慢，其中一人表示它完全卡住，无法处理任何 token。
   - 其他人选择切换到 **Sonnet**，以便在故障期间获得更好的性能。
- **Aider 的编辑功能**：一位用户询问 Aider 是否允许在执行前“接受”或“拒绝”模型的编辑计划，而不是直接运行更改。
   - 另一位用户澄清说，这本质上就是 Aider 中 **architect mode** 的设计初衷。
- **关于 AI 内容标题党的讨论**：一位用户嘲讽另一位用户倾向于发布标题党内容，幽默地将其比作推销劣质车辆的二手车推销员。
   - 回应表明大家对误导性的 AI 内容感到普遍沮丧，同时也承认需要快速更新新工具。
- **模型执行的 GPU 要求**：讨论了高效运行模型所需的 GPU 规格，一位用户建议 **RTX 4090** 就足够了。
   - 其他人讨论了在处理大型模型时对高 RAM 的需求，并提到了潜在的兼容性问题。
- **使用 Aider 构建 GitHub 机器人**：一位用户询问是否有人成功利用 **Aider** 框架创建了 GitHub 机器人。
   - 这表明人们对 Aider 工具在软件开发自动化任务中的实际应用越来越感兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF/discussions/5">unsloth/DeepSeek-V3-GGUF · 运行所需的 GPU 显存是多少，4090 是否可行以及是否支持 ollama</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i1p0yb/running_deepseek_v3_with_a_box_of_scraps_but_no">Reddit - 深入了解任何内容</a>：未找到描述</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i1p0yb/running_deepseek_v3_with_a_box_of_scraps_but_not/">Reddit - 深入了解任何内容</a>：未找到描述</li><li><a href="https://github.com/Aider-AI/aider/blob/main/benchmark/README.md">aider/benchmark/README.md at main · Aider-AI/aider</a>：aider 是你终端里的 AI 配对编程助手。欢迎在 GitHub 上为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://sillytavernai.com/llm-model-vram-calculator/">LLM 模型 VRAM 计算器 &#8211; SillyTavern</a>：未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1329005341001781268)** (22 条消息🔥): 

> `Repository-map 大小问题，用于代码探索的 Agentic 工具，Aider API 调用日志记录，Aider 的 Git commit 问题，o1-preview API 性能` 


- **Repository-map 大小引发关注**：一位成员注意到在一个中型项目中，**repository-map** 从 **2k 增长到了 8k**，并询问最近的更新是否影响了这一变化。
   - 另一位成员分享说，当文件超过 **5 个**时，小的 repo-map 会变得低效，这暗示了两者之间的相关性。
- **Agentic 工具在代码库探索中占据主导地位**：一位用户提到使用 Cursor 的 chat 和 Windsurf 等 **agentic 工具**进行代码库探索，并表示这些工具优于标准的 repo maps。
   - 他们强调，在代码实现方面，**Aider** 的表现无可比拟，这使其成为他们在探索阶段使用额外工具后的首选。
- **Aider API 调用没有日志记录**：一位用户询问 **Aider** 是否有记录 LLM API 调用内容的方法，并发现聊天记录并不包含完整的 API 调用数据。
   - 回复确认目前在本地运行 Aider 并将数据转储到数据库之前，**没有现成的解决方案**。
- **Aider 中的 Git commit 挑战**：另一位用户报告说，尽管 **Aider** 在克隆的仓库中进行了更改，但即使设置正确，它也没有执行任何 commit。
   - 成员建议使用 **architect mode** 可能会有帮助；甚至有人提交了一个 PR 来改进排查指南。
- **o1-preview 的性能问题**：用户对 **o1-preview** API 的调用速度表示担忧，一位用户评论说它似乎变慢了。
   - 另一位用户确认，虽然他们偶尔使用它，但它消耗大量 tokens 且响应时间更长，表明可能存在效率低下的问题。



**提到的链接**：<a href="https://github.com/Aider-AI/aider/pull/2877">docs: Add architect mode section to edit errors troubleshooting guide by golergka · Pull Request #2877 · Aider-AI/aider</a>：未找到描述

  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1329089771640717475)** (2 条消息): 

> `Repomix，AI 友好型代码库打包，Repopack，API 调用优化` 


- **Repomix 提供 AI 友好型打包**：一位成员推荐 [Repomix](https://repomix.com) 作为一个有用的工具，可以将代码库打包成 **AI 友好型格式**。
   - 这可以简化与 LLMs 的集成工作，从而提高效率。
- **为 Aider 最小化 API 调用**：一位用户指出，追踪自 **Repopack** 以来的进展可能有助于管理 Aider 的精简版 API 调用。
   - 这种方法旨在优化发送给 **LLMs** 的数据。



**提到的链接**：<a href="https://repomix.com">来自 Repomix 的推文</a>：将你的代码库打包成 AI 友好型格式

  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1329167856549625856)** (4 条消息): 

> `文档字体粗细更新，用户关于可读性的反馈` 


- **文档字体变粗以提高可读性**：文档的字体粗细已更新为**更粗**，旨在提高可读性。
   - “*如果觉得不好请随时分享更多反馈*”表明了根据用户输入进行进一步调整的开放态度。
- **用户对字体更改的正面反馈**：用户对此次更新表示满意，指出这一变化对可读性来说**好得多**。
   - 一位用户对这一改进表示了简单的认可，强化了正面的反响。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1328873139983093840)** (41 条消息🔥): 

> `Mojo 中的函数支持、Zed Preview 扩展、SIMD 性能、Mojo 中的递归类型` 


- **Lambda 函数仍不受支持**：用户确认 Mojo 目前不支持 `lambda` 语法，但可以将函数作为参数传递。
   - 一位用户分享了一份 [roadmap document](https://docs.modular.com/mojo/roadmap/#no-lambda-syntax)，展示了 Mojo 中计划的功能。
- **在 Zed Preview 中获取 Mojo**：成员们讨论了如何在 Zed Preview 中安装 Mojo，并指出其安装方法与稳定版相同。
   - 用户确认在添加必要设置后，代码补全（code-completion）可以正常工作，尽管有些人在缺少配置时遇到了问题。
- **使用 SIMD 的性能陷阱**：有人对使用 `SIMD` 时可能出现的性能问题表示担忧，因为结果会因 CPU 架构和实现而异。
   - 用户建议检查汇编输出（assembly output）是否存在过多的寄存器重组（register shuffling），以确保这样做是值得的。
- **Mojo 中递归类型的挑战**：讨论了在 Mojo 中实现递归类型的困难，并建议使用指针代替。
   - 一位用户引用了之前创建树结构的尝试，并指向相关的 [GitHub issues](https://github.com/modularml/mojo/issues/3917) 以获取更多背景信息。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html">Ice Lake AVX-512 Downclocking</a>：检查 Intel Ice Lake CPU 上与 AVX 相关的降频程度。</li><li><a href="https://docs.modular.com/mojo/roadmap/#no-lambda-syntax">Mojo🔥 roadmap &amp; sharp edges | Modular</a>：Mojo 计划摘要，包括即将推出的功能和需要修复的问题。</li><li><a href="https://docs.modular.com/mojo/stdlib/benchmark/benchmark/run">run | Modular</a>：runfunc SIMD[float64, 1] = SIMD(60), maxbatch_size: Int = 0) -&gt; Report</li><li><a href="https://github.com/modularml/mojo/issues/3917">[BUG] --debug-level full crashes when importing · Issue #3917 · modularml/mojo</a>：Bug 描述：使用调试器运行 mojo 脚本会发生段错误，而运行常规 mojo 则能运行完成（尽管在常规脚本中也注意到了奇怪的行为...）。</li><li><a href="https://github.com/modularml/mojo/issues/3950">[Help wanted] Evaluating optional argument to not None gives segmentation fault · Issue #3950 · modularml/mojo</a>：问题描述：我有一个需要可选参数的类。当评估为 None 时，它会报错。如果评估为 None 会报错，我该如何评估它？而且，我也可能...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1328839199469867113)** (2 条消息): 

> `命运般的 Ping，AI 排名` 


- **命运般的 Ping 带来混乱**：一位成员幽默地描述了一个 **ping** 如何导致了广泛的 **混乱** 和不可预见的后果。
   - 他们将这一时刻称为“怪胎（abomination）”，反映了随之而来的混乱氛围。
- **位列 AI 前 1%**：另一位成员评论前一条消息的质量极高，称其简直是 AI 领域的前 **1%**。
   - 这一表态表明社区非常看重富有洞察力的贡献。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1328838102352924795)** (29 条消息🔥): 

> `LLM 中的关键 Token、VinePPO 与 CoT 轨迹、NanoGPT Speedrun 记录、TruthfulQA 数据集缺陷、数据集中的人工标注问题` 


- **识别关键 Token 以提升 LLM 性能**：最近的一篇摘要引入了 **critical tokens**（关键 Token）的概念，这些 Token 对 LLM 的推理任务至关重要，并能显著影响模型的准确性，特别是在 GSM8K 和 MATH500 等数据集上。
   - 提出的方法建议，识别并最小化这些 Token 可以增强模型性能，这与对隐式 PRM 的观察结果一致。
- **澄清 VinePPO 的用法**：成员们讨论了 **VinePPO**，指出尽管最初对其需求存在困惑，但其实现并不需要思维链（CoT）轨迹示例。
   - 澄清表明正在利用离线强化学习（RL），尽管在基准测试和对比方面仍存在疑虑。
- **NanoGPT 创下新的速度记录**：Fern.bear 报告了 **modded-nanoGPT** 的新 speedrun 记录，时间为 **3.17 分钟**，展示了诸如新的 lm_head 偏置和融合操作（fused operations）等改进。
   - 该记录引发了关于进一步优化的讨论，例如使用 **Long-Short Sliding Window Attention** 来增强性能。
- **探讨 TruthfulQA 数据集的缺陷**：TurnTrout 揭示了如何利用 **TruthfulQA** 多选题数据集中的微妙弱点，在隐藏问题的情况下实现 **79% 的准确率**，凸显了即使是知名基准测试也可能存在缺陷。
   - 这引发了对数据集可靠性的重要担忧，并影响了关于在其他数据集（如 **halueval**）中观察到的类似问题的持续对话。
- **对数据集中人工标注的担忧**：讨论涉及 **halueval** 等数据集中错误人工标注的普遍性，这往往导致误导性的结果。
   - 成员们表示这一问题非常广泛，有说法称在某些视觉语言模型数据集中，高达 **30% 的条目** 可能存在歧义或错误。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Turn_Trout/status/1879710659904254081">来自 Alex Turner (@Turn_Trout) 的推文</a>：Mark Kurzeja 和我利用了多选题 TruthfulQA 数据集的弱点，同时隐藏了问题！只需几个简单的经验法则就达到了 79% 的准确率。即使是备受推崇的基准测试也可能存在缺陷。...</li><li><a href="https://x.com/hi_tysam/status/1879687807678959729">来自 Fern (@hi_tysam) 的推文</a>：新的 NanoGPT 训练速度记录：在 8xH100 上 3.17 分钟内达到 3.28 FineWeb 验证损失。之前的记录（重建）：3.32 分钟。有很多变化！- 新的 Token 相关 lm_head 偏置 - 融合了多个操作 - Multi...</li><li><a href="https://arxiv.org/abs/2411.19943">Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM&#39;s Reasoning Capability</a>：数学推理任务对大语言模型（LLM）提出了重大挑战，因为它们需要精确的逻辑演绎和序列分析。在这项工作中，我们引入了...的概念。</li><li><a href="https://openreview.net/forum?id=BGnm7Lo8oW">Towards Learning to Reason at Pre-Training Scale</a>：提示大语言模型（LLM）输出思维链（CoT）推理可以提高处理复杂问题解决任务的性能。此外，存在几种流行的“自我提升”方法...</li><li><a href="https://github.com/facebookresearch/coconut">GitHub - facebookresearch/coconut: Training Large Language Model to Reason in a Continuous Latent Space</a>：训练大语言模型在连续潜空间中进行推理 - facebookresearch/coconut</li><li><a href="https://github.com/KellerJordan/modded-nanogpt/pull/71">Long-Short Sliding Window Attention (提升 3.2 秒或 0.053 分钟)，由 leloykun 提交 · Pull Request #71 · KellerJordan/modded-nanogpt</a>：目前，我们在所有层中以相同的速率预热滑动窗口注意力的上下文长度。此尝试改为在某些层中以不同方式预热上下文长度。这导致了...
</li>
</ul>

</div>

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1328840867003236496)** (5 messages): 

> `Loss vs Compute Plot, Induction Head Behavior, Circuit Interoperability, Pythia Models Training` 


- **Loss vs Compute 图表不一致**：一位成员询问，绘制 **loss vs compute** 是否会显示所有的 **induction head bumps** 都位于一条直线上，暗示可能存在相关性。
   - *Stellaathena* 澄清说，这通常发生在相同数量的 token 之后，表明答案是 **否**。
- **来自 Anthropic 论文的见解**：分享了关于原始 **Anthropic** 帖子和后来详细描述训练期间 circuit 行为的论文的进一步见解。
   - 特别引用了一篇 **circuit 互操作性论文**，该论文展示了不同 **Pythia 模型** 在训练过程中 circuit 出现的图表 [查看此论文](https://arxiv.org/abs/2407.10827)。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1329253449040134174)** (3 messages): 

> `MATH Dataset DMCA, AOps Disclosure` 


- **MATH 数据集面临 DMCA 下架**：据 [TMK Adamcz](https://x.com/tmkadamcz/status/1879584048429105238) 报道，**Hendrycks MATH 数据集** 遭到了 **DMCA 下架通知**，导致该数据集被禁用。
   - 关于此次下架的 [Hugging Face 讨论](https://huggingface.co/datasets/hendrycks/competition_math/discussions/5) 链接提供了更多背景信息。
- **AOps 承认题目来源**：一位成员指出，**MATH 数据集** 中的题目来源于 **AOPS** (Art of Problem Solving)。
   - 有人指出，AOPS 一直在披露有关题目来源的这些信息。



**提到的链接**：<a href="https://x.com/tmkadamcz/status/1879584048429105238">来自 Tom Adamczewski (@tmkadamcz) 的推文</a>：Hendrycks MATH 刚刚收到了 DMCA 下架通知。该数据集目前已被禁用。https://huggingface.co/datasets/hendrycks/competition_math/discussions/5

  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1328968807472959499)** (6 messages): 

> `NeoX model conversion, Learning ranks attribute, Intermediate size configuration, Layer masking issues, Zero stages incompatibility` 


- **NeoX model conversion struggles**：一位用户在将 NeoX 模型转换为 HF 格式时遇到错误，并引用了一个记录该问题的 [GitHub Gist](https://gist.github.com/aflah02/864092755c28f6f489a68b6fe7c8e313)。
   - 尽管讨论中暗示已有修复方案，但用户在尝试执行转换过程时仍持续遇到错误。
- **tp_ranks 中缺失 'module' 属性**：打印 `tp_ranks[0]` 显示其缺少 'module' 属性，这引起了用户的困惑。
   - 详细的张量结构显示了多个层和参数，但并未解决用户关于 'module' 属性的疑虑。
- **Llama 配置中的中间层大小困惑**：一位用户根据[最近的 PR](https://github.com/EleutherAI/gpt-neox/pull/1309)发现，中间层大小（intermediate size）应该是预期值的 3 倍。
   - 这引发了关于 **32768** 这一标准是否合适的疑问，因为它并不等于 **3x11008**。
- **Llama 2 配置的层掩码问题**：一项更新指出，将 `scaled_upper_triang_masked_softmax_fusion` 设置为 True 会导致模型挂起，这是通过多次消融测试确定的。
   - 关闭此设置可以解决挂起问题，但这与 Llama 2 配置中指定的默认值相矛盾。
- **Zero stages 与模型并行性的不兼容**：一位用户询问了 NeoX 中 Zero stages 2 和 3 与模型并行（model parallelism）及流水线并行（pipeline parallelism）的不兼容性，并指出 DeepSpeed 应该只需要禁用流水线并行。
   - 该疑虑强调了在训练大模型时模型并行性的必要性，并质疑了在缺失该特性时 DeepSpeed 的实用性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gist.github.com/aflah02/864092755c28f6f489a68b6fe7c8e313">error.txt</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/EleutherAI/gpt-neox/issues/971">misindexing when converting llama weights to gpt-neox format · Issue #971 · EleutherAI/gpt-neox</a>：描述 Bug：在运行带有 --pipeline_parallel 参数的 convert_raw_llama_weights_to_neox.py 后，权重检查点缺失了第 2 层和第 3 层（即）：layer_02-model_-model_states.pt layer_03-m...</li><li><a href="https://github.com/EleutherAI/gpt-neox/pull/1309">fix &#39;intermediate_size&#39; in Llama configuration files after the &#39;mlp_type&#39; option was removed by tiandeyu-cs · Pull Request #1309 · EleutherAI/gpt-neox</a>：在 'mlp_type' 选项被移除后，普通类型和 Llama 类型的 MLP 共享相同的实现，现在通过...来指定 "mlp_type"。</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/f7a5a6f9da47de4d4d7cdf776c0832b257f329ef/megatron/training.py#L958-L973)">gpt-neox/megatron/training.py at f7a5a6f9da47de4d4d7cdf776c0832b257f329ef · EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - EleutherAI/gpt-neox
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1328816329834102905)** (40 messages🔥): 

> `Cursor AI funding, Transformer² adaptive models, AI tutoring impact in Nigeria, OpenBMB MiniCPM-o 2.6 model, Synthetic data generation with Curator`

- **Cursor AI 获得 Series B 融资**：令人振奋的消息，@a16z 领投了 [Cursor AI](https://x.com/sarahdingwang/status/1879279307119608142) 的 Series B 融资，这标志着他们在编程领域持续深耕的一个重要里程碑。
   - 社区对此表示支持，相关评论强调了 Cursor 作为 Anthropic 的大客户所带来的潜在影响和成功。
- **Transformer² 革新自适应模型**：来自 @SakanaAILabs 的名为 [Transformer²](https://arxiv.org/abs/2501.06252) 的新系统可以针对各种任务动态调整其权重，模糊了 pre-training 和 post-training 之间的界限。
   - 它与适应和自我改进的概念相联系，其能力类似于章鱼融入环境的方式。
- **AI 辅导在尼日利亚取得显著成效**：最近的一项试验显示，经过六周的 GPT-4 辅导后，尼日利亚学生的学习收益相当于两年，表现优于 80% 的其他教育干预措施。
   - 该试点项目展示了 AI 支持教育的潜力，特别使女孩等弱势群体受益。
- **新型多模态模型：OpenBMB MiniCPM-o 2.6**：新发布的 [MiniCPM-o 2.6](https://x.com/_philschmid/status/1879163439559389307) 拥有 80 亿参数，支持在 edge devices 上进行视觉、语音和语言处理。
   - 凭借出色的基准测试结果，它提供了同类最佳的双语语音能力，并在各种平台上实现了无缝的 multimodal 集成。
- **Curator：一款用于 Synthetic Data 的开源工具**：新工具 [Curator](https://x.com/madiator/status/1879579213554147665) 旨在提高生成用于 AI 训练和评估的 synthetic data 的效率。
   - 它解决了工具链中的空白，并预计将发布更多功能，以增强其在 post-training 数据集中的应用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/sarahdingwang/status/1879279307119608142">来自 Sarah Wang (@sarahdingwang) 的推文</a>：很高兴宣布 @a16z 领投了 @cursor_ai 的 B 轮融资。我们非常激动能继续与 Cursor 团队合作，看他们如何在编程领域掀起风暴。</li><li><a href="https://x.com/emollick/status/1879633485004165375?s=46&t=6FDPaNxZcbSsELal6Sv7">来自 Ethan Mollick (@emollick) 的推文</a>：尼日利亚学生使用 GPT-4 作为导师的新随机对照试验。6 周的课后 AI 辅导 = 2 年的典型学习收益，表现优于 80% 的其他教育干预...</li><li><a href="https://x.com/fchollet/status/1879583863368032432">来自 François Chollet (@fchollet) 的推文</a>：我将与 @mikeknoop 联手创立 Ndea (@ndeainc)，这是一个新的 AI 实验室。我们的重点是：深度学习引导的程序合成（program synthesis）。我们正押注于一条不同的路径，以构建能够实现真正发明能力的 AI...</li><li><a href="https://x.com/samuel_colvin/status/1879627376990224417">来自 Samuel Colvin (@samuel_colvin) 的推文</a>：我们刚刚发布了 @Pydantic AI v0.0.19。这是自我们发布 PydanticAI 以来最大的新功能——图（graph）支持！我最初对图持怀疑态度，但现在我真的很兴奋...</li><li><a href="https://x.com/madiator/status/1879579213554147665?s=46">来自 Mahesh Sathiamoorthy (@madiator) 的推文</a>：我们很高兴宣布 Curator，一个旨在简化合成数据生成的开源库！高质量的合成数据生成对于训练和评估 LLM/Agent/RAG 至关重要...</li><li><a href="https://x.com/hardmaru/status/1879331049383334187">来自 hardmaru (@hardmaru) 的推文</a>：Transformer²：自适应 LLM https://arxiv.org/abs/2501.06252 这篇来自 @SakanaAILabs 的新论文展示了能够根据环境自适应权重的 LLM 的力量。我认为在未来，t...</li><li><a href="https://x.com/SakanaAILabs/status/1879325924887613931">来自 Sakana AI (@SakanaAILabs) 的推文</a>：我们很高兴推出 Transformer²，一个能为各种任务动态调整权重的机器学习系统！https://sakana.ai/transformer-squared 适应性是一种非凡的自然现象...</li><li><a href="https://x.com/emollick/status/1879633485004165375?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Ethan Mollick (@emollick) 的推文</a>：尼日利亚学生使用 GPT-4 作为导师的新随机对照试验。6 周的课后 AI 辅导 = 2 年的典型学习收益，表现优于 80% 的其他教育干预...</li><li><a href="https://x.com/jxmnop/status/1879588733730906143">来自 jack morris (@jxmnop) 的推文</a>：前几天发布了关于模型蒸馏（model distillation）的内容。几乎每个人都给出了他们的理论，包括教授、领先实验室的研究人员、学生、匿名动漫头像发布者，似乎没有...</li><li><a href="https://x.com/synthesiaio/status/1879475235390660833?s=46">来自 Synthesia 🎥 (@synthesiaIO) 的推文</a>：🎉 重大新闻：我们已完成 1.8 亿美元的 D 轮融资 🎉 前方仍有很多工作要做，但前进的道路从未如此清晰。当然，如果没有我们了不起的客户，这一切都不可能实现...</li><li><a href="https://dannguyenhuu.substack.com/p/introducing-the-managed-service-as">介绍：托管服务即软件 (M-SaS) 初创公司</a>：像 AI 这样的技术颠覆，在与商业模式颠覆相结合时，对初创公司尤为强大。</li><li><a href="https://evalplus.github.io/leaderboard.html">EvalPlus 排行榜</a>：未找到描述</li><li><a href="https://podcasts.apple.com/ca/podcast/how-the-hedge-fund-magnetar-is-financing-the-ai-boom/id1056200096?i=1000679726051&l=fr-CA">对冲基金 Magnetar 如何资助 AI 热潮</a>：播客剧集 · Odd Lots · 2024-12-09 · 50 分钟</li><li><a href="https://x.com/_philschmid/status/1879163439559389307?s=46">来自 Philipp Schmid (@_philschmid) 的推文</a>：新的开源 Omni 模型发布！👀 @OpenBMB MiniCPM-o 2.6 是一款全新的 8B 参数、全能（any-to-any）多模态模型，能够理解视觉、语音和语言，并可在手机和平板电脑等边缘设备上运行...</li><li><a href="https://forum.cursor.com/t/anthropic-cannot-sustain-additional-slow-request-traffic-on-claude-3-5-sonnet-please-enable-usage-based-pricing/41361/24?">Anthropic 无法承受 Claude 3.5 Sonnet 额外的慢速请求流量。请启用基于用量的计费</a>：毫无疑问，我们是他们最大的客户。</li><li><a href="https://blogs.worldbank.org/en/education/From-chalkboards-to-chatbots-Transforming-learning-in-Nigeria">从黑板到聊天机器人：在尼日利亚通过提示词改变学习方式</a>：“AI 帮助我们学习，它可以充当导师，它可以成为你想要的任何东西，这取决于你写的提示词，”被朋友称为 “Uyi” 的学生 Omorogbe Uyiosa 说道...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i1a88y/minimaxtext01_">MiniMax-Text-01 (Reddit 帖子)</a></li>

a_powerful_new_moe_language_model/">Reddit - 探索一切</a>：未找到描述</li><li><a href="https://www.papercup.com/blog/speech-ai">语音 AI 已实现 10 倍增长：目的、评估与产品</a>：我们如何利用计算机的语言智能来改善用户体验？</li><li><a href="https://www.minimaxi.com/en/news/minimax-01-series-2">MiniMax - 与每个人共享智能</a>：未找到描述</li><li><a href="https://www.forbes.com/sites/philkirschner/2025/01/15/did-ai-cause-those-layoffs-ny-employers-may-have-to-disclose/?utm_source=chatgpt.com">AI 导致了那些裁员吗？纽约雇主可能必须披露。</a>：纽约州宣布了一项重要举措，要求企业披露明确与采用 AI 相关的裁员，以应对 AI 对劳动力潜在的影响</li><li><a href="https://news.ycombinator.com/item?id=42705935">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1329168344066297948)** (1 条消息): 

> `NVIDIA Cosmos, CES presentation, LLM Paper Club` 


- **CES 上的 NVIDIA Cosmos 发布**：一位成员宣布，<@538564162109046786> 将在 40 分钟后的 [LLM Paper Club 活动](https://lu.ma/pvh0rwa3)中介绍在 CES 上发布的论文/模型 **NVIDIA Cosmos**。
   - 鼓励参与者使用提供的链接加入，参与关于这一新模型的讨论。
- **活动注册提醒**：提醒参与者注册活动，以便接收来自 [Latent.Space](http://Latent.Space) 的新活动通知。
   - 此外，他们被指示点击 RSS 图标将活动添加到日历中。



**提到的链接**：<a href="https://lu.ma/pvh0rwa3">LLM Paper Club (NVIDIA Cosmos) · Zoom · Luma</a>：Ethan He 回归分享来自 Nvidia CES 的最新消息：Cosmos 世界基础模型：https://github.com/NVIDIA/Cosmos---我们需要你志愿参与...

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1329045450677944330)** (6 条消息): 

> `Triton dependency on Torch, cuBLAS equivalent for Triton, Triton pointer type error` 


- **Triton 的功能需要 Torch**：一位用户询问 Triton 是否在运行时依赖 **Torch**，以及是否可以在没有 Torch 的情况下将 **Triton 与 CUDA** 结合使用，理由是存在依赖版本冲突的问题。
   - 他们指出 **Triton 最新版本** 为 3.2，与 Torch 2.5.1 不兼容，这给安装带来了挑战，且难以在不丢失必要文档的情况下完成。
- **探索 Triton 的 cuBLAS 替代方案**：一位用户询问是否存在专门为 **Triton** 定制的 **cuBLAS** 等效项，表明需要 GPU 加速的线性代数运算。
   - 然而，针对这个问题，并没有讨论具体的替代方案或解决方案。
- **Triton 代码中因指针类型导致的 ValueError**：一位用户报告在他们的 **Triton 代码** 中使用 `tl.load` 函数时，遇到了与不支持的指针类型相关的 **ValueError**。
   - 另一位成员建议指针类型应该是 **float** 而不是 **int**，并强调指针应该是代表内存地址的标量。



**提到的链接**：<a href="https://triton-lang.org/">重定向至 https://triton-lang.org/main/index.html</a>：未找到描述

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1329027962284146698)** (3 条消息): 

> `RTX 50x Blackwell cards, Hopper TMA features` 


- **关于 RTX 50x 显卡是否具备 TMA 的推测**：一位成员询问 **RTX 50x Blackwell 显卡** 是否会像 **Hopper** 架构一样具备 **TMA**。
   - 遗憾的是，有人指出在没有 *白皮书* 的情况下，仍然无法给出确切答案。
- **等待特性确认**：关于 **RTX 50x Blackwell 显卡** 的特性（特别是 TMA 功能）普遍存在不确定性。
   - 社区成员对在官方 *白皮书* 发布前缺乏具体信息表示沮丧。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1329107526482132992)** (2 messages): 

> `Batching Tensors to GPU, Torch Compiler Material` 


- **最小化 CPU 开销的 GPU Tensor 批量传输**：一位成员询问是否有办法将一批 **Tensors 传输到 GPU**，而不会因为重复迭代和调用 `.to(device)` 产生高额的 CPU 开销。
   - 他们提到，如附图所示，这种方法导致了过高的延迟。
- **寻求对 Torch Compiler 的深入理解**：另一位成员表示需要 **关于 Torch Compiler 工作原理的资源**，特别是希望看到神经网络编译过程中的全面详细分解。
   - 他们已经找到了 **ASPLOS 2024** 的幻灯片，但无法找到视频内容或更深入的材料来辅助学习。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1328846464159453194)** (2 messages): 

> `MiniMax-01, Lightning Attention Architecture, Open-Source Model Release, Ultra-Long Context Processing, Cost-Effective AI Solutions` 


- **MiniMax-01 发布开源模型**：MiniMax 推出了其最新的开源模型：采用创新架构的 **MiniMax-Text-01** 和 **MiniMax-VL-01**。
   - 这些模型采用了新颖的 **Lightning Attention 机制**，为 AI 模型性能设定了新标准。
- **Lightning Attention 机制亮相**：**Lightning Attention** 架构标志着此类架构的首次大规模实现，为传统的 Transformer 提供了一个强大的替代方案。
   - 这一创新将重新定义 AI 应用中的上下文处理方式。
- **前所未有的 4M Token 超长上下文**：**MiniMax-01** 可以高效处理 **高达 4M 的 Token**，性能显著优于现有模型 20 到 32 倍。
   - 这一能力使 MiniMax-01 在即将增长的 Agent 相关应用中处于领先地位。
- **推出高性价比 AI 解决方案**：新模型提供了行业领先的价格，API 价格仅为 **每百万输入 Token 0.2 美元**，**每百万输出 Token 1.1 美元**。
   - 这种极具竞争力的定价支持了 AI 部署的持续创新。
- **提供更多信息和资源**：用户可以点击[此处](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf)访问详细的研究论文以了解 MiniMax-01 的见解。
   - 更多信息请查看 [MiniMax 新闻页面](https://www.minimaxi.com/en/news/minimax-01-series-2)上的公告。



**提到的链接**：<a href="https://x.com/minimax__ai/status/1879226391352549451">来自 MiniMax (官方) (@MiniMax__AI) 的推文</a>：MiniMax-01 现已开源：为 AI Agent 时代扩展 Lightning Attention。我们很高兴推出最新的开源模型：基础语言模型 MiniMax-Text-01 和视觉...

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1329128790454046801)** (3 messages): 

> `Training with bfloat16, GPU animation insights, GPT-3 architecture access` 


- **权重衰减（Weight decay）对 bfloat16 训练至关重要**：一个值得注意的建议强调，在使用 **bfloat16** 训练时应使用 **权重衰减** 以避免发散，正如[论文](https://arxiv.org/pdf/2310.04415)中图 8 所指出的。
   - 这一实用建议旨在提升使用 bfloat16 的模型性能。
- **A100 动画引起关注**：一条分享的推文指出，每个与 GPU 打交道的人都需要深入理解 @vrushankdes 制作的 **A100** GPU 精妙动画中的见解。
   - 该动画引发了关于 GPU 能力和优化的兴趣与讨论，链接见[此处](https://fixupx.com/fleetwood___/status/1879511438538281350)。
- **关于获取 GPT-3 架构的疑问**：一位成员询问，鉴于 **GPT-3** 并未开源，参与 **MLPerf** 的厂商是如何获取其架构和权重的。
   - 这引发了关于在机器学习竞争格局中，模型架构的可访问性和专有性质的讨论。



**提到的链接**：<a href="https://fixupx.com/fleetwood___/status/1879511438538281350">来自 Fleetwood (@fleetwood___) 的推文</a>：每个与 GPU 打交道的人都需要深入理解这一点。@vrushankdes 动画的 A100 版本。

  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1329073605866291253)** (2 messages): 

> `LLM inference with single GPU, LLM inference with multiple GPUs, VRAM requirements for serving multiple requests, Batch processing requests, Parallelism strategies for multi-GPU setups` 


- **在单张 GPU 上服务多用户**：你可以通过利用 **batched requests** 在单张 GPU 上服务多用户，但最大同时请求数取决于用于 KV cache 的 **可用 VRAM**。
   - 如果没有足够的 VRAM，在处理请求时可能会遇到性能瓶颈。
- **多 GPU 设置推理**：在多 GPU 设置中，由于 **可用 VRAM** 增加，你可以同时处理更多请求，但你需要使用 **parallelism strategy** 来划分模型权重。
   - 这种方法允许通过多张 GPU 高效工作，而不会产生重叠的内存约束。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1328847355726004264)** (2 messages): 

> `MPS Kernel Profiling, Debugging GPU Trace, Metal Compiler Flags` 


- **MPS Kernel Profiling 问题**：一位用户报告了在 MPS 上对 kernel 进行 profiling 的困难，指出尽管使用了 `getMPSProfiler()`，Xcode 仍显示 **空的 GPU trace**。他们确保了 profiling 是通过 `startCapture('kernel', stream)` 启动并由 `stopCapture(stream)` 停止的。
   - *Synchronization*（同步）问题被认为是导致空 trace 的原因，从而引发了进一步的排查。
- **遇到调试信息访问问题**：在成功捕获后，该用户在访问调试信息时面临挑战，其消息中链接的一张令人困扰的截图说明了这一点。他们在运行 Python 设置时设置了 **Metal flags**，包括 `-Wall -Wextra -gline-tables-only -frecord-sources`。
   - 这些 flag 对调试输出的影响似乎尚不明确，这引发了关于优化调试信息可见性的持续讨论。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1328839217790586880)** (1 messages): 

> `Thunder Compute, Cloud GPU Pricing, Y Combinator, Instance Management Tool` 


- **Thunder Compute 推出云端 GPU 服务**：联合创始人宣布了 **Thunder Compute**，旨在降低云端 GPU 的价格，**A100** 实例价格为 **$0.92/hr**，新用户可在其 [官网](https://thundercompute.com) 获得 **每月 $20 免费** 的试用额度。
   - 该服务目前处于 **beta** 阶段，利用 GCP 或 AWS 进行托管以最大化可用性，并包含一个用于实例管理的命令行工具。
- **Y Combinator 校友增强团队实力**：Thunder Compute 的 **三人** 团队最近于今年夏天完成了 Y Combinator 项目，增强了他们在技术领域的公信力和网络。
   - 他们正在积极寻求用户反馈以改进产品，展示了对社区参与的坚定承诺。
- **使用 CLI 轻松管理实例**：用户可以使用 CLI 无缝管理云端 GPU 实例，只需一个简单的安装命令：**pip install tnr**。
   - 该平台承诺提供高效的设置流程，允许用户以极小的麻烦创建符合其规格的实例。



**提及的链接**：<a href="https://thundercompute.com">Thunder Compute: 适用于任何 AI/ML 的低成本 GPU</a>：在 Thunder Compute 上训练、微调和部署模型。开始使用每月 $20 的免费额度。

  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1329223651194835016)** (12 messages🔥): 

> `Popcorn Bot 的 Modal 注册表, GPU 类型处理, 创建 Modal 函数, 使用 nvidia-smi 获取 GPU 能力, Discord 排行榜实现` 


- **系统化 Modal 注册表**：一名成员正在为 popcorn bot 的 **modal registry** 探索一种更优雅的解决方案，特别是尝试在不使用 hack 手段的情况下为不同 GPU 类型创建 **Modal Functions**。
   - 讨论转向了为 Modal 装饰函数的局限性，强调了在结构化基础设施中拥有不同函数版本的重要性。
- **理解 GPU 类型实现**：讨论揭示了在创建 Modal Functions 时需要在装饰器中指定 GPU 类型，这使得函数的偏函数应用（partial application）变得复杂。
   - 其根本目标是创建各种带有固定参数的 Modal 端点，以实现简单性和功能性。
- **GPU 类型的未来内省计划**：在未来的计划中，团队目标是直接在 Modal Functions 内部实现 GPU 类型的内省（introspection），从而可能动态地检查 GPU 规格。
   - 这种预期可能会简化 popcorn bot 中 Modal Functions 的创建，减少每个函数的各种手动配置。
- **利用 Device Query 获取 GPU 信息**：为了识别 GPU 的计算能力（compute capabilities），一名成员建议使用 CUDA 安装中包含的 **deviceQuery** 工具，并强调了以编程方式检查计算规格的能力。
   - 这种方法可能提供一种从 Discord Bot 端设置正确架构的变通方案，同时保持代码整洁。
- **架构设置的灵活性**：提出了一种变通方案，即从 Discord Bot 端设置正确的架构，这可以允许用户调用不同的 GPU 端点。
   - 有趣的是，该功能可以通过针对不同架构进行编译来深入了解性能差异。



**提及的链接**：<a href="https://stackoverflow.com/questions/40695455/what-utility-binary-can-i-call-to-determine-an-nvidia-gpus-compute-capability))">我可以用什么工具/二进制文件来确定 NVIDIA GPU 的计算能力？</a>：假设我有一个安装了单个 GPU 的系统，并且我也安装了最新版本的 CUDA。我想确定我的 GPU 的计算能力。如果我可以...

  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1328833651156455445)** (6 messages): 

> `入门文档, Kernel 选项, 链接资源, 手册创建` 


- **入门文档现已发布**：一名成员分享了为 TK 添加的基础[入门文档](https://docs.google.com/document/d/15-Zvf6e0NLX1si4ml4sUOWCDlXNMtOWKiuo6CKZMEYA/edit?usp=sharing)，并邀请通过评论提供反馈。
   - 已开启“评论模式”，以便用户提出澄清建议或指出可能需要解决的问题。
- **对入门文档的兴趣**：成员们对入门文档表现出极大的热情，表示有浓厚的兴趣进行审阅。
   - 一名成员特别提到他们会立即查看该文档。
- **关于资源链接的讨论**：有人提出了是否根据新文档为 TK 仓库创建一个正式手册的问题，因为大家对链接资源表现出了兴趣。
   - 原始文档所有者表示，目前的文档暂时足够，未来可能会有更广泛的资源。
- **已添加 Kernel 选项**：文档作者提到他们在入门文档中包含了一些 **kernel 选项**。
   - 他们还为任何有兴趣尝试所列选项的人提供帮助。



**提及的链接**：<a href="https://docs.google.com/document/d/15-Zvf6e0NLX1si4ml4sUOWCDlXNMtOWKiuo6CKZMEYA/edit?usp=sharing">TK 入门指南</a>：摘要 本文档规定了如何开始使用 TK 编写 kernel。请随时对本文档提出改进意见或缺失信息的评论。摘要 1 背景...

  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1328888046107623445)** (3 条消息): 

> `使用 LlamaParse 构建 RAG 应用，通过 LlamaIndex workflows 改进知识图谱，LlamaIndex 与 Vellum AI 建立合作伙伴关系` 


- **使用 LlamaParse 构建 RAG 应用**：学习如何使用 [LlamaParse](https://twitter.com/llama_index/status/1879327934378627212)、**LlamaCloud** 和 **AWS Bedrock**，通过高效的文档解析方法构建 **RAG 应用**。
   - 这份分步指南涵盖了从解析 **SEC 文档**到管理索引的所有内容。
- **使用 LlamaIndex 转换知识图谱**：**@neo4j** 的 Tomaz Bratanic 在其[详尽的文章](https://twitter.com/llama_index/status/1879596647967306122)中展示了如何通过在 **LlamaIndex workflows** 中应用 Agent 策略，显著提高知识图谱应用的准确性。
   - 他从一个简单的 text2cypher 实现开始，逐步构建出一个更健壮的 Agent 方法，通过适当的错误处理增强了整体性能。
- **LlamaIndex 与 Vellum AI 达成合作**：团队宣布与 **@vellum_ai** 建立合作伙伴关系，并在此处分享了他们共同进行的调查中获得的宝贵用例数据。
   - 此次合作旨在进一步探索生态系统内的应用和用例。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1329013862527402044)** (23 条消息🔥): 

> `XHTML 转 PDF 转换工具，选择向量数据库，包含 HITL 步骤的 Workflow 设计，Agent 与 Workflow 的区别，用户注册问题` 


- **寻求优质的 XHTML 转 PDF 转换工具**：一位成员分享了他们在 **pandoc**、**wkhtmltopdf** 和 **weasyprint** 等各种库中遇到的困难，并表示只有 Chromium 能提供良好的 XHTML 到 PDF 的转换。
   - 他们提供了 [XHTML 示例文档](https://cdn.financialreports.eu/financialreports/media/filings/3843/2024/10-K/3843_10-k_2024-07-09-133930_ed8fec32-9559-4136-8b93-024a1ba01ffd.xhtml) 和 [HTML 示例文档](https://cdn.financialreports.eu/financialreports/media/filings/4700/2024/10-K/4700_10-k_2024-12-19_32fd81af-71d1-46e4-ab48-d86953034226.html) 的链接。
- **为生产环境选择向量数据库**：由于处理 2 万份文档的成本考量，一位用户正考虑从 **Pinecone** 切换到 **pgvector** 或 **Azure AI search**。
   - 他们就做此决定时需要注意的事项寻求建议，特别是关于与 Azure 集成的方面。
- **HITL Workflow 设计中的挑战**：一位成员详细介绍了他们包含 HITL 步骤的 Workflow 实现，并表达了在暂停等待人工输入时，如何将步骤标记为已完成的挑战。
   - 他们需要确保 Workflow 正确记录 Checkpoint，以反映发出 UserInputRequestEvent 的步骤已完成。
- **理解 Agent 与 Workflow 的区别**：讨论了 Agent 和 Workflow 之间的区别，结论是 Workflow 的范畴更广，而 Agent 是涉及决策和工具使用的更具体的实现。
   - 一位成员强调 “Agent” 的定义可能有所不同，建议用户专注于构建必要的应用，而不是纠结于标签。
- **用户注册问题已解决**：一位用户报告了注册困难，但被告知该问题是由于与身份验证升级相关的临时错误引起的，目前已解决。
   - 另一位用户也表达了同样的担忧，并提到如果有关于此类错误的官方沟通将会很有帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/#vector-store-options-feature-support">Vector Stores - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/rag/">RAG Workflow with Reranking - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/python-agents-tutorial/blob/main/5_memory.py">python-agents-tutorial/5_memory.py at main · run-llama/python-agents-tutorial</a>：来自我们 Python Agent 教程的代码示例。通过在 GitHub 上创建账号，为 run-llama/python-agents-tutorial 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1328901705525366876)** (12 messages🔥): 

> `OpenInterpreter 1.0 功能, 关于智能的 Bora's Law, OI 的 Python 便捷函数, 命令行工具的局限性, AGI 开发路径` 


- **OpenInterpreter 1.0 功能变化**：成员们讨论了新的 **OpenInterpreter 1.0** 版本限制了直接运行代码的能力，现在主要通过命令行交互操作，这可能会降低其易用性。
   - 虽然存在通过外部脚本执行任务的潜力，但早期版本中广受欢迎的即时执行功能似乎有所削弱，引起了用户的担忧。
- **Bora's Law 提出新的智能模型**：一位成员分享了见解，认为 **Bora's Law** 主张智能随约束条件而非算力 (compute) 呈指数级增长，挑战了关于 AGI 开发的主流观点。
   - 该理论提出了一个数学关系式 (I = Bi(C²))，并对目前过度关注规模扩张的 GPT-4 等主流大语言模型 (LLM) 的有效性和方向提出了质疑。
- **希望改进 OI 中的 Python 支持**：一位用户表示有兴趣在 OpenInterpreter 中添加更多 **Python 便捷函数**，以促进任务的高效完成并增强“学习新技能”功能。
   - 大家达成共识，认为提高 Python 能力可以显著提升操作效率，同时保持用户对平台的参与度。
- **1.0 版本中命令行工具的局限性**：用户对 1.0 版本中命令行工具的功能表示担忧，特别是执行以前轻而易举的任务的能力。
   - 用户对可能失去允许即时执行 Python 代码的功能感到惋惜，表明需要一种更集成的方法。
- **关于 AGI 开发路径的辩论**：对话强调了在实现 AGI 路径上的意见分歧，一位用户批评 **OpenAI** 过度依赖算力而非高效的智能缩放。
   - 成员们指出，需要对 AI 开发方法论进行批判性的重新评估，以更好地与 Bora's Law 等新兴见解保持一致。



**提到的链接**：<a href="https://chrisbora.substack.com/p/boras-law-intelligence-scales-with?r=aszci">Bora's Law: Intelligence Scales With Constraints, Not Compute</a>：这是一篇探讨人工智能开发中新兴原则的工作论文。

  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1328852774867435561)** (4 messages): 

> `AI 版权法概述, 超可解释网络 (Hyper-Explainable Networks), 推理时信用分配 (Inference-Time Credit Assignment)` 


- **DougDoug 在 AI 版权法概述方面表现出色**：出乎意料地，**DougDoug** 提供了迄今为止最好的 **AI 版权法 (AI Copyright Law)** 概述，可以在[这里](https://www.youtube.com/watch?v=pt7GtDMTd3k)查看。
   - 他的见解引发了关于技术与法律框架交集的讨论。
- **构想超可解释网络 (Hyper-Explainable Networks)**：一位成员分享了关于**超可解释网络**的远见卓识，这种网络可以对训练数据对输出生成的影响进行评分。
   - 尽管对其可行性持谨慎态度，但这一概念提出了关于数据利用和创作者版税的有趣问题。
- **探讨推理时信用分配 (Inference-Time Credit Assignment)**：讨论涉及了训练数据的**推理时信用分配**概念，作为一种追踪其影响的方法。
   - 虽然这看起来雄心勃勃，但该想法持续引发了关于训练数据在机器学习模型中价值的思考。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1329105298920833134)** (3 messages): 

> `P2B 加密平台, AI21 Labs 对加密货币的态度, 关于加密货币讨论的社区准则` 


- **P2B 提供融资和社区增长**：来自**加密平台 P2B** 的授权代表介绍了他们的服务，声称可以协助加密项目的**融资、上市、社区增长**和**流动性管理**。
   - 他们询问是否可以向 **AI21** 提供更多关于其服务的信息。
- **AI21 Labs 反对加密货币讨论**：一位成员坚定地回应称，**AI21 Labs** 没有参与任何加密货币 (crypto) 项目，且永远不会这样做。
   - 他们警告说，在 Discord 中进一步提及加密货币将导致迅速被封禁。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1329062674373414943)** (1 messages): 

> `AI Agents Workshop, 2025 AI Trends, No-code AI Development` 


- **创建你的第一个 AI Agent 工作坊**：一场名为 **Build Your First AI Agent of 2025** 的工作坊定于 **1 月 16 日星期四晚上 9 点 IST** 举行，邀请所有开发者和初学者学习如何使用代码和 No-code 方法构建 AI Agents。
   - 该活动由 Sakshi 和 Satvik 主持，**仅限受邀参加**，是免费活动，需要注册审批，并包含来自 [Build Fast with AI](http://www.buildfastwithai.com) 和 [Lyzr AI](http://www.lyzr.ai) 的见解。
- **AI Agents 将主导 2025 年**：讨论强调，预计到 2025 年 **AI Agents** 将彻底改变各行各业，承担从个人助理到业务分析师的各种角色。
   - *创建一个 AI Agent 比你想象的要简单*，且有可能无需编程，这为所有参与者拓宽了准入门槛。



**提到的链接**：<a href="https://lu.ma/y8pp2fxc">Create Your First AI Agent of 2025 · Zoom · Luma</a>：AI Agents 是 2025 年的热门话题！从个人助理到业务分析师，这些数字队友正在接管每个行业。最棒的部分是？创建……

  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/)** (1 messages): 

heathcliff_ca: 成本是坚持使用成熟方案的另一个重要原因。
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1329139199114481674)** (2 messages): 

> `Qwen 2.5 fine-tuning, Llama 3.2 character limits, TV pilot script analysis` 


- **关于 Qwen 2.5 微调的咨询**：一位用户对另一位成员如何微调其版本的 **Qwen 2.5 Coder Instruct (7B)** 以及是否在 Hugging Face 上发布表示了兴趣。
   - 他们还询问了可用的更大模型以及其他人使用的成功模型。
- **Llama 3.2 字符限制的挑战**：一位用户报告称，由于字符限制，在分析 **45 页的电视剧试播集剧本**时遇到困难，并表示 **Llama 3.2 3B** 应该能够处理，但错误仍然存​​在。
   - 他们分享了一个[对比链接](https://www.prompthackers.co/compare/llama-3.2-3b/llama-3-8b)，强调了最近发布的 **Llama 3.2 3B** 和 **Llama 3 8B Instruct** 之间的差异，两者具有不同的 Token 容量。



**提到的链接**：<a href="https://www.prompthackers.co/compare/llama-3.2-3b/llama-3-8b">Compare Llama 3.2 3B vs Llama 3 8B Instruct - Pricing, Benchmarks, and More</a>：对比 Llama 3.2 3B 和 Llama 3 8B Instruct 的价格、基准测试、模型概览等。深入对比 Llama 3.2 3B 与 Llama 3 8B Instruct。

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1328849017811238984)** (1 messages): 

> `Ambient Agent Implementation, DSPy Examples` 


- **寻求 Ambient Agent 见解**：一位成员询问了使用 **DSPy** 实现 **Ambient Agent** 的过程，并请求分享经验或实现的示例。
   - 他们特别询问是否有人已经完成此项工作，并能贡献见解以帮助标准化方法。
- **对 DSPy 示例的兴趣**：另一位成员表示有兴趣查看与 Ambient Agent 相关的具体 **DSPy** 示例。
   - 这突显了社区对于如何实际实现 Ambient Agent 的广泛好奇。


  

---


---


---


---


---


---


---


{% else %}


> 完整的频道细分内容已为邮件格式进行截断。
> 
> 如果你想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}