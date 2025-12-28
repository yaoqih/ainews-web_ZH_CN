---
companies:
- anthropic
- openai
- cognition
date: '2024-06-21T07:27:45.182774Z'
description: '由 **Anthropic** 发布的 **Claude 3.5 Sonnet** 被定位为 Claude 3 Opus 的帕累托改进（Pareto
  improvement），其运行速度是后者的**两倍**，而成本仅为**五分之一**。


  它在 **GPQA、MMLU 和 HumanEval** 等基准测试中取得了行业领先（state-of-the-art）的成绩，在视觉任务上甚至超越了 **GPT-4o**
  和 Claude 3 Opus。该模型在编程能力方面展现出显著进步，测试用例通过率从 Claude 3 Opus 的 38% 提升至 **64%**，并能够自主修复拉取请求（pull
  requests）。


  此外，Anthropic 还推出了 **Artifacts** 功能，使用户能够在动态工作区中与 AI 生成的内容（如代码片段和文档）进行交互，类似于 OpenAI
  的代码解释器（Code Interpreter）。此次发布突显了模型在性能、成本效益和编程熟练度方面的提升，预示着大语言模型（LLM）在软件开发领域将发挥越来越重要的作用。'
id: 81ca9896-156e-42c9-ae5d-7dd740c11ba8
models:
- claude-3.5-sonnet
- claude-3-opus
- gpt-4o
original_slug: ainews-claude-crushes-code-92-humaneval-and
people:
- alex-albert
title: Claude 碾压编程：92% HumanEval 评分与 Claude.ai Artifacts 功能
topics:
- benchmarking
- model-performance
- coding
- model-optimization
- fine-tuning
- instruction-following
- model-efficiency
- model-release
- api
- performance-optimization
---

<!-- buttondown-editor-mode: plaintext -->**Claude 3.5 Sonnet 就足够了？**

> 2024年6月19日至6月20日的 AI 新闻。
我们为您查看了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（包含 **415** 个频道和 **3577** 条消息）。
预计节省阅读时间（按每分钟 200 字计算）：**392 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

今日头条名义上是 [Claude 3.5 Sonnet](https://x.com/AnthropicAI/status/1803790676988920098) —— 表面上是 Anthropic 对 GPT-4o 的回应：

 
![image.png](https://assets.buttondown.email/images/4797f203-e0e9-400a-8559-4c5750b6db5a.png?w=960&fit=max)
 

包括声称在 GPQA、MMLU 和 HumanEval 上达到 SOTA：

 
![image.png](https://assets.buttondown.email/images/bb851f1b-63b5-4011-b88a-bfeb88f23a72.png?w=960&fit=max)
 

以及“[在所有标准视觉基准测试中超越了 Claude 3 Opus](https://x.com/AnthropicAI/status/1803790684857536522)”。

https://www.youtube.com/watch?v=dhxrHvgXpSM&embeds_referring_euri=https%3A%2F%2Fwww.anthropic.com%2F&embeds_referring_origin=https%3A%2F%2Fwww.anthropic.com&feature=emb_title

[Model card](https://www-cdn.anthropic.com/fed9cc193a14b84131812372d8d5857f8f304c52/Model_Card_Claude_3_Addendum.pdf) 展示了原本属于 Opus 级别的上下文利用能力现在已扩展到 Sonnet：

 
![image.png](https://assets.buttondown.email/images/d9f417a7-7a67-44db-bce7-85987ac52a09.png?w=960&fit=max)
 

我们没有关于驱动这些变化的太多技术细节，但 Anthropic 将其宣传为对 Claude 3 Sonnet 和 Claude 3 Opus 的帕累托改进（Pareto improvement）：

 
![image.png](https://assets.buttondown.email/images/33a9708c-36b1-432b-86bc-fb113a0bbf66.png?w=960&fit=max)
 

> **Claude 3.5 Sonnet 的运行速度是 Claude 3 Opus 的两倍**。这种性能提升结合极具性价比的定价，使 Claude 3.5 Sonnet 成为处理复杂任务（如上下文敏感的客户支持和编排多步骤工作流）的理想选择。

然而，除了通用能力和效率提升之外，宣传的更大重点是 Claude Sonnet 的编程能力：

> “**Claude 开始变得非常擅长编程并能自主修复 pull requests**。很明显，一年之内，很大比例的代码将由 LLM 编写。” - [Alex Albert](https://x.com/alexalbert__/status/1803804682412007850)

https://www.youtube.com/watch?v=A598ESCoC70

 
![image.png](https://assets.buttondown.email/images/f8867131-8ce1-43ac-8b8c-6eaaff5bf347.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/680e9e6c-6ea1-4911-9688-9523eabaa880.png?w=960&fit=max)
 

这似乎得到了 Claude.ai 发布的 “Artifacts” 功能的支持：

> 一项扩展用户与 Claude 交互方式的新功能。**当用户要求 Claude 生成代码片段、文本文档或网站设计等内容时，这些 Artifacts 会出现在对话旁边的专用窗口中**。这创建了一个动态工作区，用户可以实时查看、编辑并基于 Claude 的创作进行构建，将 AI 生成的内容无缝集成到他们的项目和工作流中。

这似乎是 Anthropic 对 OpenAI 的 Code Interpreter 或 Cognition Labs 的 Devin 的回应。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**Anthropic 发布 Claude 3.5 Sonnet**

- **性能**：[@alexalbert__](https://twitter.com/alexalbert__/status/1803804677701869748) 指出 Claude 3.5 Sonnet 在关键评估中优于竞争对手模型，速度是 Claude 3 Opus 的 **两倍**，成本仅为 **五分之一**。它在理解细微差别、幽默和复杂指令方面表现出显著进步。[@AnthropicAI](https://twitter.com/AnthropicAI/status/1803790676988920098) 强调它现在在 **GPQA, MMLU, 和 HumanEval** 等多个基准测试中超越了 GPT-4o。
- **Artifacts 功能**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1803790681971859473) 推出了 Artifacts，允许用户生成文档、代码、图表、图形或游戏，并显示在对话框旁边进行实时迭代。[@alexalbert__](https://twitter.com/alexalbert__/status/1803804686501507418) 提到由于这个功能，他已经停止使用大多数简单的图表、绘图和可视化软件。
- **编程能力**：在 Anthropic 的内部 pull request 评估中，[@alexalbert__](https://twitter.com/alexalbert__/status/1803804682412007850) 分享了 Claude 3.5 Sonnet 通过了 **64% 的测试用例，而 Claude 3 Opus 为 38%**。[@alexalbert__](https://twitter.com/alexalbert__/status/1803804689538171351) 引用一位工程师的话说，它修复了他们正在使用的开源库中的一个 bug。
- **可用性**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1803790689408332059) 指出该模型在 claude.ai 和 Claude iOS 应用上免费提供。Claude Pro 和 Team 订阅者可获得更高的速率限制。也可通过 Anthropic API, Amazon Bedrock, Google Cloud 的 Vertex AI 获取。

**Ilya Sutskever 的新公司：Safe Super Intelligence (SSI)**

- **目标**：[@ilyasut](https://twitter.com/ilyasut/status/1803472979873128498) 表示他们将直奔目标，通过一支精干的顶尖团队实现革命性突破，专注于安全超级智能，拥有单一的焦点、目标和产品。
- **反应**：像 [@bindureddy](https://twitter.com/bindureddy/status/1803475778019164211) 这样的人称赞其对 AGI 的专注而不过分痴迷于金钱。其他人如 [@DavidSHolz](https://twitter.com/DavidSHolz/status/1803542447206879439) 将其比作 AI 领域的 Yahoo/AOL/pets dot com 时代。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1803611506908529060) 推测这破坏了美中签署具有约束力的 AGI/ASI 条约的可能性。
- **融资**：[@ethanCaballero](https://twitter.com/ethanCaballero/status/1803494001091268756) 质疑 SSI 如何在一年内筹集 100 亿美元，否则他们将“落地即成仁”。

**AI 基准测试与评估**

- **Mixture of Agents (MoA)**：[@corbtt](https://twitter.com/corbtt/status/1803813970018791845) 介绍了 MoA 模型 + FT 流水线，其表现优于 GPT-4，但成本低 25 倍。人类在 59% 的情况下更倾向于 MoA 的输出而非 GPT-4。在 Arena-Hard (84.8) 和 Alpaca Eval (LC 68.4) 上达到了新的 SOTA。
- **Infinity Instruct**：[@_philschmid](https://twitter.com/_philschmid/status/1803679786079830449) 分享了这个包含 300 万样本的去重指令数据集。计划在 6 月底发布 1000 万样本版本。Mistral 7B 的 SFT 实验在 MT Bench 上达到 7.9，将 MMLU 提升了 6%，HumanEval 提升至 50%。
- **τ-bench**：[@ShunyuYao12](https://twitter.com/ShunyuYao12/status/1803849363506237636) 在 Sierra Platform 推出了 τ-bench，用于评估当前基准测试遗漏的关键 Agent 能力：鲁棒性、复杂规则遵循和人类交互技巧。

**梗图与幽默**

- 关于 AI 鼠标上的 Logi AI Prompt Builder 的梗图：[@nearcyan](https://twitter.com/nearcyan/status/1803583533690008030)
- 关于 AI 领域的 Yahoo/AOL/pets dot com 时代的梗图：[@DavidSHolz](https://twitter.com/DavidSHolz/status/1803542447206879439)
- 关于 Claude 3.5 的加密莎士比亚十四行诗：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1803774865473696237)
- 关于 SSI 筹集 100 亿美元融资的梗图：[@bindureddy](https://twitter.com/bindureddy/status/1803546758406086767)

---

# AI Reddit 摘要回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 公司与进展**

- **Dell 与 NVIDIA 合作打造“AI 工厂”**：在一条 [推文](https://x.com/MichaelDell/status/1803385185984974941) 中，Michael Dell 宣布 Dell 正与 NVIDIA 合作建设“AI 工厂”，为 “xAI 的 grok” 提供动力，暗示这两家科技巨头之间的一项重大 AI 基础设施计划。
- **Anthropic 的 Claude AI 展示了强大的法律推理能力**：根据一项 [分析](https://adamunikowsky.substack.com/p/in-ai-we-trust-part-ii)，Anthropic 的 Claude AI 在 **37 个案例中的 27 个** 中与最高法院的裁决一致，展示了其理解和推理复杂法律问题的能力。
- **Meta 的 Chameleon 语言模型训练数据集曝光**：Meta 的 Chameleon AI 模型文件显示，它是 [基于](https://www.reddit.com/r/LocalLLaMA/comments/1dk5a5q/chameleon_model_files_list_the_datasets_meta_used/) 涵盖法律内容、代码、安全/审核数据等多样化数据集训练而成的，这让外界得以洞察 Meta 优先考虑的知识领域。

**AI 能力与基准测试**

- **Microsoft 开源 Florence-2 视觉模型**：Microsoft 以开源许可证 [发布](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de) 了其 Florence-2 视觉基础模型，该模型在视觉问答、目标检测和图像描述等任务中表现出 [强劲性能](https://i.redd.it/31t6f0q9ti7d1.png)。
- **LI-DiT-10B 宣称超越 DALLE-3 和 Stable Diffusion 3**：一张 [对比图](https://i.redd.it/4dz9eg6heh7d1.png) 表明，LI-DiT-10B 模型在图文对齐和生成质量上超过了 DALLE-3 和 Stable Diffusion 3，并计划在进一步优化后推出公共 API。
- **基于 Llama 的 70B 参数故事写作模型发布**：DreamGen Opus v1.4 是一款基于 Llama 3、专注于故事生成的 70B 参数语言模型，现已 [发布](https://www.reddit.com/r/LocalLLaMA/comments/1djo3of/llama_3_70b_roleplay_story_writing_model_dreamgen/)。随之发布的还有详细的使用指南和示例提示词，展示了其创意写作能力。

**讨论与观点** 

- **对 Stability AI 业务前景的担忧**：一篇 [评论文章](https://i.redd.it/6lns2brtsm7d1.png) 针对 Stable Diffusion 3 发布过程中出现的问题及其他因素，对 Stability AI 商业模式的可持续性和未来前景提出了质疑。

**梗图与幽默**

- AI 梗图涉及了 AI 初创公司的 [快速增长](https://i.redd.it/7tc9ugtk2n7d1.png)，[调侃](https://i.redd.it/m0pt9dvrqj7d1.jpeg) 了 OpenAI 名不副实的封闭模型，并讽刺了 Stability AI 对 Stable Diffusion 3 问题的 [处理方式](https://i.redd.it/l2d0f7wxfi7d1.png)。
- 一张梗图 [想象](https://i.redd.it/o3rirwk52i7d1.jpeg) 了 Doc Brown 对 2045 年 AI 进展的震惊反应，以此致敬技术进步的飞速。

---

# AI Discord 回顾

> 摘要之摘要的摘要

**1. 模型性能优化与基准测试**

- **[量化 (Quantization)]** 技术如 **AQLM** 和 **QuaRot** 旨在保持性能的同时，在单个 **GPU** 上运行大型语言模型 (**LLMs**)。例如：在 RTX3090 上运行 **Llama-3-70b** 的 [AQLM 项目](https://github.com/Vahe1994/AQLM)。

- 通过 **Dynamic Memory Compression (DMC)** 等方法努力**提升 Transformer 效率**，在 **H100 GPUs** 上可能将吞吐量提高多达 370%。例如：@p_nawrot 发表的 [DMC 论文](https://arxiv.org/abs/2403.09636)。

- 关于**优化 CUDA 操作**的讨论，例如融合逐元素操作（fusing element-wise operations），使用 **Thrust 库的 `transform`** 来实现接近带宽饱和的性能。例如：[Thrust 文档](https://nvidia.github.io/cccl/thrust/api/groups/group__modifying.html#function-for-each)。

- 在 **AlignBench** 和 **MT-Bench** 等基准测试中对**模型性能**进行比较，**DeepSeek-V2** 在某些领域超越了 GPT-4。例如：[DeepSeek-V2 发布公告](https://x.com/deepseek_ai/status/1787478986731429933)。

**2. 微调挑战与提示词工程策略**

- 在将 **Llama3** 模型转换为 GGUF 格式时，存在**保留微调数据**的困难，并讨论了一个[已确认的 bug](https://github.com/ggerganov/llama.cpp/issues/7062)。

- **提示词设计 (Prompt design)** 和使用正确模板（包括文本结束标记 end-of-text tokens）对于微调和评估期间影响模型性能的重要性。例如：[Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47)。

- **提示词工程 (Prompt engineering)** 策略，如将复杂任务拆分为多个提示词，研究 **logit bias** 以获得更多控制。例如：[OpenAI logit bias 指南](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api)。

- 教导 **LLMs** 在不确定时使用 `<RET>` 标记进行**信息检索**，从而提高在低频查询上的表现。例如：[ArXiv 论文](https://arxiv.org/abs/2404.19705)。

**3. 开源 AI 发展与协作**

- 发布 **StoryDiffusion**，这是一个采用 MIT 许可证的 Sora 开源替代方案，尽管权重尚未发布。例如：[GitHub 仓库](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file)。

- 发布 **OpenDevin**，这是一个基于 Cognition 的 Devin 的开源自主 AI 工程师，举办了[网络研讨会](https://lu.ma/fp0xr460)且在 GitHub 上的关注度日益增长。

- 呼吁在开源**机器学习论文**上进行协作，该论文旨在预测 IPO 成功，托管在 [RicercaMente](https://edopedrocchi.github.io/RicercaMente/Projects/IPO/indexIPO.html)。

- 围绕 **LlamaIndex** 集成的社区努力，包括在更新后遇到的 Supabase Vectorstore 问题和包导入问题。例如：[llama-hub 文档](https://github.com/run-llama/llama-hub/tree/main#how-to-add-a-loadertoolllama-pack)。

**4. 多模态 AI 与生成模型创新**

- **[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** 专注于提升聊天交互体验，而 **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** 则精进了编程能力。

- **[Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** 模型通过 WebGPU 将强大的 AI 聊天机器人带入浏览器。

- 结合 **Pixart Sigma + SDXL + PAG** 旨在实现 **DALLE-3** 级别的输出，并具有通过微调进一步完善的潜力。

- 开源项目 **[IC-Light](https://github.com/lllyasviel/IC-Light)** 专注于改进图像重光照（image relighting）技术。

**5. 其他**

- **Stable Artisan 将 AI 媒体创作引入 Discord**: Stability AI 推出了 **Stable Artisan**，这是一个集成了 **Stable Diffusion 3**、**Stable Video Diffusion** 和 **Stable Image Core** 等模型的 Discord 机器人，用于[直接在 Discord 内进行媒体生成和编辑](https://bit.ly/4aiVy6C)。该机器人引发了关于 **SD3 开源状态**以及引入 **Artisan 作为付费 API 服务**的讨论。

---

# 第一部分：高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Ollama 获得 Unsloth 支持**：工程师们对 Unsloth AI [对 Ollama 的新支持](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing)表现出浓厚兴趣，官方提供了一个用于测试的 Colab 链接，并向早期采用者征集 Bug 报告。

**分布式训练的提炼**：深入探讨了**分布式数据并行 (DDP)**，重点关注跨多 GPU 扩展模型，强调了模型准确性、token 以及训练中上下文处理的重要性。

**Anthropic 创新推出 Claude 3.5 Sonnet**：Anthropic 发布 [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) 的公告吸引了工程师们的关注，因为它树立了新的行业模型标准。

**澄清 CausalLM 的困惑**：一系列消息解决了关于训练期间 **causalLM loss 计算**的困惑，将其与传统 Masked LM 任务中的 loss 计算进行了比较，指出了其在下个词预测准确性方面的聚合特性。

**部署难题与预训练查询**：AI 工程师讨论了模型部署中的实际挑战和解决方案，例如使用 Conda 解决 **llama3 库版本兼容性**，以及**持续预训练和微调 instruct 模型**的策略，相关的有益讨论可以在[这里](https://discuss.huggingface.co/t/what-is-the-purpose-of-save-pretrained/9167/2?u=aflah)找到。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o 激发工程好奇心**：工程师们辩论了 **GPT-4o** 的推理能力，注意到它相对于其他模型的进步，并期待它在更大的模型（如假设的 GPT-5）中的实现。关注点集中在 AI 的理论极限和实际应用，特别侧重于 **OpenAI** 的产品与 **Claude 3.5** 和 **Google’s Gemini** 等竞争对手的对比。

- **挑战 ASI 的边界**：关于人工超智能 (ASI) 的讨论提出了关于实现“上帝般智能”及其伦理影响的问题。辩论在对 ASI 局限性的担忧与对其前所未有的技术进步的热情之间摇摆。

- **实际 Prompt Engineering 的苦恼**：工程师们分享了对 OpenAI assistants 中 **token 使用量**的沮丧，简单的命令却产生了意想不到的高 token 计数。在创意方面，**DALL-E** 在生成不对称图像方面的局限性促使人们建议使用更多样化的描述性短语，但承认效果有限。

- **工程师的心声：呼吁更新与替代方案**：用户对 OpenAI 停滞不前的更新（例如 **Sam Altman** 承诺的语音发布）表示不满，并讨论了使用 **Google’s AI Studio** 的聊天体验，注意到 **Gemini** 在处理长上下文窗口方面的卓越性能。

- **AI 在长输出和系统指令方面的实际局限性**：**ChatGPT** 被指出因其 token 限制而在生成可靠的长输出方面存在困难。此外，关于 **GPT-3.5-turbo-0125** 有时会忽略系统指令的报告，导致了需要更清晰、更简化的指令以确保合规性的建议。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stability AI 首席执行官备受关注**：Shan Shan Wong 已被确认为 Stability AI 的 CEO。一些成员调侃未来可能会分享独家更新，但未提供具体细节。

- **AI 创作者的许可困扰**：由 **stabilityai/stable-diffusion-xl-base-1.0** 模型生成的 AI 图像引发了关于许可的疑问，成员们正在探索使用各种 Creative Commons 许可。该模型在 CreativeML Open RAIL++-M License 下运行。

- **艺术社区频道被砍**：由于活跃度低和机器人垃圾信息，Cascade 以及其他艺术相关的社区频道被删除，这在成员中引起了骚动。一名管理员指出，如果社区表现出重新关注的兴趣，这些频道可以恢复。

- **Turbo 与微调模型的对决**：一些成员看重 Turbo 模型在速度和灵活性方面的价值，而另一些成员则主张在需要特定细节或概念准确性的任务中使用微调模型（Finetuned models），如 Juggernaut 和 Pony。

- **介绍 Mobius，去偏见模型**：Mobius 模型被强调为去偏见扩散模型的领导者，它利用领域无关（domain-agnostic）的方法来减少偏见。讨论中提到了关于其大小和要求的问题，例如 clip skip 3 及其 Lora 兼容性。

链接：[Hatsune Miku Gif](https://tenor.com/view/hatsune-miku-miku-hatsune-earthquake-plush-miku-death-gif-4018907532159793300), [Mobius on Civitai](https://civitai.com/models/490622/mobius), [ComfyUI_TensorRT GitHub](https://github.com/comfyanonymous/ComfyUI_TensorRT), [Google Colab notebook](https://colab.research.google.com/github/mkshing/notebooks/blob/main/stable_video_diffusion_img2vid.ipynb#scrollTo=9AZDrh-SUDt2)。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity CEO 对话 Lex Fridman**：在一次引人入胜的播客节目中，Perplexity 的 CEO 讨论了 AI 对搜索和互联网的强大影响，并引用了 Larry Page 的格言“用户永远不会错”来激发灵感。视频可在 [YouTube](https://youtu.be/e-gwvmhyU7A) 上观看。

- **技术故障与突破**：用户遇到了 Pro Search 在开启时无法找到来源的问题，这与 iPhone 应用的表现不一致，引发了社区的升级反馈。同时，人们对升级到 Claude 3.5 Sonnet 充满期待，尤其是它在创意写作方面的潜力，尽管其具体的集成方式仍是一个令人好奇的点。

- **AI 伦理成为焦点**：一篇 Wired 的文章引发了关于 Perplexity 是否遵守 robots.txt 的辩论，一些用户为 AI 在检索用户请求信息中的作用辩护，而另一些用户则敦促进行更严格的审查。

- **前景与迷幻剂**：对话从英国文学专业的高薪职业路径转向了围绕 Lululemon 收益的财务投机，并与关于迷幻体验如何改变个人信念系统的讨论形成了鲜明对比。

- **API 适应性的阵痛**：Perplexity API 展示了坚实的性能，尤其是在运行大型 LLM 方面表现出色，但因其受限的定制化以及缺乏诸如通过 API 访问 Pages 等功能而受到批评。不过，通过 [Perplexity API 设置页面](https://www.perplexity.ai/settings/api) 重置 API 密钥已变得非常简单。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Character.AI 推动高效 INT8 训练**：Character.AI 致力于通过 **INT8 optimization** 实现 AGI，其推理查询量已达到 **Google Search 访问量的 20% 左右**。关于其是否使用 **Adaptive Quantization** (AQT) 的探讨仍在继续。[阅读更多](https://research.character.ai/optimizing-inference/)。

**Kernel Profiling 与 Triton 攻坚**：**Nsight Compute** 是分析 CUDA kernels 以消除代码库性能 Bug 的首选工具，而 **Triton 3.0.0** 被誉为修复了众多问题的版本，并提供了详细的升级指南。[GitHub profiling 脚本](https://github.com/AnswerDotAI/bitlora/blob/master/benchmarks/forward_kernel/profile_forward_kernel.sh) 以及 [Kernel profiling YouTube 资源](https://www.youtube.com/playlist?list=PL5B692fm6--ukF8S7ul5NmceZhXLRv_lR)。

**新兴 AI 突破**：**Qwen2**、**DiscoPOP** 和 **Mixture of Agents** 的进展正在塑造 AI 的未来，并具有提升 **LLM performance** 的潜力。Open Empathic 和 Advisory Board GPT 等正在展开的研究项目为模型利用提供了创意视角。[AI Unplugged 报道](https://datta0.substack.com/p/ai-unplugged-13-qwen2-discopop-mixture)。

**通过 Quantization 进行优化并引入 FPx**：在精调细节的同时，社区评估了 **tinygemm** 的兼容性，迎接 **FP8 quantization** 的挑战，并思考 **XLA 与量化模型的集成**。**uint2 quantization** 与 **FP16** 的性能对比显示出显著的加速效果。[量化代码参考](https://github.com/pytorch/ao/blob/e6460c22284df669b95dc912114cc29b40df2030/torchao/quantization/quant_primitives.py#L280-L289)。

**利用新技术提升硬件性能**：在 **H100 box** 上对 1558M 模型的实验表明，其速度比 A100 快 2.5 倍，从前沿硬件进步中获得了切实的效率提升。速度优化持续成为焦点，文中提到了通过 **torch compile max autotune** 实现了 20% 的提升。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 2 Theta 超越 GPT-4**：**Hermes 2 Theta 70B** 在 MT-Bench 上获得了 9.04 分，超越了 GPT-4-0314 的 8.94 分，展现了更强的创造力和能力。它是 Nous Research、Charles Goddard 和 Arcee AI 合作的产物，FP16 和 GGUF 版本均已在 Hugging Face 上线。

- **General 频道热议 Claude 3.5 Sonnet**：社区对 **Claude 3.5 Sonnet** 的发布反应热烈，称赞其速度和问题解决能力，认为它是 AI 能力的一次飞跃。同时，关于模型解析的讨论强调了将特定模型的 tool calls 转换为标准格式的重要性，并建议将反向模板（reverse templates）整合进 `tokenizer_config.json`。

- **新资源预告**：成员们在 #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1253102983932154010) 频道暗示即将发布新资源，引发了同行的好奇和期待。

- **模型集成技术受到关注**：#[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1253065913649856542) 频道中的一项建议描述了一种将工具直接合并到模型 prompts 中的方法，这可能有助于更流畅地使用多个 AI 工具。

- **音乐视频活跃讨论氛围**：在一段轻松的交流中，一名成员在 #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1253394167527247964) 频道分享了一个 [YouTube 音乐视频](https://youtu.be/E3Yt_qLUGJY)，为技术讨论增添了调剂。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **直接数据流式传输指日可待**：用户强调了 **Torchtune** 目前的局限性，因为内存中的数据集仍需从 **Hugging Face (HF) locations** 下载到本地磁盘。他们正在转向 *streaming datasets*（流式数据集）以绕过磁盘保存。

- **配置 HF 数据集：轻而易举**：社区一致同意在 `torchtune.dataset.chat_dataset` 中使用 `conversation_style: openai` 配置 HF 数据集，这应该能与 Torchtune 无缝集成。

- **序列长度争论定格在 8k**：关于 **llama3** 最大序列长度的讨论达成共识，最高可达 8192 个字符，尽管有人对 **VRAM capacity limitations**（VRAM 容量限制）表示担忧。

- **内存管理速成课**：针对模型训练期间（特别是使用 qlora 和 lora 时）出现的 RAM 相关崩溃，建议将层 offload 到 **CPU**，并解决 **ROCm** 设置中的怪癖以确保运行顺畅。

- **探索 ROCm 迷宫**：关于为 **AMD GPUs** 设置 **ROCm** 的讨论揭示了几个问题，但社区分享的资源（包括一个关于在 **6900 XT** 上成功运行 ROCm 的 Reddit 帖子）被证明非常有价值。为了简单有效，从源码构建（Building from source）是推荐的途径。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**AI 集成在脚本编写中非常方便**：用户讨论了在 VSCode 中集成 **Stable Diffusion**，并建议通过编辑器内的终端运行命令。还有人提到使用 **stable-diffusion-3-medium-diffusers** 模型作为 Stable Diffusion 3 中缺失模型索引的变通方案。

**LLM 关于药物名称和微调问题的辩论**：NLP 模型表现出对通用药物名称（如对乙酰氨基酚）而非品牌名（如泰诺）的偏好，这暗示了可能存在数据污染，正如[这项研究](http://arxiv.org/abs/2406.12066)所讨论并在[排行榜](https://huggingface.co/spaces/AIM-Harvard/rabbits-leaderboard)上展示的那样。同时，一位成员在利用 TRL 和 QLoRa 微调 **Llama 3** 时遇到了问题，并链接了他们的代码和潜在解决方案。

**挑战多表数据合成的假设**：一位成员审视了生成合成多表数据库（特别是包含日期列的数据库）的挑战，一篇[文章](https://mltechniques.com/2024/06/15/synthesizing-multi-table-databases-model-evaluation-vendor-comparison/)比较了三家数据合成供应商。此外，一篇[论文](https://doi.org/10.48550/arXiv.2305.11554)提出了 **ToolkenGPT**，这是一种让 LLM 通过 tokenization 使用外部工具的方法，旨在绕过微调和 in-context learning 的限制。

**蛋白质预测获得并行处理能力提升**：用户庆祝了 **BulkProteinviz** 的更新，这是一个开源的蛋白质结构预测工具，现在支持同时进行多个预测。这可能会显著加速计算生物学的研究。

**Llama 3:70B 寻求规模升级**：一位工程师询问了如何增加通过 **Ollama** 管理的 **Llama 3:70B** 训练数据的技巧，试图从 40GB 增加到 200GB，以进行更稳健的本地训练。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**MLIR 的 Kgen Dialect 引发困惑**：社区成员对 MLIR 中的 **`kgen` dialect** 感到困惑，因为它缺乏公开文档，一位用户形容其代码非常*混乱*。在 MLIR 中实现 **256-bit integers** 的建议解决方法包括使用 **`SIMD[DType.int64, 4]`** 或 **定义 `i256` 类型**，参考自 [GitHub 引用](https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/simd.mojo#L231-L232)。

**Mojo 乘上开源浪潮**：成员们获悉 Mojo 语言已部分开源，其编译器将逐步开源，详见 [博客文章](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source)。讨论揭示了 Mojo 目前在生产环境中的实际局限性，并建议在成熟之前不要将 Mojo 用于复杂的自动化工作。

**通过包管理器和直播演进 Mojo 生态系统**：Mojo 的包管理器正在开发中，社区提出了诸如 [Hammad-hab 的 `pkm`](https://github.com/Hammad-hab/pkm) 等建议。此外，社区受邀参加 **Modular 社区直播**，讨论 MAX Engine 和 Mojo 的进展，可在 [YouTube](https://www.youtube.com/watch?v=uookgZ7Ojg8) 上观看。

**Modular “引擎室”中紧迫问题的蓝图**：针对 MAX Engine 中的 `execute` 函数提供了详细说明，指出它可以接收可变参数 `NamedTensor` 或 `Tuple[StringLiteral, EngineNumpyView]`，如 [Model 文档](https://docs.modular.com/max/api/mojo/engine/model/Model#execute) 所述。

**Nightly 版本，谨慎处理 Mojo**：宣布发布最新的 Mojo 编译器版本 `2024.6.2005`，用户可以查看 [更新日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 了解详情。此外，还推出了一款名为 "mojo_dev_helper" 的新工具，供标准库贡献者使用，更多详情见 [GitHub](https://github.com/rd4com/mojo_dev_helper)。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **垃圾信息风暴袭击 Discord**：Discord 公会内的多个频道受到垃圾信息机器人的困扰，这些机器人推广包含 OnlyFans 泄露内容的 “18+ 免费内容”，并附带一个非法 Discord 服务器链接。所有实例中共享的邀请 URL 为 [加入 Discord 服务器！](https://discord.gg/2AFWP2Qd2r)。

- **社区采取行动打击垃圾信息**：在大量不当内容出现后，成员们采取行动举报并屏蔽了垃圾信息的来源。已确认对一名被举报的用户采取了措施，表明了社区内的警惕性。

- **Nitro Boost 赠送诈骗警示**：除了成人内容垃圾信息外，还提到了所谓的 Nitro Boost 赠送活动，这很可能是与同一垃圾 Discord 链接相关的网络钓鱼尝试或诈骗的一部分。

- **重复的目标频道**：垃圾信息并非孤立存在，而是出现在从 #[committers](https://discord.com/channels/1122748573000409160/1122748682475950142/1253138286860304544) 到 #[ai-explained-cartoons](https://discord.com/channels/1122748573000409160/1249527870750195802/1253138355726843926) 的各个频道中，表明这是一个普遍存在的问题。

- **成员的担忧与迅速响应**：在垃圾信息泛滥期间，成员们表达了对需要采取迅速行动的担忧，并得到了肯定的回应，表明社区在处理此类干扰方面反应迅速且积极。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**LM Studio 0.2.23 的新里程碑**：LM Studio 0.2.23 版本因其速度提升而备受赞誉，极大地提高了效率。用户反映在运行 Deepseek Coder v2 时遇到了“不支持的架构”错误，但指出通过禁用 flash attention 并使用 0.2.25 版本的 deepseek coder 预设可以缓解该问题。

**硬件难题与 GPU 辩论**：讨论围绕大型语言模型（LLM）对 VRAM 的巨大需求展开，建议为 34GB 模型配备 38GB+ 的 VRAM 以获得流畅性能，并辩论了 Nvidia 3090 与 4090 在性价比和 VRAM 容量方面的优劣。AMD 7900XT 对 LLM 的适用性受到质疑，原因在于 ROCm 支持问题以及在某些系统上的通用检测故障。

**寻求前端灵活性**：工程师们正在探索在各种设备上部署本地 LLM 服务器的前端选项，[every-chatgpt-gui](https://github.com/billmei/every-chatgpt-gui) 和 [awesome-chatgpt](https://github.com/uhub/awesome-chatgpt) 仓库是常见的起点。一些人对 llama 相关 subreddit 中过于激进的自动审核表示不满。

**模型讨论中的技术特性**：Nvidia 的新故事叙述模型因其在强化内容方面的平衡而引起关注。Opus 的上下文容量范围引发了辩论，人们寄希望于扩展限制。DeepSeek Coder V2 Lite 有一种特殊的倾向，除非使用旧模板，否则会偏向于使用中文回答。在进行了一些实际测试后，用户表现出对新模型优于 Midnight Miqu 产品的偏好。

**Beta 版和技术预览版的瓶颈**：LM Studio 的最新 Beta 测试显示，在 Linux Mint 上存在 Nvidia 4070 GPU 的检测问题，以及 DeepseekV2 模型的运行故障。M1 Mac 用户在利用 GPU 加速时面临不一致的情况，而 AMD 用户则被引导安装 ROCm 软件包以确保 GPU 兼容性。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **更快、更便宜、更好的 Claude**：Anthropic 推出了新的 [Claude 3.5 Sonnet](https://openrouter.ai/models/anthropic/claude-3.5-sonnet)，声称其性能优于前代 Opus，同时价格便宜 5 倍，速度快 2.5 倍；它除了标准版本外还提供自我审核版本，价格详情见 [推文](https://x.com/OpenRouterAI/status/1803802819708739717)。
- **Stripe 额度显示故障**：导致额度排队错误的 Stripe 支付问题已得到解决，过去半小时内受影响的交易已成功处理。
- **Nemotron 的托管挑战**：Nemotron 在托管商中并不受欢迎，主要是因为其 3400 亿参数的庞大体积以及与流行推理引擎缺乏兼容性。
- **Dolphin Mixtral 的开放许可优势**：Dolphin Mixtral 1x22b 模型获得了赞誉，该模型可在 [HuggingFace](https://huggingface.co/cognitivecomputations/dolphin-2.9.1-mixtral-1x22b) 上获取，并被认为有潜力替代 Codestral，同时避免了许可限制。
- **澄清 DeepSeek-Coder V2 的限制**：解决了关于 DeepSeek-Coder V2 上下文长度的困惑；尽管其模型卡片声称支持 128K，但进一步澄清显示，由于 OpenRouter 的托管限制，目前上限为 32K。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **1B 互联网争论解决器？成本 vs 实用性**：关于专门训练一个 1B 模型来解决互联网争论的可行性展开了激烈辩论，关注点在于高昂成本与模型训练时间的对比，而在一个 H100 节点上，该训练时间可以缩短至两天以内。

- **技术困扰：Selectolax、Lexbor 和 NumPy 的痛苦**：工程师们面临 **Selectolax** 和 **Lexbor** 导致段错误（segmentation faults）的技术问题，并且在 `lm-eval-overview.ipynb` 中苦于 **NumPy 2.0** 的兼容性问题，即使降级后仍未解决。

- **Warc 与速度狂魔**：关于 **CC Warc 文件处理** 的讨论中，成员们分享了各种优化方案，有报告称使用 100 个进程处理一个 Warc 需要 60 秒，而另一种方法则利用 32 个进程进行并行处理。

- **Data Hub 盛宴**：**Epoch AI 的 Data Hub** 现在编目了 800 多个模型，旨在造福研究人员、政策制定者和利益相关者，并指出正如一份 CNAS 报告所讨论的，到 2030 年代前沿 AI 可能会出现计算爆炸。

- **研究财富：从 Token 数据集到 Slot SSMs**：研究频道的讨论涵盖了多样化的话题，包括来自 [DCLM-Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) 的 4T Token 数据集的性能影响，一篇[论文](https://arxiv.org/abs/2406.12272)中介绍的用于更好序列建模的 **SlotSSMs**，模型在医疗应用中难以处理药物品牌名的问题，训练后增强技术如 **LAyer-SElective Rank reduction (LASER)**，以及用于解决 LLM 中表面形式竞争（surface form competition）的领域条件 PMI。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Claude 3.5 Sonnet 占据领先地位**：[Anthropic](https://x.com/anthropicai/status/1803790676988920098?s=46) 推出了 **Claude 3.5 Sonnet**，宣称拥有更快的速度和更高的成本效益，并承诺未来将推出名为 Haiku 和 Opus 的模型。与此同时， [Character.AI](https://research.character.ai/optimizing-inference/) 专注于为其 AGI 优化推理，能够每秒处理 20,000 次查询——相当于 Google 搜索量的 20%。

- **青少年驱动的 AI 参与度**：Character.AI 的会话时长显著增加，尤其是在年轻用户中，超过了 ChatGPT 的参与度。此外，**Claude 3.5 Sonnet** 在 aider 的代码编辑排行榜上名列前茅，尤其擅长 "whole" 和 "diff" 编辑格式。

- **AI 安全领域的“酸葡萄”心理？**：成员们对 AI 安全的信任和实施表示怀疑，带有讽刺性的“相信我，兄弟”情绪，并引用了 [Eliezer Yudkowsky 对 AI 对齐计划的挑战](https://x.com/ESYudkowsky/status/1803676608320192617)。Scott Aaronson 对 **Ilya Sutskever** 寻求理论上稳健的对齐立场的叙述也浮出水面。

- **Kling 胜过 Sora**：[快手](https://kling.kuaishou.com/en)发布了 **可灵 (Kling)**，这是一款向公众开放的文本生成视频 AI 模型，它提高了标准，可以生成 1080p、30fps 的两分钟视频，这与 OpenAI 的 **Sora** 不同。此外，人们对 Meta 使用 5000 个 V100 生成合成数据的做法感到好奇，Nathan Lambert 正在重新审视这一话题。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **CrewAI 与 LlamaIndex 联手**：CrewAI 宣布通过与 LlamaIndex 集成来增强多 Agent 系统，提供了一种定义 Agent “crew” 的方法，这些 Agent 可以利用 LlamaIndex 的功能来执行任务。有关此集成的详细信息可以在[他们最新的博客文章](https://t.co/8Tjk888RL1)中找到。

- **AI Fair 的未来演讲者**：LlamaIndex 的创始人计划在 AI Engineer's World's Fair 上发表演讲，于 6 月 26 日讨论 *知识助手的未来 (Future of Knowledge Assistants)* 并发布一些重大公告，并在 6 月 27 日进行另一场会议。欲了解更多信息，爱好者可以[在此了解更多](https://t.co/JMoAOAA4bI)。

- **向量存储定制查询**：工程师们正在探索 LlamaIndex 的 VectorStoreIndex 的灵活性，提出了关于添加序列标识符、自定义相似度分数和异步节点检索的问题，尽管由于当前的限制，某些功能可能需要自定义实现。

- **从文档生成知识**：分享了关于使用 LlamaIndex 的 `DatasetGenerator` 从 PDF 生成问题的讨论，包括一个利用 OpenAI 模型完成该任务的示例。

- **索引持久化变得简单**：对话重点讨论了存储持久化索引，强调了在 LlamaIndex 中使用 `storage_context.persist()` 来存储 DocumentSummaryIndex，并附带了实用的代码说明。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Nemotrons API 速度提升**：成员们报告了 **Nemotrons API** 的改进，强调了显著的速度提升以及新发布的 **reward model**。

- **Turbcat 还是 Turbca？**：对 **Turbcat** 的争论进行了澄清；它是模型名称，而 **Turbca** 是其背后的开发者。数据集配置和 tokenization 方法的问题引发了讨论和担忧。

- **Tokenization 的困扰与解决方案**：关于 tokenization 以及如何处理 **end of text (EOT)** token 展开了激烈的辩论，一名成员展示了 [Multipack with Flash Attention documentation](https://openaccess-ai-collective.github.io/axolotl/docs/multipack.html) 以展示最佳实践。

- **Qwen 模型的偏见揭秘**：社区对 **Qwen 模型** 的偏见以及调整的需求表示担忧，并指向 [Chinese LLM censorship analysis](https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis) 以深入了解该模型潜在的宣传倾向。

- **Layer-Pruning 与 QLoRA 的完美结合**：提到了 **layer-pruning** 与 QLoRA 的交叉应用，一名成员引用了其在提高模型性能（MMLU 分数提高多达 10 分）方面的成功应用，并提供了 [a Hugging Face model card](https://huggingface.co/chargoddard/llama3-42b-v0) 获取实际应用细节。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **单引号拯救系统**：一位用户发现，在 **SystemMessage** 中用**单引号替换反引号**可以解决数据注入问题。
  
- **分块并征服长文本**：讨论了处理来自网页抓取的大型文本数据的策略，包括 token 限制以及如何有效地合并分块响应，并附带了 [LangChain documentation](https://github.com/langchain-ai/langchain/issues/17783) 的链接。

- **PDF 困扰向量数据库**：一位用户发现使用 **PDF 文档**从**向量数据库**检索数据具有挑战性，系统给出了无意义的“我不知道”回答。

- **像专家一样管理事件流**：分享了 **astream_event** 中的事件过滤技术，并指向 [LangChain documentation](https://python.langchain.com/v0.2/docs/how_to/streaming/#filtering-events) 中的特定章节来指导用户完成该过程。

- **发布美食 AI 助手和聊天机器人**：[TVFoodMaps](https://www.tvfoodmaps.com/foodtv-ai-chat) 推出了一项 AI 驱动的功能，帮助用户查找电视节目中出现的餐厅（需要高级会员）；同时分享了使用 **OpenAI & LangChain** 创建 [SQL agents 指南](https://git.new/SQLAgent)并征求反馈。一篇 [Medium 文章](https://medium.com/ai-advances/building-a-conversational-time-machine-a-langgraph-support-chatbot-745b2b08c587)介绍了一个名为 **Conversational Time Machine** 的新概念，探讨了 **LangGraph Support Chatbot** 的开发和用途。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**近似计算的赏金猎人**：为了完成在 `function.py` 中为 LOG2、EXP2 和 SIN 实现泰勒近似（Taylor approximations）的赏金任务，出现了关于在 `ops.py` 中添加位运算的问题，社区担心操作数量膨胀。实用性战胜了纯洁性，因为对新操作的需求与追求极简主义的目标产生了竞争。

**多 GPU 探索继续**：关于 NVLink 多 GPU 支持的澄清让大家了解到 GPU 是通过 PCI-E 连接的，并分享了一个 [GitHub 资源](https://github.com/tinygrad/open-gpu-kernel-modules)，证明了 NVIDIA 具有 P2P 支持的 Linux 开源 GPU 内核模块。

**Diffusion 模型的极高门槛**：一位社区成员将 diffusion 模型从 PyTorch 移植到 tinygrad，引发了关于代码质量的辩论，George Hotz 为项目准入设定了很高的标准。鼓励贡献者提交 PR 以供审查。

**Clip, Clip, 万岁？还是求救？**：针对 TinyGrad 中 `clip_grad_norm_` 的实现进行了深入的技术剖析，Metal 的限制迫使大家讨论将 tensor chunking 作为一种变通方案。这标志着在硬件限制下进行优化的持续斗争。

**权重绑定，Bug 现身**：一个涉及 TinyGrad 中权重绑定（weight tying）的疑似 Bug 被曝光，揭示了两个表面上链接的 tensor 正在被独立优化。社区正在处理此案，建议修正库以实现一致的权重优化。

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Discord 社区存续讨论**：成员们讨论了课程结束后 Discord 服务器的持续活跃度，认为这将取决于成员和版主的参与度，目前尚未列出具体计划。
  
- **专家级 LLM 直播预告**：宣布了一场与来自 **Amazon** 的 Eugene Yan 和来自 **Hex** 的 Bryan Bischof 进行的直播，讨论现实世界中的 **LLM applications**。直播将分享针对 prompt engineering、评估和工作流优化的见解。感兴趣的成员可以在[此处](https://lu.ma/e8huz3s6)注册，并探索他们在 [O'Reilly 报告](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/)中详细记录的学习成果。

- **Finetuning 见解与需求**：关于自定义 **LLM workloads**，讨论包括对欺诈检测等特定角色需要进行 fine-tuning，而语言翻译等通用任务则可能不需要。此外，**Jarvis Lab 即将推出的 Docker 功能**以及 **Modal 的用户体验增强**在 finetuning 方面引起了热议。

- **额度与访问问题成为焦点**：多名成员在 **LangSmith** 和 **OpenAI** 等平台上寻求有关额度和账户访问的帮助，通常会提供 ID 或电子邮件以寻求解决，这表明存在一定程度的困惑或技术问题。

- **技术故障与突破**：在赞扬设计良好的 eval 框架的同时，用户报告了从 **Predibase** 的 CORS 错误到 **OpenAI** 上的额度可见性等各种技术问题，反映了在将 LLM 应用于项目的实际过程中用户体验的复杂性。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **AI 讨论中超越财富的收获**：成员们开玩笑讨论 OpenInterpreter (OI) 是否能让人在财务上更富有，引发了关于实现 100% 富有而非仅 5% 的俏皮话。在另一个话题中，围绕 **Claude 3.5 Sonnet** 的讨论显示，用户更喜欢它的对话风格而非 **GPT-4**。

- **AI 模型角逐最高荣誉**：关于最佳无审查模型的辩论浮出水面，"2.8 dolphin" 和 "mistral 3/31/24" 被提及为竞争者。观点各异，表明用户对每个模型的体验不同，目前尚未出现公认的最佳模型。

- **Open Interpreter 的记忆功能**：关于 OpenInterpreter 潜在长期记忆能力的咨询引发了讨论，但尚未产生结论性的解决方案。成员们正在积极研究如何为 OI 配备持久化内存。

- **OpenInterpreter 暂定的制造里程碑**：#O1 频道的一份更新指出，根据 Ben 的公告，首批 1,000 台 OpenInterpreter 设备预计将在 10 月 31 日至 11 月 30 日之间发货。用户对订单状态和在首批货件中的排位感到好奇。

- **本地任务导向控制器的实用 AI 魔法**：一段演示展示了一个**完全本地的、控制计算机的 AI** 通过读取便签上的密码成功连接到 WiFi，说明了 AI 在执行日常任务中的有效性。该示例反映了 AI 简化日常技术交互的潜力。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **图谱化标注（Graph-Based Captions）实现跨越**：**GBC10M 数据集**（**CC12M** 的图谱化重新标注版本）现已在 [Hugging Face](https://huggingface.co/datasets/graph-based-captions/GBC10M) 上可用。目前正在努力争取更宽松的许可证，并将数据集迁移到 Hugging Face 上的 **Apple organization**，计划在 **arXiv** 上发表配套论文，并在代码完善后发布。

- **对抗鲁棒性辩论升温**：学术界爆发争论，Carlini 和 Papernot 等专家就对抗鲁棒性问题向 Glaze 作者发起挑战，特别是针对扰动预算（perturbation budgets）中未公开的代码库。

- **VAEs 通道数增加引发技术讨论**：将 VAE latent spaces 中的通道数从 4 个增加到 16 个引发了技术辩论，对比了潜空间的复杂性与计算成本，并指出全局注意力随像素数量呈二次方缩放。

- **Claude-3.5 解决了过拟合之谜？**：一位工程师的手动实验表明，**Claude-3.5-Sonnet** 展现出令人期待的能力，能够通过问题进行推理，而不会像其他模型那样在可识别的模式上产生过拟合。

- **Chameleon 模型训练陷入困境**：工程师们在 Chameleon 模型上遇到了意想不到的挑战，极端的梯度范数（gradient norms）导致了 NaN 值，通过降低学习率或切换到更高精度等标准修复方法均无济于事。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **使用 Cohere 构建多语言聊天机器人**：AI 爱好者们正利用 Cohere API 开发各种语言的聊天机器人。讨论中强调了其与 **OpenAI API** 的兼容性，允许通过 RESTful API 或 socket 集成到任何环境中。

- **紫色赞誉**：Cohere 的界面，特别是其对紫色的使用，因其时尚的设计在社区中获得了称赞，为成员们未来的设计工作激发了灵感。

- **项目开发中的问题解决**：一位社区成员分享了他们在处理可能与 API 问题相关的聊天挂起时的经验，并承诺通过 UI 调整和持续的故障排除来解决该问题。

- **社区情谊**：参与者们表现出明显的兴奋，他们欢迎新成员，并分享了对 Cohere 独特且智能的方法的正面印象。

- **平台适应性讨论**：围绕在不同平台上利用 Cohere 能力的对话展开，特别提到了在 Mac 上使用 .NET 创建聊天机器人。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Toucan TTS 打破语言障碍**：开源的 [Toucan TTS](https://x.com/reach_vb/status/1803529768861610073?s=46) 模型以其支持 7000 种语言的 TTS 能力而脱颖而出，其特点是拥有一个用于语言无关发音特征的文本前端，并利用 Meta-learning 处理缺乏数据的语言。
  
- **Claude 3.5 Sonnet 将效率提升至新高度**：全新的 [Claude 3.5 Sonnet](https://x.com/anthropicai/status/1803790676988920098?s=46) 凭借超越竞争对手的表现、更快的速度和更低的成本给社区留下了深刻印象。成员们还庆祝了 Artifacts 功能的发布，它是 Code Interpreter 的继任者，支持实时生成文档、代码和图表。

- **咨询合作创造 AI 协同效应**：市场传闻 Jason Liu 的 Parlance Labs 与 Hamel Husain 及 Jeremy Lewi 的团队合并，联手加强 AI 产品支持与开发，重点关注基础设施、Fine-tuning 和评估，正如他们的[公告](https://x.com/jxnlco/status/1803813743714844863?s=46)中所述。

- **Groq 加强 Whisper 支持，但疑虑尚存**：Groq 新增的 Whisper 模型支持实现了 166 倍实时的处理速度，为更快的 AI 处理打开了大门；然而，社区对其目前的 [Rate limits](https://x.com/sjwhitmore/status/1803811998548812140?s=46) 以及该模型的广泛适用性提出了疑问。
   



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile 旨在实现模型多样性**：在讨论中，有人提议在 Llamafile 结构中利用 **YOLOv10 PyTorch** 和 **OCR Safe Tensors**。提供的一种解决方案是利用 llama.cpp 的 Python 脚本将这些模型转换为 **gguf** 格式。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Infer 会议引发 AI/ML 讨论**：*Hudson Buzby* 和 *Russ Wilcox* 将在 [Infer: Summer '24](https://tinyurl.com/4dfvcte7) 上主持关于**现实生活中的推荐系统**和 AI/ML 挑战的对话，重点关注优化 AI Pipeline 和内容准确性，届时将有来自 Lightricks 等公司的专家参加。

- **在 RecSys Learners 虚拟见面会进行交流与学习**：由 *Rohan Singh S Rajput* 于 2024 年 6 月 29 日举办的 [RecSys Learners Virtual Meetup](https://lu.ma/7pvpp1cm) 为各级专业人士提供了一个连接并增强推荐系统知识的平台。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

**Florence 2 提升了手写 OCR 水平**：[微软的 Florence 2](https://x.com/dylfreed/status/1803502158672761113) 因其在手写识别和 OCR 方面的卓越表现而受到称赞，对新闻业尤其有用。微软的这款模型在处理公共记录方面表现出色。

**在 Hugging Face 上试用 Florence 2**：Florence 2 模型可在 [Hugging Face 上的 Florence-2](https://huggingface.co/spaces/gokaygokay/Florence-2) 进行动手实验，展示了其在视觉相关任务中的一系列能力，这对于 AI 开发和研究至关重要。

**深入了解 Florence 2 的视觉实力**：该模型对各种视觉和视觉语言任务使用基于 Prompt 的方法论，并在包含 54 亿条注释的海量 FLD-5B 数据集上进行训练，展示了在多任务学习方面的精通以及在 Zero-shot 和 Fine-tuned 场景下的适应性。



---

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord

- **"别提 AI，否则会被摔死"：** 一篇有趣的博客文章，名为 *"如果你再提 AI，我就他妈的把你摔死"*，嘲讽了 AI 的炒作周期，警告不要过度狂热且不切实际地采用 AI 技术，并警告说这是 *"为想要准备十二道菜的他妈的灾难的人准备的食谱"*。对行业文化批评感兴趣的工程师可能会发现这是一篇独特但相关的读物，点击[此处](https://ludic.mataroa.blog/blog/i-will-fucking-piledrive-you-if-you-mention-ai-again/)阅读。



---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道沉寂时间过长，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道沉寂时间过长，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道沉寂时间过长，请告知我们，我们将将其移除。


---

# 第二部分：按频道分类的详细摘要和链接


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1253065876685586486)** (437 条消息🔥🔥🔥): 

- **在 Unsloth 中测试 Ollama 支持**：成员们对 Unsloth 新支持 Ollama 感到兴奋。分享了一个 [Colab 链接](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing) 用于早期测试，并征求 Bug 报告。
- **模型参数与训练技术的知识提炼**：讨论深入探讨了分布式数据并行 (DDP) 以及跨 GPU 扩展模型的潜力。重点关注各种语境下的模型准确性，特别是训练期间和训练后的 token 和 context 处理。
- **Claude 3.5 Sonnet 发布**：Timotheeee1 宣布了 Claude 3.5 Sonnet 的发布，这是 Anthropic 推出的一款提高行业标准的新模型。详细信息和访问指南可在 [Anthropic 的新闻稿](https://www.anthropic.com/news/claude-3-5-sonnet) 中找到。
- **Unsloth 训练技巧与故障排除**：成员们就微调 (fine-tuning) 和部署过程中的模型合并及特殊 token 处理寻求建议，并分享了关于结束 token 信号和准确性评估的有价值见解。讨论还包括了使用 Hugging Face 等平台的技术。
- **即将举行的活动与演讲录音**：Theyruinedelise 预告了与 Daniel 和 Sebastien 进行的一场精彩演讲，重点是模型在 Ollama 上的微调 (fine-tuning) 和部署。详细信息通过 [Discord 邀请链接](https://discord.com/invite/EwGjYYBu?event=1251334371349233814) 分享。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://x.com/danielhanchen/status/1803446594228068474">来自 Daniel Han (@danielhanchen) 的推文</a>: 我要去旧金山待几个月！我将参加 @aiDotEngineer World's Fair，进行一场 3 小时的研讨会和演讲！研讨会将非常具有技术性且有趣！主题：1) Backpro...</li><li><a href="https://www.anthropic.com/news/claude-3-5-sonnet">介绍 Claude 3.5 Sonnet</a>: 介绍 Claude 3.5 Sonnet——我们迄今为止最智能的模型。Sonnet 现在在关键评估中超越了竞争对手模型和 Claude 3 Opus，且速度提高了一倍。</li><li><a href="https://huggingface.co/fimbulvntr/llewd-8b-64k/tree/main">fimbulvntr/llewd-8b-64k 在 main 分支</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/trl/main/en/dpo_trainer">DPO Trainer</a>: 未找到描述</li><li><a href="https://datta0.substack.com/p/ai-unplugged-13-qwen2-discopop-mixture">AI Unplugged 13: Qwen2, DiscoPOP, Mixture of Agents, YOLO v10, Grokked Transformers</a>: 洞察胜过信息</li><li><a href="https://github.com/Kryptonions/RLHF">GitHub - Kryptonions/RLHF: 高效训练 LLM 的流水线</a>: 高效训练 LLM 的流水线。通过在 GitHub 上创建账号为 Kryptonions/RLHF 的开发做出贡献。</li><li><a href="https://x.com/UnslothAI/status/1803767513215610974">来自 Unsloth AI (@UnslothAI) 的推文</a>: 我们今天美国东部时间中午 12 点将在 @Ollama 的服务器上直播，展示我们对 Ollama 的新支持！🦥🦙 首先与 Sebastien 一起学习“AI 中的情感”，然后我们将教学并提供早期访问...</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq1">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7412#issuecomment-2120427347">CUDA: 由 JohannesGaessler 提供的量化 KV cache 演示 · Pull Request #7412 · ggerganov/llama.cpp</a>: 此 PR 添加了一个仅用于研究目的的量化 KV cache 的简单实现。目标不是提供一个可以合并或适合常规使用的实现，而是...</li><li><a href="https://huggingface.co/Salesforce/SFR-Embedding-2_R">Salesforce/SFR-Embedding-2_R · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct">Alibaba-NLP/gte-Qwen2-7B-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-bnb-4bit/blob/main/tokenizer_config.json">tokenizer_config.json · unsloth/llama-3-8b-bnb-4bit 在 main 分支</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1253145504825479168)** (7 条消息): 

- **位置查询已回答**：一位用户询问了某些元素的定位问题，另一位用户确认并回复了“*yes!*”及红色打勾表情符号。
- **Shensmobile 寻求 causalLM Loss 计算的澄清**：一名成员询问：“*有人知道 causalLM 在训练期间是如何计算 Loss 的吗？*”他们对如何解读所看到的数字表示困惑，并将其与传统的 Masked LM 下游任务中更清晰的 Loss 计算进行了对比。
- **关于 causalLM 中 Loss 聚合的澄清**：在继续讨论中，一名成员反思了 causalLM 下一个词预测（next word prediction）的本质。他们提出疑问：“*‘Loss’ 是否只是每个预测单词的所有 Loss 的聚合？*”
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1253106974493507634)** (62 条消息 🔥🔥): 

- **部署 llama3 时的兼容性问题**：由于库版本兼容性和 CUDA 依赖项，一名成员在部署 llama3 时遇到了错误。Thefanciestpeanut 建议使用 Conda 来简化依赖管理并避免麻烦。
- **QLora 微调与 Adapter 问题**：Ritx8 询问了 SFTTtrainer 工作流中 `model.save_pretrained()` 和 `trainer.save_model()` 的区别。他们最终发现这两种方法本质上达到了相同的目的，相关讨论见[此处](https://discuss.huggingface.co/t/what-is-the-purpose-of-save-pretrained/9167/2?u=aflah)。
- **转换数据集格式**：Jadon1 在为 LLM 准备数据时遇到问题，通过重新上传并重构数据集以匹配所需格式（而不是在数据集查看器中重命名列）解决了该问题。
- **为单个输入创建多个响应**：karatsubabutslower 建议 Hieu1852002 使用带有 beam search 参数的 `.generate`，以便使用 "unsloth/llama-3-8b-bnb-4bit" 等模型生成多个序列或响应。
- **在 Instruct 模型上进行持续预训练（Continued pretraining）**：Gbourdin 询问了在 Instruct 模型上进行持续预训练的可行性。Shensmobile 回复称这是可能的，虽然可以遵循 Instruct 格式，但在针对特定任务进行微调之前，先对 Base 模型进行领域自适应（domain adapt）可能会更有效。

**提到的链接**：<a href="https://discuss.huggingface.co/t/what-is-the-purpose-of-save-pretrained/9167/2?u=aflah">save_pretrained() 的目的是什么？</a>：你好！就你提问的方式而言，这个问题有点奇怪：“既然 Trainer 有那个模型，为什么模型还有这个方法？”。基本的回答是：“因为它们是两个不同的对象...”

  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1253064705035997285)** (417 条消息 🔥🔥🔥): 

- **GPT-4o 的推理能力受到审视**：成员们辩论了 GPT-4o 的推理能力，指出尽管它有局限性，但通常仍优于其他非 OpenAI 模型。一位用户提到：“*在将所有这些应用到更大的模型之后肯定会更好，*”表达了对 GPT-5 潜力的好奇。
  
- **关于人工超级智能（ASI）的辩论**：关于 ASI 未来的广泛讨论随之展开，争论点在于它是否有潜力实现“*上帝般的智能*”和永生。讨论强调了对 ASI 理论和实践极限的担忧，同时也对技术的快速进步表示乐观。

- **Claude 3.5 和 Gemini 的讨论**：一些用户对 OpenAI 最新的产品（如 GPT-4o）表示怀疑，转而青睐 Claude 3.5 和 Google 的 Gemini，因为它们在大上下文窗口（large context windows）和幽默感融合方面具有先进能力。“*OpenAI 现在正被反超（outcooked），*”一位用户评论道，强调了竞争对手的进步。

- **人类与 AI 能力的辩论**：聊天的一部分探讨了人类智能与 AI 相比独特的优势和劣势，指出虽然 AI 可以处理更多数据，但现实世界的应用往往限制了其效能。这引发了围绕计算、逻辑以及 AI 所能达到的边界的理论辩论，总结为：“*很大一部分在智力方面极具天赋的人并不是工程师。*”

- **从 GPT 转向 Google AI Studio**：用户分享了使用 Google AI Studio 的经验和技巧，特别是对其 Gemini 模型的称赞。“*大量的文本……而它表现得就像该领域的专家，*”一位成员分享道，说明了 Gemini 的大上下文窗口在特定用例中如何优于 GPT 等其他模型。

**提到的链接**：<a href="https://ai.google.dev/aistudio">未找到标题</a>：未找到描述

  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1253069870862438481)** (17 messages🔥): 

- **用户要求 Sam Altman 发布新语音更新**：一位用户对延迟更新表示沮丧，敦促 Sam Altman 发布新语音。他们评论说这种延迟导致收益递减，并认为这是一种通过扣留发布来增加兴奋感的策略。

- **ChatGPT 无法可靠地生成长输出**：一名成员建议，由于 Token 限制，要求 ChatGPT 输出 5000 字的内容是不切实际的。另一名成员分享了他们的经验，即在要求以视频脚本格式输出时，字数会减少。

- **在网站上创建的 GPT 使用 GPT-4o**：当被问及在网站上创建的 GPT 模型是基于 GPT-4 还是 GPT-4o 时，一名成员确认它们使用的是 **GPT-4o**。

- **Surface 笔记本电脑作为 MacBook 的替代品**：一位用户询问适合 Web 开发的笔记本电脑，以作为 MacBook 的替代方案。受访者建议使用 **最新的 Surface 笔记本电脑**，但同时也建议考虑 MacBook 或访问像 BuildAPC 这样的服务器以获取更多建议。
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1253094360543924274)** (11 messages🔥): 

- **高 Token 使用量令用户沮丧**：一位用户抱怨说，由于必要的函数调用，即使是一个简单的 "hello" 命令也消耗了 384 个 Token。建议的解决方案包括缩短函数名称和描述，或排除不必要的工具以节省 Token 使用量。
  
- **DALL-E 在处理不对称性方面表现不佳**：一名成员询问如何让 DALL-E 生成较少对称性的图像，因为使用 "asymmetrical"（不对称）等术语的尝试失败了。另一位用户建议使用强调随机性的短语，如 "rule of thirds"（三分法）或 "complementary positions"（互补位置），但承认效果有限。

- **系统指令并不总是被遵守**：一位用户发布了截图，显示 GPT-3.5-turbo-0125 有时会忽略系统指令。另一名成员建议专注于清晰的指令，并建议移除不必要的容器以解决相关问题。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1253094360543924274)** (11 messages🔥): 

- **减少 OpenAI Assistants 中的 Token 使用量**：一名成员对 "hello" 等简单输入的 Token 高使用量表示担忧。另一名成员建议缩短函数的名称和描述，或排除不必要的工具，这有助于节省 Token。
  
- **寻求 DALL-E 生成不对称艺术**：一位对 DALL-E 对称输出感到沮丧的用户询问如何生成更独特、不对称的艺术作品。另一位用户承认了这一局限性，并建议使用强调随机性或不平衡的短语，尽管效果有限。

- **关于 OpenAI Memory 功能的问题**：成员们讨论了聊天中的新窗口是否完全是全新的，并检查了 Memory 功能。一名成员澄清说，在引入 Memory 功能后，这种行为已经发生了变化。

- **GPT-3.5-turbo-0125 忽视系统指令**：一名成员分享了截图，强调了模型间歇性忽略某些系统指令的问题。另一名成员建议问题可能出在不必要的容器指令上，并建议将其移除。
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1253061950594613349)** (401 条消息🔥🔥): 

- **Stability AI 的 CEO 已确定**：成员们讨论了 Stability AI 的 CEO，确认为 **Shan Shan Wong**。一位用户提到他们不会分享内部信息，但暗示偶尔会发布一些消息。

- **关于 AI 生成图像的许可咨询**：一名成员询问是否可以将由 **stabilityai/stable-diffusion-xl-base-1.0** 生成的图像以 Creative Commons 许可（如 **CC-0-1.0、CC-BY-4.0 或 CC-BY-SA-4.0**）进行授权。stable-diffusion-xl-base-1.0 模型本身是根据 CreativeML Open RAIL++-M License 授权的。

- **关于归档和社区频道的辩论**：成员们对某些社区频道被删除或归档表示沮丧，特别是针对 **Cascade** 和其他艺术社区。“艺术社区被移除是因为它们已经近 2 个月没有活跃，并且充斥着机器人垃圾信息，”一位管理员（mod）回应道，并补充说如果需要，这些频道可以重新同步回来。

- **关于 Turbo 和微调（Finetuned）模型的看法**：讨论强调了对“Turbo”模型与 **Juggernaut** 和 **Pony** 等微调模型之间的不同看法。一些成员更喜欢 Turbo 模型的高灵活性和速度，而另一些人则认为**微调模型能更好地服务于特定目的**，特别是在处理细节或特定概念的任务时。

- **Mobius 模型解析**：一位用户介绍了 **Mobius 模型**，称其为去偏扩散模型（debiased diffusion models）领域的新标杆（state-of-the-art），并解释了其**领域无关的去偏（domain-agnostic debiasing）**技术。针对其庞大的体积和特殊要求（如 **clip skip 3**）的疑问，通过说明其训练细节以及对 **Lora 兼容性**的潜在影响得到了解答。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/hatsune-miku-miku-hatsune-earthquake-plush-miku-death-gif-4018907532159793300">Hatsune Miku Miku Hatsune GIF - Hatsune miku Miku hatsune Earthquake - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/tasting-milk-antony-starr-the-homelander-the-boys-lick-gif-17834498">Tasting Milk Antony Starr GIF - Tasting Milk Antony Starr The Homelander - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://civitai.com/models/490622/mobius">Mobius - v1.0 | Stable Diffusion Checkpoint | Civitai</a>：Mobius：重新定义去偏扩散模型的最前沿。Mobius 是一款突破了领域无关去偏界限的扩散模型...</li><li><a href="https://github.com/comfyanonymous/ComfyUI_TensorRT">GitHub - comfyanonymous/ComfyUI_TensorRT</a>：通过在 GitHub 上创建账号来为 comfyanonymous/ComfyUI_TensorRT 的开发做出贡献。</li><li><a href="https://civitai.com/models/526316">Tsuki - v2 | Stable Diffusion Checkpoint | Civitai</a>：euler a, 30+ steps, 使用外部 vae flatcoloredponytest2+Bunny-XL2-V3-NS (0,0,0.0202546296296296,0.0787037037037037,0.171875,0.296296296296296,0.4...</li><li><a href="https://colab.research.google.com/github/mkshing/notebooks/blob/main/stable_video_diffusion_img2vid.ipynb#scrollTo=9AZDrh-SUDt2">Google Colab</a>：未找到描述</li><li><a href="https://civitai.com/articles/5800">Malicious Compliance | Civitai</a>：针对 CivitAI 选择大幅减少用户获取 buzz 的能力，以及对 NSFW 用户的特定攻击，我提议尽可能多的...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1253075566790185050)** (319 条消息🔥🔥): 

- **Perplexity CEO 做客 Lex Fridman 播客**：分享了一段名为 ["Aravind Srinivas: Perplexity CEO on Future of AI, Search &amp; the Internet | Lex Fridman Podcast #434"](https://youtu.be/e-gwvmhyU7A) 的 YouTube 视频。亮点包括 Larry Page 的哲学——“用户永远不会错”。

- **Pro Search 问题报告**：用户讨论了在开启 Pro 开关时 Pro Search 无法找到网站来源的问题，尽管在 iPhone 应用上运行正常。多名用户报告了相同问题，该问题已上报。

- **Perplexity vs. ChatGPT 及功能请求**：多位用户询问 Perplexity 与 ChatGPT 的区别，解释重点在于 Perplexity 对可靠网络资源的使用和详细的回答。此外，还有关于需要更多设置控制（如 Temperature 设置）以及为高级用户提供“高级（Advanced）”UI 模式的多次请求和讨论。

- **对 Claude 3.5 Sonnet 发布的期待**：用户期待在 Perplexity 上发布 Claude 3.5 Sonnet，一些人强调了它在创意写作方面的优势。人们对其性能提升感到兴奋，同时也关注它将如何集成到平台中。

- **关于遵守 robots.txt 的争议**：分享并讨论了一篇 Wired 文章，该文章批评了 Perplexity 对 robots.txt 的遵守情况。用户发表了看法，指出 AI 抓取信息用于训练与代表用户检索信息之间的区别，并质疑是否应适用相同的规则。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.wired.com/story/perplexity-is-a-bullshit-machine/">Perplexity 是一个废话机器</a>：WIRED 的一项调查显示，被 Forbes 指控窃取内容的 AI 搜索初创公司正在秘密抓取数据，并凭空捏造事实。</li><li><a href="https://gizmodo.com/perplexity-ai-internet-rule-robots-exclusion-protocol-1851551095">据报道 Perplexity 正允许其 AI 违反互联网的一项基本规则</a>：Perplexity 因其 AI 生成的文章而陷入困境。</li><li><a href="https://youtu.be/e-gwvmhyU7A">Aravind Srinivas：Perplexity CEO 谈 AI、搜索与互联网的未来 | Lex Fridman Podcast #434</a>：Arvind Srinivas 是 Perplexity 的 CEO，该公司旨在彻底改变人类在互联网上寻找问题答案的方式。请支持本播客...</li><li><a href="https://docs.perplexity.ai/docs/feature-roadmap">功能路线图</a>：未找到描述</li><li><a href="https://fxtwitter.com/perplexity_ai/status/1803861295432933801?s=19">来自 Perplexity (@perplexity_ai) 的推文</a>：该模型在我们的内部基准测试中优于 Claude 3 Opus 和 GPT-4o。</li><li><a href="https://x.com/AravSrinivas/status/1803870324213121362">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：Claude 3.5 现已在 Perplexity Pro 上可用。在我们的国际评估中，它优于 GPT-4o。快来试试吧！引用 Perplexity (@perplexity_ai) 🚨 Claude 3.5 Sonnet 现已在 P...</li><li><a href="https://msty.app/">Msty - 让 AI 模型的使用变得简单容易</a>：只需点击一下按钮即可与任何 AI 模型交互</li><li><a href="https://rknight.me/blog/perplexity-ai-is-lying-about-its-user-agent/">Perplexity AI 在其 User Agent 上撒谎</a>：Perplexity AI 声称它发送了 User Agent 并遵守 robots.txt，但事实并非如此</li><li><a href="https://www.perplexity.ai/search/read-the-time-A0JXqn3iR86OjAnRVX4CEQ">为我读时间
大声思考并写下你的内部想法，检查...</a>：当然，我会一步步思考：1. 首先，我正在看时针。它显然超过了 2，但还没到 3。2. 现在，分针....</li><li><a href="https://www.perplexity.ai/page/Bananaclicking-game-tops-zW.nvAhGSzuXznHHEskL1Q">香蕉点击游戏登顶 Steam</a>：Banana 是一款简单的点击游戏，玩家重复点击虚拟香蕉，该游戏席卷了 Steam 游戏平台，积累了惊人的 884,469...</li><li><a href="https://www.reddit.com/r/ClaudeAI/s/w5t5F1MtWc">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1253154378785886360)** (6 条消息): 

- **前三大自我提升平台**：用户讨论了 Mindvalley 等前三大自我提升平台，重点介绍了 Jim Kwik 的 "Unlimited Abundance" 和 "Superbrain" 等课程，涵盖了正念、健康和个人成长等主题。[了解更多](https://www.perplexity.ai/search/What-are-the-3aHdlkCeTLunnzd4eTLozA)。

- **迷幻剂与上帝邂逅**：一项调查研究显示，由迷幻剂诱发的“上帝邂逅”体验与自发产生的体验之间存在惊人的相似性，并指出这对生活满意度和意义有显著影响。相当一部分无神论者在经历此类体验后改变了他们的信仰。[阅读更多](https://www.perplexity.ai/search/What-is-the-cLSebvMCTH2CL_SgIVAQnQ)。

- **今日探索 Perplexity AI YouTube**：Perplexity AI 的 [YouTube 视频](https://www.youtube.com/embed/AXxR1aMNBls) 涵盖了各种主题，包括 AI 安全、TikTok 的创意工具以及最近的天文学发现。

- **英语文学专业的高薪职位**：对于拥有英语文学硕士学位的人来说，技术作家和编辑属于高薪职位，平均年薪分别为 74,000 美元和 63,000 美元。[探索更多职业选择](https://www.perplexity.ai/search/What-jobs-can-zdbsXD8QQp2OGw79WcpQqw#5)。

- **Lululemon 财报预测**：用户讨论了 Lululemon Athletica Inc. 即将发布的财务业绩，分析师预测每股收益约为 2.39 至 2.40 美元。推测认为这些结果可能会显著影响股价。[点击此处查看更多详情](https://www.perplexity.ai/search/LULU-is-set-eyOD1n8mQ..kH_kgodWYFg)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.perplexity.ai/search/LULU-is-set-eyOD1n8mQ..kH_kgodWYFg">LULU 计划于今日发布财报。告诉我更多关于传闻的信息...</a>：Lululemon Athletica Inc. (NASDAQ: LULU) 计划在 2024 年 6 月 5 日今日收盘后发布第一季度财务业绩。目前有...</li><li><a href="https://www.perplexity.ai/search/What-are-the-3aHdlkCeTLunnzd4eTLozA">当今前三大自我提升平台是什么？</a>：当今前三大自我提升平台是：1. Coursera。Coursera 是一个在线学习平台，与顶尖大学和组织合作提供...</li><li><a href="https://www.perplexity.ai/page/Piloting-an-AI-JDrMoyrVT6iNR_lul8DnQA">驾驶 AI 系统</a>：通过聊天与 AI 互动是否更像是驾驶一个系统，而不是进行真正的对话？这个发人深省的比喻挑战了我们通常的...</li><li><a href="https://www.perplexity.ai/search/What-jobs-can-zdbsXD8QQp2OGw79WcpQqw#5">英语文学硕士可以从事什么工作</a>：以下是英语文学硕士学位获得者的一些潜在职业路径：教学、教授英语文学的大学教授、...</li><li><a href="https://www.perplexity.ai/search/What-is-the-cLSebvMCTH2CL_SgIVAQnQ">迷幻剂与上帝之间的关系是什么</a>：根据提供的搜索结果，迷幻体验与关于上帝的邂逅或感知之间似乎存在复杂的关系，...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1253210297334894602)** (5 条消息): 

- **Perplexity API 自定义限制**：一位成员分享了他们的印象，认为 Perplexity API 托管单个 HTTPS 端点，响应中没有引用，并且为构建自定义 Agent 提供的集成选项有限。提到的一个积极方面是，当用户自行处理 Tokenization 或文本 Embedding 时，它能够运行大型开源 LLM。
- **无法通过 API 访问 Pages 功能**：一位用户询问是否可以通过 API 访问 Pages 功能，另一位用户回答说这是**不可能的**。
- **轻松重置您的 API Key**：要重置 API Key，用户可以访问 [Perplexity API 设置页面](https://www.perplexity.ai/settings/api)。该部分包含“删除”和“生成”新 Key 的选项。

**提到的链接**：<a href="https://www.perplexity.ai/settings/api">Perplexity</a>：Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的答案。

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1253404964664119306)** (2 条消息): 

- **Character.AI 实现高效 Int8 训练**：一位成员分享了一个[链接](https://research.character.ai/optimizing-inference/)，讨论了 Character.AI 通过优化推理使其更高效、更具成本效益且更具可扩展性，从而迈向 AGI 的努力。*“我们每秒处理超过 20,000 次推理查询，大约是 Google Search 处理请求量的 20%。”*
- **对 AQT 使用情况的好奇**：另一位成员询问 Character.AI 是否在其高效推理过程中使用了 Adaptive Quantization (AQT)。

**提到的链接**：<a href="https://research.character.ai/optimizing-inference/">Optimizing AI Inference at Character.AI</a>：在 Character.AI，我们正致力于构建 AGI。在未来的状态下，大语言模型 (LLMs) 将增强日常生活，提供业务生产力和娱乐，并帮助人们进行 e...

  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1253064107762909285)** (8 条消息🔥): 

- **使用 Nsight 进行 Kernel 分析**：成员们建议使用 Nsight Compute 对 CUDA kernels 进行性能分析，以识别瓶颈并优化性能。他们分享了一个[用于分析的 GitHub 脚本](https://github.com/AnswerDotAI/bitlora/blob/master/benchmarks/forward_kernel/profile_forward_kernel.sh)和一个 [YouTube 播放列表](https://www.youtube.com/playlist?list=PL5B692fm6--ukF8S7ul5NmceZhXLRv_lR)作为资源。
- **理解 'ncu' 命令**：当被问及分析脚本中的 'ncu' 时，有人解释说 'ncu' 代表 Nsight Compute。并提供了一份[用户指南](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)以给出关于使用该 CLI 工具的更多细节。
- **在 Nsight Compute 中使用 Triton**：已确认编译为 **PTX** 的 **Triton** 与 **Nsight Compute** 兼容。这允许以类似于 CUDA 的方式进行性能分析。
- **升级到 Triton 3.0.0**：一位成员建议升级到 Triton 3.0.0 以解决问题，并提供了详细的[安装指南](https://www.umerha.com/smarties/2024-06-13-installing-triton-3-0/)。该指南包括卸载当前版本、克隆新仓库以及针对开发或常规使用进行设置的步骤。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.umerha.com/smarties/2024-06-13-installing-triton-3-0/">Installing Triton 3.0.0</a>：截至 2024 年 6 月 13 日，要获取 Triton 3.0，你必须从源码安装，如下所示：</li><li><a href="https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html">4. Nsight Compute CLI &mdash; NsightCompute 12.5 documentation</a>：未找到描述</li><li><a href="https://github.com/AnswerDotAI/bitlora/blob/master/benchmarks/forward_kernel/profile_forward_kernel.sh">bitlora/benchmarks/forward_kernel/profile_forward_kernel.sh at master · AnswerDotAI/bitlora</a>：实验性 q[X]ora kernel 开发代码。通过在 GitHub 上创建账号为 AnswerDotAI/bitlora 开发做出贡献。</li><li><a href="https://www.youtube.com/playlist?list=PL5B692fm6--ukF8S7ul5NmceZhXLRv_lR">CUDA Developer Tools</a>：本视频系列将帮助你开始使用适用于 CUDA 的 NVIDIA Nsight 开发者工具。提高你对工具的熟练程度，并将示例应用到你的...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1253361205662187630)** (1 条消息): 

- **AI Unplugged 第 13 期讨论最新 AI 趋势**：第 13 期 AI Unplugged 涵盖了最近的进展，如 **Qwen2**、**DiscoPOP**、**Mixture of Agents**、**Grokked Transformers** 和 **YOLO v10**。更多细节可以在[博客文章](https://datta0.substack.com/p/ai-unplugged-13-qwen2-discopop-mixture)中找到。

- **Mixture-of-Agents 增强 LLM 能力**：来自 [Together.ai](http://together.ai) 的一篇新论文讨论了同时使用多个 LLMs 如何增强性能，详见其 [arxiv 论文](https://arxiv.org/pdf/2406.04692)和相关的[博客文章](https://www.together.ai/blog/together-moa)。

- **探索 ChatGPT 的自定义 GPT 'Advisory Board'**：有一个名为 [Advisory board](https://chatgpt.com/g/g-mhH7nIrJW-advisory-board-2-0-with-hats) 的自定义 GPT，它允许 GPT 扮演不同的角色，从多个角度评估响应，这是混合 LLMs 以提供复杂答案评估的一个实际例子。

**提到的链接**：<a href="https://datta0.substack.com/p/ai-unplugged-13-qwen2-discopop-mixture">AI Unplugged 13: Qwen2, DiscoPOP, Mixture of Agents, YOLO v10, Grokked Transformers</a>：洞察胜过信息

  

---

### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1253094885901467768)** (4 条消息): 

- **Nous Research 招聘高级 ML 工程师**：*Nous Research 正在招聘 CUDA/Triton 工程师，负责在 PyTorch 中实现建模代码，并使用 Triton 和 CUDA 优化性能。* 他们正在寻找能够编写自定义 Triton Kernels 以提高训练效率的候选人，并开放全职和合同工职位。[Nous Research](https://nousresearch.com/) [Twitter](https://twitter.com/nousresearch/) [LinkedIn](https://www.linkedin.com/company/nousresearch/)
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1253063820629246023)** (4 条消息): 

- **原生 Files 应用提升生产力**：一位成员分享了他们在 Mac 和 iPhone 上使用原生 **Files 应用**的热情，*“太棒了，因为我可以立即在 Mac 和 iPhone 的原生文件浏览器中打开它们（并在命令行中进行操作）”*。这种设置可以实现无缝的 AirPlay，将内容镜像到 Mac，并将其作为 Streamlabs 中的视频源。

- **Reflector 4 提供替代镜像方案**：另一位成员建议将 **Reflector 4** 作为镜像内容的解决方案，它允许在 AirPlay 的同时使用 Mac，但指出，*“你可能需要比 M1 Pro 配置更高的 Mac，它在串流时可能会导致我的电脑崩溃”*。
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1253415962322014288)** (3 条消息): 

- **CUDA MODE 中的量化方法**：讨论强调**支持的量化方法**非常具体，指出像 `int4_weight_only` 量化这类术语是为 **tinygemm** 等系统量身定制的。例如，*“它是与 tinygemm 兼容的 int4 仅权重（weight-only）量化，即 `uint4 weight-only asymmetric per-group quantization`”* ([代码库链接](https://github.com/pytorch/ao/blob/e6460c22284df669b95dc912114cc29b40df2030/torchao/quantization/quant_primitives.py#L280-L289))。

- **FP8 量化挑战**：有人对 **FP8** 量化表示担忧，强调需要一种流程来选择最佳 scale。建议用户应该可以选择在性能损失不可接受时跳过对某一层的量化，称 *“如果损失太大，人们应该能够选择不量化某一层”*。

- **为 XLA 导出量化模型**：提出了创建一个包含 quant/dequant 操作的**量化模型**，以便在 **XLA** 中导出和执行的想法。需要更多细节来探索如何实现这一点，但其目标是更好地与 XLA 图执行（graph execution）集成。
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1253371751820431482)** (3 条消息): 

- **寻求加速扩散模型的技巧**：一位成员询问了在不改变架构或重新训练的情况下加速扩散模型（diffusion models）的常用技巧，并提到他们已经通过使用 **torch compile max autotune** 实现了 **20% 的性能提升**。在特定频道中已自动为此讨论创建了线程。
  

---

### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1253177566055891054)** (4 条消息): 

- **HQQ 中的 Zero 导致兼容性问题**：一位用户指出 HQQ 将 float 张量量化为整数 `qweight` 以及 float 类型的 `zero` 和 `scale`，而像 RTN 和 GPTQ 这样的算法使用的是整数 `zero`。他们询问 HQQ 的量化结果是否可以复用 GPTQ 和 RTN 的 `w4a16` kernel。

- **四舍五入的 zero-point 和最快 kernel**：另一位成员解释说，为了使用 `int8`，4-bit 量化的 zero-point 进行了四舍五入，并提到使用了 [torchao kernel](https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py#L41)，这是目前分组量化（grouped quantization）中最快的 kernel。他们还提到在 channel-wise 量化中支持 Marlin，但强调从兼容性和性能角度来看，torchao 是首选。

- **Marlin 的性能提升细节**：一位用户询问了使用 Marlin 的 split float `matmul` 方法带来的性能提升。另一位成员回复了[性能数据链接](https://github.com/mobiusml/hqq/blob/master/imgs/llama_int4_4090.png?raw=true)，并建议使用 `torchao` 方案以获得更好的兼容性和速度。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py#L41">hqq/examples/backends/torchao_int4_demo.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/backends/marlin_int4_demo.py">hqq/examples/backends/marlin_int4_demo.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1253063543951982602)** (230 条消息🔥🔥): 

```html
<ul>
  <li>
    <strong>LR Schedulers PR 已就绪</strong>：GitHub 上用于添加学习率调度器（learning rate schedulers）的 <a href="https://github.com/karpathy/llm.c/pull/605">PR #605</a> 已完成重构和简化。“主文件现在更短了。”
  </li>
  <li>
    <strong>关于 SLURM 问题的讨论</strong>：几位用户讨论了 SLURM 配置和脚本的故障排除，特别是 dummy GPU 运行挂起或仅征调单个节点的问题。一位用户提到：“我昨晚重启了节点。”
  </li>
  <li>
    <strong>更新后的 Bias_Backward PR 反馈</strong>：更新后的 <a href="https://github.com/karpathy/llm.c/pull/619">bias_backward PR</a> 显示出轻微的加速，但引发了对潜在死锁的担忧。“我们正在确保不在条件分支内调用 `__syncthreads`。”
  </li>
  <li>
    <strong>H100 机型实验</strong>：一位用户分享了在 H100 机器上运行 1558M 模型的经验，与 A100 相比实现了 2.5 倍的原生加速。“预计从 8.1 天缩短至 3.2 天。”
  </li>
  <li>
    <strong>评估最新 Eval Harness</strong>：成员们讨论了不同版本 eval harness 的速度，显示出显著的性能提升。“从 40 分钟缩短到 1 分钟……建议使用最新的 eval harness 版本进行更快速的对比评估。”
  </li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/mdouglas/llmc-gpt2-774M-150B">mdouglas/llmc-gpt2-774M-150B · Hugging Face</a>：未找到描述</li><li><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html#mpi-progress">NCCL and MPI &mdash; NCCL 2.21.5 documentation</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/619">Cast Get2dNoiseUint computation to uint by gordicaleksa · Pull Request #619 · karpathy/llm.c</a>：在与 Squirrel（我们使用的 noise rnd 生成器的作者）交流后，将中间计算转换为 uint 可能是一个好主意，以避免处理具有 UB（未定义行为）的 int...</li><li><a href="https://github.com/karpathy/llm.c/pull/624">if available, use MPI env vars to initialize multi-gpu configs by ngc92 · Pull Request #624 · karpathy/llm.c</a>：看看 windows 对此怎么看</li><li><a href="https://github.com/karpathy/llm.c/pull/623">feature/nccl only (delete MPI) by karpathy · Pull Request #623 · karpathy/llm.c</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/605">Add learning rate schedulers by gordicaleksa · Pull Request #605 · karpathy/llm.c</a>：重构了学习率调度器代码 - 根据我们的线下讨论，我们将把所有定义保留在 "schedulers.h" 中。支持的 LR 调度器：带有 Warmup 的 Cosine、循环三角形...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1253185718193360956)** (22 messages🔥): 

- **FP6-LLM 中的新 FPx Kernel**：来自 FP6-LLM 的新 FPx Kernel 可以执行 FP16 与 FPx（x 范围从 1 到 7）的操作，尽管 FP1 和 FP2 的实用性较低。局限性包括缺乏对 group-wise quantization 的支持，正如文中所述：*"如果有人有时间和知识，或许可以尝试为其添加支持 😂。"*

- **FP6 基准测试显示出不同的结果**：针对 wikitext 的不同量化方法的基准测试显示出不同的 perplexity 和 token 速度。值得注意的是，与 FP5 或 FP6 等其他方法相比，FP4 和 FP3 量化会导致显著的 perplexity 失真。

- **uint2 端到端测试用例开发**：目前正在开发 uint2 量化的端到端测试用例，同时还在进行持续的基准测试任务。

- **FP2 和 FP4 相比 FP16 的加速**：使用 FPx-LLM Kernel，FP2 和 FP4 乘法相比 FP16 乘法显示出 3.77 倍到 10.64 倍的加速，具体取决于矩阵大小。

- **FP16 到 FP8 转换的挑战**：讨论强调了实现 fp16 -> fp8 转换的挑战，指出尽管硬件指令可用，但仍可能出现减速。一位成员指出 *"fp16->fp8 实际上可以通过切掉 8 位 mantissa 来完成。"* 但同时也承认了缩放需求带来的复杂性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/gau-nernst/ao/blob/fp5_llm/torchao/prototype/fp6_llm/fp6_llm.py">ao/torchao/prototype/fp6_llm/fp6_llm.py at fp5_llm · gau-nernst/ao</a>：用于量化和稀疏化的原生 PyTorch 库 - gau-nernst/ao</li><li><a href="https://github.com/gau-nernst/ao/blob/fp5_llm/torchao/csrc/cuda/fp6_llm/utils_parallel_dequant.cuh">ao/torchao/csrc/cuda/fp6_llm/utils_parallel_dequant.cuh at fp5_llm · gau-nernst/ao</a>：用于量化和稀疏化的原生 PyTorch 库 - gau-nernst/ao</li><li><a href="https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/fp6_linear.cu#L177">fp6_llm/fp6_llm/csrc/fp6_linear.cu at 5df6737cca32f604e957e3f63f03ccc2e4d1df0d · usyd-fsalab/fp6_llm</a>：针对 x-bit 量化（如 FP6, FP5）的高效 GPU LLM 推理支持。 - usyd-fsalab/fp6_llm</li><li><a href="https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/include/utils_parallel_dequant.cuh">fp6_llm/fp6_llm/csrc/include/utils_parallel_dequant.cuh at 5df6737cca32f604e957e3f63f03ccc2e4d1df0d · usyd-fsalab/fp6_llm</a>：针对 x-bit 量化（如 FP6, FP5）的高效 GPU LLM 推理支持。 - usyd-fsalab/fp6_llm
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=Tumxml3DCvM
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1253448619177279580)** (1 messages): 

- **Hermes 2 Theta 70B 备受瞩目**：公告介绍了 **Hermes 2 Theta 70B**，与前代产品相比，它被描述为“更聪明、更具创意且功能更强大”。Hermes 2 Theta 在 **MT-Bench 上获得了 9.04 分**，超过了 GPT-4-0314 的 8.94 分，并在多个基准测试中优于 Llama-3 Instruct 70B。
- **强调了高级特性和能力**：**Hermes 2 Theta** 支持 function calling、特征提取以及用于 Agent 能力的 JSON mode 输出，为用户提供了高级功能。这突显了其在更复杂的 AI 驱动任务和操作中的潜在应用。
- **与 Arcee AI 的合作**：此版本是 Nous Research 与 **Charles Goddard** 以及 MergeKit 背后的团队 **Arcee AI** 持续合作的成果。该模型是结合了 **Hermes 2 Pro** 和 **Meta 的 Llama-3 Instruct** 的合并并进一步经过 RLHF 的版本。
- **访问和下载选项**：该模型的 FP16 版本可以从 [Hugging Face](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B) 下载，而量化后的 GGUF 版本可以在[此处](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B-GGUF)获取。下载页面提供了详细的模型描述和对比。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B">NousResearch/Hermes-2-Theta-Llama-3-70B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B-GGUF">NousResearch/Hermes-2-Theta-Llama-3-70B-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1253065913649856542)** (272 条消息🔥🔥): 

- **模型解析与 Tokenization 辩论**：成员们讨论了构建解析器的复杂性，该解析器旨在将特定模型的格式处理为类似 OpenAI 的通用格式。大家达成共识，倾向于使用“反向模板（reverse templates）”，并可能将其集成到 `tokenizer_config.json` 中以增加灵活性。

- **Claude 3.5 Sonnet 令人印象深刻**：Claude 3.5 Sonnet 的发布引发了热议，多位用户注意到其性能和能力优于之前的模型。一位成员称其为“超快版 Opus”，并赞扬了它在解决复杂任务方面的能力。

- **Hermes 2 Theta 的适度去审查**：关于 **Hermes 2 Theta** 的讨论强调了它在保持信息丰富响应的同时，具有适度的去审查特性，这引发了关于拒绝回答与真实对话之间理想平衡的辩论。

- **将工具与模型集成**：一位成员解释了将工具与模型无缝集成的过程，特别是跳过 special tokens 并将工具响应直接插入 prompt。这种将工具合并到模型流中的方法被建议用于简化多工具集成。

- **Claude 3.5 Sonnet 的影响**：新的 Claude 模型展示了令人印象深刻的能力，包括准确理解和解决晦涩的编程问题。该模型的速度和专注度尤其受到赞赏，被视为一项重大进步。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/fireworks-ai/firefunction-v2">fireworks-ai/firefunction-v2 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.anthropic.com/news/claude-3-5-sonnet">Introducing Claude 3.5 Sonnet</a>: 介绍 Claude 3.5 Sonnet——我们目前最智能的模型。Sonnet 现在在关键评估中超越了竞争对手模型和 Claude 3 Opus，且速度翻倍。</li><li><a href="https://x.com/teortaxesTex/status/1803611506908529060">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: 明确地说。2024 年 6 月，Ilya Efimovich Sutskever 破坏了美中签署 AGI/ASI 约束性条约的可能性。这本来就很遥远，但从现在起，中国绝不可能...</li><li><a href="https://github.com/vllm-project/vllm">GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs</a>: 一个高吞吐量且显存高效的 LLM 推理和服务引擎 - vllm-project/vllm</li><li><a href="https://github.com/mudler/LocalAI/blob/43f0688a95ce5a5f43228ae288020bef02770e8e/pkg/functions/parse.go#L124">LocalAI/pkg/functions/parse.go at 43f0688a95ce5a5f43228ae288020bef02770e8e · mudler/LocalAI</a>: :robot: 免费、开源的 OpenAI 替代方案。自托管、社区驱动且本地优先。可在消费级硬件上运行，作为 OpenAI 的掉入式替代方案。无需 GPU。支持运行 gguf, trans...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1253102983932154010)** (2 条消息): 

- **即将推出令人兴奋的新资源**：一位成员称赞了另一位成员的问题，并暗示将创建一些特别的东西来回答它。另一位成员表达了热情，说道：“太不可思议了！等不及要看了。”
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1253394167527247964)** (3 条消息): 

- **Sonnet 3.5 发布，接下来是 Opus 3.5？**：一位成员宣布了 **3.5 Sonnet** 的发布，并推测 **Opus 3.5** 可能很快就会到来。“3.5 Sonnet 来了！推测 Opus 3.5 也不远了。😎”。
- **成员们对新发布表示兴奋**：另一位成员在公告发布后表达了兴奋。简单的反应：“令人兴奋”。
- **分享了音乐视频链接**：一位成员分享了一个 [YouTube 视频](https://youtu.be/E3Yt_qLUGJY)，标题为“L'ENTOURLOOP - Lobster Shwarama Ft. Troy Berkley & Khoe Wa (Official Video)”，推广音乐专辑“Chickens In Your Town”。

**提到的链接**: <a href="https://youtu.be/E3Yt_qLUGJY">L&#39;ENTOURLOOP - Lobster Shwarama Ft. Troy Berkley &amp; Khoe Wa (Official Video)</a>: &quot;Lobster Shwarama Feat Troy Berkley &amp; Khoe Wa&quot; 选自 L&#39;Entourloop 的 &quot;Chickens In Your Town&quot; 专辑，可在 👉 https://smarturl.it/LNTRLPChickensIYT 观看...

  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1253061758453813300)** (254 messages🔥🔥): 

- **不能直接使用 Hugging Face 位置**：一位用户澄清道，“*对于我们目前的内存数据集，它会从 HF 位置下载到本地磁盘。*”他们正在努力实现 *streaming datasets*（流式数据集），以避免将数据集保存在磁盘上。
- **在 Torchtune 中设置 HF 数据集**：用户讨论了在 `torchtune.dataset.chat_dataset` 中配置 HF 数据集，最终决定使用 `conversation_style: openai` 应该可以开箱即用（OOTB），无需额外的转换器。
- **最大序列长度混淆**：关于 **llama3** 的最大序列长度存在争议，最初认为是 4096，但最高可达 8192。然而，一位成员指出，“*我怀疑我的 VRAM 能否容纳这种大小的上下文。*”
- **处理崩溃和内存问题**：用户在训练模型时遇到了几次与 RAM 相关的崩溃，特别是在使用 qlora 和 lora 时。讨论的一个可能解决方案是将层卸载（offload）到 *CPU* 并排查 ROCm 设置问题。
- **ROCm 设置经验**：成员们分享了为 AMD GPU 设置 ROCm 的相关问题和资源，其中一人表示，“*你需要从源码构建它们，或者从替代来源获取。*”他们还提到了一个 Reddit 资源，有人在 6900 XT 上成功运行了 ROCm。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/torchtune/main/tutorials/chat.html">使用聊天数据微调 Llama3 &mdash; torchtune 主文档</a>: 未找到描述</li><li><a href="https://github.com/pytorch/torchtune/discussions/1090">有人在 AMD ROCm（特别是 RDNA3 / gfx1100）上运行过这个吗 · pytorch/torchtune · Discussion #1090</a>: 我向 ROCm 团队提交了一个 issue ROCm/hipBLASLt#831，但只是好奇是否有其他人（作者或用户）尝试过在 ROCm 上运行 torchtune？我使用的是非常标准的安装（最新版本...</li><li><a href="https://pytorch.org/torchtune/stable/install.html#install-nightly-build">安装说明 &mdash; TorchTune 文档</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1clxhtp/amd_rocm_61_local_multigpu_6900xt_vii_setup/">Reddit - 深入了解任何事物</a>: 未找到描述</li><li><a href="https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html">系统要求 (Linux) — ROCm 安装 (Linux)</a>: 未找到描述</li><li><a href="https://github.com/pytorch/torchtune/blob/ef6e196d8e47e9bc584bc9f7ce836f646443381f/recipes/lora_finetune_single_device.py#L277C9-L277C50">torchtune/recipes/lora_finetune_single_device.py at ef6e196d8e47e9bc584bc9f7ce836f646443381f · pytorch/torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/s1nMpiuc1Q">Reddit - 深入了解任何事物</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1253447468239294606)** (1 条消息): 

- **MidJourney 图像描述数据集上线 GitHub**: 一个包含 52 万条 MidJourney 图像+描述的 [数据集](https://github.com/bghira/SimpleTuner/blob/main/documentation/QUICKSTART.md) 已发布。它为训练和实验提供了丰富的视觉数据。
  
- **PixArt 模型亮点：拥有 9 亿参数**: 包含公开照片、DALLE-3 + Midjourney 数据集，拥有 900M 参数的 PixArt [模型](https://huggingface.co/ptx0/pixart-900m-1024-ft) 现已可用。该模型利用多样化的数据源进行图像生成任务。

- **Bulkproteinviz 提升蛋白质预测能力**: 更新后的 [Proteinviz](https://huggingface.co/spaces/as-cle-bert/proteinviz) 现在支持同时进行多个结构预测。对于从事计算生物学的人员来说，这是一个重大的进步。

- **Powershell 结合 AI 并支持 function calling**: [Powershell + AI 集成](https://github.com/rrg92/powershai) 现在包含对 function calling 的支持。此更新为开发人员和自动化专家提供了增强的功能。

- **药物名称替换影响模型性能**: 最近的一项研究在生物医学基准测试中将通用药物名称替换为品牌名称，结果显示大多数模型的性能都有所下降。该研究的完整论文可以在 [这里](http://arxiv.org/abs/2406.12066) 找到，排行榜可在 [Hugging Face](https://huggingface.co/spaces/AIM-Harvard/rabbits-leaderboard) 上查看。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/taha_yssne/status/1802607279809630562)">来自 Taha Yassine (@taha_yssne) 的推文</a>: 我刚刚写了一篇关于 LLM 中 temperature 参数的博客文章，但这其实只是玩玩 Transformers.js 的借口。我很高兴能实现一个关于 T 对生成影响的交互式演示...</li><li><a href="https://x.com/shan23chen/status/1803459255518769509)">来自 Shan Chen (@shan23chen) 的推文</a>: 💊 我们把你的语言模型带到了药店……它对对乙酰氨基酚（通用名）的了解比泰诺（品牌名）更好！@hughbzhang @scale_AI 上个月开发了 GSM1K，他们发现...</li><li><a href="https://blog.cubed.run/5-chunking-techniques-in-rag-1250c8e1f49f)">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1253061783716102299)** (142 条消息🔥🔥): 

- **Stable Diffusion 与 VSCode 故障排除**：一位用户询问如何将 **Stable Diffusion** 与 VSCode 集成。另一位用户澄清说，由于 VSCode 只是一个编辑器，用户应该使用其中的终端来运行所需的命令。

- **Llama 3 微调问题**：一位用户在使用 TRL 和 QLoRa 微调 **Llama 3** 时遇到问题，并得到了半无意义的输出。他们分享了代码并讨论了参数的潜在问题，链接了一些研究并尝试了不同的 `lora_rank`。

- **Stable Diffusion 3 错误**：一些用户报告在尝试使用 **Stable Diffusion 3** 时因缺少模型索引而报错。一位用户提到使用 **stable-diffusion-3-medium-diffusers** 模型作为变通方案。

- **Hugging Face 服务中断**：多位用户报告了 504 错误以及访问 **Hugging Face** 服务时的偶发问题。状态页面最初显示所有服务均在线，导致了困惑以及对服务器过载的猜测。

- **Florence-2 模型给用户留下深刻印象**：来自 Microsoft 的 **Florence-2** 大模型以其多功能性和高效性给用户留下了深刻印象，尽管体积较小，但支持字幕生成（captioning）、OCR 和目标检测等任务。讨论强调了它在 Raspberry Pi 等低功耗设备上的潜在用途，以及与 DINOv2 等其他模型的比较。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/learn/deep-rl-course/unitbonus1/train">让我们训练并与 Huggy 一起玩耍 🐶 - Hugging Face 深度强化学习课程</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb">sample_inference.ipynb · microsoft/Florence-2-large at main</a>：未找到描述</li><li><a href="https://youtu.be/S_3TVEVE8y4?si=4zw5brZDugG4yFtY">Blade Runner Off-World 2055</a>：使用 MidJourney, RunwayML, ElevenLabs, Magnific Maya &amp; Zbrush 制作的《银翼杀手》Off-World AI / 3D 短片。#sora #Ai #3dmodeling #ridleyscott #giger 场景：The...</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard - 由 open-llm-leaderboard 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Florence-2-large">microsoft/Florence-2-large · Hugging Face</a>：未找到描述</li><li><a href="https://youtu.be/iy9Z4DyHxvE?si=eHxvWI4rfSwk8YH1">如何使用 PyReft 攻击 LLM（使用你自己的数据进行微调！）</a>：🚀 订阅时事通讯 go.coursesfromnick.com/newsletter 👨‍💻 报名全栈课程并使用 YOUTUBE50 获得 50% 折扣：https://www.coursesfrom...</li><li><a href="https://en.wikipedia.org/wiki/White_Christmas_(Black_Mirror)">白色圣诞 (黑镜) - 维基百科</a>：未找到描述</li><li><a href="https://huggingface.co/blog/llama3">欢迎 Llama 3 - Meta 的新开源 LLM</a>：未找到描述</li><li><a href="https://tenor.com/view/frank-castle-wait-please-stop-please-no-please-gif-21133188">Frank Castle Wait GIF - Frank Castle Wait Please Stop - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/PWhiddy/PokemonRedExperiments/blob/master/windows-setup-guide.md">PokemonRedExperiments/windows-setup-guide.md at master · PWhiddy/PokemonRedExperiments</a>：使用强化学习玩《口袋妖怪红》。通过在 GitHub 上创建账号为 PWhiddy/PokemonRedExperiments 的开发做出贡献。</li><li><a href="https://github.com/PWhiddy/PokemonRedExperiments">GitHub - PWhiddy/PokemonRedExperiments: 使用强化学习玩《口袋妖怪红》</a>：使用强化学习玩《口袋妖怪红》。通过在 GitHub 上创建账号为 PWhiddy/PokemonRedExperiments 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17pw7bv/eternal_question_what_rank_r_and_alpha_to_use_in/">Reddit - 深入探索任何事物</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1ZyKswO6xDbTuyMQw5NTSlri1fRHw43_l?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://status.huggingface.org/">
Hugging Face 状态
</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1253168410032865371)** (3 条消息): 

- **多表数据库合成分析**：一篇分享的[文章](https://mltechniques.com/2024/06/15/synthesizing-multi-table-databases-model-evaluation-vendor-comparison/)重点讨论了生成高质量合成多表表格数据的挑战，特别是带有日期列的数据。它评估了三家供应商，强调了 SDV、Gretel 和 Mostly.ai 等库在保持数据完整性、运行时间以及符合业务规则方面的困难。
- **ToolkenGPT 提出创新方法**：这篇[论文](https://doi.org/10.48550/arXiv.2305.11554)讨论了 **ToolkenGPT**，它将外部工具嵌入为 token 供大型语言模型 (LLM) 使用，其方式类似于生成常规词 token。这旨在克服受限于上下文长度和工具数量的 finetuning 和 in-context learning 的局限性。
- **生物医学基准测试中的药物名称揭示性能下降**：一条推文强调了一项研究，在该研究中，MedQA 和 MedMCQA 等生物医学基准测试中的通用药物名称被替换为品牌名称，并指出大多数模型的性能都有所下降。这项实验在[论文](http://arxiv.org/abs/2406.12066)和 Hugging Face [leaderboard](https://huggingface.co/spaces/AIM-Harvard/rabbits-leaderboard) 中有详细说明，表明公共预训练数据集中可能存在数据污染。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://mltechniques.com/2024/06/15/synthesizing-multi-table-databases-model-evaluation-vendor-comparison/">Synthesizing Multi-Table Databases: Model Evaluation &amp; Vendor Comparison - Machine Learning Techniques</a>：与单表相比，合成多表表格数据具有其自身的挑战。当数据库包含交易日期或入院日期等日期列时（这在现实中经常发生）...</li><li><a href="https://doi.org/10.48550/arXiv.2305.11554">ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings</a>：使用外部工具增强大型语言模型 (LLM) 已成为解决复杂问题的一种很有前景的方法。然而，传统方法使用工具演示数据对 LLM 进行 finetune...</li><li><a href="https://x.com/shan23chen/status/1803459255518769509?s=46">Shan Chen (@shan23chen) 的推文</a>：💊 我们把你的语言模型带到了药店……它对 acetaminophen（通用名）的了解比对 Tylenol（品牌名）更好！@hughbzhang @scale_AI 上个月开发了 GSM1K，他们在那里发现了许多...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1253125109837860914)** (21 条消息🔥): 

- **关于在服务器中推广 AI 翻唱的辩论**：成员们讨论了在 HuggingFace 服务器上推广 AI 翻唱是否合适。一位用户澄清说，他们不希望公开分享模型，只推广 Kpop 歌曲的 AI 翻唱，而不是他们自己的原创作品。
- **生物医学 NLP 模型对通用药物名称的识别效果更好**：用户讨论了一项研究，该研究表明 NLP 模型对 acetaminophen（通用名）等通用药物名称的识别效果优于 Tylenol（品牌名）等品牌名。该研究发现了数据污染的证据，并分享了与数据集 "RABBITS" 相关的[论文](http://arxiv.org/abs/2406.12066)和[leaderboard](https://huggingface.co/spaces/AIM-Harvard/rabbits-leaderboard)。
- **开源蛋白质结构预测工具更新**：发布了关于 **BulkProteinviz** 的公告，这是开源蛋白质结构预测工具中的一个新功能，允许从 FASTA 文件进行多次预测，从而提高研究速度。
- **用于图像生成的大规模数据和模型发布**：一次全面的更新，包括指南、数据集和模型，例如用于 Midjourney v6 的 52 万个图像文件、训练一个 900M 参数的 PixArt 以及 SD3 模型的新 fine-tunes。
- **开源西洋双陆棋模拟项目**：宣布了一个新项目，该项目运行高速西洋双陆棋模拟，并在 [GitHub - C1N-S4/Backgamoon-A.I-tool](https://github.com/C1N-S4/Backgamoon-A.I-tool) 开放进一步开发。该项目旨在增加用户界面和优化增强。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/shan23chen/status/1803459255518769509?s=46">Shan Chen (@shan23chen) 的推文</a>：💊 我们把你的语言模型带到了药店……它对 acetaminophen（通用名）的了解比对 Tylenol（品牌名）更好！@hughbzhang @scale_AI 上个月开发了 GSM1K，他们在那里发现了许多...</li><li><a href="https://github.com/C1N-S4/Backgamoon-A.I-tool">GitHub - C1N-S4/Backgamoon-A.I-tool</a>：通过在 GitHub 上创建一个账户来为 C1N-S4/Backgamoon-A.I-tool 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1253065454411579443)** (6 条消息): 

- **新用户寻求漫画风格模型推荐**：一位成员表示有兴趣使用与漫画风格相关的数据集对模型进行微调。目标是“以相似/相同的风格生成新的作品”。
- **分享了 Deep Guided Posterior Regularization**：另一位成员提供了 [DeGPR for Medical Object Detection](https://paperswithcode.com/task/medical-object-detection/latest) 的链接。讨论涵盖了将通用深度学习方法应用于医学图像的挑战，例如类别不平衡和微小重叠对象。
- **请求基于 Java 的目标检测应用**：一位成员询问如何使用 Java 创建目标检测应用。他们对自定义检测和实时检测等功能感兴趣，类似于 Python 中的 YOLO。

**提到的链接**：<a href="https://paperswithcode.com/task/medical-object-detection/latest">Papers with Code - Medical Object Detection</a>：医学目标检测是识别图像中基于医学的对象的任务。
 
 <span style="color:grey; opacity: 0.6">( 图像来源: [Liver Lesion Detection from Weakly...

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1253142876699885669)** (7 条消息): 

- **分享了值得关注的资源**：一位用户分享了资源和 Notebook，例如[这篇关于使用 llama2-faiss 和 Langchain 的指南](https://github.com/murtuza753/llama2-faiss-langchain-qa-rag/blob/main/Using_llama2_faiss_and_langchain_for_question_answering_on_your_own_data.ipynb)，以及另一篇关于[使用 PEFT 进行 LLM 微调](https://github.com/ashishpatel26/LLM-Finetuning)的文章。

- **RAG 流水线中的重排序（Reranking）问题**：一位用户提出了其 RAG 流水线中的重排序问题，对于某些查询（如“我的产品是什么？”），重排序分数较低。他们正在考虑根据向量数据库的结果来调整重排序分数。

- **寻求 vLLM 微调方面的帮助**：一位用户请求关于使用 vLLM 微调模型的指导，并在加载微调模型和 LoRA 适配器时遇到了一个特定错误：*"ValueError: Cannot find any of ['adapter_name_or_path'] in the model's quantization config."*

- **模型微调的详细步骤**：同一位用户详细说明了使用 PEFT 进行微调、合并、保存和加载模型的步骤，包括尝试使用 vLLM 进行推理。步骤包括保存 Checkpoint、合并模型、处理 Device Map 以及随后遇到的错误。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/murtuza753/llama2-faiss-langchain-qa-rag/blob/main/Using_llama2_faiss_and_langchain_for_question_answering_on_your_own_data.ipynb">llama2-faiss-langchain-qa-rag/Using_llama2_faiss_and_langchain_for_question_answering_on_your_own_data.ipynb at main · murtuza753/llama2-faiss-langchain-qa-rag</a>：通过在 GitHub 上创建账号来为 murtuza753/llama2-faiss-langchain-qa-rag 的开发做出贡献。</li><li><a href="https://github.com/ashishpatel26/LLM-Finetuning">GitHub - ashishpatel26/LLM-Finetuning: LLM Finetuning with peft</a>：使用 PEFT 进行 LLM 微调。通过在 GitHub 上创建账号来为 ashishpatel26/LLM-Finetuning 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1253313451791613952)** (2 条消息): 

- **寻求 Llama 3:70B 训练方面的帮助**：一位用户提到他们通过 **Ollama** 安装了 Llama 3:70B，但指出目前的训练数据仅为 40GB。他们正在寻求关于如何训练额外数据集以在本地将规模增加到至少 200GB 的建议。
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1253135824279699496)** (53 条消息🔥): 

- **MLIR：一个内部且令人困惑的 Dialect**：一位用户对 MLIR 中的 **`kgen` dialect** 表示困惑，透露这是一个**没有公开文档的内部 dialect**。另一位用户表达了挫败感，补充道：“在 MLIR 本就混乱的基础上，代码简直乱得一塌糊涂。”
  
- **在 MLIR 中实现自定义类型**：关于在 MLIR 中实现 **256 位整数** 的讨论引出了一些建议，例如使用 **`SIMD[DType.int64, 4]`** 或直接 **定义一个 `i256` 类型**。讨论中提供了一些有用的参考资料，包括一个 [GitHub 链接](https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/simd.mojo#L231-L232)。

- **理解 MLIR 及其转换**：一位用户对 MLIR 如何转换为 LLVM IR 进行优化感到困惑，这引发了关于 **MLIR dialects 和转换基础设施** 的解释。会议澄清了 MLIR 操作可以通过自定义后端或通过 lowering 到 LLVM IR 来转换为汇编代码。

- **Mojo 包管理器的进展**：一位用户询问了 Mojo 包管理器的进展。不同的用户指出了现有的努力，例如 GitHub 上的 [Hammad-hab 的 `pkm`](https://github.com/Hammad-hab/pkm)，并提到 Modular 团队已在社区会议中讨论过此事。

- **Modular 社区直播公告**：社区收到了关于 **Modular 社区直播** 的通知，讨论了 MAX Engine 和 Mojo 的新功能，可在 [YouTube](https://www.youtube.com/watch?v=uookgZ7Ojg8) 上观看。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/roadmap#exception-is-actually-called-error">Mojo🔥 路线图与注意事项 | Modular 文档</a>：我们 Mojo 计划的摘要，包括即将推出的功能和我们需要修复的问题。</li><li><a href="https://www.youtube.com/watch?v=uookgZ7Ojg8">Modular 社区直播 - MAX 24.4 新特性</a>：MAX 24.4 现已发布！加入我们即将举行的直播，我们将讨论 MAX Engine 和 Mojo🔥 中的新功能 —— macOS 上的 MAX、MAX Engine 量化 API 等。</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/simd.mojo#L231-L">mojo/stdlib/src/builtin/simd.mojo at main · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/simd.mojo#L231-L232">mojo/stdlib/src/builtin/simd.mojo at main · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2002.11054">MLIR：应对摩尔定律终结的编译器基础设施</a>：这项工作介绍了 MLIR，一种构建可重用和可扩展编译器基础设施的新方法。MLIR 旨在解决软件碎片化问题，改进异构硬件的编译...</li><li><a href="https://github.com/Hammad-hab/pkm">GitHub - Hammad-hab/pkm: Mojo 的非官方包管理器</a>：Mojo 的非官方包管理器。通过在 GitHub 上创建账号为 Hammad-hab/pkm 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1253388280100294656)** (2 条消息): 

- **Modular 新动态提醒**：Modular 最近在其 Twitter 账号上分享了两条与新进展相关的推文。查看 [此处](https://twitter.com/Modular/status/1803828734992207974) 和 [此处](https://twitter.com/Modular/status/1803850466767573143) 的更新以获取更详细的信息。
  

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1253075360229363825)** (56 messages🔥🔥): 

- **Mojo 已部分开源**：Mojo 语言本身是开源的，可以在 [GitHub](https://github.com/modularml/mojo) 上获取。然而，编译器目前仍是专有的，但提供免费许可，其部分组件正在逐步开源（[详细博客文章](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source)）。
- **Intable 类型处理详解**：用户在 `count_many_things` 函数中使用 `Intable` 时遇到了显式类型转换的问题。官方澄清，为了防止错误，需要显式转换为 `int`，并且 `Intable` 表示该类型具有 `__int__` 方法。
- **Mojo 用于生产环境的当前局限性**：有观点认为 Mojo 尚缺乏生产环境所需的关键特性，如包管理器、高级 stdlib 函数、traits、async 等。针对实际软件开发的更完善版本预计在 12 月左右发布。
- **关于在自动化工作中使用 Mojo 的建议**：Athena 分享了对 Mojo 隐式并行性和增强类型安全性的兴趣。然而，在 Mojo 进一步成熟和稳定之前，建议不要将其集成到生产或复杂的自动化工作中。
- **Claude Sonnet 3.5 的能力**：简要提到 Claude 新的 Sonnet 3.5 模型在生成 Mojo 代码方面优于 GPT-4，这表明它对于想要实现或实验 Mojo 代码的开发者非常有用。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/blog/the-next-big-step-in-mojo-open-source">Modular: Mojo🔥 开源迈出的下一大步</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo🔥 开源迈出的下一大步</li><li><a href="https://www.perplexity.ai/">Perplexity</a>：Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md">mojo/CONTRIBUTING.md at main · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://docs.modular.com/mojo/faq#will-mojo-be-open-sourced">Mojo🔥 常见问题解答 | Modular 文档</a>：关于 Mojo 预期问题的解答。</li><li><a href="https://github.com/modularml/mojo/">GitHub - modularml/mojo: Mojo 编程语言</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1253072418176761999)** (1 messages): 

- **Engine 文档澄清 'execute' 函数细节**：一位成员指出 `execute` 方法可以接受变长参数 `NamedTensor` 或 `Tuple[StringLiteral, EngineNumpyView]`，并附带了 [文档](https://docs.modular.com/max/api/mojo/engine/model/Model#execute) 链接作为支持。他们还提供了 [NamedTensor 文档](https://docs.modular.com/max/api/mojo/engine/tensor/NamedTensor) 链接供进一步参考。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/max/api/mojo/engine/model/Model#execute">Model | Modular 文档</a>：表示已加载并准备好执行的模型。</li><li><a href="https://docs.modular.com/max/api/mojo/engine/tensor/NamedTensor">NamedTensor | Modular 文档</a>：命名的输入张量。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1253128778792829070)** (8 messages🔥): 

- **新版 Mojo 编译器发布**：Mojo 编译器的最新 nightly 更新版本 `2024.6.2005` 已发布。用户可以使用 `modular update nightly/mojo` 进行更新，并查看 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 以及 [raw diff](https://github.com/modularml/mojo/compare/d96acc9161ce91d93d9a24424cb8870906440e05...279ade23a9409a545a723236f271c5061d2f005b)。
- **stdlib 贡献者工具**：分享了一个名为 "mojo_dev_helper" 的工具，旨在为标准库（stdlib）贡献者提供帮助。更多详情请见 [此处](https://github.com/rd4com/mojo_dev_helper)。
- **导入 initialize_pointee_move 的问题**：一位成员在导入 `initialize_pointee_move` 时遇到问题。已澄清该方法在 nightly 分支的 `UnsafePointer` 中可用，但在其他版本中可能无法访问。

**提及的链接**：<a href="https://github.com/rd4com/mojo_dev_helper">GitHub - rd4com/mojo_dev_helper: 🦺 small tool for stdlib contributors.</a>：🦺 供 stdlib 贡献者使用的小工具。通过在 GitHub 上创建账号来为 rd4com/mojo_dev_helper 的开发做出贡献。

  

---



### **AI Stack Devs (Yoko Li) ▷ #[committers](https://discord.com/channels/1122748573000409160/1122748682475950142/1253138286860304544)** (7 messages): 

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Committers 频道摘要</title>
</head>
<body>
  <ul>
    <li><strong>垃圾邀请信息席卷频道</strong>：频道内发布了多条推广“18+ 免费内容”和 OnlyFans 泄露的消息。每条消息都包含重复的邀请，附带 Discord 链接和露骨内容的描述。</li>
  </ul>
</body>
</html>

**提及的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：查看 Discord 上的 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与其他 9061 名成员一起闲聊，享受免费的语音和文字聊天。

  

---


### **AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/1253138294607183904)** (6 messages): 

```html
<ul>
    <li><strong>垃圾信息和不当内容泛滥</strong>：多条消息通过附带的 <a href="https://discord.gg/2AFWP2Qd2r">Discord 链接</a> 广告“18+ 免费内容、OnlyFans 泄露和性爱视频通话”。这些消息反复 @everyone，表明存在严重的垃圾信息问题。</li>
</ul>
```

**提及的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：查看 Discord 上的 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与其他 9061 名成员一起闲聊，享受免费的语音和文字聊天。

  

---


### **AI Stack Devs (Yoko Li) ▷ #[feedback](https://discord.com/channels/1122748573000409160/1122749120885575812/1253406090700521572)** (6 messages): 

<html>
<body>
<ul>
  <li><strong>Discord 频道垃圾机器人警报</strong>：一个名为 <code>bot1198</code> 的机器人反复发布关于“18+ 免费内容”和 OnlyFans 泄露的消息。每条消息都包含一个可疑链接：<a href="https://discord.gg/2AFWP2Qd2r">discord.gg/2AFWP2Qd2r</a>。</li>
</ul>
</body>
</html>

**提及的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：查看 Discord 上的 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与其他 9061 名成员一起闲聊，享受免费的语音和文字聊天。

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1253138300449980427)** (6 messages): 

```html
- **重复推广 18+ 内容**：多条消息推送了“免费内容、OnlyFans 泄露和性爱视频通话”的链接。它们还包含加入 Discord 服务器的邀请，URL 为 [discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r)。 
```

**提及的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：查看 Discord 上的 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与其他 9061 名成员一起闲聊，享受免费的语音和文字聊天。

  

---

### **AI Stack Devs (Yoko Li) ▷ #[team-up](https://discord.com/channels/1122748573000409160/1128471951963328512/1253138304841551882)** (6 条消息): 

- **18+ 内容垃圾信息警报**：一名用户发布了多条垃圾信息，提供“18+ 免费内容”，包括 OnlyFans 泄露和实时成人视频通话链接。消息中还提到了 Nitro boost 抽奖，并提供了一个 Discord 邀请链接 [https://discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r)。

**提到的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：在 Discord 上查看 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与 9061 名其他成员一起交流，享受免费的语音和文字聊天。

  

---


### **AI Stack Devs (Yoko Li) ▷ #[events](https://discord.com/channels/1122748573000409160/1131651713204498583/1253138308125687878)** (7 条消息): 

- **不当内容垃圾信息**：发布了多条推广 **“18+ 免费内容，OnlyFans 泄露”** 和成人视频通话的消息。这些消息包含一个用于访问内容的 [Discord 邀请链接](https://discord.gg/2AFWP2Qd2r)。

**提到的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：在 Discord 上查看 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与 9061 名其他成员一起交流，享受免费的语音和文字聊天。

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1253138311409696768)** (6 条消息): 

- **垃圾信息警报：露骨内容和抽奖诈骗**：多条消息推广了 18+ 内容、OnlyFans 泄露和成人视频通话。这些消息还宣传了一个正在进行的 Nitro Boost 抽奖，并附带了 [Discord 链接](https://discord.gg/2AFWP2Qd2r)。

**提到的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：在 Discord 上查看 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与 9061 名其他成员一起交流，享受免费的语音和文字聊天。

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1253237449778397207)** (6 条消息): 

```html
- **Alert: Block the user**: A member highlighted the need to **report and block** a certain user. Another member confirmed action has been taken with *"Thank you!! Done"*.

- **Spam Attack**: The channel was hit by **repeated spam messages** promoting 18+ content and a Discord invite link. The spam included *"onlyfans leaks and she doing now sexcam video call @everyone"*.
```

**提到的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：在 Discord 上查看 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与 9061 名其他成员一起交流，享受免费的语音和文字聊天。

  

---


### **AI Stack Devs (Yoko Li) ▷ #[late-night-lounge](https://discord.com/channels/1122748573000409160/1159342774710186075/1253138322541252691)** (7 条消息): 

- **垃圾信息席卷 late-night-lounge**：该频道被推广“18+ 免费内容、OnlyFans 泄露和成人视频通话”的垃圾信息刷屏，并包含邀请链接 `https://discord.gg/2AFWP2Qd2r`。消息重复多次，并使用 `@everyone` 标签引起注意。

**提到的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：在 Discord 上查看 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与 9061 名其他成员一起交流，享受免费的语音和文字聊天。

  

---


### **AI Stack Devs (Yoko Li) ▷ #[local-ai-stack](https://discord.com/channels/1122748573000409160/1168947823920812125/1253138328320999424)** (6 条消息): 

- **18+ 免费内容垃圾信息**：发送了多条广告，宣传 *18+ 免费内容*、**OnlyFans 泄露**和*成人视频通话*。每条消息都包含一个指向 Discord 服务器的链接 ([discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r))。

**提到的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：在 Discord 上查看 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与 9061 名其他成员一起交流，享受免费的语音和文字聊天。

  

---

### **AI Stack Devs (Yoko Li) ▷ #[assets](https://discord.com/channels/1122748573000409160/1176906086368935966/1253138332662239232)** (7 条消息): 

```html
- **成人内容垃圾信息警报**：分享了多条重复消息，推广包括 OnlyFans 泄露和性爱摄像头视频通话在内的 *18+ 免费内容*。每条消息中都包含一个指向 Discord 服务器 ([https://discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r)) 的可疑链接。
```

**提到的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：在 Discord 上查看 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与 9061 名其他成员一起交流，享受免费的语音和文字聊天。

  

---


### **AI Stack Devs (Yoko Li) ▷ #[🐣ai-tamago](https://discord.com/channels/1122748573000409160/1182765527211462716/1253138341621141596)** (6 条消息): 

- **垃圾邀请信息刷屏频道**：发布了多条消息，邀请用户访问“18+ 免费内容、OnlyFans 泄露，以及她正在进行的性爱摄像头视频通话 @everyone”。同一个 Discord 链接 [https://discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r) 被反复分享。

**提到的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：在 Discord 上查看 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与 9061 名其他成员一起交流，享受免费的语音和文字聊天。

  

---


### **AI Stack Devs (Yoko Li) ▷ #[multi-modal-starter-kit](https://discord.com/channels/1122748573000409160/1224949149380771880/1253138345463386173)** (7 条消息): 

```html
- **垃圾信息警报：不当内容和钓鱼链接**：发布了多条广告消息，宣传“18+ 免费内容、OnlyFans 泄露和性爱摄像头视频通话”，这些很可能是诈骗。消息中包含一个指向可疑且可能有害的 Discord 服务器的链接：[https://discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r)。
```

**提到的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：在 Discord 上查看 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与 9061 名其他成员一起交流，享受免费的语音和文字聊天。

  

---


### **AI Stack Devs (Yoko Li) ▷ #[paper-spam](https://discord.com/channels/1122748573000409160/1227492197541220394/1253138349296844860)** (6 条消息): 

```html
- **频道垃圾信息刷屏**：多条垃圾消息反复发布，宣传“18+ 免费内容、OnlyFans 泄露和性爱摄像头视频通话”。帖子中包含一个 Discord 邀请链接：[discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r)。
- **对公开 Bot 的兴趣**：一名用户表示有兴趣了解某个 Bot 是否公开，并提到“想为我的地方抓一个 :3”。该消息的上下文尚不明确，但似乎与垃圾信息无关。
```

**提到的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：在 Discord 上查看 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与 9061 名其他成员一起交流，享受免费的语音和文字聊天。

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-raspberry-pi](https://discord.com/channels/1122748573000409160/1234912245415280742/1253138352748892261)** (4 条消息): 

```html
- **垃圾信息警报刷屏频道**：发布了多条推广 18+ 内容、OnlyFans 泄露和性爱摄像头视频通话的消息。这些垃圾消息包含同一个 Discord 邀请链接：[discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r)。
```

**提到的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：在 Discord 上查看 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与 9061 名其他成员一起交流，享受免费的语音和文字聊天。

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-explained-cartoons](https://discord.com/channels/1122748573000409160/1249527870750195802/1253138355726843926)** (5 条消息): 

```html
- **Discord 频道被成人内容链接刷屏**：多条推广“18+ 免费内容、OnlyFans 泄露以及她正在进行的性爱摄像头视频通话”的消息被反复发布。所有消息中提供的链接均为 https://discord.gg/2AFWP2Qd2r。
```

**提到的链接**：<a href="https://discord.gg/2AFWP2Qd2r">加入 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord 服务器！</a>：在 Discord 上查看 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 社区 —— 与 9061 名其他成员一起交流，享受免费的语音和文字聊天。

  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1253062485473366217)** (46 条消息🔥): 

- **LM Studio 在 0.2.23 版本中提速**：用户注意到运行 LM Studio 0.2.23 带来了显著的速度提升。一位用户表示：“它在 LM Studio 0.2.23 上可以运行，呼！而且不知为何速度快了很多。”

- **Deepseek Coder v2 面临挑战**：多位用户遇到了 Deepseek Coder v2 的问题，引用了“不支持的架构（unsupported architecture）”错误。一位用户提到，通过关闭 flash attention 并在 0.2.25 版本中使用 deepseek coder 预设取得了成功。

- **探索 LLM 服务器的前端选项**：用户讨论了为本地服务器运行前端以便在不同设备上使用 LLM 的可能性。建议包括浏览 GitHub 仓库，如 [every-chatgpt-gui](https://github.com/billmei/every-chatgpt-gui) 和 [awesome-chatgpt](https://github.com/uhub/awesome-chatgpt)。

- **对 Reddit 审查制度的沮丧**：成员们对 local llama 子版块中严厉的版务管理表示不满。一位用户哀叹一个高赞帖子被突然删除，并猜测版务管理是否是自动化的。

- **NVLink 与内存考量**：用户讨论了 GPU 升级，包括 NVLink 支持以及在运行大型模型时 GPU RAM 相对于 CPU RAM 的重要性。一位用户指出：“据我发现，CPU RAM 帮助不大。它允许你加载更大的模型，但不会加快推理（inference）速度。”

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/billmei/every-chatgpt-gui">GitHub - billmei/every-chatgpt-gui: Every front-end GUI client for ChatGPT</a>：ChatGPT 的各类前端 GUI 客户端。可以通过在 GitHub 上创建账号为 billmei/every-chatgpt-gui 的开发做出贡献。</li><li><a href="https://github.com/uhub/awesome-chatgpt">GitHub - uhub/awesome-chatgpt: A curated list of awesome ChatGPT related projects.</a>：精选的 ChatGPT 相关优秀项目列表。- uhub/awesome-chatgpt
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1253080353812254795)** (16 条消息🔥): 

- **Nvidia 的新模型针对负强化**：一位成员指出 *“Nvidia 似乎想要一个既能生成负强化内容也能生成正强化内容的模型”*。这应该会吸引那些厌倦了过度正向叙事的用户。
  
- **Opus 上下文容量受到质疑**：成员们对 **Opus** 能处理多少上下文感到好奇，推测可能是 8k tokens，但也有人希望能更多。一位成员提到，“128k token 上下文，擅长 RAG，‘无审查’”，并链接了 [Cohere 博客](https://cohere.com/blog/command-r-plus-microsoft-azure)上的一篇详细介绍。

- **DeepSeek Coder V2 Lite 的中文回复问题**：用户报告 **DeepSeek Coder V2 Lite** 在使用官方提示词模板时默认以中文回复。然而，使用旧的 DeepSeek 提示词模板或 Vicuna 模板可以解决该问题，使其以英文输出。

- **DeepSeek 语言输出结果不一**：一位成员观察到，*“我看到了各种报告，对于它为什么选择说中文没有明确的原因”*，甚至考虑完全卸载以尝试解决问题。另一位成员发现，尽管最初存在问题，但使用旧模板可以成功以英文输出。

- **Midnight Miqu 模型的性能**：一位用户表示相比于 **Midnight Miqu 70 和 103**，他们更倾向于一个新模型，在使用了几个小时后发现它表现更好。他们计划进行更多测试以验证初步印象。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF">lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF · Hugging Face</a>：暂无描述</li><li><a href="https://cohere.com/blog/command-r-plus-microsoft-azure">Introducing Command R+: A Scalable LLM Built for Business</a>：Command R+ 是一款先进的 RAG 优化模型，旨在处理企业级工作负载，并首先在 Microsoft Azure 上可用。今天，我们推出 Command R+，这是我们最强大的...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1253063049871622255)** (3 条消息): 

- **LM Studio 无法浏览克隆的仓库**：一名成员指出，即使仓库已经克隆到本地，**LM Studio** 也不具备浏览仓库的能力。另一名成员询问是否可以将内容转换为 txt 文件以便访问。
- **手动喂入（Manual feeding）让一切皆有可能**：针对关于 txt 文件的提问，另一名成员表示，只要手动为 **LLM** 提供文本，在拥有充足的上下文空间和良好硬件的情况下，处理这些内容是可行的。
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/)** (1 条消息): 

darkhunter123: 嘿，在一个模型上实现多用户推理（multi user inference）是否可行？
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1253068271356346559)** (17 条消息🔥): 

- **为 Nemotron-4-340B 组装电脑需要 H100**：当有人询问运行如此庞大模型所需的合适硬件时，一名成员建议需要多块 **H100 GPU** 才能高效运行 **Nemotron-4-340B**。
- **运行 Meta-Llama-3-70B-Instruct 的配置**：一位用户分享了他们的配置，包括 **Ryzen 9 7950X**、**64GB DDR5** 和 **RTX 4090**，并尝试运行 **Meta-Llama-3-70B-Instruct**。
- **硬件建议**：讨论中包括了一些建议，例如为 34B 的 **LLM** 配备 **~38GB+ 的 VRAM**，以及在内存受限时考虑使用 **meta llama 8B** 等较小模型以获得更好的性能。
- **对 7900XT 的支持和 GPU 加速**：用户询问 **AMD 7900XT** 是否支持 **LLM** 的 **GPU** 加速，发现 **ROCm 支持** 可能使其可行，但其他用户遇到了由于不支持的 CPU 架构导致 **LM Studio 启动失败** 的问题。
- **辩论 LLM 的 GPU 选择**：成员们辩论了等待 **5090** 与购买翻新 **3090** 或 **4090** 的优劣，其中一人指出 **3090** 提供了更好的性价比，且拥有与 4090 类似的 **VRAM**。
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1253149011318276107)** (9 条消息🔥): 

- **LM Studio 在 Linux Mint 上无法识别 4070 GPU**：一位用户报告称，尽管安装了正确的 Nvidia 驱动程序，**LM Studio** 仍无法识别其 **4070 GPU**。他们目前正在使用 GUI 并寻找解决方案。
  
- **DeepseekV2 模型出现问题**：确认 **DeepseekV2 lite 和 large 模型无法在 LM Studio 中运行**；一位用户遇到了模型错误。**GPU offloading** 似乎是一个影响因素，确认关闭 **Flash Attention** 后仍未解决问题。

- **针对 GPU 问题的建议**：有人建议检查 **Nvidia 驱动的具体版本** 以及 **libcuda** 是否存在。用户被要求从终端运行 **appimage** 以检查相关的错误消息，并考虑在相应的频道中继续进行故障排除。

- **M1 Mac GPU 加速问题**：一位使用 **M1 16G** 的用户报告称，开启 **GPU** 加速有时会导致加载失败，而使用开启或关闭 **GPU** 加速的 **llama.cpp server** 时，其模型响应速度明显快于 **LM Studio**。
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1253198243588870175)** (4 条消息): 

- **7900xt 出现 GPU 检测问题**：一位用户报告其 **7900xt GPU** 未被检测到。另一名成员询问了他们的 **OS** 以及是否安装了 **ROCm** 软件包。
- **用户寻求 ROCm 软件包安装帮助**：该用户确认他们需要安装 **ROCm 软件包**。随后他们请求关于如何安装的指导，随后获得了一个包含进一步说明的链接。

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1253361907327307858)** (2 条消息): 

- **Claude 3.5 Sonnet 以极速发布**：[Claude 3.5 Sonnet](https://openrouter.ai/models/anthropic/claude-3.5-sonnet) 的性能超越了 Anthropic 最大的模型 Opus，但价格便宜 5 倍，速度快 2.5 倍。它提供标准版和自我审查（self-moderated）两种变体；更多信息请查看[此处](https://x.com/OpenRouterAI/status/1803802819708739717)。
- **Stripe 支付问题已解决**：由于一个未知问题，Stripe 支付最初是将积分排队，而不是直接添加到用户账户。该问题已完全修复，过去 30 分钟内所有待处理的付款现已处理完毕。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/anthropic/claude-3.5-sonnet">Anthropic: Claude 3.5 Sonnet by anthropic</a>：Claude 3.5 Sonnet 提供了优于 Opus 的能力、快于 Sonnet 的速度，且价格与 Sonnet 持平。Sonnet 特别擅长：- 编程：自主编写、编辑和运行代码...</li><li><a href="https://openrouter.ai/models/anthropic/claude-3.5-sonnet:beta">Anthropic: Claude 3.5 Sonnet (beta) by anthropic</a>：这是与 Anthropic 合作提供的 [Claude 3.5 Sonnet](/models/anthropic/claude-3.5-sonnet) 的低延迟版本，具有自我审查功能：响应审核发生在...</li><li><a href="https://x.com/OpenRouterAI/status/1803802819708739717">来自 OpenRouter (@OpenRouterAI) 的推文</a>：Claude 3.5 Sonnet 现已上线！它的性能超越了 Anthropic 最大的模型 Opus，但价格便宜 5 倍，速度快 2.5 倍 🔥 引用 Leon Builds Agents (@leonjcoe) 的话：始终能够轻松访问...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1253068643260960923)** (93 条消息 🔥🔥): 

- **Nemotron 未被广泛托管**：讨论透露，由于 NVIDIA 的 NeMo 格式与主流推理引擎不兼容，且 340B 的体量巨大，Nemotron 未被许多服务商托管。一位成员指出：“大多数供应商都不愿在没有‘不会失败’的保证下托管如此庞大的模型。”
 
- **Dolphin Mixtral 1x22b 获得好评**：一位成员认为，在 [HuggingFace](https://huggingface.co/cognitivecomputations/dolphin-2.9.1-mixtral-1x22b) 上发现的 Dolphin Mixtral 1x22b 值得更多关注。他们强调了其潜力，认为它能够“挑战甚至可能完全取代 Codestral，且没有那些限制性许可的烂事”。

- **OpenRouter 网站混淆已解决**：有用户报告 OpenRouter 网站宕机，但在用户重启电脑并表示“现在似乎可以工作了，一切正常”后，确定这是与 Safari 浏览器相关的问题。

- **Sonnet 3.5 引发热议**：关于 @AnthropicAI 发布 Claude 3.5 Sonnet 的讨论提到了其极具竞争力的定价，“每百万 input tokens 3 美元，每百万 output tokens 15 美元”。一位成员评价了这一积极影响：“如果我没看错的话，价格仍然低于 Opus。”

- **DeepSeek-Coder V2 上下文冲突**：关于 DeepSeek-Coder V2 实际上下文长度的查询揭示了差异；尽管模型卡片标注为 128K，但 OpenRouter 的描述将其限制在 32K，正如一位成员澄清的那样：“它是 32k，被供应商限制了。”

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/de">de (li)</a>：未找到描述</li><li><a href="https://www.anthropic.com/pricing#anthropic-api">定价</a>：Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释且可控的 AI 系统。</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.9.1-mixtral-1x22b">cognitivecomputations/dolphin-2.9.1-mixtral-1x22b · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/alexalbert__/status/1803790943633686589/photo/1">来自 Alex Albert (@alexalbert__) 的推文</a>：Claude 3.5 Sonnet 现已向各地的 @AnthropicAI 开发者开放。它是我们迄今为止最好的模型——比 Claude 3 Opus 更聪明，速度快两倍。而且每百万 input tokens 仅需 3 美元，每百万...</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-coder/api>">DeepSeek-Coder-V2 – 使用标准化 API 运行</a>：DeepSeek-Coder-V2，一个开源的混合专家（MoE）代码语言模型。它是在 DeepSeek-V2 的中间检查点基础上，通过额外的 6 万亿 tokens 进一步预训练而成的。原始...
</li>
</ul>

</div>

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1253061821166784552)** (46 条消息🔥): 

- **训练 1B 模型用于网络争论**：成员们讨论了训练一个 1B 模型来解决网络争论的实用性。有人指出，“在 H100 节点上只需不到两天……训练模型似乎是回答该问题最简单、最直接的方法，”而另一个人则反驳说，为了争论而付出的成本太高了。
  
- **Selectolax 和 Lexbor 的挑战**：一位用户分享了在将代码转换为使用 Selectolax 和 Lexbor 后端后遇到的多个问题，导致了大量的段错误 (segmentation faults)。问题包括难以查询 HTML 注释以及处理空的 HTML 文档。

- **Warc 处理流水线的性能**：多位用户比较了使用不同流水线处理 CC Warc 文件的耗时。一位用户报告称，“处理 1 个 Warc 大约需要……在 32 个进程中并行完成，”而另一位用户则优化了其方法，使用 100 个进程在 60 秒内处理完一个 Warc。

- **Epoch AI Data Hub 更新**：Epoch AI 宣布了其 Data Hub 的新迭代，其中包含超过 800 个模型的数据。该公告包含了指向其更新后的仓库链接，并强调了其对研究人员、政策制定者和利益相关者的实用性。

- **开发非前沿 LLM 的成本**：讨论重点展示了一张显示非前沿 LLM 开发成本大幅下降的图表。一位用户链接了来自 CNAS 的一份报告，讨论了前沿 AI 开发的未来，该报告预测到 2030 年代计算量将显著增长。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://epochai.org/data">Data on the Trajectory of AI</a>：我们的公共数据库编目了超过 1300 个机器学习模型。探索展示从 1950 年至今 AI 增长和轨迹的数据与图表。</li><li><a href="https://arxiv.org/abs/2404.07647">Why do small language models underperform? Studying Language Model Saturation via the Softmax Bottleneck</a>：语言建模的最新进展在于在极大的网络挖掘文本语料库上预训练高度参数化的神经网络。此类模型的训练和推理在实际中可能成本高昂……</li><li><a href="https://www.cnas.org/publications/reports/future-proofing-frontier-ai-regulation">Future-Proofing Frontier AI Regulation</a>：制定强大、务实且有原则的国家安全和国防政策。
</li>
</ul>

</div>

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1253064515944185988)** (21 条消息🔥): 

- **模型合并与 Token 使用讨论**：成员们讨论了 [DCLM-Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)，重点介绍了其通过分类器驱动的清洗和过滤制成的 4T Token 数据集。会议提出了关于数据细节、使用情况及其对性能影响的问题，包括 Llama2、DeepSeek、Mistral-0.3 以及其他 7B 级模型表现。

- **关于序列建模技术的 SlotSSM 论文**：分享了一篇介绍 SlotSSMs 的[论文](https://arxiv.org/abs/2406.12272)，重点关注用于模块化序列建模的 SSMs。对话探讨了将状态维护为多个向量（“Slots”），并通过 Self-Attention 进行稀疏交互，如何改善视频预测和 3D 视觉推理等任务。

- **专业数据提升 Benchmark 表现的证据**：一条推文强调了来自 [@shan23chen](https://x.com/shan23chen/status/1803459255518769509?s=46) 的 GSM1K 数据集，并链接到一篇记录生物医学 Benchmark 中药物名称识别问题的[论文](http://arxiv.org/abs/2406.12066)。研究表明，由于公共数据集中潜在的数据污染，模型在处理品牌名称时表现较差，这影响了实际的医疗应用。

- **LLM 训练后改进技术**：通过一篇[论文](https://arxiv.org/abs/2312.13558)探讨了一种名为层选择性秩削减（LAyer-SElective Rank reduction, LASER）的技术。该方法涉及在训练后移除权重矩阵的高阶分量，以在不增加额外参数或数据的情况下增强模型性能。

- **表面形式竞争（Surface Form Competition）的替代评分函数**：提到了另一篇[论文](https://arxiv.org/abs/2104.08315)，提出了领域条件逐点互信息（Domain Conditional Pointwise Mutual Information）来解决 LLM 中的表面形式竞争问题。该概念建议根据选项的先验似然重新计算权重，以提高 Zero-shot 任务的性能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.12272">Slot State Space Models</a>：最近的状态空间模型（SSMs）如 S4、S5 和 Mamba 在长程时间依赖建模中显示出显著的计算优势。然而，在许多序列建模问题中，未...</li><li><a href="https://arxiv.org/abs/2104.08315">Surface Form Competition: Why the Highest Probability Answer Isn&#39;t Always Right</a>：大型语言模型在 Zero-shot 设置中表现出了可喜的结果（Brown 等，2020；Radford 等，2019）。例如，它们只需通过对问题进行条件化即可执行多项选择任务...</li><li><a href="https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0">mlfoundations/dclm-baseline-1.0 · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://x.com/shan23chen/status/1803459255518769509?s=46">Shan Chen (@shan23chen) 的推文</a>：💊 我们把你的语言模型带到了药店……它对对乙酰氨基酚（通用名）的了解比泰诺（品牌名）更好！@hughbzhang @scale_AI 上个月开发了 GSM1K，他们发现许多...</li><li><a href="https://arxiv.org/abs/2312.13558">The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction</a>：基于 Transformer 的大型语言模型（LLMs）已成为现代机器学习的固定组成部分。相应地，大量资源被投入到旨在进一步推进这一领域的研究中...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 条消息): 

arthur0511: 很确定一堆旧的 BERT 论文都这样做过，例如 https://arxiv.org/pdf/2101.04547
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1253385667598352555)** (7 条消息): 

- **LM-Eval 与 NumPy 版本的兼容性问题**：一位成员在运行 `lm-eval-overview.ipynb` 文件时遇到问题，错误提示指向与 **NumPy 2.0** 不兼容。他们提到尝试通过 `pip install "numpy<2.0"` 降级 NumPy，但问题依然存在。
- **错误诊断与建议的解决方法**：另一位成员建议在最新的 master 分支上尝试，并展示错误输出的底部内容。该问题似乎源于使用 **NumPy 1.x** 编译的模块无法与 **NumPy 2.0** 协同工作。
- **以 Colab 作为替代方案**：有建议在 Google Colab 中运行该任务，一名用户成功运行了 `lm_eval -h`。然而，由于远程代码执行（remote code execution）问题，在 Colab 中运行 `demo_boolq` 示例时仍然存在问题。
- **分支命名差异**：对分支名称进行了澄清；应该使用的正确分支是 `main` 而不是 `master`。这种混淆引起了困惑，并被指出会影响任务的运行。
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 条消息): 

stellaathena: https://x.com/stasbekman/status/1803653883350360372?s=46

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1253150792442183848)** (62 messages🔥🔥): 

- **初创公司缩写听起来很有趣**：一位成员幽默地评论了一些初创公司的缩写听起来像疾病。
- **Claude 3.5 Sonnet 发布**：[Anthropic](https://x.com/anthropicai/status/1803790676988920098?s=46) 宣布发布 Claude 3.5 Sonnet，强调了其相比前代模型在速度、成本效益和性能方面的提升。公告还提到了 3.5 模型家族的未来版本，包括 Haiku 和 Opus。
- **Character.AI 致力于高效推理**：[Character.AI](https://research.character.ai/optimizing-inference/) 强调了他们通过优化推理以处理每秒超过 20,000 次查询，从而致力于构建 AGI 的努力。这一处理量大约是 Google Search 请求量的 20%。
- **青少年对 Character.AI 的参与度很高**：成员们讨论了 Character.AI 的流行程度和高参与率，特别是在年轻用户中，并指出其平均会话时间远超 ChatGPT。
- **编程基准测试青睐 Claude 3.5**：Claude 3.5 Sonnet 在 aider 的代码编辑排行榜上[排名第一](https://x.com/paulgauthier/status/1803813637556945201?s=46&t=_jodDCDeIUnWb_Td0294bw)，表明其在“whole”和“diff”代码编辑格式方面表现强劲。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://research.character.ai/optimizing-inference/">在 Character.AI 优化 AI 推理</a>：在 Character.AI，我们正致力于构建 AGI。在未来的状态下，大语言模型 (LLMs) 将增强日常生活，提供商业生产力和娱乐，并帮助人们...</li><li><a href="https://x.com/sam_kantor/status/1803783127195677013?s=46">来自 Sam (@Sam_Kantor) 的推文</a>：@AnthropicAI Claude 3.5 Sonnet 发布了</li><li><a href="https://x.com/anthropicai/status/1803790691199336484?s=46">来自 Anthropic (@AnthropicAI) 的推文</a>：为了完善 Claude 3.5 模型家族，我们将在今年晚些时候发布 Claude 3.5 Haiku 和 Claude 3.5 Opus。此外，我们正在为企业开发新的模态和功能，以及...</li><li><a href="https://x.com/anthropicai/status/1803774865473696237?s=46">来自 Anthropic (@AnthropicAI) 的推文</a>：Fc zbvx ts temxnsq nx mzog jlbuv gusn zofg hhfs: Ebwxk vnii mzceaw tfr fvpowf sbyglovaw fmr, Nyp fgrryw xjf, lrx 'mosra xaw huvvw sbq ssnjhu'f vnf, Che rxo qeremaca ophgaf, n abkse oyw. KEY:</li><li><a href="https://x.com/anthropicai/status/1803790676988920098?s=46">来自 Anthropic (@AnthropicAI) 的推文</a>：介绍 Claude 3.5 Sonnet——我们迄今为止最智能的模型。这是我们 3.5 模型家族的首个发布版本。Sonnet 现在在关键评估中超越了竞争对手模型，速度是 Claude 3...</li><li><a href="https://x.com/TonyWangIV/status/1803510231332536564">来自 Tony Wang (@TonyWangIV) 的推文</a>：大约 18 个月前，我和我的合作者发现，所谓的超人类围棋 AI 可以被人类使用简单的对抗性策略击败。从那时起，我们一直在测试...</li><li><a href="https://x.com/testingcatalog/status/1803566884991766640?s=46">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：突发新闻 X 和 Midjouney 似乎已经达成了合作伙伴关系 👀 Grok 未来可能能够使用 Midjourney 进行图像生成。引用 DogeDesigner (@cb_doge) 突发新闻：...</li><li><a href="https://x.com/alexalbert__/status/1803790943633686589">来自 Alex Albert (@alexalbert__) 的推文</a>：Claude 3.5 Sonnet 现已面向各地的 @AnthropicAI 开发者开放。这是我们迄今为止最好的模型——比 Claude 3 Opus 更聪明，速度快两倍。而且每百万 input tokens 仅需 3 美元，每百万 output tokens 仅需 15 美元...</li><li><a href="https://x.com/paulgauthier/status/1803813637556945201?s=46&t=_jodDCDeIUnWb_Td0294bw">来自 Paul Gauthier (@paulgauthier) 的推文</a>：Claude 3.5 Sonnet 现在是 aider 代码编辑排行榜上排名最高的模型！DeepSeek Coder V2 仅在 4 天前占据了第一名。Sonnet 在 “whole” 编辑格式中排名第一。它还得分...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1253150948197666866)** (8 messages🔥): 

- **AI Safety 的信任度受到质疑**：成员们讨论了在 AI Safety 背景下信任的重要性，探讨了 **trust and safety** 的定义和作用。一位成员讽刺地评论道：“相信我，兄弟”，凸显了围绕这一问题的怀疑态度。
- **Eliezer Yudkowsky 挑战对齐计划**：链接指向 [Eliezer Yudkowsky 的推文](https://x.com/ESYudkowsky/status/1803676608320192617)，他在文中点名批评了典型的对齐计划，称：“如果你有一个我无法在 120 秒内驳倒的对齐计划，那就说来听听。”这引发了一场嘲讽过去和当前安全尝试严肃性的对话。
- **Scott Aaronson 和 Ilya Sutskever 的对齐理论**：一位成员回忆起 Scott Aaronson 曾提到 **Ilya Sutskever** 正在寻找通过复杂理论表达的对齐方式。这一评论与对 AI Safety 概念性方法的更广泛探索相呼应。

**Link mentioned**: <a href="https://x.com/ESYudkowsky/status/1803676608320192617">Tweet from Eliezer Yudkowsky ⏹️ (@ESYudkowsky)</a>: @ssi If you have an alignment plan I can&#39;t shoot down in 120 seconds, let&#39;s hear it.  So far you have not said anything different from the previous packs of disaster monkeys who all said exact...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1253249442379989002)** (3 messages): 

- **快手凭借视频 AI 模型超越 OpenAI**：中国公司 [快手](https://kling.kuaishou.com/en) 推出了首个向公众免费开放的文本生成视频生成式 AI 模型。该工具名为 **Kling**，可以生成长达两分钟、帧率为 30fps、分辨率高达 1080p 的视频，而 OpenAI 的 **Sora** 在测试数月后仍未向公众开放。
- **询问 Meta 使用合成数据的情况**：Nathan Lambert 征求有关 Meta 使用 5000 块 V100 生产合成数据的参考资料或链接。他提到自己正在再次撰写关于合成数据的想法，并寻求这些信息作为背景。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/natolambert/status/1803844567269281896">Tweet from Nathan Lambert (@natolambert)</a>: Does anyone have a link or reference to the job information regarding meta using 5000 v100s for &#34;synthetic data&#34;?  Writing some thoughts on synth again :)</li><li><a href="https://www.technologyreview.com/2024/06/19/1094027/kling-kuaishou-video-ai-china/">I tested out a buzzy new text-to-video AI model from China</a>: Kuaishou’s generative video model Kling, which could be poised to transform how short clips are created for platforms like TikTok.
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1253377985076138137)** (2 messages): 

- **CrewAI 和 LlamaIndex 增强多 Agent 系统**：CrewAI 提供了一个直观的框架来定义具有不同角色的 Agent “团队（crew）”来解决任务。现在，这些 Agent 可以轻松地通过 LlamaIndex 的功能进行增强。[阅读更多](https://t.co/8Tjk888RL1)。 
- **创始人将在 AI Engineer's World's Fair 演讲**：下周将在 @aiDotEngineer 的 World's Fair 上看到我们的创始人 @jerryjliu0 的两次露面。他将于 6 月 26 日讨论 *Future of Knowledge Assistants*，并发布一些特别公告，随后在 6 月 27 日进行另一场分享。[了解更多](https://t.co/JMoAOAA4bI)。
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1253068638458478625)** (61 条消息🔥🔥): 

- **自定义 Similarity Scores**：一位成员询问是否可以在 Vector Store 中定义自定义 Similarity Score。回复指出 LlamaIndex 并不显式支持此功能，建议用户在需要时实现自己的方法。 
- **添加带有顺序标识符的 Nodes**：一位成员寻求关于向 VectorStoreIndex 添加带有顺序标识符的新 Nodes 的建议。解决方案涉及在插入前手动管理标识符，并提供了一个代码示例来说明操作方法。
- **从 PDFs 生成问题**：针对从多个 PDFs 生成问题的查询，通过使用 LlamaIndex 的 `DatasetGenerator` 示例进行了回答。该示例演示了如何使用 OpenAI 模型设置生成器以创建问题。
- **持久化 DocumentSummaryIndex**：一位成员询问关于在 Vector Store 中存储 DocumentSummaryIndex 的问题，建议使用 `storage_context.persist()` 方法。详细的代码示例展示了如何持久化 Index。
- **异步检索 Nodes**：有一个关于如何从 PGVector Docstore 异步获取所有 Nodes 的问题。回复提到了使用 `aget_nodes`，但指出在没有现有列表的情况下，缺乏检索所有 Node IDs 的具体信息。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/examples/llm/ollama/">Ollama - Llama 3 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/storing/storing/#inserting-documents-or-nodes>))">Storing - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/finetuning/llm_judge/pairwise/finetune_llm_judge/#use-a-datasetgenerator-to-build-train_dataset-and-test_dataset>).">Knowledge Distillation For Fine-Tuning A GPT-3.5 Judge (Pairwise) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/#llama_index.core.vector_stores.types.BasePydanticVectorStore.aget_nodes>)">Index - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/docstore/#llama_index.core.storage.docstore.types.BaseDocumentStore.aget_nodes>)">Index - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/jaguar/#llama_index.vector_stores.jaguar.JaguarVectorStore.similarity_search_with_score>)).">Jaguar - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/weaviate/#llama_index.vector_stores.weaviate.WeaviateVectorStore>)).">Weaviate - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/rocksetdb/#llama_index.vector_stores.rocksetdb.RocksetVectorStore>),">Rocksetdb - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/singlestoredb/#llama_index.vector_stores.singlestoredb.SingleStoreVectorStore.query>)).">Singlestoredb - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1253089845232795668)** (42 messages🔥): 

- **Nemotrons API 显示速度提升**：一位成员宣布 **Nemotrons API** 现在快了很多，并提到 **Reward Model** 已经发布。
  
- **Turbcat 模型和数据集配置讨论**：关于 **Turbcat** 是指组织还是个人存在困惑。澄清了 **Turbca** 是人名，而 **Turbcat** 是模型名称。还对正在使用的数据集配置和 Tokenization 方法提出了担忧。

- **Tokenization 和 Sample Packing 辩论**：随后进行了关于 Tokenization 过程的详细讨论，特别是关于如何正确处理 **EOT (End of Text)** token。成员们辩论了 Sample Packing 中上下文分离的潜在问题以及 Attention Masks 的正确用法。

- **Flash Attention 和 Multipack 可视化**：一位成员提供了 [Multipack with Flash Attention 文档](https://openaccess-ai-collective.github.io/axolotl/docs/multipack.html) 的链接，以说明在训练期间应如何拼接和处理样本。

- **Qwen 模型的偏见和所需调整**：针对 **Qwen 模型** 需要去审查和去宣传化提出了担忧。讨论了偏见问题，特别是与 CCP 观点一致的部分，并引用了 Hugging Face 上关于 [中国 LLM 审查分析](https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis) 的文章。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/docs/multipack.html">Axolotl - Multipack (Sample Packing)</a>：未找到描述</li><li><a href="https://huggingface.co/bl">bl (BLIAN)</a>：未找到描述</li><li><a href="https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis">An Analysis of Chinese LLM Censorship and Bias with Qwen 2 Instruct</a>：未找到描述</li><li><a href="https://huggingface.co/turboderp/llama3-turbcat-instruct-8b">turboderp/llama3-turbcat-instruct-8b · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1253248749439029258)** (7 messages): 

- **关于使用 QLoRA 进行持续预训练 (Continuous Pretraining) 的辩论**：一位成员质疑使用 QLoRA 进行持续预训练的效果，表示：“我一直认为这只有作为全参数微调 (Full Finetune) 才有意义？”其他人指出这对于保留知识没有太大帮助。
- **层剪枝 (Layer-pruning) 和 QLoRA 展现出前景**：另一位成员分享了一项关于 [层剪枝 (Layer-pruning)](https://arxiv.org/abs/2403.17887) 及其极小性能退化的研究，详细介绍了他们使用 QLoRA “修复”剪枝模型并将 MMLU 分数提高 10 分的成功经验。该技术结合了 QLoRA 和剪枝策略以优化计算资源。
- **关于 QLoRA 和 Llama 3 的资源**：为了解更多细节，一位成员引用了一个 [Hugging Face 模型](https://huggingface.co/chargoddard/llama3-42b-v0)，解释了剪枝参数和所述方法论。这个详细的模型卡片是实现 QLoRA 和 PruneMe 的实际案例。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.17887">The Unreasonable Ineffectiveness of the Deeper Layers</a>：我们对流行的开源预训练 LLM 家族进行了一种简单的层剪枝策略的实证研究，发现直到……在不同的问答基准测试中性能几乎没有下降。</li><li><a href="https://huggingface.co/chargoddard/llama3-42b-v0">chargoddard/llama3-42b-v0 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/)** (1 messages): 

nanobitz: 我认为 Teknium 可能也收集了其中一些。

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1253085330563928134)** (27 条消息🔥): 

- **通过将反引号切换为单引号修复 Bug**：一位成员发现，在向 SystemMessage 注入数据时，使用反引号（backticks）而非单引号（single quotes）会导致问题。更改为单引号后解决了该问题。
- **处理网页抓取的大文本数据**：一位成员寻求关于网页抓取数据结构化的建议，提到了 token 限制以及合并分块响应的需求。分享了 [LangChain 文档](https://github.com/langchain-ai/langchain/issues/17783) 和数据拆分策略。
- **PDF 向量数据库问题**：一位成员在使用 PDF 文档时遇到了从向量数据库（vector database）检索响应的困难，反复出现“我不知道”的回答。
- **对 Streamlit 与 LangServe 集成的兴趣**：一位成员询问了使用 Streamlit 配合 LangServe 为其 LangGraph 聊天机器人部署 Web 应用的可行性。
- **流式传输期间的事件过滤**：一位成员询问如何在 `astream_event` 期间从特定的 LLMChain 获取响应。提供了详细的解答和示例，并参考了 [LangChain 文档](https://python.langchain.com/v0.2/docs/how_to/streaming/#filtering-events)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/langchain-ai/langchain/issues/17783>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/streaming/#by-type>).">如何进行流式传输 | 🦜️🔗 Langchain</a>: 本指南假设您熟悉以下概念：</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/streaming/#filtering-events>).">如何对 runnable 进行流式传输 | 🦜️🔗 LangChain</a>: 本指南假设您熟悉以下概念：</li><li><a href="https://www.startupweekendsf.com/">Techstars Startup Weekend</a>: Techstars Startup Weekend 是一个为期 3 天的动态加速器项目，您可以在其中开发、原型设计、设计并验证您的创业想法。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1253417002526179438)** (9 条消息🔥): 

- **难以将 LangGraph 与 Chat Playground 集成**：一位用户询问如何使用 Chat Playground 配合 **LangGraph** 来测试其对话 Agent。他们得到了关于输入模式（input schema）和 runnable 格式要求的建议，但提到 AI 提供的回复并未专门针对 LangGraph 层面进行解答。
- **LangServe 文档与持久化示例**：聊天机器人澄清说，虽然最初的回复未包含 LangGraph，但可以通过向 LangGraph Agent 传递 checkpointer 来实现持久化。提供了一个使用 `SqliteSaver` 的示例，并引导用户参考 [LangChain Python 文档](https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/#agent-constructor) 以获取详细指导。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/langserve/#chat-playground>).">🦜️🏓 LangServe | 🦜️🔗 LangChain</a>: 发行说明</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/#agent-constructor>).">对话式 RAG | 🦜️🔗 LangChain</a>: 本指南假设您熟悉以下概念：
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1253070766539542598)** (5 条消息): 

- **Manifold Research 分享双周更新**：Manifold Research Group 发布了最新的 [Research Log #040](https://www.manifoldrg.com/research-log-040/)，重点介绍了其名为 MultiNet 的全模态预训练语料库的进展。他们邀请感兴趣的人加入其 [Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com) 上的研究讨论，并查看其 [GitHub](https://github.com/ManifoldRG?ref=manifoldrg.com)。

- **TVFoodMaps 添加 AI Concierge**：TVFoodMaps 推出了 [个人美食管家](https://www.tvfoodmaps.com/foodtv-ai-chat)，帮助用户发现并计划访问电视节目中介绍的餐厅。点击 [此处](https://tvf-images.s3.amazonaws.com/prod/tvf-ai.mp4) 查看视频演示，该功能需要高级会员资格。

- **关于使用 OpenAI 和 LangChain 创建 SQL Agent 的指南**：一位用户分享了关于创建 SQL Agent 的 [指南](https://git.new/SQLAgent)，该 Agent 可以绘制图表并使用 OpenAI 和 LangChain 查询数据库。他们欢迎对其作品提出反馈。

- **构建对话式时光机**：Medium 上的一篇新文章 [Building a Conversational Time Machine](https://medium.com/ai-advances/building-a-conversational-time-machine-a-langgraph-support-chatbot-745b2b08c587) 介绍了 LangGraph 支持聊天机器人。作者是 Ankush K. Singal，文章讨论了该聊天机器人的开发和潜在应用。

- **关于检索增强的新文章**：发布了一篇题为 [Retrieval augmentation with MLX: A bag full of RAG, part 2](https://github.com/uogbuji/mlx-notes/blob/main/2024/rag-basics2.md) 的文章，涵盖了使用 Apple MLX 机器学习框架的笔记。该帖子包含了对检索增强生成 (RAG) 技术的详细见解。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.manifoldrg.com/research-log-040/">Research Log #040</a>: 欢迎来到研究日志 #040！我们记录了 Manifold Research Group 各项计划的每周研究进展，并重点介绍了我们认为更广泛的研究社区取得的突破...</li><li><a href="https://www.tvfoodmaps.com/foodtv-ai-chat">Restaurant Planner: AI Concierge to Find Restaurants on TV</a>: 您的私人助手将帮助您找到在 Diners, Drive-Ins and Dives 等热门美食电视餐厅节目中看到的所有餐厅。</li><li><a href="https://git.new/SQLAgent">Step by step guide to create a SQL Agent</a>: 这是一份关于如何创建 SQL Agent 以执行 SQL 查询并记录它们的指南。</li><li><a href="https://github.com/uogbuji/mlx-notes/blob/main/2024/rag-basics2.md">mlx-notes/2024/rag-basics2.md at main · uogbuji/mlx-notes</a>: 在使用 Apple MLX 机器学习框架时创建的共享个人笔记 - uogbuji/mlx-notes</li><li><a href="https://medium.com/ai-advances/building-a-conversational-time-machine-a-langgraph-support-chatbot-745b2b08c587">Building a Conversational Time Machine: A LangGraph Support Chatbot</a>: Ankush k Singal
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1253311700313178202)** (1 条消息): 

- **使用 LangChain 和 OpenAI 创建 SQL Agent**：一位成员分享了 [如何创建 SQL Agent 的指南](https://git.new/SQLAgent)，该 Agent 能够执行 SQL 查询、绘制图表并记录结果。他们请求对其创作提供反馈。

**提及的链接**：<a href="https://git.new/SQLAgent">Step by step guide to create a SQL Agent</a>: 这是一份关于如何创建 SQL Agent 以执行 SQL 查询并记录它们的指南。

  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1253232081736044604)** (10 条消息🔥): 

- **讨论 Taylor 近似实现的 Bounty**：一名成员开始处理涉及 `function.py` 中 LOG2、EXP2 和 SIN 的 Taylor 近似 Bounty。他们询问，鉴于社区专注于减少操作数量，向 `ops.py` 添加新的位操作（bitwise operations）是否可以接受。

- **多 GPU 支持说明**：一名成员询问了多 GPU 支持，特别是使用 NVLink 时是否可以使用两个以上的 GPU，并得到了 GPU 是通过 PCI-E 连接的澄清。讨论中包含了一个 [NVIDIA Linux open GPU with P2P support 的 GitHub 链接](https://github.com/tinygrad/open-gpu-kernel-modules)。

- **Diffusion 模型贡献**：一名成员将一个简单的 Diffusion 模型从 PyTorch 移植到了 tinygrad，并询问是否可以将其作为示例添加。George Hotz 强调代码质量必须非常高才能被包含，并建议在准备好后提交 PR。

- **新贡献者的正确性优先**：George Hotz 强调，对于处理近似值的新贡献者，在关注速度之前，正确性是首要任务。他还提到目前还没有人让所有测试都通过。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/Seachaos/Tree.Rocks/blob/main/QuickDiffusionModel/QuickDiffusionModel_torch.ipynb">Tree.Rocks/QuickDiffusionModel/QuickDiffusionModel_torch.ipynb at main · Seachaos/Tree.Rocks</a>：通过在 GitHub 上创建账号为 Seachaos/Tree.Rocks 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: NVIDIA Linux open GPU with P2P support</a>：支持 P2P 的 NVIDIA Linux 开源 GPU 驱动。通过在 GitHub 上创建账号为 tinygrad/open-gpu-kernel-modules 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1253073416542490695)** (30 条消息🔥): 

- **为什么 buffer 会在 optimizer 中被 realize？**：讨论了在 buffer 未更新时，是否有必要在 optimizer 中对其进行 realize。一位用户解释说 *“如果它们没有改变，realize 不会执行任何操作”*。
- **对 'extra' 模块的困惑**：一位用户询问某些示例所需的 'extra' 模块，其他人澄清这是 tinygrad 仓库中的 'extra' 目录。建议在命令中添加 `PYTHONPATH=.` 以解决问题。
- **Ubuntu 依赖项安装工具**：当被问及 Ubuntu 24.04 配合 Python venv 的更好设置时，建议使用 `SETUPTOOLS_ENABLE_FEATURES="legacy-editable" pip install -e .` 来处理依赖项。
- **在 TinyGrad 中实现 `clip_grad_norm_`**：围绕优化和修正 TinyGrad 中 `clip_grad_norm_` 的实现展开了长时间讨论，特别是在 Metal 上。解决了与 Metal 限制相关的问题，并建议将 tensor 分块（chunking tensors）作为临时解决方案。
- **权重共享（Weight tying）Bug**：识别并调查了 TinyGrad 中一个与 embedding 层和输出 logit 层之间权重共享相关的潜在 Bug。两个 tensor 似乎被分别优化了，这表明需要对库进行修正。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1253159944509853782)** (10 messages🔥): 

- **LLM-Finetuning：课程结束后的服务器活跃度**：一位成员询问课程结束后 Discord 服务器是否会继续保持活跃，另一位成员回答说，社区的生命力取决于成员和管理员。
  
- **Vanishing Gradients 专家直播**：宣布了一场由来自 Amazon 的 Eugene Yan 和来自 Hex 的 Bryan Bischof 等专家参加的直播。他们将讨论现实世界 LLM 应用中的经验教训，涵盖 Prompt Engineering、评估和工作流优化等主题。[在此注册](https://lu.ma/e8huz3s6?utm_source=ds)并阅读他们的 [O'Reilly 报告](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/)。

- **优化 LLM 的 Time-to-First-Token**：关于优化首字延迟（Time-to-First-Token）的讨论提出了多项建议，例如为 Prefill 和 Decoding 使用不同的 GPU，针对特定任务进行 Fine-tuning，以及尝试使用 Base 模型而非 Instruct 模型。讨论中分享了一篇[相关论文](https://arxiv.org/pdf/2401.09670v3)。

- **Maven 上的 Zoom 录像**：有人提出了关于 Maven 上 Zoom 课程录像保存期限的问题。确认录像将永久保存。

- **为金融领域项目寻找 RAG 专家**：一家专注于金融领域 LLM 的初创公司急需一名 RAG 专家，参与为期一周的 AI 聊天机器人优化项目。关键技术栈包括 MySQL, Langchain, ChatGPT API, Docker 和 Pinecone。

**提到的链接**：<a href="https://lu.ma/e8huz3s6?utm_source=ds">LESSONS FROM A YEAR OF BUILDING WITH LLMS · Luma</a>：在这次 Vanishing Gradients 特别直播录制中，Hugo 与 Eugene Yan (Amazon), Bryan Bischof (Hex), Charles Frye (Modal), Hamel Husain 等人进行了对话。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1253065398086275104)** (1 messages): 

- **欺诈检测系统的微调至关重要**：作业建议，为**独特的金融机构创建欺诈检测系统**需要进行 Fine-tuning，以处理特定的交易模式和欺诈指标。通用模型不足以胜任此任务。
- **通用翻译模型已足够**：作业结论认为，**通用语言翻译服务**不需要 Fine-tuning。这是因为通用翻译模型已经能够充分处理各种语言和语境。
- **小众产品推荐需要微调**：为稀有收藏品等**极小众产品构建推荐系统**需要进行 Fine-tuning。它必须理解该领域特有的用户偏好和产品属性。
- **通用新闻摘要无需微调**：对于**通用新闻摘要工具**，作业提到 Fine-tuning 是不必要的。通用语言模型可以有效地管理新闻摘要任务。
- **专业技术支持聊天机器人需要微调**：作业指出，**担任高度专业化技术支持角色的聊天机器人**需要进行 Fine-tuning。这对于确保机器人拥有特定技术领域的详细知识是必要的。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1253338703309570158)** (1 messages): 

- **Modal 微调快速入门**：一位成员分享了一篇[博客文章](https://gkopendev.github.io/2024/06/19/llm-finetune.html)，作为 Modal 微调示例的快速入门指南。他们鼓励社区尝试 **Modal UX**。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1253095837173813400)** (3 messages): 

- **用户请求 Jarvis Lab 提供 Docker 镜像选项**：一位用户称赞了 Jarvis Lab 在 Fine-tuning 方面的直观性，但建议增加提供自定义 Docker 镜像的选项以提高效率。他们指出目前的设置大约需要 45 分钟，而 Fine-tuning 运行仅需约 20 分钟。
- **Docker 支持即将推出**：Jarvis 团队确认，使用个人 Docker 镜像的选项将很快推出。该功能在早期版本中曾提供，并将被重新引入。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1253307683348676658)** (1 messages): 

- **关于 Argilla 使用的说明**：一位成员询问，在按照 *Mixture of Agents and Juries as Judge* 方法使用积分生成合成数据时，是否需要为每个 **LLM** 创建专用端点。目前没有进一步的讨论或链接来详细阐述这个问题。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1253076789211697192)** (9 messages🔥): 

- **LangSmith 额度已发放：Swaroopch**：一名用户为 "Mastering LLMs Course Credit" 申请了 LangSmith 额度，并提供了他们的邮箱和组织 ID。在确认付款方式后，他们确认收到了额度，并询问了有效期，确认有效期为一年。
- **来自 Shtandon 的新额度申请**：另一位用户 Shtandon 询问如何为其提供的组织 ID 和邮箱获取额度。回复指出该邮箱不在额度名单中，并建议如果他们在截止日期前提交过申请，请发送私信。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[allaire_inspect_ai](https://discord.com/channels/1238365980128706560/1242943547699888229/1253427396179398757)** (1 messages): 

- **Eval 框架获得高度赞赏**：一位成员对该 Eval 框架表示兴奋，称赞其拥有 *"极佳的开发者体验"*。他们强调了其 *"直观的 API 设计和编写良好的代码"*，使得为 LLM 使用代理端点（如自定义企业级 Base URL）变得非常容易。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1253152138855710771)** (3 messages): 

- **成员请求额度帮助**：几位成员正在寻求账户额度方面的帮助。他们在消息中提供了自己的账户 ID：`shubhi194-680421`、`cyzgab-17b4a1` 和 `mnemic-8c53ac`。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1253139895006920754)** (3 messages): 

- **Predibase Serverless 推理问题**：一位成员表示很兴奋尝试在 React 客户端上通过其 **QLora adapter** 使用 **Predibase serverless inference endpoint**。然而，他们遇到了 CORS 策略错误，并已在 Predibase Discord 频道中将其作为功能请求（Feature Request）提交。
- **邮箱注册问题**：另一位成员报告了一个问题，尽管使用了工作 ID 注册，但找不到其注册的邮箱 ID，请求协助解决。
- **解锁额度问题**：一位成员就如何解锁额度寻求建议，表示对该流程存在困惑或遇到了问题。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1253151688425078914)** (1 messages): 

- **查看额度问题**：一位成员报告称他们“看不到任何额度”，并提供了他们的 ID `org-XSyt2Grt41k7glihL6LKhuVP` 以寻求进一步帮助。这表明用户对 OpenAI 平台上的账户额度存在困惑或遇到了技术问题。
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1253068652161536133)** (27 messages🔥): 

- **OI 不能让你变富 5%，但也许能让你变富 100%**：一位用户询问 OpenInterpreter 是否能让他们变富 5%，另一位成员幽默地回答道：“它能让你变富 100% 😉”。该用户坚持认为 5% 就足够了，对话以简单的“那对 5% 也是肯定的”继续。

- **最佳无审核模型讨论**：一位成员询问“最佳无审核模型（Uncensored Model）”，并建议了如“2.8 dolphin 或 mistral 3/31/24”等选项。这引发了用户之间关于不同模型偏好和体验的讨论。

- **Open Interpreter 的长期记忆**：一位成员询问：“有没有获取 OI 长期记忆（Long-Term Memory）的解决方案？”这引起了大家的兴趣，并围绕可能的实现方式展开了讨论，但消息中未提供具体的解决方案。

- **首个 Open Interpreter 演示视频**：分享了一个名为“[open interpreter compatch demo](https://youtu.be/SOKq8RS0pR4)”的 YouTube 视频，展示了 Windows/Linux 集成 UI 和 TTS 的首个演示，该演示通过 Azure 使用 GPT-4。上传者暗示后续会有更多更新。

- **Claude 3.5 Sonnet 对话偏好**：用户讨论了他们使用 Claude 3.5 Sonnet 的体验，认为其优于 GPT-4。一位用户提到：“我不喜欢 GPT-4 和 4o 说话的方式，我觉得很烦。Claude 3.5 Sonnet 要好得多。”

**提到的链接**：<a href="https://youtu.be/SOKq8RS0pR4">open interpreter compatch demo</a>：未找到描述

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1253213263785754707)** (2 messages): 

- **制造更新置顶消息已发布**：一位用户告知，通过点击顶部的 #01 并选择“pins”选项卡，用户可以看到来自 Ben 的制造更新。**首批 1,000 台将于 10 月 31 日至 11 月 30 日之间发货**。
- **查询订单状态**：另一位用户询问是否可以查到他们的订单是否在首批 1,000 台的范围内。
  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1253430366879088650)** (1 条消息): 

- **Open Interpreter 使用 AI 连接 WiFi**：一位成员分享了一个[示例](https://x.com/hellokillian/status/1803868941040914824)，其中一个*完全本地、控制电脑的 AI* 成功读取了写有 WiFi 密码的便签并连接到了网络。这展示了 AI 在轻松管理日常任务方面的实际效用。

**提到的链接**：<a href="https://x.com/hellokillian/status/1803868941040914824">来自 killian (@hellokillian) 的推文</a>：我给一个完全本地、控制电脑的 AI 看了一张写有我 WiFi 密码的便签。它上线了。

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1253119332859252807)** (4 条消息): 

- **GBC10M 数据集发布**：**GBC10M** 数据集已发布并可在 [Hugging Face](https://huggingface.co/datasets/graph-based-captions/GBC10M) 上获取。它是使用基于图（graph-based）的方法对 **CC12M** 进行重新标注（recaptioned）的版本。
- **第一作者致谢**：一位被确认为 GBC10M 数据集第一作者的成员对他们的贡献表示了感谢。
- **致力于放宽许可限制**：团队正在努力获取限制较少的许可证，并计划将数据集移动到 Hugging Face 上的 **Apple organization**，并在 **arXiv** 上发表论文。他们还计划发布代码，但强调代码需要更多时间进行润色和审批。

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1253067054810927205)** (23 条消息🔥): 

- **关于对抗鲁棒性的学术争论**：成员们讨论了 Carlini、Papernot 和 Glaze 作者等知名人物在对抗鲁棒性（adversarial robustness）方面的分歧。有人指出了一起事件，即 Glaze 作者拒绝分享用于扰动预算（perturbation budgets）的可控代码库。
- **VAEs：通道数 vs. 空间维度**：成员们辩论了将 VAE 潜通道（latent channels）从 4 个增加到 16 个的影响。讨论点涉及潜空间（latent space）的复杂性，以及增加像素与增加通道之间的计算差异，其中一人指出全局注意力（global attention）随像素数量呈二次方缩放。
- **LLMs 在问题模式上过拟合**：一位成员解释了他们的手动实验，分析 LLMs 缺乏推理能力是否归因于对可识别问题模式的过拟合。他们注意到 Claude-3.5-Sonnet 在通过这些问题进行推理方面似乎明显优于其他模型。
- **Chameleon 模型的挑战**：尝试训练 Chameleon 模型时遇到了 Embedding 层和 Norm 层中极端的梯度范数（gradient norms）问题。通常的技术（如降低学习率和使用更高精度）均无效，导致梯度在几步后变为 NaN。

  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1253089272169369680)** (18 条消息🔥): 

- **Cohere API 适用于不同语言**：一位用户询问是否有人正在使用 Cohere 在 Mac 上开发 .NET 聊天机器人。另一位成员澄清说 **API 与 OAI 兼容**，可以通过简单的 REST/sockets 在任何语言中使用。他们强调直接使用 REST 并不复杂。
- **欢迎与问候**：几位用户互相打招呼，并对社区表示兴奋。一人表示：“很高兴加入这里的派对，必须说 Cohere 确实在以不同且智能的方式做事。”
- **对设计的赞赏**：用户对界面中紫色的风格化选择表示赞赏。一位成员表示：“讨论紫色这种风格化选择有多酷很重要，总有一天我也会利用这个酷炫的技巧做点什么。”
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1253391533613514853)** (1 条消息): 

- **用户解决聊天卡死问题**：一位用户对项目收到的反馈表示感谢，并指出了聊天挂起（hanging）的问题，怀疑这可能与连接的 API 有关。他们提到正在尝试各种 UI 更改，并承诺继续进行实验。
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1253173131820924949)** (18 条消息🔥): 

- **Toucan TTS 展示了强大的多语言能力**：成员们讨论了 [Toucan TTS](https://x.com/reach_vb/status/1803529768861610073?s=46)，强调其作为支持 7000 种语言的最强多语言开源 TTS 模型的地位。该项目包含一个用于处理与语言无关的发音特征（articulatory features）的文本前端，并利用元学习（meta-learning）来覆盖无数据的语言。

- **Claude 3.5 Sonnet 让社区印象深刻**：成员们对 [Claude 3.5 Sonnet](https://x.com/anthropicai/status/1803790676988920098?s=46) 的发布感到兴奋，指出其性能超越了竞争对手，且速度是 Claude 3 Opus 的两倍，成本仅为其五分之一。正面的评价提到了它在编程和自主 Pull Request 管理方面的高效。

- **Artifacts 功能大放异彩**：讨论重点关注了 [Claude 3.5 Sonnet](https://x.com/anthropicai/status/1803790681971859473?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 中新的 Artifacts 功能，它被描述为 Code Interpreter 的精神继任者。用户对其实时生成文档、代码、图表等的能力表示赞赏。

- **工程咨询公司宣布合并**：Jason Liu 的咨询公司 Parlance Labs 正与 [Hamel Husain 和 Jeremy Lewi 的团队](https://x.com/jxnlco/status/1803813743714844863?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w) 合并，以提供全面的 AI 产品和工程支持。合并后的团队将专注于基础设施、Fine-tuning 和 Evaluations 等服务。

- **Groq 首次推出 Whisper 模型支持**：成员们讨论了 Groq 新推出的 [Whisper 模型支持](https://x.com/sjwhitmore/status/1803811998548812140?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)，其运行速度达到实时速度的 166 倍。然而，人们对其目前的 Rate Limits 以及除技术演示之外的实际应用表示担忧。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/anthropicai/status/1803790676988920098?s=46">来自 Anthropic (@AnthropicAI) 的推文</a>：介绍 Claude 3.5 Sonnet——我们迄今为止最智能的模型。这是我们 3.5 模型家族的首次发布。Sonnet 现在在关键评估中超越了竞争对手模型，且速度是 Claude 3 Opus 的两倍...</li><li><a href="https://x.com/reach_vb/status/1803529768861610073?s=46">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：Toucan TTS：支持 7000 种语言的 MIT 许可文本转语音（TTS）！🔥 目前最强大的多语言开源 TTS 模型 ⚡ 第一步：他们构建了一个文本前端，可以将来自 IS... 的任何语言的文本转换...</li><li><a href="https://x.com/mikeyk/status/1803791011828711930?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Mike Krieger (@mikeyk) 的推文</a>：很高兴向大家介绍 Claude 3.5 Sonnet——我们迄今为止最智能的模型。这是我们 3.5 模型家族的首次发布。Sonnet 现在在关键评估中超越了竞争对手模型，且速度...</li><li><a href="https://x.com/alexalbert__/status/1803804677701869748">来自 Alex Albert (@alexalbert__) 的推文</a>：Claude 开始变得非常擅长编程和自主修复 Pull Request。很明显，一年后，很大比例的代码将由 LLM 编写。让我展示...</li><li><a href="https://x.com/anthropicai/status/1803790681971859473?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Anthropic (@AnthropicAI) 的推文</a>：我们还在 http://claude.ai 上推出了 Artifacts 的预览版。你可以要求 Claude 生成文档、代码、Mermaid 图表、矢量图形，甚至是简单的游戏。Artifacts 会出现在你的...旁边</li><li><a href="https://x.com/alexalbert__/status/1803837844798189580?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Alex Albert (@alexalbert__) 的推文</a>：这真是充实的一年😅</li><li><a href="https://x.com/jxnlco/status/1803813743714844863?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w">来自 jason liu (@jxnlco) 的推文</a>：如果你是寻求以下支持的工程领导者，请联系我们：1. 加速你的 AI 产品工作 2. 提升现有工程团队的技能 3. 构建可扩展的路线图以吸引更多人才。信息见此：https://...</li><li><a href="https://x.com/sjwhitmore/status/1803811998548812140?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Sam Whitmore (@sjwhitmore) 的推文</a>：Dot 已在 App Store 上线！为了配合发布，@jasonyuandesign 和我写了关于我们在过去一年中使用 Dot 的历程。你可以在这里阅读我们的故事：https://new.computer/ 引用...</li><li><a href="https://www.heavybit.com/library/article/ai-hidden-opportunities-for-software-developers-swyx">AI 的隐藏机遇：Shawn "swyx" Wang 谈新用例和职业生涯 | Heavybit</a>：Shawn “swyx” Wang 讨论了 AI 中的隐藏机遇，包括新的用例以及有志于成为 AI Engineer 的人的新机会。
</li>
</ul>

</div>
  

---

### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1253365331611746304)** (3 messages): 

- **将 YOLOv10 和 OCR 封装进 Llamafile**：一位成员询问了将 **YOLOv10 PyTorch** 和 **OCR Safe Tensors** 等其他模型类型整合进 Llamafile 的可能性。另一位成员建议使用 llama.cpp 的 Python 脚本将它们转换为 **gguf** 格式。
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1253287033774805053)** (2 messages): 

- **Infer: Summer '24 为 AI/ML 领域升温**：*Hudson Buzby* 和 *Russ Wilcox* 领导了关于**现实场景推荐系统**以及 AI/ML 挑战的讨论，活动地点在 [Infer: Summer '24](https://tinyurl.com/4dfvcte7)。会议将邀请来自 Lightricks 等公司的专家，重点关注优化 AI 流水线（pipelines）和消除不准确内容。

- **RecSys Learners 虚拟见面会宣布**：欢迎参加 2024 年 6 月 29 日举行的免费 [RecSys Learners 虚拟见面会](https://lu.ma/7pvpp1cm)。该活动由 *Rohan Singh S Rajput* 主办，为推荐系统领域的初学者和资深专业人士提供社交和学习机会。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://lu.ma/7pvpp1cm">RecSys Learner Virtual Meetup · Luma</a>: 加入我们，参加这场精彩且信息丰富的 RecSys Learner 虚拟见面会，专为对推荐系统充满热情的爱好者和专业人士设计。这……</li><li><a href="https://tinyurl.com/4dfvcte7">Infer Summer ‘24 by Qwak | AI 和 ML 背后的工程学</a>: Qwak 举办的 Infer Summer ‘24 邀请了 AI 领袖分享全球领先公司如何在生产环境中使用 ML 和 AI。请于 2024 年 6 月 26 日上午 11:00 (EDT) 参加直播。
</li>
</ul>

</div>
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1253063424586289152)** (2 messages): 

- **Florence 2 在 OCR 和手写识别方面表现卓越**：一位用户分享说，[微软的 Florence 2](https://x.com/dylfreed/status/1803502158672761113) 在手写识别和 OCR 方面表现出色，在公共记录上展示了令人印象深刻的结果。他们强调了其对新闻工作流的重要性。
- **在 Hugging Face 上体验 Florence 2**：另一位用户提供了 [Hugging Face 上的 Florence-2](https://huggingface.co/spaces/gokaygokay/Florence-2) 链接，用户可以在那里测试其功能。他们强调了它在各种视觉任务中的熟练程度。
- **Florence-2 模型摘要**：Florence-2 采用基于提示（prompt-based）的方法处理视觉和视觉语言任务，利用了拥有 54 亿条注释的 FLD-5B 数据集。它在 Zero-shot 和微调（fine-tuned）设置下均表现出色，精通多任务学习。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/microsoft/Florence-2-base">microsoft/Florence-2-base · Hugging Face</a>: 暂无描述</li><li><a href="https://x.com/dylfreed/status/1803502158672761113">来自 Dylan Freedman (@dylfreed) 的推文</a>: 新的开源 OCR 模型刚刚发布！微软的这款模型拥有我在开源模型中见过的最好的文本识别能力，在手写识别方面表现出色。它还能处理各种……
</li>
</ul>

</div>
  

---



### **YAIG (a16z Infra) ▷ #[ai-ml](https://discord.com/channels/958905134119784489/1013536071709118565/1253413048052617268)** (1 messages): 

- **对 AI 炒作周期的批判引发笑声**：一位成员分享了一篇题为 *“如果你再提 AI，我就把你摔在地上 (I Will Fucking Piledrive You If You Mention AI Again)”* 的博客文章，建议读者腾出 10 分钟冷静时间来充分享受阅读，并提到它对当前 AI 炒作周期的看法既搞笑又真实。他们引用了一段批评大多数机构实施复杂 AI 技术不切实际的话：“这不只是灾难的药方，这是为想要准备‘十二道菜灾难大餐’的人准备的食谱。” [阅读博客文章](https://ludic.mataroa.blog/blog/i-will-fucking-piledrive-you-if-you-mention-ai-again/)

**提及的链接**: <a href="https://ludic.mataroa.blog/blog/i-will-fucking-piledrive-you-if-you-mention-ai-again/">I Will Fucking Piledrive You If You Mention AI Again — Ludicity</a>: 暂无描述

  

---



---



---



{% else %}


> 完整的各频道详细内容已针对邮件进行截断。
> 
> 如果你想查看完整内容，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}