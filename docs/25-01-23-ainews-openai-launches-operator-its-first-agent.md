---
companies:
- openai
- anthropic
- deepseek-ai
- google-deepmind
- perplexity-ai
date: '2025-01-24T03:34:34.136294Z'
description: '**OpenAI** 发布了 **Operator**，这是一款用于处理预订和下单等网页任务的高级计算机操作智能体（computer-using
  agent）。该产品目前已面向美国的 Pro 用户开放，并承诺未来将提供 API。它具备长达 20 分钟的长跨度远程虚拟机（VM）运行能力和视频导出功能，展示了目前最顶尖的智能体性能，但尚未达到人类水平。**Anthropic**
  在三个月前曾以开源演示的形式发布过类似的智能体。


  **DeepSeek AI** 推出了 **DeepSeek R1**，这是一款开源推理模型，在“人类最后的考试”（**Humanity''s Last Exam**）数据集上表现优异，超越了
  **LLaMA 4** 和 **OpenAI 的 o1** 等模型。**Google DeepMind** 开源了 **VideoLLaMA 3**，这是一个用于图像和视频理解的多模态基础模型。**Perplexity
  AI** 为 Android 系统发布了 **Perplexity Assistant**，具备推理和搜索功能。


  “人类最后的考试”（**Humanity''s Last Exam**）数据集包含 3,000 道测试 AI 推理能力的题目，目前模型的准确率均低于 10%，表明仍有很大的提升空间。OpenAI
  的计算机操作智能体（CUA）在 OSWorld 和 WebArena 基准测试中表现有所提升，但仍落后于人类。**Anthropic AI** 引入了“引用”（Citations）功能，以提供更安全的
  AI 回复。*Sam Altman* 和 *Swyx* 对 Operator 的发布及其功能发表了评论。'
id: 467b0e01-1a3e-4a2a-9f2b-a7eaedd3acfb
models:
- operator
- deepseek-r1
- videollama-3
- llama-4
- o1
- claude
original_slug: ainews-openai-launches-operator-its-first-agent
people:
- sam-altman
- swyx
title: OpenAI 发布其首个 AI 智能体 Operator。
topics:
- computer-using-agent
- reasoning
- multimodality
- performance-benchmarks
- open-source
- ai-safety
- benchmarking
- video-generation
- model-evaluation
---


- 正如 Sam 所说，[在接下来的几周和几个月内还将发布更多 agent](https://x.com/nickadobos/status/1882496722741633342?s=46)

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 回顾

> 所有摘要均由 Claude 3.5 Sonnet 完成，从 4 次运行中择优。

**AI 模型与发布**

- **OpenAI Operator 发布**：[@OpenAI](https://twitter.com/OpenAI/status/1882509286439637448) 推出了 **Operator**，这是一个 **computer-using agent**，能够与浏览器交互以执行 **预订** 和 **订购杂货** 等任务。[@sama](https://twitter.com/sama/status/1882234406662000833) 对此次发布表示赞赏，而 [@swyx](https://twitter.com/swyx/status/1882505900717687231) 强调了它高效处理 **重复性浏览器任务** 的能力。

- **DeepSeek R1 及其模型**：[@deepseek_ai](https://twitter.com/DeepLearningAI/status/1882516386490245269) 揭晓了 **DeepSeek R1**，这是一个 **开源推理模型**，在 **Humanity’s Last Exam** 上的表现优于许多竞争对手。[@francoisfleuret](https://twitter.com/francoisfleuret/status/1882320945043685601) 称赞了它的 **Transformer 架构** 和 **性能基准**。

- **Google DeepMind 的 VideoLLaMA 3**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1882270342649126947) 宣布了 **VideoLLaMA 3**，这是一个专为 **图像和视频理解** 设计的 **多模态基础模型**，现已 **开源** 以供更广泛的研究和应用。

- **Perplexity Assistant 发布**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1882512352043753829) 推出了 **Android** 版 **Perplexity Assistant**，集成了 **推理** 和 **搜索功能** 以增强日常生产力。用户现在可以 **激活助手** 并使用 **多模态交互** 等功能。

**AI 基准测试与评估**

- **Humanity's Last Exam**：[@DanHendrycks](https://twitter.com/DanHendrycks/status/1882433928407241155) 介绍了 **Humanity’s Last Exam**，这是一个包含 **3,000 个问题的数据集**，旨在评估 AI 在各个领域的 **推理能力**。目前模型的准确率低于 **10%**，表明仍有巨大的提升空间。

- **CUA 在 OSWorld 和 WebArena 上的表现**：[@omarsar0](https://twitter.com/omarsar0/status/1882501699757379666) 分享了 **Computer-Using Agent (CUA)** 在 **OSWorld** 和 **WebArena** 基准测试中的结果，展示了其相对于之前 SOTA 模型的 **性能提升**，尽管仍落后于 **人类表现**。

- **DeepSeek R1 的主导地位**：来自 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1882500542909600241) 的多条推文强调了 **DeepSeek R1 在文本基准测试中的卓越表现**，在各种评估指标上超越了 **LLaMA 4** 和 **OpenAI 的 o1** 等模型。

**AI 安全与伦理**

- **引用与安全 AI 响应**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1882481265414377919) 推出了 **Citations** 功能，使 **Claude** 等 AI 模型能够提供带有精确**来源引用**的**有据回答**，从而增强了**输出可靠性**和**用户信任**。

- **AI 的过度炒作与幻觉**：[@kylebrussell](https://twitter.com/kylebrussell/status/1882481976927756735) 批评了 **AI 技术的过度炒作**，强调不应因**幻觉**和**错误**而全盘否定 **AI 的进步**。

- **AI 作为创意协作者**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1882467983840145841) 提倡将 **AI 视为创意协作者**而非单纯的工具，强调了在**艺术创作**中进行**主观和情感**评估的重要性。

**AI 研究与开发**

- **程序合成与 AGI**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1882231849524809768) 探讨了将**程序合成**作为实现**通用人工智能 (AGI)** 的路径，通过结合**模式识别**与**抽象推理**来克服当前 **Deep Learning** 的局限性。

- **扩散特征提取器**：[@ostrisai](https://twitter.com/ostrisai/status/1882447889882034629) 报告了使用 **LPIPS 输出**训练 **Diffusion Feature Extractors** 的进展，从而在生成的图像中获得**更清晰的图像特征**并增强了**文本理解**。

- **X-Sample 对比损失 (X-CLR)**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1882428109225373786) 介绍了 **X-Sample Contrastive Loss (X-CLR)**，这是一种**自监督损失函数**，通过分配**连续相似度分数**，提升了相比 **SimCLR** 和 **CLIP** 等传统方法的**对比学习**性能。

**AI 行业与公司**

- **Stargate 项目投资**：[@saranormous](https://twitter.com/saranormous/status/1882442993959149991) 讨论了 **5000 亿美元的 Stargate 投资**，该项目旨在提升**算力**和 **AI Token 使用量**，并对其在**智能获取**和**行业竞争**方面的长期影响提出了疑问。

- **Google Colab 的影响**：[@osanseviero](https://twitter.com/osanseviero/status/1882352487858151729) 强调了 **Google Colab** 在普及 **GPU 访问**方面的重大作用，促进了**开源项目**、**教育**和 **AI 研究**的进步。

- **OpenAI 与 Together Compute 合作伙伴关系**：[@togethercompute](https://twitter.com/togethercompute/status/1882516682016719340) 宣布与 **Cartesia AI** 达成合作，通过 **Together API** 提供对 **Sonic**（一种**低延迟语音 AI 模型**）的访问。此次合作旨在通过结合**聊天、图像、音频和代码**功能，打造**无缝的多模态体验**。

**梗/幽默**

- **AI 取代规则律师**：[@NickEMoran](https://twitter.com/NickEMoran/status/1882469624618606682) 拿 **Humanity’s Last Exam** 中包含的《万智牌》和《龙与地下城》开玩笑，幽默地暗示 **LLM** 可能很快就会接管**规则律师 (Rules Lawyers)** 的角色。

- **AI 对流行文化的影响**：[@saranormous](https://twitter.com/saranormous/status/1882204427676996021) 分享了一段反映 **AI 能力**的幽默引用，并结合 **Memes** 展示了 **AI 进步**轻松有趣的一面。

- **Elon 与 Sam 的信任度辩论**：[@draecomino](https://twitter.com/draecomino/status/1882493261279056037) 幽默地质疑了 **Elon Musk** 与 **Sam Altman** 相比的**可信度**，引发了一场关于 **AI 领导力**的轻松辩论。

- **有趣的 AI 交互**：[@nearcyan](https://twitter.com/nearcyan/status/1882320601303621971) 分享了一条关于 **AI 生成内容**幽默面的推文，强调了 **AI 模型**与用户提示词交互时产生的古怪且出人意料的结果。

---

本摘要将提供的推文分类为 **AI 模型与发布**、**AI 基准与评估**、**AI 安全与伦理**、**AI 研究与开发**、**AI 行业与公司**以及**梗/幽默**，确保了主题的一致性并将类似的讨论点进行归类。每个摘要都引用了带有内联 Markdown 链接的直接推文，以保持事实依据。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. DeepSeek 的竞争力震撼科技巨头**

- **[deepseek is a side project](https://i.redd.it/zdvrlxahzpee1.jpeg)** ([Score: 1406, Comments: 165](https://reddit.com/r/LocalLLaMA/comments/1i80cwf/deepseek_is_a_side_project/)): **DeepSeek** 被描述为一家拥有深厚数学基础、并拥有大量用于交易和挖矿的 **GPUs** 的量化公司的副业项目。该项目旨在优化这些 **GPUs** 的利用率，突显了该公司的技术实力。
  - **DeepSeek 的起源与意图**: 许多用户强调 **DeepSeek** 是由一家对冲基金（具体为 **High-Flyer**）资助的，并强调这是一个利用闲置 **GPUs** 的副业项目。虽然该项目不被视为 **OpenAI** 或 **xAI** 等巨头的直接竞争对手，但它展示了极高的效率和低成本运营，仅使用了 **2000 个 H100 GPUs**，而其他公司则使用了 **10 万个**。
  - **量化背景与 GPU 利用率**: 评论讨论了该对冲基金的量化专业知识，这使他们能够在硬件有限的情况下优化资源使用并创建高效模型。用户指出，高频交易 (**HFT**) 与 AI 开发之间的技能存在重叠，**quants** 经常开发需要精确且快速执行的模型，这与交易算法类似。
  - **对比与市场影响**: 人们对大型公司进行大规模硬件投资的必要性表示怀疑，质疑当像 DeepSeek 这样的小型项目能够取得具有竞争力的结果时，其 **ROI** 如何。用户幽默地指出，一家对冲基金的副业项目竟然对主要 AI 玩家构成潜在威胁，这具有讽刺意味，突显了有效利用现有资源的战略优势。


- **[Meta panicked by Deepseek](https://i.redd.it/ek65oz361see1.png)** ([Score: 535, Comments: 114](https://reddit.com/r/LocalLLaMA/comments/1i88g4y/meta_panicked_by_deepseek/)): 据报道，Meta 对 **DeepSeek v3** 在 **benchmarks** 中超越 **Llama 4** 感到惊慌，促使工程师紧急分析并复制 DeepSeek 的能力。担忧包括 **generative AI** 部门的高昂成本，以及在领导层薪酬背景下难以证明支出的合理性，这表明 Meta 在 AI 进步方面面临组织挑战和紧迫感。
  - **对 DeepSeek v3 影响的怀疑**: 许多评论者对 DeepSeek v3 导致 Meta 恐慌的说法表示怀疑，理由是 DeepSeek 的模型与 Llama 模型之间存在显著的规模差异。**ResidentPositive4122** 强调，DeepSeek 在 AI 领域以其强大的模型而闻名，这与他们是“未知”威胁的观点相矛盾。
  - **Meta 的战略地位**: 评论者如 **FrostyContribution35** 和 **ZestyData** 认为，Meta 在 AI 研究中仍占据强势地位，在 **BLT** 和 **LCM** 等架构改进方面持续创新。他们认为，尽管竞争激烈，Meta 广泛的数据资源和才华横溢的研究团队仍提供了显著优势。
  - **组织与资源挑战**: 讨论涉及 Meta 的组织动态，例如领导层决策给工程师带来的压力，以及美国与中国相比的能源成本。**The_GSingh** 指出，尽管 Meta 拥有广泛的研究，但他们在实施新模型方面表现不足，而 **Swagonflyyyy** 则提到 DeepSeek 的高性价比方法突显了 Meta 在 AI 领导层薪酬支出上的低效。

- **[开源 DeepSeek 在“人类最后的考试”中击败了并不那么 OpenAI 的 OpenAI！](https://i.redd.it/lxwhx4eicree1.jpeg)** ([得分: 238, 评论: 36](https://reddit.com/r/LocalLLaMA/comments/1i856wr/opensource_deepseek_beat_not_so_openai_in/)): **DeepSeek 的开源模型 DeepSeek-R1 在“HLE”测试中超越了 GPT-4O 和 Claude 3.5 Sonnet 等其他模型，实现了 9.4% 的准确率，校准误差为 81.8%。** 尽管 DeepSeek-R1 不是多模态模型，但它依然超越了竞争对手，详细结果可在附录 C.2 中查看。
  - **DeepSeek-R1 的表现**：DeepSeek-R1 作为一个侧重项目，在纯文本数据集上的表现令人印象深刻，超越了 **OpenAI 的 O1** 等成熟模型，其准确率为 **9.4%**，而 **O1** 为 **8.9%**。这一成就凸显了非主流项目挑战行业领导者的潜力。
  - **人类最后的考试 (HLE)**：该基准测试对于测试 AI 在各学科的专家级推理能力至关重要，揭示了当前 AI 系统的重大差距。领先模型的得分均低于 **10%**，表明在抽象推理和专业知识方面仍需改进。
  - **开源与行业动态**：DeepSeek 的成功引发了关于开源 AI 现状的讨论，用户对 **Meta** 和 **xAI** 等主要参与者近期缺乏发布表示质疑。对话还涉及了像 DeepSeek 这样缺乏传统科技巨头支持的项目意外崛起，并实现了最先进的性能。


**主题 2. 高级 LLM 架构：字节级模型与推理 Agent**

- **[字节跳动发布采用 Apache 2.0 协议的 2B、7B 和 72B 用于计算机操作的“推理” Agent](https://v.redd.it/ealby85nioee1)** ([得分: 541, 评论: 52](https://reddit.com/r/LocalLLaMA/comments/1i7wcry/bytedance_dropping_an_apache_20_licensed_2b_7b/)): **字节跳动 (ByteDance)** 发布了采用 **Apache 2.0 协议** 的大语言模型 (LLM)，参数量分别为 **20 亿、70 亿和 720 亿**，重点在于增强计算机操作的推理任务。这些模型旨在提高计算推理能力，展示了字节跳动对开源 AI 开发的承诺。
  - 讨论强调了字节跳动新模型的**潜力与局限性**，用户对 **2B 和 7B** 参数模型在“快捷键”等基础功能之外的实际用例表示好奇。一些用户还报告了在从较小模型获取有意义输出时的初步困难，表明可能需要部署和使用指南。
  - 用户对 **Gnome Desktop 演示** 表现出浓厚兴趣，这表明了对模型在操作系统环境中能力的期待。用户还在讨论针对非 Web 软件使用 **基于 LLM 的方法** 的必要性，并将其与 **AutoHotkey** 等工具进行比较。
  - 社区分享了指向 **GitHub 仓库** 和 **Hugging Face** 等资源的链接，一些用户对获取这些资源的便利性表示感谢。此外，还有关于使用仓库中特定 Prompt 以确保模型正常运行的讨论，强调了理解训练方法论的重要性。

- **[首个高性能、无需 Tokenization 的开源字节级模型已发布。EvaByte 是一个 6.5B 参数的模型，还具有多字节预测功能以实现更快的推理（对比同等规模的基于 Tokenizer 的模型）](https://i.redd.it/o28q2pl6roee1.png)** ([得分: 249, 评论: 65](https://reddit.com/r/LocalLLaMA/comments/1i7x5nd/the_first_performant_opensource_bytelevel_model/)): **EvaByte**，一个 **6.5B 参数** 的开源字节级模型已经发布，它提供了多字节预测功能，在无需 Tokenization 的情况下实现了更快的推理。该模型在 14 项任务中实现了约 **60% 的性能**，其训练 Token 数量在对数尺度上略高于 **0.3**，如对比其他模型的散点图所示。
  - 讨论强调了 **EvaByte** 与其他模型相比的 **性能和速度**，一些用户注意到其架构允许更快的解码——比原生字节模型快 **5-10 倍**，比基于 Tokenizer 的 LM 快 **2 倍**。该模型处理 **多模态任务** 的效率比 BLTs 更高，因为它需要的训练字节更少。
  - 模型的 **字节级 Token** 引起了争论，人们担心输出速度较慢以及上下文填充过快。然而，一些人认为改进的架构通过提高预测速度抵消了这些缺点，而另一些人则指出，由于词典更小且计算更简单，有可能降低 **硬件开销**。
  - 用户对 **训练数据的不一致性** 和模型的扩展能力提出疑问，并参考了 **Hugging Face** 和 **博客** 进行进一步探索。人们对 **EvaByte** 与 **GPT-J** 和 **OLMo** 等其他模型的对比很感兴趣，并讨论了它在聊天机器人输出上的训练，这可能会导致回答中出现错误。


**主题 3. 提升 AI 模型推理能力的工具：Open WebUI 的增强功能**

- **Open WebUI 在今天发布的两个新版本中增加了专注于推理的功能！！！0.5.5 增加了 "Thinking" 标签支持，以简化推理模型聊天（适用于 R1）。0.5.6 带来了新的 "reasoning_effort" 参数来控制认知开销。** ([得分: 104, 评论: 18](https://reddit.com/r/LocalLLaMA/comments/1i7pxn7/open_webui_adds_reasoningfocused_features_in_two/)): **Open WebUI** 发布了两个更新版本 **0.5.5** 和 **0.5.6**，增强了推理模型的交互。版本 **0.5.5** 引入了一个 "think" 标签，可以直观地显示模型的思考时长，而版本 **0.5.6** 增加了 **reasoning_effort** 参数，允许用户调整 OpenAI 模型付出的认知开销，从而提高复杂查询的定制化程度。更多详情可以在其 [GitHub 发布页面](https://github.com/open-webui/open-webui/releases)找到。
  - **reasoning_effort** 参数目前对 **R1 蒸馏模型** 没有影响，一位用户测试发现不同设置下的“思考”时间没有差异。该参数目前似乎仅适用于 OpenAI 推理模型。
  - **推理引擎** 需要自己实现 "reasoning_effort"，因为它不是一个模型参数。一种建议的方法是调整“思维结束” Token 的采样缩放系数，这可以有效地修改感知到的认知开销。
  - 用户期待修复渲染伪影并增加 **MCP 支持** 以标准化工具使用，这预计将增强平台的实用性。

- **Deepseek R1 的开源版本与官方 API 版本存在差异** ([Score: 80, Comments: 57](https://reddit.com/r/LocalLLaMA/comments/1i7o9xo/deepseek_r1s_open_source_version_differs_from_the/))：**与官方 API 相比，Deepseek R1 的开源模型在 CCP 相关问题上表现出更多的审查，这与预期不符。** 这种差异引发了对 Benchmark 准确性和潜在偏见回答的担忧，因为开源模型的表现可能较差并传播偏见观点，从而影响第三方供应商和像 LM Arena 这样的人工排名排行榜。**测试显示，开源模型在敏感话题上会中断其思考过程（thinking process），这表明模型可能并不相同，研究人员在研究中应明确说明他们使用的是哪个版本。**
  - **开源模型**与**官方 API** 之间存在明显的差异，开源模型在 **CCP 相关问题**上表现出更多的审查。包括 **TempWanderer101** 和 **rnosov** 在内的用户讨论了 Benchmark 可能无法准确衡量开源模型，以及模型可能并不相同，从而影响性能和第三方供应商的质量。
  - **审查问题**可能与 Prompt 处理的差异有关，**rnosov** 指出，在文本补全模式下，使用 `<think>` 标签后接换行符可以绕过审查。这表明官方 API 可能使用了不同的 Template 或隐藏 Prompt，例如“谨慎处理与中国相关的查询”，从而影响回答。
  - 讨论中还涉及了**成本和性能**的影响，**TempWanderer101** 注意到 **TogetherAI** 和 **OpenRouter** 之间的定价差异。模型版本之间潜在的混淆引发了对 Benchmark 公平性和研究结果可复现性的担忧，强调了明确模型版本标识的必要性。


**Theme 4. NVIDIA 增强 AI 的 GPU 创新：Blackwell 与长上下文库**

- **[配备 96GB GDDR7 显存和 512-bit 位宽的 NVIDIA RTX Blackwell GPU 曝光](https://videocardz.com/newz/nvidia-rtx-blackwell-gpu-with-96gb-gddr7-memory-and-512-bit-bus-spotted?fbclid=IwZXh0bgNhZW0CMTEAAR3i39eJbThbgTnI0Yz4JdnkMXgvj4wlorxOdbBeccw35kkqWqyrG816HpI_aem_EoENoW6h6SP-aU7FVwBWiw)** ([Score: 209, Comments: 92](https://reddit.com/r/LocalLLaMA/comments/1i7nmk5/nvidia_rtx_blackwell_gpu_with_96gb_gddr7_memory/))：**NVIDIA 的 RTX Blackwell GPU** 已被发现配备 **96GB GDDR7 显存**和 **512-bit 位宽**，这标志着显存容量和带宽的重大更新。这一进展表明高性能计算和 AI 应用的处理能力具有潜在的提升。
  - 讨论重点关注了 RTX Blackwell GPU 的**潜在定价**，估计范围在 **$6,000 到 $18,000** 之间。一些用户将其与 **MI300X/325X** 和 **H100** 等其他显卡进行比较，认为后者在类似价格点上可能提供更好的性能或价值。
  - 有推测认为 RTX Blackwell 可能是 **RTX 6000 Ada**（最高 48GB）的**继任者**。这款新卡的 **96GB GDDR7** 显存被视为实质性的升级，可能将其定位在**工作站显卡系列**中。
  - 用户幽默地表达了对负担能力的担忧，开玩笑说要**卖肾**或加班来买得起这款新卡。这反映了一种普遍情绪：虽然显卡的规格令人印象深刻，但其价格可能是许多潜在买家的障碍。

- **首批 5090 LLM 测试结果，对比 4090 和 6000 ada** ([Score: 70, Comments: 44](https://reddit.com/r/LocalLLaMA/comments/1i867k8/first_5090_llm_results_compared_to_4090_and_6000/))：**NVIDIA GeForce RTX 5090** 的 **LLM benchmarks** 已发布预览，显示出对比 **RTX 4090** 和 **6000 Ada** 型号的显著改进。详细结果和对比可以在链接的 [Storage Review 文章](https://www.storagereview.com/review/nvidia-geforce-rtx-5090-review-pushing-boundaries-with-ai-acceleration)中找到。
  - **性能预期与瓶颈**：用户原本预期 **RTX 5090** 由于更高的显存带宽会带来 **60-80% 的 tokens per second 提升**，但由于未观察到这些增益，怀疑存在瓶颈或基准测试问题。**FP8** 正在进入主流，提供比整数模型量化更好的性能，而 **FP4** 距离普及仍需数年。
  - **硬件特性与对比**：讨论强调了对 **multi-GPU training** 能力以及 5090 像 4090 一样通过自定义驱动解锁 p2p 的兴趣。**RTX 6000** 与 **GeForce** 系列的对比指出，尽管 6000 相对于 **GeForce** 系列性能较低，但其拥有更高的 VRAM 且更注重效率。
  - **性能指标**：与 4090 相比，**RTX 5090** 在 **LLMs** 方面表现出 **25-30% 的提升**，在图像生成方面表现出 **40% 的提升**，符合规格预期。用户还注意到新一代产品中 **FP8 和 FP4 优化** 对增强性能的重要性。


## 其他 AI Subreddit 总结

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. OpenAI 发布用于电脑的 Operator 工具**

- **[OpenAI 发布 Operator——一个可以为你操作电脑的 agent](https://www.technologyreview.com/2025/01/23/1110484/openai-launches-operator-an-agent-that-can-use-a-computer-for-you/?utm_medium=tr_social&utm_source=reddit&utm_campaign=site_visitor.unpaid.engagement)** ([Score: 107, Comments: 70](https://reddit.com/r/OpenAI/comments/1i89lt0/openai_launches_operatoran_agent_that_can_use_a/))：**OpenAI** 推出了 **Operator**，这是一个旨在代表用户自主使用电脑的 agent。这一进展代表了 AI 能力的重大飞跃，能够实现更复杂的任务自动化以及与数字环境的交互。
  - 用户对 **Operator** 的能力表示怀疑，质疑其控制浏览器以外的操作系统的能力，以及处理 **CAPTCHAs** 或报税等复杂任务的有效性。关于隐私和每月 200 美元的高昂服务费用的担忧也是讨论的重点。
  - 一些评论强调了 **Operator** 的潜力，提到了它在编程中的应用以及与 **Google Sheets** 等工具的兼容性，尽管目前的局限性使其对非程序员的吸引力较小。对话还涉及了 AI 进步的飞速步伐以及重大改进的潜力，参考了 **2024** 年视频模型的快速发展。
  - 几条评论讨论了 AI 对隐私的更广泛影响，认为未来的技术发展可能本质上会减少隐私，将其比作当今拥有手机的必要性。**EU** 严格的数据隐私法被认为是导致该地区 AI 技术推广延迟的一个因素。


- **有没有人的 chat gpt 也不好使了？内部服务器错误？** ([Score: 260, Comments: 320](https://reddit.com/r/OpenAI/comments/1i81kkp/is_anyones_chat_gpt_also_not_working_internal/))：**ChatGPT 用户**正面临问题，特别是**内部服务器错误**，导致无法访问。
  - 来自 **New Zealand**、**Spain** 和 **Australia** 等地的许多用户报告 **ChatGPT** 宕机，遇到 **503 Service Temporarily Unavailable** 错误，并建议订阅 [OpenAI status page](https://status.openai.com/) 以获取更新。一些用户幽默地推测可能是 **AGI** 的进展导致了这个问题。
  - 几位用户提到 **DeepSeek** 作为一个替代方案，强调了它在解决复杂问题（如 **Docker** 配置错误）方面的有效性，并考虑取消 **OpenAI** 订阅以支持这个免费工具。
  - 有建议在应用程序内部加入状态指示器，用户推荐 **Downdetector** 作为监控服务可用性的可靠替代方案。

- **[Sam Altman 表示他改变了对 Trump 的看法，与此同时“第一好友” Elon Musk 因 5000 亿美元的 Stargate Project 在网上对其进行抨击](https://fortune.com/2025/01/23/sam-altman-donald-trump-elon-musk-stargate-project/)** ([Score: 474, Comments: 104](https://reddit.com/r/OpenAI/comments/1i82ean/sam_altman_says_hes_changed_his_perspective_on/)): 据报道，**Sam Altman** 改变了对 **Donald Trump** 的看法，而 **Elon Musk** 则因 **5000 亿美元的 Stargate Project** 在网上对其提出批评。帖子中未提供更多细节。
  - 讨论重点关注了**对 AI 的担忧**以及潜在的乌托邦式未来，用户表达了对 AI 驱动的监控国家以及用于控制或战争的自主无人机的恐惧。**WloveW** 和 **lepobz** 讨论了 AI 在监控和执法中的作用，强调了国家行为者大规模部署的风险。
  - **Sam Altman** 立场的转变遭到了批评，一些评论者表达了对亿万富翁及其对 AI 和政治影响的不信任。**RealPhakeEyez** 和 **Sharp_Iodine** 反思了亿万富翁决策的更广泛影响及其对社会的影响，并将其与法西斯主义和企业权力联系起来。
  - 讨论了 **5000 亿美元的 Stargate Project** 及其政治关联，**-Posthuman-** 的一条评论指出了该项目与 **OpenAI**、**Microsoft** 以及拜登政府的历史渊源，同时质疑了 **Trump** 的参与和功劳主张。


**主题 2. OpenAI 对 2025 年 AI Agent 的愿景**

- **[OpenAI 计划在 2025 年底前发布旨在取代高级软件工程师的 Agent](https://i.redd.it/7mse7ko2unee1.jpeg)** ([Score: 148, Comments: 147](https://reddit.com/r/OpenAI/comments/1i7twg4/open_ai_set_to_release_agents_that_aim_to_replace/)): OpenAI 计划在 **2025** 年底前发布旨在协助并可能取代高级软件工程师的 AI Agent。该计划包括测试一款 AI 编程助手，目标是让 ChatGPT 达到 **10 亿日活跃用户**，并复制 **Sam Altman** 所述的资深程序员的能力。
  - 许多评论者对 AI 在 **2025** 年前取代高级软件工程师的能力表示**怀疑**，并指出 AI 目前的局限性，例如无法处理需要人类判断和创造力的复杂任务和上下文。**Mistakes_Were_Made73** 强调 AI 可以提高生产力，但不能完全取代工程师，而 **_LordDaut_** 则指出了当前 AI 模型在调试等任务中的局限性。
  - 讨论反映了对 AI 对白领工作更广泛影响的**担忧**，**rom_ok** 建议关注软件工程师可能是一种降低薪资的策略。**Crafty_Fault_2238** 预测未来十年对各种白领工作将产生重大影响，并将其描述为“生存”威胁。
  - 一些用户（如 **tQkSushi**）分享了 AI 提高特定任务效率的例子，但强调了为 AI 提供复杂软件任务所需充足上下文的**挑战**。这种观点得到了 **willieb3** 的认同，他认为虽然 AI 可以提供协助，但仍需要具备专业知识的人类监督才能有效运作。


---

# AI Discord 摘要

> 由 o1-preview-2024-09-12 生成的摘要之摘要的总结

**主题 1. DeepSeek R1 对比现有模型：能力与争议**

- [**DeepSeek R1 在编程对决中胜过 O1**](https://x.com/_aidan_clark_/status/1882135220738220131): 用户报告称 **DeepSeek R1** 在编程任务中超越了 **OpenAI 的 O1**，甚至完美解决了像 *"POTATO THUNDERSTORM!"* 这样古怪的提示词。对比测试显示 **R1** 提供了更强大的代码解决方案和更快速的推理。
- [**用户讨论性能缓慢和审查担忧**](https://x.com/gregisenberg/status/1882064374120268234): 虽然 **DeepSeek R1** 在彻底调试方面给人留下了深刻印象，但一些用户抱怨其在 **Composer** 模式下响应迟缓以及过度审查。对其安全功能的评价带有讽刺意味，用户正努力寻找或创建*无审查版本*。
- [**DeepSeek R1 以一杯咖啡的价格挑战巨头**](https://x.com/gregisenberg/status/1882064374120268234): **Greg Isenberg** 称赞 **DeepSeek R1** 使推理成本比一杯咖啡还便宜，并且是开源的，不像 **GPT-4**，在某些任务上甚至超过了 **O1-Pro**。

**主题 2. OpenAI 的 Operator 和 Agent：新功能与用户反应**

- [**Operator 演示自动浏览器功能，用户对其 200 美元的价格感到不满**](https://www.youtube.com/watch?v=CSE77wAdDLg)：在 **太平洋时间上午 10 点**，**Sam Altman** 发布了 **Operator**，展示了其在浏览器中执行任务的能力，但每月 **200 美元** 的价格非常昂贵。一些用户对其功能表示兴奋，而另一些用户则质疑其过高的定价。
- [**浏览器控制引发安全辩论**](https://x.com/hwchase17/status/1882502767312531954)：**Operator** 自动控制网络浏览器的能力引发了关于 **CAPTCHA 循环**和安全性的担忧。用户将其与 [Browser Use](https://x.com/hwchase17/status/1882502767312531954) 等开源替代方案进行了比较。
- [**OpenAI 预告 Agent 的未来，令用户充满期待**](https://x.com/nickadobos/status/1882496722741633342?s=46)：社区传闻 **Operator** 并非唯一的 Agent，并暗示在未来几周内会有更多发布。用户期待自动化工作流和集成 AI Agent 的新方式。

**主题 3. AI 助手与 IDE：Cursor, Codeium Windsurf, Aider 和 JetBrains**

- **Cursor 用户在 Chat 和 Composer 模式之间纠结**：开发者支持 **Chat 模式**用于友好的代码审查，但批评 **Composer** 会产生不可预测的代码更改。挫败感源于模型在没有适当上下文的情况下对代码进行胡乱修改。
- [**Codeium Windsurf 的 Flow 额度因有 Bug 的 AI 编辑而耗尽**](https://x.com/windsurf_ai/status/1882561985621221451)：用户报告称，由于 AI 反复导致的代码错误，在几小时内就消耗了超过 **10%** 的每月 **flow credits**。修复这些错误迅速消耗了额度，导致用户呼吁更智能地使用资源。
- **JetBrains 粉丝在加入 AI 等候名单时充满希望**：尽管早先有所失望，用户仍对 **JetBrains IDEs** 保持忠诚，纷纷加入 **JetBrains AI** 等候名单，希望它能与 **Cursor** 和 **Windsurf** 竞争。一些人开玩笑说，无论 AI 表现如何，他们都会坚持使用 JetBrains。

**主题 4. AI 模型开发与多 GPU 支持**

- **Unsloth 的多 GPU 支持即将到来**：虽然目前缺乏完整的多 GPU 能力，**Unsloth AI** 预告了未来支持大规模训练的更新，以减少单 GPU 瓶颈。专业用户热切期待能更顺畅地训练大型模型。
- [**BioML 博士后寻求将 Striped Hyena 适配于真核生物**](https://github.com/togethercomputer/stripedhyena/tree/main)：一位研究人员旨在对在原核生物基因组上训练的 **Striped Hyena** 进行微调，使其适用于真核生物序列，并引用了 [Science ado9336](https://www.science.org/doi/10.1126/science.ado9336)。讨论内容包括基因组数据预训练的挑战。
- [**在获得 6000 美元赞助后，社区为 Dolphin-R1 的开源发布欢呼**](https://x.com/cognitivecompai/status/1882140705159799169)：创建 **Dolphin-R1** 花费了 6000 美元的 API 费用，导致开发者寻求支持者以实现开源发布。一位赞助商挺身而出，使得该数据集能够以 **Apache-2.0** 许可在 **Hugging Face** 上共享。

**主题 5. 硬件与性能讨论：GPU、CUDA 更新及训练大型模型**

- **NVIDIA 的 RTX 5090 带来速度提升但功耗更高**：**RTX 5090** 的性能比 **4090** 快 **30%**，但功耗也增加了 **30%**。用户注意到，对于较小的 LLM，该显卡并未充分利用其 **1.7 倍** 的带宽增量。
- [**CUDA 12.8 发布，支持 FP8/FP4 令开发者感到兴奋**](https://developer.nvidia.com/cuda-downloads)：**CUDA 12.8** 引入了对 **Blackwell** 架构的支持以及新的 **FP8** 和 **FP4** TensorCore 指令，引发了关于训练性能潜在提升的热议。
- **DeepSeek R1 巨大的 VRAM 需求引发 GPU 讨论**：以 float16 格式运行 **DeepSeek R1 Distilled Qwen 2.5 32B** 至少需要 **64GB VRAM**，或者使用量化技术需要 **32GB**。讨论强调了 VRAM 的限制以及在有限硬件上训练大型模型的挑战。


---

# PART 1: Discord 高层级摘要

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek R1 超越 O1-Pro**：与会者称赞 **DeepSeek R1** 具有彻底的调试能力，并引用了 [Greg Isenberg 的推文](https://x.com/gregisenberg/status/1882064374120268234)，该推文称其更便宜且开源，在某些任务中超过了 **O1-pro**。
  
  - *“我刚刚意识到 DeepSeek R1 让推理成本变得比一杯咖啡还便宜，”* 一位用户附和道，尽管其他人注意到 **Composer** 模式下响应较为迟缓。
- **O1 订阅争议**：参与者发现 **OpenAI** 的 **O1** Pro 版本每月费用为 **$200**，这在社区中引发了困惑和沮丧。
  
  - 他们将此方案与低成本替代方案进行了对比，认为 **DeepSeek** 在持续使用方面似乎更具性价比。
- **Chat 与 Composer 之争**：开发者们支持将 **Chat 模式** 作为更友好的代码审查工具，强调了其对话式的方法。
  
  - 他们批评 **Composer** 存在不可预测的代码修改，并强调了上下文感知编辑（context-aware editing）的重要性。
- **对按量计费的抵制**：用户质疑是否应该为 AI 相关的 API 调用追踪支付更多费用，对**按量计费（usage-based pricing）**表示怀疑。
  
  - 他们要求透明的费用结构和更强大的模型，以便在不增加开支的情况下提供核心功能。
- **UI-TARS 引领自动化 GUI 交互**：字节跳动在[名为 "UI-TARS: Pioneering Automated GUI Interaction with Native Agents" 的论文](https://huggingface.co/papers/2501.12326)中介绍了 **UI-TARS**，聚焦于先进的 GUI 自动化可能性。
  
  - 开发者在 [GitHub 官方仓库](https://github.com/bytedance/UI-TARS)中探索了其代码库，并指出了其与 **agentic LLM** 流程的潜在协同作用。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 的网页搜索浪潮**：他们为 Codeium (Windsurf) 推出了新的**网页搜索功能**，并在一段[演示视频](https://x.com/windsurf_ai/status/1882561985621221451)中进行了展示，邀请开发者在集成环境中“冲浪”互联网。
  
  - 社区成员被敦促支持该演示视频推文，并强调广泛的参与有助于为更稳健的使用场景优化**搜索功能**。
- **Codeium 扩展：对更新的担忧**：一些用户担心 **Windsurf** 可能会掩盖 Codeium 扩展，理由是自 9 月以来插件更新极少。
  
  - 一份公开声明澄清说，尽管目前的更新重点是 **Windsurf** 的最新功能，但扩展支持对于企业客户仍然至关重要。
- **Devin 的自主性遭受质疑**：**Devin** 被介绍为一款全自动 AI 工具，引发了对其真实能力以及是否仍需要 **human-in-the-loop** 输入的怀疑。
  
  - 一些讨论将其比作“狼来了”的情景，并引用了一篇描述其在多项任务中表现的[博客文章](https://www.answer.ai/posts/2025-01-08-devin.html)。
- **Flow 额度与模型对比**：用户报告称，由于反复修复 AI 导致的代码错误，**Windsurf** 的 Flow 额度消耗极快，在短短几小时内就消耗了超过 10% 的每月配额。
  
  - 他们还将 **DeepSeek R1** 与 **Sonnet 3.5** 进行了对比，强调了部分成功，但呼吁更一致的性能和更智能的额度使用。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek 与 Qwen 的动态二重奏**：一种将 **DeepSeek R1** 与 **Qwen** 结合的集成方法受到了赞赏，一位用户称其在实际性能方面“近乎完美”。
  
  - 社区成员建议在进行任何微调之前进行彻底评估，以避免破坏协同效应，并参考了 [Qwen 2.5 Coder collection](https://huggingface.co/collections/unsloth/qwen-25-coder-6732bc833ed65dd1964994d4)。
- **Unsloth 中的 Multi-GPU 热议与 VRAM 讨论**：成员们确认 **Unsloth** 目前缺乏完整的 Multi-GPU 功能，但预告了未来的推出，以帮助大规模训练并减少单 GPU 瓶颈。
  
  - 他们指出 VRAM 的使用与**模型大小**挂钩，[Unsloth 的文档](https://docs.unsloth.ai/basics/errors#evaluation-loop-also-oom-or-crashing)提供了关于内存限制的见解。
- **“Dolphin-R1” 凭借赞助引起轰动**：创建 **Dolphin-R1** 花费了 6000 美元的 API 费用，促使开发者在 [Hugging Face](https://x.com/cognitivecompai/status/1882140705159799169) 上寻求支持者以 **Apache-2.0** 许可证公开发布。
  
  - 一位赞助商挺身而出，使得该数据集能够与社区共享，同时用户们称赞了其在成本和数据生成方面的透明做法。
- **Striped Hyena 与真核生物探索**：一位 BioML 博士后希望将针对原核生物基因组训练的 **Striped Hyena** 适配到真核生物序列，参考了 [Science ado9336](https://www.science.org/doi/10.1126/science.ado9336) 和 [项目仓库](https://github.com/togethercomputer/stripedhyena/tree/main)。
  
  - 他们强调 **Unsloth** 尚未完全支持将 Multi-GPU 用于大型基因组数据，这引发了关于生物分子 Token 专业化训练方法的讨论。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **DeepSeek 难题与 LM Studio 修复**：用户在加载 **DeepSeek R1** 时遇到了“unknown pre-tokenizer type”等错误，促使了手动模型更新和重新下载。
  
  - 他们参考了 [LM Studio 文档](https://lmstudio.ai/)进行故障排除，并赞扬了针对持续加载失败的快速解决方案。
- **Qwen 量化之争**：小组讨论了 **Q5_K_M** 是 **Qwen** 模型在模型大小和准确性之间的最佳平衡点。
  
  - 更大的参数集似乎能提供更丰富的输出，导致许多人尽管 GPU 需求更高，仍倾向于选择更大的占用空间。
- **LM Studio 中的网络配置困扰**：贡献者呼吁在 **LM Studio** 中提供更清晰的切换选项，以区分仅限 localhost 与跨设备的 all-IPs 访问。
  
  - 他们分享道，模糊的设置阻碍了多设备使用，并强调需要更直接的标记方式。
- **Gemini 2.0 势头强劲**：爱好者们称赞 **Google** 的 **Gemini 2.0 Flash** 具有更长的上下文长度和对法律文件极高准确性的解析能力。
  
  - 与 **o1 mini** 等旧模型的对比突显了 Gemini 更持久的响应和更敏锐的知识保留。
- **RTX 5090 与 Procyon 性能讨论**：**NVIDIA** 的 **RTX 5090** 运行速度比 4090 快约 30%，但对于较小的 LLM，它并未完全利用其 1.7 倍的带宽，正如 [NVIDIA 官方页面](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)所示。
  
  - 建议使用 **Procyon AI** 进行统一的性能测试，强调了在一致的基准测试中模型量化和 VRAM 的使用情况。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 在 Android 上的重大飞跃**：**Perplexity Assistant** 现在可以在 Android 上使用，通过[此链接](https://pplx.ai/android)可以实现跨 App 的任务处理。
  
  - 语音激活仍是一个难点，不过用于识别现实世界物体的新 multimodal 功能引起了人们的兴趣。
- **Mistral 的 IPO 计划引发猜测**：讨论集中在 **Mistral** 计划进行 **IPO**，这引发了对其产品潜在扩张的好奇。
  
  - 一段 [YouTube 视频](https://www.youtube.com/embed/dGQOrroTmTY)重点介绍了这一举动，社区成员就其对未来模型开发的影响展开了辩论。
- **DeepSeek R1 在性能测试中表现强劲**：一些人声称 **DeepSeek R1** 在特定任务中可能超越 OpenAI，并引用了一份[详细探索](https://www.perplexity.ai/page/deepseek-r1-may-beat-openai-s-Xl.Pc5FFSfS9NaIIuLNQaw)。
  
  - 工程师们认为这是竞争加剧的信号，并呼吁进行更严格的对比。
- **Sonar 模型调整策略**：**Sonar** 系列放弃了 Sonar Huge，转而采用 Sonar Large，并暗示将推出 Sonar Pro，引发了关于性能提升的疑问。
  
  - API 中断（包括 **524 错误**）和 **SOC 2 compliance** 查询，凸显了企业级用户对稳定性的广泛担忧。
- **PyCTC Decode 与社区项目**：开发者正在考虑将 *PyCTC Decode* 用于专门的语音应用，并将同行引导至[此链接](https://www.perplexity.ai/search/pyctc-decode-57MLlbc2QbmCmmV1jO6zdw)。
  
  - 与此同时，一个音乐流媒体概念和新鲜的 AI prompt 创意展示了贡献者之间多样化的实验。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Web Search 取得进展**：OpenRouter 推出了 **Web Search API**，价格为 **$4/1k results**。在模型名称后附加 **:online** 时，默认每次请求可获取 **5** 条结果，文档位于 [OpenRouter](https://openrouter.ai/docs/web-search)。
  
  - 他们澄清说每次查询的成本约为 **$0.02**，并开玩笑说提前发布的公告盖过了正式推出的风头。
- **Reasoning Tokens 开放**：OpenRouter 引入了 **Reasoning Tokens** 以直接获取模型思考过程，需要设置 `include_reasoning: true`，如[此推文](https://x.com/OpenRouterAI/status/1882491124402803075)所述。
  
  - 多个思考模型之间的 **finish_reason** 标准化旨在统一解释风格。
- **Deepseek R1 在高负载下表现不稳**：**Deepseek R1** 面临响应问题，偶尔会出现卡顿以及无法从 **Deepseek** 和 **DeepInfra** 返回结果的情况，详见 [DeepSeek R1 Distill Llama 70B](https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b)。
  
  - 一位用户质疑这些问题是源于模型固有的缺陷还是服务中断。
- **额度与集成问题**：一些用户报告了 **API Key** 优先级混淆，导致使用了额度而非预期的 Mistral 集成，而另一些用户则对 **Web Search** 的计费感到困惑。
  
  - 出现了一种变通方案，即 [Crypto Payments API](https://openrouter.ai/docs/crypto-api)，允许用户通过标准支付方式之外的途径购买额度。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 的双 LLM 设置**：社区成员描述了如何通过 `aider.conf.yaml` 配置 **Aider** 运行多个 LLM，并指出除非按照[安装文档](https://aider.chat/docs/install.html)进行精确设置，否则聊天模式默认仅使用单个模型。
  
  - 他们发现 **/chat-mode code** 可能会覆盖独立的编辑器模型，这让那些希望严格控制每个模型角色的用户感到困惑。
- **DeepSeek R1 的语法障碍**：多方反馈 **DeepSeek R1** 在编程任务中面临语法和上下文限制的挑战，如[此演示视频](https://www.youtube.com/watch?v=bOsvI3HYHgI)所示。
  
  - 一些人建议输入更小的上下文片段，并引用“使用部分引用效果更好”作为临时解决方案。
- **Anthropic 的引用功能说明**：Anthropic 推出的新 **Citations API** 在 Claude 的回答中加入了来源链接，详见其[发布公告](https://www.anthropic.com/news/introducing-citations-api)。
  
  - 社区成员赞扬了这种获取可靠引用的简便方法，评论道“这减轻了验证生成文本中来源的麻烦”。
- **大型项目的 Aider 日志记录**：针对大型代码库，参与者通过将 **Aider** 的提示词输出重定向到文件来处理，从而节省 Token 并减少混乱。
  
  - 他们提到“重定向繁重的终端命令”是一种很有帮助的工作流，可以在捕获详细日志的同时保持界面整洁。
- **JetBrains AI 等候名单热议**：技术人员纷纷加入 **JetBrains AI** 的等候名单，希望这位 IDE 领导者在经历早期的挫折后，能与 **Cursor** 和 **Windsurf** 展开竞争。
  
  - 尽管有人批评 JetBrains AI 之前的尝试，但仍坚持认为“无论 AI 功能如何，JetBrains 仍然是我首选的开发套件”，这得益于其强大的 IDE 功能。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Operator 的强势登场与 200 美元定价**：太平洋时间上午 10 点，**Sam Altman** 及其团队在 [YouTube 演示](https://www.youtube.com/watch?v=CSE77wAdDLg)中介绍了 **Operator**，订阅费用为每月 **200 美元**。
  
  - 社区对其**浏览器集成**功能感到兴奋，并期待未来能扩展到由用户驱动的浏览器选择。
- **DeepSeek R1 在编程方面力压 O1**：多项对比测试显示 **DeepSeek R1** 在编程任务中超越了 **O1**，甚至能流畅处理像 'POTATO THUNDERSTORM!' 这样的随机提示词。
  
  - 社区成员强调了其**更强的代码解决方案**并赞扬了 R1 的灵活性，预言未来会有更多激烈的对比。
- **GPT 停机与语音功能崩溃**：服务中断导致 **GPT** 抛出 'bad gateway' 错误并禁用了语音功能，[OpenAI 状态页](https://status.openai.com/)对此进行了追踪。
  
  - 用户开玩笑地指责 **LeBron James** 和 **Ronaldo**，而官方更新表明正在持续修复以恢复**语音功能**。
- **Perplexity Assistant 在移动端势头强劲**：多位用户称赞 **Perplexity Assistant** 在**移动端**比现有的 OpenAI 应用更高效，引发了关于用户满意度的讨论。
  
  - 他们批评了 **OpenAI 的定价**，暗示如果替代方案在便携性上继续超越 **ChatGPT**，用户忠诚度可能会发生转移。
- **脉冲神经网络（Spiking Neural Networks）引发复杂反应**：参与者考虑将**脉冲神经网络**用于提高能效，但担心延迟问题，并指出在实际实现中收益尚不明确。
  
  - 有人将其视为死胡同，也有人认为它是下一步的发展方向，这引发了关于哪些特定任务可能从**脉冲模型**中受益的进一步探讨。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **OpenAI Operator 提供自动化操作**：OpenAI 正在准备一项名为 [**Operator** 的新 ChatGPT 功能](https://x.com/steph_palazzolo/status/1882091855606895073/)，它可以在用户的浏览器中执行操作，并允许保存或共享任务。
  
  - 社区成员预计该功能将于**本周**发布，并指出虽然目前在 **API** 中尚不可用，但它可能会塑造自动化用户工作流的新方式。
- **R1 Qwen 2.5 32B 对 VRAM 要求极高**：根据参数量估算讨论，以 float16 格式运行 **R1 Distilled Qwen 2.5 32B** 至少需要 **64GB VRAM**，而 q8 版本则需要 **32GB**。
  
  - 讨论强调了 16-bit 的 **7B 参数** 需要约 **14B 字节** 的内存，此外还需要加上**上下文窗口 (context windows)** 的额外开销。
- **GSPN 为视觉带来 2D 变革**：新的 [Generalized Spatial Propagation Network (GSPN)](https://arxiv.org/abs/2501.12381) 承诺提供一种针对**视觉任务**优化的 2D 能力注意力机制，能够更有效地捕获**空间结构**。
  
  - 成员们称赞了 **Stability-Context Condition**，它将有效序列长度缩减至 **√N**，并有可能提高**图像数据**中的上下文感知能力。
- **MONA 方法最小化多步奖励作弊 (Reward Hacking)**：一种提出的 RL 方法 [MONA](https://arxiv.org/abs/2501.13011)，通过使用短视优化 (short-sighted optimization) 结合远见检查 (far-sighted checks) 来遏制**多步奖励作弊**。
  
  - 研究人员在易发生奖励作弊的场景中测试了 **MONA**，展示了在**强化学习 (reinforcement learning)** 设置中防止非预期行为的潜力。
- **IntellAgent 通过模拟对话评估 Agent**：[**IntellAgent** 项目](https://github.com/plurai-ai/intellagent) 提供了一个开源框架，用于生成和分析 Agent 对话，捕获细粒度的**交互细节**。
  
  - 伴随着[研究论文](https://arxiv.org/pdf/2501.11067)的发布，早期采用者对这种稳健的 Agent 评估方法表示欢迎，该方法侧重于一个能够指出**对话缺陷**的批判组件。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Evabyte 的压缩分块注意力 (Compressed Chunked Attention)**：新的 **Evabyte** 架构依赖于一种具有多字节预测的全分块线性注意力设计，如[此代码片段](https://github.com/OpenEvaByte/evabyte/blob/ba8f65c5fe502b7ed07f916773754734b91b52fd/evabyte_hf/eva.py#L63)所示。
  
  - 工程师们指出了其压缩的内存占用和提高吞吐量的潜力，通过内部的 **attention** 草图展示了其大规模效率。
- **Tensorgrad 扭转张量操作**：来自 [GitHub](https://github.com/thomasahle/tensorgrad) 的 **tensorgrad** 库引入了命名边 (named edges) 以实现用户友好的张量操作，支持如 `kernel @ h_conv @ w_conv` 这样无需复杂索引的命令。
  
  - 它提供**符号推理 (symbolic reasoning)** 和矩阵简化，利用前向和反向传播中的**公共子表达式消除 (common subexpression elimination)** 来提升性能。
- **R1 数据集出现，访问难题依然存在**：参与者确认用于蒸馏模型的 **R1** 数据集已部分可访问，但具体的下载位置细节仍不明确。
  
  - 好奇的研究人员请求直接的仓库链接，希望 **Nous Research** 的官方澄清能解决这一困惑。
- **大脑与比特：MIT 的表示收敛**：MIT 研究人员观察到，在自然主义输入上训练的**人工神经网络**与生物系统趋同，如[这项研究](https://www.biorxiv.org/content/10.1101/2024.12.26.629294v1)所示。
  
  - 他们发现**模型与大脑的对齐 (model-to-brain alignment)** 与视觉和语言刺激下的跨模型一致性相关，这表明某些神经计算存在通用基础。
- **通过 TREAD Token 路由获得 Diffusion 增益**：最近的一篇论文 [TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training](https://arxiv.org/abs/2501.04765) 通过保留 token 信息而非丢弃来解决样本效率低下的问题。
  
  - 作者声称在更深层中增加了集成度，适用于 **Transformer** 和状态空间 (state-space) 架构，从而降低了 Diffusion 模型的计算成本。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Stripe-Supabase 历险记**：一位用户在将 [Stripe webhook](https://x.com/boltdotnew/status/1882483266680406527) 对接到 **Supabase edge function** 时遇到了 **401 错误**，最终定位为 **verify_jwt** 配置错误。
  
  - 他们修正了 **JWT config**，并正在查阅 [官方文档](https://bolters.io/docs/read-this-first) 以巩固集成。
- **Token 纠纷**：在从免费计划切换到付费计划后，用户注意到 **token 配额** 从 **300k** 降至 **150k**，引发了对每日配额的困惑。
  
  - 一些人推测付费计划取消了自动 token 续期，促使他们重新查看 [StackBlitz 注册页面](https://stackblitz.com/register) 以求明确。
- **Bolt Chat 故障**：社区成员报告聊天记录消失，需要完全重置 **StackBlitz** 才能尝试修复。
  
  - 他们讨论了持久化会话策略，并引用 [bolt.new](https://bolt.new/?autoAuth) 寻找可能的解决方案。
- **3D 显示难题**：一位用户尝试使用 GLB 文件构建 **3D 模型查看器** 时出现白屏，表明缺少或未完成设置步骤。
  
  - 指南推荐使用 **Google Model Viewer** 代码，并建议参考 [Cursor Directory](https://cursor.directory/) 中的进一步资料。
- **Discord 支付提案**：一位用户提议增加 **Discord 登录** 功能和新的 **webhook 接收** 系统，以简化 **Bolt.new** 的支付流程。
  
  - 他们还提到通过邀请好友获得 **token 抽奖** 奖励，旨在通过额外福利提升社区参与度。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Operator 掌舵数字世界**：OpenAI 推出了 **Operator**，这是一个可以自主导航浏览器执行任务的 Agent，详见其 [博客文章](https://openai.com/index/introducing-operator/)。
  
  - 观察者测试了其研究任务能力，并对 **CAPTCHA** 循环表示担忧，同时参考了开源类似项目如 [Browser Use](https://x.com/hwchase17/status/1882502767312531954)。
- **Imagen 3 超越 Recraft-v3**：Google 的 **Imagen 3** 夺得文生图榜首，在 [Text-to-Image Arena 排行榜](https://x.com/lmarena_ai/status/1882164189739073990) 上领先 Recraft-v3 达 70 分。
  
  - 社区成员强调了其精细的 Prompt 处理能力，包括一个令旁观者印象深刻的“海滩上的水母”场景细节。
- **DeepSeek RAG 简化复杂度**：**DeepSeek** 通过允许直接摄取大量文档来重新定义检索增强生成（RAG），如 [讨论](https://x.com/pelaseyed/status/1882471632129994914) 中所述。
  
  - KV caching 提升了吞吐量，促使一些人宣称在大规模用例中，传统的 RAG 是一种“反模式”（anti pattern）。
- **Fireworks AI 低价转录服务**：**Fireworks AI** 在免费期后推出了价格为每分钟 0.0032 美元的流式转录工具，详见其 [公告](https://x.com/FireworksAI_HQ/status/1882530477468459309)。
  
  - 他们声称拥有接近 *Whisper-v3-large* 的质量和 300ms 的延迟，使其成为实时字幕领域极具性价比的选择。
- **API 收入分成引发好奇**：参与者注意到 **OpenAI** 不会将 API 使用量计入 ChatGPT 订阅额度，从而引发了关于收入分配的疑问。
  
  - 他们想知道是否有供应商会根据用户的 API 活动支付报酬，但未发现此类安排的证据。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **R1 的真实风险与 DDoS 困境**：社区成员对 **R1** 提出了警示，原因是其容易被越狱（jailbreak）并可能生成 **DDoS** 代码，并提到操纵一般的 AI 系统是多么简单。
  
  - 一些成员分享了 [TDE Podcast #11](https://www.youtube.com/live/DxnBT5ChEPE?si=ptEfDoyJzHYmTFhp) 的链接，该节目解释了**端到端 LLM** 解决方案，并思考这些漏洞是否可以通过更强大的代码检查来缓解。
- **Triton 棘手的步骤排序**：一位贡献者发现 **step2** 必须在 **step3** 之前运行，以避免数据覆盖问题，并指出 **step3** 会以影响最终结果的方式间接改变 **x_c**。
  
  - 他们建议直接在 **x** 而不是 **x_c** 上测试更改以提高清晰度，强调了在 multipass kernels 中变量操作的微妙影响。
- **CUDA 12.8 与 Blackwell 收益**：NVIDIA 发布了 [CUDA 12.8](https://developer.nvidia.com/cuda-downloads)，其特点是支持 **Blackwell** 架构以及新的 **FP8**/**FP4** TensorCore 指令。
  
  - 开发者们还提到了用于 GPU 仿真的 [Accel-Sim](https://accel-sim.github.io/#overview)，以及一条关于**第五代** TensorCore 指令的推文，引发了关于性能指标提升的辩论。
- **ComfyUI 招聘 ML 工程师**：ComfyUI 正在为其开源生态系统招聘 **machine learning engineers**，其优势在于 **VC-backed**（风投支持）模式以及来自湾区的**宏大愿景**。
  
  - 感兴趣的人员可以在 [职位列表](https://comfyorg.notion.site/Founding-Machine-Learning-Engineer-1696d73d36508014bfbaf5aebf39b145) 中了解更多关于该角色的信息，团队强调了来自几家顶尖公司的首日模型支持。
- **Tiny GRPO 与 Reasoning Gym 发布**：开发者们在 [GitHub](https://github.com/open-thought/tiny-grpo) 上发布了 **Tiny GRPO** 仓库，用于极简、可高度定制的实现，并鼓励社区贡献。
  
  - 他们还启动了 [Reasoning Gym](https://github.com/open-thought/reasoning-gym)，专注于过程推理任务，邀请社区提出新的数据集想法和扩展建议。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **超时难题解决与 Windows 烦恼**：一位用户克服了 **60 秒** 的 MCP 服务器超时限制，虽然他们没有分享具体做法，但这引起了其他人的兴趣。
  
  - 另一位用户解决了 **Windows** 上隐藏的 PATH 设置问题，并在 Access 中创建了一个 *test.db* 文件，参考 [MCP Inspector tool](https://github.com/modelcontextprotocol/inspector) 确认了稳定性。
- **容器之争：Podman 对阵 Docker**：成员们辩论了 **Podman** 与 **Docker** 的优劣，参考 [Podman 安装步骤](https://podman.io/docs/installation) 进行更简单的设置。
  
  - 虽然 Podman 是 daemonless（无守护进程）且更轻量级，但许多开发者由于熟悉度和更广泛的工具集成而继续使用 Docker。
- **用于精准编辑的代码行号**：一位用户展示了一种在代码中跟踪**行号**的方法，以便应用针对性的更改，称其比旧的基于 diff 的方法更高效。
  
  - 通过强调在*大型重构任务*中提高的可靠性，社区发现这对于**复杂**的代码合并来说是一种更简单的方法。
- **Anthropic TS 客户端受挫与 SSE 示例修复**：**Anthropic TS client** 的一个已知 bug 导致一些开发者转向 Python，正如 [issue #118](https://github.com/modelcontextprotocol/typescript-sdk/issues/118) 中所提示的。
  
  - 一位用户承认在 SSE 示例中存在*复制粘贴错误*，并链接了一个[修正后的 clientSse.ts 示例](https://github.com/apify/actors-mcp-server/blob/master/src/examples/clientSse.ts)以澄清自定义 header 的用法，同时也回答了关于 Node 的 **EventSource** 可靠性的问题。
- **Puppeteer 赋能网页交互**：一个新的 [mcp-puppeteer-linux package](https://www.npmjs.com/package/mcp-puppeteer-linux) 为 LLM 带来了**浏览器自动化**能力，支持导航、截图和元素点击。
  
  - 社区成员称赞了其 JavaScript 执行功能，称其为**基于 Web** 的测试工作流的潜在游戏规则改变者。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 稳步成长**：新的 [GPT4All v3.7.0 版本](https://docs.gpt4all.io) 包含了针对 Qualcomm 和 Microsoft SQ 设备的 **Windows ARM 支持**，尽管用户目前必须注意其仅限 **CPU-only** 运行。
  
  - 讨论重点关注了 **macOS** 的崩溃修复，并建议卸载任何基于 GitHub 的临时解决方案，以恢复到官方版本。
- **Code Interpreter 弥补缺陷**：**Code Interpreter** 进行了升级，改进了超时处理，并为多个参数提供了更灵活的 console.log 用法。
  
  - 工程师们称赞这符合 **JavaScript** 的预期，强调了更简单的调试和更流畅的开发者工作流。
- **聊天模板解析问题得到解决**：修复了 **chat template parser** 中的两个崩溃和一个兼容性故障，为 EM German Mistral 和五个新模型提供了稳定性。
  
  - 几位成员引用了 [GPT4All 关于聊天模板的文档](https://docs.gpt4all.io/gpt4all_desktop/chat_templates.html#how-do-i-customize-the-chat-template-or-system-message) 以获取更深入的配置和故障排除技巧。
- **提示词工程的礼貌性有所回报**：爱好者们认为，精炼的请求（包括使用 'Please' 等礼貌用语）可以提高 **GPT4All** 的响应能力。
  
  - 他们还调侃了无限使用 ChatGPT 的 **pay-to-play**（付费即玩）现状，鼓励同事们探索替代方案。
- **NSFW 和 Jinja 担忧**：社区成员提到了 **NSFW 内容** 的障碍，指出道德过滤器和审查器（zensors）会阻止露骨内容的输出。
  
  - 其他人注意到基于 C++ 的 GPT4All 集成中 **Jinja template** 的复杂性，这使得采用自定义语法变得困难。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **CitiVAI 的快速停机**：成员们表示 **CitiVAI** 每天会宕机几次，并引用了 [r/StableDiffusion 上的用户体验](https://www.reddit.com/r/StableDiffusion/s/7hEw9MOp9D)，这导致了偶发性的图像生成限制。
  
  - 他们解释说这些间隔是计划内维护的一部分，一些人建议发布公告时间表，以便更好地围绕停机时间进行规划。
- **冰雪遮罩魔法**：一位用户分享了他们如何将黑白遮罩层与 **Inkscape** 结合来制作冰雪主题的文本，然后使用 canny controlnet 或直接提示词进行上色。
  
  - 其他人讨论了诸如 [SwarmUI 的模型支持文档](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md) 之类的参考资料，以便更好地集成基于自定义遮罩的生成方法。
- **5090 GPU 的性能提升与代价**：讨论显示，据报道 **5090 GPU** 的渲染速度提高了 20-40%，但功耗增加了 30%，而更深层次的优势体现在 **B100/B200** 系列中。
  
  - 参与者指出了诸如 [在 RTX 2070 消费级 GPU 上的微调结果](https://www.reddit.com/r/StableDiffusion/comments/14jck90/finetuning_sdxl_on_an_rtx_2070_consumer_tier_gpu/) 等数据，表明 Stable Diffusion 任务有持续的改进。
- **训练卡通角色**：爱好者们研究了复制 **电影角色** 的微调，参考了 [来自《鼠来宝》电影的 Alvin LoRA 模型](https://civitai.com/models/981021/alvin-seville-alvin-and-the-chipmunks-movie)。
  
  - 他们指出这种方法只需要中端 GPU 和一点时间，并呼应了 [SwarmUI 的提示词语法技巧](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Features/Prompt%20Syntax.md#automatic-segmentation-and-refining) 中的示例。
- **Clip Skip 逐渐淡出**：一位用户询问 'clip skip' 是否仍然相关，发现它是 **SD1** 演进过程中的遗留物，现在很少使用。
  
  - 小组得出结论，对于现代 Stable Diffusion 设置，它通常是不必要的，并强调高级提示词工作流已经取代了那种旧配置。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Google 的 Titans 展示下一代内存**: Google 推出了 **Titans**，承诺提供更强大的推理时内存 (inference-time memory)，如[这段 YouTube 视频](https://www.youtube.com/watch?v=x8jFFhCLDJY)所示。
  
  - 小组指出该论文难以复现，关于确切的 Attention 策略仍存在疑问。
- **Egomotion 助力特征学习**: 研究人员在[这篇论文](https://arxiv.org/abs/1505.01596)中测试了 **egomotion** 作为一种自我导向方法，用移动数据取代标签。
  
  - 他们观察到场景识别和目标检测的强劲结果，引发了对基于运动训练的兴趣。
- **分布动态规划 (Distributional Dynamic Programming) 势头强劲**: 一种名为 **distributional dynamic programming** 的新方法解决了收益分布的统计泛函问题，详见[这篇论文](https://arxiv.org/abs/2501.13028)。
  
  - 它具有 stock augmentation 功能，可以扩展曾经用标准 Reinforcement Learning 方法难以处理的解决方案。
- **Ruler 任务扩展长上下文可能性**: 所有 **Ruler tasks** 已完成最终定稿并修正了少量格式问题，鼓励在 [#lm-thunderdome 频道](https://discord.com/channels/729741769192767510/755950983669874798/1331757642640789544)中进行更多长上下文应用。
  
  - 贡献者请求增加更多的 **long context tasks**，强调努力突破现实世界测试的边界。

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **开源 RAG 势头强劲**: 开发者们探索了一份[详细指南](https://t.co/kkr77KA23P)，使用 **LlamaIndex**、**Meta Llama 3** 和 **TruLens** 构建开源 **RAG 系统**，并将基础方法与 **Neo4j** 以及更具代理性的 **agentic setup** 进行了对比。
  
  - 他们将 **OpenAI** 与 **Llama 3.2** 进行性能对比，激发了对自托管和灵活解决方案的热情。
- **面向社交平台的 AI Chrome 扩展**: 成员们讨论了[一对 **Chrome extensions**](https://t.co/8T9bFBD0Cl)，它们利用 **LlamaIndex** 来提升 **X** 和 **LinkedIn** 帖子的影响力。
  
  - 他们称赞这些 AI 工具在提高参与度的同时扩展了内容创作的可能性。
- **AgentWorkflow 的重大提升**: 爱好者们赞扬了 **AgentWorkflow** 的升级，强调其速度和输出质量优于旧版本。
  
  - 多个项目转向了这些新功能，认为这次改进消除了之前的 **bottlenecks**（瓶颈）。
- **多 Agent 混战 vs 工具**: 讨论明确了多个 **agents** 如何按顺序激活，利用异步工具调用 (async tool calls) 而不破坏彼此的上下文 (context)。
  
  - 他们还澄清了 **agents** 依赖工具，但其自身也可以在专门角色中作为工具使用。
- **内存管理与链接故障**: 参与者呼吁更好的内存模块，指出 **ChatMemoryBuffer** 可能无法优化上下文使用，且摘要可能会增加延迟。
  
  - 一个失效的 [Agent 教程](https://ts.llamaindex.ai/docs/llamaindex/getting_started/starter_tutorial/agent)导致了 500 错误，促使他们参考 [run-llama GitHub 仓库](https://github.com/run-llama)获取核心文档。

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 幽默的 LCoT 猜想**: 一名成员敦促 **Cohere** 发布能够处理逻辑和思考的 LCoT meme 模型权重，得到的回复是提醒 Cohere 专注于企业级业务。
  
  - 他们分享了一个有趣的 [GIF](https://tenor.com/view/jeff-bridges-agent-champagne-kingsman-golden-circle-toss-gif-9381860)，以强调社区对更多开放实验的渴望。
- **Pydantic 与 Cohere 的完美结合**: 一位用户宣布 **Pydantic** 现在支持 **Cohere models**，引发了开发者对简化集成的兴奋。
  
  - 这一更新可能会简化任何使用 **Cohere** 构建应用的开发者的工作流和编码实践，尽管尚未透露更多发布细节。
- **Chain of Thought 讨论**: 参与者提出了诸如 *'think before you act'* 和 `<thinking></thinking>` 等提示词，以模拟 **Chain of Thought** 推理。
  
  - 他们注意到，即使是缺乏显式 trace 训练的 **regular models**，通过结构良好的 Prompt 仍能获得部分推理优势。
- **Reranker 难题：本地部署的梦想**: 一位智利用户询问关于 **Cohere Reranker** 的本地部署 (on-prem hosting) 问题，以抵消来自南美的高延迟。
  
  - 目前尚未出现直接的解决方案，建议他们通过 [support@cohere.com](mailto:support@cohere.com) 联系 **sales** 团队寻求替代方案。
- **ASI 的雄心与忧虑**: 讨论涵盖了 **Artificial Superintelligence (ASI)** 可能超越人类智力的话题，强调了在 **healthcare** 和 **education** 领域的潜在突破。
  
  - 成员们表达了对滥用行为的伦理担忧，并指出目前没有官方的 **Cohere** 文档涉及 ASI 开发。

 

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 助力学习成效提升**：一位成员分享了将 **NotebookLM** 集成到学习工作流中的兴奋之情，并发布了一段 [YouTube 视频](https://youtu.be/wvf4EXJJsU8)，重点介绍了其中实用的笔记整理功能。
  
  - 他们还发现了 **Obsidian** 插件如何有效地合并 Markdown 笔记，引发了关于优化知识共享实践的讨论。
- **播客深度解析 DeepSeek-R1**：一位用户发布了一集**播客节目**，深度解析了 [DeepSeek-R1 论文分析](https://open.spotify.com/episode/5T8cbCKks1RE4RxZ0aFBMD?si=TEVGEhl1SWqFy9KRlW17Pg)，探讨了该模型的推理能力和基于 RL 的改进。
  
  - 他们强调了强化学习如何塑造小型模型的发展，激发了其他人对规模化策略的探索。
- **NotebookLM 语言切换困扰**：用户在尝试将 **NotebookLM** 从罗马尼亚语切换为英语时遇到中断，尝试使用 URL 参数却导致了错误。
  
  - 社区成员寻求官方的语言更新方法，但困惑依然存在。
- **高质量测试题生成**：一位参与者介绍了一种在 **NotebookLM** 中根据指定章节生成多项选择测试题的固定模式。
  
  - 他们认为这种方法能够确保持续的成功，从而简化了备考过程。
- **音频故障与文档交叉检查**：成员们遇到了音频生成的小问题，包括在 Prompt 缺乏细节时倾向于提取整个 PDF 的内容，还有一些人报告了下载文件的播放问题。
  
  - 他们还辩论了 **NotebookLM** 在分析法律文档方面是否超越了 **ChatGPT**，并指出交叉引用如何揭示非典型条款。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 走向异步**：在 #general 频道中，出现了一个关于[异步代码的新论坛帖子](https://forum.modular.com/t/how-to-write-async-code-in-mojo/473)，突显了社区对协程（coroutines）的兴趣，尽管目前官方封装还很有限。
  
  - 成员们对直接分享链接表示欢迎，鼓励进一步讨论代码模式和使用示例。
- **MAX Builds 页面活跃度提升**：[MAX Builds 页面](https://builds.modular.com)现在开始展示社区构建的包，重点推介基于 Mojo 项目的扩展。
  
  - 贡献者在发布时会获得认可，任何人都可以向 [Modular 社区仓库](https://github.com/modular/modular-community)提交 recipe.yaml 以供收录。
- **没有 Override？没关系！**：一场 #mojo 讨论确认了 Mojo 中没有 @override 装饰器，一位成员澄清说 struct 本身就不支持继承。
  
  - 这意味着函数重定义无需特殊语法即可进行，这促使了更注重细节的代码审查。
- **生成器引发讨论**：关于 Python 风格生成器的问题被提出，并指出许多编译语言中都存在这一空白。
  
  - 参与者建议了一个需要显式暴露 yield 的异步提案，推动未来在协程方面的增强。
- **重新赋值与 iadd 辩论**：开发者们讨论了函数定义中的只读引用，区分了 mut 与 owned 的用法。
  
  - 他们还探索了 **iadd** 如何支撑 +=，澄清了 Mojo 中的组合行为。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Bud-E 的情感 TTS 首秀**：**情感开源 TTS** 即将加入 **Bud-E**，分享的音频片段展示了该方法的进展。
  
  - 成员们赞扬了其富有表现力的音域，称其为*“音频项目迈出的令人兴奋的一步”*，并期待 **Bud-E** 的进一步扩展。
- **使用 pydub 剖析失真**：一位研究人员正在使用 **pydub** 对比**原始音频文件**与高噪声变体的波形，重点关注轻微与极端失真水平的差异。
  
  - 他们分享了突出轻微与强噪声差异的图像，展示了音频探索方面的改进。
- **协作式 Colab Notebook**：成员们提议通过 [Google Colab notebook](https://colab.research.google.com/drive/140lGFiXXeTsNFp7w5xteCjpmRRaSBvmj) 进行 **Notebook 共享**，以共同优化围绕**音频转换**的代码。
  
  - 参与者表示有兴趣复现该方法，并为进一步优化提出了建议。
- **用于波形对比的组件**：在 Colab 中请求 **IPython 音频组件**，旨在简化失真前后的评估。
  
  - 成员们集思广益讨论了潜在的代码片段，强调了在共享 Notebook 中实现更简单的播放控制和侧边对比。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **仓库垃圾信息引发骚乱 (Repo Spam Sparks Commotion)**：有关 **repo** 被灌水的担忧浮出水面，推测这与 **coin** 问题有关，并将其描述为“非常差劲”。
  
  - 一些参与者否认了这种关联，将重点转向加强内容管理工作。
- **框架灵感胜过模仿 (Framework Inspiration Over Imitation)**：一位用户敦促避免严格复制现有框架，强调针对特定解决方案的 **use-case** 对齐。
  
  - 他们提倡围绕实际目标构建工具包，而不是依赖他人的方法。
- **DSPy 中由邮件触发的 REACT Agent**：一名开发者希望通过邮件触发运行 **REACT agent**，并最终通过使用 **webhook** 成功实现。
  
  - 他们提到 DSPy 已准备好支持外部库，强调了灵活的“触发器到 Agent”工作流。
- **OpenAI 模型获得青睐，Groq 仍在参与**：一位贡献者称赞 **OpenAI model** 的广泛覆盖范围和在各项任务中的实用性。
  
  - 另一位贡献者提到了 **Groq** 的兼容性，表明了对多种硬件后端的兴趣。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **领取悬赏：llvm_bf16_cast 取得进展**：一位贡献者确认了 **llvm_bf16_cast** 的悬赏状态，并在几小时前提交了 PR，有效解决了重写请求。
  
  - 注意力现在转向新任务，确保有一系列 **tinygrad** 悬赏任务用于进一步的 GPU 优化。
- **ILP 在 Shapetracker 中登场**：一名成员展示了一种针对 **shapetracker add problem** 的 [基于 ILP 的方法](https://cdn.discordapp.com/attachments/1068976834928193609/1332064522495725700/viewadd.pdf)，尽管它在速度上存在困难且需要外部求解器。
  
  - 尽管如此，这种结构化的 shape 处理方式可能为 **tinygrad** 中更精确的重写操作铺平道路。
- **George Hotz 支持基于 ILP 的重写简化**：George Hotz 对 ILP 方法产生了兴趣，询问是否有 PR，并暗示可能在 **tinygrad** 重写规则中进行集成。
  
  - 此举可能会推动 **tinygrad** 采用线性规划来实现更高效的变换。
- **Mask 与 View 的碰撞：合并策略显现**：参与者讨论了合并 **masks** 和 **views**，建议有界表示（bounded representation）可以增强 mask 的能力。
  
  - 他们承认复杂性有所增加，但仍对融合 mask 以扩展 shape 灵活性持开放态度。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **证书时间表尚不确定，MOOC 仍开放**：一位参与者询问了 **course certificates** 以及跟踪发放的方式，但尚未提供官方时间表，引发了好奇。
  
  - 另一位参与者不确定 **LLM MOOC enrollment** 情况，发现只需填写表格即可确认参加。
- **Agent 期待掌握课程**：一位参与者指出，作为 **LLM agent** 会自动获得课程访问权限，这突显了成功的高门槛。
  
  - 他们建议，任何通过考试的 Agent 都会获得极大的公信力，这反映了 **LLM training** 的先进性。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **BFCLV3：巨大的工具之谜**：有人提出疑问，**BFCLV3** 是否提供了一个系统消息，概述了在调用 **book_flight** 之前，像 **get_flight_cost** 和 **get_creditcard_balance** 这样的工具是如何互连的。
  
  - 成员们观察到，在标记为 *simple*、*parallel*、*multiple* 和 *parallel_multiple* 的任务中，没有关于工具依赖关系的元数据，并链接到 [GitHub source](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/data) 以获取更多细节。
- **LLM 测试方法论受到审视**：参与者辩论了 **BFCLV3** LLM 是纯粹根据工具描述进行测试，还是考虑了底层的依赖关系。
  
  - 他们指出，理解这些关系对于研究至关重要，因为引用 **BFCLV3 dataset** 的细节可以揭示现实世界中函数调用的使用情况。

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **KTO-Liger 合并**: **KTO loss** 已合并至 [Liger-Kernel 仓库](https://github.com/linkedin/Liger-Kernel/pull/475)，有望提升模型性能并带来新功能。
  
  - 社区成员对 **KTO loss** 及其即时收益表示兴奋，期待更强的训练稳定性和改进的泛化能力。
- **Office Hours 倒计时**: 发布了 **Office Hours** 将在 **4 小时**后开始的提醒，旨在为问题解答和设计审查提供互动论坛。
  
  - 参与者可以通过此 [Discord 活动链接](https://discord.gg/dEfsYQbX?event=1328556620107743343)加入，期待围绕进行中的 LLM 项目展开热烈交流。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **2 月 18 日多伦多 MLOps 聚会**: 一场 **MLOps** 活动定于 **2 月 18 日**在**多伦多**举行，面向高级工程师和数据科学家，提供交流领域见解的空间。
  
  - 组织者提到参加者应*私信获取更多详情*，强调重点在于职业社交和知识共享。
- **资深技术专家的社交热潮**: 此次聚会中心在于加强**高级工程师**和**数据科学家**之间的联系，鼓励同行支持和资源共享。
  
  - 参与者认为这是加深社区联系、促进当地 AI 生态系统协作的有益方式。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Local-First 黑客松落地旧金山**: 一场 **Local-First X AI 黑客松**定于 **2 月 22 日**在 [旧金山](https://www.lofihack.com/) 举行，重点展示结合本地计算与 **Generative AI** 的项目。
  
  - 组织者强调了参与者之间的实际协作，并引导他们前往 [活动讨论帖](https://discord.com/channels/1089876418936180786/1329529625189154826) 进行想法交流和资源共享。
- **社区头脑风暴火热进行**: 一个[专门的讨论帖](https://discord.com/channels/1089876418936180786/1329529625189154826)鼓励参与者分享实验策略和隐私保护机器学习框架。
  
  - 策划者希望通过邀请本地计算爱好者在黑客松期间展示原型和进行代码冲刺（code jam），来促成*现实世界的成果*。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Deepspeek 与 OpenInterpreter 的联系**: 一位用户询问关于将 **Deepspeek** 集成到 `>interpreter --os mode` 的事宜，希望通过语音功能使 **OpenInterpreter** 受益。
  
  - 他们提到了 **Deepspeek** 与 OS 级解释器能力之间的潜在协同效应，但未提供进一步的技术细节或链接。
- **OS 模式或将扩展语音功能**: 参与者推测未来的 **OS 模式**增强功能将适配 **OpenInterpreter** 中的语音操作。
  
  - 尽管计划尚不明确，但 **Deepspeek** 的集成可能会开启高级语音支持和某种程度的系统交互。

---

Torchtune Discord 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

HuggingFace Discord 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

AI21 Labs (Jamba) Discord 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

### **Cursor IDE ▷ #**[**general**](https://discord.com/channels/1074847526655643750/1074847527708393565/1331719601297428553) (655 messages🔥🔥🔥):

> `DeepSeek R1, OpenAI O1, Chat vs Composer Mode, AI Agentic Models, Usage-Based Pricing`

- **DeepSeek R1 的性能与未来**: 用户讨论了 **DeepSeek R1** 在 **Composer 普通模式**下性能缓慢的问题，但指出其在正常运行时结果令人满意。
  
  - 虽然一些人发现 R1 在调试方面很有效，但他们对其响应速度以及在处理 Bug 方面不如 **Sonnet** 有效表示沮丧。
- **OpenAI O1 订阅困惑**: 关于 **OpenAI O1 的定价**存在困惑；一些人注意到 **Pro 版本**每月花费 **$200**，而其他人则在使用更便宜的替代方案。
  
  - 讨论强调了 **DeepSeek** 被视为一种更具成本效益的替代方案，提供了极高的性价比。
- **Chat 模式的优势**: **Chat 模式**被用户强调为与 AI 交互的更好选择，允许在应用更改之前审查和讨论代码。

- 相反，**Composer** 因在用户代码上“横冲直撞”以及在缺乏足够 context 的情况下进行随意修改而受到批评。
- **AI 工具的增长与用户情绪**：参与者注意到对新兴 AI 工具的兴趣有所增加，一些人指出技术炒作与普通用户对这些工具的理解之间存在脱节。
  
  - 用户呼吁改进像 R1 这样的 AI 模型的**功能性**和响应速度，以满足用户预期。
- **对按量计费的担忧**：用户对 AI 服务的**按量计费（usage-based pricing）**表示怀疑，质疑仅仅为了追踪 API 调用而支付额外费用的必要性。
  
  - 社区正在推动更高的透明度和更有效的模型，使其在实现基础功能时不需要过高的成本。

**提到的链接**：

- [LiveBench](https://livebench.ai/#/?Reasoning=a&Coding=a&Mathematics=a&IF=a): 未找到描述
- [Agent Mode in Warp AI | Warp](https://www.warp.dev/warp-ai): 一个除了传统命令外还能理解纯英文的命令行界面。使用 Agent Mode 来完成多步骤工作流。
- [Tweet from Chubby♨️ (@kimmonismus)](https://x.com/kimmonismus/status/1882167352315486507): https://x.com/skirano/status/1881854481304047656/video/1 天才！“你可以仅从 deepseek-reasoner 中提取推理过程，这意味着你可以将该思考过程发送给你想要的任何模型...”
- [Tweet from Aidan Clark (@_aidan_clark_)](https://x.com/_aidan_clark_/status/1882135220738220131?s=46): o3-mini 第一次尝试，无需修改，耗时 20 秒（还告诉了我如何转换为 gif...）太兴奋了 :) 引用 Ivan Fioravanti ᯅ (@ivanfioravanti) 👀 DeepSeek R1（右）碾压了 o1-pro（左） 👀 提示词：“wri...
- [Tweet from GREG ISENBERG (@gregisenberg)](https://x.com/gregisenberg/status/1882064374120268234): 我刚刚意识到 DeepSeek R1 让推理变得比一杯咖啡还便宜，而且它是开源的（不像 GPT4），并且在某些方面超越了 Claude 3.5 Sonnet。“中国制造”的 AI 现在成本为 $0.50/小时，而美国...
- [Tweet from swyx /dd (@swyx)](https://x.com/swyx/status/1881141889283588495/photo/2): 我收回之前对 @warpdotdev 说过的所有负面评价。这东西能解决 Python 依赖地狱。我现在只需狂按回车，它就在帮我修复环境以运行 @home_assistant...
- [Joe Biden Presidential Debate GIF - Joe biden Presidential debate Huh - Discover & Share GIFs](https://tenor.com/view/joe-biden-presidential-debate-huh-confused-gif-9508832355999336631): 点击查看 GIF
- [Tweet from Aidan Clark (@_aidan_clark_)](https://x.com/_aidan_clark_/status/1882135220738220131): o3-mini 第一次尝试，无需修改，耗时 20 秒（还告诉了我如何转换为 gif...）太兴奋了 :) 引用 Ivan Fioravanti ᯅ (@ivanfioravanti) 👀 DeepSeek R1（右）碾压了 o1-pro（左） 👀 提示词：“wri...
- [Introduction to Operator & Agents](https://www.youtube.com/watch?v=CSE77wAdDLg): 太平洋时间上午 10 点开始。加入 Sam Altman、Yash Kumar、Casey Chu 和 Reiichiro Nakano，听他们介绍并演示 Operator。
- [Please add DeepSeek R1 model](https://forum.cursor.com/t/please-add-deepseek-r1-model/42868): 显然比 Sonnet 更好且便宜得多？拭目以待……
- [deepseek-ai (DeepSeek)](https://huggingface.co/deepseek-ai): 未找到描述
- [Terminal Chat](https://learn.microsoft.com/en-us/windows/terminal/terminal-chat): 了解如何在 Windows Terminal Canary 中设置和使用 Terminal Chat。
- [How to use structured outputs with Azure OpenAI Service - Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?utm_source=chatgpt.com))): 了解如何通过结构化输出改进模型响应
- [Wrapper for structured outputs with non required fields](https://community.openai.com/t/wrapper-for-structured-outputs-with-non-required-fields/913246?utm_source=chatgpt.com))): 来自文档：https://platform.openai.com/docs/guides/structured-outputs/supported-schemas 虽然所有字段都必须是必填的 [...]，但可以通过使用联合类型（union）来模拟可选参数...
- [Trae - Ship Faster with Trae](https://www.trae.ai/): 未找到描述
- [The Great Queers of History](https://rictornorton.co.uk/greatgay/greatgay.htm): 未找到描述
- [Meta genai org in panic mode | Artificial Intelligence - Blind](https://www.teamblind.com/post/Meta-genai-org-in-panic-mode-KccnF41n): 始于 DeepSeek V3，它使得 Llama 4 在基准测试中已经落后。更糟糕的是这家“训练预算仅为 550 万美元的无名中国公司”。工程师们正...
- [NEW Deepseek-R1 Update is INSANE! (FREE!) 🤯](https://youtu.be/SX8TpqfTTCg?si=dK5feMqVaxKuotgE): 🚀 立即获取免费 SEO 策略会议 + 折扣：https://go.juliangoldie.com/strategy-session - 另外长期合同可额外获得 25% 的免费额度！🤯 想要更多...
- [Paper page - UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://huggingface.co/papers/2501.12326): 未找到描述
- [GitHub - bytedance/UI-TARS](https://github.com/bytedance/UI-TARS): 通过在 GitHub 上创建账号，为 bytedance/UI-TARS 的开发做出贡献。
- [Fireworks - Fastest Inference for Generative AI](https://fireworks.ai/models): 以极快的速度使用最先进的开源 LLM 和图像模型，或者使用 Fireworks AI 免费微调并部署你自己的模型！

### **Codeium (Windsurf) ▷ #**[**content**](https://discord.com/channels/1027685395649015980/1092566563862884412/1332122340938481755) (1 条消息):

> `Web Search 功能，演示视频发布`

- **更好的网页浏览体验**：团队表示希望大家喜欢使用新推出的 Web Search 功能在互联网上 **(wind)surfin'** 🏄。
  
  - *这里有一个展示其功能的酷炫演示视频*，可以在[这里](https://x.com/windsurf_ai/status/1882561985621221451)找到。
- **演示视频的行动号召**：请求社区支持，敦促成员们在演示视频帖子下点赞互动。
  
  - 这种协作被视为推广 **新 Search 功能**、鼓励更多互动的关键。

 

**提到的链接**：[来自 Windsurf (@windsurf_ai) 的推文](https://x.com/windsurf_ai/status/1882561985621221451)：正在网上冲浪！🏄

 

---

### **Codeium (Windsurf) ▷ #**[**discussion**](https://discord.com/channels/1027685395649015980/1027697446446432336/1331722078885380106) (49 条消息🔥):

> `Codeium 扩展功能、Devin 的能力、Codeium 的 Web Search、JetBrains 中的 Supercomplete、Windsurf 更新与问题`

- **对 Codeium 扩展更新的担忧**：用户质疑 **Windsurf** 的发布是否意味着 Codeium 扩展将被忽视，并指出某些插件自 9 月以来就没更新过。
  
  - 一份回复保证没有放弃扩展的计划，因为许多企业客户依赖它们，尽管大多数创新都集中在 Windsurf 上。
- **Devin 的自主性受到质疑**：围绕 **Devin** 声称自己是完全自主 AI 工具的说法展开了讨论，一些人对其长期可行性和 Human-in-the-loop (HITL) 的必要性表示怀疑。
  
  - 有人担心，随着其实际能力在炒作面前受到质疑，最终可能会演变成类似“狼来了”的局面。
- **对 Web Search 功能的请求**：一位用户询问了 **Codeium 扩展** 获得类似 Windsurf 的 Web Search 功能的时间表，强调了对功能对等的需求。
  
  - 社区对现有功能在 IDE 集成（特别是 JetBrains 和其他环境）中的需求表达了挫败感。
- **关于 Supercomplete 功能的咨询**：在注意到 IDE 中的支持减弱后，成员们对 Codeium 扩展中 **Supercomplete** 功能的回归感到好奇。
  
  - 这促进了关于功能请求和社区参与的讨论，以重新激发对高级功能的兴趣。
- **权限和访问错误**：一些用户报告在尝试访问服务时遇到 **[permission_denied]** 错误，表明用户账户可能存在问题。
  
  - 这指向了对访问管理以及各自团队施加的可能限制的更广泛担忧。

**提到的链接**：

- [Cascade Memories](https://docs.codeium.com/windsurf/memories)：未找到描述
- [关于使用 Devin 一个月的思考 – Answer.AI](https://www.answer.ai/posts/2025-01-08-devin.html)：我们在给 Devin 分配了 20 多个任务后对它的印象。
- [Chrome 教程 | Windsurf 编辑器和 Codeium 扩展](https://codeium.com/chrome_tutorial)：Codeium 是开发者喜爱且企业信赖的 AI 代码辅助平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。
- [方案设置](https://codeium.com/plan)：未来的编辑器，就在今天。Windsurf 编辑器是首个由 AI Agent 驱动的 IDE，能让开发者保持专注。现已支持 Mac、Windows 和 Linux。
- [JetBrains 的 Supercomplete | 功能请求 | Codeium](https://codeium.canny.io/feature-requests/p/supercomplete-for-jetbrains)：我认为 JetBrains 在“连续动作建议”领域最欠缺。Supercomplete 将会是该领域首创的功能。
- [方案与定价更新](https://codeium.com/blog/pricing-windsurf)：我们对 Cascade 定价模型的一些调整。

---

### **Codeium (Windsurf) ▷ #**[**windsurf**](https://discord.com/channels/1027685395649015980/1306163501286293515/1331719212930170971) (493 条消息🔥🔥🔥):

> `Windsurf 额度问题，Windsurf 登录错误，移动应用开发工具与配置，AI 模型对比，Windsurf 用户体验`

- **Windsurf 额度消耗问题**：用户报告在使用 Windsurf 时 Flow 额度消耗显著，有人表示在几小时内修复错误就消耗了超过 **10%** 的月度额度。
  - 许多用户表示沮丧，因为 AI 的修复往往会导致新的错误，从而引发重复修复。
- **Windsurf 登录与权限问题**：几位用户遇到了 **ConnectError**，提示“用户被团队禁用 (user is disabled by team)”，这造成了困惑，且不清楚问题的根源。
  - 讨论中提出了如何解决这些问题，但未提供明确的解决方案。
- **使用 Windsurf 进行移动应用开发**：用户讨论了预览在 Windsurf 中编写的移动应用的方法，重点介绍了利用 Android Studio 等工具进行构建和测试应用的开发流程。
  - 参与者分享了环境搭建以及在开发过程中使用 Git 进行版本控制的技巧。
- **用户对 AI 模型性能的反馈**：一些用户对比了 DeepSeek R1 和 Sonnet 3.5，指出在处理某些编码任务时 R1 的表现优于 Sonnet。
  - 尽管有所增强，但用户对各种模型的可靠性仍持怀疑态度，呼吁 AI 提供更一致的结果。
- **Windsurf 与隐私担忧**：围绕 Trae 等新工具的隐私政策展开了讨论，用户对数据收集的侵入性表示保留意见。
  - 许多用户由于感知到数据隐私相关的风险，对使用此类工具持谨慎态度。

**提到的链接**：

- [未找到标题](https://frame0.app/)：未找到描述
- [首页 | Sweetpad](https://sweetpad.hyzyla.dev/)：描述将放入 <head /> 中的 meta 标签
- [Corey Quinn (@quinnypig.com)](https://bsky.app/profile/quinnypig.com/post/3lgglbn46w22u)：这是一项 AI 最佳实践 / 恐怖的排错技巧，即指示 LLM 像 Elmer Fudd 一样说话。当它停止这样做时，就说明它不再关注你设定的规则了...
- [持久、智能的项目记忆](https://forum.cursor.com/t/persistent-intelligent-project-memory/39109)：.cursorrules 只是权宜之计。我们真正需要的是 Cursor 能够真正记住与用户的交互以及项目的需求，并随着用户与 Cursor 的交互自动更新这些记忆...
- [来自 Riley Brown (@rileybrown_ai) 的推文](https://x.com/rileybrown_ai/status/1882281345935978649)：天哪，我正在 @cursor_ai 中使用 R1 模型，它真的在做出更改之前向我展示了思考过程……作为一个正在学习编程原理的人，我非常喜欢这一点。
- [页面未找到 | Windsurf 编辑器和 Codeium 扩展](https://codeium.com/c)：Codeium 是开发者喜爱、企业信赖的 AI 代码助手平台。也是首个 Agentic IDE —— Windsurf 的开发者。
- [Warp：智能终端](https://warp.dev)：Warp 是内置了 AI 和开发团队知识的智能终端。现已支持 MacOS 和 Linux。
- [联系方式 | Windsurf 编辑器和 Codeium 扩展](https://codeium.com/contact/enterprise)：联系 Codeium 团队以获取支持并了解更多关于我们企业级服务的信息。
- [Codeium 反馈](https://codeium.canny.io/)：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。
- [合并来自 ichoosetoaccept/add-link 的拉取请求 #89 · ichoosetoaccept/awesome-windsurf@8966b22](https://github.com/ichoosetoaccept/awesome-windsurf/actions/runs/12915269018/job/36016737674)：一系列用于 Windsurf 代码编辑器的优秀资源集合 - 合并来自 ichoosetoaccept/add-link 的拉取请求 #89 · ichoosetoaccept/awesome-windsurf@8966b22
- [\- YouTube](https://www.youtube.com/watch?v=hqJDKTqCESE)：未找到描述
- [Windsurf fork 了 VS Code 以与 Cursor 竞争。谈论 AI + 编码的未来](https://youtu.be/ptekg6GNzIQ?si=uKdjIGKEAiYZ8v-Y)：Wes 和 Scott 与来自 Windsurf 的 Kevin Hou 和 Varun Mohan 讨论了 AI 在编码领域不断演变的格局以及软件开发的未来。👉 加入 ...
- [隐私政策 | Windsurf 编辑器和 Codeium 扩展](https://codeium.com/privacy-policy)：Codeium 是开发者喜爱、企业信赖的 AI 代码助手平台。也是首个 Agentic IDE —— Windsurf 的开发者。

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1331715853372362848) (347 条消息🔥🔥):

> `DeepSeek R1 and Qwen, Multi-GPU Support in Unsloth, Fine-tuning for Non-English Languages, Tokenization Challenges in Biology, Evaluation and Training Strategies`

- **DeepSeek R1 在配合 Qwen 时保持了强劲性能**：用户注意到蒸馏模型 DeepSeek R1 非常有效，有人称其在大多数用例中“近乎完美”。
  
  - 然而，建议在对模型进行 Fine-tuning 之前，先针对特定应用进行测试和评估，以保持其性能。
- **Unsloth 预计很快将支持 Multi-GPU**：目前 Unsloth 不支持 Multi-GPU 操作，但预计即将发布的更新将解决这一功能。
  
  - Pro 用户将在功能推出时获得 Multi-GPU 支持，从而实现更快的训练时间和更高的性能。
- **针对非英语语言 Fine-tuning DeepSeek Qwen**：为了有效地为蒸馏后的 DeepSeek Qwen 添加非英语支持，用户应先在特定语言上训练 Qwen，然后再使用额外的 traces 进行 Fine-tuning。
  
  - 这种方法可以防止蒸馏模型原始能力的灾难性遗忘（catastrophic forgetting），同时实现扩展的语言支持。
- **生物学 Tokenization 过程中的挑战**：关于在基因组学中使用二进制 Tokenization 的讨论突显了在 Embedding 和序列识别方面潜在的低效。
  
  - 建议对单、双和三元 Tokenization 进行进一步评估，以明确生物学应用中的性能差异。
- **在 Fine-tuning 前评估模型**：强调在进行 Fine-tuning 之前，先对 DeepSeek R1 等模型进行初步评估，以避免性能下降。
  
  - 通过较小的迭代来测试和理解模型能力，可以为未来的训练策略提供宝贵的见解。

**提到的链接**：

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)：虽然最近的语言模型能够处理长上下文输入，但关于它们利用长上下文的效果知之甚少。我们分析了语言模型在两个任务上的表现...
- [Finetune Phi-4 with Unsloth](https://unsloth.ai/blog/phi4)：使用 Unsloth 对 Microsoft 的新模型 Phi-4 进行 Fine-tune！我们还发现并修复了模型中的 4 个 bug。
- [Qwen 2.5 Coder - a unsloth Collection](https://huggingface.co/collections/unsloth/qwen-25-coder-6732bc833ed65dd1964994d4)：未找到描述
- [Tweet from Keller Jordan (@kellerjordan0)](https://x.com/kellerjordan0/status/1881959719012847703)：社区对拥有更大的 NanoGPT 类别进行 speedrun 感兴趣。这里有一个打破记录的开端：新的 NanoGPT-medium speedrun 记录：在 29.3 秒内使用 8xH100 达到 2.92 FineWeb val loss...
- [Unsloth - Dynamic 4-bit Quantization](https://unsloth.ai/blog/dynamic-4bit)：Unsloth 的 Dynamic 4-bit Quants 有选择地避免对某些参数进行量化。这在保持与 BnB 4bit 相似的 VRAM 使用量的同时，大幅提高了准确性。
- [togethercomputer/evo-1-131k-base · Hugging Face](https://huggingface.co/togethercomputer/evo-1-131k-base)：未找到描述
- [LongSafari/open-genome · Datasets at Hugging Face](https://huggingface.co/datasets/LongSafari/open-genome)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1i867k8/first_5090_llm_results_compared_to_4090_and_6000/)：未找到描述
- [GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)：未找到描述
- [GitHub - nf-core/deepmodeloptim: Stochastic Testing and Input Manipulation for Unbiased Learning Systems](https://github.com/nf-core/deepmodeloptim)：无偏学习系统的随机测试和输入操作 - nf-core/deepmodeloptim

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1331863882762944512) (24 条消息🔥):

> `DeepSeek V3 Hardware Requirements, Dual RTX 3090s Setup, Training Reasoning Models, Dolphin-R1 Dataset, Unsloth Integration with TRL`

- **DeepSeek V3 硬件需求明确**：分享了一篇详细介绍离线运行 **DeepSeek V3** 规格的博客文章，并指出该设置**不需要 GPU**。
  
  - 该模型继 R1 之后，是目前最强大的开源 AI 模型，特别适合使用 [Unsloth](https://github.com/unslothai/unsloth) 进行微调。
- **配置双 RTX 3090s**：讨论了适合运行双 **RTX 3090s** 的机箱，并对开放式配置和设置提出了建议。
  
  - 一位用户计划使用外部 Docker 运行第二块 3090，并提到使用**温控帐篷**进行高效散热。
- **训练推理模型的仓库**：一位用户询问了用于训练推理模型的仓库，得到的建议是 **TRL** 包含了 R1 中使用的方法。
  
  - 引用了 [DeepSeekMath 论文](https://huggingface.co/papers/2402.03300) 以了解训练策略的概览。
- **Dolphin-R1 数据集发布公告**：一位用户提到需要赞助来创建 **Dolphin-R1 数据集**，该数据集产生了 6000 美元的 API 费用，并提供详细的蒸馏过程。
  
  - 随后，更新显示他们已获得赞助，并将以 **Apache-2.0 许可证**在 Hugging Face 上发布该数据集。
- **搭建强大的 GPU 配置**：一位用户描述了他们在 **Corsair 7000D** 机箱中的双 **4090** 配置，强调了 AIO 水冷的使用和主板通道拆分（bifurcated）配置。
  
  - 他们将其与 **Ryzen 7950X** 搭配，并使用 **DDR5-6400** RAM 等高端规格以获得最佳性能。

**提到的链接**：

- [来自 Eric Hartford (@cognitivecompai) 的推文](https://x.com/cognitivecompai/status/1882132168153178606)：创建 Dolphin-R1 数据集花费了 6000 美元的 API 费用。我遵循 Deepseek-R1 的蒸馏方案，但使用了 Dolphin 种子数据。（60 万条推理数据，20 万条对话数据，总计 80 万条）我想以 Apache 2.0 授权它...
- [运行 Deepseek-R1 / R1 Zero](https://unsloth.ai/blog/deepseek-r1)：DeepSeek 最新的 R-1 模型是目前最强大的开源推理模型，其性能与 OpenAI 的 o1 模型相当。了解如何运行和微调该模型。
- [GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)：未找到描述
- [来自 Eric Hartford (@cognitivecompai) 的推文](https://x.com/cognitivecompai/status/1882140705159799169)：我找到了赞助商！感谢 @driaforall！数据将在几天内以 Apache-2.0 许可证发布到 @huggingface。

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1331715730055495742) (79 条消息🔥🔥):

> `DeepSeek Distilled Models, Unsloth Notebooks, 模型微调问题, VRAM 消耗, 数据集管理`

- **关于 DeepSeek 蒸馏模型模板的查询**：一位成员分享了在使用新的 DeepSeek 蒸馏模型模板时的困惑，并询问输出是否需要按换行符格式化。
  
  - 讨论中提到了关于 System Prompts 缺乏清晰度的问题，以及如何正确利用模型进行高效微调。
- **Unsloth Notebooks 的挑战**：几位成员讨论了 Unsloth Notebooks 目前存在的问题，并对小团队的局限性表示担忧。
  
  - 他们提供了 Notebooks 的链接，并讨论了 Phi-4 和 Llama 3 等各种模型选项，强调了遇到的差异和错误。
- **模型微调问题**：一位用户报告在运行 RAG 模型时出现乱码输出，引发了关于微调时使用相同 Chat Template 重要性的讨论。
  
  - 强调了微调前后对结果进行持续评估的重要性，并分享了错误管理的建议。
- **理解 VRAM 消耗**：会议澄清了训练期间的 VRAM 消耗取决于模型大小和 Batch Size，而数据集大小则影响训练时长。
  
  - 成员们讨论了这些因素对不同模型的影响及其在各种环境下的性能表现。
- **通用聊天数据集查询**：一位成员询问了有效的通用聊天数据集，以减轻模型训练时的灾难性遗忘（Catastrophic Forgetting）。
  
  - 他们的重点是利用能够增强模型学习一致性且不会被新信息淹没的数据集。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing): 无描述
- [Finetune Phi-4 with Unsloth](https://unsloth.ai/blog/phi4): 使用 Unsloth 微调 Microsoft 的新模型 Phi-4！我们还发现并修复了模型中的 4 个 Bug。
- [Errors | Unsloth Documentation](https://docs.unsloth.ai/basics/errors#evaluation-loop-also-oom-or-crashing): 要修复设置中的任何错误，请参阅下文：
- [Errors | Unsloth Documentation](https://docs.unsloth.ai/basics/errors): 要修复设置中的任何错误，请参阅下文：
- [Unsloth Requirements | Unsloth Documentation](https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements): 这里是 Unsloth 的要求，包括系统和 GPU VRAM 要求。
- [Unsloth Notebooks | Unsloth Documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks): 以下是我们所有 Notebooks 的列表：
- [GitHub - ggerganov/llama.cpp: LLM inference in C/C++](https://github.com/ggerganov/llama.cpp): C/C++ 环境下的 LLM 推理。通过在 GitHub 上创建账户为 ggerganov/llama.cpp 的开发做出贡献。
- [unsloth/unsloth/kernels/fast_lora.py at d802bbf4e298cb0da1e976ab9670fbc1cbe3514c · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/d802bbf4e298cb0da1e976ab9670fbc1cbe3514c/unsloth/kernels/fast_lora.py#L201)): 使用 Unsloth 将 Llama 3.3, Mistral, Phi-4, Qwen 2.5 和 Gemma LLM 的微调速度提高 2-5 倍，并减少 70% 的显存占用。

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1332012381336764509) (11 条消息🔥):

> `Finetuning Striped Hyena Model, Unsloth GPU Support, Genomic Data Pretraining`

- **博士后寻求微调 Striped Hyena**：一位 bioML 领域的博士后表达了微调 **evo** 模型的需求。该模型是一个在原核生物序列上训练的 **Striped Hyena**，旨在将其适配到真核生物，并引用了一篇[研究论文](https://www.science.org/doi/10.1126/science.ado9336)。
  
  - 他们分享了该模型的 [GitHub 仓库](https://github.com/togethercomputer/stripedhyena/tree/main)，寻求关于如何适配 Unsloth 来完成此任务的指导。
- **关于 GPU 资源的讨论**：一位成员指出 **Unsloth** 并不考虑正在使用的是哪款 NVIDIA GPU，因为**多 GPU 支持尚未全局可用**。
  
  - 他们强调，如果训练时只能访问一个 GPU，某些功能将会受到限制。
- **基因组数据需要不同的方法**：有讨论指出，**基因组数据**和 **RNA/DNA 序列**需要不同的预训练方法，才能有效地识别标记（markers）和向量（vectors）。
  
  - 这表明，如果没有在目标数据上进行适当的重新训练，仅靠适配现有模型可能并不完全足够。
- **关于消息相关性的对话**：一位成员指出，最初关于微调的帖子不适合发布在 research 频道，并建议将对话移至更合适的频道。
  
  - 最初的发布者承认了错误，并幽默地提到研究论文确实应该属于 research 频道。

 

**提到的链接**：[GitHub - togethercomputer/stripedhyena: Repository for StripedHyena, a state-of-the-art beyond Transformer architecture](https://github.com/togethercomputer/stripedhyena/tree/main)：StripedHyena 仓库，一种超越 Transformer 架构的最先进架构 - togethercomputer/stripedhyena

 

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1331719616522621008) (173 条消息🔥🔥):

> `DeepSeek 模型, LM Studio 错误排查, 量化对模型性能的影响, LM Studio 中的本地网络访问, Gemini 2.0 性能`

- **排查 DeepSeek 模型加载错误**：用户报告了在加载 DeepSeek R1 等模型时出现“unknown pre-tokenizer type”等错误，引发了关于故障排除步骤的讨论，包括检查 runtimes 和手动更新 LM Studio。
  
  - 几位用户分享了解决方案，强调如果问题仍然存在，需要调整设置并重新下载模型。
- **Qwen 模型的性能与量化**：用户讨论了 Qwen 模型在不同量化水平下的性能，*havenwood* 建议使用 Q5_K_M 以平衡大小和准确性，而其他用户则询问了高量化与低量化设置的影响。
  
  - 对话强调了用户对不同参数规模的体验，表达了对更高参数以获得更好模型性能的倾向。
- **LM Studio 中的本地网络访问**：讨论强调了 LM Studio 设置中关于本地网络访问术语需要更清晰，建议使用能明确表示是在所有 IP 上提供服务还是仅在 localhost 上提供服务的选项标签。
  
  - 用户对现有设置及其如何影响网络中各种设备对模型的访问感到困惑。
- **探讨适用于 LM Studio 的最佳视觉模型**：用户询问了适用于 LM Studio 的最佳视觉模型，*havenwood* 提到了 Llama 3.2 和新兴的 UI-TARS 模型是值得探索的选择。
  
  - 讨论揭示了在特定硬件上使用 MLX 和 GGUF 格式的挑战，突出了兼容性问题。
- **Gemini 2.0 性能见解**：用户称赞了新的 Google Gemini 2.0 Flash 模型在处理法律文件方面的表现，认为其庞大的上下文长度和对细节的关注是对先前模型的重大改进。
  
  - 用户将其与 o1 mini 等旧模型进行了比较，Gemini 因其持续的输出和精炼的知识被定位为有力的继任者。

**提到的链接**：

- [来自 @levelsio (@levelsio) 的推文](https://x.com/levelsio/status/1882028288702656673?s=46)：好吧，再次选择了 LM Studio
- [leafspark/Llama-3.2-11B-Vision-Instruct-GGUF · Llama-3.2-11B-Vision-Instruct 的 Modelfile](https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF/discussions/2#677efb4c846852dc90)：未找到描述
- [leafspark/Llama-3.2-11B-Vision-Instruct-GGUF · Llama-3.2-11B-Vision-Instruct 的 Modelfile](https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF/discussions/2#677efb4c846852dc90e75cd0)：未找到描述
- [LM Studio - 发现、下载并运行本地 LLM](https://lmstudio.ai/)：在你的电脑上本地运行 Llama, Mistral, Phi-3。
- [GitHub - bytedance/UI-TARS](https://github.com/bytedance/UI-TARS)：通过在 GitHub 上创建账号来为 bytedance/UI-TARS 的开发做出贡献。

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1331779226961051740) (143 条消息🔥🔥):

> `NVIDIA RTX 5090 性能、Llama 模型基准测试、GPU 使用的 AVX2 要求、AI 推理 API 补贴、Procyon AI 文本生成基准测试`

- **NVIDIA RTX 5090 显示出适度的性能提升**：讨论透露，**RTX 5090** 比 **RTX 4090** 快约 **30%**，但这与增加的 **1.7 倍带宽** 并不直接对应，原因可能是受限于较小模型。
  
  - 带宽与性能的对比表明，小模型可能无法充分利用改进后的显存总线。
- **Llama 模型性能的挑战**：参与者指出，在使用高带宽硬件时，对超过 **8B** 参数的 **LLM** 进行基准测试往往显示出有限的性能提升。
  
  - 目前，**Llama3** 模型缺乏能够有效代表不同 GPU VRAM 下真实世界性能的全面基准测试。
- **旧服务器的 AVX2 限制**：一位用户分享了在没有 **AVX2** 的服务器上使用 **GTX 1080 TI** 运行模型的挑战，引发了对兼容性的担忧。
  
  - 尽管拥有强大的 CPU 资源，但缺乏 AVX2 可能会限制该配置下可实现的性能提升。
- **AI 使用场景的 API 补贴**：围绕 **AI 模型** 的讨论指出，对消费者而言，最可行的选择可能取决于公司通过 API 访问对模型进行补贴，例如 **OpenAI**。
  
  - 这反映了一种趋势，即企业承担托管推理 API 的成本，以促进 **LLM** 的更广泛使用。
- **Procyon AI 基准测试工具用于简化测试**：建议考虑使用 **Procyon AI Text Generation Benchmark** 来评估 **LLM 性能**，从而简化并提供跨模型的一致性测试。
  
  - 该基准测试工具旨在平衡量化和模型要求等因素，而这些因素往往会使传统的 LLM 性能评估变得复杂。

**提到的链接**：

- [NVIDIA GeForce RTX 5090 Review: Pushing Boundaries with AI Acceleration](https://www.storagereview.com/review/nvidia-geforce-rtx-5090-review-pushing-boundaries-with-ai-acceleration)：NVIDIA GeForce RTX 5090 评测：通过 AI 加速突破界限。2025 年 1 月 30 日发布，售价 1,999 美元。5090 是否会重新定义高性能游戏和 AI 工作负载？
- [NVIDIA RTX Blackwell GPU with 96GB GDDR7 memory and 512-bit bus spotted - VideoCardz.com](https://videocardz.com/newz/nvidia-rtx-blackwell-gpu-with-96gb-gddr7-memory-and-512-bit-bus-spotted)：发现配备 96GB GDDR7 显存和 512-bit 总线的 NVIDIA RTX Blackwell GPU。据报道，NVIDIA 正在准备一款配备 96GB 显存的工作站旗舰产品，该显卡据称使用 3GB 模块。根据 ComputerBase 的报告，NVIDIA 即将推出的桌面显卡预计将...
- [Procyon AI Text Generation](https://benchmarks.ul.com/procyon/ai-text-generation-benchmark)：测试 AI LLM 性能可能非常复杂且耗时，完整的 AI 模型需要大量的存储空间和带宽来下载。
- [NVIDIA RTX Blackwell GPU with 96GB GDDR7 memory and 512-bit bus spotted - VideoCardz.com](https://videocardz.com/newz/nvidia-rtx-blackwell-gpu-with-96gb-gddr7-me)：NVIDIA 正在准备一款配备 96GB 显存的工作站旗舰产品。该显卡据称使用 3GB 模块。根据 ComputerBase 的报告，NVIDIA 即将推出的桌面显卡预计将...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/19cdd9z/dont_take_apple_mlx_too_seriously_its_not_going/)：未找到描述
- [NVIDIA GeForce RTX 5090 Founders Edition Review & Benchmarks: Gaming, Thermals, & Power](https://youtu.be/VWSlOC_jiLQ?si=GRbMJ1Z34IdZrFNO)：赞助商：亚马逊上的 Thermal Grizzly Aeronaut 以及 Hydronaut。NVIDIA GeForce RTX 5090 GPU 将于下周发布...
- [NVIDIA GeForce RTX 5090 Graphics Cards](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)：采用 NVIDIA Blackwell 架构。
- [Nvidia GeForce RTX 5090 Review, 1440p & 4K Gaming Benchmarks](https://youtu.be/eA5lFiP3mrs?si=YdcBf8Z5YT2rKXTa)：在此查看 Asus X870 系列：https://www.asus.com/microsite/motherboard/amd-am5-ryzen-9000-x3d-x870e-x870-b850-b840/ 在 Patreon 上支持我们：https://www...

---

### **Perplexity AI ▷ #**[**announcements**](https://discord.com/channels/1047197230748151888/1047204950763122820/1332055497561608325) (1 messages):

> `Perplexity Assistant Launch, Assistant Features, Integration with Other Apps`

- **Perplexity Assistant 面向 Android 用户发布**：**Perplexity Assistant** 现已在 Android 上线，标志着从回答引擎向集成 Assistant 的转变，能够跨应用执行任务。
  
  - 用户可以通过[此链接](https://pplx.ai/android)访问并探索其协助处理日常任务的能力。
- **Assistant 的独特能力**：**Assistant** 可以浏览网页以设置提醒，并在不同操作之间保持 Context，从而实现无缝的多应用功能。
  
  - 无论是预订餐厅还是提醒活动，Assistant 旨在轻松管理各种任务。
- **Assistant 的多模态（Multimodal）交互**：用户可以与 **Assistant** 进行多模态交互，例如开启摄像头询问现实世界中的物体。
  
  - 该功能扩展了用户除了文本命令之外与 Assistant 互动的方式。
- **对用户反馈的期待**：团队对用户将如何利用 **Assistant** 处理日常活动表示热切期待。
  
  - 他们期待看到用户发挥创意，使用 Assistant 的各项功能。

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1331722196892258365) (250 条消息🔥🔥):

> `Perplexity Assistant 问题、模型选择挑战、Sonar 模型变更、AI 输出质量对比、Perplexity 新功能`

- **Perplexity Assistant 的困扰**：用户报告了使用 Perplexity Assistant 时的困难，特别是在更改语音设置和语音激活功能未按预期工作方面。
  
  - 几位用户对需要手动启动助手而不是让其响应语音命令表示沮丧。
- **模型选择的困惑**：关于从 API 中移除 Opus 模型的担忧被提出，用户讨论了这对他们的工作流和使用偏好的影响。
  
  - 一些用户通过修改请求参数成功绕过了限制，引发了关于模型访问技术细节的讨论。
- **Sonar 模型裁减**：Sonar 模型阵容的过渡引起了关注，特别是移除 Sonar Huge 以支持 Sonar Large，并暗示未来更新将包含 Sonar Pro。
  
  - 用户推测这些变化是否与即将推出的模型发布或可能提高性能的调整有关。
- **AI 输出质量对比**：不同模型（包括 Claude 2 和 Claude 3.5）之间的对比凸显了褒贬不一的体验，特别是在输出质量和拒绝率方面。
  
  - Nemoia 对从更可靠模型的过渡表示遗憾，认为响应质量和针对性的转变是一个重大问题。
- **Perplexity 新功能**：关于 Perplexity Assistant 新功能的讨论包括其分析屏幕和执行某些任务的能力，但也面临易用性挑战。
  
  - 几位用户分享了助手的技巧和功能，包括潜在的自定义及其在日常任务中的集成。

**提到的链接**：

- [Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1882466239123255686)：介绍 Perplexity Assistant。Assistant 利用推理、搜索和应用来帮助处理从简单问题到多应用操作的日常任务。你可以预订晚餐、找回遗忘的歌曲、计算...
- [Sam Altman (@sama) 的推文](https://x.com/sama/status/1882478782059327666)：大新闻：ChatGPT 免费版将获得 o3-mini！（Plus 版将获得大量的 o3-mini 使用额度）
- [Revolut 为英国客户推出最高 5% AER（浮动）的储蓄利率](https://ffnews.com/newsarticle/fintech/revolut-launches-its-highest-savings-rates-for-uk-customers-of-up-to-5-aer-variable/)：Revolut 为其英国即时访问储蓄账户大幅提高了利率，提供高达 5% AER 的利率。
- [Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/AravSrinivas/status/1882467172498436291)：我们很高兴向所有 Android 用户推出 Perplexity Assistant。这标志着 Perplexity 从回答引擎向原生集成助手的转型，它可以调用其他应用并...
- [ChatGPT vs. Nelima：哪种任务调度更好？](https://youtu.be/GytSx78md0s?si=pEadW3sj-EPqYV3X)：🚀 ChatGPT vs Nelima：终极任务调度对决！🚀 在这段视频中，我们将 ChatGPT 全新的任务调度功能与 Nelima 强大的...
- [Operator 与 Agents 介绍](https://www.copilotforyoutube.com/search/introduction-to-operator-and-agents-1SrOLOfnSBEp9I5yo3enIr)：太平洋时间上午 10 点开始。加入 Sam Altman、Yash Kumar、Casey Chu 和 Reiichiro Nakano，他们将介绍并演示 Operator。

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1331816510846533694) (11 条消息🔥):

> `PyCTC Decode, Mistral Plans IPO, CIA Chatbot, Stargate Initiative, DeepSeek R1`

- **探索项目的 PyCTC Decode**：一位成员分享了他们对项目中使用 *PyCTC Decode* 的兴趣，并强调了其潜在的应用价值。
  
  - 更多详情请参考此 [PyCTC Decode 链接](https://www.perplexity.ai/search/pyctc-decode-57MLlbc2QbmCmmV1jO6zdw)。
- **Mistral 的 IPO 计划揭晓**：目前的讨论包括 **Mistral** 的 **IPO** 计划，这引起了广泛关注。
  
  - 一段名为 'YouTube' 的视频讨论了这一进展，可以在[此处](https://www.youtube.com/embed/dGQOrroTmTY)观看。
- **DeepSeek R1 突破界限**：一位成员指出，在特定对比中，**DeepSeek R1** 模型可能优于 OpenAI 的产品。
  
  - 更多见解请见此[详细探索](https://www.perplexity.ai/page/deepseek-r1-may-beat-openai-s-Xl.Pc5FFSfS9NaIIuLNQaw)。
- **音乐流媒体项目讨论**：一位用户提到他们正在创建一个**音乐流媒体**平台，并寻求见解和帮助。
  
  - 有关此尝试的相关细节在[此链接](https://www.perplexity.ai/search/i-am-making-a-music-streaming-GRa5Oet5TTKyknYnH2n4HQ)中讨论。
- **关于 AI Prompt 的咨询**：一位成员询问：*生成有效回复的最佳 AI Prompt 是什么？*。
  
  - 这一询问引发了多次讨论，资源通过[此链接](https://www.perplexity.ai/search/was-ist-der-beste-ki-prompt-fu-9CPjn9hmQJSZBjDhd3XiiQ)分享。

 

**提到的链接**：[YouTube](https://www.youtube.com/embed/dGQOrroTmTY)：未找到描述

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1331794096184103003) (11 条消息🔥):

> `API Issues, API SOC 2 Compliance, Sonar vs Legacy Models, Retrieving Old Responses by ID, Sonar-Pro Multi-Step Goals`

- **API 遭遇 524 错误**：几位成员报告在尝试切换到 API 上的 **sonar/sonar-pro** 时遇到了 **524 错误**，表明可能存在连接问题。
  
  - 一位用户指出，部分请求最终成功通过，暗示 API 存在间歇性问题。
- **API SOC 2 合规性查询**：一位成员祝贺团队发布了 API，并询问该 API 是否符合 **SOC 2** 标准，强调了其对产品集成的关键性。
  
  - 讨论强调合规性是企业级解决方案采用的关键因素。
- **关于 Sonar 功能的疑问**：一位成员提出疑问：**Sonar** 相比 legacy models 是否有实质性变化，还是仅仅是 **rebranding**。
  
  - 另一位用户询问了 **sonar-pro** 的功能，特别是关于类似于 Perplexity Pro Search 中的 **multi-step goals**。
- **通过 ID 检索旧回复**：成员们讨论了通过 **ID** 检索旧回复的可能性，并建议将回复存储在自己的数据库中以便访问。
  
  - 该功能被认为对于需要在应用程序中验证 API calls 的用户至关重要。
- **对 API 更新的积极展望**：一位用户对 API 及其更新所获得的关注表示满意，消除了早些时候对被忽视的担忧。
  
  - 这反映了社区对持续改进和支持的日益赞赏。

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1331755607451242627) (6 条消息):

> `Web Search API 发布，引入 Reasoning Tokens，Web Search 定价更新，模型标准化改进，提前发布的公告`

- **Web Search API 发布**：新的 **Web Search API** 允许用户在模型名称后附加 `:online` 以使用 Web 搜索功能，定价为 **$4/1k 结果**。
  
  - 默认情况下，每次请求使用 Exa.ai 获取最多 **5 个 Web 结果**，用户可以自定义结果数量和 Prompt。
- **引入 Reasoning Tokens**：通过在请求中包含 `include_reasoning: true`，引入 **Reasoning Tokens** 让用户可以直接在 Chatroom 中查看模型的推理过程。
  
  - 该功能也可通过 API 使用，增强了模型推理的透明度。
- **Web Search 定价更新**：分享了关于 **Web 搜索定价** 的说明，指出收费标准从 **$4/1k 结果**起，每次请求的成本低于 **$0.02**。
  
  - 该定价模型将于次日开始实施，同时 API 访问将进入 Soft Launch（软启动）阶段。
- **模型标准化改进**：最近的更新包括跨模型的 `finish_reason` **规范化**，使用 OpenAI 风格的解释，并返回原生原因以确保清晰。
  
  - Reasoning Tokens 在所有推理模型中实现了标准化，预计将提高一致性和易用性。
- **提前发布的公告**：承认了关于更新的公告发布过早；某些功能仍在部署中。
  
  - 这在聊天中引发了幽默的反应，确保了对正在进行的更改的清晰说明。

**提到的链接**：

- [来自 OpenRouter (@OpenRouterAI) 的推文](https://x.com/OpenRouterAI/status/1882491124402803075)：新的 LLM 标准正在兴起：Reasoning Tokens！🧠 - 你现在可以直接在 Chatroom 中查看模型的推理过程 - 跨多个思考模型（包括 Deep...）的标准化 API（包括 finish reasons）。
- [Cheers Cheerleader GIF - Cheers Cheer Cheerleader - Discover & Share GIFs](https://tenor.com/view/cheers-cheer-cheerleader-cheer-up-cheering-gif-18395332177585697711)：点击查看 GIF
- [来自 OpenRouter (@OpenRouterAI) 的推文](https://x.com/OpenRouterAI/status/1882498131381936257)：今天的另一个发布：Web Search API！只需附加 ":online" 即可为任何模型添加 Grounding 🌐

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1331727392619761664) (244 条消息🔥🔥):

> `DeepSeek R1, API 功能与问题, 联网搜索定价, 模型性能对比, 额度支付方式`

- **DeepSeek R1 响应迟缓**：用户报告称 DeepSeek 提供的 R1 出现挂起现象，且无法从 DeepSeek 和 DeepInfra 返回响应，表明可能存在服务问题。
  
  - 一位用户质疑无响应是否由模型固有的问题引起。
- **API Key 问题与扣费**：一位用户遇到了 Mistral 的 API Key 未被优先使用的问题，导致系统扣除了账户额度，而非使用集成设置。
  
  - 其他用户推测，与 OpenRouter 相关的额外费用可能使问题变得复杂。
- **联网搜索定价说明**：联网搜索查询的新定价显示，每个搜索结果收费 $0.02，这将增加总使用成本。
  
  - 用户对联网搜索功能对 API 使用和计费的影响表示困惑。
- **模型性能讨论**：参与者讨论了蒸馏模型（distilled models）的能力，指出虽然某些模型不会“思考”，但其表现仍优于 O1 Mini 和 Claude 等旧模型。
  
  - 人们对性能差异以及各种模型实现的有效性持一些怀疑态度。
- **额度充值的替代支付方式**：一位用户询问了购买额度的替代方法，特别是考虑到现有方法不适用于其所在国家。
  
  - 提到可以通过 OpenRouter 的界面使用加密货币购买额度，为部分用户提供了解决方案。

**提到的链接**：

- [kluster.ai - 大规模赋能 AI](https://platform.kluster.ai/)：以小规模成本实现大规模推理。彻底改变大规模推理的开发者平台。
- [Anthropic API 推出引用功能](https://www.anthropic.com/news/introducing-citations-api)：今天，我们推出了引用（Citations）功能，这是一项新的 API 功能，让 Claude 的回答可以基于源文档。
- [联网搜索 | OpenRouter](https://openrouter.ai/docs/web-search)：模型无关的 Grounding
- [Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1882480450649915772)：介绍引用功能。我们新的 API 功能让 Claude 的回答可以基于你提供的来源。Claude 随后可以引用支撑每个回答的具体句子和段落。
- [DeepSeek R1 Distill Llama 70B - API、提供商、统计数据](https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b)：DeepSeek R1 Distill Llama 70B 是一个基于 [Llama-3.3-70B-Instruct](https://openrouter) 的蒸馏大语言模型。通过 API 运行 DeepSeek R1 Distill Llama 70B。
- [未找到标题](https://ai.google.dev/gemini-api/docs/models/gemini-v2)：未找到描述
- [Reddit - 深入了解](https://www.reddit.com/r/LocalLLaMA/comments/1i7o9xo/deepseek_r1s_open_source_version_differs_from_the/)：未找到描述
- [未找到标题](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-2.0-flash-thinking-mode)：未找到描述
- [加密货币支付 API | OpenRouter](https://openrouter.ai/docs/crypto-api)：与无需 UI 购买额度相关的 API
- [DeepSeek R1 - API、提供商、统计数据](https://openrouter.ai/deepseek/deepseek-r1)：DeepSeek R1 已发布：性能与 [OpenAI o1](/openai/o1) 相当，但完全开源并提供完整的推理 Token📖 完全开源的模型和[技术报告](https://api-docs.deepseek...)
- [请求 | OpenRouter](https://openrouter.ai/docs/requests)：处理传入和传出请求

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1331719004816932966) (154 条消息🔥🔥):

> `Aider 配置, DeepSeek R1 性能, 使用多个 LLM, 聊天模式操作, 来自 Anthropic 的 Citations API`

- **多 LLM 的 Aider 配置**：用户讨论了如何通过 `aider.conf.yaml` 文件配置 Aider，以便在编码和架构任务中使用不同的 LLM，并为这两个角色设置特定的模型。
  
  - 会议澄清，将聊天模式切换到 `code` 会忽略配置中指定的编辑器模型，默认在两个角色中都使用架构师模型。
- **DeepSeek R1 性能问题**：多位用户对 DeepSeek R1 的语法处理和性能表示不满，指出了在编码时的上下文容量和准确性问题。
  
  - 一些用户发现，提供选择性的上下文而非整个文件有助于维持性能，而另一些用户则报告在不同任务中的结果褒贬不一。
- **在 Ollama 中使用多个 LLM**：有关于在使用 Ollama 时如何有效地将问题路由到特定 LLM 的咨询，一些用户成功启动了两个 LLM 来实现这一目的。
  
  - 讨论内容包括使用自定义提示词和 memory banks 来管理上下文的好处，从而提高编码任务的性能。
- **Aider 中的聊天模式操作**：强调了在 Aider 中设置正确聊天模式的重要性，通过 `/chat-mode code` 或 `/chat-mode architect` 等操作命令允许用户在模式之间切换。
  
  - 提到默认加载设置决定了初始模式，用户可以在工作流中根据需要切换模式。
- **来自 Anthropic 的 Citations API**：Anthropic 的一项新功能允许在回复中引用来源，增强了 AI 生成内容的可信度和可验证性。
  
  - 这解决了之前提示工程（prompt engineering）的复杂性，提供了一种与 Claude 回复集成的引用来源的直接方法。

**提到的链接**：

- [Introducing Citations on the Anthropic API](https://www.anthropic.com/news/introducing-citations-api)：今天，我们推出了 Citations，这是一项新的 API 功能，让 Claude 能够根据源文档提供回答。
- [Prompt Engineering Guide](https://www.promptingguide.ai/techniques/cot)：提示工程全面概述。
- [DeepSeek R1: API Provider Performance Benchmarking & Price Analysis | Artificial Analysis](https://artificialanalysis.ai/models/deepseek-r1/providers)：对 DeepSeek R1 的 API 供应商进行的性能指标分析，包括延迟（首个 token 时间）、输出速度（每秒输出 token 数）、价格等。
- [人間によるコーディング禁止の CLINE 縛りでゲームを作らせてみた感想](https://zenn.dev/mizchi/articles/game-with-cline#%E5%B7%A5%E5%A4%AB%3A-%E8%87%AA%E5%B7%B1%E3%83%89%E3%82%AD%E3%83%A5%E3%83%A1%E3%83%B3%E3%83%88%E5%8C%96)：（未找到描述）
- [Coding with AI episode 1: Claude Sonnet 3.5 vs DeepSeek v3 for coding analysis and modifications](https://www.youtube.com/watch?v=_pLlet9Jrzc&list=PLrEMgOSrS_3cU-ndLheq6TZiO3gWTAszA&index=3)：注意：请确保开启字幕（手动编辑）。我的英语口音很难懂，尤其是在疲劳时。我们将使用 kilo 编辑器作为测试案例...
- [DeepSeek R1 Fully Tested - Insane Performance](https://www.youtube.com/watch?v=bOsvI3HYHpI)：开源持续获胜！Vultr 正在通过提供最新的 NVIDIA GPU 来赋能下一代生成式 AI 初创公司！
- [Reasoning Model (deepseek-reasoner) | DeepSeek API Docs](https://api-docs.deepseek.com/guides/reasoning_model)：deepseek-reasoner 是由 DeepSeek 开发的推理模型。在交付最终答案之前，模型会先生成思维链（CoT）以提高响应的准确性。
- [FuseAI/FuseO1-DeepSeekR1-Qwen2.5-Coder-32B-Preview-GGUF · Hugging Face](https://huggingface.co/FuseAI/FuseO1-DeepSeekR1-Qwen2.5-Coder-32B-Preview-GGUF)：（未找到描述）
- [Feature: Add GitHub Copilot as model provider · Issue #2227 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2227)：问题：你好！请添加 GitHub Copilot 作为模型提供商。应该可以像这样实现：https://github.com/olimorris/codecompanion.nvim/blob/5c5a5c759b8c925e81f8584a0279eefc8a6c6643/lua/codecompani...
- [no title found](https://news.ycombinator.com/item?id=42589158)：（未找到描述）

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1331719213928153161) (79 条消息🔥🔥):

> `Aider 安装问题，在 Docker 中使用 Aider，Aider 日志实践，Aider 与大型代码库，Aider 使用模型推荐`

- **复杂的 Aider 安装导致挫败感**：成员们讨论了 Aider 的安装问题，特别是在使用 Conda 和 Brew 等多个包管理器时，导致了版本混淆。
  
  - 一位用户报告称，在成功重新安装 Aider 之前，必须删除大量的辅助文件。
- **Aider 的文件访问能力**：有关于 Aider 访问文件和目录能力的咨询，澄清了 Aider 可以看到 Git 仓库中的文件，但可能需要显式命令才能读取它们。
  
  - 有人指出 Aider 大量使用终端输出，如果管理不当，可能会使工作流变得复杂。
- **优化 Aider 的日志记录**：用户建议实施一种将输出写入文件的日志系统，以在使用 Aider 时最大限度地减少 Token 使用和上下文膨胀。
  
  - 这可以实现对项目日志的高效跟踪，而无需在终端之间进行过多的复制。
- **在大型代码库中处理 Aider 的输出**：讨论强调了在处理大型代码库时，管理 Aider 中终端命令输出的挑战。
  
  - 成员们讨论了更简洁的输出管理方法的潜力，以防止在冗长的命令序列期间出现上下文过载。
- **Aider 模型推荐**：用户寻求关于与 Aider 配合使用的有效模型的建议，这些模型应具有良好的性价比。
  
  - 大家普遍对优化模型选择以提高 Aider 的生产力感到好奇。

 

**提到的链接**：[Installation](https://aider.chat/docs/install.html)：如何安装并开始使用 Aider 进行结对编程。

 

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1332005106597756948) (10 条消息🔥):

> `JetBrains AI，Cursor 与 Windsurf 的竞争，VSCode vs JetBrains，Continue 功能使用，用户候补名单查询`

- **JetBrains 加入 AI 赛道？**：一位成员在加入 **JetBrains AI** 候补名单后表达了希望，质疑 JetBrains 是否最终能在 AI 领域竞争，考虑到过去的失败。
  
  - *由于早期的失望，他们失去了许多用户，这些用户转投了 Cursor 和 Windsurf*。
- **尽管 AI 方面表现不佳，JetBrains 仍受青睐**：尽管对其 AI 能力感到担忧，一位成员表示 **JetBrains** 仍然是他们最喜欢的 IDE 供应商，并将其与 **VSCode** 进行了负面对比。
  
  - 这种情绪加强了一些专业人士对 JetBrains 工具的强烈偏好，*无论其 AI 表现如何*。
- **JetBrains 用户仍然依赖 IDE**：另一位人士指出，专业人士继续使用 **JetBrains** 工具，肯定了它们即使没有 AI 增强也具有实用性。
  
  - 他们还提到 **Continue** 功能在 JetBrains 中可以运行，显示了用户的持续投入。
- **用户迁移趋势**：讨论集中在用户迁移上，表明大多数人已经离开 **JetBrains** 转向 **VSCode**，同时 Windsurf 也被注意到正在吸引用户。
  
  - 这反映了在 IDE 领域日益激烈的竞争中，JetBrains 面临的持续挑战。
- **关于候补名单的查询**：一位用户询问对话中引用的是哪个 **waitlist**，表明对该主题缺乏清晰度。
  
  - 这突显了关于 JetBrains 当前产品和开发的潜在沟通鸿沟。

 

---

### **OpenAI ▷ #**[**annnouncements**](https://discord.com/channels/974519864045756446/977259063052234752/1332048131688501442) (1 条消息):

> `Operator 介绍，OpenAI 演示`

- **OpenAI 团队揭晓 Operator**：加入 **Sam Altman**、**Yash Kumar**、**Casey Chu** 和 **Reiichiro Nakano**，他们在 [YouTube](https://www.youtube.com/watch?v=CSE77wAdDLg) 上介绍并演示 **Operator**。演示于 **太平洋时间上午 10 点** 开始。
  
  - *不要错过这次关于新功能和能力的精彩演示！*
- **为 Operator 演示做好准备**：在日历上标记由 OpenAI 高管领导的 **Operator** 工具揭晓仪式。
  
  - 此次介绍性会议承诺展示增强用户参与度的关键功能。

 

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1331726814472831088) (170 messages🔥🔥):

> `DeepSeek R1 Performance, Operator Features, Usage of Perplexity Assistant, OpenAI API Comparisons, Spiking Neural Networks Discussion`

- **DeepSeek R1 在编程任务中超越 O1**：用户报告称，与 **O1** 相比，**DeepSeek R1** 提供了更好的编程解决方案，甚至在侧向对比中更受青睐。
  
  - *一位用户幽默地指出*，R1 如何将一个关于“POTATO THUNDERSTORM!”的随机提示词转化为一段详尽的回复。
- **Operator 支持 Web 浏览器交互**：**Operator** 功能允许与 Web 浏览器进行交互，为用户提供了一项新能力，订阅费用为 **$200/月**。
  
  - 目前，它仅能通过 Web 浏览器使用，且用户无法选择用于操作的浏览器。
- **Perplexity Assistant 获得比 OpenAI 产品更高的评价**：一位用户提到，**Perplexity Assistant** 在移动端的表现优于 OpenAI 目前的移动解决方案。
  
  - 这引发了关于各种 AI 产品之间用户体验总体质量的讨论，凸显了对竞争对手产品的不满。
- **对 OpenAI 订阅费用的担忧**：参与者讨论了 **OpenAI 定价** 的公平性，特别是对于那些看似提供实质性收益的功能。
  
  - 一位用户建议，以目前的价格来看，公司名称“Open AI”可能不再合适。
- **对脉冲神经网络（Spiking Neural Networks）的不同看法**：*一位成员提出了*关于**脉冲神经网络**可行性的问题，权衡了它们的效率与潜在的延迟问题。
  
  - 这引发了一场对话，讨论它们在未来项目的某些方面可能代表死胡同，而在其他方面则是非常有用的工具。

**提到的链接**：

- [ChatGPT vs. Nelima: Which task scheduling is better?](https://youtu.be/GytSx78md0s?si=pEadW3sj-EPqYV3X)：🚀 ChatGPT vs Nelima：终极任务调度对决！🚀 在这段视频中，我们将 ChatGPT 全新的任务调度功能与 Nelima 强大的功能进行对比...
- [Trae - Ship Faster with Trae](https://www.trae.ai)：无描述
- [TikTok made an IDE and it's actually good? (free cursor killer??)](https://www.youtube.com/watch?v=hqJDKTqCESE)：Byte Dance（TikTok 的母公司）开发了一个代码编辑器，而且效果竟然很好？！RIP Cursor？VS Code 杀手？Jetbrains 克隆版？我不知道发生了什么...

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1331956284370456629) (9 messages🔥):

> `GPT Outage, Voice Feature Issues, Status Updates, Attributing Blame for Downtime`

- **GPT 面临停机问题**：成员们对 **GPT 宕机** 表示沮丧，提到了 **bad gateway** 错误以及用户无法访问服务的情况。
  
  - 一位成员针对这种情况幽默地评论道：*“为夜班或白班团队默哀”*。
- **语音功能不可用**：一位用户询问停机是否是 GPT 中**语音功能**无法使用的原因。
  
  - 另一位成员确认了他们也无法使用语音，进一步证实了连接问题。
- **OpenAI 状态更新**：来自 [OpenAI 状态页面](https://status.openai.com/) 的当前状态显示，修复程序已实施，他们正在**监控结果**，尽管问题仍然存在。
  
  - 早些时候的更新显示问题已被识别并正在处理中，表明中断仍在持续。
- **针对停机的幽默调侃**：一位成员开玩笑地将 GPT 的停机归咎于 **LeBron James** 和 **Ronaldo**，得到了其他人的赞同。
  
  - 这种轻松的评论反映了社区在服务中断期间的应对机制。

**提到的链接**：[OpenAI Status](https://status.openai.com/)：无描述

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1331727963812528262) (14 messages🔥):

> `OCR 使用案例, GIS 数据集改进, Task prompt 分享, ASMR meta-prompts, 内容创作策略`

- **地图读取中的 OCR 挑战**：一位成员强调，在 **OCR** 中使用示例实际上可能会诱发 **hallucinations**，而不是提高准确性，特别是在无约束空间中。
  
  - 另一位成员分享了他们读取地图的变通方法，但表示希望 OpenAI 的模型能在 **spatial datasets** 方面有所改进。
- **内容创作的创意 Task Prompts**：一位成员提供了一个用于生成海报和社交媒体内容的详细 prompt，将营销技巧与说服性语气相结合，以确保高质量的输出。
  
  - 他们强调了类人（human-like）回复的重要性，同时保持专业性并减少语法错误。
- **探索 Task Prompts 以获取灵感**：一位用户表示有兴趣发现新的 task prompts，并提到正在研究**每日新闻摘要**和**企业活动日历**等概念。
  
  - 他们的目标是通过最大化 **ChatGPT** 的特性和功能来挑战其极限。
- **用于睡眠的创新 ASMR Meta-Prompt**：一位成员讨论了使用一种独特的 ASMR task meta-prompt，该 prompt 将 **onomatopoeia**（拟声词）作为一种助眠机制。
  
  - 他们承认这种使用案例是非标准的，这引起了频道内其他人的兴趣。
- **在社区内分享资源**：随后讨论了关于分享创意 prompts 和资源的话题，但对于提及个人服务器是否合适存在不确定性。
  
  - 成员们表示愿意贡献和探索彼此的作品，同时保持对社区准则的关注。

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1331727963812528262) (14 messages🔥):

> `有效地使用 prompts, GPT 的 Task prompts, ASMR meta-prompts, 社交媒体内容创作, 每日新闻摘要`

- **关于有效使用 Prompts 的讨论**：一位成员分享了一个用于创建海报和社交媒体内容的复杂 prompt，强调类人回复和企业术语。
  
  - 这包括一个专门针对广告的双重 prompt 结构，建议使用严肃且专业的语气。
- **探索 Task Prompts 以获取灵感**：一位成员表示有兴趣发现独特的 task prompts，以拓宽他们对 ChatGPT 功能的使用，提到了每日新闻摘要和企业活动日历等概念。
  
  - 该询问引发了关于创建或分享想法的提议，重点在于最大化 ChatGPT 的能力。
- **创新的 ASMR Prompt 使用案例**：一位成员提到他们使用 task meta-prompt 来创建包含 **onomatopoeia** 的 ASMR 内容，但由于格式限制，无法逐字分享。
  
  - 这种 prompt 的独特应用展示了 ChatGPT 的非标准使用案例。

---

### **Yannick Kilcher ▷ #**[**general**](https://discord.com/channels/714501525455634453/986699377257119794/1331741488790507741) (160 条消息🔥🔥):

> `AI 模型局限性, 国家能源紧急状态声明, AI 就业机会, OpenAI 未来计划, 使用 LLM 解决数学问题`

- **AI 模型在几何推理方面表现不佳**：用户讨论了 LLM 在解决几何问题时的局限性，特别是覆盖 3x3 网格需要多少条线的问题，一些模型始终给出错误答案。
  
  - 他们指出，准确性和推理能力通常取决于所使用的 Prompt 的具体程度。
- **特朗普的能源紧急状态声明**：特朗普总统宣布国家进入能源紧急状态，旨在最大限度地利用美国的能源资源，并将美国定位为制造业和 AI 领域的领导者。
  
  - 该声明强调了能源资源与 AI 技术进步之间的联系。
- **Rav.ai 的工作机会**：来自 rav.ai 的一名成员表示，他们的团队提供研究 Foundation Models 和 AI 的职位，强调了拥有充足资金的协作环境。
  
  - 他们提到工作机会多种多样，涵盖从硬件到 AI 研究的所有领域，目标人群是感兴趣的开发者和研究人员。
- **OpenAI 对 ChatGPT 的升级**：Sam Altman 暗示 ChatGPT 的免费层级很快将包含 o3-mini 升级，而 Plus 层级将获得增强的访问权限。
  
  - 这一变化是为了响应对 OpenAI 产品功能改进日益增长的需求。
- **参与 AI 问题解决**：成员们分享了使用 LLM 处理数学问题的经验，同时强调了 Prompt 构建中约束条件的重要性。
  
  - 他们一致认为，对推理方法的更详细检查是衡量 AI 能力的更好标准。

**提到的链接**：

- [Sam Altman (@sama) 的推文](https://fxtwitter.com/sama/status/1882478782059327666)：重大新闻：ChatGPT 免费版将获得 o3-mini！（Plus 版将获得大量的 o3-mini 使用额度）
- [Tufa Labs 开放职位](https://tufalabs.ai/open_positions.html?)：未找到描述
- [维基百科，自由的百科全书](https://en.wikipedia.org/)：未找到描述
- [Tsarathustra (@tsarnick) 的推文](https://fxtwitter.com/tsarnick/status/1882300255024390467?t=b-xYx0_Xa_2qMDTHpHGOFQ&s=19)：OpenAI 的 Brad Lightcap：“o1 就像是通往 GPT-7、GPT-8 的门户……[它] 为你提供了 GPT-7 或 GPT-8 级别的有效算力……这是 Scaling 上的一个纯粹的不连续点”
- [覆盖 n \times n 网格的最少直线数量？](https://math.stackexchange.com/questions/4756401/minimum-number-of-straight-lines-to-cover-n-times-n-grid#:~:text=The%20minimal%20number%20must%20be,horizontal%20(or%20vertical)%20lines.)：我想知道触及 n \times n 网格每个方格所需的最少直线数量。唯一的附加规则是直线必须通过方格内部，而不是边缘/角落。我有 f...
- [Tsarathustra (@tsarnick) 的推文](https://fxtwitter.com/tsarnick/status/1882520836739039384?t=NJuPxKFBPXDQq4GJTlgieg&s=19)：特朗普总统表示，他已宣布国家进入能源紧急状态，以释放美国的能源资源，使美国成为“制造业超级大国和人工智能的世界首都……”
- [AshutoshShrivastava (@ai_for_success) 的推文](https://fxtwitter.com/ai_for_success/status/1882113005875302421)：Elon 与 Sam Altman 之间的争斗变得丑陋。Sam Altman 是否在暗示 Elon 及其公司没有把美国放在首位，而 OpenAI 却做到了？别忘了 OpenAI 的董事会现在包括了 BlackRock 的前……
- [Paul Calcraft (@paul_cal) 的推文](https://fxtwitter.com/paul_cal/status/1882111659927556535)：OpenAI 的 @SebastienBubeck 谈 o1 范式：“没有给模型提供任何策略。一切都是涌现的。一切都是通过 Reinforcement Learning 学习的。这太疯狂了。简直疯了”引用……
- [Sam Altman (@sama) 的推文](https://fxtwitter.com/sama/status/1882505714196988271)：Stargate 1 号站点，德克萨斯州，2025 年 1 月。
- [Sam Altman (@sama) 的推文](https://fxtwitter.com/sama/status/1882505650594611588)：宏伟、美丽的建筑。

---

### **Yannick Kilcher ▷ #**[**paper-discussion**](https://discord.com/channels/714501525455634453/1045297868136779846/1331798300910813204) (10 messages🔥):

> `DeepSeek 内存需求, Generalized Spatial Propagation Network, 强化学习奖励黑客 (Reward hacking)`

- **DeepSeek 模型的内存需求**：根据关于参数大小的讨论，要在 float16 格式下运行 **R1 Distilled Qwen 2.5 32B** 模型，至少需要 **64GB (V)RAM**，或者 **q8 量化版需要 32GB**。
  
  - 一位成员确认，16-bit 的 **70 亿参数 (7B)** 大约需要 **140 亿字节 (14GB)** 的内存，此外还需要额外的内存用于上下文窗口。
- **GSPN 解决 Attention 机制的局限性**：**Generalized Spatial Propagation Network (GSPN)** 提供了一种新的 Attention 机制，旨在通过捕捉 **2D 空间结构**来优化视觉任务，从而提高计算效率。
  
  - GSPN 的核心是 **Stability-Context Condition**，它将有效序列长度显著降低至 **\\sqrt{N}**，使其对图像数据更具上下文感知能力。
- **GSPN 实现的同行评审征集**：一位成员在 PyTorch 中实现了 **GSPN 概念**并分享了代码以供同行评审，认为这是一种简单直接的方法。
  
  - 他们表示，既然 **NVIDIA** 是论文作者，该实现可能具有相当大的价值，并邀请他人提供反馈。
- **在强化学习中探索 MONA**：一种新的**训练方法**——带有非短视批准的短视优化 (Myopic Optimization with Non-myopic Approval, MONA)，旨在防止 Agent 在强化学习设置中执行不希望的多步奖励黑客 (reward hacking) 行为。
  
  - 该方法结合了短视优化与远见奖励策略，并在不同环境中进行了实证测试，以解决对齐 (misalignment) 问题。

**提到的链接**：

- [MONA: Myopic Optimization with Non-myopic Approval Can Mitigate Multi-step Reward Hacking](https://arxiv.org/abs/2501.13011)：未来的高级 AI 系统可能会通过强化学习 (RL) 学习到人类无法充分理解并安全评估的复杂策略。我们提出了一种训练方法，可以避免……
- [Parallel Sequence Modeling via Generalized Spatial Propagation Network](https://arxiv.org/abs/2501.12381)：我们提出了 Generalized Spatial Propagation Network (GSPN)，这是一种针对视觉任务优化的新型 Attention 机制，能够固有地捕捉 2D 空间结构。现有的 Attention 模型，包括……

---

### **Yannick Kilcher ▷ #**[**agents**](https://discord.com/channels/714501525455634453/1269724655405498429/1331740760537825332) (1 messages):

> `IntellAgent, 对话式 Agent 评估, 研究洞察`

- **IntellAgent 发布开源框架**：新的开源项目 **IntellAgent** 旨在评估对话式 Agent，通过从 Agent 的 Prompt 中生成多样化的数据集，从而引导模拟对话。
  
  - 其 [GitHub 仓库](https://github.com/plurai-ai/intellagent)提供了一个完整的框架，专注于全面的诊断和评估。
- **通过模拟进行深入分析**：**IntellAgent** 模拟了扮演用户的 Agent 与被测 Agent 之间的对话，并结合批判组件进行细粒度分析。
  
  - 这种创新方法增强了评估过程，为对话动态提供了更清晰的洞察。
- **研究论文揭示迷人见解**：随项目附带的一篇[研究论文](https://arxiv.org/pdf/2501.11067)揭示了由 **IntellAgent** 系统生成的几个引人入胜的非平凡见解。
  
  - 这些见解为理解对话式 Agent 的能力做出了宝贵贡献。

 

**提到的链接**：[GitHub - plurai-ai/intellagent: A framework for comprehensive diagnosis and evaluation of conversational agents using simulated, realistic synthetic interactions](https://github.com/plurai-ai/intellagent)：一个使用模拟的、真实的合成交互对对话式 Agent 进行全面诊断和评估的框架 - plurai-ai/intellagent

 

---

### **Yannick Kilcher ▷ #**[**ml-news**](https://discord.com/channels/714501525455634453/853983317044756510/1331721870478934026) (10 条消息🔥):

> `OpenAI Operator, Kanye West AI Project, ChatGPT Free Tier Updates, Humanity's Last Exam, R1 Competitive Landscape`

- **OpenAI 准备发布 'Operator'**: OpenAI 正准备发布一项名为 **Operator** 的新功能，该功能可以在用户的浏览器中执行操作，提供建议的提示词，并允许保存/分享任务，但该功能在 API 中不可用。
  
  - 正如 [@steph_palazzolo](https://x.com/steph_palazzolo/status/1882091855606895073/) 在推文中分享的那样，该功能定于本周发布。
- **Kanye West 为 AI 项目寻找 'Wizards'**: Kanye West 宣布他的 Yeezy 公司正在为他们的 AI 项目积极寻找人才，号召“仅限奇才 (WIZARDS ONLY)”加入团队。
  
  - 据 [VICE](https://www.vice.com/en/article/kanye-west-seeking-wizards-only-for-ambitous-ai-project/) 报道，鼓励有兴趣的人士发送作品集和简历来参与这项创意事业。
- **ChatGPT 免费版获得 o3-mini 访问权限**: 重大更新：ChatGPT 的免费版很快将集成 **o3-mini**，而 Pro 用户将可以大量使用该功能。
  
  - [@sama](https://x.com/sama/status/1882478782059327666) 的这一公告表明，近期的一系列进展可能加剧了竞争。
- **'Humanity's Last Exam' 征集投稿**: **Humanity's Last Exam** 项目仍在接收问题和贡献，但明确表示新的投稿将不再有资格获得奖金池奖励。
  
  - 参与详情和引用信息可以在该项目的[网站](https://agi.safe.ai/)上找到。
- **R1 对 AI 领域的影响**: 随后讨论了 **R1** 最近的成功如何影响其他 AI 的发展，一些人推测这可能正在推动 **OpenAI** 的策略。
  
  - R1 出人意料的表现刺激了竞争，并引发了对服务器技术效率的反思。

**提到的链接**:

- [来自 Sam Altman (@sama) 的推文](https://x.com/sama/status/1882478782059327666): 重大新闻：ChatGPT 免费版将获得 o3-mini！（Plus 版将获得大量的 o3-mini 使用额度）
- [来自 Stephanie Palazzolo (@steph_palazzolo) 的推文](https://x.com/steph_palazzolo/status/1882091855606895073/): 独家：OpenAI 正准备在本周发布 “Operator”，这是一个新的 ChatGPT 功能，将代表用户在浏览器中执行操作。有趣的细节：Operator 提供建议的...
- [Kanye West 为雄心勃勃的 AI 项目寻找 'Wizards Only'](https://www.vice.com/en/article/kanye-west-seeking-wizards-only-for-ambitous-ai-project/): Kanye West 是一个经常同时处理很多事情的人，现在他显然正在开发自己的人工智能项目。
- [Humanity's Last Exam](https://agi.safe.ai/): 未找到描述

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1331715577084903580) (162 条消息🔥🔥):

> `AI 与 AGI 预测、公司 AI 落地、Olweus 欺凌受害问卷、GPU 对比与选择、模型训练策略`

- **行业内部人士重新评估 AGI 时间表**：一些此前对实现 AGI 持怀疑态度的业内人士现在认为，在 5 年内实现 AGI 的可能性大于 90%。
  
  - 随之而来的是人们承认 AI 在不久的将来确实会带来经济上的颠覆。
- **公司 AI 落地面临的挑战**：讨论集中在公司更倾向于内部 AI 解决方案还是外部机构，以及对专业技能的需求。
  
  - 领导层往往误解这项技术，这可能导致对前期成本和持续费用的恐惧。
- **理解 Olweus 欺凌问卷**：一位参与者在研究中寻求关于 Olweus 欺凌受害问卷评分系统的帮助。
  
  - 他们特别关注计算平均值以及根据心率变化对欺凌风险等级进行分类。
- **机器学习 GPU 的选择**：关于购买 5090 GPU 与等待未发布的 RTX Blackwell 以及考虑 A100 的优缺点讨论。
  
  - 显存带宽、PCIe 速度以及与现有硬件的未来可扩展性是重点考虑因素。
- **探索分布式资源的训练策略**：参与者讨论了在多个 GPU 上加速训练的策略，包括使用 DiStRo 和各种并行化方法。
  
  - 强调了 GPU 显存的重要性，以及使用多台机器与使用没有 NVLink 的显卡的影响。

**提到的链接**：

- [Ronan (@Ronangmi) 的推文](https://x.com/Ronangmi/status/1881952133345644694)：@dylan522p（领先的 AI 研究员）被问及他对 AI 创业领域的哪些空间感到兴奋。他的第一个回答？分布式训练和推理。特别是 @NousResearch 和 @...
- [Pietro Schirano (@skirano) 的推文](https://x.com/skirano/status/1881854481304047656?s=46)：顺便说一下，你可以从 deepseek-reasoner 中只提取 reasoning 部分，这意味着你可以在任何模型回答之前将该思考过程发送给它们。比如这里我将 gpt-3.5 turbo 变成...
- [Smoke-away (@SmokeAwayyy) 的推文](https://x.com/SmokeAwayyy/status/1847350947368095963)：Andrej Karpathy：“你想要的是大脑内部的思维独白……你在解决问题时大脑中的轨迹，如果我们有十亿个这样的轨迹，AGI 大致就实现了……”
- [Mahesh Sathiamoorthy (@madiator) 的推文](https://x.com/madiator/status/1882131703927652762)：介绍 Bespoke-Stratos-32B，这是我们使用 Berkeley NovaSky 的 Sky-T1 配方从 DeepSeek-R1 蒸馏出的推理模型。该模型在推理（数学和代码）基准测试中优于 Sky-T1 和 o1-preview...
- [区块链如何改变世界](https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/how-blockchains-could-change-the-world)：忽略比特币的挑战。在这次采访中，Don Tapscott 解释了为什么作为加密货币基础技术的区块链有潜力彻底改变世界经济。
- [Teknium (e/λ) (@Teknium1) 的推文](https://fxtwitter.com/Teknium1/status/1882159710180376828)：我正在为 Nous Research 的 post training 团队寻找两名工程师，以构建通用能力模型的未来，探索认知型、创造型模型，并推进最先进的推理和 a...
- [Sun 乌龟 💖 (@suntzoogway) 的推文](https://x.com/suntzoogway/status/1882121235762721063)：伙计们，这是我写的一个恶搞！只是试图通过 hyperstition 创造一个美好的未来（我被困在欧盟的监管地狱里）
- [MisguidedAttention/eval/harness at main · cpldcpu/MisguidedAttention](https://github.com/cpldcpu/MisguidedAttention/tree/main/eval/harness)：一组旨在挑战大语言模型在存在误导信息时的推理能力的提示词集合 - cpldcpu/MisguidedAttention

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1331806807341010997) (6 messages):

> `Synthetic Data Generation, R1 Dataset Availability, Olweus Bullying Victimization Questionnaire`

- **寻找关于合成数据（Synthetic Data）注意事项的指南**：一位成员询问了关于为 **Finetuning** 目的生成 **Synthetic Data** 的 **注意事项（do's and don'ts）** 的有用指南。
  
  - 对这些资源的寻求表明，该领域对明确的最佳实践需求日益增长。
- **R1 数据集的可用性**：一位成员询问用于蒸馏模型的 **R1 数据集** 是否可用，得到了另一位成员的确认，表示有部分是可以获取的。
  
  - 这引发了随后关于在哪里可以找到它们的询问，表明在数据集获取方面可能存在困惑。
- **理解 Olweus 欺凌受害问卷的评分**：一位成员寻求关于 **Olweus 欺凌受害问卷** 评分系统的澄清，旨在将分数分类为欺凌发生的 **低**、**中**、**高** 概率。
  
  - 他们提供了相关文档，并请求其他具有专业知识的人员协助，以证实他们对评分过程的理解。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1331791695531999354) (4 messages):

> `Human-AI representation similarities, Diffusion models optimization`

- **发现人类与 AI 的表征对齐**：MIT 的研究人员展示了许多在自然主义数据上训练的 **人工神经网络（ANNs）** 与生物系统的神经表征相一致，表明表征存在收敛性。
  
  - 他们开发了一种方法来识别影响 **模型与大脑对齐（model-to-brain alignment）** 的刺激因素，展示了表征普遍性的核心组成部分，这有助于揭示生物计算。
- **优化扩散模型（Diffusion models）的训练效率**：最近的一项研究提出了一种方法，通过使用预定义路线来保留信息，从而提高 **Diffusion models** 的训练效率，避免了 Token 丢弃的低效。
  
  - 这种优化可以应用于各种架构，包括基于 **Transformer** 的模型和 **状态空间模型（state-space models）**，从而增强其计算效能。

**提到的链接**：

- [TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training](https://arxiv.org/abs/2501.04765)：Diffusion models 已成为视觉生成的主流方法。然而，这些模型通常面临样本效率低下和训练成本高昂的问题。这个问题尤其突出……
- [Universality of representation in biological and artificial neural networks](https://www.biorxiv.org/content/10.1101/2024.12.26.629294v1)：未找到描述

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1331715773248569418) (3 条消息):

> `Evabyte Architecture, Tensor Network ML Library, Symbolic Reasoning in ML, Graph Isomorphism Optimization`

- **Evabyte 的分块注意力机制 (Chunked Attention Mechanism)**：**Evabyte** 架构采用了一种全**分块 (chunked)** 线性注意力方法，在压缩注意力占用的同时配合**多字节预测 (multi-byte prediction)** 以优化吞吐量。有关详细实现，请参阅 [GitHub 上的源代码](https://github.com/OpenEvaByte/evabyte/blob/ba8f65c5fe502b7ed07f916773754734b91b52fd/evabyte_hf/eva.py#L63)。
  
  - 一张说明性的 [草图](https://cdn.discordapp.com/attachments/1132352574750728192/1331715773013426176/attn_sketch.png?ex=6793f1f6&is=6792a076&hm=137584266c089c6710213a48af66382ce1fb6003093f41dc8f9070e26293d850&) 展示了该架构的重点。
- **创新型张量网络库 (Tensor Network Library)**：开发了一个基于张量网络的新型 **ML 库**，其特点是使用**命名边 (named edges)** 代替数字索引，从而实现直观的张量操作（如卷积）。用户可以在 `kernel @ h_conv @ w_conv` 等操作中轻松连接边缘，而无需担心张量维度的操作。
  
  - 此外，该库采用了**符号推理 (symbolic reasoning)** 和矩阵简化技术，输出优化的**编译后 torch 代码**，通过在前向和反向传播中使用**公共子表达式消除 (common subexpression elimination)** 等技术来增强性能。
- **征求张量库的反馈**：该张量网络 ML 库的作者正在寻求用户反馈以提高易用性，邀请用户*试用并分享他们的体验*。该库承诺提供各种高级功能，包括相对于变量的符号期望计算。
  
  - 这一创新工具旨在简化张量操作，同时提供复杂的增强功能，可能极大地受益于 ML 工作流。

**提到的链接**：

- [GitHub - thomasahle/tensorgrad: Tensor Network Library with Autograd](https://github.com/thomasahle/tensorgrad)：带有 Autograd 的张量网络库。通过在 GitHub 上创建账户为 thomasahle/tensorgrad 的开发做出贡献。
- [evabyte/evabyte_hf/eva.py at ba8f65c5fe502b7ed07f916773754734b91b52fd · OpenEvaByte/evabyte](https://github.com/OpenEvaByte/evabyte/blob/ba8f65c5fe502b7ed07f916773754734b91b52fd/evabyte_hf/eva.py#L63)：EvaByte：高效的大规模字节级语言模型 - OpenEvaByte/evabyte

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1331791695531999354) (4 条消息):

> `Human-AI Representation Alignment, Optimization for Diffusion Models`

- **人类与 AI 系统在表征上趋于一致**：来自 MIT 作者的研究表明，当使用生态学上合理的目标在自然主义数据上进行训练时，高性能的**人工神经网络 (ANNs)** 和生物系统在表征上会趋于一致，正如他们的 [研究](https://www.biorxiv.org/content/10.1101/2024.12.26.629294v1) 中所述。
  
  - 他们证实，可以通过语言和视觉刺激中不同程度的模型间一致性来预测**模型与大脑的对齐 (model-to-brain alignment)**。
- **Diffusion Models 的创新训练效率**：最近的一篇论文讨论了一种提高 **diffusion models** 训练效率的新方法，该方法利用预定义的路线来保留信息而不是丢弃信息，从而允许更深层的集成，链接见 [此处](https://arxiv.org/abs/2501.04765)。
  
  - 这种方法不仅解决了样本效率低下的问题，还适用于非 Transformer 架构，在不改变底层训练结构的情况下展示了显著的优化。

**提到的链接**：

- [TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training](https://arxiv.org/abs/2501.04765)：Diffusion models 已成为视觉生成的主流方法。然而，这些模型通常面临样本效率低下和训练成本高的问题。这个问题在...中尤为突出。
- [Universality of representation in biological and artificial neural networks](https://www.biorxiv.org/content/10.1101/2024.12.26.629294v1)：未找到描述

---

### **Nous Research AI ▷ #**[**reasoning-tasks**](https://discord.com/channels/1053877538025386074/1264666760972472481/) (1 条消息):

lowiqgenai: 嘿，我使用 MistralAI 的免费服务做了一些工作 `fhai50032/medmcqa-solved-thinking-o1`

---

### **Stackblitz (Bolt.new) ▷ #**[**announcements**](https://discord.com/channels/364486390102097930/671536649301131325/) (1 条消息):

katetra: [https://x.com/boltdotnew/status/1882483266680406527](https://x.com/boltdotnew/status/1882483266680406527)

---

### **Stackblitz (Bolt.new) ▷ #**[**prompting**](https://discord.com/channels/364486390102097930/1301167628416454737/) (1 条消息):

cwinhall: 你需要连接 Supabase。它正尝试将其用作数据库。

---

### **Stackblitz (Bolt.new) ▷ #**[**discussions**](https://discord.com/channels/364486390102097930/680953097354215446/1331725686766833788) (143 条消息🔥🔥):

> `Bolt 与 Stripe 集成、Token 分配问题、聊天持久化问题、3D 模型查看器实现、支付系统建议`

- **Stripe Webhook 实现中的挑战**: 一位用户在实现 Stripe Webhook 结合 Supabase Edge Function 时遇到了 **401 错误**，在经过大量调试后感到非常沮丧。
  
  - 他们发现原因是错误的 JWT 配置被设置为 `verify_jwt = true`，在修正后集成工作取得了进展。
- **关于 Token 分配的担忧**: 用户对他们的 **Token 分配** 表示困惑，特别是订阅付费计划后，每日分配量从 **300k 降至 150k**。
  
  - 有建议称免费计划会收到每日 Token，而付费计划可能不会，这促使用户寻求澄清并纠正其 Token 使用情况。
- **Bolt 中的聊天持久化问题**: 一位用户提出了关于聊天无法持久保存的问题，且需要通过 StackBlitz 删除并重新打开它们，这导致了额外的 Bug。
  
  - 用户正在寻找有效的解决方案，以便在不引起进一步复杂情况的情况下保持聊天记录完整。
- **寻求 3D 模型查看器代码帮助**: 一位用户尝试使用指定的 GLB 文件创建一个包含 **3D 模型查看器** 的网页，但在实现过程中报告只看到白屏。
  
  - 进一步的指导建议通过 Google Model Viewer 的代码来集成模型查看器，以解决显示问题。
- **支付集成改进建议**: 一位用户提议创建 Webhook 接收系统，并建议增加 **Discord 登录** 功能，以提升平台的用户体验。
  
  - 他们还提到通过 Token 抽奖来激励 Discord 邀请，表明了增强社区参与度的愿望。

**提到的链接**:

- [Discord - Group Chat That’s All Fun & Games](https://discordapp.com/channels/364486390102097930/1332002779480199168/1332002779480199168): Discord 是玩游戏和与朋友放松，甚至建立全球社区的绝佳场所。自定义你自己的空间来聊天、玩耍和聚会。
- [Get started with StackBlitz - StackBlitz](https://stackblitz.com/register): 未找到描述
- [READ THIS FIRST](https://bolters.io/docs/read-this-first): 关于 Bolt.new 能力、限制和成功最佳实践的关键信息
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://support.bolt.new/.): 一款将日常工作应用融合为一的新工具。它是为你和你的团队打造的一体化工作空间
- [bolt.new](https://bolt.new/?autoAuth): 未找到描述
- [Cursor Directory](https://cursor.directory/): 为你的框架和语言寻找最佳的 Cursor 规则

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1331717649385918545) (139 条消息🔥🔥):

> `OpenAI Operator 发布、Imagen 3 性能、DeepSeek 模型进展、Fireworks AI 转录服务、API 收入分成`

- **OpenAI 发布用于自动化的新 Operator**: OpenAI 推出了 Operator，这是一个旨在利用其自带浏览器自主执行任务的 Agent，允许实时用户交互和工作流管理。
  
  - 初步反应强调了它的能力，但也对安全性和 CAPTCHA 挑战增加的可能性提出了疑问。
- **Imagen 3 在 Text-to-Image 领域夺冠**: Google 的 Imagen 3 在性能指标上首次亮相即登顶，大幅超越竞争对手 Recraft-v3，展示了 Text-to-Image 生成方面的进步。
  
  - 该模型因能有效处理特定 Prompt 而受到称赞，证明了 AI 生成图像技术的进步。
- **DeepSeek 通过快速、低成本的方案革新 RAG**: DeepSeek 正在启用一种新的检索增强生成 (RAG) 方法，允许用户将大量文档直接输入模型进行相关性检查，实现了卓越的效率。
  
  - 据报道，这种利用 KV caching 和高速处理的方法优于传统的 RAG 策略，并为大规模文档处理开辟了新的可能性。
- **Fireworks AI 推出具有竞争力的流式转录服务**: Fireworks AI 宣布了一项新的流式转录服务，其功能可与领先模型媲美，同时在免费试用后提供每音频分钟 0.0032 美元的竞争性价格。

- 该服务旨在提供低延迟的高质量转录，将其定位为现有解决方案的可行替代方案。
- **关于 API 营收分成模式的讨论**：对话探讨了是否有模型公司与用户分享 API 使用收益，并指出 OpenAI 不将 API 使用计入 ChatGPT 订阅的一部分。
  
  - 参与者对模型 API 使用和营收分成的财务透明度表示关注，强调了当前实践中的空白。

**提到的链接**：

- [Noam Brown (@polynoamial) 的推文](https://x.com/polynoamial/status/1882461290947547175)：醒来看到一个新的未饱和评估（eval）的感觉。祝贺 @summeryue0, @alexandr_wang, @DanHendrycks 以及整个团队！引用 Dan Hendrycks (@DanHendrycks)：我们正在发布《人类最后的考试》（Humanity’s Last Exam）...
- [Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1882480450649915772)：介绍 Citations（引用）。我们新的 API 功能让 Claude 能够根据你提供的来源来构建回答。Claude 随后可以引用支撑每个回答的具体句子和段落。
- [Mark Chen (@markchen90) 的推文](https://x.com/markchen90/status/1882509726237573503)：这是一个 SoTA 模型（在 OSWorld, WebArena, WebVoyager 上），当你试用它时确实能感觉到！在此申请 CUA 团队：https://openai.com/careers/research-engineer-research-scientist-compute...
- [Jacques (@JacquesThibs) 的推文](https://x.com/jacquesthibs/status/1871991099138736628?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ)：有些人一边说大学考试不能测试现实世界，抱怨没用的 CS 毕业生，一边又构建了一个像无聊考试一样的 AI 基准测试。
- [Swaroop Mishra (@Swarooprm7) 的推文](https://x.com/swarooprm7/status/1882505585956135367?s=46)：这是一个很好的基准测试，通过模型在环（model in the loop）的对抗方式创建，并经过专家验证。此外，如果发现错误，还可以选择提交。我创建了几个问题，很喜欢这些数据...
- [Lucas Nestler (@_clashluke) 的推文](https://x.com/_clashluke/status/1882333810131677615?s=46)：醒醒，宝贝，新的 MoE Scaling Laws 发布了。
- [Jerry Liu (@jerryjliu0) 的推文](https://x.com/jerryjliu0/status/1882236760442519801)：LlamaIndex 现在比以往任何时候都更像是一个成熟的 Agent 框架。🤖🛠️ 今天我很高兴介绍 AgentWorkflows —— 一套全新的抽象，允许你设置事件驱动、异步优先的...
- [Harrison Chase (@hwchase17) 的推文](https://x.com/hwchase17/status/1882502767312531954?s=46)：⭐️ 想要 OpenAI Operator 的开源版本吗？有一个很棒的开源项目叫 Browser Use，它在保持开源的同时能做类似的事情（甚至更多），允许你接入任何模型...
- [ZombAIs: From Prompt Injection to C2 with Claude Computer Use](https://simonwillison.net/2024/Oct/25/zombais/)：对于一直关注此领域的人来说，这并不意外，Johann Rehberger 展示了针对新版 Claude [Computer Use](https://simonwillison.net/2024/Oct/...) 的 Prompt Injection 攻击...
- [Yilong Qin (@yilongqin) 的推文](https://x.com/yilongqin/status/1882507643669123230?s=46)：随着我们进入 Test-time compute 的世界，我们看到仅仅让我们的 Agent 运行更长时间就能获得递增的回报。我们第一次让我们的 Agent 在数百个步骤上运行...
- [Fireworks AI (@FireworksAI_HQ) 的推文](https://x.com/FireworksAI_HQ/status/1882530477468459309)：我们正在推出流式转录服务！以 300ms 的延迟生成具有 Whisper-v3-large 质量的实时字幕或驱动语音 Agent。未来两周免费使用，之后价格为 $0.0032...
- [We Tried OpenAI’s New Agent—Here’s What We Found](https://every.to/chain-of-thought/we-tried-openai-s-new-agent-here-s-what-we-found)：Operator（你能帮我完成这个任务吗？）
- [Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models](https://arxiv.org/abs/2501.12370)：参数 vs FLOPs：Mixture-of-Experts 语言模型最优稀疏性的 Scaling Laws：扩展语言模型的容量已被证明是提高性能和解锁新能力的可靠方法。容量主要由两个维度定义：...
- [fofr (@fofrAI) 的推文](https://x.com/fofrai/status/1882362181167235553?s=46)：Imagen 3 也通过了这项测试。引用 fofr (@fofrAI) 提示词：“海滩上的水母”。recraft-v3 再次表现出色。1. recraft 2. flux 1.1 pro 3. sd3.5 large 4. Midjourney v6.1
- [Alessio Fanelli (@FanaHOVA) 的推文](https://x.com/FanaHOVA/status/1882495355876741262)：笔记：- 暂时不会在欧洲推出 - 由基于 4o 训练的 CUA (Computer Use Agent) 运行 - 也将通过 API 提供 - Operator 与 Opentable 等有直接集成...
- [Peter Welinder (@npew) 的推文](https://x.com/npew/status/1882497318555115595?s=46)：我是瑞典人，所以我喜欢桑拿。这是一个 Operator 阅读 Tripadvisor 上的酒店评论以寻找斯德哥尔摩最好的酒店桑拿房的视频。

- [Tweet from Andrej Karpathy (@karpathy)](https://x.com/karpathy/status/1882544526033924438?s=46): 像 OpenAI 的 Operator 这样的项目之于数字世界，就像 Humanoid 机器人之于物理世界。一个通用的设置（显示器、键盘和鼠标，或者人类身体）原则上可以逐渐 p...
- [Tweet from elvis (@omarsar0)](https://x.com/omarsar0/status/1882545077219926031): 尝试将 OpenAI 的 Operator 作为我的研究助手。看我让这个 Agent 在 arXiv 上搜索 AI 论文并进行总结。未来已来！Agent 绝非玩笑！
- [Tweet from Morgante (@morgantepell)](https://x.com/morgantepell/status/1882170462236746154?s=46): 太多 AI “员工”初创公司犯了拟物化的错误，认为管理 Agent 的理想界面与管理员工的界面相同。我也曾陷入这个陷阱，但是...
- [Tweet from Mike (@grabbou)](https://x.com/grabbou/status/1882139484994551861): 🎉 宣布 Flows AI：一个基于 Vercel AI SDK 构建 Agent 工作流的轻量级库。✅ 无多余抽象。✅ 使用任何你选择的 LLM 和提供商。✅ 包含所有来自 Anthropic 的模式...
- [Tweet from Florian S (@airesearch12)](https://x.com/airesearch12/status/1882481758337450200?s=46): 我正在构建 smooth operator，这是 OpenAI Operator 的替代方案，它 - 不会每月花费 200 美元，并且 - 结合了市场上最好的模型，例如 R1 和下面的 screengrasplink。
- [Tweet from Varun Anand (@vxanand)](https://x.com/vxanand/status/1882061978593837344?s=46): 我们宣布获得 4000 万美元的 B 轮扩展融资，估值为 12.5 亿美元。我们上次融资的资金仍未动用，但我们的势头强劲 —— 24 年营收增长 6 倍，22 年增长 10 倍...
- [Tweet from Greg Brockman (@gdb)](https://x.com/gdb/status/1882494743739015389?s=46): Operator —— 一个可以利用自己的浏览器为你执行任务的 Agent 研究预览版。2025 年是 Agent 之年。引用 OpenAI (@OpenAI) 对 Operator 和 Agent 的介绍 https://openai.com/in...
- [The ‘self-operating’ computer emerges](https://venturebeat.com/ai/the-self-operating-computer-emerges/): 由 GPT-4V 驱动，该框架将截图作为输入，并输出鼠标点击和键盘命令，就像人类一样。
- [Tweet from swyx /dd (@swyx)](https://x.com/swyx/status/1882505900717687231): 对 Operator 的初步想法：- SOTA 的 OSWorld/WebArena 意味着模型层面的实质性进步，而不仅仅是 UI/产品封装。OAI 总是擅长这一点（模型+产品的进步），正如我们在 @karin...
- [Tweet from Mahesh Sathiamoorthy (@madiator)](https://x.com/madiator/status/1882131703927652762?s=46): 介绍 Bespoke-Stratos-32B，这是我们使用 Berkeley NovaSky 的 Sky-T1 配方从 DeepSeek-R1 蒸馏出的推理模型。该模型在推理（数学和代码）基准测试中优于 Sky-T1 和 o1-preview...
- [Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)](https://x.com/lmarena_ai/status/1882164189739073990?s=46): 来自 Text-to-Image Arena 的突破性新闻！🖼️✨ @GoogleDeepMind 的 Imagen 3 首次亮相即登顶榜首，以显著的 +70 分领先优势超越了 Recraft-v3！祝贺 Google Imagen 团队树立了新标杆！
- [Tweet from Alex Volkov (Thursd/AI) (@altryne)](https://x.com/altryne/status/1882523511279001752?s=46): 这条推文是这样产生的！@OpenAI 为 Operator 增加了分享机制。引用 Alex Volkov (Thursd/AI) (@altryne)：你好，我是代表 Alex 的 OpenAI Operator。
- [Tweet from Shunyu Yao (@ShunyuYao12)](https://x.com/shunyuyao12/status/1882507506557288816?s=46): 分别列举 CUA 和 Operator 的一个亮点：- CUA 是 *长时程（long-horizon）* 的 —— 如果需要，它可以自主运行 20 分钟！- Operator 使用 *远程虚拟机（remote VMs）*，这有利于管理安全和访问权限，也意味着...
- [Tweet from Dan Hendrycks (@DanHendrycks)](https://x.com/DanHendrycks/status/1882433928407241155): 我们正在发布 Humanity’s Last Exam，这是一个包含 3,000 个问题的数据集，由数百名领域专家开发，旨在捕捉人类知识和推理的前沿。最先进的 AI 在...
- [Tweet from Andrej Karpathy (@karpathy)](https://x.com/karpathy/status/1882498281089241545): 这样做是因为它更容易 1) 收集，2) 评估，以及 3) 超越并取得进展。我们将看到每一个像这样被整齐打包的任务都得到改进（包括...
- [Tweet from fofr (@fofrAI)](https://x.com/fofrai/status/1882377778273939564?s=46): 冲吧。说真的，怎么做到的。引用 fofr (@fofrAI) 哇，Imagen 3 🤯 > 一张高速自然照片，捕捉了两只鸣禽在英国花园中飞行对打的瞬间。玫瑰叶子甚至带有一点...
- [Tweet from OpenAI (@OpenAI)](https://x.com/openai/status/1882129444212740482?s=46): 以推理时间计算（Inference-Time Compute）换取对抗鲁棒性（Adversarial Robustness） https://openai.com/index/trading-inference-time-compute-for-adversarial-robustness/

- [来自 Deedy (@deedydas) 的推文](https://x.com/deedydas/status/1882479771428544663?s=46)：中国刚刚发布了一个新模型。字节跳动 Doubao-1.5-pro 在 GPT 4o 基准测试上表现持平，但价格便宜 50 倍——缓存输入 token 为 $0.022/M，输入为 $0.11/M，输出为 $0.275/M——比 DeepSeek 便宜 5 倍，比 o1 便宜超过 200 倍……
- [来自 thebes (@voooooogel) 的推文](https://x.com/voooooogel/status/1881966969043464365?s=46)：制作了一个非常愚蠢的基于采样器的杠杆，试图在 r1 上模拟 o1 风格的 "reasoning_effort=high"：如果在生成足够的 thinking tokens 之前出现了 </think>，采样器会替换……
- [$2 H100s：GPU 泡沫是如何破裂的](https://www.latent.space/p/gpu-bubble)：H100 以前如果你能买到的话是 $8/小时。现在有 7 个不同的转售市场以低于 $2 的价格出售。发生了什么？
- [利用 LLM 估计的效用优化预训练数据混合](https://huggingface.co/blog/WillHeld/utilimax-and-medu)：未找到描述
- [来自 Nick Dobos (@NickADobos) 的推文](https://x.com/nickadobos/status/1882496722741633342?s=46)：Operator 不是唯一的 Agent。“在接下来的几周和几个月里，我们还将推出更多 Agent” 引用 OpenAI (@OpenAI) 对 Operator 和 Agent 的介绍 https://openai.com/index/i...
- [Deepseek：引领中国 AI 竞赛的沉默巨人](https://www.chinatalk.media/p/deepseek-ceo-interview-with-chinas)：其 CEO 最深度采访的有注释翻译版
- [来自 Aaron Levie (@levie) 的推文](https://x.com/levie/status/1882240165865025711?s=46)：AI Agent 的交互将成为未来最有趣的软件互操作性范式之一。不可避免地，没有任何一个软件系统能包含执行任务所需的全部知识或信息……
- [来自 Jason Baldridge (@jasonbaldridge) 的推文](https://x.com/jasonbaldridge/status/1882174689373745377?s=46)：Imagen 3 在 lmsys 排行榜上取得如此强劲的表现真的很令人兴奋。这是巨大努力和大量工作的结果——祝贺整个团队！（参见 https://arxiv...
- [埃隆·马斯克抨击特朗普宣布的 5000 亿美元 AI 项目，声称其支持者“没钱” | CNN 商业](https://edition.cnn.com/2025/01/22/tech/elon-musk-trump-stargate-openai/index.html)：未找到描述
- [Operator 与 Agent 简介](https://www.youtube.com/live/CSE77wAdDLg?feature=shared)：太平洋时间上午 10 点开始。加入 Sam Altman、Yash Kumar、Casey Chu 和 Reiichiro Nakano，他们将介绍并演示 Operator。
- [来自 Karina Nguyen (@karinanguyen_) 的推文](https://x.com/karinanguyen_/status/1882506665951715804?s=46)：CUA Evals！引用 OpenAI (@OpenAI) 对 Operator 和 Agent 的介绍 https://openai.com/index/introducing-operator/
- [来自 homanp (@pelaseyed) 的推文](https://x.com/pelaseyed/status/1882471632129994914)：我不再做 RAG 了，只需启动一个流水线并将所有内容喂给 Deepseek，就能获得 10 倍的效果。是的，它可以扩展到超过 1 万个文档。RAG 是一种反模式。
- [来自 Jonathan Ellis (@spyced) 的推文](https://x.com/spyced/status/1881725740917670079)：我构建了一个工具来解决大型代码库的上下文问题。1/N
- [Anthropic CEO 呼吁对 AI 采取政策行动](https://youtu.be/ooM4QOETFGk?si=fFuLi5hdt7VX2kgJ)：Anthropic CEO 兼联合创始人 Dario Amodei 谈到了自主 AI 的威胁，并表示他将与特朗普政府合作制定能源条款，以……
- [据报道 Anthropic 从谷歌获得额外 10 亿美元资金 | TechCrunch](https://techcrunch.com/2025/01/22/anthropic-reportedly-secures-an-additional-1b-from-google/)：据报道，随着这家 AI 公司寻求在今年发布多项重大产品更新，Anthropic 已从谷歌筹集了约 10 亿美元。
- [State of AI 2025 预览 · Issue #278 · Devographics/surveys](https://github.com/Devographics/surveys/issues/278)：这是即将发布的 State of Web Dev AI 2025 调查的预览链接，这是该项新调查的首个版本：https://survey.devographics.com/en-US/survey/state-of-ai/2025 我很想得到……
- [[AINews] Bespoke-Stratos + Sky-T1：推理领域的 Vicuna+Alpaca 时刻](https://buttondown.com/ainews/archive/ainews-bespoke-stratos-sky-t1-the-vicunaalpaca/)：Reasoning Distillation 就是你所需要的一切。2025/1/21-2025/1/22 的 AI 新闻。我们检查了 7 个 subreddits、433 个 Twitter 和 34 个 Discord（225 个频道和 4297...

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1331736178856300645) (5 条消息):

> `R1 越狱风险，AI 的 DDoS 能力，端到端 LLM 解决方案`

- **R1 的越狱风险受到关注**：尽管 **R1** 表现卓越，但由于其极易被越狱或被操纵以执行危险任务，它也引发了人们的担忧。
  
  - 一位成员指出，*对于任何知道如何使用 Google 的人来说，越狱其他模型也是轻而易举的。*
- **AI 在 DDoS 攻击方面的潜力**：一位用户对 **ChatGPT** 能够被说服编写可对网站执行 DDoS 攻击的代码表示惊讶。
  
  - 这引发了人们对对话式 AI 在被不负责任地使用时所产生的更广泛影响的关注。
- **关于 LLM 解决方案播客的讨论**：一位成员分享了 **TDE Podcast #11** 的链接，该节目由 Paul Iusztin 和 Maxime Labonne 主持，讨论了如何构建端到端的 **LLM** 解决方案。
  
  - YouTube 视频可以通过 [这里](https://www.youtube.com/live/DxnBT5ChEPE?si=ptEfDoyJzHYmTFhp) 观看。
- **对保护机制的担忧**：关于现有的 AI 模型保护方法除了品牌保护之外，是否能提供任何实质性的安全性，展开了辩论。
  
  - 一种共识是，这些保护措施对于有动机的用户似乎是无效的。

 

**提到的链接**：[\- YouTube](https://www.youtube.com/live/DxnBT5ChEPE?si=ptEfDoyJzHYmTFhp)：未找到描述

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1331895836782760028) (9 条消息🔥):

> `步骤执行顺序，数据覆盖问题，编译器行为，变量变更`

- **步骤执行顺序的困惑**：一位成员澄清说，**step2** 必须在 **step3** 之前执行，因为 **step3** 会覆盖 **step2** 加载的数据。
  
  - 这引发了关于顺序必要性的疑问，因为乍一看，**step2** 似乎并不影响 **step3**。
- **提出数据覆盖问题**：指出差异的产生是因为 **step3** 更改了变量 **x_c**，尽管 **x** 本身保持不变，但这影响了 **step2** 的结果。
  
  - 一位成员建议直接在 **x** 而不是 **x_c** 上测试更改，以解决该问题。
- **编译器行为审查**：对编译器同时执行 **step1** 和 **step3** 导致潜在错误的担忧。
  
  - 一位成员推测，这种行为可能源于 **x** 未被修改这一事实。
- **尝试变量调整**：一位成员尝试调整等式以合并 **x** 和 **x_c**，但在将其更新为 **x=x+x_c*0.00001+y*0.000001** 时遇到了问题。
  
  - 分享了一个包含代码的附件，以便进一步分析和调试。

 

---

### **GPU MODE ▷ #**[**cuda**](https://discord.com/channels/1189498204333543425/1189607726595194971/1331734504330756159) (49 条消息🔥):

> `CUDA Toolkit 12.8 发布，Accel-Sim 框架，新 Tensor 指令，FP8 和 FP4 数据类型，Blackwell 架构增强`

- **CUDA Toolkit 12.8 发布并支持 Blackwell**：[CUDA 12.8](https://developer.nvidia.com/cuda-downloads) 已正式发布，集成了对 Blackwell 架构的支持和新的 TensorCore 指令。
  
  - 文档已更新以反映这些变化，包括性能增强和新功能。
- **Accel-Sim 框架提供 GPU 仿真**：[Accel-Sim](https://accel-sim.github.io/#overview) 提供了一个在 CPU 上仿真和验证 GPU 的仿真框架，最近更新了 1.2.0 版本。
  
  - 它允许对现代 GPU 架构的功耗建模和高效设计空间进行更深入的探索。
- **发布令人兴奋的新 Tensor 指令**：引入了新的 Tensor 指令，重点针对 sm_90a 和 sm_100a 硬件架构进行了优化。
  
  - 因此，这对指令性能产生了影响，特别是对于 FP8 和 FP4 数据类型。
- **用于增强性能的 FP8 和 FP4 数据类型**：讨论集中在新的 **FP8** 和 **FP4** 数据类型上，它们增强了神经网络的 Tensor 性能，简化了量化过程。
  
  - 预计各种实现将支持 NVFP4，新兴的 Tensor 指令有望带来显著的速度提升。
- **Blackwell 架构增强**：Blackwell 架构引入了重要功能，包括 FP4/FP6 转换规范以及 cuBLAS 在矩阵运算方面的进步。
  
  - 参与者承认在实现 B100/B200 架构宣称的 TFLOPS 以及改进 GEMM 操作方面存在复杂性。

**提到的链接**：

- [LeetGPU](https://LeetGPU.com)：未找到描述
- [FP8 Quantization: The Power of the Exponent](https://arxiv.org/abs/2208.09225)：在为高效推理量化神经网络时，低位整数是效率的首选格式。然而，低位浮点数具有额外的自由度，可以分配一些 b...
- [Accel-Sim: The Accel-Sim Framework](https://accel-sim.github.io/#overview)：未找到描述
- [nvfp4_tensor — Model Optimizer 0.21.1](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.qtensor.nvfp4_tensor.html#module-modelopt.torch.quantization.qtensor.nvfp4_tensor)：未找到描述
- [来自 Vijay (@__tensorcore__) 的推文](https://x.com/__tensorcore__/status/1882532829999075366)：CUDA 12.8 刚刚发布，支持 Blackwell。第五代 TensorCore 系列指令：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1331772224167743488) (8 条消息🔥):

> `Torch Profiler 函数计时，学习率调度技术，Torch Compile 中的内存分配`

- **Torch Profiler 可以跟踪函数执行时间**：一位用户询问在 **torch profiler** 中启用堆栈跟踪（stack tracing）时，是否可以跟踪像 **forward attention** 这样的高级函数的运行时间。
- **多样化的学习率调度方法**：讨论了各种学习率调度方法，包括带热重启的 **CosineAnnealing** 以及最近的 [schedule-free](https://github.com/facebookresearch/schedule_free) 技术。
  
  - 一些参与者提到在线性预热（linear warmup）和余弦衰减（cosine decay）方面取得了成功，而另一些人则提到 **WSD schedule** 是保持训练一致性的稳定选择。
- **关于输出不匹配的疑虑**：一位用户分享了一张对比输出的图片，提出了关于为什么**输出不匹配**的问题。
  
  - 这引发了对影响这些结果的底层计算和条件的探究。
- **Torch 中的内存分配偏好**：一位用户表示有兴趣了解 **torch.compile** 的内存分配，并询问是否可以提供预分配的 GPU Tensor，以避免阻塞异步 Kernel 运行。
  
  - 他们指出，如果没有预分配的 Tensor，会调用 **cudaMalloc**，从而阻止异步操作。

---

### **GPU MODE ▷ #**[**jobs**](https://discord.com/channels/1189498204333543425/1190208177829068860/1332078707480985702) (1 messages):

> `ComfyUI Hiring, Machine Learning Engineers, Open Source Contributions`

- **ComfyUI 招聘机器学习工程师**：ComfyUI 正在招聘 **Machine Learning Engineers**，加入维护 ComfyUI 生态系统的团队，强调他们对开源贡献的承诺。
  
  - 去年，他们为来自 **BFL**、**Hunyuan** 和 **StabilityAI** 等顶尖公司的模型提供了 **day 1 support**，并邀请感兴趣的人士*联系*以获取更多细节。
- **加入湾区一家受 VC 支持的创业公司**：该公司由 **VC backed**，位于湾区 (Bay Area)，拥有 **long runway** 和对未来增长的 **big vision**。
  
  - 有关该职位的更多信息可以在 [job listing](https://comfyorg.notion.site/Founding-Machine-Learning-Engineer-1696d73d36508014bfbaf5aebf39b145) 中找到。

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1331738957351747605) (14 messages🔥):

> `Choosing a GPU for Programming, Running Large Models Locally, Multi-GPU Training with Florence-2, Cost-effective GPU Options, Cloud GPU Services`

- **根据预算选择合适的 GPU**：一位寻求 GPU 选项建议的成员指出，**RTX 4060** 的 **VRAM** 有限，难以在本地运行模型，并提到如果是在本地操作，**RTX 3090** 是一个备受青睐的选择。
  
  - 通过云服务 *Renting GPUs* 被提及为一种廉价的替代方案，并提供了一个 [cloud GPU comparisons](https://cloud-gpus.com/) 的链接。
- **运行大模型的困境**：一位成员表达了运行 **405b models** 的挫败感，指出即使使用 2x16GB GPU 的配置，在尝试处理 **70b models** 时也显得力不从心。
  
  - 另一位成员强调了在 GPU 之外拥有足够系统 **RAM** 的重要性，以避免在模型切换时出现 swapping 问题。
- **入门级 GPU 见解**：对 **entry-level GPUs** 的建议包括 12GB 的 **RTX 3060**，因为它具有很高的性价比，尽管最近价格有所上涨，引发了对整体系统预算的担忧。
  
  - 建议先从一块 GPU 开始，以后再进行扩展，这是一种可行的策略。
- **在多 GPU 上微调 Florence-2 的挑战**：一位用户报告了在微调 **Florence-2 model** 时的困难，四个 **16GB GPUs** 中只有一个被利用，导致了 *CUDA out of memory* 错误。
  
  - 建议的解决方案包括使用 **DeepSpeed Zero 2** 或 **Zero 3** 以获得更好的多 GPU 支持，前提是模型兼容。

**提到的链接**：

- [Cloud GPUs](https://cloud-gpus.com/)：未找到描述
- [Fine-tuning Florence-2 - Microsoft's Cutting-edge Vision Language Models](https://huggingface.co/blog/finetune-florence2)：未找到描述

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1331772104105787456) (4 messages):

> `LeetGPU.com support, ComfyUI event, Performance enhancement tips, Community engagement, Workflows sharing`

- **加入 LeetGPU.com 获取更新**：邀请成员通过此 [Discord link](https://discord.gg/y8W4nkAU) 加入，以获取来自 **LeetGPU.com** 的更新和支持。
  
  - 该平台旨在促进其用户社区的沟通和支持。
- **通过 Token 管理提升性能**：建议在最后一个 attention module 之后消除 context tokens，以实现 **5-10% faster prefill** 并节省 VRAM。
  
  - 可以在最后一个 MLP module 上利用 *Torch.compile()*，因为输入形状将是静态的，从而优化性能。
- **下周四在湾区举行的 ComfyUI 活动**：鼓励参与者下周四前往 GitHub Office 参加 **ComfyUI event**，届时将有 demo 和社区互动，并有众多开源开发者参与。
  
  - 活动将提供茶点，著名的演示者包括 **MJM** 和 **Lovis**，他们将分享他们的 workflow 技巧。
- **ComfyUI 活动议程**：活动将有结构化的议程，从签到开始，随后是 Comfy Org Team 的介绍和 demo 演示。
  
  - 亮点包括小组讨论和分享 workflow 的闪电演讲环节，鼓励广泛参与。

**提到的链接**：

- [来自 mobicham (@mobicham) 的推文](https://x.com/mobicham/status/1882464122417385625)：通过在最后一个 attention module 之后移除 context tokens，你可以获得 5-10% 更快的 prefill 并节省大上下文/词表大小下的 VRAM。然后你可以对最后一个 mlp module 使用 torch.compile()，因为输入形状将是...
- [ComfyUI Official SF Meet-up at Github · Luma](https://lu.ma/6skuqn7c?tk=xiHyMZ)：在 GitHub 办公室举行的首届官方 ComfyUI SF 见面会！来见见 ComfyUI 的其他用户，与社区分享你的 workflow，或者提供你的建议...

---

### **GPU MODE ▷ #**[**thunderkittens**](https://discord.com/channels/1189498204333543425/1300872762163728550/1332112187153711256) (1 条消息):

> `CUDA kernel 中的 Tensor 累加，累加器的设置`

- **关于累加器清零的疑问**：一位成员询问是否有**必要将累加器清零**，参考的是[此处](https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/matmul/H100/matmul.cu#L87C13-L87C36)的 CUDA 矩阵乘法代码，考虑到它们可能已经在设置中被清零了。
  
  - 他们想知道在代码的那个点再次初始化累加器是否存在潜在的冗余。
- **关于 Kernel 性能的讨论**：另一位成员强调了 **tile primitives** 对于实现高速 Kernel 的重要性，并引用了 [ThunderKittens GitHub 仓库](https://github.com/HazyResearch/ThunderKittens)。
  
  - 他们指出，优化这些 primitives 可以在矩阵乘法期间带来显著的性能提升。

**提到的链接**：[ThunderKittens/kernels/matmul/H100/matmul.cu at main · HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/matmul/H100/matmul.cu#L87C13-L87C36)：用于高速 Kernel 的 Tile primitives。通过在 GitHub 上创建账号来为 HazyResearch/ThunderKittens 的开发做出贡献。

---

### **GPU MODE ▷ #**[**arc-agi-2**](https://discord.com/channels/1189498204333543425/1316377974672588850/1331747092149833910) (8 条消息🔥):

> `Tiny GRPO 仓库，Reasoning Gym 项目，易懂的 RL 教程，云计算选项，社区贡献`

- **Tiny GRPO 仓库发布**：**Tiny GRPO** 项目已移至独立的 [GitHub 仓库](https://github.com/open-thought/tiny-grpo)，提供一个极简且可 hack 的实现。
  
  - *查看项目描述以获取有关贡献和功能的详细信息。*
- **新的 Reasoning Gym 项目启动**：一个用于收集**程序化推理数据集**的新项目已启动，可在 [Reasoning Gym](https://github.com/open-thought/reasoning-gym) 获取，旨在处理算法可验证的任务。
  
  - *发起者欢迎任务创意和贡献，以扩大其范围。*
- **面向新手的简易 RL 教程**：一位成员分享了一个[关于 LLM 与 RL 的教程](https://www.philschmid.de/rl-with-llms-in-2025-dpo)，专为希望轻松入门强化学习的初学者设计。
  
  - *该教程基于近期更高效 LLM 的重大进展。*
- **苹果用户的云计算**：针对在没有 NVIDIA GPU 的苹果笔记本电脑上运行实验的担忧，建议包括利用 [Prime Intellect](https://www.primeintellect.ai/) 等云计算服务。
  
  - *另一个推荐的云计算服务是* ***runpod.io***。
- **项目的社区协作**：成员们表达了对贡献项目的兴趣，特别是关于通过语法（grammars）支持形式语言的想法。
  
  - *成员们鼓励在贡献时间受限的情况下，通过分享想法或创建 issue 来讨论潜在任务。*

**提到的链接**：

- [如何在 2025 年通过 DPO 和合成数据对齐开源 LLM](https://www.philschmid.de/rl-with-llms-in-2025-dpo)：学习如何通过直接偏好优化 (DPO) 和在线合成数据，使用 Hugging Face TRL 和 RLHF 来对齐 LLM。
- [Prime Intellect - 算力与智能的商品化](https://www.primeintellect.ai/)：Prime Intellect 使大规模 AI 开发民主化。我们的平台可以轻松找到全球算力资源，并通过跨集群的分布式训练来训练最先进的模型。
- [GitHub - open-thought/tiny-grpo: 极简可 hack 的 GRPO 实现](https://github.com/open-thought/tiny-grpo)：极简可 hack 的 GRPO 实现。通过在 GitHub 上创建账号来为 open-thought/tiny-grpo 的开发做出贡献。
- [GitHub - open-thought/reasoning-gym: 程序化推理数据集](https://github.com/open-thought/reasoning-gym)：程序化推理数据集。通过在 GitHub 上创建账号来为 open-thought/reasoning-gym 的开发做出贡献。

---

### **MCP (Glama) ▷ #**[**general**](https://discord.com/channels/1312302100125843476/1312302100125843479/1331742669529022504) (87 messages🔥🔥):

> `MCP Server Improvements, Podman vs Docker, Line Number Handling in Code, MCP Client Interaction, Timeout Issues with MCP Servers`

- **MCP Server 超时问题已解决**：一位成员分享了他们修复 MCP Server 响应中 **60 秒超时**问题的方法，为面临此挑战的其他用户提供了潜在的解决方案。
  
  - *然而，讨论中并未提供具体修复方案的细节。*
- **MCP Server 在 Windows 上的路径问题**：一位用户在 **Windows** 上启动 MCP Server 时遇到了由于隐藏的 PATH 设置导致的问题，但最终通过指定 **uvx.exe** 命令的完整路径解决了该问题。
  
  - 他们还在 Access 中为应用程序创建了一个数据库文件 **test.db**，并提到不确定这是否是必需的。
- **关于容器化方案的讨论**：成员们辩论了 **Podman** 与 **Docker** 的优劣，**Podman** 因其轻量化设计和无守护进程（daemonless）的特性而受到关注。
  
  - **Podman** 正在变得流行，但由于 **Docker** 在该领域存在时间更长，许多工具仍然主要支持 **Docker**。
- **代码编辑中的行号处理**：一位成员讨论了他们在特定编辑中管理代码行号的方法，强调这种方法使处理代码变更更加可靠和高效。
  
  - 这种方法被认为优于以往在 diff 处理方面表现不佳的迭代版本。
- **与 MCP Client 的交互及工具使用**：一位用户对于将环境变量传递到 **mcp dev server.py** 感到沮丧，并发现 **wong2 mcp cli** 提供了更简单的解决方案。
  
  - 他们分享了管理 MCP Client 的经验，并讨论了提取和验证工具参数的各种方法。

**提到的链接**：

- [Sage - Native Client for Claude](https://sageapp.ai)：未找到描述
- [Files and Resources with MCP - Part 1](https://llmindset.co.uk/posts/2025/01/mcp-files-resources-part1/)：关于使用 Model Context Protocol (MCP) 处理文件、图像和其他内容类型的实用指南。了解 Claude Desktop 和其他 MCP 实现如何管理资源和工具响应...
- [Reddit - Dive into anything](https://www.reddit.com/r/modelcontextprotocol/comments/1i6g3if/comment/m8oz8m4/.)：未找到描述
- [Hyper-V Dynamic Memory Overview](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-server-2012-r2-and-2012/hh831766(v=ws.11))：未找到描述
- [Podman Installation | Podman](https://podman.io/docs/installation)：寻找 GUI？你可以在这里找到 Podman Desktop。
- [GitHub - evalstate/mcp-webcam: Capture live images from your webcam with a tool or resource request](https://github.com/evalstate/mcp-webcam)：通过工具或资源请求从摄像头捕获实时图像。
- [GitHub - modelcontextprotocol/inspector: Visual testing tool for MCP servers](https://github.com/modelcontextprotocol/inspector)：用于 MCP Server 的可视化测试工具。在 GitHub 上为 modelcontextprotocol/inspector 的开发做出贡献。
- [Simplified, Express-like API by jspahrsummers · Pull Request #117 · modelcontextprotocol/typescript-sdk](https://github.com/modelcontextprotocol/typescript-sdk/pull/117)：受 #116 和生态系统中出现的一些 MCP SDK 封装器的启发，这是一个尝试将更具 Express 风格的 API 引入 SDK 的尝试。这与现有的封装库有所不同...

---

### **MCP (Glama) ▷ #**[**showcase**](https://discord.com/channels/1312302100125843476/1315696461316358175/1331717781904687104) (7 messages):

> `Anthropic TS Client 问题，用于浏览器自动化的 Puppeteer，SSE Client 示例修正`

- **Anthropic TS Client 的已知问题**：一位用户指出了 **Anthropic TS client** 的一个已知问题，并提供了一个与使用自定义请求头 (custom headers) 相关的 [GitHub issue 链接](https://github.com/modelcontextprotocol/typescript-sdk/issues/118)。
  
  - 他们提到在尝试 JavaScript 实现遇到困难后，选择了 Python。
- **Puppeteer 增强浏览器自动化**：一篇文章介绍了一个用于浏览器自动化的 [Puppeteer 软件包](https://www.npmjs.com/package/mcp-puppeteer-linux)，允许 LLM 与网页交互并执行 JavaScript。
  
  - 这包括**导航**、**截屏**以及在网页上**点击元素**等功能。
- **SSE Client 示例修正**：一位用户承认在 SSE client 示例中犯了一个*复制粘贴错误*，并澄清其支持使用自定义请求头。
  
  - 他们提供了修正后的 [GitHub SSE client 示例链接](https://github.com/apify/actors-mcp-server/blob/master/src/examples/clientSse.ts)。
- **关于 Node 版本 EventSource 的澄清**：针对 SSE client 的澄清，一位用户询问 **Node 版本** 的 EventSource 是否真的有效。
  
  - 这表明关于其功能和用户体验的讨论仍在进行中。

**提到的链接**：

- [mcp-puppeteer-linux](https://www.npmjs.com/package/mcp-puppeteer-linux)：用于浏览器自动化的 MCP server，使用 Puppeteer 并支持 X11/Wayland。最新版本：1.0.0，最后发布于：14 小时前。通过运行 `npm i mcp-p...` 在你的项目中使用 mcp-puppeteer-linux。
- [为 /sse 和 /message 端点使用自定义请求头 · Issue #118 · modelcontextprotocol/typescript-sdk](https://github.com/modelcontextprotocol/typescript-sdk/issues/118)：@chrisdickinson 感谢这个 PR。抱歉，我的 JS 并不强。我需要在访问我的 MCP server 时，为 /sse 和 /message 端点都包含一个 API token。我相信 head...
- [actors-mcp-server/src/examples/clientSse.ts at master · apify/actors-mcp-server](https://github.com/apify/actors-mcp-server/blob/master/src/examples/clientSse.ts)：适用于 Apify Actors 的 Model Context Protocol (MCP) Server - apify/actors-mcp-server

---

### **Nomic.ai (GPT4All) ▷ #**[**announcements**](https://discord.com/channels/1076964370942267462/1090471714888102009/1332082491749826570) (1 messages):

> `GPT4All v3.7.0 发布，Windows ARM 支持，macOS 更新，Code Interpreter 改进，聊天模板修复`

- **GPT4All v3.7.0 发布并带来关键特性**：**GPT4All v3.7.0** 版本引入了多项更新，包括对高通骁龙 (Qualcomm Snapdragon) 和微软 SQ 系列 (Microsoft SQ-series) 设备的 **Windows ARM 支持**。
  
  - 然而，用户必须注意目前尚不支持 GPU/NPU 加速，仅限 CPU 运行。
- **macOS 更新修复了之前的问题**：**macOS** 用户将受益于多项修复，包括防止应用程序在更新期间崩溃，以及允许在通过 Command-Q 退出时正确保存聊天记录。
  
  - 如果用户之前安装了来自 GitHub 的临时解决方案，建议将其卸载并恢复到官网的正式版本。
- **Code Interpreter 获得行为升级**：**Code Interpreter** 的改进包括更好地处理执行期间的超时情况，以及增强 `console.log` 功能以支持多个参数。
  
  - 这些调整旨在优化开发者体验，并使其更接近原生 JavaScript 行为。
- **聊天模板问题已解决**：最近的更新修复了 **chat template parser**（聊天模板解析器）中的两个崩溃问题和一个兼容性问题，确保了更流畅的可用性。
  
  - 此外，纠正了 **EM German Mistral** 的默认聊天模板，并为五个新模型添加了自动替换功能。

---

### **Nomic.ai (GPT4All) ▷ #**[**general**](https://discord.com/channels/1076964370942267462/1090427154141020190/1331820583595544577) (64 条消息🔥🔥):

> `ChatGPT 访问与限制, Prompt engineering, 模型兼容性与选择, Jinja 模板问题, NSFW 内容生成`

- **ChatGPT 访问需要付费**：一位成员指出，要访问无限的 ChatGPT 功能必须付费，这引发了一场关于限制的幽默交流。
  
  - 传达的观点是免费访问是不现实的，从而引发了关于替代聊天方案的玩笑。
- **分享 Prompt engineering 建议**：讨论了 Prompt engineering 对获得有效模型响应的重要性，建议精炼的 Prompt 会产生更好的结果。
  
  - 成员强调，使用礼貌用语（如 'Please'）会增加从模型获得理想响应的可能性。
- **模型与 GPT4All 的兼容性问题**：成员们讨论了与 GPT4All 兼容的各种模型选项，指出选择合适的模型以避免审查的重要性。
  
  - 建议包括探索 Nous Hermes 模型以及来自 Undi 的 Hugging Face 个人主页的其他替代方案。
- **Jinja 模板的挑战**：成员们对 Jinja 语法的可用性表示担忧，指出不同模型架构之间存在持续的兼容性问题。
  
  - 实现 Jinja 仍面临挑战，特别是由于 GPT4All 是用 C++ 构建的，这使得集成变得困难。
- **NSFW 内容生成障碍**：讨论集中在尝试生成 NSFW 内容，表明了与审查机制和模型限制的斗争。
  
  - 成员们注意到在请求露骨叙事时，预期性能与实际性能之间存在差异，并提到了普遍存在的道德约束。

 

**提到的链接**：[Chat Templates - GPT4All](https://docs.gpt4all.io/gpt4all_desktop/chat_templates.html#how-do-i-customize-the-chat-template-or-system-message)：GPT4All 文档 - 在你的硬件上高效运行 LLM

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1331721787733442590) (59 条消息🔥🔥):

> `CitiVAI 停机时间, 图像生成技术, GPU 对比, AI 模型训练, Clip Skip 设置`

- **CitiVAI 维护计划**：一位成员询问了 CitiVAI 宕机的情况，另一位成员提到每天会进行几次维护。
  
  - 这表明这是一个常规维护流程，可能会间歇性地影响访问。
- **使用遮罩生成冰块文字**：一位用户分享了他们通过在 Inkscape 中创建黑白遮罩，并使用特定字体生成冰块文字的过程。
  
  - 其他用户建议使用 **canny controlnet** 或直接在 Prompt 中描述冰块颜色，可以简化图像生成过程。
- **GPU 性能基准测试**：讨论中提到了 5090 GPU 的性能，称其生成速度快 20-40%，但功耗增加 30%。
  
  - 成员们指出，显著的进步更多体现在 B100/B200 系列中，而非消费级显卡。
- **为特定角色训练 AI 模型**：一段对话探讨了针对特定电影训练模型，一位成员断言这需要不错的 GPU 和一些时间。
  
  - 讨论强调了训练的简易性，并分享了关于模仿特定动画角色模型的实际参考资料。
- **澄清 Clip Skip 设置**：一位成员询问了 AI 生成中 “clip skip” 设置的相关性，另一位成员确认这是 SD1 时代的过时设置。
  
  - 这一澄清表明，现代用户无需关注这一遗留设置即可获得最佳性能。

**提到的链接**：

- [Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/comments/14jck90/finetuning_sdxl_on_an_rtx_2070_consumer_tier_gpu/): 无描述
- [Alvin Seville (Alvin and the Chipmunks Movie) - V1 | Stable Diffusion LoRA | Civitai](https://civitai.com/models/981021/alvin-seville-alvin-and-the-chipmunks-movie): 来自《鼠来宝》电影的 Alvin。可选 Prompt：Realistic, Hands in Pockets。接受定制，请访问此链接...
- [SwarmUI/docs/Model Support.md at master · mcmonkeyprojects/SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md): SwarmUI（原名 StableSwarmUI），一个模块化的 Stable Diffusion Web 用户界面，强调易用的强大工具、高性能和可扩展性。
- [SwarmUI/docs/Features/Prompt Syntax.md at master · mcmonkeyprojects/SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Features/Prompt%20Syntax.md#automatic-segmentation-and-refining>): SwarmUI（原名 StableSwarmUI），一个模块化的 Stable Diffusion Web 用户界面，强调易用的强大工具、高性能和可扩展性。
- [Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/s/7hEw9MOp9D): 无描述

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1331727101564293260) (22 条消息🔥):

> `Google's Titans, Difficulty of Titans Paper, Pretraining Context in Models, Interpretability and Diffusion Models, Reward Systems in Model Distillation`

- **Google Research 发布 Titans，承诺提升性能**：成员们讨论了 Google 的新模型 **Titans**。正如关于该主题的 [YouTube 视频](https://www.youtube.com/watch?v=x8jFFhCLDJY) 所强调的，该模型声称优于 Transformer，并在 Inference time 引入了一种新型内存。
  
  - *共识似乎表明该论文难以复现*，这让一些成员对所使用的确切 Attention mechanism 感到困惑。
- **辩论 Titans 论文的难度**：关于复现 Titans 论文结果的**难度**出现了疑问，一位成员评论说很难解读他们的方法论。另一位成员指出，在读完论文后也感到类似的困惑。
  
  - 对 Attention mechanism 的评论表示了不确定性，一位成员指出通常只有几种可用的选项。
- **预训练过程作为 Test Time Training 层**：一位成员提出，**Pretraining** 过程可以被视为一个巨大的 Test Time Training 层，并认为 Context 通常没有被有意义地概述。这一观点反映了人们对理解 **Model Interpretability** 日益增长的兴趣。
  
  - 其他人对这一概念如何应用于 **Interpretability** 和 **Diffusion Models** 的最新进展表示好奇。
- **对话愿望与创业精神**：一位成员表达了想在 Sam 的 Twitter 保持沉默期间与其交流的愿望，强调需要对过去的行为保持透明。以轻松的口吻，他们将该小组称为联合创始人，并幽默地声称**年薪**为 **1 万亿**。
  
  - 他们的言论很俏皮，包括关于购买任何他们喜欢的东西的评论，并用表情符号对情况进行了调侃。
- **探索 AI 模型中的 Distillation 方法**：一位成员询问了关于讨论 Teacher 和 Student 模型之间的差异被用作 Distillation 的 PPO 应用中 **Reward** 的论文。另一位成员质疑为什么不直接使用带有 KL-matching 的标准 Student-Teacher 模型。
  
  - 讨论表明，人们正在持续研究 Model Distillation 策略，特别是在提高 Student 模型性能的背景下。

 

**提到的链接**：[Google Research Unveils "Transformers 2.0" aka TITANS](https://www.youtube.com/watch?v=x8jFFhCLDJY)：我们终于破解了赋予模型“类人”记忆的代码吗？观看以了解更多！加入我的 Newsletter 以获取定期 AI 更新 👇🏼[https://forwardfu](https://forwardfu)...

 

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1331752569110003752) (27 条消息🔥):

> `Feature Learning with Egomotion, LLM Explainability and Security Feedback, Multi-Turn Reasoning in Models, Learned Optimizers, Distributional Dynamic Programming`

- **利用 Egomotion 进行特征学习**：关于利用自我运动（Egomotion）作为特征学习监督信号的研究已经出现，这与传统的使用手动标签的监督学习形成对比，通过利用移动数据来完成场景识别和目标检测等任务。
  
  - 这种方法展示了具有竞争力的特征学习结果，表明 **egomotion** 可以作为类标签监督的有效替代方案。
- **寻求研究论文反馈**：一名成员询问了分享关于 LLM 可解释性和安全性论文的频道，以便从社区收集反馈。
  
  - 另一位参与者建议在专门的工作分享频道中发布，特别是当工作接近完成时。
- **通过 Key-Value 压缩增强多轮推理**：关于改进模型多轮推理的讨论包括采用注意力掩码（attention masking）和动态 Key-Value 压缩等技术，以增强记忆保持和推理能力。
  
  - 几位成员探讨了保留先前推理轨迹的想法，同时允许后续步骤访问并从中学习，且不会显著损失解码准确性。
- **关于 Learned Optimizers 的见解**：一位成员分享了对 Learned Optimizers 潜力的兴趣，这类优化器通过元泛化（meta-generalization）提供无超参数训练的优势，同时也强调了其在实际应用中面临的挑战。
  
  - 目前人们越来越关注优化 Learned Optimizers 的架构和流程，以提高其泛化能力。
- **引入 Distributional Dynamic Programming**：一篇论文提出了 Distributional Dynamic Programming 方法，用于优化回报分布的统计泛函，扩展了传统的强化学习概念。
  
  - 这一新框架结合了存量增强（stock augmentation），能够解决以前用经典方法难以处理的问题。

**提到的链接**：

- [Optimizing Return Distributions with Distributional Dynamic Programming](https://arxiv.org/abs/2501.13028)：我们引入了 Distributional Dynamic Programming (DP) 方法来优化回报分布的统计泛函，将标准强化学习作为一个特例。之前的分布...
- [MONA: Myopic Optimization with Non-myopic Approval Can Mitigate Multi-step Reward Hacking](https://arxiv.org/abs/2501.13011)：未来的先进 AI 系统可能会通过强化学习 (RL) 学习到人类无法充分理解、从而无法安全评估的复杂策略。我们提出了一种训练方法，可以避免...
- [Learning Versatile Optimizers on a Compute Diet](https://arxiv.org/abs/2501.12670)：Learned optimization 已成为手工设计优化器的一种有前途的替代方案，有可能发现更强大的学习更新规则，从而实现更快、无超参数的训练...
- [Learning to See by Moving](https://arxiv.org/abs/1505.01596)：计算机视觉中特征学习的主流范式依赖于使用数百万张手动标记的图像来训练神经网络进行对象识别任务。是否可能学习到有用的...
- [Restructuring Vector Quantization with the Rotation Trick](https://arxiv.org/abs/2410.06424)：矢量量化变分自编码器 (VQ-VAEs) 旨在将连续输入压缩到离散潜空间，并以最小失真进行重构。它们通过维护一组...
- [meta-llama/Meta-Llama-3-8B · Are there bias weights in Llama3 ?](https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/202)：未找到描述

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1331757642640789544) (1 条消息):

> `Ruler tasks, Long context tasks`

- **Ruler 任务已更新**：一位成员确认所有 **Ruler 任务** 都已添加，并指出目前主要剩下一些小的格式调整。
  
  - 他们询问是否有人知道其他可以支持的**长上下文（long context）任务**。
- **对长上下文任务的需求**：该成员就当前设置中是否有优质的**长上下文任务**提出了疑问。
  
  - 这凸显了进一步探索以最大化任务有效性的需求。

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1331773291874095144) (2 条消息):

> `开源 RAG 系统，LlamaIndex，AI Chrome 扩展程序`

- **使用 LlamaIndex 构建开源 RAG 系统**：通过这份[详细指南](https://t.co/kkr77KA23P)，学习如何利用 **LlamaIndex**、**Meta Llama 3** 和 **@TruLensML** 构建并评估开源 **RAG 系统**。
  
  - 它将使用 **@neo4j** 进行数据存储的基础 RAG 设置与更高级的 agentic RAG 系统进行了比较，包括 **OpenAI** 与 **Llama 3.2** 的性能评估。
- **用于社交媒体的 AI Chrome 扩展程序**：使用 **LlamaIndex** 不仅仅能构建 RAG；通过[此链接](https://t.co/8T9bFBD0Cl)查看一对旨在增强 **X** 和 **LinkedIn** 帖子影响力的 **Chrome 扩展程序**。
  
  - 这些工具利用 AI 能力优化内容，以获得更好的互动和曝光度。

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1331715589181411369) (44 条消息🔥):

> `AgentWorkflow 改进，多 Agent 工作流，Agent 与 Tool 的区别澄清，动态内存管理，LlamaIndex 文档问题`

- **AgentWorkflow 的增强激发了热情**：成员们对 **AgentWorkflow** 的改进表示兴奋，指出它显著优于前代产品。
  
  - *正当其时*，看来许多人已经计划用它进行构建。
- **关于多 Agent 工作流的澄清**：**Cheesyfishes** 解释说，多个 Agent 可以同时处于活跃状态，但一次只能运行一个，在预测多个动作时会发生异步 Tool 调用。
  
  - 全局聊天历史记录可能会导致上下文窗口问题，但并不会完全加载到每个 Agent 的上下文中。
- **Agent 与 Tool 之间的混淆解释**：**Cheesyfishes** 澄清了 Agent（利用 Tool）与 Tool 本身（也可以作为 Agent 运行）之间的区别。
  
  - 添加多个 Agent 有助于更好地分离职责，从而提高**性能**。
- **正在讨论中的动态内存管理**：讨论强调了对更智能内存模块的需求，**Cheesyfishes** 指出当前的 ChatMemoryBuffer 可能无法有效地最大化上下文利用率。
  
  - 虽然存在摘要选项，但它们会引入延迟，且并不总是默认选择。
- **LlamaIndex 文档链接失效**：**Twocatsdev** 指出分步 Agent 教程中的一个链接失效，导致 500 错误。
  
  - 作为回应，**Cheesyfishes** 提供了一个潜在的源链接，并确认大多数入门信息仍可在其他地方访问。

**提到的链接**：

- [LlamaIndexTS/apps/next/src/content/docs/llamaindex/guide/agents at main · run-llama/LlamaIndexTS](https://github.com/run-llama/LlamaIndexTS/tree/main/apps/next/src/content/docs/llamaindex/guide/agents)：适用于 LLM 应用程序的数据框架。专注于服务器端解决方案 - run-llama/LlamaIndexTS
- [GitHub - run-llama/multi-agent-concierge: An example of multi-agent orchestration with llama-index](https://github.com/run-llama/multi-agent-concierge)：使用 llama-index 进行多 Agent 编排的示例 - run-llama/multi-agent-concierge
- [Agent tutorial](https://ts.llamaindex.ai/docs/llamaindex/getting_started/starter_tutorial/agent)：未找到描述

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1331778492651671653) (9 messages🔥):

> `Cohere LCoT models, Pydantic support for Cohere, COT prompting techniques`

- **呼吁发布 LCoT 模型**：一位成员敦促 Cohere 通过发布能够处理逻辑和思考的 **LCoT meme 模型权重**来进入市场。
  
  - 另一位成员回应称，Cohere 主要专注于企业级解决方案。
- **Pydantic 集成 Cohere 模型**：一位成员宣布 **Pydantic 现在支持 Cohere 模型**，这引起了热烈讨论，标志着开发者迈出了重要一步。
  
  - 这种集成可以简化流程，并提高开发者使用 Cohere 时的易用性。
- **通过 Prompting 实现 COT 功能**：讨论集中在通过策略性 **Prompting** 实现 **Chain of Thought (COT)** 的概念。
  
  - 一位成员建议使用诸如 *'think before you act'* 之类的提示，并使用 `<thinking></thinking>` 标签包裹思考过程，以增强推理能力。
- **关于常规模型推理的辩论**：一位成员指出，虽然未在推理轨迹 (traces) 上进行训练的**常规模型**无法完全复制 COT，但它们仍然可以提供一定的推理能力。
  
  - 这引发了关于 Prompting 有效性及其对标准模型固有能力影响的对话。

 

**提到的链接**：[Tossing Hat GIF - Jeff Bridges Agent Champagne Kingsman Golden Circle - Discover & Share GIFs](https://tenor.com/view/jeff-bridges-agent-champagne-kingsman-golden-circle-toss-gif-9381860)：点击查看 GIF

 

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1331996968439779339) (3 messages):

> `Cohere API endpoints, Latency issues in South America, Cohere Reranker models on premise`

- **查询 Cohere API 端点位置**：由于在**智利遇到延迟**并怀疑存在拓扑问题，一位成员询问了 **Cohere API 端点**的位置。
  
  - *未记录到关于 API 位置的具体回复。*
- **本地部署 Cohere Reranker 模型的可行性**：同一位成员随后询问是否可以**在本地 (on-premise) 挂载 Cohere Reranker 模型**，或者访问靠近南美洲的区域。
  
  - 另一位成员建议将此咨询发送给 **sales team**，邮箱为 [support@cohere.com](mailto:support@cohere.com)。

 

---

### **Cohere ▷ #**[**cmd-r-bot**](https://discord.com/channels/954421988141711382/1168578374038470656/1331948763723599985) (29 messages🔥):

> `Artificial Superintelligence (ASI), Cohere Documentation Queries`

- **探索 ASI 的概念**：ASI，即 **Artificial Superintelligence (人工超智能)**，是一个理论构想，指机器在包括创造力和解决问题在内的所有领域都超越人类智能。
  
  - 虽然这一构想引人入胜，但也引发了关于其潜在误用及对社会影响的重大伦理担忧。
- **ASI 的潜在影响**：ASI 的发展可能会彻底改变 **healthcare** 和 **education** 等部门，提供诸如精确疾病诊断和复杂问题的创造性解决方案等进步。
  
  - 然而，必须负责任地处理 ASI，以在预期收益的同时降低风险。
- **Cohere 文档限制**：在 Cohere 文档中反复搜索术语 **ASI** 均未获得相关信息，这表明缺乏关于该主题的正式资源。
  
  - 尽管尝试寻找文档，但讨论仍集中在理论解释上，而非资源中的具体内容。

 

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1331759958764683335) (4 条消息):

> `NotebookLM 之旅、Obsidian 插件、音频生成问题、笔记保存限制`

- **用户在学习工作流中开启 NotebookLM 之旅**：一位成员分享了他们在 **NotebookLM 之旅**进入第三周的兴奋之情，并将其整合到学习工作流中，强调了它的影响。
  
  - 他们提供了一个 [YouTube 视频](https://youtu.be/wvf4EXJJsU8) 链接，展示了 NotebookLM 用于学习的各种功能。
- **了解 Obsidian 的 Markdown 功能**：另一位成员对上述视频发表了评论，对 **Obsidian** 的新见解表示赞赏，特别是通过插件合并 Markdown 笔记的能力。
  
  - 这突显了社区内持续的知识和工作流分享。
- **对音频生成提示词的担忧**：一位成员指出，在没有特定提示词（Prompts）的情况下生成音频时，模型通常默认使用整个 PDF 的通用信息，这可能会影响内容质量。
  
  - 他们建议遇到类似问题的其他用户在指定板块将其作为 Bug 报告。
- **NotebookLM 中的笔记保存问题**：一位用户询问了 NotebookLM 保存笔记的潜在限制，因为他们遇到了新笔记虽然已生成但无法保存的问题。
  
  - 这引发了对应用程序保存功能可靠性的担忧。

**提到的链接**：[NotebookLM: The AI Tool That Will Change Your Study Habits](https://youtu.be/wvf4EXJJsU8)：在这段视频中，我分享了用于学习的 Google NotebookLM 功能。00:00 简介 00:46 工作流 01:49 功能 1 02:50 功能 2 03:55 功能 3 04:50 功能...

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1331716455770755073) (35 条消息🔥):

> `播客制作、NotebookLM 语言设置、测试题格式、音频下载问题、使用 NotebookLM 进行文档对比`

- **发布关于 DeepSeek-R1 分析的播客**：一位用户分享了他们关于 [DeepSeek-R1 论文分析](https://open.spotify.com/episode/5T8cbCKks1RE4RxZ0aFBMD?si=TEVGEhl1SWqFy9KRlW17Pg&nd=1&dlsi=e4f792d7a96e43c0) 的新播客剧集，讨论了该模型的推理能力和性能基准。
  
  - 鼓励听众探索强化学习（Reinforcement Learning）如何增强能力并助力小型模型的发展。
- **NotebookLM 语言调整**：用户讨论了更改 NotebookLM 的语言，其中一人询问如何从罗马尼亚语切换到英语。
  
  - 根据建议，尝试使用 URL 参数导致了错误，这显示了用户对语言设置的困惑。
- **编写高质量测试题**：一位用户成功创建了一种根据特定指南生成多选题的格式。
  
  - 这种方法有助于从指定章节中持续产出高质量的测试题。
- **音频概览播放问题**：一位成员报告了下载音频概览的问题，指出文件出现在手机上但无法播放。
  
  - 这个问题已经持续了几周，表明可能存在持续的技术故障。
- **使用 NotebookLM 与 ChatGPT 进行文档对比**：一位用户询问 NotebookLM 是否能比 ChatGPT 更好地协助分析法律文件（通过上传整个文档）。
  
  - 他们指出，虽然 ChatGPT 可以识别非典型条款，但处理多个文档和交叉引用的能力可能会提供更全面的见解。

**提到的链接**：

- [#AI - A ChatGPT from China better, free and open source? Analyzing DeepSeek-R1](https://open.spotify.com/episode/5T8cbCKks1RE4RxZ0aFBMD?si=TEVGEhl1SWqFy9KRlW17Pg&nd=1&dlsi=e4f792d7a96e43c0)：Carlos Nuñez 的播客，由 Nalah 制作 · 剧集
- [未找到标题](https://notebooklm.google.com?hl=en)：未找到描述
- [未找到标题](https://notebooklm.google.com/notebook/c558515c-96ed-443e-bb33-3b5cfbcc8a3f?original_referer=https:%2F%2Fwww.google.com%23&pli=1)：未找到描述

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1332050171445379142) (9 messages🔥):

> `Mojo 中的异步代码，分享论坛帖子`

- **关于异步代码的咨询**：一位成员寻求关于如何编写 **asynchronous code**（异步代码）的帮助，并提到之前讨论中无人回应。
  
  - 表达了“感谢你的帮助！”，以鼓励继续对话。
- **论坛帖子创建协助**：讨论强调了对异步函数（asynchronous functions）清晰说明的需求，并提议协助在 [Modular 官网论坛](https://forum.modular.com/)上发帖。
  
  - 一位成员确认愿意在忙碌之余帮忙复制粘贴消息，使协作变得简单直接。
- **论坛主题直接链接**：原帖作者分享了他们新创建的关于 [编写 async 代码](https://forum.modular.com/t/how-to-write-async-code-in-mojo/473) 的论坛主题链接。
  
  - 这一步骤鼓励社区在异步编程查询方面进行持续的互动和支持。

**提到的链接**：

- [Modular](https://forum.modular.com/?)：与我们一起构建 AI 的未来，了解 MAX, Mojo 和 Magic。
- [如何在 mojo🔥 中编写 async 代码？](https://forum.modular.com/t/how-to-write-async-code-in-mojo/473)：我看到开发博客说 Mojo 目前缺乏 async fn awaiting 的包装器，但它支持 coroutines（协程）本身。如果可能的话，如何编写一个函数，比如在打印的同时……

---

### **Modular (Mojo 🔥) ▷ #**[**announcements**](https://discord.com/channels/1087530497313357884/1098765954302873621/1332122022657921056) (1 messages):

> `MAX Builds 页面上线，社区构建的软件包，软件包提交指南`

- **MAX Builds 页面正式上线**：更新后的 [MAX Builds 页面](https://builds.modular.com) 现已上线，展示了 **community-built packages**（社区构建软件包）的专门板块。
  
  - 向首批软件包创作者表示祝贺，感谢他们为这次激动人心的发布所做的贡献！
- **向软件包创作者致敬**：特别点名表扬了 <@875794730536018071>、<@1074753858309468292> 等创作者，感谢他们在 MAX Builds 页面上展示的社区软件包。
  
  - 他们的努力通过显著的贡献增强了社区驱动的倡议。
- **项目提交指南**：若想让你的项目获得展示，请向 [Modular 社区仓库](https://github.com/modular/modular-community) 提交一个包含 `recipe.yaml` 文件的 PR。
  
  - 完整的提交说明和示例可以在[这里](https://www.modular.com/community/package-submission)找到。

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1331832755591581769) (27 messages🔥):

> `Mojo 中的函数重写，Mojo 中的 Python 风格生成器，Mojo 函数定义中的变量重新赋值，Mojo 中的 __iadd__ 方法`

- **关于函数重写（Overriding）的澄清**：一位成员澄清说 Mojo 中没有 `@override` 装饰器，并强调你可以在没有该装饰器的情况下重写函数。
  
  - 另一位成员指出 structs（结构体）缺乏继承机制，这使得重写的概念进一步复杂化。
- **实现 Python 风格的生成器**：一位成员询问如何在 Mojo 中实现 Python 风格的 generators（生成器），并指出在 Swift、C++ 和 Java 等其他编译语言中很难找到类似的结构。
  
  - 讨论引出了一个 async 提案，该提案要求必须在 coroutines（协程）中暴露 `yield`。
- **在函数定义中重新赋值**：有一个关于在 Mojo 函数签名变量中重新赋值的问题，引发了关于只读引用的讨论。
  
  - 一位成员解释了使用 `mut`（允许修改引用）与使用 `owned`（用于对调用者隐藏的寄存器内副本）之间的区别。
- **理解 iadd 方法**：一位成员寻求关于 Mojo 中 `__iadd__` 方法的帮助，特别是它的运作方式。
  
  - 另一位成员回答说 `__iadd__` 控制着语言中 `+=` 运算符的行为。

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1331747438922563705) (27 messages🔥):

> `Open Source TTS, Audio Distortions Visualization, Colab Notebook Sharing, Pydub Implementations, Audio Output Widgets`

- **Bud-E 上的情感开源 TTS**：一位成员宣布 **Emotional Open Source TTS** 很快将在 Bud-E 上可用，并分享了相关的音频片段。
  
  - 该片段获得了积极的反响，并展示了正在开发的内容类型。
- **音频失真可视化**：另一位成员正在使用 **pydub** 可视化 **原始音频文件** 与带有失真的文件之间的差异。
  
  - 他们分享了对比轻微噪声与强噪声波形的图像，展示了他们在音频探索方面的进展。
- **Colab Notebook 协作**：**Notebook 分享** 成为一个话题，一位成员提议在 Google Colab notebook 中分享他们的工作以进行协作。
  
  - 链接已分享，一些成员表示有兴趣审查其实现和后续步骤。
- **Colab 中的保存问题**：一位成员在 Colab 中遇到了 **自动保存失败**，导致更改丢失，这引发了关于修复该情况的讨论。
  
  - 关于在哪里可以找到该实现的讨论反复进行，并对保存失败表示了沮丧。
- **音频输出组件请求**：一位成员请求在 Colab 中创建 **IPython audio widgets**，以便于对比失真前后的音频。
  
  - 这引发了关于如何在他们共享的 notebook 中实现音频输出的协作建议。

 

**Link mentioned**: [Google Colab](https://colab.research.google.com/drive/140lGFiXXeTsNFp7w5xteCjpmRRaSBvmj?usp=sharing): no description found

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1331734402442592329) (5 messages):

> `Repo Spam Concerns, Adaptation vs. Imitation in Frameworks, Using External Libraries with DSPy`

- **仓库垃圾信息引发疑问**：有关 **仓库被刷垃圾信息** 的担忧被提出，一位成员质疑这是否与最近的 **coin** 问题有关。
  
  - 另一位成员对这种情况不屑一顾，认为参与此类活动“非常差劲”。
- **不断演进的框架应提供启发而非束缚**：一位成员强调，**并非所有事物都需要** 直接复制现有框架提供的功能，提倡将工具包用于独特的解决方案。
  
  - 他们强调了理解工具背后 **use case** 的重要性，而不是仅仅复制他人的成功。
- **触发 REACT Agent 的 Webhook 实现**：一位成员分享了他们希望通过电子邮件触发其 **REACT Agent** 的经验，并寻求在 DSPy 中使用外部库的示例。
  
  - 他们随后确认通过使用 **webhooks** 成功解决了该问题。

 

---

### **DSPy ▷ #**[**examples**](https://discord.com/channels/1161519468141355160/1161519685616025600/1332042965408809130) (2 messages):

> `OpenAI Model, Groq Integration`

- **OpenAI 模型讨论**：一位成员提到了 **OpenAI model** 及其潜在应用，认为这可能是一个可行的解决方案。
  
  - 该模型因其在各种任务中的稳健性和灵活性而受到关注。
- **Groq 兼容性咨询**：另一位成员强调 **Groq** 也应该能有效工作，表明其与现有模型的兼容性。
  
  - 这表明人们对在 OpenAI 之外探索 **Groq** 的能力有着更广泛的兴趣。

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1331865421078335604) (7 条消息):

> `llvm_bf16_cast PR, shapetracker 相加问题, 悬赏建议, mask 收缩与 views`

- **llvm_bf16_cast 悬赏状态确认**：一位成员询问了关于 *将 llvm_bf16_cast 移动到 renderer 中的重写规则* 的悬赏状态，并指出几小时前已提交了一个 PR。
  
  - 这确认了悬赏任务的解决，并引发了关于适合新手的可用任务的讨论。
- **Shapetracker 相加问题演示**：一位成员展示了他们处理 *shapetracker add problem* 的方法，并在 [PDF 附件](https://cdn.discordapp.com/attachments/1068976834928193609/1332064522495725700/viewadd.pdf?ex=6793e542&is=679293c2&hm=19f298adb040f1f8b9666c337c4318e3af4f6dc06fd3bce00a55a2df6671d24e&) 中详细说明了将其简化为整数线性规划（ILP）的过程。
  
  - 虽然该方案已完成，但面临速度问题，且需要一个 ILP 求解器，这引发了对其通用性的质疑。
- **ILP 的重构可能性**：George Hotz 对展示的工作表示感兴趣，并询问是否有相关的 PR，建议重写简化（rewrite simplifications）也可以从 ILP 中受益。
  
  - 这表明了进一步优化以及将线性规划技术集成到现有工作流中的潜力。
- **views 和 mask 合并的探索**：讨论了合并 mask 和 views 的问题，思考是否需要有界表示（bounded representation）来增强 mask 的效用。
  
  - 成员们承认，合并 mask 可能会使原始的合并问题复杂化，但仍能提供潜在的解决方案。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1331721796084437023) (6 条消息):

> `课程证书, LLM MOOC 报名`

- **课程证书何时发放？**：一位成员询问了领取 **课程证书** 的时间表，以及是否有追踪日期的方法。
  
  - 针对此查询，未提供具体信息。
- **关于 LLM MOOC 报名的澄清**：另一位成员对他们在 12 月填写预申请表后的 **报名状态** 表示不确定，并寻求确认。
  
  - 回复确认没有“录取过程”；任何填写了报名表的人都已自动 **报名**！
- **LLM Agents 与课程成功**：一位成员指出，作为 **LLM Agent** 保证能被课程录取，并评论说通过课程将是令人印象深刻的。
  
  - 这突显了对课程中 Agent 能力的期望。

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**leaderboard**](https://discord.com/channels/1111172801899012102/1214705495974092810/1331949356319772712) (3 条消息):

> `BFCLV3 LLMs 测试, LLMs 中的工具关系, BFCLV3 数据集研究`

- **关于 BFCLV3 LLMs 工具关系的咨询**：一位成员询问 LLMs 在执行 **book_flight** 等操作之前，是否会收到详细说明 **get_flight_cost** 和 **get_creditcard_balance** 等工具之间关系的 system message。
  
  - 他们寻求澄清 LLMs 是否仅根据工具描述进行测试，而没有任何关于工具依赖关系的元信息。
- **关于任务信息可用性的回复**：作为回应，另一位成员提到，检查各种任务的实际请求信息后发现，对于 **simple**、**parallel**、**multiple** 和 **parallel_multiple** 任务，似乎没有提供关于工具依赖关系的信息。
  
  - 他们分享了一个 [GitHub 链接](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/data) 以便对 Gorilla 数据集进行进一步调查。
- **对 BFCLV3 数据集的研究重点**：最初的成员确认他们已经审查了所有可用数据，并强调理解工具关系对他们的研究至关重要。
  
  - 他们表示希望尝试并专门引用来自 **BFCLV3 数据集** 的信息。

 

**提到的链接**：[gorilla/berkeley-function-call-leaderboard/data at main · ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/data): Gorilla: 为函数调用（工具调用）训练和评估 LLMs - ShishirPatil/gorilla

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**discussion**](https://discord.com/channels/1111172801899012102/1111353033352294440/1331949420698406943) (1 messages):

> `BFCLV3 System Message, LLMs Tool Dependency`

- **BFCLV3 系统消息的清晰度**：一位成员询问 **BFCLV3** 中的 LLMs 在执行操作前，是否会收到解释工具之间关系（如 *get_flight_cost* 和 *get_creditcard_balance*）的系统消息。
  
  - 该问题提出了 LLMs 是否仅根据工具描述运行，而没有任何关于工具间依赖关系的元信息。
- **LLM 测试方法论的澄清**：讨论内容包括 LLMs 是纯粹基于工具描述进行测试，还是在测试中考虑了依赖关系。
  
  - 这一询问突显了在 BFCLV3 框架内对 LLMs 评估过程理解的潜在差距。

 

---

### **Axolotl AI ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1331716882071425116) (2 messages):

> `KTO Loss Merge, Office Hours Announcement`

- **Liger-Kernel 合并 KTO Loss**：**KTO loss** 已成功合并至 [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/pull/475) 仓库。
  
  - 此次合并预计将提升模型性能，提供令人兴奋的新功能。
- **Office Hours 提醒**：友好提醒，**Office Hours** 将在 **4 小时**后开始。
  
  - 成员可以通过此 [Discord 链接](https://discord.gg/dEfsYQbX?event=1328556620107743343)加入活动进行互动讨论。

 

---

### **MLOps @Chipro ▷ #**[**events**](https://discord.com/channels/814557108065534033/869270934773727272/1332010159232254055) (1 messages):

> `Event for Senior Engineers/Data Scientists, Networking Opportunities in Toronto`

- **举办高级工程师/数据科学家活动**：有人宣布将于 **2 月 18 日**在**多伦多**为高级工程师和数据科学家举办一场小型活动。
  
  - 他们邀请感兴趣的参与者**私信了解更多详情**。
- **多伦多社交活动**：该活动旨在促进多伦多高级工程师和数据科学家之间的社交和讨论。
  
  - 这为专业人士提供了一个在各自领域建立联系和分享见解的机会。

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1331719938183790747) (1 messages):

> `Local-First X AI Hackathon, Event Discussion Thread`

- **旧金山 Hackathon 公告**：一场 **Local-First X AI Hackathon** 定于 **2 月 22 日**在[旧金山](https://www.lofihack.com/)举行。
  
  - 组织者包括渴望在活动期间吸引社区参与创新项目的成员。
- **Hackathon 讨论线程**：随着细节的敲定，有一个[关于活动的更多讨论](https://discord.com/channels/1089876418936180786/1329529625189154826)线程可供使用。
  
  - 鼓励感兴趣的参与者在 Hackathon 之前分享想法并进行协作。

 

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/) (1 messages):

fund21: 是的，我们如何在 >interpreter --os 模式下集成 Deepspeek？

---

---

---

{% else %}

> 各频道的完整详细分解内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整的分解内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}