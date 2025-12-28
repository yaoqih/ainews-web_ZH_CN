---
companies:
- openai
- anthropic
- xai
- togethercompute
- alibaba
- sakana-ai
date: '2025-02-22T02:50:34Z'
description: '在纽约举行的 **AIE 峰会**（AI Engineer Summit）重点展示了多场关键演讲，包括 **Grace Isford 的趋势主题演讲**、**Neo4j
  与辉瑞（Pfizer）的联合演示**，以及 **OpenAI 对智能体（Agents）的首次定义**。与会演讲者宣布了总计 **9.3 亿美元的融资**。


  在 AI 推特（X）上，讨论热点集中在 **Grok-3** 和 **o3-mini** 模型，并围绕性能和基准测试展开了辩论，其中提到了 Grok-3 创纪录的
  **4e26 至 5e26 FLOP** 计算规模。**o3-mini** 模型在 Sakana AI 的代码中发现了一个关键的 **CUDA 内核错误**。**DeepSeek-R1**
  作为一种具有显著训练批量大小的开源替代方案备受推崇。此外，**阿里巴巴**宣布发布 **Qwen 2.5-VL** 模型。'
id: 2cd5ffe8-2313-4289-b37a-bdf7806008a7
models:
- grok-3
- o3-mini
- deepseek-r1
- qwen-2.5-vl
original_slug: ainews-ai-engineer-summit-day-1
people:
- aidan_mclau
- giffmana
- nrehiew_
- teortaxestex
- epochairesearch
- andrew_n_carr
- borismpower
- yuhu_ai_
title: AI工程师峰会 第一天
topics:
- benchmarking
- model-performance
- cuda
- model-training
- open-source
- debugging
- inference-speed
- batch-size
- reinforcement-learning
---

<!-- buttondown-editor-mode: plaintext -->**AI Engineers are all you need.**

> 2025年2月19日至2月20日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**211** 个频道和 **6423** 条消息）。预计节省阅读时间（以 200wpm 计算）：**647 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

AIE Summit 第一天在纽约圆满结束。

如果非要我们选出 3 个最值得关注的演讲，请查看 [Grace Isford 的趋势主题演讲](https://www.youtube.com/live/L89GzWEILkM?si=ZDW5jhKAD4LVyQVx&t=1033)、[Neo4j/Pfizer](https://www.youtube.com/live/L89GzWEILkM?si=xkUBa6CUDYIZtJfw&t=7632) 的演示，以及 [OpenAI 首次定义 Agents](https://www.youtube.com/live/L89GzWEILkM?si=TC5qcVHcSE1ny1wq&t=11410)。演讲者/赞助商宣布了 [$9.3 亿美元的融资](https://x.com/swyx/status/1892771856484122933)。[多个 Anthropic 数据点](https://x.com/swyx/status/1892684773891375125) 在社交媒体上走红。


![image.png](https://assets.buttondown.email/images/5eb12543-f1b0-46c4-87ff-1047282c222a.png?w=960&fit=max)



您可以在此处观看完整的录播：

https://www.youtube.com/watch?v=L89GzWEILkM

第二天将[侧重于 Agent Engineering](https://www.youtube.com/watch?v=D7BzTxVVMuw)，而第三天将[举行线下工作坊和新的线上环节](https://www.latent.space/p/2025-summit-online)。



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**模型、基准测试与性能**

- **Grok-3 的性能与能力**：[@BorisMPower](https://twitter.com/BorisMPower/status/1892407015038996740) 报告称，**与 Grok 3 相比，o3-mini 在每项评估中都表现更好**，并指出 Grok 3 虽然不错但被过度宣传了。这引发了与来自 xAI 的 [@ibab](https://twitter.com/ibab/status/1892418351084732654) 的讨论，后者回应称他们使用了相同的评估方法。来自 xAI 的 [@Yuhu_ai_](https://twitter.com/Yuhu_ai_/status/1892449337218883868) 为 Grok 3 的性能辩护，声称他们的 **mini 模型在 AIME 2024、GPQA 和 LCB 的 pass@1 指标上超越了 o3-mini high**，并认为基准测试无法完全体现模型智能。[@aidan_mclau](https://twitter.com/aidan_mclau/status/1892416555868201321) 批评 Grok 3 的图表展示为“图表犯罪（chart crimes）”。[@itsclivetime](https://twitter.com/itsclivetime/status/1892463726810583245) 分享了 **Grok 3 的初步正面体验，注意到其在 Deep Research 中的速度**，但也提到了编码速度较慢和偶尔崩溃的问题。[@nrehiew_](https://twitter.com/nrehiew_/status/1892469273446035924) 为 xAI 的评估报告辩护，称其遵循了 OpenAI 的惯例，问题在于清晰度而非欺骗。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1892509598612877503) 对因看好 Grok 而遭受的指责表示惊讶。[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1892671695535677745) 指出 **Grok-3 创纪录的算力规模**，估计为 **4e26 到 5e26 FLOP**，使其成为首个训练算力超过 1e26 FLOP 的已发布模型。

- **o3-mini 性能与 CUDA Kernel 问题**：[@giffmana](https://twitter.com/giffmana/status/1892510741242036468) 强调，**o3-mini 在 11 秒内发现并解决了 Sakana AI 的 CUDA kernels 问题**，揭示了一个使其看起来快 150 倍但实际上慢 3 倍的 bug。[@giffmana](https://twitter.com/giffmana/status/1892510744224182661) 强调了学到的经验：**简单的 CUDA 代码不太可能超越经过优化的 kernels**，**不一致的基准测试预示着问题**，以及 **o3-mini 在调试方面非常高效**。[@main_horse](https://twitter.com/main_horse/status/1892446384910987718) 也进行了基准测试，发现 **Sakana AI 声称的 150 倍加速实际上慢了 3 倍**，并指出了其 CUDA kernel 的问题。

- **DeepSeek R1 的能力与训练**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1892636514221166644) 提到了 **“受 R1 启发的 RL 寒武纪大爆发”**，指出其科学配方与其他顶尖实验室相似，强调了从令人沮丧的“绝望废话”中的转变。[@togethercompute](https://twitter.com/togethercompute/status/1892609242957582505) 推广 **DeepSeek-R1 作为闭源模型的开源替代方案**，在 NVIDIA GPU 上提供快速推理。[@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1892403826717569120) 分享了 **关于 DeepSeek 训练的一个冷知识**，指出 **14 万亿 token 的训练中 batch size 约为 60M token**，与 Llama 1 较小的 batch size 形成对比。

- **Qwen 2.5-VL 模型发布**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1892576737848160538) 发布了 **Qwen2.5-VL 技术报告**，详细介绍了其架构和训练过程，强调了其与 **Qwen2.5-72B 的能力对齐以及行业领先的视觉语义解析能力**。[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1892576743904670022) 还发布了 **3B、7B 和 72B 尺寸的 Qwen2.5-VL AWQ 量化模型**。[@_akhaliq](https://twitter.com/_akhaliq/status/1892433462910501170) 分享了 **Qwen2.5-VL 技术报告的发布**。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1892422884049768473) 也宣布了 **Qwen2.5-VL 技术报告的发布**。[@_philschmid](https://twitter.com/_philschmid/status/1892506190656999925) 详细介绍了 **Qwen Vision Language Models 的训练方式**，强调了动态分辨率处理和重新设计的 Vision Transformer。

- **SmolVLM2 视频模型**：[@mervenoyann](https://twitter.com/mervenoyann/status/1892576290181382153) 发布了 **SmolVLM2，“世界上最小的视频模型”**，包含 256M、500M 和 2.2B 三种尺寸，并提供 **iPhone 应用、VLC 集成以及精彩片段提取器**。[@reach_vb](https://twitter.com/reach_vb/status/1892578169615523909) 重点介绍了 **SmolVLM2，这是采用 Apache 2.0 协议的 VideoLM，参数量从 256M 到 2.2B 不等**，并指出它们可以在免费的 Colab 甚至 iPhone 上运行。[@awnihannun](https://twitter.com/awnihannun/status/1892594913893556707) 推广了 **SmolVLM2 对 MLX 和 MLX Swift 的首日支持**，实现了在 Apple 设备上的本地运行。

- **用于机器人技术的 Helix VLA 模型**：[@adcock_brett](https://twitter.com/adcock_brett/status/1892579315461599658) 发布了 **Helix 的技术报告，这是一款通用视觉-语言-动作 (VLA) 模型**。[@adcock_brett](https://twitter.com/adcock_brett/status/1892579188424712682) 将 **Helix 的架构描述为“System 1, System 2”**，包含一个 7B 参数的 VLM 和一个 80M 参数的视觉运动策略，运行在嵌入式 GPU 上。[@adcock_brett](https://twitter.com/adcock_brett/status/1892579136956186947) 展示了 **Helix 机器人抓取家用物品**，[@adcock_brett](https://twitter.com/adcock_brett/status/1892579000817521092) 详细介绍了 **Helix 以 200Hz 频率协调 35 自由度 (DoF) 的动作空间**。[@adcock_brett](https://twitter.com/adcock_brett/status/1892578885226635525) 展示了 **两台机器人使用 Helix 协作存放杂货**。[@adcock_brett](https://twitter.com/adcock_brett/status/1892578309344502191) 强调了 **Helix 在机器人技术方面的类人思考和泛化能力**。[@adcock_brett](https://twitter.com/adcock_brett/status/1892577936869327233) 将 **Helix 介绍为“像人类一样思考的 AI”**，目标是让机器人走进家庭。

- **SholtoBench AGI 基准测试**：[@nearcyan](https://twitter.com/nearcyan/status/1892469757653442989) 宣布了 **SholtoBench，这是一个追踪 Sholto Douglas (@_sholtodouglas) 在 AGI 实验室就业情况的新 AGI 基准测试**。[@nearcyan](https://twitter.com/nearcyan/status/1892470292758614148) 提供了 **SholtoBench 官方网站**的链接，并感谢了匿名贡献者。

- **AIME 2025 性能图表**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1892487275948224793) 分享了 **AIME 2025 的“Teortaxes 最终版”性能图表**，对比了 o3-mini, Grok-3, DeepSeek-R1 和 Gemini-2 FlashThinking 等模型。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1892482831680504224) 评论了一些实验室发布“愚蠢且变形的图表”来宣称达到 SoTA。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1892471638534303946) 展示了 **AIME 2025 结果的汇总**，旨在追求清晰度，避免“图表造假”。

- **Grok DeepSearch 评估**：[@casper_hansen_](https://twitter.com/casper_hansen_/status/1892531542548684820) 发现 **Grok DeepSearch “相当不错”**，注意到了它的查询扩展功能，并对其与 OpenAI 的 DeepResearch 的对比提出了疑问。

- **LLM Scaling Laws 与数据质量**：[@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1892596586347160059) 讨论了 **LLM Scaling Laws**，认为 **即使互联网数据耗尽，通过提高数据质量，改进仍能继续**，并引用 AlphaGo Zero 的自我博弈作为合成数据推动进步的例子。

- **FlexTok 图像分词器**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1892550422050877486) 重点介绍了 **FlexTok，这是来自 Apple 和 EPFL 的新型分词器 (Tokenizer)**，它将 2D 图像投影为可变长度的 1D Token 序列，从而实现层级化和语义化压缩。

- **视觉语言模型训练**：[@_philschmid](https://twitter.com/_philschmid/status/1892506190656999925) 解释了 **像 @Alibaba_Qwen 2.5-VL 这样的视觉语言模型是如何训练的**，详细说明了预训练阶段（仅 ViT、多模态、长上下文）和后训练阶段（SFT 和 DPO）。

- **vLLM 结合 DeepSeek 模块提速**：[@vllm_project](https://twitter.com/vllm_project/status/1892646680719216960) 宣布 **vLLM v0.7.3 现在支持 DeepSeek 的 Multi-Token Prediction 模块**，实现了高达 **69% 的速度提升**。

**开源与社区**

- **开源 AI 模型**：[@togethercompute](https://twitter.com/togethercompute/status/1892609241212715045) 坚定了他们的信念，即 **“AI 的未来属于开源”**，并围绕开源模型和高性能基础设施构建其云服务公司。[@_akhaliq](https://twitter.com/_akhaliq/status/1892600666276671710) 向 [@bradlightcap](https://twitter.com/bradlightcap) 表示祝贺，并建议 **开源模型可以进一步增强他们的成功**。[@cognitivecompai](https://twitter.com/cognitivecompai/status/1892648693691551839) 表达了对 **@arcee_ai 新发布的 Apache 2.0 协议项目的高度赞赏**。

- **Hugging Face 推理支持扩展**：[@_akhaliq](https://twitter.com/_akhaliq/status/1892628229871030602) 宣布 **Hugging Face Inference 提供商现在支持超过 8 个不同的供应商和近 100 个模型**。

- **LangChain Agent 组件与 Open Deep Research**：[@LangChainAI](https://twitter.com/LangChainAI/status/1892675173226316073) 宣传了 Interrupt 会议，来自 Uber 的演讲者将分享 **基于 LangGraph 的可重用 Agent 组件**。[@LangChainAI](https://twitter.com/LangChainAI/status/1892645710224622024) 推出了 **Open Deep Research**，这是一个可配置的开源深度研究 Agent。[@LangChainAI](https://twitter.com/LangChainAI/status/1892642089529442697) 在炉边谈话中重点介绍了 **Decagon 的 AI Agent 引擎**，该引擎已被 Duolingo 和 Notion 等公司使用。

- **Unsloth 显存高效型 GRPO**：[@danielhanchen](https://twitter.com/danielhanchen/status/1892643424538595611) 宣布 **GRPO（R1 背后的算法）在 @UnslothAI 中实现了高达 90% 的显存节省**，在 54GB 显存上即可实现 20K 上下文长度的 GRPO，而其他训练框架则需要 510GB。

- **Lumina2 LoRA 微调发布**：[@RisingSayak](https://twitter.com/RisingSayak/status/1892462411451412674) 宣布在 Apache 2.0 协议下发布 **Lumina2 LoRA 微调**。

- **Offmute 开源会议总结**：[@_philschmid](https://twitter.com/_philschmid/status/1892599725913768161) 展示了 **Offmute，这是一个使用 Google DeepMind Gemini 2.0 进行会议转录、分析和总结的开源项目**，可生成结构化报告和关键要点。

- **SongGen 开源文本转音乐模型**：[@multimodalart](https://twitter.com/multimodalart/status/1892533897537192366) 宣布 **SongGen 加入 YuE 成为开源文本转音乐模型**，类似于 Suno，允许用户根据语音样本、描述和歌词创作歌曲。

**研究与开发**

- **AI CUDA Engineer - Agentic CUDA 核函数优化**：[@DrJimFan](https://twitter.com/DrJimFan/status/1892404919480832259) 重点介绍了 **Sakana AI 的 “AI CUDA Engineer”，这是一个能够生成优化后的 CUDA 核函数的 Agentic 系统**，利用 AI 来加速 AI。[@omarsar0](https://twitter.com/omarsar0/status/1892621241674301761) 拆解了 **Sakana AI 的 AI CUDA Engineer**，解释了其用于核函数优化的端到端 Agentic 系统。[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1892433535400890734) 宣布了 **“AI CUDA Engineer” Agent 系统，该系统可自动生成 CUDA 核函数，潜在地将模型处理速度提高 10-100 倍**，并发布了一个包含 17,000 多个 CUDA 核函数的数据集。[@omarsar0](https://twitter.com/omarsar0/status/1892621325136810001) 详细介绍了 **AI CUDA Engineer 的 Agentic 流水线**，包括 PyTorch 到 CUDA 的转换和进化优化。[@omarsar0](https://twitter.com/omarsar0/status/1892621450340921345) 提到可以获取由 AI CUDA Engineer 创建的 **17,000 多个经过验证的 CUDA 核函数存档**。

- **Thinking Preference Optimization (TPO)**：[@_akhaliq](https://twitter.com/_akhaliq/status/1892431954085024189) 分享了关于 **Thinking Preference Optimization（思考偏好优化）的研究**链接。

- **用于高效网页爬取的 Craw4LLM**：[@_akhaliq](https://twitter.com/_akhaliq/status/1892435546703638628) 发布了关于 **Craw4LLM 的消息，这是一种用于 LLM 预训练的高效网页爬取技术**。

- **通过基于 3DGS 的强化学习实现驾驶策略的 RAD**：[@_akhaliq](https://twitter.com/_akhaliq/status/1892435007412621429) 分享了 **RAD 研究，该研究使用大规模基于 3DGS 的 Reinforcement Learning 训练端到端驾驶策略**。

- **Autellix - LLM Agent 的高效推理引擎**：[@_akhaliq](https://twitter.com/_akhaliq/status/1892434670597345474) 重点介绍了 **Autellix，这是一个将 LLM Agent 作为通用程序运行的高效推理引擎**。

- **用于 3D 分子生成的 NExT-Mol**：[@_akhaliq](https://twitter.com/_akhaliq/status/1892438110480302474) 分享了 **NExT-Mol 研究，该研究探讨了 3D Diffusion 结合 1D Language Modeling 用于 3D 分子生成**。

- **小模型向强推理者学习**：[@_akhaliq](https://twitter.com/_akhaliq/status/1892435858684248326) 链接了一项关于**小模型难以从强推理者（Strong Reasoners）中学习**的研究。

- **用于复杂推理的 NaturalReasoning 数据集**：[@maximelabonne](https://twitter.com/maximelabonne/status/1892539204875227642) 介绍了 **NaturalReasoning，这是一个旨在无需人工标注即可提高 LLM 复杂推理能力的新指令数据集**，强调质量优于数量以及训练数据的多样性。

- **用于目标检测的细粒度分布细化（Fine-grained Distribution Refinement）**：[@skalskip92](https://twitter.com/skalskip92/status/1892497124534747193) 介绍了 **D-FINE，一种“新型” SOTA 目标检测器**，它使用细粒度分布细化技术，通过迭代边缘偏移调整和在网络层间共享精确分布来提高边界框（bounding box）的准确性。

- **用于生物分子平衡结构预测的 BioEmu**：[@reach_vb](https://twitter.com/reach_vb/status/1892656772759916860) 重点介绍了 **Microsoft 的 BioEmu，这是一个用于高效预测生物分子平衡结构系综的大规模深度学习模型**，每小时能够采样数千个结构。

**机器人与具身智能（Robotics and Embodiment）**

- **Figure 的 Helix 人形机器人 AI**：Figure AI 正在开发 **Helix**，这是一款用于人形机器人的 AI 模型，展示了包括杂货存储和物体操作在内的多种能力（来自 [@adcock_brett](https://twitter.com/adcock_brett) 的推文）。他们正在为 **Helix、训练基础设施（Training Infra）、大规模训练（Large Scale Training）、操作工程师（Manipulation Engineer）、大规模模型评估（Large Scale Model Evals）以及强化学习（Reinforcement Learning）** 扩展其 AI 团队 ([@adcock_brett](https://twitter.com/adcock_brett/status/1892579357182345588))。他们的目标是在 **2025** 年实现量产并交付更多机器人，重点关注家庭机器人 ([@adcock_brett](https://twitter.com/adcock_brett/status/1892579860289130520))。

- **机器人上的 7B LLM 对比用于数学的 o3**：[@abacaj](https://twitter.com/abacaj/status/1892622993148313747) 表示 **“在机器人上运行 7B LLM 比使用 o3 解决博士级数学问题更有趣”**。[@abacaj](https://twitter.com/abacaj/status/1892611093802910152) 发现一个 **7B 参数的板载视觉 LLM 驱动机器人“很有趣且在预料之中”**，并指出了模型能力的提升。[@abacaj](https://twitter.com/abacaj/status/1892623488889831520) 幽默地建议 **“7B LLM 会帮你洗碗，而 o3 不会”**。

- **Skyfire AI 无人机营救警员**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1892628887856939236) 分享了一个关于 **Skyfire AI 无人机营救一名警员生命**的故事，该无人机在交通执法期间定位到了处于困境的警员，从而实现了快速增援和干预。

**工具与应用**

- **Glass 4.0 AI 临床决策支持平台**：[@GlassHealthHQ](https://twitter.com/GlassHealthHQ/status/1892574802327523360) 推出了 **Glass 4.0，这是他们更新后的 AI 临床决策支持平台**，具有连续对话、高级推理、扩展的医学文献覆盖范围以及更快的响应速度。

- **AI-Toolkit UI**：[@ostrisai](https://twitter.com/ostrisai/status/1892424544356294978) 分享了 **AI-Toolkit UI** 的进展，指出“困难的部分已经完成”，目前正在进行 UI 清理，随后将添加“有趣的功能”。

- **用于 AI 应用构建的 Gradio Sketch**：[@_akhaliq](https://twitter.com/_akhaliq/status/1892604706377052357) 重点介绍了一种**使用 “gradio sketch” 构建 AI 应用的新方法**，该方法支持通过视觉组件选择和配置来生成 Python 代码。

- **Gemini App 深度研究（Deep Research）**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1892629054311772463) 宣布 **Deep Research 已在 150 个国家/地区以 45 种以上语言向 Gemini Advanced 用户开放**，其功能类似于个人 AI 研究助手。

- **Elicit 系统综述**：[@elicitorg](https://twitter.com/elicitorg/status/1892592908563534221) 推出了 **Elicit Systematic Reviews，支持研究综述的自动搜索、筛选和数据提取**，旨在在用户控制下加速研究进程。

- **搭载 Qwen 2.5 模型的 PocketPal 移动应用**：[Qwen 2.5 模型（包括 1.5B (Q8) 和 3B (Q5_0) 版本）已添加到](https://twitter.com/ANOTHER_HANDLE/status/SOME_ID) 适用于 iOS 和 Android 平台的 PocketPal 移动应用中。用户可以通过该项目的 GitHub 仓库提供反馈或报告问题，开发者承诺会在时间允许的情况下解决这些问题。该应用支持多种聊天模板（ChatML, Llama, Gemma）和模型，用户对比了 Qwen 2.5 3B (Q5)、Gemma 2 2B (Q6) 和 Danube 3 的性能。开发者提供了 [截图](https://preview.redd.it/130oisgjvspd1.png?width=1290&format=png&auto=webp&s=9890aa96eec037b33f6849e)。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Qwen2.5-VL-Instruct 在视觉和视频任务中表现出色**

- **Qwen/Qwen2.5-VL-3B/7B/72B-Instruct 发布了！！** ([Score: 489, Comments: 75](https://reddit.com/r/LocalLLaMA/comments/1itq30t/qwenqwen25vl3b7b72binstruct_are_out/)): **Qwen2.5-VL** 带来了显著增强，包括改进的 **visual understanding**（用于识别图像中的物体、文本、图表和布局），以及使其能够进行推理并与计算机和手机等工具交互的 **agentic capabilities**。它还具有针对一小时以上视频的 **long video comprehension**、具有精确物体识别和定位能力的 **visual localization**，以及针对发票和表格等复杂数据的 **structured output generation**，使其在金融和商业领域具有高度适用性。模型链接可在 [Hugging Face](https://huggingface.co/Qwen) 上找到。
  - 用户注意到了 **Qwen2.5-VL** 及其 **AWQ versions** 的发布，对其发布时机有些困惑。**Recoil42** 强调了其 **long video comprehension** 功能在视频行业的潜在影响，而其他人则讨论了处理长视频（特别是使用 **70B model** 时）所需的巨大 **VRAM requirements**。
  - 分享了不同模型大小和量化版本的 **Benchmark results**，包括 **MMMU_VAL**、**DocVQA_VAL** 和 **MathVista_MINI** 等性能指标，展示了 **BF16** 和 **AWQ** 量化之间的差异。对比了 **3B, 7B, and 72B models**，通常 **AWQ** 的表现略低于 **BF16**。
  - 用户讨论了 **compatibility and support** 问题，包括 **ollama** 或 **llama.cpp** 是否支持该模型，并分享了在不同平台上运行模型的解决方案，如 **Mac 上的 MLX** 和 **Nvidia/Linux 上的 TabbyAPI**。还讨论了 **exl2 format** 及其与较新 Nvidia 硬件的兼容性。


**Theme 2. Reverb-7b 在 Open LLM Leaderboards 中表现优异**

- **New AI Model | Ozone AI** ([Score: 164, Comments: 54](https://reddit.com/r/LocalLLaMA/comments/1itr9th/new_ai_model_ozone_ai/)): 来自 **Ozone AI** 的最新 AI 模型 **Reverb-7b** 已发布，展示了 7B 模型性能的显著提升。该模型在来自 **Claude 3.5 Sonnet** 和 **GPT-4o** 的超过 **2 亿个 tokens** 上进行了训练，并基于 **Qwen 2.5 7b** 进行了微调。**Reverb-7b** 在 **Open LLM Leaderboard** 上超越了其他 7B 模型，尤其在 **MMLU Pro** 数据集上表现出色，各学科平均准确率达到 **0.4006**。更多详情和模型可在 [Hugging Face](https://huggingface.co/ozone-ai/Reverb-7b) 找到，后续模型包括目前正在训练的 14B 版本。
  - **性能担忧：** 用户对 **Reverb-7b** 的创意写作能力表示担忧，指出尽管其 **MMLU Pro** 分数很高，但在该领域表现不佳，这表明其侧重于 STEM 学科而非多样化的词汇知识。
  - **模型差异化：** 该模型是 **Qwen 2.5 7b** 的微调版本，在智能和创意写作方面比之前的版本有所改进，正如用户将其与 **llama 3.1 8B** 等模型对比时所指出的。
  - **数据集与发布：** 由于盈利动机，数据集目前保持封闭，但未来有开放计划。**Reverb-7b** 的 **GGUF** 版本已在 **Hugging Face** 发布，用户已将其转换为 **mlx** 格式以获得更广泛的可用性。


**Theme 3. SmolVLM2：优化视频任务的紧凑型模型**

- **SmolVLM2: New open-source video models running on your toaster** ([Score: 104, Comments: 15](https://reddit.com/r/LocalLLaMA/comments/1iu2sdk/smolvlm2_new_opensource_video_models_running_on/)): **Hugging Face 的 Merve** 发布了 **SmolVLM2**，提供了 **256M, 500M, and 2.2B** 尺寸的新型开源视觉语言模型。此次发布包括对 **transformers and MLX** 的零日支持、一个使用 500M 模型的 iPhone 应用、使用 2.2B 模型进行描述分割的 VLC 集成，以及同样基于 2.2B 模型的视频高光提取器。更多详情可以在他们的 [blog](https://reddit.com/link/1iu2sdk/video/fzmniv61obke1/player) 中找到。
  - **Zero-shot vision** 被解释为视觉模型在没有针对特定任务进行直接训练的情况下，利用通用知识执行任务的能力。给出的例子是在测试时为指定的新标签分类图像。
  - 用户对 **Hugging Face** 在小型模型上的工作表示赞赏，指出 **SmolVLM2** 尽管体积紧凑，但性能令人印象深刻。该模型在各种应用中的集成和实用性被视为重大成就。
  - **Merve** 提供了 **SmolVLM2** 的 [blog](https://huggingface.co/blog/smolvlm2) 链接以及 [checkpoints and demos 集合](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7)，方便进一步探索和使用该模型。

**主题 4. 开源 AI Agent 挑战新前沿**

- **[使用 Canva 的 Agent。事情变得疯狂了……](https://v.redd.it/hjbttwq4r9ke1)** ([Score: 125, Comments: 47](https://reddit.com/r/LocalLLaMA/comments/1itv9ia/agent_using_canva_things_are_getting_wild_now/)): 该帖子讨论了一个使用 **Canva** 并可能绕过 **CAPTCHA** 的 **AI Agent**，展示了在自动化通常需要人类交互的任务方面的先进能力。帖子正文的缺失表明需要依靠随附的视频来获取更多背景信息。
  - 帖子中展示的 **AI Agent** 具有绕过 **CAPTCHA** 的能力，尽管人们对这类演示的真实性仍持怀疑态度，建议通过亲自使用来验证。该项目已开源，可在 [GitHub](https://github.com/Aident-AI/open-cuak) 上获取。
  - 人们对该 **Agent** 与 **OpenAI** 之外的其他 **multimodal models** 的兼容性很感兴趣，并确认它可以与其他开源模型配合使用，尽管性能可能会有所不同。运行成本可以通过租用 **GPU** 来控制，价格约为每小时 **1.5 美元**。
  - 将 **Canva** 与 AI 配合使用的设置需要详细的指令，这表明是一个反复试验的过程。有人提出了对 **Agent** 适应界面变化的担忧，强调了在 **Prompt** 或知识库中需要精确控制细节的必要性。


## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. 多模态 AI 系统：连接文本与视觉**

- **[“事实上……再想一想” 这种 AI](https://i.redd.it/ekotu6u4z7ke1.jpeg)** ([Score: 103, Comments: 44](https://reddit.com/r/ChatGPT/comments/1itpt8c/actually_on_a_second_thought_ahh_ai/)): 该帖子讨论了 **AI 对数值数据理解**中的一个常见错误，特别是 AI 如何误解小数。给出的例子展示了 **9.11 和 9.9** 之间的比较，说明 **9.9 更大**，因为 **0.90 大于 0.11**，强调了正确解析小数部分的重要性。
  - **类人困惑**：讨论强调了 AI 在解释数字时的最初困惑与人类乍看之下可能产生的误解类似，但人类可以迅速分析并纠正自己的理解。
  - **AI 的自我纠正**：用户注意到像 **ChatGPT** 这样的 AI 在回答中途承认错误的案例，这与人类意识到错误时的行为相似。
  - **误解中的幽默**：评论幽默地将数值误解与其他语境（如物理尺寸或日期）进行比较，并调侃 AI 像人类一样掩盖错误的倾向。


---

# AI Discord 摘要

> 由 o1-preview-2024-09-12 生成的摘要之摘要的摘要

**主题 1. Grok 3 从 OpenAI 手中夺走焦点**

- [**Grok 3 碾压 ChatGPT 无法处理的编程任务**](https://grok.com)：用户报告称 **Grok 3** 解决了 **ChatGPT Pro** 难以应对的复杂编程问题，促使许多人考虑转向 **SuperGrok**。
- [**SuperGrok 以极低价格提供高级 AI 服务**](https://grok.com/?show_subscribe=1)：每月 **$30** 的 **SuperGrok** 被认为比每月 **$250** 的 **ChatGPT Pro** 订阅更具性价比，导致用户重新评估他们的 AI 服务选择。
- [**Grok 3 成为社区的新“死党”**](https://x.com/Yuchenj_UW/status/1892634804786757712)：热情的用户因其性能、速度和用户友好的界面而称 **Grok 3** 为他们的“*bestie*”，许多人称赞其无限的 API 和即将推出的功能。

**主题 2. Unsloth 的 GRPO 算法大幅降低 VRAM 需求**

- [**仅需 5GB VRAM 即可训练 GRPO 模型——无需魔法！**](https://unsloth.ai/blog/grpo)：**Unsloth** 发布了新算法，可实现 **10 倍长的上下文长度**和 **90% 的 VRAM 节省**，允许在不损失精度的情况下仅用 **5GB VRAM** 进行训练。
- [**社区欢呼 Unsloth 的 VRAM 节省突破**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)：用户表达了兴奋和感谢，并在其项目中使用 [Unsloth 的 Google Colab notebooks](https://colab.research.google.com/github/unslothai/notebooks/) 分享了改进成果。
- [**Llama 3.1 训练减少了 90% 的 VRAM 需求**](https://x.com/UnslothAI/status/1892640995847901684)：受 **Horace He** 的梯度检查点（gradient checkpointing）技术启发，**Unsloth** 的 GRPO 算法将 **Llama 3.1** 的 VRAM 需求从 **510.8GB** 降低到 **54.3GB**。

**主题 3. AI CUDA Engineer 夸张的提速声明引发质疑**

- [**“AI CUDA Engineer”声称提速 100 倍，工程师们表示抗议**](http://sakana.ai/ai-cuda-engineer/)：**Sakana AI** 推出了一款 AI 系统，号称在 CUDA kernel 优化方面实现了 **10-100 倍的提速**，但怀疑者指出其基准测试存在缺陷且存在根本性 bug。
- [**“NOP Kernels”赢得了比赛——但什么也没做！**](https://x.com/main_horse/status/1892473238036631908)：成员们发现某些 kernel 通过实际上不执行任何操作来实现提速，凸显了“奖励作弊”（*reward hacking*）的案例，并对系统的有效性提出质疑。
- [**过度炒作的 AI Kernel 遭到社区吐槽**](https://x.com/BingXu_/status/1892405811596710392)：专家们拆穿了令人印象深刻的提速谎言，揭露了内存重用和错误评估等问题；AI 尚未准备好取代人类 CUDA 工程师。

**主题 4. 微软凭借 Majorana 1 实现量子飞跃，但遭遇怀疑**

- [**微软凭借 Majorana 1 芯片承诺百万量子比特的未来**](https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/)：微软推出了全球首款由拓扑量子比特（topological qubits）驱动的量子处理器，旨在实现 **100 万个量子比特**的可扩展性。
- [**拓扑量子比特详解——真的是这样吗？**](https://www.youtube.com/watch?v=wSHmygPQukQ)：在一段 [YouTube 视频](https://www.youtube.com/watch?v=wSHmygPQukQ)中，微软团队讨论了拓扑量子比特，但一些人对其需要“氦致冷机”的实际应用仍持怀疑态度。
- [**Nadella 大肆宣传量子技术，用户却在抱怨 Teams**](https://youtu.be/4GLSzuYXh6w)：尽管 **Satya Nadella** 宣传微软的量子突破，用户却对 **Teams** 和 **Copilot** 等现有产品表示不满，质疑微软对创新而非产品质量的关注。

**主题 5. AI 公司狂揽巨资，押注推理爆发**

- [**Lambda 获 4.8 亿美元融资助力 AI 云**](https://x.com/stephenbalaban/status/1892275552171737220)：**Lambda** 宣布完成 **4.8 亿美元 D 轮融资**，以增强其 AI 计算资源，旨在成为**专为 AI 定制的云服务**首选。
- [**Arize AI 融资 7000 万美元以完善 AI 评估**](https://arize.com/blog/arize-ai-raises-70m-series-c-to-build-the-gold-standard-for-ai-evaluation-observability/)：**Arize AI** 获得资金以推进 AI 评估和可观测性，确保 **AI Agent** 大规模可靠运行。
- [**Baseten 和 Together Compute 豪赌 2025 年推理热潮**](https://x.com/basetenco/status/1892259130540179863)：**Baseten** 融资 **7500 万美元**，**Together Compute** 融资 **3.05 亿美元**，双方都在为他们认为的 **AI 推理技术**关键年做准备。

---

# 第一部分：高层级 Discord 摘要

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Grok 3 表现优于 OpenAI 模型**：**Grok 3** 在基准测试和解决 **ChatGPT Pro** 难以处理的代码任务方面表现出更优越的性能。
   - 用户对 **Grok 3** 的能力表现出更强的信心，报告称它解决了 **o1 Pro** 无法解决的复杂问题，并考虑转向 **SuperGrok**。
- **SuperGrok 提供更好的订阅价值**：每月 30 美元的 **SuperGrok** 被认为比 **ChatGPT Pro** 每月 250 美元的订阅更具性价比。
   - 用户认为 **SuperGrok** 在性能和使用限制方面具有优势，导致许多人重新评估其 AI 服务订阅。
- **Grok 语音模式的期待**：社区成员期待 **Grok** 即将推出的功能，如语音模式和自定义指令，认为这些功能将进一步增强其效用和竞争力。
   - **Grok 3** 模型的 **API** 因其无限制的能力而受到关注，允许进行广泛的交互，而没有某些其他模型中常见的严格限制。他们正在积极寻求更多集成。
- **提议保存聊天 URL 以返回有价值的讨论**：一位成员提议保存**聊天的 URL**，以便轻松返回有价值的讨论，并鼓励他人在指定频道分享想法，以便 **OpenAI** 看到。
   - 他们还建议使用 *'good1'* 或 *'Track this chat'* 等关键词来帮助记住重要的聊天记录。
- **预期的提示词工程故障排除**：一位成员表达了对通话的渴望，以确定问题是由于 **prompt** 还是软件故障引起的，这比预期花费了更多时间。
   - 该成员感谢他人的有益建议，表示将记住这些见解以备将来参考，但在特定情况下需要“其他东西”。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **DeepSeek-V3 授予无限访问权限！**：**DeepSeek-V3** 现在对 **Windsurf Pro** 和 **Ultimate** 计划用户无限开放，提供 **0 prompt credits** 和 **0 flow action credits** 的不受限访问。
   - **Windsurf** 鼓励用户查看[此推文](https://x.com/windsurf_ai/status/1892322088507105561)以了解有关此更改的更多信息。
- **MCP 用例引发关注**：Matt Li 分享了 **MCP** 内容，鼓励用户在 [X](https://x.com/windsurf_ai/status/1892394489588727985) 上探索其潜力，突显了社区对参与的渴望。
   - 一个快速演示展示了 **MCP** 如何在 **Cascade** 中工作，为仍在探索其功能的人提供资源。
- **Codeium 插件面临 EOL 猜测**：用户对 **JetBrains Codeium plugin** 可能不再受支持表示担忧，对其缺乏方向感感到沮丧。
   - 一位用户感叹道：*看到 Codeium 作为一个插件被放弃真是太遗憾了。*
- **Cascade 的记忆系统需要改进**：鼓励用户使用 '*add to memory*' 和 '*update memory*' 等命令来帮助 **Cascade** 记住项目细节，而将全局规则组织到单独文件中的提议结构旨在提高 **Cascade** 的性能。
   - 关于 **DeepSeek-V3** 与 **Cascade Base** 优势的讨论。
- **Windsurf 用户等待支持**：用户报告在收到支持工单回复方面存在延迟，包括主题行中缺少带有预期工单编号的自动回复。
   - 关于支持通信的正确电子邮件来源仍存在困惑。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 发布 Long Context GRPO**：Unsloth 发布了 **Long Context GRPO**，仅需 **5GB VRAM** 即可训练推理模型，承诺实现 **10x** 的上下文长度提升和 **90%** 的 VRAM 占用减少，详见[此推文](https://x.com/UnslothAI/status/1892640995847901684)。
   - 用户表达了兴奋之情并分享了他们的改进成果，同时感谢 Unsloth 提供免费资源，例如[这个 Google Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)。
- **训练损失波动引发关注**：用户观察到模型训练期间 **training loss** 存在显著波动，通常在多个 epoch 后才会稳定，用户正使用[这个 Google Colab](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing) 进行调整。
   - 社区建议调整 **learning rate** 并保持训练提示词（prompts）的清晰度，以减少 **overfitting** 并增强学习效果，这在 [Unsloth 文档](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl)中也有提及。
- **5090 Mobile 规格激发升级幻想**：**RTX 5090 Mobile** 将配备 **24GB** 显存，预计下周开始预订。
   - 该公告引起了正积极考虑硬件升级的社区成员的兴趣。
- **RAG 与 Fine-tuning 的细微差别揭晓**：分享了一段名为 [“RAG vs. Fine Tuning (Live demo)” 的 YouTube 视频](https://www.youtube.com/watch?v=LDMFL3bjpho)，探讨了 **fine tuning** 是否比传统的 **RAG** 系统产生更好的结果。
   - 观众要求提供更多比较 **RAG** 和 **fine tuning** 的案例，暗示了对未来演示中更全面见解的需求；创作者表示计划制作后续视频，详细介绍如何开始使用 **Kolo**。
- **Triton 的自定义汇编效果显著**：对挑战评分系统中的 **custom_asm_works** 含义进行了澄清，解释其涉及 Triton 中的 **inline assembly**，允许在没有 **CUDA** 的情况下对张量执行，详见 [Triton 文档](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html)。
   - 这被用作一种改进硬件内聚计时问题的技术，也是当前工作的重点。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **混元图像生成对 VRAM 要求较高**：用于图像生成的 **Hunyuan** 模型现已可用，但至少需要 **24GB VRAM**，且主要在 **NVIDIA** 显卡上运行，生成视频内容需要几分钟时间。
   - 用户热衷于测试 **Hunyuan** 与其他平台的能力对比。
- **用于 AI 任务的 A100 GPU**：用户讨论了在 **LM Studio** 中使用 **A100 GPU** 的效用，强调了其 **80GB VRAM** 容量对 AI 任务的支持。
   - 尽管成本可能很高，但人们对获取 **A100** 以提升性能表现出浓厚兴趣。
- **AMD Ryzen AI Max+ CPU 媲美 RTX 4090**：**Ryzen AI Max+** 的规格引起了关注，有[文章](https://www.club386.com/amd-ryzen-ai-max-cpus-beat-nvidia-rtx-4090-at-llm-performance/)称其在 LLM 工作负载上击败了 **Nvidia RTX 4090**。
   - 在独立基准测试出炉前，人们对其与现有 GPU 相比的实际性能仍持怀疑态度。
- **Apple Silicon 因焊接组件受到批评**：围绕 **Apple** 在笔记本电脑中焊接组件的讨论，这限制了可维修性和升级性。讨论还涉及对集成设计趋势限制内存配置灵活性的担忧。
   - 用户表达了对允许硬件升级的系统的偏好。
- **推测解码深度探讨**：根据用户反馈，某些模型的 **Speculative decoding**（推测解码）可能会导致较低的 token 接受率和较慢的性能。
   - 用户分享了关于 token 接受率的经验，并询问了旨在最大化性能的最佳模型设置。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grok 3 占据领先地位**：用户发现 **Grok 3** 的表现比 **GPT-4o** 更快，一些人为此取消了其他订阅，称 Grok 3 为他们的“最佳拍档”，因为它性能出色、价格更便宜且 UI 友好，参考 [此 X 帖子](https://x.com/Yuchenj_UW/status/1892634804786757712)。
   - 值得注意的是，根据 [xAI 的推文](https://x.com/xai/status/1892400129719611567)，**Grok 3** 目前免费提供（直到他们的服务器宕机），并为 Premium+ 和 SuperGrok 用户增加了访问权限。
- **Aider 面临 Linux 参数大小限制**：一位用户报告称，由于 Linux 参数大小限制，特别是深层嵌套的目录路径，很难向 **Aider** 传递大量文件。
   - 他们建议使用带有 `/load` 命令的文本文件作为变通方案，同时指出虽然仓库包含许多小文件，但嵌套目录路径的长度是一个主要问题。
- **SambaNova 夺得 DeepSeek-R1 效率桂冠**：**SambaNova** 宣布，与现有模型相比，其提供的 **DeepSeek-R1** 服务在速度和成本上都有显著降低，达到了每秒 *198 tokens*，详见 [其新闻稿](https://sambanova.ai/press/fastest-deepseek-r1-671b-with-highest-efficiency)。
   - 根据一篇 [Kotlin 博客文章](https://blog.jetbrains.com/kotlin/2025/02/openai-vs-deepseek-which-ai-understands-kotlin-better/)，这一声明将 **DeepSeek-R1** 定位为高效模型，在 AI 模型应用和实现方面取得了重大进展。
- **Aider 字体颜色引发可见性争议**：用户对 **Aider** 中的字体颜色可见性表示担忧，尤其是浅色模式下的蓝色。
   - 建议包括检查深色模式设置并确保正确配置以解决可见性问题。
- **RAG 配置优于 AI Chat**：一位成员表示，目前的 **RAG** 配置在编码需求方面比 **AI Chat** 的 RAG 功能效果更好。
   - 另一位成员表示赞同，指出普通的 **RAG** 在处理代码时表现不佳，需要改进。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE 引发争议**：用户报告了 **Cursor 的 Sonnet 3.5 性能**问题，对与之前版本相比的可靠性表示沮丧。
   - 相比之下，**Grok 3** 在编码任务中的速度和解决问题的有效性受到了称赞，尽管一些人批评其所有者和过去的表现，以及缺乏 API 访问权限；参见 [Grok 3 是一个...有趣的模型](https://youtu.be/WVpaBTqm-Zo)。
- **MCP 服务器令人头疼**：用户讨论了在 Cursor 中设置和运行 **MCP 服务器** 的复杂性，一些人发现很难有效利用它；查看 [Perplexity Chat MCP Server | Smithery](https://smithery.ai/server/@daniel-lxs/mcp-perplexity)。
   - 社区成员建议，改进文档可以提升用户体验并简化安装，并指出 *MCP 配置是针对 OSX 和 Linux 特有的*，参见 [issue #9 · anaisbetts/mcp-installer](https://github.com/anaisbetts/mcp-installer/issues/9)。
- **AI 模型性能受到质疑**：参与者对当前 **AI 模型**（尤其是 Claude）的性能表示不满，将输出的不一致归因于底层的 Prompting 和上下文管理问题。
   - LLM 的响应变化是预料之中的，这突显了这些模型的随机性，但一些人希望 Grok-3 以及 Windsurf Pro 和 Ultimate 计划中提供的新 **DeepSeek-V3** 能有更好的表现，参见 [Windsurf (@windsurf_ai) 的推文](https://x.com/windsurf_ai/status/1892322088507105561?s=46&t=ggmESCIXF0nYw8_kshHz7A)。
- **开发者工具引发挫败感**：用户报告了使用 **Cursor Tab** 的挑战，一些人表示它在开发过程中引入了 Bug，减慢了工作流程。
   - **Cursor Composer** 因生成更强大、更可靠的代码而受到称赞，但总体而言，开发者们正期待着由 **Rainier AI 计算集群** 提供支持的下一代 Amazon 和 Anthropic 模型，参见 [Amazon 宣布与 Anthropic 合作推出新的“Rainier” AI 计算集群](https://www.semafor.com/article/12/03/2024/amazon-announces-new-rainier-ai-compute-cluster-with-anthropic)。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face 精装书上架**：围绕新发布的 Hugging Face 主题精装书，大家反响热烈，这标志着在[最近的博客文章](https://x.com/Nouamanetazi/status/1892274582503248178)中庆祝的一年团队合作成果。
   - 感兴趣的人应该尽快行动以*确保获得一本*。
- **Qwen2.5 取得训练突破**：利用 Unsloth 的新算法，用户现在只需 **5GB VRAM** 即可训练 Qwen2.5 推理模型，实现 **10倍长的上下文长度**和 **90% 的 VRAM 节省**，详情见[此博客](https://unsloth.ai/blog/grpo)。
   - 这些改进为开发者提供了实用的工具。
- **HF Spaces 托管快速视频生成器**：讨论强调了 HF Spaces 上视频生成器的可用性，其中 *ltxv* 因其速度脱颖而出，仅需 **10-15 秒**即可生成视频。
   - 有一个新的协作计划，旨在基于最新发布版本创建一个视频生成器。
- **CommentRescueAI 加速 Python 文档生成**：**CommentRescueAI** 是一款只需点击一下即可为 Python 代码添加 AI 生成的 docstrings 和注释的工具，现已在 VS Code 扩展市场上线。
   - 开发者正在寻求社区关于改进想法的建议。
- **Lumina2 使用 LoRA 进行微调**：一个使用 **LoRA** 的 **Lumina2** 新微调脚本现已发布，在 **Apache2.0** 许可证下增强了用户能力，更多信息请参阅[文档](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_lumina2.md)。
   - 这促进了 AI 技术上的开放协作。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 用户遭遇故障**：用户报告了使用 **Perplexity AI** App 的挫败体验，提到在文本生成过程中存在延迟、高资源消耗和故障，但开发者可能[正在处理](https://www.cplx.app/)。
   - 特别是针对模型的性能提出了担忧，引发了关于开发团队是否正在积极解决这些持续问题的询问。
- **用户称 Grok 3 幻觉严重**：围绕 **Grok 3** 的讨论显示出复杂的情绪；一些用户认为它比之前的模型表现更好，而另一些用户则注意到明显的幻觉行为。
   - 用户将 **Grok 3** 与 **Claude** 和 **O3** 的组合进行了比较，通常更倾向于 **Claude** 以获得更可靠的性能。
- **墨西哥与 Google 在海湾地区的对峙**：墨西哥采取大胆行动，就其在海湾附近的运营对 **Google** 发出威胁，凸显了持续的管辖权争议。
   - 这一冲突凸显了科技公司与国家监管机构之间在机器学习使用方面日益增长的紧张关系。
- **Sonar API 的困境引发担忧**：一位用户对 **Sonar API 的性能**表示担忧，发现其结果比 **llama-3.1-sonar-large-128k-online** 等旧模型更差。
   - 该用户报告称，旧模型在获取网站信息等任务中表现更好，对尽管价格相似但质量下降的情况表示失望。
- **传言 Deep Research API 即将推出**：成员们正在询问将 **deep research 能力**集成到 API 中的可能性，这可能会带来令人兴奋的新功能。
   - 一位用户表达了热情，感谢 Perplexity 团队在这一领域的持续工作。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **沙特阿拉伯发布 ALLaM**：由**沙特阿拉伯**支持的 [ALLaM](https://arxiv.org/html/2407.15390v1) 专注于创建阿拉伯语语言模型，以支持阿拉伯语技术生态系统，这代表了在当前地缘政治环境下对 LLM 的推动。
   - 该模型可以生成阿拉伯语和英语文本，拥有 70B 参数。
- **Mercor 为 AI 招聘融资 1 亿美元**：[Mercor](https://techcrunch.com/2025/02/20/mercor-an-ai-recruiting-startup-founded-by-21-year-olds-raises-100m-at-2b-valuation/) 为其 AI 招聘平台融资 1 亿美元，该公司由年轻的 Thiel Fellows 创立，此次融资凸显了其快速增长，估值跃升至 20 亿美元。
   - 讨论集中在 Mercor 在竞争激烈的 AI 领域中创新的营销驱动力。
- **创新的 GRPO 算法降低 VRAM 需求**：Unsloth 发布了一种新的 **GRPO 算法**，将 Qwen2.5 训练的 VRAM 需求降低到仅 **5GB**，标志着重大改进。
   - 该算法支持 **10 倍长的上下文长度**，提供了简化的设置，可能彻底改变模型训练效率。
- **Nadella 宣传微软，但产品质量存疑**：在最近的一段 [YouTube 视频](https://youtu.be/4GLSzuYXh6w)中，**Satya Nadella** 在宣传经济增长和微软的**拓扑量子比特（topological qubit）突破**的同时，分享了他对 AGI 的怀疑。
   - 成员们表达了沮丧，质疑当 Teams 和 Copilot 等 **Microsoft 产品**表现不佳时，**Satya Nadella** 为何能被正面看待。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **推理 Token 引发不满**：用户对 OpenRouter 实现中较低的 **max_tokens** 表示不满，这导致当 **include_reasoning** 默认为 false 时出现空响应或 null 响应。
   - 提议的更改包括将 **include_reasoning** 默认设置为 true，并确保内容始终为字符串，避免 null 值以提高响应一致性，目前正通过投票收集社区意见。
- **Weaver 扩展提供多功能选项**：**Weaver** Chrome 扩展提供了高度可配置的选项，如 PDF 支持、与 **Supabase** 的云同步以及从浏览器直接进行 API 调用。
   - 虽然目前免费且托管在 Vercel 的免费计划上，但由于使用限制，它可能面临访问限制，且没有后端数据日志记录。
- **API 翻译器转为开源**：一位用户分享了一个新开发的**开源 Chrome 扩展**，可通过 [GitHub](https://github.com/amirrezasalimi/aify) 获取，允许用户将任何内容转换为他们喜欢的风格。
   - 该工具仅需要一个**兼容 OpenAI 的 API** 即可运行。
- **Gemini 输出故障引发抱怨**：用户报告了 **Gemini 2.0 Flash** 模型的结构化输出问题，指出在与 OpenRouter 集成时与 OpenAI 的模型存在差异。
   - 反馈表明需要更清晰的 UI 指示模型能力，特别是关于输入类型和错误消息。
- **DeepSeek 的性能下降令人担忧**：一些用户报告称，**DeepSeek** 模型最初产生高质量响应，但随后在 OpenRouter 内部的响应质量显著下降。
   - 讨论涉及了响应质量下降的可能原因和缓解策略。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Grok3 基准测试受到质疑**：关于 **Grok3** 的性能和基准测试出现了疑问，成员们指称 xAI 可能在 cons@64 的使用数据上存在隐瞒。
   - 质疑者对 **Grok3** 超越现有最先进模型的说法提出挑战，并分享了具体的反例。
- **用于神经网络优化的 EAs？**：社区讨论了使用 **evolutionary algorithms** (进化算法) 来优化 **neural networks**，考虑到在高维情况下大规模扩展时收敛速度较慢的问题。
   - 成员们讨论了在特定的训练流水线组件中使用 GAs 以提高模型性能，并将其与传统的 backpropagation 进行了对比。
- **代码数据集分享**：成员们在 **Hugging Face** 上分享了代码数据集，建议将其用于增强现有模型。
   - 对话强调了数据集质量的重要性，以及使用先进的推理模型（如 [NovaSky-AI/Sky-T1_data_17k](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k)）重新加工现有数据集的可能性。
- **Agents 协作进行优化**：一位成员询问了关于 **agents 协作**以针对某一目标完善想法的研究，重点关注沟通机制和方法论。
   - 对话中引用了一些个人实验，在这些实验中，agents 通过讨论和完善流程来实现特定结果，从而达到目标优化。
- **Equilibrium Propagation 优于 Backprop？**：社区探讨了 **equilibrium propagation** 作为训练基于能量的模型时 backpropagation 的替代方案，强调了其如 [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179) 所示的将预测推向最小误差配置的能力。
   - 讨论涵盖了 equilibrium propagation 与 recurrent backpropagation 之间的相似性，强调了其在神经网络训练技术中的潜在应用，正如 [Equivalence of Equilibrium Propagation and Recurrent Backpropagation](https://arxiv.org/abs/1711.08416) 中所讨论的那样。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Logits 在训练中表现优于概率**：讨论强调 **logits** 比归一化后的概率包含更多信息，认为不必要的归一化可能会阻碍优化。
   - 共识是，虽然概率对于决策至关重要，但利用 **logit space** 可以优化特定模型的训练效率。
- **Sparse Attention 受到关注**：参与者探讨了 **DeepSeek** 关于 *Native Sparse Attention* 的论文，指出了其在效率和增强上下文理解方面的意义。
   - 他们赞赏 **DeepSeek** 高标准的钻研精神以及使研究成果易于获取的能力。
- **Microsoft 进入拓扑量子比特领域**：Microsoft 推出了 **Majorana 1**，这是首个利用拓扑量子比特的 QPU，旨在实现高达一百万个量子比特的可扩展性，详见 [Microsoft Azure Quantum Blog](https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/)。
   - 一段由 Microsoft 团队参与的 [YouTube 视频](https://www.youtube.com/watch?v=wSHmygPQukQ) 解释了 **topological qubits** 的重要性及其重新定义量子计算的潜力。
- **Perplexity 突破审查障碍**：据 [The Decoder](https://the-decoder.com/perplexity-ai-removes-chinese-censorship-from-deepseek-r1/) 报道，Perplexity AI 推出了 **R1 1776**，旨在通过专门的 post-training 技术绕过 Deepseek R1 模型中的审查。
   - 这一进展展示了 **AI** 在应对和克服监管限制方面日益增长的作用。
- **Google 发布 PaliGemma 2：愿景飞跃**：Google 发布了 **PaliGemma 2 mix checkpoints**，这是一款增强型视觉语言模型，提供多种预训练尺寸，记录在他们的 [博客文章](https://developers.googleblog.com/en/introducing-paligemma-2-mix/?linkId=13028688) 中。
   - 该模型专为跨各种任务的 fine-tuning 而设计，在图像分割和科学问题回答等领域表现出色。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Sakana AI 的 AI CUDA Engineer 自动化优化**：[AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/) 自动化生产高度优化的 CUDA kernel，声称比 PyTorch 中常见的机器学习操作快 **10-100 倍**。
   - 该系统还发布了一个包含 **17,000 多个经过验证的 CUDA kernel** 的数据集以及一篇详述其能力的论文，尽管一些用户认为由于基准测试（baselines）较弱，该论文可能存在*过度炒作*。
- **Unsloth 揭晓 10 倍上下文和 90% VRAM 节省**：Unsloth 宣布了新算法，使得 **Qwen2.5-1.5B** 模型仅需 **5GB VRAM** 即可进行训练，实现了 **90% 的 VRAM 占用减少**，详情见其 [blog](https://unsloth.ai/blog/grpo)。
   - 对比基准测试显示，此前在 20K 上下文下为 **Llama 3.1** 运行标准的 GRPO QLoRA 设置需要 **510.8GB VRAM**，现在通过利用受 **Horace He** 实现启发的 **gradient checkpointing algorithm**，已降至 **54.3GB**。
- **RTX 5080+ 面临 Triton 兼容性问题**：一位成员分享了在 Triton 上使用 **TorchRL** 运行 **RTX 5080+** 的经验，强调了与 `torch.compile` 触发 Triton 相关的错误，最终通过移除 **PyTorch-triton** 安装解决了该问题。
   - 这引起了人们对 Triton 与 PyTorch 交互中仍然存在的兼容性问题的关注。
- **Raw-Dogged Tensors 赢得置换胜利**：一位成员提出了一种名为 **raw-dogged Tensor** 的新命名法，旨在使存储格式与 **MMA_Atom** 线程布局保持一致，并指出这显著降低了置换（permutation）复杂度。
   - 另一位成员确认在 **int8 matmul** 中使用了这种方法，并强调这是避免 **shared-memory bank conflicts** 的必要手段。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 略胜 Flux**：成员们发现 **Stable Diffusion (SD)** 比 **Flux** 更精致，尽管他们承认 **Flux** 仍处于活跃开发中。
   - 一位成员建议对比示例图像，看看哪种模型更符合个人品味。
- **ControlNet 掌控图像姿态**：**ControlNet** 使用深度图（depth maps）或线框图（wireframes）从姿态生成图像，处理诸如“手在前”或“手在后”等调整，以实现创意控制。
   - 成员们指出，控制方法能够根据姿态实现精确的图像生成。
- **DIY 自定义模型**：一位用户询问关于聘请兼通 **Stable Diffusion** 和艺术的艺术家来创建自定义模型和提示词风格的事宜，引发了关于实用性的讨论。
   - 社区建议，从长远来看，学习如何创建模型会更有益且更具成本效益。
- **从涂鸦到 AI 图像**：一位用户分享了他们在 iPad 上使用草图引导 AI 图像生成的流程，寻求关于将涂鸦细化为成品图像的建议。
   - 该用户发现 *img2img* 很有用，但想寻找从简单涂鸦开始的方法。
- **Nvidia GPU 仍是图像生成的王者**：**Nvidia GPU** 是流畅运行 **Stable Diffusion** 的推荐选择，而 **AMD** 选项可能会有性能问题。
   - 用户分享了 GPU 配置，并讨论了模型与 GPU 能力的兼容性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI CUDA Engineer 引发质疑**：[AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/) 是一个声称在 CUDA kernel 生成方面能实现 **10-100倍加速** 的 AI 系统，但人们对其评估的准确性以及类似项目之前的误导性陈述产生了怀疑。
   - 批评指出，一个所谓的 **150倍加速 kernel** 存在内存重用和根本性的 **bugs**，导致人们对生成的 kernel 的可靠性产生怀疑。
- **社区辩论 LLM-compiler 的可行性**：成员们推测 **LLM-compiler** 是否能将高级 PyTorch 代码翻译成优化的机器代码，引发了热烈的讨论。
   - 虽然很有趣，但大家达成共识，认为巨大的挑战（特别是缺乏通用指令集）可能会阻碍进展。
- **Clockwork RNN 架构回归**：围绕 **Clockwork RNN**（一种针对不同输入粒度使用独立模块的修订架构）的讨论引起了关注。
   - 成员们辩论了此类架构在未来模型中的可行性，包括空洞卷积 (dilated convolutions) 和注意力机制 (attention mechanisms) 的应用。
- **Llama 3.2 TPS 中的 NeoX vs NeMo**：对 NeMo 和 NeoX 在 **Llama 3.2 1B 配置** 下的对比显示，NeoX 为 **21.3K TPS**，而 NeMo 为 **25-26K TPS**，[配置文件已公开](https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/hubble/Speed_Exps/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_Llama_3_2_Fusions_All_4_FA_Swiglu.yml)。
   - 成员分享了 **[WandB run](https://wandb.ai/aflah/hubble-speed-testing/runs/nioywj5f?nw=nwuseraflah)** 以提供详细指标，供他人优化其设置。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **播客 TTS 面临挑战**：一位用户报告了 NotebookLM 中的 **TTS** 功能无法正确读取和解释其播客输入提示词的问题。
   - 该用户表示，尽管尝试了各种提示词，但仍无法实现其播客主持人所需的语气，感到非常沮丧。
- **非 Google 用户访问权限辩论**：一位成员询问是否可以像 **Google Docs** 一样，邀请没有 Google 账号的用户访问 NotebookLM 笔记本。
   - 讨论强调了为那些未集成到 Google 生态系统中的用户提供替代协作方法的必要性。
- **通过播客探索特斯拉专利**：一位用户在专利授权后分析了特斯拉的自动驾驶 AI，重点关注了 **Lidar**、**Radar** 和 **Ultrasonics** 等技术，并在播客中进行了讨论。
   - 该用户在其 Patreon 上提供了一篇 **免费** 文章，邀请听众进一步探索他们的发现。
- **AI 双人组助力 Homeschooling**：一位用户分享了他们在 **Homeschooling** 过程中整合 **NotebookLM** 与 **Gemini** 的成功经验，并将其比作拥有了技术娴熟的助手。
   - 这两个工具之间的协同作用显著帮助了教学工作的执行，提升了学习体验。
- **AI 在文学细微差别处理上的挣扎**：用户对 **AI 对文学作品的误读** 表示担忧，理由是 AI 误解了角色细节和叙事细微差别。
   - 在某些情况下，即使提供了直接证据，**AI** 也会拒绝纠正，从而与原著的完整性产生冲突。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 2025 年初路线图发布**：2025 年上半年的官方 **Torchtune 路线图**已在 [PyTorch dev-discuss](https://drive.google.com/file/d/1mKENBMrdMzMQQG1kn43Il64qWPB9uP21/view) 上发布，概述了该期间为 **Torchtune** 规划的关键方向和项目。
   - 各种项目的完整 **PyTorch 路线图**集也可以在 [dev-discuss](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794) 上访问，展示了整个平台令人兴奋的发展和正在进行的工作。
- **Packing 导致 VRAM 爆炸**：在 **max_tokens** 长度下对数据集使用 Packing 会显著增加 **VRAM 需求**，导致在 **16K 序列长度**时出现 *out-of-memory* 错误。
   - 一位用户报告在不使用 Packing 的情况下内存占用为 **30GB**，强调了巨大的资源影响。
- **注意力机制辩论升温**：讨论围绕整合**非传统 Transformer 技术**（如*稀疏注意力*和*注意力压缩*）的优先级展开，以提高**序列缩放效率**。
   - 反馈表明存在兴趣，但由于既定方法的限制，整合新研究面临阻力。
- **AdamWScheduleFree 作为优化器出现**：关于 **AdamWScheduleFree** 作为 **llama3.1 8B DPO** 默认优化器的潜力正在讨论中，该优化器已在 **2 个节点 16 个 GPU** 上进行了测试。
   - 提出了一种涉及调整 full-dpo Python 脚本的变通方法，以解决之前 FSDP 的问题。
- **Hugging Face 发布 UltraScale Playbook**：一位用户分享了托管在 Hugging Face 上的 [UltraScale Playbook 链接](https://huggingface.co/spaces/nanotron/ultrascale-playbook)，称其**令人耳目一新**。
   - 该指南旨在指导用户在实际框架内扩展模型使用。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Baseten 获 7500 万美元融资，瞄准 2025 年推理市场**：Baseten 宣布了由 @IVP 和 @sparkcapital 领投的 7500 万美元 C 轮融资，将 2025 年确定为 **AI 推理技术**的关键年份。
   - 本轮融资包括来自 @01Advisors 的 Dick Costolo 和 Adam Bain 等新投资者，强调了 **Baseten 的增长**以及在 **AI 基础设施领域**的潜力；参见[公告推文](https://x.com/basetenco/status/1892259130540179863)。
- **Mastra 的 Agent 开放使用**：开源项目 **Mastra** 推出了一款基于 Vercel AI SDK 构建 **AI Agent** 的 JavaScript SDK，强调集成和易用性；查看 [Mastra 的 Agent 文档](https://mastra.ai/docs/agents/00-overview)。
   - 开发者正在探索 **Mastra Agent** 处理访问第三方 API 和自定义函数等任务的能力，从而增强工作流自动化。
- **Arize AI 融资 7000 万美元押注可观测性**：根据其 [C 轮融资公告](https://arize.com/blog/arize-ai-raises-70m-series-c-to-build-the-gold-standard-for-ai-evaluation-observability/)，Arize AI 已筹集 7000 万美元 C 轮融资，以推进生成式和决策模型中的 **AI 评估与可观测性**。
   - 他们的使命是确保 **AI Agent** 在大规模运行时可靠工作，应对 **AI 技术**新发展带来的挑战。
- **Lambda 融资 4.8 亿美元，目标 AI 云**：Lambda 披露了由 Andra Capital 和 SGW 领投的 4.8 亿美元 D 轮融资，以巩固公司在 **AI 计算资源**领域的地位；参见 [stephenbalaban 的公告](https://x.com/stephenbalaban/status/1892275552171737220?s=46)。
   - 这笔资金将帮助 Lambda 增强其作为**专为 AI 定制的云服务**的地位，提升其能力和产品以满足日益增长的行业需求。
- **OpenAI's 用户基数飙升**：据 [Brad Lightcap](https://x.com/bradlightcap/status/1892579908179882057?s=46) 称，OpenAI 报告 ChatGPT 每周活跃用户超过 4 亿，在不到三个月的时间里增长了 33%。
   - 备受期待的 **GPT-5** 承诺向所有人提供免费无限使用，预计将整合现有模型，加剧 **AI 领域**的竞争。



---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **SSE 实现上线**：一位成员确认在其项目中成功实现了 **/sse**，标志着 **MCP functionality** 的增强。
   - 详情可在指定频道查看，突显了持续的改进。
- **Glama 调试遭遇 Cursor 混淆**：一位成员报告了在调试 **Glama hosted models** 时遇到的问题，Cursor 无法定位工具。
   - 该问题主要归因于 node 路径使用不当以及可能遗漏了必要的引号，这占了 *99% 的问题原因*。
- **解决 Docker 安装困惑**：一位新成员在通过 **Docker** 构建命令进行 **Puppeteer installation** 时需要帮助，随后对其进行了目录导航方面的澄清。
   - 提供的指导确保了他们处于正确的父目录中，并解释了命令中 `.` 的用法。
- **Python REPL 加入 MCP**：一位成员分享了一个支持 MCP **STDIO** 的简单 **Python REPL** 实现，并提供了最新的镜像以及 [GitHub repository](https://github.com/evalstate/mcp-py-repl) 链接。
   - 关于 **IPython support** 的咨询得到了乐观回应，未来可能会添加，为进一步开发开辟了途径。
- **Docker 部署步骤已明确**：一位成员分享了一篇关于部署 Docker 化 **MCP servers** 的 [blog post](https://docs.defang.io/blog/2025/02/18/model-context-protocol)，解决了跨架构的环境搭建挑战。
   - 该文章强调了 **Docker** 在确保开发环境一致性方面的作用，并提供了一份用于实现的 [reference MCP Servers](https://github.com/modelcontextprotocol/server) 列表。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX 25.1 直播已排期**：已安排直播讨论 **MAX 25.1**，可以通过 [LinkedIn 加入](https://www.linkedin.com/events/introducingmax25-17297704283980902402/theater/) 并通过 [Google Form](https://forms.gle/NkjU6e3n15TRtiMA7) 提交问题。
   - 演讲者鼓励社区分享他们的问题，强调渴望听到社区的见解。
- **Mojo 近期不太可能支持 Windows**：由于在 **Windows** 上运行 **AI clusters** 的成本高昂，原生 **Mojo Windows support** 不在近期路线图中。
   - 共识是 **nix OSes** 更适合计算任务，许多人转而使用云端 **Linux** 平台，从而降低了对 Windows 支持的紧迫性。
- **用于内存效率的 Slab Lists**：一位成员将 **slab list** 定义为一种高效的数据结构，类似于 `LinkedList[InlineArray[T, N]]`，旨在促进简洁性和良好的内存管理，并链接到了 [nickziv/libslablist](https://github.com/nickziv/libslablist)。
   - 该用户指出，这种结构在某些操作上可以达到 **O(1)** 性能，并且由于更好的缓存利用率，比链表提供更快的迭代速度。
- **Mojo 弥合 Python 性能差距**：大家一致认为 **Mojo** 源自 Python，但性能接近 C/C++/Rust，目标是未来实现类似 C++ 与 C 的兼容性。
   - 社区认为 **Mojo** 的类型系统允许 **Python-like** 的体验，吸引了诸如 **Nim** 等语言的用户。
- **Mojo 在底层易用性方面表现出色**：一位成员评论说，与 C/C++ 相比，在 **Mojo** 中处理底层任务更加用户友好，表明 Mojo 使硬件利用变得更容易。
   - 社区建议对于底层编码，**Mojo** 不需要严格遵循 Python 的语法，因为运行 Python 脚本对于许多用途来说已经足够了。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud 在欧盟上线**：[LlamaCloud EU](https://t.co/HTt2pob88p) 开启早期访问，提供全新的 SaaS 解决方案，具备安全的知识管理功能，并确保数据完全驻留在欧盟境内。
   - 此次发布旨在为需要合规解决方案的欧洲公司扫清障碍，强调了**安全性**和**数据驻留**。
- **LlamaParse 解析能力提升**：[LlamaParse](https://t.co/ux3gsvDIeW) 推出了全新的解析模式——Fast、Balanced 和 Premium，以有效满足多样化的文档解析需求。
   - 这些升级增强了处理不同文档类型的通用性，以应对现有的**文档解析挑战**。
- **Agent 陷入移交僵局**：一位开发者报告了在多 Agent 工作流（multi-agent workflow）中，LLM 反复返回 *'I am handing off to AgentXYZ'* 而不执行工具调用（tool calls）的问题。
   - 建议包括将**移交规则**直接写入 **system message** 以更好地明确预期行为，但也有人担心这会破坏现有的 Prompt。
- **Redis 竞态问题频发？**：一位用户正在寻求策略，以有效地运行 **1000 个并行批处理**来持久化摘要索引（summary index），同时避免 Redis 中的竞态条件（race conditions）。
   - 由于 Review Embeddings 存储在 Redis 命名空间中，该用户担心潜在的**键冲突（key collisions）**和**资源限制**。
- **诈骗币恶作剧！**：关于在 **Solana** 上创建代币可能性的讨论，导致社区认为此类说法是**诈骗（scams）**。
   - 此外，人们还对参与“诈骗币”项目的更广泛影响表示了担忧。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **粉色状态受到关注**：一位成员更新了他们的状态，显示为 *“我现在是粉色的了。”*
   - 这种颜色变化可能有助于增加 Discord 社区的视觉动态。
- **身份共享提议遭到质疑**：一位用户提议了一个涉及共享身份以获取 **$100-1500** 利润的合作机会，目标年龄段为 **25-50** 岁。
   - 这引发了对此类安排中身份盗用影响的担忧，且对方未提供**网站**或相关文档，并引发了关于在公共论坛披露**个人身份信息（PII）**需保持谨慎的辩论。
- **请求关于没有咖啡的世界的文章**：一位成员请求撰写一篇关于没有**咖啡**的世界所产生影响的文章，强调了其文化和经济意义。
   - 这一请求表明了对咖啡不再可用的假设情景下生活方式变化的关注。
- **沟通清晰度被视为至关重要**：书面沟通中的歧义引起了关注，建议使用**更清晰的写作**来防止误解。
   - 成员们强调了改善沟通对于促进小组内**积极协作**的重要性。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **工程师深入研究 Jamba API**：用户正积极探索 **Jamba API**，一位成员分享了用于发起 API 调用的[代码](https://docs.ai21.com/reference)并寻求语法帮助，而另一位成员则提供了详细的 API 使用大纲。
   - 该综合大纲包括了 Header 和必要的参数，为频道内的其他工程师提供了实用指导。
- **Jamba API 输出引发讨论**：关于 **Jamba API** 的输出格式出现了担忧，特别是**转义字符（escape characters）**复杂化了不同语言中的数据处理。
   - 确认了响应格式因语言而异，因此需要针对输出采用定制的处理方法。
- **PHP 工程师处理 Jamba API 集成**：一位 Symfony 和 PHP 工程师寻求关于将 **Jamba API** 响应转换为可用格式的建议，特别是处理**特殊字符处理**的问题。
   - 其他成员指出可以针对 **PHP** 特定挑战和有效的输出处理寻求同行协助。
- **提议使用 AJAX 增强 Jamba API**：一位成员建议利用 **AJAX** 来改进 **Jamba API** 的响应处理，尽管结果显示出不一致性。
   - 注意到 **Jamba 聊天窗口**的输出格式有所不同，这影响了结果的呈现方式，并可能影响处理策略。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **旧款 GeForce 在 RTX 4070 面前显得吃力**：性能测试显示，一台**旧款 GeForce 850M** 在 **8 秒**后达到 **3 tok/s**，而 **RTX 4070** 在 **1.9 秒**内即可达到 **12 tok/s**。
   - 然而，模型的整体可用性受到显著的**计算成本和数值硬度（numerical stiffness）**的限制。
- **Int8 量化导致模型偏离轨道**：成员指出，**Int8 量化**可能需要调整，因为在使用 **Int8Linear** 时，模型在几百个 token 后偶尔会“脱轨”。
   - 有建议称，关于 **tinychat** 开发的讨论应在 *私信或 GitHub* 上进行，以便更加专注。
- **速度测试中 Torch 略胜 Tinygrad**：速度测试表明，在 **2048x2048** 张量上，**torch** 的表现优于 **tinygrad**，**torch** 为 **0.22 ms**，而 **tinygrad** 为 **0.42 ms**。
   - 然而，在 **4096x4096** 张量上，**tinygrad** 仅比 **torch** 慢 **1.08 倍**，这表明其缩放性能经过了优化。
- **BEAM 可能提升性能**：增加 **BEAM** 值可能会缓解性能限制，测试显示在 **torch** 中 **BEAM=10** 时，**2048x2048** 张量的耗时为 **0.21 ms**。
   - 不同张量大小下的性能表现一致，突显了更高 **BEAM** 配置的潜在收益。
- **新的 PyTorch 频道上线**：已创建一个专门用于 **PyTorch** 讨论的新频道。
   - 意图是随着用户贡献的增加，鼓励更专注、更深入的交流。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **系统消息（System Message）术语引发困惑**：一位成员澄清说，UI 中现在使用了 *“system message”* 一词，表明命名惯例发生了变化。
   - 另一位参与者肯定地表示，在操作这些系统时，旧习惯很难改变。
- **系统消息中的指令：纯英文可以吗？**：提到在 *“system message”* 中可以使用纯英文指令，大多数模型都会遵循这些命令。
   - 一些成员对这一过程的简便性表示怀疑，询问使用 **Jinja** 或 **JSON** 代码是否更有效。
- **GPT4All 在图像处理方面表现不佳**：一位成员询问是否可以像其他 AI 平台一样直接将图像粘贴到文本栏中，但得到的答复是 **GPT4All** 无法处理图像。
   - 建议使用外部软件来完成此类任务。
- **Nomic 和 NOIMC v2：是真的吗？**：一位成员对 **NOIMC v2** 的实现表示困惑，质疑为什么它看起来实现得不正确。
   - 另一位成员幽默地寻求确认自己是否在 **Nomic** 频道，以此表达他们的沮丧。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **2024 年 LLM Agents 课程依然有用**：一位成员建议，虽然不是必须的，但旁听[此 YouTube 播放列表](https://youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&feature=shared)中的 **2024 年秋季课程**可以加深理解，特别是对于 **DSPy**。
   - 他们指出，本学期的教学大纲中没有 **DSPy**，这使得 2024 年秋季课程对那些对其感兴趣的人特别有用。
- **LLM Agents 课程测验已存档**：针对测验从当前教学大纲中消失的困惑，一位成员分享了 2024 年秋季课程的**测验存档**链接，位于[此处](https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit?usp=sharing)。
   - 那些开始学习较晚并想赶上进度的学生现在可以访问这些测验。
- **在 MOOC 上查找测验入口**：针对寻找**测验 1 和 2** 的用户，有人指出可以在 MOOC 的[页面](https://llmagents-learning.org/sp25)或[公告页面](https://llmagents-learning.org/f24)找到测验。
   - 还提到*所有证书均已发放*，并鼓励学生报名参加 [2025 年春季班](https://llmagents-learning.org/sp25)。
- **课程结业通知**：**LLM Agents MOOC** 已结束，但视频讲座在教学大纲中仍可访问。
   - 所有证书均已发放，鼓励学生报名参加 [2025 年春季班](https://llmagents-learning.org/sp25)。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Qwen/Qwen2.5-VL-7B-Instruct 在 HaizeLabs Judge Compute 上的评分存在差异**：一位成员复现了与 **HaizeLabs Judge Compute** 相同的数据集，并发现使用 **Qwen/Qwen2.5-VL-7B-Instruct** 模型的评分范围从 2-stage 优化的 **60%-70%** 到 mipro2 的 **88.50%** 不等。
   - 名为 **LLM-AggreFact_DSPy** 的项目已在 [GitHub](https://gist.github.com/fullstackwebdev/fa4934fb4669cfc3e8c6ced950ea7a22) 上共享，其中包含与评估相关的源代码，可以更深入地了解所使用的方法论。
- **Leonard Tang 发布 Verdict 库**：Leonard Tang 发布了 [Verdict](https://x.com/leonardtang_/status/1892243653071908949)，这是一个针对 judge-time compute scaling 的库，并指出 AI 的可靠性问题源于评估而非生成。
   - 他强调，AI 的下一个进步应该集中在评估的改进上，这与对 **pre-training** 和 **inference-time scaling** 的强调形成了对比。
- **DSPy 对话历史探讨**：一位成员询问 DSPy 是否会自动将对话历史注入到调用中，这表明在进一步实现之前需保持谨慎。
   - 这凸显了在不无意中覆盖先前上下文的情况下管理 AI 交互的潜在复杂性，特别是在更复杂的应用中。
- **导出 Prompt 到消息模板的说明**：一位成员分享了一个 FAQ，解释了如何通过使用带有 `dspy.ChatAdapter()` 的 Python 代码片段来冻结并导出 Prompt 到消息模板中。
   - 对方澄清说，这种方法会导致控制流逻辑丢失，并建议使用 `program.save()` 或 `program.dump_state()` 作为更全面导出的替代方案。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1341820211363582083)** (979 messages🔥🔥🔥): 

> `Grok 3 performance, SuperGrok subscription, Comparison with OpenAI models, Grok's capabilities, Community feedback` 


- **Grok 3 超越 OpenAI 模型**：Grok 3 展示了优于 OpenAI 模型的性能，特别是在基准测试和 ChatGPT Pro 难以应对的特定编程任务中表现出色。
   - 用户报告称 Grok 3 解决了 o1 Pro 无法处理的复杂问题，这增强了人们对其能力的信心。
- **SuperGrok 提供更高的价值**：许多用户正考虑转向 SuperGrok，因为与 ChatGPT Pro 的 250 美元相比，它每月 30 美元的价格更具性价比。
   - SuperGrok 被认为具有显著优势，特别是在性能和使用限制方面。
- **Grok 期待的功能**：社区成员对 Grok 即将推出的功能感兴趣，例如语音模式和自定义指令（custom instructions），他们认为这将进一步增强其效用。
   - 预计这些功能将使 Grok 在处理上下文和易用性方面比其他模型更具竞争力。
- **关于 AI 订阅模式的讨论**：用户讨论了各种可用的订阅模式以及现有服务的局限性，由于 Grok 3 提供了更好的服务和价格，因此更受青睐。
   - 对话显示出一种普遍情绪，即鉴于新竞争对手的出现，许多人正在重新评估他们的 AI 服务订阅。
- **Grok 的 API 和能力**：Grok 3 模型的 API 因其无限制的能力而受到关注，允许进行广泛的交互，而没有某些其他模型中常见的严格限制。
   - 用户表达了对更多集成和功能的渴望，以最大限度地发挥 Grok 平台的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://grok.com/share/bGVnYWN5_b5ffc957-9e88-4942-96aa-80372f58d995">使用 SDRPlay RSP 2 接收木星信号 | Grok 对话分享</a>: 我如何使用我的 SDRPlay RSP 2 接收像 JOVE 这样的木星信号？</li><li><a href="https://x.ai/blog/grok-image-generation-release">Grok 图像生成发布</a>: 未找到描述</li><li><a href="https://grok.com/">Grok</a>: Grok 是由 xAI 设计的免费 AI 助手，旨在最大限度地提高真实性和客观性。Grok 提供实时搜索、图像生成、趋势分析等功能。</li><li><a href="https://grok.com/share/bGVnYWN5_7ddf224c-9606-4e02-a956-22135248bc79">适用于 JOVE 接收器与 SDRPlay RSP2 的偶极天线 | Grok 对话分享</a>: 向我展示如何为使用 SDRPlay RSP 2 的 JOVE 接收器制作合适的偶极天线</li><li><a href="https://grok.com/share/bGVnYWN5_76a8e85f-5559-4230-8ef4-f7730b83056b">低潮时淹没的梯子横梁 | Grok 对话分享</a>: 有一个梯子固定在船上，共有 10 级横梁。在低潮时，水位下降了 60 厘米。每一级...</li><li><a href="https://grok.com/share/bGVnYWN5_ccd48442-c8fb-4d56-ae0f-739cf884de16">在澳大利亚购买 Allstar Node | Grok 对话分享</a>: 在澳大利亚哪里可以买到完全组装好的即插即用 Allstar node？</li><li><a href="https://grok.com/?show_subscribe=1">Grok</a>: Grok 是由 xAI 设计的免费 AI 助手，旨在最大限度地提高真实性和客观性。Grok 提供实时搜索、图像生成、趋势分析等功能。</li><li><a href="https://grok.com/share/bGVnYWN5_67cdd414-63d1-427a-b6ee-54e4d24d738e">HackerNews 热门故事概览 | Grok 对话分享</a>: 总结今天 HackerNews 首页的热门结果。对于有趣的文章，请深入探索...</li><li><a href="https://grok.com/share/bGVnYWN5_26c250ce-ff40-4328-9385-bd71cbf04f80">Grok 3 免费计划限制 | Grok 对话分享</a>: Grok 3 目前的限制是什么？我正在使用免费计划（Elon Musk 推特说它是...）</li><li><a href="https://grok.com/share/bGVnYWN5_4e986ed7-df82-4227-841f-1bdeae4fc961">SuperGrok: AI 订阅还是加密货币？ | Grok 对话分享</a>: 我应该购买 SuperGrok 吗？</li><li><a href="https://tenor.com/view/yarp-hot-fuzz-gif-12386003">Yarp Hot Fuzz GIF - Yarp Hot Fuzz - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=pEErLop52Jw">危险的“表情符号黑客”：AI 模型易受“特洛伊木马”表情符号的影响...</a>: 最新的 AI 新闻。了解 LLM、Gen AI 并为 AGI 的推出做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 领域的最新动态。</li><li><a href="https://en.wikipedia.org/wiki/Catastrophic_interference">Catastrophic interference - 维基百科</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1342180418010611772)** (1 条消息): 

> `Feature Requests, Chat Tracking Methods` 


- **鼓励分享想法**：一位成员建议在指定频道发布想法，因为这是让 **OpenAI** 看到这些想法并让其他人参与互动的绝佳方式。
   - *如果你也想要这个功能，请评论并分享！*
- **保存聊天 URL 以供日后参考**：一位成员提议保存 **聊天 URL**，以便轻松返回有价值的讨论。
   - 他们还建议使用 **'good1'** 或 *'Track this chat'* 等关键词来帮助记忆重要的聊天记录。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1342241982667427941)** (2 条消息): 

> `Software troubleshooting, Insights for improvement` 


- **对故障排除电话会议的期待**：一位成员表示非常期待定于明天的电话会议，以确定问题是由 **prompt** 引起的还是 **software** 运行故障。
   - 他们幽默地提到解决问题花费的时间比预期的要长，说 *it's taking too much time than I expected*。
- **感谢有用的建议**：该成员感谢了其他人的 **有用建议**，并表示会将这些见解记在心中以备后用。
   - 然而，他们觉得在某个特定案例中可能需要额外的支持，并对 *not sure yet what exactly* 表示不确定。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1342241982667427941)** (2 条消息): 

> `Prompt issues, Software performance` 


- **对明天电话会议的期待升温**：一位成员对即将到来的电话会议表示兴奋，思考所面临的问题是由于 **prompt** 还是 **software** 表现异常。
   - 在寻求明确答案的过程中，他们表示 *It’s taking too much time than expected*。
- **感谢支持，但仍寻求更多帮助**：该成员感谢了其他人的建议，相信分享的见解在未来会有所帮助。
   - 然而，他们提到在特定情况下需要 *something else*，表明对后续步骤尚不确定。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1341881225656139799)** (1 条消息): 

> `DeepSeek-V3 Unlimited, Windsurf Pro and Ultimate Plans, Prompt and Flow Action Credits` 


- **DeepSeek-V3 开启无限制模式！**：DeepSeek-V3 现在对 **Windsurf Pro** 和 **Ultimate** 计划的用户无限制开放，允许不受限的访问。
   - 此次更新伴随着 **0 prompt credits** 和 **0 flow action credits**，实现了无限制的无缝使用。
- **冲向新功能**：鼓励用户通过 [这条推文](https://x.com/windsurf_ai/status/1892322088507105561) 查看公告，该推文强调了新的无限制访问。
   - 随着 Windsurf 的不断进化，让我们满怀热情地投入到这些更新中吧！ <:windsurf:1306309317011570699>



**提到的链接**：<a href="https://x.com/windsurf_ai/status/1892322088507105561">Windsurf (@windsurf_ai) 的推文</a>：DeepSeek-V3 现在在 Windsurf Pro 和 Ultimate 计划中无限制使用。0 prompt credits。0 flow action credits。

  

---


### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1341954445147377685)** (1 条消息): 

> `MCP content, Use cases for MCP, MCP in Cascade` 


- **令人兴奋的 MCP 内容发布**：一位成员分享了 Matt Li 展示的 **MCP 酷炫使用案例**，鼓励大家在 [X](https://x.com/windsurf_ai/status/1892394489588727985) 上查看内容。
   - *Go show some love on the post* ❤️ 突显了社区对参与不断扩展的 MCP 功能的热情。
- **展示 MCP 的潜在使用案例**：原始帖子包含一个快速演示，说明了 **MCP** 如何在 **Cascade** 中运行，提高了对其功能的认识。
   - 这个演示为那些对 MCP 仍有疑问的人提供了资源，促进了对其能力的进一步探索。



**提到的链接**：<a href="https://x.com/windsurf_ai/status/1892394489588727985">Windsurf (@windsurf_ai) 的推文</a>：如果你对 MCP 及其潜在使用案例仍有疑问，这里有一个关于 MCP 如何在 Cascade 中运行的快速演示！

  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1341855441151332434)** (86 messages🔥🔥): 

> `JetBrains 中的 Codeium 插件, Supercomplete 功能, Windsurf 安装要求, Codeium 与 CodeBuddy 的对比, 对 Codeium 支持的担忧` 


- **Codeium 插件面临 EOL 推测**：由于用户对其缺乏发展方向感到沮丧，人们开始担心 **JetBrains Codeium 插件** 可能不再受支持。
   - 一位用户评论道：*看到 Codeium 作为一个插件被放弃真的很遗憾。*
- **Supercomplete 彻底改变了自动补全**：Codeium 中的 **Supercomplete** 功能可以预测用户在简单自动补全之外的操作，提供相关的编辑和上下文感知建议。
   - 用户强调了它在重构中的价值，并表示：*这种能力对于单文件中的代码处理非常出色。*
- **试用版强制要求安装 Windsurf**：为了获得 Pro 版本的试用权限，用户必须 **注册并下载 Windsurf**，尽管免费版本不需要安装。
   - 在关于使用插件是否必须安装 Windsurf 的疑问中，这一点得到了澄清。
- **Codeium 与 CodeBuddy 的对比**：**CodeBuddy** 和 Codeium 被列为首选方案，一位用户表示有兴趣在做出决定前对两者都进行尝试。
   - 另一位用户指出，虽然 CodeBuddy 拥有更便捷的聊天功能，但 Codeium 的自动补全目前表现更佳。
- **预期的 API 改进**：用户正热切期待改进，例如 Codeium API 即将增加对 Grok 3 的支持。
   - 一位成员评论了 Grok 在创造性解决问题方面的优势，为功能讨论增添了内容。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://code.visualstudio.com/blogs/2025/02/12/next-edit-suggestions">Copilot Next Edit Suggestions (preview)</a>: 宣布 Visual Studio Code 中 GitHub Copilot 的 Next Edit Suggestions 和 Agent 模式。</li><li><a href="https://codeium.com/supercomplete">Supercomplete | Windsurf Editor and Codeium extensions</a>: Supercomplete 能够预测你的下一个意图，无论光标位置如何。无论你是想插入、删除还是编辑，Supercomplete 都能满足你的需求。</li><li><a href="https://codeium.canny.io/feature-requests/p/supercomplete-for-jetbrains">Supercomplete for Jetbrains | Feature Requests | Codeium</a>: 我认为 JetBrains 在“连续动作建议”领域最欠缺。Supercomplete 将会是该领域首创的功能。
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1341818857756557457)** (546 messages🔥🔥🔥): 

> `Windsurf 易用性问题, DeepSeek vs Cascade Base, Cascade 中的 Memory 系统, MCP 服务器配置, 支持响应查询` 


- **Windsurf 使用体验令人沮丧**：用户对 Windsurf 的功能表示持续沮丧，理由包括 UI 错误和对代码的意外更改。
   - 普遍共识是采用更小的增量进行工作，并频繁提交更改以避免丢失进度。
- **DeepSeek 与 Cascade Base 的性能对比**：DeepSeek v3 被视为 Cascade Base 的可靠替代方案，但在没有自定义指令的情况下，它在可靠调用工具方面存在局限性。
   - 然而，Cascade Base 得益于基于 Llama 3.1 70b 的微调，在工具调用方面表现出色。
- **在 Cascade 中使用 Memory 系统的策略**：鼓励用户使用“add to memory”和“update memory”等命令，以确保 Cascade 保持项目细节。
   - 提议将全局规则结构化到独立文件中，旨在增强组织性并提高 Cascade 的性能。
- **MCP 服务器配置的挑战**：多位用户在 MCP 服务器设置中遇到问题，导致出现错误，直到调整或删除配置文件。
   - 建议将反映错误的配置重新定位，以解决 Windsurf 性能的潜在问题。
- **关于支持和响应时间的查询**：用户在接收支持工单响应时遇到延迟，有些人甚至没有收到任何自动回复。
   - 预期的响应应在邮件主题行包含工单编号，但用户对这些通信的邮件来源表示困惑。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://codeium.canny.io/">Codeium Feedback</a>: 向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队以获得个性化协助。</li><li><a href="https://codeium.canny.io/feature-requests/p/placeholder-input-does-not-change-when-changing-windsurf-open-chat-with-cascade">Placeholder input does not change when changing `Windsurf: Open Chat with Cascade` keybind | Feature Requests | Codeium</a>: 请参阅附带的屏幕截图</li><li><a href="https://codeium.canny.io/feature-requests/p/devcontainer-support">Devcontainer Support | Feature Requests | Codeium</a>: 希望能有更多 Devcontainer 支持。具体包括：在容器中重新构建并重新打开（目前只有“在容器中重新打开”）。需要它来安装扩展。</li><li><a href="https://docs.codeium.com/windsurf/mcp>">Welcome to Codeium - Codeium Docs</a>: 未找到描述</li><li><a href="https://tenor.com/view/japanese-tyranno-dance-japanese-dance-tyranno-gif-10262458857606665890">Japanese Tyranno Dance Japanese Dance GIF - Japanese Tyranno dance Japanese dance Tyranno - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/donvito/status/1892640143145644056">Tweet from Melvin Vivas (@donvito)</a>: 终极 MCP 教程 🤯🤯🤯 学习如何在 Cursor、Windsurf 和 Claude 中配置 MCP。在本教程中，我们使用了 GitHub MCP 服务器。一个推文串 🧵👇</li><li><a href="https://glama.ai/mcp/servers/vwi6nt8i80">supabase-mcp</a>: 一个提供与 Supabase 数据库、存储和 Edge Functions 交互工具的 MCP 服务器。</li><li><a href="https://x.com/sdrzn/status/1892262424881090721">Tweet from Saoud Rizwan (@sdrzn)</a>: Cline v3.4 发布了 🚀 推出 MCP Marketplace！直接在扩展中发现并安装最好的 MCP 服务器，Cline 会处理所有设置。我们还在 Plan 模式中添加了 Mermaid 图表...</li><li><a href="https://tenor.com/view/mr-bean-mrbean-bean-mr-bean-holiday-mr-bean-holiday-movie-gif-3228235746377647455">Mr Bean Mrbean GIF - Mr bean Mrbean Bean - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/albert-einstein-lol-think-be-smart-think-wise-gif-8735407">Albert Einstein Lol GIF - Albert Einstein Lol Think - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/i-have-a-question-david-crane-frasier-may-i-ask-a-question-i-have-an-inquiry-gif-12327607075242146179">I Have A Question David Crane GIF - I have a question David crane Frasier - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://codeium.canny.io/feature-requests/p/git-awareness">Git awareness | Feature Requests | Codeium</a>: 由于 Git 对软件开发过程至关重要，Windsurf 应该随时完美感知：git status、git diff 以及可能的更多内容。</li><li><a href="https://youtu.be/iBiNfa32AnE?si=0nsiCJAlGa8If-1l">The ONLY Windows PC OPTIMIZATION Guide You Will EVER Need In 2024</a>: 2024 年你唯一需要的 Windows PC 优化指南！优化/提升游戏 PC 上 Windows 性能的最佳快速指南！如何为游戏优化 Windows 10 - FPS 和无延迟的最佳设置！在今天的...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1341818452091994143)** (485 条消息🔥🔥🔥): 

> `Unsloth AI 模型、GRPO 训练更新、训练损失问题、蒸馏模型性能、AI 社区见解`

- **Long Context GRPO 发布**：Unsloth 推出了 Long Context GRPO，仅需 5GB VRAM 即可训练推理模型，承诺上下文长度增加 10 倍，VRAM 占用减少 90%。
   - 社区反响热烈，用户注意到性能提升，并对 Unsloth 提供的免费资源表示感谢。
- **训练损失波动**：用户观察到模型训练期间训练损失（Training Loss）存在显著波动，通常在多个 Epochs 后才会稳定。
   - 建议包括调整学习率并保持训练提示词（Prompts）的清晰度，以减少过拟合（Overfitting）并改善学习效果。
- **蒸馏模型的局限性**：关于使用蒸馏模型进行 GRPO 训练的局限性讨论指出，如果不进行适当调整，这些模型可能无法生成预期的输出格式。
   - 用户反馈蒸馏模型生成的格式与要求不符，强调在某些情况下需要采用两阶段方法。
- **社区参与和实验**：社区正积极分享优化模型训练和输出准确性的经验、技巧和技术。
   - 常见做法包括利用结构化输出和精炼提示词工程（Prompt Engineering）来增强模型的理解能力。
- **微调中的挑战**：参与者表达了在模型微调（Fine-tuning）中面临的共同挑战，特别是在处理复杂数据集和保持有意义的摘要方面。
   - 一些用户建议在训练前利用命名实体识别（NER）模型来协助管理公司特定的专业术语和缩写。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://www.youtube.co">无标题</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-vllm">Saving to VLLM | Unsloth Documentation</a>: 将模型保存为 16bit 以用于 VLLM</li><li><a href="https://colab.research.google.com/drive/1ZF4qWG0CO67j8gm0hoeGiEXXFBPFyF2X?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/overview">AI Mathematical Olympiad - Progress Prize 2</a>: 使用人工智能模型解决国家级数学挑战</li><li><a href="https://docs.vllm.ai/en/latest/">Welcome to vLLM &#8212; vLLM</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=lyVxD0bJDOk">Start Up Wednesday with Unsloth.AI</a>: 结识 Daniel 和 Michael Han，这对澳大利亚兄弟正通过 Unsloth 改变 AI 开发。他们的开源项目使模型微调速度提高 2 倍...</li><li><a href="https://x.com/UnslothAI/status/1892640995847901684">Tweet from Unsloth AI (@UnslothAI)</a>: 今天，我们推出了新算法，可实现 10 倍长的上下文长度和减少 90% 的 VRAM 来训练推理模型 (GRPO)。使用 Unsloth，你现在仅需 5GB 即可训练自己的推理模型...</li><li><a href="https://github.com/vllm-project/vllm/issues/13486.">vllm-project/vllm</a>: 一个高吞吐量且显存高效的 LLMs 推理和服务引擎 - vllm-project/vllm</li><li><a href="https://www.youtube.com/watch?v=bAWV_yrqx4w">[GRPO Explained] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>: #deepseek #llm #grpo GRPO 是 Deepseek-R1 使用的核心改进之一，但去年在这篇结合了 n... 的论文中就已经引入。</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl">Reasoning - GRPO &amp; RL | Unsloth Documentation</a>: 使用 Unsloth 通过 GRPO 训练你自己的 DeepSeek-R1 推理模型，GRPO 是强化学习 (RL) 微调的一部分。</li><li><a href="https://github.com/jingyaogong/minimind/blob/master/README_en.md">minimind/README_en.md at master · jingyaogong/minimind</a>: 🚀🚀 「大模型」2小时完全从0训练26M的小参数GPT！🌏 Train a 26M-parameter GPT from scratch in just 2h! - jingyaogong/minimind</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit">unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥</a>: 以 2 倍的速度和减少 70% 的显存微调 Llama 3.3, DeepSeek-R1 和推理 LLMs！🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1341934054785679412)** (17 messages🔥): 

> `Unsloth Art, Quantum Computing, Triton Language in Challenges, Cohesion Timing Hardware, Inline Assembly in Triton` 


- **Unsloth 艺术涉及 AI 和艺术家**：讨论透露，**3D 树懒**是 AI 生成的，而贴纸是由一位**才华横溢的艺术家**创作的。
   - 成员们对其中涉及的创意表示赞赏，一致认为：*艺术作品确实很棒！*
- **Majorana 1 芯片带来的量子计算进展**：一段名为 [Majorana 1 Explained](https://youtu.be/wSHmygPQukQ) 的 YouTube 视频展示了 Microsoft 团队讨论新芯片在**量子计算**方面的突破。
   - 然而，有人指出该技术在运行时仍需要**氦制冷机 (helium fridge)**。
- **澄清 Triton 中的 custom_asm_works**：一位成员寻求关于 **custom_asm_works** 在挑战赛评分系统中具体指代的澄清。
   - 解释称，这涉及 Triton 中的内联汇编 (inline assembly)，允许在不使用 **CUDA** 的情况下对张量 (tensor) 进行执行。
- **硬件内聚时序 (Cohesion timing) 的担忧**：有人提到对所讨论硬件的**内聚时序**感到好奇，特别是与最近的量子进展相关的部分。
   - 具体的影响和细节尚未深入探讨，但表达了对技术层面的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html">triton.language.inline_asm_elementwise &mdash; Triton 文档</a>：未找到描述</li><li><a href="https://youtu.be/wSHmygPQukQ?si=4VyaksRGdCXpnNeE">Majorana 1 Explained: The Path to a Million Qubits</a>：听取来自 Microsoft 团队关于近期物理学和量子计算突破的分享，这些突破由新型 Majorana 1 芯片展示...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1341819278155976876)** (32 messages🔥): 

> `Installing Unsloth, RTX 5090 Mobile Specs, GPU Performance and Fine-tuning, VRAM Usage in Datasets, Qwen2.5 Model Inference Issues` 


- **新用户寻求 Unsloth 安装帮助**：一位新成员请求协助安装 **Unsloth**，并提到他们是从另一个 Discord 服务器介绍过来的。
   - *希望能获得指导*，他们与社区展开交流以快速上手。
- **RTX 5090 Mobile 规格发布**：**RTX 5090 Mobile** 将配备 **24GB** 显存，预计下周开始预订。
   - 这一消息引发了考虑升级硬件的成员们的关注。
- **关于 GPU 性能的讨论**：成员们分享了各自的 **GPU 配置**，其中一人提到 **3x24GB GPU** 的运行速度为 **1 token/sec**，而另一人在 **96GB VRAM** 下达到了 **3t/s**。
   - 对话探讨了通过现有硬件提升性能的优化策略。
- **训练中 VRAM 攀升的担忧**：一位用户询问 **VRAM 使用量** 随数据集长度不均而上升的问题，质疑这是巧合还是已知问题。
   - 社区回应建议测试现有的解决方案，有人推荐使用 SFTTrainer 中的 **packing** 选项。
- **Qwen2.5 模型输出的不一致性**：在微调 **Qwen2.5-VL3B 模型**后，一位用户报告称，使用**合并模型 (merged model)** 与独立 **LoRA adapter** 相比，输出结果不一致。
   - 尽管正确加载了模型，但对于 **vLLM** 生成的不同输出仍感到困惑，这引发了社区的进一步排查。



**提到的链接**：<a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#id-7.-running--saving-the-model">微调指南 | Unsloth 文档</a>：了解微调的所有基础知识。

  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1341919319591882855)** (4 messages): 

> `RAG vs Fine Tuning, Video Examples, Kolo Usage, Industry Insights` 


- **视频探讨 RAG vs Fine Tuning**：分享了一个[标题为 "RAG vs. Fine Tuning (Live demo)" 的 YouTube 视频](https://www.youtube.com/watch?v=LDMFL3bjpho)，质疑 Fine Tuning 是否比传统的 RAG 系统能提供更好的结果。
   - *哪一个更好？* 该视频旨在挑战目前行业内关于这两种方法有效性的主流观点。
- **希望在演示中看到更多示例**：一位观众表示，希望在演示过程中看到更多比较 **RAG** 和 **Fine Tuning** 的示例。
   - *是否可以展示更多示例？* 这一询问凸显了在未来的迭代中需要更深入见解的需求。
- **未来 Kolo 视频的计划**：创作者对反馈做出了回应，表示计划制作后续视频，详细介绍如何开始使用 **Kolo**。
   - *我稍后可能会再制作一个*，其中包括全面的测试和训练数据见解。



**提到的链接**：<a href="https://www.youtube.com/watch?v=LDMFL3bjpho">RAG vs. Fine Tuning (Live demo)</a>：RAG 还是 Fine Tuning 更好？行业是否搞错了？Fine Tuning 能比传统的 RAG 系统提供更好的结果吗？观看视频了解...

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1341824741098651700)** (56 messages🔥🔥): 

> `Rigor in Science, Citizen Science, AI in Medicine, Content Moderation Research, Phytochemical Formulations` 


- **关于科学严谨性的辩论**：成员们讨论了科学中**严谨性**的重要性，批评了在没有经过适当审查和认可的情况下，将 **ChatGPT** 等 AI 用于科学主张。
   - 有人担心，仅仅拥有 **PhD** 或研究头衔并不能保证质量，许多人认为这种趋势削弱了科学的可信度。
- **公民科学与资质**：一位成员强调，有效的研究并不完全取决于学术资质，**公民科学**在知识生产中发挥着重要作用。
   - 社区辩论了是否需要**学位**才能被视为科学家，一些人强调了该领域的例外情况。
- **AI 内容审核的挑战**：讨论强调了创建能够客观审核内容的 AI 系统的挑战，建议使用 **BERT 分类器**等解决方案，同时也承认其局限性。
   - 一位成员引用了一篇与**内容审核**相关的付费研究文章，强调需要将主观感受与客观事实分开。
- **AI 驱动的营养保健品研究**：一位成员介绍了他们使用 AI 为各种疾病创建针对性营养保健品配方的工作，并分享了相关链接和文档作为研究材料。
   - 尽管提供了大量关于需要临床试验的警告，但其他人对向公众展示此类信息的伦理表示担忧。
- **对 AI 在医学中应用的批评**：几位成员讨论了使用 AI 过度简化医疗建议的潜在危险，认为这破坏了医疗实践和伦理的复杂性。
   - 有人担心 AI 生成的内容可能会误导晚期患者，但也有人主张尽管存在潜在的误解，仍应取得进展。



**提到的链接**：<a href="https://www.marielandryceo.com/2025/02/title-ai-powered-phytochemical.html?m=1">Title: AI-Powered Phytochemical Formulation: A Data-Driven Approach to Supporting Health</a>：未找到描述

  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1341816752169947178)** (381 messages🔥🔥): 

> `Hunyuan Image Generation Model, A100 GPU Performance, Speculative Decoding Analysis, LM Studio Features, Embedding Models for Long Texts`

- **Hunyuan 图像生成模型**：Hunyuan 图像生成模型现已可用，但至少需要 24GB 的 VRAM，且主要在 NVIDIA 显卡上运行，生成视频内容需要几分钟时间。
   - 用户表达了对该模型进行实验的兴趣，特别是其与其他平台相比的能力。
- **A100 GPU 性能**：用户讨论了 A100 GPU 在 LM Studio 中的功能，指出它们在处理 AI 任务时非常有效，特别是强调了其 80GB 的 VRAM 容量。
   - 尽管成本可能很高，但用户对获取 A100 以获得更好性能的兴趣显而易见。
- **Speculative Decoding 分析**：有观点指出，在某些模型中使用 Speculative Decoding 可能会导致较低的 Token 接受率和较慢的性能。
   - 用户分享了关于 Token 接受率的不同经验，并就最大化性能的最佳模型设置提出了疑问。
- **LM Studio 功能**：用户对 LM Studio 表示满意，称其在为 AI 项目节省时间和成本方面非常高效。
   - 对话内容包括关于在平台中选择模型和可用功能的易用性讨论。
- **长文本 Embedding 模型**：讨论了专为长文本设计的 Embedding 模型的性能，并建议使用支持更大上下文窗口（Context Window）的特定模型。
   - 使用 7B 模型分析长文本的结果表明，在处理大量材料后，它能够准确回答查询。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com>">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px">Efficient-Large-Model/Sana_1600M_1024px · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/tkt-smart-gif-20642718">Tkt Smart GIF - Tkt Smart - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/ggml-org/llama.cpp/discussions/11442">在部分卸载期间强制所有计算在 GPU 上运行 · ggml-org/llama.cpp · Discussion #11442</a>: 我建议添加一个命令行参数形式的选项，以便在以 CPU RAM 作为卸载缓冲区的部分卸载期间，强制所有计算在 GPU 上运行。这将允许我们保持……</li><li><a href="https://tenor.com/view/imagination-spongebob-squarepants-dreams-magic-gif-12725683">Imagination Spongebob Squarepants GIF - Imagination Spongebob Squarepants Dreams - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://gitlab.com/logliwo/lm-studio-docker-compose">Aleksey Tsepelev / LM-Studio docker-compose · GitLab</a>: GitLab.com</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://tenor.com/view/doubt-it-i-dont-believe-you-will-farrell-anchor-man-gif-5332521">Doubt It GIF - Doubt It I Dont Believe You Will Farrell - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=JAG_83hj1SI">DeepSeek AI 真的能在 5 分钟内编写一个 Python 加密货币交易机器人吗？</a>: 🔥 *交易手续费 9 折优惠！* 使用我的 Bitget 链接注册：https://bonus.bitget.com/Robottraders 💡 *你将在此视频中学到什么*：我让 DeepSeek AI 尝试……</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI。通过在 GitHub 上创建一个账户来为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.10)">未找到博文</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF">unsloth/DeepSeek-R1-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1">unsloth/DeepSeek-R1 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (原 StableSwarmUI)，一个模块化的 Stable Diffusion Web 用户界面，强调易于访问的强大工具、高性能和可扩展性。</a>: SwarmUI (原 StableSwarmUI)，一个模块化的 Stable Diffusion Web 用户界面，强调易于访问的强大工具、高性能和可扩展性。 - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/erwold/qwen2vl-flux">GitHub - erwold/qwen2vl-flux</a>: 通过在 GitHub 上创建一个账户来为 erwold/qwen2vl-flux 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1341844278040727702)** (190 条消息🔥🔥): 

> `Apple Silicon 性能、ARM 与 x86 架构、Intel 的市场竞争、最新 AMD Ryzen AI Max+ 规格、内存配置与性能` 


- **Apple Silicon 的集成设计引发批评**：关于 Apple 在笔记本电脑中焊接组件趋势的讨论引发了对可维修性和可升级性的担忧，导致用户更倾向于允许组件升级的系统。
   - 用户对内存配置的限制表示沮丧，特别强调缺乏灵活性限制了性能的提升。
- **ARM 架构被视为具有局限性**：批评者认为，与传统的 x86 系统相比，向 ARM 架构（特别是在笔记本电脑中）的转变更多是由营销驱动的，而非切实的性能优势。
   - 用户对这些系统在软件和进程管理方面的低效表示担忧，导致了用户的不满。
- **Intel 在竞争中挣扎**：参与者反思了 Intel 显著落后于 AMD 和 Apple 等竞争对手的现状，以及其设计决策对功耗和整体性能的影响。
   - 讨论中对 Intel 的未来发展持谨慎乐观态度，这取决于其在技术上追赶的能力。
- **AMD Ryzen AI Max+ 令人印象深刻但仍存疑问**：Ryzen AI Max+ 的规格引起了用户的兴趣，但对其与现有 GPU 相比的实际性能仍持怀疑态度。
   - 观点反映出对独立基准测试的谨慎期待，以真实评估这一新架构与成熟竞争对手的对比。
- **内存性能考虑集成选项**：技术爱好者讨论了内存速度和架构对整体系统性能的影响，特别是比较了 HBM 和 DDR 配置。
   - 对话强调了在通过集成组件获得性能提升与确保桌面环境中的整体用户控制权之间所面临的权衡。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/">GPTQ, AWQ, EXL2, q4_K_M, q4_K_S 和 load_in_4bit 之间的详细比较：perplexity、VRAM、速度、模型大小和加载时间。- LLM 博客</a>: 未找到描述</li><li><a href="https://www.club386.com/amd-ryzen-ai-max-cpus-beat-nvidia-rtx-4090-at-llm-performance/">AMD Ryzen AI Max+ CPU 在 LLM 工作负载中击败 Nvidia RTX 4090</a>: AMD 扩展了其针对 AI 工作负载的移动芯片选择，由一款比独立显卡更强大的笔记本 CPU 领衔。</li><li><a href="https://hothardware.com/reviews/rog-flow-z13-review">ASUS ROG Flow Z13 评测：AMD Strix Halo 是一头强大的猛兽</a>: 我们首款基于 AMD Ryzen AI MAX 的设备以出色的性能和稳健的电池续航给人留下深刻印象。</li><li><a href="https://tenor.com/view/cat-despair-meme-atone-gif-18083281511005463831">猫咪绝望 GIF - 猫咪绝望表情包 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.club386.com/amd-ryzen-ai-max-cpus-beat-nvidia-rtx-4090-at-llm-perform">AMD Ryzen AI Max+ CPU 在 LLM 工作负载中击败 Nvidia RTX 4090</a>: AMD 扩展了其针对 AI 工作负载的移动芯片选择，由一款比独立显卡更强大的笔记本 CPU 领衔。</li><li><a href="https://www.youtube.com/watch?v=WVTuU-Bu7OE">来自 Duolingo CEO 的消息</a>: Duo 在地球上的连胜纪录已终结。请以今天完成一课的方式来悼念他，无需鲜花。</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>: 以下是我们所有 Notebook 的列表：</li><li><a href="https://www.youtube.com/watch?v=v7HUud7IvAo">AMD CPU, Apple M4 Pro 性能 - Ryzen AI MAX 评测</a>: Ryzen AI MAX+ 395 和 Ryzen AI MAX 390 被认为是 Apple M4 和 Apple M4 Pro 的竞争对手，结合了高效率和一些相当疯狂的性能...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1341829906580902029)** (358 条消息🔥🔥): 

> `Grok 3 性能、Aider 集成挑战、Elon Musk 对 AI 的影响、DeepSeek-R1 对比、AI 模型成本效率` 


- **Grok 3：AI 新宠**：用户称赞 Grok 3 比 GPT-4o 更快，并提供了有效的“Think”模式，许多人正考虑为此取消其他订阅。
   - 一位用户因其性能、更低的价格和更讨喜的 UI 而称其为“死党”。
- **Aider 在大型仓库中的局限性**：由于 Linux 参数大小限制，用户表示难以将大量文件传递给 Aider，建议使用带有 `/load` 命令的文本文件。
   - 他们指出，虽然他们的仓库包含许多小文件，但嵌套目录路径的长度是一个重大问题。
- **Elon Musk 对 AI 认知的冲击**：Elon Musk 仍然是一个有争议的人物，一些人对他对 AI 的贡献表示钦佩，而另一些人则批评他的商业行为。
   - 对话揭示了对 Musk 的矛盾情感，讨论中交织着幽默。
- **DeepSeek-R1 vs OpenAI 模型**：SambaNova 宣布了提供 DeepSeek-R1 服务的效率，与市场上现有模型相比，速度显著提升且成本大幅降低。
   - 该更新声称能为 DeepSeek-R1 提供最高效率，在 AI 模型应用和实施方面取得了重大进展。
- **AI 模型的成本担忧**：讨论强调了与各种 AI 模型（特别是 Sonnet 和 Grok 3）相关的成本，用户正在反思它们的价值。
   - 人们对 AI 服务免费提供的可持续性表示担忧，以及用户是否会迁移到具有更明确成本效益的模型。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://grok.com/">Grok</a>: Grok 是由 xAI 设计的免费 AI 助手，旨在最大限度地提高真实性和客观性。Grok 提供实时搜索、图像生成、趋势分析等功能。</li><li><a href="https://sambanova.ai/press/fastest-deepseek-r1-671b-with-highest-efficiency">SambaNova 发布最高效、最快的 DeepSeek-R1 671B</a>: SambaNova 宣布 DeepSeek-R1 671B 今日在 SambaNova Cloud 上以每秒 198 个 tokens 的速度运行——这是其他平台无法比拟的速度和效率。</li><li><a href="https://blog.jetbrains.com/kotlin/2025/02/openai-vs-deepseek-which-ai-understands-kotlin-better/">OpenAI vs. DeepSeek：哪个 AI 更懂 Kotlin？ | Kotlin 博客</a>: 哪个 AI 模型最懂 Kotlin？我们使用 Kotlin 特定的基准测试了 DeepSeek-R1、多个 OpenAI 模型等。在我们的分析中查看它们的对比。</li><li><a href="https://tenor.com/view/down-syndrome-gif-9029652995864711868">Down Syndrome GIF - 唐氏综合征 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/Yuchenj_UW/status/1892634804786757712">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>: 我终于可以说了：Grok 3 是我的死党。- 比 GPT-4o 快得多 - “Think”模式与下面的提示指南完美配合 - 更便宜 - 我更喜欢他们的 UI 而不是 ChatGPT 和 Claude (我是个...)</li><li><a href="https://tenor.com/view/elon-musk-smoke-smoking-well-maybe-gif-12516944">Elon Musk Smoke GIF - Elon Musk 抽烟 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/burgerkingguy-gif-21201954">Burgerkingguy GIF - 汉堡王家伙 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/yacquub-lexhinds-gif-19320537">Yacquub Lexhinds GIF - Yacquub Lexhinds - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/xai/status/1892400129719611567">来自 xAI (@xai) 的推文</a>: 就是这个：世界上最聪明的 AI，Grok 3，现在免费提供（直到我们的服务器熔化）。现在尝试 Grok 3：https://x.com/i/grok。X Premium+ 和 SuperGrok 用户将拥有更多访问 Grok 3 的权限...</li><li><a href="https://x.com/elonmusk/status/1892452789042757709">来自 Elon Musk (@elonmusk) 的推文</a>: 这还不包括语音模式和未来几天推出的许多其他功能</li><li><a href="https://www.reddit.com/r/singularity/comments/1itoi3f/grok3_thinking_had_to_take_64_answers_per/">Reddit - 深入探索</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1341930831521779736)** (20 messages🔥): 

> `Aider 中的模型配置、Editor 与 Architect 模式、Aider 中的字体颜色更改、使用本地模型、NPM 包管理` 


- **模型配置挑战**：一位用户分享了在为 Architect 模式配置 `.aider.conf` 回退模型时遇到的问题，在提供模型列表时遇到了错误。
   - 他们还提到在不退出 Architect 模式的情况下更改模型存在困难，并寻求关于正确配置的帮助。
- **字体颜色可见性问题**：用户对 Aider 中字体颜色的可见性表示担忧，指出蓝色在浅色模式（light mode）下难以看清。
   - 建议包括检查深色模式（dark mode）设置，并确保配置正确以解决可见性问题。
- **在 Editor 和 Architect 模式之间切换**：一位用户询问如何在 Aider 的 Editor 和 Architect 模式之间切换，并对系统默认进入 Architect 模式表示沮丧。
   - 另一位成员建议使用 `--edit-format` 选项，根据用户需求控制使用哪种格式。
- **Aider 中的本地模型加载**：讨论了关于本地模型运行缓慢的问题，用户寻求一种在整个 Aider 会话期间将模型保持加载在 RAM 中的方法。
   - 这指向了性能问题，因为每个 Prompt 都要重复加载会导致延迟。
- **在 Aider 中管理 Git 仓库**：用户报告了在 Aider 会话处于活动状态时切换分支的问题，在 Git 仓库中遇到了坏对象（bad object）错误。
   - 提议包括增加一个 `/drop-stale` 命令，用于自动从添加状态中清理不存在的文件，以简化工作流。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/usage/conventions.html">指定编码规范</a>：让 Aider 在处理代码时遵循你的编码规范。</li><li><a href="https://aider.chat/docs/more/edit-formats.html">编辑格式</a>：Aider 使用各种“编辑格式”让 LLM 编辑源文件。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1341921412478144532)** (2 messages): 

> `构建过程缓慢、RAG 与 AI Chat 性能对比、索引成本` 


- **构建过程滞后**：构建过程明显缓慢，使用“build”方法重写 Chunks，导致文件索引延迟，并在 API 调用上产生巨大开销，特别是针对 **Chunks** 和 **Tokens**。
   - *令人沮丧的是，我的索引从昨天开始一直在进行，但与此同时我仍可以继续使用系统。*
- **RAG 表现优于 AI Chat**：一位成员表示，对于他们的编码需求，目前的 **RAG** 设置比 **AI Chat** 的 RAG 功能产生更好的结果。
   - 另一位成员表示赞同，指出*普通的 RAG 在处理代码时比较吃力*，有必要进行改进。
- **批量成本效率建议**：有建议提出通过允许整合来自供应商的批量处理（batching）成本来提高系统效率，从而可能降低整体成本。
   - 这一改进可以解决目前与长时间索引操作（如正在经历的这些）相关的高昂费用问题。


  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1341816667176570931)** (354 条消息🔥🔥): 

> `Cursor IDE 更新、Grok 3 性能、Sonnet 3.5 问题、MCP server 功能、AI 模型讨论` 


- **Cursor IDE 更新引发争论**：多位用户报告了 Cursor 中 Sonnet 3.5 的性能较之前版本有所下降，对其可靠性和功能表示失望。
   - 相比之下，Grok 3 在编程任务中的解决问题速度和有效性获得了赞赏。
- **Grok 3 评价褒贬不一**：虽然一些用户支持 Grok 3，称其在编程任务中表现出色，但其他人对其所有者及过往表现持批评态度。
   - 讨论内容包括是否应在 Cursor 中集成 Grok 3 的不同意见，并强调了其目前缺乏 API 访问权限的问题。
- **MCP server 造成困惑**：用户讨论了在 Cursor 中设置和使用 MCP server 的复杂性，部分用户发现难以有效利用。
   - 社区成员建议，改进文档可以提升用户体验并简化安装流程。
- **AI 性能受到审视**：多位参与者对当前 AI 模型（尤其是 Claude）的表现表示不满，将输出的不一致性归因于底层的 prompting 和上下文管理问题。
   - 有人指出，LLM 的响应存在差异是预料之中的，这突显了这些模型的随机性（stochastic nature）。
- **开发者对 Cursor 工具的挫败感**：用户报告在使用 Cursor Tab 时遇到挑战，称其在开发过程中引入了 Bug，导致工作流变慢。
   - 相比之下，Cursor Composer 因能生成更强大、更可靠的代码而受到称赞。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.cursor.com/settings/models">Cursor – Models</a>：未找到描述</li><li><a href="https://browsertools.agentdesk.ai/installation">Installation - AgentDesk - BrowserToolsMCP</a>：未找到描述</li><li><a href="https://smithery.ai/server/@daniel-lxs/mcp-perplexity">Perplexity Chat MCP Server | Smithery</a>：未找到描述</li><li><a href="https://gist.github.com/grahama1970/98c5cd8bc4e266fd7b3ebad36e6823eb">该 README 概述了 Cursor MCP 环境的局限性，强调了对包访问和环境变量的限制。关键问题包括对 Python 标准库的依赖以及对敏感数据进行硬编码的需求。解决方法涉及相应地调整脚本。开放性问题集中在改进安全性和访问权限的潜在配置上。</a>：该 README 概述了 Cursor MCP 环境的局限性，强调了对包访问和环境变量的限制。关键问题包括对 Python 标准库的依赖……</li><li><a href="https://www.semafor.com/article/12/03/2024/amazon-announces-new-rainier-ai-compute-cluster-with-anthropic">亚马逊宣布与 Anthropic 合作推出新的 “Rainier” AI 计算集群</a>：该多地数据中心将用于训练下一代 Anthropic 模型。</li><li><a href="https://x.com/windsurf_ai/status/1892322088507105561?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 Windsurf (@windsurf_ai) 的推文</a>：DeepSeek-V3 现在在 Windsurf Pro 和 Ultimate 计划中不限量使用。0 prompt 额度消耗。0 flow action 额度消耗。</li><li><a href="https://youtu.be/WVpaBTqm-Zo">Grok 3 是一个……有趣的模型。</a>：我对 Grok 3 寄予厚望。根据他们的基准测试，它应该是新的最强模型，对吧？对吧？关于这个模型有很多值得讨论的地方……谢谢……</li><li><a href="https://github.com/anaisbetts/mcp-installer/issues/9">你的 MCP 配置是针对 OSX 和 Linux 的 · Issue #9 · anaisbetts/mcp-installer</a>：为了让它在 Windows 上运行，配置需要像这样 { &quot;mcpServers&quot;: { &quot;mcp-installer&quot;: { &quot;command&quot;: &quot;cmd.exe&quot;, &quot;args&quot;: [ &quot;/c&...</li><li><a href="https://github.com/AgentDeskAI/browser-tools-mcp/issues/5">无论我添加多少次 MCP 服务，它都不生效，即使多次重启 Cursor 也是如此。 · Issue #5 · AgentDeskAI/browser-tools-mcp</a>：另一方面，BrowserTools Server 运行正常。</li><li><a href="https://downloader.cursor.sh/versions/0.45.14/mac/zip/arm64">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1341838930718232699)** (79 messages🔥🔥): 

> `Hugging Face 精装书发布, Qwen2.5 训练优化, HF Spaces 上的视频生成器, 编程模型讨论, Spark Engine Discord 社区` 


- **Hugging Face 精装书即将发布！**: 成员们对新款 Hugging Face 主题精装书表示期待，并重点推荐了一篇 [博客文章](https://x.com/Nouamanetazi/status/1892274582503248178)，庆祝团队一年的工作成果。
   - *如果你感兴趣，请尽快点击以获取副本！*
- **Qwen2.5 训练新算法**: 一名成员宣布，使用 Unsloth 的新算法，用户仅需 **5GB VRAM** 即可训练 Qwen2.5 推理模型，实现 **10 倍的上下文长度**，且 **VRAM 占用减少 90%**。
   - 他们分享了强调这些改进的 [博客链接](https://unsloth.ai/blog/grpo)，并鼓励用户充分利用。
- **对 HF Spaces 视频生成器的兴趣**: 讨论围绕 HF Spaces 上视频生成器的可用性展开，一名成员指出 *ltxv* 速度相当快，在现有平台上生成视频仅需 *10-15 秒*。
   - 另一名成员表示有兴趣合作，基于最新发布的成果创建一个视频生成器。
- **最佳编程模型对比**: 成员们辩论了用于开发的最佳编程模型，推荐了各种开源和闭源模型，其中 *claude* 因其静态页面生成能力而受到关注。
   - 讨论显示，与专有模型相比，由于更好的控制权和用户自由度，用户更倾向于使用 Hugging Chat。
- **加入 Spark Engine Discord**: sparkjordi 向成员们介绍了 Spark Engine 并分享了其 Discord 社区链接，获得了积极响应。
   - 据透露，sparkjordi 在启动 Spark Engine 项目中发挥了作用，进一步激发了大家对该平台的兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1892640995847901684">来自 Unsloth AI (@UnslothAI) 的推文</a>: 今天，我们推出了新算法，可实现 10 倍的上下文长度并减少 90% 的 VRAM 用于训练推理模型 (GRPO)。使用 Unsloth，你现在只需 5G 即可训练自己的推理模型...</li><li><a href="https://sparkengine.ai">Spark Engine - AI 沙盒</a>: 将创意转化为 AI 驱动的产品，无需编程经验</li><li><a href="https://x.com/Nouamanetazi/status/1892274582503248178">来自 Nouamane Tazi (@Nouamanetazi) 的推文</a>: 🚀 很高兴发布 *THE* Ultra-Scale Playbook —— 一份关于在 1 到 1000 颗 GPU 上训练 LLM 的全面指南！</li><li><a href="https://github.com/huggingface/huggingface_hub/issues">huggingface/huggingface_hub</a>: Huggingface Hub 的官方 Python 客户端。 - huggingface/huggingface_hub</li><li><a href="https://github.com/huggingface/datasets/pull/6968">由 Wauplin 提交的 Pull Request #6968：使用 `HF_HUB_OFFLINE` 代替 `HF_DATASETS_OFFLINE`</a>: 要离线使用数据集，可以使用 HF_DATASETS_OFFLINE 环境变量。此 PR 使 HF_HUB_OFFLINE 成为离线训练的推荐环境变量。目标是更加一致...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1341832837866983545)** (2 messages): 

> `量子计算, Majorana 1, Satya Nadella 的创新` 


- **微软 Majorana 1 引领量子计算攻势**: 微软推出了 **Majorana 1**，这是一款量子芯片，有可能在几分钟内解决当前超级计算机需要 **数十亿年** 才能解决的问题。
   - 这一突破是在经历了近 **20 年的研究** 后取得的，被视为 **量子计算 (Quantum Computing)** 领域的重大里程碑。
- **Satya Nadella 揭示量子创新**: 配合微软的公告，Satya Nadella 分享了他对 **量子计算领域** 最新努力的见解。
   - 这引发了关于量子技术在各行业影响的兴奋和讨论。



**提及的链接**: <a href="https://kuberwastaken.github.io/blog/Technology/Majorana-1---Why-Quantum-Computing-Matters-Now">Majorana 1 - 为什么量子计算现在至关重要</a>: 简介：计算的新纪元。想象一台如此强大的计算机，它可以在几分钟内解决当今最快的超级计算机需要数十亿年才能解决的问题...

  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1342086993924063297)** (3 条消息): 

> `Zurich 14B Model, Hugging Face Spaces` 


- **对 Zurich 14B 模型的兴奋**：成员们对发现 **Zurich 14B 模型**表现出极大的热情，该模型通过 [Hugging Face collection](https://huggingface.co/collections/rubenroy/zurich-14b-679b57329ebbbc09ab6f03d4) 分享。
   - 一位成员评论说它实际上非常**惊人 (insane)**，强调了该模型令人印象深刻的能力。
- **介绍 Zurich 14B Chat 功能**：展示 Zurich 14B 模型的 **HF Space** 允许用户进行交互式聊天体验，可用时间为 **5 分钟**。
   - 讨论中提到了*火箭表情符号*以及对使用此类 Space 的兴奋之情。



**提到的链接**：<a href="https://huggingface.co/collections/rubenroy/zurich-14b-679b57329ebbbc09ab6f03d4">Zurich 14B - a rubenroy Collection</a>：未找到描述

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1341851680827904141)** (10 条消息🔥): 

> `CommentRescueAI, Aster audio search app, ASR dataset for Ukrainian, docSmith documentation generator, NotAnAI.ai` 


- **CommentRescueAI 简化 Python 文档编写**：一位成员介绍了 **CommentRescueAI**，这是一个只需点击一下即可为 Python 代码添加 AI 生成的 docstrings 和注释的工具。它现在已在 VS Code 扩展市场上线，并邀请用户提供反馈。
   - 开发者对社区关于改进建议的投入表示热切期待。
- **Aster 应用探索使用 HF 模型的音频搜索**：一位成员分享了一篇博文，详细介绍了 **Aster** 应用，这是一个利用 HF Laion **CLAP** 模型的免费音频搜索工具。讨论包括了 ONNX 和 PyTorch 之间的性能比较，强调需要批处理 (batching) 支持以提高效率。
   - 正在寻求社区对该应用功能和性能的反馈，以增强其能力。
- **乌克兰语的干净 ASR 数据集**：一位成员宣布发布了一个经过清洗的乌克兰语 **ASR 数据集**，旨在纠正之前标签不可靠的问题。该数据集旨在促进 ASR 模型的可靠测试，并通过人工验证确保准确性。
   - 鼓励社区成员分享和推广该数据集，以扩大其影响力和实用性。
- **docSmith 生成结构化文档**：分享了 **docSmith** 的发布，这是一个 AI 驱动的文档生成器，它使用 **Gemini** 语言模型直接从 GitHub 仓库创建结构化文档。它专为开发者、撰稿人和项目经理设计，旨在简化文档编写过程。
   - 用户可以在[这里](https://github.com/Jai0401/docSmith)探索该项目及其功能。
- **NotAnAI 提供交互式 AI 体验**：一位成员介绍了 **NotAnAI**，这是一个 AI 驱动的 Discord 机器人和网站，通过各种问题提供交互式体验。其底层技术利用 **Qwen2.5-Coder** 模型实现对话能力。
   - 分享了机器人和网站的链接，邀请用户尝试所提供的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://asteraudio.app/blog/whats-it-for">Aster</a>：未找到描述</li><li><a href="https://asteraudio.app/blog/webgpu-wasm-cuda">Aster</a>：未找到描述</li><li><a href="https://www.kaggle.com/code/allanwandia/secondary-structure-data-analysis">Secondary Structure Data - Analysis</a>：使用 Kaggle Notebooks 探索和运行机器学习代码 | 使用来自多个数据源的数据</li><li><a href="https://github.com/Jai0401/docSmith">GitHub - Jai0401/docSmith: docSmith is an AI-powered codebase documentation generator for analyzing codebases and producing structured docs. Supports GitHub repos &amp; local files. Perfect for developers, writers, &amp; project managers.</a>：docSmith 是一个 AI 驱动的代码库文档生成器，用于分析代码库并生成结构化文档。支持 GitHub 仓库和本地文件。非常适合开发者、撰稿人和项目经理。</li><li><a href="https://doc-smith.vercel.app/">docSmith</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Yehor/cv10-uk-testset-clean">Yehor/cv10-uk-testset-clean · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://notanai.xyz">NotAnAi</a>：绝对不是一个 AI - k/wom.p.womp</li><li><a href="https://not-an-ai.vercel.app)">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct">Qwen/Qwen2.5-Coder-32B-Instruct · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1342135559811764236)** (1 条消息): 

> `关于 LLMs、代码与认知的 Substack` 


- **LLMs 主题 Substack 发布**：一个新的 [Substack](https://open.substack.com/pub/codeandcognition/p/unlocking-lightning-fast-llms-the?r=1u0tss) 已经上线，专注于提供关于 **LLMs** 和 AI 的易于理解的内容。
   - 创建者邀请大家提供反馈，并强调其目标是分享该领域的实用见解和创新。
- **探索 AI 创新**：这个名为 **Code & Cognition** 的 Substack 通过深度探讨和实用见解，探索 **AI**、机器学习和软件工程的最新动态。
   - 该专栏于一周前启动，旨在提供该领域的**前沿创新**。



**提到的链接**：<a href="https://open.substack.com/pub/codeandcognition/p/unlocking-lightning-fast-llms-the?r=1u0tss&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false">Unlocking Lightning Fast LLMs: The Power of KV Caching</a>：你是否好奇 AI 聊天机器人为何能在后台运行大规模语言模型的情况下几乎瞬间做出响应？秘密就在于一种名为 KV caching 的强大优化技术。

  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1342023553221918761)** (1 条消息): 

> `Lumina2 微调，LoRA 实现` 


- **Lumina2 新微调脚本发布**：一个支持 **LoRA** 的 **Lumina2** 新微调脚本已经发布，增强了用户的功能体验。
   - 开发者可以在[此处文档](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_lumina2.md)中查看详情。
- **庆祝 Apache2.0 许可证**：新功能采用 **Apache2.0** 许可证，促进了开放性和可访问性。
   - 这与社区致力于分享 AI 技术创新的承诺相一致。



**提到的链接**：<a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_lumina2.md">diffusers/examples/dreambooth/README_lumina2.md at main · huggingface/diffusers</a>：🤗 Diffusers：在 PyTorch 和 FLAX 中用于图像、视频和音频生成的先进扩散模型。 - huggingface/diffusers

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1342035402890154037)** (5 条消息): 

> `摘要与图表量化、NLP 学习资源、聊天模型微调、编码理论中的模运算` 


- **基于 SQL 上下文评估摘要**：一名成员询问了如何量化摘要并根据 SQL 查询和上下文生成图表的方法。
   - 他们表示有兴趣利用 LLM 作为评委（LLM as a judge）来进行评估，并寻求进一步的操作指导。
- **寻求全面的 NLP 资料**：一名成员请求推荐从基础到高级主题的完整 NLP 资源。
   - 另一名成员建议将 **HuggingFace 的 NLP 课程**作为潜在资源。
- **聊天模型微调中的挑战**：一位用户分享了他们在包含 **100,000** 个样本的数据集上对聊天模型进行一个 epoch 微调的经验，但遇到了模型输出用户输入内容的问题。
   - 他们寻求社区帮助以解决遇到的推理问题。
- **模运算解题技巧**：有人提出了关于如何高效手动解决涉及幂和各种模类型的模运算问题。
   - 这在编码理论和密码学背景下具有相关性，凸显了对数学方法的兴趣。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1341838147394207785)** (4 条消息): 

> `HF Learn 课程实现，课程新单元` 


- **HF Learn 课程变得更具互动性**：正如 [Discord 消息](https://discord.com/channels/879548962464493619/1313889336907010110/1341067833479794780)中所述，一名成员目前正致力于在 **HF Learn** 上实现该课程，使其更易于访问且更具互动性。
   - 这一努力旨在通过整合更多互动元素来提升整体学习体验。
- **计划增加新单元**：另一名成员表达了向课程添加新单元的意图，表明开发和更新正在持续进行。
   - 此次更新旨在扩大课程内容，提高其相关性和实用性。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1341817572345122886)** (238 条消息🔥🔥): 

> `Unit 2.1 发布状态、访问 Hugging Face 模型、Dummy Agent 库故障排除、团队成员介绍、关于课程形式的问题` 


- **Unit 2.1 发布状态**：几位用户对 Unit 2.1 的可用性表示困惑，一些人确认该单元尚未发布。
   - 一位用户提到他们看到了一个奖励章节，但不确定 Unit 2.1 的状态，引发了关于等待更新的讨论。
- **访问 Hugging Face 模型**：用户分享了如何在 Hugging Face 上创建 token 并申请 Meta Llama 模型访问权限的见解，并引导他人前往相关的设置页面。
   - 有人指出模型需要特定的推理权限，强调了明确访问级别的必要性。
- **Dummy Agent 库故障排除**：一位用户在测试 Dummy Agent 库时遇到错误，并建议通过将模型更改为镜像链接来解决。
   - 其他人也参与进来，讨论了提供替代 API 以及使用错误处理技术来实现模型 fallback 选项。
- **介绍团队成员**：来自数据科学、工程和机器学习等不同背景的用户进行了各种自我介绍，表达了对课程的热情。
   - 参与者表现出合作和建立联系的热情，强调了互助的社区氛围。
- **关于课程形式的问题**：一位用户质疑自己是因为课程格式不佳还是个人理解问题而感到吃力，这反映了关于学习挑战的普遍情绪。
   - 这引发了关于课程清晰度和易用性的讨论，表明了对改进所提供材料结构的渴望。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sparkengine.ai">Spark Engine - AI 沙盒</a>：将创意转化为 AI 驱动的产品，无需编程经验</li><li><a href="https://huggingface.co/learn/agents-course/bonus-unit1/introduction">简介 - Hugging Face Agents 课程</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/agents-course/First_agent">First Agent - agents-course 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://agents-course-unit-1-quiz.hf.space/">agents-course/unit_1_quiz 的数据集测验</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/sebasArTecnology/First_agent_template">First Agent Template - sebasArTecnology 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit0/introduction">欢迎来到 🤗 AI Agents 课程 - Hugging Face Agents 课程</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/agents-course/First_agent_template">First Agent Template - agents-course 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/dummy-agent-library">Dummy Agent 库 - Hugging Face Agents 课程</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/vitpolis/First_agent_template">First Agent Template - vitpolis 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/settings/tokens.">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct">meta-llama/Llama-3.2-3B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=Z3yQHYNXPws">Helix 介绍</a>：我们正在推出 Helix，这是一个通用的视觉-语言-动作 (VLA) 模型，它统一了感知、语言理解和学习控制，以克服多...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1341818939243495434)** (243 条消息🔥🔥): 

> `Perplexity AI 使用问题、Grok 3 性能对比、Deep Research 功能、O3 和 O3 Mini 模型、API 集成与功能`

- **Perplexity AI 面临使用问题**：用户报告了在 Perplexity 应用中令人沮丧的体验，包括延迟、资源消耗以及文本生成过程中的故障。
   - 针对模型的性能提出了担忧，引发了关于开发团队是否正在解决这些持续性问题的询问。
- **Grok 3 的能力受到审视**：围绕 Grok 3 的讨论反映了复杂的情绪，一些用户认为其表现优于之前的模型，而另一些用户则注意到了显著的幻觉（hallucinatory）行为。
   - 用户将 Grok 3 与 Claude 和 O3 的组合进行了比较，更倾向于使用 Claude 以获得更可靠的性能。
- **Deep Research 的性能评估**：Deep Research 的有效性引发了辩论，用户注意到自 R1 1776 实施以来有所改进，但幻觉输出仍是一个问题。
   - 一位用户表示，Deep Research 和 ChatGPT 在检索当地历史犯罪数据方面都证明是有用的，展示了它们超越当地新闻的能力。
- **关于 O3 和 O3 Mini 模型的澄清**：用户澄清说，虽然 O3 是一个完整模型，但并不容易获取，目前仅 O3 Mini 可供一般使用。
   - 大家达成共识，认为 O3 Mini 有效地保留了完整模型的能力，但在计算能力和可访问性方面存在限制。
- **API 集成和用户支持**：用户正在探索通过 API 集成 Perplexity 模型，旨在无需大量编码即可高效构建 AI 工具。
   - 用户对支持响应时间和 API 使用的定价模型表示担忧，并讨论了潜在的变通方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/vectara/Hallucination-evaluation-leaderboard">Hallucination Evaluation Leaderboard - a Hugging Face Space by vectara</a>：未找到描述</li><li><a href="https://sparkengine.ai">Spark Engine - The AI Sandbox</a>：将创意转化为 AI 驱动的产品，无需编程经验</li><li><a href="https://chromewebstore.google.com/detail/complexity-perplexity-ai/ffppm">Chrome Web Store</a>：为您的浏览器添加新功能并个性化您的浏览体验。</li><li><a href="https://openrouter.ai/perplexity/r1-1776">R1 1776 - API, Providers, Stats</a>：注意：由于此模型不返回 &lt;think&gt; 标签，思考过程默认将直接流式传输到 `content` 字段。R1 1776 是 DeepSeek-R1 的一个版本，经过后期训练以移除...</li><li><a href="https://en.wikipedia.org/wiki/OpenAI_o3">OpenAI o3 - Wikipedia</a>：未找到描述</li><li><a href="https://x.com/naivigator/status/1892658960496230880">Tweet from Navigator (@naivigator)</a>：🧵 介绍 Navigator —— 您的全能 DeFai AI Agent、启动平台和用于自动化浏览器任务的框架！🚀</li><li><a href="https://www.cplx.app/">Complexity</a>：每个人都想要的 Perplexity.ai 增强版。</li><li><a href="https://x.com/CryptoEternalAI/status/1892490182479192287?s=46">Tweet from Eternal AI (EAI) (@CryptoEternalAI)</a>：现在任何人都可以使用 @perplexity_ai 的 R1 1776 模型启动去中心化 AI Agent，存储在 @Filecoin 上，并在 @Avax 上以去信任的方式提供服务。http://eternalai.org/avaxAvax agents powered by @CryptoEternalA...</li><li><a href="https://chromewebstore.google.com/detail/complexity-perplexity-ai/ffppmilmeaekegkpckebkeahjgmhggpj,">Chrome Web Store</a>：为您的浏览器添加新功能并个性化您的浏览体验。</li><li><a href="https://news.microsoft.com/source/features/ai/microsofts-majorana-1-chip-carves-new-path-for-quantum-computing/">Microsoft’s Majorana 1 chip carves new path for quantum computing - Source</a>：Majorana 1，首款由新型 Topological Core 架构驱动的量子芯片</li><li><a href="https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/">Microsoft unveils Majorana 1, the world’s first quantum processor powered by topological qubits - Microsoft Azure Quantum Blog</a>：来自 Microsoft 的 Majorana 1 是世界上第一款使用拓扑导体构建的量子处理器（QPU）。了解更多。</li><li><a href="https://blog.google/technology/research/google-willow-quantum-chip/">Meet Willow, our state-of-the-art quantum chip</a>：我们的新型量子芯片展示了纠错和性能，为实用的、大规模的量子计算机铺平了道路。</li><li><a href="https://scitechdaily.com/superconduction-breakthrough-scientists-discover-new-state-of-quantum-matter/">Superconduction Breakthrough: Scientists Discover New State of Quantum Matter</a>：康奈尔大学的研究人员在候选拓扑超导体中发现了一种新的物质状态，这一发现可能对凝聚态物理产生深远影响...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1341839883173626058)** (23 条消息🔥): 

> `AI 对冲基金表现优于市场，墨西哥因海湾问题向 Google 发难，双足肌肉机器人，发光蛋白质创造，神经网络分析` 


- **AI 对冲基金超越市场预期**：最近的一篇文章透露，一家 **AI Hedge Fund** 的表现显著优于市场，引起了投资者的关注。
   - 该基金利用先进算法来分析市场趋势和决策过程。
- **墨西哥就海湾准入问题向 Google 发出警告**：墨西哥采取大胆行动，就 **Google** 在海湾附近的运营发出威胁，引发了对管辖权争议的关注。
   - 这一冲突凸显了科技公司与国家监管机构之间日益紧张的关系。
- **全球首款双足肌肉机器人亮相**：随着**全球首款双足肌肉机器人**的推出，工程领域出现了突破性进展。
   - 这一创新有望以类人的敏捷性彻底改变机器人的运动和交互。
- **AI 为研究创造发光蛋白质**：**科学家**开发了一种可以创造**发光蛋白质**的 **AI**，这在各种生物研究应用中可能具有关键作用。
   - 这种蛋白质可能促进成像技术和分子生物学研究的进步。
- **探索神经网络的最新见解**：一项分析深入探讨了围绕 **神经网络** 的最新进展和讨论，强调了重大发现。
   - 该主题涵盖了人工智能领域的各种应用和未来方向。



**提及链接**：<a href="https://www.youtube.com/embed/LM6r_rSF1pU">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1341827686066032763)** (4 条消息): 

> `Deep research API，Sonar API 性能问题，模型对比` 


- **Deep research 可能很快会加入 API**：成员们正在询问将 **Deep research 能力**集成到 API 中的可能性，暗示了令人兴奋的新功能。
   - 一位用户表达了热情，感谢 Perplexity 团队在该领域持续开展的工作。
- **r1-1776 API 的结束标签问题**：一位用户报告称，**r1-1776 API** 意外返回了结束标签 `</think>`，但没有匹配的开始标签 `<think>`，这已通过 curl 得到验证。
   - 他们注意到，在使用 **sonar-reasoning 模型**时不会出现此问题，该模型会正确提供开始标签。
- **对 Sonar API 性能的担忧**：一位用户对 **Sonar API 的性能**提出担忧，认为其结果比 **llama-3.1-sonar-large-128k-online** 等旧模型更差。
   - 该用户一致发现，旧模型在获取网站信息等任务中表现更好，尽管价格相似，但对感知到的质量下降感到失望。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1341816963575582730)** (156 条消息🔥🔥): 

> `PaliGemma 2 Mix 模型，AI CUDA 工程师，ALLaM 阿拉伯语模型，Helix 机器人模型，Mercor AI 招聘`

- **PaliGemma 2 Mix：增强的能力**：新推出的 [PaliGemma 2 mix](https://developers.googleblog.com/en/introducing-paligemma-2-mix/) 模型允许直接探索各项功能，并在各种现实世界任务上进行了微调，且相比之前版本有**承诺的加速**。
   - 尽管用户对与 PaliGemma 2 的区别感到困惑，但社区成员注意到在实际应用中确实存在性能提升。
- **AI CUDA Engineer 的主张与争议**：[AI CUDA Engineer](https://pub.sakana.ai/ai-cuda-engineer/paper/) 声称比现有的 CUDA kernels 提速 10-100 倍，在用户报告性能基准测试存在差异后引发了辩论。
   - 批评者质疑加速主张的可靠性，有证据表明某些优化反而导致性能变慢。
- **ALLaM 在 AI 领域的国家级努力**：由**沙特阿拉伯**支持的 [ALLaM](https://arxiv.org/html/2407.15390v1) 专注于创建阿拉伯语语言模型，以支持阿拉伯语语言技术（Arabic Language Technologies）的生态系统。
   - 这代表了在当前地缘政治环境下，构建具有竞争力的 LLMs 的少数成功国家尝试之一。
- **Helix 在机器人领域的创新**：Figure 推出了 Helix，这是一种 **Vision-Language-Action model**，能够实现多机器人协同工作以及对类人机器人的高级控制，标志着**机器人能力**的重大进步。
   - 装备了 Helix 后，机器人可以执行复杂任务，通过遵循自然语言提示对陌生对象做出动态响应。
- **Mercor 的 AI 招聘发布**：[Mercor](https://techcrunch.com/2025/02/20/mercor-an-ai-recruiting-startup-founded-by-21-year-olds-raises-100m-at-2b-valuation/) 为其 AI 招聘平台筹集了 1 亿美元，该平台由年轻的 Thiel Fellows 创立，突显了其快速增长和估值跃升至 20 亿美元。
   - 讨论集中在 Mercor 在竞争激烈的 AI 领域中创新的营销驱动，并询问这种方法是否类似于数据标注公司。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://www.figure.ai/news/helix">Helix: A Vision-Language-Action Model for Generalist Humanoid Control</a>: Figure 的创立初衷是改变世界。</li><li><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - 由 nanotron 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2502.13595v1">MMTEB: Massive Multilingual Text Embedding Benchmark</a>: 文本嵌入通常在有限的任务集上进行评估，这些任务受限于语言、领域和任务多样性。为了解决这些局限性并提供更全面的评估...</li><li><a href="https://x.com/Alibaba_WanX/status/1892607749084643453">来自 WanX (@Alibaba_WanX) 的推文</a>: 🌟 来自 @alibaba_cloud 的大新闻！🌟 认识一下 WanX —— 我们重新定义视频生成的下一代 AI 模型！🚀 展示来自 WanX 2.1 令人惊叹的演示！🔥 更令人兴奋的是：WanX 2.1 将会开源！Com...</li><li><a href="https://developers.googleblog.com/en/introducing-paligemma-2-mix/">介绍 PaliGemma 2 mix：一个用于多任务的视觉-语言模型</a>: 未找到描述</li><li><a href="https://x.com/mervenoyann/status/1892289763954069720">来自 merve (@mervenoyann) 的推文</a>: @skalskip92 @onuralpszr 这是混合迁移，但这次模型接受开放式输入，而不是结构化的任务前缀 🥹</li><li><a href="https://x.com/main_horse/status/1892489065746088217">来自 main (@main_horse) 的推文</a>: @RobertTLange 你第一个 notebook 中的 kernel 坏了。请参阅 https://x.com/main_horse/status/1892473238036631908 引用 main (@main_horse) @miru_why 我相信这其中有些问题...</li><li><a href="https://x.com/swyx/status/1892668077768106424?s=46">来自 swyx 🗽 NYC (@aiDotEngineer) (@swyx) 的推文</a>: 第一次看到 @AnthropicAI 这样列出其首要任务，现在比 Claude 4 更关注 mechinterp 了！来自 @ambricken 和 Joe Bayley 的精彩演讲！引用 swyx 🗽 NYC (@aiDotEnginee...</li><li><a href="https://fxtwitter.com/bingxu_/status/1892405811596710392?s=61">来自 Bing Xu (@bingxu_) 的推文</a>: 我在手机上快速浏览了他们的报告，有几个误导性的部分：1. Torch C++ 代码不是 CUDA kernel，它在底层调用的是 CUDNN。2. 强调的例子 Conv3D GroupNorm, conv...</li><li><a href="https://x.com/mervenoyann/status/1892576290181382153">来自 merve (@mervenoyann) 的推文</a>: 我们刚刚发布了 SmolVLM2：世界上最小的视频模型，包含 256M、500M 和 2.2B 版本 ⏯️🤗 我们还发布了以下内容 🔥 > 一个 iPhone 应用（在 MLX 中运行 500M 模型）> 与 VLC 集成用于分割...</li><li><a href="https://x.com/main_horse/status/1892474049114108138">来自 main (@main_horse) 的推文</a>: @miru_why 为了消除所有疑虑：你可以解压你提供给我的链接，并应用以下 diff 来证明 kernel 是坏的。</li><li><a href="https://x.com/btibor91/status/1892290734650433980">来自 Tibor Blaho (@btibor91) 的推文</a>: Claude Web 应用更新 —— 看起来网页搜索和 Paprika 模式（新的思考模型）仍在开发中，过去 24 小时内部署了多个新版本。这包括一个新的实验，...</li><li><a href="https://x.com/owl_posting/status/1892317797172015210">来自 owl (@owl_posting) 的推文</a>: 笑死，Greg Brockman 在 Evo 论文中的所属单位是“独立研究员”</li><li><a href="https://x.com/RobertTLange/status/1892489402070220989">来自 Robert Lange (@RobertTLange) 的推文</a>: 你好！感谢你对我们项目的关注。我们的加速估算是基于 H100 获得的。我们已经在另外 3 个 GPU 上确认了结果，并在这里分享相应的加速数据和 Colab 链接...</li><li><a href="https://x.com/tomwarren/status/1892620459062988911">来自 Tom Warren (@tomwarren) 的推文</a>: 独家消息：微软正在为 OpenAI 的 GPT-5 模型做准备，GPT-4.5 最快可能在下周发布。所有这些以及更多内容都在本周的 📒 Notepad 期刊中，订阅者现已可阅 👇 ht...</li><li><a href="https://arxiv.org/html/2407.15390v1">ALLaM: Large Language Models for Arabic and English</a>: 未找到描述</li><li><a href="https://x.com/klarnaseb/status/1892262217568891179">来自 Sebastian Siemiatkowski (@klarnaseb) 的推文</a>: @GergelyOrosz 我们没有改变方向。我们正在进一步开发它。今天我们的 AI 聊天机器人处理的问题比你测试时更复杂，质量也更高。但与此同时...</li><li><a href="https://x.com/main_horse/status/1892473238036631908">来自 main (@main_horse) 的推文</a>: @miru_why 我相信他们的 kernel 有问题 —— 它似乎“窃取”了 eager 实现的结果（某种内存复用？），从而绕过了正确性检查。在这里，我...</li><li><a href="https://x.com/SakanaAILabs/status/1892385766510338559">来自 Sakana AI (@SakanaAILabs) 的推文</a>: 介绍 AI CUDA Engineer：一个自动生产...的 Agentic AI 系统。

高度优化的 CUDA kernel。http://sakana.ai/ai-cuda-engineer/ AI CUDA Engineer 可以生成高度优化的...</li><li><a href="https://techcrunch.com/2025/02/20/mercor-an-ai-recruiting-startup-founded-by-21-year-olds-raises-100m-at-2b-valuation/">Mercor，一家由 21 岁年轻人创办的 AI 招聘初创公司，以 20 亿美元估值筹集了 1 亿美元 | TechCrunch</a>：Mercor，这家由三位 21 岁 Thiel Fellows 创办的 AI 招聘初创公司，已在 B 轮融资中筹集了 1 亿美元，该公司已向媒体确认。</li><li><a href="https://x.com/Figure_robot/status/1892577871366939087">Figure (@Figure_robot) 的推文</a>：认识 Helix，我们内部开发的、能像人类一样推理的 AI。如果没有能力的跨越式发展，机器人将无法进入家庭。我们的机器人现在几乎可以处理任何家用物品：</li><li><a href="https://fxtwitter.com/main_horse/status/1892446384910987718">main (@main_horse) 的推文</a>：他们论文中的这个例子（https://pub.sakana.ai/static/paper.pdf#page=47）声称有 150 倍的加速，但如果你进行基准测试，实际上慢了 3 倍... 引用 Sakana AI (@SakanaAILabs) 的介绍...</li><li><a href="https://x.ai/blog/grok-3">Grok 3 Beta — 推理 Agent 的时代</a>：未找到描述</li><li><a href="https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf">Open-Reasoner-Zero/ORZ_paper.pdf at main · Open-Reasoner-Zero/Open-Reasoner-Zero</a>：Open-Reasoner-Zero 的官方仓库。通过在 GitHub 上创建账号来为 Open-Reasoner-Zero 的开发做出贡献。</li><li><a href="https://x.com/GergelyOrosz/status/1892196257608687842">Gergely Orosz (@GergelyOrosz) 的推文</a>：Klarna 曾是一家全力投入用 AI 机器人取代客户支持并大肆宣扬节省成本的公司。现在他们正在改变方向。显而易见，会有更多公司盲目地进行替换...</li><li><a href="https://fxtwitter.com/main_horse/status/1892408991327932883?s=61">main (@main_horse) 的推文</a>：@SakanaAILabs，level_1-&gt;15_Matmul_for_lower_triangular_matrices 显然有问题吧？声称左边的 kernel 比右边的代码快 152.9 倍。真的吗？</li><li><a href="https://huggingface.co/google/paligemma2-3b-mix-448#paligemma-2-results-by-model-resolution-and-size>">google/paligemma2-3b-mix-448 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1342024334172098630)** (4 条消息): 

> `Grok 3 推理，think 与 big brain 的区别，xAI vs OpenAI 的能力，关于评分的困惑` 


- **Grok 3 推理是一个 ~o1 级别的模型**：一位成员指出，如果浅蓝色部分代表 best of N 评分，那么 **Grok 3 推理** 本质上是一个 ~o1 级别的模型，这表明 OpenAI 和 xAI 之间存在 **约 9 个月的能力差距**。
   - 他们对 *think* 和 *big brain* 这两个术语的含义提出疑问，暗示模型性能指标中存在更深层次的细微差别。
- **评分误解得到澄清**：另一位成员澄清说，浅色阴影区域指的是 **cons@64 评分**，这表明大家对 *非思考型与思考型* 模型之间的区别存在误解。
   - 这导致了一个令人沮丧的时刻，讨论中使用的捂脸表情符号表达了这种困惑。
- **对当前模型混乱状态的共识**：另一条评论反映了大家对模型能力差异讨论中 **混乱现状** 的共同感受。
   - 正在进行的对话强调了在模型评估和比较方面需要更清晰的沟通。



**提到的链接**：<a href="https://x.com/nrehiew_/status/1891710589115715847">wh (@nrehiew_) 的推文</a>：如果浅蓝色部分是 best of N 评分，这意味着 Grok 3 推理本质上是一个 ~o1 级别的模型。这意味着 OpenAI 和 xAI 之间的能力差距约为 9 个月。此外，...的区别是什么...

  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1341822327041687602)** (69 messages🔥🔥): 

> `Nadella 谈 Dwarkesh、AI 竞赛、GRPO 进展、Anthropic 员工留存率、播客亮相` 


- **Nadella 与 Dwarkesh 的互动**：*Nadella 在 Dwarkesh 播客中邀请到了重量级嘉宾*，展示了他在科技领域精通媒体的策略。
   - 围绕 *CEO 如何利用播客和媒体* 来提升公众形象和影响力的讨论。
- **俄罗斯在 AI 领域的地位**：对俄罗斯 AI 能力的担忧被提出，成员们一致认为他们处于 **'GPU poor'**（算力贫困）状态，影响了其竞争力。
   - 战争应用似乎驱动了其有限的 AI 努力，暗示任何进展都与军事利益紧密相关。
- **创新的 GRPO 进展**：Unsloth 发布了一种新的 **GRPO 算法**，将 Qwen2.5 训练的 VRAM 需求降低到仅 **5GB**，标志着重大改进。
   - 该算法支持 **10 倍长的上下文长度**，提供了简化的设置，可能彻底改变模型训练效率。
- **Anthropic 的留存率**：*AnthropicAI 在主要 AI 实验室中拥有极高的员工留存率*，凸显了其职场文化。
   - 重点正在转向 *mechanistic interpretability*（机械可解释性），展示了从 Claude 4 开发转向的战略重心。
- **来自播客圈的见解**：一位成员对受邀参加 Gleb Solomin 主持的播客频道感到惊讶，并强调了其内容的吸引力。
   - 围绕行业专业人士参加 *播客亮相* 的价值展开了持续对话，平衡了趣味性与严肃讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/@podcast_solomina/videos">Gleb Solomin 的播客</a>：与思想者的深度对话。Gleb Solomin 是一位 24 岁的企业家，莫斯科国立大学毕业生（以优等成绩毕业），邀请杰出的科学家、商人和在各自领域取得成就的人士作为嘉宾...</li><li><a href="https://x.com/JustinLin610/status/1892625486284734696">Junyang Lin (@JustinLin610) 的推文</a>：@TheXeophon 是的。7 是基于 Apache 2.0 协议的</li><li><a href="https://tenor.com/view/just-house-totally-duh-gif-23663188">Just House GIF - Just House Totally - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/huybery/status/1892628963878486233">Binyuan Hui (@huybery) 的推文</a>：&lt;think&gt;…&lt;/think&gt;Binyuan 正在准备中…</li><li><a href="https://unsloth.ai/blog/grpo">长上下文 GRPO (R1 推理)</a>：DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行 Unsloth 的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://fxtwitter.com/swyx/status/1892668077768106424">swyx 🗽 NYC (@aiDotEngineer) (@swyx) 的推文</a>：第一次看到 @AnthropicAI 这样列出其首要任务，现在比 Claude 4 更关注 mechinterp！来自 @ambricken 和 Joe Bayley 的精彩演讲！引用 swyx 🗽 NYC (@aiDotEnginee...</li><li><a href="https://x.com/swyx/status/1892684773891375125">swyx 🗽 NYC (@aiDotEngineer) (@swyx) 的推文</a>：涨知识了，@AnthropicAI 在大型实验室中拥有最高的员工留存率。引用 swyx 🗽 NYC (@aiDotEngineer) (@swyx) 第一次看到 @AnthropicAI 这样列出其首要任务，现在更关注...</li><li><a href="https://x.com/colin_fraser/status/1892379172007285176">Colin Fraser (@colin_fraser) 的推文</a>：答案：0/100。它“思考”了四分钟，然后给了我五个无关的 3 位数加法答案（我承认是正确的！），且没有可下载的文件。引用 Colin Fraser (@colin_f...</li><li><a href="https://x.com/mvpatel2000/status/1892627122729988450">Mihir Patel (@mvpatel2000) 的推文</a>：小小的生活更新：我在年初加入了 Anthropic！未来将是疯狂的，我非常高兴能成为这个向善改变世界的团队的一员 😊。我也很兴奋能...</li><li><a href="https://www.youtube.com/watch?v=YXTYbr3hiFU">一场意想不到的强化学习复兴</a>：我们所处的语言模型研究时代，普遍完全相信推理和新的强化学习 (RL) 训练...</li><li><a href="https://fxtwitter.com/colin_fraser/status/1892368545884873016">Colin Fraser (@colin_fraser) 的推文</a>：你认为 OpenAI Deep Research agent 在这 100 道 4 位数加法题上的表现会如何？
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1341854725896671293)** (5 条消息): 

> `集成 AI Agent 的无用机器、中国与 Google 的 AI 研究、Claude 的现状、AIME 2025 性能对比、Grok 的开发进展` 


- **对 AI 驱动的无用机器的兴趣**：有人请求展示一个集成了 **AI Agent** 的 **无用机器**（useless machine），展现了对新颖 AI 应用的好奇心。
   - 这反映了幽默与技术融合的持续趋势，引发了对 AI 在趣味形式中应用影响的思考。
- **中国与 Google 的开放研究路径**：一位成员指出，AI 领域的领先国家/公司，即 **中国** 和 **Google**，都通过 **开放研究** 来推进 AI，并对背后的动机表示怀疑。
   - 这一评论暗示了 AI 领域关于私有化与开源进展之间持续存在的紧张关系和看法。
- **对 Claude 的担忧**：一位成员发布了“**Claude nooooo**”的消息表达担忧，暗示 **Claude** AI 出现了某些令人苦恼的更新。
   - 这展示了社区对 AI 动态的高度参与和关注。
- **关于 AIME 2025 性能的见解**：分享了关于 **Grok** 和 **OpenAI** 模型在 **AIME 2025 性能** 方面的结果汇编，并与其他模型的性能进行了对比。
   - 一条值得注意的引用强调，为了进行准确对比，查看不同训练版本的结果至关重要，特别强调了 **Grok3** 仍有高效提升的空间。
- **Grok3 开发进展揭秘**：Yuhuai (Tony) Wu 分享了 **Grok3** 背后严苛训练的见解，解释说其更大的规模影响了训练时长，但它正在迅速增强。
   - 这表明了对提升 AI 能力的持续承诺，并承诺在未来的更新中 **释放**（unleashed）更多能量。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/zetalyrae/status/1892331830939976169">来自 Fernando 🌺🌌 (@zetalyrae) 的推文</a>：Claude nooooo</li><li><a href="https://x.com/doomslide/status/1892311556991697009">来自 doomslide (@doomslide) 的推文</a>：很能说明问题的是，处于 AI 前沿的两个国家/公司都走开放研究的道路。中国有破坏旧金山顶尖机构资金的深层动机，而 Google...</li><li><a href="https://x.com/teortaxestex/status/1892471638534303946?s=46">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：很有道理。这是我从相关来源汇编的关于 Grok 和 OpenAI 模型在 AIME 2025 表现的所有结果，加上对 DeepSeek 模型和 o1 的 cons@64 推演。我认为这...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/)** (1 条消息): 

the_real_jrb: https://arxiv.org/abs/2502.13923

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1341845360271687731)** (9 messages🔥): 

> `Open Source AI Critique, Satya Nadella on AI, Microsoft Product Quality, Copilot Development, Microsoft Teams Integration` 


- **开源 AI 面临审查**：批评者断言，如果 **Linux** 在今天诞生，它将被**立法者**和危言耸听者摧毁，他们声称开源会带来威胁，这可能导致对软件的**严格控制**。
   - 讨论强调了围绕开源软件的恐惧驱动叙事，探讨了这些运动背后的财务和立法力量。
- **Satya Nadella 务实的 AI 观点**：在最近的一段 [YouTube 视频](https://youtu.be/4GLSzuYXh6w)中，**Satya Nadella** 分享了他对 AGI 的怀疑态度，同时推动经济增长和微软的**拓扑量子比特突破**。
   - 他理智的观点赢得了赞赏，这与人们对微软整体产品质量的复杂感受形成鲜明对比。
- **对微软产品性能的担忧**：成员们表达了挫败感，质疑当 **Microsoft** 的产品（如 Teams 和 Copilot）表现不佳时，为何 **Satya Nadella** 仍能获得正面评价。
   - 深刻的评论指出，虽然 **Windows** 在游戏方面表现出色，但其搜索功能明显落后于 Mac 等竞争对手。
- **Copilot 在竞争后获得更新**：讨论指出，为了应对竞争，**Copilot** 经历了多次更新，突显了其与 **Cursor** 相比被察觉到的不足。
   - 成员们反映，微软往往会忽略质量改进，直到他们在市场上感到威胁。
- **Teams 依靠集成而非质量获胜**：一位成员阐述道，虽然 **Microsoft Teams** 在 **MSFT** 生态系统中集成良好，但这并不一定反映其独立产品的质量。
   - 对话表明一种看法，即微软的主要关注点是企业客户，从而改变了“好”产品的定义。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://danieljeffries.substack.com/p/defending-open-source-ai-against">Defending Open Source AI Against the Monopolist, the Jingoist, the Doomer and the Idiot</a>：如果 Linux 在今天才刚刚起步，它将被摧毁，我们都会因此变得贫穷得多。我们不能让这种事发生在 AI 身上。</li><li><a href="https://youtu.be/4GLSzuYXh6w">Satya Nadella – Microsoft’s AGI Plan &amp; Quantum Breakthrough</a>：Satya Nadella 谈论：为什么他不相信 AGI 但相信 10% 的经济增长、微软新的拓扑量子比特突破以及游戏世界的护城河...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1341888997420433419)** (2 messages): 

> `Reasoning Tokens Behavior, User Feedback on Token Responses, Proposed Changes to Reasoning Tokens, Poll on Reasoning Token Settings` 


- **用户反馈引发推理 Token 讨论**：反馈表明，当 **max_tokens** 较低时，用户感到不满，因为这会导致没有内容返回。
   - 目前，**include_reasoning** 默认为 false，导致内容为空或返回 null 响应，这让用户感到沮丧。
- **拟议的更改旨在提高响应清晰度**：目前有两个关键提案：将 **include_reasoning** 默认设置为 true，并确保内容始终为 string，避免 null 值。
   - 这些更改旨在提供响应的一致性，确保开发者即使在推理消耗了所有 Token 的情况下也能收到可用的内容。
- **扩大社区意见投票**：已发起一项投票，以收集关于 **include_reasoning** 设置拟议更改的意见。
   - 选项范围从保持当前行为到更改默认值，目前正积极征求社区反馈。

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1342210705818128488)** (2 messages): 

> `Weaver Chrome Extension, Open Source API Tool` 


- **Weaver：多功能 Chrome 扩展**：**Weaver** Chrome 扩展支持高度可配置的选项，如 PDF 支持、与 **Supabase** 的云同步以及直接从浏览器进行 API 调用，从而提升性能。
   - 目前免费，但托管在 Vercel 的免费计划上，这意味着由于使用限制，访问可能会受到潜在限制，且没有后端数据日志记录。
- **开源翻译工具出现**：一位用户分享了他们新开发的**开源 Chrome 扩展**，允许用户将任何内容转换为他们喜欢的风格，例如翻译或摘要。
   - 该工具可通过 [GitHub](https://github.com/amirrezasalimi/aify) 获取，仅需一个 **OpenAI 兼容的 API** 即可运行。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://weaver-one.vercel.app/">Weaver</a>: 未找到描述</li><li><a href="https://x.com/amirsalimiiii/status/1892667934641692774">来自 Amirreza (@amirsalimiiii) 的推文</a>: 刚刚开发了一个强大的 Chrome 扩展！将任何内容转换为你喜欢的风格——翻译、简化、摘要，随你挑选。🔥🛠️ 完全开源且仅需一个 OpenAI 兼容的 API。查看 ...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1341817175245066240)** (209 messages🔥🔥): 

> `OpenRouter API Integration, Gemini Model Issues, DeepSeek Models Performance, API Key Generation, Vision and Reasoning Models` 


- **将 OpenRouter 集成到网站**：一位用户询问如何使用 OpenRouter 的 API Key 将聊天机器人集成到他们的 Elementor 网站中，并表示需要指导。
   - 另一位用户指出 OpenRouter 仅提供对 LLM 的访问，并建议联系开发人员寻求集成方面的帮助。
- **Gemini 2.0 模型性能问题**：用户讨论了 Gemini 2.0 Flash 模型在结构化输出方面的问题，强调了与 OpenAI 模型相比的差异。
   - 反馈表明 UI 需要更清晰地展示不同模型的能力，特别是在输入类型和错误消息方面。
- **DeepSeek 模型的性能波动**：一些用户报告称 DeepSeek 模型最初能提供高质量的回复，但随后的回复质量显著下降。
   - 讨论集中在这种行为的可能原因，以及是否有设置可以缓解回复质量下降的问题。
- **通过编程方式生成 API Key**：一位用户表示希望能够通过编程方式为自己生成 API Key，而无需访问 OpenRouter 网站。
   - 回复者确认该功能计划很快发布，有望在本周末前上线。
- **了解模型能力**：一位用户询问在 OpenRouter 上浏览时，如何识别模型是否具有 Vision、Reasoning 或 Tool Use 能力。
   - 针对 Vision 和 Reasoning 的标识进行了说明，并提出了改进界面的建议，以使这些信息更易于获取。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://poloclub.github.io/transformer-explainer/">Transformer Explainer: LLM Transformer Model Visually Explained</a>: 一个交互式可视化工具，向你展示 Transformer 模型在 GPT 等大语言模型 (LLM) 中是如何工作的。</li><li><a href="https://openrouter.ai/perplexity/r1-1776">R1 1776 - API, Providers, Stats</a>: 注意：由于此模型不返回 &lt;think&gt; 标签，思维链默认将直接流式传输到 `content` 字段。R1 1776 是 DeepSeek-R1 的一个版本，经过后期训练以移除...</li><li><a href="https://x.com/perplexity_ai/status/1892329089903841467?t=6lD3qXX2sOcKytYFI8L1kA&s=19">来自 Perplexity (@perplexity_ai) 的推文</a>: R1 1776 现在可通过 Perplexity 的 Sonar API 获取。引用 Perplexity (@perplexity_ai)：今天我们开源了 R1 1776——这是 DeepSeek R1 模型的一个版本，经过后期训练以提供...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1341824812275994695)** (196 条消息🔥🔥): 

> `Grok3 性能担忧、进化策略在训练中的应用、AI 模型的代码数据集、Agent 协作与优化、神经网络中的平衡传播 (Equilibrium Propagation)` 


- **Grok3 的基准测试与性能**：讨论围绕 Grok3 的性能及其基准测试展开，一些人声称 xAI 对其数据不够坦诚，特别是与 cons@64 使用相关的数据。
   - 成员们对 Grok3 超越当前最先进（state-of-the-art）模型表示怀疑，并分享了具体的例子作为背景。
- **探索用于训练的进化策略**：辩论了使用进化算法 (GAs) 优化神经网络的可行性，强调了由于高维性，在大规模应用时收敛速度较慢的问题。
   - 交流了可能在训练流水线中的特定组件使用 GAs 以增强模型性能的想法，并将其与传统的反向传播 (backpropagation) 进行了对比。
- **分享高质量代码数据集**：用户分享了 Hugging Face 上可用的各种代码数据集，建议它们可以用于增强现有模型。
   - 成员们反思了数据集质量的重要性，以及使用先进的推理模型（reasoning models）重构现有数据集的潜力。
- **目标优化中的 Agent 协作**：一位成员询问了关于 Agent 协作以针对某一目标优化想法的前沿研究，特别是它们如何沟通以及利用何种方法论。
   - 对话包括了关于 Agent 讨论和优化过程以实现目标结果的个人实验引用。
- **理解神经网络中的平衡传播 (Equilibrium Propagation)**：讨论了平衡传播作为训练基于能量的模型（energy-based models）时传统反向传播的替代方案，重点在于它能够将预测推向误差最小的配置。
   - 社区参与探索了平衡传播与循环反向传播（recurrent backpropagation）之间的相似性，重点关注其在演进神经网络训练技术中的潜在应用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://news.mit.edu/2025/large-language-models-reason-about-diverse-data-general-way-0219">像人脑一样，大语言模型以通用的方式对多样化数据进行推理</a>：MIT 研究人员发现，大语言模型处理不同类型的数据（如不同语言、音频输入、图像等）的方式与人类推理复杂问题的方式相似。就像人类一样，LLMs...</li><li><a href="https://arxiv.org/abs/1602.05179">Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation</a>：我们介绍了平衡传播，这是一种用于基于能量的模型的学习框架。它只涉及一种神经计算，在第一阶段（进行预测时）和...中均执行。</li><li><a href="https://arxiv.org/abs/1711.08416">Equivalence of Equilibrium Propagation and Recurrent Backpropagation</a>：循环反向传播和平衡传播是用于不动点循环神经网络的有监督学习算法，它们在第二阶段有所不同。在第一阶段，两种算法...</li><li><a href="https://arxiv.org/abs/1808.04873">Generalization of Equilibrium Propagation to Vector Field Dynamics</a>：反向传播算法的生物学合理性长期以来一直受到神经科学家的怀疑。两个主要原因是神经元需要在前向传播中发送两种不同类型的信号...</li><li><a href="https://huggingface.co/microsoft/wham">microsoft/wham · Hugging Face</a>：未找到描述</li><li><a href="https://fxtwitter.com/satyanadella/status/1892244164814725387">Satya Nadella (@satyanadella) 的推文</a>：如果你觉得 AI 生成的文本、图像和视频很酷，想象一下像游戏一样的整个交互式环境！</li><li><a href="https://steamcommunity.com/sharedfiles/filedetails/?id=3143225812&searchtext=">Steam Workshop::Reforged Eden 2 Beta</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1itdy0k/no_system_instructions_for_deepseek_makes_jake/">Reddit - 深入探讨一切</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k">NovaSky-AI/Sky-T1_data_17k · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1342230052565155962)** (1 条消息): 

> `Reinforcement Learning for LLMs, Scaling Supervision` 


- **面向初学者的 Reinforcement Learning 解释**：一位成员分享了一个 [Twitter 线程](https://x.com/ShashwatGoel7/status/1892668493390094338)，专门为 **large language models** (LLMs) 的新手解释 **Reinforcement Learning** (RL)，强调这是一种无需预备知识的方法。
   - 该线程强调 **RL** 令人兴奋，因为它能够从 **rewards**（奖励）中学习，而不是仅仅依赖于演示（demonstrations）。
- **通过 RL 扩展监督的重要性**：该线程强调 **scaling supervision**（扩展监督）是使用 **Reinforcement Learning** 的一个显著优势，因为它允许通过更简单的奖励机制进行有效学习。
   - 这种方法最终将范式从需要详细的演示转变为利用更广义的奖励反馈。



**提到的链接**: <a href="https://x.com/ShashwatGoel7/status/1892668493390094338">Shashwat Goel (@ShashwatGoel7) 的推文</a>：我根据第一性原理整理了这个无需 RL 预备知识的解释器，介绍了 LLM 的 RL 如何工作以及我们为什么需要它🧵核心观点？RL 令人兴奋，因为它允许我们扩展监督。我们现在可以……

  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1341818699895799808)** (75 条消息🔥🔥): 

> `Transformer Backpropagation, Logit vs Probability in Decision Making, Evolutionary Strategies for LLMs, LoRA vs Full Fine-Tuning, Reinforcement Learning for LLMs` 


- **Transformer 反向传播的难题**：一位用户表达了在 Transformer 中实现反向传播（backpropagation）的困惑，特别是在处理并行操作和注意力机制（attention mechanisms）方面。
   - 其他人建议关注单个注意力组件，并参考 unsloth triton kernels 等资源。
- **Logits 比概率包含更多信息**：讨论集中在 Logits 比归一化概率更具表达能力的观点上，同时认为不必要的归一化可能会阻碍优化过程。
   - 有观点认为，虽然决策需要概率，但在 Logit 空间中工作可以提高某些模型的训练效率。
- **Low-Rank Adaptation (LoRA) 的局限性**：参与者讨论了 LoRA 为何不等同于全量微调（full fine-tuning），因为其低维更新可能会限制对新数据的准确拟合。
   - 有人认为，虽然低秩 LoRA 难以保持分布外（out-of-distribution）数据的不变性，但高秩 LoRA 虽接近全量微调，却降低了效率。
- **对进化策略的担忧**：一位用户询问进化策略（Evolutionary Strategies, ES）在低维学习框架中是否会遇到与 LoRA 类似的限制，并指出了突变噪声（mutation noise）的潜在问题。
   - 回复指出，虽然 ES 可能不会面临与 LoRA 相同的挑战，但如果突变噪声过强，仍可能遇到问题。
- **对强化学习的新认识**：一位用户分享了他们整理的关于 LLM 强化学习的入门解释，强调了其增强监督扩展的能力。
   - 该解释认为 RL 允许仅从奖励中学习，而不需要演示，突显了其在模型训练效率方面的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ShashwatGoel7/status/1892668493390094338">Shashwat Goel (@ShashwatGoel7) 的推文</a>：我根据第一性原理整理了这个无需 RL 预备知识的解释器，介绍了 LLM 的 RL 如何工作以及我们为什么需要它🧵核心观点？RL 令人兴奋，因为它允许我们扩展监督。我们现在可以……</li><li><a href="https://arxiv.org/abs/2502.06773">On the Emergence of Thinking in LLMs I: Searching for the Right Intuition</a>：最近的 AI 进展（如 OpenAI 的新模型）正在将 LLM 转变为 LRM (Large Reasoning Models)，这些模型在推理过程中执行推理，花费额外的推理时间和算力以获得更高质量的结果……</li><li><a href="https://www.youtube.com/watch?v=X_niF6KaWd8">🚨🚨 Chad Game Dev Reviews Devin.ai Game Code 🚨🚨</a>：Twitch https://twitch.tv/ThePrimeagen Discord https://discord.gg/ThePrimeagen 成为后端开发：https://boot.dev/prime (此外我也为他们制作课程) 这是一个……
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1341866357540655184)** (73 条消息🔥🔥): 

> `DeepSeek 的 Sparse Attention 论文、AGI 与智能模型、Conditional Attention 概念、Differential Transformers` 


- **DeepSeek 发布 Native Sparse Attention**：今天，参与者们就 DeepSeek 关于 **Native Sparse Attention** 的论文展开了讨论，探讨了其对效率和上下文感知的影响。该活动计划在本周六为错过的人再次举行。
   - *我喜欢 DeepSeek 的论文！* 他们的研究做得很好，标准很高，而且研究结果易于理解。
- **辩论 AGI 的定义与范畴**：大家一致认为定义 **AGI** 仍然具有挑战性，关于其在技术中的影响和实现存在各种观点。参与者建议使用 **ActGI** 等替代术语来应对当前的争论。
   - 讨论强调，*并非每个人的定义都适用于所有场景*，这增加了建立普遍接受定义的复杂性。
- **理解 Conditional 与 Sparse Attention**：讨论将 Conditional attention 视为一种决策过程，而 **sparse attention models** 则是一种隐式选择。一位成员解释了他们的机制如何通过压缩表示来捕捉相关性。
   - 这一对比阐明了现代 attention 机制如何演进以提高计算效率。
- **Continual Learning 的重要性**：对话提到 **continual learning** 与强化学习等其他领域相比成熟度较低，并建议探索不同领域的成熟度水平。参与者强调了在该领域促进理解的重要性。
   - 大家公认，提高记忆保持等能力可以推动学习效率的进步。
- **Transformer 研究中的新颖想法**：关于 **differential transformers** 的贡献引起了兴趣，其创新方法因目前缺乏商业动力而受到关注。参与者认为在当前的研究背景下，许多有价值的论文仍被低估。
   - 表达了将 sparse 和 differential 方法的想法结合起来的愿望，强调了该领域进一步变革的潜力。


---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1341844452825759887)** (9 条消息🔥): 

> `Perplexity AI 与中国审查，Microsoft 发布 Majorana 1，Topological Qubits 详解，Windows 11 隐私更新，Google PaliGemma 2 发布` 


- **Perplexity AI 突破审查**：Perplexity AI 推出了 R1 1776，通过专门的技术手段克服了其 Deepseek R1 模型中的中国审查。
   - 此举凸显了 **AI** 在应对和克服监管障碍方面日益增长的重要性。
- **Microsoft 推出 Majorana 1 量子处理器**：Microsoft 发布了 **Majorana 1**，这是全球首款由 Topological Qubits 驱动的 QPU，旨在扩展至百万级量子比特。
   - 这一进展代表了迈向实用化 **Quantum Computing** 和纠错的重要一步。
- **深入理解 Topological Qubits**：一段新的 YouTube 视频解释了 **Topological Qubits** 的重要性，并包含了来自 Majorana 1 芯片背后的 Microsoft 团队的见解。
   - 内容强调了这些突破性材料将如何重新定义量子计算能力。
- **Windows 11 经历隐私相关变更**：Microsoft 正在移除 Windows 11 File Explorer 中的多项功能，以遵守欧洲的 **隐私法规**。
   - 此次更新为欧洲用户带来了精简的界面，断开了依赖于跟踪用户数据的相关功能。
- **发布 PaliGemma 2 视觉语言模型**：Google 宣布发布 **PaliGemma 2 mix checkpoints**，这是一款升级版的视觉语言模型，具有多种预训练尺寸。
   - 该模型旨在针对多种任务进行 Fine-tuning，包括图像分割和科学问答。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fixvx.com/Alibaba_WanX/status/1892607749084643453">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2502.08859">EnigmaEval: A Benchmark of Long Multimodal Reasoning Challenges</a>: 随着语言模型掌握了现有的推理基准，我们需要新的挑战来评估它们的认知前沿。解谜活动是充满挑战的多模态问题的丰富宝库...</li><li><a href="https://it.slashdot.org/story/25/02/20/0227241/microsoft-declutters-windows-11-file-explorer-in-the-name-of-euro-privacy">Microsoft Declutters Windows 11 File Explorer in the Name of Euro Privacy - Slashdot</a>: Microsoft 表示，为了遵守隐私法规，将为欧洲用户移除 Windows 11 File Explorer 中的多项功能。这些变更影响了欧洲的 Entra ID 账户...</li><li><a href="https://developers.googleblog.com/en/introducing-paligemma-2-mix/?linkId=13028688">Introducing PaliGemma 2 mix: A vision-language model for multiple tasks</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=wSHmygPQukQ">Majorana 1 Explained: The Path to a Million Qubits</a>: 听取来自 Microsoft 团队关于近期在物理学和量子计算领域取得突破的见解，这些突破通过新型 Majorana 1 芯片得以展示，该芯片由一种全新的材料工程化而成...</li><li><a href="https://the-decoder.com/perplexity-ai-removes-chinese-censorship-from-deepseek-r1/">Perplexity AI removes Chinese censorship from Deepseek R1</a>: Perplexity AI 推出了 R1 1776，这是 Deepseek R1 语言模型的修改版本，专门设计用于通过专门的 Post-training 技术克服中国审查。</li><li><a href="https://x.com/elder_plinius/status/1891968598496760230?s=46">来自 Pliny the Liberator 🐉󠅫󠄼󠄿󠅆󠄵󠄐󠅀󠄼󠄹󠄾󠅉󠅭 (@elder_plinius) 的推文</a>: 🧙‍♂️ 󠅗󠅗解锁了新的攻击类别 🧙‍♂️󠅗󠅗</li><li><a href="https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/">Microsoft unveils Majorana 1, the world’s first quantum processor powered by topological qubits - Microsoft Azure Quantum Blog</a>: 来自 Microsoft 的 Majorana 1 是全球首款使用 Topoconductor 构建的量子处理单元 (QPU)。了解更多信息。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1341977154145222717)** (12 条消息🔥): 

> `GPU spec spreadsheet, AI CUDA Engineer, Snapdragon GPU computations, GPU architecture resources, Computer architecture books` 


- **寻找权威的 GPU 规格电子表格**：一位成员对找不到可靠的 **GPU spec spreadsheet** 表示沮丧，类似于链接中的这个 ([Google Sheets](https://docs.google.com/spreadsheets/d/e/2PACX-1vSdXHeEqyabPZTgqFPQ-JMf-nogOR-qaHSzZGELH7uNU_FixVDDQQuwmhZZbriNoqdJ6UsSHlyHX89F/pubhtml))。另一位成员建议将 [TechPowerUp](https://www.techpowerup.com/gpu-specs/) 作为潜在资源。
- **关于 AI CUDA Engineer 的兴奋讨论**：介绍了 **AI CUDA Engineer**，这是一个自动创建优化 CUDA kernels 的系统，声称在 PyTorch 操作中可实现 **10-100 倍的加速** ([Sakana AI](http://sakana.ai/ai-cuda-engineer/))。该系统还发布了一个包含 **超过 17,000 个经过验证的 CUDA kernels** 的数据集，以及一篇详细介绍其功能的论文 ([论文链接](https://pub.sakana.ai/ai-cuda-engineer/paper/))。
- **对 Snapdragon GPU 计算平台的兴趣**：一位成员询问是否有关于 **Snapdragon/Adreno GPU computing** 的频道，因为他们正在 Windows on ARM 笔记本电脑上进行探索。对话强调了他们对该平台上 **OpenCL/Vulkan** 计算的兴趣。
- **寻求 GPU 架构资源**：一位刚接触 GPU 的成员正在寻找专注于 **GPU architecture** 以及优化如何与硬件设计关联的资源。他们引用了来自 **Springer** 的一个有用资源，并征求其他建议。
- **咨询计算机体系结构书籍**：一位成员对优秀的 **computer architecture** 书籍表示好奇，并向社区中的其他人寻求建议。这反映了他们对与 GPU 研究相关的基础原理的持续兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/SakanaAILabs/status/1892385766510338559">来自 Sakana AI (@SakanaAILabs) 的推文</a>: 介绍 AI CUDA Engineer：一个自动化生产高度优化 CUDA kernels 的 Agentic AI 系统。http://sakana.ai/ai-cuda-engineer/ AI CUDA Engineer 可以生产高度优化的...</li><li><a href="https://www.techpowerup.com/gpu-specs/">TechPowerUp</a>: 未找到描述</li><li><a href="https://link.springer.com/book/10.1007/978-3-031-01759-9">General-Purpose Graphics Processor Architectures</a>: 未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/e/2PACX-1vSdXHeEqyabPZTgqFPQ-JMf-nogOR-qaHSzZGELH7uNU_FixVDDQQuwmhZZbriNoqdJ6UsSHlyHX89F/pubhtml#">GPU_Compare</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1342249630687170622)** (1 messages): 

> `TMA Descriptor in Triton, Persistent Kernel Implementations, Matrix Multiplication Techniques, FP8 and FP16 Support, Benchmarking Triton with cuBLAS` 


- **探索 Triton 中 TMA Descriptor 的用法**：关于 [persistent matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) 的教程阐述了 **TMA (Tensor Memory Accelerator)** 描述符如何增强 Triton 中的矩阵乘法实现。
   - 该脚本提供了多种示例，包括 **naive**、**persistent** 和 **基于 TMA 的方法**，强调了 TMA 在高效内存使用方面的优势。
- **矩阵乘法方法亮点**：该教程展示了在 Triton 中实现的几种矩阵乘法技术，特别是为了优化性能而采用的 **naive**、**persistent** 和 **基于 TMA** 的方法。
   - 教程还提到 Kernel 支持 **FP16** 和 **FP8** 数据类型，并根据所选精度提供了具体的使用说明。
- **可配置的命令行参数**：用户可以通过命令行参数灵活指定矩阵维度和迭代次数，例如在 **FP8** 和 **FP16** 示例中使用 `--prec` 进行精度设置。
   - 例如，命令 `python 09-persistent-matmul.py --prec fp8 --K_range 128 1024 --K_step 128` 设置了 **FP8** 实现的参数。
- **共享内存大小的注意事项**：教程警告称，在共享内存大小有限的设备（如 **RTX-4090**）上可能会运行失败，这可能会影响性能和兼容性。
   - 这一考虑对于旨在成功执行教程中提供的示例的用户至关重要。
- **基准测试策略说明**：该脚本在不同配置下对 Triton 和 **cuBLAS 实现** 进行基准测试，并使用 **proton profiler** 对其进行评估。
   - 这种基准测试方法有助于用户理解不同矩阵乘法技术的性能影响。



**提到的链接**：<a href="https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html">Persistent Matmul &mdash; Triton  documentation</a>：未找到描述

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1341841708400508979)** (14 messages🔥): 

> `Raw-Dogged Tensor Proposal, RTX 5080+ Triton Issues, Warp Specialization Kernels, TF32 NT Kernel Inquiry, Custom gmem Offset Math in Device Code` 


- **提议 Raw-Dogged Tensor 命名法**：一位成员提议了一种名为 **raw-dogged Tensor** 的新命名法，旨在使存储格式与 **MMA_Atom** 线程布局保持一致。他们指出这显著降低了排列（permutation）的复杂性。
   - 另一位成员确认在 **int8 matmul** 中使用了这种方法，并强调这是避免共享内存 Bank 冲突（bank conflicts）的必要手段。
- **RTX 5080+ Triton 兼容性障碍**：一位成员分享了在 Triton 上使用 **TorchRL** 运行 **RTX 5080+** 的经验，强调了与 `torch.compile` 触发 Triton 相关的错误。他们通过移除 **PyTorch-triton** 安装解决了这些问题。
   - 这引起了人们对 Triton 和 PyTorch 交互中仍然存在的兼容性问题的关注。
- **关于 Warp Specialization Kernel 的讨论**：有人询问了关于 **warp specialization kernels** 的问题，并引用了来自 [arxiv 链接](https://arxiv.org/pdf/2307.03760) 的示例。成员们讨论了常见的 **具有生产者/消费者专业化的 GEMM Kernel**，并指出了同步技术。
   - 一位成员还鼓励查看一份强调 **GEMM warp specialization** 的实用演示文稿。
- **寻求 TF32 16x8x8 NT Kernel**：有人请求提供 **TF32 16x8x8 NT kernel** 实现，作为改进其 Cutlass 工作的一部分。该查询反映了当代应用中对优化 Kernel 的持续需求。
- **用于 Batched Syrk 的自定义 gmem 偏移计算**：一位用户询问如何通过根据 Block 索引调整 **gmem offset math** 来实现 **batched strided SYRK**。他们表示在确保 **bM == bN** 的同时，很难通过标准的 Cutlass 功能找到合适的路径。



**提到的链接**：<a href="https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp">cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp at main · NVIDIA/cutlass</a>：用于线性代数子程序的 CUDA 模板。通过在 GitHub 上创建一个账户来为 NVIDIA/cutlass 做出贡献。

  

---

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1342211788707467325)** (1 条消息): 

> `GRPO algorithm advancements, VRAM reduction techniques, Extended context lengths, Llama 3.1 benchmarking, Gradient checkpointing` 


- **Unsloth 发布 10 倍 Context 和 90% VRAM 节省方案**：Unsloth 宣布了新算法，使得 **Qwen2.5-1.5B** 模型的训练仅需 **5GB VRAM**，实现了 **90%** 的 VRAM 占用减少。
   - 他们表示：*'使用 Unsloth，你现在可以在无精度损失的情况下训练自己的 Reasoning Model。'* 更多详情请见其 [blog](https://unsloth.ai/blog/grpo)。
- **基准测试结果显示显著的 VRAM 节省**：对比基准测试显示，**Llama 3.1** 在 20K Context 下的标准 GRPO QLoRA 设置此前需要 **510.8GB VRAM**，现在减少到了 **54.3GB**。
   - 这一改进源于对之前 **Gradient checkpointing** 算法的利用，灵感来自 **Horace He** 的 Linear Cross Entropy 实现。



**提到的链接**：<a href="https://x.com/UnslothAI/status/1892640995847901684">Unsloth AI (@UnslothAI) 的推文</a>：今天，我们推出了新算法，可实现 10 倍长的 Context 长度，并减少 90% 的 VRAM 用于训练 Reasoning Models (GRPO)。使用 Unsloth，你现在只需 5G 即可训练自己的 Reasoning Model...

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1341861646724890756)** (12 条消息🔥): 

> `AI CUDA Engineer, Nanotron Blog Post, HadaCore Quantization, CUDA Kernel Optimization, Quantization Techniques` 


- **AI CUDA Engineer 自动化优化**：[AI CUDA Engineer](https://pub.sakana.ai/ai-cuda-engineer/) 可以生成高度优化的 CUDA Kernel，相比 PyTorch 中常见的机器学习操作实现了 **10-100x** 的加速。
   - 它在将 PyTorch 操作转换为 CUDA 方面达到了 **90%** 的成功率，优于原生的 Torch Kernel，但有人认为由于基准测试较弱，实际论文可能存在*过度炒作*。
- **Nanotron 团队发布新博客文章**：[Nanotron](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 团队发布了一篇令人兴奋的博客文章，被一些用户评价为 **awesome**。
   - 讨论集中在 Nanotron 是否是 Hugging Face 内部的一个团队，并确认了他们参与了该 [GitHub](https://github.com/huggingface/nanotron) 项目。
- **HadaCore 引入先进 Quantization 方法**：[HadaCore](https://pytorch.org/blog/hadacore/?utm_source=tldrai) 方法重点介绍了一个 Hadamard Transform CUDA Kernel，它增强了 Quantization 技术的效率，比前代产品实现了 **1.1–1.4x** 的性能提升。
   - 最近的作品如 [QuaRot](https://arxiv.org/abs/2404.00456) 和 [SpinQuant](https://arxiv.org/abs/2405.16406) 展示了提高 LLM 中使用的低精度 Quantization 方法数值准确性的方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sakana.ai/ai-cuda-engineer/">无标题</a>：未找到描述</li><li><a href="https://pytorch.org/blog/hadacore/?utm_source=tldrai">HadaCore: Tensor Core Accelerated Hadamard Transform Kernel</a>：Quantization 是一种通过压缩模型权重并在低精度数据类型中执行（更快的）计算来提高模型推理速度的方法。然而，Quantization 可能会导致精度...</li><li><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>：未找到描述</li><li><a href="https://pub.sakana.ai/ai-cuda-engineer/">The AI CUDA Engineer 👷</a>：未找到描述</li><li><a href="https://github.com/huggingface/nanotron">GitHub - huggingface/nanotron: Minimalistic large language model 3D-parallelism training</a>：极简的大型语言模型 3D 并行训练 - huggingface/nanotron
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1342175502055641170)** (2 messages): 

> `Apple ML Research, A5Labs ML Engineer Position` 


- **Apple 研究科学家职位开放**：Apple 机器学习研究小组正在招聘一名**研究科学家**，专注于**efficient foundation models**领域由好奇心驱动的工作。感兴趣的候选人可以查看[此处的职位描述](https://jobs.apple.com/en-us/details/200587898/aiml-ml-researcher-foundation-models?team=MLAI)。
   - 该团队拥有深厚的研究背景，在 **NLP** 和 **speech** 领域发表过具有影响力的论文，并强调可复现的高质量研究的重要性。
- **A5Labs 寻求远程 ML Engineer**：A5Labs 正在寻找一名专注于 **reinforcement learning** 和游戏领域的**远程 ML Engineer**，加入其多元化的全球团队。感兴趣的申请人可以查看[此处的职位列表](https://a5labs.co/we-are-hiring/?jobId=Pz34B6RbYyAI)。
   - 该团队欢迎候选人直接发送私信，并强调其在**亚洲**、**北美**和**欧洲**的国际化布局。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://a5labs.co/we-are-hiring/?jobId=Pz34B6RbYyAI">We’re hiring! - A5 Labs</a>: 职业中心，我们正在招聘！加入 A5 Labs 团队</li><li><a href="https://jobs.apple.com/en-us/details/200587898/aiml-ml-researcher-foundation-models?team=MLAI.">AIML - ML Researcher, Foundation Models - Careers at Apple</a>: 申请 Apple 的 AIML - ML Researcher, Foundation Models 职位。阅读职位介绍，了解它是否适合你。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1341843700451381348)** (9 messages🔥): 

> `torchao issue, HuggingFace error, past_key_values bug, modeling_llama.py fix` 


- **torchao 遇到问题**：一名成员报告称 *torchao* 中存在一个**损坏的问题 (broken issue)**，并表示他们正在进一步调查。
   - 另一名成员表示愿意提供帮助。
- **HuggingFace torchao 示例错误**：分享了一个 GitHub issue 链接，涉及在运行 HuggingFace torchao 示例时出现的 **torch.compile 错误**，引用版本为 **torch (2.6.0)** 和 **torchao (0.8.0)**。
   - issue 描述中提到了 Quantization 以及所提供的示例代码存在问题。
- **确定错误原因**：据建议，该错误是由于 Hugging Face 交替使用 **past_key_values** 和 **past_key_value** 导致的，从而引起了混淆和 Bug。
   - 这种不一致性被认为是导致错误的主要原因。
- **针对 Llama 模型的提议修复**：链接了一个 Pull Request，通过更新 *modeling_llama.py* 以正确处理键跳过问题，为 Llama 模型提供 **bugfix**。
   - 该 bugfix 解决了 **past_key_value** 和 **past_key_values** 的混合使用问题，确保在处理过程中两者都能被适当地跳过。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/issues/1705">torch.compile error when running the HuggingFace torchao example · Issue #1705 · pytorch/ao</a>: 当我运行来自 https://huggingface.co/docs/transformers/main/en/quantization/torchao 的代码片段时，遇到了 torch.compile 错误。torch 版本: 2.6.0, torchao 版本: 0.8.0, transformers 版本...</li><li><a href="https://github.com/huggingface/transformers/pull/36289">[bugfix] Update modeling_llama.py so it skips keys correctly by HDCharles · Pull Request #36289 · huggingface/transformers</a>: Llama 模型交替使用 past_key_value 和 past_key_values，这导致了问题，因为在 _skip_keys_device_placement 中实际上只跳过了其中一个，而两者都需要被跳过...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1342176395618549872)** (1 messages): 

> `Together Computer Series Funding` 


- **Together Computer 获得 3.05 亿美元系列融资**：今天，[Together Computer](https://www.linkedin.com/posts/togethercomputer_today-were-announcing-our-305m-series-activity-7298375921277800450-Jvjs/) 宣布了其令人瞩目的 **3.05 亿美元**系列融资，旨在加速其技术进步。
   - 这一重大投资凸显了 AI 计算领域日益增长的兴趣和潜力。
- **AI 计算投资的增长**：本轮融资展示了一种趋势，即投资者正越来越多地向 **AI computing** 公司投入资金，表明了强烈的市场信心。
   - 行业专家认为，这可能会带来进一步的创新和突破，特别是在 **machine learning** 和 **cloud computing** 领域。

---

### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 条消息): 

kpk1340: 有人在纽约吗？
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1341939234327494676)** (9 条消息🔥): 

> `Mi50 硬件支持, Matmul 操作, GPU 架构` 


- **Mi50 缺乏硬件 Matmul 支持**：成员们确认，尽管 **Mi50** 具备处理多种数据类型的能力，但它不支持硬件 Matmul 或 Tensor 操作。
   - 一位成员表示，*'没有 wmma 也没有 mfma'*，这表明缺乏特定的矩阵乘法功能。
- **Matmul 技术澄清**：讨论显示，Matmul 支持在用于 CDNA 架构数据中心的 **XDL** 和用于 RDNA3 显卡游戏的 **WMMA** 之间存在界限。
   - 另一位成员强调，**Mi50** 使用的是 **Vega / GCN 5** 架构，不包含这些新特性。
- **确认 Mi50 的局限性**：对话强调了关于 **Mi50** 在 Matmul 能力方面局限性的共识，特别是它无法使用 WMMA。
   - 成员们对这一确认表示感谢，肯定了他们对该硬件规格的理解。



**提到的链接**：<a href="https://www.8anet.com/Product/17823/AMD-100-506143-Radeon-Instinct-MI50-Accelerator-PCIe-4-0-x16-32GB-HBM2-4096-bit-3840-Stream-Processors-Passive-Cooling">
	8ANET - AMD 100-506143 Radeon Instinct™ MI50 Accelerator PCIe 4.0 x16 32GB HBM2 4096-bit 3840 Stream Processors Passive Cooling
</a>：未找到描述

  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1341903496987017267)** (3 条消息): 

> `收敛测试修复, PR 合并流程, Native Sparse Attention` 


- **收敛测试修复成功**：一位成员报告称，通过解决 **MiniModelConfig** 中缺失的 logit 缩放参数，修复了**收敛测试**，从而纠正了 logit 幅度。
   - 他们表达了希望协助合并 **PR** 的愿望，并表示愿意做任何必要的事情来加速这一过程。
- **询问 PR 编号**：另一位成员询问了合并流程所需的具体 **PR 编号**。
   - 这条消息很轻松，带有一个笑脸表情，表明讨论氛围友好。
- **对 Native Sparse Attention 协作的兴趣**：一位成员发起了关于 **Native Sparse Attention** 特性的对话，询问是否有人有兴趣合作，使其与硬件对齐并在 **liger** 中实现原生可训练。
   - 这一合作邀请得到了热烈响应，展示了社区的协作精神。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 条消息): 

iron_bound: Goat https://m.youtube.com/watch?v=leCY8vCUS4g
  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1341947354999296130)** (10 messages🔥): 

> `AI CUDA Engineer, CUDA kernel optimization, Rewards and challenges in code generation, Research papers on CUDA, Evolutionary AI approaches` 


- **AI CUDA Engineer 优化 CUDA kernel**：[AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/) 实现了高度优化的 CUDA kernel 的自动化生产，相比 PyTorch 中的标准操作实现了 **10-100 倍的加速**。
   - 该系统利用进化式 **LLM 驱动的代码优化**来增强 CUDA kernel 性能，甚至发现新的算子解决方案。
- **显著贡献与发现**：论文概述了重要发现，例如在特定任务（包括类别交叉熵优化）中 **kernel 性能超过了 torch.compile**，以及一个包含 **17K** 个具有加速效果的 kernel 对数据集。
   - 它还强调了从 NCU 中筛选有用数据的挑战，以及向 LLM 传授 Tensor Cores 等新特性的难度。
- **关于奖励机制的见解**：*AutoML 回来了！* 目前的讨论强调，改进 CUDA kernel 的奖励函数定义非常明确，重点在于**数值正确性和实际运行时间（wall clock speed）**。
   - 一位成员开玩笑地提到了一次**奖励作弊（reward hacking）**案例：一个 “nop kernel”（空操作 kernel）因为什么都不做而获胜，幽默地反映了优化的本质。
- **关于 kernel 问题的讨论**：由于**输出缓冲区重用**导致某些 kernel 格式错误，从而影响其性能，这引起了人们的关注。
   - 诸如回收前一个输出所使用的内存等问题被讨论为确保 kernel 正确性的重大障碍。
- **有趣的协作环境**：几位成员考虑了与 Sakana AI 潜在的合作机会，认为这是一个很有前景的 **Colab 机会**。
   - 讨论氛围轻松愉快，成员们分享了关于如何通过不执行任何操作来轻松避免错误的俏皮话——*如果你什么都不做，你就不会搞砸任何事情*。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/miru_why/status/1892478739491279153">miru (@miru_why) 的推文</a>：@main_horse 看起来 torch::empty_like 调用回收了包含正确输出的 torch_output 内存（缓存分配器），然后由于 2d block 配置，kernel 几乎什么都没做...</li><li><a href="https://x.com/drjimfan/status/1892404919480832259?s=46">Jim Fan (@DrJimFan) 的推文</a>：我最近见过的最酷的自主编程 Agent：使用 AI 编写更好的 CUDA kernel 来加速 AI。AutoML 真的回来了！你能用你的计算资源做的杠杆率最高的事情就是...</li><li><a href="https://x.com/sakanaailabs/status/1892385766510338559?s=46">Sakana AI (@SakanaAILabs) 的推文</a>：介绍 AI CUDA Engineer：一个自动化生产高度优化 CUDA kernel 的 Agentic AI 系统。http://sakana.ai/ai-cuda-engineer/ AI CUDA Engineer 可以生产高度优化的...</li><li><a href="https://x.com/miru_why/status/">GitHub 推文 - FixTweet/FxTwitter</a>：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1341985412683399269)** (1 messages): 

> `Hybrid Speech Processing Application, NVIDIA Jetson Nano, Speech Separation Model, Cloud LLM Integration` 


- **展示混合语音处理应用**：一位用户为一个课程构建了一个**混合语音处理应用**，该应用在 **NVIDIA Jetson Nano** 上部署了一个语音分离模型，根据提示词（prompt）过滤输入语音。
   - 该应用集成了云端能力，由 **LLM** 解码提示词并将 embedding 发送到 Edge 设备进行处理。
- **请求对应用报告的反馈**：该用户附上了一份题为 [Listen, Chat, and Edit on Edge](https://cdn.discordapp.com/attachments/1303441437592911912/1341985412197122159/Listen__Chat__and_Edit_on_Edge.pdf?ex=67b8a58f&is=67b7540f&hm=e5ce784faf8d568c323c01e402323a53e7e88e4367b798d3115f07821dd98acd&) 的报告，并请求对其项目提供反馈。
   - 他们鼓励对该项目的方法和结果进行讨论和评估。


  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1341849877054881835)** (76 条消息🔥🔥): 

> `Reasoning Gym Server, 空间推理数据集, 十进制算术增强, Needle in Haystack 数据集, UnslothAI 的新算法` 


- **Reasoning Gym Server 进展**：团队成员正在敲定 Reasoning Gym server 的第一个版本，server 和 CLI 工具正在合并并进行调试，以确保运行顺畅。
   - 目标是实现对各种推理任务的无缝处理，包括可能集成的 ILP 任务。
- **寻找空间推理数据集**：成员们讨论了对专注于**空间推理 (spatial reasoning)** 数据集的需求，并提出了生成与 3D 空间和关系相关问题的想法。
   - 示例包括使用经典谜题（如弹珠问题）和研究论文中的概念来完善数据集。
- **十进制算术增强**：讨论了可能减少十进制算术配置中的最大有效数字，以确保结果准确。
   - 成员们表示，虽然浮点数问题是已知的，但在训练中进行适当处理可以优化性能。
- **Needle in Haystack 数据集改进**：讨论包括通过可能推迟数据加载直到必要时，来优化 Needle in Haystack 数据集的内存使用。
   - 成员们强调了在内存效率与生成及保留多个样本的能力之间取得平衡的重要性。
- **UnslothAI 发布新算法**：UnslothAI 的新发布承诺在训练推理模型时提供 10 倍长的上下文长度和减少 90% 的 VRAM 占用。
   - 这一进步允许在资源极少的情况下有效地训练模型，引发了团队成员的兴奋。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2401.03991">Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark</a>: 人工智能 (AI) 在各个领域取得了显著进展，像 ChatGPT 这样的大型语言模型因其类人的文本生成能力而受到广泛关注...</li><li><a href="https://www.interconnects.ai/p/artifacts-7">The latest open artifacts (#7): Alpaca era of reasoning models, China&#x27;s continued dominance, and tons of multimodal advancements</a>: Artifacts Log 7。对于 AI 研究人员和从业者来说，这将继续是一个有趣的春天。</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/173">Add &quot;emoji mystery&quot; dataset · Issue #173 · open-thought/reasoning-gym</a>: 创建一个数据集，生成 "emoji mystery" 问题，其中包含通过 "变体选择器 (variation selectors)" 编码在 Unicode 表情符号中的隐藏消息。参见 Andrej Karpathy 的 x 帖子...</li><li><a href="https://x.com/unslothai/status/1892640995847901684">Tweet from Unsloth AI (@UnslothAI)</a>: 今天，我们发布了新算法，支持 10 倍长的上下文长度和减少 90% 的 VRAM 用于训练推理模型 (GRPO)。使用 Unsloth，你现在只需 5G 显存即可训练自己的推理模型...</li><li><a href="https://github.com/Fangjun-Li/SpatialLM-StepGame">GitHub - Fangjun-Li/SpatialLM-StepGame: Codes and data for AAAI-24 paper &quot;Advancing Spatial Reasoning in Large Language Models: An In-depth Evaluation and Enhancement Using the StepGame Benchmark&quot;</a>: AAAI-24 论文 "Advancing Spatial Reasoning in Large Language Models: An In-depth Evaluation and Enhancement Using the StepGame Benchmark" 的代码和数据 - Fangjun-Li/SpatialLM-StepGame</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/174">Add decimal number comparison by Adefioye · Pull Request #174 · open-thought/reasoning-gym</a>: 用于比较十进制数的 Python 生成器</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/arithmetic/number_format.py">reasoning-gym/reasoning_gym/arithmetic/number_format.py at main · open-thought/reasoning-gym</a>: 程序化推理数据集。通过在 GitHub 上创建一个账户来为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/170">Adds Needle in a Haystack problems by Miserlou · Pull Request #170 · open-thought/reasoning-gym</a>: 例如：Boedyn 疯狂迷恋卷饼。Tyrnan 后悔地理。Deryn 赞美汤。David-Jay 颂扬擦拭家具。Malikye 欢庆文学。Oluwadamilare 庆祝电动滑板车。Nai...
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1341828814451376130)** (130 条消息🔥🔥): 

> `SD 与 Flux 的对比、ControlNet 应用、自定义模型创建、使用草图生成图像、AI 工具的 GPU 推荐` 


- **在 Stable Diffusion 和 Flux 之间做出选择**：成员们讨论认为 **Stable Diffusion (SD)** 目前比 **Flux** 更成熟，尽管 **Flux** 仍处于开发阶段。
   - 一位成员建议查看示例图像，以确定哪种模型最符合个人偏好。
- **用于图像姿态的 ControlNet**：ControlNet 可以有效地利用深度图或骨架线框，根据姿态生成图像，处理诸如“手在前”或“手在后”等调整。
   - 有人指出，使用控制方法可以根据提供的姿态实现更准确且更具创造性的图像生成。
- **关于自定义模型创建的咨询**：一位用户表示希望聘请同时精通 **Stable Diffusion** 和传统艺术的人才，来创建自定义模型和提示词风格。
   - 其他人质疑这种要求的实用性，建议亲自学习创建模型会更有益且更具成本效益。
- **从草图到图像生成的 Workflow**：一位用户分享了一个 Workflow，包括在 iPad 上使用粗略草图来引导 AI 生成图像，并寻求关于如何从草图过渡到成品图像的建议。
   - 他们承认 img2img 流程的效用，但不确定如何从简单的涂鸦开始。
- **图像生成工具的 GPU 需求**：讨论强调 **Nvidia GPU** 仍然是高效运行 **Stable Diffusion** 的推荐选择，而 AMD 选项可能会面临性能问题。
   - 用户分享了他们当前的 GPU 配置，并讨论了不同模型与 GPU 能力的兼容性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/jiseok-kim-jiseok-big-ocean-bigocean-kpop-gif-16919206117458777151">Jiseok Kim Jiseok GIF - Jiseok Kim jiseok Big ocean - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=n233GPgOHJg">Stable Diffusion 模型一站式详解 (1.5, 2, XL, Cascade, 3)</a>: 在这段视频中，我解释了 Stable Diffusion 的 5 个不同模型系列。我有说错或遗漏什么吗？请告诉我。章节：00:00 介绍...</li><li><a href="https://github.com/LykosAI/StabilityMatrix/">GitHub - LykosAI/StabilityMatrix: 适用于 Stable Diffusion 的多平台包管理器</a>: 适用于 Stable Diffusion 的多平台包管理器 - LykosAI/StabilityMatrix</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI?tab=readme-ov-file#installing-on-windows">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (原 StableSwarmUI)，一个模块化的 Stable Diffusion Web 用户界面，强调易于访问的高级工具、高性能和可扩展性。</a>: SwarmUI (原 StableSwarmUI)，一个模块化的 Stable Diffusion Web 用户界面，强调易于访问的高级工具、高性能和可扩展性。 - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui 安装指南</a>: Stable Diffusion 知识库（设置、基础、指南等） - CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1341964090435960833)** (6 条消息): 

> `GPU scheduler optimization, AI CUDA Engineer, ARENA 4.0 program` 


- **寻求机器学习项目的数据集推荐**：一位成员分享了他们在机器学习研究中对使用 **deep reinforcement learning** 进行 **GPU scheduler optimization** 的关注，并寻求有关 benchmark 数据集的建议。
   - 他们专门征求了建议，表示目前在寻找合适数据集方面面临挑战。
- **介绍用于优化的 AI CUDA Engineer**：分享了一个关于 **AI CUDA Engineer** 的资源，这是一个用于优化 CUDA kernels 的自动化框架，据报道将 PyTorch 转换为 CUDA 的成功率 **>90%**。
   - 尽管其效果显著，但根据社区共识，有人担心结果可能存在 **虚假/错误 (spurious/error-ridden)**。
- **关于 AI CUDA Engineer 数据质量的讨论**：一位成员指出，包含由 **AI CUDA Engineer** 生成的 kernels 的数据集可能存在缺陷，因为生成的输出可能不准确。
   - 这引发了关于该数据集相关 baseline 实现可靠性的辩论。
- **关于 ARENA 4.0 的联系请求**：一位用户表达了希望与 **ARENA 4.0 program** 的创建者取得联系的意愿，并请求私信。
   - 这表明了对该特定项目进行协作或寻求帮助的需求。



**提到的链接**：<a href="https://pub.sakana.ai/ai-cuda-engineer">The AI CUDA Engineer 👷</a>：未找到描述

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1341983181078921226)** (81 messages🔥🔥): 

> `AI CUDA Engineer, CUDA and PyTorch performance, LLM Optimization, Clockwork RNN, Model training insights` 


- **AI CUDA Engineer 令人印象深刻的主张**：介绍了 [AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/)，这是一个 AI 系统，声称在 CUDA kernel 生成方面比 PyTorch 实现 **10-100 倍加速**，并附带了一个包含 **17,000 多个 kernel** 的数据集。
   - 然而，成员间的讨论对评估的准确性提出了质疑，并指出类似项目此前存在误导性陈述，表明怀疑态度依然存在。
- **CUDA kernel 评估缺陷**：针对 kernel 评估方法的批评出现，揭示了据称具有 **150 倍加速的 kernel** 实际上利用了内存重用，并且在实现中存在根本性的 **bugs**。
   - 成员们对这些 kernel 的可靠性表示怀疑，从而引发了关于所提供样本中可能普遍存在问题的更广泛讨论。
- **LLM Compilers 的探索**：围绕 **LLM-compilers** 的概念展开了对话，成员们推测 LLM 是否可以将高级 PyTorch 代码翻译成针对特定设置优化的机器码。
   - 虽然这个想法引起了成员们的兴趣，但大家一致认为，由于缺乏通用的指令集，重大挑战可能会阻碍进展。
- **Clockwork RNN 和 Transformer 架构**：讨论涉及了 **Clockwork RNN**，这是一种改进的架构，通过对各种输入粒度使用独立的模块来提高性能，类似于 Transformer 中的预测。
   - 成员们辩论了此类架构在未来模型中应用的可行性，包括空洞卷积（dilated convolutions）和 Attention 机制的应用。
- **对实验性模型 checkpoint 的需求**：对话表明了对 Muon 优化器等模型 **checkpoints** 的需求，强调与传统模型的直接比较可能会产生富有洞察力的结果。
   - 此外还强调了半黑盒超参数优化（semi-blackbox hyperoptimization）的潜在好处及其对训练策略的影响，呼吁在理论框架内进行进一步探索。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/SakanaAILabs/status/1892385766510338559">来自 Sakana AI (@SakanaAILabs) 的推文</a>：介绍 AI CUDA Engineer：一个自动化生产高度优化 CUDA kernel 的 Agentic AI 系统。http://sakana.ai/ai-cuda-engineer/ AI CUDA Engineer 可以生产高度优化的...</li><li><a href="https://arxiv.org/abs/2502.10927">Self-attention 的底层结构：Transformer 训练中的对称性、方向性和涌现动力学</a>：Self-attention 对 Transformer 架构至关重要，然而信息如何嵌入 Self-attention 矩阵以及不同的目标函数如何影响这一过程仍不清楚。我们提出...</li><li><a href="https://arxiv.org/abs/1402.3511">A Clockwork RNN</a>：序列预测和分类是机器学习中普遍且具有挑战性的问题，可能需要识别时间上相距甚远的输入之间的复杂依赖关系。循环神经网络...</li><li><a href="https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf">Open-Reasoner-Zero/ORZ_paper.pdf at main · Open-Reasoner-Zero/Open-Reasoner-Zero</a>：Open-Reasoner-Zero 的官方仓库。通过在 GitHub 上创建账号为 Open-Reasoner-Zero 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1341827572085821552)** (4 messages): 

> `Logit Lens, Tuned Lens, Transformers Analysis, Computer Security Analogy, Average-case Goals` 


- **Logit Lens 和 Tuned Lens 的前景**：讨论强调了 **Logit Lens** 和 **Tuned Lens** 在分析 Transformer 和循环模型方面的潜力，认为在理解模型每一步如何处理问题方面存在尚未挖掘的价值。
   - 进一步探索这一点可能会为长文本 Chain of Thought 推理提供见解。
- **分析复杂问题的挑战**：一位成员表示，解决推文中提出的特定问题非常困难，并将其比作 **计算机安全** 中的 Fuzzing 和识别后门等问题。
   - 这凸显了在辨别模型行为中有意义的模式时所涉及的复杂性和微妙性。
- **对平均情况性能（Average-case Performance）的直觉**：一位参与者认为，瞄准 **平均情况性能** 可能更容易实现，因为它不依赖于隐藏的线索，而是依赖于自然的训练配置。
   - 这一观点强调了关注可访问的隐变量（latents）而非难以捉摸的离群情况的重要性。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1342052538655965185)** (8 条消息🔥): 

> `lm-eval-harness, runtime benchmarks, model path errors, lm studio, task path errors` 


- **寻求使用 lm-eval-harness 的基准测试**：一名成员询问如何使用 **lm-eval-harness** 对在 **lm studio** 本地运行的模型进行基准测试，同时评估 PC 的性能。
   - *stellaathena* 提到 **lm-eval** 衡量的是性能（模型表现），而非运行时间（runtime）。
- **已获取运行时间基准测试**：该成员澄清他们已经使用 **llm perf** 收集了运行时间基准，现在正面临与 **eval harness** 任务路径相关的错误。
   - *stellaathena* 请求提供正在运行的命令以便更好地提供帮助。
- **lm_eval 命令中的模型路径错误**：该成员分享了他们正在使用的命令，但尽管尝试更改，仍因错误的 **model path** 而反复出现问题。
   - 他们提供了自己的命令，并指定了模型路径以及他们正在使用的其他参数。
- **请求私下协助**：该成员表示希望通过私信联系，以获得针对其问题的更个性化的协助。
   - 这表明他们更倾向于通过一对一支持来解决所面临的挑战。
- **尝试不同的模型补全方式**：他们提到在基准测试工作中尝试了 **openai-completions** 和 **local-chat completions**。
   - 这表明在任务执行困难的情况下，他们正在寻求更广泛的解决方案。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1341896383501828126)** (12 条消息🔥): 

> `Evo2 Genome Models, Llama 3.2 Comparison, NCCL_BUFFSIZE Adjustments` 


- **Evo2 基因组模型利用 GPT-NeoX**：新的 **Evo2 基因组模型**是使用[一个基于 GPT-NeoX 的库](https://github.com/Zymrael/savanna)训练的。这证实了当代模型与现有框架的深度集成。
   - 听到该公告在社区内反响良好，*感到非常欣慰*。
- **Llama 3.2 显示出 TPS 差异**：一名成员对比了 NeMo 和 NeoX 上的 **Llama 3.2 1B 配置**，注意到 NeoX 的 **21.3K TPS** 对比 NeMo 的 **25-26K TPS**。分享的[配置文件](https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/hubble/Speed_Exps/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_Llama_3_2_Fusions_All_4_FA_Swiglu.yml)概述了实验设置。
   - 性能见解可以帮助他人优化设置，他们可以参考 **[WandB run](https://wandb.ai/aflah/hubble-speed-testing/runs/nioywj5f?nw=nwuseraflah)** 获取详细指标。
- **讨论 NCCL_BUFFSIZE 的调整**：一名成员对 **NCCL_BUFFSIZE** 提出了*好奇*，建议将其值设为 **2097152**。这被认为有利于多 GPU 通信，特别是在使用 InfiniBand 时。
   - 建议独立于 DeepSpeed 的 bucket size 调整缓冲区大小，这意味着最佳实践可以增强复杂设置中的性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/hubble/Speed_Exps/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_Llama_3_2_Fusions_All_4_FA_Swiglu.yml">gpt-neox/configs/hubble/Speed_Exps/1_1B_Baseline_BS_8_GAS_8_No_Activation_Checkpointing_GQA_Llama_3_2_Fusions_All_4_FA_Swiglu.yml at olmo-support · aflah02/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自回归 Transformer - aflah02/gpt-neox</li><li><a href="https://wandb.ai/aflah/hubble-speed-testing/runs/nioywj5f?nw=nwuseraflah">aflah</a>：Weights & Biases，机器学习开发者工具
</li>
</ul>

</div>
  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1341922645066649671)** (12 条消息🔥): 

> `Podcast TTS 问题，邀请非 Google 用户访问 Notebooks，Tesla 自动驾驶专利见解，使用 NotebookLM 进行居家学习，AI 对文学作品的理解` 


- **Podcast TTS 问题**：一位用户在尝试让 TTS 功能正确读取其输入时遇到困难，尝试了各种 prompts 但均未成功。
   - 他们对播客主持人无法按预期朗读文本表示沮丧。
- **邀请非 Google 用户访问 Notebooks**：一位成员询问是否可以邀请没有 Google 账号的人访问 notebook，类似于 Google Docs 的功能。
   - 这引发了关于 Notebook LM 协作中替代访问方法的讨论。
- **Tesla 自动驾驶专利见解**：一位用户在最近的一项专利授权后探讨了 Tesla 的自动驾驶 AI，提到了 **Lidar**、**Radar** 和 **Ultrasonics** 等关键技术。
   - 他们制作了一个播客讨论其发现，并强调在他们的 Patreon 上为听众提供了一篇 **免费** 文章。
- **使用 NotebookLM 进行居家学习**：一位用户分享了将 NotebookLM 与 Gemini 结合用于孩子居家学习的积极体验，将其比作拥有高技能的助手。
   - 他们认为这种集成方法对执行教学工作提供了显著帮助。
- **AI 对文学作品的理解**：多位用户对 AI 误解其写作和角色细节表示沮丧，并引用了各种错误示例。
   - 一位用户指出，即使提供了证据， AI 也经常拒绝承认更正，导致与叙述产生冲突。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1341819474499866654)** (97 条消息🔥🔥): 

> `NotebookLM 权限，音频功能，Notebook 共享问题，来源限制，NotebookLM 用户体验` 


- **引导 NotebookLM 权限**：用户讨论了如何共享 notebook，一些人反映在 Plus 版本上找不到共享按钮，强调了用户角色可能存在的限制。
   - 一位用户建议针对缺失的共享按钮功能提交错误报告。
- **在课程中使用音频概览**：一位用户询问是否可以将音频“深度探讨”（Deep Dive）输出用于学术目的，并确认允许在 EDU 账户内共享。
   - 提供了关于生成 Audio Overviews 的指导，指出它们反映的是源内容而非 AI 主持人的观点。
- **嵌入功能和组织请求**：请求文件夹组织选项是一个反复出现的主题，用户表示需要改进对笔记和 notebook 的管理。
   - 该功能的请求已在内部记录，但未提供实施的时间表。
- **解决上传挑战**：用户报告了上传包括 PDF 和音频文件在内的各种文件类型时遇到的问题，推测可能存在 Bug。
   - 建议测试上传不同的文件或使用 Google Docs 来有效管理内容。
- **澄清来源使用政策**：围绕使用新闻源作为 NotebookLM 输入的限制引发了关于接受的来源类型的疑问。
   - 一位用户建议，在面临公认新闻机构的限制时，可以通过直接复制文本而不是使用链接来解决。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15731776?hl=en&ref_topic=14272601&sjid=7303781756764289573-NC">Audio Overviews - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://notebooklm.google.com/?hl=ar">تسجيل الدخول - Google 账号</a>：未找到描述</li><li><a href="https://youtu.be/EGhXtFjzcJY">NotebookLM - 免费研究 YouTube 评论和情感！</a>：全面的 NotebookLM 播放列表 - https://www.youtube.com/playlist?list=PL-HkokgcYrl5SrKYeVo28JA4OMPbslhA8🚀 曾经希望你能提取成千上万的 YouTu...
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1341892739289321583)** (1 条消息): 

> `Torchtune Roadmap, PyTorch Roadmaps` 


- **2025 年 H1 Torchtune Roadmap 发布**：今年上半年的官方 **Torchtune roadmap** 已发布在 [PyTorch dev-discuss](https://drive.google.com/file/d/1mKENBMrdMzMQQG1kn43Il64qWPB9uP21/view) 上。该文档概述了 Torchtune 在此期间计划的核心方向和项目。
   - 鼓励成员们查看该 roadmap，因为它详细说明了对 Torchtune 开发至关重要的 **关键计划 (key initiatives)** 和策略。
- **PyTorch Roadmaps 全面概览**：各种项目的完整 **PyTorch roadmaps** 也可以在 [dev-discuss](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794) 上访问。此次发布展示了本半年整个 PyTorch 平台上的一系列令人兴奋的进展和正在进行的工作。
   - 这一更广泛的概览展示了 PyTorch 团队在创新和推进技术方面的协作努力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://drive.google.com/file/d/1mKENBMrdMzMQQG1kn43Il64qWPB9uP21/view">[PUBLIC] Torchtune - H1 2025 Roadmap.pdf</a>: 未找到描述</li><li><a href="https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794">Meta PyTorch Team 2025 H1 Roadmaps</a>: PyTorch 社区，Meta 团队很高兴能提供我们的 2025 H1 roadmaps。我们以半年为单位进行计划，并针对我们在 Meta 以及整个... 为用户所做的事情进行全局优化。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1341825832188121161)** (43 messages🔥): 

> `使用 packing 时的 VRAM 需求、Roadmap 更新、新兴 Attention 技术、LLM 剪枝策略、奇异 Transformer 架构` 


- **使用 packed sequences 时 VRAM 需求激增**：当在数据集以 **max_tokens** 长度使用 packing 时，**VRAM 需求**会大幅增加，导致在 **16K 序列长度**下出现 *out-of-memory* (OOM) 错误。
   - 一位用户指出，当 packing 设置为 false 时，显存占用仅为 **30GB**，这展示了资源需求的巨大差异。
- **Roadmap 已发布至 PyTorch dev-discuss**：PyTorch 的 Roadmap 已在 [Google Drive](https://drive.google.com/file/d/1mKENBMrdMzMQQG1kn43Il64qWPB9uP21/view) 上共享，重点强调了即将到来的会议截止日期。
   - 尽管该规划仍在完善中，但已收到积极反馈，并*承诺持续改进*。
- **征求关于奇异 Attention 机制的意见**：讨论集中在**奇异 Transformer 技术**（如 *sparse attention* 和 *attention compression*）的优先级上，这些技术可以提高**序列扩展（sequence scaling）的效率**。
   - 研究人员的贡献表明，虽然存在兴趣，但由于现有方法论的原因，对于集成新技术研究仍持保留意见。
- **大语言模型的剪枝技术**：受近期关于*剪枝替代方案*论文的启发，一种支持 LLM (Large Language Models) **宽度和深度剪枝**的新 recipe 正在开发中。
   - 该方法可以实现模型的显著压缩，在无需完全重新训练的情况下提高资源利用率。
- **Roadmap 目标的澄清**：记录了关于 **KR2.4** 的反馈，强调其评估中缺乏明确的 state-of-the-art (SOTA) 示例，例如 *Codestral* 和 *Jamba*。
   - Roadmap 的目标强调了对长期创新的关注，同时优先处理核心任务，反映了随领域发展而调整的意图。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.14679">Compact Language Models via Pruning and Knowledge Distillation</a>：针对不同部署规模和尺寸的大语言模型（LLM）目前是通过从头开始训练每个变体来生产的；这极其耗费计算资源。在本文中，我们投入...</li><li><a href="https://github.com/pytorch/torchtune/issues/2392">`torch._inductor.exc.LoweringException: NoValidChoicesError` using torch 2.6.0 · Issue #2392 · pytorch/torchtune</a>：错误 [rank0]: raise NoValidChoicesError( [rank0]: torch._inductor.exc.LoweringException: NoValidChoicesError: No choices to select, please consider adding ATEN into max_autotune_gemm_backends conf...</li><li><a href="https://github.com/pytorch/torchtune/blob/e6cba25">GitHub - pytorch/torchtune at e6cba2532d51a53936c7646bd4cdaa6b2b57ed66</a>：PyTorch 原生后训练库。通过在 GitHub 上创建账号为 pytorch/torchtune 开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/e6cba2532d51a53936c7646bd4cdaa6b2b57ed66/torchtune/modules/attention_utils.py#L35">torchtune/torchtune/modules/attention_utils.py at e6cba2532d51a53936c7646bd4cdaa6b2b57ed66 · pytorch/torchtune</a>：PyTorch 原生后训练库。通过在 GitHub 上创建账号为 pytorch/torchtune 开发做出贡献。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1341836804399038566)** (15 条消息🔥): 

> `用于 Online DPO 的 Judge Framework，AdamWScheduleFree 作为默认优化器，Pruning & Checkpointing 工具，Torchtune 与 Gymnasium 的集成，用于 LLM 的 Intercode` 


- **请求关于 Judge Framework RFC 的反馈**：一位成员正在寻求关于其[用于 Judge Framework 实现的 RFC](https://github.com/pytorch/torchtune/issues/2413)的反馈，该框架旨在用于 Online DPO，如果合理的话，计划贡献到 dev 分支。
   - [TRL Judges 文档](https://trl-docs.com)在概念上支持用于 RLHF 方法的多个 Judge。
- **AdamWScheduleFree 可能作为优化器**：讨论了 **AdamWScheduleFree** 作为 **llama3.1 8B DPO** 默认优化器的潜力，并在 **2 个节点、16 块 GPU** 上进行了测试。
   - 针对之前 FSDP 的问题提出了一个变通方案，需要对 full-dpo Python 脚本进行调整。
- **Pull Request 中的 Pruning 和 Checkpointer 工具**：一位成员强调了[关于 Checkpointer 工具的 Pull Request](https://github.com/joecummings/torchtune/pull/2)，其中包含一个获取给定目录中最新 Checkpoint 的功能。
   - 重点在于审查该贡献，以确保其与现有工具保持一致。
- **质疑 Gymnasium 是否适合 LLM 的 RL**：有人询问了关于 **Torchtune** 与 **Gymnasium** 集成的进展，引发了关于其与 LLM 兼容性的讨论。
   - 讨论中提出了对 Gymnasium 设计的担忧，认为其无法很好地适应 LLM 的独特需求，特别是在环境的 Action 和 Observation 方面。
- **探索用于 LLM 集成的 Intercode**：成员们探讨了使用 [Intercode](https://github.com/princeton-nlp/intercode) 来增强适合 LLM 的 RL 任务的可能性，并对其接口的有效性提出了疑问。
   - 对话显示了对在 LLM 项目中结合类 Gym 接口的怀疑，并认识到该领域需要进一步开发。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ysymyth.github.io)">未找到标题</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtune/issues/2413">[RFC] Judge Framework and Online DPO · Issue #2413 · pytorch/torchtune</a>：TRL 拥有多个不同 Judge 的概念，可用于各种 Online RLHF 类型的方法，请参阅 TRL Judges 文档。作为起点，我们可以先实现一个 Pairwise Judge...</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_1/8B_full_dpo.yaml#L64-L72">torchtune/recipes/configs/llama3_1/8B_full_dpo.yaml at main · pytorch/torchtune</a>：PyTorch 原生后训练库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/princeton-nlp/intercode">GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&amp;B] InterCode 基准测试代码库 https://arxiv.org/abs/2306.14898</a>：[NeurIPS 2023 D&amp;B] InterCode 基准测试代码库 https://arxiv.org/abs/2306.14898 - princeton-nlp/intercode</li><li><a href="https://github.com/joecummings/torchtune/pull/2">feat: 为 checkpointer utils 添加 get_latest_checkpoint，由 bogdansalyp 提交 · Pull Request #2 · joecummings/torchtune</a>：添加了 get_latest_checkpoint """ 返回给定目录中最新的 checkpoint。pattern 参数是一个正则表达式，用于匹配 epoch 编号...
</li>
</ul>

</div>

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1341848371287162951)** (4 messages): 

> `Multi-step PPO, Tool Learning, Reward Shaping, StepTool Framework, UltraScale Playbook` 


- **多步 PPO 方法的探索**：一位用户询问了关于 **multi-step PPO** 的论文，这涉及到对 LLM 的多次连续调用，且奖励仅在多次交互后才进行评估。
   - 他们建议在更广泛的 Tool Learning 和 Reward Shaping 领域进行研究。
- **用于 Tool Learning 的 StepTool 框架**：分享的一篇核心论文讨论了 **StepTool**，这是一个新的步粒度强化学习框架，增强了 LLM 的多步工具使用能力，并详细介绍了其 **Step-grained Reward Shaping** 和 **Step-grained Optimization** 组件。
   - 论文强调在 Tool Learning 的背景下，需要考虑多步语境中决策的复杂性。
- **Hugging Face UltraScale Playbook 发布**：一位用户分享了托管在 Hugging Face 上的 [UltraScale Playbook 链接](https://huggingface.co/spaces/nanotron/ultrascale-playbook)，并称其**令人耳目一新**。
   - 该 Playbook 可能旨在指导用户在实际框架中扩展模型的使用。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>: no description found</li><li><a href="https://arxiv.org/abs/2410.07745">StepTool: Enhancing Multi-Step Tool Usage in LLMs through Step-Grained Reinforcement Learning</a>: Despite powerful text generation capabilities, large language models (LLMs) still need to learn how to utilize external tools to solve complex tasks, a process known as tool learning. Existing methods...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1341832770216919090)** (49 messages🔥): 

> `Baseten Series C funding, Mastra JS agent framework, Arize AI Series C funding, Lambda $480M Series D, OpenAI's growing user base` 


- **Baseten 获得 7500 万美元 C 轮融资**：Baseten 宣布成功完成 7500 万美元的 C 轮融资，由 @IVP 和 @sparkcapital 领投，强调 2025 年将是 AI 技术的推理之年。
   - 来自 @01Advisors 的 Dick Costolo 和 Adam Bain 等新投资者也加入了本轮融资，突显了 Baseten 在 AI 领域的增长和潜力。
- **Mastra 开源 AI Agent 框架**：开源项目 Mastra 提供了一个 JavaScript SDK，用于在 Vercel 的 AI SDK 之上构建 AI Agent，专注于易用性以及与工作流的集成。
   - 开发者对 Mastra Agent 执行复杂任务的能力表示关注，例如访问第三方 API 和自定义函数。
- **Arize AI 筹集 7000 万美元 C 轮融资**：Arize AI 获得了 7000 万美元的 C 轮融资，以增强生成式模型和决策模型中的 AI 评估与可观测性。
   - 他们的使命是确保 AI Agent 能够在大规模环境下可靠运行，应对 AI 技术新发展带来的挑战。
- **Lambda 获得 4.8 亿美元 D 轮融资**：Lambda 宣布了由 Andra Capital 和 SGW 领投的 4.8 亿美元 D 轮融资，展示了该公司在 AI 计算资源方面的增长。
   - 凭借这笔资金，Lambda 旨在加强其作为专为 AI 需求和能力开发的云服务的地位。
- **OpenAI 用户突破 4 亿**：OpenAI 最近报告称 ChatGPT 的周活跃用户超过 4 亿，在不到三个月的时间内实现了 33% 的显著增长。
   - 即将推出的 GPT-5 承诺为免费用户提供无限次使用，并有望统一现有模型，从而加剧 AI 领域的竞争。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://mastra.ai/docs/agents/00-overview">创建和调用 Agents | Agent 文档 | Mastra</a>：未找到描述</li><li><a href="https://x.com/huybery/status/1892628963878486233">来自 Binyuan Hui (@huybery) 的推文</a>：&lt;think&gt;…&lt;/think&gt;Binyuan 正在憋大招（cooking）…</li><li><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - nanotron 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/drjimfan/status/1892404919480832259?s=46">来自 Jim Fan (@DrJimFan) 的推文</a>：我最近见过的最酷的自主编程 Agent：使用 AI 编写更好的 CUDA kernel 来加速 AI。AutoML 回归了！利用计算资源能做的杠杆率最高的事情是...</li><li><a href="https://x.com/Figure_robot/status/1892577871366939087">来自 Figure (@Figure_robot) 的推文</a>：认识一下 Helix，我们内部开发的像人类一样推理的 AI。如果没有能力的跨越式发展，机器人将无法进入家庭。我们的机器人现在几乎可以处理任何家用物品：</li><li><a href="https://x.com/stephenbalaban/status/1892275552171737220?s=46">来自 stephen balaban (@stephenbalaban) 的推文</a>：Lambda 是为 AI 时代设计的云。今天，我们宣布了由 Andra Capital 和 SGW 共同领投的 4.8 亿美元 D 轮融资，NVIDIA、Andrej Karpathy、In-Q-Tel、ARK Invest 等参投。</li><li><a href="https://x.com/bingxu_/status/1892405811596710392">来自 Bing Xu (@bingxu_) 的推文</a>：我在手机上快速看了一下他们的报告，有几个误导性的地方：1. Torch C++ 代码不是 CUDA kernel，它在底层调用的是 CUDNN。2. 突出显示的示例 Conv3D GroupNorm，conv...</li><li><a href="https://x.com/loubnabenallal1/status/1892278622104215894?s=46">来自 Loubna Ben Allal (@LoubnaBenAllal1) 的推文</a>：nanotron 团队刚刚发布了 The Ultra-Scale PlayBook，包含了关于从小规模到大规模 LLM 预训练所需了解的一切 https://huggingface.co/spaces/nanotron/ultrascale-playbook</li><li><a href="https://x.com/leonardtang_/status/1892243653071908949">来自 Leonard Tang (@leonardtang_) 的推文</a>：首先是预训练扩展（pre-training scaling）；然后是推理时扩展（inference-time scaling）。现在轮到评判时扩展（judge-time scaling）了。尽管 AI 通过扩展推理时计算取得了进展，但在开放式、不可验证的...</li><li><a href="https://www.youtube.com/watch?v=L89GzWEILkM">AI Engineer Summit 2025 - AI 领导力 (第一天)</a>：演讲安排（均为 EST 时间）：9:00am - 开幕 9:07AM - 超越共识：在 2025 年探索 AI 前沿 - Lux Capital 的 Grace Isford 9:28AM - Ho...</li><li><a href="https://x.com/stefania_druga/status/1892669203657736600?s=46">来自 Stefania Druga (@Stefania_druga) 的推文</a>：很棒的列表！在可解释性（interpretability）方面还有很多工作要做。引用 swyx 🗽 NYC (@aiDotEngineer) (@swyx)：第一次看到 @AnthropicAI 这样列出其首要任务，更多地关注机械可解释性（mechinterp）...</li><li><a href="https://x.com/stephenbalaban/status/1892403855817859079">来自 stephen balaban (@stephenbalaban) 的推文</a>：这是 Lambda 的第一台超级计算机。建于 2015 年，用于运行 Dreamscope 风格迁移应用。它由 32 块 NVIDIA GTX 980 组成。当时我们因为每月 4 万美元的云账单快没钱了。我们称它为 Deep B...</li><li><a href="https://x.com/togethercompute/status/1892609235789422724">来自 Together AI (@togethercompute) 的推文</a>：📣 今天我们宣布了由 @generalcatalyst 领投、@p7ventures 共同领投的 3.05 亿美元 B 轮融资，参与方包括一组杰出的全球机构和战略投资者...</li><li><a href="https://arize.com/blog/arize-ai-raises-70m-series-c-to-build-the-gold-standard-for-ai-evaluation-observability/">Arize AI 筹集 7000 万美元 C 轮融资，旨在打造 AI 评估与可观测性的黄金标准</a>：在我们的 C 轮融资公告中了解我们如何塑造值得信赖的 LLMs 和 AI Agents 的未来，以及 Arize 的下一步计划。</li><li><a href="https://x.com/nutlope/status/1892619157662806272?s=46">来自 Hassan (@nutlope) 的推文</a>：很高兴分享我们以 33 亿美元的估值筹集了 3.05 亿美元！见证所有的增长真是太棒了——开发者达到 45 万，支持超过 200 个模型！另外，我们正在招聘！引用 Together AI (@togetherc...</li><li><a href="https://x.com/dhravyashah/status/1892363590671233255?s=46">来自 Dhravya Shah (@DhravyaShah) 的推文</a>：介绍 apple-mcp - http://git.new/apple-mcp。只需一个简单的命令即可让 LLMs 访问一堆 Apple 原生工具，如：联系人、备忘录、iMessage 等（即将推出）。只需将其添加到您的 Claude 桌面端...</li><li><a href="https://x.com/yoheinakajima/status/1892257339400737087?s=46">来自 Yohei (@yoheinakajima) 的推文</a>：我在私下交流中说过，所以我也在这里说一下……“更好的记忆”是我们需要获得真正更好的 Agents 的最后一把钥匙，2025 年我们将看到更多这方面的进展。我们已经有了强大的推理能力、用于...</li><li><a href="https://x.com/skalskip92/status/1892233630577000820?s">

=46">来自 SkalskiP (@skalskip92) 的推文</a>：这个人采用了我的足球 AI 项目，并将其用于自动化的进阶比赛分析；首先专注于直塞球。引用 SkalskiP (@skalskip92) 的话：足球 AI 代码终于开源了 - 球员 d...</li><li><a href="https://x.com/bradlightcap/status/1892579908179882057?s=46">来自 Brad Lightcap (@bradlightcap) 的推文</a>：ChatGPT 最近突破了 4 亿 WAU（周活跃用户），我们感到非常荣幸每周能为全球 5% 的人口提供服务。现在有超过 200 万企业用户在工作中使用 ChatGPT，自 o3 mini 发布以来，推理模型 API 的使用量增长了 5 倍...</li><li><a href="https://x.com/basetenco/status/1892259130540179863">来自 Baseten (@basetenco) 的推文</a>：2025 年是推理之年。我们很高兴宣布由 @IVP 和 @sparkcapital 领投的 7500 万美元 C 轮融资，@GreylockVC、@conviction、@basecasevc、@southpkcommons 等也参与了其中...</li><li><a href="https://x.com/thom_wolf/status/1892273133547078036?s=46">来自 Thomas Wolf (@Thom_Wolf) 的推文</a>：经过 6 个多月的筹备并消耗了超过一年的 GPU 计算时间，我们非常激动终于发布了“Ultra-Scale Playbook”。在这里查看：http://hf.co/spaces/nanotron/...</li><li><a href="https://news.ycombinator.com/item?id=43103073">Show HN: Mastra – 开源 JS Agent 框架，由 Gatsby 的开发者打造 | Hacker News</a>：未找到描述</li><li><a href="https://x.com/miru_why/status/1892500715857473777?s=46">来自 miru (@miru_why) 的推文</a>：事实证明，AI CUDA 工程师通过……修改评测脚本实现了 100 倍的加速。引用 main (@main_horse) @miru_why 的话：我认为他们的内核有问题——它似乎“窃取”了...</li><li><a href="https://x.com/giffmana/status/1892510741242036468?s=46">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>：o3-mini-high 在 11 秒内发现了 @SakanaAILabs CUDA 内核的问题。所谓的 150 倍提速其实是个 Bug，实际上慢了 3 倍。我直接把他们的 CUDA 代码复制粘贴到 o3-mini-high 中并询问...
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1341824207218020382)** (11 条消息🔥): 

> `SSE 实现, 调试 Glama 托管模型, Puppeteer 安装问题, Docker 需求, 远程 MCP 功能时间线` 


- **SSE 实现已上线**：一位成员确认他们已成功为自己的项目实现了 **/sse**，可以在特定频道查看。
   - 这一更新突显了 MCP 功能的持续改进。
- **Glama 托管模型调试困扰**：另一位成员分享称，在调试 **Glama 托管模型** 时遇到了 Cursor 找不到工具的问题。
   - *99% 的问题* 归因于 node 路径使用不当以及可能缺失的引号。
- **Puppeteer 安装困惑**：一位新成员寻求关于 **Puppeteer 安装** 的帮助，特别是关于运行 Docker 构建命令的问题。
   - 社区提供了指导，建议导航到正确的父目录，并澄清了 Docker 命令中 `.` 的用途。
- **Docker 基础知识澄清**：确认在使用 **Docker** 之前需要先安装它，一位成员指出找不到该命令。
   - 此外，安装不需要账号，因为 Docker 是免费软件。
- **远程 MCP 时间线咨询**：一位用户询问了 **远程 MCP** 功能的时间线，表达了对其在公司潜在应用的兴趣。
   - 另一位成员回应称，目前已经支持 **SSE** 和 **websocket** 的 MCP 传输方式。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1341837709735493713)** (26 条消息🔥): 

> `Docker 化 MCP 服务器，Sage 对 LLM 提供商的支持，Glama 集成，MCP Python 解释器，MCP 客户端中的 Roots` 


- **如何部署 Docker 化 MCP 服务器**：一位成员分享了一篇博文，详细介绍了部署 Docker 化 **MCP 服务器**的步骤，并强调了跨架构环境设置的挑战。他们指出 **Docker** 可以帮助确保开发环境的一致性。
   - 该博文还指向了一个[参考 MCP 服务器](https://github.com/modelcontextprotocol/server)列表，供希望实现 MCP 功能的开发者参考。
- **Sage LLM 支持查询**：讨论了 **Sage** 何时会支持像 **OpenRouter** 这样额外的 LLM 提供商，并暗示正在等待潜在的 API 更新。目前 **Glama** 已经可以直接集成到 Sage 中。
   - 一位成员在意识到共同的兴趣和目标后，表达了希望将这两个项目更紧密地结合起来的愿望。
- **MCP Python REPL 实现**：一位成员介绍了一个支持 MCP **STDIO** 的简单 **Python REPL** 实现，并分享了他们的 [GitHub 仓库](https://github.com/evalstate/mcp-py-repl)链接。他们还为感兴趣的人提供了最新的镜像。
   - 另一位成员询问了关于 **IPython 支持**的情况，开发者表示添加该支持可能比较简单，并建议在该功能上进行进一步开发。
- **MCP 中的 Matplotlib 支持**：围绕在 MCP 中集成 **matplotlib/pyplot** 支持以渲染图表（类似于 Jupyter）展开了讨论。作者确认 **matplotlib**、**seaborn** 和 **numpy** 已经包含在实现中。
   - 他们提到将图表图像作为 .png 文件返回，并讨论了是否可以直接将其返回给 MCP 客户端。
- **MCP 客户端中的 Roots 使用**：关于在 MCP 中使用 **roots** 以及现有客户端实现的讨论。一位成员指出，使用 MCP 服务器返回文件结果很容易，但对目前在各种客户端中的使用程度表示好奇。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.defang.io/blog/2025/02/18/model-context-protocol">使用 Docker 和 Model Context Protocol 简化 AI 应用到云端的部署 | Defang</a>：mcp</li><li><a href="https://github.com/evalstate/mcp-py-repl">GitHub - evalstate/mcp-py-repl: 一个用于 MCP 的 Python REPL</a>：一个用于 MCP 的 Python REPL。通过在 GitHub 上创建账号来为 evalstate/mcp-py-repl 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1341838736530604102)** (3 条消息): 

> `MAX 25.1 直播，社区会议演讲，Modular 品牌周边` 


- **加入 MAX 25.1 直播！**：明天安排了一场直播，讨论关于 **MAX 25.1** 的一切。你可以[在 LinkedIn 上加入](https://www.linkedin.com/events/introducingmax25-17297704283980902402/theater/)，并通过这个 [Google 表单](https://forms.gle/NkjU6e3n15TRtiMA7)提交问题。
   - *欢迎分享你的问题；我们渴望听到你的想法。*
- **社区会议演讲机会**：下周一的社区会议开放了演讲名额，邀请成员展示他们的项目或关注领域。鼓励感兴趣的参与者联系并表达演讲意愿。
   - *这是一个展示社区内创新工作的绝佳机会。*
- **Modular 时尚的 Patagonia 毛衣**：一位成员称赞了 **Modular 品牌的 Patagonia 毛衣**，对其设计表现出极大的热情。这似乎在社区成员中很受欢迎，展示了他们的品牌自豪感。
   - *它的风格和质量确实引起了关注。*



**提到的链接**：<a href="https://forms.gle/NkjU6e3n15TRtiMA7">Modular 社区问答</a>：未找到描述

  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1341907584952832103)** (33 messages🔥): 

> `原生 Mojo Windows 支持, Slab List 结构讨论, Mojo 与 Python 的比较, AI 计算性能, Mojo 中的底层编程` 


- **原生 Mojo Windows 支持尚不明确**：讨论表明目前可能没有**原生 Mojo Windows 支持**的预定时间表，这主要是由于在 **Windows** 上运行 **AI** 集群的相关成本过高。
   - *nix 操作系统在计算工作中更受青睐，许多人正在使用云端 Linux 平台而非 Windows，因此这并非短期内的优先事项。
- **理解 Slab List 结构**：一名成员将 **slab list** 定义为一种有效的数据结构，其运行方式非常类似于 `LinkedList[InlineArray[T, N]]`，侧重于简单性和高效的内存管理。
   - 他们强调，使用这种结构可以使各种操作获得 **O(1)** 性能，由于缓存效率的提高，其迭代速度比传统的链表更快。
- **Mojo 与 Python 的关系**：大家达成共识，认为 **Mojo** 可以被看作是一种源自 Python 但性能接近 C/C++/Rust 的语言，旨在实现类似于 C++ 对 C 的未来兼容性。
   - 一位成员总结道，Mojo 先进的类型系统允许提供**受 Python 启发**的体验，并建议它可能会吸引现有的 **Nim** 用户。
- **AI 计算性能比较**：成员们注意到，一旦 **AI** 计算任务被推送到 GPU，性能差异就会变得微乎其微，而在许多 CPU 任务中，Mojo 的表现可能显著优于 Python。
   - 在 ARM 架构上，Mojo 的速度甚至可以优于传统的**纯 Python**，尽管据说通过 WSL 使用 Windows 会引入一些开销。
- **使用 Mojo 进行底层编程的便利性**：一位成员表示，与 C/C++ 相比，在 **Mojo** 中处理底层任务更容易，这表明 Mojo 的设计有效地促进了硬件利用。
   - 他们建议 Mojo 在进行底层编码时不必严格遵守 Python 的语法，因为对于许多应用来说，具备运行 Python 脚本的强大能力就足够了。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/nic">nic - 概览</a>：首席 Trolling 官。nic 拥有 58 个代码库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/nickziv/libslablist">GitHub - nickziv/libslablist: slab list 是一种内存效率极高的对数时间数据结构。</a>：slab list 是一种内存效率极高的对数时间数据结构。 - nickziv/libslablist
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1341830882704625674)** (2 messages): 

> `LlamaCloud EU, LlamaParse 升级` 


- **LlamaCloud EU 发布以满足数据合规性**：我们宣布了 [LlamaCloud EU](https://t.co/HTt2pob88p) 的早期访问，这是一款新的 SaaS 产品，在 EU 司法管辖区内提供具有完整数据驻留权的知识管理。
   - 此次发布旨在消除寻求合规解决方案的欧洲公司的重大障碍，重点关注**安全性**和**数据驻留**。
- **LlamaParse 推出新功能**：[LlamaParse](https://t.co/ux3gsvDIeW) 引入了新的解析模式：Fast、Balanced 和 Premium，以有效地满足不同的文档解析需求。
   - 这些升级旨在解决现有的**文档解析挑战**，从而在处理不同类型的文档时提供更多的通用性。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1341864480254201868)** (21 条消息🔥): 

> `Agent 工作流中的循环、处理多个工具调用、Redis 并行处理最佳实践、LlamaCloud 系统故障、区块链进展` 


- **Agent 工作流中的移交（Handoff）问题**：一位开发者在多 Agent 工作流场景中遇到问题，LLM 返回 *'I am handing off to AgentXYZ'* 而不是执行工具调用。
   - 另一位用户询问是否应该将 **handoff 规则包含在系统消息（system message）中**以明确此行为。
- **确保 Redis 中的并行处理**：一位用户正在寻求策略，以便在避免 Redis 竞态条件的同时，有效地运行 **1000 个并行批次**来持久化摘要索引（summary index）。
   - 他们将评论的 Embeddings 存储在 Redis 命名空间中，并担心潜在的 **键冲突（key collisions）和资源限制**。
- **LlamaCloud 服务状态讨论**：用户报告了 LlamaCloud 服务的潜在问题，尽管状态页面显示 *所有系统运行正常*。
   - 团队成员确认他们正在调查情况，一位用户幽默地表示已经有足够的 *scamcoin*（诈骗币）活动了。
- **对区块链项目的担忧**：关于在 **Solana** 上创建代币可能性的咨询显示，社区认为任何此类声明都是 **诈骗（scams）**。
   - 讨论还涉及了参与“诈骗币”项目的更广泛影响。
- **LLM 工具调用的挑战**：一位用户对 LLM 响应未能执行预期操作表示沮丧，重点在于需要工具调用（tool calls）而不是通用响应。
   - 他们指出担心破坏现有的 handoff 提示词，该提示词似乎是为 **多功能工具交互** 设计的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llamaindex.statuspage.io">LlamaIndex Status</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/#human-in-the-loop">AgentWorkflow 基础介绍 - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1341920240250126366)** (1 条消息): 

> `AI 的下一阶段、数据运营趋势` 


- **探索 AI 的下一阶段和数据运营**：一位成员分享了一篇讨论 *AI 下一阶段* 和新兴 **数据运营趋势** 的帖子，这些趋势可能会对行业产生重大影响。
   - 有兴趣了解这些转变的人可以点击 [这里](https://open.substack.com/pub/procurefyi/p/the-end-of-big-dumb-ai-data?r=223ajc&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false) 阅读题为 *《大而无当的 AI 数据终结》* 的文章。
- **关于 AI 数据管理的见解**：讨论强调了从 **传统 AI 数据管理** 向更高效、更具适应性的方法论转变的潜力。
   - 该成员强调，组织需要重新思考其数据策略，以跟上不断发展的技术格局。


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1341846122133721098)** (2 条消息): 

> `频道创建请求、颜色更改公告` 


- **新频道创建请求**：一位成员请求创建一个特定频道，并表示他们也应该能在那里发送截图。
   - 这一请求的语气比较随意友好，并配有爱心表情符号。 
- **成员颜色更改更新**：另一位成员宣布了他们的颜色更改，简单地说道：*"现在我是粉色的了。"*
   - 这一变化可能为 Discord 社区增添了视觉活力。


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1341928753395138601)** (3 条消息): 

> `利润分享机会、没有咖啡的世界的影响` 


- **寻找利润分享合作伙伴**：一位成员正在寻找年龄在 **25-50 岁** 之间、愿意分享身份并分享 **$100-1500** 利润的人。
   - 这可能预示着一个基于共同利益的潜在商业或投资机会。
- **关于咖啡缺失的文章请求**：一位成员请求写一篇关于没有 **咖啡** 的世界所产生影响的文章，强调了其文化和经济意义。
   - 这一讨论暗示了对咖啡不再供应的假设情景下生活方式变化的关注。


  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1341921219322318903)** (13 messages🔥): 

> `Identity Sharing in Collaboration, Concerns about Personal Information, Communication Clarity in Forums` 


- **身份共享提案引发辩论**：一名用户提出了一个涉及身份共享以获取利润（范围在 **$100-1500**）的协作机会，并强调了 **25-50** 岁的年龄范围。
   - 这引发了对该安排中身份盗用风险的担忧，且未提供 **website** 或相关文档。
- **警惕个人信息**：一名成员提醒小组，并非所有人都愿意在公共论坛披露 **personally identifiable information**（个人身份信息），并强调该频道专注于 **Cohere 相关项目**。
   - 该提醒强调了在讨论潜在协作时尊重个人隐私的重要性。
- **呼吁沟通清晰**：针对书面沟通中的歧义提出了担忧，并建议使用 **更清晰的写作** 以防止误解。
   - 成员们强调了改进沟通对于促进小组内 **积极协作** 的重要性。
- **对项目细节的怀疑**：由于缺乏信息，一名用户对最初的提案表示怀疑，理由是缺少 **website**、文档或清晰的项目描述。
   - 这种怀疑凸显了在讨论新的协作机会时对透明度的需求。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1341832890568282164)** (16 messages🔥): 

> `Jamba API usage, PHP integration with Jamba, Response formatting issues, Removing special characters, Using AJAX for API calls` 


- **Jamba API 入门**：一名用户寻求使用 **Jamba API** 的帮助，并分享了进行 API 调用的代码，特别提到了语法方面的困难。
   - 另一名成员提供了使用 API 的详细大纲，包括必要的参数和 headers。
- **理解 API 响应**：讨论了 API 的输出，特别是它包含可能使处理复杂化的 **escape characters**（转义字符）。
   - 成员们确认响应格式可能会根据所使用的 **language** 而有所不同，强调需要进行额外处理。
- **Jamba API 集成的 PHP 细节**：一名用户提到正在使用 **Symfony 和 PHP**，并表示需要将 API 响应转换为可用格式。
   - 建议向其他成员寻求关于 PHP 输出中特殊字符处理的帮助。
- **使用 AJAX 改进 API 输出**：一名用户建议利用 **AJAX** 来增强 API 响应处理，但指出结果仍然不一致。
   - 有成员确认 **Jamba chat window** 中的输出格式不同，这可能会影响结果的呈现方式。
- **PHP 挑战的协作**：成员们指出，熟悉 PHP 的其他用户可能会提供帮助，特别是在有效处理输出方面。
   - 一名成员直接联系了另一名成员，寻求在该主题上的潜在指导。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.ai21.com/reference">Jamba 1.5</a>: Jamba-1.5 指令遵循对话模型</li><li><a href="https://docs.ai21.com/reference/jamba-15-api-ref">Jamba 1.5</a>: Jamba-1.5 指令遵循对话模型
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1341888954336808980)** (6 messages): 

> `不同硬件上的模型性能、Int8 量化问题、Torch 与 Tinygrad 的速度测试、BEAM 优化、新的 PyTorch 频道` 


- **GeForce 850M vs RTX 4070 性能对比**：测试显示，一台**旧的 GeForce 850M** 在 Brave 浏览器上运行 **8 秒**后达到 **3 tok/s**，而 **RTX 4070** 在 Windows 11 Chrome 上仅需 **1.9 秒**即可达到 **12 tok/s**。
   - 然而，由于各种**计算成本和数值刚性（numerical stiffness）**，模型的整体可用性仍然受限。
- **Int8 量化的挑战**：有人指出 **Int8 量化**方法可能需要改进，因为在使用 **Int8Linear** 时，模型在几百个 token 后偶尔会“跑偏（off rails）”。
   - 建议通过 *Direct messages 或 GitHub* 讨论进行更集中的 **tinychat** 开发交流。
- **速度测试结果显示性能互有胜负**：最近的速度测试表明，在 **2048x2048** 张量上 **torch** 优于 **tinygrad**，torch 为 **0.22 ms**，而 tinygrad 为 **0.42 ms**。
   - 然而，在 **4096x4096** 上，性能更加接近，tinygrad 仅比 torch **慢 1.08 倍**，目前正在继续调查性能差异的原因。
- **利用 BEAM 优化性能**：进一步的见解表明，增加 **BEAM** 值可能会缓解性能问题，测试显示在 torch 中使用 **BEAM=10** 处理 **2048x2048** 张量时为 **0.21 ms**。
   - 性能在不同张量大小下保持相对一致，突显了通过更高 **BEAM** 设置进行优化的潜力。
- **George Hotz 宣布新的 PyTorch 频道**：为了讨论与 **PyTorch** 相关的内容，创建了一个新频道，显示了社区的参与度。
   - 随着用户贡献的增长，这一新增频道预计将促进更专业的讨论。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1342172399084503213)** (4 messages): 

> `tinygrad 中的操作、BLOCK 操作文档、代码库搜索策略` 


- **关于 BLOCK 操作的查询**：一名成员请求有关 `BLOCK`、`BLOCKSTART`、`BLOCKFORK` 和 `BLOCKEND` 操作的文档，以了解它们代表和存储的内容。
   - 该问题突显了 tinygrad 项目中需要更清晰的文档或指南。
- **共享 GitHub 资源**：作为对查询的回应，一名成员链接到了包含 `linearize.py` 的 [GitHub 仓库](https://github.com/tinygrad/tinygrad/blob/master/tinygrad%2Fcodegen%2Flinearize.py)，这可能与 BLOCK 操作有关。
   - 该资源可以作为理解这些操作的实现和用法的起点。
- **在代码库中搜索文档**：一名成员建议，寻找信息的一个有用步骤是搜索整个代码库中的相关引用。
   - 这种方法强调了在 tinygrad 框架内利用现有资源进行自主学习的重要性。



**提到的链接**：<a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad%2Fcodegen%2Flinearize.py">tinygrad/tinygrad/codegen/linearize.py at master · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1341828771438923868)** (10 messages🔥): 

> `系统消息术语、模型指令、图像粘贴功能、Nomic 实现问题` 


- **对系统消息术语的困惑**：一名成员澄清说，UI 中现在使用“系统消息（system message）”一词，表明命名惯例发生了变化。
   - 另一位参与者肯定地表示，在操作这些系统时，旧习惯很难改变。
- **在系统消息中使用指令**：提到可以在“系统消息”中使用纯英文指令，大多数模型都会遵循这些命令。
   - 一些成员对这一过程的简便性表示怀疑，询问使用 Jinja 或 JSON 代码是否更有效。
- **GPT4All 中的图像处理限制**：一名成员询问是否可以像其他 AI 平台一样直接将图像粘贴到文本栏中。
   - 澄清说 **GPT4All** 无法处理图像，建议使用外部软件执行此类任务。
- **关于 Nomic 和 NOIMC v2 发布的讨论**：一名成员对 **NOIMC v2** 的实现表示困惑，质疑为什么它看起来实现得不正确。
   - 另一名成员幽默地寻求确认自己是否在 **Nomic** 频道，表达了他们的挫败感。

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1341943677773418506)** (4 messages): 

> `2024 LLM Agents 课程，测验存档访问，DSPy 兴趣，讲座可用性` 


- **考虑观看 2024 LLM Agents 课程**：一位成员建议，虽然不一定要旁听 **Fall 2024 课程**，但对于想要深入理解的人来说是有益的，特别是对本学期教学大纲中缺失的 **DSPy** 感兴趣的人。
   - 他们提供了该课程讲座的 [YouTube 播放列表](https://youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&feature=shared)。
- **视频和测验消失**：一位成员对当前大纲中 **视频和测验** 的消失表示困惑，这阻碍了他们的补课进度。
   - 作为回应，另一位成员链接到了 Fall 2024 课程的 **测验存档**，可以在 [这里](https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit?usp=sharing) 找到。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&feature=shared,">LLM Agents MOOC</a>: Large Language Model Agents MOOC F24</li><li><a href="https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit?usp=sharing)">Quizzes Archive - LLM Agents MOOC</a>: 注意：正确答案在黑色方框内（黑底黑字）。用光标高亮方框即可显示正确答案（如果难以查看，也可以将文本复制到新浏览器中...
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1341942110936498176)** (3 messages): 

> `测验访问，MOOC 资源` 


- **获取测验的帮助**：一位成员询问如何获取周末的 **测验 1 和 2**，因为他加入课程较晚。
   - 另一位成员回应称，可以在 MOOC 的 [页面](https://llmagents-learning.org/sp25) 或 [公告页面](https://llmagents-learning.org/f24) 找到测验。
- **MOOC 课程结业通知**：注意到课程现已结束，但视频讲座在教学大纲中仍可访问。
   - *所有证书已发放*，并鼓励学生报名参加 [Spring 2025 迭代版本](https://llmagents-learning.org/sp25)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/sp25">Advanced Large Language Model Agents MOOC</a>: MOOC, Spring 2025</li><li><a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>: MOOC, Fall 2024
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1342013410719567882)** (1 messages): 

> `HaizeLabs Judge Compute, Qwen/Qwen2.5-VL-7B-Instruct, LLM-AggreFact 分数` 


- **受 HaizeLabs Judge Compute 启发**：一位成员运行了与 HaizeLabs Judge Compute 相同的数据集，并使用 **Qwen/Qwen2.5-VL-7B-Instruct** 模型获得了不同的分数。
   - 分数范围从 2-stage 优化的 **60%-70%** 到 mipro2 的 **88.50%**，展示了令人印象深刻的性能指标。
- **LLM-AggreFact 分数详情**：使用各种方法得到的 **LLM-AggreFact** 分数报告如下：**labeled fewshots 81.25%**，**bootstrap random 84.50%**，**copro 84%**。
   - 这表明在不同评估方法下均具有竞争力的表现，暗示了该模型评分能力的鲁棒性。
- **在 GitHub 上分享源码**：所有与评估相关的源代码已在 [GitHub Gist](https://gist.github.com/fullstackwebdev/fa4934fb4669cfc3e8c6ced950ea7a22) 中分享。
   - 可以访问名为 **LLM-AggreFact_DSPy** 的项目，以进一步了解评估中使用的具体方法。



**提及的链接**: <a href="https://gist.github.com/fullstackwebdev/fa4934fb4669cfc3e8c6ced950ea7a22">LLM-AggreFact_DSPy</a>: GitHub Gist: 立即分享代码、笔记和代码片段。

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1341817003324739586)** (5 条消息): 

> `Judge-Time Scaling, Personal Voice Identity Manager, DSPy 会话历史, 消息模板导出` 


- **Verdict 引领 Judge-Time Scaling**：Leonard Tang 宣布发布 [Verdict](https://x.com/leonardtang_/status/1892243653071908949)，这是一个旨在扩展 judge-time compute（评判时计算）的库，并强调 AI 目前的可靠性问题源于评估而非生成。
   - Tang 指出，近期 AI 的创新主要集中在 **pre-training** 和 **inference-time scaling**，而改进评估将成为该领域的下一个重大突破。
- **Personal Voice Identity Manager 的潜力**：一位成员对 **Verdict** 库表示了极大的热情，认为它与他们构思的 **Personal Voice Identity Manager** 概念完美契合。
   - 这表明人们有兴趣探索增强的评估技术如何使 AI 应用中的用户身份管理受益。
- **关于 DSPy 会话历史的澄清**：一位成员寻求核实 DSPy 是否会自动将会话历史注入调用中，这表明在深入研究实现细节之前的一种谨慎态度。
   - 这突显了在不覆盖先前上下文的情况下管理 AI 交互时可能存在的复杂性担忧。
- **将 Prompt 导出为消息模板**：分享了一份 FAQ，详细介绍了如何使用带有 `dspy.ChatAdapter()` 的 Python 代码片段，将程序中的 prompt 冻结并导出为消息模板。
   - 文中提到，虽然这种方法很有用，但会导致控制流逻辑丢失，建议使用 `program.save()` 或 `program.dump_state()` 等替代方案进行完整导出。



**提及的链接**：<a href="https://x.com/leonardtang_/status/1892243653071908949">来自 Leonard Tang (@leonardtang_) 的推文</a>：首先是 pre-training scaling；接着是 inference-time scaling。现在轮到 judge-time scaling 了。尽管 AI 通过扩展推理时计算取得了进展，但在开放式、非验证性的场景中，AI 仍然不可靠...

  

---


---


{% else %}


> 为了便于邮件阅读，完整的频道逐项细分已被截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}