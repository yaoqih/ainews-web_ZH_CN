---
companies:
- meta-ai-fair
- openai
- allenai
- google-deepmind
date: '2024-09-26T22:52:11.195444Z'
description: '**Meta AI** 发布了 **Llama 3.2** 模型，包括 **1B、3B 纯文本**版本以及 **11B、90B 视觉**版本。这些模型具备
  **128K 令牌（token）的上下文长度**，并采用适配器层（adapter layers）来实现图像与文本的集成。其性能超越了 **Gemma 2** 和
  **Phi 3.5-mini** 等竞争对手，并已获得 **AWS、Azure 和 Google Cloud** 等主流平台的支持。


  **OpenAI 首席技术官 (CTO) Mira Murati** 宣布离职。**Allen AI** 发布了 **Molmo**，这是一个开源多模态模型系列，其表现优于部分专有（闭源）系统。**谷歌**改进了
  **Gemini 1.5**，推出了 Flash 和 Pro 模型。**Meta** 展示了 **Project Orion AR 眼镜**，并暗示将推出售价
  300 美元的 **Quest 3S**。此外，相关讨论还涵盖了多模态模型的新基准、模型优化以及 AI 安全与对齐等议题。'
id: ecf9bc48-0db1-4ed2-b326-536bd9bc10ee
models:
- llama-3-2
- llama-3
- gemma-2
- phi-3-5-mini
- claude-3-haiku
- gpt-4o-mini
- molmo
- gemini-1.5
- gemini
original_slug: ainews-not-much-happened-today-2295
people:
- mira-murati
- demis-hassabis
- ylecun
- sama
title: 今天没发生什么。
topics:
- multimodality
- model-optimization
- benchmarks
- ai-safety
- model-distillation
- pruning
- adapter-layers
- open-source-models
- performance
- context-windows
---

<!-- buttondown-editor-mode: plaintext -->**一个安静的日子就是你所需要的。**

> 2024年9月25日至9月26日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discord 社区（**224** 个频道，**3282** 条消息）。预计节省阅读时间（以 200wpm 计算）：**342 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

许多人仍在消化 OpenAI 突如其来的管理层变动。[Sama](https://x.com/sama/status/1839093415226524114) 和 [gdb](https://x.com/gdb/status/1839391073296408577) 都发布了声明。看来 Anthropic 的传闻被推迟了，但与此同时，关于新的 [blueberry 模型传闻](https://x.com/ArtificialAnlys/status/1839333788817702920)才刚刚开始。

---

既然今天是安静的一天，你可以通过**[查看 Weights and Biases 的 RAG++ 课程](https://wandb.me/ainews-course)**来帮助 AINews！我们昨天介绍了它，但忘了包含文本链接。抱歉！

> Swyx：我们昨天初步扫描时漏掉的还有关于响应合成和优化的第 6 章和第 7 章。**特别是第 6 章，正是我们构建 AINews 时必须做的事情**——由于这些技术，你下面看到的一切都是 AI 生成的。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 简报

> 所有简报均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**Meta 发布 Llama 3.2 模型**

- **新模型变体**：Meta AI 宣布发布 Llama 3.2，包括[适用于边缘设备的 1B 和 3B 纯文本模型](https://twitter.com/AIatMeta/status/1838993953502515702)，以及支持多模态任务的 [11B 和 90B 视觉模型](https://twitter.com/AIatMeta/status/1838993953502515702)。所有模型均支持 [128K token 上下文长度](https://twitter.com/_philschmid/status/1838998169293615318)。

- **性能**：[1B 和 3B 模型在关键任务上优于 Gemma 2 2.6B 和 Phi 3.5-mini](https://twitter.com/AIatMeta/status/1839018085329809831)，而 [11B 和 90B 视觉模型与 Claude 3 Haiku 和 GPT4o-mini 具有竞争力](https://twitter.com/danielhanchen/status/1838991771948425652)。

- **技术细节**：[视觉模型使用 adapter 层进行图像-文本集成](https://twitter.com/AIatMeta/status/1839033482015895600)，而 [1B 和 3B 模型是通过对 Llama 3.1 8B 进行剪枝（pruning）和蒸馏（distillation）创建的](https://twitter.com/AIatMeta/status/1839018079529087158)。

- **生态系统支持**：模型[首日即支持 Arm、MediaTek 和 Qualcomm](https://twitter.com/AIatMeta/status/1839018083207491888)，并在包括 AWS、Azure 和 Google Cloud 在内的 [25 个以上合作伙伴平台上可用](https://twitter.com/AIatMeta/status/1838993953502515702)。

- **开源**：模型可从 [llama.com 和 Hugging Face 下载](https://twitter.com/rohanpaul_ai/status/1839009997440880812)，并在 150 多个跨语言基准数据集上进行了评估。

**其他 AI 新闻**

- **OpenAI CTO 离职**：[OpenAI 首席技术官 Mira Murati 宣布离职](https://twitter.com/miramurati/status/1839025700009030027)。

- **Molmo 发布**：[Allen AI 发布了 Molmo](https://twitter.com/rohanpaul_ai/status/1839004028690186438)，这是一个开源多模态 AI 模型系列，据报道其最强模型表现优于专有系统。

- **Gemini 更新**：[Google 宣布了对 Gemini 1.5 的改进](https://twitter.com/demishassabis/status/1839085152259158336)，Flash 和 Pro 生产模型提供了极具竞争力的性能/价格比。

- **Meta Connect 发布会**：Meta 展示了 [Project Orion，一个全增强现实（AR）眼镜原型](https://twitter.com/ylecun/status/1839038551457161256)，并[暗示了售价 300 美元的 Quest 3S](https://twitter.com/ylecun/status/1839091926118654316)。

**AI 研究与开发**

- **基准测试**：围绕[多模态模型的新基准测试](https://twitter.com/DrJimFan/status/1839012622441787430)以及[开源与闭源模型之间比较](https://twitter.com/Tim_Dettmers/status/1838959756377313750)的讨论。

- **模型优化**：分享了[提高模型性能](https://twitter.com/rohanpaul_ai/status/1839090102233870623)和[降低计算成本](https://twitter.com/omarsar0/status/1838995198476460461)的技术。

- **AI 安全**：在新模型发布的背景下，关于 [AI 安全和对齐（alignment）](https://twitter.com/mustafasuleyman/status/1838956259871277100)的持续讨论。

---

# AI Reddit 简报

## /r/LocalLlama 简报

**主题 1. 开源视觉语言模型挑战专有巨头**

- **[Molmo 是我发现的第一个能读懂模拟时钟的视觉模型，这是 Claude/GPT/Gemini 无法做到的。它在手表照片中混淆了分针和时针，但位置判断正确](https://i.redd.it/hedu6vjwvyqd1.png)** ([Score: 57, Comments: 13](https://reddit.com//r/LocalLLaMA/comments/1fp62xq/molmo_is_the_first_vision_model_ive_found_that/)): **Molmo** 作为一个视觉模型，展示了**读取模拟时钟**的能力，这是 **Claude**、**GPT** 和 **Gemini** 等其他主流模型未能完成的任务。虽然 Molmo 成功解读了指针的位置，但在分析手表图像时，在区分分针和时针方面出现了错误。
  - **Molmo** 的论文明确提到在**模拟时钟读取数据**上进行了训练，这或许解释了其优于其他模型的原因。包含特定训练数据凸显了多样化数据集对模型能力的重要性。
  - 该模型在读取多个手表时表现出令人印象深刻的准确性，即使其中一个手表慢了一个小时。这表明其在解读**图表和图形**等各种视觉呈现方面具有潜在应用价值。
  - 一项用户测试显示，**Molmo** 对时钟图像提供了详细、甚至可能过于详尽的回答。这种细节水平与其它模型倾向于关注单一假设的做法形成对比，可能表明其分析方法更为全面。
- **[Molmo：由 AllenAI 开发的开源 SOTA 多模态 AI 模型系列](https://molmo.allenai.org/)** ([Score: 184, Comments: 85](https://reddit.com//r/LocalLLaMA/comments/1fp5gut/molmo_a_family_of_open_stateoftheart_multimodal/)): Allen AI 发布了 **Molmo**，这是一个能够同时处理**文本和图像**的开源**多模态 AI 模型**系列。Molmo 模型的大小从 **3 亿到 30 亿参数**不等，在包括 **VQAv2**、**GQA** 和 **OKVQA** 在内的各种基准测试中达到了 SOTA 性能，在某些任务上甚至超越了 **GPT-4V** 等更大的闭源模型。这些模型可以通过 [Hugging Face](https://huggingface.co/allenai/molmo) 获取，并可用于视觉问答、图像描述和多模态聊天等任务。
  - **Molmo** 模型展示了令人印象深刻的能力，包括**辨认模拟时钟的时间**和执行**空间感知任务**。用户使用多只手表测试了该模型，发现它可以准确识别不同的时间，尽管它在转录钢琴谱等任务上表现吃力。
  - 模型架构使用 **OpenAI 的 ViT-L/14 CLIP** 进行视觉编码，在实验中其表现优于 **SigLIP**。作者 **Matt** 解释说，SigLIP 在单裁剪训练中表现良好，但在 Molmo 使用的多裁剪/高分辨率训练中表现较差。
  - Molmo 包含了多个模型的**完全开源数据集和训练代码**。团队计划发布 Checkpoints 以及各种视觉编码器消融实验的结果，并对在未来迭代中尝试不同的语言和视觉骨干网络持开放态度。
- **[Ovis 1.6 - 一个基于 Gemma 2 的 10B 视觉语言模型，在 MMMU 上超越了 Llama 3.2 11B 和 GPT-4o-mini](https://huggingface.co/AIDC-AI/Ovis1.6-Gemma2-9B)** ([Score: 49, Comments: 25](https://reddit.com//r/LocalLLaMA/comments/1fppvov/ovis_16_a_gemma_2based_10b_visionlanguage_model/)): **Ovis 1.6** 是一个基于 **Gemma 2** 的 **10B 参数视觉语言模型**，现已发布。它在 **MMMU** 基准测试中表现优于 **Llama 3.2 11B** 和 **GPT-4o-mini** 等更大规模的模型。该模型在各种视觉语言任务中取得了 SOTA 结果，展示了设计高效的小型模型在多模态理解方面与大型模型竞争并超越它们的潜力。
  - 用户对 **Ovis 1.6** 超越 **Llama 3.2 11B** 的说法表示**怀疑**，注意到对比表中缺少 Llama 3.2，并质疑在 Llama 3.2 发布后 24 小时内进行的快速性能评估。
  - 一位用户通过 [Spaces 演示](https://preview.redd.it/eebzp82x54rd1.png?width=2752&format=png&auto=webp&s=4bc40af9f2e1cffdd7db5620c3fe79255e403c99)测试了 **Ovis 1.6**，发现其主观上与他们尝试过的其他模型相当。另一位用户认为，与 **MiniCPM v2.6** 和 **Qwen 2 VL 7B** 等模型相比，**Llama 3.2 11B** 在视觉任务上稍逊一筹。
  - 原帖作者（OP）澄清说，性能对比是基于两个模型都已公布的 **MMMU 基准测试**。一些用户同意 Ovis 在个人测试中可能更好，但强调需要更全面、更具体的数值对比。


**主题 2. Llama 3.2：Meta 在开源 AI 领域的多模态飞跃**

- **llama.cpp 尚未支持 Llama-3.2 vision** ([Score: 32, Comments: 34](https://reddit.com//r/LocalLLaMA/comments/1fppoch/llama32_vision_is_not_yet_supported_by_llamacpp/)): **llama.cpp** 项目目前不支持 **Llama-3.2 vision** 功能，正如该项目 GitHub 仓库中的一个公开 issue 所示。[issue #9643](https://github.com/ggerganov/llama.cpp/issues/9643) 表明，需要开展工作来实现对最新 Llama 模型版本视觉功能的支持。
  - **Ollama** 正在独立于 llama.cpp 开发对 **Llama-3.2 vision** 的支持，正如其 [发布博客](https://ollama.com/blog/llama3.2) 和相关的 [PRs](https://github.com/ollama/ollama/pull/6963) 中提到的。一些用户建议关注 Ollama 或考虑使用 **mistral.rs** 等其他工具以获得更好的模型支持。
  - llama.cpp 仓库所有者 **Ggerganov** 表示，添加多模态支持是具备软件架构技能的新贡献者的机会。他在 [GitHub 评论](https://github.com/ggerganov/llama.cpp/issues/8010) 中强调，需要更多具备此类技能的人才来维持项目质量。
  - 用户对 llama.cpp 缺乏对 **Phi3.5 Vision**、**Pixtral** 和 **Qwen-2 VL** 等各种视觉模型的支持表示失望。一些人推测了实现中的挑战，而另一些人则开玩笑说地理屏蔽（geoblocking）问题可能会影响模型的获取。
- **Llama 3.2 多模态** ([Score: 244, Comments: 87](https://reddit.com//r/LocalLLaMA/comments/1fp9had/llama_32_multimodal/)): Meta 发布了 **Llama 3.2**，这是其开源 AI 模型的一次更新，具有新的**多模态能力**和额外的**模型尺寸**。虽然帖子正文未提供发布的具体细节，但标题表明 Llama 3.2 现在可以处理和生成文本及视觉内容，从而可能扩展其在各个领域的应用。
  - **Llama 3.2** 模型（11B 和 90B）在多模态基准测试中表现强劲，在数学推理和视觉问答等领域超越了 **Claude3-Haiku**，并与 **GPT-4o-mini** 展开竞争。**90B** 模型在多语言任务中表现尤为出色，在 VQAv2 测试中得分为 **86.9%**。
  - Meta 出人意料地在发布大版本的同时发布了更小的 **1B** 和 **3B** 模型，分别在高达 **9T tokens** 上训练了 37 万和 46 万小时。这些模型在 tooling 和 function-calling 方面展示了令人印象深刻的能力，达到了 8B 模型的性能水平。
  - 此次发布面临一些争议，Hugging Face 上的模型**禁止欧盟（EU）访问**。这引发了关于 **AI Act** 对模型可用性影响的讨论，以及个人和公司潜在的变通方案。
- **在手机上运行 Llama 3.2 3B - 支持 iOS 和 Android** ([Score: 151, Comments: 47](https://reddit.com//r/LocalLLaMA/comments/1fppt99/run_llama_32_3b_on_phone_on_ios_android/)): **PocketPal AI** 应用现在为 **iOS** 和 **Android** 设备提供了 **Llama 3.2 3B 模型**（Q4_K_M GGUF 变体），允许用户在智能手机上运行此 AI 模型。由于 Q8 版本可能存在降频问题，开发者目前仅在默认模型中添加了 **Q4 变体**，但设备内存充足的用户可以将 GGUF 文件作为本地模型导入，并确保选择 **"llama32" chat template**。
  - **PocketPal AI** 应用的 UI 收到了用户的详细反馈，建议进行改进，例如将标签重命名为 "Downloaded" 和 "Available Models"，并使界面更加直观。开发者积极回应了这些反馈。
  - 用户报告了性能指标，其中一位指出在其设备上达到 **11 tokens/sec**，另一位分享了在 **iPhone 14 iOS 18.0** 上的 **CPU 使用率**。一位用户在拥有 12GB RAM 的智能手机上成功运行了 **Q4K** 格式的 **Mistral Nemo 12B** 模型。
  - 该应用使用 **llama.cpp** 进行推理，并使用 **llama.rn** 进行 React Native 绑定。目前在 Android 上使用 **CPU**，虽然尚未开源，但开发者提到未来可能会考虑开源。


**主题 3. Qwen 2.5：阿里巴巴在开源 LLM 领域的突破**

- **Qwen 2.5 vs Llama 3.1 对比图** ([Score: 30, Comments: 17](https://reddit.com//r/LocalLLaMA/comments/1fp8v9h/qwen_25_vs_llama_31_illustration/)): 作者在获得 **3090 GPU** 后对比了 **Qwen 2.5** 和 **Llama 3.1** 模型，并制作了一张插图来评估它们的性能。在使用 **32B Qwen 模型** 几天后，他们分享了这张图片以突出 Alibaba 的成就，并指出了该模型令人印象深刻的能力。
  - 用户讨论了 **32B 模型** 的可用性，有人推荐 **70B 模型**，因为其性能达到 **16 T/s**。原帖作者询问 **32B 和 70B** 模型之间是否有显著提升，以证明购买第二块 **3090 GPU** 的合理性。
  - 一些用户赞扬了 **Alibaba** 对开源的贡献，对 **Alibaba 和 Meta** 都在 AI 社区获得尊重感到惊讶。其他人注意到了 **Qwen 70B 模型** 令人印象深刻的能力，将其性能与 **4000 亿+ 参数模型** 进行比较。
  - 关于在消费级硬件上运行大模型的讨论，原帖作者分享了他们使用支持上下文量化的 **ollama fork** 的设置，在 **3090 GPU** 上运行 **"q4 32b q4 64k" 或 "q6 14b q4 128k"** 配置。
- **qwen2.5:72b 是目前最强的编程模型吗？** ([Score: 66, Comments: 66](https://reddit.com//r/LocalLLaMA/comments/1fpq1jq/is_qwen2572b_the_strongest_coding_model_yet/)): 用户报告称，通过 **Hugging Face Spaces** 访问的 **Qwen 2.5 72B Instruct** 模型提供了卓越的编程辅助，并认为它在特定需求下优于 **Claude** 和 **ChatGPT-4**。他们询问该模型是否在客观上是编程任务的最佳选择，并提供了 [Hugging Face space 链接](https://huggingface.co/spaces/Qwen/Qwen2.5-72B-Instruct) 作为参考。
  - **Qwen2.5 72B** 的编程性能受到称赞，**32B 版本** 的能力也几乎旗鼓相当。用户期待 **qwen2.5 32b-coder** 的发布，预计它在编程任务上将超越 72B 模型。
  - 关于模型对比的辩论：一些人认为 **Qwen2.5 72B** 在复杂任务上并不优于 **Claude** 或 **Mistral-Large2-123B**，而另一些人则认为开源模型现在足以满足大多数编程需求。**Context window size** 被强调为大型项目的关键。
  - 用户讨论了在本地运行大模型的硬件设置，建议包括多块 **RTX 3090** 或 **P40** GPU。提到了 **Q4** 和 **AWQ** 等 **Quantization** 技术，以实现高效的模型部署。


**Theme 4. 欧盟 AI 法规对模型可用性和开发的影响**

- **[LLAMA 3.2 不可用](https://i.redd.it/mupq13jgk2rd1.jpeg)** ([Score: 1060, Comments: 388](https://reddit.com//r/LocalLLaMA/comments/1fpmlga/llama_32_not_available/)): 由于监管限制，Meta 的 **LLAMA 3.2** 模型目前对 **欧盟** 用户 **不可用**。这一限制影响了通过 **Meta AI 网站** 和 **第三方平台**（如 Hugging Face）访问模型。这种情况突显了 **欧盟法规** 对该地区 AI 模型可用性的影响。
  - **Meta 的 LLAMA 3.2** 模型在 **欧盟** 不可用，原因是可能存在从 Facebook 照片中非法抓取用户数据进行训练的问题。**1B 和 3B 文本模型** 仍然可以访问，但 **Vision 模型** 被禁止。
  - 用户辩论了 **GDPR** 等 **欧盟法规** 的优劣，一些人称赞其在消费者保护方面的努力，而另一些人则认为这抑制了 **AI 竞赛** 中的创新和竞争力。**AI Act** 旨在监管高风险 AI 系统和生物识别分类。
  - 关于 **Meta 对欧盟法规的合规性** 以及 LLAMA 是否真正 **开源** 的讨论正在进行。一些人猜测这可能是 Meta 的一项政治举措，旨在向欧盟施压，要求其宣布 LLAMA 为开源，从而免受某些法规的约束。

**Theme 5. 大型语言模型在扩展和可靠性方面的挑战**

- **更大且更具指令遵循能力的 AI 模型变得更不可靠** ([Score: 109, Comments: 23](https://reddit.com//r/LocalLLaMA/comments/1fpjo5w/larger_and_more_instructable_ai_models_become/)): 一篇 **Nature 论文** 揭示，经过更多指令和对齐训练的 **更大 AI 模型** 在五个困难任务类别中变得 **更不可靠**。虽然在简单任务上的表现有所提高，但模型在处理更难的变体时，越来越多地给出 **错误答案** 而不是拒绝回答，且 **人类读者无法准确辨别** 这些自信但错误的回答的正确性。这一趋势在包括 **OpenAI GPT**、**Meta's Llama** 和 **BLOOM** 在内的多个模型家族中都有观察到。
  - **RLHF 方法** 因未能奖励模型准确表达其 **epistemic status**（认识状态）而受到批评。一些人认为这项研究可能已经 **过时**，因为它使用的是 **GPT-3.5** 和 **Llama 1** 等较旧的模型，而另一些人则认为这一趋势仍然具有相关性。
  - 研究中对 **"avoidant responses"**（规避性回答）的定义受到质疑，**几乎所有** 此类回答都被归类为“不合规的规避”。批评者认为，根据 [补充信息](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07930-y/MediaObjects/41586_2024_7930_MOESM1_ESM.pdf) 中的定义，这些回答并不一定更可靠。
  - 该论文发表在 **Nature** 而非顶级的 ML 会议上被认为是不寻常的。它于 **2023 年 6 月 2 日** 提交并于近期发表，这对于通常青睐更快速的会议发表的计算机科学研究来说并不典型。

- **为什么大多数模型的上下文窗口“仅”为 100K tokens，而 Gemini 却达到了 2M tokens？** ([Score: 99, Comments: 93](https://reddit.com//r/LocalLLaMA/comments/1fp4s7e/why_do_most_models_have_only_100k_tokens_context/)): 该帖子讨论了大多数语言模型（**100K tokens**）与 **Gemini**（**2M tokens**）之间 **上下文窗口大小的差异**。作者质疑为什么其他模型无法达到或超过 Gemini 的上下文窗口，特别是考虑到 Gemini 的有效性以及 **Gemini 2.0** 进一步扩展的可能性。他们试图了解阻止其他模型实现类似上下文窗口大小的技术限制。
  - **Google 的硬件** 能力，包括其具有 **256 路快速芯片间互连** 和每个 pod **8,192 GB 内存** 的 **TPUs**，显著优于典型的 **Nvidia** 配置。这种硬件优势可能是 **Gemini** 拥有超大上下文窗口的关键因素。
  - 大多数模型的 **有效上下文长度** 通常远低于其宣传值，通常约为其声明的上下文大小的 **1/4**。**Google** 似乎在解决长上下文理解和信息检索问题方面取得了进展。
  - **Google Research** 发表了关于 **Infinite Context Windows** 的工作，在点积注意力层中引入了 **compressive memory**（压缩内存）。这与 **Ring Attention** 等技术一起，可能有助于 Gemini 高效处理更长的上下文。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型进展与发布**

- **OpenAI 的高级语音模式 (Advanced Voice Mode)**：OpenAI 为 ChatGPT 发布了高级语音模式，具备[唱歌、哼唱和声音模仿](https://www.reddit.com/r/singularity/comments/1fp1ifc/chatgpts_advanced_voice_mode_can_sing_hum/)等功能，尽管它被指令不得使用某些特性。系统提示词限制了调情和浪漫互动。

- **带有语音功能的 Meta AI**：Meta 宣布了 [OpenAI 高级语音模型的竞争对手](https://www.reddit.com/r/singularity/comments/1fpaeew/meta_announces_meta_ai_with_voice_a_competitor_to/)，允许用户将自己置于 AI 虚拟化身中。

- **Salesforce xLAM-1b**：Salesforce 发布了 xLAM-1b，这是一个 10 亿参数的模型，尽管体积相对较小，但在[函数调用 (function calling) 方面实现了 70% 的准确率，超越了 GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。

- **Phi-3 Mini 更新**：Rubra AI 在 6 月发布了更新后的 Phi-3 Mini 模型，[具备函数调用能力](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)，可与 Mistral-7b v3 竞争。

**AI 研究与技术**

- **Google DeepMind 的多模态学习**：一篇 [Google DeepMind 论文](https://arxiv.org/html/2406.17711v1)展示了如何通过联合样本选择进行数据策展，从而加速多模态学习。

- **Microsoft 的 MInference**：[Microsoft 的 MInference 技术](https://arxiv.org/abs/2407.02490)能够在保持准确性的同时，为长上下文任务实现高达数百万 token 的推理。

- **扩展合成数据生成**：一篇关于[扩展合成数据生成](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/)的论文利用大型语言模型中的多样化视角，从 10 亿个网络策展的角色 (personas) 中生成数据。

**AI 行业动态**

- **OpenAI 重组**：OpenAI 正在[取消非营利组织的控制权，并给予 Sam Altman 股权](https://www.reuters.com/technology/artificial-intelligence/openai-remove-non-profit-control-give-sam-altman-equity-sources-say-2024-09-25/)。与此同时，[多名关键人员离职](https://www.reddit.com/r/singularity/comments/1fpj6ls/more_people_leaving_openai/)，包括 CTO Mira Murati。

- **Google 的 AI 人才引进**：Google [支付了 27 亿美元请回 AI 研究员 Noam Shazeer](https://www.reddit.com/r/singularity/comments/1fpbc6u/article_google_paid_27_billion_to_bring_back_an/)，他此前离开并创办了 Character.AI。

- **OpenAI 的数据中心计划**：Sam Altman 提出了一个计划，[在多个州建设多个 5 GW 的数据中心](https://www.bloomberg.com/news/articles/2024-09-24/openai-pitched-white-house-on-unprecedented-data-center-buildout)，从一个 5GW 的设施开始。

**AI 应用与演示**

- **阿里巴巴的 MIMO**：阿里巴巴展示了 MIMO，这是一个通过空间分解建模实现[可控角色视频合成](https://www.reddit.com/r/singularity/comments/1fp0ti3/alibaba_presents_mimo_controllable_character/)的系统。

- **FaceFusion 3.0.0**：[FaceFusion 3.0.0](https://www.reddit.com/r/StableDiffusion/comments/1fpbm3p/facefusion_300_has_finally_launched/) 的发布展示了换脸技术的进步。

- **Looney Tunes 背景 LoRA**：一位用户为 Stable Diffusion 1.5 训练了一个 [Looney Tunes 背景图像风格 LoRA](https://www.reddit.com/r/StableDiffusion/comments/1fp94dn/still_having_fun_with_15_trained_a_looneytunes/)，展示了微调技术的多功能性。

**AI 伦理与监管**

- **欧盟 AI 法规**：欧盟的《AI 法案》(AI Act) 包含[限制在工作场所和学校使用情绪识别技术](https://www.reddit.com/r/singularity/comments/1fp4789/new_voice_is_illegal_in_eu_workplaces_and_schools/)的条款，这可能会影响高级 AI 语音模型在这些环境中的部署。

**硬件与基础设施**

- **Meta 的 AR 眼镜**：Meta 推出了 [Orion，这是他们的首款真正增强现实眼镜](https://about.fb.com/news/2024/09/introducing-orion-our-first-true-augmented-reality-glasses/)，标志着可穿戴 AI 技术的进步。


---

# AI Discord 回顾

> 由 O1-mini 生成的摘要之摘要之摘要

**主题 1. Llama 3.2 模型发布与性能**

- [**Llama 3.2 发布多个尺寸版本**](https://discord.com/channels/879548962464493619/879548962464493622/1288578069032079464)：**Meta** 发布了四个尺寸的 **Llama 3.2** (**90B**, **11B**, **3B**, **1B**)，目标指向**医疗领域**。尽管如此，**Llama-3.1 70B** 的表现仍优于它，平均得分为 **84%**，在 **MMLU College Biology** 中得分为 **95.14%**。

- [**基准测试差异凸显性能差距**](https://discord.com/channels/879548962464493619/1110598183144399058/1288579271870386207)：在 **LM Studio** 中，用户报告 **Llama 3.2 1B** 达到了 **49.3%**，**3B** 达到了 **63.4%**，量化模型运行速度为 **15-17 tokens/sec**，展示了显著的性能差异。

- [**社区对 Llama 3.2 局限性的批评**](https://www.youtube.com/watch?v=_MQVHyEeER4)：成员们对 **Llama 3.2** 相比 **Llama 3.1** 的表现表示失望，强调了在执行文件计数等基础任务时的问题，详见社区分享的 [YouTube 视频](https://www.youtube.com/watch?v=_MQVHyEeER4)。

**主题 2. AI 模型微调与优化**

- [**Unsloth AI 提升微调效率**](https://discord.com/channels/1179035537009545276/1179035537529643040/1288580232340836406)：**Unsloth AI** 优化了 **Llama 3.2** 的微调，实现了 **2倍的训练速度提升** 并减少了 **60% 的内存** 占用，使其能在 **低 VRAM 配置** 上运行。用户成功实现了 **QLoRA** 配置，并期待 **vision model** 的支持。

- [**讨论有效的 LLM 训练策略**](https://discord.com/channels/1179035537009545276/1179035537529643040/1288580232340836406)：社区成员交流了关于 **LLM 训练技术** 的见解，强调了数据集配置、不同的 batch sizes 以及精细的参数调整，以优化性能并减少错误。

- [**WeightWatcher 辅助模型诊断**](https://github.com/CalculatedContent/WeightWatcher)：讨论重点介绍了 **WeightWatcher**，这是一个用于分析模型权重和分布的工具，通过详细的诊断促进明智的训练决策并增强优化策略。

**主题 3. AI 硬件与 GPU 讨论**

- [**探索免费平台上的 GPU 可用性**](https://discord.com/channels/879548962464493619/1110598183144399058/1288579271870386207)：用户辩论了在 **Google Colab 免费版** 上运行模型的潜力，质疑在没有资金投入的情况下什么才算“相对高性能”，强调了 AI 模型部署的可及性。

- [**NVIDIA RTX 5090 与 RTX 5080 规格泄露引发讨论**](https://videocardz.com/newz/nvidia-geforce-rtx-5090-and-rtx-5080-specs-leaked)：**NVIDIA** 即将推出的 **RTX 5090** 拥有 **21,760 CUDA cores**、**32GB GDDR7 显存** 和 **600W** 功耗，而 **RTX 5080** 则配备 **16GB VRAM**。这引发了内容创作者和游戏玩家关于 **VRAM 与速度** 权衡的辩论。

- [**VRAM 限制影响大模型部署**](https://discord.com/channels/1110598183144399058/1153759714082033735/1288781696388431883)：对话强调 **24GB GPU** 在处理 **70B** 模型时非常吃力，更倾向于能维持至少 **15 tok/s** 速度的配置。成员们探索了多 GPU 集成和模型量化等解决方案，以克服这些限制。

**主题 4. AI 政策与企业转型**

- [**OpenAI 领导层变动引发担忧**](https://x.com/papers_anon/status/1839131401322639805?s=46)：包括 **Mira Murati** 和 **Barret Zoph** 在内的关键人员近期离职，引发了关于 **OpenAI** 从 **初创公司** 向 **企业结构** 转型的猜测，这可能影响创新并吸引监管审查。

- [**许可限制导致 Llama 3.2 在欧盟可用性受限**](https://discord.com/channels/1053877538025386074/1053877538025386074/1288615728127148032)：由于许可协议分歧，**Meta AI** 的 **Llama 3.2** 模型（尤其是 **11B** 和 **90B Vision Instruct**）面临 **欧盟访问限制**，限制了当地开发者的可用性并引发了关于合规性的辩论。

- [**OpenAI 非营利结构中的利润权益单位 (PIUs)**](https://riveron.com/posts/accounting-for-pius/)：关于 **OpenAI 非营利** 身份下的 **Profit Interests Units (PIUs)** 讨论浮出水面，引发了对利用非营利框架实现营利目的的担忧，可能招致 **加州总检察长** 等机构的监管行动。

**主题 5. 社区工具与集成**

- [**Aider 为模型引入高级 (Senior) 与初级 (Junior) 角色**](https://discord.com/channels/1131200896827654144/1131200896827654149/1288580759694868500)：**Aider** 推出了 **'Senior'** 和 **'Junior'** 角色，通过划分规划与执行的职责来简化编码流程。用户建议使用 **'Planner'** 和 **'Executor'** 等替代名称以提高清晰度。

- [**OpenRouter 发布 Vision Llama 并更新 Token 计费方式**](https://x.com/OpenRouterAI/status/1839157099747479553)：**OpenRouter** 推出了首个带有 [免费端点](https://x.com/OpenRouterAI/status/1839157099747479553) 的 **Vision Llama**，并新增了 **五个新端点**。他们还宣布 **Gemini models** 将从按 **字符计费转向按 Token 计费**，这将使 Token 计数减少约 **4 倍**，并计划在 **10 月 1 日**后将价格 **翻倍**。

- [**LM Studio 面临 Llama 3.2 Vision 模型的兼容性问题**](https://discord.com/channels/1110598183144399058/1110598183144399061/1288579271870386207)：**LM Studio** Discord 频道中的用户指出，**llama.cpp** 尚不支持 **Llama 3.2 Vision Instruct** 模型。尽管集成存在挑战，用户仍表达了部署这些模型的兴趣，并强调了量化框架（quantization frameworks）未来支持的必要性。

- [**LangChain 讨论源文档检索逻辑**](https://discord.com/channels/1038097195422978059/1038097196224086148/1288887526488019116)：**LangChain** 用户讨论了基于 **LLM** 置信度的 **源文档（source documents）** 条件检索，主张更直观的响应行为，并讨论了如 **Langfuse** 等替代调试工具，以便在不损害数据隐私的情况下进行监控。

- [**Tinygrad 的自定义内核生成增强了优化**](https://github.com/mesozoic-egg/tinygrad-notes/tree/main)：在 **tinygrad** 内部，用户强调了 **自定义内核生成（custom kernel generation）** 优于 **PyTorch** 固定内核的优势，为特定应用提供了更大的优化空间和潜在的性能收益。


---

# 第 1 部分：Discord 高层摘要




## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama 3.2 发布并具备多模态功能**：Meta 推出了 **Llama 3.2**，提供四种规格（90B, 11B, 3B, 1B），旨在应用于医疗领域，但讽刺的是，**Llama-3.1 70B** 的表现大幅领先于它。
   - 在基准测试中，**Meta-Llama-3.1-70B-Instruct** 获得了 **84% 的平均分**，在 **MMLU College Biology** 测试中表现尤为出色，达到 **95.14%**。
- **Tau 的最新创新成果揭晓**：一篇新[文章](https://dev.to/p3ngu1nzz/from-data-expansion-to-embedding-optimization-taus-latest-innovations-4h3n)重点介绍了 **P3ngu1nzz** 在数据扩展和嵌入优化（embedding optimization）方面的创新，以及包含 **1 亿步（100 million steps）** 的 **Tau** [训练运行](https://huggingface.co/p3nGu1nZz/Tau/tree/main/results/tau_agent_D10_100M)。
   - 这些进展专注于提高上下文理解能力，这对于各种 AI 应用至关重要。
- **Gemini 在目标检测领域引起关注**：**Gemini 的目标检测（object detection）** 功能已发布，详细见解可在此处[获取](https://huggingface.co/spaces/saq1b/gemini-object-detection)。
   - 其目标是利用尖端技术增强 AI 在目标检测任务中的能力。
- **构建用于法律推理的 AGENTIC RL SWARM**：一名成员正在开发一种 **AGENTIC RL SWARM** 设置，旨在通过集成 **RAG** 和 **graphrag** 等工具来处理复杂的法律任务。
   - 这种集成旨在增强上下文检索和功能，重点是对输出进行严格评估。
- **Colab 免费版的性能潜力**：用户讨论了在 **Google Colab 免费版** 中运行模型的巨大潜力，并探讨了在这种环境下什么才算“相对高性能”。
   - 这对在没有资金限制的情况下部署 AI 模型的可访问性具有重要意义。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 3.2：微调的飞跃**：Unsloth 团队宣布已优化 **Llama 3.2** 的微调，实现了 **2 倍的训练速度提升** 并减少了 **60% 的显存 (VRAM)** 占用，使其在低显存配置上也可运行。
   - 用户报告了使用 QLoRA 配置的成功实现，而视觉模型支持预计很快推出，促使大家呼吁更新 Unsloth。
- **NVIDIA 新产品线引发连锁反应**：泄露的 NVIDIA 即将推出的 **RTX 5090** 和 **RTX 5080** GPU 规格显示 CUDA 核心数增加，但显存容量各异，引发了当前用户对升级合理性的讨论。
   - 有人担心为了更快的规格而牺牲显存，特别是对于需要性能稳定性的内容创作者和游戏玩家。
- **OpenAI 的企业转型引发投资者疑虑**：社区内注意到的担忧表明，OpenAI 正在从其令人兴奋的初创根基转向企业结构，从而影响了创新。
   - 投资者正在推测缺乏显著增长的原因，并传言如果未能达到目标（尤其是 **10 倍增长**），将面临内部审查。
- **高效 LLM 训练策略**：关于**用于营销分析的 LLM 训练**的咨询引发了关于数据集配置和微调实践以优化性能的深入讨论。
   - 用户交流了包括不同 Batch Size 和训练技术在内的方法见解，强调了仔细调整参数以减少错误的必要性。
- **使用 Alpaca 的微调灵感**：社区成员分享了在微调过程中使用 **Alpaca 指令模板**的经验，重点关注 Tokenizer 配置。
   - 寻求关于集成该模板的指导，强调了其复杂性和带来的训练挑战。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Llama 3.2 性能基准测试**：用户对 Llama 3.2 模型进行了基准测试，显示 **1B** 模型得分 **49.3%**，**3B** 模型得分 **63.4%**，展示了量化模型达到约 **15-17 tokens/sec** 的显著性能差异。
   - 更广泛的对比突出了这如何影响跨平台的 Token 吞吐量。
- **Llama 3.2 Vision 模型暂不支持**：**Llama 3.2 Vision Instruct** 模型在 **llama.cpp** 中尚不支持，使用户对未来的集成和量化挑战感到不确定。
   - 尽管存在集成障碍，部署这些模型的兴趣依然浓厚。
- **显存限制大模型部署**：参与者一致认为显存 (VRAM) 对于大模型至关重要，**24GB** 的 GPU 在运行 **70B** 模型时非常吃力，更倾向于能维持至少 **15 tok/s** 速度的配置。
   - 讨论集中在显存权衡和可行的模型选择上。
- **跨 GPU 的性能指标**：基准测试显示，AMD **RX 5700 XT** 系统约为 **35 tokens/sec**，**NVIDIA RTX 4060** 系统约为 **40 tokens/sec**。
   - 用户注意到 Apple **M3 Max** 芯片达到了 **61 tokens/sec** 的惊人结果，强调了硬件能力的差异。
- **LLM 硬件需求讨论**：关于适用于配备 **32GB RAM** 的 **Intel i7-8750H** 的 LLM 讨论中，推荐了 **Qwen 2.5** 等选项，并指出了 Intel 集成显卡的局限性。
   - 对系统内存 (RAM) 的依赖意味着运行大型模型时的处理速度较慢。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **新的 Sr. & Jr. 角色让编码更轻松**：Aider 的最新更新为模型引入了 **'Senior'** 和 **'Junior'** 角色，通过明确定义规划与执行之间的职责来简化编码过程。
   - 用户建议使用 **'Planner'**（规划者）和 **'Executor'**（执行者）等替代名称，以减少对这些角色的混淆。
- **用户体验追求更快的节奏**：围绕 Aider UI 的讨论指出，应使两步过程变为可选，在允许通过新角色配置进行规划的同时，提供更快速的编辑选项。
   - 正在提议如 **/fast** 命令之类的想法来切换模式，以便在不牺牲高级功能的情况下增强用户体验。
- **Aider 的最佳模型搭配**：社区成员讨论了最佳模型配置，建议将 **OpenAI 的 o1-preview** 用于 Senior 角色，将 **Claude 3.5 Sonnet** 用于 Junior 任务。
   - 在实现过程中，当速度是首要任务时，也会考虑使用 **Deepseek** 模型。
- **Mend Renovate 自动化依赖管理**：对话强调了 **Mend Renovate**，这是一个通过识别较新的软件包版本并促进代码集成来自动更新依赖项的工具。
   - 用户希望 LLM 能够独立处理软件包版本控制，以简化项目设置。
- **Sonnet 的可靠性受到质疑**：用户注意到 **Sonnet** 的性能在没有明确触发因素的情况下可靠性下降，对此表示担忧。
   - 社区推测，重叠的系统错误修复可能会影响 Sonnet 的功能。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 3 登陆 HuggingChat**：Nous Research 在 [HuggingChat](https://huggingface.co/chat/settings/NousResearch/Hermes-3-Llama-3.1-8B) 上发布了 **8B** 规模的 **Hermes 3** 模型，展示了在指令遵循方面的增强。
   - 该模型旨在提升 AI 应用中的交互性，体现了 Nous Research 致力于推进用户响应型 AI 的承诺。
- **Llama 3.2 Vision Encoder 规模巨大**：**Llama 3.2 Vision Encoder** 拥有惊人的规模，**11B 模型**的 Encoder 接近 **3B** 参数，而 **90B 模型**的则达到 **18B**。
   - *成员们强调了其巨大的规模*，并指出了这对各种应用中处理能力的影响。
- **推理 Llama 3.2 需要强大的算力**：为了推理 **90B Llama 3.2**，用户建议可能需要 **3x H100 GPU**，对于更大的 Batch 或张量并行（tensor parallelism）可能需要 **4x**。
   - 这指出了高效模型部署所需的实际 GPU 基础设施考量，特别是在 **Runpod** 等平台上。
- **Wordware 应用集成 O1Mini**：**更新后的 Wordware 应用**现在包含了 **O1Mini**，通过利用 **Sonnet 3.5** 进行模型排名的 [OPUS Insight](https://app.wordware.ai/explore/apps/aa2996a0-93c9-4c19-ade2-1796c5c8a409) 增强了功能。
   - 此次更新凭借全面的排名功能，增强了在模型评估和用户参与方面的竞争优势。
- **判断与奖励建模增强 Hermes 3**：关于 **Hermes 3** 的**判断与奖励建模（judgement and reward modelling）**改进的咨询确认了其在训练中使用了**合成数据（synthetic data）**。
   - 这种方法旨在将模型性能提升到传统公共数据集所能提供的水平之外。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **General 频道诈骗链接警报**：成员们对一个潜在的欺诈链接表示担忧，并确保已对发布者采取措施。
   - *“绝对是个骗局，”* 一位成员指出，强调了社区内的警惕性。
- **Triton Conference 2024 录像已发布**：[Triton Conference 2024](https://www.youtube.com/watch?v=NZz5sczZ_30) 的录像现已可以观看，其中包含行业领袖的主旨演讲。
   - 下午的会议包括 **Meta** 关于其 Triton 策略的见解，可通过[此链接](https://www.youtube.com/watch?v=ONrKkI7KhU4)查看。
- **高级 PyTorch Profiling 技术**：成员们探索了检查 **PyTorch** 中内存分配的方法，重点关注层、权重和优化器状态。
   - 讨论了使用 **torchdispatchmode** 进行自动 Profiling 等技术，以优化内存利用率。
- **针对边缘计算推出 Llama 3.2**：Meta 推出了 **Llama 3.2**，其特点是针对边缘设备优化的轻量级**视觉 LLM**，增强了开发者的可访问性。
   - 针对其在 **EU**（欧盟）可用性受限的问题引发了担忧，这影响了当地开发者获取先进资源。
- **危地马拉社区聚会规划**：提出了一项在**危地马拉**组织聚会的倡议，邀请当地爱好者建立联系。
   - 规划强调了区域协作以及建立当地 AI 社区的重要性。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **小众兴趣推动 AI 进步**：一位成员强调了像 **PonyDiffusion** 这样的特定兴趣如何推动 AI 艺术生成的创新，挑战创意边界。
   - *粉丝圈塑造了对 AI 内容的认知*，表明用户参与度与技术进步之间的互联性日益增强。
- **Stable Diffusion 的 GPU 问题激增**：一位新手询问如何在没有 GPU 的情况下运行 **Stable Diffusion**，引发了关于使用 **Kaggle** 而非 Colab 以获得更好资源的建议。
   - 共识强调了在图像生成任务中，高性能 GPU 对于 **Stable Diffusion** 最佳性能的必要性。
- **LoRA 模型效果不尽如人意**：一位用户报告其 **LoRA 模型** 在输出图像中产生的变化不足，不像 Hugging Face 上看到的那些高质量示例，这引发了关注。
   - 澄清显示模型确实有细微变化，但未能达到基准图像所设定的*高预期*。
- **Colab 上的 RVC 安装咨询**：成员们讨论了如何在 **Colab Pro** 上安装用于语音转换的 **RVC**，并推荐了 Hugging Face 上提供的众多 **RVC 模型**。
   - 这些资源共享帮助那些投身于语音处理任务的人员简化了设置过程。
- **图像生成时间受到关注**：一位用户注意到在相同参数下，其本地设置的图像生成时间不稳定，引发了关于 **VRAM 使用情况**和基准测试效率的讨论。
   - 关于系统流量影响输出的推测，展示了用户对优化 **Stable Diffusion** 运行的持续追求。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **明确禁止广告政策**：Discord 社区有严格的**禁止广告**政策，支持研究共享，但禁止推广公司和产品。
   - 参与者强调遵守特定频道的规则，以确保社区指南的清晰。
- **关于 LLM 中 Filler Tokens 的探讨**：讨论围绕 **Filler Tokens** 在 LLM 架构中的有效性展开，承认其在合成任务中取得了成功，但对其泛化能力表示怀疑。
   - *LLM 究竟如何从 Filler Tokens 中真正获益？* 仍然是一个紧迫的问题，表明需要进一步调查。
- **寻求 Chinchilla Scaling Laws 数据集**：一位成员正在寻找展示 **参数量 (# params)**、**Token 数 (# tokens)** 和 **Loss** 之间相关性的数据集，以便在不进行多次模型训练的情况下分析低阶项，参考了 [Chinchilla scaling laws 论文](https://arxiv.org/pdf/2405.15074)。
   - 这凸显了研究人员需要更多可获取的资源来验证缩放结果。
- **在 H100 上集成 FA3 的尝试**：出现了关于在 **H100** 上为小模型训练添加 **FA3** 支持的讨论，预期该集成可能比较直接。
   - 由于 H100 获取权限有限，挑战依然存在，这使得测试和实施工作变得复杂。
- **调试 Token 生成问题**：一位用户报告在 Token 生成过程中超过了最大序列长度，发现了 `tok_batch_encode` 方法的潜在问题。
   - 同行的回应强调了需要集体调试努力来有效解决这些挑战。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 在上下文保留方面遇到困难**：用户对 **Perplexity AI** 无法记住过去的问题表示担忧，特别是在追问时，这种情况最近有所恶化。
   - 一位用户提到：*“这个平台在日常使用中仍然有用，但确实变得越来越糟了。”*
- **对 Llama 3.2 发布的兴奋**：一名成员宣布 **Llama 3.2** 已在 llama.com 上发布，并以“LFG”的口号激发了大家的兴奋。
   - 然而，另一名成员表示尚未在 Perplexity 的界面上看到它。
- **Mira Murati 从 OpenAI 离职**：Mira Murati 已正式**离开 OpenAI**，引发了关于 AI 领域人才迁移的讨论，详见此 [YouTube 视频](https://www.youtube.com/embed/zk1GwCIEvVU)。
   - 对该组织及整个 AI 技术格局的影响仍在推测中。
- **AI 攻克 reCAPTCHA 挑战**：分享的一项分析显示 **AI 击败了 reCAPTCHA** 系统，引发了对网络安全和更新验证方法的担忧。
   - [此处详情](https://www.perplexity.ai/search/ai-beats-recaptcha-IJvLzX98RkeMdh.kpXIqXw) 展示了 AI 不断进化的能力。
- **澄清 Zapier 中的 Perplexity 结构**：一名成员寻求关于在 Zapier 中使用 **Perplexity** 的澄清，特别是关于与 webhooks 集成的部分。
   - *消息结构是否有特定的格式要求？*

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Meta AI 访问限制令用户沮丧**：成员们对访问 **Meta AI** 表示沮丧，特别是在美国境外，一些用户尝试通过 VPN 进行规避。**Llama 3.2 license** 与欧盟（EU）的不兼容加剧了这些访问挑战。
   - 讨论强调了阻碍用户有效利用所需 AI 工具的关键限制。
- **Llama 3.2 发布引发争议**：随着 **Llama 3.2** 的推出，用户分析了其新的多模态能力，并努力解决欧盟用户的兼容性问题以及 **Hugging Face** 的托管问题。
   - 用户对开发所需核心模型的各种功能和访问权限表示担忧。
- **游戏开发中 AI IDE 的投资回报率（ROI）**：成员们分享了他们在游戏开发中首选的 AI IDE，强调了 **Cursor** 和 **GitHub Copilot** 等高效代码生成的选项。
   - 一位用户分享说，他们成功地将 **ChatGPT** 与 SSH 集成以进行实时代码修改，优化了工作流程。
- **Advanced Voice Mode 表现不及预期**：用户对 **Advanced Voice Mode** 感到沮丧，抱怨其缺乏互联网搜索能力，且需要繁琐地切换回文本模式。
   - 尽管存在限制，成员们仍对随着 ChatGPT-5 到来而预期的改进抱有希望。
- **o1 在文件上传方面表现挣扎**：成员们讨论了 **o1** 缺乏文件上传能力的问题，导致许多人换回 **GPT-4o**，这影响了生产力。
   - 用户对 **o1** 模型在遵循复杂指令方面与 **GPT-4o** 相比的性能表示担忧。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 领导层大洗牌引发质疑**：OpenAI 最近的人员离职（包括核心领导层）引发了对公司发展方向的怀疑，成员们表达了对 *除了 Sam 之外，所有 OG OpenAI 成员都离开了* 的担忧。
   - 这些辞职的时机引发了关于内部紧张局势的猜测，暗示该组织可能正处于十字路口。
- **对 Molmo 性能声明的怀疑**：在 **Molmo** 优于 **LLaMA 3.2** 的声称中，成员们对这些断言的真实性表示怀疑，有人指出偏见背书 *没有证据*。
   - 关于 Molmo 发布时间线的澄清指出，它仅在 LLaMA 3.2 发布前几小时推出，但鼓励通过个人测试来验证性能。
- **利润权益单位（Profit Interest Units）引发争议**：成员们讨论了在非营利机构中引入 *Profit Interest Units (PIUs)* 的影响，质疑潜在的监管后果。
   - 有人担心利用非营利地位谋取利润动机可能会招致加州总检察长等实体的审查。
- **NeurIPS 投稿被拒凸显偏见**：**Rewardbench** 在 NeurIPS 被拒成为成员间幽默与沮丧的话题，评论中提到了关于使用 **C++** 的轻慢反馈。
   - 成员对学术门槛表示担忧，一位成员表示在非营利组织中提供任何形式的“股权”补偿似乎 *很奇怪*。
- **轻松的会议结构提高生产力**：成员们反思了减少会议数量且更加轻松的会议效果，一位成员指出，尽管安排了 **3.5 小时**，但更倾向于在当天早些时候举行。
   - 大家一致认为在必要时堆叠会议，建议专注于高效利用时间，而非过度的日程安排。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Vision Llama 登陆 OpenRouter 并提供免费端点**：首个视觉版 **Llama** 现已在 **OpenRouter** 上线，并提供 [免费端点](https://x.com/OpenRouterAI/status/1839157099747479553)。总共推出了 **五个新端点**，由多个供应商提供支持。
   - 鼓励用户体验最新功能，并附带庆祝图标 🎁🦙。
- **Gemini Tokenization 简化成本**：OpenRouter 将转为对 Gemini 模型计算 **tokens** 而非字符数，使表观 token 计数减少约 **4 倍**。此举旨在为开发者规范并降低成本。
   - 这些变化将导致当前价格 **翻倍**，因为它们将 tokens 与每 token 计价模型对齐，并计划在 **10 月 1 日** 后进一步调整。
- **OpenRouter 充值与发票问题**：用户报告了 OpenRouter 上的信用交易困难，指出付款后交易可能需要一段时间才能显示。后端延迟或供应商问题可能导致查看交易历史记录时出现中断。
   - 一位用户展示了他们最终收到的信用额度，引发了对信用系统可靠性的担忧。
- **Llama 3.2 对欧盟用户的限制**：Meta 在欧盟使用其视觉模型的政策引发了对该地区用户可访问性和合法性的担忧。成员们指出，供应商所在地和遵守 Meta 规则方面的困惑可能会带来问题。
   - 这引发了关于在欧洲提供 **Llama 3.2** 推理服务的讨论。
- **BYOK Beta 测试参与请求**：一位成员询问如何加入 **Bring Your Own Key (BYOK)** beta 测试。他们提出通过私信提供 **电子邮件地址** 以方便参与。
   - 该成员表示愿意分享个人联系信息以协助 beta 测试过程。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **许可证合规性引发挫败感**：成员们讨论了许可证合规性问题，强调由于与监管条例的分歧，**欧盟访问被封锁**，导致了对访问限制的挫败感。
   - 一位成员幽默地评论说 **Mistral 现在成了一个梗 (meme)**，指出了这种情况的荒谬性。
- **OpenAI CTO 辞职引发猜测**：OpenAI CTO 的辞职引发了热议，成员们开玩笑说这引发了对公司现状的各种猜测。
   - 成员们对 OpenAI 的发展方向表示担忧，并建议内部问题或许可以拍成一部有趣的 **Netflix 迷你剧**。
- **新 Molmo 模型令人印象深刻的能力**：最近的 **Molmo 模型** 因其在**图像中定位**的能力而受到称赞，展示了开源开发的进步。
   - 成员们讨论了**语音标注图像训练**方法，标志着在整合多模态数据集方面取得了重大进展。
- **Tokenizer 缺少 Padding Token**：一位用户提出了 Tokenizer 在预训练期间缺少 Padding Token 的问题，这可能会干扰变长输入序列的处理。
   - *提供的选项包括将 Pad Token 设置为 EOS Token，或使用 `tokenizer.add_special_tokens({'pad_token': '[PAD]'})` 添加一个新的 Pad Token*。
- **计划 Llama 3.2 推理**：有人询问推理 **900 亿参数 (90 billion parameters)** 的 **Llama 3.2** 需要多少个 **H100 GPU**，以防止显存溢出 (OOM) 错误。
   - 用户计划获取 **Runpod GPU**，但旨在确保它们能够处理该模型，而无需因 OOM 问题而被迫删除它们。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Mira Murati 离开 OpenAI**：[Mira Murati](https://x.com/miramurati/status/1839025700009030027?s=46) 宣布在工作 6.5 年后离开 OpenAI，Sam Altman 对她的重要贡献表示感谢。
   - 这一变动引发了对组织内部领导层动态演变的疑问，特别是在最近几位关键人物离职之后。
- **Meta 展示 Orion AR 眼镜**：Meta 推出了 [Orion](https://about.meta.com/realitylabs/orion)，被誉为其最先进的 **AR 眼镜**，尽管由于制造挑战而选择暂不销售。
   - 初步反馈强调了其**美学吸引力**，突显了 Meta 整合数字与物理体验的野心。
- **Google 突破性的 AlphaChip**：Google 推出了 [AlphaChip](https://x.com/googledeepmind/status/1839306984480231852?s=46)，这是一款具有变革意义的微芯片，有望简化 AI 模型的设计，并附带公开可用的模型权重。
   - 这一进步增强了 Google 为 AI 设计最先进 **TPU** 的能力，标志着其芯片生产的一次重大飞跃。
- **Arcade 为 AI 工具融资 1700 万美元**：Arcade 已筹集 **1700 万美元**，用于构建一个变革性的 AI 产品创作平台，声称能帮助将创意愿景变为现实。
   - 该项目旨在使产品开发民主化，可能催化 AI 领域的创新。
- **GitHub Copilot 扩展到浏览器**：开发者现在可以直接在浏览器中访问 GitHub Copilot 的功能，使其与 Sourcegraph 的 **Cody Chat** 等类似产品展开竞争。
   - 这一扩展强调了详尽文档对于开发者充分利用该工具能力的重要性。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 正在招聘工程人才**：LlamaIndex 正在**旧金山**招聘一系列 **ML/AI** 工程职位，包括全栈职位。感兴趣的候选人可以在 [Twitter](https://twitter.com/llama_index/status/1839055997291344050) 上找到更多详情。
   - 此次扩张凸显了他们的增长，以及在应对即将到来的项目时增强工程团队的承诺。
- **NVIDIA 竞赛提供丰厚奖励**：由 **NVIDIA** 主办的竞赛提供超过 **$10,000** 的现金和硬件奖励，包括一块 **NVIDIA® GeForce RTX™ 4080 SUPER GPU**。开发者在 **11月10日** 之前可以提交创新的 LLM 应用，详情见[此处](https://developer.nvidia.com/llamaindex-developer-contest/join)。
   - 鼓励参与者探索不同领域的 *RAG* 应用，[条款和条件](https://developer.download.nvidia.com/licenses/nvidia-and-lla)可供查阅。
- **ReAct Agent 消息格式化**：成员们讨论了如何将用户和系统消息传递给 **ReAct agents**，强调了对适当类和格式化工具的需求。`ReActChatFormatter` 类对于正确构建聊天历史记录至关重要。
   - 澄清消息格式可以简化与 Agent 的通信，确保更顺畅的交互。
- **VectorStoreIndex 困惑澄清**：围绕 **VectorStoreIndex** 产生了一些困惑，引发了关于索引与其底层向量存储之间连接的对话。用户确认了如何在不初始化新向量存储的情况下访问 `vector_store` 属性。
   - 此次讨论旨在消除误解并改善用户与索引的交互。
- **关于 KnowledgeGraph RAG 与 QueryFusion 的辩论**：一位成员询问了如何正确使用 `QueryFusionRetriever` 而非 `KnowledgeGraphRAGRetriever` 进行知识索引。小组讨论了 RAG 检索器是否能更好地满足他们的查询需求。
   - 对话指向了在为特定应用选择最有效检索器方面的潜在改进。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Langtrace 增加 DSPy 实验支持**：Langtrace 引入了运行 DSPy 实验的功能，提供自动 **trace** 捕获、checkpoint、成本以及**评估分数可视化**。
   - 这一创新允许用户为每个流水线块创建专用项目，从而增强实验和优化。
- **访问 STORM 研究资源**：成员们讨论了 STORM 论文的资源链接，确认其在 [GitHub](https://github.com/stanford-oval/storm) 和 [arXiv](https://arxiv.org/abs/2402.14207) 上可用。
   - STORM 论文探讨了使用 LLM 撰写结构化文章，这引发了更多关于结构化知识生成的咨询。
- **在 DSPy 中构建 Agent**：分享了一个在 DSPy 中构建 Agent 的教程，强调了该框架的探索性质和现有局限性。
   - 本教程的目标是帮助他人学习如何利用 DSPy 创建有效的 **Agent 应用**。
- **类别数量优化**：关于模型中类别数量的讨论兴起，一位成员正在处理 **5 个类别**，并建议 **10 个类别** 可能更有益。
   - 这次对话强调了类别数量在实现有效**分类**和模型性能方面的重要性。
- **处理细微的类别差异**：强调了类别签名中细微差别的显著性，因为这些细微差别会使描述和模型清晰度变得复杂。
   - 成员们一致认为，准确地突出这些**差异**对于提高模型性能和理解至关重要。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Yamashi 幽默的绿卡诉求**：在一个轻松的时刻，Yamashi 幽默地问道：*“谁能施舍张绿卡？”*，表达了对法律和合规障碍的沮丧。
   - 他建议：*“是时候在特拉华州开一家假公司了，”* 反思了与绿卡获取相关的挑战。
- **Llama 3.2 的访问困境**：成员们表示，欧盟的限制阻碍了对 **Llama 3.2** 的访问，使得直接使用对他们来说变得很困难。
   - Yamashi 指出：*“但我无法直接使用 Llama 3.2，”* 强调了在访问该模型时面临的障碍。
- **Torchtune 遭遇 PackedDataset 错误**：一位成员遇到了与序列长度限制相关的 **PackedDataset** 错误，并引用了 [GitHub issue #1689](https://github.com/pytorch/torchtune/issues/1689)。
   - 他们提供了一个潜在的修复方案，并表示在评估测试要求后愿意提交 PR。
- **欧盟用户的 MetaAI 访问限制**：成员们对 **MetaAI** 的登录问题表示担忧，称欧盟用户无法访问其账户。
   - Yamashi 评论道：*“啊，确实我也无法登录 MetaAI，”* 指出了这些连接挑战。
- **对视觉问答数据集的热情**：一位成员对 [Hugging Face](https://huggingface.co/datasets?task_categories=task_categories:visual-question-answering&sort=trending) 集合中新提供的视觉问答（Visual Question Answering）数据集表示兴奋。
   - 他们指出了这些数据集在 finetuning 应用中的潜力。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **解决 MOToMGP Pass Manager 错误**：团队正在处理 **'failed to run the MOToMGP pass manager'** 错误，并邀请用户就 **Max / Mojo** 问题提供反馈以寻求潜在改进。
   - 鼓励成员分享与 Pass Manager 相关的抱怨或建议，以获得更流畅的体验。
- **对 Mojo/MAX 品牌背景图的兴趣**：一项投票调查了用户对 **Mojo / MAX** 品牌桌面背景的兴趣，主题包括可爱的 Mojo 火焰和 MAX 宇航员。
   - 用户通过表情符号投票（**是**或**否**）参与，表达了他们对这些创意设计的偏好。
- **验证机器人回归以确保安全**：验证机器人要求成员点击“我是人类 ✅”以维护社区安全并防止垃圾信息。
   - 未经验证的成员将在指定频道面临发帖限制，从而鼓励更好地遵守验证流程。
- **Mojo 直接编译为机器码**：一位成员澄清说，**Mojo** 直接编译为机器码，而不是像 Python 那样创建 **.pyc** 文件。
   - *“.pyc 是字节码缓存，Mojo 直接编译为机器码。”* 强调了 Mojo 在编译执行路径方面的高效性。
- **征求 MAX API 用户反馈**：正在向 **MAX API** 的用户寻求反馈，特别是关于使用中的挫折和潜在的改进建议。
   - 该成员鼓励就他们的 API 体验进行友好的思想交流，包括任何增强建议。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LLM 关于可用性的沟通误区**：当被问及问题时，尽管检索到了相关的**源文档（source documents）**，**LLM** 有时仍会回答*“对不起，我不知道”*。
   - 成员建议，文档检索应以 LLM 拥有有用信息为前提，以避免混淆。
- **不必要的源文档导致的困惑**：同一位成员批评说，即使 LLM 表示没有相关信息，也会返回**源文档**。
   - 他们指出，虽然大多数回答是令人满意的，但在否定回答中收到不必要的文档可能会产生误导。
- **调试工具的抉择**：一位参与者质疑了使用 **Langsmith** 等**调试（debugging）**工具的必要性，而发帖者因隐私问题拒绝使用。
   - 提出了 **Langfuse** 等替代方案，以便在不泄露敏感数据的情况下进行监控。
- **要求代码清晰化**：有人请求提供代码示例，以澄清发帖者在 LLM 交互中面临的问题。
   - 发帖者同意第二天分享示例，强调了对协作排查问题的承诺。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Generative AI 泡沫面临质疑**：一名成员表示担忧，认为 **Generative AI** 行业，特别是 **ChatGPT**，由于包括 **Mira Murati** 在内的近期关键人物离职，正接近崩溃。
   - 他们引用了一份令人警觉的通讯，声称 Generative AI 热潮是**不可持续的**，冒着损害重大技术声誉和公众认知的风险。
- **博士生在 Cohere 找到了归属**：一位新成员强调，在博士学业即将结束之际，他们有兴趣持续关注 AI 讨论，使 **Cohere** 成为他们的首选资源。
   - 这展示了该社区对于希望参与前沿 AI 话题讨论的学术界人士的价值。
- **关于 Avg Hard Negatives 计算的问题**：一位用户询问了 **'Avg Hard Negatives per Query'** 是如何计算的，并指出其数据集中硬负样本（hard negatives）比例不足 **10%**。
   - Cohere 澄清说他们不会在后台添加负样本，并建议核查数据质量。
- **训练后的模型性能**：在训练过程结束后，一位用户报告称该模型的表现仅略优于 **default English v3 reranker**。
   - 他们推测**数据质量**可能是导致这种不尽如人意表现的一个因素。
- **社区对新人表现出热情**：多位成员积极欢迎新人，并鼓励他们提出关于 **Cohere** 的问题，营造了友好的氛围。
   - 这体现了社区在 AI 学习中致力于协作和支持的承诺。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **发布了任意视图可合并性（Arbitrary View Mergeability）的证明**：GitHub 上分享了一个无需 mask 或 reshape 的 **arbitrary view mergeability** 证明，详细说明了 **Tinygrad** 中视图管理的见解。你可以在[这里找到证明](https://github.com/pschilliOrange/Tinygrad-view-merging-proof/blob/8672f35c1147798c8e9a78bfab28b9ff79bf45e6/Proof%20for%20when%20a%20new%20view%20must%20be%20appended.pdf)。
   - 该文档对当前视图合并技术中的挑战进行了可靠的概述。
- **识别出 Tinygrad 训练瓶颈**：用户报告称，即使使用 **4090 GPU**，**Tinygrad** 的训练也受到性能低下的阻碍，原因在于采样代码而非训练速度。他们澄清说，输出质量受损源于实现错误，而非硬件本身。
   - 这突显了改进**采样逻辑（sampling logic）**中的调试和功能的必要性。
- **Metal 双精度错误困扰**：一位用户遇到了与 **double precision**（双精度）相关的 Metal 错误，这是由于 **NumPy** 默认使用双精度值引起的。他们通过将 tensor 转换为 **float32** 解决了此问题，尽管随后出现了新的 buffer 问题。
   - 这场对话强调了针对 **Metal** 后端特性适配 Tinygrad 的挑战。
- **Tinygrad 与 PyTorch 的对决**：关于 **Tinygrad** 作为 **PyTorch** 更快替代方案（特别是在直接操作 **CUDA** 方面）的优势存在活跃讨论。虽然 Tinygrad 编译为 CUDA，但 PyTorch 受益于高度优化的 CUDA kernel。
   - 这种区别指向了可定制性与预优化性能之间的权衡。
- **Tinygrad 中未被发掘的优化**：成员们指出，与 **PyTorch** 的固定 kernel 相比，**Tinygrad** 的 **custom kernel generation**（自定义内核生成）提供了更多的优化机会。这种灵活性可能会显著影响特定应用中的整体性能。
   - 讨论集中在利用这些特性来实现定制化的性能提升。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LLaMA 3.2 Vision 在 Image Captioning 方面表现出色**：成员们注意到 **LLaMA 3.2 Vision 90B** 在 **image captioning** 方面能力极强，**11B** 版本也开始受到关注。
   - *一位成员幽默地建议对整个 LAION 数据集进行 captioning*，以展示其潜力。
- **OpenAI 的 Function Calling API 受到关注**：一位成员询问了 **OpenAI 的 function calling API** 是如何运作的，质疑它是依赖于 fine-tuned 模型还是输出检查。
   - 这反映了人们对 **API 设计细节和性能增强** 的持续兴趣。
- **宣布免费开放 LLaMA 3.2 Vision 访问权限**：TogetherCompute 与 **AI at Meta** 合作，为开发者免费提供 **LLaMA 3.2 11B Vision**，以便进行多模态 AI 实验。
   - 他们在[此链接](https://api.together.ai/playground/chat/meta-llama/Llama-Vision-Free)提供了一个免费的模型端点，并为增强性能提供了付费选项。
- **MaskBit 重塑图像生成技术**：**MaskBit** 通过 **bit tokens** 引入了 embedding-free 的图像生成，对传统的 **VQGAN** 模型进行了改进。
   - 该模型在 ImageNet 上仅凭 **305M 参数** 就实现了 **1.52 的 FID**，展示了 embedding-free 方法的有效性。
- **MonoFormer 简化生成过程**：**MonoFormer** 提出了一种统一的 transformer 架构，可以同时管理生成过程中的 **autoregression** 和 **diffusion**。
   - 该模型保持了具有竞争力的图像生成和文本输出，更多细节可在其[项目页面](https://monoformer.github.io/)查看。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Quiz 3 问题引发困惑**：一位成员对 **Quiz 3 的一个问题** 表示困惑，该问题在演讲者关于受限流（constrained flows）和非受限流（unconstrained flows）的解释中并未涵盖。另一位成员指出相关信息确实在幻灯片中，澄清了测验内容。
   - 这次交流凸显了将测验材料与课程内容对齐所面临的持续挑战。
- **RAG 模型在多模态数据上遇到困难**：人们对最新模型的 **RAG 能力** 表示担忧，特别是针对文本、表格和图像等多模态数据的表现。值得注意的是，**Claude 3** 在解释流程图方面表现出色。
   - 这表明模型需要更好地适应多样化的数据类型，以提高功能性。
- **Agentic RAG 项目初具规模**：一位成员分享了他们的 **ccmp_ai 项目**，这是一个提供新术语的非受限 RAG 模型，被称为具有动态问题域扩展能力的 **agentic RAG**。这突显了同行在项目概念化方面的创新。
   - 另一位成员认为这些术语非常有用，激发了对该模型应用进一步探索的兴趣。
- **医疗保健多智能体系统研究摘要**：题为 [AgentClinic: A Multimodal Agent Benchmark](https://open.substack.com/pub/yanpan0508/p/agentclinic-a-multimodal-agent-benchmark?r=ad7en) 的研究专注于医疗保健多智能体系统（multi-agent systems），分析了方法论和研究结果。它强调了这些系统在医疗保健领域的协作潜力，增强了 AGI 的应用。
   - 此类研究为多智能体系统的未来发展提供了信息，并强化了它们在 AI 领域的重要性。
- **Yvaine 的 Substack 发布**：Yvett 的 Substack 专栏“Embracing AGI”旨在与社区就 AI 领域的进展进行交流，特别是在医疗保健领域。她最近发布的内容包括强调 AGI 在医疗保健背景下作用的讨论。
   - 这一举措强调了在快速发展的 AGI 领域中，社区驱动的知识共享的重要性。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Llama 3.2 表现不佳**：在测试了 **Llama 3.2** 90b 后，一位成员表示失望，称其无法与 **Llama 3.1** 70b 媲美。他们引用了一个标题为 'Llama-3.2 (1B, 3B, 11B, 90B) : The WORST New LLMs EVER!?' 的 [YouTube 视频](https://www.youtube.com/watch?v=_MQVHyEeER4)，详细说明了他们的发现。
   - 该视频批评了新模型在各项指标上的缺陷，引发了关于其实际应用的讨论。
- **Open Interpreter 无法统计文件**：一位成员报告称，在使用 **3b** 模型配合 **Open Interpreter** 统计桌面文件时，它**未能**执行该任务。这引发了人们对该模型处理基础任务可靠性的担忧。
   - 社区正在质疑此类局限性将如何影响开发中更广泛的使用场景。
- **对 Tech Week SF 聚会的期待**：一位用户表达了参加旧金山 **Tech Week** 的兴奋之情，并建议见面击掌。这突显了社区在技术活动期间进行社交和联系的热情。
   - 成员们热衷于在这个高能量的活动中讨论他们的项目并分享见解。
- **NERD 任务的挑战**：一位成员描述了一个 **NERD** 任务，重点是将文本链接到新闻文章中提到的人物 **wiki** 条目。由于提取和匹配相关信息的复杂性，该任务被认为非常困难。
   - 对话强调了需要改进方法论，以应对文本分析中此类具有挑战性的任务。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **寻找 partition_pdf 的替代方案**：一位成员请求推荐 **unstructured 'partition_pdf'** 的替代方案，以便更好地从 PDF 中提取图像和表格。
   - *他们正在寻找针对这一特定任务更有效的工具。*
- **频道礼仪提醒**：另一位成员强调，在多个频道发布相同问题将被视为 **spam**，并采取了删除重复内容的行动。
   - *这一提醒强调了维护频道秩序的重要性。*

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **关于推广的疑虑**：一位成员表达了挫败感，质疑为何某些话题不被视为**推广 (promotion)**，暗示某些审查是有道理的。
   - 这一评论突显了社区内部关于讨论中推广界限的持续争论。
- **讨论缺乏清晰度**：由于只记录了一条消息，讨论缺乏上下文，导致被批评的主题存在歧义。
   - 成员们通常认为，更清晰的推广指南可以防止此类误解。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla AI 登上 Nature**：Mozilla AI 及其倡议在 Nature 的文章《忘掉 ChatGPT：为什么研究人员现在在笔记本电脑上运行小型 AI》中受到关注。讨论集中在**本地运行 AI 模型日益增长的趋势**，这增强了用户能力。
   - 文章包含了来自 Mozilla 开源 AI 负责人的见解，强调了向赋能个人用户使用自主模型的转变。
- **LLMs 获得系统通用性**：文章中展示的一个著名项目旨在促进 **Large Language Models (LLMs)** 在多个系统上运行，反映了它们的适应性。
   - 这一进步标志着在使强大的 AI 工具可用于多样化环境、弥合不同技术基础设施之间的差距方面取得了飞跃。
- **Continue 工具人气上升**：在最近的一次演讲中强调的 **Continue** 工具，因其在 **AI 辅助编码**中的实用性而受到认可，提高了开发者的生产力。
   - 这一认可信号表明，作为提高编码效率的资源，它在 AI 工程社区中的重要性日益增加。
- **获取 Nature 的完整洞察**：感兴趣的读者可以通过点击[此处完整文章](https://discord.com/channels/1089876418936180786/1288648264069025883/1288648264069025883)查看详细分析。
   - 该直接链接是进一步了解社区讨论的创新成果的重要资源。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **用户对 Function Calling 评估感到困惑**：一位用户对代码库中的 **Function Calling 评估** 提出了疑问，具体询问是否可以在提交 API/LLM 的同时提交自己的 **自定义评估数据集**。
   - 他们指出，关于如何集成由 **<prompt>, <llm_response>, <ideal response>** 组成的数据集以进行有效的错误分解，目前**缺乏清晰度**。
- **对自定义数据集错误洞察的需求**：同一位用户表示希望有一种工具能够 **分析他们的数据集**，并提供类似于 [BFCL 指标](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#metrics) 中概述的见解。
   - 这表明用户明确需要能够增强对自定义数据集中错误理解的功能。



---


**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将予以移除。


---


**DiscoResearch Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将予以移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将予以移除。


---

# 第 2 部分：频道详细摘要与链接


{% if medium == 'web' %}




### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1288925046416343071)** (1 条消息): 

> - `Tau 创新`
> - `Gemini 目标检测`
> - `LLama 3.2 评测`
> - `用于软件工程的推理模型`
> - `自定义阿拉伯语语义搜索模型` 


- **Tau 最新创新揭秘**：一篇新[文章](https://dev.to/p3ngu1nzz/from-data-expansion-to-embedding-optimization-taus-latest-innovations-4h3n)详细介绍了 **P3ngu1nzz** 在数据扩展和 Embedding 优化方面的创新。
   - 他们还分享了 **Tau** 的一个包含 **1 亿步** 的 [训练运行](https://huggingface.co/p3nGu1nZz/Tau/tree/main/results/tau_agent_D10_100M)。
- **Gemini 在目标检测领域引起关注**：**Gemini 目标检测** Space 已发布，可以在 [此处](https://huggingface.co/spaces/saq1b/gemini-object-detection) 进行探索。
   - 这个新 Space 旨在增强 AI 在目标检测应用中的能力，展示最新进展。
- **探索 LLama 3.2 的有效性**：一位成员提出了一个问题，*
- **软件工程中对推理模型的需求**：一场关于日常软件工程任务是否需要 [推理模型](https://huggingface.co/blog/onekq/daily-software-engineering-work-reasoning-models) 的讨论展开了。
   - 这引发了对新模型如何集成到传统软件流程中以提高效率的思考。
- **构建自定义阿拉伯语搜索模型**：一篇文章讨论了使用 **阿拉伯语 Matryoshka Embeddings** [开发自定义阿拉伯语语义搜索模型](https://huggingface.co/blog/Omartificial-Intelligence-Space/building-custom-arabic-semantic-search-model)。
   - 该模型利用 **Sentence Transformers** 在阿拉伯语语境下实现强大的检索增强生成 (RAG) 应用。



**提到的链接**：<a href="https://youtu.be/yWxWJfQ3Tcg)">Is LLama 3.2 any good ?</a>：让我们试着看看 Llama 3.2 是否真的好用。

  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1288578069032079464)** (491 条消息 🔥🔥🔥): 

> - `OpenAI 动态`
> - `Hugging Face 模型`
> - `机器学习与神经科学`
> - `Llama 3.2 更新`
> - `AI 部署方案`

- **OpenAI 的领导层变动及其影响**：包括 Mira Murati 在内的关键人员近期从 OpenAI 离职，引发了关于公司未来以及是否会催生专注于更安全 AI 的新初创公司的讨论。
   - 许多人认为，鉴于人才的流失，内部分歧可能会导致 OpenAI 创新能力的下降。
- **Hugging Face 模型数量达到 100 万**：Hugging Face 平台达成了一个重要的里程碑，模型数量突破 100 万个，并庆祝社区为此目标做出的贡献。
   - 这一增长反映了各种应用中对多样化机器学习模型日益增长的普及程度和需求。
- **探索 Llama 3.2 模型**：新的 Llama 3.2 模型带来了更小的变体和增强的功能，包括在 iPhone 15 Pro 等设备上高效运行的能力。
   - 然而，用户注意到与 ChatGPT 等模型相比，其性能表现各异，特别是在需要特定知识的特定任务中。
- **将 AI 与人类大脑研究相结合**：成员们讨论了机器学习与神经科学的交叉领域，并分享了关于处理脑机接口和神经活动的过去项目的见解。
   - 对话强调了理解意识与无意识思维的复杂性，并建议机器学习可以辅助解码这些过程。
- **AI 部署策略**：关于在生产环境中部署机器学习解决方案的咨询，引出了使用 Hugging Face 的推理部署服务的建议。
   - 参与者请求关于管理特定任务的多个模型的最佳实践指导，反映了在 SaaS 应用中集成 AI 的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/miramurati/status/1839025700009030027">来自 Mira Murati (@miramurati) 的推文</a>：我今天与 OpenAI 团队分享了以下便签。</li><li><a href="https://x.com/awnihannun/status/1839330067039887622">来自 Awni Hannun (@awnihannun) 的推文</a>：Llama 3.2 1B 4-bit 版本在我的 iPhone 15 Pro 上使用 MLX Swift 运行速度约为 60 toks/sec。它表现相当不错，且能轻松在端侧运行：</li><li><a href="https://ui.endpoints.huggingface.co/">Inference Endpoints - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/collections/Joseph717171/gguf-llama-32-instruct-oq8-0-f32ef32iq4-k-q8-0-iquants-66f49cf761597948c6fa789d">GGUF Llama-3.2-Instruct-OQ8_0-F32.EF32.IQ4_K-Q8_0 IQuants - Joseph717171 收藏集</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/discord-community/LevelBot/discussions/23">discord-community/LevelBot · 修复了在机器人私聊中发送消息可获得无限经验的问题，其他更改已注释</a>：未找到描述</li><li><a href="https://huggingface.co/SandLogicTechnologies/Llama-3.2-3B-Instruct-GGUF">SandLogicTechnologies/Llama-3.2-3B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct">meta-llama/Llama-3.2-11B-Vision-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/v4.44.2/main_classes/quantization">Quantization</a>：未找到描述</li><li><a href="https://www.scientificamerican.com/article/there-is-no-such-thing-as-conscious-thought/#:~:text=We%20are%20not%20simply%20puppets%20manipulated%20by%20our%20unconscious%20thoughts,">不存在意识思维这种东西 | Scientific American</a>：未找到描述</li><li><a href="https://x.com/GaryMarcus/status/1839037069307555905">来自 Gary Marcus (@GaryMarcus) 的推文</a>：等等，什么？Murati 离职了，Schulman 离职了，Karpathy 离职了，Ilya 离职了，Leike 离职了，Brockman 正在休假，可能还有十几个人也离开了，GPT-5 还没发布，Sora 还没出货，公司曾有一个开...</li><li><a href="https://github.com/Deadsg/DQNAgent/tree/main/Q_Layered_Network">DQNAgent/Q_Layered_Network at main · Deadsg/DQNAgent</a>：通过在 GitHub 上创建账户来为 Deadsg/DQNAgent 的开发做出贡献。</li><li><a href="https://www.technologyreview.com/2024/04/19/1091505/companies-brain-computer-interfaces/">超越 Neuralink：了解其他正在开发脑机接口的公司</a>：Synchron、Paradromics 和 Precision Neuroscience 等公司也在竞相开发脑植入物</li><li><a href="https://huggingface.co/docs/transformers/v4.44.2/llm_tutorial_optimization">为速度和内存优化 LLM</a>：未找到描述</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok 开源发布</a>：Grok 开源发布。通过在 GitHub 上创建账户来为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://www.hetzner.com/dedicated-rootserver/matrix-gpu/">专用服务器托管</a>：未找到描述</li><li><a href="https://www.biorxiv.org/content/10.1101/2020.07.01.183384v1.full">通过想象书写实现高性能脑机通信</a>：脑机接口 (BCI) 可以为失去移动或说话能力的人恢复沟通。迄今为止，BCI 研究的主要焦点一直是恢复粗大运动技能，例如...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1288583802490454128)** (5 条消息): 

> - `AGENTIC RL SWARM`
> - `Legal Reasoning Models` (法律推理模型)
> - `RAG Tool Utilization` (RAG 工具利用)
> - `RL Models in Legal Field` (法律领域的 RL 模型)
> - `CUDA and FP8 Training` (CUDA 和 FP8 训练)


- **为法律推理构建 AGENTIC RL SWARM**：一位成员正在构建一个专门用于处理长上下文**法律任务**的 **AGENTIC RL SWARM** 设置，涉及对重大案件的代理式分解。
   - 这将整合 **RAG** 工具，如 **graphrag、retriever、reranker** 和 **colbert v2**，以增强**上下文检索**和功能，同时对输出进行严格评估。
- **为法律代理探索 RL 模型**：另一位成员正在深入研究 **RL 模型**，以确定哪些模型在创建法律领域案件构建代理的代理能力 (agency) 方面最有效。
   - 这一探索将集中于这些模型如何优化复杂案件的法律推理和任务管理。
- **CUDA 和 FP8 学习**：一位成员报告了学习过程中的收获，特别是使用 **CUDA** 实验 **7b FP8** 模型。
   - 这反映了在机器学习工作流中掌握性能优化技术的实践方法。

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1288798457376673833)** (3 条消息): 

> - `Llama 3.2 发布`
> - `OpenAI 的 ChatGPT 新功能`
> - `新研究论文` 


- **具备多模态功能的 Llama 3.2 发布**：Meta 推出了四种规模（90B, 11B, 3B, 1B）的 **Llama 3.2**，旨在应用于医疗领域，但 **Llama-3.1 70B** 的表现出人意料地优于它。
   - **Meta-Llama-3.1-70B-Instruct** 以 **84% 的平均分** 登顶基准测试，尤其在 **MMLU College Biology** 中表现出色，得分达 **95.14%**。
- **分析 OpenAI 的 ChatGPT 新功能**：一段强烈推荐的 [YouTube 视频](https://www.youtube.com/watch?v=QDfE0HwDBo8)，标题为 **'OpenAI’s New ChatGPT: 7 Incredible Capabilities!'**，对新功能进行了精彩的解释。
   - 观众可以通过提供的链接探索 **Lambda 的 GPU Cloud**，还可以玩 *Tron 游戏*。
- **关于 AI 的最新研究论文**：可以在 [ACL Anthology](https://aclanthology.org/2024.acl-long.43.pdf) 访问一篇新的研究论文，涵盖了该领域的最新发现。
   - 这篇论文为 AI 研究社区内正在进行的讨论和发展做出了贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/aadityaura/status/1839233111927750830">来自 Aaditya Ura (@aadityaura) 的推文</a>: 针对医疗与保健领域的 Llama-3.2 90B, 11B, 3B 和 1B 的分析 🩺🧬💊 有趣的观察：- Llama-3.1 70B 表现优于 Llama-3.2 90B - Meta-Llama-3.2-90B-Vision Instruct 和 Base a...</li><li><a href="https://www.youtube.com/watch?v=QDfE0HwDBo8">OpenAI 的新 ChatGPT：7 个令人惊叹的能力！</a>: ❤️ 在这里查看 Lambda 并注册其 GPU Cloud: https://lambdalabs.com/paper 玩 Tron 游戏: https://agpallav.com/tron.html 来源: https://www.y...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1288580703239540908)** (142 条消息🔥🔥): 

> - `AI Model Comparison Tool` (AI 模型对比工具)
> - `Ollama Integration` (Ollama 集成)
> - `Hugging Face Models` (Hugging Face 模型)
> - `Community Feedback` (社区反馈)
> - `Future Enhancements` (未来增强功能)


- **AI 模型对比工具发布**：一位用户分享了一个新网站 [Countless.dev](https://countless.dev/)，该网站允许对比各种 AI 模型及其价格，是作为一个个人项目构建的。
   - 开发者旨在保持工具更新，并收集反馈以进行进一步改进。
- **社区对概览的反应**：成员们对该工具表示赞赏，一位用户指出，*"该死，这是一个非常棒的概览"*。
   - 围绕该工具的实用性和价值展开了讨论，特别是关于价格更新方面。
- **关于 Ollama 角色的讨论**：关于将 Ollama 作为本地托管模型包含在内的必要性产生了疑问，引发了关于它在 Hugging Face 数据之外是否增加价值的讨论。
   - 有建议考虑将其从主列表中移除，但为想要探索所有平台的用户保留。
- **提升用户体验的未来增强计划**：开发者表示计划增加更多功能，如模型评分和链接，以便在不同平台间进行更好的对比。
   - 社区成员强调了链接模型的重要性，以便在不同的托管平台之间获得无缝的对比体验。
- **集成 OpenLLM 排行榜功能的讨论**：讨论指向了将 OpenLLM 排行榜与对比工具集成的潜力，以增强用户洞察。
   - 共识是，对于涉及多个托管平台的任何稳健集成，正确的模型链接至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/qamarsidd/FreeTranscriptMaker">FreeTranscriptMaker - 一个由 qamarsidd 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.youtube.com/@BatCountryEnt/videos#:~:text=Share%20your%20videos%20with%20friends,%20family,%20and%20the%20world">BatCountryEnt</a>：未找到描述</li><li><a href="https://huggingface.co/p3nGu1nZz/Kyle-b0a">p3nGu1nZz/Kyle-b0a · Hugging Face</a>：未找到描述</li><li><a href="https://youtu.be/uyxRV1PP12Y">Llama3.2 全天候监视我的屏幕并向我发送每日摘要邮件</a>：https://screenpi.pe</li><li><a href="https://youtu.be/yWxWJfQ3Tcg">Llama 3.2 真的好用吗？</a>：让我们试着看看 Llama 3.2 是否真的好用</li><li><a href="https://countless.dev/">Countless.dev | AI 模型对比</a>：轻松对比 AI 模型！所有供应商一站式汇总。</li><li><a href="https://x.com/ahmetdedeler101/status/1839313737561551359">来自 Ahmet ☕ (@ahmetdedeler101) 的推文</a>：介绍 http://Countless.dev - 一个我（17 岁）仅用几天时间完全使用 @v0 和 @cursor_ai 构建的 Web 应用。对比每个 AI 模型，检查价格，并为您的用例找到最佳模型...</li><li><a href="https://github.com/p3nGu1nZz/Tau/blob/dev-pca-optimization-script/MLAgentsProject/Scripts/optimizer.py">Tau/MLAgentsProject/Scripts/optimizer.py at dev-pca-optimization-script · p3nGu1nZz/Tau</a>：使用 Unity 6 ML Agents 制作的 Tau LLM。通过在 GitHub 上创建账号为 p3nGu1nZz/Tau 的开发做出贡献。</li><li><a href="https://github.com/mediar-ai/screenpipe/blob/main/examples/typescript/pipe-email-daily-log/README.md">screenpipe/examples/typescript/pipe-email-daily-log/README.md at main · mediar-ai/screenpipe</a>：基于你所看、所说或所听内容构建个性化 AI 的库。支持 Ollama。Rewind.ai 的开源替代方案。安全且数据自持。使用 Rust 构建。- mediar-ai/screenpipe</li><li><a href="https://github.com/p3nGu1nZz/oproof/blob/main/oproof/main.py">oproof/oproof/main.py at main · p3nGu1nZz/oproof</a>：使用 Ollama 和 Python 验证 Prompt-Response 对。- p3nGu1nZz/oproof</li><li><a href="https://huggingface.co/p3nGu1nZz/Tau/tree/main/results/tau_agent_D10_100M">p3nGu1nZz/Tau at main</a>：未找到描述</li><li><a href="https://github.com/p3nGu1nZz/Tau">GitHub - p3nGu1nZz/Tau: 使用 Unity 6 ML Agents 制作的 Tau LLM</a>：使用 Unity 6 ML Agents 制作的 Tau LLM。通过在 GitHub 上创建账号为 p3nGu1nZz/Tau 的开发做出贡献。</li><li><a href="https://github.com/ytdl-org/youtube-dl">GitHub - ytdl-org/youtube-dl: 从 YouTube.com 和其他视频网站下载视频的命令行程序</a>：从 YouTube.com 和其他视频网站下载视频的命令行程序 - ytdl-org/youtube-dl
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1288767971204010004)** (5 条消息): 

> - `Multimodal Models`
> - `Contrastive Learning`
> - `Visual Learning`
> - `3D/4D Understanding` 


- **VLMs 预示未来**：今天发布了**两个多模态模型**，凸显了向**视觉语言模型 (VLMs)** 转变是 AI 的一个关键方向。
   - 鉴于目前的进展，*VLMs 正在成为未来的必经之路*已变得显而易见。
- **对比学习铺平道路**：在讨论中指出，**多模态**一直是目标，但由于难以关联不同模态而面临挑战。
   - 像 **CLIP** 这样的模型和**对比学习 (Contrastive Learning)** 等技术为整合多模态数据提供了新路径。
- **视觉学习者青睐 VLMs**：一位成员强调，人类倾向于成为**视觉学习者**，这促使 **VLMs 和 MLLMs** 在当前的 AI 讨论中日益普及。
   - 有推测认为，未来的发展将集中在 **3D 和 4D 理解**上，而不仅仅局限于语言或 2D 图像。
- **对 3D 和 4D 进展的质疑**：尽管出现了关于 **3D 理解**的讨论，但成员们对其可行性表示怀疑，称渲染 **3D 数据**并非易事。
   - 一位成员评论道，目前 *4D 理解还太遥远*，强调了持续存在的挑战。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1288891994734723072)** (1 条消息): 

> - `Word Injection in Embeddings`
> - `Parameter Freezing Techniques` 


- **单词注入解析**：文章讨论了现实生活中的学习是如何循序渐进的，并强调 Embedding 模型需要为新单词扩展词汇表，这需要对模型进行 Fine-tuning 并对 Embedding 进行重新索引。
   - *我们很少会经历一个令人震撼的时刻，让我们重新思考所知道的一切*。
- **微调 Embedding 的挑战**：在扩展词汇表并对模型进行 Fine-tuning 后，由于参数改变，之前计算的 Embedding 会失效，这可能会使处理数百万份文档变得复杂。
   - 这引发了一个重要的探究：如何在允许更新新输入 Embedding 的同时，冻结模型中的参数。
- **参数冻结的困惑**：关于如何冻结现有的输入 Embedding 同时允许新输入的 Embedding 可训练，目前还存在不确定性，这表明在参数管理方面可能存在潜在的复杂性。
   - 关注点在于是否可以将所有 Embedding 存储在单个参数中而不影响训练过程。



**提到的链接**：<a href="https://www.kacperlukawski.com/posts/word-injection/?_gl=1*1a2g1t4*_ga*MTgzMTMyNjgzMS4xNzI3MDkxMjE3*_ga_SL41LGM35F*MTcyNzM2NjA4MS4yLjEuMTcyNzM2NjA4NC4wLjAuMA..">Old dog, new tricks: Word Injection in the text embedding models</a>：在现实生活中，我们很少经历那种让我们重新思考所知一切的震撼时刻。当下的经验往往建立在过去经验之上，知识的积累是一个循序渐进的过程...

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1288597400281288769)** (2 条消息): 

> - `Colab Free Tier Performance`
> - `Default Training Logic in Diffusers`
> - `MSE Loss Computation in Diffusion Models` 


- **Colab 免费版性能潜力**：一位用户指出，在讨论性能标准时，几乎可以在 **Google Colab 的免费版**中运行任何模型。
   - 这引发了关于将“相对高性能”作为该环境下模型执行描述符的具体细节问题。
- **使用 UNet2D 训练 Diffusion 模型**：关于 Hugging Face diffusers 教程中的**默认训练逻辑**出现了一个问题，特别是围绕无条件图像生成。
   - 该教程指导创建一个旨在从 Smithsonian 数据集生成蝴蝶图像的 UNet2D 模型，并提供了更多资源的链接。
- **Diffusion 模型中 MSE Loss 的困惑**：一位用户质疑为什么 **MSE Loss** 是针对随机高斯噪声与残差 (Residual) 计算的，而不是结合 Timestep 交互。
   - 他们强调模型的重点是学习残差，这对于理解在每个 Timestep 从当前噪声图像中减去多少噪声至关重要。



**提到的链接**：<a href="https://huggingface.co/docs/diffusers/en/tutorials/basic_training">Train a diffusion model</a>：未找到描述

  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1288580232340836406)** (486 条消息🔥🔥🔥): 

> - `Llama 3.2 Fine-Tuning`
> - `NVIDIA GPU Specifications`
> - `LLM Training Techniques`
> - `Marketing Data Datasets`
> - `OLLaM and LM Studio Issues`

- **Llama 3.2 Fine-Tuning 支持更新**：Unsloth 团队宣布 Llama 3.2 Fine-Tuning 已修复并现已可用，预计很快将支持 vision model。QLoRA 配置允许模型在特定的 VRAM 最小值下运行，提升了可用性。
   - 多位用户已成功微调模型，并大幅节省了 VRAM，目前正在等待即将推出的功能，特别是关于图像处理的部分。
- **NVIDIA 新 GPU 发布**：NVIDIA 即将推出的 GPU（RTX 5090 和 RTX 5080）信息已泄露，显示 CUDA cores 数量有所增加，但 VRAM 规格各异。这引起了现有 GPU 持有者对性能必要性与升级成本之间关系的担忧。
   - 讨论围绕着为了更高的速度规格而牺牲 VRAM 展开，强调了这对内容创作者和游戏玩家可能带来的不利影响。
- **LLM 训练策略**：关于使用 LLM 进行营销分析和广告增强的咨询引发了对有效数据集和训练方法的讨论。提出了多种方法，从 Fine-Tuning 同一个模型到训练多个不同的模型。
   - 用户们就构建用于训练的最佳数据集交换了意见，同时寻求创建能够处理特定营销任务模型的最佳实践。
- **OLLama 和 LM Studio 加载问题**：一位用户报告了在 OLLama 和 LM Studio 中加载模型的问题，收到了与 tokenizer 合并相关的错误。在排查问题时，向有经验的用户寻求了指导。
   - 社区成员帮助识别了错误的根本原因，并为模型加载差异提供了潜在的解决方案。
- **WeightWatcher 工具讨论**：WeightWatcher 被讨论为一种有价值的模型诊断工具，能够提供对权重分布的见解，并协助做出明智的训练决策。用户对其在模型分析中的应用表示好奇。
   - WeightWatcher 与各种训练技术的集成可以帮助优化模型性能，从而更好地管理训练指标。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html">兼容性矩阵 (WSL) &#8212; 在 Radeon GPU 上使用 ROCm</a>: 未找到描述</li><li><a href="https://x.com/Unsloth">来自未定义用户的推文</a>: 未找到描述</li><li><a href="https://x.com/UnslothAI/status/1839036956245897685">来自 Unsloth AI (@UnslothAI) 的推文</a>: Llama 3.2 版本（包括 GGUF + bnb 4 bit 版本 + 重新上传的版本）现已上线 @HuggingFace！在此查看 Llama 3.2 的所有版本：https://huggingface.co/collections/unsloth/llama-32-66f...</li><li><a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/">Llama 3.1 | Model Cards 与 Prompt 格式</a>: Llama 3.1 - 最强大的开源模型。</li><li><a href="https://huggingface.co/Joseph717171/Llama-3.2-1B-Instruct-OQ8_0.EF32.IQ4_K-Q8_0-GGUF/resolve/main/Llama-3.2-1B-Instruct-OF32.EF32.IQ8_0.gguf">未找到标题</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/15OyFkGoCImV9dSsewU1wa2JuKB4-mDE_?usp=sharing#scrollTo=juQiExuBG5Bt">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf">Llama 3.2 - meta-llama 集合</a>: 未找到描述</li><li><a href="https://tenor.com/view/why-whyyy-neden-a%C4%9Flamak-%C3%BCzg%C3%BCn-gif-19603232">Why Whyyy GIF - Why Whyyy Neden - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fpet4v/llama_32_multimodal_">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/Loc">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF">unsloth/Llama-3.2-1B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://gist.github.com/fullstackwebdev/81e64e8faca496e5390d09a4756d8db4">llama32_3b_failwhale.py</a>: GitHub Gist: 即时分享代码、笔记和代码片段。</li><li><a href="https://tenor.com/view/no-gif-6533142189269812111">No GIF - No - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/UnslothAI/status/1839340091241869698">来自 Unsloth AI (@UnslothAI) 的推文</a>: 你现在可以在 Colab 上免费微调 Llama-3.2 了！Unsloth 使微调速度提升 2 倍，并减少 60% 的 VRAM 使用，且无精度损失。Llama 3.2 (1B) QLoRA 适用于 4GB GPU，(3B) 适用于 7GB...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fpeb5g/llama_32_versions_gguf_4bit_bnb_more/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fpet4v/llama_32_multimodal_ggufs_4bit_bitsandbytes/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/Joseph717171/gguf-llama-32-instruct-oq8-0-f32ef32iq4-k-q8-0-iquants-66f49cf761597948c6fa789d">GGUF Llama-3.2-Instruct-OQ8_0-F32.EF32.IQ4_K-Q8_0 IQuants - Joseph717171 集合</a>: 未找到描述</li><li><a href="https://chromewebstore.google.com/detail/page-assist-a-web-ui-for/jfgfiigpkhlkbnfnbobbkinehhfdhndo">Page Assist - 本地 AI 模型 Web UI - Chrome 网上应用店</a>: 使用本地运行的 AI 模型辅助你浏览网页。</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct#instruction-tuned-models),">meta-llama/Llama-3.2-1B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://datta0.substack.com/p/ai-unplugged-20-score-self-correction">AI Unplugged 20: 通过 RL 进行 SCoRE 自我修正、OpenAI o1 模型、Qwen 2.5 coder、Spectrum 微调。</a>: 洞察重于信息</li><li><a href="https://videocardz.com/newz/nvidia-geforce-rtx-5090-and-rtx-5080-specs-leaked">NVIDIA GeForce RTX 5090 和 RTX 5080 规格泄露 - VideoCardz.com</a>: GeForce RTX 5090 将配备 21760 个 CUDA 核心、32GB GDDR7 显存和 600W 功耗，RTX 5080 配备 16GB VRAM。消息源自 Kopite7kimi 本人。最可靠的 NVIDIA 爆料者之一现已确认了这些规格...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fq0f6m/llama_32_1b_4gb_vram_finetuning_2x_faster_colab/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/1040#issuecomment-2377762522">加载 Qwen2 聊天界面的正确方式是什么？· Issue #1040 · unslothai/unsloth</a>: 我遇到了这个错误：chat_template, stop_word, yes_map_eos_token, ollama_modelfile = CHAT_TEMPLATES[chat_template] ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^ KeyError: 'Qwen2-1.5B' 出自这段代码：def test_un...</li><li><a href="https://weightwatcher.ai/">WeightWatcher: 深度学习的无数据诊断</a>: 未找到描述</li><li><a href="https://github.com/CalculatedCo">GitHub - CalculatedCo</a></li>

ntent/WeightWatcher">GitHub - CalculatedContent/WeightWatcher: 用于预测 Deep Neural Networks 准确率的 WeightWatcher 工具</a>: The WeightWatcher tool for predicting the accuracy of Deep Neural Networks - CalculatedContent/WeightWatcher
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1288908901181423707)** (1 messages): 

> - `Llama 3.2 发布`
> - `新 Notebooks`
> - `模型上传` 


- **Unsloth 正式支持 Llama 3.2**：Unsloth 现在集成了 **Llama 3.2** 的文本模型，实现了 **2 倍的训练速度提升** 和 **减少 60% 的内存** 占用。Vision 支持预计很快推出，建议用户 **更新 Unsloth**。
   - *请务必检查更新以享受这些增强功能带来的好处。*
- **探索适用于 Llama 3.2 的新 Notebooks**：适用于 Llama 3.2 的新 **Google Colab** notebooks 已发布，包括 **[Llama 3.2 (3B)](https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing)** 和 **[Kaggle notebook](https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-2-1b-3b-conversational-unsloth/notebook)**。请在 **[其余 notebooks 列表](https://docs.unsloth.ai/get-started/unsloth-notebooks)** 中查看完整清单。
   - *利用这些资源来加强你在 Llama 3.2 及更高版本上的实验。*
- **Llama 3.2 模型上传公告**：**Llama 3.2 集合** 包含多种模型版本，包括 1B、3B、11B 和 90B 参数的 **Base** 和 **Instruct** 版。用户可以通过 **[集合链接](https://huggingface.co/collections/unsloth/llama-32-all-versions-66f46afde4ca573864321a22)** 访问所有版本。
   - *此次上传旨在简化针对各种应用的不同配置的访问。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1qN1CEalC70EO1wGKhNxs1go1W9So61R5?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1oCEHcED15DzL8xXGU1VTx5ZfOJM8WY01?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks)">Unsloth Documentation</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models)">Unsloth Documentation</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1288590341439754241)** (7 条消息): 

> - `OpenAI 的企业化转型`
> - `投资者对 OpenAI 的担忧`
> - `GPU 上的 Llama 3.2`
> - `Petscan API 的使用`
> - `管理层归属锁定 (Vesting Lock-ins)` 


- **OpenAI 转向企业模式**：有人担心 OpenAI 正在失去其作为**令人兴奋的初创公司 (exciting startup)** 的吸引力，并正在转向**企业模式 (corporate mode)**。
   - *一位成员感叹道*，“不再是一家令人兴奋的初创公司”，并指出这可能会扼杀创新。
- **投资者审视 OpenAI 的增长**：成员们推测**投资者**正在质疑资金的去向，以及为什么公司没有实现预期的 **10 倍增长**。
   - 有人评论说，如果结果没有改善，投资者可能会开始指责工程团队。
- **Llama 3.2 在 CuPy 上运行**：[GitHub](https://github.com/githubpradeep/llm_np_cp/blob/main/llama3.2_model.py) 展示了 Llama 3.2 完全使用 GPU 上的 **CuPy** 运行，这标志着效率的提升。
   - 一位成员分享了他们的代码，宣传了结合 **CuPy** 和 **NumPy** 执行模型的能力。
- **关于 Petscan API 使用的问题**：一位成员询问了 **Petscan API**，寻求可能以编程方式使用过它的其他人的见解。
   - 这表明社区有兴趣扩展该 API 在实际应用中的使用。
- **管理层归属锁定 (Vesting lock-ins)**：关于高层管理人员的**归属锁定**可能结束的推测出现，这可能预示着即将到来的变化。
   - 这可能意味着随着公司进行企业化演变，领导层动态或激励机制将发生转变。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/miramurati/status/1839025700009030027">来自 Mira Murati (@miramurati) 的推文</a>：我今天向 OpenAI 团队分享了以下简报。</li><li><a href="https://github.com/githubpradeep/llm_np_cp/blob/main/llama3.2_model.py">githubpradeep/llm_np_cp 项目 main 分支下的 llm_np_cp/llama3.2_model.py</a>：在 CuPy 和 NumPy 上运行 Llama 和 Gemma。通过在 GitHub 上创建账号为 githubpradeep/llm_np_cp 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1288598201867304991)** (110 条消息🔥🔥): 

> - `微调 Llama 3.1 & 3.2`
> - `Unsloth 库问题`
> - `训练配置`
> - `Qwen 模型错误`
> - `用于微调的 Alpaca 模板` 


- **Llama 3.1 & 3.2 微调问题**：用户在微调 Llama 3.1 和 3.2 模型时遇到了挑战，包括训练过程中的 NaN 错误以及对数据集配置的困惑。
   - 许多人建议使用更小的 batch sizes，减少 gradient accumulation（梯度累积），甚至更换数据集来解决这些问题。
- **Unsloth 库兼容性**：Unsloth 库正在积极开发中，包括对各种模型版本的支持，但目前缺乏对 vision 模型的兼容性，并出现与未训练 token 相关的错误。
   - 用户建议更新依赖项，特别是 transformers，以缓解其中的一些问题。
- **训练配置和超参数**：几位用户分享了他们的训练参数设置，讨论了各种设置（如 learning rates、gradient accumulation 和 batch sizes）对性能的影响。
   - 有人指出，调整这些参数可以显著影响内存使用情况和训练期间 NaN 的出现。
- **Qwen 模型微调尝试**：用户报告了尝试微调 Qwen 模型时的不同结果，在使用较大模型时，特定错误指向了 embedding 和 tokenization 问题。
   - 有建议提议在训练过程中将 `embed_tokens` 和 `lm_head` 包含在 `target_modules` 中，以解决这些问题。
- **用于微调的 Alpaca 模板使用**：关于以指定格式使用 Alpaca 指令模板的问题不断出现，这在微调模型时会影响 tokenizer 配置。
   - 用户寻求关于如何成功将此模板集成到其训练过程中的指导。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/17d3U-CAIwzmbDRqbZ9NnpHx">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/17d3U-CAIwzmbDRqbZ9NnpHxCkmXB6LZ0?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/reward-modelling-dpo-and-orpo">奖励建模 - DPO &amp; ORPO | Unsloth 文档</a>: 要在 Unsloth 中使用 DPO 或 ORPO，请遵循以下步骤：</li><li><a href="https://www.youtube.com/watch?v=eIziN2QUt8U">微调多模态 LLaVA 视觉和语言模型</a>: ➡️ 高级 Vision 微调仓库: https://trelis.com/advanced-vision/➡️ Trelis 时事通讯: https://blog.Trelis.com➡️ Trelis 资源与支持: https:/...</li><li><a href="https://docs.unsloth.ai/b">Unsloth 文档</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM</a>: 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1288732722403217428)** (15 条消息🔥): 

> - `使用 KTO 微调 Unsloth`
> - `用于 Low-Rank Adaptation 的 Flexora`
> - `INT-FlashAttention GitHub 资源`
> - `LLM 预训练中的 Data Packing`
> - `使用 DeepSpeed 预训练 GPT-2` 


- **使用 Unsloth 进行 KTO 微调**：一位用户询问了使用 **Unsloth** 处理二元反馈数据进行微调的可行性。
   - 这引发了关于在此类场景中有效利用 KTO 方法的讨论。
- **引入用于 Low-Rank Adaptation 的 Flexora**：消息中链接的论文提出了 **Flexora**，这是一种通过选择关键层进行优化微调来改进 Low-Rank Adaptation (LoRA) 的方法。
   - 它解决了现有 LoRA 技术在针对特定任务微调模型时的过拟合问题。
- **INT-FlashAttention 资源**：一位成员分享了 **INT-FlashAttention** 的 GitHub 资源链接，包括 **flash_atten_int8.py** 的 Python 代码。
   - 该项目似乎是增强 Attention 机制的更广泛开发工作的一部分。
- **关于 LLM 预训练中 Data Packing 的讨论**：一位用户询问了像 GPT 或 LLaMA 这样的 LLM 在训练期间如何处理不同的上下文长度，特别是关于 Data Packing 的问题。
   - 成员们讨论了诸如训练框架屏蔽无关部分以及同时上传多个示例的可能性等方法。
- **使用 DeepSpeed 预训练 GPT-2**：有人建议利用 **DeepSpeed** 预训练一个小型 **GPT-2** 模型，并强调了其效率。
   - 进一步的询问涉及 DeepSpeed 是否支持 Data Packing 以优化训练。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.10774v1">Flexora: Flexible Low Rank Adaptation for Large Language Models</a>: 大型语言模型 (LLMs) 通过增加模型参数规模推动了人工智能的进步，这显著增强了泛化能力并解锁了新的 c...</li><li><a href="https://github.com/INT-FlashAttention2024/INT-FlashAttention/blob/main/flash_atten_int8.py">INT-FlashAttention/flash_atten_int8.py at main · INT-FlashAttention2024/INT-FlashAttention</a>: 通过在 GitHub 上创建账户，为 INT-FlashAttention2024/INT-FlashAttention 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1288579271870386207)** (175 条消息🔥🔥): 

> - `Llama 3.2 性能`
> - `Llama 3.2 视觉模型`
> - `模型兼容性与使用`
> - `LM Studio 中的 Embedding 问题`
> - `新模型与更新` 


- **Llama 3.2 性能基准测试**：用户讨论了各种 Llama 3.2 模型的性能，指出 **Llama 3.2 1B** 的基准测试得分为 **49.3%**，**3B** 为 **63.4%**。
   - 性能比较还包括关于量化模型的讨论，以及这些模型如何影响 Token 吞吐量，部分模型达到了约 **15-17 tokens/sec**。
- **Llama 3.2 视觉模型支持**：像 **Llama 3.2 Vision Instruct** 这样的视觉模型目前在 **llama.cpp** 中尚不支持，未来的兼容性仍不确定。
   - 用户表达了使用这些模型的兴趣，但在量化和集成到现有框架方面仍面临挑战。
- **Embedding 与 API 问题**：一位用户报告了在升级到 LM Studio **0.3.2** 后，Embedding 在处理超过上下文窗口的较大输入时返回错误的问题。
   - 该问题通过对 Embedding 请求进行分块处理（chunking）得到解决，这突显了在 API 使用中管理上下文窗口限制的重要性。
- **安装与运行新模型**：围绕安装更新模型以及通过 LM Studio API 使用第三方模型的讨论引起了对正确模型版本的关注。
   - 用户被引导下载最新的 LM Studio beta 版以及从 Hugging Face 下载特定模型，以确保兼容性和功能性。
- **模型定制与行为**：用户对某些模型（如 **Minicpm 2.6**）表现出的限制性行为表示担忧，认为需要通过提示词工程（Prompt Engineering）才能产生理想的输出。
   - 建议包括使用不同的提示策略来绕过内置的审查机制，并引发了关于无审查模型版本的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Joseph717171/Llama-3.2-3B-Instruct-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF/resolve/main/Llama-3.2-3B-Instruct-OF32.EF32.IQ8_0.gguf">未找到标题</a>：未找到描述</li><li><a href="https://imgur.com/a/suUJuyV">imgur.com</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐平台。通过幽默的笑话、流行的梗图、有趣的 GIF、励志故事、病毒视频等来振奋精神...</li><li><a href="https://imgur.com/a/88T9yJI">imgur.com</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐平台。通过幽默的笑话、流行的梗图、有趣的 GIF、励志故事、病毒视频等来振奋精神...</li><li><a href="https://medium.com/@mlabonne/uncensor-any-llm-with-abliteration-d30148b7d43e">通过 Abliteration 取消任何 LLM 的审查</a>：无需重新训练的微调</li><li><a href="https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19">Molmo - allenai 集合</a>：未找到描述</li><li><a href="https://huggingface.co/collections/Joseph717171/gguf-llama-32-instruct-oq8-0-f32ef32iq4-k-q8-0-iquants-66f49cf761597948c6fa789d">GGUF Llama-3.2-Instruct-OQ8_0-F32.EF32.IQ4_K-Q8_0 IQuants - Joseph717171 集合</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fptwvm/molmo_a_new_model_that_outperforms_llama_32/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/chigkim/Ollama-MMLU-Pro">GitHub - chigkim/Ollama-MMLU-Pro</a>：通过创建账号为 chigkim/Ollama-MMLU-Pro 的开发做出贡献。</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1fpckrw/qwen25_selfreported_now_on_official_mmlupro/">Qwen2.5 (自报数据) 现已登上官方 MMLU-Pro 排行榜，击败 Gemini 1.5 Pro 和 Claude 3 Opus</a>：https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro</li><li><a href="https://github.com/ollama/ollama/pull/6963">pdevine 为 llama3.2 增加的图像处理 · Pull Request #6963 · ollama/ollama</a>：用于运行 llama3.2 的图像处理例程。未来可能需要重构以支持其他多模态模型。</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8010#issuecomment-2376339571">server: 恢复多模态支持 · Issue #8010 · ggerganov/llama.cpp</a>：自 #5882 以来多模态已被移除，取决于 llava 的重构，我们将能够恢复支持：#6027。创建此 Issue 主要是为了跟踪目的。如果有人想...</li><li><a href="https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1">视觉模型 (GGUF) - lmstudio-ai 集合</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1288781696388431883)** (165 条消息🔥🔥): 

> - `针对硬件的 LLM 推荐`
> - `不同 GPU 上的性能表现`
> - `VRAM 限制`
> - `多 GPU 集成`
> - `即将发布的 GPU` 


- **关于硬件适用 LLM 的讨论**：成员们讨论了在配备 Intel i7-8750H 和 32GB RAM 的系统上运行 LLM 的建议，建议关注 Qwen 2.5 等模型。
   - 同时也提出了使用集成 Intel GPU 的选项，但指出由于依赖系统 RAM，速度会受到限制。
- **不同 GPU 上的性能指标**：分享了性能基准测试，结果显示 AMD Radeon RX 5700 XT 约为 **35 tokens/sec**，配备 NVIDIA GeForce RTX 4060 的 Surface Laptop Studio 2 约为 **40 tokens/sec**。
   - 其他用户指出 Apple M3 Max 芯片达到了 **61 tokens/sec**，凸显了不同硬件之间显著的性能差异。
- **模型部署的 VRAM 限制**：聊天参与者一致认为 VRAM 是大模型的关键限制因素，**24GB** 的 GPU 在高效运行 **70B** 模型时显得力不从心。
   - 讨论还涉及了低 VRAM 模型及其性能权衡，并表示 **15 tok/s** 是可接受的使用速度。
- **NVIDIA 与 AMD GPU 的集成**：用户讨论了在单一配置中同时使用 NVIDIA 和 AMD GPU 的挑战，以及这种组合可能如何限制性能。
   - 共识是软件兼容性可能决定哪些模型能从多 GPU 设置中受益。
- **对未来 GPU 发布的期待**：讨论了即将推出的 GPU，如 NVIDIA GeForce RTX **5090**，传闻其拥有 **32 GB** VRAM 和高性能规格。
   - 用户对未来型号相较于当前一代产品的价格承受能力表示担忧，同时也讨论了二手 GPU 的可行性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/14gnkfw/think_twice_about_getting_the_rtx_4060_ti/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://wccftech.com/nvidia-geforce-rtx-5090-32-gb-rtx-5080-16-gb-specs-5090-20k-cores-600w-5080-10k-cores-400w/">NVIDIA GeForce RTX 5090 32 GB &amp; RTX 5080 16 GB 规格曝光：5090 超过 20K 核心 &amp; 600W，5080 超过 10K 核心 &amp; 400W</a>：Kopite7kimi 揭晓了 NVIDIA 为玩家打造的下一代 GPU —— GeForce RTX 5090 和 RTX 5080 的规格。</li><li><a href="https://wccftech.com/nvidia-geforce-rtx-5090-32-gb-rtx-5080-16-gb-specs-5090-20k-cores-600w-5080-10k">NVIDIA GeForce RTX 5090 32 GB &amp; RTX 5080 16 GB 规格曝光：5090 超过 20K 核心 &amp; 600W，5080 超过 10K 核心 &amp; 400W</a>：Kopite7kimi 揭晓了 NVIDIA 为玩家打造的下一代 GPU —— GeForce RTX 5090 和 RTX 5080 的规格。</li><li><a href="https://videocardz.com/newz/nvidia-geforce-rtx-5090-and-rtx-5080-specs-leaked">NVIDIA GeForce RTX 5090 和 RTX 5080 规格泄露 - VideoCardz.com</a>：GeForce RTX 5090 将配备 21760 个 CUDA 核心、32GB GDDR7 显存和 600W 功耗，RTX 5080 配备 16GB VRAM。消息来自 Kopite7kimi 本人，他是最可靠的 NVIDIA 爆料者之一，现已确认相关规格...</li><li><a href="https://www.canadacomputers.com/index.php?cPath=43_557_559&sf=:3_22&co=&mfr=&pr=">选购 Powered By Nvidia 及更多产品 - Canada Computers</a>：未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1288580759694868500)** (230 条消息🔥🔥): 

> - `Aider 更新`
> - `Aider 中的模型角色`
> - `用户体验改进`
> - `Aider 模式`
> - `配置选项` 


- **Aider 引入 Senior 和 Junior 模式**：Aider 的最新更新为模型引入了 “Senior” 和 “Junior” 角色，通过将职责划分为规划与执行来增强编码过程。
   - 用户表示需要明确这些角色的运作方式，一些人建议使用 “Planner” 和 “Executor” 等替代术语以减少混淆。
- **优化 Aider 的用户体验**：讨论了如何让 Aider 的两步流程变为可选，从而在保留通过 Senior/Junior 配置进行规划的能力的同时，允许使用更快的模式进行快速编辑。
   - 提议包括使用 `/fast` 等命令来高效切换模式，并根据用户需求改进模型选择过程。
- **Aider 的最佳模型组合**：用户讨论了 Senior 和 Junior 角色的最佳组合，探索了包括 OpenAI 的 o1-preview、Claude 3.5 Sonnet 和 Deepseek 在内的各种模型。
   - 默认设置的建议是使用 o1-preview 担任 Senior 职位，Sonnet 担任 Junior 职位，同时也在考虑将快速的 Deepseek 模型用于实现任务。
- **用户对 Aider 功能的偏好**：用户更倾向于让 Aider 的默认功能保持快速和简单，而高级功能则在需要故障排除时按需提供。
   - 其核心思想是在保持最新更新的同时，确保现有用户在处理简单的编码需求时仍能获得原始的 Aider 体验。
- **反馈与持续改进**：社区强调了用户对新功能反馈的重要性，承认需要简化 UX 和模型交互。
   - 建议包括允许 Aider 智能检测重复的提示词，并推荐适当的模式转换以提高用户效率。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.napkins.dev/">Napkins.dev – Screenshot to code</a>: 通过截图生成你的下一个应用</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/aider/website/_posts/2024-09-26-senior-junior.md">aider/aider/website/_posts/2024-09-26-senior-junior.md at main · paul-gauthier/aider</a>: aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/tldraw/make-real">GitHub - tldraw/make-real: Draw a ui and make it real</a>: 绘制 UI 并将其变为现实。通过在 GitHub 上创建账号来为 tldraw/make-real 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider.git">GitHub - paul-gauthier/aider: aider is AI pair programming in your terminal</a>: aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://fireworks.ai/blog/cursor">How Cursor built Fast Apply using the Speculative Decoding API </a>: AI 原生 IDE Cursor 利用 Fireworks 推理栈增强了其 Instant Apply、Smart Rewrites 和 Cursor Prediction 等功能。该博文介绍了 Speculative Decoding API...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1288617146133385238)** (21 messages🔥): 

> - `Augmenting model inputs` (增强模型输入)
> - `Issues with Sonnet performance` (Sonnet 性能问题)
> - `Using Aider with GUI frameworks` (在 GUI 框架中使用 Aider)
> - `Dependency update tools` (依赖更新工具)
> - `File creation issues in Aider` (Aider 中的文件创建问题)


- **增强代码生成的模型输入**：用户表达了 LLM 在生成正确代码方面的挑战，特别是需要来自外部库和工具（如针对 **Rust** 等语言的分析器）的提示。
   - 一位用户评论了在代码生成过程中准确反映 **typing** 和 **lifetime** 规则的重要性。
- **对 Sonnet 可靠性的担忧**：一名成员对 **Sonnet** 的**不可靠性**提出了担忧，并观察到尽管没有明显变化，但其表现比平时更差。
   - 社区讨论了持续存在的问题，其中一人将其归因于另一个系统中正在修复的 Bug 影响了 Sonnet 的性能。
- **在 GUI 中有效使用 Aider**：一位用户询问如何改善在使用 **Streamlit** 时的反馈循环，旨在寻求除基本命令和错误报告之外的更多交互。
   - 另一位用户回应称，Streamlit 的局限性将需要重新设计 Aider 以实现交互式前端功能。
- **使用 Mend 自动更新依赖**：讨论了 **Mend Renovate**，它通过检测较新的软件包版本并将更新集成到代码中来自动执行依赖项更新。
   - 虽然一位用户了解此工具，但也有人表示希望 LLM 能够独立地更好地处理软件包版本管理。
- **Aider 的文件创建问题**：一位用户报告了 Aider 在创建或编辑文件时不一致的问题，尽管它指示正在执行这些操作。
   - 这引发了关于如何确保 Aider 默认在指定目录中可靠地创建和修改文件的疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/faq.html#how-do-i-turn-on-the-repository-map">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://www.mend.io/renovate/">Mend Renovate Products</a>：使用 Mend Renovate 产品自动更新您的依赖项。向仓库提交 Pull Request 并放心地合并更新。</li><li><a href="https://github.com/zed-industries/zed/pull/18363">Fix sending alt-enter in terminal by notpeter · Pull Request #18363 · zed-industries/zed</a>：在此报告：如何将 option-enter (alt-enter) 转发到终端？#18149。修复前：peter@scruffy zed % fish_key_reader，按下按键：bind \eenter 'do something'。修复后：peter@scruffy z...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

fry69_61685: https://simonwillison.net/2024/Sep/25/o1-preview-llm/
  

---



### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1288615728127148032)** (1 messages): 

> - `Hermes 3`
> - `Llama-3.1 8B` 


- **Hermes 3 在 HuggingChat 上发布**：Nous Research 在 [HuggingChat](https://huggingface.co/chat/settings/NousResearch/Hermes-3-Llama-3.1-8B) 上发布了参数量为 **8B** 的 **Hermes 3** 模型，该模型能够严格遵循指令。
   - 此次发布展示了 **Hermes** 系列的重大进展，使该模型可供社区随时使用。
- **Hermes 3 的主要特性**：**8B** 规格的 **Hermes 3** 模型旨在紧密遵循用户指令并增强交互能力。
   - 这与 **Nous Research** 项目中为提供更具响应性的 AI 解决方案而进行的持续改进工作相一致。



**提到的链接**：<a href="https://huggingface.co/chat/settings/NousResearch/Hermes-3-Llama-3.1-8B">HuggingChat</a>：让社区最好的 AI 聊天模型可供所有人使用。

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1288576185357570131)** (224 messages🔥🔥): 

> - `Nous Hermes 3`
> - `Llama 3.2 models`
> - `GPU advancements` (GPU 进展)
> - `Generative AI concerns` (生成式 AI 担忧)
> - `AI event in Singapore` (新加坡 AI 活动)

- **关于 Nous Hermes 3 和 Steerability 的担忧**：用户对 Nous Hermes 3 的 Steerability（可控性）和审查制度表示困惑，将其设计预期与实际部署中的限制进行了对比。
   - 讨论内容包括 system prompts 的影响，以及模型基于其预设配置生成的响应的僵化性。
- **Llama 3.2 3B Base 模型的可用性**：有关于 GGUF 格式的 Llama 3.2 3B Base 模型可用性的询问，用户质疑其是否已经发布。
   - 几位用户表示有兴趣对该模型进行 finetuning 以获得更好的性能。
- **RTX 5090 规格与预期**：讨论了 RTX 5090，强调了其预期的 32GB 显存以及未来升级到 48GB 的潜力，确保其具有吸引力。
   - 用户推测了即将推出的 RTX 系列显卡的显存配置及其对性能的影响。
- **二手 GPU 市场**：一位用户寻求购买二手 GPU，特别是像 A6000 这样的 Ada 系列设备，其他用户则提供了较低端的选项或分享了个人升级情况。
   - 讨论强调了在当前技术进步背景下，二手市场中高性能 GPU 的稀缺性。
- **新加坡 AI 活动**：披露了即将在新加坡举行的一场 AI 活动，届时将有来自 AI 领域新兴公司的 lightning talks，以及社交互动的机会。
   - 该活动旨在展示创新的 AI 工具以及对未来技术发展的见解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/voooooogel/status/1839144530995450346">来自 thebes (@voooooogel) 的推文</a>：让 Advanced Voice Mode 疯狂重复 "wheedle" 这个词，以至于它开始自言自语，最后因为违反准则被切断了。</li><li><a href="https://huggingface.co/Joseph717171/Llama-3.2-1B-Instruct-OQ8_0.EF32.IQ4_K-Q8_0-GGUF/resolve/main/Llama-3.2-1B-Instruct-OQ8_0.EF32.IQ8_0.gguf">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/chat/settings/NousResearch/Hermes-3-Llama-3.1-8B">HuggingChat</a>：让社区最好的 AI 聊天模型惠及每一个人。</li><li><a href="https://huggingface.co/spaces/huggingface-projects/llama-3.2-3B-Instruct">Llama 3.2 3B Instruct - 由 huggingface-projects 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/teknium1/status/1839040366512844917?s=46">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：如果你没能在 OpenAI 的 Advanced Voice Mode 中体验到《她》(HER) 的感觉——好吧——我在 http://Play.AI 的朋友们带来了货真价实的 HERmes。在这里尝试：https://play.ai/agent/HERMES-m3i3jU81_52ruL6_0tw2R</li><li><a href="https://x.com/teknium1/status/1839040366512844917?s=4">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：如果你没能在 OpenAI 的 Advanced Voice Mode 中体验到《她》(HER) 的感觉——好吧——我在 http://Play.AI 的朋友们带来了货真价实的 HERmes。在这里尝试：https://play.ai/agent/HERMES-m3i3jU81_52ruL6_0tw2R</li><li><a href="https://huggingface.co/collections/alpindale/llama-32-re-upload-66f463d7940e8a6c7f5b7bbc">Llama 3.2 Re-upload - alpindale 的收藏集</a>：未找到描述</li><li><a href="https://x.com/kopite7kimi/status/1839343725727941060?s=19">来自 kopite7kimi (@kopite7kimi) 的推文</a>：GeForce RTX 5090 PG144/145-SKU30 GB202-300-A1 21760FP32 512-bit GDDR7 32G 600W</li><li><a href="https://console.groq.com/docs/models">GroqCloud</a>：体验世界上最快的推理速度</li><li><a href="https://huggingface.co/collections/Joseph717171/gguf-llama-32-instruct-oq8-0-f32ef32iq4-k-q8-0-iquants-66f49cf761597948c6fa789d">GGUF Llama-3.2-Instruct-OQ8_0-F32.EF32.IQ4_K-Q8_0 IQuants - Joseph717171 的收藏集</a>：未找到描述</li><li><a href="https://x.com/OfficialLoganK/status/1839310682530959367?t=jJkHuEtuJlc956Q58ZdNjw&s=19">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：地堡的备用智能</li><li><a href="https://lu.ma/mlqwqi6x">新加坡 AI 展示会 · Luma</a>：欢迎参加 10 月 8 日举行的独家 AI 展示会，拉开新加坡科技周 (TechWeek Singapore) 的序幕。本次活动将邀请创新的 AI 团队，每支团队都将进行闪电演讲……</li><li><a href="https://x.com/JStaatsCPA/status/1838984688917954621?t=SRmzuonj2li2suACqaEWUA&s=19">来自 Jason Staats⚡ (@JStaatsCPA) 的推文</a>：这个全新的 ChatGPT Advanced Voice Mode 🤯🤯</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1fp9wem/llama_32_1b_3b_benchmarks/)">Llama 3.2 1B &amp; 3B 基准测试</a>：由 u/TKGaming_11 发布于 r/LocalLLaMA • 122 点赞和 20 条评论</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1fpb4m3/molmo_models_outperform_llama_32_in_most_vision/">Molmo 模型在大多数视觉基准测试中表现优于 Llama 3.2 🌟</a>：由 u/shrewdeenger 发布于 r/LocalLLaMA • 204 点赞和 27 条评论</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-llm/">Qwen2.5-LLM：扩展 LLM 的边界</a>：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介 在这篇博客中，我们将深入探讨最新的 Qwen2.5 系列语言模型的细节。我们开发了一系列仅解码器 (decoder-only) 的稠密模型……
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1288612439390294086)** (17 messages🔥): 

> - `Llama 3.2 Vision Encoder`
> - `Llama 3.2 的推理需求`
> - `Llama 3.2 模型可用性`
> - `DisTRo 优化器`
> - `Hermes 3 中的判断与奖励建模` 


- **Llama 3.2 Vision Encoder 规模巨大**：**Llama 3.2 Vision Encoder** 的尺寸非常可观，**11B 模型**的 Encoder 接近 **3B** 参数，而 **90B 模型**的则达到了 **18B**。
   - 成员们使用 *Gigantic*（巨大）一词来强调该 Encoder 的规模。
- **90B Llama 3.2 的推理 GPU 需求**：一位成员建议 **3x H100 GPU** 应该足以推理 **90B Llama 3.2**，尽管为了张量并行（tensor parallelism）或更大的 batch sizes 可能需要 **4x GPU**。
   - 推理问题指向了在 **Runpod** 上实际的 GPU 配置方案。
- **在欧盟访问新的 Llama 3.2 模型**：新的 **Llama 3.2 模型** 现在可以通过 **Hugging Face** 上的微调版本和量化副本（例如 **Unsloth 的集合**）在欧盟下载。
   - 此外，**Alpindale 已经重新上传了未经修改的权重**，以便于访问。
- **关于 DisTRo 优化器实现的问题**：一位成员对 **DisTRo 优化器** 表示了兴趣，并询问在它可用之前，如何使用 **Elixir** 构建可靠的基础设施。
   - 他们考虑最初使用标准的优化器，如 **SGD 或 ADAM**，同时为以后更好的集成做准备。
- **Hermes 3 的判断与奖励建模**：有关于 **Hermes 3** 在**判断与奖励建模（judgement and reward modelling）**方面的改进，以及 SFT 数据集中是否使用了合成数据的询问。
   - 官方确认这些改进是由**合成数据**支持的，而不是仅仅依赖于公开数据集。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

trre: https://arxiv.org/abs/2409.16897
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1288590690539929713)** (2 messages): 

> - `Wordware 应用更新`
> - `Opus Insight 增强功能`
> - `ThinkLab 特性`
> - `O1-Preview 限制` 


- **Wordware 应用更新支持 O1Mini**：主站 Wordware 应用的更新版本现在包含了 **O1Mini**，增强了功能和评价表现。
   - **Opus Insight** 模板利用 **Sonnet 3.5** 进行初步评价，随后由 O1Mini 进行全面的模型排名。
- **用于扩展搜索的 ThinkLab**：**ThinkLab** 由 **Sonar Huge 405b** 驱动，专注于 scratchpad 的使用以及随后的搜索，以进行更广泛的探索。
   - 该工具旨在通过高效的搜索能力简化用户的探索过程。
- **O1-Preview 速率限制挑战**：**O1-Preview** 是 Wordware 应用中的一个选项，但它受到**速率限制（rate limited）**，如果无法返回数据，可能会导致应用性能停滞。
   - 尽管它已被添加到 Wordware 应用中，但目前仍保持**禁用（disabled）**状态以避免停滞，但用户可以创建自己的版本来实验专用的 O1-Preview 工作流。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://app.wordware.ai/explore/apps/aa2996a0-93c9-4c19-ade2-1796c5c8a409">OPUS Insight : 最新模型排名 - o1mini</a>：此 Prompt 使用最新模型处理问题，并提供全面的评价和排名。更新：2024/9/25 - 已添加：o1mini, Gemini 1.5 Flash, Command R+。注意：o1-preview 是其中的一部分...</li><li><a href="https://app.wordware.ai/explore/apps/999cc252-5181-42b9-a6d3-060b4e9f858d">_Think-Lab 修订版 - o1mini</a>：(版本 1.10) 利用 ScratchPad-Think 的力量进行日常网络搜索。以 JSON 格式导出精炼的搜索查询。scratchpad 是一个强大的工具，可以帮助你保持连贯性和准确性...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

trre: https://arxiv.org/abs/2409.16897
  

---



### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1288753349172133888)** (5 messages): 

> - `诈骗链接识别`
> - `教科书讨论` 


- **用户识别诈骗链接**：成员们对一个潜在的欺诈链接表示担忧，强调这**绝对是一个骗局**。
   - 一名用户标记了相关角色以提醒他人，随后对发布该链接的用户采取了行动。
- **PMPP 作为课堂教科书**：一位成员提到 **PMPP** 是他们的课堂教科书，并使用 emoji 表情表达了热情。
   - 这一评论突显了 **PMPP** 在小组内正在进行的教育讨论中的相关性。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1288699558334959646)** (6 messages): 

> - `Triton Conference 2024`
> - `Proton kernel 优化`
> - `适用于 Windows 的 Triton 编译版 wheel` 


- **Triton Conference 2024 视频发布**：[Triton Conference 2024](https://www.youtube.com/watch?v=NZz5sczZ_30) 的录像现已发布，包含由 OpenAI 的 Keren Zhou 主讲的上午场等多个环节。
   - 下午场包含由 Aparna Ramani 主讲的 Meta Triton 策略主题演讲，可通过[此链接](https://www.youtube.com/watch?v=ONrKkI7KhU4)观看。
- **Proton 与 Kernel 调用优化讨论**：一位用户注意到 proton-viewer 中显示的 Triton kernel 数量少于其 output_code.py 中的数量，并询问是否在驱动层发生了优化。
   - 另一位用户澄清说，每一个 `kernel[grid](args...)` 都保证是一个 kernel 调用，而 **proton** 会将这些调用组合在一起。
- **寻求适用于 Windows 的 Triton 编译版 Wheel**：一位成员正在寻找与 **Windows** 和 **Python 3.10** 兼容的最新 Triton 编译版 wheel。
   - 这一需求突显了 Triton 社区对更新安装包的持续需求。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=NZz5sczZ_30">Triton Conference 2024: Morning Session</a>: 教程 2:30 开发工具：Proton/Interpreter, Keren Zhou (George Mason University &amp; OpenAI) 幻灯片: https://bit.ly/proton-interpreter-tutorial 1:04:24 编译...</li><li><a href="https://www.youtube.com/watch?v=ONrKkI7KhU4">Triton Conference 2024: Afternoon Session</a>: 1:25 欢迎致辞: Ajit Mathews, META 6:05 主题演讲: Triton Strategy at Meta, Aparna Ramani, META https://bit.ly/aparna-keynote 17:05 主题演讲: Triton State of the Un...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1288631725509050452)** (9 messages🔥): 

> - `PyTorch 内存分析 (Memory Profiling)`
> - `TorchDispatchMode`
> - `Autograd 优化`
> - `PyTorch Profiler`
> - `内存可视化` 


- **调查 PyTorch 层中的内存分配**：成员们讨论了在 **PyTorch** 中检查每一层内存分配的方法，并询问了有关 **weights**、**grads** 和 **optimizer states** 的内存数据。
   - *Memorypaladin* 引用了过去会议的一张幻灯片，暗示该问题有肯定的答案。
- **手动与自动内存分析技术**：一位成员提到，可以通过在 PyTorch 中点击特定层的可视化界面来手动获取内存分配信息。
   - *p0.tato* 强调有一种使用 **torchdispatchmode** 解析内存数据的自动化方法。
- **分享关于 Autograd 优化的有趣 GitHub Issue**：一位成员分享了一个 [GitHub issue](https://github.com/pytorch/pytorch/issues/136733)，讨论了与 autograd 公式相关的优化，并指出这可能取决于输入或输出。
   - 他们认为这个优化非常有趣，值得探索。
- **使用 PyTorch Profiler 获取内存数据**：一位成员建议使用 **PyTorch Profiler** API 来获取内存使用的总体概览，并称赞了其彩色的可视化效果。
   - 然而，他们指出，虽然这对于理解很有帮助，但与用于调试的内存快照 (memory snapshot) 方法相比，它缺乏交互性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile.export_memory_timeline,">torch.profiler &mdash; PyTorch 2.4 文档</a>: 未找到描述</li><li><a href="https://github.com/pytorch/pytorch/issues/136733">Be smart about autograd formulas saving either the input or output, depending on context · Issue #136733 · pytorch/pytorch</a>: 🚀 特性、动机与构想 参见 NVIDIA/apex#1715。其核心思想是，对于某些算子，原则上 autograd 公式可以根据输入或输出来编写。F.....
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1288584505241899079)** (2 messages): 

> - `Llama 3.2 Models`
> - `Llama 3.1 Impact`
> - `Edge and Mobile AI`
> - `Availability Issues` 


- **Llama 3.2 带来新模型**：Meta 宣布发布 **Llama 3.2**，其特点是推出了针对 **edge and mobile devices**（边缘和移动设备）优化的新型中小型视觉 LLM（11B 和 90B）以及轻量级文本模型（1B 和 3B）。
   - 这一扩展旨在为开发者提供更易获得的资源，而无需庞大的计算能力。
- **Llama 3.1 模型使用量翻倍**：自 **Llama 3.1** 发布以来，其影响力在两个月内见证了使用量**翻倍**，展示了该模型作为拥有 **405B 参数**的领先开放前沿级 AI 的实力。
   - Meta 的 **Zuckerberg** 在 Connect 活动上强调了这些模型性能所带来的震撼。
- **为提高效率而剪枝的新模型**：Llama 3.2 的 **1B 和 3B 文本模型**是从较大的 **8B 和 70B** 模型中经过剪枝（Pruned）和蒸馏（Distilled）而来的版本，专门面向资源有限的开发者。
   - 这种战略性的剪枝允许在计算能力受限的设备上运行应用程序。
- **欧盟可用性限制**：**Llama 3.2** 的 11B 和 90B 视觉模型在**欧盟国家**不可用，限制了该地区开发者获取这些模型的途径。
   - 这一限制引起了渴望利用这些强大工具的欧盟开发者的关注。



**Link mentioned**: <a href="https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/">no title found</a>: no description found

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1288617009235361875)** (8 messages🔥): 

> - `CUDA project setup`
> - `clangd configuration issues`
> - `RTX 3070 compute capabilities`
> - `Neovim vs. other IDEs`
> - `Channel organization` 


- **为集中讨论创建频道**：创建了一个新频道以确保讨论保持在正确的地方，让成员能够集中精力进行交流。
   - *Thanks will repost it there*（谢谢，我会在那里重新发布）确认了文章将分享到指定频道，以获得更好的上下文。
- **CUDA 项目中 clangd 的问题**：一位新用户在 Neovim 中使用 clangd 设置 CUDA 项目时遇到问题，诊断输出显示了各种不支持的选项。
   - 成员们指出 clangd 不接受与 nvcc 相同的选项，强调了这可能是一个混淆点。
- **RTX 3070 计算能力（compute capability）澄清**：讨论围绕用户的 RTX 3070 被错误设置为 `sm_20` 展开，而它应该是 `sm_86`。
   - 澄清计算能力应该能解决之前提到的一些编译问题，包括版本冲突。
- **探索 Neovim 的替代方案**：一位成员正在考虑使用 Visual Studio Code 以利用 CUDA 工具链接口，从而获得更好的易用性。
   - 讨论表明，对于大型项目，相比使用 Neovim，人们更倾向于使用像 GDB 和其他 CUDA profilers 这样更有效的工具。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1288975422062723092)** (1 messages): 

> - `Reading Images in C`
> - `Chapter 3 Overview` 


- **在 C 语言中读取图像的技巧**：一位成员询问了关于第 3 章中使用 **C** 语言读取图像的建议，该章节内容包含图像处理。
   - 消息中未提供具体建议，但该问题表明了对章节内实际应用的关注。
- **第 3 章包含视觉内容**：讨论指出第 3 章利用**图像**来增强对材料的理解。
   - 这种方法表明了一个实践环节，其中视觉元素在学习体验中起着至关重要的作用。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1288648844287934516)** (49 条消息🔥): 

> - `torchao 兼容性`
> - `CogVideoX 上的 FP8`
> - `Windows 对 ML 的支持`
> - `NVIDIA 训练问题`
> - `关于开发的社区讨论` 


- **torchao 在 NVIDIA 上遇到困难**：一位用户报告称 `torchao` 在 NVIDIA 硬件上无法正常运行，在训练过程中遇到了 dtype 转换错误。
   - 尽管能够在 RTX 3060 上使用 FP8 进行推理，但在转向训练场景时出现了问题。
- **关于 Windows 支持重要性的辩论**：几位用户讨论了 Windows 对 ML 框架支持的相关性，一些人断言不使用 Linux 的用户对此表现出越来越浓厚的兴趣。
   - 虽然有人认为过分强调 Linux 是偏颇的，但另一位指出目前的实现主要依赖于以 Linux 为中心的 Triton。
- **CogVideoX 中 FP8 的挑战**：一位用户指出 CogVideoX 需要 H100 硬件才能实现 FP8 功能，强调了某些模型在兼容性方面的局限性。
   - 用户对于在特定配置下（尤其是利用 NVIDIA 平台等功能时）使用 FP8 存在的障碍感到沮丧。
- **社区关于开发者责任的紧张局势**：讨论变得激烈，一名用户批评开发者不提供 Windows 支持，促使其他社区成员为开发者的选择和方法辩护。
   - 这次对话凸显了对开发者期望所带来的挫败感，特别是关于支持多种操作系统的期望。
- **训练方法的澄清**：一位用户分享了他们在不使用 autocast 的情况下初始化训练的方法，利用 bf16 权重，但在低精度优化器需要梯度上采样（upcasting）时面临挑战。
   - 这引发了对许多人在尝试优化 ML 训练性能时面临的技术障碍的深入见解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://scholar.google.com/citations?user=_2_KAUsAAAAJ">Furkan Gözükara</a>：Toros 大学助理教授，计算机工程师 - 被引用 20 次 - 数据挖掘 - 情感分析 - 文本分类 - 产品聚类 - 聚类</li><li><a href="https://github.com/pytorch/ao/issues/957">这只适用于 linux 吗？ · Issue #957 · pytorch/ao</a>：我安装在 windows 上，从 torchao.quantization import quantize_ 报错。pip freeze Microsoft Windows [版本 10.0.19045.4894] (c) Microsoft Corporation。保留所有权利。R:\CogVideoX_v1\...</li><li><a href="https://huggingface.co/THUDM/CogVideoX-5b">THUDM/CogVideoX-5b · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/bghira/SimpleTuner/pull/986/files#diff-327015d4d445c4efaaa945a93701df4c68e3bc401dc4ddb7e55f2b5dc7854d6fR103-R116>">(进行中，尚不可用) torchao: fp8/int8 by bghira · Pull Request #986 · bghira/SimpleTuner</a>：未找到描述</li><li><a href="https://github.com/pytorch/pytorch/blob/b408591b5380d6b856e0a4ce32cf386c56660c54/torch/_inductor/kernel/mm_scaled.py#L110-L190">pytorch/torch/_inductor/kernel/mm_scaled.py at b408591b5380d6b856e0a4ce32cf386c56660c54 · pytorch/pytorch</a>：Python 中具有强大 GPU 加速的张量和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/ao/pull/276/files#diff-9f968cc00e2ee60006f8747d55f99cb54f367ca98f8d360731d306ab1d5db2b4L239)">由 HDCharles 向 TorchAO 添加 Llama · Pull Request #276 · pytorch/ao</a>：摘要：此 PR 为 torchao 代码库中的 llama 模型添加了稳定的评估/基准测试功能。模型相关内容位于 torchao/_models/llama，评估已移至 _models/_eval.py...</li><li><a href="https://github.com/p">p - 概览</a>：p 有 153 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py">llama-models/models/llama3/reference_impl/model.py at main · meta-llama/llama-models</a>：旨在与 Llama 模型配合使用的实用程序。通过在 GitHub 上创建账户为 meta-llama/llama-models 开发做出贡献。</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L266">gpt-fast/model.py at main · pytorch-labs/gpt-fast</a>：简单高效的 PyTorch 原生 Transformer 文本生成，Python 代码少于 1000 行。- pytorch-labs/gpt-fast
</li>
</ul>

</div>

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1288576049625567384)** (10 messages🔥): 

> - `Easy to Use GPU Solutions` (易于使用的 GPU 解决方案)
> - `Prime Intellect on Lambda` (Lambda 上的 Prime Intellect)
> - `IRL Hackathon Planning` (IRL Hackathon 策划)
> - `Edge LLM Challenge` (Edge LLM 挑战赛)
> - `Team Collaboration for Challenges` (挑战赛团队协作)


- **GPU 解决方案证明了其用户友好性**：一位用户提到，根据他们的经验，某款特定的 GPU 是**最易于使用的 (easiest to use)**，并建议在考虑从 **3070** 升级时，可以将其视为一次**成功的转换 (successful conversion)**。
   - 这表明用户寻求更易获得的 GPU 选项的趋势可能正在增长。
- **Prime Intellect 利用 Lambda**：据分享，**Prime 运行在 Lambda 之上**，并可以通过 **prime-intellect** 进行访问。
   - 这突显了在用户工作流中集成用于 GPU 处理的云解决方案。
- **社区组织首个 IRL Hackathon**：一场关于与 **GPU Mode Discord 服务器**成员共同组织**首个此类 Hackathon** 的讨论已经展开。
   - 该活动旨在将成员聚集在 IRL（线下），促进围绕 GPU 进展的社区参与。
- **Edge LLM 挑战赛需要创新**：**Edge LLM Challenge** 包含让团队为 **Phi-2** 和 **Llama-3-8B** 等模型开发压缩方法的任务，重点是使其能在**智能手机 (smartphones)** 上运行。
   - 参与者还被鼓励从头开始训练语言模型，从而在 AI 模型训练中培养创造力和创新。
- **为 Edge 挑战赛寻找队伍**：一位用户表达了为 Edge LLM Challenge **组队**的愿望，以促进参与者之间的协作。
   - 另一位用户确认已分享了挑战赛的**有用信息**，鼓励团队合作和知识共享。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://edge-llms-challenge.github.io/edge-llm-challenge.github.io/challenge">未找到标题</a>：未找到描述</li><li><a href="https://x.com/caseyaylward/status/1839358642241536262">来自 Casey Aylward (@caseyaylward) 的推文</a>：1/ 几个月前，一些来自 GPU Mode (原名 CUDA Mode) Discord 服务器的朋友和我开始讨论将社区带到 IRL，以策划一场首创的 Hackathon 🧵
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1288969786352472065)** (1 messages): 

> - `Guatemala Meetups` (危地马拉见面会)
> - `Regional Connections` (区域联系) 


- **危地马拉见面会公开招募**：一位成员表达了在**危地马拉**组织见面会的兴趣，鼓励附近的成员进行联系。
   - 他们还提到愿意与来自**伯利兹 (Belize)** 或**墨西哥 (Mexico)** 的成员见面。
- **寻求跨国界联系**：该成员正在接触潜在的见面会参与者，强调了该地区**社区重要性**。
   - 这一倡议旨在加强当地爱好者和专业人士之间的联系。


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1288866450110550210)** (2 messages): 

> - `Numba`
> - `Convolution and Deep Learning` (卷积与深度学习)
> - `Attention and Transformers` 


- **Numba 的深度学习章节**：*mr.osophy* 指出讨论实际上是关于 **Numba** 的，并指出它包含一个专门讨论**卷积和深度学习 (convolution and deep learning)** 的章节。
   - 他们质疑在此背景下为何没有提及 **Attention** 或 **Transformers**。
- **Numba 讨论中缺失 Attention**：讨论强调了 Numba 关注点中的一个潜在疏忽，即未提及在现代深度学习中至关重要的 **Attention 机制**。
   - *mr.osophy* 对缺乏 **Transformers**（该领域的基础课题）的相关覆盖表示惊讶。


  

---

### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1288576075860938793)** (25 messages🔥): 

> - `RoPE 前向集成`
> - `Fused RMSNorm 实现`
> - `SwiGLU 性能`
> - `前向传播匹配`
> - `REPKV 反向 Kernel 评审` 


- **RoPE 前向集成成功**：RoPE 前向集成已添加，并确认运行顺畅。
   - 该集成已推送到分支，准备进行进一步测试。
- **Fused RMSNorm 实现完成**：Fused **RMSNorm** 已成功集成并正常运行。
   - 下一步是实现 **SwiGLU** 层。
- **SwiGLU 层确认工作正常**：**SwiGLU** 现已投入使用，标志着开发进程又向前迈进了一步。
   - 开发者即将实现模型的完整前向传播。
- **前向传播与 PyTorch 匹配**：**Llama 3.1** 的前向传播结果现在与 **PyTorch** 一致，这是一个重要的里程碑。
   - 观察到的 Loss 约为 **0.021**，对于 8B 模型来说是可以接受的。
- **REPKV 反向 Kernel 正在评审中**：针对 `repkv_backward` Kernel 的新 PR 已准备好进行评审，目标分支为 **llama3**。
   - 该 Kernel 已更新并完成测试，目前正在寻求对 **llama3** 后续工作的进一步协助。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/764">Adding backward kernel for repkv on `llama3` branch (cudamode-irl) by insop · Pull Request #764 · karpathy/llm.c</a>：请查阅，repkv_backward 已更新并测试。一旦此 PR 合并，我将更新 repkv.cuh。抄送：@karpathy。这是一个正在进行中的 repkv 反向 Kernel，最初作为 cudamode-irl 项目启动。</li><li><a href="https://github.com/karpathy/llm.c/pull/763">rmsnorm backward simple baseline kernel by ngc92 · Pull Request #763 · karpathy/llm.c</a>：Baseline Kernel 和 CPU 版本。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1288594256478601327)** (3 messages): 

> - `GroupedGemm 示例`
> - `架构开发` 


- **澄清 GroupedGemm 示例选择**：一位成员指出在多个 **GroupedGemm 示例**中进行选择时存在不确定性，暗示在选择哪个示例进行实现时存在困惑。
   - 这凸显了在选择合适示例时需要更清晰的文档或指南。
- **构建初始架构**：讨论集中在将**构建一个小架构**作为开发的基础步骤。
   - *从小处着手*可能有助于简化流程并尽早发现潜在问题。


  

---


### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1288607273018589305)** (5 messages): 

> - `iPhone 16 Pro`
> - `bfloat16 启用`
> - `M4 SME 探索`
> - `Apple GPU 微架构` 


- **在 iPhone 16 Pro 上测试 Metal 基准测试**：一位用户表示有兴趣很快在他们的新 **iPhone 16 Pro** 上尝试基准测试，并引用了 GitHub 上的相关仓库：[metal-benchmarks](https://github.com/philipturner/metal-benchmarks)。
   - 该仓库探索了 **Apple GPU 微架构**，是进行性能评估的绝佳资源。
- **ExecutorCH 中启用 bfloat16**：一位用户分享了他们最近在 ExecutorCH 框架中**启用 bfloat16** 的工作，表明在性能增强方面取得了进展。
   - 这一进展可能在优化支持架构上的数据处理方面发挥关键作用。
- **Apple M4 处理器的可扩展矩阵扩展 (SME)**：讨论强调了 [m4-sme-exploration](https://github.com/tzakharko/m4-sme-exploration) 项目，该项目研究了 Apple M4 处理器的**可扩展矩阵扩展 (Scalable Matrix Extension)**。
   - 这一探索旨在将优化能力扩展到 Apple 生态系统之外，未来可能在其他 ARM 芯片上可用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/tzakharko/m4-sme-exploration">GitHub - tzakharko/m4-sme-exploration: Exploring the scalable matrix extension of the Apple M4 processor</a>：探索 Apple M4 处理器的可扩展矩阵扩展。</li><li><a href="https://github.com/philipturner/metal-benchmarks">GitHub - philipturner/metal-benchmarks: Apple GPU microarchitecture</a>：Apple GPU 微架构。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1288596491774066688)** (1 messages): 

> - `Block Scheduling`
> - `Randomization in Scheduling` 


- **Block Scheduling 表现出随机性**：一位成员指出，在实践中，**blocks** 基本上是随机调度的，这表明其缺乏可预测性。
   - 这种随机性可能会影响近期对调度过程进行的任何潜在改进。
- **对调度改进表示怀疑**：另一位成员对近期能否在调度过程中引入任何增强功能表示怀疑。
   - 鉴于目前调度的随机性质，立即发生改变的可能性似乎不大。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1288619866445582366)** (3 messages): 

> - `GPU Performance Optimization`
> - `CUDA Virtual Connect`
> - `Llama 3.2 with CuPy` 


- **提升你的 GPU 性能见解**：一篇题为 [An Introduction to GPU Performance Optimization for Deep Learning](https://www.digitalocean.com/community/tutorials/an-introduction-to-gpu-optimization) 的博客文章强调了学习最新 GPU 架构特性和性能监控工具以实现最佳使用的重要性。
   - 它强调了通过实验和基准测试（benchmarking）来实现更好硬件利用率的重要性，特别是在深度学习场景下。
- **与专家一起参加 CUDA Virtual Connect**：即将于 2024 年 9 月 27 日举行的 [CUDA Virtual Connect](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/connect-with-experts) 活动提供了一个与 NVIDIA 的 CUDA 库开发人员交流的机会。
   - 与会者可以讨论 CUDA Python 生态系统，包括 CuPy 和 Numba 等库，这对各种技能水平的开发人员都有好处。
- **Llama 3.2 现在完全利用 CuPy**：据报道，[Llama 3.2](https://github.com/githubpradeep/llm_np_cp/blob/main/llama3.2_model.py) 已完全在 GPU 上通过 CuPy 运行，这标志着性能优化方面的重大进展。
   - 这种集成强调了利用 CuPy 等库发挥 GPU 能力以提高模型效率的日益重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.digitalocean.com/community/tutorials/an-introduction-to-gpu-optimization">An Introduction to GPU Performance Optimization for Deep Learning | DigitalOcean</a>: 未找到描述</li><li><a href="https://github.com/NVIDIA/accelerated-computing-hub/tree/main/connect-with-experts">accelerated-computing-hub/connect-with-experts at main · NVIDIA/accelerated-computing-hub</a>: NVIDIA 策划的与通用 GPU 编程相关的教育资源集合。 - NVIDIA/accelerated-computing-hub
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[diffusion](https://discord.com/channels/1189498204333543425/1288899271193526342/1288899818743398420)** (11 messages🔥): 

> - `Creating a separate channel`
> - `Flux models benchmarks`
> - `DiffusionKit usage`
> - `Block diagram for Flux`
> - `Reddit diagram sharing` 


- **为持续讨论创建独立频道**：一位成员提议为正在进行的工作创建一个独立频道，并表示不能让 **LLM flop maxers** 独占所有乐趣。
   - 另一位成员回应道：*“噢不错，哈哈”*，对这一举动表示支持。
- **获取 Flux 模型的基准测试数据**：一位用户询问是否有不同硬件配置下 Flux 模型的 **benchmark numbers**（基准测试数据）。
   - 另一位成员分享了他们拥有的 **M2 Pro benchmarks**，表明人们对性能指标的兴趣日益增加。
- **测试非量化模型**：一位用户表示有兴趣测试**非量化模型**，但提到其 **16GB RAM** 限制了此类测试。
   - 这突显了用户在进行性能评估时面临的持续技术挑战。
- **请求 Flux 的架构图**：一位用户询问是否有人有 Flux 的**手绘架构图**，并建议视觉辅助工具可以增强理解。
   - 随后讨论了**方框图（block diagram）**的实用性，强调了复杂主题中视觉呈现的重要性。
- **分享来自 Reddit 的图表**：一位成员提到了之前在 **Reddit** 上分享过的一张图表，暗示它是理解 Flux 的宝贵资源。
   - 这显示了社区利用外部资源进行知识共享的倾向。



**提及的链接**：<a href="https://github.com/argmaxinc/DiffusionKit">GitHub - argmaxinc/DiffusionKit: On-device Inference of Diffusion Models for Apple Silicon</a>: Apple Silicon 上的 Diffusion 模型端侧推理 - argmaxinc/DiffusionKit

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1288576922648842405)** (135 条消息🔥🔥): 

> - `AI 艺术讨论`
> - `Stable Diffusion 用户查询`
> - `模型测试与问题`
> - `使用 RVC 进行语音转换`
> - `Lora 模型用户体验` 


- **关于用户兴趣对 AI 进步影响的辩论**：一位成员评论了许多技术进步是如何源于小众兴趣的，例如 **PonyDiffusion** 促进了新的艺术风格。
   - 这引发了关于粉丝群体对创意的影响以及对 AI 相关内容看法的讨论。
- **关于 Stable Diffusion 和 GPU 需求的查询**：一位 Stable Diffusion 新手寻求在没有 GPU 的情况下生成图像的指导，询问关于 Google Colab 的选项。
   - 其他人推荐使用 **Kaggle** 而非 Colab，因为其资源更好，并指出高效运行 Stable Diffusion 需要良好的 GPU。
- **Lora 模型的测试与问题**：一位用户在使用 Lora 模型时遇到困难，声称该模型对生成的图像没有任何效果。
   - 经过多次询问，澄清了虽然该模型产生了细微差别，但并未达到 Hugging Face 示例中预期的高质量图像。
- **Google Colab 上的语音转换查询**：一位成员请求协助在 Colab Pro 上安装 **RVC** 以进行语音转换任务。
   - 作为回应，有人指出 Hugging Face 上有许多可用的 **RVC 模型**，可能会有所帮助。
- **关于图像生成速度变动性的讨论**：一位用户观察到在本地设置中，相同参数的图像生成时间存在显著差异。
   - 这引发了关于 VRAM 使用情况和系统流量可能影响输出时间的推测。



**提到的链接**：<a href="https://huggingface.co/bingbangboom/flux_dreamscape">bingbangboom/flux_dreamscape · Hugging Face</a>：未找到描述

  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1288623850178871306)** (2 条消息): 

> - `广告政策`
> - `自学成才的 ML 工程师介绍` 


- **讨论中禁止广告**：Discord 中有严格的**禁止广告**政策，鼓励分享研究，但不允许推广公司、产品或活动。
   - 参与者被提醒参考特定频道以获取更多细节，从而提高社区规则的清晰度。
- **欢迎自学成才的 ML 工程师**：一位新成员介绍自己是正在开发初创公司的**自学成才的 ML 工程师**，并表达了关注并为社区研究做出贡献的兴趣。
   - 这一举动突显了社区成员之间协作和知识共享的愿望。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1288576112510894110)** (67 条消息🔥🔥): 

> - `高掩码率的 MLM transformer`
> - `对 Qwen2 基准测试的担忧`
> - `FP6 与 BF16 的性能对比`
> - `Hyperbolic Vision Transformer`
> - `LLM 中的逐字记忆 (Verbatim memorization)` 


- **探索 MLM 中的高掩码率**：有人提议训练一个掩码率高达 **85% 中间 token** 的 MLM transformer，并采用类似于学习率的掩码调度方法。
   - 讨论围绕此前是否有人尝试或研究过这种策略展开。
- **对 Qwen2 基准测试的担忧**：有报告称 **Qwen2 的 VL 基准测试** 两次无法复现，导致人们对其在人类偏好评估中的表现产生怀疑。
   - 评论指出，根据 [Molmo 发布公告](https://molmo.allenai.org/blog) 的最新发现，共识认为这些基准测试结果看起来并不可靠。
- **FP6 在量化测试中优于 BF16**：据报道，**FP6** 在多项评估和测试中表现优于 **BF16**，并讨论了这些意外结果的潜在原因。
   - 还有推测认为量化可能会引入正则化效应并影响模型路径。
- **引入 Hyperbolic Vision Transformer**：一篇新论文介绍了 **Hyperbolic Vision Transformer (HVT)**，它利用双曲几何来增强自注意力机制。
   - 该论文声称在图像分类任务中表现有所提升，特别是在 **ImageNet 数据集**上。
- **LLM 逐字记忆动态研究**：探索 LLM 中**逐字记忆 (verbatim memorization)** 的研究表明，其发生需要基于序列重复和模型状态的特定条件。
   - 该研究揭示了对数据隐私的影响，并提供了评估 Unlearning 技术的方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://molmo.allenai.org/blog">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/lexin_zhou/status/1838961179936293098">来自 Lexin Zhou (@lexin_zhou) 的推文</a>: 1/ Nature 新论文！人类对任务难度的预期与 LLM 错误之间的差异损害了可靠性。在 2022 年，Ilya Sutskever @ilyasut 预测：&#34;也许随着时间的推移，这种差异会...</li><li><a href="https://arxiv.org/abs/2409.16422">Is All Learning (Natural) Gradient Descent?</a>: 本文展示了一大类有效的学习规则——那些在给定时间窗口内提高标量性能指标的规则——可以被重写为关于...的自然梯度下降。</li><li><a href="http://arxiv.org/abs/2409.15371">Bone: Block Affine Transformation as Parameter Efficient Fine-tuning Methods for Large Language Models</a>: 随着大语言模型 (LLMs) 规模的不断扩大，其计算和内存需求也相应增加。因此，探索成本效益高且高效的微调...</li><li><a href="https://arxiv.org/abs/2210.02671">A Logic for Expressing Log-Precision Transformers</a>: 解释基于 Transformer 的语言模型推理能力的一种方法是描述它们可以在某些输入文本上解析的逻辑规则类型。最近，Chiang 等人 (2023) 表明...</li><li><a href="https://x.com/rodrimora/status/1839329810864390611">来自 Rodri Mora aka Bullerwins (@rodrimora) 的推文</a>: @AlpinDale 我使用 Aphrodite 运行了 FP 量化的 MMLU-Pro 测试。FP6-7 的结果非常出色，几乎与 BF16 持平。完整结果见此处：https://docs.google.com/spreadsheets/d/17JUJPfDgeAY...</li><li><a href="https://arxiv.org/abs/2202.08005">Should You Mask 15% in Masked Language Modeling?</a>: 掩码语言模型 (MLMs) 习惯上掩码 15% 的 token，因为人们认为更多的掩码会导致学习良好表示所需的上下文不足；这种掩码率已被广泛使用...</li><li><a href="https://arxiv.org/abs/2409.16897">HVT: A Comprehensive Vision Framework for Learning in Non-Euclidean Space</a>: 非欧几里得空间中的数据表示已被证明能有效捕捉现实世界数据集中的层次结构和复杂关系。特别是双曲空间，提供了高效的嵌入...</li><li><a href="https://arxiv.org/abs/2407.17817">Demystifying Verbatim Memorization in Large Language Models</a>: 大语言模型 (LLMs) 经常逐字记忆长序列，这通常会带来严重的法律和隐私影响。此前的许多工作通过观察性研究探讨了这种逐字记忆...
</li>
</ul>

</div>

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1288595616850772069)** (8 messages🔥): 

> - `Scaling laws dataset`
> - `Broken neural scaling laws`
> - `Chinchilla scaling laws`
> - `Google research on scaling laws` 


- **寻找 Scaling Laws 数据**：一位成员正在寻找一个包含**参数量 (# params)**、**Token 数 (# tokens)**与**损失 (loss)**之间关系数据点的数据集，这些数据点源自针对不同参数和数据集大小训练的模型，参考了 [Chinchilla scaling laws 论文](https://arxiv.org/pdf/2405.15074)。
   - 他们的目标是在不承担训练大量模型负担的情况下，探索缺失低阶项的潜在影响。
- **利用 Broken Neural Scaling 数据集**：另一位成员建议使用 **Broken Neural Scaling Laws** 数据集，该数据集被认为很全面，但缺乏架构细节。
   - 这给那些希望通过自己的模型训练工作来复现或扩充该数据集的人带来了挑战。
- **参考 Google Research GitHub**：一位成员指出，论文参考了 Google Research 在 GitHub 上分享的数据，链接见[此链接](https://github.com/google-research/google-research/blob/master/revisiting_neural_scaling_laws/README.md)。
   - 虽然该仓库提供了原始数据点，但缺乏用于复现或扩展实验的直接方法。
- **复现 Scaling Law 研究的挑战**：讨论指出，复现实际的 scaling law 研究本身可能既昂贵又复杂。
   - 这突显了研究人员在尝试验证发现或在该领域进行类似实验时面临的困难。



**提到的链接**：<a href="https://github.com/google-research/google-research/blob/master/revisiting_neural_scaling_laws/README.md">google-research/revisiting_neural_scaling_laws/README.md at master · google-research/google-research</a>：Google Research。通过在 GitHub 上创建账号来为 google-research/google-research 的开发做出贡献。

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1288640717119422478)** (1 messages): 

> - `Filler tokens in LLMs`
> - `Parallelization of tasks`
> - `Supervision in natural language context` 


- **探索 LLM 中的填充 Token (Filler Tokens)**：讨论强调，虽然在**合成任务**中取得了成功，但这并不一定能转化为 **LLMs** 中的表现，正如结论中所强调的那样。
   - *一个关键问题仍然存在*：LLMs 在多大程度上能有效地在其架构中利用**填充 Token**？
- **NLP 中的并行化限制**：参与者表示，关于并行化的观察仅限于 **Token 可并行化 (token-parallelizable)** 的算法问题，但对于复杂的语言任务可能并不成立。
   - 这引发了关于**可并行监督**与**自然语言计算**本质之间动态关系的进一步疑问。
- **Token 计算中对充足监督的需求**：对话表明，自然语言文本是否能为填充 Token 计算提供足够的**监督 (supervision)** 尚不确定。
   - *自然语言是否过于微妙？* 讨论建议，真正的语言理解可能需要**实例自适应思维链 (instance-adaptive chains of thought)**。

  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1288578173419917384)** (22 messages🔥): 

> - `Exact Match Metric Adjustment` (精确匹配指标调整)
> - `Debugging Token Generation` (调试 Token 生成)
> - `Improving Documentation for Utils` (改进 Utils 文档)
> - `Task Execution Default Behavior` (任务执行默认行为)
> - `Token Counting Script Enhancement` (Token 计数脚本增强)


- **YAML 中的精确匹配指标调整**：一位用户指出需要修改 `_aexams.yaml` 以使用 `exact_match` 聚合指标而非 `acc`，并指出了获取更多细节的路径。
   - 另一位用户确认他们实施了更改，但在生成总体的 group aexam 时遇到了问题。
- **调试 Token 生成问题**：一名成员报告了在 Token 生成过程中超出最大序列长度的错误，并深入调试到特定代码行以追踪问题。
   - 确认了覆盖 `tok_batch_encode` 可能是导致问题的原因，并促使进行审查以解决该问题。
- **改进 Utils 文档**：一名成员对指出文档缺失表示感谢，并提议创建一个 Issue 以提高 `Collator` 和 `Reorderer` 等工具的清晰度。
   - 引用了一个特定的 GitHub Issue 来跟踪文档中需要的增强内容。
- **明确任务执行默认行为**：澄清了执行任务时使用的默认 split，除非在任务 YAML 中另有规定，否则标准使用 `test` split。
   - 注意到 fewshots 在可用时取自 validation 或 training 集，否则取自 test 集。
- **讨论 Token 计数脚本增强**：一名成员寻求在生成过程中使用 `write_out.py` 计算 Token，并指出 `cost_estimate.py` 中的差异，表达了对一致复现条件的期望。
   - 一位用户分享了一个更新脚本以进行 Token 计数的 PR 草案，并提到需要进一步更新以准确处理成本估算。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2359).">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/2358">通过骨架模板代码 + 对 `Collator` 和 `Reorderer` 等工具的描述来改进 `docs/model_guide.md` · Issue #2358 · EleutherAI/lm-evaluation-harness</a>：正如标题所述，我们在 HFLM 中围绕请求排序、批处理和缓存使用的一些机制对大多数用户来说仍然是不透明的。如果能通过以下方式扩展 model_guide.md 就太好了：在 te...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

sky.moo: 我该如何在本地运行视觉 LLM？
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1288766396242595892)** (4 messages): 

> - `FA3 support` (FA3 支持)
> - `Training on H100s` (在 H100 上训练)
> - `Testing assistance` (测试协助)
> - `Grant opportunities from Modal` (来自 Modal 的资助机会)


- **H100 上的 FA3 集成正在推进中**：成员们讨论了将 **FA3** 加入待办列表以在 **H100** 上训练较小模型的可能性，并建议集成可能只需要几行代码。
   - 然而，目前持续访问 **H100** 是一个限制，测试仍然是一个挑战。
- **愿意在 H100 上进行测试**：一名成员提出帮助在 **H100** 上为小模型测试 **FA3**，表达了学习和贡献的意愿。
   - 他们请求关于如何着手集成 **FA3** 的指导，根据官方仓库，该项目目前仍处于 beta 阶段。
- **用于项目支持的 Modal 资助**：建议通过获取来自 **Modal** 的资助，作为改善项目对 **H100** 访问权限的一种手段。
   - Modal 以积极支持该领域的项目而闻名，这可能有助于克服当前的访问限制。

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1288575814090096781)** (87 条消息🔥🔥): 

> - `Perplexity AI 上下文保留问题`
> - `Llama 3.2 发布`
> - `Cloudflare 验证带来的挫败感`
> - `功能请求提交`
> - `AI 模型在歌词生成方面的能力` 


- **Perplexity AI 在上下文保留方面表现不佳**：用户对 **Perplexity AI** 无法记住先前问题的上下文表示担忧，尤其是在追问时，这种情况最近似乎有所恶化。
   - 一位用户指出：*“这个平台日常使用仍然有用，但确实变差了。”*
- **对 Llama 3.2 发布感到兴奋**：一位成员分享了 **Llama 3.2** 已在 llama.com 上发布的消息，引发了用户的兴奋，并伴随着“LFG”的号召。
   - 另一位成员提到在 Perplexity 的界面上还没看到它出现。
- **对 Cloudflare 验证的挫败感**：用户对 **Cloudflare** 反复的验证过程表示不满，声称即使是 Pro 用户也深受这一障碍的困扰。
   - 一位用户指出：*“这让我怀疑我的订阅是否值得”*，强调了这对用户体验的影响。
- **如何提交功能请求**：成员们讨论了如何为 Perplexity **提交功能请求**，并建议针对账户管理相关问题向支持团队发送电子邮件。
   - 一位用户询问其 Pro 订阅是否可以转移到朋友的账户，并获得了流程指导。
- **歌词生成的限制**：一位用户哀叹无法让 **Perplexity** 生成有趣的歌词，觉得它不再像以前那样响应创意提示。
   - 作为回应，大家讨论了各种解决方案，包括在进行歌词生成任务时切换到 Sonar 等其他模型。



**提到的链接**：<a href="https://x.com/apostraphi/status/1839303673480098285?s=61">Phi Hoang (@apostraphi) 的推文</a>：大海捞针也不在话下

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1288620910269698091)** (8 条消息🔥): 

> - `Mira Murati 离开 OpenAI`
> - `AI 对抗 reCAPTCHA`
> - `AI 的能源消耗`
> - `垃圾进，垃圾出 (Garbage In, Garbage Out)`
> - `城市中的自行车道` 


- **Mira Murati 从 OpenAI 离职**：Mira Murati 已正式 **离开 OpenAI**，引发了关于这对组织和 AI 技术影响的讨论。正如最近的一段 [YouTube 视频](https://www.youtube.com/embed/zk1GwCIEvVU)所强调的，这开启了关于行业人才迁移的新对话。
   - 还提到了 *James Cameron 的转变*以及其他有趣的话题，或许暗示了科技领域叙事演变。
- **AI 战胜 reCAPTCHA 挑战**：一项分析指出 **AI 击败了 reCAPTCHA** 系统，引起了人们对 AI 能力进步的关注。成员们讨论了对网络安全的潜在影响，并呼吁更新验证方法。
   - [详情点击此处](https://www.perplexity.ai/search/ai-beats-recaptcha-IJvLzX98RkeMdh.kpXIqXw)。
- **理解“垃圾进，垃圾出”**：围绕 **垃圾进，垃圾出 (Garbage In, Garbage Out)** 的讨论强调了 AI 训练和数据质量方面的挑战。一篇详细的 [文章](https://www.perplexity.ai/search/garbage-in-garbage-out-underst-RhMLBvuiSYycAm0BY4o9nw) 对这一概念进行了扩展，揭示了常见的陷阱。
   - 更多见解请参考 [转换后的页面](https://www.perplexity.ai/page/garbage-in-garbage-out-Z5ZG6D5ZScCBOFJ4JYGhQg)。
- **调查 AI 的能源消耗**：一份报告强调了 **AI 模型的能源消耗**，引发了关于 AI 发展可持续性的讨论。研究结果引发了关于在扩展 AI 技术时如何管理资源的询问。
   - [在此查看摘要](https://www.perplexity.ai/search/how-much-energy-does-ai-and-ll-jcyr1ba3S9eVkzCinpLk_g#0)。
- **拥有最多自行车道的城市**：**城市中自行车道最多**的话题引起了兴趣，成员们分享了对城市规划和基础设施的见解。这可能预示着可持续城市生活方式的增长趋势。
   - [探索更多关于此话题的内容](https://www.perplexity.ai/search/city-with-the-most-bike-lanes-hhNCIS6oRRCli0fdq8Z32g)。



**提到的链接**：<a href="https://www.youtube.com/embed/zk1GwCIEvVU">YouTube</a>：未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1288589988216307713)** (4 条消息): 

> - `Zapier 中的 Perplexity`
> - `System 和 User 内容`
> - `外部模型的可用性` 


- **澄清 Zapier 中的 Perplexity 结构**：一位成员正在 Zapier 内部使用 Perplexity，并寻求对其结构的澄清，特别是关于 Webhooks 的部分。
   - *消息结构是否有特定的格式要求？*
- **理解 System 和 User 内容**：另一位成员确认 **System content** 指的是给 AI 模型的指令，而 **User content** 是提供给模型的输入。
   - 这种区分对于在 Perplexity 框架内进行有效交互至关重要。
- **询问外部模型的可用性**：一位成员询问外部模型是否只能通过 Web 界面访问，而不能通过 API 访问。
   - *他们询问了 Perplexity 是否支持使用脚本操作 Web 界面。*


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1288578108609265715)** (76 条消息🔥🔥): 

> - `Meta AI 访问问题`
> - `Llama 3.2 许可担忧`
> - `AI IDE 推荐`
> - `AI 搜索工具的挑战`
> - `Reinforcement Learning 资源` 


- **Meta AI 访问仍然受限**：多位成员对访问 **Meta AI** 表示沮丧，指出尤其是美国以外的用户即使尝试使用 VPN 也会面临限制。
   - 一位用户强调 **Llama 3.2 许可证** 显然与 EU 不兼容，导致了进一步的访问问题。
- **Llama 3.2 发布亮点**：随着 **Llama 3.2** 的发布，几位成员讨论了新的 Multimodal 模型及其兼容性限制，特别是针对 EU 用户。
   - 提到 [Hugging Face](https://huggingface.co/blog/llama32) 托管了这些模型，突显了缺乏核心 AI 工具访问权限的问题。
- **游戏开发的最佳 AI IDE 选择**：对于游戏开发，成员们推荐了多种 AI IDE 选项，包括用于代码生成的 **Cursor**、**Continue**、**Amazon Q** 和 **GitHub Copilot**。
   - 一位用户分享了他们的个人配置，即结合使用 **ChatGPT** 和 SSH 操作来集成实时代码修改。
- **对 AI 搜索工具的挫败感**：用户对现有的 AI 搜索工具表示不满，特别是关于多步响应等功能，并提到了 **Perplexity** 及其在没有 Pro 订阅时的局限性。
   - 成员们讨论了创建高效搜索引擎的挑战以及索引互联网的复杂性。
- **Reinforcement Learning 资源**：有人询问学习 **Reinforcement Learning** 的资源，建议利用 **ChatGPT 4o** 获取深入信息和参考资料。
   - 成员们还指出网上有大量的研究论文，同时建议通过 YouTube 学习 RL 的实际应用。



**提及的链接**：<a href="https://huggingface.co/blog/llama32">Llama can now see and run on your device - welcome Llama 3.2</a>：未找到描述内容。

  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1288636990140584027)** (20 条消息🔥): 

> - `Advanced Voice Mode 的局限性`
> - `o1 中的文件上传问题`
> - `GPT 模型的性能对比`
> - `对违反政策的担忧`
> - `对未来版本的期待功能` 


- **Advanced Voice Mode 无法搜索互联网**：用户对 **Advanced Voice Mode** 目前缺乏互联网搜索功能表示沮丧，这导致从语音切换回文本输入时需要开启新的对话。
   - 尽管有此限制，许多人对该功能以及即将推出的 ChatGPT-5 等模型的未来持乐观态度。
- **o1 缺失文件上传功能**：成员们对无法向 **o1** 上传文件感到遗憾，这迫使他们不得不换回 **GPT4o** 进行处理，影响了工作效率。
   - 一位成员回想起 *Sam Altman* 关于 GPT-4 的名言，对必须在不同模型间倒腾数据表示失望。
- **o1 在推理方面优于 GPT4o**：讨论强调 **o1** 在遵循长指令方面表现出色，并且事实证明其能力显著强于在基础任务上表现吃力的 **GPT4o**。
   - 一位用户指出，在写作中可能需要使用 **Python tools** 来追踪字符和估算页数。
- **小说写作的合规困扰**：一位用户担心使用 ChatGPT 编辑暴力题材的虚构小说可能会违反使用政策，特别是在收到警告之后。
   - 另一位成员建议，持续的警告可能会导致账号受到审查，并敦促遵守服务条款。
- **对未来功能发布的期待**：成员们讨论了对 **file uploads**、**o1 中的 memory** 以及与 GPTs 集成等未来功能的期望，并强调在发布过程中需要保持耐心。
   - 正如一位成员所说，“OpenAI 的一切发布都是缓慢且谨慎的”，这反映了大家对未来改进的共同乐观态度。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 条消息): 

blckreaper: 如果你使用 o1 mini 就不会了，它会告诉你你很差劲，让你做得更好
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 条消息): 

blckreaper: 如果你使用 o1 mini 就不会了，它会告诉你你很差劲，让你做得更好
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1288584856971776095)** (20 messages🔥): 

> - `3.2 许可证对比`
> - `OpenAI 领导层变动`
> - `Molmo 指向功能引发关注`
> - `Barret Zoph 离职`
> - `Opus 项目更新` 


- **Llama 3.2 许可证的显著变化**：一名成员指出，**3.2 许可证**与 **3.1** 相比有显著差异，并敦促大家仔细查看 [diff 对比](https://www.diffchecker.com/O4ijl7QY/)。
   - 这引发了使用该许可证的开发者和组织对其潜在影响的疑问。
- **OpenAI 领导层动荡**：最近的讨论表明，包括 **Chief Technology Officer** (首席技术官) 在内的几位关键人物正在辞职，引发了对公司发展方向的猜测。
   - 一位成员注意到这些离职的时间点非常不同寻常，暗示团队内部可能存在紧张局势。
- **Molmo 指向功能引发热议**：一名成员表示，**Molmo 指向功能**（pointing feature）对于产品构建的影响可能比更高的 **AIME** 分数更大，并称其“通过了 vibe check”，这让大家感到非常兴奋。
   - 这反映了一些开发者的观点，即新功能带来的即时价值有时会超过单纯的性能指标。
- **Barret Zoph 宣布离职**：在一篇感人至深的帖子中，**Barret Zoph** 宣布他将离开 OpenAI，并回顾了他在 **post-training** 团队中的积极经历和贡献。
   - 他表达了对同事的感激之情以及对 OpenAI 未来的乐观态度，预示着个人职业生涯的新阶段。
- **Opus 发布的不确定性**：几位成员对 **Opus** 项目表示怀疑，其中一位成员开玩笑说，如果把房子押在上面可能会输得很惨。
   - 成员们对 Opus 不确定的时间表感到焦虑，以及这可能如何影响他们的预期。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/BenjaminDEKR/status/1839112669808275816">来自 Benjamin De Kraker 🏴‍☠️ (@BenjaminDEKR) 的推文</a>：等等，所以……Sam 今天早些时候最初的“便签”只提到了 Mira 的离开。然后他删掉了它并发布了新的（现在的版本），把他们都归类在一起并说“这很合理……”</li><li><a href="https://x.com/bobmcgrewai/status/1839099787423134051?t=3SqLlhVJgATHMsRWY7neTA&s=19">来自 Bob McGrew (@bobmcgrewai) 的推文</a>：我刚刚与 OpenAI 分享了这些：</li><li><a href="https://x.com/miramurati/status/1839025700009030027?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Mira Murati (@miramurati) 的推文</a>：我今天与 OpenAI 团队分享了以下便签。</li><li><a href="https://x.com/andersonbcdefg/status/1839030313659564424">来自 Ben (e/treats) (@andersonbcdefg) 的推文</a>：我已经很久没有像对 Molmo 指向功能这样对新的 AI 模型能力感到兴奋了。对于想要构建产品（而不是神）的人来说，我认为这可能更具影响力……</li><li><a href="https://x.com/bobmcgrewai/status/1839099787423134051?t=3SqLlhVJgATHMsRWY7">来自 Bob McGrew (@bobmcgrewai) 的推文</a>：我刚刚与 OpenAI 分享了这些：</li><li><a href="https://x.com/barret_zoph/status/1839095143397515452?s=46">来自 Barret Zoph (@barret_zoph) 的推文</a>：我向 OpenAI 发布了这份便签。大家好，我决定离开 OpenAI。这是一个非常艰难的决定，因为我在 OpenAI 度过了一段不可思议的时光。我在 ChatGPT 之前就加入了……</li><li><a href="https://www.diffchecker.com/O4ijl7QY/">Llama 3.2 vs 3.1 - Diffchecker</a>：Llama 3.2 vs 3.1 - LLAMA 3.1 社区许可证协议 Llama 3.1 版本发布日期：2024 年 7 月 23 日 “协议”是指……
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1288586685629857943)** (44 条消息🔥): 

> - `OpenAI leadership changes` (OpenAI 领导层变动)
> - `Profit interest structures` (利润权益结构)
> - `California oversight on non-profits` (加州对非营利组织的监管)
> - `Impact of Anthropic on OpenAI` (Anthropic 对 OpenAI 的影响)
> - `Tax implications for employees` (对员工的税务影响) 


- **OpenAI 领导层大换血引发担忧**：讨论集中在 OpenAI 最近的领导层变动上，多位成员评论称 *除了 Sam 之外，所有 OG OpenAI 成员都离开了*，并质疑这是否是一个负面信号。
   - 有人指出，*这事儿闻起来太臭了。非常可疑，* 反映了对 OpenAI 发展方向的普遍担忧。
- **利润权益单位 (PIUs) 引发关注**：成员们讨论了在非营利组织中提供 *Profit Interests Units (PIUs)* 是否合适，一些人认为这像是利用漏洞，可能会引发监管行动。
   - 有人评论说，以非营利组织身份筹集资金，然后在获得股权的同时转为营利性实体，这似乎令人不安，突显了业界对税务影响的担忧。
- **加州可能调查 OpenAI 的结构**：围绕加州在监管非营利组织方面的角色展开了讨论，一名成员建议最近的变化可能会使 OpenAI 面临该州总检察长 (Attorney General) 的审查。
   - 这种潜在的调查在更广泛的 AI 立法和治理背景下可能具有重大意义。
- **对 Anthropic 日益增长的影响力的担忧**：成员们推测了如果 *Logan* 离开 OpenAI 加入 Anthropic 会产生什么影响，有人调侃说这会导致 *世界真正崩塌*。
   - 另一位成员表示，Anthropic 被认为是一个 *远没那么不严肃的公司*，这引发了对其在 AI 领域竞争地位的质疑。
- **税务问题使员工薪酬复杂化**：成员们讨论了利润权益的税务影响，指出获得利润权益的员工可能面临与传统 RSUs 不同的待遇，后者在归属 (vesting) 时立即征税。
   - 有人评论道，*在非营利组织中给予任何形式的“股权”补偿确实看起来很奇怪，* 突显了员工补偿结构中持续存在的混乱。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://vxtwitter.com/Yuchenj_UW/status/1839030011376054454">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://www.levels.fyi/blog/openai-compensation.html">OpenAI PPUs：OpenAI 独特的股权补偿运作方式</a>：深入了解当今最热门、最神秘的 AI 公司之一。</li><li><a href="https://riveron.com/posts/accounting-for-pius/">利润权益单位 (PIUs) 的会计处理 - Riveron</a>：当今许多公司寻求通过利润权益单位 (PIUs) 留住核心人才。以下是会计团队应该了解的内容。</li><li><a href="https://x.com/miramurati/status/1839025700009030027?s=46">来自 Mira Murati (@miramurati) 的推文</a>：我今天与 OpenAI 团队分享了以下简报。</li><li><a href="https://news.bloomberglaw.com/us-law-week/openais-example-is-lesson-for-nonprofit-tandem-entity-formation">OpenAI 的案例是非营利组织双重实体形成的教训</a>：加州大学伯克利分校的 Jesse Finfrock 解释了公司利用 AI 产生利润所面临的压力，以及实体形成时必要的考虑因素。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1288584168036630569)** (21 条消息🔥): 

> - `LLaMA Stack`
> - `FTC AI Crackdown`
> - `NeurIPS Submission Rejections`
> - `Rewardbench Benchmark`
> - `C++ Knowledge in Academia` 


- **LLaMA Stack 困惑**: 关于 **LLaMA Stack** 集成工具是否重要引发了讨论，一名成员在看到博文后表示并不感兴趣。
   - 另一位成员评论说这似乎并不重要，认为它只是工具集成。
- **FTC 宣布 AI 监管打击**: 一条推文透露 **FTC** 宣布了针对 AI 的新监管打击，引发了对其有效性的怀疑，并断言该领域存在许多骗局。
   - 成员们对什么是“真正的 AI”这一模糊表述进行了思考，反映出对监管定义的困惑。
- **NeurIPS 拒稿引发关注**: 一位成员幽默地指出了 **Rewardbench** 被 **NeurIPS D&B** 拒绝的情况，尽管结果令人遗憾，但他形容这非常滑稽。
   - 据指出，拒绝评分中包含了一次具有伤害性的个人攻击，且该项目是该领域的第一个 Benchmark，这一点似乎被忽视了。
- **C++ 知识批判**: 成员们对一项 NeurIPS 投稿被拒做出了反应，该投稿因使用 **C++** 编写而受到批评，质疑在学术界掌握这门语言的相关性。
   - 一位成员评论了 C++ 在其课程作业中的普遍性，对这种狭隘的批评表示沮丧。
- **社交媒体回应**: 一位成员认为，在 Twitter 上抱怨现状对解决问题几乎没有帮助，且项目本身仍然是成功的。
   - 他们反思了如果学生依赖于这个被拒绝的 Benchmark，可能会产生不同的后果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/cdolan92/status/1839024340689371356">来自 Charlie Dolan (@cdolan92) 的推文</a>: FTC 宣布了与 AI 相关的打击行动。锐评：好！有很多骗局。进一步阅读后：什么鬼！？例如：这到底是什么意思？“...我们的技术人员可以弄清楚 [如果你的...</li><li><a href="https://x.com/eugenevinitsky/status/1839275401903780300?s=46">来自 Eugene Vinitsky 🍒 (@EugeneVinitsky) 的推文</a>: 得分相当高的 NeurIPS 投稿因“模拟器是用 C++ 编写的，而人们不懂 C++”而被拒绝，这真是一个滑稽的新低点。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1288580074018312203)** (6 条消息): 

> - `Molmo model skepticism`
> - `Impact of social media`
> - `LLaMA vs Molmo performance` 


- **对 Molmo 赞誉的质疑**: 成员们对大量声称 **Molmo** 优于 **LLaMA 3.2** 的帖子表示怀疑，质疑这些说法的真实性。
   - 然而，一位成员建议亲自测试该模型，并表示*没有证据*表明存在第三方付费的好评，强调验证结果非常容易。
- **关于社交媒体粉丝价值的辩论**: 讨论引发了关于积累 **推文和粉丝** 是否真的有益的好奇，一位成员幽默地质疑其价值。
   - 另一位成员建议这可能对个人的 **网络存在感 (net presence)** 更有利，为对话增添了轻松的基调。
- **Molmo 发布时间的澄清**: 一位成员澄清说 **Molmo** 与 **LLaMA 3.2** 在同一天发布，但早了几个小时，纠正了最初对其发布时间线的困惑。
   - 他们强调了自己使用该模型的个人经验，并指出其性能令人印象深刻。



**提到的链接**: <a href="https://old.reddit.com/r/LocalLLaMA/comments/1fptwvm/molmo_a_new_model_that_outperforms_llama_32/lp1utnq/">Molmo - 一款性能优于 Llama 3.2 的新模型，已在欧盟可用</a>: 由 u/franklbt 发布在 r/LocalLLaMA • 104 点赞和 51 条评论

  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1288579946385772658)** (6 messages): 

> - `Meeting duration`
> - `One-on-ones`
> - `Project leadership` 


- **3.5 小时的会议似乎不寻常**：一位成员对当天安排的 **3.5 小时** 会议表示惊讶，称其非常 **crazy**。
   - 该成员指出，尽管持续时间很长，但最好还是把这些会议安排在当天早些时候。
- **轻松的会议结构**：另一位成员提到安排的会议很 **chill**，指的是一种更放松的方式。
   - 这包括一对一会议以及与其领导的项目相关的讨论。
- **总体会议较少**：一位成员指出，他们通常会议不多，暗示了一种更专注的工作风格。
   - 他们建议，如果一定要开会，倾向于将会议堆叠在一起进行。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1288716719635959919)** (2 messages): 

> - `Vision Llama Release`
> - `Gemini Tokenization Changes`
> - `Gemini Pricing Adjustments`
> - `OpenRouter Endpoints` 


- **Vision Llama 登陆 OpenRouter 并提供免费 Endpoint**：首个视觉 **Llama** 现已在 **OpenRouter** 上线，并提供 [免费 Endpoint](https://x.com/OpenRouterAI/status/1839157099747479553)。总共推出了 **五个新 Endpoint**，由多个供应商提供支持。
   - 鼓励用户体验最新功能，并配有庆祝图标 🎁🦙。
- **Gemini Tokenization 简化成本**：OpenRouter 将转为对 Gemini 模型计算 **tokens** 而非字符数，使表观 token 数量减少约 **4 倍**。此举旨在为开发者实现标准化并降低成本。
   - 由于将 token 对应到每 token 计价模式，这些变化也将导致当前价格 **翻倍**，该模式计划在 **10 月 1 日** 后进一步调整。
- **即将到来的 Gemini 价格调整**：作为即将到来的变更的一部分，**Gemini** 价格将进行调整，以匹配 AI Studio 当前较低层级的每 token 价格。这一定价策略旨在提升开发者体验和标准化。
   - 预计 **10 月 1 日** 的降价将为用户带来进一步的优惠，从而在未来实现整体改进的成本结构。



**提及的链接**：<a href="https://x.com/OpenRouterAI/status/1839157099747479553">来自 OpenRouter (@OpenRouterAI) 的推文</a>：今日发布的 Llama 3.2 的五个新 Endpoint 现已上线，包括一个免费的 vision Llama！🎁🦙

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1288579887602470912)** (87 messages🔥🔥): 

> - `OpenRouter 更新`
> - `Llama 模型限制`
> - `Chat Completion 模型`
> - `Gemini 性能变化`
> - `数学测验重写模型` 


- **OpenRouter 额度与发票问题**：用户报告了在 OpenRouter 上进行额度交易和查看额度时遇到的困难，称付款后交易可能需要一段时间才会显示，正如一位最终收到额度的用户所描述的那样。
   - 另一位用户提到，后端延迟或提供商问题可能导致交易历史显示中断。
- **针对欧盟用户的 Llama 3.2 限制**：Meta 关于在欧盟使用其 Vision 模型的政策引发了对该地区用户访问权限和合法性的担忧，特别是关于推理服务的提供。
   - 成员们对提供商所在地和许可证的影响表示困惑，认为遵守 Meta 的规则可能会有问题。
- **Chat Completion 模型限制**：用户讨论了 OpenRouter 对 Completion 模型支持的局限性，指出此类功能可能需要特定的模板或条件才能正确运行，特别是对于 Codestral Mamba 模型。
   - 对话强调了普遍缺乏 Completion 支持的问题，并对 OpenRouter 内部的模型能力提出了疑问。
- **Gemini Token 计数异常**：有报告称 Gemini 的输入/输出 Token 计数出现意外跳变，用户推测可能存在影响结果的潜在更改或错误。
   - 一位参与者建议 Token 计数包含了隐藏的 Reasoning Tokens，导致用户对实际 Token 使用情况感到困惑。
- **vLLM 中的 VLM 支持**：讨论集中在 vLLM 对 Vision Language Models (VLMs) 不断发展的支持上，反馈表明虽然之前存在问题，但最近的更新已改进了功能。
   - 用户确认 vLLM 的支持取决于与 Hugging Face 的 Transformers 的兼容性，这表明新架构对外部框架存在依赖。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.llama.com/llama3_2/use-policy/">Llama 3.2 可接受使用政策</a>: Llama 3.2 Acceptable Use Policy</li><li><a href="https://openrouter.ai/credits">Credits | OpenRouter</a>: 管理您的额度和付款历史
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1288923984175431762)** (2 messages): 

> - `Bring Your Own Key Beta 测试` 


- **申请参与 BYOK Beta 测试**：一位成员询问是否可以被纳入 **Bring Your Own Key (BYOK)** Beta 测试。
   - 他们提出可以通过私信 (DM) 提供其 **电子邮件地址** 以推进该流程。
- **愿意分享联系方式**：该成员表示愿意分享个人联系方式以协助参与 Beta 测试。
   - 他们特别提到，如果需要，很乐意 **DM** 他们的电子邮件地址。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1288581042885623914)** (65 条消息🔥🔥): 

> - `License Compliance Issues` (许可证合规问题)
> - `OpenAI's CTO Resignation` (OpenAI CTO 辞职)
> - `Mistral Discussions` (Mistral 讨论)
> - `Open Source Advancements` (开源进展)
> - `Continued Pretraining Example` (持续预训练示例)


- **许可证合规引发挫败感**：成员们讨论了许可证合规性，强调由于与监管条例存在分歧，**EU 访问被屏蔽**，导致对访问限制感到沮丧。
   - 一位成员幽默地评论说 **Mistral 现在成了一个梗 (meme)**，指出了这种情况的荒谬性。
- **OpenAI CTO 辞职引发猜测**：提到了 OpenAI CTO 的辞职，一位成员开玩笑说这引发了对公司现状的猜测。
   - 成员们对 OpenAI 的发展方向表示担忧，并建议其内部问题可以拍成一部有趣的 Netflix 迷你剧。
- **新 Molmo 模型能力惊人**：最近的 **Molmo 模型** 因其在**图像中指向位置**的能力而受到称赞，展示了开源开发的进步。
   - 此外，还讨论了**语音标注图像训练**方法，标志着向集成多模态数据集迈出了重要一步。
- **对 AI 军事应用的担忧**：成员们猜测了 AI 模型的潜在军事应用，对识别军事目标或从无人机镜头估算高度等任务表示担忧。
   - 一位成员分享说，他们的测试显示在识别低质量航拍图像中的关键物体方面表现出**卓越的性能**，引发了伦理问题。
- **需要持续预训练示例**：一位成员询问了特定模型的 **continued pretraining** 示例，包括配置和数据集处理。
   - 这反映了对各种 AI 技术实际实现的日益增长的兴趣，并表明了优化训练过程的协作努力。



**提到的链接**：<a href="https://x.com/miramurati/status/1839025700009030027?t=pVYyCN8C7RnV0UruM9H2Lg&s=19">来自 Mira Murati (@miramurati) 的推文</a>：我今天与 OpenAI 团队分享了以下便签。

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1288930053371072565)** (4 条消息): 

> - `LoRA+ Issue` (LoRA+ 问题)
> - `Multimodal/VLM Assistance` (多模态/VLM 协助)


- **Commit 破坏了 LoRA+ 功能**：一位成员报告说 [这个 commit](https://github.com/axolotl-ai-cloud/axolotl/commit/b98d7d7098f5d64a07c5a96855c4e08dca7afd91) 在与官方 runpod 模板一起使用时似乎破坏了 **LoRA+**，而不是替换它。
   - 另一位成员询问该问题是否与 YAML 配置中新添加的 `loraplus_lr_ratio` 有关。
- **寻求多模态/VLM 方面的帮助**：一位成员表示愿意在 **multimodal** 和 **VLM** 项目上提供协助，并指出由于预处理优化，他们目前面临挑战。
   - 他们提到多模态操作的数据预处理并未产生理想的结果。
- **LoRA+ 可能存在的配置问题**：另一位成员推测问题是否与最近添加的参数 `loraplus_lr_ratio` 或 `loraplus_lr_embedding` 有关。
   - 他们分享了在更改几小时后启动新 pod 时功能失效的经历。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1288669230237224970)** (3 条消息): 

> - `Llama 3.2 Inference` (Llama 3.2 推理)
> - `GPU Requirements` (GPU 需求)
> - `Runpod H100 GPUs` (Runpod H100 GPU)
> - `Quantization Effects` (量化影响)


- **计划进行 Llama 3.2 推理**：有人询问推理 **900 亿参数** 的 **Llama 3.2** 需要多少个 **H100 GPU**，以避免显存溢出 (out-of-memory) 错误。
   - 该用户旨在获取 **Runpod GPU**，但希望确保它们能处理该模型，而无需因 OOM 问题而删除它们。
- **探索显存 (VRAM) 需求**：有人询问高效推理 **Llama 3.2** 所需的 **GB VRAM** 数量。
   - 这反映了用户优化 GPU 分配以运行大型模型的意图。
- **提出了量化考虑**：一位成员通过询问 *What quant?*（什么量化？）来了解该模型正在考虑的量化选项。
   - 这表明量化可能会影响 GPU 需求和整体模型性能。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1288658334622154842)** (5 条消息): 

> - `Tokenizer Padding Token`
> - `Adding Special Tokens`
> - `Model Resizing for Tokenization` 


- **Tokenizer 缺少 padding token**：一位用户提出了在预训练期间 Tokenizer 缺少 padding token 的问题，这可能会干扰变长输入序列的处理。
   - *提供的选项包括将 pad token 设置为 EOS token，或者使用 `tokenizer.add_special_tokens({'pad_token': '[PAD]'})` 添加一个新的 pad token*。
- **将 pad token 设置为 EOS**：要使用现有 token，可以通过 `tokenizer.pad_token = tokenizer.eos_token` 将 padding token 指定为句子结束 token。
   - 该解决方案允许 Tokenizer 将指定的 token 同时视为 padding 和 end-of-sequence，从而简化输入流程。
- **添加 token 后调整模型 embeddings 大小**：当添加新的 padding token 时，必须使用 `model.resize_token_embeddings(len(tokenizer))` 更新模型的 embeddings。
   - 这确保了模型在运算中能正确处理新添加的 token，并保持兼容的输入处理。
- **带有 padding 的 tokenization 示例**：分享了一个完整的代码片段，演示了在 Tokenizer 设置中处理 padding token 的两种方法。
   - 该示例包含了对输入进行 tokenizing 的过程，同时确保应用适当的 padding 以保持统一的输入长度。



**提及链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=ff7885b4-7628-45b9-8547-0f162351c8d1)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1288576581899653160)** (73 条消息🔥🔥): 

> - `Mira Murati 离职`
> - `Meta 的 Orion AR 眼镜`
> - `Google 的 AlphaChip`
> - `AI 产品创作平台 Arcade`
> - `GitHub Copilot 浏览器集成` 


- **Mira Murati 从 OpenAI 离职**：[Mira Murati](https://x.com/miramurati/status/1839025700009030027?s=46) 宣布从 OpenAI 离职，引发了 Sam Altman 的感谢以及对其过去 6.5 年贡献的回顾。
   - 这一变动似乎与其他关键人物的离职同时发生，引发了外界对领导层动态转变的猜测。
- **Meta 发布 Orion AR 眼镜**：Meta 推出了 [Orion](https://about.meta.com/realitylabs/orion)，声称这是其迄今为止最先进的 AR 眼镜，旨在将数字体验无缝集成到物理世界中。
   - 初步反应强调了其美学吸引力，尽管由于制造复杂性，公司选择暂不销售该产品。
- **Google 发布 AlphaChip 及其权重**：Google 宣布了 [AlphaChip](https://x.com/googledeepmind/status/1839306984480231852?s=46)，彻底改变了微芯片设计，并公开了其模型权重供公众使用。
   - 该芯片因其在为 AI 模型设计 TPU 方面的作用而备受推崇，是 Google 芯片生产能力的重大进步。
- **Arcade 为 AI 产品创作平台筹集 1700 万美元**：Arcade 宣布筹集 1700 万美元用于开发首个 AI 产品创作平台，将其定位为将创意想法转化为实物产品的工具。
   - 其愿景核心是通过 AI 使产品开发民主化，强调了该平台对创新的潜在影响。
- **GitHub Copilot 现已支持浏览器**：GitHub Copilot 已将其功能扩展到浏览器，允许开发者从更灵活的环境中访问其功能。
   - 这一变化使其与 Sourcegraph Cody Chat 等竞争对手保持一致，并强调了详细文档对于最大化发挥其能力的重要性。


<div class="linksMentioned">

<strong>提及链接</strong>：

<ul>
<li>

<li><a href="https://x.com/miramurati/status/1839025700009030027?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Mira Murati (@miramurati) 的推文</a>：我今天向 OpenAI 团队分享了以下便条。</li><li><a href="https://x.com/barret_zoph/status/1839095143397515452?s=46">来自 Barret Zoph (@barret_zoph) 的推文</a>：我向 OpenAI 发布了这条便条。大家好，我决定离开 OpenAI。这是一个非常艰难的决定，因为我在 OpenAI 度过了一段不可思议的时光。我在 ChatGPT 推出前夕加入...</li><li><a href="https://x.com/RihardJarc/status/1839014234266755473">来自 Rihard Jarc (@RihardJarc) 的推文</a>：$META 的 iPhone 时刻</li><li><a href="https://x.com/Teknium1/status/1839040366512844917">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：所以你没能通过 OpenAI 的高级语音模式获得 HER 的体验——好吧——我在 http://Play.AI 的朋友们有真正的干货，HERmes。在这里尝试：https://play.ai/agent/HERMES-m3i3jU81_52ruL6_0tw2R</li><li><a href="https://x.com/googledeepmind/status/1839306984480231852?s=46">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：我们的芯片设计 AI 方法 AlphaChip 已经改变了我们设计微芯片的方式。⚡ 从帮助设计用于构建 AI 模型的最先进 TPU，到数据中心的 CPU——其广泛的影响...</li><li><a href="https://x.com/mnaficy/status/1839342011788439580?s=46&t=b7l37rB6wtbyAh6ah1NpZQ">来自 Mariam Naficy (@mnaficy) 的推文</a>：1/ 宣布获得总计 1700 万美元的融资，用于构建 Arcade，全球首个 AI 产品创作平台。引用 Arcade (@arcade_ai)：推出 Arcade 1.0，首个 AI 产品创作平台...</li><li><a href="https://x.com/willdepue/status/1839098732534722570">来自 will depue (@willdepue) 的推文</a>：第二波辞职潮席卷了 OpenAI 总部</li><li><a href="https://x.com/andrewcurran_/status/1839037623756796196?s=46">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：哇。</li><li><a href="https://x.com/bobmcgrewai/status/1839099787423134051?s=46">来自 Bob McGrew (@bobmcgrewai) 的推文</a>：我刚刚向 OpenAI 分享了这些内容：</li><li><a href="https://www.theverge.com/24253908/meta-orion-ar-glasses-demo-mark-zuckerberg-interview">Orion 上手体验，Meta 的首款 AR 眼镜</a>：Meta 的 AR 眼镜令人印象深刻地展示了马克·扎克伯格认为将取代智能手机的产品。</li><li><a href="https://x.com/multimodalart/status/1839060917960474671">来自 apolinario 🌐 (@multimodalart) 的推文</a>：在随机图像上测试 Diffusers Image Fill 演示功能</li><li><a href="https://gist.github.com/csellis/4b08be1757b545d1497853df0b0d730e">CSAT.md</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://x.com/gdb/status/1839391073296408577">来自 Greg Brockman (@gdb) 的推文</a>：我深深感激 Barret、Bob 和 Mira 为 OpenAI 带来的一切。我们共事多年，我们都是帮助 OpenAI 成就今日地位的团队成员。他们...</li><li><a href="https://x.com/benthompson/status/1839065543766429926?s=46">来自 Ben Thompson (@benthompson) 的推文</a>：它们是真实的，而且非常壮观。</li><li><a href="https://x.com/Yuchenj_UW/status/1839030011376054454">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：开始时 vs 现状。引用 Sam Altman (@sama)：我以此回复。Mira，感谢你所做的一切。很难言表 Mira 对 OpenAI、我们的使命以及我们所有人的意义...</li><li><a href="https://x.com/sama/status/1839093415226524114">来自 Sam Altman (@sama) 的推文</a>：我刚刚向 OpenAI 发布了这条便条：大家好——在过去的 6.5 年里，Mira 对 OpenAI 的进步和增长起到了至关重要的作用；她是我们从一个默默无闻的...</li><li><a href="https://x.com/boztank/status/1838999636402647453">来自 Boz (@boztank) 的推文</a>：我们刚刚揭晓了 Orion，这是我们研发近十年的全功能 AR 眼镜原型。当我们开始这段旅程时，我们的团队预测成功的概率最高只有 10%...</li><li><a href="https://x.com/JacquesThibs/status/1839030342788809062">来自 Jacques (@JacquesThibs) 的推文</a>：我的天。@gwern 又说对了：引用 Mira Murati (@miramurati)：我今天向 OpenAI 团队分享了以下便条。</li><li><a href="https://x.com/ArtificialAnlys/status/1838993593056247965">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>：Gemini 1.5 的智能升级和降价显著加强了 Google 在 AI 市场的质量与价格定位。我们对 @Google 的 Gemini 模型进行的独立评估...</li><li><a href="https://github.com/copilot">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://x.com/smokeawayyy/status/1839036947400143073?s=46">来自 Smoke-away (@SmokeAwayyy) 的推文</a>：Ilya 宣布...</li>

在 GPT-4o 发布后的第二天宣布离职。Mira 在 ChatGPT Voice 发布后的第二天宣布离职。</li><li><a href="https://youtu.be/oX7OduG1YmI">马克·扎克伯格试图构建的未来</a>：与马克·扎克伯格的深度对话...我在 Connect 大会前采访了 Meta CEO 马克·扎克伯格。没有多少人比他对我们的未来拥有更大的影响力...</li><li><a href="https://stratechery.com/2024/an-interview-with-meta-cto-andrew-bosworth-about-orion-and-reality-labs/">专访 Meta CTO Andrew Bosworth：关于 Orion 和 Reality Labs</a>：专访 Meta CTO Andrew Bosworth，探讨 Orion 和 Reality Labs</li><li><a href="https://about.fb.com/news/2024/09/introducing-orion-our-first-true-augmented-reality-glasses/">介绍 Orion，我们的首款真正增强现实眼镜 | Meta</a>：今天我们揭晓了 Orion，我们相信这是有史以来最先进的一款 AR 眼镜。</li><li><a href="https://youtu.be/6pxmdmlJCG0?feature=shared">我采访了 ChatGPT 的创始人</a>：你可能知道他是 OpenAI 的 CEO —— 但你知道 Sam Altman 也是一位热衷写作的人吗？作为当今最成功的企业家之一，Sam 倡导...</li><li><a href="https://x.com/mattturck/status/1839054212040171638">来自 Matt Turck (@mattturck) 的推文</a>：Destiny’s Child —> Beyoncé；One Direction —> Harry Styles；OpenAI —> Sama</li><li><a href="https://github.com/meta-llama/llama-stack/issues/6">RFC-0001 - Llama Stack · Issue #6 · meta-llama/llama-stack</a>：作为 Llama 3.1 发布的一部分，Meta 正在发布一份关于 ‘Llama Stack’ 的 RFC，这是一套为在 Llama 基础模型之上构建应用的 ML 开发者提供的全面接口 / API。我们正在寻找...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: 由 <@656968717883670570> 领导的 <@&1284244976024424630> 新聚会！ https://lu.ma/i8ulstlw
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1288615241835479101)** (2 条消息): 

> - `工程职位招聘`
> - `NVIDIA 竞赛`
> - `LLM 应用`
> - `奖品与奖励`
> - `开发者资源` 


- **LlamaIndex 正在旧金山招聘工程师**：LlamaIndex 正在寻找充满活力的 **ML/AI** 爱好者加入他们不断壮大的团队，扩展从全栈到各种职位的工程角色。感兴趣的候选人可以在 [Twitter](https://twitter.com/llama_index/status/1839055997291344050) 上找到更多细节。
- **与 NVIDIA 竞争现金和硬件奖品**：有一项与 **NVIDIA** 合作的竞赛，提供超过 **$10,000** 的奖品，包括现金和 **NVIDIA® GeForce RTX™ 4080 SUPER GPU** 等硬件。开发者可以在 **11 月 10 日** 之前参加比赛，创建创新的 LLM 应用，更多信息请点击 [此处](https://developer.nvidia.com/llamaindex-developer-contest/join)。
   - 鼓励参与者在各个领域构建 **RAG** 应用，可查阅 [条款和条件](https://developer.download.nvidia.com/licenses/nvidia-and-lla)。



**提到的链接**：<a href="https://t.co/rtMpetSyu1">NVIDIA 与 LlamaIndex 开发者大赛</a>：有机会赢取现金奖励、GeForce RTX GPU 等。

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1288594955769741325)** (47 条消息🔥): 

> - `ReAct Agent 消息`
> - `VectorStoreIndex 困惑`
> - `LlamaTrace 项目问题`
> - `评分评估`
> - `KnowledgeGraph RAG vs QueryFusion` 


- **向 ReAct Agent 传递消息**：一位成员询问如何将系统和用户消息传递给 ReAct Agent，强调了特定类和格式化工具的使用。
   - 回复指出 `ReActChatFormatter` 类对于将聊天历史格式化为兼容的输入格式至关重要。
- **理解 VectorStoreIndex**：关于 `VectorStoreIndex` 存在一些困惑，用户澄清了索引与底层向量存储（vector stores）之间的关系。
   - 小组讨论了如何在无需初始化新向量存储的情况下访问索引的 `vector_store` 属性。
- **LlamaTrace 项目问题**：一位用户提到在登录其 LlamaTrace 项目后遇到错误，该错误在一段时间后自行解决。
   - 他们建议如果其他人遇到同样的问题，可以尝试清除 Cookie。
- **评分评估差异**：一位用户发现自己的正确性评分（correctness score）高于答案相关性（answer relevancy），引发了关于评分可比性的讨论。
   - 建议认为这些分数不可直接比较，因为它们取决于 LLM 评估的不同方面。
- **使用 KnowledgeGraph RAG vs QueryFusion**：一位成员询问他们是否正确使用了 `QueryFusionRetriever` 而非 `KnowledgeGraphRAGRetriever` 进行知识索引。
   - 讨论集中在潜在的改进方案，以及 RAG 检索器是否能更好地满足他们查询多个索引的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://microsoft.github.io/autogen/docs/tutorial/code-executors/">Code Executors | AutoGen</a>: Open In Colab</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/output_parsers/pydantic/#llama_index.core.output_parsers.PydanticOutputParser>)">Pydantic - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/output_parsing/llm_program/#initialize-with-pydantic-output-parser>)">LLM Pydantic Program - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/nvidia_agent/#view-prompts>)">Function Calling NVIDIA Agent - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/agent/react/#llama_index.core.agent.react.ReActChatFormatter>)">React - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/nvidia_agent/#customizing-the-prompt>)">Function Calling NVIDIA Agent - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1288947311778599004)** (3 条消息): 

> - `Langtrace DSPy 集成`
> - `DSPy 项目设置`
> - `使用 Langtrace 进行实验追踪`
> - `DSPy 分类和摘要的代码示例` 


- **Langtrace 现在支持 DSPy 实验**：Langtrace 引入了对运行 DSPy 实验的支持，具有**自动捕获 Trace、检查点（checkpoints）、成本**以及**评估分数可视化**的功能。
   - 此功能使用户能够为 Pipeline 的每个模块创建单独的项目，从而增强实验和优化。
- **征集实验代码示例**：**Chiggly007** 表示有兴趣获取屏幕截图中分享的实验的 Notebook 或代码示例。
   - 这突显了社区希望通过使用 DSPy 进行分类和摘要任务的实际案例进行学习的愿望。
- **分享 DSPy 设置说明**：**Kaykay0403** 提供了将 DSPy 与 Langtrace 集成的分步设置说明，并指向了相关文档。
   - 鼓励用户遵循 [DSPy 安装指南](https://github.com/stanfordnlp/dspy?tab=readme-ov-file#1-installation)和 Langtrace 设置来简化集成。
- **简化的 DSPy 集成流程**：将 Langtrace 与 DSPy 集成是一个简单的过程，只需**两行代码**即可设置 SDK 并运行实验。
   - Kaykay0403 表示愿意协助用户解决设置过程中的任何问题或疑虑。



**提到的链接**：<a href="https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy#dspy)">DSPy - Langtrace AI Docs</a>: 未找到描述

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1288584086859939871)** (38 条消息🔥): 

> - `模型中的对话历史集成`
> - `STORM 论文及 GitHub 资源`
> - `使用 DSPy 创建 Agent`
> - `Dspy.LM 与 Azure OpenAI API`
> - `TypedChainOfThought 面临的挑战` 


- **探索对话历史集成**：一位成员询问了如何使用 `dspy.LM` 方法将对话历史添加到模型中，并建议通过之前的轮次进行摘要优化。
   - 另一位成员建议为 Predictors 使用 `multi_turn=True` 标志，这将简化 Signature 以有效处理对话历史。
- **STORM 论文资源**：一位成员索要了 STORM 论文链接，该链接可以在其 [GitHub 仓库](https://github.com/stanford-oval/storm)中找到。
   - 另一位成员提供了 [arXiv 上的 STORM 论文](https://arxiv.org/abs/2402.14207)链接，该论文讨论了使用 LLM 编写结构化文章。
- **在 DSPy 中创建 Agent**：一位成员分享了在 DSPy 中构建 Agent 的教程，强调了其探索性质并指出了当前框架的局限性。
   - 该成员表示，本教程旨在帮助理解如何使用 DSPy 有效地创建 Agent 应用。
- **切换至带有 Azure API 的 dspy.LM**：一位成员分享了从 `dspy.AzureOpenAI` 迁移到 `dspy.LM` 的经验，在使用 litellm 时遇到了 API 路径构建问题。
   - 不过，他们随后报告称已解决该问题（归因于 gpt-4o 模型），并对新 LM 的性能表示满意。
- **TypedChainOfThought 的问题**：一位成员指出，在迁移过程中发现了与 TypedChainOfThought 相关的解析问题，并遇到了 JSON 解包挑战。
   - 他们计划进一步调查此问题，但仍推荐使用新的 LM，因为它具有更顺畅的集成和改进的功能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.14207">Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models</a>: 我们研究了如何应用大语言模型从头开始撰写有据可查且组织严密的详尽文章，其广度和深度可与维基百科页面相媲美。这个尚未被充分探索的问题提出了新的...</li><li><a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>: 一个由 LLM 驱动的知识策展系统，可研究特定主题并生成带有引用的完整报告。 - stanford-oval/storm</li><li><a href="https://learnbybuilding.ai/tutorials/dspy-agents-from-scratch">A first attempt at DSPy Agents from scratch</a>: 这篇文章将尝试使用 DSPy 从零开始创建 Agent。其目的是教学，并探索如何在 DSPy 中从头构建 Agent。
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1288683231436210260)** (3 条消息): 

> - `模型中的类别数量`
> - `Class Signature 中的细微差别` 


- **探索模型中的类别数量**：@okhattab 询问了类别数量，引发了关于有效分类所需最佳数量的讨论。
   - 一位参与者提到他们目前正在处理 **5 个类别**，但建议 **10 个类别** 可能值得探索。
- **细微类别区分的挑战**：一位成员强调了类别区分的重要性，指出细微的差别可能会使 Signature 描述变得复杂。
   - 他们指出，强调这些细微差别对于提高模型性能和清晰度至关重要。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1288581828231561258)** (35 条消息🔥): 

> - `Yamashi 的绿卡困境`
> - `Llama 3.2 访问问题`
> - `Torchtune 数据集错误`
> - `MetaAI 登录限制`
> - `视觉问答 (Visual Question Answering) 数据集` 


- **Yamashi 的绿卡困境**: *“谁能给我一张绿卡？”* 在一段充满喜剧色彩的对话中，Yamashi 在讨论法律和合规障碍时开玩笑地恳求获得绿卡。
   - 他提到，*“是时候在 Delaware 开一家假公司了，”* 表现出对法律限制的沮丧。
- **Llama 3.2 访问问题**: 讨论围绕访问 **Llama 3.2** 展开，成员们注意到欧盟 (EU) 的限制阻止了直接使用该模型。
   - *“但我无法直接使用 Llama 3.2，”* Yamashi 哀叹道，反映了所面临的挑战。
- **Torchtune 数据集错误**: 一位成员讨论了在使用 **PackedDataset** 时遇到的关于序列长度限制的错误，并引用了 [GitHub issue #1689](https://github.com/pytorch/torchtune/issues/1689)。
   - 他们提出了一个简单的修复方案，并表示在进一步考虑测试要求后愿意提交 PR。
- **MetaAI 登录限制**: 成员们质疑对 **MetaAI** 的访问，强调欧盟用户面临限制且无法登录。
   - Yamashi 观察到，*“啊，确实我无法登录 MetaAI，”* 表明了连接限制。
- **丰富的视觉问答数据集**: 一位成员分享了对可用视觉问答 (Visual Question Answering) 数据集的兴奋之情，并链接到了 [Hugging Face](https://huggingface.co/datasets?task_categories=task_categories:visual-question-answering&sort=trending) 上的一系列热门集合。
   - 他们热衷于将这些资源用于 finetuning 的潜在应用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets?task_categories=task_categories:visual-question-answering&sort=trending">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://github.com/pytorch/torchtune/issues/1689">当使用 split_across_pack=True 时，PackedDataset 无法处理长度大于 2*max_seq_len 的长序列 · Issue #1689 · pytorch/torchtune</a>: 正如标题所述，它报告了 self._pad_pack() 中的运行时错误 f = {&quot;tokens&quot;:list(range(121)),&quot;labels&quot;:list(range(121))} x = PackedDataset([f],max_seq_len=60,split_across_pack=Tr...</li><li><a href="https://github.com/mirceamironenco/torchtune/blob/fix-packedds-seqlen/torchtune/datasets/_packed.py#L144.">mirceamironenco/torchtune 的 fix-packedds-seqlen 分支下的 torchtune/datasets/_packed.py</a>: 一个用于 LLM Fine-tuning 的原生 PyTorch 库。通过在 GitHub 上创建账号来为 mirceamironenco/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1288666337035423756)** (7 messages): 

> - `NeMo ASR 优化`
> - `针对 INT8 的 INT-FlashAttention`
> - `量化收益`
> - `Triton kernel 托管` 


- **NeMo ASR 优化显著提升性能**：根据 [Piotr Zelasko 的推文](https://x.com/PiotrZelasko/status/1838653087529209990)，由于针对 RNN-T、TDT 和 CTC 模型的一系列优化，NeMo ASR 现在在 **NVIDIA GPU** 上的运行速度比实时快 **2000-6000 倍**。
   - 这些模型不仅目前领跑 **HF Open ASR Leaderboard**，而且被描述为快速且具有成本效益，全部采用 **纯 PyTorch** 实现。
- **INT-FlashAttention 实现速度与精度的双重提升**：关于 [INT-FlashAttention](https://x.com/papers_anon/status/1839131401322639805?s=46) 的讨论强调了它能够实现 **全 INT8 激活** 和 GEMM kernel，与 FP16 和 FP8 格式相比，**推理速度提升了 72%**，**量化误差降低了 82%**。
   - 成员们指出，这对于缺乏 FP8 支持的 **Ampere 显卡** 可能非常有用，尽管仍需对其在 PTQ 和训练方面的影响进行更深入的分析。
- **关于 INT-FlashAttention 的 Kernel 托管讨论**：大家达成共识，认为 INT-FlashAttention 的新 kernel 理想情况下应该托管在 **torchao** 中，因为这可能是更合适的平台。
   - 一名成员建议开启一个 issue 来分享相关见解，并强调了进一步完善这些开发成果的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/papers_anon/status/1839">来自 crystal (@crystal) 的推文</a>：噢，太漂亮了~！</li><li><a href="https://x.com/PiotrZelasko/status/1838653087529209990">来自 Piotr Żelasko (@PiotrZelasko) 的推文</a>：看好了：NeMo ASR 现在在 @nvidia GPU 上可以轻松实现比实时快 2000-6000 倍 (RTFx) 的运行速度。我们开发了一系列优化，让 RNN-T、TDT 和 CTC 模型飞速运行！🔥 除了登顶...</li><li><a href="https://x.com/papers_anon/status/1839131401322639805?s=46">来自 PapersAnon (@papers_anon) 的推文</a>：INT-FlashAttention：为 INT8 量化启用 Flash Attention。全 INT8 激活和 GEMM kernel。对缺乏 FP8 的 Ampere 显卡非常有用。实现了 72% 的推理速度提升和 82% 的量化误差降低...
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1288929344072454146)** (2 messages): 

> - `MOToMGP pass manager 错误`
> - `Mojo/MAX 品牌背景图` 


- **团队正在处理 MOToMGP pass manager 问题**：团队正在积极处理 **'failed to run the MOToMGP pass manager'** 错误，并欢迎关于可以改进的 **Max / Mojo** 错误的反馈。
   - 鼓励用户分享与此问题相关的任何不满或建议。
- **对 Mojo/MAX 桌面背景的兴趣**：发起了一项投票，以调查大家对印有可爱的 Mojo 火焰和 MAX 宇航员的 **Mojo / MAX 品牌桌面背景** 的兴趣。
   - 参与者被要求通过表情符号投票，表达对这些背景设计的 **赞成** 或 **反对**。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1288869066634690654)** (1 messages): 

> - `验证流程`
> - `引导提问`
> - `频道发言限制` 


- **验证机器人回归以增强安全性**：验证机器人已重新上线，以帮助创建一个安全、无垃圾信息的社区；成员必须通过点击 "I'm human ✅" 并授权共享电子邮件来进行验证。
   - 附带了一个 GIF，演示如何成功验证成员身份。
- **未验证成员面临发言限制**：从 9 月 27 日开始，未验证成员只能在指定频道发言，但仍保留对所有其他频道的只读访问权限。
   - 此举旨在鼓励用户进行验证以获得完整的社区访问权限。
- **引入新的引导提问**：新增了两个引导提问：关于加入的主要原因以及对 MAX 的 GPU 产品早期访问的兴趣。
   - 这些问题旨在提升社区成员的整体体验。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1288724503861395561)** (19 条消息🔥): 

> - `Mojo compilation`
> - `Rust interoperability`
> - `Error handling improvements` 


- **Mojo 直接编译为机器码**：一位成员澄清说，Mojo 不会像 Python 那样为字节码缓存创建 **.pyc 文件**，因为它直接编译为机器码。
   - *“.pyc 是字节码缓存，Mojo 直接编译为机器码。”*
- **用于 Mojo 调用的 Rust 库**：一位成员分享了一个展示 `mojo calls to rust` 的项目，其中包含一个示例库以及详细的集成过程。
   - 该项目托管在 GitHub 上，可在[此处](https://github.com/better-mojo/learn-mojo/tree/main/packages/mojo-ffi/mojo-call-rust)查看。
- **将 .exe 反编译回 Mojo**：讨论了从 **.exe** 转换回 Mojo 代码的可能性，成员建议虽然付出努力可能实现，但更有可能还原为 **C**。
   - *“虽然非常痛苦，但理论上可行。更有可能的是它们被转换为 C。”*
- **错误信息需要改进**：一位成员表达了希望 Mojo 提供更好的错误信息，强调了错误来源和模块边界的清晰度。
   - 另一位成员表示赞同，建议错误信息应突出问题源头以提高可读性。
- **Variant 中缺乏标称和类型 (nominal sum type)**：关于 `Variant` 的 `__init__` 方法的讨论指出，由于缺乏标称和类型，导致错误信息不够明确。
   - *“我将其归咎于缺乏标称和类型。”*



**提到的链接**：<a href="https://github.com/better-mojo/learn-mojo/tree/main/packages/mojo-ffi/mojo-call-rust">learn-mojo/packages/mojo-ffi/mojo-call-rust at main · better-mojo/learn-mojo</a>：学习 Mojo。通过在 GitHub 上创建账号来为 better-mojo/learn-mojo 做出贡献。

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1288887854147178599)** (3 条消息): 

> - `MAX API usage`
> - `User feedback on MAX API` 


- **征求 MAX API 用户反馈**：一位成员向最近使用 **MAX API** 加载和执行模型的用户征求**反馈**。
   - 他们表示有兴趣了解用户希望看到的任何**挫败感**或改进，并鼓励在讨论串中进行公开交流。
- **邀请分享关于 MAX API 的想法**：该成员重申，希望用户分享关于 MAX API 体验的**任何及所有想法**。
   - 他们加入了一个轻松的表情符号，以营造友好的反馈分享环境。


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1288887526488019116)** (21 条消息🔥): 

> - `LLM Response Behavior`
> - `Source Document Retrieval`
> - `Debugging Tools Usage` 


- **LLM 响应“未找到信息”**：一位成员解释说，在调用 chain 提问时，**LLM** 有时会回答“对不起，我不知道”，但仍然检索了源文档。
   - 他们希望文档检索能以模型实际拥有相关信息为前提。
- **源文档被不必要地检索**：同一位成员指出，如果 LLM 表示没有有用信息，则不应返回**源文档**，否则会导致困惑。
   - 他们澄清说，虽然大多数响应令人满意，但在 LLM 响应为否定时，他们不希望出现源文档。
- **对调试工具的担忧**：另一位成员询问原作者是否使用了像 Langsmith 这样的**调试**或分析工具，原作者因隐私考量拒绝了。
   - 建议使用 **Langfuse** 托管等替代方案，以满足监控需求而不损害数据隐私。
- **请求代码示例**：一位参与者请求查看代码示例，以便更好地理解原作者面临的情况和挑战。
   - 原作者同意在第二天提供示例，因为当时已是深夜。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1288586714339741829)** (14 messages🔥): 

> - `Generative AI 泡沫`
> - `博士生参与`
> - `指令头部格式化`
> - `Token 推测`
> - `社区欢迎氛围` 


- **Generative AI 泡沫面临审查**：一位成员对 **Generative AI** 行业（尤其是 **ChatGPT**）表示担忧，认为其正处于崩溃边缘，受 **Mira Murati** 等人近期离职的影响。
   - 他们引用了一份富有洞察力的通讯，声称生成式 AI 热潮是**不可持续的**，可能会损害大型科技公司和公众认知。
- **博士生关注 Cohere 动态**：一位新成员表示，他们在博士学业即将结束之际加入社区，以跟进相关活动。
   - 这突显了 Cohere 是对 AI 讨论感兴趣的学术界人士的宝贵资源。
- **指令头部格式化讨论**：一位成员寻求关于**指令头部格式化**的建议，并询问 **RAG 包含项**在提交给 LLM 时将如何呈现。
   - 这显示了社区内对改进教学设计的浓厚兴趣。
- **社区欢迎新人**：多位成员欢迎新人加入社区，并鼓励他们提出关于 **Cohere** 的问题。
   - 这展示了一个专注于学习和参与的支持性环境。
- **不鼓励“锡纸帽”式的推测**：一位成员主张在有关 AI 行业未来的推测中坚持事实，强调 **Cohere 的立足点**在于现实。
   - 这反映了比起煽情理论，人们更渴望深思熟虑的讨论。



**提到的链接**：<a href="https://www.wheresyoured.at/subprimeai/">The Subprime AI Crisis</a>：我在本通讯中所写的任何内容都不是为了播种怀疑或“仇恨”，而是对我们现状以及在当前路径下可能走向何方的冷静评估。我相信人工...

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1288822057575911435)** (7 messages): 

> - `Rerank 微调`
> - `每个查询的平均硬负样本数 (Avg Hard Negatives per Query)`
> - `模型性能与数据质量` 


- **关于平均硬负样本计算的问题**：一位用户询问 **'Avg Hard Negatives per Query'** 是如何计算的，并指出其数据集中硬负样本比例不足 **10%**。
   - *Cohere 澄清说他们不会在后台添加负样本*，并建议验证数据质量。
- **训练后的模型性能**：在训练过程结束后，一位用户报告说该模型的表现仅略好于**默认的 English v3 reranker**。
   - 他们推测**数据质量**可能是导致这种表现不佳的一个因素。
- **讨论数据特征**：在关于数据集的对话中，提到其中包含许多**财务表格**，尽管许多数值已被删除。
   - 这种数据呈现方式的转变可能会影响模型从训练中有效学习的能力。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1288699898971029574)** (5 messages): 

> - `Arbitrary View Mergeability Proof`（任意 View 可合并性证明）
> - `Pairwise vs Global Mergeability`（两两合并与全局合并的对比）
> - `View Merging Optimization`（View 合并优化）
> - `View Merging and Symbolic Reduction`（View 合并与符号化规约）


- **讨论了任意 View 可合并性证明**：在 GitHub 上分享了一个关于**任意 View 可合并性**的证明（不考虑 masks 或 reshapes），详细阐述了 **Tinygrad** 中 View 管理的重要方面。
   - 该 [证明可以在此处找到](https://github.com/pschilliOrange/Tinygrad-view-merging-proof/blob/8672f35c1147798c8e9a78bfab28b9ff79bf45e6/Proof%20for%20when%20a%20new%20view%20must%20be%20appended.pdf)，并附有项目概览。
- **两两合并与全局合并的区别**：有人提出担心，目前的证明只解决了两个 View 的**两两合并（pairwise mergeability）**，但没有涵盖多个组合 View 的**全局合并（global mergeability）**或 **shapeTrackers** 的等价性。
   - 在追加新 View 时折叠所有 View 的做法被认为在实际实现中可能**成本过高**。
- **将 Offsets 和 Masks 纳入证明**：有意将 **offsets** 和 **masks** 引入现有的合并性证明中，以增强其全面性。
   - 计划最终将这项工作适配到 **Lean**（一种用于数学形式化的证明助手）中。
- **View 合并作为符号化规约**：有建议认为 **View 合并**可能等同于**符号化规约（symbolic reduction）**，这为优化开辟了新途径。
   - 这一概念可能允许通过**模式重写（pattern rewrites）**来重写 View，正如 **Tinygrad** 仓库中的一个 [单元测试示例](https://github.com/tinygrad/tinygrad/blob/master/test/unit/test_simplify_valid_idx.py) 所暗示的那样。



**提到的链接**：<a href="https://github.com/pschilliOrange/Tinygrad-view-merging-proof/blob/8672f35c1147798c8e9a78bfab28b9ff79bf45e6/Proof%20for%20when%20a%20new%20view%20must%20be%20appended.pdf">Tinygrad-view-merging-proof/Proof for when a new view must be appended.pdf at 8672f35c1147798c8e9a78bfab28b9ff79bf45e6 · pschilliOrange/Tinygrad-view-merging-proof</a>：通过在 GitHub 上创建账号，为 pschilliOrange/Tinygrad-view-merging-proof 的开发做出贡献。

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1288613336568692766)** (12 messages🔥): 

> - `Tinygrad performance issues`（Tinygrad 性能问题）
> - `Metal compatibility errors`（Metal 兼容性错误）
> - `Comparison between Tinygrad and PyTorch`（Tinygrad 与 PyTorch 的对比）
> - `Tinygrad customization benefits`（Tinygrad 自定义优势）


- **Tinygrad 训练性能滞后**：一位用户报告称使用 **Tinygrad** 训练极其缓慢，并收到了一块 **4090 GPU** 以提升性能，但由于采样代码中的 bug，输出质量很差。
   - 他们指出问题不在于训练速度，而在于采样逻辑中的错误实现。
- **Metal 的双精度错误**：一位用户遇到了 Metal 错误，提示不支持 '**double**'，这与 **NumPy** 默认使用双精度（double precision）有关。
   - 将 tensor 转换为 **float32** 解决了该问题，但随后他们遇到了另一个错误，提示可能存在缓冲区资源位置问题。
- **Tinygrad 在 Metal 上的限制**：一位参与者指出 **Tinygrad** 中的融合内核（fused kernel）触及了 **Metal 的缓冲区限制（buffer limit）**，并将在接下来的更新中尝试解决方法。
   - 这表明在 **Tinygrad** 中使用 **Metal** 作为后端时可能存在内存问题。
- **Tinygrad 与 PyTorch 及 CUDA 的对比**：讨论了 **Tinygrad** 作为 **PyTorch** 快速替代方案的可能性，并询问其与直接在 **CUDA** 中编码相比如何。
   - **Tinygrad** 被定位为使用 **CUDA** 作为后端的编译器，但 **PyTorch** 的优势在于经过深度优化的手工定制 **CUDA** kernel。
- **Tinygrad 中的自定义与优化**：有人指出，与 **PyTorch** 中固定的 kernel 相比，**Tinygrad** 通过自定义 kernel 生成允许更广泛的优化可能性。
   - **Tinygrad** 的一个关键特性是其探索各种 kernel 优化的能力，这可以带来性能提升。



**提到的链接**：<a href="https://github.com/mesozoic-egg/tinygrad-notes/tree/main">GitHub - mesozoic-egg/tinygrad-notes: Tutorials on tinygrad</a>：Tinygrad 教程。通过在 GitHub 上创建账号，为 mesozoic-egg/tinygrad-notes 的开发做出贡献。

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1288638152457584702)** (9 条消息🔥): 

> - `LLaMA 3.2 Vision`
> - `OpenAI's Function Calling API`
> - `Voice Cloning Technology` 


- **LLaMA 3.2 Vision 90B 在 Image Captioning 方面表现卓越**：成员们注意到 **LLaMA 3.2 Vision 90B** 在 **image captioning** 方面似乎能力极强，甚至 **11B** 版本也引起了关注。
   - 一位成员幽默地建议使用该无限制模型为整个 **LAION dataset** 生成描述，这显示了其潜力。
- **OpenAI's Function Calling API 咨询**：一位成员询问了 **OpenAI's function calling API** 的内部工作原理，推测它是运行了一个微调模型，还是对其他模型的输出进行了检查。
   - 这个问题反映了人们对 API 设计细节和性能增强的持续好奇。
- **免费获取 LLaMA 3.2 Vision**：TogetherCompute 宣布与 **AI at Meta** 合作，**免费提供 LLaMA 3.2 11B Vision**，允许开发者尝试多模态 AI。
   - 他们在[此链接](https://api.together.ai/playground/chat/meta-llama/Llama-Vision-Free)提供了一个免费的模型端点，同时也提供付费选项以获得更强的性能。
- **对 Voice Cloning 演示感到兴奋**：一位成员兴奋地分享了一段 YouTube 视频，内容是他们与自己的声音克隆进行对话，表达了对这项技术的喜爱。
   - 相关的[视频](https://youtu.be/X8KhTlKoagg)突出了 Voice Cloning 技术的持续进步及其有趣的应用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/togethercompute/status/1839071026728333778">来自 Together AI (@togethercompute) 的推文</a>: 🚀 我们与 @AIatMeta 合作，免费提供 Llama 3.2 11B Vision，以便开发者可以零成本尝试开源多模态 AI。除了我们的免费额度外，在有限的时间内...</li><li><a href="https://youtu.be/X8KhTlKoagg">我正在与使用我声音的 BUD-E 交谈！ :D</a>: https://discord.gg/pCPJJXP7Qx
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1288576993222459412)** (6 条消息): 

> - `MaskBit`
> - `MonoFormer`
> - `HuggingFace papers`
> - `VQGAN enhancements`
> - `Multimodality transformer models` 


- **MaskBit 彻底改变了图像生成**：这项研究介绍了 **MaskBit**，这是一种通过 **bit tokens** 实现的无嵌入图像生成方法，显著改进了用于类别条件图像生成的经典 VQGAN Transformer 模型。
   - 他们的新模型在 ImageNet 基准测试中仅凭 **305M 参数** 就达到了惊人的 **FID 1.52**，表明无嵌入方法可以超越现有技术。
- **MonoFormer 统一了生成过程**：**MonoFormer** 提出了一种单一的 Transformer 架构，能够熟练地处理 **autoregression** 和 **diffusion**，消除了对文本和图像生成使用独立模型的需求。
   - 通过利用共享的训练方法，它在有效生成文本的同时保持了具有竞争力的图像生成性能，更多细节可以在其[项目页面](https://monoformer.github.io/)找到。
- **HuggingFace 展示了最新进展**：HuggingFace 分享了两篇**颇有趣的论文**，可能对多模态模型和图像合成的未来产生影响。
   - 虽然目前只是粗略浏览，但初步发现表明 Transformer 模型的效率和性能有了令人期待的进步。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1288643369005875304)** (8 条消息🔥): 

> - `Quiz 3 问题`
> - `RAG 模型能力`
> - `LLM Agents 的社会对齐`
> - `CCMP_AI 项目`
> - `课程注册确认` 


- **Quiz 3 问题引发困惑**：一位成员对 **Quiz 3 的一个问题**表示困惑，指出在演讲者关于受限流（constrained flows）和非受限流（unconstrained flows）的解释中并未提及该内容。
   - 另一位成员指出，相关信息确实在幻灯片中，确认了测验内容。
- **RAG 模型在多模态数据上遇到挑战**：有人对最新模型的 **RAG 能力**表示担忧，特别是它们在处理文本、表格和图像等多模态数据时的表现。
   - 提到了 **Claude 3** 的出色表现，它在解释流程图方面表现良好。
- **Agentic RAG 项目初具规模**：一位成员分享了他们的 **ccmp_ai 项目**是一个非受限的 RAG 模型，并提供了令另一位成员觉得非常有用的新术语。
   - 他们将其称为具有动态问题域扩展能力的 **agentic RAG**。
- **关于 LLM Agent 对齐的查询**：一位成员询问接下来的课程是否会涉及 LLM Agent 的**社会对齐（social alignment）**，并指出过去的讲座似乎侧重于技术对齐。
   - 他们指出最后两节课仅涵盖了 **AI safety** 的相关方面。
- **课程注册的不确定性**：一位成员寻求关于通过 Google 表单进行课程注册的澄清，询问报名后是否会收到确认。
   - 他们希望尽快解决此问题，以了解如何获得课程访问权限。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1288647904462110780)** (1 条消息): 

> - `医疗多智能体系统`
> - `AgentClinic 基准测试` 


- **医疗多智能体系统研究摘要**：最近一项名为 [AgentClinic: A Multimodal Agent Benchmark](https://open.substack.com/pub/yanpan0508/p/agentclinic-a-multimodal-agent-benchmark?r=ad7en) 的研究专注于医疗多智能体系统，讨论了关键方法论和发现。
   - 该研究强调了多智能体系统在医疗领域的协作潜力，推动了 AGI 应用的边界。
- **Yvaine 的 Substack 发布**：Yvett 的 Substack 专栏“Embracing AGI”于两个月前发布，提供了对人工智能不断发展领域的见解。
   - Yvaine 在她的博客中强调了 AGI 在医疗背景下的重要性，旨在吸引对多智能体系统进展感兴趣的社区。



**提到的链接**：<a href="https://open.substack.com/pub/yanpan0508/p/agentclinic-a-multimodal-agent-benchmark?r=ad7en&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true">AgentClinic: a multimodal agent benchmark</a>：论文阅读：模拟临床环境中的 AI 评估

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1288579347724107870)** (6 条消息): 

> - `Llama 3.2 实验`
> - `Open Interpreter 问题`
> - `旧金山科技周 (Tech Week SF) 聚会` 


- **对 Llama 3.2 实验的期待**：一位用户询问了计划针对 Llama 3.2 进行的**实验**和**测试**，表明了对其能力的持续关注。
   - 社区似乎渴望探索该模型新的潜在应用和功能。
- **Open Interpreter 无法统计文件数量**：一位成员报告称，在使用 **3b** 模型配合 Open Interpreter 统计桌面文件时，模型**未能**执行该任务。
   - 这一失败引发了对该模型执行此类任务可靠性的担忧。
- **旧金山科技周击掌机会**：一位用户表达了参加旧金山 **Tech Week** 的兴奋之情，并建议见面击掌。
   - 这突显了社区在技术活动期间进行社交和联系的热情。


  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1288631996456632342)** (3 messages): 

> - `Llama 3.2 Testing`
> - `NERD Task Challenges` 


- **Llama 3.2 表现平平**：在测试了 **Llama 3.2** 90b 后，一位成员表示失望，称其表现不如 **Llama 3.1** 70b。
   - 他们引用了一个名为 [YouTube video](https://www.youtube.com/watch?v=_MQVHyEeER4) 'Llama-3.2 (1B, 3B, 11B, 90B) : The WORST New LLMs EVER!?' 的视频，其中详细介绍了他们的发现。
- **NERD 任务的挑战**：一位成员描述了一个 **NERD** 任务，重点是将文本链接到新闻文章中提到的个人的 **wiki** 条目。
   - 由于提取和匹配相关信息涉及复杂的细节，该任务被认为非常复杂。



**提及的链接**: <a href="https://www.youtube.com/watch?v=_MQVHyEeER4">Llama-3.2 (1B, 3B, 11B, 90B) : The WORST New LLMs EVER!? (Fully Tested &amp; Beats &quot;Nothing&quot;)</a>: 加入此频道以获得特权：https://www.youtube.com/@AICodeKing/join 在这段视频中，我将全面测试所有新的 Llama-3.2 Vision &amp; Small La...

  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1288850703275659335)** (3 messages): 

> - `Alternatives to partition_pdf`
> - `Channel etiquette` 


- **寻找 partition_pdf 的替代方案**：一位成员询问是否有人能推荐 **unstructured 'partition_pdf'** 的替代方案，用于从 PDF 中提取图像和表格。
   - 他们正在寻找一个更有效的工具来处理这项任务。
- **频道礼仪提醒**：另一位成员提醒，在多个频道发布相同的问题将被视为 **spam**。
   - 他们采取了行动，删除了其他频道中的重复问题以维持秩序。


  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

rusch: 兄弟，这怎么不算推广
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1288648747076685906)** (1 messages): 

> - `Mozilla AI`
> - `Locally-run AI models`
> - `Large Language Models (LLMs)`
> - `Continue tool`
> - `Nature article` 


- **Mozilla AI 在 Nature 文章中备受关注**：Mozilla AI 及其关键项目在 Nature 的文章《忘掉 ChatGPT：为什么研究人员现在在笔记本电脑上运行小型 AI》中被提及。
   - Mozilla 的开源 AI 负责人分享了见解，强调了赋能用户的**本地运行 AI 模型**的兴起。
- **适用于所有系统的 LLMs**：文章提到了一个关于让 **Large Language Models (LLMs)** 在各种系统上运行的项目，展示了其通用性。
   - 这被强调为使强大的 AI 在不同平台上可用的重大进展。
- **Continue 工具获得认可**：Continue（由一位成员在最近组织的演讲中介绍）被强调为 **AI-assisted coding** 的宝贵工具。
   - 这一认可展示了它在 AI 社区中的有效性和相关性。
- **全文直接链接**：摘要提供了[全文链接](https://discord.com/channels/1089876418936180786/1288648264069025883/1288648264069025883)，供感兴趣的读者了解更多细节。
   - 包含此链接有助于引导社区成员访问所讨论见解的原始出处。


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1288953002731765780)** (1 条消息): 

> - `Function Calling Evaluation`
> - `Custom Evaluation Dataset`
> - `LLM Integration`
> - `Berkeley Function-Calling Leaderboard` 


- **澄清代码库中的 Function Calling 评估**：一位用户对代码库中的 Function Calling 评估表示困惑，询问是否可以在使用自定义 API/LLM 的同时提供自己的评估数据集。
   - 他们强调，在集成包含 **<prompt>, <llm_response>, <ideal response>** 的数据集以进行错误细分（error breakdown）方面缺乏清晰度。
- **希望对自定义数据集进行错误细分**：用户正在寻找一个可以分析其数据集并提供类似于 [BFCL metrics](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#metrics) 中错误见解的工具包。


**提及的链接**：<a href="https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#metrics">Berkeley Function Calling Leaderboard</a>：未找到描述

  

---



---



---



{% else %}


> 完整的频道细分内容已在邮件中截断。 
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}