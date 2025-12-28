---
companies:
- mistral-ai
- meta-ai-fair
- hugging-face
- arcee-ai
- deepseek-ai
- openai
- anthropic
date: '2024-09-12T00:30:22.330132Z'
description: '**Mistral AI** 发布了 **Pixtral 12B**，这是一款权重开放的**视觉语言模型**。该模型以 **Mistral
  Nemo 12B** 为文本主干，并配备了一个 4 亿参数的视觉适配器，具有 **131,072 个词元**的超大词汇量，并支持 **1024x1024 像素的图像**。此次发布在推出开源多模态模型方面显著领先于
  **Meta AI**。在 Mistral AI 峰会上，官方分享了其架构细节和基准测试表现，展示了强大的 OCR（光学字符识别）和屏幕理解能力。


  此外，**Arcee AI** 宣布推出 **SuperNova**，这是一款蒸馏自 **Llama 3.1 70B 和 8B** 的模型，在基准测试中表现优于
  Meta 的 Llama 3.1 70B 指令微调版。**DeepSeek** 发布了 **DeepSeek-V2.5**，在 **HumanEval** 编程测试中获得
  **89 分**，在代码任务上超越了 **GPT-4-Turbo**、Opus 和 Llama 3.1。**OpenAI** 计划近期将 **Strawberry**（草莓）作为
  ChatGPT 的一部分发布，尽管其具体能力仍存争议。**Anthropic** 推出了 **Workspaces**（工作区），旨在通过增强的访问控制功能来管理多个
  Claude 部署。'
id: 2c30c16c-0df7-443b-93bd-09097bce542a
models:
- pixtral-12b
- mistral-nemo-12b
- llama-3-1-70b
- llama-3-1-8b
- deeps-eek-v2-5
- gpt-4-turbo
- llama-3-1
- strawberry
- claude
original_slug: ainews-pixtral-12b-mistral-beats-llama-to
people:
- reach_vb
- devendra_chapilot
- _philschmid
- rohanpaul_ai
title: Pixtral 12B：Mistral 在多模态领域击败 Llama
topics:
- vision
- multimodality
- ocr
- benchmarking
- model-release
- model-architecture
- model-performance
- fine-tuning
- model-deployment
- reasoning
- code-generation
- api
- access-control
---

<!-- buttondown-editor-mode: plaintext -->**Vision Language Models are all you need.**

> 2024年9月10日至9月11日的 AI News。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务器（**216** 个频道和 **3870** 条消息）。预计节省阅读时间（以 200wpm 计算）：**411 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

昨晚深夜，Mistral 恢复了往日的风格——与 Mistral Large 2（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-mistral-large-2/)）不同，**Pixtral** 是以 [磁力链接的形式发布](https://x.com/MistralAI/status/1833758285167722836) 的，没有附带论文或博客文章。此时正值 Mistral AI Summit 今日召开，庆祝公司成立一周年。

[Huggingface 的 VB](https://x.com/reach_vb/status/1833779749430124692) 给出了最详尽的解析：

![Mistral 发布了 Pixtral 12B Vision Language Model。关于此次发布的一些说明：1. 文本骨干网络：Mistral Nemo 12B 2. 视觉适配器：400M 3. 使用 GeLU（用于视觉适配器）和 2D RoPE（用于视觉编码器） 4. 更大的词表 - 131,072 5. 三个新的特殊 Token - `img`, `img_break`, `img_end` 6. 图像尺寸：1024 x 1024 像素 7. Patch 尺寸：16 x 16 像素 8. mistral_common 中的 Tokenizer 支持 9. 模型权重采用 bf16 10. 尚未看到推理代码 11. 权重已上传至 Hugging Face Hub 🤗 祝贺 Mistral 在多模态领域成功领先 Meta 🐐](https://assets.buttondown.email/images/dd044b56-f77a-4891-a855-8b4715c1ecda.png?w=960&fit=max)


VB 正确地指出，Mistral 在发布权重开放的多模态模型方面击败了 Meta。您可以在 [mistral-common 更新](https://github.com/mistralai/mistral-common/releases/tag/v1.4.0) 中看到新的 ImageChunk API：


![image.png](https://assets.buttondown.email/images/fbbe0318-4123-485c-a733-d33ea08864a3.png?w=960&fit=max)


对技术细节感兴趣的朋友可以在这里查看 [更多超参数 (hparams)](https://discord.com/channels/822583790773862470/1283320559240876052/1283322542202818570)。

在峰会上，Devendra Chapilot 分享了更多关于架构的细节（专为 [任意尺寸和交错输入](https://x.com/swyx/status/1833932883347865802) 设计）


![image.png](https://assets.buttondown.email/images/ab872886-ec72-42b5-9fdf-eddf1e34ea0f.png?w=960&fit=max)


同时展示了令人印象深刻的 [OCR](https://x.com/swyx/status/1833934254834942047) 和 [屏幕理解](https://x.com/swyx/status/1833935106605809993) 示例（虽然有错误！），以及与开源模型替代方案相比具有优势的 Benchmark 性能（尽管一些 [Qwen](https://x.com/_philschmid/status/1833954941624615151) 和 [Gemini Flash 8B](https://x.com/OfficialLoganK/status/1833951504232780014) 的数据有所偏差）：


![image.png](https://assets.buttondown.email/images/caf1509d-2538-42bb-81f4-c9cce3531c6c.png?w=960&fit=max)



这仍然是一项极其令人印象深刻的成就，对于 Mistral 来说是实至名归的胜利，他们还展示了其 [模型优先级](https://x.com/swyx/status/1833927941824414157) 和产品组合。


![image.png](https://assets.buttondown.email/images/8d637413-cad5-4dd6-b5a7-7368557f70be.png?w=960&fit=max)


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

**AI 模型更新与基准测试**

- **Arcee AI 的 SuperNova**：[@_philschmid](https://twitter.com/_philschmid/status/1833599779902787713) 宣布发布 SuperNova，这是一个经过蒸馏推理的 Llama 3.1 70B & 8B 模型。它在**各项基准测试中超越了 Meta Llama 3.1 70B instruct**，并且是 **IFEval 上表现最好的开源 LLM**，超过了 OpenAI 和 Anthropic 的模型。

- **DeepSeek-V2.5**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833638752842887533) 报告称，新的 DeepSeek-V2.5 模型在 **HumanEval 上得分为 89**，在编程任务中超越了 GPT-4-Turbo、Opus 和 Llama 3.1。

- **OpenAI 的 Strawberry**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833596635273658449) 分享称，OpenAI 计划在未来两周内将 Strawberry 作为其 ChatGPT 服务的一部分发布。然而，[@AIExplainedYT](https://twitter.com/AIExplainedYT/status/1833527132498112532) 指出关于其能力的报告存在矛盾，有人称其为“对人类的威胁”，而早期测试者则认为“它略好一些的回答不值得等待 10 到 20 秒”。

**AI 基础设施与部署**

- **Anthropic Workspaces**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1833529395765776615) 在 Anthropic Console 中引入了 Workspaces，允许用户管理多个 Claude 部署，设置自定义支出或速率限制，分组 API 密钥，并通过用户角色控制访问权限。

- **SambaNova Cloud**：[@AIatMeta](https://twitter.com/AIatMeta/status/1833517936134545571) 强调 SambaNova Cloud 正在为 405B 模型的推理设定新标准，开发者今天即可开始构建。

- **Groq 性能**：[@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1833436848221261849) 声称 Groq 创造了新的速度记录，并计划进一步提升。

**AI 开发工具与框架**

- **LangChain Academy**：[@LangChainAI](https://twitter.com/LangChainAI/status/1833529605262872770) 推出了他们的第一门课程《LangGraph 简介》，教授如何使用基于图的工作流构建可靠的 AI Agent。

- **Chatbot Arena 更新**：[@lmsysorg](https://twitter.com/lmsysorg/status/1833582238983934078) 在其排行榜上添加了一个新的“Style Control”按钮，允许用户将其应用于总榜和硬核提示词（Hard Prompts）榜单，以观察排名变化。

- **Hugging Face 集成**：[@multimodalart](https://twitter.com/multimodalart/status/1833459429557088314) 分享称，现在可以轻松地将图像添加到 Hugging Face 上的 LoRA 模型库中。

**AI 研究与洞察**

- **Sigmoid Attention**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833654018109055391) 讨论了 Apple 的一篇论文，该论文提出了 Flash-Sigmoid，这是一种硬件感知且内存高效的 Sigmoid Attention 实现，在 H100 GPUs 上相比 FlashAttention2-2 可实现高达 **17% 的推理算子加速**。

- **Mixture of Vision Encoders**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833630574872789411) 分享了关于使用混合视觉编码器（Mixture of Vision Encoders）增强 MLLM 在各种视觉理解任务中性能的研究。

- **引用生成**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833628769216827495) 报告了一种用于长文本 QA 引用生成的新方法，提升了性能和可验证性。

**行业新闻与趋势**

- **Klarna 的技术栈变革**：[@bindureddy](https://twitter.com/bindureddy/status/1833603866207916475) 指出 Klarna 关闭了 Salesforce 和 Workday，取而代之的是由 AI 创建的更简单的技术栈，其运行成本可能比传统的 SaaS 应用程序**便宜 10 倍**。

- **AI 影响力人物争议**：[@corbtt](https://twitter.com/corbtt/status/1833633946644713582) 报道了关于 Reflection-70B 模型的争议，称经过调查，他们认为达到声称基准测试水平的模型从未存在过。

- **Mario Draghi 的欧盟报告**：[@ylecun](https://twitter.com/ylecun/status/1833600877606945002) 分享了 Mario Draghi 对欧洲生产力停滞的分析及解决方法，强调了欧盟与美国之间的竞争力差距。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

> 抱歉，我们的流水线今天出了点问题，正在修复中。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 研究与技术**

- **AI 唇语识别**：一段展示 AI 驱动的唇语识别技术的视频引发了关于其潜在应用和隐私影响的讨论。一些评论者对大规模监控和 Deepfake 的潜力表示担忧，而另一些人则看到了其在无障碍辅助方面的益处。[来源](https://www.reddit.com/r/singularity/comments/1fdkpls/lipreading_with_ai/)

- **中国拒绝签署 AI 核武器禁令**：中国拒绝签署一项禁止 AI 控制核武器的协议，引发了对未来 AI 战争的担忧。文章指出，中国希望在相关决策中保留“人为因素”。[来源](https://www.reddit.com/r/singularity/comments/1fdwt1q/china_refuses_to_sign_agreement_to_ban_ai_from/)

- **无人驾驶 Waymo 车辆安全性提升**：一项研究发现，无人驾驶 Waymo 车辆发生的严重事故远少于人类驾驶的车辆，且大多数事故是由其他车辆的人类驾驶员造成的。这突显了自动驾驶技术的潜在安全优势。[来源](https://www.reddit.com/r/singularity/comments/1fdyeje/driverless_waymo_vehicles_get_into_far_fewer/)

**AI 模型开发与发布**

- **OpenAI 的 GPT-4.5 "Strawberry"**：有报告称 OpenAI 可能会在两周内发布名为 "Strawberry" 的新纯文本 AI 模型。据称，该模型在回答前会“思考” 10-20 秒，旨在减少错误。然而，一些测试者发现，与 GPT-4 相比，其改进并不明显。[来源](https://www.reddit.com/r/singularity/comments/1fdit9r/new_details_on_openais_strawberry_openai_may/)

- **OpenAI 研究负责人离职**：参与 GPT-4 和 GPT-5 开发的一位 OpenAI 核心研究人员已离职并创办自己的公司，引发了关于 AI 行业人才留存和竞争的讨论。[来源](https://www.reddit.com/r/OpenAI/comments/1fe3i8q/openai_research_lead_for_gpt4ogpt5_leaves_to/)

- **Flux 微调改进**：开发者通过针对特定层进行微调，在 Flux AI 模型上取得了进展，这可能会提高训练速度和推理质量。这展示了在优化 AI 模型性能方面的持续努力。[来源](https://www.reddit.com/r/StableDiffusion/comments/1fdczqy/flux_fine_tuning_with_specific_layers/)

**AI 在娱乐与媒体中的应用**

- **James Earl Jones 授权达斯·维达声音版权**：演员 James Earl Jones 已签署协议，授权 AI 重塑其标志性的达斯·维达（Darth Vader）配音，突显了 AI 在娱乐行业日益增长的应用，并引发了关于配音行业未来的讨论。[来源](https://www.reddit.com/r/singularity/comments/1fdcesl/james_earl_jones_signed_over_rights_for_ai_to/)

- **Domo AI 视频放大工具发布**：Domo AI 发布了一款快速视频放大工具，可将视频增强至 4K 分辨率，展示了 AI 驱动的视频处理技术的进步。[来源](https://www.reddit.com/r/singularity/comments/1fdw8xm/domo_ai_just_launched_its_video_upscaler_its_fast/)

**AI 行业与研究趋势**

- **Sergey Brin 对 AI 的关注**：Google 联合创始人 Sergey Brin 表示，由于对近期 AI 的进展感到兴奋，他现在每天都在 Google 工作，这表明科技行业领袖对 AI 的高度关注和投入。[来源](https://www.reddit.com/r/singularity/comments/1fdtp0g/sergey_brin_says_he_is_working_at_google_every/)

- **公众对 AI 取代工作的看法**：一个梗图（Meme）帖子引发了关于公众对 AI 可能取代工作的态度的讨论，突显了围绕 AI 对就业影响的复杂情绪和担忧。[来源](https://www.reddit.com/r/singularity/comments/1fdljfg/the_public_be_like/)


---

# AI Discord 摘要回顾

> 由 GPT4O-Aug (gpt-4o-2024-08-06) 生成的摘要之摘要的摘要

**1. 模型性能与基准测试**

- **Pixtral 12B 表现优于竞争对手**：来自 Mistral 的 **[Pixtral 12B](https://x.com/MistralAI/status/1833758285167722836)** 在 Mistral 峰会上展示了其在 OCR 任务中优于 **Phi 3** 和 **Claude Haiku** 等模型的表现。
  - 现场演示强调了 **Pixtral** 在处理图像尺寸方面的灵活性，引发了关于其与竞争对手相比准确性的讨论。
- **Llama-3.1-SuperNova-Lite 在数学方面表现出色**：**[Llama-3.1-SuperNova-Lite](https://huggingface.co/arcee-ai/Llama-3.1-SuperNova-Lite)** 在数学任务中表现优于 **Hermes-3-Llama-3.1-8B**，在吠陀乘法（Vedic multiplication）等计算中保持了准确性。
  - 尽管两个模型都面临挑战，但该模型在处理数字方面的卓越表现得到了关注，SuperNova-Lite 显示出更好的数字完整性。


**2. AI 与多模态创新**

- **Mistral 的 Pixtral 12B 视觉模型**：Mistral 发布了视觉多模态模型 **[Pixtral 12B](https://x.com/MistralAI/status/1833758285167722836)**，该模型拥有 **220 亿参数**，并针对单 GPU 使用进行了优化。
  - 虽然目前仅限于 **4K context size**，但人们对 11 月份推出的长上下文模型寄予厚望，以增强多模态处理能力。
- **Hume AI 的 Empathic Voice Interface 2**：**[Hume AI](https://x.com/hume_ai/status/1833906262351974483)** 推出了 **Empathic Voice Interface 2 (EVI 2)**，融合了语言和语音，以增强情感智能应用。
  - 该模型现已可用，邀请用户创建需要更深层次情感参与的应用，标志着语音 AI 的进步。


**3. 软件工程与 AI 协作**

- **SWE-bench 凸显 GPT-4 的效率**：**[SWE-bench](https://www.swebench.com/index.html)** 结果显示，**GPT-4** 在 15 分钟以内的任务中表现优于 **GPT-3.5**，在没有人类基准对比的情况下展示了更高的效率。
  - 尽管有所改进，但两个模型在超过四小时的任务上都表现不佳，这表明了问题解决能力的局限性。
- **AI 与软件工程集成的挑战**：关于 AI 与软件工程集成的讨论反映了日益增长的兴趣，AI 模型虽然展现出潜力，但缺乏细致的人类洞察力。
  - AI 在软件工程任务中的作用正在迅速发展，但其在有效性和洞察力方面仍难以与经验丰富的工程师相媲美。


**4. 开源 AI 工具与框架**

- **对 Modular 的 Mojo 24.5 发布的期待**：随着社区会议讨论解决接口清晰度问题，人们对预计在一周内发布的 **Mojo 24.5** 充满期待。
  - 用户热切期待改进产品时间表的沟通，以防止误解并确保为变化做好准备。
- **OpenRouter 增强编程工具集成**：OpenRouter 提供了 Claude API 的高性价比替代方案，强调了多模型的集中式实验。
  - 讨论强调了绕过初始 **rate limits** 和更低的成本，使其成为开发者的首选。


---

# 第一部分：高层级 Discord 摘要

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 用户反馈机会**：团队正在积极寻找尚未与 **Magic** 互动的用户，通过 30 分钟的通话提供反馈，并提供独家周边奖励；感兴趣的人员可以[在此预约](https://modul.ar/user-feedback)。关于未来周边获取的咨询得到了积极回应，表明未来可能有更广泛的获取渠道。
   - 成员们对开设周边商店以提供更多选择表示了兴趣，反映出社区对额外参与机会的热情。
- **Mojo 24.5 版本发布倒计时**：备受期待的 **Mojo 24.5** 版本预计将在一周内发布。近期社区会议讨论了关于条件式 Trait 一致性（conditional trait conformance）导致的用户困惑。成员们特别渴望解决复杂系统中与接口相关的清晰度和可见性问题。
   - 讨论强调了在产品时间线上进行更好沟通的必要性，以防止误解并确保用户为变化做好充分准备。
- **对 Mojo 复制行为的担忧**：成员们对 **Mojo** 的隐式复制行为表示担忧，特别是在使用 `owned` 参数约定时，这可能导致大型数据结构发生计划外的复制。对于从 **Python** 等语言转过来的用户，正在考虑将显式复制设置为默认选项的建议。
   - 这引发了关于不同编程语言如何管理复制的进一步辩论，用户主张在数据处理方式上应更加透明。
- **所有权语义引发困惑**：Mojo 中的所有权语义引发了讨论，因为隐式复制可能导致函数行为发生不可预测的变化，被描述为“幽灵般的远距离作用”（spooky action at a distance）。用户呼吁在 API 变更中提供更好的清晰度，并对 `ExplicitlyCopyable` Trait 进行更严格的规定，以防止诸如双重释放（double frees）等意外问题。
   - 几位成员强调了文档和社区指南的重要性，以帮助开发者更有效地应对这些复杂性。
- **Mojodojo.dev 受到关注**：社区重点介绍了由 Jack Clayton 最初创建的开源项目 **Mojodojo.dev**，认为它是 Mojo 的重要教育资源。成员们表达了增强该平台的愿望，并受邀贡献以 Mojo 构建的项目为中心的内容。
   - Caroline Frasca 强调了扩展博客和 YouTube 频道内容的重要性，以更好地展示 Mojo 开发者的可用项目和资源。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Mistral 的 Pixtral 模型亮相**：Mistral 推出了 **Pixtral 12b**，这是一款视觉多模态模型，拥有 **220 亿参数**，并针对单 GPU 运行进行了优化，尽管其 **4K 上下文大小** 较为有限。
   - 预计 11 月将推出完整的长上下文模型，提升了人们对多模态处理即将推出的功能的期待。
- **Gemma 2 表现优于 Llama 3.1**：**Gemma 2** 在多语言任务中始终优于 **Llama 3.1**，尤其是在瑞典语和韩语等语言方面表现出色。
   - 尽管关注点多在 **Llama 3**，但用户已经认可了 Gemma 2 在高级语言任务中的优势。
- **小数据集的训练效率**：用户发现，在模型优化过程中，规模较小且多样化的数据集能显著降低训练损失（training loss）。
   - 他们强调**质量胜过数量**，并指出当数据集经过良好策划且同质化程度较低时，结果会有所改善。
- **Unsloth 支持 Flash Attention 2**：成员们正在将 **Flash Attention 2** 与 **Gemma 2** 集成，但已注意到遇到了一些兼容性问题。
   - 尽管面临挑战，但大家乐观地认为最终的调整将解决冲突并提升性能值。
- **在 phi-3.5 上使用 LoRa 的微调挑战**：一位用户报告称，在 **phi-3.5** 模型上应用 **LoRa** 时，损失下降停滞，最初从 **1 降至 0.4** 后不再变化。
   - 鉴于微调 phi 模型的复杂性，建议包括尝试不同的 alpha 值以进一步优化性能。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **SWE-bench 显示了 GPT-4 相对于 GPT-3.5 的卓越实力**：SWE-bench 的表现表明 **GPT-4** 显著优于 **GPT-3.5**，尤其是在 15 分钟以内的任务中，标志着效率的提升。
   - 然而，由于缺乏人类基准数据，使得将这些结果与人类工程师进行对比评估变得复杂。
- **GameNGEN 通过实时模拟突破界限**：**GameNGEN** 令人印象深刻地实时模拟了游戏 DOOM，为世界建模（world modeling）应用开辟了道路。
   - 尽管取得了进步，它仍然依赖于现有的游戏机制，这引发了关于 3D 环境原创性的疑问。
- **GPT-4o 在基准测试中胜过 GPT-3.5**：在处理 SWE-bench 框架中的简单任务时，GPT-4o 的表现比 GPT-3.5 提升了 **11 倍**。
   - 尽管如此，这两个模型在超过四小时的任务上都表现不佳，揭示了它们解决问题能力的局限性。
- **AI 在软件工程协作中面临挑战**：关于 **AI 与软件工程师集成**以完成基准测试任务的讨论日益增多，反映出人们对此兴趣激增。
   - 虽然 AI 充满前景，但它仍缺乏资深人类工程师那种细致入微的洞察力和效能。
- **GAIA 基准测试重新定义了 AI 难度标准**：**GAIA 基准测试**对 AI 系统进行了严格测试，同时允许人类在挑战性任务中获得 80-90% 的分数，这与传统基准测试有显著区别。
   - 这表明需要重新进行评估，因为许多现有基准测试甚至对于熟练的从业者来说也变得越来越难以应对。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DeepSeek 2.5 结合 238B MoE 的优势**：[DeepSeek 2.5](https://huggingface.co/collections/deepseek-ai/deepseek-v25-66d97550c81167fc5e5e32e6) 的发布集成了 **DeepSeek 2 Chat** 和 **Coder 2** 的特性，拥有 **238B MoE** 模型、**128k 上下文长度**以及新的编码功能。
   - **Function calling** 和 **FIM completion** 为聊天和编码任务提供了开创性的新标准。
- **AI 变革医疗保健**：AI 通过增强**诊断**、实现**个性化医疗**以及加速**药物研发**，改变了**医疗保健**行业。
   - 集成**可穿戴设备**和 IoT 健康监测有助于疾病的早期发现。
- **韩语词干提取器寻求 AI 助力**：一位成员开发了一个韩语词干提取器（lemmatizer），并正在寻求利用 AI 解决词义歧义的方法。
   - 他们表达了对 **2024** 年生态系统进步以获得更好解决方案的希望。
- **CSV 仅为图像加载提供 ID**：在关于图像加载的讨论中，有人指出 CSV 文件仅包含图像 ID，因此需要获取图像或预先将它们拆分到目录中。
   - 与从组织好的文件夹创建 DataLoader 对象相比，这种方法可能会略微增加延迟。
- **Multi-agent 系统提升性能**：Transformers 现在支持 [Multi-agent 系统](https://x.com/AymericRoucher/status/1831373699670315257)，允许 Agent 在任务上进行协作，从而提高基准测试中的整体效能。
   - 这种协作方法允许对子任务进行专门化处理，从而提高效率。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **优化 Aider 的工作流**：用户分享了 Aider 的 `ask first, code later`（先询问，后编码）工作流如何通过使用 plan 模型来增强代码实现的清晰度。
   - 这种方法改进了上下文并减少了对 `/undo` 命令的依赖。
- **Prompt Caching 的优势**：Aider 的 Prompt Caching 功能通过对关键文件的策略性缓存，显示出可减少 **40%** 的 token 使用量。
   - 该系统保留了系统提示词（system prompts）等元素，有助于在交互过程中最大限度地降低成本。
- **Aider 与其他工具的对比**：用户将 Aider 与 Cursor 和 OpenRouter 等其他工具进行了对比，强调了 Aider 提升生产力的独特功能。
   - 智能功能（如从 zsh 历史记录自动生成别名和速查表）凸显了 Aider 的能力。
- **探索 OpenRouter 的优势**：成员们指出了使用 OpenRouter 优于直接使用 Claude API 的优势，强调了成本降低和绕过初始速率限制（rate limits）。
   - OpenRouter 促进了对多个模型的集中实验，使其成为首选。
- **Mistral 发布 Pixtral 模型种子**：Mistral 以种子（torrent）形式发布了 **Pixtral (12B)** 多模态模型，适用于图像分类和文本生成。
   - 可通过磁力链接 `magnet:?xt=urn:btih:7278e625de2b1da598b23954c13933047126238a` 下载，并支持 PyTorch 和 TensorFlow 等框架。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AI 图像中一致性是关键**：用户正在探索在 AI 生成的图像中保持**角色一致性 (character consistency)** 的技术，即使在更换服装或背景时也是如此。
   - 目标是确保**角色的面部特征和身体**在不同的分镜中保持可辨识性。
- **Token 处理中的 GPU 对决**：关于 Token 处理的讨论显示，一位用户在 **6900XT** 上达到了 **45 tokens/s**，凸显了不同 GPU 型号之间的差异。
   - 几位成员建议通过刷 BIOS 来提升性能，同时也对意料之外的结果表示沮丧。
- **LM Studio 爱好者聚会**：LM Studio 用户正在伦敦组织一场**聚会**，重点讨论 Prompt Engineering，并向所有用户开放讨论。
   - 鼓励参与者寻找带笔记本电脑的非学生群体进行高效交流。
- **聚焦 RTX 4090D**：讨论集中在 **RTX 4090D** 上，这是一款中国特供的 GPU，其特点是与同类产品相比拥有更多 VRAM，但 CUDA 核心较少。
   - 尽管游戏性能较低，但由于其显存容量，它可能是 AI 工作负载的战略性选择。
- **Surface Studio Pro：升级的挫败感**：用户对 **Surface Studio Pro** 有限的升级选项表示沮丧，并讨论了诸如 **eGPU** 或 SSD 等增强方案。
   - 建议包括投资一台专用的 AI 设备，而不是升级笔记本电脑。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 模型展开对决**：用户展示了旧模型（如 '1.5 ema only'）与新选项之间的性能差异，强调了图像生成质量的进步。
   - 社区指出，在 AI 任务中 **RTX 4060 Ti** 的表现优于 **7600** 和 **Quadro P620**，强调了 GPU 选择的重要性。
- **分辨率在图像生成中至关重要**：建议早期模型的最佳生成分辨率为 **512x512**，以减少放大时的伪影。
   - 用户分享了有效的工作流，建议从较低分辨率开始可以提高最终输出的质量。
- **AI 模型及其相似性**：由于共享训练数据和技术影响了原创性，人们对各种 LLM 的相似性表示担忧。
   - 然而，一些人指出新模型在生成逼真的手部等方面有了显著改进，显示出可喜的进展。
- **AI 训练的 GPU 摊牌**：社区成员争论 NVIDIA 的 GPU 是 AI 模型训练的首选，主要是因为 **CUDA** 的兼容性。
   - 共识倾向于选择具有 **20GB** VRAM 的高端 GPU 以获得卓越性能，即使低显存选项可以运行特定模型。
- **Reflection LLM 受到审查**：被吹捧具有“思考”和“反思”能力的 Reflection LLM，在实际表现与宣传不符方面面临批评。
   - API 与开源版本之间的差异引发了用户对其有效性的怀疑。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Novita 端点遭遇故障**：所有 **Novita 端点** 面临停机，导致过滤请求出现 **403 状态错误** 且没有备选方案。
   - 问题解决后，所有用户恢复了正常功能。
- **编程工具建议引发讨论**：一位用户探索将 **AWS Bedrock** 与 **Litelm** 结合进行速率管理，引发了用户对 **Aider** 和 **Cursor** 等其他工具的建议。
   - 关于这些工具有效性的意见各不相同，引发了关于用户体验和功能的激烈辩论。
- **关于 Hermes 模型定价的推测**：用户对 **Hermes 3** 是否会保持免费表示不确定，预计更新后的端点可能会收取 **$5/M** 的费用。
   - 这引发了关于预期性能提升的讨论，同时也提到了现有的免费替代方案可能仍然可用。
- **洞察 Pixtral 模型的能力**：**Pixtral 12B** 可能主要接受图像输入以产生文本输出，这表明其文本处理能力有限。
   - 该模型的表现预计与 **LLaVA** 相似，侧重于专门的图像任务。
- **OpenRouter 与 Cursor 集成的挑战**：一些用户在将 **OpenRouter** 与 **Cursor** 配合使用时遇到障碍，涉及激活模型功能所需的配置调整。
   - 贡献者指出了 Cursor 仓库中存在的问题，特别是与特定模型内的硬编码路由相关的问题。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **使用 cuDNN 优化 Matmul**：成员们讨论了各种 **matmul 算法**（如 Grouped GEMM 和 Split K）的资源，并建议查看 **Cutlass 示例**。重点仍然是利用现有的优化技术，在机器学习中实现高效的矩阵运算。
- **神经网络量化挑战**：一位成员正在重新实现 **Post-Training Quantization**，并在激活量化过程中面临准确率下降的问题，并在 [torch forum](https://discuss.pytorch.org/t/significant-accuracy-drop-after-custom-activation-quantization-seeking-debugging-suggestions/209396) 上分享了见解。社区提供了建议，强调了调试对于保持量化模型准确率的重要性。
- **Multi-GPU 使用的令人兴奋的进展**：分享了关于 **Multi-GPU** 增强的创新想法，旨在延长上下文长度并提高内存效率，并附带了[详细信息](https://docs.google.com/document/d/1YuCvBeMD5wlwI0iAV1xf3aokf4tj53epLNyRFeUuf1U/edit)。鼓励参与者追求在优化资源利用的同时最小化开销的项目。
- **OpenAI RSU 和市场洞察**：OpenAI 员工讨论了如果不出售，RSU 将增值到 **6-7x**，并分享了允许套现的二级交易的复杂性，以及对未来 IPO 的影响。关于这些二级交易对股票定价和估值影响的推测，揭示了对风险投资谈判的见解。
- **FP6 已添加到主 API**：宣布将 **fp6** 添加到项目的 README 主文件中，引发了关于与 **BF16** 和 **FP16** 集成挑战的讨论。公认需要为用户提供清晰的说明，以确保在不同精度类型之间进行高效的性能管理。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 经历重大人员离职**：OpenAI 遭遇重大人才流失，[Alex Conneau](https://x.com/alex_conneau/status/1833535309902189015?s=46) 宣布离职创业，而 [Arvind](https://x.com/arvind_io/status/1833571886766399773?s=46) 分享了加入 Meta 的兴奋之情。讨论暗示对 **GPT-5** 的提及可能预示着即将推出的模型，但对这些推测仍持怀疑态度。
- **Meta 的巨型 AI 超级计算集群**：Meta 即将完成一个拥有 **100,000 个 GPU Nvidia H100** 的 AI 超级计算集群，用于训练 **Llama 4**，且未选择 Nvidia 的专有网络设备。这一大胆举措凸显了 Meta 对 AI 的承诺，尤其是在行业竞争加剧的情况下。
- **Adobe 的生成式视频举措**：Adobe 即将推出其 **Firefly Video Model**，这标志着自 2023 年 3 月推出以来的重大进展，并计划将其集成到 Creative Cloud 功能中。今年晚些时候的 Beta 版可用性展示了 Adobe 对生成式 AI 驱动的视频制作的关注。
- **Pixtral 模型超越竞争对手**：在 Mistral 峰会上，据报道 **Pixtral 12B** 在图像尺寸灵活性和任务性能方面优于 **Phi 3** 和 **Claude Haiku** 等模型。活动期间的现场演示展示了 **Pixtral 强大的 OCR 能力**，引发了关于其与竞争对手相比准确性的辩论。
- **Surge AI 的合同挑战**：据报道，Surge AI 未能向 **HF** 和 **Ai2** 交付数据，直到面临潜在的法律诉讼，这引发了对其在小型合同上可靠性的警惕。担忧集中在他们在延迟期间缺乏沟通，让人对其优先级产生怀疑。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 注册活动进入最后阶段**：各校区仅剩 **5 天时间** 来争取 **500 个注册量**，以解锁一年的免费 **Perplexity Pro**。请访问 [perplexity.ai/backtoschool](https://perplexity.ai/backtoschool) 参与活动！
   - 更新后的倒计时器显示为 **05:12:11:10**，进一步强化了这一行动号召——*这是最后的冲刺！*
- **学生在 Perplexity 优惠活动中面临差异**：虽然提供了免费一个月的 **Perplexity Pro** 学生优惠，但仅限于美国学生或达到足够注册量的特定校区。
   - 德国等其他国家的学生也表达了对不公平待遇的担忧，他们同样希望获得促销优惠。
- **对新 API 功能的期待升温**：用户对即将到来的开发者日（dev day）发布的新 **API** 功能充满期待，特别是 **4o 语音和图像生成** 功能。
   - 此外，还有关于为需求低于全额 Pro 权限的用户创建 **hobby tier**（兴趣层级）的讨论。
- **Neuralink 分享患者更新与 SpaceX 的雄心壮志**：Perplexity AI 推广了一段 **YouTube 视频**，详细介绍了 *Neuralink 的首位患者更新* 以及 **SpaceX** 在 2026 年前往火星的目标。
   - 该视频深入探讨了这两个项目及其对未来的宏伟目标。
- **Bounce.ai 就 API 问题发出紧急支持请求**：**Bounce.ai** 的 CTO Aki Yu 报告了一个影响超过 **3,000 名活跃用户** 的 **Perplexity API** 紧急问题，强调需要立即协助。
   - 尽管已尝试联系 **4 个月**，Bounce.ai 仍未收到 **Perplexity 团队** 的回复，这凸显了支持渠道可能存在的局限性。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Llama-3.1-SuperNova-Lite 在数学方面表现出色**：成员们注意到，与 Hermes-3-Llama-3.1-8B 相比，[Llama-3.1-SuperNova-Lite](https://huggingface.co/arcee-ai/Llama-3.1-SuperNova-Lite) 在处理吠陀乘法（Vedic multiplication）等计算时表现更优，且能保持准确性。
   - 尽管两个模型都面临挑战，但 SuperNova-Lite 在保持数字完整性方面表现明显更好。
- **模型对比揭示性能差距**：测试显示 **LLaMa-3.1-8B-Instruct** 在数学任务中表现挣扎，而 **Llama-3.1-SuperNova-Lite** 取得了更好的结果。
   - 社区对 Hermes-3-Llama-3.1-8B 的偏好显现，突显了它们在算术能力上的差异。
- **高质量数据增强性能**：讨论中的反馈强调，随着参数规模（parameters）的扩大，更高质量的数据能显著提升模型性能。
   - 这强调了使用**高质量数据集**对于实现 **LLM** 最佳效果的重要性。
- **更好的选择：用于简单任务的小型模型**：一位成员询问是否有比 Llama 3.1 8B 更小、适用于基础任务的模型，并提到了 **Mistral 7B** 和 **Qwen2 7B** 作为潜在选项。
   - 讨论引发了对 3B 参数以下模型更新列表的需求，表明了社区对效率的关注。
- **渴望空间推理创新的更新**：人们对**空间推理（Spatial Reasoning）**及其相关领域是否取得了任何**革命性进展**感到好奇。
   - 成员们积极寻求有关最新创新的见解，这些创新可能会重塑对 AI 推理能力的理解。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Mistral 展示 Pixtral 12B 模型**：在一次受邀参加的会议上，Mistral 发布了 **Pixtral 12B 模型**。据 [Mistral AI](https://x.com/MistralAI/status/1833758285167722836) 指出，该模型的表现优于 **Phi 3** 和 **Claude Haiku** 等竞争对手。
   - 该模型支持任意图像尺寸和 interleaving，在有 **Jensen Huang** 出席的活动中展示了其取得的显著基准测试成绩。
- **Klarna 与 SaaS 供应商断绝关系**：Klarna 的 CEO 宣布公司正在裁撤其 **SaaS 供应商**，包括那些曾被认为不可替代的供应商，这引发了关于潜在运营风险的讨论，详见 [Tyler Hogge](https://x.com/thogge/status/1833627582551757143?s=46) 的报道。
   - 与此同时，据报道 Klarna 裁员 **50%**，这一决定可能受财务挑战驱动。
- **Jina AI 发布 HTML 转 Markdown 模型**：Jina AI 推出了两个语言模型 **reader-lm-0.5b** 和 **reader-lm-1.5b**，专门为高效将 HTML 转换为 Markdown 而优化，提供多语言支持和强劲性能 [点击阅读更多](https://x.com/JinaAI_/status/1833861180445860168?s=46)。
   - 这些模型在保持极小体积的同时，性能超越了更大的模型，简化了可访问内容的转换流程。
- **Trieve 获得融资增长**：Trieve AI 成功获得了由 Root Ventures 领投的 **350 万美元融资**，旨在简化各行业的 AI 应用部署，正如 Vaibhav Srivastav 在[此处](https://x.com/skeptrune/status/1833954889904652737?s=46)分享的那样。
   - 凭借新资金，Trieve 现有系统目前每天为数万名用户提供服务，显示出强劲的市场兴趣。
- **Hume 发布 Empathic Voice Interface 2**：Hume AI 推出了 **Empathic Voice Interface 2 (EVI 2)**，将语言和语音融合以增强情感智能应用 [点击查看](https://x.com/hume_ai/status/1833906262351974483?s=46)。
   - 该模型现已面向渴望创建需要深层情感交互应用的开发者开放。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **在 Open Interpreter 中使用自定义 Python 代码**：一位用户询问如何在 **Open Interpreter** 中利用特定的 **Python 代码**执行情感分析任务，引发了对数据库进行更广泛自定义查询的兴趣。
   - 社区渴望确认在终端应用中使用各种 **Python** 库（如用于格式化的 [rich](https://github.com/Textualize/rich)）的可行性。
- **文档改进引发关注**：反馈指出，虽然用户觉得 **Open Interpreter** 很有吸引力，但文档缺乏组织，阻碍了查阅。
   - 有人提议通过协作努力来增强文档，并鼓励提交 Pull Requests 进行改进。
- **桌面应用早期访问临近**：用户渴望了解即将推出的**桌面应用**早期访问的时间线，该应用旨在简化安装过程。
   - 社区预计在未来几周内会有更多的 Beta 测试人员加入，旨在提升用户体验。
- **关于 01 Light 的退款和转型**：围绕已停产的 **01 light** 的退款问题爆发了讨论，导致一条泄露的推文确认将转向新的**免费 01 app**。
   - 制造材料的开源也在议程之中，同时配合 **01.1 update** 进行进一步开发。
- **突出显示来自 JSONL 数据的 RAG 上下文**：初步测试运行显示，为 **RAG** 设计的 **JSONL 数据**提供上下文具有前景，主要集中在新闻 RSS 订阅源。
   - 在完成 **NER** 流程和数据加载到 **Neo4j** 后，将开始编写教程，以增强 AI 应用的可用性。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 的工单支持集成**：一位成员计划将 [Cohere](https://cohere.com) 与 Link Safe 集成，用于**工单支持**和**文本处理**，并对此次合作表示期待。
   - *我迫不及待想看到这将如何增强我们目前的工作流程！*
- **Mistral 发布视觉模型**：**Mistral** 推出了一个新的视觉模型，引发了对其功能和即将开展项目的兴趣。
   - 成员们推测 C4AI 推出视觉模型的可能性，并将其与需要更多时间开发的 *Maya* 项目联系起来。
- **人类监督的长期需求**：成员们一致认为，在 AI 的进步过程中，**人类监督**仍然至关重要，主张采用可靠的方法而非单纯追求机器智能。
   - *让我们专注于使现有的东西变得可靠*，而不是追求理论上的能力。
- **Discord FAQ 机器人初具规模**：正在努力为 Cohere 创建一个 **Discord FAQ 机器人**，以简化社区内的沟通。
   - 讨论还开启了举办虚拟黑客松活动的可能性，鼓励创新想法。
- **询问 Aya-101 的状态**：*Aya-101 是否已达到生命周期终点（End-of-life）？* 这一问题引发了关于向性能可能超越竞争对手的新模型过渡的猜测。
   - 一位成员将其称为潜在的 *Phi-killer*，引起了大家的好奇。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **lm-evaluation-harness 指导请求**：一位用户寻求使用 **lm-evaluation-harness** 在 **swe-bench** 数据集上评估 OpenAI **gpt4o** 模型的帮助。
   - 他们欢迎任何指导，并表示实践建议能显著帮助他们的评估过程。
- **Pixtral 模型发布**：社区分享了新发布的 **Pixtral-12b-240910** 模型权重（checkpoint），暗示其部分与 Mistral AI 最近的更新保持一致。
   - 用户可以在发布说明中找到下载详情和磁力链接，以及指向 [Mistral Twitter](https://x.com/MistralAI/status/1833758285167722836) 的链接。
- **RWKV-7 展现潜力**：**RWKV-7** 被视为潜在的 **Transformer 杀手**，其特点是具有源自 **DeltaNet** 的恒等加低秩（identity-plus-low-rank）转移矩阵。
   - [arXiv](https://arxiv.org/abs/2406.06484) 上展示了一项关于优化序列长度并行化的相关研究，增强了该模型的吸引力。
- **多节点训练的陷阱**：一位用户对慢速以太网链路下的**多节点训练**表示担忧，特别是关于 **8xH100** 机器之间的 **DDP** 性能。
   - 讨论表明训练可能会受到速度限制，并且跨节点使用 **DDP** 的效率可能低于预期。
- **数据集分块实践**：一位成员询问将数据集拆分为 **128-token 块** 是否为标准做法，暗示这一决定往往源于直觉而非实证研究。
   - 回复指出许多从业者可能会忽视分块对模型性能的潜在影响，凸显了理解上的空白。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LLM 时代的 RAG Maven 课程**：查看名为 *Search For RAG in the LLM era* 的 [Maven 课程](https://twitter.com/llama_index/status/1833584685664067833)，包含客座讲座和现场代码演示。
   - 参与者可以与行业资深人士一起练习代码示例，以增强学习体验。
- **构建 RAG 的快速教程**：现已提供使用 LlamaIndex [构建检索增强生成（RAG）的简明教程](https://twitter.com/llama_index/status/1833611545370366281)。
   - 该教程专注于有效地实现 RAG 技术。
- **Kotaemon：构建基于 RAG 的文档问答系统**：学习使用 [Kotaemon](https://twitter.com/llama_index/status/1833907464355647906) 构建*基于 RAG 的文档问答系统*，这是一个用于与文档聊天的开源 UI。
   - 本次会议涵盖了可定制 RAG UI 的设置以及如何组织 **LLM 和 embedding 模型**。
- **AI 调度器动手研讨会**：参加 9 月 20 日在 AWS Loft 举行的研讨会，使用 **Zoom**、**LlamaIndex** 和 **Qdrant** *构建用于智能会议的 AI 调度器*。
   - 参与者将使用 **Zoom 的转录 SDK** 创建一个专注于会议效率的 RAG 推荐引擎。
- **探索用于索引的任务队列设置**：发起了一场关于使用 **FastAPI** 和 **Celery 后端** 创建构建索引任务队列的讨论，重点关注文件和索引信息的数据库存储。
   - 鼓励参与者查看可能满足这些要求的现有设置。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **查询生成的 POC 开发**：一位成员正在使用 **LangGraph** 开发用于 **查询生成** 的 **POC**，面临着随着表数量增加而导致的 Token 大小增加的挑战。
   - 他们正在利用 **RAG** 创建 Schema 的向量表示以进行查询构建，并对增加更多 LLM 调用感到犹豫。
- **OppyDev 重大更新发布**：**OppyDev** 团队宣布了一项重大更新，增强了该 AI 辅助编码工具在 **Mac 和 Windows** 上的可用性，并支持 **GPT-4** 和 **Llama**。
   - 用户可以通过限时促销代码获得 **一百万免费 GPT-4 Token**；详情可按需获取。
- **构建 RAG 应用的见解**：关于在存入向量数据库之前，是否保留 **RAG** 应用中通过 Web Loader 检索到的文本中的 **换行符** 引起了讨论。
   - 确认保留 **换行符** 是可以接受的，这能确保文本格式保持完整。
- **OppyDev 中的实时代码审查功能**：最新的 **OppyDev** 更新包括一个 **带颜色编码的可编辑 Diff** 功能，用于实时代码更改监控。
   - 此次升级显著增强了开发人员有效跟踪和管理代码修改的能力。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 缺乏 FP16 支持**：一位成员指出 **Torchtune** 不支持 **FP16**，需要额外的工作来保持与混合精度模块的兼容性，而 **bf16** 被认为是更优的替代方案。
   - 缺乏支持可能会给使用旧型号 GPU 的用户带来问题。
- **Qwen2 接口分词怪癖**：**Qwen2** 接口允许 `eos_id` 为 `None`，这导致在 `encode` 方法中添加它之前需要进行检查，引发了对其意图的疑问。
   - 由于代码的另一部分没有执行此检查，指示存在疏忽，从而产生了潜在的 Bug。
- **None EOS ID 处理问题**：关于在 `eos_id` 设置为 `None` 时允许 `add_eos=True` 的担忧被提出，这暗示了 **Qwen2** 模型分词过程中行为不一致。
   - 这种不一致可能会困扰用户并破坏预期功能。
- **关于 padded_collate 功效的疑问**：一位成员质疑 **padded_collate** 的实用性，指出它在任何地方都没有被使用，同时指出了关于 **input_ids 和 labels** 序列长度缺失逻辑的问题。
   - 这引发了后续询问，即 **padded_collate** 逻辑是否已正确合并到 **PPO Recipe** 中。
- **PPO Recipe 需要澄清**：围绕 **PPO Recipe** 中的 `padded_collate` 逻辑是否完整展开了讨论，一位成员表示他们已经集成了一部分。
   - 这进一步引发了关于 **input_ids** 和 **labels** 之间长度通常匹配的讨论。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Sci Scope 发布用于 Arxiv 洞察**：**Sci Scope** 是一个新工具，使用 LLM 对最新的 Arxiv 论文进行分类和总结，可在 [Sci Scope](https://www.sci-scope.com/) 免费使用。用户可以订阅 AI 研究的 **每周摘要**，增强对文献的关注。
   - 讨论了如何确保输出的 **真实性** 并减少摘要中的幻觉，反映了对 AI 生成内容可靠性的担忧。
- **为客户需求定制 DSPy**：一位成员询问如何将 **针对特定客户的定制** 集成到 DSPy 为聊天机器人生成的提示词中，希望避免硬编码客户数据。他们考虑使用 **后处理步骤** 进行动态适配，并征求关于更好实现策略的反馈。
   - 这种交流体现了小组内的协作精神，成员们通过分享见解和解决方案积极互相支持。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **在 tinygrad 中探索音频模型**：一位用户寻求关于如何使用 **tinygrad** 运行音频模型的指导，特别是希望了解仓库中现有的 **Whisper** 示例之外的内容。
   - 这一询问引发了关于在 tinygrad 中探索音频应用潜在切入点的建议。
- **学习的哲学方法**：一位成员引用了 *“千里之行，始于足下”*，强调了直觉在学习过程中的重要性。
   - 这种观点鼓励了对社区内资源的深思熟虑式探索。
- **链接到有用的资源**：另一位成员分享了 Eric S. Raymond 编写的 [smart questions](http://www.catb.org/~esr/faqs/smart-questions.html) FAQ 链接，概述了在线寻求帮助的礼仪和策略。
   - 该资源可作为撰写有效查询并最大限度获得社区帮助的指南。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Mistral 的 Pixtral 奠定多模态基础**：**Mistral 的 Pixtral** 正在推进 **multi-modal support**（多模态支持）的工作，呼应了 AI 能力的最新发展。
   - *考虑到当今的技术进步，这是一个具有先见之明的举动*。
- **Axolotl 项目获得新的消息结构**：**Axolotl** 项目中关于新消息结构的 pull request 旨在增强消息的表示方式，以支持改进的功能。
   - 欲了解更多信息，请参阅 [New Proposed Message Structure](https://github.com/axolotl-ai-cloud/axolotl/pull/1904) 的详细信息。
- **LLM 模型进行速度与性能测试**：最近的一段 [YouTube 视频](https://youtu.be/w6CJtAlGygQ?si=0MzkKj5m2MUiSN59) 评估了截至 2024 年 9 月领先的 LLM 模型在速度和性能方面的表现，重点关注 **tokens per second**。
   - 测试强调了 **latency**（延迟）和 **throughput**（吞吐量），这是生产环境中任何性能评估的关键指标。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI 开发者为 NYX 模型寻找合作伙伴**：一位 AI 开发者宣布正在开发拥有超过 **6000 亿参数**的 **NYX 模型**，并正在积极寻找合作伙伴。
   - *如果你具备 AI 方面的专业知识且时区相近以便有效协作，让我们聊聊吧！*
- **关于训练大模型的咨询**：一位开发者询问了用于训练 **600B 参数模型**的训练资源，并提到了在 **15 万亿 tokens** 上训练的 **LLaMA-405B**。
   - 好奇心集中在此类超大模型的数据来源方法论上，表明了对底层流程的浓厚兴趣。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Literal AI 在易用性方面表现出色**：用户称赞 [Literal AI](https://literalai.com/) 具有直观的界面，增强了 LLM 应用的可访问性和用户体验。
   - 这反映了在 LLM 技术竞争日益激烈的格局中，对**用户友好型工具**日益增长的需求。
- **可观测性提升 LLM 生命周期健康度**：强调了 LLM 可观测性的重要性，因为它使开发者能够快速迭代并有效地处理调试过程。
   - 利用日志可以提升较小模型的性能，同时降低成本，推动高效的模型管理。
- **监控 prompt 防止回归**：在部署新版本的 prompt 之前，持续跟踪 prompt 的表现对于防止回归（regressions）至关重要。
   - 这种主动评估保护了 LLM 应用免受潜在故障的影响，并增加了部署信心。
- **LLM 监控确保生产环境可靠性**：强大的日志记录和评估机制对于监控生产环境中的 LLM 性能至关重要。
   - 实施有效的分析使团队能够保持监督并增强应用的稳定性。
- **集成 Literal AI 非常简便**：Literal AI 支持跨应用的轻松集成，允许用户接入完整的 LLM 生态系统。
   - 提供自托管选项，以满足欧盟用户和管理敏感数据的用户需求。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Ground Truth Data 在 AI 中的关键作用**：一篇新博客文章强调了 **Ground Truth Data 的重要性**，它能提升 AI 应用中的 **模型准确性 (model accuracy)** 和可靠性，并敦促读者参与正在进行的讨论 [加入讨论](https://discord.com/channels/1089876418936180786/1283463258635898922)。
   - Ground Truth Data 被认为是推动 AI 系统在不同语境下性能提升的核心要素。
- **Mozilla 开启校友资助申请**：Mozilla 邀请之前的 Mozilla Fellowship 参与者申请针对 **可信 AI (trustworthy AI)** 和更健康互联网倡议的 [项目资助](https://foundation.mozilla.org/en/blog/mozilla-opens-call-for-alumni-connection-grant-applications/)，这反映了在 AI 领域进行 **结构性变革 (structural changes)** 的努力。
   - *“互联网，尤其是人工智能 (AI)，正处于一个转折点。”* Hailey Froese 的这句话强调了在该领域进行变革性努力的号召。



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **评估脚本错误困扰用户**：用户在运行 `openfunctions_evaluation.py` 并带有 `--test-category=non_live` 参数时遇到了 **'No Scores'** 问题，指定文件夹中没有生成结果。
   - *尝试使用新的 API 凭据重新运行* 未能成功，导致了进一步的复杂情况。
- **API 凭据已更新但问题依旧**：在设置中，用户在 `function_credential_config.json` 中添加了四个新的 API 地址，希望能解决问题。
   - 尽管进行了这些更改，评估期间仍然出现错误，证实了凭据更新无效。
- **Urban Dictionary API 超时问题**：在评估过程中，出现了与 Urban Dictionary API 相关的 **连接错误 (Connection Error)**（涉及术语 'lit'），表明存在超时问题。
   - *怀疑网络问题* 是用户面临连接困难的根源。



---


**Alignment Lab AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1283158447403307048)** (336 条消息🔥🔥): 

> - `Mojo 用户反馈`
> - `周边 (Swag) 讨论`
> - `Mojo 24.5 发布`
> - `Trait 一致性与接口 (Interfaces)`
> - `Go 接口 (Interfaces)` 


- **用户反馈机会**：目前正在寻找尚未与 **Magic** 互动的用户，通过 30 分钟的通话提供反馈，并提供酷炫的独家周边 (Swag) 作为奖励。感兴趣的人可以[在此预约](https://modul.ar/user-feedback)。
   - 团队对未来更广泛地获取周边或开设潜在周边商店的询问给予了积极回应。
- **对 Mojo 24.5 发布的期待**：有用户询问 **Mojo 24.5** 的预期发布日期，回复建议它可能会在一周内发布。还讨论了近期社区会议中关于条件 Trait 一致性的细节，这反映了用户的困惑。
   - 探讨了 Trait 一致性的运作方式及其影响，成员们对复杂系统中接口的清晰度和可见性表示担忧。
- **关于 Trait 一致性与接口的讨论**：对话转向了不同编程语言如何处理接口实现，特别是关注 **Go** 及其接口。有人担心某些接口在大型组织中不够清晰，或会导致意外后果。
   - 考察了 **Rust** 和 **Swift** 的对比设计，特别关注了在系统编程语境下扩展接口的影响。
- **对 Go 接口的批评**：辩论了 **Go** 接口的效率和实用性，重点在于它们依赖开发者遵守特定的 API 合约。参与者对 Go 的组合模型 (Composition Model) 的平衡性和优雅性与其潜在陷阱之间的关系表达了复杂的情绪。
   - 讨论涵盖了在使用 Go 的大型代码库和组织中，由于验证过程不足而产生的问题。
- **关于组合与接口设计的总体印象**：讨论中强调了一个观点：如果一个函数依赖于组合 (Composition)，那么它就不需要关心实现细节。参与者承认程序员有责任确保接口的设计便于验证和错误处理。
   - 还强调了程序员和接口在通过正确实现防止数据损坏方面的责任，突显了关于语言设计中应嵌入多少验证逻辑的持续争论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols/#Adding-Con">文档</a>：未找到描述</li><li><a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols#Protocol-Extensions">文档</a>：未找到描述</li><li><a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols/#Adding-Constraints-to-Protocol-Extensions">文档</a>：未找到描述</li><li><a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols#Protocol-Ex">文档</a>：未找到描述</li><li><a href="https://docs.python.org/3/tutorial/classes.html#inheritance">9. 类 (Classes)</a>：类提供了一种将数据和功能捆绑在一起的方法。创建一个新类会创建一个新类型的对象，从而允许创建该类型的新实例。每个类实例可以具有...</li><li><a href="https://www.youtube.com/watch?v=OWfxexSE2aM">408 Swift 中的面向协议编程</a>：翻译 Yandex 浏览器的视频</li><li><a href="https://docs.oracle.com/en/database/oracle/oracle-database/19/admin/repairing-corrupted-data.html">数据库管理员指南</a>：您可以检测并纠正数据块损坏。</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Classes/extends">extends - JavaScript | MDN</a>：extends 关键字用于类声明或类表达式中，以创建一个作为另一个类子类的类。</li><li><a href="https://modul.ar/user-feedback">预约</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1283155331093889035)** (394 messages🔥🔥): 

> - `Mojo Copy Behavior`
> - `Ownership in Mojo`
> - `ExplicitlyCopyable Trait`
> - `Mojodojo.dev` 


- **关于 Mojo 复制行为的讨论**：几位成员对 Mojo 的隐式复制行为表示担忧，特别是在使用 `owned` 参数约定时，这可能导致大型数据结构发生意外复制。
   - 建议包括将显式复制设为默认行为，以避免意外行为，特别是对于从 Python 等语言转过来的用户。
- **对 Ownership 语义的担忧**：成员们讨论了 Mojo 中的 Ownership 语义如何产生“幽灵般的远距离作用 (spooky action at a distance)”，导致局部更改由于隐式复制而显著改变函数行为。
   - 对话强调了 API 变更需要更好的清晰度，以及可能需要围绕 `ExplicitlyCopyable` trait 制定更严格的规则，以避免无意中的 double frees。
- **关于显式复制方法的提案**：有一项提案要求 `ExplicitlyCopyable` trait 实现 `copy()` 方法，这可以提高开发者的可用性和清晰度。
   - 成员们建议内置的 `copy` 函数可能更符合 Pythonic 风格，同时也讨论了链式方法对于函数式编程风格的重要性。
- **Mojodojo.dev 介绍**：社区讨论了 Mojodojo.dev 的开源状态，这是一个最初由 Jack Clayton 创建的用于学习 Mojo 的资源。
   - 成员们表现出合作增强 Mojodojo.dev 的兴趣，强调了其作为 Mojo 生态系统早期教育资源的价值。
- **邀请贡献者**：Caroline Frasca 邀请社区成员为博客和 YouTube 频道做贡献，表达了希望看到更多围绕使用 Mojo 构建的项目的内容。
   - 一位用户对游戏代码资源表示感谢，这些资源促进了他们对 Mojo 的理解。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/deep-dive-into-ownership-in-mojo">Modular: Deep dive into ownership in Mojo</a>: 这篇博客是 Mojo Ownership 系列的第二部分。请务必查看第一部分《Ownership 的本质：心理模型方法》，因为我们将在此基础上进行构建...</li><li><a href="https://github.com/modularml/mojo/commit/a597b9009ecf743f99e01263e570a43aa6c1cfbd">[External] [stdlib] Fix soundness issues in InlinedFixedVector on cop… · modularml/mojo@a597b90</a>: …y/del (#46832) [External] [stdlib] 修复 InlinedFixedVector 在 copy/del 上的健全性问题。`InlinedFixedVector` 存在明显的 double free（在多次调用 `_del_old()` 时），这很容易意外发生...</li><li><a href="https://github.com/modularml/mojo/issues/3390">Generalize `init` argument convention with named result slots · Issue #3390 · modularml/mojo</a>: 审查 Mojo 的优先级。我已阅读路线图和优先级，并相信此请求符合优先级。你的请求是什么？最近，Mojo 以...的形式添加了命名结果槽。
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1283144865202307084)** (608 messages🔥🔥🔥): 

> - `Pixtral Model Launch`
> - `Gemma 2 vs Llama 3.1 Performance`
> - `Fine-tuning Techniques`
> - `Unsloth Features`
> - `Flash Attention 2 Issues`

- **Mistral 发布 Pixtral 模型**：Mistral 推出了一款名为 **Pixtral 12b** 的新型视觉多模态模型，该模型旨在适配单个 GPU，并拥有 220 亿个参数。
   - 虽然有些人对测试其功能感到兴奋，但也有人指出其上下文窗口（context size）相对较小，仅为 **4K**，预计 11 月将推出完整的长上下文模型。
- **Gemma 2 在多语言任务中表现出色**：用户分享的经验表明，**Gemma 2** 在多种语言（特别是瑞典语和韩语）中的表现优于 **Llama 3.1**，使其成为多语言应用的强力竞争者。
   - 尽管大家都在忙于 **Llama 3**，但 **Gemma 2** 的能力依然受到赞赏，因为许多用户看到了其先进的语言处理潜力。
- **关于微调（Fine-tuning）技术的讨论**：参与者讨论了如何优化模型，并指出在数据集条目中“**质量重于数量**”对于有效微调至关重要。
   - 有人建议通过过滤数据集来提高效率，并强调了增加 batch size 或尝试梯度累积（gradient accumulation）步骤等策略，以提高训练速度。
- **Unsloth 的功能与支持**：**Unsloth** 最近开始支持 **Flash Attention 2**，用户正尝试将其集成到使用 **Gemma 2** 的工作流中。
   - 虽然一些人遇到了问题，但社区成员表示希望最终的调整能解决兼容性问题，从而带来更好的性能。
- **Flash Attention 2 相关咨询**：用户询问了在使用 **Gemma 2** 时 **Flash Attention 2** 的最佳配置，并确认推荐使用最新版本。
   - 虽然 **Flash Attention 3** 需要更先进的硬件，但目前的共识是使用 **Flash Attention 2** 以保证兼容性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Joseph717171/Llama-3.1-SuperNova-Lite-8.0B-OQ8_0.EF32.IQ4_K-Q8_0-GGUF">Joseph717171/Llama-3.1-SuperNova-Lite-8.0B-OQ8_0.EF32.IQ4_K-Q8_0-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Linguistic_frame_of_reference">Linguistic frame of reference - Wikipedia</a>: 未找到描述</li><li><a href="https://huggingface.co/Etherll/Herplete-LLM-Llama-3.1-8b">Etherll/Herplete-LLM-Llama-3.1-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.google.com/document/d/1OjbjU5AOz4Ftn9xHQrX3oFQGhQ6RDUuXQipnQ9gn6tU/edit?usp=sharing">SHARED Continuous Finetuning By Rombodawg</a>: 使用 LoRA 和 Mergekit 进行无损连续微调。在这篇文章中，我们将讨论如何使用 LoRA 适配器和 Mergekit 对开源 AI 模型进行持续微调...</li><li><a href="https://huggingface.co/upstage/solar-pro-preview-instruct">upstage/solar-pro-preview-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored">Orenguteng/Llama-3-8B-Lexi-Uncensored · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/danielhanchen/status/1833764749538119921">来自 Daniel Han (@danielhanchen) 的推文</a>: Mistral 刚刚发布了一个名为 Pixtral 12b 的新视觉多模态模型！还下载了参数 json - 视觉适配器使用了 GeLU 和 2D RoPE。词表大小也变大了 - 131072。此外 Mist...</li><li><a href="https://huggingface.co/arcee-ai/Llama-3.1-SuperNova-Lite-GGUF">arcee-ai/Llama-3.1-SuperNova-Lite-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2405.09673">LoRA Learns Less and Forgets Less</a>: 低秩自适应 (LoRA) 是一种广泛使用的针对大语言模型的高效参数微调方法。LoRA 通过仅对选定的权重矩阵训练低秩扰动来节省显存。在本文中...</li><li><a href="https://x.com/mistralai/status/1833758285167722836?s=46">来自 Mistral AI (@MistralAI) 的推文</a>: magnet:?xt=urn:btih:7278e625de2b1da598b23954c13933047126238a&dn=pixtral-12b-240910&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fannoun...</li><li><a href="https://huggingface.co/arcee-ai/Llama-3.1-SuperNova-Lite">arcee-ai/Llama-3.1-SuperNova-Lite · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/Yuchenj_UW/status/1833627813552992722">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>: 这是关于我在 @hyperbolic_labs 上托管 Reflection 70B 的故事：9 月 3 日，Matt Shumer 联系了我们，说他想发布一个 70B 的 LLM，它应该是顶级的 OSS 模型（远超 405B）...</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-V2-Llama-3.1-8b">Replete-AI/Replete-Coder-V2-Llama-3.1-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c9dv7h/comment/l4emxvx/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/lighteval/MATH-Hard/viewer/number_theory>">lighteval/MATH-Hard · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/lighteval/MATH-Hard/viewer/number_theory">lighteval/MATH-Hard · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/)** (1 条消息): 

mahiatlinux: https://www.reddit.com/r/ChatGPT/comments/1fdphr6/blowing_out_the_candles/
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1283168993062158510)** (48 条消息🔥): 

> - `Intel Gaudi 上的 Unsloth`
> - `训练损失（Training Loss）与数据集大小`
> - `在非英语数据集上微调 LLM`
> - `视觉模型（Vision Models）支持`
> - `在 phi-3.5 上使用 LoRa` 


- **Unsloth 在 Intel Gaudi 系统上运行困难**：成员们讨论了在 Intel Gaudi 系统上运行 Unsloth 的情况，遇到了 **Torch not compiled with CUDA enabled** 的错误。
   - 一位成员指出 Unsloth 主要针对 Nvidia GPU 工作，这表明其在 Gaudi 兼容性方面存在挑战。
- **较小的数据集可提高训练效率**：一位用户分享了在使用 Unsloth 时，减小数据集规模如何改善了其训练损失（training loss）表现。
   - 讨论强调，为了获得更好的结果，应专注于更小、更多样化的数据集，而不是大规模且同质化的数据集。
- **在自定义数据集上微调 LLM 的指导**：一位新手询问了关于使用 Unsloth 在自定义非英语数据集上微调 LLM 的指导，特别是针对敏感数据的处理。
   - 资深用户建议查看相关的 YouTube 教程，以获取该主题的实用建议。
- **视觉模型微调的现状**：参与者确认像 **phi-3.5-vision** 这样的视觉模型目前无法使用 Unsloth 进行微调。
   - 大家乐观地认为，对此类模型的支持可能会在今年年底或明年年初推出。
- **在 phi-3.5 上使用 LoRa 的挑战**：一位用户报告称，在使用 LoRa 训练 phi-3.5 模型时，损失（loss）改善停滞不前，最初从 **1 降至 0.4** 后便不再变化。
   - 建议尝试不同的 alpha 值，因为微调 phi 模型可能特别具有挑战性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/Gpyukc6c0w8?si=60-tnWqbnTEnqunU">Unsloth: How to Train LLM 5x Faster and with Less Memory Usage?</a>: 🚀 与 Unsloth 一起深入 AI 模型微调的世界！在这个全面的教程中，我们探索了如何将 MRAL Jemma Llama 模型的微调速度提高多达 5 倍...</li><li><a href="https://youtu.be/rpAtVIZB72U?si=xbfosm-KVI8G0tvi">LLAMA-3.1 🦙: EASIET WAY To FINE-TUNE ON YOUR DATA 🙌</a>: 学习如何使用 Unsloth、LoRa 和 QLoRa 技术高效地微调 Llama 3.1 模型。链接：Colab: https://tinyurl.com/bdzxhy5n Unsloth: https:/...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1283152186288767006)** (466 messages🔥🔥🔥): 

> - `SWE-bench 性能`
> - `GameNGEN 能力`
> - `GPT-4o vs GPT-3.5 基准测试`
> - `软件工程中的 AI 能力`
> - `针对 AI 的 GAIA 基准测试` 


- **SWE-bench 性能指标**：观察到 GPT-4 相比 GPT-3.5 解决了显著更多的任务，特别是在 <15 分钟类别中，表明效率有所提高。
   - 尽管 GPT-4o 显示出令人期待的结果，但由于缺乏人类基准（human baseline），很难全面评估模型相对于人类工程师的表现。
- **GameNGEN 的模拟能力**：GameNGEN 因创建了一个能够实时模拟游戏《DOOM》的神经模型而受到关注，这暗示了在世界建模（world modeling）应用中的可能性。
   - 尽管它在模拟环境方面迈出了令人印象深刻的一步，但它仍然依赖于既有的游戏机制和资产，而不是开发全新的 3D 环境。
- **GPT-4o vs GPT-3.5 基准测试**：GPT-4o 在解决 SWE-bench 基准测试中较低复杂度的任务方面，表现比 GPT-3.5 提高了 11 倍。
   - 然而，这两个模型在处理耗时超过 4 小时的任务时都遇到了很大困难，表明它们在解决问题能力上存在潜在局限。
- **软件工程中的 AI 能力**：人们对了解 AI 在基准测试问题上与软件工程师协作的表现越来越感兴趣。
   - 讨论表明，虽然 AI 模型展现出潜力，但它们缺乏经验丰富的人类工程师那种细致入微的理解和效率。
- **为 AI 难度设计的 GAIA 基准测试**：GAIA 基准测试旨在挑战 AI 系统，同时对人类参与者保持可控，人类在困难任务上的得分率为 80-90%。
   - 这与传统基准测试形成鲜明对比，后者对于即使是技术娴熟的毕业生来说也变得越来越难以解决。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/airesearch12/status/1833572278103380211,">Florian S (@airesearch12) 的推文</a>: 让我告诉你关于 Reflection 和 @mattshumer_ 发生了什么的理论。这是唯一能同时解释时间线和令人费解的声誉自毁行为的解释...</li><li><a href="https://gamengen.github.io/">GameNGen</a>: 扩散模型是实时游戏引擎</li><li><a href="https://www.swebench.com/index.html">SWE-bench</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=6x-Xb_uT7ts&t=900s">[CVPR&#39;23 WAD] Keynote - Ashok Elluswamy, Tesla</a>: 在 2023 年 CVPR 自动驾驶研讨会上的演讲：https://cvpr2023.wad.vision/。00:00 介绍 02:09 占用网络（Occupancy Networks）回顾 04:04 生成式模型...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1283190159252258927)** (12 messages🔥): 

> - `Android 应用复制问题`
> - `GPT 访问错误`
> - `GPT 困惑与性能下降`
> - `聊天记忆加载问题`
> - `即将到来的 GPT-5 发布日期` 


- **Android 应用在 Markdown 处理上遇到困难**：用户报告了 Android 应用中的一个问题，即复制文本会导致**没有 Markdown 格式的纯文本**，这个问题刚刚开始出现。
   - 此外，用户对无法在聊天中切换到之前的**提示词/消息（prompts/messages）**表示沮丧。
- **GPTs 的访问问题**：一位用户在使用他们的 GPT 时遇到了访问问题，收到了错误消息：“Oops, an error occurred! Try again.”
   - 他们对发生这种情况的原因表示困惑，表明这可能是一个普遍问题。
- **GPT 表现出困惑迹象**：一位用户表示，他们的 GPT 突然变得很困惑，指出它会重复同样的错误，并且似乎意识到自己的错误。
   - 他们推测 **temperature** 设置被调整到了 0，影响了模型的性能。
- **聊天记忆无法加载**：多位用户报告称，浏览器版本的 ChatGPT 经常无法加载聊天记忆，导致无法生成回复。
   - 一位用户提到他们已经放弃了浏览器版本，转而使用应用版本。
- **关于 GPT-5 发布日期的推测**：一位用户询问了 **GPT-5** 的发布日期，另一位成员建议可能在 **2025-2026** 年左右。
   - 这一建议引发了对等待时间的**沮丧**，导致一位用户表示不敢置信。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1283159217922117673)** (17 条消息🔥): 

> - `Prompt Library 访问`
> - `使用 ChatGPT 实现 ECHO`
> - `ChatGPT 回复的多样性`
> - `Custom Instructions 的影响` 


- **寻找 Prompt Library**：一位成员询问如何访问 Prompt Library，该库现在位于特定频道 <#1019652163640762428>。
   - 另一位成员迅速提供了更新后的频道信息。
- **质疑在 ChatGPT 上实现 ECHO 的可行性**：关于在 ChatGPT 上实现 ECHO 潜力的讨论，一些人认为这可能需要 Orion 和 Strawberry 等未来模型。
   - 一位成员询问客户洞察是否能澄清这一话题。
- **ChatGPT 回复中的重复现象**：一位成员注意到，在多次重新生成（regenerations）后，他们收到了相同的笑话，这表明 ChatGPT 的输出变异性有限。
   - 这引发了关于模型一致性的幽默讨论，并对相同笑话出现的频率发表了评论。
- **Custom Instructions 与消息引导**：成员们讨论了 Custom Instructions 如何影响 ChatGPT 的回复，引导其提供更具创意的输出而非标准答案。
   - 另一位成员建议，当被要求时，模型会遵守简洁回复的请求，即使它通常倾向于提供更充实的内容。
- **ChatGPT 鼓励探索**：一位成员遇到了模型发出的幽默提示，建议在多次重新生成后休息一下或进行其他活动。
   - 这展示了在用户鼓励模型探索多样化主题时，尽管输出具有随机性，但仍面临随机性不足的挑战。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1283159217922117673)** (17 条消息🔥): 

> - `Prompt Library 位置`
> - `ECHO 与未来模型`
> - `重新生成回复`
> - `引导 GPT 输出` 


- **查找 Prompt Library 频道**：一位用户询问如何访问 Prompt Library，该频道现已重命名为 <#1019652163640762428>。
   - 一位成员迅速提供了新的频道名称以协助导航。
- **关于当前模型实现 ECHO 的辩论**：一位用户质疑 ECHO 是否可以通过 ChatGPT 实现，或者是否需要 Orion 和 Strawberry 等未来模型。
   - 另一位成员建议，使用当前的设置也可以获得客户洞察。
- **重新生成笑话的问题**：一位用户对多次重新生成回复后重复收到同一个笑话表示沮丧，指出 10 次中有 9 次是相同的。
   - 值得注意的是，其中一次重新生成产生了一个涉及奶牛变体幽默的不同回复。
- **GPT-4 的交互式笑话**：与重新生成的回复形成对比，据报道 GPT-4 会通过提问（如询问 "knock knock"）与用户互动。
   - 一位成员称赞了它的交互性，称原始 GPT-4 在生成新鲜内容方面是佼佼者。
- **鼓励 GPT 产生独特输出**：一位用户分享了从 GPT 获取独特输出的策略，即指令它尽管有之前的交互也要创造新颖的内容。
   - 他们提到他们的 Custom Instructions 会引导模型建议不同的探索方向，从而影响其随机性。


  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1283141072914219122)** (1 messages): 

> - `DeepSeek 2.5`
> - `Mini Omni`
> - `Multi-agent systems`
> - `Transformers.js v3`
> - `Reflection-Tuning` 


- **DeepSeek 2.5 融合了 238B MoE 的优势**：[DeepSeek 2.5](https://huggingface.co/collections/deepseek-ai/deepseek-v25-66d97550c81167fc5e5e32e6) 的发布结合了 **DeepSeek 2 Chat** 和 **Coder 2** 的特性，拥有一个 **238B MoE** 模型，具备 **128k context length** 和全新的编程功能。
   - 它包含 **function calling** 和 **FIM completion** 等功能，为聊天和编程任务树立了新标准。
- **Multi-agent systems 提升性能**：Transformers Agents 现在支持 [Multi-agent systems](https://x.com/AymericRoucher/status/1831373699670315257)，允许多个 Agent 协作完成任务，从而提高各项基准测试的效率。
   - 这些系统允许对子任务进行专门化处理，与传统的单 Agent 模型相比，显著提高了运行效率。
- **通过 Mini Omni 实现实时语音交互**：[Mini Omni](https://huggingface.co/gpt-omni/mini-omni) 引入了一个支持 **实时语音对话** 的模型，扩展了实时交互的能力。
   - 这一创新为对话式 AI 开辟了新途径，实现了即时且动态的交流。
- **WebGPU 助力更快速的背景移除**：一种全新的 [图像背景移除](https://x.com/xenovacom/status/1828116951186710795) 方法采用了 **WebGPU 加速**，实现了低成本且高隐私标准的浏览器内推理。
   - 正如所提到的，它提供了 **快速** 且 **高质量** 的结果，无需数据离开用户设备。
- **Reflection-Tuning 取得了令人瞩目的成果**：一个新的 [distilabel recipe](https://x.com/gabrielmbmb_/status/1832078861296668748) 展示了如何使用 **Reflection-Tuning** 生成数据集，证明了 **Reflection 70B** 模型具有竞争力的性能。
   - 该方法利用 **Llama 3.1** 指导模型生成回复，从而通过反思性思维提高输出质量。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/TheZachMueller/status/1831002292440469519)">Zach Mueller (@TheZachMueller) 的推文</a>: 今天 @huggingface accelerate 0.34.0 正式发布，这是一个内容丰富的版本！从 `torchpippy` 更新到可恢复的 dataloader 支持，以及改进的 TransformerEngine 支持，有很多内容...</li><li><a href="https://x.com/AymericRoucher/status/1831373699670315257)!">Aymeric (@AymericRoucher) 的推文</a>: 🥳 Transformers Agents 现在支持 Multi-agent systems！Multi-agent systems 最初由 Microsoft 的 Autogen 框架引入。它简单来说就是让多个 Agent 协同工作来解决...</li><li><a href="https://x.com/vllm_project/status/1833257997814096245)">vLLM (@vllm_project) 的推文</a>: 我们很高兴看到 @vllm_project 成为 @huggingface hub 本地应用的一个选项！它附带了简单的代码片段，可以快速测试模型。</li><li><a href="https://x.com/xenovacom/status/1828116951186710795)">Xenova (@xenovacom) 的推文</a>: 最近关于图像背景移除的最佳方法有很多争论。这是我的尝试：- 使用 🤗 Transformers.js 进行浏览器内推理 - WebGPU 加速（快！）- 成本 $0 ...</li><li><a href="https://x.com/multimodalart/status/1833459429557088314)">apolinario 🌐 (@multimodalart) 的推文</a>: 现在在 @huggingface 上为你的 LoRA 画廊添加图片变得非常简单 🤯 🪄 ① 使用 Widget 生成一张图片 🖼️ ② 点击 "Add to model card gallery" 🔥</li><li><a href="https://x.com/vanstriendaniel/status/1833188523207496058)">Daniel van Strien (@vanstriendaniel) 的推文</a>: @huggingface 的 Semantic Dataset Search 重新上线了！通过 ID 查找相似数据集，或对 dataset cards 进行语义搜索。快来试试：https://huggingface.co/spaces/librarian-bots/hug...</li><li><a href="https://x.com/gabrielmbmb_/status/1832078861296668748)">Gabriel Martín Blázquez (@gabrielmbmb_) 的推文</a>: 昨天 Reflection 70B 发布了，这是一个使用 Reflection-Tuning 微调的模型，在 MMLU 等多个基准测试中取得了令人印象深刻的分数。用于微调的数据集并不...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1283141593842450493)** (241 messages🔥🔥): 

> - `HuggingFace community mapping`
> - `New datasets features`
> - `SQL integration with datasets`
> - `Best AI models for different purposes`
> - `Using cloud for model training`

- **HuggingFace 社区图谱发布**：一位用户分享了 HuggingFace 社区的交互式可视化，展示了生态系统内的各种连接。
   - *Charlesddamp* 宣布了这一发布，引发了社区贡献者的兴奋和认可。
- **引入新的 datasets 功能**：用户讨论了 HuggingFace datasets 的最新功能，包括 SQL 能力和 DuckDB 集成。
   - 一些用户报告了运行 SQL 查询导致内存溢出（out-of-memory）错误的问题，引发了关于错误处理的讨论。
- **探索使用 SQL 进行数据集分析**：一位用户演示了使用 SQL 命令进行数据集查询，特别关注 HuggingFace 中的 Fineweb 数据集。
   - 讨论中提出了关于潜在的 SQL 分析与自然语言处理集成的有趣观点。
- **关于目前最佳 AI 模型的讨论**：用户对比了当前的 AI 模型，建议开源需求使用 *Llama 3.1*，封闭系统使用 *ChatGPT* 或 *Claude*。
   - 讨论了模型大小与硬件兼容性的考虑，特别是与 M1 Mac 性能相关的部分。
- **受益于云端模型**：一位用户建议，对于硬件资源有限的人来说，使用云服务可以获得更好的模型访问权限。
   - 这被认为是处理大型模型用户的一个重要考虑因素，包括关于量化（quantization）的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://course.fast.ai/">Practical Deep Learning for Coders - Practical Deep Learning</a>：一门为有一定编程经验、想要学习如何将深度学习和机器学习应用于实际问题的人设计的免费课程。</li><li><a href="https://discuss.huggingface.co/t/giving-ai-a-large-dataset-with-json/106338">使用 JSON 为 AI 提供大型数据集</a>：嘿！我想请教如何在一个巨大的（非常巨大的）JSON 文件上训练现有的 LLM，该文件包含一堆转换为 JSON 格式的文件/目录。我希望 LLM 能够理解...</li><li><a href="https://discuss.huggingface.co/t/how-to-install-flash-attention-on-hf-gradio-space/70698">如何在 HF Gradio Space 上安装 Flash Attention</a>：我尝试将 flash-attn 放入 requirements.txt 文件中以在我的 Space 上安装 Flash Attention，但报错提示未安装 Torch。我也尝试将 Torch 放在 flash-attn 之上，但仍然无法...</li><li><a href="https://x.com/Charlesddamp/status/1833852121290088957">来自 Charles de Dampierre (@Charlesddamp) 的推文</a>：探索我们的 HuggingFace 社区图谱！在这里查看交互式可视化：https://lnkd.in/eXwuKgYw @LysandreJik @JustineTunney @maximelabonne @Dorialexander @Thom_Wolf</li><li><a href="https://huggingface.co/learn/nlp-course">简介 - Hugging Face NLP 课程</a>：未找到描述</li><li><a href="https://tenor.com/view/ok-oh-yes-yes-o-yeah-yes-no-yes-go-on-yea-yes-gif-14382673246413447193">Ok Oh Yes Yes O Yeah Yes No Yes Go On Yea Yes GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://llama.meta.com/docs/how-to-guides/fine-tuning">微调 | 操作指南</a>：全参数微调（Full parameter fine-tuning）是一种对预训练模型所有层的所有参数进行微调的方法。</li><li><a href="https://tenor.com/view/baby-face-palm-really-sigh-stupid-gif-16058491">婴儿捂脸 GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/datasets/nyu-mll/glue?sql_console=true&sql=SELEC">nyu-mll/glue · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/airtrain-ai/fineweb-edu-fortified?sql_console=true">airtrain-ai/fineweb-edu-fortified · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/nyu-mll/glue">nyu-mll/glue · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/nyu-mll/glue?sql_console=true&sql=SELECT+*+FROM+ax+LIMIT+10">nyu-mll/glue · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/pgvector/pgvector">GitHub - pgvector/pgvector: 适用于 Postgres 的开源向量相似度搜索</a>：适用于 Postgres 的开源向量相似度搜索。通过在 GitHub 上创建账号为 pgvector/pgvector 的开发做出贡献。</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L344">transformers/src/transformers/models/gpt2/modeling_gpt2.py at main · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的前沿机器学习。</li><li><a href="https://github.com/huggingface/transformers/pull/33088">由 Manalelaidouni 添加 include_loss_for_metrics · Pull Request #33088 · huggingface/transformers</a>：此 PR 的作用是什么？修复了 #32307。此 PR 通过 include_loss_for_metrics 训练参数标志在 compute_metrics 函数中包含 loss，这对于计算 loss 相关的指标特别有用...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1283391167177691136)** (6 messages): 

> - `Fine-tuning Llama 2`
> - `PEFT for Fine-tuning`
> - `Computer Vision Community Course` 


- **实习生寻求微调 Llama 2 的指导**：一位新实习生表示，在使用 **Llama 2 7b** 模型通过微调创建自定义 LLM 时感到不知所措。
   - 他们请求资源指导，表现出快速学习的热情。
- **推荐使用 PEFT 进行微调**：一位成员建议将 **PEFT** 作为使用 Hugging Face 库进行微调的最佳方法，并提到其仓库中提供了相关教程。
   - 对于拥有单个或多个 GPU 的用户，建议包括使用 **TRL** 和 **Accelerate** 库进行优化训练。
- **关于冒充者综合征的讨论**：另一位成员分享了他们在面对 Linux 输出中的 **PERL** 时产生的**冒充者综合征 (impostor syndrome)** 困扰。
   - 这突显了许多人在技术环境中持续面临的不安全感。
- **开始计算机视觉社区课程**：一位成员宣布他们今天开始学习 Hugging Face 上的**计算机视觉社区课程**。
   - 这展示了参与者在社区内提升技能的承诺。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1283221469568040971)** (5 messages): 

> - `AI in Healthcare`
> - `Retrieval Augmented Generation (RAG)`
> - `Learning Resources on Hugging Face`
> - `AI Applications` 


- **AI 变革医疗保健**：AI 在今年显著改变了**医疗保健**领域，改善了**诊断**、实现了**个性化医疗**并加速了**药物研发**。
   - 随着**可穿戴设备**和 IoT 健康监测的集成，AI 促进了疾病的早期发现和准确诊断，重塑了治疗方法。
- **为初学者简化的 RAG**：[检索增强生成 (RAG)](https://learnbybuilding.ai/tutorials/rag-from-scratch#areas-for-improvement) 使大语言模型能够利用其自身数据，增强其能力。
   - 一篇教程旨在揭秘 RAG，为初学者提供了一种简单直接的方法来构建 RAG 应用，而无需使用晦涩的术语。
- **Hugging Face 学习资源**：Hugging Face 上的**社区计算机视觉课程**教授用户如何使用 HF 库和模型在计算机视觉中应用机器学习。
   - 参与者注意到 [Hugging Face 平台](https://huggingface.co/learn)上提供了大量宝贵的学习资源，增强了他们对 AI 的理解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.forbes.com/sites/bernardmarr/2023/05/10/15-amazing-real-world-applications-of-ai-everyone-should-know-about/">每个人都应该了解的 15 个令人惊叹的 AI 现实应用</a>：未来已来，它由人工智能驱动。阅读本文了解 2023 年正在重新定义行业并影响我们日常生活的 15 个顶级 AI 现实应用。</li><li><a href="https://huggingface.co/learn">Hugging Face - 学习</a>：暂无描述</li><li><a href="https://ai.meta.com/research/publications/transfusion-predict-the-next-token-and-diffuse-images-with-one-multi-modal-model/">无标题</a>：暂无描述</li><li><a href="https://learnbybuilding.ai/tutorials/rag-from-scratch#areas-for-improvement">从零开始构建检索增强生成 (RAG) 应用的初学者指南</a>：这篇文章将教你 RAG 背后的基本直觉，同时提供一个简单的教程帮助你入门。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1283158502671384607)** (23 messages🔥): 

> - `NLP Dataset Release`
> - `Gradio Applications in R`
> - `Agentic Framework in Java`
> - `Image Similarity Demo`
> - `DebateThing AI Debate Generator`

- **波斯语 NLP 数据集发布**：一位用户发布了他们的第一个 NLP 数据集，这是来自 Wikipedia 的 **6,000 多条句子**的波斯语翻译，可在 [Hugging Face](https://huggingface.co/datasets/Reza2kn/OLDI-Wikipedia-MTSeed-Persian) 上获取。他们对未来的数据集发布表示期待。
   - 该数据集的提交日期为 **2024 年 9 月 10 日**。
- **在 R 中构建 Gradio 应用**：一位用户分享了他们的 GitHub 仓库，教授如何使用 R 语言构建 **Gradio 应用程序**，并强调了该过程的简便性（[仓库地址](https://github.com/Ifeanyi55/Gradio-in-R)）。鼓励贡献者为该项目点赞（star）。
   - 该仓库突出了 Gradio 与 R 编程集成的便捷性。
- **Java 版 Agentic 框架**：展示了一个用 Java 实现的 **agentic framework** 演示，并就所采用的方法寻求反馈（[LinkedIn 帖子](https://www.linkedin.com/pulse/af4j-agentic-framework-java-vishal-mysore-8ykrc/?trackingId=BhAYt7NgAfVpR6dW8T0x7A%3D%3D)）。该实现基于 **JADE** 和 **IEEE** 标准。
   - 作者欢迎各种想法和建设性的反馈。
- **使用 Pixtral 的图像相似度演示**：一位用户分享了他们使用 Hugging Face 上的 **pixtral-12b-240910** 模型构建的图像相似度演示，邀请他人测试其功能（[演示地址](https://huggingface.co/spaces/Tonic/Pixtral)）。他们提到在特定 GPU 上进行文本生成时会导致内存溢出（memory overflow）的问题。
   - 他们正在寻求关于如何使用该模型生成描述（caption）的帮助。
- **DebateThing AI 辩论生成器介绍**：一位用户介绍了 **DebateThing.com**，这是一个 AI 驱动的辩论生成器，支持 TTS 和多轮辩论中最多 **4 名参与者**（[网站地址](https://debatething.com/)）。该项目是开源的，使用 **Deno Fresh** 构建。
   - 主要功能包括自定义设置、主持人声音选项以及用于设置辩论的简单用户界面。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mis">Mis (Unknow)</a>: 未找到描述</li><li><a href="https://huggingface.co/mistral-community/pixtral-12b-240910">mistral-community/pixtral-12b-240910 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Tonic/Pixtral">Pixtral Image Similarity - a Hugging Face Space by Tonic</a>: 未找到描述</li><li><a href="https://youtu.be/w6CJtAlGygQ?si=0MzkKj5m2MUiSN59">Who ?</a>: 在这段视频中，我们将测试 2024 年 9 月全球领先的 LLM 模型在速度和性能方面的表现。#tokensperseconds #GPT4o #LLM #SOTA #Cl...</li><li><a href="https://youtu.be/e-RfalOKSMI?si=poGP7w3IJDPA0erW">Contributing to Open Source Changes Your Life ✨ | How to Contribute ⭐️ | Dhanush N</a>: GitHub 拥有超过 4.2 亿个仓库，其中至少有 2800 万个公共仓库。GitHub 上超过 80% 的贡献是针对私有仓库的...</li><li><a href="https://github.com/atlantis-nova/simtag">GitHub - atlantis-nova/simtag: Implementation of Semantic Tag Filtering</a>: 语义标签过滤（Semantic Tag Filtering）的实现。通过在 GitHub 上创建账号来为 atlantis-nova/simtag 的开发做出贡献。</li><li><a href="https://github.com/Ifeanyi55/Gradio-in-R">GitHub - Ifeanyi55/Gradio-in-R</a>: 通过在 GitHub 上创建账号来为 Ifeanyi55/Gradio-in-R 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/Reza2kn/OLDI-Wikipedia-MTSeed-Persian">Reza2kn/OLDI-Wikipedia-MTSeed-Persian · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/U-C4N/HOPE-Agent">GitHub - U-C4N/HOPE-Agent: HOPE (Highly Orchestrated Python Environment) Agent simplifies complex AI workflows. Manage multiple AI agents and tasks effortlessly.  Features: • JSON-based configuration • Rich CLI • LangChain &amp; Groq integration • Dynamic task allocation • Modular plugins  Streamline your AI projects with HOPE Agent.</a>: HOPE (Highly Orchestrated Python Environment) Agent 简化了复杂的 AI 工作流。轻松管理多个 AI Agent 和任务。功能：• 基于 JSON 的配置 • 丰富的 CLI • LangChain 和 Groq 集成 • 动态任务分配 • 模块化插件。使用 HOPE Agent 优化您的 AI 项目。</li><li><a href="https://huggingface.co/learn/cookbook/multiagent_web_assistant">Have several agents collaborate in a multi-agent hierarchy 🤖🤝🤖 - Hugging Face Open-Source AI Cookbook</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Csplk/multi-agent-web-browser">Multi Agent Web Browser - a Hugging Face Space by Csplk</a>: 未找到描述</li><li><a href="https://debatething.com/">DebateThing.com</a>: 使用 AI 就任何话题生成有趣的辩论并免费收听！
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1283309415331663872)** (2 条消息): 

> - `CSV Image Loading`
> - `PyTorch DataLoader Best Practices` 


- **CSV 仅提供用于图像加载的 ID**：讨论指出，CSV 文件仅包含图像 ID，这需要实时获取图像，或者按类别预先将它们分割到训练和测试文件夹中。
   - 与直接从文件夹层级结构创建 DataLoader 对象相比，这种方法可能会引入轻微的延迟。
- **预先应用变换以提高效率**：一位成员强调了 PyTorch 数据加载的最佳实践，建议预先应用图像变换 (transformations)，而不是在运行时实时处理。
   - 这一建议与一篇[讨论此行为的博文](https://blog.dailydoseofds.com/p/a-counterintuitive-behaviour-of-pytorch)中的见解一致，该博文强调了性能方面的考量。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1283154636278206506)** (6 条消息): 

> - `Korean lemmatizer enhancement with AI`
> - `Building NLP models with PyTorch`
> - `Fine-tuning models on specific use cases`
> - `NSFW text detection datasets` 


- **韩语词形还原器 (Lemmatizer) 寻求 AI 助力**：一位成员开发了一个不含 AI 的韩语词形还原器，并寻求关于利用 AI 解决一个词具有多个词元 (lemmas) 的歧义情况的建议。
   - 关键问题是 *“我应该关注哪个方向？”*，因为他们希望 2024 年的生态系统已经更加先进。
- **关于使用 PyTorch 构建 NLP 模型的问题**：一位成员正在探索如何使用 PyTorch 从头开始创建 NLP 模型，但不清楚输入和输出所需的参数数量。
   - 他们提到之前的经验完全在 Computer Vision 领域，表达了想要涉足 NLP 领域的愿望。
- **微调模型的仓库请求**：一位成员正在寻找能够为特定用例微调模型提供指导的 GitHub 仓库。
   - 另一位成员链接了 Hugging Face 的 [transformers examples](https://github.com/huggingface/transformers/tree/main/examples) 作为潜在资源。
- **询问 NSFW 文本检测数据集**：一位成员询问是否存在类似于 MNIST 之于图像识别的 NSFW 文本检测标准学术数据集。
   - 他们提到了 CensorChat 和一篇基于 Reddit 的论文，但指出缺乏全面的数据集。



**提到的链接**：<a href="https://github.com/huggingface/transformers/tree/main/examples">transformers/examples at main · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch, TensorFlow 和 JAX 的前沿机器学习。- huggingface/transformers

  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1283140211005587594)** (141 条消息🔥🔥): 

> - `Aider 功能与工作流`
> - `Aider 中的 Prompt 缓存`
> - `模型性能与对比`
> - `配合工具与 API 使用 Aider`
> - `Aider 用户体验与技巧` 


- **优化 Aider 的工作流**：用户分享了在 Aider 中使用“先询问，后编码（ask first, code later）”的工作流，这在配合 plan 模型时能更清晰地进行代码实现决策。
   - 这种工作流改善了上下文构建，并减少了频繁使用 `/undo` 命令的需求。
- **Prompt 缓存的好处**：Aider 的 Prompt 缓存功能已被证明非常有效，一些用户报告通过策略性地缓存关键文件和指令，Token 使用量减少了高达 **40%**。
   - 缓存系统会保留系统 Prompt 和只读文件等各种元素，从而在交互过程中节省成本。
- **Aider 与其他工具的对比**：用户将 Aider 的能力与 Cursor 和 OpenRouter 等其他工具进行了对比，指出 Aider 具有一些可以节省时间并提高生产力的独特功能。
   - Aider 的智能化功能（例如从 zsh 历史记录中生成有用的别名和速查表）展示了其多功能性。
- **API 性能与问题**：报告指出 Anthropic API 出现了过载问题，影响了多位用户连接和使用服务的能力。
   - 相比之下，用户发现 EU Vertex AI 在停机期间运行良好，凸显了 API 性能的可变性。
- **新模型特性与成本效益**：讨论显示，最新的 GPT-4o 模型在支持结构化输出的同时，大幅降低了输入和输出 Token 的成本。
   - 对于希望优化 GPT 技术使用的用户来说，该模型是一个极具吸引力的选择，尤其是配合特定的模型参数使用时。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/mistralai/status/1833758285167722836?s=46">来自 Mistral AI (@MistralAI) 的推文</a>: magnet:?xt=urn:btih:7278e625de2b1da598b23954c13933047126238a&dn=pixtral-12b-240910&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fannoun...</li><li><a href="https://supermaven.com/">Supermaven: 免费 AI 代码补全</a>: 最快的 Copilot。Supermaven 使用 100 万 Token 的上下文窗口来提供最高质量的代码补全。</li><li><a href="https://www.swift.org/blog/swift-on-windows/">在 Windows 上引入 Swift</a>: Swift 项目正在为 Windows 引入新的可下载 Swift 工具链镜像！这些镜像包含在 Windows 上构建和运行 Swift 代码所需的开发组件。</li><li><a href="https://aider.chat/docs/troubleshooting/support.html">使用 /help</a>: 使用 “/help” 询问有关使用 Aider、自定义设置、故障排除、使用 LLM 等方面的帮助。</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>: 查看你在 OpenRouter 上使用模型的情况。</li><li><a href="https://aider.chat/docs/config/options.html#--chat-language-chat_language">选项参考</a>: 关于 Aider 所有设置的详细信息。</li><li><a href="https://aider.chat/docs/config/options.html#--chat-language-">选项参考</a>: 关于 Aider 所有设置的详细信息。</li><li><a href="https://github.com/pwsacademy/swift-setup/blob/main/platforms/windows/README.md">swift-setup/platforms/windows/README.md at main · pwsacademy/swift-setup</a>: 针对支持 Swift 的平台、编辑器和 IDE 的学生友好型安装指南。 - pwsacademy/swift-setup</li><li><a href="https://openrouter.ai/models/anthropic/claude-3.5-sonnet">Claude 3.5 Sonnet - API, Providers, Stats</a>: Claude 3.5 Sonnet 提供了优于 Opus 的能力、快于 Sonnet 的速度，且价格与 Sonnet 持平。通过 API 运行 Claude 3.5 Sonnet。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1283174777556701185)** (105 messages🔥🔥): 

> - `Using OpenRouter with Aider` (在 Aider 中使用 OpenRouter)
> - `OpenAI Compatibility and API Differences` (OpenAI 兼容性与 API 差异)
> - `YAML Configuration Issues` (YAML 配置问题)
> - `Aider as a Python Tool` (Aider 作为 Python 工具)
> - `Handling Git .gitignore Files in Aider` (在 Aider 中处理 Git .gitignore 文件)


- **探讨 OpenRouter 的优势**：成员们讨论了使用 OpenRouter 相比 Claude API 的优势，指出 OpenRouter 绕过了初始速率限制，且由于税收差异成本更低。
   - 重点讨论了它在以集中方式进行多模型实验方面的实用性。
- **澄清 OpenAI API 与 Ollama API**：明确了 `ollama` 使用 OpenAI 兼容的端点，而 `ollama_chat` 利用原生 API，这影响了它们与模型的交互方式。
   - 这一区别引发了关于哪种 API 配置可能为 Aider 用户带来更好性能的讨论。
- **YAML 配置问题**：用户遇到了 YAML 配置不被 Aider 识别的问题，特别是涉及多行设置时。
   - 有人指出 Aider 使用的是 YAML 的子集，这增加了其配置的复杂性。
- **在 Python 脚本中使用 Aider**：一位用户成功演示了在 Python 脚本中使用 Aider，并询问了关于定义脚本名称和文件修改的问题。
   - 针对如何有效地让 Aider 指向特定文件位置以进行脚本创建，提供了相关建议。
- **Git 忽略文件的问题**：一位用户报告了 Aider 无法编辑 .gitignore 中列出的文件的问题，这在尝试保存编辑时导致了错误。
   - 讨论了如何在不禁用 Git 建议设置的情况下解决此问题，从而增强 Aider 的功能。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/config/adv-model-settings.html">Advanced model settings</a>：配置 LLM 的高级设置。</li><li><a href="https://aider.chat">Home</a>：aider 是你终端里的 AI 配对编程工具。</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>：你可以通过命令行或 Python 对 aider 编写脚本。</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>：aider 是你终端里的 AI 配对编程工具。</li><li><a href="https://aider.chat/docs/leaderboards/#code-editing-leaderboard">Aider LLM Leaderboards</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://pypi.org/project/ConfigArgParse/">ConfigArgParse</a>：argparse 的替代品，允许通过配置文件和/或环境变量设置选项。</li><li><a href="https://aider.chat/docs/leaderboards/#code-editing-leader">Aider LLM Leaderboards</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion">ollama/docs/api.md at main · ollama/ollama</a>：快速上手 Llama 3.1, Mistral, Gemma 2 以及其他大语言模型。 - ollama/ollama</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/openai.md#v1chatcompletions">ollama/docs/openai.md at main · ollama/ollama</a>：快速上手 Llama 3.1, Mistral, Gemma 2 以及其他大语言模型。 - ollama/ollama
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1283163603339575306)** (3 messages): 

> - `Pixel art tools` (像素艺术工具)
> - `Pixtral model release` (Pixtral 模型发布)


- **Pixel Lab 工具受到关注**：一位成员推荐了 [Pixel Lab](https://www.pixellab.ai/)，认为它是为游戏资产生成像素艺术和 2D 精灵图动画的绝佳工具。
   - 他们还分享了一个[关于使用该工具创建出拳动画的 YouTube 教程](https://youtu.be/LQS4J4ub8G4?si=LeegBAaYOzwWRbE0)，并指出该工具仍处于早期开发阶段。
- **Mistral 发布 Pixtral 模型种子**：Mistral 以种子形式发布了一个名为 **Pixtral (12B)** 的新多模态模型，旨在用于包括图像分类和文本生成在内的各种应用。
   - 下载的磁力链接为 `magnet:?xt=urn:btih:7278e625de2b1da598b23954c13933047126238a&dn=pixtral-12b-240910`，它兼容 PyTorch 和 TensorFlow 等框架。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.pixellab.ai/">no title found</a>：未找到描述</li><li><a href="https://youtu.be/LQS4J4ub8G4?si=LeegBAaYOzwWRbE0">Tutorial: How to quickly create punching animations</a>：这是一个关于如何使用骨骼动画工具创建动画的教程。该工具仍处于开发早期，因此预计会有很多改进...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1283182919355072563)** (174 条消息🔥🔥): 

> - `在 AI 图像中保持角色一致性`
> - `用于 Token 处理的 GPU 性能`
> - `LM Studio 用户见面会`
> - `Pixtral 支持与推理代码`
> - `LM Studio 功能与更新` 


- **在 AI 图像中保持角色一致性**：一位用户询问了在更换服装或背景时，确保角色在 AI 生成的图像中保持一致的有效技术。
   - 重点是寻找能让角色的面部和身体在不同画幅中保持可辨识度的方法。
- **用于 Token 处理的 GPU 性能**：用户讨论了不同 GPU 在 Token 处理速度上的差异，其中一位成员报告 6900XT 的速度为 45 tokens/s，并与其他用户进行了对比。
   - 有人建议通过刷新 BIOS 来提升性能，用户们也对分享出的意外性能数据感到沮丧。
- **LM Studio 用户见面会**：几位 LM Studio 用户宣布将在伦敦市中心举行见面会，讨论 Prompt Engineering，并邀请其他人参加。
   - 详情包括时间和物流安排，鼓励参与者寻找正在使用笔记本电脑的非学生群体。
- **Pixtral 支持与推理代码**：有人咨询了关于 Pixtral 的支持情况，特别是关于推理代码及其与现有库的集成。
   - 回复指出，虽然 Mistral 发布了一些代码，但目前 Transformers 或 llama.cpp 尚不支持。
- **LM Studio 功能与更新**：用户讨论了 LM Studio 的更新功能，并注意到 Apple Silicon 上缺少 GPU 滑块。
   - 用户分享了如何通过选项菜单访问模型设置，以调整卸载到 GPU 的层数（Layers）。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/MistralAI/status/1833758285167722836">来自 Mistral AI (@MistralAI) 的推文</a>: magnet:?xt=urn:btih:7278e625de2b1da598b23954c13933047126238a&dn=pixtral-12b-240910&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fannoun...</li><li><a href="https://maps.app.goo.gl/1bHCRW5DP79fKapUA">  Google Maps  </a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1283142661519446077)** (67 条消息🔥🔥): 

> - `AMD vs NVIDIA 性能`
> - `Surface Studio Pro 升级`
> - `RTX 4090D 特性`
> - `AI 模型需求`
> - `多 GPU 基准测试` 


- **AMD 与 NVIDIA：大辩论**：一位用户指出 **Blender** 在非 NVIDIA 硬件上表现不佳，引发了关于 **AMD** 和 **NVIDIA** GPU 性能差异的讨论。
   - 会议强调，在某些条件下 **4090D** 的表现可能优于标准显卡，这引发了关于 GPU 选择的不同意见。
- **Surface Studio Pro 的升级**：一位用户对 **Surface Studio Pro** 有限的升级潜力表示沮丧，并寻求关于 **eGPU** 或 **SSD** 等可能增强方案的建议。
   - 建议包括与其升级现有笔记本电脑，不如为新的专用 AI 设备预留预算。
- **RTX 4090D 的特性**：讨论围绕 **RTX 4090D** 展开，这是一款中国专供的 GPU，拥有更多 VRAM 但 CUDA 核心较少，尽管在某些任务中性能较低，但仍是一个独特的选择。
   - 一些人指出，由于其显存容量较高，尽管在游戏方面较慢，但对于 AI 应用来说可能是一项值得的投资。
- **AI 模型与 RAM 需求**：参与者讨论了同时运行多个 AI 模型对 RAM 的显著需求，并将其与各种 GPU 配置的性能联系起来。
   - 共识是，充足的 VRAM 对于 AI 工作负载的流畅运行至关重要。
- **系统中多 GPU 的基准测试**：一位用户报告了对同时包含 **AMD Radeon 7** 和 **Intel Arc A380** 的系统进行基准测试的情况，强调了系统如何默认使用 **Intel** GPU。
   - 建议移除一个 GPU，以解决使用来自不同制造商的多块显卡时的兼容性问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.microsoft.com/en">您的请求已被拦截。这可能是由于多种原因造成的。</a>：未找到描述</li><li><a href="https://www.tomshardware.com/pc-components/gpus/nvidias-made-for-china-rtx-4090d-is-only-5-slower-in-gaming-performance-than-the-original-rtx-4090">Nvidia 为中国制造的 RTX 4090D 在游戏性能上仅比原始 RTX 4090 慢 5%</a>：符合制裁标准且速度依然很快，在 AI 工作负载中大约慢 10%。</li><li><a href="https://gamerant.com/asus-rtx-4090d-outperforming-rtx-4090/">华硕 RTX 4090D 性能超越 RTX 4090</a>：华硕提升了 Nvidia 中国专供 RTX 4090D GPU 的性能，使其超越了美国市场上的 RTX 4090。</li><li><a href="https://videocardz.com/newz/nvidia-geforce-rtx-4090d-with-48gb-and-rtx-4080-super-32gb-now-offered-in-china-for-cloud-computing">配备 48GB 的 NVIDIA GeForce RTX 4090D 和 32GB 的 RTX 4080 SUPER 现已在中国提供云服务 - VideoCardz.com</a>：RTX 4090D 和 RTX 4080 SUPER 显存翻倍。有需求的地方就有定制解决方案。专为规避美国出口限制而设计的中国核心 RTX 4090D 显卡...</li><li><a href="https://videocardz.com/newz/nvidia-geforce-rtx">NVIDIA GeForce RTX 2050 和 MX500 笔记本 GPU 在 3DMark TimeSpy 中测试 - VideoCardz.com</a>：RTX 2050 和 MX570/550 的首批测试结果。在意外发布仅几小时后，NVIDIA 新入门级 GPU 的首批基准测试结果已经公布。</li><li><a href="https://www.microsoft.com/en-us/d/surface-studio-2-plus/8vlfqc3597k4?activetab=pivot:overviewtab">购买 Surface Studio 2+ - 查看桌面规格、价格、屏幕尺寸 | Microsoft Store</a>：从 Microsoft Store 购买 Surface Studio 2+。这款引人注目的 28 英寸触摸屏可从桌面模式转换为画布模式，配备第 11 代 Intel® Core™ H 系列处理器和 NVIDIA GeForce RTX...
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1283149433067929784)** (197 messages🔥🔥): 

> - `Stable Diffusion 模型对比`
> - `文生图生成技术`
> - `AI 图像生成技术讨论`
> - `AI 训练硬件推荐`
> - `Reflection LLM 概览` 


- **Stable Diffusion 模型对比**：用户讨论了各种 Stable Diffusion 模型，强调了像 '1.5 ema only' 这样的旧模型与性能更好的新选项之间的差异。
   - 进行了 GPU 之间的对比，提到 RTX 4060 Ti 在 AI 任务中的表现优于 7600 和 Quadro P620。
- **文生图生成技术**：强调模型应在最佳分辨率下生成，例如早期模型为 512x512，以减少放大时的伪影。
   - 用户分享了图像生成的工作流，建议先使用低分辨率生成，然后进行放大以获得更高质量的输出。
- **AI 图像生成技术讨论**：关于 AI 模型的讨论揭示了人们对各种 LLM 因共同的训练数据和技术而趋于雷同的持续担忧。
   - 用户注意到，在较新的模型中，手部生成不再是图像生成中的难题，这表明 AI 能力正在快速进步。
- **AI 训练硬件推荐**：社区辩论了用于训练 AI 模型的 GPU 的有效性，由于 CUDA 的兼容性，更倾向于 Nvidia 而非其他品牌。
   - 用户一致建议，虽然低显存（VRAM）的 GPU 可能适用于某些模型，但为了获得最佳性能，首选 20GB 等高端 GPU。
- **Reflection LLM 概览**：用户讨论了 Reflection LLM 的特性，该模型本应通过“思考”和“反思”超越其他模型，但其实际表现却遭到了批评。
   - 有人对 API 与开源版本之间的差异表示担忧，这导致了对其宣称能力的怀疑。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://dsc.gg/vexel">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 非常适合玩游戏、与朋友一起放松，甚至建立全球社区。自定义你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://academictorrents.com/details/9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4">Reddit comments/submissions 2005-06 to 2023-12</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1283229985338953758)** (1 messages): 

> - `Novita Endpoint 故障` 


- **Novita Endpoints 遭遇故障**：所有 **Novita endpoints** 目前正经历故障，导致那些过滤到 Novita 且没有备选方案（fallbacks）的用户出现 **403 状态错误**。
   - *如果你允许备选方案（fallbacks），那么你的请求应该会照常进行。*
- **Novita 故障已解决**：之前报告的 **Novita endpoints** 问题现已解决。
   - 故障结束后，用户可以期待恢复正常功能。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1283139989974290634)** (171 条消息🔥🔥): 

> - `编程工具建议`
> - `关于 Hermes 模型定价的讨论`
> - `Pixtral 模型能力`
> - `OpenRouter 与 Cursor 的集成`
> - `Novita 服务中断` 


- **编程工具推荐**：一位用户询问了编程工具，提到计划利用 AWS Bedrock 配合 Litelm 进行速率管理和成本优化。
   - 其他用户推荐了 Aider 和 Cursor 等工具，并对其有效性和用户体验发表了不同看法。
- **关于 Hermes 模型定价的困惑**：关于 Hermes 3 模型是否会保持免费存在不确定性，一位用户推测更新后的端点可能会收取 **$5/M** 的费用。
   - 成员们表达了对开始收费后性能提升的期待，而一些人则认为仍会提供免费的替代方案。
- **Pixtral 模型的使用案例**：用户讨论了 Pixtral 12B 模型的能力，判断它可能只接受图像输入以产生文本输出，这意味着其纯文本处理能力有限。
   - 共识似乎倾向于它的功能将类似于 LLaVA，可能在图像任务中提供专业性能。
- **将 OpenRouter 与 Cursor 集成**：一位用户在将 OpenRouter 与 Cursor 配合使用时遇到问题，引发了关于启用模型功能所需配置调整的讨论。
   - 用户分享了对 Cursor 仓库中现有问题的见解，强调了在使用特定模型时的硬编码路由问题。
- **Novita 服务中断讨论**：成员们报告了影响与 OpenRouter 关联的 Novita 服务的临时停机，并对问题持续时间不明表示沮丧。
   - 一些用户推测了“NOT_ENOUGH_BALANCE”错误背后的原因，倾向于认为是提供商端的身份验证问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/chat?models=meta-llama/llama-3.1-8b-instruct:free>">Chatroom | OpenRouter</a>：LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在您的浏览器中。</li><li><a href="https://blog.arcee.ai/meet-arcee-supernova-our-flagship-70b-model-alternative-to-openai/">Meet Arcee-SuperNova: Our Flagship 70B Model, Alternative to OpenAI</a>：认识 Arcee-SuperNova：一款具有突破性的模型，在指令遵循方面具有最先进的能力，并与人类偏好高度对齐。</li><li><a href="https://github.com/mistralai/mistral-common/releases/tag/v1.4.0">Release  v1.4.0 - Mistral common goes 🖼️  · mistralai/mistral-common</a>：Pixtral 发布了！Mistral common 现已支持图像！您现在可以将图像和 URL 随文本一起传递到用户消息中。pip install --upgrade mistral_common。图像编码方式如下...</li><li><a href="https://huggingface.co/Sao10K/L3.1-70B-Hanami-x1/">Sao10K/L3.1-70B-Hanami-x1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mistral-community/pixtral-12b-240910/discussions/6#66e1b9d052c91424e2374dd7">mistral-community/pixtral-12b-240910 · Any Inference code?</a>：未找到描述</li><li><a href="https://status.openrouter.ai/">OpenRouter Status</a>：OpenRouter 事件历史</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:free">Hermes 3 405B Instruct (free) - API, Providers, Stats</a>：Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理、多轮对话、长上下文连贯性...</li><li><a href="https://github.com/getcursor/cursor/issues/1511">Can&#39;t use claude 3.5 sonnet with openrouter, seems like a cursor issue · Issue #1511 · getcursor/cursor</a>：在 Windows 11 上使用 Cursor。直到最近（至少上周五）还能正常工作。如果我使用 anthropic/claude-3.5-sonnet，会收到 API key 无效的错误。在模型首选项中验证 API key 时...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1283159997966057522)** (23 条消息🔥): 

> - `Matmul Algorithms`
> - `Cudamode-IR Online Discussions`
> - `Neural Network Quantization`
> - `Interview Preparation Strategies` 


- **探索 Matmul 算法**：一位成员询问了寻找各种 **matmul 算法**（如 Grouped GEMM 和 Split K）资源的渠道。
   - 另一位成员建议查看 **Cutlass examples**，认为它是这些算法的全面参考源。
- **Cudamode-IR 走向线上**：关于 **cudamode-irl** 在线版本可行性的讨论引发了对社区驱动项目的建议。
   - 成员们指出，目前的社区已经是一个有效的在线平台，欢迎各种贡献和讨论。
- **冷启动与 GPU 担忧**：一位成员询问 **cold starts**（冷启动）是否会对本地推理场景下的 GPU 性能产生负面影响。
   - 另一位成员回应询问为何会认为这会损害 GPU，并指出在启动时加载权重是常见操作。
- **量化挑战**：为了深入理解 [Post-Training Quantization](https://github.com/satabios/quantization/tree/master/quant/layer_wise_weights_activation)，一位成员在面临精度下降后，就其实现寻求反馈。
   - 社区建议包括对激活值使用动态量化，以在量化过程中提高精度。
- **面试准备的差异性**：成员们讨论了不同公司面试问题的 **high variance**（高方差/巨大差异），强调了在平衡 CUDA、DL 算法等方面存在的挑战。
   - 分享的经验表明，面试内容涵盖了 ML 理论和实际编程，一些人还提到增加了系统设计（system design）环节。



**提到的链接**：<a href="https://discuss.pytorch.org/t/significant-accuracy-drop-after-custom-activation-quantization-seeking-debugging-suggestions/209396">Significant Accuracy Drop After &quot;Custom&quot; Activation Quantization – Seeking Debugging Suggestions</a>：为了加深对神经网络量化的理解，我正在从头开始重新实现 Post-Training Quantization (PTQ)，尽可能减少对 PyTorch 函数的依赖。代码可以在这里找到：Git...

  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1283450831059943556)** (7 条消息): 

> - `Kernel Outputs Garbage with Autotune`
> - `Utilizing Tensor Cores in Triton`
> - `Support for uint4 in Triton`
> - `Using Cutlass for Tensor Operations` 


- **Autotune 导致 Kernel 输出乱码**：一位用户报告说，在 Triton 中使用 autotune 时，即使使用完全相同的配置，Kernel 也会输出乱码（garbage），而不使用 autotune 时则运行正常。
   - *非常奇怪*，但该问题通过 **重新安装 Triton** 得到了解决。
- **探索 Tensor Core 利用率**：有人提问 Triton 是否有特殊机制来利用 Tensor Cores，因为它具有硬件无关性。
   - 另一位成员澄清说，开发者只需调用 `tl.dot` 并确保输入具有 **正确的形状（shapes）** 即可。
- **关于 Triton 支持 uint4 的讨论**：一位成员表达了对 Triton 支持 **uint4** 数据类型的兴趣。
   - 然而，目前官方表示 Triton **不支持** **uint4**。
- **使用 Cutlass 进行高级张量操作**：建议对于像 **uint4** 这样不支持的类型，可能需要使用 **Cutlass** 来进行张量操作。
   - 分享了 **Cutlass GitHub 仓库** 的链接，作为处理此类情况的资源。



**提到的链接**：<a href="https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm80.hpp#L1384">cutlass/include/cute/arch/mma_sm80.hpp at main · NVIDIA/cutlass</a>：用于线性代数子程序的 CUDA 模板。通过在 GitHub 上创建账号为 NVIDIA/cutlass 做出贡献。

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1283228672823787542)** (1 条消息): 

> - `FlexAttention speedup`
> - `flash_attn_varlen_func comparison` 


- **FlexAttention 表现出显著的加速**：一位用户报告尝试了带有文档掩码（document masking）的 **FlexAttention**，与 padding 方案相比，实现了超过 **60% 的加速**。
   - 他们询问了同样使用文档掩码的 **flash_attn_varlen_func()** 的性能数据，以寻求对比。
- **关于 flash_attn_varlen_func 性能的咨询**：同一位用户对 **flash_attn_varlen_func()** 在文档掩码方面的性能指标表示好奇。
   - 这突显了对不同 Attention 机制之间更详细性能对比的需求。


  

---

### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1283387326650912786)** (39 条消息🔥): 

> - `OpenAI RSU 讨论`
> - `OpenAI 股票的二级市场`
> - `Microsoft 对 OpenAI 的投资`
> - `Liquid 上的合作机会` 


- **关于 OpenAI RSU 的讨论**：成员们讨论了 OpenAI 的员工（特别是四年前入职的员工）如果未出售其 RSU，其价值可能已经增长了 **6-7 倍**，强调了对其价值的*膨胀感知*。
   - 一位成员指出，在 OpenAI IPO 之前，这些都只是*纸面财富*，而其他人则承认二级市场交易的现实性，这允许部分员工套现。
- **二级市场交易见解**：对话涉及 OpenAI 如何进行了**三次二级市场交易**，为员工提供了套现股份的机会，[更多细节见此](https://www.crunchbase.com/organization/openai/company_financials)。
   - 成员们推测了这些二级市场轮次如何随着时间的推移影响股票定价和估值，并暗示 VC 在这些交易中进行谈判。
- **Microsoft 对 OpenAI 的历史性投资**：回顾了 Microsoft 在 **2020 年** 向 OpenAI 投资了 **10 亿美元**，当时公司估值为 **150 亿美元**。
   - 这次投资使 Microsoft 成为 AI 领域的重要参与者，并引发了关于 OpenAI 财务里程碑时间线的讨论。
- **Liquid 上的潜在合作**：一位成员提出可以为任何有兴趣与 Liquid 合作的人牵线搭桥，展示了对社交机会的开放态度。
   - 其目的是在这些技术发展的背景下促进关系和伙伴关系的建立。



**提及的链接**：<a href="https://cbg.com.cy/investors-are-valuing-openai-at-over-100-billion-in-the-secondary-market/">no title found</a>: no description found

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1283265887583080549)** (9 条消息🔥): 

> - `FP6 API 引入`
> - `BF16 与 FP16 的混淆`
> - `训练后量化 (Post-Training Quantization) 的挑战`
> - `torchao v0.5.0 发布`
> - `量化 TTS 模型的局限性` 


- **FP6 成为主要 API**：一名成员宣布，由于 **fp6** 出色的性能，他们正将其作为重要 API 添加到主 README 中，详情见此 [Pull Request](https://github.com/pytorch/ao/pull/867)。
   - 会上讨论了将 **BF16** 与 **FP16** 集成的困难，这引发了对性能依赖可能导致用户混淆的担忧。
- **激活量化的挑战**：一名成员正在从头重新实现 **Post-Training Quantization**，但在激活量化过程中遇到了**显著的精度下降**，正在寻求建议。
   - 他们在 [Torch 论坛](https://discuss.pytorch.org/t/significant-accuracy-drop-after-custom-activation-quantization-seeking-debugging-suggestions/209396)上分享了一篇详细帖子以获取更多见解。
- **torchao v0.5.0 发布亮点**：社区庆祝了 **torchao v0.5.0** 的发布，该版本引入了 **float8 训练与推理**以及对 **HQQ** 的支持等功能。
   - [发布说明 (Release Notes)](https://github.com/pytorch/ao/releases/tag/v0.5.0) 详细介绍了内存高效推理和量化训练方面的增强功能。
- **探索开源 (OSS) 量化 TTS 模型**：一名成员讨论了在易于搜索的开源 **量化 TTS 模型**方面的空白，并提到 **torchao** 可能很容易填补这一空白。
   - 在进一步研究时，他们对当前 **量化 API** 在 [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2) 等 TTS 模型上的局限性提出了疑问。
- **讨论 torchao 的局限性**：一名成员概述了 **torchao** 表现不佳的情况，指出其与 **compile** 的不兼容性，以及主要对 **CPU** 和卷积的依赖。
   - 他们强调了在执行非常规操作时面临的挑战，特别是与 torchao 处理方式不匹配的线性函数。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/coqui/XTTS-v2">coqui/XTTS-v2 · Hugging Face</a>：未找到描述</li><li><a href="https://discuss.pytorch.org/t/significant-accuracy-drop-after-custom-activation-quantization-seeking-debugging-suggestions/209396">Significant Accuracy Drop After &quot;Custom&quot; Activation Quantization – Seeking Debugging Suggestions</a>：为了加深我对神经网络量化的理解，我正在从头重新实现训练后量化 (PTQ)，且尽量减少对 PyTorch 函数的依赖。代码可以在这里找到：Git...</li><li><a href="https://github.com/pytorch/ao/releases/tag/v0.5.0">Release v0.5.0 · pytorch/ao</a>：亮点：我们很高兴地宣布 torchao 0.5 版本发布！此版本增加了对内存高效推理、float8 训练与推理、int8 量化训练、HQQ、自动混合精度 (automatic mi...) 的支持。</li><li><a href="https://github.com/pytorch/ao/pull/867">README and benchmark improvements by HDCharles · Pull Request #867 · pytorch/ao</a>：摘要：量化 README：在基准测试中添加了 fp6；重写了自动量化 (autoquant) 部分，在深入细节前提供更高层级的解释；重新排序了仿射量化 (affine quantization) 部分，首先展示...
</li>
</ul>

</div>

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1283444121935282229)** (1 条消息): 

> - `Neural Network Quantization`
> - `Post-Training Quantization (PTQ)`
> - `Weight Quantization`
> - `Activation Quantization`
> - `Debugging Accuracy Drop` 


- **从零开始实现 PTQ**：一位成员正在尝试在尽可能少依赖 PyTorch 函数的情况下，从零开始重新实现 Post-Training Quantization (PTQ)，并已成功实现了 **weight-only quantization**。
   - 然而，他们在尝试 activation quantization 时报告了显著的 **accuracy drop**（精度下降），并寻求改进建议。
- **寻求精度问题的帮助**：该成员在 [torch forum](https://discuss.pytorch.org/t/significant-accuracy-drop-after-custom-activation-quantization-seeking-debugging-suggestions/209396) 上发帖分享了他们在实现 activation quantization 后的挑战以及实现后的精度下降问题。
   - 他们正在寻求社区的 **debugging suggestions**（调试建议）和见解，以解决所面临的问题。



**提到的链接**：<a href="https://discuss.pytorch.org/t/significant-accuracy-drop-after-custom-activation-quantization-seeking-debugging-suggestions/209396">Significant Accuracy Drop After &quot;Custom&quot; Activation Quantization – Seeking Debugging Suggestions</a>：为了加深对 Neural Network quantization 的理解，我正在尽可能少地依赖 PyTorch 函数，从零开始重新实现 Post-Training Quantization (PTQ)。代码可以在这里找到：Git...

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1283151469268176916)** (46 条消息🔥): 

> - `Activation Function Save`
> - `FP8 Custom Implementation`
> - `Memory Management in Optimizers`
> - `Tensor Scaling Approaches`
> - `Debugging Fused Classifier` 


- **为 Backward Pass 保存激活值**：有人指出，在应用激活函数后会为 backward pass 保存激活值，且可选的 activation checkpointing 正在分支上开发。
   - 一位成员强调了重新计算某些输出（如 **GELU** 或 layer normalization）以节省内存的能力。
- **探索自定义 FP8 实现**：讨论了将 *Tile-wise scaling* 作为一种替代的 FP8 方法，这有别于典型的 per-tensor absmax tracking，可以显著简化代码库。
   - 成员们提到，可能需要自定义 GEMM kernel，因为现有方法似乎有限，且主要集中在 per-tensor 方法上。
- **Buffer 内存管理策略**：一位成员建议实现一个固定大小的双端栈（double-ended stack）来管理中间 buffer 分配，这将有效解决典型的分配模式问题。
   - 这一想法被认为既适用于 scratch tensor 也适用于 temporary tensor，同时能简化整体内存管理工作。
- **Tensor Scaling 技术的进展**：讨论围绕实现 per-row/column absmax scaling 策略以提高精度展开，特别是在 FP16 和 tensor-core 操作中。
   - 对话还涉及了在大模型架构中优化分类相关的内存使用以及准确的输入缩放。
- **调试 Fused Classifier 中的问题**：分享了一个幽默的失误：在验证期间 fused_classifier 错误地写入了 **dlogits**，该调试事件已得到解决。
   - 该 bug 追溯到代码逻辑中的一个小错误，突显了验证条件处理的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_mixed_dtype_gemm.cu#L330">cutlass/examples/55_hopper_mixed_dtype_gemm/55_hopper_mixed_dtype_gemm.cu at main · NVIDIA/cutlass</a>：用于线性代数子程序的 CUDA 模板。通过在 GitHub 上创建账号为 NVIDIA/cutlass 开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/bd457aa19bdb7c0776725f05fe9ecb692558aed8/llmc/cuda_common.h#L46">llm.c/llmc/cuda_common.h at bd457aa19bdb7c0776725f05fe9ecb692558aed8 · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 开发做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/1283457784528244828)** (7 条消息): 

> - `cuSparse 用法`
> - `稀疏矩阵乘法`
> - `压缩感知理论` 


- **寻求 cuSparse 批量乘法技巧**：一位成员询问如何使用 **cuSparse** 将稀疏矩阵 **S** 与具有重复模式的稠密矩阵 **D** 进行批量相乘，且无需复制稠密矩阵。
   - 他们提供了一段代码片段，并就使用 API 时如何优化内存分配寻求建议。
- **对 cuSparse 性能的提醒**：另一位成员警告说，**cuSparse** 是针对极稀疏矩阵（少于 1% 的非零元素）优化的，对于典型的机器学习稀疏度水平，其性能可能并不高效。
   - 他们建议如果数据中存在可利用的结构，可以考虑自定义实现。
- **对压缩感知的兴趣**：一位用户建议使用**压缩感知理论（compressed sensing theory）**来实现与处理超稀疏矩阵类似的效果，但使用更小的稠密矩阵。
   - 然而，原提问者表示担心，由于各批次之间的稀疏模式各异，一致的投影（projection）可能并不可行。
- **压缩感知中的投影挑战**：作为回应，一位成员指出，只要遵循正确的分布，具有单位范数列的固定随机矩阵就可以编码所有信息。
   - 他们强调需要确定合适的投影尺寸，同时提到基础投影可能会随着稀疏模式的变化而改变。


  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1283284963042525234)** (9 条消息🔥): 

> - `黑客松参与`
> - `多 GPU 增强`
> - `GPU 供应商更新`
> - `赞助`
> - `云端额度` 


- **寻求参加黑客松**：一位成员表达了为黑客松做贡献的兴趣，并强调了从零开始构建 **PyTorch/cuDNN** 的经验以及即将在 **Tenstorrent** 入职的情况。他们分享了一个关于卷积核工作的 [GitHub 仓库](https://github.com/yugi957/Journey/tree/convolution)。
   - 这一角色可以为算子内核（kernel）开发提供宝贵的见解。
- **分享多 GPU 构思**：成员们讨论了关于 **Multi-GPU** 使用的令人兴奋的新想法，包括延长上下文长度和优化内存效率。他们为合作者提供了[更多细节链接](https://docs.google.com/document/d/1YuCvBeMD5wlwI0iAV1xf3aokf4tj53epLNyRFeUuf1U/edit)。
   - 目标是让参与者能够以最小的开销开展他们的项目。
- **发布 GPU 供应商更新**：团队确认已为黑客松争取到 **30 万美元的云端额度**，以及一个 **10 节点的 GH200 集群**和一个 **4 节点的 8 H100 集群**。他们计划与赞助商合作，将访问权限延伸到活动结束后的与会者。
   - 感谢包括 **Fal**、**Anyscale** 和 **NVIDIA** 在内的多家赞助商的大力支持，更多详情可在[活动网站](https://events.accel.com/cudamode)上找到。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.google.com/document/d/1YuCvBeMD5wlwI0iAV1xf3aokf4tj53epLNyRFeUuf1U/edit">Multi-gpu Track</a>：多 GPU 赛道。让 llama-405B 在 4090 或性能较低的 GPU 上运行得更快。目前，可以将 llama-405B 装入 4 张 48GB 的 4090 中，但速度很慢。我们能否将 torch.compile 作为一等公民引入？目前，它协同...</li><li><a href="https://github.com/yugi957/Journey/tree/convolution">GitHub - yugi957/Journey at convolution</a>：通过在 GitHub 上创建账号来为 yugi957/Journey 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1283160019025793034)** (3 条消息): 

> - `SGD Implementation` (SGD 实现)
> - `Label Smoothing in FLCE` (FLCE 中的 Label Smoothing)


- **在等待资源期间尝试 SGD**：一位成员提到，他们在等待获取 **80GB A100 实例** 的访问权限时，正在尝试使用 **SGD**。
   - 这突显了社区在其项目中对效率和资源利用率的持续追求。
- **为 FLCE 添加 Label Smoothing 支持**：一位贡献者修复了 **FLCE 的 label_smoothing 支持**，并向 Liger Kernel 仓库提交了一个 [pull request](https://github.com/linkedin/Liger-Kernel/pull/244)。
   - 该更新包括在 **RTX-3080** 上进行的测试，以确保正确性、代码风格和收敛性，解决了之前记录的一个问题。



**提及的链接**：<a href="https://github.com/linkedin/Liger-Kernel/pull/244">Add label smoothing to FLCE and unit tests by Tcc0403 · Pull Request #244 · linkedin/Liger-Kernel</a>：摘要：修复 #243，测试已完成。硬件类型：RTX-3080。运行 make test 以确保正确性，运行 make checkstyle 以确保代码风格，运行 make test-convergence 以确保收敛性。

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1283145265892687882)** (63 条消息🔥🔥): 

> - `OpenAI Departures` (OpenAI 人员离职)
> - `Meta's AI Supercomputing Cluster` (Meta 的 AI 超级计算集群)
> - `Adobe Firefly Video Model` (Adobe Firefly 视频模型)
> - `Pixtral Model Performance` (Pixtral 模型性能)
> - `Government Bureaucracy and Automation` (政府官僚机构与自动化)


- **OpenAI 经历重大人员变动**：OpenAI 今天出现了重大的人才流失，前员工 [Alex Conneau](https://x.com/alex_conneau/status/1833535309902189015?s=46) 宣布离职并创办新公司，而 [Arvind](https://x.com/arvind_io/status/1833571886766399773?s=46) 分享了加入 Meta 的兴奋之情。
   - 有传言称，个人简介中提及 **GPT-5** 可能预示着即将推出的模型，尽管对这些说法存在怀疑。
- **Meta 构建 10 万个 GPU 的 AI 超级计算集群**：据报道，Meta 即将完成一个由 **100,000 个 Nvidia H100 GPU** 组成的 AI 超级计算集群，用于训练 Llama 4，且未选择 Nvidia 专有的网络设备。
   - 这一举措的规模强调了 Meta 对 AI 能力的投入，尤其是在该领域竞争加剧的情况下。
- **Adobe Firefly 视频模型发布**：Adobe 宣布即将推出 **Firefly Video Model**，强调了自 2023 年 3 月推出 Firefly 以来的快速进步，及其与热门 Creative Cloud 功能的集成。
   - 新的视频模型将于今年晚些时候发布测试版，表明 Adobe 对利用生成式 AI 进行视频制作的浓厚兴趣。
- **Pixtral 模型展现竞争优势**：在最近的 Mistral 峰会上，据报道 **Pixtral 12B** 模型的表现优于 **Phi 3** 和 **Claude Haiku** 等同类模型，在图像尺寸和任务处理方面提供了灵活性。
   - 现场演示展示了 **Pixtral 在 OCR 任务中的强劲表现**，引发了关于其相对于竞争对手准确性的讨论。
- **政府通过 LLM 实现自动化**：讨论了 LLM 如何显著自动化政府的官僚流程，潜在地节省数十亿美元的纳税人资金。
   - 然而，也有人担心这种自动化可能会暴露那些不愿简化操作的隐性利益。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/mistralai/status/1833758285167722836?s=46">来自 Mistral AI (@MistralAI) 的推文</a>: magnet:?xt=urn:btih:7278e625de2b1da598b23954c13933047126238a&dn=pixtral-12b-240910&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fannoun...</li><li><a href="https://x.com/alex_conneau/status/1833535309902189015?s=46">来自 Alexis Conneau (@alex_conneau) 的推文</a>: 职业更新：在 @OpenAI 参与构建 #Her 的奇妙旅程后，我决定创办一家新公司。</li><li><a href="https://x.com/swyx/status/1833926630861070359">来自 swyx.io (@swyx) 的推文</a>: **触手可及的前沿 AI** 我在今天 @MistralAI 峰会上的现场笔记，嘉宾包括 Jensen Huang 和 @arthurmensch 及其团队，此处为推文线程。</li><li><a href="https://blog.adobe.com/en/publish/2024/09/11/bringing-gen-ai-to-video-adobe-firefly-video-model-coming-soon">通过 Adobe Firefly Video Model 为视频带来生成式 AI | Adobe 博客</a>: Firefly Video Model 的最新进展。</li><li><a href="https://x.com/amir/status/1833898418026275089?s=46">来自 Amir Efrati (@amir) 的推文</a>: 新闻：Meta 即将完成一个由超过 10 万块 Nvidia H100 组成的 AI 超级计算集群，用于训练 Llama 4。在此过程中，它将不再使用 Nvidia 的专用网络设备。https://www.theinform...</li><li><a href="https://x.com/apples_jimmy/status/1833595024543781088?s=46">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>: 好了，回到 10 月。我们应该会在 10 月迎来一个 4.x 模型（也许仍被称为 4.5，我的老朋友）。至于大块头 GPT 5，我听说最早在 12 月，但为了大家的理智，我建议关注明年第一/第二季度...</li><li><a href="https://x.com/cthorrez/status/1833631799593078878?s=46">来自 Clayton Thorrez (@cthorrez) 的推文</a>: 在今天进行的 24 场比赛中的 14 场里，539 的表现并非超人。它的正确率为 8/14 (57%)，而 1960 年代发明、仅使用历史数据的 Elo 等级分的正确率为 11/14 (78%)...</li><li><a href="https://x.com/arvind_io/status/1833571886766399773?s=46">来自 Arvind Neelakantan (@arvind_io) 的推文</a>: 很高兴加入 @AIatMeta！过去 4.5 年在 @OpenAI 工作，参与了 embeddings、GPT-3 & 4、API 和 ChatGPT 的开发，这是我职业生涯的高光时刻。现在，我很激动能参与下一代 Llama 的研发...</li><li><a href="https://x.com/abacaj/status/1833942228365987915">来自 anton (@abacaj) 的推文</a>: Pixtral 把这题做错了笑死，Qwen2-VL 搞定了这题... 第 5 题 "vulnerable" 应该是 "unlawful"。引用 swyx.io (@swyx) @GuillaumeLample @ArtificialAnlys @dchaplot ...</li><li><a href="https://fxtwitter.com/swyx/status/1833933507590324483">来自 swyx.io (@swyx) 的推文</a>: 在 Mistral 的受邀制会议上，Jensen Huang 同台发布了 Pixtral 的新细节，包括基准测试！！Pixtral > Phi 3, Qwen VL, Claude Haiku, LLaVA。引用 swyx.io (@swyx) @Guill...</li><li><a href="https://fxtwitter.com/swyx/status/1833932883347865802">来自 swyx.io (@swyx) 的推文</a>: @GuillaumeLample @ArtificialAnlys 和 @dchaplot 上台发布了 Pixtral 12B 的 alpha 信息（抄送 @altryne）。与 @imhaotian 的 LLaVA 风格融合模型不同——固定图像尺寸、较少的...
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1283197990659686512)** (9 messages🔥): 

> - `Matt Shumer 的公告`
> - `Reflection 70B 模型问题`
> - `社区反应`
> - `透明度与问责` 


- **Matt Shumer 项目发布引发的抵制**：Matt Shumer 对他关于 **Reflection 70B 项目** 的过早发布表示遗憾，称他当时是根据手头掌握的信息过快地做出了决定。
   - *我根据当时掌握的信息，决定发布这种新方法。*
- **Reflection 70B 在错误模型上的基准测试**：在一名成员声称由于配置错误导致测试实际上是在之前的 **Sonnet 3.5** 模型上运行后，关于 **Reflection 70B** 基准测试的担忧浮出水面。
   - 该成员指出，Matt *在不知情的情况下在错误模型上运行了基准测试*，并尽管存在问题仍发布了结果。
- **要求 Matt Shumer 提高透明度**：一位开发者在投入大量资源托管其模型后，对 Matt 缺乏沟通表示沮丧，敦促其对性能差异保持透明。
   - 该开发者指出，在付出巨大努力后，他们对有关 **Reflection 70B 模型** 的沟通中断感到失望。
- **社区对托管工作的反思**：社区成员反思了托管 **Reflection 70B 模型** 所花费的时间和资源，表示如果近期得不到回复，他们将转向更有生产力的项目。
   - 一位成员总结了这一困境，称 *“Attention is not all you need”*，暗示了从这次经历中吸取的教训。
- **对关于 Matt 的持续讨论感到厌烦**：成员们对关于 Matt 的持续讨论感到日益疲劳，一些人幽默地建议彻底跳过这个话题。
   - 一位成员的评论概括了这种情绪：*“当沉寂的 AI 新闻周期最需要他时，英雄出现了”*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/airesearch12/status/1833572283992146183">来自 Florian S (@airesearch12) 的推文</a>：3. 在为 Reflection 70b 重新配置时，他一定做错了什么，代理仍然指向他之前设置的 Sonnet 3.5。Matt *在不知情的情况下* 在错误的...上运行了基准测试。</li><li><a href="https://x.com/yuchenj_uw/status/1833636690877100488?s=46">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：@mattshumer_ 嗨 Matt，我们花了很多时间、精力和 GPU 来托管你的模型，很遗憾看到你在过去的 30 多个小时里停止了回复，我认为你可以更加透明地说明...</li><li><a href="https://x.com/mattshumer_/status/1833619390098510039">来自 Matt Shumer (@mattshumer_) 的推文</a>：我在宣布这个项目时操之过急了，我很抱歉。那不是我的本意。我根据当时掌握的信息决定发布这种新方法。我知道...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1283140696831692933)** (14 messages🔥): 

> - `Gemini 与 Cursor 集成`
> - `Aider vs Cursor`
> - `API UX 的挫败感`
> - `Anthropic 的 SDK 性能`
> - `Stripe API 订阅问题` 


- **探索 Gemini 与 Cursor 的连接**：成员们讨论了尝试将 **Gemini** 集成到 **Cursor** 中，其中一人认为这是一个**非常有用的 AI 聊天机器人界面**。
   - *Cursor* 获得的评价褒贬不一，一些人对其整体体验表示平平。
- **Aider 受到青睐**：成员们对 **Aider** 的偏好日益增长，其中一人指出，尽管编码存在困难，但他们一直很喜欢它。
   - 对于那些不习惯终端命令的人来说，Aider 似乎很有吸引力，正如正在积极考虑它的成员所指出的那样。
- **对 API UX 的不满**：成员们表达了对围绕 **OpenAI API 格式** 标准化的不满，称其 UX 糟糕且复杂。
   - 一位用户批评了使用该 API 的复杂性，指出简单的任务也需要多个步骤。
- **API 对比：OpenAI vs Google**：讨论中对比了 **OpenAI API** 和 Google 的 API，用户一致认为两者都没有提供直观的体验。
   - 成员们表达了对像 **Stripe** 或 **Python Requests** 这样更符合人体工程学的 API 的渴望，强调了对这两项服务的不满。
- **Anthropic SDK 的现状**：一位成员建议总结一下 **API 的现状** 会很有帮助，重点关注 **Anthropic** 今年波动的 SDK 性能。
   - 评论表明 Anthropic 面临着挑战，用户热衷于跟踪其支持方面的进展。


  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1283401102540275743)** (58 messages🔥🔥): 

> - `Surge AI 合同问题`
> - `数据标注劳动力`
> - `Google 合同工组建工会`
> - `Turing 对比 Scale AI`
> - `针对私有模型的 RLHF` 


- **Surge AI 面临合同交付问题**：据报道，Surge AI 直到受到法律诉讼威胁才向 **HF** 和 **Ai2** 交付数据，这引发了人们对其优先处理较小合同方式的担忧。
   - 延迟期间没有任何沟通，导致人们对其可靠性产生质疑。
- **内部数据标注的挑战**：成员们讨论了将数据标注转为内部进行的困难，因为大多数团队不愿承担此类任务，这凸显了科技公司面临的**业务风险**。
   - 这种情绪呼应了对 **Nvidia** 等公司运营复杂性的担忧，这些公司并未追求内部制造。
- **Google 合同工成功组建工会**：一组负责训练 **Bard AI 聊天机器人** 的 Google 合同工投票以压倒性多数加入 **Alphabet Workers Union**，以寻求更好的工作条件。
   - 这些工人强调，他们的工作任务非常艰巨，其中包括处理**淫秽和冒犯性提示词**。
- **Turing 与 Scale AI 的对比**：关于 Turing 和 Prolific 的讨论强调了它们在数据标注中的作用，以及它们在扩展到编码劳动力之前最初对工程招聘的关注。
   - 虽然大家对行业格局感到好奇，但由于这些公司属于私有性质，很难获得准确的指标。
- **了解私有模型的 RLHF**：成员们表达了对 **Reinforcement Learning from Human Feedback (RLHF)** 在定制模型（尤其是企业场景）中如何运作的了解欲望。
   - 对话指出，虽然 RLHF 旨在根据人类偏好调整模型，但在**材料科学**和**有机化学**等专业领域仍面临挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.theverge.com/2023/11/7/23950392/google-contractors-accenture-obscene-bard-prompts-unionizing">Google contractors objected to reading obscene Bard prompts — now they’re unionizing</a>：更多 Google 合同工正在组建工会。</li><li><a href="https://www.theverge.com/2023/11/7/23950392/google-contractors-accenture">Google contractors objected to reading obscene Bard prompts — now they’re unionizing</a>：更多 Google 合同工正在组建工会。
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1283498082633515191)** (1 messages): 

> - `Perplexity Pro 注册活动`
> - `注册最后倒计时`
> - `学生免费月` 


- **Perplexity Pro 注册活动进入最后阶段**：各校区仅剩 **5 天时间** 达到 **500 人注册**，即可获得一年的免费 **Perplexity Pro**。查看详情并前往 [perplexity.ai/backtoschool](https://perplexity.ai/backtoschool) 注册。
   - *这是最后的冲刺！* 鼓励参与者在截止日期前召集同学进行注册。
- **倒计时器增加紧迫感**：倒计时器强调了紧迫性，显示距离达成注册目标还剩 **05:12:11:10**。这个滴答作响的时钟旨在激励学生迅速行动。
   - 公告附带的视觉效果突出了**限时优惠**，并营造了围绕该活动的兴奋感。
- **为学生提供一个月免费 Perplexity Pro**：学生可以使用学生邮箱注册，以解锁 **一个月免费** 的 **Perplexity Pro**。此优惠旨在激励新用户参与该平台。
   - 注册不仅可以访问高级功能，还有助于达成校区注册目标。



**提到的链接**：<a href="https://perplexity.ai/backtoschool">Perplexity - Race to Infinity</a>：欢迎回到学校！在短短两周内，即可兑换一个月免费的 Perplexity Pro。推荐你的朋友，因为如果你的学校达到 500 人注册，我们将把免费月份升级为一整年免费...

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1283141437122412665)** (86 条消息🔥🔥): 

> - `Perplexity 订阅`
> - `学生优惠`
> - `API 功能`
> - `促销与折扣`
> - `通用用户体验` 


- **学生从 Perplexity 订阅中获益**：目前有一项针对学生的优惠，可以获得一个月的免费 Pro 会员，尽管有人指出这仅限于美国学生或有足够注册人数的特定学校。
   - 成员们对这些优惠中的国际不平等现象表示不满，并提到像德国这样的国家也有促销活动。
- **关于 API 和功能的讨论**：用户讨论了对即将到来的 Dev Day 中新 API 功能的期待，推测将发布 4o 语音和图像生成功能。
   - 还有用户希望为不需要完全 Pro 权限的频繁用户提供 Hobby Tier（爱好者层级）订阅。
- **用户体验反馈**：一些用户在 Perplexity 平台管理多个附件时遇到了问题，特别是在尝试删除特定附件而不丢失全部附件时。
   - 还有关于通用用户体验的其他讨论，将 Perplexity 与竞争对手进行了对比，评价较为正面。
- **促销与折扣讨论**：促销和折扣是一个热门话题，提到了面向不同用户群体的各种优惠券以及特定地区的促销活动。
   - 社区强调了推荐计划（referral program）的存在，但对其可访问性的评价褒贬不一。
- **社区参与和支持**：成员们对社区支持表示感谢，特别是在寻求有关订阅和账户状态的解答时。
   - 讨论展示了一个活跃的社区，在故障排除和功能请求方面有着共同的经验。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1283211580544122930)** (17 条消息🔥): 

> - `Neuralink 患者更新`
> - `SpaceX Starship 2026 年火星目标`
> - `商业太空行走`
> - `情报主管更新`
> - `诸神之战 (Clash of Titans) 见解` 


- **Neuralink 首位患者更新**：Perplexity AI 分享了一个名为 *Neuralink's First Patient Update and SpaceX's Starship Targets Mars 2026* 的 **YouTube 视频**。
   - 该视频提供了关于 **Neuralink** 和 **SpaceX** 的见解，详细阐述了它们的未来雄心。
- **首次商业太空行走成功**：一位用户对 [首次商业太空行走](https://www.perplexity.ai/page/the-first-commercial-spacewalk-AA5WBBNtSMq9DtAFlwFK7w) 页面的呈现效果表示满意。
   - 也有人对页面缺乏 **embeds**（嵌入内容）表示担忧，表明希望看到更多互动内容。
- **情报主管的最新见解**：记录了关于 **Intelligence Chiefs**（情报主管）的更新，并分享了一个便于访问的页面转换链接 [点击此处](https://www.perplexity.ai/page/intelligence-chiefs-sound-alar-.ecnEe0OS8KbZVsHKqXXLQ)。
   - 这突显了关于国家安全和当前情报方法论的持续讨论。
- **诸神之战 (Clash of Titans) 核心要点**：用户将关于“诸神之战”的讨论转换成了页面，强调了重要的收获，特别是 [核心要点 5](https://www.perplexity.ai/page/clash-of-titans-5-key-takeaway-NKVAf71vSR2GNltNQODgpw)。
   - 这反映了将见解汇编成易于理解的格式以供未来参考的协作努力。
- **AI 处于前沿分析**：在此 [搜索查询](https://www.perplexity.ai/search/ai-at-the-forefront-analyzing-qo5TpqUwSSqRzs7B2BSa0w#2) 中强调了关于 **AI** 在当前技术进步中相关性的讨论。
   - 这表明了人们对 **AI** 在各个领域不断演变的角色感兴趣。



**提到的链接**：<a href="https://www.youtube.com/embed/oQjlH0CUTDo">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1283206009464291348)** (2 条消息): 

> - `Bounce.ai`
> - `Perplexity API 使用`
> - `支持请求` 


- **Bounce.ai 的紧急支持请求**：Bounce.ai 的 CTO 兼联合创始人 Aki Yu 就一个影响其拥有超过 **3,000 名活跃用户** 平台的 **Perplexity API** 紧急问题寻求帮助。
   - 尽管在过去的 **4 个月** 里尝试通过各种渠道联系 **Perplexity 团队**，但他们尚未收到回复。
- **需要立即联系**：Aki 强调了他们情况的严重性，请求 **Perplexity 团队** 的成员尽快通过 **yutian@gobounce.ai** 与他们联系。
   - 这凸显了 **Perplexity** 在支持或沟通方面可能存在的缺口，需要加以解决以维持用户信任。

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1283151920776741015)** (74 条消息🔥🔥): 

> - `Llama-3.1-SuperNova-Lite 性能`
> - `模型对比：Hermes vs Llama`
> - `蒸馏技术及其影响`
> - `对 Hermes 3 API 的需求`
> - `训练小型 LLM` 


- **Llama-3.1-SuperNova-Lite 展示了更好的数学能力**：一位成员强调，Llama-3.1-SuperNova-Lite 在处理像 Vedic multiplication 这样的计算时似乎比其他模型表现更好，特别是相比 Hermes-3-Llama-3.1-8B，它在保持数字准确性方面更出色。
   - 尽管两个模型在处理该任务时都比较吃力，但 SuperNova-Lite 在计算过程中保持数字完整性方面表现出优势。
- **模型对比揭示了性能差异**：在测试中，LLaMa-3.1-8B-Instruct 在数学任务上始终表现不佳，而 Llama-3.1-SuperNova-Lite 表现相对较好，这引发了对这些模型之间的对比。
   - 用户表达了对 Hermes-3-Llama-3.1-8B 的强烈偏好，而非 LLaMa-3.1-8B-Instruct，并指出了它们在数学表现上的差异。
- **关于蒸馏技术的讨论**：成员们辩论了在大型语言模型上使用蒸馏方法的有效性和成本，质疑其收益是否超过了所涉及的挑战。
   - 有人对蒸馏模型在处理 out-of-distribution 数据时的局限性表示担忧，因为其训练依赖于精选的数据集。
- **对 Hermes 3 API 的兴趣**：由于 Hyperbolic 等现有服务的结果不尽如人意，有人请求提供 Hermes 3 70B 的 API。
   - 社区注意到 Hermes 3 405B 的托管选项已经可用，但关于 Hermes 3 70B 的信息仍在等待中。
- **探索小型 LLM 训练**：一位成员表示有兴趣使用 Grokadamw 优化器预训练一个小型语言模型（约 10M 参数），并在不同软件之间权衡速度和易用性的选择。
   - 有人对 Transformer trainer 在速度方面相较于 Axolotl 等替代方案的表现表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/MistralAI/status/1833758285167722836">来自 Mistral AI (@MistralAI) 的推文</a>: magnet:?xt=urn:btih:7278e625de2b1da598b23954c13933047126238a&dn=pixtral-12b-240910&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fannoun...</li><li><a href="https://x.com/corbtt/status/1833633946644713582">来自 Kyle Corbitt (@corbtt) 的推文</a>: @mattshumer_ @ArtificialAnlys 关于 Reflection-70B 的最终报告：经过调查，我不相信曾存在过一个能达到声称基准测试结果的模型。我非常不清楚...</li><li><a href="https://x.com/ailozovskaya/status/1833610156745363788">来自 Alina Lozovskaya (@ailozovskaya) 的推文</a>: 🧵 (1/7) 我得到了 mattshumer/ref_70_e3 和 mattshumer/Reflection-Llama-3.1-70B 的结果，包含和不包含系统提示词的情况！简而言之，这些模型的表现都不如 Meta-Llama-3.1-70B...</li><li><a href="https://huggingface.co/spaces/featherless-ai/try-this-model">HF 缺失的推理组件 - featherless-ai 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.meta.ai/?utm_source=ai_meta_site&utm_medium=web&utm_content=AI_nav&utm_campaign=April_mo">Meta AI</a>: 使用 Meta AI 助手完成任务，免费创建 AI 生成的图像，并获得任何问题的答案。Meta AI 构建在 Meta 最新的 Llama 大型语言模型之上，并使用 Emu,...</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B">meta-llama/Meta-Llama-3.1-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/LWDCLS/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF-IQ-Imatrix-Request/tree/main">LWDCLS/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF-IQ-Imatrix-Request at main</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">meta-llama/Meta-Llama-3.1-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.meta.ai/?utm_source=ai_meta_site&utm_medium=web&utm_content=AI_nav&utm_campaign=April_moment">Meta AI</a>: 使用 Meta AI 助手完成任务，免费创建 AI 生成的图像，并获得任何问题的答案。Meta AI 构建在 Meta 最新的 Llama 大型语言模型之上，并使用 Emu,...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1283145355847929937)** (13 messages🔥): 

> - `Quality Data Scaling` (高质量数据缩放)
> - `Best Small Models for Instruction Following` (最佳指令遵循小模型)
> - `Llama-3.1-SuperNova-Lite Launch` (Llama-3.1-SuperNova-Lite 发布)
> - `Models Under 3B Parameters` (3B 参数以下模型)
> - `Open LLM Leaderboard Resources` (Open LLM Leaderboard 资源)


- **高质量数据提升性能表现**：成员们指出，随着模型参数规模的扩大，更高质量的数据能显著增强性能。
   - 持续的反馈表明，使用**高质量数据集**对于获得最佳结果至关重要。
- **关于最佳指令遵循小模型的讨论**：有人询问 7B 以下最适合指令遵循的小模型，并提到了包括 **qlora 3.1 8B** 在内的多种选择。
   - 回复倾向于认为 **Llama 3.1 8B** 是指令任务的一个可行选择，表明了**它在该类别中的有效性**。
- **Llama-3.1-SuperNova-Lite 发布**：分享了一个新模型 [Llama-3.1-SuperNova-Lite](https://huggingface.co/arcee-ai/Llama-3.1-SuperNova-Lite)，这是一个在指令遵循能力方面表现出色的 8B 参数模型。
   - 它是大型 Llama-3.1-405B-Instruct 模型的蒸馏版本，旨在**提高任务效率和有效性**。
- **寻求适用于简单任务的小型模型**：一位成员寻求比 **Llama 3.1 8B** 更小、能够处理简单任务的模型建议。
   - 建议包括 **Mistral 7B**、**Qwen2 7B** 以及 3B 参数以下的可能选项，并请求提供更新的列表。
- **LLM 排名资源共享**：一位成员提到发现了一个 [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)，详细列出了各种模型的性能。
   - 该资源预计将帮助用户对**当前性能最佳的 LLM** 做出明智的决策。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/arcee-ai/Llama-3.1-SuperNova-Lite">arcee-ai/Llama-3.1-SuperNova-Lite · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - a Hugging Face Space by open-llm-leaderboard</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1283346630438948874)** (1 messages): 

> - `Spatial Reasoning` (空间推理)
> - `Neuro-Symbolic AI` (神经符号 AI)
> - `Program Search` (程序搜索)
> - `Program Synthesis` (程序合成)


- **空间推理：最后的希望？**：一位成员表示，**Spatial Reasoning** 可能是推理能力进步的最后希望，可能与 **neuro-symbolic AI**、**program search** 或 **program synthesis** 并列。
   - 他们表达了一种不确定感，询问最近在这些领域是否出现了任何**革命性的进展**。
- **关于近期创新的讨论**：对话强调了对 **Spatial Reasoning** 及相关技术**近期创新**更新的渴望。
   - 成员们渴望了解该研究领域是否取得了任何显著进展。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1283346630438948874)** (1 messages): 

> - `Spatial Reasoning` (空间推理)
> - `Neuro-symbolic AI` (神经符号 AI)
> - `Program Search` (程序搜索)
> - `Program Synthesis` (程序合成)


- **空间推理作为 AI 的最后希望**：一位成员认为 **Spatial Reasoning** 可能是 AI **Reasoning** 能力的终极前沿。
   - 他们建议探索 **neuro-symbolic AI**、**program search** 或 **program synthesis** 作为这些创新的一部分。
- **询问近期进展**：该成员提出了一个问题，即最近在 **Spatial Reasoning** 及相关领域是否出现了任何**革命性的进展**。
   - 这反映了社区对 AI 技术最新趋势的持续关注和好奇。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1283145357068337202)** (83 messages🔥🔥): 

> - `Pixtral 12B model` (Pixtral 12B 模型)
> - `Klarna's SaaS strategy` (Klarna 的 SaaS 策略)
> - `New AI models and tools` (新 AI 模型与工具)
> - `Trieve's funding round` (Trieve 的融资轮次)
> - `Hume's Empathic Voice Interface` (Hume 的共情语音接口)

- **Mistral 大会发布 Pixtral 12B 模型**：Mistral 宣布了 Pixtral 12B 模型，其性能超越了 **Phi 3** 和 **Claude Haiku** 等其他模型。这一消息是在一场由 **Jensen Huang** 出席的受邀制会议上披露的。
   - 该模型支持任意图像尺寸和交错（interleaving），并在活动中展示了重要的基准测试结果。
- **Klarna 将弃用传统 SaaS 供应商**：Klarna 的 CEO 表示，他们正在解雇其 **SaaS 供应商**，包括以前被认为无法替代的系统，这引发了人们对运营风险的关注。
   - 除了切断与 SaaS 的关系外，据报道 Klarna 还裁减了 **50% 的员工**，这可能与财务压力有关。
- **用于 HTML 转 Markdown 的新 AI 模型发布**：Jina AI 推出了两个小型模型 **reader-lm-0.5b** 和 **reader-lm-1.5b**，专门针对将 HTML 干净高效地转换为 Markdown 进行了训练。
   - 这些模型支持多语言且专为高性能设计，在体积显著缩小的同时，性能超越了体量更大的同类模型。
- **Trieve 获得 350 万美元融资**：Trieve AI 宣布成功完成由 Root Ventures 领投的 **350 万美元融资轮**，旨在让各行各业更容易构建 AI 应用。
   - 这笔资金将支持其增长，目前的系统每天已有数万名用户使用。
- **Hume 发布新型共情语音模型**：Hume AI 推出了 **Empathic Voice Interface 2 (EVI 2)**，它结合了语言和语音来训练情感智能。
   - 该模型已开放供用户尝试，并开始构建需要情感交互的应用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/MistralAI/status/1833758285167722836">来自 Mistral AI (@MistralAI) 的推文</a>：magnet:?xt=urn:btih:7278e625de2b1da598b23954c13933047126238a&dn=pixtral-12b-240910&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fannoun...</li><li><a href="https://x.com/thogge/status/1833627582551757143?s=46">来自 tyler hogge (@thogge) 的推文</a>：Klarna CEO 表示他们正在解雇他们的 SaaS 供应商，甚至是那些我们认为不可能被替换的“记录系统 (systems of record)”。全都没了……这太疯狂了。</li><li><a href="https://x.com/patrickc/status/1833648360194265318?s=46">来自 Patrick Collison (@patrickc) 的推文</a>：a16z 制作了这份实用的前 50 名 AI Gen AI Web 产品列表：https://a16z.com/100-gen-ai-apps-3/。我们检查了一下，结果显示 82% 的产品都在使用 @Stripe。我们一直在构建一系列功能...</li><li><a href="https://x.com/swyx/status/1833933507590324483?s=46">来自 swyx.io (@swyx) 的推文</a>：在 Mistral 的受邀制会议上（Jensen Huang 出席）发布了新的 Pixtral 细节，包括 Benchmark！！Pixtral > Phi 3, Qwen VL, Claude Haiku, LLaVA。引用 swyx.io (@swyx)...</li><li><a href="https://x.com/eshamanideep/status/1833759328521867505?s=46">来自 Esha (@eshamanideep) 的推文</a>："dim": 5120, "n_layers": 40, "head_dim": 128, "hidden_dim": 14336, "n_heads": 32, "n_kv_heads": 8, "rope_theta": 1000000000.0, "norm_eps"...</li><li><a href="https://api.ynab.com/v1#/)">YNAB API Endpoints - v1</a>：未找到描述</li><li><a href="https://x.com/wgussml/status/1833615864131948756?s=46">来自 william (@wgussml) 的推文</a>：🚀 我很高兴宣布 Prompt Engineering 的未来：𝚎𝚕𝚕。ell 是根据我在 OpenAI 期间的想法开发的，是一个轻量级、函数式的 LM 编程库：- 自动版本控制与追踪...</li><li><a href="https://x.com/alwayslaunch/status/1833683514090303874?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 aiur (@alwayslaunch) 的推文</a>：引用 tyler hogge (@thogge)：Klarna CEO 表示他们正在解雇他们的 SaaS 供应商，甚至是那些我们认为不可能被替换的“记录系统 (systems of record)”。全都没了……这太疯狂了。</li><li><a href="https://x.com/hume_ai/status/1833906262351974483?s=46">来自 Hume (@hume_ai) 的推文</a>：推出 Empathic Voice Interface 2 (EVI 2)，我们新的语音到语音基础模型。EVI 2 将语言和语音合并为一个专门为情感智能训练的单一模型。你可以尝试...</li><li><a href="https://x.com/code/status/1833249742274314260">来自 Visual Studio Code (@code) 的推文</a>：📣 新的 @code 版本包含最新且最棒的 GitHub Copilot 更新。让我们来看看吧… 🧵</li><li><a href="https://x.com/tom_doerr/status/1833619034425770227?s=46">来自 Tom Dörr (@tom_doerr) 的推文</a>：我最近见过的最令人兴奋的项目之一。这个爬虫可以为你提供页面的截图，允许你选择输出格式（JSON, 清理后的 HTML, Markdown），并且还有大量其他功能...</li><li><a href="https://x.com/danielhanchen/status/1833764749538119921?s=46">来自 Daniel Han (@danielhanchen) 的推文</a>：Mistral 刚刚发布了一个名为 Pixtral 12b 的新视觉多模态模型！还下载了参数 JSON - Vision Adapter 使用了 GeLU 和 2D RoPE。词表大小也变大了 - 131072。此外 Mist...</li><li><a href="https://x.com/TimSuchanek/status/1833538423954804948">来自 Tim Suchanek (@TimSuchanek) 的推文</a>：🚀 在 Stellate 度过了一段美好的时光后，我决定开始一项新业务。我创立了 http://expand.ai，我们正在参加当前的 YC 批次 - S24！对于技术人员：http://expand.ai 瞬间...</li><li><a href="https://x.com/langchainai/status/1833529605262872770?s=46">来自 LangChain (@LangChainAI) 的推文</a>：LangChain Academy 上线了！我们的第一门课程——LangGraph 简介——将教你构建可靠 AI Agent 的方方面面。在本课程中，你将学习如何：🛠️ 使用 LangGraph 构建 Agent...</li><li><a href="https://x.com/jxnlco/status/1833555318590329073?s=46">来自 jason liu (@jxnlco) 的推文</a>：祝贺 http://expand.ai！对于其他人，可以在家尝试 expand ai ;)</li><li><a href="https://x.com/reach_vb/status/1833801060659372071?s=46">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：🚨 新的强大开源 Text to Speech 模型：Fish Speech 1.4 - 在 70 万小时的语音上训练，多语言（8 种语言）🔥 > 即时声音克隆 > 超低延迟 > 约 1GB 模型权重...</li><li><a href="https://x.com/skeptrune/status/1833954889904652737?s=46">来自 skeptrune (@skeptrune) 的推文</a>：我很高兴地宣布 @trieveai 获得了由 Root Ventures 领投的 350 万美元融资！我和 @cdxker 创立 Trieve 是因为我们觉得构建 AI 应用应该更容易。我们正在寻找...</li><li><a href="https://x.com/pelaseyed/status/1833851894260699174">来自 homanp (@pelaseyed) 的推文</a>：1. 写入的成本...</li>

...软件的成本正在趋于零。2. 编写软件所需的技能也正在趋于零。3. 传统的商业模式（例如 SaaS）正在被颠覆...</li><li><a href="https://x.com/jinaai_/status/1833861180445860168?s=46">来自 Jina AI (@JinaAI_) 的推文</a>：宣布推出 reader-lm-0.5b 和 reader-lm-1.5b，https://jina.ai/news/reader-lm-small-language-models-for-cleaning-and-converting-html-to-markdown?nocache=1 这是两个受 J 启发的 Small Language Models (SLMs)，用于清理并将 HTML 转换为 Markdown...</li><li><a href="https://x.com/draecomino/status/1833940572706668934">来自 James Wang (@draecomino) 的推文</a>：Nvidia 首次开始向 AI 芯片初创公司失去市场份额。在过去几个月的每一场 AI 会议的走廊里都能听到这种声音。</li><li><a href="https://x.com/martin_casado/status/1833642258178150402?s=46">来自 martin_casado (@martin_casado) 的推文</a>：呵呵，有人会了解到状态一致性和集成是很困难的。</li><li><a href="https://x.com/reach_vb/status/1833866688254583239">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：端到端语音模型正火热——LLAMA-OMNI 8B - Apache 许可！🔥  &gt; Speech Encoder - Whisper Large v3 &gt; LLM backbone - Llama 3.1 8B Instruct &gt; Speech Decoder - HuBERT (UnitY)  &gt; ...</li><li><a href="https://www.reworkd.ai/">Reworkd AI</a>：端到端网页抓取</li><li><a href="https://www.cnbc.com/2024/08/27/buy-now-pay-later-firm-klarna-swings-to-first-half-profit-ahead-of-ipo.html">先买后付公司 Klarna 在 IPO 前上半年扭亏为盈</a>：Klarna 表示其在 2024 年上半年实现了调整后运营利润，在公司临近备受期待的 IPO 之际实现盈利。</li><li><a href="https://www.npmjs.com/package/@philschmid/clipper">@philschmid/clipper</a>：一个用于从网页剪辑文章并将其保存为 Markdown 文件的 CLI 工具。最新版本：0.2.0，最后发布于 8 个月前。通过运行 `npm i @philschmid...` 开始在你的项目中使用 @philschmid/clipper。
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1283216318123999374)** (36 条消息🔥): 

> - `Open Interpreter 能力`
> - `文档清晰度`
> - `桌面端应用的早期访问`
> - `01 Light 停产`
> - `使用 Open Interpreter 探索硬件` 


- **使用自定义 Python 代码指导 Open Interpreter**：一位成员询问是否可以将特定的 Python 代码作为工具在 Open Interpreter 中使用，以执行情感分析等任务。
   - 人们对确认该功能在数据库更广泛的自定义查询中的可行性表现出兴趣。
- **收到文档反馈**：一位成员表示，虽然他们觉得 Open Interpreter 很酷，但在查阅文档时遇到了问题，感觉文档比较分散。
   - 另一位参与者提出愿意通过建议来帮助改进文档，并欢迎提交 Pull Requests。
- **桌面端应用的早期访问即将开启**：一位用户寻求关于桌面端应用早期访问时间表的信息，并被鼓励留意更新。
   - 据指出，未来几周内可能会包含更多的 Beta 测试人员。
- **退款及 01 Light 项目更新**：一位成员分享说，他们收到了与 01 Light 停产相关的退款，这引发了关于向新的免费 01 应用过渡的讨论。
   - 相关更新确认，所有制造材料将随 01.1 更新一起开源。
- **在 01 停产后探索硬件选项**：一位用户询问在 01 项目停产后，是否有论坛可以探索 Open Interpreter 的硬件可能性。
   - 他们被引导至一个包含相关讨论的特定频道，而他们之前并未看到过该频道。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/hellokillian/status/1833215071880941972">来自 killian (@hellokillian) 的推文</a>：今天我们将停止 01 light，为所有人退款，并推出一款免费的 01 应用。我们还将开源所有的制造材料以及一个重大的 01.1 更新。为什么？为了专注。这个软...</li><li><a href="https://docs.openinterpreter.com/language-models/custom-models">未找到标题</a>：未找到描述</li><li><a href="https://github.com/Textualize/rich">GitHub - Textualize/rich: Rich 是一个用于在终端中实现富文本和精美格式的 Python 库。</a>：Rich 是一个用于在终端中实现富文本和精美格式的 Python 库。 - Textualize/rich
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1283192967577997343)** (45 messages🔥): 

> - `Open Interpreter 安装问题`
> - `移动端 App 需求`
> - `更新 01`
> - `即将推出的桌面端 App`
> - `Open Interpreter 与 01 的区别` 


- **Open Interpreter 安装问题依然存在**：用户对在各种系统（尤其是 Intel MacBook）上安装 **Open Interpreter** 的复杂性表示沮丧。
   - 几位成员正在等待即将推出的**桌面端 App**（目前处于 Beta 测试阶段），以获得更简单的安装选项。
- **移动端 App 需要 Livekit 服务器**：要使用新的**移动端 App**，用户必须安装 **Livekit** 服务器，并在其 .env 文件中为 ElevenLabs 和 Anthropics 等服务设置 API Key。
   - 有人指出，**Ngrok** 仅在需要外网暴露时才需要，在本地网络使用时可以不需要它。
- **通过更新 01 解决问题**：运行 `poetry run 01 --local --qr` 等命令时遇到的问题引发了关于使用 `git pull` 和 `poetry install` 等命令更新 **01** 的讨论。
   - 当问题持续存在时，建议用户克隆一个新的仓库以确保拥有最新版本。
- **对桌面端 App Beta 测试的兴趣**：一位用户询问如何协助即将推出的**桌面端 App** 的 Beta 测试，该 App 承诺简化安装过程。
   - 社区对桌面端 App 简化 Windows 和 Mac 用户安装流程的潜力持乐观态度。
- **关于 Open Interpreter 与 01 的澄清**：用户对 **Open Interpreter** 和 **01** 之间的区别存在困惑。
   - 鼓励成员在专门的频道中发布安装问题，以优化支持流程。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://01.openinterpreter.com/software/server/introduction">选择服务器 - 01</a>：未找到描述</li><li><a href="https://01.openinterpreter.com/software/server/livekit-server">Livekit 服务器 - 01</a>：未找到描述</li><li><a href="https://01.openinterpreter.com/software/configure">配置 - 01</a>：未找到描述</li><li><a href="https://01.openinterpreter.com/software/server/introd">简介 - 01</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1283429473601196033)** (2 messages): 

> - `来自 JSONL 数据的 RAG 上下文`
> - `Pheme News GitHub 仓库`
> - `NER 流程与 Neo4j 加载`
> - `新闻中的信息区分` 


- **RAG 上下文预览**：一位用户展示了为 **RAG** 提供 **JSONL** 数据上下文的初步测试运行预览，目前已针对**新闻 RSS 订阅源**进行了配置。
   - 该流程包括耗时一天的 **NER** 处理，以及将数据加载到 **Neo4j** 中所需的类似时间，之后将制作关于可用性的视频教程。
- **Pheme News 仓库亮点**：推荐的 **GitHub** 仓库 [Pheme-News](https://github.com/CodeAKrome/Pheme-News) 旨在利用 **NLP** 技术区分新闻中的**误导信息/虚假信息/恶意信息（mis/dis/mal-information）**。
   - 它试图追踪参与者及其与全球事件的**互联性**，展示了一种更全面的信息分析方法。
- **鼓励加载示例数据**：一位用户提醒其他人，用于测试目的的示例数据已在 off-topic 频道中分享。
   - 这鼓励大家参与并对提供的资源进行实验。



**提及的链接**：<a href="https://github.com/CodeAKrome/Pheme-News">GitHub - CodeAKrome/Pheme-News: 使用 NLP 区分新闻中的误导/虚假/恶意信息，以整体方式追踪参与者及其与彼此及世界事件的互联性。</a>：使用 NLP 区分新闻中的误导/虚假/恶意信息，以整体方式追踪参与者及其与彼此及世界事件的互联性。- CodeAKrome/Pheme-News

  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1283151631956971531)** (52 messages🔥): 

> - `Cohere 集成项目`
> - `Mistral 视觉模型`
> - `AI 中的人类监督`
> - `Discord FAQ 机器人开发` 


- **用于工单支持的 Cohere 集成**：一位成员计划在 Link Safe 中集成 [Cohere](https://cohere.com)，用于工单支持和文本处理等多种用途。
   - 他们对在持续的实验中结合使用 Cohere 表示兴奋。
- **Mistral 发布视觉模型**：随着 **Mistral** 发布新的视觉模型，成员们反应热烈，引发了关于潜在功能和未来项目的讨论。
   - 一位成员表示有兴趣了解 C4AI 团队是否会开发视觉模型，而另一位成员分享说他们正在开发 'Maya'，还需要几个月的时间。
- **AI 中的长期人类监督**：一位成员强烈认同在未来的 AI 发展中，人类监督可能仍然至关重要。
   - 另一位成员表示赞同，强调 Cohere 的重点应该是使现有功能在企业场景中变得可靠且相关，而不是追求终极的机器智能。
- **Discord FAQ 机器人开发**：一位成员目前正在为 Cohere 构建专门的 **Discord FAQ 机器人**，旨在简化服务器中的沟通。
   - 他们还在组织一场虚拟 Hack 活动，鼓励大家参与并探索新想法。
- **Cohere 社区的互动创意**：讨论激发了关于服务器更多互动用例的头脑风暴，包括正式的 Chatbot 和 AI 地牢探险体验。
   - 成员们互相鼓励，就如何增强社区内的用户互动和参与度进行创造性思考。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1283151897301225635)** (5 messages): 

> - `Aya-101 寿命终止 (End-of-life)`
> - `微调 LLM`
> - `Aya-23 发布` 


- **询问 Aya-101 的未来**：*Aya-101 是否已到寿命终点？* 引起了成员的关注，暗示可能正在向新模型过渡。
   - 同一位成员推测新模型可能会超越竞争对手，并将其称为 *Phi-killer*。
- **探索 Aya-23 的功能**：另一位成员推荐尝试 [Aya 23](https://huggingface.co/collections/CohereForAI/c4ai-aya-23-664f4cda3fa1a30553b221dc)，声称它覆盖的语言范围较小，但 *比 Aya-101 更强大*。
   - 他们提到 Aya-23 提供了微调脚本，感兴趣的用户可以通过其 Hugging Face 模型页面获取。
- **寻求微调资源**：成员们积极寻求关于如何微调 LLM 的建议视频或书籍，表明了对针对性学习资源的渴望。
   - 此外，他们表示有兴趣在发布新模型的同时获取微调相关信息。



**提到的链接**：<a href="https://huggingface.co/collections/CohereForAI/c4ai-aya-23-664f4cda3fa1a30553b221dc">C4AI Aya 23 - a CohereForAI Collection</a>：未找到描述

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1283186572291735686)** (6 messages): 

> - `Cohere API 功能`
> - `API 反馈流程`
> - `JSON 中的多态对象` 


- **Cohere API 仍面临功能问题**：用户对 **Cohere API** 表示沮丧，指出方法不断变化，且对于当前哪些功能正常运行缺乏明确沟通。
   - *一位用户质疑 API 的整体可靠性*，而另一位用户建议缩小具体问题的范围以明确问题。
- **关于 API 体验的反馈**：对用户反馈的确认反映了 Cohere 团队成员希望改善 API 功能整体用户体验的愿望。
   - 建议包括提供有关不可用方法的更明确细节，并在文档中使用反馈表单。
- **请求 min-p 参数**：一位用户请求将 `min-p` 参数作为功能增强添加到 **Cohere API** 中。
   - 该请求是伴随着对现有 API 功能缺乏清晰度的抱怨提出的。
- **JSON 中多态对象的挑战**：一位用户在 **Cohere** 的结构化 JSON 中使用**多态对象**时遇到困难，特别指出缺乏对 `anyOf` 的支持。
   - *分享了两种尝试创建多态结构的方法*，但均被 API 拒绝。


  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1283449816650747985)** (1 条消息): 

> - `AI developer seeking project` 


- **AI 开发者寻找项目**：一位 AI 开发者表达了合作新项目的兴趣，鼓励他人与其联系。
   - *如果有人有项目机会，请随时联系我*。
- **合作邀请**：该消息强调了在 AI 开发社区内进行合作的公开邀请。
   - 鼓励其他人就潜在项目与该成员进行交流。


  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1283302281193984055)** (15 条消息🔥): 

> - `lm-evaluation-harness`
> - `pile-t5 performance`
> - `benchmark paper finalization`
> - `huggingface implementation` 


- **获取 lm-evaluation-harness 的指导**：一位用户寻求使用 **lm-evaluation-harness** 评估 OpenAI **gpt4o** 模型在代码生成数据集 **swe-bench** 上表现的帮助。
   - 他们对提供的任何指导预先表示感谢。
- **Benchmark 论文接近完成**：一位成员提到 **benchmark** 和论文定稿已接近完成，并邀请他人在发布时联系。
   - 另一位成员表示有兴趣在草案可用时查看并提供反馈。
- **关于 pile-t5 代码库和性能的问题**：一位用户询问了用于评估 **pile-t5** 的代码库，指出其性能低于预期的 **google/t5-v1_1-xl**。
   - 他们不确定这是 Hugging Face 的实现问题，还是该模型不适合他们的任务。
- **使用 lm-eval-harness 进行评估**：一位成员确认他们使用了 **lm-eval-harness** 进行评估，并说明是针对完整模型而非仅针对 encoder。
   - 这一澄清帮助用户了解到他们的特定用例尚未经过测试。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1283461234129113210)** (4 条消息): 

> - `Pixtral-12b-240910`
> - `RWKV-7 improvements`
> - `Dynamic state evolution in RWKV-7` 


- **Pixtral-12b-240910 模型发布**：社区分享了 **Pixtral-12b-240910** 模型权重，并指出它是按原样提供的，可能不是最新的，镜像了 Mistral AI 发布的种子文件。
   - 下载链接包含一个 magnet URI，更多详情可在 [Mistral 的 Twitter](https://x.com/MistralAI/status/1833758285167722836) 上找到。
- **RWKV-7 展示了作为 Transformer 杀手的潜力**：RWKV-7 被描述为 **DeltaNet** 的改进版本，其转移矩阵为单位矩阵加低秩矩阵（identity-plus-low-rank），这可能会挑战现有的 Transformer。
   - 一项关于优化 DeltaNet 以实现序列长度并行化的相关研究已在 [arXiv](https://arxiv.org/abs/2406.06484) 上发布。
- **RWKV-7 中的动态状态演化受到关注**：RWKV-7 “Goose” 预览版强调利用结构化矩阵进行 **dynamic state evolution**（动态状态演化），在修复一个隐藏 bug 后，其 loss 曲线看起来更具可扩展性。
   - 正如 [BlinkDL](https://x.com/BlinkDL_AI/status/1833863117480280528) 在其 X 平台上分享的那样，这一增强标志着显著的进展。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/BlinkDL_AI/status/1833863117480280528">来自 BlinkDL (@BlinkDL_AI) 的推文</a>: RWKV-7 &#34;Goose&#34; 预览版，具有动态状态演化（使用结构化矩阵） 🪿 在修复了一个隐藏 bug 后，现在 loss 曲线看起来具有可扩展性😀</li><li><a href="https://x.com/SonglinYang4/status/1833912864203309562">来自 Songlin Yang (@SonglinYang4) 的推文</a>: 据说 RWKV7 是 DeltaNet 的改进版本，其转移矩阵是单位矩阵加低秩矩阵。查看我们之前关于如何跨序列长度并行化 DeltaNet 的工作：https://arx...</li><li><a href="https://huggingface.co/mistral-community/pixtral-12b-240910">mistral-community/pixtral-12b-240910 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1283282110865477752)** (6 messages): 

> - `Chunking datasets` (数据集分块)
> - `Performance rationale in training` (训练中的性能依据)
> - `EOS token usage` (EOS token 的使用)


- **数据集分块是标准做法吗？**: 一位用户询问将数据集拆分为 **128-token chunks** 进行训练是否为标准做法，并指出了这可能对模型输入上下文产生影响。
   - 回复建议，关于分块的决定可能源于**直觉推测**，而非经验研究。
- **性能收益 vs. 上下文损失**: 另一位用户强调，这是一种常见做法，但不一定是因为有优化性能的研究支持，往往是因为缺乏可靠的依据。
   - 对话暗示，许多人可能在没有考虑分块对模型性能真实影响的情况下使用它。
- **在分块中引入 EOS token**: 一位成员提到在将数据分块为窗口长度之前，使用 **<EOS> token** 作为分隔符，但该方法缺乏强有力的依据。
   - 他们将这种做法比作报纸上的文本分割，结论是虽然位置效应可能会被平均化，但这仍然不是确凿的证据。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/)** (1 messages): 

rimanv_51850: 我正在为一个任务准备 pull request，如果可以的话，其中将包含该修复补丁。
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1283357568177148024)** (3 messages): 

> - `image-text multimodal LLM positioning` (图文多模态 LLM 定位)
> - `pixtral` 


- **图像文本 Token 的定位**: 一位成员询问是否有关于图像-文本多模态 LLM 中 **image tokens** 相对于 **text tokens** 最佳位置的研究。
   - *zaptrem* 回复称，**attention 是位置不变的 (position invariant)**，这意味着位置并不重要，除非是在微调一个可能已经学到了近因偏差 (recency bias) 的现有模型。
- **对新 Pixtral 的期待**: 一位成员对 **new pixtral** 的能力表示热烈期待，认为这是一个很酷的进展。
   - 讨论暗示了积极的反响，但未提供其功能的具体细节。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1283519909283762338)** (3 messages): 

> - `Multinode training` (多节点训练)
> - `DDP across nodes` (跨节点 DDP)
> - `Global batch size impact` (全局批大小的影响) 


- **多节点训练性能担忧**: 一位用户询问了在 **8xH100** 机器之间通过较慢的以太网连接进行**多节点训练**的经验，特别是关于 DDP 的功能性。
   - 有人担心性能可能会因链路速度而受到显著影响，从而影响训练时间。
- **DDP 见解**: 讨论了在单节点内训练模型但在多个节点之间使用 **DDP** 的可能性。
   - 建议虽然可行，但由于连接速度较慢，效率可能会降低。
- **Batch Size 建议**: 一位成员建议增加 **train_batch_size** 以饱和 **VRAM**，从而优化训练效率。
   - 有人指出，在 **Pythia** 预训练期间，**4M-8M tokens** 的全局批大小被确定为有效收敛的阈值。
- **训练技术讨论**: 强调了提高批大小以增加节点间梯度规约 (gradient reductions) 间隔时间的重要性。
   - 该用户表示在使用 **1-bit Adam** 和 **topk grad sparsification** 等技术时未获成功，表明在优化方面仍面临挑战。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1283144295561433131)** (4 messages): 

> - `RAG 课程`
> - `检索增强生成 (Retrieval-Augmented Generation) 教程`
> - `用于文档问答的 Kotaemon UI`
> - `AI 调度器工作坊` 


- **LLM 时代的 RAG Maven 课程**：查看名为 [Search For RAG in the LLM era](https://twitter.com/llama_index/status/1833584685664067833) 的 **Maven 课程**，由 **@jerryjliu0** 担任客座讲师，并包含现场代码演示。
   - 与行业资深人士一起*参与代码示例和实现*，以提升你的学习体验。
- **构建 RAG 的快速教程**：现已推出一个使用 LlamaIndex [构建检索增强生成的快速教程](https://twitter.com/llama_index/status/1833611545370366281)。
   - 它提供了一种有效实现 RAG 技术的简单方法。
- **Kotaemon：构建基于 RAG 的文档问答系统**：了解如何使用 [Kotaemon](https://twitter.com/llama_index/status/1833907464355647906) 构建*基于 RAG 的文档问答系统*，这是一个用于与文档聊天的开源 UI。
   - 涵盖的主题包括设置可定制的 RAG UI 以及组织 **LLM & embedding models**。
- **实战 AI 调度器工作坊**：参加 9 月 20 日在 AWS Loft 举办的工作坊，使用 **Zoom**、**LlamaIndex** 和 **Qdrant** *构建用于智能会议的 AI 调度器*。
   - 参与者将创建一个用于提高会议效率的 RAG 推荐引擎，并利用 **Zoom 的转录 SDK** 进行分析。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1283159881909538969)** (24 messages🔥): 

> - `用于构建索引的任务队列`
> - `QueryPipeline 的 run_multi_with_intermediates`
> - `在 ChromaDB 中保存向量`
> - `LlamaIndex 中的内存管理`
> - `使用不同的 LLM 提供商` 


- **探索任务队列设置**：发起了关于使用 **FastAPI** 配合 **Celery 后端**和数据库来存储文件及索引信息，从而创建构建索引的任务队列的讨论。
   - 有建议提出检查现有的、可能满足这些要求的设置。
- **QueryPipeline 的 run_multi_with_intermediates**：询问了如何在 Workflows 中实现与 **QueryPipeline 的 run_multi_with_intermediates** 类似的功能，强调了其在查询执行后检查结果的实用性。
   - 该方法被确认为查看中间结果的有效手段，一些用户分享了处理 Workflows 的编码技巧。
- **存储来自 Semantic Chunker 的向量**：提出了一个关于将 **Semantic Chunker** 的向量保存到 **ChromaDB** 的问题，并指出目前尚无此选项。
   - 为了存储向量，建议用户对 Semantic Chunker 进行子类化，并编写自定义代码将向量推送到数据库解决方案中。
- **在 LlamaIndex 中管理内存**：用户讨论了在使用 **LlamaIndex** 时内存管理的实现策略，详细介绍了如何使用 **ChatMemoryBuffer** 来存储聊天消息。
   - 分享的最佳实践包括如何在检索先前消息历史以获取上下文的同时追加新消息。
- **集成替代 LLM 提供商**：提供了关于如何使用 LlamaIndex 从 **OpenAI** 切换到其他 LLM 提供商（如 **Ollama**）的说明，从而提高了集成的灵活性。
   - 分享了设置和使用替代 LLM 的资源，包括一个用于实际演示的 **Colab notebook** 链接。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/understanding/workflows/stream/">Streaming events - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/ollama/">Ollama - Llama 3.1 - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1283168631467020419)** (15 messages🔥): 

> - `使用 LLM 进行查询生成`
> - `连接年轻企业家`
> - `剥离 LLM 响应内容`
> - `构建 RAG 应用`
> - `Upstash Redis Memory 调试` 


- **构建查询生成的 POC**：一位成员正在开发一个通过 LLM 使用 **LangGraph** 进行**查询生成**的 POC，面临着随着表数量增加而导致 token 大小增加的挑战。
   - 他们实现了 **RAG** 将 schema 转换为向量以生成查询，并寻求其他潜在解决方案，同时表示不愿增加额外的 LLM 调用。
- **组建志同道合的团队**：一位成员表示有兴趣组建一个由**充满抱负的 AI 爱好者、程序员和企业家**组成的团体，进行旨在产生影响的每日头脑风暴会议。
   - 他们强调了积极参与的必要性，并表示：“让我们行动起来，不要再等待其他人或公司去做你希望自己已经完成的事情。”
- **剥离 LLM 响应的输出**：在询问如何剥离响应以仅显示 LLM 输出时，一位成员被引导至 LangChain 提供的 **StrOutputParser**。
   - 该解析器旨在从 LLM 结果中提取可能性最高的字符串，从而实现更整洁的输出。
- **构建 RAG 应用的见解**：一位 **RAG** 应用的新手询问在将通过网络加载器获取的文本存储到向量数据库之前，是否应保留或删除换行符。
   - 另一位成员建议没有必要删除**换行符**，这意味着保留格式是可以的。
- **调试 Upstash Redis Memory**：一位用户提出了在使用 **ChatTogetherAI** 时遇到的 **Upstash Redis** 历史记录 bug，并指出切换到 **ChatOpenAI** 可以解决该问题。
   - 这表明库中可能存在需要进一步调查的潜在兼容性问题。



**提及链接**：<a href="https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html">langchain_core.output_parsers.string.StrOutputParser &mdash; 🦜🔗 LangChain 0.2.16</a>：未找到描述

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1283327546267664414)** (2 messages): 

> - `OppyDev 更新`
> - `优惠码`
> - `插件系统`
> - `RAG 系统`
> - `实时代码审查` 


- **OppyDev 重大更新发布**：团队宣布了其 **AI 辅助编程工具** ***OppyDev*** 的重大更新，通过简化与 AI 的交互来增强开发者体验。
   - 新功能包括针对 Mac 和 Windows 的简易安装，以及与 **GPT-4** 和 **Llama** 等多个 LLM 的集成。
- **提供专属优惠码**：他们限时向用户提供优惠码，可访问拥有 **100 万免费 GPT-4 token** 的订阅账户。
   - 感兴趣的用户可以私信获取优惠码，以开始使用新版本的 OppyDev。
- **引入全新的插件系统**：此次更新包含一个**全新的插件系统**，允许用户构建自定义工具并增强 **OppyDev** 的功能。
   - 在 [文档](https://oppydev.ai/documentation/#plugins) 中了解更多关于插件功能的信息。
- **用于增强编程任务的 RAG 系统**：OppyDev 利用 **RAG 系统**，允许开发者询问有关其代码库的问题，并跨文件管理多步骤编程任务。
   - 这有助于弥合开发与 AI 辅助之间的差距，提高效率。
- **实时代码审查功能**：新更新包括**颜色标记、可编辑的 diff** 功能，可在编程过程中进行实时更改审查。
   - 此功能确保开发者能够有效地监控其编程进度。



**提及链接**：<a href="https://oppydev.ai/documentation/#plugins.">Documentation - OppyDev</a>：观看我们的入门视频，详细了解如何使用 OppyDev 的 AI Agent 驱动的编程助手。

  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1283363921926815829)** (4 messages): 

> - `Torchtune FP16 Support`
> - `Qwen2 Interface Discrepancies`
> - `EOS ID Handling` 


- **Torchtune 不支持 FP16**：一位成员解释说，在混合精度模块与其他特性之间保持兼容性需要额外的工作，而 **bf16** 半精度训练被认为更优越。
   - 缺乏 FP16 支持的主要缺点是它与旧款 GPU 不兼容，这可能会影响部分用户。
- **Qwen2 接口具有不同的 Tokenization 行为**：一位成员指出，Qwen2 接口允许为 `eos_id` 传递 `None`，从而在 `encode` 方法中添加它之前进行 `None` 检查。
   - 然而，这可能是一个疏忽，因为代码中的另一个案例没有执行此检查，这引发了关于此行为是故意为之还是 Bug 的潜在疑问。
- **允许 None EOS ID 似乎存在问题**：有人建议，不应明确允许在 `eos_id` 设置为 `None` 的同时设置 `add_eos=True`。
   - 这引发了对 Qwen2 模型内 Tokenization 过程的一致性和预期行为的担忧。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/main/torchtune/models/qwen2/_tokenizer.py#L161)">torchtune/torchtune/models/qwen2/_tokenizer.py at main · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/models/qwen2/_tokenizer.py#L373).">torchtune/torchtune/models/qwen2/_tokenizer.py at main · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1283454746346131629)** (5 messages): 

> - `padded_collate utility`
> - `ppo recipe modifications` 


- **关于 padded_collate 用法的查询**：一位成员注意到 `padded_collate` 工具函数没有在任何地方被使用，并询问了其预期用途，特别是关于 **padding direction 参数**。
   - 他们指出 `padded_collate_sft` 中存在逻辑问题，提到缺少 **匹配 input_ids 和 labels 序列长度** 的逻辑。
- **关于 ppo recipe 的澄清**：另一位成员认为他们已经将 `padded_collate` 逻辑添加到了 **ppo recipe** 中，并询问缺少了什么。
   - 这引发了关于 **padded_collate** 实现的当前状态，以及 len(input_ids) 和 len(labels) 通常是否匹配的讨论。



**提及的链接**：<a href="https://github.com/pytorch/torchtune/blob/eb92658a360d7a7d4ce1c93bbcf99c99a2e0943b/torchtune/data/_collate.py#L204">torchtune/torchtune/data/_collate.py at eb92658a360d7a7d4ce1c93bbcf99c99a2e0943b · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。

  

---



### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1283353462108721235)** (2 messages): 

> - `Sci Scope Tool`
> - `Evaluating AI Outputs` 


- **介绍用于 Arxiv 论文摘要的 Sci Scope**：开发了一个名为 **Sci Scope** 的新工具，用于按相似性对最新的 Arxiv 论文进行分组，并使用 LLM 生成简明摘要，该工具在[网站](https://www.sci-scope.com/)上免费托管。
   - 用户可以注册以直接在收件箱中接收所有 AI 研究的**每周摘要**，并提供源列表以供深入探索。
- **关于输出真实性的查询**：一位成员对该工具如何确保输出的**真实性**并最大限度地减少生成的摘要中的幻觉表示感兴趣。
   - 这引发了关于 AI 生成内容的可靠性以及评估此类输出背后的方法论的重要讨论。



**提及的链接**：<a href="https://www.sci-scope.com/">Sci Scope</a>：一份关于 AI 研究的 AI 生成报纸

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1283507812248457359)** (2 条消息): 

> - `DSPy Customizations`
> - `Dynamic Prompting Techniques` 


- **针对 DSPy 的客户特定定制化**：一名成员提出了一个关于如何将**客户特定定制化**集成到 DSPy 生成的简单聊天机器人提示词中的问题，且无需将客户信息硬编码到模块中。
   - 他们正在考虑使用**后处理步骤 (post-processing step)** 来进行动态适配，并寻求关于实现定制化的更好方法的建议。
- **表达对帮助的感谢**：一名成员感谢另一名成员提供了他们正在寻找的关于 DSPy 的信息。
   - 这次交流凸显了小组的协作精神，成员们在咨询中积极地互相支持。


  

---



### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1283416010816225313)** (3 条消息): 

> - `Running audio models with tinygrad`
> - `Whisper example`
> - `Getting help online` 


- **探索使用 Tinygrad 运行音频模型**：一位用户寻求关于如何使用 **tinygrad** 运行音频模型的指导，特别是希望寻找该仓库中现有 **Whisper** 示例之外的内容。
   - 这一询问引发了关于在 tinygrad 中探索音频应用潜在切入点的建议。
- **学习的哲学方法**：一位成员引用了 *“千里之行，始于足下”*，强调了直觉在学习过程中的重要性。
   - 这种观点鼓励对社区内的资源进行反思性探索。
- **链接到有用的资源**：另一位成员分享了 Eric S. Raymond 编写的 [提问的智慧 (smart questions)](http://www.catb.org/~esr/faqs/smart-questions.html) FAQ 链接，其中概述了在线寻求帮助的礼仪和策略。
   - 该资源可作为撰写有效查询并最大限度获得社区协助的指南。



**提到的链接**：<a href="http://www.catb.org/~esr/faqs/smart-questions.html">How To Ask Questions The Smart Way</a>：未找到描述

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1283423700431142954)** (1 条消息): 

> - `Mistral's Pixtral Release`
> - `Multi-modal Support`
> - `New Message Structure` 


- **对 Mistral 的 Pixtral 发布感到兴奋**：一名成员提到正在致力于推动**多模态支持 (multi-modal support)** 的更改，这与最近发布的 **Mistral Pixtral** 相契合。
   - *考虑到当今的技术进步，这是一个具有先见之明的举动*。
- **正在进行中的新消息结构提案**：已经创建了一个 Pull Request，旨在在 **Axolotl** 项目中引入一种新的消息结构，以增强消息的表示。
   - 查看 [wip add new proposed message structure](https://github.com/axolotl-ai-cloud/axolotl/pull/1904) 的详细信息以获取更多见解。



**提到的链接**：<a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1904">wip add new proposed message structure by winglian · Pull Request #1904 · axolotl-ai-cloud/axolotl</a>：未找到描述

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1283197900868030474)** (1 条消息): 

> - `Speed/Performance of LLM Models`
> - `September 2024 LLM Testing` 


- **LLM 模型速度与性能竞赛**：最近发布的 [YouTube 视频](https://youtu.be/w6CJtAlGygQ?si=0MzkKj5m2MUiSN59) 标题为 “Who?”，测试了截至 2024 年 9 月领先的 LLM 模型的速度和性能。
   - 演示者旨在揭示哪种模型在**每秒 Token 数 (tokens per second)** 方面脱颖而出，展示了最新的 **SOTA** 技术。
- **分析 LLM 性能指标**：在讨论的视频中，测试重点关注了定义目前可用领先 AI 模型速度和效率的各种指标。
   - 测量的关键方面包括**延迟 (latency)** 和**吞吐量 (throughput)**，这对于任何评估生产环境下性能的人来说都至关重要。



**提到的链接**：<a href="https://youtu.be/w6CJtAlGygQ?si=0MzkKj5m2MUiSN59">Who ?</a>：在这段视频中，我们将测试 2024 年 9 月全球领先的 LLM 模型在速度和性能方面的表现。#tokensperseconds #GPT4o #LLM #SOTA #Cl...

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1283191466096726086)** (2 messages): 

> - `NYX 模型开发`
> - `AI 协作`
> - `大模型的数据来源` 


- **寻求 NYX 模型协作**：一位 AI 开发者目前正在开发拥有超过 **6000 亿参数**的 **NYX 模型**，并正在寻找充满热情的合作伙伴来共同改进它。
   - *让我们聊聊吧！* 如果你有 AI 方面的经验，并且处于适合协作的时区。
- **关于大模型训练资源的疑问**：询问了训练 **600B 参数模型**所需的资源，并特别参考了在 **15 万亿 tokens** 上训练的 **LLaMA-405B**。
   - 好奇心集中在如何获取此类大模型的数据，表明了对所用方法论的兴趣。


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1283529301815459860)** (1 messages): 

> - `Literal AI 易用性`
> - `LLM 可观测性`
> - `LLM 评估`
> - `LLM 监控`
> - `LLM 集成` 


- **Literal AI 在易用性方面表现出色**：一位用户对 [Literal AI](https://literalai.com/) 的易用性表示高度认可，赞赏其为 LLM 应用提供的友好界面。
   - 这种热情凸显了对以用户为中心的 LLM 工具日益增长的需求。
- **强调全生命周期健康的 LLM 可观测性**：强调了 LLM 可观测性的重要性，指出它能让开发者和 Product Owners 快速迭代并调试问题。
   - 利用日志可以帮助微调更小的模型，在降低成本的同时潜在地提升性能。
- **追踪 Prompt 性能以降低风险**：监控 Prompt 性能对于确保在部署新 Prompt 版本之前不发生回归至关重要。
   - 这种持续评估的方法是防止 LLM 应用出现问题的保障。
- **LLM 监控与分析：生产环境成功的关键**：建立 LLM 日志和评估体系对于监控生产环境中 LLM 系统的性能至关重要。
   - 有效的分析使团队能够保持监督并提高应用的可靠性。
- **与 Literal AI 的无缝集成**：Literal AI 允许轻松集成到应用中，连接整个 LLM 生态系统。
   - 此外，它还提供自托管选项，这对于欧盟用户或处理敏感信息的用户至关重要。



**提到的链接**：<a href="https://literalai.com/">Literal AI - RAG LLM 可观测性与评估平台</a>：Literal AI 是专为开发者和 Product Owners 构建的 RAG LLM 评估与可观测性平台。 

  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1283463711809343650)** (1 messages): 

> - `AI 中的 Ground Truth 数据`
> - `Mozilla Fellowship 资助` 


- **理解 Ground Truth 数据在 AI 中的作用**：一篇新博文强调了 **Ground Truth 数据**在 AI 应用中的重要性，讨论了其在提高**模型准确性**和可靠性方面的关键作用。鼓励读者在讨论帖中分享想法：[加入讨论](https://discord.com/channels/1089876418936180786/1283463258635898922)。
- **Mozilla 开放校友资助申请**：Mozilla 邀请之前的 Mozilla Fellowship 或 Awards 参与者申请[项目资助](https://foundation.mozilla.org/en/blog/mozilla-opens-call-for-alumni-connection-grant-applications/)，旨在解决**可信 AI** 问题并促进更健康的互联网。Hailey Froese 强调了 AI 领域进行**结构性变革**的必要性，以便在利用其益处的同时减轻其危害。
   - *“互联网，尤其是人工智能 (AI)，正处于一个转折点。”*


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1283347588170649621)** (1 条消息): 

> - `评估脚本错误`
> - `API 凭据问题`
> - `Urban Dictionary API 连接问题` 


- **openfunctions_evaluation.py 脚本中遇到错误**：用户运行了带有 `--test-category=non_live` 参数的 `openfunctions_evaluation.py` 脚本，但在结果文件夹中没有得到任何分数。
   - 他们尝试使用新的 API 凭据运行 `eval_runner.py`，但反而遇到了问题。
- **对 function_credential_config.json 的补充**：在申请了另外四个 API 地址后，他们将其填入 `function_credential_config.json` 作为设置的一部分。
   - 然而，这一步并没有解决问题，因为他们在运行评估时遇到了进一步的错误。
- **Urban Dictionary API 连接超时错误**：运行评估导致了与 Urban Dictionary API 相关的 `requests.exceptions.ConnectionError`，特别是针对术语 'lit'。
   - 错误消息指出由于超时无法建立连接，暗示可能存在网络问题。


  

---



---



---



---



{% else %}


> 完整的频道细分内容已针对电子邮件进行了截断。 
> 
> 如果您想查看完整的细分内容，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}