---
companies:
- google-deepmind
- openai
- apple
date: '2024-12-12T03:16:07.864299Z'
description: '**Google DeepMind** 推出了 **Gemini 2.0 Flash**，这是一款性能超越 Gemini 1.5 Pro
  和 o1-preview 的新型多模态模型，具备视觉和语音 API、多语言能力以及原生工具调用功能。它为 **Project Astra** 和 **Project
  Mariner** 等新型 AI 智能体提供动力，其中 Project Mariner 在 WebVoyager 基准测试中达到了 **83.5%** 的业界领先水平。**OpenAI**
  宣布了 ChatGPT 与 **Apple** 设备的集成，实现了 Siri 接入和视觉智能功能。**Claude 3.5 Sonnet** 被指出是 Opus
  的蒸馏版本。AI 社区在 **NeurIPS 2024** 上的反响非常积极，标志着谷歌在 AI 创新领域的强势回归。关键主题包括**多模态**、**智能体开发**、**多语言性**、**基准测试**和**模型发布**。'
id: f85cc97e-71fc-4e40-a6fc-93604214255a
models:
- gemini-2.0-flash
- gemini-1.5-pro
- gemini-exp-1206
- claude-3.5-sonnet
- opus
original_slug: ainews-google-wakes-up-gemini-20-et-al
people:
- demis-hassabis
- sundar-pichai
- paige-bailey
- bindureddy
title: '以下是几种不同语气的翻译供你参考：


  *   **标准直译：** 谷歌觉醒：Gemini 2.0 及其他

  *   **更具冲击力（新闻标题风）：** 谷歌发力：Gemini 2.0 及其系列产品

  *   **意译：** 谷歌苏醒：Gemini 2.0 等重磅发布'
topics:
- multimodality
- agent-development
- multilinguality
- benchmarking
- model-releases
---

<!-- buttondown-editor-mode: plaintext -->**TPUs are all you need.**

> 2024年12月10日至12月11日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discords（**207** 个频道，**6549** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**649 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

这是 NeurIPS 的第一天会议，正如[之前通过各种 Gemini-Exp 版本所预热的那样](https://x.com/scaling01/status/1865086810214289910?s=46)，[Sundar Pichai 强势推出了](https://x.com/sundarpichai/status/1866868228141597034) Google 的首个官方 Gemini 2 模型 —— [Gemini Flash](https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/#building-responsibly)。
没人预料到 2.0 Flash 能击败 1.5 Pro，但事实确实如此：


![image.png](https://assets.buttondown.email/images/cd6e7365-5104-45a7-bd61-67aae2763573.png?w=960&fit=max)


它在 LMArena 上也[击败了 o1-preview](https://x.com/lmarena_ai/status/1866873983569891378)（但仍落后于 Gemini-Exp-1206，即疑似的 2.0 Pro 模型）。

价格是“免费”的 —— 只要 2.0 Flash 仍处于实验阶段。仿佛这还不够，2.0 Flash [还发布了多模态（视觉和语音）API](https://x.com/officiallogank/status/1866873298027446465?s=46&t=PW8PiFwluc0tdmv2tOMdEg)，[Paige Bailey 甚至顺道参加了](https://x.com/swyx/status/1866958171560173966)今天的 [Latent Space LIVE](https://x.com/saranormous/status/1866933642401886707)/[Thrilla on Chinchilla](https://x.com/dylan522p/status/1866630813074461060) 活动，展示了它如何实现 OpenAI 今天不敢发布的功能：


![image.png](https://assets.buttondown.email/images/8252a1dc-f456-48fa-9f22-f2d8d01463da.png?w=960&fit=max)


图像**输出**（Image output）也经过了训练并进行了[预热](https://www.youtube.com/watch?v=7RqFLp0TqV0)但尚未发布，但它可以以你从未见过的方式[“画出猫头鹰的其余部分”](https://x.com/m__dehghani/status/1866921587322261998?s=46)（draw the rest of the owl）。

他们还宣布了一系列限量预览的功能：

- [Deep Research](https://x.com/sundarpichai/status/1866868489140772928)：“一个可以深入研究复杂主题并为您创建包含相关来源链接报告的研究助手。”
- [Project Mariner](https://x.com/sundarpichai/status/1866868770678988850)：一个浏览器 Agent，“能够理解并推理浏览器屏幕上的各种信息 —— 像素、文本、代码、图像 + 表单 —— 然后利用这些信息为您完成任务”，在 WebVoyager 基准测试中达到了 83.5% 的 SOTA。
- [Project Astra 更新](https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/#project-astra)：多语言支持、新的工具使用（tool use）、10 分钟的会话记忆、流式/原生音频延迟。
- [Jules](https://developers.googleblog.com/en/the-next-chapter-of-the-gemini-era-for-developers/)，一个将使用 Gemini 2.0 的实验性 AI 驱动代码 Agent。Jules 异步工作并集成到您的 GitHub 工作流中，在您专注于真正想要构建的内容时，处理 Bug 修复和其他耗时的任务。Jules 创建全面的多步骤计划来解决问题，高效地修改多个文件，甚至准备 Pull Request 以将修复直接提交回 GitHub。

NeurIPS 现场以及线上 X/Reddit/Discord 上的每个人评论和印象都非常正面。Google 强势回归！


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

以下是按相关类别整理的关键讨论：

**重大模型发布与更新**

- **Gemini 2.0 Flash 发布**：[@demishassabis 宣布了](https://twitter.com/demishassabis/status/1866872643615592544) Gemini 2.0 Flash，其性能超越 1.5 Pro 且速度快两倍，具备原生 Tool Use、多语言能力以及包括图像生成和 Text-to-Speech 在内的全新多模态功能。该模型将驱动 Project Astra 和 Project Mariner 等新的 Agent 原型。

- **ChatGPT + Apple 集成**：[OpenAI 宣布](https://twitter.com/OpenAI/status/1866943282795938013) ChatGPT 已集成到 iOS、iPadOS 和 macOS 的 Apple 体验中，允许通过 Siri、视觉智能功能和写作工具进行访问。

- **Claude 性能**：[@scaling01 指出](https://twitter.com/scaling01/status/1866768283992531098) Claude 3.5 Sonnet 似乎是 Opus 的蒸馏版本，且 [Opus 训练完成已得到确认](https://twitter.com/scaling01/status/1866767823005159506)。

**行业发展与分析**

- **Google 的进展**：[多位研究人员观察到](https://twitter.com/bindureddy/status/1866877034108190985) Google 的进步，Gemini 2.0 Flash 表现强劲，但指出其尚未达到生产就绪状态。该模型在 SWE-bench 等基准测试中取得了令人印象深刻的成绩。

- **竞争动态**：[@drjwrae 强调](https://twitter.com/drjwrae/status/1866912496327831591)，虽然 1.5 Flash 因其性价比而受欢迎，但 2.0 带来的性能已匹配或超过 1.5 Pro。

- **商业影响**：关于市场动态的讨论，[@saranormous 指出](https://twitter.com/saranormous/status/1866952426890166372) AI 行业的博弈才刚刚开始，并将其与互联网发展历经数十年才尘埃落定的过程进行了类比。

**研究与技术进展**

- **NeurIPS 会议**：多位研究人员分享了来自 #NeurIPS2024 的更新，包括 [@gneubig 关于](https://twitter.com/gneubig/status/1866691769150116273) Agent、LLM + 神经科学以及 Alignment 的演讲。

- **LSTM 讨论**：[@hardmaru 分享了](https://twitter.com/hardmaru/status/1866896953730273698) Sepp Hochreiter 关于 xLSTM 在推理速度和参数效率方面优于 Attention Transformer 的主题演讲。

**幽默与梗**

- **行业评论**：[@nearcyan 提到](https://twitter.com/nearcyan/status/1866931118521581822)：“Twitter 对 OpenAI 的 12 天发布活动感到惊艳的唯一方式，就是它以 GPT-5 开始并以 GPT-17 结束。”

- **AI 模型命名**：关于 AI 模型命名惯例的讨论，[各种幽默的观点](https://twitter.com/nearcyan/status/1866939351592210573) 针对不同公司的做法发表了看法。

---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. Gemini 2.0 Flash 的成就与对比**

- **[Gemini 2.0 Flash 在 SWE-Bench 上击败 Claude Sonnet 3.5 出乎我的意料](https://i.redd.it/xn57o94tw96e1.png)** ([得分: 287, 评论: 53](https://reddit.com/r/LocalLLaMA/comments/1hc276t/gemini_20_flash_beating_claude_sonnet_35_on/))：**Gemini 2.0 Flash** 在 **SWE-bench Verified** 基准测试中达到了 **51.8%** 的性能，超过了 **Claude Sonnet 3.5** 的 **50.8%**。**GPT-4o** 和 **o1-preview** 等其他模型的得分在 **31.0%** 到 **41.0%** 之间，显示出显著的性能差距。
  - **Scaffolding 与测试方法**：讨论强调了 **Scaffolding** 在模型性能中的重要性，一些用户指出 **Gemini 2.0 Flash** 利用多种采样方法来获得高分，而不像 **Claude Sonnet 3.5** 被认为是一个更直接的模型。争论点在于，由于测试方法论的差异（例如使用数百个样本对比 Single-shot 方法），模型之间的比较是否公平。
  - **Context Window 与性能**：**Gemini 2.0 Flash** 因其更大的 Context Window 而受到关注，一些用户认为这使其优于 **Claude** 和 **o1** 等模型。这种能力被视为处理现实世界软件工程任务的关键，促成了其在 **SWE-bench Verified** 基准测试中的高分。
  - **行业观点与担忧**：关于 **Google** 和 **OpenAI** 等公司在 AI 领域主导地位的更广泛讨论，用户对这种控制权带来的影响表示担忧。一些人因为 **Google** 对开源项目的贡献而更青睐它，而另一些人则担心 **OpenAI** 等公司快节奏的开发方式。

- **Gemini Flash 2.0 experimental** ([Score: 141, Comments: 53](https://reddit.com/r/LocalLLaMA/comments/1hbw529/gemini_flash_20_experimental/)): **Gemini Flash 2.0** 正在被讨论，这与 **Sundar Pichai** 通过其 [Twitter 帖子](https://x.com/sundarpichai/status/1866868228141597034?s=46) 宣布的一项实验性更新有关。该更新据推测包含新功能或改进，尽管帖子中未提供具体细节。
  - **Gemini 2.0 Flash 性能**: 该模型在 **natural code 上表现出 92.3% 的准确率**，取得了显著进步，相比 **1.5 Pro** 版本提升了 **7%**，使 Google 成为 OpenAI 的强力竞争对手。然而，与之前的 1.5 Flash 模型相比，它在 MRCR 长上下文基准测试中的表现较差，这表明在通用改进与特定能力之间存在权衡。
  - **API 与使用**: 用户可以通过 **Google AI Studio** 以“按需付费”模式访问该模型，费用为 **每 1M token 输入 $1.25** 以及 **每 1M token 输出 $5**。AI Studio 中有 **每日 1500 次回复的限制**，部分用户遇到了 **QUOTA_EXHAUSTED** 问题，这可能是由于 API key 的配置导致的。
  - **市场与未来预期**: 用户对具有增强多模态能力的 **Gemma 3** 充满期待，反映了用户对未来发展的兴趣。该模型的定价策略被视为潜在的市场主导因素，其对原生工具使用和实时应用的集成被强调为一项关键创新。


- **[Gemini 2.0 Flash Experimental, 有人试过吗？](https://i.redd.it/xhzxaey1i86e1.png)** ([Score: 96, Comments: 44](https://reddit.com/r/LocalLLaMA/comments/1hbvegm/gemini_20_flash_experimental_anyone_tried_it/)): **Gemini 2.0 Flash Experimental** 提供了 **多模态理解与生成** 能力，支持处理代码以及生成文本和图像等用例。界面详细列出了 **定价**，在 **128K tokens** 以内及以上，输入和输出 token 均为 $0.00，**知识截止日期** 为 2024 年 8 月，**速率限制** 设置为每分钟 15 次请求。
  - **Gemini 2.0** 的 **物体定位 (object localization)** 能力给用户留下了深刻印象，它可以检测指定的物体类型并绘制边界框，而无需进行自定义 ML 训练，这是 **ChatGPT** 所不具备的功能。
  - 用户注意到了 Gemini 2.0 的 **速度**，一些人将其在数据科学任务中的表现与 **Claude** 进行了比较。虽然这两个模型在错误纠正方面都表现欠佳，但用户赞赏 Google 采用多个模型相互测试的方法，尽管遇到了需求限制，且与 Claude 更复杂的功能相比，其提供的功能较为基础。
  - 据报道存在一些兼容性问题，例如与 **cline** 和 **cursor composer** 的兼容性，不过建议采用编辑扩展文件等变通方法。此外，根据公告，**图像生成** 目前仅限于早期测试人员。


**主题 2. QRWKV6-32B 与 Finch-MoE-37B-A11B：线性模型的创新**

- **新型线性模型：QRWKV6-32B（基于 Qwen2.5-32B 的 RWKV6）和基于 RWKV 的 MoE：Finch-MoE-37B-A11B** ([Score: 81, Comments: 28](https://reddit.com/r/LocalLLaMA/comments/1hbv2yt/new_linear_models_qrwkv632b_rwkv6_based_on/)): Recursal 发布了两个实验性模型 **QRWKV6-32B** 和 **Finch-MoE-37B-A11B**，它们利用高效的 RWKV Linear attention 机制来降低时间复杂度。**QRWKV6** 将 Qwen2.5 架构与 RWKV6 相结合，允许在不从头开始重新训练的情况下进行转换；而 **Finch-MoE** 是一个 Mixture-of-experts 模型，总参数量为 37B，激活参数量为 11B，并承诺未来会进行扩展和改进。更多模型如 **Q-RWKV-6 72B Instruct** 和 **Q-RWKV-7 32B** 正在开发中。欲了解更多详情，请访问其 [Hugging Face model cards](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1) 和 [Finch-MoE](https://huggingface.co/recursal/Finch-MoE-37B-A11B-v0.1-HF)。
  - **RWKV 的潜力与局限性**：评论者讨论认为，虽然 **RWKV** 具有理论上的速度优势并能处理长 context lengths，但目前的推理引擎尚未针对充分实现这些优势进行优化。尽管由于训练资源有限导致对短 context lengths 有所担忧，但人们对将 Transformer 转换为 RWKV 的实用性仍感兴趣。
  - **实施挑战**：在 **koboldcpp** 等平台上实施 **QRWKV6** 等新架构存在挑战，因为这通常需要社区投入专门精力进行适配和实现。RWKV 社区被认为有潜力最终克服这些障碍。
  - **未来发展与预期**：评论者对 **RWKV 7** 等未来模型以及 **QwQ** 模型表示期待。人们对线性推理模型抱有希望，讨论中还涉及了需要 reasoning-style 数据来改进模型转换和推理时的思考过程。


- **QwQ-32B Preview 的 Speculative Decoding 可以使用 Qwen-2.5 Coder 7B 完成！** ([Score: 69, Comments: 28](https://reddit.com/r/LocalLLaMA/comments/1hbm7e3/speculative_decoding_for_qwq32b_preview_can_be/)): 该帖子讨论了使用 **Qwen-2.5 Coder 7B** 作为 **QwQ-32B** Speculative Decoding 的 draft model，并指出这两个模型的 vocab sizes 匹配。在 16 GB VRAM 的系统上，性能提升有限，但作者预计在更大 VRAM 的 GPU（如 24 GB）上会有显著改进。主观上，配合 Qwen Coder 的 QwQ 显得更加自信且逻辑清晰，尽管它使用了更多的字符和时间；作者邀请其他人进行实验并分享结果。提供了详细输出的 [PDF link](https://miscpublicbucket.s3.us-east-2.amazonaws.com/testing.pdf)。
  - **Speculative Decoding 技术**：关于在较大的 **QwQ-32B** 模型中使用较小的 draft model 的有效性存在争论。一些用户建议，当较小模型的尺寸显著小于较大模型的十分之一（例如 **0.5B 或 1.5B** 模型）时，速度提升才明显，而模型过大则效果不佳。
  - **性能观察**：用户报告称，在某些配置下，Speculative Decoding 可以带来 **1.5x 到 2x** 的加速，尽管逻辑或质量上的感知提升是主观的。建议使用固定 seed 来验证感知的改进是源于 Speculative Decoding，还是由于 GPU offload 不准确等其他因素。
  - **Speculative Decoding 方法**：提到了两种 Speculative Decoding 方法：一种是从两个模型中采样，仅在采样一致时才使用较小模型；另一种是使用较小模型的 logits 进行 rejection sampling。**llama.cpp** 中实现的具体方法对某些用户来说仍不明确。


## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**Theme 1. Google's Gemini 2.0: Strategic Release Amidst OpenAI Announcements**

- **Google 在 OpenAI 每日直播前几小时发布 Gemini 2.0：** ([Score: 244, Comments: 64](https://reddit.com/r/OpenAI/comments/1hbybhi/google_releasing_gemini_20_a_few_hours_before_the/))：**Google** 在 **OpenAI 每日直播活动**开始前几小时发布了 **Gemini 2.0**，引发了关于 **GPT-5/Epsilon** 可能获得确认的猜测。这一时机暗示了这两家科技巨头在 AI 领域的竞争态势。
  - **Gemini 2.0 性能**：用户注意到 **Gemini Flash 2.0** 在基准测试中表现异常出色，有人认为它的表现可能超过 **Sonnet 3.5** 等模型，且性价比可能高于 **4o-mini**。其上下文窗口（context window）比前代产品显著增大，使其在编程任务中特别有用。
  - **市场动态**：舆论强烈支持 **Google** 与 **OpenAI** 之间的竞争，因为这能推动创新并防止垄断导致的停滞。用户赞赏这种竞争格局，它促使公司不断改进其产品。
  - **采用与普及**：尽管对其知名度存在一些怀疑，但 **Gemini** 已集成到许多 **Google 产品**中，随着其性能得到更多认可，这可能会促进其普及。它已在 **AI Studio** 等平台上开放测试，用户发现它易于访问且功能强大。


- **[Google 刚刚推出了 Gemini 2.0](https://youtu.be/Fs0t6SdODd8)** ([Score: 202, Comments: 45](https://reddit.com/r/OpenAI/comments/1hbxtbt/google_just_introduced_gemini_20/))：**Google** 在 **OpenAI 周**期间推出了 **Gemini 2.0**，这表明其采取了战略举措来展示 AI 技术和可访问性方面的进展。
  - 讨论中强调了对 **Google Gemini 2.0** 的怀疑，一些用户因缺乏新的核心智能特性以及对 Google 历史上在演示中过度承诺的担忧而表示失望。另一些人则反驳称，Gemini 已经可以在 **aistudio** 上进行测试，预计 **Agents** 将于 1 月推出，部分用户报告了对该技术的积极体验。
  - 讨论还强调了 **Google** 凭借其 **TPUs** 的使用在价格上压低 **OpenAI** 的潜力，暗示 Google 可能会主导 AI 市场。对 **OpenAI 定价策略**和产品发布（如每月 200 美元的模型和 **Sora** 的发布）的批评，暗示了其财务困境以及相对于 Google 集成化方法的竞争劣势。
  - 一些用户猜测公告的战略时机，**Google 的发布时机**可能旨在抢在 OpenAI 预期的公告之前。竞争格局被视为一种动态的“针锋相对”环境，一些用户表示乐于看到这些主要的 AI 公司进行竞争。


**主题 2. Google GenCast：15 天 AI 天气预报引领未来预测**

- **Google 称其 AI 天气模型精通 15 天预报** ([Score: 286, Comments: 38](https://reddit.com/r/OpenAI/comments/1hby7zl/google_says_ai_weather_model_masters_15day/))：据报道，**DeepMind** 的 **GenCast** AI 模型在 15 天天气预报中实现了超过 **97% 的精度**，在 **35 个以上国家**的表现优于各种传统模型。欲了解更多详情，请参阅 [phys.org](https://phys.org/news/2024-12-google-ai-weather-masters-day.html) 上的文章。
  - 用户对 AI 驱动的天气预报的准确性表示怀疑，一些人指出**目前超过 5 天的预报**通常不可靠。一位评论者认为，对于两周以上的预报，**历史平均值**可能比预测更准确。
  - **GenCast 模型的代码库**可通过[出版物中的链接](https://www.nature.com/articles/s41586-024-08252-9)访问，这对于有兴趣进一步检查或利用该模型的人来说非常有用。
  - 讨论涉及了 AI 模型潜在的数据来源，一些用户猜测像 GenCast 这样的模型可能依赖 **NOAA 数据**。然而，其他人强调 AI 不会取代卫星和气象站等传统数据收集方法。


**主题 3. ChatGPT 停机：提升稳定性和用户依赖性的困扰**

- **我来问问 ChatGPT 是不是挂了... 看来其他 5,000,000 人也是这么想的...** ([Score: 299, Comments: 186](https://reddit.com/r/ChatGPT/comments/1hc7ndo/i_came_to_ask_it_chat_gpt_was_down_seems_like/)): **ChatGPT** 遭遇宕机，在期末考试期间给学生带来了困扰。这一问题引起了广泛关注，许多用户都在寻求确认服务是否中断。
  - 期末周期间 **ChatGPT** 的宕机引发了学生的极大不满，凸显了它作为学习和完成作业工具的重要性。用户表达了在编程和头脑风暴等任务中对它的依赖，并强调它不仅仅是作弊的手段。
  - 服务中断引发了幽默又无奈的反应，一些用户开玩笑地责怪自己导致了崩溃，而另一些人则回忆起使用 **Chegg** 等替代方案的经历。这一事件凸显了在关键学术时期对 AI 工具的高需求和依赖。
  - 多条评论指出此类宕机非常罕见，认为这是由于使用量增加和近期更新导致的倒霉时机。宕机引发了幽默的调侃，同时也引发了对学术表现受影响的严重担忧。


- **[⚠️ ChatGPT, API & SORA 目前已宕机！重大故障 | 2024年12月11日](https://i.redd.it/25fc77iu4b6e1.png)** ([Score: 195, Comments: 80](https://reddit.com/r/ChatGPT/comments/1hc7u3t/chatgpt_api_sora_currently_down_major_outage/)): **重大故障**：在 **2024年12月11日**，严重的宕机影响了 **ChatGPT、API 和 Sora** 服务，OpenAI 的状态页面显示正在调查中。90 天运行时间图表显示 API 和 ChatGPT 处于“Major Outage”状态，而 Labs 和 Playground 仍保持“Operational”。
  - 许多用户推测 **iOS 18.2 更新**和最近推出的 **Sora** 导致了此次宕机，集成到 Siri 和 Writing Tools 中的 ChatGPT 支持等新功能可能给服务器带来了压力。**dopedub** 指出，这些更新的时机可能选得不好，导致了系统过载。
  - 舆论普遍反映了对 **ChatGPT 的依赖**，如 **legend503** 的评论强调了在 AI 工具不可用时，这种依赖是多么脆弱。用户对宕机表达了沮丧和幽默，一些人建议在宕机期间使用替代平台或方法来访问 ChatGPT。
  - 用户分享了各种变通方法，例如尝试通过 **iPhone** 或使用 **mobile app** 访问 ChatGPT，并报告取得了一些成功。**InspectorOk6664** 等人指出，他们设法访问了服务，尽管功能有限，这表明宕机的影响在不同平台上有所不同。


**主题 4. Sora AI 批评：产出逊于竞争对手，引发用户不满**

- **Sora 太糟糕了** ([Score: 350, Comments: 193](https://reddit.com/r/OpenAI/comments/1hbos9w/sora_is_awful/)): **Sora** 的表现受到严厉批评，因为它无法准确生成视频，即使是让猫跳舞这样的简单任务，产出的质量也很差。用户对 Sora 的成本表示不满，认为与其他更有效且通常更便宜或免费的 text-to-video 生成器相比，除非有显著改进，否则该服务的价格并不合理。
  - **Sora 的性能与局限性**：许多用户一致认为 **Sora** 的表现不如 **Runway Gen-3** 和 **Luma** 等替代方案，并抱怨它无法有效处理 image-to-video 生成等任务。用户注意到，公开版的 Sora 是一个缩减版的 “Turbo” 模型，缺乏早期展示的演示版本所具备的计算能力。
  - **技术与市场挑战**：评论认为 **Sora** 的局限性源于计算限制以及平衡需求与可用资源的需要，导致发布的是一个“缩水”版本。这让那些期待早期演示中展示的功能的用户感到失望，因为那些演示可能使用了比公开版更多的资源。
  - **社区情绪与对比**：社区对 **OpenAI** 处理 Sora 的方式表示怀疑，一些人推测该产品是为了与 **Google 的 Gemini Pro** 竞争而战略性发布的。用户还批评 **OpenAI 的 Dalle** 不如 **MidJourney**，表明了对 OpenAI 产品相对于竞争对手的普遍不满。

- **[使用 Sora AI 重现我最喜欢的 AI 视频](https://v.redd.it/kapq475ob76e1)** ([得分: 2574, 评论: 140](https://reddit.com/r/ChatGPT/comments/1hbr2ii/used_sora_al_to_recreate_my_favorite_al_video/)): **Sora AI** 因**无法完成预期任务**而受到批评，导致了负面的用户反馈。该帖子提到尝试使用 Sora AI 重现一段最喜欢的 AI 视频，突显了用户对其表现的不满。
  - 用户压倒性地更喜欢**原始视频**而非 Sora AI 的重制版，理由是原版更具幽默感和吸引力。许多评论对 AI 生成内容的真实感和质量表示不满，一些用户还注意到了不当元素，如色情帧。
  - 用户对 Sora AI 使用的 **Prompt 和训练数据** 存在推测和好奇，一些用户质疑它是否在不当内容上进行了训练。讨论暗示了用户希望了解该 AI 的开发过程及其局限性。
  - 对话中包含了对 AI 输出的幽默和批评性评论，例如将其与**游戏角色**进行比较，以及提及 9/11 等**文化事件**。尽管存在批评，但人们也认可了该 AI 在展示技术演进和潜在未来应用方面的作用。


---

# AI Discord 摘要

> 由 O1-mini 生成的摘要之摘要之摘要

**主题 1. 新 AI 模型与重大更新**

- **Gemini 2.0 Flash 以卓越性能发布**：**Google DeepMind** 推出了 **[Gemini 2.0 Flash](https://x.com/lmarena_ai/status/1866873983569891378)**，在 Chatbot Arena 总榜中首次亮相即排名第 3，表现优于 **Flash-002** 等模型。此次发布增强了**多模态能力**和**编程性能**，为 AI 模型开发树立了新标杆。
- **Nous Research 发布 Hermes 3 以增强推理能力**：**Nous Research** 在 **Hugging Face** 上发布了 **[Hermes 3 3B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B)**，提供针对尺寸和性能优化的量化 GGUF 版本。Hermes 3 在**用户对齐**、**Agent 性能**和**推理**方面引入了高级功能，标志着其较前代产品的重大升级。
- **Windsurf Wave 1 增强开发者工具**：**Windsurf Wave 1** 正式发布，集成了 **Cascade Memories** 和**自动终端命令**等主要自主工具。此次更新还提升了**图像输入能力**，并支持 **WSL** 和 **devcontainers** 等开发环境。查看完整[变更日志](https://www.codeium.com/changelog)了解详细增强功能。

**主题 2. AI 工具性能与对比分析**

- **Windsurf 在 AI 工具对比中超越 Cursor**：社区讨论强调 **Windsurf** 是比 **Cursor** 更优越的 AI 工具，强调其更可靠的通信和透明的变更日志。用户赞赏 **Windsurf** 的更新方式和响应速度，使其在 AI 工具领域处于领先地位。
- **Muon 优化器成为 AdamW 的强力替代方案**：**Muon 优化器**因其稳健的基准性能和坚实的数学基础而受到关注，被定位为 **AdamW** 等传统优化器的可行替代方案。尽管它尚未超越 AdamW，但其在 **Llama 3** 等模型中的韧性突显了其在未来开发中的潜力。
- **Gemini 2.0 Flash 在编程任务中超越竞争对手**：**[Gemini 2.0 Flash](https://x.com/lmarena_ai/status/1866873983569891378)** 因其在**空间推理**和**迭代图像编辑**方面的卓越表现而受到称赞，超越了 **Claude 3.5** 和 **o1 Pro** 等模型。用户注意到了其极具竞争力的基准测试结果，引发了关于其相对于现有产品进步的进一步讨论。

**主题 3. 功能集成与平台增强**

- **ChatGPT 与 Apple 生态系统无缝集成**：在 **[12 Days of OpenAI](https://www.youtube.com/live/mBhkD0iFf4w?si=uMokZAeHp68wBwp2)** 活动期间，**ChatGPT** 成功集成到 **iOS** 和 **macOS**。由 **Sam Altman** 及其团队成员演示，此次集成包括增强的**节日主题功能**，旨在节日期间吸引用户。
- **NotebookLM 通过集成 Gemini 2.0 提升功能**：**NotebookLM** 确认集成了 **[Gemini 2.0](https://x.com/lmarena_ai/status/1866873983569891378)**，增强了其在 Discord 频道内进行实时 AI 交互的能力。尽管品牌选择受到了一些幽默的批评，但此次升级预计将增强 **NotebookLM** 的性能。
- **Supabase 集成增强 Bolt.new 工作流**：**Bolt.new** 在直播中预览了其 **Supabase 集成**，承诺为开发者提供改进的工作流功能。此次集成旨在简化现有流程，并通过增强平台的实用性来吸引更多用户。

**主题 4. 定价、使用透明度和订阅模式**

- **Windsurf 推出透明定价和使用情况更新**：**Windsurf** 推出了更新的定价系统，其特点是新增了 **[快速设置面板](https://codeium.com/redirect/windsurf/learn-pricing)**，可显示当前方案的使用情况和试用期到期时间。该更新还在 **Cascade** 中包含了一个“Legacy Chat”模式，在耗尽 **Flow Credits** 后激活，提供有限的功能而无需额外费用。
- **自助升级方案简化订阅管理**：**Windsurf** 推出了**自助升级方案按钮**，允许用户通过[此链接](https://www.codeium.com/plan)轻松访问更新的方案。此功能简化了根据项目需求扩展订阅的过程，提升了用户体验和灵活性。
- **30 美元 Pro 方案扩展 Open Interpreter 的应用能力**：**Killianlucas** 宣布 **Open Interpreter** 的 **30 美元每月桌面应用方案**增加了使用限制，并为免费用户提供无需 API key 的应用访问。他建议除非用户发现扩展功能非常有益，否则可以坚持使用免费方案，因为该应用在 Beta 阶段仍在快速演进。

**主题 5. 训练、微调和前沿研究**

- **Eleuther 分析训练 Jacobian 矩阵以揭示参数依赖性**：**Eleuther** 的研究人员在 [arXiv](https://arxiv.org/abs/2412.07003) 上发表了一篇论文，探讨了**训练 Jacobian**，通过分析导数矩阵揭示了最终参数如何依赖于初始参数。该研究区分了 **bulk** 和 **chaotic** 子空间，为**神经训练动力学**提供了见解。
- **Qwen 2.5 微调中的挑战凸显了集成难度**：在 **Unsloth AI** Discord 中，用户报告了从微调后的 **Qwen 2.5** 模型获取数值输出的困难，特别是在简单的乘法查询中。讨论强调集成特定领域知识具有挑战性，建议 **pre-training** 或采用 **RAG 解决方案**可能会提供更有效的结果。
- **LLM 训练创新：COCONUT 和 RWKV 架构**：**[COCONUT (Chain of Continuous Thought)](https://arxiv.org/abs/2412.06769)** 的引入使 LLM 能够在连续潜空间（latent space）内进行推理，通过直接嵌入（embedding）方法优化处理。此外，发布了新的 **RWKV 架构**，如 **[Flock of Finches](https://huggingface.co/rwkv/Finch-MoE-37B-A11B-HF)** 和 **[QRWKV-6 32B](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)**，强调在不牺牲性能的情况下优化训练成本。

---

# 第 1 部分：高层 Discord 摘要

## [Codeium / Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Wave 1 发布增强了 Windsurf 的能力**：Windsurf Wave 1 正式发布，引入了主要的自主工具，如 **Cascade Memories** 和 **自动化终端命令**。此次更新还包括增强的 **图像输入功能**，以及对 **WSL** 和 **devcontainers** 等开发环境的支持。查看完整的 [changelog](https://www.codeium.com/changelog) 获取详细信息。
   - 用户对新功能表示兴奋，强调 **Cascade Memories** 的集成是一项重大改进。此次发布旨在通过自动化终端命令和支持多样化的开发环境来简化开发者的工作流。
- **Windsurf 价格和使用透明度更新**：Windsurf 正在推出更新的定价系统，其特点是新增了 **快速设置面板**，可显示当前方案的使用情况和试用有效期。有关更改的详细信息可在 [定价页面](https://codeium.com/redirect/windsurf/learn-pricing) 查看。
   - 更新在 **Cascade** 中引入了 “Legacy Chat” 模式，当用户耗尽 **Flow Credits** 时该模式会激活，允许在不使用额外额度的情况下使用有限的功能。此更改旨在提供更清晰的使用指标，并提高用户管理订阅的透明度。
- **Windsurf 中增强的 Python 支持**：Windsurf 改进了对 **Python 的语言支持**，为开发者提供更好的集成和高级功能。这一增强是该平台致力于为开发者提供更有效工具的一部分。
   - 此外，还引入了自助升级方案按钮，使用户能够通过 [此链接](https://www.codeium.com/plan) 轻松访问更新的方案。此功能简化了根据项目需求扩展订阅的过程。
- **Cascade 图像上传和功能增强**：**Cascade 图像上传** 不再受 1MB 限制，允许用户无缝分享更大的文件。这项改进是持续提升平台可用性和功能努力的一部分。
   - 提高的上传限制旨在支持更广泛的工作流，使用户能够将更高分辨率的图像合并到项目中，而不会遇到大小限制。这一变化有助于提供更灵活、更高效的用户体验。
- **报告 Cascade 模型性能问题**：用户报告 **Cascade Base** 正在经历性能问题，如卡死和无响应，这影响了其在编码任务中的可靠性。一些用户在使用过程中遇到了 HTTP 504 错误。
   - 有推测认为这些不稳定性问题可能与 **OpenAI** 持续存在的问题有关，导致几位用户考虑降级或切换到替代工具。社区正在积极讨论潜在的解决方案，以缓解这些性能挑战。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Apple Integration**: 在 [YouTube 演示](https://www.youtube.com/live/mBhkD0iFf4w?si=uMokZAeHp68wBwp2)中，Sam Altman、Miqdad Jaffer 和 Dave Cummings 在 **12 Days of OpenAI** 活动期间展示了 **ChatGPT** 集成到 **iOS** 和 **macOS** 的功能。
   - 节日演示中，团队成员身穿**节日毛衣**，旨在增强观众互动并体现**节日季**精神。
- **Gemini 2.0 Flash Outperforms**: **Gemini 2.0 Flash** 因其卓越性能受到用户称赞，尤其是在**空间推理**和**迭代图像编辑**方面，表现优于 **o1 Pro** 等模型。
   - 社区对比突出了 **Gemini 2.0 Flash** 的竞争性基准测试，引发了关于其相对于 **OpenAI** 产品进步的讨论。
- **OpenAI Services Experience Downtime**: 根据[状态更新](https://status.openai.com/incidents/ctrsv3lwd797)报告，**OpenAI** 经历了影响 **ChatGPT** 及相关 API 工具的服务中断。
   - 用户注意到中断与 Apple 集成公告同时发生，截至 2024 年 12 月 11 日，**API 流量**已部分恢复，但 **Sora** 仍处于宕机状态。
- **Challenges in Fine-Tuning OpenAI Models**: 开发者报告称，尽管完成了训练，经过微调的 **OpenAI** 模型仍会生成**泛泛的回答**，并就其 **JSONL** 配置寻求帮助。
   - 正在征求社区反馈以识别训练数据中的潜在问题，旨在提高模型的针对性和性能。
- **Chaining Custom GPT Tools**: 用户在单个提示词中链接多个 **Custom GPT** 工具时遇到困难，通常只有第一个工具的 API 调用被执行。
   - 建议包括添加**元功能指令**和使用规范的工具名称，以改进工具交互管理。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Slows Amid OpenAI Model Hiccups**: 用户报告 **Cursor** 持续出现性能问题，特别是请求响应缓慢和代码库索引问题。这些问题似乎与 **OpenAI 模型** 的上游问题有关，导致平台性能下降。[Cursor Status](https://status.cursor.com/) 显示持续的停机影响了核心功能。
   - 社区成员对 **Cursor** 在停机期间无法有效编码表示沮丧。关于[长上下文模式的功能请求](https://forum.cursor.com/t/feature-request-long-context-mode/32187/2)引起了广泛关注，但正式支持仍待定。
- **Agent Mode Hangs Code Flow**: **Cursor** 的 **Agent 模式** 因在处理代码或访问代码库时频繁卡死而面临广泛投诉。用户发现重新索引或重启 **Cursor** 可以暂时缓解这些问题。
   - 尽管有临时修复方案，但 **Agent 模式** 中断的反复发生继续阻碍开发者的生产力，引发了社区内关于潜在长期解决方案的讨论。
- **Windsurf Surges Past in AI Tool Comparisons**: 讨论强调 **Windsurf** 是比 **Cursor** 更优秀的 AI 工具，强调其更可靠的沟通和透明的更新日志。用户赞赏 **Windsurf** 处理更新和反馈集成的方式。
   - 参与者指出，虽然 **Cursor** 提供了独特的功能，但其更新日志的透明度和响应速度落后于 **Windsurf** 等竞争对手，这表明在满足用户期望方面仍有改进空间。
- **Gemini 2.0 Shines Twice as Fast**: [Sundar Pichai](https://x.com/sundarpichai/status/1866868228141597034) 宣布推出 **Gemini 2.0 Flash**，其在关键基准测试上的表现优于 **Gemini 1.5 Pro**，且速度提升了两倍。这一进步标志着 **Gemini** 模型开发的一个重要里程碑。
   - AI 社区正密切关注 **Gemini 2.0** 的进展，对其性能提升以及对 **Claude** 等现有模型的潜在影响寄予厚望。
- **Windsurf's Transparent Changelogs Impress**: **Windsurf** 用户称赞该工具在 [Windsurf Editor Changelogs](https://codeium.com/changelog) 中提供的清晰沟通和详细记录。这种透明度展示了对用户反馈和持续改进的坚定承诺。
   - **Windsurf** 互动模式的好评与 **Cursor** 形成对比，引发了关于采用类似透明做法以增强用户信任和满意度的讨论。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **训练 Jacobian 分析揭示参数依赖性**：最近发表在 [arXiv](https://arxiv.org/abs/2412.07003) 上的一篇论文探讨了**训练 Jacobian**，通过分析导数矩阵揭示了最终参数如何取决于初始参数。研究强调，训练将参数空间操纵为**主体 (bulk)** 和**混沌 (chaotic)** 子空间。
   - 由于计算限制，该研究使用了 5K 参数的 MLP 进行，但在 62K 参数的图像分类器中观察到了类似的频谱模式，这表明对**神经训练动力学 (neural training dynamics)** 具有更广泛的意义。
- **Muon 优化器脱颖而出成为强力竞争者**：**Muon** 因其稳健的基准性能和坚实的数学基础而受到关注，使其成为 **AdamW** 等现有优化器的可行替代方案。[Keller Jordan](https://kellerjordan.github.io/posts/muon/) 详细介绍了其在优化神经网络隐藏层方面的潜力。
   - 虽然 **Muon** 尚未果断超越 **AdamW**，但它在应对 **Llama 3** 等模型中观察到的问题时表现出的韧性，凸显了其在未来优化器开发中的前景。
- **解决大语言模型中的固有偏见**：即将发表的一篇论文讨论了大型语言模型中的**有害偏见**是其当前架构的内在结果，主张对 AI 设计原则进行根本性的重新评估。研究强调，偏见源于数据集近似和模型架构。
   - 这一观点鼓励 AI 社区专注于理解偏见的根本原因，而不仅仅是调整具体的实现，从而促进更有效的偏见缓解策略。
- **在 lm_eval_harness 中集成 HumanEval 以增强困惑度指标**：**lm_eval_harness** 正被用于评估模型在 **jsonl 文件** 格式数据集上的**困惑度 (perplexity)**，成员们分享了自定义任务配置。一个特定的 Pull Request ([#2559](https://github.com/EleutherAI/lm-evaluation-harness/pull/2559)) 旨在促进批量推理，解决处理效率低下的问题。
   - 此次集成旨在提供对比研究必不可少的 **per-token perplexity** 结果，社区成员正积极贡献解决方案以简化评估工作流。
- **新 RWKV 架构发布增强模型效率**：新的 **RWKV** 架构，即 [Flock of Finches](https://huggingface.co/rwkv/Finch-MoE-37B-A11B-HF) 和 [QRWKV-6 32B](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1) 已经发布，强调在不牺牲性能的情况下优化训练成本。
   - 这些模型展示了与大型同类模型相当的能力，同时保持了较低的计算需求，使其对可扩展的 AI 应用具有吸引力。



---



## [Bolt.new / Stackblitz](https://discord.com/channels/364486390102097930) Discord

- **Cole 在 Twitter 上推广 OSS Bolt**：在一次直播会议中，Cole 讨论了 **OSS Bolt**，分享了其历程和进展，社区在 [Twitter](https://x.com/stackblitz/status/1866867336730628513) 上观看了相关内容。
   - **OSS Bolt** 正在获得关注，最新的 **Bolt Office Hours: Week 8** 现已在 [YouTube](https://www.youtube.com/watch?v=xlbKobsqfdc) 上线，其中包含增强观众参与度的资源链接。
- **Supabase 预览增强 Bolt**：一次直播提供了 **Supabase 集成** 的**预览 (sneak peek)**，旨在增强现有工作流并吸引开发者的兴趣。
   - 正如直播中所揭示的，这次 **Supabase 集成** 有望改进工作流功能，令开发者社区感到兴奋。
- **在 Stripe 集成方面 Supabase 优于 Firebase**：几位用户讨论了从 **Firebase** 迁移到 **Supabase** 以获得更好的 **Stripe 集成**，并评估了每个数据库的优缺点。
   - 选择 **Supabase** 的决定是出于避免供应商锁定 (vendor lock-in) 的愿望，使其成为一些应用开发者的首选。
- **Shopify API 集成助力 Web 应用**：一位正在开发与其 **Shopify** 商店同步的 Web 应用的成员收到了集成 **Shopify API** 的建议，并引用了 [Shopify API 文档](https://shopify.dev/docs/api)。
   - 实施 **Shopify API 集成** 可确保该应用独有的安全数据访问，并利用全面的文档进行无缝开发。
- **Bolt AI Token 消耗引发担忧**：用户报告了对 **Bolt AI** 反复出错并在未解决问题的情况下消耗大量 Token 的沮丧情绪，其中一位用户在没有产生任何变化的情况下使用了 **200k tokens**。
   - 问题归因于底层的 AI 模型 **Claude**，这促使人们建议改进 Prompt 的构建方式，以尽量减少调试过程中的 Token 浪费。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 2.5 微调面临数字输出问题**：一位用户报告了在从微调后的 **Qwen 2.5** 模型获取数字输出时遇到的挑战，特别是在处理简单的乘法查询时。
   - 讨论强调，整合特定领域知识非常困难，并建议 **pre-training**（预训练）或采用 **RAG 解决方案** 可能会提供更有效的结果。
- **Gemini Voice 超越竞争对手**：社区成员对 **Gemini Voice** 的功能表示兴奋，称其表现“疯狂”且优于现有的某些替代方案。
   - 对比指出，**OpenAI 的实时语音 API** 成本更高且功能较弱，使 Gemini Voice 成为更优的选择。
- **管理训练期间的内存峰值**：用户对训练阶段 **RAM** 和 **GPU RAM** 使用量出现意外峰值表示担忧，这导致了 **Colab** 上的运行时崩溃。
   - 用户建议通过调整 **batch sizes**、平衡 **dataset lengths** 以及确保 **high-quality data** 等策略来缓解内存问题。
- **使用 llama.cpp 进行有效的模型转换**：用户讨论了利用 **llama.cpp** 进行模型转换，分享了经验和故障排除方法以简化流程。
   - 一位成员通过执行全新安装并利用 **PyTorch 2.2** 作为特定的依赖版本，成功解决了转换问题。
- **WizardLM Arena 数据集现已发布**：一位成员将 **WizardLM Arena** 论文中使用的所有数据集上传到了 [该仓库](https://huggingface.co/datasets/forcemultiplier/arena-paper-datasets-jsonl/tree/main)，提交记录 [b31fa9d](https://huggingface.co/datasets/forcemultiplier/arena-paper-datasets-jsonl/commit/b31fa9dba65f1931cfbd1daf37f35b7c94afecf2) 已于一天前通过验证。
   - 这些数据集为复制 **WizardLM Arena** 实验和进一步研究提供了全面的资源。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 3 发布，推理能力升级**：**Hermes 3 3B** 已在 [Hugging Face](https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B) 上发布，并提供量化的 GGUF 版本以优化大小和性能，在 **user alignment**（用户对齐）、**agentic performance**（Agent 性能）和 **reasoning**（推理）方面引入了先进功能。
   - 此次发布标志着从 **Hermes 2** 开始的重大升级，增强了 AI 工程师使用该模型的整体效率和有效性。
- **Google DeepMind 首次推出 Gemini 2.0**：**Google DeepMind** 揭晓了 **Gemini 2.0**，包括具有改进的多模态输出和性能的实验性 **Gemini 2.0 Flash** 版本，正如其 [官方推文](https://x.com/GoogleDeepMind/status/1866869343570608557) 所宣布的那样。
   - **Gemini 2.0 Flash** 的引入旨在通过增强的 **tool use capabilities**（工具使用能力）促进新的 **agentic experiences**（Agent 体验）。
- **DNF 推进 4D 生成建模**：**DNF** 模型引入了一种用于生成建模的新 4D 表示，使用字典学习（dictionary learning）方法捕捉可变形形状的高保真细节，详见 [项目页面](https://xzhang-t.github.io/project/DNF/)。
   - [YouTube 概览](https://youtu.be/l1UhJmTbKIo) 展示了 DNF 的应用，体现了时间一致性和卓越的形状质量。
- **COCONUT 增强 LLM 推理过程**：**COCONUT (Chain of Continuous Thought)** 使 LLM 能够在连续潜空间（continuous latent space）内进行推理，通过更直接的 embedding 方式优化处理，如 [研究论文](https://arxiv.org/abs/2412.06769) 中所述。
   - 该方法建立在 Schmidhuber 关于循环高速公路网络（Recurrent Highway Networks）的工作基础上，通过梯度下降提供端到端优化，以提高推理效率。
- **Forge Bot 简化 Discord 访问**：成员现在可以通过 Discord 机器人访问 **Forge**，无需 API 审批，只需从 **customize** 页面选择 **Forge Beta** 角色并导航至相应频道即可。
   - 这一增强功能方便了 AI 工程师在服务器内进行即时测试和协作。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **VRAM 使用与管理**：用户讨论了以 JSON 格式缓存 prompt 如何影响内存使用，指出存储 **20 个 prompt** 仅消耗几兆字节。他们还探索了在 **FP8** 等较低精度下运行模型以降低内存成本。
   - 优化 **FP8 精度** 对于在大规模 AI 部署中高效管理 VRAM 至关重要。
- **图像增强的 AI 模型推荐**：一位用户请求生成 **spaceships**（飞船）的模型推荐，提到他们正在使用 **Dream Shaper** 和 **Juggernaut**。另一位成员建议，如果现有模型不能满足需求，可以训练一个 **LoRA**。
   - 训练 **LoRA** 可以为特定的图像生成需求提供定制化增强。
- **GPU 黄牛问题**：成员们分享了关于黄牛可能如何部署网络爬虫在发布当日抢购 **GPUs** 的见解。一位用户表达了对在黄牛之前获取 GPU 的担忧，提到由于身处美国境外，他们避免了排队。
   - 防止 GPU 黄牛需要物理排队以外的策略，特别是对于国际用户。
- **用于图像分类和打标签的 AI**：一位用户询问了从图像中提取标签或 prompt 的工具，引发了关于 **图像分类** 技术的讨论。**Clip Interrogator** 被推荐为一种可以在没有先验元数据的情况下描述图像的工具。
   - **Clip Interrogator** 可以自动执行打标签过程，提高管理大型图像数据集的效率。
- **语音训练的 AI 程序**：用户讨论了通常用于针对特定声音训练 AI 的程序，其中 **Applio** 被提及为一个潜在工具。对话强调了对 **语音操作** 和 AI 训练应用的兴趣。
   - 选择像 **Applio** 这样的合适程序可以改进特定语音 AI 训练任务的工作流程。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **将 NotebookLM 集成到 Discord**：一位成员讨论了在 Discord 中添加 **聊天功能**，使用户能够直接查询 **NotebookLM** 以增强交互性。
   - 这种集成旨在为寻求在 Discord 公会内进行实时 AI 互动的玩家提供无缝连接。
- **用于 TTRPG 规则转换的 NotebookLM**：一位成员展示了使用 **100 页的自定义 TTRPG 规则书** 配合 **NotebookLM**，将叙述性描述转换为特定的游戏机制。
   - 他们强调了该工具在将复杂规则转化为可操作机制方面的有效性，从而促进更流畅的游戏体验。
- **利用 NotebookLM 增强播客**：用户探索了利用 **NotebookLM** 进行播客创作，包括自定义主持人以及利用来自 [Talking Heads Green Screen Podcast](https://youtu.be/MjbH5neAAAo?feature=shared) 的视觉效果。
   - 成员们建议采用 **chromakey（色键）技术** 来个性化播客背景，提升视觉吸引力。
- **Gemini 2.0 集成到 NotebookLM**：讨论确认了 **Gemini 2.0** 与 **NotebookLM** 的集成，成员们幽默地注意到品牌推广中缺少 Gemini 符号。
   - 尽管对品牌选择有轻松的批评，但预计这一转变将增强 **NotebookLM** 的能力。
- **NotebookLM 功能和 AI 工具体验**：用户研究了 **NotebookLM** 的限制，包括每个笔记本最多 **1,000 条笔记**、每条 **100,000 个字符** 以及 **50 个来源**，并对 PDF 大小限制提出了疑问。
   - 此外，成员们分享了使用 **Snapchat AI** 和 **Instagram 聊天 AI** 等工具的积极体验，强调了 **NotebookLM** 在挑战时期对他们工作流程的实用性。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemini 2.0 Flash 发布**：Google 推出了 **Gemini 2.0 Flash**，强调了其与前代版本相比卓越的 **multimodal capabilities**（多模态能力）和增强的 **coding performance**（编程性能）。[Google DeepMind blog](https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/) 详细报道了此次发布活动，并强调了其顶级的 benchmark 分数。
   - 用户称赞了 Gemini 2.0 Flash 强大的 **coding functionalities**，并将其与 **Claude** 等平台进行了比较。然而，正如多条 [tweets](https://x.com/officiallogank/status/1866916665902108785) 中讨论的那样，一些人对其相对于其他模型的 **stability**（稳定性）和 **effectiveness**（有效性）提出了担忧。
- **NeurIPS 上的 Scaling Laws 辩论**：**Dylan Patel** 将在 NeurIPS 参加一场关于 **scaling laws** 的 **Oxford Style Debate**（牛津式辩论），挑战“仅靠 scaling 就能推动 AI 进步”的观点。该活动承诺将回应来自行业影响力人士的批评，注册详情可见其 [tweet](https://x.com/dylan522p/status/1866630813074461060)。
   - 社区表现出强烈的参与意愿，认为这种辩论形式既有趣又有见地。讨论强调了 **search**（搜索）、**adaptation**（适配）和 **program synthesis**（程序合成）比单纯的模型 scaling 更重要，反映了更广泛的 **AI development trends**（AI 发展趋势）。
- **加入 Meta 的 Llama 团队**：一位成员宣布了他们即将在 **[AI at Meta's Llama team](https://x.com/_arohan_/status/1866621771451076812)** 担任的角色，旨在开发下一代 **Llama models**。此举强调了致力于培养一个与 **AGI goals** 和 **open-source initiatives**（开源倡议）相一致的更健康的 **AI ecosystem**。
   - 尽管过去在 **Llama 3.2-vision** 上遇到过挑战，该公告还是引发了关于将 **Gemini** 的见解整合到 Llama 项目中的讨论。成员们幽默地辩论了展示 **Kambhampathi's works** 可能带来的干扰，突显了社区对这一新尝试的支持。
- **OpenAI 以产品为中心的策略**：**OpenAI** 继续强调 **product-centric approach**（以产品为中心的方法），通过专注于 **user-friendly products**（用户友好型产品）保持其在 AI 领域的领导地位。正如社区讨论中所观察到的，这种策略使 OpenAI 在竞争对手滞后时保持领先。
   - 对 **usability**（易用性）的关注确保了 OpenAI 的发展保持相关性和影响力，强化了 **practical applications**（实际应用）在持续的 AI 进步中的重要性。
- **LLM 创造力基准测试**：关于评估 **LLM capabilities** 在创意任务中表现的 [讨论](https://gwern.net/creative-benchmark) 出现，揭示了 **benchmark scores** 与 **user satisfaction**（用户满意度）之间的差异。尽管 **Claude-3** 并不总是在创造力指标上领先，但它仍然很受欢迎。
   - 社区成员正在探索更好地评估 LLM **creative outputs**（创意输出）的方法论，旨在使 **benchmarking practices**（基准测试实践）与实际的 **user experiences**（用户体验）和期望相一致。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **在 LM Studio 中最大化 GPU 利用率**：讨论集中在如何使用多个 **GPUs**（包括 **两个 3070** 和一个 **3060ti**），目前 **LM Studio** 的 **GPU offload** 功能仅为一个简单的开关。
   - 由于现有的限制，用户正在探索通过 **environment variables**（环境变量）来增强多 GPU 性能的权宜之计。
- **利用生成式 AI 推进文档合并**：一位成员寻求使用生成式 AI 合并两个文档的方法，但建议倾向于传统技术，如 **MS Word merge option**。
   - 替代方案包括编写自定义脚本进行精确合并，而不是依赖模糊的 AI prompts。
- **通过 API 集成启用模型 Web 访问**：关于为模型提供 **web access** 的咨询得到的回答指出，这需要 **custom API solutions**，而不是标准的聊天界面。
   - 这表明人们对将模型与 **external tools and websites**（外部工具和网站）集成以增强操作能力的兴趣日益浓厚。
- **在硬件配置中集成 Alphacool 的 D5 泵**：**Alphacool** 推出了预装 **D5 pump** 的型号，给一些成员留下了深刻印象。
   - 然而，*一位成员对没有选择这种配置表示遗憾*，原因是他们包含 **4 GPUs** 和 **8 HDDs** 的庞大配置存在空间限制。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Gemini 2.0 Flash 发布**：Google [宣布](https://x.com/lmarena_ai/status/1866873983569891378)推出 **Gemini 2.0 Flash**，该模型在 Chatbot Arena 综合排名中位列第 3，表现优于之前的 Flash-002 等模型。
   - **Gemini 2.0 Flash** 在困难提示词（hard prompts）和编程任务中提供了增强的性能，并集成了实时多模态交互。
- **Hyperbolic 完成 1200 万美元 A 轮融资**：Hyperbolic 成功筹集了 **1200 万美元 A 轮**资金，旨在开发一个开放的 AI 平台，其开放 GPU 市场以 **$0.99/小时**的价格提供 H100 SXM GPU。
   - 此次融资强调了 AI 基础设施开发中的**透明度**和**社区协作**。
- **Stainless 获得 2500 万美元 A 轮融资**：Stainless API 宣布完成由 **a16z** 和 **Sequoia** 领投的 **2500 万美元 A 轮**融资，旨在增强其 AI SDK 产品。
   - 这笔投资将支持为 AI 开发者构建一个强大的生态系统。
- **Nous Simulators 发布**：Nous Research 推出了 [Nous Simulators](https://sims.nousresearch.com)，用于在社交语境中实验人类与 AI 的交互。
   - 该平台旨在提供对 **AI 行为**和交互的见解。
- **实时多模态 API 亮相**：Logan K 介绍了由 Gemini 2.0 Flash 驱动的新型 [Realtime Multimodal API](https://x.com/officiallogank/status/1866873298027446465?s=46)，支持实时音频、视频和文本流。
   - 该 API 促进了后台的动态工具调用，从而实现无缝的交互体验。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **C23 标准化纯函数（Pure Functions）**：通过在 **C23** 中包含此特性，C 和 C++ 中的**纯函数**标准化已经实现，详细指南参考 [n2956](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2956.htm) 和 [p0078](https://wg21.link/p0078)。
   - 此次更新通过清晰地标记**纯函数**来促进更好的优化，增强了代码的可预测性和性能。
- **Modular 论坛链接位置隐蔽**：用户对 **Modular** 网站的论坛链接难以找到表示沮丧，该链接被**埋**在 **Company** 栏目下。
   - 团队正致力于改进论坛，并计划在 **1 月份正式公开发布**，以提高可访问性。
- **Mojo 的 Multi-Paxos 实现问题**：一个用 C++ 实现的 **Multi-Paxos** 共识协议在初步测试中失败，因为它无法有效处理多个提案。
   - 缺乏超时、领导者切换和重试等基本功能，这类似于神经网络中反向传播（back-propagation）的必要性。
- **关于 Mojo Struct __del__ 设计的辩论**：社区讨论了 Mojo struct 的 `__del__` 方法应该是选择性加入（opt-in）还是选择性退出（opt-out），权衡了**一致性**与开发者**人机工程学（ergonomics）**。
   - 一些成员倾向于减少样板代码（boilerplate code），而另一些成员则主张在 Trait 和方法之间采用统一的方法。
- **命名结果（Named Results）带来的性能提升**：Mojo 中的**命名结果**允许直接写入地址，避免了昂贵的**移动操作（move operations）**，并在函数返回期间提供性能保证。
   - 虽然该特性主要提供保证，但在移动操作不可行的情况下，它能提高效率。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Maya 多模态模型发布**：介绍 [Maya](https://x.com/karthik_kanjula/status/1866668743918567469?s=46)，这是一个开源、多语言的视觉语言模型，支持 **8 种语言**，专注于文化多样性，基于 **LLaVA 框架**构建。
   - 在 **Apache 2.0 许可证**下开发，Maya 的[论文](https://arxiv.org/abs/2412.07112)强调了其指令微调（instruction-finetuned）能力，社区正热切期待**博客文章**和额外资源。
- **Rerank 3.5 英文模型计划**：一位成员询问了 **Rerank 3.5 英文模型**的后续开发计划，寻求对未来增强功能的见解。
   - 截至目前，该询问**尚未得到回应**，使得 Rerank 3.5 模型的计划**悬而未决**。
- **Aya Expanse 增强 Command 系列**：讨论强调 **Aya Expanse** 可能受益于 **Command 系列**的性能，暗示了改进的指令遵循（instruction following）能力。
   - 成员们暗示，由于 **Aya Expanse** 可能是基于 Command 系列构建的，它在处理和执行命令方面可能提供**增强的性能**。
- **持续的 API 403 错误**：用户报告在使用 API 请求生成器时遇到 **403 错误**，即使在禁用 **VPN** 并使用 **Trial API key** 后也是如此。
   - 分享的细节包括使用来自**中国的 IPv6 地址**和特定的 **curl** 命令，但该问题在社区内仍**未解决**。
- **寻求用于量化的高质量数据集**：成员们正在寻找适合**量化**的高质量数据集，特别是对 **aya_dataset** 中的 **'re-annotations'** 标签感兴趣。
   - 社区强调需要具有**大量样本**的数据集，以支持稳健的量化任务。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **黑客松提交截止至 12 月 17 日**：参与者必须在 [12 月 17 日](https://forms.gle/Paf6cSsCgfmGvHg47)之前提交其**黑客松作品**，并遵守提供的指南。
   - 此外，**书面文章作业**的截止日期为 **PST 时间 12 月 12 日晚上 11:59**，这与之前要求的 **12 份讲座总结**是分开的。
- **书面文章作业的详细指南**：学生需要在 [Twitter](https://twitter.com)、[Threads](https://threads.net) 或 [LinkedIn](https://linkedin.com) 等平台上发布一篇约 **500 字**的帖子，并链接到 MOOC 网站。
   - 文章提交按**通过或不通过**计分，且必须使用注册课程时相同的电子邮件提交，以确保获得相应学分。
- **ToolBench 平台在 ICLR'24 上展示**：[ToolBench 项目](https://github.com/OpenBMB/ToolBench)在 **ICLR'24** 上展出，作为一个用于工具学习的大语言模型训练、服务和评估的开放平台。
   - 该平台旨在为 AI 工程师提供增强的资源和框架，以推进大语言模型中的工具集成。
- **AI 模型函数调用的进展**：AI 模型正越来越多地利用详细的**函数描述（function descriptions）**和**签名（signatures）**，根据用户提示设置参数，以提高泛化能力。
   - 这一进展表明 AI 模型与预定义函数之间的交互趋向于更加复杂，增强了它们的运行复杂度。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **O1 Pro 启用高级控制功能**：一位成员提议在 OS 模式下使用 **Open Interpreter** 来控制 **O1 Pro**，从而实现网页搜索、canvas 和文件上传等功能。
   - 另一位用户考虑通过逆向工程 **O1 Pro** 来控制 **Open Interpreter**，并评论道 *“这开启的可能性……天哪。”*
- **Open Interpreter App Beta 访问权限仅限 Mac**：成员们确认 **Open Interpreter app** 目前处于 beta 阶段，需要邀请码，且仅限 **Mac** 用户使用。
   - 一位成员对身处等候名单却无法访问表示沮丧，而另一位成员分享了获取邀请的联系方式。
- **对新网站设计的评价褒贬不一**：用户对新网站设计给出了不同的反馈，有人表示最初看起来 *“有点突兀”*，但后来逐渐习惯了。
   - 其他人指出该设计仍在开发中，并计划在未来的更新中加入更酷的叠加效果。
- **30 美元的 Pro 计划扩展了 App 功能**：**Killianlucas** 解释说，每月 30 美元的桌面应用计划增加了使用限制，并为免费用户提供无需 API key 即可使用的应用。
   - 他建议除非用户觉得非常有益，否则可以坚持使用免费计划，因为该应用在 beta 阶段发展迅速。
- **Actions Beta App 专注于文件修改**：**Actions** 功能被强调为一个专注于文件修改的 beta 应用，这与仅在终端中可用的 OS 模式有所不同。
   - 成员们被鼓励探索这一新功能，尽管一些人遇到了限制，其中一人指出在测试时用尽了他们的 token 限制。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **QRWKV6-32B 实现了 1000 倍的计算效率**：基于 [Qwen2.5-32B 架构](https://x.com/rohanpaul_ai/status/1866971776737218564) 构建的 **QRWKV6-32B** 模型在匹配原始 32B 性能的同时，在推理中提供了 **1000 倍的计算效率**。
   - 训练在 **16 台 AMD MI300X GPU**（192GB VRAM）上耗时 **8 小时** 完成，展示了计算成本的显著降低。
- **Finch-MoE-37B-A11B 引入 linear attention**：作为新 **RWKV 变体** 的一部分，**Finch-MoE-37B-A11B** 模型采用了 linear attention 机制，以提高处理 **长上下文（long contexts）** 的效率。
   - 这一转变突显了 RWKV 架构在增强计算性能方面的持续发展。
- **DoraLinear 增强了参数初始化**：**DoraLinear** 通过利用 `to_empty` 方法进行幅度初始化（magnitude initialization）提升了用户体验，确保其不会破坏现有功能。
   - 在 `to_empty` 方法中实现 `swap_tensors` 有助于在初始化期间进行正确的设备处理，这对于不同设备上的 tensors 至关重要。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **O1 系列简化了 DSPy 工作流**：一位成员询问了 **O1 系列模型** 对 **DSPy 工作流** 的影响，认为 **MIPRO 的推荐参数** 可能需要针对优化进行调整。
   - 他们推测新模型可能会导致 **更少的优化周期** 或 **评估更少的候选程序**。
- **DSPy 中的通用优化错误**：一位用户报告在优化过程中遇到了 **奇怪的通用错误**，并提到了在特定频道发布的相关 bug。
   - 这一问题突显了社区在优化 DSPy 工作流时面临的持续 **挑战**。
- **DSPy Settings 中的 'backtrack_to' 属性错误**：一位成员分享了一个错误，即 **'backtrack_to'** 不是 DSPy 中 **Settings** 的属性，并寻求帮助。
   - 另一位用户指出该问题 **早些时候已解决**，可能是由于某些 **async** 使用导致的。
- **关于视频和音频 IO 重点的辩论**：一位用户发起了关于 DSPy 内部 **视频和音频 IO** 的讨论，引发了成员们的不同意见。
   - 一位成员主张集中精力处理 **文本和图像输入**，理由是现有功能的有效性。

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Grassroots Science 推动多语言 LLM 开发**：**Grassroots Science** 倡议将于 2025 年 2 月启动，旨在通过与 [SEACrowd](https://seacrowd.github.io/) 和 [Masakhane](https://www.masakhane.io/) 等伙伴合作进行众包数据收集，开发多语言 **LLM**。
   - 该项目专注于创建全面的多语言数据集，并评估人类偏好数据，以使模型符合多样化的用户需求。
- **LLaMA 3.2 吞吐量故障排除**：一位用户正在 A100 GPU 上优化 **LLaMA 3.2** 在 10,000 token 输入下的推理吞吐量，目标是达到每秒约 **200 tokens**，但目前性能较慢。
   - 讨论建议采用 Batching、Prompt 缓存以及利用量化模型等技术来提升吞吐量。
- **TGI 3.0 在 Token 处理上超越 vLLM**：据 Hugging Face 报告，**TGI 3.0** 处理 **token** 的数量是 **vLLM** 的三倍，运行速度快 **13 倍**，提升了对长 Prompt 的处理能力。
   - 凭借更低的内存占用，**TGI 3.0** 在 **LLaMA 3.1-8B** 上支持高达 **30k tokens**，而 **vLLM** 的限制为 **10k tokens**。
- **模型缩放怀疑论者青睐小模型**：一位成员质疑将模型扩展到超过 10 亿参数的必要性，并表示：*“10 亿参数对任何人来说都足够了。”*，同时倡导*超高效的小模型*。
   - 讨论批评了“规模即一切（scale-is-all-you-need）”的方法，强调了模型训练效率方面的优势。
- **COCONUT 为 LLM 引入连续推理**：**Chain of Continuous Thought (COCONUT)** 被作为一种新的 **LLM** 推理方法提出，详情见此 [推文](https://x.com/iScienceLuvr/status/1866353795502158163/photo/1)。
   - **COCONUT** 将最后的隐藏状态（hidden state）作为输入嵌入（embedding），通过梯度下降实现端到端优化，而非传统的隐藏状态与 token 映射。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **默认设置建议**：用户建议 **'default'** 应该成为默认设置。
- ****： 

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla AI 招聘社区参与负责人**：Mozilla AI 正在招聘一名 [社区参与负责人 (Head of Community Engagement)](https://job-boards.greenhouse.io/mozillaai/jobs/4600382007)，该职位为远程办公，直接向 CEO 汇报。
   - 该角色负责领导和扩展跨多个渠道的社区计划，以提升参与度。
- **推出用于 LLM 选择的 Lumigator**：Mozilla 正在开发 **Lumigator**，这是一款旨在帮助开发者为其项目选择最佳 **LLM** 的工具。
   - 该产品是 Mozilla 致力于向开发者社区提供可靠开源 AI 解决方案的一部分。
- **开发者中心优化 AI 资源**：Mozilla AI 正在启动 **Developer Hub**，提供用于构建开源 AI 的精选资源。
   - 该倡议旨在增强 AI 开发过程中的用户自主权和透明度。
- **Blueprints 开源 AI 集成**：**Blueprints** 倡议专注于通过入门代码仓库开源 AI 集成，以启动 AI 项目。
   - 这些资源旨在帮助开发者快速在应用中实现 AI 解决方案。
- **社区参与负责人职位咨询**：感兴趣的申请人可以在专门的 [话题](https://discord.com/channels/1089876418936180786/1316478017530495007/1316478017530495007) 中提出关于社区参与负责人职位的问题。
   - 这一职位彰显了 Mozilla AI 对社区驱动倡议的专注。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

# 第 2 部分：各频道详细摘要与链接

{% if medium == 'web' %}

### **Codeium / Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1316507848754200698)** (1 messages): 

> `Windsurf Wave 1 发布，Cascade Memories 与终端自动化，更新的定价与使用透明度，改进的 Python 语言支持，Cascade 图片上传` 


- **Windsurf Wave 1 发布引发关注**：Windsurf Wave 1 现已上线，具有主要的自主工具，包括 **Cascade memories** 和 **自动化终端命令**。查看完整的 [changelog](https://www.codeium.com/changelog) 获取所有激动人心的更新。
   - 此次发布强调了增强的 **图片输入能力**，以及对 **WSL** 和 **devcontainers** 等开发环境的支持。
- **更新的定价与使用透明度**：Windsurf 正在推出更新的使用和定价系统，配备新的 **快速设置面板**，显示当前方案的使用情况和试用有效期。有关更改的详细信息可以在 [pricing](https://codeium.com/redirect/windsurf/learn-pricing) 页面找到。
   - **Cascade** 中的新“Legacy Chat”模式在用户耗尽 **Flow Credits** 时激活，提供无需额外额度的有限功能。
- **Cascade 扩大了图片上传限制**：**Cascade image uploads** 不再受 1MB 限制，增强了用户上传大文件的能力。此更改是可用性和功能持续改进的一部分。
   - 更广泛的功能旨在为在工作流中使用图片功能的用户提供更无缝的用户体验。
- **Python 语言支持得到提升**：Windsurf 改进了对 **Python 的语言支持**，为用户提供更好的集成和功能。这一增强符合通过更有效的工具赋能开发者的目标。
   - 此外，还添加了一个自助升级方案按钮，使用户更容易通过[此链接](https://www.codeium.com/plan)访问更新的方案。
- **期待后续更多 Waves**：公告以对进一步更新的预告结束，称：*“这只是开始——请期待新一年中更多的 waves！”*
   - 有关持续更新，用户可以在 [blog](https://codeium.com/blog/windsurf-wave-1) 阅读完整公告，并在 [Twitter](https://x.com/windsurf_ai/status/1866948850205986926) 上关注团队。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Windsurf 编辑器的最新更新和更改。</li><li><a href="https://www.codeium.com/plan">Plan Settings</a>: 未来的编辑器，就在今天。Windsurf 编辑器是首款由 AI Agent 驱动的 IDE，让开发者保持心流状态。现已在 Mac、Windows 和 Linux 上可用。</li><li><a href="https://codeium.com/blog/windsurf-wave-1">Windsurf Wave 1</a>: 介绍 Wave 1，我们对 Windsurf 编辑器的第一批更新。</li><li><a href="https://x.com/windsurf_ai/status/1866948850205986926">来自 Windsurf (@windsurf_ai) 的推文</a>: 介绍 Wave 1。本次更新包含：🧠 Cascade Memories 和 .windsurfrules 💻 自动化终端命令 🪟 WSL、devcontainer、Pyright 支持... 以及更多。
</li>
</ul>

</div>
  

---


### **Codeium / Windsurf ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1316160459014930524)** (1 messages): 

> `Windsurf AI Twitter 抽奖活动` 


- **Windsurf 宣布周边抽奖活动**：Windsurf AI 兴奋地在 [Twitter](https://x.com/windsurf_ai/status/1866600392165048329) 上宣布了他们的首次周边抽奖活动，鼓励用户分享他们使用该平台构建的内容，以有机会赢取大礼包。
   - 参与者必须关注才能获得 **#WindsurfGiveaway** 的资格，这是用户展示其项目的好机会。
- **邀请在 Twitter 上展示构建成果**：邀请用户在 Twitter 上分享他们使用 Windsurf 创作的作品，以参与抽奖活动。
   - 这一行动号召旨在吸引社区参与并推广与 Windsurf 相关的用户生成内容。



**提到的链接**: <a href="https://x.com/windsurf_ai/status/1866600392165048329">来自 Windsurf (@windsurf_ai) 的推文</a>: 很高兴宣布我们的首次周边抽奖活动 🏄 分享你用 Windsurf 构建的作品，就有机会赢取大礼包 🪂 #WindsurfGiveaway 必须关注才有资格

  

---

### **Codeium / Windsurf ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1316132366317518878)** (239 条消息🔥🔥): 

> `Windsurf 中的额度问题、客户支持担忧、Cascade 和扩展的功能性、关于功能的社区讨论、产品更新与反馈` 


- **用户报告额度问题**：几位用户对无法使用购买的 Flex Credits 表示沮丧，其中一名成员指出，尽管已经购买，但额度并未计入其账户。
   - 另一位用户也分享说他们无法访问 Windsurf，并一直在寻求支持团队的帮助，但未得到回应。
- **对客户支持的担忧**：多位参与者提出了客户支持缺乏回应的问题，有人表示他们已经等待工单回复好几天了。
   - 一位用户讽刺地强调了糟糕的客户支持，建议需要改进沟通。
- **Cascade 的集成与功能**：用户讨论了 Cascade 的局限性，一位参与者指出，与手动使用相比，通过编程方式集成它非常困难。
   - 另一位用户表达了他们更倾向于向 Cascade 提问并处理其输出，而不是追求一个开发者友好的界面。
- **对 Codeium 扩展的反馈**：参与者分享了他们使用 Codeium 的 VS Code 扩展的经验，推测可能会有对不同模型的无限访问权限，但遇到了功能性问题。
   - 反馈中还包括对 Codeium 与聊天功能相关的操作错误的担忧。
- **产品更新与社区投入**：社区成员参与了关于最近更新的讨论，特别是关于新功能的博客文章中错误的视频链接。
   - 用户提出了一些改进建议，强调了他们的使用体验，并表达了对更好产品一致性的期望。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/disappointed-disbelief-slap-gif-14729546">失望怀疑 GIF - 失望怀疑扇耳光 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/bobawooyo-dog-confused-dog-huh-dog-meme-shocked-dog-gif-16491396616893958961">Bobawooyo 狗狗困惑 GIF - Bobawooyo 狗狗困惑 狗狗哈 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/clapping-leonardo-dicaprio-leo-dicaprio-well-done-applause-gif-1867988353188143738">鼓掌的莱昂纳多·迪卡普里奥 GIF - 鼓掌的莱昂纳多·迪卡普里奥 小李子 - 做得好 掌声 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>: 需要帮助？联系我们的支持团队以获得个性化协助。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h7sjyt/windsurf_cascade_leaked_system_prompt/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://codeium.com/blog/pricing-windsurf">计划与定价更新</a>: 我们对 Cascade 的定价模型进行了一些更改。</li><li><a href="https://github.com/codelion/optillm">GitHub - codelion/optillm: 针对 LLM 的优化推理代理</a>: 针对 LLM 的优化推理代理。通过在 GitHub 上创建账户为 codelion/optillm 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Codeium / Windsurf ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1316132365051101204)** (580 条消息🔥🔥🔥): 

> `Windsurf 更新、Cascade 模型问题、定价的用户体验、Windsurf 的功能请求、关于 AI 性能的社区反馈`

- **Windsurf 更新挑战**：用户报告了近期更新后的各种问题，包括“Accept All”按钮的问题以及与 Cascade 性能相关的错误，部分用户遇到了 HTTP 504 错误。
   - 尽管存在这些挫折，许多用户仍对新功能和 rules 功能表示兴奋，并为改进产品的讨论做出了积极贡献。
- **Cascade 模型性能担忧**：多位用户指出 Cascade Base 出现卡死或响应不当的情况，导致用户感到沮丧，并对其在编程任务中的可靠性表示担忧。
   - 有建议认为这种不稳定性可能与持续存在的 OpenAI 问题有关，导致许多用户考虑降级或切换到其他工具。
- **定价与订阅投诉**：针对 Windsurf 定价模型的批评较多，特别是与 flex credits 相关的限制，这些限制似乎不足以支持高强度使用。
   - 用户将其与 Cursor 的选项进行了对比，后者允许更灵活的订阅安排，凸显了用户对当前结构的不满。
- **用户需求功能**：用户希望获得关于 rules 使用的更好文档，并希望能够集成自定义 API endpoints 以实现更个性化的功能。
   - 社区还讨论了加入网页爬取和文件上传等功能的潜力，以增强可用性和有效性。
- **对 Windsurf 的总体评价**：用户的总体情绪从对 Windsurf 能力的赞赏到对近期小故障和功能限制的沮丧不等。
   - 许多用户积极寻求解决方案，分享经验并提供反馈，以帮助改进该工具。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/windsurf_ai/status/1866600392165048329">来自 Windsurf (@windsurf_ai) 的推文</a>：很高兴宣布我们的首个周边抽奖活动 🏄 分享你用 Windsurf 构建的作品，就有机会赢取礼包 🪂 #WindsurfGiveaway 必须关注才有资格</li><li><a href="https://tenor.com/view/exit-abort-wipe-out-surf-alex-dim-gif-14510313">Exit Abort GIF - Exit Abort Wipe Out - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://codeium.canny.io/feature-reques">Codeium</a>：未找到描述</li><li><a href="https://cursorlist.com/">CursorList - 为 Cursor AI 提供的 .cursorrule 文件及更多</a>：未找到描述</li><li><a href="https://status.openai.com/">OpenAI Status</a>：未找到描述</li><li><a href="https://status.openai.com">OpenAI Status</a>：未找到描述</li><li><a href="https://codeium.canny.io/feature-requests/">功能请求 | Codeium</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://codeium.canny.io/feature-requests/p/export-individual-chat-history">导出单个聊天记录 | 功能请求 | Codeium</a>：我能在一个对话会话中构建一个应用。我想将其导出并查看我执行的步骤，以便复盘并复制到另一个应用的构建中。</li><li><a href="https://docs.codeium.com/windsurf/cascade#memories">Windsurf - Cascade</a>：未找到描述</li><li><a href="https://codeium.canny.io/feature-requests">功能请求 | Codeium</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>：需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://tenor.com/view/burning-late-omg-what-happened-pizza-gif-11504119">Burning Late GIF - Burning Late Omg - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=thLNfIkCpg0">我测试了 AI 编程工具，结果可能会让你大吃一惊</a>：建议以 1.5 倍速观看，伙计们 🙏 如果你想一起学习 👉 https://discord.gg/CBC2Affwu300:00 介绍 + 测试提示词 01:25 Cursor 07:30 Windsurf 16:48 Aider 25:46...</li><li><a href="https://youtu.be/lED0yLrUelM?feature=shared">Gemini 2.0 Flash：史上最强 LLM！击败 Claude 3.5 Sonnet + o1！（全面测试）</a>：Gemini 2.0 Flash：史上最强 LLM？！我们将 Google 的新 AI 模型投入终极测试！它真的击败了 Claude 3.5 Sonnet 和其他模型吗？！在本视频中...</li><li><a href="https://tenor.com/view/windsurf-surf-tenerife-medano-e737-gif-18189634">Windsurf Tenerife GIF - Windsurf Surf Tenerife - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/PatrickJS/awesome-cursorrules">GitHub - PatrickJS/awesome-cursorrules: 📄 精选的优质 .cursorrules 文件列表</a>：📄 精选的优质 .cursorrules 文件列表。通过在 GitHub 上创建账号为 PatrickJS/awesome-cursorrules 的开发做出贡献。</li><li><a href="https://codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf 编辑器的最新更新和变化。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1316462684782329859)** (1 条消息): 

> `ChatGPT 与 Apple 集成，OpenAI 的 12 天活动，节日主题演示` 


- **ChatGPT 集成至 Apple 设备**：在[标题为](https://www.youtube.com/live/mBhkD0iFf4w?si=uMokZAeHp68wBwp2)“ChatGPT x Apple Intelligence—12 Days of OpenAI: Day 5”的 YouTube 视频中，Sam Altman、Miqdad Jaffer 和 Dave Cummings 穿着节日毛衣展示了 ChatGPT 在 **iOS** 和 **macOS** 中的新集成。
   - 通过在 <id:customize> 中领取 <@&1261377106890199132> 角色来*保持关注*，以获取 **12 Days of OpenAI** 期间的所有更新。
- **演示期间的节日氛围**：在演示过程中，团队成员穿着**节日毛衣**，在展示 ChatGPT 功能的同时增添了节日气氛。
   - 这种轻松的方式旨在吸引观众，并体现**节日季**的精神。



**提到的链接**：<a href="https://www.youtube.com/live/mBhkD0iFf4w?si=uMokZAeHp68wBwp2">ChatGPT x Apple Intelligence—12 Days of OpenAI: Day 5</a>：Sam Altman、Miqdad Jaffer 和 Dave Cummings 穿着节日毛衣介绍并演示了 ChatGPT 在 iOS 和 macOS 中的集成。

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1316140912606445569)** (517 条消息🔥🔥🔥): 

> `Gemini 2.0 Flash, OpenAI 服务停机, ChatGPT 使用策略, 图像生成性能, API 工具对比` 


- **Gemini 2.0 Flash 给用户留下深刻印象**：用户报告称 Gemini 2.0 Flash 的表现令人印象深刻，一些人将其基准测试结果与包括 o1 Pro 在内的 OpenAI 模型进行了对比，结果非常理想。
   - 空间推理和迭代图像编辑等功能受到关注，展示了其能力的显著进步。
- **OpenAI 服务遭遇停机**：据报道，由于系统问题，ChatGPT 服务无法使用，团队正在积极修复。
   - 几位用户注意到他们的 API 工具和聊天机器人也已宕机，这恰逢 Apple 智能集成发布之际。
- **最大化订阅使用率的策略**：用户寻求关于如何有效利用 o1 和 canvas 等新功能的建议，以确保从订阅中获得最大价值。
   - 建议包括使用 DALL-E 进行图像生成，使用 o1 处理推理任务，以及使用 canvas 构建项目，同时也承认某些功能目前处于离线状态。
- **对图像生成性能的担忧**：一些用户对 Gemini 2.0 的图像生成能力表示怀疑，提到他们在尝试图像变形（morphing）时得到了不理想的结果。
   - 虽然迭代编辑的演示看起来很有前景，但来自 Beta 测试人员的报告显示结果褒贬不一。
- **与其他模型（如 Llama 和 Google）的对比**：讨论强调了 Gemini 与 o1 和 Llama 模型的对比，特别是在定价和 Token 效率方面。
   - 用户注意到了模型功能和输出方面的竞争格局，特别是近期 Gemini 和 OpenAI 的发布。



**相关链接**: <a href="https://www.youtube.com/results?search_query=what+is+chatgpt+o1">未找到标题</a>: 未找到描述

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1316161807924334612)** (25 条消息🔥): 

> `Custom GPT Actions, API 错误处理, 平台停机, 场景文件格式` 


- **Custom GPTs 在 Action 调用方面存在困难**：成员们讨论了 **Custom GPTs** 在生成失败周期中可能多次调用外部 API 的问题。
   - 一位成员建议引入 Session ID，以避免重复调用，同时确保正常的 API 交互。
- **ChatGPT 停机状态更新**：已确认发生影响 ChatGPT 和 API 的**停机**事件，更新通过 [状态链接](https://status.openai.com/incidents/ctrsv3lwd797) 分享。
   - 更新报告称 API 流量已部分恢复，但指出截至 2024 年 12 月 11 日，**Sora** 仍处于宕机状态。
- **有效处理 API 错误**：围绕如果初始调用失败是否应**重复 API 调用**直到获得成功响应展开了讨论。
   - 成员们辩论了错误处理逻辑的有效性，暗示像 **403** 或 **500** 这样的错误响应对于修正逻辑可能很有用。
- **场景数据的最佳文件格式**：一位用户询问了用于 Custom GPT 的 **25 个场景**的最佳文件格式，这些场景目前存储在 Word 文档表格中。
   - 一位成员建议使用基础文本文件以简化数据并避免格式问题，并建议总结冗长的场景以便更好地检索。



**相关链接**: <a href="https://status.openai.com/incidents/ctrsv3lwd797">API, ChatGPT &amp; Sora 面临问题</a>: 未找到描述

  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1316152755185913932)** (11 messages🔥): 

> `微调问题，自定义 GPT 指令，工具集成挑战，Canmore 工具函数` 


- **用户在模型微调方面遇到困难**：一位正在开发基于 Node.js 的 OpenAI 应用的成员报告称，模型在微调后未能学习到上下文，导致输出 **通用回复 (generic responses)**。
   - 他们请求社区对其训练 JSONL 提供反馈，希望能从中识别出潜在问题。
- **操纵 GPT 中的工具使用顺序**：一位成员询问 GPT 指令是否可以规定其使用工具的顺序，并指出它总是先搜索知识库文件，然后再搜索 RAG API。
   - 另一位成员解释说，这可能是由于对编码回复的依赖，建议在远程 RAG 中合并文档，而不是将它们分开。
- **将多个工具链式调用**：同一位成员表示在单个 Prompt 中链式调用多个工具时遇到困难，指出模型成功执行了 API 调用，但没有后续的工具使用。
   - 回复建议包含元功能指令 (meta-functional instructions) 并使用规范工具名称 (canonical tool names)，以便更好地管理这些交互。
- **关于 Canmore 工具函数的细节**：分享了关于 Canmore 工具及其三个主要函数的详细描述：**create_textdoc**、**update_textdoc** 和 **comment_textdoc**，每个函数都有特定的参数。
   - 针对如何有效使用这些函数进行协作内容管理提出了建议。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1316152755185913932)** (11 messages🔥): 

> `微调 OpenAI 模型，自定义 GPT 工具使用，链式调用多个工具，Canmore 工具函数，受保护的聊天策略` 


- **微调 OpenAI 模型的挑战**：一位开发者表达了对他们基于 Node.js 的 OpenAI 应用的挫败感，称尽管完成了训练，微调后的模型仍然提供 **通用答案 (generic answers)**。
   - 他们请求协助审查其训练 JSONL 以发现潜在问题。
- **自定义 GPT 工具未按预期使用**：一位成员询问是否可以 Prompt GPT 按指定顺序使用工具，并指出其 API 经常被忽略，而优先使用了知识库文件。
   - 另一位成员强调，如果模型在其路径中检测到文档，可能会随机地 (stochastically) 无法优先处理 RAG API。
- **链式调用多个工具的困难**：一位用户报告了让 GPT 在单个 Prompt 中链式调用工具的挑战，指出它仅完成了第一个工具的 API 调用。
   - 建议包含元功能指令，以帮助 GPT 理解如何在回复中列出多个函数调用。
- **澄清 Canmore 工具函数**：一位成员详细介绍了 **Canmore** 工具的功能，包括以结构化方式创建、更新和评论文本文档的命令。
   - 他们分享了每个函数如何配合特定的 JSON 参数工作，以增强交互性和协作性。
- **聊天审核策略的讨论**：关于一个被屏蔽讨论的对话指向了 ChatGPT 设置中可能存在的措辞问题和保护策略。
   - 附带了一张图片来阐明审核声明，尽管没有分享具体细节。


  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1316132350169710645)** (410 条消息🔥🔥🔥): 

> `Cursor 性能问题, Agent 模式功能, 与其他 AI 工具的对比, Gemini 模型能力, Windsurf 的沟通与特性` 


- **Cursor 面临性能挑战**：用户报告了 Cursor 持续存在的问题，特别是请求响应时间缓慢以及代码库 indexing（索引）功能的问题，这让许多人感到沮丧，无法高效编码。
   - 停机似乎与 OpenAI 模型的上游问题有关，导致整个平台的性能下降。
- **Agent 模式表现不佳**：用户普遍抱怨 Cursor 的 Agent 模式功能，指出它在尝试处理代码或访问代码库时经常卡死。
   - 一些用户建议重新索引或重启 Cursor 可以暂时解决问题，但这仍然是一个反复出现的问题。
- **与其他 AI 工具的对比**：讨论强调了 Windsurf 等工具的优势，特别是与 Cursor 相比，它具有更可靠的沟通和更透明的 changelog（更新日志）。
   - 用户表示，虽然 Cursor 提供了独特的功能，但其 changelog 和用户反馈可以进一步改进，以更好地与竞争对手竞争。
- **Gemini 模型性能反馈**：用户分享了使用 Gemini 模型的不同经验，一些人注意到它的能力与其他模型相当，但在某些领域仍力有不逮，特别是与 Claude 的表现相比。
   - 尽管评价褒贬不一，用户仍对 Gemini 持续发展带来的潜在进步保持兴趣。
- **Windsurf 增强的社区参与度**：Windsurf 用户称赞该工具在其社区中清晰的沟通和响应能力，表示这展示了对用户反馈和产品改进的承诺。
   - Windsurf 良好的互动模式被强调为与 Cursor 的对比，引发了关于 Cursor 如何采取类似做法的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/sundarpichai/status/1866868228141597034">来自 Sundar Pichai (@sundarpichai) 的推文</a>：我们正在开启 Gemini 2.0 时代，推出了 Gemini 2.0 Flash，它在关键 benchmarks（基准测试）上以 2 倍的速度超越了 1.5 Pro（见下图）。我特别高兴看到在 c... 方面的快速进展。</li><li><a href="https://supermaven.com/">Supermaven: 免费 AI 代码补全</a>：最快的 Copilot。Supermaven 使用 100 万 token 的 context window（上下文窗口）来提供最高质量的代码补全。</li><li><a href="https://forum.cursor.com/t/feature-request-long-context-mode/32187/2">功能请求：长上下文模式</a>：我看到对这条评论做出反应的人比给该功能投票的人还多。提醒一下，你需要点击正式的投票按钮。</li><li><a href="https://x.com/aryanvichare10/status/1866561638712881172?s=46">来自 Aryan Vichare (@aryanvichare10) 的推文</a>：介绍 WebDev Arena，这是一个两个 LLM 竞争构建 Web 应用的竞技场。你可以投票选出表现更好的 LLM，并查看最佳模型的排行榜。100% 免费且开源，由 @lmarena_ 提供支持...</li><li><a href="https://x.com/OpenAI/status/1866578914233159928">来自 OpenAI (@OpenAI) 的推文</a>：Canvas——一种与 ChatGPT 协作起草、编辑并获取写作与代码反馈的新方式——现在已向所有 4o 模型用户开放。它已在 Web 端和 ChatGPT 桌面端全面推出...</li><li><a href="https://tenor.com/view/unemployment-unemployed-laid-off-layoffs-layoff-gif-17329141">失业 GIF - Unemployment Unemployed Laid Off - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/wait-what-wait-a-minute-huh-gif-17932668">等等 什么 等一下 GIF - Wait What Wait A Minute Huh - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/wtff-cat-confused-orange-gif-25858653">Wtff 猫 困惑 橘猫 GIF - Wtff Cat Confused Orange - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://status.cursor.com/">Cursor 状态</a>：未找到描述</li><li><a href="https://codeium.com/changelog">Windsurf 编辑器 Changelogs | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf 编辑器的最新更新和变化。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1316505161794846851)** (1 条消息): 

> `Neural network training, Training Jacobian analysis, Parameter dependence, Bulk and chaotic subspaces, Training dynamics` 


- **揭秘神经网络参数依赖关系**：在一篇新论文中，研究人员分析了 **训练 Jacobian**，通过检查导数矩阵揭示了最终参数如何依赖于初始参数。
   - 他们发现训练会拉伸并旋转参数空间，导致产生受训练影响极小的 **bulk** 区域，以及扰动会被放大的 **chaotic** 区域。
- **训练中的 Bulk 与 Chaotic 动力学**：论文识别了两个关键子空间：**bulk** 子空间的奇异值 (SVs) 接近 1，训练不会对其产生显著改变；而 **chaotic** 子空间的奇异值大于 1，其中的变化会被放大。
   - 有趣的是，与真实数据相比，在白噪声上训练时 bulk 的维度更小，这表明在处理无结构输入时需要更激进的参数压缩。
- **训练 Jacobian 的计算挑战**：计算限制了对完整训练 Jacobian 的分析，因为它对于大型网络来说仍然是不可计算的，因此他们的实验重点关注了一个较小的 5K 参数 MLP。
   - 尽管如此，在一个 62K 参数的图像分类器中也观察到了类似的谱模式，表明这可能为更广泛的神经训练动力学提供见解。
- **即将推出的训练动力学研究系列**：这项研究是关于 **神经网络训练动力学** 和损失景观几何系列研究的第一篇，鼓励社区参与到正在进行的工作中。
   - 欲了解更多信息和文献，请参阅专门的频道以获取有关该主题的协作和见解。
- **探索论文与代码**：论文已发布在 [arXiv](https://arxiv.org/abs/2412.07003)，并提供了 PDF 链接和实验 HTML。
   - 其他资源包括相关的 [GitHub 仓库](https://github.com/EleutherAI/training-jacobian)，用于为该项目做出贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.07003">Understanding Gradient Descent through the Training Jacobian</a>：我们利用训练后的网络参数相对于其初始值的 Jacobian 矩阵来研究神经网络训练的几何结构。我们的分析揭示了训练中存在的低维结构...</li><li><a href="https://github.com/EleutherAI/training-jacobian">GitHub - EleutherAI/training-jacobian</a>：通过在 GitHub 上创建账号来为 EleutherAI/training-jacobian 的开发做出贡献。</li><li><a href="https://x.com/norabelrose/status/1866943688993370381">来自 Nora Belrose (@norabelrose) 的推文</a>：神经网络的最终参数如何依赖于其初始参数？在这篇新论文中，我们通过分析训练 Jacobian（即最终参数的导数矩阵）来回答这个问题...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1316142935607345315)** (78 条消息🔥🔥): 

> `HumanEval 评估，OpenAI 员工对训练数据的见解，RWKV 架构与模型，AdamW 权重衰减` 


- **对 HumanEval 评估的兴趣**：一名成员表示有兴趣运行 [HumanEval 评估](https://github.com/EleutherAI/lm-evaluation-harness/pull/1992)，并询问了关于其集成的长期未解决的 Pull Request。
   - 他们正在寻求有关是否有计划审查并推进该 PR 的信息。
- **训练数据优先**：一篇来自 OpenAI 员工的帖子讨论了训练数据的重要性，断言在经过广泛实验后，模型表现高度趋近于其数据集。
   - 对话中认识到，虽然模型行为似乎源于数据集，但也涉及模型架构内部的潜在偏见。
- **发布新的 RWKV 模型**：一名成员宣布发布了新的 RWKV 架构，特别是 [Flock of Finches](https://huggingface.co/rwkv/Finch-MoE-37B-A11B-HF) 和 [QRWKV-6 32B](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)，两者都声称优化了训练成本。
   - 他们强调了这些模型在保持较低计算需求的同时，具有与更大规模模型相当的性能。
- **呼吁澄清 AdamW 设置**：一名成员建议 AdamW 中的权重衰减（weight decay）设置对于理解对奇异值的影响至关重要，应在相关论文中提及。
   - 他们指出权重衰减会影响矩阵的谱（spectrum），并指出在实验中澄清权重衰减是开启还是关闭会很有帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://nonint.com/2023/06/10/the-it-in-ai-models-is-the-dataset/">The &#8220;it&#8221; in AI models is the dataset. &#8211; Non_Interactive &#8211; Software &amp; ML</a>: 未找到描述</li><li><a href="https://gwern.net/aunn">Absolute Unit NNs: Regression-Based MLPs for Everything · Gwern.net</a>: 未找到描述</li><li><a href="https://substack.recursal.ai/p/flock-of-finches-rwkv-6-mixture-of">Flock of Finches: RWKV-6 Mixture of Experts</a>: 迄今为止最大的 RWKV MoE 模型！</li><li><a href="https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1">recursal/QRWKV6-32B-Instruct-Preview-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://substack.recursal.ai/p/q-rwkv-6-32b-instruct-preview">Q-RWKV-6 32B Instruct Preview</a>: 迄今为止最强、最大的 RWKV 模型变体：QRWKV6 32B Instruct Preview</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1992">Add HumanEval by hjlee1371 · Pull Request #1992 · EleutherAI/lm-evaluation-harness</a>: 你好，我添加了广泛使用的 HumanEval 基准测试。这部分解决了 #1157。该实现依赖于 HF evaluate 模块的 pass@k，因此需要环境变量 HF_ALLOW_COD...</li><li><a href="http://www.incompleteideas.net/IncIdeas/BitterLesson.html">The Bitter Lesson</a>: 未找到描述</li><li><a href="https://gwern.net/scaling-hypothesis">The Scaling Hypothesis · Gwern.net</a>: 未找到描述</li><li><a href="https://gwern.net/scaling-hypothesis#scaling-hypothesis">The Scaling Hypothesis · Gwern.net</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=YEUclZdj_Sc">Why next-token prediction is enough for AGI - Ilya Sutskever (OpenAI Chief Scientist)</a>: 完整剧集：https://youtu.be/Yf1o0TQzry8 访谈文本：https://www.dwarkeshpatel.com/p/ilya-sutskever Apple Podcasts：https://apple.co/42H6c4D Spotify：https://...</li><li><a href="https://x.com/karpathy/status/1733299213503787018?lang=en">Tweet from Andrej Karpathy (@karpathy)</a>: # 关于“幻觉问题”：当我被问及 LLM 中的“幻觉问题”时，我总是感到有些纠结。因为从某种意义上说，幻觉是 LLM 唯一在做的事情。它们是梦...</li><li><a href="http://prize.hutter1.net/#:~:text=hrules.htm.-,Motivation,-This%20compression%20contest">500'000&euro; Prize for Compressing Human Knowledge</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1316132511591432223)** (296 条消息🔥🔥): 

> `Muon 优化器性能，理解 LLM 中的有害偏见，神经网络中的正则化技术，Transformer 中权重衰减的影响，梯度正交化的益处`

- **Muon 优化器展现出潜力**：Muon 已被公认为近期优化器开发中的有力竞争者，具有前景良好的 baseline 性能和深刻的底层数学原理。
   - 虽然尚未确认其性能优于 AdamW，但其潜力已得到认可，特别是考虑到在 Llama 3 等现有模型中观察到的优化器问题。
- **解决 LLMs 中的偏见问题**：即将发表的一篇论文指出，大型语言模型中的有害偏见是其当前设计的必然结果，这需要重新评估 AI 的基础假设。
   - 这一观点强调需要更深入地理解偏见产生的原因，而不是仅仅关注具体的实现方式。
- **探索正则化方法**：关于 weight decay 等正则化项在神经网络训练中的复杂作用，目前正在进行持续讨论，特别是在 Transformer 的 attention 层中。
   - L2 regularization 与特征表示之间的关系凸显了进一步研究不同正则化技术如何影响模型性能的必要性。
- **Gradient orthogonalization 作为正则化项**：先前关于 gradient orthogonalization 的研究表明，它作为一种强正则化项非常有效，能够促进深度学习模型中的特征多样性。
   - 实施 batch whitening 而非 batch normalization 的提议可能会产生类似的收益，特别是在小 batch size 的场景下。
- **Weight decay 在训练中的影响**：一项研究讨论了 weight decay 和 L2-regularization 对深度神经网络的影响，特别是关于 attention 机制中参数矩阵内的乘法交互。
   - 研究结果表明，L2-regularized 损失可能会迅速收敛到涉及 nuclear norm regularization 的相同公式，这引发了对其在训练期间效用的质疑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.07752">FlashRNN: Optimizing Traditional RNNs on Modern Hardware</a>: 虽然 Transformers 和其他序列并行化神经网络架构似乎是目前序列建模的最先进技术，但它们特别缺乏状态跟踪能力。这些...</li><li><a href="https://arxiv.org/abs/2412.06464">Gated Delta Networks: Improving Mamba2 with Delta Rule</a>: Linear Transformers 作为标准 Transformers 的高效替代方案受到了关注，但它们在检索和长上下文任务中的表现一直有限。为了解决这些局限性，...</li><li><a href="https://kellerjordan.github.io/posts/muon/">Muon: An optimizer for hidden layers in neural networks | Keller Jordan blog</a>: 未找到描述</li><li><a href="https://x.com/YouJiacheng/status/1866734331559071981">Tweet from YouJiacheng (@YouJiacheng)</a>: 新的 NanoGPT 训练速度记录：3.80 分钟内达到 3.28 FineWeb 验证损失。更新日志：拆分 Value Embs，DDP(gradient_as_bucket_view=Tr...</li><li><a href="https://arxiv.org/abs/2410.23819">Weight decay induces low-rank attention layers</a>: 训练深度神经网络时，权重衰减（weight decay）等正则化项的作用尚不完全清楚。我们研究了训练神经网络时权重衰减以及 $L2$-正则化的影响...</li><li><a href="https://recall2imagine.github.io/">Recall to Imagine</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2412.05270">APOLLO: SGD-like Memory, AdamW-level Performance</a>: 大语言模型（LLMs）在训练期间的内存消耗极高，尤其是在使用流行的 AdamW 优化器时。这种内存负担使得必须使用更多或更高规格的 GPU，或者减少...</li><li><a href="https://arxiv.org/abs/2203.00555">DeepNet: Scaling Transformers to 1,000 Layers</a>: 在本文中，我们提出了一种简单而有效的方法来稳定极深层的 Transformers。具体来说，我们引入了一种新的归一化函数（DeepNorm）来修改残差连接...</li><li><a href="https://arxiv.org/abs/2406.13138">Large Language Models are Biased Because They Are Large Language Models</a>: 本文的主要目标是引发关于偏见与大语言模型基本属性之间关系的深入讨论。我们试图通过说服读者...</li><li><a href="https://arxiv.org/abs/1505.00387">Highway Networks</a>: 大量的理论和实证证据表明，神经网络的深度是其成功的关键因素。然而，随着深度的增加，网络训练变得更加困难...</li><li><a href="https://arxiv.org/abs/2410.17897">Value Residual Learning For Alleviating Attention Concentration In Transformers</a>: Transformers 可以利用 Self-Attention 捕捉长程依赖，允许 Token 直接关注所有其他 Token。然而，堆叠多个注意力层会导致注意力集中（attention concentration）。我们的...</li><li><a href="https://arxiv.org/abs/2402.02622">DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging</a>: Vaswani 等人（2017）提出的 Transformer 架构现在在从自然语言处理到语音处理和图像理解的各个应用领域无处不在。我们提出了 DenseFormer...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1316323582032220181)** (7 条消息): 

> `lm_eval_harness, Perplexity evaluation, Batch processing in inference frameworks, Token processing utility, AOTriton updates` 


- **使用 lm_eval_harness 评估模型 Perplexity**：一位成员正在使用 **lm_eval_harness** 评估 **jsonl 文件** 中数据的模型 **perplexity**，并分享了他们的自定义任务配置。
   - 他们需要 **per-token perplexity** 结果以便与另一项研究进行对比，并就潜在问题寻求建议。
- **单样本处理导致 Inference 变慢**：有成员担心 **inference** 速度缓慢，因为 **jsonl 文件** 中的每个样本都是单独处理的，这可能导致效率低下。
   - 该成员指出，每个样本都会被单独进行 **tokenized** 和 **inferred**，这表明需要进行 **batch processing**。
- **启用 Batch Inference 的 Pull Request**：一位成员通过提交 [PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/2559) 提供了解决处理延迟的方案，该 PR 实现了跨推理请求的 batch 处理。
   - 他们确认了确保请求之间没有 **attention** 干扰的重要性，并对及时的帮助表示感谢。
- **关于 Token 处理函数的疑问**：讨论了 token 处理函数，指出它会根据 tokenizer 输入计算不同的 perplexity。
   - 虽然计算了 **token_perplexity**、**word_perplexity** 和 **byte_perplexity** 等关键指标，但成员们仍在寻求关于效率提升的进一步说明。
- **[ROCm] 更新至 AOTriton 0.8b**：分享了一个与 **AOTriton 0.8b** 相关的 commit，该版本为 AMD 系统上的 **SDPA 算子** 引入了新功能，包括 **nested tensor** 支持。
   - commit 消息中强调了显著的新特性，突出了此次更新带来的改进。



**提到链接**: <a href="https://github.com/pytorch/pytorch/commit/424156c26c5a80c9221197c09c2d1c12006f11d1">[ROCm] Update to AOTriton 0.8b (#140172) · pytorch/pytorch@424156c</a>：来自 AOTriton 0.8b 的 AMD 系统 SDPA 算子的显著新特性：1. Nestedtensor 支持；2. MQA/GQA 支持；3. 恢复对 causal=True 且 seqlen_q != seqle... 的 Efficient attention 支持。

  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 条消息): 

tensor_kelechi: https://machinelearning.apple.com/research/multimodal-autoregressive
  

---


### **Bolt.new / Stackblitz ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1316480454404083713)** (3 条消息): 

> `OSS Bolt, YouTube Streams, Supabase Integration` 


- **Cole 直播谈论 OSS Bolt**：Cole 在直播环节中讨论了 **OSS Bolt**，社区成员在 [Twitter](https://x.com/stackblitz/status/1866867336730628513) 上进行了观看。
   - 这场互动对话深入探讨了围绕 OSS Bolt 的历程和发展。
- **在 YouTube 上观看 Bolt Office Hours**：由 Cole 主持的最新一期 **Bolt Office Hours: Week 8** 现已在 [YouTube](https://www.youtube.com/watch?v=xlbKobsqfdc) 上线。
   - 本期节目包含了与讨论相关的各种资源链接，增强了观众的参与感。
- **Supabase 集成预览**：社区在直播期间看到了令人兴奋的 Supabase 集成**预览**。
   - 这一集成看起来将增强现有工作流的功能，并吸引开发者的兴趣。



**提到链接**: <a href="https://www.youtube.com/watch?v=xlbKobsqfdc">Bolt Office Hours: Week 8</a>：🔗 相关链接 Bolt.diy: https://bolt.diy Cole Medin YouTube: https://www.youtube.com/@ColeMedin Bolt.diy 公告: https://twitter.com/stackblitz/status/18668673...

  

---

### **Bolt.new / Stackblitz ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1316155032344854630)** (7 条消息): 

> `Web App Development, Shopify API Integration, Data Transformation Tools, Airtable Integration, Webhook Scenarios` 


- **构建带有产品同步功能的内部仪表盘**：一位成员正在为公司开发一个 Web App，旨在创建一个能与 Shopify 商店同步的内部仪表盘。
   - *Isoprophlex0* 建议通过集成 Shopify API 来确保数据仅供该应用访问，并强调了 [Shopify API Docs](https://shopify.dev/docs/api) 中提供的详尽文档。
- **利用 Airtable 进行库存管理**：另一位成员确认他们已将 Airtable 上的完整库存与使用 Bolt 构建的 Web App 进行了同步。
   - *Isoprophlex0* 建议将文档转换为 Bolt 可读的格式，并可能利用另一个 AI 来辅助此过程。
- **使用 Make.com 进行文件操作**：一位用户分享了使用 Make.com 进行文件操作和 LLM 传输的经验，称其通常比传统方法更高效。
   - 他们建议利用 Webhook 场景来简化将数据发送到数据库以供显示的过程。



**提到的链接**：<a href="https://shopify.dev/docs/api">Shopify API, libraries, and tools</a>：了解 Shopify API、库和工具，并为您的用例选择合适的选项。

  

---


### **Bolt.new / Stackblitz ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1316155963564097667)** (235 条消息🔥🔥): 

> `Bolt AI performance, Firebase vs Supabase, Error handling in Bolt, Token usage concerns, Community support in Bolt` 


- **Bolt AI 在错误处理方面表现挣扎**：用户反馈 Bolt AI 经常犯重复错误，且在消耗大量 Token 的情况下仍无法实施更改或解决问题。
   - 一位用户强调花费了 **200k tokens** 却没有任何实质性改变，并将部分问题归因于底层的 AI 模型 Claude。
- **Stripe 集成：Firebase vs Supabase**：几位用户讨论了从 Firebase 迁移到 Supabase 以获得更好的 Stripe 集成体验，并权衡了每种数据库的优缺点。
   - 值得注意的是，虽然两者都是可行的选择，但最终决定取决于具体的应用需求，一些用户倾向于选择 Supabase 以避免供应商锁定（vendor lock-in）。
- **Bolt 中的 Token 消耗与调试**：用户普遍担心调试期间 Token 消耗过快，许多用户分享了在未解决问题的情况下产生巨额 Token 支出的经历。
   - 建议包括优化 Prompt 的构建方式，以减少与 AI 协作时浪费的 Token。
- **社区支持与资源**：用户对缺乏直接技术支持表示沮丧，目前主要依靠社区协助进行故障排除。
   - 参与者指出 Discord 和 GitHub 等社区频道对于分享挑战和解决方案的重要性。
- **跨平台兼容性**：关于在 Bolt 与 Cursor 等其他平台之间传输代码的能力提出了疑问，强调了对更流畅集成的需求。
   - 用户在 Bolt 中遇到困难时，寻求更直接的代码迁移路径。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1316137030903664662)** (152 messages🔥🔥): 

> `Qwen 2.5 Fine-tuning, Gemini Voice, AWQ and Adapter Models, Context Length Capabilities, Performance Evaluation of Voice Models` 


- **Qwen 2.5 的微调挑战**：一位用户指出在微调 Qwen 2.5 模型时难以获得数字输出，特别是在处理简单的乘法查询时。
   - 讨论强调了添加特定领域知识的挑战性，表明预训练或利用 RAG 解决方案可能更有效。
- **对 Gemini Voice 的反馈**：成员们对 Gemini Voice 的功能感到兴奋，称其表现“疯狂”，优于某些替代方案。
   - 用户将其与其他语音模型进行了比较，一些用户指出 OpenAI 的实时语音 API 成本高昂且功能较弱。
- **关于 AWQ 和 Adapter 使用的见解**：讨论提到，虽然无法直接使用 AWQ 模型进行训练，但使用 Adapters 是可行且有效的。
   - 成员们评论道，即使在不支持直接训练模型的场景下，巧妙地使用 Adapters 也是一种良策。
- **Qwen 2.5 的上下文长度测试**：一位用户正在 Qwen 2.5 的 3B 版本上实验 65k 上下文，并报告了对性能提升的积极预期。
   - 针对模型处理 65,536 个上下文 Token 的能力存在期待，这预示着微调过程的整体效率将有所提高。
- **语音模型的性能问题**：关于特定语音模型（如 GLM）特性的讨论，其中出现了输出中缺失 Logits 等问题。
   - 成员们对架构表示困惑，提到模型输出的是二进制 Token 而非标准概率，这引起了混淆。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/troubleshooting/errors">Errors | Unsloth Documentation</a>：要修复设置中的任何错误，请参阅下文：</li><li><a href="https://unsloth.ai/blog/llama3-3">Fine-tune Llama 3.3 with Unsloth</a>：微调 Meta 的 Llama 3.3 (70B) 模型，其性能优于 GPT 4o，通过 Unsloth 开源实现 2 倍提速！对初学者友好。现已加入 Apple 的 Cut Cross Entropy 算法。</li><li><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF">unsloth/Llama-3.3-70B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1fnvlla/qwen25_bugs_issues_fixes_colab_finetuning_notebook/lonlt7s/">Qwen2.5 Bugs &amp; Issues + fixes, Colab finetuning notebook</a>：当我使用 4-bit bnb 和 transformers（在 ooba 中）加载 Qwen 32b base，并在 notebook 模式下仅使用 <|im_start|> 进行提示时，我想...</li><li><a href="https://docs.unsloth.ai/basics/reward-modelling-dpo-orpo-and-kto">Reward Modelling - DPO, ORPO &amp; KTO | Unsloth Documentation</a>：要通过 Unsloth 使用 DPO, ORPO 或 KTO，请遵循以下步骤：</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hbaioc/llama_33_70b_finetuning_now_with_90k_context/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1316343144186642563)** (10 messages🔥): 

> `Unsloth Merch, Dropshipping Challenges, Ecommerce Concerns` 


- **Unsloth 周边需求旺盛**：社区对推出 **Unsloth 周边**有浓厚兴趣，但有人对物流配送提出了担忧。
   - 一位社区成员表示：“配送将会是一场噩梦”。
- **Dropshipping 作为解决方案**：有人建议利用 Printful 等 **Dropshipping 服务**来缓解配送担忧。
   - 这种方法可能消除直接管理配送的需求。
- **Dropshipping 的质量和成本问题**：尽管 Dropshipping 表面上很简便，但成员们警告可能会出现 **QA（质量保证）挑战**。
   - 一位成员提到，如果**质量受损**，*RMA（退货授权）成本可能会超过产品的利润*。
- **电商创业者需警惕**：分享了一个警告，提醒许多尝试涉足**电子商务和 Dropshipping** 的人往往会面临意想不到的成本。
   - 成员们建议“谨慎行事”，因为他们可能会面临沉重的财务负担。
- **Dropshipping 可能是一个陷阱**：对话以一个隐喻结束，指出 Dropshipping 可能是一个**充满挑战的陷阱**。
   - 一位成员幽默地指出，这个过程中“处处是危机（there will be dragons）”。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1316142136521134125)** (58 条消息🔥🔥): 

> `学习 CUDA 和 Triton，使用自定义数据集进行微调，训练期间的内存管理，使用 llama.cpp 进行模型转换，领域自适应的数据质量` 


- **寻求 CUDA/Triton Kernel 开发资源**：一位用户询问了学习 **CUDA** 和 **Triton kernel 开发**的优质资源，寻求社区指导。
   - 另一位成员建议参考该主题早期的博客文章，特别提到了 Unsloth 的介绍，其中概述了其性能优势。
- **使用自定义数据进行微调的挑战**：一位成员提到在使用 **Unsloth** 进行微调时遇到困难，因为 **UnslothTrainer** 要求其自定义数据集中包含单个 `text` 列。
   - 建议他们可以尝试添加一个模拟文本列，并确保 input-output 对被 data collator 正确处理。
- **管理训练期间的内存峰值**：有用户对训练阶段 **RAM** 和 **GPU RAM** 使用量突然飙升表示担忧，这导致了 Colab 上的运行崩溃。
   - 几位用户建议调整 batch sizes，平衡数据集长度，并强调了高质量数据对有效训练的重要性。
- **使用 Llama.cpp 进行模型转换**：围绕使用 **llama.cpp** 进行模型转换展开了讨论，用户分享了经验并排查了遇到的问题。
   - 一位成员通过执行全新安装并使用特定版本的依赖项（包括 **PyTorch 2.2**）成功解决了问题。
- **Collators 中的 Tokenization 问题**：一位用户在尝试使用自定义 collator 让模型在最后一个 token 上进行训练时，遇到了关于无法识别 token 的 **ValueError**。
   - 该 collator 被分享了出来，凸显了在处理文本和图像数据时管理 label masking 以进行有效训练的复杂性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/introducing">Introducing Unsloth</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/unsloth/comments/1bnm3yd/validation_dataset/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://pastebin.com/JCBTDFex">accelerate==1.2.0aiohappyeyeballs==2.4.4aiohttp==3.11.10aiosignal==1.3.1 - Pastebin.com</a>: Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户，为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1316188345163845774)** (5 条消息): 

> `AI 系统中的角色，受限生成，特征提取，审核技术` 


- **探索 AI 系统中的角色**：一位成员建议**深入探讨** AI 系统中的各种角色（system/user/assistant），涵盖**人格卡片 (personality cards)** 和**角色扮演 (roleplay)** 等方面。
   - 讨论强调了更好地理解角色对于增强功能和用户交互的重要性。
- **利用受限生成 (Constrained Generation)**：提出了另一个想法，即通过 **JSONSchema** 和语法解释**受限生成**，强调其在改进代码和**特征提取**方面的优势。
   - 重点在于这如何带来更好的 **RAG** (Retrieval-Augmented Generation) 并确保**完美的函数调用 (function calling)**。
- **主题优先级决策困境**：关于先处理哪个主题展开了一场有趣的辩论，成员们表达了兴奋和幽默。
   - *“两个都要！！！拜托了...”* 凸显了解决这两个课题的渴望，展现了协作精神。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1316165434029903952)** (4 messages): 

> `WizardLM Arena 数据集, OpenPlatypus 数据集, QwQ 模型转换, MATH 数据集` 


- **WizardLM Arena 数据集可用**：一位成员将 **WizardLM Arena** 论文中使用的所有数据集上传到了[此仓库](https://huggingface.co/datasets/forcemultiplier/arena-paper-datasets-jsonl/tree/main)。提交记录 [b31fa9d](https://huggingface.co/datasets/forcemultiplier/arena-paper-datasets-jsonl/commit/b31fa9dba65f1931cfbd1daf37f35b7c94afecf2) 在一天前刚经过验证。
- **OpenPlatypus 数据集见解**：分享了另一个名为 **OpenPlatypus** 的数据集，包含 **25k 个样本**，并指出在 OpenRouter 上进行测试的成本为 **30 美元**。建议排除长度大于 **5000** 且小于 **100** tokens 的回答，以获得更干净的结果。
- **创建模型的 QwQ 版本**：一位成员提到，可以利用此方法将任何模型转换为 **QwQ** 版本，目前正在制作 **14B** 和 **3B** 的 Qwen 模型。这些模型的测试尚未进行。
- **数学能力测试数据集可用**：**MATH** 数据集包含带有详细解答的竞赛题目，可以在[这里](https://huggingface.co/datasets/hendrycks/competition_math)找到。它可用于训练模型生成答案推导和解释。
- **使用 MATH 数据集进行基准测试**：建议使用 **qwq** 为 **MATH** 数据集中的题目生成说明，并与 ground truth 答案进行对比。指出该数据集有时被用于 benchmark，但可以进行过滤以移除 benchmark 题目。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/hendrycks/competition_math">hendrycks/competition_math · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/forcemultiplier/arena-paper-datasets-jsonl/tree/main">forcemultiplier/arena-paper-datasets-jsonl at main</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/forcemultiplier/QwQ_OpenPlatypus_25k_jsonl">forcemultiplier/QwQ_OpenPlatypus_25k_jsonl · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1316142733068337192)** (3 messages): 

> `新项目频道, Forge Discord 机器人访问, Hermes 3 LLM 发布` 


- **新项目协作频道上线**：为项目创建了一个新的协作频道，允许成员们有效地共同工作。
   - 这是成员们在服务器中建立联系并共同构建的机会。
- **通过 Discord 机器人即时访问 Forge**：成员现在可以通过 Discord 机器人访问 **Forge**，无需 API 审批，方便立即进行测试。
   - 要开始使用，请在 **customize** 页面选择 **Forge Beta** 角色，然后前往相应的频道。
- **介绍 Hermes 3 LLM：小而强大**：**Hermes 3 3B** 现已在 [Hugging Face](https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B) 上线，并提供量化 GGUF 版本以优化体积和性能。
   - Hermes 3 在用户对齐 (user alignment)、Agent 性能和推理 (reasoning) 方面带来了先进的能力，标志着从 **Hermes 2** 开始的重大升级。



**提及的链接**: <a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B">NousResearch/Hermes-3-Llama-3.2-3B · Hugging Face</a>: 未找到描述

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1316132741913514084)** (90 条消息🔥🔥): 

> `Nous Forge Access, Quantum Computing Updates, Neurofeedback Research, AI Collaboration Proposals, Creative AI Simulation` 


- **解锁 Nous Forge 访问权限**：一位成员询问如何获得 Nous Forge beta 版的访问权限，表达了在漫长等待后希望得到回复的愿望。
   - 另一位成员提到，用户可以通过 <id:customize> 链接并点击相应的按钮来获取 beta 版。
- **用于开发的 AI 额度**：一位用户分享了一个提供免费 AI 额度的链接，表示这对于那些正在扩展 AI 开发项目的用户很有帮助。
   - 该帖子提到，通过与 Eleven Labs 的合作，提供了来自顶级平台（包括 BFL API）的 50 美元以上的额度。
- **神经反馈领域的令人兴奋的新研究**：分享了一项关于通过 fMRI 成像的神经反馈在脑中“写入”新学习模式的技术研究。
   - 这种方法在隐性学习方面展现了极具前景的见解，并可能为神经精神疾病带来新的治疗方法。
- **量子计算突破**：一位成员分享了一篇关于量子设备噪声以及逻辑量子比特（logical qubits）如何帮助减轻误差的文章，强调了它们在量子计算中的重要性。
   - 论文指出，逻辑量子比特代表了量子设备迈向实际应用过程中的关键进展。
- **创意 AI 探索**：讨论集中在成员们尝试 World Simulator 和创意 AI 工具（如模拟个人体验）。
   - 成员们对构建 Agent 和探索 AI 的创意潜力表现出浓厚兴趣，并指出了模拟过程中的不可预测性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blogs.nvidia.com/blog/logical-qubits-cuda-q-demo/">Turn Down the Noise: CUDA-Q Enables Industry-First Quantum Computing Demo With Logical Qubits</a>: Infleqtion 发布了突破性成果，利用 NVIDIA CUDA-Q 平台设计并演示了涉及两个逻辑量子比特的实验。</li><li><a href="https://x.com/nousresearch/status/1866584568548995538?s=61">Tweet from Nous Research (@NousResearch)</a>: 宣布推出 Nous Simulators！这是我们所有涉及社交领域人机交互实验的大本营。 http://sims.nousresearch.com</li><li><a href="https://neurosciencenews.com/neurofeedback-learning-neuroscience-28219/">Can We Program the Brain to Learn Without Teaching? - Neuroscience News</a>: 研究人员开发了一种突破性技术，利用来自 fMRI 成像的实时神经反馈，直接在大脑中“写入”新的学习模式。</li><li><a href="https://neurosciencenews.com/sensory-perception-noise-signals-28195/">How the Brain Sorts Noise from Signal to Maintain Stable Perception - Neuroscience News</a>: 新研究揭示了大脑如何将内部产生的噪声与感官信号分离，以确保稳定的感知。</li><li><a href="https://medicalxpress.com/news/2024-12-dopamine-neuroscientists.html">New look at dopamine signaling suggests neuroscientists' model of reinforcement learning may need to be revised</a>: 多巴胺是大脑中的一种强大信号，影响着我们的情绪、动机、运动等。这种神经递质对于基于奖励的学习至关重要，而这一功能在许多疾病中可能会受到干扰……</li><li><a href="https://x.com/bfl_ml/status/1866891754974199832">Tweet from Black Forest Labs (@bfl_ml)</a>: 我们很高兴推出 AI Engineer Pack，以加速您的 AI 开发。我们与 @elevenlabsio 合作，为顶级平台（包括 BFL API）提供 50 美元以上的额度。无论您是……</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1316287055805808693)** (62 条消息🔥🔥): 

> `Coconut 模型, KV-cache 机制, LLM 中的 Thought tokens, 模型中的 Amnesia 模式, iOS 上的自定义 LLM` 


- **Coconut 模型处理思维**：Coconut 模型利用 Embedding 层处理初始序列，在生成响应之前创建一系列 Thought tokens。
   - 该方法在整个响应生成过程中，优先处理初始序列与连续思维之间的 Attention。
- **LLM 中的 KV-cache 机制**：讨论揭示了为某些层提供更大 KV-cache 以增强 LLM 状态构建的潜力。
   - 预计 Attention 机制将利用额外的 KV 向量来改进处理。
- **Thought tokens 增强推理**：有人提议使用特殊的 Thought tokens，作为一种在推理过程中促进推理且不拥塞初始层的方法。
   - 实验表明，此类 Token 可能会带来更有效的输出，但之前的研究表明，循环 Embedding 可能更高效。
- **Amnesia 模式的可用性**：关于新的 smol Hermes 模型是否具有 Amnesia 模式存在疑问，但成员报告称该模式似乎不存在。
   - 即使在 Prompt 为空的情况下也会出现随机延续，这表明该功能并不一致。
- **iOS LLM 应用能力**：关于运行 LLM 的 iOS 应用程序的咨询显示，大多数应用支持自定义下载模型。
   - 这为用户尝试包括 Hermes 在内的各种模型开辟了途径。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/gurvanson/status/1866390303118164302">来自 Gurvan (@gurvanson) 的推文</a>：@iScienceLuvr Schmidhuber 已在 Recurrent Highway Networks 中实现 https://arxiv.org/abs/1607.03474，此处有更好的说明 https://arxiv.org/abs/1707.05589</li><li><a href="https://arxiv.org/abs/2412.06769">在连续潜空间中训练大语言模型进行推理</a>：大语言模型 (LLM) 被限制在“语言空间”中进行推理，它们通常使用思维链 (CoT) 来表达推理过程，以解决复杂的推理问...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1316235559256981548)** (8 条消息🔥): 

> `DNF 用于 4D 生成，Chain of Continuous Thought (COCONUT)，qtip 的 GitHub 仓库，模型容量利用率见解，AI 中的通信理论` 


- **DNF 突破了 4D 建模的界限**：DNF 模型通过利用字典学习（dictionary learning）方法，提出了一种用于生成可变形形状的新型 4D 表示，在解耦运动和形状的同时增强了高保真细节。
   - 关于该方法的更多细节和见解可以在 [项目主页](https://xzhang-t.github.io/project/DNF/) 和 [YouTube 视频](https://youtu.be/l1UhJmTbKIo) 中找到。
- **COCONUT 重新定义了 LLM 推理**：Chain of Continuous Thought (COCONUT) 的引入将允许 LLM 在连续潜空间（continuous latent space）内进行推理，通过更直接的嵌入（embedding）方法优化处理过程。
   - 讨论中出现了将其更名为“高速公路网络”（highway network）的观点，参考了 Schmidhuber 关于循环高速公路网络（Recurrent Highway Networks）的相关工作。
- **GitHub 仓库 'qtip' 发布**：为项目 [Cornell-RelaxML/qtip](https://github.com/Cornell-RelaxML/qtip) 启动了一个新的 GitHub 仓库，旨在为该模型的开发做出贡献。
   - 该仓库邀请贡献与合作，表明人们对高质量实现的兴趣日益浓厚。
- **关于模型容量利用率的见解**：初步观察表明，某些模型（如 llama3）由于模型容量利用率问题而出现性能下降，这标志着性能分析的一个趋势。
   - 社区对无需重新训练即可实现的效率提升感到好奇，这暗示了信号处理技术的复兴。
- **通信理论卷土重来**：AI 从业者对通信理论论文的兴趣日益增加，暗示了模型优化基础概念的转变。
   - 讨论表明，这些理论可能在未来的模型设计和改进中发挥关键作用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/iScienceLuvr/status/1866353795502158163/photo/1">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：训练大语言模型在连续潜空间中推理。引入了一种名为 Chain of Continuous Thought (COCONUT) 的 LLM 推理新范式。极其简单的改变：不再映射...</li><li><a href="https://x.com/gurvanson/status/1866390303118164302">来自 Gurvan (@gurvanson) 的推文</a>：@iScienceLuvr Schmidhuber 在循环高速公路网络（Recurrent Highway Networks）中已经做过了 https://arxiv.org/abs/1607.03474，这里有更好的说明 https://arxiv.org/abs/1707.05589</li><li><a href="https://github.com/Cornell-RelaxML/qtip">GitHub - Cornell-RelaxML/qtip</a>：通过在 GitHub 上创建账号来为 Cornell-RelaxML/qtip 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2412.05161">DNF: Unconditional 4D Generation with Dictionary-based Neural Fields</a>：虽然基于扩散的 3D 形状生成模型已取得显著成功，但由于物体随时间变形的复杂性，4D 生成建模仍然具有挑战性。W...</li><li><a href="https://xzhang-t.github.io/project/DNF/">DNF: Unconditional 4D Generation with Dictionary-based Neural Fields</a>：未找到描述</li><li><a href="https://youtu.be/l1UhJmTbKIo">DNF: Unconditional 4D Generation with Dictionary-based Neural Fields</a>：项目主页：https://xzhang-t.github.io/project/DNF/。虽然基于扩散的 3D 形状生成模型已取得显著成功，但 4D 生成...
</li>
</ul>

</div>

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1316448872695992412)** (16 messages🔥): 

> `Gemini 2.0, Gemini Flash, Deep Research feature, Maya: Multilingual Vision-Language Model` 


- **Google 的 Gemini 2.0 发布**：Google DeepMind 推出了 **Gemini 2.0**，并发布了一个名为 **Gemini 2.0 Flash** 的实验版本，其特点是增强的多模态输出和改进的性能，正如其 [官方公告](https://x.com/GoogleDeepMind/status/1866869343570608557) 中所强调的。
   - Gemini 2.0 旨在通过其 **tool use** 能力为新的 **Agent** 体验铺平道路。
- **Deep Research 功能发布**：**Gemini Advanced** 中的 **Deep Research** 功能旨在综合复杂信息并协助处理详细任务，正如 **Sundar Pichai** 所强调的。
   - 该功能旨在帮助用户生成包含相关来源 **链接** 的综合报告，从而增强 AI 的实用性。
- **Maya 模型预印本分享**：一位成员宣布了他们在 **Maya: Multilingual Vision-Language Model** 上的工作，并通过其 [Twitter 帖子](https://twitter.com/nahidalam/status/1866667770114609217) 分享了预印本链接。
   - **Maya** 的推出展示了多语言视觉语言模型（Vision-Language Models）在多语言能力方面的进展，引起了参与者的兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/GoogleDeepMind/status/1866869343570608557">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>: 欢迎来到这个世界，Gemini 2.0 ✨ 我们迄今为止最强大的 AI 模型。我们首先发布 2.0 Flash 的实验版本 ⚡ 它具有更好的性能、新的多模态输出、@Google 工具使用 - ...</li><li><a href="https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/">介绍 Gemini 2.0：我们面向 Agent 时代的新 AI 模型</a>: 今天，我们宣布推出 Gemini 2.0，这是我们迄今为止最强大的 AI 模型。</li><li><a href="https://x.com/JeffDean/status/1866884077988810988">来自 Jeff Dean (@🏡) (@JeffDean) 的推文</a>: 我们今天还在 Gemini Advanced 中推出了一项名为 "Deep Research" 的新功能（目前使用 Gemini 1.5 Pro 模型），它将进行大量的独立工作来综合...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1316235559256981548)** (8 条消息🔥): 

> `4D Generative Modeling, Chain of Continuous Thought (COCONUT), Model Capacity Utilization, Signal Processing in AI, QTIP GitHub Repository` 


- **DNF：4D 生成建模的飞跃**：提出了一种名为 **DNF** 的新型 4D 生成建模表示方法，利用字典学习（dictionary learning）方法有效地捕捉可变形形状的高保真细节。
   - 该方法提供了时间一致性和卓越的形状质量，并集成了 [YouTube 概览视频](https://youtu.be/l1UhJmTbKIo) 展示其应用。
- **引入用于 LLM 推理的 COCONUT**：引入了一种名为 **Chain of Continuous Thought (COCONUT)** 的新范式，用于训练 Large Language Models 在连续潜空间（continuous latent space）中进行推理，简化了现有方法。
   - 这一改进允许通过连续思维进行梯度下降，实现端到端优化，从而提高处理语言 token 的效率。
- **关于模型容量利用率的性能见解**：讨论表明模型容量利用率显著影响性能，观察到 **Llama3** 在某些条件下表现出更明显的下降。
   - 成员们注意到受信号处理（signal processing）启发的方法再次兴起，暗示回归到早期的通信理论原则。
- **QTIP GitHub 仓库发布**：Cornell-RelaxML 发布了一个名为 **QTIP** 的新仓库，在 GitHub 上提供模型开发和协作资源。
   - 用户可以通过其 [GitHub 页面](https://github.com/Cornell-RelaxML/qtip) 为该项目做出贡献，旨在促进相关领域的进步。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/iScienceLuvr/status/1866353795502158163/photo/1">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：训练 Large Language Models 在连续潜空间中推理。引入了一种名为 Chain of Continuous Thought (COCONUT) 的 LLM 推理新范式。极其简单的改变：不再映射...</li><li><a href="https://x.com/gurvanson/status/1866390303118164302">来自 Gurvan (@gurvanson) 的推文</a>：@iScienceLuvr Schmidhuber 在 Recurrent Highway Networks 中已经做过了 https://arxiv.org/abs/1607.03474，这里有更好的说明 https://arxiv.org/abs/1707.05589</li><li><a href="https://github.com/Cornell-RelaxML/qtip">GitHub - Cornell-RelaxML/qtip</a>：通过在 GitHub 上创建账号为 Cornell-RelaxML/qtip 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2412.05161">DNF: Unconditional 4D Generation with Dictionary-based Neural Fields</a>：虽然基于扩散的 3D 形状生成模型取得了显著成功，但由于物体随时间变形的复杂性，4D 生成建模仍然具有挑战性。W...</li><li><a href="https://xzhang-t.github.io/project/DNF/">DNF: Unconditional 4D Generation with Dictionary-based Neural Fields</a>：未找到描述</li><li><a href="https://youtu.be/l1UhJmTbKIo">DNF: Unconditional 4D Generation with Dictionary-based Neural Fields</a>：项目页面：https://xzhang-t.github.io/project/DNF/。虽然基于扩散的 3D 形状生成模型取得了显著成功，但 4D g...
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1316132883806814218)** (176 条消息🔥🔥): 

> `VRAM 使用与管理，图像增强的 AI 模型推荐，GPU 抢购，使用 AI 进行图像分类和打标，语音训练 AI 程序` 


- **理解 VRAM 使用**：用户讨论了以 JSON 格式缓存 prompt 如何影响显存使用，有人指出 20 个 prompt 仅占用几 MB。
   - 还讨论了以 FP8 等低精度运行模型以降低显存开销。
- **飞船 AI 模型推荐**：一位用户询问专门用于生成飞船的模型推荐，并提到他们正在使用 Dream Shaper 和 Juggernaut。
   - 另一位用户建议，如果现有模型不能满足需求，可以训练一个 LoRA。
- **GPU 抢购**：成员们分享了关于黄牛可能如何利用网络爬虫在发布日抢购 GPU 的见解。
   - 一位用户表达了对在黄牛之前买到 GPU 的担忧，并提到由于身处美国境外，他们选择不排队等待。
- **使用 AI 进行图像分类和打标**：一位用户询问了从图像中提取标签或 prompt 的工具，引发了关于图像分类技术的讨论。
   - Clip Interrogator 被推荐作为一种可以帮助在没有原始元数据的情况下描述图像的工具。
- **语音训练 AI 程序**：一位用户询问通常使用哪些程序来训练特定声音的 AI，Applio 被提为一个潜在工具。
   - 讨论突显了对语音处理工具和 AI 训练应用的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/facepalm-face-palm-picard-trek-gif-15072590366303305471">Facepalm Picard GIF - Facepalm Face Palm - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/blog/vision_language_pretraining">深入了解视觉语言模型 (A Dive into Vision-Language Models)</a>: 未找到描述</li><li><a href="https://github.com/lllyasviel/IC-Light">GitHub - lllyasviel/IC-Light: 更多重光照！</a>: 更多重光照！通过在 GitHub 上创建账号，为 lllyasviel/IC-Light 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1316156991021776937)** (13 条消息🔥): 

> `Notebook 的 Discord 集成、TTRPG 规则书实用工具、实验输出风格、Podcast 增强策略、单人冒险生成` 


- **将 Notebook 集成到 Discord**：一位成员询问如何向 Discord 添加聊天功能，以便用户可以直接向 Notebook 提问，从而增强互动性。
   - 这可以为想要更直接地与 AI 互动的玩家提供无缝连接。
- **利用 TTRPG 规则书**：一位成员分享了成功使用其自定义 TTRPG 的 **100 页规则书**作为机制转换工具的经验。
   - 他们强调了该书在将**叙事描述**转化为特定游戏机制方面的有效性。
- **输出实验**：一位成员尝试强制 NotebookLM 生成短输出，并在生成过程中利用名言警句。
   - 他们记录了一些有趣的发现，并暗示了内容响应中更深层的含义。
- **通过视觉效果提升 Podcast**：有人建议通过使用视觉效果来增强 Podcast，包括一段名为“Talking Heads Green Screen Podcast”的 **YouTube 视频**。
   - 鼓励成员通过采用 **chromakey**（色键）技术来个性化内容，打造独特的背景。
- **使用 AI 进行单人 DnD 冒险**：有人提出了关于利用 TTRPG 资源通过聊天互动来促进类似单人 DnD 冒险的问题。
   - 成员们分享了褒贬不一的体验，表明这一概念的成功程度各异，仍有进一步探索的空间。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/ufe-EWH3_gc?si=C8lneeoL82j1kT1T">Women Suck at EVERYTHING?! Debating Fresh and Fit&#39;s Myron Gaines @StandOutTV_</a>: 来自 FreshandFit Podcast 的 Myron Gaines 认为女性在所有事情上都很差劲。从驾驶到运动，他认为女性不如男性。加入我的辩论...</li><li><a href="https://www.youtube.com/watch?v=3OFeH9YFxjM">UNREAL MYSTERIES 6: The Christmas Special - a Post-Apocalyptic Musical</a>: 每一部好剧都有圣诞特辑，而每一部好的圣诞特辑都是音乐剧……David 和 Hannah 对抗僵尸驯鹿、澳大利亚外星人以及 l...</li><li><a href="https://youtu.be/MjbH5neAAAo?feature=shared">Talking Heads Green Screen Podcast</a>: 🎙️ 淹没在 AI 生成的 Podcast 海洋中？让你的脱颖而出！🌊需要视觉效果来提升你的 Podcast 吗？🎥 下载此视频并随心使用...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1316138669886930945)** (93 条消息🔥🔥): 

> `使用 NotebookLM 制作播客, NotebookLM 功能与限制, Gemini 2.0 集成, NotebookLM 的输入方法, AI 工具的用户体验` 


- **探索 NotebookLM 的播客功能**：许多用户对使用 NotebookLM 创建播客表示好奇，其中一位成员询问了是否可以自定义播客主持人。
   - 有人提到他们已经有一段时间没用 NotebookLM 了，这凸显了在了解其播客功能方面的脱节。
- **了解 NotebookLM 对笔记的限制**：用户讨论了 NotebookLM 的限制，例如每个笔记本最多可包含 1,000 条笔记（每条最多 100,000 个字符）以及 50 个来源。
   - 关于作为来源上传的 PDF 文件大小限制及其对整体功能的影响也引发了疑问。
- **Gemini 2.0 即将接入 NotebookLM**：关于 NotebookLM 集成 Gemini 2.0 的猜测非常普遍，一些人确认这种过渡确实正在发生。
   - 讨论中还包括一些幽默的评论，关于错失使用 Gemini 符号进行品牌推广的机会。
- **AI 与工具的用户体验**：用户分享了他们使用各种 AI 工具和功能的经验，提到 Snapchat AI 和 Instagram 的聊天 AI 体验愉快。
   - 一位用户提到 NotebookLM 在困难时期非常有帮助，强调了它在工作流（workflows）中的作用。
- **视频裁剪与使用**：一位成员分享了使用在线工具裁剪 YouTube 视频的经验，强调了该工具在有效编辑内容方面的易用性。
   - 他们提到了该工具的功能，即无需注册即可创建指向已编辑片段的直接链接。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://openinterx.com">OpenInterX 的推文 - 具有上下文记忆的多模态大规模视频分析</a>: OpenInterX 是一个领先的多模态大规模视频分析平台，具有上下文记忆功能，提供先进的基于 AI 的工具以进行高效的视频分析。</li><li><a href="https://medicalxpress.com/news/2024-12-brain-mechanisms-underpinning-loss-consciousness.html">确定了支撑意识丧失的大脑机制</a>: 从清醒状态到无意识状态的转变是一个长期以来吸引科学家和哲学家兴趣的现象，但它是如何发生的直到现在仍然是一个谜。通过...</li><li><a href="https://www.instagram.com/p/DDMKRTZRFlS/?utm_source=ig_web_copy_link">Dr. Ganapathi Pulipaka 在 Instagram 上: &quot;最适合高级学习者的 #Statistics 书籍。#BigData #Analytics #DataScience #IoT #IIoT #PyTorch #Python #RStats #TensorFlow #Java #JavaScript #ReactJS #GoLang #CloudComputing #Serverless #DataScientist #Linux #Books #Programming #Coding #100DaysofCode 
https://geni.us/Statistics-Advanced&quot;</a>: 62 个赞，2 条评论 - gp_pulipaka 于 2024 年 12 月 5 日发布: &quot;最适合高级学习者的 #Statistics 书籍。#BigData #Analytics #DataScience #IoT #IIoT #PyTorch #Python #RStats #TensorFlow #Java #JavaScript...</li><li><a href="https://www.instagram.com/reel/DDaTgGKzG7o/?utm_source=ig_web_button_share_sheet&igsh=MzRlODBiNWFlZA==">Instagram 上的 Volley</a>: 0 个赞，0 条评论 - somebiohacker 于 2024 年 12 月 10 日发布</li><li><a href="https://www.youtube.com/watch?v=FJiXHRdW6ws">nice bonk football (加速到 1.5 或 2.0)</a>: 未找到描述</li><li><a href="https://www.youtubetrimmer.com/">裁剪 YouTube 视频 - YouTubeTrimmer.com</a>: YouTubeTrimmer.com: 在线裁剪 YouTube 视频。一个免费的在线工具。</li><li><a href="https://youtu.be/JHfSDJOXEPo?si=eLLG5V42o33mgdBD">Hellfire</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=aG0ixD3OY80">你必须知道的 10 个 NotebookLM 播客提示词</a>: NotebookLM 播客正在改变游戏规则——为什么要满足于通用的双主持人聊天？在这段视频中，我将揭示 10 个秘密提示词，它们将提升你的 NotebookLM...</li><li><a href="https://www.progress.org.uk/crispr-genome-editing-can-be-controlled-with-ultrasound/">CRISPR 基因组编辑可以通过超声波控制 | PET</a>: 聚焦超声波束被定向到含有 CRISPR/Cas9 的肿瘤，引起局部加热并激活 CRISPR 系统。</li><li><a href="https://pubmed.ncbi.nlm.nih.gov/15217335/">时间处理的神经基础 - PubMed</a>: 对感觉和运动处理的完整理解需要描述神经系统如何在几十到几百毫秒 (ms) 的范围内处理时间。时间处理...</li><li><a href="https://www.youtube.com/shorts/MRQJr7Qaqvs">2024 年 12 月 11 日</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1316139252857176155)** (13 条消息🔥): 

> `Microwave Gang, Discord 个人资料命名, 公开 Hangouts 安排, Whova App 偏好` 


- **欢迎加入 Microwave Gang**：一名成员建议，如果成员们有足够的兴趣，可以创建一个专门针对 **Microwave Gang** 的频道。
   - 另一名成员幽默地提到，他们太懒了不想起名字，导致首字母缩写刷屏。
- **带有家庭气息的 Discord 命名**：一位成员透露，他们的 Discord 主个人资料名称和头像是由女儿选择的，为他们的在线身份增添了个人色彩。
   - 这个有趣的细节在聊天中引发了笑声和轻松的气氛，增强了社区的凝聚力。
- **宣布周末举行公开 Hangouts**：一名成员宣布了定于周四和周五 **1:30-2:30pm** 举行的两次公开 Hangouts，并计划最终确定地点。
   - 他们还暗示，可能会为已经知情的 *付费参与者* 提前开始。
- **对 Whova App 表示不适**：一名成员幽默地表示拒绝使用 **Whova** App，表明更倾向于其他选择。
   - 这一评论引发了围绕 App 选择的简短讨论，突显了小组内多样化的偏好。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/chicken-microwave-spin-gif-22764274">Chicken Microwave GIF - Chicken Microwave Spin - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/microwavegang/">Reddit - 深入探索一切</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1316404689247408180)** (49 条消息🔥): 

> `Gemini 2.0 Flash, OpenAI 产品重点, 视频生成模型, Sora 注册, 编程能力` 


- **Gemini 2.0 Flash 惊艳亮相**：Google 宣布推出 **Gemini 2.0 Flash**，其在功能和编程性能方面表现出色，特别是在基准测试中超越了之前的版本。
   - 用户对其多模态（Multimodal）特性感到兴奋，包括在工具使用和编程任务方面的改进，但也有人对其稳定性和相对于其他模型的有效性表示担忧。
- **OpenAI 坚持以产品为中心的方法**：社区观察到，**OpenAI** 对交付强大产品的关注使其在领域内保持领先，而其竞争对手则难以跟上步伐。
   - 这种观点呼应了这样一种想法：在 AI 技术的持续发展中，产品的可用性将始终占据首位。
- **视频生成模型评价褒贬不一**：来自**中国社区**关于各种视频生成模型（包括 **Kling 1.5** 和**混元 Huyuan**）的反馈表明，相比之下，**Gemini** 的视频功能表现平平。
   - 尽管最初有一些炒作，但用户指出 Gemini 存在 Bug，且未能达到其前代产品和竞争对手设定的预期。
- **Sora 重新开放注册**：**Sora** 注册已恢复，随着一些成员报告获得访问权限并寻求进一步探索其功能，用户情绪高涨。
   - 这种兴趣的复苏出现在关于 AI 模型功能和可用性的持续讨论中。
- **Gemini 的编程能力令用户印象深刻**：许多用户评论说 **Gemini 2.0 Flash** 在编程任务中表现出色，甚至可以与 **Claude** 等成熟平台相媲美。
   - 普遍共识是，其在编程方面的优势为开发人员未来的发展提供了令人兴奋的机会。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/hanqing_me/status/1866688869711954339">汗青 HQ (@hanqing_me) 的推文</a>: 玩了一天下来，不得不说Sora的稳定性真的是很好，特别稳</li><li><a href="https://x.com/testingcatalog/status/1866844765293355496">TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: 突破性新闻 🚨: Gemini Flash 2.0 Experimental 已经开始向部分用户推送！看来 Google 今天有大动作 👀 它出现在下拉菜单中，并带有 "Experimental" 标签...</li><li><a href="https://x.com/officiallogank/status/1866868435722047927?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Logan Kilpatrick (@OfficialLoganK) 的推文</a>: 宣布推出 Gemini 2.0 Flash，这是一款新的多模态 Agentic 模型（最初为实验版），在功能和基准测试方面取得了我们有史以来最好的结果：https://developers.googleblog.com/en/the-next-chapter-o...</li><li><a href="https://x.com/bio_bootloader/status/1866916624168784033">BioBootloader (@bio_bootloader) 的推文</a>: 这非常令人印象深刻，但 Anthropic 的帖子显示他们使用了相当基础的脚手架——Agent 选择工具直到决定提交，而从这里看来 2.0 Flash 的得分是...</li><li><a href="https://x.com/lmarena_ai/status/1866873983569891378">lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>: 来自 Chatbot Arena 的突发新闻⚡@GoogleDeepMind Gemini-2.0-Flash 首次亮相即位列总榜第 3 —— 相比 Flash-002 是一个巨大的飞跃！亮点（相比 Flash-002 的提升）：- 总榜：#11 → #3 - 硬核提示词（Hard Prompts）：#15 → ...</li><li><a href="https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/">介绍 Gemini 2.0：我们为 Agent 时代打造的新 AI 模型</a>: 今天，我们宣布推出 Gemini 2.0，这是我们迄今为止最强大的 AI 模型。</li><li><a href="https://x.com/OfficialLoganK/status/1866916665902108785">Logan Kilpatrick (@OfficialLoganK) 的推文</a>: Gemini 2.0 Flash 在编程方面非常出色 : )</li><li><a href="https://deepmind.google/technologies/gemini/">Gemini</a>: Gemini 2.0 我们迄今为止最强大的 AI 模型，专为 Agent 时代打造。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1316240558963822603)** (6 条消息): 

> `Scaling Laws 辩论, Latent Space Podcast 现场活动, 影响力人物与 Scaling, 社区参与, AI 讨论中的温馨氛围` 


- **NeurIPS 上的 Scaling Laws 辩论**：明天下午 4:00 NeurIPS 期间，加入 @dylan522p 参加一场关于 Scaling Laws 的现场牛津式辩论（Oxford Style Debate），他将迎接所有挑战者的挑战。
   - 鼓励*兴奋的*参与者参加，因为这场辩论将探讨 **Scaling 已经撞墙** 的观点，旨在回应来自 Twitter 影响力人物的批评。
- **Latent Space Podcast 现场活动公布**：**Latent Space Podcast** 正在举办一场与 @dylan522p 合作的现场活动，他正在挑战任何有勇气在 Scaling Laws 问题上与他辩论的人。
   - 赞助使活动可以**免费**参加，活动还为参与者准备了**奖杯**，吸引大胆的竞争者。
- **社区对辩论的参与**：一位成员表达了兴奋之情，并觉得辩论的概念*非常滑稽*，展示了社区对该活动的积极参与。
   - 社区的热情反映了他们对 AI 领域紧迫话题的兴趣，尤其是以这种娱乐化的形式呈现。
- **对个人内容博弈的支持**：另一位成员幽默地建议让一名支持者就其内容“对战 Gary”，为对话增添了轻松的基调。
   - 社区的反应凸显了一个相互支持的环境，成员们可以在其中分享他们对内容的经验和情感。
- **温馨的 AI 社区氛围**：一位成员关于讨论性质温馨的评论培养了 AI 社区内的同僚情谊。
   - 对话的基调暗示了一种**令人振奋**的氛围，在严肃辩论的同时鼓励积极性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/dylan522p/status/1866630813074461060">来自 Dylan Patel ✈️ NeurIPS (@dylan522p) 的推文</a>：Scaling 结束了吗？明天（周三）下午 4:00 在 NeurIPS 现场，我将与来自 Lottery Ticket 和 MosaicML / Databricks 的杰出人物 Jonathan Frankle 进行一场精彩的辩论。现在注册！https:/...</li><li><a href="https://bsky.app/profile/mdagost.bsky.social/post/3ld2gnb7vns25">Michelangelo D’Agostino (@mdagost.bsky.social)</a>：我没有内部消息，只有 @natolambert.bsky.social 在那篇帖子中写的内容：
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1316278029063815178)** (11 条消息🔥): 

> `加入 Llama 团队，Gemini 秘密，关于推理的演讲，关于推理的重要论文，对 Nous Dunks 的称赞` 


- **激动人心的消息：加入 Llama 团队！**：一名成员宣布他们将于下个月加入 [AI at Meta](https://x.com/_arohan_/status/1866621771451076812) 的 Llama 团队，致力于研发下一代 Llama 模型，并对即将开展的项目表示兴奋。
   - 他们还表达了希望为建立一个惠及所有人的更健康生态系统做出贡献的愿望，并引用了扎克伯格关于 AGI 目标和开源努力的言论。
- **期待 Gemini 的秘密**：成员们表示，在经历了对 **Llama 3.2-vision** 的失望后，希望 **Gemini** 的技术秘密能整合进多模态 **Llama** 团队中。
   - 席间还有一段幽默的交流，讨论在演示过程中探讨 **Kambhampathi 的作品** 所带来的见解，尽管有人指出这可能会分散对主旨的注意力。
- **准备关于推理的演讲**：一位成员分享说，他们将在 4 小时后进行一场关于推理（Reasoning）的演讲，并寻求可以在幻灯片中加入的重要论文或著名评论。
   - 另一位成员因无法访问自己的资源而感到困扰，感叹自己身处偏远地区，并暗示会讨论一些 **STaR** 的相关工作。
- **地理位置笑话中的闲聊**：一些讨论幽默地强调了成员们身处偏远地区的共同经历，其中一人将巴黎称为“荒郊野岭”。
   - 随意的言谈与对所在地舒适度的表达交织在一起，维持了轻松的氛围。
- **对 Nous Dunks 的称赞**：一位成员的 **Nous Dunks** 获得了称赞，并评论道：“你根本不知道关起门来都在发生什么”。
   - 这一评论引发了幽默的反应，展示了频道内讨论的随意性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/_arohan_/status/1866621771451076812">来自 rohan anil (@_arohan_) 的推文</a>：秘密公开了：我将于下个月加入 @AIatMeta 的 Llama 团队，致力于下一代 Llama 模型。是的，在下一个模型发布前，我已经准备好了一些关于 Llama 的双关语...</li><li><a href="https://docs.google.com/presentation/d/1PNipMudHb5HTNnVosve0lqdrgwKSWck4SCE9TSPCJkY/edit?usp=sharing">[12112024, Latent Space @ NeurIPS] 推理</a>：推理的状态，Nathan Lambert Ai2 // Interconnects.ai Latent Space // NeurIPS 2024 Lambert | 关于推理的思考 1
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1316215678650286155)** (8 条消息🔥): 

> `AI 的可扩展性，对 GM 的批评，对网络言论的回应` 


- **Scaling Laws 受到审视**：针对批评，@fchollet 澄清说，他们并没有赌 **Scaling Laws** 会失效，而是反对“仅仅增加模型大小就足以取得进步”的观点。
   - *他们指出，进步来自于搜索、适配和程序合成，而不仅仅是增加模型规模*。
- **GM 的声誉受损**：@kvogt 直言不讳地表示“**GM 是一群笨蛋**”，这表明了对该公司及其领导层的严厉批评。
   - 在随后的评论中有人指出，虽然这对像 Kyle 这样的人来说可能很难接受，但 **GM** 以高达 **10 亿美元** 的价格收购了他的公司，这引发了关于该收购价值的讨论。
- **Twitter 上日益增长的挫败感**：一位用户评论了 Twitter 上言论不断升级的本质，@natolambert 承认他们增加了直接回应的频率，从最初只说“不”的立场开始转变。
   - *他们承认当前对话的激烈程度，并认为这是他们互动方式的一个重大变化*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/kvogt/status/1866612270815494639?s=46">来自 Kyle Vogt (@kvogt) 的推文</a>：如果以前还不清楚，现在很清楚了：GM 是一群笨蛋。</li><li><a href="https://x.com/fchollet/status/1866348355204595826?s=46">来自 François Chollet (@fchollet) 的推文</a>：伙计，你在说什么？1. 我根本不知道你是谁，所以我对你没有任何“看法”。2. 我从未赌过 Scaling Laws 会失效。相反，我一直反对那种认为……的观点。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1316450275019456552)** (1 messages): 

> `CV 频道参与度, MLLMs, VLMs` 


- **提升 CV 频道的参与度**：一位成员表示希望能增加 CV 频道的参与度，寻求与更多对计算机视觉感兴趣的人建立联系。
   - 他们自荐为专注于 **MLLMs** 和 **VLMs** 的 **CV / perception** 人员。
- **关注多模态学习**：讨论强调了频道内对 **MLLMs** (Multimodal Large Language Models) 和 **VLMs** (Vision-Language Models) 的特定兴趣。
   - 这反映了在 AI 中整合各种模态以增强感知系统能力的日益增长的趋势。


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1316256265441382461)** (11 messages🔥): 

> `AI Scaling Laws, LLM 创意基准测试, Inference Time Compute, RL LLMs, Scaling LLM Test-Time Compute` 


- **围绕 AI Scaling Laws 的恐惧与疑虑**：关于 **AI Scaling Laws** 的恐惧、不确定性和怀疑 (FUD) 呈上升趋势，各种预测声称这些模型的快速进步已到头，正如一篇[近期文章](https://semianalysis.com/2024/12/11/scaling-laws-o1-pro-architecture-reasoning-infrastructure-orion-and-claude-3-5-opus-failures/)所强调的那样。记者们正利用关于模型可扩展性失败的*嘈杂泄密*和模糊信息来支持这些叙述。
- **探索 LLM 在创意任务中的能力**：一项关于衡量 **LLM 能力** 在软性“创意”任务（如头脑风暴）中表现的[讨论](https://gwern.net/creative-benchmark)浮出水面，并揭示了在创意写作领域的缺陷。讨论指出，用户的意见往往与基准测试分数相左，特别注意到 **Claude-3** 尽管在创意基准测试中并不总是排名最高，但却非常受欢迎。
- **RL LLMs 和 Test-Time Compute 的挑战**：成员们强调了他们对两段最近关于 **RL LLMs** 和 **Scaling Test-Time Compute** 的 YouTube 视频的兴趣，其中一段由来自加州大学伯克利分校的 Charlie Snell 演讲，重点是增强 LLM 的输出。另一段教程旨在解释 **'Inference-Time Compute'** 背后的关键技术，这对于优化 OpenAI 的 O1 性能至关重要。
- **PDF 资源共享变得混乱**：一位用户分享了一份与 **Scaling Laws** 主题相关的 PDF，但随后指出该文档缺少高级内容，并对请求被误解表示沮丧。这引发了一场关于资源可用性和完整性的轻松交流。
- **视频学习资源的不确定性**：一位成员评论说，他们很难抽出时间观看关于 LLM 推理及相关主题的深刻 **YouTube 视频**，并指出内容质量可能参差不齐。其他人分享了这些资源的链接，对其深度和可用信息的积极程度各不相同。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gwern.net/creative-benchmark">迈向 LLM 多样性与创意基准测试 · Gwern.net</a>: 未找到描述</li><li><a href="https://semianalysis.com/2024/12/11/scaling-laws-o1-pro-architecture-reasoning-infrastructure-orion-and-claude-3-5-opus-failures/">Scaling Laws &#8211; O1 Pro 架构、推理训练基础设施、Orion 和 Claude 3.5 Opus 的“失败”</a>: 关于 AI Scaling laws 的恐惧、不确定性和怀疑 (FUD) 日益增多。一众兼职 AI 行业预言家们正紧盯着任何看跌的叙述……</li><li><a href="https://archive.is/xoKR3">Scaling Laws &#x2013; O1 Pro 架构、推理训练基础设施……</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=S5l5OvJ01ws&list=WL&index=25&t=21s)">推理漫游指南</a>: 关于 LLM 推理的演讲，涵盖了各种方法、核心问题和未来的研究方向！链接：- Slides: https://docs.google.com/presentation/d/e/...</li><li><a href="https://youtu.be/OXwGp9YeuBg?si=TOZgkr2hG7BfvJmT">Charlie Snell, 加州大学伯克利分校。标题：Scaling LLM Test-Time Compute</a>: 摘要：使 LLM 能够通过使用更多的 test-time computation 来改进其输出，是构建通用自我改进 Agent 的关键一步……</li><li><a href="https://youtu.be/T1SeqBapMBo?si=evbYnI0AUBlZ74HH">LTI Yi Wu 特别研讨会</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=_Bw5o55SRL8">Inference Time Compute</a>: 本教程旨在解释“Inference-Time Compute”背后的关键技术，据称这是 OpenAI O1 的核心。我将讨论我们如何……
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1316440929913016354)** (2 条消息): 

> `` 


- **Discord 中表达的沮丧情绪**：用户对极少的回复表示焦虑，正如在交流中一位用户惊呼 *'bruh'* 并紧接着说 *'cmon'*，表明了对讨论现状的不满。
- **注意到参与度极低**：用户之间简短的互动表明该频道缺乏深入的对话或参与，说明了对正在进行的对话可能存在的挫败感。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1316149531292991559)** (81 条消息🔥🔥): 

> `Merging Documents with AI, Running Models on Multiple GPUs, Using LM Studio with Web Access, Handling Model Parameters, Updating LM Studio` 


- **使用生成式 AI 合并文档的技巧**：一名成员询问了关于使用生成式 AI 合并两个文档的建议，但收到的建议是使用传统方法，如 MS Word 的合并选项。
   - 另一位成员建议编写脚本执行精确合并，而不是依赖模糊的 Prompt。
- **并行 GPU 使用讨论**：一位用户询问关于使用多个 GPU 运行模型的问题，特别是两块 3070 和一块 3060ti。
   - 有人指出 LM Studio 的 GPU offload 目前只是一个简单的开关，用户需要通过环境变量来探索变通解决方案。
- **LM Studio 中模型的 Web 访问**：一名成员询问他们的模型是否可以访问 Web，回复指出这需要通过 API 进行自定义解决，而不是通过聊天界面。
   - 这反映了用户对将模型与外部工具和网站集成以增强功能的广泛兴趣。
- **模型训练挑战**：讨论者研究了训练大语言模型 (LLMs) 的挑战，强调虽然创建一个模型并不太难，但创建一个高质量的模型却很难。
   - 针对在数据和参数不足的情况下训练的模型性能提出了担忧，并分享了关于架构重要性的见解。
- **更新和自定义 LM Studio**：一位用户注意到他们使用的是过时版本的 LM Studio，并得知最新的更新必须手动完成。
   - 讨论还涉及了自定义 GUI，并在 GitHub 上找到了可能进行修改的资源。



**提及的链接**：<a href="https://huggingface.co/SicariusSicariiStuff/LLAMA-3_8B_Unaligned_BETA_GGUFs">SicariusSicariiStuff/LLAMA-3_8B_Unaligned_BETA_GGUFs · Hugging Face</a>：未找到描述

  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1316163085740015678)** (5 条消息): 

> `Alphacool D5 Pump Setup, LMStudio GPU Usage` 


- **Alphacool 的 D5 泵集成**：Alphacool 发布了一款已经安装了 **D5 pump** 的型号，一些成员觉得这令人印象深刻。
   - *一位成员表示遗憾*，因为他们的大机箱空间问题没有选择这种设置，机箱内已经装满了 **4 GPUs** 和 **8 HDDs**。
- **确认 LMStudio 支持多 GPU**：有确切消息确认 **LMStudio** 确实可以同时利用多个 GPU，从而提升性能。
   - *另一位成员询问*了集成在客户端中的浏览器发现功能，表明了对其更广泛能力的兴趣。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1316149341299408968)** (52 条消息🔥): 

> `Nous Simulators Announcement, Hyperbolic Series A Funding, Gemini 2.0 Flash Launch, Stainless Series A Update, Realtime Multimodal API Introduction`

- **Nous Simulators 发布**：Nous Research 宣布推出 [Nous Simulators](https://sims.nousresearch.com)，用于在社交语境中实验人类与 AI 的交互。
   - 该平台旨在提供对 AI 行为和交互的见解。
- **Hyperbolic 完成 1200 万美元 A 轮融资**：Hyperbolic 成功筹集了 **1200 万美元 A 轮融资**，目标是开发一个具有开放 GPU 市场的开放 AI 平台。
   - 他们致力于透明度和社区协作，为 H100 SXM GPU 提供低至 **$0.99/小时** 的价格。
- **Gemini 2.0 Flash 发布**：Google 的 [Gemini 2.0 Flash](https://x.com/lmarena_ai/status/1866873983569891378) 在 Chatbot Arena 总榜位列第 3，表现显著优于 Flash-002 等前代模型。
   - 关键改进包括在复杂提示词（hard prompts）和编程任务中表现更好，具备支持多模态交互的实时能力。
- **Stainless A 轮融资成功**：Stainless API 宣布了由 **a16z** 和 **Sequoia** 等知名投资者领投的 **2500 万美元 A 轮融资**。
   - 这笔资金旨在增强其在流行 AI SDK 背后的产品能力，助力构建稳健的开发生态系统。
- **实时多模态 API 亮相**：Logan K 展示了由 Gemini 2.0 Flash 驱动的新型 [Realtime Multimodal API](https://x.com/officiallogank/status/1866873298027446465?s=46)，支持实时音频、视频和文本流。
   - 该工具承诺在后台进行动态工具调用，以实现无缝的交互体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/_arohan_/status/1866621771451076812">来自 rohan anil (@_arohan_) 的推文</a>：秘密公开了：我下个月将加入 @AIatMeta 的 Llama 团队，致力于下一代 Llama 模型。是的，我已经准备好了一些关于 Llama 的双关语，在下一代模型发布前...</li><li><a href="https://x.com/lmarena_ai/status/1866873983569891378">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：来自 Chatbot Arena 的突发新闻⚡ @GoogleDeepMind Gemini-2.0-Flash 首次亮相即位列总榜第 3 —— 相比 Flash-002 是一个巨大的飞跃！亮点（相比 Flash-002 的提升）：- 总榜：#11 → #3 - 困难提示词：#15 → ...</li><li><a href="https://x.com/dylan522p/status/1866693150632473048?s=46">来自 Dylan Patel ✈️ NeurIPS (@dylan522p) 的推文</a>：Scaling Laws、O1 Pro 架构、推理基础设施、Orion 和 Claude 3.5 Opus 的“失败”、AI 实验室合成数据基础设施、Test Time Compute 的推理代币经济学、数据墙、评估的...</li><li><a href="https://x.com/stainless">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/andykonwinski/status/1867015050403385674">来自 Andy Konwinski (@andykonwinski) 的推文</a>：我将向第一个在这一全新的、无污染版本的 SWE-bench 上达到 90% 分数的开源 AI 提供 100 万美元奖金 - http://kprize.ai</li><li><a href="https://x.com/cerebrassystems/status/1866530273502044240?s=46">来自 Cerebras (@CerebrasSystems) 的推文</a>：推出 CePO —— 一个用于 Llama 的 Test Time 推理框架。Llama3.3-70B + CePO 的表现优于 Llama 3.1 405B，并接近 GPT-4 和 Sonnet 3.5。CePO 实现了实时推理。尽管使用了超过 10 倍的...</li><li><a href="https://x.com/yuchenj_uw/status/1866514943815880847?s=46">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：🚀 很高兴分享我们筹集了 1200 万美元的 A 轮融资！在 Hyperbolic，我们的使命是构建一个开放的 AI 平台。所谓“开放”，我们指的是：> 开放 GPU 市场：可以将其视为 GPU 版的 Airbnb —— 任何人都可以...</li><li><a href="https://x.com/officiallogank/status/1866873298027446465?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：推出全新的 Realtime Multimodal API，由 Gemini 2.0 Flash 提供支持！你可以输入音频、视频和文本流，同时后台会进行动态工具调用（搜索、代码执行和函数调用...）</li><li><a href="https://x.com/legit_rumors/status/1866708804584296459?s=46">来自 ʟᴇɢɪᴛ (@legit_rumors) 的推文</a>：这个刚刚在幕后被添加了 👀</li><li><a href="https://x.com/simonw/status/1866942603020910866?s=46">来自 Simon Willison (@simonw) 的推文</a>：如果你今天不想尝试别的，请务必试试 https://aistudio.google.com/live 上的演示 —— 它允许你直接向 Gemini 2.0 Flash 传输视频和音频流并获取音频反馈，这样你就可以进行实时的...</li><li><a href="https://x.com/scaling01/status/1866895964826378376?s=46">来自 Lisan al Gaib (@scaling01) 的推文</a>：天哪！那是真的聊天吗？Google 的 Gemini 2.0 Flash 简直绝了。Gemini 2.0 Flash 在 SWE-bench verified 中击败了所有 o1 模型和 Sonnet 3.5。引用 wh (@nrehiew_) 更新的图表，Gemini...</li><li><a href="https://x.com/kepano/status/1866891181138797049?s=46">来自 kepano (@kepano) 的推文</a>：我对今天发布的 Gemini 2.0 Flash 的简评 —— 它非常快，而且免费（目前？）。在我的 Web Clipper 测试中，感觉与 Claude Haiku 3.5 旗鼓相当。这里有一个例子：</li><li><a href="https://x.com/NousResearch/status/1866584568548995538">来自 Nous Research (@NousResearch) 的推文</a>：宣布 Nous Simulators！这是我们所有涉及社交领域人机交互实验的大本营。http://sims.nousresearch.com</li><li><a href="https://x.com/m__dehghani/status/1866921587322261998?s=46">来自 Mostafa Dehghani (@m__dehghani) 的推文</a>：Gemini 2 Flash 挑战了互联网一直以来的诉求：将“画出剩下的猫头鹰部分”分解为实际步骤，并进行交错生成。虽然还不完美，但已经在路上了...</li><li><a href="https://x.com/jeffdean/status/1866884077988810988?s=46">来自 Jeff Dean (@🏡) (@JeffDean) 的推文</a>：我们今天还在 Gemini Advanced 中推出了一项名为“Deep Research”的新功能（目前使用 Gemini 1.5 Pro 模型），它将进行大量的独立工作来综合...</li><li><a href="https://x.com/altryne/status/1866863870553493790?s=46">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：哇，看来我们要迎来 Gemini 2 Flash 了！@googleaistudio 中支持 128k 多模态 👏</li><li><a href="https://x.com/davidsholz/status/1866932443456082432?s=46">来自 David (@DavidSHolz) 的推文</a>：12 月 11 日 Restream 直播 https://x.com/i/broadcasts/1OwxWNdLkDZJQ</li><li><a href="https://x.com/m__dehghani/status/1866937033052262651?s=46">来自 Mostafa Dehghani (@m__dehghani) 的推文</a>：交互式和交错式图像生成是 Gemini 2 Flash 的闪光点之一！这里有一些很酷的例子线程：</li><li><a href="https://x.com/techcrunch/status/">来自 TechCrunch 的推文</a>：...</li>

1866476101825896455?s=46">来自 TechCrunch (@TechCrunch) 的推文</a>：OpenAI 支持的 Speak 以 10 亿美元估值融资 7800 万美元，旨在通过大声朗读帮助用户学习语言 https://tcrn.ch/3D9gZw4</li><li><a href="https://x.com/stainlessapi/status/1866503595690180657?s=46">来自 Stainless (@StainlessAPI) 的推文</a>：很高兴分享我们已完成 2500 万美元 A 轮融资，由 @JenniferHLi @a16z 领投，@sequoia, @thegp, @felicis, @zapier 和 @mongoDB Ventures 参投：https://www.stainlessapi.com/blog/stainless-series-a</li><li><a href="https://x.com/hardmaru/status/1866287722543116371?s=46">来自 hardmaru (@hardmaru) 的推文</a>：很高兴宣布 Sakana AI 的新论文：“An Evolved Universal Transformer Memory” 🧠https://arxiv.org/abs/2410.13166 这项工作引入了经过进化以优化...的 Neural Attention Memory Models</li><li><a href="https://www.youtube.com/watch?v=wT636THdZZo">Latent Space LIVE! - 2024 年度回顾：Startups, Vision, Open Src, Reasoning, &amp; The Great Scaling Debate</a>：https://lu.ma/LSLIVE</li><li><a href="https://ai.google.dev/pricing">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1316314622344695848)** (2 条消息): 

> `Latent Space Live 2024, NeurIPS Conference, AI Agents Debate, Bolt Success, YouTube Streaming Event` 


- **Latent Space Live 2024 阵容公布**：已确认多位重量级演讲嘉宾参加在 [NeurIPS Conference](https://x.com/swyx/status/1864423257266639166) 举办的 **Latent Space Live!**，主题涵盖 **2024 AI Startups/Macro** 和 **Open Models**。
   - 组织者正寻求敲定最后几个演讲席位，并邀请关于 **Multi Agents 和 End of Scaling** 的讨论。
- **AI Engineers 助力学术会议**：该活动强调填补主要学术会议的知识空白，**新加坡将成为 [ICLR 2025](https://lu.ma/LSLIVE) 的下一站**。
   - 这一举措旨在为今后参加大型会议的 AI Engineers 创造一个支持性的空间。
- **Bolt 在 AI 工程领域的快速成功**：Bolt 报告称作为 Claude Wrapper，在短短 2 个月内 **ARR 超过 800 万美元**，显示出市场对 code agent 工程的浓厚兴趣。
   - 关键讨论包括 **复杂任务分解策略** 和维护用户界面，突显了 Bolt 在市场中的独特地位。
- **直播 Latent Space LIVE!**：在 [YouTube](https://www.youtube.com/watch?v=wT636THdZZo) 上观看讨论 **Best of 2024** 的直播活动，主题包括 **Startups, Vision, 和 The Great Scaling Debate**。
   - 观众可以通过此 [链接](https://lu.ma/LSLIVE) 获取活动详情以了解更多信息。
- **频道更新与 Zoom 使用**：**#llm-paper-club** 频道已暂时更名，以便更好地组织围绕该活动的讨论。
   - 鼓励 Discord 成员通过提供的 [Zoom 链接](https://us06web.zoom.us/j/86263968708?pwd=PapZ6BkWafamK0rnntuIkqMPiNAXf8.1) 加入，参与实时讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/swyx/status/1864423257266639166):">来自 swyx @LatentSpacepod LIVE! (@swyx) 的推文</a>：虽然是次要新闻，但非常自豪地宣布在 @NeuripsConf 第一天举行的 LS LIVE! 首批演讲嘉宾：- 2024 AI Startups/Macro @saranormous - 2024 Vision: @roboflow + @vikhyatk - 2024 Open Models: ...</li><li><a href="https://us06web.zoom.us/j/86263968708?pwd=PapZ6BkWafamK0rnntuIkqMPiNAXf8.1">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，可跨移动设备、桌面和会议室系统进行视频和音频会议、聊天及网络研讨会。Zoom ...</li><li><a href="https://www.youtube.com/watch?v=wT636THdZZo">Latent Space LIVE! - 2024 年度回顾：Startups, Vision, Open Src, Reasoning, &amp; The Great Scaling Debate</a>：https://lu.ma/LSLIVE
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1316438144081727509)** (7 messages): 

> `Zoom 会议安排、关于 Chinchilla 的 Thriaal、YouTube 直播` 


- **分享了 Zoom 会议链接**：一名成员分享了即将于上午 9 点开始的会议 [Zoom 链接](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09)。
   - 由于一名成员需要快速加入通话，现场一度比较紧急，促使他们寻求更多信息。
- **关于 Chinchilla 上的 Thriaal 的询问**：一名成员询问了 **Thriaal on Chinchilla** 的状态，寻求关于该主题进展的澄清。
   - 这一询问凸显了对该项目的持续关注，并引发了小组内的讨论。
- **YouTube 直播公告**：宣布了一个名为 [YouTube stream](https://www.youtube.com/watch?v=wT636THdZZo) “Latent Space LIVE! - Best of 2024: Startups, Vision, Open Src, Reasoning, & The Great Scaling Debate”的直播。
   - 直播的 [描述](https://lu.ma/LSLIVE) 中提供了更多细节，暗示后续将有一场重大的事件讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可跨移动设备、桌面和会议室系统进行视频和音频会议、聊天和网络研讨会。Zoom ...</li><li><a href="https://www.youtube.com/watch?v=wT636THdZZo">Latent Space LIVE! - Best of 2024: Startups, Vision, Open Src, Reasoning, &amp; The Great Scaling Debate</a>: https://lu.ma/LSLIVE
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1316362551105425428)** (7 messages): 

> `C 和 C++ 标准化、Modular 网站论坛访问` 


- **C 和 C++ 纯函数（pure functions）成功实现标准化**：在 C 和 C++ 中将函数标记为 **pure** 的标准化努力已取得成功，C 语言部分已被包含在 **C23** 中。
   - 参考资料包括 [n2956](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2956.htm) 和 [p0078](https://wg21.link/p0078) 以获取技术见解。
- **Modular 网站的论坛链接难以找到**：一名成员对 Modular 网站没有清晰链接到论坛表示沮丧，称其似乎被“埋”在 **Company** 栏目下。
   - 另一名成员指出，团队目前正专注于改进论坛，并计划在 1 月份进行 **正式发布**，之后再增加更多链接。
- **对 Modular 网站社区功能的建议**：建议在 Modular 菜单中增加 **Community** 功能，以改善论坛的访问。
   - 回复指出，不从首页直接链接是刻意的选择，目的是在初始发布期间为用户简化体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.modular.com">Modular: 控制、开发和部署高性能 AI</a>：Modular Accelerated Xecution (MAX) 平台是全球唯一能够跨 CPU 和 GPU 解锁生成式 AI (Generative AI) 性能和便携性的平台。将 AI 集成到各处的每一个产品中。</li><li><a href="https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2956.htm">Unsequenced functions</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1316153872234053735)** (49 条消息🔥): 

> `Multi-Paxos 协议实现，Mojo Struct 设计，Named Results 性能，编程环境偏好，Mojo 开源时间线` 


- **Multi-Paxos 协议初始测试失败**：一位用户用 C++ 实现了 Multi-Paxos 共识协议，但反馈指出初始版本未能满足 Multi-Paxos 的核心要求，例如高效处理多个提案。
   - 超时、Leader 切换和重试等关键特性对于完整的实现至关重要，类似于神经网络中反向传播（back-propagation）的必要性。
- **关于 Mojo Struct 设计一致性的辩论**：讨论围绕 Mojo structs 的 `__del__` 方法应该是选择性加入（opt-in）还是选择性退出（opt-out）展开，意见在一致性与开发者体验（ergonomics）的影响上存在分歧。
   - 一些成员主张采用符合人体工程学的设计以减少样板代码，而另一些成员则更倾向于在 trait 和方法的使用上保持一致。
- **Named Results 的性能保证**：Mojo 中的 Named Results 允许直接写入地址，避免了有时代价高昂的 move 操作，从而在函数返回期间提供性能保证。
   - 虽然这一特性更多被视为一种保证而非直接的性能提升，但它优化了可能无法进行 move 的情况。
- **Mojo 的首选编码环境**：成员们讨论了他们的编码环境，指出许多人因为直接的学习体验而更喜欢为 Mojo 使用 CLI，而其他人则使用 VS Code，因为它提供基础的 linting 支持。
   - 一位用户提到期待使用带有针对新 Magic CLI 兼容性更新的 Zed，展示了社区对不同工具的尝试。
- **关于 Mojo 开源未来的咨询**：一位用户询问了 Mojo 开源的时间线，表达了对社区更广泛的可访问性和协作的渴望。
   - 这反映了当前用户群之外对该平台日益增长的兴趣，突显了对未来发展的期望。


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1316308446739431424)** (13 条消息🔥): 

> `AI 工具和聊天机器人，DM 沟通，支持请求` 


- **特定行业的 AI 工具**：一位成员指出，包括聊天机器人在内的 AI 工具的使用在很大程度上取决于**行业**。
   - 他们暗示，根据应用场景的不同，特定的工具可能会更有效。
- **针对紧急查询的私信**：在对话中，成员们建议针对紧急咨询直接向他们或支持团队发送私信（DM），而不是发送通用请求。
   - 这突显了在处理关键问题时对更个性化沟通的偏好。
- **支持团队协助**：一位成员保证支持团队目前已配备人员，可以回答疑问。
   - 另一位成员重申，具体的沟通应发送至提供的支持邮箱。



**提到的链接**：<a href="https://tenor.com/view/magic-eight-eightball-gif-8220138296338768220">Magic Eight GIF - Magic Eight Eightball - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1316164446275506216)** (8 messages🔥): 

> `Rerank 3.5 English Model, CmdR+Play Bot Status, Aya Expanse Performance, API Request 403 Error, Dataset Recommendations for Quantification` 


- **关于 Rerank 3.5 English 模型的询问**：一名成员询问了关于 **Rerank 3.5 English 模型** 的任何计划。
   - 目前尚未收到针对此话题的回复。
- **CmdR+Play Bot 正在休息**：一名成员对 **CmdR+Play Bot** 的状态感到好奇，另一名成员回复称它目前正在休息，请关注后续更新。
   - *
- **Aya Expanse 与 Command 系列的性能**：有成员提出疑问，**Aya Expanse** 是否受益于 **Command 系列** 的性能，暗示其能力可能存在相关性。
   - 这暗示由于 **Aya Expanse** 可能是基于该系列构建的，它在执行指令方面可能会提供更强的性能。
- **API 请求构建器出现 403 错误**：一名成员报告在尝试使用其 API 请求构建器时收到 **403 错误**，并询问背后的原因。
   - 该问题似乎尚未解决，社区尚未提供任何解释。
- **高质量数据集建议**：一名成员正在寻求建议，想了解所提供的数据集中哪一个对于**量化 (quantification)** 具有更好的质量，并提到需要大量的样本。
   - 他们特别表达了对 **aya_dataset** 中 **'re-annotations'** 标签的数据感兴趣，以用于他们的分析。



**提及的链接**：<a href="https://github.com/hiyouga/LLaMA-Factory">GitHub - hiyouga/LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ LLMs (ACL 2024)</a>：100+ LLMs 的统一高效微调 (ACL 2024) - hiyouga/LLaMA-Factory

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1316224360947781663)** (9 messages🔥): 

> `API Response 403 Issues, VPN Connection Effects, Trial API Key Limitations` 


- **用户在 API 请求时遇到 403 错误**：一名用户报告在尝试使用 API 请求构建器时收到 **403 响应**。
   - 他们提供了 ISP 和位置详情，表明他们正在华盛顿州西雅图使用 **VPN**。
- **疑似与 VPN 相关的限制**：另一名用户建议 **VPN 可能会路由通过受限区域**，从而触发 403 错误。
   - 他们建议原用户尝试在不使用 VPN 的情况下发送请求，看看问题是否仍然存在。
- **分享 API 请求详情**：原用户分享了用于 API 请求的完整 **curl 命令**，包括 headers 和数据负载。
   - 该请求旨在与 Chat 模型交互，询问 Bot 模型的名称。
- **用户尝试了使用和不使用 VPN**：用户确认即使不使用 VPN，使用来自中国的 IPv6 地址时仍然面临 **403 响应**。
   - 他们澄清自己使用的是 **Trial API key**，这可能存在限制。


  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1316250792491094096)** (9 条消息🔥): 

> `Maya 多模态模型, 开源开发, 反馈与支持, 未来视频发布, 具备文化意识的 VLM` 


- **Maya：一个新的多模态多语言模型**：介绍 [Maya](https://x.com/karthik_kanjula/status/1866668743918567469?s=46) —— 一个完全开源的多语言视觉语言模型（VLM），旨在处理 **8 种语言**并关注文化多样性。
   - Maya 基于 LLaVA 框架构建，包含一个新制作的预训练数据集，重点关注**数据质量**和**文化敏感性**。
- **社区对 Maya 的兴奋**：社区成员表达了兴奋之情，评论如 *“太疯狂了！”* 并鼓励大家尝试该模型。
   - 一位成员请求提供视频或录音以更好地了解该模型，预计很快会发布一篇**博客文章**。
- **Maya 的模型细节与可访问性**：Maya 由 Cohere For AI 社区开发，并在 **Apache 2.0 许可证**下运行，确保了可访问性。
   - 该模型相关的论文可以在[这里](https://arxiv.org/abs/2412.07112)找到，强调了其在多语言环境下的指令微调（instruction-finetuned）能力。
- **征集对 Maya 的反馈**：Karthik 鼓励成员尝试 Maya 并提供他们的**反馈**，强调了社区输入的重要性。
   - 团队正积极与用户互动，对从社区获得的持表示感谢。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/karthik_kanjula/status/1866668743918567469?s=46">来自 Karthik reddy Kanjula (@Karthik_kanjula) 的推文</a>：介绍 Maya – 一个新的多模态多语言视觉语言模型。Maya 完全开源、开放权重并开放数据集，旨在处理 8 种语言、文化多样性和细微差别...</li><li><a href="https://huggingface.co/maya-multimodal/maya">maya-multimodal/maya · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/nahidalam/maya/">GitHub - nahidalam/maya: Maya: An Instruction Finetuned Multilingual Multimodal Model using Aya</a>：Maya：一个使用 Aya 的指令微调多语言多模态模型 - nahidalam/maya</li><li><a href="https://huggingface.co/maya-multimodal">maya-multimodal (Maya: Multilingual Multimodal model)</a>：未找到描述</li><li><a href="https://huggingface.co/papers/2412.07112">论文页面 - Maya: An Instruction Finetuned Multilingual Multimodal Model</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1316288802485964861)** (1 条消息): 

> `MOOC 反馈, Hackathon 反馈` 


- **征集 MOOC 反馈**：鼓励参与者使用提供的[匿名反馈表单](https://forms.gle/Paf6cSsCgfmGvHg47)分享他们对 MOOC 的想法。
   - 如果愿意，也可以通过此表单提交 *Hackathon 反馈*。
- **鼓励匿名反馈**：该消息邀请每位参与者提供可以保持匿名的反馈，为分享见解营造一个安全空间。
   - 这一举措旨在收集多元化的观点，以改进课程体验。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1316154188862066789)** (31 条消息🔥): 

> `Hackathon 提交指南、书面文章作业、API Key 使用、反馈提交、结业要求` 


- **Hackathon 提交指南已明确**：确认 **Hackathon 提交**截止日期为 12 月 17 日，而书面文章的截止日期为 **PST 时间 12 月 12 日晚上 11:59**。
   - 学生可以为 Hackathon 提交一篇文章，这与之前要求的 **12 份课程总结**不同。
- **书面文章作业详情**：学生必须在 Twitter、Threads 或 LinkedIn 等平台发布一篇约 **500 字**的帖子，并链接到 MOOC 网站。
   - 文章提交的评分标准为 **通过或不通过 (pass or no pass)**，要求使用与课程注册相同的电子邮件以确保获得学分。
- **API Key 使用并非强制**：参与者在实验作业中使用自己的 **个人 API Key**，但不得提交这些 Key。
   - 明确表示，即使没有收到 OpenAI 的 API Key，在课程作业中使用自己的 API Key 也是可以接受的。
- **提供匿名反馈提交**：创建了一个**匿名反馈表单**，用于收集课程反馈和 Hackathon 建议。
   - 鼓励参与者通过 [反馈表单](https://forms.gle/Paf6cSsCgfmGvHg47) 分享对课程的想法。
- **允许社交媒体提交**：学生可以创建并提交发布在社交媒体平台上的文章链接，包括最终草案的提交。
   - 这在满足**书面文章作业**要求的同时，允许更具互动性的提交过程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>：MOOC，2024 秋季</li><li><a href="https://forms.gle/Paf6cSsCgfmGvHg47">LLM Agents MOOC (匿名) 反馈表单！</a>：如果你愿意，Hackathon 的反馈也可以填在这里 :)</li><li><a href="https://forms.gle/7ekobPNSWDLBWnDT6">书面文章作业提交</a>：说明：创建一个大约 500 字的 Twitter、Threads 或 LinkedIn 帖子。你可以直接在喜欢的平台发布文章，或者在 Medium 上写好后发布链接...
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1316140868456943637)** (2 条消息): 

> `AI 中的 Function Calling、ToolBench 平台、重要研究论文` 


- **AI 中的 Function Calling 受到关注**：提到 AI 模型可能会利用对函数描述和签名的详细理解，根据用户的 Prompt 设置参数，从而增强其泛化能力。
   - 这表明 AI 模型与预定义函数交互的复杂性正在增加。
- **ToolBench：一个新的开放平台出现**：一名成员重点介绍了在 ICLR'24 上展示的 [ToolBench 项目](https://github.com/OpenBMB/ToolBench)，这是一个旨在为工具学习（tool learning）训练、服务和评估 LLM 的开放平台。
   - 其意义在于为推动 AI 工具领域的发展提供资源和框架。
- **关于关键研究论文的讨论**：成员们分享了可能很重要的研究论文链接，包括 [论文 1](https://arxiv.org/pdf/2305.16504) 和 [论文 2](https://arxiv.org/abs/2304.08354)，以便就其与当前主题的相关性进行持续对话。
   - 有人对这些论文是否属于各自主题的顶级资源表示了不确定。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenBMB/ToolBench">GitHub - OpenBMB/ToolBench: [ICLR'24 spotlight] 一个用于工具学习的 LLM 训练、服务和评估的开放平台。</a>：[ICLR'24 spotlight] 一个用于工具学习的 LLM 训练、服务和评估的开放平台。 - OpenBMB/ToolBench</li><li><a href="https://arxiv.org/abs/2304.08354">使用基础模型进行工具学习 (Tool Learning with Foundation Models)</a>：人类拥有创造和利用工具的非凡能力，使他们能够克服生理限制并探索新领域。随着基础模型的出现，AI 系统已经...
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1316148997685116939)** (25 条消息🔥): 

> `Open Interpreter Desktop App, O1 Pro Capabilities, Website Design Feedback, Pricing for Pro Plan, Actions Beta App` 


- **探索 O1 Pro 控制的可能性**：一位成员提议在 OS mode 下使用 Open Interpreter 来控制 **O1 Pro**，认为这可以启用网页搜索、canvas 和文件上传功能。
   - 另一位成员思考了逆向工程 **O1 Pro** 来控制 Open Interpreter 的潜力，并补充道：*'这开启的可能性……太惊人了。'*
- **关于 Open Interpreter 应用可用性的说明**：成员们讨论了 **Open Interpreter app**，确认其目前处于 beta 阶段且需要邀请，目前仅限 **Mac** 用户使用。
   - 一位成员对身处等待名单却尚未获得访问权限表示沮丧，而另一位成员分享了获取邀请的联系方式。
- **对网站新设计的反馈**：对新网站设计的反馈褒贬不一，一位成员表示起初看起来 *'有点突兀'*，但后来逐渐习惯并喜欢上了。
   - 其他人评论说设计仍处于开发中（work-in-progress），并期待未来能有更酷的叠加效果。
- **了解 30 美元的 Pro 方案**：**Killianlucas** 解释说，每月 30 美元的桌面应用方案增加了使用限制，并为免费用户提供无需 **API key** 即可使用的应用。
   - 他建议除非用户觉得非常有益，否则可以坚持使用免费方案，因为该应用在 beta 阶段发展迅速。
- **Actions Beta 应用详解**：**Actions** 功能被强调为一个专注于文件修改的 beta 应用，与仅在终端中可用的 OS mode 不同。
   - 成员们被鼓励探索这一新功能，尽管有人遇到了限制，其中一位指出他们在测试时达到了 **token** 限制。



**提到的链接**：<a href="https://www.openinterpreter.com/">Open Interpreter</a>：未找到描述

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/)** (1 条消息): 

pradipdutta9392: 我认为这对研究人员很有用，就像他们在演示中展示的那样
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1316166462460657796)** (16 条消息🔥): 

> `DoraLinear 初始化、Module 设备处理、梯度管理、参数复制技术、方法签名中 Optional 的使用` 


- **DoraLinear 通过 `to_empty` 提升用户体验**：在当前的 PR 中建议利用 `to_empty` 方法来处理 magnitude 初始化，确保不会破坏现有功能。
   - *在 magnitude 上设置 `requires_grad`* 至关重要，以避免 **self.magnitude.requires_grad=False** 导致的非预期行为。
- **Module 实现中的设备管理**：指出在 `to_empty` 方法中应使用 `swap_tensors` 以在初始化期间进行正确的设备处理。
   - 这允许复制设备设置，当 Tensor 驻留在不同设备上时，这一点非常重要。
- **理解参数初始化中的 `requires_grad`**：讨论明确了只要预先管理好设备考虑因素，复制参数就不会产生问题。
   - 有人对赋值捕获参数表示担忧，建议应谨慎处理赋值操作。
- **`copy_` 和 `swap_tensors` 的功能与需求**：强调了在初始化逻辑中使用 `copy_` 和 `swap_tensors` 的区别，倾向于使用 `copy_` 因为其标准实现。
   - 在 `to_empty` 和 `initialize_dora_magnitude` 中使用 `copy_` 可能被视为一种更直接的解决方案。
- **关于可选方法参数使用的疑问**：针对 `to_empty` 方法中 `device` 参数是否需要 `Optional` 类型提出了疑问，因为它缺少默认值。
   - 结论是 `Optional` 类型旨在通过允许 **None** 值来提供灵活性，从而保留现有的设备设置。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/pull/113647.">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/ebsmothers/ebs-torchtune/blob/5da01406658f9079ebb5bcd6eab0e4261d4188f9/torchtune/modules/peft/dora.py#L123-L126">ebs-torchtune/torchtune/modules/peft/dora.py at 5da01406658f9079ebb5bcd6eab0e4261d4188f9 · ebsmothers/ebs-torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 ebsmothers/ebs-torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L939).">pytorch/torch/nn/modules/module.py at main · pytorch/pytorch</a>: Python 中具有强 GPU 加速的 Tensor 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L963).">pytorch/torch/nn/modules/module.py at main · pytorch/pytorch</a>: Python 中具有强 GPU 加速的 Tensor 和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1316530837415792700)** (1 messages): 

> `QRWKV6-32B, Finch-MoE-37B-A11B, Computational Efficiency Improvements, RWKV-V6 Attention Mechanism, Language Support Limitations` 


- **QRWKV6-32B 性能比肩原始 32B**：基于 Qwen2.5-32B 架构构建的新型 QRWKV6-32B 模型在实现与原始 32B 相同性能的同时，在推理中提供了 **1000 倍的计算效率**。
   - 训练在 **16 块 AMD MI300X GPU** (192GB VRAM) 上仅用 **8 小时** 即可完成，展示了计算成本的显著降低。
- **Finch-MoE-37B-A11B 发布**：Finch-MoE-37B-A11B 模型基于 RWKV 架构构建，是目前正在开发的系列新型 **RWKV 变体** 的一部分。
   - 它强调了向使用线性 Attention 机制的转变，这对于处理 **长上下文 (long contexts)** 特别有效。
- **转换过程支持 RWKV 变换**：一种新颖的转换过程允许将 QKV Attention 模型转换为 RWKV 变体，而无需进行完整的重新训练，从而显著降低了计算开销。
   - 值得注意的是，该模型保留了其父模型的前馈网络 (feedforward network) 架构，导致其与现有的 RWKV 推理代码不兼容。
- **当前上下文限制与能力**：QRWKV6 模型支持约 **30 种语言**，受限于其父模型 Qwen（传统上支持 100 多种语言）。
   - 此外，虽然当前的上下文长度限制在 **16k**，但即使超出此边界也表现出稳定性。



**提到的链接**：<a href="https://x.com/rohanpaul_ai/status/1866971776737218564">Rohan Paul (@rohanpaul_ai) 的推文</a>：新型线性模型：QRWKV6-32B（基于 Qwen2.5-32B 的 RWKV6）和基于 RWKV 的 MoE：Finch-MoE-37B-A11B🚀 Recursal AI 将 Qwen 32B Instruct 模型转换为 QRWKV6 架构，替换了 Transformer attention...

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1316161303118876762)** (10 messages🔥): 

> `O1 Series Impact on DSPy Workflows, Generic Optimization Errors, Backtrack_to Attribute Error, Async Usage Issues, Video and Audio Input Discussions` 


- **O1 系列可能优化 DSPy 工作流**：一位成员询问了 **O1 系列模型** 对 **DSPy 工作流** 的影响，并指出 MIPRO 为优化模块推荐的参数可能需要调整。
   - 他们推测这些新模型可能需要 **更少的优化周期** 或 **评估更少的候选程序**。
- **成员在优化过程中遇到通用错误**：一位用户报告在优化时遇到了 **奇怪的通用错误**，并提到已在特定频道发布了该 Bug 以寻求进一步帮助。
   - 这一问题突显了社区在优化过程中面临的持续挑战。
- **Backtrack_to 属性错误排查**：一位成员分享了一个与 **'backtrack_to'** 不是 DSPy 中 **Settings** 属性相关的错误，并寻求解决办法。
   - 另一位用户表示该问题已在 **早些时候解决**，且可能与某些 **async 使用** 有关。
- **关于视频和音频 IO 的讨论**：一位用户提出了关于 **视频和音频 IO** 看法的问题，引发了成员间的讨论。
   - 一位成员表示，鉴于现有功能，目前专注于 **文本和图像输入** 会更有益。


  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1316137785316479088)** (3 条消息): 

> `Grassroots Science 倡议，优化推理吞吐量，库性能对比，研究论文知识图谱` 


- **多语言 LLM Grassroots Science 项目启动**：一个名为 **Grassroots Science** 的协作倡议定于 2025 年 2 月启动，旨在通过众包数据收集，开发符合多样化人类偏好的多语言 LLM。
   - 该倡议的合作伙伴包括 [SEACrowd](https://seacrowd.github.io/) 和 [Masakhane](https://www.masakhane.io/)，目标是创建一个全面的多语言数据集，并对人类偏好数据进行评估。
- **LLaMA 3.2 推理吞吐量优化**：一位用户寻求在 A100 GPU 上优化 **LLaMA 3.2** 处理 10,000 token 输入的吞吐量，预期速度约为 **每秒 200 个 token**，但感觉这比预期的要慢。
   - 讨论内容包括探索 Batching、Prompt Caching 以及使用量化版本来增加 Batch Size 的潜在收益。
- **TGI 3.0 表现出优于 vLLM 的性能**：根据 Hugging Face 的数据，**TGI 3.0** 处理的 **token 数量可多出 3 倍**，且速度比 **vLLM 快 13 倍**，使其非常适合高效处理长 Prompt。
   - TGI 显著降低了内存占用，使其能够在 **LLaMA 3.1-8B** 上处理高达 **30k token**，而 vLLM 在处理 **10k token** 时就已显得吃力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/text-generation-inference/conceptual/chunking">TGI v3 概览</a>：未找到描述</li><li><a href="https://artificialanalysis.ai/models/llama-3-2-instruct-3b?utm_source=chatgpt.com">Llama 3.2 3B - 质量、性能与价格分析 | Artificial Analysis</a>：对 Meta 的 Llama 3.2 Instruct 3B 的分析，并在质量、价格、性能（每秒 token 数和首个 token 响应时间）、上下文窗口等关键指标上与其他 AI 模型进行对比...</li><li><a href="https://grassroots.science/">Grassroots Science</a>：一个专注于通过基层努力开发最先进多语言语言模型的全球倡议。</li><li><a href="https://forms.gle/i8mG999yRbznK8JE9">Grassroots Science 意向表单</a>：Grassroots Science 是一个为期一年的全球协作项目，旨在通过众包收集多语言数据，由相信集体力量的基层社区发起...</li><li><a href="https://x.com/GrassrootsSci">来自 undefined 的推文</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1316201570659991613)** (6 条消息): 

> `非 LLM 泛化，十亿以下参数模型，COCONUT 范式，高效小模型` 


- **关于模型缩放 (Scaling) 的辩论**：一位成员对缩放模型的必要性表示怀疑，称 *“十亿参数对任何人来说都应该足够了”*。他们表达了对“规模即一切 (scale-is-all-you-need)”运动的不同意见。
   - 成员们还分享了对*超高效小模型*的热情，强调了它们在模型训练中的优势。
- **COCONUT：LLM 的新推理范式**：重点介绍了连续思维链 (Chain of Continuous Thought, **COCONUT**) 的引入，这是一种训练大语言模型在连续潜空间中进行推理的方法，详见此[推文](https://x.com/iScienceLuvr/status/1866353795502158163/photo/1)。
   - 与传统的通过隐藏状态和语言 token 进行映射不同，COCONUT 将最后一个隐藏状态作为输入嵌入 (embedding)，从而通过梯度下降实现端到端优化。



**提到的链接**：<a href="https://x.com/iScienceLuvr/status/1866353795502158163/photo/1">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：训练大语言模型在连续潜空间中推理。引入了一种名为连续思维链 (COCONUT) 的 LLM 推理新范式。极其简单的改变：不再映射...

  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/)** (1 条消息): 

c.gato: 那应该是默认设置

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1316478576693874819)** (1 messages): 

> `Mozilla AI hiring, Community Engagement Head role, Lumigator product, Developer Hub, Blueprints initiative` 


- **Mozilla AI 正在寻找人才**：Mozilla AI 正在招聘 [Head of Community Engagement](https://job-boards.greenhouse.io/mozillaai/jobs/4600382007)，提供远程办公机会，直接向 CEO 汇报。
   - 该职位将领导并扩展跨渠道的社区计划，以增强参与度。
- **推出用于 LLM 选择的 Lumigator**：Mozilla 正在开发 **Lumigator**，这是一款旨在帮助开发者自信地为项目选择最佳 LLM 的产品。
   - 这是他们为开发者社区提供值得信赖的开源 AI 解决方案努力的一部分。
- **Developer Hub 简化 AI 资源**：Mozilla AI 正在创建一个 **Developer Hub**，开发者可以在其中找到用于构建开源 AI 的精选资源。
   - 该计划支持 AI 开发中的用户代理权（User Agency）和透明度。
- **Blueprints：开源 AI 集成**：**Blueprints** 计划旨在通过入门代码库开源 AI 集成，以启动 AI 项目。
   - 对于希望快速实现 AI 解决方案的开发者来说，这些资源将非常宝贵。
- **社区参与咨询**：潜在申请人可以在此 [thread](https://discord.com/channels/1089876418936180786/1316478017530495007/1316478017530495007) 中询问有关 Head of Community Engagement 职位的相关问题。
   - 该职位体现了 Mozilla AI 对社区驱动计划的承诺。



**提及的链接**：<a href="https://job-boards.greenhouse.io/mozillaai/jobs/4600382007)">Head of Community Engagement</a>：远程

  

---


---


---


---


---


{% else %}


> 完整的各频道详细分类已在邮件中截断。 
> 
> 如果你想查看完整分类，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}