---
companies:
- anthropic
- meta-ai-fair
- openai
- deepseek-ai
- llamaindex
- langchainai
date: '2024-09-21T01:37:46.121441Z'
description: '以下是该文本的中文翻译：


  **Anthropic** 推出了一种名为“上下文检索”（Contextual Retrieval）的 RAG 技术，通过使用提示词缓存（prompt caching）将检索失败率降低了
  67%。**Meta** 在 Meta Connect 大会前夕预热了多模态 **Llama 3**。**OpenAI** 正在为其多智能体研究团队招聘人才，该团队专注于利用其
  **o1 模型**提升 AI 推理能力，而该模型目前引发了褒贬不一的反应。**DeepSeek 2.5** 被视为 **GPT-4** 和 **Claude 3.5
  Sonnet** 的高性价比替代方案。文中还重点介绍了用于 3D 资产生成的 **3DTopia-XL** 和用于图生视频的 **CogVideoX** 等新模型。此外，分享了通过重读问题以及将检索与提示词缓存相结合来增强推理能力的技术。行业洞察强调了企业采用
  AI 的必要性以及对传统机器学习业务的颠覆。**LangChainAI 的 LangGraph 模板**和 **LlamaIndex 的 LlamaParse
  Premium** 等工具增强了智能体应用和多模态内容提取。关于大语言模型评估（evals）和缓存的讨论突出了生产环境中的挑战与改进。一个核心观点是：“不允许开发人员使用
  AI 的公司不太可能成功”。'
id: bcaa22ff-74a7-41b9-ba6f-51b5b3fc1ea5
models:
- llama-3
- o1
- deepseek-2.5
- gpt-4
- claude-3.5-sonnet
- 3dtopia-xl
- cogvideox
original_slug: ainews-not-much-happened-today-5059
people: []
title: 今天没发生什么特别的事。
topics:
- retrieval-augmented-generation
- prompt-caching
- multimodality
- multi-agent-systems
- reasoning
- diffusion-models
- image-to-video
- prompting
- enterprise-ai
- agentic-ai
- long-context
- model-evaluation
- caching
- model-cost-efficiency
---

<!-- buttondown-editor-mode: plaintext -->**定制化的 AINews 可能很快就是你所需要的一切...**

> 2024年9月19日至9月20日的 AI News。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务（**221** 个频道和 **2035** 条消息）。预计节省阅读时间（按 200wpm 计算）：**258 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 来进行 AINews 讨论！

Anthropic 撰写了关于 [Contextrual Retrieval](https://www.anthropic.com/news/contextual-retrieval) 的文章，这是一种利用其 prompt caching 功能的 RAG 技术，结果显示 Reranked Contextual Embedding 和 Contextual BM25 将 top-20-chunk 的检索失败率降低了 67%（5.7% → 1.9%）：


![image.png](https://assets.buttondown.email/images/9d4ebb6a-2651-4877-aaf7-114554443199.png?w=960&fit=max)


然而，这仅仅是一项 RAG 技术，所以我们觉得它还不值得作为头条新闻。

Meta 团队正在为下周 Meta Connect 上的 multimodal Llama 3 进行[大量](https://reddit.com//r/LocalLLaMA/comments/1fkyiim/metas_llama_has_become_the_dominant_platform_for/)[预热](https://reddit.com//r/LocalLLaMA/comments/1fl2l86/zuck_is_teasing_llama_multimodal_over_on_ig/)，但在它正式发布之前，我们还不能将其作为头条新闻。

与此同时，如果你一直渴望拥有自己的个人 AINews，或者想为我们提供一些 inference 资金，你现在可以[注册我们的 “AINews Plus” 服务](https://buy.stripe.com/dR602I7Sv7FYfN69AA)，并拥有**针对你选择的任何主题的定制化 AI News 服务**！

https://youtu.be/iDCUYZgnAjY

如果你在旧金山，本周末的 [LLM as Judge Hackathon](http://wandb.me/swyx-hack) 见！

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要回顾

> 所有摘要由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 研究与开发**

- **OpenAI 的 o1 模型**：[@polynoamial](https://twitter.com/polynoamial/status/1836872735668195636) 宣布 OpenAI 正在为新的 multi-agent 研究团队招聘 ML 工程师，将 multi-agent 视为通往更好 AI 推理能力的路径。[@scottastevenson](https://twitter.com/scottastevenson/status/1836811502340252020) 指出 o1 模型在技术专家中引起了困惑和怀疑，类似于早期对 GPT-3 和 ChatGPT 的反应。[@nptacek](https://twitter.com/nptacek/status/1836832186558734662) 观察到 o1 在 prompting 方面感觉不同，需要更多以目标为导向而非指令驱动的方法。

- **AI 模型进展**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1836750072639369419) 将 DeepSeek 2.5 与 GPT-4 进行了比较，指出其价格比 Claude 3.5 sonnet 便宜 21 倍，比 GPT-4 便宜 17 倍。[@_akhaliq](https://twitter.com/_akhaliq/status/1836754453644398667) 分享了关于 3DTopia-XL 的信息，这是一个使用 Diffusion Transformer 的高质量 3D PBR 资产生成模型。[@multimodalart](https://twitter.com/multimodalart/status/1836780383813185541) 强调了 CogVideoX 的图生视频能力，特别是对于延时摄影（timelapse）视频。

- **AI 研究洞察**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1836890159314522445) 讨论了一种强大但简单的 prompting 技术，即要求 LLM 重新阅读问题，这显著提升了不同任务和模型类型的推理能力。[@alexalbert__](https://twitter.com/alexalbert__/status/1836854956785352776) 分享了关于 Contextual Retrieval 的研究，该技术在结合 prompt caching 时，可将错误的 chunk 检索率降低高达 67%。

**AI 行业与应用**

- **企业中的 AI**：[@svpino](https://twitter.com/svpino/status/1836857830470717514) 表示，不允许开发人员使用 AI 的公司不太可能成功。[@scottastevenson](https://twitter.com/scottastevenson/status/1836767834833145986) 指出 LLM 如何颠覆了传统的 ML 业务，深厚的护城河在几个月内就消失了。

- **AI 工具与平台**：[@LangChainAI](https://twitter.com/LangChainAI/status/1836789918250500355) 宣布了 LangGraph Templates，这是一系列用于创建 agentic 应用的参考架构。[@llama_index](https://twitter.com/llama_index/status/1836798520394686917) 推出了 LlamaParse Premium，结合了多模态模型的视觉理解能力与长文本/表格内容提取。

- **生产环境中的 AI**：[@HamelHusain](https://twitter.com/HamelHusain/status/1836816587200024658) 分享了关于使用 LLM evals 改进 AI 产品的建议，展示了如何创建数据飞轮以从 demo 转向生产就绪的产品。[@svpino](https://twitter.com/svpino/status/1836737020485656818) 讨论了 LLM 应用中缓存（caching）的重要性及挑战，以提高速度和成本效益。

**AI 伦理与监管**

- [@ylecun](https://twitter.com/ylecun/status/1836805202076180909) 讨论了研究误导信息（misinformation）的科学家的政治倾向，指出科学家通常倾向于左派，因为他们关注事实，而目前误导信息主要来自右派。[@ylecun](https://twitter.com/ylecun/status/1836807353708269718) 还分享了一封由行业领袖签署的公开信，敦促欧盟统一 AI 监管，以防止该地区成为技术落后地区。

- [@fchollet](https://twitter.com/fchollet/status/1836809075440660805) 澄清说，ARC-AGI 基准测试并非专门为难倒 LLM 而设计，而是为了突出深度学习的局限性，而 LLM 作为同一范式的一部分也具有这些局限性。

**迷因与幽默**

- 各种推文展示了幽默的 AI 生成内容，包括 [@nearcyan](https://twitter.com/nearcyan/status/1836779472080527375) 为了“像海盗一样说话日”包装了他们的整个 Twitter 应用，以及 [@jxnlco](https://twitter.com/jxnlco/status/1836821893078471037) 分享了一张有趣的 AI 生成图片。


---

# AI Reddit 摘要回顾

## /r/LocalLlama 摘要回顾

**主题 1. Llama 3 多模态：Meta 的下一个重大 AI 发布**

- **“Meta 的 Llama 已成为构建 AI 产品的核心平台。下一个版本将是多模态的，并能理解视觉信息。”** ([Score: 74, Comments: 21](https://reddit.com//r/LocalLLaMA/comments/1fkyiim/metas_llama_has_become_the_dominant_platform_for/))：**Yann LeCun** 在 LinkedIn 上宣布，**Meta 的 Llama 3** 将是一个具有**视觉理解**能力的**多模态**模型。Llama 的下一次发布旨在进一步巩固 Meta 在 AI 产品开发领域的地位。

- **Zuck 正在 IG 上预热 Llama 多模态模型。** ([Score: 164, Comments: 42](https://reddit.com//r/LocalLLaMA/comments/1fl2l86/zuck_is_teasing_llama_multimodal_over_on_ig/)): **Mark Zuckerberg** 在 **Instagram** 上暗示了 **Llama** 的**多模态能力**。这一进展预计将在下周举行的 **Meta Connect** 大会上正式揭晓。
  - **Llama.cpp** 开发者对**多模态模型**和**工具调用 (tool calling)** 缺乏支持，令用户感到失望。该团队专注于**极致的底层效率**以及 **CPU/CPU+GPU 推理**，导致多模态实现需要从零开始重做。
  - 用户讨论了 **Llama.cpp** 与 **TabbyAPI**、**ExllamaV2** 和 **KTransformers** 等其他后端的性能对比。一些人认为通过更好的优化、**投机解码 (speculative decoding)** 和**张量并行 (tensor parallelism)**，**Llama.cpp** 的 **GPU 性能**仍有提升空间。
  - 社区对 **Llama.cpp** 缺乏对 **Meta** 的 **Chameleon** 模型支持表示不满，尽管一位 Meta 开发者提供了帮助。一个实现该支持的 Pull Request 未被合并，导致贡献者们感到失望。


**主题 2. Qwen2.5 32B：在 GGUF 量化中表现出色**


- **Qwen2.5 32B GGUF 评估结果** ([Score: 78, Comments: 38](https://reddit.com//r/LocalLLaMA/comments/1fkm5vd/qwen25_32b_gguf_evaluation_results/)): 对 **Qwen2.5 32B GGUF** 模型的评估显示，其在 **MMLU PRO** 的**计算机科学类别**中表现强劲，其中 **Q4_K_L** 量化 (**20.43GB**) 得分为 **72.93**，**Q3_K_S** 量化 (**14.39GB**) 得分为 **70.73**，性能损失仅为 **3.01%**。这两个 Qwen2.5 32B 量化版本的表现都显著优于 **Gemma2-27b-it-q8_0** 模型 (**29GB**)，后者在同一类别中得分为 **58.05**。
  - **Qwen2.5 32B** 的量化版本表现令人印象深刻，用户注意到尽管在**世界知识和审查 (censorship)** 方面可能存在缺陷，但在某些领域有显著改进。
  - 用户建议测试 **IQ 变体量化**，这被认为是 **4-bit 以下的 SOTA**，通常优于旧的 Q_K 类型量化。对于 24GB VRAM 用户，比较 **72B IQ3_XXS (31.85GB)** 和 **IQ2_XXS (25.49GB)** 版本的兴趣很高。
  - 围绕 **Hugging Face** 上官方 **Qwen/Qwen2.5 GGUF 文件**的讨论指出，官方量化版本的表现往往不如社区创建的版本。


- **手机上的 Qwen 2.5：PocketPal 已添加 1.5B 和 3B 量化版本** ([Score: 74, Comments: 34](https://reddit.com//r/LocalLLaMA/comments/1fkogmk/qwen_25_on_phone_added_15b_and_3b_quantized/)): **Qwen 2.5** 模型，包括 **1.5B (Q8)** 和 **3B (Q5_0)** 版本，已添加到适用于 [iOS](https://apps.apple.com/us/app/pocketpal-ai/id6502579498) 和 [Android](https://play.google.com/store/apps/details?id=com.pocketpalai) 平台的 **PocketPal** 移动端 AI 应用中。用户可以通过该项目的 [GitHub 仓库](https://github.com/a-ghorbani/PocketPal-feedback/issues)提供反馈或报告问题，开发者承诺会抽空解决这些问题。
  - 用户表达了对添加 **语音转文字 (speech-to-text)** 功能和修改 **系统提示词 (system prompt)** 的兴趣。开发者确认大多数设置都是可自定义的，并分享了可用选项的[截图](https://preview.redd.it/i5j52257sspd1.png?width=1290&format=png&auto=webp&s=acdf079983770322c5c4bf50881cbb208f380d76)。
  - 一位用户询问了 **上下文大小 (context size)** 设置，引发了关于**上下文长度**与**生成时间参数**之间区别的讨论。开发者解释了其放置位置背后的逻辑，并将该问题添加到了 [GitHub 仓库](https://github.com/a-ghorbani/PocketPal-feedback/issues/10)中。
  - 该应用支持多种 **聊天模板 (chat templates)** (ChatML, Llama, Gemma) 和模型，用户对比了 **Qwen 2.5 3B (Q5)**、**Gemma 2 2B (Q6)** 和 **Danube 3** 的性能。开发者提供了相关截图。


**主题 3. 欧盟 AI 监管：平衡创新与控制**

- **爱立信（Ericsson）发布公开信，由 Meta 协调，探讨欧洲碎片化的监管如何阻碍 AI 机遇** ([Score: 87, Comments: 16](https://reddit.com//r/LocalLLaMA/comments/1fkh311/open_letter_from_ericsson_coordinate_by_meta/))：**爱立信 CEO Börje Ekholm** 警告称，碎片化的欧盟监管正在阻碍欧洲的 AI 发展，可能使欧洲人无法享受到其他地区已有的技术进步。信中强调，**开放模型（open models）**能增强主权和控制力；据估计，**生成式 AI（Generative AI）**在未来十年内可使全球 GDP 增长 **10%**。信中敦促制定清晰、一致的规则，以便能够使用欧洲数据进行 AI 训练。[点击此处阅读公开信全文](https://www.ericsson.com/en/news/2024/9/open-letter-on-fragmented-regulation-risks-to-eu-in-ai-era)。
  - 评论者们就欧盟监管对 **AI 创新**的影响展开辩论，一些人认为这可能导致**欧洲在未来 AI 技术上依赖美国**。另一些人则主张建立类似于 **GDPR** 的**通用框架**，以明确全欧洲的规则并促进投资。
  - 讨论集中在 AI 监管的范围上，有建议认为应侧重于禁止监控和歧视等**“1984 式的行为”**，而不是监管模型本身。发帖者澄清说，问题的核心在于监管**用于 AI 训练的数据**，而非 AI 的使用。
  - 文中分享了 [euneedsai.com](https://euneedsai.com/) 的链接，可能为欧洲的 AI 需求和监管环境提供更多背景信息。


- **快速提醒：SB 1047 尚未签署成为法律，如果你住在加州，请给州长留言** ([Score: 209, Comments: 57](https://reddit.com//r/LocalLLaMA/comments/1fkfkth/quick_reminder_sb_1047_hasnt_been_signed_into_law/))：**加州的 SB 1047** 是一项受《终结者》启发的 AI “安全”法案，虽然已经通过但尚未签署成为法律。该帖子敦促**加州居民**通过官方[联系页面](https://www.gov.ca.gov/contact/)向州长表达反对意见，在主题中选择“**An Active Bill**”和“**SB 1047**”，并选择“**Con**”（反对）作为立场。
  - 批评者认为 **SB 1047** 是一项**监管俘获（regulatory capture）法案**，可能会阻碍**开放研究**，同时让那些在没有安全检查的情况下进行封闭式、利润驱动研究的企业受益。一些人认为该法案可能**违宪**，但也有人认为它可能是合法的。
  - 评论者强调了**开源 AI** 对研究、通用用途以及通过协作开发实现长期安全的重要性。他们建议在联系官员时提及所在地、选民身份以及开源 AI 带来益处的个人案例。
  - 对**中国 AI 进步**的担忧被作为反对监管的一个理由。文中分享了一个专门的网站 [stopsb1047.com](https://stopsb1047.com)，用于提交反对该法案的评论，一些用户反馈已发送了详细的回复。


**主题 4. Mistral Small 2409 22B：量化影响分析**



- **Mistral Small 2409 22B GGUF 量化评估结果** ([Score: 106, Comments: 25](https://reddit.com//r/LocalLLaMA/comments/1fl2ck8/mistral_small_2409_22b_gguf_quantization/))：该帖子展示了 **Mistral Small Instruct 2409 22B** 模型的量化评估结果，重点关注 **MMLU PRO** 基准测试中的计算机科学类别。测试了各种量化级别，令人惊讶的是 **Q4_K_L** 变体以 **60.00%** 的准确率优于其他变体，模型大小范围从 **9.64GB** 到 **18.35GB**。作者还包含了 **Qwen2.5-32B** 和 **Gemma2-27b** 模型的对比结果，并提供了 GGUF 模型、后端、评估工具和所用配置的链接。
  - **Q4_K_L** 量化表现优于 **Q5_K_L** 引发了讨论，用户推测这可能是由于**随机性**或**层差异**造成的。测试在 **0 temperature** 下运行，Q4_K_L 达到了 **60.20%** 的准确率（407 道题中答对 245 道）。
  - **Qwen2.5-32B** 的性能受到称赞。用户要求与 **Mistral Nemo 12B** 进行对比，作者确认已完成评估并稍后发布。
  - 讨论涉及了量化效应，有传闻称某些模型的 **5-bit** 量化表现比 **4-bit** 更差。一位用户的测试表明，在某些场景下 **Q4** 变体可能比 **Q6** “更聪明”。


**主题 5. AI 模型大小之争：效率 vs 能力**

- **热门观点：Llama3 405B 可能确实太大了** ([Score: 104, Comments: 94](https://reddit.com//r/LocalLLaMA/comments/1fkpdks/hot_take_llama3_405b_is_probably_just_too_big/)): **Llama3.1-405B** 最初在开源模型中处于领先地位，但与 **Mistral Large (~120B)** 等更高效的模型相比，现在被认为在实际应用中过于庞大。该帖子认为 **27-35B** 和 **120B** 模型将成为行业标准，公司会先部署现成的 120B 模型，然后通过微调 30B 模型来降低超过 **50%** 的成本。在承认 Meta AI 贡献的同时，作者强调需要更多 **100B+** 模型，因为它们比更大的模型在训练、微调和托管方面成本更低。
  - AI 模型的**行业标准**引发了辩论，一些人认为公司会使用任何效果最好的模型，而不考虑其大小。**405B 模型**被认为对研究、蒸馏以及关注**数据隐私**的大型组织内部使用非常有用。
  - 像 **Llama 405B** 这样的大型模型被视为突破边界以及与传闻中拥有 **1.7T 参数**的 GPT-4 等模型竞争的重要力量。一些用户认为，创建 SOTA 模型对于研究和收集训练数据具有重要价值。
  - 讨论了大型模型的实际应用，一些用户报告称每天通过 API 使用 **405B 模型**以获得更好的响应。人们对如何在不产生过高成本或复杂性的情况下**微调 70B+ 模型**的教程表现出浓厚兴趣。

## 其他 AI Subreddit 综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 研究与技术**

- **Google Deepmind 通过联合样本选择推进多模态学习**：一篇 [Google Deepmind 的论文](https://arxiv.org/html/2406.17711v1) 展示了通过联合样本选择（joint example selection）进行数据策展如何进一步加速多模态学习。该技术可以提高训练大型多模态模型的效率。

- **Microsoft 的 MInference 显著加快了长上下文任务的推理速度**：[Microsoft 的 MInference 技术](https://arxiv.org/abs/2407.02490) 能够在保持准确性的同时，实现多达数百万个 Token 的长上下文任务推理，大幅提升了支持模型的运行速度。这使得处理超长文档或对话变得更加高效。

- **利用 10 亿个网络策划的角色扩展合成数据生成**：一篇关于[扩展合成数据生成](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/)的论文利用 LLM 中的多样化视角，从网络数据策划的 10 亿个角色（personas）中生成数据。这种方法有助于创建更具多样性和代表性的训练数据集。

**AI 模型发布与改进**

- **Salesforce 的“小巨人” xLAM-1b 模型在函数调用方面超越 GPT 3.5**：Salesforce 发布了 xLAM-1b，这是一个 10 亿参数的模型，在[**函数调用（function calling）中实现了 70% 的准确率，超越了 GPT 3.5**](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。这展示了更小、更高效模型的显著进步。

- **具备函数调用能力的 Phi-3 Mini (六月版)**：Rubra AI 发布了更新后的 Phi-3 Mini 模型 [**具备函数调用功能**](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)。它具有与 Mistral-7b v3 竞争的实力，并优于基础版 Phi-3 Mini，显示了小型开源模型的快速进展。

- **OmniGen 多模态模型**：一篇[新的研究论文](https://arxiv.org/pdf/2409.11340)介绍了 OmniGen，这是一种内置 LLM 和视觉模型的多模态模型，通过 Prompting 提供前所未有的控制力。它可以根据文本指令操作图像，而无需专门的训练。

**AI 发展与行业趋势**

- **OpenAI 融资轮因需求高涨即将结束**：OpenAI 的[最新一轮融资即将结束](https://www.bloomberg.com/news/articles/2024-09-19/openai-to-decide-which-backers-to-let-into-6-5-billion-funding?srnd=homepage-americas)，由于需求极高，他们不得不拒绝了“数十亿美元”的超额认购。这表明投资者对公司未来充满信心。

- **关于 LLM API 与 ML 产品开发的争论**：[r/MachineLearning 上的一场讨论](https://www.reddit.com/r/MachineLearning/comments/1fl5be0/d_i_feel_like_ever_since_llm_apis_have_become_a/)引发了关注，即 LLM API 的普及导致人们过度关注 Prompt Engineering，而非更基础的 ML 研究与开发。这反映了关于 AI 研发方向的持续争论。

- **不可磨灭的 5D 存储水晶**：[新技术](https://interestingengineering.com/innovation/5d-memory-crystals-to-store-humanitys-genome)允许在坚固的水晶中存储高达 360 TB 的数据并保存数十亿年，这可能为长期保存人类知识提供一种方式。


---

# AI Discord 回顾

> 由 O1-preview 提供的摘要之摘要的总结

**主题 1. 新 AI 模型在社区引起轰动**

- [**Qwen 2.5 成为焦点**](https://openrouter.ai/models/qwen/qwen-2.5-72b-instruct)：Unsloth AI 确认支持 **Qwen 2.5**，用户正积极训练 **Qwen2.5-14b**。OpenRouter 推出 **Qwen2.5 72B**，具备增强的代码和数学能力，以及高达 **131,072** 的 **context size**。
- [**Mistral 进军 Multimodal AI**](https://openrouter.ai/models/mistralai/pixtral-12b)：**Mistral Pixtral 12B** 作为 Mistral 的首个 **multimodal** 模型首次亮相，现已可在 OpenRouter 上访问。这次发布标志着一个关键时刻，扩展了 Mistral 在多功能 AI 应用领域的产品线。
- [**Flux 模型点燃 Stability.ai 用户热情**](https://huggingface.co/nyanko7/flux-dev-de-distill)：**Flux** 模型凭借卓越的 **prompt adherence** 和图像质量给人留下深刻印象，克服了最初的资源障碍。尽管对审美相似性存在一些担忧，但对其性能的乐观情绪依然高涨。

**主题 2. 模型微调：胜利与磨难**

- [**LoRA 微调激发创新**](https://github.com/huggingface/diffusion-models-class)：HuggingFace 用户建议利用 **LoRA** 微调基础模型，灵感来自 **ipadapter** 方法。这可以在不进行大规模重新训练的情况下提升模型性能。
- [**Phi-3.5-Mini 带来意外挑战**](https://github.com/unslothai/unsloth/issues/946)：Unsloth AI 用户在微调 **Phi-3.5-Mini** 时遇到 **AttributeError**，尽管遵循了推荐的修复方法，仍需处理 **LongRopeRotaryEmbedding** 问题。社区正在寻找可行的变通方案。
- **Quantization 权衡引发辩论**：成员们讨论认为，未量化的模型在批处理中可能会提供更好的速度和吞吐量。在决策中，**speed**、**size** 和 **cost** 之间的关键平衡占据了中心位置。

**主题 3. AI 工具考验用户耐心**

- **Aider 与 API 问题搏斗**：用户正努力解决 Aider 无法从 `.env` 文件读取的问题，这导致了配置混乱以及 **Anthropic API** 的过载错误。记录 LLM 对话历史成为首选的排查方法。
- [**LangChain 的分块输出令人懊恼**](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/openai_assistants/#using-existing-assistant)：LangChain v2.0 用户报告称，在使用 OpenAI **streaming** 时，**function call information** 会间歇性地以分块形式输出。对潜在 bug 的怀疑促使人们呼吁修复。
- [**LM Studio 连接难题获解**](https://support.apple.com/en-gb/guide/mac-help/mh14129/mac)：在 macOS 上切换到 **IPv4** 为面临连接困扰的 LM Studio 用户化解了危机。关于调整设置的清晰指导将沮丧转为宽慰。

**主题 4. AI 编程助手引发讨论**

- **O1 模型在代码编辑中受到质疑**：Aider 用户对 O1 模型与 **Sonnet-3.5** 相比的性能表示怀疑，特别是在代码重构任务中。人们仍对未来增强交互能力的改进抱有很高期望。
- [**Wizardlm 在 OpenInterpreter 中展现魔力**](https://github.com/microsoft/Wizardlm)：**Wizardlm 8x22B** 在 OpenInterpreter 中的表现优于 **Llama 405B**，更频繁地在第一次尝试时就搞定任务。用户对其效率和效果印象深刻。
- [**Devin 加倍改进**](https://x.com/cognition_labs/status/1836866696797401118)：**Devin** 现在提供更快、更准确的代码编辑以及改进的企业安全支持。虽然许多人称赞这些更新，但反馈仍然褒贬不一，一些用户对局限性表示沮丧。

**主题 5. 社区活动与协作努力**

- [**黑客松热潮拉开序幕**](https://rsvp.withgoogle.com/events/web-ai-summit-2024)：CUDA MODE 成员正为黑客松做准备，审批流程正在进行，并鼓励通过论坛想法组建团队。有机会让 Wen-mei Hwu 教授在 **PMPP book** 上签名，这增添了额外的兴奋感。
- [**OpenAI 征集 Multi-Agent 奇才**](https://jobs.ashbyhq.com/openai/form/oai-multi-agent)：OpenAI 正在为一个新的 **multi-agent** 研究团队招聘 ML 工程师，认为这一领域对于增强 AI 推理至关重要。他们鼓励即使没有 **multi-agent** 经验的人也积极申请。
- [**Web AI Summit 2024 即将来临**](https://rsvp.withgoogle.com/events/web-ai-summit-2024)：成员们对即将举行的峰会中的社交机会表示热忱。该活动承诺在热情的参与者之间就 Web AI 主题进行宝贵的交流。


---

# 第 1 部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **探索高级 Tokenization 技术**：一篇名为 [This Title Is Already Tokenized](https://huggingface.co/blog/apehex/this-title-is-already-tokenized) 的新博客文章解释了高级方法，引起了人们对其在现代 NLP 中应用的兴趣。
   - 内容详细介绍了 Tokenization 的复杂性，推动了关于其与当前项目相关性的讨论。
- **用于语言模型训练的 Unity ML Agents**：观看关于使用 Unity ML Agents 和 sentence transformers 从头开始训练 LLM 的最新 [YouTube 视频](https://youtube.com/live/0foHMTPWa4Y?feature=share)。
   - 本集重点介绍了 **Oproof 验证成功**，展示了 Tau LLM 系列的关键里程碑。
- **发布新的 GSM8K 推理数据集**：一位用户介绍了一个基于 GSM8K 的新[推理数据集](https://huggingface.co/datasets/thesven/gsm8k-reasoning)，旨在用于 AI 模型训练。
   - 预计将通过其结构化的挑战增强 AI 的批判性推理能力。
- **分形生成器的新缩放功能**：一个分形生成器项目现在通过“Aiming Better”部分实现了缩放功能，允许用户调整网格长度并生成新输出。
   - 社区建议包括实现滚轮输入以实现更平滑的交互。
- **使用 LoRA 微调基础模型**：有人建议利用 **LoRA** 微调基础模型，并从 **ipadapter** 方法论中汲取灵感。
   - 这可以通过调整参数而无需进行广泛的重新训练来增强模型性能。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 在 API 交互方面遇到困难**：用户面临 Aider 无法读取 `.env` 文件的问题，导致配置挑战和 Anthropic API 的重载错误。建议将记录 LLM 的对话历史作为一种潜在的诊断方法。
   - 在此背景下，进一步调查配置问题对于确保更顺畅的 API 交互至关重要。
- **O1 模型与 Sonnet-3.5 的对比**：对于 Aider 中 `O1` 模型相对于 Sonnet-3.5 的性能（特别是在编辑和代码重构等任务中）存在怀疑。用户期待能够增强 Aider 与 O1 模型之间交互能力的改进。
   - 这种对比引发了关于模型集成和在编码任务中可用性的更广泛讨论。
- **Chain of Thought 引发辩论**：一位成员质疑 [Chain of Thought 方法](https://huggingface.co/spaces/cerebras/chain-of-thought) 的有效性，认为先前的训练对性能影响更大。讨论表明，针对结果进行务实的微调对于定制化应用至关重要。
   - 这突出了 AI 讨论中的一个共同主题，即通过适当的方法论实现模型性能。
- **Anthropic 通过 Contextual Retrieval 增强 LLM 运行**：Anthropic 引入了一种 [Contextual Retrieval 方法](https://www.anthropic.com/news/contextual-retrieval)，可改进 Prompt 缓存以实现高效的 LLM 运行。该方法的实现被认为对 Aider 等项目至关重要。
   - 总的来说，它强调了持续改进 AI 交互管理以简化功能的必要性。
- **Aider 中的函数重命名问题**：Aider 尝试重命名函数导致了部分更新，从而引发未定义函数错误，这引起了对其搜索/替换有效性的担忧。用户注意到，尽管有提示，Aider 仅修复了一个 linter 错误实例。
   - 增强引用更新功能的必要性显现出来，暗示 Aider 的架构有很大的改进空间。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 2.5 发现 Unsloth 兼容性**：用户确认 Unsloth 已支持 **Qwen 2.5**，尽管 Qwen 团队正在解决聊天模板（chat templates）的 bug。
   - “最好是这样，我正在训练 Qwen2.5-14b” 这种情绪表达了对功能支持的紧迫需求。
- **成功微调 Qwen 2.5 等模型**：对于在有限数据集上进行文本分类的 LLM 微调，**Qwen 2.5** 和 **BERT** 等模型非常理想，一位用户使用 **Llama 8.1 8B** 达到了 **71% 的准确率**。
   - 成员们正寻求提高这些分数，并分享成功经验与挑战。
- **关键的量化权衡**：讨论表明，未量化的模型可能具有更好的速度和吞吐量，特别是在批处理（batch processing）场景中。
   - 成员们在决定模型量化时，对**速度**、**尺寸**和**成本**之间的关键权衡进行了辩论。
- **AGI 进展引发辩论**：有观点认为，实现 **AGI** 不仅仅是寻找答案，更多的是有效地解释答案，这暗示着未来面临巨大挑战。
   - 呼应 **80/20 法则**，有人指出对 AGI 长达 **60 年** 的投入表明了其实现路径的艰辛。
- **BART 模型的输入机制受到关注**：关于 BART 的输入格式出现了疑问，强调它使用 **EOS token** 而不是预期的 **BOS token** 来开始生成。
   - 计划通过实验进一步分析这种行为的影响。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **修复 Triton 和置信度函数中的 Bug**：成员报告了 `assert_verbose_allclose` 的 bug，促使 [Pull Request #261](https://github.com/linkedin/Liger-Kernel/pull/261) 进行修复，旨在增强其在多种场景下的可靠性。
   - 此外，还出现了关于 KL Divergence 计算在较大输入尺寸下产生意外结果的担忧，表明需要与交叉熵（cross-entropy）等既定函数保持一致。
- **黑客松团队与 CDP 签名会**：参与者正在为黑客松做准备，确认了审批情况，并鼓励利用论坛中发布的想法自行组建团队。
   - 值得注意的是，活动中有一个让 **Wen-mei Hwu** 教授在 **PMPP 书籍**上签名的机会，这增加了额外的参与感。
- **Web AI Summit 2024 社交机会**：即将举行的 [Web AI Summit 2024](https://rsvp.withgoogle.com/events/web-ai-summit-2024) 令人期待，成员们表达了参加并在 Web AI 话题周围进行社交的兴趣。
   - 该峰会为寻求分享该领域见解和经验的参与者提供了宝贵的交流机会。
- **关于 Apple ML 框架和 MLX 平台的见解**：Apple 特有的 ML 框架专注于 autodiff 和 JIT 编译等技术，以增强在 Apple silicon 上的性能，这与 PyTorch 的 kernel 开发方法有相似之处。
   - 成员们讨论了 **MLX**，这是一个类似 NumPy 的平台，专为 Metal 后端（Metal backends）的最佳性能而设计，增强了与 Apple 硬件能力的兼容性。
- **探索 Modal 的 Serverless 功能**：成员们寻求利用 **Modal** 获取免费 GPU 访问的信息，讨论了其不支持 SSH 的 Serverless 部署，但为新账户提供免费额度。
   - 建议探索一个 [GitHub 仓库](https://github.com/charlesfrye/cuda-modal) 中的示例，以便在 Modal 上无缝启动 CUDA 工作流。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 订阅困惑**：新用户对使用 [Perplexity Pro](https://discord.com/channels/1047197230748151888/1047649527299055688/1286404868285796448) 处理优化简历等任务表示困惑，建议 **ChatGPT** 可能会更有效。
   - 关于现有的 Pro 账户持有者在降级订阅后是否可以应用新的 Xfinity 奖励代码的讨论仍在继续。
- **o1 Mini 模型表现参差不齐**：用户提供了关于 **o1 Mini 模型**的反馈，报告结果褒贬不一，一些任务生成的响应较为基础且缺乏深度。
   - 在与 **Claude Sonnet 3.5** 进行对比时，用户强烈呼吁改进特定的 Prompting 技术以获得更好的结果。
- **AI 模型在编程中的多功能性**：几位用户强调了尝试使用最新的 AI 模型进行编程，将 **o1 Mini** 作为一个选项，但指出其在复杂项目中的局限性。
   - 他们强调了互联网搜索能力和实时反馈的必要性，以提升 AI 工具内的编程性能。
- **Sonar 与 Llama-3.1 的性能差异**：用户报告 **llama-3.1-sonar-large-128k-online** 模型表现不佳，特别是在响应格式方面，与 Web 端应用的结果相比存在差距。
   - 具体问题包括输出质量下降、过早截断以及在遵循 Prompt 指令方面的不一致。
- **获取 Beta 功能访问权限**：有关于 **return_citations** Beta 功能访问权限的咨询，建议联系 **api@perplexity.ai** 进行申请。
   - 用户要求澄清 **search_recency_filter** 是否处于封闭测试阶段，以及检索近期内容的潜力。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Pony XL 与原始模型的难题**：**Pony XL** 是其前身的一个精炼迭代版本，但引起了关于 **Text Encoder 层不匹配**以及与其他 Embeddings 混淆模型分类的担忧。
   - 一位用户恰当地将围绕 Pony 的热潮比作“郁金香狂热”，建议根据具体项目需求，**SDXL** 可能会表现得更好。
- **Flux 模型展示出强劲性能**：**Flux** 模型现在因克服了初始障碍（特别是资源需求和速度方面）而受到认可，从而在 **Prompt 遵循度**和图像质量方面建立了声誉。
   - 尽管有一些关于生成图像中**审美相似性**的反馈，社区仍对 Flux 实现顶级性能的能力保持期待。
- **SDXL 与旗帜：一个难点**：用户报告 **SDXL** 和 **SD3M** 在准确渲染国旗等常见符号时表现挣扎，对其可靠性提出质疑。
   - 社区建议包括专门训练一个 **Lora** 模型，旨在提高 SDXL 输出中旗帜的准确性。
- **优化 ComfyUI 以获得更好的工作流**：关于高效使用 **ComfyUI** 的讨论强调了云端工作流，并探索了如 **Backblaze** 等 Serverless 选项用于模型存储。
   - 成员们对在多个 GPU 上最大化利用 **VRAM** 表现出兴趣，并分享了在处理高负载任务时增强性能的技巧。
- **缺失 Inpainting 模型引发咨询**：一位用户对 **IOPaint** 中缺少 Inpainting 和擦除模型表示沮丧，需要通过命令行访问来解锁这些功能。
   - 这引发了关于命令行参数如何影响各种 UI 中模型可用性和操作的更广泛讨论。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **视频放大变得更简单**：一位成员建议使用 [video2x](https://github.com/k4yt3x/video2x?tab=readme-ov-file) 通过放大模型处理每一帧来放大视频。
   - 另一位成员考虑在放大前降低帧率以减少工作量，尽管对视频质量仍存疑虑。
- **猫咪 AI 革新音乐制作**：一位用户展示了他们的猫咪 AI 聊天机器人，用于音乐制作，能够生成 MIDI 文件并推荐合成方法。
   - 计划迁移到 Llama 以获得更好的性能，强调其对拍号和音乐风格的理解。
- **对 Forge 技术的兴趣增长**：成员们询问了 Forge 的功能，特别是它与 Hermes 和其他模型的关系。
   - 链接的 Discord 消息可能揭示了 Forge 在此背景下的能力。
- **探索 Hermes 3 的可访问性**：关于访问 Hermes 3 的讨论包括一个指向 [OpenRouter](https://openrouter.ai/) 的链接以供探索。
   - 参与者分享了关于 Hermes 3 性能和数据处理的看法。
- **关于 AI 意识的哲学思考**：一篇关于意识作为智能流形（intelligence manifolds）梯度的奇特论文被提及，引发了对其有效性的怀疑。
   - 围绕 AI 对音乐理论等复杂概念的理解程度展开了辩论。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Hyung Won Chung 的范式转移**：Hyung Won Chung 在 MIT 的演讲中介绍了 AI 的范式转移，强调最近推出的 [o1](https://x.com/hwchung27/status/1836842717302943774?s=46) 是该领域的重大进展。
   - 他表示，由于 AI 理解方面的重大进步，这次演讲正值关键时刻。
- **OpenAI 为 multi-agent 团队招聘 ML 工程师**：OpenAI 正在为一个新的 multi-agent 研究团队招聘 ML 工程师，认为这个细分领域对于增强 AI 推理至关重要；[申请详情点击此处](https://jobs.ashbyhq.com/openai/form/oai-multi-agent)。
   - 他们强调，先前在 multi-agent 系统方面的经验并非先决条件，鼓励更广泛的申请。
- **Devin 提升了速度和准确性**：**Devin** 最近的增强功能带来了**更快**、**更准确**的代码编辑以及改进的企业级安全支持 [来源](https://x.com/cognition_labs/status/1836866696797401118)。
   - 虽然许多用户对更新表示赞赏，但反馈褒贬不一，一些人对其局限性表示沮丧。
- **新的 RAG 提案减少了检索错误**：Anthropic 关于检索增强生成（RAG）的最新提案建议将错误分块检索率降低 **67%** [链接](https://www.anthropic.com/news/contextual-retrieval)。
   - 对话强调了人们对增强 RAG 有效性策略日益增长的兴趣。
- **关于 GitHub Copilot 模型的疑问**：用户提出了关于 GitHub Copilot 所使用模型标准的问题，推测其使用了 **GPT-4o**，并对性能一致性表示担忧 [来源](https://github.blog/changelog/2024-09-19-sign-up-for-openai-o1-access-on-github/)。
   - 讨论集中在上下文对各种 AI 工具性能的影响上。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **聊天室增强了用户交互**：聊天室现在支持**可编辑消息**，允许用户轻松修改自己的消息或机器人的回复。
   - 此次更新包括一个**重新设计**的统计界面，旨在提升用户参与度。
- **Qwen 2.5 树立了新标准**：**Qwen 2.5 72B** 提供了增强的代码和数学能力，具备 **131,072 上下文**大小。更多信息请点击[此处](https://openrouter.ai/models/qwen/qwen-2.5-72b-instruct)。
   - 该模型代表了 AI 能力的重大进步，推高了性能预期。
- **Mistral 进入多模态领域**：**Mistral Pixtral 12B** 标志着该公司在多模态 AI 领域的首次亮相，免费版本可在此处访问：[此处](https://openrouter.ai/models/mistralai/pixtral-12b)。
   - 此次发布被证明是一个关键时刻，将 Mistral 的产品扩展到了通用的 AI 应用中。
- **Hermes 3 转向付费结构**：随着 **Hermes 3** 转向 **$4.5/月**的付费结构，用户正在重新考虑服务使用选项。更多详情请点击[此处](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b)。
   - 价格变动缺乏通知，引发了社区内关于依赖免费额度的担忧。
- **自定义 API 集成受到关注**：有请求提出希望能够使用自定义的兼容 OpenAI 的 **API Key 端点**，以便更好地与私有 LLM 服务器集成。
   - 几位成员对这种灵活性对于未来集成能力的重要性表示赞同。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **令人兴奋的 Opik 合作伙伴关系，助力 RAG 自动日志记录**：LlamaIndex 宣布与 [Opik](https://t.co/Z3KdwjAKKv) 达成合作伙伴关系，将为开发和生产环境自动记录所有 RAG/Agent 调用，简化认证流程。
   - 这种自动化简化了复杂多步工作流中的用户体验。
- **RAGApp v0.1 发布：无代码多 Agent 应用**：团队发布了 [RAGApp v0.1](https://t.co/wyRNnnrmig)，支持在无需任何编码的情况下创建多 Agent 应用程序。
   - 用户可以轻松添加 Agent、分配角色、设置 Prompt，并利用各种工具进行应用增强。
- **LlamaIndex ID 给 Pinecone 带来麻烦**：由于 LlamaIndex 自动生成的 ID，用户在 Pinecone 中面临 ID 控制方面的挑战，导致删除操作变得复杂。
   - 社区建议通过手动编辑 ID 和创建节点来更好地管理这些限制。
- **Pandas Query Engine 表现异常**：Notebook 与 Python 脚本在使用 Pandas Query Engine 时的查询输出存在差异，影响了使用 `df.head()` 时的功能。
   - 将 `df.head()` 切换为 `df.head(1)` 证明可以解决该问题，这表明列数可能会影响查询解析。
- **Graph RAG 面临查询兼容性问题**：用户发现了 Graph RAG 中的查询模式问题，提供的模式与检索到的块 (chunks) 不一致。
   - 进一步分析显示，在数据获取过程中，GraphRAGQueryEngine 的预期存在不匹配。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **o1 Mini 与 4o 性能对决**：用户认为 **o1 mini** 不如 **4o**，理由是它缺乏现实世界的经验和智能推理，只是打字速度快但缺乏实质内容。
   - *一位用户评论说 o1 感觉与 4o 没有区别，* 引发了关于 AI 认知能力的讨论。
- **关于 AI 意识的热烈辩论**：一场关于 AI 是否能真正推理还是仅仅是模拟的激烈讨论爆发了，对于意向性持有不同意见。
   - 一位成员提出，让 AI 专注于任务完成，而不是类人推理，可能会产生更安全、更高效的结果。
- **澄清 GPT-4 记忆功能**：有关于 **GPT-4** API **Memory** 功能的咨询，明确了这些功能仅提供给 ChatGPT Plus 用户。
   - *一位用户指出，尽管 ChatGPT 界面缺乏此功能，但使用 Pinecone 等替代方案实现自己的记忆工具非常容易。*
- **IDE 集成反馈汇总**：有建议提出增强集成在 IDE 中的 AI 工具，特别是呼吁增加类似于 **ClaudeAI** 的实时预览功能。
   - *许多用户希望 ChatGPT 增加此功能，* 而其他人则建议探索各种 IDE 以获得更好的兼容性。
- **分享并改进 Prompt 使用**：一位成员分享了来自 [Prompt 指南](https://chatgpt.com/g/g-ssHSsvECZ-guidegpt) 的**有用 Prompt**，强调了其持续的相关性。
   - *视觉辅助工具被认为对增强 Prompt 理解很有价值，* 突显了它们在有效沟通想法中的作用。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **HyperCloning 加速模型初始化**：讨论集中在利用 **HyperCloning** 使用较小的预训练模型来初始化大型语言模型，旨在提高训练效率。
   - *一位成员建议*训练一个微型模型，对其进行缩放，并从较大的模型中进行蒸馏，以优化计算资源的使用。
- **IDA 在 AI Alignment 中获得关注**：用于对齐 AI 系统的 **Iterated Distillation and Amplification** (IDA) 方法因其迭代有效性而受到认可。
   - *一位参与者对“蒸馏”一词表示怀疑*，认为它无法代表所需的压缩和信息丢弃。
- **发现关键的 FP8 训练不稳定性**：据报告，由于在长时间训练运行中 SwiGLU 激活函数的离群值放大，**FP8 训练**存在不稳定性。
   - *一位听众询问*其他激活函数在扩展训练背景下是否会面临类似问题。
- **Tokenized SAEs 提升性能**：[Tokenized SAEs](https://www.lesswrong.com/posts/P8qLZco6Zq8LaLHe9/tokenized-saes-infusing-per-token-biases) 引入了每个 token 的解码器偏置，增强了 **GPT-2** 和 **Pythia** 等模型，促进了更快的训练。
   - *该方法解决了训练类别不平衡问题*，通过“unigram 重建”能够更好地学习局部上下文特征。
- **对 Gemma 模型中 BOS Token 的担忧**：有人担心 **Gemma 模型**中的 **BOS token** 在序列中可能只添加一次，从而影响滚动 **loglikelihood** 计算。
   - *同一位成员确认*，他们在调试期间发现某些情况下 **llh_rolling** 中缺失了 **BOS token**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **IPv4 切换解决连接问题**：一位成员指出，在 MacOS 上切换到 **IPv4** 通常可以解决连接问题，另一位成员确认*“它确实有效”*。
   - 有关调整 TCP/IP 设置的明确指南可以在[这里](https://support.apple.com/en-gb/guide/mac-help/mh14129/mac)找到。
- **攻克 LM Studio 连接挑战**：成员们在将 **LM Studio API** 连接到 **CrewAI** 时遇到了问题，探索了不同的解决方案，但未达成共识。
   - 有人建议观看一段有用的 [YouTube 视频](https://www.youtube.com/watch?v=fnchsJd9pfE)，以深入了解正确的设置方法。
- **3090 功率限制引发辩论**：一位成员分享了关于将 **3090** 限制在 **290W** 与降压（undervolting）的见解，并推荐了相关资源以供进一步了解。
   - 建议包括查阅文档，各方对每种方法的有效性持有不同意见。
- **Windows 与 Linux 电源管理**：对比显示，在 **Windows** 中调整 GPU 电源设置需要手动设置，而 **Linux** 用户可以通过单个命令进行优化。
   - 成员们辩论了跨系统电源管理的易用性，确认 Windows 提供了更快的设置调整。
- **RAM 速度与 CPU 推理瓶颈**：讨论围绕 **RAM 速度和带宽** 是否显著阻碍了 CPU 推理展开，并提出了使用 **DDR6** 主板的建议。
   - 成员们分享了对多个 **CPU 核心**利用不足的沮丧，强调了对当前 CPU 设计效率的担忧。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo LLMs API 引发关注**：一位用户表达了对使用基于 **Mojo 的 LLMs API** 以及 [Pythagora](https://www.pythagora.ai)（一种旨在通过对话交互构建应用程序的开发工具）的浓厚兴趣。
   - 他们提出了关于服务**成本**的问题，并强调了该服务在**软件开发转型**中所扮演角色的令人兴奋之处。
- **GitHub Discussions 关闭：请记好日期！**：**Mojo** 和 **MAX** 的 GitHub Discussions 将于 **9 月 26 日**正式关闭，重要的讨论将被转换为 GitHub Issues。
   - 为了确保有价值的讨论得以保留，随着社区集中化其沟通方式，成员可以标记组织者以请求进行转换。
- **Packed Structs：Mojo 的兼容性疑问**：聊天中强调了对 Mojo 缺乏 **packed structs** 支持的担忧，这使得将 bitflags 作为 `__mlir_type.i1` 列表处理变得复杂。
   - 尽管对其可靠性仍存疑虑，但人们希望 **LLVM** 能通过字节对齐（byte alignment）来解决这一问题。
- **对可变位宽整数的需求**：成员们讨论了为 TCP/IP 实现**可变位宽整数**（variable bit width integers）的问题，特别是对 UInt{1,2,3,4,6,7,13} 等类型的需求。
   - 虽然提出了*位运算符和掩码（bitwise operators and masks）*作为替代方案，但它们被认为不够易用（ergonomic），因此希望 **Mojo** 能提供原生支持。
- **Mojo 的功能请求堆积**：出现了一项功能请求，允许在没有类型参数的情况下使用**空列表**，以便更好地与 Mojo 中的 Python 兼容，此外还有其他语法咨询。
   - 显式 trait 实现的提及很常见，并有请求要求提供关于定义具有多个 trait 的 **generic struct** 的更清晰指南。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **输入位置系统得到简化**：最近的 [PR](https://github.com/pytorch/torchtune/pull/1449/files#diff-99a96ce497241e82b0c4d56f4bef3437e29dd596881b8f6d4db4d93178f88af5L227) 移除了 **input_pos** 的自动设置，以简化 **generation/cacheing logic**。
   - 此举旨在通过消除在各个类中寻找默认设置的过程，来防止用户产生困惑。
- **内存优化备受关注**：讨论强调了正在开发中的**内存优化**（如 **activation offloading**），并鼓励使用 **chunked cross entropy**。
   - 成员们承认，之前被否定的方法现在正被重新评估，以用于 **mem opt tutorial**。
- **通过批大小提升生成效率**：重点集中在**生成效率**上，强调 **generate script** 在执行期间仅支持 **bsz 1**。
   - 成员们在考虑提高批大小的缺点的同时，也在思考循环遍历批次的简单性。
- **关于生成过程子方法的辩论**：围绕引入 **sample** 和 **batched_sample** 等子方法展开了激烈辩论，旨在改进生成方法。
   - 观点各异，一些人倾向于方法分离，而另一些人则更喜欢类似于 **gpt-fast** 实践的精简方法。
- **保持 Generate Recipe 简洁的挑战**：一位成员对在用户报告的与较大批大小相关的问题中保持 **generate recipe** 的简单性表示紧迫感。
   - 目前正在努力简化逻辑，这被认为对于 **generate functionalities** 的清晰度至关重要。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Wizardlm 在任务表现上超越了 Llama**：实验表明 **microsoft/Wizardlm 8x22B** 的表现始终优于 **llama 405B**，在第一次尝试时成功完成任务的频率更高。
   - *成员们对 Wizardlm 在各种任务中的效率印象深刻*，引发了对其潜在更广泛应用的讨论。
- **O1 目前缺乏实用功能**：参与者指出 **O1** 仍处于开发阶段，目前没有任何可用于部署的功能。
   - 他们对它无法与应用程序交互表示担忧，强调了进一步增强的必要性。
- **提议讨论 O1 的功能**：有人呼吁针对 **O1** 的功能进行专门讨论，旨在澄清其潜在用例并收集见解。
   - 为了最大限度地提高参与度，鼓励成员分享他们的空闲时间，特别是 **GMT 时区**。
- **Firebase/Stripe 集成困难**：一位用户报告了他们的 **FarmFriend** 项目在集成 **Firebase** 和 **Stripe** 时遇到的持续问题，特别是在处理 CORS 和身份验证域名方面。
   - *他们描述在服务配置中遇到了“死循环”*，并寻求有此类集成维护经验的人员提供帮助。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **替换 CLANG dlopen 的悬赏**：关于用 **mmap** 替换 **CLANG dlopen** 的悬赏讨论出现了，这可能需要像[这个 pull request](https://github.com/tinygrad/tinygrad/pull/6299) 中所示的那样手动处理重定位。
   - *“此时我非常好奇谁会拿到这个悬赏”* 凸显了对该任务的竞争兴趣。
- **Tinygrad 与 Intel GPU 的兼容性**：一位用户询问 **Tinygrad** 是否支持多个 **Intel GPU**（类似于其对 **AMD** 和 **NVIDIA** GPU 的支持），并得到了积极反馈。
   - 建议是进一步调查兼容性，表明对 Intel 硬件的兴趣日益增长。
- **排查 IPMI 凭据问题**：报告的 **IPMI** 问题指向可能错误的凭据，引发了关于重置凭据最佳方法的讨论。
   - 建议包括使用显示器和键盘进行设置，并确保 **web BMC** 密码与显示的密码匹配。
- **对 GPU 设置连接的困惑**：关于在 GPU 设置中应使用 **HDMI** 还是 **VGA** 出现了一个问题，明确的共识是在初始连接期间**仅需使用 VGA**。
   - 这种困惑凸显了硬件配置实践中常见的疏忽。
- **关于 ShapeTrackers 可合并性的本科论文**：一位用户表示有兴趣在 Lean 中解决**两个任意 ShapeTrackers 的可合并性**问题，作为其本科论文，并询问了悬赏状态。
   - 他们注意到该任务似乎尚未完成，为项目的新贡献提供了机会。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Discord：学习之地**：成员们对加入 [Cohere Discord](https://discord.com/channels/954421988141711382/954421988783444043/1286401892078845954) 社区表示兴奋，鼓励营造一个学习 AI 和 Cohere 产品的协作氛围。
   - 向新人发送了 *“欢迎！”* 消息，为知识共享营造了积极的环境。
- **获取试用密钥并开始 Hack**：一位成员建议使用每月提供 **1000 次免费调用** 的试用密钥，强调通过项目进行实践学习。
   - 另一位成员表示赞同，称**应用是最好的学习方式**，并提到他们将在完成毕业设计后进一步探索。
- **Rerank Multilingual v3 在英语处理上的困难**：一位成员报告了使用 **rerank_multilingual_v3** 时的差异，指出英语查询的分数低于 **0.05**，而使用 **rerank_english_v3** 时分数为 **0.57** 和 **0.98**。
   - 这种不一致性正对他们的 **RAG 结果** 产生负面影响，导致相关分块被意外过滤。
- **使用 Curl 命令测试 Rerank 模型**：另一位成员建议使用 **curl** 命令切换模型进行测试，提议使用诸如 **'what are the working hours?'** 和 **'what are the opening times?'** 之类的查询。
   - 这可以实现模型之间更好的性能对比。
- **对新闻简报的兴趣**：一位成员提到他们是通过 **classify newsletter** 被吸引到社区的，展示了其在社区参与中的重要性。
   - 另一位成员表示希望看到更多新闻简报，表明对社区持续更新和信息的需求。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **寻求快速的 Whisper 解决方案**：一名成员请求帮助以最大化 **Whisper** 的速度，重点是针对**超大规模数据集**的转录任务使用多 GPU。
   - 高效处理至关重要，讨论中强调了对 batching（批处理）的需求。
- **Whisper-TPU：一个快速的选择**：**Whisper-TPU** 被强调为转录需求中一个显著的快速替代方案，迎合了需要高速处理的用户。
   - 它处理高要求任务的潜力引起了讨论者的兴趣。
- **探索 Transfusion 架构的使用**：人们对利用 **Transfusion** 架构进行多模态应用产生了好奇，暗示了其创新的能力。
   - [Transfusion 的 GitHub 仓库](https://github.com/lucidrains/transfusion-pytorch)展示了其预测 tokens 和扩散图像的潜力。
- **Diffusion 与 AR 训练的挑战**：结合 **diffusion** 和 **AR 训练** 的实验揭示了显著的稳定性挑战，凸显了一个关键的集成障碍。
   - 社区正在积极寻求有效的策略来增强这些训练方法的稳定性。
- **询问 Qwen-Audio 训练的不稳定性**：讨论中提到了 **Qwen-Audio** 论文中的训练不稳定性，并将其与多模态设置中的问题联系起来。
   - 成员们表示打算重新审视该论文，以明确这些挑战及其相关性。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Qwen 2.4 对比 o1-mini 令人失望**：新发布的 **qwen2.4-math-72b-instruct** 在使用代码执行和 **n=256** 生成的 ensemble 方法测试中，未能超越 **o1-mini**。
   - 这一结果凸显了在没有反思型 **CoT** 的情况下，很难在 AIME 等指标上实现公平比较。
- **欧盟暂停 Llama 多模态发布**：一位开发者提到，他们的团队热衷于创建一个 **多模态版本的 Llama**，但由于监管的不确定性，他们不会在欧盟发布。
   - 这一决定反映了人们对碎片化的技术监管可能扼杀欧洲 AI 创新的广泛担忧。
- **社区担忧欧盟的反技术立场**：围绕欧盟被感知的**反技术**情绪展开了讨论，成员们认为监管虽然初衷良好，但诱发了巨大的不确定性。
   - 呼吁制定更清晰的法规，以更好地平衡技术领域的创新与安全。
- **OpenAI 扩展视频的见解**：OpenAI 的扩展视频表明，与人类能力相比，带有 **RL** 的模型现在在发现 **CoT** 步骤方面更具优势。
   - 提出的关键点包括基础设施在算法性能中的重要性，以及 **self-critique**（自我批判）作为一项重大进展的出现。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain v2.0 中的分块输出问题**：用户报告称 **LangChain v2.0** 在使用 OpenAI streaming 时，会间歇性地以分块形式输出 **function call 信息**，这暗示可能存在 bug。
   - 这种情况引发了对函数调用期间配置设置和输出格式稳定性的担忧。
- **Ell 与 LangChain 的对比**：讨论强调了 **Ell** 和 **LangChain** 之间的区别和比较，显示了社区对评估 AI 框架可靠性的兴趣。
   - 参与者正在细致地检查框架，以确定当前项目有效的模型集成方案。
- **澄清 LangGraph 支持渠道**：关于去哪里咨询 **LangGraph** 问题的询问表明，社区对适当的支持渠道存在困惑。
   - 这表明需要为探索各种工具和库的用户提供更明确的支持途径。
- **新 Agent 平台 Beta 测试招募**：一份公告邀请 Beta 测试人员参与一个使用原生 tokens 启动 Agent 的**新平台**，预示着创新的机会。
   - 该平台旨在增强 Agent 的部署方法，引发了围绕集成策略的热议。
- **OpenAI Assistants 文档请求**：成员们请求根据最新文档使用其自定义 **OpenAI assistants** 的指导，展示了对 API 变化的适应。
   - 随着社区成员应对版本修订，理解新 **Assistants API** 功能的重要性被反复强调。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Moshi 模型发布引发关注**：[Moshi 模型](https://huggingface.co/kyutai/moshiko-pytorch-bf16) 已作为一款 **speech-text foundation model**（语音-文本基础模型）发布，它采用了一种全新的文本转语音方法，支持 **full-duplex spoken dialogue**（全双工语音对话）。
   - 这一进展显著增强了 **conversational dynamics**（对话动态）和 **speech recognition**（语音识别）能力。
- **GRIN MoE 以极少参数实现卓越性能**：[GRIN MoE](https://huggingface.co/microsoft/GRIN-MoE) 模型表现出色，仅凭 **6.6B active parameters**（激活参数）就实现了高性能，尤其在编程和数学任务中表现优异。
   - 该模型采用 **SparseMixer-v2** 进行梯度估计，通过规避标准的 **expert parallelism**（专家并行）方法挑战了技术极限。
- **Mistral Small 发布引发关注**：关于 **Mistral Small** 的讨论确认了它是一个 instruction-only（仅指令）版本，成员们对此反应不一。
   - 参与者指出 **memory intensity**（内存占用）问题是困扰多位用户的一个显著限制。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 中的 Bootstrapping 概念澄清**：一位成员澄清了 DSPy 中的 **bootstrapping** 用于在 pipeline 中生成中间示例，确保成功的预测能捕获完整的流程轨迹。
   - 强调了即使 LLM 具有非确定性，只要最终结果正确，中间步骤也应该是有效的。
- **MathPrompt 论文引起兴趣**：一位成员分享了一篇关于 **MathPrompt** 的研究论文，认为它有潜力扩展对增强数学推理的理解，并提供了[论文链接](https://arxiv.org/pdf/2409.11445)。
   - 这一参考资料可能为针对数学任务的更强大的 prompt engineering 策略铺平道路。
- **TypedPredictors JSON 技巧**：一位成员分享了 **TypedPredictors** 的新技巧，展示了如何模拟 JSON 解析来优化输出预处理，从而增强数据处理能力。
   - 该方法包括删除多余文本、处理无效转义序列以及记录来自其 [GitHub Gist](https://gist.github.com/tkellogg/246d7928b2fc26821db582be583d8b7a) 的错误。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **金融科技初创公司寻找 LLM 工程师**：一家金融科技初创公司正在寻找一名资深的 **LLM Engineer**，参与为期一周的冲刺，旨在利用 **LLama 3.1** 或 **Qwen2** 模型增强其多语言 **real-time translation service**（实时翻译服务）。
   - 该计划有望显著改善跨语言障碍处理数百万笔金融交易的方式。
- **Qwen 2.5 的多语言潜力**：一位参与者建议探索 **Qwen 2.5** 的 **multilingual capabilities**（多语言能力），认为它可能非常符合项目目标。
   - 这一建议指出了在选择 LLM 的同时增强 **Whisper model** 的方向，以进一步提高翻译准确性。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：各频道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1286418710332182529)** (1 条消息): 

> - `Tokenization 技术`
> - `Unity ML Agents`
> - `GSM8K 推理数据集`
> - `Nemotron-Mini-4B 演示`
> - `微调 Parler TTS` 


- **探索 Tokenization 技术**：一篇由认证用户发布的新博文 [This Title Is Already Tokenized](https://huggingface.co/blog/apehex/this-title-is-already-tokenized) 解释了先进的 Tokenization 方法。
   - 内容深入探讨了 Tokenization 的复杂性，同时展示了其在现代 NLP 中应用的潜力。
- **Unity ML Agents 预训练**：观看最新的 [YouTube 视频](https://youtube.com/live/0foHMTPWa4Y?feature=share)，了解如何使用 Unity ML Agents 和 Sentence Transformers 从头开始训练语言模型。
   - 本集重点介绍了 **Oproof 验证成功** 以及 Tau LLM 系列中的重要里程碑。
- **新 GSM8K 推理数据集发布**：一位贡献者引入了一个基于 GSM8K 的新 [推理数据集](https://huggingface.co/datasets/thesven/gsm8k-reasoning)，旨在用于 AI 模型训练。
   - 该数据集预计将增强 AI 系统的批判性推理能力。
- **Nemotron-Mini-4B 演示可用**：查看 Nemotron-Mini-4B 模型的 [演示](https://huggingface.co/spaces/Tonic/Nemotron-Mini-4B)，展示其各项功能。
   - 该演示旨在向 AI 从业者和研究人员展示该模型的实用性。
- **为特定语言微调 Parler TTS**：一篇详细的博文讨论了 [微调 Parler TTS](https://huggingface.co/blog/PHBJT/french-parler-tts) 以适配特定语言的过程。
   - 本指南提供了关于如何利用现有 TTS 模型服务于小众语言社区的见解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtube.com/live/0foHMTPWa4Y?feature=share)">Unity ML-Agents | 使用 Sentence Transformers 从头预训练 LLM | 第 20 部分</a>: **欢迎回到我们的 Tau LLM 系列！🌟**在本集中，我们很高兴能分享一些重大里程碑和新挑战：- **Oproof 验证成功**: ...</li><li><a href="https://medium.com/@visrow/ai-multi-agent-system-in-java-and-fipa-standards-f0a4d048c446)">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1286402118168875123)** (162 messages🔥🔥): 

> - `GPT models and unsupervised learning`
> - `Llava model quantization`
> - `Triplet loss explanation`
> - `AI tools support for Apple Silicon`
> - `Nanotron project discussion` 


- **GPT 模型简化了无监督学习**：GPT 模型以无监督方式运行，学习预测下一个 Token，而无需进行标注（例如 POS tags）。
   - 研究强调了从这些模型中提取语法的方法，展示了它们涵盖经典 NLP 特性的潜力。
- **了解 Llava 模型量化**：一位用户询问了如何量化 Llava 模型，以便在 12 GB 的 Nvidia GPU 上高效使用，并寻求最佳实践。
   - 建议集中在量化方法的指南上，并关注计算资源。
- **解释用于 Embeddings 的 Triplet loss**：Triplet loss 通过计算 Embeddings 之间的欧几里得距离来聚类相似样本，同时拉开差异样本的距离。
   - 图表中的视觉清晰度可能有助于更有效地传达 Anchor、Positive 和 Negative Embeddings 之间的关系。
- **Apple Silicon 与 AI 工具支持**：利用其 NPU 和统一内存（Unified RAM），Apple Silicon 上的 ML 工具支持具有提升潜力。
   - 讨论强调了这些工具的新兴发展及其在机器学习应用中的能力。
- **对 Nanotron 项目的热情**：Nanotron 项目获得了热烈反响，并提及了其与流行文化的融合。
   - 用户交流了兴奋的评论，表现出对该项目相关的游戏和创意应用的兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/cognitivecomputations/samantha-data">cognitivecomputations/samantha-data · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/settings/organizations>">Hugging Face – 建设未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=Cb13DAB59Po">在 VSCode 中快速轻松地设置 Google Cloud GPU VM 以进行深度学习（2024 指南）</a>: 在此视频中，我将向您展示如何快速设置 Google Cloud GPU 虚拟机，以便使用 Visual Studio Code (VSCode) 进行模型训练和深度学习。...</li><li><a href="https://arxiv.org/abs/1905.05950">BERT 重新发现经典 NLP 流水线</a>: 预训练文本编码器迅速推进了许多 NLP 任务的技术现状。我们专注于其中一个模型 BERT，旨在量化语言信息在网络中的捕获位置...</li><li><a href="https://gist.github.com/Getty/f5a6ebdea7de441215e4a8cd546f5cb8">gist:f5a6ebdea7de441215e4a8cd546f5cb8</a>: GitHub Gist: 立即分享代码、笔记和代码片段。</li><li><a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">让我们构建 GPT：从零开始，在代码中详细解析。</a>: 我们按照论文 "Attention is All You Need" 以及 OpenAI 的 GPT-2 / GPT-3 构建了一个 Generatively Pretrained Transformer (GPT)。我们讨论了其中的联系...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1286550468297883648)** (1 messages): 

> - `HF tutorials`
> - `Image creation guide`
> - `User collaboration` 


- **令人兴奋的新 HF 教程发布**：一名成员发布了他们的第一个教程，题为 **'Y01E01 - Make an image'**，旨在简化初学者的使用，可在此处查看 [here](https://huggingface.co/OFT/HF4Noobs/tree/main/Y01E01%20-%20Make%20an%20image)。
   - 他们提出了一个有用的建议，即重命名教程文件以提高清晰度，并请求更有经验的用户提供反馈以改进教程。
- **邀请用户贡献**：该成员对关于其教程的任何反馈或更正表示欢迎，鼓励社区投入以增强内容。
   - 他们表示愿意吸收资深用户的额外信息和改进建议。



**提及的链接**：<a href="https://huggingface.co/OFT/HF4Noobs/tree/main/Y01E01%20-%20Make%20an%20image">OFT/HF4Noobs at main</a>: 未找到描述

  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1286639230629449759)** (7 条消息): 

> - `FastAPI 中的 GLiNER 模型`
> - `自动 Notebook 生成器`
> - `Logo 生成模型`
> - `3D 内容生成框架`
> - `Stable Fast 3D` 


- **GitHub 上的 GLiNER 模型介绍**：一位用户在 GitHub 上分享了一个将 **GLiNER 模型** 实现为 FastAPI 微服务的项目，可以通过 [这里](https://github.com/henrikalbihn/gliner-as-a-service) 访问。
   - 对于那些对 AI 模型部署感兴趣的人来说，这可能是一个**非常酷**的工具。
- **自动 Notebook 生成器现已发布**：发现了一个自动 Notebook 生成器，可在 [此 Hugging Face Space](https://huggingface.co/spaces/asoria/auto-notebook-creator) 中使用。
   - 令人**耳目一新**，因为它可以简化 AI 项目的 Notebook 创建流程。
- **酷炫 Logo 生成模型发现**：分享了一个名为 **Huggieverse** 的新 Logo 生成模型，并在 [Hugging Face](https://huggingface.co/Chunte/flux-lora-Huggieverse) 上展示了各种 prompts。
   - 生成的图像包括**快乐的星星**和**柠檬**，展示了其在趣味品牌设计方面的潜力。
- **3D 内容生成框架揭晓**：一位用户指出了一个用于 **3D 内容生成** 的统一框架的 GitHub 仓库，详见 [这里](https://github.com/threestudio-project/threestudio/tree/main)。
   - 他们表达了对**快速 3D 对象生成**的需求，正在寻找支持保存为 PLY 格式的解决方案。
- **Stable Fast 3D 提供快速 3D 生成**：有人建议使用 **Stable Fast 3D**，它可以在 **不到一秒** 的时间内从图像生成 3D 对象。
   - 它拥有一个 Hugging Face Space，为用户的需求提供了一个高效的选择。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/asoria/auto-notebook-creator">Auto notebook creator - asoria 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/Chunte/flux-lora-Huggieverse">Chunte/flux-lora-Huggieverse · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/henrikalbihn/gliner-as-a-service">GitHub - henrikalbihn/gliner-as-a-service: FastAPI 微服务中的 GLiNER 模型。</a>：FastAPI 微服务中的 GLiNER 模型。通过在 GitHub 上创建账号来为 henrikalbihn/gliner-as-a-service 的开发做出贡献。</li><li><a href="https://github.com/threestudio-project/threestudio/tree/main">GitHub - threestudio-project/threestudio: 一个统一的 3D 内容生成框架。</a>：一个统一的 3D 内容生成框架。通过在 GitHub 上创建账号来为 threestudio-project/threestudio 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1286401465216143370)** (220 条消息🔥🔥): 

> - `Fractal Generator`
> - `Interactive World & Character Generative AI`
> - `Self-Supervised Learning Workshop at ECCV 2024`
> - `OCR Demos by PleIAs` 


- **Fractal Generator 的缩放功能**: 一位用户讨论了他们的 Fractal Generator 项目，重点介绍了通过“Aiming Better”部分更改网格长度并生成新输出来实现缩放的功能。
   - 另一位用户建议检测滚轮输入以实现更平滑的交互，并表示很喜欢测试该工具。
- **AI 平台的 Beta Testing**: 一个交互式 AI 平台正在寻找 Beta 测试人员，以探索角色和世界生成体验，邀请感兴趣的用户通过私信联系。
   - 该平台旨在结合用户生成内容和 AI 能力，创造沉浸式体验。
- **ECCV 2024 关于 Self-Supervised Learning 的 Workshop**: 分享了一篇文章，总结了即将举行的 ECCV 2024 Workshop 中多篇论文的技术，重点是提高 Self-Supervised Learning 中的数据效率和模型可解释性。
   - 该 Workshop 探讨了 Joint-embedding pre-training 过程中的 Representation collapse 等主题，强调了 Augmentations 的重要性。
- **来自 PleIAs 的 OCR Demo**: 一位用户分享了由 PleIAs 创建的 OCR Demo，据报道该 Demo 在 CPU 上运行，并已在 Hugging Face 上线。
   - 该 Demo 展示了 OCR 技术的实际应用，并激发了用户探索相关功能的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://dev.to/p3ngu1nzz/tau-llm-series-enhancements-and-debugging-part-18-19-n01">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Tonic/florence-pdf">Florence Pdf - 由 Tonic 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.lightly.ai/post/self-supervised-learning-at-eccv-2024">ECCV 2024 的 Self-Supervised Learning</a>: 这篇文章总结了 ECCV 2024 Workshop "Self-Supervised Learning: What is Next?" 的论文。它涵盖了提高数据效率、模型可解释性等多种方法...</li><li><a href="https://huggingface.co/p3nGu1nZz/Tau/tree/main/results/tau_agent_ppo_A3_2M/TauAgent">p3nGu1nZz/Tau at main</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Aryanne/Another_Fractal_Generator">Another Fractal Generator - 由 Aryanne 提供的 Hugging Face Space</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1286410950668193974)** (3 messages): 

> - `reCAPTCHAv2 100% 成功率`
> - `Qwen2-VL-72B-Instruct 介绍`
> - `模型创建者在 HF Hub 上的互动` 


- **reCAPTCHAv2 实现了 100% 的成功率**：一篇新论文揭示，**reCAPTCHAv2** 现在在破解验证码方面达到了 **100% 的成功率**，超过了之前约 **68% 到 71%** 的水平 [在此查看 PDF](https://arxiv.org/abs/2409.08831)。
   - 该研究使用先进的 YOLO 模型进行图像分割，并表明当前的 AI 技术可以有效地利用**基于图像的验证码 (image-based captchas)**。
- **Qwen2-VL-72B-Instruct 登场**：**Qwen2-VL-72B-Instruct** 刚刚发布，引入了**原生动态分辨率 (naive dynamic resolution)** 和**多模态旋转位置嵌入 (M-RoPE)**，以实现有效的信息融合 [阅读更多](https://arxiv.org/abs/2409.12191)。
   - 据开发者称，该模型现在可以处理长度超过 **20 分钟** 的视频，并具有增强的理解能力。
- **在 HF Hub 上与模型创建者互动**：建议直接在 **HF Hub 的 Community 标签页** 提问，因为模型创建者通常会监控该区域的咨询 [链接到集合](https://huggingface.co/collections/llava-hf/llava-onevision-66bb1e9ce8856e210a7ed1fe)。
   - 这一建议可能有助于用户与模型创建者之间进行更有效的沟通。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/llava-hf/llava-onevision-66bb1e9ce8856e210a7ed1fe">LLaVA-Onevision - 一个 llava-hf 集合</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2409.08831">Breaking reCAPTCHAv2</a>：我们的工作研究了使用先进机器学习方法破解 Google reCAPTCHAv2 系统验证码的功效。我们评估了自动化系统在破解验证码方面的有效性...</li><li><a href="https://arxiv.org/abs/2409.12191">Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution</a>：我们推出了 Qwen2-VL 系列，这是对之前 Qwen-VL 模型的先进升级，重新定义了视觉处理中传统的预定分辨率方法。Qwen2-VL 引入了 Naive...</li><li><a href="https://github.com/QwenLM/Qwen2-VL">GitHub - QwenLM/Qwen2-VL: Qwen2-VL 是由阿里巴巴云 Qwen 团队开发的多模态大语言模型系列。</a>：Qwen2-VL 是由阿里巴巴云 Qwen 团队开发的多模态大语言模型系列。 - QwenLM/Qwen2-VL
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1286478418485186582)** (2 messages): 

> - `GPT2SP 论文见解`
> - `使用 GPT 模型进行故事点估算`
> - `处理 Embedding 中的非标准语言` 


- **探索用于敏捷估算的 GPT2SP**：一位成员正在撰写硕士论文，旨在通过实验 **GPT-3** 和 **GPT-4** 等先进模型，基于 [GPT2SP 论文](https://link.to.paper) 的见解，改进敏捷开发方法中的**故事点估算 (story point estimation)**。
   - 他们正在寻求最合适模型的建议，以及来自任何有类似经验的人的见解。
- **语言序列中的奇特模式**：另一位成员观察到了非常规序列，如 *I'll* 的 **'ll'** 和 **'yes!do it'**，并提到了单词中缺少空格的问题。他们对这些情况在 **ST embedding pipeline** 中的处理方式，以及缺乏适应这些情况的现有 Embedding 模型表示担忧。
   - 他们强调了非标准语言模式带来的挑战及其对 Embedding 有效性的影响。


  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1286447792377561108)** (11 条消息🔥): 

> - `Fine-tuning Base Models`
> - `Best GPUs for Micro Datacenters`
> - `Liquid AI's Foundational Model`
> - `Mathematical Resources for Model Development`
> - `Diffusion Models Discussion Channel` 


- **使用 LoRA 进行 Base Model Fine-tuning**: 有建议提出使用 **LoRA** 以类似于 **ipadapter** 的方式对 **base model** 进行 **fine-tune**。
   - 这种方法旨在通过调整模型参数来增强性能，而无需进行大规模的重新训练。
- **小型集群配置的最佳 GPU**: 分享了针对小型集群的 **GPU 型号** 综合对比，重点关注了 **price, VRAM** 和 **TFLOPS** 等因素。
   - **NVIDIA RTX 4070** 位居效率榜首，与其他型号相比，以更低的单位 TFLOPS 成本提供了可观的性能。
- **Liquid AI 的基础模型资源**: 一位用户寻求关于 **Liquid AI** 新模型背后基础原理的资源，旨在利用现有的数学和 C++ 知识进行深入研究。
   - 推荐资源包括近期与 **LLMs** 相关的白皮书，以及用于实践落地的 **Unity ML Agents** 等资源。
- **模型构建的数学基础**: 讨论强调了在着手模型创建之前，扎实的数学背景的重要性。
   - 成员们分享了各种资源，以帮助那些希望加深对模型训练中涉及的复杂算法理解的人。
- **关于 Diffusion Models 频道的澄清**: 针对讨论 **Hugging Face Diffusion Models** 相关话题的正确频道进行了澄清。
   - 似乎有些用户误将该频道用于 **LLMs** 讨论，但它实际上是专门用于 Diffusion 相关的讨论。



**提及的链接**: <a href="https://github.com/huggingface/diffusion-models-class">GitHub - huggingface/diffusion-models-class: Materials for the Hugging Face Diffusion Models Course</a>: Hugging Face Diffusion Models 课程材料 - huggingface/diffusion-models-class

  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1286401893244862505)** (169 条消息🔥🔥): 

> - `Aider API interactions`
> - `O1 model performance`
> - `Using proxies with Aider`
> - `Sonnet's coding capabilities`
> - `Providing coding conventions` 


- **Aider API 交互问题**: 用户在使用 Aider 时遇到了无法读取 `.env` 文件的问题，导致配置困难以及 Anthropic API 出现过载错误。
   - 建议通过记录 LLM 的对话历史来帮助诊断这些问题。
- **O1 模型在 Aider 中的性能**: 对于 `O1` 模型与 Sonnet-3.5 相比的性能（特别是在编辑和代码重构任务中）存在怀疑态度。
   - 用户希望在集成更多功能后，未来能够改进 Aider 与 O1 模型之间的交互。
- **在 Aider 中使用代理**: 用户正在探索将代理设置与 Aider 集成的方法，包括为了方便连接而配置 Shadowsocks。
   - 在实现代理无缝工作方面成败参半，用户分享了各自的具体方法和挑战。
- **Sonnet 对大型函数的处理**: 有人担心 Sonnet 在只需要进行小改动时，倾向于尝试替换整个大型函数，从而导致错误。
   - 用户表示 LLMs 需要处理更小的代码块而不是大块代码，以减少代码编辑中的错误。
- **在 Aider 中加入编程规范**: 讨论了通过预索引文档为 Aider 提供相关编程指南作为潜在改进方案的可能性。
   - 用户认为这一功能可以增强 Aider 作为结对编程工具的有效性，确保其遵循特定的编码标准。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="http://127.0.0.1:1081")```">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/dani_avi">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/cerebras/chain-of-thought">Chain Of Thought - 由 cerebras 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://x.com/dani_avila7/status/1836760988982366533">来自 Daniel San (@dani_avila7) 的推文</a>: Cerebras + Llama 3.1：VSCode 中闪电般的代码助手 ⚡️ @CerebrasSystems 的服务速度极快，你几乎察觉不到它对代码所做的更改。在这个例子中，@AIatMeta Llama 3....</li><li><a href="https://aider.chat/docs/repomap.html">仓库映射 (Repository map)</a>: Aider 使用你的 git 仓库映射来为 LLM 提供代码上下文。</li><li><a href="https://arxiv.org/abs/2409.12186">Qwen2.5-Coder 技术报告</a>: 在本报告中，我们介绍了 Qwen2.5-Coder 系列，这是对其前身 CodeQwen1.5 的重大升级。该系列包括两个模型：Qwen2.5-Coder-1.5B 和 Qwen2.5-Coder-7B。作为一个专门针对代码的...</li><li><a href="https://aider.chat/docs/llms/anthropic.html">Anthropic</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/llms/warnings.html">模型警告</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/usage/conventions.html">指定编码规范</a>: 告诉 aider 在处理你的代码时遵循你的编码规范。</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>: 未找到描述</li><li><a href="https://aider.chat/docs/config/options.html#--llm-history-file-llm_history_file">选项参考</a>: 关于 aider 所有设置的详细信息。</li><li><a href="https://github.com/r-lib/tree-sitter-r/blob/main/queries/tags.scm">r-lib/tree-sitter-r 项目 main 分支下的 tree-sitter-r/queries/tags.scm</a>: 通过在 GitHub 上创建账号来为 r-lib/tree-sitter-r 的开发做出贡献。</li><li><a href="https://draftjs.org/docs/getting-started">概览 | Draft.js</a>: Draft.js 是一个用于在 React 中构建富文本编辑器的框架，由不可变模型驱动，并抽象了跨浏览器差异。</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/CONTRIBUTING.md#setting-up-a-development-environment)">paul-gauthier/aider 项目 main 分支下的 aider/CONTRIBUTING.md</a>: aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/tests/basic/test_io.py">paul-gauthier/aider 项目 main 分支下的 aider/tests/basic/test_io.py</a>: aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/fry69/files-to-prompt-ts">GitHub - fry69/files-to-prompt-ts：一个以结构化方式将文件和目录连接成单个提示词的命令行工具，适用于大语言模型和其他应用。</a>: 一个以结构化方式将文件和目录连接成单个提示词的命令行工具，适用于大语言模型和其他应用。 - fry69/files-to-prompt-ts</li><li><a href="https://github.com/simonw/files-to-prompt">GitHub - simonw/files-to-prompt：将包含文件的目录连接成单个提示词，供 LLM 使用</a>: 将包含文件的目录连接成单个提示词，供 LLM 使用 - simonw/files-to-prompt</li><li><a href="https://cursor.directory/">Cursor 目录</a>: 为你的框架和语言寻找最佳的 Cursor 规则</li><li><a href="https://aider.chat/docs/llms">连接到 LLM</a>: Aider 可以连接到大多数 LLM 进行 AI 结对编程。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1286465520182231090)** (54 条消息🔥): 

> - `Aider URL 抓取`
> - `Aider 中的文件管理`
> - `Aider 的测试驱动开发方法`
> - `模型配置问题`
> - `函数重命名错误` 


- **停止 Aider 抓取 URL**：用户对 Aider 询问他们粘贴的 URL 表示沮丧，除非他们明确使用了 `/web`。
   - 为了缓解这一问题，有人建议在配置中明确偏好，但尚未提供最终解决方案。
- **使用 Aider 管理文件**：有人指出，新文件需要手动添加才能被 Aider 识别，而对现有文件的修改无需提交即可被察觉。
   - 为了获得最佳功能，建议用户在 VSCode 外部运行 Aider，以最大化其交互式终端能力。
- **Aider 的测试驱动开发 (TDD) 实践**：几位用户讨论了在测试驱动开发 (TDD) 环境中使用 Aider，更倾向于手动编辑测试文件。
   - 建议包括将文件设为只读，以便 Aider 专注于实现生产代码而不修改测试。
- **模型配置和 Token 限制问题**：一位用户报告在使用 Bedrock/Anthropic Claude 3.5 时遇到 Token 限制问题，响应在中途被截断。
   - 尝试通过 JSON 文件调整模型设置未达到预期效果，引发了对配置的进一步调查。
- **Aider 中的函数重命名和 Linter 错误**：Aider 尝试重命名函数导致所有代码引用中的更新不完整，从而引发未定义函数错误。
   - 尽管被提示解决 Linter 错误，Aider 仅修复了一个实例，这表明其搜索/替换功能存在局限性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: 使用 chat, ask 和 help 聊天模式。</li><li><a href="https://www.youtube.com/watch?v=QlUt06XLbJE">SECRET SAUCE of AI Coding? AI Devlog with Aider, Cursor, Bun and Notion</a>: 高产出 AI 编程的秘诀是什么？🔗 更多关于 AIDER 的 AI 编程：https://youtu.be/ag-KxYS8Vuw🚀 更多关于 Cursor 的 AI 编程：https://youtu.be/V9_Rzj...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1286528820509278259)** (13 条消息🔥): 

> - `Anthropic 的上下文检索 (Contextual Retrieval)`
> - `Cerebras 的思维链 (Chain of Thought)`
> - `RAG 的挑战与解决方案`
> - `Google CTR 助推机器人`
> - `AI 开发平台竞争` 


- **Anthropic 发布上下文检索方法**：Anthropic 推出了一种[上下文检索方法](https://www.anthropic.com/news/contextual-retrieval)，可增强 Prompt 缓存，这对于高效的 LLM 运行至关重要。
   - 一位用户指出，实施该方法应是 Aider 等项目的首要任务，因为它普遍适用于所有 LLM。
- **思维链引发疑问**：一位成员质疑 [思维链 (Chain of Thought) 方法](https://huggingface.co/spaces/cerebras/chain-of-thought) 是否真的能提高性能，暗示早期更依赖于训练。
   - 回复强调结果的复杂性需要针对预期应用进行特定调整。
- **RAG 需要仔细调优才能生效**：围绕 RAG 的讨论表明，它最初看似简单实则具有误导性，因为获得最佳结果非常具有挑战性。
   - 一位成员强调了集成全文搜索和重排序 (re-ranking) 过程以获得理想结果的重要性。
- **开源 Google CTR 助推机器人发布**：一位用户分享了他们的 [Google CTR 助推机器人](https://github.com/alextobias78/Google-CTR-Bot)，该机器人使用 Javascript 编写，用于模拟人类行为以提高点击率。
   - 他们指出 Aider 协助构建了约 **25-33%** 的代码，使开发过程非常愉快。
- **AI 平台竞争：Anthropic vs OpenAI**：成员们讨论了 **Anthropic** 和 **OpenAI** 作为领先 AI 开发平台之间日益激烈的竞争。
   - 随着不断进步，在这种竞争格局中对更简单工具的需求变得更加突出。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/cerebras/chain-of-thought">Chain Of Thought - a Hugging Face Space by cerebras</a>: 暂无描述</li><li><a href="https://github.com/alextobias78/Google-CTR-Bot">GitHub - alextobias78/Google-CTR-Bot: Google CTR bot - Use it to simulate click-through for your websites.</a>: Google CTR 机器人 - 用于模拟网站的点击。 - alextobias78/Google-CTR-Bot
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1286401962379841537)** (198 条消息🔥🔥): 

> - `Qwen 2.5 支持`
> - `模型微调`
> - `模型性能对比`
> - `量化技术`
> - `生产力工具` 


- **Unsloth 已支持 Qwen 2.5**：成员们确认 Unsloth 已支持 **Qwen 2.5**，尽管聊天模板（chat templates）存在一些 Bug，Qwen 团队正在处理中。
   - 一位用户强调了支持的重要性，提到：*“最好是支持了，我正在训练 Qwen2.5-14b”*。
- **模型微调与数据量**：建议在针对文本分类微调 LLM 时，**Qwen 2.5** 和 **BERT** 是合适的选择，尤其是在数据集有限的情况下。
   - 一位用户指出，他们使用 **Llama 8.1 8B** 达到了 **71% 的准确率**，并表示希望进一步提高这一分数。
- **量化关注点**：关于量化的讨论表明，使用未量化的模型可以提供更好的速度和吞吐量，特别是在批处理（batch processing）场景下。
   - 成员们强调了在决定是否对模型进行量化时，需要在速度、体积和成本之间进行权衡。
- **生产力工具讨论**：一位用户分享了 **MetaDock** 的链接，这是一款支持分屏和多账号登录功能的生产力工具；而其他用户则提到了用于窗口管理的 **FancyZones**。
   - 有人提出了 *“有没有平铺式窗口管理器（tiled window manager）？”* 的疑问，引发了对免费替代方案的讨论。
- **模型兼容性与性能**：用户讨论了不同模型之间的性能差异，特别注意到 **Qwen 2.5** 在某些任务上可能比 **Llama 3.1 8B** 更快。
   - 对话表明，计算环境和数据集大小等外部因素会显著影响模型性能和训练效果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://learn.microsoft.com/en-us/windows/powertoys/fancyzones">适用于 Windows 的 PowerToys FancyZones 工具</a>：一款用于将窗口排列并吸附到高效布局中的窗口管理器工具。</li><li><a href="https://www.metadock.net/">MetaDock 官网</a>：告别频繁的窗口切换。MetaDock 通过其独特的分屏和多布局系统，让你无缝管理多个任务。立即尝试！</li><li><a href="https://docs.wandb.ai/guides/integrations/huggingface/">Hugging Face Transformers | Weights &amp; Biases 文档</a>：Hugging Face Transformers 库使 BERT 等最先进的 NLP 模型以及混合精度和梯度检查点等训练技术变得易于使用。W&amp;B 集成增加了丰富的……
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1286467838932156416)** (7 条消息): 

> - `AGI 进展`
> - `AI 中的帕累托法则`
> - `查询与数据库限制`
> - `L3.1 托管的经济性` 


- **关于 AGI 临近度的见解**：在研究复杂话题时，成员们表达了对我们距离 **AGI** 尚远的担忧，强调这不仅关乎获取答案，更关乎如何有效地解释答案。
   - 这一认识强调了 **80/20 法则**，暗示通往 AGI 的最后一步极其耗时，而我们已经投入了 **60 年** 的努力。
- **理解 AI 智能**：一位成员对 AI 现有的智能表示怀疑，称其仅仅是对有损数据库的**高级查询**，而非真正的智力。
   - 他们进一步阐述，目前的 AI 能力本质上是**曲线拟合（curve fitting）**，并不反映真正的理解。
- **托管 L3.1 的成本效益分析**：一位成员分享了关于使用 **L3.1 70b** 托管 **50 万份文档**（平均 **7k tokens**）的经济性担忧，在是用 **RunPod 搭配 vLLM** 还是按 Token 向 API 提供商付费之间进行权衡。
   - 他们寻求关于哪种方案对于大规模数据分析需求更经济的见解。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1286403991001104577)** (11 条消息🔥): 

> - `phi-3.5-mini 微调问题`
> - `TGI 预量化权重支持`
> - `结合聊天历史使用 model.generate`
> - `LoRa 权重加载崩溃`
> - `训练中的预测损失评估` 


- **phi-3.5-mini 微调问题的变通方法**：一名用户报告在尝试微调 **phi-3.5-mini** 并将最大长度设置为超过 4095 时，遇到了与 **LongRopeRotaryEmbedding** 相关的 *AttributeError*，详见 [此 issue](https://github.com/unslothai/unsloth/issues/946)。尽管尝试了建议的解决方案，该问题似乎依然存在，促使社区寻求额外的变通方法。
- **关于 TGI 是否支持预量化权重的疑问**：有人提出了关于 **TGI** 是否支持加载 **pre-quantized weights** 的查询，这表明可能存在兼容性问题。目前尚未提供明确的答复来澄清此功能。
- **结合对话历史实现 model.generate**：为了结合对话历史使用 **model.generate**，一位用户建议使用传递给 tokenizer 的结构来格式化聊天，并引用了 [Hugging Face 文档](https://huggingface.co/docs/transformers/main/en/chat_templating)。然而，另一位用户对他们的 prompt 与该方法的兼容性表示困惑。
- **加载 LoRa 权重时崩溃**：一名成员在使用 **unsloth FastLanguageModel** 加载 **LoRa weights** 时遇到崩溃，而使用 PeftModel 进行推理则正常。该问题被怀疑与可能缺失的依赖项有关，例如 **flash-attn**。
- **理解 prediction_loss_only 参数**：关于在训练循环中使用 **prediction_loss_only = True** 进行评估以减少 VRAM 占用的讨论，成员们寻求对其确切功能的澄清。特别是，有人提出了关于此设置是否仅影响评估阶段的问题，并指出已经在使用 **DataCollatorForCompletionOnlyLM** 来限制损失计算。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Chat Templates</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/946">AttributeError: &#39;LongRopeRotaryEmbedding&#39; object has no attribute &#39;inv_freq&#39; when finetuning Phi3.5 mini · Issue #946 · unslothai/unsloth</a>: 你好，我在微调 Phi3.5 时遇到了标题中的错误。我相信我使用的是最新版本的 unsloth（通过 pip 从 git 安装）。背景：使用已在其他 unsloth 模型上运行的代码微调 Phi3.5...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1286616401326506025)** (3 messages): 

> - `Contacting Authors` (联系作者)
> - `BART Model Behavior` (BART 模型行为)
> - `Torchtune Activation Offloading` (Torchtune 激活卸载)
> - `Memory Consumption Techniques` (内存消耗优化技术)
> - `W&B Charts` (W&B 图表)


- **理解 BART 的输入机制**：一位成员询问了 **input_ids** 和 **labels** 的输入格式，注意到 BART 模型是从 **EOS token** 而不是传统的 **BOS token** 开始生成的。
   - 这种行为被认为很奇怪，但计划通过实验进一步调查其影响。
- **探索 Torchtune 的激活卸载 (Activation Offloading)**：分享了一个指向 [Torchtune GitHub repository](https://github.com/pytorch/torchtune/blob/9f329a261fce1935b40029914e38ee31d952c50a/torchtune/training/_activation_offloading.py#L4) 的链接，重点介绍了其在激活卸载方面的功能。
   - 该功能是用于 LLM 微调的原生 PyTorch 库的一部分，非常值得进一步探索。
- **来自 PyTorch 大会的见解**：分享了一条关于 **Jane Xu** 在 PyTorch 大会上演讲的推文，讨论了减少模型训练中 **memory consumption**（内存消耗）的技术。
   - 提到的技术包括 **OffloadActivations**、**Activation Checkpointing** 以及各种优化器，如 **AdamW4bit**。
- **参与 W&B 图表**：该成员注意到演讲中展示的 **Weights & Biases** (W&B) 图表非常直观，有助于追踪这些新的省显存技术。
   - 这些图表在理解实验过程中的性能指标方面起着至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/9f329a261fce1935b40029914e38ee31d952c50a/torchtune/training/_activation_offloading.py#L4">torchtune/torchtune/training/_activation_offloading.py at 9f329a261fce1935b40029914e38ee31d952c50a · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。可以通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://x.com/morgymcg/status/1837140988457779587">来自 Morgan McGuire (Hack @ W&B Sep 21/22) (@morgymcg) 的推文</a>：非常喜欢 Jane Xu 在 PyTorch 大会上关于逐步减少内存消耗的演讲 ⬇️OffloadActivations (torchtune 中的新功能) ⬇️Activation Checkpointing ⬇️AdamW4bit / Adafacto...
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1286404847742091364)** (6 messages): 

> - `Nvidia's Triton Inference Server` (Nvidia 的 Triton 推理服务器)
> - `Google Cloud GPU VM Setup` (Google Cloud GPU 虚拟机设置)
> - `BaM Reproduction Systems` (BaM 复现系统)
> - `Peer-to-Peer GPU Communication` (GPU 点对点通信)


- **关于 Triton Inference Server 的澄清**：一位用户澄清说他们指的是 **Nvidia's Triton Inference Server**，而不是 OpenAI 的版本。
   - 另一位成员提到他们之前在一个小项目中使用过它，展示了其在实际中的应用。
- **关于 BaM 复现系统的讨论**：一位用户询问了针对 **BaM** 项目的低预算系统建议，特别是关于 **Gigabyte G292-Z22** 在点对点 (P2P) 通信方面的能力。
   - 他们寻求确认该系统是否同时支持 GPU-to-GPU 和 GPU-to-NVMe 的连接。
- **Google Cloud 设置的新视频指南**：一位成员分享了一个 [YouTube video](https://www.youtube.com/watch?v=Cb13DAB59Po) 指南，详细介绍了如何为深度学习设置 **Google Cloud GPU VM instance**，包括通过 SSH 在 VSCode 中安装 PyTorch 的说明。
   - 他们提到发现设置过程非常繁琐，旨在帮助社区中从事模型训练的其他成员。


  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1286545358029062144)** (9 messages🔥): 

> - `GroupNorm 实现`
> - `性能挑战`
> - `内存优化策略`
> - `Triton kernel 调整`
> - `YouTube 录像查询` 


- **GroupNorm 实现进展**：一位成员正在 Triton 中实现 **GroupNorm**，并已成功运行前向/反向传播，但在某些张量尺寸（如 **512 x 64 x 64**，**num_groups=32**）上遇到了性能下降。
   - 他们推测这种下降是由于 **T4 GPU 的内存带宽**限制，并正在寻求提升性能的建议。
- **性能优化见解**：另一位成员建议，该实现需要读取两次输入 **X**：一次用于计算统计数据，另一次用于归一化，这对于较大的张量至关重要。
   - 他们指出，使用较大的 **BLOCK_SIZE** 可能会降低占用率（occupancy）；对于像这样受内存限制（memory-bound）的 kernel，256 通常是更好的值。
- **通过减小 BLOCK_SIZE 提高效率**：最初的成员承认降低 **BLOCK_SIZE** 可以提高性能，并引用了在 **4096** 和 **16384** 尺寸下的成功基准测试。
   - 他们计划在重写 kernel 时考虑这些优化，特别是针对大张量的场景。
- **大张量面临的挑战**：对话强调，较大的空间维度（如 **128x128** 或 **512x512**）会导致无法将整个 group 加载到 SRAM 中，从而使均值/标准差计算变得复杂。
   - 该成员描述了他们当前的实现，即使用 for 循环计算统计数据，但在**内存带宽**方面遇到了瓶颈。
- **YouTube 录像查询**：一位成员提到 YouTube 上应该有与讨论相关的录像，但尚未向组织者确认。
   - 他们表示打算在第二天联系组织者，以确认录像是否可用。



**提到的链接**：<a href="https://colab.research.google.com/drive/1jbBmYi0QulrsQMMe2kRh2LkM71RKelTw?usp=sharing">Google Colab</a>：未找到描述

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607595451895918/1286415494202789898)** (3 messages): 

> - `模型优化演讲`
> - `Flash Attention 实现` 


- **寻求推荐的模型优化演讲**：一位成员询问有关*模型优化演讲*的信息，回忆起之前的讨论并请求相关链接或建议。
   - 这促使其他人考虑成员们过去的贡献，例如 Driss 与此主题相关的多个 PR。
- **Flash Attention 实现见解**：另一位成员提到了来自 **cuDNN** 的 **Flash Attention 实现**，强调了其相比 **FA3** 的速度优势。
   - 他们表示，这是在持续优化模型性能工作中的一个值得关注的进展。


  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1286580140578701416)** (2 messages): 

> - `Llama2-7B 训练`
> - `FP8 精度`
> - `SwiGLU 激活问题`
> - `优化技术` 


- **英特尔使用 FP8 精度训练 Llama2-7B**：英特尔成功在 **2 万亿 token** 的数据集上使用 **FP8 精度**训练了 **Llama2-7B** 模型，比之前的限制增加了 **20 倍**。论文讨论了 FP8 训练中的不稳定性，追溯到 **SwiGLU** 激活函数对离群值的放大，并引入了 **Smooth-SwiGLU** 修改以确保稳定性。
   - 这种转变不仅影响激活稳定性，还允许对 Adam 优化器的两个动量进行 **FP8 量化**，展示了对模型参数更好的管理。
- **通过随机投影平滑激活**：一位成员提议探索**随机投影**技术来平滑激活并降低维度，建议如果激活足够稀疏，这可以减轻离群值问题。他们注意到在 **Quip#** 和 **Quarot** 等现有模型中使用了 **Hadamard 变换**来实现类似目的。
   - 这引发了关于降维方法如何影响大型语言模型（LLM）的稳定性和性能的讨论。



**提到的链接**：<a href="https://arxiv.org/abs/2409.12517">Scaling FP8 training to trillion-token LLMs</a>：我们首次在高达 2 万亿 token 的数据集上使用 FP8 精度训练大型语言模型——比之前的限制增加了 20 倍。通过这些扩展的训练运行，我们发现……

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

iron_bound: https://wunkolo.github.io/post/2024/09/gpu-debug-scopes/
  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1286473895217598506)** (4 messages): 

> - `Yudkowski's Rationality`
> - `Nous Research Merch` 


- **每日深入阅读 Yudkowski 的《Rationality: From AI to Zombies》**：一位成员决定每天为 **Yudkowski 的《Rationality: From AI to Zombies》** 中的一个章节做笔记。
   - 他们对快速阅读的实用性表示怀疑，认为这需要深思熟虑的参与。
- **Nous Research 周边发布好评**：成员们热烈讨论了最近的 **Nous Research 周边发布**，重点介绍了他们收到的单品：**Decentralize T 恤**。
   - 然而，一位成员指出 **90 美元的连帽衫**相当昂贵，尽管价格不菲，但仍表达了购买欲望。


  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1286427885548212335)** (3 messages): 

> - `Latent Space server`
> - `San Francisco meeting spots` 


- **Latent Space 服务器见解**：一位成员分享了在 Latent Space 服务器中发现的标题为 [San Francisco meeting spots](https://www.alessiofanelli.com/posts/san-francisco-meeting-spots) 的帖子链接。
   - 该帖子可能对规划未来在 **San Francisco** 的聚会有所帮助。
- **讨论计划尚不明确**：一位成员询问了另一位成员的未来计划，表现出对潜在讨论的兴趣。
   - 另一位成员回复称，目前按现状来看**没有计划**。


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1286407901027762206)** (17 messages🔥): 

> - `L2 Side Aware optimization`
> - `Stochastic rounding hack`
> - `CI support for Llama 3`
> - `Travel fatigue`
> - `Friend requests on Discord` 


- **探索 L2 Side Aware 优化**：一位成员计划撰写一篇关于 **L2 Side Aware 优化**和 **NVIDIA 内存层级**的文章，倾向于在 WiFi 改善后再进行。
   - 他们指出，在当前条件下通过 SSH 进行开发会非常痛苦。
- **酷炫的随机舍入 (Stochastic rounding) 技巧想法**：一位成员介绍了一种 **stochastic rounding** 技巧，通过强制将某些尾数位 (mantissa bits) 置零来节省功耗，称其为一个非常酷的想法。
   - 这种技巧可能为计算提供一种更高效的方法。
- **协作 LLM.c 想法**：讨论了一个关于编写极其模糊的 **'GH200 和/或 Llama3.1 上的 llm.c'** 大规模黑客想法，一些成员表示处理起来不太自在。
   - 他们邀请小组中其他对该任务感到更自在的人参与贡献。
- **感谢 CI 团队对活动的支持**：一位成员对 **AAA+E** 团队在即将举行的活动中的参与表示感谢，同时承认他们目前在芝加哥。
   - 他们感谢成员们的协作，并提出将来会提供 **Llama 3** 的 CI 更新。
- **旅行疲劳再次袭来**：几位成员分享了他们的旅行经历，其中一位表示虽然只睡了 7 小时，但在长距离散步后感觉良好。
   - 讨论还包括关于旅行中管理睡眠挑战的轻松评论。


  

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1286458638927527946)** (9 messages🔥): 

> - `LLM 压缩方法`
> - `乘积量化 (Product Quantization) 技术`
> - `BitNet 训练实现`
> - `量化效率`
> - `内存优化策略` 


- **关于 LLM 压缩策略的新见解**：最近的讨论强调了**无数据压缩 (data-free compression)** 方法的有效性，该方法可实现 **50-60%** 的稀疏度，从而在保持困惑度 (perplexity) 等性能指标的同时，减少 LLM 的内存占用。
   - 引入了*知识密集型压缩 LLM 基准测试 (LLM-KICK)* 以重新定义评估协议，使其与稠密模型对应版本保持高度一致。
- **关于乘积量化方法的辩论**：针对乘积量化方法在实现与 BitNet 类似的压缩率方面的表现提出了疑虑，对于 **2-bit 和 3-bit** 模型的有效性意见不一。
   - 一位成员指出，**微调 2-bit 模型**可以产生积极的结果，特别是使用 *LoRA* 或 *hqq+ 风格* 的方法。
- **量化方法效率受到质疑**：一位成员批评了当前量化方法的**效率**，指出在 **H100** 上处理 **Llamav2-70B 模型**可能需要 **3 到 11 小时**，表明性能缓慢。
   - 另一位参与者幽默地评论道，这个时间范围表明人们对效率的含义存在误解。
- **BitNet 训练调整取得进展**：报告了 BitNet 训练的进展，提到了 **int8 混合精度**训练的集成以及量化方法的更改，这预示着潜在的改进。
   - 一次小型调试运行显示出良好的前景，表明对**前向传播量化 (forward pass quantization)** 的改进方法可能会带来更好的速度和性能。
- **探索内存优化技术**：讨论包括利用来自 **gemlite 的 A8W2** 来降低量化策略中的内存占用，同时探索其相对于 **A8W8** 对速度的影响。
   - 成员们分享了关于平衡内存优化与处理速度的见解，建议这些调整可以专门针对资源效率进行优化。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2310.01382">Compressing LLMs: The Truth is Rarely Pure and Never Simple</a>：尽管取得了显著成就，现代大语言模型 (LLMs) 仍面临极高的计算和内存占用。最近，多项工作展示了在免训练 (training-free) 方面取得的显著成功...</li><li><a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf at master · microsoft/unilm</a>：跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1286734523127693417)** (2 messages): 

> - `Web AI Summit 2024` 


- **令人兴奋的 Web AI Summit 2024 公告**：一位成员分享了 [Web AI Summit 2024](https://rsvp.withgoogle.com/events/web-ai-summit-2024) 的链接，表达了参加该活动的兴趣。
   - 他们邀请其他人加入，并建议在峰会期间见面交流。
- **即将在 Web AI Summit 举行的聚会**：一位成员宣布了他们参加 Web AI Summit 的计划，并鼓励其他人碰面。
   - 该邀请暗示将创造一个围绕 Web AI 进行社交和讨论的机会。



**提到的链接**：<a href="https://rsvp.withgoogle.com/events/web-ai-summit-2024">Web AI Summit 2024</a>：未找到描述

  

---

### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1286420954511183924)** (32 条消息🔥): 

> - `Hackathon 邀请`
> - `寻找队伍`
> - `PMPP 书籍签名`
> - `Hackathon 项目创意`
> - `停车选项` 


- **Hackathon 邀请正在发放中**：<@sr2132> 确认 **QQ** 和 **Ishan** 都已通过 Hackathon 审核，邀请函应在明天前送达。
   - 如果参与者尚未收到邀请，建议分享其队友的详细信息。
- **Hackathon 自行组队**：<@marksaroufim> 建议参与者通过查看论坛中发布的创意，或在活动前与各环节负责人（session leads）沟通来自行组队。
   - 此流程旨在简化组队过程，并最大限度地增加 Hackathon 期间的编码时间。
- **PMPP 书籍签名机会**：与会者在 **CUDA-MODE IRL** 活动期间，将有难得的机会获得作者 Wen-mei Hwu 教授在 **PMPP 书籍**上的亲笔签名。
   - 提醒参与者携带书籍以便签名。
- **鼓励为 Hackathon 贡献项目创意**：<@andreaskoepf> 鼓励与会者思考潜在的开发项目，并查看论坛上的现有创意。
   - 他强调，小团队理想情况下应由 **2-4** 名成员组成，但也欢迎个人项目。
- **活动停车建议**：建议参与者考虑搭乘 Uber 前往 **CUDA-MODE IRL** 活动，而不是自行驾车。
   - <@marksaroufim> 提示停车可能不是最佳选择，另一位成员也询问了可用车位的情况。


---

### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1286617357925482538)** (89 条消息🔥🔥): 

> - `assert_verbose_allclose 错误`
> - `KL 散度问题`
> - `RMSNorm 修复`
> - `Triton Kernel 限制` 


- **assert_verbose_allclose 中的错误**：多位成员发现了与 `assert_verbose_allclose` 函数相关的错误，以及在处理某些输入时的不正确行为。
   - 通过 [Pull Request #261](https://github.com/linkedin/Liger-Kernel/pull/261) 分享了一个拟议的修复方案，旨在解决不同场景下的这些错误。
- **KL 散度计算之谜**：有人对 KL 散度 (`kldiv`) 计算产生意外结果表示担忧，特别是当输入大小超过 **16384** 时。
   - 观察到标准实现与 `LigerKLDIVLoss` 函数之间的结果存在显著差异，引发了关于潜在修复方案的讨论。
- **RMSNorm 相关调整**：成员们讨论了 RMSNorm 的实现，确认其工作正常，但指出存在细微的数值稳定性问题。
   - 提出了一项详细分析，涉及错误的权重访问以及旨在提高效率的潜在调整。
- **Triton Kernel 限制与改进**：讨论强调了 Triton 在处理较大 Kernel 大小时（特别是超过 **64kb** 限制）不发生崩溃的局限性。
   - 成员们建议根据 Triton 教程的见解更改 Grid 大小以优化性能，并参考了 `cross_entropy` 实现中采用的方法。
- **调试建议**：成员们建议使用 `torch.allclose` 进行测试，而不是自定义断言，以便更好地调试 Loss 函数。
   - 达成共识认为，将 KL 散度的计算方式与 Cross Entropy 中使用的计算方式对齐可以解决这些差异。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gist.github.com/Tcc0403/67aa7f8eaf536ae63f21f83405298047">kldiv_bug.py</a>: GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/261">Fix assert_verbose_allclose bugs by Tcc0403 · Pull Request #261 · linkedin/Liger-Kernel</a>: 摘要：修复 #259。WIP：我还没有检查修复是否有效。先开启一个 PR 进行测试。测试已完成，硬件类型：运行 make test 以确保正确性，运行 make checkstyle 以确保代码规范...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/255">RMSNorm aggregation by Tcc0403 · Pull Request #255 · linkedin/Liger-Kernel</a>: 摘要：解决 #179。WIP：解决较大 hidden_size (4096) 的数值稳定性问题。测试已完成，硬件类型：RTX-3080，运行 make test 以确保正确性，运行 make checkstyle 以确保...</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/layer_norm.py#L189">Liger-Kernel/src/liger_kernel/ops/layer_norm.py at main · linkedin/Liger-Kernel</a>: 用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/259">test.utils.assert_verbose_allclose has multiple bugs · Issue #259 · linkedin/Liger-Kernel</a>: 🐛 错误描述：当 diff 为 nan 时返回 False。Liger-Kernel/test/utils.py 第 59 行 ce71d59，当 num_mismatched 为 1 时，mismatched = diff > tolerance 条件为 False。Liger-Kernel/test/utils.py 第 7...</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/ce71d59b0b0894f9f3e7512f5a3bf3780c5a1499/src/liger_kernel/ops/kl_div.py#L121">Liger-Kernel/src/liger_kernel/ops/kl_div.py at ce71d59b0b0894f9f3e7512f5a3bf3780c5a1499 · linkedin/Liger-Kernel</a>: 用于 LLM 训练的高效 Triton Kernels。</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/ce71d59b0b0894f9f3e7512f5a3bf3780c5a1499/src/liger_kernel/ops/cross_entropy.py#L205">Liger-Kernel/src/liger_kernel/ops/cross_entropy.py at ce71d59b0b0894f9f3e7512f5a3bf3780c5a1499 · linkedin/Liger-Kernel</a>: 用于 LLM 训练的高效 Triton Kernels。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[irl-sponsor-qa](https://discord.com/channels/1189498204333543425/1285287931828768940/1286509707636248680)** (9 messages🔥): 

> - `Modal`
> - `PrimeIntellect`
> - `Lambda Cloud`
> - `CUDA workflows` 


- **探索 Modal 和免费 GPU 访问**：一位成员正在调研 **Modal, PrimeIntellect** 和 **Lambda Cloud**，希望在今晚和明天准备周末活动时能**免费访问 1 台 H100 机器**。
   - *这是否有可能实现？*
- **对 Modal 的 Serverless 功能的疑问**：有人认为 **Modal** 作为一个“Serverless”且无忧的部署服务运行，但它可能不提供直接的 **SSH** 访问。
   - 尽管如此，**Modal** 在创建账户时会提供一些**免费额度 (free credits)**。
- **在 Modal 上开始使用 CUDA**：建议查看[这个 GitHub 仓库](https://github.com/charlesfrye/cuda-modal)中的**示例代码**，旨在 **Modal** 上启动 **CUDA workflows**。
   - 仓库中的 **Jupyter Lab 示例**被强调为一个非常有用的界面，其中包含 shell 访问权限。
- **在 Modal 上启动 VSCode**：提出了另一种使用 **Modal** 的方法，通过命令 `modal launch vscode --gpu h100`，允许用户挂载 `--volume` 或 `--mount` 来保存他们的工作。
   - 该建议旨在简化使用 **Modal** 时的开发工作流。
- **关于 Modal 使用体验的反馈**：在收到社区提供的有用命令后，初始用户表达了感谢并表示：“Tysm! Trying this now!”，他们正在测试 **Modal** 的各项功能。
   - 另一位成员鼓励反馈使用体验，并提到他们并不经常使用该工作流。



**提到的链接**：<a href="https://github.com/charlesfrye/cuda-modal">GitHub - charlesfrye/cuda-modal: Enter CUDA MODE on Modal</a>：在 Modal 上进入 CUDA MODE。通过在 GitHub 上创建账户来为 charlesfrye/cuda-modal 的开发做出贡献。

  

---


### **CUDA MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1286414595820359751)** (4 messages): 

> - `Apple ML Framework`
> - `MLX Platform`
> - `Metal Backends` 


- **Apple 的专用 ML 框架**：一位成员指出，该框架是专为 **Apple 计算机**设计的，采用了 **autodiff**, **vmap** 和 **JIT** 编译等特定技术，以在 **Apple silicon** 上获得更好的性能。
   - 这种方法由于其定制的 kernel 开发，与 **PyTorch** 更为相似，而不是 **Triton**。
- **介绍 MLX：一个类 NumPy 平台**：另一位成员分享道，**MLX** 是一个从头开始构建的**类 NumPy** 平台，表明其专注于优化性能。
   - 他们强调了其独特的结构，将其与传统库区分开来。
- **MLX 中的 Metal 后端**：讨论强调了 MLX 包含其自身的 Metal 后端，特别是 **'steel'**，并利用了 **Metal Performance Shaders (MPS)**。
   - 这种集成有助于增强性能并充分利用 **Apple 硬件**的能力。
- **惰性求值 (Lazy Evaluation) 提升性能**：指出 MLX 以**惰性 (lazily)** 方式处理操作，仅在特定调用时进行求值，从而提高了整体性能。
   - 这种技术减少了不必要的计算，优化了资源利用。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1286404868285796448)** (164 条消息🔥🔥): 

> - `Perplexity Pro 订阅问题`
> - `o1 Mini 模型性能`
> - `使用 AI 模型进行编程`
> - `AI 模型的 Prompting 技巧`
> - `Perplexity 中的 Pro Search 功能` 


- **Perplexity Pro 的困扰**：新用户对使用 Perplexity Pro 处理简历定制等任务表示困惑，认为 ChatGPT 等其他工具可能更有效。
   - 关于现有的 Pro 账户持有者在降级订阅后是否可以应用新的 Xfinity 奖励代码，目前正在进行讨论。
- **o1 Mini 模型的评价褒贬不一**：用户报告了在 Perplexity 中使用 o1 Mini 模型的不同体验，指出虽然它能很好地处理某些任务，但其他任务的回答较为基础，缺乏推理深度。
   - 用户将 o1 Mini 的性能与 Claude Sonnet 3.5 等模型进行了对比，认为其表现稍逊一筹，并强调了特定 Prompting 技巧的必要性。
- **用于编程的 AI 模型**：几位用户讨论了使用最新的 AI 模型进行编程任务，强调 o1 Mini 是一个潜在工具，但也指出了它在更复杂项目中的局限性。
   - 互联网搜索能力和实时反馈机制的整合被视为增强 AI 模型编程性能的关键。
- **有效的 AI Prompting 技巧**：讨论围绕 Prompting AI 模型的核心最佳实践展开，用户建议更简单、更清晰的 Prompt 会带来更好的结果。
   - 强调了测试和审查输出的重要性，这是更好地了解模型能力过程的一部分。
- **AI 的未来与 Neuralink**：讨论涉及了 Neuralink 等进展的潜在影响，暗示了一个拥有更智能 AI 和增强人类能力的未来。
   - 关于 AI 的现状以及创造可能超越人类智能的人工智能所带来的伦理影响，出现了截然不同的观点。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://chromewebstore.google.com/detail/complexity/ffppmilmeaekegkpckebkeahjgmhggpj)">Chrome 网上应用店</a>：为您的浏览器添加新功能并个性化您的浏览体验。</li><li><a href="https://x.com/apostraphi/status/1837219719495176299?s=61">来自 Phi Hoang (@apostraphi) 的推文</a>：说实话...我们没预料到会有这么多学生加入 Perplexity 的返校季活动！欢迎大家，我们才刚刚开始。</li><li><a href="https://chromewebstore.google.com/detail/complexity/ffppmilmeaekegkpckebkeahj">Chrome 网上应用店</a>：为您的浏览器添加新功能并个性化您的浏览体验。</li><li><a href="https://tenor.com/view/holo-spice-and-wolf-holo-the-wise-wolf-horo-korbo-gif-13009516793083034180">Holo Spice And Wolf GIF - 贤狼赫萝 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-huge-128k-online">Llama 3.1 Sonar 405B Online - API、提供商、统计数据</a>：Llama 3.1 Sonar 是 Perplexity 最新的模型系列。通过 API 运行 Llama 3.1 Sonar 405B Online。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1286497325761691669)** (14 条消息🔥): 

> - `AI 益处`
> - `以色列-黎巴嫩紧张局势`
> - `Linkin Park 的遗产`
> - `烹饪 Chicken Keraguen`
> - `AI 创新者简介` 


- **探索 AI 益处**：分享的一个链接讨论了 [AI 的益处](https://www.perplexity.ai/search/what-are-the-benefits-of-ai-ag-k9RzcVpSgqRT7JdzAc6qA)，提供了关于 AI 对各行业积极影响的见解。
   - 关键点包括效率提升和增强的决策能力。
- **当前以色列-黎巴嫩紧张局势**：围绕 [以色列和黎巴嫩之间持续的紧张局势](https://www.perplexity.ai/search/israel-lebanon-tensions-curren-hrOSTOE2R9KNBWl643CQbQ) 展开的讨论，使用了 CPLX 0.3.0 版本的最新见解。
   - 对话指出了历史背景以及对地区稳定的当前影响。
- **引人入胜的 Linkin Park 遗产讨论**：深入探讨 [Linkin Park 的遗产](https://www.perplexity.ai/page/linkin-park-s-legacy-gEPo7lN1SGmRcjaf43oaWQ) 揭示了该乐队的文化影响力和持续的影响。
   - 成员们分享了对乐队音乐的个人感悟，面对情感叙事及其与粉丝的共鸣。
- **Chicken Keraguen 烹饪指南**：关于 [如何烹饪 Chicken Keraguen](https://www.perplexity.ai/search/how-to-cook-a-chicken-keraguen-odoyUzlwTfmBNvETq4KRDg) 的链接提供了分步食谱和技巧。
   - 这道菜源于丰富的文化传统，提供了充满风味的烹饪体验。
- **拆解 AI 创新者的见解**：通过 [创新者](https://www.perplexity.ai/page/ai-innovator-the-enigmatic-str-JwemnZS0TGuYd2WLcXj8FA) 的经验观察 AI 的塑造过程，展示了想法如何演变为具有影响力的工具。
   - 参与者讨论了创新者在快速变化的技术格局中所面临的挑战。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1286408460220764233)** (10 条消息🔥): 

> - `Perplexity API 的变更`
> - `Sonar 与 Llama-3.1 模型性能对比`
> - `Beta 功能访问权限`
> - `搜索时效性过滤器 (Search Recency Filter)`
> - `API 输出限制` 


- **Perplexity API 网站更新**：Perplexity API 网站上的 [模型页面](https://docs.perplexity.ai/guides/model-cards) 已更新，不再使用过时的 "pplx" 模型，现在统称为 **Sonar**。
   - *Pplx 模型已过时且不再受支持*，因此鼓励用户切换到 Sonar。
- **Sonar 与 Web 应用相比的性能问题**：一位用户报告称，与 Perplexity Web 应用程序相比，**llama-3.1-sonar-large-128k-online** 模型的结果明显较差，特别是在响应格式化方面。
   - 他们指出了输出质量的具体问题，包括提前停止以及难以遵循特定的 Prompt 指令。
- **获取引用的 Beta 功能权限**：一位用户询问如何获取 API 的 **return_citations** Beta 功能访问权限，这需要特定的申请流程。
   - 建议他们通过电子邮件 **api@perplexity.ai** 联系以获取进一步的帮助和指导。
- **了解 search_recency_filter 的可用性**：有一个关于 **search_recency_filter** 功能是属于封闭 Beta 测试还是对所有用户开放的问题。
   - 如果设置得当，该功能可以帮助收集过去一小时内发布的来源中的最新信息。
- **改进输出质量的建议流程**：一位用户建议通过缓存链接并在多次运行中格式化文本的多步流程来提高输出质量，以确保准确的引用。
   - 他们表示有兴趣了解为什么 Web 应用程序的表现似乎优于 API 模型，以及使用 GPT 是否可以增强该过程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://perplexity.typeform.com/apiaccessform">API 表单 </a>：很高兴听到您充分利用了 Perplexity API！使用此表单申请访问额外功能或更高的速率限制。   </li><li><a href="https://docs.perplexity.ai/guides/model-cards>">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1286407278257377281)** (132 messages🔥🔥): 

> - `Pony 和 XL 模型对比`
> - `Flux 模型能力`
> - `SDXL 与旗帜生成的问题`
> - `高效使用 ComfyUI`
> - `Inpainting 与擦除模型` 


- **Pony 和 XL 模型非常相似**：**Pony XL** 本质上是原始模型的精炼版本，但人们对其与其他 **embeddings** 的**文本编码器层不一致 (text encoder layer disalignment)** 表示担忧，这导致了关于其是否应被分类为独立模型的困惑。
   - 一位用户将围绕 Pony 的炒作比作“郁金香狂热”，并建议根据具体需求，**SDXL** 中有更好的选择。
- **Flux 模型的性能亮点**：据报道，**Flux** 已经克服了包括资源需求和速度在内的初始障碍，在**提示词遵循度 (prompt adherence)** 和图像质量方面确立了其作为领先模型的地位。
   - 尽管社区对生成图像的**审美相似性 (aesthetic similarities)** 有一些抱怨，但用户对 Flux 保持模型性能优势的潜力持乐观态度。
- **SDXL 和旗帜生成的挑战**：用户注意到 **SDXL** 和 **SD3M** 在有效渲染国家旗帜和常见符号方面都存在困难，这引发了对模型在这一领域能力的质疑。
   - 一位用户建议专门为旗帜训练一个 **Lora** 模型，以增强 SDXL 在生成准确描述方面的效能。
- **高效使用 ComfyUI**：讨论了如何优化云端的 **ComfyUI** 工作流，建议使用 **serverless** 平台或探索像 **Backblaze** 这样的模型存储选项。
   - 用户对在多个 **GPUs** 上最大化 **VRAM** 利用率表现出兴趣，寻求优化工作负载性能和效率的建议。
- **Inpainting 和擦除模型的缺失**：一位用户提出了关于 **IOPaint** 中缺失 **inpainting** 和擦除模型的问题，发现需要通过命令行选项才能访问这些功能。
   - 这引发了关于命令行参数如何影响各种 **UI** 设置中特定模型和功能可用性的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/nyanko7/flux-dev-de-distill">nyanko7/flux-dev-de-distill · Hugging Face</a>: 未找到描述</li><li><a href="https://civitai.com/images/25279078">49RpK5dY 发布的图片</a>: 未找到描述
</li>
</ul>

</div>

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1286419881410953321)** (73 条消息🔥🔥): 

> - `使用 AI 进行视频超分辨率 (Upscaling)`
> - `音乐制作聊天机器人`
> - `Forge 技术`
> - `Hermes 3`
> - `AI 中的意识` 


- **视频超分辨率变得更简单**：一位成员建议使用 [video2x](https://github.com/k4yt3x/video2x?tab=readme-ov-file)，通过超分辨率模型处理每一帧来提升视频分辨率。
   - 另一位成员考虑在超分辨率处理前降低帧率以减少工作量，尽管他们对最终的视频质量表示不确定。
- **猫咪 AI 革新音乐制作**：一位用户讨论了他们创建的一个专注于音乐制作的猫咪形象 AI 聊天机器人，它可以生成 MIDI 文件并推荐合成方法。
   - 尽管目前存在局限性，他们计划将其迁移到 Llama 以提升性能，并强调了它对拍号和音乐风格的理解。
- **对 Forge 技术的兴趣日益增长**：一位成员询问了 Forge 的功能及其与 Hermes 和 World Sim 等其他模型的关系。
   - 另一位成员分享了一个 Discord 消息链接，该链接可能提供了关于 Forge 能力和特性的见解。
- **探索 Hermes 3 的可用性**：成员们讨论了在哪里可以尝试 Hermes 3，并提供了 [OpenRouter](https://openrouter.ai/) 的链接供大家探索。
   - 对话中包含了对 Hermes 3 整体性能以及处理即时信息能力的看法。
- **关于 AI 意识的哲学思考**：一位用户提到了一篇奇特的论文，该论文将意识描述为智能流形 (intelligence manifolds) 中的梯度，并对其有效性表示怀疑。
   - 这引发了关于 AI 对音乐理论等概念的理解，以及模型以更复杂方式进行训练的潜力的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tome.app/k4don/f-cm188fgq10fhn7xgq8bz93udc">Tome</a>：未找到描述</li><li><a href="https://news.lambdalabs.com/news/today">ML Times</a>：未找到描述</li><li><a href="https://tenor.com/view/tldr-gif-25251690">Tldr GIF - Tldr - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://openrouter.ai/">OpenRouter</a>：LLM 路由与市场</li><li><a href="https://github.com/k4yt3x/video2x?tab=readme-ov-file">GitHub - k4yt3x/video2x: A lossless video/GIF/image upscaler achieved with waifu2x, Anime4K, SRMD and RealSR. Started in Hack the Valley II, 2018.</a>：一款通过 waifu2x, Anime4K, SRMD 和 RealSR 实现的无损视频/GIF/图片超分辨率工具。始于 2018 年 Hack the Valley II。- k4yt3x/video2x
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1286411413257977898)** (29 messages🔥): 

> - `Hermes-3 功能`
> - `用于海事法聊天机器人的 RAG`
> - `在 RAG 中使用规则`
> - `Together AI 的性价比` 


- **寻求 Hermes-3 模型 Function Arguments 方面的帮助**：一位用户在他们的仓库（特别是 [functions.py](https://github.com/NousResearch/Hermes-Function-Calling/tree/main)）中向 Hermes-3 模型添加具有多种参数类型的函数时寻求帮助。另一位用户建议转回通过 Langchain 使用 OpenAI 的工具，因为某些设计决策可能会限制参数类型。
   - 他们被引荐给另一位成员，该成员可能会在有空时提供进一步帮助。
- **推荐将 RAG 用于海事法聊天机器人**：讨论集中在利用 **Retrieval Augmented Generation (RAG)** 构建针对海运和法律的聊天机器人，因为 Fine-tuning 的效果可能较差。成员们表示，通过 RAG 引用特定的法律文件对于避免 Hallucination 至关重要。
   - 分享了一个丰富的资源链接，重点介绍了如何使用 Langchain 实现 RAG 以有效处理问答应用。
- **探索用于基于规则的应用的 RAG**：一位用户询问如何将 RAG 适配于多人文字游戏中的基于规则的操作，强调了在不同条件下检索适用规则的挑战。他们思考了规则中的通用术语如何影响实际场景中的规则检索和应用。
   - 另一位成员解释了如何创建一个与 Google Sheets 接口的 API Server 来高效管理规则，并建议将规则管理与针对特定 Prompt 的模型 Fine-tuning 相结合。
- **使用 Together AI 的高性价比 AI 模型**：一位用户注意到使用 **Together AI** 模型的经济性，并考虑将其应用于 RAG 功能中。大家普遍认为，由于低成本和高效性能，使用 Together AI 构建金融应用具有潜在优势。
- **在不同对话中设置规则**：讨论包括了关于在不同对话中实现一致指令的担忧，特别是在 MUD 的背景下。一位用户评论说，目前的模型可能难以持久地处理复杂的、带有 Context 的指令。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/docs/tutorials/rag/">构建检索增强生成 (RAG) 应用 | 🦜️🔗 LangChain</a>：LLM 实现的最强大应用之一是复杂的问答 (Q&amp;A) 聊天机器人。这些应用可以针对特定的源信息回答问题。这些...</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/tree/main">GitHub - NousResearch/Hermes-Function-Calling</a>：通过创建账号为 NousResearch/Hermes-Function-Calling 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1286423949219074068)** (3 messages): 

> - `ReST-MCTS 论文`
> - `Iteration of Thought 框架` 


- **ReST-MCTS 论文受到关注**：一位参与者注意到 **ReST-MCTS** 论文（结合了 **STaR**、**PRM** 和 **MCTS**）尽管无缝集成了多种方法论，但似乎未得到应有的重视。
   - 这篇论文可能会引发关于其与其他作品重叠概念的深刻讨论。
- **提出新的 Iteration of Thought 框架**：**Iteration of Thought (IoT)** 框架旨在通过动态 Prompt 生成来增强 **LLM** 的响应，这与 **CoT** 或 **ToT** 等传统方法不同。
   - 它包含一个 **Inner Dialogue Agent (IDA)** 来创建特定于 Context 的 Prompt，并根据不断变化的对话 Context 调整推理。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2409.12917">通过强化学习训练语言模型进行自我修正</a>：自我修正是大语言模型 (LLMs) 非常理想的能力，但在现代 LLM 中一直被发现很大程度上是无效的。现有的训练自我修正的方法...</li><li><a href="https://arxiv.org/abs/2409.12618">Iteration of Thought: 利用内部对话进行自主大语言模型推理</a>：迭代的人类参与是利用大语言模型 (LLMs) 先进语言处理能力的一种常见且有效的方式。在对话方式中使用结构良好的 Prompt...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1286713807011446825)** (2 messages): 

> - `Promptriever`
> - `Twitter Insights` 


- **Promptriever：稠密检索（Dense Retrieval）的新纪元**：介绍了 [Promptriever](https://github.com/orionw/promptriever)，这是**首个可以像语言模型一样进行提示（prompted）的稠密检索模型**，为信息检索提供了全新的视角。
   - 它能够被提示的能力带来了一种**魅力与好奇**的结合，引发了关于它是“诅咒”还是“美丽”的讨论。
- **神秘的 Twitter 见解**：来自 [@unknown](https://vxtwitter.com/reach_vb/status/1836432149018288157) 的一条推文引发了好奇，但未定义具体背景，以其模糊的吸引力吸引了社区。
   - 参与者们思考了这条推文的含义，反映出它在没有明确解释的情况下激发兴趣和神秘感的能力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://vxtwitter.com/reach_vb/status/1836432149018288157">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://github.com/orionw/promptriever">GitHub - orionw/promptriever: The first dense retrieval model that can be prompted like an LM</a>：首个可以像 LM 一样进行提示的稠密检索模型 - orionw/promptriever
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1286423949219074068)** (3 messages): 

> - `ReST-MCTS`
> - `Iteration of Thought framework`
> - `Large Language Models engagement` 


- **ReST-MCTS 论文引起关注**：一位成员强调 **ReST-MCTS 论文**被低估了，并提到它有效地融合了 **STaR**、**PRM** 和 **MCTS** 方法论。
   - 他们指出，该论文以一种**无缝的方式**结合了这些技术，并敦促其他人探索其中的见解。
- **引入 Iteration of Thought 框架**：提出了一种名为 **Iteration of Thought (IoT)** 的新框架，旨在通过自适应对话方法增强 **LLM 响应**。
   - 该框架由三个组件组成，包括一个用于生成特定上下文提示的 **Inner Dialogue Agent**，旨在提高 LLM 交互的思考深度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2409.12618">Iteration of Thought: Leveraging Inner Dialogue for Autonomous Large Language Model Reasoning</a>：迭代式的人类参与是利用大语言模型（LLMs）先进语言处理能力的常用且有效手段。通过以对话方式使用结构良好的提示...</li><li><a href="https://arxiv.org/abs/2409.12917">Training Language Models to Self-Correct via Reinforcement Learning</a>：自我修正是大语言模型（LLMs）一种非常理想的能力，但在现代 LLMs 中一直被发现基本无效。现有的训练自我修正的方法...
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1286407382796075129)** (38 messages🔥): 

> - `Hyung Won Chung's MIT Talk`
> - `OpenAI Hiring for Multi-Agent Research`
> - `Advancements in Devin`
> - `Improvement Techniques in RAG`
> - `GitHub Copilot Updates`

- **Hyung Won Chung 展示了新范式**：Hyung Won Chung 分享了他在 MIT 的演讲，讨论了范式转移以及最近发布的 [o1](https://x.com/hwchung27/status/1836842717302943774?s=46)，他认为这体现了该领域的一个新范式。
   - 他强调了这次演讲在理解 AI 重大进展方面的及时性。
- **OpenAI 为 Multi-Agent 研究招募 ML Engineers**：OpenAI 目前正在为一个新的 Multi-Agent 研究团队招聘 ML Engineers，认为这一领域对于增强 AI 推理至关重要 [详情点击这里](https://jobs.ashbyhq.com/openai/form/oai-multi-agent)。
   - 他们指出，不要求具备 Multi-Agent 系统的先前经验，鼓励感兴趣的候选人申请。
- **Devin 展示了改进和社区反馈**：最近的更新显示，Devin 现在在代码编辑方面更快、更准确，并改进了对企业安全要求的支持 [来源](https://x.com/cognition_labs/status/1836866696797401118)。
   - 然而，用户的反馈从对其能力的沮丧到对其 Demo 表现的赞赏不等。
- **RAG 应用的新技术**：Anthropic 关于 Contextual Retrieval 的一项新提案建议，将错误块检索率降低多达 **67%** [链接](https://www.anthropic.com/news/contextual-retrieval)。
   - 参与者注意到，旨在改进 Retrieval-Augmented Generation (RAG) 的策略及其有效性正在增加。
- **GitHub Copilot 的模型混淆**：用户对 GitHub Copilot 中使用的模型标准进行了推测，声称它采用了 GPT-4o，引发了关于性能的问题 [来源](https://github.blog/changelog/2024-09-19-sign-up-for-openai-o1-access-on-github/)。
   - 对话强调了理解 Context 的重要性，以及性能在不同 AI 工具之间的差异。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/DimitrisPapail/status/1835791517316747725">来自 Dimitris Papailiopoulos (@DimitrisPapail) 的推文</a>：嗯，o1-preview 可以解决具有负权重的 100 个顶点图的最短路径问题。你需要 Bellman-Ford 算法来完成这个任务（即动态规划）。在 GPT-4 还在……的时候，它能做到这一点真是太疯狂了。</li><li><a href="https://x.com/polynoamial/status/1836872735668195636?s=61">来自 Noam Brown (@polynoamial) 的推文</a>：.@OpenAI 正在为新的 multi-agent 研究团队招聘 ML 工程师！我们将 multi-agent 视为实现更好 AI 推理的路径。不需要之前的 multi-agent 经验。如果你想研究……</li><li><a href="https://x.com/alexalbert__/status/1836854956785352776">来自 Alex Albert (@alexalbert__) 的推文</a>：很高兴分享我们关于 Contextual Retrieval 的最新研究——这项技术可以将错误的分块检索率降低高达 67%。当与 prompt caching 结合时，它可能是最好的技术之一……</li><li><a href="https://github.blog/changelog/2024-09-19-sign-up-for-openai-o1-access-on-github/">在 GitHub 上申请 OpenAI o1 访问权限 · GitHub Changelog</a>：在 GitHub 上申请 OpenAI o1 访问权限</li><li><a href="https://x.com/hwchung27/status/1836842717302943774?s=46">来自 Hyung Won Chung (@hwchung27) 的推文</a>：这是我在 @MIT 的演讲（延迟了一段时间😅）。这个演讲是我去年在思考范式转变时准备的。这次延迟发布非常及时，因为我们刚刚发布了 o1，我相信这是一个新的范式……</li><li><a href="https://x.com/cognition_labs/status/1836866696797401118">来自 Cognition (@cognition_labs) 的推文</a>：Devin 变得更快了，代码编辑更准确，遵循指令更可靠，独立决策能力更强。我们还改进了对企业安全的支持……</li><li><a href="https://www.youtube.com/watch?v=tEzs3VHyBDM">构建 OpenAI o1（加长版）</a>：顶排（从左到右）：Mark Chen, Giambattista Parascandolo, Trapit Bansal, Łukasz Kaiser, Hunter Lightman, Karl Cobbe, Łukasz Kondraciuk, Szymon Sidor, No...</li><li><a href="https://github.com/o1-waitlist-signup">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://www.youtube.com/watch?v=kYWUEV_e2ss&feature=youtu.be">MIT EI 研讨会，来自 OpenAI 的 Hyung Won Chung。“不要教导，要激励。”</a>：这个演讲是我去年在思考范式转变时准备的。这次延迟发布非常及时，因为我们刚刚发布了 o1，我相信这是一个新的范式……</li><li><a href="https://m.youtube.com/watch?v=IityUpVVD38">与 Dharmesh Shah 探讨 AI Agents 的未来 | INBOUND 2024</a>：免费访问 Agent.AI：https://clickhubspot.com/dlxp HubSpot 联合创始人兼 CTO Dharmesh Shah 对 AI agents 的未来做出了预测。他认为……</li><li><a href="https://arxiv.org/search/?query=rag+improvement&searchtype=all&abstracts=show&order=-announced_date_first&size=50">搜索 | arXiv 电子打印仓库</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/allenai/MADLAD-400">allenai/MADLAD-400 · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: 关于 SOTA Prompting 的新播客上线了！ https://x.com/latentspacepod/status/1837206370573041758
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1286778767464726603)** (53 条消息🔥): 

> - `Cursor usage` (Cursor 使用情况)
> - `Emoji reactions issues` (Emoji reactions 问题)
> - `Discord message editing problems` (Discord 消息编辑问题)
> - `Cody and Claude alternatives` (Cody 和 Claude 替代方案)
> - `Zoom meeting link` (Zoom 会议链接)


- **Cursor 占据了对话主导地位**：成员们主要在讨论他们使用 **Cursor** 的体验，许多人表达了满意之情，并希望了解更多关于其功能的信息。
   - 一位成员提到，他们在 Cursor 中的工作流已经非常完善，以至于尝试其他工具变得很有挑战性。
- **Emoji reactions 无法正常工作**：几位参与者反映了 Discord 中 **Emoji reactions** 无法正常运行的问题，引发了关于潜在修复方案的讨论。
   - 一位成员表示他们正在重启应用以解决 Emoji reaction 问题。
- **对 Discord 功能的挫败感**：多位成员报告了在 Discord 中 **编辑消息** 的问题，引发了关于 Discord 可靠性的讨论。
   - 一位用户指出，他们不确定问题是出在自己的系统还是 Discord 的全局性问题。
- **讨论了 Phind 的替代方案**：一位成员表示他们已经停止使用 **Phind**，因为他们发现了更好的替代方案，如 **Cody** 和 **Cursor**。
   - 这一转变凸显了用户在不断探索旨在提高 AI 应用生产力的工具。
- **分享了 Zoom 会议坐标**：一位用户分享了 **Zoom 会议链接** 以及会议 ID 和密码，供其他人加入。
   - 该链接作为正在进行的聊天的一部分被分享，反映了成员之间的协作。



**提到的链接**：<a href="https://zoom.us/j/8715206103?pwd=Tnp0VnlMUjZZSlYvRnB5dzJGVk13QT09">Join our Cloud HD Video Meeting</a>：Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，适用于移动端、桌面端和会议室系统的视频和音频会议、聊天及网络研讨会。Zoom ...

  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1286464007640846397)** (1 条消息): 

> - `Chatroom improvements` (Chatroom 改进)
> - `New Model Releases` (新模型发布)
> - `Hermes 3 Pricing Update` (Hermes 3 价格更新)


- **Chatroom 引入可编辑消息**：Chatroom 现在支持 **可编辑消息**，允许用户通过点击重新生成按钮来编辑自己的消息或机器人的回复，以获取新的回复。
   - 此外，统计数据（stats）进行了 **重新设计**，增强了用户交互体验。
- **Qwen 2.5 提高了 AI 模型的标准**：**Qwen 2.5 72B** 在编程和数学方面能力显著提升，并拥有令人印象深刻的 **131,072 context** 窗口。更多详情请参阅 [这里](https://openrouter.ai/models/qwen/qwen-2.5-72b-instruct)。
   - 该模型展示了显著的知识跨越，为 AI 能力设定了新标准。
- **Mistral Pixtral 作为多模态模型首次亮相**：**Mistral Pixtral 12B** 标志着 Mistral 首次进军多模态 AI 领域，同时也为用户提供了免费版本。详情请参阅 [这里](https://openrouter.ai/models/mistralai/pixtral-12b)。
   - 这一举措代表了通过多模态功能提供通用 AI 解决方案的进步。
- **Neversleep Lumimaid 升级公告**：**Neversleep Lumimaid v0.2 8B** 现在是 Llama 3.1 8B 的 finetune 版本，与前代相比，在 **数据集方面有了巨大提升**。更多信息请参阅 [这里](https://openrouter.ai/models/neversleep/llama-3.1-lumimaid-8b)。
   - 此次升级展示了致力于通过提高数据集质量来提升模型性能的承诺。
- **Hermes 3 转为付费模式但保留免费版本**：**Hermes 3** 正在转型为每月 **$4.5** 的付费模型，不过目前仍提供免费版本。详情请参阅 [这里](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b)。
   - 总体而言，这一转变反映了市场上模型产品的不断演进。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/qwen/qwen-2.5-72b-instruct">Qwen2.5 72B Instruct - API, Providers, Stats</a>: Qwen2.5 72B 是 Qwen 大语言模型（LLM）的最新系列。通过 API 运行 Qwen2.5 72B Instruct</li><li><a href="https://openrouter.ai/models/neversleep/llama-3.1-lumimaid-8b">Lumimaid v0.2 8B - API, Providers, Stats</a>: Lumimaid v0.2 8B 是 Llama 3 的 finetune 版本。通过 API 运行 Lumimaid v0.2 8B</li><li><a href="https://openrouter.ai/models/mistralai/pixtral-12b">Pixtral 12B - API, Providers, Stats</a>: Mistral AI 的首个图像转文本模型。其权重按照惯例通过种子（torrent）发布：https://x. 通过 API 运行 Pixtral 12B</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API, Providers, Stats</a>: Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更出色的角色扮演、推理、多轮对话、长 context 连贯性...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1286415291445936170)** (79 条消息🔥🔥): 

> - `OpenRouter 的前端`
> - `SillyTavern 功能`
> - `模型价格变动`
> - `集成 API 问题`
> - `功能请求与进一步开发` 


- **AI 聊天前端推荐**：一名用户寻求在 OpenRouter 上与 AI 聊天的前端推荐，并提到需要在不同 PC 之间共享对话。另一位成员推荐了 [every-chatgpt-gui](https://github.com/billmei/every-chatgpt-gui/blob/main/README.md) 作为全面的前端选项。
   - 有人指出，虽然 SillyTavern 擅长角色扮演，且支持代码片段的语法高亮，但对于编程任务可能并非最佳选择。
- **Hermes 模型的近期价格变动**：提醒注意 **nousresearch/hermes-3-llama-3.1-405b** 的价格已从免费上调至 **$4.5/M**，这让依赖免费额度的用户感到意外。用户对突然的价格变动缺乏通知表示担忧。
   - 有信息显示，由于 OpenRouter 在转发缓存细节方面的限制，缓存的 Token 未能反映在用量统计中。
- **集成 API 挑战**：一名用户遇到了集成 API Key 无法正常工作的问题，尽管有保证称 Lambda 的服务仍然免费。建议检查活动页面以验证 Prompt Caching 的有效性。
   - 多位用户交流了利用缓存的技巧，并提供了功能实现的更新，以更好地辅助 API 使用。
- **模型性能讨论**：用户对比了不同模型的性能，特别提到 **deepinfra qwen72b** 速度较慢（5-8 tok/s），而 **hyperbolic** 则明显更快。一些人质疑 OpenRouter 在大型应用中的可扩展性和预期用途。
   - 几位用户报告了在大量 Token 使用场景下成功应用 OpenRouter 的案例，强调了其在个人及更广泛应用场景中的潜力。
- **功能请求与未来发展**：有人询问如何提交功能请求，并被引导至特定频道。讨论还涉及对更强大的聊天记录共享选项的需求。
   - 社区期待即将发布的增强功能更新，包括围绕缓存和用户参与度的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/activity">Activity | OpenRouter</a>: 查看您在 OpenRouter 上使用模型的情况。</li><li><a href="https://www.anthropic.com/news/contextual-retrieval">Introducing Contextual Retrieval</a>: Anthropic 是一家 AI 安全与研究公司，致力于构建可靠、可解释且可控的 AI 系统。</li><li><a href="https://simonwillison.net/2024/Sep/20/introducing-contextual-retrieval/">Introducing Contextual Retrieval</a>: 这是一种有趣的全新 Embedding/RAG 技术，由 Anthropic 描述，但它应适用于任何 Embedding 模型及其他 LLM。在实现语义搜索时面临的一大挑战是...</li><li><a href="https://padolsey.medium.com/using-llms-to-parse-and-understand-proposed-legislation-9eec469d9830#:~:text=A%20rundown%20of%20the%20entire%20process>">Using LLMs to parse and understand proposed legislation</a>: 众所周知，法律条文的阅读和理解极具挑战性。事实上，这类文件甚至不是为了让普通人阅读而设计的……</li><li><a href="https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb">anthropic-cookbook/misc/prompt_caching.ipynb at main · anthropics/anthropic-cookbook</a>: 展示使用 Claude 的一些有趣且有效方法的 Notebook/Recipe 集合。- anthropics/anthropic-cookbook</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>: ChatGPT 的各类前端 GUI 客户端。通过在 GitHub 上创建账号为 billmei/every-chatgpt-gui 的开发做出贡献。
</li>
</ul>

</div>

### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1286705139557859420)** (1 messages): 

> - `Custom API Integration`
> - `Private LLM Servers` 


- **自定义 API Key 集成请求**：一名成员建议增加添加自定义 OpenAI 兼容的 **URL / API KEY 端点**的功能，以方便与**私有 LLM 服务器**集成。
   - 该请求强调了对更灵活集成选项的需求，以支持多样的用户环境和部署。
- **关于集成灵活性的讨论**：几位成员表达了在当前和未来的**集成**中允许自定义端点的重要性，以实现更广泛的兼容性。
   - 大家的共识是，启用这些功能可以通过适应不同的系统架构来提升整体用户体验。


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1286409345831145574)** (2 messages): 

> - `RAG integrations`
> - `Opik partnership`
> - `RAGApp v0.1 release` 


- **Opik 自动记录 RAG 调用**：团队很高兴宣布与 [Opik](https://t.co/Z3KdwjAKKv) 达成合作伙伴关系，该工具可自动记录开发和生产环境中所有的 RAG/Agent 调用和追踪。
   - Opik 通过自动化简化了认证流程，优化了多步工作流中的用户体验。
- **RAGApp v0.1：无代码多 Agent 应用**：今天发布的 [RAGApp v0.1](https://t.co/wyRNnnrmig) 允许用户在不编写任何代码的情况下构建多 Agent 应用程序。
   - 用户可以自由添加 Agent、分配角色、设置系统提示词（system prompts），并利用各种工具来增强其应用程序。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1286419361506267147)** (68 条消息🔥🔥): 

> - `LlamaIndex 与 Pinecone 集成`
> - `Pandas Query Engine 行为`
> - `Graph RAG 查询问题`
> - `Gemini LLM 错误`
> - `Contextual Retrieval 功能` 


- **LlamaIndex 与 Pinecone ID 的问题**：用户反映在使用 LlamaIndex 后，Pinecone 中的 ID 控制存在困难，并指出 Pinecone 自动生成的 ID 使得删除操作变得繁琐。
   - 建议包括手动创建 Node 或编辑 ID，并指出目前的删除限制使得必须使用前缀（prefixes）。
- **Pandas Query Engine 中的查询不一致**：一位用户观察到在 Notebook 和 Python 脚本中使用 Pandas Query Engine 时，查询输出存在差异，特别是在利用 `df.head()` 时。
   - 将 `df.head()` 修改为 `df.head(1)` 解决了该问题，这表明 DataFrame 中的列数可能会影响查询解析。
- **Graph RAG 查询困难**：一位使用 Graph RAG 的用户在查询索引时遇到问题，尽管使用了相同的 Notebook，但提供的模式（pattern）与检索到的块（chunks）不匹配。
   - 对 `GraphRAGQueryEngine` 模式的调查表明，在获取数据期间存在预期不匹配的情况。
- **Gemini LLM 兼容性错误**：用户在使用 Gemini LLM 时遇到了 'AttributeError: to_dict'，这指向了与某些库版本的潜在不兼容性。
   - 建议包括降级版本，以及提交 Pull Request 来解决观察到的问题。
- **LlamaIndex 中的 Contextual Retrieval 和 Hybrid Retrieval**：LlamaIndex 支持类似于 Anthropic 的 Contextual Embeddings 的上下文元数据提取，允许用户通过摘要和问答来增强索引。
   - 讨论中还涉及了添加 BM25 检索方法和使用 `QueryFusionRetriever`，强调了多种检索策略的集成。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.pinecone.io/guides/data/manage-rag-documents#delete-all-records-for-a-parent-document">管理 RAG 文档 - Pinecone 文档</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_metadata_extractor/#metadata-extraction-usage-pattern">元数据提取 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/#hybrid-retriever-with-bm25-chroma">BM25 检索器 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/existing_data/pinecone_existing_data/">指南：在现有 Pinecone 向量存储中使用 Vector Store Index - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/1d49e15f4b91f6e4b931d8ae42f69dc678ce8ee4/llama-index-integrations/llms/llama-index-llms-gemini/llama_index/llms/gemini/utils.py#L32-L62">llama_index/llama-index-integrations/llms/llama-index-llms-gemini/llama_index/llms/gemini/utils.py (位于 1d49e15f4b91f6e4b931d8ae42f69dc678ce8ee4) · run-llama/llama_index</a>：LlamaIndex 是适用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/issues/14897">[Bug]: Pandas Query Engine 无法给出正确响应（特别是在通过 API 暴露时，在本地 Jupyter Notebook 中运行良好） · Issue #14897 · run-llama/llama_index</a>：Bug 描述：如果 Pandas Query Engine 在 Jupyter Notebook 中运行，它会给出正确结果，但如果作为单个 .py 文件运行，则无法给出正确结果。版本 llama-index==0.10.50。步骤...</li><li><a href="https://github.com/run-llama/llama_index/blob/a18b94699ac4e49b17f3f49879adf29dfc7c3ed3/llama-index-core/llama_index/core/indices/property_graph/base.py#L308">llama_index/llama-index-core/llama_index/core/indices/property_graph/base.py (位于 a18b94699ac4e49b17f3f49879adf29dfc7c3ed3) · run-llama/llama_index</a>：LlamaIndex 是适用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/26205a0e96d36382cd4a09432e51731ddb5170a1/llama-index-integrations/vector_stores/llama-index-vector-stores-pinecone/llama_index/vector_stores/pinecone/base.py#L170">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-pinecone/llama_index/vector_stores/pinecone/base.py (位于 26205a0e96d36382cd4a09432e51731ddb5170a1) · run-llama/llama_index</a>：LlamaIndex 是适用于 LLM 应用程序的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/)** (1 条消息): 

stk_vnluser: 是的
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1286411140473032934)** (59 条消息🔥🔥): 

> - `o1 vs 4o performance` (o1 与 4o 性能对比)
> - `GPT task efficiency and reasoning` (GPT 任务效率与推理)
> - `Local server memory implementation` (本地服务器内存实现)
> - `AI development feedback` (AI 开发反馈)
> - `AI consciousness discussion` (AI 意识讨论)


- **o1 mini 被认为不如 4o**：用户表示 **o1 mini** 感觉缺乏现实世界的经验，没有表现出智能推理，断言它仅仅是比 **4o** 打字更快。
   - *一位用户提到 o1 感觉并不比 4o 强，* 引发了关于 AI 实际认知能力的讨论。
- **关于 AI 推理与意识的辩论**：随后引发了一场关于 AI 是真正具有推理能力还是仅仅在模拟推理的长篇辩论，对 AI 的意图性和理解能力持有不同看法。
   - 一位参与者建议，一个专注于任务完成而忽略类人推理的 AI 可能会更安全、更高效。
- **GPT-4 API 的 Memory 能力**：用户询问是否可以通过 API 为 **GPT-4** 提供 **memory** 功能，并澄清这些功能目前仅限 ChatGPT Plus 用户在界面内使用。
   - *一位用户指出，尽管 ChatGPT 界面缺乏此功能，但使用 Pinecone 等替代方案很容易实现自己的 memory 工具。*
- **对 IDE 集成反馈的请求**：分享了关于改进集成在 IDE 中的 AI 工具的建议，讨论了对实时预览以增强工作流的需求，类似于 **ClaudeAI** 中的功能。
   - *几位用户表示希望 ChatGPT 加入此功能，* 而其他人则建议探索各种 IDE 以获得更好的集成。
- **讨论 AI 的能力限制**：对话强调了对 AI 在分布外（out-of-distribution）场景中处理推理能力的怀疑，以及构建普适对齐伦理框架的挑战。
   - 参与者指出，在单一伦理真相下的任何对齐策略都可能无意中导致更广泛的问题或偏见，强调了 AI 交互的复杂性。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 条消息): 

null.user: hmm
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1286435511912759379)** (4 条消息): 

> - `Effective Prompts` (有效提示词)
> - `ChatGPT Usage` (ChatGPT 使用)


- **分享有用的提示词**：一位成员分享了他们过去编写的一个**有用的提示词**，并表示它在今天仍然适用。他们链接到了自己的 [prompt guide](https://chatgpt.com/g/g-ssHSsvECZ-guidegpt) 供他人探索。
   - 这一行为鼓励社区成员交换宝贵的资源和提示词，以增强使用 GPT 模型的体验。
- **提示词理解的改进**：另一位成员提到，截图显示的提示词有效地捕捉到了预期的想法。他们乐观地认为这有助于更好地理解概念。
   - 这突显了视觉辅助工具在理解如何有效构建提示词方面的重要性。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1286435511912759379)** (4 条消息): 

> - `Prompt sharing` (提示词分享)
> - `GuideGPT utility` (GuideGPT 工具)


- **看起来很有效！**：一位成员提到，截图显示的提示词似乎有效地捕捉到了预期的想法。
   - 这表明视觉示例可以增强对提示词结构的理解。
- **表达感谢！**：另一位成员对收到的帮助表示感谢，展示了互助的氛围。
   - 这种认可促进了社区参与。
- **分享一个有用的提示词**：一位成员分享了他们之前编写的一个特别有用的提示词。
   - 这表明了社区内协作和资源共享的文化。
- **分享了 GuideGPT 链接**：提供了 GuideGPT 提示词的直接链接，方便他人访问。
   - 分享此类资源促进了对组内有效工具的探索。


  

---



### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1286401826618474527)** (46 条消息🔥): 

> - `Model Initialization Techniques` (模型初始化技术)
> - `Iterated Distillation and Amplification` (迭代蒸馏与放大)
> - `Challenges in FP8 Training` (FP8 训练中的挑战)
> - `Llama1 Checkpoints Status` (Llama1 Checkpoints 状态)
> - `Overcomplete Color Space in Image Models` (图像模型中的过完备色彩空间)

- **探索用于模型初始化的 HyperCloning**：讨论利用 **HyperCloning** 使用较小的预训练模型来初始化大型语言模型（LLMs），从而可能节省训练时间并提高准确性。
   - *一位贡献者建议*采用训练微型模型、对其进行扩展，然后从较大模型中进行蒸馏的方法，以此来优化计算资源的使用。
- **迭代放大（Iterated Amplification）受到关注**：**迭代蒸馏与放大（IDA）** 的概念因其通过迭代过程有效对齐 AI 系统的潜力而获得认可。
   - *一位参与者对“蒸馏”一词表示怀疑*，认为它没有捕捉到压缩的必要属性以及丢弃信息的目的。
- **FP8 训练中遇到的困难**：对话强调了在 FP8 训练中发现的**关键不稳定性**，这是由于在长期的训练运行中，SwiGLU 激活函数导致了离群值放大。
   - *一位听众询问*其他激活函数（如 relu^2）在长时间训练场景中是否可能面临类似问题。
- **原始 Llama1 Checkpoints 的状态**：询问了原始 **Llama1 checkpoints** 的状态，并分享了关于其访问权限和潜在泄露的见解。
   - *一位用户指出*，这些 checkpoints 并没有被公开上传，并引用了一个包含泄露版本的特定 Hugging Face 仓库。
- **使用过完备颜色空间（Overcomplete Color Spaces）**：提出了一项有趣的建议，即**过完备颜色空间**是否可以通过冗余的输入表示来增强图像模型的性能。
   - *该想法在哲学上与感质（qualia）的概念相关联*，暗示类似的方法可能对语言模型也有益。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2409.12517">Scaling FP8 training to trillion-token LLMs</a>：我们首次在高达 2 万亿 Token 的数据集上使用 FP8 精度训练大型语言模型，这比之前的限制增加了 20 倍。通过这些扩展的训练运行，我们发现了……</li><li><a href="https://arxiv.org/abs/2409.12903">Scaling Smart: Accelerating Large Language Model Pre-training with Small Model Initialization</a>：语言模型的预训练阶段通常从随机初始化的参数开始。随着模型规模化的当前趋势，训练其大量的参数可能会极其缓慢……</li><li><a href="https://arxiv.org/abs/2409.12618">Iteration of Thought: Leveraging Inner Dialogue for Autonomous Large Language Model Reasoning</a>：迭代式的人类参与是利用大型语言模型（LLMs）先进语言处理能力的常用且有效手段。通过在对话方式中使用结构良好的提示……</li><li><a href="https://arxiv.org/abs/2409.12917">Training Language Models to Self-Correct via Reinforcement Learning</a>：自我纠错是大型语言模型（LLMs）一种非常理想的能力，但在现代 LLMs 中一直被发现很大程度上是无效的。现有的训练自我纠错的方法……</li><li><a href="https://x.com/BoshiWang2/status/1836938361216409750">Boshi Wang (@BoshiWang2) 的推文</a>：OpenAI o1 能解决困难的推理问题吗？我们在 Grokked Transformers 论文中的复杂推理任务上对其进行了测试。结果表明，o1-preview 像早期的 LLMs 一样也面临很大困难；在……</li><li><a href="https://arxiv.org/abs/2302.02774">The SSL Interplay: Augmentations, Inductive Bias, and Generalization</a>：自监督学习（SSL）已成为一种强大的框架，可以从原始数据中学习表示而无需监督。但在实践中，工程师们面临着诸如调优优化中的不稳定性等问题……</li><li><a href="https://arxiv.org/abs/2303.00633">An Information-Theoretic Perspective on Variance-Invariance-Covariance Regularization</a>：方差-不变性-协方差正则化（VICReg）是一种自监督学习（SSL）方法，在各种任务上都显示出了良好的结果。然而，其背后的基本机制……</li><li><a href="https://www.alignmentforum.org/posts/vhfATmAoJcN8RqGg6/a-guide-to-iterated-amplification-and-debate">迭代放大与辩论指南 — AI Alignment Forum</a>：这篇文章关于两种以可扩展方式对齐 AI 系统的建议：……</li><li><a href="https://arxiv.org/abs/2205.11508">Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods</a>：自监督学习（SSL）推测输入和成对的正向关系足以学习有意义的表示。尽管 SSL 最近达到了一个里程碑：表现优于有监督……
</li>
</ul>

</div>

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1286648125905436712)** (4 条消息): 

> - `Tokenized SAEs`
> - `Spectral Filters`
> - `Whisper Interpretability`
> - `Attention-MLP Interactions`
> - `Interpretable Sequence Continuation` 


- **Tokenized SAEs 增强架构性能**：一篇关于 [tokenized SAEs](https://www.lesswrong.com/posts/P8qLZco6Zq8LaLHe9/tokenized-saes-infusing-per-token-biases) 的文章介绍了一种 per-token decoder bias（逐 token 解码器偏置），通过显著加快训练速度来改进 **GPT-2** 和 **Pythia** 等现有模型。
   - *该方法解决了训练类别不平衡问题*，通过 'unigram reconstruction'（一元语法重构）促进了局部上下文特征的学习。
- **AI 中 Spectral Filters 的探索**：讨论中提到了 [spectral filters](https://arxiv.org/abs/2402.09221)，尽管聊天中未提供具体细节。
   - 提到了 Spectral Filters 与近期模型性能的相关性。
- **关于 Whisper Interpretability 的见解**：分享了一篇关于 [Whisper Interpretability](https://er537.github.io/blog/2023/09/05/whisper_interpretability.html) 的博客文章，重点在于增强对模型运行机制的理解。
   - 博文中提出的见解强调了可解释性在复杂模型中的重要性。
- **对即将发表的 EMNLP 论文感到兴奋**：*非常自豪有 2 篇论文入选 #EMNLP2024！* 根据 @FazlBarez 的帖子，标题包括《Interpreting Context Look-ups in Transformers》，重点关注 **Attention-MLP interactions**。
   - 另一篇题为《Towards Interpretable Sequence Continuation》的论文旨在分析 **Large Language Models** 中的 **shared circuits**（共享电路）。
- **可解释性研究中对 Key/Value Cache 的关注**：一位成员评论道，在可解释性讨论中，通常的焦点是 *residual stream 的统计数据*，而不是 **Key/Value Cache** 机制。
   - 这种视角的转变强调了理解模型解释中底层机制的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/FazlBarez/status/1837229484543726036">Fazl Barez (@FazlBarez) 的推文</a>：非常自豪有 2 篇论文入选 #EMNLP2024！🚀 1️⃣ &#34;Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions&#34;  2️⃣&#34;Towards Interpretable Sequence Continuati...</li><li><a href="https://www.lesswrong.com/posts/P8qLZco6Zq8LaLHe9/tokenized-saes-infusing-per-token-biases)">Tokenized SAEs: Infusing per-token biases. — LessWrong</a>：tl;dr * 我们引入了向 SAEs 添加 per-token decoder bias 的概念。换句话说，我们添加了一个由最后看到的 token 索引的查找表。T…
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1286655475512774716)** (2 条消息): 

> - `Gemma Models`
> - `BOS Token Application` 


- **关于 Gemma 模型中 BOS Token 的担忧**：一位成员根据他们在语言建模任务中的发现，对 **Gemma 模型** 的序列可能只添加了一次 **BOS token** 表示担忧。
   - 他们质疑是否应该每次都一致地添加 BOS token，特别是在滚动 **loglikelihood** 计算的背景下。
- **在调试器中验证输入**：该成员确认他们在调试器中停止了模型调用并验证了其输入，注意到在某些情况下 **llh_rolling** 中缺少 **BOS token**。
   - 这一发现引发了对建模过程中 BOS token 应用的进一步审查。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1286401433301946428)** (33 messages🔥): 

> - `MacOS 上的 IPv4 切换`
> - `LM Studio API 连接问题`
> - `处理模型加载错误`
> - `在 LM Studio 中追踪 API 调用者`
> - `Qwen2.5-Coder 兼容性` 


- **切换到 IPv4 解决连接问题**：一位成员建议切换到 **IPv4** 通常能解决 MacOS 上常见的连接问题，并明确了如何调整设置。
   - 经过一番排障后，另一位成员确认该方法有效，并表示：*“它直接就能用了。”*
- **连接 LM Studio 到 CrewAI 的困扰**：成员们讨论了将 **LM Studio API** 连接到 **CrewAI** 的挑战，探索了各种解决方案，但尚未达成定论。
   - 一位成员引用了一个 [YouTube 视频](https://www.youtube.com/watch?v=fnchsJd9pfE)，以获取在 **LM Studio** 中使用 **CrewAI** 的指导。
- **模型加载错误及潜在修复**：一位成员概述了 **llama.cpp** 中关于模型加载的一个错误，另一位成员解释说该模型在 **LM Studio** 中不受支持。
   - 讨论还围绕图形处理单元 (GPU) 的选择以及从 0.2.* 版本到 0.3.* 版本的变化展开。
- **识别 LM Studio 中的 API 调用者**：一位成员询问如何追踪 **LM Studio** 的 API 调用者，得到的建议是使用 Python 为 API key 创建一个自定义封装器 (wrapper)。
   - 然而，官方表示目前近期没有开发内置追踪功能的计划。
- **Qwen2.5-Coder 与 LM Studio 的兼容性**：成员们询问 **Qwen2.5-Coder** 是否适用于 **LM Studio**，因为他们只能找到 **Qwen** 发行版的引用。
   - 这引发了人们对 Qwen 在 LM Studio 平台背景下能力的兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://support.apple.com/en-gb/guide/mac-help/mh14129/mac">在 Mac 上更改 TCP/IP 设置</a>：在 Mac 上，使用 TCP/IP 网络设置来配置 IPv4 或 IPv6 连接，或续订 DHCP 租约。</li><li><a href="https://www.youtube.com/watch?v=fnchsJd9pfE">CrewAI：使用 LM Studio, Ollama, JanAI &amp; TextGen 的 AI 驱动博客 Agent</a>：🌟 欢迎来到 AI 驱动博客世界的精彩旅程！🌟在今天的视频中，我将带你完成一个使用 Crew AI 的综合教程...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1286413272785879132)** (13 messages🔥): 

> - `3090 功耗限制 vs 降压 (Undervolting)`
> - `跨操作系统的电源管理`
> - `比较 GPU 功耗限制设置`
> - `RAM 速度与 CPU 推理瓶颈`
> - `针对 DDR6 和 CPU 推理的主板设计` 


- **3090 功耗限制与降压策略**：一位成员讨论了将 **3090** 限制在每张 **290W**，并询问在考虑时钟频率的情况下如何将其与降压进行比较。
   - 有人建议参考 *RTFM*（阅读手册），并指向了各种资源以进行更深入的了解。
- **电源管理：Windows vs Linux**：对比了如何实现更低的 GPU 功耗，指出在 Windows 中需要手动调整设置，而在 Linux 中只需通过 `nvidia-smi` 执行一条命令。
   - 这引发了关于操作系统易用性的辩论，其他人确认 Windows 中的设置速度更快。
- **设置 GPU 功耗限制的有效性**：一位成员分享说，将 **4090** 的功耗限制从 **450W** 降至 **320W** 仅导致 **FPS 下降了 10%**。
   - 这引发了关于功耗限制作为电源管理方法是否比降压更充分的对话。
- **辩论 RAM 速度与 CPU 推理**：一位成员质疑 **RAM 速度和带宽** 是否是 CPU 推理过程中的最大瓶颈。
   - 他们提出了一个可以直接使用 **DDR6** 进行 CPU 推理的主板概念，引发了关于为什么这种设计不存在的好奇。
- **CPU 核心利用不足**：一位成员对大量 **CPU 核心** 基本处于闲置状态表示沮丧，尤其是因为他自 **2011** 年以来就没买过电子游戏。
   - 这提出了关于当前 CPU 设计在通用计算工作负载中效率的问题。



**提及的链接**：<a href="https://www.youtube.com/watch?v=FqpfYTi43TE">RTX 3080 / 3090 降压 | 功耗降低 100W = 性能不变？</a>：在 Amazon 查看价格：Nvidia RTX 3090: https://geni.us/4o7Xj Nvidia RTX 3080: https://geni.us/Dk9g3 GPU 降压指南（深度）：https://youtu.be/z...

  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1286635550425747487)** (4 messages): 

> - `Mojo LLMs API`
> - `Pythagora Dev Tool`
> - `Magic 反馈` 


- **对 Mojo LLMs API 的兴趣**：一位用户表达了将基于 **Mojo 的 LLMs API** 与 [Pythagora](https://www.pythagora.ai) 结合使用的兴趣，Pythagora 是一款通过对话交互构建应用的开发工具。
   - 他们询问了使用该服务的相关**费用**，并强调了对其开启**软件开发新时代**的兴奋之情。
- **Pythagora AI 编程助手咨询**：一位用户询问另一位用户是否打算在 Mojo 中使用 **Pythagora 的 AI 编程助手**。
   - 这再次引发了关于将 Mojo 与 Pythagora 集成以增强编程能力的讨论。
- **Magic 反馈最后召集**：发出了一项提醒，邀请参与者加入 **Magic 反馈聊天**以获取产品洞察，特别针对那些尚未尝试过 Magic 的用户。
   - 该活动承诺为反馈提供者提供 **30 分钟**的快速交流环节及专属周边，感兴趣的人员可以在[此处](https://modul.ar/user-feedback)预约。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://modul.ar/user-feedback">Zoom 调度器</a>：未找到描述</li><li><a href="https://www.pythagora.ai">Pythagora</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1286434243412492389)** (2 messages): 

> - `关闭 GitHub Discussions`
> - `即将举行的社区会议` 


- **关闭 Mojo 和 MAX 的 GitHub Discussions**：为了集中社区力量，[Mojo](https://github.com/modularml/mojo/discussions) 和 [MAX](https://github.com/modularml/max/discussions) 仓库的 GitHub Discussions 将于 **9 月 26 日**关闭。
   - *评论超过 10 条的重要讨论将被转换为 GitHub Issues*，成员可以通过标记组织者来请求转换特定讨论。
- **下一次社区会议时间调整**：下一次**社区会议**已移至太平洋时间 **9 月 30 日星期一上午 10 点**。成员可以通过此[链接](https://modul.ar/community-meeting)将会议添加到日历，并在 [Google doc](https://docs.google.com/document/d/1Hdy52tJXbUR2jZSYt-IFdaEJRRBHvHCQkODAZnuXsNc/edit#heading=h.hthojob043vc) 中提交他们的演讲内容。
   - 鼓励参与者提前准备并提交演示文稿，以提升会议体验。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://modul.ar/community-meeting">Google 日历 - 登录以访问和编辑您的日程</a>：未找到描述</li><li><a href="https://docs.google.com/document/d/1Hdy52tJXbUR2jZSYt-IFdaEJRRBHvHCQkODAZnuXsNc/edit#heading=h.hthojob043vc)">[公开] MAX + Mojo 社区会议</a>：MAX + Mojo 社区会议文档链接：https://modul.ar/community-meeting-doc。这是一个公开文档；欢迎所有人查看、评论或提出建议。所有会议参与者必须遵守...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1286434670237454469)** (23 messages🔥): 

> - `Variable Bit Width Integers` (可变位宽整数)
> - `Packed Structs in Mojo` (Mojo 中的紧凑结构体)
> - `Set Implementation and __copyinit__` (Set 实现与 __copyinit__)
> - `Custom Decorators` (自定义装饰器)
> - `Generic Struct Syntax` (泛型结构体语法)


- **TCP/IP 实现需要可变位宽整数**：一位成员询问了在 TCP/IP 实现中使用类似 UInt{1,2,3,4,6,7,13} 的**可变位宽整数**的可能性。
   - 虽然有人建议使用**位运算符和掩码**，但认为其不够易用（ergonomic），同时表达了希望 **Mojo** 能支持此特性的愿望。
- **对 Packed Structs 支持的关注**：讨论中提到了 Mojo 缺乏 **packed structs**（紧凑结构体），这使得将位标志（bitflags）作为 `__mlir_type.i1` 列表处理变得复杂。
   - 一位成员希望 LLVM 能处理字节对齐问题，但对其可靠性持怀疑态度。
- **Set 实现中对 __copyinit__ 的需求**：一位成员询问为何 **Set** 没有实现 `__copyinit__`，这影响了它对 CollectionElement trait 的遵循。
   - 他们查阅了 GitHub issues 以寻求关于此遗漏的解释，但未发现令人满意的说明。
- **Mojo 中空列表的功能请求**：有人提出了一项功能请求，允许在 Mojo 中使用不带类型参数的**空列表**，以获得更好的 Python 兼容性。
   - 虽然这个想法得到了认可，但也有建议考虑从空列表字面量进行隐式转换。
- **实现多个 Trait 的泛型结构体语法**：关于定义实现多个 trait 的**泛型结构体**的当前（nightly）语法出现了疑问。
   - 成员们注意到在组合类似 `T: Trait1 & Trait2` 的 trait 时可能存在的语法问题。


  

---



### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1286478827744133130)** (27 messages🔥): 

> - `Input Position Settings` (输入位置设置)
> - `Memory Optimisations` (内存优化)
> - `Generation Efficiency` (生成效率)
> - `Batch Sizes Management` (批大小管理)
> - `Generate Recipe Simplification` (生成脚本简化)


- **输入位置系统正在审查中**：围绕在最近的 [PR](https://github.com/pytorch/torchtune/pull/1449/files#diff-99a96ce497241e82b0c4d56f4bef3437e29dd596881b8f6d4db4d93178f88af5L227) 中移除 **input_pos** 自动设置的决定展开了讨论，理由是简化**生成/缓存逻辑**。
   - 此次更改旨在通过避免用户在各种类中搜索默认设置来减少困惑。
- **讨论可配置的内存优化**：成员们讨论了各种**内存优化**，指出**激活卸载**（activation offloading）等方案正在开发中，而**分块交叉熵**（chunked cross entropy）等其他方案则鼓励使用。
   - 值得注意的是，有人提到过去的技术大多被认为是浪费的，但现在正被考虑用于**内存优化教程**。
- **不同批大小下的生成效率**：对话围绕不同批大小下的**生成效率**展开，特别强调了**生成脚本**在执行期间仅允许 **bsz 1**。
   - 成员们考虑了在配置中循环处理批次的简便性，但也认识到增加批大小带来的限制。
- **关于生成子方法的辩论**：关于引入 `sample` 和 `batched_sample` 等子方法进行了一场诙谐的辩论，这标志着对完善生成过程的关注。
   - 尽管一些成员倾向于拆分方法，但其他人建议采用更精简的方法，效仿 **gpt-fast** 等其他框架的实践。
- **生成工作流中的挑战与修订**：一位成员表示，由于报告的用户问题较多，迫切需要保持**生成脚本**（generate recipe）的简洁，特别是在向支持更大批大小过渡的过程中。
   - 目前正在努力精简逻辑，这被认为是降低用户在使用**生成功能**时复杂性的**必要**举措。



**提到的链接**：<a href="https://github.com/pytorch/torchtune/pull/1449/files#diff-99a96ce497241e82b0c4d56f4bef3437e29dd596881b8f6d4db4d93178f88af5L227">[RFC] Adding overrides for max cache seq length by SalmanMohammadi · Pull Request #1449 · pytorch/torchtune</a>：上下文 此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档，还是其他（请在此处添加） #1364 变更日志 此 PR：增加了对覆盖 th...

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1286415209237581885)** (5 messages): 

> - `OpenInterpreter Models`
> - `Task Success with OpenInterpreter`
> - `Enhancing Existing Projects`
> - `Firebase/Stripe Integrations` 


- **Wizardlm 在 OpenInterpreter 任务中表现优于 Llama**：在尝试了多个模型后，一位成员报告称 **microsoft/Wizardlm 8x22B** 的表现明显优于 **llama 405B**，完成任务所需的尝试次数更少。
   - *Wizardlm 经常能一次性成功完成任务*，展示了其在用户场景下的高效性。
- **使用 OpenInterpreter 成功完成任务**：一位成员成功使用 OpenInterpreter 组织和排序大型数据集，并创建了桌面快捷方式。
   - 他们表示希望了解其他成员使用该工具完成的其他任务。
- **使用 OpenInterpreter 进行增强时需要清晰的指令**：在要求 OpenInterpreter 修改现有应用程序时，需要给出明确的指令，因为它倾向于生成新项目。
   - 一位成员提到，为了避免在修改过程中出现非预期结果，必须保持指令的具体性和重复性。
- **寻求 Firebase/Stripe 故障排除帮助**：一位用户在多次尝试并更换新凭据后，仍无法使其集成了 **Firebase** 和 **Stripe** 的 **FarmFriend** 项目正常运行。
   - *陷入了“死循环”*，他们遇到了与 CORS、service accounts 和 authentication domains 相关的问题，请求熟悉这些领域的专家提供帮助。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1286476857583206433)** (11 messages🔥): 

> - `O1 Installation Video`
> - `Functionalities of O1`
> - `Discussion on O1`
> - `Scheduling Test Sessions` 


- **O1 安装视频更新**：一位用户询问了 O1 安装视频的进度，表示一旦视频发布就渴望协助测试。
   - 他们提到已向开发者发送了功能请求和关于视频的消息。
- **O1 仍处于开发阶段**：一位参与者评论道，O1 仍在开发中，目前尚未实现任何实用功能。
   - 据指出，O1 目前缺乏对应用程序的控制或应用间的交互能力。
- **关于 O1 讨论的提议**：一位成员建议就 O1 进行详细讨论，以深入了解其功能。
   - 他们提议在双方约定的时间使用特定频道进行讨论。
- **安排讨论以获取反馈**：同一位成员鼓励其他人提供参加 O1 讨论的空闲时间，并注明需要 GMT 时区。
   - 此举旨在确保开发者和用户之间能有充分的参与，进行全面的讨论。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1286402687612489928)** (15 条消息🔥): 

> - `CLANG dlopen 替换`
> - `配备 Intel GPU 的 Tinybox`
> - `IPMI 凭据问题`
> - `Lean 中 ShapeTrackers 的可合并性` 


- **探讨使用 mmap 替换 CLANG dlopen**：一场关于使用 **mmap** 替换 **CLANG dlopen** 的 Bounty 讨论兴起，这可能导致需要手动处理对象文件中的重定位（relocations），正如 [此 Pull Request](https://github.com/tinygrad/tinygrad/pull/6299) 所示。
   - 有人提到 *“目前我非常好奇谁会拿到这个赏金”*，显示出对完成此任务的兴趣。
- **关于 Tinygrad 和 Intel Arc GPU 的咨询**：一位用户询问 **Tinygrad** 在多块 **Intel GPU** 上是否运行良好（考虑到目前已支持 **AMD** 和 **NV**）。他们得到了肯定的答复，并被建议进一步深入研究。
- **受困于 IPMI IP 问题**：另一位用户报告了 **IPMI** 的问题，怀疑是凭据不正确，并寻求重置建议。建议包括使用显示器和键盘进行设置，并确认 **web BMC** 密码与显示的密码一致。
- **设置过程中的 GPU 连接问题**：有一个关于初始设置时应使用 **GPU HDMI** 还是 **VGA** 的问题，得到的明确回答是必须仅使用 **VGA**。
   - 这表明硬件连接是配置过程中常见的疏忽点。
- **关于 ShapeTrackers 的潜在本科论文**：一位用户询问了关于 **Lean 中两个任意 ShapeTrackers 的可合并性** 相关 Bounty 的状态，表示有兴趣将其作为本科毕业论文课题。
   - 他们观察到该任务似乎尚未完成，这标志着项目中存在新的贡献机会。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/6299">Replace dlopen with mmap in CLANG by christopherm99 · Pull Request #6299 · tinygrad/tinygrad</a>：在 M1 MacBook Pro 上进行了性能测试。来自 tinygrad.runtime.ops_clang import ClangProgram。使用 open(&quot;test.o&quot;, &quot;rb&quot;) as f: lib = f.read() for _ in range(1000): C...</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4492">Clang jit by uuuvn · Pull Request #4492 · tinygrad/tinygrad</a>：未找到描述。
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1286401892078845954)** (11 条消息🔥): 

> - `Cohere Discord 社区`
> - `Cohere 试用密钥`
> - `Connector 上的自定义超时`
> - `毕业设计项目 (Capstone Projects)`
> - `新闻通讯 (Newsletters)` 


- **Cohere Discord：学习之地**：成员们表达了加入 Cohere Discord 社区的兴奋之情，新人渴望了解 AI 和 Cohere 的产品。
   - 社区对新成员表示了 *欢迎！*，鼓励大家在协作的氛围中分享知识。
- **获取试用密钥并开始 Hack**：一位成员建议使用 Cohere 的试用密钥，该密钥每月免费提供 **1000 次调用**，并强调通过项目进行动手学习。
   - 另一位成员表示赞同，称 **实践是最好的学习方式**，并提到在完成毕业设计项目后会进行更多探索。
- **关于技术细节的讨论**：有人提出了响应时间为 **504** 的超时问题，随后询问了如何在 Connector 上设置自定义超时。
   - 这突显了关于在 Cohere 框架内管理服务交互的持续技术讨论。
- **对新闻通讯的兴趣**：一位成员提到自己是被 **classify newsletter** 吸引到社区的，这表明了新闻通讯在吸引参与者方面的价值。
   - 另一位成员表达了希望看到更多新闻通讯的愿望，显示出对社区持续互动和更新的兴趣。


  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1286405525533364268)** (3 messages): 

> - `Rerank Multilingual v3 问题`
> - `Rerank 模型对比`
> - `对 RAG 结果的影响` 


- **Rerank Multilingual v3 在处理英文时表现不佳**：一位成员报告了使用 **rerank_multilingual_v3** 时的差异，指出对于相似的英文查询，其得分低于 **0.05**，而使用 **rerank_english_v3** 时得分则为 **0.57** 和 **0.98**。
   - 这种不一致性正在影响他们的 **RAG 结果**，导致相关的 chunk 被意外过滤掉。
- **建议使用 Curl 命令进行测试**：另一位成员建议使用 **curl** 命令切换模型进行测试，并提出了诸如 **'what are the working hours?'** 和 **'what are the opening times?'** 之类的查询。
   - 这可能有助于更有效地比较模型性能。
- **缺乏文档可见性阻碍了支持工作**：一位用户表示，由于无法访问完整的文档集，他们无法提供进一步的帮助。
   - 他们提到，从他们的角度来看，模型似乎在正常工作，这增加了对所报告问题的困惑。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1286478046081187840)** (7 messages): 

> - `Whisper 优化`
> - `转录项目`
> - `GPU 利用率` 


- **寻求快速的 Whisper 解决方案**：一位成员请求关于如何最大化 **Whisper** 速度的帮助，询问了关于利用多 **GPU** 和支持转录任务 batching 的问题。
   - 他们提到需要转录一个**非常庞大的数据集**，表明了对高效处理的需求。
- **Whisper-TPU：一个快速的选择**：**Whisper-TPU** 被推荐作为一个显著快速的处理替代方案，代表了提升性能的一个潜在途径。
   - 该选项可以满足需要高速转录能力的用户的需求。


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1286559674380062720)** (4 messages): 

> - `Transfusion 架构`
> - `Diffusion 和 AR 训练稳定性`
> - `Qwen-Audio 训练挑战` 


- **探索 Transfusion 架构的使用**：人们对新模型是否可能利用 **Transfusion** 架构或类似方法（特别是针对多模态应用）感到好奇。
   - 一个相关的 [Transfusion GitHub 仓库](https://github.com/lucidrains/transfusion-pytorch) 提供了 Pytorch 实现，突出了其预测 token 和对图像进行 diffusion 的能力。
- **Diffusion 和 AR 训练的挑战**：实验揭示了在结合 **diffusion** 和 **AR 训练** 时难以实现**稳定性**，这表明方法集成中存在一个关键障碍。
   - 社区正在寻求提高这些训练方法稳定性的有效策略。
- **询问 Qwen-Audio 训练的不稳定性**：一位成员提到记得 **Qwen-Audio** 研究论文中讨论过多模态设置中的训练不稳定性问题，表明了这一挑战的普遍性。
   - 他们表示打算重新查阅该论文以澄清这些细节，并承认这与当前的讨论相关。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.12917">Training Language Models to Self-Correct via Reinforcement Learning</a>: 自我修正是大语言模型 (LLMs) 一项非常理想的能力，但在现代 LLM 中一直被发现很大程度上是无效的。现有的训练自我修正的方法...</li><li><a href="https://github.com/lucidrains/transfusion-pytorch">GitHub - lucidrains/transfusion-pytorch: Transfusion 的 Pytorch 实现，来自 MetaAI 的 &quot;Predict the Next Token and Diffuse Images with One Multi-Modal Model&quot;</a>: Transfusion 的 Pytorch 实现，来自 MetaAI 的 &quot;Predict the Next Token and Diffuse Images with One Multi-Modal Model&quot; - lucidrains/transfusion-pytorch
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1286415037170188365)** (11 messages🔥): 

> - `Qwen vs o1-mini`
> - `Llama 多模态开发`
> - `欧盟监管环境`
> - `OpenAI 扩展视频见解` 


- **Qwen 2.4 数学模型表现不如 o1-mini**：尽管是新发布的 **SOTA 开源数学模型**，**qwen2.4-math-72b-instruct** 在使用代码执行和 **n=256** 的集成方法时，仍未能超越 **o1-mini**。
   - 作为参考，它在 AIME 上与 **o1-mini** 持平，这说明在没有反思型 **CoT** 的情况下，仅通过 **256 次生成** 进行公平比较仍存在挑战。
- **Llama 多模态版本搁置**：一位开发者分享了其团队在 **Llama 多模态版本** 上的工作进展，但由于目前的不确定性，他们不会在欧盟发布该版本。
   - 这与此前关于碎片化监管可能阻碍欧洲 AI 创新的担忧相一致；他们的立场旨在支持欧洲开发者。
- **对欧盟反技术情绪的担忧**：社区成员对欧盟表现出的 **反技术 (anti-tech)** 倾向表示忧虑，认为虽然监管背后的意图可能是好的，但往往会制造不确定性。
   - 讨论强调需要更好的监管透明度，以便在维护技术领域安全的同时促进创新。
- **OpenAI 扩展视频的见解**：OpenAI 扩展视频中的亮点提到，据报道，一个 **带有 RL 的模型** 在寻找新的 **CoT** 步骤方面比人类更出色，这表明 AI 推理方法正在演进。
   - 围绕该模型的讨论包括基础设施相对于算法的重要性，以及 **自我批判 (self-critique)** 的出现作为一个值得关注的进步。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ahmad_al_dahle/status/1836839278468538629?s=46">Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：我对团队在 Llama 多模态版本上的工作感到兴奋，但在我们得到明确答复之前，我们不会在欧盟发布它。我们希望看到欧洲开发者在 AI 时代蓬勃发展——我希望……</li><li><a href="https://x.com/HaveFunWithAI/status/1836749726554702027">HaveFunWithAI (@HaveFunWithAI) 的推文</a>：o1-mini 擅长数学，参考：qwen2.4-math-72b-instruct（刚发布的 SOTA 开源数学模型）在使用代码执行和集成方法 (n=256) 时并不优于 o1-mini https://qwe...</li><li><a href="https://x.com/natolambert/status/1837232801235755174">Nathan Lambert (@natolambert) 的推文</a>：这段较长的 o1 视频中值得注意的事项（不多）：1. “带有 RL 的模型在寻找新的 CoT 步骤方面比人类更强” 2. “自我批判的出现是一个强大的时刻” 3. 提到了一个字面上的……
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1286525075557711925)** (6 messages): 

> - `LangChain v2.0 问题`
> - `LangGraph 咨询`
> - `新 Agent 平台`
> - `OpenAI Assistant 使用` 


- **修复 LangChain v2.0 中的分块输出**：一位用户在使用 **LangChain v2.0** 配合 OpenAI 流式传输时，遇到了间歇性的 **函数调用信息 (function call information)** 以分块形式输出的问题，并寻求帮助。
   - 这引发了对可能影响输出的 Bug 或配置问题的担忧。
- **对比 Ell 和 LangChain**：一位成员发起了关于 **Ell** 和 **LangChain** 之间差异及潜在对比的讨论。
   - 这表明用户对评估 AI 框架及可用选择有着持续的兴趣。
- **寻求 LangGraph 的帮助**：一位用户询问在哪里可以提交关于 **LangGraph** 的问题，这为寻求支持的用户指出了困惑点。
   - 这凸显了针对特定工具和库建立更清晰的社区支持渠道的必要性。
- **新 Agent 平台招募 Beta 测试人员**：有人发布了一个用于启动带有原生代币的 **新 Agent 平台** 的公告，邀请感兴趣的 Beta 测试人员私信了解更多详情。
   - 这反映了社区内对创新型 **Agent** 部署方案日益增长的兴趣。
- **根据最新文档使用 OpenAI Assistants**：一位成员请求关于遵循最新文档使用其自定义 **OpenAI assistant** 的指导，强调了新变化中清晰度的重要性。
   - 他们提到了 **Assistants API** 的特定功能，包括交互和工具能力。



**提到的链接**：<a href="https://python.langchain.com/v0.1/docs/modules/agents/agent_types/openai_assistants/#using-existing-assistant">OpenAI assistants | 🦜️🔗 LangChain</a>：Assistants API 允许您在自己的应用程序中构建 AI 助手。助手拥有指令，并可以利用模型、工具和知识来响应用户查询。助手……

  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/)** (1 条消息): 

degen_cap: https://x.com/degencap777/status/1836483857614541266
希望能分享您的看法
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1286412167620329644)** (6 条消息): 

> - `Moshi 模型发布`
> - `GRIN MoE`
> - `Mistral Small 发布` 


- **Moshi 模型亮相**：[Moshi 模型](https://huggingface.co/kyutai/moshiko-pytorch-bf16)已发布，被描述为一个**语音-文本基础模型**，使用独特的方法从文本生成语音。
   - 该模型支持**全双工语音对话**，显著增强了对话动态和语音识别能力。
- **GRIN MoE 以更少参数表现出色**：[GRIN MoE](https://huggingface.co/microsoft/GRIN-MoE) 模型以仅 **6.6B 激活参数**实现了令人印象深刻的高性能，尤其在编程和数学方面表现优异。
   - 通过利用 **SparseMixer-v2** 进行梯度估计，并避免传统的**专家并行 (expert parallelism)** 技术，GRIN 为 MoE 训练提供了一种新颖的方法。
- **关于 Mistral Small 发布的讨论**：成员们注意到 **Mistral Small** 的发布，但表示这仅是一个指令 (instruction) 版本。
   - 同时也提出了对其**内存占用强度**的担忧，指出这对某些用户来说是一个限制因素。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/kyutai/moshiko-pytorch-bf16">kyutai/moshiko-pytorch-bf16 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/GRIN-MoE">microsoft/GRIN-MoE · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1286433861038899362)** (5 条消息): 

> - `DSPy 中的 Bootstrapping`
> - `MathPrompt 论文`
> - `TypedPredictors 技巧` 


- **理解 Bootstrapping 的目的**：一位成员澄清说，DSPy 中 **bootstrapping** 的目的是在流水线 (pipeline) 中生成中间示例，确认成功的预测捕获了过程的完整追踪 (trace)。
   - 这基于一个假设：如果最终结果正确，尽管 LLM 本质上具有非确定性，但中间步骤也是有效的。
- **MathPrompt 简介**：一位成员分享了关于 **MathPrompt** 的有趣发现，并引用了一篇[研究论文](https://arxiv.org/pdf/2409.11445)。
   - 这为理解 Prompt 如何增强数学推理提供了进一步的扩展。
- **使用 TypedPredictors 优化 JSON 解析**：一位成员分享了处理 **TypedPredictors** 的技巧，通过模拟 JSON 解析功能来改进输出预处理。
   - 他们的方案包括删除不必要的文本、修复无效的转义序列，并记录来自其 [GitHub Gist](https://gist.github.com/tkellogg/246d7928b2fc26821db582be583d8b7a) 的解析失败日志。



**提到的链接**：<a href="https://gist.github.com/tkellogg/246d7928b2fc26821db582be583d8b7a">fix-json.py</a>：GitHub Gist：即时分享代码、笔记和片段。

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1286665206759686266)** (2 条消息): 

> - `招聘 LLM 工程师`
> - `多语言翻译`
> - `Qwen 2.5`
> - `实时金融通信` 


- **利用 LLM 变革全球金融**：一家金融科技 (FinTech) 初创公司正在寻求一位才华横溢的 **LLM 工程师**，进行为期一周的冲刺，以使用 **LLama 3.1** 或 **Qwen2** 模型增强其多语言实时翻译服务。
   - 该项目旨在打破全球金融中的语言障碍，有望对全球数百万笔交易的处理方式做出重大贡献。
- **探索 Qwen 2.5 的多语言能力**：一位用户建议考虑 **Qwen 2.5**，强调了其在多语言功能方面的潜力，认为它可能符合项目需求。
   - 这一见解可以指导在所选 LLM 之外增强 **Whisper 模型** 的方向。


  

---



---



---



---



---



---



{% else %}


> 为了便于邮件阅读，完整的频道明细已被截断。
> 
> 如果您想查看完整明细，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}