---
companies:
- meta-ai-fair
- groq
- fireworks
date: '2024-07-24T00:13:31.329222Z'
description: '**Meta AI** 发布了 **Llama 3.1**，其中包括一个 **405B（4050亿）参数模型**，该模型触及了诸如**《欧盟人工智能法案》**和
  **SB 1047** 等监管考量。该模型在**代码**、**数学**、**多语言**、**长上下文**和**工具使用**的微调中大量采用了**合成数据**技术，并利用来自
  Llama 2 的合成偏好数据进行了 **RLHF**（基于人类反馈的强化学习）。


  此次发布与各大推理服务提供商协同进行，其中 **Groq** 展示了每秒 **750 个 token** 的推理速度，而 **Fireworks** 在价格方面具有领先优势。更新后的许可协议明确允许生成合成数据，这标志着开源前沿级大语言模型迈出了重要一步，也体现了自三月以来成本效益的显著提升。'
id: 829b011e-ad99-4da3-87a9-13e0bf8ccffd
models:
- llama-3-405b
- llama-3-1
- llama-3
original_slug: ainews-llama-31-the-synthetic-data-model
people:
- bindureddy
- thomas
title: Llama 3.1：合成数据模型
topics:
- synthetic-data
- fine-tuning
- reinforcement-learning
- multilinguality
- long-context
- tool-use
- code-generation
- math
- model-licensing
- inference-speed
- model-deployment
---

<!-- buttondown-editor-mode: plaintext -->**Synthetic Data is all you need.**

> 2024年7月22日至7月23日的 AI 新闻。我们为您检查了 7 个 subreddits、[384 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 30 个 Discord（474 个频道和 5128 条消息）。预计节省阅读时间（按 200wpm 计算）：473 分钟。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

**Llama 3.1 来了！** ([网站](https://llama.meta.com/)、[视频](https://x.com/aiatmeta/status/1815766327463907421?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)、[论文](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)、[代码](https://github.com/meta-llama/llama-models)、[模型](https://ai.meta.com/blog/meta-llama-3-1/)、[Zuck](https://news.ycombinator.com/item?id=41046773)、[Latent Space 播客](https://x.com/latentspacepod/status/1815781241398104085))。包括 405B 模型，它同时触发了 [欧盟 AI 法案 (EU AI act)](https://x.com/deanwball/status/1815826885663658445?s=46) 和 [SB 1047](https://x.com/martin_casado/status/1815865505204576389)。[完整论文](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) 包含了你想要的所有 Frontier Model 对比：

 
![image.png](https://assets.buttondown.email/images/12252d7e-4c23-4cf4-8afa-1ea166b0ff26.png?w=960&fit=max)
 

我们假设您已经阅读了 [昨天](https://buttondown.email/ainews/archive/ainews-llama-31-leaks/) 的头条新闻。它目前还没有在 LMsys 上线，但 [SEAL](https://x.com/summeryue0/status/1815776426999877643) 和 [Allen AI 的 ZeroEval](https://x.com/billyuchenlin/status/1815841947468353700?s=46) 的独立评估结果令人期待（尽管存在一些 [分歧](https://x.com/hrishioa/status/1815811349777375649?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)）。这是一次跨越行业内几乎所有推理提供商的协同发布，当然也包括 Groq 展示的 [750tok/s](https://x.com/JonathanRoss321/status/1815777714642858313) 惊人推理速度演示。推理定价也已公布，[Fireworks 处于领先地位](https://x.com/HamelHusain/status/1815852454027940135)。

虽然人们普遍推测 8B 和 70B 是 405B 的 [“离线蒸馏 (offline distillations)”](https://x.com/kalomaze/status/1815797116104556565)，但 Llama 3.1 中的 Synthetic Data 元素比预期的要多得多。论文中明确指出：

- **代码 SFT**：[3 种 Synthetic Data 方法](https://x.com/swyx/status/1815771160841425113) 用于 405B 的自我引导（bootstrapping），包括代码执行反馈、编程语言翻译和文档回译（docs backtranslation）。
- **数学 SFT**：
![https://pbs.twimg.com/media/GTLqFD9aMAAwYUQ?format=png&name=900x900](https://pbs.twimg.com/media/GTLqFD9aMAAwYUQ?format=png&name=900x900)

- **多语言 SFT**：“为了在非英语语言中收集更高质量的人类标注，我们通过从预训练运行中分支出来，并继续在由 90% 多语言 Token 组成的数据混合上进行预训练，从而训练出一个多语言专家模型。”
- **长上下文 SFT**：“由于阅读冗长上下文的性质既乏味又耗时，让模型人类标注此类示例在很大程度上是不切实际的，因此**我们主要依靠 Synthetic Data 来填补这一空白。** 我们使用早期版本的 Llama 3，根据关键的长上下文用例生成 Synthetic Data：(可能是多轮的) 问答、长文档摘要以及代码库推理，并在下文中进行更详细的描述。”
- **工具使用 SFT**：针对 Brave Search、Wolfram Alpha 和 Python 解释器（一个新的特殊 [`ipython`](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1) 角色）进行了训练，支持**单次、嵌套、并行和多轮函数调用 (function calling)**。
- **RLHF**：DPO 偏好数据广泛应用于 Llama 2 的生成结果。正如 [Thomas 在播客中所说](https://www.latent.space/p/llama-3)：“Llama 3 的后训练（post-training）基本上没有任何人类编写的答案……**它只是利用了来自 Llama 2 的纯 Synthetic Data。**”

最后但同样重要的一点是，Llama 3.1 [获得了许可证更新](https://x.com/AIatMeta/status/1815766335219249513)，明确允许将其用于 Synthetic Data 生成。

我们终于拥有了一个 [Frontier 级别的开源 LLM](https://x.com/karpathy/status/1815842603377779140)。值得注意的是，自三月份以来，整个行业在 [单位智能成本 (cost per intelligence)](https://x.com/swyx/status/1815892458519289946/photo/1) 方面取得了巨大的进步，而且未来只会变得更好。

 
![image.png](https://assets.buttondown.email/images/f446a928-ef16-41f6-b461-efde87ac6ecf.png?w=960&fit=max)
 

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**Meta AI**

- **Llama 3.1 405B 模型**：[@bindureddy](https://twitter.com/bindureddy/status/1815443198459990098) 指出 **Llama-3.1 405B 的基准测试在 Reddit 上泄露，表现优于 GPT-4o**。[@Teknium1](https://twitter.com/Teknium1/status/1815443354735571232) 分享了一张对比 **Llama-3.1 405/70/8b 与 GPT-4o 的图片，展示了目前已开源的 SOTA 前沿模型**。[@abacaj](https://twitter.com/abacaj/status/1815484377167466997) 提到 **Meta 训练和发布 open weights 模型速度比 OpenAI 发布闭源模型还要快**。
- **Llama 3 70B 性能**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1815457648814194694) 强调 **70B 模型在体积缩小 6 倍的情况下达到了 GPT-4 的水平**。这是 base 模型，而非经过 instruct tuned 的模型。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1815458893759475974) 指出 **70B 模型正在蚕食 405B 的领地，而大模型的作用将是用于蒸馏 (distill)**。
- **开源模型进展**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1815486069883752862) 称之为 **“老当益壮” (Old Man Strength) 开源模型的黎明**。[@abacaj](https://twitter.com/abacaj/status/1815484683452649687) 提到 **OpenAI 的模型并没有显著提升，因此 Meta 的模型将在 open weights 领域追赶上来**。

**AI 助手与 Agent**

- **Omnipilot AI**：[@svpino](https://twitter.com/svpino/status/1815360679072653704) 介绍了 **@OmnipilotAI，这是一款 AI 应用，可以在任何你能打字的地方输入，并利用屏幕内容的完整 context**。它适用于所有 macOS 应用程序，并使用 **Claude Sonet 3.5, Gemini 和 GPT-4o**。示例包括回复电子邮件、自动补全终端命令、完成文档以及发送 Slack 消息。
- **Mixture of agents**：[@llama_index](https://twitter.com/llama_index/status/1815518744829169807) 分享了 @1littlecoder 的视频，介绍了 **“mixture of agents”——使用多个本地语言模型，性能可能超越单个模型**。其中包括使用 **LlamaIndex 和 Ollama** 实现的教程，在分层架构中结合了 **Llama 3, Mistral, StableLM** 等模型。
- **Agent 规划**：[@hwchase17](https://twitter.com/hwchase17/status/1815404685500821950) 讨论了 **Agent 规划的未来**。虽然模型的改进会有所帮助，但**优秀的 prompting 和自定义认知架构始终是使 Agent 适应特定任务所必需的**。

**基准测试与评估**

- **LLM-as-a-Judge**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1815405425866518846) 概述了 **LLM-as-a-Judge，即使用更强大的 LLM 来评估另一个 LLM 输出的质量**。关键要点包括使用足够能力的 judge 模型、prompt 设置（pairwise vs pointwise）、提高 pointwise 评分稳定性、chain-of-thought prompting、temperature 设置以及考虑位置偏差 (position bias)。
- **事实不一致检测**：[@sophiamyang](https://twitter.com/sophiamyang/status/1815389350013899106) 分享了关于使用 @weights_biases **微调和评估 @MistralAI 模型以检测文本摘要中的事实不一致和幻觉 (hallucinations)** 的指南。该工作基于 @eugeneyan 的研究，是 Mistral Cookbook 的一部分。
- **复杂问题回答**：[@OfirPress](https://twitter.com/OfirPress/status/1815379379188293872) 推出了一项新的基准测试，用于**评估 AI 助手回答复杂自然语言问题的能力**，例如“我附近有哪些餐厅提供 25 美元以下的素食和无麸质主菜？”，目标是推动开发更好的助手。

**框架与工具**

- **DSPy**：[@lateinteraction](https://twitter.com/lateinteraction/status/1815423177272824022) 分享了一篇论文，发现 **DSPy 优化器在优化权重和 prompt 之间交替进行，比仅优化其中之一可获得高达 26% 的提升**。[@lateinteraction](https://twitter.com/lateinteraction/status/1815423187418763308) 指出**基于模块化 NLP 程序的组合式优化器是未来**，并建议组合使用 BootstrapFewShot 和 BootstrapFinetune 优化器。
- **LangChain**：[@hwchase17](https://twitter.com/hwchase17/status/1815442482290978932) 指向了**新的 LangChain 更新日志 (Changelog)，以便更好地沟通他们发布的所有内容**。[@LangChainAI](https://twitter.com/LangChainAI/status/1815439685117993349) 强调了 **LangGraph.js 中无需额外配置的无缝 LangSmith 追踪**，使得利用 LangSmith 的功能构建 Agent 变得更加容易。
- **EDA-GPT**：[@LangChainAI](https://twitter.com/LangChainAI/status/1815426831430123585) 介绍了 **EDA-GPT，一个开源的数据分析助手**，可简化数据探索、可视化和洞察。它具有可配置的 UI 并与 LangChain 集成。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 在本地运行大语言模型**

- **如果你必须问如何本地运行 405B** ([Score: 287, Comments: 122](https://reddit.com//r/LocalLLaMA/comments/1e9nybe/if_you_have_to_ask_how_to_run_405b_locally/))：该帖子探讨了**在本地运行 4050 亿参数模型的不可能性**。文章直言不讳地指出，如果有人需要询问如何做到这一点，那么他们根本无法实现，暗示这项任务超出了典型消费级硬件的能力范围。

- **请为我们这些 GPU 贫民分享你们的 LLaMA 3.1 405B 使用体验** ([Score: 52, Comments: 30](https://reddit.com//r/LocalLLaMA/comments/1e9t8n2/please_share_your_llama_31_405b_experiences_below/))：该帖子请求用户分享在本地运行 **LLaMA 3.1 405B** 的经验，特别是针对那些 GPU 资源有限的用户。虽然正文中没有提供具体的经验，但标题表明了人们对了解这种超大语言模型在消费级硬件上的表现，以及 GPU 性能较低的用户所面临的挑战有着浓厚兴趣。

- **我希望当初那个笨拙的自己能早点知道的 Ollama 网站“专业技巧”：** ([Score: 72, Comments: 24](https://reddit.com//r/LocalLLaMA/comments/1e9hju5/ollama_site_pro_tips_i_wish_my_idiot_self_had/))：该帖子重点介绍了使用 **Ollama 网站**下载和运行 AI 模型的几个**“专业技巧”**。关键功能包括：通过“Tags”链接访问模型的**不同量化版本 (quantizations)**；通过搜索框访问隐藏的**模型类型排序功能**；在模型表格中查找**最大上下文窗口大小 (max context window sizes)**；以及使用顶部搜索框访问更广泛的模型列表（包括用户提交的模型）。作者已经使用 Ollama **6-8 个月**，分享这些见解是为了帮助其他可能忽略了这些功能的人。

**主题 2. LLaMA 3.1 405B 模型发布与基准测试**

- **[Azure Llama 3.1 基准测试] (https://github.com/Azure/azureml-assets/pull/3180/files)** ([Score: 349, Comments: 268](https://reddit.com//r/LocalLLaMA/comments/1e9hg7g/azure_llama_31_benchmarks/))：Microsoft 发布了 **Azure Llama 3.1** 的基准测试结果，显示出比之前版本的改进。该模型在 **MMLU** 基准测试中获得了 **94.4%** 的分数，超过了 GPT-3.5 并接近 GPT-4 的性能。Azure Llama 3.1 在**代码生成**和**多轮对话**方面也表现出强大的能力，使其成为 AI 模型领域中极具竞争力的选择。

- **[Llama 3.1 405B, 70B, 8B 指令微调版基准测试] (https://i.redd.it/62ov7fzck5ed1.jpeg)** ([Score: 137, Comments: 28](https://reddit.com//r/LocalLLaMA/comments/1e9sinx/llama_31_405b_70b_8b_instruct_tuned_benchmarks/))：Meta 发布了 **LLaMA 3.1**，包含 **405B**、**70B** 和 **8B** 参数的模型，所有模型都经过了指令微调 (instruct-tuned)。**405B** 模型在各种基准测试中达到了业界领先 (state-of-the-art) 的水平，在多项任务上超越了 **GPT-4**，而 **70B** 模型在面对 **Claude 2** 和 **GPT-3.5** 时也表现出极具竞争力的结果。

- **LLaMA 3.1 405B 基座模型开放下载** ([Score: 589, Comments: 314](https://reddit.com//r/LocalLLaMA/comments/1e98zrb/llama_31_405b_base_model_available_for_download/))：大小为 **764GiB** (~820GB) 的 **LLaMA 3.1 405B 基座模型 (base model)** 现已开放下载。可以通过 [Hugging Face 链接](https://huggingface.co/cloud-district/miqu-2)、**磁力链接**或 **种子文件**获取该模型，来源归功于 4chan 论坛的一个帖子。
  - 用户讨论了运行 **405B 模型**的可能性，建议包括使用 **2x A100 GPU** (160GB VRAM) 配合低量化版本，或者在 **Hetzner** 上以每月 200-250 美元的价格租用拥有 **数 TB RAM** 的服务器，在 Q8/Q4 量化下可能达到 **每秒 1-2 个 token**。
  - 关于在 **Nintendo 64** 上运行该模型或“**下载更多 VRAM**”的幽默评论引发了对硬件限制的讨论。用户推测，消费级 GPU 可能需要 **5-10 年**才能处理如此巨大的模型。
  - 一些人质疑泄露的真实性，指出其与之前的泄露（如 **Mistral medium (Miqu-1)**）有相似之处。另一些人则争论这是否是 **Meta** 在正式发布前为了营销目的而进行的有意“泄露”。


**主题 3. 分布式与联邦 AI 推理**

- **LocalAI 2.19 发布！支持 P2P、自动发现、联邦实例（Federated instances）和分片模型加载（sharded model loading）！** ([Score: 52, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1e9inr9/localai_219_is_out_p2p_autodiscovery_federated/)): LocalAI 2.19 引入了**联邦实例**和**通过 P2P 进行的分片模型加载**，允许用户跨多个节点结合 **GPU 和 CPU 算力**来运行大型模型，而无需昂贵的硬件。该版本包括一个新的 **P2P 仪表板**用于轻松设置联邦实例，在二进制发行版中集成了 **Text-to-Speech**，并改进了 **WebUI**、**安装脚本**以及支持 **embeddings** 的 **llama-cpp 后端**。

- **Ollama 已更新以适配 Mistral NeMo，现已提供正式下载** ([Score: 63, Comments: 13](https://reddit.com//r/LocalLLaMA/comments/1e9wsv6/ollama_has_been_updated_to_accommodate_mistral/)): **Ollama** 已更新，增加了对 **Mistral NeMo** 模型的支持，现已开放下载。用户报告称，在拥有 **16GB VRAM 的 4060 Ti GPU** 上，**NeMo** 的表现比 **Llama 3 8b** 和 **Gemma 2 9b** 模型更快、更好，并指出这是继 Gemma 发布后本地 AI 模型的又一重大进展。
  - 用户称赞了 **Mistral NeMo 12b** 的性能，其中一位指出它“**完美通过（NAILED）**”了 **48k context** 测试，并表现出流利的法语水平。然而，随着即将发布的 **Llama 3.1 8b**，它的使用寿命可能较短。
  - 一些用户对下载该模型表示兴奋，而另一些人则认为它与 **tiger-gemma2** 相比令人失望，特别是在多轮对话中的指令遵循方面。
  - **Mistral NeMo** 的发布时机被描述为对开发者来说“非常遗憾”，因为它紧随其他重要模型的发布之后。


**主题 4. 新 AI 模型发布与泄露**

- **Nvidia 发布了两个新的基础模型：Minitron 8B 和 4B，是 Nemotron-4 15B 的剪枝版本** ([Score: 69, Comments: 3](https://reddit.com//r/LocalLLaMA/comments/1e9vaqy/nvidia_has_released_two_new_base_models_minitron/)): **Nvidia** 发布了 **Minitron 8B 和 4B**，这是其 **Nemotron-4 15B** 模型的剪枝（pruned）版本，与从头开始训练相比，它们需要的 **训练 token 减少了多达 40 倍**，并节省了 **1.8 倍的计算成本**。这些模型在 **MMLU 评分**上比从头训练的模型提高了多达 **16%**，性能与 **Mistral 7B** 和 **Llama-3 8B** 等模型相当，仅用于研究和开发目的。
  - **剪枝模型（Pruned models）** 在 AI 领域并不常见，**Minitron 8B 和 4B** 是值得注意的例外。这种稀缺性引起了研究人员和开发者的兴趣。
  - **剪枝（pruning）** 的概念在直觉上与 **量化（quantization）** 相似，尽管一些用户推测对剪枝后的模型进行量化可能会对性能产生负面影响。
  - **AWQ** (Activation-aware Weight Quantization) 被拿来与剪枝进行比较，剪枝可能通过减少整体模型维度而不仅仅是压缩位表示来提供更大的收益。
- **llama 3.1 download.sh 提交记录** ([Score: 66, Comments: 18](https://reddit.com//r/LocalLLaMA/comments/1e9vjkc/llama_31_downloadsh_commit/)): **Meta Llama GitHub 仓库**最近的一次提交表明 **LLaMA 3.1** 可能即将发布。该提交可在 [https://github.com/meta-llama/llama/commit/12b676b909368581d39cebafae57226688d5676a](https://github.com/meta-llama/llama/commit/12b676b909368581d39cebafae57226688d5676a) 查看，其中包含一个 **download.sh** 脚本，可能预示着模型分发的准备工作。
  - 提交内容显示了 **Base 和 Instruct** 版本的 **405B 模型**，变体标记为 **mp16**、**mp8** 和 **fp8**。用户推测 “mp” 可能代表 **混合精度（mixed precision）**，暗示了针对打包混合精度模型的量化感知训练（quantization-aware training）。
  - 关于 Instruct 模型中 **fb8** 标签的讨论得出结论，这很可能是 **fp8** 的拼写错误，文件中的证据支持了这一点。用户对分析权重精度以实现更好的低比特量化的潜力感到兴奋。
  - 提交作者 **samuelselvan** 此前曾向 Hugging Face 上传过一个被认为可疑的 **LLaMA 3.1 模型**。用户对 Meta 直接发布模型的量化版本表现出极大的热情。

- **[Llama 3 405b 在 4chan 上泄露了？太期待了！还有一天！！](https://www.reddit.com/gallery/1e99uaa)** ([Score: 210, Comments: 38](https://reddit.com//r/LocalLLaMA/comments/1e99uaa/llama_3_405b_leaked_on_4chan_excited_for_it_just/)): 关于 **LLaMA 3.1 405B** 模型在 **4chan** 上泄露的消息正在流传，但这些说法尚未经过证实，且很可能是虚假的。据称的泄露发生在预期官方发布的前一天，引发了对其真实性的怀疑。对待此类泄露信息应保持谨慎，并等待来自 **Meta** 或其他可靠来源的官方确认。
  - 据报道，一个包含该模型的 **HuggingFace 仓库** 在 **2 天前** 曾公开可见，这可能让潜在的泄露者获得了访问权限。用户对该模型的 **70B 和 8B** 版本表现出了浓厚兴趣。
  - 相比于等待官方发布，一些用户对没有对齐 (alignment) 或护栏 (guardrails) 的 **纯基础模型 (base model)** 更感兴趣。**/r/LocalLLaMA** 上的另一个帖子讨论了所谓的 **405B base model** 下载。
  - 用户正尝试运行该模型，其中一人计划使用 **7x24GB GPU 配置** 将其转换为 **4-bit GGUF 量化**。另一位用户分享了他们尝试运行该模型的 [YouTube 链接](https://www.youtube.com/watch?v=LoUbZt9gtZs)。

## AI Reddit 全摘要

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. OpenAI 的全民基本收入 (UBI) 实验结果**

- [/r/singularity] **[OpenResearch 团队发布了他们 UBI 研究的首个结果 (OpenAI)](https://www.openresearchlab.org/findings)** ([Score: 280, Comments: 84](https://reddit.com//r/singularity/comments/1e9jppo/the_openresearch_team_releases_the_first_result/)): **OpenResearch**（**OpenAI** 的一个团队）发布了其 **全民基本收入 (UBI) 研究** 的初步结果。这项在 **肯尼亚村庄** 进行的研究发现，**1,000 美元的现金转移** 产生了显著的积极影响，包括资产增加了 **400 美元**，以及挨饿的可能性降低了 **40%**。这些发现为支持通过直接现金转移缓解贫困的有效性提供了更多证据。

- [/r/OpenAI] **[OpenAI 创始人 Sam Altman 秘密向随机人群发放了 4500 万美元——作为一项实验](https://www.forbes.com.au/news/innovation/openai-founder-sam-altman-gave-thousands-of-people-free-money/)** ([Score: 272, Comments: 75](https://reddit.com//r/OpenAI/comments/1e984lw/openai_founder_sam_altman_secretly_gave_out_45/)): **Sam Altman 的 4500 万美元 UBI 实验揭晓**：这位 **OpenAI 创始人** 秘密地向美国两个州的 **3,000 人** 发放了 **4500 万美元**，作为 **全民基本收入 (UBI) 实验** 的一部分。参与者每月领取 **1,000 美元**，持续长达 **五年**。该研究旨在评估无条件现金转移对受助者的生活质量、时间利用和财务状况的影响。
  - **3,000 名参与者** 每月领取 **1,000 美元或 50 美元**，持续长达 **五年**，许多 Redditor 表达了加入该实验的愿望。该研究针对 **德克萨斯州和伊利诺伊州** 城市、郊区和农村地区，家庭收入低于 **联邦贫困线 300%** 的 **21-40 岁** 人群。
  - 一些用户批评该实验是 **科技亿万富翁** 的公关手段 (PR move)，旨在缓解人们对 **AI 驱动的失业** 的担忧；而另一些人则认为，鉴于政府在这一问题上行动迟缓，私人的 UBI 实验是必要的。
  - 讨论中出现了关于就业未来的看法，一些人预测由于 AI 的进步，失业率将突然飙升，当各行各业的传统工作变得稀缺时，可能会导致 **UBI 的广泛实施**。

**主题 4. AI 研究员对 AGI 时间线的预测**

- [/r/singularity] **[前 OpenAI 研究员的预测](https://i.redd.it/4fmq8fb6e2ed1.png)** ([Score: 243, Comments: 151](https://reddit.com//r/singularity/comments/1e9d5pb/former_openai_researcher_predictions/)): **前 OpenAI 研究员预测 AGI 时间线**：前 OpenAI 研究员 Paul Christiano 估计，到 **2030 年有 20-30% 的机会实现 AGI**，到 **2040 年有 60-70% 的机会**。他认为目前的 AI 系统距离 AGI 还很远，但在推理 (reasoning) 和规划 (planning) 等领域的快速进展可能会在未来几年带来重大突破。

- [/r/singularity] **["顶尖秘密实验室的大多数员工都在认真地根据 2027 年数字神灵的出现来规划他们的生活"](https://twitter.com/jam3scampbell/status/1815311644853256315)** ([Score: 579, Comments: 450](https://reddit.com//r/singularity/comments/1e9flnw/most_of_the_staff_at_the_secretive_top_labs_are/)): **AI 研究人员预见数字神灵**：根据该帖子，据报道，**顶尖秘密 AI 实验室**的**大多数员工**正根据 2027 年预期出现的**数字神灵**来**规划他们的生活**。虽然没有提供具体的来源或证据，但这一说法表明 AI 研究人员在看待未来 AI 系统的潜在能力和影响方面，心态发生了重大转变。

- [/r/singularity] **[Nick Bostrom 表示，在 AI 能够完成人脑所能做的一切事情后不久，它将学会更好、更快地完成这些事情，而人类智能将变得过时](https://v.redd.it/q9qjbqh707ed1)** ([Score: 323, Comments: 258](https://reddit.com//r/singularity/comments/1e9yhzx/nick_bostrom_says_shortly_after_ai_can_do_all_the/)): **Nick Bostrom** 警告称 **AI 将以快速且变革性的方式超越人类智能**。他预测，一旦 AI 能够匹配人脑的能力，它将迅速在所有领域超越人类，使人类智能变得过时。这种加速的进步暗示了潜在的**智能爆炸**，即 AI 的能力迅速超过人类，从而导致重大的社会和生存影响。
  - **Nick Bostrom** 的警告引发了辩论，一些人因 AI 能够连接 **100k GPUs** 而称其为“**显而易见的废话**”，而另一些人则考虑到目前关于 AI 能力的持续争论，捍卫了他信息的重要性。
  - 讨论范围从幽默的梗图到关于“**被解决的世界**”的哲学思考，一位用户描述了一个假设的 **2055** 年场景，即 **AGI** 和 **ASI** 带来了医学突破、全沉浸式 VR 和模拟现实。
  - 一些用户对 AI 解决海洋退化等重大问题表示乐观，而另一些人则对潜在的负面结果表示担忧，例如人口减少的情景，或由于阻力而实施必要变革所面临的挑战。


**主题 5. AI 训练基础设施的新进展**

- [/r/singularity] **[Elon 表示，今天一个模型已开始在世界上最新且最强大的 AI 集群上进行训练](https://x.com/elonmusk/status/1815325410667749760)** ([Score: 239, Comments: 328](https://reddit.com//r/singularity/comments/1e9ahwl/elon_says_that_today_a_model_has_started_training/)): **Elon Musk 宣布突破性的 AI 进展**：一个新的 AI 模型已开始在 Musk 声称的**全球最强大的 AI 集群**上进行训练。这一公告标志着 AI 计算能力的重大里程碑，有可能推高大语言模型训练和性能的边界。

---

# AI Discord 摘要

> 摘要之摘要的摘要

**1. LLM 进展与基准测试**

- **Llama 3.1 发布热潮**：**Llama 3.1** 模型（包括 **8B** 和 **405B**）现已发布，在社区中引发了巨大反响。用户分享了他们的经验和故障排除技巧，以解决如本地运行模型以及在微调期间管理高 loss 值等问题。
   - 社区赞扬了该模型的性能，一些人指出它在基准测试中超越了现有的私有模型，而另一些人则强调了实际部署中的挑战。
- **Meta 对开源 AI 的承诺**：Meta 发布了包含 **405B** 等模型的 **Llama 3.1**，推向了开源 AI 的新边界，提供 **128K token context** 并支持多种语言。此举符合 Mark Zuckerberg 通过开放协作促进创新的愿景。
   - 社区讨论了此次发布的战略意义，强调了该模型挑战 **GPT-4** 等顶级闭源替代方案的潜力。
    


**2. 优化 LLM 推理与训练**

- **高效微调技术讨论**：**ReFT 论文**介绍了一种比 LoRA 参数效率高 **15x-60x** 的方法，该方法通过作用于 **residual stream**，在结合训练任务与优化参数方面提供了灵活性。
   - 社区成员与主作者进行了交流，以了解其实际应用，强调了该方法在提高微调效率方面的潜力。
- **GPU 兼容性挑战**：用户报告了 Linux 上的 GPU 检测问题，特别是 **Radeon RX5700XT**，引发了对 **RDNA 1 support** 的关注。讨论强调了正确配置对于 GPU 识别的重要性。
   - 一些用户确认扩展包未能解决问题，表明需要进一步的故障排除以及开发者的潜在更新。
    


**3. 开源 AI 框架与社区努力**

- **LlamaIndex 关于高效文档检索的网络研讨会**：即将举行的网络研讨会将于本周五上午 **9am PT** 讨论**使用 Vision Language Models 进行高效文档检索**。参与者可以报名学习文档处理领域的前沿技术。
   - 研讨会旨在探索 ColPali 使用 Vision Language Models 嵌入 **page screenshots** 的创新方法，从而增强对复杂文档的检索性能。
- **Magpie 论文引发辩论**：成员们辩论了 **Magpie 论文**中见解的实用性，质疑生成的指令是提供了实质性的效用，还是仅仅是一个 *party trick*（噱头）。
   - 讨论突显了对指令生成新兴技术的持续评估，反映了社区对新研究的批判性参与。
    


**4. 多模态 AI 与生成模型创新**

- **UltraPixel 创建高分辨率图像**：**UltraPixel** 是一个能够生成极高质量细节的高分辨率图像的项目，通过专注于 **clarity**（清晰度）和 **detail**（细节）推向了图像生成的边界。
   - 社区对该项目的功能表现出浓厚兴趣，探索了其潜在应用，并分享了项目链接以供进一步参与。
- **Idefics2 和 CodeGemma：新型多模态模型**：**[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** 专注于提升聊天交互体验，而 **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** 则精进了编程能力。
   - 这些模型代表了多模态 AI 的重大进步，社区讨论了它们在增强用户交互和编程任务方面的潜力。

---

# 第一部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **NuminaMath 数据集发布**：**NuminaMath** 数据集已发布，包含约 **1M** 个数学问题-解答对，曾助力获得 AI Math Olympiad 的 **Progress Prize**。其中包括专为 **Chain of Thought** 和 **Tool-integrated reasoning** 设计的子集，显著提升了在数学竞赛基准测试中的表现。
   - 在这些数据集上训练的模型展示了*同类最佳性能*，超越了现有的私有模型。请在 [🤗 Hub](https://huggingface.co/collections/AI-MO/numinamath-6697df380293bcfdbc1d978c) 查看发布内容。
- **Llama 3.1 发布引发热潮**：近期 **Llama 3.1** 的发布引发了广泛关注，**8B** 和 **405B** 等模型现已开放测试。用户正积极分享使用经验，包括在本地运行模型时的故障排除。
   - 社区参与并分享了各种见解，并为早期采用者面临的运行挑战提供支持。
- **模型微调中的挑战**：在针对特定任务微调模型时，高 Loss 值和性能问题引发了一些挫败感。社区建议了一些资源和实践方法来有效应对这些挑战。
   - 知识交流旨在改进模型训练和评估流程。
- **UltraPixel 创作高分辨率图像**：**UltraPixel** 作为一个能够生成极高质量细节的高分辨率图像项目被展示。该计划专注于**清晰度**和**细节**，推向了图像生成的边界。
   - 访问[此链接](https://huggingface.co/spaces/gokaygokay/UltraPixel)查看该项目。
- **对分割技术的兴趣**：另一位成员对在使用扩散模型去除背景的同时，如何应用有效的**分割技术 (segmentation techniques)** 表示出兴趣。他们正在寻求关于成功方法或模型的建议。
   - 对话旨在探索图像分割的最佳实践。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Magpie 论文引发辩论**：成员们讨论了 [Magpie 论文](https://arxiv.org/abs/2406.08464) 中的见解是否具有实质性效用，还是仅仅是一个*噱头 (party trick)*，重点关注生成指令的**质量**和**多样性**。
   - 这一探讨突显了对指令生成领域新兴技术的持续评估。
- **ReFT 论文揭示高效微调方法**：[ReFT 论文](https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/) 的第一作者澄清说，该方法通过在 **residual stream** 上操作，比 LoRA 的参数效率高出 **15x-60x**。
   - 这为结合训练任务与优化参数提供了灵活性，强化了高效微调策略的相关性。
- **Bud-E 语音助手获得关注**：**Bud-E 语音助手** 演示强调了其开源潜力，目前已针对 **Ubuntu** 进行了优化，Christoph 正在领导黑客松活动以吸引社区参与。
   - 此类协作努力旨在促进志愿者的贡献，扩大项目范围。
- **Llama 3.1 在基准测试中表现出色**：Llama 3.1 405B Instruct-Turbo 在 GSM8K 上排名第一，在逻辑推理方面与 GPT-4o 旗鼓相当，尽管在 MMLU-Redux 上的表现似乎稍弱。
   - 这种差异强化了在不同基准数据集上进行全面评估的重要性。
- **推荐 Kuzu 图数据库**：成员们推荐了与 **LlamaIndex** 集成的 **Kuzu GraphStore**，特别是其 [MIT license](https://github.com/kuzudb/kuzu) 确保了开发者的可访问性。
   - 采用先进的图数据库功能为数据管理提供了可行的替代方案，尤其是在复杂系统中。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 性能见解**：用户强调了 **Llama 3.1** 模型之间的性能差异，指出运行大型模型需要大量的 GPU 资源，特别是 **405B** 变体。
   - 一位用户幽默地评论说，为了有效运行这些模型，需要 *一个小国家的电力供应*。
- **模型下载困扰**：由于 DNS 问题以及 **Llama 3.1** 的流行导致 **Hugging Face** 流量激增，几位成员指出下载模型存在困难。
   - 一位用户建议在应用内提供禁用 **IPv6** 的选项，以缓解其中的一些下载挑战。
- **GPU 兼容性挑战**：新的 Linux 用户报告了 LM Studio 在识别 **Radeon RX5700XT** 等 GPU 时遇到麻烦，引发了对 **RDNA 1 support** 的担忧。
   - 讨论强调了正确配置 GPU 识别的重要性，一些用户确认 **extension packs** 并不能解决这些问题。
- **Llama 3.1 提供新特性**：**Llama 3.1** 已经发布并带来了改进，包括高达 **128k** 的上下文长度，可在 Hugging Face 上下载。
   - 鼓励用户探索该模型增强的性能，特别是针对内存密集型任务。
- **更新后 ROCm 性能问题**：一位用户指出，更新到 **ROCm 0.2.28** 导致推理速度显著变慢，其 **7900XT** 的功耗降至 **150w**。
   - 回退到 **0.2.27** 恢复了性能，这表明需要明确新版本中的功能变化。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Llama 3.1 405B 发布与 API 集成**：备受期待的 **Llama 3.1 405B** 模型现已在 Perplexity 上线，可与 **GPT-4o** 和 **Claude Sonnet 3.5** 媲美，增强了平台的 AI 能力。
   - 用户询问了将 **Llama 3.1 405B** 添加到 Perplexity API 的情况，询问是否很快会推出，并分享了关于模型性能的各种体验。
- **对 Llama 3.1 性能的担忧**：用户报告了 **Llama 3.1 405B** 的问题，包括回答重复以及理解亚洲符号方面的困难，导致许多人考虑回退到 **Claude 3.5 Sonnet**。
   - 对比评估表明，虽然 **Llama 3.1** 是一个飞跃，但 **Claude** 在速度和编程任务方面仍具有优势。
- **探索 Dark Oxygen 和 Mercury's Diamonds**：最近的一项讨论集中在 **Dark Oxygen** 上，提出了关于其对大气研究和生态平衡影响的问题。
   - 此外，关于 **Diamonds on Mercury** 的见解也浮出水面，引发了人们对可能导致其形成的地球物理过程的兴趣。
- **Beach-Cleaning Robots 大放异彩**：**beach-cleaning robot technology** 的创新受到关注，展示了有效应对海洋污染的努力。
   - 这些机器人对 **marine ecosystems** 的影响是讨论的一个重点，会上分享了来自试验的实时数据。
- **Perplexity API 的 DSGVO 合规性**：用户对 **Perplexity API** 是否符合 DSGVO 标准表示担忧，寻求数据保护合规性方面的澄清。
   - 对话中分享了引用 GDPR 合规性的 [服务条款](https://www.perplexity.ai/hub/legal/perplexity-api-terms-of-service)。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **AI 模型排名，Kolors 位居榜首**：在最新的讨论中，用户对 AI 模型进行了排名，**Kolors** 因其出色的速度和性能位列第一，随后是 **Auraflow**、**Pixart Sigma** 和 **Hunyuan**。
   - Kolors 的表现非常符合用户对 **SD3** 的预期。
- **训练 Lycoris 遇到兼容性障碍**：讨论集中在使用 **ComfyUI** 和 **Kohya-ss** 等工具训练 **Lycoris**，用户对需要 Python **3.10.9 或更高版本**的兼容性要求表示沮丧。
   - 期待 **Onetrainer** 可能会发布更新以简化这一过程。
- **社区对 Stable Diffusion 的反应**：用户辩论了社区对 **Stable Diffusion** 的看法，认为最近的批评通常源于对模型许可（licensing）的误解。
   - 用户对营销策略以及针对 **Stability AI** 的负面情绪表示担忧。
- **AI 采样方法的创新**：引入了一个新的采样器节点，实现了 **Strong Stability Preserving Runge-Kutta** 和隐式变步长求解器，引起了用户对提升 AI 性能的兴趣。
   - 用户热烈讨论了这些更新为 AI 模型效能带来的潜在改进。
- **关于 AI 体验的闲聊**：随着用户分享 AI 的个人体验（包括学习编程语言和评估与健康相关的专注力挑战），一般性讨论也蓬勃发展。
   - 这些闲聊增加了对 AI 日常应用的深入理解。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Llama 3 405B 发布，价格极具竞争力**：**Llama 3 405B** 以 **$3/M tokens** 的价格发布，与 **GPT-4o** 和 **Claude 3.5 Sonnet** 展开竞争，同时展示了用于生成合成数据的惊人 **128K token 上下文**。
   - 用户反应热烈，评论道“这是目前最好的开源 LLM”，并对该模型的能力表示兴奋。
- **对模型性能的担忧日益增加**：关于 **Llama 405B** 的反馈显示性能结果褒贬不一，特别是在翻译任务中，其表现不如 **Claude** 和 **GPT-4**。
   - 一些用户报告 **70B 版本**在生成几个 token 后会出现“乱码”，引发了对其在特定任务使用中可靠性的担忧。
- **令人兴奋的 OpenRouter 功能更新**：**OpenRouter** 的新功能包括**追溯发票（Retroactive Invoices）**、**自定义密钥**以及 **Playground** 的改进，全面增强了用户功能。
   - 鼓励社区成员在[这里](http://openrouter.ai/chat)分享反馈，以进一步优化用户体验。
- **发起多 LLM 提示词竞赛**：宣布了一项针对 **Llama 405B**、**GPT-4o** 和 **Claude 3.5 Sonnet** 的**提示词竞赛（prompting competition）**，参与者有机会赢取 **15 个免费额度**。
   - 参与者渴望了解评审标准，特别是关于什么才算作“高难度提示词”。
- **DeepSeek Coder V2 推理提供商公布**：引入了 **DeepSeek Coder V2** 新的私有推理提供商，在不进行输入训练的情况下运行，这显著拓宽了 OpenRouter 的服务范围。
   - 用户可以开始通过 [DeepSeek Coder](https://openrouter.ai/models/deepseek/deepseek-coder) 探索该服务。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 中 Flash Attention 的困惑**：一名成员对 **Flash Attention** 中寄存器的高效管理提出了疑问，并对其在 CUDA 编程中与共享内存（shared memory）结合使用的方式表示关注。
   - 这引发了在高性能计算（high-performance computing）背景下对寄存器分配策略进行更清晰说明的广泛需求。
- **Torch Compile 的内存挑战**：在一个小型 **Bert** 模型中使用 `torch.compile` 导致了显著的 RAM 占用，迫使 batch size 从 **512** 削减至 **160**，且性能落后于 eager mode。
   - 测试表明，尽管存在这些担忧，模型仍成功编译，这突显了 PyTorch 中的内存管理问题。
- **Meta Llama 3.1 专注于文本**：Meta 发布的 **Llama 3.1 405B** 将上下文长度扩展至 **128K** 并支持**八种语言**，目前暂不支持多模态（multi-modal）功能，这引发了战略层面的讨论。
   - 这一功能的缺失符合对其潜在财务收益以及在财报发布前竞争定位的预期。
- **优化 CUDA Kernel 性能**：用户经验表明，转向分块矩阵乘法（tiled matrix multiplication）带来的性能提升有限，这与一篇关于 CUDA 矩阵乘法基准测试的相关文章结论相似。
   - 讨论强调了计算强度（compute intensity）对于优化 Kernel 性能的重要性，尤其是在早期阶段。
- **AMD 上的 Stable Diffusion 加速**：一篇文章详细介绍了如何使用适用于 AMD RDNA3 GPU 的 [Composable Kernel 库](https://github.com/ROCm/composable_kernel/discussions/1032) 来优化 **Stable Diffusion** 在 **RX7900XTX** 上的推理（inferencing）。
   - 此外，最近的一个 [GitHub pull request](https://github.com/Dao-AILab/flash-attention/pull/1010) 强调了对 AMD ROCm 上 Flash Attention 的支持，该支持对 **mi200 & mi300** 有效。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GEMINI 竞赛引发关注**：一名成员对 Google 的 **GEMINI 竞赛** 表现出极大热情，正在为该黑客松（hackathon）寻找潜在的合作伙伴。
   - *如果你有兴趣合作，请联系！*
- **Llama 3.1 模型引发褒贬不一的反应**：成员们对 **Llama-3.1** 模型做出了反应，一些人认为与 **Claude** 和 **Gemini** 等早期迭代版本相比，它显得“**缺乏灵魂**”，后者被认为保留了更多的创作深度。
   - 这一讨论指出了用户对近期模型体验和预期的分歧。
- **微调 Llama 3.1 以获得无审查输出**：一位用户正致力于微调 **Llama-3.1 405B** 的无审查版本，目标是在经过几周的训练后，在 Hugging Face 上发布 **Llama3.1-406B-uncensored**。
   - 这一努力凸显了开发受限模型替代方案的持续兴趣。
- **Discord 中的语音 AI 面临挑战**：围绕创建能够参与 **Discord 语音频道** 的 AI 机器人展开了讨论，强调了由于目前的限制，该任务具有复杂性。
   - 成员们指出了有效实施所需解决的技术挑战。
- **热切期待 Alpha 版本发布**：成员们正焦急地等待 Alpha 版本的发布，有些人每 20 分钟检查一次应用，对它是会在 7 月底还是更早发布表示不确定。
   - 呼吁开发者就时间表进行更清晰的沟通。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 社区会议演讲征集**：8 月 12 日的 [Mojo 社区会议](https://modul.ar/community-meeting-doc) 正在公开征集演讲，旨在展示开发者在 Mojo 中构建的内容。
   - 成员可以报名分享经验和项目，增强社区参与度。
- **字符串和缓冲区优化成为焦点**：标准库中关于 **short string optimization**（短字符串优化）和 **small buffer optimization**（小缓冲区优化）的工作正被提议进行演示，突显了其在未来会议中的相关性。
   - 这项工作与过去以性能增强为中心的讨论主题相一致。
- **在 Ubuntu 虚拟机上安装 Mojo 变得简单**：讨论了在 Windows 的 Ubuntu 虚拟机中安装 Mojo 的方法，建议将 **WSL** 和 **Docker** 作为可行方案。
   - 虽然有人担心可能出现的安装问题，但普遍共识是使用虚拟机是合适的。
- **Mojo：游戏引擎开发的未来**：讨论了 Mojo 在创建下一代游戏引擎方面的潜力，强调了其对通过 GPU 进行异构计算（heterogeneous compute）的强大支持。
   - 提到了分配器（allocator）处理方面的挑战，表明在游戏开发模式中存在一些障碍。
- **将 Mojo 与 C 库链接**：关于改进 Mojo 与 C 库（特别是利用 **libpcap**）链接能力的对话正在进行中。
   - 成员们主张在 Linux 上的 Mojo 默认使用 **ktls**，以增强网络功能。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **FSDP 与 nn.Parameters 的性能困扰**：一位用户在使用 **FSDP** 添加 `nn.Parameters` 时遇到了 **20 倍的减速**，但 **16** 的参数大小显著提升了性能。
   - 他们讨论了 **buffer alignment**（缓冲区对齐）如何影响 **CPU** 性能，尽管 GPU kernel 运行很快。
- **在高端硬件上访问 Llama 3.1 Instruct**：一位成员成功在 **8xH100 80GB** 上托管了 **Llama 3.1 405B instruct**，可通过 [聊天界面](https://chat.tune.app/) 和 [API](https://studio.tune.app/) 访问。
   - 然而，访问需要登录，引发了关于成本和硬件限制的讨论。
- **引入 Switch SAE 以实现高效训练**：**Switch SAE** 架构改进了稀疏自编码器（SAEs）的扩展性，解决了跨层的训练挑战。
   - 相关论文表明，这可能有助于从超智能语言模型中恢复特征。
- **对 Llama 3 图像编码的担忧**：讨论中提到 **Llama 3** 的图像编码器分辨率限制为 **224x224**，建议使用 **vqvae-gan** 风格的 tokenizer 进行增强。
   - 建议关注 Armen 的小组，强调了潜在的改进空间。
- **评估任务分组策略**：成员们建议对嵌套任务使用 **groups**（组），对简单安排使用 **tags**（标签），这一建议得到了 **Hailey Schoelkopf** 的认可。
   - 该方法旨在有效地简化任务组织。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Meta 推出 Premium Llama 405B**：推测 Meta 可能会在 **7 月 23 日** 宣布 **Premium** 版本的 **Llama 405B**，此前 Meta 最近取消了对 Llama 模型的限制，为更多样化的应用铺平了道路。
   - 这一变化引发了关于更广泛用例的讨论，不再仅仅局限于增强其他模型。
- **NVIDIA 的市场策略**：有人担心 **NVIDIA** 可能会垄断 AI 领域，旨在将 **硬件**、**CUDA** 和模型产品结合起来。
   - 一位用户指出，这种主导地位可能会带来巨额利润，尽管监管挑战可能会阻碍这一愿景。
- **OpenAI 的定价动态**：OpenAI 推出 **gpt-4o-mini** 每天高达 **2M tokens** 的免费微调，引发了关于 AI 竞争性定价环境的讨论。
   - 成员们将定价格局描述为混乱，这是为了应对日益激烈的竞争而出现的。
- **Llama 3.1 超出预期**：[Llama 3.1](https://llama.meta.com/) 的发布引入了拥有 **405B 参数** 的模型并增强了多语言能力，在评估中表现出与 **GPT-4** 相似的性能。
   - 随后展开了关于潜在 **模型水印** 和用户下载跟踪的讨论，重点关注合规性和隐私问题。
- **Magpie 的合成数据创新**：[Magpie 论文](https://arxiv.org/abs/2406.08464) 强调了一种为 LLM 生成 **高质量指令数据** 的方法，该方法在 **词汇多样性** 方面超过了现有的数据源。
   - 值得注意的是，在 **Magpie IFT 数据集** 上微调的 **LLaMA 3 Base** 在 **AlpacaEval** 上的表现比原始 **LLaMA 3 Instruct** 模型高出 9.5%。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Llama 3.1 发布引发褒贬不一的反应**：**Llama 3.1** 的发布引起了复杂的情绪，特别是针对其在 **Mistral** 等模型背景下的**实用性**。一些成员表达了不满，正如一位成员所说：*“该死，他们不喜欢 Llama 的发布”*。
   - 尽管有热度，但反馈表明需要更好的性能指标以及相对于前代产品更明显的优势。
- **用户面临 Llama 3.1 的训练挑战**：在训练 **Llama 3.1** 时，与 `rope_scaling` 配置相关的错误导致了社区的挫败感。通过更新 Transformers 找到了解决方法，展示了用户的韧性，一位用户评论道：*“似乎起作用了，谢谢！”*。
   - 这突显了新模型发布时持续存在的故障排除这一更广泛的主题。
- **对 Llama 3.1 语言包含性的担忧**：**Llama 3.1** 排除对**中文**的支持引发了关于其全球影响的讨论。虽然 Tokenizer 包含中文，但缺乏优先级被批评为战略失误。
   - 这次对话指出 AI 模型中**语言包容性**的持续必要性。
- **评估分数对比：Llama 3.1 vs Qwen**：社区讨论集中在比较 **Llama 3.1** 的 **cmmlu** 和 **ceval** 分数，结果显示仅有边际提升。成员指出，虽然 **Qwen** 自报的分数更高，但评估指标的差异使得直接比较变得复杂。
   - 这反映了社区对不断发展的模型性能基准的持续关注。
- **探索 LLM 蒸馏流水线**：一位成员分享了 [LLM Distillery GitHub 仓库](https://github.com/golololologol/LLM-Distillery)，重点介绍了一个专注于预计算 Logits 和 KL divergence（KL 散度）进行 LLM 蒸馏的流水线。这表明了改进蒸馏过程的积极态度。
   - 社区对优化此类流水线的兴趣反映了对提高模型训练效率的持续承诺。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Code Confluence 工具生成 GitHub 摘要**：受 **DSPy** 启发，一位成员介绍了 **Code Confluence**，这是一个使用 **Antlr**、**Chapi** 和 DSPy 流水线构建的 OSS 工具，旨在创建 GitHub 仓库的详细摘要。该工具的性能很有前景，正如在其 [DSPy 仓库](https://github.com/unoplat/unoplat-code-confluence/blob/main/unoplat-code-confluence/examples/python/dspy/dspy_v1.md)中演示的那样。
   - 他们还分享了资源，包括 [Unoplat Code Confluence GitHub](https://github.com/unoplat/unoplat-code-confluence/) 和名为 [OSS Atlas](https://github.com/unoplat/unoplat-oss-atlas/tree/main) 的摘要汇编。
- **新 AI 研究论文预警**：一位成员分享了一篇题为 [2407.12865](https://arxiv.org/pdf/2407.12865) 的 AI 研究论文链接，引发了对其研究结果的兴趣。鼓励社区成员分析并讨论其影响。
   - 有人请求任何在代码中复现该发现或找到现有实现的人进行分享。
- **JSON 生成库的比较**：成员们讨论了 **Jsonformer** 和 **Outlines** 等库在结构化 JSON 生成方面的优势，指出 **Outlines** 对 Pydantic 格式提供了更好的支持。虽然 *Jsonformer* 在严格合规性方面表现出色，但 *Guidance* 和 *Outlines* 提供了灵活性，但也增加了复杂性。
   - 考虑到社区的反馈，他们正在探索每个库在工作流中的实际应用。
- **Llama3 结构化输出的挑战**：用户表示在使用 DSPy 从 **Llama3** 获取正确结构化输出时遇到困难。他们建议将 `dspy.configure(experimental=True)` 与 *TypedChainOfThought* 结合使用，以提高成功率。
   - 有人对即使在类型检查失败时也能查看模型输出表示担忧，发现 `inspect_history` 在调试方面存在局限性。
- **探索 ColPali 用于医疗文档**：由于之前使用 **ColBert** 和标准 Embedding 模型失败，一位成员分享了使用 **ColPali** 对**带有图像的医疗文档进行 RAG** 的经验。目前正在计划研究其他的 Vision-Language Models。
   - 这一探索旨在增强从复杂文档类型中检索信息的有效性。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **关于高效文档检索的 LlamaIndex 网络研讨会**：参加本周五**上午 9 点（太平洋时间）**举行的**使用视觉语言模型进行高效文档检索**的网络研讨会。[在此注册](https://lu.ma/9q4ldrwc)以探索前沿技术。
   - ColPali 引入了一种创新技术，直接使用 Vision Language Models 嵌入**页面截图**，增强了对传统解析难以处理的复杂文档的检索。
- **TiDB Future App 黑客松提供 30,000 美元奖金**：参加 [TiDB Future App Hackathon 2024](https://t.co/vTV3t8daqT)，有机会赢取总计 **30,000 美元**的奖金池，其中冠军可获得 **12,000 美元**。该竞赛敦促使用最新的 **TiDB Serverless with Vector Search** 构建创新的 AI 解决方案。
   - 鼓励开发者与 @pingcap 合作，展示他们在构建高级应用方面的最佳成果。
- **探索 LlamaIndex 的 Mixture-of-Agents**：一段新视频展示了使用多个本地语言模型的 "mixture of agents" 方法，其表现有可能超越 **GPT-4** 等独立模型。查看[分步教程](https://t.co/EqF2RM3jeB)以深入了解这一增强技术。
   - 支持者认为这种方法可以提供竞争优势，特别是在需要多样化模型能力的项目中。
- **Llama 3.1 模型现已发布**：**Llama 3.1** 系列现在包括 **8B**、**70B** 和 **405B** 模型，可通过带有 Ollama 的 LlamaIndex 访问，尽管最大的模型需要大量的计算资源。探索 [Fireworks AI](https://t.co/NMckK14nZf) 的托管解决方案以获取支持。
   - 用户在选择较大的模型时应评估其计算能力，以确保最佳性能。
- **澄清 context_window 参数以改进模型使用**：`context_window` 参数定义了影响模型输入和输出能力的总 Token 限制。计算错误可能会因超出限制而导致 ValueError 等错误。
   - 建议用户调整输入大小或选择具有更大 Context 能力的模型，以优化输出效率。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Llama 3.1 发布引发关注**：[Llama 3.1](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) 的发布包括 **405B 模型**，标志着开源 LLM 的一个重要里程碑，其**卓越的能力**足以与封闭模型媲美。
   - 初步评估显示它是第一个具有前沿能力的开源模型，因其对迭代研发的可访问性而受到称赞。
- **国际语言学奥林匹克竞赛 (IOL)**：**国际语言学奥林匹克竞赛 (IOL)** 开幕，挑战学生利用逻辑翻译冷门语言，类似于高风险的数学竞赛。
   - 参赛者在长达六小时的时间内解决看似不可能的问题，突显了逻辑推理与语言的交集。
- **Llama 定价见解**：Llama 3.1 的 **405B 模型**在 Fireworks 和 Together 等平台上的定价约为**每百万 Token 4-5 美元**。
   - 这种具有竞争力的定价策略旨在随着采用率的增长，在潜在提价之前占领市场份额。
- **Llama 性能评估**：早期评估表明 Llama 3.1 在 **GSM8K** 和 **ZebraLogic** 逻辑推理等基准测试中排名很高，介于 **Sonnet 3.5** 和 **GPT-4o** 之间。
   - 在对比测试中注意到，在超长 Token 长度后保持 Schema 一致性等挑战。
- **GPT-4o mini 微调发布**：OpenAI 宣布了 **GPT-4o mini** 的微调功能，面向 Tier 4 和 5 用户开放，在 9 月 23 日之前，每天前 **200 万个训练 Token 免费**。
   - 这一举措旨在扩大访问和定制化，用户正在评估其相对于新发布的 Llama 3.1 的表现。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **AgentState vs InnerAgentState 详解**：一场讨论澄清了 `AgentState` 与 `InnerAgentState` 之间的区别，提供了 `AgentState` 的定义，并建议查看 [LangChain 文档](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/#basic-example-using-the-docker-container)以获取更多细节。
   - `AgentState` 的关键字段包括 `messages` 和 `next`，这对于 LangChain 中依赖上下文的操作至关重要。
- **设置 Chroma 向量数据库**：分享了如何在 Python 中使用开源解决方案将 **Chroma** 设置为向量数据库的说明，需要安装 `langchain-chroma` 并通过 Docker 运行服务器。
   - 示例展示了 `.add`、`.get` 和 `.similarity_search` 等方法，并强调了使用 `OpenAIEmbeddings` 时必须持有 OpenAI API Key。
- **使用 Composio 创建调度 Agent**：一份关于使用 **Composio**、**LangChain** 和 **ChatGPT** 创建**调度 Agent**（Scheduler Agent）的指南，可以实现通过电子邮件进行流线化的活动调度。指南可在[此处](https://git.new/scheduler)获取。
   - Composio 为 Agent 增强了有效的工具，如[调度示例](https://git.new/scheduler)中所示，强调了任务处理的效率。
- **YouTube 笔记生成器上线了！**：宣布推出 **YouTube Notes Generator**，这是一个用于从 YouTube 视频生成笔记的开源项目，旨在方便用户直接从视频内容中更轻松地记录笔记。
   - 在 [LinkedIn](https://www.linkedin.com/posts/isham-rashik-5a547711b_machinelearning-artificialintelligence-deeplearning-activity-7221165319464095747-DMDS?utm_source=share&utm_medium=member_desktop) 上了解更多关于此工具及其功能的信息。
- **使用 AI 进行高效代码审查**：一段名为 **'AI Code Reviewer Ft. Ollama & Langchain'** 的新视频介绍了一个 CLI 工具，旨在增强开发者的代码审查流程；点击[此处](https://youtu.be/g_VRsjpC4e8)观看。
   - 该工具旨在通过促进开发团队中高效的代码评估来简化工作流。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **新成员加入 Cohere 社区**：新成员展示了加入 Cohere 的热情，引发了社区的积极欢迎。
   - *社区张开双臂欢迎新手*，为讨论营造了亲切的氛围。
- **使用 Midicaps 数据集进行创新微调**：基于以往成功项目的经验，使用 **midicaps** 进行微调的工作取得了进展，展现出良好的前景。
   - 成员们强调了过去努力取得的*良好结果*，预示着未来可能的突破。
- **澄清 Cohere 的 OCR 解决方案**：**Cohere** 利用 [unstructured.io](https://unstructured.io) 来实现其 OCR 能力，并对外部集成保持开放选择。
   - 社区就 OCR 功能的定制和增强进行了富有成效的讨论。
- **探讨 RAG 聊天机器人系统**：**RAG** 架构聊天机器人系统中的对话历史管理成为热门话题，重点讨论了向量数据库的使用。
   - 提出了点赞/点踩等反馈机制，以优化交互体验。
- **发布具有重大改进的 Rerank 3 Nimble**：**Rerank 3 Nimble** 亮相，在保持准确性的同时提供 **3 倍的吞吐量**，现已在 [AWS SageMaker](https://cohere.com/blog/rerank-3-nimble) 上可用。
   - *迎接企业搜索速度的提升！* 这一基础模型提升了检索增强生成（RAG）的性能。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Llama 3.1 正式发布！**：Meta 今天早上发布了最新的模型 **Llama 3.1**，支持 **8B** 和 **70B** instruct 模型。查看 [Llama 3.1 Model Cards and Prompt formats](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1) 了解详情。
   - 现场气氛非常热烈，甚至出现了一些因兴奋而导致的拼写错误和低级错误。
- **MPS 支持 Pull Request 讨论**：标题为 [MPS support by maximegmd](https://github.com/pytorch/torchtune/pull/790) 的 Pull Request 引入了对 **MPS 设备上 BF16** 的检查，旨在改进本地 Mac 电脑上的测试。讨论指出由于共同祖先差异 (common ancestor diff) 可能存在问题，建议采用 rebase 可能是更好的方法。
   - 该 PR 被强调为对于使用 MPS 的开发者的关键更新。
- **LoRA 问题依然存在**：有人提出了关于 **LoRA** 实现未按预期运行的持续性问题，并给出了调试建议。一位贡献者指出在最近的尝试中遇到了 **CUDA 硬编码 (hardcoding)** 的挑战。
   - 这一问题凸显了对模型性能进行更深层次排查的必要性。
- **Git 工作流挑战重重**：**Git 工作流** 挑战一直是热门话题，许多人在解决之前的冲突后又陷入了新的冲突循环。有人建议调整工作流以尽量减少这些冲突。
   - 对于贡献者来说，有效的冲突解决策略似乎是一个始终紧迫的需求。
- **引入 Pad ID 错误修复 PR**：[Pull Request #1211](https://github.com/pytorch/torchtune/pull/1211) 解决了 generate 过程中显示 **pad ID** 的关键 bug，旨在防止此问题再次发生。它明确了 **utils.generate** 中 Pad ID 默认为 **0** 的隐含假设。
   - 这一修复对于确保在未来的生成任务中正确处理特殊 token 至关重要。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **需要帮助重现 matmul-free-llm**：有人请求协助在 tinygrad 中重现 **matmul-free-llm**，旨在利用 [高效算子 (kernels)](https://github.com/ridgerchu/matmulfreellm/blob/master/mmfreelm/ops/fusedbitnet.py) 并整合 **fp8**。
   - *希望能尽快无缝适配 Blackwell fp4*。
- **M1 结果与 CI 不同**：一位 M1 用户遇到了与 CI 不同的结果，正在寻求关于如何使用 **conda** 和环境变量正确设置测试的说明。
   - *由于启用 `PYTHON=1` 时会导致测试中出现 IndexError，目前存在困惑。*
- **cumsum 性能问题**：一位新成员正在探索 tinygrad 中 **nn.Embedding** 的 **O(n)** 实现，以及如何借鉴 PyTorch 的技术将 **cumsum** 从 O(n^2) 优化到 O(n)。
   - *有人推测某些约束使得这项任务具有挑战性，尤其是这还是一个 **$1000 的悬赏任务**。*
- **寻求使用 PyTorch 进行增量测试的模式**：一位成员询问了在 PyTorch 中按 **Linear, MLP, MoE** 和 **LinearAttentionMoE** 顺序增量测试模型性能的有效模式。
   - *他们质疑从头开始测试是否比增量测试更有效率。*
- **在 tinygrad 中开发分子动力学引擎**：一个团队正尝试在 tinygrad 中实现 **分子动力学 (Molecular Dynamics) 引擎**，以训练预测分子构型能量的模型，但在梯度计算方面面临挑战。
   - *他们需要预测能量相对于输入位置的梯度来计算力，但由于对模型权重进行了两次反向传播 (backpropagate)，导致出现了问题。*

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Int8 实现已确认**：成员们讨论了使用 **Int8** 的情况，其中一人确认其有效，显示了开发者对优化技术的兴趣。
   - 有人请求“等一下（Hold a sec）”，表明在实现过程中可能需要额外的指导和社区支持。
- **ComfyUI Flow 脚本指导**：一位用户请求一个 **ComfyUI flow** 脚本，随后得到了利用该框架实现更平滑设置流程的建议。
   - 这反映了社区在处理复杂系统集成时，倾向于追求效率和首选工作流的趋势。
- **Llama 3.1 设定新标准**：[**Llama 3.1 405B**](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) 的发布引入了 **128K** 的上下文长度，在八种语言中提供了显著的能力。
   - 这一飞跃使 Llama 3.1 成为领先模型的强力竞争者，讨论集中在其多样化的功能上。
- **Meta 的开源承诺**：正如 Mark Zuckerberg 的信中所述，Meta 强调了其对 [**开源 AI**](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/) 的奉献，突出了开发者和社区的利益。
   - 这与其在 AI 生态系统中促进协作的愿景一致，旨在实现工具和资源的更广泛可及性。
- **Llama 3.1 中的上下文大小增强**：讨论批评了之前的 **8K** 上下文大小不足以处理大型文档，现在 Llama 3.1 中新的 **128K** 大小解决了这一问题。
   - 这一改进被认为对于需要广泛文档处理的任务至关重要，显著提升了模型性能。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Llama 3.1 405B 令用户惊叹**：据报道，**Llama 3.1 405B** 与 **OpenInterpreter** 配合得非常出色。与 **GPT-4o** 不同，它不需要不断的提醒或重启来完成多项任务。
   - 用户强调，与 **GPT-4o** 相比，**Llama 3.1 405B** 提供的体验显著提高了生产力。
- **对 GPT-4o 的挫败感**：一位用户表达了对 **GPT-4o** 的挑战，需要频繁的提示才能在电脑上执行任务。这种挫败感凸显了用户在使用 **Llama 3.1 405B** 时获得的无缝体验。
   - 这种对比表明用户更倾向于选择 **Llama 3.1 405B** 来进行高效的任务管理。
- **在 MacOS 上使用 Coqui 模型进行语音输入？**：有人询问在 **MacOS** 上使用本地 **Coqui model** 进行语音输入的问题。目前尚未有成功的实现报告。
   - 社区参与保持开放，但尚未出现进一步的回复来澄清该应用的可行性。
- **Expo App 对 Apple Watch 的支持能力**：讨论确认 **Expo app** 理论上应该能够为 **Apple Watch** 构建应用程序。然而，没有提供进一步的细节或确认。
   - 虽然持乐观态度，但社区仍在等待在 *Apple Watch* 环境下对该能力的实际验证。
- **设备的发货时间线**：一位成员询问了特定设备的 **发货时间线**，表达了对其状态的好奇。对话中没有分享更新或时间表。
   - 信息的缺乏表明在发货状态方面需要更清晰的沟通。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **关于 OpenOrca 数据集许可的澄清**：一位成员询问适用于 **OpenOrca** 数据集的 **MIT License** 是否允许商业化使用源自 **GPT-4 Model** 的输出。
   - *其输出可以用于商业目的吗？* 这一问题突出了围绕 AI 数据集许可的持续讨论。
- **开源合成数据集的计划**：另一位成员透露了开源 **合成数据集（synthetic dataset）** 的意图，旨在支持商业和非商业项目，强调了其在 AI 生态系统中的相关性。
   - 他们提到正在评估对 **OpenOrca** 可能存在的依赖，这引发了关于其在更广泛数据集领域中许可影响的问题。

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **迈阿密聚会意向引发讨论**：一名成员询问了在 **迈阿密 (Miami)** 举办潜在 **meetups** 的可能性，寻求与该地区其他人的联系以进行聚会。
   - 到目前为止，关于这次聚会查询，还没有进一步的回复或安排。
- **8月纽约聚会获得关注**：另一名成员表示有兴趣参加 8 月下旬在 **纽约 (NYC)** 举行的 meetup，表达了社区互动的愿望。
   - 这一讨论暗示了纽约地区本地 AI 爱好者的活动协调可能性。



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **艺术家寻求合作**：Aria，一位 **2D/3D 艺术家**，表达了与社区内其他人合作的兴趣。他们邀请感兴趣的成员通过 DM 联系以开展潜在项目。
   - 这为 Guild 中任何希望将 **艺术** 技能融入其 AI 项目（特别是在可视化或游戏领域）的人提供了机会。
- **AI 工程师的参与机会**：合作呼吁强调了将 **AI engineering** 与艺术和设计等创意领域融合的日益增长的兴趣。
   - 此类合作可以增强 **AI 项目** 的视觉效果，从而可能带来更具吸引力的用户体验。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla Accelerator 申请截止日期临近**：**Mozilla Accelerator** 的申请截止日期即将到来，该项目提供为期 **12 周** 的计划，以及高达 **$100k** 的非稀释性资金。
   - 参与者还将在与 Mozilla 共同举办的 **demo day** 上展示他们的项目，为获取反馈和曝光提供关键时刻。[有问题吗？](https://discord.com/channels/1089876418936180786/1245083732319408195)
- **准备参加 Zero Shot Tokenizer Transfer 活动**：提醒即将举行的与 Benjamin Minixhofer 合作的 **Zero Shot Tokenizer Transfer** 活动，计划于本月举行。
   - 详情可以在 [活动链接](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732) 中找到，鼓励感兴趣的工程师参与。
- **介绍 AutoFix：开源 Issue 修复工具**：**AutoFix** 是一个开源工具，可以直接从 **Sentry.io** 提交 PR，从而简化 Issue 管理。
   - 在此处链接的详细帖子中了解有关此工具功能的更多信息：[AutoFix 信息](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732)。



---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道划分的详细摘要与链接


{% if medium == 'web' %}




### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1265310564222111884)** (1 条消息): 

> - `NuminaMath datasets`
> - `Docmatix dataset`
> - `SmolLM models`
> - `Chameleon model`
> - `Followgraph tool`

- **NuminaMath 数据集发布**：**NuminaMath** 数据集已发布，包含约 **1M** 个数学竞赛题目-解答对，曾助力获得 AI Math Olympiad 的 **Progress Prize**。这包括旨在增强数学推理能力的 **Chain of Thought** 和 **Tool-integrated reasoning** 子集。
   - 在 NuminaMath 上训练的这些模型实现了**同类最佳性能**，在数学竞赛基准测试中超越了私有模型，并可在 [🤗 Hub](https://huggingface.co/collections/AI-MO/numinamath-6697df380293bcfdbc1d978c) 上获取。
- **Docmatix 数据集介绍**：**Docmatix** 数据集作为文档理解的巨量资源推出。它旨在解决此前阻碍开源模型在文档任务中表现的数据覆盖不足问题。
   - 该数据集旨在提升各种文档任务的性能，此前由于缺乏充足的开源数据，这些任务通常更倾向于闭源模型。
- **SmolLM 模型发布**：发布了一系列名为 **SmolLM** 的新模型，包含 **135M**、**360M** 和 **1.7B** 参数规模。它们的表现优于 **MobileLLM**、**Phi1.5** 和 **Qwen2**，且是在高质量语料库上训练而成的。
   - 该系列模型应对了 **large language models** (LLMs) 端侧部署日益增长的重要性，满足了多样化的应用需求。
- **Chameleon 模型现已可用**：Meta 的多模态模型 **Chameleon** 现已集成到 **transformers** 中，提供 **7B** 和 **34B** 参数版本。该模型旨在增强各种多模态任务。
   - Chameleon 的集成代表了 **transformers** 在处理多样化输入和输出能力方面的重大进步。
- **通过 Followgraph 探索 ML 社交**：推出了一款名为 **Followgraph** 的新工具，旨在方便用户关注有趣的 ML 领域人士。其目标是增强 ML 社区内的协作和交流机会。
   - 该工具允许用户发现并联系机器学习领域的有影响力人物，为专业互动增添了社交维度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/_lewtun/status/1814958635732140336">来自 Lewis Tunstall (@_lewtun) 的推文</a>：我们刚刚发布了 ✨NuminaMath 数据集：这是最大的数学竞赛问题-解答对集合，约有 100 万条，难度从初级挑战到数学奥林匹克预选赛不等。这些...</li><li><a href="https://x.com/mervenoyann/status/1813963500513058849">来自 merve (@mervenoyann) 的推文</a>：介绍 Docmatix：一个巨大的文档理解数据集 📑 到目前为止，由于缺乏数据覆盖，闭源模型在文档任务中的表现优于开源模型 💔 但 @huggingface M4 来了...</li><li><a href="https://x.com/LoubnaBenAllal1/status/1813252390692303069">来自 Loubna Ben Allal (@LoubnaBenAllal1) 的推文</a>：LLM 的端侧部署比以往任何时候都更加重要。今天我们发布了 SmolLM，一个新的 SOTA 系列，包含 135M、360M 和 1.7B 模型：- 表现优于 MobileLLM、Phi1.5 和 Qwen2 小型模型 - 训练...</li><li><a href="https://x.com/NielsRogge/status/1814310702162551247">来自 Niels Rogge (@NielsRogge) 的推文</a>：我们刚刚发布了视觉语言模型 (VLMs) 的聊天模板！🔥 像 LLaVa、LLaVa-NeXT 和 LLaVa-Interleave 这样的模型现在都可以使用 messages API 进行调用。文档：https://huggingface.co/do...</li><li><a href="https://x.com/TheZachMueller/status/1813218332050358522">来自 Zach Mueller (@TheZachMueller) 的推文</a>：延迟加载模型权重已合并到 @huggingface transformers 主分支！一条推文解释这到底意味着什么... 通常当你加载 PyTorch 权重时，它是瞬时的（也就是说当...</li><li><a href="https://x.com/KonradSzafer/status/1815726520939212985">来自 Konrad Szafer (@KonradSzafer) 的推文</a>：我们刚刚在 Transformers Tokenizer 类中添加了一个新方法，以改进跟踪和可复现性。你现在可以检索 Tokenizer 使用的确切聊天模板了！🚀</li><li><a href="https://x.com/mervenoyann/status/1814278511785312320">来自 merve (@mervenoyann) 的推文</a>：@Meta 的 Chameleon 🦎 现在已在 @huggingface transformers 中可用 😍 这是一个多模态模型，有 7B 和 34B 两种尺寸 🤩 但这个模型有什么特别之处呢？继续阅读 ⇣</li><li><a href="https://x.com/NielsRogge/status/1810284458412573052">来自 Niels Rogge (@NielsRogge) 的推文</a>：@huggingface Transformers 现在新增了 2 个深度估计模型！Depth Anything v2 和 ZoeDepth - Depth Anything v2 是相对的，告诉你像素之间的相对距离 - ZoeDepth 是绝对的...</li><li><a href="https://x.com/julien_c/status/1814310812393120077">来自 Julien Chaumond (@julien_c) 的推文</a>：周五 @huggingface 更新。对于图像生成模型和 LoRAs，我们现在直接在用户个人资料上显示模型的微型预览。祝周末愉快！🔥</li><li><a href="https://x.com/severo_dev/status/1815684824893436246">来自 Sylvain Lesage (@severo_dev) 的推文</a>：[新工具] 使用 Followgraph 关注有趣的 ML 人士 👩‍🎨 👨‍🎤 👩‍🏫 https://huggingface.co/spaces/severo/followgraph</li><li><a href="https://x.com/vanstriendaniel/status/1814298698383585692">来自 Daniel van Strien (@vanstriendaniel) 的推文</a>：在分享模型微调 Notebook 时，显示输入数据集会很有帮助。你现在可以直接在 @GoogleColab Notebook 中嵌入数据集查看器。这是一个编辑过的 @UnslothAI Notebook ...</li><li><a href="https://x.com/RemiCadene/status/1813675172492411170">来自 Remi Cadene (@RemiCadene) 的推文</a>：🚨 我们现在可以直接在 Hugging Face Hub 上可视化 LeRobot 数据集。在我刚刚记录的数据集上试试吧 😇 https://huggingface.co/spaces/lerobot/visualize_dataset Hugging Face 具有潜力...</li><li><a href="https://x.com/abhi1thakur/status/1813892464144798171">来自 abhishek (@abhi1thakur) 的推文</a>：我们刚刚在 AutoTrain 中集成了数据集查看器 💥 所以，现在你可以在训练模型之前查看数据集，识别正确的拆分和列，而无需离开页面 🚀</li><li><a href="https://x.com/reach_vb/status/1815434084572688581">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：我们整理了一篇详细的博客文章，介绍了在 Mac 上运行 Mistral 的步骤，以及 Apple 在 WWDC 期间宣布的所有更新：https://huggingface.co/blog/mistral-coreml</li><li><a href="https://x.com/evijitghosh/status/1814003112128172255">来自 Avijit Ghosh (@evijitghosh) 的推文</a>：http://x.com/i/article/1814002459108691968</li><li><a href="https://x.com/calebfahlgren/status/1814116515328807226">来自 Caleb (@calebfahlgren) 的推文</a>：写了一篇博客文章，介绍如何使用 Datasets Explorer 在 @huggingface 数据集上发现非常有趣的见解 🔥 甚至还有几个使用 @duckdb 空间扩展的例子...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1265020925708079125)** (1104 条消息🔥🔥🔥):

> - `Llama 3.1 release`
> - `Kanye West controversy`
> - `Building PC setups`
> - `Model fine-tuning practices`
> - `Textbook recommendations for LLMs` 


- **Llama 3.1 发布热潮**：Llama 3.1 的发布引发了广泛关注，8B 和 405B 等模型现已可用于测试和部署。
   - 用户正在分享他们的经验，并排查在尝试本地运行模型时遇到的 ValueError 等问题。
- **Kanye West 在音乐界的影响力**：尽管围绕 Kanye West 存在争议，许多用户（如 kebab_addict）表达了对他音乐才华及其对行业影响的欣赏。
   - 讨论还强调了将艺术家的作品与其个人争议区分开来的复杂性。
- **组装 PC 配置与 GPU 讨论**：用户正在讨论用于组装高性价比 PC 的各种 GPU 选项，并推荐了 3060 和 4060ti 等型号。
   - 一些人对组件成本上升表示担忧，同时分享了获取硬件的个人经历。
- **模型 Fine-tuning 实践**：针对特定任务进行模型 Fine-tuning 的挑战正在被讨论，用户对高 Loss 值和性能问题表示沮丧。
   - 有人建议利用相关资源和实践来更好地处理模型训练和评估。
- **LLM 教科书推荐**：一位用户正在寻找涵盖 LLM 最新创新的综合性教科书，表示相比视频内容更倾向于文字材料。
   - 诸如《Transformers for Natural Language Processing》之类的书籍被提及为潜在资源，尽管它们主要侧重于应用学习。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://huggingface.co/starsnatched/MemeGPT">starsnatched/MemeGPT · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/enzostvs/zero-gpu-spaces">— Zero GPU Spaces — - enzostvs 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://x.com/osanseviero/status/1815769303188205678">来自 Omar Sanseviero (@osanseviero) 的推文</a>: Llama 3.1 发布了 🔥 尽情体验吧！ - 了解更多信息 https://hf.co/blog/llama31 - 模型 https://hf.co/meta-llama - 社区量化版本 https://hf.co/hugging-quants - 如何使用它 https://github.com/huggingf...</li><li><a href="https://huggingface.co/chat/settings/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8">HuggingChat</a>: 让每个人都能使用社区最优秀的 AI 聊天模型。</li><li><a href="https://huggingface.co/spaces/Xenova/whisper-speaker-diarization">Whisper Speaker Diarization - Xenova 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.nbcnews.com/news/world/snoop-dogg-will-carry-olympic-torch-final-leg-paris-rcna163234">Snoop Dogg 将在通往巴黎的最后一程传递奥运圣火</a>: 这位在文化领域无处不在的饶舌歌手将在周五的开幕式前见证圣火传统的延续。</li><li><a href="https://www.amazon.com/ASUS-ProArt-GeForce-Graphics-DisplayPort/dp/B0CCC7MP3H">未找到标题</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2407.14561">NNsight 和 NDIF：让基础模型内部访问大众化</a>: 尖端基础模型的巨大规模限制了科学家对其的访问，因为在大模型尺寸上进行定制化实验需要昂贵的硬件和复杂的工程...</li><li><a href="https://tenor.com/view/what-gif-21384529">What GIF - What - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://arxiv.org/abs/1810.04805">BERT: 用于语言理解的深度双向 Transformer 预训练</a>: 我们介绍了一种名为 BERT 的新语言表示模型，它代表来自 Transformer 的双向编码器表示。与最近的语言表示模型不同，BERT 旨在...</li><li><a href="https://tenor.com/view/patrick-stupid-drooling-patrick-star-spongebob-gif-12221001666588210206">派大星变笨 GIF - 派大星流口水 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/stfu-kanye-kanye-west-shut-up-dance-gif-23839788">闭嘴 Kanye GIF - 闭嘴 Kanye Kanye West - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/spongebob-squarepants-begging-pretty-please-beg-on-your-knees-pray-for-mercy-gif-26344462">海绵宝宝哀求 GIF - 海绵宝宝哀求拜托拜托 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/blog/nroggendorff/train-with-llama-architecture">从零开始训练 Llama 模型</a>: 未找到描述</li><li><a href="https://doc.rust-lang.org/book/#the-rust-programming-language">Rust 程序设计语言 - Rust 程序设计语言</a>: 未找到描述</li><li><a href="https://youtu.be/01g_EfO-Dms?si=tMF70x7MhxKw8S95">让你的 Agent 可靠性提升 10 倍？Flow Engineer 入门</a>: 深入探讨 Flow Engineer 和 Lang Graph，构建可靠的 SQL Agent。获取 Codeium（免费的 GitHub Copilot 替代方案）：https://codeium.com/?utm_source=youtube&amp;u...</li><li><a href="https://tenor.com/view/sad-upset-violin-sponge-bob-mr-crab-gif-3466351">悲伤小提琴 GIF - 悲伤沮丧的小提琴 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tinyurl.com/2nfwn2xy">Vision Card</a>: 未找到描述</li><li><a href="https://tenor.com/view/lindsey-stirling-lindsey-stirling-cute-adorable-gif-19359953">Lindsey Stirling 可爱 GIF - Lindsey Stirling Lindsey Stirling - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/openai/whisper-large-v3">openai/whisper-large-v3 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/waiting-waiting-patiently-waiting-for-you-waiting-on-you-gif-15489516379864441176">等待耐心等待 GIF - 等待耐心等待等你 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/wizard-dance-ena-gif-27696814">巫师跳舞 GIF - 巫师跳舞 Ena - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/good-morning-gif-11437316614611695342">早安 GIF - 早安 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/mark-zuckerberg-gif-14169217">Mark Zuckerberg GIF - Mark Zuckerberg - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/scared-dog-shivering-dog-dog-shaking-meme-gif-26566244">受惊的狗发抖的狗 GIF - 受惊的狗发抖的狗 狗狗颤抖 Meme - 发现并分享</a>: 点击查看 GIF</li>

<li><a href="https://tenor.com/view/cat-twitching-tweaking-blink-blinking-gif-15542945703716446313">猫咪抽搐 GIF - Cat Twitching Tweaking - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/batman-mad-angry-tell-me-interogating-gif-17869813">蝙蝠侠生气 GIF - Batman Mad Angry - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/biggest-boy-family-guy-chris-griffin-dancing-gif-17316116">《盖酷家庭》Biggest Boy GIF - Biggest Boy Family Guy Chris Griffin - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/subida-gif-18379274">Subida GIF - Subida - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh 猫咪 GIF - Huh Cat - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/bh187-spongebob-patrick-star-derp-duh-gif-21500047">Bh187 海绵宝宝 GIF - Bh187 Spongebob Patrick Star - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/kotmadam-odilon-old-man-no-sigh-gif-18163378">Kotmadam Odilon GIF - Kotmadam Odilon Old Man - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/hello-street-cat-huge-bite-little-scraggly-guy-kibble-gif-8033892186058013617">你好流浪猫大口吃肉 GIF - Hello street cat Huge bite Little scraggly guy - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/bugs-bunny-no-no-bunny-bugs-gif-7909500831201365932">兔八哥说不 GIF - Bugs bunny no No Bunny - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/patrick-menacingly-spongebob-standing-there-gif-19452999">派大星气势逼人 GIF - Patrick Menacingly Spongebob - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/dead-gif-18865199">死亡 GIF - Dead - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/dpowe-gif-24107728">Dpowe GIF - Dpowe - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/spongebob-spongebob-meme-spongebob-mafia-mafia-money-gif-12714856527416165903">海绵宝宝迷因 GIF - Spongebob Spongebob meme Spongebob mafia - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/caveman-spongebob-spongegar-gif-5620708">原始人海绵宝宝 GIF - Caveman Spongebob Spongegar - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/lag-android-glitch-kid-gif-15712794">安卓卡顿 GIF - Lag Android Glitch - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://youtu.be/53kr_dvof1w">我乔装成好莱坞造型去买电脑 - 垃圾场大战 2024 第一部分</a>: https://jawa.link/ScrapyardWars 感谢 Jawa 赞助本季垃圾场大战！与 Jawa 一起加入：买卖...的平台</li><li><a href="https://tenor.com/view/troll-lol-gta-gta-san-andreas-running-gif-25040072">Troll Lol GIF - Troll Lol Gta - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/kanye-west-kanye-ai-dance-gif-6290223825845382767">Kanye West AI GIF - Kanye west Kanye Ai - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/journey-car-kissing-gif-17893723">旅途车载 GIF - Journey Car Kissing - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/homelander-based-the-boys-homelander-the-boys-facts-gif-26206051">护国超人 Based GIF - Homelander Based The Boys - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/zeng-this-guy-right-here-this-right-here-point-out-point-gif-23913867">Zeng 就是这家伙 GIF - Zeng This Guy Right Here This Right Here - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/oliver-twist-gif-26543489">雾都孤儿 GIF - Oliver Twist - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/kanye-haircut-kanye-west-stare-mattscrub-gif-13171403811728519930">Kanye 理发 GIF - Kanye Haircut Kanye west - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/blog/llama31">Llama 3.1 - 405B, 70B & 8B，支持多语言和长上下文</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces?search=Whi">Spaces - Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/ye-kanye-kanye-west-dj-khaled-khaled-gif-24604192">Ye Kanye GIF - Ye Kanye Kanye West - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/learn/nlp-course">简介 - Hugging Face NLP 课程</a>: 未找到描述</li>

d</li><li><a href="https://huggingface.co">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://huggingface.co/docs">Hugging Face - 文档</a>: 未找到描述</li><li><a href="https://github.com/huggingface/huggingface-llama-recipes">GitHub - huggingface/huggingface-llama-recipes</a>: 通过在 GitHub 上创建账户，为 huggingface/huggingface-llama-recipes 的开发做出贡献。</li><li><a href="https://tenor.com/view/wizard-crawly-crawly-wizard-mall-wizard-mall-gif-14992836518596419882">Wizard Crawly GIF - Wizard Crawly Crawly wizard - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/biggest-boy-family-guy-gif-10780370951992646584">Biggest Boy Family Guy GIF - Biggest boy Family guy - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.packtpub.com/en-us/product/transformers-for-natural-language-processing-9781800565791">Transformers for Natural Language Processing | Data | eBook</a>: 使用 Python, PyTorch, TensorFlow, BERT, RoBERTa 等构建创新的 NLP 深度神经网络架构。即时交付。顶级移动应用开发产品。</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216">NVIDIA GeForce RTX 5090 规格</a>: NVIDIA GB202, 2520 MHz, 20480 Cores, 640 TMUs, 192 ROPs, 28672 MB GDDR7, 2500 MHz, 448 bit</li><li><a href="https://www.techpowerup.com/gpu-specs/voodoo3-3000-agp.c3555#:~:text=it%20might%20not%20be%20able%20to%20run%20all%20the%20latest%20games)">3dfx Voodoo3 3000 AGP 规格</a>: 3dfx Avenger, 166 MHz, 1 Pixel Shaders, 0 Vertex Shaders, 2 TMUs, 1 ROPs, 16 MB SDR, 166 MHz, 128 bit</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-5060.c4219">NVIDIA GeForce RTX 5060 规格</a>: NVIDIA GB206, 2520 MHz, 4608 Cores, 144 TMUs, 48 ROPs, 8192 MB GDDR7, 2500 MHz, 128 bit
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1265079285471903837)** (4 条消息): 

> - `Speaker Diarization & Transcription` (说话人日志与转录)
> - `Sankey Plots Visualization` (桑基图可视化)
> - `Dynamic Graph Node Management` (动态图节点管理)
> - `PEFT Model Loading Methods` (PEFT 模型加载方法)
> - `Adapter Configuration in Models` (模型中的 Adapter 配置)


- **自动化说话人日志与转录**：一位成员正在寻求一种方法，将上传的 WAV 文件的 **speaker diarization**、**whisper transcriptions** 和时间戳自动整合到单个数据库中。
   - 他们正在寻找 **开源仓库** 或模型来实现这一流水线 (pipeline)。
- **使用 Matplotlib 绘制桑基图**：一位用户分享了使用 **matplotlib** 绘制 **Sankey plots**（也称为流向图）的经验，并指出该实现在可视化数据集过滤能力方面仍有改进空间。
   - 他们表示希望进行许多 **更改**，以增强数据集过滤的可视化能力。
- **图中的动态节点管理**：一位用户询问了在图中 **动态添加和删除节点** 以逐步构建信息数据库的可行性。
   - 他们的目标是避免一次性解析大量文件的需要，从而建议一个更精简的流程。
- **PEFT 模型加载见解**：一位成员强调了加载 **PEFT 模型** 的两种方法，并为这两种技术提供了代码片段示例。
   - 他们质疑第一种方法如何从 adapter 链接中检索整个模型，推测 **adapter config** 可能包含必要的 base model 详细信息。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1265030075150106675)** (5 条消息): 

> - `Willing Suspension of Disbelief`
> - `nanoLLaVA model`
> - `Meta's Llama 3.1 release`
> - `Mark Zuckerberg's vision for open-source AI` 


- **探索叙事中的沉浸感**：一项名为 *Willing Suspension of Disbelief* 的研究调查了意志在观众参与故事中的作用，强调了深入探索叙事体验的重要性。
   - 该研究可以在[此处](https://www.researchgate.net/publication/298068504_Willing_Suspension_of_Disbelief_A_study_of_the_role_of_volition_in_the_experience_of_delving_into_a_story)访问。
- **nanoLLaVA 模型讨论**：一位成员重点介绍了 [nanoLLaVA 模型](https://huggingface.co/spaces/qnguyen3/nanoLLaVA)，并指出它是从另一个名为 *llava-next* 的模型复制而来的。
   - 对话中包含与该模型相关的图像，但未做进一步详细说明。
- **Llama 3.1 AI 模型发布**：Meta 宣布发布 Llama 3.1 系列，称赞其性能可与顶尖闭源模型相媲美，尤其是 **405B** 版本。
   - 此次发布旨在推广开源 AI 精神，并提供了 [Mark Zuckerberg 的信函](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/)，阐述了为什么开源对开发者和社区有益。
- **Mark Zuckerberg 倡导开源 AI**：Meta 首席执行官分享了他对开放 AI 生态系统的愿景，断言 Llama 3.1 的特性将帮助开发者解锁新能力，例如合成数据生成 (synthetic data generation)。
   - Zuckerberg 强调了开源 AI 的相关性，称其是开发者和 Meta 未来战略的 *前进之路 (the path forward)*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/qnguyen3/nanoLLaVA">nanoLLaVA-1.5 - a Hugging Face Space by qnguyen3</a>: 未找到描述</li><li><a href="https://www.neowin.net/news/mark-zuckerberg-explains-why-open-source-ai-is-good-for-developers/">Mark Zuckerberg explains why open source AI is good for developers</a>: Mark Zuckerberg 认为开源 AI 是 AI 的未来，能够促进不受限制的创新，类似于开源开发如何加速其他领域的进步。</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1/">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1265036255784341584)** (10 条消息🔥): 

> - `UltraPixel 高分辨率图像`
> - `Gradio 的 Rust 客户端库`
> - `SmolLM Arena 更新`
> - `YouTube 笔记生成器`
> - `Mistral-NeMo 12B Instruct` 


- **UltraPixel 创建高分辨率图像**：一位用户展示了名为 **UltraPixel** 的项目，能够在该 [链接](https://huggingface.co/spaces/gokaygokay/UltraPixel) 生成极其精细的高分辨率图像。
   - 该项目旨在突破图像生成的界限，重点关注**清晰度**和**细节**。
- **用于 Gradio 开发的 Rust 库**：宣布了一个新的 **Gradio Rust 客户端库**项目，目前正使用 `hf-audio/whisper-large-v3` 和其他模型进行积极测试，可在 [GitHub](https://github.com/JacobLinCool/gradio-rs) 上获取。
   - 该库处于**早期阶段**，邀请社区在开发过程中提供贡献和反馈。
- **SmolLM Arena 获得新界面**：**SmolLM Arena** 引入了带有聊天机器人（而非文本框）的新界面，提升了速度和用户体验，详见[此链接](https://huggingface.co/spaces/as-cle-bert/smolLM-arena)。
   - 用户现在可以**比较**小型语言模型并为他们最喜欢的模型投票，兼具趣味性和互动性。
- **YouTube 笔记生成器项目亮相**：宣布了一个新的 **YouTube Notes Generator** 项目，可以从 YouTube 视频中创建详细笔记，其代码托管在 [GitHub](https://github.com/di37/youtube-notes-generator) 上。
   - 它具有一个易于使用的 **Streamlit UI**，允许用户生成视频内容笔记并与之交互。
- **极速 Mistral-NeMo 12B Instruct 演示**：分享了一个使用 llama.cpp 的 **Mistral-NeMo 12B Instruct** 演示，展示了其极速的聊天能力，可在[此链接](https://huggingface.co/spaces/gokaygokay/Mistral-NeMo-llamacpp)体验。
   - 该项目强调在产生快速响应交互方面的性能。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/gokaygokay/UltraPixel">UltraPixel - gokaygokay 的 Hugging Face Space</a>：暂无描述</li><li><a href="https://huggingface.co/spaces/gokaygokay/Mistral-NeMo-llamacpp">Mistral NeMo llama.cpp - gokaygokay 的 Hugging Face Space</a>：暂无描述</li><li><a href="https://huggingface.co/spaces/Sergidev/HD-Pony-Diffusion-v6">HD Pony Diffusion - Sergidev 的 Hugging Face Space</a>：暂无描述</li><li><a href="https://github.com/qompassai/KO">GitHub - qompassai/KO: Kyber Odyssey: 在后 Crowdstrike 世界中为安全创新指明方向</a>：Kyber Odyssey: Charting a course for secure innovation in a post-Crowdstrike world - qompassai/KO</li><li><a href="https://github.com/JacobLinCool/gradio-rs">GitHub - JacobLinCool/gradio-rs: Rust 版 Gradio 客户端。</a>：Gradio Client in Rust. 通过在 GitHub 上创建账户为 JacobLinCool/gradio-rs 的开发做出贡献。</li><li><a href="https://github.com/di37/youtube-notes-generator">GitHub - di37/youtube-notes-generator: AI 驱动的 YouTube 笔记生成器：从 YouTube 视频创建详细笔记。易于使用的 Streamlit UI。</a>：AI-powered YouTube Notes Generator: Create detailed notes from YouTube videos. Streamlit UI for easy use. - di37/youtube-notes-generator
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1265367731813879960)** (2 条消息): 

> - `Anything V5 的动漫风格数据集`
> - `微调 SD 模型` 


- **关于 Anything V5 动漫风格数据集的咨询**：一位成员询问了 [Anything V5 API 推理](https://huggingface.co/stablediffusionapi/anything-v5) 中用于**动漫风格生成**的数据集。他们提供了一个生成图像的链接，以及获取 API key 和代码示例的信息。
   - 他们分享说 API key 不需要付费，并链接到 [Stable Diffusion API](http://stablediffusionapi.com/) 以获取更多详情。
- **关于微调 SD 模型的讨论**：一位成员询问了如何使用定制数据集微调 **Stable Diffusion (SD) 模型**。这突显了人们对为特定应用定制模型的持续兴趣。



**提及的链接**：<a href="https://huggingface.co/stablediffusionapi/anything-v5">stablediffusionapi/anything-v5 · Hugging Face</a>：暂无描述

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1265093121142816831)** (17 条消息🔥): 

> - `在 SFTTrainer 中使用非打包数据集`
> - `张量创建中的错误处理`
> - `用于数值数据的 Embedding 模型`
> - `使用 Donut 进行生成`
> - `Transformers 库中的修改`

- **SFTTrainer 中非打包数据集（Non Packed Datasets）的挑战**：一位用户询问是否有人在针对 LLM 的 **SFTTrainer** 中使用过非打包数据集，并对示例有限以及在实现过程中遇到的错误表示担忧。
   - 建议将精细的 **Prompt Engineering** 与 **PEFT** 结合使用以提高硬件效率，作为潜在的解决方案。
- **Tensor 创建错误调查**：另一位用户遇到了提示“Unable to create tensor”的错误，建议在设置期间激活 **truncation** 和 **padding** 选项。
   - 为了获得更多帮助，他们分享了一个 [Hugging Face 论坛帖子](https://discuss.huggingface.co/t/unable-to-create-tensor-you-should-probably-activate-truncation-and-or-padding-with-padding-true-truncation-true/98833) 的链接，详细说明了该问题。
- **寻求数值数据 Embedding 模型**：一位成员请求推荐针对**数值数据**优化的 **embedding** 模型，寻求专门的选项。
   - 针对该询问，没有直接建议具体的模型。
- **探索用于文本生成的 Donut**：一位用户分享了他们使用 **GitHub** 上的 **Donut** 模型进行生成的经验，强调了需要适应 **Transformers** 库两个版本之间的变化。
   - 他们链接了相关的 [GitHub Pull Requests](https://github.com/huggingface/transformers/pull/22748)，解释了针对 **Donut** 生成的调整及其影响。
- **为 LLM 切分大型 Embedded 文本**：一位用户请求关于切分大型 **embedded** 文本以有效配合 **LLM** 使用的见解。
   - 对话中没有提供解决这一问题的具体策略或方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/clovaai/donut/">GitHub - clovaai/donut: Official Implementation of OCR-free Document Understanding Transformer (Donut) and Synthetic Document Generator (SynthDoG), ECCV 2022</a>：OCR-free Document Understanding Transformer (Donut) 和 Synthetic Document Generator (SynthDoG) 的官方实现，ECCV 2022 - clovaai/donut</li><li><a href="https://discuss.huggingface.co/t/unable-to-create-tensor-you-should-probably-activate-truncation-and-or-padding-with-padding-true-truncation-true/98833">Unable to create tensor, you should probably activate truncation and/or padding with &#39;padding=True&#39; &#39;truncation=True&#39;</a>：我正尝试通过设置 ‘packing=False’ 在 SFTTrainer 中使用非打包数据集，但我遇到了错误：Unable to create tensor, you should probably activate truncation and/or padding with ‘padding=Tr...</li><li><a href="https://github.com/huggingface/transformers/pull/22748">Generate: handle text conditioning with multimodal encoder-decoder models by gante · Pull Request #22748 · huggingface/transformers</a>：此 PR 做了什么？在单一位置整合了所有未来 PT 和 TF 上的多模态 Encoder-Decoder 模型的 decoder_input_ids 准备工作。简而言之，此 PR 泛化了以下内容...</li><li><a href="https://github.com/huggingface/transformers/pull/22955">Generate: Add exception path for Donut by gante · Pull Request #22955 · huggingface/transformers</a>：此 PR 做了什么？#22748 中添加的多模态泛化导致了 Donut 的回归——Donut 永远不期望 BOS token，而是由一个特定任务的 token 代替。此 PR 添加了一个异常...</li><li><a href="https://github.com/clovaai/donut/blob/master/donut/model.py#L210">donut/donut/model.py at master · clovaai/donut</a>：OCR-free Document Understanding Transformer (Donut) 和 Synthetic Document Generator (SynthDoG) 的官方实现，ECCV 2022 - clovaai/donut</li><li><a href="https://github.com/clovaai/donut/blob/master/donut/model.py#L468">donut/donut/model.py at master · clovaai/donut</a>：OCR-free Document Understanding Transformer (Donut) 和 Synthetic Document Generator (SynthDoG) 的官方实现，ECCV 2022 - clovaai/donut</li><li><a href="https://github.com/huggingface/transformers/releases/tag/v4.29.0">Release v4.29.0: Transformers Agents, SAM, RWKV, FocalNet, OpenLLaMa · huggingface/transformers</a>：Transformers Agents 是一个新的 API，让你通过自然语言提示 Agent（即大语言模型）来使用该库和 Diffusers。然后该 Agent 将输出...</li><li><a href="https://github.com/huggingface/transformers/compare/v4.28.1...v4.29.0">Comparing v4.28.1...v4.29.0 · huggingface/transformers</a>：🤗 Transformers：用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。 - Comparing v4.28.1...v4.29.0 · huggingface/transformers
</li>
</ul>

</div>

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1265285987924316283)** (1 messages): 

> - `Background removal`
> - `Segmentation`
> - `Diffusion models` 


- **寻求背景移除指导**：一名成员请求协助使用 **Diffusion models** 和 **Segmentation** 技术实现 **背景移除 (background removal)**。
   - 他们询问是否有人能提供关于入门最佳方法或可用资源的指导。
- **对分割技术的兴趣**：另一位成员对在使用 **Diffusion models** 进行 **背景移除** 时能有效配合的 **Segmentation** 技术表示感兴趣。
   - 他们询问是否有其他人认为成功的特定模型或方法。


  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1265178267250593825)** (2 messages): 

> - `Magpie Paper`
> - `Nous Research AI`
> - `Instruction Generation Techniques` 


- **关于 Magpie 论文实用性的讨论**：一位成员询问 [Magpie 论文](https://arxiv.org/abs/2406.08464) 中的见解是代表了一种有用的技术，还是仅仅是一个 *噱头 (party trick)*。
   - 他们对生成的指令的 **质量 (quality)** 和 **多样性 (diversity)** 表示好奇。
- **Nous Research 作者与合作**：一篇著名论文的作者包括 [Jaden Fiotto-Kaufman](https://arxiv.org/search/cs?searchtype=author&query=Fiotto-Kaufman,+J)、[Alexander R Loftus](https://arxiv.org/search/cs?searchtype=author&query=Loftus,+A+R) 等。
   - 这一协作成果展示了为当前 AI 讨论做出贡献的广泛专业知识。



**提到的链接**：<a href="https://arxiv.org/abs/2407.14561">NNsight and NDIF: Democratizing Access to Foundation Model Internals</a>：最先进的基础模型规模巨大，限制了科学家对它们的可访问性，因为在大模型尺寸上进行定制化实验需要昂贵的硬件和复杂的工程……

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1265052200615284797)** (9 messages🔥): 

> - `ReFT paper discussion`
> - `YouTube video on ReFT`
> - `Oxen AI community activity`
> - `PC Agent Demo`
> - `Emoji duplication in server` 


- **ReFT 论文简化**：进行了一场由 [ReFT 论文](https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/) 主作者参与的讨论，解释了其高效的微调技术，该技术比 LoRA 的参数效率高出 15 到 60 倍。
   - 该方法在 **residual stream** 上运行，使其能够灵活地将各种训练任务与被称为“干预 (interventions)”的学习参数相结合。
- **《How ReFT Works》YouTube 视频发布**：一段名为 [How ReFT Works w/ Author Zhengxuan Wu](https://www.youtube.com/watch?v=to2oKwnknUk&t=2770s) 的新视频深入探讨了 ReFT 论文及其对机器学习的影响。
   - 视频中作者与 Greg 进行了引人入胜的解释，并与之前 Paper Club 讨论过的其他论文建立了联系。
- **活跃中的 Oxen AI 社区**：[Oxen AI 社区](https://oxen.ai/community) 正在积极成长，专注于通过每周对各种研究论文的讨论来推动 ML 和 AI 的进步。
   - 参与者可以订阅未来的 Paper Club 日历邀请，以便与学术研究人员和开发人员进行交流。
- **PC Agent 演示公开**：分享了一个名为 [PC Agent Demo](https://youtu.be/97tUynaJusY?si=pi-K8F4trJDE3Kt8) 的 YouTube 视频链接，详细介绍了 PC Agent 的功能。
   - 描述中链接了关于该演示的更多资源，表明该领域正在进行的创新。
- **表情符号重复归咎于用户**：一名成员质疑服务器上为何出现大量重复的表情符号，导致另一名成员暗示这是某个特定用户的错。
   - 这种轻松的交流点缀在关于机器学习话题的严肃讨论之中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=to2oKwnknUk&t=2770s">How ReFT Works w/ Author Zhengxuan Wu</a>：我们与作者之一 Zhengxuan Wu 一起深入探讨来自斯坦福的 ReFT 论文。--使用 Oxen AI 🐂 https://oxen.ai/ Oxen AI 让您的数据版本化……</li><li><a href="https://youtu.be/97tUynaJusY?si=pi-K8F4trJDE3Kt8">PC Agent Demo</a>：gate-app.com/research/pc-agent</li><li><a href="https://www.reddit.com/r/MachineLearning/comments/1e8qwnl/r_discussion_of_reft_paper_with_lead_author/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://oxen.ai/community">Community Resources | Oxen.ai</a>：使用 Oxen AI 管理您的机器学习数据集。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1265312427541794880)** (62 条消息🔥🔥): 

> - `Bud-E Voice Assistant`
> - `Llama 3.1 Models`
> - `Synthetic Dataset Creation`
> - `Graph RAG by Microsoft`
> - `DSPy Python Library` 


- **Bud-E Voice Assistant 势头强劲**：**Bud-E voice assistant** 的演示展示了其在无障碍辅助和开源改编方面的潜力，目前代码库已针对 **Ubuntu** 笔记本电脑进行了优化。
   - Christoph 主持每日黑客松会议，以引导新志愿者并协调项目工作，从而实现社区贡献。
- **Llama 3.1 为开源 AI 模型开辟新天地**：**Llama 3.1 405B** 模型被描述为最大的开源模型，提供可与顶级闭源替代方案媲美的能力，同时可用于商业和研究用途。
   - 开发者可以利用其功能进行 **synthetic data generation**（合成数据生成）和模型改进等任务，尽管运营成本较高。
- **关于合成数据集创建的讨论**：人们对使用 **Llama 3.1-405B** 创建合成数据集相关的成本表示担忧，从而引发了关于使用 **70B model** 替代方案可行性的询问。
   - 虽然 **70B model** 被认为足以胜任许多任务，但其在数据集创建中的成本效益仍然是一个关键讨论点。
- **Microsoft 的 Graph RAG 提案**：Microsoft 推出了 **GraphRAG**，这是一种旨在通过将 LLM 与私有数据集集成以进行语义理解和聚类来增强模型的方法。
   - 该方法旨在通过利用知识图谱（knowledge graphs）获得更好的上下文回答，提升 LLM 分析其不熟悉数据的能力。
- **DSPy Python 库发布**：一个为集成 **DSPy** 优化器而开发的新 **Python library** 声称能显著提高 AI 应用中的评估指标。
   - 该库便于集成到现有 App 中，允许开发者有效地优化其系统，并鼓励在社交媒体上进行社区互动。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/jail-right-to-jail-right-away-parks-and-recreation-parks-and-rec-gif-16177531">Jail Right To Jail GIF - Jail Right To Jail Right Away - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1/">无标题</a>: 未找到描述</li><li><a href="https://news.ycombinator.com/item?id=41013693">无标题</a>: 未找到描述</li><li><a href="https://youtu.be/7EYifjGAbg0">Cheevly 入门教程 (第 1 部分)</a>: Cheevly 入门第 1 部分。</li><li><a href="https://github.com/stanfordnlp/dspy">GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models</a>: DSPy：用于对基础模型进行编程（而非提示）的框架 - stanfordnlp/dspy</li><li><a href="http://groq.link/llama3405bblog">现已在 Groq 上可用：迄今为止最大且最强大的开源基础模型 Llama 3.1 405B - Groq 是快速 AI 推理的代表</a>: 迄今为止最大的开源基础模型 Llama 3.1 405B 现已在 Groq 上可用。Groq 很荣幸能参与这一重要的行业发布。</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">meta-llama/Meta-Llama-3.1-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE">llama-models/models/llama3_1/LICENSE at main · meta-llama/llama-models</a>: 旨在与 Llama 模型配合使用的实用程序。通过在 GitHub 上创建账户为 meta-llama/llama-models 的开发做出贡献。</li><li><a href="https://youtu.be/O4IXfa8CROs">BUD-E - 演示</a>: 加入我们的 Discord 社区，亲自尝试 BUD-E 并帮助我们构建我和 BUD-E 在视频中谈论的语音助手：https://discord.gg/sTKSB2AwBvhttps...</li><li><a href="https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/">GraphRAG：一种利用复杂信息进行发现的新方法</a>: Microsoft 正在通过 GraphRAG 改变检索增强生成（RAG），使用 LLM 生成的知识图谱在分析复杂信息时显著改善问答效果，并持续超越...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1265022548857327668)** (489 条消息🔥🔥🔥): 

> - `Llama 3.1 Performance`
> - `Quantization and Fine-tuning`
> - `Tool Calling Methods`
> - `Model Inference and Evaluation`
> - `Open Source Licensing`

- **Llama 3.1 在基准测试中表现优于竞争对手**：Llama 3.1 405B Instruct-Turbo 在 GSM8K 上排名第一，在 ZebraLogic 等逻辑推理任务中表现与 GPT-4o 和 Sonnet 3.5 接近。
   - 然而，它在 MMLU-Redux 上的表现较弱，表明在不同数据集上的结果参差不齐。
- **对 Llama 3.1 微调（Fine-tuning）的担忧**：有人担心基础模型在预训练期间的对齐（alignment）会对微调效果产生负面影响，从而可能导致结果不佳。
   - 专家希望随着用户调整训练技术，未来的微调工作将产生更好的性能。
- **关于工具调用（Tool Calling）机制的讨论**：关于 Llama 3.1 如何管理工具调用存在持续的讨论，有人推测其内部处理方式可能与用户预期不符。
   - 不同框架之间的工具调用方法进行了对比，引发了关于与现有工具兼容性的疑问。
- **Llama 3.1 的推理与性能**：用户报告 8B 量化模型具有令人印象深刻的推理速度，并将其与 GPT-4o 进行对比。
   - 这种快速性能对于需要大规模并行生成的应用至关重要。
- **开源许可变更**：Meta 更改了 Llama 3.1 的许可协议，允许使用其输出结果来改进其他模型，这标志着其开源策略的转变。
   - 此举旨在促进创新，而不限制开发者只能使用他们的模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/casper_h">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://livebench.ai">LiveBench</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Half-precision_floating-point_format">Half-precision floating-point format - 维基百科</a>：未找到描述</li><li><a href="https://huggingface.co/qresearch/llama-3.1-8B-vision-378">qresearch/llama-3.1-8B-vision-378 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/billyuchenlin/status/1815841947468353700">来自 Bill Yuchen Lin 🤖 (@billyuchenlin) 的推文</a>：对 Llama-3.1-405B-Instruct-Turbo（在 @togethercompute 上）的快速独立评估 ⬇️ 1️⃣ 它在 GSM8K 上排名第一！2️⃣ 它在 ZebraLogic 上的逻辑推理能力与 Sonnet 3.5 非常相似，并且...</li><li><a href="https://huggingface.co/collections/hugging-quants/llama-31-gptq-awq-and-bnb-quants-669fa7f50f6e713fd54bd198">Llama 3.1 GPTQ, AWQ, and BNB Quants - hugging-quants 集合</a>：未找到描述</li><li><a href="https://huggingface.co/Salesforce/xLAM-7b-fc-r">Salesforce/xLAM-7b-fc-r · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/maya-rudolph-ho-raise-the-roof-excited-comedy-gif-9228610">Maya Rudolph Ho GIF - Maya Rudolph Ho Raise The Roof - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/well-no-randy-marsh-south-park-s7e12-all-about-mormons-gif-22233922">Well No Randy Marsh GIF - Well No Randy Marsh South Park - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/meta-llama/Llama-Guard-3-8B-INT8">meta-llama/Llama-Guard-3-8B-INT8 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/SillyTilly/Meta-Llama-3.1-70B">SillyTilly/Meta-Llama-3.1-70B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mlx-community/Meta-Llama-3-70B-Instruct-4bit">mlx-community/Meta-Llama-3-70B-Instruct-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://api.together.xyz/playground/chat/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo">未找到标题</a>：未找到描述</li><li><a href="https://x.com/casper_hansen_/status/1815769985861493056">来自 Casper Hansen (@casper_hansen_) 的推文</a>：Llama 3.1 的 AWQ 模型已完成并上传 ✅ 应该可以在 vLLM 中开箱即用！链接如下👇👇👇</li><li><a href="https://github.com/meta-llama/llama-toolchain/blob/9fb50bbd99b1dcf8f85c269cef5cb0bb48266964/llama_toolchain/inference/inference.py#L68">meta-llama/llama-toolchain 中的 llama-toolchain/llama_toolchain/inference/inference.py</a>：Llama Stack APIs 的模型组件。通过在 GitHub 上创建账号来为 meta-llama/llama-toolchain 的开发做出贡献。</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct">meta-llama/Meta-Llama-3.1-405B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1/">未找到标题</a>：未找到描述</li><li><a href="https://github.com/exo-explore/exo">GitHub - exo-explore/exo: 使用日常设备在家里运行你自己的 AI 集群 📱💻 🖥️⌚</a>：使用日常设备在家里运行你自己的 AI 集群 📱💻 🖥️⌚ - exo-explore/exo</li><li><a href="https://www.deepspeed.ai/docs/config-json/">DeepSpeed 配置 JSON</a>：DeepSpeed 是一个深度学习优化库，使分布式训练变得简单、高效且有效。</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">meta-llama/Meta-Llama-3.1-8B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/casper_hansen_/status/1815764551821856833">来自 Casper Hansen (@casper_hansen_) 的推文</a>：Llama 3.1 发布了！我拿到了下载链接，但需要 Hugging Face 格式。有人有可用的 HF 链接吗？</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1">未找到标题</a>：未找到描述</li><li><a href="https://llama.meta.com/">Llama 3.1</a>：你可以随时随地进行微调、蒸馏和部署的开源 AI 模型。我们的最新模型提供 8B、70B 和 405B 版本。</li><li><a href="https://github.com/meta-llama/llama-agentic-system">GitHub - meta-llama/llama-agentic-system: Llama Stack APIs 的 Agentic 组件</a>：Llama Stack APIs 的 Agentic 组件。通过在 GitHub 上创建账号来为 meta-llama/llama-agentic-system 的开发做出贡献。</li><li><a href="https://x.com/terryyuezhuo/status/1815796835677790539">来自 Terry Yue Zhuo (@terryyuezhuo) 的推文</a>：Llama-3.1-405b-instruct 通过 @nvidia API 在 BigCodeBench-Hard 上的初步结果：Complete: 30.4 Instruct: 22.3 Average: 26.4 优于 @AnthropicAI 的 Claude-3-Opus，并接近 @OpenAI 的 GPT-4o...</li><li><a href="https://x.com/astonzhangAZ/status/1815763885380747422">来自 Aston Zhang (@astonzhangAZ) 的推文</a>：我们的 Llama 3.1 405B 现已公开可用！经过一年的专注努力，从 pr...</li>

oject 计划启动审查，我们很高兴能开源 Llama 3 系列模型并分享我们的发现……</li><li><a href="https://github.com/meta-llama/llama-agentic-system/blob/main/custom_tools/base.py">llama-agentic-system/custom_tools/base.py at main · meta-llama/llama-agentic-system</a>：Llama Stack APIs 的 Agentic 组件。通过在 GitHub 上创建账户，为 meta-llama/llama-agentic-system 的开发做出贡献。</li><li><a href="https://github.com/meta-llama/llama-toolchain">GitHub - meta-llama/llama-toolchain: Model components of the Llama Stack APIs</a>：Llama Stack APIs 的模型组件。通过在 GitHub 上创建账户，为 meta-llama/llama-toolchain 的开发做出贡献。</li><li><a href="https://tenor.com/twuK.gif">Cat Keyboard GIF - Cat Keyboard Cats - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1265125784075370658)** (18 messages🔥): 

> - `训练更大的 bitnet 模型`
> - `模型微调的差异`
> - `微调 Llama 3.0`
> - `多语言微调资源` 


- **对训练更大 bitnet 模型的兴趣**：成员们讨论了训练具有更多参数的 **1.58 bitnet** 的潜力，并指出 Hugging Face 上缺乏与 **Llama** 相当的模型。
   - 一位成员提到在 **Nous** 上发现了一个较小的模型，但对更大参数量的模型表示好奇。
- **关于 Qwen 模型差异的辩论**：一位成员推测 **Qwen2** 相对于 **Qwen1.5** 的改进可能源于更好的基座模型，而不仅仅是不同的微调技术。
   - 另一位成员质疑了基座模型的 Benchmark 在评估变化时的相关性，特别是考虑到 **Mistral Nemo** 和 **Llama-3-8b** 较低的 Benchmark 结果。
- **微调 Llama 3.0 的挑战**：随着 **Llama 3 405b** 的发布，成员们承认微调该模型面临重大挑战，特别是担心在 **Lora FTing** 之外的实际执行。
   - 一位成员表示希望这能推动开源软件中 **DoRA fine-tuning** 的成功实现。
- **普什图语（Pashto）的微调资源**：一位成员正在寻找专门针对**普什图语**微调模型的资源，强调尽管该语言有 **6000 万**使用者，但可用材料却很稀缺。
   - 另一位成员建议探索最近的研究，指出 **Aya23 model papers** 是一个潜在的指导资源。
- **多语言任务的协作**：一位成员询问了关于多语言微调工作的协作，另一位提到 **coheres** 正在这一领域进行大量工作。
   - 讨论还涉及了具有高计算需求（如某团队使用的众多 **H100s**）的微调计划的物流问题。


  

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1265220957560377374)** (4 条消息): 

> - `Kuzu Graph Database`
> - `GraphRAG and Outlines`
> - `Entity Deduplication Techniques`
> - `Property Graph Index`
> - `Duplicate Detection in Graph Databases` 


- **推荐集成 Kuzu Graph Database**：一位成员推荐尝试 **Kuzu** GraphStore，它采用 [MIT license](https://github.com/kuzudb/kuzu)，并已与 **LlamaIndex** 集成用于知识图谱。
   - 对于寻求增强型 GraphStore 功能的用户来说，这可能是一个很有前景的替代方案。
- **GraphRAG 的 outline 功能可增强输出**：关于 **GraphRAG** 的讨论强调了使用 outlines 来约束输出的潜力，这有助于执行 deduplication 任务。
   - 集成此功能可以通过减少数据输出中的冗余来简化工作流程。
- **Tomaz Bratanic 的 Entity Deduplication 方案**：针对 **entity deduplication**，文中引用了 **Tomaz Bratanic** 的深入研究，并分享了一篇[博客文章](https://neo4j.com/developer-blog/property-graph-index-llamaindex/#deduplication)。
   - 该方法结合了 text embedding 相似度与 word distance，通过 **Cypher queries** 来识别并合并重复项。
- **Property Graph Index 增强**：**Property Graph Index** 被视为 **LlamaIndex** 的一项价值升级，它现在拥有完善的 property graph 结构，增强了数据表示能力。
   - 与之前的 triple 表示法相比，这一改进允许进行更详细的 node labeling 和 property 存储。
- **Atlas 的重复检测能力**：另一位成员提到 **Atlas** 也提供 duplicate detection 功能，显示出图数据库领域的竞争态势。
   - 虽然可能需要一些数据预处理，但据报道其 duplicate detection 功能表现不错。



**提到的链接**：<a href="https://neo4j.com/developer-blog/property-graph-index-llamaindex/#deduplication">Customizing Property Graph Index in LlamaIndex</a>：了解如何使用 LlamaIndex 执行 entity deduplication 和自定义检索方法，以提高 GraphRAG 的准确性。

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/)** (1 条消息): 

jmiles38: <@414158939555364865> 你是 worldsim/world client 的贡献者吗？
  

---

### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1265026584201531473)** (74 条消息🔥🔥): 

> - `Open Reasoning Tasks`
> - `Schema and Formatting Improvements`（Schema 与格式改进）
> - `Reasoning Techniques and Tools`（推理技术与工具）
> - `Master List for Reasoning Papers`（推理论文总表）
> - `SMT Solvers for Reasoning`（用于推理的 SMT Solvers）


- **探索 Open Reasoning Tasks 框架**：讨论集中在改进 Open Reasoning Tasks 仓库的结构和美观度，建议采用能区分任务并包含示例的总表（master list）格式。
   - 输入结构的提案包括使用带有标题的 Markdown 格式和用于示例输出的表格，以平衡贡献者的清晰度和易用性。
- **纳入多模态任务**：参与者商讨了如何处理多轮任务和各种模态，考虑是否利用表格进行结构化输入，同时确保贡献者的灵活性。
   - 提出了将复杂任务排除在表格要求之外，同时允许贡献者自行决定的想法。
- **协作与未来贡献**：团队成员表达了为仓库提供更新和改进的意愿，并确认了由分享的论文引发的持续讨论。
   - 引用了外部资源，特别是在 Bayesian 推理和结构化问题解决技术方面，被强调为未来发展的宝贵输入。
- **开发推理论文总表 (Master List)**：讨论了为推理相关论文和资源创建一份全面总表的可能性，并就如何构建展示结构以提高清晰度提供了意见。
   - 示例包括潜在的标题和摘要格式，旨在增强贡献者和读者的可访问性。
- **利用 SMT Solvers 增强推理**：一位用户提到利用 SMT Solvers 将应用题翻译成 SMTLIB 格式的潜力，暗示了通过创建合成数据来增强推理能力。
   - 这种方法与最近关于将逻辑框架与 LLM 集成以提高推理应用准确性的讨论相一致。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/SMT_Solvers/status/1815856006427205672">Chad Brewbaker (@SMT_Solvers) 的推文</a>：正如我对 @Teknium1 所说，如果我们能教会 LLM 将英语/德语的应用题翻译成 SMTLIB，我们就可以通过 SMT Solvers 获得大量的推理能力。这是一个 MADLIBS 合成数据问题，如果你...</li><li><a href="https://arxiv.org/abs/2402.06557">The Quantified Boolean Bayesian Network: Theory and Experiments with a Logical Graphical Model</a>：本文介绍了量化布尔贝叶斯网络 (QBBN)，它提供了逻辑推理和概率推理的统一视角。QBBN 旨在解决大模型（Larg...）的一个核心问题。</li><li><a href="https://x.com/swarnaNLP/status/1815430142870908971">Swarnadeep Saha (@swarnaNLP) 的推文</a>：🚨 新消息：我最后一篇博士论文 🚨 介绍 System-1.x，一个基于 LLM 的可控规划框架。它从双加工理论 (Dual-Process Theory) 中汲取灵感，该理论主张快速/直觉的 Sy...</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/issues/5">Structuring suggestions · Issue #5 · NousResearch/Open-Reasoning-Tasks</a>：你好，这是一个了不起的倡议！编译一份用于语言模型评估的潜在推理任务的全面列表非常有价值。我有几个建议：这是...</li><li><a href="https://github.com/mlabonne/llm-course">GitHub - mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks.</a>：通过路线图和 Colab 笔记本进入大语言模型 (LLMs) 领域的课程。 - mlabonne/llm-course</li><li><a href="https://arxiv.org/abs/2401.14295">Demystifying Chains, Trees, and Graphs of Thoughts</a>：自然语言处理 (NLP) 领域近年来取得了显著进展，重点是通过创新的提示 (pro...) 提高大语言模型 (LLM) 的性能。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1265028243950665728)** (197 条消息🔥🔥): 

> - `LM Studio 性能`
> - `模型下载问题`
> - `Linux 与 GPU 的兼容性`
> - `Llama 3.1 能力`
> - `ROCm 安装` 


- **Llama 模型之间的性能比较**：用户讨论了 Llama 3.1 8B 和 405B 模型之间的性能差异，强调运行较大的模型需要大量的 GPU 资源。
   - *一位用户开玩笑说，为高容量模型所需的 GPU 集群供电需要“一个小国家的电力供应”*。
- **模型下载问题**：一些用户报告了下载模型的问题，将其归因于 DNS 问题，另一些用户则注意到由于 Llama 3.1 的流行导致 Hugging Face 流量增加，从而出现了减速。
   - 一位用户推测他们的问题是由 IPv6 引起的，并提到*希望在应用中有一个选项可以避免使用它*，而不影响系统全局设置。
- **Linux 上的 GPU 检测问题**：新的 Linux 用户表示 LM Studio 在检测其 GPU 时遇到困难，特别是提到从 Windows 切换到 Linux Mint 后，Radeon RX5700XT 出现的问题。
   - 一位用户指出他们已经安装了扩展包，但系统仍然无法识别其 GPU，并对 RDNA 1 的支持提出了疑问。
- **关于模型能力的讨论**：用户讨论了各种模型的功能差异，包括提到 Llama 3.1 支持多种语言以及在某些任务中表现更好。
   - 一位用户指出，对于日语，4o-mini 模型的表现优于 Llama 3.1，这显示了考虑特定用例模型的重要性。
- **ROCm 安装建议**：分享了关于为 AMD GPU 手动安装 ROCm 以提高与 LM Studio 兼容性的建议，特别是针对那些使用 Radeon 显卡遇到问题的用户。
   - 用户被引导至特定的 GitHub 页面以获取安装说明和故障排除技巧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLM</a>: 查找、下载并实验本地 LLM</li><li><a href="https://huggingface.co/YorkieOH10/Meta-Llama-3.1-8B-Instruct-hf-Q4_K_M-GGUF">YorkieOH10/Meta-Llama-3.1-8B-Instruct-hf-Q4_K_M-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://lmstudio.ai/snapdragon">👾 LM Studio - 发现并运行本地 LLM</a>: 查找、下载并实验本地 LLM</li><li><a href="https://huggingface.co/YorkieOH10/Meta-Llama-3.1-8B-Instruct-Q8_0-GGUF">YorkieOH10/Meta-Llama-3.1-8B-Instruct-Q8_0-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1/">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#amd-rocm">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8650">功能请求：在 llama.cpp 中提供完善的 Llama 3.1 支持 · Issue #8650 · ggerganov/llama.cpp</a>: 前提条件：我正在运行最新代码。如果可能请注明版本。我仔细阅读了 README.md。我使用与问题相关的关键词进行了搜索，以确保我正在创建...</li><li><a href="https://llama.meta.com/">Llama 3.1</a>: 您可以随时随地进行微调、蒸馏和部署的开源 AI 模型。我们最新的模型提供 8B、70B 和 405B 版本。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1265043935970201742)** (92 条消息🔥🔥): 

> - `Qwen 2 的高内存占用`
> - `LM Studio 模型兼容性`
> - `Meta-Llama 模型推荐`
> - `Gemini 和 Deepseek 的进展`
> - `用于高级编程的 LLM Compiler` 


- **Qwen 2 的高内存占用**：一位用户报告在使用 **llama.cpp** 加载 **Qwen 2 72B** 时内存占用极高，超过了模型本身的大小。
   - 另一位成员建议降低上下文长度（context length）以帮助管理内存利用率。
- **LM Studio 模型兼容性**：一位成员指出 LM Studio 中的模型存在兼容性问题，特别是 **Meta-Llama 3.1-8B** 和 **70B**，在当前版本中 GPU offloading 无法正常工作。
   - 其他人建议升级到 **0.2.28** 版本以获得更好的支持，因为 **llama.cpp** 的更新尚在处理中。
- **Meta-Llama 模型推荐**：关于 **Meta-Llama 3.1** 模型展开了讨论，对其性能（尤其是推理任务）评价不一。
   - 一位成员提到 **8B 版本** 逻辑较弱但尚可接受；另一位建议关注 **70B 版本** 以获得更好的输出。
- **Gemini 和 Deepseek 的进展**：对话涉及了 **Gemini Pro 1.5** 的性能及其在编程任务中的适用性，强调了其编程能力但指出其缺乏写作能力。
   - 成员们期待即将推出的模型（特别是来自 **Deepseek** 的模型）能在推理能力上有所提升。
- **用于高级编程的 LLM Compiler**：一位成员推荐了基于 **Code Llama** 构建的 **LLM Compiler**，用于涉及高级编程概念的任务，并提到它支持代码优化和编译器推理。
   - 该模型提供 **7B** 和 **13B** 版本，适合显存（VRAM）有限的用户的内存容量需求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.producthunt.com/posts/llama-7"> Llama - 3.1-405B：一个可与 GPT-4o / Claude-3.5 竞争的开源模型 | Product Hunt</a>：Meta 正在发布三个模型：全新的 3.1-405B 以及对其较小模型的升级：3.1-70B 和 3.1-8B。如果 405B 表现如基准测试所示，这将是开源模型首次...</li><li><a href="https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f">Llama 3.1 - meta-llama 集合</a>：未找到描述</li><li><a href="https://x.com/YouJiacheng/status/1815817670954213710">YouJiacheng (@YouJiacheng) 的推文</a>：刚看到 deepseek-coder 将在 7 月 24 日 10:00 UTC+8 进行升级。</li><li><a href="https://github.com/xcapt0/gpt2_chatbot">GitHub - xcapt0/gpt2_chatbot: ☕ 用于日常对话的 GPT-2 聊天机器人</a>：☕ 用于日常对话的 GPT-2 聊天机器人。通过在 GitHub 上创建账号为 xcapt0/gpt2_chatbot 的开发做出贡献。</li><li><a href="https://dubesor.de/benchtable">Dubesor LLM 基准测试表</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1265039071646974127)** (1 条消息): 

> - `搜索功能` 


- **搜索功能回归！**：影响应用内搜索功能的问题现已**解决**，用户现在可以重新进行搜索。
   - 对停机期间造成的**不便**深表歉意。
- **提高解决透明度**：团队承诺提供有关应用问题的更新，确保用户了解已修复的功能。
   - 用户对有关搜索功能状态的及时沟通表示赞赏。


  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1265025042324914177)** (4 条消息): 

> - `hf-mirror.com`
> - `Llama 3.1 模型的 Latex 支持` 


- **hf-mirror.com 展示了巨大的潜力**：一位成员介绍了 [hf-mirror.com](https://hf-mirror.com)，这是一个 Hugging Face API 的镜像网站，其源代码可在 [GitHub](https://github.com/padeoe/hf-mirror-site) 上获得，尽管目前是中文界面。
   - 该网站使用 Caddy 作为反向代理，并提供了一个用于断点续传的脚本 [hfd.sh](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f)，建议 LM Studio 可以通过集成这些功能来显著提高用户的适应性。
- **Llama 3.1 即将支持 Latex**：一位成员对新 Llama 3.1 模型中*巨大的* **Latex 支持**表示热切期待，强调了这对于询问数学和编程相关问题的用户的重要性。
   - 另一位成员确认 **Latex 支持**即将推出，以满足社区对增强数学能力的需求。


  

---

### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1265238329943654410)** (12 条消息🔥): 

> - `Llama 3 Configuration`
> - `GPU Settings`
> - `Roleplay Scenarios`
> - `Context Length Settings` 


- **寻求关于 Llama 3 角色扮演设置的建议**：一位用户正在寻求在 LMStudio 中设置角色扮演场景的指导，试图防止助手为用户角色编写对话或动作。
   - *鉴于基于 Llama 3 模型的进展，我正重新深入研究 LMStudio。*
- **Llama 3.1 的配置设置**：一位用户请求关于 Llama 3.1 配置值的建议，因为他们是初次接触该设置。
   - 另一位成员在确认其已更新到 v0.2.28 后，建议使用 Llama 3 的 v2 预设。
- **上下文长度建议**：讨论显示 **Llama 3** 支持高达 **128k** 的上下文长度，并建议将其设置为 **32k** 以实现最佳 GPU 利用率。
   - 一位用户询问是否应将上下文长度保持在 **2048**，不确定之前的增加操作是否正确。
- **GPU 兼容性问题**：一位用户提到 Llama 3.1 似乎无法完全加载到他们的 GPU（具体为 **3080ti**）中。
   - 在将上下文长度设置为最大值 **(-1)** 后，用户注意到重新加载时它会恢复原状。


  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1265022201867010048)** (23 条消息🔥): 

> - `Fine-tuning with 3090s`
> - `GGUF Fine-tuning Limitations`
> - `GPU Acceleration on RX 6700 XT`
> - `Quantized Model Fine-tuning`
> - `GPU Requirements for LLMs` 


- **使用双 3090 进行微调**：一位成员考虑购买 **两块二手 3090** 进行微调，另一位成员指出这适用于最高 **13b** 的模型，尽管速度较慢。
   - 建议考虑 **租用 GPU** 来进行自定义模型微调，以获得更好的效率。
- **微调 GGUF 的挑战**：有人声称微调 **GGUF** 基本上是不可能的，一位成员表示这可能会产生较差的结果。
   - 然而，另一位成员指出量化 LLM 可以进行微调，但结果可能会损坏模型的权重。
- **RX 6700 XT 缺乏 GPU 加速支持**：一位用户询问 Linux 上 **RX 6700 XT** 的 GPU 加速情况，确认由于 **OpenCL 弃用** 而不支持。
   - 成员们强调 RX 6700 XT 不支持 **ROCM**，进一步限制了其能力。
- **量化模型微调见解**：围绕使用 **unsloth/QLora** 等方法微调 **量化模型** 的可行性展开了讨论，尽管可能存在问题。
   - 成员们澄清说，支持的量化模型通常是 **bitsandbytes/awq quantized**，而不支持 GGUF。
- **运行 LLM 的 GPU 需求**：有人指出，为了实现 LLM 的 **GPU 加速**，相比 RX 系列等 AMD 模型，更推荐使用兼容的 NVIDIA GPU。
   - 成员们引用了 **LM Studio** 网站上关于支持硬件的指南，强调 NVIDIA GPU 可以“即插即用”。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/fine-tuning">LLM 微调入门</a>：大语言模型 (LLM) 微调是将预训练模型适应特定任务的过程。该过程通过在核新数据集上更新其参数来完成。具体来说，LLM 会经过...
</li>
</ul>

</div>

### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1265124756294991943)** (112 条消息🔥🔥): 

> - `Beta UI Improvements` (Beta UI 改进)
> - `Feedback on Model Loading` (模型加载反馈)
> - `User Experience Concerns` (用户体验关注点)
> - `Issues with GPU Usage` (GPU 使用问题)
> - `Beta Testing Process` (Beta 测试流程)


- **Beta UI 收到褒贬不一的反馈**：用户赞赏 [Beta 1](https://lmstudio.ai) 中新 UI 的简洁性，但一些人觉得重要功能被隐藏在了过多的标签页和菜单之后。
   - 一些用户认为界面需要为追求深度定制的用户保留高级设置。
- **模型加载参数造成困惑**：几位用户报告称，在新界面中很难找到并使用 Batch size 和 GPU offloading 设置等模型加载参数。
   - 反馈提到增加了 *mmap* 等功能，对于习惯旧版本的用户来说，最初可能并不清晰。
- **GPU 自动设置未能有效利用硬件**：用户注意到，将 GPU layers 设置为 auto 并不能有效利用可用的 GPU，特别是在 4080S 等高性能 GPU 平台上。
   - 手动设置在 GPU 使用方面表现更好，这引发了关于自动功能应如何运作的疑问。
- **Beta 测试流程和反馈处理**：社区强调了 Beta 测试期间反馈的重要性，一些用户积极鼓励他人报告 Bug 或建议。
   - 参与者对早期的 Bug 修复表示感谢，并鼓励在 LM Studio 的持续开发方面进一步提高透明度。
- **系统设置和限制的澄清**：一些用户寻求澄清为什么存在某些系统资源限制，例如限制为 8 个 CPU threads，特别是对于高端系统。
   - 其他人分享了他们对新功能的体验，承认由于功能重新设计而产生的最初误解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/blog/lms#bootstrap-lms-on--your-system">Introducing `lms` - LM Studio's companion cli tool | LM Studio</a>：今天，随 LM Studio 0.2.22 一起，我们发布了第一个版本的 lms —— LM Studio 的配套 cli 工具。</li><li><a href="https://forms.gle/kDYvduhQmDZmeKkG7">LM Studio 0.3.0 - Private Beta Sign Up</a>：感谢您有兴趣帮助测试我们即将发布的版本。LM Studio 0.3.0 包含大量新功能，我们希望在向全球发布之前，能得到您的帮助来排查 Bug...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1265061480605417482)** (5 条消息): 

> - `ROCm 0.2.28 performance issues` (ROCm 0.2.28 性能问题)
> - `Llama 3.1 compatibility with AMD cards` (Llama 3.1 与 AMD 显卡的兼容性)


- **ROCm 0.2.28 导致推理变慢**：更新到 **0.2.28 ROCm preview** 后，一位用户注意到推理性能显著下降，只有一张 **7900XT** 显卡显示 **150w** 功耗，而不是通常两张显卡各 **300w**。
   - 用户回退到 **0.2.27** 后恢复了性能，并请求他人调查 **0.2.28** 的 **inference** 发生了什么变化。
- **Llama 3.1 需要针对 AMD 的 Tokenizer 修复**：一位用户表示有兴趣在 AMD 显卡上运行 **Llama 3.1**，但提到 **llama.cpp** 无法识别 **smaug-bpe** 作为 tokenizer。
   - 他们强调这个问题是实现 AMD 硬件兼容性需要解决的挑战。


  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1265343880698662983)** (1 条消息): 

> - `Llama 3.1`
> - `longer context improvements` (更长上下文改进)


- **Llama 3.1 发布带来令人兴奋的更新**：**Llama 3.1** 现已发布，**8B 模型**目前已在 [Hugging Face 页面](https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF) 上线供下载。
   - 鼓励用户尝试，因为与初始版本相比，它有**巨大改进**，特别是支持高达 **128k** 的**更长上下文**。
- **鼓励下载 Llama 3.1**：消息强调需要立即**下载 Llama 3.1** 以体验其增强功能。
   - 凭借其在长上下文方面的**性能提升**，强烈建议用户参与体验。


  

---

### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1265242937621479435)** (4 条消息): 

> - `Mistral 下载问题`
> - `VPN 连接问题`
> - `用于评分的 LLM 模型`
> - `CHROMA 数据使用` 


- **VPN 环境下 Mistral 下载失败**：一名成员在远程桌面通过 VPN 连接时，在 LM Studio 中遇到了 **Mistral** 的 **下载失败** 问题。
   - *已知代理（Proxies）无法在模型浏览器（model explorer）中正常工作，* 这使得解决该问题变得具有挑战性。
- **使用 LLM 进行评分**：一位用户正在开发一个用于评分的 **LLM 模型**，利用答案文件和文档文件来约束 Bot 的回答。
   - 他们对如何有效地利用输入到 **CHROMA** 中的数据来实现这一目标表示困惑。


  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1265397562408435742)** (1 条消息): 

> - `Llama 3.1 405B`
> - `Perplexity 移动端应用` 


- **Llama 3.1 405B 在 Perplexity 上线**：被誉为最强开源模型的 **Llama 3.1 405B** 模型现已在 Perplexity 上可用，可与 **GPT-4o** 和 **Claude Sonnet 3.5** 媲美。
   - 此次发布标志着该平台可用能力的重大提升。
- **Llama 3.1 即将集成至移动端**：Perplexity 接下来正致力于将 **Llama 3.1 405B** 添加到其移动端应用程序中，承诺提供对该先进模型的无缝访问。
   - 随着开发工作的推进，建议用户关注后续更新。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1265022026515480588)** (273 条消息🔥🔥): 

> - `Llama 3.1 405B 的性能`
> - `Llama 3.1 405B 与 Claude 3.5 Sonnet 的对比`
> - `Perplexity AI 的功能与问题`
> - `对 AI 回复的反馈`
> - `API 及使用体验` 


- **Llama 3.1 405B 性能担忧**：用户对 **Llama 3.1 405B** 的表现表示不满，称其经常重复答案，且无法有效处理 Prompt，特别是在处理亚洲符号时。
   - 许多人正在考虑切换回 Claude 3.5 Sonnet，以获得更好的速度和性能。
- **AI 模型对比评估**：一些用户认为，尽管 **Llama 3.1 405B** 是开源 AI 的重大进步，但在编程任务中可能无法超越 Claude 3.5。
   - 其他人指出，与 Llama 相比， Sonnet 3.5 在速度和处理编程咨询方面仍然表现出色。
- **Perplexity AI 功能问题**：有报告称 **Llama 3.1 405B** 在 Perplexity AI 上无法正常运行，引发了对其在不同平台上的状态和稳定性的疑问。
   - 用户建议等待几天再评估性能，因为之前的模型在初始发布后都有所改进。
- **对 AI 回复的反馈**：几位用户评论称模型无法正确理解或生成某些符号，导致评价褒贬不一。
   - 反馈表明，虽然 Llama 可以简化概念，但其整体功能可能落后于 Claude 等竞争对手。
- **API 使用体验**：用户讨论了不同供应商之间的体验差异，指出 AWS 和 Fireworks 在 Llama 新版本上存在特定问题。
   - 有人提到，通过 Perplexity AI 平台访问模型可能与其他应用程序有所不同，预计随着时间的推移会有所改进。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://jason-mendoza.vercel.app/">
      Jason Mendoza - Fullstack Developer (Web &amp; Blockchain &amp; AI Tech)
    </a>: 未找到描述</li><li><a href="https://x.com/rypearts/status/1815868829169328349?s=61">来自 Ryan Putnam (@RypeArts) 的推文</a>: ✧ 　 　 ✧ ˚ * 　 　.　 　　　　 　　 · · 　　 　 + ✧ 　　　 · 　 · ˚ . 𝓈𝓊𝓂𝓂ℯ𝓇 𝓋𝒾𝒷ℯ𝓈</li><li><a href="https://x.com/perplexity_ai/status/1603441221753372673?lang=en">来自 Perplexity (@perplexity_ai) 的推文</a>: 介绍 Bird SQL，这是一个由 Perplexity 结构化搜索引擎驱动的 Twitter 搜索界面。它使用 OpenAI Codex 将自然语言翻译成 SQL，让每个人都能...</li><li><a href="https://scale.com/leaderboard">SEAL 排行榜</a>: 未找到描述</li><li><a href="https://x.com/perplexity_ai/status/1815431484767142272?s=61">来自 Perplexity (@perplexity_ai) 的推文</a>: 懂的都懂。</li><li><a href="https://tenor.com/view/cryptoflash-crypto-flash-tattoo-vintage-gif-27569875">Cryptoflash Tattoo GIF - Cryptoflash Crypto Flash - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.aboutamazon.com/news/aws/meta-llama-3-1-models-AWS-generative-ai">Meta 的 Llama 3.1 模型已在 AWS 上提供，用于生成式 AI 应用</a>: Meta 最先进的大语言模型 (LLMs) 为客户在构建、部署和扩展生成式 AI 应用时提供了更多选择。</li><li><a href="https://x.com/minchoi/status/1815812112796565690">来自 Min Choi (@minchoi) 的推文</a>: Llama 3.1 8B + Groq 的即时智能太疯狂了 🤯 </li><li><a href="https://tenor.com/bnLYV.gif">Balloons Up GIF - 气球升起 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://build.nvidia.com/explore/discover#llama-3_1-405b-instruct">尝试 NVIDIA NIM APIs</a>: 立即体验领先的模型来构建企业级生成式 AI 应用。</li><li><a href="https://lexfridman.com/aravind-srinivas-transcript/#:~:text=(01%3A46%3A42,by%20the%20way.">Aravind Srinivas 访谈录：Perplexity CEO 谈 AI 的未来、搜索与互联网 | Lex Fridman Podcast #434 - Lex Fridman</a>: 这是 Lex Fridman Podcast #434 与 Aravind Srinivas 的访谈录。访谈录中的时间戳是可点击的链接，可直接带您进入视频中的相应时间点。请注意...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1265065560614633613)** (12 messages🔥): 

> - `Dark Oxygen`
> - `Mercury's Diamonds`
> - `Beach-Cleaning Robots`
> - `Munger's Inversion Technique`
> - `Llama 3 Release` 


- **暗氧 (Dark Oxygen) 的发现**：一场关于最近发现的**暗氧 (Dark Oxygen)**的讨论展开，强调了其对大气研究的潜在影响。
   - 成员们对**暗氧 (Dark Oxygen) 的本质**及其在生态平衡中的作用表示好奇。
- **水星钻石 (Mercury's Diamonds) 的探索**：聊天中重点介绍了关于**水星钻石 (Mercury's Diamonds)**的发现，分享了当前研究的迷人见解。
   - 参与者对可能导致该行星上钻石形成的**地质过程**非常感兴趣。
- **海滩清洁技术的创新**：海滩清洁机器人是一个热门话题，展示了有效针对海洋污染的**新型机器人技术**。
   - 社区讨论了这些机器人对**海洋生态系统**的潜在影响，并强调了来自试验的实时数据。
- **芒格逆向思维法 (Munger's Inversion Technique) 解析**：分享的一段 [YouTube 视频](https://www.youtube.com/embed/EtjBA3DGCrg) 重点介绍了**芒格逆向思维法 (Munger's Inversion Technique)**，详细说明了它如何应用于决策。
   - 观众被鼓励**考虑使用这种技术**，以便在日常生活中进行更好的批判性思考。
- **Meta 发布 Llama 3**：一个值得注意的亮点是 Meta 发布了 **Llama 3**，引发了对其先进能力的关注。
   - 社区讨论了 **Llama 3** 在各种 AI 任务中的潜在应用及其对开发者的影响。



**Link mentioned**: <a href="https://www.youtube.com/embed/EtjBA3DGCrg">YouTube</a>: 未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1265344369943384206)** (13 messages🔥): 

> - `Llama model updates`
> - `Perplexity API and DSGVO`
> - `Search site limitations` 


- **关于 API 中 Llama 3.1 模型的咨询**：用户对在 Perplexity API 中添加 **Llama 3.1 405b** 模型表示出兴趣，一些人询问了其可用性的细节。
   - 一位用户特别问道：“是否有计划在 API 中提供 Llama 3 405b？”这引发了后续的查询。
- **关于使用特定站点搜索方法的澄清**：一位用户建议利用 `site:example.com` 或 `site:arxiv.org` 进行学术搜索，表明可以将搜索限制在特定域名。
   - 然而，他们指出存在一个限制，即每个请求只能检索来自 **5 个来源**的结果。
- **Perplexity API 的隐私合规性咨询**：一位用户提出了关于 **Perplexity API** 是否符合 DSGVO 标准的问题，寻求其对数据保护条例合规性的澄清。
   - 另一位用户分享了[服务条款链接](https://www.perplexity.ai/hub/legal/perplexity-api-terms-of-service)，并提到其中引用了 GDPR 合规性。

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1265049794116583554)** (282 条消息🔥🔥): 

> - `Stable Diffusion models comparison` (Stable Diffusion 模型对比)
> - `Training Lycoris and Loras` (训练 Lycoris 和 LoRA)
> - `Community perceptions of Stable Diffusion` (社区对 Stable Diffusion 的看法)
> - `New developments in AI models` (AI 模型的新进展)
> - `General discussions and inquiries` (一般性讨论与咨询)


- **当前 AI 模型排名**：用户讨论了他们对 AI 模型的排名，其中 **Kolors** 被评为最高，随后是 **Auraflow**、**Pixart Sigma** 和 **Hunyuan**。
   - **Kolors** 因其速度和性能受到关注，符合用户对 **SD3** 的预期。
- **使用 ComfyUI 训练 Lycoris**：讨论了目前训练 Lycoris 的能力，提到了 **Kohya-ss** 等工具以及 **Onetrainer** 中可能的更新。
   - 用户对 **Kohya-ss** 的兼容性问题表示沮丧，特别是需要 Python 3.10.9 或更高版本。
- **社区对 Stable Diffusion 的情绪**：用户表达了对 **Stable Diffusion** 社区看法的观点，认为最近的批评可能源于对模型许可（licensing）的误解。
   - 一些用户指出了针对 **Stability AI** 的营销策略和感知到的恶意言论（toxicity）。
- **AI 采样技术的更新**：引入了一个新的采样器节点，实现了 Strong Stability Preserving Runge-Kutta 和隐式可变步长求解器，引起了用户的兴趣。
   - 用户讨论了这些新方法可能为 AI 模型带来的潜在性能提升。
- **关于 AI 和个人经历的闲聊**：用户分享了关于 AI 的个人经历，例如学习新的编程语言，以及讨论影响他们注意力的健康决策。
   - 围绕 AI 在各种日常应用中的使用进行了随意的交流。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://m3.material.io/```">
   Material Design
  </a>: 更快地构建美观、易用的产品。Material Design 是一个可扩展的系统——由开源代码支持——帮助团队构建高质量的数字体验。</li><li><a href="https://chat.tune.app/">Tune Chat - 由开源 LLM 驱动的聊天应用</a>: 通过 Tune Chat，访问 Prompt 库、PDF 聊天和 Brand Voice 功能，以增强内容写作和分析，并在所有创作中保持一致的基调。</li><li><a href="https://huggingface.co/dataautogpt3/PixArt-Sigma-900M">dataautogpt3/PixArt-Sigma-900M · Hugging Face</a>: 未找到描述</li><li><a href="https://llama.meta.com/">Llama 3.1</a>: 您可以在任何地方进行微调、蒸馏和部署的开源 AI 模型。我们最新的模型提供 8B、70B 和 405B 版本。</li><li><a href="https://youtu.be/ROCKGuuviis?feature=shared&t=33">Dennis reads Charlie's campaign speech</a>: 出自《费城永远阳光灿烂》第 2 季第 8 集。</li><li><a href="https://tenor.com/view/jump-to-conclusion-think-again-go-wild-moot-no-gif-17140256">Jump To Conclusion Think Again GIF</a>: 点击查看 GIF</li><li><a href="https://studio.tune.app/">TuneStudio</a>: 未找到描述</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1e8d4l3/new_twostage_pixart_ensemble_of_experts_2x_900m/">New two-stage PixArt ensemble of experts (2x 900M)</a>: 由 u/terminusresearchorg 发布于 r/StableDiffusion • 98 分和 33 条评论</li><li><a href="https://civitai.com/models/64471/real-mechanical-parts?modelVersionId=460254">Real Mechanical Parts - RealMech*Pony Alpha v1 | Stable Diffusion LoRA | Civitai</a>: 关于 PONY XL - 真实机械零件版本！！！！重要的是你应该知道，它处于 alpha 阶段，还有很大的改进空间。我...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1265336536107454466)** (41 条消息🔥): 

> - `Llama 3 405B 发布`
> - `模型性能对比`
> - `OpenRouter 功能更新`
> - `提示词竞赛公告`
> - `DeepSeek Coder V2 推理提供商` 


- **Llama 3 405B 以极具竞争力的价格发布**：**Llama 3 405B** 已经发布，定价为 **$3/M tokens**，足以与 **GPT-4o** 和 **Claude 3.5 Sonnet** 竞争，并为合成数据生成提供了令人印象深刻的 **128K token 上下文**。
   - 用户表现出极大的兴奋，评论如 *“天哪！太疯狂了，这现在是最好的开源 LLM”* 以及 *“多么巨大的飞跃”*，突显了其预期的影响力。
- **用户对模型性能的反馈**：关于 Llama 3 405B 性能的反馈开始出现，一位用户指出在翻译任务中它 *“比 gpt4o 差，甚至无法与 claude 3.5 相比”*。
   - 有人担心 **70B 版本** 在输出几个 token 后会产生 *“乱码”*，而 **405B** 则被拿来与 **gemini 1.5 pro** 进行比较。
- **OpenRouter 功能更新**：OpenRouter 宣布了新功能，包括 **追溯发票 (Retroactive Invoices)**、**自定义密钥 (custom keys)** 以及 **Playground** 的改进。
   - 鼓励用户在 [OpenRouter Chat](http://openrouter.ai/chat) 对新产品提供反馈，以提升用户体验。
- **多模型提示词竞赛**：推出了一项 **多 LLM 提示词竞赛**，邀请用户为 **Llama 405B**、**GPT-4o** 和 **Sonnet** 提交具有挑战性的提示词 (Prompts)，有机会赢取 15 个免费额度。
   - 竞赛旨在测试这些模型的极限，用户正热切期待公布结果的公告。
- **DeepSeek Coder V2 推理提供商**：宣布了 **DeepSeek Coder V2** 的新 **私有推理提供商**，该提供商在运行时不进行输入训练。
   - 用户可以通过 [DeepSeek Coder](https://openrouter.ai/models/deepseek/deepseek-coder) 探索新的提供商，这丰富了 OpenRouter 的产品线。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM 聊天室是一个多模型聊天界面。添加模型并开始聊天！聊天室将数据本地存储在您的浏览器中。</li><li><a href="https://openrouter.ai/chat?models=meta-llama/llama-3.1-405b-instruct,openai/gpt-4o,anthropic/claude-3.5-sonnet">Chatroom | OpenRouter</a>: LLM 聊天室是一个多模型聊天界面。添加模型并开始聊天！聊天室将数据本地存储在您的浏览器中。</li><li><a href="https://x.com/OpenRouterAI/status/1815860614755147961">来自 OpenRouter (@OpenRouterAI) 的推文</a>: DeepSeek Coder V2 现在在 OpenRouter 上有一个私有提供商处理请求，且不进行输入训练！在这里查看：https://openrouter.ai/models/deepseek/deepseek-coder</li><li><a href="https://x.com/OpenRouterAI/status/1815837707505131699">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 🏆 多 LLM 提示词竞赛。在下方回复对 Llama 405B、GPT-4o 和 Sonnet 具有挑战性的提示词！获胜者将获得 15 个免费额度 ✨。示例：</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct">Meta: Llama 3.1 405B Instruct (由 meta-llama 提供)</a>: 备受期待的 400B 级 Llama3 来了！拥有 128k 上下文和令人印象深刻的评估分数，Meta AI 团队继续推动开源 LLM 的前沿。Meta 最新的 c...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct">Meta: Llama 3.1 8B Instruct (由 meta-llama 提供)</a>: Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。这个 8B 指令微调版本快速且高效。与 ... 相比，它展示了强大的性能。</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-70b-instruct">Meta: Llama 3.1 70B Instruct (由 meta-llama 提供)</a>: Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。这个 70B 指令微调版本针对高质量对话场景进行了优化。它展示了强...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1265021926913605692)** (190 条消息🔥🔥): 

> - `Llama 405B Model Performance` (Llama 405B 模型性能)
> - `Custom API Keys Integration` (自定义 API Keys 集成)
> - `Comparison of Llama Models` (Llama 模型对比)
> - `Prompting Competition for Llama 405B` (Llama 405B Prompt 竞赛)
> - `Fine-Tuning and Instruction Challenges` (微调与指令挑战)


- **Llama 405B 模型展现出强大的能力**：用户讨论了新发布的 **Llama 405B 模型** 的性能，指出其在推理能力方面表现出色，特别是在英语环境下。不过，也有人提到与 **Claude** 和 **GPT-4** 等模型相比，它在其他外语方面的表现仍有差距。
   - 一些用户发现该模型会产生无意义的回复，不同用户的体验各不相同。
- **访问自定义 API Keys**：讨论涉及了获取**每个供应商自定义 API Keys** 的流程，强调这种集成方式可能因供应商而异，并可能涉及特定的账户设置。
   - 用户渴望了解如何有效地管理和利用这些 Key。
- **Llama 3 与 Llama 3.1 的对比**：参与者对比了 **Llama 3** (8B/70B) 和 **Llama 3.1**，强调 3.1 是从更大的 **405B 模型** 蒸馏而来的，并将上下文长度限制从 8k 提升到了 **128k**。
   - 新版本预计在各种 Benchmark 中表现更好。
- **Llama 405B Prompt 竞赛**：Alex Atallah 宣布了一项针对 Llama 405B 的 **Prompt 竞赛**，获胜者将获得 **15 个免费积分**。竞赛重点是那些能挑战模型能力的 Prompt。
   - 参与者对竞赛的标准感到好奇，特别是关于什么才算是一个“困难”的 Prompt。
- **使用指令模型（Instruction Models）的挑战**：几位用户报告了在使用 **Instruct 模型** 时的 Bug，特别是提到了在多轮对话场景中调用 JSON 响应的问题。
   - 参与者正在分享代码片段和排错技巧，以努力解决这些挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://embracethered.com/blog/posts/2024/chatgpt-gpt-4o-mini-instruction-hierarchie-bypasses/"> Breaking Instruction Hierarchy in OpenAI&#39;s gpt-4o-mini &middot;  Embrace The Red</a>: 未找到描述</li><li><a href="https://x.com/elder_plinius/status/1815759810043752847">来自 Pliny the Prompter 🐉 (@elder_plinius) 的推文</a>: 🌩️ JAILBREAK ALERT 🌩️  META: PWNED 🦾😎 LLAMA-3-405B: LIBERATED 🦙💨  快来见证全新的 SOTA 开源 AI 输出家庭实验室生物武器指南、如何黑进 WiFi、受版权保护的歌词，以及……</li><li><a href="https://llama.meta.com/llama-downloads/">下载 Llama</a>: 申请 Llama 访问权限。</li><li><a href="https://tenor.com/view/the-shawshank-redemption-pie-finger-gif-23305361">肖申克的救赎 GIF - The Shawshank Redemption - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/OpenRouterAI/status/1815837707505131699">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 🏆 多 LLM Prompt 竞赛。在下方回复对 Llama 405B、GPT-4o 和 Sonnet 具有挑战性的 Prompt！获胜者将获得 15 个免费积分 ✨。示例：</li><li><a href="https://openrouter.ai/docs/integrations">集成 (Beta) | OpenRouter</a>: 在 OpenRouter 中使用你自己的供应商 Key</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e9hg7g/azure_llama_31_benchmarks/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://github.com/vikyw89/llmtext">GitHub - vikyw89/llmtext: 一个简单的 LLM 库</a>: 一个简单的 LLM 库。通过创建账户为 vikyw89/llmtext 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1265040534259634306)** (7 messages): 

> - `Register Allocation in Flash Attention`（Flash Attention 中的寄存器分配）
> - `Kernel Fusion of Q, K, V Projections`（Q, K, V 投影的 Kernel Fusion）
> - `Challenges with SVD Parallelization`（SVD 并行化的挑战）
> - `Open Source GPU Kernel Modules`（开源 GPU 内核模块）


- **Flash Attention 中的寄存器分配问题**：@vkaul11 询问了 **Flash Attention** 中寄存器的显式分配，对寄存器与 shared memory 的协同使用表示困惑。
   - 该问题强调了在 CUDA 编程中需要明确如何高效管理寄存器资源。
- **关于 Q, K, V 投影的 Kernel Fusion 疑问**：有人提出疑问，询问 **Q, K 和 V** 矩阵的初始投影是否可以融合（fused）进单个 kernel 中，并对其在大尺寸下的可行性表示担忧。
   - 这指向了关于优化神经网络计算中内存和处理需求的持续讨论。
- **SVD 的并行化难题**：@danikhan632 指出，虽然 **SVD** 难以并行化，但相比将数据传回 **CPU**，并行化仍是首选。
   - 社区还表现出开发用于 SVD 的 **Triton kernel** 的兴趣，这暗示了一个潜在的社区项目，旨在实现更优化的计算。
- **NVIDIA 转向开源 GPU 内核模块**：分享了一个关于 NVIDIA 转向 **开源 GPU 内核模块** 的链接，该进程始于 2022 年 5 月的 R515 驱动程序，支持 **GPL** 和 **MIT** 双重许可。
   - 该更新概述了性能提升和诸如**异构内存管理（heterogeneous memory management）**等功能，并承诺将完全取代闭源驱动程序。



**提到的链接**：<a href="https://developer.nvidia.com/blog/nvidia-transitions-fully-towards-open-source-gpu-kernel-modules/">NVIDIA Transitions Fully Towards Open&#x2d;Source GPU Kernel Modules | NVIDIA Technical Blog</a>：伴随 R515 驱动程序，NVIDIA 于 2022 年 5 月发布了一套开源的 Linux GPU 内核模块，采用 GPL 和 MIT 双重许可。初始版本针对数据中心计算 GPU……

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1265395795671384195)** (9 messages🔥): 

> - `torch.compile performance`（torch.compile 性能）
> - `Bert model inference issues`（Bert 模型推理问题）
> - `CUDA graphs usage`（CUDA graphs 使用）
> - `PyTorch profiler tools`（PyTorch profiler 工具）
> - `Inductor configuration changes`（Inductor 配置更改）


- **torch.compile 在小型 Bert 模型上导致内存问题**：一位成员报告在小型 **Bert** 模型上测试 `torch.compile` 进行推理时，观察到显著的 RAM 占用，迫使 batch size 从 **512** 降低到 **160**。
   - 他们发现其性能比使用大 batch size 的 eager mode 还要慢，尽管模型通过 `full_graph=True` 成功编译，表明架构没有问题。
- **关于 CUDA graphs 利用率的疑问**：另一位成员询问是否使用了 **CUDA graphs** 以及是否使用了最新的 nightlies 版本，这表明可能需要通过调整来提高性能。
   - 他们强调这可能会影响 `torch.compile` 过程的整体有效性及其内存影响。
- **使用 PyTorch profiler 获取深度见解**：为了进一步调查，一位成员建议结合使用 **PyTorch profiler** 和 memory trace 工具，以分析底层发生的情况。
   - 该工具可以为推理过程中的内存使用模式和低效之处提供有价值的见解。
- **Inductor 配置查询**：一位成员询问是否更改了 Inductor 配置，或者是否使用了默认设置调用 `torch.compile`。
   - 标准配置结合 `inference_mode` 错误也可能导致观察到的内存挑战。
- **不同编译模式没有效果**：用户确认，无论在编译调用中使用 `reduce-overhead` 还是 `fullgraph` 选项，内存使用量都保持不变。
   - 这种一致性表明，可能有其他因素在影响推理期间的内存消耗。


  

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1265325582766313482)** (17 messages🔥): 

> - `Meta Llama 3.1 Release`
> - `GPU Allocations`
> - `Multi-modal Features`
> - `VLM Capabilities`
> - `CUDA Performance` 


- **Meta Llama 3.1 专注于文本功能**：Meta 最新发布的版本包括 **Llama 3.1 405B**，将上下文长度扩展至 **128K** 并支持**八种语言**，详见 [扎克伯格的信](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/)。然而，本次发布不包含多模态部分，成员们讨论认为这种遗漏可能是财报发布前的战略安排。
- **对 GPU 的高需求**：一位成员表达了难以获取单个 **A100 GPU** 的沮丧，而另一位成员指出 **xAI** 正在使用惊人的 **100,000 H100 GPUs**。可用 GPU 的数量凸显了用户之间资源获取能力的巨大差异。
- **讨论中的 VLM 能力**：成员们注意到此版本仅提供文本模型，VLM (Vision Language Model) 功能预计稍后推出。一位成员分享了他们通过利用 **GPT-4o** 生成大量 Python 实现，在 ARC-AGI 上实现 **50% 准确率**的方法心得。
- **通过特征工程提升结果**：讨论围绕通过特征工程而非过度依赖视觉能力来改善结果展开，并强调了一个通过对问题网格进行工程化处理取得成功的案例。一位用户提到利用额外技术来优化其方法的性能。
- **CUDA 的未来计划**：一位成员预告了即将发布的 CUDA 版本，声称他们计划在各种矩阵尺寸上超越 **cuBLAS**，特别是对 **FP16/FP32** 的支持。关于 **Nvidia 硬件内联函数 (hardware intrinsics)** 对 FP16 支持的讨论展示了对潜在性能提升的期待。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ai.meta.com/blog/meta-llama-3-1">无标题</a>: 未找到描述</li><li><a href="https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt">Getting 50% (SoTA) on ARC-AGI with GPT-4o</a>: 你只需要抽取更多样本
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1265158276895608862)** (4 messages): 

> - `Performance of CUDA Kernels`
> - `Tiled Matrix Multiplication`
> - `Compute Intensity` 


- **ncu 输出解析**：一位成员询问执行 `ncu ./kernel` 提供的是 CUDA Kernel 的速度还是耗时，并指出普通矩阵乘法的耗时为 **10.30us**，而分块 (tiled) 乘法为 **9.18us**。
   - 他们表示困惑，因为性能提升幅度与 PMPP 教科书中的预期不符。
- **分块 (Tiling) 带来的提升有限**：另一位成员分享了他们的经验，即从朴素矩阵乘法过渡到分块矩阵乘法并没有产生显著的速度提升，这与 [这篇文章](https://siboehm.com/articles/22/CUDA-MMM) 中的 Kernel 对比结果相似。
   - 他们指出，显著的加速通常只有通过线程分块 (thread tiling) 才能观察到，并引用了链接资源中的 Kernel 实现 4 和 5。
- **计算强度的重要性**：一位成员强调，增加计算强度 (Compute Intensity) 对于获得更好的性能至关重要，特别是为了摆脱 Roofline 模型左侧的限制。
   - 他们表示，这将是优化 CUDA Kernel 初始阶段最具影响力的策略。


  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/)** (1 messages): 

iron_bound: 赞 https://github.com/AnswerDotAI/fsdp_qlora/tree/llama400b
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1265020763904147518)** (182 messages🔥🔥): 

> - `Performance of LLMs`
> - `KV Caching Implementation`
> - `MuP vs Other Optimizations`
> - `Floating Point Precision Techniques`
> - `Training Stability Methods`

- **分析性能指标**：成员们讨论了实验中 ZeRO-1 和 ZeRO-2 之间性能指标的差异，并指出了 ZeRO-2 中 stochastic rounding 的潜在优势。
   - 在 2 x 4060Ti 系统上的初步测试显示，由于额外的通信，存在轻微的性能开销。
- **KV Caching 成果**：报告了模型推理中 KV caching 逻辑的实现进展，部分操作已正常运行，但仍需提高效率。
   - 正在探索对 `matmul_cublaslt` 和 attention kernels 的调整，以在不改变最终结果的情况下增强计算。
- **关于 MuP 与替代方案的见解**：讨论了 muP 与其他方法论之间感知的性能差异，表明 muP 在某些场景下可能表现不佳。
   - 成员们比较了 baseline 优化，指出 muP 旨在获得更好的稳定性和结果，但并不总是能兑现这一承诺。
- **浮点精度技术**：团队探索了使用不同浮点精度（如 BF16 和 FP8）对模型训练性能和稳定性的影响。
   - 由于存在潜在的 underflows 和 overflows，人们对 FP8 训练中维持稳定性的挑战表示担忧。
- **提高训练稳定性**：成员们对增强训练稳定性的各种技术感兴趣，例如最新文献中讨论的 z-loss 和 soft clamping 方法。
   - 有人指出，构建训练期间 tensor 变化的视觉表示可能有助于理解和防止不稳定。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.05872">Scaling Exponents Across Parameterizations and Optimizers</a>：从小宽度到大宽度的模型稳健且有效的 Scaling 通常需要精确调整许多算法和架构细节，例如参数化和优化器的选择...</li><li><a href="https://arxiv.org/abs/2405.18710v1">To FP8 and Back Again: Quantifying the Effects of Reducing Precision on LLM Training Stability</a>：与大语言模型（LLM）预训练相关的巨大计算成本激发了人们对降低精度浮点表示以加速该过程的极大兴趣。作为结果...</li><li><a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>：我们提出了 Adam-mini，这是一种优化器，其性能与 AdamW 相当或更好，但内存占用减少了 45% 到 50%。Adam-mini 通过削减学习率资源来减少内存...</li><li><a href="https://x.com/rosieyzh/status/1811790177246888075">Rosie Zhao @ ICML (@rosieyzh) 的推文</a>：在我们关于评估 LLM 训练优化器的新工作中，我们进行了一系列实验，以研究 Adam 等优化器中的自适应性在实现良好性能和稳定性方面的作用...</li><li><a href="https://arxiv.org/abs/2407.07972">Deconstructing What Makes a Good Optimizer for Language Models</a>：随着规模的扩大，训练语言模型的成本变得越来越高，这促使了许多提高优化效率的尝试。尽管做出了这些努力，Adam 优化器仍然是使用最广泛的...</li><li><a href="https://github.com/microsoft/mup/issues/76">Not getting perf improvements from muP at ~1.5B scale · Issue #76 · microsoft/mup</a>：嘿伙计们，首先感谢你们的出色工作！我在 llm.c 项目中实现了 muP（见此处），坐标检查（coord checks）看起来是平坦/正确的（我运行到了 15 步，仍然是平坦的！），但我...</li><li><a href="https://github.com/karpathy/llm.c/pull/707">Add KV cache for inference by gordicaleksa · Pull Request #707 · karpathy/llm.c</a>：开发中（WIP）。目前非常简陋，正在实验中。待草案取得进展后将更新描述。:)</li><li><a href="https://github.com/karpathy/llm.c/blob/master/llmc/matmul.cuh#L134">llm.c/llmc/matmul.cuh at master · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账户来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/708">Add high perf mode by gordicaleksa · Pull Request #708 · karpathy/llm.c</a>：添加：进入次优分支时的警告；如果未运行所有最优化分支则立即退出的高性能模式；还添加了一个将使用的 fwd kernel 配置...</li><li><a href="https://www.diffchecker.com/nbHQZode/">3.1 vs 3 - llama license - Diffchecker</a>：3.1 vs 3 - llama 许可证 - META LLAMA 3 社区许可协议 Meta Llama 3 版本发布日期：2024 年 4 月 18 日 “协议”</li><li><a href="https://github.com/karpathy/llm.c/pull/307">Improve tanh derivative in backward gelu by akbariyeh · Pull Request #307 · karpathy/llm.c</a>：将 tanh 的导数计算为 1 - tanh^2 比计算 1/(cosh^2) 更便宜。这可能不会产生可衡量的差异。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1265165032480837712)** (6 messages): 

> - `Stable Diffusion 在 RX7900XTX 上的运行`
> - `Flash Attention 对 AMD ROCm 的支持` 


- **关于 RX7900XTX 上 Stable Diffusion 的讨论**：分享了一篇关于使用 [Composable Kernel 库](https://github.com/ROCm/composable_kernel/discussions/1032) 在 **RX7900XTX** 上加速 **Stable Diffusion** 推理的文章。
   - 该讨论被指出略显过时，但提供了关于 ROCm5.7 能力的见解。
- **Flash Attention 现已支持 AMD ROCm**：[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/pull/1010) 项目引入了对 AMD ROCm 的支持，并说明目前仅适用于 **mi200 和 mi300**。
   - 此更新由 **Composable Kernel** 提供支持，相关细节在最近的一个 Pull Request 中共享。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/ROCm/composable_kernel/discussions/1032">在 ROCm5.7 上使用 RX7900XTX 运行 Stable Diffusion · ROCm/composable_kernel · Discussion #1032</a>：使用 Composable Kernel 库在 AMD RDNA3 GPU 上加速推理。你好，欢迎阅读这篇关于 AMD RDNA3 GPU 高性能推理的博客文章。在本博文中，我们将讨论如何使用...</li><li><a href="https://github.com/Dao-AILab/flash-attention/pull/1010">由 rocking5566 在 FlashAttention 2 中支持 AMD ROCm · Pull Request #1010 · Dao-AILab/flash-attention</a>：此 PR 实现了 C++ Flash API 的 AMD / ROCm 版本，包括 mha_fwd、mha_varlen_fwd、mha_bwd、mha_varlen_bwd。Kernel 实现来自 Composable Kernel，C++ API 与原始版本相同...
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1265024691765252178)** (196 messages🔥🔥): 

> - `GEMINI 竞赛`
> - `Meta AI`
> - `Llama 3.1 模型`
> - `语音频道 AI 机器人`
> - `微调 Llama 模型` 


- **关于 GEMINI 竞赛的讨论**：一名成员对 Google 的 **GEMINI 竞赛** 表示出兴趣，并寻求他人帮助一起参加黑客松。
   - *如果你有兴趣合作，请联系！*
- **对 Llama-3.1 模型的反应**：成员们对 **Llama-3.1** 的评价褒贬不一，有人认为它与前几代模型相比显得**缺乏灵魂**。
   - 其他人指出 **Claude** 和 **Gemini** 似乎保留了一些创作深度。
- **无审查版 Llama-3.1 微调**：一名用户正在微调 **Llama-3.1 405B** 以创建一个无审查版本，预计需要数周时间。
   - 他们计划在训练完成后将其发布在 Hugging Face 上，命名为 **Llama3.1-406B-uncensored**。
- **语音频道中 AI 的挑战**：讨论了创建可以在 **Discord 语音频道** 中交互的 AI 机器人的复杂性。
   - 成员们对目前在构建有效的语音交互机器人时面临的局限性表示担忧。
- **AI 模型的成本与可访问性**：成员们讨论了使用 **GPT-4o** 等高级模型的 API 相关成本，并指出了获取更高层级权限的挑战。
   - 一些人对低层级用户需要大量交互才能解除限制的现状表示沮丧。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1265198813455450185)** (7 messages): 

> - `Alpha 版本发布时间`
> - `用户沟通问题`
> - `应用测试` 


- **澄清 Alpha 版本发布时间**：成员们对 Alpha 版本的发布时间表感到不确定，特别是它是在 7 月的最后一天还是提前几天发布。
   - 成员们对预期是否明确提出了质疑，强调开发者需要更好的沟通。
- **用户焦急等待 Alpha 访问权限**：一名成员表示每 20 分钟检查一次应用，希望能被选中作为 Plus 用户参与 Alpha 测试。
   - 另一名用户确认 Alpha 测试预计在 7 月底开始，暗示需要保持耐心。
- **对信息陈旧的担忧**：在讨论中，一名用户指出关于 Alpha 版本发布的共享链接已经快一个月了，这表明信息已经过时。
   - 这引发了关于与付费客户缺乏持续沟通的更广泛讨论。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1265096789577695303)** (7 条消息): 

> - `Meta-Prompting`
> - `Plagiarism in AI Output`
> - `Prompting Techniques` 


- **Meta-Prompting 革新 Prompt Engineering**：一位成员强调，Prompt Engineering 中的 AI 指引被称为 **meta-prompting**，被描述为学习如何构建有效 Prompt 的最佳方法。
   - 通过 meta-prompting，用户最终可以创建能够生成进一步 Prompt 的 Prompt，从而提升其 Prompt 技巧。
- **对输出内容抄袭的担忧**：一位成员表达了挫败感，称使用博客内容会导致 Prompt 生成的内容出现 **100% 抄袭**。
   - 他们正在寻找减轻这一问题的解决方案或思路。
- **寻求 Prompt 改进方案**：针对抄袭问题，一位成员建议分享 Prompt 和 custom instructions，以便从他人那里获得见解和建议。
   - 他们通过引用另一位成员的话来鼓励透明度，并表示：*“也许有人能帮你看一下并提供建议！”*。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1265096789577695303)** (7 条消息): 

> - `Meta-Prompting`
> - `Plagiarism in Generated Content`
> - `Prompt Improvement Suggestions` 


- **通过 Meta-Prompting 学习 Prompt 编写**：**Meta-prompting** 被公认为掌握 Prompt Engineering 的顶级方法，允许用户创建能够生成进一步 Prompt 的 Prompt。
   - 这种技术可以显著增强基于 AI 指引构建有效 Prompt 的能力。
- **对博客内容抄袭的担忧**：一位用户提出担忧，称利用博客内容会导致每个生成的 Prompt 都出现 **100% 抄袭**。
   - 这引发了关于寻找提高生成内容原创性解决方案的讨论。
- **更好 Prompt 编写的建议**：一位成员建议分享之前 Prompt 的具体细节和上下文，以便获得更具针对性的建议。
   - 他们强调了明确表达对响应质量差异化需求的重要性，以便获得有效的建议。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1265076277887373443)** (39 条消息🔥): 

> - `Mojo 社区会议演讲`
> - `标准库中的字符串优化`
> - `在虚拟机上安装 Mojo`
> - `使用 Mojo 进行游戏引擎开发`
> - `链接 C 语言库` 


- **Mojo 社区会议演讲公开征集**：[Mojo 社区会议](https://modul.ar/community-meeting-doc)现提供演讲机会，8 月 12 日有可用时段。
   - 如果你想展示你在 Mojo 中构建的内容或分享经验，可以通过链接文档进行报名。
- **短字符串和缓冲区优化提案**：一位成员确认，他们在标准库中关于 **short string optimization**（短字符串优化）和 **small buffer optimization**（小缓冲区优化）的工作非常适合作为演讲主题。
   - 另一位成员对此表示支持，并指出 **optimizations**（优化）是过去会议中的热门相关话题。
- **在 Ubuntu 虚拟机上安装 Mojo**：一位用户询问在 Windows 的 Ubuntu 虚拟机中安装 Mojo 的可行性，对此其他成员回应称，配合 WSL 和 Docker 等解决方案通常运行良好。
   - 虽然有人担心潜在的安装问题，但使用虚拟机被认为是合适的。
- **评估 Mojo 用于游戏引擎开发**：讨论强调 Mojo 可能适合构建下一代游戏引擎，特别是由于其通过 GPU 支持具备良好的异构计算能力。
   - 然而，也有人指出游戏开发模式中分配器（allocator）处理方面的挑战，暗示可能会遇到一些**棘手的问题**（rough spots）。
- **在 Mojo 中链接 C 语言库**：关于将 Mojo 链接到 C 语言库的讨论正在进行中，有建议认为改进的功能将使利用 libpcap 的项目受益。
   - 成员指出，在 Linux 上的 Mojo 应该默认使用 **ktls**，以增强底层网络的可定制性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo">GitHub - modularml/mojo: The Mojo Programming Language</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/3262">Ability to Link to C Libraries · Issue #3262 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并相信此请求符合优先级。你的请求是什么？理想情况下应该有类似 @link(…) 装饰器的东西...</li><li><a href="https://modul.ar/community-meeting-doc.">[Public] Mojo Community Meeting</a>：Mojo 社区会议。此文档链接：https://modul.ar/community-meeting-doc。这是一个公开文档；欢迎所有人查看和评论/建议。所有会议参与者必须遵守...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1815463417391837596>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1265037280339300404)** (50 messages🔥): 

> - `SDL Bindings`
> - `Mojo Game Frameworking`
> - `Physics Engine Development`
> - `Contributing to Mojo`
> - `Pygame with Mojo` 


- **SDL Bindings 正在进行中**：一名成员正在为 Mojo 开发 **SDL bindings**，并指出 **Pygame** 本质上是 SDL 的封装，这使得集成成为可能。
   - 另一位用户提到他们自己的 SDL binding 项目已经停滞，但计划更新并改进其 API。
- **游戏框架实验**：关于 **游戏框架和物理引擎** 实验的讨论引发了兴趣，一位成员分享了构建自定义物理引擎的个人经验。
   - 该用户希望将来能将其数学部分转化为一个名为 **Infrared** 的通用几何代数包。
- **创建小型 Socket 库**：一位新成员正在使用 **external_call** 集成 C 函数，为 Mojo 开发一个 **mini socket library**，并寻求将其以 **Apache 2.0** 协议授权的许可。
   - 他们表达了为 Mojo 做出贡献的兴趣，并得到了社区积极响应的鼓励。
- **贡献与社区资源**：成员们讨论了为 Mojo 贡献代码的可用资源，包括指向标记为 **good first issue** 的 GitHub issue 链接。
   - 一位用户计划阅读贡献指南，以更好地了解如何参与社区项目。
- **期待 v24.5 的发布**：一位成员询问了 **v24.5** 的发布日期，提到 **v24.4** 是在 6 月初发布的。
   - 有人建议，目前关于 GPU 特性的讨论可能会推迟新版本的发布，并引发了对版本命名规范的推测。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/saviorand/lightbug_http/tree/main/external">lightbug_http/external at main · saviorand/lightbug_http</a>：适用于 Mojo 的简单且快速的 HTTP 框架！🔥。可以通过在 GitHub 上创建账号来为 saviorand/lightbug_http 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc+label%3A%22good+first+issue%22">Issues · modularml/mojo</a>：Mojo 编程语言。可以通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1265100055052816465)** (9 messages🔥): 

> - `Modular's Industry Relationships`
> - `NVIDIA Support`
> - `OpenCL and SYCL Usage` 


- **Modular 对行业关系保持保密**：一名成员指出 *Modular 的行业关系* 是保密的，在正式发布公告前不会发表评论，但会在合适的时机公开分享。
   - 这在他们准备好以官方身份披露信息之前，保持了一定程度的机密性。
- **NVIDIA 支持助力 Modular 的方案**：来自 **NVIDIA** 的支持被视为重大增强，一名成员表示渴望在支持上线后立即使用。
   - 有建议称，时机成熟时可以在专门的频道中进一步讨论。
- **OpenCL 的历程与相关性**：讨论强调了 *OpenCL* 的起源及其在实现高级编程中的重要性，特别是在 SYCL 和 OneAPI 等平台内。
   - 尽管有人对 OpenCL 的未来使用表示担忧（特别是考虑到旧硬件的淘汰），但其在 *特定数据库和防火墙* 中的相关性得到了认可。
- **使用 GPU 和 FPGA 进行通用计算**：成员们讨论了不仅将 **GPU** 和 **FPGA** 用于图形处理，还用于通用计算（General-purpose compute），特别是在数据库场景下。
   - 人们认识到这些技术在传统角色之外有效处理工作负载的能力。


  

---


### **Modular (Mojo 🔥) ▷ #[max-gpu](https://discord.com/channels/1087530497313357884/1212827673257316453/1265101767893454939)** (2 messages): 

> - `XLA`
> - `MAX engine`
> - `GPU performance` 


- **MAX engine：XLA 之后的下一步**：**MAX engine** 被视为 **XLA** 的继任者，它利用了从 XLA 中获得的经验，同时解决了其缺点，例如具备可扩展性并原生支持 **动态和参数化形状 (dynamic and parametric shapes)**。
   - 与 **XLA** 相比，人们预期其 **CPU 和 GPU 性能** 将有显著提升。
- **推进 MAX/GPU 发布之路**：虽然在今年晚些时候发布之前无法透露 **MAX/GPU** 的具体细节，但团队致力于实现艰难但正确的解决方案。
   - 对 **GPU 在 AI 世界中重要性** 的信念驱动着这一努力，这也为产品发布的进展带来了期待。


  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1265031218026778895)** (86 条消息🔥🔥): 

> - `memcpy 的更改`
> - `Mojo 的文档`
> - `Mojo 中 Reference 的使用`
> - `Mojo Nightly 的更新`
> - `MAX 与 Mojo 的关系` 


- **memcpy 函数的更改**：用户讨论了最近对 `memcpy` 函数进行的更改，注意到针对指针类型的三个重载，并对新的函数签名感到困惑。
   - 成员们探讨了这些更改可能对现有代码产生的影响，特别是关于 `DTypePointer` 和 `LegacyPointer` 等类型，并提供了潜在的解决方案。
- **Mojo 需要更好的文档**：用户对 Mojo 文档的现状表示不满，认为过于技术化的解释对初学者来说缺乏清晰度。
   - 用户还担心 Discord 的格式问题也增加了理解难度，呼吁改进文档格式。
- **关于 Reference 和相等性的讨论**：一位成员质疑为什么 `Reference` 缺少 `__eq__` 方法，推测其是否被设计为唯一或排他的。
   - 另一位用户支持这一观点，并指出直接比较内存地址比解引用更高效。
- **Mojo Nightly 更新公告**：分享了关于最新 Mojo nightly 编译器更新的通知，重点介绍了更新和错误修复。
   - 鼓励用户更新版本，并提供了指向变更日志（changelog）的链接以查看详细更改。
- **MAX 与 Mojo 之间的关系**：成员们讨论了 MAX 是如何使用 Mojo 构建的，强调这两个系统随着共同的编译器更改而同步演进。
   - 提到了 MAX Kernel 开发中 Mojo 和 C++ 的融合，阐明了两者的联系。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sourcegraph.com/search">Sourcegraph</a>：未找到描述</li><li><a href="https://youtu.be/_QVs626Vn2k?t=3617">Mojo 🔥 社区会议 #4</a>：Mojo 社区会议 #4 的录音🫓 Flat Buffers：内存高效的序列化⚒️ Forge Tools：扩展 Mojo 🔥 标准库🔄 Mojo 🔥 Gen...</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/memory/__init__.mojo">mojo/stdlib/src/memory/__init__.mojo at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/16cc60dc3fbed1eff01a8f5fee94f97cf97cca33/stdlib/src/memory/__init__.mojo">mojo/stdlib/src/memory/__init__.mojo at 16cc60dc3fbed1eff01a8f5fee94f97cf97cca33 · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1265051433703702528)** (1 条消息): 

> - `Intel CPUID 库`
> - `AMD CPUID 映射` 


- **Intel 的 CPUID 库简化了访问**：Intel 的库封装了 **CPUID** 并将其转换为更易理解的格式，无需用户查阅处理器文档。
   - 这为使用 Intel 处理器的开发者提供了一种更用户友好的方法。
- **AMD 和 Intel 使用独立的 CPUID 映射**：会议指出，除了区分处理器制造商外，**AMD** 和 **Intel** 还维护着独立的 **CPUID** 映射。
   - 因此，开发者需要针对每个制造商使用不同的映射来访问特定的处理器功能。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1265026280664072223)** (84 条消息🔥🔥): 

> - `FSDP 性能问题`
> - `Llama 3.1 托管`
> - `生成式 ML 贡献` 


- **nn.Parameters 导致的 FSDP 性能困扰**：一位用户在向其使用 **FSDP** 的模型添加 `nn.Parameters` 时经历了 **20 倍的减速**，但发现使用大小为 **16** 的参数能显著提高性能。
   - 他们讨论了与**缓冲区对齐（buffer alignment）**相关的潜在问题，以及尽管 GPU kernel 运行很快，但对齐不良如何影响 **CPU** 性能。
- **托管 Llama 3.1 405B**：一位成员宣布他们已在 **8xH100 80GB** 硬件上托管了 **Llama 3.1 405B instruct**，可通过[聊天界面](https://chat.tune.app/)和 [API](https://studio.tune.app/) 访问。
   - 遗憾的是，访问受登录限制，且托管安排涉及成本，引发了关于硬件限制和托管替代方案的讨论。
- **对开放 AI 研究的贡献**：一位用户介绍自己正在一家初创公司从事生成式 ML 工作，表示有兴趣为**开放 AI 研究**做出贡献，并讨论了一篇关于从较少样本中学习推理的论文。
   - 他们过去的经验包括 **3D Computer Vision** 和**机器翻译（Machine Translation）**领域的工作，强调了其在有限数据下推进 AI 的目标。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://chat.tune.app/">Tune Chat - 由开源 LLM 驱动的聊天应用</a>：通过 Tune Chat，可以访问提示词库、与 PDF 聊天以及品牌声调功能，以增强您的内容写作与分析，并在所有创作中保持一致的语调。</li><li><a href="https://studio.tune.app/">TuneStudio</a>：未发现描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1265056944616640623)** (43 条消息🔥): 

> - `新的 SAE 架构`
> - `Monte Carlo Dropout 对比`
> - `分层 3D Gaussians`
> - `Llama 3 模型细节`
> - `Transformer 性能与稀疏性` 


- **引入了用于高效训练的新 SAE 架构**：一种名为 **Switch SAE** 的新颖架构利用条件计算（conditional computation）来高效扩展稀疏自编码器（SAEs），解决了在跨层训练宽 SAE 时的计算挑战。
   - *相关论文链接*强调了这种方法在从超智能语言模型中恢复特征方面的潜力。
- **关于不确定性的 Monte Carlo Dropout 对比**：一位用户指出，他们的方法结果应与 **Monte Carlo dropout** 进行对比，后者被认为是贝叶斯不确定性量化（Bayesian uncertainty quantification）的一种次优近似。
   - 另一位成员分享的见解表明，存在许多对比这些方法的论文，并强调了关于 MC dropout 有效性的担忧。
- **Llama 3 的图像编码限制**：人们对 **Llama 3** 模型的图像编码器提出了担忧，特别是其 **224x224** 的分辨率限制。
   - 一些人建议，使用 Armen 团队倡导的 **vqvae-gan 风格分词器（tokenizer）**可能会增强图像处理能力。
- **Transformer 模型及其性能影响**：讨论集中在 **Transformers** 的扩展以及随着模型尺寸增加，由多头注意力（MHA）产生的 FLOPs 比例如何下降，可能降至 **33% 或更低**。
   - 分享了关于 **V** 和 **O 投影（projection）**必要性的见解，引发了对其在模型解释和有效性方面影响的思考。
- **Transformer 模型中的稀疏性**：引用了一篇论文，讨论了利用 **Transformer 层**中的稀疏性如何在降低训练成本和提高效率的同时，产生具有竞争力的性能。
   - 研究结果表明，稀疏变体可以达到与传统 Transformer 相似的困惑度（perplexity）水平，使其适用于更长的序列处理。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.14561">NNsight and NDIF: Democratizing Access to Foundation Model Internals</a>: 最先进的基础模型其巨大的规模限制了科学家对它们的访问，因为在大模型尺寸上进行定制化实验需要昂贵的硬件和复杂的工程...</li><li><a href="https://arxiv.org/abs/2111.12763">Sparse is Enough in Scaling Transformers</a>: 大型 Transformer 模型在许多任务上取得了令人印象深刻的结果，但训练甚至微调的成本都很高，且解码速度慢到让人无法使用和研究。我们解决了这个问题...</li><li><a href="https://ai.meta.com/research/publications/the-llama-3-herd-of-models/">无标题</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2403.13097">Simple Ingredients for Offline Reinforcement Learning</a>: 离线强化学习算法已在与目标下游任务高度相关的数据集上证明了其有效性。然而，利用一个新的测试平台 (MOOD)，其中的轨迹来自异构...</li><li><a href="https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/">A Hierarchical 3D Gaussian Representation for Real-Time Rendering of
    Very Large Datasets</a>: 未找到描述</li><li><a href="https://www.lesswrong.com/posts/47CYFbrSyiJE2X5ot/efficient-dictionary-learning-with-switch-sparse">Efficient Dictionary Learning with Switch Sparse Autoencoders — LessWrong</a>: 作为 ML Alignment &amp; Theory Scholars 项目（2024 年夏季批次）的一部分产出 …
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 条消息): 

alofty: https://arxiv.org/abs/2407.14561
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1265022144937459804)** (23 条消息🔥): 

> - `任务分组建议`
> - `lm-eval Harness 更新`
> - `vLLM 与 Logits 问题`
> - `自动化单元测试讨论`
> - `Transformers 版本问题` 


- **任务分组中的 Groups vs Tags**：建议对嵌套任务和聚合评分使用 **groups**，而 **tags** 则适用于更简单的场景。
   - *Hailey Schoelkopf* 确认该方法对任务组织非常有效。
- **lm-eval Harness 增强功能**：lm-eval harness 的更新包括一个用于 API 模型的新超类，增强了模块化和功能性，详见 [Pull Request #2008](https://github.com/EleutherAI/lm-evaluation-harness/pull/2008)。
   - 成员现在可以使用 `local-completions` 模型类型配合 vLLM 的 OpenAI 服务器，在所有任务类型上评估 Llama-405B。
- **关于 vLLM 和 Logits 的澄清**：针对 **vLLM** 是否提供 logits 展开了讨论，对其能力的看法存在分歧；然而，最终澄清它确实提供 continuation logits。
   - 对话引用了来自 [vLLM 仓库](https://github.com/vllm-project/vllm/issues/185) 和 [Triton 推理服务器仓库](https://github.com/triton-inference-server/server/issues/6895) 的 issue。
- **对自动化单元测试的关注**：一名成员对目前缺乏自动化单元测试表示担忧，强调了其在防止代码库出现破坏性变更方面的重要性。
   - Hailey Schoelkopf 承认需要改进测试，并提到了现有的回归测试，尽管它们的样本量有限。
- **Transformers 版本问题**：Layernorm 发现，在 **Transformers** 最近的一次 commit 之后，他们的 deepseek 模型被错误地识别为 Llama 模型。
   - 固定 Transformers 版本解决了该问题，表明这与该库的最新更新有关。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#:~:text=yourip%7D%3A8000/v1%2C-,num_concurrent%3D1,-%2Cmax_retries%3D3%2Ctokenized_requests">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/vllm-project/vllm/issues/185#issuecomment-1600931023">Can I directly obtain the logits here? · Issue #185 · vllm-project/vllm</a>: 嗨，非常棒的工作！我想知道是否有简单的方法获取 logits，因为有时我只需要计算特定序列的 perplexity/language modeling loss。我看到了代码...</li><li><a href="https://github.com/triton-inference-server/server/issues/6895">vllm backend - logit probabilities at inference · Issue #6895 · triton-inference-server/server</a>: 关于当前的 vllm 后端：https://github.com/triton-inference-server/vllm_backend/tree/main 我想知道在推理时是否也有可能获取 logit 概率...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2008">Refactor API models by baberabb · Pull Request #2008 · EleutherAI/lm-evaluation-harness</a>: 此 PR 为 API 请求模型引入了一个新的超类，提供：下游类的模块化、用于请求转换的可重载方法、API 请求和响应解析、Tokeniza...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1265084194590036120)** (5 条消息): 

> - `Nerdsniping 评估`
> - `不可作弊的评估 Harness` 


- **利用评估进行 Nerdsniping**：一名成员表达了一个轻松的意图，称：*“总有一天我会用评估来对你进行 nerdsnipe。”*
   - 该评论暗示了围绕评估方法复杂性的趣味性挑战。
- **不可作弊评估（Uncheatable Eval）的挑战**：在一次询问中，一名成员询问是否可以将 **uncheatable eval** 整合进评估 harness 中，并对其可行性提出了疑问。
   - 另一名成员幽默地评论道：*“一旦你把它加入 harness，它就不再是‘不可作弊’的了。”*
- **Fresh Scrape 防御**：一名成员断言，只要 **uncheatable eval** 指向 **fresh scrape**（新鲜抓取的数据），它就依然有效。
   - 这一说法遭到了质疑，认为将其与 harness 结合使用可能会限制其效能。
- **对效能与可复现性的担忧**：一名成员警告说，不可作弊评估方法的想法过于强大，并引用可复现性问题质疑其可行性。
   - 他们指出，由于该概念对标准实践的影响，可能会面临审查或拒绝。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1265020956783673344)** (69 条消息🔥🔥): 

> - `Meta 的 AI 策略`
> - `NVIDIA 的市场地位`
> - `OpenAI 价格战`
> - `Llama 3.1 发布` 


- **Meta 的 AI 策略与高级服务**：讨论了 Meta 可能推出的 **Llama 405B** 的 **Premium** 版本，并猜测可能在 **7 月 23 日**发布公告。
   - 成员们注意到，最近取消了对 Llama 模型的限制，这为除了改进其他模型之外的更广泛用途打开了大门。
- **NVIDIA 的潜在垄断**：人们对 NVIDIA 整合硬件、CUDA 和模型的野心表示担忧，这可能会形成类似于历史上涉及 IBM 的反垄断案件的潜在垄断。
   - 一位用户建议，如果 NVIDIA 控制了整个技术栈，他们基本上可以“印钞”，但监管障碍会阻止这种整合。
- **OpenAI 的竞争性定价策略**：OpenAI 宣布 **gpt-4o-mini** 每天前 200 万个 token 免费进行微调，这引发了关于 AI 领域激进定价格局的讨论。
   - 成员们反思了行业内作为对竞争加剧的回应而出现的混乱价格战状态。
- **Llama 3.1 与性能指标**：重点介绍了 **Llama 3.1** 的发布，成员们讨论了将其纳入 **RewardBench** 的情况，显示其在某些任务上与 **GPT-4** 保持一致。
   - 据报道，这些模型主要受到安全性问题的挑战，用户指出这可能是有益的。
- **行业洞察与参考**：一位用户赞赏 Ben Thompson 的 *Stratechery* 提供的见解，表明其与正在进行的市场动态讨论具有相关性。
   - 其他成员分享了他们对技术策略周期性的看法，指出公司经常重复历史模式。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.joelonsoftware.com/2002/06/12/strategy-letter-v/">策略信 V (Strategy Letter V)</a>：我在大学时修了两门经济学入门课程：宏观经济学和微观经济学。宏观经济学充满了诸如“低失业率导致通货膨胀”之类的理论，但这些理论从未真正站得住脚……</li><li><a href="https://x.com/natolambert/status/1815813221410037957">来自 Nathan Lambert (@natolambert) 的推文</a>：通过 @togethercompute 将 Llama 3.1 模型添加到 RewardBench。主要受限于安全性，有人认为这是好事。在其他挑战任务上与 GPT-4 持平。</li><li><a href="https://x.com/testingcatalog/status/1815439546722451493?s=46">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：除此之外，Llama 405B 似乎可能成为 Premium 服务的一部分，在这种情况下，Meta AI Premium 也可能在 7 月 23 日宣布（在代码中发现）。此外，还提到了 AI St...</li><li><a href="https://fxtwitter.com/moyix/status/1815840634013639086?s=46">来自 Brendan Dolan-Gavitt (@moyix) 的推文</a>：抱歉，OpenAI 现在在做什么？！gpt-4o-mini 的微调每天前 200 万 token 免费？！</li><li><a href="https://x.com/kalomaze/status/1815547484376076460?s=46">来自 kalomaze (@kalomaze) 的推文</a>：LLM-Distillery！一个由 AMOGUS 和我过去几个月构建的开源训练流水线，用于通过……收集和训练“学生”语言模型以模仿“教师”模型。</li><li><a href="https://www.diffchecker.com/nbHQZode/">3.1 vs 3 - llama 许可证 - Diffchecker</a>：3.1 vs 3 - llama 许可证 - META LLAMA 3 社区许可协议 Meta Llama 3 版本发布日期：2024 年 4 月 18 日 “协议”</li><li><a href="https://web.archive.org/web/20240722214257/https://huggingface.co/huggingface-test1/test-model-1">huggingface-test1/test-model-1 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1265197069404999732)** (16 条消息🔥): 

> - `关于合成数据生成的 Magpie 论文`
> - `LLaMA 3 Instruct 性能`
> - `Instruction finetuning 技术`
> - `Vocabulary size 与推理速度` 


- **Magpie 论文揭示了合成数据生成技术**：[Magpie 论文](https://arxiv.org/abs/2406.08464) 提出了一种仅使用模板为 LLM 生成**高质量指令数据**的方法，允许像 **LLaMA 3** 这样的模型在极少输入的情况下生成用户查询。
   - 据报道，该技术可以大规模生成数据，并且与 Alpaca 和 UltraChat 等现有数据集相比，展现出**更大的词汇多样性**。
- **LLaMA 3 在 Magpie 数据集上的惊人表现**：即使只有 **300k 样本**，在 **Magpie IFT 数据集**上微调的 **LLaMA 3 Base** 在 **AlpacaEval** 上的表现也比**原始 LLaMA 3 Instruct** 模型高出 9.5%。
   - 这引发了人们对传统指令蒸馏（instruction distillation）技术与新型数据集生成方法相比有效性的质疑。
- **来自 Raschka 博客文章的指令微调见解**：Sebastian Raschka 在其博客文章中介绍了**指令微调**的进展，强调了生成微调数据集的新型高性价比方法。
   - 他强调了大型科技公司在 LLM 集成方面的潜在应用和最新进展，以及高质量指令数据的重要性。
- **关于词表大小对推理速度影响的辩论**：针对 Raschka 声称**更大的词表大小**可能会减慢推理速度的观点引发了讨论，这与通常认为更少但更密集的 Token 会加速过程的观点形成对比。
   - 成员们注意到词表大小增加对较小模型与较大模型相比的相对影响，建议寻找最佳平衡点至关重要。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://magazine.sebastianraschka.com/p/instruction-pretraining-llms">Instruction Pretraining LLMs</a>: 指令微调的最新研究</li><li><a href="https://arxiv.org/abs/2406.08464">Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing</a>: 高质量的指令数据对于对齐大语言模型 (LLM) 至关重要。虽然 Llama-3-Instruct 等一些模型具有开放权重，但它们的对齐数据仍然是私有的，这阻碍了...</li><li><a href="https://magazine.sebastianraschka.com/i/146761957/running-the-dataset-generation-locally">Instruction Pretraining LLMs</a>: 指令微调的最新研究
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1265270069462827049)** (43 messages🔥): 

> - `Llama 3 发布`
> - `Mark Zuckerberg 的 AI 时代`
> - `模型水印担忧`
> - `公众对 Zuckerberg 的看法` 


- **Llama 3 基础模型发布**：[Llama 3](https://llama.meta.com/) 的发布包含了一系列支持多语言和工具使用的语言模型，其中最大的模型拥有 **405B 参数**和 **128K tokens** 的上下文窗口。关于该模型的一篇论文详细介绍了评估结果，显示 Llama 3 的性能与 **GPT-4** 相当。
   - 存在关于模型水印和追踪下载的讨论，因为用户在获取权重（weights）之前需要**提供信息**并同意许可协议。
- **观察 Zuck 的 AI 形象转变**：一位成员分享了他们观看关于 Mark Zuckerberg 的 [YouTube 视频](https://youtu.be/YuIc4mq7zMU?si=UgEu2onfXdlblT9j)后的想法，提到这感觉像是一篇围绕他新树立的“酷”形象的软文。他们指出，这在很大程度上强化了 Zuckerberg 需要适应公众认知的叙事。
   - 评论中包含了对 Zuckerberg 关于 **Windows 主导地位**归功于其开放性的历史叙述的反思，一些用户认为这是在改写历史。
- **关于 AI 模型下载追踪的辩论**：人们对 Meta 可能如何**追踪模型下载**表示担忧，用户需要提供信息才能接收链接。这引发了猜测，即追踪可能是为了确保符合协议。
   - 对话暗示了这可能出于**分析（analytics）**目的，但也引发了关于数据收集的隐私问题。
- **关于 Llama 3 和开放策略的个人研究**：一位成员表达了对研究 Llama 3 及其在工具使用和策略方面的更广泛影响的兴奋，并指出可能需要数周时间才能消化所有内容。他们计划从大局观的文章开始，然后再深入研究技术帖子。
   - 人们期待这些知识将如何影响对 **AI 模型**及其社会影响的理解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/togethercompute/status/1815796476775461019?s=46">Together AI (@togethercompute) 的推文</a>：@altryne @satpal_patawat @GroqInc @Teknium1 @JonathanRoss321 @xenovacom @altryne prompt 是否长于 4K？目前限制了上下文，但我们很快就会开放。如果你看到一个...</li><li><a href="https://youtu.be/YuIc4mq7zMU?si=UgEu2onfXdlblT9j">Inside Mark Zuckerberg's AI Era | The Circuit</a>：如果 AI 战争的最新战役是在开放模型和封闭模型之间展开，Meta 首席执行官兼创始人 Mark Zuckerberg 正处于前线。自从更名为 M...</li><li><a href="https://llama.meta.com/">Llama 3.1</a>：你可以随处微调、蒸馏和部署的开源 AI 模型。我们最新的模型提供 8B、70B 和 405B 版本。</li><li><a href="https://www.producthunt.com/posts/llama-7"> Llama - 3.1-405B：一个足以与 GPT-4o / Claude-3.5 抗衡的开源模型 | Product Hunt</a>：Meta 正在发布三个模型：新的 3.1-405B 以及对其较小模型的升级：3.1-70B 和 3.1-8B。如果 405B 正如基准测试所示的那样出色，这将是开源模型第一次...</li><li><a href="https://ai.meta.com/research/publications/the-llama-3-herd-of-models/">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1265053168799318139)** (3 messages): 

> - `Claude AI 边界`
> - `AI 中的神圣文本`
> - `GPT-3.5 Opus 的发布` 


- **Claude AI 限制关于神圣文本的言论**：一位用户指出，在尝试使用 **Claude** 进行演示时，遇到了关于“神圣文本（Sacred Texts）”的**强力护栏（guardrails）**，具体引用了他选择的《我有一个梦想》（I Have a Dream）。
   - *Claude 有很强的倾向避免使用敏感文本*，这在这次互动中显而易见。
- **用户与马丁·路德·金博士之间的比较**：在一个轻松的评论中，一位用户通过宣布自己的论文为“神圣文本”，将自己比作**马丁·路德·金博士**。
   - 这一幽默的比较得到了祝贺的回应，突显了在讨论个人作品时的崇敬主题。


  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1265193010833002557)** (7 条消息): 

> - `OpenAI vs Llama 3.1`
> - `ChatGPT Memory Management`
> - `Mark Zuckerberg's AI Era`
> - `Snail Appreciation` 


- **OpenAI 的意外退款**：一位用户报告称 **OpenAI 承认败给了 Llama 3.1**，并随机向他们退还了一笔意想不到的金额。
   - 他们用简单的确认表达了感谢：*
- **像玩游戏一样管理 ChatGPT Memory**：一位成员将 **ChatGPT** 的 Memory 管理比作**游戏中的库存管理**，并指出当 Memory 满时他们通常会选择退出。
   - 这个类比突出了用户在平台内高效使用 Memory 所面临的挑战。
- **走进 Mark Zuckerberg 的 AI 时代**：一位用户分享了一段 [名为 'Inside Mark Zuckerberg's AI Era' 的 YouTube 视频](https://www.youtube.com/watch?v=YuIc4mq7zMU)，讨论了 AI 领域正在进行的博弈。
   - 该视频强调了 **Meta CEO Mark Zuckerberg** 在 AI 开源与闭源模型竞争的最前沿地位。
- **分享对蜗牛的热情**：一位用户幽默地与社区互动，发送了一张正在移动的**蜗牛**图片，引发了积极的反响。
   - 这种新分享的对蜗牛的热爱进一步活跃了讨论氛围。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/xlr8harder/status/1815633845422584171">来自 xlr8harder (@xlr8harder) 的推文</a>：对此感到非常震撼</li><li><a href="https://www.youtube.com/watch?v=YuIc4mq7zMU">Inside Mark Zuckerberg&#39;s AI Era | The Circuit</a>：如果 AI 战争的最新战线是在开源和闭源模型之间，那么 Meta CEO 兼创始人 Mark Zuckerberg 正处于最前线。自从更名为 M...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1265052470636187790)** (3 条消息): 

> - `Distillation`
> - `Llama 3.1` 


- **寻找关于 Distillation 的博客文章**：一位成员询问是否有人有关于 **distillation** 的推荐博客文章，表示对该主题感兴趣。
   - 这引发了关于该主题缺乏全面资源的讨论。
- **缺少 Lilian Wang 的全面文章**：另一位成员对 Lilian Wang 没有写过关于 distillation 的 **2 万字**长文表示惊讶。
   - 这一评论反映了社区对详细讨论和资源的渴望。
- **关于 Llama 3.1 Distillation 的潜在文章**：一位成员提到，如果 **Llama 3.1** 被 distilled，他们可能会写一两段话。
   - 这表明了对新进展及此类过程记录的持续关注。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1265421370112606269)** (4 条消息): 

> - `SnailBot updates`
> - `User engagement timings` 


- **SnailBot 新闻公告**：发布了 **SnailBot News** 的通知，针对用户角色 <@&1216534966205284433> 进行更新。
   - *请继续关注来自 SnailBot 的精彩公告和更新！*
- **记录参与时长**：提到了 **45 分钟**，可能强调了用户参与时长或相关的时间框架。
   - *这一见解可能会为未来考虑用户交互规模的讨论或活动提供参考。*
- **用户对内容的反馈**：一位成员表示正在进行的讨论中有些**有趣**的东西。
   - *用户的积极参与表明频道内存在动态的对话流。*


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1265052493142819078)** (73 messages🔥🔥): 

> - `Llama 3.1 Release`
> - `Mistral and Nemo Concerns`
> - `Training Issues`
> - `Language Inclusion in Models`
> - `Evaluation Scores Comparison` 


- **Llama 3.1 发布引发褒贬不一的反应**：围绕 **Llama 3.1** 发布的期待显而易见，但一些人对其效用和性能表示担忧，尤其是针对 **Mistral** 等模型。
   - *Duh Kola* 感叹道，*“该死，他们不喜欢 Llama 的发布”*，表明了对整体反响的不满。
- **Llama 3.1 的训练挑战**：用户在训练 **Llama 3.1** 时遇到了错误，特别是关于 `rope_scaling` 配置的问题，这在社区中引起了挫败感。
   - 一位成员通过更新 Transformers 成功运行，在克服重大障碍后表示，*“似乎成功了，谢谢！”*。
- **关于语言包含情况的讨论**：有人对 **Llama 3.1** 排除**中文**支持表示担忧，成员们认为考虑到中文在全球的重要性，这是一个不利的疏忽。
   - 评论指出，虽然模型 Tokenizer 包含了中文，但未将该语言列入优先级被视为战略上的失误。
- **评估分数对比：Llama 3.1 vs Qwen**：随后讨论了 **Llama 3.1** 的 **CMMLU** 和 **C-Eval** 分数，评估显示其较前代仅有轻微提升。
   - 成员们注意到 **Qwen** 自报的分数显示出更好的性能，但由于评估方法的差异，可能无法直接比较。
- **Qwen 模型的许可担忧**：有人提出了关于 **Qwen** 许可状态的问题，特别是它是否仍受阿里巴巴的限制，还是已经完全开放。
   - *Noobmaster29* 提到，*“只要是公开权重，我并不介意许可证”*，反映了社区在模型获取方面的务实态度。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://llama.meta.com/llama-downloads/">Download Llama</a>：申请 Llama 访问权限。</li><li><a href="https://llama.meta.com/">Llama 3.1</a>：您可以随时随地进行微调、蒸馏和部署的开源 AI 模型。我们最新的模型提供 8B、70B 和 405B 版本。
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1265135798655909938)** (33 条消息🔥): 

> - `LLM Distillation`
> - `DPO Training Issues`
> - `Adapter Fine Tuning`
> - `Reward Modeling`
> - `ChiPO Algorithm` 


- **探索 LLM Distillation 流水线**：一名成员分享了 [LLM Distillery GitHub repo](https://github.com/golololologol/LLM-Distillery) 的链接，该仓库概述了一个 LLM distillation 的流水线。
   - 讨论重点介绍了他们在磁盘上预计算 logits 并随后进行 KL divergence 的实现方式。
- **对 DPO 训练停滞的担忧**：成员们对 DPO 集成缺乏进展表示担忧，一名成员指出该项目已有两周没有动静。
   - 虽然存在一些困惑，但一名成员确认他们正在重新审查该 issue 以寻求解决方案。
- **Adapter 微调阶段**：一名成员询问了关于 [GitHub issue #1095](https://github.com/axolotl-ai-cloud/axolotl/issues/1095) 中提到的多阶段 Adapter 微调的看法。
   - 他们提议使用先前的权重来初始化后续阶段，以提高 DPO 训练的效率。
- **DPO 和 NLL Loss 的数学复杂性**：讨论了围绕 DPO 和引入 NLL loss 的复杂性，并对其经验有效性（empirical validity）持怀疑态度。
   - 成员们表示有兴趣将近期论文中的数学理论整合到实际应用中。
- **Reward Modeling 与 PPO 方法的对比**：大家达成共识，尽管 Reward Modeling 存在局限性，但目前仍优于 Proximal Policy Optimization (PPO)。
   - 成员们探讨了实施 stepwise-DPO 的策略，并考虑可能通过 LoRA adapters 进行增强。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.13399">Correcting the Mythos of KL-Regularization: Direct Alignment without Overoptimization via Chi-Squared Preference Optimization</a>：语言模型对齐方法（如来自人类反馈的强化学习 RLHF）在语言模型能力方面取得了显著进展，但现有技术受到多种限制...</li><li><a href="https://github.com/golololologol/LLM-Distillery">GitHub - golololologol/LLM-Distillery: A pipeline for LLM distillation</a>：一个用于 LLM distillation 的流水线。通过在 GitHub 上创建账号来为 golololologol/LLM-Distillery 的开发做出贡献。</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1095)">Issues · axolotl-ai-cloud/axolotl</a>：尽管向 axolotl 提问。通过在 GitHub 上创建账号来为 axolotl-ai-cloud/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1265114145204994110)** (3 条消息): 

> - `LLM for verb tense conversion`
> - `Spacy script for perspective change`
> - `Third-person to first-person conversion`
> - `Dataset for tense conversion examples` 


- **寻求用于时态和视角调整的 LLM**：一名成员询问是否有人知道可以有效更改任意文本动词时态和视角的 **LLM** 或 **使用 Spacy 的脚本**。
   - 他们特别需要将文本从**第三人称/过去时转换为第一人称/现在时**。
- **未完成的时态转换数据集**：另一名成员分享了他们大约一年前构建的包含 **10k 样本的时态转换数据集** 的工作，但由于其他事务而未能完成。
   - 他们表示，如果其他人发现了任何相关的工具或资源，希望能得到通知。


  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1265227814009835530)** (8 条消息🔥): 

> - `Code Confluence 工具`
> - `DSPY 集成`
> - `Zenbase/Core 库发布` 


- **Code Confluence 工具生成 GitHub 摘要**：受 **DSPY** 启发，一位成员介绍了 **Code Confluence**，这是一个使用 **Antlr**、**Chapi** 和 DSPY 流水线构建的 OSS 工具，旨在创建 GitHub 仓库的详细摘要。该工具的结果可能优于现有的开源产品，正如他们在 [DSPY 仓库](https://github.com/unoplat/unoplat-code-confluence/blob/main/unoplat-code-confluence/examples/python/dspy/dspy_v1.md)中所展示的那样。
   - 他们提供了额外的资源，包括 [Unoplat Code Confluence GitHub](https://github.com/unoplat/unoplat-code-confluence/) 和名为 [OSS Atlas](https://github.com/unoplat/unoplat-oss-atlas/tree/main) 的摘要汇编。
- **对 Code Confluence 的参与和反馈**：新工具欢迎反馈，用户的参与表明了对其功能的兴趣和兴奋。一位用户评论说他们会 **去看看 (check it out)**，并用 **🔥** 表情符号表达了热情。
   - 另一位用户注意到最近分享了大量有趣的进展，为 **DSPY** 社区营造了热烈的氛围。
- **Zenbase/Core 库在 Twitter 上发布**：一位成员宣布在 Twitter 上发布 **zenbase/core**，这是一个 Python 库，允许用户在现有的 **Instructor** 和 **LangSmith** 代码中使用 DSPY 的优化器。他们请求对他们的发布公告进行转发、点赞和 Star，公告可以在[这里](https://twitter.com/cyrusofeden/status/1815858216389300383?s=61&t=WwA-PFs585hhcOplJkLRbQ)查看。
   - 该库的推出表明了将 DSPY 功能集成到更广泛的编程实践中以增强用户体验的持续努力。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1265108584442892370)** (2 条消息): 

> - `AI 研究论文`
> - `实现请求` 


- **新 AI 研究论文提醒**：一位成员分享了一篇名为 [2407.12865](https://arxiv.org/pdf/2407.12865) 的 AI 研究论文链接，引发了对其研究结果的关注。
   - 鼓励其他人查看该论文并在社区中讨论其影响。
- **征集代码复现**：一位成员请求，如果有人编写代码来复现该论文的研究结果，或者发现了现有的实现，请分享出来或私信 (DM) 他。
   - 这突显了推进该研究讨论的协作方式。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1265022761856667809)** (83 条消息🔥🔥): 

> - `DSPy 与 Outlines 的比较`
> - `使用 DSPy 进行实体提取`
> - `Llama3 的结构化输出问题`
> - `DSPy 中的 Optimizer 更新`
> - `LOTUS 与 Phoenix 的集成` 


- **JSON 生成库的比较**：成员们讨论了 **Jsonformer**、**Outlines** 和 **Guidance** 等库在生成结构化 JSON 方面的优缺点，并指出 **Outlines** 对 Pydantic 格式和 JSON schemas 提供了更好的支持。
   - *Jsonformer* 因严格遵守 Schema 而受到赞赏，而 *Guidance* 和 *Outlines* 提供了更多灵活性，但可能会增加复杂性。
- **DSPy 中的实体提取模块**：一位用户询问在执行 DSPy 中的 **EntityExtractor** 模块时如何观察内部步骤，这引出了使用 `inspect_history` 方法的建议。
   - 该方法旨在帮助用户在处理输入时了解模块的内部工作原理。
- **Llama3 结构化输出的挑战**：用户表示在使用 DSPy 从 **Llama3** 模型获取正确的结构化输出时遇到困难，建议将 `dspy.configure(experimental=True)` 与 *TypedChainOfThought* 结合使用。
   - 然而，关于即使在类型检查失败时如何查看模型输出仍存在疑问，并指出 `inspect_history` 的实用性存在局限。
- **对 DSPy Optimizer 更新的关注**：一位用户提出了关于将 DSPy 的后端重构合并到主分支计划的问题，特别是对实验新 Optimizer 感兴趣。
   - 这表明 DSPy 正在进行持续开发，且用户积极参与其功能增强。
- **LOTUS 与 Phoenix 的集成**：一位用户询问如何将 **LOTUS** 与 **Phoenix** 连接以检查查询，显示出对探索 DSPy 生态系统内集成机会的兴趣。
   - 另一位成员确认了 LOTUS 与 Modin 的活跃使用，表明这些集成已有实际应用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://localhost:30000"))">未找到标题</a>：未找到描述</li><li><a href="https://github.com/sksarvesh007/dspy-rag-application/blob/main/embedder.ipynb">dspy-rag-application/embedder.ipynb (位于 sksarvesh007/dspy-rag-application 的 main 分支)</a>：通过在 GitHub 上创建账号来为 sksarvesh007/dspy-rag-application 的开发做出贡献。</li><li><a href="https://github.com/outlines-dev/outlines">GitHub - outlines-dev/outlines: 结构化文本生成</a>：结构化文本生成。通过在 GitHub 上创建账号来为 outlines-dev/outlines 的开发做出贡献。</li><li><a href="https://github.com/lm-sys/RouteLLM?tab=readme-ov-file">GitHub - lm-sys/RouteLLM: 一个用于服务和评估 LLM 路由器的框架 - 在不牺牲质量的情况下节省 LLM 成本！</a>：一个用于服务和评估 LLM 路由器的框架 - 在不牺牲质量的情况下节省 LLM 成本！ - lm-sys/RouteLLM</li><li><a href="https://github.com/stanfordnlp/dspy">GitHub - stanfordnlp/dspy: DSPy: 用于对基础模型进行编程（而非提示）的框架</a>：DSPy: 用于对基础模型进行编程（而非提示）的框架 - stanfordnlp/dspy</li><li><a href="https://github.com/stanfordnlp/dspy/tree/rc">GitHub - stanfordnlp/dspy 的 rc 分支</a>：DSPy: 用于对基础模型进行编程（而非提示）的框架 - GitHub - stanfordnlp/dspy at rc</li><li><a href="https://github.com/stanfordnlp/dspy/issues/590">如何更改生成长度 · Issue #590 · stanfordnlp/dspy</a>：嘿，我正在按如下方式初始化我的 LM，extras={'max_tokens':4000,'temperature':0.7} vllm = dspy.HFClientVLLM(model="Mistral-7B-Instruct-v0.1",url="https://my_vllm_u...</li><li><a href="https://github.com/stanfordnlp/dspy/blob/31ac32ba1a0b51cb7b9a8728b0bb7d4f3f2860a5/dsp/modules/hf.py#L30">dspy/dsp/modules/hf.py (位于 stanfordnlp/dspy)</a>：DSPy: 用于对基础模型进行编程（而非提示）的框架 - stanfordnlp/dspy
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1265184598908731443)** (3 messages): 

> - `ColPali 使用案例`
> - `ColBert 与 RAG`
> - `Qdrant 对 ColBert 的支持` 


- **探索用于医疗文档的 ColPali**：一位成员分享了他们使用 ColPali 的经验，表示他们正在测试将其用于**带有图像的医疗文档 RAG**，因为 **ColBert 和标准 embedding 模型**此前在该领域表现不佳。
   - 他们还计划探索训练和使用其他 vision-language 模型以提高效果。
- **Qdrant 对 ColBert 的采用**：另一位成员强调 **Qdrant** 现在支持 ColBert，并提供了自 v1.10.0 起可用的 [Hybrid and Multi-Stage Queries](https://qdrant.tech/documentation/concepts/hybrid-queries/) 文档。
   - 多查询功能的引入允许利用每个点的命名向量（named vectors）来实现复杂的搜索场景，从而增强检索过程。



**提到的链接**：<a href="https://qdrant.tech/documentation/concepts/hybrid-queries/">Hybrid Queries - Qdrant</a>：Qdrant 是一个用 Rust 编写的开源向量数据库和向量搜索引擎。它通过便捷的 API 提供快速且可扩展的向量相似度搜索服务。

  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1265342139064127499)** (1 messages): 

> - `LlamaIndex 网络研讨会`
> - `ColPali 文档检索`
> - `Vision Language Models`
> - `ViDoRe 基准测试` 


- **关于高效文档检索的 LlamaIndex 网络研讨会**：欢迎参加本周五 **PT 时间上午 9 点**由 ColPali 作者主持的即将举行的网络研讨会，讨论**使用 Vision Language Models 进行高效文档检索**。[在此注册](https://lu.ma/9q4ldrwc)以了解文档处理领域的尖端技术。
- **ColPali 创新的文档检索方法**：ColPali 引入了一种新技术，直接使用 Vision Language Models (VLMs) 嵌入**页面截图**，从而提高了在**复杂文档**上的检索性能。这种方法避免了传统解析和 OCR 通常会遇到的关键视觉信息丢失问题。
- **新的文档检索基准测试**：ColPali 提出的新 **ViDoRe 基准测试**能更好地解决与各种文档元素相关的具有挑战性的检索任务，增强了对检索系统的评估。该基准测试旨在通过专注于视觉表示来补充传统方法。
- **多模态文档检索的未来**：网络研讨会将深入探讨**多模态文档检索**的未来，整合来自 ColPali 和 LlamaParse 的技术。讨论将重点介绍一个在文档检索中达到 State-of-the-art 结果的端到端系统。



**提到的链接**：<a href="https://lu.ma/9q4ldrwc">LlamaIndex Webinar: ColPali - Efficient Document Retrieval with Vision Language Models · Zoom · Luma</a>：企业级 RAG 系统在处理具有复杂布局、表格和图表的 PDF 时面临重大挑战。传统的 RAG 流水线通常……

  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1265021000773533766)** (8 messages🔥): 

> - `TiDB Future App Hackathon 2024`
> - `Mixture-of-Agents with LlamaIndex`
> - `Llama 3.1 Performance`
> - `LlamaParse Features`
> - `MongoDB AI Applications Program` 


- **加入奖金 30,000 美元的 TiDB Future App Hackathon！**：我们正在赞助为期一个月的 [TiDB Future App Hackathon 2024](https://t.co/vTV3t8daqT)，奖金总额超过 **30,000 美元**，其中第一名奖金为 **12,000 美元**，合作伙伴包括 @pingcap 等。
   - 参与其中，利用最新的 [TiDB Serverless with Vector Search](https://www.pingcap.com/ai/) 构建创新的 AI 应用。
- **探索 LlamaIndex 的 Mixture-of-Agents！**：在一段新视频中，@1littlecoder 介绍了一种名为 'Mixture-of-Agents' 的新方法，该方法通过使用多个本地语言模型，其表现有可能超越 **GPT-4** 等单一模型。
   - 查看 [分步教程](https://t.co/EqF2RM3jeB)，了解此方法如何增强您的 AI 项目。
- **Llama 3.1 模型现已发布**：包含 **8B**、**70B** 和 **405B** 模型的 **Llama 3.1** 系列现在可以通过 Ollama 在 LlamaIndex 中使用，不过 405B 模型需要强大的计算资源。
   - 对于托管版本，请咨询我们的合作伙伴 [Fireworks AI](https://t.co/NMckK14nZf) 以获取支持。
- **探索 LlamaParse 的功能**：在一段视频中，@seldo 强调了 **LlamaParse** 的关键特性，包括 **Markdown 和 JSON 输出** 选项，以及增强的 **OCR 支持**。
   - 该工具旨在实现跨多种语言的更强大的元数据提取，使其在文档处理方面具有极高的通用性。
- **MongoDB AI Applications Program 正式启动！**：@MongoDB 宣布其 **AI Applications Program (MAAP)** 已全面开放，旨在帮助企业高效地构建和部署 AI 驱动的应用。
   - 在[此处](https://t.co/rCz3DfUe3A)了解更多关于 MAAP 的服务以及它如何加速您的 AI 之旅。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://t.co/vTV3t8daqT">TiDB Future App Hackathon 2024</a>: 创新并创建令人惊叹的 AI 应用</li><li><a href="https://t.co/rCz3DfUe3A">MongoDB AI Applications Program</a>: 获取加速 AI 应用之旅所需的支持，并以信心和速度发布。
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1265139668752662644)** (61 messages🔥🔥): 

> - `context_window 参数`
> - `chunk_size 和 chunk_overlap`
> - `模型可用性与上下文大小`
> - `LlamaIndex 中的 ValueError`
> - `使用具有更大上下文窗口的模型` 


- **理解 context_window 参数**：`context_window` 参数指定了模型可以处理的最大 token 数量，包括输入和输出 token。
   - 如果输入文本过长，可能会限制输出生成，如果超过 token 限制会导致错误。
- **定义 chunk_size 和 chunk_overlap**：`chunk_size` 设置处理过程中每个分块的最大 token 数，而 `chunk_overlap` 定义了相邻分块之间重叠的 token 数。
   - 这些参数有助于控制 embedding 的精度，并确保跨分块保留上下文。
- **解决 LlamaIndex 中的 ValueError**：指示上下文大小为负值的 ValueError 表明输入文本超过了当前模型的 `context_window` 限制。
   - 减少输入大小或切换到具有更大上下文窗口的模型是潜在的解决方案。
- **通过 context_window 最大化模型效率**：在达到上下文窗口的情况下，可能会显著限制模型的输出能力。
   - 根据输入长度选择具有适当 `context_window` 值的模型对于获得最佳性能至关重要。
- **关于 context_window 范围的讨论**：分享了关于 `context_window` 是仅涵盖输入 token 还是也包括输出的澄清。
   - 已确认 `context_window` 包含两者，因此需要仔细管理输入大小。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/api_reference/llms/cleanlab/#llama_index.llms.cleanlab.CleanlabTLM>).">Cleanlab - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/models/llms/#usage-pattern>).">Using LLMs - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/node_parsers/token_text_splitter/#llama_index.core.node_parser.TokenTextSplitter>)">Token text splitter - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies/#chunk-sizes>).">Basic Strategies - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1265123362246627450)** (53 messages🔥): 

> - `Llama 3.1 发布`
> - `IOL 语言学奥林匹克`
> - `Llama 定价`
> - `Llama 性能评估`
> - `GPT-4o Mini 微调` 


- **Llama 3.1 发布引发关注**：[Llama 3.1](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) 的发布包括了 405B 模型，这标志着开源 LLM 的一个重要里程碑，其卓越的能力可与闭源模型相媲美。
   - 初步评估表明，它是第一个定位在尖端（frontier）能力的开源模型，得到了 @karpathy 等人物的认可，称赞其在迭代研发中的可访问性。
- **国际语言学奥林匹克 (IOL)**：国际语言学奥林匹克 (IOL) 开幕，挑战学生纯粹利用逻辑翻译冷门语言，类似于 IMO 等高风险数学竞赛。
   - 参赛者正在苛刻的六小时时限内解决看似不可能的问题，引发了人们对逻辑推理如何弥合语言鸿沟的兴趣。
- **Llama 3.1 定价见解**：Llama 3.1 405B 模型的定价因供应商而异，Fireworks 和 Together 等平台的输入和输出成本约为每百万 token 4-5 美元。
   - 这种具有竞争力的定价策略被视为可能旨在在随着采用率增长而逐渐提高费率之前夺取市场份额。
- **Llama 性能评估**：Llama 3.1 的早期评估显示其在各种基准测试中表现良好，在 GSM8K 和 ZebraLogic 的逻辑推理能力等任务中排名很高。
   - 在对比测试中，其整体性能介于 Sonnet 3.5 和 GPT-4o 之间，尽管也注意到了一些挑战，例如在超长 token 长度后难以维持 schema 遵循。
- **GPT-4o Mini 微调发布**：OpenAI 宣布了 GPT-4o mini 的微调功能，现已面向 tier 4 和 5 用户开放，在 9 月 23 日之前每天前 200 万个训练 token 免费。
   - 该计划旨在随着时间的推移扩大访问和定制选项，用户已经在针对新发布的 Llama 3.1 评估其性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/corbtt/status/1815829444009025669?s=46">来自 Kyle Corbitt (@corbtt) 的推文</a>：伙计们，微调后的 Llama 3.1 8B 简直强得离谱。刚刚用我们的微调测试套件运行了一下，在每项任务上都完胜 GPT-4o mini。从来没有过这么小的开源模型...</li><li><a href="https://x.com/deanwball/status/1815826885663658445?s=46">来自 Dean W. Ball (@deanwball) 的推文</a>：根据欧盟及其 AI Act，Llama 3 405b 对社会构成了“系统性风险”。</li><li><a href="https://x.com/sullyomarr/status/1815788922737225771?s=46">来自 Sully (@SullyOmarr) 的推文</a>：小扎（zucc）推出的 Llama 3.1 真的太棒了，它是最好的开源模型，几乎可以媲美最顶尖的闭源模型。</li><li><a href="https://x.com/naklecha/status/1815808346735378487?s=46">来自 naklecha (@naklecha) 的推文</a>：今天，我很高兴发布 factorio-automation-v1。使用这个模组，你的 Agent 可以执行游戏操作，如合成、寻路、采矿、研究等。这个模组可以作为一个很好的实验场...</li><li><a href="https://x.com/hrishioa/status/1815811349777375649?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Hrishi (@hrishioa) 的推文</a>：Llama3 405B 现在已加入 Mandark (https://github.com/hrishioa/mandark)。代码编写测试：表现如何？* 需要更多的 Prompt tuning，在约 1K tokens 后难以维持 schema * 只是...</li><li><a href="https://x.com/neuralmagic/status/1815769704415342890?s=46">来自 Neural Magic (@neuralmagic) 的推文</a>：vLLM 现在支持在单个 8xH100 或 8xA100 节点上部署 Llama-3.1-405B，使推理变得更加简单和便宜！这是 Neural Magic 工程师们的一项巨大成就，他们贡献了 3 个关键特性...</li><li><a href="https://x.com/JonathanRoss321/status/1815777714642858313">来自 Jonathan Ross (@JonathanRoss321) 的推文</a>：拥有 Llama 的质量和 Groq 的速度能做什么？你可以实现“即时”（Instant）。就是这样。在 http://groq.com 上体验 Llama 3.1 8B 的即时智能。</li><li><a href="https://llama.meta.com/">Llama 3.1</a>：你可以随处进行微调、蒸馏和部署的开源 AI 模型。我们最新的模型提供 8B、70B 和 405B 版本。</li><li><a href="https://x.com/togethercompute/status/1815769272536445292">来自 Together AI (@togethercompute) 的推文</a>：随着 Meta Llama 3.1 405B 的发布，今天标志着开源 AI 的转折点。它是目前最大的开源基础模型，在快速加速的 AI 领域中足以媲美最顶尖的闭源模型...</li><li><a href="https://x.com/openaidevs/status/1815836887631946015?s=46">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：通过微调为你的应用定制 GPT-4o mini。今天起对第 4 级和第 5 级用户开放，我们计划逐步向所有级别开放。在 9 月之前，每天前 2M 训练 tokens 免费...</li><li><a href="https://x.com/corbtt/status/1815843764960911549">来自 Kyle Corbitt (@corbtt) 的推文</a>：@altryne @eugeneyan 评测（EVALS）正在运行中</li><li><a href="https://arxiv.org/abs/2406.03368">IrokoBench：大语言模型时代非洲语言的新基准</a>：尽管大语言模型（LLMs）已被广泛采用，但其卓越的能力仍局限于少数高资源语言。此外，许多低资源语言（例如非洲语言）...</li><li><a href="https://x.com/billyuchenlin/status/1815841947468353700?s=46">来自 Bill Yuchen Lin 🤖 (@billyuchenlin) 的推文</a>：对 Llama-3.1-405B-Instruct-Turbo（在 @togethercompute 上）的一次快速独立评估 ⬇️ 1️⃣ 它在 GSM8K 上排名第一！2️⃣ 它在 ZebraLogic 上的逻辑推理能力与 Sonnet 3.5 非常相似，而且...</li><li><a href="https://x.com/aiatmeta/status/1815766327463907421?s=46&t=b7l37rB6wtbyAh6ah1NpZQ">来自 AI at Meta (@AIatMeta) 的推文</a>：从今天开始，开源正在引领潮流。介绍 Llama 3.1：我们迄今为止最强大的模型。今天我们将发布一系列新的 Llama 3.1 模型，包括期待已久的 405B。这些模型...</li><li><a href="https://x.com/deedydas/status/1815222838623883614">来自 Deedy (@deedydas) 的推文</a>：IMO 是最难的高中数学考试。一个较少为人知的兄弟赛事 IOL（国际语言学奥林匹克竞赛）明天开始！学生们被要求纯粹通过翻译鲜为人知的语言...</li><li><a href="https://x.com/aravsrinivas/status/1815800336642367590?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：Scale AI 的 SEAL 评测（我认为这比竞技场排行榜更好，因为你不想在虚假端点上盲目追求高分，也不想让随机的人根据感觉来评分）表明...</li><li><a href="https://x.com/thexeophon/status/1815780557445648648?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Xeophon (@TheXeophon) 的推文</a>：Llama-405B 定价：Fireworks: 3/3 Together: 5/15 Replicate: 9.5/9.5 Groq: 目前仅限企业版。引用 Xeophon (@TheXeophon)：考虑到 Llama 3 的定价和其他供应商的时间表，这将是...</li>

li><li><a href="https://x.com/francis_yao_/status/1815434157893267554?s=46">来自 Yao Fu @ICML (@Francis_YAO_) 的推文</a>：作为一名 LLM 从业者，这场演讲是我目前听过的关于 LLM 科学领域信息量最大的一场。我深受启发。对于我自己的学生，我会要求他们背诵……</li><li><a href="https://x.com/AIatMeta/status/1815814535313514960">来自 AI at Meta (@AIatMeta) 的推文</a>：关于我们今天发布的全新 Llama 3.1 模型的更多技术细节。🦙🧵</li><li><a href="https://x.com/karpathy/status/1815842603377779140?s=46">来自 Andrej Karpathy (@karpathy) 的推文</a>：热烈祝贺 @AIatMeta 发布 Llama 3.1！几点笔记：今天随着 405B 模型的发布，是前沿能力的 LLM 首次开放给所有人使用和构建……</li><li><a href="https://x.com/realgeorgehotz/status/1815818855190782198?s=46">来自 George Hotz 🌑 (@realGeorgeHotz) 的推文</a>：不仅发布了 405B Llama 的权重，他们还发布了一篇解释其制作过程的论文。太棒了！任何有自尊心的 ML 研究员怎么还能在封闭实验室工作？你并不是在拯救……</li><li><a href="https://x.com/thexeophon/status/1815780557445648648?s=46&t=6FDPaNx">来自 Xeophon (@TheXeophon) 的推文</a>：Llama-405B 定价：Fireworks: 3/3 Together: 5/15 Replicate: 9.5/9.5 Groq: 目前仅限企业版。引用 Xeophon (@TheXeophon)：考虑到 Llama 3 的定价和其他供应商的时间安排，这将是……</li><li><a href="https://x.com/summeryue0/status/1815776426999877643">来自 Summer Yue (@summeryue0) 的推文</a>：🚀 我们将 Llama 3.1 405B 添加到了 SEAL 排行榜中，它没有让人失望！以下是它的表现：- 🥇 指令遵循 (Instruction Following) 排名第 1 - 🥈 GSM1k 排名第 2 - 💻 编程 (Coding) 排名第 4。SEAL 评估是私有的……
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1265340057406148638)** (3 条消息): 

> - `Llama 3 播客`
> - `合成数据 (Synthetic Data)`
> - `RLHF`
> - `Galactica Instruct`
> - `Llama 4 Agents` 


- **Llama 3 播客发布**：由 [@ThomasScialom](https://x.com/latentspacepod/status/1815781241398104085) 主讲的新播客集讨论了 **Llama 2, 3 & 4**，重点关注 **Synthetic Data**、**RLHF** 以及 Agent 通往 **Open Source AGI** 的道路。
   - 鼓励听众点击[此处](https://latent.space/p/llama-3)查看并参与播客互动！
- **Galactica Instruct 的潜在影响**：播客强调了为什么 **@ylecun 的 Galactica Instruct** 本可以有效解决 **@giffmana 的 Citations Generator** 问题。
   - 这一见解展示了先进模型在现实场景中的实际应用。
- **Chinchilla 性能见解**：讨论包括 **@jefrankle** 提到的 **100x Chinchilla** 等进展，强调了超越传统模型的趋势。
   - 这提出了关于优化模型效率和性能的有趣观点。
- **原生 INT8 训练探索**：本集涵盖了 **@NoamShazeer** 对 **原生 INT8 训练 (native INT8 training)** 的看法，强调了其对模型训练和部署的影响。
   - 这可能会塑造未来 AI 模型训练策略的方法论。
- **Llama 4 和 Agent 的未来**：讨论涉及 **Llama 4** 关于 **Agents** 的计划，并质疑了避免使用 **MoE** 的原因。
   - 这些考量可能指向影响未来 AI 模型能力的重要设计选择。



**提到的链接**：<a href="https://x.com/latentspacepod/status/1815781241398104085">来自 Latent.Space (@latentspacepod) 的推文</a>：🆕 与 @AIatMeta 的 @ThomasScialom 合作的播客！Llama 2, 3 & 4：Synthetic Data、RLHF、通往 Open Source AGI 之路上的 Agents https://latent.space/p/llama-3 特别鸣谢：- 为什么 @ylecun 的 Galactica Instruct……

  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1265170932016877709)** (23 条消息🔥): 

> - `AgentState vs InnerAgentState`
> - `使用 Chroma 向量数据库`
> - `LangChain 中的多角色聊天机器人` 


- **AgentState 与 InnerAgentState 探讨**：有人提出了关于 `AgentState` 和 `InnerAgentState` 之间区别的问题。虽然 `AgentState` 的定义已经明确，但指出关于 `InnerAgentState` 的信息不足，建议用户查阅官方 LangChain 文档。
   - 关于 `AgentState` 的详细信息包括 `messages`、`next` 等字段（取决于上下文），并提供了进一步探索的参考资料。
- **在 Python 上设置 Chroma 向量数据库**：提供了关于如何使用 Python 设置 Chroma 作为向量数据库的指令，包括安装 `langchain-chroma` 以及在 Docker 容器中运行 Chroma 服务器。
   - 示例包括使用 `.add`、`.get` 和 `.similarity_search` 等方法，并强调了使用 `OpenAIEmbeddings` 需要 OpenAI API Key。
- **使用 LangChain 开发即兴聊天机器人**：有人咨询关于使用 LangChain 创建多角色即兴聊天机器人的问题。虽然尚未确认显式支持，但提到 LangChain 提供的流式传输（streaming）和消息历史管理等功能可以实现此类功能。
   - 分享了来自 LangChain 文档的有用资源，包括关于对话式 RAG、Agents 和消息历史管理的教程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/#basic-example-using-the-docker-container>)">Chroma | 🦜️🔗 LangChain</a>：Chroma 是一款专注于开发者生产力和幸福感的 AI 原生开源向量数据库。Chroma 采用 Apache 2.0 许可。</li><li><a href="https://github.com/langchain-ai/langchain/issues/19211>)),">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/22191>)).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1265042962673434725)** (3 条消息): 

> - `使用 Composio 构建调度 Agent`
> - `LangGraph 与 MapReduce`
> - `Llama 3.1 托管` 


- **使用 Composio 创建调度 Agent**：分享了一份指南，详细介绍了利用 **Composio**、**LangChain** 和 **ChatGPT** 创建 **调度 Agent** 的步骤，该 Agent 可根据收到的电子邮件安排活动。如果您觉得有用，可以查看指南并 [Star 该仓库](https://git.new/scheduler)。
   - 该指南强调了 Composio 如何为 Agent 配备精心设计的工具，以有效地处理复杂任务。
- **用于并行处理的 LangGraph 和 MapReduce**：一篇文章讨论了 **LangGraph** 和 **MapReduce** 如何作为“黄金搭档”协作处理大数据中的并行处理任务。深入见解可以在这篇详细的 [文章](https://ai.gopubby.com/langgraph-and-mapreduce-a-dynamic-duo-for-parallel-processing-744cb10da377) 中找到。
   - 引言强调了将任务分解以进行并行执行是如何改变复杂计算游戏规则的。
- **Llama 3.1 托管现已可用**：一位成员宣布托管了 **Llama 3.1 405B** 并邀请其他人尝试。聊天界面可在 [此处](https://chat.tune.app/) 访问，API 可在 [此处](https://studio.tune.app/) 访问。
   - 此次托管为成员们提供了一个在用户友好型环境中与最新模型版本交互的机会。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://git.new/scheduler">composio/python/examples/scheduler_agent at master · ComposioHQ/composio</a>：Composio 为 Agent 配备了精心设计的工具，使其能够处理复杂任务 - composio/python/examples/scheduler_agent at master · ComposioHQ/composio</li><li><a href="https://chat.tune.app/">Tune Chat - 由开源 LLM 驱动的聊天应用</a>：通过 Tune Chat，访问提示词库、PDF 聊天和品牌声音（Brand Voice）功能，以增强您的内容写作和分析，并在所有创作中保持一致的基调。</li><li><a href="https://studio.tune.app/">TuneStudio</a>：未找到描述</li><li><a href="https://ai.gopubby.com/langgraph-and-mapreduce-a-dynamic-duo-for-parallel-processing-744cb10da377">LangGraph and MapReduce: A Dynamic Duo for Parallel Processing</a>：Ankush k Singal
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1265043135684280321)** (5 条消息): 

> - `Scheduler Agent`
> - `YouTube Notes Generator`
> - `LangGraph and Flow Engineer`
> - `AI Code Reviewer`
> - `Fully Local Tool Calling with Ollama` 


- **使用 Composio 创建你自己的 Scheduler Agent**：分享了一份指南，详细介绍了如何使用 Composio、LangChain 和 ChatGPT 创建一个 **Scheduler Agent**，用于根据收到的电子邮件安排事件。点击[此处](https://git.new/scheduler)查看。
   - Composio 为 Agent 配备了工具，使其能够有效地处理复杂任务，这在 [scheduler examples](https://git.new/scheduler) 中得到了展示。
- **YouTube Notes Generator 发布！**：宣布了一个新的开源项目 **YouTube Notes Generator**，旨在帮助用户从 YouTube 视频中生成笔记。更多信息可以在[此处](https://www.linkedin.com/posts/isham-rashik-5a547711b_machinelearning-artificialintelligence-deeplearning-activity-7221165319464095747-DMDS?utm_source=share&utm_medium=member_desktop)找到。
   - 该项目旨在简化直接从视频内容中提取笔记的过程，提高学习效率。
- **使用 LangGraph 构建 10 倍可靠的 Agent**：发布了一个视频教程，演示了如何使用 **LangGraph** 和 **Flow Engineer** 构建高度可靠的 Agent。在 YouTube 上观看：[此处](https://youtu.be/01g_EfO-Dms?si=tMF70x7MhxKw8S95)。
   - 该视频简化了流程，显著提升了 Agent 的可靠性，推广了高效的开发实践。
- **使用 Ollama 和 LangChain 的 AI Code Reviewer**：一段名为 **'AI Code Reviewer Ft. Ollama & Langchain'** 的新 YouTube 视频介绍了一个用于高效代码审查的 CLI 工具。点击[此处](https://youtu.be/g_VRsjpC4e8)查看视频。
   - 该工具旨在彻底改变代码审查流程，增强开发者的工作流和生产力。
- **请求关于 Fully Local Tool Calling 的 Notebook**：一位成员请求获取 **'Fully local tool calling with Ollama'** 的 Notebook，希望能获取当天早些时候分享的信息。该环节被社区公认为非常出色。
   - 这反映了社区对本地工具集成技术实际应用的浓厚兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/g_VRsjpC4e8">AI Code Reviewer Ft. Ollama &amp; Langchain</a>：欢迎来到 Typescriptic！在本视频中，我们将介绍我们的 Code Reviewer，这是一个旨在彻底改变代码审查方式的 CLI 工具。由 LangChain 驱动...</li><li><a href="https://git.new/scheduler">composio/python/examples/scheduler_agent at master · ComposioHQ/composio</a>：Composio 为 Agent 配备了精心打造的工具，使其能够应对复杂任务 - composio/python/examples/scheduler_agent at master · ComposioHQ/composio
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1265034850059030640)** (26 条消息🔥): 

> - `Welcome New Members`
> - `Model Fine-tuning`
> - `Cohere's OCR Capabilities`
> - `RAG Chatbot Discussions`
> - `Community Feedback Evaluation` 


- **欢迎新成员**：包括 @thetimelesstraveller 和 @fullc0de 在内的新成员介绍了自己，并表达了对使用 Cohere 的兴奋。
   - @xvarunx 等社区成员对他们表示了热烈欢迎，营造了友好的氛围。
- **模型 Fine-tuning 进度**：@thetimelesstraveller 分享了使用名为 **midicaps** 的数据集进行模型 Fine-tuning 的新尝试，其中涉及一些后处理。
   - 他们引用了以往类似项目的**良好结果**，表明他们的努力取得了进展。
- **澄清 Cohere 的 OCR 解决方案**：针对关于 OCR 能力的问题，@co.elaine 告知 Cohere 使用了 [unstructured.io](https://unstructured.io)。
   - 社区讨论显示，集成外部解决方案是可行的，并允许进行自定义。
- **ChatBot 和 RAG 实现疑问**：用户 @coco.py 提出了关于在基于 **RAG** 的 ChatBot 系统中管理聊天历史和反馈的问题。
   - 回复建议将之前的对话放入 Context 中或使用 **Vector Databases**，同时提到了点赞/点踩等反馈方法。
- **积极的社区氛围**：社区庆祝了最近的一次发布，用户在各种评论中表达了兴奋和积极的态度。
   - @mrdragonfox 重申了社区准则，确保环境保持专注且友好。


  

---

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1265348202534670427)** (1 messages): 

> - `Rerank 3 Nimble`
> - `Cohere and Fujitsu Partnership` 


- **推出具有卓越性能的 Rerank 3 Nimble**：**Rerank 3 Nimble** 正式发布，其吞吐量比前代 Rerank 3 高出 **3 倍**，同时保持了极高的准确度水平。它现在已在 [AWS SageMaker](https://cohere.com/blog/rerank-3-nimble) 上可用。
   - *向我们的新基础模型 Rerank 3 Nimble 问好！* 该模型承诺为企业级搜索和检索增强生成 (RAG) 系统提供更快的速度。
- **Cohere 与 Fujitsu 的战略合作伙伴关系**：Cohere 宣布与 **Fujitsu** 建立**战略合作伙伴关系**，专门为日本企业提供 AI 服务。详情请参阅 [博客文章](https://blog.fujitsu-partnership)。
   - 此次合作旨在利用两家公司的优势，提升该地区的 AI 服务交付能力。



**提及链接**: <a href="https://cohere.com/blog/rerank-3-nimble">Introducing Rerank 3 Nimble: Faster Reranking for Enterprise Search &amp; Retrieval-Augmented Generation (RAG) Systems</a>: 今天，Cohere 推出了 Rerank 3 Nimble：这是我们 Cohere Rerank 模型系列中最新的基础模型，旨在增强企业级搜索和 RAG 系统，其速度比 Rerank 3 快约 3 倍，同时...

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1265066201013424149)** (22 messages🔥): 

> - `Llama 3.1 release`
> - `MPS support and conflicts`
> - `Issues with LoRA`
> - `Git workflow challenges` 


- **Llama 3.1 正式发布！**：Meta 今天早上发布了最新的模型 **Llama 3.1**，并已提供对 8B 和 70B instruct 模型的支持 {@everyone}。
   - 现场气氛非常热烈，甚至出现了一些关于拼写错误和因兴奋导致失误的幽默评论。
- **MPS 支持及相关冲突**：提到了一个关于 MPS 支持的 **Pull Request**，其中将检查 **MPS 设备上的 BF16** 作为一个关键更新。
   - 此外，代码库中持续存在的冲突也受到了关注，贡献者指出由于频繁的更改，保持分支更新具有挑战性。
- **LoRA 问题依然存在**：提出了一个关于 **LoRA** 未能按预期工作的问题，并给出了调试实现的建议。
   - 一位贡献者回忆起在最近的工作中遇到了 **CUDA 硬编码** 问题。
- **应对 Git 工作流挑战**：讨论了 **Git 工作流** 的挑战，特别是那种在解决完之前的冲突后又不断面临新冲突的感觉。
   - 有人建议调整工作流以尽量减少重复发生的冲突，强调了有效冲突解决策略的必要性。


<div class="linksMentioned">

<strong>提及链接</strong>:

<ul>
<li>
<a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1">Llama 3.1 | Model Cards and Prompt formats</a>: Llama 3.1 - 最强大的开源模型。</li><li><a href="https://pytorch.org/torchtune/0.2/install.html#install-nightly-build)">Install Instructions &mdash; torchtune 0.2 documentation</a>: 未找到描述</li><li><a href="https://github.com/pytorch/torchtune/pull/790">MPS support by maximegmd · Pull Request #790 · pytorch/torchtune</a>: 背景：出于测试目的，直接在本地 Mac 电脑上运行非常有用。变更日志：检查 MPS 设备对 BF16 的支持。添加了针对 MPS 的配置，对路径进行了更改...
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1265292188871299184)** (3 条消息): 

> - `Torchtune 中的 MPS 支持`
> - `Pad ID 错误修复`
> - `GitHub Pull Request 工作流` 


- **MPS 支持 Pull Request 讨论**：标题为 [MPS support by maximegmd](https://github.com/pytorch/torchtune/pull/790) 的 Pull Request 引入了对 MPS 设备上 BF16 的检查，旨在改进在本地 Mac 电脑上的测试。
   - 讨论指出，由于 diff 源自共同祖先，可能会出现潜在问题，并建议可能进行了 rebase 而非 merge。
- **引入 Pad ID 错误修复 PR**：一名成员指出一个关于 **generate 中显示 pad ID** 的严重 Bug，随后创建了 [Pull Request #1211](https://github.com/pytorch/torchtune/pull/1211) 以防止此问题。
   - 该 PR 旨在解决 **utils.generate 中隐式假设 Pad ID 为 0** 的问题，并澄清了其对特殊 token 的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/pull/1211">Prevent pad ids, special tokens displaying in generate by RdoubleA · Pull Request #1211 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档还是其他（请在此处添加）。utils.generate 中隐式假设 Pad ID 为 0，...</li><li><a href="https://github.com/pytorch/torchtune/pull/790">MPS support by maximegmd · Pull Request #790 · pytorch/torchtune</a>：上下文：出于测试目的，直接在本地 Mac 电脑上运行很有用。变更日志：检查 MPS 设备对 BF16 的支持。添加了针对 MPS 的配置，对路径进行了更改...
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1265091468771725428)** (15 条消息🔥): 

> - `使用 tinygrad 实现 matmul-free-llm`
> - `M1 性能差异`
> - `使用 PYTHON=1 时的测试挑战`
> - `tinygrad 中的 cumsum 优化`
> - `TensorFlow 与 PyTorch 张量操作对比` 


- **寻求重现 matmul-free-llm 的帮助**：有人请求协助使用 tinygrad 重现 **matmul-free-llm**，旨在利用 [高效 kernel](https://github.com/ridgerchu/matmulfreellm/blob/master/mmfreelm/ops/fusedbitnet.py) 并整合 **fp8**。
   - *希望能尽快无缝适配 Blackwell fp4*。
- **M1 结果与 CI 不同**：一位 M1 用户遇到了与 CI 不同的结果，正在寻求关于如何使用 **conda** 和环境变量正确设置测试的说明。
   - *由于启用 `PYTHON=1` 时出现差异而产生困惑，因为这会导致测试中出现 IndexError。*
- **cumsum 性能关注**：一位新成员正在探索 tinygrad 中 **nn.Embedding** 的 **O(n)** 实现，以及如何借鉴 PyTorch 的技术将 **cumsum** 从 O(n^2) 提升到 O(n)。
   - *有人推测某些约束使得这项任务具有挑战性，尤其是考虑到这是一个 **1000 美元的悬赏任务**。*
- **TensorFlow 和 PyTorch 张量操作差异**：关于 **TensorFlow bitcast** 和 **PyTorch view** 行为差异的讨论正在进行，主要集中在维度处理方式上。
   - *增加或减少维度可能会引起混淆，一些人认为 TensorFlow 的行为在这种情况下更有意义。*
- **bitcast 和 view 的测试问题**：由于设备对 **view** 和 **bitcast** 的支持差异，使用 `PYTHON=1` 时出现了测试问题，导致形状兼容性问题。
   - *大家一致认为，虽然 PyTorch 和 NumPy 会扩展或收缩维度，但 TensorFlow 的方法是添加一个新维度。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/ridgerchu/matmulfreellm/blob/master/mmfreelm/ops/fusedbitnet.py">matmulfreellm/mmfreelm/ops/fusedbitnet.py at master · ridgerchu/matmulfreellm</a>：MatMul-free LM 的实现。通过在 GitHub 上创建账号为 matmulfreellm 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/issues/1612),">Issues · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - Issues · tinygrad/tinygrad
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1265319278228541532)** (6 messages): 

> - `PyTorch 中的增量测试`
> - `Tinygrad 中的分子动力学引擎`
> - `梯度计算`
> - `神经网络势能 (Neural Network Potentials)` 


- **寻求 PyTorch 增量测试模式**：一位成员询问了在 PyTorch 中按 **Linear, MLP, MoE,** 和 **LinearAttentionMoE** 顺序增量测试模型性能的有效模式。
   - 他们质疑从头开始测试是否比增量测试更有效率。
- **在 Tinygrad 中开发分子动力学引擎**：一个小组正尝试在 tinygrad 中实现一个**分子动力学引擎**，用于训练预测分子构型能量的模型，但在梯度计算方面遇到了挑战。
   - 他们需要预测能量相对于输入位置的梯度来计算力，但由于对模型权重进行了两次反向传播，出现了问题。
- **需要高效的梯度计算**：开发者解释了通过不同图计算能量/位置梯度的挑战，类似于 PyTorch 中的 **torch.autograd.grad**。
   - 这对于确保第一次梯度计算不影响 Loss 计算至关重要，他们计划分享一个最小示例（minimal example）以寻求帮助。
- **鼓励提交带有最小复现的 PR**：George Hotz 建议开发者发布问题的最小复现（minimal reproduction）以及预期行为，以便更好地提供帮助。
   - 他建议这个最小示例最好能作为测试添加到 Pull Request (PR) 中。
- **与神经网络势能的联系**：另一位成员 James Wiles 询问分子动力学项目是否与 **Neural Network Potentials** 有关。
   - 这表明了对这些概念在工作背景下如何交汇的兴趣。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1265055411216846869)** (9 messages🔥): 

> - `Int8 使用`
> - `ComfyUI 工作流`
> - `Llama 3.1 发布`
> - `Whisper Speech 工具`
> - `扎克伯格关于 Llama 3.1 的谈话` 


- **关于 Int8 实现的讨论**：成员们询问了关于使用 **Int8** 的问题，一位成员确认他们可以使其正常工作。
   - 讨论过程中有人请求“稍等片刻”，暗示会有进一步的支持。
- **ComfyUI 工作流指导**：分享脚本的请求得到了使用 **ComfyUI flow** 进行设置的回应。
   - 这反映了社区对流线型工作流的偏好。
- **分享 Llama 3.1 更新**：一位成员在特定频道提到了 **Llama 3.1 博客**，表明了对更新的极大兴趣。
   - 这突显了围绕 Llama 模型进展的持续讨论。
- **关于 Whisper Speech 工具的查询**：有人提问所提供链接中的 **Whisper Speech** 工具的工作状态。
   - 成员们参与了检查该工具当前状态的行动，展示了活跃的社区参与。
- **扎克伯格讨论 Llama 3.1**：一位成员分享了一个 [YouTube 视频](https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61)，马克·扎克伯格在其中讨论了 **Llama 3.1** 及其竞争优势。
   - 该视频强调 Llama 3.1 是有史以来第一个开源的前沿 AI 模型，在多个基准测试中表现出色。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61">Mark Zuckerberg on Llama 3.1, Open Source, AI Agents, Safety, and more</a>：Meta 刚刚发布了 Llama 3.1 405B —— 这是有史以来第一个开源的前沿 AI 模型，在多个基准测试中击败了 GPT-4o 等顶级封闭模型。我坐下来...</li><li><a href="https://collabora-whisperspeech.hf.space/">Gradio</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1265065927959908464)** (5 messages): 

> - `Meta 对开源 AI 的承诺`
> - `Llama 3.1 的能力`
> - `上下文长度的改进` 


- **Meta 倡导开源 AI**：正如马克·扎克伯格的信函中所详述，Meta 致力于[开放获取的 AI](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/)，强调其对开发者和更广泛社区的益处。
   - 这与其通过 AI 生态系统中的协作来促进创新的愿景相一致。
- **Llama 3.1 树立了新标杆**：[Llama 3.1 405B](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) 的发布引入了前所未有的能力，包括 **128K** 的上下文长度和对八种语言的支持。
   - 该模型提供了灵活性和控制力，使其在与顶级闭源替代方案的竞争中占据优势。
- **解决了对上下文大小的批评**：讨论强调了之前的 **8K** 上下文大小被认为不足以高效处理大型文档。
   - 向 **128K** 上下文大小的跨越被视为对需要大量文档处理任务的重大改进。



**提及的链接**：<a href="https://ai.meta.com/blog/meta-llama-3-1/">未找到标题</a>：未找到描述

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1265406571203137587)** (1 messages): 

> - `Llama 3.1 405 B`
> - `GPT-4o 性能` 


- **Llama 3.1 405 B 令用户惊叹**：据报道，**Llama 3.1 405 B** 与 **OpenInterpreter** 配合使用时开箱即用，效果极佳。
   - 用户注意到，与 **GPT-4o** 不同，它不需要不断的提醒或重启即可完成多项任务。
- **对 GPT-4o 的挫败感**：一位用户表达了对 **GPT-4o** 的挑战，需要频繁的提示才能在他们的计算机上执行任务。
   - 这种挫败感凸显了用户在使用 **Llama 3.1 405 B** 时获得的无缝体验。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1265049476913954969)** (3 messages): 

> - `使用 Coqui Model 的语音输入`
> - `适用于 Apple Watch 的 Expo App`
> - `设备发货时间线` 


- **在 MacOS 上使用 Coqui Model 进行语音输入？**：有人提出了关于在 **MacOS** 上使用本地 **Coqui model** 进行语音输入的可行性查询。
   - 目前尚未提供详述任何成功实现的回复。
- **Expo App 针对 Apple Watch 的能力**：有一场讨论确认了 **Expo app** 理论上应该能够为 **Apple Watch** 构建应用程序。
   - *未提供进一步的细节或确认*。
- **设备的发货时间线**：一名成员询问了特定设备的发货时间线，表明了对其状态的持续关注。
   - 对话中未分享任何更新或时间线。


  

---



### **Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

spirit_from_germany: https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61
  

---


### **Alignment Lab AI ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/1265354431889674271)** (2 messages): 

> - `OpenOrca 数据集许可`
> - `合成数据集公告` 


- **关于 OpenOrca 数据集许可的澄清**：一名成员寻求关于 **OpenOrca** 数据集许可的澄清，特别是质疑其 **MIT License** 是否允许将其输出用于商业用途，因为该数据集源自 **GPT-4 Model**。
   - *其输出可以用于商业目的吗？*
- **开源合成数据集的计划**：另一名成员宣布了开源一个**合成数据集**的计划，该数据集将支持非商业和商业应用。
   - 他们提到在创建数据集时正在评估对 **OpenOrca** 的依赖，表明了对其许可影响的兴趣。


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/1265144542382919792)** (2 messages): 

> - `迈阿密聚会`
> - `8 月份对纽约市（NYC）的兴趣` 


- **迈阿密聚会查询**：一名成员询问是否有人在**迈阿密**附近，可能是在寻找聚会或集会。
   - 关于此查询，未分享进一步的细节或回复。
- **对纽约市（NYC）集会的兴趣**：另一名成员表达了参加 8 月下旬在 **NYC** 举行的聚会的兴趣。
   - 这一询问为该地区成员之间的联系提供了潜在机会。


  

---

### **AI Stack Devs (Yoko Li) ▷ #[team-up](https://discord.com/channels/1122748573000409160/1128471951963328512/)** (1 条消息): 

ari991963: 大家好，我是 Aria，一名 2D/3D 艺术家，如果你有兴趣合作请私信。
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1265387661103726703)** (1 条消息): 

> - `Mozilla Accelerator 申请截止日期`
> - `Zero Shot Tokenizer Transfer 活动`
> - `AutoFix 项目概览` 


- **Mozilla Accelerator 申请截止日期临近**：**Mozilla Accelerator** 的申请截止日期即将到来，该项目提供为期 **12 周** 的计划，以及高达 **10 万美元** 的非稀释性资金。
   - 参与者还将有机会在与 Mozilla 共同举办的 **demo day** 上展示他们的项目。[有问题吗？](https://discord.com/channels/1089876418936180786/1245083732319408195)
- **准备参加 Zero Shot Tokenizer Transfer 活动**：提醒本月即将举行的一场由 Benjamin Minixhofer 主讲的 **Zero Shot Tokenizer Transfer** 活动。
   - 详情可以在活动链接中找到。[活动信息](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732)
- **介绍 AutoFix：开源问题修复器**：**AutoFix** 是一款开源的问题修复工具，可以直接从 **Sentry.io** 提交 PR，提供了一种高效的问题管理方式。
   - 您可以在链接的详细文章中了解更多关于这款创新工具的信息。[AutoFix 信息](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732)


  

---



---



---



---



{% else %}


> 由于邮件长度限制，完整的频道细分内容已被截断。
> 
> 如果你想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}