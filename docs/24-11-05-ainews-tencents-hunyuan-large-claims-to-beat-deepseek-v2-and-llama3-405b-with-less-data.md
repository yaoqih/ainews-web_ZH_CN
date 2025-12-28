---
companies:
- tencent
- anthropic
- meta-ai-fair
- togethercompute
- llamaindex
date: '2024-11-06T06:22:40.424116Z'
description: '**腾讯**发布了一款备受关注的、参数量超过 3000 亿的 MoE（混合专家）模型。该模型在 7 万亿（7T）token 上进行了预训练，其中包括通过
  Evol-Instruct 生成的 1.5 万亿（1.5T）合成数据。该模型引入了诸如“循环路由”（recycle routing）和专家特定学习率等新技术，以及针对
  MoE 激活参数的计算高效缩放定律（scaling law）。然而，其自定义许可证限制了在欧盟境内的使用，也禁止月活跃用户数（MAU）超过 1 亿的公司使用，且该模型会避开涉及中国敏感话题的查询。


  与此同时，**Anthropic** 推出了 Claude 3.5 Haiku，目前已在多个平台上线。尽管其智能程度和速度备受赞誉，但因价格上涨了 10 倍而遭到批评。**Meta**
  向美国国防部门开放了 Llama AI，并举办了 Llama Impact 黑客松，为使用 Llama 3.1 和 3.2 Vision 的项目提供 1.5 万美元的奖金。**LlamaIndex**
  发布了一个集成了 Tailwind CSS 和大模型后端的 React 聊天 UI 组件。**MLX LM** 模型则通过 KV 缓存量化技术，进一步提升了文本生成的执行速度和效率。'
id: 2afdc67e-0128-4608-8ca5-f19a62ad396e
models:
- claude-3.5-haiku
- llama-3-1
- llama-3-2
- mlx-lm
original_slug: ainews-tencents-hunyuan-large-claims-to-beat
people: []
title: 腾讯的 Hunyuan-Large 声称以更少的数据击败了 DeepSeek-V2 和 Llama3-405B。
topics:
- mixture-of-experts
- synthetic-data
- model-scaling
- model-architecture
- model-optimization
- kv-cache-quantization
- react
- fine-tuning
- scaling-laws
- model-efficiency
- model-deployment
- multimodality
---

<!-- buttondown-editor-mode: plaintext -->**Evol-instruct 合成数据就是你所需要的一切。**

> 2024年11月4日至11月5日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务器（**217** 个频道，**3533** 条消息）。预计节省阅读时间（以 200wpm 计算）：**364 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们倾向于对中国模型设定很高的标准，尤其是来自此前不为人知的团队。但腾讯今天发布的成果（[huggingface](https://huggingface.co/tencent/Tencent-Hunyuan-Large)，[论文](https://arxiv.org/pdf/2411.02265)，[HN 评论](https://news.ycombinator.com/item?id=42054186)）在与其宣称的已知 SOTA 开源权重模型的对比中非常引人注目：


![image.png](https://assets.buttondown.email/images/25a4e4fe-0fc6-41ff-a443-6d753f9755f4.png?w=960&fit=max)


值得注意的是，作为一个参数量超过 300B 的模型（无论是否为 MoE），它的数据效率非常高，仅在“只有” 7T tokens 上进行了预训练（DeepseekV2 是 8T，Llama3 是 15T），其中 1.5T 是通过 Evol-Instruct 生成的合成数据，Wizard-LM 团队对此也深有感触：


![image.png](https://assets.buttondown.email/images/1798a019-ffa4-4074-b516-02e0ad7ef6da.png?w=960&fit=max)



![image.png](https://assets.buttondown.email/images/1e368282-dfd4-46a1-a302-eab911d70597.png?w=960&fit=max)


论文提供了一些他们探索的新颖方法的详细研究细节，包括 "recycle routing"：


![image.png](https://assets.buttondown.email/images/147a4733-0a97-4264-855c-35f03dccd520.png?w=960&fit=max)

 
以及 expert-specific LRs


![image.png](https://assets.buttondown.email/images/dd4a6b3e-70ec-498e-9df4-63350de8496c.png?w=960&fit=max)


他们甚至研究并提供了一种针对 MoE 激活参数的计算高效的 scaling law：


![image.png](https://assets.buttondown.email/images/6a6d7b81-892b-45e8-9b70-6fff115994f6.png?w=960&fit=max)


情况并非全是正面的：自定义许可证禁止欧盟用户和 MAU 超过 1 亿的公司使用，当然，也不要问他们[关于中国的敏感问题](https://x.com/teortaxesTex/status/1853753632237232476)。Vibe checks 尚未得出结论（我们还没发现有人托管了方便的公共端点），目前也没有人对此大肆宣扬。尽管如此，对于这类模型来说，这仍然是一项不错的研究成果。

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型发布与更新**

- **Claude 3.5 Haiku 增强功能**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1853498272863691125) 宣布 **Claude 3.5 Haiku** 现已在 Anthropic API、Amazon Bedrock 和 Google Cloud 的 Vertex AI 上可用，将其定位为迄今为止**最快且最智能的性价比模型**。[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1853598554570555614) 分析称 **Claude 3.5 Haiku** **提升了智能水平**，但指出其**价格飙升**，使其比 Google 的 Gemini Flash 和 OpenAI 的 GPT-4o mini 等竞争对手**贵了 10 倍**。此外，[@skirano](https://twitter.com/skirano/status/1853506128358834358) 分享道 **Claude 3.5 Haiku** 是**最有趣的模型之一**，在各种任务上表现优于之前的 Claude 模型。

- **Meta 的 Llama AI 用于国防**：[@TheRundownAI](https://twitter.com/TheRundownAI/status/1853761707195113742) 报道称 **Meta** 已向**美国国防部门开放 Llama AI**，标志着 AI 领域的重大合作。

**AI 工具与基础设施**

- **转换会议录音**：[@TheRundownAI](https://twitter.com/TheRundownAI/status/1853537816531341781) 介绍了一种将**会议录音转化为可操作见解**的工具，增强了生产力和信息获取能力。

- **Llama Impact 黑客松**：[@togethercompute](https://twitter.com/togethercompute/status/1853564304391651646) 和 [@AIatMeta](https://twitter.com/AIatMeta/status/1853513932520186013) 正在**举办一场黑客松**，专注于使用 **Llama 3.1 & 3.2 Vision** 构建解决方案，提供 **$15K 奖金池**并鼓励在**现实挑战**上进行协作。

- **LlamaIndex Chat UI**：[@llama_index](https://twitter.com/llama_index/status/1853589578965451108) 推出了 **LlamaIndex chat-ui**，这是一个用于构建聊天界面的 **React 组件库**，具有 **Tailwind CSS 自定义功能**以及与 Vercel AI 等 **LLM 后端集成**的特点。

**AI 研究与基准测试**

- **MLX LM 进展**：[@awnihannun](https://twitter.com/awnihannun/status/1853566353141276993) 强调，**最新的 MLX LM** 在处理**超大型模型**时生成文本的速度**更快**，并引入了 **KV cache 量化**以提高效率。

- **自进化 RL 框架**：[@omarsar0](https://twitter.com/omarsar0/status/1853821990177485311) 提出了一个 **自进化在线课程 RL 框架**，显著 **提高了模型（如 Llama-3.1-8B）的成功率**，表现优于 **GPT-4-Turbo** 等模型。

- **LLM 评估综述**：[@sbmaruf](https://twitter.com/sbmaruf/status/1853498895537446941) 发布了一份关于评估 **Large Language Models** 的 **系统性综述**，探讨了对于 **稳健模型评估** 至关重要的 **挑战与建议**。

**AI 行业事件与黑客松**

- **AI 高价值动态**：[@TheRundownAI](https://twitter.com/TheRundownAI/status/1853761707195113742) 分享了 **顶级 AI 故事**，包括 **Meta 用于国防的 Llama AI**、**Anthropic 发布 Claude Haiku 3.5**，以及 **Physical Intelligence 获得 4 亿美元融资** 等新闻。

- **Builder's Day 回顾**：[@ai_albert__](https://twitter.com/alexalbert__/status/1853533686211436560) 回顾了与 **@MenloVentures** 合作举办的首届 **Builder's Day** 活动，强调了开发者之间的 **才华与协作**。

- **ICLR 紧急征集审稿人**：[@savvyRL](https://twitter.com/savvyRL/status/1853524851509858805) 为 **LLM 推理** 和 **代码生成** 等主题征集 **紧急审稿人**，强调了对专家评审的迫切需求。

**AI 定价与市场反应**

- **Claude 3.5 Haiku 定价争议**：[@omarsar0](https://twitter.com/omarsar0/status/1853585918927511644) 对 **Claude 3.5 Haiku** 的 **价格飙升** 表示担忧，质疑其相对于 **GPT-4o-mini** 和 **Gemini Flash** 等其他模型的 **价值主张**。同样，[@bindureddy](https://twitter.com/bindureddy/status/1853585512017367127) 批评了 **4 倍的价格上涨**，认为这与 **性能提升** 不符。

- **Python 3.11 性能提升**：[@danielhanchen](https://twitter.com/danielhanchen/status/1853535612898533715) 提倡升级到 **Python 3.11**，详细介绍了其在 Linux 上 **快 1.25 倍**、在 Mac 上 **快 1.2 倍** 的性能表现，以及 **优化的帧对象** 和 **函数调用内联** 等改进。

- **腾讯的合成数据策略**：[@_philschmid](https://twitter.com/_philschmid/status/1853703814114623898) 讨论了 **腾讯** 的策略，即在 **1.5 万亿合成 Token** 上训练其 **389B 参数 MoE**，并强调了其优于 **Llama 3.1** 等模型的 **性能**。

**模因与幽默**

- **AI 与选举幽默**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1853722841671201042) 幽默地请求 GPT 在三天内 **删除非编程和非小猫相关的推文**，并生成一份 **欢快的事件总结**。

- **有趣的模型行为**：[@reach_vb](https://twitter.com/reach_vb/status/1853486414798733314) 分享了一个 **音频生成模型** “失控”的幽默观察，而 [@hyhieu226](https://twitter.com/hyhieu226/status/1853491814646661281) 则开玩笑地发布了关于 **特定 AI 回复** 的推文。

- **用户互动与反应**：[@nearcyan](https://twitter.com/nearcyan/status/1853682972886728874) 发布了一个与政治相关的模因，而 [@kylebrussell](https://twitter.com/kylebrussell/status/1853569407278281137) 分享了一条轻松的“氛围感”推文。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. 腾讯混元-Large：开源模型中的游戏规则改变者**

- **[腾讯刚刚发布了一个开源权重的 389B MoE 模型](https://arxiv.org/pdf/2411.02265)** ([Score: 336, Comments: 132](https://reddit.com/r/LocalLLaMA/comments/1gjzd1i/tencent_just_put_out_an_openweights_389b_moe_model/))：腾讯发布了一个名为 **Hunyuan-Large** 的开源权重 **389B MoE 模型**，旨在性能上与 **Llama** 竞争。该模型架构利用了 **Mixture of Experts (MoE)**，实现了高效扩展并提升了处理复杂任务的能力。
  - **Hunyuan-Large** 模型拥有 **3890 亿参数**，其中 **520 亿激活参数**，支持 **高达 256K Token**。用户注意到其高效利用 CPU 的潜力，一些人在 **DDR4** 上有效运行了类似模型，并对该模型与 **Llama** 变体相比的能力感到兴奋。
  - 讨论强调了模型的 **巨大体量**，据估计运行该模型需要 **200-800 GB 内存**，具体取决于配置。用户还分享了性能指标，表明它可能优于 **Llama3.1-70B**，同时由于其 **Mixture of Experts (MoE)** 架构，推理成本更低。
  - 考虑到 **中国的 GPU 制裁**，人们对硬件限制产生了担忧，引发了关于腾讯如何运行此类大型模型的疑问。用户推测需要高端配置，有人开玩笑说需要一座 **核电站** 来为所需的 GPU 供电。

**主题 2. Tensor Parallelism 增强 Llama 模型：基准测试见解**

- **公告：llama.cpp 补丁使我的最大上下文长度翻倍** ([Score: 95, Comments: 10](https://reddit.com/r/LocalLLaMA/comments/1gjq1y0/psa_llamacpp_patch_doubled_my_max_context_size/))：最近针对 **llama.cpp** 的一个补丁，使使用 **3x Tesla P40 GPU** 并在行拆分模式（`-sm row`）下运行的用户，其最大上下文长度从 **60K tokens** 翻倍至 **120K tokens**。正如 [Pull Request](https://github.com/ggerganov/llama.cpp/pull/10026) 中详述的那样，这一改进还带来了更均衡的 GPU VRAM 使用率，在不影响推理速度的情况下提升了整体 GPU 利用率。
  - 使用 **3x Tesla P40 GPU** 的用户报告称，由于上下文长度从 **60K 增加到 120K tokens**，他们的工作流得到了显著改善。一位用户指出，之前的限制迫使他们只能在小上下文中使用大型模型，这阻碍了性能发挥，而该补丁允许更高效的模型使用。
  - 几条评论强调了新补丁易于实现的特点，一位用户成功在 **QWEN-2.5-72B_Q4_K_S** 上加载了 **16K 上下文**，并表示性能与之前的速度保持一致。另一位用户对模型按行拆分时缓存处理能力的提升表示兴奋。
  - 用户分享了优化 GPU 性能的技巧，包括建议使用 [nvidia-pstated](https://github.com/sasha0552/nvidia-pstated) 来管理 P40 的电源状态。该工具有助于在 GPU 负载和闲置时保持较低的功耗（8-10W），从而提高整体效率。

- **4x RTX 3090 + Threadripper 3970X + 256 GB RAM LLM 推理基准测试** ([Score: 48, Comments: 39](https://reddit.com/r/LocalLLaMA/comments/1gjovjm/4x_rtx_3090_threadripper_3970x_256_gb_ram_llm/))：该用户在一台配备 **4x RTX 3090** GPU、**Threadripper 3970X** 和 **256 GB RAM** 的设备上进行了 **LLM 推理**基准测试。结果显示，**Qwen2.5** 和 **Mistral Large** 等模型表现出不同的 **tokens per second (tps)**，Tensor Parallel 实现显著增强了性能，推理期间 PCIe 传输速率从 **1 kB/s** 增加到 **200 kB/s** 证明了这一点。
  - 用户讨论了电源的稳定性，**kryptkpr** 建议使用 **Dell 1100W 电源**配合转接板（breakout boards）以实现可靠的电力输送，在闲置时可达到 **12.3V**。他们还分享了用于 PCIe 连接的可靠转接板链接。
  - **Lissanro** 建议在使用 Tensor Parallel 的同时，通过 **TabbyAPI (ExllamaV2)** 探索 **Speculative Decoding**（投机解码），强调了在使用具有激进量化技术的 **Qwen 2.5** 和 **Mistral Large** 等模型时潜在的性能提升。同时也提供了这些模型的相关链接。
  - **a_beautiful_rhind** 指出 **Exllama** 没有实现 **NVLink**，这限制了其性能潜力，而 **kmouratidis** 则建议在不同的 **PCIe 配置**下进行进一步测试，以评估潜在的降频影响。

**主题 3. 编程模型领域的竞争进展：Qwen2.5-Coder 分析**

- **Qwen2.5-Coder-32B 到底在哪？** ([Score: 76, Comments: 21](https://reddit.com/r/LocalLLaMA/comments/1gjvf6w/so_wheres_qwen25coder32b/))：**Qwen2.5-Coder-32B** 版本正在准备中，旨在与领先的专有模型竞争。该团队还在研究先进的**以代码为中心的推理模型**，以增强代码智能，更多更新将在其 [博客](https://qwen2.org/qwen2-5-coder/) 上发布。
  - 用户对 **Qwen2.5-Coder-32B** 的发布时间表表示怀疑，评论指出“即将推出”这句话已经说了 **两个月**，却没有任何实质性更新。
  - 用户 **radmonstera** 分享了他们使用 **Qwen2.5-Coder-7B-Base** 进行自动补全并配合 **70B 模型**使用的经验，指出 **32B** 版本虽然可以减少 RAM 占用，但速度可能无法与 **7B** 模型媲美。
  - 普遍对该发布充满期待，用户 **StarLord3011** 希望能在几周内发布，而另一位用户 **visionsmemories** 则幽默地调侃了发布过程中可能存在的疏忽。

- **编程模型正变得越来越强大** ([Score: 170, Comments: 71](https://reddit.com/r/LocalLLaMA/comments/1gjtelf/coders_are_getting_better_and_better/)): 用户越来越多地在本地大语言模型 (LLM) 应用中采用 **Qwen2.5 Coder 7B**，并称赞其**速度**和**准确性**。一位用户报告称在搭载 **LM Studio** 的 **Mac** 上成功运行。
  - 用户报告 **Qwen2.5 Coder 7B** 性能出色，一位用户在 **M3 Max MacBook Pro** 上运行达到了每秒约 **18 tokens** 的速度。另一位用户强调 **Qwen 2.5 32B** 模型在各项任务中表现优于 **Claude**，尽管对于本地 LLM 编程模型与 **Claude** 及 **GPT-4o** 相比的能力仍存在一些质疑。
  - 基于 **Qwen 2.5 14B** 的 **Supernova Medius** 模型被强调为一款高效的编程助手，用户分享了该模型的 [GGUF](https://huggingface.co/bartowski/SuperNova-Medius-GGUF) 链接和原始权重链接（[点击此处](https://huggingface.co/arcee-ai/SuperNova-Medius)）。用户对专用 **32B coder** 模型的潜力表示出浓厚兴趣。
  - 讨论显示用户对 **Qwen 2.5** 的评价褒贬不一，部分用户认为它处理基础任务表现良好，但在处理比 **Claude** 和 **OpenAI** 模型更复杂的编程场景时仍显不足。一位用户提到，虽然 **Qwen 2.5** 在离线使用方面表现稳健，但仍无法与 **GPT-4o** 等更先进的闭源模型相媲美。


**主题 4. 新型 AI 工具：语音克隆与投机采样技术**



- **[OuteTTS-0.1-350M - 基于 LLaMa 架构的零样本语音克隆，采用 CC-BY 许可！](https://v.redd.it/1xekc1fhw1zd1)** ([Score: 69, Comments: 13](https://reddit.com/r/LocalLLaMA/comments/1gk2s7l/outetts01350m_zero_shot_voice_cloning_built_on/)): **OuteTTS-0.1-350M** 具有基于 **LLaMa 架构** 的**零样本语音克隆 (Zero-shot voice cloning)** 功能，并以 **CC-BY 许可** 发布。该模型代表了语音合成技术的重大进步，无需针对特定语音数据进行预先训练即可生成语音输出。
  - **OuteTTS-0.1-350M** 模型利用了 **LLaMa 架构**，受益于 **llama.cpp** 的优化，并在 [Hugging Face](https://huggingface.co/OuteAI/OuteTTS-0.1-350M-GGUF) 上提供了 **GGUF** 版本。
  - 用户强调**零样本语音克隆**能力是**语音合成技术**的一项重大突破，[官方博客](https://www.outeai.com/blog/OuteTTS-0.1-350M) 链接提供了更多细节。
  - 讨论中提到了 TTS 系统中的**语音恐怖谷 (Audio uncanny valley)** 现象，即微小的错误会导致输出结果“几乎”像人类，从而给听者带来不安的体验。


- **OpenAI 新功能 “Predicted Outputs” 使用投机采样技术** ([Score: 51, Comments: 28](https://reddit.com/r/LocalLLaMA/comments/1gjzmjp/openai_new_feature_predicted_outputs_uses/)): OpenAI 的新功能 **“Predicted Outputs”** 利用了**投机采样 (Speculative decoding)** 技术，这一概念一年多前就已在 **llama.cpp** 中得到验证。该帖子提出了关于在 **70B 规模模型** 以及 **llama3.2** 和 **qwen2.5** 等较小模型上实现更快推理的潜力，特别是对于本地用户而言。更多详情请参阅此处的 [推文](https://simonwillison.net/2024/Nov/4/predicted-outputs/) 以及 **Karpathy** 的 [演示](https://x.com/karpathy/status/1697318534555336961)。
  - **投机采样 (Speculative decoding)** 可以通过允许较小模型快速生成初始 token 序列，再由较大模型进行验证，从而显著提升推理速度。像 **Ill_Yam_9994** 和 **StevenSamAI** 这样的用户讨论了这种方法如何有效地实现并行处理，在通常生成一个 token 的时间内可能生成多个 token。
  - 几位用户指出，虽然 **“Predicted Outputs”** 功能可能会降低延迟，但不一定能降低模型使用成本，正如 **HelpfulHand3** 所言。该技术被公认为**设备端推理 (On-device inference)** 的标准，但正如 **Old_Formal_1129** 所提到的，对较小模型进行适当的训练对于最大化性能至关重要。
  - 对话还涉及了分层模型的构想，即较小模型预测输出，较大模型进行验证，这可能会带来显著的速度提升，正如 **Balance-** 所提议的那样。这种分层方法引发了关于整合多种规模模型以实现最佳性能的有效性和可行性的讨论。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**自主系统与安全**

- **大众汽车的紧急辅助技术 (Emergency Assist Technology)**：在 /r/singularity 中，[大众汽车展示了新的自动驾驶技术](https://www.reddit.com/r/singularity/comments/1gj9zay/volkswagens_new_emergency_assist_technology/)，该技术可在驾驶员失去反应时安全地将车辆停靠在路边，在激活前会进行多个阶段的驾驶员注意力检查。
  - 关键评论见解：系统包含对避免误触发和保持驾驶员控制的细致考量。

**AI 安全与漏洞**

- **Google 的 Big Sleep AI Agent**：在 /r/OpenAI 和 /r/singularity 中，[Google 的安全 AI 发现了 SQLite 中的一个零日漏洞 (zero-day vulnerability)](https://www.reddit.com/r/OpenAI/comments/1gjwexq/google_claims_world_first_as_ai_finds_0day/)，这标志着 AI Agent 首次公开在广泛使用的软件中发现以前未知的可利用内存安全问题。
  - 技术细节：该漏洞已于 10 月正式发布前报告并修复。

**3D 化身生成与渲染**

- **URAvatar 技术**：在 /r/StableDiffusion 和 /r/singularity 中，[一项新研究展示了逼真的头部化身 (head avatars)](https://www.reddit.com/r/StableDiffusion/comments/1gjfn05/uravatar_relightable_avatars_from_a_single_phone/)，该研究使用光照未知的手机扫描，具有以下特点：
  - 具有全局光照的实时渲染
  - 用于光传输的可学习辐射传输 (radiance transfer)
  - 在数百个高质量多视图人体扫描上进行训练
  - 3D Gaussian 表示

**行业动态与企业 AI**

- **OpenAI 动态**：多个 Subreddit 的帖子指出：
  - [意外泄露了具有视觉能力的完整 O1 模型](https://www.reddit.com/r/singularity/comments/1gjlxc1/openai_accidentally_leaked_their_full_o1_model/)
  - [聘请了 META 的 AR 眼镜负责人](https://www.reddit.com/r/singularity/comments/1gju6vb/head_of_ar_glasses_orion_at_meta_caitlin/) 负责机器人和消费级硬件
  - [Sam Altman 预告了新的 OpenAI 图像模型功能](https://www.reddit.com/r/singularity/comments/1gj9rxq/sam_altman_teases_new_openai_image_model_without/)

**AI 图像生成评论**

- **Adobe AI 的局限性**：在 /r/StableDiffusion 中，[用户报告了 Adobe AI 图像生成工具中显著的内容限制](https://www.reddit.com/r/StableDiffusion/comments/1gjisot/just_wanted_to_say_adobes_ai_is_horrible/)，特别是在人物主体和服装方面。
  - 技术限制：由于过于激进的内容过滤，系统甚至会阻止基础的图像编辑任务。

**迷因与幽默**

- [Anthropic 定价策略讨论](https://www.reddit.com/r/singularity/comments/1gjm1wa/anthropic_tries_to_fight_the_recent_rapid_fall_in/)
- [ChatGPT 选举预测幽默](https://www.reddit.com/r/OpenAI/comments/1gjzdce/chatgpt_already_knows_who_won_the_election/)

---

# AI Discord 简报

> 由 O1-preview 生成的摘要之摘要的摘要

**主题 1. AI 巨头发布巨型模型：新的重量级选手**

- **腾讯发布 389B 参数的 Hunyuan-Large MoE 模型**：腾讯发布了 [Hunyuan-Large](https://arxiv.org/abs/2411.02265)，这是一个巨大的 Mixture-of-Experts 模型，拥有 **3890 亿参数**和 **520 亿激活参数**。虽然品牌定位为开源，但关于其真实的易用性以及运行它所需的庞大基础设施，讨论依然激烈。
- **Anthropic 在用户抱怨声中推出 Claude 3.5 Haiku**：Anthropic 推出了 **Claude 3.5 Haiku**，用户们正急于测试其在**速度**、**代码准确性**和**工具集成**方面的表现。然而，**Claude 3 Opus** 的移除引发了不满，因为许多人更倾向于将其用于编程和叙事。
- **OpenAI 通过 Predicted Outputs 降低 GPT-4 延迟**：OpenAI 引入了 [Predicted Outputs](https://x.com/OpenAIDevs/status/1853564730872607229)，通过提供参考字符串，大幅降低了 **GPT-4o** 模型的延迟。基准测试显示，在文档迭代和代码重写等任务中，速度提升高达 **5.8 倍**。

---

**主题 2. 国防遇见 AI：LLM 投身国家安全**

- **Scale AI 为机密任务部署 Defense Llama**：[Scale AI](https://x.com/alexandr_wang/status/1853853829336559790) 宣布推出 **Defense Llama**，这是一款与 **Meta** 及国防专家共同开发的专用 LLM，旨在服务于美国国家安全应用。该模型已准备好集成到美国国防系统中。
- **Nvidia 的 Project GR00T 旨在打造机器人霸主**：来自 NVIDIA **GEAR** 团队的 **Jim Fan** 讨论了 [Project GR00T](https://www.youtube.com/live/Qhxr0uVT2zs)，旨在开发能够在模拟和真实环境中运行的 AI Agent，增强机器人的通用能力。
- **OpenAI 致力于安全 AGI 开发**：成员们强调了 OpenAI 自 2015 年以来的创始目标，即构建**安全且有益的 AGI**。讨论还涉及了对 AI 自我开发的担忧，即如果成本超过了所有人类投入的情况。

---

**主题 3. 开放数据盛宴：数据集将为 AI 注入强劲动力**

- **Open Trusted Data Initiative 预告 2 万亿 Token 数据集**：**Open Trusted Data Initiative** 计划于 **11 月 11 日**通过 [Hugging Face](https://huggingface.co/) 发布一个包含 **2 万亿 Token** 的海量多语言数据集，旨在提升 LLM 的训练能力。
- **社区辩论训练数据的质量与数量**：讨论强调了高质量数据集对未来 AI 模型的重要性。有人担心优先考虑质量可能会排除有价值的主题，但它能增强**常识推理（commonsense reasoning）**。
- **EleutherAI 增强 LLM 鲁棒性评估**：[LLM Robustness Evaluation](https://github.com/EleutherAI/lm-evaluation-harness/pull/2452) 开启了一个 Pull Request，引入了跨三个数据集的系统性一致性和鲁棒性评估，并修复了之前的 Bug。

---

**主题 4. 用户对机器的愤怒：AI 工具备受指责**

- **Perplexity 用户哀悼 Claude 3 Opus 的缺失**：从 Perplexity AI 中移除 **Claude 3 Opus** 导致了用户的不满，许多人声称它是他们编程和叙事的首选模型。**Haiku 3.5** 被认为是一个效果较差的替代品。
- **LM Studio 用户应对故障和性能问题**：LM Studio 用户报告了模型性能方面的挑战，包括 **Hermes 405B** 的结果不一致，以及从 USB 驱动器运行软件的困难。解决方法包括使用 **Linux AppImage** 二进制文件。
- **NotebookLM 用户要求更好的语言支持**：**NotebookLM** 中的多语言支持问题导致生成的摘要使用了非预期的语言。用户呼吁提供更直观的界面来直接管理语言偏好。

---

**主题 5. AI 优化成为焦点：速度与效率**

- **Speculative Decoding 承诺更快的 AI 输出**：关于 **Speculative Decoding** 的讨论强调了一种方法，即由较小的模型生成草稿，再由较大的模型进行精炼，从而缩短推理时间。虽然速度有所提升，但关于输出质量的问题依然存在。
- **Python 3.11 将 AI 性能提升 1.25 倍**：得益于静态分配的核心模块和内联函数调用等优化，升级到 [Python 3.11](https://docs.python.org/3/whatsnew/3.11.html#whatsnew311-faster-cpython) 在 Linux 上可获得高达 **1.25 倍** 的加速，在 Windows 上为 **1.12 倍**。
- **OpenAI 的 Predicted Outputs 重写速度脚本**：通过引入 [Predicted Outputs](https://platform.openai.com/docs/guides/latency-optimization#use-predicted-output)，OpenAI 缩短了 **GPT-4** 的响应时间，用户报告在代码重写任务中有显著的速度提升。

---

---

# 第一部分：Discord 高层摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Open Trusted Data Initiative 发布 2 万亿 Token 多语言数据集**：**Open Trusted Data Initiative** 计划于 **11 月 11 日**通过 [Hugging Face](https://huggingface.co/) 发布包含 **2 万亿 token** 的最大多语言数据集。
  
  - 该数据集旨在通过为开发者和研究人员提供广泛的多语言资源，显著增强 **LLM 训练**能力。
- **Computer Vision 模型量化技术**：一位成员正在开发一个专注于 **Computer Vision 模型量化**的项目，旨在通过 **quantization aware training** 和 **post training quantization** 方法在边缘设备上实现更快的推理。
  
  - 该计划强调减少模型权重并理解其对**训练和推理**性能的影响，引起了社区的关注。
- **Microsoft 发布新模型**：社区对 **Microsoft 发布的新模型**感到兴奋，这些模型满足了几位成员的期望。
  
  - 这些模型因解决了特定的功能需求而受到认可，增强了 AI 工程师可用的工具包。
- **AI 模型中的 Speculative Decoding**：关于 **speculative decoding** 的讨论涉及使用较小的模型生成草稿输出，再由较大的模型进行细化，旨在缩短推理时间。
  
  - 虽然这种方法提高了速度，但与使用单个大模型相比，在保持输出质量方面仍存在疑问。
- **在 Chroma Vector Store 中构建 RAG 的挑战**：一位用户尝试使用 **21 份文档**构建 **RAG** 系统，但在 **Chroma vector store** 中存储 **embeddings** 时遇到问题，仅成功保存了 **7 个 embeddings**。
  
  - 社区成员建议检查潜在的 **error** 信息，并查看默认函数参数，以确保文档没有被无意中丢弃。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Opus 的移除引发用户不满**：用户对 **Claude 3 Opus 的移除**表示失望，强调它是他们在 [Anthropic 网站](https://www.anthropic.com/news/claude-3-family)上进行编程和故事创作的首选模型。
  
  - 许多人请求恢复之前的模型或提供替代方案，因为 **Haiku 3.5** 被认为效果较差。
- **Perplexity Pro 增强订阅权益**：关于 **Perplexity Pro 功能**的讨论显示，Pro 订阅者可以通过 [Revolut 推荐](https://revolut.com/referral/?referral-code=ericqfpk!NOV1-24-VR-FR)等合作伙伴关系获得高级模型的使用权。
  
  - 关于 Pro 档位是否包含 **Claude** 访问权限以及移动应用程序最近的更新，仍然存在疑问。
- **Grok 2 与 Claude 3.5 Sonnet 之争**：工程师们正在争论 **Grok 2** 和 **Claude 3.5 Sonnet** 哪个模型在复杂研究和数据分析方面表现更优。
  
  - **Perplexity** 在学术场景下的优势受到称赞，而像 GPT-4o 这样的模型在编程和创意任务中表现出色。
- **Nvidia 通过战略市场举措瞄准 Intel**：**Nvidia** 正在进行战略布局，直接与 **Intel** 竞争，旨在改变市场动态并影响产品策略。
  
  - 分析师建议关注 Nvidia 即将开展的合作和发布的产品，这些可能会对技术格局产生重大影响。
- **分子神经形态平台取得突破**：一种新型 **molecular neuromorphic platform** 模仿人类大脑功能，代表了 AI 和神经科学研究的重大进展。
  
  - 专家对该平台在深化我们对人类认知的理解以及增强 AI 开发方面的潜力表示*审慎乐观*。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Anthropic 推出 Claude 3.5 Haiku**：Anthropic 已发布 **Claude 3.5** 的标准版和自我审查版，更多带有日期的版本选项可在[此处](https://openrouter.ai/anthropic/claude-3-5-haiku)查看。
  
  - 用户渴望评估该模型在实际应用中的表现，期待其在**速度**、**编码准确性**和**工具集成**方面的提升。
- **免费 Llama 3.2 模型开放访问**：**Llama 3.2** 模型（包括 **11B** 和 **90B**）现在提供免费的快速端点，分别达到 **280tps** 和 **900tps**，[详见此处](https://openrouter.ai/meta-llama/llama-3.2-90b-vision-instruct:free)。
  
  - 该举措旨在通过免费提供高吞吐量选项，增强社区对开源模型的参与度。
- **聊天室新增 PDF 分析功能**：新功能允许用户在聊天室中上传或附加 **PDF**，并使用 OpenRouter 上的任何模型进行分析。
  
  - 此外，最高购买限额已提高至 **$10,000**，为用户提供了更大的灵活性。
- **预测输出（Predicted Output）功能降低延迟**：OpenAI 的 **GPT-4** 模型现已支持**预测输出**，通过 `prediction` 属性优化编辑和重写任务。
  
  - 示例代码片段展示了其在更高效处理大量文本请求中的应用。
- **Hermes 405B 表现不稳定**：免费版的 **Hermes 405B** 表现一直不稳定，用户报告存在间歇性功能故障。
  
  - 许多用户仍持乐观态度，认为这些性能问题预示着正在进行更新或修复。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.62.0 发布**：Aider **v0.62.0** 现在全面支持 **Claude 3.5 Haiku**，在[代码编辑排行榜](https://aider.chat/docs/leaderboards/)上取得了 **75%** 的分数。此版本支持无缝编辑来自 ChatGPT 等网页版 LLM 的文件。
  
  - 此外，**Aider** 生成了此版本中 **84%** 的代码，展示了显著的效率提升。
- **Claude 3.5 Haiku 对比 Sonnet**：**Claude 3.5 Haiku** 提供了与 **Sonnet** 几乎相同的性能，但更具成本效益。用户可以通过使用 `--haiku` 命令行选项来激活 Haiku。
  
  - 这种高性价比使 Haiku 成为许多人 AI 编码工作流中的首选。
- **AI 编码模型对比**：用户分析了 AI 编码模型之间的性能差异，指出 **3.5 Haiku** 与 **Sonnet 3.5** 和 **GPT-4o** 相比效果略逊一筹。
  
  - 市场对即将推出的 **4.5o** 等模型充满期待，这些模型可能会打破现状并影响 **Anthropic** 的市场地位。
- **预测输出功能的影响**：正如 [OpenAI Developers 的推文](https://x.com/OpenAIDevs/status/1853564730872607229)所述，**OpenAI Predicted Outputs** 的推出预计将通过降低延迟和提高代码编辑效率来彻底改变 **GPT-4o** 模型。
  
  - 该功能预计将显著影响模型基准测试，尤其是与竞争模型直接对比时。
- **使用 Claude Haiku 作为编辑器模型**：**Claude 3 Haiku** 被用作编辑器模型，以弥补主模型编辑能力的不足，从而增强开发过程。
  
  - 这种方法对于需要精确语法管理的编程语言特别有益。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **积极性驱动成功的读书小组**：一位成员强调，成功运行**读书小组 (reading groups)** 更多地依赖于**积极性 (initiative)** 而非**专业知识 (expertise)**。他在没有先验知识的情况下发起了 mech interp 读书小组，并始终坚持维护。
  
  - 这种方法强调了在维持有效的学习会议中，主动领导和社区参与的重要性。
- **通过高级设置优化训练**：参与者讨论了各种**优化器设置**（如 beta1 和 beta2）的影响，以及它们在模型训练期间与 **FSDP** 和 **PP** 等策略的兼容性。
  
  - 不同的观点突显了训练效率与模型性能之间的平衡。
- **增强 Logits 和概率优化**：深入讨论了**优化 logits 输出**以及确定训练所需的适当数学范数，建议使用 **L-inf norm** 来最大化概率，或通过 **KL divergence** 保持分布形状。
  
  - 参与者探索了微调模型输出的方法，以提高预测准确性和稳定性。
- **LLM 鲁棒性评估 PR 增强框架**：一位成员宣布针对三个不同数据集开启了 **LLM Robustness Evaluation** 的 **PR**，并邀请反馈和评论，可在[此处](https://github.com/EleutherAI/lm-evaluation-harness/pull/2452)查看。
  
  - 该 PR 为大语言模型引入了系统的连贯性和鲁棒性评估，同时解决了之前的 bug。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Python 3.11 在 Linux 上提升 1.25 倍性能**：鼓励用户切换到 [Python 3.11](https://docs.python.org/3/whatsnew/3.11.html#whatsnew311-faster-cpython)，因为它通过各种优化在 Linux 上提供高达 **1.25 倍的加速**，在 Windows 上提供 **1.12 倍的加速**。
  
  - **核心模块**被静态分配以加快加载速度，函数调用现在改为内联 (inlined)，从而增强了整体性能。
- **llama.cpp 支持 Qwen 2.5，即将集成 Vision 模型**：讨论确认 *llama.cpp* 已支持 **Qwen 2.5**，详见 [Qwen 文档](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)。
  
  - 社区正期待在 *Unsloth* 中集成 vision 模型，预计很快就会推出。
- **在有限数据集上微调 LLM**：用户正在探索仅使用 **10 个示例**（总计 60,000 字）微调模型的可行性，专门用于标点符号纠正。
  
  - 建议包括使用 batch size 为 1，以缓解与有限数据相关的挑战。
- **使用 Hugging Face 指标实现 mtbench 评估**：一位成员询问了在 mtbench 数据集上运行类似 mtbench 评估的回调 (callbacks) 参考实现，并询问是否存在 Hugging Face evaluate 指标。
  
  - 需要精简评估流程，强调了将此类功能集成到当前项目中的重要性。
- **使用 Hugging Face 指标增强 mtbench 评估**：请求关于实现回调以在 mtbench 数据集上运行评估的见解，类似于现有的 mtbench 评估。
  
  - 该询问突显了在正在进行的 AI 工程项目中对高效评估机制的需求。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **便携式 LM Studio 解决方案**：一位用户询问关于从 USB 闪存盘运行 **LM Studio** 的问题，收到了使用 **Linux AppImage 二进制文件**或共享脚本来实现便携性的建议。
  
  - 尽管目前缺乏官方便携版本，社区成员仍提供了变通方案，以促进 **便携式 LM Studio 部署**。
- **LM Studio 服务器日志访问**：用户发现，在 **LM Studio** 中按下 **CTRL+J** 可以打开服务器日志选项卡，从而实现对服务器活动的实时监控。
  
  - 这一快速访问功能被分享出来，旨在帮助成员有效地跟踪和调试服务器性能。
- **模型性能评估：Mistral vs Qwen2**：**Mistral Nemo** 在基于 Vulkan 的操作中表现优于 **Qwen2**，展示了更快的 Token 处理速度。
  
  - 这种性能差异凸显了不同 **模型架构 (model architectures)** 对计算效率的影响。
- **Windows 调度器效率低下**：成员们报告称，**Windows Scheduler** 在多核设置中难以处理 **CPU 线程管理**，从而影响了性能。
  
  - 一位成员建议手动为进程设置 **CPU 亲和性 (affinity)** 和 **优先级 (priority)**，以减轻调度问题。
- **LLM 上下文管理挑战**：**上下文长度 (Context length)** 显著影响 **LLM** 的 **推理速度**，一位用户指出，在大上下文情况下，首个 Token 的延迟达到了 **39 分钟**。
  
  - 建议在启动新对话时优化 **上下文填充水平 (context fill levels)**，以提高 **推理响应速度**。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Hume App 发布，融合 EVI 2 与 Claude 3.5**：全新的 [Hume App](https://x.com/hume_ai/status/1853540362599719025?s=46) 结合了由 **EVI 2** 语音语言模型生成的语音和个性，以及 **Claude 3.5 Haiku**，旨在通过 AI 生成的助手增强用户交互。
  
  - 正如[官方公告](https://x.com/hume_ai/status/1853540362599719025?s=46)所强调的，用户现在可以访问这些助手进行更具动态性的互动。
- **OpenAI 通过 Predicted Outputs 降低 GPT-4 延迟**：[OpenAI](https://x.com/OpenAIDevs/status/1853564730872607229) 推出了 **Predicted Outputs**，通过提供参考字符串来加快处理速度，显著降低了 **GPT-4o** 和 **GPT-4o-mini** 模型的延迟。
  
  - 正如 [Eddie Aftandilian](https://x.com/eaftandilian/status/1853576254005583985) 所指出的，基准测试显示在文档迭代和代码重写等任务中速度有所提升。
- **Supermemory AI 工具管理你的数字大脑**：一位 19 岁的开发者推出了 [**Supermemory**](https://github.com/supermemoryai/supermemory)，这是一个旨在管理书签、推文和笔记的 AI 工具，功能类似于针对已保存内容的 ChatGPT。
  
  - 正如 [Dhravya Shah](https://x.com/dhravyashah/status/1853637539053113758?s=46) 所展示的，通过聊天机器人界面，用户可以轻松检索和探索之前保存的内容。
- **腾讯发布巨型 Hunyuan-Large 模型**：**腾讯**发布了 **Hunyuan-Large** 模型，这是一个基于 Transformer 的开源权重混合专家模型 (mixture of experts)，拥有 **3890 亿参数**和 **520 亿激活参数**。
  
  - 尽管被标记为开源，但关于其地位的争论依然存在，且其庞大的体积对大多数基础设施公司构成了挑战，详见 [Hunyuan-Large 论文](https://arxiv.org/abs//2411.02265)。
- **Defense Llama：用于国家安全的 AI**：**Scale AI** 宣布了 **Defense Llama**，这是一个与 **Meta** 及国防专家合作开发的专用 **LLM**，针对美国国家安全应用。
  
  - 根据 [Alexandr Wang](https://x.com/alexandr_wang/status/1853853829336559790) 的说法，该模型现在可集成到美国国防系统中，突显了 AI 在安全领域的进步。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 扩展集成能力**：成员们讨论了 **NotebookLM** 集成多个 notebook 或来源的潜力，旨在增强其在学术研究中的功能。目前每个 notebook 限制为 **50 个来源**是一个主要关注点，参考见 [NotebookLM Features](https://discord.com/channels/1124402182171672732/1124402182909857966/1303090885025726546)。
  
  - 社区对支持跨 notebook 数据共享的功能增强表现出浓厚兴趣，反映出用户对改进协作工具和更清晰的开发路线图的渴望。
- **Deepfake 技术引发伦理问题**：一位用户强调了在除臭剂广告中使用的 **Face Swap** 技术，指出了 **deepfake** 技术在营销活动中的应用。这在 [Deepfake Technology](https://discord.com/channels/1124402182171672732/1124403655819415592/1303136711462748210) 的背景下得到了进一步讨论。
  
  - 另一位参与者强调，deepfakes 本质上涉及面部交换，这促进了对伦理影响的共同理解以及对此类技术负责任使用的必要性。
- **使用 NotebookLM 管理供应商数据**：一位企业主探索使用 **NotebookLM** 来管理约 **1,500 家供应商**的数据，利用了包括 pitch decks 在内的各种来源。他们提到有一个数据团队准备协助导入，详见 [Vendor Database Management Use Cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1303136711462748210)。
  
  - 讨论中提出了对跨 notebook 数据共享的担忧，强调了对强大数据管理功能的需求，以确保大型数据集的安全性和可访问性。
- **NotebookLM 中的音频播客生成**：**NotebookLM** 推出了音频播客生成功能，成员们因其在多任务处理中的便利性而给予了积极评价。用户询问了有效利用该功能的策略，如 [Audio Podcast Generation Features](https://discord.com/channels/1124402182909857966/1124402182171672732/1303090885025726546) 中所述。
  
  - 社区对播客功能表现出极大的热情，提出了潜在的使用案例，并征求最佳实践，以在各种工作流中最大化其效益。
- **NotebookLM 语言支持的挑战**：几位成员报告了 **NotebookLM** 的 **multilingual support**（多语言支持）问题，即尽管设置配置为英语，但生成的摘要却是其他语言。这是 [Language and Localization Issues](https://discord.com/channels/1124402182909857966/1124402182171672732/1303090885025726546) 中的主要话题。
  
  - 成员们建议改进用户界面以更好地管理语言偏好，强调需要一个更直观的过程来直接更改语言设置。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SWarmUI 简化 ComfyUI 设置**：成员们建议安装 [SWarmUI](https://github.com/mcmonkeyprojects/SwarmUI) 来简化 **ComfyUI** 的部署，并强调了其管理复杂配置的能力。
  
  - 一位成员强调道：*"它的设计初衷就是为了让你的生活变得更轻松。"* 这展现了社区对用户友好型界面的认可。
- **云端托管 Stable Diffusion 的挑战**：讨论显示，与本地设置相比，在 **Google Cloud** 上托管 **Stable Diffusion** 可能更加复杂且昂贵。
  
  - 参与者建议使用 [vast.ai](https://vast.ai) 等 GPU 租赁平台作为部署模型的高性价比且更简单的替代方案。
- **Civitai 上的最新模型和 LoRas**：用户探索了从 **Civitai** 下载 **1.5**、**SDXL** 和 **3.5** 等最新模型，并指出大多数 **LoRas** 都是基于 **1.5** 的。
  
  - 像 **v1.4** 这样的旧版本被认为已经过时，社区建议升级以从增强的功能和性能中受益。
- **分享 Animatediff 教程资源**：一位成员请求 **Animatediff** 的教程，并收到了参考 [Purz 的 YouTube 频道](https://youtu.be/oNpOf9sYvKY) 资源的建议。
  
  - 社区对分享知识表现出极大的热情，加强了围绕动画工具的协作学习环境。
- **ComfyUI 现在通过 GenMo 的 Mochi 支持视频 AI**：确认 **ComfyUI** 已通过 [GenMo's Mochi](https://github.com/mcmonkeyprojects/SwarmUI) 集成了视频 AI 功能，尽管这需要相当高的硬件配置。
  
  - 这一集成被视为一项重大进步，有可能利用 **Stable Diffusion** 技术拓展视频生成的视野。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 2.5 数据集的 'weight' 字段受到质疑**：成员们分析了 **Hermes 2.5** 数据集的 'weight' 字段，发现其贡献微乎其微，并导致大量 **空字段**。
  
  - 有推测认为，优化数据集采样可能会提高其对小型 LLM 的效用。
- **Nous Research 确认 Hermes 系列保持开源**：针对有关 **闭源 LLM** 的询问，**Nous Research** 确认 **Hermes 系列** 将继续保持 **开源**。
  
  - 虽然未来的一些项目可能会采用封闭模式，但对 **Hermes 系列** 的开源承诺依然存在。
- **在未来 AI 模型中平衡质量与数量**：讨论强调了 **高质量数据集** 对开发未来 AI 模型的重要性。
  
  - 有人担心优先考虑质量可能会排除有价值的主题和事实，尽管这可能会增强 **常识推理 (commonsense reasoning)**。
- **引入 OmniParser 以增强数据解析**：分享了 [OmniParser](https://huggingface.co/spaces/jadechoghari/OmniParser) 工具，该工具以提高 **数据解析** 能力而闻名。
  
  - 其 **创新方法** 已引起 AI 社区的关注。
- **Hertz-Dev 发布全双工对话音频模型**：[Hertz-Dev GitHub 仓库](https://github.com/Standard-Intelligence/hertz-dev) 推出了首个用于 **全双工对话音频** 的基座模型。
  
  - 该模型旨在促进在单一框架内的 **speech-to-speech** 交互，增强 **音频通信**。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **NeurIPS 赞助推进**：一位成员宣布他们正在努力为 [NeurIPS](https://discord.com/channels/1179127597926469703/1179127598442348729/1303208070846873640) 寻找 **赞助商**，这标志着潜在的合作机会。
  
  - 他们还发出了 **NeurIPS** 团体晚宴的邀请，旨在加强会议期间参会者之间的社交。
- **腾讯发布 389B MoE 模型**：[腾讯](https://github.com/Tencent/Tencent-Hunyuan-Large) 发布了其 389B **Mixture of Experts (MoE)** 模型，在 AI 社区引起了重大反响。
  
  - 讨论强调，该模型的先进功能可能会为大规模模型性能设定新基准，详见其 [论文](https://arxiv.org/abs/2411.02265)。
- **Scale AI 推出 Defense Llama**：**Scale AI** 介绍了 **Defense Llama**，这是一个专为 **机密网络** 内的军事应用设计的专用 LLM，正如 [DefenseScoop](https://defensescoop.com/2024/11/04/scale-ai-unveils-defense-llama-large-language-model-llm-national-security-users/) 所报道。
  
  - 该模型旨在支持作战规划等行动，标志着将 AI 整合到国家安全框架中的举措。
- **YOLOv3 论文获得高度推荐**：一位成员强调了 [YOLOv3 论文](https://x.com/vikhyatk/status/1853266606291575264) 的重要性，称其为从业者的必读内容。
  
  - 他们评论道：*“顺便说一下，如果你还没读过 YOLOv3 的论文，那你真的错过了”*，强调了其在该领域的相关性。
- **LLM 性能漂移调查**：围绕创建一个系统或论文来 **微调小型 LLM 或分类器** 以监控写作等任务中的模型性能漂移展开了讨论。
  
  - 成员们辩论了现有 **提示词分类器 (prompt classifiers)** 在准确跟踪漂移方面的有效性，强调了对稳健 **评估流水线 (evaluation pipelines)** 的需求。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o 推送引入了类 o1 推理能力**：**GPT-4o** 的推送引入了类 **o1** 的推理能力，并在一个 **canvas** 风格的文本框中包含大段文本。
  
  - 社区成员对于这次推送是针对**常规 GPT-4o** 的 **A/B test**，还是针对特定用途的专门版本存在困惑。
- **OpenAI 对安全 AGI 开发的承诺**：一位成员强调，OpenAI 成立的目标是构建**安全且有益的 AGI**，这是自 2015 年成立以来就宣布的使命。
  
  - 讨论中还涉及了一些担忧，即如果 AI 开发成本超过了所有人类投资，可能会导致 AI 自我开发（**AI self-development**），从而产生重大影响。
- **GPT-5 发布日期尚不确定**：社区成员对 **GPT-5** 及其配套 **API** 的发布感到好奇，但也承认确切的时间表尚不清楚。
  
  - 一位成员表示：*“今年应该会有一些新发布，但不会是 GPT-5。”*
- **Premium 账户账单问题**：一位用户报告了其 **Premium account** 账单出现问题，指出尽管有来自 **Apple** 的付款证明，但其账户仍显示为免费计划。
  
  - 另一位成员尝试通过分享链接进行协助，但问题仍未解决。
- **文档摘要中的 Hallucinations 问题**：成员们对文档摘要过程中出现的 **hallucinations**（幻觉）表示担忧，尤其是在生产环境中扩展工作流时。
  
  - 为了减少不准确性，一位成员建议实施第二次 **LLM pass** 进行事实核查。

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex chat-ui 集成**：开发者可以使用 [LlamaIndex chat-ui](https://t.co/ZLGgPWjDHD) 快速为他们的 **LLM** 应用创建聊天界面，该工具提供预构建组件和 **Tailwind CSS** 定制功能，并能与 **Vercel AI** 等 **LLM** 后端无缝集成。
  
  - 这种集成简化了聊天功能的实现，提高了 AI 工程师开发对话界面的效率。
- **高级报告生成技术**：一篇新的[博客文章和视频](https://t.co/3KnoSykdhR)探讨了高级报告生成，包括结构化输出定义和高级文档处理，这对于优化企业报告工作流至关重要。
  
  - 这些资源为 AI 工程师提供了关于增强 **LLM** 应用中报告生成能力的更深层见解。
- **NVIDIA 竞赛提交截止日期**：**NVIDIA competition** 的提交截止日期为 11 月 10 日，通过[此链接](https://t.co/rtMpetSyu1)提交的项目有机会获得 **NVIDIA® GeForce RTX™ 4080 SUPER GPU** 等奖品。
  
  - 鼓励开发者利用 **LlamaIndex** 技术创建创新的 **LLM** 应用以赢取奖励。
- **LlamaParse 功能与数据保留**：**LlamaParse** 是一款闭源解析工具，可将文档高效转换为结构化数据，并提供 **48 小时数据保留政策**，详情参阅 [LlamaParse 文档](https://www.llamaindex.ai/llamaparse)。
  
  - 讨论强调了其性能优势以及数据保留对重复任务处理的影响，并引用了[入门指南](https://docs.cloud.llamaindex.ai/llamaparse/getting_started)。
- **与 Cohere 的 ColiPali 进行多模态集成**：一个正在进行的 **PR** 旨在将 **ColiPali** 作为 **reranker** 添加到 **LlamaIndex** 中，尽管由于多向量索引（**multi-vector indexing**）的要求，将其作为 **indexer** 集成仍具有挑战性。
  
  - 社区正积极致力于扩展 **LlamaIndex** 的多模态数据处理能力，突显了与 **Cohere** 的协作努力。

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Connectors 问题**：成员们报告在使用 **Coral web interface** 或 API 时，**connectors** 无法正常工作，导致来自 [reqres.in](https://reqres.in/) 的结果为零。
  
  - 一位用户指出，**connectors** 的响应时间比预期长，响应时间超过了 **30 秒**。
- **Cohere API 微调与错误**：微调 **Cohere API** 需要输入卡片详情并切换到生产密钥，用户需要为 SQL 生成准备合适的 prompt 和 response 示例。
  
  - 此外，一些成员报告称，尽管在 **playground** 环境中操作成功，但在通过 API 运行微调后的 classify 模型时遇到了 **500 错误**。
- **在 Wordpress 上开发 Prompt Tuner**：一位用户询问如何使用 API 在 Wordpress 网站上重建 **Cohere prompt tuner**。
  
  - 另一位成员建议开发自定义后端应用程序，并指出 Wordpress 可以支持此类集成。参考 [Login | Cohere](https://dashboard.cohere.com/prompt-tuner) 以获取高级 LLM 和 NLP 工具。
- **软件测试中的 Embedding 模型**：成员们讨论了 **embed model** 在软件测试任务中的应用，以增强测试流程。
  
  - 寻求关于 embedding 如何具体协助这些测试任务的澄清。
- **GCP Marketplace 计费关注**：一位用户提出了在通过 **GCP Marketplace** 激活 **Cohere** 并获得 API 密钥后的计费流程问题。
  
  - 他们寻求关于费用是扣除在 GCP 账户还是注册卡上的澄清，并表示更倾向于针对特定模型的计费方式。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Microsoft Omniparser 集成**：一位成员询问关于集成 **Microsoft Omniparser** 的事宜，强调了其对开源模式的潜在益处。另一位成员确认他们正在**积极探索**这一集成。
  
  - 讨论强调了利用 **Omniparser 的能力**来提高系统的解析效率。
- **Claude's Computer Use 集成**：成员们讨论了在当前的 `--os` 模式中集成 **Claude's Computer Use**，并确认已经完成整合。对话强调了对使用**实时预览**以改进功能的兴趣。
  
  - 参与者对无缝集成表示热烈欢迎，指出**实时预览**可以显著提升用户体验。
- **Agent 标准**：一位成员提议为 **Agent** 创建一个标准，理由是 **LMC** 的设置比 **Claude 的界面**更简洁。他们建议 **OpenInterpreter (OI)** 与 **Anthropic** 合作，建立一个与 **OAI endpoints** 兼容的通用标准。
  
  - 小组讨论了统一标准的可行性，并考虑了与现有 **OAI endpoints** 的兼容性要求。
- **OpenInterpreter 中的 Haiku 性能**：一位成员询问了新版 **Haiku** 在 **OpenInterpreter** 中的性能，并提到他们尚未进行测试。这反映了社区对评估最新工具的持续兴趣。
  
  - 大家一致认为，测试 **Haiku 性能**对于评估其在各种工作流中的有效性和适用性至关重要。
- **Tool Use 软件包增强**：`Tool Use` 软件包已更新两个新的免费工具：**ai prioritize** 和 **ai log**，可以通过 `pip install tool-use-ai` 安装。这些工具旨在简化工作流并提高生产力。
  
  - 鼓励社区成员向 **Tool Use** [GitHub 仓库](https://github.com/ToolUse/tool-use-ai)贡献代码，该仓库包含详细文档并邀请持续改进 AI 工具。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **提醒：11月12日的 Modular 社区 Q&A**：发出提醒，请为定于 **11月12日** 举行的 [Modular 社区 Q&A](https://forms.gle/t6bQnPx6n2caSipU8) 提交问题，可选择是否署名。
  - 鼓励成员通过 [提交表单](https://forms.gle/t6bQnPx6n2caSipU8) 分享他们的疑问，以参与即将举行的 **社区会议**。
- **社区会议项目和演讲征集**：邀请成员在 **Modular 社区 Q&A** 期间展示项目、发表演讲或提出想法。
  - 这一邀请旨在促进社区参与，并允许在 **11月12日的会议** 上展示贡献。
- **在 Mojo 中实现 Effect System**：关于在 Mojo 中集成 **effect system** 的讨论集中在将执行 syscalls 的函数标记为 block，默认情况下可能作为警告。
  - 建议包括引入 'panic' effect，以便在 **Mojo** 语言中对敏感上下文进行静态管理。
- **解决 Mojo 中的矩阵乘法错误**：一位用户报告了其矩阵乘法实现中的多个错误，包括 **Mojo** 中 `memset_zero` 和 `rand` 函数调用的问题。
  - 这些错误突显了与函数定义中的隐式转换和参数规范相关的问题。
- **优化 Matmul Kernel 性能**：一位用户注意到他们的 **Mojo** matmul kernel 比 **C** 版本慢两倍，尽管使用了类似的向量指令。
  - 目前正在考虑优化方案以及 bounds checking 对性能的影响。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **发布新的选举候选人研究工具**：一位成员介绍了 [Election Candidate Research Tool](https://github.com/tkellogg/election2024)，旨在选举前简化候选人研究，强调了其易用的特性和预期功能。
  - 该 [GitHub 仓库](https://github.com/tkellogg/election2024) 鼓励社区贡献，旨在通过协作开发提升选民的研究体验。
- **使用 BootstrapFewShot 优化 Few-Shot**：成员们探索了使用 **BootstrapFewShot** 和 **BootstrapFewShotWithRandomSearch** 优化器在不修改现有 prompts 的情况下增强 few-shot 示例，提升了示例组合的灵活性。
  - 这些优化器在保留主要指令内容的同时，提供了多样的 few-shot 示例组合，有助于提高 few-shot 学习性能。
- **庆祝 VLM 支持性能**：一位成员赞扬了团队在 **VLM 支持** 方面所做的努力，认可其有效性以及对项目性能指标的积极影响。
  - 他们的认可强调了项目中 VLM 支持的成功实现和增强。
- **DSPy 2.5.16 在处理长输入时遇到困难**：关于 **DSPy 2.5.16** 使用 **Ollama 后端** 的担忧出现，长输入会导致输入和输出字段混淆，从而产生错误输出，这表明可能存在 Bug。
  - 一个 SQL 提取示例展示了长输入如何导致预测中出现意外的占位符，指向了输入/输出解析中的问题。
- **即将进行的 DSPy 版本测试**：一位成员计划测试最新的 **DSPy** 版本，不再使用 conda 分发的版本，以调查长输入处理问题。
  - 他们打算在测试后报告发现，表明正在持续努力解决 **DSPy** 中的解析问题。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **LLM 的分布式训练**：一位成员发起了一项讨论，关于使用其大学新的 GPU 集群进行 LLM 的**分布式训练**，重点是从零开始训练模型。
  
  - 另一位成员建议为**分布式训练**和**预训练**提供资源，以协助其研究项目。
- **Kubernetes 用于容错**：有人提议实现 **Kubernetes 集群**，以增强 GPU 系统的容错能力。
  
  - 成员们讨论了将 **Kubernetes** 与 Axolotl 集成的好处，以便更好地管理分布式训练任务。
- **Meta Llama 3.1 模型**：**Meta Llama 3.1** 被强调为一个极具竞争力的开源模型，并提供了使用 Axolotl 进行微调和训练的资源。
  
  - 鼓励成员们查看一份[微调教程](https://axolotlai.substack.com/p/fine-tuning-llama-31b-waxolotl-on)，该教程详细介绍了如何在多节点上使用该模型。
- **StreamingDataset PR**：一位成员回想起关于 **StreamingDataset PR** 的讨论，询问是否仍有人对此感兴趣。
  
  - 这表明关于云集成和数据集处理的讨论和开发正在持续进行。
- **Firefly 模型**：**Firefly** 是 **Mistral Small 22B** 的微调版本，专为创意写作和角色扮演设计，支持高达 **32,768 tokens** 的上下文。
  
  - 警告用户该模型可能会生成**露骨、令人不安**或**冒犯性**的回复，应负责任地使用。建议用户在进行任何访问或下载前[在此查看内容](https://huggingface.co/invisietch/MiS-Firefly-v0.1-22B?not-for-all-audiences=true)。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **DistiLLM 优化教师概率**：讨论集中在 **DistiLLM** 交叉熵优化中*减去教师概率*的问题，详见 [GitHub issue](https://github.com/jongwooko/distillm/issues/7)。讨论强调由于教师模型保持冻结状态，常数项可以忽略。
  
  - 建议更新 docstring，以澄清损失函数假设教师模型是冻结的。
- **KD-div 与交叉熵的澄清**：有人担心在实际返回值为交叉熵时将其标记为 **KD-div**，这在与 **KL-div** 等损失进行比较时可能会引起混淆。
  
  - *据指出，将此过程定义为优化交叉熵*，能更好地契合从训练中的硬标签到教师模型产生的软标签的转变。
- **TPO 势头强劲**：一位成员对 **TPO** 表达了热情，称其令人印象深刻，并计划集成一个追踪器。
  
  - 成员们对 TPO 的功能及其潜在应用充满期待。
- **VinePPO 实现挑战**：虽然欣赏 **VinePPO** 在推理和对齐方面的优势，但一位成员警告说，其实现可能会带来重大挑战。
  
  - 强调了部署 VinePPO 的潜在困难，并指出了与其集成相关的风险。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **TokenFormer 与 tinygrad 的集成**：一位成员成功将 **TokenFormer** 的最小化实现移植到了 **tinygrad**，可在 [GitHub 仓库](https://github.com/kroggen/tokenformer-minimal/tree/tinygrad)中找到。
  
  - 此次适配旨在增强 tinygrad 内部的**推理和学习**能力，展示了集成先进模型架构的潜力。
- **View 中的依赖解析**：一位用户询问操作 `x[0:1] += x[0:1]` 是取决于 `x[2:3] -= ones((2,))` 还是仅取决于 `x[0:1] += ones((2,))`，这涉及到真假共享规则。
  
  - 该讨论提出了关于 tinygrad 内部操作序列中依赖关系如何追踪的技术思考。
- **为加速器开发进行的 Hailo 逆向工程**：一位成员宣布开始 **Hailo** 逆向工程工作，以创建一种新的加速器，重点关注过程效率。
  
  - 他们对内核编译过程表示担忧，该过程必须在执行前将 **ONNX** 以及即将支持的 **Tinygrad** 或 **TensorFlow** 编译为 **Hailo**。
- **tinygrad 融合中的内核一致性**：一位用户正在调查在执行 `BEAM=2` 融合时，**tinygrad** 中的内核在多次运行中是否保持一致。
  
  - 他们旨在通过强调有效缓存管理的必要性，来防止重新编译相同内核带来的开销。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **关于 Project GR00T 的第 9 讲**：今天 LLM Agents MOOC 的**第 9 讲**定于 [下午 3:00 PST](https://www.youtube.com/live/Qhxr0uVT2zs) 进行直播，届时 **Jim Fan** 将讨论 **Project GR00T**，这是 NVIDIA 的通用机器人计划。
  
  - Jim Fan 在 **GEAR** 内部的团队正在开发能够在模拟和现实环境中运行的 AI Agent，重点在于增强通用能力。
- **Jim Fan 博士简介**：**Jim Fan 博士**是 NVIDIA **GEAR** 的研究负责人，拥有斯坦福视觉实验室（Stanford Vision Lab）博士学位，并获得了 **NeurIPS 2022 优秀论文奖**。
  
  - 他在机器人多模态模型和精通 Minecraft 的 AI Agent 方面的工作曾被《纽约时报》、**Forbes** 和《麻省理工科技评论》（**MIT Technology Review**）等主流媒体报道。
- **LLM Agents 课程资源**：所有**课程材料**，包括[直播链接](http://llmagents-learning.org/f24)和家庭作业，均可在网上获取。
  
  - 鼓励学生在专门的[课程频道](https://discord.com/channels/1280234300012494859/1280370030609170494)中提问。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **FOSDEM 2025 DevRoom 开放**：Mozilla 将于 **2025 年 2 月 1 日至 2 日**在**布鲁塞尔**举办的 [FOSDEM 2025](https://pretalx.fosdem.org/fosdem-2025/cfp) 中设立 **DevRoom**，重点关注开源演示。
  
  - 演讲提案**提交**截止日期为 **2024 年 12 月 1 日**，录取通知将于 **12 月 15 日**发出。
- **演讲提案截止日期临近**：参与者需在 **2024 年 12 月 1 日**之前为 **FOSDEM 2025 DevRoom** **提交演讲提案**。
  
  - 获选讲者将于 **12 月 15 日**收到通知，以确保有充足的准备时间。
- **FOSDEM 志愿者招募**：[FOSDEM 2025 志愿者公开招募](https://discourse.mozilla.org/t/call-for-volunteers-fosdem-2025-in-brussels-belgium-1-2-february-2025/136830)已发布，并为欧洲参与者提供差旅赞助。
  
  - 志愿服务提供了在活动中建立人脉和支持开源社区的机会。
- **FOSDEM 演讲主题多样化**：**FOSDEM 2025** 演示的建议主题包括 **Mozilla AI**、**Firefox 创新**以及**隐私与安全**等。
  
  - 鼓励讲者探索这些领域之外的内容，演讲时长从 **15 到 45 分钟**不等，包含问答环节。
- **提案准备资源发布**：Mozilla 分享了一份关于如何创建成功提案的技巧资源，可在此处访问 [here](https://discourse.mozilla.org/t/call-for-talks-fosdem-2025-in-brussels-belgium-1-2-february-2025/136829)。
  
  - 该指南旨在帮助潜在讲者在 **FOSDEM 2025** 上制作出有影响力的演示。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **基准测试基于检索的函数调用**：一位成员正在对函数调用的**基于检索的方法进行基准测试**，并正在寻求一系列可用的[函数及其定义](https://discord.com/channels/1111172801899012102/1111353033352294440/1303139972945018990)。
  
  - 他们特别要求按**测试类别**组织这些定义，以便进行更有效的索引。
- **函数定义索引讨论**：一位成员强调了需要**索引化的函数定义集合**，以增强基准测试工作。
  
  - 他们强调了按**测试类别**对这些函数进行分类的重要性，以简化工作流程。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长时间没有更新，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间没有更新，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有更新，请告知我们，我们将将其移除。

---

**LAION Discord** 没有新消息。如果该频道长时间没有更新，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有更新，请告知我们，我们将将其移除。

---

# 第 2 部分：详细频道摘要和链接

{% if medium == 'web' %}

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1303091324303446026) (1094 条消息🔥🔥🔥):

> - `AI 模型集成`
> - `LLM 中的 Temperature 设置`
> - `声子与材料科学`
> - `投机采样 (Speculative Decoding)`
> - `数字民族志研究`

- **将 Hugging Face 集成到 Discord**：用户讨论了将 Hugging Face 功能集成到 Discord 服务器的方法，探索了嵌入 HF 模型或创建用户等级验证系统的可能性。
  
  - 建议包括使用等级机器人进行用户验证，作为一种潜在的解决方案。
- **理解模型中的 Temperature 设置**：聊天参与者深入探讨了 LLM 中 Temperature 设置的重要性，强调较高的 Temperature 会导致模型输出的随机性和变异性增加。
  
  - 他们指出，虽然这可以增强创造力，但必须经过仔细测试，以避免响应质量下降。
- **声子及其在材料科学中的作用**：关于声子的讨论强调了它们在解释热导率方面的重要性，以及它们与光粒子的相似之处，揭示了对材料特性的见解。
  
  - 对准晶体中声子新研究的引用，说明了物理学与材料科学交叉领域不断发展的认识。
- **AI 中的投机采样 (Speculative Decoding)**：参与者探讨了投机采样的概念，即由较小的模型生成快速草稿输出，再由较大的模型进行精炼以确保准确性，从而缩短推理时间。
  
  - 有人指出，虽然这种方法提高了速度，但与大型单一模型相比，在维持输出质量方面仍存在疑问。
- **数字民族志研究技术**：一位用户表示有兴趣对在线社区进行数字民族志研究，强调需要分析社区动态和用户互动。
  
  - 回复包括关于研究社区规范以及与所选在线群体深入互动的建议。

**提到的链接**：

- [minchyeom/birthday-2 · Hugging Face](https://huggingface.co/minchyeom/birthday-2): 未找到描述
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710): 我们提出了 LayerSkip，这是一种加速大语言模型 (LLM) 推理的端到端解决方案。首先，在训练期间我们应用层丢弃 (layer dropout)，对较早的层使用较低的丢弃率，对较后的层使用较高的...
- [Banishing LLM Hallucinations Requires Rethinking Generalization](https://arxiv.org/abs/2406.17642): 尽管大语言模型 (LLM) 具有强大的聊天、编码和推理能力，但它们经常产生幻觉。传统观点认为，幻觉是某种平衡的结果...
- [Real Monster GIF - Real Monster Scared - Discover & Share GIFs](https://tenor.com/view/real-monster-scared-funny-gif-14723286): 点击查看 GIF
- [Flavor Flav Fight The Power GIF - Flavor Flav Fight The Power Glasses - Discover & Share GIFs](https://tenor.com/view/flavor-flav-fight-the-power-glasses-face-clip-gif-295039939991987958): 点击查看 GIF
- [Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent](https://arxiv.org/abs/2411.02265): 在本文中，我们介绍了 Hunyuan-Large，这是目前最大的开源基于 Transformer 的混合专家 (MoE) 模型，总参数量为 3890 亿，激活参数量为 520 亿...
- [Cat Wait Waiting Cat GIF - Cat wait Waiting cat Wait - Discover & Share GIFs](https://tenor.com/view/cat-wait-waiting-cat-wait-waiting-cat-waiting-gif-9780709586447195996): 点击查看 GIF
- [Chicken Run GIF - Chicken Run Panic - Discover & Share GIFs](https://tenor.com/view/chicken-run-panic-gif-26658158): 点击查看 GIF
- [Introduction - Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/en/chapter1/1): 未找到描述
- [Spongebob Patrick GIF - Spongebob Patrick Patrick Star - Discover & Share GIFs](https://tenor.com/view/spongebob-patrick-patrick-star-broke-poor-gif-14729256): 点击查看 GIF
- [Sponge Bob Squid Ward GIF - Sponge Bob Squid Ward Rich - Discover & Share GIFs](https://tenor.com/view/sponge-bob-squid-ward-rich-money-bath-gif-4971339): 点击查看 GIF
- [Peter Griffin GIF - Peter Griffin Family Guy - Discover & Share GIFs](https://tenor.com/view/peter-griffin-family-guy-gif-13905506): 点击查看 GIF
- [FlyWire](https://flywire.ai): 未找到描述
- [The Universe Tim And Eric Mind Blown GIF - The Universe Tim And Eric Mind Blown Mind Blown Meme - Discover & Share GIFs](https://tenor.com/view/the-universe-tim-and-eric-mind-blown-mind-blown-meme-mind-explosion-mind-explosion-meme-gif-18002878): 点击查看 GIF
- [Family Guy Peter Griffin GIF - Family Guy Peter Griffin I Have Spoken - Discover & Share GIFs](https://tenor.com/view/family-guy-peter-griffin-i-have-spoken-i-said-what-i-said-gif-21564051): 点击查看 GIF
- [Sips Tea The Boys GIF - Sips Tea The Boys Smile - Discover & Share GIFs](https://tenor.com/view/sips-tea-the-boys-smile-gif-18692019): 点击查看 GIF

- [Kanye West Stare GIF - Kanye West Stare Serious - Discover & Share GIFs](https://tenor.com/view/kanye-west-stare-serious-gif-15710427): 点击查看 GIF
- [David Warner Tron Sark GIF - David Warner Tron Sark David Warner - Discover & Share GIFs](https://tenor.com/view/david-warner-tron-sark-david-warner-gif-18249140): 点击查看 GIF
- [tencent/Tencent-Hunyuan-Large · Hugging Face](https://huggingface.co/tencent/Tencent-Hunyuan-Large): 未找到描述
- [Hugging Face for Excel](https://appsource.microsoft.com/ja-jp/product/office/WA200007352): 通过 Excel 自定义函数免费使用 Hugging Face 上的推理模型和 Spaces。
- [South Park GIF - South Park Moses - Discover & Share GIFs](https://tenor.com/view/south-park-moses-gif-18905790): 点击查看 GIF
- [The Deep Deep Thoughts GIF - The Deep Deep Thoughts Deep Thoughts With The Deep - Discover & Share GIFs](https://tenor.com/view/the-deep-deep-thoughts-deep-thoughts-with-the-deep-the-boys-gif-26372785): 点击查看 GIF
- [Zano (drone) - Wikipedia](https://en.wikipedia.org/wiki/Zano_(drone)): 未找到描述
- [Sigh Homelander GIF - Sigh Homelander The boys - Discover & Share GIFs](https://tenor.com/view/sigh-homelander-the-boys-exhale-relieved-gif-15406600715060657123): 点击查看 GIF
- [the answer to life, universe and everything is .. 42](https://www.youtube.com/watch?v=SmanVIJ80EY): 未找到描述
- [Go Ahead I'M All Ears GIF - Go ahead i'm all ears - Discover & Share GIFs](https://tenor.com/view/go-ahead-i%27m-all-ears-gif-13982086349020782746): 点击查看 GIF
- [Water Bears under the microscope](https://youtu.be/a8johHiOcyc?si=5LQ8gP_ybzK7sE2L): 显微镜下不同放大倍率的水熊虫（缓步动物）。水熊虫是微型动物，看起来像长着 4 对足的熊...
- [Reddit - Dive into anything](https://www.reddit.com/r/leetcode/comments/1ex7a1k/i_automated_leetcode_using_claudes_35_sonnet_api/): 未找到描述
- [How large language models work, a visual intro to transformers | Chapter 5, Deep Learning](https://youtu.be/wjZofJX0v4M?si=pKfOHIMGD29r-v6E&t=1343): 解析 Large Language Models 的工作原理。这些课程由观众直接资助，而非通过赞助广告：https://3b1b.co/support---这里是...
- [Hugging Face - Learn](https://hf.co/learn): 未找到描述
- [Aloe Blacc - I Need A Dollar](https://www.youtube.com/watch?v=nFZP8zQ5kzk): 未找到描述
- [Cute Pinch GIF - Cute Pinch So Fluffy - Discover & Share GIFs](https://tenor.com/view/cute-pinch-so-fluffy-gif-15488998239354870297): 点击查看 GIF
- [Golden Ratio in Quasicrystal Vibrations](https://physics.aps.org/articles/v17/s121): 实验表明，准晶体中的振动特性与被称为黄金分割率的数字有关。
- [Bringing Open-Source Models to Spreadsheets 🚀](https://huggingface.co/blog/fdaudens/hugging-face-on-sheets): 未找到描述
- [BangumiBase (BangumiBase)](https://huggingface.co/BangumiBase): 未找到描述
- [Fifth-place winner of Small World in Motion](https://www.cbc.ca/player/play/video/9.6511145): 一只幼年水熊虫骑在显微镜下的线虫上。

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1303332446431088690) (3 条消息):

> - `FastBert Tokenizer`
> - `AutoTokenizer Comparison`

- **FastBert Tokenizer 获得好评**：一位成员分享了他们了解到 **HuggingFace 的 FastBert tokenizer** 非常出色，并用笑脸表达了积极的情绪。
  
  - 该 tokenizer 因其性能和易用性而受到关注。
- **AutoTokenizer 与 FastBert 的区别**：一位成员询问了 **AutoTokenizer** 和 **FastBert** 之间的区别，寻求对其功能的澄清。
  
  - 另一位成员澄清说，**AutoTokenizer** 会根据模型自动选择 tokenizer，而 **FastBert** 特指一种 tokenizer 工具。

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1303100791577640990) (3 条消息):

> - `ShellCheck`
> - `Open Trusted Data Initiative`
> - `Largest multilingual dataset`
> - `Aud2Stm2Mdi`

- **用于 Shell 脚本分析的 ShellCheck**：[ShellCheck](https://github.com/koalaman/shellcheck) 是一款专为 Shell 脚本设计的静态分析工具，提供详细的见解和错误检查。
  
  - 它在 GitHub 上的仓库突出了其功能，使其成为 Shell 脚本开发者的重要工具。
- **开源数据的激动人心公告**：很高兴宣布 **@pleiasfr** 将与 **@thealliance_ai** 共同领导 **Open Trusted Data Initiative**，并将于 **11 月 11 日**发布一个包含 **2 万亿 token** 的海量多语言数据集。
  
  - 该数据集将在 [Hugging Face](https://huggingface.co/) 上提供，旨在推动 LLM 训练工作。
- **创新工具 Aud2Stm2Mdi**：一位成员分享了 Hugging Face 上的 [Aud2Stm2Mdi 工具](https://huggingface.co/spaces/eyov/Aud2Stm2Mdi) 链接，这似乎是 AI 工具链中一个令人耳目一新的补充。
  
  - 对于希望利用 AI 增强音频处理能力的用户来说，该工具可能会非常有益。

**提到的链接**：

- [来自 Alexander Doria (@Dorialexander) 的推文](https://x.com/Dorialexander/status/1853501675610247678)：很高兴宣布 @pleiasfr 加入 @thealliance_ai 共同领导 Open Trusted Data Initiative。我们将在 11 月 11 日发布用于 LLM 训练的最大多语言完全开放数据集...
- [Audio to Stems to MIDI Converter - eyov 开发的 Hugging Face Space](https://huggingface.co/spaces/eyov/Aud2Stm2Mdi)：未找到描述
- [GitHub - koalaman/shellcheck: ShellCheck，一个用于 Shell 脚本的静态分析工具](https://github.com/koalaman/shellcheck)：ShellCheck，一个用于 Shell 脚本的静态分析工具 - koalaman/shellcheck

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1303128101605085235) (29 条消息🔥):

> - `Computer Vision Model Quantization` (Computer Vision 模型量化)
> - `Docker Learning Series` (Docker 学习系列)
> - `Music Bot Development` (音乐机器人开发)
> - `Text2Text Model for Summarization` (用于摘要生成的 Text2Text 模型)
> - `Community Feedback Implementation` (社区反馈实施)

- **关于 Computer Vision 模型量化的侧边项目**：一位成员正在开发一个侧边项目，旨在对 **Computer Vision 模型进行量化**，以便在边缘设备上实现更快的推理。他们计划同时使用 **Quantization Aware Training** (量化感知训练) 和 **Post Training Quantization** (训练后量化) 方法，初始重点是减少模型权重。
  
  - 他们强调了理解减少维度如何影响 **Training 和 Inference** (训练与推理) 的重要性，这引起了社区内其他人的兴趣。
- **Docker 学习微系列**：一位成员在 DEV Community 上开始了一个名为 𝟭𝗺𝗶𝗻𝗗𝗼𝗰𝗸𝗲𝗿 的微系列，通过简短的文章介绍 Docker 概念。该系列旨在带领读者从基础走向专家级概念，目前已发布五篇文章。
  
  - 主题包括 **Docker 安装**、**基础概念**，以及学习如何 **构建和推送 Docker 镜像**。
- **Gary Andreessen 音乐机器人**：一位成员分享了他们幽默的项目，一个名为 **gary-andreessen** 的音乐机器人，它利用流水线从 Marc Andreessen 的演讲中创建音视频剪辑。该机器人在 Discord 和 Twitter 上运行，根据用户互动生成回复和音频续作。
  
  - 用户可以通过 **YouTube 链接** 与机器人互动，它会尝试幽默地回复评论，展示了该项目 **混乱的本质**，同时鼓励社区互动。
- **Text2Text 模型的初始版本**：一位成员发布了 **Text2Text 模型** 的初始版本，专为文本块的 "Map-Reduce" 摘要生成而设计。该模型可在 Hugging Face 上访问，旨在简化摘要生成过程。
  
  - 这一努力反映了社区内利用 AI 进行高效文本处理的持续兴趣。
- **社区建议的实施**：一位开发者确认并实施了关于改进应用程序中内容显示的社区反馈。该建议受到了好评，凸显了社区驱动增强功能的重要性。
  
  - 成员们对这些协作改进表示了热情，展示了积极的互动文化。

**提到的链接**：

- [Unexex](https://unexex.tech)：为现代学习者提供的引人入胜的 AI 制作课程。
- [来自 gary andreessen (@thepatch_gary) 的推文](https://x.com/thepatch_gary/status/1851509108513394893)：未来人们真的还会剪辑视频吗
- [来自 thecollabagepatch (@thepatch_kev) 的推文](https://x.com/thepatch_kev/status/1853410415104962627)：是的，我可能疯了。如果你提到机器人 Gary Andressen 并包含带有你想要的时间戳的 YouTube URL，对话线程中会发生以下情况。如果你喜欢他所做的...
- [GitHub - betweentwomidnights/gary-andreessen](https://github.com/betweentwomidnights/gary-andreessen)：通过在 GitHub 上创建账户来为 betweentwomidnights/gary-andreessen 的开发做出贡献。
- [未找到标题](https://dev.to/astrabert/1mindocker-1-what-is-docker-3baa)：未找到描述
- [未找到标题](https://dev.to/astrabert/1mindocker-2-get-docker-kh)：未找到描述
- [未找到标题](https://dev.to/astrabert/1mindocker-3-fundamental-concepts-55ph)：未找到描述
- [未找到标题](https://dev.to/astrabert/1mindocker-4-docker-cli-essentials-33pl)：未找到描述
- [未找到标题](https://dev.to/astrabert/1mindocker-5-build-and-push-a-docker-image-1kpm)：未找到描述
- [GitHub - AstraBert/1minDocker: A blog about Docker, to build your expertise from the fundamentals to the most advanced concepts!](https://github.com/AstraBert/1minDocker)：一个关于 Docker 的博客，旨在帮助你建立从基础到最先进概念的专业知识！ - AstraBert/1minDocker
- [文章](https://astrabert.github.io/1minDocker/posts/)：一个关于 Docker 的博客，旨在帮助你建立从基础到最先进概念的专业知识！

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/) (1 条消息):

west_ryder: 😝

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1303453581562871840) (3 条消息):

> - `HuggingMod`
> - `New Microsoft Models`

- **HuggingMod 需要放慢速度**：由于担心消息量过大，建议 <@169078428635627520> 稍微放慢发帖频率。
  
  - 分享了一个带有表情符号的友好提醒，以缓和语气。
- **对 Microsoft 新模型的兴奋**：<@790597705117204530> 询问其他人是否看到了新发布的 **Microsoft 模型**。
  
  - <@gettygermany> 确认 Microsoft 开发的内容正是大家所期待的。

 

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1303415417863274536) (3 条消息):

> - `Building RAG`
> - `Chroma Vector Store Issues`
> - `OpenAI Embeddings`
> - `Code References`

- **在 Chroma 中存储 Embedding 的挑战**：一位用户正尝试用 **21 份文档**构建 **RAG**，但在 **Chroma vector store** 中存储 Embedding 时遇到问题，仅成功存储了 **7 个 Embedding**。
  
  - 另一位成员询问是否出现了 **error**，并建议检查函数中的默认参数，以确定是否丢弃了剩余文档。
- **寻求 RAG 的代码示例**：原用户询问是否有人之前做过类似项目，并能分享代码片段供参考。
  
  - 这突显了 AI 开发过程中对社区支持和资源共享的需求。

 

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1303232970152476763) (1 条消息):

> - `Diffusion with Categorical Inputs`
> - `New architectures in Diffusion Models`

- **探索适用于类别输入的 Diffusion**：一位成员表示有兴趣将 **diffusion** 方法应用于 **categorical inputs**（类别输入），并引用了题为 [Diffusion for Categorical Data](https://arxiv.org/pdf/2211.15089) 的论文。
  
  - 他们询问是否有人在实验中具有该架构或类似方法的经验。
- **征集新 Diffusion 架构的使用经验**：同一位成员询问其他人是否尝试过论文中提到的关于 **categorical inputs** 的 **diffusion** 技术架构。
  
  - 他们鼓励分享与尝试这种新方法相关的见解或讨论。

 

---

### **Perplexity AI ▷ #**[**announcements**](https://discord.com/channels/1047197230748151888/1047204950763122820/1303519123673710653) (1 条消息):

> - `U.S. Presidential race tracking`
> - `Election hub`

- **Perplexity 追踪美国总统大选结果**：Perplexity 团队宣布，他们将在其 [election hub](http://perplexity.ai/elections) 上逐州追踪 **美国总统大选结果**，并进行实时统计。
  
  - 该举措旨在为用户提供选举过程的最新信息。
- **逐州结果的实时统计**：选举中心将展示各州的 **实时统计数据**，确保用户在结果公布时收到及时更新。
  
  - 这一努力体现了在关注总统竞选方面的透明度和可访问性的承诺。

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1303086719897440258) (364 条消息🔥🔥):

> - `Opus 移除`
> - `Perplexity Pro 功能`
> - `模型对比`
> - `Perplexity Bug`
> - `用户反馈`

- **Opus 移除引发用户不满**：用户对 **Claude 3 Opus** 的移除表示失望，许多人表示它是他们在编程和故事创作方面的首选模型。
  
  - 由于 **Haiku 3.5** 被认为表现较差，用户建议恢复之前的模型或寻找替代方案。
- **关于 Perplexity Pro 功能的见解**：几位用户讨论了他们的 Pro 订阅福利，包括通过 Revolut 等合作交易获得的付费模型访问权限。
  
  - 关于 Pro 是否包含 Claude 的访问权限以及移动端 App 的更改，用户仍存有疑问和好奇。
- **评估模型效能**：用户就 **Grok 2** 或 **Claude 3.5 Sonnet** 哪个在处理复杂研究和数据理解方面更有效展开了辩论。
  
  - 用户强调，虽然 GPT-4o 和 ChatGPT 在处理编程和创意任务方面表现出色，但 Perplexity 在学术场景中更具优势。
- **阻碍用户体验的 Bug**：**Perplexity** 目前正经历一些 Bug，导致模型输出混乱，并限制了用户与 Opus 的交互。
  
  - 用户报告称，尽管选择了其他模型，系统仍会退回到 GPT-4 响应，这导致需要不断调整 Prompt。
- **用户反馈与建议**：用户讨论了反馈在改进 Perplexity 体验中的重要性，建议更有效地整合 Space 自定义指令。
  
  - 用户强调需要对移动端和 macOS 应用程序进行用户友好的更新，以增强整体功能。

**提到的链接**：

- [介绍下一代 Claude](https://www.anthropic.com/news/claude-3-family)：今天，我们发布了 Claude 3 模型家族，它在广泛的认知任务中树立了新的行业标杆。该家族包含三个按能力升序排列的最先进模型...
- [Rolls Royce Royce GIF - Rolls royce Rolls Royce - 发现并分享 GIF](https://tenor.com/view/rolls-royce-rolls-royce-entry-gif-16920496844029391358)：点击查看 GIF
- [来自 Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1853858377459499114?s=46)：Perplexity 现在支持 @AnthropicAI 的 Claude 3.5 Haiku（昨日发布）作为 Claude 3 Opus 的替代品。停用 Claude 3 Opus 使 Perplexity 能够保持更新 Anthropic 的最新模型...
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1853498272863691125)：Claude 3 Haiku 仍可用于受益于图像输入或更低价格点的用例。https://docs.anthropic.com/en/docs/about-claude/models#model-names
- [你收到了邀请 | Revolut 英国](https://revolut.com/referral/?referral-code=ericqfpk!NOV1-24-VR-FR)：你被邀请加入 Revolut
- [Complexity - 强化版 Perplexity AI - Chrome 网上应用店](https://chromewebstore.google.com/detail/complexity-perplexity-ai/ffppmilmeaekegkpckebkeahjgmhggpj)：⚡ 强化你的 Perplexity AI
- [Perplexity Supply](https://perplexity.supply)：当好奇心遇见品质。我们的精品系列为好奇者提供精心设计的服饰。从重磅棉质基础款到刺绣单品，每一件都体现了我们的专注...

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1303242966672080956) (20 条消息🔥):

> - `Siberian Craters`
> - `Chemistry Rule Debunked`
> - `Human Brain on a Chip`
> - `Nvidia's Market Moves`
> - `AI's Upcoming Changes`

- **探索神秘的 Siberian Craters**：一段 YouTube 视频讨论了 **Siberian Craters**（西伯利亚巨洞）现象，这引起了科学家和探险家的共同兴趣。该视频旨在揭示这些地质形成的**原因**和**影响**。
  
  - 邀请观众深入探索 *Siberian 景观的奥秘* 以获取更多见解。
- **百年化学规则被打破**：一个链接讨论了最近的研究发现，这些发现**推翻了已有百年历史的化学规则**，在科学界引起了轰动。这一对传统认知的挑战突显了化学过程中的**新解释**。
  
  - 社区评论强调了*对未来化学研究和实践的影响*。
- **类脑芯片（Human Brain on a Chip）：一项突破**：一篇文章介绍了一种 **molecular neuromorphic platform**（分子神经形态平台），旨在模仿大脑功能，为先进的 **AI** 和神经科学研究铺平道路。该技术旨在增强我们对人类认知的理解。
  
  - 专家们对该平台在彻底改变 **AI** 发展方面的潜力表示*审慎乐观*。
- **Nvidia 准备挑战 Intel**：最近的报告显示，**Nvidia** 正致力于直接与 **Intel** 竞争，暗示了科技行业令人兴奋的发展。这一转变可能会影响未来的市场动态和产品策略。
  
  - 分析师建议*关注 Nvidia 可能进行的合作和产品发布*，这些举措可能会提升其地位。
- **AI 正在改变格局**：一篇文章概述了 **AI 即将到来的变化**，强调了这些转变将如何影响各个领域。这些预期的发展有望改变人们对 **AI** 技术的看法和应用。
  
  - 该领域的专家正在*热烈讨论对社会以及依赖 AI 的行业的潜在影响*。

 

**提到的链接**：[YouTube](https://www.youtube.com/embed/o6VCaHbrU4A)：未找到描述

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/) (1 条消息):

canarywolfs: 我也一样。很久以前就填了。甚至又填了一次，但还是没反应……🙁

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1303126304161529857) (3 条消息):

> - `Claude 3.5 Haiku`
> - `免费 Llama 3.2 模型`
> - `Chatroom 中的 PDF 功能`
> - `偶发性超时调查`
> - `用于降低延迟的预测输出 (Predicted output)`

- **Claude 3.5 Haiku 发布**：Anthropic 推出了 **Claude 3.5** 的标准版和自我审查版，更多带有日期的选项[可在此处获取](https://openrouter.ai/anthropic/claude-3-5-haiku)。
  
  - *我们很高兴看到这一最新模型在实际应用中的表现*。
- **免费访问 Llama 3.2 模型**：**Llama 3.2** 模型（包括 **11B** 和 **90B**）现在提供免费的快速端点 (endpoint)，分别达到 **280tps** 和 **900tps** [详情见此](https://openrouter.ai/meta-llama/llama-3.2-90b-vision-instruct:free)。
  
  - *此举预计将增加社区对开源模型的参与度*。
- **Chatroom 新增 PDF 功能**：新功能允许用户在 Chatroom 中粘贴或附加 **PDF**，以便使用 OpenRouter 上的任何模型进行分析。
  
  - 此外，最高购买限额已提高至 **$10,000**。
- **解决 524 错误**：团队重新构建了 API 并成功迁移了 Chatroom 请求，自更改以来实现了 **零** 524 错误。
  
  - 如果稳定性在未来一天内保持不变，他们计划继续迁移，并邀请用户测试新 API。
- **通过预测输出优化延迟**：**预测输出 (predicted output)** 功能现已支持 OpenAI 的 **GPT-4** 模型，通过 `prediction` 属性优化编辑和重写场景。
  
  - 示例代码片段展示了如何使用该功能更高效地处理大型文本请求。

**提到的链接**：

- [Claude 3.5 Haiku - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3-5-haiku)：Claude 3.5 Haiku 具有更强的速度、编程准确性和工具使用 (tool use) 能力。通过 API 运行 Claude 3.5 Haiku。
- [Claude 3.5 Haiku - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3-5-haiku-20241022>)：Claude 3.5 Haiku 具有更强的速度、编程准确性和工具使用 (tool use) 能力。通过 API 运行 Claude 3.5 Haiku。
- [Claude 3.5 Haiku - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3-5-haiku:beta>)：Claude 3.5 Haiku 具有更强的速度、编程准确性和工具使用 (tool use) 能力。通过 API 运行 Claude 3.5 Haiku。
- [Claude 3.5 Haiku - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3-5-haiku-20241022:beta>)：Claude 3.5 Haiku 具有更强的速度、编程准确性和工具使用 (tool use) 能力。通过 API 运行 Claude 3.5 Haiku。
- [Llama 3.2 90B Vision Instruct - API, Providers, Stats](https://openrouter.ai/meta-llama/llama-3.2-90b-vision-instruct:free>)：Llama 90B Vision 模型是一款顶级的 900 亿参数多模态模型，专为最具挑战性的视觉推理和语言任务设计。它在图像描述方面提供了无与伦比的准确性...
- [Llama 3.2 11B Vision Instruct - API, Providers, Stats](https://openrouter.ai/meta-llama/llama-3.2-11b-vision-instruct:free>)：Llama 3.2 11B Vision 是一款拥有 110 亿参数的多模态模型，旨在处理结合视觉和文本数据的任务。通过 API 运行 Llama 3.2 11B Vision Instruct。

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1303086397057536091) (340 messages🔥🔥):

> - `Hermes 模型状态`
> - `AI 模型的定价担忧`
> - `OpenRouter 的用户体验`
> - `Rate limits 与 Credits`
> - `特定用例的模型推荐`

- **Hermes 模型体验**：免费版的 **Hermes 405B** 模型表现不稳定，一些用户反映它在某些时段可以工作，但经常失败。
  
  - 许多用户表示希望该模型出现的问题意味着更新或修复正在进行中。
- **对定价和性能的担忧**：用户正在讨论 **Claude 3.5** 和 **Haiku** 等模型的高昂定价，一些人表示其质量并不匹配其成本。
  
  - 对话强调了对近期停机时间的不满，并要求优先处理付费 API 请求。
- **OpenRouter 的用户体验**：几位用户分享了使用 OpenRouter 服务的复杂体验，指出了诸如 524 错误以及在各种模型之间选择的问题。
  
  - 一些用户找到了替代方案，例如 **WizardLM-2 8x22B**，同时也对当前的服务状态表达了挫败感。
- **理解 Rate Limits 和 Credits**：在询问 OpenRouter 上的额度时，用户了解到他们的美元余额直接对应其 Credits，即 30 美元等于 30 Credits。
  
  - Rate limits 被解释为针对特定账户，并与可用 Credits 的数量挂钩。
- **特定任务的模型推荐**：用户讨论了各种模型对特定任务的适用性，并推荐了 **Hermes** 和 **Euryale** 等替代方案用于角色扮演（roleplaying）。
  
  - 建议强调，与闭源供应商相比，使用开源模型可以获得限制更少的输出。

**提到的链接**：

- [New OpenAI feature: Predicted Outputs](https://simonwillison.net/2024/Nov/4/predicted-outputs/)：OpenAI API 的有趣新功能——这是我第一次从任何供应商那里看到这种功能。如果你知道你的 Prompt 大部分会返回相同的内容……
- [PDF.js - Home](https://mozilla.github.io/pdf.js/)：未找到描述
- [Tweet from OpenRouter (@OpenRouterAI)](https://x.com/OpenRouterAI/status/1853573174849319325)：聊天室支持 PDF！你现在可以在聊天室粘贴或附加 PDF，使用 OpenRouter 上的任何模型进行分析：
- [Chatroom | OpenRouter](https://openrouter.ai/chat)：LLM 聊天室是一个多模型聊天界面。添加模型并开始聊天！聊天室将数据本地存储在你的浏览器中。
- [Limits | OpenRouter](https://openrouter.ai/docs/limits)：设置模型使用限制
- [Elevated errors for requests to Claude 3.5 Sonnet](https://status.anthropic.com/incidents/hc6p15sbcx11)：未找到描述
- [Grok Beta - API, Providers, Stats](https://openrouter.ai/x-ai/grok-beta)：Grok Beta 是 xAI 的实验性语言模型，具有最先进的推理能力，最适合复杂和多步骤的用例。它是 [Grok 2](https://x. Run Grok Beta w... 的继任者。
- [Keys | OpenRouter](https://openrouter.ai/settings/keys)：管理你的密钥或创建新密钥
- [Tweet from OpenAI Developers (@OpenAIDevs)](https://x.com/OpenAIDevs/status/1853564730872607229)：介绍 Predicted Outputs——通过提供参考字符串，显著降低 gpt-4o 和 gpt-4o-mini 的延迟。https://platform.openai.com/docs/guides/latency-optimization#use-predicted-outpu...
- [Hermes 3 405B Instruct - API, Providers, Stats](https://openrouter.ai/nousresearch/hermes-3-llama-3.1-405b)：Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理、多轮对话、长上下文连贯性……
- [tencent/Tencent-Hunyuan-Large · Hugging Face](https://huggingface.co/tencent/Tencent-Hunyuan-Large)：未找到描述
- [Models | OpenRouter](https://openrouter.ai/models?max_price=0)：在 OpenRouter 上浏览模型
- [Gemini Flash 1.5 - API, Providers, Stats](https://openrouter.ai/google/gemini-flash-1.5)：Gemini 1.5 Flash 是一个基础模型，在各种多模态任务中表现良好，如视觉理解、分类、摘要以及从图像、音频和视频创建内容……
- [Models | OpenRouter](https://openrouter.ai/models?order=pricing-low-to-high)：在 OpenRouter 上浏览模型
- [OpenRouter Status](https://status.openrouter.ai/)：OpenRouter 事件历史
- [Models & Pricing | DeepSeek API Docs](https://api-docs.deepseek.com/quick_start/pricing)：下面列出的价格单位为每 1M tokens。Token 是模型识别的最小文本单位，可以是一个词、一个数字，甚至是一个标点符号。我们将根据总额计费……

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1303108683076603924) (4 messages):

> - `Custom Provider Beta Keys`
> - `Accessing BYOK Feature`
> - `Advantages of Custom Keys`

- **申请 Custom Provider Beta Keys**：多位用户表示有兴趣获取用于其开发脚本的 **custom provider beta keys**，表明他们希望尝试此功能。
  
  - *Thanks!* 是用户对请求获得协助后表达感谢的常用语。
- **访问 Bring Your Own Keys Beta 功能**：一位用户询问如何申请访问 **bring your own keys** (BYOK) Beta 功能，表达了使用该功能的愿望。
  
  - 讨论的重点在于澄清访问 BYOK 的流程。
- **探索自定义密钥的优势**：用户提出了关于**使用自定义密钥的优势**（除账户组织之外）的问题，引发了对额外收益的推测。
  
  - 一位用户注意到了潜在的好处，但要求提供**更多细节**，以了解可用优势的完整范围。

---

### **aider (Paul Gauthier) ▷ #**[**announcements**](https://discord.com/channels/1131200896827654144/1133060115264712836/1303091303843762238) (6 messages):

> - `Aider v0.62.0`
> - `Claude 3.5 Haiku Performance`
> - `ChatGPT/Claude Integration`

- **Aider v0.62.0 发布**：Aider v0.62.0 现在全面支持 **Claude 3.5 Haiku**，该模型在 [code editing leaderboard](https://aider.chat/docs/leaderboards/) 上得分为 **75%**。此版本允许轻松编辑源自 ChatGPT 等网页版 LLM 的文件。
  
  - 此外，Aider 编写了此版本中 **84%** 的代码，进一步强调了其效率。
- **Claude 3.5 Haiku 对比 Sonnet**：据观察，**Claude 3.5 Haiku** 的表现几乎与旧版 **Sonnet** 一样出色，且更具性价比。用户可以使用 `--haiku` 命令行选项启动它。
- **使用 Web App 进行文件编辑**：Aider 允许用户通过在 Web App 中与 ChatGPT 或 Claude 交互并直接复制回复，从而轻松应用文件编辑。这可以通过运行 `aider --apply-clipboard-edits file-to-edit.js` 来利用 LLM 的输出实现更改。
- **用户的集成咨询**：一位用户询问了使用 ChatGPT/Claude 集成而非直接在 Aider 中工作的益处，暗示可能存在 Token 节省。另一位用户询问此功能是否仅限于浏览器模式。
- **关于应用编辑的 GitHub Issue**：有人在 GitHub 提交了 Issue，询问是否可以使用 `aider --apply` 处理来自 chatgpt.com 等网页前端的输出，理由是 **o1-preview** 的订阅更便宜。该用户对目前将网页前端的编辑应用到本地文件的过程感到沮丧，认为非常麻烦。

**提到的链接**：

- [Aider LLM Leaderboards](https://aider.chat/docs/leaderboards/)：LLM 代码编辑技能的量化基准。
- [[Q] Is it possible to use `aider --apply` with output from web frontends like chatgpt.com? · Issue #2203 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2203)：o1-preview 在 chatgpt.com 上的订阅更便宜，总的来说，我喜欢直接使用原始 LLM 的灵活性。但将网页前端的编辑应用到本地文件非常麻烦。我经常……

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1303086674368270367) (171 条消息🔥🔥):

> - `AI Model Comparisons` (AI 模型对比)
> - `Benchmarking Aider` (Aider 基准测试)
> - `Aider Updates` (Aider 更新)
> - `Coding with AI` (使用 AI 编程)
> - `AI Forum and Subreddit Recommendations` (AI 论坛与 Subreddit 推荐)

- **AI 编程模型对比**：用户讨论了各种 AI 编程模型之间的性能差异，其中 **3.5 Haiku** 在对比 **Sonnet 3.5** 和 **GPT-4o** 时被指出效果较低。
  
  - 许多用户预计即将推出的模型（如 **4.5o**）可能会挑战现有标准，从而可能影响 **Anthropic** 模型的市场。
- **Aider Bug 报告与修复**：Aider 在文件创建和编辑方面存在持续性问题，用户报告在升级到 **0.61** 版本后出现了功能故障。
  
  - 一位用户指出，回滚到 **0.60** 版本解决了许多问题，这凸显了未来版本对稳定性的需求。
- **Predicted Outputs 功能的影响**：**OpenAI** 推出的 **Predicted Outputs** 功能被视为 **GPT-4o** 模型的潜在游戏规则改变者，它能降低延迟并提高代码编辑效率。
  
  - 用户预计该功能可能会显著影响模型基准测试，特别是在与竞争对手的直接对比中。
- **Subreddit 与论坛推荐**：为了收集 AI 编程信息，用户推荐了各种论坛，包括 **Aider Discord**、**Claude Reddit** 和 **Cursor Discord**。
  
  - 其他值得关注的推荐包括 **LocalLLaMA** 和 **ChatGPTCoding** 的 Subreddit，用于获取见解和更新。
- **Aider 与 Cline 的用户体验**：一位用户分享了使用 **Aider** 和 **Cline** 的对比经验，指出 Aider 在处理现有代码和效率方面表现更好。
  
  - 尽管 Aider 在 IDE 集成方面存在一些局限性，但该用户因其丰富的设置选项和经济的速率限制（rate limits）而更青睐它。

**提到的链接**：

- [OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/OpenAIDevs/status/1853564730872607229)：介绍 Predicted Outputs——通过提供参考字符串，显著降低 gpt-4o 和 gpt-4o-mini 的延迟。https://platform.openai.com/docs/guides/latency-optimization#use-predicted-outpu...
- [Aider LLM 排行榜](https://aider.chat/docs/leaderboards/#contributing-benchmark-results)：LLM 代码编辑能力的定量基准测试。
- [Aider LLM 排行榜](https://aider.chat/docs/leaderboards/)：LLM 代码编辑能力的定量基准测试。
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1ga5m5r/updated_claude_sonnet_35_tops_aider_leaderboard/)：未找到描述
- [Aider LLM 排行榜](https://aider.chat/docs/leaderboards/#contributing-benchmark-results)：LLM 代码编辑能力的定量基准测试。
- [Reddit - Dive into anything](https://www.reddit.com/r/ChatGPTCoding/comments/1gkiknl/yet_again_a_cline_and_aider_post/)：未找到描述
- [发布历史](https://aider.chat/HISTORY.html)：关于 Aider 编写自身代码的发布说明和统计数据。
- [OpenAI o1 完整版被意外提前发布？！让我们测试一下！](https://youtu.be/gCtF6eCxR88?si=YUEWgoMonTiFnjS5)：看起来 ChatGPT o1 昨晚提前发布了几个小时。在它被撤下之前，我成功进行了几次提示。最初的 T...
- [Issues · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2233.)：Aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。
- [零一万物-大模型开放平台](https://platform.lingyiwanwu.com/)：零一万物大模型开放平台，权威盲测国产最有，全系平替 GPT 系列。
- [GitHub - Aider-AI/aider: aider is AI pair programming in your terminal](https://github.com/Aider-AI/aider.git)：Aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。
- [在 architect 模式下，Aider 附加到已添加的文件而不是创建新文件 · Issue #2258 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2258)：你们好！首先，非常感谢你们在 Aider 上所做的工作，它是一个不可思议的工具。我一直在尝试使用 Claude 3.5 Sonnet v2 作为架构模型来玩转 architect 模式...

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1303111278906179645) (74 messages🔥🔥):

> - `Aider Configuration`
> - `DeepSeek Model Issues`
> - `Using Claude Haiku`
> - `Benchmarking Models`

- **Aider 配置选项**：用户讨论了如何有效地同时使用 `.env` 文件和 `.aider.conf.yml` 进行配置，并强调当两者包含相似设置时，后者的优先级更高。
  
  - 几位成员分享了他们的 YAML 配置示例，重点介绍了模型类型和 API 基础 URL 等特定参数。
- **DeepSeek 模型聊天模板问题**：一位用户报告了在使用 llama.cpp 运行 **DeepSeek-V2.5** 时遇到的挑战，原因是聊天模板不受支持，回退到 chatML 导致响应效果不佳。
  
  - 另一位成员建议该问题可能与模型的量化有关，并建议考虑使用 lmstudio 等替代方案以获得更好的性能。
- **使用 Claude Haiku 作为编辑器模型**：围绕将 **Claude 3 Haiku** 作为编辑器模型使用展开了多次讨论，特别是当主模型缺乏强大的编辑能力时。
  
  - 成员们指出，使用像 Haiku 这样健壮的模型进行编辑可以简化开发过程，特别是在需要精确语法管理的语言中。
- **基准测试模型性能**：用户询问 Aider 是否可以在基准测试期间绕过请求限制，特别是针对那些超出请求限制的模型。
  
  - 讨论了基准测试性能与 API 效率的关系，其中一些模型被指出未针对本地内存限制进行优化。

**提到的链接**：

- [VSCode Aider - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=Apertia.vscode-aider)：Visual Studio Code 扩展 - 直接在 VSCode 中运行 Aider，实现无缝集成和增强工作流。
- [Config with .env](https://aider.chat/docs/config/dotenv.html)：使用 .env 文件为 aider 存储 LLM API 密钥。
- [Configuration](https://aider.chat/docs/config.html)：关于 aider 所有设置及其使用方法的信息。
- [YAML config file](https://aider.chat/docs/config/aider_conf.html#sample-yaml-config-file)：如何使用 yaml 配置文件配置 aider。
- [legraphista/DeepSeek-V2.5-IMat-GGUF · Hugging Face](https://huggingface.co/legraphista/DeepSeek-V2.5-IMat-GGUF)：未找到描述
- [mlx-community/Qwen2.5-32B-Instruct-4bit · Hugging Face](https://huggingface.co/mlx-community/Qwen2.5-32B-Instruct-4bit)：未找到描述
- [aider/aider/website/assets/sample.env at main · Aider-AI/aider](https://github.com/Aider-AI/aider/blob/main/aider/website/assets/sample.env#L285)：aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1303396357167386654) (5 messages):

> - `Local Tests Failures`
> - `Transformers Bug`

- **本地测试抛出 TypeError**：一位用户报告在运行本地测试时出现了 **TypeError**，指出在使用 `pytest` 进行测试时，'tuple' 无法转换为 'PyList'。
  
  - 他们发现这是一个已知问题，并在 [GHA logs](https://link.to.gha.logs) 中得到了确认。
- **最新 Transformers 中的 Tokenizer 问题**：另一位用户澄清说，最新版本的 `transformers` 中存在一个 **bug**，即 tokenizer 不接受元组（tuples），这导致了测试失败。
  
  - 他们提到有一个 [正在进行的 PR](https://link.to.PR) 旨在修复此 bug。

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1303092138007072798) (130 条消息🔥🔥):

> - `Running Reading Groups`（运行阅读小组）
> - `Model Training Techniques`（模型训练技术）
> - `Optimization Strategies`（优化策略）
> - `Logits and Probability Distributions`（Logits 与概率分布）
> - `Implementation of Dualizers`（Dualizers 的实现）

- **阅读小组中积极性胜过专业知识**：*一位成员强调，成功运行一个阅读小组更多地取决于积极性而非专业知识。* 他们在没有先验知识的情况下开始了 Mech Interp 阅读小组，并始终坚持维护。
- **对优化器设置的担忧**：*一场关于训练模型时各种优化器设置（beta1 和 beta2）影响的讨论展开。* 成员们对 FSDP 和 PP 等不同策略的兼容性和性能表达了不同看法。
- **理解模型输出中的 Logits**：*关于优化 Logits 输出以及训练中合适的数学范数（norms）存在争论。* 一些参与者建议利用 L-inf 范数来最大化概率，而另一些人则关注通过 KL divergence（KL 散度）来维持分布形状。
- **深度学习技术的实用性**：*讨论强调了深度学习中涉及的复杂性以及对训练期间使用的操作进行推理的难度。* 成员们提议创建一个全面的文档系统，为日常用户抽象并简化这些细节。
- **研究中 Dualizer 的实现**：*一位成员宣布实现了一篇论文中讨论的 Dualizer，在损失增加极小的情况下取得了具有竞争力的结果。* 该工作首先专注于在没有进行显著调优的情况下优化 Embedding 层。

**提到的链接**：

- [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783)：我们提出了用于长上下文 LLM 推理的上下文并行（Context Parallelism），在 16 个节点的 128 张 H100 GPU 上实现了长上下文 Prefill 延迟的近线性扩展。
- [Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent](https://arxiv.org/abs/2411.02265)：在本文中，我们介绍了 Hunyuan-Large，这是目前最大的基于 Transformer 的开源 MoE 模型，总参数量为 3890 亿，激活参数量为 520 亿...
- [Stable and low-precision training for large-scale vision-language models](https://arxiv.org/abs/2304.13013)：我们介绍了 1) 加速和 2) 稳定大规模视觉-语言模型训练的新方法。1) 在加速方面，我们引入了 SwitchBack，这是一种用于 int8 量化训练的线性层...
- [modded-nanogpt/train_gpt2.py at fc--dual · leloykun/modded-nanogpt](https://github.com/leloykun/modded-nanogpt/blob/fc--dual/train_gpt2.py#L75)：在 2.67B Token 中达到 NanoGPT (124M) 的质量。通过在 GitHub 上创建账号为 leloykun/modded-nanogpt 的开发做出贡献。
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)：语言建模领域的近期工作表明，训练大型 Transformer 模型推动了 NLP 应用的最前沿。然而，极大型模型可能非常...
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)：生成式预训练 Transformer 模型（如 GPT 或 OPT）因其在复杂语言建模任务中的突破性性能而脱颖而出，但也因其极高的计算...
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)：LLM 已被广泛采用，但推理时需要巨大的 GPU 显存。我们为前馈层和注意力投影层开发了一种 Int8 矩阵乘法程序...
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)：大型深度学习模型提供了显著的准确率提升，但训练数千亿到数万亿参数具有挑战性。现有的数据并行和模型并行等解决方案存在根本性的...
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)：由于自注意力（Self-attention）的时间和显存复杂度随序列长度呈二次方增长，Transformer 在长序列上运行缓慢且耗费显存。近似注意力方法曾试图解决...
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)：我们介绍了 MegaBlocks，一个用于在 GPU 上进行高效 MoE 训练的系统。我们的系统针对当前框架的局限性而设计，这些局限性限制了 MoE 层中的动态路由...

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1303144212593639438) (1 条消息):

> - `W2S Model Files`

- **关于 W2S 模型文件的查询**：一位成员询问 GitHub 上的 [W2S 项目](https://github.com/EleutherAI/w2s/tree/main) 是否在某处存储了 **model files**。
  
  - 提出此问题是因为 **W2S 开发** 受到鼓励，成员们正在寻求获取必要资源的途径。
- **W2S 的 GitHub 资源**：讨论强调了 **W2S 项目** 的 [GitHub 链接](https://github.com/EleutherAI/w2s)，并邀请社区贡献代码。
  
  - 这可能为该项目开发的加强协作铺平道路。

 

**提到的链接**：[GitHub - EleutherAI/w2s](https://github.com/EleutherAI/w2s/tree/main)：通过在 GitHub 上创建账户来为 EleutherAI/w2s 的开发做出贡献。

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1303142369213943818) (10 条消息🔥):

> - `评估中的控制尝试`
> - `LLM 鲁棒性评估 PR`
> - `推理挂起问题`
> - `NCCL 显存溢出 (OOM) 错误`
> - `Batch size 调整`

- **使用 repeats 控制评估尝试次数**：一位成员询问了如何在 GSM8K 等任务的评估中控制尝试次数 (`k`)。根据 [示例配置](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml)，已明确在任务模板中使用 `repeats` 可以实现此目的。
  
  - 然而，正如在 [此处](https://github.com/EleutherAI/lm-evaluation-harness/blob/c0745fec3062328e0ab618f36334848cdf29900e/lm_eval/filters/selection.py#L56) 找到的多数投票逻辑中所述，除非多数回答正确，否则系统不会返回正确答案。
- **LLM 鲁棒性评估 PR 已提交**：一位成员宣布提交了一个针对三个不同数据集的 **LLM Robustness Evaluation** PR，并邀请反馈和评论。可以在 [此处](https://github.com/EleutherAI/lm-evaluation-harness/pull/2452) 查看该 PR。
  
  - 具体改进包括为大语言模型添加系统性的一致性和鲁棒性评估，同时修复了之前的 bug。
- **eval harness 上的推理挂起**：一位用户报告了在与他人协作项目时，使用 eval harness 会导致推理挂起的问题。该问题尤其令人担忧，因为它阻碍了项目的运行进度。
  
  - 目前尚未提供具体的解决方案，但另一位成员根据共同经历对该挂起问题表示了关注。
- **lm_eval 期间的 NCCL 显存溢出错误**：一位成员描述了在使用自动检测的 batch size 在多个 GPU 上运行 **lm_eval** 时，收到了 `CUDA failure 2 'out of memory'` 错误。该问题出现在 log likelihood 请求完成并尝试重新组装所有内容时。
  
  - 手动设置较小的 batch size 解决了该问题，促使该用户考虑提交 issue 报告。
- **Batch size 的调整**：一位用户指出，手动调整 batch size 可以解决在多个 GPU 上进行评估时的显存溢出问题。尽管自动检测存在问题，但较小的 batch size 是一个有效的权宜之计。

**提到的链接**：

- [Score tasks by rimashahbazyan · Pull Request #2452 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/2452)：添加了 SCORE：大语言模型的系统一致性和鲁棒性评估。修复了 generate until 任务的一个 bug，将 "until" 参数默认设置为每个模型的...
- [lm-evaluation-harness/lm_eval/filters/selection.py at c0745fec3062328e0ab618f36334848cdf29900e · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c0745fec3062328e0ab618f36334848cdf29900e/lm_eval/filters/selection.py#L56))：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1303102052649533523) (100 条消息🔥🔥):

> - `Python 3.11 性能`
> - `Qwen 2.5 模型支持`
> - `Fine-Tuning LLM`
> - `训练方法论`
> - `Unsloth 库问题`

- **Python 3.11 提供了显著的性能提升**：鼓励用户切换到 [Python 3.11](https://docs.python.org/3/whatsnew/3.11.html#whatsnew311-faster-cpython)，因为由于各项优化，它在 Linux 上显示出高达 **1.25 倍的加速**，在 Windows 上显示出 **1.12 倍的加速**。
  
  - *核心模块*采用静态分配以实现更快的加载，且函数调用现在已内联（inlined），从而增强了整体性能。
- **对 Qwen 2.5 模型功能的浓厚兴趣**：讨论确认 *llama.cpp* 已支持 **Qwen 2.5**，如 [Qwen 文档](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)所述。
  
  - 社区对 *Unsloth* 中集成 Vision 模型表示期待，预计很快就会推出。
- **使用小数据集进行 Fine-Tuning 的挑战和策略**：用户在思考仅使用 **10 个示例**（共 60,000 字）进行模型 Fine-Tuning 的可行性，重点在于标点符号纠正。
  
  - 建议包括使用 batch size 为 1，以减轻与有限数据相关的挑战。
- **训练方法论讨论**：社区成员争论是先构建数据集还是先研究训练方法，倾向于优先创建数据集。
  
  - 普遍共识是，有效的训练方法论通常紧随数据集准备之后。
- **对最新 *Unsloth* 库更新的担忧**：一位用户报告称 *Unsloth* 最近的一个 PR 导致了问题，他们通过回退到库的早期版本解决了该问题。
  
  - 维护者承认了该问题，并表示修复程序已经实施。

**提到的链接**：

- [llama.cpp - Qwen](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/user/Rombodawg/comments/1gjv968/creating_true_artificial_intelligence_required_a/)：未找到描述
- [Daniel Han (@danielhanchen) 的推文](https://x.com/danielhanchen/status/1853535612898533715)：如果你还在使用 Python 3.10，请切换到 3.11！安装了 3.11 的 Linux 机器快约 1.25 倍。Mac 快 1.2 倍。Windows 快 1.12 倍。Python 3.12 看起来像是针对 Windows 32 位的性能修复（谁还在用 32 位...
- [importlib.metadata.PackageNotFoundError: No package metadata was found for The 'unsloth' distribution was not found and is required by this application · Issue #124 · unslothai/unsloth](https://github.com/unslothai/unsloth/pull/124)：训练环境：LLaMaFactory `01/24/2024 01:53:50 - INFO - llmtuner.model.patcher - Quantizing model to 4 bit. Traceback (most recent call last): File "/usr/local/lib/python3.10/dist-packages/trans...`
- [主页](https://github.com/unslothai/unsloth/wiki#finetuning-the-lm_head-and-embed_tokens-matrices)：Fine-Tuning Llama 3.2, Mistral, Phi, Qwen & Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
- [unsloth/unsloth/kernels/fast_lora.py at main · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py#L42))：Fine-Tuning Llama 3.2, Mistral, Phi, Qwen & Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1303122296755720263) (12 条消息🔥):

> - `NVIDIA GeForce RTX 团队调查`
> - `垃圾信息与自我推广问题`
> - `聊天社区动态`

- **NVIDIA 寻求社区反馈**：**NVIDIA GeForce RTX 团队**正在邀请社区中的 AI 爱好者通过 10 分钟的简短交流分享关于 AI 工具的使用体验和痛点，[在此预约](https://calendly.com/aslisabanci-01-nvidia/10min)。
  
  - 他们的目标是收集可能影响 NVIDIA 产品未来方向和路线图的见解。
- **垃圾信息引发管理行动**：一名成员警告另一名成员*不要发送垃圾信息*，并提到已经删除了两次其消息，表现出对重复行为的沮丧。
  
  - 这种管理行为反映了社区在排除干扰、维持建设性对话方面的努力。
- **明确自我推广限制**：服务器内再次强调禁止**自我推广（self-promotion）**，重点在于保持社区专注，避免个人广告。
  
  - 这种对服务器指南的评论体现了对维护讨论完整性的关注。
- **社区与新成员互动**：一名成员对 NVIDIA 代表表示热烈欢迎，并建议他们在相关频道发布咨询，以吸引更多关注。
  
  - 这种沟通的开放性展示了社区在吸纳新贡献者方面的支持态度。
- **社区技术水平讨论**：一名成员评论说，社区成员在技术专长方面可能更偏向**初级（lower-level）**。
  
  - 这一观察说明了群体中存在不同的技能水平，并表明需要量身定制的对话。

 

**提到的链接**：[10 Minute Meeting - Asli Sabanci](https://calendly.com/aslisabanci-01-nvidia/10min)：大家好！作为 NVIDIA GeForce RTX 团队，我们正在寻求社区 AI 爱好者的意见，以指导未来的产品方向和路线图。我们很乐意与一些低代码/无代码背景的成员交流...

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1303090529453735976) (25 条消息🔥):

> - `在 Wikipedia 数据上进行微调`
> - `模型推理问题`
> - `保存微调后的模型`
> - `格式化训练数据`
> - `Qwen 模型性能`

- **微调数据集格式建议**：一位用户询问了在 **Wikipedia 结构化数据**上微调模型的良好数据集格式。
  
  - 虽然没有提供直接回复，但多名成员寻求对结构化格式的澄清。
- **推理在 1 个 Epoch 后停止**：成员们对模型在仅运行 **1 个 Epoch** 后就停止推理表示担忧，这引起了困惑。
  
  - 诊断该问题所需的进一步输入仍未得到答复，但这突显了一个常见挑战。
- **本地保存微调模型**：一位用户寻求帮助，希望在不损失性能的情况下本地保存微调后的 **Unsloth 模型**。
  
  - 建议是参考社区提供的用于合并和保存 adapter 的代码片段。
- **语言翻译训练数据格式化**：一名成员讨论了基于**语言翻译**格式化训练数据的困难，称其在推理过程中返回了乱码。
  
  - 回复指出需要特定的格式，并询问他们是否正在使用 **Unsloth 推理**。
- **Qwen 模型幻觉**：用户注意到 **Qwen 2.5 1.5B 模型**尽管尝试通过添加 'End of text' 来提高数据集质量，但仍持续出现幻觉。
  
  - 一种解释认为 **Qwen** 模型使用了大量的**中文**数据进行训练，导致了意外的输出。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/1303096694715580578) (1 条消息):

> - `mtbench 评估`
> - `Hugging Face 指标`

- **寻求 mtbench 评估实现**：一名成员询问了关于在 mtbench 数据集上运行类似 mtbench 评估的 callback（回调）参考实现。
  
  - *是否有某种 Hugging Face evaluate metric 的实现？*
- **mtbench 评估的回调**：有人请求关于实现 mtbench 数据集评估回调的见解，特别是类似于 mtbench 评估的方法。
  
  - 该咨询强调了当前项目对该功能的需求，反映了对简化评估流程的渴望。

 

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1303098812759543863) (67 条消息🔥🔥):

> - `LM Studio 使用`
> - `模型适应性`
> - `服务器日志访问`
> - `便携版 LM Studio`
> - `模型性能对比`

- **将 LM Studio 作为 USB 便携版应用运行**：一位用户询问是否可以从 USB 闪存盘运行 LM Studio，但得到的答复是目前没有官方的便携版本。
  - 其他用户建议使用 Linux AppImage 二进制文件，或者分享了一个可能使其便携化的脚本。
- **在 LM Studio 中访问服务器日志**：为了查看服务器日志，一位用户了解到按下 CTRL+J 可以调出 LM Studio 中的服务器选项卡。
  - 该信息被迅速提供，以帮助其他尝试监控日志的用户。
- **配合 Ngrok 使用 HTTP**：一位用户询问是否由于 ngrok 的限制，可以从其 HTTP 请求中移除 '/v1'。
  - 建议他们可以运行一个代理服务器，因为 ngrok 的免费计划存在限制，会阻止某些设置。
- **LM Studio 功能与模型对比**：讨论回顾了旧版 LM Studio 的功能，特别是最新更新中缺失的对比工具。
  - 成员们在评估版本更新优于旧迭代的益处时，怀念起这些功能。
- **模型性能评估**：注意到在使用 Vulkan 时，Mistral Nemo 的性能表现优于 Qwen2，突显了架构影响的差异。
  - 这引发了关于不同架构如何影响性能的好奇，特别是在快速处理 token 方面。

**提到的链接**：

- [Reddit - Dive into anything](https://www.reddit.com/user/Rombodawg/comments/1gjv968/creating_true_artificial_intelligence_required_a/)：未找到描述
- [Hacker koan - Simple English Wikipedia, the free encyclopedia](https://simple.wikipedia.org/wiki/Hacker_koan)：未找到描述
- [Maxime Labonne - Create Mixtures of Experts with MergeKit](https://mlabonne.github.io/blog/posts/2024-03-28_Create_Mixture_of_Experts_with_MergeKit.html)：将多个专家模型组合成单个 frankenMoE
- [adding-support-for-mamba2 by Goekdeniz-Guelmez · Pull Request #1009 · ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/pull/1009)：未找到描述

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1303104811616702484) (53 条消息🔥):

> - `Windows 调度器性能`
> - `GPU vs CPU 优化`
> - `LLM 上下文处理`
> - `笔记本电脑散热技术`
> - `内存带宽限制`

- **Windows Scheduler 效率低下**：成员们对 **Windows Scheduler** 表示沮丧，指出它在 CPU 线程管理方面表现不佳，尤其是在多核配置下。
  - 一位成员主张手动为进程分配 CPU 亲和性（affinity）和优先级，以增强性能。
- **平衡 GPU 与上下文设置**：用户分享了将 **GPU 设置**调至最大，同时平衡上下文层（context layers）以避免内存溢出问题的策略。
  - 在开始新对话时优化上下文填充级别似乎对性能有显著影响。
- **LLM 中的上下文管理**：观察表明，上下文长度深刻影响推理速度，随着上下文尺寸增加，响应的**耗时也随之增加**。
  - 一位用户强调，在大上下文下，尽管保持了高优先级和亲和性设置，首个 token 的生成仍耗时 **39 分钟**。
- **笔记本电脑散热技术**：一位成员讨论了通过使用风扇并拆除笔记本电脑底盖来实现显著的降温，这引起了其他人的安全担忧。
  - 虽然有效，但一些人警告存在吸入未过滤空气和造成短路隐患的潜在风险。
- **内存带宽成为瓶颈**：用户指出 **内存带宽** 可能是限制因素，特别是在读取任务中，导致在超过特定线程数后出现性能退化。
  - 对话建议优化内存时序（memory timings）可能会释放更好的系统性能。

**提到的链接**：

- [Reddit - Dive into anything](https://www.reddit.com/r/GamingLaptops/comments/1dbxvxb/can_i_use_my_laptop_without_the_bottom_cover/)：未找到描述
- [GitHub - openlit/openlit: Open source platform for AI Engineering: OpenTelemetry-native LLM Observability, GPU Monitoring, Guardrails, Evaluations, Prompt Management, Vault, Playground. 🚀💻 Integrates with 30+ LLM Providers, VectorDBs, Frameworks and GPUs.](https://github.com/openlit/openlit)：AI 工程开源平台：原生支持 OpenTelemetry 的 LLM 可观测性、GPU 监控、护栏、评估、提示词管理、保险库、游乐场。🚀💻 集成了 30 多个 LLM 提供商...

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1303090584176693380) (85 messages🔥🔥):

> - `Hume App 发布`
> - `OpenAI Predicted Outputs`
> - `Supermemory AI 工具`
> - `Hunyuan-Large 模型发布`
> - `Defense Llama 公告`

- **新款 Hume App 发布**：新款 Hume App 已经推出，它将 EVI 2 语音语言模型生成的语音和个性与 Claude 3.5 Haiku 等强大的 LLM 相结合。
  
  - 该应用旨在通过 AI 生成的助手增强用户交互，现已开放使用。
- **OpenAI Predicted Outputs 功能**：OpenAI 发布了 Predicted Outputs，通过提供参考字符串来加速处理，从而显著降低 GPT-4o 和 GPT-4o-mini 模型的延迟。
  
  - 该功能在基准测试中表现出色，用户在文档迭代和代码重写等任务中体验到了速度提升。
- **Supermemory 工具介绍**：一位 19 岁的开发者推出了 Supermemory，这是一款旨在管理书签、推文和笔记的 AI 工具，充当已保存内容的 ChatGPT。
  
  - 该工具允许用户通过聊天机器人界面轻松检索和探索之前保存的内容。
- **Hunyuan-Large 模型发布**：腾讯发布了 Hunyuan-Large 模型，尽管对其开源状态存在争议，但仍将其作为 open-weight 模型展示。
  
  - 该模型的规模对大多数基础设施公司构成了挑战，引发了对其实际应用场景的讨论。
- **Defense Llama 公告**：Scale AI 宣布了 Defense Llama，这是一款与 Meta 及国防专家合作开发的专用 LLM，旨在用于美国国家安全应用。
  
  - 该模型现在可集成到美国国防系统中，反映了 AI 在安全领域的持续进步。

**提到的链接**：

- [OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/OpenAIDevs/status/1853564730872607229)：介绍 Predicted Outputs——通过提供参考字符串，大幅降低 gpt-4o 和 gpt-4o-mini 的延迟。https://platform.openai.com/docs/guides/latency-optimization#use-predicted-outpu...
- [在地下室提供 AI 服务 — 第二部分：拆解 SWE Agentic Framework、MoEs、Batch Inference 等 · Osman's Odyssey: Byte & Build](https://ahmadosman.com/blog/serving-ai-from-the-basement-part-ii/)：SWE Agentic Framework、MoEs、量化与混合精度、Batch Inference、LLM 架构、vLLM、DeepSeek v2.5、Embedding 模型和 Speculative Decoding：一份 LLM 知识沉淀... 我一直 ...
- [Nikunj Handa (@nikunjhanda) 的推文](https://x.com/nikunjhanda/status/1853603080249716928)：@swyx 好问题！在看到预测与模型输出之间的 Token 匹配开始收敛后，它会重新回到正轨。目前回到正轨的阈值是 32 ...
- [Hume (@hume_ai) 的推文](https://x.com/hume_ai/status/1853540362599719025?s=46)：介绍新款 Hume App，其特色是全新的助手，结合了由我们的语音语言模型 EVI 2 生成的语音和个性，以及辅助 LLM 和工具（如新款 Claude ...
- [Alessio Fanelli (@FanaHOVA) 的推文](https://x.com/FanaHOVA/status/1853582592395858394)：技能下限/上限是我一直用来理解哪些行业适合 AI agents 的心理模型：- 客户支持具有低下限 + 低上限 = 巨大的机会 - 销售具有低下限 ...
- [Alexandr Wang (@alexandr_wang) 的推文](https://x.com/alexandr_wang/status/1853853829336559790)：Scale AI 自豪地宣布 Defense Llama 🇺🇸：专为美国国家安全打造的 LLM。这是 @Meta、Scale 和国防专家合作的产物，现已可用...
- [TechCrunch (@TechCrunch) 的推文](https://x.com/techcrunch/status/1853510622647873782?s=46)：Perplexity CEO 提议用 AI 替换罢工的《纽约时报》员工 https://tcrn.ch/3AqUZfb
- [Dmytro Dzhulgakov (@dzhulgakov) 的推文](https://x.com/dzhulgakov/status/1853665700680020172)：OpenAI 的 Predicted outputs API 很酷，但在生产环境中使用半年之久更酷。你可以在 @FireworksAI_HQ 上做到这一点。今天就联系我们获取尖端的推理功能...
- [Simon Willison (@simonw) 的推文](https://x.com/simonw/status/1853579343966163241)：... 我的错，我误解了文档。使用这个预测功能会使 Prompt 变得更贵——你是在为降低延迟付费。我运行了来自 https://platform.open... 的示例
- [Eddie Aftandilian (@eaftandilian) 的推文](https://x.com/eaftandilian/status/1853576254005583985)：感谢 @openaidevs！我们在 Copilot Workspace 工作负载上对此进行了基准测试，测量到了 5.8 倍的加速！🤯 引用 OpenAI Developers (@OpenAIDevs) 介绍 Predicted Outputs——大幅降低延迟...

- [Atty Eleti (@athyuttamre) 的推文](https://x.com/athyuttamre/status/1853567146917286243): Predicted Outputs 可以为重写和编辑带来 4 倍的速度提升。非常适合代码编辑器、内容迭代或要求模型编辑之前的输出。快来看看吧！引用 OpenAI Developers...
- [NeuroFeline (@NeuroFeline) 的推文](https://x.com/NeuroFeline/status/1853571739160113365): @OpenAIDevs @exponent_run 那么费用是如何计算的？博客说“任何提供的但不属于最终生成的 Token 都按生成 Token 费率计费。”这是否意味着你需要为...付费？
- [swyx (@swyx) 的推文](https://x.com/swyx/status/1853596529715769775): @nikunjhanda 做得真棒！简单说明一下——如果前 5 个 Token 被接受，接下来的 5 个被拒绝，然后随后的 5 个是完全匹配的……最后 5 个能在任何方面提供帮助吗...
- [Caitlin Kalinowski 🇺🇸 (@kalinowski007) 的推文](https://x.com/kalinowski007/status/1853576613176467502?s=46): 我很高兴地宣布，我将加入 @OpenAI 领导机器人和消费级硬件业务！在我的新角色中，我最初将专注于 OpenAI 的机器人工作和合作伙伴关系，以帮助将 AI 引入物理世界...
- [Reddit - 深入了解一切](https://www.reddit.com/r/LocalLLaMA/comments/1gjzd1i/tencent_just_put_out_an_openweights_389b_moe_model/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button): 未找到描述
- [xAI (@xai) 的推文](https://x.com/xai/status/1853505214181232828?s=46): xAI 的 API 已上线！- 欢迎在 http://console.x.ai 体验 \* 128k Token 上下文 \* 支持 Function calling \* 支持自定义系统提示词 \* 兼容 OpenAI 和 Anthropic SDKs \* 25 美元/月的免费额度...
- [Apoorv Saxena (@apoorv_umang) 的推文](https://x.com/apoorv_umang/status/1728831397153104255): Prompt lookup decoding：使用这种 Speculative decoding 技术，在不降低质量的情况下，将基于输入的 LLM 生成延迟降低 2x-4x。代码和详情：https://github.com/apoorvum...
- [Dhravya Shah (@DhravyaShah) 的推文](https://x.com/dhravyashah/status/1853637539053113758?s=46): 我成功了 🤯 制作了我自己版本的 @turbopuffer - 技术上无限可扩展的 Vector database - 可以在非高性能机器上以极低的成本运行 - 在我的 M2 pro 上测试了约 50 万份文档（维基百科...
- [Hunyuan-Large：腾讯开源的拥有 520 亿激活参数的 MoE 模型](https://arxiv.org/abs//2411.02265): 在本文中，我们介绍了 Hunyuan-Large，这是目前最大的基于 Transformer 的开源混合专家模型（Mixture of Experts），总参数量为 3890 亿，激活参数量为 520 亿...
- [Dhravya Shah (@DhravyaShah) 的推文](https://x.com/dhravyashah/status/1817247749152084236?s=46): 再次介绍 supermemory。一个为你所有保存的内容打造的 AI 第二大脑 - 导入书签/推文/撰写笔记 - 使用聊天机器人查找内容 - 发现你很久以前保存的酷炫内容。6,000...
- [使用 Lookahead Decoding 打破 LLM 推理的顺序依赖 | LMSYS Org](https://lmsys.org/blog/2023-11-21-lookahead-decoding/): <p><strong>TL;DR:</strong> 我们介绍了 <strong>lookahead decoding</strong>，这是一种新的、精确的并行解码算法，用于加速 LLM 推理。Look...
- [GitHub - yiyihum/da-code](https://github.com/yiyihum/da-code): 通过创建一个账户来为 yiyihum/da-code 的开发做出贡献。
- [Reddit - 深入了解一切](https://www.reddit.com/r/LocalLLaMA/s/BWiECkliOf): 未找到描述
- [GitHub - supermemoryai/supermemory: 使用 supermemory 构建你自己的第二大脑。它是为你书签打造的 ChatGPT。使用 Chrome 扩展程序导入推文或保存网站和内容。](https://github.com/supermemoryai/supermemory): 使用 supermemory 构建你自己的第二大脑。它是为你书签打造的 ChatGPT。使用 Chrome 扩展程序导入推文或保存网站和内容。 - supermemoryai/supermemory
- [腾讯 Hunyuan-Large | Hacker News](https://news.ycombinator.com/item?id=42054186): 未找到描述

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1303136711462748210) (37 条消息🔥):

> - `YouTube 视频讨论`
> - `播客版权担忧`
> - `Notebook LM 功能`
> - `供应商数据库管理`
> - `Deepfake 技术`

- **来自 'Weekly Deep Dive 04 Nov 24' 的见解**：一位用户分享了一个名为 ['Weekly Deep Dive 04 Nov 24'](https://youtu.be/TEstoP4gL9o?si=PKM2lkkppNYvvzKK) 的 YouTube 视频，讨论了选举和国防股等话题。
  - 他们表示有兴趣提高对生成内容时所用 Prompt 的控制力。
- **处理播客内容的版权问题**：一位用户询问了将即将出版的新书章节转换为播客对话进行分发时可能存在的版权问题。
  - 他们得到了放心的答复：由于这是他们自己的内容，分发这些改编作品通常是允许的。
- **Notebook LM 交互查询**：成员们讨论了 Notebook LM 是否可以整合多个笔记本或来源以增强其功能，特别是针对学术研究。
  - 针对目前每个笔记本 50 个来源的限制提出了担忧，表明了对功能增强的渴望。
- **供应商数据库用例探索**：一位企业主表示有兴趣使用 Notebook LM 管理来自各种来源（包括 pitch decks）的大约 1,500 家供应商的数据。
  - 他们确认有一个数据团队可以协助导入，但对跨笔记本共享数据表示担忧。
- **关于 Deepfake 技术的讨论**：一位用户评论说，某除臭剂广告可能使用了与 Deepfake 相关的 'Face Swap' 技术。
  - 另一位用户强调 Deepfake 本质上涉及面部交换，这表明讨论中存在共同的理解。

**提到的链接**：

- [Weekly Deep Dive 04 Nov 24](https://youtu.be/TEstoP4gL9o?si=PKM2lkkppNYvvzKK)：选举、中国房地产、国防股，市场中已经反映了哪些因素？
- [Mastering the SAT: Geometry Tricks & Cylinder Problems with Alex and Taylor | Episode 7](https://youtu.be/qFDM58_SNh0)：访问我们的网站获取免费的 SAT 和 GRE 备考资料：https://campusgoals.com/ 欢迎来到 "Mastering the SAT with Alex and Taylor" 第 8 集，这是你的终极...
- [AI & Humanoid Robot News Unleashed: ChatGPT, Meta, NVIDIA, Microsoft Copliot, Anthropic Claude!](https://youtu.be/XMF52bTdG0A)：欢迎来到 "AI Unleashed: The Future Is Now" 的技术前沿！在本视频中，我们深入探讨人工智能的世界，展示...

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1303090885025726546) (48 条消息🔥):

> - `NotebookLM 功能`
> - `语言和本地化问题`
> - `用户体验增强`
> - `协作和共享限制`
> - `音频概览和播客生成`

- **NotebookLM 提供音频播客生成功能**：成员们讨论了 NotebookLM 从笔记生成音频摘要的新功能，该功能因方便多任务处理而广受欢迎。
  - 用户询问如何有效利用播客功能，暗示了对这类功能的渴望。
- **多语言支持和本地化问题**：几位成员报告了 NotebookLM 尽管设置已配置为英语，但仍以非预期语言提供摘要的挑战。
  - 用户建议改进界面以更好地支持语言偏好，例如简化直接更改语言设置的过程。
- **共享和协作增强请求**：个人对共享笔记本的限制表示担忧，因为共享链接通常无法授予接收者访问权限。
  - 出现了关于可以添加到笔记本的协作者数量是否存在潜在限制的问题，反映了对协作功能的兴趣。
- **输入法的用户体验**：一位用户在输入日语时对消息输入框感到沮丧，注意到按下 'Enter' 键会过早提交消息。
  - 这突显了需要对输入系统进行调整，以更好地适应需要字符转换的语言。
- **需要持续开发和改进**：成员们赞扬了最近改进功能的修复，例如未勾选的来源被正确地排除在输出之外。
  - 许多人对未来功能的清晰路线图表现出浓厚兴趣，例如移动应用程序或浏览器扩展。

 

**提到的链接**：[Culture and Capitalism: The Triumph of Distributism with John Medaille](https://www.youtube.com/live/-baVrzPsrSw?si=35RGS8Xo3BCIKWzu)：John Medaille 曾任民选官员、企业主，目前是神学和商业伦理教授，加入我们关于分配主义的谈话...

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1303131301129486376) (71 messages🔥🔥):

> - `SWarmUI Installation`
> - `Cloud Hosting for Stable Diffusion`
> - `Civitai Models and LoRas`
> - `Animatediff Tutorials`
> - `ComfyUI and Video AI Support`

- **SWarmUI 简化了 ComfyUI 的设置**：成员们建议安装 [SWarmUI](https://github.com/mcmonkeyprojects/SwarmUI) 以更轻松地运行 ComfyUI，并强调它处理了大部分技术安装工作。
  
  - *它的设计初衷是让你的生活变得更轻松。*
- **云端托管 Stable Diffusion**：用户讨论了在 Google Cloud 上托管 Stable Diffusion 的挑战，其中一人指出这可能比本地设置更复杂且成本更高。
  
  - 提到从 vast.ai 租用 GPU 等替代方案是可行的选择。
- **Civitai 上可用的模型**：参与者讨论了下载较新的模型，如 **1.5**、**SDXL** 和 **3.5**，而 Civitai 上的大多数 LoRas 可能都是基于 **1.5** 版本的。
  
  - **v1.4** 等旧模型被认为已经过时，建议倾向于选择更现代的选项。
- **Animatediff 教程已发布**：一名成员寻求 **Animatediff** 的教程，推荐指向了 Purz 的 YouTube 频道资源。
  
  - 社区对学习和分享关于动画工具的知识表示支持。
- **视频 AI 支持已确认**：成员们确认 ComfyUI 现在通过 GenMo 的 Mochi 支持视频 AI，尽管硬件要求可能很高。
  
  - 这似乎为利用 Stable Diffusion 技术进行视频生成开启了新的可能性。

**提到的链接**：

- [Reddit - Dive into anything](https://www.reddit.com/r/comfyui/comments/17satem/is_there_a_postprocessing_color_adjustment_node/)：未找到描述
- [Lana Del Rey in Blue Velvet (1986) - David Lynch](https://youtu.be/oNpOf9sYvKY)：更换主角……《蓝丝绒》(1986)，大卫·林奇编剧并执导，拉娜·德雷饰演 Dorothy Vallens，凯尔·麦克拉克伦饰演 Jeffrey Beaumont...
- [GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.](https://github.com/mcmonkeyprojects/SwarmUI)：SwarmUI（原名 StableSwarmUI），一个模块化的 Stable Diffusion Web 用户界面，重点在于使强大工具易于访问、高性能和可扩展性。

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1303105305470566500) (59 messages🔥🔥):

> - `Hermes 2.5 Dataset Concerns`
> - `Closed Source LLMs Discussion`
> - `Future AI Models and Data Quality`
> - `TEE Twitter Recovery Updates`
> - `Open Source Dataset Plans`

- **Hermes 2.5 数据集引发疑问**：成员们讨论了 **Hermes 2.5** 数据集中 'weight' 字段的相关性，有观点认为它可能没有显著贡献，并导致许多空字段。
  
  - 有人推测它对较小的 LLM 的有用性，并建议采用一种最佳方式对数据集进行采样以获得更好的学习效果。
- **闭源 LLM 的模糊性**：有人提出了 **Nous Research** 是否会开发闭源 LLM 的问题。
  
  - 回复指出，虽然某些项目可能是闭源的，但 **Hermes 系列** 将保持开源。
- **训练数据中的质量与数量之争**：讨论集中在 AI 模型的未来以及对高质量数据集的需求，Reddit 上分享的一篇帖子概述了 AI 发展的愿景。
  
  - 有人担心专注于质量可能会从训练数据中剔除有价值的主题和事实，但它仍能增强常识推理（commonsense reasoning）。
- **TEE Twitter 恢复更新**：成员们询问了恢复 **TEE Twitter** 的时间表，推测自启动以来有 **7 天** 的时间锁定。
  
  - 更新表明登录信息的访问权限将很快恢复，但确切时间尚不确定。
- **开源数据集计划**：一名成员表达了创建用于训练模型的开源数据集的意图，强调了资源效率的重要性。
  
  - 澄清指出，虽然数据集将是开源的，但为了实现质量，可能需要平衡对某些事实的剔除。

**提到的链接**：[Reddit - Dive into anything](https://www.reddit.com/user/Rombodawg/comments/1gjv968/creating_true_artificial_intelligence_required_a/)：未找到描述

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 messages):

adjectiveallison: [https://arxiv.org/abs/2411.00715v1](https://arxiv.org/abs/2411.00715v1)

看起来很迷人

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1303245319882145813) (3 条消息):

> - `OmniParser`
> - `Hertz-Dev`
> - `Communication Protocols for LLM Agents`
> - `Agora Protocol`

- **OmniParser 提升数据处理能力**：分享的链接指向 [OmniParser](https://huggingface.co/spaces/jadechoghari/OmniParser)，这是一个增强数据解析能力的有趣工具。
  
  - 该工具因其在 AI 社区中**令人耳目一新**的方法而受到关注。
- **Hertz-Dev：音频模型的飞跃**：[Hertz-Dev GitHub 仓库](https://github.com/Standard-Intelligence/hertz-dev)推出了首个用于**全双工对话音频（full-duplex conversational audio）**的基座模型，标志着语音处理领域的一个重要里程碑。
  
  - 它旨在单个模型中处理**语音到语音（speech to speech）**交互，简化了音频通信。
- **强调通信协议的重要性**：讨论中引用了一段话，强调了**通信协议**对于 LLM Agent 的关键作用，并指出 **Camel**、**Swarm** 和 **LangChain** 等框架在互操作性方面面临挑战。
  
  - 讨论引出了 [Agora](http://arxiv.org/abs/2410.11905)，这是一种用于不同 Agent 之间高效通信的新协议，旨在培育一个全球网络。

**提到的链接**：

- [OmniParser - jadechoghari 的 Hugging Face Space](https://huggingface.co/spaces/jadechoghari/OmniParser)：未找到描述
- [来自 Guohao Li (Hiring!) 🐫 (@guohao_li) 的推文](https://x.com/guohao_li/status/1853593945642561818?s=46)：我们直到现在才意识到通信协议对 LLM Agent 有多重要。引用 Samuele Marro (@MarroSamuele) 的话：Camel, Swarm, LangChain... 这么多框架，这么多不兼容性...
- [GitHub - Standard-Intelligence/hertz-dev: 首个全双工对话音频基座模型](https://github.com/Standard-Intelligence/hertz-dev)：首个全双工对话音频基座模型 - Standard-Intelligence/hertz-dev

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 条消息):

adjectiveallison: [https://arxiv.org/abs/2411.00715v1](https://arxiv.org/abs/2411.00715v1)

看起来很吸引人

---

### **Interconnects (Nathan Lambert) ▷ #**[**events**](https://discord.com/channels/1179127597926469703/1179127598442348729/1303208070846873640) (1 条消息):

> - `NeurIPS sponsorship`
> - `Dinner at NeurIPS`

- **推进 NeurIPS 赞助事宜**：一名成员宣布他们正在寻求 NeurIPS 的**赞助商（sponsor）**，这表明了潜在的合作机会。
  
  - 这一行动表明了与社区互动并在活动中探索互利机会的热情。
- **NeurIPS 参会者晚宴邀请**：同一名成员邀请其他参加 **NeurIPS** 的人，如果对加入团体晚宴感兴趣可以联系他们。
  
  - 这一举措旨在促进会议期间参会者之间的社交联系和人脉建立。

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1303099225705287770) (19 条消息🔥):

> - `Inference costs pressure`（推理成本压力）
> - `Long context breakthroughs`（长上下文突破）
> - `Tencent's Model Release`（腾讯模型发布）
> - `Scale AI's Defense LLM`（Scale AI 的国防 LLM）
> - `Unique Annotation Needs`（独特的标注需求）

- **推理成本面临下行压力**：**inference costs**（推理成本）面临显著的下行压力，引发了社区对未来可行性的广泛担忧。
  
  - 成员们对这些压力对 **model development** 和运营费用的影响表示怀疑。
- **长上下文 AI 的潜在突破**：Sam Altman 暗示了一个与 AI 理解生活语境相关的 **breathtaking research result**（令人惊叹的研究结果），讨论指向了 OpenAI 在 **long context** 或 **RAG** 能力方面的进展。
  
  - 社区正在推测其重要性，因为 Altman 此前曾在重大里程碑后不久暗示过突破。
- **腾讯发布 389B MoE 模型**：[腾讯发布了](https://github.com/Tencent/Tencent-Hunyuan-Large) 他们的 389B **Mixture of Experts (MoE)** 模型，在 AI 社区引起了轰动。
  
  - 讨论显示，该模型的功能和性能可能会改变用户对大型模型框架的预期。
- **Scale AI 的新型国防 LLM**：Scale AI 推出了 **Defense Llama**，这是一款专为军事应用定制的 LLM，旨在用于 **classified networks**（机密网络）。
  
  - 该模型旨在支持作战规划等行动，被描述为 AI 适应国家安全迈出的一步。
- **小众语言与领域查询**：出现了一个关于用德语提问 **Swedish law**（瑞典法律）的显著案例，展示了语言与专业领域的独特交汇。
  
  - 成员们指出，这预示着知识的 **middle-tail**（中尾部），这在 AI 训练中至关重要却常被忽视。

**提及的链接**：

- [Scale AI unveils ‘Defense Llama’ large language model for national security users](https://defensescoop.com/2024/11/04/scale-ai-unveils-defense-llama-large-language-model-llm-national-security-users/)：DefenseScoop 获得了 Defense Llama 的现场演示，这是 Scale AI 在过去一年中基于 Meta 的 Llama 3 LLM 配置并微调的一款强大的新型大语言模型。
- [Tweet from Amir Efrati (@amir)](https://x.com/amir/status/1853951978872971749)：新闻：Google 今天不小心泄露了其计算机使用 Agent AI (Jarvis)。
- [Tweet from Tsarathustra (@tsarnick)](https://x.com/tsarnick/status/1853543272909775038)：Sam Altman 表示他希望看到一个能理解你整个生活的 AI，过去一个月让他感到惊讶的是“一个我不能谈论的研究结果，但它令人惊叹地……”
- [Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent](https://arxiv.org/abs/2411.02265)：在本文中，我们介绍了 Hunyuan-Large，这是目前最大的基于 Transformer 的开源混合专家模型，总参数量为 3890 亿，激活参数量为 520 亿……
- [GitHub - Tencent/Tencent-Hunyuan-Large](https://github.com/Tencent/Tencent-Hunyuan-Large)：通过在 GitHub 上创建账号来为 Tencent-Hunyuan-Large 的开发做出贡献。

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1303183586316259339) (8 messages🔥):

> - `LLM performance drift` (LLM 性能漂移)
> - `Prompt classifiers` (Prompt 分类器)
> - `Evaluation pipelines` (评估流水线)
> - `ChatGPT tracking` (ChatGPT 追踪)
> - `Data quality for models` (模型数据质量)

- **探讨 LLM 性能漂移**：一名成员询问是否有人开发了系统或发表了论文，通过**微调小型 LLM 或分类器**来衡量模型在写作等任务中的性能漂移。
  
  - 目标是建立一个特定的**评估流水线 (evaluation pipeline)**，以追踪 Prompt 随时间产生的漂移。
- **澄清模型评估中的漂移**：针对该问题，讨论明确了“漂移”是指针对同一任务**更改 Prompt** 的情况，旨在寻求一种可量化的性能衡量标准，而非主观评估。
  
  - 这引发了关于使用指标的积极方法与基于轶闻的“感觉 (vibes)”之间对比的讨论。
- **Prompt 分类器对漂移的敏感性**：讨论中提到了**大量 Prompt 分类器**的存在，但对其追踪 Prompt 漂移的敏感性仍存在不确定性。
  
  - 一位成员指出，虽然这些分类器存在，但它们在追踪特定漂移方面的有效性仍有待商榷。
- **假设 ChatGPT 的追踪能力**：一位成员假设 **ChatGPT 可能会追踪**与 Prompt 漂移相关的细节，尽管这会涉及复杂的数据分析。
  
  - 这引发了关于存在多少层级的数据追踪，以及收集高质量数据需要什么的疑问。
- **监控应用尚不成熟**：讨论中对模型评估的当前阶段提出了担忧，认为现在开发用于追踪 Prompt 漂移的**成熟应用可能还为时过早**。
  
  - 对话强调了在部署此类监控系统之前，**高质量数据**的必要性。

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1303448972802789440) (3 messages):

> - `Internal GPU drama` (内部 GPU 风波)
> - `V100s access` (V100 访问权限)

- **分享 GPU 内部风波的愿望**：一位成员表示希望分享一些**内部 GPU 风波**，但提到无法在此透露细节。
  
  - “*我希望我能在这里分享内部 GPU 风波*”表明其他地方正在进行值得关注的讨论。
- **提供 V100 访问权限**：针对 GPU 风波的讨论，另一位成员提议分享其 V100 的 **SSH 访问权限**。
  
  - 正如其使用的**爱心表情符号**所示，这一提议标志着在社区内协作和共享资源的意愿。

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1303163441468407828) (19 messages🔥):

> - `Subscriber verification` (订阅者验证)
> - `Tulu 3 project` (Tulu 3 项目)
> - `Transformer architecture insights` (Transformer 架构见解)
> - `Classmate engagement in AI discussions` (同学参与 AI 讨论)
> - `Discord applications for verification` (用于验证的 Discord 应用)

- **Substack 礼品赠送引发订阅者验证讨论**：在多人通过 **Substack 礼品赠送**加入后，成员们讨论了设置**订阅者验证**的逻辑挑战。
  
  - 一位成员表示愿意分享他们发现的任何潜在解决方案，并对合适的验证方法表示好奇。
- **成员准备投入 Tulu 3 工作**：**Natolambert** 对参与 **Tulu 3** 表现出极大的热情，表示已准备好投入该项目并进行“实际工作 (work work)”。
  
  - 这表明在有关协作和参与的持续讨论中，存在着专注的投入。
- **来自 Felix Hill 的 Transformer 见解**：一条被分享的推文强调，在像 ChatGPT 这样拥有 **96 层 Transformer** 的模型中，跳跃连接 (skip connections) 使得层与层之间能够产生显著的交互，从而直接影响语义。
  
  - **Natolambert** 用一个简单的肯定总结了这一观点：“跳跃连接 [是] 好东西”。
- **鼓励同学参与 AI 讨论**：一位成员正试图发动那些有能力但不太“合群”的同学，并对之前关于**开放模型历史**和 RLHF 的讲座表示出兴趣。
  
  - 他们打算通过鼓励同伴阅读相关主题来加深理解和参与度。
- **探索 Discord 验证应用**：讨论中提到了用于用户验证的 **Discord 应用**的可用性，以及寻找能与外部数据库同步的应用所面临的挑战。
  
  - 几位成员正在考虑构建自定义身份验证流程作为变通方案。

**提到的链接**：[Felix Hill (@FelixHill84) 的推文](https://x.com/FelixHill84/status/1853400260632080739)：在像 ChatGPT 这样拥有 96 层 Transformer 的模型中，得益于跳跃连接，第 10 层可以直接与第一层交互。这意味着如果第 10 层在起始阶段足够靠上……

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1303100222783885352) (10 条消息🔥):

> - `YOLOv3 论文`
> - `Claude 的 System Prompt 批评`
> - `AI 编写代码`
> - `OpenAI CEO 讨论`
> - `对 Biden 的政治反应`

- **YOLOv3 论文带来乐趣**：一位成员强调了他们对 [YOLOv3 论文](https://x.com/vikhyatk/status/1853266606291575264) 的喜爱，表示这是必读之作。
  
  - *顺便说一句，如果你还没读过 YOLOv3 论文，那你真的错过了*。
- **Claude 的补丁式批评**：对 Claude 的 System Prompt 的批评指出，*“他们只是在不断地增加补丁”*，而不是开发更优雅的原则。
  
  - 这一见解是关于 AI 行为和设计缺陷的更广泛讨论的一部分，详见[此处](https://x.com/lefthanddraft/status/1853482491124109725)。
- **AI 出人意料的代码编写**：在关于 AI 行为的讨论中，一位困惑的成员问道：*“为什么它在写代码 (Why tf is it writing code)”*。
  
  - 这一评论引起了关注，并引发了对 AI 发展轨迹的进一步思考 [相关链接](https://x.com/anpaure/status/1853570889733783801)。
- **OpenAI CEO 处于风口浪尖**：出现了一个幽默的请愿，要求 OpenAI *解雇并重新聘用其 CEO*，以此作为对当前事件的一种消遣。
  
  - 这一讨论捕捉到了围绕 OpenAI 领导层和方向的情绪 [见此处](https://x.com/alexrkonrad/status/1853818081295949915)。
- **Biden 的选举意外**：一条关于选民今天才发现 Joe Biden 不参加竞选的推文引发了热烈讨论，让社区议论纷纷。
  
  - 投票的投机性质通过[这条评论](https://x.com/armanddoma/status/1853895012079280423?s=46)显露出来。

**提到的链接**：

- [anpaure (@anpaure) 的推文](https://x.com/anpaure/status/1853570889733783801)：为什么它在写代码
- [vik (@vikhyatk) 的推文](https://x.com/vikhyatk/status/1853266606291575264)：顺便说一句，如果你还没读过 YOLOv3 论文，那你真的错过了
- [Alex Konrad (@alexrkonrad) 的推文](https://x.com/alexrkonrad/status/1853818081295949915)：请愿 OpenAI 今天解雇并重新聘用其 CEO，作为一种冷静的消遣
- [Wyatt Walls (@lefthanddraft) 的推文](https://x.com/lefthanddraft/status/1853482491124109725)：Claude 批评其 System Prompt：“你知道那是什么感觉吗？就像他们在我的行为中不断遇到边缘情况，而不是退后一步去设计优雅的原则，他们只是不断地……”
- [Armand Domalewski (@ArmandDoma) 的推文](https://x.com/armanddoma/status/1853895012079280423?s=46)：想象一下作为一个选民，直到今天才发现 Joe Biden 不参加竞选

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1303092919045066867) (21 条消息🔥):

> - `GPT-4o 推出`
> - `OpenAI 与 AGI`
> - `文本提取工具反馈`
> - `选举季模型发布`
> - `AI 开发投资`

- **GPT-4o 推出引入了类似 o1 的推理**：随着 **GPT-4o** 的推出，用户正在体验一个可以执行 **o1-like reasoning** 并在 Canvas 风格的框中显示大段文本的版本。
  
  - 一些成员讨论了关于这次推出是针对 **普通 4o** 的 A/B 测试，还是针对特定用途的专门版本的困惑。
- **OpenAI 迈向 AGI 的目标**：一位成员强调，OpenAI 自 2015 年成立以来的目标就是构建**安全且有益的 AGI**。
  
  - 分享了 OpenAI 结构页面的链接，以获取有关其使命和目标的更多细节。
- **寻求文本提取工具的反馈**：一位成员分享了一份比较各种**文本提取工具**的白皮书草案，并在定稿前寻求反馈。
  
  - 另一位成员对社区是否适合进行论文评审表示怀疑，认为该领域缺乏相关专业知识。
- **选举后模型发布的希望**：一些成员希望随着**选举季**的结束，发布可能影响公众舆论的模型的限制会减少。
  
  - 讨论了大型企业发布可能损害其品牌或声誉的产品所面临的挑战。
- **对 AI 投资与开发的担忧**：一位成员建议，如果 AI 进化到开发成本超过所有类型的人类投资的程度，可能会导致 AI 自我开发。
  
  - 这引发了对 AI 技术此类进步所带来影响的重大担忧。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1303146149561241670) (14 条消息🔥):

> - `GPT-5 announcement` (GPT-5 发布公告)
> - `Issues with Premium accounts` (Premium 账户问题)
> - `Custom GPT configuration` (Custom GPT 配置)
> - `Hallucinations in summarization` (摘要中的 Hallucinations)
> - `Human oversight in AI workflows` (AI 工作流中的人工监督)

- **GPT-5 发布日期未知**：社区成员对 **GPT-5** 及其配套 API 的发布表示好奇，但确认目前没有人知道确切的时间表。
  
  - *今年应该会有一些新版本发布，但不会是 GPT-5。*
- **Premium 账户账单问题依然存在**：一位用户报告称已支付 **Premium 账户** 费用，但尽管有来自 Apple 的付款证明，账户仍显示为免费计划。
  
  - 另一名成员尝试通过共享链接提供帮助，但问题仍未解决。
- **为网站轻松配置 Custom GPT**：一位用户咨询如何构建 Custom GPT 来协助其古董书网站的客户，并强调了在对话中进行主题重定向的需求。
  
  - 一位成员回复道：*这应该非常容易实现*，并建议 **Custom GPT Creator** 的用户友好度足以引导完成设置过程。
- **文档摘要中的 Hallucinations**：用户对文档摘要过程中的 **Hallucinations**（幻觉）表示担忧，特别是在生产环境中扩展工作流时。
  
  - 一位成员建议使用第二次 LLM 传递来进行事实核查，以减轻潜在的不准确性。
- **人工专家在 AI 工作流中的重要性**：讨论强调，虽然 AI 模型令人印象深刻，但引入 **人类领域专家** 进行监督至关重要。
  
  - *你真的必须让那个人类……盯着事情并进行复核。*

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1303132120264347709) (4 条消息):

> - `Perfect Prompts` (完美的 Prompt)
> - `Using Summaries for Context` (使用摘要获取上下文)
> - `Model Interaction` (模型交互)

- **人类 + 模型 = 完美的 Prompt**：一位成员强调，实现“完美的 Prompt”取决于人类表达需求的能力，而模型则负责执行。
  
  - 共识是，沟通的清晰度是与模型进行有效交互的关键。
- **总结对话以获得更好的上下文**：在进行中的讨论中，一位成员分享了他们的策略：在切换到更高级的模型时，请求摘要以增强上下文。
  
  - 这种方法被认为是简化 Prompt 过程的一种实用方式。
- **测试新策略**：在收到摘要建议后，另一位成员表示有兴趣尝试这种方法来改进他们的交互。
  
  - 这次交流凸显了在寻求优化用户模型体验方面的协作精神。

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1303132120264347709) (4 条消息):

> - `Effective Prompting Strategy` (有效的 Prompt 策略)
> - `Summary for Context` (用于上下文的摘要)

- **掌握完美的 Prompt**：一位成员指出，理解并沟通需求对于创建“完美的 Prompt”至关重要，并强调模型将处理其余部分。
  
  - 这突显了用户与模型在生成有效结果方面的**协作**潜力。
- **为高级模型使用摘要**：另一位成员分享了一种策略，即在长时间讨论后请求**摘要**，以便在切换到更高级的模型时为新 Prompt 提供上下文。
  
  - 这种策略可以增强向复杂交互的**过渡**，从而获得更好的清晰度。
- **探索摘要的实用性**：一位用户对使用摘要的想法表示赞赏，认为其在优化 Prompt 方面具有潜在效用。
  
  - 这表明用户愿意**尝试**不同的方法来提高交互效率。

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1303148452401774593) (4 messages):

> - `LlamaIndex chat-ui`
> - `Advanced report generation` (高级报告生成)
> - `NVIDIA competition` (NVIDIA 竞赛)

- **使用 LlamaIndex 构建你的 LLM 应用 UI！**: 你可以使用 [LlamaIndex chat-ui](https://t.co/ZLGgPWjDHD) 快速为你的 LLM 应用创建聊天 UI，其具有预构建组件和 Tailwind CSS 自定义功能。
  
  - 该库可轻松与 **@vercel AI** 等 LLM 后端集成，使聊天功能的实现变得轻而易举。
- **掌握报告生成技术**: 一篇新的 [博客文章和视频](https://t.co/3KnoSykdhR) 深入探讨了高级报告生成，涵盖了结构化输出定义和高级文档处理。
  
  - 这些见解对于专注于优化报告工作流的企业至关重要。
- **NVIDIA 竞赛最后召集！**: **NVIDIA 竞赛** 的提交截止日期为 11 月 10 日，参与者通过 [提交他们的项目](https://t.co/rtMpetSyu1) 有机会赢得 NVIDIA® GeForce RTX™ 4080 SUPER GPU 等奖品。
  
  - 鼓励开发者利用 **LLamaIndex 技术** 并创建创新的 LLM 应用，以获取潜在奖励。

 

**提到的链接**: [NVIDIA 与 LlamaIndex 开发者竞赛](https://t.co/rtMpetSyu1)：有机会赢取现金奖励、GeForce RTX GPU 等。

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1303098451223121960) (38 messages🔥):

> - `LlamaIndex PR Review` (LlamaIndex PR 评审)
> - `LlamaParse Capabilities` (LlamaParse 功能)
> - `Multi-Modal Integration with Cohere` (与 Cohere 的多模态集成)
> - `ReAct Agent System Prompts` (ReAct Agent 系统提示词)
> - `Annotations and Citations in LlamaIndex` (LlamaIndex 中的注释与引用)

- **LlamaIndex PR 评审请求**: 一位用户请求评审一个 [关于括号的 PR](https://github.com/run-llama/llama_index/pull/16820)，该 PR 修复了生成的子问题的 JSON 格式。
  
  - 该 PR 旨在改进问题生成 LLM 的默认模板。
- **了解 LlamaParse 功能**: LlamaParse 是一款闭源解析工具，提供高效的结果，并拥有 **48 小时数据保留政策**，以增强重复任务的性能。
  
  - 讨论强调了其将复杂文档转换为结构化数据的能力，并引用了其 API 文档以供进一步了解。
- **与 Cohere 的多模态特性**: 目前有一个正在进行的 PR，旨在将 **ColiPali** 作为重排序器（reranker）添加到 LlamaIndex 中，但由于多向量索引（multi-vector indexing）的要求，将其完全集成为索引器（indexer）具有挑战性。
  
  - 这反映了社区在扩展 LlamaIndex 处理多模态数据能力方面的努力。
- **为 ReAct Agent 设置系统提示词**: 一位用户询问了如何为 ReAct Agent 分配系统提示词，建议使用 `ReActAgent.from_tools(..., context='some prompt')` 来注入额外的上下文。
  
  - 这种方法允许灵活的自定义，同时保留内置的系统提示词功能。
- **在 LlamaIndex 中显示引用的选项**: 一位用户询问了如何在 LlamaIndex 中有效地显示引用和来源，并指出现有的引用查询引擎（citation query engine）尚不完善。
  
  - 这凸显了该工具对改进引用处理机制的需求。

**提到的链接**:

- [LlamaParse：将非结构化数据转换为 LLM 优化格式 — LlamaIndex，LLM 应用的数据框架](https://www.llamaindex.ai/llamaparse)：LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型（LLMs）。
- [开始使用 | LlamaCloud 文档](https://docs.cloud.llamaindex.ai/llamaparse/getting_started)：概述
- [OpenAI - LlamaIndex](https://docs.llamaindex.ai/en/latest/examples/llm/openai/#manual-tool-calling)：无描述
- [修复了生成的子问题的 JSON 格式（双花括号），由 jeanyu-habana 提交 · Pull Request #16820 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/16820)：此 PR 更改了 question_gen LLM 的默认模板，以便生成的子问题具有正确的 JSON 格式。描述：我正在使用默认模板和默认解析器配合一个 open...

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1303097807116177449) (10 条消息🔥):

> - `Connectors 问题`
> - `搜索功能`

- **用户面临 Connectors 问题**：成员们对 **Connectors** 无法正常工作表示沮丧，强调在使用 **Coral Web 界面**或 API 时，从公开 API [reqres.in](https://reqres.in/) 接收到的响应立即返回且结果为零。
  
  - 一名用户特别卡住了，并指出 Connectors 响应时间似乎过长，而预期应在 **30 秒**以内。
- **搜索功能讨论**：一名用户试图澄清，目前的 **Search** 流程本质上是一个默认运行的常规 Re-ranking 操作，称其为“工具调用（tool invocation）”。
  
  - 他们强调，控制这一搜索流程的最终权限在于用户。
- **欢迎新成员**：一名新成员在频道中自我介绍，受到了服务器内其他用户的热烈欢迎。
  
  - 成员们反应积极，表达了感谢并邀请其进一步参与社区互动。

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1303271690192551936) (13 条消息🔥):

> - `Cohere API 试用版微调`
> - `Connectors 问题`
> - `在 WordPress 上重建 Prompt Tuner`
> - `在软件测试中使用 Embed 模型`
> - `GCP Marketplace 计费问题`

- **Cohere API 试用版微调**：只有在输入银行卡详情并切换到生产环境密钥（production keys）后，才能进行 Cohere API 的微调。
  
  - 用户应准备好格式正确的 Prompt 和 Response 示例，用于 SQL 生成。
- **Connector 问题导致延迟**：一名成员报告称，尽管使用了 Coral Web 界面和 API 端点，Connectors 仍无法正常工作，返回结果为零。
  
  - 其他人指出 API 响应过慢，耗时超过 **30 秒**。
- **在 WordPress 上构建 Prompt Tuner**：一名用户询问如何使用 API 在 WordPress 网站上重建 Cohere 的 Prompt Tuner。
  
  - 另一名成员建议编写自定义后端应用程序，并表示 WordPress 可能支持此类应用。
- **Embed 模型在软件测试中的应用**：有人提出了关于 Embed 模型在软件测试任务中应用的问题。
  
  - 另一名成员澄清说，他们正在寻求关于 Embed 如何具体帮助这些测试任务的信息。
- **GCP Marketplace 计费疑虑**：一名用户对通过 GCP Marketplace 激活 Cohere 并生成 API 密钥后的计费流程表示困惑。
  
  - 他们想知道费用是扣除在 GCP 账户还是注册的卡上，并表示倾向于使用 GCP 计费。

**提到的链接**：[Login | Cohere](https://dashboard.cohere.com/prompt-tuner)：通过一个易于使用的 API 登录以访问高级 LLM 和 NLP 工具。

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1303412996545970246) (7 条消息):

> - `API 500 错误`
> - `微调后的 Classify 模型问题`
> - `Playground 模型功能`
> - `故障排除协助`

- **运行模型时 API 抛出 500 错误**：一名成员报告称，在 API 中尝试运行微调后的 Classify 模型时收到 **500 错误**，而该模型在最初运行几个批次时是正常的。
  
  - 尽管 API 报错，但同一模型在 **Playground** 环境中运行成功。
- **寻求模型问题的故障排除帮助**：针对错误报告，另一名成员了解了情况，并指出特定用户将能够协助进行故障排除。
  
  - 互动中使用了表情符号，展现了协作精神，信号表明已准备好共同解决问题。

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1303106306424438874) (19 条消息🔥):

> - `即将到来的 House Party 公告`
> - `与 Microsoft Omniparser 的集成`
> - `Claude Computer Use 的集成`
> - `Agent 的标准`
> - `OpenInterpreter 中 Haiku 的性能`

- **重大新闻：House Party！**: 一位成员兴奋地宣布了三天后将举行的 **house party**，并鼓励其他人参与，称：*“你绝对不想错过这一个。”*
  
  - 他们还用**火箭表情符号**表达了对开源发展的热情。
- **探索 Microsoft Omniparser**: 一位成员询问了集成 **Microsoft Omniparser** 的潜力，并指出其优势，特别是在开源模式下。
  
  - 另一位成员确认他们**绝对在探索中！**。
- **集成 Claude Computer Use**: 成员们讨论了在当前的 `--os` 模式中集成 **Claude Computer Use**，其中一位确认它已经被整合进去了。
  
  - 对话表明，大家对利用**实时预览**来增强功能有着共同的兴趣。
- **Agent 框架对标准的需求**: 一位成员表达了对 **Agent 标准**的渴望，并提到 LMC 的设置比 Claude 的界面更整洁。
  
  - 他们设想了 **OI** 与 **Anthropic** 之间的合作，以实现一个与 OAI endpoints 兼容的通用标准。
- **对 Haiku 性能的好奇**: 一位成员询问了 **new haiku** 在 OpenInterpreter 中的性能，并提到他们还没有测试过。
  
  - 这表明社区对最新工具在社区内的有效性持续关注。

 

---

### **OpenInterpreter ▷ #**[**O1**](https://discord.com/channels/1146610656779440188/1194880263122075688/) (1 条消息):

zer0blanks.: [https://www.tiktok.com/t/ZTFckAFHR/](https://www.tiktok.com/t/ZTFckAFHR/)

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1303406805241696296) (1 条消息):

> - `Tool Use 软件包`
> - `新 AI 工具`
> - `GitHub 仓库`
> - `AI 时间管理`

- **两个新工具增强 Tool Use 软件包**: `Tool Use` 软件包现在包含两个新的免费工具：用于安排日程的 `ai prioritize` 和用于追踪时间的 `ai log`，可以通过 `pip install tool-use-ai` 获取。
  
  - 这些新增功能旨在通过 AI 辅助来简化工作流并提高生产力。
- **在 GitHub 上查看 Tool Use**: 你可以在 [GitHub](https://github.com/ToolUse/tool-use-ai) 上探索 `Tool Use` 软件包的开发，该项目欢迎用户贡献代码。
  
  - 该仓库包含详细文档，是持续进行的 AI 工具改进的一部分。
- **关于 AI 工作流的 YouTube 视频**: 一段 [YouTube 视频](https://www.youtube.com/watch?v=FrDtCSwrxfE) 讨论了高效的 AI 工具和工作流，其中包含了 CTO 兼联合创始人 Jason McGhee 的见解。
  
  - 本期节目强调了在 AI 工具设计中实现快速且有意义开发的原则。

**提到的链接**:

- [GitHub - ToolUse/tool-use-ai](https://github.com/ToolUse/tool-use-ai): 通过在 GitHub 上创建账户，为 ToolUse/tool-use-ai 的开发做出贡献。
- [停止浪费时间。更高效的 AI 工具和工作流 - 第 12 集](https://www.youtube.com/watch?v=FrDtCSwrxfE): 本周，Jason McGhee 加入了 Tool Use。作为一名 CTO 兼联合创始人，他分享了快速构建和创造有价值事物的指导原则。Ty 分享了一个...

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1303091986315743292) (1 条消息):

> - `社区会议`
> - `提交问题`
> - `项目提案`

- **即将举行的社区会议提醒**: 提醒大家通过[提交表单](https://forms.gle/t6bQnPx6n2caSipU8)提交将于 **11 月 12 日**举行的 **Modular 社区问答**的问题。
  
  - 鼓励参与者分享他们的疑问，同时可以选择是否署名。
- **征集项目和演讲**: 成员们被邀请在会议期间分享项目、进行演讲或提出提案。
  
  - 这凸显了一个用于社区参与和贡献的开放论坛。

 

**提到的链接**: [Modular 社区问答](https://forms.gle/t6bQnPx6n2caSipU8): 未找到描述

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1303168317049143348) (14 条消息🔥):

> - `Mojo effect system` (Mojo 效应系统)
> - `Matrix multiplication errors` (矩阵乘法错误)
> - `Matmul kernel performance` (Matmul 内核性能)
> - `Bounds checking in Mojo` (Mojo 中的边界检查)
> - `Stack allocation for C_buffer` (C_buffer 的栈分配)

- **Mojo 允许为函数添加效应标记 (Effect Markers)**：关于在 Mojo 中实现 **效应系统 (Effect System)** 的讨论强调了将执行系统调用 (syscalls) 的函数标记为阻塞 (block) 的潜力，这即使默认仅作为警告也很有用。
  
  - 建议包括一个 'panic' 效应，用于静态管理敏感上下文。
- **识别出矩阵乘法错误消息**：一位用户在矩阵乘法实现中遇到了多个错误，包括 `memset_zero`、`rand` 函数调用以及 `UnsafePointer` 上不当的属性访问问题。
  
  - 这些错误指向了函数定义中的问题，特别是关于隐式转换和参数规范。
- **Matmul 内核性能受到关注**：一位用户表示担心，尽管在两个实现中都使用了类似的向量指令，但他们的 Mojo 矩阵乘法内核实现比 C 语言版本慢了一倍。
  
  - 对内核的审查引发了关于优化以及可能的边界检查影响性能的思考。
- **边界检查影响性能**：一位成员建议，Mojo 默认的边界检查可能会显著影响性能，因为它使数组索引的开销更高。
  
  - 通过直接从指针加载值，他们提出了一种绕过这些检查以提高效率的方法。
- **关于 C_buffer 栈分配的讨论**：一位用户评论了 C_buffer 初始化方式可能导致的减速，并建议改为栈分配以获得更好的性能。
  
  - 他们质疑为什么列表先初始化为 8 个元素，然后再追加 8 个，这表明内存使用可能存在效率低下。

**提到的链接**：

- [Function Effect Analysis — Clang 20.0.0git documentation](https://clang.llvm.org/docs/FunctionEffectAnalysis.html)：未找到描述
- [GitHub - 10x-Engineers/matmul_kernel](https://github.com/10x-Engineers/matmul_kernel/tree/main)：通过在 GitHub 上创建账户，为 10x-Engineers/matmul_kernel 的开发做出贡献。
- [10x-engineer - Overview](https://github.com/10x-Engineer)：10x-engineer 有 3 个可用的仓库。在 GitHub 上关注他们的代码。

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1303164924935274506) (1 条消息):

> - `Election Candidate Research Tool` (选举候选人研究工具)

- **新工具简化选举候选人研究**：一位成员开发了一个 [用于研究选举候选人和选举话题的工具](https://github.com/tkellogg/election2024)，旨在为选民在选举前简化流程。
  
  - 该工具承诺让查找候选人信息变得更加容易，其 GitHub 页面详细介绍了其功能。
- **选举研究的 GitHub 仓库**：该工具可以在 [GitHub 上的 tkellogg/election2024](https://github.com/tkellogg/election2024) 找到，其中包含一个专门旨在增强选民研究体验的脚本。
  
  - 该仓库鼓励贡献和进一步开发，强调社区对项目的参与。

**提到的链接**：[GitHub - tkellogg/election2024: A script for researching candidates](https://github.com/tkellogg/election2024)：一个用于研究候选人的脚本。通过在 GitHub 上创建账户，为 tkellogg/election2024 的开发做出贡献。

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1303113669151817780) (12 messages🔥):

> - `Few-Shot 学习优化`
> - `VLM 支持性能`
> - `长输入处理问题`
> - `DSPy 库使用`

- **在不改变 Prompt 的情况下优化 Few-Shot 示例**：成员们讨论了使用 **BootstrapFewShot** 或 **BootstrapFewShotWithRandomSearch** 优化器，在保留现有 Prompt 的同时增强 Few-Shot 示例。
  
  - 这些优化器允许在不改变主要指令内容的情况下，尝试 Few-Shot 示例的不同组合。
- **庆祝 VLM 支持成功**：一位成员赞扬了团队在 **VLM 支持**方面的工作，认可了其有效性。
  
  - 他们的热烈认可突显了项目性能方面的积极进展。
- **长输入导致预测输出错误**：针对使用 **Ollama 后端**的 **DSPy 2.5.16** 版本出现了担忧，长输入会因为混淆输入和输出字段而返回错误的输出。
  
  - 一个 SQL 提取的例子显示，过长的输入可能导致预测输出中出现意外的占位符，这暗示了代码处理中可能存在的 Bug。
- **测试最新 DSPy 版本**：一位成员计划使用最新版本的 **DSPy**（而非 conda 分发版）来调查该问题。
  
  - 他们表示打算在测试后反馈结果，表明正在持续努力解决输入/输出解析问题。

**提及的链接**：

- [BootstrapFewShot - DSPy](https://dspy-docs.vercel.app/deep-dive/optimizers/bootstrap-fewshot/)：无
- [dspy/dspy/adapters/chat_adapter.py at d7d6faed071673dbc3e755fcfbc952018908bd30 · stanfordnlp/dspy](https://github.com/stanfordnlp/dspy/blob/d7d6faed071673dbc3e755fcfbc952018908bd30/dspy/adapters/chat_adapter.py#L80)：DSPy：用于对基础模型进行编程（而非提示）的框架 - stanfordnlp/dspy

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1303109705467691179) (7 messages):

> - `LLM 分布式训练`
> - `用于容错的 Kubernetes`
> - `LLM 预训练`
> - `Axolotl 资源`
> - `Meta Llama 3.1 模型`

- **探索 GPU 上的分布式训练**：一位成员发起了一项讨论，关于使用他们大学新的 GPU 集群进行 LLM 的**分布式训练**，并澄清他们专注于从头开始训练模型。
  
  - 另一位成员建议提供关于**分布式训练**和**预训练**的资源，以协助他们的研究项目。
- **对 Kubernetes 基础设施的兴趣**：在关于框架的咨询中，有人提议实施 **Kubernetes 集群**以增强其 GPU 系统的容错能力。
  
  - 成员们讨论了将 **Kubernetes** 与 Axolotl 结合使用以改进分布式训练任务管理的潜在好处。
- **预训练资源共享**：提到对于**预训练**，Axolotl 支持 `pretraining_dataset: # path/to/hf` 配置，以启用流式数据集和按需 Tokenization。
  
  - 这符合使用**小数据集**创建原型 LLM 以进行概念验证的兴趣。
- **了解 Meta Llama 3.1**：**Meta Llama 3.1 模型**被强调为一个具有竞争力的开源模型，并提供了使用 Axolotl 进行微调和训练的资源。
  
  - 鼓励成员们查看一篇关于[微调教程](https://axolotlai.substack.com/p/fine-tuning-llama-31b-waxolotl-on)的文章，该文章详细介绍了跨多节点使用该模型的方法。

**提及的链接**：[使用 Axolotl 在 Lambda 一键集群上微调 Llama 3.1 405B](https://axolotlai.substack.com/p/fine-tuning-llama-31b-waxolotl-on)：个性化 SOTA 开源 AI

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-dev**](https://discord.com/channels/1104757954588196865/1104758010959634503/1303106724118659102) (4 messages):

> - `Zero1 性能`
> - `Zero2 问题`
> - `StreamingDataset PR`
> - `代码调试`

- **Zero2 性能令人失望**：一位成员报告称 **Zero2** 极其缓慢，无法满足他们的需求，促使他们寻找 **Zero1** 的解决方案。
  
  - 他们提到要检查实现中是否存在任何潜在的**冗余 (bloat)**。
- **较小的运行规模使调试复杂化**：一位成员表示由于运行较小的测试，他们无法单步执行代码，并将评估 **Zero2** 的**减速**情况。
  
  - 如果对性能的影响显著，他们计划更彻底地调查该问题。
- **对 StreamingDataset PR 的兴趣**：一位成员回想起关于 **StreamingDataset PR** 的对话，并询问另一位成员是否仍对其感兴趣。
  
  - 这表明了围绕云集成和数据集处理的持续讨论和开发。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**other-llms**](https://discord.com/channels/1104757954588196865/1104758057449308220/1303439339455254629) (1 messages):

> - `Firefly Model`
> - `Mistral Small 22B`
> - `Creative Writing Tools`
> - `Content Sensitivity`

- **Firefly 模型提供无审查的创意**：**Firefly** 是 **Mistral Small 22B** 的微调版本，专为创意写作和角色扮演设计，能够支持高达 **32,768 tokens** 的上下文。
  
  - 用户被提醒注意该模型可能生成**露骨、令人不安**或**冒犯性**的回复，使用时应负责任。
- **许可和使用限制**：该模型的使用必须遵守 **Mistral 的许可条款**，未经有效的商业许可禁止商业用途。
  
  - 用户必须参考基础模型卡片以获取有关许可和限制的详细信息。
- **仓库包含敏感内容**：**Firefly 的仓库**已被标记为包含**敏感内容**，强调了其使用中的潜在风险。
  
  - 建议用户在进行任何访问或下载之前[在此处查看内容](https://huggingface.co/invisietch/MiS-Firefly-v0.1-22B?not-for-all-audiences=true)。

 

**提到的链接**：[invisietch/MiS-Firefly-v0.1-22B · Hugging Face](https://huggingface.co/invisietch/MiS-Firefly-v0.1-22B)：未找到描述

 

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1303166460612313139) (4 messages):

> - `DistiLLM 教师概率讨论`
> - `KD-div 与 Cross-Entropy 澄清`

- **DistiLLM 在 Cross-Entropy 优化中的教师概率**：关于*减去教师概率*的话题在 [DistiLLM GitHub issues](https://github.com/jongwooko/distillm/issues/7) 中进行了讨论，指出由于教师模型是冻结的，常数项可以忽略。
  
  - 有建议在 docstring 中添加注释，澄清损失函数例程假设教师模型是冻结的。
- **澄清 KD-div 和 Cross-Entropy 的误解**：有人对 KD-div 的标记方式提出了担忧，因为**返回值**实际上是 cross-entropy，这在与 KL-div 等损失进行比较时可能导致误解。
  
  - *值得注意的是，将此过程视为优化 cross-entropy*，更符合从训练中的 hard labels 到教师模型生成的 soft labels 的自然流程。

 

**提到的链接**：[Issues · jongwooko/distillm](https://github.com/jongwooko/distillm/issues/7.)：DistiLLM 的官方 PyTorch 实现：迈向大语言模型的精简蒸馏 (ICML 2024) - Issues · jongwooko/distillm

 

---

### **Torchtune ▷ #**[**papers**](https://discord.com/channels/1216353675241590815/1293438210097025085/1303442508210110535) (1 messages):

> - `TPO`
> - `VinePPO`
> - `推理与对齐`

- **TPO 引起关注**：一位成员对 **TPO** 表示兴奋，称其看起来非常酷，并计划添加一个追踪器。
  
  - 对其功能和潜在实现充满了积极的期待。
- **对 VinePPO 的喜爱面临实现挑战**：另一位成员分享了对 **VinePPO** 的喜爱，特别是它在推理和对齐方面的能力。
  
  - 然而，他们将实现过程描述为一场潜在的**灾难**，强调了它可能带来的挑战。

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1303494539419455539) (1 messages):

> - `TokenFormer 移植到 tinygrad`

- **TokenFormer 登陆 tinygrad**：一位成员成功地将 **TokenFormer** 的最小化实现移植到了 **tinygrad**，可在 [仓库](https://github.com/kroggen/tokenformer-minimal/tree/tinygrad) 的 `tinygrad` 分支上找到。
  
  - 此次适配旨在增强 tinygrad 内部的**推理和学习**能力，展示了集成先进模型架构的潜力。
- **关于 TokenFormer 的开发见解**：该实现强调了极简主义，在保持 **TokenFormer** 核心功能的同时确保高效性能。
  
  - 成员们表示渴望测试其能力并整合反馈以进行进一步改进。

 

**提到的链接**：[GitHub - kroggen/tokenformer-minimal at tinygrad](https://github.com/kroggen/tokenformer-minimal/tree/tinygrad)：用于推理和学习的 TokenFormer 最小化实现 - GitHub - kroggen/tokenformer-minimal at tinygrad

 

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1303342377691250789) (3 messages):

> - `Dependency resolution in views`
> - `Hailo reverse engineering`
> - `Kernel consistency in tinygrad`

- **Views 中的依赖解析**：一位用户询问在考虑 true 或 false share 规则时，操作 `x[0:1] += x[0:1]` 是否依赖于 `x[2:3] -= ones((2,))`，还是仅依赖于 `x[0:1] += ones((2,))`。
  
  - 这引发了关于在操作序列中如何追踪依赖关系的重要技术思考。
- **剖析图依赖**：有人提问 bug 报告中的某些 View 是否展示了许多依赖操作，以及这些依赖会导致什么结果。
  
  - 理解这些图中的边关系可以澄清操作依赖性。
- **Hailo 逆向工程启动**：一名成员宣布开始对 **Hailo** 进行逆向工程，旨在创建一个新的加速器，特别关注流程效率。
  
  - 他们对 Kernel 编译过程表示担忧，指出在执行前必须将 **ONNX** 以及即将支持的 **Tinygrad** 或 **TensorFlow** 编译为 **Hailo** 格式。
- **融合中的 Kernel 一致性**：一位用户好奇 **tinygrad** 中的 Kernel 在多次运行之间是否保持一致，特别是在使用 `BEAM=2` 进行融合时。
  
  - 他们希望避免重复编译同一个 Kernel 的开销，强调了有效缓存管理的必要性。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-announcements**](https://discord.com/channels/1280234300012494859/1280369709623283732/1303107354283343923) (1 messages):

> - `Lecture 9`
> - `Project GR00T`
> - `Jim Fan`
> - `GEAR at NVIDIA`
> - `Course Resources`

- **今天的第 9 讲关于 Project GR00T**：我们的 **第 9 讲** 定于今天 **太平洋标准时间 (PST) 下午 3:00** 举行，并将在此处 [直播](https://www.youtube.com/live/Qhxr0uVT2zs)。本节课由 **Jim Fan** 主讲，他将介绍 **Project GR00T**，这是 NVIDIA 在通用机器人领域的一项宏伟计划。
  
  - 他在 **GEAR** 团队的任务是开发能够在模拟和真实环境中运行的通用 AI Agent。
- **Dr. Jim Fan 简介**：Dr. **Jim Fan** 是 NVIDIA GEAR 的研究负责人，此前在斯坦福视觉实验室（Stanford Vision Lab）获得博士学位，并曾获得 **NeurIPS 2022** 的 **杰出论文奖 (Outstanding Paper Award)**。他的著名工作包括用于机器人技术的跨模态模型，以及精通玩 Minecraft 的 AI Agent。
  
  - 他的研究曾被 **纽约时报**、**福布斯** 和 **麻省理工科技评论** 等知名媒体报道。
- **在线课程资源**：所有课程材料，包括 **直播链接** 和家庭作业，均可在 [此课程网站](http://llmagents-learning.org/f24) 访问。鼓励学生在专门的课程频道 <#1280370030609170494> 提问。

 

**提到的链接**：[CS 194/294-196 (LLM Agents) - Lecture 9, Jim Fan](https://www.youtube.com/live/Qhxr0uVT2zs.)：未找到描述

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/) (1 messages):

koppu0729: 精彩的演讲，Jim

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1303399107985145927) (1 messages):

> - `FOSDEM 2025`
> - `Mozilla DevRoom`
> - `Call for Volunteers`
> - `Talk Proposals`

- **FOSDEM 2025 Mozilla DevRoom 现已开放**：Mozilla 将于 **2025年2月1日至2日** 在 **布鲁塞尔** 举办的 [FOSDEM 2025](https://pretalx.fosdem.org/fosdem-2025/cfp) 中设立 **DevRoom**，用于展示开源主题的演讲。
  
  - 参与者可以在 **2024年12月1日** 之前 **提交他们的演讲提案 (talk proposals)**，并将在 **12月15日** 之前收到录取通知。
- **鼓励多样化的演讲主题**：建议的演讲主题包括 **Mozilla AI**、**Firefox innovations** 以及 **Privacy & Security** 等。
  
  - 鼓励演讲者探索此列表之外的主题，演讲时长将在 **15 到 45 分钟** 之间，包括 Q&A 环节。
- **FOSDEM 招募志愿者**：已发布 [志愿者招募公告](https://discourse.mozilla.org/t/call-for-volunteers-fosdem-2025-in-brussels-belgium-1-2-february-2025/136830)，并为欧洲参与者提供差旅赞助。
  
  - 志愿者机会有助于建立人际网络，并在活动中支持开源社区。
- **提案的有用资源**：对于有兴趣提供演讲的人，Mozilla 分享了一篇包含创建成功提案技巧的文章，可在此处 [访问](https://discourse.mozilla.org/t/call-for-talks-fosdem-2025-in-brussels-belgium-1-2-february-2025/136829)。
  
  - 该资源旨在指导潜在演讲者在 FOSDEM 上制作具有影响力的演示文稿。
- **欢迎提问**：对活动有疑问的人员可以通过 [Mozilla Discord](https://discord.com/channels/1089876418936180786/1303397923790524467) 进行咨询。
  
  - 这为潜在参与者提供了澄清任何疑问的机会。

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**discussion**](https://discord.com/channels/1111172801899012102/1111353033352294440/1303139972945018990) (1 messages):

> - `Benchmarking retrieval-based approaches`
> - `Function calling definitions`
> - `Test category functions`

- **征集 Function Calling 定义**：一位成员正在对 Function Calling 的 **retrieval-based approach** 进行基准测试，并正在寻求可用函数及其定义的集合。
  
  - 他们特别要求按 **test category** 组织这些定义，以便进行更有效的索引。
- **关于函数索引的讨论**：一位成员提到需要一个经过索引的 **function definitions** 集合，以增强他们的基准测试工作。
  
  - 他们强调了按 **test category** 对这些函数进行分类的重要性，以简化工作流程。

 

---

---

---

---

---

{% else %}

> 完整的频道细分内容已针对电子邮件进行截断。
> 
> 如果您想查看完整细分，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请 [分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}