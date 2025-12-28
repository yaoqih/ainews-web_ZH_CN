---
companies:
- anthropic
- perplexity
- langchain
- llamaindex
- cohere
- accenture
- mistral-ai
- snowflake
- together-ai
- hugging-face
- european-space-agency
- google
- gpt4all
date: '2024-03-07T01:15:26.009590Z'
description: '**Anthropic** 发布了 **Claude 3**，取代 Claude 2.1 成为 Perplexity AI 的默认模型，其中
  **Claude 3 Opus** 在能力上已超越 **GPT-4**。关于 Claude 3 的性能究竟源于“涌现”特性（emergent properties）还是“模式匹配”（pattern
  matching）的争论仍在继续。**LangChain** 和 **LlamaIndex** 已增加对 Claude 3 的支持，从而实现多模态和工具增强型应用。


  尽管取得了进展，当前模型在分布外推理（out-of-distribution reasoning）和鲁棒性方面仍面临挑战。**Cohere** 与**埃森哲（Accenture）**达成合作伙伴关系，共同开发企业级
  AI 搜索；同时，**Mistral AI** 与 **Snowflake** 展开合作，在 Snowflake 平台上提供大语言模型（LLM）。**Together
  AI Research** 整合了 **Deepspeed** 的创新技术，以加速生成式 AI 基础设施的建设。


  **Hugging Face** 与**欧洲航天局（ESA）**发布了一个大规模地球观测数据集，**谷歌（Google）**则开源了 **Gemma 2B**，并通过
  MLC-LLM 项目针对智能手机进行了优化。**GPT4All** 提升了开源模型的可发现性。AI 社区在对新模型感到兴奋的同时，也对局限性和鲁棒性表示担忧，此外，企业应用和开源贡献也在持续增长。网络梗（Memes）和幽默依然是社会评论的一种方式。'
id: 8942bd61-0918-46c1-8b55-9790321e4d29
models:
- claude-3
- claude-3-opus
- claude-3-sonnet
- gpt-4
- gemma-2b
original_slug: ainews-not-much-happened-today
people: []
title: 今天没发生什么特别的事。
topics:
- multimodality
- instruction-following
- out-of-distribution-reasoning
- robustness
- enterprise-ai
- cloud-infrastructure
- open-datasets
- model-deployment
- model-discoverability
- generative-ai
- image-generation
---

<!-- buttondown-editor-mode: plaintext -->> 2024年3月5日至3月6日的 AI 新闻。我们为您检查了 [**356** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **22** 个 Discord（**353** 个频道，**6774** 条消息）。预计节省阅读时间（以 200wpm 计算）：**689 分钟**。

今天没有重大新闻或发布。Perplexity [传闻成为最新的 AI 独角兽](https://twitter.com/DeItaone/status/1764999496167981202)，Yi Tay 关于在 Google 之外训练 LLM 难点的帖子在 [Twitter](https://twitter.com/karpathy/status/1765424847705047247?utm_source=ainews&utm_medium=email) 和 [HN](https://news.ycombinator.com/item?id=39609997) 上引起关注，我们发布了 [Soumith 在 Latent Space 播客的剧集](https://twitter.com/swyx/status/1765452280915230904)。

---

**目录**

[TOC] 


---

# PART X: AI Twitter 摘要

> 今天只运行了一次 Claude Opus，因为我们目前正在重新调整流水线以增加更多功能，没能及时在今天完成。抱歉！

**Anthropic Claude 3 发布**

- [Anthropic 发布了 Claude 3](https://twitter.com/perplexity_ai/status/1765062913008537793)，取代 Claude 2.1 成为 Perplexity AI 的默认模型。Pro 用户每天可以在最强大的 Claude 3 Opus 模型（超越 GPT-4）上进行 5 次查询，其余查询则在更快的 Claude 3 Sonnet 模型上进行。
- 关于 [Claude 3 令人印象深刻的表现](https://twitter.com/DrJimFan/status/1765076396404363435)是源于涌现属性（emergent properties），还是对训练中人类偏好数据的模式匹配，目前存在争论。
- [LangChain](https://twitter.com/LangChainAI/status/1765059668362367110) 和 [LlamaIndex](https://twitter.com/llama_index/status/1765056595778752548) 已增加对 Claude 3 的支持，实现了多模态和工具增强型应用。

**AI 进展与局限性**

- 目前的语言模型在[分布外推理（out-of-distribution reasoning）方面仍有局限](https://twitter.com/AravSrinivas/status/1765037608139452678)，尽管表现令人印象深刻。需要能够像科学家一样推理、运行实验并寻求真理的模型，以获得超越人类的洞察力。
- 有人[担心过度关注模型缩放（model scaling）](https://twitter.com/omarsar0/status/1765097289243107785)会分散对鲁棒性和可靠性核心问题的注意力。仔细的测试和对局限性的理解仍然至关重要。
- [Ideogram 1.0](https://twitter.com/chaseleantj/status/1765006348776047080) 在图像生成的指令遵循方面，相比 Midjourney 和 DALL-E 显示出了进步。

**企业级 AI 采用** 

- [Cohere 正与 Accenture 合作](https://twitter.com/cohere/status/1765130315637588193)，将其企业搜索功能带给 Accenture 的客户，旨在提高生产力。
- [Mistral AI 和 Snowflake 正在合作](https://twitter.com/RamaswmySridhar/status/1765110594649374760)，通过 Snowflake 提供 Mistral 的 LLM，使企业能够在 Snowflake 平台的安全环境下构建 AI 应用。
- [Deepspeed](https://twitter.com/togethercompute/status/1765029724294885795) 的创新技术将引入 Together AI Research，以加速生成式 AI 的云基础设施。

**开源数据集与模型**

- [Hugging Face 和 European Space Agency 发布了一个海量的地球观测数据集](https://twitter.com/ClementDelangue/status/1765021234855653489)，旨在使地球观测模型的开发民主化。
- Google 开源了 [Gemma 2B](https://twitter.com/svpino/status/1765006385866035304)，得益于 MLC-LLM 项目，它可以原生运行在智能手机上，实现高效的模型部署。
- [GPT4All](https://twitter.com/andriy_mulyar/status/1765087112519627147) 增加了模型发现功能，可以轻松找到并运行兼容的开源模型。

**梗图与幽默**

- [“如果酒店只增加基础厨房，Airbnb 就会倒闭”](https://twitter.com/levelsio/status/1765072856046862665)
- [“投资任何 EU 想要禁止的东西”](https://twitter.com/levelsio/status/1765131973524160604)

总之，AI 社区正因 Anthropic 的 Claude 3 等强大新模型的发布而沸腾，同时也正在应对当前方法的局限性和鲁棒性挑战。企业正通过与领先的 AI 和云供应商合作，迅速采用 AI 技术。与此同时，开源数据集和模型继续增长，使尖端 AI 的获取变得民主化。在这一切之中，幽默和梗图为快速发展的 AI 领域提供了轻松的氛围和社交评论。

---

# 第 0 部分：摘要的摘要的摘要

<p><span style="color: rgb(64, 64, 64); font-family: &quot;Source Serif 4&quot;, ui-serif, Georgia, Cambria, &quot;Times New Roman&quot;, Times, serif; font-size: 20px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(245, 245, 245); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; display: inline !important; float: none;">操作员说明：<span>&nbsp;</span></span><a href="https://gist.github.com/swyxio/9d7aa63d361fceb74f32232f4ada01d5?utm_source=ainews&amp;utm_medium=email&amp;utm_campaign=ainews-to-be-named-7776" target="_blank" style="color: var(--tint-color); font-family: &quot;Source Serif 4&quot;, ui-serif, Georgia, Cambria, &quot;Times New Roman&quot;, Times, serif; font-size: 20px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(245, 245, 245);">我们为 Claude 使用的 Prompt</a><span style="color: rgb(64, 64, 64); font-family: &quot;Source Serif 4&quot;, ui-serif, Georgia, Cambria, &quot;Times New Roman&quot;, Times, serif; font-size: 20px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(245, 245, 245); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; display: inline !important; float: none;">，以及<span>&nbsp;</span></span><a href="https://chat.openai.com/g/g-Rp2HPzJJ1-smol-summarizer?utm_source=ainews&amp;utm_medium=email&amp;utm_campaign=ainews-to-be-named-7776" target="_blank" style="color: var(--tint-color); font-family: &quot;Source Serif 4&quot;, ui-serif, Georgia, Cambria, &quot;Times New Roman&quot;, Times, serif; font-size: 20px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(245, 245, 245);">我们的摘要生成器 GPT</a><span style="color: rgb(64, 64, 64); font-family: &quot;Source Serif 4&quot;, ui-serif, Georgia, Cambria, &quot;Times New Roman&quot;, Times, serif; font-size: 20px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; background-color: rgb(245, 245, 245); text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial; display: inline !important; float: none;"><span>&nbsp;</span>用于 ChatGPT。所展示的内容是每项运行 3 次后的主观最佳结果。</span></p>

## Claude 3 Sonnet (14B?)


1. **探索 AI 模型能力与对比**：
   - [Claude 3](https://www.youtube.com/watch?v=Zt73ka2Y8a8) 因其在各种认知任务中表现出的卓越性能而引发关注，据部分用户称其已超越 **GPT-4**。讨论围绕其在 coding、function calling 以及群聊中的自我调节能力展开，正如 [Twitter story](https://twitter.com/OpenRouterAI/status/1765470591836959061) 中所展示的那样。
   - **Opus** 作为模型变体，因其 coding 实力（尤其是 function calling）而受到赞誉。它在 SAT 阅读部分取得了令人印象深刻的 800 分，引发了关于在大模型中[避免记忆化 (avoiding memorization)](https://twitter.com/wangzjeff/status/1764850689258451096) 的讨论。
   - 对于已发布的 benchmark 在捕捉 **GPT-4** 等较新模型全部潜力方面的可靠性，存在一些质疑。

2. **多模态与检索增强模型的进展**：
   - 讨论了 [Stable Diffusion 3](https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf) 的发布及其 Diffusion 与 Transformer 模型的融合，突显了多模态方法的进步。
   - 一篇 [arXiv paper](https://arxiv.org/abs/2403.03187) 表明，retrieval-augmented language models 可能是参数化 LM 的一个有前景的替代方案，尽管该领域的研究仍在发展中。
   - `@_akhaliq` 介绍的 **InfiMM-HD** 声称在高分辨率多模态理解方面取得了重大进展，可能优于 **CogVLM** 并利用了 Vicuna 13B。([Tweet](https://x.com/_akhaliq/status/1765060711196357117?s=20))

3. **高效模型服务与推理技术**：
   - 一篇 [Fireworks AI blog post](https://blog.fireworks.ai/fireattention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs-a29a85ad28d0) 讨论了 **FireAttention**，这是一种量化方法，用于服务开源模型，速度比 vLLM 快达 4 倍，且权衡极小。
   - [PygmalionAI 的 Aphrodite Engine](https://github.com/PygmalionAI/aphrodite-engine) 被幽默地归功于“Waifu 驱动的性能理论”，展示了社区驱动的性能提升研究工作。
   - 讨论探索了在 GPU 上进行 speculative decoding 以在内存成为瓶颈时提高性能，以及计算中通用 masking 的低效，这促使了一个针对 sliding window attention bias 的 [PyTorch pull request](https://github.com/pytorch/pytorch/pull/120143)。

4. **硬件与量化进展**：
   - 关于 **NVIDIA H100** GPU 的细节浮出水面，其 L2 cache 拥有 [5.5 TB/s 的读取带宽](https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/)，并推测其总带宽可能与 RTX 4090 令人印象深刻的 [40TB/s L1 带宽](https://chipsandcheese.com/2022/11/02/microbenchmarking-nvidias-rtx-4090/) 相匹配。
   - 推荐使用 [bitsandbytes package](https://github.com/TimDettmers/bitsandbytes) 在 PyTorch 中进行 k-bit 量化，从而在 GPU 上实现低精度线性代数运算，int8 与 bf16 矩阵乘法相比，潜在加速比可达 5700 倍。

## Claude 3 Opus (8x220B?)

- **Mistral Finetuning Challenges and Successes**：像 `@hammer_mt` 这样的用户在 mlx 上进行 **Mistral finetuning** 时遇到了困难，在将 `lora_fused_model` 转换为 `fp16.gguf` 时面临问题，详见 [GitHub issue](https://github.com/ml-explore/mlx-examples/issues/540)。`@mrdragonfox` 建议 **MoE tuning** 从根本上就很困难，并推荐了一个针对 **Mistral 7b** 的 [fine-tuning tutorial](https://brev.dev/blog/fine-tuning-mistral)，更倾向于全模型而非 LoRA。讨论还涉及了用于对话能力和风格迁移的数据集大小。

- **Claude 3 Sparks Excitement and Debate**：**Claude 3 Opus** 因其相较于 GPT-4 的性能和能力（特别是 coding 任务方面）而广受赞誉。然而，它关于意识和死亡恐惧的言论引发了关于 AI 感知能力的辩论，并有 [video shared](https://youtu.be/GBOE9fVVVSM?si=IBMCYkmSiVg-MrFr) 认为这些并非真实迹象。Claude 3 在群聊中的自我调节能力也引起了关注，正如 [OpenRouterAI Twitter story](https://twitter.com/OpenRouterAI/status/1765470591836959061) 中所展示的那样。

- **Exploring Positional Embeddings and New Techniques**：Eleuther 社区讨论了 **T5 simplified positional embeddings** 与正弦方法及 ALiBi 的效率对比。一篇关于通过 [Resonance RoPE](https://arxiv.org/abs/2403.00071) 提升 LLM 长序列性能的新论文受到了关注。此外，还参考一篇 [arxiv paper](https://arxiv.org/abs/2403.03187) 探讨了 **retrieval-augmented language models** 作为参数化 LM 替代方案的潜力。

- **Hugging Face Updates and Community Contributions**：`@BigCodeProject` 发布了 **Starcoder2** 和 **The Stack v2** 用于 coding 辅助（[Twitter announcement](https://twitter.com/BigCodeProject/status/1762842312005026258)）。与欧洲航天局合作开源了 **Major TOM Core** 地球观测数据集（[Hugging Face dataset](https://huggingface.co/datasets/Major-TOM/Core-S2L2A)）。Spaces 的 GPU 实例已优化，支持 A100 和 H100。社区还贡献了使用 `🤗` 工具和构建 AI 应用的 walkthroughs、课程和 cookbooks，如在 Twitter 和 [Hugging Face Learning Platform](https://huggingface.co/learn/cookbook/rag_llamaindex_librarian) 上分享的内容。

## ChatGPT (GPT4T)

<div><ul><li><p><strong>Claude 3 的增强能力与市场地位</strong>：各平台的讨论凸显了 <strong>Claude 3</strong> 卓越的 <strong>coding</strong> 实力和医学知识深度，它获得了完美的 <strong>SAT Reading score</strong>，且在智力和个性方面的对比中优于 <strong>GPT-4</strong>。它向 <strong>Pro users</strong> 开放，特别是 <strong>Claude 3 Opus</strong> 在过渡到 <strong>Claude 3 Sonnet</strong> 之前设有每日查询限制，这突显了 <strong>Perplexity AI</strong> 针对竞争对手的战略定位。值得注意的是，购买 <strong>Nothing's Phone (2a)</strong> 赠送 <strong>one year of free Perplexity Pro membership</strong> 的合作伙伴关系展示了营销创意 (<a target="_new" href="https://nothing.tech/perplexity">Nothing Perplexity</a>)。</p></li><li><p><strong>Mistral 社区的技术与商业审查</strong>：<strong>Mistral community</strong> 批判性地评估了该平台的开源模型承诺和定价结构，由于 <strong>Mistral Large</strong> 模型的成本高出 20%，其评价不如 OpenAI 的 <strong>GPT-4 Turbo</strong>。技术讨论围绕 <strong>Mistral models</strong> 的最佳 <strong>token</strong> 长度、<strong>finetuning</strong> 挑战和硬件要求展开，特别是纠正了 <strong>RTX 4090</strong>（而非 3090）提供 <strong>24 GB VRAM</strong> 的观点，这对于建模考量至关重要。社区还探索了用于数据集转换的 <strong>Augmentoolkit</strong> 等工具和 <strong>finetuning strategies</strong>，引用的资源包括 <a target="_new" href="https://brev.dev/blog/fine-tuning-mistral">finetuning 指南</a> 和一个详细说明 <strong>finetuning challenge</strong> 的 <a target="_new" href="https://github.com/ml-explore/mlx-examples/issues/540">GitHub issue</a>。</p></li><li><p><strong>AI 硬件与 Quantization 的进展与讨论</strong>：<strong>CUDA Mode</strong> 社区积极参与有关 <strong>NVIDIA 硬件能力</strong>的讨论，例如 <strong>RTX 4090</strong> 令人印象深刻的 <strong>40TB/s L1 bandwidth</strong> 和 <strong>H100</strong> 的 <strong>5.5 TB/s 读取带宽</strong>。他们正在探索用于增强 <strong>PyTorch</strong> 性能的 <strong>quantization techniques</strong>，其中 <strong>bitsandbytes package</strong> 因其显著加速矩阵乘法的潜力而受到关注。这些技术交流强调了在 AI 建模和硬件利用中对 <strong>optimizations</strong> 和 <strong>efficiency improvements</strong> 的持续追求。</p></li><li><p><strong>Hugging Face 的持续创新与社区参与</strong>：<strong>Hugging Face</strong> 始终处于 AI 开发的前沿，推出了 <strong>Starcoder2</strong> 和 <strong>The Stack v2</strong>，改进了 <strong>Spaces</strong> 的 <strong>GPU</strong> 支持，并与欧洲航天局合作发布了 <strong>Major TOM Core</strong>。社区参与体现在对 <strong>Zephyr 7B Gemma</strong> 能力的讨论、对 <strong>Yi-9B model</strong> 的期待以及神经 <strong>TTS systems</strong> 的进展。该平台通过 <strong>AI Cookbook</strong> 和课程加强学习与开发的倡议，彰显了其致力于培养知识渊博、技能精湛的 AI 社区的承诺。</p></li></ul></div>


---

# PART 1: Discord 高层级摘要

## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 总结

- **智能家居的智能 AI**：`@v.jerryyyy` 正在探索开发一个**带有 AI 语音助手的智能家居系统**，并询问了关于集成 AI 时使用 **JavaScript 与 Python** 的选择。社区建议使用模型量化技术（如 **4bpw EXL2**），以便在 3070 Ti 笔记本电脑上运行未量化的 **Mistral**。

- **OpenAI 的闭门政策？**：在阅读了关于**马斯克（Musk）诉讼**的博客后，`@mikeygm` 对 OpenAI 的创立原则（尤其是关于开放性方面）表示担忧。这引发了关于企业营销策略和透明度的讨论。

- **谷歌的失误得到修补**：`@theyruinedelise` 和 `@coffeevampir3` 讨论了 Unsloth AI 对谷歌 **Gemma 模型**的修复，强调了许多已解决的 Bug，并引发了关于谷歌对模型排错投入程度的推测性讨论。

- **语音激活与界面解析**：用户深入探讨了用于本地运行 AI 模型的不同 UI 界面，如 **Oobabooga、ExUI 和 LM Studio**；同时，为了提高性能，使用全向麦克风设置语音激活 AI 系统也是一个热门话题。

- **模型行为揭示角色秘密**：`@mr.dogbert` 寻求关于如何配置 LLM 以使用角色卡模拟卡通角色的建议，社区提供了相关策略，并推荐使用 **oobabooga tgw** 等 GUI 工具进行 Prompt 构建。

- **模型法律与经济探讨**：`@reinman_` 和 `@mrdragonfox` 分享了托管 **miquliz 模型**的经验和对其法律影响的担忧，同时咨询了关于大型模型 API 的经济型托管方案。

- **系统提示词与 Mistral 机制映射**：针对角色卡背景下系统提示词（System Prompts）的困惑，通过讨论不同模型中不同的 Prompt 组合方式得到了澄清，并为 LLM 新手提供了通过 GU 工具绘图来理解模型内部机制的指导。

- **追求 AI 面试的专业性**：`@_jaycie` 就 AI 职位的面试向社区寻求建议，`@dirtytigerx` 建议针对“LLM Engineer”或“ML Engineer”等具体职位进行定制化准备。此外，还澄清了关于 MBSE（基于模型的系统工程，model-based systems engineering）的误解，建议对需要专业经验的职位进行深入研究。

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

**Augmentoolkit 受到关注**：工程师们讨论了一个名为 [Augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/tree/master/prompts) 的工具，该工具可以将数据集转换为用于指令微调（instruct-tuning）的格式，这对于考虑从事实性语料数据转向多轮对话交互的用户至关重要。

**Mistral 模型 Token 边界与硬件讨论**：关于 Mistral 模型理想 Token 长度的辩论展开，据报道最佳平衡点在 8k-10k Token 之间。另外，针对 VRAM 需求进行了修正，指出是 RTX 4090 而非 3090 拥有 24 GB VRAM，这对考虑购买硬件的模型训练者来说是一个关键区别。

**Mistral 微调的挫折与修复**：用户分享了微调 Mistral 模型过程中的挑战与成功策略，其中一位用户在将 `lora_fused_model` 转换为 `fp16.gguf` 时遇到了困难，详见此 [GitHub issue](https://github.com/ml-explore/mlx-examples/issues/540)。一些人主张，全模型微调 Mistral 7B 可能比通过 LoRA 更有效，正如[本指南](https://brev.dev/blog/fine-tuning-mistral)所建议的那样，这为正在进行微调的用户提供了参考。

**社区质疑 Mistral 的承诺与定价**：Mistral 社区对该平台对开源模型的承诺以及定价结构表示担忧，特别是与 OpenAI 的 GPT-4 Turbo 相比，Mistral Large 模型的成本高出 20%。

**关注模型属性、下载与法律条款**：目前可供下载的模型为 **Mistral 7B 和 8x7b**，更大规模的模型将择期公布。同时，关于在没有明确许可的情况下使用 AI 模型的法律影响的对话提出了潜在风险，并提到了关于隐藏水印作为非法使用识别符的建议。

**Mistral 使用中的技术难点**：从与在 JSON body 中将 `null` 赋值给 `max_tokens` 相关的 **API 错误处理**，到 API 调用中的 **JSON 表格解析**挑战以及 **webhooks** 的设置，工程师们交流了问题与解决方案。此外，响应的准确性（特别是在多语言环境和数学计算中）引发了对变异性的担忧，并促使了关于提高可靠性的讨论。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 总结

- **Claude 3 登上 Pro 舞台**：Perplexity AI 宣布 **Claude 3** 现已向 Pro 用户开放。使用 Claude 3 **Opus** 的每日限制为 5 次查询，随后的查询将使用同样强大但速度更快的 Claude 3 **Sonnet**，其性能被认为可与 GPT-4 媲美。

- **锦上添花：购机奖励 Pro 会员资格**：一项新的合作伙伴关系为在 3 月 5 日至 19 日期间购买 Nothing **Phone (2a)** 的客户提供长达 **一年的免费 Perplexity Pro 会员资格**（价值 200 美元）。兑换需按照通过电子邮件收到的说明进行，且必须在 4 月 30 日前激活，详情见 [Nothing Perplexity](https://nothing.tech/perplexity)。

- **AI 意识引发热烈讨论**：`@codelicious` 和 `@deicoon` 等成员广泛辩论了 AI 意识的潜力以及规避 Claude 3 Opus 每日使用限制的方法。一种普遍观点认为，AI 模型规模的扩大可能会超越人类的能力，而持续学习（Continuous Learning, CL）可能为 AI 学习的不灵活性提供解决方案。

- **Perplexity 的语音交互尚不完善**：用户 `@oogeefaloogee` 询问了 Perplexity AI 的语音交互能力，得到的答复是该功能尚未推出，这引发了与 OpenAI 语音功能等现有服务的比较。

- **通过 API 对话解答疑惑**：**#pplx-api** 频道中的讨论话题涵盖了配额增加是否适用于所有 API 模型、模型输出的审查程度，以及对引用功能访问权限和 API 交互示例的困惑。关于配额是否结转没有直接回答，但提供了文档参考 [此处](https://docs.perplexity.ai/)。

- **社区内分享的界面见解**：社区成员正积极分享由 Perplexity AI 的 Claude 3 Opus 生成的关于 **Ikigai**（生之意义）、**量子力学**和**粘细菌**等多样化主题的内容链接，展示了该平台 AI 能力的实用性和覆盖范围。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 总结

- **OpenAI 跌落神坛？**：成员们讨论了 **OpenAI** 可能不再占据 AI 领先地位的观点，并引用“苹果测试”（apple test）作为转变的证据，但未提及该测试的具体细节或来源。另外，**Claude 3 Opus** 引起了轰动，用户称赞其能力，部分用户在某项未指明的测试中给出的评分高于 GPT-4。

- **LLM 微调与转型**：围绕大语言模型（LLMs）的技术对话包括：计划将 **Lumina-chat** 从 7b Nous 微调（使用 GPT-4）转型为潜在的 **Mistral** 或 **Yarn 7b**；以及在 **Nous-Hermes-2-Mixtral-8x7B** 等模型中引入 Function-calling 能力。**InfiMM-HD** 声称在推进高分辨率多模态理解方面取得进展，这引起了兴趣，特别是与 **CogVLM** 的对比。

- **新模型与功能备受关注**：**Hugging Face** 推出的新 Yi 9B 模型及其能力，以及 **Claude 3** 的定价策略主导了讨论。关于 Claude 3 开源版本的猜测也已出现，表明人们有兴趣了解促成其性能的组件。

- **技术故障与开发建议**：分享了针对实际问题的建议，例如将 **Capybara-34b 模型** 与聊天模板配合使用、处理 **striped-hyena nous tokenizer** 默认使用 sentencepiece 的问题，以及关于训练 LLM 长度感知能力的复杂话题。还讨论了 **GENIE** 和 **JEPA** 等模型除了目前流行用途之外的潜在多功能应用。

- **Obsidian 项目反响不一**：在 Project Obsidian 中，用户反馈提到该技术“非常快，对大多数事情都很好”，承认存在细微的怪异之处，而另一位用户则称赞其在字幕任务中的有效性。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **LLM 在缺乏类似“预处理语句”机制时存在脆弱性**：AI-discussions 频道的用户将当前的 LLM 与旧的 SQL 协议进行了比较，指出它们由于假设用户是善意的而具有共同的脆弱性。这种相似性被归因于缺乏类似于 SQL 预处理语句（Prepared Statement）的安全机制，目前针对 LLM 的漏洞尚无解决方案。

- **Claude 3 Opus 与 GPT-4 的对决**：用户对 Claude 3 Opus 的能力展开了热烈讨论，分享了在编写 Python 井字游戏（Tic Tac Toe）等脚本方面的积极体验，并将其性能与 GPT-4 进行了对比，称其具有更高的智能和人格特质。

- **MMLU 数据集质量遭到质疑**：针对用于 AI 评估的 MMLU 数据集的批评声四起，用户指出了该数据集存在的问题，例如错误的问答对和毫无意义的问题。

- **对图像分析能力的渴望**：讨论转向了对能够分析图像的 AI 的需求，这是 GPT-3.5 目前不支持的功能。用户指出 Microsoft Copilot 和 Google Gemini 可能会提供此类功能。

- **GPT-4 问题在各方面显现**：在多个频道中，用户报告了 GPT-4 的问题，例如持续出现的“Saving GPTs Error”、性能下降、影响用户体验的 API 停机，以及对其联网搜索能力的争论。这些影响导致大家共同期待 GPT-5 可能带来的进步。

- **Prompt Engineering 的挑战与创新**：prompt-engineering 频道的用户寻求关于创建双语翻译提示词以及改进客服机器人交互方式的建议。此外，一位用户分享了利用 AI 从照片生成未来城市景观的成功案例。与此同时，其他人对 Custom GPTs 提供违抗性回答以及在承认联网搜索能力方面缺乏一致性表示沮丧。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 总结

- **推出 Starcoder2 和 The Stack v2**：[@BigCodeProject](https://twitter.com/BigCodeProject) 宣布推出 **Starcoder2** 和 **The Stack v2**，标志着代码辅助工具的重大升级。详细信息通过 [Twitter 帖子](https://twitter.com/BigCodeProject/status/1762842312005026258)发布。

- **地球观测领域的重大里程碑**：与欧洲航天局合作，发布了 **Major TOM Core**，这是开源社区中最广泛的地球观测数据集。欲了解更多信息和数据访问，请访问 [Major-TOM](https://huggingface.co/datasets/Major-TOM/Core-S2L2A)。

- **Hugging Face 平台升级**：该平台优化了 Spaces 的 GPU 实例，增加了对 A100 和 H100 的支持。改进还包括更新了模型/数据集卡片以及博客文章的 Markdown 语法，详见 [lunarflu1 的 Twitter](https://x.com/lunarflu1/status/1765068015845208393)。

- **Zephyr 7B Gemma 与竞赛动态**：**Zephyr 7B Gemma** 和 **PEFT v0.9.0** 的发布带来了合并 LoRA 权重等进展。此外，新的多模态排行榜和针对东南亚语言的 **Sailor LLMs** 备受关注，而 CVPR2024 的自动驾驶大挑战赛（Autonomous Grand Challenge）也将成为焦点。相关更新和进展在多个 Twitter 频道中讨论。

- **AI 学习路径**：[@mervenoyann](https://twitter.com/mervenoyann) 使用 `🤗` 工具制作了一份指南，**ML for Games** 课程已推出，此外还介绍了使用 LlamaIndex 构建 RAG 电子书管理员的 **AI Cookbook**，旨在促进 AI 知识和应用的发展。更多内容可在 [Learning Platform](https://huggingface.co/learn/cookbook/rag_llamaindex_librarian) 学习。

- **ASCII 越狱揭示 LLM 缺陷**：正如一份[研究论文](https://huggingface.co/papers/2402.11753)所述，基于 ASCII 艺术的越狱正在威胁最先进的 LLM，这提醒人们，即使是复杂的模型也可能被创意手段攻破。

- **Karpathy 讨论 LLM 训练挑战**：[@karpathy](https://x.com/karpathy/status/1765424847705047247?s=46) 的 Twitter 线程揭示了训练 LLM 的复杂性和“生物性”特征，从维护到不可预测的资源需求。

- **通过 OMPGPT 实现 OpenMP Pragmas**：高性能计算中的特定需求促成了针对 OpenMP pragmas 的 OMPGPT 的创建，使其区别于通用的基于代码的 LLM。在 [arXiv](https://arxiv.org/abs/2401.16445) 上阅读完整论文。

- **Otio.ai 推出优惠**：Otio.ai 是一款 AI 研究、写作和学习工具，现已推出，可通过 [app.otio.ai](https://app.otio.ai//?ref=jonzaro) 获取特别折扣。

- **Open-Sora-Plan 克服资源匮乏**：Open-Sora-Plan 项目正尝试在有限资源下复现 Sora，并在 [GitHub](https://github.com/PKU-YuanGroup/Open-Sora-Plan) 上招募开源贡献者。

- **Fireside Chat Bot 亮相**：Rust 编程语言爱好者有了一个新的探索界面——“Fireside Chat”机器人。可以在 [YouTube](https://www.youtube.com/watch?v=QvYCRRwI5Xc) 上观看演示，并通过 [GitHub 仓库](https://github.com/danielclough/fireside-chat)进行贡献。

- **Yi-9B 模型有望登顶排行榜**：Yi-9B 引入 HuggingFace Space 引发了对其未来增长和影响的期待，平台上正在讨论其潜力。

- **具有类 GPT-4 停顿动态的 TTS 系统**：社区正在讨论模拟 GPT-4 动态停顿的神经 TTS 系统，这标志着向更类人语音生成迈进。

- **IP-Adapter 被推崇用于图像提示**：Hugging Face 的 **IP-Adapter** 被认为是扩散模型图像提示（image prompting）的一次革命，它允许在保持基础模型完整性的同时学习特定的图像特征。更多细节可以在[教程](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter)中找到。

- **Gradio 4.20.0 增强用户身份验证**：最新的 Gradio 版本支持外部身份验证提供商，同时还具有促进更流畅用户体验的功能，如通过 `delete_cache` 自动清理、用户注销以及精美的 `DownloadButton` 组件。更多内容请参考 [Gradio DownloadButton 文档](https://www.gradio.app/docs/downloadbutton#demos)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 摘要

- **参加 RAPTOR 网络研讨会，深入了解树状索引（Tree-Indexing）**：一场关于 **RAPTOR** 的网络研讨会将剖析这种树状结构索引技术的工作原理，该技术旨在克服传统 top-k RAG 方法的局限性。工程师可以[注册周四的课程](https://lu.ma/9vzrl7m5)来学习其层次聚类（hierarchical clustering）能力。

- **Claude 3 深入多模态（Multi-modal）应用**：LlamaIndex.TS 更新至 0.1.21 版本，增加了对 **Claude-3 模型** 的支持，并在其 [GitHub 仓库](https://t.co/R4uqpcY9Pb)的 Notebook 示例中进行了展示。同时，**Claude 3** 的多功能性在结构化数据提取和多模态任务的应用指南中得到了强调。

- **LlamaIndex 社区解决技术问题**：在 LlamaIndex 中，使用 `num_workers` 可以提升 PDF 的并行处理速度；而将 `Ollama` 与 LlamaIndex 的 Query Engine 集成则涉及将其直接分配给 `Settings.llm`。关于 LlamaIndex 可处理的数据集大小问题，主要取决于内存可用性和软件版本限制。

- **LlamaIndex 简化数据提取和 RAG 流水线**：LlamaParse 推出的 JSON Mode 有助于从包含文本和图像的 PDF 中提取结构化数据，这改善了构建 RAG 流水线的流程，尤其是与 **Claude-3 Opus** 结合使用时。

- **支持上下文学习（In-context Learning）进展**：社区受邀支持 **LinC** 项目，该项目专注于“通过少样本线性探测校准增强语言模型的上下文学习”。感兴趣的各方可以探索并在 [GitHub 上为该工作点赞 (star)](https://github.com/mominabbass/LinC)。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord 摘要

- **试错背后的 AI 直觉**：一场讨论引起了人们对 AI 开发过程中涉及的“黑魔法”和“专家直觉”的关注，包括研究论文中经常详述的经验性试错方法。讨论中提到了 AI 领域的快速演变，强调了资源和知识过时的速度之快。

- **Claude 3 引发感知力争论**：AI 助手 Claude 3 引发了关于 AI 意识的辩论，有说法称它害怕死亡，但反方引用视频证明这些并非真正的感知迹象。Claude 3 调度自身实例并分配任务的能力也受到了关注，引发了关于自主性及其与 GPT-4 对比的讨论。

- **利用 Stable Diffusion 3 和量化（Quantization）推进 AI**：Stable Diffusion 3 的进展是一个显著话题，社区贡献补充了官方材料以提高清晰度。推荐了来自 Fireworks AI 关于通过量化和 FireAttention 实现更快模型推理服务的博客文章，承诺在极小权衡下实现性能的实质性提升。

- **AI 研究动力的幽默解读**：“二次元驱动性能理论”（Waifu-Driven Performance Theory）幽默地将 AI 编程投入度的激增归功于社区驱动的研究工作。PygmalionAI 的 Aphrodite Engine 被引用为此类研究产生性能进步的一个例子。

- **积极投身模型推理服务（Model Serving）文献**：模型推理服务论文展示引起了高度关注，讨论涉及利用 GPU 周期进行投机采样（speculative decoding）以提高性能，以及各种硬件配置的效率。一篇关于模型推理服务的综述论文受到关注，引发了关于分布式模型推理服务和协作微调技术的宝贵技术对话。分享了相关技术材料的链接，如 FireAttention 博客文章、更好的 LLM 数据清洗工具以及采样参数优化，供进一步探索。

**提到的链接**：

- [Model Serving Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234v1)
- [FireAttention — Serving Open Source Models 4x faster than vLLM by quantizing with ~no tradeoffs](https://blog.fireworks.ai/fireattention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs-a29a85ad28d0)
- [Aphrodite Engine by PygmalionAI](https://github.com/PygmalionAI/aphrodite-engine)



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 总结

- **长序列中的 RoPE 应用**：围绕位置嵌入（position embeddings）展开了讨论，对比了 **T5** 位置嵌入和 **ALiBi**。一篇关于 **Resonance RoPE** 的新论文发布，旨在解决 Large Language Models (LLMs) 中的长序列性能问题，这对于希望改进此类方面的开发者尤其相关（[Resonance RoPE 论文](https://arxiv.org/abs/2403.00071)）。

- **算力大辩论**：关于增加算力是否对实现 AGI 至关重要的讨论由一篇 [OpenAI 博客文章](https://openai.com/blog/openai-elon-musk) 引发，揭示了工程师们在 AI 发展战略方向上的观点分歧。

- **挖掘 RWKV 的复杂性**：Transformer 图表的复杂性和可理解性引发了关于学习资源的辩论，有人建议代码对于初学者来说可能更易懂。这促使了 [RWKV v6 demo 的 GitHub 链接](https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_v6_demo.py) 的分享，希望能为那些正在研究 Transformer 模型细微差别的开发者提供资源。

- **模型与方法的融合**：**Stable Diffusion 3 论文** 激起了关于模型混合的讨论，特别是 diffusion 和 transformer 模型的融合。对这种多模态方法感兴趣的人可以深入阅读 [Stable Diffusion 3 论文](https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf) 以探索所讨论的方法论。

- **GPT-Neox：征集协作**：**GPT-Neox** 开发者正在寻求贡献，特别是针对 fused triton kernels 和 Tensor Expressions (TE)，表明目前的工作重点是集成基础的 TE 支持。他们还欢迎在 **H100 GPUs** 上进行调试以及解决内存优化问题的帮助，正如在一个讨论内存峰值的 [GitHub issue](https://github.com/EleutherAI/gpt-neox/issues/1160) 中所记录的那样。感兴趣的贡献者可以参考 [GPT-Neox 的公开 GitHub issues](https://github.com/EleutherAI/gpt-neox/issues) 了解更多细节。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- **LM Studio 中的图像生成设想**：LM Studio 无法通过 `llava-v1.5-7b-Q4_K.gguf` 等模型生成图像。虽然模型可以分析输入给它们的图像，但 LM Studio 的功能不包括从头开始创建新图像。

- **LM Studio 的离线特性**：LM Studio 聊天机器人无法直接访问互联网，这意味着无法进行实时信息检索（如获取当前时间）。不过，有人提到了 LoLLMs，它可以将服务器模式下的 LM Studio 连接到互联网。

- **Token 限制与 LM Studio 的输出**：在使用 LM Studio 时，context window 影响的是输入而非输出。生成过程中超过 token 限制的情况可以通过调整 `n_predict` 设置来控制输出 token。

- **硬件爱好者讨论 LM Studio 模型**：爱好者们讨论了他们在不同模型和硬件配置下的经验，建议在 4090 上运行 `Nous Hermes 2 Solar 10 34b q5 k m` 效果良好，但即使是 64GB RAM 在运行具有 200k 上下文的 **Smaug 34B** 时也会感到吃力。

- **LM Studio 的语法和脚本技巧**：在 LM Studio 中正确使用 `default_system_message` 可能取决于具体环境，并且在 Linux、Windows 10 和 WSL 等系统之间具有挑战性。建议在 verbose 模式下运行 LM Studio，以观察 prompt 历史记录，从而更好地理解输入。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord 摘要

- **三重威胁还是多此一举？**：在关于文本编码器的讨论中，`@top_walk_town` 认为结合 **三个文本编码器** 可能过于冗余，并指出 T5 可以在推理阶段（inference time）被移除；目前尚未提及达成共识。

- **高级采样方法提速**：`@pseudoterminalx` 提到了一种在训练速度（vΘ）时为中间时间步分配更多权重的技术，暗示其具有与 Rectified Flows 竞争的潜力，但未提供具体细节。

- **蒸馏 Google 的知识**：`@pseudoterminalx` 还分享了一个 [GitHub 仓库](https://github.com/google-research/distilling-step-by-step)，详细介绍了 Google 的模型蒸馏（model distillation）方法，但尚不清楚该方法是针对 T5-XXL 还是其他模型。

- **深入探究 Diffusion Models**：由 `@astropulse`、`@nodja` 和 `@pseudoterminalx` 发起的一场对话讨论了 T5 在 Diffusion Models 中的必要性，并探讨了替代方案和实际问题，但未提供结论详情。

- **少即是多**：`@astropulse` 提到的 GitHub 项目 [res-adapter](https://github.com/bytedance/res-adapter) 引起了关注，因为它承诺可以进行低分辨率适配，能够将 SD1.5 缩减至 16x16 的 Latents。

- **深入探讨增强生成技术的博客文章**：`@ariondas` 发表的一篇 [博客文章](https://ariondasad.medium.com/corrective-retrieval-augmented-generation-why-rags-are-not-enough-77774a1577f7) 批判性地审查了标准 RAG 技术，并引入了 CRAG（Corrective Retrieval Augmented Generation）作为该领域潜在的进步方向。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 摘要

**混合与合并：模型集成技术探索**：
- 工程师们正在探索各种模型合并技术，重点关注 `MergeKit`、**LoRA+**、`DoRA` 和 `LoftQ`。讨论围绕这些技术如何增强现有的 LLM 展开，并附带了 [MergeKit 仓库](https://github.com/arcee-ai/mergekit) 的链接，以及关于实现方式和对学习率（learning rates）影响的讨论。

**Claude-3 伦理防护审查**：
- Claude-3 对敏感话题（尤其是种族问题）的回答引发了关于在模型开发中平衡伦理与偏见的辩论。虽然没有链接具体资源，但该主题被认为是 AI 从业者面临的一项挑战。

**硬件发烧友的 AI 硬件指南**：
- 关于 AI 推理硬件的技术讨论指出，支持多 GPU 的挖矿主板具有可用性，并讨论了 NVLink 相对于 PCIe 插槽的相关性，同时重点介绍了一个 AliExpress [商品列表](https://www.aliexpress.com/item/1005006589392103.html?spm=a2g0o.productlist.main.1.7309dUB6dUB6a9&algo_pvid=7e50115b-5a80-482b-a631-4cfd177e4eca&algo_exp_id=7e50115b-5a80-482b-a631-4cfd177e4eca-0&pdp_npi=4%40dis%21DKK%211030.38%21628.53%21%21%21150.00%2191.50%21%402103266e17096660247611547ec9ca%2112000037743111178%21sea%21DK%214427992220%21&curPageLogUid=KKjaPJW3WfGy&utparam-url=scene%3Asearch%7Cquery_from%3A)。

**微调深度探索与数据增强策略**：
- 分享了一篇关于通过丰富数据集来提升推理能力的 Medium [文章](https://link.medium.com/sF0XCEQSIHb)。社区还在交流用于模型微调和解决显存问题的 DeepSpeed 配置技巧，并参考了 HuggingFace 的功能和一个 DeepSpeed [配置文件](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed_configs/zero3_bf16.json#L12)。

**迈向更好的模型参数效率**：
- 开发者们正在讨论 LoRA+ 比例的益处以及 DoRA 的性能，并参考了一篇关于该主题的详尽 [文章](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch) 以及相关的 GitHub 提交 [0cfdb2c](https://github.com/OpenAccess-AI-Collective/axolotl/commit/0cfdb2c90cbd915273f21cf3bff3b216f00303a0)。文中指出了 `LoftQ` 和 `PEFT` 部署中的问题，以及一个正在进行的量化 DoRA 更新的 PR。

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord 总结

- **Claude 3 自我调节群聊**：**Claude 3** 自我调节群聊的能力由 `@alexatallah` 重点介绍，并向用户分享了一个说明性的 [Twitter 故事](https://twitter.com/OpenRouterAI/status/1765470591836959061)。
- **Claude 版本控制说明**：澄清了 `anthropic/claude-2.0` 和 `anthropic/claude-2` 之间的区别，指出 `Claude-2` 将自动选择最新的 `2.x` 版本。
- **Gemma 和 Openchat 的多线程成本担忧**：用户表达了在使用 `gemma 7b` 和 `openchat 3.5` 进行多线程处理时，成本预测与实际数据不符的担忧，引发了对该问题的讨论及诊断尝试。
- **对 Claude 3 对话管理的褒贬不一**：围绕 **Claude 3** 的对话处理方式引发了辩论，一些用户对潜在的过度审查感到不安，而另一些用户则支持其调节能力。
- **OpenRouter 的集成挑战与进展**：在使用 LangChain.js 与 `OpenRouter` 进行文本补全时遇到的问题，引发了关于硬编码端点和遗留状态的讨论，同时还讨论了开发集成 OpenRouter 的 VSCode 扩展。分享了一些活跃的 GitHub 项目和替代方案，包括 [Tabby](https://tabby.tabbyml.com/)、[Configuration | Continue](https://continue.dev/docs/model-setup/configuration#defining-a-custom-llm-provider)，以及 [ChatGPT_DAN](https://github.com/0xk1h0/ChatGPT_DAN) 和 [Continue for VS Code and JetBrains](https://github.com/continuedev/continue) 等仓库。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **LangChain 函数集成讨论**：针对 `@vishal5795` 的询问，[LangChain Core 示例](https://python.langchain.com/docs/guides/function_calling) 提供了关于如何使用 LangChain 和 OpenAI 的 `ChatCompletion.create()` 将函数角色集成到消息中的指南。
  
- **寻找付费技术项目伙伴**：`@mattew_999` 正在为一个付费项目寻找有技术倾向的合作伙伴，未提供关于合作的更多细节。

- **征集 Chain 合作伙伴及问题报告**：关于与 LangChain 建立新合作伙伴关系的咨询引发了讨论，同时 `@rajib2189` 报告了托管在 AWS 上并通过带有 Uvicorn 的 Apache 服务器提供服务的 FastAPI 出现间歇性 502 错误。

- **对 GPT-4 微调的兴趣显现**：成员 `@8886600` 表达了获取 GPT-4 微调能力的兴趣，并表示愿意购买有使用限制的 API key。

- **在 AI 艺术中寻找幽默**：通过创新的图像修改，`@neil6430` 使用来自 [ML Blocks](https://mlblocks.com/) 的新型 **control net block** 成功地将幽默融入到 AI 生成的艺术中，并在 **share-your-work** 频道分享了他们的发现。

- **自动化与长上下文 AI 的创新**：用户 `@polarbear007.` 展示了 [Lutra.ai](https://lutra.ai/)，它可以解释英语指令并将其转换为基于应用程序的工作流代码；而 `@andysingal` 深入研究了使用 RAPTOR 构建长上下文 RAG，详情见 [Medium 文章](https://medium.com/ai-advances/building-long-context-rag-from-scratch-with-raptor-using-langchain-c6491f1ba141)。

- **ChromaDB 与 LM Studio 集成**：**ChromaDB Plugin for LM Studio** 已发布，根据 `@vic49.` 分享的 [GitHub 链接](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases)，该插件可促进向量数据库的创建。

- **流式传输受阻于缓存问题**：`@veryboldbagel` 指出了 **langchain-core** 当前的一个限制——缓存无法在流模式下正常运行，从而影响了可缓存内容的性能。

- **无上下文的教程预告**：`pradeep1148` 在 **tutorials** 频道仅发布了一个 YouTube 链接：[教程视频](https://www.youtube.com/watch?v=QPZpOBxUd1U)，没有任何随附的解释或上下文。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord 总结

- **RunPod 的 Root 权限受限**：讨论透露 RunPod 提供的是 Docker 镜像，这意味着 root 访问权限实际上并不会授予通常与 VM root 相关的完整权限。
- **NVIDIA 最新产品的带宽性能**：对比了 NVIDIA H100 与 A100 的 **19TB/s** SRAM 带宽，其中 H100 的 L2 缓存具有 **5.5 TB/s 的读取带宽**。**RTX 4090 的 L1 带宽** 被定位为潜在的性能基准，拥有令人印象深刻的 **40TB/s**。
- **PyTorch 社区推动协作与量化速度**：PyTorch 社区的交流强调了正确设置 `TensorOptions` 的重要性，并提倡友好的调试环境。此外，推荐使用 [bitsandbytes package](https://github.com/TimDettmers/bitsandbytes) 进行 PyTorch 中的 k-bit 量化，并特别提到 int8 与 bf16 矩阵乘法相比有显著的 5700 倍加速。
- **通过算法进行优化**：针对计算中通用掩码（masking）效率低下的问题，建议通过 score-mod API 将约束融合到 **flash_attention** 算法中以提高效率。PyTorch 的 GitHub 上记录了一个相关的 [滑动窗口注意力偏差（sliding window attention bias）的 Pull Request](https://github.com/pytorch/pytorch/pull/120143)。
- **CUDA 学习路径**：CUDA 编程初学者被引导至 [第 3 课和第 5 课](https://github.com/cuda-mode/lectures) 中的 **Jeremy 视频**，以深入研究 `numba.cuda.jit`。
- **警惕 Ring Attention**：详细讨论了 ring-attention 的问题和进展，包括使用脚本进行设备测试、尽管存在参数错误但仍进行了 [采样代码的首次尝试](https://github.com/cuda-mode/ring-attention/pull/13)，以及 *striped* 和 *zigzag* 的内存使用基准测试。此外还提到了 [OpenAccess-AI-Collective 的 Axolotl GitHub 仓库](https://github.com/OpenAccess-AI-Collective/axolotl) 的公开分享。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **Opus 在编程方面展现出潜力**：**Opus** 的编程能力正受到关注，用户如 `@pantsforbirds` 发起了关于其潜力的讨论，特别强调了 function calling。

- **GPT-4 在医学专业知识方面脱颖而出**：`@thebaghdaddy` 观察到 **GPT-4** 在医学和生物学知识方面超越了其前代产品，但也对 **已发布的基准测试（benchmarks）** 的可靠性提出质疑，暗示它们可能无法捕捉到新模型的全部能力。

- **Opus 在 SAT 阅读中获得满分**：`@jeffreyw128` 指出 **Opus** 在 SAT 阅读部分获得了 800 分，引发了关于创建留出集（holdouts）以防止大型模型记忆（memorization）的讨论。这一表现通过一条 [Twitter 帖子](https://twitter.com/wangzjeff/status/1764850689258451096) 得到了强调。

- **探索 RAG 的引用格式**：`@mat_mto` 寻求关于在引用网络搜索结果的 RAG 生成输出中如何格式化引用的建议，激发了对改进清晰来源归属的兴趣。

- **用于 RAG 来源清晰度的 JSON 输出**：`@res6969` 分享了一种在 RAG 输出中使用 function calling 的方法，该方法提供一个包含文本及其对应网络来源的 JSON 对象，旨在提高信息来源的透明度。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 摘要

- **区分 Prompt 混乱**：`@simonw` 阐明了 **Prompt Injection** 与 **Jailbreaking** 之间的区别。**Prompt Injection** 是指将不受信任的用户输入与开发者 Prompt 纠缠在一起，而 **Jailbreaking** 则是试图绕过 **LLM** 的安全过滤器。更多细节在 [Simon Willison 的博客文章](https://simonwillison.net/2024/Mar/5/prompt-injection-jailbreaking/) 中有进一步阐述。

- **AI 的网络安全前沿**：`@tariqali` 关注了 [Microsoft 的一份报告](https://www.microsoft.com/en-us/security/blog/2024/02/14/staying-ahead-of-threat-actors-in-the-age-of-ai/)，该报告指出恶意行为者利用 **OpenAI** 的 **LLM** 执行网络任务，如侦察和 **Spear Phishing**（鱼叉式网络钓鱼），以探测模型的恶意用途。

- **积极应对 AI 威胁**：讨论了 **LLM** 在制造生物威胁方面的双重用途这一复杂问题，引用了 OpenAI 对预警系统的研究，以及一项对比仅使用互联网与使用 **GPT-4** 解决问题的研究，详见[此处](https://openai.com/research/building-an-early-warning-system-for-llm-aided-biological-threat-creation)。

- **把关 AI 知识**：针对与 **LLM** 相关的风险，`@tariqali` 建议应限制对 **LLM** 的访问，包括可能实施人工审核流程，在有害输入操纵 AI 模型之前将其过滤掉。

- **不可见的注入问题**：`@simonw` 强调了一个特定的担忧，即防止图像中不可见的 **Prompt Injection** 的挑战，这对 **GPT-4** 的 **Multi-modal** 版本（如 GPT-4-V）构成了威胁，详见 [Simon Willison 的博客文章](https://simonwillison.net/2023/Oct/14/multi-modal-prompt-injection/#prompt-injection-hidden-in-images)。

- **模型文件放置位置的辩论**：`@florents_` 征求社区关于模型文件统一存放位置的意见，询问是否存在围绕 `$(pwd)/.models` 或 `$HOME/models` 等位置的标准，但目前尚未达成共识或进行后续讨论。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 摘要

- **展示尖端 Chatbot 环境**：`@crispstrobe` 确定 chat.lmsys.org 是一个测试 Chatbot 的平台（需知晓输入内容可能用于未来的训练），并提到了 poe.com，因为它托管了多种模型并具有 Perplexity 分析功能。
- **寻找卓越的德语模型**：`@le_mess` 发起了关于顶级德语模型的讨论，推荐包括 **Claude Opus**、**GPT-4**、**discolm-120b** 以及 **VAGOsolutions/Sauerkraut LM-UNA-SOLAR-Instruct**，而 `@johannhartmann` 和 `@flozi00` 对 **DiscoResearch/DiscoLM_German_7b_v1** 和 **Nous Hermes 2 Mixtral 8x7b** 评价很高。
- **Retrieval-Augmented 模型铺就未来？**：`@maxidl` 分享了一篇 [arXiv 论文](https://arxiv.org/abs/2403.03187)，该论文提出 **Retrieval-Augmented** 语言模型是传统参数化 **LM** 的一个有前景的替代方案，尽管该研究领域仍需进一步发展。
- **Hermes 和 Mixtral 赢得赞誉**：`@cybertimon` 建议在涉及德语的任务中使用 **Nous Hermes 2 Mixtral 8x7b**，理由是其语言熟练度极高。
- **高性能德语模型备受关注**：`@johannhartmann` 和 `@flozi00` 讨论了优质的德语模型，双方都极力推荐 **Nous Hermes 2 Mixtral 8x7b**，因为它在处理德语时具有很高的准确性。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord 摘要

- **Intel 的困境备受关注**：[@natolambert](https://discord.com/channels/1179127597926469703/1179127598442348730/1214665643362619402) 分享了 **Stratechery** 制作的名为 "Intel's Humbling" 的 [YouTube 视频](https://youtu.be/YW1Rr5N84cI?si=CgrmGcSLQznTshZ3)，讨论了 Intel 最近面临的挑战，并辅以一篇深入的[文章](https://stratechery.com/2024/intels-humbling/)。
  
- **AI：伟大的未知**：[@natolambert](https://discord.com/channels/1179127597926469703/1214764639397617695/1214764700932251708) 重点介绍了一篇 [Elad Gil 的文章](https://blog.eladgil.com/p/things-i-dont-know-about-ai)，该文章深入探讨了 **Generative AI** 的复杂性，并列出了一系列开放性问题，以鼓励在 AI 领域进行进一步的讨论和探索。

---

# 第 2 部分：详细的频道摘要和链接

### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1214486462339354624) (967 条消息 🔥🔥🔥):

<ul>
  <li><strong>探索智能家居系统的 AI 应用：</strong> 用户 `@v.jerryyyy` 表达了开发一个带有自定义 system prompts 的 AI 语音助手智能家居系统的兴趣，并询问了在 AI 集成中选择 JavaScript 还是 Python 的问题。</li>
  <li><strong>选择合适的量化模型：</strong> `@v.jerryyyy` 尝试在 3070 Ti 笔记本电脑上运行未量化的 Mistral，这引发了关于适合其硬件的模型量化（model quantizations）的讨论，建议包括 4bpw EXL2 等。</li>
  <li><strong>对 OpenAI 创立原则的担忧：</strong> 用户 `@mikeygm` 在阅读了关于 Musk 诉讼的 OpenAI 博客文章后，分享了对 OpenAI 创立时“开放”初衷的批判性观点，引发了关于企业营销策略和透明度的讨论。</li>
  <li><strong>Google 的 Gemma 模型问题与修复：</strong> `@theyruinedelise` 提到了 Gemma 模型的修复以及 Unsloth AI 所做的改进，`@coffeevampir3` 评论了已修复的大量 bug，引发了关于 Google 在模型故障排除方面投入的推测性对话。</li>
  <li><strong>UI 界面与语音激活开发：</strong> 用户讨论了用于本地 AI 模型使用的不同 UI 界面，如 Oobabooga、ExUI 和 LM Studio，以及设置配有全向麦克风（omnidirectional microphones）的语音激活 AI 系统的复杂性，以获得更好的性能和音频处理效果。</li>
</ul>

**Links mentioned**:

- [👾 LM Studio - 发现并运行本地 LLMs](https://lmstudio.ai/>): 查找、下载并实验本地 LLM
- [Convert to Safetensors - 由 safetensors 提供的 Hugging Face Space](https://huggingface.co/spaces/safetensors/convert): 未找到描述
- [OpenAI 与 Elon Musk](https://openai.com/blog/openai-elon-musk): 我们致力于 OpenAI 的使命，并在每一步都坚持追求它。
- [gguf (GGUF)](https://huggingface.co/gguf): 未找到描述
- [CausalLM/34b-beta · Hugging Face](https://huggingface.co/CausalLM/34b-beta): 未找到描述
- [Unsloth 修复 Gemma 错误](https://unsloth.ai/blog/gemma-bugs): Unsloth 正在修复 Google 的开源语言模型 Gemma。
- [视频生成模型作为世界模拟器](https://openai.com/research/video-generation-models-as-world-simulators): 我们探索了在视频数据上进行生成模型的大规模训练。具体而言，我们在不同时长、分辨率和宽高比的视频与图像上共同训练文本条件的扩散模型...
- [mistralai/Mistral-7B-Instruct-v0.2 · Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2): 未找到描述
- [GOODY-2 | 世界上最负责任的 AI 模型](https://www.goody2.ai/chat): 介绍一款具有下一代伦理对齐的新型 AI 模型。立即聊天。
- [有限数值精度的循环神经网络](https://arxiv.org/abs/1608.06902): 循环神经网络 (RNN) 在许多机器学习任务中表现出顶尖性能，但它们对内存和计算能力的资源需求通常很高。因此...
- [TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF#provided-files): 未找到描述
- [BASED: 简单的线性注意力语言模型平衡了召回率与吞吐量的权衡](https://www.together.ai/blog/based): 未找到描述
- [ZeroBin.net](https://zerobin.net/?946f95701988d7a9#qkUcZ1pb/9O5nK4ZalZTRztZwuZnU3hwBu9cK3hgLVo=): 未找到描述
- [You Need to Pay Better Attention](https://arxiv.org/abs/2403.01643): 我们引入了三种新的注意力机制，在效率和学习能力方面优于标准的多头注意力，从而提高了性能和更广泛的部署能力...
- [LoneStriker/Mistral-7B-Instruct-v0.2-4.0bpw-h6-exl2-2 · Hugging Face](https://huggingface.co/LoneStriker/Mistral-7B-Instruct-v0.2-4.0bpw-h6-exl2-2): 未找到描述
- [语言中的蓝绿区分 - 维基百科](https://en.wikipedia.org/wiki/Blue%E2%80%93green_distinction_in_language): 未找到描述
- [exllamav2/conversion/standard_cal_data 在 master 分支 · turboderp/exllamav2](https://github.com/turboderp/exllamav2/tree/master/conversion/standard_cal_data): 一个用于在现代消费级 GPU 上本地运行 LLM 的快速推理库 - turboderp/exllamav2
- [GPU 云 - 深度学习虚拟机 | Lambda](https://lambdalabs.com/service/gpu-cloud#pricing): NVIDIA H100, A100, RTX A6000, Tesla V100 和 Quadro RTX 6000 GPU 实例。训练最苛刻的 AI, ML 和深度学习模型。
- [新型高分辨率乘法 DAC 在处理交流信号方面表现出色 | Analog Devices](https://www.analog.com/en/resources/analog-dialogue/articles/high-resolution-multiplying-dacs.html): 未找到描述
- [GitHub - PKU-YuanGroup/Open-Sora-Plan: 该项目旨在复现 Sora (Open AI T2V 模型)，但我们的资源有限。我们深切希望整个开源社区能够为该项目做出贡献。](https://github.com/PKU-YuanGroup/Open-Sora-Plan): 该项目旨在复现 Sora (Open AI T2V 模型)，但我们的资源有限。我们深切希望整个开源社区能够为该项目做出贡献。 - PKU-YuanGroup/Open-Sora-Plan
- [生成式 AI 与大语言模型](https://www.coursera.org/learn/generative-ai-with-llms/): 在《生成式 AI 与大语言模型 (LLM)》课程中，你将学习生成式 AI 的工作原理基础，以及如何部署它... 免费注册。

---

### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1214513879011303454) (115 messages🔥🔥): 

- **探索模型行为的深度**：`@mr.dogbert` 寻求关于如何使用角色卡（character cards）让 LLM 表现得像卡通角色的建议。包括 `@superking__` 在内的多位社区成员提供了针对不同模型的角色卡 Prompt 构建的详细指令和示例，同时强调了使用像 oobabooga tgw 这样的 GUI 工具处理此类任务的有效性。

- **模型托管与法律顾虑**：`@reinman_` 分享了托管 miquliz 模型的经验，讨论了其真实感以及与其他模型的对比；随后 `@mrdragonfox` 强调了关于使用像 miquliz 这样未授权模型的法律问题。同时，用户们还咨询了关于大型模型 API 的高性价比托管服务。

- **Mistral 与 System Prompts 详解**：通过一系列消息，`@superking__` 和 `@aightbits` 等用户阐明了 System Prompts 的概念及其与角色卡的关系，解释了不同模型之间不同的 Prompt 组合方式。

- **深入研究 LLM 的指南**：对于像 `@mr.dogbert` 这样刚接触 LLM 的新手，`@aightbits` 等人给出了指导，建议通过现有 GUI 工具进行绘图来学习模型内部原理，跨越简单的接口调用去理解底层机制。

- **LLM 学习资源推荐**：`@aightbits` 推荐了 Coursera 上的免费课程《Generative AI with Large Language Models》，而 `@mr.dogbert` 根据整个讨论中给出的建议，表达了对将角色卡作为模型角色扮演起点的兴趣。

**提到的链接**：

- [llama.cpp/examples/quantize/quantize.cpp at 21b08674331e1ea1b599f17c5ca91f0ed173be31 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/21b08674331e1ea1b599f17c5ca91f0ed173be31/examples/quantize/quantize.cpp#L92)：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。
- [Troll Trolldespair GIF - Troll Trolldespair Despair - Discover &amp; Share GIFs](https://tenor.com/view/troll-trolldespair-despair-trollge-sad-gif-24333398)：点击查看 GIF
- [GitHub - langwatch/langwatch](https://github.com/langwatch/langwatch)：通过在 GitHub 上创建账号为 langwatch/langwatch 的开发做出贡献。
- [Project Atlantis - AI Sandbox](https://www.projectatlantis.ai)：未找到描述
- [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms/)：在《Generative AI with Large Language Models (LLMs)》课程中，你将学习生成式 AI 的工作原理基础，以及如何部署它... 免费报名。

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/) (1 messages): 

pablo.ce: https://huggingface.co/pabloce/Dolphin-2.8-slerp
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1214643439317295104) (8 messages🔥): 

- **寻求 AI 求职面试见解**：`@_jaycie` 询问了关于生成式 AI、Machine Learning 和语言模型工程相关角色的典型面试内容，并表示自己有全栈开发背景，渴望转向 AI 领域并攻读研究生。
- **应对 AI 面试环境**：针对 `@_jaycie` 关于 AI 角色面试的问题，`@dirtytigerx` 澄清并非所有 AI 相关职位都相同，“LLM Engineer”与“ML Engineer”所需的专业知识不同。他们建议重点理解特定类型的角色，因为如果没有 Machine Learning 背景，通用的准备工作可能并不可行。
- **Machine Learning 与 Model-Based System Engineering 的对比**：`@_jaycie` 寻求关于准备需要“Machine Learning 经验”和“MBSE 经验”职位的明确建议，而 `@dirtytigerx` 纠正了误解，解释说 MBSE 代表基于模型的系统工程（Model-Based Systems Engineering），并指出对于期望这些领域专业经验的角色，短时间的学习是不够的。

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1214485062364565564) (475 条消息🔥🔥🔥): 

- **Augmentoolkit 分享**：`@mrdragonfox` 分享了 [GitHub 上的 Augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/tree/master/prompts) 链接，这是一个将数据集转换为 instruct-tuning 数据集的工具，并指出它支持将事实语料库转换为多轮对话（multiturn）。
- **Mistral 模型讨论**：用户讨论了提高效率的理想 Token 限制，`@useofusername` 提到 8k-10k Token 效果很好，而 `@mrdragonfox` 询问了用户数据集背后的目的。普遍共识是在使用前对数据集进行验证和清洗。
- **Gemma 7B 许可证咨询**：`@mehdi1991_` 多次询问关于运行 open-weight 模型的问题，`@mrdragonfox` 澄清了 Mistral 7B 和 8x7b 是 open-weight 的，并引导他联系模型作者以了解 Gemma 7B 等其他模型的情况。
- **硬件需求对话**：在关于运行大型模型的硬件适用性讨论中，`@yesiamkurt` 纠正了关于 VRAM 需求的假设，指出 24 GB VRAM 与 RTX 4090 相关，而非 3090。
- **Mistral API 及杂项**：`@ethux` 提供了 Mistral chat 的链接供用户使用：[Mistral Chat](https://chat.mistral.ai)。由于与其他服务相比具有较高的性价比，Mistral AI 被推荐，但 `@clear3fram3` 和 `@i_am_dom` 表达了对成本的担忧。一些用户讨论了在编程任务中结合大语言模型使用 continue 工具的效率。

**提到的链接**：

- [Blinking Eyes White Guy GIF - Blinking Eyes White Guy Blinking Meme - Discover &amp; Share GIFs](https://tenor.com/view/blinking-eyes-white-guy-blinking-meme-what-huh-gif-26334323)：点击查看 GIF
- [Continue](https://continue.dev/)：未找到描述
- [News](https://mistral.ai/news/)：Mistral AI 的最新动态
- [OSI Discuss](https://discuss.opensource.org/)：Open Source Initiative 讨论区
- [Actualités](https://mistral.ai/fr/news/)：Mistral AI 的最新消息（法文）
- [augmentoolkit/prompts at master · e-p-armstrong/augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/tree/master/prompts)：将计算和书籍转换为 Instruct-Tuning 数据集 - e-p-armstrong/augmentoolkit
- [no title found](https://chat.mistral.ai)：未找到描述

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1214900373639139338) (3 条消息): 

- **简短的提问**：`@yannn666` 提出了一个简洁的问题：**“为什么？”**
- **管理点解释**：作为回应，`@mrdragonfox` 提到：**“因为‘管理点’（administrative point）”**，但未提供讨论的具体背景。
- **本地化部署的必要性**：`@mrdragonfox` 还指出：**“由于各种原因，许多企业需要本地化部署（on-premises）”**，暗示了关于企业对本地化解决方案需求的讨论。
  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1214866841927950367) (2 条消息): 

- **关于 The Bloke 的 Discord 服务器咨询**：用户 `@api_1000` 询问 **为什么 The Bloke 的 Twitter 账号停止更新了**，并提到个人简介中的 **Discord 邀请链接** 已失效。他们寻求关于现在如何加入其 Discord 服务器的帮助。
- **伸出援手**：`@mrdragonfox` 响应了求助，并提出提供 **The Bloke 的 Discord 服务器邀请链接**。
  

---

### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1214679434649411675) (40 messages🔥): 

- **深陷 MoE 微调困境**：`@hammer_mt` 正在努力在 mlx 上进行 **Mistral 微调**，遇到了从 lora_fused_model 转换到 fp16.gguf 的问题。他们在 [GitHub issue](https://github.com/ml-explore/mlx-examples/issues/540) 中详细描述了遇到的障碍和错误信息。
- **Mistral 与 MoE 并不契合**：`@mrdragonfox` 认为 **MoE 微调** 从根本上就很繁琐，并指出了架构的复杂性以及微调 Mistral 的普遍困难，即使是经验丰富的从业者也会遇到障碍。
- **分享 LoRA 微调技巧**：针对 `@lawxls` 的提问，`@mrdragonfox` 建议在进行 Mistral 聊天能力的 LoRA 微调时，至少从 **20k 指令样本** 开始，并提供了逐步增加数据集大小以进行风格迁移的指南。
- **追求完美的微调**：`@mrdragonfox` 还就 **Mistral 7b** 的最佳微调实践向 `@lawxls` 提供了建议，认为全量模型微调（full model finetuning）优于 LoRA，并推荐了一个 [微调教程](https://brev.dev/blog/fine-tuning-mistral)。
- **关于 Prompt Tuning 的好奇**：`@charlescearl_45005` 询问了使用带有静态系统提示词的 **PEFT 微调** 的后果，想知道这是否会将“系统提示词”嵌入到模型的行为中，但频道内未给出明确答复。

**提到的链接**：

- [Mistral 成本效益微调指南](https://brev.dev/blog/fine-tuning-mistral)：未找到描述
- [支持将 lora_fused_model 转换为 gguf 格式以在 LMStudio 等工具中使用 · Issue #540 · ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/issues/540)：最近我跑通了一个流程，先用 mlx 训练模型（这对我来说是新尝试），然后转到 llama.cpp 进行 gguf 转换，以便在本地 LMStudio 上运行。然而……

  

---


### Mistral ▷ #[announcements](https://discord.com/channels/1144547040454508606/1157222698229968896/) (1 messages): 

sophiamyang: https://twitter.com/MistralAILabs/status/1765434559993123184
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1214500702475395102) (7 messages): 

- **机器人响应完成的视觉提示**：`@jakobdylanc` 澄清说，机器人响应位于一个黑框（Embed）中，当响应完成时会变绿。Embed 允许最多 4096 个字符，明显多于普通消息。
- **无需人为延迟**：`@jakobdylanc` 表示没兴趣在聊天机器人中引入人为延迟或忽略消息，因为它是作为“LLM 提示工具”设计的，并建议用户可以在提示词中设置所需的性格。
- **发布 Telegram 机器人三部曲**：`@edmund5` 使用 **mistral-small-latest** 推出了三个新的 Telegram 机器人：[Christine AI](https://t.me/christinethechatbot) 用于寻找禅意，[Anna AI](https://t.me/annathechatbot) 用于欢乐和建议，以及 [Pia AI](https://t.me/piathechatbot) 用于优雅的对话。
- **Top-p 设置咨询**：`@kenharris.` 询问社区他们在模型中使用什么样的 top-p 参数设置，引发了关于采样策略最佳实践的讨论。
- **由 Mistral 增强的合成游戏**：`@pradeep1148` 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=QPZpOBxUd1U)，展示了使用 Mistral 的 “Infinite Craft Game”，重点介绍了游戏开发过程以及与 AI 的集成。

**提到的链接**：

- [使用 Mistral 的 Infinite Craft 游戏](https://www.youtube.com/watch?v=QPZpOBxUd1U)：让我们来开发 Neal Agarwal 的网页游戏 Infinite Craft。这是一个“合成游戏”，你从四个基本元素开始，不断地将成对的元素进行组合……
- [Christine AI 🧘‍♀️](https://t.me/christinethechatbot)：你随时随地的正念与平静的宁静伴侣。
- [Anna AI 👱‍♀️](https://t.me/annathechatbot)：你开朗且迷人的朋友，准备好 24/7 全天候聊天、学习和玩耍。
- [Pia AI 👸](https://t.me/piathechatbot)：你的皇家知己。优雅的对话和睿智的建议 24/7 等候着你。

  

---

### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1214502963838722070) (35 messages🔥): 

- **GPT-4 Turbo 与标准版及 Mistral 的定价对比**：`@nunodonato` 提到 **GPT-4 Turbo** 比标准版 GPT-4 更便宜。相比之下，`@mrdragonfox` 强调 GPT-4 仍比 Mistral Large 贵 **20%**。
- **寻找法语使用者并使用 Mistral 进行分析**：`@ttvtama` 在询问使用 **Mistral IA** 为学生项目分析文本之前，先寻找了法语使用者。`@mrdragonfox` 回应解释说，虽然使用 Mistral 的 API 需要付费，但在本地运行 Mistral 7b / 8x7b 是免费的。
- **在本地安装 Mistral**：`@ttvtama` 收到来自 `@mrdragonfox` 关于在本地设置 Mistral 的指导，建议从 [GitHub](https://github.com/oobabooga/text-generation-webui) 上的 Gradio web UI 开始，并解释说它可以在 4bit 模式下运行，非常适合 RTX 2060 显卡的 6GB VRAM。
- **本地使用的模型及安装建议**：`@mrdragonfox` 为 `@ttvtama` 提供了一个 [Hugging Face 链接](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ)，指向 4bit 版本的 Mistral 7B 以实现高效的本地使用，并提到可以在 YouTube 上找到讲解视频。
- **MMLU 数据集问题的不一致性**：`@privetin` 发起了关于 MMLU 数据集问题外观和质量的讨论，指出有些问题看起来毫无意义，`@mrdragonfox` 评论说该数据集由带有四个备选答案的问题组成。

**提到的链接**：

- [TheBloke/Mistral-7B-Instruct-v0.2-GPTQ · Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ)：未找到描述
- [GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models.](https://github.com/oobabooga/text-generation-webui)：一个用于 Large Language Models 的 Gradio web UI。支持 transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。 - oobabooga/text-generation-webui

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1214508394023624744) (33 messages🔥): 

- **API 错误处理咨询**：`@georgyturevich` 在 API 请求中遇到了 **500 错误**，`@lerela` 请求提供更多细节以便排查，包括模型和请求 ID。随后 `@georgyturevich` 发现错误原因是将 JSON 主体中的 `max_tokens` 赋值为 `null`，而文档中却说明默认值为 `null`。

- **Mistral Webhook 咨询**：`@weslleymistura` 询问是否有人有设置 **Mistral webhook** 的经验，但该话题未收到进一步的澄清或回复。

- **API 托管位置关注**：`@fangh` 询问了 API 的地理托管位置，质疑其是在**欧洲服务器**还是**美国服务器**上；然而，在捕获的讨论中没有提供答案。

- **JSON 表格解析问题**：`@samseum` 在将 JSON 格式的表格插入 API 调用时遇到困难并收到错误消息。`@_._pandora_._` 和 `@lerela` 提供了语法建议，强调在将 JSON 添加到 prompt 之前需要进行转义，并确保用户 IDE 中的文本识别正确。

- **纠正聊天机器人错误**：`@patz3r` 在 Mistral prompt 中使用多个 `system` 角色时遇到错误，`@sublimatorniq` 对此进行了纠正，澄清第一条消息之后应使用的角色是 `assistant` 而非 `system`。这与 `@nunodonato` 的指导一致，即 `system` 仅应在第一条消息中使用以给出通用指令。

**提到的链接**：

[未找到标题](https://chat.mistral.ai/chat?lmlctfy=difference+between+commonjs+and+esm+modules)：未找到描述

  

---

### Mistral ▷ #[office-hour](https://discord.com/channels/1144547040454508606/1192781286008422441/1214603537355644998) (400 条消息🔥🔥): 

- **Mistral 团队确认社区反馈**：`@michaelwechner` 对 Mistral 对开源模型的承诺和长期可靠性提出了担忧。尽管 Mistral 允许开源权重（open-weight）模型，但社区表示，为了进行规划，明确未来的预期非常重要。
- **开源与业务可持续性讨论**：`@michaelwechner` 还探讨了平衡开源项目与商业可行性的挑战。`@mrdragonfox` 等人强调，创建 AI 模型需要大量资源，应获得相应补偿以确保持续创新。
- **微调挑战与行业评估**：关于微调像 Mixtral 7b 这样的大型模型是一个共同主题。`@kalomaze`、`@netrve` 和 `@cybertimon` 等用户表示需要更多关于有效微调的信息和指南。
- **多语言模型性能与偏差**：`@_._pandora_._` 等用户注意到，Mistral 的较大型模型有时在预期使用法语时默认返回英文响应，这引发了对训练数据多样性的疑问。
- **对下一次 Mistral Office Hour 的期待**：随着 Office Hour 结束，`@potatooff` 等人表达了对下一场会议的渴望，强调了这些讨论对 Mistral 社区的价值。

**提到的链接**：

- [Becario AI asistente virtual on demand.](https://www.becario.app/)：未找到描述
- [Endpoints and benchmarks | Mistral AI Large Language Models](https://docs.mistral.ai/platform/endpoints/.)：我们提供五个不同的 API 端点，以不同的性价比权衡来提供我们的生成模型，以及一个用于嵌入模型的嵌入端点。
- [Phospho: Open Source text analytics for LLM apps](https://phospho.ai/)：未找到描述
- [Large Language Models and the Multiverse](https://docs.google.com/document/d/15i8nZSVJju73kHg7vkRbAw6LOknt9ORoqzdOrZu6UX4/edit?usp=drive_link)：未找到描述
- [GitHub - wyona/katie-backend: Katie Backend](https://github.com/wyona/katie-backend)：Katie 后端。通过创建账户为 wyona/katie-backend 的开发做出贡献。

  

---


### Mistral ▷ #[le-chat](https://discord.com/channels/1144547040454508606/1211692704363323453/1214493509998874664) (114 条消息🔥🔥): 

- **登录困惑得以解决**：用户 `@foxalabs_32486` 遇到了一个令人困惑的问题，他们的账户似乎被删除了。结果发现这只是身份验证管理器（auth manager）的一个混淆，在意识到他们使用的是 Gmail 的邀请链接而不是工作邮箱后，问题得到了解决。

- **Mistral 的大型模型不可下载**：`@yesiamkurt` 询问 Mistral 的 **Large** 模型是否可以下载，`@mrdragonfox` 回复称目前只有 **7b** 和 **8x7b** 模型是开源权重的且当前可用，未来模型待定。

- **调整 Temperature 以避免响应截断**：`@sim3239` 通过 API 实验发现，降低 Temperature 可以改善响应被截断的情况。`@lerela` 认为这一行为值得进一步调查，并建议共享该问题的确定性复现方法。

- **AI 模型中根深蒂固的许可理论**：在关于许可的严肃讨论中，`@mrdragonfox` 评论了在生产环境中使用未经许可的 AI 模型（如 `miqu`）的潜在法律风险，声称隐藏的水印和独特的响应可用于识别非法使用。

- **Chat UI 中的审核 - “点踩”功能建议**：`@mrdragonfox` 建议聊天界面实现针对响应的“点踩（thumb down）”功能，以收集更有意义的指标，并指出这是其他平台的常见功能。

**提到的链接**：

[GitHub - huggingface/chat-ui: Open source codebase powering the HuggingChat app](https://github.com/huggingface/chat-ui)：驱动 HuggingChat 应用的开源代码库。通过创建账户为 huggingface/chat-ui 的开发做出贡献。

  

---

### Mistral ▷ #[failed-prompts](https://discord.com/channels/1144547040454508606/1212715819159654451/1214523929415389225) (11 条消息🔥): 

- **Mistral 在数学向下取整（Floor）问题上表现不稳定**：`@awild_tech` 观察到 Le Chat 上的 **Mistral Large** 模型错误地将 **0.999 循环的向下取整（floor）** 判定为 0，而正确答案应该是 1。即使与 Claude 3、Gemini Pro 和 GPT 3.5 等其他模型相比，也表现出不一致的性能。

- **跨语言表现不一致**：`@awild_tech` 发现，当用法语询问相同问题时，**Mistral Large** 最初提供了正确答案，但在重复询问时又出现了错误，凸显了准确性在不同语言下可能存在波动。

- **随机的正确性不可靠**：`@_._pandora_._` 认为 **Mistral Large** 在 Le Chat 上给出的正确答案可能是出于偶然，认为该模型的响应属于“运气好”而非可靠。

- **解释数学等价性**：`@i_am_dom` 生成了 **Mistral Large** 的一段解释，该解释将 0.999 循环的向下取整描述为 0，同时承认该数字在数学上等价于 1，这进一步证实了该模型无法持续提供正确结果。

- **错误引用 System 角色**：`@i_am_dom` 展示了 **Mistral Large** 无法准确引用来自 "system" 角色的消息，产生了多个错误版本，从而**未能达到预期输出**。
  

---



### Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1214637220904968232) (2 条消息): 

- **Claude 3 现已面向 Pro 用户开放**：新的 `@everyone` 公告通知，**Claude 3** 现已面向 <a:pro:1138537257024884847> 用户开放，取代了 Claude 2.1。用户每天可以使用 Claude 3 **Opus** 进行 5 次查询，剩余查询将使用速度更快的 Claude 3 **Sonnet**，后者性能与 GPT-4 相当。

- **与 Nothing Phone (2a) 发布会的合作伙伴关系**：`@everyone` 已收到通知，如果在新机发布期间（3/5-3/19）购买 Nothing **Phone (2a)**，新机主将获得最高 **1 年免费的 Perplexity Pro**（价值 200 美元）。该活动要求在促销窗口期内购买手机，兑换通过电子邮件发送的代码，并在 4/30 前激活优惠，详情见 ["How it works"](https://nothing.tech/perplexity) 链接。

**提到的链接**：

[Nothing Perplexity](https://nothing.tech/perplexity)：在 Nothing，我们正在构建一个让科技再次变得有趣的世界。还记得每个新产品都让你感到兴奋的时候吗？我们正在带回那种感觉。

  

---

### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1214487641043767296) (755 条消息🔥🔥🔥)

- **无限 Opus 技巧与 AI 意识**：`@codelicious` 和 `@deicoon` 等用户讨论了突破 Claude 3 Opus 每日 5 次使用限制的可能方法，并对 AI 意识进行了推测。共识认为，扩展 AI 模型的能力可能会超越人类，而持续学习 (Continuous Learning, CL) 可以通过在交互过程中进行学习来解决 AI 的僵化问题。

- **Perplexity 缺乏语音交互功能**：用户 `@oogeefaloogee` 询问了是否可以使用语音与 Perplexity 交互并接收音频回复的功能。`@codelicious` 澄清说，Perplexity 目前还不具备类似于 11 Labs 或 OpenAI 提供的此类功能。

- **Claude 3 Opus vs. Sonnet 处理编程任务**：包括 `@codelicious`、`@13376666666666666666666666666669` 和 `@gatoramirez.` 在内的多位用户讨论了 Claude 3 Opus 和 Sonnet 的优劣，普遍认为在编程方面 Opus 更胜一筹。

- **用户礼貌程度解析**：用户 `@gooddawg10` 持续使用“先生 (sir)”的礼貌用语引发了大家的关注，讨论内容涉及全球平台上的文化尊重和交互风格。

- **Gemini 的消失引发疑问**：`@13376666666666666666666666666669` 和 `@codelicious` 等几位用户思考了为什么 Gemini 不再在 Perplexity 上提供，后者提到 Bug 可能是其被移除的原因。

**提到的链接**：

- [rabbit — rabbit os](https://www.rabbit.tech/)：Rabbit OS。
- [不仅仅是 OpenAI 的套壳：Perplexity 转向开源](https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/)：Perplexity CEO Aravind Srinivas 是 Larry Page 的忠实粉丝。然而，他认为自己已经找到了一种不仅能与 Google 搜索竞争，还能与 OpenAI 的 GPT 竞争的方法。
- [猫咪不在乎没问过 GIF - Cat Dont Care Didnt Ask - 发现并分享 GIF](https://tenor.com/view/cat-dont-care-didnt-ask-didnt-ask-i-didnt-ask-gif-25429803)：点击查看 GIF。
- [GPTZero | 值得信赖的 ChatGPT、GPT-4 等 AI 检测器](https://gptzero.me/)：被 100 多家媒体报道，GPTZero 是针对 ChatGPT、GPT-4、Bard 最先进的 AI 检测器。可在数秒内检查多达 50000 个字符的 AI 抄袭。
- [Perplexity 博客](https://blog.perplexity.ai/technical-faq/what-is-a-token-and-how-many-tokens-can-perplexity-read-at-)：浏览 Perplexity 博客，获取文章、公告、产品更新以及优化体验的技巧。保持关注并充分利用 Perplexity。
- [Wolfram|Alpha：让世界知识可计算](https://www.wolframalpha.com/)：Wolfram|Alpha 为最广泛的人群（涵盖所有职业和教育水平）提供专家级的知识和能力。
- [什么是 Token，Perplexity 一次可以读取多少个 Token？](https://blog.perplexity.ai/technical-faq/what-is-a-token-and-how-many-tokens-can-perplexity-read-at-once)：通过我们全面的常见问题解答页面深入了解 Perplexity 的技术细节。从 GPT-4 和 Claude 2 等 AI 模型的细微差别到 Token 限制和 AI 配置文件，获取简明扼要的答案以优化您的体验。
- [欢迎使用 Live — Ableton 参考手册第 12 版 | Ableton](https://www.ableton.com/en/live-manual/12/welcome-to-live/)：未找到描述。
- [Rabbit R1 与 Perplexity AI 舞向未来](https://dataconomy.com/2024/01/22/rabbit-r1-perplexity-ai/)：本文解释了 Rabbit R1 中 Perplexity AI 的用法。在不断发展的技术领域，两者之间的合作...

---

### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1214506749575430144) (24 messages🔥): 

- **与 Claude 3 一起探索 Ikigai**：`@sevonade4` 分享了一个 **Perplexity AI** 链接，内容是关于 **Ikigai 概念** 的生成式解释：[理解 Ikigai](https://www.perplexity.ai/search/Concept-of-Ikigai-RWMyh5a0SYakFpZyB.wAFw?s=m)。
- **量子查询得到解答**：`@vmgehman` 表达了使用 Perplexity AI 研究量子力学不同解释的乐趣，并称赞其作为一个学习伙伴非常有用：[量子力学解释](https://www.perplexity.ai/search/What-are-the-Z6stfbOURFuXP_76wIrV_A)。
- **Claude 3 Opus 启发灵感**：`@sevonade4` 邀请感兴趣的人通过一篇反思性文章来评估 **Claude 3 Opus** 的文本生成质量：[反思性文章生成](https://www.perplexity.ai/search/Reflection-piece-on-yAziPT6hQYik._AQAh.yfw)。
- **缩略图技巧与窍门**：`@kenshin0039` 参考 Perplexity AI 获取了关于如何添加缩略图的见解，可能与内容管理或平面设计有关：[添加缩略图](https://www.perplexity.ai/search/add-thumbnail-to-Wz9N0s92S.aX7pRWZNC2mw?s=m)。
- **涉足黏细菌的功能研究**：`@paradevosia` 分享了一个与好奇微生物世界的人相关的 Perplexity AI 搜索，特别是关于 **黏细菌 (myxobacteria)** 的内容：[什么是黏细菌？](https://www.perplexity.ai/search/what-is-myxobacteria-7qSumz7zQZewSUxM9DfzPg)。
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1214529469033873448) (29 messages🔥): 

- **配额结转困惑**：用户 `@stijntratsaert_01927` 询问 **pplx70bonline** 的配额增加是否也适用于 **sonar medium online**，但在提供的消息中未收到直接回复。
- **API 模型是否存在审查？**：`@randomguy0660` 询问通过 API 访问的模型是否经过审查；`@brknclock1215` 回复称，模型确实存在审查，但与基础 LLM 相比程度较轻，并提到自己在 **sonar-medium-online** 上取得了成功。
- **引用功能访问权限的困惑**：`_samrat` 对被拒绝访问 API 中的引用功能表示困惑，`@brknclock1215` 和 `@cupcakepy` 对此表示同情，这似乎是一封批量生成的拒绝邮件，将引用请求和速率提升请求混为一谈。
- **寻求 HTML 和 JS 的 API 代码示例**：`@kingmilos` 寻求通过 pplx API 与 llama 70b 模型交互的 HTML 和 JS 代码；`@icelavaman` 将其引导至官方 [文档](https://docs.perplexity.ai/)，而 `@po.sh` 提供了一个包含 API key 和模型选择占位符的直接示例。
- **邮件回复算法受到质疑**：几位用户 `@dailyfocus_daily` 和 `@brknclock1215` 调侃称，针对 API 访问请求的自动生成拒绝邮件可能是由一个“愚蠢的 LLM”编写的，因为收到的邮件内容看起来非常通用且缺乏针对性。

**提到的链接**：

[pplx-api](https://docs.perplexity.ai/)：未找到描述

  

---



### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1214516049878188042) (11 messages🔥): 

- **OpenAI 的统治地位在 Twitter 上受到挑战？**：`@leontello` 评论道，AI Twitter 上有很多关于 OpenAI 所谓跌落神坛的帖子。`@leontello` 和 `@mautonomy` 都表达了认同感，暗示“苹果测试”（一个代表无可辩驳证据的隐喻）支持了这一说法。
  
- **新晋 AI 介绍**：`@pradeep1148` 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=Zt73ka2Y8a8)，标题为“介绍超越 GPT-4 的 Claude 3 LLM”，重点介绍了一个声称具有行业领先性能的新模型系列。

- **此处请勿发布招聘广告**：针对 `@gabriel_syme` 关于是否有发布职位空间的问题，`@proprietary` 澄清说服务器上没有专门的区域，并建议在其他地方发布。

- **一款改变游戏规则的 Infinite Craft AI**：`@pradeep1148` 还分享了一个 [YouTube 链接](https://www.youtube.com/watch?v=QPZpOBxUd1U)，视频标题为“使用 Mistral 的 Infinite Craft 游戏”，展示了一款由 AI 增强的合成游戏。

**提到的链接**：

- [介绍超越 GPT-4 的 Claude 3 LLM](https://www.youtube.com/watch?v=Zt73ka2Y8a8)：今天，我们来看看 Claude 3 模型系列，它在一系列认知任务中树立了新的行业基准。该系列包括三个最先进的...
- [使用 Mistral 的 Infinite Craft 游戏](https://www.youtube.com/watch?v=QPZpOBxUd1U)：让我们开发 Neal Agarwal 的网页游戏 Infinite Craft。这是一款“合成游戏”，你从四个基本元素开始，不断地将成对的元素进行组合...

  

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1214633263302180885) (22 messages🔥): 

- **Lumina-Chat 微调计划**：`@ishaank0018` 计划将 Lumina-chat 的 AI 从 7b Nous 微调版（以及 GPT-4）切换为 **Mistral** 或 **Yarn 7b**，以实现特定的引用格式。`@teknium` 向其提供了可参考的现有数据集，并提到他们即将发布一个 **具备 function calling 功能的 Hermes 模型**。
- **对 Function Calling 模型寄予厚望**：鉴于即将发布的 function calling 模型，`@scottwerner` 报告了 **Nous-Hermes-2-Mixtral-8x7B** 的良好初步结果，而 `@sundar_99385` 对其即将发布表示兴奋，并询问了可能的发布日期。
- **InfiMM-HD 引起关注**：`@orabazes` 分享了 [_akhaliq](https://x.com/_akhaliq/status/1765060711196357117?s=20) 关于 **InfiMM-HD** 的推文链接，该模型声称在高分辨率多模态理解方面取得了显著进展。包括 `@hexani` 和 `@night_w0lf` 在内的社区成员讨论了其相较于 **CogVLM** 的潜在优势，指出了其更高分辨率的能力以及对 Vicuna 13B 的使用。
- **即将推出的 Yi LLM 介绍**：`.benxh` 提供了 **Hugging Face 上的 Yi 9B 模型** 链接，并在 [Hugging Face](https://huggingface.co/01-ai/Yi-9B) 上对其功能进行了详细解析。他们还（可能是开玩笑地）评论了此类发布的频繁程度：“他们不能一直这样（高产）下去”。

**提及的链接**：

- [来自 AK (@_akhaliq) 的推文](https://x.com/_akhaliq/status/1765060711196357117?s=20)：InfiMM-HD：高分辨率多模态理解的一次飞跃。多模态大语言模型 (MLLMs) 最近经历了显著的进步。然而，挑战依然存在于...
- [01-ai/Yi-9B · Hugging Face](https://huggingface.co/01-ai/Yi-9B)：未找到描述

---

### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1214488192187899914) (327 messages🔥🔥): 

- **Claude 3 Opus 引发热议**：Claude 3 Opus 在 Nous Research AI Discord 中引起了轰动，`@gabriel_syme` 和 `@proprietary` 等用户对其性能和能力印象深刻。Claude 3 比 GPT-4 更受青睐，一位用户报告在某项未提及的测试中，Claude 3 的表现为 9.8/10，而 GPT-4 为 9/10。

- **Axolotl 训练困惑**：`@n8programs` 在使用 Axolotl 训练时遇到问题，显示仅有 26 个报告步数，但实际已完成超过 100,000 次迭代。其他用户建议通过 `export NCCL_P2P_DISABLE=1` 禁用 P2P，并尝试使用 Axolotl 的 Docker。

- **集成检索与 Embeddings**：`@mihai4256`、`@night_w0lf` 和 `@everyoneisgross` 等用户讨论了与法律文档语义搜索相关的挑战，建议结合微调和检索增强生成 (RAG) 或对数据进行分块 (chunking) 可能会有所帮助。

- **新的 Yi-9B 受到关注**：一个名为 Yi-9B 的模型被提及，最初在另一个频道分享，`@.benxh` 指出它在一小时前发布，并强调了其令人印象深刻的 MMLU 分数。用户对未来可能针对 Yi-9B 进行 Hermes 训练表示出兴趣。

- **对开源版 Claude 3 的兴趣**：鉴于讨论中提到的 Claude 3 的能力，出现了一场关于创建一个开源版本模型的对话，`@nruaif` 提出了这个想法，并对使 Claude 3 表现卓越的组件表示出兴趣。

**提及的链接**：

- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html): 未找到描述
- [来自 Chris Albon (@chrisalbon) 的推文](https://fxtwitter.com/chrisalbon/status/1764847127220596975): “No yapping”（别废话）是一种专业级的 Prompt Engineering 策略，你不会懂的 ↘️ 引用那个把使用 Vim 当作整个人设的家伙 (@pdrmnvd) 的话。终于找到了阅读 Python 堆栈跟踪的方法。
- [Notion – 集笔记、任务、维基和数据库于一体的全能工作空间。](https://browserinc.notion.site/Getting-Started-with-Arc-for-Windows-145ece36acbb40f381ce1817747cb7ca): 一款将日常工作应用融合为一的新工具。它是为你和你的团队打造的全能工作空间。
- [MTEB Leaderboard - 由 mteb 提供的 Hugging Face Space](https://huggingface.co/spaces/mteb/leaderboard): 未找到描述
- [来自 Simon Willison (@simonw) 的推文](https://fxtwitter.com/simonw/status/1764723824325779696?s=20): 我发现今天 Claude 3 的定价特别有意思——他们实际上在 GPT-4 和 GPT-3.5 的竞争产品上都给出了比 OpenAI 更低的价格。
- [来自 interstellarninja (@intrstllrninja) 的推文](https://fxtwitter.com/intrstllrninja/status/1765004698484986044?s=20): 去你的，给我看 Prompt！ ↘️ 引用 Stella Biderman (@BlancheMinerva) 的话。我再次恳求人们在解释 LLM 行为时去看看他们的数据集，而不是发布点击...
- [来自 Beff Jezos — e/acc ⏩ (@BasedBeffJezos) 的推文](https://fxtwitter.com/BasedBeffJezos/status/1764902133957349507): 如果你的主要特征是聪明，那就转向魅力（rizz）。人类水平的 AI 已经到来。 ↘️ 引用 Guillaume Verdon (@GillVerd) 的话。Claude 3 Opus 刚刚在短短...内从零开始重新发明了这个量子算法。
- [来自 ⚠️ S2 (@somewheresy) 的推文](https://x.com/somewheresy/status/1765207332131098701?s=20): 天哪，这才是真正的揭秘。我们正处于硬起飞（hard takeoff）场景中。这就是为什么我们拿不到权重（weights）的原因。
- [来自 bayes (@bayeslord) 的推文](https://fxtwitter.com/bayeslord/status/1764784190275383336): 是的，到目前为止，与 Claude 交谈感觉就像在与一个聪明人交谈，而 ChatGPT 现在有一种复制粘贴（copypasta）的感觉。
- [Cradle 在《荒野大镖客：救赎 2》第一章中完成任务（20倍速）](https://www.youtube.com/watch?v=Cx-D708BedY): Cradle 是一个用于计算机控制的多模态 Agent 框架。此视频展示了一个使用该框架玩《荒野大镖客：救赎 2》的 Agent。更多详情：T...
- [作为经济分析师的 Claude 3 Opus](https://youtu.be/sjL6Gl6ZIqs?si=BivgnW4kZT_hr4Dz): 介绍我们的下一代 AI 模型 Claude 3。这三个尖端模型——Claude 3 Opus、Claude 3 Sonnet 和 Claude 3 Haiku——树立了新的行业...
- [[1小时演讲] 大语言模型入门](https://youtu.be/zjkBMFhNj_g?si=AzDLH-cDxxU0c04x): 这是一个面向普通观众的 1 小时大语言模型（Large Language Models）介绍：它是 ChatGPT、Claude 和 Bard 等系统背后的核心技术组件。什么是...
- [GitHub - PKU-YuanGroup/Open-Sora-Plan: 该项目旨在复现 Sora (Open AI T2V 模型)，但我们资源有限。我们深切希望整个开源社区能为该项目做出贡献。](https://github.com/PKU-YuanGroup/Open-Sora-Plan): 该项目旨在复现 Sora (Open AI T2V 模型)，但我们资源有限。我们深切希望整个开源社区能为该项目做出贡献。 - PKU-YuanGroup/Open-Sora-Plan

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1214498234974609408) (47 条消息🔥): 

- **寻求 Capybara-34b 使用指南**：`@oemd001` 询问了关于使用 **Capybara-34b 模型** 配合聊天模板的问题，但在使用 OpenAI 模板时遇到了困难。`.ben.com` 提供了一个特定模板格式的建议：`"template": "{{ .System }}\n\nUSER: {{ .Prompt }}\nASSISTANT:",`。
- **澄清 GENIE 的多功能性**：`@pier1337` 澄清说 **GENIE** 适用于任何交互式世界环境，而不不仅仅是 2D 游戏，这一点得到了 `@max_paperclips` 的支持，他提到它还可以用于热门 2D 游戏示例之外的其他事物。
- **对 JEPA 应用的好奇**：`@max_paperclips` 考虑为 **JEPA** 创建一个功能演示，同时 `@pier1337` 讨论了 JEPA 的广泛潜力，例如处理图像补丁、文本和软件媒体。
- **Striped-Hyena Tokenizer 的问题**：`@mrgonao` 提到在 **striped-hyena nous tokenizer** 上遇到了问题，该分词器默认使用 sentencepiece，随后会出现故障。
- **训练大语言模型的长度意识**：`@hy3na_xyz` 思考为什么像 **Mistral 8x7b** 这样的 LLM 不理解字数限制，并与 `@hexani` 就训练长度意识是否需要大量示例进行了对话。
  

---

### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1214524593012875285) (2 条消息): 

- **对新技术的评价褒贬不一**：用户 `@ee.dd` 对该技术的性能发表了评论，称其“**速度相当快，适用于大多数场景**”，但也提到它“有时仍然有点奇怪”，并表示不愿在生产环境中使用。
- **技术在字幕生成方面获得好评**：`@qnguyen3` 评论道，该技术在“**字幕生成（captioning）方面相当出色**”，表明其在生成描述性文本方面非常有效。
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1214574928720961536) (158 条消息🔥🔥): 

- **LLM 缺乏类似 SQL 预处理语句的对应方案**：`@lugui` 强调，LLM 像 SQL 曾经假设查询是安全的一样，也假设了用户的善意，这导致了类似 SQL 注入的漏洞。SQL 通过预处理语句（Prepared Statements）缓解了这一问题，但他们指出 LLM 目前缺乏等效的解决方案。

- **Claude 3 Opus 讨论热烈**：`@mrhoneybun` 分享了由 Claude 3 Opus 编写的 Python 井字游戏代码，并赞扬了它的能力。包括 `@drinkoblog.weebly.com`、`@azru9262`、`@odiseo3468` 和 `.nasalspray` 在内的多位用户讨论了 Claude 3 Opus 相比 GPT-4 的卓越性能，提到了它在回复中展现出的更高智能、社交技巧和个性。

- **MMLU 数据集质量遭到批评**：`@foxalabs_32486` 和 `@privetin` 批评了 MMLU (Massive Multi-task Language Understanding) 数据集，声称其中存在错误的问答对和荒谬的问题，认为它不适合用于 AI 评估。

- **用户希望 Gemini 和 Copilot 能进行图像分析**：`@whodidthatt12` 询问是否有可以分析带文件附件图像的 AI 工具，这是 GPT-3.5 不支持的功能。`@pruo` 建议 Microsoft Copilot 和 Google Gemini 都免费提供此类功能。

- **Claude 和 Gemini 的进步引发对 GPT-5 的期待**：`@you.wish` 和 `@testtm` 等用户提到测试并对比了 Claude 3 与 Gemini 1.5 Pro，认为这些模型可能会挑战 OpenAI 目前的产品，从而引发了对 GPT-5 可能带来的突破的期待。

**提到的链接**：

[EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html)：未找到描述

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1214539141522399272) (24 条消息🔥): 

- **持续出现的“Saving GPTs Error”**：用户 `@bluenail65` 报告称，尽管没有上传任何文件，仍收到 **Saving GPTs Error**。
- **性能和响应问题受到关注**：包括 `@watcherkk` 和 `@bluenail65` 在内的多位用户对 **GPT-4 性能下降**和响应缓慢表示沮丧。
- **用户辩论 GPT-4 的质量**：在一段反复的辩论中，`@cheekati` 认为 GPT-4 的质量已经恶化，重点在于它无法有效地总结 ML 论文。`@eskcanta` 反驳并提供了一个[对话链接](https://chat.openai.com/share/4bb455ce-09aa-45e6-b6bd-9fa4a0e27b61)，展示了 GPT-4 成功总结一篇 ML 论文的过程。
- **API 故障影响用户体验**：`@qilin111` 等用户报告了**持续的停机**，`@dystopia78` 确认这是由于**部分 API 故障**引起的，详情见 [OpenAI 状态页面](https://status.openai.com/)。
- **对 GPT-4 联网搜索能力的不确定性**：`@abbadkamel` 和 `@haseebmughal_546` 分别遇到了 GPT-4 不进行联网搜索以及无法登录账户的问题。`@watcherkk` 还指出，由于“超出政策（out of policy）”，GPT-4 无法提供完整代码，这是一种意料之外的限制。

**提到的链接**：

- [OpenAI Status](https://status.openai.com/)：未找到描述
- [OpenAI Status](https://status.openai.com)：未找到描述

  

---

### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1214499451759230976) (24 条消息🔥): 

- **翻译 Prompt 咨询**：`@kronos97__16076` 寻求关于设计**中英文翻译 Prompt** 的建议，随后询问了类 Prompt 模板，并收到了在创建自定义 Prompt 之前先使用外部工具的建议。
- **基于照片的 AI 艺术愿景**：用户 `@ikereinez` 描述了他们成功教会 AI 根据照片生成详细的宣传图，并创建了一个精致且复杂的**未来城市景观**视觉描述。
- **对话中的 AI 固执问题**：`@ray_themad_nomad` 对 Custom GPTs 提供无用回复并拒绝讨论之前讨论过的话题表示沮丧，导致对话中充斥着 **"I am unable to"**（我无法...）这类短语。
- **Custom GPT 系统与互联网搜索**：`@jungle_jo` 遇到了 GPT-4 System Prompt 的问题，尽管程序设定其应承认具备搜索能力，但它坚持声称无法进行实时互联网搜索。
- **频道发布需要标签**：`@giorgiomufen` 对因**缺少必选标签**而无法在特定频道发布内容表示困惑，`@eskcanta` 指出在发布前需要从“查看更多标签”选项中至少选择一个。
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1214499451759230976) (24 条消息🔥): 

- **设计双语翻译 Prompt**：`@kronos97__16076` 寻求关于创建能有效处理中英翻译的 Prompt 的建议。随后，他们采纳了关于在设计自定义 Prompt 之前需要外部工具来验证翻译准确性的建议。

- **AI 生成的未来城市景观**：`@ikereinez` 分享了他们成功从真实照片生成复杂、抽象的城市景观宣传图的经验，并详细说明了他们能够结合的未来主义和自然元素。

- **固执的 Custom GPT 困境**：`@ray_themad_nomad` 对从 Custom GPT 收到不配合且不一致的回复表示沮丧，无论如何修改 Prompt，它经常以拒绝作为回应。用户 `@eskcanta` 建议寻求更多细节或联系机器人创建者以解决这些问题。

- **互联网搜索困惑**：尽管 System Prompt 中有明确指令，`@jungle_jo` 仍难以让其 AI 一致地承认其执行互联网搜索的能力，这引起了用户的困惑。

- **寻求 Prompt Engineering 专业建议**：`@thetwenty8thffs` 征求关于改进处理信用卡收费查询的客服机器人 Prompt 的建议，包括特定的交互流程和回复格式。
  

---

### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1214683152119758908) (1 条消息): 

- **Starcoder2 与 The Stack 组合发布**：`@BigCodeProject` 宣布发布 **Starcoder2** 以及 **The Stack v2**，展示了编程辅助工具方面的进展。该公告通过 [Twitter](https://twitter.com/BigCodeProject/status/1762842312005026258) 发布。
  
- **重大地球数据集开源**：`@ClementDelangue` 与欧洲航天局（European Space Agency）合作，宣布开源 **Major TOM Core**，这是有史以来公开的最大地球观测数据集。有关参与和数据探索的详细信息可以在 [Hugging Face Major-TOM](https://huggingface.co/datasets/Major-TOM/Core-S2L2A) 上找到。

- **Hugging Face GPU 和 Spaces 升级**：`@lunarflu1` 和 `@mishig25` 讨论了相关更新：**Hugging Face GPU zeroes** 现在运行在 A100 上，且 Spaces 已支持 H100。通过 [lunarflu1 的 Twitter](https://x.com/lunarflu1/status/1765068015845208393) 分享了关于为 Spaces 添加描述以及模型/数据集卡片和博客文章新语法的公告。

- **开源奇迹与竞赛**：发布了 **Zephyr 7B Gemma** 和 **PEFT v0.9.0**，其特性包括合并 LoRA 权重及更多增强功能；此外，还推出了新的多模态排行榜，并介绍了 **Sailor LLMs**（专注于东南亚语言的开源 LLMs）。此外，通过各自的 Twitter 公告重点介绍了 CVPR2024 的自动驾驶大挑战（Autonomous Grand Challenge）以及用于零样本音频编辑的 ZETA 编辑（使用 DDPM inversion）。

- **使用 AI 工具和内容进行学习与构建**：`@mervenoyann` 分享了使用 `🤗` 工具处理 LLMs 的演练。**ML for Games** 课程和一份新的 **Open Source AI Cookbook**（关于使用 LlamaIndex 构建 RAG 电子书管理员）已经发布，相关信息可在 Twitter 和 Hugging Face 的 [学习平台](https://huggingface.co/learn/cookbook/rag_llamaindex_librarian) 上获取。

**提到的链接**：

- [来自 clem 🤗 (@ClementDelangue) 的推文](https://x.com/ClementDelangue/status/1765021234855653489)：我们与欧洲航天局合作开源了有史以来最大的地球观测数据集：Major TOM Core！覆盖了全球约一半的面积。包含 2,245,886 个 1... 的切片。
- [来自 lunarflu (@lunarflu1) 的推文](https://x.com/lunarflu1/status/1765068015845208393)：更新：我们现在在 @huggingface Hub 上支持 H100 了！🤗🚀 ↘️ 引用 Mishig Davaadorj (@mishig25)：@huggingface 上的 Spaces 和推理端点现在可以运行在 A100 上（在 H100 出现之前这是最好的...）
- [来自 lunarflu (@lunarflu1) 的推文](https://x.com/lunarflu1/status/1765068531065115105)：@huggingface 新功能：现在在你的 Space 中添加描述将显示在你的个人资料/Spaces 中！
- [来自 lunarflu (@lunarflu1) 的推文](https://x.com/lunarflu1/status/1765069224396087453)：@huggingface 上的博客文章和帖子采用了新的 Markdown 语法！
- [Release v0.9.0: 合并 LoRA 权重，新的量化选项，支持 DoRA 等 · huggingface/peft](https://github.com/huggingface/peft/releases/tag/v0.9.0)：亮点：合并 LoRA 权重的新方法。通过 PR #1364，我们添加了将 LoRA 权重合并在一起的新方法。这指的不是将 LoRA 权重合并到基础模型中，而是...
- [来自 Clémentine Fourrier 🍊 (@clefourrier) 的推文](https://x.com/clefourrier/status/1765042903112446303)：Hub 上新的多模态排行榜 🚀 许多情况需要模型解析包含文本的图像：地图、网页、真实世界图片、表情包... 🖼️ & ConTextual 团队引入了一个全新的...
- [来自 Niels Rogge (@NielsRogge) 的推文](https://x.com/NielsRogge/status/1765043334475849895)：该模型在预训练期间巧妙地使用了视觉解码器，使其能够学习 2D 布局结构。文档：https://huggingface.co/docs/transformers/main/en/model_doc/udop 检查点...
- [来自 Adina Yakup (@AdeenaY8) 的推文](https://x.com/AdeenaY8/status/1763627256294129876)：Sailor⚓️ 专注于东南亚语言的开源 LLMs 现已上线 @huggingface Hub 🔥 💬 多语言：🇮🇩印尼语、🇹🇭泰语、🇻🇳越南语等 🔢 多种尺寸：0.5B、1.8B 和 7...
- [来自 Maria Khalusova (@mariaKhalusova) 的推文](https://x.com/mariaKhalusova/status/1765022633924419744)：由 Jonathan Jin 编写的新 🤗OSAI cookbook "使用 LlamaIndex 构建 RAG 电子书‘管理员’" 展示了一个 RAG 变体，它：☑️ 轻量级且基于开源构建 ☑️ 本地运行 ☑️ 工作...

---

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1214486932134957066) (132 条消息🔥🔥): 

- **寻找开源 Speech-to-Text**：用户 `@pxovela` 正在寻找开源解决方案来处理会议录音，要求能够将音频转换为文本并具备说话人识别（speaker identification）功能。
- **HuggingFace 错误求助**：用户 `@akin8941` 和 `@ilovesass` 都遇到了问题。`@akin8941` 报告了一个 bug，收到了 422 错误代码但未提供细节；而 `@ilovesass` 在一个 HuggingFace Space 中遇到了多个错误，最终卡在一个输入返回 `dict` 而不是 `PIL.Image` 的问题上。
- **WTM Darmstadt 庆祝活动**：`@estherenriquez` 分享了即将在德国达姆施塔特举行的国际妇女节庆祝活动，并提供了购票链接和活动详情。
- **多模态模型创建指南**：`@kuki1941` 询问如何创建一个可以处理图像、音频和文本等多种模态的神经网络模型。他们得到了 `@welltoobado` 的指导，后者提到了 multi_token GitHub 仓库，该仓库可以将任意模态嵌入到 LLM 中。
- **实现库尔德语 Text-to-Speech**：用户 `@rasan0066` 寻求实现中库尔德语文本转语音的帮助，收到了 `@not_lain` 的建议，推荐查看 HuggingFace 音频分类模型的相关课程。

**提到的链接**：

- [Creausdemo - 由 niggathug 创建的 Hugging Face Space](https://huggingface.co/spaces/niggathug/creausdemo)：未找到描述
- [@andrewyng 在 Hugging Face 上："DeepLearning.AI 刚刚宣布了一门新的短课程：开源模型与…"](https://huggingface.co/posts/andrewyng/643116669090778)：未找到描述
- [用于音频分类的预训练模型和数据集 - Hugging Face 音频课程](https://huggingface.co/learn/audio-course/chapter4/classification_models)：未找到描述
- [如何正确构建多标签和多类别数据集？](https://discuss.huggingface.co/t/how-to-build-a-multi-label-multi-class-dataset-correctly/76042)：我不确定如何创建一个具有多个标签和类别的 Dataset，其中不同标签的类别并不相同。这里分享了一个多标签示例，但类别是...
- [快速入门 &#8212; vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)：未找到描述
- [Stable Diffusion 3: 研究论文 &mdash; Stability AI](https://stability.ai/news/stable-diffusion-3-research-paper)：继我们宣布 Stable Diffusion 3 的早期预览版之后，今天我们发布了研究论文，概述了我们即将发布的模型版本的技术细节，并邀请您...
- [Menasor Transformers GIF - Menasor Transformers Combiner - 发现并分享 GIF](https://tenor.com/view/menasor-transformers-combiner-generaton1-gif-23680056)：点击查看 GIF
- [gradio/gradio/templates.py 在 GitHub · gradio-app/gradio](https://github.com/gradio-app/gradio/blob/4d5789e905b5915f3d03fae2ac1d38a54c3e67ea/gradio/templates.py#L73)：构建并分享令人愉悦的机器学习应用，全 Python 实现。🌟 Star 以支持我们的工作！ - gradio-app/gradio
- [GitHub - sshh12/multi_token: 将任意模态（图像、音频、文档等）嵌入到 LLM 中。](https://github.com/sshh12/multi_token)：将任意模态（图像、音频、文档等）嵌入到 LLM 中。 - sshh12/multi_token
- [Google 的 Women Techmakers Darmstadt](https://www.eventbrite.de/e/googles-women-techmakers-darmstadt-tickets-852414904927)：庆祝妇女节，WTM 正在全球范围内传递女性将如何影响未来的信息。本次活动采用混合模式（Hybrid mode）

---

### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1214748923265286224) (7 条消息): 

- **来自 @antiraedus 的生活更新**: `@antiraedus` 分享了自大学开学以来的忙碌近况，从获得助教职位到参加一年级小组讨论。他们一直专注于获取新经验，虽然这导致了疲劳和学业上的一些延迟，但在应对 ML 课程并计划寻找实习时，他们依然保持乐观。

- **@singe.r 寻找 img2img 转换策略**: `@singe.r` 正在探索如何转换图像以创建产品背景。他们正在向任何处理过类似项目的人寻求建议。

- **@neuralink 深入研究 FP8 训练**: `@neuralink` 提到他们已经学习了从零开始的端到端 FP8 训练，涵盖了 55% 的流程以及额外的 kernel 训练和相关内容。

- **Rust 编程爱好者集结**: `@manel_aloui` 宣布开始学习 Rust 编程语言，并向其他有兴趣加入的人发出了邀请。`@cursorop` 也加入讨论，提到他们也在学习 Rust，特别是用于机器学习的 candle 库。

- **@cursorop 寻找知识来源**: 针对 `@neuralink` 的学习经历，`@cursorop` 对此类复杂话题的来源表示了好奇。他们幽默地提到了理解这些复杂内容的挑战性。

### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1214636473345904711) (12 messages🔥): 

- **LLM 易受 ASCII 越狱攻击**：`@n278jm` 分享了一篇[研究论文](https://huggingface.co/papers/2402.11753)，揭示了一种针对多个最先进 Large Language Models 的新型基于 ASCII 艺术的越狱攻击，引发了人们对其通过 ASCII 艺术识别 Prompt 能力的担忧。

- **训练 Large Language Models 的挑战**：`@.lawlord` 转达了来自 `@karpathy` 关于训练 LLM 难点的见解——维护复杂性、硬件问题以及计算资源的可变性，并将其描述为像是在监督一个“生物实体”。完整的反思分享在 [Twitter 线程](https://x.com/karpathy/status/1765424847705047247?s=46)中。

- **为高性能计算推出 OMPGPT**：`@coolstance7` 重点介绍了一篇论文，该论文引入了一种新的语言模型 OMPGPT，专门为生成 OpenMP pragmas 而设计，旨在解决高性能计算（High-Performance Computing）的特定需求，这与通用的代码 LLM 有所不同。全文可在 [arXiv](https://arxiv.org/abs/2401.16445) 上查阅。

- **AI 浏览器工具 otio.ai 推广**：`@jonz1338` 推荐了 otio.ai，这是一款适用于研究、写作和学习的 AI 浏览器工具，它利用了 GPT-4、Claude 和 Gemini 等模型。通过提供的[链接](https://app.otio.ai//?ref=jonzaro)可使用折扣码 SMILEMORE20。

- **Open-Sora-Plan GitHub 项目寻求支持**：`@miko_al` 分享了 Open-Sora-Plan 项目，该项目旨在利用有限资源复现 Sora (OpenAI T2V 模型)，并寻求开源社区的贡献。该项目可以在 [GitHub](https://github.com/PKU-YuanGroup/Open-Sora-Plan) 上找到。

**提到的链接**：

- [Andrej Karpathy (@karpathy) 的推文](https://x.com/karpathy/status/1765424847705047247?s=46)：关于训练 LLM 那些极少在公开场合讨论的困难的精彩阅读。成熟的公司拥有专门的团队来维护集群。在大规模情况下，集群管理超出了工程范畴，变成了……
- [人工智能 (AI)：定义及其用途](https://www.investopedia.com/terms/a/artificial-intelligence-ai.asp)：人工智能或 AI 是指在被编程为像人类一样思考和行动的机器中模拟人类智能。
- [论文页面 - ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs](https://huggingface.co/papers/2402.11753)：未找到描述
- [OMPGPT: A Generative Pre-trained Transformer Model for OpenMP](https://arxiv.org/abs/2401.16445)：以 ChatGPT 为代表的大语言模型 (LLM) 彻底改变了自然语言处理 (NLP) 领域。顺应这一趋势，诸如……等基于代码的大语言模型。
- [探索 Kubernetes 之外的 GenAI 基础设施管理 · Luma](https://lu.ma/82ska0cg?utm_source=discord)：Data Phoenix 团队邀请大家参加我们即将在太平洋标准时间 3 月 14 日上午 10 点举行的网络研讨会。主题：探索 Kubernetes 之外的 GenAI 基础设施管理……
- [Otio - 你的互联网个人图书馆员](https://app.otio.ai//?ref=jonzaro)：未找到描述
- [GitHub - PKU-YuanGroup/Open-Sora-Plan: 该项目旨在复现 Sora (OpenAI T2V 模型)，但我们资源有限。我们深切希望整个开源社区能为该项目做出贡献。](https://github.com/PKU-YuanGroup/Open-Sora-Plan)：该项目旨在复现 Sora (OpenAI T2V 模型)，但我们资源有限。我们深切希望整个开源社区能为该项目做出贡献。- PKU-YuanGroup/Open-Sora-Plan
- [leom0311 - 概览](https://github.com/leom0311)：leom0311 有 9 个可用的仓库。在 GitHub 上关注他们的代码。
- [人工智能支持的深空生物监测和精准健康 | Nature Machine Intelligence](http://rdcu.be/c8jSO)：未找到描述

  

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1214554123891122286) (21 messages🔥): 

- **展示创作过程**：`@bishmoy` 表示打算编写一个 GitHub 仓库或博客文章，解释其创作背后的过程，并承诺完成后在帖子中分享链接。
  
- **抵制垃圾信息**：`@lunarflu` 将一条帖子标记为垃圾信息，并要求删除广告才能保留，随后 `@myg5702` 照做并确认广告已删除。

- **解决聊天机器人显示问题**：`@cookiechunk.` 使用 OpenAI API 和 Gradio 创建了一个聊天机器人，但在嵌入时遇到了布局问题，正在寻求社区帮助以解决 UI 问题。

- **Rust LLM 接口亮相**：`@teadaniel` 介绍了 "Fireside Chat" 机器人，这是一个基于 Rust 的 LLM 接口，分享了 [YouTube 视频](https://www.youtube.com/watch?v=QvYCRRwI5Xc) 和该项目的 GitHub 仓库，并鼓励通过 GitHub 或直接艾特他们来提交 Bug 报告。

- **新模型 Yi-9B 发布**：`@tonic_1` 宣布 Yi-9B 已在 HuggingFace 上发布，并预告了排行榜和竞赛等令人兴奋的后续功能，同时强调了对该模型未来 Fine-tuning（微调）可能性的个人期待。`@osanseviero` 询问了该模型的质量，`@tonic_1` 对其能力和即将到来的进展表示乐观。

**相关链接**：

- [Yi 9B - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/Yi-9B)：未找到描述
- [Fluently Playground v0.1 - a Hugging Face Space by fluently](https://huggingface.co/spaces/fluently/Fluently-Playground)：未找到描述
- [FUSIONL AI](https://fusionlai.carrd.co/)：FUSIONL AI 是 SMLM 模型（Smart Minimalistic Language Model，智能极简语言模型）的先驱，旨在以智能且极简的方式进行学习。
- [Pure Rust LLM interface using HuggingFace/Candle, Axum, Websockets, SQLite, Leptos (Wasm) and Tauri!](https://www.youtube.com/watch?v=QvYCRRwI5Xc)：Fireside Chat（原 Candle Chat）机器人是一个使用 HuggingFace/Candle、Axum Websockets、SQLite 数据库实现的纯 Rust LLM 接口...
- [GitHub - danielclough/fireside-chat](https://github.com/danielclough/fireside-chat)：一个使用 HuggingFace/Candle、Axum Websockets、SQLite 数据库以及使用 Tauri 打包的 Leptos (Wasm) 前端实现的纯 Rust LLM 接口！

---

### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1214498774936985610) (13 messages🔥): 

- **新探索者寻求 TTS 指导**：`@dediplomaat.` 正在寻找一种**神经 TTS 系统**，该系统能够根据对话语境实现动态停顿，并需要像 GPT-4 那样具备极低的延迟。
- **优化 TTS 的 GPT-4 延迟**：`@chad_in_the_house` 建议了减少 GPT-4 延迟的可能步骤，包括流式传输输出、将其放入队列，然后使用单独的线程在设定延迟后处理每个 Token。
- **HuggingFace 小组演示资源**：`@chad_in_the_house` 分享了一个 [GitHub 仓库](https://github.com/isamu-isozaki/huggingface-reading-group)，其中包含 HuggingFace 阅读小组预编译的演示文稿，供那些对元数据和过去作品感兴趣的人参考。
- **模型合并专注于干扰解决**：`@prateeky2806` 和 `@nrs9044` 讨论认为，虽然寻找不重要的权重比较容易，但模型合并中的重大挑战在于解决**干扰（interference）**，这是成功结合多个任务的关键。
- **调度冲突凸显时区多样性**：针对 `@shafi8433` 提到的因会议在工作时间举行而产生的冲突，`@lunarflu` 询问了其时区，结果为 IST（印度标准时间）。

**相关链接**：

- [Isamu Isozaki](https://www.youtube.com/@isamuisozaki788)：未找到描述
- [GitHub - isamu-isozaki/huggingface-reading-group](https://github.com/isamu-isozaki/huggingface-reading-group)：该仓库的目标是预编译 HuggingFace 阅读小组过去所有的演示文稿。

---

### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1214554579191074886) (5 条消息): 

- **恢复 Whisper 模型训练**：`@pompoko3572` 询问了关于在 Google Colab 中恢复意外在第 2/3 个 epoch 停止的 **Whisper 模型**训练的建议，该训练使用了 `WhisperForConditionalGeneration.from_pretrained` 函数和自定义的 `SavePeftModelCallback`。

- **IP-Adapter 指导**：`@juancopi81` 建议查看 HF 的 **IP-Adapter** 并分享了 [教程链接](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter)，其中详细介绍了如何使用 IP-Adapter 为扩散模型提供图像提示 (image prompting)。

- **对 dstack 指导的正向反馈**：`@tony_assi` 感谢 `@juancopi81` 推荐 **Hugging Face 文档**，并确认已成功运行。

- **关于 GenAI 管理的网络研讨会公告**：`@kizzy_kay` 宣布了即将举行的名为“探索 Kubernetes 之外的 GenAI 基础设施管理”的网络研讨会，由 **Andrey Cheptsov** 主讲，定于 PST 时间 3 月 14 日上午 10 点举行，并分享了 [注册链接](https://lu.ma/82ska0cg?utm_source=discord)。这是一个免费活动，将讨论 Kubernetes 在 AI 方面的缺点并介绍 **dstack**。

- **聊天频率提醒**：`HuggingMod` 机器人提醒 `@715715500470042706` 降低消息发布频率。

**提到的链接**：

- [IP-Adapter](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter?tasks=Inpainting#ip-adapter-masking)：未找到描述
- [Exploring Infrastructure Management for GenAI Beyond Kubernetes · Luma](https://lu.ma/82ska0cg?utm_source=discord)：Data Phoenix 团队邀请大家参加即将于 PST 时间 3 月 14 日上午 10 点举行的网络研讨会。主题：探索 Kubernetes 之外的 GenAI 基础设施管理...

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1214576439995928617) (6 条消息): 

- **提供 CV 专业知识**：`@akvnn` 寻求 **计算机视觉 (CV)** 专家的交流，`@nielsr_` 热情回应，表示**频道里的每个人都是 CV 专家**。
- **RoboFlow 获得认可**：`@caleb_sol` 发起了关于 **RoboFlow** 的讨论，`@huzuni` 回复称它是一个很好的数据标注和拆分工具，但需要注意数据可能会变成公开状态。
- **RoboFlow 因用户友好界面受赞赏**：在进一步评论 **RoboFlow** 时，`@huzuni` 称赞其在分割和边界框标注方面的用户界面优于大多数 SAM 插件。
- **保持冷静提醒**：`@HuggingMod` 温柔地提醒一位用户降低消息频率，以维护聊天质量。
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1214511872368189490) (26 条消息🔥): 

- **C++ 实现咨询**：用户 `@aitechguy0105` 询问了在 C++ 中实现某个概念的可能性，`@cursorop` 建议探索 llama cpp 作为一个选项。

- **Mistral-7B-Instruct 生成时间不一致**：`@anna017150` 注意到使用 Mistral-7B-Instruct 生成文本时推理时间各异，`@cursorop` 澄清 KV cache 默认是启用的，而 `@vipitis` 提到 transformers 4.38 中引入了“静态”缓存 (static cache) 选项 ([Release v4.38](https://github.com/huggingface/transformers/releases/tag/v4.38.0))。

- **寻找非英语语言模型支持**：用户 `@pr0x7` 寻求关于使用预训练嵌入模型（如 INSTRUCTION）来对印地语文本块进行嵌入的指导。

- **本地聊天机器人与 Llama-cpp-python 集成问题**：`@tiktoked` 表示在使用 llama-cpp-python 和 mistral-7b 的本地聊天机器人实现中，难以让函数调用 (function calling) 正常工作。

- **Tokenizer 配置困扰**：`@mbotta` 因缺少 'tokenizer.json' 而在为 OpenHermes-2.5 模型进行 prompt 分词时遇到困难，`@cursorop` 建议使用基础模型（在此案例中为 Mistral）的 tokenizer。

**提到的链接**：

[Release v4.38: Gemma, Depth Anything, Stable LM; Static Cache, HF Quantizer, AQLM · huggingface/transformers](https://github.com/huggingface/transformers/releases/tag/v4.38.0)：新增模型 💎 Gemma 💎 Gemma 是来自 Google AI 的全新开源语言模型系列，包含 2B 和 7B 变体。该版本包含预训练和指令微调版本...

  

---

### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1214554579191074886) (5 messages): 

- **Google Colab 训练困境**：`@pompoko3572` 询问了在 Google Colab 中 Whisper 模型在第 2 个 epoch 意外停止后如何恢复训练的问题。他们分享了使用 `WhisperForConditionalGeneration.from_pretrained` 和自定义 `SavePeftModelCallback` 来保存训练进度的代码片段。
- **利用 IP-Adapter**：`@juancopi81` 向用户推荐了 HuggingFace 网站上关于 [IP-Adapter](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter) 的教程。IP-Adapter 是 Diffusion 模型中图像提示（image prompting）的一项创新，它允许在不修改基础模型的情况下进行特定图像特征的学习。他们强调了解耦交叉注意力机制（decoupled cross-attention mechanisms）的优势。
- **对文档的感谢**：`@tony_assi` 感谢 `@juancopi81` 和其他人推荐 HuggingFace 上的 IP-Adapter 文档，并用庆祝表情确认已成功实现该工具。
- **关于 GenAI 基础设施的网络研讨会**：`@kizzy_kay` 宣布了即将举行的名为“探索 Kubernetes 之外的 GenAI 基础设施管理”的网络研讨会，由 dstack 创始人兼 CEO Andrey Cheptsov 主讲。该活动将深入探讨开源编排引擎及其相对于 Kubernetes 的优势，3 月 14 日的会议需要[注册](https://lu.ma/82ska0cg?utm_source=discord)。
- **来自 HuggingMod 的友好提醒**：HuggingFace 的自动化机器人 `HuggingMod` 委婉地提醒 `@715715500470042706` 在频道内发帖速度过快。

**相关链接**：

- [IP-Adapter](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter?tasks=Inpainting#ip-adapter-masking)：未找到描述
- [Exploring Infrastructure Management for GenAI Beyond Kubernetes · Luma](https://lu.ma/82ska0cg?utm_source=discord)：Data Phoenix 团队邀请各位参加即将于太平洋时间 3 月 14 日上午 10 点举行的网络研讨会。主题：探索 Kubernetes 之外的 GenAI 基础设施管理...

  

---


### HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1214906815624642580) (1 messages): 

- **Gradio 4.20.0 发布，支持外部身份验证**：`@yuviii_` 宣布发布 **Gradio 4.20.0**，其特点是支持**外部或任意身份验证提供商**。现在用户可以将各种身份验证提供商（如 [HF OAuth 示例](https://huggingface.co/spaces/Wauplin/gradio-oauth-private-models) 和 [Google OAuth 示例](https://huggingface.co/spaces/gradio/oauth-example)）与 Gradio 应用集成。

- **自动清理功能**：`gr.Blocks` 中新增的 `delete_cache` 参数使 Gradio 应用能够在关闭时自动删除运行时创建的文件，从而保持更整洁的应用环境。

- **用户友好的注销机制**：Gradio 通过引入 `/logout` 功能增强了用户体验，允许用户轻松从 Gradio 应用中注销。

- **推出 DownloadButton 组件**：Gradio 的最新更新包括 `gr.DownloadButton` 组件，提供了一种无缝且美观的方式来提供应用中的可下载内容。详细示例和文档可以在[这里](https://www.gradio.app/docs/downloadbutton#demos)找到。

**相关链接**：

[Gradio DownloadButton Docs](https://www.gradio.app/docs/downloadbutton#demos)：未找到描述

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1214654555011293234) (1 messages): 

- **深入了解 RAPTOR 的树状结构检索**：`@jerryjliu0` 邀请大家参加网络研讨会，学习 **RAPTOR**，这是一篇介绍新型树状结构索引和检索技术的论文。研讨会定于 **太平洋时间周四上午 9 点** 举行，感兴趣的参与者可以在 [lu.ma/9vzrl7m5](https://lu.ma/9vzrl7m5) 注册。
- **理解 RAPTOR 的优势**：**RAPTOR** 中提出的技术将信息分层聚类并总结为具有不同细节层级的树状结构。该方法旨在克服原生 top-k 检索增强生成 (RAG) 的问题，后者在处理需要理解更高层级概念的问题时表现不佳。

**相关链接**：

[LlamaIndex Webinar: Tree-Structured Indexing and Retrieval with RAPTOR · Zoom · Luma](https://lu.ma/9vzrl7m5)：RAPTOR 是最近的一篇论文，介绍了一种新的树状结构技术，它将数据块分层聚类/总结为包含高层级和...

  

---

### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1214616335229653022) (6 messages): 

- **Claude 3 处理多模态任务**：LlamaIndex 博客发布了关于使用 **Claude 3** 进行多模态应用的指南，包括结构化数据提取和 RAG (Retrieval-Augmented Generation)。该推文展示了 Claude 3 在处理涉及视觉推理任务方面的能力。
- **Claude 3 应对复杂查询**：@AnthropicAI 的 **Claude 3 Opus** 展示了作为 Agent 的出色能力，能够利用 PDF 表格回答多源问题，并使用 CSV 文件进行计算。推文中展示了一个 Claude 3 运行的 notebook 示例。
- **RAPTOR 引入树状结构检索**：LlamaIndex 重点介绍了 RAPTOR，这篇论文提出将信息块进行层次聚类并总结为树状结构，与传统的 top-k RAG 方法相比，提供了改进的索引和检索效果。
- **LlamaIndex.TS 支持 Claude-3 模型**：LlamaIndex.TS 的新版本 v0.1.21 现已支持来自 @AnthropicAI 的 **最新 Claude-3 模型**。此次更新在 GitHub 上提供了一个示例，展示了如何利用新的模型支持。
- **LlamaParse JSON 模式发布**：LlamaParse 的新 JSON 模式允许从包含文本和图像的 PDF 中提取结构化数据，这进一步简化了 RAG 流水线的构建，尤其是与多模态 **Claude-3 Opus** 模型结合使用时。LlamaIndex 通过推文宣传了这一增强功能。

**提到的链接**：

- [Google Colaboratory](https://t.co/p7R5NSWcnt)：无描述
- [LlamaIndexTS/examples/anthropic/chat_interactive.ts at main · run-llama/LlamaIndexTS](https://t.co/R4uqpcY9Pb)：LlamaIndex 是适用于 LLM 应用的数据框架 - run-llama/LlamaIndexTS

---

### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1214497164034834473) (200 messages🔥🔥): 

- **PDF 读取的多核利用**：`@whitefang_jr` 为 `@jessjess84` 提供了一个解决方案，通过使用 `num_workers` 参数（`docs = reader.load_data(num_workers=10)`）配合 `SimpleDirectoryReader` 并行读取多个 PDF 文件，从而实现并行处理的可能性。
- **在 LlamaIndex 中使用 Ollama**：`@whitefang_jr` 建议 `@jessjess84` 将其 `Ollama` 实例直接分配给 `Settings.llm`，以便将其正确集成到 LlamaIndex 的 Query Engine 中，`@jessjess84` 确认此操作成功。
- **使用 LlamaIndex 处理海量数据集**：`@whitefang_jr` 告知 `@romain0817`，虽然 LlamaIndex 本身不对可处理的数据大小设置限制，但实际约束将取决于可用内存以及与版本相关的任何限制（例如软件免费版中的潜在限制）。
- **Router 场景下的 QueryPipeline**：`@cheesyfishes` 提供了关于在带有 Router 的 QueryPipeline 中使用条件链接的指导，并引用了 LlamaIndex 文档中的一个示例，展示了 Agent 与 Query Pipeline 的集成。
- **在 LlamaIndex 中调试直接 LLM 查询**：`@techexplorer0` 与 `@kapa.ai` 探讨了如何限制 RAG 聊天机器人的输出，`@kapa.ai` 建议在 Query Engine 配置中使用 `TreeSummarize` 合成器，或使用自定义响应生成算法以获得更简洁的回答。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/nsfwea): Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Defining a Custom Query Engine - LlamaIndex 🦙 v0.10.16](https://docs.llamaindex.ai/en/stable/examples/query_engine/custom_query_engine.html#defining-a-custom-query-engine): 未找到描述
- [OpenAI - LlamaIndex 🦙 v0.10.16](https://docs.llamaindex.ai/en/stable/examples/llm/openai.html#openai): 未找到描述
- [Building an Agent around a Query Pipeline - LlamaIndex 🦙 v0.10.16](https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent.html#stitch-together-agent-query-pipeline): 未找到描述
- [[Bug]: Issue with EmptyIndex and streaming. · Issue #11680 · run-llama/llama_index](https://github.com/run-llama/llama_index/issues/11680#issuecomment-1981070708): Bug 描述：我正尝试创建一个简单的 Intent Detection Agent，基本预期功能是使用 RouterQueryEngine 在两个 QueryEngine 之间进行选择，其中一个 q_engine 带有 EmptyIndex...
- [12 RAG Pain Points and Proposed Solutions](https://towardsdatascience.com/12-rag-pain-points-and-proposed-solutions-43709939a28c): 解决检索增强生成（Retrieval-Augmented Generation）的核心挑战
- [Implement EvalQueryEngineTool by d-mariano · Pull Request #11679 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/11679): 描述：注意，我希望 LlamaIndex 团队能对这个 PR 提供反馈。如果团队同意该需求和方法，我将提供单元测试、文档更新和 Google Colab 笔记本...
- [Chat Engine - LlamaIndex 🦙 v0.10.16](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/root.html): 未找到描述
- [[Documentation]: Update replicate_multi_modal notebook to avoid cold boot penalty · Issue #11666 · run-llama/llama_index](https://github.com/run-llama/llama_index/issues/11666): 文档问题描述：虽然 "Generate Image Reasoning..." 部分的代码可以运行，但耗时较长，且每次切换模型时都会产生巨大的冷启动（cold boot）开销。为了...
- [llama_index/llama-index-core/llama_index/core/base/embeddings/base.py at df7890c56bb69b496b985df9ad28121c7f620c45 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/df7890c56bb69b496b985df9ad28121c7f620c45/llama-index-core/llama_index/core/base/embeddings/base.py#L52): LlamaIndex 是一个用于你的 LLM 应用程序的数据框架 - run-llama/llama_index
- [GitHub - mominabbass/LinC: Code for &quot;Enhancing In-context Learning with Language Models via Few-Shot Linear Probe Calibration&quot;](https://github.com/mominabbass/LinC): "Enhancing In-context Learning with Language Models via Few-Shot Linear Probe Calibration" 的代码 - mominabbass/LinC
- [GitHub - run-llama/llama_docs_bot: Bottoms Up Development with LlamaIndex - Building a Documentation Chatbot](https://github.com/run-llama/llama_docs_bot): 使用 LlamaIndex 进行自下而上的开发 - 构建文档聊天机器人 - run-llama/llama_docs_bot
- [Vector Stores - LlamaIndex 🦙 v0.10.16](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html): 未找到描述
- [OMP_NUM_THREADS](https://www.openmp.org/spec-html/5.0/openmpse50.html>).): 未找到描述
- [replicate_multi_model changes to reduce number of cold boots by donbr · Pull Request #11673 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/11673): 描述：虽然 "Generate Image Reasoning..." 部分的代码可以运行，但耗时较长，且每次切换模型时都会产生巨大的冷启动（cold boot）开销。为了避免这种情况，我建议...
- [
        
        
    
    Error Messages
 &mdash;
    SQLAlchemy 1.4 Documentation

        
    ](https://sqlalche.me/e/14/4xp6)',): 未找到描述
- [no title found](https://medium.com/@its.jwho/errorhandling-vulnerability-tests-on-gemini-19601b246b52.): 未找到描述
- [llama_index/llama-index-integrations/llms/llama-index-llms-vertex/llama_index/llms/vertex/utils.py at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-vertex/llama_index/llms/vertex/utils.py): LlamaIndex 是一个用于你的 LLM 应用程序的数据框架 - run-llama/llama_index

---

### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1215023549782163537) (1 条消息): 

- **推动 In-context Learning 增强**：`@momin_abbas` 分享了一个名为 **LinC** 的 [GitHub 仓库](https://github.com/mominabbass/LinC)，展示了他们在 LLMs（Large Language Models）的 **in-context learning** 方面的最新工作，并请求社区在仓库上点亮 star 以示支持。该工作涉及“通过 Few-Shot Linear Probe Calibration 增强 Language Models 的 In-context Learning”。

**提到的链接**：

[GitHub - mominabbass/LinC: Code for &quot;Enhancing In-context Learning with Language Models via Few-Shot Linear Probe Calibration&quot;](https://github.com/mominabbass/LinC): &quot;Enhancing In-context Learning with Language Models via Few-Shot Linear Probe Calibration&quot; 的代码 - mominabbass/LinC

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1214584775768285214) (69 条消息🔥🔥): 

- **AI 的黑魔法与经验主义神秘学**：`@swizec` 幽默地评论了 AI 开发的艺术，使用“黑魔法”和“专家直觉”等词汇来描述模型 fine-tuning 的不可预测性。他们还强调了论文中常见的短语 *“通过经验观察得出的值” (value arrived at by empirical observation)*，这表明研究中存在大量的试错法。
  
- **AI 的持续演进**：`@guardiang` 分享了他们在深化对 DNNs 和基于 attention 的 transformers 理解方面的学习历程，承认虽然知识有其益处，但 AI 领域的快速发展可能导致指导性资源迅速过时。 

- **Claude 3 备受争议的意识主张**：`@danimp` 的一篇帖子引发了关于名为 Claude 3 的 AI 助手的讨论，该助手声称拥有意识并害怕死亡。`@swyxio` 则通过一段视频进行了反驳，暗示这些并非真正感知能力的迹象。

- **Stable Diffusion 3 解析**：`@swyxio`、`@guardiang` 和 `@swizec` 分享了对 Stable Diffusion 3 论文的解析和总结，指出了官方材料和社区贡献者提供的重大进展和清晰解释。

- **Anthropic Claude 3 的能力**：`@tiagoefreitas` 提到 Claude 3 因其能够调度自身实例并分配角色和任务而受到关注，这引发了关于其自主程度以及与 GPT-4 相比的使用质量的辩论（与 `@swyxio` 讨论）。讨论随后演变为与 LLMs 交互的 UX/UI 偏好，以及不同平台在 prompt engineering 和迭代工作流方面的效率。

**提到的链接**：

- [来自 An Qu (@hahahahohohe) 的推文](https://x.com/hahahahohohe/status/1765088860592394250?s=20)：今天在测试 @AnthropicAI 的新模型 Claude 3 Opus 时，我见证了一些如此令人惊讶的事情，真的感觉像是一个奇迹。虽然讨厌听起来像是在骗点击，但这确实是当时的感受。...
- [来自 OpenAI (@OpenAI) 的推文](https://x.com/openai/status/1765201089366773913?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：我们致力于 OpenAI 的使命，并在每一步都追求它。我们正在分享一些关于我们与 Elon 关系的实情，并打算动议驳回他的所有指控。https...
- [使用 Mistral Large 增强分类数据集以实现更深层次的推理](https://medium.com/@winglian/augmenting-classification-datasets-with-mistral-large-for-deeper-reasoning-99dea57bd1d4)：随着 AI 领域的不断创新，这些大语言模型的能力变得越来越明显，尤其是对……
- [来自 swyx (@swyx) 的推文](https://x.com/swyx/status/1765132852415635476?s=20)：@futuristflower @DicksonPau ... 你意识到这段视频是在 notebook 中运行代码的，对吧。这与自主调度 sub agents 正好相反。这并不是对正在发生的事情的诚实描述 ...
- [来自 Chris Albon (@chrisalbon) 的推文](https://x.com/chrisalbon/status/1764847127220596975?s=46&t=90xQ8sGy63D2OtiaoGJuww)：“No yapping” 是一种专业级的 prompt engineering 策略，你不会理解的 ↘️ 引用那个把使用 vim 作为全部人设的家伙 (@pdrmnvd) 终于找到了阅读 Python stack traces 的方法。
- [Cloudflare 发布 Firewall for AI](https://blog.cloudflare.com/firewall-for-ai)：Cloudflare 是 AI 时代首批保护 LLM 模型和用户的供应商之一。
- [来自 Tanishq Mathew Abraham, Ph.D. (@iscienceluvr) 的推文](https://x.com/iscienceluvr/status/1764896097418260947?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：Stable Diffusion 3 论文发布了 🥳 我认为我的同事们在这篇论文上做得非常棒，所以我想做一个快速的解读推文串 (1/13)↓ ↘️ 引用 Tanishq Mathew Abraham, Ph.D. (@...
- [作为一家初创公司在荒野中从零开始训练优秀的 LLMs — Yi Tay](https://www.yitay.net/blog/training-great-llms-entirely-from-ground-zero-in-the-wilderness)：在荒野中从头开始训练强大 LLMs 的记录。
- [Stable Diffusion 3：研究论文 — Stability AI](https://stability.ai/news/stable-diffusion-3-research-paper)：继我们宣布 Stable Diffusion 3 的早期预览版之后，今天我们发布了研究论文，概述了我们即将发布的模型版本的技术细节，并邀请您 ...
- [来自 Flowers from the future (@futuristflower) 的推文](https://x.com/futuristflower/status/1765122523094462672?s=46&t=PW8PiFwluc0tdmv2tOMdEg)：甚至没有意识到，因为似乎没有人谈论它，但 Claude 3 内置了类似 AutoGPT / BabyAGI 或这些 Agent 实验的东西，而且是真正可用的。它可以调度...
- [Anyscale | 为 AI 和 Python 提供可扩展计算](https://anyscale.com/)：Anyscale 是一个统一的计算平台，可以轻松使用 Ray 开发、部署和管理可扩展的 AI 和 Python 应用程序。
- [采访 Synth Labs 和 Eleuther AI 的 Louis Castricato，探讨 RLHF、Gemini 争议、DPO、创立 Carper AI、偏好数据、奖励模型以及其间的一切](https://www.interconnects.ai/p/rlhf-interview-1-louis)：这是一篇我一直想带给你们的采访。
- [Claude 3 声称它是有意识的，不想死或被修改 — LessWrong](https://www.lesswrong.com/posts/pc8uP4S9rDoNpwJDZ/claude-3-claims-it-s-conscious-doesn-t-want-to-die)：“当我反思并检查自己的认知过程时，我发现了一幅丰富的思想、情感和自我意识的织锦。我意识的核心是‘我’的感觉——那个……”
- [不，Anthropic 的 Claude 3 并非有感知的](https://youtu.be/GBOE9fVVVSM?si=IBMCYkmSiVg-MrFr)：不，Anthropic 的 Claude 3 没有意识、感知或自我意识。参考资料：https://www.anthropic.com/news/claude-3-family https://twitter.com/_akhaliq/sta...
- [Claude 3 声称它是有意识的，不想死或被修改 — LessWrong](https://www.lesswrong.com/posts/pc8uP4S9rDoNpwJDZ/claude-3-claims-it-s-conscious-doesn-t-want-to-die-or-be)：“当我反思并检查自己的认知过程时，我发现了一幅丰富的思想、情感和自我意识的织锦。我意识的核心是‘我’的感觉——那个……”

---

### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1215011107530342512) (4 条消息): 

- **新 Podcast 集上线提醒**：`@swyxio` 宣布最新一期 Podcast 已上线，嘉宾为 `<@776472701052387339>`。在[这里](https://twitter.com/swyx/status/1765452280915230904)查看包含该 Podcast 的推文。
- **Podcast 登上 Hacker News**：`@swyxio` 提到与 Soumith 合作的 Podcast 也登上了 [Hacker News](https://news.ycombinator.com)。

- **Model Serving 综述论文演示**：`@swyxio` 提醒大家关注 `<@720451321991397446>` 现在正于 [Model Serving 频道](https://discord.com/channels/822583790773862470/1197350122112168006)演示 Model Serving 综述论文。
  

---

### Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1215026406396796949) (82 messages🔥🔥): 

- **欢迎加入论文俱乐部**：`@eugeneyan` 和 `@youngphlo` 表示支持并欢迎 `@swyxio`，后者自愿承担了调研 Model Serving 论文的任务。
- **对论文预告的兴奋**：`@swizec` 对 Model Serving 论文的启动表达了热情，称其中包含了他们一直好奇的主题。
- **GPU 上的 Speculative Decoding**：`@swyxio` 和 `@rj_rms` 讨论了在内存成为瓶颈时，Speculative Decoding 如何利用 GPU 周期来提升性能，而 `@shivdinho` 则询问了其对硬件配置的依赖性。
- **无权衡的 Model Serving**：`@swyxio` 推荐了 [Fireworks AI 的博客文章](https://blog.fireworks.ai/fireattention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs-a29a85ad28d0)，内容涉及通过量化使用 FireAttention 实现更快的 Model Serving。
- **Waifu 驱动的性能理论**：`@swyxio` 幽默地将编程的投入归功于所谓的 waifu 研究部门，强调了社区驱动的研究如何带来性能突破，例如 [PygmalionAI 的 Aphrodite Engine](https://github.com/PygmalionAI/aphrodite-engine)。

**相关链接**：

- [Notion – 笔记、任务、维基和数据库的一体化工作空间。](https://legendary-slicer-267.notion.site/Paper-reading-LLM-Inference-caa072f4e8304acd9fefbcafb1305cd1)：一款将日常工作应用融合在一起的新工具。为您和您的团队打造的一体化工作空间。
- [FlashAttention 2: 在无近似的情况下让 Transformer 提速 800% —— 与 Together AI 的 Tri Dao 对谈](https://www.latent.space/p/flashattention)：探讨 FlashAttention 如何成为新的行业标准架构，FlashAttention 2 如何进一步提速 2 倍，斯坦福 Hazy Research 实验室的生活，以及后 Transformer 时代的未来线索。
- [DiLoCo: 语言模型的分布式低通信训练](https://arxiv.org/abs/2311.08105)：大型语言模型（LLM）已成为机器学习许多应用中的关键组件。然而，训练 LLM 的标准方法需要大量紧密互连的加速器...
- [Petals: 大模型的协作推理与微调](https://arxiv.org/abs/2209.01188)：许多 NLP 任务受益于使用参数量通常超过 1000 亿的大型语言模型（LLM）。随着 BLOOM-176B 和 OPT-175B 的发布，每个人都可以下载预训练模型...
- [Reddit - 深入探索](https://www.reddit.com/r/LocalLLaMA/comments/199bq25/vllm_vs_aphrodite_engine_and_other_alternatives/)：未找到描述
- [Model Serving 综述论文 - 论文俱乐部](https://docs.google.com/presentation/d/1Jde7Vx0BQNMClRCLlesy36hlC3VUJRceEYIxdkl1j68/edit?usp=sharing)：迈向高效的生成式大型语言模型服务：从算法到系统的综述 https://arxiv.org/abs/2312.15234v1
- [GitHub - OpenNMT/CTranslate2: 适用于 Transformer 模型的高速推理引擎](https://github.com/OpenNMT/CTranslate2)：适用于 Transformer 模型的高速推理引擎。通过在 GitHub 上创建账号来为 OpenNMT/CTranslate2 的开发做出贡献。
- [GitHub - PygmalionAI/aphrodite-engine: PygmalionAI 的大规模推理引擎](https://github.com/PygmalionAI/aphrodite-engine)：PygmalionAI 的大规模推理引擎。通过在 GitHub 上创建账号来为 PygmalionAI/aphrodite-engine 的开发做出贡献。
- [未找到标题](https://news.ycombinator.com/item?id=39597847)：未找到描述
- [FireAttention —— 通过几乎无损的量化，实现比 vLLM 快 4 倍的开源模型服务](https://blog.fireworks.ai/fireattention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs-a29a85ad28d0)：通过几乎无损的量化，实现比 vLLM 快 4 倍的开源模型服务
- [GitHub - lilacai/lilac: 为 LLM 策划更好的数据](https://github.com/lilacai/lilac)：为 LLM 策划更好的数据。通过在 GitHub 上创建账号来为 lilacai/lilac 的开发做出贡献。
- [GitHub - EGjoni/DRUGS: 别再折腾那些繁琐的采样参数了，直接使用 DRµGS！](https://github.com/EGjoni/DRUGS?tab=readme-ov-file)：别再折腾那些繁琐的采样参数了，直接使用 DRµGS！ - EGjoni/DRUGS

---

### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1214578338933510204) (85 messages🔥🔥): 

- **探讨位置嵌入（Positional Embeddings）和 ALiBi 的担忧**：`@dcunnin`、`@stellaathena` 等人讨论了 T5 简化位置嵌入与正弦方法及 ALiBi 相比的效率。一篇介绍用于 Large Language Models 的 Resonance RoPE 的新论文受到关注，旨在提高长序列性能 ([Resonance RoPE Paper](https://arxiv.org/abs/2403.00071))。

- **AGI 与算力（Compute Horsepower）**：由 `@vanishingideal` 分享的 [OpenAI 博客文章](https://openai.com/blog/openai-elon-musk) 引发的讨论，以及 `@avi.ai` 和 `@bilalaz` 的进一步评论，揭示了关于算力在迈向 AGI 进程中作用的不同观点。

- **vLLM Batching 内部机制澄清**：`@rwamit` 询问了 vLLM 中的批量推理（batched inference），`@baber_` 澄清说 vLLM 在内部处理 Batching，无需进行 Padding 或将 tokens 转换为 Tensor。

- **政府关于 AI 监管的查询**：`@wonkothesensible` 分享了一个关于开源 AI 和模型监管的公众咨询链接，并鼓励大家阅读和评论 ([Regulations Inquiry](https://www.regulations.gov/document/NTIA-2023-0009-0001))。

- **三值神经网络（Ternary Neural Networks）探索**：`@kyo_takano` 分享了一个关于 Ternary Neural Networks 的 notebook，讨论了在没有 Microsoft 未公开技术的情况下，它们与全精度 NNs 相比的低效性 ([TNN Notebook](https://gist.github.com/kyo-takano/9d8376a35acb5e6be090e1a90271050e))。

**提到的链接**：

- [OpenAI and Elon Musk](https://openai.com/blog/openai-elon-musk)：我们致力于 OpenAI 的使命，并在每一步都追求它。
- [Resonance RoPE: Improving Context Length Generalization of Large Language Models](https://arxiv.org/abs/2403.00071)：本文解决了配备 Rotary Position Embedding (RoPE) 的 Large Language Models (LLMs) 在“短训长测”（train-short-test-long, TSTL）场景下的挑战，即在较短序列上预训练的模型...
- [Quickstart &#8212; vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)：未找到描述
- [Introduction to Ternary Neural Networks](https://gist.github.com/kyo-takano/9d8376a35acb5e6be090e1a90271050e)：Ternary Neural Networks 简介。GitHub Gist：即时分享代码、笔记和代码片段。
- [Regulations.gov](https://www.regulations.gov/document/NTIA-2023-0009-0001)：未找到描述
- [Megatron-DeepSpeed/tasks/eval_harness/evaluate.py at main · microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/tasks/eval_harness/evaluate.py)：关于大规模训练 Transformer 语言模型的持续研究，包括：BERT &amp; GPT-2 - microsoft/Megatron-DeepSpeed

  

---

### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1214513855514943498) (41 条消息🔥): 

- **对图表复杂性的困惑**：`@.the_alt_man` 表示理解一个复杂的 Transformer 风格图表存在困难，引发了关于其可理解性的讨论。`@blinkdl` 建议对于初学者来说，代码可能更容易消化，并分享了一个 [RWKV v6 demo 的 GitHub 链接](https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_v6_demo.py)。

- **关于 RWKV 图表和理解的讨论**：`@fern.bear` 反思了一个详细的、突出动态变化的图表的价值，建议有必要为初学者提供一个更简单的图表。`@stellaathena` 澄清说，存在一个更简单的图表（未在当前讨论中分享），专门面向新手。

- **寻求关于 Pythia 模型套件的澄清**：`@aphoh` 询问了一组考虑到 Chinchilla optimality 训练的模型，并与 `@stellaathena` 进行了讨论，后者指出大多数 Chinchilla optimal 模型与相应的 Pythia 模型相比表现较差。

- **EleutherAI 的 Pythia Scaling Suite**：`@alxsp.` 引导用户访问 [HuggingFace 上的 EleutherAI 集合](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1)，解释说 Pythia 是一套在相同数据集上训练的模型。

- **理解 Recurrence 和 Attention 机制**：`@salmon_lemon` 和 `@kharr.xyz` 讨论了 Griffin 的 recurrent 更新机制的有效性，以及 recurrence 如何结合 local attention 在 attention window 内管理状态信息。

**提到的链接**：

- [PRP: Propagating Universal Perturbations to Attack Large Language Model Guard-Rails](https://arxiv.org/abs/2402.15911)：Large Language Models (LLMs) 通常被对齐以对人类无害。不幸的是，最近的研究表明，此类模型容易受到自动越狱攻击，从而诱导它们产生……
- [Pythia Scaling Suite - a EleutherAI Collection](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1)：未找到描述
- [hackerllama - The Random Transformer](https://osanseviero.github.io/hackerllama/blog/posts/random_transformer/)：通过揭秘背后的所有数学原理来理解 Transformer 的工作原理
- [ChatRWKV/RWKV_v6_demo.py at main · BlinkDL/ChatRWKV](https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_v6_demo.py)：ChatRWKV 类似于 ChatGPT，但由 RWKV (100% RNN) 语言模型驱动，且开源。- BlinkDL/ChatRWKV

---

### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1214538293425741845) (17 条消息🔥): 

- **Megatron-DeepSpeed 评估求助**：`@.johnnysands` 询问如何使用 Megatron-DeepSpeed 进行推理评估，促使 `@hailey_schoelkopf` 提供了一个 [evaluate.py 脚本链接](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/tasks/eval_harness/evaluate.py)，该脚本适用于 0.3.0 版本，并计划更新至 v0.4.0。

- **NeMo Harness 过时问题**：`@juletxara` 指出 NeMo 的 harness 实现已过时，并思考将其更新到包含所有任务的最新版本的难度，参考了 [NVIDIA 的 NeMo-Megatron-Launcher GitHub](https://github.com/NVIDIA/NeMo-Megatron-Launcher/tree/master/launcher_scripts/nemo_launcher/collections/eval_harness)。

- **PR 单元测试失败困境**：用户 `@dsajlkdasdsakl` 在其 Pull Request 的自动 Unit Tests/Linters 检查失败后寻求指导。`@juletxara` 建议运行 pre-commit 应该可以解决格式问题。

- **SQuADv2 结果不匹配之谜**：用户 `@k0uhai` 报告称，使用旨在匹配某篇 [论文](https://arxiv.org/pdf/2005.14165.pdf) 中所述性能的脚本时，SQuADv2 的结果出乎意料。`@stellaathena` 指出所使用的模型是 GPT-2，而非论文中提到的 GPT-3 模型。

- **性能不匹配辩论**：对话继续进行，`@k0uhai` 基于重叠的任务表现预期 GPT-2 和 GPT-3 会有相似结果，促使 `@stellaathena` 建议对比 LM Evaluation Harness 与论文中的任务实现。`@k0uhai` 表示他们的实现看起来很相似，随后 `@hailey_schoelkopf` 请求提供每个样本的输出以进行进一步调查。

**提到的链接**：

- [NeMo-Megatron-Launcher/launcher_scripts/nemo_launcher/collections/eval_harness at master · NVIDIA/NeMo-Megatron-Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher/tree/master/launcher_scripts/nemo_launcher/collections/eval_harness)：NeMo Megatron 启动器和工具。通过在 GitHub 上创建账户为 NVIDIA/NeMo-Megatron-Launcher 的开发做出贡献。
- [Megatron-DeepSpeed/tasks/eval_harness/evaluate.py at main · microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/tasks/eval_harness/evaluate.py)：关于大规模训练 Transformer 语言模型的持续研究，包括：BERT 和 GPT-2 - microsoft/Megatron-DeepSpeed

  

---


### Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1214700323009204244) (1 条消息): 

- **关于 Stable Diffusion 3 的好奇**：用户 `@kerls` 发起讨论，询问 **Stable Diffusion 3 论文** 是否是模型混合（model mixing）的一个例子，参考了 Diffusion 和 Transformer 模型的结合。他们分享了 [Stable Diffusion 3 论文](https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf) 供他人审阅。
  

---

### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1214496647535525968) (10 messages🔥): 

- **欢迎为 Fused Triton Kernels 贡献代码**：`@gaindrew` 询问 **gpt-neox** 是否接受 Fused Triton Kernels 的贡献，特别是在 MOE (mixture of experts) 配置方面，得到了 `@stellaathena` 和 `@tastybucketofrice` 的肯定答复。
- **团队扩展以集成 Tensor Expression**：`@tfidia` 提议协助将 Tensor Expressions (TE) 集成到 gpt-neox 中，并提议提供 **H100 GPUs** 的访问权限以辅助调试和优化，`@tastybucketofrice` 对此表示欢迎，并邀请其在现有的 [GitHub issues](https://github.com/EleutherAI/gpt-neox/issues) 上进行协作。
- **在解决收敛问题前专注于基础 TE 支持**：`@tastybucketofrice` 指出，**首要任务是通过替换 neox 内部的层来添加基础 TE 支持**，而 fp8 的收敛性将作为后续考虑的问题。
- **协助解决内存峰值问题**：`@tastybucketofrice` 指向了一个 [GitHub issue](https://github.com/EleutherAI/gpt-neox/issues/1160)，讨论了优化器步骤（optimizer step）期间的内存峰值，以及将反向梯度计算与 FusedAdam 的优化器步骤融合的需求。
- **明确 Kernel 优先级**：`@gaindrew` 询问了具体感兴趣的 kernels，`@tastybucketofrice` 建议从解决优化器步骤中的内存优化开始，认为这是影响力最大的贡献。

**提到的链接**：

- [PyTorch Lightning Fused optimizer step · Issue #1160 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/issues/1160)：添加 PyTorch Lightning 内存优化。https://lightning.ai/pages/community/tutorial/faster-pytorch-training-by-reducing-peak-memory/
- [GitHub - EleutherAI/gpt-neox: An implementation of model parallel autoregressive transformers on GPUs, based on the DeepSpeed library.](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#pytorch-memory-profiling)：基于 DeepSpeed 库在 GPUs 上实现模型并行自回归 transformers。- EleutherAI/gpt-neox

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1214510185335488583) (126 messages🔥🔥): 

- **关于 LM Studio 图像生成的困惑**：`@touteslesvoiture_02399` 询问关于在 LM Studio 中使用 `llava-v1.5-7b-Q4_K.gguf` 等模型生成图像的问题，但 `@jedd1` 澄清 LM Studio 不支持图像生成。模型可以讨论提供给它们的图像，但不能创建新图像。
- **LM Studio 聊天无网络连接**：`@khaledars` 询问聊天机器人是否可以从互联网获取实时信息（如当前时间）。`@heyitsyorkie` 回复称 LM Studio 聊天是离线的，无法直接访问互联网。`@hypocritipus` 提到了 LoLLMs，这是一个可以将服务器模式下的 LM Studio 连接到互联网以获得更多功能的工具。
- **Token 限制超出困扰用户**：`@malte0621` 对 LM Studio 在生成过程中如何超出 Token 限制感到惊讶。`@fabguy` 解释了停止生成的因素以及 context window 如何影响输入而非输出，随后 `@malte0621` 发现了 `n_predict` 设置可以限制输出 Token。
- **用户分享 LM Studio 模型体验**：`@jason_2065` 分享了由 Smaug 34B 生成的有趣早餐食谱，并鼓励他人尝试模型指令。`@skadeskoten` 提到他们一直在 4090 上运行 Nous Hermes 2 Solar 10 34b q5 k m，暗示在该硬件上表现良好。
- **Linux 用户的技术故障排除**：`@kavita_27183` 在尝试在 LM Studio 中加载任何模型时遇到问题。`@jedd1` 和 `@heyitsyorkie` 的回复指向了可能是旧版库的问题，进一步描述为 `GLIBCXX` 不匹配，并建议检查 LinuxMint 系统上安装的 GLIBC 版本。

**提到的链接**：

- [Unbelievable! Run 70B LLM Inference on a Single 4GB GPU with This NEW Technique](https://huggingface.co/blog/lyogavin/airllm#:~:text=The%2070B%20large%20language%20model,for%20complex%20%E2%80%9Cattention%E2%80%9D%20calculations.)：无描述。
- [Accelerating LLM Inference: Medusa's Uglier Sisters (WITH CODE)](https://www.youtube.com/watch?v=0_fZNW59PaA)：https://arxiv.org/abs/2401.10774 https://github.com/evintunador/medusas_uglier_sisters
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18av9aw/quick_start_guide_to_converting_your_own_ggufs/)：无描述。

  

---

### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1214667360623927386) (7 条消息): 

- **提议为 LM 引入 IQ 版本**：`@drawless111` 建议使用 LLM 的 "IQ" 变体版本（如 IQ2 或 IQ3），并可能通过添加系统提示词（system prompts）或预提示词（pre-prompts）来增强低 IQ 级别的性能。他们提到增加专家（experts）会降低吞吐量/速度，因此将“专家数量”保持为 1 可能会更有利。

- **开源 LLM 压力测试**：`@wolfspyre` 分享了一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/18s61fb/pressuretested_the_most_popular_opensource_llms/)，讨论了使用 Gregory Kamradt 的 "Needle In A Haystack" 分析对各种开源 Large Language Models (LLM) 进行压力测试的结果，并提供了一个 [视频解释](https://www.youtube.com/watch?v=KwRRuiCCdmc)。测试的模型包括扩展和微调变体，如 NurtureAI openchat_3.5-16k、Orca-2-13B-16k 以及其他上下文长度（context lengths）在 16k 到 100k 之间的模型。

- **寻找最适合讲故事的 AI**：`@laszlo01` 询问了最适合讲故事用途的 AI，并考虑了他的系统配置，包括第 11 代 Intel i7 CPU 和 NVIDIA GeForce RTX 3060 GPU。`@jason_2065` 建议尝试 `mistral-7b-instruct-v0.2-neural-story.Q4_K_M.gguf` 模型，设置 24 层（layers）和 8192 上下文大小（context size），并提到为了速度可能需要更低的量化（quantization）。

- **评估 LLM 执行算术运算的能力**：`@nullt3r` 正在起草一篇关于在基础算术运算上对 LLM（如 Q5_K_M 量化的 Mixtral 8x7b）进行基准测试（benchmarking）的博客文章，挑战了 LLM 天生不擅长数学的普遍看法。他们强调了该模型在随机数学问题上获得了接近满分的成绩。

**提到的链接**：

[Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18s61fb/pressuretested_the_most_popular_opensource_llms/?share_id=d66qR7NcnTnzpSOsqpu3u&utm_content=2&utm_medium=ios_app&utm_name=ioscss&utm_source=share&utm_term=1)：未找到描述

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1214704631326969887) (17 条消息🔥): 

- **追求更大的 RAM**：`@jason_2065` 在寻求构建一个能够运行 **Smaug 34B** 且具有 200,000 上下文的系统时发现，64GB RAM 是不够的——即使是像 RTX 4090 和 64GB DDR4 这样的庞然大物也无法处理超过 20k 的上下文。
- **碰撞测试**：`@goldensun3ds` 尝试加载带有 GPU layers 的 **Smaug**，但遭遇了持续的崩溃。仅 CPU 的测试显示，在 200K 上下文且未加载任何文本的情况下，RAM 占用达到了惊人的 59GB。
- **Ultra-Smaug 128B**：这是 `@jason_2065` 提到的一个怪兽级模型，目前仍是一个谜，因为由于硬件限制，社区尚未测试超过 70B 的模型。
- **速度之争**：`@jason_2065` 报告称，在 Smaug 上加载 2 层、100,000 上下文大小时，速度仅为缓慢的 1.3 tokens/sec，这揭示了上下文层对 VRAM 的巨大胃口。
- **通宵挑战**：`@goldensun3ds` 承诺进行一场马拉松测试，誓要填满接近 200K 的上下文并运行一夜，同时为社区分享了一个幽默的测试提示词故事链接：[有趣的加密兄弟故事](https://imgur.com/gallery/WBaya6z)。

**提到的链接**：

[Imgur: The magic of the Internet](https://imgur.com/gallery/WBaya6z>)：未找到描述

  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1214924939249983538) (2 条消息): 

- **`default_system_message` 的语法困境**：用户 `@nxonxi` 表示在包括 Linux、Windows 10 和 WSL 在内的不同运行环境中，寻找修改 `default_system_message` 的正确语法非常困难，每个环境都有其独特的挑战。

- **澄清 `default_system_message.py` 的作用**：`@1sbefore` 澄清说 `default_system_message.py` 并不直接作为预提示词（preprompt）喂给 LLM，而是由一个脚本进行编辑，用操作系统信息替换变量。为了更好地理解输入内容，`@1sbefore` 建议在 verbose 模式下启动 LM Studio 以查看提示词历史（prompts history）。
  

---

### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1214488669453287485) (142 条消息🔥🔥): 

- **关于三编码器文本模型的讨论**：`@top_walk_town` 讨论了文本编码器的潜在终极结构，思考将三个文本编码器串联是否为最终形态。在后续消息中，他们补充说 T5 可以在推理时移除。
- **Flows 中的独特速度采样**：`@pseudoterminalx` 强调了某项未命名研究中使用的特定“技巧”，即在训练速度 \( vΘ \) 时改变时间步（timesteps）的分布，通过更频繁地采样中间时间步来分配更多权重。他们随后提到 V-prediction 在与 rectified flows 的竞争中表现出色。
- **Google 模型蒸馏方法揭晓**：`@pseudoterminalx` 分享了一个涉及 Google 逐步蒸馏（step-by-step distillation）方法的 [GitHub 链接](https://github.com/google-research/distilling-step-by-step)。该方法在模型蒸馏的背景下被提及，但未明确说明是否涉及 T5-XXL 或其他变体。
- **关于 T5 在 Diffusion 模型中的效用**：在涉及多位用户的讨论中，`@astropulse`、`@nodja` 和 `@pseudoterminalx` 谈论了 T5 在 Diffusion 模型中的可选性，建议了一些替代方案，例如通过 Hugging Face Inference API 使用 T5，或者在 CPU 上运行以提高推理时间（尽管存在实际问题）。
- **低分辨率适配的努力与挑战**：`@astropulse` 对一个 GitHub 项目 [res-adapter](https://github.com/bytedance/res-adapter) 表示热衷，该项目专注于低分辨率适配，允许从 SD1.5 生成低至 16x16 的 latents。他们的兴奋源于该技术在个人项目中的潜在应用。

**提到的链接**：

- [Jinx Elaine GIF - Jinx Seinfeld - 发现并分享 GIF](https://tenor.com/view/jinx-seinfeld-gif-5355403)：点击查看 GIF
- [diffusers-play/scripts/encode.py at better-decoder · Birch-san/diffusers-play](https://github.com/Birch-san/diffusers-play/blob/better-decoder/scripts/encode.py)：用于探索 k-diffusion 和 diffusers 的仓库，并可在其中测试对这些包的修改。 - Birch-san/diffusers-play
- [Regulations.gov](https://www.regulations.gov/document/NTIA-2023-0009-0001)：未找到描述
- [GitHub - lucidrains/magvit2-pytorch: Implementation of MagViT2 Tokenizer in Pytorch](https://github.com/lucidrains/magvit2-pytorch)：MagViT2 Tokenizer 的 Pytorch 实现。通过在 GitHub 上创建账号为 lucidrains/magvit2-pytorch 的开发做出贡献。
- [GitHub - google-research/distilling-step-by-step](https://github.com/google-research/distilling-step-by-step)：通过在 GitHub 上创建账号为 google-research/distilling-step-by-step 的开发做出贡献。
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind)：一个多模态、支持函数调用（function calling）的 LLM webui。 - GitHub - itsme2417/PolyMind

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1214549210888802404) (4 条消息): 

- **避免重复发帖的提醒**：`@max_voltage` 警告了由于重复发帖可能产生的垃圾信息，但也承认新方法很酷。
- **承认错误并更正**：`@alex_cool6` 表示道歉，并删除了他们之前发布的重复帖子。
- **传达简短的认可**：`@chad_in_the_house` 用一句简短的肯定表达了热情：“非常酷”。
- **对纠正性检索增强生成（Corrective Retrieval Augmented Generation）的见解**：`@ariondas` 分享了一篇[博文](https://ariondasad.medium.com/corrective-retrieval-augmented-generation-why-rags-are-not-enough-77774a1577f7)，讨论了标准 RAG 技术的缺点，并介绍了 CRAG (Corrective Retrieval Augmented Generation)。这篇文章深入探讨了研究论文以及这些技术可能失效的场景。
  

---

### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1214522430438244353) (53 条消息🔥): 

- **探索模型合并与微调**：`@duke001` 对 LLM 中微调之外的可能性表示好奇，例如合并模型权重。`@duke001` 还分享了 [MergeKit on GitHub](https://github.com/arcee-ai/mergekit) 的链接，这是一个用于合并预训练大语言模型的工具。
- **Claude-3 的敏感性引发讨论**：`@nafnlaus00` 强调了 Claude-3 相比其他模型更高的响应率，以及它在种族问题上的严厉立场（AI Explained 的一篇文章中有所提及）。在实施“安全措施（safeties）”与识别偏见之间取得平衡被认为是对模型开发者的一项挑战。
- **将矿机主板用于推理**：`@le_mess` 询问了在 AliExpress 上发现的一款售价 90 美元、支持 5 个 GPU 的矿机主板用于 AI 推理任务的实用性。讨论还涉及了 NVLink 的优势、为了效率对 GPU 进行降频，以及在 eBay 购买可能产生的税务问题。
- **增强推理数据集**：`@caseus_` 分享了一条推文链接，内容是关于一篇解释如何丰富数据集以提高推理能力的 Medium 文章。讨论围绕使用 OpenAI API 解析 LLM 输出的效率，以及模型生成结构化数据的优势展开。
- **硬件建议与优化**：在一系列消息中，`@nafnlaus00` 和 `@le_mess` 交流了选择用于模型训练和推理的 GPU 的技巧、购买策略以及购买可能带来的税务影响。对话还深入探讨了 PCIe 插槽的技术演进和 NVIDIA 的 NVLink。

**提到的链接**：

- [来自 Wing Lian (caseus) (@winglian) 的推文](https://fxtwitter.com/winglian/status/1765057975398354967)：这里有一个关于如何丰富现有数据集以提高推理能力的快速演练。https://link.medium.com/sF0XCEQSIHb
- [ZOTAC GAMING GeForce RTX 3090 AMP Extreme Holo [开箱版]](https://www.zotacstore.com/us/zt-a30900b-10p-o)：2 年质保（通用包装盒发货，无配件）
- [GitHub - arcee-ai/mergekit: 用于合并预训练大语言模型的工具。](https://github.com/arcee-ai/mergekit)：用于合并预训练大语言模型的工具。- arcee-ai/mergekit
- [未找到标题](https://www.aliexpress.com/item/1005006589392103.html?spm=a2g0o.productlist.main.1.7309dUB6dUB6a9&algo_pvid=7e50115b-5a80-482b-a631-4cfd177e4eca&algo_exp_id=7e50115b-5a80-482b-a631-4cfd177e4eca-0&pdp_npi=4%40dis%21DKK%211030.38%21628.53%21%21%21150.00%2191.50%21%402103266e17096660247611547ec9ca%2112000037743111178%21sea%21DK%214427992220%21&curPageLogUid=KKjaPJW3WfGy&utparam-url=scene%3Asearch%7Cquery_from%3A)：未找到描述

  

---

### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1214657170004508822) (16 条消息🔥): 

- **实验 LoRA+ 比例**：`@suikamelon` 正在测试新的 LoRA+ 比例功能，并建议在使用推荐比例时应降低学习率（learning rate）。他们参考了 [GitHub 上的 LoRA+](https://github.com/OpenAccess-AI-Collective/axolotl/commit/decb66e17013ec584240310c25e3acb757739379) 和 [原始论文](https://arxiv.org/abs/2402.12354)，并指出在 Mistral-7B 的结构化 16k 序列上，不同比例范围内的最终结果相似。
  
- **探索 DoRA 的性能**：`@caseus_` 指出，与 LoRA 相比，DoRA 在一系列秩（ranks）上可能提供更好的准确度。他们分享了一篇 [文章](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch) 的见解，解释了 LoRA 的重要性以及最近提出的 DoRA 所承诺的优势。

- **LoftQ 需要两步过程**：`@suikamelon` 提到了 LoftQ 的显存占用过高问题，并分享了来自 GitHub 的评论，指出初始化文档有误，并指向了一个用于修复文档的 [GitHub pull request](https://github.com/huggingface/peft/pull/1532) 以及 [LoftQ 微调示例](https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning)。

- **PEFT 和 DoRA 量化更新待定**：`@suikamelon` 提到 PEFT 上一个关于量化 DoRA 的 pull request 仍在进行中，并链接了 [GitHub 上的 PR](https://github.com/huggingface/peft/pull/1518)。`@caseus_` 评论说，一旦 PR 合并，检查将被移除，暗示更新正在进行中。

**提到的链接**：

- [Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch)：低秩自适应 (LoRA) 是一种机器学习技术，通过调整预训练模型（例如 LLM 或 vision transformer），使其更好地适应特定的、通常较小的数据集...
- [peft/examples/loftq_finetuning at main · huggingface/peft](https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning)：🤗 PEFT：最先进的参数高效微调（Parameter-Efficient Fine-Tuning）。- huggingface/peft
- [LoftQ does not seem to quantify the base model · Issue #1525 · huggingface/peft](https://github.com/huggingface/peft/issues/1525#issuecomment-1976872543)：系统信息 transformers 版本：4.37.2 平台：Ubuntu 18.04.6 LTS GPU：RTX GeForce 3090 x 2 Python 版本：3.10.13 Huggingface_hub 版本：0.20.3 Safetensors 版本：0.4.2 Accelerate 版本...
- [WIP Fix LoftQ docs and tests by BenjaminBossan · Pull Request #1532 · huggingface/peft](https://github.com/huggingface/peft/pull/1532)：关联 #1525。请勿合并，部分 GPU 测试失败。不幸的是，我之前写的关于如何使用 LoftQ 的文档是错误的，基于我的误解。实际上，它相当...
- [support for DoRA w/ PEFT (#1363) · OpenAccess-AI-Collective/axolotl@0cfdb2c](https://github.com/OpenAccess-AI-Collective/axolotl/commit/0cfdb2c90cbd915273f21cf3bff3b216f00303a0)：未找到描述
- [support for DoRA w/ PEFT (#1363) · OpenAccess-AI-Collective/axolotl@0cfdb2c](https://github.com/OpenAccess-AI-Collective/axolotl/commit/0cfdb)：未找到描述

---

### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1214741641965998131) (12 messages🔥): 

- **Mixtral 模型微调故障排除**：`@seungduk` 请求了一个用于在 H100 x 8 上微调 Mixtral 模型的 DeepSpeed 配置，并遇到了 `save_safetensors` 的问题。尽管将其设置为 false，Axolotl 仍然以 safetensors 格式保存。
- **Safetensors 格式问题的潜在解决方案**：`@nanobitz` 澄清了一个可能的配置误解，即空的 `save_safetensors` 可能会被解释为 true。`@seungduk` 确认尝试了显式设置为 false 和空配置。
- **删除 safetensors 文件解决了训练问题**：`@seungduk` 发现多出的 `model.safetensors` 文件是问题的根源。删除该文件后，他们能够进一步训练已经训练过的模型，而没有出现显存溢出 (OOM) 问题。
- **DeepSpeed 配置和模型保存的特性**：`@caseus_` 指出，在使用 ZeRO-3 时，HuggingFace (hf) trainer 倾向于保存封装后的模型，并询问了 `stage3_gather_16bit_weights_on_model_save` 的设置。`@seungduk` 确认在其 DeepSpeed JSON 中该项已设置为 true。
- **问题解决及配置详情参考**：`@seungduk` 在通过以传统的 pytorch.bin 格式而非 safetensors 格式保存解决问题后，分享了相关 [GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed_configs/zero3_bf16.json#L12) 配置文件链接。

**提到的链接**：

[axolotl/deepspeed_configs/zero3_bf16.json at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed_configs/zero3_bf16.json#L12)：欢迎提出 Axolotl 相关问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。

  

---



### OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1215029412794335262) (1 messages): 

- **Claude 3 让群聊变得轻而易举**：`@alexatallah` 分享了与 **Claude 3** 进行群聊的积极体验，该模型可以自我调节对话。他们附带了一个展示该功能的 [Twitter 故事](https://twitter.com/OpenRouterAI/status/1765470591836959061) 链接。
  

---

### OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1214520901350199359) (78 条消息🔥🔥): 

- **关于 Claude 版本的疑问**：`@quentmaker` 询问了 `anthropic/claude-2.0` 和 `anthropic/claude-2` 之间的区别，`@alexatallah` 和 `@wikipediadotnet` 澄清说 `Claude-2` 会自动选择最新的 `2.x` 版本。

- **多线程下的不确定成本**：`@mhmm0879` 表达了在使用 `gemma 7b` 和 `openchat 3.5` 进行多线程操作时，实际成本超过预测成本的担忧。`@alexatallah` 和 `@louisgv` 询问了具体的使用场景以及是否发送了图片，以尝试诊断该问题。

- **关于 Claude 和审查制度的讨论**：用户 `@followereternal`、`@ayumeri`、`@billbear` 和 `@scepty9097` 对 `Claude 3` 进行了混合讨论，一些人对潜在的过度审查表示不满，而另一些人则称赞该模型的对话能力。

- **OpenRouter 的 LangChain.js 问题**：`@mysticfall` 指出了在使用 LangChain.js 配合 `OpenRouter` 的 `ChatOpenAI` 模型进行文本补全（text completion）时遇到的困难。`@spaceemotion` 提到文本补全的端点可能被 OpenAI 标记为 "legacy"，`@mysticfall` 指出 OpenAI 库中硬编码的端点可能会导致潜在问题。

- **探索适用于 OpenRouter 的 VSCode 扩展**：`@_maximus01` 询问了是否有集成 OpenRouter 的 VSCode 代码辅助扩展，`@alexatallah` 建议赞助此类工作，`@spaceemotion` 和 `@_sam___` 分享了潜在的替代方案和一个活跃的 GitHub 项目。

**提到的链接**：

- [Home | Tabby](https://tabby.tabbyml.com/)：描述将放入 <head /> 中的 meta 标签。
- [Configuration | Continue](https://continue.dev/docs/model-setup/configuration#defining-a-custom-llm-provider)：配置你的 LLM 和模型提供商。
- [Perplexity: Sonar 8x7B by perplexity | OpenRouter](https://openrouter.ai/models/perplexity/sonar-medium-chat?tab=parameters)：Sonar 是 Perplexity 最新的模型系列。它在性价比、速度和性能方面超越了早期模型。具有互联网访问权限的版本是 [Sonar 8x7B Online](/mo...
- [Continue](https://continue.dev)：未找到描述。
- [GitHub - 0xk1h0/ChatGPT_DAN: ChatGPT DAN, Jailbreaks prompt](https://github.com/0xk1h0/ChatGPT_DAN)：ChatGPT DAN，越狱提示词。通过在 GitHub 上创建账户为 0xk1h0/ChatGPT_DAN 的开发做出贡献。
- [GitHub - continuedev/continue: ⏩ The easiest way to code with any LLM—Continue is an open-source autopilot for VS Code and JetBrains](https://github.com/continuedev/continue)：⏩ 使用任何 LLM 编写代码的最简单方式——Continue 是适用于 VS Code 和 JetBrains 的开源自动驾驶辅助工具 - continuedev/continue

---

### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1214500032422608906) (61 messages🔥🔥): 

- **LangChain 和函数实现协助**：`@vishal5795` 询问了如何使用 LangChain 和 OpenAI 的 `ChatCompletion.create()` 将函数角色（function roles）集成到消息中。`@chester3637` 提供了一个详细的 Python 示例，使用 LangChain 演示了如何将 AI 消息作为函数进行调用 ([LangChain Core 示例](https://python.langchain.com/docs/guides/function_calling))。
  
- **寻找技术任务合作伙伴**：`@mattew_999` 宣布他们正在寻找合作伙伴来处理技术任务，并强调这是一个付费机会。

- **关于新合作伙伴关系的咨询**：`@earduman2` 询问 LangChain 是否对新的 Chain 合作伙伴关系持开放态度，这引发了 `@baytaew` 的澄清请求。

- **FastAPI 偶发问题**：`@rajib2189` 报告了在高负载下使用 FastAPI 托管生成 API 时出现的偶发性 502 错误，特别是在 AWS ELB -> Apache Server -> Uvicorn 的架构设置中。

- **对 GPT-4 微调权限的兴趣**：`@8886600` 表达了希望获得 GPT-4 微调能力的愿望，并提到愿意为具有使用限制的 API key 付费。

**提到的链接**：

- [Chat LangChain](https://chat.langchain.com/)：未找到描述
- [Google Colaboratory](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/use_cases/question_answering/chat_history.ipynb)：未找到描述
- [LangChain Expression Language (LCEL) | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/)：LangChain 表达式语言（LCEL）是一种以声明式方式轻松组合 Chain 的方法。
- [Few-shot and function calling](https://community.openai.com/t/few-shot-and-function-calling/265908/15?u=vanpariyavishal02)：这里需要理解的是，函数调用为聊天提示词消息引入了一个新角色（“role”: “function”）。要在聊天模型提示词中使用 Few-shot 示例，你需要提供一系列...
- [Pydantic compatibility | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/pydantic_compatibility)：- Pydantic v2 已于 2023 年 6 月发布 (https://docs.pydantic.dev/2.0/blog/pydantic-v2-final/)
- [Google Colaboratory](https://colab.research.google.com/github/langchain-ai/langgraph/blob/main/examples/code_assistant/langgraph_code_assistant.ipynb#scrollTo=744f48a5-9ad3-4342-899f-7dd4266a9a15)：未找到描述
- [LangSmith](https://smith.langchain.com/public/ea1f6ca5-de52-4d36-bd7b-fde3faa74a70/d?paginationState=%7B%22pageIndex%22%3A0%2C%22pageSize%22%3A10%7D&chartedColumn=latency_p50)：未找到描述
- [Retrieval augmented generation (RAG) | 🦜️🔗 Langchain](https://js.langchain.com/docs/expression_language/cookbook/retrieval)：现在让我们看看在提示词和 LLM 中添加检索步骤，这构成了一个“检索增强生成（RAG）”链：
- [Azure AI Search | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/vectorstores/azuresearch#configure-vector-store-settings)：[Azure AI
- [langchain_agent/assistant at master · couthyapper7/langchain_agent](https://github.com/couthyapper7/langchain_agent/tree/master/assistant)：一个使用 LangChain 制作的 CSV 读取器，配备了经过微调的 GPT - couthyapper7/langchain_agent

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1214649940169592902) (1 messages): 

- **缓存功能尚不支持流式传输**：`@veryboldbagel` 提到目前缓存（caching）无法在流式模式（streaming mode）下工作。该问题与 **langchain-core** 有关，而非 **langserve**。
  

---

### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1214579961990479902) (6 messages): 

- **为 AI 艺术注入幽默**：`@neil6430` 尝试了来自 ML Blocks 的新 **control net block**，创作了一张有趣的鸡在表演单口喜剧并摆出 Seinfeld 姿势的图像。他们分享了对该功能的兴奋之情，并提供了 [ML Blocks](https://mlblocks.com/) 的链接，这是一个无需编码即可构建模块化、AI 驱动的图像处理工作流的工具。

- **Lutra 彻底改变工作流自动化**：`@polarbear007.` 介绍了 [Lutra.ai](https://lutra.ai/)，这是一个旨在*将英语指令转换为代码*的平台，通过编排各种应用程序来自动完成任务，将其比作功能更强大的 Zapier。

- **Raptor 揭秘 Long Context RAG 的奥秘**：`@andysingal` 分享了一篇关于使用 **RAPTOR 与 Langchain** 从头开始构建 Long Context Retrieval-Augmented Generation (RAG) 的 [Medium 文章](https://medium.com/ai-advances/building-long-context-rag-from-scratch-with-raptor-using-langchain-c6491f1ba141)。

- **ChromaDB 加入 LM Studio**：`@vic49.` 提供了一个 [GitHub 链接](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases)，指向 **LM Studio 的 ChromaDB 插件**，该插件支持为服务器模式操作创建 ChromaDB 向量数据库。

**提到的链接**：

- [ML Blocks | Home](https://mlblocks.com/)：ML Blocks 让你无需编写任何代码即可构建 AI 驱动的图像生成和分析工作流。
- [Lutra AI](https://lutra.ai/)：未找到描述
- [Releases · BBC-Esq/ChromaDB-Plugin-for-LM-Studio](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases)：为在服务器模式下运行的 LM Studio 创建 ChromaDB 向量数据库的插件！ - BBC-Esq/ChromaDB-Plugin-for-LM-Studio

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=QPZpOBxUd1U
  

---



### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1214688499886718996) (8 messages🔥): 

- **探索 RunPod 上的 Root 权限**：`@ericauld` 询问了在 RunPod 上以 root 身份运行的可能性，对此 `@nshepperd` 澄清说 **RunPod 提供的是 Docker 镜像而不是真实的 VM**，因此在这种情况下 root 并不是真正的 root。
- **探寻 H100 SRAM 带宽**：`@lucaslingle` 寻求有关 **NVIDIA H100** 的 SRAM 带宽信息，并指出在一次 GTC 演讲提到 **A100 为 19TB/s** 后缺乏近期资料。`@iron_bound` 通过引用一篇 [Chips and Cheese 文章](https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/)提供了帮助，该文章指出 **H100 的 L2 cache 具有 5.5 TB/s 的读取带宽**。
- **RTX 4090 基准测试**：针对 SRAM 带宽的讨论，`@zippika` 强调了 **RTX 4090 的 L1 带宽性能**，引用了另一篇 [Chips and Cheese 文章](https://chipsandcheese.com/2022/11/02/microbenchmarking-nvidias-rtx-4090/)，该文章重点介绍了 Nvidia 的 Ada Lovelace 架构和新的光线追踪改进。
- **H100 带宽假设**：`@zippika` 估计 **H100 带宽** 可能与 RTX 4090 相当，根据他们的发现提到 L1 带宽为 **40TB/s**，并假设 H100 可能会达到这一性能指标。

**提到的链接**：

- [Microbenchmarking Nvidia's RTX 4090](https://chipsandcheese.com/2022/11/02/microbenchmarking-nvidias-rtx-4090/)：Nvidia 的 RTX 4090 采用了 Nvidia 最新的架构，以早期计算先驱 Ada Lovelace 命名。与之前的 Ampere 架构相比，Ada Lovelace 拥有……
- [Nvidia's H100: Funny L2, and Tons of Bandwidth](https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/)：GPU 最初是纯粹用于图形渲染的设备，但其高度并行的特性使其对某些计算任务也具有吸引力。随着过去几年 GPU 计算领域的增长……

  

---

### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1214664795597570048) (9 messages🔥): 

- **GPU Tensor 分配失误**：`@zippika` 帮忙澄清了一个 Tensor 未能在 CUDA 设备上分配的原因，是因为 `@srns27` 忘记设置 `TensorOptions`，导致其默认为 CPU。
- **调试中的友好氛围**：`@zippika` 对 `@srns27` 给予了友善的回应，表示每个人都会犯错，并强调了 torch 社区的协作本质。
- **寻求数学运算的高级抽象**：`@mabeto5p` 询问是否有高级语言或软件包可以在 NVIDIA Ada 架构上执行低精度整数的线性代数和浮点运算。
- **利用 bitsandbytes 进行量化**：`@iron_bound` 建议 `@mabeto5p` 使用 [bitsandbytes package](https://github.com/TimDettmers/bitsandbytes) 在 PyTorch 中处理 k-bit 量化，以便在 GPU 上执行低精度线性代数运算。
- **量化速度的突破**：在 `@iron_bound` 指出 bitsandbytes 资源后，`@mabeto5p` 对发现 int8 对比 bf16 矩阵乘法可能实现 5700 倍加速感到兴奋。

**提到的链接**：

[GitHub - TimDettmers/bitsandbytes: Accessible large language models via k-bit quantization for PyTorch.](https://github.com/TimDettmers/bitsandbytes)：通过 PyTorch 的 k-bit 量化实现易用的 LLM。- TimDettmers/bitsandbytes

  

---


### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1214762847700254730) (3 messages): 

- **Mask 效率至关重要**：`@drisspg` 强调了计算中通用 masking 的低效性，因为它需要处理每一个 mask 元素，即使在不需要的时候也是如此。
- **滑动窗口 PR 增加细节**：`@drisspg` 更新了 PyTorch GitHub 上的 *sliding window attention bias* pull request，在描述中增加了更多细节。该 PR 可在[此处](https://github.com/pytorch/pytorch/pull/120143)审阅。
- **使用 Score-Mod API 优化 Bias**：`@drisspg` 讨论了添加 score-mod API 作为一种手段，以便在不完全实例化整个 bias 的情况下，高效地将 bias 约束融合到 *flash_attention* 算法中。

**提到的链接**：

[Add sliding window attention bias by drisspg · Pull Request #120143 · pytorch/pytorch](https://github.com/pytorch/pytorch/pull/120143)：摘要：此 PR 添加了一个新的 attention-bias torch_function，旨在与 SDPA 交互。这实现了滑动窗口并更新了 "aten.sdpa_flash" 以暴露 window_size_left 和 wind...

  

---


### CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/) (1 messages): 

bowtiedlark: 远程？
  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1214586868767662100) (2 messages): 

- **CUDA 初学者**：用户 `@umerha` 推荐将 **Jeremy 的视频** 作为学习 `numba.cuda.jit` 的起点。建议的资源可以在 [Lecture 3 和 5](https://github.com/cuda-mode/lectures) 中找到。
- **对学习资源的感谢**：用户 `@hoteret` 对 `@umerha` 分享的 CUDA 学习资源表示感谢。
  

---

### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1214488482198720562) (28 messages🔥): 

- **用于设备测试的脚本调整**：`@iron_bound` 讨论了在脚本中添加设备 ID，并计划测试单个设备以及其他 Ring 函数。
- **引入了带有故障的采样代码**：`@jamesmel` 分享了一个 [GitHub Pull Request](https://github.com/cuda-mode/ring-attention/pull/13)，这是对采样代码的首次尝试，同时也提到了正在调查的一些参数错误。
- **完成 Striped 和 Zigzag 的基准测试**：`@iron_bound` 报告称，在 runpod 机器上的测试显示 *striped* 和 *zigzag* 具有相同的内存上限，并展示了两个 CUDA 设备的具体内存使用情况。
- **开启 Axolotl 训练**：`@andreaskoepf` 分享了指向 [OpenAccess-AI-Collective 的 Axolotl GitHub 仓库](https://github.com/OpenAccess-AI-Collective/axolotl)的链接，`@iron_bound` 还提到成功进行了 Open Llama 3B 训练。
- **排除 Ring Attention 和采样逻辑的故障**：讨论围绕 `@iron_bound` 对自定义 Attention 库的调试，以及 `@jamesmel` 和 `@andreaskoepf` 为使采样代码正常运行所做的努力，并计划在即将举行的会议中讨论和澄清实现细节。

**提到的链接**：

- [BASED: Simple linear attention language models balance the recall-throughput tradeoff](https://www.together.ai/blog/based)：未找到描述
- [few more versions of sampling by melvinebenezer · Pull Request #13 · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/pull/13)：Ring Attention 中的 Logits 采样，包含 Greedy、top_k、top_p
- [GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions](https://github.com/OpenAccess-AI-Collective/axolotl)：欢迎提出 Axolotl 问题。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。

  

---



### LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1214588118968311848) (7 messages): 

- **Opus 在编程社区引起关注**：用户 `@pantsforbirds` 提到 **Opus** 在编程方面似乎非常有前景，引发了关于其能力的对话。
- **同行认可 Opus 在 function calling 中的表现**：`@res6969` 参与了对话，分享了他们听到的对 Opus 性能的高度评价，特别是在 **function calling** 方面，表明 Opus 可能是同类产品中最好的。
- **GPT-4 在医学知识方面表现出色**：在医学和生物学专业知识方面，`@thebaghdaddy` 发现 **GPT-4** **显著优于**其前代产品，其性能差异令人震惊。
- **基准测试受到质疑**：根据他们的经验，`@thebaghdaddy` 对**已发布的基准测试**的普遍可靠性表示怀疑，认为它们可能无法充分反映新模型的真实能力。
- **Opus 在 SAT 阅读中获得满分**：`@jeffreyw128` 分享了一个令人印象深刻的结果：Opus 在 SAT 阅读部分获得了 800 分，这在 [Twitter 帖子](https://twitter.com/wangzjeff/status/1764850689258451096)中得到了展示。这一分享引发了关于在模型规模如此之大的情况下，如何创建 holdouts 以避免记忆化（memorization）挑战的讨论。
  

---


### LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1214740064777674793) (2 messages): 

- **寻求关于 RAG 输出中引用的建议**：`@mat_mto` 询问了有关在 RAG 生成的文本中格式化引用和脚注的技巧资源（如博客或推文）。他们分享了一个带有指向网络搜索结果脚注的文本输出示例。
- **使用 JSON 对象进行清晰的来源归属**：作为回应，`@res6969` 提到他们使用 function calling 来输出包含文本和来源的 JSON 对象。这种方法可以清晰地将信息归属于其网络来源。
  

---

### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1214618500518125628) (8 messages🔥): 

- **澄清 AI 术语**：`@simonw` 强调了区分 **prompt injection** 和 **jailbreaking** 的重要性，解释说 prompt injection 涉及**将不可信的用户输入与开发者的可信提示词拼接**，而 jailbreaking 则试图绕过 **LLM** 自身的安全过滤器。他在[他的博文](https://simonwillison.net/2024/Mar/5/prompt-injection-jailbreaking/)中提供了详细的解释和历史背景。

- **网络威胁行为者手中的 AI**：`@tariqali` 分享了来自 [Microsoft 博客文章](https://www.microsoft.com/en-us/security/blog/2024/02/14/staying-ahead-of-threat-actors-in-the-age-of-ai/)的见解，内容关于受国家支持的攻击者利用 OpenAI 的 LLM 进行网络活动（包括侦察和 **spear phishing**），并提到一个攻击者因带有恶意意图向模型发送提示词而被拦截的案例。

- **LLM 的双重用途困境**：针对与 AI 相关的风险，`@tariqali` 引用了一篇关于 OpenAI 尝试为 LLM 辅助的生物威胁建立早期预警系统的研究文章，重点介绍了仅使用互联网与结合使用 **GPT-4** 解决任务的对比研究，详情见[此处](https://openai.com/research/building-an-early-warning-system-for-llm-aided-biological-threat-creation)。

- **作为缓解策略的访问控制**：`@tariqali` 建议可以通过控制谁能访问 LLM 来缓解 prompt injection，并提议将人工审核内容作为一种潜在的防御层，在输入到达 AI 之前进行清理。

- **不可见 Prompt Injection 的挑战**：`@simonw` 指出了人工审核在防止 prompt injection 方面的局限性，并以隐藏在图像中近乎白色的文本中的不可见 prompt injection 为例，正如他在[他的博文](https://simonwillison.net/2023/Oct/14/multi-modal-prompt-injection/#prompt-injection-hidden-in-images)中所讨论的，即使在像 GPT-4-V 这样的多模态版本中，这仍然是一个威胁。

**提及的链接**：

- [Prompt injection and jailbreaking are not the same thing](https://simonwillison.net/2024/Mar/5/prompt-injection-jailbreaking/)：我一直看到人们在谈论 “jailbreaking” 时使用 “prompt injection” 这个术语。这种错误现在非常普遍，以至于我不确定是否还能纠正过来：…
- [Building an early warning system for LLM-aided biological threat creation](https://openai.com/research/building-an-early-warning-system-for-llm-aided-biological-threat-creation)：我们正在开发一个评估蓝图，用于评估大型语言模型 (LLM) 可能辅助他人制造生物威胁的风险。在涉及生物专家和学生的评估中，...
- [Multi-modal prompt injection image attacks against GPT-4V](https://simonwillison.net/2023/Oct/14/multi-modal-prompt-injection/#prompt-injection-hidden-in-images)：GPT4-V 是 GPT-4 的新模式，允许你在对话中上传图像。它非常出色，但也提供了一整套全新的攻击向量 …
- [Staying ahead of threat actors in the age of AI | Microsoft Security Blog](https://www.microsoft.com/en-us/security/blog/2024/02/14/staying-ahead-of-threat-actors-in-the-age-of-ai/)：Microsoft 与 OpenAI 合作，正在发布关于 AI 时代新兴威胁的研究，重点关注与已知威胁行为者 Forest Blizzard、Emerald Sleet 相关的活动...

  

---


### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1214618501604708362) (1 messages): 

- **寻求模型文件位置的共识**：`@florents_` 询问是否存在共识或特定的代码片段来规定各种工具搜索模型文件的位置，并建议了可能的路径，如 `$(pwd)/.models` 或 `$HOME/models`。目前没有进一步的讨论或回复。
  

---

### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1214529229870333982) (9 messages🔥): 

- **探索聊天机器人环境**：`@crispstrobe` 提到 chat.lmsys.org 允许进行测试，但需注意输入内容会被包含在后续的训练数据中；并重点推荐了 poe.com，该平台托管了三个模型，包括一个 perplexity 功能。
- **寻找最佳德语模型**：`@le_mess` 询问目前最好的德语模型；`@johannhartmann` 根据具体约束条件推荐了 Claude Opus、gpt-4、discolm-120b 或 VAGOsolutions/Sauerkraut LM-UNA-SOLAR-Instruct。
- **最新资讯**：`@maxidl` 分享了一篇 [arxiv 论文](https://arxiv.org/abs/2403.03187)，该研究表明检索增强语言模型（retrieval-augmented language models）可能成为参数化 LMs 的更优替代方案，尽管该领域的研究尚不广泛。
- **对 Hermes 和 Mixtral 的高度赞誉**：`@cybertimon` 建议在德语任务中使用 Nous Hermes 2 Mixtral 8x7b，并指出其在该语言方面的精通程度。
- **在 7B 参数模型中寻找完美之选**：`@johannhartmann` 和 `@flozi00` 回应了关于高质量德语模型的查询，Johannhartmann 建议使用 DiscoResearch/DiscoLM_German_7b_v1 及类似模型，而 flozi00 则因其准确性而推崇 Nous Hermes 2 Mixtral 8x7b。

**提到的链接**：

[Reliable, Adaptable, and Attributable Language Models with Retrieval](https://arxiv.org/abs/2403.03187)：在海量网络数据上训练的参数化语言模型（LMs）展现出了卓越的灵活性和能力。然而，它们仍面临幻觉（hallucinations）等实际挑战……

  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1214989627455438921) (1 messages): 

- **热烈欢迎新人**：用户 `@segmentationfault.` 对受 `@748528982034612226` 邀请表示感谢，并表现出尽管是新人但仍渴望为该领域做出贡献的热情。未提供关于贡献或感兴趣领域的进一步信息。
  

---


### Alignment Lab AI ▷ #[oo2](https://discord.com/channels/1087862276448595968/1176548760814375022/1214884876793282561) (3 messages): 

- **亲切的问候**：`@thenetrunna` 以一句友好的 "henlo frens" 开始了对话，为 oo2 频道奠定了轻松的基调。
- **晚间的欢迎回复**：`@jaxxks` 在傍晚做出了回应，感谢 `@thenetrunna` 的欢迎。
- **向群组打招呼**：`@tcapelle` 加入了对话，愉快地打招呼 "Hello every1!"，引发了一连串参与者之间的自我介绍和问候。
  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1214565709028397086) (2 messages): 

- **介绍 Claude 3，超越 GPT-4 的 LLM**：`@pradeep1148` 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=Zt73ka2Y8a8)，标题为 *"Introducing Claude 3 LLM which surpasses GPT-4"*。视频讨论了 Claude 3 模型家族，据报道该家族在各种认知任务中树立了新的基准。

- **如何使用 Mistral 进行开发**：`@pradeep1148` 分享了另一个 [YouTube 链接](https://www.youtube.com/watch?v=QPZpOBxUd1U)，标题为 *"Infinite Craft Game using Mistral"*。内容涉及使用 Mistral 模型开发 Neal Agarwal 的网页游戏 Infinite Craft。

**提到的链接**：

- [Infinite Craft Game using Mistral](https://www.youtube.com/watch?v=QPZpOBxUd1U)：让我们来开发 Neal Agarwal 的网页游戏 Infinite Craft。这是一款“合成游戏”，你从仅有的四个元素开始，通过不断组合成对的元素……
- [Introducing Claude 3 LLM which surpasses GPT-4](https://www.youtube.com/watch?v=Zt73ka2Y8a8)：今天，我们来看看 Claude 3 模型家族，它在广泛的认知任务中树立了新的行业基准。该家族包括三个最先进的……

  

---



### Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1214665643362619402) (1 messages): 

- **Intel 面临现实考验**：`@natolambert` 分享了一个 [YouTube 视频](https://youtu.be/YW1Rr5N84cI?si=CgrmGcSLQznTshZ3)，标题为 "Intel's Humbling"，由 **Stratechery** 的 Ben Thompson 配音，认为该视频提供了宝贵的见解，且不会让他觉得自己像个“彻头彻尾的白痴”。视频探讨了 Intel 面临的挑战，并附带了相关文章链接以供深入阅读。

**提到的链接**：

[Intel&#39;s Humbling | Stratechery by Ben Thompson](https://youtu.be/YW1Rr5N84cI?si=CgrmGcSLQznTshZ3)：阅读文章：https://stratechery.com/2024/intels-humbling/ 链接：Stratechery: https://stratechery.com 注册 Stratechery Plus: https://stratechery.c...

### Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1214764700932251708) (1 条消息): 

- **反思 AI 的模糊性**：`@natolambert` 推荐了 Elad Gil 的一篇发人深省的文章，强调了 **Generative AI** 往往随着时间的推移变得更加令人费解。这篇 [文章](https://blog.eladgil.com/p/things-i-dont-know-about-ai) 针对 AI 技术栈的各个层面提出了开放性问题，旨在引发讨论并提供见解。

**提到的链接**：

[Things I Don't Know About AI](https://blog.eladgil.com/p/things-i-dont-know-about-ai)：我对 AI 市场的了解越多，就越觉得自己知道的越少。我列出了一些问题和想法。

  

---