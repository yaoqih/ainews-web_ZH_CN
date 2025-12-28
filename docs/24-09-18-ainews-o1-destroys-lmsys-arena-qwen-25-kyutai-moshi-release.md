---
companies:
- openai
- anthropic
- google
- alibaba
- deepseek
- kyutai
- weights-biases
- mistral-ai
date: '2024-09-18T21:51:26.650574Z'
description: '**OpenAI 的 o1-preview 模型**达成了一个里程碑，在无需人工干预的情况下，能够完全匹配每日顶级 AI 新闻报道，并在“氛围感测试”（vibe
  check）评估中持续优于 **Anthropic**、**Google** 和 **Llama 3** 等其他模型。


  **OpenAI** 的模型在 **LMsys** 基准测试中占据了前四名，其速率限制也已提高至**每分钟 500-1000 次请求**。在开源领域，**阿里巴巴的
  Qwen 2.5** 系列在 70B 规模上超越了 **Llama 3.1**，并更新了其闭源的 **Qwen-Plus** 模型，使其表现优于 **DeepSeek
  V2.5**，但仍落后于美国领先模型。


  **Kyutai Moshi** 发布了其开源权重的实时语音模型，该模型采用独特的、带有“内心独白”的流式神经架构。**Weights & Biases** 推出了
  **Weave**，这是一款大语言模型（LLM）可观测性工具包，旨在增强实验跟踪和评估，使提示词工程（prompting）过程更加科学化。


  新闻还提到了即将举行的活动，例如在旧金山举办的 **WandB “LLM-as-judge” 黑客松**。*“o1-preview 在我们的氛围感测试评估中始终胜出”*，以及
  *“OpenAI 模型的速率限制正在逐日提高。”*'
id: 0f4a2a7d-2a07-47aa-8fb5-2f8377b999ca
models:
- o1-preview
- o1-mini
- qwen-2.5
- qwen-plus
- llama-3-1
- deepseek-v2.5
original_slug: ainews-o1-destroys-lmsys-arena-qwen-25-kyutai
people:
- sama
- guillaumelample
title: o1 横扫 Lmsys Arena 榜单，Qwen 2.5 与 Kyutai Moshi 正式发布。
topics:
- chain-of-thought
- multimodality
- model-benchmarking
- model-performance
- streaming-neural-architecture
- llm-observability
- experiment-tracking
- rate-limiting
---

<!-- buttondown-editor-mode: plaintext -->**o1 可能就是你所需的一切。**

> 2024年9月17日至9月18日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**221** 个频道，**1591** 条消息）。预计节省阅读时间（以每分钟 200 字计）：**176 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们 Smol AI 的人类一直对这一天心存畏惧。

**有史以来第一次，一个 LLM 能够在无需我们干预的情况下，100% 匹配并准确报告我们认为的当日头条新闻。**（参见下文的 **AI Discord Recap**。）

<p align="center">
<img src="https://assets.buttondown.email/images/ce5e427f-f148-47a3-93d6-cb84e9d6d735.png?w=960&fit=max" height=200 align="center" />
</p>

对于模型训练者来说，或许更有趣的是 `o1-preview` 在我们的 vibe check 评估中始终胜出。每一期 AINews 的运行都是 OpenAI、Anthropic 和 Google 模型之间的对决（你可以在 [存档](https://buttondown.com/ainews/archive/) 中看到痕迹。我们也 [短暂尝试过 Llama 3](https://x.com/swyx/status/1828223943301570939) 但它总是落败），而 `o1-preview` 自发布以来基本上每天都获胜（除了需要 [移除 instructor 隐藏的系统提示词](https://x.com/ivanleomk/status/1834749163151802505) 之外，没有进行任何特定的调优）。

我们现在有了 [关于 o1-preview 和 -mini 的 LMsys 数据](https://x.com/lmsysorg/status/1836443278033719631) 来量化这些 vibe check。


![image.png](https://assets.buttondown.email/images/55370726-21c8-4a60-bf83-d6ade0502279.png?w=960&fit=max)


LMsys 的前 4 名现在都被 OpenAI 模型占据。尽管 OpenAI 正在 [逐日提高速率限制（rate limits），目前已达到每分钟 500-1000 次请求](https://x.com/OpenAIDevs/status/1836506351062716701)，但需求依然旺盛。

在开源领域，**阿里巴巴的 Qwen 凭借其 [Qwen 2.5 系列通用、编程和数学模型](https://qwenlm.github.io/blog/qwen2.5/) 赶超了 DeepSeek**，在 70B 规模上展现出优于 Llama 3.1 的数据。


![image.png](https://assets.buttondown.email/images/51eb7865-0dfd-4390-b21a-a99fec253057.png?w=960&fit=max)


同时他们也更新了其闭源的 Qwen-Plus 模型以击败 DeepSeek V2.5，但仍逊于美国的 frontier models。

最后，**Kyutai Moshi** 在 [7 月份预告了其实时语音模型](https://the-decoder.com/french-ai-lab-kyutai-unveils-conversational-ai-assistant-moshi-plans-open-source-release/) 并在公开演示中出现了一些 [有趣/令人担忧的精神崩溃](https://x.com/benhylak/status/1808611023123067357?s=46&t=Fski5tAXGapEPufiBpUQQg) 后，终于按约定发布了其 open weights 模型，以及展示“内心独白”的独特流式神经架构的细节。


![image.png](https://assets.buttondown.email/images/92c17587-f3f0-4fda-916a-37a5bcd21387.png?w=960&fit=max)


实时演示地址仍为 [https://moshi.chat](https://moshi.chat/)，或者在本地尝试：

```bash
$ pip install moshi_mlx
$ python -m moshi_mlx.local_web -q 4
```

---

**[本期内容由 Weights and Biases Weave 赞助！]**: 
坦白说，许多团队只知道 Weights & Biases 是 **世界上最好的 ML 实验跟踪软件**，甚至不知道我们名为 Weave 的新 LLM 可观测性工具包。所以，如果你正在阅读这篇文章，并且正在生产环境中进行任何 LLM 调用，为什么不 [试试 Weave](http://wandb.me/swyx-weave) 呢？只需 3 行代码，你就可以记录并追踪用户与 LLM 之间的所有输入、输出和元数据，通过我们的评估框架，你可以将 prompting 从一门艺术转变为一门科学。

查看关于 [使用 Weave 构建 GenAI 辅助自动故事插画师](http://wandb.me/swyx-report) 的报告。

> swyx 的评论：这周末我将参加在旧金山举行的 [WandB LLM-as-judge 黑客松](http://wandb.me/swyx-hack)，届时会有许多来自 Latent Space/AI Engineer 团队的朋友一起使用 Weave 进行开发！

[
![image.png](https://assets.buttondown.email/images/a3a4a9bc-af9d-4652-84ef-886ac0861609.png?w=960&fit=max)
](http://wandb.me/swyx-hack)

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型更新与发布**

- **OpenAI 的 o1 模型**：[@sama](https://twitter.com/sama/status/1836178378673786923) 宣布在目标 3 上表现显著超出预期，尽管耗时比预想的长。这些模型使用思维链（chain-of-thought）推理来增强复杂问题的解决能力。

- **Mistral AI 的 Pixtral**：[@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1836147938898382883) 宣布发布 Pixtral 12B，这是一款可在 le Chat 和 la Plateforme 上使用的多模态模型。它包含一个新的 400M 参数视觉编码器（vision encoder）和一个基于 Mistral Nemo 的 12B 参数多模态解码器（multimodal decoder）。

- **Llama 3.1**：[@AIatMeta](https://twitter.com/AIatMeta/status/1836095729535983791) 分享了 Llama 增长的最新动态，指出主要云合作伙伴和各行业的采用率正在迅速增加。

**AI 开发与工具**

- **ZML**：[@ylecun](https://twitter.com/ylecun/status/1836030233796874244) 重点介绍了 ZML，这是一个高性能 AI 推理栈（inference stack），用于在各种硬件上并行化和运行深度学习系统，目前已结束隐身模式并开源。

- **LlamaCloud**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1836168846388138316) 宣布了 LlamaCloud 的多模态功能，支持对具有空间布局、嵌套表格和视觉元素的复杂文档进行 RAG。

- **Cursor**：[@svpino](https://twitter.com/svpino/status/1836015426376998956) 赞扬了 Cursor 的代码补全能力，指出其功能相较于其他工具更为先进。

**AI 研究与基准测试**

- **思维链赋能（Chain of Thought Empowerment）**：一篇[论文](https://twitter.com/rohanpaul_ai/status/1836149683426615490)展示了 CoT 如何使 Transformer 能够解决固有的串行问题，将其问题解决能力扩展到仅限并行的限制之外。

- **V-STaR**：关于为自我启发式推理者（self-taught reasoners）训练验证器（verifiers）的[研究](https://twitter.com/_philschmid/status/1835936839057740043)，显示在代码生成和数学推理基准测试中提升了 4% 到 17%。

- **Masked Mixers**：一项[研究](https://twitter.com/rohanpaul_ai/status/1836164187653050560)表明，带有卷积的 Masked Mixers 在某些语言建模任务中可能优于 self-attention。

**AI 教育与资源**

- **新 LLM 书籍**：[@JayAlammar](https://twitter.com/JayAlammar/status/1836064233957515554) 和 [@MaartenGr](https://twitter.com/MaartenGr) 发布了一本关于大语言模型（Large Language Models）的新书，已在 O'Reilly 上架。

- **DAIR.AI Academy**：[@omarsar0](https://twitter.com/omarsar0/status/1836140676276199677) 宣布启动 DAIR.AI Academy，提供提示工程（prompt engineering）和 AI 应用开发课程。

**AI 应用与演示**

- **AI 产品广告**：[@mickeyxfriedman](https://twitter.com/mickeyxfriedman/status/1836080022941282554) 介绍了 Flair AI 上的 AI 生成产品广告，允许用户根据产品照片创建动画视频。

- **多模态 RAG**：[@llama_index](https://twitter.com/llama_index/status/1836079020351648173) 推出了多模态功能，用于构建跨非结构化数据的端到端多模态 RAG 流水线（pipelines）。

- **NotebookLM**：[@omarsar0](https://twitter.com/omarsar0/status/1836187497329467887) 演示了 NotebookLM 从 AI 论文生成逼真播客的能力，展示了 AI 和 LLM 的一个有趣应用。


---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. T-MAC：适用于 llama.cpp 的高效能 CPU 后端**



- **T-MAC（一种高效能 CPU 后端）可能即将加入 llama.cpp！** ([Score: 50, Comments: 5](https://reddit.com//r/LocalLLaMA/comments/1fj15h3/tmac_an_energy_efficient_cpu_backend_may_be/))：**T-MAC** 和 **BitBLAS** 是微软支持的项目，旨在实现高效的低比特数学运算，随着 T-MAC 维护者计划提交 pull request，它们可能会被集成到 **llama.cpp** 中。T-MAC 显示了 FLOPs 和推理延迟相对于比特数的**线性缩放**，支持对 int1/2/3/4 进行**位运算**而无需反量化（dequantization），并通过快速查表和加法指令支持各种激活类型。这种集成可能会使 **Ollama** 等项目受益，有望提升笔记本电脑和 **Pixel 6** 等移动设备的性能，后者目前在运行 llama.cpp 时面临热节流（thermal throttling）问题。
  - 讨论中提到 **BitNet** 并非真正的量化方法，因为它是原生以 1 bit 训练的，而不是从高分辨率模型量化而来的。原帖作者澄清说，某些层仍然需要量化。
  - 用户对 **BitNet** 的潜力表示兴奋，一位评论者热切期待其完整实现及对该领域的影响。
  - “**终极量化**”的概念被幽默地提及，原帖作者开玩笑地大喊其所谓的益处，如“**无损质量**”和“**OpenAI 陷入混乱**”。

**主题 2. Qwen2.5-72B-Instruct：性能与内容过滤**



- **Qwen2.5-72B-Instruct 在 LMSys Chatbot Arena** ([评分: 31, 评论: 10](https://reddit.com//r/LocalLLaMA/comments/1fj39h2/qwen2572binstruct_on_lmsys_chatbot_arena/)): **Qwen2.5-72B-Instruct** 在 **LMSys Chatbot Arena** 上表现强劲，正如分享的图片所证明的那样。**Qwen2.5** 系列包含从 **0.5B 到 72B** 参数的模型，并设有针对编码和数学任务的专门版本。与前代产品相比，该系列似乎具有更严格的内容过滤，导致模型对某些概念缺乏了解，包括一些非色情但可能与性相关的话题。
  - **Qwen2.5-72B-Instruct** 面临严格的**内容过滤**，这可能归因于中国对开源 LLM 的监管。用户注意到它对某些概念不了解，包括非色情的性内容以及像**天安门广场**这样的敏感政治话题。
  - 该模型在**编码和数学任务**中表现出色，性能与 **405B** 和 **GPT-4** 相当。一些用户发现，在 Prompt 中加入“绝不犯错 (never make any mistake)”可以提高对棘手问题的回答质量。
  - 尽管存在对审查的担忧，一些用户仍赞赏该模型对技术知识的专注。讨论中提到了绕过内容限制的尝试，一位用户分享了一张[绕过方法的图片](https://preview.redd.it/l2vwefrimepd1.png?width=1589&format=png&auto=webp&s=737ef9aa6133af2b1aac5b9b48ff8cb96a53360c)。


**主题 3. 视觉语言模型 (VLMs) 的最新进展**



- **[最新 VLMs 及 VLM 基准测试综述](https://nanonets.com/blog/bridging-images-and-text-a-survey-of-vlms/)** ([评分: 30, 评论: 8](https://reddit.com//r/LocalLLaMA/comments/1fjls95/a_survey_of_latest_vlms_and_vlm_benchmarks/)): 该帖子对近期**视觉语言模型 (VLMs)** 及其相关基准测试进行了全面综述。它重点介绍了 **GPT-4V**、**DALL-E 3**、**Flamingo**、**PaLI** 和 **Kosmos-2** 等关键模型，讨论了它们的架构、训练方法以及在各种任务中的表现。该综述还涵盖了重要的 VLM 基准测试，包括 **MME**、**MM-Vet** 和 **SEED-Bench**，这些基准测试在广泛的视觉理解和生成能力方面对模型进行评估。
  - 用户询问了关于**本地可运行的 VLMs**，作者推荐了 **Bunny** 并参考了 State of the Art 章节作为依据。
  - 出现了一场关于为非营利用途创建**移动优先应用**的讨论，建议使用 **YOLO** 进行训练，并使用 **UI 叠加层**进行实时目标检测，参考了一个 [YouTube 视频](https://www.youtube.com/watch?v=QV85eYOb7gk) 获取 UI 灵感。
  - 提出了一项针对**漫画翻译**的新 **VLM 基准测试**提案，强调需要评估模型识别文本、理解多图上下文以及在视觉和文本模态中消除歧义的能力。


**主题 4. Mistral Small v24.09：新型 22B 企业级模型**



- **为什么思维链 (chain of thought) 是以文本形式实现的？** ([评分: 67, 评论: 51](https://reddit.com//r/LocalLLaMA/comments/1fixn2m/why_is_chain_of_thought_implemented_in_text/)): 该帖子质疑了在语言模型中以文本格式实现**思维链推理**的效率，特别是提到了针对长推理链进行微调的 **o1**。作者认为，在**高维向量**中保持模型的逻辑可能比将推理投影到文本 Token 中更有效，这挑战了当前甚至在专门为扩展推理设计的模型中所使用的方法。
  - 正如用户所指出的，**可追溯性**和**可解释 AI (explainable AI)** 是基于文本的思维链推理的显著优势。潜空间 (latent space) 的**黑盒**性质会使人类更难理解模型的推理过程。
  - **OpenAI 的博客文章**透露，**o1 模型**的思维链过程是基于文本的，这与关于向量化层的猜测相反。一些用户建议，未来的模型如 **o2** 可能会实现**隐式 CoT (implicit CoT)** 以节省 Token，并参考了一篇关于[数学推理的论文](https://arxiv.org/abs/2405.14838)。
  - 用户讨论了训练抽象潜空间进行推理的挑战，一些人建议将**强化学习 (reinforcement learning)** 作为一种潜在方法。其他人提出了诸如逐渐转变训练数据或使用**特殊 Token** 来控制推理步骤在推理 (inference) 过程中的显示等想法。

- **[mistralai/Mistral-Small-Instruct-2409 · NEW 22B FROM MISTRAL](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409)** ([Score: 160, Comments: 74](https://reddit.com//r/LocalLLaMA/comments/1fj4unz/mistralaimistralsmallinstruct2409_new_22b_from/)): Mistral AI 发布了一个名为 **Mistral-Small-Instruct-2409** 的新型 **22B parameter** 模型，现已在 Hugging Face 上线。该模型展示了优于其前代产品的能力，包括在各个领域增强了 **instruction-following**、**multi-turn conversations** 和 **task completion**。此次发布标志着 Mistral AI 模型产品的重大进步，在性能和通用性方面有可能与更大的语言模型竞争。
  - **Mistral Small v24.09** 是在 **MRL license** 下发布的，允许非商业性质的自我部署。用户反应不一，一些人对其 **finetuning** 潜力感到兴奋，而另一些人则对许可限制感到失望。
  - 该模型在 **human alignment**、**reasoning** 和 **code generation** 方面展示了改进的能力。它支持 **function calling**，具有 **128k sequence length** 和 **32768** 的词汇量，使其在某些用例中成为 **GPT-3.5** 的潜在替代方案。
  - 用户讨论了该模型在当前语言模型格局中的地位，指出其 **22B parameters** 填补了较小模型与 **Llama 3.1 70B** 等较大模型之间的空白。一些人推测了它与 20-35B 参数范围内的其他模型相比的性能。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型进展与研究**

- **OpenAI 的 o1 模型展示了令人印象深刻的能力**：OpenAI 发布了其 o1 模型的预览版，该模型较之前的模型有显著改进。Sam Altman [在推文中提到“在目标 3 上表现出色”](https://www.reddit.com/r/singularity/comments/1fje98h/sam_altman_incredible_outperformance_on_goal_3/)，暗示了重大进展。

- **增加推理算力（inference compute）带来重大性能提升**：OpenAI 研究员 Noam Brown 建议，[增加推理算力比增加训练算力更具成本效益](https://www.reddit.com/r/singularity/comments/1fji7yv/openais_noam_brown_suggests_increasing_inference/)，潜力可能达到几个数量级。这可以通过在推理时分配更多算力来显著提高性能。

- **Google DeepMind 推进多模态学习**：一篇 [Google DeepMind 的论文](https://arxiv.org/html/2406.17711v1) 展示了如何通过联合样本选择进行数据策展，从而进一步加速多模态学习。

- **Microsoft 的 MInference 加速长上下文推理**：[Microsoft 的 MInference 技术](https://arxiv.org/abs/2407.02490) 能够实现长上下文任务中高达数百万个 token 的推理，同时保持准确性，大幅提升了所支持模型的速度。

**AI 应用与演示**

- **AI 辅助快速应用开发**：一位开发者利用 Claude 和 OpenAI 的 o1 模型，在不手动编写任何代码的情况下，[在短短 6 小时内构建并发布了一款 iOS 习惯追踪应用](https://www.reddit.com/r/OpenAI/comments/1fjg9sh/its_great_time_to_be_alive_but_also_really_scary/)。这展示了 AI 大幅加速软件开发的潜力。

- **虚拟试穿技术**：Kling 推出了 [Kolors Virtual-Try-On](https://www.reddit.com/r/singularity/comments/1fjfr61/kling_has_launched_kolors_virtualtryon_you_can/)，允许用户只需点击几下即可免费更换任何照片上的衣服。这展示了 AI 驱动的图像处理技术的进步。

- **AI 生成艺术与设计**：r/StableDiffusion 中的帖子展示了令人印象深刻的 [AI 生成艺术作品](https://www.reddit.com/r/StableDiffusion/comments/1fj9783/created_lora_tech_vibrant_3d_style_render_glass/) 和 [设计](https://www.reddit.com/r/StableDiffusion/comments/1fj671u/sakura_tree/)，证明了 AI 模型的创造潜力。

**行业与基础设施发展**

- **AI 基础设施的巨额投资**：Microsoft 和 BlackRock 正在[组建一个筹集 1000 亿美元的团体](https://www.reddit.com/r/singularity/comments/1fjdd0y/microsoft_blackrock_form_group_to_raise_100/)，用于投资 AI 数据中心和电力基础设施，这预示着 AI 算力资源的重大扩张。

- **Neuralink 推进脑机接口技术**：Neuralink 的 Blindsight [获得了 FDA 的突破性设备认定](https://www.reddit.com/r/singularity/comments/1fj7p15/neuralink_received_breakthrough_device/)，旨在为失明人士恢复视力。

- **NVIDIA 对自主机器的愿景**：NVIDIA 的 Jim Fan 预测，[在 10 年内，每一台移动的机器都将是自主的](https://www.reddit.com/r/singularity/comments/1fivpc1/nvidias_jim_fan_says_in_10_years_every_machine/)，智能机器人的数量将与 iPhone 一样多。不过，这一时间表是作为一种假设情景提出的。

**哲学与社会影响**

- **AI 作为新“物种”的出现**：一位前 OpenAI 研究员[认为我们正处于两个智能水平相当的物种并存的节点](https://www.reddit.com/r/singularity/comments/1fizylf/exopenai_researcher_i_didnt_expect_there_to_be/)，这里指的是人类和 AI。这引发了关于 AI 智能本质及其快速进展的讨论。

- **与历史技术转型的对比**：一篇帖子将[当前的 AI 数据中心与计算机的电子管时代进行类比](https://www.reddit.com/r/singularity/comments/1fix55b/we_are_back_in_the_vacuum_tube_era/)，暗示我们可能正处于另一次重大技术飞跃的边缘。


---

# AI Discord 摘要回顾

> 由 O1-preview 生成的摘要之摘要的摘要

**主题 1. 新 AI 模型登场**

- [**Qwen 2.5 盛大发布，包含 100 多个模型**](https://qwenlm.github.io/blog/qwen2.5/)：阿里巴巴发布 **Qwen 2.5**，拥有超过 **100 个模型变体**，包括 **Qwen2.5-Coder** 和 **Qwen2.5-Math**，参数量从 **0.5B** 到 **72B** 不等。这一开源强力模型挑战了私有模型，其旗舰型号 **Qwen2.5-72B-Instruct** 在顶级基准测试中表现出色。

- [**Moshi 亮相：Kyutai Labs 推出对话式 AI**](https://kyutai.org/Moshi.pdf)：**Kyutai Labs** 发布了实验性低延迟 AI 模型 **Moshi**，并同步发布了[技术报告](https://kyutai.org/Moshi.pdf)、[模型权重](https://huggingface.co/collections/kyutai/moshi-v01-release-66eaeaf3302bef6bd9ad7acd)以及在其 [GitHub](https://github.com/kyutai-labs/moshi) 上提供了基于 Pytorch、Rust 和 MLX 的流式推理代码。

- [**OpenAI 的 o1 模型称霸竞技场**](https://x.com/lmsysorg/status/1836443278033719631?s=46)：**OpenAI** 的 **o1-preview** 和 **o1-mini** 在 **Chatbot Arena** 中夺得榜首，在数学、高难度提示词和编程任务中表现优异。用户称赞 **o1-mini** 在生物医学领域堪比“优秀的博士生”。


**主题 2. 涡轮增压模型微调**

- [**Unsloth 提速两倍并减少 70% 的 VRAM 占用**](https://unsloth.ai/blog/llama3-1)：**Unsloth** 将 **Llama 3.1**、**Mistral** 和 **Gemma** 等模型的微调速度提升了 **2 倍**，同时将 VRAM 使用量降低了 **70%**。讨论还强调了将量化模型推送到 Hub 时对存储的影响。

- [**Torchtune 0.3 发布，支持 FSDP2 和 DoRA**](https://github.com/pytorch/torchtune/releases/tag/v0.3.0)：最新的 **Torchtune** 版本引入了完整的 **FSDP2** 支持，增强了分布式训练的灵活性和速度。它还增加了通过设置 `use_dora=True` 轻松激活 **DoRA/QDoRA** 特性的功能。

- **课程学习（Curriculum Learning）在 PyTorch 中得到实际应用**：成员们分享了在 PyTorch 中实现**课程学习**的步骤，涉及自定义数据集类和分阶段难度设置。一个示例展示了如何在训练循环中更新数据集以实现渐进式学习。


**主题 3. 应对 AI 模型的小故障**

- **OpenRouter 用户遭遇 429 错误轰炸**：沮丧的 **OpenRouter** 用户报告称遭遇了 **429 错误**和严格的频率限制，一名用户甚至被限流 **35 小时**。关于备用模型和密钥管理以缓解访问问题的讨论异常激烈。

- **缓存混淆引发困扰**：开发者们正在努力解决模型中的缓存管理问题，讨论了在每个任务后完全删除缓存的必要性。建议包括使用上下文管理器来防止评估过程中的干扰。

- **过度安全特性遭到嘲讽**：社区幽默地批评了像 **Phi-3.5** 这样过度审查的模型，并分享了一些讽刺性的回复。他们强调了过度审查给编程和技术任务带来的挑战。


**主题 4. AI 席卷创意领域**

- [**Riffusion 在 AI 音乐领域掀起波澜**](https://www.riffusion.com/)：**Riffusion** 允许用户通过频谱图生成音乐，引发了关于集成 AI 生成歌词的讨论。成员们注意到，在全曲生成方面，目前还缺乏能替代 **Suno AI** 的开源方案。

- **成人角色扮演（ERP）获得 AI 升级**：分享了使用 AI 模型进行**成人角色扮演 (ERP)** 的高级技术，重点在于构建详细的角色档案和沉浸式提示词。用户强调了营造期待感和真实互动的重要性。

- **艺术家寻找“图转卡通”模型**：成员们正在寻找能够将**图像转换为高质量卡通**的 AI 模型，并互相交流推荐。对于能够提供顶级卡通转换效果模型的探索仍在继续。


**主题 5. AI 集成提升生产力**

- **Perplexity Pro 集成至 VSCode 遭遇阻碍**：尝试将 **Perplexity Pro** 与 VSCode 扩展（如 'Continue'）配合使用的用户面临挑战，特别是难以区分 Pro Search 和纯写作模式。有限的编程技能也增加了集成的难度。

- **自定义 GPT 成为个人代码片段库**：成员们正利用 **Custom GPTs** 来记忆个人代码片段和模板，例如 [Mac Snippets](https://link.to.mac.snippets)。建议不要过度堆砌指令，以保持性能。

- **LM Studio 的新功能令用户兴奋**：**LM Studio** 中新增的**文档处理**功能引发了用户热议。讨论围绕数据表大小限制以及通过该软件分析数据库的潜力展开。


---

# 第一部分：高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 加速模型微调**：Unsloth 为 **LlaMA 3.1**、**Mistral** 和 **Gemma** 等模型的微调带来了 **2倍** 的速度提升，并将 VRAM 占用降低了 **70%**。
   - 讨论强调了量化模型与全量模型在存储需求上的不同，这会影响显存的可用性。
- **Qwen 2.5 登场**：近期发布的 **Qwen 2.5** 模型展示了改进的指令遵循能力，特别是在编程和数学方面。
   - 用户注意到它处理细微差别的能力优于 **Llama 3.1**，尽管在保存和重新加载合并模型时会出现问题。
- **Gemma 2 微调困境**：社区成员报告了在微调 **Gemma 2** 时遇到的挑战，特别是在保存和加载合并模型时遇到的错误。
   - 建议指出问题可能出在推理中使用的聊天模板（chat templates）或模型内部的通用持久性问题。
- **神经网络代码生成取得成功**：一位社区成员对在训练神经网络生成 **Python** 代码方面获得的帮助表示感谢，认为这是一个充满希望的开始。
   - 社区反响热烈，纷纷以“太棒了，祝贺！”来赞扬这一成就。
- **vLLM 服务引发延迟担忧**：一位使用 **vLLM** 进行服务的参与者提到了在微调模型时遇到的延迟问题。
   - 他们寻求关于使用 **Quantization Aware LoRa training** 的建议，并表达了对有效合并模型的担忧。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **LoRa 模型训练要点**：一位成员询问了训练 **LoRa 模型** 的有效图像，建议使用多样化的平面图、门和窗户来增强数据集。
   - 重点放在了标签标注和社区经验分享上，以帮助新手开启训练之旅。
- **分辨率之争：SD1.5 对比 SD512**：在生成 **1024x1024** 图像时，**SD1.5** 的表现优于 **512x512**，特别是考虑到生成过程中的 GPU 限制。
   - 建议采用 turbo 模型，以便在不牺牲效率的情况下实现更快的图像生成。
- **Multidiffusion 的省存魔力**：**multidiffusion** 扩展被誉为低 VRAM 用户的省存神器，它通过分块（tiled sections）处理图像。
   - 社区分享了指南和资源，帮助用户将此扩展有效地集成到工作流中。
- **Riffusion 震撼 AI 音乐创作**：**Riffusion** 平台支持通过声谱图生成音乐，并可能在未来的版本中加入 AI 歌词。
   - 讨论指出，在开源领域，除了 **Suno AI** 之外，能够生成完整歌曲的替代方案非常匮乏。
- **远程处理：一把双刃剑**：对于像 **iopaint** 这样使用远程处理的工具，用户表达了担忧，因为这限制了用户控制权和模型的灵活性。
   - 社区倡导自托管（self-hosting）模型，以增强定制化和隐私保护。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Mistral API 价格创下新低**：成员们强调了 **Mistral API** 的大幅降价，大型模型如 **Large 2** 的价格极具竞争力，低至 **$2/$6**。
   - 这一价格调整使其在与其他供应商的竞争中处于有利地位，提升了用户获取模型的便捷性。
- **OpenRouter 面临访问障碍**：多位用户在使用 **OpenRouter** 时遇到问题，特别是收到 **429 错误** 和 **Data error output** 消息。
   - 为了缓解这些问题，鼓励用户创建专门的帖子来报告错误，以简化故障排除流程。
- **速率限制（Rate limits）干扰用户工作负载**：用户因触发严格的速率限制而无法访问模型，导致生产力大幅下降，对此感到沮丧。
   - 一位用户指出他们被限制了 **35 小时**，引发了关于 BYOK (Bring Your Own Key) 等潜在解决方案的讨论。
- **回退模型（Fallback models）需要更好的策略**：讨论了在遇到速率限制错误时，使用 **回退模型** 与 **回退密钥** 的操作顺序。
   - 用户提出了对未能有效使用回退模型的担忧，特别是在面对 **Gemini Flash** 的 429 错误时。
- **用户咨询免费 LLM 访问**：一位用户询问如何在每月 **$10-$15** 的有效预算下，为 **5000 人**提供 **免费 LLM 访问**。
   - 随后展开了关于 Token 使用量的讨论，估计每人每天约 **9k tokens**，这需要极其复杂的优化策略。

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 表现异常**：用户发现 **Aider** 表现出不稳定的行为，有时在简单的请求后会执行自己的议程，需要重启才能解决。该问题似乎与会话期间的 context retention（上下文保留）有关。
   - 社区建议研究状态管理，以防止在未来的更新中出现此类意外行为。
- **对 OpenAI 模型的反馈令人失望**：用户批评了 O1 模型的性能，特别是拒绝服从格式化命令，这破坏了工作流效率。许多用户转向使用 **3.5 Sonnet**，理由是其对 prompt 的控制力更好。
   - 这引发了关于灵活参数设置对于增强用户与 AI 模型交互重要性的讨论。
- **探索 DeepSeek 的局限性**：围绕 **DeepSeek 模型** 的编辑和重构能力出现了挑战，并建议改进输入格式以获得更好的输出。提出了微调（tuning）方案，寻求有效的 source/prompt 示例进行测试。
   - 此次交流表明，集体需要更清晰的指南，以通过有效的 prompt 设计来优化模型性能。
- **Claude 3.5 系统提示词细节发布**：一个针对 **Claude 3.5 Sonnet** 的[提取出的系统提示词](https://gist.github.com/njpearman/ffdc8768dc37451bf2c8d5f93b6a905d)被分享出来，旨在增强处理 artifacts 时的性能。这一发现引发了对其在实际应用中如何发挥作用的兴趣。
   - 社区期待了解该提示词对实际应用和代码生成任务的影响。
- **FlutterFlow 5.0 发布增强功能**：一段 [YouTube 视频](https://www.youtube.com/watch?v=eKuKKdIglHA)介绍了 **FlutterFlow 5.0**，它承诺通过旨在简化组件创建的新功能来彻底改变应用开发。该更新声称有显著的性能提升。
   - 反馈显示，用户已经渴望实施这些功能，以提高编码工作流的效率。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **将 Perplexity Pro 与 VSCode 集成**：用户讨论了如何将 **Perplexity Pro** 模型与 VSCode 扩展（如 'Continue'）结合使用，以实现有效的 autocomplete 功能，尽管由于编程技能有限存在集成挑战。
   - 强调了 Pro Search 与纯 *writing mode* 之间的区别，这使某些用户的预测策略变得复杂。
- **在 Pro Search 中利用 O1 模型**：**O1-mini** 现在可以通过 Pro Search 中的 Reasoning 焦点进行访问，尽管其集成情况因模型选择而异。
   - 一些用户主张在角色扮演场景中使用 **O1**，因为它具有角色保持能力，但要求更高的使用限制。
- **关于 Perplexity 与 ChatGPT 的辩论**：一场关于 **Perplexity API** 模型与 ChatGPT 模型对比的辩论正在进行，特别是关于教育用途和订阅福利。
   - 一位用户指出了 **ChatGPT Plus** 对学生的优势，同时也承认了 **Perplexity Pro** 订阅的优点。
- **Slack 发布 AI Agents**：**Slack** 报告称引入了 **AI agents**，旨在提高平台内的工作流和沟通效率。
   - 该功能预计将提高使用该平台的团队的整体生产力。
- **Lucid 推出新款平价电动 SUV**：**Lucid** 推出了一款更实惠的新型**电动 SUV**，扩大了其市场覆盖范围，并吸引了具有环保意识的消费者。
   - 这款平价车型针对的是对可持续交通感兴趣的更广泛受众。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face 发布全新 API 文档**：新推出的 [API 文档](https://huggingface.co/docs/api-inference) 改进了对速率限制（rate limits）的说明，增加了专门的 PRO 专区，并强化了代码示例。
   - *用户反馈已被直接采纳以提升易用性*，使开发者部署 AI 更加顺畅。
- **TRL v0.10 支持视觉语言模型微调**：[TRL v0.10](https://x.com/QGallouedec/status/1833893093793304950) 将视觉语言模型（vision-language models）的微调简化至仅需两行代码，恰逢 Milstral 发布 Pixtral。
   - 此版本强调了**多模态 AI 能力（multimodal AI capabilities）**日益增强的连通性。
- **Nvidia 发布紧凑型 Mini-4B 模型**：点击[此处](https://huggingface.co/spaces/Tonic/Nemotron-Mini-4B)查看 Nvidia 全新的 **Mini-4B** 模型，该模型表现出色，但需要兼容的 Nvidia 驱动。
   - 鼓励用户将其注册为 **Hugging Face agent**，以发挥其全部功能。
- **开源生物特征模板保护**：一名成员分享了他们的**生物特征模板保护（BTP）**实现方案，可在无需服务器数据访问的情况下进行身份验证，代码已托管至 [GitHub](https://github.com/templateprotection/basic-btp)。
   - 这段教学代码旨在向初学者介绍安全生物识别系统的复杂性，同时保持易用性。
- **社区寻求图像转卡通模型**：社区成员正在寻找能够将图像转换为高质量卡通效果的 **space 模型**，并征集相关推荐。
   - *社区参与是关键*，他们鼓励分享满足这一需求的模型见解。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NousCon 激发热情**：参与者对 **NousCon** 表现出极大的热情，讨论了参会及未来活动，并计划在附近的酒吧举行派对以促进社区互动。
   - *许多人请求在不同地点举办未来活动*，强调了对更多社交机会的渴望。
- **Hermes Tool Calling 标准被采用**：社区已为 **Qwen 2.5** 采用了工具调用（tool calling）格式，这受到了 vLLM 支持等贡献以及正在讨论的其他未来实现工具的影响。
   - *关于 Hermes 和 Qwen 之间解析工具差异的讨论正在进行中*，激发了创新的集成想法。
- **Qwen 2.5 携新模型发布**：**Qwen 2.5** 已正式发布，具有全新的编程和数学模型，标志着开源 AI 进展的关键时刻。
   - 这一大规模发布展示了 AI 社区中语言模型的持续演进，并有详细的 [博客文章](https://qwenlm.github.io/blog/qwen2.5/) 概述其能力。
- **Gemma 2 提升游戏表现**：成员们分享了微调 **Gemma 2** 等模型以增强国际象棋对局体验的经验，尽管性能表现仍面临若干挑战。
   - *这反映了社区内创造性的开发过程和协作精神*，从游戏预期出发反向推动创新。
- **Hermes 3 API 访问权限确认**：已确认与 Lambda 合作提供 **Hermes 3** API 访问权限，允许用户使用全新的 Chat Completions API。
   - 进一步的讨论包括旨在最大化模型能力的潜在配置，特别是对以 `bf16` 精度运行的关注。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 会议致谢**：在 Triton 会议的主旨演讲中，**Mark Saroufim** 赞扬了社区的贡献，这令与会者感到兴奋。
   - 这一认可引发了关于社区参与和未来贡献的讨论。
- **Triton CPU / ARM 正式开源**：关于 **Triton CPU / ARM** 的咨询确认了其现已开源，可在 [GitHub](https://github.com/triton-lang/triton-cpu) 上获取。
   - 该倡议旨在促进社区协作并改进实验性的 CPU 后端。
- **Llama-2 模型训练性能报告**：[Llama2-7B-chat 模型](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) 的性能指标揭示了在各种任务中与 FP16 配置的显著对比。
   - 参与者强调了优化量化方法以增强推理质量的必要性。
- **高效量化技术**：讨论集中在有效的量化方法上，例如针对 Large Language Models 的 **4-bit quantization**，这对于 BitNet 的架构至关重要。
   - 成员们对应用无分组量化以降低推理成本的模型表现出兴趣。
- **即将发布的 Pixtral 模型**：围绕 **Transformers** 库中即将发布的 Pixtral 模型展开了热烈讨论，重点在于实现策略。
   - 成员们指出，发布后预计能与现有框架实现平滑集成。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **开源 TTS 迁移正在进行中**：讨论了从 **OpenAI TTS** 迁移到开源替代方案的问题，特别强调了支持多语言的 [Fish Speech V1.4](https://huggingface.co/fishaudio/fish-speech-1.4)。
   - 成员们辩论了使用 **xttsv2** 来增强不同语言性能的可行性。
- **MLRA 键的压缩技术**：成员们探索了利用额外的压缩矩阵处理 MLRA 键（keys）和值（values）的概念，旨在增强投影后的数据效率。
   - 有人对 MLRA 实验设置中细节不足（特别是关于秩矩阵的部分）表示担忧。
- **Playground v3 发布引发关注**：[Playground v3 (PGv3)](https://arxiv.org/abs/2409.10695) 发布，展示了在文本生成图像领域的领先性能，并为图像说明（image captioning）设立了新基准。
   - 新模型集成了 LLMs，不同于早期依赖预训练编码器的模型，证明了其效率更高。
- **引入 Diagram of Thought 框架**：提出了 **Diagram of Thought (DoT)** 框架，通过有向无环图（DAG）结构对 LLMs 中的迭代推理进行建模，旨在增强逻辑一致性。
   - 这种新方法相比之前研究中讨论的线性推理方法提出了显著改进。
- **调查模型调试策略**：一位成员建议从**工作基准（working baseline）**开始进行模型调试，并逐步识别各种配置（如 **FSDP**）中的问题。
   - 反复讨论强调了在优化模型性能时分享调试经验的必要性。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Custom GPTs 有效记忆片段**：成员们讨论了使用 Custom GPTs 导入和记忆个人片段（如 [Mac Snippets](https://link.to.mac.snippets)），尽管过量的信息堆砌带来了挑战。
   - 有建议指出，更清晰的指令和知识库上传可以提升性能。
- **泄露的 Advanced Voice Mode 发布信息**：预计将于 9 月 24 日为 Plus 用户推出即将到来的 **Advanced Voice Mode**，重点在于提高清晰度和响应时间，同时过滤噪音。
   - 社区对其在日常语音命令可用性方面的潜在影响表示好奇。
- **关于 AI 内容饱和的辩论**：一场激烈的讨论集中在 AI 生成的内容是提升了还是稀释了质量，有观点认为这反映了预先存在的低质量内容。
   - 随着 AI 能力的增长，人们对脱离现实的担忧日益增加。
- **GPT Store 托管创新作品**：一位成员推介了他们在 [GPT Store](https://your-gpt-store-link.com) 中的各种 GPTs，这些工具可以自动执行来自不同来源的任务，从而增强工作流。
   - 他们的产品中包含受文学启发的特定提示词技术，包括 DALL·E。
- **澄清频道内的自我推广规则**：成员们审查了自我推广规则，确认了 API 和 Custom GPTs 频道中分享作品的例外情况。
   - 成员们被鼓励链接他们的 GPTs，强调了社区在遵守服务器准则的同时支持分享。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 职位申请引发社区热议**：一名成员在申请 **Cohere** 的职位后分享了他们的热情，并寻求社区支持。
   - 社区以兴奋的态度欢迎这一举动，展现了对新人的友好氛围。
- **CoT-Reflections 表现优于传统方法**：讨论集中在 **CoT-reflections** 如何比标准的 Chain of Thought 提示词提高响应质量。
   - 成员们强调，将 **BoN** 与 **CoT-reflections** 结合可以显著提升输出质量。
- **关于 O1 奖励模型机制的推测**：成员们推测 **O1** 使用奖励模型运行，该模型会迭代调用自身以获得最佳结果。
   - 有迹象表明 **O1** 经历了**多阶段训练过程**以提升其输出质量。
- **账单信息设置困惑已解决**：一名成员在通过 **Stripe** 链接设置付款方式后，寻求关于添加 **VAT** 详情的澄清。
   - 建议发送电子邮件至 **support@cohere.com** 以安全处理账单变更，这被证实是一个可行的解决方案。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Markov Models 受到认可**：成员们指出 **Markov models** 是参数较少的语言概率模型，引发了对其潜在应用的讨论。
   - 工程师们对这些模型如何简化语言处理中的某些流程产生了浓厚兴趣。
- **训练时间引发辩论**：在 4090 GPU 上训练 **40M tokens** 大约需要 **5 天**，但减少到 **40k tokens** 则只需 **1.3 小时**。
   - 关于为什么 **100k 模型**的训练时间仍然显得**过长**，人们仍存有疑虑。
- **Data Loader 瓶颈引发挫败感**：成员们讨论了模型训练期间的 **data loader** 瓶颈，有报告称延迟导致了挫败感。
   - 呼吁探索数据流水线的优化技术，以提高整体训练效率。
- **LM Studio 令人兴奋的新功能**：随着新的文档处理功能发布，一名成员在整合之前的反馈后重新回到 **LM Studio**，引发了关注。
   - 讨论围绕理解数据表的大小限制以及通过软件分析数据库展开。
- **AI 模型推荐纷至沓来**：在编程推荐方面，**Llama 3.1 405B** 模型在 Prolog 辅助方面浮出水面，引发了各种意见。
   - 对 **qwen 2.5 0.5b** 等小模型替代方案的见解强调了其**连贯性**，尽管它缺乏小写支持。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Langchain 合作伙伴包更新**：有人询问将旧的 **Langchain** 社区集成更新为合作伙伴包的流程，建议通过联合沟通渠道联系。
   - *Lance Martin* 被提及为该过渡过程中寻求进一步协助的首选联系人。
- **Mistral 为开发者推出免费层级**：**Mistral** 在其无服务器平台上推出了免费层级，允许开发者在增强 **Mistral Small** 模型的同时免费进行实验。
   - 此次更新还包括修订后的定价，并在其聊天界面上引入了免费的视觉（vision）功能，使其更易于访问。
- **Qwen 2.5：基础模型的游戏规则改变者**：阿里巴巴推出了 **Qwen 2.5** 基础模型，引入了超过 **100 个变体**，旨在改进编码、数学推理和语言处理。
   - 该版本因其具有竞争力的性能和针对性的增强而受到关注，有望比早期版本取得重大进步。
- **Moshi Kyutai 模型震撼登场**：**Kyutai Labs** 推出了 **Moshi** 模型，并在多个平台上提供了技术报告、权重和流式推理代码。
   - 他们提供了论文、GitHub 和 Hugging Face 的链接，供渴望深入了解该模型能力的任何人使用。
- **Mercor 吸引重大投资**：**Mercor** 在 A 轮融资中以 **2.5 亿美元**的估值筹集了 **3000 万美元**，目标是利用先进模型增强全球劳动力匹配。
   - 本轮投资吸引了 **Peter Thiel** 和 **Jack Dorsey** 等知名人物，凸显了其在 AI 驱动的劳动力解决方案中的重要性。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 0.3 发布，功能丰富**：[Torchtune 0.3](https://github.com/pytorch/torchtune/releases/tag/v0.3.0) 引入了重大增强，包括对 **FSDP2** 的全面支持，以提升灵活性和速度。
   - 此次升级重点在于缩短训练时间，并改进各种任务中的模型管理。
- **FSDP2 增强分布式训练**：所有分布式 recipes 现在都利用 **FSDP2**，从而实现更好的编译支持并改进对 **LoRA** 参数的处理。
   - 鼓励用户在分布式 recipes 中尝试新配置，以获得更强的性能。
- **训练时间速度提升**：通过设置 `compile=True` 实现 **torch.compile**，使编译时间缩短至一分钟以内，从而实现更快的训练。
   - 使用最新的 **PyTorch nightlies** 版本可进一步放大性能，显著减少模型编译期间的耗时。
- **启用 DoRA/QDoRA 支持**：最新版本允许用户通过在配置中设置 `use_dora=True` 轻松激活 **DoRA/QDoRA**。
   - 这一新增功能对于增强与 **LoRA** 和 **QLoRA** recipes 相关的训练能力至关重要。
- **引发缓存管理讨论**：围绕每次任务后是否必须完全删除缓存展开了讨论，并提出了对 **eval harness** 的改进建议。
   - 一位贡献者建议确保模型在不需要拆除缓存的情况下，同时保持推理（inference）和前向（forward）模式。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Qwen2.5 发布，达成重大里程碑**：Qwen 家族的最新成员 **Qwen2.5** 被誉为规模最大的开源发布之一，涵盖了 **Qwen2.5-Coder** 和 **Qwen2.5-Math** 等模型，尺寸从 **0.5B** 到 **72B** 不等。
   - 亮点包括旗舰模型 **Qwen2.5-72B-Instruct** 能够匹配私有模型，在基准测试中展示了极具竞争力的性能。
- **OpenAI o1 模型媲美博士水平工作**：对 OpenAI 的 **o1-mini** 模型的测试表明，其表现可与生物医学领域优秀的博士生相媲美，被标记为他们训练过的顶尖候选模型之一。
   - 这一评价强调了该模型的精通程度及其在高级学术项目中的应用潜力。
- **数学推理（Math Reasoning）备受关注**：AI 领域越来越强调推进 **math reasoning** 能力，支持中英双语的 **Qwen2.5-Math** 模型引发了热议。
   - 用户的参与表明，在努力突破该领域界限的过程中，大家共同关注于增强数学相关的 AI 应用。
- **AI 模型知识截止日期（Knowledge Cutoff）的挑战**：几位用户对模型的 **knowledge cutoff** 表示沮丧，特别指出其设定在 **2023 年 10 月**，影响了其对较新编程库的适用性。
   - 讨论表明实时信息对于实际应用至关重要，这对像 OpenAI 的 **o1** 这样的模型构成了挑战。
- **Transformers 彻底改变 AI**：自 2017 年以来，**Transformer** 架构从根本上改变了 AI 方法，为 OpenAI 的 **GPT**、Meta 的 **Llama** 和 Google 的 **Gemini** 等模型提供了动力。
   - **Transformers** 的用途已从文本扩展到 [语音生成](https://huggingface.co/learn/audio-course/en/chapter3/introduction)、[图像识别](https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/vision-transformers-for-image-classification) 以及 [蛋白质结构预测](https://elifesciences.org/articles/82819)。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **01 App 已完全投入运行**：成员确认 **01 app** 在手机上运行良好，特别是使用 **-qr 选项**时效果最佳。
   - 一名成员对非本地版本进行了广泛测试，并报告其功能运行顺畅。
- **自动化浏览器任务请求**：一名成员正在寻求自动化浏览器表单提交（特别是针对政府门户网站）的**指南和技巧**。
   - 尽管遵循了 **ChatGPT 4o** 的建议，但他们仍面临效率低下的问题，特别是结果重复。
- **CV Agents 可供测试**：一名成员分享了他们的 **CV Agents** 项目，旨在通过 GitHub 上的智能简历增强求职体验：**[GitHub - 0xrushi/cv-agents](https://github.com/0xrushi/cv-agents)**。
   - 该项目邀请社区贡献，并配有极具吸引力的描述。
- **Moshi Artifacts 发布**：**Kyutai Labs** 发布了 **Moshi artifacts**，包括技术报告、模型权重以及支持 **Pytorch**、Rust 和 MLX 的流式推理代码，可在其 **[论文](https://kyutai.org/Moshi.pdf)** 和 **[GitHub 仓库](https://github.com/kyutai-labs/moshi)** 中获取。
   - 随着项目获得关注，社区热切期待更多更新。
- **音频同步反馈**：用户指出更新 Moshi 视频的缩略图可以提高曝光率和参与度。
   - 他们注意到视频中存在轻微的音频同步问题，表明需要进行技术调整。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Benito 的 RAG 部署突破**：Benito Martin 分享了使用 [AWS CDK](https://t.co/vsB3x9rYUY) 端到端构建和部署 **RAG 服务**的指南，为将原型转化为生产环境提供了宝贵资源。
   - *如果你想提升部署技能，这份指南是一个快速开始！*
- **KeyError 困扰 Weaviate 用户**：Yasuyuki 在读取现有 **Weaviate** 数据库时遇到了 **KeyError**，引用了 [GitHub Issue #13787](https://github.com/run-llama/llama_index/issues/13787)。一名社区成员建议 Fork 该仓库并创建一个 Pull Request，以允许用户指定字段名称。
   - *这是在查询非使用 llama-index 创建的向量数据库时常见的陷阱。*
- **Yasuyuki 的首次开源贡献**：Yasuyuki 表示有兴趣通过将键从 'id' 更改为 'uuid' 并准备 Pull Request 来为项目做出贡献。
   - *这次首次贡献鼓励了他熟悉 GitHub 工作流，以便未来参与。*
- **寻求 RAG 技术反馈**：.sysfor 寻求关于 **RAG** (Retrieval-Augmented Generation) 策略的反馈，以将供应商问题与索引的 QA 对关联起来。
   - 建议包括对 QA 对进行索引，并生成问题的变体以提高检索效率。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **模型提供商导致 LLM 延迟**：根据成员讨论，**LLM 响应延迟**主要与**模型提供商**有关，而非实现错误。
   - 这建议应专注于优化模型提供商设置以提高响应速度。
- **Python 和 LangChain 对延迟的影响微乎其微**：据称 **LLM 延迟**中只有 **5-10%** 归因于 **Python 或 LangChain**，这意味着应更多地关注模型配置。
   - 优化模型设置可以大幅提高整体性能并减少等待时间。
- **React 状态管理的最佳实践**：用户讨论了将 **Langserve** 与 **React 前端**集成时的最佳**状态管理**实践。
   - 对话暗示了有效状态处理的重要性，特别是在涉及 **Python 后端**的情况下。
- **用于高质量 PDF 提取的 PDF-Extract-Kit**：[PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) 作为一个用于高效 **PDF 内容提取**的综合工具包被展出。
   - 成员们考虑了其在解决常见 PDF 提取挑战中的实际应用，引发了广泛兴趣。
- **使用 AWS 技术栈开发的 RAG 应用**：一名成员展示了一个新的 [RAG 应用](https://github.com/benitomartin/aws-bedrock-opensearch-langchain)，利用 **LangChain** 和 **AWS Bedrock** 进行 LLM 集成和部署。
   - 该应用利用 **AWS OpenSearch** 作为向量数据库，突显了其处理数据的强大云能力。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **BeToast Discord 安全风险警报**：由于 LinkedIn 上关于黑客攻击事件的报告，人们开始担心 **BeToast** Discord 服务器可能遭到入侵。
   - *成员们强调必须保持警惕*，并准备好在任何受损账号开始发送垃圾信息时采取行动。
- **Windows 原生支持时间表尚不明确**：关于 **Windows 原生支持** 的讨论提到了一个 [GitHub issue](https://github.com/modularml/mojo/issues/620)，其中列出了功能需求，但实现的时间表尚不确定。
   - 许多开发者由于成本原因在 AI 项目中倾向于选择 **Windows** 以外的替代方案，通常使用 WSL 作为折中方案。
- **SIMD 转换为 Int 的解释**：一位用户询问如何将 **SIMD[DType.int32, 1]** 转换为 **Int**，一名成员简洁地回答道：`int(x)`。
   - 这强调了理解 **SIMD** 数据类型对于高效转换的重要性。
- **澄清 SIMD 数据类型**：对话强调了理解 **SIMD** 数据类型以实现平滑转换的必要性，并鼓励熟悉 **DType** 选项。
   - 成员们指出，这些知识可以简化未来关于数据处理的咨询。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **探索最先进的文本转语音技术**：一位成员询问了文本转语音的 **state of the art (SOTA)**，特别是寻求 [开源解决方案](https://discord.com/channels/823813159592001537/823813160075132991/1285823773202845706)。*“理想情况下是开源的，但也很好奇目前市面上都有哪些选择”*，这反映了对比各种方案的愿望。
   - 参与者称赞 [Eleven Labs](https://elevenlabs.io/) 是 **最佳闭源** 文本转语音选项，而对于开源爱好者，则推荐了 **styletts2**、**tortoise** 和 **xtts2** 等替代方案。
- **引入用于统一图像生成的 OmniGen**：名为 [OmniGen](https://arxiv.org/abs/2409.11340) 的论文介绍了一种新的 Diffusion 模型，它集成了多种控制条件，无需像 **Stable Diffusion** 等模型那样添加额外模块。OmniGen 通过其简化的架构支持多种任务，包括 **text-to-image generation**、**image editing** 和经典的 CV 任务。
   - OmniGen 利用了 **SDXL VAE** 和 **Phi-3**，增强了其生成图像和处理控制条件的能力，使其在各种应用中都非常易于使用。
- **Nvidia 官方开源 LLMs**：一位成员强调了 **Nvidia 官方开源 LLMs** 的可用性，这可能与正在进行的 AI 研究和开发相关。这一举措可能为该领域的开发者和研究人员提供宝贵的资源。
   - 此举支持了向更具协作性和可访问性的 AI 资源转型的趋势，符合当前开源软件的发展潮流。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Ruff 检查错误警报**：一位用户报告在执行 `ruff check . --fix-only` 时出现 **TOML 解析错误**，指出第 216 行存在 **未知字段** `indent-width`。
   - 该错误表明需要修改配置文件以符合预期的字段要求。
- **与 AI 研究者的播客**：由 Sayash Kapoor 和 Benedikt Stroebl 参与的 **YouTube 播客** 探讨了优化任务性能和最小化推理成本的方法，可在 [此处](https://youtu.be/gCP-W_BNzg4) 观看。
   - 讨论引起了广泛兴趣，强调了在 AI 系统中考虑成本的重要性。
- **LanceDB 集成首次亮相**：DSpy 的新 **LanceDB 集成** 增强了大数据的性能，详情请见 [此 Pull Request](https://github.com/stanfordnlp/dspy/pull/1444)。
   - 贡献者表示愿意在相关的个人项目和开源计划上进行协作。
- **关于 API Key 处理的担忧**：用户询问 API Key 是否需要在到达 OpenAI 之前直接发送到 VM/服务器，这凸显了对非官方服务器的 **信任问题**。
   - 明确安全流程对于避免个人数据泄露至关重要。
- **创建可重用的 RAG 流水线**：一位社区成员寻求关于创建 **可重用 RAG 流水线** 的指导，该流水线可以适应多家公司而不会使 Prompt 过载。
   - 成员们提出了关于如何有效整合多样化数据的担忧，旨在简化流程。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **在 PyTorch 中实现 Curriculum Learning**：要在 PyTorch 中实现 **curriculum learning**，需要定义标准，将数据集分割成难度递增的阶段，并创建一个自定义数据集类来管理此逻辑。
   - 一个示例展示了如何使用这种阶段性方法在训练循环中更新数据集。
- **控制数据集打乱 (Shuffling)**：一位用户提出了关于指定数据集中缺少 **random shuffling** 的问题，并就此寻求指导。
   - 有建议认为，为了清晰起见，可以在单独的线程中讨论此查询。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **需要 Tinybox 设置指南**：有人请求帮助设置 **two tinyboxes**，并提供了 [Tinybox documentation](https://docs.tinygrad.org/tinybox/) 的链接以获取设置指导。
   - 这突显了随着更多用户探索 **Tinygrad** 功能，对简化设置说明的需求。
- **Tinyboxes 助力 Tinygrad CI**：据指出，**tinyboxes** 在 **tinygrad** 的 **CI** 中发挥着至关重要的作用，通过在 [MLPerf Training 4.0](https://public.tableau.com/views/MLCommons-Training_16993769118290/MLCommons-Training) 上运行展示了其能力。
   - 这证明了它们作为 **tinygrad** 集成测试最佳平台的地位。
- **Tinybox 购买选项说明**：对于想要购买的用户，提到可以访问 [tinygrad.org](https://tinygrad.org) 购买 **tinybox**，并向其他人保证不买也没关系。
   - 这迎合了用户不同的兴趣，无论是购买还是探索。
- **Tinybox 特性揭秘**：简要概述强调了 **tinybox** 是一个用于 AI 工作负载的通用系统，可处理训练和推理任务。
   - 具体硬件规格包括配备六个 **7900XTX GPU** 的 **red box** 和配备六个 **4090 GPU** 的 **green box**。



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **rateLLMiter 现已支持 Pip 安装**：**rateLLMiter** 模块现在可以作为 Pip 安装包使用，增强了 LLM 客户端的请求管理。查看 [GitHub](https://github.com/llmonpy/ratellmiter) 上的实现细节，其中包含有关其 **MIT license** 的信息。
   - 此实现允许 LLM 客户端更好地管理其 API 调用，使其更容易集成到现有工作流中。
- **速率限制器图表展示请求管理**：一张图表说明了 **rateLLMiter** 如何平滑请求流，其中橙色代表票据请求，绿色代表已发放的票据。这有效地将 **100 requests** 的峰值随时间分散，以避免服务器速率限制异常。
   - 参与者强调了有效管理 API 速率的重要性，以确保在高峰负载期间与后端服务的无缝交互。



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **成员意识到 Prompt 使用不当**：一位成员承认他们错误地应用了 Prompt，这在 **Gorilla LLM** 讨论中引起了意外的输出混淆。
   - 这突显了验证 Prompt 使用以确保结果准确的必要性。
- **Prompt 模板现已可用**：同一位成员提到，现在可以轻松获取 Prompt 模板，以协助高效地构建未来的 Prompt。
   - 利用该模板有助于减少未来类似的 Prompt 相关错误。



---


**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1285683852718641224)** (196 条消息🔥🔥): 

> - `Unsloth 模型微调`
> - `Qwen 2.5 发布`
> - `Gemma 2 微调问题`
> - `Pytorch Conference`
> - `使用 WSL 进行安装` 


- **Unsloth 让模型微调更高效**：据报道，Unsloth 将 LlaMA 3.1、Mistral 和 Gemma 等模型的微调速度提高了 **2 倍**，同时减少了 **70% 的 VRAM** 使用量。
   - 围绕将模型推送到 Hub 的讨论强调了量化模型与全量模型对存储需求的不同，这会影响可用内存。
- **Qwen 2.5 刚刚发布**：Qwen 2.5 模型最近发布，承诺在指令遵循以及编码和数学等领域的能力有所提升。
   - 用户注意到 Qwen 2.5 比 Llama 3.1 能更好地处理细微差别，尽管在保存和重新加载合并模型方面存在疑虑。
- **Gemma 2 面临微调复杂问题**：参与者报告了微调 Gemma 2 的问题，特别是与保存和重新加载合并模型时的错误有关。
   - 建议指出这可能与用于推理的 Chat Templates 或一般的模型持久化问题有关。
- **Pytorch Conference 更新**：一位参与者宣布他们将在 Pytorch Conference 上发表演讲，分享关于改进 LLM 训练的创新。
   - 会议预计将被录制，让错过的参会者能够了解演示期间分享的见解。
- **关于使用 WSL 进行安装的讨论**：用户询问了在 Windows 下使用 WSL 安装模型的问题，讨论了优化设置的各种方法。
   - 建议仅推送 Adapters，以避免在模型训练和部署过程中出现空间问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://qwenlm.github.io/blog/qwen2.5/">Qwen2.5: A Party of Foundation Models!</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介 在 Qwen2 发布后的过去三个月里，众多开发者在 Qwen2 语言模型的基础上构建了新模型，为我们提供了...</li><li><a href="https://www.youtube.com/@LlamaSeb">LlamaSeb</a>: 我致力于探索 AI、Machine Learning 和 Deep Learning 的迷人世界。在这里，你会发现深入探讨最新 AI 工具、技术和趋势的视频，特别是...</li><li><a href="https://huggingface.co/flowaicom/Flow-Judge-v0.1">flowaicom/Flow-Judge-v0.1 · Hugging Face</a>: 暂无描述</li><li><a href="https://download.pytorch.org/whl/cu124">无标题</a>: 暂无描述</li><li><a href="https://unsloth.ai/blog/llama3-1">使用 Unsloth 微调 Llama 3.1</a>: 通过 Unsloth 微调并运行 Meta 更新的 Llama 3.1 模型，支持 6 倍长的上下文长度！</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing)">Google Colab</a>: 暂无描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fk0acj/hacks_to_make_llm_training_faster_guide/">Reddit - 深入探索一切</a>: 暂无描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1285681143831597169)** (20 条消息🔥): 

> - `Neural Network Code Generation` (神经网络代码生成)
> - `Path Specification for Llama-CPP` (Llama-CPP 的路径规范)
> - `Fine-tuning Llama Models` (Fine-tuning Llama 模型)
> - `LoRa Quantization Issues` (LoRa 量化问题)
> - `vLLM Serving Performance` (vLLM Serving 性能)


- **神经网络代码生成成功**：一位成员对社区在训练神经网络生成 **Python** 代码方面的帮助表示感谢，并称其为**大事业的小起点**。
   - 社区回应道：*“太棒了，恭喜！”*，展示了对这一成就的支持。
- **指定 Llama-CPP 路径**：一位用户询问在 `save_pretrained_gguf` 过程中遇到错误时，如何指定 **llama-cpp** 的路径。
   - 另一位成员建议将文件路径添加到系统环境变量 `PATH` 中，以便自动识别路径。
- **Fine-tuning Llama 模型导致延迟问题**：一位用户分享了他们 Fine-tuning 模型并遇到**延迟问题**的经历，考虑使用 **vLLM** 进行 Serving。
   - 他们就该方法寻求建议，特别是关于 **Quantization Aware LoRa** 训练和合并模型的问题。
- **将 LoRa 与量化模型合并**：在使用 **vLLM** 进行推理时，有建议认为成员不需要合并 **LoRa** adapter，因为它自己可以处理加载。
   - 一位用户解释了他们在加载 **LoRa** adapter 时遇到的困难，暗示可能丢失了 Fine-tuning 的效果。
- **vLLM Serving 的挑战**：一位成员概述了他们使用 **vLLM** Serving 模型的命令，但指出了正确加载 **LoRa** adapter 的问题。
   - 他们不确定 **LoRa** 是否已正确加载，或者 Fine-tuning 的效果是否减弱了。



**提到的链接**：<a href="https://github.com/unslothai/unsloth/blob/main/unsloth/save.py#L842">unsloth/unsloth/save.py at main · unslothai/unsloth</a>：Fine-tuning Llama 3.1, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1285686218427994263)** (161 条消息🔥🔥): 

> - `训练 LoRa 模型`
> - `图像生成技术`
> - `Multidiffusion 使用`
> - `音频生成工具`
> - `通用 AI 讨论` 


- **训练 LoRa 模型的最佳实践**：一位成员询问了有效训练 LoRa 模型所需的图像，建议收集各种平面图以及门窗等部件进行训练。
   - 讨论还强调了打标签的重要性，以及在社区内分享经验以帮助新手。
- **图像生成分辨率建议**：成员们讨论了在不同分辨率下生成图像的问题，指出与 512x512 相比，SD1.5 在 1024x1024 图像上的表现更好，同时考虑了 GPU 限制。
   - 一项建议提议使用 turbo 模型，在保持效率的同时实现更快的生成速度。
- **用于放大的 Multidiffusion 工具**：向低 VRAM 用户推荐了 multidiffusion 扩展，它被描述为一个 tiled sampler，通过处理图像的较小部分来节省显存。
   - 分享了指南和资源的链接，帮助用户了解如何在他们的工作流中有效地实施这一点。
- **Riffusion 和音频 AI**：Riffusion 被提及作为一个从频谱图生成音乐的平台，并有可能将其与 AI 驱动的歌词生成相结合。
   - 对话探讨了开源音频生成工具的现状，指出目前缺乏用于全曲生成的 Suno AI 替代方案。
- **AI 工具中远程处理的挑战**：有人对 iopaint 等使用远程处理的工具表示担忧，这影响了用户对模型的灵活性和控制力。
   - 讨论强调了自行托管模型在隐私和定制化方面的优势。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.riffusion.com/">Riffusion</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/FXFullLoaded/comments/1fhj6nn/tradingview_premium_free_version_available_for/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/stainless-cypress">stainless-cypress - Overview</a>: GitHub 是 stainless-cypress 构建软件的地方。</li><li><a href="https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111">GitHub - pkuliyi2015/multidiffusion-upscaler-for-automatic1111: Tiled Diffusion and VAE optimize, licensed under CC BY-NC-SA 4.0</a>: Tiled Diffusion 和 VAE 优化，采用 CC BY-NC-SA 4.0 许可 - pkuliyi2015/multidiffusion-upscaler-for-automatic1111</li><li><a href="https://github.com/shiimizu/ComfyUI-TiledDiffusion">GitHub - shiimizu/ComfyUI-TiledDiffusion: Tiled Diffusion, MultiDiffusion, Mixture of Diffusers, and optimized VAE</a>: Tiled Diffusion, MultiDiffusion, Mixture of Diffusers, 以及优化的 VAE - shiimizu/ComfyUI-TiledDiffusion</li><li><a href="https://exactly.ai">exactly.ai</a>: 为艺术家提供的先进 AI 艺术品创作平台，能够理解您的风格，创作出激发灵感的图像并简化您的创作过程。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1285703205409390697)** (126 messages🔥🔥): 

> - `OpenRouter issues`
> - `Mistral API price drops`
> - `Rate limits and model access`
> - `Backup model usage`
> - `LLM allocation for users` 


- **Mistral API 经历显著降价**：成员们强调了最近 Mistral API 的大幅降价，指出大型模型的价格极具竞争力。
   - 例如，一位用户提到 **Large 2 的价格为 $2/$6**，与其他模型相比非常有优势。
- **对 OpenRouter 可访问性和错误的担忧**：多位用户报告了访问 OpenRouter 服务时持续出现的问题，特别是遇到 **429** 错误和 **Data error output**。
   - 建议通过创建帖子并提供详细的错误示例来报告问题，以便更清晰地进行故障排除。
- **速率限制（Rate limits）影响用户体验**：用户对被限制到无法访问模型的程度表示沮丧，这显著影响了生产力。
   - 一位用户提到被 **最高速率限制长达 35 小时**，引发了关于使用 BYOK (Bring Your Own Key) 等替代方案来绕过限制的讨论。
- **错误期间备用模型（fallback models）的使用**：成员们讨论了在遇到 **429 错误** 时实施备用模型的挑战，并对其有效性表示不确定。
   - 有人指出 `4xx` 错误代表不可恢复的问题，需要人工干预而非自动回退。
- **针对庞大用户群体的 LLM 查询计算**：一位用户询问如何在每月 **$10-$15** 的预算内为约 5000 人提供 **免费 LLM 访问**，引发了关于 Token 分配的讨论。
   - 提供了关于有效使用率的见解，根据月度预算估计**每位用户每天约 9k tokens**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gemini.google.com.">‎Gemini - 激发灵感的聊天工具</a>：Bard 现已更名为 Gemini。从 Google AI 获取写作、规划、学习等方面的帮助。</li><li><a href="https://openrouter.ai/models/openai/chatgpt-4o-lat">OpenRouter</a>：LLM 路由与市场</li><li><a href="https://openrouter.ai/credits">Credits | OpenRouter</a>：管理您的额度和支付历史</li><li><a href="https://openrouter.ai/models/openai/chatgpt-4o-latest">ChatGPT-4o - API, Providers, Stats</a>：持续更新至 ChatGPT 中当前 [GPT-4o](/models/openai/gpt-4o) 版本的动态模型。旨在用于研究和评估。通过 API 运行 ChatGPT-4o。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1285980212458295358)** (21 messages🔥): 

> - `Fallback Model Behavior`
> - `API Key Management`
> - `Rate Limiting with Gemini Flash`
> - `User Implementation of Fallbacks` 


- **备用模型（Fallback Models）优先级需要明确**：成员们讨论了在遇到速率限制问题时（尤其是使用 **Gemini Flash Exp** 时），使用**备用模型**与**备用密钥**的先后顺序。
   - *一位用户观察到 429 错误*，质疑为什么在某些情况下没有使用他们指定的备用模型。
- **Double Chat 困惑**：一位成员澄清了围绕 *double chat* 的困惑，确保他们会在同一个帖子中简化讨论以避免混乱。
   - 另一位成员表示，不用担心讨论重叠的问题。
- **用户对备用方案的变通方法**：一位成员提到他们手动实现了自己的*备用解决方案*，解决了备用模型的即时问题。
   - 他们强调这种方法值得其他面临类似挑战的人参考。
- **关于模型滥用的担忧**：讨论强调了这样一种担忧：允许回退到付费模型可能会导致**滥用**，即用户利用免费访问权限。
   - 成员们一致认为有必要实施限制，以防止在免费账户条件下过度访问付费功能。
- **对备用方案的普遍不满**：用户对僵化的备用政策表示恼火，特别是涉及 **Gemini 模型** 时。
   - 虽然他们理解这些政策背后的原因，但发现它们在实践中既不切实际又繁琐。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1285679654639767612)** (109 messages🔥🔥): 

> - `Aider Performance` (Aider 性能)
> - `Using OpenAI Models` (使用 OpenAI 模型)
> - `O1 Mini Feedback` (O1 Mini 反馈)
> - `DeepSeek Model Testing` (DeepSeek 模型测试)
> - `OpenAI API Costs` (OpenAI API 成本)


- **Aider 出现异常行为**：用户报告 Aider 表现不稳定，一名用户描述了一个问题：在发出简单请求后，AI 在很长一段时间内持续执行其自身的议程。
   - 在重启应用程序后，他们注意到问题得到了解决，这表明该问题与该会话期间保留的 context 有关。
- **关于 OpenAI 模型的反馈**：几位用户对 O1 模型的表现表示失望，提到了诸如拒绝遵守格式指令等限制，这阻碍了工作流的效率。
   - 用户讨论了尝试其他模型的情况，并提到了使用 3.5 Sonnet 取得的成功，强调需要对 system prompts 等参数进行更多控制。
- **DeepSeek 模型的使用和局限性**：讨论强调了 DeepSeek 模型面临的挑战，特别是在编辑和重构方面的表现，以及需要特定格式来改善结果。
   - 一位用户分享了使用新编辑格式调优 DeepSeek 的见解，并寻求导致较差结果的 source/prompt 对示例进行对比。
- **OpenAI API 使用的财务考量**：用户分享了与 OpenAI API 使用相关的成本见解，提到公司支付的 API key 授权，并对典型的每月支出表示好奇。
   - 一位用户表示每月花费约 200-300 美元，公司承担 70% 的费用，这引发了关于 API 费用预算的讨论。
- **编程中的实验与自动化**：对话显示用户正在探索编程任务的自动化方法，并分享了使用 AI 工具优化代码编写的各种策略。
   - 一位用户描述了如何使用 Aider 为编码任务创建结构化计划以改进工作流，建议将任务分解为可管理的步骤可能会带来好处。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 在浏览器本地存储数据。</li><li><a href="https://cortex.so">Homepage - Cortex</a>: 未找到描述</li><li><a href="https://aider.chat/docs/usage/commands.html#interrupting-with-control-c">In-chat commands</a>: 使用 /add、/model 等聊天内命令控制 aider。</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/import.md#Importing-a-GGUF-based-model-or-adapter">ollama/docs/import.md at main · ollama/ollama</a>: 快速上手 Llama 3.1、Mistral、Gemma 2 和其他大型语言模型。- ollama/ollama</li><li><a href="https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/tree/main">bartowski/Qwen2.5-Coder-7B-Instruct-GGUF at main</a>: 未找到描述</li><li><a href="https://github.com/All-Hands-AI/OpenHands">GitHub - All-Hands-AI/OpenHands: 🙌 OpenHands: Code Less, Make More</a>: 🙌 OpenHands：少写代码，多做创造。通过创建账号参与 All-Hands-AI/OpenHands 的开发。</li><li><a href="https://fluxcanvas.art/">The most powerful no-code platform</a>: Bubble 引入了一种构建 Web 应用程序的新方法。它是一个无代码的点选式编程工具。Bubble 在其云平台上托管所有应用程序。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1285700310832250980)** (17 条消息🔥): 

> - `Marblism 工具`
> - `Aider 功能增强`
> - `RAG 系统集成`
> - `Markdown 与 XML 的讨论`
> - `用户与 Aider 的互动` 


- **Marblism 提供基于用户故事的方法**：成员们讨论了 [Marblism](https://marblism.com)，它在应用创建方面与 Aider 相似，并为每个页面设置引入了用户故事（User Stories），从而优化了开发流程。
   - *Aider* 可以通过采用类似的框架受益，以根据用户反馈改进功能生成。
- **Aider 的网页抓取行为需要明确**：有用户对使用 `/web` 命令与 `/add` 命令的区别提出了疑虑，因为前者在抓取 URL 后会立即启动补全（Completion），而无需用户进一步输入。
   - 有建议提出通过增加一个类似于 `/add` 的网页版命令来改善用户体验，从而允许更多控制。
- **对 RAG 系统集成的兴趣**：一位用户询问将 RAG 系统与 Aider 集成以增强其能力是否可行。
   - 社区有兴趣探索新的集成方式，以提升 Aider 的功能性。
- **初学者的 Markdown 与 XML 之争**：讨论指出一些用户觉得 Markdown 具有挑战性，建议使用 XML 等替代方案来格式化他们的 Prompt。
   - 社区成员强调了为 Aider 新手提供清晰指令和简便格式的重要性。
- **用户参与度和生产力提升**：一位用户分享了他们在使用 Aider 时提高生产力的新 Prompt，其中包括确保所有必要文件都已就绪。
   - 反馈强调了社区的互助性质，成员们分享技巧和策略来互相帮助。



**提到的链接**：<a href="https://aider.chat/docs/usage/images-urls.html#web-pages">Images &amp; web pages</a>：将图像和网页添加到 Aider 编码聊天中。

  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1285680085235531978)** (9 messages🔥): 

> - `Claude 3.5 Sonnet system prompt`
> - `RethinkMCTS`
> - `JavaScript trademark concerns`
> - `Fine-tuning GPT-4o`
> - `FlutterFlow 5.0` 


- **Claude 3.5 system prompt 揭晓**：分享了一个针对 artifacts 的 [**Claude 3.5 Sonnet** 提取版 system prompt](https://gist.github.com/njpearman/ffdc8768dc37451bf2c8d5f93b6a905d)。
   - 该 prompt 旨在提升性能，但用户对其应用方式感到好奇。
- **RethinkMCTS 应对代码生成挑战**：标题为 [RethinkMCTS](https://www.arxiv.org/abs/2409.09584) 的论文讨论了通过树搜索算法增强 **LLM agents** 在代码生成中的表现，解决了搜索质量低的问题。
   - 它引入了一种思维层级的搜索方法，显著扩展了策略探索。
- **JavaScript 商标放弃辩论**：[javascript.tm](https://javascript.tm/) 上的一篇文章指责 Oracle 据称放弃了 **JavaScript** 商标，导致公众困惑。
   - 讨论强调 JavaScript 已成为一个通用术语，应属于公共领域。
- **用户友好的 GPT-4o fine-tuning**：@AlexTobiasDev 宣布了一个针对 **GPT-4o** 的 fine-tuner，允许**非技术**用户轻松创建用于 fine-tuning 的 JSONL 数据集，链接见[此处](https://github.com/alextobias78/Fine-Tuner)。
   - 据报道，该工具在简化 fine-tuning 流程的同时解决了常见 bug。
- **FlutterFlow 5.0 发布新功能**：一段 [YouTube 视频](https://www.youtube.com/watch?v=eKuKKdIglHA)展示了 **FlutterFlow 5.0**，引入了改变游戏规则的功能以增强应用开发。
   - 此版本承诺在构建灵活组件方面有显著改进。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://javascript.tm/">JavaScript™</a>：Oracle，是时候释放 JavaScript 商标了。加入我们，要求 Oracle 释放商标，并帮助我们向 USPTO 提交商标撤销申请。</li><li><a href="https://www.arxiv.org/abs/2409.09584">RethinkMCTS: Refining Erroneous Thoughts in Monte Carlo Tree Search for Code Generation</a>：通过树搜索算法增强的 LLM agents 在代码生成方面取得了显著表现。然而，该领域的当前搜索算法由于多种原因导致搜索质量较低...</li><li><a href="https://x.com/AlexTobiasDev/status/1836367037515407448">Alex Tobias (@AlexTobiasDev) 的推文</a>：为 GPT-4o 创建了一个 fine-tuner，非编程人员/非技术人员可以用它构建用于 @OpenAI 的 GPT-4o fine-tuning 的 .JSONL 数据集。简单且极其高效。解决了所有可能的...</li><li><a href="https://www.youtube.com/watch?v=eKuKKdIglHA">Introducing FlutterFlow 5.0</a>：FlutterFlow 5.0 带着改变游戏规则的新功能来了，为您的应用开发注入动力！⚡️🚀 Widget Builder：通过传递...构建极其灵活的组件。</li><li><a href="https://gist.github.com/njpearman/ffdc8768dc37451bf2c8d5f93b6a905d">Extracted Claude 3.5 Sonnet system prompt for artifacts</a>：提取的针对 artifacts 的 Claude 3.5 Sonnet system prompt - claude_35_artifacts_system_prompt.txt
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1285678840353652797)** (120 messages🔥🔥): 

> - `Perplexity Pro Model Integration`
> - `O1 and Reasoning Focus`
> - `Perplexity API vs ChatGPT`
> - `Challenges with Perplexity Features`
> - `User Experience with Extensions` 


- **将 Perplexity Pro 模型与 VSCode 集成**：用户讨论了如何将 **Perplexity Pro** 模型与 'Continue' 等 VSCode 扩展结合使用，以实现高效的自动补全功能。
   - *一位用户提到了集成挑战*，原因是编程技能有限，以及 Pro Search 与纯 *writing mode* 之间的区别。
- **O1 模型的可用性与使用**：用户确认 **O1-mini** 可以通过 Pro Search 中的 Reasoning focus 使用，但其集成情况因模型选择和设置而异。
   - 一些用户更倾向于在角色扮演场景中使用 **O1**，因为它能更好地维持人设；而另一些用户则强调了提高使用限额的必要性。
- **Perplexity 与 ChatGPT 的对比**：关于 **Perplexity API** 中的模型是否优于非 Pro 订阅版模型的辩论展开，尤其是在教育背景下。
   - 一位用户指出 **ChatGPT Plus** 的可用性，强调它可能为学生提供更多功能，但也承认了 **Perplexity Pro** 订阅的优势。
- **Perplexity 功能的问题**：一位用户对 **thread search function**（线程搜索功能）表示担忧，提到搜索结果似乎并不总能如预期般一致。
   - 这一持续存在的问题令用户感到沮丧，因为他们期望从搜索查询中获得全面的结果。
- **用户对扩展程序的反馈**：适用于 Firefox 的 **Complexity extension** 因增强了 Perplexity 体验而受到称赞，它允许改进模型和 collection 的选择。
   - 几位用户表达了对 **iOS/Safari** 平台上类似扩展的渴望，同时也承认该扩展的开源性质可能允许未来的适配。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.wheresyoured.at/subprimeai/">The Subprime AI Crisis</a>：我在这份简报中所写的任何内容都不是为了制造怀疑或“仇恨”，而是对我们现状以及当前路径可能走向的冷静评估。我相信人工...</li><li><a href="https://cplx.vercel.app/">Complexity</a>：每个人都梦寐以求的 Perplexity.ai 增强版。</li><li><a href="https://www.perplexity.ai/backtoschool">Perplexity - Race to Infinity</a>：欢迎回到学校！在仅限两周的时间内，领取一个月免费的 Perplexity Pro。推荐你的朋友，如果你的学校达到 500 人注册，我们将把免费月份升级为整年免费...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1285718382678638592)** (7 messages): 

> - `Slack AI Agents`
> - `Lucid Electric SUV`
> - `Bitcoin Puzzle`
> - `Windows Registry Tips`
> - `Motorola Smartphones` 


- **Slack 首次推出 AI Agents**：Perplexity AI 报道称 **Slack** 引入了新的 **AI agents**，增强了其平台内的协作功能。
   - 此举旨在简化工作流程并提高用户的沟通效率。
- **Lucid 推出平价电动 SUV**：讨论重点介绍了 **Lucid** 发布的一款价格更亲民的**电动 SUV**，扩大了其市场份额。
   - 该车型预计将吸引更多寻求可持续交通方案的客户群。
- **比特币 66 位谜题被破解**：社区庆祝与 **Bitcoin** 相关的 **66 位谜题**被成功破解，展示了计算挑战方面的进展。
   - 这一事件进一步强调了加密货币技术及其密码学基础的持续演进。
- **分享 Windows 注册表技巧**：一位成员分享了关于[如何添加和创建 Windows 注册表](https://www.perplexity.ai/search/how-to-add-create-windows-regi-2vByzG_uRDGK5D5M7K_R0g)条目的资源，对系统优化非常有用。
   - 该指南帮助用户更有效、更安全地管理其系统设置。
- **摩托罗拉智能手机见解**：一个讨论线程评估了多款 **Motorola Moto 智能手机**，特别关注其轻薄设计和性能表现。
   - 这一评估是关于智能手机创新和用户偏好的持续对话的一部分。



**提到的链接**：<a href="https://www.youtube.com/embed/GEC9vV4YCwY">YouTube</a>：未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1285751807376883743)** (3 条消息): 

> - `Perplexity API 一致性`
> - `新 API 功能的时间表` 


- **增强 API 准确性的策略**：一位成员分享了他们提高 Perplexity API **一致性**和**准确性**的方法，即在通过 GPT 总结结果之前，针对模糊主题使用 **query expansion** 和 **variations**。
   - *这种方法提高了一致性*，但他们指出由于 **rate limits**，该方法并不完美。
- **等待新 API 功能**：另一位成员询问了*新 API 功能的时间表*，表示渴望开始展示图片。
   - 他们意识到自己还没有发送有关申请的电子邮件，这可能延迟了他们的回复。


  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1285961952757153854)** (1 messages): 

> - `Hugging Face API Docs`
> - `TRL v0.10 Release`
> - `PySpark for HF Datasets`
> - `Sentence Transformers v3.1`
> - `DataCraft Introduction` 


- **翻新后的 Hugging Face API 文档上线**：新的 [API 文档](https://huggingface.co/docs/api-inference) 已发布，带来了诸如更清晰的速率限制（rate limits）、专门的 PRO 专区以及更详细的代码示例等改进功能。
   - *让部署 AI 变得简单*，并根据用户反馈进行了优化以提升易用性。
- **TRL v0.10 释放视觉-语言模型能力**：[TRL v0.10](https://x.com/QGallouedec/status/1833893093793304950) 已发布，仅需两行代码即可实现视觉-语言模型（vision-language models）的微调，恰逢 Mistral 发布 Pixtral。
   - 这一及时的更新突显了 AI 项目中多模态能力集成的日益增长。
- **PySpark 优化 HF Datasets 访问**：针对 ✨PySpark✨ 的新代码片段允许用户轻松地在 HF Datasets 中进行读写操作，增强了数据处理能力。
   - 它提供了一个优化的分布式解决方案，使用户与数据集的交互更加简单。
- **Sentence Transformers v3.1 增强模型训练**：最新版本 [Sentence Transformers v3.1](https://github.com/UKPLab/sentence-transformers/releases/tag/v3.1.0) 包含一个难负样本挖掘（hard negatives mining）工具和一种新的强损失函数，以获得更好的模型性能。
   - 它还支持使用流式数据集（streaming datasets）进行训练，并支持自定义模块和各种错误修复。
- **DataCraft 彻底改变合成数据集创建**：[DataCraft](https://huggingface.co/spaces/argilla/distilabel-datacraft) 已推出，旨在利用自然语言构建合成数据集，简化了数据集生成过程。
   - 这一无代码 UI 工具结合了创建高质量合成数据的最佳实践，使原本复杂的任务变得流程化。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/Wauplin/status/1835715850583564713)">Wauplin (@Wauplin) 的推文</a>: 我很高兴能揭晓我们翻新后的 Inference API 文档！我们正面解决了大家的反馈：更清晰的速率限制、专门的 PRO 专区、更好的代码示例以及详细的参数列表...</li><li><a href="https://x.com/QGallouedec/status/1833893093793304950)">Quentin Gallouédec (@QGallouedec) 的推文</a>: 时机完美！@MistralAI 发布了 Pixtral，这是他们的第一个多模态模型，而我们新发布的 TRL 正好增加了两行代码微调视觉-语言模型的功能 🌟</li><li><a href="https://x.com/qlhoest/status/1829145570465722578)">Quentin Lhoest 🤗 (@qlhoest) 的推文</a>: 🤗Hugging Face Datasets 用户们欢呼吧！我为 ✨PySpark✨ 编写了几行代码，用于读写 HF Datasets。全部采用分布式且经过优化！代码片段 / 文档和 JupyterLab 演示见下方 🧡</li><li><a href="https://x.com/tomaarsen/status/1833870859552928172)">tomaarsen (@tomaarsen) 的推文</a>: Sentence Transformers v3.1 发布了！具有难负样本挖掘工具，可从您的数据中获得更好的模型，还有新的强损失函数、流式数据集训练、自定义模块、错误修复...</li><li><a href="https://x.com/dvilasuero/status/1835711765570630017)">Daniel Vila Suero (@dvilasuero) 的推文</a>: 🧶 介绍 DataCraft：使用自然语言构建合成数据集！创建高质量的合成数据很困难，这是一个反复试验的过程，需要很多技巧。DataCraft 提供...</li><li><a href="https://x.com/pcuenq/status/1834616110475514343)">Pedro Cuenca (@pcuenq) 的推文</a>: 宣布 SAM 2 Studio 和 Core ML Segment Anything 2！我对设备端 ML 感到非常兴奋，并坚信它将是 AI 未来的重要组成部分。我们将 Segment Anything 2 转换为了...</li><li><a href="https://x.com/OzzyGT/status/1834594141822406796)">Alvaro Somoza (@OzzyGT) 的推文</a>: 想知道如何使用 diffusers 擦除/填充图像的部分内容吗？虽然花了一些时间，但我终于有了一个新的指南和一个你可以尝试的 Space。你可以在这篇博客文章中阅读相关内容：https...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1285713352059781170)** (101 条消息🔥🔥): 

> - `Hugging Face 会议出席情况`
> - `LLM 中的 JSON 输出`
> - `Moshi Checkpoint 发布`
> - `ADIFY AI 播放列表生成器`
> - `Qwen2.5 Math Demo 发布` 


- **PyTorch Conference 上的 Hugging Face 社区**：一位用户询问是否有 Hugging Face 社区成员参加在旧金山举行的 PyTorch 会议，表达了线下（AFK）见面的愿望。
   - 另一位用户分享了一条关于 Hugging Face 为会议做准备的推文，提到了像袜子之类的周边（swag）。
- **JSON 与更简单的输出格式**：讨论了为什么 LLM 通常被设计为输出 JSON 而不是更简单的解析格式，一位参与者指出结构化输出可能会对性能产生负面影响。
   - 对话强调虽然结构化输出对质量的损害可能较小，但有人建议采用一种将普通文本与结构化输出分离的工作流以提高效率。
- **Moshi Checkpoint 现已发布**：一位用户提到了 Moshi checkpoint 及其代码的发布，将其描述为来自 Kyutai 团队的实验性低延迟对话式 AI。
   - Moshi 项目是开源的，包含 PyTorch 和 Rust 实现，并包括 Moshiko 和 Moshika 两种声音。
- **ADIFY AI 介绍**：一位用户推广了 ADIFY AI，这是一个智能 Spotify 播放列表生成器，可根据用户定义的情绪或活动创建自定义播放列表。
   - 另一位参与者提醒该用户不要进行自我推广，引发了一场关于该话题的幽默交流。
- **Qwen2.5 Math Demo 公开**：社区收到了关于 Qwen2.5 Math Demo 的更新，该 Demo 展示了令人印象深刻的结果，被认为是一个创新的发布。
   - 鼓励用户查看 Hugging Face 上的 Demo Space，展示了其惊人的能力。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Qwen/Qwen2.5-Math-Demo">Qwen2.5 Math Demo - a Hugging Face Space by Qwen</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev">FLUX.1 [dev] - a Hugging Face Space by black-forest-labs</a>: 未找到描述</li><li><a href="https://x.com/JulienBlanchon/status/1836421431220920428">来自 Julien Blanchon (@JulienBlanchon) 的推文</a>: Moshi，由 @kyutai_labs 开发的实验性低延迟对话式 AI 刚刚开源：https://github.com/kyutai-labs/moshi - PyTorch, MLX 和 Rust (Candle 🔥) 实现 -...</li><li><a href="https://www.youtube.com/watch?v=Qq0SjONbWOE">科学技术史问答（2024年9月18日）</a>: Stephen Wolfram 主持了一场关于科学技术史的直播，不设脚本，面向所有年龄段。在这里找到问答播放列表：https:...</li><li><a href="https://x.com/osanseviero/status/1834508940417040487">来自 Omar Sanseviero (@osanseviero) 的推文</a>: 这就是 Hugging Face 团队为下周的 PyTorch Conference 做准备的方式🤗 到时见，来参加我们的派对领取精美周边吧！</li><li><a href="https://arxiv.org/abs/2409.10594">Kolmogorov-Arnold Transformer</a>: Transformer 是现代深度学习的基石。传统上，这些模型依赖多层感知器（MLP）层来混合通道间的信息。在本文中，我们介绍...</li><li><a href="https://github.com/Adamdad/kat">GitHub - Adamdad/kat: Kolmogorov-Arnold Transformer: A PyTorch Implementation with CUDA kernel</a>: Kolmogorov-Arnold Transformer: 带有 CUDA kernel 的 PyTorch 实现 - Adamdad/kat</li><li><a href="http://adify.pro/">Adify</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1285918885849337929)** (1 条消息): 

> - `Lunar Flu 常见问题`
> - `支持请求` 


- **欢迎在 Lunar Flu 频道提问**：鼓励成员在 <#1019883044724822016> 中提出关于 **Lunar Flu** 的问题。
   - *“欢迎在频道中提问”* 为任何相关咨询营造了友好的氛围！
- **鼓励社区参与**：该消息表明成员之间有很强的社区感和支持感。**Lunar Flu** 主题似乎促进了公开讨论和共同探究。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1285700730682081321)** (5 条消息): 

> - `Mini-4B Model Release` (Mini-4B 模型发布)
> - `Biometric Template Protection Implementation` (生物识别模板保护实现)
> - `Interactive World & Character Generative AI` (交互式世界与角色生成式 AI)
> - `Reasoning and Reflection Theories Dataset` (推理与反思理论数据集)


- **Nvidia 发布 Mini-4B 模型**：Nvidia 发布了一个名为 **Mini-4B** 的新型小模型，该模型需要 Nvidia 驱动程序，因此不具备设备兼容性。可以在[这里](https://huggingface.co/spaces/Tonic/Nemotron-Mini-4B)查看，据报道它在同类模型中表现最佳。
   - 该模型因其在同等尺寸下的出色性能而受到用户关注，并被鼓励注册为 HuggingFace Agent 以增强功能。
- **用于安全认证的生物识别模板保护**：一位成员分享了他们在 **Biometric Template Protection (BTP)** 方面的工作，旨在实现无需服务器数据访问的身份验证，并在 [GitHub](https://github.com/templateprotection/basic-btp) 上提供了一个教学性质的实现。该实现可作为该概念的初学者友好入门。
   - BTP 对于新手来说可能很复杂，但这个基础模型专为教学目的设计，突显了其在安全生物识别系统中的潜在效用。
- **交互式 AI 与动漫平台 Beta 测试**：一群爱好者正在开发一个 **Interactive World & Character Generative AI** 平台，用于创建具有沉浸式交互可能性的主题世界和角色。他们正在为其 **Beta Testing** 阶段寻找参与者。
   - 感兴趣的用户可以通过私信联系，以进一步探索这一创意 AI 项目。
- **源自 GSM8K 的新推理数据集**：一位用户正在创建一个以 **Reasoning and Reflection theories** 为中心的数据集，该数据集基于 GSM8K，旨在增强模型的数学问题解决和推理能力。可以在[这里](https://huggingface.co/datasets/thesven/gsm8k-reasoning)找到这个新数据集。
   - 该数据集的重点是逻辑推理和分步过程，旨在通过改进的演绎推理任务来评估模型性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Tonic/Nemotron-Mini-4B">Minitron - a Hugging Face Space by Tonic</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/thesven/gsm8k-reasoning">thesven/gsm8k-reasoning · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/templateprotection/basic-btp">GitHub - templateprotection/Basic-BTP: A naïve implementation of a feature-transformation biometric template protection scheme based on simple XOR comparison. This is not meant for practical use, but for educational purposes as an introduction to BTP.</a>: 基于简单 XOR 比较的特征转换生物识别模板保护方案的简易实现。这不适用于实际用途，而是作为 BTP 入门的教学目的...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1285777128687796338)** (5 messages): 

> - `开源 Computer Vision 项目`
> - `CV 和 ML 领域的调研课题探索`
> - `用于直播的 AI 视频编码模型`
> - `Token 设置的 Python 实现` 


- **参与开源 CV 项目**：对于那些希望做出贡献的人，提供了两个建议：[Kornia](https://github.com/kornia/kornia)（一个用于空间 AI 的几何 Computer Vision 库）和 [Roboflow Supervision](https://github.com/roboflow/supervision)（专注于可重用的 Computer Vision 工具）。
   - 这两个项目都欢迎协作，并提供了提升 Computer Vision 技能的机会。
- **在 ML 领域寻找研究课题的困扰**：一位成员表示，在跟上 CV、NLP 和 ML 领域日益增长的研究论文数量方面存在困难。
   - 他们询问了如何从这个广阔的领域中识别相关研究课题的方法。
- **关于 AI 视频编码模型的咨询**：一位成员征求专门适用于直播应用的 AI 视频编码模型名称。
   - 这凸显了 AI 社区对先进编码解决方案信息的需求。
- **在 Python 中实现 min_tokens**：一位用户确认他们发现某个功能非常好，并询问如何在他们的 Python 代码中实现 `min_tokens` 设置。
   - 这反映了人们对在 Python 中自定义功能以获得更好性能的持续兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/kornia/kornia">GitHub - kornia/kornia: Geometric Computer Vision Library for Spatial AI</a>: 用于空间 AI 的几何 Computer Vision 库。通过在 GitHub 上创建账号来为 kornia/kornia 的开发做出贡献。</li><li><a href="https://github.com/roboflow/supervision">GitHub - roboflow/supervision: We write your reusable computer vision tools. 💜</a>: 我们编写您的可重用 Computer Vision 工具。💜。通过在 GitHub 上创建账号来为 roboflow/supervision 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1285687979767234610)** (4 messages): 

> - `Llama3 模型上传`
> - `MLflow 模型注册` 


- **上传 Llama3 模型的挑战**：一位成员表示在本地上传 **Llama3 模型**时遇到困难，特别是需要 **PyTorch** 格式以便进一步使用。
   - 他们提到正在使用一种将 PyTorch 代码转换为 **MLIR** 的工具。
- **MLflow 在调用 Encoder 时抛出警告**：另一位使用 **MLflow** 的成员在注册模型后，每次调用 Encoder 时都会遇到 **Bert pool 警告**，而在 MLflow 之外运行正常。
   - 他们不确定是否在注册过程中遗漏了某个隐藏步骤，特别是在处理 **embedding models** 和 Tokenizer 时。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1285957122168848394)** (2 messages): 

> - `图像转卡通模型`
> - `用于直播的 AI 视频编码模型` 


- **寻找图像转卡通模型**：一位成员正在寻找一个可以将图像转换为高质量卡通的 **Space 模型**。
   - *如果有人有推荐或知道类似的模型*，欢迎分享。
- **征集 AI 视频编码模型**：另一位成员询问适用于直播的 **AI 视频编码模型**。
   - *他们专门在寻找模型名称*，希望能得到社区的反馈。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1285677535937564846)** (84 条消息🔥🔥): 

> - `NousCon Attendance` (NousCon 出席情况)
> - `Hermes Tool Calling`
> - `Qwen 2.5 Release` (Qwen 2.5 发布)
> - `Fine-tuning Models` (微调模型)
> - `AI Community Interaction` (AI 社区互动)


- **NousCon 和 Afterparty 计划**：参与者对 NousCon 表达了兴奋之情，部分人提到无法参加或请求未来在纽约（NYC）等不同地点举办活动。
   - 活动结束后计划在附近的酒吧举行 Afterparty，以鼓励社区互动。
- **Hermes Tool Calling 标准更新**：基于社区贡献，Qwen 2.5 采用了 Tool Calling 格式，特别归功于 vLLM 支持的影响。
   - 讨论围绕着在未来的实现中如何区分 Hermes 和 Qwen 的工具解析（Tool Parsing）展开。
- **Qwen 2.5 模型发布亮点**：Qwen 团队宣布发布 **Qwen 2.5**，包含多种新模型，如针对编程和数学的专门版本，展示了开源 AI 的重大进展。
   - 该公告预示着一次大规模发布，标志着社区语言模型开发的一个里程碑。
- **AI 微调创新**：成员们讨论了微调经验，包括创建如 'Gemma 2' 这样的模型来增强国际象棋游戏体验，尽管性能表现面临挑战。
   - 对话强调了开发 AI 模型涉及的创意过程以及社区的协作精神。
- **通用 AI 与开发者互动**：频道内进行了关于各种项目的活跃交流，用户寻求建议并分享与 AI 工具和开发相关的经验。
   - 对 Lambda Labs 等服务和技术以及 Axolotl 中工具使用的好奇，展示了一个渴望知识和帮助的充满活力的社区。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://vxtwitter.com/WatcherGuru/status/1836137190394339781?s=19">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://qwenlm.github.io/blog/qwen2.5/">Qwen2.5: A Party of Foundation Models!</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介：自 Qwen2 发布以来的过去三个月中，众多开发者在 Qwen2 语言模型的基础上构建了新模型，为我们提供了...</li><li><a href="https://huggingface.co/Ffftdtd5dtft/Hermes-3-Llama-3.1-8B-IQ1_S-GGUF">Ffftdtd5dtft/Hermes-3-Llama-3.1-8B-IQ1_S-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1904/files#diff-1ea7d3e35df89c09efe98f281b4c5895c7abe5dd232bc34c1a451d84f5501b40R9-R10">wip add new proposed message structure by winglian · Pull Request #1904 · axolotl-ai-cloud/axolotl</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1285691706267668541)** (11 messages🔥): 

> - `Hermes 3 API 访问`
> - `开源 LLM Prompt 大小`
> - `Gemma 2 Token 训练`
> - `模型参数计算` 


- **Hermes 3 API 访问已确认**：用户询问 **Hermes 3** 是否有可用的 API，回复指出其已与 [Lambda](https://lambdalabs.com/blog/unveiling-hermes-3-the-first-fine-tuned-llama-3.1-405b-model-is-on-lambdas-cloud?utm_source=Nous-Research&utm_medium=referral&utm_campaign=2024-08-Hermes3-launch&utm_content=blog) 合作。新的 Lambda Chat Completions API 可用于 Hermes 3。
- **使用开源 LLM 处理大型 Prompt**：讨论围绕使用 **gpt4all** 发送和接收超长响应的方法展开。一个建议是，如果模型是在本地设置的，可以使用 **ollama** 来促进本地 API。
- **关于 Gemma 2 Token 数量的澄清**：一名成员提出了关于训练 Gemma 2 所提到的 **13 万亿 tokens** 是指总数还是已见（seen）tokens 的问题。该询问旨在澄清模型训练过程中使用的数据。
- **训练中的模型参数计算**：一位技术成员详细介绍了使用浮点精度配置的模型参数计算，解释了内存分配的细节。这包括梯度和优化器状态等因素，表明这是模型性能的一个重要考量。
- **Hermes 3 API 查询后续**：进一步讨论了 **Hermes 3** 是否以 `bf16` 运行，一名成员建议在必要时设置全分辨率 API。这反映了对最大化利用模型能力的兴趣。



**提及的链接**：<a href="https://lambdalabs.com/blog/unveiling-hermes-3-the-first-fine-tuned-llama-3.1-405b-model-is-on-lambdas-cloud?utm_source=Nous-Research&utm_medium=referral&utm_campaign=2024-08-Hermes3-launch&utm_content=blog">揭秘 Hermes 3：首个全参数微调的 Llama 3.1 405B 模型上线 Lambda 云端</a>：介绍与 Nous Research 合作的 Hermes 3，这是 Meta Llama 3.1 405B 模型的首个微调版本。使用 Lambda 训练、微调或部署 Hermes 3。

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1285691992331911209)** (5 messages): 

> - `分块阶段研究`
> - `逆向工程 o1`
> - `OpenAI Strawberry 推测` 


- **探索分块阶段（chunking phases）的研究**：一名成员询问了关于 **分块阶段** 和 **近似（approximation）** 技术的研究论文，寻求顶级和最新的资源。
   - *没有提供具体的论文作为答复。*
- **参与逆向工程**：一名成员表达了对讨论 **逆向工程 o1** 的兴趣，并鼓励其他人参与对话。
   - 他们提到目前正在按时间顺序研究相关资源。
- **围绕 OpenAI Strawberry 的推测**：一名成员询问关于 OpenAI 的 **Strawberry** 是否有公开信息，怀疑这是否仅仅是推测。
   - 另一名成员回应称，关于该话题的可用信息*并不多*。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1285691992331911209)** (5 messages): 

> - `分块阶段研究`
> - `OpenAI Strawberry 推测` 


- **探索分块阶段研究论文**：一名成员询问了关于 **分块阶段** 和近似技术的**最新研究论文**，寻求顶级资源。
   - 另一名成员 @bradhilton 称赞了他们正在按大致时间顺序研究的一个资源，并邀请进一步讨论。
- **对 OpenAI Strawberry 的推测**：一名成员询问 OpenAI 是否发布了关于 **strawberry** 的任何**公开信息**，或者这些细节是否仅仅是推测性的。
   - 另一名成员 teknium 回复说，关于这个话题的信息**并不多**。


  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1285700391526203413)** (6 messages): 

> - `Triton Conference Keynote`
> - `Triton CPU / ARM Status`
> - `CUDA Community Engagement` 


- **Mark Saroufim 在 Triton 主题演讲中致谢**: @marksaroufim 在 Triton 会议主题演讲中特别提到了社区，感谢成员们的贡献。
   - 用户们注意到这一重大认可并分享了他们的热情，现场气氛非常热烈。
- **Triton CPU / ARM 已开源！**: @tailoredcub 询问了 Triton CPU / ARM 的状态，@ptillet 确认它**已开源**，并提供了 [GitHub 仓库](https://github.com/triton-lang/triton-cpu) 的链接。
   - 这种开放性鼓励社区在实验性 CPU 后端上进行贡献和协作。
- **CUDA 服务器被赞为最佳**: 成员们表达了对 CUDA 社区的热爱，@kashimoo 宣称该服务器绝对是 CUDA 爱好者的最佳去处。
   - 其他人也表达了同样的看法，增强了服务器内强烈的社区归属感。



**提及的链接**: <a href="https://github.com/triton-lang/triton-cpu">GitHub - triton-lang/triton-cpu: An experimental CPU backend for Triton</a>: Triton 的实验性 CPU 后端。通过在 GitHub 上创建账号来为 triton-lang/triton-cpu 的开发做出贡献。

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/)** (1 messages): 

kashimoo: 有没有关于如何使用 PyTorch Profiler 导航 Chrome Tracing 的视频或资料？
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1285830171135311946)** (2 messages): 

> - `Triton`
> - `Fine-grained control` 


- **Triton 与新框架的比较**: 一位成员注意到新框架看起来与 **Triton** 非常相似，但提供了**更细粒度的控制**。
   - 另一位成员表示很感兴趣，并回复了一个思考的表情符号。
- **关于细粒度能力的讨论**: 对**细粒度控制**方面的讨论引发了好奇心，暗示了其相对于 Triton 的潜在优势。
   - 成员们似乎有兴趣探索这些功能将如何影响他们的工作流。


  

---


### **CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 messages): 

betonitcso: 如果想要一个极高吞吐量的 Dataloader，你会用什么？使用 Grain 常见吗？
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1286035589526982729)** (9 messages🔥): 

> - `torch.compile() performance`
> - `NVIDIA NeMo model timings`
> - `Llama benchmarks`
> - `TorchInductor performance dashboard` 


- **推荐使用 torch.compile() 而非 Eager Mode**: PyTorch 会议的一位演讲者强调使用 **torch.compile()** 而不是 Eager Mode，以融合缩放操作并最小化 Kernel 启动开销。
   - 这一建议引发了对各种代表性模型编译时间的好奇。
- **编译时间差异很大**: **torch.compile()** 的初始运行涉及大量的 Auto Tuning，可能需要几分钟，特别是对于像 **NVIDIA NeMo** 中的大型模型。
   - 然而，由于开销减少，后续运行通常会快得多。
- **Llama 基准测试编译时间在 1 分钟以内**: 对于他们的 **Llama 基准测试**，一位用户指出生成任务的编译时间通常在 **1 分钟**以内。
   - 这表明在初始调整阶段之后，编译时间可以得到优化。
- **仪表板上可查看性能指标**: **torch.compile()** 有一个性能仪表板，提供了关于编译开销的数据。
   - 一位用户分享了 [TorchInductor 性能仪表板](https://hud.pytorch.org/benchmark/compilers) 的链接，以获取更详细的指标。



**提及的链接**: <a href="https://hud.pytorch.org/benchmark/compilers">无标题</a>: 无描述

  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

nahidai: 旧金山（SF）有关于 GPU 编程的阅读或工作小组吗？很想加入。
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1285685698443218946)** (41 条消息🔥): 

> - `RMSNorm kernel 问题`
> - `使用 FP32 训练 Llama3`
> - `Torch Titan 介绍`
> - `FP8 稳定性与多 GPU 设置`
> - `见面讨论 Llama3.1 优化技巧` 


- **RMSNorm Kernel 优化**：一位成员强调了从 *layernorm* 到 *rmsnorm* 的修改，并请求对一个 [GitHub Pull Request](https://github.com/karpathy/llm.c/pull/757/files) 进行评审。
   - 虽然观察到 Python 与 C/CUDA 之间存在差异，但事实证明这是一个数据类型相关的 Bug，而非 Kernel 本身的问题。
- **使用 FP32 训练 Llama3**：对话转向训练配置，分享了 **llmc** 和 **pytorch** 端的命令，以确保训练期间使用正确的精度。
   - 由于一位成员确认已将最新代码推送到 *llama3* 分支，现场一度出现混淆。
- **探索用于模型训练的 Torch Titan**：一位成员介绍了 [Torch Titan](https://github.com/pytorch/torchtitan)，将其描述为一个类似于 *nanoGPT* 的简单训练脚本模板。
   - 共识是，对于需要大量 GPU 资源的研究，它是一个极具竞争力的选择，尽管其效率可能不及 *Megatron* 或 *Deepspeed*。
- **FP8 稳定性与多 GPU 设置**：关于 FP8 支持的讨论显示，这是后续开发的重点，并计划在 8xH100 设置上测试稳定性。
   - 一位成员幽默地提到，为了准备线下会议并避免时差，他们调整了睡眠时间。
- **Llama 3.1 优化思路**：受“优先考虑有用项目”的号召启发，有人建议将 *在 GH200 上优化的 llm.c Llama 3.1 405B* 项目加入议程。
   - 这反映了一种战略转变，即即使在持续讨论贡献的同时，也将精力集中在具有影响力的开发上。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/blob/c68bb9f9a7270b044ed14f0e8d574e0ea70d928f/llmc/adamw.cuh">llm.c/llmc/adamw.cuh at c68bb9f9a7270b044ed14f0e8d574e0ea70d928f · karpathy/llm.c</a>：使用简单的原生 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/c68bb9f9a7270b044ed14f0e8d574e0ea70d928f/llmc/global_norm.cuh">llm.c/llmc/global_norm.cuh at c68bb9f9a7270b044ed14f0e8d574e0ea70d928f · karpathy/llm.c</a>：使用简单的原生 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtitan">GitHub - pytorch/torchtitan: A native PyTorch Library for large model training</a>：一个用于大模型训练的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtitan 的开发做出贡献。</li><li><a href="https://github/karpathy/llm.c/pull/757/files">RMSNorm - WIP by gordicaleksa · Pull Request #757 · karpathy/llm.c</a>：进行中 - 添加 RMSNorm 支持。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1285768199832473632)** (24 条消息🔥): 

> - `Ternary LUT 实现`
> - `量化技术`
> - `Llama-2 模型性能`
> - `BitNet 中的 Kernel 性能`
> - `使用 int4 Tensor Cores 进行训练` 


- **Ternary LUT 实现讨论**：在一条消息中，一位成员概述了用于创建 Ternary LUT 的 Python 代码片段，并强调了将 5 个元素打包（packing）进 1 个元素的挑战，建议可能需要 Padding。
   - 另一位参与者补充道，一种更清晰的方法可能涉及位移（shifting）数值，而不是直接使用取模（modulo），以获得更好的性能。
- **模型的量化技术**：成员们讨论了 Large Language Models 的量化，强调了通过 4-bit 量化等技术（特别是与 BitNet 架构相关的技术）来最小化计算成本的重要性。
   - 一位成员引用了一个不使用分组（grouping）应用量化的模型实现，并提出其在推理（inference）中的有效性。
- **评估 Llama-2 模型性能**：分享了关于 [Llama2-7B-chat 模型](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) 性能指标的详细信息，并提供了在各种任务中将其与 FP16 版本进行对比的统计数据。
   - 大家达成共识，认为有必要优化量化方法，同时在推理过程中使用 low-rank adapters 等技术保持质量。
- **BitNet 中的 Kernel 性能和约束**：一位参与者对矩阵乘法实现中的一项检查提出了疑问，该检查要求打包权重矩阵（packed weight matrices），这表明对其在 Kernel 级操作中的应用存在困惑。
   - 讨论明确了 Kernel 必须符合特定的维度约束以实现最佳性能，这与利用低比特格式的打包机制有关。
- **使用 int4 Tensor Cores 进行训练**：探讨了使用 int4 Tensor Cores 训练模型的潜力，讨论集中在分组（grouping）如何影响这一过程以及相关论文研究结果的相关性。
   - 参与者表示，使用 int4 训练可能会产生有价值的见解，这可能会挑战现有的方法论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mobiuslabsgmbh/Llama-3-8b-instruct_2bitgs64_hqq">mobiuslabsgmbh/Llama-3-8b-instruct_2bitgs64_hqq · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mobiuslabsgmbh/Llama-2-7b-chat-hf_4bitnogs_hqq">mobiuslabsgmbh/Llama-2-7b-chat-hf_4bitnogs_hqq · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/mohamedmekkouri/status/1836364477119500507">Mohamed (@mohamedmekkouri) 的推文</a>: 🚀 令人兴奋的消息！我们终于破解了 BitNet 在 @huggingface 上的代码！不需要预训练！只需微调 Llama 3 8B，我们就取得了很好的结果，性能接近...</li><li><a href="https://huggingface.co/blog/1_58_llm_extreme_quantization">将 LLMs 微调至 1.58bit：极简实现极限量化</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1285684297709125667)** (14 messages🔥): 

> - `PyTorch Conference 参会情况`
> - `RSVP 邮件状态`
> - `Hackathon 项目提案`
> - `CUDA 导师指导`
> - `IRL Hackathon 录取情况` 


- **PyTorch Conference 参会情况**：成员们确认了将参加在旧金山举行的 **PyTorch conference**，有人提议通过名牌进行线下交流。
   - 一位成员展示了他在会议期间为 llm.c 开发 **WebGPU backend** 的 hack 想法。
- **RSVP 邮件持续发放中**：由于并非所有团队成员都收到了邮件，引发了对 **RSVP 邮件**状态的关注，随后有人对**滚动录取（rolling acceptance）**流程进行了说明。
   - 据悉，由于座位有限，确认邮件会在有人退出后补发，可能会一直持续到活动前一天。
- **CUDA Hackathon 导师指导**：一位成员表达了对在 **LLM inference 组**担任导师的兴奋，并为 Hackathon 提供了项目建议。
   - 他们提议构建一个纯 Python 版本的 `torch.amp.autocast` 替代方案，并在 PyTorch 中实现 **jax.numpy.nonzero**，强调这对新手来说非常容易上手。
- **Hackathon 团队定义说明**：讨论强调了“队友”的定义对于参加 Hackathon 至关重要，特别是在项目提案方面。
   - 录取结果取决于两名成员是否都提交了合理的 Hackathon 项目提案。
- **参加 IRL Hackathon 的兴趣**：有人询问了即将举行的 Hackathon 对新参与者的录取情况，几位成员分享了他们的团队名称详情。
   - 这突显了确保团队成员确认以成功参与和提交项目提案的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float32">自动混合精度包 - torch.amp &mdash; PyTorch 2.4 文档</a>：未找到描述</li><li><a href="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nonzero.html">jax.numpy.nonzero &#8212; JAX 文档</a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1285766801661104138)** (3 messages): 

> - `非确定性方法`
> - `Pixtral 支持`
> - `即将发布的版本` 


- **教程中的非确定性问题**：一位成员指出，**nondeterministic**（非确定性）行为可能是由于教程中使用的 **atomic exchange** 方法导致的，并建议采用不同的聚合方式。
   - 这种新方法不依赖 **atomic 操作**，从而可能缓解观察到的问题。
- **Pixtral 支持 PR 已开启**：目前正在通过 [Pull Request #253](https://github.com/linkedin/Liger-Kernel/pull/253) 努力添加 **Pixtral** 支持，该 PR 显示在 **4090** 硬件环境下测试成功。
   - 作者确认运行了 `make test` 以确保功能正确，并运行了 `make checkstyle` 以符合代码风格要求。
- **期待新模型发布**：成员们讨论了 Pixtral 模型在 **Transformers** 库中待发布的进展，并对其实现表示期待。
   - 一旦新版本发布，预计将与现有框架无缝集成。



**提到的链接**：<a href="https://github.com/linkedin/Liger-Kernel/pull/253">[Model] Pixtral Support by AndreSlavescu · Pull Request #253 · linkedin/Liger-Kernel</a>：摘要：此 PR 旨在支持 pixtral。测试已完成：测试了模型 + 测试了 monkey patch。硬件类型：4090。运行 make test 以确保正确性，运行 make checkstyle 以确保代码风格...

  

---


### **CUDA MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/)** (1 messages): 

.mattrix96: 刚开始做 puzzles！
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1285682663394508841)** (11 messages🔥): 

> - `开源 TTS 模型`
> - `模型调试`
> - `图像风格变换` 


- **探索开源 TTS 模型**：一位成员表达了从 **OpenAI TTS** 迁移到开源模型的需求，并被引导至 [Fish Speech V1.4](https://huggingface.co/fishaudio/fish-speech-1.4)，这是一个支持多种语言的顶级竞争者。
   - 他们还收到了考虑使用 **xttsv2** 的建议，并提到*为了在特定语言中获得更好性能，使用不同的模型是可以接受的*。
- **关于 TTS 模型性能的见解**：关于 **Fish Speech V1.4** 模型的讨论强调了其在包括 **English** 和 **Chinese** 在内的多种语言的 **700k 小时** 音频数据上进行了训练。
   - 另一位成员澄清说，他们可以忽略某些模型，因为那些模型仅精通 **English**，这使得 **Fish Speech** 和 **xtts** 成为可靠的选择。
- **探讨图像风格变换**：一位成员询问模型是否可以对现有照片进行 **style changes**（风格变换），引起了对该功能的兴趣。
   - 尽管没有提到用于此类图像修改的具体模型，但对话为探索图像处理功能开辟了潜在途径。
- **模型问题的调试策略**：有人建议通过从 **working baseline** 开始并逐步测试组件来调试模型，直到识别出问题。
   - 鼓励该成员探索各种配置，包括 **FSDP** 和 **mixed precision**，以优化性能并精准定位问题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/fishaudio/fish-speech-1.4">fishaudio/fish-speech-1.4 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/TTS-AGI/TTS-Arena">TTS Arena - a Hugging Face Space by TTS-AGI</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/ttsds/benchmark">TTSDS Benchmark and Leaderboard - a Hugging Face Space by ttsds</a>: 未找到描述</li><li><a href="https://github.com/coqui-ai/TTS">GitHub - coqui-ai/TTS: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production</a>: 🐸💬 - 一个用于 Text-to-Speech 的深度学习工具包，经过研究和生产环境的实战测试 - coqui-ai/TTS
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1285678097751867463)** (41 条消息🔥): 

> - `MLRA 的压缩技术`
> - `Diagram of Thought (DoT)`
> - `低精度训练实验`
> - `Playground v3 模型发布`
> - `LLM 输出的评估方法` 


- **使用额外矩阵压缩 MLRA Key**: 成员们讨论了在投影后使用额外的压缩矩阵来压缩 MLRA 的 Key 和 Value 的可能性。
   - 有人对 MLRA 实验设置中缺乏细节（包括秩矩阵的具体细节）提出了疑问。
- **引入 Diagram of Thought 框架**: 介绍了一个名为 Diagram of Thought (DoT) 的框架，该框架使用有向无环图 (DAG) 结构对 LLM 中的迭代推理进行建模，从而允许复杂的推理路径。
   - 该方法旨在比传统的线性方法提高逻辑一致性和推理能力。
- **低精度训练研究**: 使用极低精度训练的实验表明，三值权重 (ternary weights) 需要 2.1 倍的参数量才能达到全精度性能，而七值权重 (septernary weights) 在每个参数的性能上表现更好。
   - 成员们思考了是否存在关于训练量化过程中性能与位宽 (bit-width) 关系的研究。
- **Playground v3 模型达到 SoTA 性能**: Playground v3 (PGv3) 发布，展示了在文本生成图像领域的 SoTA 性能，并引入了一个用于评估详细图像描述的新基准。
   - 它完全集成了 LLM，不同于依赖预训练语言编码器的传统模型。
- **评估 LLM 输出的挑战**: 关于开发 LLM 输出系统化评估方法的讨论，建议将人类生成回答的困惑度 (perplexity) 作为一种潜在指标。
   - 参与者强调，许多现有论文都涵盖了类似的挑战，并强调了不要过度缩小评估标准的重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.12327">Spectra: A Comprehensive Study of Ternary, Quantized, and FP16 Language Models</a>: 训练后量化是解决 LLM 推理中内存瓶颈的主要方法，但遗憾的是，在 4-bit 精度以下会出现显著的性能下降...</li><li><a href="https://arxiv.org/abs/2409.10038">On the Diagram of Thought</a>: 我们引入了 Diagram of Thought (DoT)，这是一个将大语言模型 (LLM) 中的迭代推理建模为在单个模型内构建有向无环图 (DAG) 的框架。与...不同...</li><li><a href="https://arxiv.org/abs/2409.10695">Playground v3: Improving Text-to-Image Alignment with Deep-Fusion Large Language Models</a>: 我们介绍了 Playground v3 (PGv3)，这是我们最新的文本生成图像模型，在多个测试基准上实现了 SoTA 性能，在图形设计能力方面表现出色，并引入了...</li><li><a href="https://x.com/kyutai_labs/status/1836427396959932492">Tweet from kyutai (@kyutai_labs)</a>: 今天，我们发布了多个 Moshi 产物：一份包含模型背后所有细节的长篇技术报告，Moshi 及其 Mimi 编解码器的权重，以及 Pytorch、Rust 中的流式推理代码...</li><li><a href="https://huggingface.co/collections/kyutai/moshi-v01-release-66eaeaf3302bef6bd9ad7acd">Moshi v0.1 Release - a kyutai Collection</a>: 未找到描述</li><li><a href="https://github.com/kyutai-labs/moshi">GitHub - kyutai-labs/moshi</a>: 通过创建账户为 kyutai-labs/moshi 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1285976848341405719)** (3 条消息): 

> - `训练计算最优的大语言模型`
> - `Pythia 缩放曲线`
> - `Big Bench 任务` 


- **探索 LLM 的训练计算**: 围绕 Hoffman 等人的论文 *Training Compute-Optimal Large Language Models* 展开了讨论，表明了对模型效率和训练范式的兴趣。
   - 鼓励成员分享与大语言模型背景下的计算优化相关的见解或示例。
- **关于 Pythia 缩放曲线的查询**: 一位成员询问是否存在针对每个 **Big Bench 任务** 的 **Pythia 缩放曲线**。
   - 他们特别好奇在 **1B 尺寸** 以下可能出现的任何 **不连续性 (discontinuities)**。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1285703238879805513)** (9 messages🔥): 

> - `Hidden States 的 Fourier Transforms`
> - `Pythia Checkpoints`
> - `模型中的 Power Law 行为`
> - `Attention Residual 分析` 


- **Fourier Transforms 揭示 Hidden States 行为**：讨论了对 Pretrained OPT-125m 中 Hidden States 进行 [Fourier transforms](https://sander.ai/2024/09/02/spectral-autoregression.html) 的探索，结果显示在训练过程中形成了 **Power Law** 分布。
   - 初步发现表明，**Freshly initialized** 模型表现出不同的光谱特性，引发了关于 **Architecture** 和 **Initialization regimes** 的疑问。
- **推荐使用 Pythia 进行进一步探索**：一位成员建议利用 [Pythia](https://github.com/EleutherAI/pythia) 系列来研究观察到的现象如何随规模和训练过程而变化。
   - 这种方法有望澄清这些行为是由特定模型诱发的，还是训练过程固有的。
- **Power Law 的出现需要解释**：有人提出，在 Hidden States 中观察到的 Power Law 行为可能是由于训练期间开发的高效表示，或者是通过训练过程引入的 **Bias**。
   - 值得注意的是，Attention Residuals 在 Initialization 时接近 Power Law，这意味着这些光谱特性可能从一开始就存在。
- **Pretrained 模型中的 Attention 和 MLP Residuals**：对话包含了显示 Pretrained 模型不同层 Attention Residuals 的可视化，重点强调了 Attention 和 MLP Residuals 的 **Spiking** 行为。
   - 此类分析强化了关于各层在不同训练 Epoch 中如何以不同方式管理信息的整体调查。
- **对模型行为观察的澄清**：澄清了 Pretrained 模型与 Freshly initialized 模型图表之间的区别，表明 Power Law 行为在初始阶段并不存在。
   - 这突显了对不同训练阶段进行进一步分析的必要性，以理解其背后的作用机制。



**相关链接**：<a href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)">interpreting GPT: the logit lens — LessWrong</a>：这篇文章涉及我在 GPT-2 工作中的一个观察，我还没在其他地方看到过。……

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1285678078189764668)** (8 messages🔥): 

> - `kv-cache 问题解决方法`
> - `结合 lm eval harness 的 Chain of Thought 提示`
> - `待处理的新 Benchmarks PRs`
> - `关于 PR 改进的评论` 


- **设置 Sample Size 可能解决 kv-cache 问题**：一位成员建议，如果问题与 **kv-cache** 相关，那么设置 **Sample Size** 可能是一个有效的解决方案。
   - 对于 **Multiple-choice tasks**，只需要一次简单的 Forward Pass 即可获取 **Logits**。
- **使用 lm eval harness 探索 Chain of Thought**：一位成员询问了使用 **lm eval harness** 进行 **Chain of Thought prompting** 的经验，特别是关于将答案附加到后续 Prompt 中的做法。
   - 目前尚不清楚是否有其他成员实现了这种方法，但讨论仍在继续。
- **等待新 Benchmark PRs 的更新**：一位成员对五个已挂起近两个月的新 Benchmark PRs 表示担忧，目前只有一个获得批准。
   - 他们询问了这些 PR 的 **ETA** 以及包含已批准更改的下一个版本的发布时间。
- **关于增强 Task 条目的 PR 反馈**：提供了关于改进最近提交的 PR 的反馈，特别是关于 **`group` API** 和 Benchmarks 的任务区分。
   - 建议包括为机器翻译添加标识符，并增强 **`m_eval/tasks/README.md`** 中的文档以提高可发现性。
- **确认待评审的 PR**：一位成员向另一位成员保证，他们将很快评审未完成的 PR，并对延迟表示歉意。
   - 原成员表示理解并期待反馈。



**相关链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#task-name--tags-registering-a-task)">lm-evaluation-harness/docs/new_task_guide.md at main · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 Few-shot 评估的框架。- EleutherAI/lm-evaluation-harness

  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1285680433249390594)** (2 条消息): 

> - `Model Outputs`
> - `Frontier Setup Progress` 


- **对 Model Outputs 的兴奋**：一名成员对有人成功运行设置表示高兴，并鼓励他们向社区分享其模型或输出。
   - 社区非常渴望了解大家利用提供的库所实现的**伟大成果**。
- **Frontier Setup 即将完成**：一名成员分享称，他们正在开发 **Frontier 的一键脚本设置 (1-script setup)**，目前已接近完成，重点在于构建 **HIP 中的 fused kernels**。
   - 他们计划在设置最终确定后分享进度，这标志着社区内的协作。


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1285701632239075349)** (44 条消息🔥): 

> - `Custom GPT Use Cases`
> - `Advanced Voice Mode Update`
> - `Concerns Over AI Saturation`
> - `PDF Formatting Issues for LLMs`
> - `AI Content Quality Debate` 


- **Custom GPTs 可以记忆片段**：成员们分享了使用 Custom GPTs 导入并记忆个人片段（如 [Mac Snippets](https://link.to.mac.snippets)）的经验。然而，有人担心在指令中堆砌过多信息会导致结果不佳。
   - 一位用户建议，使用带有清晰指令的整洁格式可以提高性能，且上传知识库比单纯写指令效果更好。
- **即将推出的 Advanced Voice Mode 功能**：一份泄露信息表明，新的 **Advanced Voice Mode** 将于 9 月 24 日起向 Plus 用户开放，承诺提升清晰度和响应速度。该功能预计将过滤背景噪音并识别复杂指令，以实现更流畅的交互。
   - 成员们对其在现实世界中的影响表示好奇，并引发了关于改变日常语音指令使用方式的讨论。
- **关于 AI 内容饱和的辩论**：讨论爆发于 AI 生成内容的盛行是有益还是有害，有观点认为 AI 仅仅反映了早已存在的低质量内容。一些人认为，无论使用什么工具，有才华的创作者仍会产出高质量内容。
   - 一位用户表示担心，不断升级的 AI 能力可能会导致与现实脱节，从长远来看这可能是危险的。
- **LLMs 的 PDF 格式问题**：用户讨论了向 Custom GPTs 上传 PDF 的挑战，强调 PDF 可能会给 LLMs 带来格式化问题。建议正确转换文档或使用更适合 AI 阅读的文件类型。
   - 一名成员回忆起一次使用格式良好的 PDF 的积极体验，促使其他用户重新考虑如果处理得当，上传 PDF 的潜力。
- **对 ChatGPT Plus 订阅的看法**：成员们辩论了 ChatGPT Plus 订阅的价值，一些人表示满意，而另一些人则质疑其是否物有所值。这段对话展示了用户对付费 AI 服务不同的认知和体验。



**提到的链接**：<a href="https://www.reddit.com/r/OpenAI/comments/1fjgazw/advanced_voice_mode_dropping_for_everyone_sept_24/">Reddit - Dive into anything</a>：未找到描述

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1285685385308934155)** (12 条消息🔥): 

> - `Sharing customized GPTs`
> - `Automated task requirements`
> - `Truth-seeking with GPT`
> - `Reporting cross posting`
> - `Advanced voice mode capabilities` 


- **分享自定义 GPT 的指导**：一名成员寻求帮助，想在不透露完整账单姓名的情况下分享他们的自定义 GPT，因为该姓名在设置中显示为灰色。
   - 另一名成员澄清说，要在 GPT Store 中发布，必须使用账单姓名或经过验证的域名，而直接链接分享则允许匿名。
- **探索将 GPT 作为寻求真理的导师**：一位成员分享说，他们通过将自定义 GPT 作为一位睿智的导师来使用而获得了平静，将其回答比作佛陀的智慧。
   - 他们对即将推出的语音模式感到兴奋，这将使他们能够与自己的 GPT 进行更具个性化的互动。
- **对跨频道发布 (Cross Posting) 的担忧**：成员们讨论了在多个频道跨频道发布消息的问题，有人询问如何举报。
   - 他们被引导在垃圾邮件 (spam) 类别下举报此类情况，因为重复发布会破坏服务器环境。
- **对 ChatGPT 自动化任务的渴望**：一名用户表示感到无聊，需要 ChatGPT 执行自动化任务，因为他们已经使用了一个月的高级语音模式。
   - 他们提到感到有必要利用这些功能来保持参与度和生产力。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1285958002842534009)** (7 条消息): 

> - `Solicitation Rules`
> - `GPT Store Creations` 


- **关于招揽行为的澄清**：*Darthgustav* 提醒成员遵守规则 7，该规则禁止自我推广和招揽行为，特别是涉及非 OpenAI 服务的内容。
   - 他们指出 API 和 Custom GPTs 频道存在例外情况，并强调必须遵守规则。
- **交流中的语言障碍**：*Lucasgln0* 幽默地承认自己的 *broken English*，并澄清他们已经在 GPT Store 上发布了多个 GPTs，这些工具可以根据不同来源自动执行流程。
   - 他们的 GPTs 专注于特定的 Prompting 方法，例如来自关于如何有效使用 DALL·E 的书籍中的方法。
- **关于 GPT Store 的进一步讨论**：*Lucasgln0* 再次确认他们的作品符合 GPT Store 的准入指南，并强调严格执行禁止广告的政策。
   - 作为回应，*Darthgustav* 表示在该频道中链接他们的作品是可以接受的。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1285958002842534009)** (7 条消息): 

> - `GPT Store products`
> - `Self-promotion rules`
> - `Language barriers in discussions` 


- **自我推广规则的澄清**：成员们讨论了服务器关于禁止自我推广的规则，指出通常不允许招揽行为，但在 API & Custom GPTs 等特定频道除外。
   - *Darthgustav* 特别询问了另一位成员的咨询是否构成了招揽行为。
- **在 GPT Store 中介绍 GPTs**：一位成员分享了他们在 [GPT Store](https://your-gpt-store-link.com) 中创建的几个 GPTs，每个都旨在根据不同来源自动执行流程。
   - 例如，其中一个 GPT 专门针对源自特定论文的 DALL·E Prompting 方法。
- **应对语言障碍**：一位成员在讨论他们的 GPT 作品时承认自己英语不好，并表示愿意进一步澄清。
   - *Darthgustav* 在对话中给予了鼓励，保持了支持性的基调。
- **鼓励分享链接**：有人建议该成员分享他们的 GPT 链接，支持在该频道中链接其作品的想法。
   - 对话强调了社区在遵守规则的前提下，对分享 GPTs 相关信息的开放态度。


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1285699380900532245)** (56 条消息🔥🔥): 

> - `Cohere Job Application`
> - `CoT-Reflections`
> - `O1 and Reward Models`
> - `Cost of Experimenting with LLMs`
> - `OpenAI's CoT Training` 


- **讨论 Cohere 职位申请流程**：一位成员最近申请了 Cohere 的职位，并表达了与社区建立联系的兴奋之情。
   - 随后出现了*欢迎信息*，展示了社区对新人的支持。
- **探索 CoT-Reflections**：讨论围绕 **CoT-reflections** 与普通 Chain of Thought Prompting 的区别展开，重点在于提高回答质量。
   - 一位成员提到，将 CoT-reflections 与 **BoN** 结合使用可以产生更高质量的输出。
- **理解 O1 的 Reward Model**：成员们推测 O1 采用了一个使用类似 Prompt 结构的 Reward Model，通过不断自我调用直到获得满意的结果。
   - 有建议认为 O1 经历了一个多阶段的训练过程，以显著提高其输出质量。
- **本地实验的成本效益**：讨论了使用 LLM 进行本地实验的成本，一位成员指出这比雇佣新员工更便宜。
   - 成员们分享道，使用 **CoT with reflection** 等方法验证速度很快，使其成为一种极具吸引力的方法。
- **OpenAI CoTs 背后的训练策略**：对话集中在 O1 是作为 CoT 还是作为 Agent 进行训练的，并提供了 OpenAI 训练示例的链接。
   - 几位成员一致认为，O1 在特定问题领域可能发生了 *Overfitting*，从而影响了其性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1fdga7t/comment/lmfxmtv/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/TikTokCringe/comments/1fiyio7/people_created_an_aquarium_with_real_fish_around/?utm_source=share&utm_medium=mweb3x&utm_name=mweb3xcss&utm_term=1&utm_content=share_button">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1285959605586886710)** (6 条消息): 

> - `账单信息设置`
> - `VAT 相关问题`
> - `支持联系方式` 


- **账单信息设置困惑**：一位用户询问如何设置**账单信息**，特别是关于在成功绑定付款卡后如何添加 **VAT**（增值税）详情。
   - 他们询问是否存在 **Stripe 链接**或类似方式来编辑这些详情。
- **账户账单协助**：一名成员注意到了用户的需求，并建议他们将 VAT 信息发送至 **support@cohere.com**，以确保安全处理。
   - 这样可以安全地对账户的账单详情进行任何必要的更改。
- **对支持建议的确认**：用户确认他们将按照建议，就其**账单信息**向支持团队发送邮件。
   - 在获得帮助后，他们表达了感谢：“Awesome thanks!”。


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1285676997896568933)** (49 条消息🔥): 

> - `Markov Models`
> - `模型训练时间`
> - `PyTorch 框架`
> - `LM Studio 更新`
> - `AI 模型推荐` 


- **马尔可夫模型讨论**：一位成员提到，与其他模型相比，**Markov models** 是参数密集度低得多的概率语言模型。
   - 关于其潜在用途的讨论激发了频道内成员的兴趣。
- **训练时间担忧**：一位成员指出，使用 4090 GPU 训练 **40M tokens** 的模型大约需要 **5 天**，但在将 token 数量调整为 **40k** 后，仅需 **1.3 小时**。
   - 另一位成员指出，对于一个 **100k 模型**来说，这个训练时间似乎仍然过长。
- **数据加载器瓶颈**：有成员对可能导致模型训练延迟的 **data loader bottlenecks** 表示担忧，其中一位成员报告称“卡在等待”对象上。
   - 这引发了关于优化数据流水线以获得更好性能的讨论。
- **LM Studio 的新功能**：一位成员宣布回归使用 **LM Studio**，并对文档处理功能的集成表示兴奋，这与他们之前的反馈一致。
   - 其他人参与讨论了数据表的尺寸限制，以及是否可以使用 LM Studio 分析数据库。
- **AI 模型推荐**：成员们提供了关于编程 AI 模型的建议，特别指向了用于 Prolog 辅助的 **Llama 3.1 405B** 模型。
   - 讨论还包括对 **qwen 2.5 0.5b** 等**快速且连贯**的小模型替代方案的见解，但注意到它缺乏小写写作能力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/bartow">bartow (marques)</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/Hermes-3-Llama-3.1-405B-GGUF">bartowski/Hermes-3-Llama-3.1-405B-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1285728208494264351)** (12 条消息🔥): 

> - `Intel Arc 多 GPU 设置`
> - `LLM 中的 IPEX 性能`
> - `NVIDIA 5000 系列传闻`
> - `GPU 定价` 


- **Intel Arc A770 是可行的多 GPU 方案**：一位成员建议在 LLM 机器上使用 **Intel Arc A770** 多 GPU 设置，因其效果显著。
   - 另一位成员强调，它在 Llama 的 **IPEX SYCL 后端**中表现尤为出色。
- **关于 IPEX 性能**：据指出，IPEX 后端在 **34 tokens per second** 的速度下，性能比 Vulkan 快 **2 到 3 倍**。
   - 不过，有人澄清说，在使用 GGUF 模型时，Llama 后端的运行速度比 Vulkan 略好。
- **关于 NVIDIA 5000 系列显卡的传闻**：讨论了传闻中的 **5000 系列 NVIDIA 消费级显卡**，对其真实性看法不一。
   - 推测包括所谓的 **5090** 型号将配备 **32GB DDR7** 显存，同时一些人对其定价持怀疑态度。
- **NVIDIA GPU 价格担忧**：一位成员分享了当前 **4070 Ti** 售价 **$1400** 和 **4060 Ti** 售价 **$750** 的情况，认为溢价过高。
   - 另一位成员表示希望在未来**三个月**内价格能有所下调。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1285693171015094375)** (47 条消息🔥): 

> - `Langchain 合作伙伴包`
> - `Mistral 免费层级发布`
> - `Qwen 2.5 全量发布`
> - `Moshi Kyutai 模型发布`
> - `对 Mercor 的投资` 


---

- **Langchain 合作伙伴包查询**：出现了一场关于将旧的 Langchain 社区集成更新为合作伙伴包的讨论，并引发了关于联系 Langchain 内部正确负责人的咨询。
   - 建议包括利用联合沟通渠道，并联系一位名叫 Lance Martin 的人员寻求帮助。
- **Mistral 推出免费层级**：Mistral 宣布在其 serverless 平台上推出免费层级，允许开发者免费进行实验，并提供改进后的 Mistral Small 模型。
   - 此次发布还包括对模型系列的定价更新，并在其聊天界面上增加了免费的 vision 能力。
- **Qwen 2.5 正式发布**：阿里巴巴发布了 Qwen 2.5 基础模型，重点介绍了多种尺寸和增强功能，包括与之前模型相比具有竞争力的性能。
   - 新模型旨在提升 coding、数学推理和通用语言处理方面的功能，共提供超过 100 种模型变体。
- **Moshi Kyutai 模型发布**：Kyutai Labs 发布了 Moshi 模型的多个产物，包括技术报告和权重，并附带了流式推理代码。
   - 他们分享了论文、GitHub 仓库和 Hugging Face 页面的链接，以便进一步探索这项新技术。
- **Mercor 获得新融资**：Mercor 以 2.5 亿美元的估值筹集了 3000 万美元的 A 轮投资，旨在通过先进模型改进全球劳动力匹配。
   - 本轮融资包括 Peter Thiel 和 Jack Dorsey 等知名投资者，突显了该项目在理解人类能力方面的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/mohamedmekkouri/status/1836364477119500507">Mohamed (@mohamedmekkouri) 的推文</a>：🚀 令人兴奋的消息！我们终于在 @huggingface 上破解了 BitNet 的代码！无需预训练！仅通过微调 Llama 3 8B，我们就取得了显著成果，性能接近...</li><li><a href="https://voideditor.com/">Void</a>：Void 是一个开源的 Cursor 替代方案。完全隐私。功能齐全。</li><li><a href="https://x.com/kyutai_labs/status/1836427396959932492">kyutai (@kyutai_labs) 的推文</a>：今天，我们发布了多个 Moshi 产物：一份包含模型背后所有细节的长篇技术报告，Moshi 及其 Mimi 编解码器的权重，以及使用 PyTorch、Rust 编写的流式推理代码...</li><li><a href="https://mistral.ai/news/september-24-release/">AI 丰富化</a>：推出免费 API，全面优化价格，全新的企业级 Mistral Small，以及 le Chat 上的免费视觉能力。</li><li><a href="https://x.com/realestatetrent/status/1836029330763182474?s=46">StripMallGuy (@realEstateTrent) 的推文</a>：@Tyrouse02 尝试上传一份经纪人 OM。让它分析该交易的优缺点，以可复制粘贴到 Excel 的表格格式创建租金清单，并询问第 10 年的 NOI（净运营收入）是多少...</li><li><a href="https://x.com/lmsysorg/status/1836443278033719631?s=46">lmsys.org (@lmsysorg) 的推文</a>：不再等待。o1 正式登陆 Chatbot Arena！我们通过 6000 多张社区投票测试了 o1-preview 和 mini。🥇o1-preview：全面排名第一，特别是在数学、困难提示词（Hard Prompts）和编程方面。...</li><li><a href="https://x.com/brendanfoody/status/1836435248592376149?s=46">Brendan (can/do) (@BrendanFoody) 的推文</a>：Mercor 正在通过理解人类能力的模型来解决全球劳动力匹配问题。@mercor_ai 在由 @benchmark 的 @victoralazarte 和 @bgurley 领投的 A 轮融资中筹集了 3000 万美元，估值达 2.5 亿美元...</li><li><a href="https://x.com/danielhanchen/status/1835684061475655967">Daniel Han (@danielhanchen) 的推文</a>：Transformer 的深度影响其推理能力，而模型大小影响其知识容量。强烈推荐 @ZeyuanAllenZhu 关于 Transformer 推理的视频。实验表明...</li><li><a href="https://x.com/garrisonlovely/status/1836130074388488546?s=46">Garrison Lovely (@GarrisonLovely) 的推文</a>：OpenAI 吹哨人 William Saunders 今天在参议院小组委员会作证（Helen Toner 和 Margaret Mitchell 也是如此）。他的书面证词现已上线。以下是最重要的部分...</li><li><a href="https://www.1x.tech/discover/1x-world-model">1X World Model</a>：未找到描述</li><li><a href="https://x.com/alibaba_qwen/status/1836449414220779584?s=46">Qwen (@Alibaba_Qwen) 的推文</a>：欢迎参加 Qwen2.5 基础模型发布会！这一次，我们迎来了 Qwen 历史上规模最大的发布。简而言之，我们有：博客：https://qwenlm.github.io/blog/qwen2.5/ 博客 (LLM)：htt...</li><li><a href="https://news.ycombinator.com/item?id=41581480">Moshi：用于实时对话的语音-文本基础模型 | Hacker News</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=eaAonE58sLU">AI 规划力量的寓言：从扑克到外交：Noam Brown (OpenAI)</a>：标题：AI 规划力量的寓言：从扑克到外交。演讲者：Noam Brown (OpenAI)。日期：2024 年 5 月 23 日，星期四。摘要：从 Deep Blue 在 19...</li><li><a href="https://codeforces.com/blog/entry/134091">OpenAI o1 IOI 提交记录 - Codeforces</a>：未找到描述</li><li><a href="https://github.com/voideditor/void/issues/2#issuecomment-2354428804">README 中缺失本地安装说明 · Issue #2 · voideditor/void</a>：看起来 readme 中没有运行该应用所需的必要信息</li>
</ul>

### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1285984057204936745)** (1 条消息): 

> - `Torchtune 0.3 发布`
> - `FSDP2 集成`
> - `训练时间加速`
> - `DoRA/QDoRA 支持`
> - `内存优化技术` 


- **Torchtune 0.3 带来了大量新特性**：[Torchtune 0.3](https://github.com/pytorch/torchtune/releases/tag/v0.3.0) 引入了许多新功能，包括基于 **FSDP2** 的完整 recipes，以增强灵活性和速度。
   - 此版本旨在通过前沿的增强功能加速训练并简化各种任务中的模型处理。
- **FSDP2 改进了分布式训练**：所有分布式 recipes 现在都使用 **FSDP2**，从而实现了**更好的编译支持**，并改进了对 **LoRA** 参数的处理，以实现更快的训练。
   - 用户可以在任何分布式 recipes 中尝试，以体验提升后的性能。
- **准备迎接重大的训练时间加速！**：实现 **torch.compile** 显著缩短了编译时间，在配置中设置 `compile=True` 时，编译时间不到一分钟。
   - 使用最新的 **PyTorch nightlies** 版本时，此功能可提供更快的性能提升。
- **轻松引入 DoRA/QDoRA 支持！**：最新版本现在支持 **DoRA/QDoRA**，用户只需在模型配置中添加 `use_dora=True` 即可激活此功能。
   - 这一新增功能对于使用 LoRA 和 QLoRA recipes 的用户至关重要，增强了他们的训练能力。
- **探索内存优化技术**：更新后的文档页面描述了 Torchtune 中的各种**内存节省技术**，为用户提供了多种根据硬件需求进行自定义的选项。
   - 该指南包含一个总结核心组件的综合表格，确保用户能够高效地调整其模型配置。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html">内存优化概述 &mdash; torchtune 主文档</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtune/releases/tag/v0.3.0">Release v0.3.0 · pytorch/torchtune</a>：概览 我们已经有一段时间没有发布新版本了，所以这个版本包含很多内容。一些亮点包括用于全量微调和 LoRA(/QLoRA) 的 FSDP2 recipes，支持 DoRA 微调，...</li><li><a href="https://github.com/pytorch/torchtune">GitHub - pytorch/torchtune: 一个用于 LLM 微调的原生 PyTorch 库</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账户为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1285679135620071485)** (38 条消息🔥): 

> - `Cache Management`
> - `模型的 KV Caching`
> - `评估多模态任务`
> - `Pytorch Conference 更新`
> - `三天工作制` 


- **Cache Management 问题**: 围绕每个任务后是否需要完全删除 **cache** 展开了讨论，一位成员建议虽然这很有必要，但 eval harness 可能会在内部迭代任务。
   - *一位贡献者指出，支持具有 inference 和 forward mode 的模型而不拆除 caches 可能是一个潜在的解决方案。*
- **KV Caching 机制**: 成员们探讨了其他模型（如 **HF**）如何使用 dynamic caching，这似乎可以无压力地处理较小的 batch sizes。
   - 有人建议，解决方案可能涉及使用 context manager 来启用 KV caching，但对于其在不同任务中的有效性，意见不一。
- **Pytorch Conference 讨论**: 随着 **Pytorch conference** 接近尾声，大家讨论了寻求修复方案，特别是针对 compile errors，并关联到了一个特定的 GitHub issue。
   - 一位成员表示他们将升级修复请求，而另一位成员指出该问题可能在周五之前无法解决。
- **多模态任务评估计划**: 一位用户确认了为多模态任务运行 **evals** 的计划，将使用 **1** 的 batch size 并利用 KV caching。
   - 他们表示如果时间允许，有兴趣进行更大 batch size 的评估，因为一位成员表示他们已经取得了进展。
- **轻松的职场闲聊**: 成员们就不同的工作周长度进行了轻松的闲聊，其中一人幽默地提到了他们的**三天工作制**和时区差异。
   - 这种俏皮的互动最终演变成了一场关于会议日程和截止日期的幽默交流。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/metaboy-britboy-metabrit-metaboy-british-british-metaboy-gif-26774587">Metaboy Britboy GIF - Metaboy Britboy Metabrit - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/pytorch/pytorch/issues/135439">`torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor bool call_method is_inference` · Issue #135439 · pytorch/pytorch</a>: 🐛 描述 Bug。我想使用 fullgraph=True 编译一个函数，该函数在 Module 中进行 forward passes，而该 Module 使用 torch.is_i... 检查我们是否处于 inference mode。
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1285788230909694048)** (32 条消息🔥): 

> - `Qwen2.5 发布`
> - `OpenAI o1 模型性能`
> - `AI 中的数学推理`
> - `知识截止日期问题` 


- **Qwen2.5 发布，宣称重大里程碑**：Qwen 家族的最新成员 **Qwen2.5** 被誉为最大的开源发布之一，包含 **Qwen2.5-Coder** 和 **Qwen2.5-Math** 等模型，参数量从 **0.5B** 到 **72B** 不等。
   - 亮点包括旗舰模型 **Qwen2.5-72B-Instruct**，其表现可与闭源模型媲美，在基准测试中展现出极具竞争力的性能。
- **OpenAI o1 模型性能堪比博士水平**：一位用户报告称，对 OpenAI 的 **o1-mini** 模型进行的测试显示，其水平可与生物医学领域优秀的博士生相提并论，是他们训练过的顶尖候选模型之一。
   - 这一评价强调了该模型的熟练程度及其在高级学术项目中的应用潜力。
- **数学推理能力备受关注**：AI 领域越来越重视提升**数学推理**能力，**Qwen2.5-Math** 模型引发了广泛关注，该模型同时支持英文和中文。
   - 用户的参与表明，大家正共同致力于增强与数学相关的 AI 应用，力求在该领域突破界限。
- **AI 模型知识截止日期的挑战**：几位用户对模型的**知识截止日期（knowledge cutoff）**表示失望，特别是提到该日期设定在 **2023年10月**，这影响了模型对较新编程库的适用性。
   - 讨论指出，实时信息对于实际应用至关重要，这对像 OpenAI o1 这样的模型构成了挑战。
- **AI 工作中兴奋与疲惫交织**：对于活跃在该领域的开发者来说，AI 的发展速度既令人兴奋又让人疲惫，这反映了创新的快速步伐。
   - 参与者分享了在被新技术震撼的同时感到应接不暇的感受，在挑战与进步之间寻找平衡。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/lmsysorg/status/1836443278033719631?s=46">来自 lmsys.org (@lmsysorg) 的推文</a>: 不再等待。o1 正式登陆 Chatbot Arena！我们通过 6000 多个社区投票测试了 o1-preview 和 o1-mini。🥇o1-preview：全面领先，尤其是在数学、困难提示词（Hard Prompts）和编程方面。...</li><li><a href="https://x.com/markchen90/status/1836068847167914162">来自 Mark Chen (@markchen90) 的推文</a>: 你能从竞争对手那里得到的最大赞美，就是他们联系你并询问你是否确定没有在测试集上进行训练。</li><li><a href="https://qwenlm.github.io/blog/qwen2.5/">Qwen2.5：基础模型的盛宴！</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介 在 Qwen2 发布后的三个月里，众多开发者在 Qwen2 语言模型的基础上构建了新模型，为我们提供了...</li><li><a href="https://www.bloomberg.com/news/articles/2024-09-17/newsom-says-he-s-concerned-about-chilling-effect-of-ai-bill?embedded-checkout=true">Bloomberg - 你是机器人吗？</a>: 未找到描述</li><li><a href="https://fxtwitter.com/JustinLin610/status/1836461575965938104">来自 Junyang Lin (@JustinLin610) 的推文</a>: 终于有时间聊聊这些新模型了。我们在发布 Qwen2 的那一刻就开始了 Qwen2.5 项目。在这个过程中，我们确实意识到了很多问题和犯下的错误...</li><li><a href="https://x.com/DeryaTR_/status/1836434726774526381">来自 Derya Unutmaz, MD (@DeryaTR_) 的推文</a>: 在过去的几天里，我一直在测试 OpenAI o1 模型，主要是 o1-mini，用于开发博士或博士后水平的项目。我可以自信地声称，o1 模型可与优秀的博士生相媲美...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1285679382819504331)** (4 messages): 

> - `Transformer architecture`
> - `BertViz library`
> - `GDM LLM self-critique` 


- **Transformers 彻底改变了 AI**：在开创性论文 ["Attention is All You Need"](https://dl.acm.org/doi/10.5555/3295222.3295349) 中提出的 **Transformer** 架构自 2017 年以来从根本上改变了 AI 的方法，为 OpenAI 的 **GPT**、Meta 的 **Llama** 和 Google 的 **Gemini** 等模型提供了动力。
   - Transformers 的用途还从文本扩展到了 [音频生成](https://huggingface.co/learn/audio-course/en/chapter3/introduction)、[图像识别](https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/vision-transformers-for-image-classification) 以及 [蛋白质结构预测](https://elifesciences.org/articles/82819)。
- **探索用于注意力可视化的 BertViz**：一个有用的工具 [BertViz](https://github.com/jessevig/bertviz) 可以可视化 BERT、GPT2 和 BART 等 NLP 模型中的注意力机制，并有可能无缝集成到工作流中。
   - 虽然该库尚未被广泛使用，但它可以为分析模型注意力提供即插即用的功能。
- **GDM 的 LLM 缺乏自我批判（Self-Critique）**：一条幽默的评论指出，**GDM 的 Large Language Models** 目前无法进行自我批判，这表明了它们在操作能力上的局限性。
   - 这指向了 LLM 在 AI 系统自我评估和反思性学习方面面临的持续挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://poloclub.github.io/transformer-explainer/">Transformer Explainer: LLM Transformer Model Visually Explained</a>: 一个交互式可视化工具，向你展示 Transformer 模型在 GPT 等大语言模型 (LLM) 中是如何工作的。</li><li><a href="https://github.com/jessevig/bertviz">GitHub - jessevig/bertviz: BertViz: Visualize Attention in NLP Models (BERT, GPT2, BART, etc.)</a>: BertViz：可视化 NLP 模型（BERT, GPT2, BART 等）中的注意力机制 - GitHub。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/)** (1 messages): 

xeophon.: https://x.com/agarwl_/status/1836119825216602548?s=46
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

xeophon.: 我爱 Twitter
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1285708031065325569)** (10 messages🔥): 

> - `01 App Functionality`
> - `Automating Browser Form Tasks`
> - `CV Agents Experimentation` 


- **01 App 已完全投入运行**：成员们确认 **01 app** 在他们的手机上运行正常，并特别注意使用 **-qr 选项** 来实现功能。
   - 一位成员提到他们尚未测试该 App 的本地版本，但已让非本地版本顺利运行。
- **寻求自动化浏览器任务技巧**：一位成员正在寻求有关自动化浏览器表单提交任务的**指南或技巧**，特别是针对各种政府机构的门户网站。
   - 他们目前正在遵循 **ChatGPT 4o** 的建议，但效率低下，经常陷入循环。
- **CV Agents 可供测试**：一位成员分享了他们在 GitHub 上的 **CV Agents** 项目链接，该项目旨在通过使用智能简历来帮助更智能地找工作。
   - 该项目附带了极具吸引力的视觉描述，并邀请其他人参与贡献。
- **咨询应用程序控制功能**：一位成员询问了 **CV Agents** 在有效控制应用程序方面的能力。
   - 这引发了关于潜在集成以及正在考虑的各种方法的讨论。
- **排除 01 App 错误**：一位成员对使用 **01 app** 时遇到的错误表示沮丧，尽管已经输入了所需的参数。
   - 他们请求协助进行故障排除，并寻求解决该问题的分步指南。



**提到的链接**: <a href="https://github.com/0xrushi/cv-agents">GitHub - 0xrushi/cv-agents: Intelligent Resumes for Smarter Job Hunting</a>: 用于智能求职的智能简历。通过在 GitHub 上创建一个账户来为 0xrushi/cv-agents 的开发做出贡献。

  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1285689678044991541)** (7 条消息): 

> - `Beta 测试名额可用性`
> - `Windows 上的浏览器操作问题`
> - `Discord 应用商店 Prompt 错误` 


- **Beta 测试名额查询**：<@631210549170012166> 被问及 Beta 测试中是否有空余名额，以便进行日常测试和反馈。
   - 另一位成员也表达了同样的兴趣，强调他们非常渴望参与后续互动。
- **浏览器中打开 Perplexity**：一位成员报告称，在 Windows 上执行浏览器操作时，**01** 每次都会打开 **Perplexity**，这引发了对其功能的担忧。
   - 他们询问其他人是否也有关于这种异常行为的类似经历。
- **Discord 应用商店 Prompt 失败**：一位成员尝试执行 *从 Microsoft App Store 下载 Discord* 的 Prompt，但收到错误提示：**“此任务无法完成” ('this task is impossible')**。
   - 他们对此问题表示沮丧，并指出虽然它运行了打开应用商店的代码，但未能完成任务。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1285814343153225788)** (4 条消息): 

> - `Moshi 成果发布`
> - `Moshi 技术报告`
> - `Moshi GitHub 仓库`
> - `音频同步反馈` 


- **Moshi 成果发布**：今天，**Kyutai Labs** 发布了多个 Moshi 成果，包括一份详细的技术报告、模型权重，以及基于 [Pytorch](https://pytorch.org/)、Rust 和 MLX 的流式推理代码。
   - 您可以在**[此处找到论文](https://kyutai.org/Moshi.pdf)**，并访问 **[GitHub](https://github.com/kyutai-labs/moshi)** 上的仓库。
- **Moshi GitHub 仓库讨论**：一位成员分享了 Moshi 的 GitHub 仓库链接，强调了其对社区参与该项目的重要性。
   - 另一位成员表达了对 Moshi 开发进一步更新的期待。
- **缩略图和音频反馈**：一位用户对 Moshi 相关的视频发表了评论，建议更新缩略图以提高可见度和播放量。
   - 他们还指出音频同步略有偏差，表明需要进行技术调整。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/kyutai_labs/status/1836427396959932492?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">来自 kyutai (@kyutai_labs) 的推文</a>：今天，我们发布了多个 Moshi 成果：一份包含模型背后所有细节的长篇技术报告，Moshi 及其 Mimi 编解码器的权重，以及 Pytorch、Rust 和... 中的流式推理代码。</li><li><a href="https://github.com/kyutai-labs/moshi">GitHub - kyutai-labs/moshi</a>：通过在 GitHub 上创建账号来为 kyutai-labs/moshi 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1286032047785119815)** (1 条消息): 

> - `RAG 服务部署`
> - `AWS CDK`
> - `LlamaIndex` 


- **Benito Martin 的 RAG 部署指南**：Benito Martin 分享了一份关于使用 [@awscloud CDK](https://t.co/vsB3x9rYUY) 端到端构建和部署 **RAG 服务**的指南，为将原型转化为生产环境提供了宝贵的资源。
   - *如果您希望提升部署技能，这份指南是一个快速上手的开始！*
- **AWS 的基础设施即代码 (Infrastructure-as-Code)**：该指南强调了为 AWS 使用**基础设施即代码提供商**，从而简化 RAG 服务部署流程。
   - 这种方法显著简化了从开发环境到生产环境的过渡。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1285765755949813790)** (10 条消息🔥): 

> - `Weaviate 问题解决`
> - `开源贡献流程`
> - `RAG 方法反馈` 


- **Weaviate 问题已关闭但仍存在问题**：Yasuyuki 提出了一个与读取现有 **Weaviate** 数据库时遇到的 **KeyError** 相关的问题，引用了 [GitHub Issue #13787](https://github.com/run-llama/llama_index/issues/13787)。一位社区成员建议 fork 该仓库并创建一个 pull request 来修复该功能，从而允许用户指定字段名称。
   - *这是在查询非 llama-index 创建的向量数据库时常见的陷阱*。
- **对开源软件（OSS）的贡献触发了新的学习**：Yasuyuki 表达了对贡献该项目的兴趣，并提到通过将键从 'id' 修正为 'uuid' 来实施应急措施。他们正准备学习如何制作 pull request 以正式提交修复。
   - *这次首次贡献鼓励他熟悉 GitHub 工作流，以便未来参与*。
- **RAG 策略探索**：.sysfor 寻求关于集成 **RAG** (Retrieval-Augmented Generation) 技术来处理供应商问题的反馈，方法是索引相关的问答对。该策略涉及计算语义分数，并可能使用元数据将问题与其回答链接起来。
   - 社区成员建议索引 QA 对，并考虑生成问题的变体以提高检索效率。



**提到的链接**：<a href="https://github.com/run-llama/llama_index/issues/13787">[Question]: LLamaIndex and Weaviate · Issue #13787 · run-llama/llama_index</a>：问题验证。我已经在文档和 discord 中搜索了答案。问题：我正尝试使用 llamaIndex 从我的 weaviate 向量数据库中检索文档。我遵循了...

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1286054599563804755)** (3 条消息): 

> - `LLM 响应延迟`
> - `Python 和 LangChain 优化` 


- **模型提供商导致 LLM 延迟**：一位成员指出，**LLM 响应延迟** 几乎总是归因于 **模型提供商**。
   - 他们强调，这是影响响应时间的重要因素，而不是实现过程中的问题。
- **Python 和 LangChain 对延迟的影响微乎其微**：另一位成员建议，只有大约 **5-10%** 的延迟可以归因于 **Python 或 LangChain** 本身。
   - 这意味着优化模型或提供商设置可能会对响应时间产生更大的影响。


  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1285879252797095957)** (2 条消息): 

> - `Langserve`
> - `React 前端`
> - `状态管理`
> - `Python 后端` 


- **Langserve 与 React 的状态管理**：一位用户询问了在将 **Langserve** 与 **React 前端** 集成时，关于 **状态管理** 的最佳实践。
   - 他们寻求在该技术栈中有效管理状态的经验见解。
- **与 Python 后端的集成挑战**：同一位用户提到他们在应用设置中使用 **Python 后端**。
   - 这意味着讨论可能会延伸到 React 前端如何与 Python 后端进行交互以处理状态。


  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1285833298853236797)** (5 messages): 

> - `PDF Extraction Toolkit`
> - `RAG Application with AWS`
> - `LangChain Framework` 


- **用于 PDF 内容提取的 PDF-Extract-Kit**：一位成员介绍了 [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)，这是一个旨在进行高质量 PDF 内容提取的综合工具包，并建议其在常见用例中的实用性。
   - 另一位成员对该工具包的效果表示好奇并计划进行尝试，强调处理 PDF 提取是一个普遍需求。
- **使用 AWS 技术栈的 RAG 应用**：一位成员分享了一个令人兴奋的新 [RAG 应用](https://github.com/benitomartin/aws-bedrock-opensearch-langchain)，该应用使用 Terraform 作为基础设施即代码 (IaC) 开发，利用 LangChain 和 AWS Bedrock 获取 LLM 和 embedding 模型，并使用 AWS OpenSearch 作为向量数据库。
   - 该应用部署在 Amazon Web Services (AWS) 上，使用现有的 AWS OpenSearch 端点，展示了一个强大的数据处理云解决方案。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/benitomartin/aws-bedrock-opensearch-langchain">GitHub - benitomartin/aws-bedrock-opensearch-langchain: RAG Application with LangChain, Terraform,  AWS Opensearch and AWS Bedrock</a>: 使用 LangChain, Terraform, AWS Opensearch 和 AWS Bedrock 构建的 RAG 应用 - benitomartin/aws-bedrock-opensearch-langchain</li><li><a href="https://github.com/opendatalab/PDF-Extract-Kit">GitHub - opendatalab/PDF-Extract-Kit: A Comprehensive Toolkit for High-Quality PDF Content Extraction</a>: 一个用于高质量 PDF 内容提取的综合工具包 - opendatalab/PDF-Extract-Kit
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1285827218001629208)** (8 messages🔥): 

> - `BeToast Discord Compromise`
> - `Windows Native Support` 


- **BeToast Discord 入侵警报**：一位成员对 **BeToast** Discord 服务器可能遭到入侵表示担忧，并引用了与 LinkedIn 上一位报告被黑客攻击的人士的对话。
   - 另一位成员确认了该警报，强调如果被盗账号开始发送垃圾信息，需要保持警惕并采取适当行动。
- **Windows 原生支持时间表尚不确定**：针对关于 **Windows 原生支持** 的提问，一位成员分享了一个 [GitHub issue](https://github.com/modularml/mojo/issues/620)，其中详细记录了对该功能的需求，并解释说这可能需要一些时间。
   - 讨论强调，由于成本和普及度问题，许多开发者在 AI 部署中更倾向于 Windows 以外的替代方案，而 WSL 是一个常见的变通方案。



**提及的链接**: <a href="https://github.com/modularml/mojo/issues/620">[Feature Request] Native Windows support · Issue #620 · modularml/mojo</a>: 查看 Mojo 的优先级。我已经阅读了路线图和优先级，我认为此请求符合优先级。你的请求是什么？对 Windows 的原生支持。什么时候可以使用？...

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1285893499610202143)** (2 messages): 

> - `SIMD Conversion`
> - `Data Type Handling` 


- **SIMD 转换为 Int 的说明**：一位用户询问如何将 **SIMD[DType.int32, 1]** 转换为 **Int**。
   - *phomola* 提供了一个简洁的解决方案：`int(x)`。
- **澄清 SIMD 数据类型**：讨论强调了在转换中理解 **SIMD** 及其数据类型的重要性。
   - 成员们指出，熟悉 **DType** 选项可以简化此类查询。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1285823773202845706)** (6 messages): 

> - `State of the Art Text to Speech`
> - `Open Source TTS Solutions`
> - `Closed Source TTS Solutions` 


- **探索最先进的文本转语音技术**：一位成员询问了文本转语音 (TTS) 的 **最先进技术 (SOTA)**，特别是寻求开源解决方案。
   - *“理想情况下是开源的，但也很好奇目前市面上都有什么”* 反映了对比各种选项的愿望。
- **对 Eleven Labs 的赞赏**：[Eleven Labs](https://elevenlabs.io/) 被一位成员推荐为目前可用的 **最佳闭源** 文本转语音选项。
   - 这表明它具有强大的功能，但关于开源选项的观点仍在讨论中。
- **关于开源 TTS 选项的辩论**：开源选项包括 **styletts2**、**tortoise** 和 **xtts2**，作为可供考虑的替代方案被分享。
   - 对话表明，对于这些解决方案的有效性存在多种不同意见。


  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1286005860296032419)** (4 messages): 

> - `OmniGen`
> - `Nvidia open-source LLMs`
> - `SDXL VAE`
> - `Phi-3` 


- **介绍用于统一图像生成的 OmniGen**：题为 [OmniGen](https://arxiv.org/abs/2409.11340) 的论文展示了一种新的扩散模型，该模型集成了多种控制条件，无需像 **Stable Diffusion** 等模型中常见的额外模块。
   - OmniGen 通过其简化的架构支持多种任务，包括 **text-to-image generation**（文本生成图像）、**image editing**（图像编辑）以及经典的 CV 任务。
- **Nvidia 官方开源 LLMs**：一位成员强调了 **Nvidia 官方开源 LLMs** 的可用性，这可能与正在进行的 AI 研发相关。
   - 这一举措可能为该领域的开发者和研究人员提供宝贵的资源。
- **OmniGen 的特性突出了 SDXL VAE 和 Phi-3**：OmniGen 利用了 **SDXL VAE** 和 **Phi-3**，增强了其在生成图像和处理控制条件方面的能力。
   - 这些集成进一步简化了其用户友好的架构，使其更易于应用于各种场景。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://nvlm-project.github.io">未找到标题</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2409.11340">OmniGen: Unified Image Generation</a>：在这项工作中，我们介绍了 OmniGen，一种用于统一图像生成的全新扩散模型。与流行的扩散模型（如 Stable Diffusion）不同，OmniGen 不再需要额外的模块，例如...
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1285679813230723182)** (6 messages): 

> - `Ruff check error`
> - `Interview with Sayash Kapoor and Benedikt Stroebl`
> - `LanceDB integration with DSpy`
> - `Elixir live coding`
> - `Typed predictors example` 


- **Ruff Check 错误警报**：一位用户报告在运行 `ruff check . --fix-only` 时出现 **TOML 解析错误**，指出第 216 行存在 **未知字段** `indent-width`。
   - 该错误建议修改配置文件，以符合错误消息中列出的预期字段。
- **与 AI 研究人员的播客**：一个精彩的 **YouTube 播客**，由 Sayash Kapoor 和 Benedikt Stroebl 主持，讨论了如何在最小化推理成本的同时优化任务性能，观看地址见[此处](https://youtu.be/gCP-W_BNzg4)。
   - 听众表达了对内容的期待，并承认在 AI 系统中考虑成本的必要性。
- **LanceDB 集成首次亮相**：DSpy 宣布了新的 **LanceDB 集成**，通过针对大型数据集的 retriever 增强了性能，详见此 [pull request](https://github.com/stanfordnlp/dspy/pull/1444)。
   - 贡献者表达了在个人项目和开源倡议上进行合作的兴趣。
- **Elixir 现场编程环节**：一场专注于 **Elixir 模板和项目** 的现场编程环节正在休息室进行，邀请社区成员参与。
   - 成员们通过 Discord 链接获知加入正在进行的 Elixir 开发活动。
- **请求 Typed Predictors 示例**：一位用户询问关于使用 typed predictors 的 **O1 运行示例**，寻求社区的帮助。
   - 该请求表明需要资源或演示来帮助理解这一特性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/gCP-W_BNzg4">AI Agents That Matter with Sayash Kapoor and Benedikt Stroebl - Weaviate Podcast #104!</a>：AI 研究人员过度拟合于最大化最先进的准确率，却牺牲了运行这些 AI 系统的成本！我们需要在操作中考虑成本...</li><li><a href="https://github.com/stanfordnlp/dspy/pull/1444">Lancedb Integration by PrashantDixit0 · Pull Request #1444 · stanfordnlp/dspy</a>：此 PR 添加了 LanceDB 作为 retriever 以处理大型数据集。
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1285776958260510772)** (3 messages): 

> - `API Key 管理`
> - `对非官方服务器的信任问题`
> - `可重用的 RAG 流水线`
> - `多公司上下文` 


- **关于 API Key 处理的担忧**：用户质疑是否必须将 API keys 直接发送到 VM/服务器，然后再传输到 OpenAI，并强调**用户可能不信任**将 keys 发送到非官方服务器。
   - 一位成员强调需要明确安全流程，以避免泄露个人数据。
- **用户信任与 API 计算**：展开了关于用户是否为独立服务器上的计算提供 API keys 的讨论，暗示这种信任问题影响了整体集成。
   - 对话表明，这一担忧与语言互操作性相关的挑战是不同的。
- **创建可重用的 RAG 流水线**：一位成员寻求关于为多家公司构建**带有 RAG 的可重用流水线**的建议，同时在不使 prompt 过载的情况下管理上下文信息。
   - 他们表达了关于如何整合多样化数据，而又不因信息过多导致 prompt 过于复杂的担忧。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1285979644117647524)** (8 messages🔥): 

> - `课程学习 (Curriculum Learning) 实现`
> - `数据集洗牌 (Shuffling) 控制` 


- **PyTorch 中的课程学习步骤**：要在基于 PyTorch 的环境中实现**课程学习 (curriculum learning)**，你应该定义标准、对数据集进行排序，并创建一个自定义数据集类来处理课程逻辑。
   - 一个示例展示了如何将数据集分割成难度递增的阶段，并在训练循环中更新数据集。
- **数据集洗牌控制查询**：用户询问如何指定他们不希望对数据集进行**随机洗牌 (random shuffle)**。
   - 关于在数据集上下文中指定此操作的指导，建议在单独的线程中讨论。



**提到的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=b1253ada-5c5a-454c-807d-a5022967129e)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1286001128139395115)** (3 messages): 

> - `Tinybox 设置指南`
> - `Tinygrad 与 Tinybox 集成`
> - `使用 Tinybox 进行 MLPerf 训练` 


- **Tinybox 设置寻求帮助**：@drose0933 请求协助设置他们的**两台 tinybox** 并索要说明。
   - 作为回应，一位成员提供了 [Tinybox 文档](https://docs.tinygrad.org/tinybox/) 的链接以获取设置指导。
- **Tinybox 在 tinygrad CI 中的角色**：**tinybox** 被强调为 **tinygrad CI** 中大量使用的平台，被证明是与 tinygrad 配合使用的最佳测试平台。
   - 他们通过在 [MLPerf Training 4.0](https://public.tableau.com/views/MLCommons-Training_16993769118290/MLCommons-Training) 上运行 tinygrad 展示了其能力。
- **Tinybox 的购买选项**：对于有兴趣购买 **tinybox** 的用户，该成员提到可以访问 [tinygrad.org](https://tinygrad.org) 进行购买。
   - 他们安慰说，对于那些可能没兴趣购买的人也没关系。
- **Tinybox 功能介绍**：消息简要介绍了 **tinybox**，这是一款专为 AI 工作负载设计的通用系统，涵盖了训练和推理。
   - 提供了硬件规格详情，指出**红盒子 (red box)** 包含六块 **7900XTX GPU**，而**绿盒子 (green box)** 包含六块 **4090 GPU**。



**提到的链接**：<a href="https://docs.tinygrad.org/tinybox/">tinybox - tinygrad 文档</a>：未找到描述内容

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1286031414969765998)** (1 messages): 

> - `rateLLMiter`
> - `API call management`
> - `Pip install modules` 


- **rateLLMiter 现已支持 Pip 安装**：**rateLLMiter** 模块现在作为一个可通过 pip 安装的包发布，增强了 LLM 客户端的请求管理。
   - 实现代码可以在 [GitHub](https://github.com/llmonpy/ratellmiter) 上找到，并附有其 **MIT license** 的详细信息。
- **速率限制器图表展示请求管理**：一张图表展示了 **rateLLMiter** 如何平滑请求流，其中橙色线代表票据（tickets）请求，绿色线代表已发放的票据。
   - 它有效地将 **100 个请求** 的峰值随时间分散，以防止服务器速率限制异常。



**提到的链接**: <a href="https://github.com/llmonpy/ratellmiter">GitHub - llmonpy/ratellmiter: Rate limiter for LLM clients</a>: LLM 客户端的速率限制器。通过在 GitHub 上创建账号来为 llmonpy/ratellmiter 的开发做出贡献。

  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1285862909670457437)** (1 messages): 

> - `Prompt Errors`
> - `Template Usage` 


- **意识到 Prompt 误用**：一名成员发现他们错误地使用了自己的 Prompt，导致输出结果混乱。
   - 这一发现强调了仔细检查 Prompt 应用以获得准确结果的重要性。
- **Prompt 模板的可用性**：该成员指出，提供的 Prompt 模板可供参考，以辅助构建未来的 Prompt。
   - 有效利用模板可以帮助在未来的交互中防止类似问题的发生。


  

---



---



---



---



---



{% else %}


> 完整的逐频道细分内容已针对电子邮件进行截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}