---
companies:
- discoresearch
- fireworks-ai
- hugging-face
- mistral-ai
date: '2023-12-09T23:30:00.926075Z'
description: '**Mixtral 的权重**在没有代码的情况下发布，促使 **Disco Research 社区**和 **Fireworks AI**
  迅速对其进行了实现。尽管付出了努力，但并未报告显著的基准测试提升，这限制了其在本地大语言模型（LLM）使用中的效用，但标志着**小模型社区**的进步。


  DiscoResearch Discord 中的讨论涵盖了 **Mixtral 的性能**与 **Hermes 2.5** 和 **Hermes 2** 等模型的对比，并在
  **winogrande**、**truthfulqa_mc2** 和 **arc_challenge** 等基准测试上进行了评估。技术话题包括 GPU 需求、多
  GPU 设置以及通过 **GPTQ** 进行的量化。


  此外，研究人员还探索了基于语法的评估、思维链（CoT）和 min_p 采样等基准测试策略，以及 Min P 和 Top P 等模型采样技术，以增强响应的稳定性和创造力。用户还讨论了
  GPT 的学习局限性以及模型在不同条件下的适应性，并强调了 min_p 采样在允许更高温度设置以激发创造力方面的作用。'
id: fd6ab5f1-7943-4163-bff1-af1c235853f7
models:
- mixtral
- hermes-2.5
- hermes-2
- mistral-yarn
- ultrachat
original_slug: ainews-1292023-the-mixtral-rush
people:
- bjoernp
- the_bloke
- rtyax
- kalomaze
- solbus
- calytrix
title: 2023年12月9日：Mixtral 狂潮
topics:
- benchmarking
- gpu-requirements
- multi-gpu
- quantization
- gptq
- chain-of-thought
- min-p-sampling
- top-p-sampling
- model-sampling
- model-merging
- model-performance
- small-models
- reasoning-consistency
- temperature-sampling
---

<!-- buttondown-editor-mode: plaintext -->Mixtral 的权重在没有代码的情况下发布了，因此 Disco Research 社区（新加入的）一夜之间爆发，致力于实现它：

 
![image.png](https://assets.buttondown.email/images/0b171e47-332c-435c-b61e-b3b0a2eb851c.png?w=960&fit=max)
 

我们也看到了来自 Fireworks AI 的类似努力：

 
![image.png](https://assets.buttondown.email/images/b79af424-deb3-42ec-85a3-b15c3a328bc2.png?w=960&fit=max)
 

不幸的是，目前还没有人报告显著的基准测试提升，而且它可能不太适合本地 LLM 使用。尽管如此，这对于 smol models 社区来说仍是巨大的进步。


[TOC] 


## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 摘要

- 多个频道中关于 **Mixtral 模型的性能和实现**的讨论。这包括它在 Hermes 2.5 和 Hermes 2 等新旧模型背景下的功能。例如，讨论了 Mixtral 在各种测试（如 `winogrande`、`truthfulqa_mc2` 和 `arc_challenge`）中表现出的**性能行为**。此外，还涉及了 GPU 需求、内存限制的影响以及多 GPU 设置问题等技术方面。

    *"基础模型由用户 `@bjoernp` 使用 HuggingFace (HF) transformers 实现，发现在约 `70B` 的性能水平下，计算量仅需约 `12B`，内存需求约为 `47B`。"* - [mixtral_implementation, @the_bloke](https://discord.com/channels/1178995845727785010/1182759434326396998/)

- 针对不同数据集的**基准测试模型评估和检测策略**。`@bjoernp` 引入了基于语法的评估、思维链 (CoT) 和 min_p 采样方法等考量。提议将 Hellaswag 基准测试和 FastEval 作为潜在工具，用户 `@rtyax` 提出了将 llama.cpp 整合到 FastEval 中的想法。讨论了关于 CoT 或 Tree of Thought 的澄清思路以及 min_p 采样的应用。

    *"提出了检测作弊措施的建议，例如打乱问题顺序或保留一定比例的问题不予发布。"* - [benchmark_dev, @.calytrix](https://discord.com/channels/1178995845727785010/1183158791605330051/)

- 关于**模型采样技术**（包括 Min P 和 Top P）及其对生成响应的稳定性、连贯性和创造力的各自影响的深入辩论。

    *"他建议通过 10 次重复运行的过程来确定模型的推理一致性。"* - [general, @kalomaze](https://discord.com/channels/1178995845727785010/1182877486854451271/)
    
- 用户强调了 **GPTs 的学习过程**和局限性。`@solbus` 关于 Agent 如何将上传的文件作为“知识”存储并利用的澄清值得关注。

    *"上传的文件作为‘知识’存储供 Agent 参考，但不会持续修改其基础知识。"* - [general, @solbus](https://discord.com/channels/1178995845727785010/1182877486854451271/) 

- **模型在不同条件下的适应性和通用性**是一个重点话题。讨论了通过 min_p 采样方法启用更高温度模型设置的潜在好处。

    *"Min P 采样在启用更高温度设置方面的作用，使模型能够以合适且受控的方式更具创造性。"* - [benchmark_dev, @kalomaze](https://discord.com/channels/1178995845727785010/1183158791605330051/)

**DiscoResearch 频道摘要**

### ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/) (1 条消息): 
        
cryptossssun: 有开发 Mixtral 模型的计划吗？

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (651 条消息🔥🔥🔥): 
        
- **Hermes 2.5 与 Hermes 2 性能对比**：用户讨论了名为 **Hermes 2.5** 的新版 Hermes 实现的性能。一位用户报告称，它在各种 Benchmark 中表现优于 Hermes 2。
- **新 Mixtral 模型实现**：多位用户讨论并报告了他们在实现新发布的 Mixtral 模型方面的进展。用户 `@bjoernp` 使用 HuggingFace (HF) Transformers 实现了基础模型，发现其在约 `12B` 的计算量和约 `47B` 的内存需求下，达到了 `70B` 级别的性能。用户 `@the_bloke` 还通过 GPTQ 实现了该模型的量化，但目前仍处于测试阶段。
- **模型合并策略讨论**：提出了各种合并技术，其中一位用户建议将 `UltraChat` 与基础 `Mistral` 之间的差异应用于 `Mistral-Yarn`。
- **模型性能评估**：几位用户报告了 Mixtral 实现的 Benchmark 结果。初始 Benchmark 在 `winogrande`、`truthfulqa_mc2` 和 `arc_challenge` 等各种评估中表现不一。在修复了 softmax+topk 的 bug 后，性能结果有所提升。据报告，进一步的 Finetuning 正在进行中。
- **模型加载与 GPU 需求讨论**：用户讨论了加载新 Mixtral 模型的各种问题和技术，包括解决内存限制、优化加载时间以及多 GPU 设置的问题。GPU 显存讨论表明，该模型可以在具有约 `24GB` VRAM 的 GPU 上以 `4bit` 模式加载。用户还分享了将该模型集成到 `textgen-webui`、`exllama` 等现有工具中的问题。


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (222 条消息🔥🔥): 
        
- **Mistral 模型讨论**：`@bjoernp` 和 `@sinan2` 交流了 **Mistral** 模型在 Hermes 2.5 和 Hermes 2 上的性能，以及将 Mistral 扩展到 8K 以上的相关问题。
- **GPTs Agent 学习问题**：`@tilanthi` 对 GPTs Agent 在初始训练后无法从额外信息中学习表示担忧。`@solbus` 澄清说，上传的文件是作为“知识”存储供 Agent 参考的，但并不会持续修改其基础知识。
- **聊天机器人模型性能对比**：`@cryptossssun` 分享了 MistralAi 的一个 MoE 模型初步 HuggingFace 实现 [mixtral-7b-8-expert](https://huggingface.co/DiscoResearch/mixtral-7b-8expert)，并讨论了 Mistral 原始模型与 Mixtral 之间可能存在的性能差异。
- **模型采样技术讨论**：`@kalomaze` 发表了关于 Top P 局限性的看法，并建议在典型的 1.0 Temperature 条件下采用 "min P 0.1" 采样方法。他建议通过 10 次重复运行的过程来确定模型的推理一致性。
- **模型 Benchmark 的潜在改进**：`@bjoernp` 提出了一种新的模型 Benchmark 方法，结合了用于自一致性的 10 倍重采样、基于语法的评估、Chain of Thought (CoT) 以及 min_p 采样方法。他与 `@kalomaze` 讨论了编程约束和实现细节，并邀请后者可能牵头开展这项工作。

### ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/) (111 messages🔥🔥): 
        
- **改进 Benchmark 评估**：该频道旨在讨论如何改进 Benchmark 的评估，核心想法包括使用 CoT 或 Tree of Thought，对每个问题进行多次评估以规避 token 概率问题，采用 min_p sampling 以获得更好的问题解决效果，以及在 CoT 推理后应用基于 grammar 的方法以获得更多有效答案。特别提到了 `@kalomaze` 关于多次运行问题的价值的思考，这不仅影响二元对错判断，还能突出模型错误的程度。
- **采样方法**：围绕各种采样方法（特别是 Min P 和 Top P）及其对生成响应的连贯性、创造性和稳定性的影响展开了广泛讨论。`@kalomaze` 提出了 Min P sampling 的优势，证明了其在截断（truncation）方面的优越性，并通过分享多个模型响应示例进行了演示。他的主张遭到了 `@.calytrix` 的质疑，后者指出人类偏好并不总是与模型所能达到的最佳推理能力一致。
- **Benchmarks 与工具**：[Hellaswag benchmark](https://allenai.org/data/hellaswag) 和 [FastEval](https://github.com/FastEval/FastEval) 都被视为潜在资源，尽管它们与所提议方法的契合度尚未确认。用户 `@rtyax` 提到了将 llama.cpp 集成到 FastEval 中的可能性。
- **Benchmarking 的标准化**：用户对 Benchmark 测试缺乏标准化和可靠性表示担忧，提到了评估技术和 sampler 设置的多样性。有人建议采取检测作弊的措施，例如打乱题目顺序或保留一定比例的题目不公开。
- **模型可扩展性与多功能性**：`@kalomaze` 报告了 Min P sampling 在支持更高 temperature 设置方面的潜力，使模型能够以合适且受控的方式更具创造性，甚至适用于编程用途。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- **Mistral 7B** 的 Benchmarking 方法，并将其与 **Hermes 2.5** 和 **Hermes 2** 等其他模型进行了比较。分享了一系列与这些模型的 Benchmarking 和改进相关的推文：[Tweet1](https://twitter.com/abacaj/status/1733292527904592350), [Tweet2](https://twitter.com/tsengalb99/status/1733222467953422702)。
- 讨论了 **Mixtral** 的内存需求分析、其 GPU 使用的内存优化以及具体的 Mixtral 推理实现。
- **微调 MOE (Mixture of Experts)** 和像 **GPT-4** 这样的大型模型的潜力和可行性。*“讨论了微调 MOE (Mixture of Experts) 模型的潜力，并认为企业从基于基础 MOE 架构的持续预训练中获益，要优于单纯微调像 GPT-4 这样的大型模型”*。
- 比较了 **GGUF, GPTQ, AWQ 等量化方法**，将 AWQ 描述为一种更“动态”的方法。频道内还讨论了关于“2.5 bits”一词的困惑。
- 参考 Tim Dettmers 关于仅用 5GB RAM 运行模型的说法，讨论了 **Mixtral 8x7B** 等模型的 VRAM 需求。
- 为寻求深入了解 **MoEs (Mixture of Experts)** 结构和功能的用户提供了共享资源。
- 在 *off-topic* 频道中交流了关于 Nous 的运营结构和不同象征性物品的查询。具体而言，出现了关于用户 `@coffeebean6887` 提出的旧金山某不明物品的报价问题，以及关于 Nous 是否有员工或是否为全志愿者组织的查询。
- 对 AI 未来的推测通常涉及未来对 AI 的监管，并建议寻找限制较少的地区继续 AI 项目。其中一个建议特别关注预期的 EU AI Act 限制。

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (7 条消息): 
        
- **关于物体外观的讨论**：用户 `@eas2535` 对某些物体的外观发表了评论，称：“*那些头没接对*。”
- **旧金山物体赠送**：`@coffeebean6887` 向位于旧金山的任何人提供某种未指明物体的多余部分，幽默地暗示他们可能拿得比预想的要多。
- **索要物体**：`@gabriel_syme` 表达了对“那个女孩”（推测是之前提到的比喻性物体）的渴望，尽管他离旧金山很远。他们补充说可以承担邮寄费用。该用户随后询问了关于该物体美感的确认，问道：“*看起来不错吧？*”
- **分享链接**：`@euclaise` 分享了一个推文链接，未作进一步评论。[查看推文](https://vxtwitter.com/tarantulae/status/1733263857617895558)
- **Nous 雇佣咨询**：`@nasw` 询问 Nous 是否有员工，或者是否是一个全志愿者组织，并提到了用户 `@jade` 和 `@teknium`。他们为该问题是否不适合该频道表示歉意，称自己正在找工作并感到好奇。


### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (7 条消息): 
        
- `@nonameusr` 分享了一系列与 **Mistral** 模型 Benchmark 相关的推文。推文来自 Anton [@abacaj](https://twitter.com/abacaj)，讨论了 **Mistral-7B** 在标准测试下的评估。
- 在一条推文中，Anton [@abacaj](https://twitter.com/abacaj/status/1733292527904592350) 报告了 33.54% 的分数，相比标准 **Mistral-7B** 的 30.5% 有所提高。 
- `@gabriel_syme` 对这些测试所使用的代码表现出兴趣，随后意识到这些代码可以在公共仓库中获取。


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (34 条消息🔥): 
        
- **DeepSpeed v0.5 Mixture of Experts (MoE) 训练**：`@everyoneisgross` 分享了 [DeepSpeed v0.5](https://www.deepspeed.ai/tutorials/mixture-of-experts/) 的链接，该版本支持训练 Mixture of Experts (MoE) 模型。指出 MoE 模型是一类新兴的稀疏激活模型，其计算成本相对于参数呈次线性增长，并重点介绍了 Switch Transformer 的例子。
- **MoE 实现与功能**：`@everyoneisgross` 建议查看 GitHub 页面 [megablocks-public/megablocks/layers/moe.py](https://github.com/mistralai/megablocks-public/blob/main/megablocks/layers/moe.py) 上的注释，以便更清楚地了解 MoE 的工作原理。
- **Mistral/Mixtral 的推理代码**：`@fullstack6209` 分享了 GitHub 页面 [llama-mistral](https://github.com/dzhulgakov/llama-mistral) 的链接，该页面提供了将 Mistral 和 Mixtral 模型修改并整合进原始 Llama 实现中的推理代码。
- **用于文本生成的 8x7B MoE 基础模型**：`@if_a` 链接到了 Replicate 平台上的 [MistralAI 新模型](https://replicate.com/nateraw/mixtral-8x7b-32kseqlen)，并指出该模型运行在 4x Nvidia A100 (80 GB) GPU 上。
- **新的 2-bit 量化方法**：`@cyborgdream` 分享了来自 @tsengalb99 的推文信息，介绍了一种新的 2-bit 量化方法 [QuIP#](https://twitter.com/tsengalb99/status/1733222467953422702)，该方法适用于大语言模型，且具有接近 fp16 的性能。

### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (667 条消息🔥🔥🔥): 
        
- **Mistral AI 模型讨论**：用户讨论了 Mistral AI 模型的各个方面，包括它们的性能、潜在改进，以及它们与 Hermes 2.5 和 Hermes 2 等其他模型的对比。`@fblgit` 分享了关于不同模型性能的见解，提到 **Xaberius 34B** 在 LLM 排行榜上处于领先地位。

- **Mixtral 推理实现**：关于 Mixtral 正确推理协议的辩论。多位用户提出了不同的框架。随后达成共识，即在 topk 之后应用 softmax 会带来更好的 benchmark 结果。

- **内存与性能权衡**：讨论了 Mixtral 的内存需求，以及如何优化它以更有效地利用 GPU 内存。有人指出，尽管 Mixtral 占用大量 VRAM，其推理速度与 **Mistral 7B** 模型相似。此外，有人建议混合精度（mixed-precision）方法可能是一个可行的解决方案。

- **微调 AI 模型**：讨论了微调 MOE (Mixture of Experts) 模型的潜力，有观点认为，与简单地微调像 **GPT-4** 这样的大型模型相比，企业从基于 MOE 架构的持续预训练（pretraining）中获益更多。此外，还交流了关于增强数据集以获得更好 GSM 分数的想法。

- **监管担忧**：用户对 AI 监管的前景表示担忧，特别是关于欧洲的 EU AI Act 以及对开源 AI 项目可能存在的限制。一些人讨论了寻找监管较少的地方来继续他们的 AI 项目。


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (40 条消息🔥): 
        
- **量化方法 GGUF, GPTQ, AWQ**：用户 `@akhxl` 询问了这些量化方法之间的区别。用户 `@cyborgdream` 解释说 GGUF 是一种文件格式，而不是一种量化方法。而 GPTQ 和 AWQ 是不同的量化方法，AWQ 是一个更“动态”且“更聪明”的选择。`@cyborgdream` 还澄清了关于此背景下“2.5 bits”含义的困惑，指出它们意味着 2 bits，然后每“几个参数就有一个带有额外信息的额外字节”。
- **微调 Mistral 7B**：`@.beowulfbr` 寻求帮助微调 Mistral 7B 的 notebook。`@russselm` 分享了一个他们作为参考使用的 [notebook](https://github.com/brevdev/notebooks/blob/main/mistral-finetune.ipynb) 的 GitHub 链接。 
- **理解 MoE (Mixture of Experts)**：用户 `@russselm` 请求了解 MoE 的资源。用户 `@random_string_of_character` 推荐了几个资源，包括：[Mixture of Experts](https://lilianweng.github.io/posts/2021-09-25-train-large/#mixture-of-experts-moe)、[MoE 阅读小组](https://docs.google.com/document/d/12CR7jLJNA4vuFvvWjRZIG6_dcbArZT4kM5193XPUjZc/edit) 以及一个 [关于 MoE 的 YouTube 播放列表](https://youtube.com/playlist?list=PLvtrkEledFjoTA9cYo_wX6aG2WT5RFBY9&si=vO8sItJIGbpfYitU)。
- **StripedHyena-Nous-7B 和 LLamachop 实现**：用户 `@yobibyte` 发起了关于新架构 StripedHyena-Nous-7B 的讨论。为了兼容 Hugging Face Transformer，必须对 Llamachop 的建模代码进行更新。 
- **Mixtral 8x7B 的 VRAM 需求**：`@gerred` 发起了关于运行 Mixtral 8x7B 巨大的 VRAM 需求的讨论。有人提到 bitsandbytes 的创作者 Tim Dettmers 声称他可以在 5GB RAM 中运行 Mixtral 8x7B。 
- **Encoder 中的位置编码**：`@ex3ndr` 分享了他们对 Encoder 中位置编码（position encodings）的使用和应用的困惑，特别是与音频 token 相关的部分。讨论集中在他们对编码过程如何运作的理解，以及在整个编码过程中由此产生的潜在问题。
- **LLM 损失函数中的阶梯状行为**：`@nmg4914` 分享了一篇关于 Large Language Models (LLMs) 中异常训练行为的 [博客](https://www.fast.ai/posts/2023-09-04-learning-jumps/)，并询问其他人是否能在他们的实验中复现这些发现。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- 讨论了 **Google 在 AI 进步中的参与**及其优劣势，并对 Google 与 Atari 和 Kodak 等公司的过往记录发表了看法；还提到了该公司在 Artificial Superintelligence 方面的工作。提到了核心作者因 Google 未能将研究成果转化为实际应用而离职。
- 围绕 **GPT-4 访问**的使用问题和技术挑战，主要问题包括速度变慢、登录困难以及消息数量限制。提供了**浏览器建议**以解决网络错误，并处理了**账号恢复**咨询。
- 探讨了 **ChatGPT 的效用与性能**，包括对 Bard/Gemini Pro、GPT-3.5 和 GPT-4 鲁棒性的对比辩论。提出了关于持续用户验证以及随时间推移实用性下降的担忧。
- **提示词工程（Prompt Engineering）**的不同方法，例如使用 "show-and-tell"、EmotionPrompt 技术、Strunk 和 White 的《风格要素》（Elements of Style）等风格指南，或对角色特征进行显式详细描述，旨在塑造引人入胜且独特的 AI 输出。
- 讨论了 **API 相关问题与策略**，包括处理重复短语、模拟工作面试以及操纵指令以引导 AI 行为。强调了清晰的用户理解和明确的需求，以获得有效的 AI 响应。
- 关于 **DALL·E 使用**的对话，推荐了 MS Designer Create 和 Bing Image Creator 中的 DALL·E 功能。澄清了 ChatGPT 内部（特别是针对 Plus 订阅者）的 DALL·E 实现。
- 围绕 **Unified Neural Alignment (UNA)** 和**自定义 GPT 助手**的提问与建议，反映了对各种 OpenAI 技术和功能的兴趣。然而，没有关于 UNA 技术的解答。
- 提到了文件读取中**分析器下拉菜单**的移除，以及对自定义 GPT 配额限制和 AI 辅助内容审核的担忧。


**OpenAI 频道总结**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (54 条消息🔥): 
        
- **Google 的表现**：`@thewizzard___` 指出，尽管 Google 拥有强大的研究团队，但在社交媒体、手机、AI 和桌面操作系统等多个领域都有许多失败的产品。该用户将 Google 与 Atari 和 Kodak 进行了比较，认为这些公司未能将其行业地位转化为长久的成功。
- **DALL-E 的使用**：用户 `.ggvoid` 询问了 DALL·E 的用法，`@rjkmelb` 建议使用 Bing Image Creator，`@i_am_dom_ffs` 推荐了 MS Designer Create 或 Bing Create，两者都具备 DALL·E 功能。
- **Unified Neural Alignment (UNA)**：`@readyplayeremma` 询问了关于解释几种公开可用 AI 模型中所使用技术的出版物。对此没有收到任何回复。
- **Bard/Gemini Pro vs GPT-3.5** & **GPT-4**：`@thepitviper` 认为 Bard/Gemini Pro 似乎比 GPT-3.5 更好，而 Gemini Ultra 可能与 GPT-4 旗鼓相当。`@zyrqlo` 表示，目前的体验显示 GPT-4 优于 Bing 或 Bard，但预测如果问题得到解决，Gemini 可能会超越 GPT。`@bambooshoots` 强调，与 OpenAI 相比，Google 在 AI 模型开发方面大幅落后。
- **Google 在 AI 进步中的参与**：`@zyrqlo` 指出 Google Deepmind 正在研究 Artificial Superintelligence，这可能显著优于任何现有的 AI。然而，`@bambooshoots` 表示，《Attention is all you need》论文的核心作者离开了 Google，因为 Google 缺乏将研究转化为实际行动的动力。
- **续写 AI 生成的故事**：`@spectre120` 询问了续写其 AI 生成故事的 AI 推荐，因为对 ChatGPT 感到沮丧。`@the_only_alexander` 回应建议需要改进用户故事的方向。

### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (106 条消息🔥🔥): 
        
- **账户相关问题**：`@dawnx.` 询问如何更改与其 OpenAI 账户绑定的 Discord Account。`@satanhashtag` 建议通过私信 modmail 寻求帮助。
- **关于 GPT 版本和功能的讨论**：包括 `@eksynn`、`@jessicant.`、`@satanhashtag`、`@sooswastaken` 在内的用户对未来 GPT 版本的潜在发布和定价，以及对现有版本的影响进行了推测。
- **消息数量限制**：`@mrcrack_`、`@tariqali`、`@bad3r`、`@【ｐｅｎｕｌｔｉｍａｔｅ】`、`@eskcanta`、`@ragnarlothbrok` 等人讨论了 GPT-4 在单位时间内允许的消息数量限制，并将其与其他版本进行了比较。
- **性能和可用性问题**：`@luculentlady` 和 `@mrcrack_` 报告称遇到了 ChatGPT 变慢和访问困难的问题，`@satanhashtag` 建议这可能发生在高峰使用时段。
- **GPT-3 和 GPT-4 的质量**：`@ragnarlothbrok` 和 `@offline` 讨论了观察到的 GPT-3 和 GPT-4 回答质量下降的问题，包括回答质量莫名其妙的倒退以及随时间推移实用性降低的情况。


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (55 条消息🔥🔥): 
        
- **访问 GPT-4**：一些用户（如 `@signate` 和 `@pr0xymo`）反映尽管是付费客户，但在访问和使用 GPT-4 时遇到了问题。他们指出的问题包括自 11 月以来无法访问程序、浏览器卡死、响应时间慢以及“stop”命令失效。 
- **浏览器推荐**：在解决浏览器相关问题时，针对 `@maguiresfuture` 在使用 Chrome 时遇到的网络错误，`@rjkmelb` 建议尝试使用 Firefox 等其他浏览器。 
- **账户恢复和计费问题**：用户 `@gprapcapt3l` 询问是否可以在注销后恢复账户，`@rjkmelb` 回复称不可以。该用户还担心账户注销后仍会产生费用。`@iversusai` 在成功续订 Plus 订阅后仍无法访问 GPT-4，`@rjkmelb` 建议通过 OpenAI support 进行申诉。
- **DALL·E 的使用和计费**：`@life_9999` 询问了使用 ChatGPT Plus 和 DALL·E 的费用，`@solbus` 澄清说 ChatGPT Plus 每月费用为 20 USD，但用户可以商业化使用 DALL·E 3 生成的图像。通过 Bing 的图像创建器可以免费访问 DALL·E，但商业使用政策有所不同。
- **自定义 GPT Assistants 和 Attachments 功能**：用户 `@killymbapps` 提出了关于在自定义 GPT assistants 中使用和实现 “Attachments” 功能的问题，特别是关于如何对附件进行 Prompt 和结构化。讨论中未给出答案。

### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (32 条消息🔥): 
        
- **读取文件中的分析器下拉菜单**：`@zeriouszhit` 询问了文件读取中分析器下拉菜单被移除的问题。该工具被认为有助于判断 AI 是否在避免完整读取文件。
- **训练 GPT - 每条回复中遵循的步骤**：`@happyg` 讨论了构建 Custom Instructions 与 Knowledge 的最有效方式。他们提出了一种方法，即要求 GPT 根据特定指令重写 Prompt，然后按照规范进行回复。
- **使用 GPT-4 进行内容审核**：`@theunknown7` 询问了使用 GPT-4 进行内容审核的问题，`@solbus` 建议使用 OpenAI 的 API moderations 端点。讨论进一步探讨了在 OpenAI 使用政策下管理自定义规则的困难。
- **Custom Actions 与 Trello API 的问题**：`@hachuman` 寻求将 Trello 的 REST API 集成到 GPT 中的帮助，并在从 Swagger 导入完整 Schema 时遇到了问题。
- **ChatGPT 的用户验证**：`@yuriy700` 在使用 ChatGPT 时频繁遇到用户验证提示。`@readyplayeremma` 建议这可能是由浏览器插件或 VPN 引起的。
- **GPT-4 限额**：`@karajan` 提出了关于自定义 GPTs 限制的担忧。`@thepitviper` 澄清说，自定义 GPTs 的限额是 25条/3小时，而使用原生 GPT-4 则允许额外的 Prompt，限额最高为 40条/3小时。
- **在 ChatGPT 中使用 Dall-E**：`@life_9999` 询问了在 ChatGPT 中使用 Dall-E 的情况。`@pietman` 澄清说，这仅对 GPT Plus 订阅者开放。
- **使用 RAG 触发 GPT 中的搜索**：`@a1vx` 询问了如何指示 GPT 使用 RAG 搜索其知识文件。
- **ChatGPT 回复中的用户数据保护**：`@jobydorr` 分享了 ChatGPT 拒绝涉及个人信息请求的经历。他们询问拒绝转录 Instagram 用户名或电子邮件地址是否是新实施的规定。
- **Dall-E 图像与 API Actions 的集成**：`@chandan8764` 建议将 ChatGPT UI 中由 Dall-E 生成的图像发送到 GPT 内部的某些 API action 路由。


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (40 条消息🔥): 
        
- **对话创作的“Show-and-tell”技巧**：`@tryharder0569` 讨论了一种使用“show-and-tell”技巧创建引人入胜对话的方法，即在不直接陈述的情况下暗示细节。该方法侧重于使用比喻、隐喻和表现力强的语言来含蓄地展示信息的效果。([来源](https://discord.com/channels/974519864045756446/1079083340637941760))
  
- **提升输出质量的 EmotionPrompt 技术**：`@madame_architect` 提到了在给 AI 的指令中加入情感利害关系或暗示可能带来的益处。研究表明，特别是在情感化回复的范围内， AI 往往在产生目标结果方面表现得更好（来源：论文《Large Language Models Understand and Can Be Enhanced by Emotional Stimuli》）。

- **将 Strunk and White 的《风格要素》用于 AI 写作**：`@laughteronwater` 建议根据 Strunk and White 的《风格要素》（Elements of Style）来引导 AI 的写作风格，使用类似于《国家地理》、《科学美国人》或《大众科学》杂志的语调。用户警告不要使用陈词滥调或口语。

- **Emoji 对 Token 的影响**：`@pythoncodrr` 警告说，在 AI 输出中添加 Emoji 可能会消耗比预期更多的 Token，因为一个 Emoji 可能对应 2-8 个 Token。

- **RPG 风格对话的行为引导 Prompt**：`@eskcanta` 讨论了使用特定角色 Prompt 来导向 AI 行为的有效性。通过指示 AI “扮演一个带有阴暗秘密的黑色电影侦探”，可以产生忧郁且神秘的对话。用户强调了在编写 Prompt 时给出明确指令、准确描述性格特征以及明确要求的重要性。

### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (40 messages🔥): 
        
- **避免重复使用相似短语**：`@Ted` 询问了关于如何防止 GPT 多次重复使用相似短语的建议。`@mysticmarks1` 建议融合知名作者的写作风格，甚至谈到了模仿特定时代、时期和言语障碍的能力，以实现独特的写作风格。

- **行为引导**：针对 `@Ted` 关于实现独特文本的问题，`@mysticmarks1` 强调了给 AI 良好的行为引导以避免短语重复的重要性。讨论强调了使用特定的性格特征（如反派角色）来限制并多样化 AI 的词汇量。

- **技术问题**：`@laughteronwater` 报告了在使用某些符号创建表格和节奏记号时，ChatGPT 系统出现的问题。他们还讨论了希望限制模型输出中的陈词滥调和口语，并提到了他们的自定义指令，以获得更专业和学术的写作风格，类似于《国家地理》或《科学美国人》杂志。

- **模拟真实的求职面试**：`@eudk` 和 `@tryharder0569` 讨论了如何提示 AI 模拟真实的求职面试。`@tryharder0569` 建议在指令中指定某些行为特征，例如做一个“严厉的面试官”。

- **自定义指令**：`@laughteronwater` 和 `@tryharder0569` 讨论了避免 AI 回复中出现陈词滥调的策略。他们尝试了不同的指令来改善 AI 的直接性，`@tryharder0569` 建议使用 "show-not-tell"（展示而非叙述）的语言。

- **为 AI 角色命名**：`@eskcanta`、`@madame_architect` 和 `@tryharder0569` 讨论了为 AI 模型赋予特定角色、性格和动机的有益效果，以更有效地引导其语言和回复风格。

- **情感提示词**：`@madame_architect` 记录了在提示词中进行情感操纵的优点，并引用了一篇名为 "Large Language Models Understand and Can Be Enhanced by Emotional Stimuli" 的学术论文。

- **对 AI 的看法**：`@eskcanta` 主张用户需要清楚地了解自己想从 AI 那里得到什么，并准确地进行沟通以获得有效结果。讨论强调了在提示 AI 时需要清晰的沟通，以避免“魔幻思维”或不当情感操纵的风险。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- 关于 **Mixture of Experts (MoE) 范式和 Switch Transformer** 的教育性讨论。用户承认其复杂性，并讨论了其解决 VRAM 限制和加速 AI 训练的潜力。关于在批处理时是否加载所有专家存在显著分歧，这可能会影响整体 VRAM 容量。分享了一些关于该主题的视频和资源以供进一步学习。
- 持续讨论 **HuggingFace hub 上的几个数据集** 以及一个名为 [Megablocks-Public](https://github.com/mistralai/megablocks-public) 的 GitHub 项目，该项目对公众贡献开放。这些资源的好处伴随着加载问题的报告。此外，还有成员之间的微调进度更新和信息共享，对扩大词汇量及相关实验结果的兴趣，以及对 Grok 的 LLM 微调过程的批评。
- 关于 **AI 模型的开发、训练和精炼** 的许多观点，特别关注 Mixtral 和 qLoRA。分享了关于社区贡献的代码更新、训练期间的 VRAM 使用情况，以及在 Transformers 库中遇到检查点保存问题的见解。这些问题后来在 HuggingFace 的 GitHub 上进行了讨论。
- 讨论了用于从 PDF 脚本中**提取文本以用于机器学习的工具**，将 PyMuPDF 与 Apache Tika™ REST 服务等其他解决方案进行了比较，并请求推荐。分享了一个指向 [GitHub 上的 Tika-Python](https://github.com/chrismattmann/tika-python) 的链接，以获得更好的提取结果。
- 分享了关于使用 Axolotl 的 readme 中支持的工具**将超大 PyTorch 模型转换为 safetensor 格式**的指南。建议包括对模型文件使用 "axolotl.cli.shard" 以简化脚本创建。
- `@propback` 更新了针对多 GPU 推理过程中报告的 **nccl 相关问题** 的排查进度。然而，目前尚未收到团队的进一步更新。

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (47 条消息🔥): 
        
- **混合专家 (MoE) 范式**：`@noobmaster29` 分享了一个教学 [YouTube 视频](https://youtu.be/U8J32Z3qV8s?si=V2qPqASNUjr2N-FM)，解释了 *混合专家 (MoE) 范式和 Switch Transformer*。他们还提到这个概念比 *集成模型 (ensemble model)* 更复杂。
- **HuggingFace Hub 上的数据库**：`@noobmaster29` 提供了 HuggingFace 网站上两个不同数据库的链接。一个是名为 [FreedomIntelligence/evol-instruct-japanese](https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-japanese) 的日语数据集。另一个名为 [sharegpt-japanese](https://huggingface.co/datasets/FreedomIntelligence/sharegpt-japanese?row=0)，但遇到了加载问题。此外，`@noobmaster29` 还分享了 GitHub 上的 [Megablocks-Public](https://github.com/mistralai/megablocks-public)，这是一个开放公众贡献的项目。
- **为医疗 AI 模型命名**：`@yamashi` 征求医疗模型命名的建议。`@noobmaster29` 建议使用 *Viper*，参考了医学中使用的蛇形符号，其他建议包括 *Internist.ai v0.1* 和 *Amoxitron*。`@nanobitz` 建议使用药用植物的名字。
- **微调 (Fine-Tuning) 讨论**：新成员 `@joshuasundance` 表示对学习微调感兴趣。`@yamashi` 澄清说该话题仍在取得进展，并提到在 Hugging Face 上发布了一个微调模型，但它可能基于复制粘贴的信息。
- **混合专家 (MoE) 与 VRAM 讨论**：`@nafnlaus00` 建议 MoE 模型可以解决 VRAM 限制并加速 AI 训练和推理。`@yamashi` 表示反对，称 *在进行批处理 (batching) 时，你不可避免地会一次性加载所有专家*。


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (135 条消息🔥🔥): 
        
- **DiscoReseach/mixtral-7b-8expert 更新**：`@caseus_` 分享了关于 `@bjoernp` 在该项目上代码更改的更新链接，详见 [此处](https://huggingface.co/DiscoResearch/mixtral-7b-8expert/blob/main/modeling_moe_mistral.py)。
 
- **使用 qLoRA 微调 Mixtral**：`@faldore`、`@yamashi`、`@bjoernp` 和 `@casper_ai` 讨论了使用 qLoRA 微调 Mixtral 及其有效性。`@bjoernp` 表示这应该是可行的，因为 qLoRA 本质上是带有路由器的标准 Mistral 架构。

- **Mixtral 训练与 VRAM 占用**：`@faldore` 分享了他调整 Mixtral 模型的经验和挑战。他报告说该模型可以在 4x A100 80gb GPU 和 8k 序列长度下工作，但他不得不将序列长度减少到 4096。

- **扩展模型词汇表**：`@seungduk` 分享了他通过微调为新添加的 token 扩展模型词汇表的实验。他分享了代码段链接 [此处](https://github.com/yanqiangmiffy/GoGPT/blob/ec2b9de8df73621745f8bc0e8908ccbb163aa359/backup/llama1/step2_train_pt.py#L642)，并提到在训练新添加 token 的嵌入 (embeddings) 时，不会损害预训练模型。

- **Transformer Checkpoint 保存问题**：`@faldore` 报告了在 dolphin-mixtral 训练过程中保存第一个 Checkpoint 时遇到的 Transformers 问题。`@caseus_` 分享了一个链接 [此处](https://github.com/huggingface/transformers/issues/27925)，指向 HuggingFace GitHub 上的同一个 issue。


### ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/) (1 条消息): 
        
- **Grok LLM 微调批评**：`@nafnlaus00` 批评了 Elon Musk 的 "Grok" LLM 的微调过程，称负责人 **没有从中剥离 OpenAI 的提示词 (prompts)**。


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (11 条消息🔥): 
        
- **将大型 PyTorch 模型转换为 safetensors**：`@.wooser` 询问如何将 14GB 的大型 `pytorch_model.bin` 转换为更小、易于管理的 `safetensors` 文件，以确保用户安全。`@nanobitz` 建议查看 Axolotl 的 readme，它支持转换过程。他们建议 `@.wooser` 加载模型，然后设置配置以保存为 safetensor 格式。为了帮助 `@.wooser` 简化脚本创建，`@nanobitz` 还推荐对模型文件使用 `axolotl.cli.shard`。

### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (10 messages🔥): 
        
- **PDF 转文本脚本推荐**：用户 `@visuallyadequate` 发起了一场讨论，寻求能够从 PDF（主要是机械手册）中提取原始文本的库或脚本的推荐。
- **PyMuPDF 与其他工具**：`@visuallyadequate` 分享了他们一直在使用 **PyMuPDF**，效果尚可；而 `@noobmaster29` 也提到尝试过不同的解决方案，但仍在寻找完美的工具。
- **Tika-Python 推荐**：`@nruaif` 推荐了 **Tika-Python**，这是 Apache Tika™ REST 服务的 Python 绑定。据 `@nruaif` 反馈，该工具的效果优于 PyMuPDF。该工具的链接为：[https://github.com/chrismattmann/tika-python](https://github.com/chrismattmann/tika-python)。


### ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/) (1 messages): 
        
- **排除 NCCL 相关问题**：`@propback` 提到他们目前正在解决多 GPU 推理过程中的 **NCCL 相关问题**，这可能有助于解决当前背景下的问题。他们还指出，目前尚未收到团队关于此问题的任何更新。


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord 摘要

- 关于各种模型（如 Lora 4-bit 和 HuggingFace LLM）、逻辑回归以及 OpenAI 的 XABERIUS-34B beta 的操作/聚合方面的查询。用户还就 ElevenLabs、Unity Inference 和 Neovim llm 插件等工具和 API 寻求建议。具体话题包括将 safeTensor 输出转换为 gguf 格式、sidhu moose wala 模型移除、在 HuggingFace 上使用 ElevenLabs 的问题，以及设置“官方” Neovim llm 的困难。
- 几位成员就不同的技术方面寻求见解，包括创建四维 Gaussian Splat、解决 TensorFlow-gpu v1.15 与 NVIDIA GeForce RTX 4090 之间的兼容性问题、通过 Gradio API 获取图像，以及将本地语言模型（Local Language Models）集成到应用中的有效方法。分享的资源包括 [GitHub issue 讨论](https://github.com/tensorflow/tensorflow/issues/62002) 和一段关于 [Mamba transformers 的视频](https://youtu.be/ouF-H35atOY)。社区提出的一个解决方案是使用 ONNX 来提高本地语言模型的便携性。
- 用户展示了自主开发的项目，包括 SDXL Transfer Style 演示、Web3 API Gateway、带有 AI 的 Discord 机器人、XABERIUS-34B-UNA 模型以及 Overall V1 Model。这些项目的创建者请求社区提供反馈、建议和测试；并分别分享了项目链接。
- 社区讨论围绕高分辨率图像数据集的利用以及 Google 的 Gemini 多模态系统的性能展开。建议利用深度模型（depth models）、point-e 模型，以及 Transformers 库中提供的 LLaMa 和 Mistral-7b 的多模态扩展，并链接到了特定模型，如 [3D-Room-Layout-Estimation_LGT-Net](https://huggingface.co/spaces/zhigangjiang/3D-Room-Layout-Estimation_LGT-Net) 和 [LLaVa](https://huggingface.co/llava-hf)。


**HuggingFace Discord 频道摘要**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (57 条消息🔥🔥): 
        
- **微调 Lora 4bit 模型**：用户 `@.dafa` 寻求帮助，希望将微调后的 Lora 4bit 模型的 safeTensor 输出转换为 gguf 格式。该用户还反映没有获取到 `adapter_model.bin`，只得到了 safeTensor。
- **ElevenLabs 使用问题**：`@felipegabriel1995` 表达了在 Hugging Face 上使用 ElevenLabs 的困难，并询问是否有任何变动或停止服务的计划。
- **模型移除请求**：`@RAJDEEP SINGH` 请求从 Hugging Face 网站上移除 sidhu moose wala 模型，这是应模型制作者父母的要求。他提供了一个 YouTube 链接作为证明 (https://www.youtube.com/shorts/v7ZAGyFY_20?feature=share)。
- **Neovim llm 插件与 HuggingFace LLM API 问题**：`@zalasur` 询问如何配置“官方” Neovim llm 插件以配合 HuggingFace LLM API 或本地运行的模型使用。该用户还报告在 HuggingFace 平台上使用 inference API 时遇到 500 错误。
- **语音转文本工具咨询**：`@starkroyale` 询问是否有可以将语音转换为文本的工具。该用户表现出对更好地理解歌词的兴趣。
- **Unity Inference APIs 中的语言配置**：`@pyl29` 寻求在 Unity inference APIs 中更改语言设置的帮助。`@doctorpangloss` 建议用户可能需要针对其官方端点为 Unity 生成 openapi 客户端。
- **逻辑回归资源**：`@gabdos` 分享了一个关于 Logistic Regression（逻辑回归）的 YouTube 视频链接 (https://youtu.be/ux12Lj8gXZ0)，并称其为该主题的综合资源。


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (7 条消息): 
        
- **4维高斯泼溅 (4 Dimensional Gaussian Splat) 教程请求**：`@sims_` 寻求学习如何创建 4维高斯泼溅的教程或课程。目前尚未收到回复或解决方案。
- **TensorFlow-gpu v1.15 与 NVIDIA GeForce RTX 4090 兼容性问题**：`@hussain_muhammed` 在 NVIDIA GeForce RTX 4090 上运行 tensorflow-gpu 1.15 版本的代码库时遇到了 cuBLAS 错误。怀疑是 Tensorflow 版本与 GPU 版本之间的兼容性问题，并请求协助。
- **TensorFlow 兼容性问题解决方案**：`@tryharder0569` 建议问题可能是由于版本不匹配造成的。他们建议 `@hussain_muhammed` 启动一个新的 conda 环境并从头开始重新安装所有内容。他们还分享了一个相关的 [GitHub issue](https://github.com/tensorflow/tensorflow/issues/62002) 链接，可能有助于解决该问题。
- **学习 Mamba Transformers**：`@caleb_sol` 提到他们正在学习 Mamba transformers，并分享了一个标题为 "Mamba - a replacement for Transformers?" 的 [YouTube 链接](https://youtu.be/ouF-H35atOY)。


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (1 条消息): 
        
fblgit：隆重推出.. Xaberius 34B，排名第 1 的 LLM 🙂 而且这还只是 beta 版... 最弱的一个 checkpoint 🙂


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (7 条消息): 
        
- **SDXL Transfer Style Demo 创建**：用户 `@tonic_1` 宣布创建了一个 **SDXL Transfer Style demo**，并在此分享了项目链接 [here](https://huggingface.co/spaces/Tonic1/TonicsStyleAlign)。他们邀请社区提供建议和 PRs。

- **Web3 API 网关**：`@dsimmo` 讨论了一个利用 Web3 技术提供 API 无缝货币化的项目。该系统允许高吞吐量，每用户限制高达每秒 50 次请求，且仅增加 400ms 的响应时间。该项目的官方网站可以在[这里](https://dhali.io)找到。

- **使用 AI 和猫娘创建 Discord 机器人**：用户 `@devilin_` 创建了一个集成了开源语言模型的 Discord 机器人，并提供多种交互模式，例如能够同时询问所有模型并比较结果。该机器人还包含 **DAN 模式**。机器人可以在[这里](https://top.gg/bot/1094198651846414336)找到。

- **XABERIUS-34B-UNA 介绍**：`@fblgit` 介绍了一个新模型 **XABERIUS-34B-UNA**，解释说该模型表现出预训练/基础模型（pretrained/foundational）行为，并邀请用户尝试。

- **新 Overall V1 模型发布**：`@dak.off1` 宣布发布了一个新模型 **Overall V1**，该模型基于 **SD-1.5** 训练，具有生成出色图像的能力。该模型拥有 .CKPT、.SAFETENSORS 和 .ONNX 格式的权重。模型可以在[这里](https://hf.co/openskyml/overall-v1)下载和测试。

### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (3 messages): 
        
- **高质量图像数据集查询**：用户 `@jfischoff` 询问是否有人知道高分辨率、高质量的图像数据集，最好是规模较小的。
- **Gradio API 图像获取**：用户 `@_thunderlord` 询问如何使用 Gradio API 从特定路径 (tmp/gradio/...) 获取图像 (png 或 jpg)。
- **对 Gemini 的看法**：用户 `@yamayamakawa` 征求专家对 Google 新推出的 multi-model 系统 **Gemini** 的看法。


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (3 messages): 
        
- **深度模型推荐**：`@jo_pmt_79880` 建议查看深度模型或 point-e 模型，并分享了 Hugging Face Spaces 上的 [3D-Room-Layout-Estimation_LGT-Net](https://huggingface.co/spaces/zhigangjiang/3D-Room-Layout-Estimation_LGT-Net) 链接。
- `@n278jm` 对提供的信息表示感谢，认为非常有用。
- **LLaVa 和 BakLLaVa 模型发布**：`@nielsr_` 宣布 Transformers 库中已支持 LLaVa 和 BakLLaVa（分别是 LLaMa 和 Mistral-7b 的多模态扩展），并附带了 Hugging Face 上的 [LLaVa 模型](https://huggingface.co/llava-hf) 链接。该用户还分享了一个 [demo notebook](https://colab.research.google.com/drive/1_q7cOB-jCu3RExrkhrgewBR0qKjZr-Sx#scrollTo=PuWVAAOinC8q) 的链接。


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (4 messages): 
        
- **将本地语言模型集成到 App 中**：用户 `@sayingwhateverr` 询问有关将 **本地语言模型 (LLMs)** 集成到应用程序（特别是 Flutter 或 Web 应用）中的资源，旨在通过提供数据洞察和建议来增强用户体验。该用户寻求一种既不需要终端用户设置 LLMs，也不需要用户对其有深入了解的解决方案。
- **Localhost vs 捆绑模型**：`@sayingwhateverr` 还提到，目前大多数教程都指导如何暴露 localhost 供应用检查，但更倾向于将所有内容都集成在应用本身中。
- **模型的资源考量**：提到首选那些不会消耗太多资源的模型。
- **ONNX 作为可能的解决方案**：`@vipitis` 建议使用 **ONNX** 作为便携性的潜在解决方案。


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (3 messages): 
        
- **高分辨率图像数据集**：用户 `@jfischoff` 询问是否有人了解 **高分辨率、高质量的图像数据集**。未指定该数据集的具体要求或使用场景。
  
- **Gradio API 图像获取**：用户 `@_thunderlord` 想知道如何通过 Gradio API 获取图像，特别是位于 'tmp/gradio' 路径下的图像。

- **Google 的 Gemini 多模态模型**：用户 `@yamayamakawa` 征求有关 **Google Gemini**（其新的多模态 AI 项目）的专家意见。社区对该查询的回复未包含在提供的对话中。


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 摘要

- 介绍了个人（'**teknium**'）并在 '**general-chat**' 频道分享了链接。
- 在 '**oo**' 频道中关于 **Mixtral 和 Mamba** 的广泛对话，`*@Teknium*` 分享了相关的 Twitter 链接。讨论围绕 Mixtral 与 **Mistral 7b** 模型的比较展开。`*@teknuim*` 提到了一个 **AI 见面会 (meetup)**。
- 在 '**oo2**' 中，有一些关于**客厅装饰**的建议，包括 `@ufghfigchv` 分享的用*巨型白板*代替电视的想法。此外，`@gabriel_syme` 将一个图表描述为*半成品 (half-cooked)*，并提议通过变异 system prompts 来修改交互。
- `@danfosing` 在 '**general-chat**' 频道对 '**Gemini Ultra & Bard Advanced 计划**' 发表了评论，指出 Gemini Ultra 将包含在付费的 Bard Advanced 计划中。


**Alignment Lab AI 频道摘要**

### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (4 messages): 
        
- **介绍**：`@teknium` 在频道中打招呼。
- **链接分享**：`@teknium` 分享了一个 [Twitter 链接](https://fxtwitter.com/Teknium1/status/1733233296962953567)。
- **模型缩放讨论**：`@rusch` 评论了 AI 模型的扩展性，提到了 MoE (Mixture of Experts) 的可能性，可能类似于 Mistral 模型。
- **Gemini Ultra & Bard Advanced 计划**：`@danfosing` 分享称 **Gemini Ultra** 将包含在 **付费 Bard Advanced 计划** 中。

### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (18 messages🔥): 
        
- **关于 Mixtral 和 Mamba 的对话**：`@Teknium` 分享了一个[链接](https://fxtwitter.com/Teknium1/status/1733233296962953567)，重点介绍了 **Mixtral** 的工作，这是一种 Transformer 和类 Mamba 架构的混合体。然而，它尚未实现线性扩展（linear scaling）。`@Alpindale` 作出回应，表示计划在明年发布他们自己的线性架构以及预训练模型。
- **Mixtral 与 Mistral 的比较**：`@Teknium` 指出 **Mixtral** 与 **Mistral 7b** 的表现非常接近。
- **AI Meetup 引用**：`@Teknium` 提到 `@1090682143425966100` 在最近的 a16z 开源 AI 聚会上向 `@410352626421465089` 表示了致敬。


### ▷ #[oo2](https://discord.com/channels/1087862276448595968/1176548760814375022/) (5 messages): 
        
- **渴望在客厅放一块大白板**：`@teknium` 表示需要在他们的客厅里放一块大白板。
- **图表状态**：`@gabriel_syme` 提到一个图表，称其为**半成品**（half-cooked）。
- **脚手架想法**：`@gabriel_syme` 建议，通过变异系统提示词（system prompts）作为**变异交互**的一种方式，可能会有一些有趣发现。
- **客厅装饰创意**：`@ufghfigchv` 分享了一个想法，认为客厅应该安装**巨大的白板**而不是电视。


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- 宣布 **LangChain 新版本**发布，langchain-core==0.1，为 langchain-community 做准备。用户 `@hwchase17` 确认了向后兼容性，并鼓励用户反馈任何问题。最新版本可以通过 `pip install langchain==0.0.349rc2` 安装。此外，`@hwchase17` 为发现回归问题（regressions）的用户提供免费的 LangChain 周边礼品。
- 关于 **LangChain 序列化问题**的持续讨论，用户 `@b0otable` 提出了序列化方面的挑战，强调了输出解析器和内置序列化方法的局限性，并建议将 `json dumps` 作为目前的最佳解决方案。
- 用户 `@p4y4` 询问了关于 **Langsmith 访问权限**的问题，而 `@nagar502` 寻求关于**利用自定义 Love Language Model (LLM) 进行流式传输**的帮助。
- 用户 `@seththunder` 提出了一个关于在 **ConversationalRetrievalChain 中使用 `.arun`** 的问题。
- 用户 `@daemon966` 在多个频道重复分享了一则**招聘广告**，并提供了潜在申请者的 Discord 服务器链接：[https://discord.gg/cryptojob](https://discord.gg/cryptojob)。

**LangChain AI 频道总结**

### ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/) (1 messages): 
        
- **LangChain 新版本发布**：用户 `@hwchase17` 宣布发布 **langchain-core==0.1**，为 **langchain-community** 做准备。这个新版本是**向后兼容**的，但希望用户能标记出任何问题。最新版本可以通过 `pip install langchain==0.0.349rc2` 安装。
- 该用户还为发现任何回归问题的用户提供**免费 LangChain 周边**。


### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (9 messages🔥): 
        
- **LangChain 序列化问题**：用户 `@b0otable` 讨论了在 LangChain Web 应用中遇到的序列化挑战。重点提到了不可序列化的对象——例如 Documents 和 AIMessages。提到了输出解析器和内置序列化方法的局限性。不过，`json dumps` 被认为是目前发现的最佳解决方案。
- **Langsmith 访问请求**：用户 `@p4y4` 询问如何获得 Langsmith 的访问权限。
- **自定义 LLM 查询**：`@nagar502` 请求帮助利用自定义 Love Language Model (LLM) 进行流式传输，并提供了一段代码片段以获取反馈。该用户目前未能收到响应。
- **在 ConversationalRetrievalChain 中使用 .arun**：用户 `@seththunder` 提出了一个关于 `.arun` 是否可以在 ConversationalRetrievalChain 中使用的问题。
- **招聘广告**：`@daemon966` 向社区分享了一条招聘信息，并链接到了一个 Discord 服务器 ([https://discord.gg/cryptojob](https://discord.gg/cryptojob))。


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (1 messages): 
        
- **职位招聘**：在 LangChain AI Discord 聊天机器人中，用户 `@daemon966` 发布了一条招聘公告以及 [https://discord.gg/cryptojob](https://discord.gg/cryptojob) 的链接。他们已在 `langserve` 频道通知了所有人。


### ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/) (1 messages): 
        
- 用户 `@daemon966` 发布了关于**招聘**的公告，并分享了 [cryptojob Discord 服务器的邀请链接](https://discord.gg/cryptojob)，同时艾特了 **everyone** 和 **here** 以触达所有可能的参与者。

### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (1 messages): 
        
- **招聘公告**：`@daemon966` 正在招聘，并提供了一个 [职位 Discord 服务器链接](https://discord.gg/cryptojob)。提到要通知 `@everyone` 和 `@here`。


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages): 
        
- **工作机会**：用户 `@daemon966` 分享了一个工作机会，并附带了 LangChain AI 小组的申请链接，点击 [此处](https://discord.gg/cryptojob) 申请。为此公告召唤了所有人及当前群组成员。


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- 在 **opensource** 频道中，`@lhl` 发起了一场关于 [Llama-Mistral](https://github.com/dzhulgakov/llama-mistral) 的讨论。提到的关键话题包括在 2x80G 显卡上运行该模型、与 2x48G GPU 的潜在兼容性，以及一条展示了 Llama-Mistral 令人期待的初步结果的 [推文](https://fxtwitter.com/jphme/status/1733412003505463334)。

- 在 **speed** 频道中，用户们就 **Azure vs GPT-4** 的性能展开了辩论，并分享了个人经验。此外，`@laikhtewari` 分享了一篇 [博客文章](https://hf.co/blog/optimum-nvidia)，讨论了在 Hugging Face 上使用 Optimum-NVIDIA 以提高 LLM 推理速度。

- 在 **rag** 频道中，`@sandkoan` 触发了关于 **Claude** 模型能力的对话，讨论了不同序列长度和输入序列中上下文放置位置的影响。他们还强调了应用于 100k 版本的 Claude 和 8k 版本的 Mistral 模型时所采用的不同技术。

- 在 **offtopic** 频道中发现了一些缺乏上下文的对话，其中 `@res6969` 分享了一个 [链接](https://x.com/chatgptapp/status/1733569316245930442?s=46)，但没有提供额外的评论或背景信息。

**LLM Perf Enthusiasts AI 频道总结**

### ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (4 messages): 
        
- **在特定显卡上运行 Llama-Mistral**：`@lhl` 提到目前有人在 2x80G 显卡上运行 [Llama-Mistral](https://github.com/dzhulgakov/llama-mistral)。
- **运行 Llama-Mistral 所需的资源**：`@lhl` 还分享道，根据列出的要求，推理代码在 2x48G GPU 上运行可能也没问题。
- **Llama-Mistral 的初步结果**：`@lhl` 链接了一条 [推文](https://fxtwitter.com/jphme/status/1733412003505463334)，展示了使用 Llama-Mistral 取得的一些令人期待的初步结果。


### ▷ #[offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594/) (2 messages): 
        
- 用户 `@res6969` 分享了一个 [链接](https://x.com/chatgptapp/status/1733569316245930442?s=46)，没有任何额外的评论或上下文。
- 用户 `@res6969` 随后评论了 "lol"，同样没有进一步的上下文。


### ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/) (5 messages): 
        
**Azure vs GPT-4 性能对比**：

- `@nosa_.` 提到 **Azure 整体看起来更好**，但提醒这并不总是绝对的。
- `@res6969` 分享道，他们使用 **GPT-4 和 Azure 之间的切换系统**，以最大化速率限制并最小化延迟。
- `@wenquai` 声称 Azure 对他们来说几乎总是快 40-60%，不过他们也指出这可能取决于地理位置和 Azure 实例的配置。

**在 Hugging Face 上使用 Optimum-NVIDIA 实现快速 LLM 推理**：

- `@laikhtewari` 分享了 Hugging Face 的一篇 [博客文章](https://hf.co/blog/optimum-nvidia) 链接，解释了 Optimum-NVIDIA 如何通过仅一行代码的更改来实现快速 LLM 推理（1,200 tok/s，据称快了 28 倍）。他们还征求了对该博客文章的反馈。


### ▷ #[rag](https://discord.com/channels/1168579740391710851/1169086375686053890/) (3 messages): 
        
- **模型在不同序列长度下的注意力**：用户 `@sandkoan` 讨论了模型的有效性在很大程度上取决于其 **在不同序列长度下保持注意力的能力**。
- **输入序列中的上下文放置**：`@sandkoan` 解释说，如果查询（query）在上下文（context）之前给出，**Claude** 模型很可能会 **忘记查询**，因此上下文通常被放置在查询之前。
- **不同模型能力的差异**：`@sandkoan` 提醒道，适用于 **100k 版 Claude** 的规则不一定适用于 **8k 版 Mistral**。


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- 社区成员 `@tonic_1` 提供了一个使用 diffusers 库和 sdxl 生成风格化图像的 [工具演示](https://huggingface.co/spaces/Tonic1/TonicsStyleAlign/)，并邀请大家提供反馈和讨论。
- 用户 `@aardvarkoncomputer` 和 `@dimfeld` 详细讨论了 “Lazy” GPT，特别是其在 0613 API 中的出现情况。
- `@guardiang` 通过分享一条 [推文](https://x.com/aravsrinivas/status/1732825206023201273?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 庆祝 Perplexity AI 成立一周年，重点介绍了该公司优先发展搜索业务的决定。
- 关于 Mistral/Open Hermes 7B 微调的问题：`@.beowulfbr` 寻求相关的 Notebook 建议，而 `@btdubbins` 询问了所需的算力（compute）量。
- swyxio 在 #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) 中提到了关于 “[INST]” 的评论，尽管上下文有限。

**Latent Space 频道总结**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (6 条消息): 
        
- **tonic_1 的工具演示**：`@tonic_1` 分享了一个 [演示](https://huggingface.co/spaces/Tonic1/TonicsStyleAlign/)，该工具使用 diffusers 库和 sdxl 根据参考图像风格生成新图像。 
- **Lazy GPT**：用户 `@aardvarkoncomputer` 和 `@dimfeld` 讨论了 “Lazy” GPT，`@aardvarkoncomputer` 提到了 0613 API 中 “懒惰” 现象的发生率。 
- **Perplexity AI 周年庆**：`@guardiang` 分享了关于 Perplexity AI 成立一周年的 [推文](https://x.com/aravsrinivas/status/1732825206023201273?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)，重点介绍了 Aravind 关于公司专注于搜索决策的帖子。 
- **微调 Mistral/Open Hermes 7B**：`@.beowulfbr` 询问是否有可用于微调 Mistral/Open Hermes 7B 的 Notebook。
- **微调所需的算力**：针对之前的询问，用户 `@btdubbins` 询问了微调 Mistral/Open Hermes 7B 所需的算力。


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (1 条消息): 
        
swyxio: 这字面上就是 “[INST]”，是……的一部分。

        

---

## [Ontocord (MDEL discord)](https://discord.com/channels/1147858054231105577) Discord 总结

只有一个频道有活动，因此无需总结...

xa9ax: 谁都要去 NeurIPS 吗？
        

---
Skunkworks AI Discord 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。

---
MLOps @Chipro Discord 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。

---
AI Engineer Foundation Discord 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。

---
Perplexity AI Discord 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。

---
YAIG (a16z Infra) Discord 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。