---
companies:
- microsoft-research
- anthropic
- nvidia
- hugging-face
date: '2024-07-03T01:30:30.158799Z'
description: '以下是为您翻译的中文内容：


  **微软研究院（Microsoft Research）**开源了 **GraphRAG**，这是一种检索增强生成（RAG）技术。它通过从源数据中提取知识图谱并进行聚类，从而提升大语言模型（LLM）的回答质量，但同时也增加了
  Token 消耗和推理时间。**Gemma 2** 系列模型正式发布，专注于高效的小型大模型，并引入了滑动窗口注意力机制（sliding window attention）和
  RMS 归一化（RMS norm）等创新技术，其性能已接近规模更大的 **Llama 3 70B**。Anthropic 的 **Claude 3.5 Sonnet**
  在指令遵循和编程基准测试中处于领先地位，而**英伟达（Nvidia）**也在 6 月发布了 **Nemotron 340B** 模型。**Qwen2-72B（通义千问）**在
  HuggingFace 开源大模型排行榜上名列前茅，在数学和长程推理方面表现卓越。有关 RAG 的讨论重点关注了其局限性，以及如何通过函数调用来优化上下文的使用。一种角色驱动（persona-driven）的合成数据生成方法引入了
  10 亿个角色，基于此微调的 7B 规模模型在数学基准测试中达到了与 GPT-4 相当的水平。此外，拥有 200GB 数据的 **AutoMathText 数据集**在数学数据合成方面的作用也受到了关注。'
id: b5462156-6ed0-48d6-854b-9091926ceb0f
models:
- gemma-2
- llama-3-70b
- claude-3.5-sonnet
- nemotron-340b
- qwen2-72b
- llama-3
original_slug: ainews-graphrag
people:
- travis-fischer
- rasbt
- alexandr-wang
- osanseviero
- rohanpaul_ai
- hamelhusain
- svpino
- aaaazzam
- omarsar0
title: GraphRAG：知识图谱与 RAG 的结合（或：知识图谱与 RAG 的联姻）
topics:
- retrieval-augmented-generation
- knowledge-graphs
- token-usage
- inference-time
- attention-mechanisms
- instruction-following
- coding
- math
- long-range-reasoning
- synthetic-data
- dataset-release
- fine-tuning
- context-windows
- function-calling
---

<!-- buttondown-editor-mode: plaintext -->**KG 是 LLM 所需的一切。**

> 2024年7月1日至7月2日的 AI 新闻。
我们为您检查了 7 个 subreddit、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**419** 个频道和 **2518** 条消息）。
预计节省阅读时间（以每分钟 200 字计）：**310 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

神经符号（Neurosymbolic）的拥趸们欢呼吧！

 
![image.png](https://assets.buttondown.email/images/6ff5146a-947a-4da1-b7ec-79f1e753371c.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/8a486066-fa4b-4814-9132-82a435e39eb0.png?w=960&fit=max)
 

Microsoft Research [最初在 4 月份发布了 GraphRAG](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)，在上周 AI Engineer World's Fair 的 Neo4j 工作坊和演讲中，它[出人意料地受欢迎](https://x.com/altryne/status/1808218232861647171)（视频尚未上线，所以我们还没看到，但很快就会有了 (tm)）。他们现在已经[开源了代码](https://x.com/MSFTResearch/status/1808168761565798833)。正如 [Travis Fischer 所说](https://x.com/transitive_bs/status/1808218183809355887)：

1. 使用 LLM 从源数据中提取知识图谱（Knowledge Graph）
2. 将该图谱聚类为不同细节层级的相关实体社区
3. 对于 RAG，遍历（map）所有社区以创建“社区回答”，并进行规约（reduce）以生成最终答案。

或者用他们相对不那么通俗易懂的话来说：

 
![image.png](https://assets.buttondown.email/images/cb8ca75b-2559-4ce7-be16-87998f06a199.png?w=960&fit=max)
 


然而，这类性能提升技术都有一个公开的秘密：[Token 使用量和推理时间都会增加](https://x.com/donpark/status/1808232638878306429) 🙃

同样值得注意的还有：他们的 [Prompt 重写方法](https://x.com/yoheinakajima/status/1808214283471446309)

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**LLM 模型发布与改进**

- **Gemma 2 模型发布**：[@rasbt](https://twitter.com/rasbt/status/1807764328159850817) 指出 Gemma 2 模型在不增加数据集大小的情况下探索了新技术，专注于开发小型且高效的 LLM。关键设计选择包括 **sliding window attention、group-query attention 和 RMS norm**。Gemma 2 的表现几乎与体积大 3 倍的 Llama 3 70B 相当。
- **Anthropic 的 Claude 3.5 Sonnet 模型**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1807828398523249099) 报告称，Claude 3.5 Sonnet 在 ScaleAI 的隐藏评估中目前在 **Instruction Following 和 Coding 方面排名第一**。然而，与其他顶级模型相比，它在写作风格和格式方面失分较多。
- **Nvidia 的 Nemotron 340B 模型**：[@osanseviero](https://twitter.com/osanseviero/status/1807761331191189790) 分享了 Nemotron，这是一个 **340B 参数模型**，作为 6 月开源模型发布的一部分推出。 
- **Qwen2-72B 登顶 HuggingFace Open LLM Leaderboard**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1807769879526695153) 指出 Qwen2-72B 平均得分 43.02，在 **数学、长程推理和知识** 方面表现出色。有趣的是，Llama-3-70B-Instruct 在 GPQA 上的得分比其预训练版本低了 15 分。

**检索增强生成 (RAG) 技术与挑战**

- **RAG 基础演讲**：[@HamelHusain](https://twitter.com/HamelHusain/status/1807799818972262650) 分享了来自 LLM 大会的 RAG 基础演讲，**涵盖了核心概念和技术**。
- **RAG 的局限性**：[@svpino](https://twitter.com/svpino/status/1807748211076968899) 讨论了 RAG 的局限性，包括 **检索、长上下文窗口、评估以及配置系统以提供答案来源等方面的挑战**。
- **改进 RAG 中的 LLM 上下文使用**：[@AAAzzam](https://twitter.com/AAAzzam/status/1807788631135801415) 分享了一个让 LLM 更有效地使用上下文的技巧——**LLM 对来自函数调用的信息的利用远高于普通上下文**。将上下文转换为伪函数调用（pseudo function calls）可以改善结果。

**合成数据生成与应用**

- **角色驱动的数据合成**：[@omarsar0](https://twitter.com/omarsar0/status/1807827401122238628) 分享了一篇论文，提出了一种角色驱动的数据合成方法来生成多样化的合成数据。它引入了 **10 亿个多样化角色**，以促进创建涵盖广泛视角的模型。在 107 万个合成数学问题上微调的模型在 MATH 测试中达到了 64.9%，以 7B 的规模达到了 GPT-4 的性能。
- **AutoMathText 数据集**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1807911697429954732) 强调了 **200GB 的 AutoMathText 数据集**，包含用于预训练数学语言模型的数学文本和代码。该数据集由来自 arXiv、OpenWebMath 以及编程仓库/网站的内容组成。
- **数学能力的合成数据**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1807911688135397566) 指出一篇论文显示，在提高 LLM 的数学能力方面，**合成数据几乎与真实数据一样有效**。在合成数据上训练的 LLaMA-2 7B 模型在 GSM8K 和 MATH 基准测试中比之前的模型高出 14-20%。

**其他**

- **通过分块量化的 8-bit 优化器**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1807806733781200898) 回顾了 2022 年一篇关于 8-bit 优化器的论文，该优化器能 **保持 32-bit 优化器的性能**。关键创新包括分块量化（block-wise quantization）、动态量化和稳定嵌入层，以在不损失效率的情况下减少内存占用。
- **理解并减轻语言混淆**：[@seb_ruder](https://twitter.com/seb_ruder/status/1807849542538481999) 介绍了一篇分析 LLM 无法按用户要求的语言生成文本的论文。**Language Confusion Benchmark 衡量了 15 种语言的这一现象**。即使是强大的 LLM 也会表现出混淆，以英语为中心的指令微调会产生负面影响。论文提出了推理和训练阶段的缓解措施。
- **托管 LLM 价格对比**：[@_philschmid](https://twitter.com/_philschmid/status/1807790240599294169) 分享了来自不同供应商的托管 LLM 的最新价格表。核心洞察：**顶级闭源 LLM 每 1M 输出 tokens 约 15 美元，Deepseek v2 最便宜，为 0.28 美元/M，Gemini 1.5 Flash 性价比最高，Llama 3 70B 约 1 美元/1M tokens**。

---

# AI Reddit 摘要回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**LLM 开发与能力**

- **Gemma 2 27B 模型问题**：在 /r/LocalLLaMA 中，用户质疑 [**Gemma 2 27B 模型在最近的 llamacpp 更新中是否已损坏**](https://www.reddit.com/r/LocalLLaMA/comments/1dsqn30/gemma_2_27b_it_gguf_broken/)。对比显示其回答与 aistudio.google.com 相似。
- **规模化合成数据生成**：/r/singularity 讨论的一项新研究利用 [**10 亿个多样化人格（personas）集合来大规模创建合成数据**](https://www.reddit.com/r/singularity/comments/1dszq6h/the_problems_with_llms_and_the_paths_being/) 以训练 LLM，从而能够利用多种视角进行多功能数据合成。
- **LLM 比想象中更线性**：在 /r/MachineLearning 中，新研究揭示了 [**Transformer 解码器中近乎完美的线性关系**](https://www.reddit.com/r/MachineLearning/comments/1dso3pg/r_large_language_models_are_much_more_linear_than/)。移除或近似线性块不会显著影响性能，这挑战了关于 Transformer 架构的假设。

**Stable Diffusion 模型与训练**

- **原生 SD 2.1 用于超写实面部**：在 /r/StableDiffusion 中，[**原生 SD 2.1 基础模型正配合扩展程序使用**](https://www.reddit.com/r/StableDiffusion/comments/1dsw5pl/that_new_scramble_prompts_extension_works_really/)，如 Forge UI 中的 Scramble Prompts 和 Mann-E_Dreams-0.0.4，以生成令人印象深刻的超写实面部编辑结果。
- **Halcyon 1.7 位居 SDXL 排行榜首位**：根据 /r/StableDiffusion 的对比，Halcyon 1.7 在 [**提示词遵循度（prompt adherence）和丰富结果方面的 SDXL 模型排名中名列前茅**](https://www.reddit.com/r/StableDiffusion/comments/1dsy6w9/i_compared_top_ai_search_engines_chatgpt/)。
- **使用 IC-light 保持面部一致性**：在 /r/StableDiffusion 中，一位用户正在 [**寻求在项目中使用 IC-light 时保持跨帧和光照条件下面部一致性的技巧**](https://www.reddit.com/r/StableDiffusion/comments/1dsnjcs/how_do_i_maintain_facial_consistency_when_using/)，寻找能够实现稳定性的技术、设置和工具。

**硬件与性能**

- **RTX 3090 水冷方案**：在 /r/LocalLLaMA 中，一位用户正在 [**寻求关于 4x 3090 阵列水冷化的建议**](https://www.reddit.com/r/LocalLLaMA/comments/1dsnud9/watercooling_rtx3090s/) 以提高空间效率，询问单回路方案是否可行以防止降频（throttling）。
- **多 GPU 对推理速度的影响**：另一篇 /r/LocalLLaMA 帖子质疑 [**增加另一块 GPU 是否真的能提高推理速度**](https://www.reddit.com/r/LocalLLaMA/comments/1dt4d9w/do_multiple_gpus_p40_increase_inference_speed/)（通过 Ollama 使用 llama.cpp），还是仅仅提供更大的显存池，在购买前寻求澄清。
- **多 GPU 训练：Deepspeed 对比 Unsloth**：/r/LocalLLaMA 的一个线程对比了 [**不带 Unsloth 的 Deepspeed 与带数据并行（data parallelism）的 Unsloth 的有效性**](https://www.reddit.com/r/LocalLLaMA/comments/1dt3tmb/whats_the_most_effective_training_for_multigpu/)，如果 stage 2 Deepspeed 能带来差异，计划使用它。

**优化与基准测试**

- **超越 NumPy 矩阵乘法**：在 /r/LocalLLaMA 中，一位用户分享了一个 [**高度优化的 C 语言矩阵乘法实现**](https://www.reddit.com/r/LocalLLaMA/comments/1dt3rqc/beating_numpys_matrix_multiplication_in_150_lines/)，遵循 BLIS 设计，仅需 3 行 OpenMP 代码进行并行化，性能便超过了 NumPy/OpenBLAS。
- **在 Hugging Face 上使用 Gemma 2**：/r/LocalLLaMA 链接的一个 Twitter 线程涵盖了 [**在 Hugging Face Transformers 中正确使用 Gemma 2 的方法**](https://www.reddit.com/r/LocalLLaMA/comments/1dsvpp2/thread_on_running_gemma_2_correctly_with_hf/)，包括错误修复、Logits 软截断（soft capping）以及为获得最佳结果的精度设置。
- **Anthropic 资助第三方基准测试**：Anthropic [**宣布了一项资助开发第三方基准测试的计划**](https://www.anthropic.com/news/a-new-initiative-for-developing-third-party-model-evaluations)，用于评估 AI 模型。

---

# AI Discord 回顾

> 摘要的摘要之摘要

1. **LLM 性能与基准测试进展**：

   - 来自 Microsoft 的 [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) 和来自 Google 的 [Gemma 2](https://huggingface.co/google/gemma-1.1-2b-it-GGUF) 等新模型在指令遵循和性能方面表现出显著提升。

   - AI 社区正在积极讨论和比较模型性能，并围绕 [AlignBench 和 MT-Bench](https://x.com/deepseek_ai/status/1787478986731429933) 等基准测试展开辩论。

   - 人们对可复现的基准测试越来越感兴趣，并致力于使用 `lm_eval` 等工具为 [Gemma 2](https://huggingface.co/google/gemma-2b/discussions/18) 等模型复现结果。

2. **优化 LLM 训练与推理**：

   - 各个 Discord 频道的讨论都强调了高效训练技术的重要性，重点关注 [Gemma2 模型的 eager attention](https://openaccess-ai-collective/axolotl) 等方法。

   - 社区正在探索各种量化技术，例如在 [vLLM](https://github.com/vllm-project/vllm) 和 [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/6844) 中实现的技术，以提高推理性能。

   - AI 任务的硬件考量是一个热门话题，围绕 [GPU vs CPU 性能](https://blog.runpod.io/amd-mi300x-vs-nvidia-h100-sxm-performance-comparison-on-mixtral-8x7b-inference/) 以及 [Apple 的 variable bit quantization](https://machinelearning.apple.com/research/talaria) 等专用硬件的潜力展开了辩论。

3. **开源 AI 开发与社区协作**：

   - [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) 和 [LlamaIndex](https://github.com/run-llama/llama_index) 等项目正在促进社区驱动的 AI 工具和框架开发。

   - 对专有模型的开源替代方案的兴趣日益浓厚，例如作为 Sora 替代方案的 [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file)。

   - 围绕 [OpenDevin](https://github.com/OpenDevin/OpenDevin) 和 [Microsoft 的 Graph RAG 架构](https://github.com/microsoft/graphrag) 等项目，协作努力正在不断涌现。

4. **多模态 AI 与生成式建模**：

   - 视觉语言模型的进展正被广泛讨论，包括针对越南语的 [Vistral 7B](https://github.com/hllj/Vistral-V) 以及通过 WebGPU 在本地运行的 [Florence-2](https://x.com/xenovacom/status/1805990110065803492)。

   - 文本转视频生成正受到关注，[Runway 的 Gen-3](https://runwayml.com/) 等工具引发了关于功能和定价的讨论。

   - 社区正在探索模型和技术的组合，以实现 DALLE-3 级别的输出，这表明了向更复杂的多模态系统发展的趋势。

---

# 第一部分：高层级 Discord 摘要

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 的崭新装甲**：**LM Studio 0.2.27** 版本发布，重点改进了对 **Gemma 9B 和 27B** 模型的支持，建议用户下载 [新版本](https://lmstudio.ai) 或使用自动更新功能。发布的 [更新日志](https://github.com/lmstudio-ai/lmstudio.js/issues/45) 显示了多项 Bug 修复以及更新的 `llama.cpp` commit ID。
   - 社区成员还热烈讨论了 Microsoft 将 **Phi-3 Mini** 升级为 **Phi 3.1** 的举动，强调了其在性能和指令遵循方面的飞跃——可以前往 [Hugging Face](https://huggingface.co/lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF) 体验这一新浪潮。
- **硬件的热浪与障碍**：在 LLM 任务中，**VRAM** 优于 RAM 的地位是热门话题，共识倾向于将 **8GB VRAM** 作为避免性能陷阱的最低推荐配置。一位用户展示了拥有 **120GB VRAM**（配备 5 块**水冷 4090s**）的装备，其处理 LLM 推理的强力方案引发了广泛关注。
   - 然而，并非一切都顺风顺水，有成员报告在 **P40 GPU** 等硬件上使用 **LM Studio** 时出现 GPU 待机温度问题，以及更新后特别是 **AMD GPUs** 的兼容性担忧，详见 [Extension-Pack-Instructions.md](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md)。
- **开发者聊天中的协作调试**：**LM Studio** 中的 SDK 版本冲突导致用户在升级到 0.2.26 后陷入了 `await client.llm.load(modelPath);` 错误的困境。**lmstudio.js** 的 [GitHub issue](https://github.com/lmstudio-ai/lmstudio.js/issues) 记录了这一过程，为社区排查故障提供了平台。
   - Discord 机器人也未能幸免；一个由于缺少 **MessageContent** intents 导致的 **TokenInvalid** 错误案例，通过社区协作得到了解决，彰显了集体解决问题的精神。
- **量化精要与 Gemma 2**：对 **Gemma 2** 模型更新的热情与 **Phi 3.1** 的增强不相上下，两者都承诺在最新的 **LM Studio** 迭代中运行更加顺畅。社区向量化版本的转向（如 **Gemma 2 9b 和 27b**）表明了对性能优化的敏锐关注。
   - 尽管 **Gemma 2** 在 **AMD 6900 XT** GPU 上遇到了意外挫折（如“模型加载失败”问题所示），目前的趋势似乎倾向于“重新下载并重试”的方法，不过成员们仍在等待更稳妥的修复方案。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Transformers 的技术凯旋**：随着 **Transformers 4.42** 的发布，用户现在可以访问各种新鲜模型和功能，包括 **Gemma 2**、**RT-DETR** 等，详见 [更新日志](https://github.com/huggingface/transformers/releases/tag/v4.42.0)。
   - 社区反应显示出对此次更新的高度兴奋，期待其带来的**增强功能**和高效的 RAG 支持。
- **Chronos 数据纪元**：**AWS** 在 Hugging Face 上发布了其 **Chronos 数据集**及评估脚本，提供了用于预训练和评估的数据集，详情参阅 [此处](https://s3.amazonaws.com/openresearch.com/chronos-datasets.html)。
   - 这些数据集的发布获得了社区的赞誉，被认为是对比特时间数据处理和分析感兴趣者的**重大贡献**。
- **大规模指标：Hub 的重要里程碑**：已有 **10 万个公开模型**利用 Hub 存储 `tensorboard` 日志，功能汇总见 [此处](https://huggingface.co/models?filter=tensorboard)，展示了该平台不断扩展的实用性。
   - 这一成就被视为**模型监控的重要枢纽**，简化了在模型 Checkpoints 旁跟踪训练日志的过程。
- **越南语视觉语言探索**：越南 AI 的新浪潮由 **Vistral 7B** 引领，这是一款基于 **LLaVA** 和 Gemini API 的**视觉语言模型 (Vision-Language model)**，旨在增强图像描述能力，如其 [GitHub](https://github.com/hllj/Vistral-V) 所示。
   - 团队开启了社区参与渠道，通过交互式 [Demo](https://964c3125aaea36d527.gradio.live/) 寻求关于模型性能的见解，以进一步挖掘视觉能力。
- **时尚界的 AI 艺术：电子商务的未来**：**Tony Assi** 展示了多个为电子商务量身定制的 AI 驱动项目，特别关注利用计算机视觉和机器学习在时尚领域进行创新。
   - 这些应用的多样性（可在 [此处](https://huggingface.co/spaces/tonyassi/AI-Ecommerce-Fashion) 查看）凸显了 **AI 在转型电子商务行业方面的巨大潜力**。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Conclave Crunch**：一场**[仅限 CUDA 的黑客松 (CUDA-only hackathon)](https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf)** 将于 **7 月 13 日**在旧金山举行。该活动由 Chris Lattner 主办，并提供 **H100 访问权限**，这体现了 Nebius AI 的支持。
   - 活动旨在培养硬核黑客精神，从**中午 12:00 持续到晚上 10:00**，严格遵守 **CUDA** 标准，并在 [Twitter](https://twitter.com) 上引发了讨论热潮。
- **矩阵乘法大师 (Matrix Multiplication Mastery)**：**[Mobicham 发布了矩阵乘法指南](https://salykova.github.io/matmul-cpu)**，演示了在 CPU 上的计算，对于专注于基础操作的 AI 工程师来说是一个出色的资源。
   - 这一经常被忽视的 AI 模型效率核心成为了关注焦点，为围绕 AI 工作流中计算优化的讨论铺平了道路。
- **INTx 基准测试闪击战 (INTx's Benchmarking Blitz)**：INTx 性能**[基准测试](https://github.com/pytorch/executorch/tree/main/examples/models/llama2#quantization)** 超过了 fp16，其中 **int2** 和 **int4** 在每秒 token 数 (tokens per second) 方面打破了记录。
   - **torchao** 量化实验表明，**int8-weight** 和 **intx-4** 展示了惊人的速度，而在 batch size 为 1 时优化的评估则强调了未来的性能探索方向。
- **基准测试脚本集市 (Benchmark Script Bazaar)**：MobiusML 分享了对 AI 社区至关重要的 [基准测试脚本](https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py)，以及每秒 token 数的测量方法。
   - 这些基准测试对于性能指标至关重要，特别是在解决了最近关于 Transformers 及其缓存逻辑的问题之后。
- **从 CUDA 琐事到核心重构 (CUDA Chores to Core Overhaul)**：工程师们在 **CUDA MODE** 中重构了辅助函数和 kernel 函数，提升了效率和内存优化，同时倡导 **[GenericVector](https://github.com/karpathy/llm.c/pull/641/)** 的应用。
   - AI 从业者仔细研究了**训练稳定性 (training stability)**，剖析了数据集的影响，简化了设置，并讨论了推理优化，这在对 **[llm.c](https://github.com/karpathy/llm.c/)** 等仓库的贡献中可见一斑。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **与 Perplexity 的 Android 应用畅谈**：Perplexity AI 为 Android 发布了**语音对语音功能 (voice-to-voice feature)**，实现了无缝的免提交互，并通过[指定频道](https://discord.com/channels/1112135181567000697)征集用户反馈。该应用提供 *Hands-free*（免提）和 *Push-to-talk*（一键通）两种模式。
   - **Pro Search 更新**增强了 Perplexity 处理**复杂查询**的能力，坚实支持多步推理、**Wolfram|Alpha** 和代码执行。点击[此处](https://pplx.ai/FVyJaIH)探索增强功能。
- **使用 Perplexity AI 整理知识**：用户讨论中反映了 Perplexity AI 搜索引擎的局限性，例如对印度新闻的偏见和不稳定的来源选择，并将其与 [Morphic.sh](https://www.morphic.sh/share/W1Kd9iO) 等同类工具进行了评估。显然需要增强来源设置并平衡全球内容。
   - 社区对话揭示了 Claude 在 Perplexity AI 中存在 **32k tokens** 的瓶颈，这降低了用户对宣传中 **200k tokens** 的预期。这标志着它正从理想的 AI Assistant 转向更以搜索为中心的模型。
- **探讨 Perplexity 的绘图潜力**：关于在 Perplexity AI 中进行数据可视化的咨询得出结论，虽然它本身不生成图表，但可以辅助生成用于 Google Colab 等外部平台的代码。成员们建议利用 [AIlin](https://gitlab.com/monnef/ailin) 等扩展来实现集成的图形输出。
   - 针对**推荐链接 (referral links)** 的泛滥，Perplexity AI 用户群期待潜在的试用版本，尽管由于过去的滥用行为目前似乎并未提供。社区在订阅前获得亲身体验的渴望仍未得到满足。
- **从魔方周年庆到 Meta 的危机**：为纪念半个世纪的认知挑战，**魔方 (Rubik's Cube)** 迎来了 50 周年纪念并发布了特别致敬视频，可在[此处](https://www.youtube.com/embed/UrJp4OuxFGM)观看。在魔方的光辉之下，还有一系列杂乱的新闻，从 Meta 的欧盟指控到电动飞行的创新。
   - 出现了关于通过 Perplexity AI 创建业务计划的讨论，展示了一个用于构建创业战略的**精益画布 (lean canvas)** 教程。热衷于商业的头脑被该指南所吸引，详见[此处](https://www.perplexity.ai/search/when-should-you-do-a-lean-canv-z_lDH7CJStuuX.MpyRGNMA)。
- **Perplexity 管线中的 API 焦虑与期待**：Perplexity AI 的 API 生态系统遇到了 Chrome 设置无法加载的小故障，促使用户探索 Safari 作为替代方案，社区建议通过清除缓存进行修复。
   - 虽然 Sonnet 3.5 仍未进入 Perplexity API 的范畴，但通过 API 使用搜索引擎的兴趣引发了关于可用模型及其与 Hugging Face 实现的功能对等性的讨论。详细的模型文档请参考[此处](https://docs.perplexity.ai/docs/model-cards)。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **十亿角色突破攻克 MATH 难题**：[Aran Komatsuzaki](https://x.com/arankomatsuzaki/status/1807593343007818065) 详细介绍了 **Persona Hub** 项目，该项目通过 **10 亿个 personas** 生成数据，将 MATH 分数从 **49.6 提升至 64.9**。他们在 [GitHub repo](https://github.com/tencent-ailab/persona-hub) 和 [arXiv 论文](https://arxiv.org/abs/2406.20094) 中分享了这一概念，引发了关于数据价值高于代码的讨论。
   - 讨论围绕 **Persona Hub** 的复制难度以及扩展合成数据生成的潜力展开，一位成员强调：“数据远比代码重要”。
- **Phi-3 Mini 获得性能增强更新**：Microsoft 宣布升级 **Phi-3 Mini**，增强了 [Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) 上的 4K 和 128K 上下文模型权重，并引发了对其未公开的高级训练方法的猜测。
   - LocalLLaMA 社区反应热烈，有人幽默地将其与 OpenAI 的保密做法进行对比，调侃道 *“ClosedAI 的 CriticGPT”*。
- **Unsloth 采用 SPPO 以实现流线化集成**：*theyruinedelise* 确认了 SPPO 与 Unsloth 的兼容性，讨论了它如何与 TRL 协同运行，为 Unsloth 用户提供直接的集成方案。
   - 集成方面的进展仍在继续，Unsloth 通过 TRL 可靠地扩展了对 SPPO 的支持，简化了开发者的工作流。
- **在 Discord 中与聊天机器人闲聊**：由 **jakobdylanc** 开发的 **llmcord.py** 脚本已在 [GitHub](https://github.com/jakobdylanc/discord-llm-chatbot) 上线，它将 Discord 重新利用为 LLM 界面，因其实用性和易用性赢得了社区的赞誉。
   - 社区成员受到该发布的启发，纷纷表达赞赏，如“做得好！”等评价，强调了对这一创新的集体支持。
- **征集 Notebook 反馈**：一个新的支持多个数据集的 [Colab notebook](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing) 向社区征求反馈，旨在优化用户体验和功能。
   - 该 Notebook 的显著特性，如 **Exllama2 quant 支持** 和 **LoRA rank scaling**，反映了社区的协作精神，期待获得建设性的反馈。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **解决显存 (VRAM) 困扰的 Diffusion 难题**：一位成员询问如何在仅有 12GB VRAM 的情况下使用 **Stable Diffusion**，因为 **everydream2 trainer** 需要 16GB。另一位成员分享了在 8GB VRAM 系统上经过 4 小时高强度运行后成功生成微型 checkpoint 的经验。
   - 对话转向了在显存受限的系统上运行 **Stable Diffusion** 的策略，成员们交流了关于不同模型和设置的技巧，以更好地适应硬件限制。
- **Stable Diffusion 3 面临挑战**：社区剖析了 **Stable Diffusion 3 (SD3)** 的弱点，指出其在精细可调性方面未达预期，且功能实现不完整，例如 **Low-Rank Adaptation (LoRA)** 的训练效率低下。
   - 一位参与者对 SD3 在复杂姿势下的图像质量缺陷表示失望，主张进一步改进功能以克服模型性能不稳定的领域。
- **LoRA 的学习循环**：关于为特定风格（从 3D 分形到游戏截图）训练 **LoRA (Low-Rank Adaptation)** 的讨论激增，尽管 **Stable Diffusion 3** 对 LoRA 训练的限制是一个制约因素。
   - 社区成员热烈地交流了辅助专业训练的变通方法和工具，反复强调在追求自定义模型精通的过程中，尝试与错误（trial and error）的价值。
- **Stable Diffusion 的风格光谱**：参与者分享了他们在 **Stable Diffusion** 广泛风格光谱上的成功与尝试，从线条艺术的锐利到恐怖风格的诡异，每一个案例都包含了使用成功的经验和提示词（prompt）失控的不可预测性。
   - 尽管提示词有时会带来莫名其妙的混淆，成员们仍对模型交织不同视觉叙事的能力感到欣喜。
- **艺术 vs AI：反 AI 算法的困境**：公会辩论了开发软件以保护艺术家作品不被 AI 吸收的问题，参考了 Glaze 和 Nightshade 等工具，并承认这些方法在面对持续存在的漏洞时存在不足。
   - 辩论揭示了在艺术作品中嵌入反 AI 功能的实用性，并讨论了与数字艺术复制相关的道德问题，强调了有效防止 AI 同化的挑战。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **价格冲击与硅片阻碍**：工程界对 **8 H100 GPUs** 昂贵的价格议论纷纷，据称价格超过 **50 万美元**，且采购存在瓶颈，需要 NVIDIA 企业账户。
   - 侧边栏讨论揭示了对 **Google TPUs** 和 **Paperspace** 等替代方案的兴趣，后者提供 **每小时 2.24 美元的 H100**，被视为具有成本效益的训练解决方案。
- **AI 创作者评估工具**：人工智能图像爱好者对 **Luma Dream Machine** 和 **Runway Gen-3** 等工具表示不满，认为其定价过高，**15 美元** 仅能生成寥寥 **6-7 个输出**。
   - 随着人们认为这些工具相较于前代产品缺乏实质性进步，热情有所减退，这促使生成式工具市场需要更高的效率和创造力。
- **模型 Prompt 中的多步失误**：社区讨论了 **GPT** 在执行多步任务时尽管有明确指令却仍倾向于跳过步骤的问题，这损害了模型在任务执行中的彻底性。
   - 有人分享了一种按顺序构建指令的技巧，即使用“这个在那个之前”的逻辑，引导 **GPT** 完整执行预定流程而不遗漏步骤。
- **意图查询的探究**：讨论深入探讨了如何为意图检查设计精巧的 **RAG-on-GPT-4o** Prompt，将响应细分为模糊、历史记录或需要新搜索查询。
   - 开发者担心 Bot 会混淆历史记录与新查询，因此呼吁在上下文和意图解释方面保持一致性。
- **局限性与编排层**：对话深入探讨了 **multimodal models** 的局限性及其在特定领域的挣扎，并以 **Python** 编程等专业任务作为计算对比。
   - 像 **LangChain** 这样的编排器因其在扩展 AI 上下文限制（突破 **4k token** 阈值）以及构建 **GPT-4** 和 **LLama** 等高级模型架构中的作用而备受关注。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **vLLM 驰援 HF 救援**：一位用户建议使用 **vLLM** 作为 HF 已知问题的解决方案，参考 [wiki 指南](https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm) 将模型转换为 **16 Bit**，从而提高效率。
   - 社区对这种模型部署的替代方案表示赞赏，因为 vLLM 被证明是有效的，用户对该建议表示感谢。
- **Gemma 2 复现问题受到关注**：使用 `lm_eval` 对 **Gemma 2** 进行的准确率基准测试与官方指标不符，促使用户尝试使用 `bfloat16` 和特定的 Transformer 版本，详见[此处](https://x.com/LysandreJik/status/1807779464849273343)。
   - 在 `model_args` 中添加 `add_bos_token=true` 后，得分更接近模型的论文基准，特别是 **lambada_openai** 的准确率从 **0.2663 跃升至 0.7518**。
- **LLM Tokenization 中的语义混淆**：一篇新[论文](https://arxiv.org/abs/2406.20086)审视了 LLM Tokenization 中的“擦除”效应，指出 **Llama-2-7b** 有时会将“northeastern”等单词拆分为不相关的 Token。
   - 该研究因深入探讨从随机 Token 组到高层表示的转换而引起关注，有助于理解 Tokenization 过程。
- **“分解诅咒”成为头条**：一篇[论文](https://arxiv.org/abs/2406.05183)将“反转诅咒”重新定义为“分解诅咒（factorization curse）”，揭示了 LLM 在信息检索方面的问题，并引入了 WikiReversal 来模拟复杂任务。
   - 报告暗示在损坏或改写的数据上进行训练后，模型泛化能力更好，受此启发，有人建议采用 UL2 风格的目标。
- **Fewshot Prompts 的图形可视化**：正在研究 Fewshot Prompting 对准确率影响的用户正在寻求可视化 Prompt 效果的方法，并尝试使用 `--log_samples` 进行更深入的分析。
   - 展示的一种方法是保存输出以供检查，旨在发现 Fewshot Prompting 的负面影响，并指导评估准确率的提升。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Apple 在端侧表现惊艳**：**Apple 的可变比特量化 (variable bit quantization)** 优化了端侧 LLM，正如其 [Talaria 工具](https://machinelearning.apple.com/research/talaria) 所展示的那样，该工具自发布以来已增强了 3,600 多个模型。该技术获得了**最佳论文荣誉提名 (Best Paper Honorable Mention)**，功劳归于 Fred Hohman 和 Chaoqun Wang 等知名研究人员。
   - 在 **WWDC 2024** 上，随着 **Apple Foundation Models** 的推出，端侧智能得到了进一步推动。这些模型整合了 iOS 18、iPadOS 18 和 macOS Sequoia。根据 [Apple 的报告](https://machinelearning.apple.com/research/introducing-apple-foundation-models)，展示了一个用于日常任务的 30 亿参数端侧 LLM。
- **Runway 变革：视频生成成本引发热议**：[Runway 最先进的视频生成工具](https://runwayml.com/)尽管技术领先，但 60 秒视频 12 美元的价格标签引发了争论。
   - 社区成员 *Mautonomy* 领头呼吁更合理的 **每生成一次 0.5 美元** 的价格，并将其与其他溢价服务进行了比较。
- **Genstruct 7B 催化指令创作**：NousResearch 的 **Genstruct 7B** 模型脱颖而出，成为从原始文本构建指令数据集的工具包，其灵感源自 [Ada-Instruct](https://arxiv.org/abs/2310.04484)。
   - *Kainan_e* 强调，Genstruct 旨在简化 **LLM** 的训练，使其成为寻求增强指令生成能力的开发人员关注的焦点。
- **VLLMs：动画中的视觉与语言融合**：讨论涉及训练 **VLLMs** 以实现动画自动化，其中推介了一种[基于扩散的关键帧补间方法](https://setarehc.github.io/CondMDI/)。
   - 社区一致认为 *Hermes Pro* 在该领域可能具有潜力，而 *Verafice* 则带头批评并寻求更有效的解决方案。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的树莓派难题**：在 **Raspberry Pi 5** 的 **Ubuntu 24.04** 上运行 Mojo 时出现了一个问题，引发了对故障排除支持的寻求。
   - 该情况在对话中仍未解决，突显了对社区驱动解决方案或进一步对话的需求。
- **Mojo 拓展视野**：讨论揭示了 Mojo 在推进 **基于模型和模拟器的 RL**（用于 LLM Agent）、**符号推理 (symbolic reasoning)** 以及 **亚符号模型引导 (sub-symbolic model steering)** 等领域的潜力。
   - 参与者集中讨论了这些新兴应用，社区成员表达了合作和交流见解的渴望。
- **基准测试的利与弊**：社区努力为 Mojo 的测试和基准测试框架带来了显著增强，特别是提高了性能指标。
   - 尽管取得了进展，但由于舍入误差导致的基准测试失败以及 GFlops/s 测试期间的死循环等挑战，凸显了优化工作的迭代本质。
- **Nightly 编译器复杂性**：Mojo 编译器 Nightly 版本 **2024.7.205** 的发布促使了关于使用 `modular update nightly/mojo` 进行版本控制的讨论，并解决了 CI 转换问题。
   - 周边商品查询得到了社区的直接支持，而一些成员面临的 Nightly/Max 软件包更新挑战最终也得到了解决。
- **矩阵乘法思索**：大量活动集中在 `src/main` 编译错误上，特别是 `matrix.mojo` 中的 `DTypePointer`，导致了使用稳定构建版本的建议。
   - 参与者集思广益改进矩阵乘法，提出了向量化 (vectorization)、分块 (tiling) 以及集成 **Strassen** 和 **Winograd-Copper** 等算法的方案。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **模型页面改版**：宣布了 **/models 页面**的**更新**，承诺带来新的改进，并在此征求社区**反馈** [此处](https://openrouter.ai/models_feedback)。
   - Gemini 和 PaLM 的 **token 尺寸**变更将使统计数据标准化，但也会改变**定价和上下文限制 (context limits)**；社区将面临调整。
- **弃用默认模型**：OpenRouter 设置中的**默认模型 (Default Model)** 正面临**弃用 (deprecation)**，取而代之的是特定模型设置或自动路由 (auto router) 等替代方案。
   - 针对 OpenAI API 密钥的**自定义认证头 (Custom auth headers)** 也正在逐步淘汰，这表明正转向更新、更可靠的身份验证方法。
- **Discord 机器人对话精简化**：社区成员分享了优化对话机器人的**技巧**，强调了**高效利用 token (token-efficient)** 的策略，例如在 prompt 中仅包含必要的消息部分。
   - 这引发了关于平衡模型**上下文限制 (context limits)** 与保持对话吸引力的讨论，并参考了 [SillyTavern Discord](https://sillytavern.app) 的方法。
- **Claude 3.5 的代码纠纷**：**Claude 3.5** 间歇性出现的**错误**引起了轰动，而 Claude 3.0 则未受影响，这表明可能存在**特定于模型的问题**。
   - 社区正在积极分享变通方法并等待**修复**，突显了工程领域协作调试的本质。
- **iOS 前端发现 OpenRouter 的乐趣**：关于支持 OpenRouter 的 **iOS 应用**查询得到了推荐，Pal Chat 和 Typingmind 在修复 bug 后处于领先地位。
   - 寻找适合 OpenRouter 集成的**多样化前端平台**的参与度表明，移动 AI 应用生态系统正在不断壮大。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Eager vs Flash：Gemma2 的注意力机制升温**：训练 Gemma2 模型时[推荐使用 Eager attention](https://openaccess-ai-collective/axolotl)，在 **AutoModelForCausalLM.from_pretrained()** 中将设置修改为 'eager' 以增强性能。
   - 为 eager attention 配置 **YAML 文件**得到了全面覆盖，为训练 **Gemma2** 以达到性能基准提供了细粒度控制。
- **优化讨论：评估 Adam-mini 和 CAME**：关于在 *axolotl* 中集成 [CAME](https://arxiv.org/abs/2307.02047) 和 [Adam-mini](https://arxiv.org/abs/2406.16793) 优化器的讨论非常热烈，主要围绕它们较低的内存占用和潜在的训练稳定性。
   - [Adam-mini](https://arxiv.org/abs/2406.16793) 作为 AdamW 在内存效率方面的竞争对手出现，引发了关于其在大模型优化中务实使用的讨论。
- **衡量标准：数值精度优于文本描述**：一位用户寻求在模型输出中优先考虑精确的数值响应而非解释性文本，思考使用加权交叉熵 (weighted cross-entropy) 来引导模型行为。
   - 虽然对这种微调 (fine-tuning) 方法的研究仍在进行中，但社区将加权交叉熵视为提高模型准确性的一个有前景的途径。
- **Gemma2 的微调挫折：应对巨大的梯度范数 (Gradient Norm)**：分享了关于 **Gemma2 27b** 的微调 (finetuning) 困扰，报告显示其 grad_norm 值很高，与 9b 等较小模型更平滑的训练体验形成对比。
   - 降低学习率 (learning rates) 和利用 'flash attention' 是提出的解决 grad_norm 巨兽并缓解 **Gemma2** 训练情绪的方案之一。
- **ORPO 的硬性规则：训练必须成对**：一位成员在为 **ORPO** 生成接受/拒绝 (accepted/rejected) 样本对时感到困扰，寻求确认训练数据中的每一行是否都需要这两者。
   - 社区共识强调了样本对生成在 **ORPO** 对齐 (alignment) 过程中的关键作用，并强调了其在实践中的复杂性。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Llama-Leap 迈向微服务**：[Mervin Praison 制作了一个深入的视频教程](https://t.co/eW9mT6IHlk)，介绍了全新的 **llama-agents** 框架，概述了高层概念和实际实现。
   - 该教程被广泛认为非常详尽，深入探讨了将 Python 系统转换为微服务的复杂性，社区对其涵盖的**高级特性**给予了高度评价。
- **知识助手变得更聪明**：**AI Engineer World Fair** 展示了关于增强知识助手的突破性讨论，强调了需要**创新数据模块**来提升其效率。
   - 专家们主张转向更复杂的数据处理方式，这是从朴素 RAG 结构向下一代**知识深化（knowledge deepening）**迈出的关键一步。
- **微软 Graph RAG 详解**：微软创新的 [Graph RAG Architecture](https://github.com/microsoft/graphrag) 在社区引起了轰动，它被构想为一个灵活的、基于图的 RAG 系统。
   - 这一发布引发了专业人士的好奇和热情，许多人渴望剖析其在**建模架构**方面的潜力。
- **Pinecone 的元数据困境**：在处理 **DocumentSummaryIndex** 信息时，**Pinecone** 的元数据限制引发了技术挑战，迫使一些人编写变通方案。
   - 这一排错过程引发了关于替代框架的广泛讨论，强调了 Pinecone 在元数据处理上的僵化，并激发了对**动态元数据模式（dynamic metadata schema）**的呼吁。
- **聊天机器人：RAG 革命**：AI 工程师探讨了开发**基于 RAG 的聊天机器人**，旨在利用 LlamaHub 的数据库读取器接入跨不同 SQL 和 NoSQL 平台的公司数据。
   - 对话开启了制定用户文本查询的策略，关于数据库路由的知识共享展示了社区**协作排错**的精神。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **图嫁接（Graph Grafting）取得进展**：Tinygrad 的讨论集中在增强现有的 `graph rewrite followup, speedup / different algorithm`，目前尚未就首选算法达成共识。Egraphs/muGraphs 已被列入后续关注名单。
   - 尽管不是图灵完备的，成员们仍**大胆支持**基于规则的方法，因为它易于推理。同时，也有人呼吁在 graph rewrite 中嵌入更多算法（如 scheduler）。
- **揭开 'image dtype' 的神秘面纱**：**Tinygrad 的 'image dtype'** 在代码库中无处不在却又显得神秘，引发了辩论；目前尚未记录消除它的具体行动。
   - 诸如“你尝试过移除它吗？”之类的询问在盘旋，讨论悬而未决，缺乏详细的探索或结果。
- **错误信息的低语**：Tinygrad 反复出现的错误 **RuntimeError**: *failed to render UOps.UNMUL* 强调了在 Tinygrad 迈向 v1.0 里程碑之际，迫切需要彻底改革错误消息机制。
   - George Hotz 认为这更多是一个 assert 问题，表示它“永远不应该发生”，而一位社区成员正准备通过一个**带有失败测试用例的 PR** 来解决此问题。
- **内存混乱与梯度处理**：热烈的讨论指出了 Tinygrad 在梯度累积期间的 **CUDA memory overflow** 问题，用户交流了诸如每步减少 loss 和管理梯度内存等策略。
   - `Tensor.no_grad = True` 成为 Tinygrad 在推理期间对抗梯度计算的利器，类似于 `torch.no_grad()`；而 `a = a - lr * a.grad` 是目前的操作方式，因为 `a -= lr * a.grad` 会触发断言。
- **文档困境需要深思熟虑**：为了追求清晰度，Tinygrad 频道的参与者呼吁提供更丰富的文档，特别是涵盖 TinyJit 和细致的梯度累积等高级主题。
   - 随着开发者在创新与用户支持之间寻求平衡，制作全面指南和生动示例以照亮 Tinygrad 认知盲区的倡议得到了支持。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **RAG 策略大比拼：HydeRetrieval vs. MultiQueryRetrieval**：围绕检索策略的优劣展开了激烈讨论，**HydeRetrieval** 与 **MultiQueryRetrieval** 互不相让。几位用户参与了对话，其中一位在使用 **MultiQueryRetrieval** 时遇到了结果为空的情况，引发了关于潜在回退方案（fallbacks）和修复方法的讨论。
   - 讨论延伸到了分片数据库（sharded databases）领域，一位求知者寻求实现此类系统的建议。大家分享了见解，并提到了 serverless MongoDB Atlas，但社区对于分片查询映射（shard-query mapping）的具体细节仍渴望更多信息。
- **API 愿景与文件上传查询**：在 LangServe 领域，出现了一个关于将 **fastapi-users** 与 **langserve** 结合的难题，旨在通过用户特定逻辑保护端点。目前讨论区较为安静，尚无明确的指引路径。
   - 另一位 LangChain 爱好者寻求关于启用文件上传的指导，希望摆脱静态文件路径的限制。技术群体分享了代码片段和见解，但分步解决方案仍未完全明朗，依然笼罩在技术迷雾中。
- **LangChain 聊天机器人：行动 Agent**：在 LangChain 开发者中，一项新任务是赋予聊天机器人预约演示和建立人工连接的能力。有人给出了回应，概述了使用 **Agents** 和 **AgentDirector** 的方法，并展示了通过 **LangSmith** 进行调试的潜力。
   - 随着对 Python 代码实现聊天机器人执行动作技能的需求增加，讨论进一步扩展。回应中包含了丰富的方法步骤和教程，为那些敢于为其数字作品启用动作功能的人提供了社区知识。
- **CriticGPT 的征程与 RAFT 的启示**：OpenAI 的 **CriticGPT** 凭借一段视频揭秘走进聚光灯下，剖析了其优化 GPT-4 输出的方法，并提高了代码生成精准度的标准。敏锐的思想者吸收了这些智慧，思考着论文相关视频评测所标志的进步。
   - 前瞻者们并未停歇，深入探讨了 **RAFT** 方法论的深度，分享了学术文章，并将其与传统的 **RAG** 机制进行对比。一个关于使用 **LLM** 构建聊天机器人的协作号召广泛传播，公开邀请各方加入创新。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Runway Revolution with Gen 3 Alpha**：Runway 推出了 **Gen-3 Alpha Text to Video**，这是一个面向所有用户的**高保真、快速且可控的视频生成**工具。可以在[这里](https://runwayml.com)体验，或在[这里](https://x.com/runwayml/status/1807822396415467686)查看公告。
   - 社区进行的与 **SORA** 的正面**对比**突显了 Gen-3 Alpha 在即时可用性方面的独特优势，让人们得以窥见视频生成的未来。你可以在[这里](https://x.com/altryne/status/1807868306361094153)查看对比评测。
- **Sonnet Syncs with Artifacts**：**Sonnet 与 Artifacts** 的融合因提高了可视化和操作流程图的效率而受到赞誉，促进了更直观、更快速的设计工作流。
   - 爱好者们对这种能以思维速度合成视觉概念的能力表示赞赏，它消除了手动调整的繁琐，并简化了创作过程。
- **Figma Debunks Design Data Doubts**：Figma 针对用户的担忧做出了回应，澄清其 **'Make Design' 功能**并未在 Figma 的专有内容上进行训练，官方声明可见[这里](https://x.com/zoink/status/1808045655082033483)。
   - 尽管 Figma 进行了说明，但社区讨论仍对输出中出现的识别度较高的元素（如 **Apple 的天气应用**）表示怀疑，引发了关于 AI 生成设计伦理的持续辩论。
- **Microsoft's Magnified Phi-3 Mini**：Microsoft 增强了 **Phi-3 Mini**，提升了其在代码理解和多语言支持方面的能力极限，以实现更高效的 AI 开发。在[这里](https://x.com/reach_vb/status/1808056108319179012)查看更新。
   - 改进涵盖了 4K 和 128K 模型，重点在于丰富上下文和结构化响应能力，预示着在代码细微理解方面的进步。
- **Magic Dev's Market Magic**：初创公司 Magic Dev 尽管缺乏具体的产品或收入，但其目标是实现雄心勃勃的 **15 亿美元估值**，这凸显了 AI 领域狂热的市场投机。详情见 [Reuters](https://www.reuters.com/technology/artificial-intelligence/ai-coding-startup-magic-seeks-15-billion-valuation-new-funding-round-sources-say-2024-07-02/) 的讨论。
   - 这个由 20 人精干团队设定的估值目标，指向了 AI 投资领域的看涨趋势，并在缺乏坚实基本面的情况下，再次引发了对潜在泡沫的担忧。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llama.cpp 在轻量级硬件上运行？再想想吧**：关于运行 **llama.cpp** 的讨论揭示，**iPhone 13** 和 **Raspberry Pi Zero W** 都不符合成功运行所需的 64-bit 系统前提条件，特定的型号和内存规格至关重要。
   - 社区成员指出，尽管便携式设备很有吸引力，但像 **Raspberry Pi Zero** 这样的型号由于系统和内存限制而力不从心，这促使人们重新评估合适的硬件。
- **Llamafile v0.8.9 随着 Android 支持的加入实现飞跃**：Mozilla 宣布 [发布 llamafile v0.8.9](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.9)，增强了 Android 兼容性，且 **Gemma2** 模型更贴近 Google 的框架。
   - 反馈称赞了该版本对 **Gemma2** 的对齐改进，根据公开评估，这表明它现在可以与更大的模型竞争。
- **通过简单的开关修复 Mxbai 模型怪癖**：**mxbai-embed-large-v1** 模型的一个奇怪行为（对不同的文本输入返回相同的向量）通过将输入键从 'text' 更改为 'content' 得到了解决。
   - 社区建议在 [Hugging Face 上更新模型](https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile) 以提高清晰度并简化未来的部署，这标志着模型的持续优化。
- **探索大语言模型的硬件迷宫**：AI 爱好者聚集在一起，为运行重量级语言模型确定最佳硬件配置，对话中 **VRAM** 容量和 **CPU 内存** 性能占据了主导地位。
   - 用户的实际见解倾向于将 **3090/4090 GPU** 用于主流用途，同时为那些追求极限的用户推荐 **A6000/RTX 6000** 工作站，强调了对有利硬件配置的尝试与平衡。
- **模型训练中 CPU 与 GPU 的大辩论**：对于模型训练，**GPU** 仍然是优于 CPU 的首选平台，这归功于它们的计算实力，正如多核 CPU 在处理大型模型时表现出的缓慢且不足的处理能力所证明的那样。
   - 关于 **基于 CPU 训练** 可行性的推测正在酝酿，社区使用 **llm.c** 训练 GPT-2 等模型的测试凸显了 CPU 在大规模学习应用中的明显局限性。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Windows 上的苦与乐**：用户在 **Windows 上构建 01** 时遇到了挑战，**piranha__** 寻找指导未果，但偶然发现了一个可能具有挽救意义的 [Pull Request](https://github.com/OpenInterpreter/01/pull/203)，该请求承诺更新安装指南。
   - 讨论中的 **Pull Request** 汇集了过去用户克服 Windows 安装难题的尝试，有望为未来的尝试扫清障碍。
- **排除 OI 中的并发故障**：**chaichaikuaile_05801** 讨论了 **OI 部署** 中 **并发** 和资源隔离的困惑，辩论了 **OI 多实例 (Multiple Instances)** 与其他上下文解决方案的优劣。
   - 交流中考虑了使用 `.reset()` 来规避代码共享障碍，结论是不同的实例可以避免共享 Python 执行环境带来的混乱。
- **OI 中的图像难题**：**chaichaikuaile_05801** 的一份恳切请求强调了通过 OI 的 `MatPlotLib.show()` 显示图像时面临的障碍，并引用了一个开放的 [GitHub Issue](https://github.com/OpenInterpreter/open-interpreter/issues/1301)，展示了版本之间的差异。
   - 随着用户在 0.1.18 和 0.2.5 版本之间切换，他们呼吁未来的版本加强图像返回功能，表明了对可视化改进的渴望。
- **寻找高性能本地 AI Agent**：**blurrybboi** 正在寻找具有超越基础查询的网页搜索能力的本地 AI Agent，目标是那些能够从冗余信息中筛选出优质输出的 Agent。
   - 尽管发出了请求，但目前还没有收到回复，关于 AI Agent 在高级在线过滤方面的能力问题仍未得到解答。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **WandB 助力工作流**：一位用户庆祝了他们成功**微调模型 (finetuning a model)**，通过利用在线资源克服了对个人 GPU 的需求，并采用了 **WandB** 以获得更好的训练洞察。
   - 社区讨论得出结论，根据分享的 [WandB logger 文档](https://pytorch.org/torchtune/main/deep_dives/wandb_logging.html)，精简 YAML 配置并采用 **WandB's logger** 可以简化训练过程。
- **评估 AMD 的 AI 表现**：成员们交流了使用 **AMD GPUs** 进行 AI 开发的经验，尽管有些人在 **ROCm** 和 **torchtune** 上取得了成功，但仍推荐 NVIDIA 替代方案，详情见 [Reddit 指南](https://www.reddit.com/r/LocalLLaMA/s/VRNQWhh2fh)。
   - 一位用户分享了他们在 **6900 XT** 上艰难但最终成功的历程，强调了社区在 AMD 硬件上对 **torchtune** 进行故障排除的支持。
- **为 Torchtune 模型提供 HuggingFace 支持**：有人询问如何将 **torchtune 模型转换**为 **HuggingFace** 格式，表明用户有意合并工具集。
   - 虽然没有讨论转换过程的具体细节，但参与者分享了命名规范和集成策略，展示了在模型兼容性方面的工程努力。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Toolkit 的多步处理奇迹**：用户确认了 [Toolkit 的多步处理能力 (multi-step capabilities)](https://discord.com/channels/954421988141711382/954421988783444043/1257517558375120947)，具有已启用的前端和后端支持，并分享了实际运行的示例。
   - 讨论还赞扬了 **Sandra Kublik** 在 **SF AI Engineer** 会议上的分享，并强调了 **LLM-UNIVERSITY 频道** 的关闭，引导用户前往 [Cohere 的 API 文档](https://docs.cohere.com/reference/about)以获取持续支持。
- **Slack Bot 的快速同步**：一位用户创建了一个 Cohere Slack bot，表示其在工作区中非常易于使用，并得到了社区的即时好评。
   - 对话强调了快速模型处理的重要性，因为 Slack 对 bot 有 3 秒响应规则，突显了对响应式 AI 模型的需求。
- **伦敦 AI 爱好者集会**：**Cohere For AI** 宣布将于 **7 月 10 日**在伦敦举行活动，重点关注**多语言 AI (multilingual AI)**，包括闪电演讲和 [Expedition Aya](https://sites.google.com/cohere.com/expedition-aya/home) 的启动等精彩活动。
   - **Expedition Aya** 是一项旨在突破多语言 AI 模型界限的全球挑战，为团队提供独家资源、API 额度，并有机会因显著贡献赢得周边产品和奖品。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **模型评估迷宫 (Model Evaluation Maze)**：一篇文章详细介绍了评估微调后的 **LLM** 进行结构化数据提取的复杂现状，强调了复杂的指标体系以及在没有可靠服务来维护评估的情况下过程的乏味。重点在于准确率指标和经常拖慢工作进度的隐藏代码层。
   - 用户强调了 **LM 评估**日益增长的挑战，理由是资源需求大以及随着时间的推移难以保持评估的完整性。没有分享额外的资源或图表。
- **phi-CTNL 的重大胜利**：一篇开创性的论文介绍了 **phi-CTNL**，这是一个仅有 100 万参数的精简 **LLM**，展示了它在各种学术基准测试中的完美得分，以及类似于 grokking 的金丝雀预测（canary prediction）能力。该论文的[摘要](https://arxiv.org/abs/2309.08632)介绍了该模型实力的全部细节。
   - 这个基于 **Transformer** 的 **LLM** 在专门策划的数据集上进行了预训练，因其能够以极高的精度预测评估基准而脱颖而出，引发了 **AI** 工程社区关于这种灵活而强大的模型潜在应用的讨论。
- **AIW+ 问题被破解**：一位用户展示了 **AIW+ 问题正确解法**已获验证的证据，建议使用图表来展示经过正式检查的答案。他们引用了 **Claude 3 Opus** 模型的准确回答作为验证。
   - 该解决方案的确认引发了关于问题陈述中所使用的假设及其如何直接影响结果的讨论。研究人员被敦促仔细检查支撑这些 **AI** 谜题的逻辑。
- **Terminator 架构师**：提出了一种名为“Terminator”的新模型架构，通过消除残差（residuals）、点积注意力（dot product attention）和归一化（normalization），彻底背离了传统设计。
   - **Terminator** 模型在社区中被分享，并为那些有兴趣探索其独特结构和对模型开发潜在影响的人提供了[论文](https://arxiv.org/pdf/2401.17948)链接。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Chainlit 打造音频解决方案**：一位 **AI** 工程师将用于语音检测的 **SileroVAD** 与用于转录的 **whisper-fast** 相结合，并通过 [Chainlit 语音助手示例](https://github.com/Chainlit/cookbook/tree/main/audio-assistant) 为同行提供建议。还评估了 **elevenlabs (turbo)**、**playht** 和 **deepgram** 等 **TTS** 替代方案，以优化音频工作流。
   - 进一步的讨论围绕**知识图谱**以及 **Lang Graph** 在 **AI** 中的利用展开，社区成员积极寻求将图技术嵌入 **AI** 系统的更深层见解。
- **Dask 深入数据维度**：在处理大型数据集（特别是来自 **USPTO Kaggle 竞赛**的数据）时，使用 **Dask** 导致了**内存溢出 (OOM) 错误**。讨论转向了使用 **Modal** 高效执行 **Dask** 作业的策略，旨在管理海量数据。
   - 一位从业者向小组询问了在 **Modal** 上运行 **Dask 作业**的成功案例，暗示了 **Modal** 在更好地适应高需求计算工作负载方面的潜力。
- **Autotrainer 还是别的？这是个问题**：**Autotrainer** 的角色受到质疑，不确定它是属于 **Axolotl** 的功能还是 **Huggingface autotrain**。社区正努力确定其归属。
   - **Autotrainer** 的来源仍不清楚，至少有一位成员寻求澄清，并在调查后推测其与 **Huggingface autotrain** 有关。
- **OpenAI 的慷慨引发愧疚感**：**OpenAI** 的定价在用户中引起了幽默与愧疚交织的情绪，有人感叹无法在提供的 3 个月期限内用完其 **$500 额度**。
   - 另一位成员也表达了这种情绪，他风趣地使用了*倒脸表情符号*来表达对这种低消耗额度的复杂心情。
- **幻灯片支线任务**：围绕视频幻灯片的位置展开了对话——Remi1054 和 **jt37** 等成员正在搜寻通常托管在 **Maven** 上的难以找到的演示材料。
   - 搜寻仍在继续，**hamelh** 表示并非所有演讲者（如 **Jo**）都会轻易分享他们的幻灯片，这迫使从业者直接请求访问权限。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Hexagen.World 揭晓新地点**：来自 [Hexagen.World](https://Hexagen.World) 的全新**地点**（locations）已推出，为用户扩展了数字地形。
   - 该公告引发了成员们对这些 **Hexagen.World** 新增内容的潜在用途和开发的兴趣。
- **AI Town 停靠 Docker 之岸**：社区呼吁提供 AI Town 的 **Docker 适配**，讨论了增强便携性和简化设置的好处。
   - 一位积极的成员分享了一个 [GitHub 指南](https://github.com/Ikkitsuna/AI-Town-Windows-Page-Setup-WSL-method)，用于在 **Windows 上使用 WSL** 设置 AI Town，并建议将其集成到主仓库中。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Apple 潇洒夺得 OpenAI 席位**：**Phil Schiller** 将作为观察员加入 OpenAI 董事会，使 **Apple** 在旨在增强其 Apple Intelligence 产品的战略合作伙伴关系中占据优势，详见[此处](https://www.bloomberg.com/news/articles/2024-07-02/apple-to-get-openai-board-observer-role-as-part-of-ai-agreement)。
   - **AI 社区**反应热烈，**Microsoft 的巨额投资**与 Apple 的精明举动形成鲜明对比，引发了关于 Microsoft 劣势的辩论和一些幸灾乐祸，如[此讨论](https://x.com/BartokGabi17/status/1808242102750568799)所示。
- **科技巨头在 OpenAI 关系上展开角逐**：社区对话深入探讨了 **OpenAI 合作伙伴关系的业务动态**，审视了 Apple 获得观察员席位与 Microsoft 直接财务投资相比的战略胜利。
   - 见解和嘲讽贯穿于讨论中，参与者将“观察员席位交易”比作一场国际象棋比赛，Apple 以极小的支出实现将军，而 Microsoft 的巨额支出既引起了旁观者的钦佩，也引发了笑声。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **网络以智慧取胜**：[Be Better, Not Smaller](https://www.dbreunig.com/2024/07/01/be-better-not-smaller.html) 分析了早期移动互联网服务（如 **WAP**）的失误，将其比作现代 AI 产品。它解释了 iPhone 时代之前移动浏览的局限性，并为当前产品提出了更好的方法。
   - 文章鼓励当前的 AI 开发者优先考虑提升用户体验，而不仅仅是适应更小的平台。**移动浏览**曾经就像是*“通过钥匙孔窥视互联网”*。
- **政府礼品引人注目**：一篇 [Scoop 文章](https://thescoop.org/archives/2024/06/22/all-foreign-gifts-around-us/index.html) 揭示了美国官员从外国实体收到的各种不寻常礼品，例如**鳄鱼保险**和**金牌**。
   - 它讨论了这些礼品在数据管理方面的困难，指出它们通常是非结构化格式且存在存储问题，这反映了政府数据处理中更大的问题。*“这些外国礼品确实就是数据。”*




---

# 第 2 部分：按频道划分的详细摘要和链接


{% if medium == 'web' %}

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1257413266599055522)** (193 条消息🔥🔥): 

> `LM Studio 更新与问题`, `LM Studio 中的 TTS 集成`, `本地使用的模型推荐`, `加载大模型的挑战`, `Gemma 2 模型性能与更新`

- **LM Studio 更新与问题**：成员们讨论了与 **LM Studio** 相关的各种问题和更新，包括网络错误、更新特定模型（如 **Gemma 2**）的需求，以及关于系统要求和性能改进的问题。
   - “我们计划明天（美国时间）发布更新”的消息被公布，随后社区互动解决了有关 **VRAM** 和 **加载大模型** 的问题。
- **LM Studio 中的 TTS 集成疑问**：一位成员询问了 **LM Studio** 内部的 **Text-to-Speech (TTS) 集成** 情况，以及在 **同一台服务器** 上同时运行 **TTS** 和 **LM Studio** 的可行性。
   - 目前尚未给出明确答复，这表明用户对探索该应用内的 **TTS 功能** 有进一步的兴趣。
- **本地模型推荐与问题**：成员们推荐了各种用于游戏翻译等任务的 **本地模型**，特别是 **Meta-LLaMA 3**，并讨论了模型翻译过于 **字面化** 的问题。
   - 专家们更青睐 **Gemma 2** 等模型的写作风格和 **性能**，特别推荐在 **VRAM** 有限的系统上使用特定的量化版本。
- **LM Studio 加载大模型的挑战**：尽管硬件配置充足，仍有几位用户报告了在 **LM Studio** 中加载 **30B 和 70B 参数** 等大模型时遇到的问题。
   - 据指出，分片模型可能会遇到与 **RAM** 限制相关的加载问题，且通常需要 **全 GPU offload** 才能获得最佳性能。
- **Gemma 2 模型性能与更新**：**Gemma 2** 的最新更新显著提升了性能，用户注意到更新后模型具有更好的 **连贯性和叙事能力**。
   - 讨论内容包括为了获得最佳性能以及与 **Langchain** 和 **Local Server** 等应用的集成，有必要重新下载更新后的 **GGUF** 文件。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://huggingface.co/bartowski/gemma-2-9b-it-GGUF">bartowski/gemma-2-9b-it-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/google/gemma-1.1-2b-it-GGUF">google/gemma-1.1-2b-it-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://huggingface.co/bartowski/WizardLM-2-8x22B-GGUF/tree/main">bartowski/WizardLM-2-8x22B-GGUF at main</a>：未找到描述</li><li><a href="https://github.com/facebookresearch/fairseq/tree/nllb/?tab=readme-ov-file">GitHub - facebookresearch/fairseq at nllb</a>：用 Python 编写的 Facebook AI Research 序列到序列工具包。 - GitHub - facebookresearch/fairseq at nllb</li><li><a href="https://huggingface.co/docs/transformers/main/en/model_doc/nllb#generating-from-any-other-language-than-english">NLLB</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8197">由 abetlen 提交的 Pull Request #8197 · ggerganov/llama.cpp：为 Gemma2 添加 attention 和最终 logit soft-capping，更新缩放因子</a>：此 PR 添加了缺失的 attention 层和最终 logit soft-capping。实现参考自 Hugging Face Transformers。此外，Gemma2 应用了 hidden_size / ... 的 pre-attention 缩放。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1257460639459315773)** (148 条消息🔥🔥): 

> `模型性能与加载时间`, `Phi-3 Mini 更新`, `在不同硬件上运行模型`, `LM Studio 导致 GPU 待机温度过高`, `为 LM Studio 量化视觉模型`

- **模型性能与加载时间**：用户讨论了大型模型在不同硬件配置上的性能问题和加载缓慢，例如由于 RAM 不足，**70b 模型**在 NVME 存储上运行速度仅为 **0.04 tokens/s**。
   - 一位用户建议尝试更小的量化模型，如 **Mistral Grok 7b Q5**，它在受限硬件上的表现更好，速度可达 **3-5 tokens/s**。
- **Phi-3 Mini 更新发布**：Microsoft 发布了 **Phi-3 Mini** 的更新，解决了**指令遵循 (instruction following)**和**输出结构 (output structure)**问题。更新后的模型链接可在 [Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) 上找到。
   - 用户正在测试更新后模型的性能，一些人指出加载 **128k context model** 时存在问题，而另一些人则难以让 **Mini 4k Q8_0** 正常运行。
- **用户受困于 LM Studio 中的 GPU 待机温度**：成员们观察到运行 LM Studio 时 GPU 的**待机温度较高**，报告的案例包括 P40 GPU 在 **45% 风扇速度下待机温度为 47C**。
   - 该问题在 **Ollama** 或 **ComfyUI** 等其他应用中并未出现，表明这可能是 LM Studio 特有的问题。
- **在各种硬件配置上运行模型**：讨论包括配置硬件以有效运行大型模型，建议如使用**不同的磁盘**进行输出以加快进程，以及利用 **NVME 而非 HDD**。
   - 还有建议使用**外部脚本**或 **langchain** 等程序，使模型能够访问互联网或运行时数据，突显了 LM Studio 的灵活性和局限性。
- **探索适用于 LM Studio 的量化视觉模型**：用户表示有兴趣在 LM Studio 中运行 **dolphin-vision-72B** 等**大型视觉模型**，并讨论了相关的挑战。
   - 虽然量化可能使这些模型适应可用的 VRAM，但在 LM Studio 中使用视觉模型的经验和指导仍然有限。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://rentry.org/LMSTudioFAQ">非官方 LMStudio FAQ！</a>：欢迎来到非官方 LMStudio FAQ。在这里，您可以找到 LMStudio Discord 中最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1dtgylv/microsoft_updated_phi3_mini/">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1257798297414795357)** (1 条消息): 

> `LM Studio 0.2.27 发布`, `改进了对 Gemma 2 模型的支持`, `修复了 'invalid creation parameter' 问题`, `关于新更新的高级信息`, `Windows 版 ROCm 扩展包说明`

- **LM Studio 0.2.27 发布，支持 Gemma 2！**：LM Studio 0.2.27 现已支持 Mac (M1/M2/M3)、Windows (x86 和 ARM64) 以及 Linux (x86)，并提升了 **Gemma 9B 和 27B** 模型的性能。可以从[此处](https://lmstudio.ai)下载，或重启应用以触发自动更新。
   - 这些改进归功于 `llama.cpp` 的开发者，包括 [abetlen](https://github.com/abetlen)、[ngxson](https://github.com/ngxson)、[slaren](https://github.com/slaren) 和 [ggerganov](https://github.com/ggerganov)。社区成员可以从 [Hugging Face](https://huggingface.co/lmstudio-community) 下载模型，例如 [Gemma 9B](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF) 和 [Gemma 27B](https://huggingface.co/lmstudio-community/gemma-2-27b-it-GGUF)。
- **Bug 修复解决了 'invalid creation parameter' 问题**：最新更新修复了 [lmstudio.js issue #45](https://github.com/lmstudio-ai/lmstudio.js/issues/45) 中的 **'invalid creation parameter'** bug。用户看到的关于 GPU 支持的错误消息应该会减少。
   - [相关的 GitHub issue](https://github.com/lmstudio-ai/lmstudio.js/issues/45) 详细说明了该问题及其解决方案，确保了更流畅的性能。
- **LM Studio 中的高级更新**：`llama.cpp` 的 commit ID 已更新为 `d08c20eddedb24515a3212e2de66bdff41a26b8c`，并且 `OpenCL` 后端已重新打包进 Windows 和 Linux 版的 LM Studio 中。值得注意的是，**Gemma 2 不支持 OpenCL**。
   - 对于 Windows 上的 AMD ROCm 用户，请[参阅说明](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#amd-rocm)了解如何更新您的 ROCm 扩展包。Linux 版 ROCm 扩展包仍待发布。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://github.com/lmstudio-ai/lmstudio.js/issues/45)">Issues · lmstudio-ai/lmstudio.js</a>：LM Studio TypeScript SDK (发布前公开测试版) - Issues · lmstudio-ai/lmstudio.js</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#amd-rocm).">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1257441211250118750)** (9 条消息🔥): 

> `挪威语频道请求`, `LM Studio 更新与 AMD Radeon 7800 XT 的兼容性问题`, `处理 LM Studio 中 GPU 兼容性错误的建议`, `LM Studio 0.2.26 中 ROCM 处理方式的变化`, `Vulkan 后端支持作为潜在解决方案`

- **挪威语频道请求得到顺利处理**：一名成员请求创建挪威语频道，并提到了对指南的遵守。另一名成员澄清说，他们可以不受限制地自行创建主题，确保了灵活性。
   - 一位成员安慰道：*你应该能够自己创建主题*，强调了平台的灵活性。请求者指出：*我只是在遵守规则*，对指南表示认可。
- **AMD Radeon 7800 XT 遭遇意外兼容性障碍**：一名成员报告称，在最新的 LM Studio 更新后，他们的 AMD Radeon 7800 XT 16GB VRAM GPU 被标记为不兼容，尽管之前运行正常。他们提到在更新前体验到了出色的性能。
   - 共享了指向 [Extension-Pack-Instructions.md](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md) 的链接，解释了 ROCM 处理方式的变化。另一名成员解释说：*0.2.26 更改了 ROCM 的处理方式*。
- **LM Studio 的错误信息需要改进**：社区反馈表明需要改进错误提示信息，以帮助用户自行排除 GPU 兼容性问题。自最新更新以来，已收到多份报告。
   - *很好的建议。0.2.27 恢复了 OpenCL*，确认了一项旨在解决这些问题的更新。另一名成员建议 *Vulkan 后端支持将解决 90% 的此类问题*。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai)">未找到标题</a>：未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1257465678328500224)** (47 条消息🔥): 

> `Transformer ASIC 市场潜力`, `Sohu 预期定价`, `LLM 推理中的 VRAM 与 RAM 对比`, `LM Studio 中的 AMD GPU 支持问题`, `用于 LLM 处理的增强型 GPU 配置`

- **Transformer ASIC 具有市场潜力**：讨论表明对专为 LLM 定制的 GPU 存在需求，强调 **144GB VRAM** 和仅用于 LLM 任务的专用芯片。
   - 对初始高成本的担忧，并声称 ASIC 是大规模推理（如在 **Etched** 中）唯一可行的解决方案。
- **LLM 推理中的 VRAM 与 RAM 对比**：成员建议 **VRAM** 对于 LLM 推理任务更为关键，8GB VRAM 比 4GB *更受青睐*，以获得更好的性能。
   - 一名成员强调，与使用系统 RAM 分流相比，完全在 VRAM 中运行模型可以获得更卓越的速度。
- **LM Studio 中的 AMD GPU 支持问题**：注意到 LM Studio 0.2.24 版本无法识别 **AMD GPU**，甚至是嵌入式 GPU。
   - 一名成员指出，**OpenCL GPU 支持在 llama.cpp 中已被弃用**，从而导致了这些问题。
- **用于 LLM 处理的增强型 GPU 配置**：一名成员正在将用于 LLM 推理的设备从 **3x 风冷升级为 5x 水冷 4090**，总计达到显著的 **120GB VRAM**。
   - *散热和功耗管理*是主要关注点，讨论涉及 PSU（电源）需求和外部散热器。
- **在不同硬件上优化 LLM**：成员讨论了在包括 **Snapdragon** 和多 GPU 系统在内的各种配置上运行 LM Studio，成功程度各异。
   - 报告称在 7b 模型上达到 **19 t/s**，并需要调整配置以获得最佳效果，特别是与 **SillyTavern** 等集成时。
  

---

### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1257771555648438312)** (2 messages): 

> `LM Studio 与 AutoGen 中 group chat 功能的兼容性`, `Llama 7b Instruct 的错误处理`, `在 LM Studio 中使用多个本地模型的解决方案`

- **LM Studio Group Chat 功能的兼容性问题**：一位成员在使用 Llama 7b Instruct 时遇到了 LM Studio 的 **groupchat 功能**问题，由于“content”字段不能为空的要求，触发了 **BadRequestError: Error code: 400**。他们注意到该错误在 **OpenAI 模型**上不会发生。
   - 另一位成员确认多位用户也遇到了类似问题，并建议在 Discord 中搜索解决方案。他们分享了一个 [notebook](https://microsoft.github.io/autogen/docs/topics/non-openai-models/local-lm-studio/) 链接，演示了如何通过 LM Studio 在 AutoGen 中使用多个本地模型。
- **使用 LM Studio 的多模型服务 (Multi-Model Serving)**：分享的 [notebook](https://microsoft.github.io/autogen/docs/topics/non-openai-models/local-lm-studio/) 详细介绍了如何使用自 **0.2.17** 版本起提供的 LM Studio **multi-model serving** 功能。它展示了如何启动“Multi Model Session”并加载模型进行本地托管。
   - 示例包括使用两个不同的本地模型 **Phi-2** 和 **Gemma** 创建一个喜剧聊天，并演示了如何创建配置以及启动多模型服务服务器。该成员强调此方案可解决任何类似的集成问题。

**提及的链接**: <a href="https://microsoft.github.io/autogen/docs/topics/non-openai-models/local-lm-studio/">LM Studio | AutoGen</a>: Open In Colab

---

### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1257459017085882471)** (17 messages🔥): 

> `ROCm 扩展性能`, `代码基准测试结果`, `用于 AI 的 NVIDIA GPU`, `Gemma 2 模型问题`, `ROCm 特定 GPU`, `Linux ROCm 扩展包测试`

- **ROCm 扩展提升性能**：一位成员指出，在最新的 **24.6.1 Adrenalin 驱动**下，其 **6900 XT** GPU 使用 ROCm 扩展后性能有了**显著提升**。基准测试显示，使用 **codestral 22b q4_k_m** 时，速度从 **9.29 tok/s** 提升到了 **26.75 tok/s**。
   - 另一位成员建议对比差异，而另一位成员则在重新考虑是否购买 **NVIDIA GPU** 以获得额外速度。
- **Gemma 2 模型加载失败**：一位用户报告无法运行 **Gemma 2**，在 LM Studio **0.2.27** 版本中清除缓存并运行脚本后，收到 **“failed to load model”** 错误。错误提示在 **AMD 6900 XT** GPU 上 **“gemma2”** 的 **“unknown model architecture”**。
   - “重新下载/干净安装总是一个好的调试步骤，”另一位成员建议，如果问题持续，请在支持频道发布帖子。
- **ROCm 在特定 GPU 上运行**：一位成员提到 **ROCm 仅在特定 GPU 上运行**，并将 **6900 XT** 列为兼容型号。他们报告在 **8k tokens** 下运行 **Gemma 2** 且未耗尽显存，相比 **2k tokens** 有了显著改进。
   - 该用户还表示，由于性能表现良好，他们可能会取消下载更高量化版本的计划。
- **征集 Linux ROCm 扩展测试**：社区发起号召，请成员协助测试适用于 **0.2.27** 版本的最新 **Linux ROCm 扩展包**。说明包括安装步骤，以及在运行特定脚本后检查 **“Settings->GPU Backend Type”** 下是否存在 **“ROCm llama.cpp”**。
   - 预先感谢测试者，并鼓励他们反馈测试结果。

---

### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1257704241749819555)** (2 条消息): 

> `Microsoft 将 Phi 3 mini 更新至 Phi 3.1`, `lmstudio community 的 Gemma 2 模型更新`

- **Microsoft 发布 Phi 3.1 Mini，带来巨大改进**：Microsoft 已将 **Phi 3 mini** 更新至 **Phi 3.1**，宣称其性能大幅提升，指令遵循能力更强，且增强了输出结构化能力。该更新现已在 [lmstudio community](https://huggingface.co/lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF) 上线。
   - 一位用户强调了此次更新的重要性，表示：*我们认为它值得拥有自己的名字，因为这次更新非常巨大。* 该模型采用了由 [bartowski](https://huggingface.co/bartowski) 基于 `llama.cpp` 版本 [b3278](https://github.com/ggerganov/llama.cpp/releases/tag/b3278) 提供的 **GGUF 量化**。
- **Gemma 2 模型已针对最新的 lmstudio 更改进行更新**：lmstudio community 中的 **Gemma 2 模型** 已更新以包含最新更改。用户可以放心重新下载并在 **0.2.27** 版本中使用。
   - 提供了更新模型的链接：[Gemma 2 9b it GGUF](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF) 和 [Gemma 2 27b it GGUF](https://huggingface.co/lmstudio-community/gemma-2-27b-it-GGUF)。

**提到的链接**：<a href="https://huggingface.co/lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF">lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF · Hugging Face</a>：未找到描述

  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1257422676465811626)** (50 条消息🔥): 

> `更新后 LM Studio 加载命令出错`, `Discord 机器人出现 TokenInvalid 错误`, `配置 LM StudioClient`, `Discord 机器人 intents 和权限`, `调试并修复 Discord 机器人代码`

- **更新后 LM Studio 加载命令失败**：在将 LM Studio 从 **0.2.25 更新至 0.2.26** 以及 lmstudio.js **SDK 从 0.0.3 更新至 0.0.12** 后，由于通信协议不兼容，用户在调用 `await client.llm.load(modelPath);` 时遇到错误。
   - 错误信息提示需要 `creationParameter.loadConfigStack`，这促使用户在 [GitHub 上提交了 issue](https://github.com/lmstudio-ai/lmstudio.js/issues)。
- **Discord 社区解决了 TokenInvalid 错误**：一位用户在设置 Discord 机器人时遇到 **TokenInvalid** 错误，随后咨询了 Discord.js 社区寻求指导。问题追溯到未允许的 **MessageContent** intents。
   - 在为机器人启用必要的 intents 后，用户成功解决了问题，使机器人能够正常登录并运行。
- **带有有效 token 的机器人 Client.login 失败**：尽管 token 配置正确，用户的 Discord 机器人仍登录失败并提示 token 无效。基础运行代码验证了 token 本身没有问题。
   - 进一步调查发现，未启用 **MessageContent** intent 导致了登录失败。
- **LM StudioClient 配置**：一位用户分享了他们在 LM Studio 中加载特定模型的配置，包含 `gpuOffload` 和 `contextLength` 等自定义设置。
   - 虽然最初被注释掉了，但他们的 daemon 设置确保了模型保持加载状态，允许用户直接引用。
- **为自定义机器人交互提供协助**：一位用户寻求帮助以扩展其 Discord 机器人的功能，包括响应提及（mentions）和本地对话保存。
   - 协作调试和指导促成了机器人权限的调整，并修复了其 TypeScript 代码中的一个实现缺陷。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/lmstudio-ai/lmstudio.js/issues">Issues · lmstudio-ai/lmstudio.js</a>：LM Studio TypeScript SDK (pre-release public alpha) - Issues · lmstudio-ai/lmstudio.js</li><li><a href="https://github.com/mrdjohnson/lmstudio-discord-bot/tree/main">GitHub - mrdjohnson/lmstudio-discord-bot: A tutorial for creating a Discord bot that responds using LM Studio! This code is based on a blogpost found here: https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6</a>：创建一个使用 LM Studio 进行响应的 Discord 机器人的教程！此代码基于此处的博客文章：https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6 - mrdjohnson/lm...</li><li><a href="https://github.com/lmstudio-ai/lmstudio.js#using-an-already-loaded-model">GitHub - lmstudio-ai/lmstudio.js: LM Studio TypeScript SDK (pre-release public alpha)</a>：LM Studio TypeScript SDK (pre-release public alpha) - lmstudio-ai/lmstudio.js
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1257796484473294908)** (1 条消息):

> `通过 KerasNLP 访问大量新的 Transformers 模型微调版本。`, `使用新 API 通过列名搜索 Hugging Face 数据集。`, `Transformers 4.42 发布，包含新模型和功能。`, `近 10 万个公开模型使用 Hub 存储 tensorboard 日志。`, `宣布推出本地 Gemma，用于私密和安全的使用。`, `AWS 在 Hugging Face 上发布 Chronos 数据集。`, `Google 发布高质量的 Gemma 2 LLM。`, `Real-time Detection Transformer (RT-DETR) 已在 Hugging Face 上线。`, `Florence-2 通过 WebGPU 在浏览器中本地运行。`, `视觉语言模型入门介绍发布。`, `发布了新的具有挑战性的 LLM 排行榜。`, `Argilla 宣布推出 Data Explorer 视频系列。`, `用于分布式训练的高效 PyTorch dataloader。`, `使用 elastic search 的新 Gemma RAG 方案。`

- **Transformers 4.42 带来新特性**：根据 [发行说明](https://github.com/huggingface/transformers/releases/tag/v4.42.0)，**Transformers 4.42** 已发布，包含 **Gemma 2**、**RT-DETR**、**InstructBlip** 和 **LLaVa-NeXT-Video** 等新模型，以及 tool usage、RAG 支持、GGUF 微调和 quantized KV cache。
   - 该更新被描述为拥有**惊人的功能**，并获得了*大量热情的社区响应*。一位成员评论道：*Enjoy! 🥳*。
- **AWS 发布 Chronos 数据集**：**AWS** 已在 Hugging Face 上发布了 **Chronos 论文中使用的所有数据集**，包括预训练和评估数据集（[更多信息](https://x.com/solitarypenman/status/180642160568323294)）。这还包括一个用于评估 Chronos 模型的脚本。
   - 一份热情的公告提到，此次发布包含了与论文中相同设置的评估，并被描述为 *🚀🚀🚀*。
- **近 10 万个公开模型使用 Hub**：近 **10 万个公开模型**使用 Hub 存储 `tensorboard` 日志，允许在 checkpoint 旁追踪训练日志（[来源](https://x.com/Wauplin/status/1808074557128855750)）。Metrics 选项卡将所有内容整合在一个地方。
   - 一位社区成员庆祝了这一里程碑，称其为*在一个地方追踪所有内容*的一种方式。
- **Google 带来高质量 Gemma 2 LLM**：Google 发布了拥有 27B + 9B 参数的 **Gemma 2**，旨在提供**高质量**且对开发者友好的尺寸（[详情](https://huggingface.co/blog/gemma2)）。
   - 此次发布被强调为一项**重要**补充，社区成员对该模型*令人印象深刻的能力*赞不绝口。
- **Florence-2 通过 WebGPU 在本地运行**：得益于 Transformers.js，Microsoft 的新视觉基础模型 **Florence-2** 可以使用 **WebGPU** 在浏览器中 **100% 本地**运行（[演示](https://x.com/xenovacom/status/1805990110065803492)）。
   - 它支持 **image captioning**、**optical character recognition** 和目标检测等任务，被一位成员用 *WOW!* 来形容。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/julien_c/status/1806366482269352232)">来自 Julien Chaumond (@julien_c) 的推文</a>：Keras 🤝 HF</li><li><a href="https://x.com/vanstriendaniel/status/1807814430262202465)">来自 Daniel van Strien (@vanstriendaniel) 的推文</a>：使用新的实验性 API 通过列名搜索 @huggingface 数据集！该 API 允许你：- 搜索包含 context 的问答数据集 - 查找 alpaca 风格的数据集 - 定位 D...</li><li><a href="https://x.com/osanseviero/status/1806440622007447631)">来自 Omar Sanseviero (@osanseviero) 的推文</a>：Transformers 4.42 发布了，包含许多惊人的功能🥳 🔥新模型：Gemma 2、RT-DETR（目标检测）、InstructBlip 和 LLaVa-NeXT-Video 🔧工具使用和 RAG 支持 👀GGUF 微调 🤏Qu...</li><li><a href="https://x.com/Wauplin/status/1808074557128855750)">来自 Wauplin (@Wauplin) 的推文</a>：已有近 10 万个公开模型使用 Hub 存储 𝚝𝚎𝚗𝚜𝚘𝚛𝚋𝚘𝚊𝚛𝚍 日志！将训练日志与 checkpoint 一起存储，让你可以在 Metrics 选项卡中一站式跟踪所有内容...</li><li><a href="https://x.com/reach_vb/status/1807830966515519667)">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：介绍 Local Gemma！💎 100% 本地、私密且安全 - 随时随地运行！支持 CUDA, mps, cpu - 带有预设以实现最快速度 - 基于 Transformers 构建，我们确保 1:1 的 ge...</li><li><a href="https://x.com/solitarypenman/status/1806421605683232947)">来自 Abdul Fatir (@solitarypenman) 的推文</a>：🚀🚀🚀 我们刚刚在 Hugging Face 上发布了 Chronos 论文中使用的所有数据集。这包括预训练和评估（域内和 zero-shot）数据集。我们还开源了一个脚本来...</li><li><a href="https://x.com/mervenoyann/status/1807790959884665029)">来自 merve (@mervenoyann) 的推文</a>：Real-time DEtection Transformer (RT-DETR) 已登陆 @huggingface Transformers 🤩 采用 Apache 2.0 许可证 😍 DETR 在实时目标检测上能击败 YOLO 吗？继续阅读 👀</li><li><a href="https://x.com/xenovacom/status/1805990110065803492)!">来自 Xenova (@xenovacom) 的推文</a>：Florence-2，来自 Microsoft 的新视觉基础模型，现在可以通过 Transformers.js 在你的浏览器中利用 WebGPU 100% 本地运行！🤗🤯 它支持图像字幕、光学字符识别...</li><li><a href="https://x.com/mervenoyann/status/1805910433024380978)">来自 merve (@mervenoyann) 的推文</a>：刚刚发布：视觉语言模型（又称 image-text-to-text）简介</li><li><a href="https://x.com/ben_burtenshaw/status/1806291858835837333)">来自 Ben Burtenshaw (@ben_burtenshaw) 的推文</a>：🚀 很高兴推出我们的新系列，由 @argilla_io 制作的 Data Explorer！🎥 我们深入探讨数据集及其对模型性能的影响。我们的第一集探索了由 @hannahrosekir 制作的 PRISM 数据集...</li><li><a href="https://x.com/TheZachMueller/status/1807394438689214930)">来自 Zach Mueller (@TheZachMueller) 的推文</a>：如何让 @PyTorch dataloaders 在分布式训练期间高效工作？这是我使用 @huggingface Accelerate 的 dataloaders 制作的一个视频教程，展示了我们是如何做到的 https://www.yo...</li><li><a href="https://x.com/mervenoyann/status/1806267855559623115)">来自 merve (@mervenoyann) 的推文</a>：使用 @elastic search 和 @huggingface 的新 Gemma RAG 方案 🧑🏻‍🍳📖 见下方 ⇓
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1257412386755838143)** (318 条消息🔥🔥): 

> `下载 Falcon40B 时遇到问题`, `Falcon 40B 与 LLaMA 3 等其他模型的比较`, `RAG (Retrieval-Augmented Generation) 技术与挑战`, `在 LLM 中解析和管理大型转录文本`, `转录文本的摘要技术`

- **Falcon40B 下载困扰**：一位用户在 Linux 机器上花费了 30 分钟下载约 90GB 的 Falcon40B 文件，结果却发现文件不见了。讨论指出，这些文件很可能存储在 HF 的缓存目录中，并建议使用 Python 代码片段检查路径。
   - *“它们存储在 HF 的缓存目录中，不会再次下载，且 HF 会自动识别其位置。”*
- **Falcon 40B 对比 LLaMA 3**：关于 Falcon 40B 与 LLaMA 3 等新模型的性能和相关性存在争论。有人指出 Falcon 40B 已被视为过时，目前已有更好且更小的模型可用。
   - *“现在有比它好得多的更小的模型，例如 LLaMA 3 8B。”*
- **RAG 中的挑战**：用户讨论了利用 chunks, overlaps, L2 thresholds 以及 FAISS top_p 来实现有效 RAG 的复杂性。必须在检索文档时取得平衡，以避免破坏上下文。
   - *“这是平衡 chunk_size, overlap, L2 thresholds, top_p (在 FAISS 中) 的精妙艺术……因此需要创建一个评估数据集，并在不同的配置上运行批量推理。”*
- **在 LLM 中处理大型转录文本**：一位用户寻求关于为 LLaMA 3 模型管理大型转录文本的建议，但被建议考虑具有更大上下文窗口的模型（如 Mistral）。该转录文本超过了 20k tokens，给处理能力有限的模型带来了挑战。
   - *“取决于那 1000 行的具体内容，但我认为那确实有点太多了。改用 Mistral 吧。”*
- **摘要策略**：建议对大型转录文本进行摘要，以防止破坏上下文并提高处理效率。强调了智能分段策略对于有效摘要至关重要。
   - *“无论如何，如果不能智能地拆分转录文本，解决方案就不会奏效……阅读这篇关于 RAG 中 chunking 策略的文章以了解更多背景信息。”*
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lumiere-video.github.io>">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Cognitive-Lab/Tokenizer_Arena">Tokenizer Arena - Cognitive-Lab 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/?">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="http://huggingface.co">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2-7B-Instruct">Qwen/Qwen2-7B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/llava-hf/bakLlava-v1-hf">llava-hf/bakLlava-v1-hf · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/ClementDelangue/status/1808131474694037935">来自 clem 🤗 (@ClementDelangue) 的推文</a>: 这是我的招聘日。谁绝对应该加入 @huggingface？</li><li><a href="https://tenor.com/view/dancing-cat-dance-cat-cat-meme-chinese-cat-gif-12629347036627000898">Dancing Cat Dance GIF - Dancing cat Dance Cat - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/sanchitgandhi99/status/1807877591501820260">来自 Sanchit Gandhi (@sanchitgandhi99) 的推文</a>: 它是如何工作的？🧐 得益于 CPU offloading，只有最大的层需要加载到 GPU 上 🏋️‍♀️ 对于 Gemma-2，这就是 LM Head，使用 4-bit bitsandbytes 量化仅需 5GB 🔍 ...</li><li><a href="https://huggingface.co/discord-community">discord-community (Hugging Face Discord 社区)</a>: 未找到描述</li><li><a href="https://tenor.com/view/lil-yachty-drake-oprahs-bank-account-meme-laptop-gif-20803826">Lil Yachty Drake GIF - Lil Yachty Drake Oprahs Bank Account - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/reach_vb/status/1807830966515519667">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>: 介绍 Local Gemma！💎 100% 本地、私密且安全 - 随时随地运行！cuda, mps, cpu - 带有预设以尽可能快地运行 - 站在 transformers 的肩膀上，我们确保 1:1 ge...</li><li><a href="https://tenor.com/view/ineedit-needit-spongebob-squarepants-need-it-gif-4883495">Need It GIF - Ineedit Needit Spongebob Squarepants - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/reach_vb/status/1807830966515519667https://x.com/reach_vb/status/1806731975618626004">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>: 介绍 Local Gemma！💎 100% 本地、私密且安全 - 随时随地运行！cuda, mps, cpu - 带有预设以尽可能快地运行 - 站在 transformers 的肩膀上，我们确保 1:1 ge...</li><li><a href="https://tenor.com/view/handcuffs-shackles-arrest-gif-11341464">수갑 쇠고랑 체포 GIF - Handcuffs Shackles Arrest - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/chefs-kiss-french-chef-perfect-dish-excellent-food-perfection-gif-20341505">Chefs Kiss French Chef GIF - Chefs Kiss French Chef Perfect Dish - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/Vipitis/shadertoys-dataset">GitHub - Vipitis/shadertoys-dataset: 数据集的 WIP 重构</a>: 数据集的 WIP 重构。通过在 GitHub 上创建账号来为 Vipitis/shadertoys-dataset 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/nroggendorff/dequantize">Dequantize - nroggendorff 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/alirezamika/autoscraper">GitHub - alirezamika/autoscraper: 一个智能、自动、快速且轻量级的 Python 网络爬虫</a>: 一个智能、自动、快速且轻量级的 Python 网络爬虫 - alirezamika/autoscraper
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1257569575596199987)** (2 条消息): 

> `用于学习的 diffusion models`, `高级 CNN 主题资源`

- **学习 Diffusion Models 时算力耗尽**: 一位用户提到他们在学习 **diffusion models** 时“算力耗尽”，当时使用的是免费的 Google Colab 资源。
   - *未提供额外的讨论或链接。*
- **寻求高级 CNN 资源**: 另一位用户询问推荐的书籍或资源，以学习 **CNN** 的高级主题，如 **ViT** 和 **Unets**，包括视频处理。
   - *未提供额外的讨论或链接。*
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1257462307143815208)** (10 条消息🔥): 

> `AI 攻克 Meshes 生成`, `前 5 名 Python 框架`, `AI 与社会论文`, `在机器人上运行 Transformer`

- **AI 刚刚攻克了 Meshes**：分享了一个名为 *AI just figured out Meshes* 的视频，展示了一个 [YouTube 视频](https://www.youtube.com/watch?v=rQolOT4tuUY) 并链接了 [原始论文](https://huggingface.co/papers/2406.10163)、[Demo](https://huggingface.co/spaces/Yiwen-ntu/MeshAnything) 和 [代码](https://github.com/buaacyw/MeshAnything)。该项目与 Hugging Face 有密切关联。
   - *IndividualKex* 感谢了一位用户对该 AI Mesh 项目酷炫程度的称赞。*osanseviero* 强调这项工作出自才华横溢的 <@947993236755054633> 之手。
- **在机器人上运行 Transformer**：分享了一个 [GitHub 仓库](https://github.com/mbodiai/embodied-agents) 链接，展示了一个名为 *embodied-agents* 的项目，该项目允许仅用几行 Python 代码在任何机器人上运行 Transformer。该仓库详细介绍了如何将最先进的 Transformer 模型无缝集成到机器人技术栈（robotics stacks）中。
   - 该项目因其简化 Transformer 在机器人应用中的承诺而备受关注。“只需几行 Python 代码即可在任何机器人上运行 Transformer”是一个引人注目的亮点。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/Purring_Lynx/status/1805608003833352634">来自 Purring Lynx (ℵ/acc) (@Purring_Lynx) 的推文</a>: Serial Experiments: Lain  Join the Wired: https://discord.gg/8ukzB9H2Yv  Lain Thread: https://discord.com/channels/1233520657778868235/1254443013493624927  引用 Purring Lynx (ℵ/acc) (@Purring_Lynx)...</li><li><a href="https://www.youtube.com/watch?v=hQDRMmYxpi4">45 秒内了解前 5 名 Python 框架：从最难到最简单</a>: 探索用于 Web 开发的最佳 Python 框架，从最难到最简单进行排名！无论你是在构建大型应用程序还是简单的...</li><li><a href="https://www.youtube.com/watch?v=rQolOT4tuUY&ab_channel=IndividualKex">AI 刚刚攻克了 Meshes</a>: 原始论文: https://huggingface.co/papers/2406.10163 demo: https://huggingface.co/spaces/Yiwen-ntu/MeshAnything 代码: https://github.com/buaacyw/MeshAnythi...</li><li><a href="https://github.com/mbodiai/embodied-agents">GitHub - mbodiai/embodied-agents: 将最先进的 Transformer 模型无缝集成到机器人技术栈中</a>: Seamlessly integrate state-of-the-art transformer models into robotics stacks - mbodiai/embodied-agents
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1257495653622349863)** (5 条消息): 

> `越南语视觉语言模型`, `AI 与电子商务项目`, `用于代码纠错的 CriticGPT`, `Embodied Agents 工具包稳定版发布`

- **视觉语言模型创新图像描述**：一个团队推出了针对越南语的视觉语言模型 **Vistral 7B**，该模型在图像描述任务中表现出色。该模型基于 **LLaVA**、Vistral LLM 以及来自 *Gemini API* 的合成生成数据集；详细信息和资源可在其 [GitHub](https://github.com/hllj/Vistral-V) 和 [HuggingFace](https://huggingface.co/Vi-VLM/Vistral-V-7B) 上获取。
   - 他们使用了 **Siglip** 图像编码器，并利用 Llava 方法进一步增强视觉能力。欢迎通过其发布的 [demo](https://964c3125aaea36d527.gradio.live/) 提供反馈并对模型性能进行进一步研究。
- **AI 电子商务项目展示**：[Tony Assi](https://huggingface.co/spaces/tonyassi/AI-Ecommerce-Fashion) 分享了一系列专注于计算机视觉和 Stable Diffusion 的 AI 与电子商务项目。这些项目涵盖了电子商务领域的各种创新实现。
   - *尽情体验！*
- **CriticGPT 视频分析**：分享了一段 YouTube 视频 [OpenAI 发布 CriticGPT 以纠正 GPT-4 的错误](https://youtu.be/4PgcaIfwLjo)，讨论了 OpenAI 的新模型：**CriticGPT**。该模型旨在识别 GPT-4 生成代码中的错误，标志着在改进纠错能力方面迈出了重要一步。
   - *Harshit Tyagi* 邀请大家对视频中关于论文的分析提供反馈。
- **Embodied Agents 工具包发布**：开源库 **Embodied Agents** 的稳定版本已在 [GitHub](https://github.com/MbodiAI/mbodied-agents) 上发布。该工具包使用户能够以极少的代码将最先进的多模态 Transformer 集成到机器人技术中。
   - 该工具包包括 **Gradio** 界面支持以及与 HuggingFace 数据集的集成。开发者正在寻求提升用户体验，欢迎提供反馈。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/tonyassi/AI-Ecommerce-Fashion">AI E-Commerce Fashion - tonyassi 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://youtu.be/4PgcaIfwLjo">OpenAI 发布 CriticGPT 以纠正 GPT-4 的错误 | 与我一起阅读论文</a>：OpenAI 推出了 CriticGPT，这是一款基于 GPT-4 的新 AI 模型，旨在识别 ChatGPT 生成的代码中的错误，标志着在迈向改进...</li><li><a href="https://github.com/MbodiAI/mbodied-agents">GitHub - mbodiai/embodied-agents: 将最先进的 Transformer 模型无缝集成到机器人技术栈中</a>：将最先进的 Transformer 模型无缝集成到机器人技术栈中 - mbodiai/embodied-agents</li><li><a href="https://huggingface.co/Vi-VLM/Vistral-V-7B">Vi-VLM/Vistral-V-7B · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/hllj/Vistral-V">GitHub - hllj/Vistral-V: Vistral-V: 针对 Vistral 的视觉指令微调 - 越南语大型视觉语言模型。</a>：Vistral-V: 针对 Vistral 的视觉指令微调 - 越南语大型视觉语言模型。 - hllj/Vistral-V</li><li><a href="https://964c3125aaea36d527.gradio.live/">LLaVA</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1257443528699547809)** (5 条消息): 

> `HyperZ⋅Z⋅W Operator Connects Slow-Fast Networks`, `Terminator architecture`, `Terminator code repository`, `Fast training convergence with Terminator architecture`

- **Terminator 架构现已在 GitHub 上发布**: [Terminator architecture](https://github.com/hyperevolnet/Terminator) 代码已在 GitHub 上公开。该架构以连接慢速-快速网络以实现全上下文交互而闻名。
   - 成员们对其快速的训练收敛表示了兴趣；*它仅需 50~100 个训练 epochs 即可获得令人满意的结果*，这仅是**其他架构所需 epochs 数的 1/4~1/2**。
- **Terminator 的高效训练展现出令人印象深刻的结果**: 一位成员提到 **Terminator 架构** 可以在短短 50~100 个训练 epochs 内取得显著成果。这明显少于通常需要 200+ epochs 的其他架构。
   - *快速收敛率*被强调为 Terminator 架构的一个突出特点，使其成为模型训练中潜在的游戏规则改变者。

**提到的链接**: <a href="https://github.com/hyperevolnet/Terminator">GitHub - hyperevolnet/Terminator: The official repository for HyperZ⋅Z⋅W Operator Connects Slow-Fast Networks for Full Context Interaction.</a>: HyperZ⋅Z⋅W Operator 连接慢速-快速网络以实现全上下文交互的官方仓库。 - hyperevolnet/Terminator

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1257578907666743296)** (4 条消息): 

> `Resources for learning advanced CNN topics like ViT and Unets`, `Interest in working on computer vision tasks`, `Prompting etiquette reminders`

- **AI 工程师寻求高级 CNN 资源**: **一位成员**询问了关于 **ViT** 和 **Unets** 等高级 CNN 主题的书籍或资源，包括视频处理。他们正在寻求*进一步学习的建议*。
   - *未提供任何回复或进一步详情*。
- **Discord 中的提示礼仪提醒**: **HuggingMod** 提醒一位用户不要发布得太快。该提醒旨在维持频道内*流畅的对话流程*。
   - *关于此话题没有进一步的讨论*。
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1257666306476343407)** (1 条消息): 

> `embedding numbers`, `specific embedding models`

- **寻求数字嵌入方法**: **Etienne83** 询问了关于嵌入数字的方法或针对数字的特定 Embedding 模型。**未提供详细回复。**
   - *随后没有关于此话题的进一步讨论或意见。*
- **无进一步互动**: 对于关于数字嵌入的问题，没有额外的回复或跟进。**该话题未引发进一步讨论。**
   - *对话在初始查询后未再继续。*
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1257571404015206420)** (2 条消息): 

> `Problem with LLama.cpp causing access violation error`, `Seeking pre-trained model recommendation for fake voice detection`

- **LLama.cpp 访问冲突错误**: 一位成员报告在 Python 脚本中使用 **LLama.cpp** 时遇到 _OSError: access violation reading_ 错误，特别是在模型加载阶段。
   - 该错误发生在 `llama_cpp::llama_load_model_from_file`，导致在地址 `0x0000000000000000` 发生**访问冲突 (access violation)**。
- **寻求用于伪造语音检测的预训练模型**: 一位计算机科学专业的学生正在寻求在**伪造语音检测 (fake voice detection)** 方面表现良好的**预训练模型**推荐。
   - 他们提到正在进行一个研究伪造语音检测模型的项目，并寻求社区关于最佳使用模型的建议。
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1257764254547705877)** (3 messages): 

> `在旧金山 AGI House 举办的仅限 CUDA 的黑客松`, `参与者将在黑客松期间获得 H100 使用权限`

- **由 H100 驱动的黑客松正式发布**: [一场仅限 CUDA 的黑客松](https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf)将于 **7 月 13 日**在旧金山的 AGI House 举行，由 Chris Lattner 主持。每位参与者都将在活动期间获得由 Nebius AI 赞助的 **H100 使用权限**。
   - 强调 **CUDA** 的使用，所有项目必须遵守这一要求。组织者旨在设定一个高标准：*"让我们打破一些基准（baselines）。"*
- **AGI House 黑客松活动详情**: 活动于 **7 月 13 日**星期六**中午 12:00** 开始，**晚上 10:00** 结束。地点详情将分享给名单上的成员。
   - 该活动旨在打造硬核黑客文化，对 CUDA 的使用有严格规定。讨论正在 [Twitter](https://twitter.com) 上进行。

**提到的链接**: <a href="https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf">预约硬核 CUDA 黑客松 | Partiful</a>：*所有的演讲和项目必须使用 CUDA 编写*。每位硬核黑客当天都能获得一块 H100。全部由 Nebius.ai 赞助并提供！让我们打破一些基准。

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

mobicham: https://salykova.github.io/matmul-cpu
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1257527707462402098)** (30 messages🔥): 

> `INTx 性能基准测试`, `torchao 中的量化技术`, `仅 INT8-weight 与 INTx-4 实现对比`, `评估脚本的 batch sizes`, `不同 INTx 实现的 token per second 趋势`

- **INTx 击败 fp16**: [INTx 性能测试](https://github.com/pytorch/executorch/tree/main/examples/models/llama2#quantization)在准确性和速度方面均优于 fp16，其中 int2 和 int4 显示出显著更高的 tokens per second。
   - 成员确认 **int8-weight** 和 **intx-4** 的性能来自 `torchao.quantization.quant_api`，并指出 **int8 及更高版本**可能没那么有用，因为 INT8 的准确度已经足够。
- **评估的 Batch size**: 评估脚本在进行 INTx 性能测试时使用的 batch size 为 **1**。
   - 成员指出，这种设置预计在 batch size 为 1 时效果最好，且[评估工作正在进行中](https://github.com/pytorch/executorch/tree/main/examples/models/llama2#quantization)。
- **Token per second 趋势**: Token per second 速率显示出明显的趋势：**2 的倍数**（如 int2 和 int4）由于其 shard 的使用方式，性能优于其他格式。
   - INT4 是最快的，因为它使用单个 4-bit shard，而 INT7 是最慢的，因为它涉及多个 shard（4-bit, 2-bit 和 1-bit）。

**提到的链接**: <a href="https://github.com/pytorch/executorch/tree/main/examples/models/llama2#quantization">executorch/examples/models/llama2 at main · pytorch/executorch</a>: 适用于移动端、嵌入式和边缘设备的 PyTorch 端侧 AI - pytorch/executorch

  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1257567259933540413)** (5 messages): 

> `基准测试结果脚本请求`, `Token/second 计算代码请求`, `Transformer 更新问题`

- **MobiusML 分享基准测试脚本**: MobiusML 分享了用于基准测试结果的脚本，链接到 [torchao_int4_demo](https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py) 和 [marlin_int4_demo](https://github.com/mobiusml/hqq/blob/master/examples/backends/marlin_int4_demo.py)。
   - *由于 transformers 将 static cache 移到了模型之外，目前仍然无法运行，计划本周修复*。
- **Tokens/Second 计算代码**: MobiusML 解释了如何通过运行 `gen.generate("Write an essay about large language models.", print_tokens=False)` 来计算 tokens/second。
   - 这在最新的 transformers 更新中也失效了，但*已修复并可在 master 分支中使用*。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py">hqq/examples/backends/torchao_int4_demo.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/backends/marlin_int4_demo.py">hqq/examples/backends/marlin_int4_demo.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1257410704596668597)** (204 messages🔥🔥):

> `改进 CUDA MODE 中的辅助函数和核函数`，`关于 GenericVector 及其应用的讨论`，`不同数据集和模型规模下的训练稳定性`，`精简 llm.c 的设置和内存使用优化`，`调查各种配置和设置对性能的影响`，`探索推理优化和平台简化`

- **CUDA 辅助函数：改进功能**：成员们讨论了对辅助函数和核函数（kernel functions）的改进，以增强代码的清晰度和效率。关键改进包括使核函数设置更整洁以及优化内存使用。
   - 讨论内容包括 **GenericVector** 以及针对不同 GPU 任务的更简洁的模板函数：*"ElementCount 应该是可调的；但我认为提供一些有意义的便捷 typedef 是合理的。"*
- **数据集的训练稳定性**：发现在 1.5B 模型上使用 FineWeb 数据集存在训练不稳定性，用户 akakak1337 和 eliebak 确认了这一点。Edu FineWeb 数据集表现更好，引发了进一步分析。
   - 用户 akakak1337 指出，与 **Classic** 变体相比，**FineWeb-EDU** 的不稳定性更少，这表明数据集质量会影响训练稳定性。
- **llm.c 设置与优化**：讨论集中在提高 llm.c 设置过程的效率，移除 miniconda 和 Python 等依赖项。新的设置仅需 1-5 分钟，方便用户快速上手。
   - 关键步骤包括安装必要的库并克隆仓库，设置指令非常精简：*"指令极其简单：安装 cudnn、cudnn-frontend、openmpi，克隆仓库，下载 starter_pack、fineweb 分片，然后运行！"*
- **性能与配置调查**：内存使用和批大小（batch sizes）的对比显示 **llm.c** 的表现显著优于 PyTorch，能高效地在内存中容纳更大的批大小。值得注意的是，在某些配置下会出现批处理问题，导致在较小的批大小时发生崩溃。
   - 用户 akakak1337 在批大小小于 4 时遇到了崩溃，原因是除以零错误：*"整数除以零错误 - 导致运行崩溃"。*
- **推理优化讨论**：关于优化 **inference** 的对话强调了简单性和效率的必要性，避免了单用户设置中诸如量化（quantization）之类的开销。讨论的重点包括精简实现的潜在好处。
   - 讨论了高效的使用案例和可能的未来方向，强调了精简的方法：*"……如果比其他框架简单得多，那么对于单用户推理来说，一些极其简单且表现尚可的东西可能会很有用。"*
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/imbue-ai/cluster-health">GitHub - imbue-ai/cluster-health</a>: 通过在 GitHub 上创建账户，为 imbue-ai/cluster-health 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards">karpathy/fineweb-edu-100B-gpt2-token-shards · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/karpathy/llmc-starter-pack">karpathy/llmc-starter-pack · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/650/">muP (maximum update parametrization) by gordicaleksa · Pull Request #650 · karpathy/llm.c</a>: 主要变更：修改随机初始化；将 Attention 分数按 1/d 而非 1/sqrt(d) 进行缩放，并添加 attn_mult；在映射到 Logits 之前按 1/width_mult 缩放 Activation；更新 Learning Rate 及...</li><li><a href="https://huggingface.co/mlx-community">mlx-community (MLX Community)</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/commit/a876282eb845f89aef70c780033ee150aba044b0">Merge pull request #653 from ademeure/cublaslt_refactor · karpathy/llm.c@a876282</a>: 使用 cuBLASLt + GELU Fusion 重构 Matmul</li><li><a href="https://github.com/karpathy/llm.c/pull/641/">Add check versions of functions by gordicaleksa · Pull Request #641 · karpathy/llm.c</a>: 添加 Socket 关闭检查函数 - 与代码库其余部分保持一致。</li><li><a href="https://github.com/karpathy/llm.c/pull/650">muP (maximum update parametrization) by gordicaleksa · Pull Request #650 · karpathy/llm.c</a>: 主要变更：修改随机初始化；将 Attention 分数按 1/d 而非 1/sqrt(d) 进行缩放，并添加 attn_mult；在映射到 Logits 之前按 1/width_mult 缩放 Activation；更新 Learning Rate 及...</li><li><a href="https://github.com/karpathy/llm.c/pull/666">Improved style &amp; comments for gpt2_forward() by ademeure · Pull Request #666 · karpathy/llm.c</a>: 显然非常主观，但对我来说这是一个巨大的改进！我关于 Residual+LN Fusion 以及它如何对应 Pre-LN 相关论文的注释是一个很好的例子...</li><li><a href="https://github.com/karpathy/llm.c/pull/657">Remove per-layer attproj and fcproj activation tensors by ademeure · Pull Request #657 · karpathy/llm.c</a>: 我不确定我们是怎么漏掉这个的，但我们在 Backward Pass 中实际上根本不需要这些 Tensor！可能在实现 Residual/LayerNorm/Recompute 时情况并非如此...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1257532460900679721)** (1 messages): 

> `AMD MI300X 与 Nvidia H100 SXM 在 Mixtral 8x7B 推理上的性能对比`, `Nvidia CUDA 相对于 AMD ROCm 在 AI 工作负载中的优势`, `开发者在 AI 生产环境中对 Nvidia GPU 的偏好`

- **AMD MI300X 挑战 Nvidia H100 SXM**: [RunPod 博客](https://blog.runpod.io/amd-mi300x-vs-nvidia-h100-sxm-performance-comparison-on-mixtral-8x7b-inference/) 发布的基准测试显示，在运行 **Mistral** 的 **Mixtral 8x7B** 推理时，**AMD MI300X** 在小批量和大批量下的表现均优于 **Nvidia H100 SXM**。基准测试结果突显了 MI300X 卓越的硬件规格。
   - 然而，*大多数开发者在实际生产负载中并不使用 AMD 显卡*，因为在为机器学习应用编写软件方面，**Nvidia** 的 **CUDA** 远超 **AMD** 的 **ROCm**。
- **Nvidia CUDA 主导 AI 软件开发**: 尽管 MI300X 拥有更好的原始规格，但由于 **CUDA** 在编写机器学习应用软件的易用性方面远超 **ROCm**，开发者在生产负载中更青睐 **Nvidia GPU**。尽管 AMD 在硬件上有所进步，但卓越的软件生态系统使 Nvidia 保持主导地位。
   - 该帖子强调，**Nvidia 在 AI 训练和推理方面的历史主导地位**是**几乎所有生产环境中的 AI 负载都在其显卡上运行**的主要原因。

**提到的链接**: <a href="https://blog.runpod.io/amd-mi300x-vs-nvidia-h100-sxm-performance-comparison-on-mixtral-8x7b-inference/">AMD MI300X vs. Nvidia H100 SXM: Performance Comparison on Mixtral 8x7B Inference</a>: 不可否认 Nvidia 在 AI 训练和推理方面的历史主导地位。几乎所有生产环境中的 AI 工作负载都在其显卡上运行。然而，目前出现了一些乐观的情绪...

  

---


### **CUDA MODE ▷ #[sparsity](https://discord.com/channels/1189498204333543425/1247663759434977453/)** (1 messages): 

iss_llm: 非常感谢 <@1213148470664495114> ！
  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1257449278998188063)** (2 条消息): 

> `Android 应用上线 Voice-to-voice 体验`, `Pro Search 更新带来更深层的研究能力`

- **Voice-to-voice 体验在 Android 端首发**：最新的 Android 应用版本引入了 **voice-to-voice 体验**，包含两种模式：*Hands-free* 和 *Push-to-talk*。鼓励用户在 [指定频道](https://discord.com/channels/1112135181567000697) 提供反馈。
   - Hands-free 模式在屏幕打开时立即开始监听，而 Push-to-talk 模式通过按住麦克风按钮进行操作。
- **Pro Search 升级增强了复杂查询处理能力**：更新后的 **Pro Search** 现在通过涉及 **multi-step reasoning**、**Wolfram|Alpha** 和 **code execution** 的更深层研究来处理更复杂的查询。升级后的工具旨在辅助 *深度研究、数学问题解决和代码调试*。
   - 改进后的搜索提供了更具参考价值的答案，支持复杂且详细的查询。更多信息可以在 [这里](https://pplx.ai/FVyJaIH) 找到。

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1257413650935844925)** (211 条消息🔥🔥): 

> `Perplexity AI 搜索引擎问题与更新`, `关于 Perplexity AI 上 Claude 的 token 限制讨论`, `使用 Perplexity 生成图表或可视化的方法`, `Perplexity AI 的推荐链接和促销代码`, `Perplexity AI 对印度新闻网站的偏好`, `AI 模型的使用与搜索结果准确性`, `关于 Perplexity AI 移动端应用功能的常见查询`

- **Perplexity AI 搜索引擎问题与更新**：用户对 Perplexity AI 的搜索功能表示不满，包括在搜索过程中难以更改来源，以及即使在学术模式下也会使用非学术来源（[示例链接](https://www.perplexity.ai/search/tsutomu-yamaguchi-bydQvm2fRq.YQCMnTz0C8w)）。他们强调了 Perplexity 与 Morphic.sh 等其他平台在来源可靠性方面的差异。
   - 讨论中提到了印度新闻网站和 LinkedIn 在搜索结果中频繁出现，一些用户认为这存在偏见且令人困扰。成员们建议搜索引擎可能会利用用户位置或个人资料设置来确定搜索结果的优先级。
- **Claude 在 Perplexity AI 上的 token 限制**：用户讨论了 Perplexity AI 上的 Claude 并不提供完整的 **200k tokens**，而是被限制在约 **32k tokens**。这一限制打消了一些用户购买订阅的念头。
   - 对话凸显了对 token 限制的不满，认为 Perplexity AI 更多是一个搜索引擎，而非真正的助手。
- **使用 Perplexity 生成图表/可视化**：成员们询问如何在 Perplexity 中生成**图表和可视化内容**，回复澄清它可以为 Google Colab 等平台生成代码，但缺乏内置支持（[示例链接](https://www.youtube.com/watch?v=ht3XV_nbduQ)）。社区建议使用外部工具或集成（如 Mermaid）来制作图表。
   - 讨论还包括了使用代码块和用户脚本自动渲染图表的指南，并指出 AIlin 或 Complexity 扩展可能提供集成支持。
- **Perplexity AI 的推荐链接和促销代码**：用户频繁分享**推荐链接**和**促销代码**，以帮助他人获得 Perplexity AI 订阅折扣（[示例链接](https://perplexity.ai/pro?referral_code=LL3DA3XH)）。一个普遍的情绪是对试用版的期待和需求。
   - 一些人对目前没有**试用版**表示遗憾，原因是过去曾被黑客滥用，导致该服务受限。
- **Perplexity AI 移动端应用功能与崩溃问题**：关于移动端应用是否提供 Wolfram Alpha 和代码生成等**新功能**的咨询非常普遍。一些用户报告在 iPhone 上进行 Pro 搜索时遇到崩溃。
   - 讨论还涉及对移动端与 Web 端 AI 模型性能的反馈和比较，部分用户表达了对写作模式和搜索方式的偏好。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.datacamp.com/">在线学习 R, Python 和数据科学</a>：在浏览器中舒适地学习数据科学和 AI，按照自己的节奏通过 DataCamp 关于 R, Python、统计学等的视频教程和编程挑战进行学习。</li><li><a href="https://www.youtube.com/watch?v=0IZSHWW26dE">在 Perplexity AI 中复制答案时移除文中引用</a>：在本视频中，我将向您展示一个简单的技巧，以便在 Perplexity.AI 中复制答案时移除文中引用和参考文献编号。</li><li><a href="https://youtu.be/ht3XV_nbduQ">什么是 Perplexity Copilot？</a>：你需要答案，而且现在就要。但有时，你需要的不仅仅是快速的 Google 搜索。进入 Perplexity Copilot，你的新搜索助手...</li><li><a href="https://gitlab.com/monnef/ailin">monnef / AIlin · GitLab</a>：AIlin 是一个将 Perplexity.ai 等 AI 服务与本地计算机连接的工具。</li><li><a href="https://www.morphic.sh/share/W1Kd9iO">Morphic</a>：一个完全开源、由 AI 驱动且具备生成式 UI 的回答引擎。
</li>
</ul>

</div>

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1257516674278490213)** (9 messages🔥): 

> `Perplexity AI 的最佳用例`, `魔方 50 周年`, `欧盟起诉 Meta`, `电动飞行器发展`, `AI 视频进展`, `使用 Perplexity 创业`, `眼泪的情感影响`, `Microsoft Connect Test 搜索`

- **魔方庆祝 50 周年**：**魔方 (Rubik's Cube)** 迎来 50 周年，标志着这款经典益智玩具风靡全球五十载。[在此观看庆祝视频](https://www.youtube.com/embed/UrJp4OuxFGM)。
   - 视频还涵盖了其他重要新闻，包括欧盟起诉 Meta 以及电动飞行器的进展。
- **使用 Perplexity 编写商业计划书**：了解如何使用 [Perplexity AI 平台](https://www.perplexity.ai/search/help-me-create-a-business-plan-Xjj7Hy2XROuh7vxXQPqq2g) 创建一份详尽的商业计划书。本指南提供了制定有效商业策略的详细步骤。
   - 深入讨论了适用于初创公司的 **精益画布 (lean canvas)** 技术，强调了其实用性。在此观看详细**指南**：[精益画布指南](https://www.perplexity.ai/search/when-should-you-do-a-lean-canv-z_lDH7CJStuuX.MpyRGNMA)。
- **探索 Perplexity AI 的最佳用例**：通过此搜索链接发现 **Perplexity AI** 最有效的用例：[最佳用例](https://www.perplexity.ai/search/what-are-the-best-use-cases-fo-it2HP6tcRIyxEtVf55bXmA)。它重点介绍了最大化 AI 功能的最佳应用场景。
   - 讨论涉及多个行业，包括医疗保健和金融，Perplexity AI 在这些领域已展现出良好的应用前景。

**提及的链接**：<a href="https://www.youtube.com/embed/UrJp4OuxFGM">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1257433231389753585)** (8 messages🔥): 

> `API 设置加载问题`, `Perplexity 引用请求跟进`, `Sonnet 3.5 与 Perplexity API 使用`, `API 模型可用性`, `通过 API 使用搜索引擎`

- **Chrome 中的 API 设置加载问题**：一位用户报告在 Chrome 上访问 API 设置时出现问题，页面无限期加载。切换到 Safari 解决了该问题，并建议清除缓存。
   - *“似乎运行正常。你试过其他浏览器吗？”* 另一位用户回应道。通过尝试不同的浏览器并清除缓存，问题得到了解决。
- **Perplexity 引用请求跟进**：一位用户询问了两周前提交的 Perplexity 引用 (citations) 请求的状态。另一位用户表达了同样的担忧，并提到已向 api@perplexity.ai 发送了跟进邮件。
   - 用户们正焦急地等待 Beta API 访问权限的回复，其中一人表示：*“还没收到任何消息。我这周早些时候给 api@perplexity.ai 发了邮件跟进。”*
- **Sonnet 3.5 无法通过 Perplexity API 获取**：一位用户询问关于在 Perplexity API 中使用 Sonnet 3.5 的事宜。官方澄清 Sonnet 无法通过 API 使用，并提供了可用模型列表。
   - 共享了可用模型的列表和详细信息，并提到尽可能匹配 Hugging Face 的实现。强调了 *“Sonnet 不通过我们的 API 提供”*。
- **对通过 API 使用搜索引擎的兴趣**：另一位用户在得知无法使用 Sonnet 后，表达了通过 API 使用搜索引擎的兴趣。
   - *“嗯，好吧，我很想通过 API 使用搜索引擎，”* 该用户回复道，表明尽管存在最初的限制，他们仍保持浓厚兴趣。

**提及的链接**：<a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>：未找到描述

  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1257411314385424414)** (130 messages🔥🔥): 

> `扩展合成数据生成`, `Unsloth 在多 GPU 设置下的问题`, `Unsloth 与 Ollama 的兼容性`, `Unsloth 的新功能与更新`, `在 Unsloth 中使用 RAG (Retrieval-Augmented Generation)`

- **使用 10 亿个 Persona 扩展合成数据**：[Aran Komatsuzaki](https://x.com/arankomatsuzaki/status/1807593343007818065) 分享了关于 **Persona Hub** 的信息，这是一个包含 **10 亿** 个多样化 Persona 的集合，使 MATH 基准测试的成绩从 **49.6 大幅提升至 64.9**。它在 [GitHub 仓库](https://github.com/tencent-ailab/persona-hub) 中作为概念验证（proof of concept）呈现，并在 [arXiv 论文](https://arxiv.org/abs/2406.20094) 中进行了进一步讨论。
   - 一位成员评论说，该项目与 augmentoolkit 类似，但 Persona Hub 使用的 Persona 数量远多于前者。另一位成员表示打算阅读该论文以获取更多见解。
- **Unsloth 在多 GPU 设置下的问题**：多名用户在多 GPU 设置下运行 **Unsloth** 时遇到困难，遇到了各种运行时错误，例如 `subprocess.CalledProcessError`。一个特别被注意到的错误与在多 GPU 环境下运行 Unsloth 下载和量化过程有关。
   - Unsloth 目前官方不支持多 GPU 设置，建议用户尝试变通方案，或联系团队获取 Beta 版本。一种变通方案涉及使用环境变量，但未能解决该问题。
- **Unsloth 与 Ollama 的集成**：一位用户询问了将 **Unsloth** 与 **Ollama** 集成的问题，另一位用户确认它们是独立的，但可以通过一些设置协同工作。**Unsloth** 可以转换训练好的模型，以便与 Ollama 无缝使用。
   - 对话强调了由于模型构建方式的原因，自动化兼容性流水线（pipeline）的复杂性。目前正在准备一份将 Unsloth 模型集成到 Ollama 的指南。
- **Unsloth 的功能与最新更新**：Unsloth 因其在 Fine-tuning 方面的高效性以及新功能（如具有非扩展上下文大小的 **Gemma 27B** 和 **Phi-3 Mini** 的最新更新）而受到称赞。用户报告称，使用 Unsloth 为特定任务微调模型，达到了惊人的准确度。
   - 讨论包括了 Unsloth 高效改进大规模数据任务的潜力。特别是，一位用户在放射学相关的数据提取中实现了超过 **95% 的准确率**，展示了 Unsloth 实现超越最先进（state-of-the-art）结果的能力。
- **将 RAG 与 Unsloth 集成**：微软的 **GraphRAG** 系统被介绍为一种可能与 **Unsloth** 集成的工具，以增强检索增强生成（retrieval-augmented generation）能力。这个想法受到了好评，人们对可能的集成表现出兴趣。
   - 对话以对 **Unsloth** 如何从集成 RAG 中受益的好奇结束。这种集成将允许用户在进行检索增强生成的同时进行 Fine-tuning。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/trl/en/sft_trainer#advanced_usage">Supervised Fine-tuning Trainer</a>: 未找到描述</li><li><a href="https://x.com/arankomatsuzaki/status/1807593343007818065?s=46">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>: Scaling Synthetic Data Creation with 1,000,000,000 Personas - 展示了从网络数据中自动策划的 10 亿个多样化角色集合 - 在 MATH 数据集上取得巨大提升：49.6 -> 64.9 代码库: https://g...</li><li><a href="https://arxiv.org/abs/2406.20094">Scaling Synthetic Data Creation with 1,000,000,000 Personas</a>: 我们提出了一种新颖的角色驱动数据合成方法，利用大语言模型 (LLM) 中的各种视角来创建多样化的合成数据。为了充分利用这种方法...</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-to-gguf">主页</a>: 微调 Llama 3, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/microsoft/graphrag">GitHub - microsoft/graphrag: 一个模块化的基于图的检索增强生成 (RAG) 系统</a>: 一个模块化的基于图的检索增强生成 (RAG) 系统 - microsoft/graphrag</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: C/C++ 中的 LLM 推理</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户来为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6844">jubruckne 提交的自定义量化方案 · Pull Request #6844 · ggerganov/llama.cpp</a>: 这尚未准备好合并，但我希望听听您的意见，看您是否有兴趣将其包含在内。如果是这样，我可以对其进行清理和改进。这个想法是允许创建一个自定义的...</li><li><a href="https://huggingface.co/datasets/proj-persona/PersonaHub">proj-persona/PersonaHub · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/933d9fe2cb2459f949ee2250e90a5b610d277eab/unsloth/tokenizer_utils.py#L962">unsloth/unsloth/tokenizer_utils.py (版本 933d9fe) · unslothai/unsloth</a>: 微调 Llama 3, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/933d9fe2cb2459f949ee2250e90a5b610d277eab/unsloth/models/llama.py#L1199">unsloth/unsloth/models/llama.py (版本 933d9fe) · unslothai/unsloth</a>: 微调 Llama 3, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1257622203583828038)** (6 条消息): 

> `微软更新了 Phi-3 Mini`, `对新训练配方的猜想`, `Anthropic 的转向 (steering) 相关内容和 Sonnet 3.5`, `OpenAI 开发了 CriticGPT`

- **微软更新 Phi-3 Mini，提升性能**: 微软发布了 **Phi-3 Mini** 模型的更新，专门增强了 4K 和 128K 上下文的模型检查点。新模型的链接可以在 [Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (4K) 和 [Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) (128K) 上找到。
   - 一位用户推测，最近巨大的分数提升可能是由于某种未公开的训练方法，类似于 **Anthropic** 和 **OpenAI** 使用的技术（例如 steering 和 CriticGPT）。*“分数基本上是跳跃式增长”* 总结了用户的兴奋之情。
- **社区对意外的模型升级做出反应**: LocalLLaMA 子版块的用户对 **Phi-3 Mini** 的更新表示惊讶和热情。原始帖子链接至 [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1dtgylv/microsoft_updated_phi3_mini/)，描述了这些更新的性质。
   - 一位用户幽默地将 **OpenAI** 称为 *“ClosedAI 的 CriticGPT”*，突显了社区的参与度和活跃度。

**提到的链接**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1dtgylv/microsoft_updated_phi3_mini/">Reddit - 深入了解一切</a>: 未找到描述

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1257418785351270511)** (44 条消息🔥): 

> `Unsloth 对 SPPO 的支持`, `xformers 与 PyTorch 的常见错误`, `使用 Unsloth 微调模型并部署到 Ollama`, `在 AMD GPU 上运行 Unsloth 模型`, `处理模型训练过程中的显存溢出 (OOM) 错误`

- **SPPO 可在 Unsloth 上运行**：*theyruinedelise* 确认如果 SPPO 能在 TRL 上运行，那么它也能在 Unsloth 上运行。
   - *theyruinedelise* 重申 Unsloth 通过 TRL 实现了对 SPPO 的兼容，使其成为一个可用的集成。
- **PyTorch 2.2 的 Xformers 错误**：一位用户在使用 Unsloth 期间遇到了与 xformers 和 PyTorch 2.2 相关的 IndexError。
   - *unclemusclez* 提到正在排查此问题，并指出可能与 PyTorch 的 nightly 版本不兼容。
- **在 Ollama 上部署微调模型**：*ozzy.khal* 在本地 CPU 上运行微调模型时遇到问题，出现了 Unsloth 框架的属性错误（attribute error）。
   - *edd0302* 建议使用可在 CPU 上运行的 Ollama，并提供了一个 [Colab notebook 链接](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing) 以方便操作。
- **在 AMD GPU 上运行 Unsloth**：*unclemusclez* 分享了在 AMD GPU 上运行 Unsloth 的困难，并提到需要从源码编译。
   - *theyruinedelise* 承认了这一挑战，将其归因于 PyTorch nightly 版本可能存在的不兼容性和不稳定性。
- **处理 T4 GPU 上的显存溢出 (OOM) 错误**：*tom250* 提出了在 T4 GPU 上微调 Qwen2-0.5B 时遇到 OOM 错误的问题。
   - 错误提示内存不足，*tom250* 引用了具体的 PyTorch 内存统计数据并推荐了环境变量设置。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16">主页</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">主页</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)">CUDA 语义 — PyTorch 2.3 文档</a>：未找到描述</li><li><a href="https://github.com/ROCm">AMD ROCm™ 软件</a>：AMD ROCm™ 软件有 263 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://huggingface.co/akashcsd/Llama3_project_management_assistant/tree/main">akashcsd/Llama3_project_management_assistant at main</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1257680036757110855)** (2 条消息): 

> `llmcord.py`, `Discord LLM 前端`, `jakobdylanc 的 GitHub 项目`, `社区对 llmcord.py 的反馈`

- **使用 llmcord.py 将 Discord 作为 LLM 前端**：**jakobdylanc** 介绍了一个名为 [**llmcord.py**](https://github.com/jakobdylanc/discord-llm-chatbot) 的新脚本，可将 Discord 转换为 LLM 的前端。该工具旨在简单且健壮，允许用户直接在 Discord 上与 LLM 交互。
   - *一位社区成员称赞该项目说*，“Nice great job!”，这表明了社区的积极反响和兴趣。
- **社区对 llmcord.py 的赞赏**：**theyruinedelise** 通过评价 “Nice great job!” 表达了对该项目的欣赏。这显示了社区对 **jakobdylanc** 提议的积极认可。
   - 此类反馈突显了围绕 Discord 和 LLM 集成的社区协作和支持性质。

**提到的链接**：<a href="https://github.com/jakobdylanc/discord-llm-chatbot">GitHub - jakobdylanc/discord-llm-chatbot: llmcord.py • 与你的朋友一起和 LLM 聊天！</a>：llmcord.py • 与你的朋友一起和 LLM 聊天！通过在 GitHub 上创建账号为 jakobdylanc/discord-llm-chatbot 的开发做出贡献。

  

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1257842275682095285)** (2 条消息): 

> `协作教程与 Notebook 发布`, `数据集支持增强`, `微调指南已发布`, `征求 Colab Notebook 反馈`

- **协作教程发布**：一位成员分享了一个新的 [Colab notebook](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing)，支持多个数据集和各种 prompt 格式。该 notebook 尚未经过彻底测试，正在寻求反馈。
   - 主要功能包括 **为 Kaggle 用户提供 Exllama2 量化支持**、LoRA rank 自动缩放，以及包含额外技巧的通用微调指南。他们特别征求了讨论中被标记人员的意见。
- **征求 Colab Notebook 反馈**：分享该 Colab notebook 的成员请求社区其他成员提供反馈，特别是标记了两位成员。他们寻求改进 notebook 功能和修复潜在问题的见解与建议。
   - 目前尚未记录到即时回复或见解，突显了对详细审查和评论的期待。部署重点在于增强 **数据集支持** 和微调方法论。

**提及的链接**: <a href="https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing">Google Colab</a>: 未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1257412077090377851)** (11 条消息🔥): 

> `通过 10 亿个 Persona 扩展合成数据生成`, `Persona 驱动的数据合成方法论`, `Google 发布 Gemma 2`, `Google DeepMind 的广义知识蒸馏 (GKD)`

- **通过 10 亿个 Persona 扩展合成数据生成**：一条推文介绍了一个从网络数据中自动策划的 **10 亿个多样化 Persona** 集合，展示了在 MATH 上的巨大提升：**49.6 -> 64.9**。相关链接包括 [GitHub 仓库](https://github.com/tencent-ailab/persona-hub) 和 [Arxiv 论文](https://arxiv.org/abs/2406.20094)。
   - 成员们讨论了数据相对于代码的重要性，其中一位指出 *'数据比代码重要得多'*，而其他人则提到了复制的简易性。
- **Gemma 2：Google 最新的开源 LLM 发布**：Google 发布了 [Gemma 2](https://huggingface.co/blog/gemma2#knowledge-distillation)，在 Hub 上提供了 4 个开源权重模型。这些模型已与 Hugging Face Transformers 以及 Google Cloud & Inference Endpoints 集成。
   - 成员们对与 Google 的合作表示兴奋，以确保在 Hugging Face 生态系统中实现最佳集成。
- **Google DeepMind 的广义知识蒸馏 (GKD)**：Google DeepMind 关于 [广义知识蒸馏 (GKD)](https://arxiv.org/abs/2306.13649) 的论文通过在学生模型自身生成的输出序列上进行训练，解决了自回归序列模型中的“分布不匹配”问题。这种方法提供了在学生和教师模型之间采用替代损失函数的灵活性。
   - 一位成员强调了 GKD 如何修复训练期间学生模型的输出与推理期间不同的问题。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2306.13649">On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes</a>: 知识蒸馏 (KD) 广泛用于压缩教师模型，通过训练较小的学生模型来降低其推理成本和内存占用。然而，目前的自回归序列 KD 方法...</li><li><a href="https://x.com/arankomatsuzaki/status/1807593343007818065?s=46">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>: 通过 10 亿个 Persona 扩展合成数据生成 - 展示了从网络数据自动策划的 1B 多样化 Persona 集合 - MATH 上的巨大提升: 49.6 -&gt; 64.9 仓库: https://g...</li><li><a href="https://arxiv.org/abs/2406.20094">Scaling Synthetic Data Creation with 1,000,000,000 Personas</a>: 我们提出了一种新型的 Persona 驱动数据合成方法论，利用大型语言模型 (LLM) 中的各种视角来创建多样化的合成数据。为了充分利用这一方法论...</li><li><a href="https://huggingface.co/blog/gemma2#knowledge-distillation">欢迎 Gemma 2 - Google 的新开源 LLM</a>: 未找到描述
</li>
</ul>

</div>

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1257412154680676372)** (155 条消息🔥🔥): 

> `在有限的 VRAM 下运行 Stable Diffusion`, `Stable Diffusion 3 的问题`, `为特定风格训练 LoRA (Low-Rank Adaptation)`, `适用于不同图像风格的 Stable Diffusion`, `反 AI 艺术软件的潜力`

- **在有限的 VRAM 下运行 Stable Diffusion**：一位成员询问如何在 12GB VRAM 的本地环境下运行 Stable Diffusion，并提到 everydream2 训练器需要 16GB VRAM 的挑战。另一位用户分享了在 8GB 系统上生成一个微型 checkpoint 耗时 4 小时的经历。
   - 讨论强调了运行 Stable Diffusion 的硬件要求，成员们建议尽管有 VRAM 限制，仍可尝试不同的模型或设置。
- **Stable Diffusion 3 的问题**：成员们讨论了 Stable Diffusion 3 (SD3) 的缺点，指出它缺失了某些微调方面的内容且功能不完整。一位用户强调缺乏在上面高效训练 LoRA 的简便方法。
   - 一位成员对图像质量的限制表示沮丧，特别是在躺姿或坐姿等更复杂的场景中，并提到了对更高级功能的优先需求。
- **为特定风格训练 LoRA**：用户分享了为特定需求训练 LoRA 模型的愿望，例如包含 3D 分形或游戏截图。一位成员指出 Stable Diffusion 3 并不容易支持 LoRA 训练。
   - 尽管存在障碍，社区成员仍就使用各种工具和资源实现专业化训练提供了建议，并强调了频繁进行实验的必要性。
- **适用于不同图像风格的 Stable Diffusion**：几位用户交流了 Stable Diffusion 在多种图像风格上的优势，包括线稿、卡通、动漫、恐怖、肖像、自然摄影等。一些用户详细说明了特定的用例，例如结合某些风格来增强模型能力。
   - *一位用户幽默地提到了尝试不同 prompt 的不确定性以及结果的多样性*，突显了该工具在创意使用方面的多功能性。
- **反 AI 艺术软件的潜力**：讨论围绕着创建软件以保护艺术家作品不被用于 AI 生成模型的想法展开，类似于 Glaze 和 Nightshade。共识是，由于存在绕过方法，此类措施目前是无效的。
   - 成员们辩论了在艺术作品中进行反 AI 编码的可行性，考虑了实际限制和潜在的对策，同时一些人指出了道德影响以及数字内容可复制性的现实。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/AuraDiffusion/16ch-vae">AuraDiffusion/16ch-vae · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/TheLastBen/fast-stable-diffusion">GitHub - TheLastBen/fast-stable-diffusion: fast-stable-diffusion + DreamBooth</a>: fast-stable-diffusion + DreamBooth。通过在 GitHub 上创建账户来为 TheLastBen/fast-stable-diffusion 的开发做出贡献。</li><li><a href="https://tenor.com/view/michael-jackson-eating-popcorn-enjoy-i-like-nom-nom-gif-11040065238845078056">Michael Jackson Eating Popcorn GIF - Michael Jackson Eating Popcorn Enjoy - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://civitai.com/models/257749/pony-diffusion-v6-xl">Pony Diffusion V6 XL - V6 (start with this one) | Stable Diffusion Checkpoint | Civitai</a>: Pony Diffusion V6 是一款多功能的 SDXL 微调模型，能够生成各种兽人、野兽或类人生物的精美 SFW 和 NSFW 视觉效果...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1257412795134247054)** (118 messages🔥🔥): 

> `AI 硬件（H100 和 H200 GPU，Google TPUs）的高成本与性能`, `对 AI 图像生成工具（Luma Dream Machine, Runway Gen-3）的失望`, `使用 TPU V3-8 和 Paperspace 进行 AI 训练的潜力`, `多模态模型及其在特定领域的局限性`, `像 LangChain 这样的编排器以及 AI 中的上下文限制`

- **AI 硬件成本引发争议**：讨论围绕购买 **8 个 H100 GPU** 的高昂成本展开，据称成本超过 **50 万美元**，且在没有企业账户的情况下很难直接从 NVIDIA 购买。
   - 成员们探索了 **Google TPUs** 等替代方案以及 **Paperspace** 等租赁服务以降低高昂成本，并指出了显著的价格差异和不同的性能水平。
- **用户对 AI 图像生成器表示不满**：用户对 **Luma Dream Machine** 和 **Runway Gen-3** 等 AI 图像生成工具表示失望，称其为**过度炒作**且**价格过高**。
   - 一位用户指出，在 Runway Gen-3 上 **15 美元**仅能生成 **6-7 次**，且性能并不比 Luma Dream Machine 好多少，导致用户很快失去兴趣。
- **探索 TPU 和基于云的训练**：对话强调了使用 **Google TPU V3-8** 作为训练大型模型的 H100 GPU 可行替代方案的潜力，尽管速度较慢。
   - 推荐使用 **Paperspace** 等服务按小时租用强大的 GPU，**H100 的价格为每小时 2.24 美元**，提供了一个具有成本效益的解决方案。
- **关于多模态模型效能的辩论**：讨论承认虽然**多模态/通用模型**提供了令人印象深刻的多功能性，但它们目前面临技术限制。
   - 针对特定任务（如编程）进行微调的特定模型被认为更有效，特别是对于 **Python** 等语言，尽管对于更复杂或多样化的任务仍然存在挑战。
- **关于编排器和 AI 上下文限制的聊天**：有人提出了关于 OpenAI API 中 AI 上下文限制未来是否可能超过当前 **4k tokens** 的问题，强调了改进的需求。
   - 成员们还讨论了像 **LangChain** 这样的编排器，并分享了在 Prompt Engineering 系统中集成或替换 **GPT-4** 和 **LLama** 等模型的经验。
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1257718828020797511)** (3 messages): 

> `确保 GPT 完成任务中的所有步骤`, `GPT-4o 上 RAG 意图检查的 Prompt 设计`, `GPT 在多步骤 Prompt 中跳过步骤的问题`

- **Customer-GPT 跳过步骤**：一位成员询问如何确保其 **Customer-GPT** 在完成任务后进行自我复查，并指出尽管 Prompt 布局逻辑清晰，它仍会跳过步骤。
   - *Solbus* 建议使用“此步骤在彼步骤之前”的方法来构建指令，以帮助模型通过在达到上下文限制时从中断处继续，从而避免跳过指令。
- **RAG 的意图检查 Prompt**：另一位成员寻求关于为 **GPT-4o 上的 RAG** 创建意图检查 Prompt 的建议，将输入分为三类：模糊、基于历史或需要新的 VectorDB 搜索。
   - 他们报告了连贯性问题，机器人会混淆问题与历史并启动不必要的搜索，并询问他人是否有可靠的 Prompt 设计。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1257718828020797511)** (3 messages): 

> `Customer-GPT 在任务完成后进行自我检查`, `GPT-4 上 RAG 的意图检查 Prompt`, `构建 GPT 指令以避免跳过步骤`

- **Customer-GPT：确保彻底完成任务**：*allenamenwarenschonvergeben* 寻求关于让 **Customer-GPT** 在任务完成后重新检查步骤的建议。问题在于尽管 Prompt 布局合理，GPT 仍会跳过步骤，他们更倾向于自动检查而非手动 Prompt。
   - *solbus* 建议按“此步骤在彼步骤之前”的顺序构建指令，以确保模型按顺序执行每一步，通过使步骤依赖于前一步骤来解决跳过指令的问题。
- **为 GPT-4 上 RAG 的意图检查制作可靠的 Prompt**：*tvl4121* 在创建 Prompt 以让 GPT-4 将用户输入分类为三类（模糊、可由对话历史回答、或需要 VectorDB 搜索）时面临挑战。他们报告了不一致的结果，特别是在上下文切换的问题上。
   - 目前没有提供直接的解决方案，但该咨询突显了在 RAG 应用中处理上下文和意图检查时对更可靠方法的需求。
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1257420272903127191)** (20 条消息🔥): 

> `vLLM Deployment`, `Probing Learnt Representations in Diffusion Models`, `GPT-4 Parameter Count and Mixture of Experts`

- **vLLM 解决 HF 问题**：一位用户指出了 HF 的一个已知问题，并建议改用 **vLLM**，同时分享了一个 [wiki 链接](https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm) 用于将模型保存为 16 Bit。另一位用户在遵循建议后确认使用 vLLM 取得了成功。
   - 社区认可了 vLLM 在高效模型部署方面的效用，用户对提供的指导表示感谢。
- **Diffusion Models 与学习到的表征**：一位用户请求推荐关于探究 Diffusion Models（尤其是语言方面）中学习到的表征的论文，并分享了一篇关于 Latent Diffusion Models (LDMs) 的[论文链接](https://arxiv.org/abs/2306.05720)。讨论强调了 LDMs 的可解释性和内部激活。
   - 论文指出，LDMs 在去噪过程早期就编码了 3D 深度和显著对象区分的线性表征，这有助于图像合成和高级编辑。
- **未来模型参数推测**：一位用户询问 OpenAI 是否会披露未来模型的参数，以及为什么 GPT-4 的参数不可用。有传言称 GPT-4 拥有 **1.7T 参数**，归功于 Mixture of Experts。
   - 用户对这一数字的真实性进行了辩论，引用了据称 Nvidia 在 GTC 期间的确认，与 220B 乘以 **8 个模型大小**相吻合。其他人仍持怀疑态度，质疑 Nvidia 的 NDA 义务。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2306.05720">Beyond Surface Statistics: Scene Representations in a Latent Diffusion Model</a>: Latent Diffusion Models (LDMs) 展现出令人印象深刻的生成逼真图像的能力，但这些模型的内部运行机制仍然神秘。即使纯粹在没有显式...的图像上训练</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 使用 Unsloth 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3, Mistral, Phi &amp; Gemma LLMs - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1257436742257086465)** (73 条消息🔥🔥): 

> `Tokenization in LLMs`, `Factorization Curse in Language Models`, `Covert Malicious Fine-Tuning`, `Memory Efficiency in PEFT Methods`, `Estimating Networking Hardware Costs for ML Clusters`

- **LLM 中的 Tokenization 挑战**：一篇新[论文](https://arxiv.org/abs/2406.20086)探讨了 LLM 中的单个 token 往往缺乏语义含义，例如 **Llama-2-7b** 将 'northeastern' 拆分为无关的 token。该研究调查了 token 表示中的“擦除（erasure）”效应。
   - 社区对该研究表示热烈欢迎，特别是关于将任意 token 组转换为高级表示的见解，并指出其对理解 Tokenization 的贡献。
- **分解诅咒（Factorization Curse）阻碍信息检索**：最近备受关注的一篇[论文](https://arxiv.org/abs/2406.05183)将“反转诅咒（reversal curse）”重新定义为“分解诅咒”，由于 LLM 无法学习不同分解下的联合分布，从而影响了信息检索。它引入了 WikiReversal 来模拟知识密集型任务。
   - 成员们注意到该论文的启示，即使用损坏或改写的数据进行训练可以带来更好的泛化能力，并强调“UL2 风格的目标”可能是关键。
- **隐蔽恶意微调威胁**：一项[研究](https://arxiv.org/abs/2406.20053)提出了“隐蔽恶意微调”，展示了此类方法如何在规避检测的同时损害模型安全性，在有害指令方面的成功率达到 99%。研究结果引发了对保护黑盒微调接口安全性的担忧。
   - 社区成员推测了其影响，并将其与防御机制以及保护语言模型免受复杂攻击的挑战等更广泛的讨论联系起来。
- **革新 PEFT 内存效率**：一篇[论文](https://arxiv.org/abs/2306.00477)研究了参数高效微调（PEFT）方法中的内存效率低下问题，提出了一种可逆模型方法。这旨在降低激活内存，同时保持性能。
   - 讨论强调了将 PLM 修改为可逆变体的难度，以及在 PEFT 方法中保留起始点的重要作用。
- **估算 ML 集群的网络成本**：一份分享以征求反馈的草案文档旨在通过分析 H100 SuperPOD 的组件来估算 ML 集群中的网络硬件成本。该草案在详细的电子表格中使用了零售价格。
   - 建议包括咨询 SemiAnalysis 等专家以获得准确的成本估算，并比较较小 DGX 集群的租赁价格，以推断更大规模建设的成本。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2306.00477">Make Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning</a>：预训练语言模型（PLMs）的参数高效微调（PEFT）已成为一种非常成功的方法，只需训练少量参数而不牺牲性能...</li><li><a href="https://arxiv.org/abs/2406.20053">Covert Malicious Finetuning: Challenges in Safeguarding LLM Adaptation</a>：黑盒微调是一种新兴的接口，用于将最先进的语言模型适配到用户需求。然而，这种访问权限也可能让恶意行为者破坏模型的安全性。为了演示...</li><li><a href="https://arxiv.org/abs/1707.04585">The Reversible Residual Network: Backpropagation Without Storing Activations</a>：深度残差网络（ResNets）显著推动了图像分类领域的最前沿技术，随着网络变得更深更宽，性能也不断提高。然而，内存消耗...</li><li><a href="https://arxiv.org/abs/2406.20086">Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs</a>：LLMs 将文本处理为 Token 序列，这些 Token 大致对应于单词，其中较少见的单词由多个 Token 表示。然而，单个 Token 通常在语义上与...</li><li><a href="https://arxiv.org/abs/2407.01178">$\text{Memory}^3$: Language Modeling with Explicit Memory</a>：LLMs 的训练和推理共同构成了一个昂贵的过程，将知识从原始数据传输到有意义的计算中。受人类记忆层级的启发...</li><li><a href="https://www.semianalysis.com/">SemiAnalysis | Dylan Patel | Substack</a>：弥合世界上最重要的行业——半导体与商业之间的鸿沟。点击阅读由 Dylan Patel 撰写的 SemiAnalysis，这是一个拥有数十万订阅者的 Substack 出版物...</li><li><a href="https://arxiv.org/abs/2406.05183">The Factorization Curse: Which Tokens You Predict Underlie the Reversal Curse and More</a>：如今最好的语言模型仍然在与幻觉作斗争：即事实错误的生成，这阻碍了它们可靠地检索训练期间看到的信息的能力。反转诅咒...</li><li><a href="https://docs.google.com/document/d/1X38uYXCT1mG-rkM8dL9ivSydCCpkkouxXl8zPm1w54Q">Estimating cluster networking hardware costs</a>：状态：草案，开放外部评审，目前请勿发布到 Eleuther Discord 之外。估算 AI 集群中网络硬件成本 摘要 我们调查了网络硬件的成本...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1257464914461855764)** (23 条消息🔥): 

> `使用 lm_eval 复现 Gemma 2 指标时遇到的问题`, `Gemma 2 指标复现问题的可能修复方案`, `Fewshot Prompting 的可视化及其对准确率的影响`, `关于 lm_eval 中评估框架和评分机制的说明`, `OpenAI 的 evals 库与 lm_eval 的集成`

- **Gemma 2 复现问题说明**：用户在使用 `lm_eval` 复现 **Gemma 2** 指标时遇到困难，并发现 **piqa** 等准确率基准测试与 Model Card 报告的指标存在差异。他们尝试了使用正确版本的 Transformers 并将 dtype 设置为 **bfloat16** [(详情)](https://x.com/LysandreJik/status/1807779464849273343)。
   - 社区成员建议检查 Instruction Tuning 和特定模型的 Token 设置，例如使用 `add_bos_token=true` 以更好地匹配结果 [(讨论)](https://huggingface.co/google/gemma-2b/discussions/18)。
- **BOS Token 修复提升 Gemma 2 评分**：在 `model_args` 中添加 `add_bos_token=true` 显著提升了 **Gemma-2** 的评估指标，使其得分更接近 **Gemma 2 模型论文**中报告的数值。
   - Hailey 展示了清晰的前后对比结果，其中 **lambada_openai** 的准确率从 **0.2663 跃升至 0.7518**。
- **探索 Fewshot Prompting 的影响**：成员们观察到 Fewshot Prompting 对评估准确率产生了负面影响，并正在寻找可视化和检查这些 Prompt 的方法。
   - Hailey 建议使用 `--log_samples` 来保存和检查输出，并提到将审查 `--write_out` 标志存在的问题。
- **框架机制与文档**：有人对评估框架的机制提出了疑问，例如 **lm_eval** 中的隐藏 Prompt 和答案提取方法。
   - Hailey 澄清说并没有使用 **LLM-as-a-Judge**；相反，他们依赖预定义的 Prompt 和正则匹配（Regex Matching）等过滤步骤来对输出进行评分。
- **与 OpenAI Evals 的有限集成**：有人请求在 `lm_eval` 中运行 **OpenAI 的 `evals` 库**，但目前还没有直接的集成。
   - Hailey 指出，虽然在许可允许的情况下集成是可能的，但这需要额外的配置，并非开箱即用。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/LysandreJik/status/1807779464849273343)">Lysandre (@LysandreJik) 的推文</a>: 上周 Gemma 2 发布了。从那时起，实现已经过调整以反映模型性能：pip install -U transformers==4.42.3。我们看到有报告称工具（transformers, llama.cpp）未能...</li><li><a href="https://huggingface.co/google/gemma-2b/discussions/18">google/gemma-2b · 使用 lm-evaluation-harness 无法在服务器基准测试上复现结果</a>: 未找到描述</li><li><a href="https://github.com/openai/evals">GitHub - openai/evals: Evals 是一个用于评估 LLM 和 LLM 系统的框架，也是一个开源的基准测试注册库。</a>: Evals 是一个用于评估 LLM 和 LLM 系统的框架，也是一个开源的基准测试注册库。 - openai/evals
</li>
</ul>

</div>

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1257613231250341918)** (3 条消息): 

> `Apple 在端侧 LLM 中的可变位量化`, `Apple 用于模型可视化和优化的 Talaria 工具`, `WWDC 2024 上介绍的 Apple Foundation Models`, `介绍不含 residuals、dot product attention 或 normalization 的 Terminator 架构`

- **Apple 的可变位量化为端侧 LLM 赋能**：Apple 在其端侧大语言模型 (LLM) 中利用 **variable bit quantization**（可变位量化）来增强性能。该技术的详细信息记录在 2024 年 4 月发布的 [Talaria tool](https://machinelearning.apple.com/research/talaria) 中。
   - 该论文获得了 **Best Paper Honorable Mention**，研究人员包括 Fred Hohman 和 Chaoqun Wang 等。自内部部署以来，Talaria 已有 800 多名从业者提交了超过 3,600 个模型。
- **Apple Foundation Models 在 WWDC 2024 首次亮相**：在 2024 年 [Worldwide Developers Conference](https://developer.apple.com/wwdc24/) 上，Apple 推出了 **Apple Intelligence**，这是一个深度集成在 iOS 18、iPadOS 18 和 macOS Sequoia 中的个人智能系统。这些基础模型包括一个约 30 亿参数的端侧 LLM 和一个可通过 [Private Cloud Compute](https://security.apple.com/blog/private-cloud-compute/) 使用的服务器端模型。
   - 根据 [Apple 的报告](https://machinelearning.apple.com/research/introducing-apple-foundation-models)，这些生成式模型专为日常任务量身定制，如写作、总结通知和创作趣味图像。
- **LeopolisDream 预告 Terminator 架构问世**：@LeopolisDream 通过 Twitter 揭晓了一种名为 **Terminator** 的新架构，详见[此处](https://x.com/leopolisdream/status/1804627325583327358?s=46&t=BsqYoGA8vIHGcXwORlMk7w)。该架构显著的特点是省略了 residuals、dot product attention 和 normalization。
   - 此公告链接到一篇详细的 [arXiv paper](https://arxiv.org/pdf/2401.17948)，承诺在模型设计和优化方面采用新颖的方法。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/leopolisdream/status/1804627325583327358?s=46&t=BsqYoGA8vIHGcXwORlMk7w">来自 Alex Yanko 🇺🇦 (@LeopolisDream) 的推文</a>：欢迎新架构：Terminator。无 residuals，无 dot product attention，无 normalization... https://arxiv.org/pdf/2401.17948</li><li><a href="https://machinelearning.apple.com/research/introducing-apple-foundation-models">介绍 Apple 的端侧和服务器基础模型</a>：在 2024 年全球开发者大会上，我们推出了 Apple Intelligence，这是一个深度集成的个人智能系统……</li><li><a href="https://machinelearning.apple.com/research/talaria">Talaria：交互式优化机器学习模型以实现高效推理</a>：端侧机器学习 (ML) 将计算从云端转移到个人设备，保护用户隐私并实现智能用户体验……</li><li><a href="https://arxiv.org/abs/2404.03085">Talaria：交互式优化机器学习模型以实现高效推理</a>：端侧机器学习 (ML) 将计算从云端转移到个人设备，保护用户隐私并实现智能用户体验。然而，在资源受限的设备上适配模型……
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1257447302054809710)** (1 messages): 

> `PersonaHub 介绍及潜在用例`, `多场演出节日的排期与物流关键考量`, `哈利法克斯公共服务的分布与组织`, `合成数据在 LLM 研发中的应用`

- **PersonaHub 通过多样化人格（personas）扩展合成数据生成**：这个 [Hugging Face 仓库](https://huggingface.co/datasets/proj-persona/PersonaHub) 介绍了 PERSONA HUB，这是一个包含 10 亿个多样化人格的集合，用于创建合成数据。该方法促进了各种应用的高质量数据合成，可能推动 LLM 研发的范式转移。
   - *一种利用 LLM 视角的新型人格驱动数据合成方法*，允许进行通用、可扩展且灵活的数据创建。论文强调了其在逻辑推理问题和游戏 NPC 创建等方面的应用，展示了其深远的影响。
- **多场演出节日的物流排期**：在 **Broward Center** 等表演艺术中心进行排期和物流的关键考量包括运营方面、设施和节目管理。协调多个演出时间表以确保无缝衔接和观众管理至关重要。
   - 有效的物流规划应考虑到*各种设施的利用*和观众动态。管理多场演出需要仔细协调，以避免中心运营中的冲突和瓶颈。

**提及的链接**：<a href="https://huggingface.co/datasets/proj-persona/PersonaHub">proj-persona/PersonaHub · Hugging Face 数据集</a>：未找到描述

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1257664695305900104)** (2 messages): 

> `NotDevin 的 MatrixBridge AI 演示`, `Yaya Labs 在意大利举办的黑客松`

- **MatrixBridge AI 发布 NotDevin 演示**：专注于企业级浏览器 Agent 的 MatrixBridge AI 发布了其最新 Agent **NotDevin** 的[演示](https://x.com/AnandaSaiA/status/1806763965420331478)，展示了它如何通过其平台和 Google 的 Project IDX 取代 Devin。他们鼓励用户注册候补名单，开始构建 NoCode 专家级浏览器 Agent。
   - 邀请成员提供反馈，并鼓励[联系](http://matrixbridgeai.com)以获取更多信息，通过转发推文积累好评。
- **意大利黑客松：Hypersensual Overclocking 2024**：Yaya Labs 与 Umanesimo Art 合作宣布了一场名为 [Hypersensual Overclocking 2024: Making Machines Feel](https://x.com/yaya_labs_/status/1808097567739121980) 的黑客松。该活动在意大利举行，现已开放申请，有机会赢取 Yaya Labs 的驻留名额。
   - 有兴趣加入的参与者可以通过 Yaya Labs 简介中的链接申请，参与体验并可能获得 **Yaya Labs 的驻留机会**。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/yaya_labs_/status/1808097567739121980">来自 Yaya Labs (@yaya_labs_) 的推文</a>：我们与 @umanesimoart 合作的黑客松“Hypersensual Overclocking 2024: Making Machines Feel”现已开放申请 🇮🇹 立即通过我们简介中的链接申请，有机会...</li><li><a href="https://x.com/AnandaSaiA/status/1806763965420331478">来自 Sai (@AnandaSaiA) 的推文</a>：🚀 在你的浏览器中取代一家估值 20 亿美元的初创公司！ 🚀 介绍 NotDevin - 由 @Google 的 Project IDX 提供支持，使用 @MatrixBridgeAI 构建的 AI 工作者。注册我们的候补名单，开始构建 NoCode 专家级浏览器...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

harvie_zhang: https://x.com/leopolisdream/status/1804627325583327358?s=46&t=BsqYoGA8vIHGcXwORlMk7w
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1257470957275058236)** (82 messages🔥🔥): 

> `使用具备视觉能力的语言模型 (VLLMs) 进行动画制作`, `Runway 的高保真视频生成`, `对 Runway AI 工具成本的担忧`, `VLLMs 和多模态模型的新技术`, `Llama 模型的自定义量化方案`

- **使用 VLLMs 进行动画制作的想法引发讨论**：一位用户提出，通过合适的数据集和工作流，可以教会 **Vision Capable Language Models (VLLMs)** 进行绑定并自动化基础动画，这引发了技术讨论。他们建议模型可以编写动画程序并进行迭代自我修正，从而提高动画制作效率。
   - *Verafice* 指出了一个更高效的解决方案，并链接到了一个[基于 Diffusion 的关键帧补间方法](https://setarehc.github.io/CondMDI/)，该方法可以生成精确且多样化的角色动作。*Teknium* 等人一致认为 *Hermes Pro* 在这方面可能具有潜力。
- **Runway 的视频生成尽管成本高昂仍引起关注**：Runway 新推出的高保真、可控视频生成工具在社区中引起了热烈讨论，重点围绕其潜力展开。然而，**高昂的成本**（60 秒 12 美元）引发了关于负担能力的争论。
   - *Mautonomy* 等人批评了其定价结构，认为 **每次生成 0.5 美元** 会更合理。一些用户甚至将其与其他昂贵的服务（如音频转录）进行了比较。
- **vLLMs 的支持与实验**：关于支持和实验视觉能力语言模型 (**vLLMs**) 的讨论非常活跃，一些用户参与了训练和增强其功能的工作。值得注意的是，*Teknium* 提到协助开发者训练视觉模型并支持相关实现。
   - 社区分享了关于 **vLLMs** 内部工具调用 (tool calling) 的各种努力和[支持](https://github.com/vllm-project/vllm/pull/5649)亮点，表明社区正在持续推动这些模型的发展。
- **Llama.cpp 特征提取改进**：讨论了 **Llama 模型** 的特征提取工作，特别是将表征工程 (representation engineering) 集成到 llama.cpp 中。*Teknium* 分享了一个[相关推文链接](https://x.com/NousResearch/status/1769748540392387039)，强调了对 GGUF 的支持。
   - 研究人员还在为 **llama.cpp** 开发[自定义逐层量化 (custom per-layer quantization)](https://github.com/ggerganov/llama.cpp/pull/6844)，这被视为模型优化和效率提升方面一个很有前景的进展。
- **苹果用于设备端 ML 优化的 Talaria 工具**：提到了苹果用于设备端 ML 优化的 Talaria 工具，强调其专注于模型大小、延迟和功耗指标。该工具可以可视化模型统计数据并模拟优化方案。
   - *Azure2089* 引用了一篇详细介绍 Talaria 的[论文](https://arxiv.org/abs/2404.03085)，称赞其在优化和评估设备端模型方面的作用。**这对于开源 LLMs 来说可能是一个极好的补充。**
<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://x.com/justin_hart/status/1807967646375371107">来自 Justin Hart (@justin_hart) 的推文</a>：好的，这是我最喜欢的 @runwayml 作品之一。</li><li><a href="https://x.com/summeryue0/status/1807806106108101092?s=46">来自 Summer Yue (@summeryue0) 的推文</a>：1. Claude 3.5 Sonnet 现在在 SEAL 排行榜的指令遵循（Instruction Following）类别中排名第一 (http://scale.com/leaderboard) 🏆</li><li><a href="https://arxiv.org/abs/2404.03085">Talaria：交互式优化机器学习模型以实现高效推理</a>：设备端机器学习 (ML) 将计算从云端转移到个人设备，保护用户隐私并实现智能用户体验。然而，在受限设备上部署模型...</li><li><a href="https://runwayml.com/">Runway - 用人工智能推动创意。</a>：Runway 是一家应用 AI 研究公司，致力于塑造艺术、娱乐和人类创意的新时代。</li><li><a href="https://setarehc.github.io/CondMDI/">使用 Diffusion Models 的灵活动作插值</a>：未找到描述</li><li><a href="https://www.refuel.ai/blog-posts/labeling-with-confidence">带着信心进行标注</a>：未找到描述</li><li><a href="https://tenor.com/view/suicide-jump-pink-pantern-gif-14045004">Suicide Jump GIF - Suicide Jump Pink Pantern - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/jamesyeung18/status/1807673154636227015">来自 James Yeung (@jamesyeung18) 的推文</a>：📽️ 我在 @runwayml GEN-3 上制作的 8 个新视频，帖子中附带了每个视频的提示词。你最喜欢哪一个？（第 7 个是我的最爱）👇 1. 一辆警车在旋转的...无人机镜头。</li><li><a href="https://x.com/NousResearch/status/1769748540392387039">来自 Nous Research (@NousResearch) 的推文</a>：我们自豪地宣布 llama.cpp 中 @voooooogel 的 repeng 自定义控制向量实现已支持 GGUF。用户现在可以使用自然语言生成逐层向量，并添加到...</li><li><a href="https://github.com/vllm-project/vllm/pull/5649">支持允许 OpenAI API 风格工具调用和“auto”工具选择的开源模型，由 K-Mistele 提交 · Pull Request #5649 · vllm-project/vllm</a>：草案：OpenAI 工具调用清单。此（草案）PR 将以对工具使用格式和提示词格式保持最小偏见的方式，添加对 OpenAI 风格工具调用的支持。以下功能...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6844">由 jubruckne 提交的自定义量化方案 · Pull Request #6844 · ggerganov/llama.cpp</a>：这尚未准备好合并，但我希望征求你们的意见，看是否对此感兴趣。如果是的话，我可以清理并改进它。这个想法是允许创建一个自定义的...</li><li><a href="https://machinelearning.apple.com/research/talaria">Talaria：交互式优化机器学习模型以实现高效推理</a>：设备端机器学习 (ML) 将计算从云端转移到个人设备，保护用户隐私并实现智能用户体验...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1257773242156847146)** (3 条消息): 

> `从文档创建对话数据集或指令数据集`, `用于生成数据集的开源工具`, `预算充足时用于数据集生成的 Anthropic 解决方案`, `用于从原始文本语料库生成指令的 Genstruct 7B 模型`, `受 Ada-Instruct 模型启发进行指令生成`

- **从文档创建对话数据集**：**Macacodomato4260** 询问了关于从文档创建**对话数据集或指令数据集**的线索。**Kainan_e** 回应称，这取决于**文档类型和可用预算**。
   - **Kainan_e** 建议使用由 NousResearch 开发的**开源工具**，或者利用 **Anthropic 的工具**配合适当的提示词和预算（[Genstruct 7B](https://huggingface.co/NousResearch/Genstruct-7B), [Ada-Instruct](https://arxiv.org/abs/2310.04484)）。
- **Genstruct 7B：数据集生成工具**：**Kainan_e** 分享了 **[Genstruct 7B](https://huggingface.co/NousResearch/Genstruct-7B)** 的链接，这是 NousResearch 开发的一个指令生成模型。该模型旨在从原始文本语料库中创建有效的指令，以促进合成指令微调（instruction finetuning）数据集的生成。
   - Genstruct 7B 的开发受到了 **[Ada-Instruct](https://arxiv.org/abs/2310.04484)** 的启发，后者训练了一个自定义的指令生成模型。他指出，以前的方法主要依赖于 **in-context（上下文内）方法**。

**提到的链接**：<a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>：未找到描述

  

---

### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1257430538323755159)** (4 messages): 

> `将机器人打磨为链接至 Nous 的问答机器人`, `apyh 修复了问题`, `用户对 AI 场景的赞赏`, `Teknium 对机器人未来可能性的展望`

- **将机器人打磨为问答工具**：一位用户询问是否可以将机器人**打磨（sharpening）**成链接到各种 Nous 相关领域的问答机器人。这将利用 Nous 提供的所有资源来简化信息获取。
   - 另一位用户确认**总有一天这会实现**，暗示了社区内机器人能力的未来增强。
- **apyh 实施的修复**：一位名为 apyh 的用户提到**修复了一个问题**，并自信地表示现在应该可以正常工作了。没有提供关于该问题的具体细节。
   - 另一位用户表达了对 *Super cool AI* 的赞赏，他开始尝试新的场景，并对 AI 的能力感到兴奋。
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1257436032526450730)** (5 messages): 

> `在 Raspberry Pi 5 的 Ubuntu 24.04 上运行 Mojo`, `AI 工程师世界博览会演讲链接`, `从子目录导入 .mojopkg`

- **请求 AI 工程师世界博览会演讲时间戳**：一位用户请求 AI 工程师世界博览会演讲的链接，另一位用户提供了 [YouTube 视频链接](https://www.youtube.com/live/vaIiNZoXymg?t=1326s) 以及时间戳。
   - 该用户提到有几个长达 8 小时的视频，不确定哪一个包含相关的演讲。
- **在 Raspberry Pi 5 上运行 Mojo**：一位用户提到了在 **Raspberry Pi 5** 的 **Ubuntu 24.04** 上运行 Mojo 时遇到的问题，并正在寻求帮助。
   - 该消息没有包含关于此问题的进一步讨论或解决方案。
- **从子目录导入 .mojopkg**：一位用户询问是否可以将 **.mojopkg** 放在子目录中并从中导入。
   - 该消息未收到任何回复或解决方案。

**提到的链接**：<a href="https://www.youtube.com/live/vaIiNZoXymg?t=1326s">AI Engineer World’s Fair 2024 - Keynotes &amp; Multimodality track</a>: https://twitter.com/aidotengineer1. 开场音乐 - 00:00；2. 公告 - 03:26；3. AI Engineer Summit 开幕致辞 - 17:12；4. Benjamin 演讲 - 17:22...

  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1808228006068212110>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1257764978602016818)** (2 messages): 

> `Mojo 充满前景的领域`, `针对 LLM Agent 的基于模型和模拟器的强化学习 (RL)`, `符号推理和推理时搜索`, `亚符号模型引导`, `受限推理`

- **Mojo 在各种应用中充满前景**：Mojo 在调整 Attention 和 Scaling 之外的领域显示出潜力，例如针对 LLM Agent 的**基于模型和模拟器的强化学习 (RL)**、**符号推理**、类似 **MCTS** 的**推理时搜索**以及**亚符号模型引导**。
   - *是否有人在 Mojo 中研究这些应用？* 用户非常希望与探索这些领域的其他人建立联系。
- **关于受限推理的讨论**：在 Mojo 的背景下，像 **guidance**、**outlines** 或 **sglang** 这样的受限推理（Constrained inference）方法似乎也很有前景。
   - Mojo 也可以作为探索这些受限推理方法的强大工具。

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1257435983910015086)** (27 条消息🔥): 

> `Mojo 生命周期参数`, `在 Mojo 中使用 MutableLifetime 和 AnyLifetime`, `Mojo 对包管理器的必要性`, `Mojo 中 parallelize 和 sync_parallelize 的区别`, `在 Mojo 中使用 List 并打印其元素`

- **Mojo 生命周期参数：MutableLifetime 和 AnyLifetime**：讨论始于一位用户在 Mojo 中实现生命周期时遇到的困难，提到了 **MutableLifetime** 作为一个参数。另一位成员建议使用 **AnyLifetime** 来实现通用的可变性，并提供了一个涉及 **UnsafePointer** 和 **mlir_attr** 的工作示例。
   - 引用中提到了 stable 和 nightly 版本之间的不同行为，以及生命周期如何与引用转换（reference casting）相关联。讨论以针对 stable 版本的可行解决方案结束。
- **Mojo 包管理器需求**：成员们表达了对 **包管理器** 的需求，以处理 **Mojo** 中日益增长的社区项目和依赖关系。他们讨论了过去提到的非官方包管理器的努力。
   - 共识是，对于处理外部包和社区贡献的人来说，官方包管理器将非常有益。
- **Mojo 中的并行化：parallelize vs sync_parallelize**：一位用户询问了 Mojo 中 `parallelize` 和 `sync_parallelize` 的区别，并对 `parallelize` 中的运行时警告表示担忧。他们就应该在内核上下文（kernel contexts）中使用哪个函数寻求建议。
   - 对话包含了关于本地运行时创建和销毁警告的细节，并暗示 `sync_parallelize` 可能没有同样的警告，但没有得出最终结论。
- **在 Mojo 中处理 List**：一位用户在 Mojo 中向 **List** 追加和打印项时遇到困难，指出直接打印存在问题。其他成员解释说 **List** 缺少 **Stringable** trait，并建议使用 `[]` 解引用来进行迭代和打印。
   - 此外，还指向了 Mojo 标准库的 GitHub，以获取有关 trait 实现和数据类型处理的更多细节。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/types#list)">Types | Modular 文档</a>: 标准 Mojo 数据类型。</li><li><a href="https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/stdlib/src/builtin/str.mojo#L23">modularml/mojo 的 GitHub 源码</a>: Mojo 编程语言。通过在 GitHub 上创建一个账户为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1257418870621474899)** (7 条消息): 

> `Mojo 编译器 nightly 更新`, `关于 stdlib 贡献者和 Mojo 周边的讨论`, `nightly/max 包更新问题`, `CI 迁移及产生的问题`

- **新的 Mojo 编译器 nightly 更新发布**：新的 nightly Mojo 编译器版本 **2024.7.205** 已发布；你可以使用 `modular update nightly/mojo` 进行更新。[更新日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)和 [原始差异 (raw diff)](https://github.com/modularml/mojo/compare/faaf19c73a9c0f76f9799d361df4199724868905...5b77a66cb42143ffbcf39db635964ae344e63d25) 已在线发布。
   - 社区成员对更新表示兴奋，而其他一些人则遇到了包更新的问题。
- **联系获取 Mojo 周边**：一位用户询问作为 stdlib 贡献者应该联系谁获取 Mojo 周边。**Jack Clayton** 回复确认他将通过私信（DM）联系。
   - *很高兴看到你为了它们而留下来！应该很快就会修复*，Jack Clayton 在处理包更新问题时评论道。
- **nightly/max 包更新问题仍然存在**：一位成员报告说，尽管发布了新版本，但 `nightly/max` 包没有可用的更新。该用户决定暂时恢复使用 `modular update nightly/mojo`。
   - **Jack Clayton** 承认了这个问题，解释说在最初的修复之后出现了进一步的问题，由于 CI 迁移，这些问题应该在下次更新中得到解决。
  

---

### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1257412352362414200)** (44 messages🔥): 

> `关于 src/main 编译问题的讨论`, `由于舍入误差导致基准测试失败`, `提供用于故障排除的系统规格`, `基准测试改进讨论`, `测试和基准测试框架已更新`, `关于矩阵乘法数据类型的讨论`, `关于改进矩阵乘法算法的集思广益`

- **`src/main` 中的编译问题被标记**：一位用户报告了在尝试运行 `mojo main.mojo` 时出现的编译错误，具体错误与 `matrix.mojo` 中的 `DTypePointer` 属性有关。
   - *建议的修复方案是使用最新的 stable 版本进行构建，而不是 nightly 版本*。
- **由于舍入误差导致基准测试失败**：一名成员遇到了基准测试问题，函数产生的微小结果差异导致了断言失败。
   - *修复补丁已推送到 GitHub 并解决了问题，尽管随后一些用户在进行 GFlops/s 基准测试时遇到了死循环*。
- **改进的基准测试框架**：得益于社区的贡献，测试和基准测试框架已更新，以更好地衡量所有数据类型和规模的性能。
   - *这些更新鼓励参与者更新其分支以确保兼容性*。
- **讨论了矩阵乘法的数据类型**：关于在矩阵乘法基准测试中使用 `DType.float16` 是否合适存在争论，因为它可能不是最常用的数据类型。
   - *系统将测试各种类型，包括不同大小（8/16/32/64）的 `uint/int/float`*。
- **集思广益矩阵乘法的改进方案**：参与者分享了优化矩阵乘法的想法，包括 vectorization、unrolling 和 tiling 技术。
   - *还讨论了结合 Strassen 和 Winograd-Copper 等不同算法以及 zero padding 逻辑的潜力。*
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://developer.arm.com/documentation/102476/0100/Introducing-SVE">Documentation – Arm Developer</a>：未找到描述</li><li><a href="https://github.com/Benny-Nottonson/Mojo-Marathons/pull/1">Improve the performance reporting by gabrieldemarmiesse · Pull Request #1 · Benny-Nottonson/Mojo-Marathons</a>：以下是现在的性能报告样式：✅ Passed test with M = 1 , N = 1 , K = 1 ✅ Passed test with M = 1 , N = 47 , K = 97 ✅ Passed test with M = 53 , N = 1 , K = 101 ✅ Passed test with...</li><li><a href="https://github.com/aws/aws-graviton-getting-started/blob/main/README.md">aws-graviton-getting-started/README.md at main · aws/aws-graviton-getting-started</a>：帮助开发者使用 AWS Graviton2 和 Graviton3 处理器，这些处理器为第 6 代和第 7 代 Amazon EC2 实例提供动力 (C6g[d], M6g[d], R6g[d], T4g, X2gd, C6gn, I4g, Im4gn, Is4gen, G5g, C7...</li><li><a href="https://docs.google.com/spreadsheets/d/1TBz9Lp0JT1Ph7ndfbWqp-B30FQcRYl1959hP2lZ6yH4/edit">Matrix Multiplication</a>：表格 1 约束、参数 / 调优 Vectorization、Contiguous Access、Nelts、Unrollable Parallelization、Unrollable Unrolling、Contiguous Operations Tiling Square Optimized、Amorized Increase、Recursiv...
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1257776175862779985)** (1 messages): 

> `/models 页面重大更新`, `更改 Gemini 和 PaLM 模型的 Google Token 大小`, `弃用默认模型（Default Model）设置`, `弃用用于 OpenAI API 密钥的自定义认证头（auth headers）`

- **/models 页面重大更新**：**/models 页面**即将迎来**重大更新**。官方分享了预览，并鼓励在指定的反馈频道提供反馈。
   - *让我们知道你想看到什么。*
- **Google 模型 Token 大小变更**：**Gemini 和 PaLM** 模型将迁移到 **GPT 等效的 Token 长度**，以使其统计数据更加标准化，这将导致**价格上涨**和上下文限制减少。
   - 这将保持相同的模型和 API，但具有不同的 Token 大小和定价。
- **弃用默认模型设置**：**/settings 页面**上的 **Default Model** 设置现已进入弃用队列。
   - 如果你有合理的理由需要使用它，请提供反馈，因为大多数应用都在单独设置模型或使用自动路由。
- **弃用自定义认证头**：正在弃用使用**自定义认证头（custom auth headers）**发送 OpenAI API 密钥的方式。
   - 很快将有更好的解决方案取代它；该方式在 6 月中旬被少数请求使用，但从未被正式记录在文档中。
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

lastrosade: 我做了一个简易的封装（wrapper），如果有人需要的话。
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1257419901266821140)** (68 条消息🔥🔥): 

> `Mistral API 错误处理`, `LiteLLM 与 OpenRouter 的区别`, `提高对话机器人的效率`, `Claude 3.5 间歇性错误`, `iOS 上的 OpenRouter 前端应用`

- **Mistral API 错误排查**：成员们讨论了在使用 OpenRouter 配合 Sonnet 3.5 时收到 Mistral API 错误的问题，尽管他们并没有直接使用 Mistral。正如 [Activity 页面](https://openrouter.ai/activity) 所解释的，问题似乎在于当主模型请求失败时可能会回退（fallback）到 Mistral。
   - 有人指出，错误源自推理提供商（inference provider）的后端，而非 OpenRouter 本身。建议联系 Aider 的支持团队，以核实为何会发生回退。
- **LiteLLM vs. OpenRouter**：成员们询问了 LiteLLM 和 OpenRouter 之间的区别，了解到 LiteLLM 本质上是软件，而 OpenRouter 是在线 API 服务。[文档说明](https://openrouter.ai/docs) 澄清了 OpenRouter 会原样转发来自提供商的消息。
   - 讨论强调了 OpenRouter 作为中间商的角色，用户需要设置偏好以避免回退到不想要的模型。
- **增强 Discord 机器人对话**：讨论集中在如何通过仅将消息的相关部分附加到 Prompt 来提高使用 OpenRouter 的 Discord 机器人的效率，从而节省 Token。经证实，将完整对话保存在别处并仅包含核心片段是常见做法。
   - 正如 [SillyTavern Discord](https://sillytavern.app) 用户所讨论的，由于可用模型的上下文窗口（context size）有限，使用高效的 Prompt 管理策略对于长对话至关重要。
- **处理 Claude 3.5 错误**：成员们报告了 Claude 3.5 模型间歇性出现 500 错误，影响了自我审核（self-moderated）和普通审核版本。Claude 3.0 仍可正常工作。
   - 这些错误似乎是特定于模型的故障，而非更广泛的基础设施问题。有建议称修复工作正在进行中。
- **寻找适用于 OpenRouter 的 iOS 应用**：关于支持 OpenRouter 的 iOS 应用的咨询引荐了 Pal Chat 和 Typingmind。据报道，这两款应用都修复了之前与 OpenRouter 相关的 Bug，现在支持多种 API。
   - 社区还讨论了 LibreChat 和 SillyTavern 等其他前端，尽管一些用户对专注于角色扮演（roleplay）的界面表示不满。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sillytavern.app)">未找到标题</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>：为模型消耗转换数据</li><li><a href="https://www.getimg.ai">使用 AI 创建图像所需的一切 | getimg.ai</a>：神奇的 AI 艺术工具。生成原创图像、修改现有图像、扩展图片边界等。</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>：查看你在 OpenRouter 上使用模型的情况。
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1257422066534191237)** (49 条消息🔥): 

> `指令微调模型讨论`, `在 IT 模型而非 Base 模型上持续训练的缺点`, `计划在 axolotl 中添加 CAME 或 Adam-mini 优化器`, `微调 gemma 27b 时出现高 grad_norm 值`, `优先考虑数值答案的准确性而非解释性文本`

- **指令微调（IT）模型引发辩论**：一位成员讨论了使用模型的指令微调（IT）版本的用法，指出了直接使用 IT 版本或进一步微调的潜在好处。另一位成员回应称，对预训练模型进行去审查可能更容易，而通过对 IT 版本进行微调来改变其风格则更难。
   - 成员们补充了关于开源 IT 数据集的见解，认为它们通常比闭源 IT 表现差，这影响了训练方法。
- **对 CAME 和 Adam-mini 优化器的兴趣**：一位成员询问是否在 axolotl 项目中添加 CAME 或 Adam-mini 优化器，强调了它们节省显存的优势。他们分享了关于 [CAME (arxiv.org/abs/2307.02047)](https://arxiv.org/abs/2307.02047) 和 [Adam-mini (arxiv.org/pdf/2406.16793)](https://arxiv.org/abs/2406.16793) 的论文。
   - 有人指出 CAME 提供了训练稳定性和性能，而 Adam-mini 在显存占用更少的情况下，性能可以与 AdamW 持平甚至更好。
- **微调 Gemma 27b 时的 grad_norm 过高问题**：一位成员报告在尝试微调 Gemma 27b 时看到了巨大的 grad_norm 值，而 9b 模型没有此类问题。另一位成员建议使用较低的学习率和 Flash Attention，因为该模型存在 softcap 约束。
   - *Gemma* 被描述为训练起来很痛苦，存在过拟合问题，并被拿来与训练 Llama3 的困难程度做类比。
- **加权交叉熵用于响应优先级**：一位正在进行项目的成员希望在模型输出中强调数值答案而非解释性文本，并分享了一个使用权重的结构化 Prompt 示例。他们正在寻找该领域的研究，但未能找到相关的论文或信息。
   - 在注意到 Trainer 中关于加权交叉熵的部分后，他们决定进一步探索该方法。
- **在 ORPO 中生成接受/拒绝（accepted/rejected）对的困难**：一位成员询问 ORPO 是否要求训练数据中的每一行都必须有严格的接受-拒绝对，并表示生成这些对很困难。另一位成员确认两者都是必需的，并提到这就是 ORPO 计算奖励的方式。
   - 这意味着生成接受/拒绝对对于 ORPO 的偏好对齐（preference alignment）至关重要，这使得该过程对某些用户来说比较棘手。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/trl/orpo_trainer">ORPO Trainer</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2307.02047">CAME: Confidence-guided Adaptive Memory Efficient Optimization</a>: 自适应梯度方法（如 Adam 和 LAMB）在训练大语言模型中表现出卓越的性能。然而，自适应性需要维护二阶矩...</li><li><a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: 我们提出了 Adam-mini，这是一种优化器，其性能与 AdamW 持平或更好，但显存占用减少了 45% 到 50%。Adam-mini 通过削减学习率资源来减少显存...
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1257558367111610369)** (3 条消息): 

> `错误消息调查`, `配置问题`

- **Nanobitz 请求日志详情以进行错误调查**：**Nanobitz** 请求提供更多日志细节，以确定其设置中出现问题的原因。该请求表明需要进行更深入的错误分析。
   - *你能提供更多日志以确定是哪一步导致的吗？*
- **Louist4455 分享完整的错误消息详情**：**Louist4455** 分享了完整的错误消息，以响应之前对更多信息的需求。这旨在帮助故障排除过程。
   - *这是完整的错误消息：*
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1257513996722901033)** (13 条消息🔥): 

> `使用 eager attention 训练 Gemma2 模型`, `为 Gemma2 的 eager attention 修改 YAML`, `Batch Size 是否随 GPU 数量增加`

- **在 Gemma2 训练中使用 eager attention**：[一项重要建议](https://openaccess-ai-collective/axolotl) 提议在训练 Gemma2 模型时使用 `eager` attention 实现，而不是 `flash_attention_2`。具体操作是在 `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>')` 中将 `attn_implementation` 设置为 'eager'。
   - 深入讨论强调了为了获得更好的性能需要进行此项调整，并指出在训练阶段调整该参数至关重要。
- **为 Gemma2 的 eager attention 调整 YAML**：为了在 Gemma2 模型中使用 `eager` attention，有必要[调整 YAML 配置文件](https://openaccess-ai-collective/axolotl)，在模型加载参数下添加 `attn_implementation: 'eager'`。
   - 详细说明提供了修改 YAML 文件的步骤，为在实现中遇到困难的用户简化了流程。
- **GPU 数量与训练 Batch Size 的相关性**：[增加 GPU 数量](https://openaccess-ai-collective/axolotl) 实际上会增加训练的 Batch Size，因为它将每设备的 Batch Size 乘以 GPU 的数量。
   - *框架会自动处理跨多个 GPU 的数据划分*，确保高效的并行处理，以充分利用硬件性能。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=6009ab3c-72a1-4338-a2ba-1275044f1a0d)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=39e28a61-76e3-448b-ae9b-f83c37aef52a)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=75180382-43ef-400d-bf49-320aca482082)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1257492564974637097)** (2 条消息): 

> `将 Python 多 Agent 系统转换为微服务`, `关于 'llama-agents' 框架的全面视频教程`, `构建超越朴素 RAG 的更优知识助手`, `知识助手的高级数据和检索模块组件`

- **将 Python 系统转换为微服务的全面指南**：@MervinPraison 制作了一个关于新 `llama-agents` 框架的详细视频教程，涵盖了高层概念和实践层面。查看[完整视频教程](https://t.co/eW9mT6IHlk)获取分步指南。
   - 该教程被描述为迄今为止最全面的教程，不仅涵盖了基础知识，还包括了关键细节和高级特性。一位成员强调说：*“这是我们目前看到的关于 `llama-agents` 框架最全面的视频。”*
- **利用高级模块构建卓越的知识助手**：在 @aiDotEngineer World Fair 上，讨论集中在如何构建超越朴素 RAG 的知识助手，强调了高级数据和检索能力。三个主要组件包括能够提升性能的创新数据模块。
   - 社区讨论了未来知识助手所需的要素，强调了数据处理复杂性的重要性。一位成员表示：*“构建更好的知识助手需要高级的数据和检索模块。”*
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1257417076897742919)** (43 条消息🔥): 

> `sub-agents 教程与应用`, `新 agents 发布`, `集成自定义消息队列`, `基于 RAG 的公司数据聊天机器人`, `LlamaIndex 中的对话历史`, `构建 llama-cpp-python 的问题`, `Microsoft 的 graph RAG 架构`, `Pinecone 的 DocumentSummaryIndex 元数据问题`

- **Sub-agents 教程与应用**：一名成员询问了关于 **sub-agents** 的优质教程或用途，并被引导至 `llama-agents`。其他成员对启动该功能表示兴奋。
   - 讨论提到了集成 **RabbitMQ** 以及将 **Kafka** 作为 Agent 间通信的自定义消息队列的潜力。一条引用提到 *“一旦代码抽象完全添加，可能会帮助你设置任何消息队列”*。
- **Microsoft 的 Graph RAG 架构发布**：社区成员分享了 Microsoft 新的 [Graph RAG Architecture](https://github.com/microsoft/graphrag) 链接，将其描述为一个模块化的、基于图的检索增强生成 (RAG) 系统。
   - 该发布引发了成员们的好奇和期待，其中一人提到他们 *“很想听听社区对新架构的看法”*。
- **Pinecone 的 DocumentSummaryIndex 元数据问题**：一名成员遇到了 **DocumentSummaryIndex** 元数据超过 Pinecone 限制的问题，并建议通过修改代码来排除多余的元数据键。
   - 其他人建议了 **Pinecone** 的替代方案，并讨论了潜在的修复方法，例如删除 `_node_content` 的元数据并在稍后添加，强调了 Pinecone 的局限性。
- **为公司数据构建基于 RAG 的聊天机器人**：有人发布了关于使用 LlamaHub 的不同数据库读取器，结合多个 SQL 和 NoSQL 数据库构建 **基于 RAG 的聊天机器人** 的查询。
   - 问题包括如何根据用户文本形成合适的查询并将其路由到正确的数据库，社区成员对此提供了见解和想法。
- **构建 llama-cpp-python Wheel 的问题**：一名用户在 Google Colab 上构建 **llama-cpp-python** 的 wheel 时遇到失败，错误提示显示子进程问题。
   - 一名社区成员建议进一步检查上方的错误消息以获取线索，暗示问题出在子进程内部，而非 `pip` 本身。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/orgs/run-llama/projects/5?pane=issue&itemId=68386242">llama-agents • run-llama</a>: llama-agents</li><li><a href="https://github.com/run-llama/llama_index/blob/722cb67ca4e52c8c4d6ef8c5e99b7f6c9f57e244/llama-index-core/llama_index/core/indices/document_summary/base.py#L203">llama_index/llama-index-core/llama_index/core/indices/document_summary/base.py at 722cb67ca4e52c8c4d6ef8c5e99b7f6c9f57e244 · run-llama/llama_index</a>: LlamaIndex 是 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/microsoft/graphrag">GitHub - microsoft/graphrag: A modular graph-based Retrieval-Augmented Generation (RAG) system</a>: 一个模块化的、基于图的检索增强生成 (RAG) 系统 - microsoft/graphrag
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1257413825477742773)** (9 条消息🔥): 

> `图重写（graph rewrite）后续及加速/不同算法`，`特殊的 dtype 'image dtype'`，`Tinygrad 中的错误信息和开发工具`，`包含失败测试和最小复现（minimal repro）的 PR`

- **图重写（Graph Rewrite）讨论中**：成员们讨论了“图重写后续、加速/不同算法”议程项；目前尚未决定采用哪种具体的图算法。Egraphs/muGraphs 的优先级被降低，目前重点是将更多算法（如 scheduler）移入图重写中。
   - 一位成员强调，*规则不是图灵完备的，因此易于推理*。另一位成员提到，他们还没准备好押注于某个特定算法。
- **Image Dtype 辩论**：鉴于源代码中特殊的 'image dtype' 受到广泛的“特殊待遇”，有人对其重要性提出了疑问。目前没有提到具体的挑战或尝试删除它的行为。
   - *你尝试过删除它吗？* 是讨论的主要问题，尽管没有记录到其他成员的详细后续。
- **Tinygrad 错误信息亟需大修**：Tinygrad 的 cstyle.py 文件中反复出现的错误引发了关于在 1.0 版本之前改进错误信息和开发工具的讨论。特别提到了 **RuntimeError**: *failed to render UOps.UNMUL*。
   - George Hotz 认为这更多是一个断言（assert）问题，并表示它*永远不应该发生*。一位成员表示赞同，并指出他们将创建一个包含失败测试用例的 PR。
- **最小复现用例的 PR**：已启动一个草案 PR，以解决 Tinygrad 中一个具有最大复现（maximal reproduction）用例的错误。目前正在努力将其转换为合适的最小复现（minimal reproduction）用例。
   - 计划是确保 PR 包含一个*最小复现的失败测试*，以便在应用正式修复之前彻底解决底层问题。

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1257417955436925119)** (34 messages🔥): 

> `Tinygrad 中 zero_grad 的 JIT 处理`, `Tinygrad 中梯度累积的内存问题`, `Tinygrad 中 torch.no_grad() 的等效实现`, `Tinygrad 中的梯度处理与参数更新`, `改进 Tinygrad 的文档与示例`

- **JIT handling of zero_grad in Tinygrad**: 用户询问了 JIT 如何处理 Tinygrad 中的 `zero_grad`，随后明确了 JIT 在运行时使用相同的输入/输出缓冲区连接，这会强制在第二步进行不同的计算和连接。该问题在 [tensor.py at 3df47bc](https://github.com/tinygrad/tinygrad/blob/3df47bc21ee6f7006a42046902df44e2b496abac/tinygrad/tensor.py#L749) 中有记录。
   - 强调了 `zero_grad` 需要在 JIT 函数内部调用，否则从第一个宏步骤（macro step）记住的梯度缓冲区可能会在后续步骤中导致问题。
- **Memory issues with gradient accumulation in Tinygrad**: 用户分享了在 Tinygrad 梯度累积步骤中遇到 CUDA 内存溢出的困扰。建议的一种方法是将 loss 除以 `grad_accum_steps` 并以可微的方式更新梯度。
   - 一位用户建议使用 `loss.detach()` 来帮助维持计算图的持久性，而另一位用户则建议在训练步骤中使用 `param.grad.assign(0).realize()` 重新分配梯度内存。
- **Equivalent of torch.no_grad() in Tinygrad**: Tinygrad 中与 `torch.no_grad()` 等效的是 `Tensor.no_grad = True` 或使用 `@Tensor.inference_mode()` 装饰器。这些方法的示例可以在[此处](https://github.com/tinygrad/tinygrad/blob/master/examples/mlperf/model_train.py)找到。
   - 关于算子限制的查询指出，`a -= lr * a.grad` 会触发 assert，而 `a = a - lr * a.grad` 则可以正常工作，这表明 Tinygrad 对原地操作（in-place operations）的处理仍有改进空间。
- **Gradient handling and parameter updates in Tinygrad**: 用户讨论了 `zero_grad` 仅在开始时隐式地将梯度初始化为零，而不会在每一步显式地将其清零。有人指出，如果没有 `zero_grad`，梯度将会累积并可能导致问题。
   - 一位用户指出需要在训练步骤中对梯度参数调用 `realize()`，以避免持久性内存问题，这展示了 Tinygrad 梯度处理系统的复杂性。
- **Improving documentation and examples for Tinygrad**: 有人呼吁改进 Tinygrad 关于 TinyJit 和梯度累积等复杂主题的文档，因为用户在这些方面遇到了困难和不一致。建议为这些高级主题创建官方示例以帮助理解。
   - 提议包括专门为 TinyJit 添加文档页面以及更详细的示例，以更好地支持用户使用这些功能。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://www.catb.org/~esr/faqs/smart-questions.html">提问的智慧 (How To Ask Questions The Smart Way)</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py#L228">tinygrad/tinygrad/tensor.py at master · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/3df47bc21ee6f7006a42046902df44e2b496abac/tinygrad/tensor.py#L749">tinygrad/tinygrad/tensor.py at 3df47bc · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad
</li>
</ul>

</div>

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1257557522001432636)** (21 messages🔥): 

> `最佳 RAG 策略：HydeRetrieval vs. MultiQueryRetrieval`，`LangChain 用于确认消息的用法`，`Edge 中 Copilot 的图片存储`，`Embedding 数据库的分片`

- **检索任务的最佳 RAG 策略**：一位用户询问哪种 RAG 策略更好：**HydeRetrieval** 还是 **MultiQueryRetrieval**。
   - 社区成员讨论了潜在问题，其中一位用户在代码中使用 **MultiQueryRetrieval** 时遇到了结果为空的情况。
- **如何处理 LangChain 中的消息偏离**：有人提问如何使用 **LangChain** 确认输入的偏离消息并继续执行下一条 AI 消息。
   - 提供了一个示例代码片段，演示了如何使用 **invoke** 方法有效处理这种情况。
- **关于 Edge Copilot 中图片内容存储的担忧**：一位用户询问了在 Edge Copilot 中上传图片的存储位置和时长。
   - 讨论暗示了对隐私和数据处理的担忧，但未提供具体答案。
- **Embedding 数据库中的分片 (Sharding)**：一位用户询问是否有人在为 embedding 数据库实现分片。
   - 回复中包括一位使用 serverless MongoDB Atlas 的用户，但对方请求了关于如何根据查询访问特定分片的详细信息。

**提到的链接**：<a href="https://github.com/langchain-ai/langchain/issues/9195>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 做出贡献。

  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1257571468473008168)** (3 messages): 

> `在 LangServe 中使用 FastAPI-Users`，`在 LangChain 项目中允许文件上传`，`调试 CSV playground 的输出显示问题`

- **在 LangServe 中实现 FastAPI-Users**：一位成员询问是否有人将 **langserve** 与 **fastapi-users** 结合使用，以实现针对每个用户的逻辑并保护某些端点。
   - *该查询未收到直接回答或建议。*
- **在 askyourcsv 项目中启用 CSV 文件上传**：一位成员需要帮助，希望允许用户直接上传 **CSV** 文件，而不是在 **askyourcsv** 项目中提供文件路径。他们目前的实现仅允许在代码中设置文件路径，不支持上传。
   - 他们分享了 [代码片段](https://link.url) 并询问如何修改其 **FastAPI** 和 **LangChain** 设置以接受文件上传。
- **CSV Playground 输出未显示**：一位成员报告称，尽管正确设置了 **FastAPI** 应用和链，但 **csv/playground/** 中没有显示任何输出。
   - 他们还请求了调试建议，并分享了其 **FastAPI** 服务器和链实现的代码片段。
  

---


### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1257575317909016618)** (12 messages🔥): 

> `使用 LangChain 和 LangChain Expression Language 创建聊天机器人`，`添加动作能力，如预约演示和连接用户到销售团队`，`在 LangChain 中定义聊天机器人的动作`，`在 LangChain 中创建和使用 Agents`，`使用 LangSmith 调试和追踪应用`，`为 LangChain 聊天机器人能力提供 Python 代码`

- **使用 LangChain 基础构建聊天机器人**：一位成员使用 **LangChain** 和 **LangChain Expression Language** 创建了一个聊天机器人，并寻求关于添加动作功能（如预约演示和将用户连接到销售团队）的指导。
   - 另一位成员概述了在 LangChain 中使用 **Agents** 的方法，建议了定义动作和使用 **AgentExecutor** 等步骤，并指出使用 **LangSmith** 进行调试的重要性。
- **LangChain 动作的 Python 指导**：一位成员请求 Python 代码，以便为其 LangChain 聊天机器人添加执行动作的能力，特别是执行预约演示和基于用户的动作等任务。
   - 提供的回复包括定义动作、使用 **from_llm_and_tools** 创建 Agent、使用 **AgentExecutor** 的步骤，并指向了详细指导的教程，强调要根据具体用例进行调整。
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1257753796034166967)** (4 messages): 

> `使用 LLM 构建聊天机器人`, `RAFT 流水线开发`, `RAFT 与传统 RAG 的比较`, `OpenAI 的 CriticGPT`, `经验分享与合作邀约`

- **LLM 聊天机器人合作**：来自德里的 **Prince** 提议合作构建使用 **Large Language Models (LLM)** 的聊天机器人。感兴趣的人士可以通过 WhatsApp [点击此处](https://wa.me/917827250299) 联系他。
   - Prince 分享了他在该领域的丰富经验，并对新项目持开放态度，邀请用户联系进行合作。
- **实现 RAFT 流水线**：一位成员询问了 **RAFT 流水线** 的开发情况，并分享了论文链接 [此处](https://arxiv.org/abs/2403.10131) 以供进一步阅读。该论文讨论了 **Retrieval Augmented FineTuning (RAFT)**，这是一种通过忽略无关文档来提高 LLM 响应质量的方法。
   - 此次咨询旨在收集关于 RAFT 与传统 **RAG** 方法相比在性能方面的见解和反馈。
- **OpenAI CriticGPT 视频评论**：一位用户引用了一个讨论 OpenAI 新推出的 **CriticGPT** 细节的 **YouTube 视频** [此处](https://youtu.be/4PgcaIfwLjo)，强调其目的是识别并纠正 GPT-4 输出中的错误。该视频回顾了 **CriticGPT** 论文，并研究了其提高代码生成准确性的方法。
   - *尝试捕捉了 OpenAI 关于 CriticGPT 最新论文在 YouTube 上解释的主要细节和方法。非常期待听到您的反馈。*

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.10131">RAFT: Adapting Language Model to Domain Specific RAG</a>：在大型文本语料库上预训练 Large Language Models (LLMs) 现已成为标准范式。在将这些 LLMs 用于许多下游应用时，通常会额外注入新的知识...</li><li><a href="https://youtu.be/4PgcaIfwLjo">OpenAI releases CriticGPT to correct GPT-4&#39;s mistakes | Read the paper with me</a>：OpenAI 推出了 CriticGPT，这是一款基于 GPT-4 的新 AI 模型，旨在识别 ChatGPT 生成的代码中的错误，标志着向改进...迈出了重要一步。</li><li><a href="https://wa.me/917827250299">Prince</a>：商业账号
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1257413158230954045)** (36 messages🔥): 

> `Runway Gen 3 发布`, `Sonnet + Artifacts 的使用`, `Anthropic 关于优秀评估 (evals) 的原则`, `Chatbot Arena 排名问题`, `高盛关于 AI 投资的报告`, `Figma AI 功能引发的担忧`, `最佳 TTS 模型`, `Claude 3.5 Sonnet 更新`, `微软更新的 Phi-3 Mini`, `Magic Dev 估值上涨`

- **Runway Gen 3 以短片惊艳亮相**：Runway Gen-3 Alpha Text to Video 已发布并向所有人开放，主打**高保真、快速且可控的视频生成**。[立即体验](https://runwayml.com) 或在此查看 [发布公告](https://x.com/runwayml/status/1807822396415467686)。
   - *@amebagpt 对 Runway Gen-3 和 SORA 的对比* 强调了目前只有 Runway Gen-3 可供使用。[查看对比](https://x.com/altryne/status/1807868306361094153)。
- **Sonnet + Artifacts 令人震撼**：**Sonnet 和 Artifacts** 的组合因其在流程可视化和图表操作上的快速迭代能力而广受好评。用户分享了他们的**高效体验**，并提到可能需要专门的 AIA 环节。
   - 一位用户描述了他们的体验：*让我能以思考的速度迭代我想要完善/探索的流程视觉描述*，且无需手动调整节点。
- **对 Figma AI 训练数据的担忧**：Figma 为其 **'Make Design' 功能** 辩护，澄清其并未在 Figma 内容、社区文件或应用设计上进行训练，反驳了关于数据滥用的指控。在此阅读 [官方说明](https://x.com/zoink/status/1808045655082033483)。
   - 尽管 Figma 进行了澄清，但社区中仍存在担忧，因为生成的模型与现有应用（尤其是 **Apple 的天气应用**）高度相似。
- **Microsoft 提升 Phi-3 Mini 能力**：Microsoft 发布了 Phi-3 Mini 的更新，提升了多种语言的**代码理解**能力、多轮指令遵循以及长上下文理解，使其功能更加强大。详情请见 [此处](https://x.com/reach_vb/status/1808056108319179012)。
   - 这些更新已应用于 4K 和 128K 上下文模型 Checkpoints，显著增强了**结构化输出和推理能力**。
- **Magic Dev 估值达到 15 亿美元**：AI 编程初创公司 Magic Dev 在最新一轮融资中寻求 **15 亿美元估值**，尽管该公司目前尚无产品或收入，这引起了业界的关注。详情请见 [Reuters](https://www.reuters.com/technology/artificial-intelligence/ai-coding-startup-magic-seeks-15-billion-valuation-new-funding-round-sources-say-2024-07-02/)。
   - 这家仅有 20 名员工的初创公司暗示了当前技术投资领域的投机性质，若非有知名支持者，可能会引发关于市场泡沫的评论。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.goldmansachs.com/intelligence/pages/gen-ai-too-much-spend-too-little-benefit.html">Gen AI：投入过多，收益太少？</a>：未找到描述</li><li><a href="https://aider.chat/2024/07/01/sonnet-not-lazy.html">Sonnet 与“懒惰”相反</a>：Claude 3.5 Sonnet 可以轻松编写出超过单次 4k token API 响应容量的高质量代码。</li><li><a href="https://x.com/zoink/status/1808045655082033483">来自 Dylan Field (@zoink) 的推文</a>：(1) 正如我们上周在 Config 大会上，以及在我们的博客、网站和许多其他渠道所分享的那样——Make Design 功能并非基于 Figma 内容、社区文件或应用设计进行训练。换句话说...</li><li><a href="https://x.com/altryne/status/1807868306361094153">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：由优秀的 @amebagpt 制作的 @runwayml Gen-3 与 SORA 的对比 👏 你可以并排观察并做出判断，同时请记住，目前只有其中一个是可以实际使用的...</li><li><a href="https://x.com/eugeneyan/status/1807912280874705032?s=46">来自 Eugene Yan (@eugeneyan) 的推文</a>：你对 evals 痴迷吗？！加入 Anthropic 致力于：• AI 安全等级评估 • 高级能力与安全指标 • 构建 evals 的基础设施、工具和方法 https://www.anthropic.co...</li><li><a href="https://en.wikipedia.org/wiki/Flowers_for_Algernon">《献给阿尔吉侬的花束》 - 维基百科</a>：未找到描述</li><li><a href="https://x.com/reach_vb/status/1808056108319179012">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：Microsoft 悄悄更新了 Phi-3 Mini！🚀 &gt; 显著提升了对 Python, C++, Rust 和 Typescript 的代码理解能力。 &gt; 增强了 post-training 以获得更好的结构化输出。 &gt; 改进了...</li><li><a href="https://x.com/RealJosephus/status/1807840624852586938">来自 Joseph (@RealJosephus) 的推文</a>：在仔细审查 lmsys 数据后，我得出结论：模型身份（名称）和开场白都是影响评分的关键因素。</li><li><a href="https://x.com/runwayml/status/1807822396415467686">来自 Runway (@runwayml) 的推文</a>：Gen-3 Alpha 文本生成视频（Text to Video）现已向所有人开放。高保真、快速且可控视频生成的新前沿。立即在 http://runwayml.com 体验。</li><li><a href="https://github.com/Vaibhavs10/open-tts-tracker">GitHub - Vaibhavs10/open-tts-tracker</a>：通过在 GitHub 上创建账号，为 Vaibhavs10/open-tts-tracker 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1257422972554182799)** (35 条消息🔥): 

> `运行 llama.cpp 的硬件要求`, `llamafile v0.8.9 发布`, `测试 mxbai-embed-large-v1 模型`, `选择运行大语言模型的硬件`, `AI 模型训练与推理中的 CPU vs GPU`

- **关于 Llamafile 最低硬件要求的讨论**：针对运行 **llama.cpp** 最低硬件要求的咨询引发了关于 **iPhone 13** 或 **Raspberry Pi Zero W** 是否能够胜任的讨论。澄清表明需要 64 位系统，且具体的模型和内存要求起着至关重要的作用。
   - 社区成员确认 **Raspberry Pi Zero** 由于 64 位系统要求而不适用，并指出模型的大小决定了所需的内存，某些模型可以在极小的 RAM 上运行。
- **Llamafile v0.8.9 发布，支持 Android**：最新的 [llamafile v0.8.9 发布](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.9) 确认了对 Android 的支持，并改进了 **Gemma2** 模型与 Google 预期用法的一致性。新的 embedding 服务器也取得了更多进展。
   - 社区对这些改进表示赞赏，提到了更好的 **Gemma2 支持** 以及与 Google 托管版本高度一致的潜力，后者在公开评估中表现优于几个更大的模型。
- **修复 mxbai-embed-large-v1 模型问题**：用户遇到了 **mxbai-embed-large-v1** 模型的一个问题，即所有文本输入都返回相同的向量。将输入键从 **'text'** 更改为 **'content'** 解决了该问题。
   - 一位用户请求 [更新 Hugging Face 仓库](https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile) 以反映此修复，从而避免误导未来的用户。
- **为 llamafile 选择最佳硬件**：爱好者们讨论了运行大语言模型的最佳硬件配置，平衡了 **VRAM** 和 **CPU memory**。对于消费级用途，建议范围从 **3090/4090 GPU** 到高端工作站解决方案如 **A6000/RTX 6000**。
   - 用户分享了经验，指出了 **高 VRAM** GPU 的实用性，并探索了 **消费级平台** 在内存容量方面对于有效模型性能的极限。
- **探索 CPU 在模型训练中的潜力**：关于 CPU 与 GPU 能力的讨论强调了由于计算需求，GPU 在模型训练方面相对于 CPU 拥有的**护城河**。其中一个例子提到，尽管有多个核心，在 CPU 上运行大型模型仍然可能很**慢**。
   - **基于 CPU 的学习** 可能性受到了质疑，并建议尝试使用 **llm.c** 训练 GPT-2 模型，这突显了 CPU 在大规模训练场景中的实际限制。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile">Mozilla/mxbai-embed-large-v1-llamafile · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=iaJqMFTBgGw">Android 版 llamafile 教程</a>: llamafile github - https://github.com/Mozilla-Ocho/llamafile (右侧发布页) tinyllama - https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-...</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.9">发布 llamafile v0.8.9 · Mozilla-Ocho/llamafile</a>: 此版本使 Gemma2 的运行更接近 Google 的预期。af22695 使 gemma2-27b-it 与 aistudio.google.com 一致；41678c8 为 Gemma2 添加滑动窗口掩码；140eed5 为 Ge... 添加 soft-capping...</li><li><a href="https://huggingface.co/jartine/gemma-2-27b-it-llamafile">jartine/gemma-2-27b-it-llamafile · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/jartine/gemma-2-9b-it-llamafile">jartine/gemma-2-9b-it-llamafile · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#memorydisk-requirements">GitHub - ggerganov/llama.cpp: C/C++ 中的 LLM 推理</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1257446144653398128)** (25 messages🔥): 

> `Building 01 on Windows`, `Docker capabilities with OI`, `Concurrency and resource isolation in OI deployments`, `Handling image displays in OI`, `Local AI agents for web browsing`

- **在 Windows 上构建 01 仍具挑战**：**piranha__** 请求一份在 **Windows** 上构建 01 的指南，因为现有的文档无法正常工作。**techfren** 建议使用 Discord 的搜索功能查找用户分享的步骤。
   - *techfren* 指出 **piranha__** 可能在 <#1194880263122075688> 中找到了一些有用的步骤。
- **通过 OI 将 Nmap 容器化**：**lagoonghost** 询问如何结合使用 Docker 和 OI，通过 Jan 作为界面在容器内运行 **nmap** 等工具。目前尚未提供具体的回答或指南。
   - 未提供更多细节。
- **OI 部署中的并发问题**：**chaichaikuaile_05801** 提出了将 OI 作为服务部署时的并发和资源隔离问题，考虑使用 **OI Multiple Instances** 或为每个请求重新分配上下文。**notnaton** 建议使用多个实例是最佳方案，尽管成本较高。
   - *chaichaikuaile_05801* 和 *notnaton* 讨论了使用 `.reset()` 重置可能解决代码共享问题，但在测试后确认，多个实例并不共享同一个 Python 执行环境。
- **在 OI 中显示图像**：**chaichaikuaile_05801** 询问在执行 `MatPlotLib.show()` 等代码时如何在 OI 中返回图像，并提到了 [GitHub](https://github.com/OpenInterpreter/open-interpreter/issues/1301) 上的一个未解决问题。用户注意到 0.1.18 和 0.2.5 版本之间的功能差异。
   - 此外，*chaichaikuaile_05801* 表示希望看到图像返回能力的增强。
- **寻求高效的 Local AI Agents**：**blurrybboi** 询问是否有能够高效浏览网页和 YouTube 的 Local AI Agents，而不仅仅是肤浅的搜索。他们正在寻找能够进行深度搜索并过滤掉低质量结果的 AI。
   - 关于此类 Local AI Agents 目前的能力，尚未给出回复。

**提及的链接**：<a href="https://github.com/OpenInterpreter/open-interpreter/issues/1301">When I use interpreter.chat(stream=True), in what scenarios will type return &#39;image&#39;? · Issue #1301 · OpenInterpreter/open-interpreter</a>：描述 Bug：当我使用 interpreter.chat(stream=True) 时，在什么情况下类型会返回 'image'？我在 0.1.18 版本中尝试使用时它会返回图像，但 0.2.5 版本似乎不会...

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1257435057644372008)** (2 messages): 

> `Portaudio installation issue on Windows 11`, `Pull request for updating Windows installation documentation`

- **Windows 11 上的 Portaudio 安装问题**：一位用户报告了在 **Windows 11** 上安装 **Portaudio** 时的问题，收到错误提示称找不到该软件包。
   - 他们随后发现了一个更新 Windows 安装文档的 [Pull Request](https://github.com/OpenInterpreter/01/pull/203)，这可能有助于解决该问题。
- **OpenInterpreter 的 Windows 安装指南已更新**：发现了一个旨在更新 **OpenInterpreter** 的 Windows 安装文档的 [Pull Request](https://github.com/OpenInterpreter/01/pull/203)。
   - 该更新汇总了之前用户尝试在 Windows 上安装时的经验教训，包括 Discord 上 Zorcon 的讨论。

**提及的链接**：<a href="https://github.com/OpenInterpreter/01/pull/203">Update documentation for Windows installation by dheavy · Pull Request #203 · OpenInterpreter/01</a>：问题：文档中未提供具有关键差异的 Windows 安装说明。解决方案：汇总之前用户的尝试经验（包括 Discord 上 Zorcon 的经验等...）

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1257432986928939110)** (26 条消息🔥): 

> `在 GPU 上微调模型并使用 WandB 记录日志`, `torchtune 中的训练配置和评估方法`, `在 AI 任务中使用 AMD GPU 的挑战`, `针对小数据集不经过 SFT 直接进行 DPO 的指导`, `将 torchtune 模型转换为 HuggingFace 格式`

- **在社区帮助下成功完成首次模型微调**：一位用户分享了他们在社区支持下完成首个模型微调的成就，并提到如果没有付费购买在线 GPU，他们无法在自己的机器上完成。
   - 该用户讨论了学习日志文件的复杂性与使用 WandB 等工具以获得更好洞察力之间的区别，并计划在遵循此建议前后运行评估。
- **在 Torchtune 中切换到 WandB 日志记录**：一位成员建议使用 [Weights & Biases (W&B) logger](https://pytorch.org/torchtune/main/deep_dives/wandb_logging.html) 代替日志文件，以便在 torchtune 训练中获得更好的洞察。
   - 在讨论了 YAML 配置问题后，建议删除冗余的日志组件以避免错误；并分享了一个简化的 W&B 配置。
- **AMD GPU 用于 AI 任务的挑战**：一位用户分享了他们使用 AMD GPU 进行 AI 工作的经验以及一份 [Reddit 指南](https://www.reddit.com/r/LocalLLaMA/s/VRNQWhh2fh)，尽管在 ROCm 和 torchtune 上取得了一些成功，但仍推荐使用 NVIDIA 以获得更好的支持。
   - *“我从未想过能在我的 6900 XT 上微调任何东西”*，详细描述了遇到的困难以及在 torchtune Discord 中获得的社区帮助。
- **不经过 SFT 的 DPO 训练建议**：一位成员建议参考 torchtune 的 DPO recipe，直接在小数据集上尝试 DPO 而不经过 SFT。
   - 另一位成员提供了关于数据集格式的进一步支持，并强调在用户指定的场景下，DPO 可能在没有 SFT 的情况下也能表现良好。
- **将 Torchtune 模型转换为 HuggingFace**：一位用户询问了如何将 torchtune 模型转换为 HuggingFace 格式，并寻求相应的层名称（layer names）。
   - 虽然没有提供太多关于转换过程的细节，但这突显了用户在 AI 训练过程中工具集成的进展和成果。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/torchtune/main/deep_dives/wandb_logging.html">Logging to Weights &amp; Biases &mdash; torchtune 主文档</a>: 未找到描述</li><li><a href="https://wandb.ai/lmn07r/torchtune/workspace?nw=nwuserlemon07r">lmn07r</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html#generation)">使用 torchtune 的端到端工作流 &mdash; TorchTune 文档</a>: 未找到描述</li><li><a href="https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html#run-evaluation-using-eleutherai-s-eval-harness)">使用 torchtune 的端到端工作流 &mdash; TorchTune 文档</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/VRNQWhh2fh">Reddit - 深入探索</a>: 未找到描述
</li>
</ul>

</div>

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1257517558375120947)** (17 条消息🔥): 

> `工具包的多步执行能力`, `旧金山 AI Engineer 大会的活动与会议`, `Sandra Kublik 关于 Command R 模型的会议`, `LLM-UNIVERSITY 频道的停用与支持`, `使用 Cohere 的 API 和资源进行开发`

- **工具包支持多步执行能力**：@meor.amer 确认工具包已经启用了多步执行能力，并分享了一个示例来演示该功能。
   - @sssandra 补充说，前端和后端都已提供对此功能的支持。 
- **Sandra Kublik 赞扬 AI Engineer 活动**：@sssandra 分享了她在旧金山参加 AI Engineer 大会的经历，对活动和派对表示赞赏。她主持了一场关于 Command R 模型的会议，链接见[此处](https://twitter.com/itsSandraKublik/status/1807106578979639712)。
   - 她提到会议中包含了服务器机器人。其他成员也认为这是一个值得关注的活动。
- **LLM-UNIVERSITY 频道已停用**：@.overwhelmed 询问关于 LLM-UNIVERSITY 及其集成模型的示例。
   - @xvarunx 澄清该频道已经停用，但可以通过其他频道和像 @meor 这样的讲师获得支持。此外，Cohere 的[文档和 API 参考](https://docs.cohere.com/reference/about)提供了丰富的资源。
- **开始使用 Cohere API 的资源**：@xvarunx 分享了各种资源，如 [cookbooks](https://docs.cohere.com/docs/cookbooks) 和 GitHub [示例](https://github.com/cohere-ai/notebooks)。
   - 他鼓励用户分享他们的项目以获得更多支持和反馈。@.overwhelmed 对社区的积极支持表示感谢。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/reference/chat">Chat</a>: 未找到描述</li><li><a href="https://docs.cohere.com/">Cohere Enterprise Group</a>: 未找到描述</li><li><a href="https://docs.cohere.com/docs/cookbooks">Cookbooks</a>: 未找到描述</li><li><a href="https://github.com/cohere-ai/notebooks">GitHub - cohere-ai/notebooks: Cohere 平台的代码示例和 Jupyter Notebooks</a>: Cohere 平台的代码示例和 Jupyter Notebooks - cohere-ai/notebooks
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1257811755141173379)** (3 条消息): 

> `Cohere Slack 机器人创建`, `Slack 机器人的性能要求`, `模型处理速度`

- **Cohere Slack 机器人简化工作区访问**：一位用户表示，“我创建了一个在我的工作区内运行的 Cohere Slack 机器人”，强调了该集成对团队的便利性。
   - 另一位成员回复了一个兴奋的“火”表情符号，对这种快速高效的设置表示强烈赞同。
- **Slack 机器人要求快速响应**：一位用户提到 Slack 要求机器人必须在 *3 秒内完成请求*，这凸显了快速模型性能的必要性。
   - 这展示了“速度”对于维持无缝交互至关重要，强调了高性能模型的重要性。
  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1257725407629938750)** (1 条消息): 

> `Cohere For AI 伦敦活动`, `Expedition Aya 倡议`, `Expedition Aya 的收益与活动`, `多语言 AI 研究`, `Crew Connections 会议`

- **Cohere For AI 宣布伦敦活动**：Cohere For AI 将于 **7 月 10 日**晚在伦敦举办活动，通过闪电演讲、美食和饮品来庆祝和讨论**多语言 AI**。此次活动将启动他们最新的倡议 [Expedition Aya](https://sites.google.com/cohere.com/expedition-aya/home)，这是一项为期 6 周的全球挑战。
   - 与会者可以与 ML 研究人员建立联系，获得独家资源和 API 额度，并获得启动多语言研究想法的支持。优秀项目将获得特别的 **swag**（周边礼品）和**独家奖品**。
- **Expedition Aya：全球开放构建挑战赛**：Expedition Aya 是 Cohere For AI 发起的一项为期 **6 周的全球性倡议**，专注于**多语言 AI 模型** Aya 23 和 Aya 101。参赛团队可以获得资源、API 额度以及启动研究想法的支持。
   - 成功的团队将有资格获得**限量版 swag** 和针对顶级项目的**独家奖品**。该倡议旨在连接全球研究人员，并推动多语言 AI 的边界。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tinyurl.com/c4ai-london">Form</a>: 未找到描述</li><li><a href="https://sites.google.com/cohere.com/expedition-aya/home">Expedition Aya</a>:   
</li>
</ul>

</div>
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1257671332250914909)** (2 条消息): 

> `提醒升级 openssh 软件包`, `Figure 1 使用像素到动作神经网络执行完整的端到端 BMW 案例`

- **提醒：升级你的 OpenSSH 软件包**：一位成员提醒所有人，如果运行任何连接到互联网的服务器，请**升级 OpenSSH 软件包**。
   - 消息中未提供更多细节或链接。
- **Figure 1 实现了精确的 BMW 使用案例**：[Corey Lynch](https://x.com/coreylynch/status/1807816113129955696?t=Ayn8KUIcP8LXn893GH_nPw&s=19) 宣布 **Figure 1** 正在执行完整的端到端 **BMW 使用案例**，所有操作均通过 200hz 的像素到动作（pixel-to-action）神经网络学习。
   - *学习到的行为需要极其精确（小于 1 厘米的钣金插入），并且能在长时程（long horizon）内工作。*

**提到的链接**：<a href="https://x.com/coreylynch/status/1807816113129955696?t=Ayn8KUIcP8LXn893GH_nPw&s=19">Corey Lynch (@coreylynch) 的推文</a>：Figure 1 正在执行完整的端到端 BMW 使用案例，所有操作均通过 200hz 的像素到动作神经网络学习。学习到的行为需要极其精确（小于 1 厘米的钣金插入）...

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1257423485710635101)** (15 条消息🔥): 

> `ML 模型评估的复杂性`, `新型 LLM：phi-CTNL`, `AIW+ 问题正确解法的验证`, `解题假设的澄清`, `介绍新模型架构：Terminator`

- **评估微调模型非常痛苦**：在[我最近的文章](https://mlops.systems/posts/2024-07-01-full-finetuned-model-evaluation.html)中，一位用户概述了评估用于从新闻稿中提取结构化数据的微调 LLM 的复杂性和痛点，强调了核心指标（准确率）和其他评估指标。
   - 他们强调了隐藏代码和过程缓慢的问题，并指出如果没有合适的系统，维护评估的复杂性会不断累积。
- **phi-CTNL 优于所有已知模型**：最近一篇论文的[摘要](https://arxiv.org/abs/2309.08632)介绍了一个拥有 100 万参数的基于 Transformer 的 LLM —— phi-CTNL，它在一种新型的、经过策划的数据混合物上进行了预训练。
   - 该模型在各种学术基准测试中取得了完美的结果，并展现出一种前所未有的、类似于 grokking 的预测评估基准测试 canary（金丝雀数据）的能力。
- **AIW+ 正确解法已验证**：一位用户表示，AIW+ 问题的正确解法已被他人验证，并得到了更强大模型的支持。
   - 他们建议使用所有关系的图表来进行正式的检查验证，并提供了来自 Claude 3 Opus 的罕见正确响应示例，验证了答案。
- **澄清题目陈述中的假设**：用户讨论了题目陈述中的假设，特别是关于 Alice 的兄弟姐妹和表亲数量的 AIW+ 问题。
   - 一位用户质疑了得出结论的推理过程，认为 Alice 可能有兄弟，从而影响问题的解法。
- **新模型架构：Terminator**：@LeopolisDream 介绍了一种名为“Terminator”的新架构，其特点是没有残差（residuals）、没有点积注意力（dot product attention）且没有归一化（normalization）。
   - 他们提供了[论文链接](https://arxiv.org/pdf/2401.17948)以获取该架构的更多细节。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2309.08632">Pretraining on the Test Set Is All You Need</a>：受到最近展示了在精心策划的数据上预训练的小型 Transformer 语言模型前景的工作启发，我们通过投入大量精力策划...</li><li><a href="https://github.com/LAION-AI/AIW/blob/main/collected_responses/AIW_AIW_plus.json">AIW/collected_responses/AIW_AIW_plus.json at main · LAION-AI/AIW</a>：用于实验的爱丽丝梦游仙境代码库和原始实验数据 - LAION-AI/AIW</li><li><a href="https://x.com/leopolisdream/status/1804627325583327358?s=46&t=BsqYoGA8vIHGcXwORlMk7w">Alex Yanko 🇺🇦 (@LeopolisDream) 的推文</a>：欢迎新架构：Terminator。没有残差，没有点积注意力，没有归一化... https://arxiv.org/pdf/2401.17948</li><li><a href="https://mlops.systems/posts/2024-07-01-full-finetuned-model-evaluation.html">Alex Strick van Linschoten - 我的微调模型击败了 OpenAI 的 GPT-4</a>：对于我的测试数据，Mistral、Llama3 和 Solar LLM 的微调版本比 OpenAI 的模型更准确。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1257417612082675724)** (7 messages): 

> `Knowledge graphs` 和 `Lang Graph` 在 AI 项目中的应用, `Voice detection` 和转录模型, 最近演讲的录音, 构建 `LLMs` 一年来的经验教训

- **使用 Chainlit Cookbook 简化音频处理**：一位成员讨论了使用 **SileroVAD** 实现 **voice detection model** 以及使用 **whisper-fast** 进行转录，并通过少量的预处理和快速的 **TTS-API** 对其进行了增强。他们推荐了一个 [Chainlit cookbook 示例](https://github.com/Chainlit/cookbook/tree/main/audio-assistant) 用于快速测试。
   - 社区还提到了使用 **elevenlabs (turbo)**、**playht** 或 **deepgram** 作为 **TTS** 解决方案，从而简化了高效音频处理的工作流程。
- **关于 Knowledge Graphs 的咨询**：一位社区成员询问另一位成员在 AI 项目中关于 **knowledge graphs** 和 **Lang Graph** 的工作情况。该问题旨在深入探讨之前分享的经验。
   - 这引起了其他成员的兴趣，反映了社区对利用 **graph technologies** 进行高级 AI 应用的关注。
- **近期演讲录音的可用性**：当被问及**最近演讲**的录音是否可用时，一位成员确认 **<@916924724003627018>** 负责上传录音。
   - 讨论明确了“最后一次演讲”是指 **

**提及的链接**：<a href="https://github.com/Chainlit/cookbook/tree/main/audio-assistant">cookbook/audio-assistant at main · Chainlit/cookbook</a>：Chainlit 的 cookbook 仓库。欢迎通过在 GitHub 上创建账号来为 Chainlit/cookbook 的开发做出贡献。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1257751477909455000)** (1 messages): 

> 处理来自 `Kaggle` 竞赛的大型数据集, 使用 `Dask` 进行数据处理, `Dask` 中的 `Out of Memory (OOM)` 错误, 在 `Modal` 上执行 `Dask` 任务

- **Dask 在 USPTO Kaggle 数据集规模上遇到困难**：一位社区成员描述了处理来自 **Kaggle 上的 USPTO 竞赛**的大型数据集的努力，但在使用 **Dask** 合并数据时遇到了 **OOM errors**。
   - 他寻求关于在 **Modal** 上执行 **Dask jobs** 的建议，引发了关于处理大型数据集最佳实践的讨论。
- **Modal 能否实现无缝的 Dask 执行？**：用户询问是否有人成功在 **Modal** 上执行过 **Dask jobs** 以缓解内存问题。
   - 这一询问凸显了 **Modal** 在更高效处理计算任务方面的潜力，并促使其他人分享他们的经验或替代方案。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[paige_when_finetune](https://discord.com/channels/1238365980128706560/1242224662142779530/)** (1 messages): 

shamik_53759: 是的，现在已经上线了。谢谢！
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1257475202271416412)** (2 messages): 

> `Autotrainer` 建议, 关于 `Autotrainer` 的澄清

- **提出 Autotrainer 建议**：一位成员建议尝试 **autotrainer**，并提到：*'也许可以试试 autotrainer？'*。
   - 随后出现了关于 **autotrainer** 是 **axolotl** 的功能还是与 **Huggingface autotrain** 相关的询问，凸显了参与者之间的一些不确定性。
- **寻求关于 Autotrainer 的澄清**：另一位成员要求澄清 **autotrainer**，询问它是 **axolotl** 的功能还是 **Huggingface autotrain** 的一部分。
   - 一则回复表示对该术语不熟悉，在查询后建议它可能与 **Huggingface autotrain** 有关。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1257701018020614154)** (1 messages): 

> `OpenAI` 额度消耗反馈

- **对 OpenAI 定价的负罪感**：一位成员表示 **OpenAI's pricing** 非常实惠，以至于他们因为无法在接下来的 **3 个月**内用完 **$500 credits** 而感到愧疚。
   - 他们用一个*倒脸表情 (🙃)* 幽默地表达了这种情绪，强调了 OpenAI 产品的感知价值。
- **实惠的 AI 服务引发复杂情绪**：**OpenAI's credits** 的实惠性导致一些用户因为无法在过期前充分利用分配的 **$500 credits** 而产生负罪感。
   - 一位用户用倒脸表情对这种情况进行了幽默的评论，传达了对这一困境的轻松看法。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[bergum_rag](https://discord.com/channels/1238365980128706560/1252713659243827251/1257422905072025703)** (4 messages): 

> `视频中幻灯片的位置`, `Jo 的幻灯片请求`

- **关于幻灯片位置的困惑**：*Remi1054* 询问视频中提到的幻灯片是否已经分享，并表示他们可能错过了其发布位置。
   - 另一位成员 **jt37** 澄清说，幻灯片通常发布在 Maven 上，但也承认那里目前没有。
- **Jo 的幻灯片查询**：**hamelh** 提到并非所有人都会分享他们的幻灯片，但他已经向 **Jo** 索要了其幻灯片，这提供了进一步的说明。
   - 这似乎是一个特定的案例，必须直接联系 **Jo** 才能获取他的幻灯片。
  

---



### **AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/)** (1 messages): 

mikhail_ee: 来自 https://Hexagen.World 的一些新位置
  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1257787172396859494)** (5 messages): 

> `Docker 端口请求`, `建议提交 Docker 端口的 PR`, `分享了在 Windows 上使用 WSL 设置 AI Town 的 GitHub 页面`

- **AI Town 的 Docker 端口请求**：一位成员提到，如果 AI Town 能有一个 **Docker 端口** 就太棒了。他们表达了希望实现这一功能的愿望。
   - *“听起来如果能放在 README 里会很棒。你能提交一个 PR 吗？”* 另一位成员回应道，鼓励提交 PR。
- **在 Windows 上使用 WSL 设置 AI Town 的 GitHub 页面**：一位成员分享了一个 [GitHub 页面](https://github.com/Ikkitsuna/AI-Town-Windows-Setup-WSL-method)，详细介绍了如何**在 Windows 上使用 WSL 设置 AI Town**。该页面为设置开发环境提供了全面的指南。
   - 一位成员强调有必要将此指南包含在主仓库中，并重申了对 AI Town 的 Docker 镜像的长期需求。

**提到的链接**：<a href="https://github.com/Ikkitsuna/AI-Town-Windows-Setup-WSL-method">GitHub - Ikkitsuna/AI-Town-Windows-Setup-WSL-method: Guide for setting up AI Town on Windows using WSL</a>：在 Windows 上使用 WSL 设置 AI Town 的指南。通过在 GitHub 上创建账户为 Ikkitsuna/AI-Town-Windows-Setup-WSL-method 的开发做出贡献。

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1257821853884616804)** (5 messages): 

> `Apple 在 OpenAI 的观察员席位`, `Phil Schiller 作为观察员加入 OpenAI 董事会`, `Microsoft 的投资 vs. Apple 的曝光度交易`, `对 Apple 与 OpenAI 交易的反应`, `Apple 与 Microsoft 与 OpenAI 交易的对比`

- **Apple 获得 OpenAI 董事会观察员席位**：作为 Apple Intelligence 合作伙伴关系的一部分，Apple 将于今年晚些时候在 **OpenAI** 获得一个董事会观察员席位。App Store 负责人、前营销主管 **Phil Schiller** 将出任该席位 [来源](https://www.bloomberg.com/news/articles/2024-07-02/apple-to-get-openai-board-observer-role-as-part-of-ai-agreement)。
   - 社区幽默地指出，虽然 **Microsoft** 向 OpenAI 投资了数十亿美元，但 Apple 仅凭极少的投资就获得了显著优势，凸显了 **Tim Cook** 的天才之处。
- **Microsoft vs. Apple 的 OpenAI 交易**：讨论围绕 Microsoft 对 OpenAI 的十亿美元投资与 Apple 财务投入较少但同样有效的交易之间的差异展开。Apple 的合作催生了一款出色的应用和深度的 iPhone 集成，引发了幽默的对比 [来源](https://x.com/BartokGabi17/status/1808242102750568799)。
   - 成员们评论说 **Microsoft 在这笔交易中被欺负了**，对这种情况表示觉得很有趣。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/markgurman/status/1808240961522159862">Mark Gurman (@markgurman) 的推文</a>：最新消息：作为 Apple Intelligence 合作伙伴关系的一部分，Apple 将于今年晚些时候在 OpenAI 获得一个董事会观察员席位。获得该席位的人选是：App Store 负责人、前营销主管 Phil Schiller...</li><li><a href="https://x.com/BartokGabi17/status/1808242102750568799">Bartok Gabriel (@BartokGabi17) 的推文</a>：@markgurman Microsoft 向 OpenAI 投资了数十亿却没得到一个应用，Apple 简直是用“曝光度”来支付，OpenAI 却做出了一个很棒的应用并进行了深度的 iPhone 集成。利润？？Tim Apple 真是个天才。
</li>
</ul>

</div>
  

---

### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1257723447866560522)** (2 条消息): 

> `关于移动电话互联网浏览演变的讨论。`, `美国联邦官员收受外国礼物。`, `早期移动互联网服务的历史背景。`, `作为数据的外国礼物及其记录问题。`

- **大多数 AI 产品正在犯一个熟悉的错误**：[dbreunig.com](https://www.dbreunig.com/2024/07/01/be-better-not-smaller.html) 上的一篇文章讨论了早期的移动互联网服务（如 **WAP**）是如何受到限制且未被充分利用的，并将其与当前的 AI 产品错误进行了类比。作者描述了 iPhone 时代之前为了让更多用户采用而优化移动浏览的尝试。
   - 文章强调，现代 AI 产品应该专注于改善用户体验，而不仅仅是适应更小的平台。*“内容就像是通过钥匙孔窥视互联网。”*
- **美国官员收到的外国礼物分析**：一篇 [Scoop 文章](https://thescoop.org/archives/2024/06/22/all-foreign-gifts-around-us/index.html) 详细介绍了美国联邦官员经常收到来自外国政府的异常且昂贵的礼物。例子包括 **鳄鱼保险** 和 **金牌**。
   - 该文章强调了这些礼物的记录和访问方式所面临的挑战，它们通常以 PDF 等非结构化格式存储。这突显了政府数据处理中更广泛的问题。*“这些外国礼物确实就是数据。”*
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.dbreunig.com/2024/07/01/be-better-not-smaller.html">Be Better, Not Smaller</a>: 新技术只有在被用来让事物变得更好，而不仅仅是更小、更便宜时，才会真正流行起来。</li><li><a href="https://thescoop.org/archives/2024/06/22/all-foreign-gifts-around-us/index.html">Derek Willis - All Foreign Gifts Around Us</a>: 未找到描述
</li>
</ul>

</div>
  

---



---



---



---



---



{% else %}


> 为了便于邮件阅读，完整的逐频道明细已被截断。
> 
> 如果您想查看完整的明细，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}