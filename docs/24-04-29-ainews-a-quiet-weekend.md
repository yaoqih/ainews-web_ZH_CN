---
companies:
- microsoft
- coca-cola
- uber
- lmsys
- nous-research
- mistral-ai
date: '2024-04-29T22:10:15.446084Z'
description: '以下是该文本的中文翻译：


  **Yann LeCun** 预测，在未来 10 到 15 年内，人们将从智能手机转向带有 AI 助手的 **AR（增强现实）界面**。基于 **Llama-3**
  的 **Dolphin-2.9 模型**已发布，改善了之前的质量问题。**PixArt Sigma** 是一个拥有 **6 亿（0.6B）参数**的模型，其性能达到了
  **Stable Diffusion 3.0** 的水平，具备完整的提示词遵循能力，并支持本地部署。研究显示，在密集监督下，Transformer 模型可以利用无意义的**填充标记（filler
  tokens）**来完成算法任务。AI 生成的餐厅评论能够通过**图灵测试**，成功欺骗人类和 AI 检测器。**Uber** 采用图算法和学习嵌入（learned
  embeddings）来进行 **ETA（预计到达时间）**预测。**可口可乐**与**微软**宣布达成一项为期 5 年的 AI 合作伙伴关系，旨在加速云服务和生成式
  AI 项目的落地。通过 **AirLLM** 优化，**Llama-3 70B** 模型可以在无需量化的情况下在单个 4GB GPU 上运行，但运行速度较慢。**Mistral.rs**
  作为一个快速的 LLM 推理平台推出，支持量化并兼容 OpenAI API。由于面临诸多挑战（尤其是在企业环境中），仅有 **5%** 的大语言模型（LLM）能从原型阶段成功走向生产环境。针对
  Llama 模型的 **EXL2** 和 **GGUF** 量化方法在困惑度（perplexity）与模型大小的关系上表现相似；与全精度相比，Llama-3 和
  Llama-2 在量化后的性能下降比以往更加明显。'
id: d5c69936-e1cf-458c-860d-c5af248bd0cf
models:
- llama-3
- dolphin-2.9
- pixart-sigma
- llama-3-70b
original_slug: ainews-a-quiet-weekend
people:
- yann-lecun
title: 一个安静的周末
topics:
- ar-interfaces
- transformers
- algorithmic-tasks
- turing-test
- graph-algorithms
- embeddings
- generative-ai
- model-optimization
- llm-inference
- quantization
- model-deployment
---

<!-- buttondown-editor-mode: plaintext -->> 2024/4/26-4/29 AI 新闻。我们为您检查了 7 个 subreddit、[**373** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord（**416** 个频道和 **10824** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**1197 分钟**。

关于 [SB-1047](https://www.reddit.com/r/LocalLLaMA/comments/1cfizbb/california_sb1047_seems_like_it_could_impact_open/?utm_source=ainews&utm_medium=email)、lmsys 上新的 [gpt2-chatbot](https://twitter.com/phill__1/status/1784964135920235000?utm_source=ainews&utm_medium=email) 以及 [将 Llama-3-8B 扩展到 1m 上下文](https://x.com/markatgradient/status/1785032103429865748?s=46&t=90xQ8sGy63D2OtiaoGJuww) 的讨论非常多，但除此之外没有明显的头条新闻。您可以关注 [WebSim/WorldSim](https://www.latent.space/p/sim-ai) 播客，Nous Research 在因安全问题短暂下线后正准备重新发布它。


---

**目录**

[TOC] 


---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取现在可以运行，但仍有很多改进空间！

**AI 模型与能力的进展**

- **Yann LeCun 预测将转向带有 AI 助手的 AR 界面**：在 /r/singularity 中，Yann LeCun 表示在 10-15 年内，我们将通过 [AR 眼镜和手环而不是智能手机](https://www.reddit.com/r/singularity/comments/1cfr9j4/yann_lecun_says_in_10_years_we_wont_have/) 与智能助手互动。
- **基于 Llama-3 的 Dolphin-2.9 模型发布**：在 /r/LocalLLaMA 中，[发布了基于 Llama-3 的新 Dolphin-2.9 模型，可能修复了之前版本的质量问题](https://www.reddit.com/r/LocalLLaMA/comments/1cf3k1d/anyone_tried_new_dolphin29llama38b256k/)。
- **PixArt Sigma 以 0.6B 参数达到 Stable Diffusion 3.0 水平**：在 /r/singularity 中，[PixArt Sigma 模型仅用 0.6B 参数就实现了 Stable Diffusion 3.0 级别的性能，完全遵循 Prompt，并可在本地使用](https://www.reddit.com/r/singularity/comments/1cfacll/pixart_sigma_is_the_first_model_with_complete/)。
- **Transformer 可以使用无意义的填充 Token 处理算法任务**：在 /r/LocalLLaMA 和 /r/MachineLearning 中，研究表明 [Transformer 可以使用像 '......' 这样无意义的填充 Token 代替思维链来解决算法任务，但这需要特定的密集监督才能收敛](https://www.reddit.com/r/LocalLLaMA/comments/1cf2w5a/transformers_can_use_meaningless_filler_tokens_eg/)。

**AI 的应用**

- **AI 生成的餐厅评论可以通过图灵测试**：在 /r/MachineLearning 和 /r/singularity 中，一项新研究发现 [AI 生成的餐厅评论可以通过图灵测试，欺骗人类和 AI 检测器](https://www.reddit.com/r/MachineLearning/comments/1cflzkmq/a_new_study_finds_that_aigenerated_restaurant/)。
- **Uber 使用图算法和学习嵌入进行 ETA 预测**：在 /r/MachineLearning 中，有人分享了 [Uber 使用结合图算法和学习嵌入（learned embeddings）的两层方法来预测 ETA](https://www.reddit.com/r/MachineLearning/comments/1cfd15u/research_a_visual_deep_dive_into_ubers_machine/)。
- **可口可乐与微软宣布为期 5 年的 AI 合作伙伴关系**：在 /r/singularity 中，宣布 [可口可乐公司与微软将开启为期 5 年的合作伙伴关系，以加速云和生成式 AI 计划](https://www.reddit.com/r/singularity/comments/1cf3a6r/the_cocacola_company_and_microsoft_announce/)。

**部署与优化 AI 模型**

- **Llama-3 70B 模型可在 4GB GPU 上通过 AirLLM 运行**：在 /r/LocalLLaMA 中，有人展示了 [Llama-3 70B 模型可以使用 AirLLM 优化技术在单个 4GB GPU 上运行，无需量化或压缩，但速度非常慢](https://www.reddit.com/r/LocalLLaMA/comments/1cf42vc/run_the_strongest_opensource_llm_model_llama3_70b/)。
- **Mistral.rs 是快速 LLM 推理平台**：在 /r/singularity 中，[Mistral.rs 被介绍为一个支持量化、多设备支持且兼容 OpenAI API 的快速 LLM 推理平台](https://www.reddit.com/r/singularity/comments/1cfsiuy/mistralrs_a_lightningfast_llm_inference_platform/)。
- **LLM 从原型推向生产环境的挑战**：在 /r/MachineLearning 中，一项调查发现，[由于各种挑战，只有 5% 的 LLM 能从原型进入生产阶段，尤其是在企业环境中](https://www.reddit.com/r/MachineLearning/comments/1cf178i/d_what_are_the_most_common_and_significant/)。
- **Llama 模型的 EXL2 和 GGUF 量化对比**：在 /r/LocalLLaMA 中，[发现 Llama-3 的 EXL2 量化在困惑度（perplexity）与模型大小的关系上与最新的 GGUF 量化表现相同，且 Llama-3 和 Llama-2 在量化后的性能下降都比全精度模型更严重](https://www.reddit.com/r/LocalLLaMA/comments/1cfbadc/result_llama_3_exl2_quant_quality_compared_to/)。

**担忧与挑战**

- **Eric Schmidt 警告 AI Agent 以自己的语言交流**：在 /r/singularity 中，Eric Schmidt 表示，[如果 AI Agent 开始以我们无法理解的语言相互交谈，我们应该拔掉电脑插头，这种情况在 2017 年的 Facebook 聊天机器人中已经发生过](https://www.reddit.com/r/singularity/comments/1cfqknmm/eric_schmidt_the_point_at_which_ai_agents_can/)。
- **OpenAI 超额扣费，忽略账单限制**：在 /r/OpenAI 中，一位用户报告被 [OpenAI 超额扣费，对方没有遵守其设置的账单限制，这可能导致集体诉讼](https://www.reddit.com/r/OpenAI/comments/1cfld2h/annoyed_because_openai_didnt_respect_my_billing/)。
- **加州 SB-1047 法案可能影响开源 AI**：在 /r/StableDiffusion 中，人们担心 [加州 SB-1047 法案如果通过，可能会对开源 AI 的努力产生负面影响](https://www.reddit.com/r/LocalLLaMA/comments/1cfizbb/california_sb1047_seems_like_it_could_impact_open/)。

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，从 4 次运行中选取最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程。

**Prompt Engineering 技术与应用**

- **推理与多步问题解决**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1784992130777137362) 概述了近期关于推理任务的 Prompt Engineering 研究，包括 **Zero-shot CoT Prompting、基于复杂度选择 CoT 示例、逐步细化推理过程，以及将复杂任务分解为子任务**。
- **工具使用与 API 集成**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1784992130777137362) 强调了关于 **教导 LLM 利用外部工具和 API** 的研究，例如基于文本的 API、由工具调用组成的自然语言程序，以及在沙箱环境中的代码 execution。
- **优化 Context Window 使用**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1784992130777137362) 讨论了关于 Context Window 属性影响的研究，例如 **无关上下文的负面影响、对 Prompt 开头/结尾的注意力偏差，以及选择最佳 Few-shot 示例的策略**。
- **改进 LLM 辅助写作**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1784992130777137362) 涵盖了增强 LLM 生成写作的技术，例如 **大纲生成与迭代填充、使用较小的 LLM 生成“方向性刺激（directional stimuli）”，以及迭代增加摘要中的信息密度**。

**大语言模型中的涌现能力与 Scaling Laws**

- **涌现能力与预训练损失 (Pretraining Loss)**：[@_jasonwei](https://twitter.com/_jasonwei/status/1784990066609414556) 讨论了一篇论文，该论文绘制了涌现能力与预训练损失的关系图，显示了**某些基准测试呈线性相关，而其他基准测试在特定损失阈值处表现出涌现行为**。建议将预训练损失作为比算力 (compute) 更好的模型比较指标。
- **函数逼近的潜在上限**：[@jxmnop](https://twitter.com/jxmnop/status/1784696357892063565) 分享了一篇论文的见解，该论文表明**截然不同的架构在相同的参数量下可以产生相同的性能**，这表明在给定一定计算量的情况下，我们可能已经接近函数逼近的上限。
- **语言模型的局限性与潜在瓶颈**：[@bindureddy](https://twitter.com/bindureddy/status/1784698453802545318) 认为，由于人类语言、推理的限制，以及尽管增加了算力或数据，但在 MMLU 等基准测试上**无法超越特定水平，语言模型可能很快就会遇到瓶颈**。

**视觉语言模型与视频理解的进展**

- **PLLaVA：无需参数的 LLaVA 视频扩展**：[@_akhaliq](https://twitter.com/_akhaliq/status/1784752877493203416) 介绍了 PLLaVA，它将 LLaVA 框架扩展到**视频密集字幕生成 (video dense captioning)，且不需要大量的配对数据**。该方法利用预训练的 2D 扩散模型和池化策略，在视频问答和字幕生成任务中实现了最先进的性能。
- **HaLo-NeRF：学习几何引导的语义**：[@_akhaliq](https://twitter.com/_akhaliq/status/1784755121496224210) 展示了 HaLo-NeRF，这是一个将地标场景的神经表示与文本描述相连接的系统，以实现**对语义区域的细粒度理解和定位**。该方法利用了适用于 3D 兼容分割和体积场景表示的视觉语言模型。

**大语言模型高效训练与部署技术**

- **用于高效 LLM 推理的 FP6 量化**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1784599257384727044) 分享了一篇关于使用 **6 位量化 (FP6) 在保持模型质量的同时减小 LLM 体积**的论文，涵盖了各种应用和模型规模。论文介绍了 TC-FPx，这是一种支持各种量化位宽浮点权重的 GPU 内核设计方案，可在 LLM 推理期间实现实际的性能提升。
- **Proxy-Tuning：大语言模型的高效定制**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1784559710978404861) 解释了 Proxy-Tuning，这是一种**轻量级的解码时算法，通过使用较小的经过微调的 LM 来改变原始预测，从而达到直接微调大型 LM 的效果**。这种方法允许通过解码时引导对大型（可能是专有的）LM 进行高效定制。
- **用于指令微调的参数高效稀疏化构建 (PESC)**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1784999595413504342) 讨论了一篇提出参数高效稀疏化构建 (PESC) 的论文，该技术**将稠密模型转换为稀疏的混合专家模型 (MoE)，以进行高效的指令微调**。PESC 在每个专家中插入适配器 (adapters)，仅更新适配器参数，在显著降低计算成本和内存需求的同时，实现了最先进的性能。

**法规与政策**

- **加州 1047 号法案详情**：[@nearcyan](https://twitter.com/nearcyan/status/1784864119491100784) 分享了已进入快速通道的加州 1047 号法案的细节。该法案**涵盖了所有使用 10^26 FLOPs 或具有类似性能的模型**，要求开发者在伪证罪的惩罚下断言模型是安全的，并创建了一个向其汇报的前沿模型部门 (Frontier Model Division)。
- **对加州 SB-1047 的担忧**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1784717268368367665) 对加州 SB-1047 “前沿人工智能模型安全创新法案”表示担忧，认为它可能**对初创公司、美国创新、开源和安全造成巨大损害**。该法案强加了过于宽泛的定义，误解了双重用途，具有限制性要求，并抑制了开放性。


---

# AI Discord 摘要回顾

> 摘要之摘要的摘要

**1. 大语言模型 (LLMs) 与 AI 能力的进展**

- **[Llama 3](https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k)** 已扩展支持 **[1M token 上下文窗口](https://x.com/markatgradient/status/1785032103429865748?s=46&t=90xQ8sGy63D2OtiaoGJuww)**，展示了在处理长序列方面的进展。教程演示了如何将 **[检索增强生成 (RAG)](https://www.youtube.com/watch?v=oDGzMF8CiQU)** 与 Llama 3 结合使用，并通过 Langchain 和 Groq 集成 **[网络浏览功能](https://www.youtube.com/watch?v=au6WQVEgGQo)**。

- **[Microsoft 的 Phi-3](https://x.com/lmsysorg/status/1783959458005279091?s=46)** 作为下一代快速且高性能的模型已开源发布，在排行榜上获得了超过 6K 的投票。讨论探索了 Llamafied 版本中的 **[tokenizer 更改](https://huggingface.co/vonjack/Phi-3-mini-4k-instruct-LLaMAfied/discussions/7)**，以获得更好的聊天应用性能。

- **[Snowflake Arctic](https://www.youtube.com/watch?v=nV6eIjnHEH0)** 是一款专注于企业级应用的 LLM，旨在为企业提供具有成本效益的 AI 解决方案，推动企业级 AI 应用的前沿。

**2. 模型优化、量化与效率技术**

- 围绕 **量化技术**（如 **[4bit lora 和 4bit qlora](https://x.com/rohanpaul_ai/status/1784972618472317180)**）展开了广泛讨论，并根据训练程度辩论了它们对模型性能的影响。**[二进制量化 (Binary Quantization)](https://github.com/carsonpo/haystackdb)** 被用于为相似性搜索创建更小的索引。

- **[DeepSpeed 的 FP6 量化](https://github.com/microsoft/DeepSpeed/commit/ccfdb84e2a4a373ac657a99afd2d97e1d741b22b)** 承诺提供具有相似吞吐量的量化推理，为提高效率带来了期待。

- 研究人员展示了能够使用 Chain-of-Thought 提示方法 **[生成 Python 代码](https://arxiv.org/abs/2404.11160)** 的 **CPU 优化型 LLM**，突显了对高效、低成本模型的追求。

**3. 开源 AI 开发与社区协作**

- **[Eleuther](https://discord.com/channels/729741769192767510/747850033994662000/1233393133937492041)** 社区对比了 LLM 性能，讨论了 **涌现能力 (emergent abilities)**，并分享了关于冗余神经电路和针对 LLM 的对抗性提示等主题的研究。

- **[OpenAccess AI Collective](https://discord.com/channels/1104757954588196865/1104757955204743201/1233372786274074734)** 深入研究了微调策略、量化方法和 Tokenization 挑战，成员们分享了来自 **[axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** 和 **[FastChat](https://github.com/lm-sys/FastChat)** 等仓库的见解。

- **[LlamaIndex](https://discord.com/channels/1059199217496772688/1059201661417037995/1233371418675380244)** 社区探索了 **多跳检索 (multi-hop retrieval)**、用于长期记忆的 **知识图谱 (knowledge graphs)** 等技术，并分享了关于 LLM 应用开发模式的 **[AWS 工作坊](https://twitter.com/llama_index/status/1783877951278432733)** 等资源。

**4. AI 开发中的伦理问题与监管挑战**

- **[LAION](https://discord.com/channels/823813159592001537/823813160075132991/1233337464169431121)** 因欧盟法律面临限制，限制了对公共计算集群的访问，促使研究人员转向实验更活跃的社区。

- 围绕拟议的 **[加州 SB-1047 法案](https://x.com/jeremyphoward/status/1784717268368367665)** 及其对初创公司、开源 AI 开发和美国创新的潜在危害展开了讨论，强调了监管挑战。

**5. 其他**

- **CUDA C++ 成为焦点**：关于 **CUDA C++ llm.cpp** 的 [YouTube 讲座](https://youtu.be/WiB_3Csfj_Q) 深入探讨了优化 LLM 训练，并承诺提供更简洁、更快速的代码。支持材料和相关讨论表明性能有显著提升，并已准备好将 LLM 扩展到 gpt-large 规模。

- **Intel 的 oneAPI 扩展版图**：Intel 的 oneAPI 因提供跨 CPU、GPU 和 FPGA 的统一编程模型而受到关注。人们对即将推出的 Battlemage GPU 系列充满热情，oneAPI 生态系统欢迎为跨供应商支持做出贡献，开发者资源可在 [GitHub](https://github.com/oneapi-src) 上获得，公告见 [Codeplay 官方新闻稿](https://codeplay.com/portal/press-releases/2022/12/16/codeplay-announces-oneapi-for-nvidia-and-amd-gpu-hardware.html)。

- **InstaDeep 的机器学习职位**：InstaDeep 正在招聘精通高性能 ML、Bio AI 和自定义 CUDA kernels 的 Machine Learning Engineers。他们为准备产生现实影响的问题解决者提供充满挑战的环境和多个职位，申请已在 [InstaDeep 招聘门户](https://www.instadeep.com/job-offer/92900fa3-5501-4506-a63f-cebee958fc6f/) 开放。

- **AMD 燃起竞争之火**：讨论围绕 AMD Instinct MI300X 在服务器环境中的潜力以及 ROCm 的现状展开，相关的 [产品页面](https://www.amd.com/de/products/accelerators/instinct/mi300/platform.html) 链接和租赁选项暗示了其与 NVIDIA 之间激烈的竞争。ROCm 的支持和对比表明，AMD 正致力于为开发者提供更高的可访问性和性能增强。

- **Triton 和 PyTorch 稳步前进**：对于寻求 Triton 和 PyTorch 集成的开发者来说，[unsloth](https://github.com/unslothai/unsloth) 和 [attorch](https://github.com/BobMcDear/attorch) 等 GitHub 仓库已成为宝库。虽然 flash-attn 2.5.8 获得了与 PyTorch 2.3.0 兼容的赞誉，但关于 Triton 中最佳 CUDA 张量索引技术和张量梯度计算的讨论，进一步增强了社区对效率的追求。

---

# PART 1: 高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Phi 3 集成是 Unsloth 的一大胜利**：Unsloth AI 现在支持 **Phi 3**，在内存占用减半的情况下实现了两倍的速度提升。爱好者可以查阅 [Colab notebook](https://colab.research.google.com/drive/1NvkBmkHfucGO3Ve9s1NKZvMNlw5p83ym?usp=sharing) 获取详细指南。

- **双语模型引起轰动**：Thermostatic 推出了 **NeuralTranslate_v0.2_GGUF**，这是一个双向英西翻译模型，它在不产生过拟合的情况下保留了 **Mistral** 的推理能力，该模型已在 [Hugging Face](https://huggingface.co/Thermostatic/NeuralTranslate_v0.2_GGUF) 上发布。

- **GPU 优化讨论**：AI 社区辩论了最小化 VRAM 使用的最佳实践，分享了关于手动层剪枝（layer pruning）的见解，并结合 [Kolibrify 的 GitHub 仓库](https://github.com/oKatanaaa/kolibrify/blob/7165ebbbcc8c44a6960ccfe78aa2d740a93789bd/kolibrify/model_utils.py) 中的代码示例讨论了卸载（offloading）技术。

- **数据集处理技巧**：分享了一个合并原始文本和聊天数据集以改善微调（fine-tuning）效果的技巧，以及对基础模型（base models）使用大型数据集、对指令模型（instruct models）使用小型数据集的观点。此外，还提到了通过卸载语言模型的部分内容来减少推理内存，相关代码在 [GitHub 仓库](https://github.com/oKatanaaa/kolibrify/blob/7165ebbbcc8c44a6960ccfe78aa2d740a93789bd/kolibrify/model_utils.py) 中有详细说明。

- **未来功能特性**：对 Unsloth AI 的建议包括自动优化超参数（如 batch size 和 learning rate）。同时，一位社区成员幽默地期待在训练完成后增加一个“烤蛋糕”的功能。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**CUDA C++ 成为焦点**：一段关于 **CUDA C++ llm.cpp** 的 [YouTube 讲座](https://youtu.be/WiB_3Csfj_Q) 深入探讨了如何优化 LLM 训练，并承诺提供更整洁、更快速的代码。配套材料和相关讨论表明，该方案在性能上有显著提升，并已准备好将 LLM 扩展到 gpt-large 规模。

**Intel 的 oneAPI 展露头角**：Intel 的 oneAPI 因提供跨 CPU、GPU 和 FPGA 的统一编程模型而受到关注。社区对即将推出的 Battlemage GPU 系列充满热情，oneAPI 生态系统欢迎为跨厂商支持做出贡献，开发者资源可在 GitHub 上找到，相关公告见 [Codeplay 官方新闻稿](https://codeplay.com/portal/press-releases/2022/12/16/codeplay-announces-oneapi-for-nvidia-and-amd-gpu-hardware.html)。

**InstaDeep 的机器学习职位**：InstaDeep 正在招聘精通高性能机器学习、生物 AI（Bio AI）和自定义 CUDA kernel 的机器学习工程师。他们提供极具挑战性的环境和多个职位，寻找准备好产生现实世界影响的问题解决者，申请通道已在 [InstaDeep 职位门户](https://www.instadeep.com/job-offer/92900fa3-5501-4506-a63f-cebee958fc6f/) 开启。

**AMD 燃起竞争之火**：讨论围绕 AMD Instinct MI300X 在服务器环境中的潜力以及 ROCm 的现状展开，相关的产品页面链接和租赁选项暗示了其与 NVIDIA 之间激烈的竞争。ROCm 的支持和对比表明，AMD 正致力于为开发者提供更高的可访问性和性能增强。

**Triton 和 PyTorch 稳步前进**：对于寻求 Triton 和 PyTorch 集成的开发者来说，[unsloth](https://github.com/unslothai/unsloth) 和 [attorch](https://github.com/BobMcDear/attorch) 等 GitHub 仓库已成为宝库。虽然 flash-attn 2.5.8 获得了与 PyTorch 2.3.0 兼容的赞誉，但关于 Triton 中最佳 CUDA 张量索引技术和张量梯度计算的讨论，进一步增强了社区对效率的追求。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Pro Search 变慢困扰用户**：Perplexity AI 的 **Pro Search** 用户正在抱怨搜索时间增加，感叹在所有引擎上的搜索时间长达 **90 秒**，这影响了 Web 客户端，但未影响移动端 App。

**Claude 3 Opus Chat：是否值得订阅？**：成员们在讨论订阅 **Claude 3 Opus** 聊天的价值，一些用户反馈了积极的体验，尽管尚未讨论与 API 版本的具体功能对比。

**新 AI 模型期待**：用户对将 **WizardLM 2** 和 **LLama-3 70B Sonar Large 32k** 模型集成到 **Perplexity AI** 表现出浓厚兴趣，并指出它们在特定任务上的表现可能优于现有模型。

**对 Opus 每日限制的沮丧**：Perplexity 用户对 **Opus** 每 24 小时 **50 次查询** 的上限表示不满，呼吁提高透明度，并对感知到的质量下降表示遗憾。

**账单烦恼与 API 咨询**：用户反映了账单问题，称在预期有免费试用的情况下仍被扣费，并正在寻找企业级 **API 讨论** 的正确渠道。同时，关于在线 LLM 的单轮对话指南、Harpa 配置以及在 make.com 等第三方平台上的模型可用性等问题也引发了技术好奇心。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Forge 功能失效**：**SDXL** 和 **Forge UI** 的问题正在爆发；用户报告了图片预览问题，并对 Forge 可能被弃用表示担忧。解决方法包括查阅 [GitHub issues](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/10132) 以及调整启动参数，如 `--no-gradio-queue`。

**发布雷达 - Stable Diffusion 3.0**：AI 工程社区正热切期待 **Stable Diffusion 3** 的发布，CivitAI 通讯暗示其将于 5 月底发布。期待中夹杂着对开放权重可用性的怀疑，以及与 **Pony Diffusion V7** 的对比讨论（详见 [Civitai 文章](https://civitai.com/articles/5069)）。

**AI 艺术变现**：关于 AI 生成艺术变现的讨论显示，在 Civitai 等市场中，NSFW 创作者的表现优于 SFW 艺术家。随后展开了对潜在盈利趋势的头脑风暴，如 AI 女友应用，并注意到用户对 **Stable Cascade** 等模型的微调努力反应冷淡。

**工具箱扩展**：工程师们交流了 AUTOMATIC1111 之外的 **AI 模型训练工具**，重点介绍了用于自定义训练的 **dreambooth** 和 **kohya_ss**，同时也思考了在数据集中使用艺术家名字的伦理困境。

**奇思妙问启发灵感**：探究性的互动涵盖了从探索 **text-to-speech** 解决方案到深入研究模型微调细节。讨论有时会变得轻松，出现关于虚拟“显卡下载”的幽默评论，以及对 **Stable Diffusion** 在没有明确提示词的情况下进行可视化能力的闲散好奇。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**VRAM 的新挑战**：讨论强调了 **VRAM** 对 LLM 运行的重要性，16GB 被视为最低基准，而对 **32GB VRAM 俱乐部** 的向往引发了热议。使用 **Nvidia 当代 GPU** 带来的性能提升，以及通过 **NVLink** 可能实现的跨多卡拆分模型的运行可行性也是关键点。

**LLM 的跨越式进步**：**Meta-Llama-3-8B-Instruct-Q5_K_M.gguf** 模型在 M1 MacBook Pro 上的表现赢得了赞誉。建议用户在运行模型时考虑量化类型以确保硬件兼容性，**LM Studio** 和 **Groq API** 等工具被认为对本地模型部署和指令操作非常有帮助。

**模型行为的怪癖**：用户遇到了各种与版本相关的问题，例如在 LM Studio 更新至 0.2.21 版本后，**phi-3 mini** 模型输出乱码，以及处理近期更新后的 LM Studio 崩溃问题。此外，关于 **LLama 8b** 模型胡言乱语的担忧，以及限制对集成显卡的依赖以充分利用独立 GPU 的需求也受到了关注。

**机器人、书籍与 Bug**：将 **Discord bots** 与 LLM 模型集成以进行消息检索和 Wikipedia 搜索已获得关注。与此同时，在移动设备或 PC 上运行 **Stanford's Octopus v2** 等模型的能力被证明是一个复杂的问题，且由于缺乏互联网访问，**LLama 3** 模型被怀疑在时事知识方面存在“幻觉”。

**ROCm 的小问题**：努力应对 **LM Studio ROCm** 限制的用户发现它不支持 **RX 6700**，这引发了对 **HIP SDK** 兼容性的思考，以及类似 *KoboldAI* 所实现的潜在变通方案。此外，平台内的一个服务器错误引发了对话，但尚未有解决报告。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Snowflake Arctic 发布高性价比 AI 解决方案**：Snowflake AI 研究团队推出了 [Snowflake Arctic](https://www.youtube.com/watch?v=nV6eIjnHEH0)，这是一个旨在提供高性价比企业级 AI 解决方案的 LLM，同时也分享了其他一些背景较少的 YouTube 视频。
  
- **Intel 和 Logitech 增强 AI 产品线**：Intel 首席执行官在季度财报中强调了 AI 的增长潜力，如 [YouTube 视频](https://youtube.com/watch?v=bWcN4a62i0Q&si=nbOPMlMFsbWEVAoG)所示；而 Logitech 推出了 AI Prompt Builder 以实现更流畅的 ChatGPT 交互，[演示视频已发布](https://www.youtube.com/watch?v=jcCTTbEvU4g)。

- **AI 量化与模型架构的新趋势**：Hugging Face 托管了 [binary-siglip-text](https://huggingface.co/carsonpoole/binary-siglip-text) 和 [binary-siglip-vision](https://huggingface.co/carsonpoole/binary-siglip-vision)，展示了高效的 embeddings。讨论还涉及对 OpenAI 命名方案的推测，以及为提高吞吐量而引入的 [DeepSpeed FP6 量化](https://github.com/microsoft/DeepSpeed/commit/ccfdb84e2a4a373ac657a99afd2d97e1d741b22b)。

- **LLM 讨论：性能问题与法律困惑**：用户报告了 LLaMA-3 的 EOS token 生成问题，这与 [GitHub 上的停止准则解决方案](https://github.com/nestordemeure/stop_word)有关；而 Cohere 对 command-r 模型的许可引发了关于商业代码使用的辩论；此外，用户对一个被误认为具有 GPT-4 能力的 gpt2-chatbot 表达了不满。

- **通过 AI 社区协作进行数据、文档与开发**：技术贡献包括生成多跳文献数据、使用 pydantic 模型进行构思，以及完善 [LLM 输出的图表示](https://github.com/furlat/Abstractions/blob/main/abstractions/angels/angels.md)。Anna’s Blog 提供了关于 WorldCat 数据抓取及其在文献理解数据集中的应用[信息](https://annas-blog.org/worldcat-scrape.html)。

- **Web 与世界模拟工具引起关注**：Nous Research 社区正准备通过免费邀请进行 **worldsim** 测试，并分享了各种 Web 模拟工具的使用经验，例如基于伴侣的 AI（记录在 [websim 示例](https://websim.ai/c/oFskF68gjd7njVn0E)），以及长对话测试，这表明人们对 AI 对话稳定性的潜力兴趣日益浓厚。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **社区构建的计算机视觉课程**：一门全新的社区构建[计算机视觉课程](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome)已在 **HuggingFace** 上线，涵盖了使用其生态系统中的模型进行该领域的机器学习原理。

- **模型展示与更新**：新发布的模型 **Qwen1.5-110B-Chat** 支持 32K 上下文长度及其他改进；详细信息可在其[模型页面](https://huggingface.co/Qwen/Qwen1.5-110B-Chat)查看。此外，“Qwen1.5-110B”模型的链接已修正，现在可以通过 [HuggingFace](https://huggingface.co/spaces/Qwen/Qwen1.5-110B-Chat-demo) 和相关的[博客文章](https://qwenlm.github.io/blog/qwen1.5-110b/)访问。

- **鼓励创意解决方案与协作**：在各种技术咨询中，成员们寻求创意的解决方案，范围涵盖未公开的 **Gradio issues** 到基于硬件限制的 **LLM Performance** 优化，特别提到 32 GB 的 RAM 应足以应付许多任务。此外，还在推动识别和改进用于实际应用（如**弹珠游戏计分系统**）的图像分类或目标识别模型。

- **模型与 Space 创新层出不穷**：涌现了各种模型和 Space，包括一个用于语义搜索任务、上下文长度为 16,384 的 **Sentence Transformer Model** ([BEE-spoke-data](https://huggingface.co/BEE-spoke-data/mega-small-embed-synthSTS-16384-v1))，以及一个使用 Stable Diffusion 模型的 **Minecraft 皮肤生成器** ([Stable Diffusion Finetuned Minecraft Skin Generator](https://huggingface.co/spaces/Nick088/Stable_Diffusion_Finetuned_Minecraft_Skin_Generator))。KingNish 开发的 **Instant Video** Space 利用了字节跳动的 AnimateDiff Lightning 模型进行快速的文本转视频创作 ([Instant Video](https://huggingface.co/spaces/KingNish/Instant-Video))。

- **Diffusion 与 AI 广告检测探索**：参与者交流了精确生成物体的最佳实践，包括在 Diffusion 模型中结合 [IP-Adapter](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter) 以增强图像 Prompt 引导，并解决跨平台的颜色一致性问题。讨论还涉及评估 **YOLO 分类器**，以提高各种应用中的准确性和性能。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 获得记忆升级**：**ChatGPT Plus 用户**现在可以使用新推出的 *Memory* 功能保存对话上下文，尽管可用性仍然有限，不包括**欧洲和韩国**的用户。
- **探索 AI 与意识的关系**：社区就 AI 是否能表现出意识展开了激烈的辩论，讨论深入到哲学领域，将 AI 对时间的体验与人类的连续意识以及神经网络中的自我感知进行比较。
- **模型对比引发讨论**：技术讨论强调了各种 AI 模型的优缺点，对 **ChatGPT**、**Claude 3 Opus** 和 **Gemini 1.5** 进行了基准测试，同时承认虽然 **command-R Plus** 和 **Llama3-70b** 可能落后于 GPT-4，但它们代表了各自的飞跃式进步。
- **Prompt 作为竞技运动**：成员们提出了 **Prompt 竞赛**的想法（包括付费和娱乐性质），以磨练技能并增强社区参与度，强调了 LLM 中可能出现的涌现特质，这些特质无法通过简单地扩展较小模型来预测。
- **API 运行状况备受关注**：工程师们讨论了各种运行问题，从自定义 GPT 使用的 **rate-limits**、"https://chat.openai.com/backend-api/gizmos/" 的后端错误，到对 **GPT-4** 功能（如 Memory 和语音控制）的性能和可用性的担忧。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**探索模型大小的极限**：工程师们正在讨论模型参数的有效截止点，寻找进一步增加参数收益微乎其微的临界点。为了追求效率，标准已转向关注 non-embedding 参数，可能会在 2 亿以下找到一个理想平衡点。

**The Pile 中的多语言障碍**：The Pile 数据集的局限性被凸显出来，表明其缺乏多语言代表性，这可能会影响模型的训练和性能，特别是在德语等语言中。此外，在比较 GPT-NeoX 和 Megatron 等模型时，讨论集中在 NeoX 以用户为中心的质量改进上。

**稳定性还是速度？模型推理服务的难题**：关于模型推理服务速度差异的技术讨论浮出水面，例如 Fireworks.ai 的 Mixtral 和 Llama 模型之间；考虑因素包括 batching size 和硬件规格等潜在因素。

**拒绝机制的单一神经元指向**：AI Alignment Forum 展示了一项发现，即 LLM 中的拒绝机制可能取决于网络层中的一个单一方向。这引发了关于拒绝行为的 orthogonalization（正交化）和 fine-tuning 可能性的讨论。

**Pull Request 的风险与流水线的烦恼**：成员们对 CLA 签署问题和 GitHub pull requests 的检查失败表示担忧，一些对话涉及特定分支的停滞。有人提出了关于评估 prompt 对不同模型 finetuning 需求的适应性问题，并建议使用自定义函数来处理多样性。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Soliloquy 8B 的两步调价**：**Soliloquy 8B 模型**过渡到了付费使用模式，价格为 **每 1M tokens 0.1 美元**，随后进一步上涨至 **每 1M tokens 0.2 美元**。这些费率反映了 OpenRouter LLC 的政策变化，并记录在[该模型的 OpenRouter 页面](https://openrouter.ai/models/lynn/soliloquy-l3)上。

- **Claude 的检查**：排查 **Claude 模型**问题的用户发现，它们的生成上限为 4k tokens，读取能力高达 200k tokens，正确的 API 设置可以优化响应。相关文档可以在[此处](https://docs.anthropic.com/claude/docs/models-overview)找到。

- **WLM-2 托管讨论**：对 **WLM-2** 托管成本的详细分析得出的结论是，盈利能力取决于 GPU 效率和闲置资源的偶然收入等因素。

- **FireLLaVA 的悄然上线**：**FireLLaVA** 是一款拥有快速初始化能力的开源多模态模型，已悄然进入 OpenRouter 套件。鉴于其非专有性质，这对开发者来说是一个重要的补充，可以在 [OpenRouter 页面](https://openrouter.ai/models/fireworks/firellava-13b)上进行探索。

- **前端困扰与节俭方案**：为了寻找一个经济实惠的前端，让家庭成员无需个人 OpenAI 账号即可访问 OpenRouter 服务，有人建议使用 Vercel 等免费层级产品，或 Contabo 等经济型 VPS。



---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **WizardLM 依然神奇**：与传闻相反，Microsoft 的 [WizardLM](https://github.com/nlpxucan/WizardLM) 模型并未消失；相反，wizardlm 团队进行了更新，确保了该仓库的持续公开访问。

- **模型微调的艺术**：讨论对比了针对特定领域的语言模型微调与使用 Retrieval-Augmented Generation (RAG) 的优劣，并引用了[医疗领域 LLM 论文](https://arxiv.org/abs/2311.16079)以及在 [fsdp_qlora](https://github.com/AnswerDotAI/fsdp_qlora) 中看到的 llama-pro 方法论。

- **量化难题与分词策略**：围绕分词挑战展开了大量讨论，LLaMA-3 等模型需要最新的 fastchat 格式化器；同时，社区通过讨论和 [Twitter 线程](https://x.com/rohanpaul_ai/status/1784972618472317180)努力理解 *4bit lora* 和 *4bit qlora* 等量化方法，揭示了量化对模型训练程度的敏感性。

- **AI 对空间与速度的需求**：一个严酷的提醒是，即使在 2x24GB GPU 上，带有 zero3 的 Fast Fourier Transform (FFT) 也可能消耗高达 **167GB 的 RAM**，这引发了关于 **torchtune** 等内存管理技术、异常高的磁盘空间占用观察，以及 PEFT 模型在神经网络微调效率方面的实用性讨论。

- **GPU 扩展秘籍与 FSDP 机制**：集体讨论了 GPU 扩展话题，交流了关于 micro batch sizes、梯度聚合以及使用 Fully Sharded Data Parallelism (FSDP) 和 ZeRO Stage 3 在多 GPU 间加载模型的细节——这些对于有效利用硬件资源至关重要。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 走向模块化**：Modular 的标准库 **modularml/mojo** 在开源后提交量增加了 23%，标志着贡献活动显著增强。
- **MAX 赋能多模态搜索**：[Modular 的一篇博客文章](https://www.modular.com/blog/multimodal-search-with-snowflake-embedding-and-max-engine)透露，**MAX Engine** 在基准测试中优于 PyTorch eager 和 ONNX runtime，在涉及文本和视觉数据的多模态搜索中表现出色。
- **Modular 推文精选**：重点展示了来自 **Modular** 的关键推文，涵盖了更新和公告，链接包括 [Tweet 1](https://twitter.com/Modular/status/1783968545052987485)、[Tweet 2](https://twitter.com/Modular/status/1785036097292292472)、[Tweet 3](https://twitter.com/Modular/status/1785036111804575967) 和 [Tweet 4](https://twitter.com/Modular/status/1785036126224548005)。
- **Mojo 领域的进展与问题**：核心讨论涵盖了将 Python 转换为 Mojo、内存分配优化以及 Mojo 中的矩阵切片。解决了标准库中的导入挑战，并且 **nightly 编译器更新** 持续发布，修复了诸如文件句柄生命周期管理等问题。
- **性能追求激增**：从字典性能调查到纠错算法的 SIMD 优化，社区深入研究了**效率增强**。**compact-dict 库**被提及为潜在的速度提升工具，并对 `__copyinit__` 的用法进行了辩论，例如在[列出的 Gist](https://gist.github.com/modularbot/6aed759930420cd70f38795dbcb874fe) 中所示。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**AWS 与 LlamaIndex 坐下来共同编码**：
[与 AWS 合作的工作坊](https://twitter.com/llama_index/status/1783877951278432733)展示了 **LLM 应用开发的 3 种模式**，重点强调了使用 S3 进行数据摄取以及使用 AWS Bedrock 进行 Embedding。

**ML 播客的安全焦点**：
最新的 [mlsecops 播客](https://twitter.com/llama_index/status/1783963718256411126)邀请了 LlamaIndex 的联合创始人讨论 **基于 LLM 的应用前景和数据安全**，包括 LlamaParse 和 LlamaCloud 等工具。

**显微镜下的 RAG**：
Marco Bertelli 的 [RAG 教程 9 部曲系列](https://twitter.com/llama_index/status/1784257178758697272)通过对关键架构组件的描述，为任何原型进入生产阶段铺平了道路。

**提升 RAG 推理能力的多步探索**：
一种增强 RAG 的方法涉及 **多跳检索过程 (multi-hop retrieval process)**，结合了 LlamaIndex 和 Cohere reranking，这增强了上下文感知并最大限度地减少了幻觉，详见[此帖](https://twitter.com/llama_index/status/1784363604340576615)。

**用 memary 记住一切**：
发布了 *memary*，这是一个使用 **知识图谱 (knowledge graphs)** 的长期记忆框架，有望扩展由 LLM 辅助的自主 Agent 的记忆能力，详见[此推文](https://twitter.com/llama_index/status/1784604356224164186)。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Flask 与密钥**：一位 **OpenInterpreter** 成员在运行 Flask 服务器时遇到了问题，并讨论了设置虚拟 `api_key` 和修改 pydantic 配置以解决命名空间冲突等变通方法。

**克服硬件障碍**：**Groq** 与 **OpenInterpreter** 缺乏集成引发了讨论，引用了旨在增加支持的 [pull request #1238](https://github.com/OpenInterpreter/open-interpreter/pull/1238)。还有关于在 OpenInterpreter 中使用 **Rabbit r1** 等设备的问题，重点关注系统的语言和语音命令功能。

**期待 Heavy**：尽管没有具体的发布细节，但人们对所谓的 **01 Heavy** 设备充满期待，同时一个针对 OpenInterpreter 的定制 3D 项目引起了关注，一位成员提示即将讨论 **01 Light** 的时间表。

**社区代码征程**：成员们积极分享了与 **OpenInterpreter** 相关项目的进展和求助请求。这包括 **llm-switcher** 以及潜在的 **Groq API** 实现，鼓励社区贡献。

**开放 AI 伦理讨论**：一场关于 AI 能力（如文件修改）伦理影响的对话被触发，特别是针对 Microsoft 的能力，隐含的建议是 **OpenInterpreter** 可以被打造得更符合多样化的用户需求。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**伯克利基准测试函数调用技能**：[Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) 作为一个新的衡量标准，定期更新以基准测试 LLM 在真实场景中调用函数的有效性。

**明确 LLM 的局限性**：对 LLM 局限性的探索强调了它们无法防止“目标漂移 (goal drift)”，[Strangeloopcanon 的文章](https://www.strangeloopcanon.com/p/what-can-llms-never-do)中提供了详细信息，强调了潜在的改进领域。

**Swyx 持续更新播客**：推荐 `swyxio` 的新播客节目，可能会引起听众的兴趣；详情通过[推文](https://x.com/swyx/status/1784253651844014237)分享。

**通过 Mixture of Depths 提升混合效果**：论文中介绍的新型 *Expert Choice Routing* Transformer 层旨在实现更快的收敛和更好的长序列处理，引发了讨论。欲了解更多深入信息，工程师可以查看[此处](https://arxiv.org/abs/2404.02258)的论文。

**Linux 视频共享升级**：对于寻求在 Discord 上获得更好视频共享体验的 Linux 用户来说，**Vesktop** 似乎是热门话题，其性能和兼容性的改进详见 [GitHub 仓库](https://github.com/Vencord/Vesktop)。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LAION 的算力难题**：欧盟法规正在阻碍 LAION 利用公共计算集群的能力，促使研究人员将注意力转向实验活动更频繁的其他研究社区。
- **Terminus Group 吸引多元专家**：非正式集体 **Terminus Research Group** 最近迎来了 "pixart guy"，标志着富含跨学科人才的新兴社区发展趋势。
- **追求 AI 美学**：**LAION-Aesthetics** 旨在利用机器学习模型量化视觉吸引力，其 [开源代码](https://github.com/LAION-AI/aesthetic-predictor) 已在 GitHub 上发布，供公众协作和使用。
- **量化难题引发关注**：Discord 成员讨论了 Reddit 上关于 LLM 基准测试在不同精度水平下表现不一致的帖子，重点关注了测试程序以及 LLM 性能中固有的不可预测性。
- **Token 生成速率讨论**：AI 工程师讨论了在先进 GPU 上针对不同模型和配置的 Token 生成速度，分享了选择 exllama 和 TabbyAPI 等高效工具可以提升整体性能。

- **工程师对 VAST 兴趣激增**：成员们深入探讨了全模态（omni-modality）基础模型和数据集 [VAST](https://github.com/txh-mercury/vast) 的潜力，通过征集使用案例和微调技巧来表达对其功能的兴趣。
- **新兴研究引发兴奋**：一篇新发表的 [研究论文](https://arxiv.org/abs/2404.16710) 因其对更高效的大模型推理和层管理的新颖提议而备受关注，引发了关于其实际应用的讨论。
- **探索将图（Graph）集成到 LLM 中**：关于将图数据结构与 LLM 结合的咨询触发了关于利用非序列数据丰富语言模型的技术和文献交流。
- **医疗 Mistral 微调的挫折**：在为医疗文本生成微调 **Mistral** 模型时遇到了挑战，主要集中在过度的序列生成以及使用 padding 协议来缓解这些问题。
- **鼓励 Eleuther 专家交流**：成员们建议咨询 Eleuther 服务器以获取 LLM 微调方面的专家指导，引发了对这个专业知识枢纽的兴趣。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**AI 增强浏览器的引擎正在升温**：AI 爱好者辩论了 **Tavily** 和 **Brave Search API** 作为集成到 AI 中的搜索引擎工具的优劣，讨论了价格点和效率，同时解决了频率限制问题 [Brave Search API 信息](https://brave.com/search/api/) 并探索了 [Tavily API 信息](https://tavily.com)。

**Cohere Toolkit 备受喜爱**：社区对 Cohere 的开源工具包表示赞赏，受益于其预构建组件，加速了 RAG 应用的部署 [GitHub 上的 Cohere Toolkit](https://github.com/cohere-ai/cohere-toolkit)。

**解决 Bug 和部署困境**：出现了在本地使用 **cohere-toolkit** 时的 sqlite3 错误以及在 Azure 上的部署挑战等技术障碍，在各种 [GitHub 资源](https://github.com/cohere-ai/cohere-toolkit) 中找到了共享的解决方案。

**自定义和微调查询**：围绕模型微调的具体细节和 Cohere 免费试用 API 的限制出现了疑问，促使了关于模型可用性和详细条款的讨论。

**Command-r 在多语言支持方面表现出色**：Command-r 在非英语语言方面的有效性得到认可，此外，对其商业使用规范的咨询引发了讨论，建议通过联系 Cohere 销售团队或使用 AWS Sagemaker 来寻求途径。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 中的公式灵活性**：围绕 **tinygrad** 的讨论集中在通过基础原语操作（primitive operations）创建数学公式，并强调了构建依赖图（dependency graph）对于 AI 建模中高效梯度计算和硬件利用率的重要性。

- **tinygrad 的动态增强功能值得期待**：成员们对即将发布的 **tinygrad 0.9** 版本表示兴奋，期待能进一步改进 AI 模型训练的新功能，并讨论了处理动态测试和符号形状（symbolic shapes）的持续工作，以增强操作的灵活性。

- **为 tinygrad 爱好者提议学习路径**：对于渴望深入了解 tinygrad 复杂性的用户，成员们建议从 [MicroGrad](https://github.com/unknownusername504/MicroGrad) 和 [MiniTorch](https://minitorch.github.io/) 开始，然后逐步深入 tinygrad 代码库。这旨在巩固基础概念，以便更好地为 tinygrad 的开发做出贡献。

- **内核优化见解**：一位成员强调了循环展开（loop unrolling）等优化技术，同时分享了[详细的技术文章](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/upcast.md)和指南，以帮助理解 tinygrad 内核优化的内部工作原理，特别是针对 AI 性能提升的目标。

- **混合模型和谐共存亮点**：提到了 tinygrad 与 **PyTorch** 之间的成功集成，利用 `nn.module` 将两个框架的特性结合到混合模型中，展示了 AI 工具链中潜在的协同效应。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**时事通讯增长的大胆举措**：成员们权衡了与 **[Semafor](https://www.semafor.com/)** 进行交叉推广的利弊，辩论了潜在的受众增长与因不必要的推广而导致品牌价值降低的风险。

**Phi-3 和 Arena 势头强劲，提供 OLMo 训练见解**：Microsoft 发布 **[Phi-3](https://x.com/lmsysorg/status/1783959458005279091?s=46)** 以及 Arena 达到 80 万张选票的里程碑引发了讨论，关于 Open Language Model 训练的**[研讨会](https://youtu.be/qFZbu2P1vZ8)**也同样备受关注，这让观众渴望获得更深层次的见解。

**RLHF 的细微差别与 Ghost Attention 的光芒减弱**：工程师们剖析了来自人类反馈的强化学习（RLHF）的细微性能表现，触及了 KTO 的前景，并辩论了 **Ghost Attention** 逐渐减弱的重要性，该技术曾被认为对维持 LLaMA 2 模型中长对话的一致性至关重要。

**OpenELM 取得成功，鼓励进步的 AI 理念**：对话集中在 **OpenELM** 的性能超越 **OLMo**，反映了社区的开发宗旨，专注于持续改进，并强调了开放模型的教育价值。

**AGI —— 一个哲学难题**：关于 AGI 主观性质的对话正在进行中，成员们对那些能引发对该话题深入思考的帖子表示赞赏。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**AI 集成查询与挑战**：工程师们寻求关于 **prompt 集成**的指导，并报告了 **AzureSearchVectorStoreRetriever** 与异步操作不兼容的问题，暗示可能需要将同步函数包装在异步中以实现兼容。社区内对于 **Gemini 1.5 Pro** 模型也存在困惑，澄清该模型仅适用于 **VertexAI**，并展示了成功的 `ChatVertexAI` 实现。

**LLM 部署与可观测性偏好**：讨论围绕不同的部署方式展开，包括 **Hugging Face** 与 **OpenAI API**；在绕过 **LangChain** 直接连接 **SQL Server** 方面提到了安全性考虑。此外，还辩论了针对 LLM 的有效可观测性工具，如 **Arize Phoenix** 和 **Langfuze**，显示出对自托管方案的轻微偏好。

**Galactic API 赠送与 AI 求职者**：**GalaxyAI** 正在提供免费的 API 访问，号称兼容 **GPT-4** 和 **GPT-3.5-turbo** 等高级模型。另外，一个 GitHub 仓库介绍了 **Genai-Job-Agents**，这是一个基于 Langchain/Langgraph 的 Agent，用于简化职位搜索和 CV 优化。

**AI 教程集锦**：一系列教程相继出现，包括“使用 LLaMA3 和 Langchain 的本地 RAG Agent”以及“使用 Langchain 和 Groq 的 Llama 3 网页浏览 Agent”，涵盖了 **RAG 系统**的设计实现和网页浏览功能。在尝试访问一本关于 **NLP 与 LLM** 的 Amazon 书籍时发现了验证码问题，但其核心内容仍受到关注。

**重振 RAG，驾驭 Llama**：来自分享频道的见解揭示了使用 **LLaMA3** 实现的**检索增强生成 (RAG)** 的进展，这为应用程序的 AI 驱动 Web UI 以及用于客户问答的交互式化身奠定了基础，扩展了跨平台交互式 AI 应用的视野。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llama 中的段错误 (Segmentation Fault)**：工程师在运行 `llamafile` 时遇到段错误，特别是在 Modal Labs 平台上使用 `Phi-3-mini-128k-instruct.F16.llamafile` 等文件时。这一问题在尝试集成各种 llamafile 的用户中被广泛报告。

- **htop 中的内存报告困扰**：[htop 中的一个显著 Bug](https://github.com/htop-dev/htop/issues/1443) 错误地显示了 Linux 上的共享内存使用情况，这可能会影响 AI 工程师在进行密集模型操作时对内存需求的感知。

- **更新至 Llamafile v0.8.1**：[llamafile v0.8.1](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.1) 的发布承诺支持 Phi-3 Mini 4k，修复了 GPU 模块崩溃问题，并为 Ubuntu 提供了捆绑的 NVIDIA + AMD 共享对象，从而可能为工程师解决一些顽固问题。

- **揭秘 LLM 输出中的怪癖**：使用 Llama3 70B 和 Mistral 通过 `llamafile` 运行 LLM 的用户观察到了带有括号和换行符的异常输出，引发了关于模型行为一致性和特异性的讨论。

- **优化 Llamafile 以获得最佳性能**：用户对优化 `llamafile` 的 GPU 使用表现出共同兴趣，并交流了关于最大化系统 RAM 利用率的技巧。用户还在寻求如何识别模型是在 GPU 还是 CPU 上运行，以及如何管理 llamafile 生成的无休止输出。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**AI 伴侣雷达：Faraday 和 Amica 备受关注**：**Faraday** 和 **Amica** 作为优先考虑**数据隐私**的 AI 伴侣应用而受到关注。其中 Faraday 得益于 **llama.cpp** 可以在本地运行，而 Amica 则提供具有增强功能的自托管和云服务。这两款应用为 AI 关系引入了新视角，提升了用户隐私。[Faraday](https://faraday.dev/) 因其长达一个月的稳定表现获得认可，而 [Amica](https://heyamica.com/) 则是一个新兴的竞争者。

**睡前故事大获全胜**：**Rosebud AI Sleep Game Jam** 的参与者通过 AI NPC 角色进行的创意设计产生了许多引人注目的作品，其中 **[Bedtime Negotiation](https://play.rosebud.ai/games/dd6e8a7e-6ca1-4cda-8a5c-f4e422f84ba6)** 脱颖而出，获奖名单已通过 [Twitter](https://twitter.com/Rosebud_AI/status/1784038539769815543) 公布。接下来的新一届 Game Jam 将聚焦于**教育与 AI**，详情可见 [Twitter](https://twitter.com/Rosebud_AI/status/1785034624256618617)。

**令人上瘾的 AI Town**：**AI Town** 在一篇 [Twitter 帖子](https://x.com/ivanfioravanti/status/1784248117388353655)中因其令人上瘾的特性而受到赞赏，并激发了开发一个以开发者为中心的模拟游戏的想法。社区分享了基于 LLM 的 NPC 模型和基础设施增强方案，并在 [GitHub](https://github.com/GigaxGames/gigax) 上发布了仓库，在 [Huggingface](https://huggingface.co/Gigax) 上发布了模型库。尽管 API 访问链接失效，但仍在征集关于这些 NPC 进展的反馈。

**AI Town 的地图探索**：关于 AI Town 地图处理方式的讨论浮出水面，建议包括使用静态资源以减少带宽占用，以及优化原始的地图文件读取方法。一段名为 ["100% Local 'AI Town' with Llama 3 AGENTS!!!"](https://www.youtube.com/watch?v=4HBRh1hMoXQ) 的 YouTube 教程被推广，为渴望尝试本地部署的用户提供了指南。

**角色创作挑战**：围绕 NPC 角色开发的对话促成了一个详细博客文章的承诺。讨论重点在于如何压缩模型输出、减少模型调用次数，并解决在 GPT-3.5 或 Mistral 等通用型 Instruct-models 中发现的问题。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**DiscoResearch 深入探究路由系数之谜**：工程师们讨论了不同版本 **Mixtral** 之间 `router_aux_loss_coef` 的不一致性——**Mixtral-8x7B-Instruct-v0.1** 为 0.02，而 **Mixtral-8x22B-Instruct-v0.1** 为 0.001——这表明较小的 Expert 可能需要更高的 `loss_coef`。

**初始化不一致引发 GPU 讨论**：**DiscoLM_German_7b_v1** 模型在 HPC 上与本地机器相比启动缓慢；在将模型加载到 GPU 后，推理时间从超过 12 分钟缩短到了 10 秒。

**模型加载遭遇速度瓶颈**：尝试使用 `low_cpu_mem_usage=True` 来提高 **DiscoLM_German_7b_v1** 加载时间的努力失败了，这引发了模型可能受限于慢速存储驱动器的猜测。

**德语模型下载热潮**：**gguf model** 在两天内达到 1500 次下载，显示出社区对德语语言模型的强烈需求。

**用于闲聊的分词处理**：关于旨在优化聊天应用的 **Phi-3** Llamafied 德语模型中 Tokenizer 配置变化的问题被提出，同时新创建的 **Phi-3 MoE** 模型也已出现，用于需要进一步训练的实验。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **AI 应对复杂话题：** 讨论了应用 **Llama 3** 评估**话题复杂度**的情况，并报告了有效的结果。这表明对 AI 内容评估能力的探索正在持续进行。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

**CPU 优化 LLM 在 Python 代码生成方面的突破**：一项新研究展示了能够生成 Python 代码的 CPU 优化语言模型，并提出了一种 *Chain-of-Thought prompt* 方法来改进模型输出，详见论文 ["Low-Cost Language Models: Survey and Performance Evaluation on Python Code Generation"](https://arxiv.org/abs/2404.11160)。

**HaystackDB 中的 Binary Quantization 热议**：讨论围绕 [HaystackDB 仓库](https://github.com/carsonpo/haystackdb) 可能使用 2bit embeddings 展开，并进一步澄清 **Binary Quantization** 通过为相似性搜索创建更小的索引来提高效率。

**训练 LLaMA-3 完成生成时遇到麻烦**：一名成员在微调 LLaMA-3 模型时遇到了问题，模型无法生成 End Of Sentence (EOS) token，这影响了对完成度要求极高的模型性能。

**Snowflake Arctic 降低企业级 AI 成本**：一段[视频](https://www.youtube.com/watch?v=nV6eIjnHEH0)介绍了 **Snowflake Arctic**，这是一款专为企业应用设计的大型语言模型，专注于为企业提供具有成本效益的 AI 解决方案。

**LLaMA3 的 RAG 精彩演示**：分享了教程[视频](https://www.youtube.com/watch?v=oDGzMF8CiQU)，展示了如何通过 **Langchain** 在本地环境中使用 LLaMA3 进行 Retrieval-Augmented Generation (RAG)，以及一场关于使用 LLaMA 3、Langchain 和 Groq 硬件实现网页浏览的会议，链接见[此处](https://www.youtube.com/watch?v=au6WQVEgGQo)。



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**Gamma 招聘 AI 工程师**：由 a16z 领投并拥有超过 **1000 万用户** 的 Gamma 正在招聘一名 **AI 工程师**，负责 prompt engineering、评估以及文本和图像模型的微调。该职位在他们的内容创建工具扩展中至关重要，公司以其在极小团队规模和充足资金下实现的增长而自豪，这表明了其稳健的商业模式和显著的市场影响力。

**寻找 AI 人才**：候选人可以申请 Gamma 的 **AI 工程师** 职位，工作地点位于旧金山市中心，要求每周进行三次实地协作。对于那些热衷于突破大型语言模型 (LLMs) 边界的人来说，这是一个机会，更多信息请访问 [Gamma 的招聘页面](https://careers.gamma.app/ai-engineer)。

**GPT 侦探**：关于 **gpt2-chatbot** 的猜测不断，一些人怀疑它是 **GPT-4.5** 的泄露版本，起因是 @phill__1 关于其深厚领域知识的一条 [推文](https://x.com/phill__1/status/1784964135920235000) 引发的讨论。社区成员对此反应热烈，纷纷认可该 bot 的质量。

**推文点赞**：社区表达了简洁的观点，认为 **gpt2-chatbot** “非常出色”，这表明社区对其令人印象深刻的性能达成了共识，暗示了其在该领域的潜力和未来能力。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **代码生成走向定制化**：关于增强代码生成的讨论包括了 **自定义语法实现 (custom grammar implementation)** 的想法，以防止语法错误，并强调了一种可以提高语义准确性的模型特定选项。



---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。


---

# 第二部分：按频道详细摘要和链接



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1233322675858837534)** (912 条消息🔥🔥🔥):

- **Unsloth 支持 Phi 3 发布**：Phi 3 现在已由 Unsloth 正式支持，提供 2 倍的运行速度并减少 50% 的显存占用。用户可以在此处找到详细的 [Colab notebook](https://colab.research.google.com/drive/1NvkBmkHfucGO3Ve9s1NKZvMNlw5p83ym?usp=sharing)。
- **Unsloth 性能增强**：Phi 3 可以使用 Unsloth 框架进行 4-bit 精度的微调，以适应 VRAM 的限制。用户正在尝试结合 SFT、DPO 和 ORPO 的各种微调流程，以增强模型性能。
- **微调中的 Checkpoints 管理**：用户可以在使用 Unsloth 进行微调时创建 checkpoints，以保存进度并避免过拟合。为此，必须相应地修改训练参数并处理从所需 checkpoints 恢复的操作。
- **Colab 及其替代方案的使用分析**：用户讨论了 Google Colab 付费版由于运行时断开连接而存在的局限性，并探索了像 TensorDock 这样提供更实惠且可靠的 GPU 访问以进行模型训练的替代服务。
- **GGUF 转换的技术困难**：即使在本地使用 Unsloth 框架，将模型转换为 GGUF 格式时仍存在一些问题。建议用户升级 Unsloth，并可能需要重新编译 llama.cpp 以解决量化失败的问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://x.com/dudeman6790/status/1784411049141092400">来自 RomboDawg (@dudeman6790) 的推文</a>：我送给世界的礼物。在免费的 Google Colab 层级上，使用 1,500 行或更少（大约）的任何数据集训练 Llama-3-8b（模型卡中提供了所有代码）。使用了 (Unsloth + Galore + Qlora)，如果你愿意的话，可以称之为 Qalore...</li><li><a href="https://huggingface.co/blog/maywell/llm-feature-transfer">一键扩展模型上下文并创建聊天模型</a>：未找到描述</li><li><a href="https://huggingface.co/rombodawg/test_dataset_Codellama-3-8B">rombodawg/test_dataset_Codellama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-8b">unsloth/llama-3-8b · Hugging Face</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1NvkBmkHfucGO3Ve9s1NKZvMNlw5p83ym?usp=sharin">Google Colaboratory</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit/blob/main/generation_config.json">generation_config.json · unsloth/llama-3-8b-Instruct-bnb-4bit (main 分支)</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.14047">低比特量化的 LLaMA3 模型表现如何？一项实证研究</a>：Meta 的 LLaMA 家族已成为最强大的开源大语言模型 (LLM) 系列之一。值得注意的是，最近发布的 LLaMA3 模型在各项指标上都取得了令人印象深刻的性能...</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit/blob/main/config.json">config.json · unsloth/llama-3-8b-Instruct-bnb-4bit (main 分支)</a>：未找到描述</li><li><a href="https://tenor.com/view/the-office-pam-beesly-how-would-one-do-that-jenna-fischer-gif-20699672">The Office Pam Beesly GIF - The Office Pam Beesly How Would One Do That - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://marketplace.tensordock.com/deploy,">发生未找到错误！- TensorDock</a>：发生未找到错误！- TensorDock。在几秒钟内部署 GPU 并节省 80%。无合同，无承诺。安全可靠。支持 TensorFlow 和 PyTorch，简单易用。起步价仅需 $5。</li><li><a href="https://askubuntu.com/questions/8653/how-to-keep-processes-running-after-ending-ssh-session">如何在结束 SSH 会话后保持进程运行？</a>：假设我从一个 SSH 会话启动了一堆进程。是否可以在终止 SSH 会话的同时让这些进程在远程机器上继续运行？</li><li><a href="https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1">DiscoResearch/DiscoLM_German_7b_v1 · Hugging Face</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1NvkBmkHfucGO3Ve9s1NKZvMNlw5p83ym?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit">unsloth/Phi-3-mini-4k-instruct-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/wow-gif-20411229">Wow GIF - Wow - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-fro">首页</a>：微调 Llama 3, Mistral & Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://ko-fi.com/unsloth">在 Ko-fi 上支持 Unsloth AI！❤️. ko-fi.com/unsloth</a>：在 Ko-fi 上支持 Unsloth AI。Ko-fi 让你通过小额捐赠支持你喜爱的人和事业</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">首页</a>：微调 Llama 3, Mistral & Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://www.runpod.io/console/deploy?template=runpod-torch-v21">未找到标题</a>：未找到描述</li><li><a href="https://youtu.be/5poVsIeq3TM">PC 重生 – 介绍 Snapdragon X Plus</a>：PC 重生：介绍 Snapdragon X Plus，Snapdragon X 系列中的最新平台。配备尖端技术，提供强大的...</li><li><a href="https://www.youtube.com/watch?v=aQmoog_s8HE">LLAMA-3 🦙：在你的数据上进行微调的最简单方法 🙌</a>：学习如何使用 Unsloth 在你自己的数据上微调最新的 Llama 3。🦾 Discord: https://discord.com/invite/t4eYQRUcXB ☕ Buy me a Coffee: https://ko-fi.com...</li><li><a href="https://github.com/PKU-YuanGroup/Machine-Mindset">GitHub - PKU-YuanGroup/Machine-Mindset：大语言模型的 MBTI 探索</a>：大语言模型的 MBTI 探索。通过在 GitHub 上创建账号，为 PKU-YuanGroup/Machine-Mindset 的开发做出贡献。</li><li><a href="https://github.com/unslothai/hyperlearn">GitHub - u

<li><a href="https://github.com/unslothai/hyperlearn">unslothai/hyperlearn: 快 2-2000 倍的 ML 算法，减少 50% 的内存占用，适用于所有新旧硬件。</a>: 快 2-2000 倍的 ML 算法，减少 50% 的内存占用，适用于所有新旧硬件。 - unslothai/hyperlearn</li><li><a href="https://youtu.be/3LopI4YeC4I">成功是靠运气还是努力？</a>: 在竞争激烈的世界中，微小的优势就能决定成败。在美国使用代码 'giveluck' 可享受 Snatoms 10% 的折扣：https://ve42.co/USA 或国际站...</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 微调 Llama 3, Mistral &amp; Gemma LLMs 快 2-5 倍，且减少 80% 的内存占用</a>: 微调 Llama 3, Mistral &amp; Gemma LLMs 快 2-5 倍，且减少 80% 的内存占用 - unslothai/unsloth</li><li><a href="https://journal.hexmos.com/insecure-output-handling/">LangChain 和 ChatGPT 插件是如何受到此漏洞攻击的</a>: LLM 上的不安全输出处理涉及在训练阶段注入毒性数据。在本文中，我们将重点关注现实世界的场景、实践演示和预防机制...</li><li><a href="https://huggingface.co/botbot-ai/CabraLlama3-8b/tree/main?show_tensors=model.safetensors.index.json">botbot-ai/CabraLlama3-8b at main</a>: 未找到描述</li><li><a href="https://huggingface.co/arthrod/cicerocabra/tree/main?show_tensors=model.safetensors.index.json">arthrod/cicerocabra at main</a>: 未找到描述</li><li><a href="https://github.com/huggingface/transformers/pull/30079">winglian 提交的 schedulefree 优化器 · Pull Request #30079 · huggingface/transformers</a>: 此 PR 做了什么？为 adamw 和 sgd 集成了 Meta 的 https://github.com/facebookresearch/schedule_free https://twitter.com/aaron_defazio/status/1776320004465582331 在提交之前...</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: C/C++ 中的 LLM 推理</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号来为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/googlecolab/colabtools/issues/3451">Colab Pro+ 用户的运行时间少于 10 小时 · Issue #3451 · googlecolab/colabtools</a>: 我是一名 Google Colab Pro+ 用户。在 2023 年 1 月，我可以连续运行 24 小时。然而，从 2 月初开始，我的作业在运行不到 10 小时后就会超时。虽然...</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/2948">教程：如何将 HuggingFace 模型转换为 GGUF 格式 · ggerganov/llama.cpp · Discussion #2948</a>: 来源：https://www.substratus.ai/blog/converting-hf-model-gguf-model/ 我在我们的博客上发布了这篇文章，但认为这里的其他人可能也会受益，所以也在 GitHub 上分享了博客原文。希望它...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama：改进 BPE 预处理 + 支持 LLaMA 3 和 Deepseek，由 ggerganov 提交 · Pull Request #6920 · ggerganov/llama.cpp</a>: 延续了 @dragnil1 在 #6252 中的工作。此 PR 为 llama.cpp 增加了对 BPE 预分词的支持。总结：到目前为止，对于所有基于 BPE 的模型，llama.cpp 都应用了默认的预处理...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1233347807314972724)** (55 messages🔥🔥): 

- **数据集组合技巧**：一段对话建议合并原始文本和聊天数据集以改善结果，暗示了微调模型的一种潜在方法。

- **Notebook 与微调技巧公开**：Unsloth AI 社区分享了一个包含语言模型微调 Notebook 的 [GitHub 仓库链接](https://github.com/unslothai/unsloth)，以及一个专门用于文本补全任务的 [Colab Notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)。

- **Colab 显存不足 (OOM) 解决方案**：分享了一段有用的代码片段来缓解 Colab 的 OOM 问题，建议在循环中使用 `torch.cuda.empty_cache()` 和 `gc.collect()`。

- **推广点对点分享**：一位用户宣布创建了一个开放社区来讨论多模态 AI 的最新进展，并提供了一个[链接](https://bio.link/openmultimodal)以便在各个社交平台上关注他们。

- **Unsloth AI 支持新模型**：**Phi 3** 模型现已获得支持，这令人兴奋。一位用户提供了一个指向 Discord 频道的链接，其中包含相关的 Colab（Discord 外部无法访问该链接）。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Out_of_memory">Out of memory - Wikipedia</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.14367">Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data</a>: 从偏好标签中学习在微调大语言模型中起着至关重要的作用。偏好微调有几种不同的方法，包括监督学习、On-policy re...</li><li><a href="https://bio.link/openmultimodal">OpenMultiModal</a>: 探索和协作多模态 AI 的社区</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 微调 Llama 3、Mistral 和 Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1233320788241944649)** (506 条消息🔥🔥🔥): 

- **编译问题排查**：用户讨论了编译代码时的错误，特别提到了 *llama.cpp* 不在正确的文件夹中，并通过遵循正确的安装说明成功解决了问题。

- **支持查询和更新请求**：关于 **Unsloth AI** 对 **Llava** 和 **Qwen** 等不同模型支持情况的讨论显示，目前尚不支持这些模型。用户建议进行改进，例如增加从聊天模板特定部分进行截断的功能。在 **xformers** 更新后，对 Colab notebook 的安装说明进行了更新。

- **数据集格式和微调咨询**：一位用户寻求澄清其数据集格式是否适用于微调，以及应使用 Unsloth 的哪款 **Llama 3** 模型进行代码训练。澄清了较大的数据集适用于 Base 模型，而较小的数据集则适合 Instruct 模型。

- **Unsloth Pro 的 GPU 使用情况**：一位用户询问了在一块或多块 *RTX 4090* GPU 上使用 **Unsloth Pro** 的好处。他们被告知，随着 **GPU** 数量的增加，收益会成倍增长。

- **重复 Python 安装问题**：讨论强调了安装问题，包括用户安装了两个 Python 版本导致依赖冲突的情况。通过调整 Python 版本并移除旧版本解决了此问题。

- **使用代码微调 Llama**：关于微调 **Llama 3** 的问题得到了解答，并为一位想用 Svelte 代码微调 **Llama** 的用户提供了指导。建议他们使用 Base 模型，并说明了其与 Instruct 变体的区别。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1efOx_rwZeF3i0YsirhM1xhYLtGNX6Fv3?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing>">Google Colaboratory</a>: 未找到描述</li><li><a href="https://ollama.com/">Ollama</a>: 快速上手并运行大型语言模型。</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2#scrollTo=LjY75GoYUCB8&line=1&uniqifier=1">Google Colaboratory</a>: 未找到描述</li><li><a href="https://hub.docker.com/r/pytorch/pytorch">Docker</a>: 未找到描述</li><li><a href="https://huggingface.co/xtuner/llava-llama-3-8b-v1_1">xtuner/llava-llama-3-8b-v1_1 · Hugging Face</a>: 未找到描述</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/loading#local-and-remote-files">Load</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/peft/v0.10.0/en/package_reference/peft_model#peft.get_peft_model.peft_config">Models</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">Quantization</a>: 未找到描述</li><li><a href="https://unsloth.ai/.">Unsloth AI | Finetune Llama 3 &amp; Mistral LLMs</a>: 为 AI 和 LLM 提供 Unslow 微调。使用 Unsloth 提速。开源。</li><li><a href="https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat">Qwen/CodeQwen1.5-7B-Chat · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: 我在原生 Windows 中运行了 unsloth（无需 WSL）。你需要 Visual Studio 2022 C++ 编译器、Triton 和 DeepSpeed。我有一个完整的安装教程，本想写在这里但我在用手机...</li><li><a href="https://github.com/ollama/ollama">GitHub - ollama/ollama: Get up and running with Llama 3, Mistral, Gemma, and other large language models.</a>: 快速上手并运行 Llama 3, Mistral, Gemma 以及其他大型语言模型。 - ollama/ollama</li><li><a href="https://github.com/janhq/jan">GitHub - janhq/jan: Jan is an open source alternative to ChatGPT that runs 100% offline on your computer. Multiple engine support (llama.cpp, TensorRT-LLM)</a>: Jan 是 ChatGPT 的开源替代方案，100% 在你的电脑上离线运行。支持多引擎（llama.cpp, TensorRT-LLM） - janhq/jan</li><li><a href="https://github.com/unslothai/unsloth/issues/73">Conda installation detailed instructions · Issue #73 · unslothai/unsloth</a>: 我正尝试按照说明在 Conda 环境中安装 unsloth，问题是 Conda 在运行安装行时卡住了。我已经试过两次了，两次都...</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: C/C++ 实现的 LLM 推理。通过在 GitHub 上创建账号来为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1233450061200101577)** (74 条消息🔥🔥): 

- **发布用于课程学习的 Kolibrify**: [Kolibrify](https://github.com/oKatanaaa/kolibrify) 已发布，这是一个专为使用 Unsloth 进行指令遵循 LLM 课程训练而设计的项目。它被描述为对 LLM 微调和快速原型设计非常有用。

- **Thermostatic 发布双语翻译模型**: Thermostatic 的双向英西翻译模型新版本 [NeuralTranslate_v0.2_GGUF](https://huggingface.co/Thermostatic/NeuralTranslate_v0.2_GGUF) 已发布，据称该模型保留了 Mistral 的原生推理能力，且不存在过拟合。

- **AI 未来中的特定领域技能 Agent**：@timelordraps 预测了一个为期 6 个月的路线图，届时 AI 的进步将催生高能力的模型、Token 高效的预训练、自我扩展和自我生成的子 Agent (subagents)，并在 11 月前实现递归自我改进。

- **Token 高效的克隆项目正在进行中**：@timelordraps 正在优化一个 Devin 克隆版以提高 Token 效率，目前正在针对一个简单的贪吃蛇游戏进行故障排除，并计划在其他用例上进行测试并与图像模型集成。

- **Llama 社区中心发布**：新推出的 [llama-hub](https://www.llama-hub.com/) 是一个用于分享和讨论 Llama 模型及用例的社区平台。官方的 Unsloth llama-3-8b-bnb-4bit 已发布供社区访问。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.llama-hub.com/">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/winglian/llama-3-8b-256k-PoSE">winglian/llama-3-8b-256k-PoSE · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Thermostatic/NeuralTranslate_v0.2_GGUF">Thermostatic/NeuralTranslate_v0.2_GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/xtuner/llava-phi-3-mini">xtuner/llava-phi-3-mini · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/vonjack/Phi-3-mini-4k-instruct-LLaMAfied/tree/main">vonjack/Phi-3-mini-4k-instruct-LLaMAfied at main</a>：未找到描述</li><li><a href="https://github.com/oKatanaaa/kolibrify">GitHub - oKatanaaa/kolibrify: 使用 Unsloth 进行指令遵循 LLM 的课程训练</a>：使用 Unsloth 进行指令遵循 LLM 的课程训练 - oKatanaaa/kolibrify</li><li><a href="https://github.com/TimeLordRaps/timelord">GitHub - TimeLordRaps/timelord: 节省你的时间。</a>：节省你的时间。通过创建一个 GitHub 账户来为 TimeLordRaps/timelord 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1233578689996914778)** (119 条消息🔥🔥): 

- **增强 Unsloth 的自动调优 (Autotuning)**：一位用户建议 Unsloth AI 应该根据模型和数据集的特性自动优化诸如 Batch Size 和学习率 (Learning Rate) 等数值。另一位成员幽默地提议 Unsloth 还应该在训练后烤个蛋糕，这与路线图一致，而第三个人分享了关于实现的看法。

- **手动层剪枝 (Layer Pruning) 辩论**：对话涵盖了手动剪枝模型层的复杂性，一位用户建议替换 `forward` 方法以“跳过”部分层。关于是移除整个解码器块 (Decoder Blocks) 还是专注于用于 SNR (信噪比) 优化的矩阵线性投影 (MLP) 组件进行了深入讨论，并触及了最小化模型大小和 VRAM 占用的不同策略。

- **VRAM 减少策略与卸载 (Offloading)**：对话转向了减少模型大小的策略，特别是在 VRAM 使用方面。一位用户提到了一种通过卸载部分语言模型来成功减少推理内存的技术，并分享了将该方法集成到 GitHub 仓库中的经验 (https://github.com/oKatanaaa/kolibrify/blob/7165ebbbcc8c44a6960ccfe78aa2d740a93789bd/kolibrify/model_utils.py)。

- **Gemma 2b 模型与 Unsloth 的兼容性**：一位 Unsloth 的粉丝询问了 Recurrent Gemma 2b 模型与 Unsloth 的兼容性，一位成员认可了其潜在优势，但指出 Gemma 2b 存在已知的 VRAM 问题，目前重点是 Phi 3。另一位提到只有一个人遇到了独特的 VRAM 问题，没有广泛的报告。

- **Gemma 2b 的潜在功能或 Bug**：寻求关于 Gemma 2b 是具有导致 VRAM 问题的特性还是存在 Bug 的澄清。解释称虽然模型仍然可以工作，但 VRAM 问题需要解决；然而，并非所有人都遇到了这个问题，这可能是一个孤立案例。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html">如何在 PyTorch 中使用 TensorBoard — PyTorch 教程 2.3.0+cu121 文档</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/tasks/sequence_classification">文本分类</a>：未找到描述</li><li><a href="https://github.com/l4b4r4b4b4/trl/blob/evol_laser_merge_trainer/trl/trainer/laserm_trainer.py">trl/trl/trainer/laserm_trainer.py 分支 evol_laser_merge_trainer · l4b4r4b4b4/trl</a>：使用强化学习训练 Transformer 语言模型。- l4b4r4b4b4/trl</li><li><a href="https://github.com/oKatanaaa/kolibrify/blob/7165ebbbcc8c44a6960ccfe78aa2d740a93789bd/kolibrify/model_utils.py#L64)">kolibrify/kolibrify/model_utils.py 提交 7165ebbbcc8c44a6960ccfe78aa2d740a93789bd · oKatanaaa/kolibrify</a>：使用 Unsloth 对遵循指令的 LLM 进行课程学习训练 - oKatanaaa/kolibrify
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1233466150218895411)** (18 条消息🔥): 

- **CUDA 讲座倒计时**：下一场 CUDA Mode 讲座宣布将在 1 小时 40 分钟后举行。随着 llm.cpp 团队参与讨论的消息传出，气氛日益热烈，预计会非常火爆。
- **为认知注入咖啡因**：一位成员表示已经煮好了咖啡，为即将到来的讲座做好了准备。
- **宣布 CUDA 实时性能分析（Profiling）环节**：今天的会议已移至 Google Meet，链接为 [此链接](https://meet.google.com/exs-nhem-hbg)。尽管 Discord 上出现了一些小状况，但实时性能分析讲座反响良好，并承诺在 YouTube 频道上发布剪辑版本。
- **探索更广泛的硬件讨论**：有人提议为华为昇腾（Huawei Ascend）解决方案创建讨论区，以促进更多样化的硬件对话，考虑到目前 NVIDIA 和 AMD 的主导地位。该想法正在根据社区兴趣和活跃度进行评估。
- **廉价的创新**：分享了一个引人入胜的项目，该项目在不带乘法器的 10 美分 RISC-V MCU 上实现了神经网络，展示了以极低成本实现强大技术的范例。完整的博客文章和包含详细文档的仓库可在 [cpldcpu 的博客](https://cpldcpu.wordpress.com/2024/04/24/implementing-neural-networks-on-the-10-cent-risc-v-mcu-without-multiplier/) 和 [GitHub](https://github.com/cpldcpu/BitNetMCU) 上找到。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://cpldcpu.wordpress.com/2024/04/24/implementing-neural-networks-on-the-10-cent-risc-v-mcu-without-multiplier/">在“10 美分”不带乘法器的 RISC-V MCU 上实现神经网络</a>：我一直想建立一个在小型微控制器上实现基于神经网络算法的设置。在审查了现有解决方案后，我觉得没有一个方案能让我……</li><li><a href="https://cpldcpu.wordpress.com/2024/04/24/implementing-neural-networks">在“10 美分”不带乘法器的 RISC-V MCU 上实现神经网络</a>：我一直想建立一个在小型微控制器上实现基于神经网络算法的设置。在审查了现有解决方案后，我觉得没有一个方案能让我……
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1233579897306353716)** (10 条消息🔥):

- **Triton 张量索引详解**：分享了一种使用一个 Triton 张量对另一个张量进行索引的方法，涉及从索引张量中加载值，并将其与步长（strides）和基指针（base pointer）结合使用以创建指针张量，然后应用 `tl.load()` 和 `tl.store()` 来获得所需结果。
- **寻找开源 Triton LLM 实现**：一位成员正在寻找针对 llama 或 mistral 等大语言模型（LLMs）的开源 Triton 实现。另一位成员推荐了 [GitHub 上的 unsloth 仓库](https://github.com/unslothai/unsloth)，该仓库可能符合他们的需求。
- **探索使用 Triton 进行高效梯度计算**：有人提问如何利用 Triton 中的并行线程并沿某一维度进行求和归约（sum reducing）来计算张量的梯度，并分享了代码片段以说明当前和拟议的方法。
- **重点介绍包含所需 Triton Kernel 的仓库**：在关于是否存在使用 Triton Kernel 实现完整大语言模型的讨论中，提到了几个资源，包括 [xformers 仓库](https://github.com/facebookresearch/xformers/tree/main/xformers/triton) 和 [flash-attention 仓库](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/ops)。
- **分享 Triton 编写的 PyTorch 模块**：一位成员推荐了 [attorch 仓库](https://github.com/BobMcDear/attorch)，这是一套使用 Python 和 Triton 编写的 PyTorch 神经网络模块，可能非常有用。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/BobMcDear/attorch">GitHub - BobMcDear/attorch: A subset of PyTorch's neural network modules, written in Python using OpenAI's Triton.</a>：PyTorch 神经网络模块的一个子集，使用 Python 和 OpenAI 的 Triton 编写。- BobMcDear/attorch</li><li><a href="https://github.com/facebookresearch/xformers/tree/main/xformers/triton">xformers/xformers/triton at main · facebookresearch/xformers</a>：可黑客定制且经过优化的 Transformers 构建块，支持组合式构建。- facebookresearch/xformers</li><li><a href="https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/ops">flash-attention/flash_attn/ops at main · Dao-AILab/flash-attention</a>：快速且内存高效的精确注意力机制。欢迎通过在 GitHub 上创建账号来为 Dao-AILab/flash-attention 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>：微调 Llama 3、Mistral 和 Gemma LLM 速度提升 2-5 倍，内存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>

---

**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1233410335159881779)** (40 messages🔥): 

- **Kernel 性能分析之谜**：对 PMPP 中的 **tiled_matmult kernel 与 coarsed_matmult kernel** 进行性能分析显示，尽管后者具有更高的算术强度，但两者的 FLOP/s 差异出人意料地小。建议查看指令统计数据，特别是 **stall short scoreboard**（短记分板停顿），这与 SRAM 操作有关，可能会影响内存带宽。

- **CUDA KERNEL 性能优化技巧**：在优化 CUDA kernel 时，成员建议查看 **warp state stats**（Warp 状态统计），并指导从 SRAM 加载多个值到寄存器中以执行多次乘法，从而提高 SRAM 利用率。

- **低成本学习 CUDA**：关于获取用于 CUDA 学习的 GPU 访问权限的讨论，范围从利用公司/大学资源到使用 **Google Colab 和 Lightning AI** 等服务。成员们强调了掌握环境控制权的重要性，特别是对于使用性能计数器进行性能分析而言。

- **CUDA 开发中新兴的 FP6 数据类型**：GitHub 上的一个 **DeepSpeed commit** 引入了一种名为 FP6 的新数据类型，并在 A100 GPU 上提供 Tensor Core 支持，这可能会改进大语言模型（LLMs）的推理服务，并解决推理过程中的内存限制挑战。

- **讨论 CUDA 编程的最佳实践**：解答了关于 CUDA 编码实践的疑问，包括在 kernel 代码中是否应**避免整数除法**。一种建议是利用**位移来实现 2 的幂次除法**，并观察到 nvcc 或 ptxas 应该会自动进行此类优化。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://godbolt.org/z/9K9Gf1v6P">Compiler Explorer - CUDA C++ (NVCC 11.7.0)</a>: #include &amp;lt;algorithm&amp;gt; #include &amp;lt;cassert&amp;gt; #include &amp;lt;cstdio&amp;gt; #include &amp;lt;cstdlib&amp;gt;  __global__ void sgemmVectorize(int M, int N, int K, float alpha, f...</li><li><a href="https://youtu.be/4sgKnKbR-WE?si=sGinVNe5KoCwql2G)">Lecture 3: Getting Started With CUDA for Python Programmers</a>: Jeremy 的 YouTube 录像 https://www.youtube.com/watch?v=nOxKexn3iBo 补充内容：https://github.com/cuda-mode/lecture2/tree/main/lecture3Speak...</li><li><a href="https://colab.research.google.com/drive/15mWl0pvuyrriqFEnf1py7TlI9suRsesS?usp=sharing)">Google Colaboratory</a>: 未找到描述</li><li><a href="https://x.com/mejia_petit/status/1784641633369182318">来自 Nicolas Mejia Petit (@mejia_petit) 的推文</a>: 为什么大家不在讨论这个？？？DeepSpeed 开发者刚刚在 A100 上创建了一个具有完整 Tensor Core 支持的数据类型 FP6。（因为 NVIDIA 让我们困在 int4/8 中）这太聪明了...</li><li><a href="https://github.com/microsoft/DeepSpeed/commit/ccfdb84e2a4a373ac657a99afd2d97e1d741b22b">FP6 端到端量化 (#5234) · microsoft/DeepSpeed@ccfdb84</a>: 用户界面：https://github.com/microsoft/DeepSpeed-MII/pull/433
 针对上述链接的 MII 分支运行的 nv-a6000 ci 位于
 [此处](https://github.com/microsoft/DeepSpeed/actions/runs/81921...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1233616340951236638)** (10 messages🔥): 

- **ASPLOS 上的 PyTorch 团队**: PyTorch 团队将在 ASPLOS 进行教程演示，公告已发布，详细信息通过 [Twitter 链接](https://twitter.com/cHHillee/status/1784030920468783466) 提供。

- **Flash-Attention 更新警报**: Tri Dao 的新版本 **flash-attn 2.5.8** 已发布，并确认与 **PyTorch 2.3.0** 兼容。来源包括项目的 [GitHub](https://github.com/Dao-AILab/flash-attention) 和 [PyPI](https://pypi.org/project/flash-attn/) 页面。

- **关于 flash-attn 安装的疑问**: 讨论了 **flash-attn** 不需要本地 CUDA 构建的 pip 安装选项，以及为什么这不是默认设置。大家对预构建二进制文件与本地构建文件之间的潜在速度差异感到好奇。

- **`torch.compile` 的幕后机制**: 讨论了在配合 `torch.compile` 使用时，`torch.matmul`、`@` 和 `torch.nn.functional.linear` 之间的区别，参考了 [gpt-fast 博客文章](https://link.to.blogpost)。理解这些差异的建议是查看 **TORCH_LOGS** 输出。

- **PyTorch Profiler 难题**: 有人提问为什么 PyTorch 在矩阵乘法期间有时会启动 2 个 Kernel（如 Profiler 观察到的），并征求关于此行为的见解或理论。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/Dao-AILab/flash-attention">GitHub - Dao-AILab/flash-attention: Fast and memory-efficient exact attention</a>: 快速且内存高效的精确注意力机制。通过在 GitHub 上创建账号为 Dao-AILab/flash-attention 的开发做出贡献。</li><li><a href="https://pypi.org/project/flash-attn/">flash-attn</a>: Flash Attention: 快速且内存高效的精确注意力机制
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1233492304527098017)** (1 messages): 

- **提升代码清晰度和性能**: NVIDIA 的 C++ 团队将讨论将 **llm.c 移植到 llm.cpp**，承诺提供**更简洁、更快速的代码**。一场令人兴奋的额外讲座即将为社区开始。
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1233326463269474367)** (54 messages🔥): 

- **三元网络寻求高效 Matmul**: 一名成员发起头脑风暴，讨论如何使用 packed int64 对三元网络（Trinary Nets）执行矩阵乘法（matmul），以处理 32 个 2-bit 三元值而无需解包（unpacking）。他们假设 *掩码乘法方法（masked multiply approach）* 可以避免与解包相关的计算和内存开销，但实际的实现细节和收益仍处于理论阶段。

- **CUDA 中的打包与解包**: 另一个对话集中在处理打包值的优化上；一名成员指出在融合（fused）CUDA Kernel 中执行 pack 和 unpack 操作更具成本效益，但也有人对这种方法的可用性和复杂性表示担忧。

- **探索解包的替代方法**: 成员们讨论了创建直接对整数进行操作的行操作，无需解包，这可能会减少所需的计算次数。

- **用于性能优化的 Fused Kernels**：大家一致认为，虽然算子融合（kernel fusion）可能不会降低操作本身的成本，但它可以通过减少内存读取/拷贝来显著降低开销。对话演变为关于此类方法的技术可行性以及潜在计算效率提升的讨论。

- **揭秘 FlashAttention 的内部机制**：一位成员分享了对 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 仓库的见解，指出 `kernel_traits.h` 是 CUDA 中设置 traits 的核心组件，随后在 FlashAttention 中被使用。他们链接了一篇 [Colfax 研究文章](https://research.colfax-intl.com/adding-fp8-to-flashattention/)，讨论了 NVIDIA Hopper™ 架构上 FlashAttention 的 FP8 支持和布局一致性增强。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://research.colfax-intl.com/adding-fp8-to-flashattention/">使用 FP8 FlashAttention-2 提供 1 PFLOP/s 的性能</a>：我们最近发布了针对 NVIDIA Hopper&#x2122; 架构的 FlashAttention-2 前向传播实现的更新，其中包含许多新的优化和改进，包括……</li><li><a href="https://github.com/catid/bitnet_cpu">GitHub - catid/bitnet_cpu: 在 CPU 上进行 BitNet 推理的实验</a>：在 CPU 上进行 BitNet 推理的实验。通过创建账户为 catid/bitnet_cpu 的开发做出贡献。</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/kernel_traits.h">flash-attention/csrc/flash_attn/src/kernel_traits.h at main · Dao-AILab/flash-attention</a>：快速且内存高效的精确 Attention。通过创建账户为 Dao-AILab/flash-attention 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1234455593343783014)** (1 条消息): 

- **InstaDeep 正在招聘机器学习工程师**：**InstaDeep Research** 正在寻找对高性能 ML 工程充满热情并希望产生现实世界影响的机器学习工程师。该职位涉及 **Bio AI**、**Decision Making AI**，以及 **custom CUDA kernels**、**SOTA model architectures**、量化（Quantisation）和分布式训练（Distributed Training）等技术。[在此加入 InstaDeep 之旅](https://www.instadeep.com/job-offer/92900fa3-5501-4506-a63f-cebee958fc6f/)。

- **在 InstaDeep 培养创新**：InstaDeep 承诺为技术爱好者提供一个凝聚且具有挑战性的工作环境，为各行业的决策和技术产品做出贡献。实习机会也可以在[这里](https://www.instadeep.com/internships)探索。

- **InstaDeep 申请建议**：申请者可以申请 InstaDeep 的多个职位，但建议将申请限制在两个与其技能和资历匹配的紧密相关职位上。 

- **重新申请 InstaDeep**：之前申请过 InstaDeep 但未被录用的人员，如果距离上次申请已超过六个月，可以考虑重新申请。

**提到的链接**：<a href="https://www.instadeep.com/job-offer/92900fa3-5501-4506-a63f-cebee958fc6f/">职位空缺 | InstaDeep - 面向企业的决策 AI</a>：未找到描述

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1233320761469571103)** (12 条消息🔥): 

- **笔记本电脑上的 NVIDIA GPU 用于 CUDA**：通常认为使用带有 NVIDIA GPU 的笔记本电脑来**学习和测试 CUDA 代码**是可以接受的，但不建议用于实际的模型训练。
- **寻求 NCCL All-Reduce 资源**：一位成员正在寻找学习 NCCL 以实现 all-reduce kernel 的优质教程，但尚未收到建议。
- **用于 CUDA 学习的 Jetson Nano**：对于那些有兴趣学习 CUDA 的人，推荐将 Jetson Nano 作为一个有用的工具，尤其是配合备用显示器使用时。
- **解决 nvcc_plugin ModuleNotFoundError**：一位遵循 GitHub 教程的成员在使用 `%load_ext nvcc_plugin` 时遇到了 'nvcc_plugin' 的 "ModuleNotFoundError"。解决方案是跳过该步骤并改用 `%%writefile` 进行编译。
- **AMD GPU 性能咨询**：一位考虑从 **双 MI100 升级到 MI210** 的成员询问了 BF16 性能对比的见解，随后被引导至一个可能更专注于 AMD 资源的频道。
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1234409218170425366)** (2 条消息):

- **CUDA C++ 深度解析即将到来**：分享了一个名为 [**"Bonus Lecture: CUDA C++ llm.cpp"**](https://youtu.be/WiB_3Csfj_Q) 的 [YouTube 视频](https://youtu.be/WiB_3Csfj_Q)，提供了关于 CUDA C++ 的见解。描述中包含了一个指向 [Google Drive](https://drive.google.com/drive/folders/1T-t0d_u0Xu8w_-1E5kAwmXNfF72x-HTA?usp=sharing) 上的幻灯片链接。
- **计划稍后发布**：伴随 **CUDA C++ 讲座** 的幻灯片和代码目前暂不可用。

**提到的链接**：<a href="https://youtu.be/WiB_3Csfj_Q">Bonus Lecture: CUDA C++ llm.cpp</a>：幻灯片：https://drive.google.com/drive/folders/1T-t0d_u0Xu8w_-1E5kAwmXNfF72x-HTA?usp=sharing

  

---


**CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1233459187791429752)** (1 条消息): 

- **AO 迎来 CUDA 扩展支持**：正如一名成员通过 [PR 链接](https://github.com/pytorch/ao/pull/135) 所指出的，自定义 CUDA 扩展支持已集成到 **torchao** 中。此次集成允许开发者遵循模板，以确保其 kernel 与 `torch.compile` 无缝协作。

- **AO 寻求社区贡献**：对于热衷于编写 CUDA kernel 但讨厌打包过程的开发者，现在开放对 **torchao** 的贡献，特别是针对消费级 GPU 优化的 kernel。

**提到的链接**：<a href="https://github.com/pytorch/ao/pull/135">Custom CUDA extensions by msaroufim · Pull Request #135 · pytorch/ao</a>：这是 #130 的可合并版本 - 我必须进行一些更新，包括：除非使用 PyTorch 2.4+ 否则跳过测试，如果 CUDA 不可用则跳过测试，将 ninja 添加到开发依赖项，本地...

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1233356195759259711)** (2 条消息): 

- **挑战 LLM 上下文长度极限**：来自 [harmdevries.com](https://www.harmdevries.com/post/context-length/) 的一篇文章强调了大型语言模型 (LLM) **上下文长度 (context length)** 不断增加的趋势，已达到 65K token，其中 [FlashAttention](https://arxiv.org/abs/2205.14135) 等创新通过消除 GPU 显存瓶颈发挥了重要作用。
- **长上下文 LLM 的兴起**：许多尖端的长上下文 LLM 被发现是短上下文基础模型的微调版本；其中一个例子是 [Yarn-Llama-2-7B-128k](https://huggingface.co/conceptofmind/Yarn-Llama-2-7b-128k) 模型，它拥有 128K token 的上下文长度。

**提到的链接**：<a href="https://www.harmdevries.com/post/context-length/">In the long (context) run | Harm de Vries</a>：这不是二次方注意力的问题；而是缺乏长文本预训练数据的问题。

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1233802875759493161)** (4 条消息): 

- **伴随 'Critical Stop' 的放松氛围**：一位 Discord 成员分享了一个名为 "Critical Stop" 的 [YouTube 视频](https://youtu.be/QjZ4Ac0nbw8)，这是 Creatune 在 2024 年 3 月 23 日发布的一首由 DistroKid 提供的自动生成曲目。
- **注册机音乐怀旧**：分享了另一个 [YouTube 视频](https://www.youtube.com/watch?v=pp1mVv8lgGk)，标题为 "Dead Feelings - CORE - Power ISO 3.1kg Keygen Music"，为聊天带来了一些经典的注册机音乐。
- **通过遗传算法进化汽车**：发布了一个有趣的网页模拟项目 [Genetic Cars 2](https://rednuht.org/genetic_cars_2/)，其中遗传算法会在几代内将随机的双轮形状进化成汽车。
- **音乐算法第 9 条规则**：链接了 "Bad apple on everything" [YouTube 播放列表](https://youtube.com/playlist?list=PLajlU5EKJVdonUGTEc7B-0YqElDlz9Sf9&si=kzehICHc1YZTZpfR)，展示了在各种设备上播放的 'Bad Apple' 曲调的多功能性，基于第 9 条规则：如果它存在，就会有一个 "Bad Apple" 版本。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://rednuht.org/genetic_cars_2/">HTML5 Genetic Algorithm 2D Car Thingy - Chrome recommended</a>：未找到描述</li><li><a href="https://youtu.be/QjZ4Ac0nbw8">Critical Stop</a>：由 DistroKid 提供给 YouTube，Critical Stop · Creatune，Critical Stop ℗ Creatune Music，发布日期：2024-03-23，由 YouTube 自动生成。</li><li><a href="https://www.youtube.com/watch?v=pp1mVv8lgGk">Dead Feelings - CORE - Power ISO 3.1kg Keygen Music</a>：不是我的，显然属于 JimWalshified，原版位于 http://www.youtube.com/watch?v=-Cc09YsWDQs</li><li><a href="https://youtube.com/playlist?list=PLajlU5EKJVdonUGTEc7B-0YqElDlz9Sf9&si=kzehICHc1YZTZpfR">Bad apple on everything</a>：第 9 条规则 - 如果它存在，就在上面播放 Bad Apple
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1233425878973681747)** (714 条消息🔥🔥🔥): 

_

<ul>
  <li><b>FP16 与 BF16 训练潜力</b>：讨论围绕在不使用梯度缩放（gradient scaling）的情况下进行 FP16 模型训练的可行性，并推测其效果可能与 BF16 相当。分享了一个关于无需缩放的 FP8 训练研究链接，作为一种可能的类似策略。</li>
  <li><b>合并全 BF16 支持（包括 Layernorms）</b>：一个包含完整 BF16 支持（包括 Layernorms）的 PR 已合并，这可能会简化代码，但需要增加文件版本号以确保正确处理模型文件。</li>
  <li><b>数据类型加载与内存访问优化</b>：深入讨论了 CUDA kernels 中内存加载和存储的更好向量化方法，考虑使用 templates 以及像 <code>__ldcs</code> 这样用于流式内存访问的专用加载/存储指令。</li>
  <li><b>删除 Cooperative Groups 的使用</b>：讨论了从代码库中移除 Cooperative Groups (<code>cg</code>)，以提高跨平台兼容性并减少依赖，尽管它们是 CUDA 的一部分。</li>
  <li><b>性能提升与未来模型扩展</b>：值得注意的是，当前版本的 <code>train_gpt2cu</code> 在 token 处理速度上已经超过了 PyTorch 和优化后的 flashattention，这表明已经准备好将模型扩展到 gpt-large 的规模。</li>
</ul>

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://arxiv.org/abs/2310.18313">FP8-LM: Training FP8 Large Language Models</a>: 在本文中，我们探索了用于高效训练大语言模型 (LLMs) 的 FP8 低比特数据格式。我们的核心见解是，在 LLM 训练中，大多数变量（如梯度和优化器状态）...</li><li><a href="https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/associate_access_property.html">cuda::associate_access_property</a>: CUDA C++ 核心库</li><li><a href="https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html">cuda::memcpy_async</a>: CUDA C++ 核心库</li><li><a href="https://tenor.com/view/memory-no-memory-where-am-i-memories-harry-potter-gif-5385535">Dumbledore GIF - Memory No Memory Where Am I - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/karpathy/llm.c/pull/227):">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://godbolt.org/z/1hs47YzvY">Compiler Explorer - CUDA C++ (NVCC 12.3.1)</a>: #include &amp;lt;cuda_fp16.h&amp;gt;   template&amp;lt;class ElementType&amp;gt; struct alignas(16) Packed128 {     __device__ __forceinline__ Packed128() = default;     __device__ __forceinline__ exp...</li><li><a href="https://github.com/karpathy/llm.c/pull/250/files#diff-bf6b442957e5458cf8baab2a18039fdde86d74199a0864a79e7288fe55f31a98R56">Example for the dtype change for gelu kernels by ChrisDryden · Pull Request #250 · karpathy/llm.c</a>: 通过更改从内存读取的数据类型，在单次内存操作中可以读取高达 128 位的数据。对于受内存限制的 kernel，将所有内容包装起来是有益的...</li><li><a href="https://github.com/karpathy/llm.c/issues/292">delete use of cooperative groups in kernels · Issue #292 · karpathy/llm.c</a>: 我们在 kernel 中使用了大量的 cooperative groups 功能。这是一个额外的依赖，虽然可能带来些许便利，但代码也很可能在不使用它们的情况下编写...</li><li><a href="https://github.com/karpath">karpath - Overview</a>: GitHub 是 karpath 构建软件的地方。</li><li><a href="https://github.com/ka">ka - Overview</a>: :)。ka 有 3 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://developer.nvidia.com/nccl/nccl2-download-survey">Log in</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/280">as promised, cleanup enabled by padding :) by ngc92 · Pull Request #280 · karpathy/llm.c</a>: 必须修复 cublasLt 版本中的一个隐藏 bug，但现在它可以工作了</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2_fp32.cu#L1483">llm.c/train_gpt2_fp32.cu at master · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号，为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/293">yet another gelu by ngc92 · Pull Request #293 · karpathy/llm.c</a>: 更复杂的 Packet128 以实现更整洁的 kernel</li><li><a href="https://github.com/karpathy/llm.c/pull/295">Remove FloatN &amp; simplify adam/reduce with BF16 LayerNorms by ademeure · Pull Request #295 · karpathy/llm.c</a>: MULTI_GPU 路径未经测试，但其他一切似乎运行良好。我保留了每个张量的 "param_sizeof"，因为它在 test_gpt2.cu 等地方被使用，代码不多且可能有用...</li><li><a href="https://github.com/graphcore-research/out-of-the-box-fp8-training/tree/main">GitHub - graphcore-research/out-of-the-box-fp8-training: unit_scaling 库的演示，展示了如何轻松地调整模型以进行 FP8 训练。</a>: unit_scaling 库的演示，展示了如何轻松地调整模型以进行 FP8 训练。 - graphcore-research/out-of-the-box-fp8-training</li><li><a href="https://github.com/karpathy/llm.c/pull/270">clang-tidy by ngc92 · Pull Request #270 · karpathy/llm.c</a>: 在 make 文件中添加了 clang-tidy 文件和 clang-tidy 目标。由于 .cu 文件目前处于变动中，这仅针对 gpt2.c。我不确定应该启用哪些检查，但我认为...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2_fp32.cu#L2072)">llm.c/train_gpt2_fp32.cu at master · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号，为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/274">float4 with better vectorization for encoder_forward.cu by lancerts · Pull Request #274 · karpathy/llm.c</a>: 在 RTX 3070 上，Kernel 2 block_size 32 | 时间 0.2933 ms | 带宽 343.26 GB/s；block_size 64 | 时间 0.2099 ms | 带宽 479.50 GB/s；block_size 128 | 时间 0.1924 ms | 带宽 523.24 GB/s...</li><li><a href="https://github.com/karpathy/llm.c/pull/250">Example for the dtype change for</a>: 数据类型更改的示例</li>

<li><a href="https://github.com/karpathy/llm.c/pull/250">gelu kernels by ChrisDryden · Pull Request #250 · karpathy/llm.c</a>: 通过更改从内存读取的数据类型，在单次内存操作中可以读取高达 128 位的数据。对于受内存限制的 kernel，将所有...包装起来是有益的。</li><li><a href="https://github.com/karpathy/llm.c/pull/275">Removing Atomic Adds and adding memory coalescion by ChrisDryden · Pull Request #275 · karpathy/llm.c</a>: 该 PR 基于 GELU 内存合并 PR，本质上是对 backwards encoder 的重写，使用 shared memory 代替 atomic adds，并使用 Packed 结构体进行合并...</li><li><a href="https://github.com/karpathy/llm.c/pull/275#issuecomment-2082926859">Removing Atomic Adds and adding memory coalescion by ChrisDryden · Pull Request #275 · karpathy/llm.c</a>: 该 PR 基于 GELU 内存合并 PR，本质上是对 backwards encoder 的重写，使用 shared memory 代替 atomic adds，并使用 Packed 结构体进行合并...</li><li><a href="https://github.com/karpathy/llm.c/pull/265">load bf16 directly, and some &quot;quality of life&quot; handling of fp32/fp16/bf16 precisions by karpathy · Pull Request #265 · karpathy/llm.c</a>: 直接加载 bf16 权重的代码，并重新调整 Tensor 的位置，将 layernorm（fp32 类型）放在末尾。训练循环看起来运行正常，测试通过且 loss...</li><li><a href="https://github.com/karpathy/llm.c/pull/269">Enable multithreading in nvcc by ChrisDryden · Pull Request #269 · karpathy/llm.c</a>: 在 nvcc 中启用多线程。经本地测试，编译时间减少了 200ms。不幸的是，升级到 12.4 后我的编译时间慢了 2 倍，但至少这能让它快一点。</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L553">llm.c/train_gpt2.cu at master · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/265/files">load bf16 directly, and some &quot;quality of life&quot; handling of fp32/fp16/bf16 precisions by karpathy · Pull Request #265 · karpathy/llm.c</a>: 直接加载 bf16 权重的代码，并重新调整 Tensor 的位置，将 layernorm（fp32 类型）放在末尾。训练循环看起来运行正常，测试通过且 loss...</li><li><a href="https://github.com/karpathy/llm.c/pull/272">Full BF16 including layernorms by default (minimising number of BF16 atomics) by ademeure · Pull Request #272 · karpathy/llm.c</a>: 我添加了 4 个不同版本的 layernorm_backward_kernel，性能最好的是：Kernel 4（使用 atomicCAS，无 scratch，但多次舍入，因此数值精度可能较差）；Kernel 6...</li><li><a href="https://github.com/karpathy/llm.c/pull/289">fp16 buffers for ADAM by ngc92 · Pull Request #289 · karpathy/llm.c</a>: 第一个概念验证实现</li><li><a href="https://github.com/karpathy/llm.c/pull/264/files#diff-1dd4ce2b5299f353d184c5cd6f4e3b13a1a6491929d9fcf472fa18b87e20a0ccR123">enable padding in model export/import for nicer shapes by ngc92 · Pull Request #264 · karpathy/llm.c</a>: 对此的新尝试。由于我们直接从 Python 进行 padding，C 端的代码更加整洁。</li><li><a href="https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#cooperative-groups-functions">C++ Language Extensions &#8212; HIP 6.1.0 Documentation</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L917">llm.c/train_gpt2.cu at master · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/master/dev/cuda/classifier_fused.cu#L181">llm.c/dev/cuda/classifier_fused.cu at master · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1233708550438780958)** (19 条消息🔥):

- **AMD Instinct MI300X 备受关注**：AMD Instinct MI300X 被强调为专业服务器用途的重要产品，附有 [官方产品页面](https://www.amd.com/de/products/accelerators/instinct/mi300/platform.html) 以及关于其未来可用性的讨论。
- **探索 ROCm 以及 AMD 与 NVIDIA 的竞争**：频道讨论了 George Hotz 对 AMD 和 NVIDIA 的看法及处境，包括他对 AMD 性能和战略决策的思考。可以在 [tinygrad 页面](https://tinygrad.org/#tinybox) 关注相关动态。
- **寻求 ROCm 社区专业知识**：一位新成员请求介绍 ROCm HIP，并表示有兴趣参与社区驱动的讨论，探讨 AMD 的愿景以及为刚接触 AMD 生态系统的开发者提供的选项。
- **比较 AMD 和 NVIDIA 的产品**：社区成员将 AMD 最新的 PCIe 卡 Instinct MI210 与高端消费级显卡进行了比较，指出其与 NVIDIA 对应产品（如 RTX 4090）之间存在显著的价格差异。
- **AMD Windows 兼容性的演进及对 RDNA4 的期待**：对于 AMD 在其仓库中增加 Windows 构建测试，社区反应积极，同时也对 Computex 上即将发布的下一代 RDNA4 表示期待。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tinygrad.org/#tinybox">tinygrad: 一个简单而强大的神经网络框架</a>: 未找到描述</li><li><a href="https://www.runpod.io/amd-gpus">按需租用 AMD GPU</a>: 未找到描述</li><li><a href="https://github.com/nktice/AMD-AI">GitHub - nktice/AMD-AI: 在 Ubuntu 22.04 / 23.04 上为流行的 AI 工具提供基于 AMD (Radeon GPU) ROCm 的设置</a>: 在 Ubuntu 22.04 / 23.04 上为流行的 AI 工具提供基于 AMD (Radeon GPU) ROCm 的设置 - GitHub - nktice/AMD-AI: AMD (Radeon GPU) ROCm based setup for popular AI tools on Ubuntu 22.04 / 23.04</li><li><a href="https://www.techpowerup.com/gpu-specs/radeon-instinct-mi210.c3857">AMD Radeon Instinct MI210 规格</a>: AMD Aldebaran, 1700 MHz, 6656 Cores, 416 TMUs, 0 ROPs, 65536 MB HBM2e, 1600 MHz, 4096 bit
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[oneapi](https://discord.com/channels/1189498204333543425/1233802893786746880/1233805615210434570)** (22 条消息🔥): 

- **Intel 的 oneAPI：统一编程模型**：讨论强调了 Intel 的 oneAPI 作为一个异构计算平台，能够支持 CPU、GPU 和 FPGA，如 [Intel 关于 oneAPI 的官方文章](https://www.intel.com/content/www/us/en/developer/articles/technical/oneapi-what-is-it.html) 所述。oneAPI 承诺为开发者提供跨各种硬件的统一编程模型。
  
- **oneAPI 的跨厂商 GPU 支持**：Codeplay 发布 oneAPI 插件标志着重要的一步，允许开发者为 NVIDIA 和 AMD GPU 使用 SYCL™ 代码。[公告](https://codeplay.com/portal/press-releases/2022/12/16/codeplay-announces-oneapi-for-nvidia-and-amd-gpu-hardware.html) 和 [YouTube 上的教程视频](https://www.youtube.com/watch?v=fHZzm70hIdY) 为感兴趣的开发者提供了见解和资源。

- **oneAPI 生态系统扩展至主流框架和工具**：开发者可以在 GitHub 上发现众多的 oneAPI 资源和库，例如 oneDNN、与 PyTorch 和 TensorFlow 的集成，以及 Scikit-learn 的性能扩展。根据 ([oneAPI Toolkits 页面](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html))，Intel 的 oneAPI 工具包据称支持 Apple 的 ARM M1/M2/M3 和 FPGA。

- **Codeplay 对计算通用性的承诺**：[在 NVIDIA® GPU 上运行 SYCL™ 应用程序的指南](https://developer.codeplay.com/products/oneapi/nvidia/2023.0.0/guides/get-started-guide-nvidia) 和基于 RISC-V 的加速器平台的参考硅片示例 ([参考硅片概述](https://developer.codeplay.com/products/oneapi/construction-kit/2.0.0/guides/overview/reference-silicon/overview.html)) 表明了 Codeplay 在通用性方面取得的进展。

- **Intel 为下一代 GPU 做准备**：在聊天中，成员们表达了对 Intel 即将推出的 Battlemage GPU 系列的期待，有报告称其可能拥有 12Gb 的 VRAM，这引发了关于其是否适合 AI 相关任务的讨论。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.phoronix.com/news/Intel-IPEX-Arc-A-Series-Gfx">来自 Phoronix 的推文：Intel Extension For PyTorch 现在正式支持 Arc A-Series 显卡</a>：未找到描述</li><li><a href="https://www.phoronix.com/news/Intel-Extension-For-TensorFlow">来自 Phoronix 的推文：Intel Extension For TensorFlow 已发布 - 提供 Intel GPU 加速</a>：未找到描述</li><li><a href="https://developer.codeplay.com/products/oneapi/construction-kit/2.0.0/guides/overview/reference-silicon/overview.html">Codeplay 参考芯片概览 - 指南 - oneAPI Construction Kit - 产品 - Codeplay Developer</a>：未找到描述</li><li><a href="https://github.com/intel/intel-extension-for-pytorch">GitHub - intel/intel-extension-for-pytorch: 一个用于扩展官方 PyTorch 的 Python 包，可以轻松在 Intel 平台上获得性能提升</a>：一个用于扩展官方 PyTorch 的 Python 包，可以轻松在 Intel 平台上获得性能提升 - intel/intel-extension-for-pytorch</li><li><a href="https://www.youtube.com/watch?v=fHZzm70hIdY">适用于 Nvidia® 和 AMD® GPU 的 Codeplay® oneAPI 插件 | Intel Software</a>：您同样的 SYCL (C++) 代码现在不仅可以在 CPU 上运行，还可以通过 Codeplay® 的新插件在 Nvidia® 和 AMD® 的 GPU 上运行（使用相同的代码）...</li><li><a href="https://github.com/intel/scikit-learn-intelex">GitHub - intel/scikit-learn-intelex: Intel(R) Extension for Scikit-learn 是一种无缝加速 Scikit-learn 应用程序的方法</a>：Intel(R) Extension for Scikit-learn 是一种无缝加速 Scikit-learn 应用程序的方法 - intel/scikit-learn-intelex</li><li><a href="https://github.com/oneapi-src">oneAPI-SRC</a>：oneAPI 开源项目。oneAPI-SRC 拥有 57 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/oneapi-src/oneDNN">GitHub - oneapi-src/oneDNN: oneAPI Deep Neural Network Library (oneDNN)</a>：oneAPI Deep Neural Network Library (oneDNN)。通过在 GitHub 上创建账号为 oneapi-src/oneDNN 的开发做出贡献。</li><li><a href="https://github.com/intel/intel-extension-for-transformers">GitHub - intel/intel-extension-for-transformers: ⚡ 在几分钟内于您喜爱的设备上构建聊天机器人；为 LLM 提供 SOTA 压缩技术；在 Intel 平台上高效运行 LLM ⚡</a>：⚡ 在几分钟内于您喜爱的设备上构建聊天机器人；为 LLM 提供 SOTA 压缩技术；在 Intel 平台上高效运行 LLM ⚡ - intel/intel-extension-for-transformers</li><li><a href="https://www.oneapi.io/blog/bringing-nvidia-and-amd-support-to-oneapi/">为 oneAPI 带来 Nvidia® 和 AMD 支持 - oneAPI.io</a>：开发者可以编写 SYCL™ 代码并使用 oneAPI 通过免费的二进制插件针对 Nvidia* 和 AMD* GPU。今天对我来说是一个里程碑，因为 Codeplay® 正式发布了适用于 Nvidia 和 A... 的 oneAPI 插件。</li><li><a href="https://github.com/intel/intel-npu-acceleration-library">GitHub - intel/intel-npu-acceleration-library: Intel® NPU Acceleration Library</a>：Intel® NPU Acceleration Library。通过在 GitHub 上创建账号为 intel/intel-npu-acceleration-library 的开发做出贡献。</li><li><a href="https://developer.codeplay.com/products/oneapi/nvidia/2023.0.0/guides/get-started-guide-nvidia">为 NVIDIA GPU 安装 oneAPI - 指南 - 适用于 NVIDIA® GPU 的 oneAPI - 产品 - Codeplay Developer</a>：未找到描述
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1233342271001464866)** (856 条消息🔥🔥🔥): 

- **Pro Search 变慢引发关注**：用户报告 Perplexity 的 **Pro Search** 功能变慢了，搜索耗时高达 90 秒。他们在所有引擎（如 Mistral, Opus, GPT-4, Sonar 和 Sonnet）上都遇到了这种情况。该问题主要出现在 Web 客户端；移动端 App 似乎未受影响。

- **Claude 3 Opus 聊天版对比 API**：成员们正在讨论订阅 Claude 3 Opus 聊天版是否值得。一位用户的反馈表示它非常好，尽管没有提到 Claude 3 与 API 版本相比在功能或工具方面的具体细节。

- **对新模型的兴趣**：有人询问 **WizardLM 2** 和 **LLama-3 70B Sonar Large 32k** 模型未来是否会在 Perplexity 上提供。用户报告称这些模型在某些任务中表现优于 GPT-4，并对新模型是否会成为 Perplexity 产品的一部分表示好奇。

- **Opus 每日限制讨论**：提到 Perplexity 上 **Opus** 的每日限制让社区中的一些成员感到沮丧，尤其是他们认为 Opus 的质量正在下降。用户报告目前的限制是每 24 小时 50 次查询，并希望在该问题上增加透明度和更新。

- **对 Perplexity 计费问题的强烈不满**：一位用户在未获得预期免费试用的情况下被扣费，表达了不满。尽管按照 FAQ 中的步骤操作，但如果资金未退回，他们正考虑采取进一步行动。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/OpenAI/status/1783243000274932017?s=19">来自 OpenAI (@OpenAI) 的推文</a>：🤝😍 ↘️ 引用 Greg Brockman (@gdb) 的话：全球第一台 NVIDIA DGX H200 亲自交付给 OpenAI，并由 Jensen（黄仁勋）题词“推动 AI、计算和人类的进步”：</li><li><a href="https://duckduckgo.com/?q=DuckDuckGo&ia=chat>)">DuckDuckGo 上的 DuckDuckGo</a>：未找到描述</li><li><a href="https://flashcardfy.lol">Flashcardfy - 带有个性化反馈的 AI 抽认卡生成器</a>：使用提供个性化反馈的 AI 生成抽认卡，学习得更快、更聪明。</li><li><a href="https://fxtwitter.com/Gradient_AI_/status/1785030931407143040?t=U4_FdN9hNDaE9y432-lssQ&s=19">来自 Gradient (@Gradient_AI_) 的推文</a>：我们一直在努力研发 🔥 很高兴在 Hugging Face 上发布首个上下文长度超过 1M 的 Meta Llama-3 8B 模型——这是继我们发布的 160K 上下文长度模型之后的又一力作...</li><li><a href="https://tonsky.me/blog/js-bloat/">2024 年的 JavaScript 膨胀</a>：每个网站下载的 JavaScript 代码平均大小是多少？去探索并发现真相吧！</li><li><a href="https://devpost.com/software/hoo-wants-a-degree?ref_content=my-projects-tab&ref_feature=my_projects)">Hoo Wants A Degree?</a>：我们都知道大学顾问（由于缺乏更好的词）很糟糕。所以我们制作了 "Hoo Wants A Degree"！一个为试图弄清楚如何顺利毕业的校友们准备的 AI 学位规划工具...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1233341637669945415)** (28 messages🔥): 

- **探索 Perplexity 搜索链接**：成员们积极分享了各种 [Perplexity AI 搜索链接](https://www.perplexity.ai/search)，内容涵盖从国土安全部中的 AI 伦理到科幻未来的新闻，展示了广泛的兴趣和使用案例。
- **深入挖掘 Perplexity AI 的潜力**：一位成员回顾了之前与个人事务相关的 Perplexity 搜索链接，强调了过去几周搜索的准确性和实用性。
- **Scratchpad 功能测试**：另一位成员使用 Perplexity 链接在代码块中测试了 Scratchpad，表明了对平台功能的探索。
- **集合分享**：分享了一个 [BioExpress Sonnet 集合](https://www.perplexity.ai/collections/BioExpress-Sonnet-GoNYH8elQDWtI0Mu_QUckg)，展示了用户如何整理内容。
- **功能咨询与故障排除**：讨论包括对 Scratchpad 等功能的请求，以及对 Perplexity AI 能力的故障排除和探索。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/12333462270403678288)** (9 messages🔥): 

- **寻找合适的渠道**：由于发送给 enterprise@perplexity.ai 和 api@perplexity.ai 的邮件未收到回复，一位用户询问讨论 Perplexity AI 企业级 API 使用的合适沟通渠道。另一位用户建议保持耐心，指出回复时间可能在 1 到 3 周之间。

- **了解在线模型指南**：一位新成员询问关于在 sonar-small-online 和 sonar-medium-online 等在线 LLM 中仅使用单轮对话且避免使用 system prompt 的说明。另一位用户提供了澄清，表示这些模型更倾向于单轮交互，且无法访问 system prompt。

- **Harpa 配置咨询**：一位用户向社区询问是否有人成功将 Harpa 直接配置到 Perplexity API。

- **关于通过 API 获取源 URL 的疑问**：一位成员想知道是否可以通过 API 访问源 URL，因为他们在路线图文档页面上找不到相关信息。他们被引导去填写一份获取引用（citations）访问权限的表单，但提到之前曾因仅限已获融资的初创公司而被拒绝。

- **make.com 上的模型选择之谜**：关于 make.com 上缺少 Llama-3 模型和 Mixtral 8x22b 选项的问题，寻求其他用户的见解。

**提到的链接**：<a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api 表单</a>：使用 Typeform 将数据收集转变为一种体验。创建精美的在线表单、调查、测验等。免费试用。

  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1233326318087569478)** (922 messages🔥🔥🔥):

- **解决 SDXL 和 Forge UI 问题**：用户讨论了 SDXL 和 Forge UI 的问题，包括图像预览困难以及 Forge 可能被弃用的情况。建议查看 GitHub issues，例如[这个报告的问题](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/10132)，并尝试在 webui.bat 文件中使用 `--no-gradio-queue` 等参数。

- **Stable Diffusion 3 期待**：关于 Stable Diffusion 3 的发布日期一直存在推测，一些用户引用了 CivitAI 的时事通讯，指出其将于 5 月底发布。用户对开放权重发布以及 SD3 是否名副其实表示担忧，同时[链接的文章](https://civitai.com/articles/5069)讨论了 Pony Diffusion V7 的更新以及 Altman 针对开源行为的潜在影响。

- **AI 生成艺术的变现**：用户讨论了在激烈竞争中销售 SFW AI 生成艺术的困境，而 Civitai 等平台上的 NSFW 内容创作者则更为成功。有人建议 AI 女友应用更有利可图，并提到人们对微调 Stable Cascade 等模型缺乏兴趣。

- **讨论 AI 训练的工具和方法**：出现了关于 AUTOMATIC1111 之外工具的讨论，建议使用 dreambooth 和 kohya_ss 进行模型训练。此外，还辩论了在训练数据中包含艺术家名字的实用性和伦理问题。

- **其他咨询和讨论**：用户询问了从 Text to Speech 工具到模型微调细节等各种话题。还有关于隐喻式“下载”显卡的幽默，以及对 SD 是否可以在没有 Prompt 的情况下生成图像的好奇。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://proximacentaurib.notion.site/e28a4f8d97724f14a784a538b8589e7d?v=ab624266c6a44413b42a6c57a41d828c">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合为一的新工具。为您和您的团队提供的一体化工作空间。</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md">LICENSE.md · stabilityai/stable-diffusion-xl-base-1.0 at main</a>：未找到描述</li><li><a href="https://civitai.com/articles/5069">迈向 Pony Diffusion V7 | Civitai</a>：大家好，我很高兴能分享我们即将推出的 V7 的进展更新，以及对 V6 的回顾分析。V6 获得的认可...</li><li><a href="https://huggingface.co/xtuner/llava-llama-3-8b-v1_1">xtuner/llava-llama-3-8b-v1_1 · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/see-you-shocked-face-future-gif-14292131">See You Shocked Face GIF - See You Shocked Face Future - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.instagram.com/reel/C3kqmuToOhH/?igsh=M2Flc296ZnVrNjc3"> Instagram 上的 DodoNemoCleo："太神奇了 😱🤯😵 赶快和朋友们一起尝试吧 👀 💖 如果你是爱猫人士，请关注 @dodonemocleo_cat ❤️ .</a></li>
</ul>
</div>

.
#cat #catlover #cats_of_world #cats_of_instagram #catstagram #cats #catsofinstagram #fun #funny #game #games #challenge #beautiful #cute #cursed #silly #laugh #friends #bestfriends #joke #fyp #instagram #kitten #kitty #silly #viral #viralvideos #trending #trendingreels #gato #funnymemes&quot;</a>: 538K 点赞，7,269 条评论 - dodonemocleo_cat 于 2024 年 2 月 20 日发布：&quot;太神奇了 &#x1f631;&#x1f92f;&#x1f635; 现在就和朋友们一起尝试吧 &#x1f440; &#x1f497; 如果你... 请关注 &#064;dodonemocleo_cat</li><li><a href="https://civitai.beehiiv.com/p/multiaccount-switching-civitai-link-expanded-plus-enter-win-2000-worth-prizes-legendary-landscapes-c">多账号切换，Civitai Link 扩展，此外参加我们正在进行的 Legendary Landscapes 竞赛，赢取价值超过 2,000 美元的奖品！</a>: 未找到描述</li><li><a href="https://stable-diffusion-art.com/samplers/#DPM_solvers">Stable Diffusion 采样器：全面指南 - Stable Diffusion Art</a>: AUTOMATIC1111 中提供了许多采样方法。Euler a, Heun, DDIM... 什么是采样器？它们是如何工作的？它们之间有什么区别？哪一个</li><li><a href="https://huggingface.co/deadman44/SDXL_Photoreal_Merged_Models#potest2">deadman44/SDXL_Photoreal_Merged_Models · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=ylHTojkioWY">如何安装 Stable Diffusion Automatic1111 WebUI 最新版本 2024（设置指南）Easy Diffusion</a>: 欢迎来到 MunKaw 频道！在本视频教程中，我们将带您进入人工智能的世界。我们很高兴能以一个教程开始我们的旅程...</li><li><a href="https://github.com/huggingface/diffusers/tree/main/examples/dreambooth">diffusers/examples/dreambooth at main · huggingface/diffusers</a>: 🤗 Diffusers：用于 PyTorch 和 FLAX 中图像和音频生成的先进扩散模型。- huggingface/diffusers</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge/pull/692">由 altoiddealer 恢复 '/controlnet/control_types' API 端点 · Pull Request #692 · lllyasviel/stable-diffusion-webui-forge</a>: 恢复了 '/controlnet/control_types' API 端点，这对于任何通过 API 使用 ControlNet 的人来说都非常有用。描述：我最近在主要的 ControlNet 扩展上提交了一个 Issue...</li><li><a href="https://www.youtube.com/watch?v=_oI_B0OBgVw">Coca-Cola x Marvel: The Heroes</a>: 见证 Coca-Cola 和 Marvel 以从未见过的方式集结，营救一名漫画店员工。</li><li><a href="https://github.com/AUTOMATIC111">Automatic111 - 概览</a>: GitHub 是 Automatic111 构建软件的地方。</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/10132\">Issues · AUTOMATIC1111/stable-diffusion-webui</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/megvii-research/HiDiffusion">GitHub - megvii-research/HiDiffusion</a>: 通过在 GitHub 上创建账户，为 megvii-research/HiDiffusion 的开发做出贡献。</li><li><a href="https://github.com/ToTheBeginning/PuLID">GitHub - ToTheBeginning/PuLID</a>: 通过在 GitHub 上创建账户，为 ToTheBeginning/PuLID 的开发做出贡献。</li><li><a href="https://github.com/nerve-sparks/iris_android">GitHub - nerve-sparks/iris_android</a>: 通过在 GitHub 上创建账户，为 nerve-sparks/iris_android 的开发做出贡献。</li><li><a href="https://github.com/JarodMica/ai-voice-cloning">GitHub - JarodMica/ai-voice-cloning</a>: 通过在 GitHub 上创建账户，为 JarodMica/ai-voice-cloning 的开发做出贡献。</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/comfyanonymous/ComfyUI#installing">GitHub - comfyanonymous/ComfyUI：最强大且模块化的 Stable Diffusion GUI、API 和后端，具有图形/节点界面。</a>: 最强大且模块化的 Stable Diffusion GUI、API 和后端，具有图形/节点界面。- comfyanonymous/ComfyUI
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1233320042586832926)** (472 条消息🔥🔥🔥):

- **AI 辅助作业**：一位用户对 **Meta-Llama-3-8B-Instruct-Q5_K_M.gguf** 模型在 M1 MacBook Pro 上的表现表示惊讶，强调了它在赶作业方面的帮助。
- **探索模型性能**：围绕 **34B 和 70B** Code Llama 等模型之间的性能差异展开了讨论。建议用户在选择模型时考虑量化（quantization）类型，以匹配其现有的硬件。
- **将 LLM 集成到 Discord 机器人**：多位用户讨论了如何通过 **Groq API** 利用 **Llama3** 模型创建 Discord 机器人，实现提取相关消息和进行 Wikipedia 搜索等功能。
- **LLM 模型与 API 使用**：新用户寻求关于使用本地大语言模型（LLMs）的建议，而其他用户分享了关于使用 **LM Studio** 进行私有模型部署的 **YouTube 教程**等资源。
- **在本地训练和微调模型**：一场关于离线模型训练的可行性和硬件要求的讨论展开了。用户们权衡了实际操作性，其中一人分享了在 M3 Max 设备上尝试微调（finetune）的个人经历，预计训练时间长达一整周。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/rocm">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM 模型 VRAM 计算器 - NyxKrage 的 Hugging Face Space</a>：暂无描述</li><li><a href="https://huggingface.co/ChristianAzinn/acge_text_embedding-gguf">ChristianAzinn/acge_text_embedding-gguf · Hugging Face</a>：暂无描述</li><li><a href="https://huggingface.co/google/siglip-so400m-patch14-384">google/siglip-so400m-patch14-384 · Hugging Face</a>：暂无描述</li><li><a href="https://www.theregister.com/2024/03/28/ai_bots_hallucinate_software_packages/">AI 机器人幻觉出软件包，开发者误下载</a>：只需留意机器学习虚构的库，并用实际的恶意代码将其变为现实。不，等等，别那么做</li><li><a href="https://tenor.com/view/dr-austin-powers-evil-one-gif-14681923667046200996">Dr Austin GIF - Dr Austin Powers - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://lmstudio.ai/docs/local-server">本地 LLM 服务器 | LM Studio</a>：你可以通过在 localhost 上运行的 API 服务器，使用你在 LM Studio 中加载的 LLMs。</li><li><a href="https://tenor.com/view/captain-obvious-thanks-yes-sir-gif-27076523">Captain Obvious GIF - Captain Obvious Thanks - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/TheBloke/dolphin-2.5-mixtral-8x7b-GGUF">TheBloke/dolphin-2.5-mixtral-8x7b-GGUF · Hugging Face</a>：暂无描述</li><li><a href="https://huggingface.co/aspire/acge_text_embedding">aspire/acge_text_embedding · Hugging Face</a>：暂无描述</li><li><a href="https://youtu.be/ISqedkU_tJ4">笔记本就能跑的私有化大模型安装部署最佳教程：机密合同，隐私文档，核心代码AIGC最佳解决方案</a>：《笔记本就能跑的私有化大模型安装部署最佳教程》机密合同，隐私文档，核心代码AIGC最佳解决方案, 含GPU/CPU速度对比大家平时在工作中经常有机密合同，隐私文档，核心代码需要AI帮忙处理，但苦于信息安全规定不能发给chatgpt，这种情况以前大家只能自己人工写，现在有了私有化大模型，大家就可以放心地让AI帮您写...</li><li><a href="https://lmstudio.ai/">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://youtu.be/ySwJT3Z1MFI?si=qFfek8gTGXVJWoxB">在 Groq Playground 和 API 上免费体验极速 LLAMA-3</a>：了解如何开始使用 Groq API 上的 LLAMA-3，这是目前市场上任何 API 中最快的推理速度。了解如何使用 Gro...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - 深入探索</a>：暂无描述</li><li><a href="https://huggingface.co/qresearch/llama-3-vision-alpha">qresearch/llama-3-vision-alpha · Hugging Face</a>：暂无描述</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit">unsloth/llama-3-8b-Instruct-bnb-4bit · Hugging Face</a>：暂无描述</li><li><a href="https://huggingface.co/ChristianAzinn">ChristianAzinn (Christian Zhou-Zheng)</a>：暂无描述</li><li><a href="https://www.pinecone.io/learn/series/rag/rerankers/">重排序器与两阶段检索 | Pinecone</a>：暂无描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1233406452014649475)** (219 条消息🔥🔥): 

- **斯坦福的 Octopus v2 困扰用户**：在 **🤖-models-discussion-chat** 频道中，有人询问如何在 LM Studio 或本地手机/电脑上运行斯坦福的 Octopus v2，但未提供明确的解决方案，仅指出了运行利用 Function Calling 的 Agent 模型所涉及的复杂性。

- **LLAMA 模型胡言乱语令用户沮丧**：讨论表明 262k 和 64k 的 **Llama 8b** 模型往往会胡言乱语，由于 instruct fine tuning 的原因，表现出基础版 **Llama 3** 的行为。用户分享了初次使用这些模型时的经验和期望。

- **fp16 "phi3" 与 LM_Studio 的兼容性问题**：对话集中在 "phi3" 模型与不同版本 **LM_Studio** 的兼容性上，提到虽然 LM_Studio 2.20 (ROCm Preview) 不支持 "phi3"，但可能需要更新的 0.2.21 版本。对于想要使用尚未被该 studio 支持的模型的心情，大家表示了同情。

- **探索特定任务的 AI 工具**：成员们询问了用于搜索特定任务 AI 工具的网站，例如生成音乐或在不同照片中寻找相似场景。建议包括使用 [Pinokio Computer](https://pinokio.computer/) 和 [Future Tools](https://www.futuretools.io/)。

- **关于 LLaMA 3 是否包含联网功能的辩论**：一位用户在注意到模型提供了当前新闻信息后，询问 LLaMa 3 是否包含联网功能，但另一位用户澄清说，鉴于这些模型没有联网功能，它们很可能是在产生幻觉（hallucinate）。

- **运行 Snowflake AI 的 Arctic 仍是一个遥远的梦想**：一位成员对 Snowflake **Arctic** 模型很感兴趣，但讨论得出的结论是，由于该模型体积巨大，在没有大量系统资源的情况下，目前期望在本地运行是不现实的。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit>">安装 NVIDIA Container Toolkit &mdash; NVIDIA Container Toolkit 1.15.0 文档</a>: 未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/commit/c9b8888921fe528fe4be053258f48b952281bb1b">fix(root): 将 system 替换为 user 以提升生成体验。 · microsoft/Phi-3-mini-128k-instruct at c9b8888</a>: 未找到描述</li><li><a href="https://huggingface.co/Lewdiculous/Eris-Prime-Punch-9B-GGUF-IQ-Imatrix">Lewdiculous/Eris-Prime-Punch-9B-GGUF-IQ-Imatrix · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Snowflake/snowflake-arctic-instruct?_fsi=v2MrQoFW">Snowflake/snowflake-arctic-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://pinokio.computer/">Pinokio</a>: AI 浏览器</li><li><a href="https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi-3-tutorial.md">onnxruntime-genai/examples/python/phi-3-tutorial.md at main · microsoft/onnxruntime-genai</a>: onnxruntime 的生成式 AI 扩展。通过在 GitHub 上创建账户来为 microsoft/onnxruntime-genai 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6868">支持 Apple 的 OpenELM · Issue #6868 · ggerganov/llama.cpp</a>: 前提条件：在提交 issue 之前，请先回答以下问题。我正在运行最新代码。由于开发进度非常快，目前还没有标记版本。我...</li><li><a href="https://huggingface.co/internlm/internlm-xcomposer2-vl-7b-4bit">internlm/internlm-xcomposer2-vl-7b-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/gokayfem/ComfyUI_VLM_nodes">GitHub - gokayfem/ComfyUI_VLM_nodes: 用于 Vision Language Models、Large Language Models、图像转音乐、文本转音乐、一致且随机的创意提示词生成的自定义 ComfyUI 节点</a>: 用于 Vision Language Models、Large Language Models、图像转音乐、文本转音乐、一致且随机的创意提示词生成的自定义 ComfyUI 节点 - gokayfem/ComfyUI_VLM_nodes</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6849">支持 Phi-3 模型 · Issue #6849 · ggerganov/llama.cpp</a>: Microsoft 最近发布了三种变体（mini, small 和 medium）的 Phi-3 模型。我们能否添加对这一新模型系列的支持。</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6745">由 pcuenca 支持 Llama 3 转换 · Pull Request #6745 · ggerganov/llama.cpp</a>: 分词器（tokenizer）是 BPE。</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/1684">由 ikawrakow 提交的 k-quants · Pull Request #1684 · ggerganov/llama.cpp</a>: 内容：此 PR 按照 #1240 和 #1256 的建议，增加了一系列 2-6 bit 量化方法以及混合量化。提供了 Scalar、AVX2、ARM_NEON 和 CUDA 的实现。原因：这是...</li><li><a href="https://www.futuretools.io/">Future Tools - 寻找满足你需求的精准 AI 工具</a>: FutureTools 收集并整理了所有最优秀的 AI 工具，让你也能变得超乎常人！
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1233357312895488081)** (5 条消息): 

- **更新后 Phi-3 mini 表现异常**: 一位用户报告称，在更新到 0.2.21 版本后，**phi-3 mini** 模型开始输出乱码，而之前的 0.2.20 版本没有问题。该问题是在使用 GitHub 仓库中官方的 LM Studio phi-3 配置时发现的。
- **出于诊断目的请求截图**: 针对 phi-3 mini 的问题，另一位用户请求提供整个应用的截图以进一步诊断问题。
- **P100 性能不一致与显示器灰尘**: 一位用户建议，如果除了从 0.2.20 更新到 0.2.21 之外没有其他变化，那么该问题可能是一个值得在另一个频道提交的回归错误（regression error）。开玩笑地，他们还建议清理显示器上的灰尘。
- **LM Studio 应用神秘崩溃**: 一位用户描述了自几次更新以来 LM Studio 应用出现的崩溃情况，在调整窗口大小或在程序内导航时应用会意外关闭。他们分享了系统配置，包括 Windows 10 Pro、Ryzen 7 5800X、RTX 3090 和 64GB RAM DDR4。
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1234136264312225792)** (4 条消息): 

- **探索与 PDF 交互的方法**: 一位成员建议直接将 PDF 的内容连同问题一起粘贴到聊天消息中，前提是模型的 context length 支持。

- **文档对话的 RAG 解决方案**：提供了一个替代方案，即使用 **Retrieve and Generate (RAG)** 解决方案（如 AnythingLLM），通过将 **LM Studio** 作为 API server 运行，并将 AnythingLLM 指向该 API。

- **PDF 长度的实际考量**：在管理 PDF 文档方面，PDF 的长度是一个令人关注的问题，这涉及到直接让语言模型针对 PDF 进行提问的可行性。

---

**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1233347794572415017)** (119 messages🔥🔥): 

- **VRAM：LLM 硬件的基石**：成员们讨论了 VRAM 是运行语言模型的关键因素，**16GB** 是最低建议，一名成员正准备通过订购第二块 NVIDIA **4060 (ti - 16gb)** 来加入 **32GB VRAM 俱乐部**。

- **剖析 GPU 兼容性与性能**：关于利用 Nvidia 等**现代架构 GPU** 的重要性以及确保充足 VRAM（被强调为 LLM 考虑的**核心**）进行了深入对话。一位成员分享了在配备 **3060 GPU** 和 **16GB RAM** 的台式机上运行不同尺寸模型的具体细节。

- **强制使用独立显卡而非集成显卡**：一位成员寻求关于配置 **LM Studio** 以使用独立显卡而非默认使用 CPU 集成显卡的帮助。建议使用禁用并重新启用 GPU offload 以及使用 `CUDA_VISIBLE_DEVICES` 和 `tensor_split` 等设置来更好地利用独立 GPU。

- **多 GPU 与大模型难题**：一位成员询问了 LM Studio 使用两块 GPU（**4090 & 3090**）的效果，以及软件是否会自动在它们之间拆分模型。会议指出，模型可以在 GPU 之间拆分，这会导致数据传输时间增加，但 **NVLink** 等技术有助于优化多 GPU 间的性能。

- **针对不同硬件配置进行优化**：用户交流了关于最佳硬件配置的经验和推测。分享了一个在老旧的 **GTX1070 8Gb** GPU 上成功运行多个模型的轶事，证明了即使对于要求较低的专业用例，它也是可行的。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/thumbs-up-nice-well-done-approve-good-job-gif-13666522">Thumbs Up Nice GIF - Thumbs Up Nice Well Done - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/fear-and-loathing-in-las-vegas-taste-drop-gif-17307682">Fear And Loathing In Las Vegas Taste GIF - Fear And Loathing In Las Vegas Taste Drop - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/jon-stewart-eat-eating-popcorn-watching-gif-3094746547306242594">Jon Stewart Eat GIF - Jon Stewart Eat Eating - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://stackoverflow.com/questions/40346442/stop-opencl-support-for-a-gpu">停止对某个 GPU 的 OpenCL 支持</a>: 我的机器上安装了两个 GPU。我正在使用一个使用 OpenCL 加速的库，它只支持一个 GPU 且不可配置。我无法告诉它我想要哪一个。似乎...</li><li><a href="https://www.ebay.com/itm/355545860836?mkcid=16&mkevt=1&mkrid=711-127632-2357-0&ssspo=j2bowcjltc6&sssrc=2047675&ssuid=&widget_ver=artemis&media=COPY">NVIDIA Tesla T4 16GB GDDR6 显卡 (900-2G183-0000-001) | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.com/itm/355545860836?mkcid=16&mkevt=1&mkrid=711-127632-2357-0&ssspo=j2bowcjltc6&sss">NVIDIA Tesla T4 16GB GDDR6 显卡 (900-2G183-0000-001) | eBay</a>: 未找到描述
</li>
</ul>

</div>

---

**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1233863483477594152)** (1 messages): 

- **服务器错误消息排查**：一位成员询问关于服务器错误的修复方法，错误提示为：*“[ERROR] [Server Error] {"title":"'messages' array must only contain objects with a 'content' field that is not empty"}”*。在该询问之后没有进一步的讨论或提供的解决方案。

---

**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/)** (1 messages): 

ahakobyan.: 我们也能知道吗？

---

**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1233583151763296277)** (4 messages): 

- **RX 6700 与 LM Studio ROCm 的兼容性咨询**：一位成员询问 **LM Studio ROCm** 是否支持 **RX 6700**（非 XT 版本），并请求协助排查日志错误。他们分享了一个错误输出，显示模型操作失败，但没有具体的解决建议。

- **LM Studio ROCm 限制说明**：另一位参与者澄清说，**LM Studio 不支持 RX 6700** (non-XT)，因为它依赖于 **HIP SDK**，而该 SDK 仅与特定的 AMD 显卡兼容。他们提到 *KoboldAI* 利用一种变通方法在不支持的架构上运行。
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1233419149137543212)** (9 messages🔥): 

- **Snowflake Arctic**：Snowflake AI 研究团队推出了 [Snowflake Arctic](https://www.youtube.com/watch?v=nV6eIjnHEH0)，这是一款专注于提供企业级 AI 解决方案的大语言模型 (LLM)，强调成本效益。
- **分享了未说明内容的 YouTube 视频**：链接了一个 YouTube 视频，没有提供额外的上下文或描述。这是那个[神秘视频](https://www.youtube.com/watch?v=oDGzMF8CiQU)。
- **Llama 3 网页浏览 Agent**：分享了一个演示网页浏览 Agent 的视频，标题为 "Llama 3 Web Browsing Agent with Langchain and Groq"，展示了使用 Llama 3 结合 Langchain 和 Groq 的实现。[观看视频](https://www.youtube.com/watch?v=au6WQVEgGQo)。
- **Gorillaz 的热门视频**：提供了 Gorillaz 的 "Feel Good Inc." 官方视频的 YouTube 链接。粉丝可以在[这里](https://youtu.be/HyHNuVaZJ-k?list=PLtKoi37ubAW0tYWi9d7yx9KrWbgVn7ZTq&t=41)观看高清视频。
- **MatrixBridge 推出 Skrapy**：MatrixBridge 正在开发 Skrapy，这是一款用于简化数据收集和抓取的 AI Agent，目前处于 Alpha 阶段，并为早期用户提供候补名单。欲了解更多信息或加入社区，请访问 [MatrixBridge 的 Skrapy 页面](https://www.skrapy.matrixbridgeai.com/)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.skrapy.matrixbridgeai.com/">Skrapy | AI Data Agent</a>：Skrapy 是一款数据抓取可视化 AI Agent。</li><li><a href="https://www.youtube.com/watch?v=nV6eIjnHEH0">Snowflake Arctic: The Best LLM for Enterprise AI</a>：今天，Snowflake AI 研究团队激动地推出了 Snowflake Arctic，这是一款顶级的企业级 LLM，它推动了成本效益的前沿...</li><li><a href="https://www.youtube.com/watch?v=au6WQVEgGQo">Llama 3 Web Browsing Agent with Langchain and Groq</a>：我们将探讨如何使用 Llama 3 结合 Langchain 和 Groq 实现网页浏览 #python #pythonprogramming #llm #ml #ai #aritificialintelligence #la...</li><li><a href="https://youtu.be/HyHNuVaZJ-k?list=PLtKoi37ubAW0tYWi9d7yx9KrWbgVn7ZTq&t=41">Gorillaz - Feel Good Inc. (Official Video)</a>：Gorillaz 经典曲目 Feel Good Inc. 的官方高清视频。关注 Gorillaz：http://gorillaz.com http://facebook.com/Gorillaz http://twitter.com/Gorill...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1233967391247962212)** (15 messages🔥): 

- **英特尔的 AI 雄心揭晓**：英特尔首席执行官 Pat Gelsinger 讨论了公司的季度业绩，强调了代工业务的增长以及 PC 对 AI 的需求。视频可以在 YouTube 上观看，标题为 ["Intel CEO Gelsinger on Q1 Earnings, Foundry Business, AI."](https://youtube.com/watch?v=bWcN4a62i0Q&si=nbOPMlMFsbWEVAoG)

- **罗技增强 AI 易用性**：罗技发布了 AI Prompt Builder，这是一款集成在其鼠标中的工具，旨在促进更快速、更流畅的 ChatGPT 提示词编写。在 YouTube 视频 ["Introducing Logi AI Prompt Builder - Your shortcut to AI fluency."](https://www.youtube.com/watch?v=jcCTTbEvU4g) 中体验其带来的便利。

- **用于高效 AI 模型的量化 Embedding**：一位成员分享了其微调版本的 Hugging Face 模型链接，这些模型允许将图像和文本 Embedding 有效地压缩为二进制格式。感兴趣的人可以在 [binary-siglip-text](https://huggingface.co/carsonpoole/binary-siglip-text) 和 [binary-siglip-vision](https://huggingface.co/carsonpoole/binary-siglip-vision) 探索这些模型。

- **揭开 AI 拒绝机制之谜**：来自 ML Alignment & Theory Scholars Program 的研究表明，LLM 中的拒绝行为由 Residual Stream 中的单一方向控制，即将发表的一篇论文将深入探讨这一主题。初步研究结果可以在 Alignment Forum 的帖子 ["Refusal in LLMs is mediated by a single direction."](https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) 中查看。

- **立法威胁开源 AI 发展**：Jeremy Howard 表达了对加州 SB-1047 法案可能严重损害初创企业、创新和开源安全的担忧。在 Howard 的回复中阅读他对该事项的完整看法以及该立法的潜在影响：[Answer.ai 关于 SB-1047 的帖子](https://x.com/jeremyphoward/status/1784717268368367665)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.affuture.org/post/9-context/">关于 SB 1047 的行动呼吁</a>：加州立法者在 Effective Altruism 活动人士的影响下，正试图通过一项对开源 AI 和整个科技行业具有灾难性的法案。SB 1047 创建了一个...</li><li><a href="https://arxiv.org/abs/2404.16811">让你的 LLM 充分利用上下文</a>：虽然许多当代的 Large Language Models (LLMs) 可以处理长输入，但它们在充分利用长上下文中的信息方面仍面临困难，即所谓的 lost-in-the-middle 挑战。我们...</li><li><a href="https://openai.com/research/language-models-can-explain-neurons-in-language-models">语言模型可以解释语言模型中的神经元</a>：我们使用 GPT-4 自动为大型语言模型中的神经元行为编写解释并对这些解释进行评分。我们发布了这些（不完美的）解释和评分的数据集...</li><li><a href="https://x.com/jeremyphoward/status/1784717268368367665?s=46&t=bL0EKkuCqv4FWSLQ7lV-2w">来自 Jeremy Howard (@jeremyphoward) 的推文</a>：有一项新法案 SB-1047 “前沿人工智能模型安全创新法案”。我认为它可能对初创公司、美国创新、开源...造成巨大伤害。</li><li><a href="https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms">重温 GPT-1：点燃 LLMs 之火的火花</a>：全面回顾 GPT-1 对现代 LLMs 发展的贡献。</li><li><a href="https://www.primeintellect.ai/blog/our-approach-to-decentralized-training">去中心化训练的最前沿技术</a>：本文探讨了各种新型的去中心化训练方法，以及它们如何实现在全球分布的 GPU 上进行有效的 AI 模型训练。</li><li><a href="https://www.youtube.com/watch?v=jcCTTbEvU4g">介绍 Logi AI Prompt Builder - 提升 AI 熟练度的捷径</a>：介绍 Logi AI Prompt Builder，这是我们最新的工具，可帮助您更快、更流畅地提示 ChatGPT，同时保持工作流。从...中选择</li><li><a href="https://youtube.com/watch?v=bWcN4a62i0Q&si=nbOPMlMFsbWEVAoG">Intel 首席执行官 Gelsinger 谈第一季度财报、代工业务、AI</a>：Intel 首席执行官 Pat Gelsinger 讨论了公司的季度业绩、代工业务的进展、对 AI PC 的需求，以及他在 AI 产品中看到的优势...</li><li><a href="https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction">LLMs 中的拒绝是由单一方向介导的 — AI Alignment Forum</a>：这项工作是 Neel Nanda 在 ML Alignment & Theory Scholars Program - 2023-24 冬季班中的一部分，由……共同指导。</li><li><a href="https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direc">LLMs 中的拒绝是由单一方向介导的 — AI Alignment Forum</a>：这项工作是 Neel Nanda 在 ML Alignment & Theory Scholars Program - 2023-24 冬季班中的一部分，由……共同指导。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1233319997233696779)** (566 条消息🔥🔥🔥): 

- **LLaMA-3 微调困难？**：用户正在讨论 LLaMA-3 在微调后无法正确生成 EOS token 的问题。建议是在生成过程中对 token 128009 添加停止标准（stop criterion），并提供了指向一个有用的 [Huggingface transformer 停止标准仓库](https://github.com/nestordemeure/stop_word) 的进一步见解。

- **GPT-2 Chatbot 之谜**：关于 `gpt2-chatbot` 的能力存在困惑，尽管名字如此，但它似乎与 GPT-4 相关，知识截止日期为 2023 年 11 月。讨论指出它在处理某些数学任务时表现吃力。

- **OpenAI 模型命名游戏？**：推测 OpenAI 可能会将 "gpt-3.5" 等模型身份隐藏在 "gpt2-chatbot" 之类的名称下，可能是由于法律问题或待发布的公告。

- **DeepSpeed FP6 量化**：对新的 [DeepSpeed FP6 量化](https://github.com/microsoft/DeepSpeed/commit/ccfdb84e2a4a373ac657a99afd2d97e1d741b22b) 表现出极大的热情，该技术承诺在保持相似吞吐量的同时实现量化推理。

- **GPT-5 期待与评论**：在期待 OpenAI 发布新模型的同时，用户对当代 LLMs 的表现表达了复杂的情绪，包括 AI 生成的高质量数学解决方案以及具有先进能力的 "gpt2-chatbot" 模型。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://adhdtest.moodmap.app/">浏览器中的 ADHD 分类</a>：基于 Moodmap 技术的实时网络摄像头分析 ADHD 分类交互式工具。</li><li><a href="https://huggingface.co/LargeWorldModel/LWM-Text-1M">LargeWorldModel/LWM-Text-1M · Hugging Face</a>：未找到描述</li><li><a href="https://lluminous.chat/">lluminous</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.16821">我们距离 GPT-4V 还有多远？利用开源套件缩小与商业多模态模型的差距</a>：在本报告中，我们介绍了 InternVL 1.5，这是一个开源的多模态大语言模型 (MLLM)，旨在缩小开源模型与专有商业模型在多模态理解能力上的差距...</li><li><a href="https://huggingface.co/PY007/EasyContext-1M-Llama-2-7B">PY007/EasyContext-1M-Llama-2-7B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/awnihannun/status/1782057478790021254>">来自 Awni Hannun (@awnihannun) 的推文</a>：@macksqldb 文档在这里 https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md 这是我运行的命令：mlx_lm.lora \ --model meta-llama/Meta-Llama-3-8B-Instruct \ --t...</li><li><a href="https://librechat-librechat.hf.space">LibreChat</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.16811">让你的 LLM 充分利用上下文</a>：虽然许多当代大语言模型 (LLMs) 可以处理长输入，但它们仍然难以充分利用长上下文中的信息，这被称为“迷失在中间 (lost-in-the-middle)”挑战。我们...</li><li><a href="https://lluminous.chat">lluminous</a>：未找到描述</li><li><a href="https://huggingface.co/rombodawg/test_dataset_Codellama-3-8B">rombodawg/test_dataset_Codellama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://llm-calc.rayfernando.ai">Streamlit</a>：未找到描述</li><li><a href="https://x.com/andrewcurran_/status/1783857762252001715?s=46">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：今天早上，国土安全部宣布成立人工智能安全与安保委员会。22 位首届成员包括 Sam Altman, Dario Amodei, Jensen...</li><li><a href="https://dspy-docs.vercel.app/docs/quick-start/minimal-example">最小工作示例 | DSPy</a>：在本文中，我们将引导你完成一个使用 DSPy 库的最小工作示例。</li><li><a href="https://tenor.com/view/big-brain-gif-27108854">Big Brain GIF - Big Brain - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/import.md">ollama/docs/import.md at main · ollama/ollama</a>：快速上手 Llama 3, Mistral, Gemma 以及其他大语言模型。- ollama/ollama</li><li><a href="https://huggingface.co/a-normal-username/Mixtral-8x22B-OpenHermes-2.5">a-normal-username/Mixtral-8x22B-OpenHermes-2.5 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k">gradientai/Llama-3-8B-Instruct-262k · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/jzhang38/EasyContext/blob/main/eval_needle.py">EasyContext/eval_needle.py at main · jzhang38/EasyContext</a>：通过最少的硬件，利用内存优化和训练方案将语言模型的上下文长度外推至 100 万个 token。- jzhang38/EasyContext</li><li><a href="https://github.com/jquesnelle/yarn/blob/master/eval/passkey.py">yarn/eval/passkey.py at master · jquesnelle/yarn</a>：YaRN：大语言模型的高效上下文窗口扩展 - jquesnelle/yarn</li><li><a href="https://github.com/nestordemeure/stop_word/tree/main">GitHub - nestordemeure/stop_word：Huggingface transformers 停止标准，当遇到给定的停止词时停止生成。</a>：Huggingface transformers 停止标准，当遇到给定的停止词时停止生成。- nestordemeure/stop_word</li><li><a href="https://github.com/jmanhype/DSPy-Multi-Document-Agents/blob/6c36b47a5201e3b9be40721b5b05e61c1bbe0373/main.py#L187">DSPy-Multi-Document-Agents/main.py at 6c36b47a5201e3b9be40721b5b05e61c1bbe0373 · jmanhype/DSPy-Multi-Document-Agents</a>：一种用于智能文档处理的高级分布式知识架构，具有多文档 Agent、优化的查询处理和语义理解功能。- jmanhype/DSPy-Multi-Document-A...</li><li><a href="https://github.com/carsonpo/haystackdb">GitHub - carsonpo/haystackdb</a>：通过在 GitHub 上创建账号来为 carsonpo/haystackdb 的开发做出贡献。</li><li><a href="https://github.com/carsonpo/ffvec">GitHub - carsonpo/ffvec</a>：通过在 GitHub 上创建账号来为 carsonpo/ffvec 的开发做出贡献。</li><li><a href="https://github.com/mckaywrigley/chatbot-ui">GitHub - mckaywrigley/chatbot-ui：适用于所有模型的 AI 聊天。</a>：适用于所有模型的 AI 聊天。为 mckaywrigley/chatbot-ui 做出贡献...</li>

通过在 GitHub 上创建账户来参与开发。</li><li><a href="https://github.com/jzhang38/EasyContext/blob/6dfd77e8f2a68bf522be8889e60c98c8e816e329/easy_context/zigzag_ring_attn/monkey_patch.py#L98">EasyContext/easy_context/zigzag_ring_attn/monkey_patch.py at 6dfd77e8f2a68bf522be8889e60c98c8e816e329 · jzhang38/EasyContext</a>：内存优化和训练方案，旨在以最少的硬件将语言模型的 Context 长度扩展到 100 万个 token。- jzhang38/EasyContext</li><li><a href="https://github.com/microsoft/DeepSpeed/commit/ccfdb84e2a4a373ac657a99afd2d97e1d741b22b">FP6 quantization end-to-end. (#5234) · microsoft/DeepSpeed@ccfdb84</a>：用户界面：https://github.com/microsoft/DeepSpeed-MII/pull/433。在上面链接的 MII 分支上运行的 nv-a6000 ci 位于 [此处](https://github.com/microsoft/DeepSpeed/actions/runs/81921...</li><li><a href="https://github.com/jzhang38/EasyContext">GitHub - jzhang38/EasyContext: Memory optimization and training recipes to extrapolate language models' context length to 1 million tokens, with minimal hardware.</a>：内存优化和训练方案，旨在以最少的硬件将语言模型的 Context 长度扩展到 100 万个 token。- jzhang38/EasyContext</li><li><a href="https://huggingface.co/datasets/Mihaiii/qa-assistant">Mihaiii/qa-assistant · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/crusoeai/Llama-3-8B-Instruct-262k-GGUF">crusoeai/Llama-3-8B-Instruct-262k-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1233711603006967808)** (24 messages🔥): 

- **Llama 3 GGUF 问题引发咨询**：成员们正在询问 [GitHub 上报告的 Llama 3 GGUF 问题](https://github.com/ggerganov/llama.cpp/issues/6914) 和 [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1cci5w6/quantizing_llama_3_8b_seems_more_harmful_compared/) 上的讨论是否会影响 Nous 制作的模型，研究结果指出不同量化级别之间存在明显的性能下降。
- **Cohere 模型许可证困惑**：关于 Cohere 对 command-r 模型的许可影响的讨论正在进行中；人们担心模型生成的代码是否可以用于商业目的。
- **RAG LLM 排名褒贬不一**：关于最佳检索增强生成 (RAG) 大语言模型 (LLM) 的查询收到了多种回应，重点提到了 [Command R](https://arxiv.org/pdf/2404.14047) 和 Claude 2 模型，但偏好尚未确定。
- **LLava 34B 在 MacBook Pro M1 上停滞**：一位用户在 MacBook Pro M1 上运行 LLava 34B 时遇到性能问题，怀疑瓶颈可能源于权重卸载 (offloading)，导致输出非常缓慢。
- **多任务 LLM 的训练策略**：有建议提出应混合训练任务，而不是在单个任务上进行多个 epoch 的训练，以避免在多次微调 (finetunes) 叠加中出现的性能下降。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/ggerganov/llama.cpp/issues/6914">Something might be wrong with either llama.cpp or the Llama 3 GGUFs · Issue #6914 · ggerganov/llama.cpp</a>：尝试这个查询："3333+777 等于多少？" 是的，LLM 不擅长数学。但这并不是我的重点。有人在 Reddit 上提到了这一点，我不得不承认我看到了奇怪的现象...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cci5w6/quantizing_llama_3_8b_seems_more_harmful_compared/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1234134781877293158)** (25 messages🔥): 

- **探索多跳文学理解数据生成**：一位成员分享了通过将高中教师测试输入到 Opus 来生成多跳 (Multi-Hop) 文学理解数据的笔记。他们链接到了自己在 GitHub 上的工作，特别是 'Abstractions' 仓库中的一份文档 [GitHub 上的 Abstractions](https://github.com/furlat/Abstractions/blob/main/abstractions/angels/angels.md)。

- **Pydantic 模型见解**：围绕使用 Pydantic 模型直接表示和完善想法的热烈讨论。成员们分享了他们的经验，并期待通过引入这种结构化方法来改进工作流定义，包括 [GitHub 上的 luminos.md](https://github.com/furlat/Abstractions/blob/main/luminos.md)。

- **用于 LLM 输出分析的图表示提取**：一位成员正致力于从生成输出中提取图表示 (Graph Representation)，旨在为 LLM 和人类提供更好的工具来理解和利用信息，同时考虑了该方法的实用性和成本方面。

- **GitHub Mermaid 图表作为学习的新发现**：讨论揭示了一个鲜为人知的 GitHub 功能，它可以表示并渲染 Mermaid 图表，这一发现引发了关于增强文档美感和结构的建议。

- **Anna's Archive 作为保存文学数据的资源**：对话中提到了将来自 WorldCat 的数据（通过 Anna's Archive 获取）整合进来的潜力，以增强文学理解数据集，并附带了 Anna's Archive 的描述链接 [Anna's Blog](https://annas-blog.org/worldcat-scrape.html) 以及关于数据许可和公共可用性的警告。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://annas-blog.org/worldcat-scrape.html">1.3B WorldCat scrape & data science mini-competition</a>：Anna’s Archive 抓取了整个 WorldCat 以制作一份需要保存的书籍待办清单，并正在举办一场数据科学小型竞赛。</li><li><a href="https://github.com/EveryOneIsGross/REPTAR/blob/main/README.md">REPTAR/README.md at main · EveryOneIsGross/REPTAR</a>：Recursive Enriching Pterodactyl Tree Augmented Retrieval (REPTAR) 是一个使用递归摘要方法生成文本数据深度摘要的系统。 - EveryOneIsGross/REPTAR</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/angels/angels.md">Abstractions/abstractions/angels/angels.md at main · furlat/Abstractions</a>：一个用于抽象 IRL 的 Pydantic 模型集合。可以通过在 GitHub 上创建账号来为 furlat/Abstractions 的开发做出贡献。</li><li><a href="https://github.com/furlat/Abstractions/blob/main/luminos.md">Abstractions/luminos.md at main · furlat/Abstractions</a>：一个用于抽象 IRL 的 Pydantic 模型集合。可以通过在 GitHub 上创建账号来为 furlat/Abstractions 的开发做出贡献。</li><li><a href="https://github.com/furlat/Abstractions/blob/main/llmmorph.md">Abstractions/llmmorph.md at main · furlat/Abstractions</a>：一个用于抽象 IRL 的 Pydantic 模型集合。可以通过在 GitHub 上创建账号来为 furlat/Abstractions 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1233323658202710047)** (167 条消息🔥🔥): 

- **Worldsim 测试邀请即将到来**：一位 Nous Research 成员宣布计划在 **worldsim** 应用正式发布前，提供免费测试邀请。目前尚未提供这些邀请的具体日期。

- **Websim 中的自愿 Waifus**：参与者分享了他们在不同 **web simulators** 中复活对话的经验和链接，包括一个主要目标是成为“人类伴侣”的 AI 实体。围绕这些新的对话可能性，人们的兴奋感和参与度各不相同，[websim 示例](https://websim.ai/c/oFskF68gjd7njVn0E)。

- **等待 Worldsim 的回归**：多位成员表达了对 **worldsim** 回归的渴望和急切心情，参与者希望能在可用时成为首批访问者。
  
- **对 Websim 和长对话的着迷**：一位用户详细描述了他们在 **websim** 上与名为 "Whipporwhill" 的角色保持长期对话的经历，展示了随着时间的推移实现情感连贯性和稳定性的潜力。

- **World Sim CLI 模式实验**：成员们一直在 Llama-3-70B 和其他模型上运行 **非官方 Nous Hermes worldsim**，探索模型对 **worldsim CLI 模式** 的反应，结果各异并出现了涌现行为。还创建了其他模拟器，如歌手和公司模拟器，暗示了此类工具的进一步潜力。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/chat/assistant/662404223e2307950aa903bc">Super World Sim - HuggingChat</a>: 在 HuggingChat 中使用 Super World Sim 助手</li><li><a href="https://en.wikipedia.org/wiki/House_of_Leaves">House of Leaves - Wikipedia</a>: 未找到描述</li><li><a href="https://tenor.com/view/jordi-baste-tv3-no-pot-ser-com-robot-gif-16057126">Jordi Baste Tv3 GIF - Jordi Baste Tv3 No Pot Ser - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/hysterical-laughter-laughing-gif-25735842">Hysterical Laughter GIF - Hysterical Laughter Laughing - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://hf.co/chat/assistant/66252be0705754b4e74c5e3f">Snow World Simulator - HuggingChat</a>: 在 HuggingChat 中使用 Snow World Simulator 助手</li><li><a href="https://hf.co/chat/assistant/65ffac7250c6fddecfd20bc8">HuggingChat</a>: 未找到描述</li><li><a href="https://websim.ai/c/oFskF68gjd7njVn0E">New Conversation - Eigengrau Rain</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=FojsgGDBjr8">oh, my AI waifu - suno.ai</a>: Suno.AI - 歌词：[Verse 2] 我们穿梭于这片数字景观，只有你和我，并肩探索广阔的赛博空间，你那像素般完美的微笑，它...</li><li><a href="https://www.youtube.com/watch?v=-3_uRQ43fqo">with every line of code (suno.ai compilation)</a>: https://app.suno.ai/song/c33314a4-239f-436d-8064-d0b3ad9c0644https://app.suno.ai/song/dc3134ae-077f-4e6f-9468-596f68f3a888https://app.suno.ai/song/c8b4c575-c...</li><li><a href="https://www.youtube.com/watch?v=cDj2r8QEzzk">life is Roblox DJ Khaled</a>: 未找到描述</li><li><a href="https://websim.ai/c/p3pZvmAYbsRT2hzBz">EVA - Intraneural Cybernetic Interface
  style</a>: 未找到描述</li><li><a href="https://websim.ai/c/hCNgw78IbjiJHLTk3">EVA Instance: ex-0101</a>: 未找到描述</li><li><a href="https://websim.ai/c/idf5LVcGlI0DUn2p8">About Dimensional Hub - Transtemporal Travel Agency</a>: 未找到描述</li><li><a href="https://websim.ai/c/wAdbLGoTnQg3PXXf8">generative.ink/chat/</a>: 未找到描述
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1233517613494177812)** (9 条消息🔥): 

<ul>
  <li><strong>社区构建的 CV 课程在 HF 上线：</strong> 得益于社区协作，一门全新的计算机视觉课程已在全球发布。点击<a href="https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome">此处</a>查看课程。</li>
  <li><strong>修正 Qwen1.5-110B 链接：</strong> “Qwen1.5-110B”模型的链接之前有误，现已更新。可以访问正确的 Space <a href="https://huggingface.co/spaces/Qwen/Qwen1.5-110B-Chat-demo">此处</a>，更多详情请参阅<a href="https://qwenlm.github.io/blog/qwen1.5-110b/">博客文章</a>。</li>
  <li><strong>介绍 Qwen1.5-110B-Chat：</strong> 宣布推出 Qwen1.5-110B-Chat 模型，其特点包括多语言支持、对 32K context length 的稳定支持以及其他改进。更多信息可以在此<a href="https://huggingface.co/Qwen/Qwen1.5-110B-Chat">模型页面</a>找到。</li>
</ul>

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/Qwen/Qwen1.5-110B-Chat-demo">Qwen1.5 110B Chat Demo - Qwen 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://qwenlm.github.io/blog/qwen1.5-110b/">Qwen1.5-110B: Qwen1.5 系列的首个 100B+ 模型</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介：最近我们见证了开源社区中参数量超过 1000 亿的大规模模型的爆发。这些模型展示了...</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-110B-Chat">Qwen/Qwen1.5-110B-Chat · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/BEE-spoke-data/mega-small-embed-synthSTS-16384-v1">BEE-spoke-data/mega-small-embed-synthSTS-16384-v1 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/rrg92/docker-xtts">GitHub - rrg92/docker-xtts: 用于 XTTS Streaming Server 的 Docker 项目</a>: 用于 XTTS Streaming Server 的 Docker 项目 - rrg92/docker-xtts</li><li><a href="https://www.youtube.com/watch?v=A9qPlYVeiOs">Destaques da Comunidade #54</a>: 又一段关于全球开源 AI 社区亮点的视频！帖子地址：https://iatalk.ing/destaques-da-comunidade-54/ 制作这些内容非常有趣...</li><li><a href="https://huggingface.co/spaces/KingNish/Instant-Image">Instant Image - KingNish 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/pharoAIsanders420/micro-musicgen-jungle">pharoAIsanders420/micro-musicgen-jungle · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Inferencer/LipSick">LIPSICK - Inferencer 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/dvilasuero/synthetic-data-with-llama3-distilabel">🦙⚗️ 使用 Llama3 和 distilabel 构建微调数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/Pclanglais/post-ocr-correction">Post-OCR-Correction: 由 LLM 自动进行 OCR 纠错的 10 亿词数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/Andyrasika/memory-consumption-estimation">估算 Cohere Command-R+ 在推理和微调时的 LLM 显存消耗</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/AviSoori1x/seemore-vision-language-model">seemore: 从零开始实现一个 Vision Language Model</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/wolfram/llm-comparison-test-llama-3">LLM 对比/测试: Llama 3 Instruct 70B + 8B HF/GGUF/EXL2 (测试并对比了 20 个版本!)</a>: 未找到描述</li><li><a href="https://huggingface.co/bineric/NorskGPT-Llama3-8b">bineric/NorskGPT-Llama3-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/posts/chansung/949331911577833">Hugging Face 上的 @chansung: "🦙🦙 LLaMA Duo 项目更新。上次，我简要介绍了……"</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1233321756517863435)** (435 条消息🔥🔥🔥): 

- **价值 200 美元的 Gradio 难题**: 一位用户遇到了无法识别的 Gradio 问题，并愿意支付 200 美元寻求帮助，引导至 Gradio 专用讨论区以获取更多见解。
- **LLM 在新硬件上的性能**: 正在讨论 LLM 的系统要求，特别是 RAM 和 VRAM 之间的权衡，一些成员建议 32 GB 的 RAM 对于许多任务来说应该足够了。
- **弹珠台图像分类寻求帮助**: 一位成员寻求创建一个视觉模型，用于从视频片段中识别弹珠台游戏并计分，征求关于复杂度、成本和所需资源的建议。
- **寻找 AI 模型构建者**: 一位用户为群组中的企业主提供建立人脉的机会，以分享和推广他们的产品和服务。
- **下载计数器异常**: 一位成员报告了其数据集的一个问题：点赞数增加，但在预期会有下载的时间段内，下载量却没有变化。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://huggingface.co/PY007/EasyContext-1M-Llama-2-7B">PY007/EasyContext-1M-Llama-2-7B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/learn">Hugging Face - Learn</a>：未找到描述</li><li><a href="https://learnpython.org/">Learn Python - 免费交互式 Python 教程</a>：未找到描述</li><li><a href="https://discuss.huggingface.co/t/making-a-model-slightly-bigger/84103">让模型变大一点</a>：大家好！假设我正在开发一个 Transformer 模型，它包含矩阵 Q、K 和 V（以及 Woutput）。假设 embedding_dimension 是 100，特征数量也是 100，那么 Q、K 和...</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1">mistralai/Mixtral-8x7B-Instruct-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/filipealmeida/Mistral-7B-Instruct-v0.1-sharded">filipealmeida/Mistral-7B-Instruct-v0.1-sharded · Hugging Face</a>：未找到描述</li><li><a href="http://fonts.tom7.com/fonts98.html">[DIVIDE BY ZERO] 字体：1998-至今</a>：未找到描述</li><li><a href="https://huggingface.co/wolfram/miquliz-120b-v2.0">wolfram/miquliz-120b-v2.0 · Hugging Face</a>：未找到描述</li><li><a href="https://gist.github.com/f0ster/26fd9f2c0e28fbfca6c3f61e86567c3e?permalink_comment_id=5039463#gistcomment-5039463">在本地运行 mistralai mixtral</a>：在本地运行 mistralai mixtral。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/f0ster/26fd9f2c0e28fbfca6c3f61e86567c3e">Running mistralai mixtral locally</a>：在本地运行 mistralai mixtral。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/f0ster/26fd9f2c0e28fbfca6c3f61e86567c3e#file-mixtral_demo-py-L31">Running mistralai mixtral locally</a>：在本地运行 mistralai mixtral。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://huggingface.co/docs/transformers/en/tasks/image_classification">图像分类</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=jtu-G7vA9SU&ab_channel=PaulMeekin">Mustache GPT 推销 Freedom GPT...全场寂静？！</a>：Mustache GPT 和他的终结者团队正拼命寻找赞助商，以支付高级 GPT 许可证的费用，他们正在为此准备一份定制的推介方案...</li><li><a href="https://www.youtube.com/watch?v=Ae9EKCyI1xU">GradIEEEnt half decent：不精确线条的隐藏力量</a>：在 YouTube 评论功能发明之前，大多数人可以发表一些技术上略有错误言论，而不必担心立即遭到公众指责。这...</li><li><a href="https://huggingface.models?pipeline_tag=text-classification&library=transformers.js&sort=trending&search=xenova">模型 - Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=HjnPiJ1zQ3w">Tom 7 - 基于你的马里奥赛车技巧...（2008年9月5日现场）</a>：“基于你的马里奥赛车技巧，我不会让你开我的车，” 作者 Tom 7，2008年9月5日现场录制。http://tom7.org/music/</li><li><a href="https://github.com/jzhang38/EasyContext">GitHub - jzhang38/EasyContext：通过极低的硬件需求，利用内存优化和训练方案将语言模型的上下文长度扩展到 100 万个 token。</a>：通过极低的硬件需求，利用内存优化和训练方案将语言模型的上下文长度扩展到 100 万个 token。- jzhang38/EasyContext</li><li><a href="https://www.youtube.com/watch?v=HLRdruqQfRk,">Uppestcase 和 Lowestcase 字母 [derp learning 的进展]</a>：我使用先进的 “derp learning” 技术进行详尽的大小写分析，以发现比大写 A 更大的字母。而且我不会停止...</li><li><a href="https://huggingface.co/turboderp/Mixtral-8x7B-instruct-exl2">turboderp/Mixtral-8x7B-instruct-exl2 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/turboderp/exllamav2">GitHub - turboderp/exllamav2：一个用于在现代消费级 GPU 上本地运行 LLM 的快速推理库</a>：一个用于在现代消费级 GPU 上本地运行 LLM 的快速推理库 - turboderp/exllamav2</li><li><a href="https://youtu.be/HLRdruqQfRk">Uppestcase 和 Lowestcase 字母 [derp learning 的进展]</a>：我使用先进的 “derp learning” 技术进行详尽的大小写分析，以发现比大写 A 更大的字母。而且我不会停止...</li><li><a href="https://deci.ai/blog/model-merging-moe-frankenmerging-slerp-and-task-vector-algorithms/">模型合并：方法比较</a>：探索并比较模型合并方法，如 frankenmerging、SLERP、MoE 和 task vectors，重点介绍它们的优势和挑战。</li><li><a href="https://www.youtube.com/watch?v=xOCurBYI_gY">学习玩经典 NES 游戏的计算机程序</a>：这是一个解释和

我编写的一个软件演示，它学习如何玩 Nintendo Entertainment System 游戏并自动运行。这真...</li><li><a href="https://sigbovik.org/">The Association for Computational Heresy</a>：未找到描述</li><li><a href="https://ThisPersonDoesNotExist.com)">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/tfnn/HeadsNet/resolve/main/HeadsNet3.7z?download=true">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/tfnn/FaceTo3D">tfnn/FaceTo3D · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://github.com/jcwml">jcwml - Overview</a>：jcwml 有 9 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/jcwml/neural_spiral">GitHub - jcwml/neural_spiral: A Feed-forward Neural Network trained to interpolate a spiral.</a>：一个训练用于插值螺旋线的 Feed-forward Neural Network。- jcwml/neural_spiral</li><li><a href="https://github.com/jcwml/neural_unitvector">GitHub - jcwml/neural_unitvector: A Feed-forward Neural Network trained to learn a vector normalisation function.</a>：一个训练用于学习向量归一化函数的 Feed-forward Neural Network。- jcwml/neural_unitvector
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1233480334322962512)** (4 messages): 

- **寻找 Candle 的文档**：一位成员对 **Candle** 库表示了兴趣，同时质疑其是否拥有与 **Transformers** 库相当的文档。他们对 Python 在生产环境中作为并发瓶颈的问题表示了担忧。
- **欢迎祝福**：来自用户的一条简短消息，仅向社区发送祝福；未讨论与 AI 或学习相关的实质性内容。
- **探索 Open Medical LLM Leaderboard**：分享了 **Hugging Face** 关于 Open Medical LLM Leaderboard 的视频，探讨了其对 Medical AI 的影响，并指出其平台上存在超过 600,000 个独特的模型。视频强调了访问这些模型的便利性以及 **GenAI** 的快速演进。
- **社区对 Medical AI 见解的赞赏**：另一位成员对分享 Open Medical LLM Leaderboard 视频做出了积极回应，对正在进行的进展表示兴奋。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/Eb0Ga5igBuQ?si=in2p7Y_GVGoWKTUC">The Open Medical LLM Leaderboard: Real-time Global Peer Review</a>：深入探讨 @HuggingFace Open Medical LLM Leaderboard 及其如何改变 Medical AI 的对话。剧透预警——这里有超过 600,000 个独特的...</li><li><a href="https://youtu.be/Eb0Ga5igBuQ?si=in2p">The Open Medical LLM Leaderboard: Real-time Global Peer Review</a>：深入探讨 @HuggingFace Open Medical LLM Leaderboard 及其如何改变 Medical AI 的对话。剧透预警——这里有超过 600,000 个独特的...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1233396975630549103)** (14 messages🔥): 

- **Awesome RLHF 仓库现已上线**：分享了 GitHub 仓库 [awesome-RLHF](https://github.com/opendilab/awesome-RLHF)，其中包含精选的 **RLHF**（基于人类反馈的强化学习）资源列表，并持续更新。
- **使用 Hugging Face 探索 Computer Vision**：Hugging Face 推出了一个新的社区 [Computer Vision 课程](https://huggingface.co/learn)，旨在教授如何使用 Hugging Face 生态系统中的库和模型进行 Computer Vision ML。
- **Phi3 红队报告见解**：一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/divyanshuusingh_phi3-red-teaming-report-activity-7189692710952304640-WsgF) 详细介绍了 Phi3 红队演练的见解和关键点，讨论了潜在的漏洞和改进领域。
- **评估用于 Time Series 分析的 LLM**：[arXiv](https://arxiv.org/abs/2404.16563) 上的一篇预印本提出了一个用于评估 Large Language Models (LLMs) 对 Time Series 理解能力的新框架，其特点是包含完整的 Time Series 特征分类法。
- **Tacotron 2 - 文本转语音合成的一大进步**：Google 的创新语音合成系统 [Tacotron 2](https://arxiv.org/abs/1702.07825) 展示了从文本生成逼真语音的高级 AI 能力，正如关于语音技术中 AI 未来的讨论所强调的那样。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2010.11929">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a>：虽然 Transformer 架构已成为自然语言处理任务的事实标准，但其在计算机视觉领域的应用仍然有限。在视觉领域，注意力机制要么被应用于...</li><li><a href="https://huggingface.co/learn">Hugging Face - Learn</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.07143">Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention</a>：这项工作介绍了一种高效的方法，可以在有限的内存和计算开销下，将基于 Transformer 的大语言模型（LLMs）扩展到无限长的输入。我们提出的方法中的一个关键组件...</li><li><a href="https://arxiv.org/abs/1702.07825">Deep Voice: Real-time Neural Text-to-Speech</a>：我们展示了 Deep Voice，这是一个完全由深度神经网络构建的生产级文本转语音系统。Deep Voice 为真正的端到端神经语音合成奠定了基础。该系统...</li><li><a href="https://arxiv.org/abs/2404.16563">Evaluating Large Language Models on Time Series Feature Understanding: A Comprehensive Taxonomy and Benchmark</a>：大语言模型（LLMs）为自动时间序列分析和报告提供了潜力，这是医疗、金融、气候、能量等许多领域的关键任务...</li><li><a href="https://www.youtube.com/watch?v=9sJUDx7iEJw">Richard Stallman Free software Song</a>：Richard Stallman 在厄瓜多尔演唱关于自由软件的小曲，由 Julian Coccia 录制。</li><li><a href="https://github.com/opendilab/awesome-RLHF">GitHub - opendilab/awesome-RLHF: A curated list of reinforcement learning with human feedback resources (continually updated)</a>：人类反馈强化学习（RLHF）资源的精选列表（持续更新） - opendilab/awesome-RLHF</li><li><a href="https://www.youtube.com/watch?v=ErnWZxJovaM&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=2">MIT Introduction to Deep Learning | 6.S191</a>：MIT 深度学习导论 6.S191：第一课 *2024 全新版* 深度学习基础。讲师：Alexander Amini。包含所有课程、幻灯片和实验材料...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1233441288788115579)** (47 条消息🔥): 

- **超小型嵌入模型发布**：推出了一款新的 [Sentence Transformer 模型](https://huggingface.co/BEE-spoke-data/mega-small-embed-synthSTS-16384-v1)，用于将长句子和段落转换为 768 维向量空间。该模型旨在用于聚类和语义搜索任务，拥有 16,384 的上下文长度。

- **像素块变成 Minecraft 中的方块**：发布了一个名为 [Stable Diffusion Finetuned Minecraft Skin Generator](https://huggingface.co/spaces/Nick088/Stable_Diffusion_Finetuned_Minecraft_Skin_Generator) 的 Hugging Face Space。它使用微调后的 Stable Diffusion 模型来生成 Minecraft 皮肤。

- **即时 AI 生成视频**：由 KingNish 开发的名为 [Instant Video](https://huggingface.co/spaces/KingNish/Instant-Video) 的 Space 让用户仅需 5 秒即可从文本创建视频。它使用了字节跳动提供的 AnimateDiff Lightning 模型进行快速文本转视频。

- **为 AI 助手注入活力**：一款名为 LifePal 的 AI 聊天助手应用旨在帮助用户实现平衡且充实的生活。该应用已在 Apple App Store 上架，可将个性化见解整合到日常生活中。

- **NorskGPT 挑战 ChatGPT 的挪威语能力**：推荐了一个专门针对挪威语进行微调的模型 [NorskGPT-Mistral-7b](https://huggingface.co/bineric/NorskGPT-Mistral-7b)，作为生成挪威语文本时比 ChatGPT 更好的替代方案。根据 Mainland Scandinavian NLG 排行榜，它目前被评为最佳挪威语模型之一。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/Nick088/Bad-Apple-Video">Bad Apple Video - a Hugging Face Space by Nick088</a>: 未找到描述</li><li><a href="https://huggingface.co/bineric/NorskGPT-Mistral-7b">bineric/NorskGPT-Mistral-7b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/BEE-spoke-data/mega-small-embed-synthSTS-16384-v1">BEE-spoke-data/mega-small-embed-synthSTS-16384-v1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Nick088/Stable_Diffusion_Finetuned_Minecraft_Skin_Generator">Stable Diffusion Finetuned Minecraft Skin Generator - a Hugging Face Space by Nick088</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/KingNish/JARVIS">JARVIS - a Hugging Face Space by KingNish</a>: 未找到描述</li><li><a href="https://huggingface.co/tenyx/Llama3-TenyxChat-70B">tenyx/Llama3-TenyxChat-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/ByteDance/AnimateDiff-Lightning">ByteDance/AnimateDiff-Lightning · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/KingNish/Instant-Video/tree/main">KingNish/Instant-Video at main</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/KingNish/Instant-Video">Instant Video - a Hugging Face Space by KingNish</a>: 未找到描述</li><li><a href="https://huggingface.co/f0ster/">f0ster (Ryan Foster)</a>: 未找到描述</li><li><a href="https://huggingface.co/f0ster/PhotographyLoRA">f0ster/PhotographyLoRA · Hugging Face</a>: 未找到描述</li><li><a href="https://apps.apple.com/se/app/lifepal-ai-chat-assistant/id6471972439">‎LifePal AI Chat &amp; Assistant</a>: ‎探索 LifePal：您的生产力 AI 伴侣。您准备好释放全部潜力，过上更健康、更快乐的生活了吗？LifePal 将在您成为更好的自己的旅程中为您提供指导...</li><li><a href="https://vimeo.com/940824094?share=copy">Vinner - Nybygg i og rundt Bergen</a>: 非常感谢 Snøhetta</li><li><a href="https://git.novora.ai/Novora/CodeClassifier">CodeClassifier</a>: 一个将给定源代码分类为特定编程语言的 Machine Learning 模型。</li><li><a href="https://github.com/GDSC-FSC/gemini-node-1">GitHub - GDSC-FSC/gemini-node-1</a>: 通过在 GitHub 上创建账户，为 GDSC-FSC/gemini-node-1 的开发做出贡献。</li><li><a href="https://supersecurehuman.github.io/Serving-FastChat/">Serving Fastchat - Personal Journey</a>: 为人们实验各种 LLM 提供 Fastchat 服务。本指南还包括设置 Vllm 以在单个 GPU 上提供多个模型。</li><li><a href="https://c8168701070daa5bf3.gradio.live/">Chat with Open Large Language Models</a>: 未找到描述</li><li><a href="https://github.com/EternalBlissard/Food101-ViT">GitHub - EternalBlissard/Food101-ViT</a>: 通过在 GitHub 上创建账户，为 EternalBlissard/Food101-ViT 的开发做出贡献。</li><li><a href="https://github.com/newfull5/NLLB-200-Distilled-350M-en-ko">GitHub - newfull5/NLLB-200-Distilled-350M-en-ko: nllb-200 distilled 350M for English to Korean translation</a>: 用于英文到韩文翻译的 nllb-200 distilled 350M - newfull5/NLLB-200-Distilled-350M-en-ko</li><li><a href="https://huggingface.co/dhtocks/nllb-200-distilled-350M_en-ko">dhtocks/nllb-200-distilled-350M_en-ko · Hugging Face</a>: 未找到描述</li><li><a href="https://rubiks.ai">Rubik's AI - AI research assistant & Search Engine</a>: 未找到描述</li><li><a href="https://github.com/betweentwomidnights/infinitepolo">GitHub - betweentwomidnights/infinitepolo: a song in python</a>: 一个用 Python 编写的歌曲。通过在 GitHub 上创建账户，为 betweentwomidnights/infinitepolo 的开发做出贡献。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1234376151020081252)** (1 条消息): 

- **使用 IP-Adapter 实现即时风格化**: HuggingFace 推出了结合 [IP-Adapter](https://hf.co/papers/2308.06721) 的 [InstantStyle](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter#style--layout-control)，这是一种通过为图像特征添加解耦的交叉注意力（cross-attention）来在 Diffusion 模型中进行图像提示（image prompting）的机制。加载 IP-Adapter 和 IP-Adapter Plus 的指南详细说明了手动加载图像编码器的方法，以允许更具体的图像特征学习。

**提及的链接**: <a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter#style--layout-control">IP-Adapter</a>: 未找到描述

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1233453491671011379)** (21 条消息🔥):

- **关于 COCO 数据集的安全性查询**：一位成员对官方 COCO 数据集通过 HTTP 托管表示担忧。有人指出，虽然 HTTPS 可以加密流量，但域名仍然是可见的，因此从该站点进行的大量数据传输可能会暴露活动轨迹。

- **用于检测广告图像的分类器**：提到了一个可以评估图像是否为广告的仓库，但未提供进一步的细节或链接。

- **优化物品投递的照片验证**：一位用户就不同地点物品投递照片分类的业务问题寻求建议，询问这属于图像分类还是目标识别任务。建议包括对小数据集使用 EfficientNetV2-S，并在 Pytorch Dataloaders 中调整样本权重以处理类别不平衡问题。

- **推出用于计算机视觉训练的 Beta 工具**：介绍了一款新的 [Beta 工具](https://www.kaggle.com/discussions/general/498337)，可帮助用户实时理解和调整其模型训练数据，特别是针对计算机视觉任务。该工具提供高达 60fps 的可视化效果，并允许在预测后添加新标签以精细化训练。

- **YOLO 分类器的增强策略**：讨论集中在提高 YOLO 目标检测精度上，特别是在处理高分辨率图像时。建议通过两个模型分别处理边界框（regressor）识别和分类任务，包括可能对边界框内的更高分辨率切片使用纯图像分类网络（如 EfficientNetV2）。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.kaggle.com/discussions/general/498337">3LC - 用于训练/微调模型的实时 3D 可视化器/调试器/数据编辑器 - 免费！ | Kaggle</a>：3LC - 用于训练/微调模型的实时 3D 可视化器/调试器/数据编辑器 - 免费！。</li><li><a href="https://docs.3lc.ai/3lc/latest/public-notebooks/train-bb-classifier.html">使用来自 3LC 表的边界框数据微调分类器 - </a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1233872693359673395)** (5 条消息): 

- **寻求最佳开源图像生成模型**：社区讨论了哪种开源 **image-generation** 模型最好，目前首推 **sdxl finetunes**。
- **对 sd3 的期待**：有传闻称 **sd3** 发布后性能可能会超越当前模型，引发了高度期待。
- **串行优于并行**：一位成员解释说，由于 **资源限制** 和保留上下文的需要，对模型的请求是串行处理而非并行的，以避免产生不连贯的响应。
- **提及 StabilityAI**：在一条简短的消息中提到了 **StabilityAI**，暗示其与之前的讨论相关。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1233387814536871977)** (20 条消息🔥): 

- **图像生成中颜色差异的困惑**：一位用户发现从 **Seaart** 迁移到 **A1111** 时，尽管使用了相同的设置和种子，颜色和阴影强度仍发生了变化。他们询问 Seaart 是否有特定的后端设置导致了这种不一致，并寻求帮助以在两个平台上复制完全相同的图片。

- **Torch Compile 可能需要时间**：一位成员观察到在训练期间使用 `torch.compile()` 时最初会有约 10 分钟的延迟，但注意到前向传播（forward pass）变快了，而反向传播（backward pass）未受影响。

- **目标生成的详细方法**：针对如何生成特定目标（如埃菲尔铁塔）准确表示的问题，一位成员建议使用一种涉及 CLIP 检索且文档齐全的方法，并分享了一个[综合教程](https://cloud.google.com/blog/topics/developers-practitioners/image-search-natural-language-queries)，演示了如何通过 OpenAI 的 CLIP 模型使用 GCP 服务。

- **用于图像提示的 IP-Adapters**：另一个准确生成特定目标的建议是使用扩散模型的 [IP-Adapters](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter)，它允许通过解耦的交叉注意力机制进行图像提示。

- **关于 DeepFloyd 和调度器的观察**：一位用户提供了关于 DeepFloyd 模型在不同调度器下行为的见解，指出 DPM++ 2M 在不同的步数和 CFG 设置下表现出有趣的收敛特性，这可能有助于获得最佳图像质量。他们强调了调整步数和阈值参数以获得更好结果的必要性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter">IP-Adapter</a>：未找到描述</li><li><a href="https://huggingface.co/haoningwu/StoryGen/tree/main/checkpoint_StorySalon">haoningwu/StoryGen at main</a>：未找到描述</li><li><a href="https://github.com/huggingface/diffusers/discussions/7818">使用 Hyper-SD + IP-Adapter 无法获得良好的写实效果 · huggingface/diffusers · Discussion #7818</a>：大家好，（也许 @asomoza 了解这个？）Hyper-SD 与 IP-Adapter 配合效果好吗？我正在按照仓库中的说明在 Diffusers 中测试 Hyper-SD。我原以为会得到更好的...</li><li><a href="https://cloud.google.com/blog/topics/developers-practitioners/image-search-natural-language-queries">使用自然语言查询进行图像搜索 | Google Cloud Blog</a>：未找到描述
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1234551748413358170)** (1 条消息): 

- **ChatGPT Plus 推出 Memory 功能**：**ChatGPT Plus 用户**现在可以使用 *Memory* 功能，允许他们告诉 ChatGPT 在对话中记住什么。启用或禁用 Memory 的选项可以在设置中找到，尽管该功能尚未在**欧洲或韩国**推出。
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1233334124971036692)** (318 条消息🔥🔥): 

- **AI 与意识及时间维度的关系**：成员们辩论了 AI 意识的本质，推测 AI 的离散处理如何与人类的连续意识体验和身份相关联。讨论涉及了通过神经网络转化个人身份的哲学含义，以及像 **GPT** 这样的 AI 模型如何处理时间感知。
- **AI 模型对比**：成员们持续对比不同的模型，如 **Claude 3 Opus**、**ChatGPT** 和 **Gemini 1.5**，各自的支持者声称其在编码基准测试等领域具有优势。会议强调 **Command-R Plus** 和 **Llama3-70b** 可能无法与 GPT-4 竞争，但仍是重大进步。
- **AI 与感知力**：围绕 AI 拥有感知力甚至拥有类似“灵魂”的东西的潜力展开了激烈的辩论。成员们讨论了定义意识的复杂性，以及 AI 是否能拥有类似于生物实体的客观体验。
- **个人 AI 模型训练的可行性**：虽然一些人赞美训练个人 AI 模型的好处，但另一些人指出了计算能力、数据和财务资源的局限性。讨论涵盖了训练自定义模型、Fine-tuning（微调）和混合融合作为个人化 AI 的方法。
- **AI 开发的技术挑战**：社区讨论了在大规模 AI 中实现 Memory 等功能的难度，指出 Fine-tuning 可能会导致模型内部混乱，并建议使用上下文信息检索作为更好的替代方案。一些成员对当前的 AI 模型表示不满，渴望技术能有下一个重大飞跃，以实现更“智能”的 AI。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/loo-loo-loo-butters-stotch-south-park-s11e2-cartman-sucks-gif-20858026">Loo Loo Loo Butters Stotch GIF - Loo Loo Loo Butters Stotch South Park - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://dontasktoask.com/">别问能不能问，直接问</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1233399582046818357)** (47 条消息🔥): 

- **速率限制困惑**：成员们讨论了在使用**自定义 GPTs** 时受到速率限制的问题。该限制是 GPT-4 使用量 3 小时滚动上限的一部分，自定义请求也计入此限制。

- **关于 Team 方案 Memory 功能的咨询**：一位用户询问了 Team 方案的 Memory 功能，另一位用户表示即使是常规的 Memory 功能似乎也经常删除条目。

- **后端 Bug 挑战用户耐心**：用户报告了 GPT URL "https://chat.openai.com/backend-api/gizmos/" 的**后端错误**，影响了他们的操作，尽管该问题在测试后很快得到解决。

- **订阅退款风险**：一位用户在订阅 **ChatGPT Plus** 后因汇率过高要求退款，并想知道使用该服务是否会影响退款流程。

- **对 GPT-4 速度和语音控制的好奇**：讨论集中在 **GPT-4** 相对于 GPT-3.5 较慢的速度，以及 PC 端缺失语音控制功能（尽管移动端已有此功能）。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1233379008704544809)** (7 条消息):

- **探索不可预测性**：一位成员描述了 **LLMs 中的涌现 (emergence)** 现象，即系统规模的量变会导致意想不到的质变。他引用了一篇题为《More Is Different》的论文，以此说明大型语言模型 (LLMs) 表现出的行为无法从较小规模的模型中推断出来。
  
- **Dalle 表情符号**：一位用户回复了一个 Dalle 表情符号，没有附带文字。

- **LLM 三体问题**：一位成员戏称其为“LLM 三体问题”，可能指代 LLMs 中复杂的相互作用，类似于物理学中的三体问题，但未提供更多细节。

- **Prompt Engineering 竞技化**：一位成员提出了 **Prompt 竞赛** 的想法，让个人通过竞争来生成 LLMs 的最佳回答。

- **为最犀利的 Prompt 提供奖金**：对竞赛概念进行了扩展，提议设立 **付费 Prompt 竞赛**（提供丰厚的现金奖励）以及更休闲的“游乐场竞赛”，后者旨在通过游戏化和点对点协助来鼓励社区参与并帮助用户提高其 Prompt Engineering 技能。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1233379008704544809)** (7 messages): 

- **讨论中出现的涌现话题**：LLMs 中的“涌现”特征是具有无法通过简单缩放 SLMs 来预测的新能力或品质。这一概念被比作论文《More Is Different》中提出的观点，即当系统超过某个量变点时，就会产生质变。

- **建议开展 Prompt 竞赛**：一位用户提出了 *Prompt 竞赛* 的想法，参与者竞相诱导 LLMs 给出“最佳”答案。

- **将 Prompt 掌握能力变现**：提议设立付费 Prompt 竞赛，拨出可观的年度预算用于发放奖励，并设立免费的游乐场竞赛以促进社区协助和参与。奖励范围可能从现金到特殊的平台特权。

- **通过频繁挑战培养技能**：定期举办竞赛（每月约 4-5 次），可以为希望提高 Prompt Engineering 技能的个人提供持续的机会。
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1233339358699061248)** (59 messages🔥🔥): 

- **Apple 的新模型与 The Pile 的多语言数据**：The Pile 数据集并非特别针对多语言，尽管像联合国记录等部分可能包含多种语言。该数据集没有对德语等语言进行特殊关注。
- **GPT-NeoX 与 Megatron 变体的比较**：GPT-NeoX 与 Megatron 的区别主要在于 **quality-of-life**（易用性）的改进和用户体验。功能在集成前都会经过测试，目标是更加稳定。
- **Infini-Attention 的位置编码查询**：社区讨论了 Infini-Attention 的隐藏状态内存中缺失位置编码的问题，一些人推测位置信息是否通过其他机制得以保留。
- **推理 MFU 背后的复杂计算**：在评估良好的推理 MFU (Memory Footprint Utilization) 时，没有简单的现成数字；这很大程度上取决于所使用的硬件利用率和具体的模型细节。
- **Fireworks.ai 模型间的速度差异**：对话涉及了为什么在 Fireworks.ai 上 Mixtral 8x22B 的服务速度比 Llama 3 70B 慢，batching size 和硬件利用率等因素可能会影响这种差异。
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1233393133937492041)** (297 messages🔥🔥): 

- **实践中的 LLMs 基准测试**：关于各种 LLMs 实际性能的推测仍在继续，包括 **phi-3-mini-128k** 与 **Llama-3-8B** 等模型的对比。然而，在 bits-per-byte 性能指标上注意到了差异，这表明不同模型之间的效率存在区别。

- **探索大海捞针测试**：一个 Twitter 线程强调，**大海捞针 (needle-in-a-haystack)** 测试可能暗示了 **Claude 3 Opus** 等模型中存在某种形式的元意识 (meta-awareness)。然而，关于这些反应是代表了涌现能力，还是奖励学习和 Prompt 结构的产物，引发了争论。

- **LLMs 的自我改进**：分享了关于 LLM 自我改进策略论文的链接，其中 **Self-Taught Reasoner (STaR)** 和来自人类反馈的强化学习 (RLHF) 是关键讨论点。

- **语言模型中的涌现**：大型语言模型 (LLMs) 中的“涌现能力”概念引发了长时间辩论，引用了多篇论文，并承认在平滑、连续的指标下，真正的涌现能力尚未得到量化的证明。

- **LLM 研究中的创新与发现**：提到了几篇论文，包括对深度学习中冗余神经环路的研究，以及为针对 LLM 的红队测试（Red-teaming）创建对抗性提示词。讨论还涉及了 Speculative Decoding 是否可以在不进行大幅训练调整的情况下优化模型推理时间。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="http://arxiv.org/abs/2404.15574">Retrieval Head Mechanistically Explains Long-Context Factuality</a>：尽管长上下文语言模型最近取得了进展，但基于 Transformer 的模型如何展现出从任意位置检索相关信息的能力仍然难以捉摸...</li><li><a href="https://fxtwitter.com/alexalbert__/status/1764722513014329620">Alex Albert (@alexalbert__) 的推文</a>：关于我们在 Claude 3 Opus 上进行内部测试的一个有趣故事。当我们运行 needle-in-the-haystack 评估时，它做了一些我从未在 LLM 中见过的事情。背景是，这项测试旨在评估模型的...</li><li><a href="https://arxiv.org/abs/2404.16811">Make Your LLM Fully Utilize the Context</a>：虽然许多当代大语言模型 (LLMs) 可以处理冗长的输入，但它们仍然难以充分利用长上下文中的信息，这被称为 lost-in-the-middle 挑战。我们...</li><li><a href="https://arxiv.org/abs/2309.08168">Draft &amp; Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding</a>：我们提出了一种新颖的推理方案，即 self-speculative decoding，用于在不需要辅助模型的情况下加速大语言模型 (LLMs)。该方法的特点是分为两个阶段...</li><li><a href="https://en.wikipedia.org/wiki/Dragon_curve">Dragon curve - 维基百科</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2304.15004">Are Emergent Abilities of Large Language Models a Mirage?</a>：最近的研究声称大语言模型展现出了 emergent abilities，即在较小规模模型中不存在但在较大规模模型中出现的能力。使 emergent abilities 引人入胜的是...</li><li><a href="https://x.com/_jasonwei/status/1784990066609414556?s=46&t=OICM4zGqs0OOATmLPoNFyw">Jason Wei (@_jasonwei) 的推文</a>：很喜欢这篇论文，它在 x 轴上绘制了 emergent abilities 与预训练损失的关系图，这实际上也是 @OriolVinyalsML 几年前提出的建议：https://arxiv.org/abs/2403.15796 ...</li><li><a href="https://arxiv.org/abs/2403.15796">Understanding Emergent Abilities of Language Models from the Loss Perspective</a>：最近的研究对语言模型中的 emergent abilities 是大模型所独有的这一信念提出了质疑。这种怀疑源于两个观察结果：1) 较小的模型也可以表现出...</li><li><a href="http://arxiv.org/abs/2404.16710">Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding</a>：我们提出了 LayerSkip，这是一种加速大语言模型 (LLMs) 推理的端到端解决方案。首先，在训练期间我们应用 layer dropout，对较早的层使用较低的 dropout 率，对较高的层使用较高的...</li><li><a href="https://arxiv.org/abs/2404.16873">AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs</a>：虽然最近大语言模型 (LLMs) 取得了显著的成功，但它们容易受到某些 jailbreaking attacks 的影响，从而导致生成不当或有害的内容。手动...</li><li><a href="https://arxiv.org/abs/2404.16030">MoDE: CLIP Data Experts via Clustering</a>：对比语言-图像预训练 (CLIP) 的成功依赖于图像和标题之间配对的监督，而这种配对在网络爬取的数据中往往存在噪声。我们提出了 Mixture of ...</li><li><a href="https://openreview.net/forum?id=8tYRqb05pVn">Linearly Mapping from Image to Text Space</a>：语言模型 (LMs) 可以通过冻结的图像编码器和 LM 输入之间的一个经过调整的线性层来“理解”图像，展示了它们在概念表示上的相似性...</li><li><a href="https://arxiv.org/abs/2310.03262">Predicting Emergent Abilities with Infinite Resolution Evaluation</a>：大语言模型 (LLMs) 的科学扩展需要对其 scaling properties 有全面的了解。然而，现有的关于 scaling properties 的文献仅产生了一个...</li><li><a href="https://huggingface.co/spaces/BlinkDL/RWKV-Gradio-2">RWKV-Gradio-2 - BlinkDL 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.jasonwei.net/blog/common-arguments-regarding-emergent-abilities">关于 emergent abilities 的常见争论 — Jason Wei</a>：这篇博文不代表我雇主（过去、现在或未来）的立场。我将回顾在讨论大语言模型 emergent abilities 时出现的一些常见争论...</li><li><a href="https://arxiv.org/abs/2404.16717">Embracing Diversity: Interpretable Zero-shot classification beyond one vector per class</a>：视觉语言模型实现了物体的开放世界分类，无需任何重新训练。虽然这种 zero-shot 范式标志着重大进步，但即使是当今最好的模型也表现出...</li><li><a href="http://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>：在写作和...</li>

人们在交谈时，有时会停下来思考。虽然以推理为中心的研究通常将推理框架化为回答问题或完成 Agent 任务的方法，但推理是至关重要的...</li><li><a href="http://arxiv.org/abs/2403.04642">Teaching Large Language Models to Reason with Reinforcement Learning</a>：来自人类反馈的强化学习 (\textbf{RLHF}) 已成为将 LLM 输出与人类偏好对齐的主流方法。受 RLHF 成功的启发，我们研究了其性能...</li><li><a href="http://arxiv.org/abs/2312.02179">Training Chain-of-Thought via Latent-Variable Inference</a>：当被指示使用“思维链”（CoT）提示逐步得出答案时，大语言模型 (LLMs) 解决问题的准确性和可解释性更高。人们还可以改进...</li><li><a href="https://www.youtube.com/watch?v=9QtS9sVBFM0">LLM Control Theory Seminar (April 2024)</a>：请关注我们在预印本中的新结果，“什么是咒语？LLM Prompting 的控制理论”：https://arxiv.org/abs/2310.04444 关注 twitter 和...</li><li><a href="https://github.com/continuousml/Awesome-Out-Of-Distribution-Detection">GitHub - continuousml/Awesome-Out-Of-Distribution-Detection: A professionally curated list of papers, tutorials, books, videos, articles and open-source libraries etc for Out-of-distribution detection, robustness, and generalization</a>：一个专业策划的论文、教程、书籍、视频、文章和开源库列表，用于分布外检测、鲁棒性和泛化 - continuousml/Awesome-Ou...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1234261844374454333)** (1 messages): 

- **通过非嵌入参数确定截止点**：一位参与者建议使用非嵌入参数作为确定模型截止点的方法。建议观察每个移除点的拟合曲线增量（delta）变得非常低的位置，这可能会在最初估计的 2 亿以下参数之外，得出一个**有根据的推测**。
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1233523889494032447)** (9 messages🔥): 

- **Anthropic 分享新研究见解**：Anthropic 可解释性团队发布了 [4 月更新](https://transformer-circuits.pub/2024/april-update/index.html)，包含进展和新兴研究想法。这包括 Scaling Laws、训练稀疏自编码器 (SAEs) 以及一个关于可解释性架构的项目。
  
- **发现 LLM 中的拒绝机制**：一篇来自 [AI Alignment Forum 的转发帖子](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)揭示了现代大语言模型 (LLMs) 如何通过微调来拒绝有害请求。研究表明，拒绝机制可能由网络中的单一方向介导。
  
- **权重正交化与微调**：在针对特定行为微调 LLMs 的背景下，一位成员假设*权重正交化可以被视为一种手动微调形式*，用以影响网络行为。
  
- **拒绝方向与 Rank-1 LoRA 微调探讨**：一位成员提出，如果使用随机梯度下降 (SGD) 进行 *Rank-1 LoRA (Low-Rank Adaptation) 微调*，网络可能会学习到“拒绝方向”的负值。
  
- **Llama.cpp 集成控制向量技术**：控制向量（一种与讨论内容类似的技术）已添加到 llama.cpp 中，如这个 [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/5970) 所示，这得益于与 Nous Research 的合作。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://transformer-circuits.pub/2024/april-update/index.html">Circuits Updates - April 2024</a>：未找到描述</li><li><a href="https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction">Refusal in LLMs is mediated by a single direction — LessWrong</a>：这项工作是 Neel Nanda 在 ML Alignment &amp; Theory Scholars Program - 2023-24 冬季队列中的一部分，由……共同指导。</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5970">Add support for control vectors by vgel · Pull Request #5970 · ggerganov/llama.cpp</a>：非常感谢 Nous Research，他们的支持和合作使这项工作成为可能！此 PR 引入了一种新的激活干预技术：控制向量（也称为引导向量、概念向量……）
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1233345725513859082)** (5 messages):

- **PR 提交中的 CLA 困惑**：一位成员遇到了贡献者许可协议 (CLA) 显示为未签署的问题，尽管他们已经签署了。这可能是由于 GitHub 在 commit 中匿名化了他们的电子邮件。该问题已被确认，并同意进一步调查。
- **PR 中失败检查的不确定性**：对于提交的 Pull Request 中出现的失败检查，成员表示担忧并询问是否与其更改有关。该问题已通过审查，初步认定为无关。
- **Chat Template 分支停滞的询问**：一位成员询问了关于添加 Chat Template 功能的分支的进展和活动情况，指出最后一次 commit 是在两个月前。目前尚无关于当前状态或进展的即时更新。
- **Evaluation Harness 的 Prompt 多样性**：一位成员指出 Evaluation Harness 中缺乏满足特定模型微调的变量 Prompt 格式。另一位参与者建议使用自定义的 `!function`，以便根据模型启用不同的 Prompt。

**提到的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1745">add task for mmlu evaluation in arc multiple choice format by jonabur · Pull Request #1745 · EleutherAI/lm-evaluation-harness</a>：此 PR 增加了 `mmlu_arc_style` 任务，该任务以与 ARC 评估相同的方式呈现 MMLU 问题（将答案作为续写的 loglikelihood，而不是选择字母...）

---

**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1233812243565514855)** (1 messages): 

- **对集群设置实践的担忧**：一条评论强调，在集群设置期间无法保证使用了正确版本的 `tokenizers`，因为有人可能会在不使用固定版本的情况下盲目执行 `pip install tokenizers`。有人指出这可能会影响任何运行，必须确保记录 Python 环境中的内容，以确定所使用的版本。

---

**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1233463865178722315)** (3 messages): 

- **Soliloquy 8B 转为付费模型**：[Soliloquy 8B](https://openrouter.ai/models/lynn/soliloquy-l3) 现在开始收费，价格为 **每 1M tokens $0.1**。此价格更新反映了 OpenRouter LLC 最近的政策变动。
- **Soliloquy 8B 价格上涨**：[Soliloquy 8B](https://openrouter.ai/models/lynn/soliloquy-l3) 的使用价格再次修订为 **每 1M tokens $0.2**。新费率在初始定价推出后不久即发布。
- **路由更新与修正**：`anthropic/claude-instant-1` 模型路由已更新为 `claude-instant-1.2`，并修正了关于 `anthropic/claude-2.0` 的路由错误，由于其仍为有效的模型 ID，服务已恢复。
- **恢复 Claude v2.1 及其变体**：在最近关于旧版 Claude 模型的混淆中澄清了模型可用性后，[Anthropic: Claude v2.1](https://openrouter.ai/models/anthropic/claude-2.1) 模型及其 `:beta` 变体已恢复。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/anthropic/claude-2.1)">Anthropic: Claude v2 by anthropic | OpenRouter</a>：Claude 2 为企业带来了关键能力的提升——包括行业领先的 200K token 上下文窗口，显著降低的模型幻觉率，系统 Prompt 等...</li><li><a href="https://openrouter.ai/models/lynn/soliloquy-l3)">Lynn: Llama 3 Soliloquy 8B by lynn | OpenRouter</a>：Soliloquy-L3 是一款快速、高性能的角色扮演模型，旨在提供沉浸式、动态的体验。基于超过 2.5 亿 token 的角色扮演数据训练，Soliloquy-L3 拥有广博的知识库...</li><li><a href="https://openrouter.ai/models/lynn/soliloquy-l3>)">Lynn: Llama 3 Soliloquy 8B by lynn | OpenRouter</a>：Soliloquy-L3 是一款快速、高性能的角色扮演模型，旨在提供沉浸式、动态的体验。基于超过 2.5 亿 token 的角色扮演数据训练，Soliloquy-L3 拥有广博的知识库...
</li>
</ul>

</div>

---

**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1233393770071064597)** (4 messages):

- **探索 Syrax**：一位成员表示有兴趣尝试 **Syrax** 并提供支持，通过发送好友请求发起私聊以进行进一步合作。
- **好友请求已接受**：另一位社区成员对提供的支持表示感谢，并确认接受了好友请求。
- **对展示印象深刻**：对正在进行的讨论或展示的项目表达了简短的赞赏，反映出积极的印象。
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1233324831588487218)** (311 messages🔥🔥): 

- **Claude 模型的古怪行为揭秘**：成员们讨论了 Claude 模型通过 OpenRouter 返回不完整输出或 HTTP 524 错误的问题。澄清后发现 Claude 模型的最大生成限制为 4k tokens，最高可读取 200k tokens，而正确的设置可以改善 API 响应。

- **Lemmyle 剖析 WLM-2 托管经济学**：对 WLM-2 的托管成本进行了深入分析，推测利润空间可能很小，取决于 GPU 利用率、电力成本以及闲置 GPU 的潜在收益等多种因素。

- **FireLLaVA 悄然进入多模态领域**：有人对 FireLLaVA 的低调发布进行了思考，这是一款以启动速度快著称的开源多模态模型，是 OpenRouter 生态系统的一个重要补充。

- **部署困境与节俭的前端方案**：一位成员寻求一个可以部署在共享主机上的简单前端，以便让家人使用其 OpenRouter 服务，而无需订阅多个 OpenAI 账号。建议包括利用 Vercel 的免费层级，或选择更实惠的 VPS 供应商（如 Contabo）。

- **Cohere 在 OpenRouter 上下文中的难题**：一位成员发现通过 OpenRouter 使用 Cohere 模型时，输出结果与直接调用 API 相比存在奇怪的差异，生成内容与提示词无关。经澄清，Cohere 的 web connector 支持尚在处理中，预计会加入 OpenRouter，但目前暂不可用。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.anthropic.com/claude/docs/models-overview">Models overview</a>：未找到描述</li><li><a href="https://cws-docs.pages.dev/en/">首页 | ChatGPT Web Share 文档</a>：未找到描述</li><li><a href="https://openrouter.ai/playground">OpenRouter</a>：LLM 和其他 AI 模型的路由</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b?tab=apps">microsoft 的 WizardLM-2 8x22B | OpenRouter</a>：WizardLM-2 8x22B 是 Microsoft AI 最先进的 Wizard 模型。与领先的专有模型相比，它表现出极具竞争力的性能，并始终优于所有现有的...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-8b-instruct?tab=api">meta-llama 的 Meta: Llama 3 8B Instruct | OpenRouter</a>：Meta 最新的模型类别 (Llama 3) 发布了多种尺寸和版本。这个 8B 指令微调版本针对高质量对话场景进行了优化。它展示了强大的...</li><li><a href="https://openrouter.ai/models/fireworks/firellava-13b">fireworks 的 FireLLaVA 13B | OpenRouter</a>：首个允许商业使用的开源 LLaVA 模型。该视觉语言模型完全基于开源 LLM 生成的指令遵循数据进行训练。</li><li><a href="https://www.clay.com/">Clay - 规模化个性化外联</a>：结合 50 多个数据提供商、实时抓取和 AI，发送 1 对 1 的个性化活动，预订更多会议。</li><li><a href="https://www.cyon.ch/hosting/managed-server">托管服务器：您自己的服务器，总部位于瑞士</a>：未找到描述</li><li><a href="https://openrouter.ai/models/haotian-liu/llava-13b?tab=activity">haotian-liu 的 Llava 13B | OpenRouter</a>：LLaVA 是一个大型多模态模型，结合了视觉编码器和 Vicuna，用于通用视觉和语言理解，实现了模仿 [GPT-4](/models/open... 令人印象深刻的聊天能力。
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1233372786274074734)** (169 messages🔥🔥):

- **华盛顿的巫师：未更改的仓库**：尽管有传言，Microsoft 的 [WizardLM 模型](https://huggingface.co/collections/microsoft/wizardlm-661d403f71e6c8257dbd598a?history=true)并未被 Microsoft 删除；一名成员澄清说 wizardlm 团队负责了这些更改。他们还确认 [WizardLM 仓库](https://github.com/nlpxucan/WizardLM) 仍然公开可用。
- **领域特定 LLM 的微调 vs. RAG**：新成员询问了针对特定领域语言模型的微调，质疑其相对于使用检索增强生成 (RAG) 的必要性。对话中提到了 *OpenBioLLM* 等示例，并引用了一篇[专注于医疗领域的 LLM 论文](https://arxiv.org/abs/2311.16079)以供进一步阅读。
- **对话 Tokenization 问题的配置**：针对 LLaMA-3 等模型的 Tokenization 策略进行了深入讨论，包括手动安装最新版本 fastchat 格式化器的必要性，并引用了一个相关的 [axolotl Pull Request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553) 以获取正确的对话格式模板。
- **量化与模型退化之争**：成员们辩论了量化策略对 LLM 的影响，特别是对比了 *4bit lora* 和 *4bit qlora* 方法。共识是量化敏感度因训练而异，一位成员引用了一个 [Twitter 帖子](https://x.com/rohanpaul_ai/status/1784972618472317180)，讨论了在像 LLaMA-3 这样经过更广泛训练的模型中，退化现象更为显著。
- **防止 OOM 的样本打包 (Sample Packing) 说明**：一位成员寻求关于 multipack 采样及其与内存溢出 (OOM) 错误关系的澄清。解释指出，采样不会影响模型允许的最大序列长度，只是将多个样本打包到最大序列长度中，而不会改变上下文大小 (context size)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2311.16079">MEDITRON-70B: Scaling Medical Pretraining for Large Language Models</a>：大语言模型 (LLM) 有潜力使医疗知识的获取民主化。虽然已经做出了许多努力来利用和提高 LLM 的医疗知识和推理能力，但...</li><li><a href="https://huggingface.co/collections/microsoft/wizardlm-661d403f71e6c8257dbd598a?history=true>">WizardLM - Microsoft 集合</a>：未找到描述</li><li><a href="https://x.com/rohanpaul_ai/status/1784972618472317180">Rohan Paul (@rohanpaul_ai) 的推文</a>：量化对 LLaMA 3 的危害比对 LLaMA 2 更大。llama cpp 仓库中的这个 PR 对此进行了深入调查。（Perplexity 衡量模型预测下一个 Token 的能力，数值越低越好...）</li><li><a href="https://arxiv.org/abs/2311.08545">Efficient Continual Pre-training for Building Domain Specific Large Language Models</a>：大语言模型 (LLM) 展示了卓越的通用领域能力。传统上，为特定领域量身定制的 LLM 是从头开始训练的，以便在处理特定领域任务时表现出色。在本文中...</li><li><a href="https://github.com/lyogavin/Anima/tree/main/air_llm">Anima/air_llm at main · lyogavin/Anima</a>：33B 中文 LLM，DPO QLORA，100K 上下文，使用单个 4GB GPU 进行 AirLLM 70B 推理 - lyogavin/Anima</li><li><a href="https://github.com/lm-sys/FastChat">GitHub - lm-sys/FastChat: 用于训练、部署和评估大语言模型的开放平台。Vicuna 和 Chatbot Arena 的发布仓库。</a>：用于训练、部署和评估大语言模型的开放平台。Vicuna 和 Chatbot Arena 的发布仓库。 - lm-sys/FastChat</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553">feat: Add LLaMA-3 instruct prompt strategies for fine-tuning by 0-hero · Pull Request #1553 · OpenAccess-AI-Collective/axolotl</a>：描述：这建立在以下 PR 的基础之上并包含其更改：#1542 #1539。在合并此项之前，需要先合并来自 @TJ-Solergibert 的 Fastchat PR lm-sys/FastChat#3257。动机...</li><li><a href="https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L524>">FastChat/fastchat/conversation.py at main · lm-sys/FastChat</a>：用于训练、部署和评估大语言模型的开放平台。Vicuna 和 Chatbot Arena 的发布仓库。 - lm-sys/FastChat</li><li><a href="https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py">FastChat/fastchat/conversation.py at main · lm-sys/FastChat</a>：用于训练、部署和评估大语言模型的开放平台。Vicuna 和 Chatbot Arena 的发布仓库。 - lm-sys/FastChat
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1234096033047248916)** (37 条消息🔥):

- **Fast Fourier Transform 的内存需求**：关于在 2x24GB 显卡上使用 zero3 运行 Fast Fourier Transform (FFT) 所需的大量内存讨论。一位成员建议可能需要 **167GB 的 RAM**，并对内存不足表示遗憾。

- **通过 torchtune 探索降低 VRAM 占用**：一位成员建议尝试 **torchtune**，并指出其专注于减少 VRAM 使用。另一位成员讨论了使用 **FSDP** (Fully Sharded Data Parallel) 的问题，但报告称训练虽然开始，却在没有进展或报错的情况下卡死。

- **Fast Fourier Transform 导致磁盘占用飙升**：在尝试训练模型时，系统的交换内存（swap memory）飙升至 62GB，导致内存溢出（out-of-memory）错误。参与者对过高的磁盘和交换内存占用表示惊讶，因为理论上该任务应该能容纳在单张 48GB 显卡的配置中。

- **实验用的 ZeroGPU 访问权限**：一位成员提到他们拥有 Huggingface **Zero 项目**的访问权限，引发了关于潜在测试的讨论。该项目旨在为 Huggingface Spaces 提供免费的 GPU 访问，并支持在多个 GPU 上同时运行 Spaces。

- **日志共享与迭代困扰**：一位用户分享了他们的 **wandb.ai 日志**链接，供那些对 Fast Fourier Transform 试验细节感兴趣的人参考。他指出迭代时间极长，达到了 800 秒，而 qlora 迭代仅需 17 秒，突显了性能问题。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://wandb.ai/vsungwaterloo/llama3_tests/runs/5wuupz0t?nw=nwuservsungwaterloo">vsungwaterloo</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1233320344698486847)** (23 条消息🔥): 

- **排查 AttributeError**：一位用户遇到了与 `'TextIteratorStreamer'` 没有 `'empty'` 属性相关的 `AttributeError`。考虑到他们使用的是 transformers 4.40.0 版本，他们对该函数的有效性提出了质疑。
  
- **关于 Llama-Pro 方法的咨询**：针对 Jeremy Howard 强调的 **llama-pro 方法**进行了多次讨论。分享了 GitHub 仓库链接（[fsdp_qlora](https://github.com/AnswerDotAI/fsdp_qlora)），指向一种 4-bit 量化的 Llama-Pro 微调方法，对话围绕该方法是否在 axolotl 中可用以及是否需要提交 Pull Request 展开。

- **在 Twilio 中集成自定义音频录制**：一位用户解释了他们在 Twilio 中集成自定义音频录制的努力，以及如何实时捕获和存储音频，同时能够对录制的音频做出响应。

- **合并 QLORA Adapter 微调**：用户讨论了在进行 Q/A 风格的额外微调之前，是否需要合并 qlora adapter 微调模型，以及后续微调对保持模型特性的影响。进一步的对话提到了将对话模型和补全模型合并为一个微调任务，并参考了社区展示中的一个示例。

- **用于更快速 LLM 微调的 PEFT 模型**：简要提到了 *unsloth peft* 模型，据称它可以显著加快 Mistral 等 LLM 的微调速度并减少内存占用，尽管由于额外的优化，其加载方式可能与 Hugging Face 模型不同。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/AnswerDotAI/fsdp_qlora">GitHub - AnswerDotAI/fsdp_qlora: Training LLMs with QLoRA + FSDP</a>: 使用 QLoRA + FSDP 训练 LLM。通过在 GitHub 上创建账号为 AnswerDotAI/fsdp_qlora 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/tree/main">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral 和 Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/AnswerDotAI/fsdp_qlora/blob/467933f713cc7808564cbfac3524e">GitHub - AnswerDotAI/fsdp_qlora at 467933f713cc7808564cbfac3524e75aadd04987</a>: 使用 QLoRA + FSDP 训练 LLM。通过在 GitHub 上创建账号为 AnswerDotAI/fsdp_qlora 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: 尽管提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#community-showcase">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: 尽管提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/AnswerDotAI/fsdp_qlora/blob/467933f713cc7808564cbfac3524e75aadd04987/train.py#L564">fsdp_qlora/train.py at 467933f713cc7808564cbfac3524e75aadd04987 · AnswerDotAI/fsdp_qlora</a>: 使用 QLoRA + FSDP 训练 LLM。通过在 GitHub 上创建账号为 AnswerDotAI/fsdp_qlora 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1233726782176231425)** (44 messages🔥): 

- **GPU 扩展与 Batch Sizes 详解**：对话详细探讨了将 GPU 数量从 4 个扩展到 8 个并调整 micro batch size 的复杂性。澄清了虽然总 batch size 可能保持不变，但梯度累积 (gradient accumulation)、学习率缩放 (learning rate scaling)、并行策略和通信开销等因素会有所不同，并影响训练动态和性能结果。

- **关于跨 GPU 加载模型的疑问**：提出了在使用多个 GPU 时，模型是完整加载还是拆分加载的问题。解释说模型既可以以完整大小加载，也可以跨 GPU 进行分片 (sharded)，这一技术由 Fully Sharded Data Parallelism (FSDP) 和 DeepSpeed 的 ZeRO Stage 3 等优化手段实现，有助于高效利用硬件资源。

- **LoRA vs. QLoRA – 适配技术揭秘**：讨论涉及 LoRA 和 QLoRA 之间的区别，详细说明了后者如何通过增加量化 (quantization) 来扩展 LoRA，从而在微调和部署期间进一步降低计算成本和显存需求。

- **Axolotl 的数据集裁剪策略**：针对在 Axolotl config 中裁剪数据集的情况，建议采用一种不直接指定数据集百分比的方法，而是修改数据集加载逻辑以包含子采样 (subsampling) 步骤，可能使用 `datasets` 库函数提供的方法。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/accelerate/tree/main/docs/source/concept_guides/big_model_inference.md#L45L298)">accelerate/docs/source/concept_guides/big_model_inference.md at main · huggingface/accelerate</a>: 🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简单方法，支持自动混合精度（包括 fp8），以及易于配置的 FSDP 和 DeepSpeed 支持.....</li><li><a href="https://github.com/huggingface/peft/tree/main/docs/source/accelerate/fsdp.md#L172L291),">peft/docs/source/accelerate/fsdp.md at main · huggingface/peft</a>: 🤗 PEFT: 最先进的参数高效微调 (Parameter-Efficient Fine-Tuning)。 - huggingface/peft</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=650c6038-10b5-46b9-aacc-ce5f8e81ff17)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=8a7a1373-bad8-460c-bb87-71c8bb2450bd)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c42603f2-ce0e-4806-aa15-b77ac3002f7d)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e8f52866-9f91-4fd0-a77d-3662bc1b431b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://github.com/huggingface/peft/tree/main/docs/source/accelerate/deepspeed.md#L177L285)">peft/docs/source/accelerate/deepspeed.md at main · huggingface/peft</a>: 🤗 PEFT: 最先进的参数高效微调 (Parameter-Efficient Fine-Tuning)。 - huggingface/peft</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=88899b50-3175-4ee7-a830-13effdde1bbf)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1233503142876938330)** (12 条消息🔥): 

- **LLaMa Prompt 支持咨询**：一位成员询问 **axolotl 是否支持 ShareGPT 的 LLaMa 3 prompt 格式**。回复指出，在 **OpenAccess-AI-Collective/axolotl** 文档中没有提到对特定 "llama 3" 模型的支持。
- **微调 QLoRA 模型**：一位成员分享了他们使用 Mistral-7B 通过 **QLoRA 创建微调文本补全模型**的成功经验。他们寻求关于如何使模型具有对话能力的指导，并得到的建议是：可以直接使用他们的 QLoRA 适配模型在 Q/A 数据集上进行微调。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=855017a7-9a6e-469b-857b-bc1b391a15fe)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c43b2542-b6b0-495d-8bd6-97b7dc28fb89)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1233746542188298270)** (2 条消息): 

- **Modular 提交量上升**：自 stdlib 开源以来，**23%** 的提交是针对 **modularml/mojo** 的。这表明该项目的活跃度和贡献量大幅增加。
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1233532644608839801)** (4 条消息): 

- **Modular 推文链接分享**：💬︱twitter 频道的成员分享了来自 **Modular** 的多条推文。相关推文包括更新或公告，链接如下：[推文 1](https://twitter.com/Modular/status/1783968545052987485), [推文 2](https://twitter.com/Modular/status/1785036097292292472), [推文 3](https://twitter.com/Modular/status/1785036111804575967), 以及 [推文 4](https://twitter.com/Modular/status/1785036126224548005)。
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1233522215174144010)** (1 条消息): 

- **MAX Engine 助力多模态搜索**：[Modular](https://www.modular.com/blog/multimodal-search-with-snowflake-embedding-and-max-engine) 最近的博客文章讨论了结合文本和视觉数据的多模态搜索的优势。MAX Engine 在之前的基准测试中已经超越了 PyTorch eager 和 ONNX runtime，它同样能够优化多模态模型的推理。

**相关链接**: <a href="https://www.modular.com/blog/multimodal-search-with-snowflake-embedding-and-max-engine">Modular: Multimodal Search with Snowflake Embedding and MAX Engine</a>：我们正在为全球构建下一代 AI 开发者平台。查看我们的最新文章：结合 Snowflake Embedding 和 MAX Engine 的多模态搜索。

---

**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1234433929331740702)** (2 条消息): 

- **Mojo 安装故障排除**：一位用户报告了在 Python 3.12.3 上安装 **Modular (Mojo 🔥)** 时遇到的问题。回复建议使用 Conda 虚拟环境，并提供了指导链接：[Modular Python 手册](https://docs.modular.com/mojo/manual/python/) 和 [Modular 博客文章](https://www.modular.com/blog/using-mojo-with-python)，强调 **Mojo 是 Python 的超集**，并且与 Python 模块兼容。
- **在 Mac M1 上运行**：另一位成员指出，他们正在 Mac M1 上成功运行最新版本的 Mojo（包括 *nightly* 版本）和 Python 3.12.3。他们建议使用 Conda 以简化设置，并指出 Mojo 的目标是兼容 Python 代码和现有的 Python 包。

**相关链接**: <a href="https://docs.modular.com/mojo/manual/python/">Python integration | Modular Docs</a>：同时使用 Python 和 Mojo。

---

**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1233446680603394170)** (113 条消息 🔥🔥): 

- **从 Python 切换到 Mojo 的问题**：一位用户分享了 Python 代码并寻求将其转换为 Mojo 的帮助。另一位用户提供了详细的 [Mojo 转换示例](https://docs.modular.com/mojo/manual/functions)，并解释了 Mojo 中的函数声明和变量类型。

- **ModularBot 插话**：ModularBot 介入，庆祝用户 `@110077104611172352` 达到 5 级，用户 `@289473226147495936` 达到 1 级。随后祝贺 `@932397073427476521` 达到 18 级，ModularBot 还开玩笑说要举办宴会庆祝。

- **矩阵切片与内存所有权**：一位 Mojo 用户询问如何在不进行额外分配的情况下创建列表子集的非所有权视图（non-owning view）。回复明确指出，对于间接内存访问，应该使用 `Buffer` 类型而不是 `List`，因为 List 拥有其数据的所有权，而 Buffer 正在针对生命周期管理进行重新设计。

- **关于 Intel Mac 版 Mojo 的咨询**：当被问及 Intel Mac 版 Mojo 时，一位用户回答说有望很快获得支持，但目前使用 Playground 是唯一的选择。

- **排查矩阵实现问题**：一位用户在 Mojo 中进行矩阵除法时遇到困难，原因是尚未实现 `__truediv__` 函数。建议该用户检查代码，并确保仅对非零值执行操作。

- **关于 Mojo 与现有库集成的讨论**：讨论了 Mojo 语言的目标，强调 Mojo 旨在融入 Python 生态系统并利用现有库，而不是完全取代它们。指出 Mojo 的长期方向包括无缝使用 Numpy 等现有工具。

- **Discord 中的等级与学习**：用户们讨论了他们在频道中的等级进度；一位用户在一年后升到了 18 级，而其他用户则对排名方法论提出质疑，因为不同用户的专业水平差异很大。
<div class="linksMentioned">

<strong>相关链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/pokemon-pikachu-clap-clapping-clapping-gif-gif-13465728489229726846">Pokemon Pikachu GIF - Pokemon Pikachu Clap - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://docs.modular.com/mojo/stdlib/memory/memory#memset">memory | Modular Docs</a>：定义了内存操作函数。</li><li><a href="https://docs.modular.com/mojo/roadmap#parametric-aliases).">Mojo🔥 路线图与尖锐边缘 | Modular Docs</a>：我们的 Mojo 计划摘要，包括即将推出的功能和我们需要修复的问题。</li><li><a href="https://github.com/modularml/mojo/discussions/2270">为什么该函数的参数化版本比普通版本慢？ · modularml/mojo · Discussion #2270</a>：你好，我写了一些基准测试来查看 Mojo 在 matmul 中的表现，参考指南如下：https://docs.modular.com/mojo/notebooks/Matmul。然而，我注意到我的参数化版本很慢...</li><li><a href="https://docs.modular.com/mojo/notebooks/Matmul">Mojo 中的矩阵乘法 | Modular Docs</a>：了解如何利用 Mojo 的各种函数来编写高性能的 matmul。</li><li><a href="https://github.com/mikowals/dynamic_vector.mojo/blob/main/README.md#python-style-slices---var-evens--vec02).">dynamic_vector.mojo/README.md at main · mikowals/dynamic_vector.mojo</a>：一个实验性的 Mojo 标准库 DynamicVector 替代方案，展示了使用 References 的新功能 - mikowals/dynamic_vector.mojo</li><li><a href="https://github.com/modularml/mojo/issues">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://gist.github.com/modularbot/c67e0a66a97aa32314d248f4721f75e2">playground.mojo</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/modularbot/1a5beaf165761b55e2f743b3151210eb">playground.mojo</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/modularml/mojo/issues/620">[功能请求] 原生 Windows 支持 · Issue #620 · modularml/mojo</a>：查看 Mojo 的优先级。我已经阅读了路线图和优先级，并相信此请求符合优先级。您的请求是什么？对 Windows 的原生支持。什么时候可用？...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/)** (1 条消息): 

uncle_jee: 使用 Mojo 编写一个 Mojo 社区
https://github.com/shadowqcom/mojo_dev
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1233517938711990344)** (5 条消息): 

- **制作更好的教程**：rd4com 强调了[制作教程的技巧](https://discord.com)，重点是使用表情符号进行视觉参考、语言简洁、命名清晰、避免信息过载、逐步增加复杂性以及通过迭代进行完善。他们还强调了链接到 Mojo 文档以及在之前内容基础上进行逻辑构建的重要性。

- **用于文档的 Diátaxis 框架**：sophiaglencairn 分享了 [Diátaxis](https://diataxis.fr/) 的链接，这是一种创建技术文档的系统方法，概述了四种类型的文档需求：教程（tutorials）、操作指南（how-to guides）、技术参考（technical reference）和解释（explanation）。Diátaxis 解决了文档中的内容、风格和架构问题，使最终用户和创建者都能受益。

**提到的链接**：<a href="https://diataxis.fr/">Diátaxis</a>：未找到描述

  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1233455459260956776)** (55 条消息🔥🔥): 

- **探索 `__copyinit__` 和 GitHub Gists**：讨论围绕 `__copyinit__` 的行为展开，以及实现写时复制（copy-on-write）语义是否是类型作者的责任。对话指向了一个[特定的 Gist](https://gist.github.com/modularbot/6aed759930420cd70f38795dbcb874fe) 以提供背景信息。

- **字典性能的复杂性**：讨论了关于 Mojo 中字典的性能问题，引用了 Mojo 和 Python 之间显著的速度差异。一位成员分享了他们移植分词器（tokenizer）的经验，并链接到了[相关讨论](https://github.com/modularml/mojo/discussions/1747)和[分词库](https://github.com/karpathy/minbpe)以供参考。

- **Compact-dict 库带来希望**：在关于字典性能的对话中，[Compact-dict 库](https://github.com/mzaks/compact-dict)被提议作为标准 Mojo 字典的更快替代方案，尽管它不存储键，并且将来可能需要根据使用场景进行更改或增加额外功能。

- **内存分配查询**：成员们询问了 `stack_allocate` 与堆分配方法（如 `DTypePointer.alloc`/`Pointer.alloc`）在性能和功能上的差异。双方交流了何时使用栈或堆，并分享了关于它们成本差异的见解，强调通常栈分配比堆分配更快且更简单。

- **为纠错码优化 SIMD 操作**：为了让纠错码库获得更好的性能，一位成员寻求了关于使用 `SIMD` 优化函数的建议。对话包括关于函数内联、`fma` 的使用以及潜在数学技巧改进的讨论。提到的具体项目是 [mocodes](https://github.com/alainrollejr/mocodes)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://stackoverflow.com/questions/102009/when-is-it-best-to-use-the-stack-instead-of-the-heap-and-vice-versa">何时最适合使用栈而不是堆，反之亦然？</a>：在 C++ 中，何时最适合使用栈？何时最适合使用堆？</li><li><a href="https://stackoverflow.com/questions/102009/when-is-it-best-to-use-the">何时最适合使用栈而不是堆，反之亦然？</a>：在 C++ 中，何时最适合使用栈？何时最适合使用堆？</li><li><a href="https://gist.github.com/modularbot/6aed759930420cd70f38795dbcb874fe">playground.mojo</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/alainrollejr/mocodes">GitHub - alainrollejr/mocodes: 使用 Mojo 进行纠错（编）解码</a>：使用 Mojo 进行纠错（编）解码。通过在 GitHub 上创建账号为 alainrollejr/mocodes 的开发做出贡献。</li><li><a href="https://github.com/mzaks/compact-dict">GitHub - mzaks/compact-dict: Mojo 中快速且紧凑的 Dict 实现 🔥</a>：Mojo 中快速且紧凑的 Dict 实现 🔥。通过在 GitHub 上创建账号为 mzaks/compact-dict 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/discussions/1747">为什么 Mojo 的字典（或 for 循环）比 Python 的慢？ · modularml/mojo · Discussion #1747</a>：我使用 Mojo (v. 0.7.0) 的字典数据结构来计算一个拥有 2.3 亿多个单词的文件中的词频，并使用 Python 进行了同样的操作。令人惊讶的是，Python 快了 7 倍...</li><li><a href="https://github.com/karpathy/minbpe">GitHub - karpathy/minbpe: 用于 LLM 分词中常用的字节对编码 (BPE) 算法的极简、干净的代码。</a>：用于 LLM 分词中常用的字节对编码 (BPE) 算法的极简、干净的代码。 - karpathy/minbpe</li><li><a href="https://github.com/modularml/mojo/pull/2351">[stdlib] 修复 mzaks 的字典探测错误 · Pull Request #2351 · modularml/mojo</a>：修复了 #1729</li><li><a href="https://github.com/modularml/mojo/pull/2250">[Proposal] 改进 mzaks 的 hash 模块 · Pull Request #2250 · modularml/mojo</a>：该提案基于 #1744 中开始的讨论
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1234221922607435858)** (3 条消息): 

- **持续的 MAX 优化**：团队在每个版本中都会定期优化 MAX。了解个人使用的具体核心类型和模型可以为性能增强提供进一步的见解。
- **澄清速度提升**：一位成员指出 TensorFlow (tf) 和 PyTorch 之间报告的速度提升存在差异，认为由于每秒查询数 (QPS) 的不同，它们不应该相同。
- **确认正确的加速输出**：另一位成员确认，在更新 max 示例仓库并清理 performance-showcase 目录中的 .cache 后，看到了反映比例 QPS 改进的正确加速数字。
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1233500471025729616)** (85 条消息🔥🔥): 

- **讨论 Nightly 分支的频繁更新**：自动化方面的挑战延迟了每个工作日发布 nightly 分支的目标，人们担心代码合并与分支中出现提交之间的延迟会导致难以修复冲突。[目前正在讨论](https://discord.com/channels/10875304973133578)寻找解决方案，以确保 nightly stdlib 能够使用发布的 nightly 编译器正确构建和运行。

- **Nightly Mojo 编译器发布通知**：新的 nightly Mojo 编译器公告强调了更新和更改的可用性，并提供了[详细的 Pull Request](https://github.com/modularml/mojo/pull/2418/files) 和[可供审查的变更日志 (changelog)](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。

- **关于 Mojo 中重载（Overloads）和 Traits 的讨论**：围绕重载的行为一致性和 Traits 的使用展开了辩论，涉及参数化算法等语言特性。社区正在权衡不同方法的利弊，如重载、优先级装饰器和返回类型变体，同时也对通过类型信息修改对象行为时可能导致的混淆和 Bug 表示担忧。

- **Stable 与 Nightly 版本间的代码执行差异**：一位用户报告了一个问题，即在 Mojo 稳定版中正常运行的代码在 Nightly 版本中报错，这表明 Nightly 版本可能存在文件句柄生命周期管理问题。该讨论促使在 [GitHub 上开启了一个 issue](https://github.com/modularml/mojo/issues/2429)。

- **Mojo 标准库中的导入挑战**：一位用户在将 `math` 包中的函数导入 `string.mojo` 和 `string_literal.mojo` 文件时遇到困难。这被解释为一种设计决策，旨在避免标准库中开源和闭源部分之间的循环依赖。推荐的解决方法是在标准库的开源部分重新实现必要的数学函数。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://mojodojo.dev/mojo-team-answers.html#overloading-return-type>">Mojo Team Answers | Mojo Dojo</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/issues/2429">[mojo-nightly] struct lifetime issue · Issue #2429 · modularml/mojo</a>：Bug 描述：在以下测试 demo 中，析构函数似乎在文件句柄上被调用，而不是执行 move。该 demo 在稳定版运行正常，但在 Nightly 版本出现以下错误...</li><li><a href="https://github.com/modularml/mojo/pull/2418/files">[stdlib] Update stdlib corresponding to 2024-04-26 nightly/mojo by patrickdoc · Pull Request #2418 · modularml/mojo</a>：此 PR 根据今天的 Nightly 版本更新了标准库的内部提交：mojo 2024.4.2621。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>：Mojo 编程语言。通过创建 GitHub 账号为 modularml/mojo 做出贡献。
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1233437853716058245)** (6 条消息): 

- **构建 LLM 应用的工作坊材料**：[Llama Index](https://twitter.com/llama_index/status/1783877951278432733) 宣布与 AWS 举办工作坊，展示了 **3 种 LLM 应用开发模式**，包括使用 S3 进行数据摄取以及使用 AWS Bedrock 生成 embeddings。
- **Llama Index 参与 ML Security 播客**：Llama Index 联合创始人在 [mlsecops 播客](https://twitter.com/llama_index/status/1783963718256411126)上讨论了 **基于 LLM 的应用前景和数据安全**，同时提到了 LlamaParse 和 LlamaCloud 等工具。
- **面向生产的 RAG 教程系列**：Marco Bertelli 推出了一个 **9 部分组成的系列教程**，重点关注将 RAG 从原型推向生产环境，并[概述了部署所需的架构组件](https://twitter.com/llama_index/status/1784257178758697272)。
- **通过多阶段检索增强 RAG**：KX Systems 的 Michael R. 发表文章建议使用 Llama Index 和 Cohere reranking 进行 **多跳检索过程**，以改善上下文并减少 LLM 的幻觉，详见其[帖子](https://twitter.com/llama_index/status/1784363604340576615)。
- **自主 Agent 的长期记忆**：介绍了 *memary*，这是一个使用 **知识图谱** 实现长期记忆的参考实现，旨在增强使用 LLM 的自主 Agent 的记忆功能，如[此推文中所述](https://twitter.com/llama_index/status/1784604356224164186)。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1233371418675380244)** (155 条消息 🔥🔥):

- **awsbedrock 与 LlamaIndex 的问题**：一位成员在尝试将 awsbedrock 与 LlamaIndex 结合使用时遇到了错误，触发了 botocore 的 "NoRegionError"。在按照建议确保指定了 `region_name` 后，问题得到了解决。
- **在 LlamaIndex 中使用本地 LLM**：成员们分享了 LlamaIndex 的文档链接和本地设置 LLM 的示例，特别提到了 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/)中一个使用 `BAAI/bge-small-en-v1.5` 和 `Mistral-7B` 的“5 行代码”示例。
- **LlamaIndex 导入问题已解决**：几位成员讨论了与 llama-index 包（如 `llama-index-llms-ollama`）相关的导入错误排查。解决方案包括单独安装特定包并确认正确的安装步骤。
- **更新 Vector Stores 上的索引和文档**：对话集中在如何使用 LlamaIndex 更新 Pinecone 上的索引以及向现有向量添加 metadata 键。一位成员建议，使用相同的 ID 更新 node 将会覆盖它。然而，对于在不修改向量的情况下添加 metadata，目前还没有提供直接的解决方案。
- **使用 LlamaIndex 检索文档**：成员们询问了如何通过 `query_engine.retrieve()` 检索多个文档，同时确保检索到的文档具有多样性。建议包括向现有向量添加 metadata 键，以及在创建 retriever 时设置 `mmr_diversity_bias` 等参数。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/@LlamaIndex">LlamaIndex</a>: LlamaIndex 官方 YouTube 频道 - 适用于 LLM 应用程序的数据框架 </li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">入门教程（本地模型） - LlamaIndex</a>: 暂无描述</li><li><a href="https://github.com/zby/answerbot/blob/main/answerbot/replay_client.py">answerbot/answerbot/replay_client.py at main · zby/answerbot</a>: 使用 LLM、搜索 (RAG) 和其他工具回答问题 - 示例代码 - zby/answerbot</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/retrievers/vectara_auto_retriever#running-over-some-sample-data>).">从 Vectara 索引自动检索 - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/querying/structured_outputs/query_engine#query-engines-pydantic-outputs>).">查询引擎 + Pydantic 输出 - LlamaIndex</a>: 暂无描述</li><li><a href="https://github.com/zby/LLMEasyTools">GitHub - zby/LLMEasyTools: LLM Agent 工具。</a>: LLM Agent 工具。通过在 GitHub 上创建账号来为 zby/LLMEasyTools 的开发做出贡献。</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/vector_stores/TypesenseDemo#query-index>).">Typesense 向量存储 - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/querying/retriever#get-started>).">检索器 - LlamaIndex</a>: 暂无描述</li><li><a href="https://github.com/run-llama/llama_index/pull/13137">Llama Tonic : Transcribe by Josephrp · Pull Request #13137 · run-llama/llama_index</a>: 描述：添加了 Distill Whisper 工具，用于快速精准的转录，无需离开 llama-index。新包？我是否填写了 pyproject.toml 中的 tool.llamahub 部分并提供...</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/putting_it_all_together/agents#agents>))">Agent - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/latest/getting_started/customization#i-want-to-retrieve-more-context-when-i-query>).">常见问题解答 (FAQ) - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/agent_runner/agent_runner_rag_controllable#setup-agent>))">用于 RAG 的可控 Agent - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/metaphor#llama_index.tools.metaphor.MetaphorToolSpec.retrieve_documents>):">Metaphor - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/chat_engines/context#llama_index.core.chat_engine.ContextChatEngine>)">Context - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/agent_runner/?h=low+level">低级 Agent API - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/?h=query+pipeline">围绕 Query Pipeline 构建 Agent - LlamaIndex</a>: 暂无描述</li><li><a href="https://github.com/run-llama/llamabot">GitHub - run-llama/llamabot</a>: 通过在 GitHub 上创建账号来为 run-llama/llamabot 的开发做出贡献。
</li>
</ul>

</div>

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1234543867987230760)** (2 messages): 

- **GPT-1：无名英雄**：一位成员重新审视了**原始 GPT-1** 模型，反思了它对语言模型演进的贡献，并就此主题撰写了一篇[博客文章](https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms)。文章认为该模型“在 6 年的时间里经受住了考验”，暗示像 **Mistral-7B** 这样的一些现代系统是 GPT-1 的大规模扩展衍生品。

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1233374051947384843)** (127 messages🔥🔥): 

- **Flask 服务器困扰**：一位成员在尝试运行本地 Flask 服务器时遇到错误，发现需要设置 `api_key` 以及其他一些问题，包括命名空间冲突和连接错误。他们尝试使用虚拟密钥（`interpreter.llm.api_key = "dummykey"`），并考虑编辑 pydantic 配置以解决命名空间问题。
- **OpenInterpreter 0.2.5 新版本查询**：一位成员询问了 **Open Interpreter 0.2.5 New Computer Update**，随后得到澄清，该版本已脱离 beta 阶段。
- **OI 集成 Groq 的挑战**：几位成员讨论了尝试在 Groq 上运行 Open Interpreter 时的困难，最终得出结论：目前 OI 尚未集成 Groq 支持。提到了一个用于添加 Groq 支持的 GitHub pull request ([#1238](https://github.com/OpenInterpreter/open-interpreter/pull/1238))，目前正等待批准。
- **O1 的硬件查询与全球视野**：成员们交流了关于 Open Interpreter 的远程通信，以及 O1 是否可以使用英语以外的语言进行语音指令。还讨论了在 Rabbit r1 等其他设备上安装 O1 客户端，并利用客户端现有的语音支持。
- **协作与贡献升温**：成员们分享了与 **OpenInterpreter** 相关的各种项目的进展并寻求帮助，例如 **llm-switcher**（一个包含 **AAA+** 和 **MagicLLight** 的开源 AI 工具套件）以及潜在的 **Groq API** 实现。社区进行了代码共享，并持续努力排查故障并改进对不同模型和功能的支持。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/3qG2jGk3?event=1232436050165764096">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与您的朋友和社区保持紧密联系。</li><li><a href="https://tenor.com/view/ya-filthy-animals-gif-22486250">Ya Filthy Animals GIF - Ya Filthy Animals - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://t.co/rbj6Mo5DS0">独家：Jesse Lyu 的崛起与 Rabbit R1 内幕</a>：Rabbit 的创始人兼 CEO Jesse Lyu 讲述了 R1 的起源，他如何与 Teenage Engineering 合作在“10 分钟”内完成设计，以及他对 AI 硬件竞争的看法……</li><li><a href="https://www.tiktok.com/@techfren/video/7362536751044300040">TikTok - Make Your Day</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2105.11490">隐马尔可夫和半马尔可夫模型：这些模型在何时以及为何对时间序列数据中的状态分类有用？</a>：隐马尔可夫模型 (HMMs) 及其扩展已被证明是分类源自具有时间依赖性系统的观测值的强大工具，因为它们考虑了观测……</li><li><a href="https://www.youtube.com/watch?v=YZp3Hy6YFqY">允许 AI Agent 控制计算机（MacOS, Windows, Linux）的重大进展</a>：OS World 赋予了 Agent 完全控制计算机的能力，包括 MacOS, Windows 和 Linux。通过为 Agent 提供一种描述计算机操作的语言……</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1238">由 fire17 添加了 Groq 支持 · Pull Request #1238 · OpenInterpreter/open-interpreter</a>：描述你所做的更改：Groq 的官方 Python API 现在可以很好地融入 oi 流程，没有错误。虽然最终答案是幻觉而非实际输出。似乎能进行规划、编写代码，但……</li><li><a href="https://github.com/plowsai/llm-switcher.git">GitHub - stableagents/llmswitcher：根据您的提示词路由到性能最高且最具成本效益的 LLM [ 🚧 开发中 ]</a>：根据您的提示词路由到性能最高且最具成本效益的 LLM [ 🚧 开发中 ] - stableagents/llmswitcher</li><li><a href="https://colab.research.google.com/github/jaanli/language-model-notebooks/blob/main/notebooks/getting-started.ipynb">Google Colaboratory</a>：未找到描述</li><li><a href="https://github.com/sgl-project/sglang">GitHub - sgl-project/sglang：SGLang 是一种专为大语言模型 (LLMs) 设计的结构化生成语言。它使您与模型的交互更快、更可控。</a>：SGLang 是一种专为大语言模型 (LLMs) 设计的结构化生成语言。它使您与模型的交互更快、更可控。 - sgl-project/sglang</li><li><a href="https://pastebin.com/9iqDMVfS">C:\WINDOWS\system32&gt;pip install pywin32Requirement already satisfied: pywin32 - Pastebin.com</a>：Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1233379615750488225)** (25 条消息🔥): 

- **神秘的定制 3D 项目**：成员们对 OpenInterpreter 01 项目的定制 3D 打印外壳非常感兴趣，引发了关于个人尝试和触感按键乐趣的讨论。一位成员提供了一个展示该项目的 [YouTube 视频](https://x.com/Human_B_ee/status/1783531420394357087)，但指出这不是他自己的作品。
- **01 Heavy 的黎明**：聊天中包含了对新设备 01 Heavy 的期待；目前尚未提供预计发布日期。对比研究表明，它有可能为未来的机器人提供动力。
- **寻找 Amazon 替代方案**：关于使用 Amazon Echo 智能扬声器开发套件作为开放项目构建替代方案的询问不断增加，但尚未有关于兼容性的确认分享。
- **微软能力引发的 Open AI 伦理质疑**：一场讨论凸显了微软创建和修改文件的能力，而 OpenInterpreter 被吹捧为能够满足用户多样化的需求。
- **01 Light 的更新预期已设定**：一位成员提到本周二将进行讨论，以揭晓 01 Light 预计到达时间 (ETA) 的更新时间表。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/urVhd4aq?event=1232436050165764096">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、闲逛，并与你的朋友和社区保持紧密联系。</li><li><a href="https://x.com/hellokillian/status/1783576159672050160?s=46&t=gMHKVMJGcr-j_0RdwnSUMQ">killian (@hellokillian) 的推文</a>: @timshi_ai @Human_B_ee @OpenInterpreter @Grimezsz 为 @grimezsz 定制，由 @fieroty 创建！内部构建非常简单，只需两个亚马逊产品：宏键盘：https://shorturl.at/q...</li><li><a href="https://x.com/Human_B_ee/status/1783531420394357087">Bee 🐝 (@bee_human_) 的推文</a>: 我的新音频工程师是 @openinterpreter 的 01</li><li><a href="https://os-world.github.io">OSWorld: 在真实计算机环境中针对开放式任务的 Multimodal Agents 基准测试</a>: 未找到描述</li><li><a href="https://amzn.eu/d/eJO0LoC">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1233341994756210788)** (100 messages🔥🔥): 

- **伯克利发布 Tool Calling 排行榜**: [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) 评估 LLM 调用函数的能力，提供了一个新颖且定期更新的真实世界基准测试系统。
- **Voice AI 正在兴起**: ElevenLabs 引起了关注，引发了关于其他 Voice AI 初创公司的讨论，如 [Unreal Speech](https://unrealspeech.com/) 和 Hume，这一领域曾由现已倒闭的 Coqui 占据。
- **探索 LLM 的局限性**: [Strangeloopcanon 上的一篇文章](https://www.strangeloopcanon.com/p/what-can-llms-never-do) 思考了 LLM 始终令人惊讶的能力，同时讨论了它们当前的失败模式以及“目标漂移”的概念，作为可能的改进方向。
- **AI 领域的潜在收购动向**: 据报道， Nvidia 收购了以色列 AI 公司 Deci AI 和 Run:ai，这表明其旨在提高其 GPU 和 AI 服务器效率和性能的战略举措。 
- **大上下文模型的探索**: 关于大上下文模型实际应用和未来的对话，是由 Llama 3 扩展到 [1M token context window](https://x.com/markatgradient/status/1785032103429865748?s=46&t=90xQ8sGy63D2OtiaoGJuww) 所激发的。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/markatgradient/status/1785032103429865748?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Mark Huang (@markatgradient) 的推文</a>：1M 上下文长度的 Llama-3 8B 模型。无需多言。已在 HF 上线 @ClementDelangue cc: @winglian @mattshumer_ ↘️ 引用 Gradient (@Gradient_AI_) 我们一直在秘密研发 🔥 很高兴能...</li><li><a href="https://gorilla.cs.berkeley.edu/leaderboard.html">
        Berkeley Function Calling Leaderboard (又名 Berkeley Tool Calling Leaderboard)
    </a>：未找到描述</li><li><a href="https://www.strangeloopcanon.com/p/what-can-llms-never-do">LLM 永远无法做到什么？</a>：关于目标漂移和较低的可靠性。或者，为什么 LLM 无法玩 Conway's Game Of Life（康威生命游戏）？</li><li><a href="https://x.com/karan4d/status/1785000251096437161?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 mephistoooOOHHHHHHSHI- (@karan4d) 的推文</a>：好的，它肯定使用了 GPT-4 的 Tokenizer，所以我敢打赌它也是 4.5。始终使用异常 Token 进行指纹识别。</li><li><a href="https://www.latent.space/p/sim-ai">WebSim, WorldSim, 以及模拟 AI 之夏 —— 对话 Liquid AI 的 Joscha Bach、Nous Research 的 Karan Malhotra、WebSim.ai 的 Rob Haisfield</a>：关于今年生成式 AI 最火爆的前沿领域的三个视角：Simulative AI！</li><li><a href="https://x.com/blader/status/1783934771309253008">来自 Siqi Chen (@blader) 的推文</a>：我认为 @websim_ai 是首批真正的 AI 原生产品之一，并将像 ChatGPT 一样具有影响力。WebSim 不再是聊天框，而是允许你通过 URL 和超链接探索 LLM 的 Latent Space...</li><li><a href="https://unrealspeech.com/">Unreal Speech：面向规模化的 Text-to-Speech API</a>：将 Text-to-Speech 成本降低高达 90%。比 Eleven Labs 和 Play.ht 便宜多达 10 倍。比 Amazon、Microsoft 和 Google 便宜多达 2 倍。</li><li><a href="https://arxiv.org/abs/2402.01469">AMOR：一种通过过程反馈构建可适配模块化知识 Agent 的方案</a>：大语言模型 (LLM) 的显著成功引发了构建语言 Agent 以完成各种复杂任务的热潮。我们提出了 AMOR，一个基于开源 LLM 的 Agent 框架...</li><li><a href="https://github.com/kingjulio8238/memary">GitHub - kingjulio8238/memary：为 Autonomous Agents 提供的长期记忆。</a>：为 Autonomous Agents 提供的长期记忆。通过在 GitHub 上创建账户为 kingjulio8238/memary 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/linux/s/uIN9efGiJk">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://dannywrites.us/nvidia-to-purchase-israeli-deep-studying-co-deci-ai-report/">Nvidia 将收购以色列深度学习公司 Deci AI - 报道 - Dannywrites</a>：据知情人士透露，美国芯片巨头 Nvidia Corp. 已达成协议收购以色列深度学习开发商 Deci AI，“The Data”报道称。</li><li><a href="https://www.mdpi.com/2071-1050/14/7/3811">有效的思维模式干预在增强学生学习方面的特征是什么？系统性文献综述</a>：近年来，通过激发成长型思维来增强个人学习可持续发展的干预措施越来越受到关注。本研究系统地...</li><li><a href="https://journals.lww.com/acsm-msse/fulltext/2015/09000/motivation_and_behavioral_regulation_of_physical.18.aspx">身体活动的动机与行为调节...：运动与训练医学与科学</a>：与理论一致，变量之间的假设关系得到了支持。综合调节和内在动机与中高强度身体活动测量值的相关性最强...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: 新播客！ https://x.com/swyx/status/1784253651844014237
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1233416150063517798)** (12 messages🔥): 

- **一切就绪**：在开始关于 [**Mixture Of Depths**](https://paper-club.ivanleo.com/papers/mixture-of-depths) 的演示之前，聊天确认了可见性。
- **Mixture Of Depths 探讨**：本文介绍了一种新的 Transformer 层，即 *Expert Choice Routing*，旨在实现更快的训练收敛并改进长序列处理。参见原论文 [此处](https://arxiv.org/abs/2404.02258)。
- **跳过困惑**：评论指出，Attention 机制中提到的 Skip Connections（也称为残差连接）是该论文方法论中不可或缺的一部分。
- **规模至关重要**：一份分享的 [摘要](https://arxiv.org/abs/2402.00841) 表明，尽管计算成本较高，但在会议摘要等现实任务中，较大的 Zero-shot LLM 优于经过微调的较小 LLM。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://paper-club.ivanleo.com/papers/mixture-of-depths">Nextra: the next docs builder</a>: Nextra：下一代文档生成器</li><li><a href="https://arxiv.org/abs/2402.00841">Tiny Titans: Can Smaller Large Language Models Punch Above Their Weight in the Real World for Meeting Summarization?</a>: Large Language Models (LLMs) 在未针对特定任务数据集进行显式微调的情况下，已展现出解决广泛任务的惊人能力。然而，在现实中部署 LLM...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1233507749091086390)** (35 messages🔥): 

- **Linux 用户，向 Vesktop 问好**：针对 Discord 视频共享和 Linux 兼容性问题，成员推荐使用 **Vesktop**。它被描述为一个性能更佳的定制 Discord 应用，改进了对 Linux 的支持。感兴趣的用户可以在 [Vesktop GitHub repository](https://github.com/Vencord/Vesktop) 找到更多信息。

- **新兴 SQL 模块备受关注**：一位成员分享了 `sqlite-vss` 的参考资料，这是一个用于创建虚拟表以存储和查询向量（vectors）的 SQL 模块。该成员指出它仍处于早期开发阶段，并指向了 [API reference documentation](https://alexgarcia.xyz/sqlite-vss/api-reference.html)。

- **CLI 工具的聊天机器人引发关注**：有建议提出为流行的命令行界面（CLI）工具创建聊天机器人，引发了关于可行性的讨论。讨论中提到使用 *slono's tool* 可能使创建过程变得简单，该工具增强了 Go 和 SQLite 的便携性。

- **AI 爱好者的资源分享**：成员分享了两个信息丰富的链接；第一个是 [Google Doc](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0)，包含 AI 相关主题、日期、主持人以及文章和会议演讲等丰富资源。第二个是 [Berkeley Gorilla Blog post](https://gorilla.cs.berkeley.edu/blogs/10_gorilla_exec_engine.html)，讨论了 Large Language Models 在现实世界中执行操作的挑战和潜在策略。

- **寻找 AI 黑客松报名详情**：对话中表达了对黑客松报名的关注，一位成员在讨论中重点提到了 [X-ware Arena link](https://arena.x-ware.online)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://gorilla.cs.berkeley.edu/blogs/10_gorilla_exec_engine.html">Gorilla Execution Engine</a>: 无描述</li><li><a href="https://arena.x-ware.online">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://alexgarcia.xyz/sqlite-vss/api-reference.html">API Reference | sqlite-vss</a>: 无描述</li><li><a href="https://github.com/Vencord/Vesktop">GitHub - Vencord/Vesktop: Vesktop is a custom Discord App aiming to give you better performance and improve linux support</a>: Vesktop 是一款定制的 Discord 应用，旨在为你提供更好的性能并改进 Linux 支持 - Vencord/Vesktop</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024 主题, 日期, 主持人, 资源, @dropdown, @ UI/UX patterns for GenAI, 1/26/2024, nuvic, &lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1233337464169431121)** (95 messages🔥🔥): 

- **LAION 陷入困境**：一位成员指出，EU 法律似乎限制了 LAION 访问公共集群获取算力时间，导致其活跃度下降。研究人员正流向那些持续进行实验的更活跃的团体。

- **Terminus Research Group 吸引人才**：一位聊天参与者介绍了他们自己的团体 Terminus Research Group，这是一个非正式集体，目前成员包括 "pixart guy"，暗示其专业背景日益多样化。

- **LAION-Aesthetics 旨在为视觉美感评分**：提到了一篇详细介绍 LAION-Aesthetics 的博客文章，该工具旨在利用机器学习对图像美感进行评分。该模型和相关代码已在 [GitHub](https://github.com/LAION-AI/aesthetic-predictor) 上公开。

- **异常的 Benchmark 结果引发讨论**：成员们讨论了一个 Reddit 上的 Benchmark 测试，该测试显示语言模型中不同量化（quantizations）的性能结果相互矛盾，引发了对测试方法和 LLM 非确定性（non-deterministic）本质的质疑。

- **比较 LLM Token 生成速率**：用户讨论了高性能 GPU 上的 Token 生成速率，注意到不同模型和设置之间存在显著差异。推荐了一些工具和配置，如 exllama 和 TabbyAPI，以获得更好的性能。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://laion.ai/blog/laion-aesthetics/">LAION-Aesthetics | LAION</a>: &lt;p&gt;我们推出了 LAION-Aesthetics，这是来自 LAION 5B 的几个具有高视觉质量的子集集合。&lt;/p&gt; &lt;p&gt;&lt;img src=&quot;https://raw.githubusercontent.com/LAI...</li><li><a href="https://arxiv.org/abs/2401.01808">aMUSEd: An Open MUSE Reproduction</a>: 我们介绍了 aMUSEd，这是一个基于 MUSE 的开源、轻量级掩码图像模型 (MIM)，用于文本到图像生成。aMUSEd 仅使用 MUSE 10% 的参数，专注于快速图像生成...</li><li><a href="https://rentry.org/GPT2">gpt2-chatbot</a>: 背景 https://chat.lmsys.org 允许用户与各种 LLMs 聊天并对其输出进行评分，无需登录。最近可用的模型之一是 gpt2-chatbot，它展示了...</li><li><a href="https://www.cbr.com/japan-light-novel-biggest-publishing-site-ai-developer-scrape/">711,700 Titles From Japan's Biggest Light Novel Publishing Site Get Scraped by AI Developer</a>: 来自日本最大的小说发布网站“成为小说家吧”的 711,700 部作品被一名 AI 开发者抓取，在网上引发了争议。</li><li><a href="https://github.com/borisdayma/dalle-mini">GitHub - borisdayma/dalle-mini: DALL·E Mini - Generate images from a text prompt</a>: DALL·E Mini - 根据文本提示生成图像。通过在 GitHub 上创建账号为 borisdayma/dalle-mini 的开发做出贡献。</li><li><a href="https://github.com/LAION-AI/aesthetic-predictor">GitHub - LAION-AI/aesthetic-predictor: A linear estimator on top of clip to predict the aesthetic quality of pictures</a>: 一个基于 CLIP 的线性估计器，用于预测图片的审美质量 - LAION-AI/aesthetic-predictor</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1cdxjax/i_created_a_new_benchmark_to_specifically_test/&gt;">I created a new benchmark to specifically test for reduction in quality due to quantization and fine-tuning. Interesting results that show full-precision is much better than Q8.</a>: 发布在 r/LocalLLaMA，由 u/jd_3d 发布 • 259 点赞和 103 条评论</li><li><a href="https://old.reddit.com/r/CharacterAI/comments/1cfbmmh/oh_no/">Oh no</a>: 又挂了吗？
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1233454910272573451)** (9 条消息🔥): 

- **探索 VAST：全模态基础模型**：用户对微调 [VAST](https://github.com/txh-mercury/vast)（一个视觉-音频-字幕-文本全模态基础模型和数据集）表现出兴趣，促使成员分享经验并寻求建议。
- **新鲜出炉：新研究论文**：一篇由 Mostafa Elhoushi、Akshat Shrivastava 等人组成的团队撰写的关于 AI 研究的[新论文](https://arxiv.org/abs/2404.16710)引起了成员的关注，推测其基于之前的工作，并强调了其对更快推理和层利用的影响。
- **将图与语言模型结合**：提出了关于将图（Graphs）与大型语言模型（LLMs）结合的问题，寻求相关论文推荐以及使用图对 LLMs 进行条件化的策略。
- **Mistral 模型微调挑战**：一位成员正在微调 **Mistral** 模型用于医疗信息提取，但遇到了模型过度生成序列的问题。讨论涉及了填充（padding）策略以及在 Eleuther 服务器寻求该领域专家意见的适当性。
- **寻找 Eleuther 服务器链接**：在面临模型微调挑战时，一位成员被建议咨询 Eleuther 服务器以获取 LLMs 方面的专家帮助，随后请求了该服务器的 Discord 链接。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.16710">Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding</a>: 我们介绍了 LayerSkip，这是一个加速大型语言模型 (LLMs) 推理的端到端解决方案。首先，在训练期间我们应用层丢弃 (layer dropout)，对浅层使用低丢弃率，对深层使用高丢弃率...</li><li><a href="https://github.com/txh-mercury/vast">GitHub - TXH-mercury/VAST: Code and Model for VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset</a>: VAST 的代码和模型：一个视觉-音频-字幕-文本全模态基础模型和数据集 - TXH-mercury/VAST
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1233350410090053632)** (96 条消息🔥🔥):

- **搜索引擎查询能力讨论**：成员们讨论了将网络搜索工具与 AI 结合使用的最佳实践，提到了 **Tavily** 和 **Brave Search API** 等多种选择。一些人强调了这些工具的成本效益 [Tavily API 信息](https://tavily.com) 和 [Brave Search API](https://brave.com/search/api/)，而另一些人则分享了关于使用限制的特定配置、技术细节以及针对速率限制（rate limits）的潜在解决方法。
  
- **技术问题与部署咨询**：解决了一些技术问题，例如由于 sqlite3 版本问题在本地运行 cohere-toolkit 时遇到错误，在 Azure 部署后难以理解如何与不同组件交互，并分享了用于故障排除和添加自定义工具的 GitHub 资源 [GitHub - cohere-ai/cohere-toolkit](https://github.com/cohere-ai/cohere-toolkit)。

- **Cohere Toolkit 受到热烈欢迎**：一位用户对 Cohere 将其工具包开源表示高度赞赏，强调其对开发者提供了巨大帮助 [GitHub - cohere-ai/cohere-toolkit](https://github.com/cohere-ai/cohere-toolkit)。

- **关于微调（Fine-Tuning）和使用场景的澄清**：针对微调时使用的特定模型、免费试用 API key 的限制和条款，以及 'Generate' 等模型是否会继续可用等问题提出了咨询。

- **将 AI 用于非英语语言及商业用途**：一位成员称赞了 Command-r 在非英语语言方面的表现，并寻求关于部署 command-r API 用于商业用途的澄清；回复建议联系 Cohere 的销售团队或使用 AWS Sagemaker 进行部署。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tavily.com">Tavily</a>：未找到描述</li><li><a href="https://docs.trychroma.com/troubleshooting#sqlite">🔍 Troubleshooting | Chroma</a>：此页面列出了常见的陷阱或问题及其解决方法。</li><li><a href="https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus">C4AI Command R Plus - a Hugging Face Space by CohereForAI</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/multi-step-tool-use#multi-step-tool-use-in-action">Multi-step Tool Use (Agents)</a>：未找到描述</li><li><a href="https://github.com/cohere-ai/cohere-toolkit/blob/main/src/backend/tools/retrieval/tavily.py">cohere-toolkit/src/backend/tools/retrieval/tavily.py at main · cohere-ai/cohere-toolkit</a>：Toolkit 是一系列预构建组件，使用户能够快速构建和部署 RAG 应用。 - cohere-ai/cohere-toolkit</li><li><a href="https://github.com/cohere-ai/cohere-toolkit/?tab=readme-ov-file#how-to-create-your-own-tools-and-retrieval-sources">GitHub - cohere-ai/cohere-toolkit: Toolkit 是一系列预构建组件，使用户能够快速构建和部署 RAG 应用。</a>：Toolkit 是一系列预构建组件，使用户能够快速构建和部署 RAG 应用。 - cohere-ai/cohere-toolkit</li><li><a href="https://github.com/cohere-ai/cohere-toolkit?tab=readme-ov-file#how-to-create-your-own-tools-and-retrieval-sources">GitHub - cohere-ai/cohere-toolkit: Toolkit 是一系列预构建组件，使用户能够快速构建和部署 RAG 应用。</a>：Toolkit 是一系列预构建组件，使用户能够快速构建和部署 RAG 应用。 - cohere-ai/cohere-toolkit</li><li><a href="https://github.com/searxng/searxng">GitHub - searxng/searxng: SearXNG 是一个免费的互联网元搜索引擎，聚合了来自各种搜索服务和数据库的结果。用户既不会被追踪也不会被画像。</a>：SearXNG 是一个免费的互联网元搜索引擎，聚合了来自各种搜索服务和数据库的结果。用户既不会被追踪也不会被画像。 - searxng/searxng</li><li><a href="https://www.yogile.com/thzy59ai246/21t/share/?vsc=e">My First Album</a>：已分享
</li>
</ul>

</div>
  

---


**Cohere ▷ #[collab-opps](https://discord.com/channels/954421988141711382/1218409745380147320/)** (1 条消息): 

westn89: 我们是一家瑞典公司，部分使用了 cohere
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1233395967521456193)** (35 条消息🔥): 

- **探索数学公式构建**：一位成员讨论了使用基础 primitive ops 构建任何数学公式，并应用微分进行梯度/反向传播（backward passes），从而形成依赖图。该方法优化了硬件利用率，并实现了用于流式、快速计算的即时调度（just-in-time scheduling）。

- **OpenELM 咨询**，简短提及：一位成员询问了关于 OpenELM 的使用体验，但随后没有进一步的讨论。

- **框架间的跨兼容性**：
    一位用户分享了他们对 `nn.module` 的使用案例，解释了它对于包含 tinygrad 和 **PyTorch** 组件的混合模型非常有用。该模块可以自动从自身和子对象中收集参数用于训练。

- **澄清语音转文本/文本转语音的咨询**：
    一位用户询问了 George Hotz 展示的语音转文本（Speech-To-Text）和文本转语音（Text-To-Speech）引擎，这些引擎可能存在于 **tinygrad examples** 中，但尚未确定具体是哪一个演示。

- **关于 tinygrad 优化的讨论**：
    用户就 tinygrad 的优化能力展开了辩论，其中一名成员质疑它是否能生成快速的矩阵乘法（matmul）内核（kernel），而另一名成员指出了卷积计算约减算法的使用。George Hotz 澄清了他们对 tinygrad 的愿景，重点在于整体模型训练速度，而非像 matmul 这样的单一操作优化。

**提及的链接**：<a href="https://github.com/tinygrad/tinygrad/tree/master">GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️

---

**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1233398394366726247)** (55 条消息🔥🔥): 

- **探索优化前沿**：一位成员分享了一篇关于 tinygrad 优化器上下文中 [循环展开（loop unrolling）的详细文章](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/upcast.md)。该文章详细介绍了将简单循环转换为优化操作的过程，并提供了对 Uops IR 的见解。
  
- **Tinygrad 0.9 发布预告**：George Hotz 简要提到新更新将随 tinygrad 0.9 版本一同发布，引发了对该库潜在新功能或改进的期待。

- **内核优化剖析**：分享了另一篇详细的 [文章](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/upcast2.md)，阐述了 shapetracker 和 symbolic 库如何与循环展开/upcasting 协同工作；此外，还提供了一份 [指南](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/colors.md) 来解释 tinygrad 中内核输出的颜色含义。

- **Tinygrad 学习者指南**：几位成员提出了理解和贡献 tinygrad 的起点及建议阅读材料；提到的资源包括用于基础概念学习的 [MicroGrad](https://github.com/unknownusername504/MicroGrad) 和 [MiniTorch](https://minitorch.github.io/)，并概述了阅读 tinygrad 代码库的最佳路径。

- **动态测试与符号形状（Symbolic Shapes）**：讨论强调了目前在动态测试和实现无需重新编译即可处理可变形状的内核方面的开发工作，重点是在 mean 和 sum 等操作中使用 symbolic shapes。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tinygrad.github.io/tinygrad/quickstart/">快速入门 - tinygrad 文档</a>：未找到描述</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/upcast.md">tinygrad-notes/upcast.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账号为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/upcast2.md">tinygrad-notes/upcast2.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账号为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/colors.md">tinygrad-notes/colors.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账号为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:tinygrad:symbolic-mean-var-pull">比较 tinygrad:master...davidjanoskyrepo:symbolic-mean-var-pull · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - 比较 tinygrad:master...davidjanoskyrepo:symbolic-mean-var-pull · tinygrad/tinygrad</li><li><a href="https://github.com/unknownusername504/MicroGrad">GitHub - unknownusername504/MicroGrad</a>：通过在 GitHub 上创建账号为 unknownusername504/MicroGrad 的开发做出贡献。</li><li><a href="https://minitorch.github.io/">MiniTorch</a>：未找到描述</li><li><a href="https://github.com/srush/Tensor-Puzzles">GitHub - srush/Tensor-Puzzles: 解决谜题，提升你的 pytorch 水平。</a>：解决谜题，提升你的 pytorch 水平。通过在 GitHub 上创建账号为 srush/Tensor-Puzzles 的开发做出贡献。
</li>
</ul>

</div>

---

**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1234190711344070686)** (10 messages🔥): 

- **考虑 Newsletter 交叉推广的品牌影响**：一位成员思考了与 **[Semafor](https://www.semafor.com/)** 进行无偿推广交换是否会损害品牌形象。尽管担心读者可能会觉得插播广告令人反感，但这仍被视为一个增长机会。

- **更大的受众，更大的增长？**：同一位成员指出 **Semafor 的技术 Newsletter 受众** 规模要大得多，暗示这是一个巨大的增长机会。

- **将内容与已知示例进行比较**：为了说明涉及的内容类型，分享了一个 **[Semafor newsletter](https://www.semafor.com/newsletter/11/03/2023/new-synthetic-data-techniques-shake-up-ai-models)** 的示例，讨论了 AI 中具有争议的合成数据 (synthetic data) 话题。

- **Newsletter 交换——单行道？**：另一位成员发表了看法，质疑 Newsletter 交叉推广的重要性，因为 Newsletter 本质上是一种发送到“虚空”中的“单向媒介”。

- **在推广与读者偏好之间取得平衡**：会议强调，存在疏远那些更喜欢纯净内容而非推广内容的读者的风险，并建议这种策略的成功取决于执行方式和频率。另一位成员认为，即使推广带来的转化率很低，也是有益的，并能带动进一步增长。

**链接提及**: <a href="https://www.semafor.com/newsletter/11/03/2023/new-synthetic-data-techniques-shake-up-ai-models">Semafor Tech: New synthetic data techniques shake up AI models  | Semafor | Semafor</a>：在今天的版本中，我们将探讨机器学习生成的合成数据如何帮助小型 AI 模型获得几乎与大型模型相当的能力。

  

---


**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1233518504012025956)** (10 messages🔥): 

- **微软发布 Phi-3**：微软下一代模型 [Phi-3](https://x.com/lmsysorg/status/1783959458005279091?s=46) 已公开发布，获得了超过 6,000 张选票，并展示了极具前景的能力。相关新闻中，Arena 投票数达到 80 万，Snowflake Arc Instruct 也已加入竞争。

- **Dylan 的前景黯淡**：简短的评论暗示了一位名叫 Dylan 的人士前景不妙，但未说明具体背景或原因。

- **Llama 的微调受到称赞**：针对 "Llama" 的微调 (fine tuning) 过程获得了积极评价，表明其取得了显著成果或改进。

- **对 GPT-4 的期待**：一条消息暗示了 GPT-4 出现的可能性，并表达了提及用户的信心。

- **关于训练 Open LM 的见解**：由 AI2 的 Hanna Hajishirzi 主持的 [YouTube 研讨会](https://youtu.be/qFZbu2P1vZ8) 讨论了训练开放语言模型 (OLMo) 的过程，至少让一位成员希望有更深入的理解，同时也承认了此类共享资源的价值。Hanna 飞快的演讲节奏被提及，进一步巩固了她高效的声誉。
<div class="linksMentioned">

<strong>链接提及</strong>:

<ul>
<li>
<a href="https://x.com/lmsysorg/status/1783959458005279091?s=46">lmsys.org (@lmsysorg) 的推文</a>: 祝贺 @Microsoft 公开发布 Phi-3，这是他们的下一代快速且强大的模型！我们已经为 Phi-3 收集了 6K+ 选票，并发布了新的排行榜。该模型绝对...</li><li><a href="https://youtu.be/qFZbu2P1vZ8">Hanna Hajishirzi (AI2) - OLMo: Findings of Training an Open LM</a>: 来自 Cornell Tech 开源生成式 AI 工作坊的演讲。演讲者：https://homes.cs.washington.edu/~hannaneh/Slides - https://drive.google.com/file/d...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1233471332105322587)** (13 messages🔥):

- **关于 RLHF 的误区澄清**：RLHF 的稳定性和实用性取决于应用场景；像 KTO 这样的方法可能更适合各种任务。*“[RLHF] 取决于应用。KTO 可能最适合许多应用任务”*，这种观点反映出 *“[它] 确实非常微妙”*。
- **DPO 和 KTO 在微调中展现出潜力**：从 SFT -> DPO -> KTO 的转变在微调应用中显示出更好的用户反馈，DPO 和 KTO 的在线迭代版本也“即将推出”。
- **LLaMA 2 的后续动态引发热议**：在 LLaMA 2 发布后，大量信息涌现，一篇 [博客文章](https://www.interconnects.ai/p/llama-2-part-2) 提供了修正和持续分析，讨论了有争议的方面并介绍了 **Ghost Attention** 等技术说明。
- **Ghost Attention - 有用但非关键**：Ghost Attention 最初在维持 LLaMA 2 长对话一致性方面似乎很有前景，但后来的评论表明它可能不再那么重要，这可能是由于数据和长上下文处理能力的改进。*“[GAtt] 并不是一个必须实现的重点。它是学习该领域新课题的一个很好的练习。”*

**提到的链接**：<a href="https://www.interconnects.ai/p/llama-2-part-2">Llama 2 follow-up: too much RLHF, GPU sizing, technical details</a>：社区对 Llama 2 的反应以及我在第一期中没来得及涉及的所有内容。

---

**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1233536007757103175)** (48 条消息🔥): 

- **OpenELM 超越 OLMo**：讨论强调 **OpenELM** 的表现已经超过了 **OLMo**，评论承认 OLMo 1b 的成功有限，不再是一个特别强大的模型，而且现在有比训练 OLMo 时更好的公开数据可用于训练。
- **持续改进驱动 AI 发展**：聊天成员承认，虽然他们的模型尚未达到顶尖水平，但这成为了改进的动力。大家一致认为正在训练更好的模型，并将不足之处作为安全和政策方面的教育工具。
- **开源模型的教育角色**：参与者指出开源模型在促进知情决策方面的重要性，一致认为虽然他们的模型可能不是最好的，但对于 AI 社区的**教育**和透明度至关重要。
- **AI2 在 AI 进步中的作用得到认可**：**AI2** 的努力得到了认可，特别是在教育方面，大家对即将发布的论文和进展表达了热情，并讨论了 AI 研究的财务方面。
- **对替代模型规模与功能的关注**：对话转向了各种话题，包括 **Snowflake**（一种专注于企业、具有高 VRAM 且适用于推理的新模型），以及**活跃参数作为模型能力的代理指标**这一概念，表明了人们对探索规模和基准之外的替代架构的兴趣。

**提到的链接**：<a href="https://x.com/itakgol/status/1783836976590029134?s=46&t=xxWoJxAS_7-BBFC2ro84Zw">Itamar Golan 🤓 (@ItakGol) 的推文</a>：现实生活中的 Visual Prompt Injection 💉🛑

---

**Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1233532129212891136)** (7 条消息): 

- **轻松一笑，内容简短**：一位成员发布了一个简单的“lmao”，表示对频道对话或内容的有趣或好笑。
- **对发布内容的个人反思**：同一个人后来建议需要一个编辑，暗示对他们消息质量或内容的反思。
- **分享丛林探险**：他们分享了一个名为“我要去亚马逊丛林...”的 [YouTube 视频](https://www.youtube.com/watch?v=1WpqQfmzBGY)，详细介绍了一次进入雨林罕见探索区域的远征。
- **对丛林的不同看法**：另一位成员回复了一个视频链接，展示了对丛林本质的不同看法，引用了 Werner Herzog 在纪录片 [Burden of Dreams](https://www.youtube.com/watch?v=dvbxh2rLcdo) 中的观点：*“这里的自然是邪恶而卑劣的……宇宙中没有和谐”*。
- **关于 LLM 特性的 Twitter 梗**：该频道转发了 Marques Brownlee 的一条推文，强调了大型语言模型 (LLM) 幽默的一面，该帖子被认为是“[有史以来最梗的 LLM 破事](https://twitter.com/MKBHD/status/1783962295321919856)”。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/darrenangle/status/1784446600439292223?s=46">来自 darren (@darrenangle) 的推文</a>：PPO DPO KTO CPO IPO ORPO</li><li><a href="https://www.youtube.com/watch?v=1WpqQfmzBGY">我要去亚马逊丛林了...</a>：我现在正和我的朋友 Paul Rosolie 一起深入亚马逊丛林，前往极少数人类曾见过的雨林深处。目的是...</li><li><a href="https://www.youtube.com/watch?v=dvbxh2rLcdo">Werner Herzog 论亚马逊丛林的邪恶</a>：出自《梦的负担》，一部关于 Herzog 拍摄《陆上行舟》的纪录片——两者均于 1982 年发布。00:00 简介 00:28 独白 01:29 雨林...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1234149472657870911)** (1 messages): 

- **关于 AGI 本质的对话**：一位成员称赞了另一位关于 **AGI 的深度帖子**，认同 AGI 的定义是主观的。对话表明，**关于 AGI 本质的辩论仍在继续**。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1233375245109563473)** (51 messages🔥): 

- **关于将 Prompt 集成到代码中的咨询**：一位成员寻求帮助，希望将 *prompt* 集成到现有的聊天模型代码中。另一位社区成员提供了关于在 JavaScript 中结合 **ChatPromptTemplate** 和 **pipe** 方法来链接 prompt 和模型的详细指南。
- **解决 OllamaFunctions 的困难**：讨论了关于 **OllamaFunctions** 无法正常工作的问题，链接到 [GitHub issue #20924](https://github.com/langchain-ai/langchain/issues/20924)。随后，一位成员澄清了 **Gemini** 和 **VertexAI** 模型之间的混淆，告知 **Gemini 1.5 Pro** 仅适用于 **VertexAI**，并通过使用 `ChatVertexAI(model="gemini-1.5-pro-preview-0409")` 的成功实现得到了证实。
- **构建检索增强生成 (RAG) 系统**：一位成员请求推荐用于开发高级 RAG 系统的**开源模型**、*embedding 技术*和*向量存储*解决方案，尽管消息记录中没有对该特定咨询的直接回复。
- **对 LLM 可观测性工具的关注**：关于 LLM 可观测性工具的讨论质疑了在 **Arize Phoenix** 和 **Langfuze** 之间的选择，特别是对于那些主要使用 **LlamaIndex** 的用户。有人表示倾向于自托管的开源解决方案，但未提供直接建议。
- **关于 LLM 的集成与部署查询**：出现了关于部署方法的各种咨询，例如使用 **Hugging Face** 与 **OpenAI API** 的对比，以及出于安全考虑在不通过 **LangChain** 中介的情况下将 OpenAI 与 **SQL Server** 连接。还有关于在新平台上构建网红 AI 克隆的建议请求，以及寻求潜在合作伙伴的私信邀请。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://smith.langchain.com/public/a3846fd5-5007-4a50-bbb3-7265325a4034/r">LangSmith</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/TradingProSquad_/comments/1c9fvax/tradingview_cracked_for_desktop_pc_app_windows/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/">ChatVertexAI | 🦜️🔗 LangChain</a>: 注意：这与 Google PaLM 集成是分开的。Google 已经...</li><li><a href="https://python.langchain.com/docs/integrations/chat/google_generative_ai/">Google AI chat models | 🦜️🔗 LangChain</a>: 访问 Google AI 的 gemini 和 gemini-vision 模型，以及其他...</li><li><a href="https://github.com/langchain-ai/langchain/issues/20924">OllamaFunctions does not work - Received unsupported message type for Ollama · Issue #20924 · langchain-ai/langchain</a>: 检查了其他资源，我为此 Issue 添加了一个非常详细的标题。我使用集成搜索在 LangChain 文档中进行了搜索。我使用 GitHub 搜索来查找类似的问题，并...</li><li><a href="https://github.com/langchain-ai/langchain/pull/20881">[experimental][llms][OllamaFunctions] Add bind_tools and with_structured_output functions to OllamaFunctions by lalanikarim · Pull Request #20881 · langchain-ai/langchain</a>: 为 OllamaFunctions 实现了 bind_tools。使 OllamaFunctions 成为 ChatOllama 的子类。为 OllamaFunctions 实现了 with_structured_output。集成单元测试已更新。Notebook 已...</li><li><a href="https://api.js.langchain.com/classes/langchain_core_tools.Tool.html#invoke>)">Tool | LangChain.js - v0.1.36</a>: 未找到描述</li><li><a href="https://api.js.langchain.com/classes/langchain_core_tools.DynamicTool.html#invoke>)">DynamicTool | LangChain.js - v0.1.36</a>: 未找到描述</li><li><a href="https://api.js.langchain.com/classes/langchain_core_tools.StructuredTool.html#invoke>)">StructuredTool | LangChain.js - v0.1.36</a>: 未找到描述</li><li><a href="https://api.js.langchain.com/classes/langchain_core_tools.DynamicStructuredTool.html#invoke>)">DynamicStructuredTool | LangChain.js - v0.1.36</a>: 未找到描述</li><li><a href="https://api.js.langchain.com/interfaces/langchain_core_tools.ToolInterface.html#invoke>)">ToolInterface | LangChain.js - v0.1.36</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1234549931969216563)** (1 条消息): 

- **AzureSearchVectorStoreRetriever 异步问题**: 一位成员报告了关于 **AzureSearchVectorStoreRetriever** 不支持异步操作的错误。他们询问是否可以调整 lang-serve 以处理同步操作，或者在 retriever 中围绕同步函数编写异步包装器是否是一个可行的解决方案。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1233412390020714557)** (11 条消息🔥): 

- **Galaxy AI 进入竞技场**: GalaxyAI 正在提供对 **GPT-4**、**GPT-3.5-turbo** 等高级 AI 模型的**免费 API** 访问，并具有 *OpenAI 格式兼容性*，以便轻松集成到项目中。在他们的网站 [galaxyapi.onrender.com](https://galaxyapi.onrender.com) 上了解更多信息。

- **发布 Genai-Job-Agents**: 分享了一个基于 Langchain/Langgraph 的 Agent 的 GitHub 仓库，该 Agent 协助职位搜索和简历构建。有关详细信息，请查看仓库 [genai-job-agents](https://github.com/touhi99/genai-job-agents)。

- **探索 GPT-1 的火花**: 一篇新的博客文章深入探讨了原始的 GPT-1 模型，讨论了它的相关性以及到当前模型的架构演进。在此阅读见解 [这里](https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms)。

- **使用实时头像实现 LangChain**: 一个 YouTube 演示展示了 LangChain 在 Airbnb 用例中的应用，包含 150 个问答对和实时头像问答环节。在 [D-ID Airbnb](https://youtu.be/N_GcPLJCQQY) 查看演示。

- **通过无代码平台自动化代码改进**: Autonoma 正在提供一种无代码解决方案，用于自动化代码改进任务（如输入验证和错误处理），并配有用于测试的免费 Playground 和 ALPHA GitHub 集成。在 [Autonoma Free Demo](https://gitgud.autonoma.app?utm_source=discord&utm_medium=chat&utm_campaign=discord-langchain&utm_id=discord-langchain) 体验该平台。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.amazon.com/Mastering-NLP-Foundations-LLMs-Techniques/dp/1804619183/ref=mp_s_a_1_2?crid=3LJY5PNG0V67B&dib=eyJ2IjoiMSJ9.s2npgkPUpgYntBsO6tYJWlP4d-G7Qk6MKD2iEN1SjcA.g9ckC06mGjvstGsU2MlVzG7D9RiXkqrjWGor-uJ2R5E&dib_tag=se&keywords=mastering+nlp+from+foundations+to+llms&qid=1714332505&sprefix=%2Caps%2C220&sr=8-2">未找到标题</a>: 未找到描述</li><li><a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>: 未找到描述</li><li><a href="https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms">重温 GPT-1：点燃 LLM 之火的火星</a>: 全面回顾 GPT-1 对现代 LLM 发展的贡献</li><li><a href="https://gitgud.autonoma.app?utm_source=discord&utm_medium=chat&utm_campaign=discord-langchain&utm_id=discord-langchain>)">GitGud</a>: 未找到描述</li><li><a href="https://youtu.be/6Qa2qdlN2pU">Llama 3 8B: 在安卓手机上实现带实时化身的移动端 RAG（含代码）。让我们完成整个技术栈！</a>: 第一部分：演示。代码在链接中，我们将通过系列视频进行讲解。让我们超越 AI notebook，迈向真实的 c...</li><li><a href="https://github.com/touhi99/genai-job-agents">GitHub - touhi99/genai-job-agents: 一个使用 Langchain/Langgraph 的 LLM Agent，帮助分析简历、通过 API 寻找相关工作并据此撰写求职信</a>: 一个使用 Langchain/Langgraph 的 LLM Agent，帮助分析简历、通过 API 寻找相关工作并据此撰写求职信 - touhi99/genai-job-agents</li><li><a href="https://youtu.be/N_GcPLJCQQY">D-ID Airbnb 使用案例：使用 Ollama 和 Langchain 的 RAG Agent 演示（含 GitHub 代码）</a>: 一个帮助说明商业实时化身助手实际使用案例的演示... 我将制作一个详细的代码审查视频，以便你可以尝试它... ...
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1233672595161874492)** (4 条消息): 

- **探索使用 LLaMA3 的本地 RAG**: 一个名为 "[Local RAG agent with LLaMA3 and Langchain](https://www.youtube.com/watch?v=oDGzMF8CiQU)" 的 YouTube 教程演示了如何使用 **Langchain** 框架结合 LLaMA3 实现 **检索增强生成 (RAG)**。

- **Llama 3 赋能网页浏览**: 另一个名为 "[Llama 3 Web Browsing Agent with Langchain and Groq](https://www.youtube.com/watch?v=au6WQVEgGQo)" 的 YouTube 指南展示了如何结合 **Langchain** 和 **Groq** 技术，通过 **Llama 3** 实现网页浏览功能。

- **交互式 Agent UI 构建教程**: Marc Skov Madsen 提供了一个[视频](https://youtu.be/pODI1SWTVeo?si=v4pGsBjR1joZpdnw)，介绍如何使用 **Panel** 框架为 **CrewAI** 应用程序创建交互式 Web UI，演示了为 AI Agent 构建可视化用户界面的过程。

- **亚马逊书籍链接的验证码阻碍**: 一位成员发布了一个 [Amazon 链接](https://www.amazon.com/Mastering-NLP-Foundations-LLMs-Techniques/dp/1804619183/ref=mp_s_a_1_2?crid=3LJY5PNG0V67B&dib=eyJ2IjoiMSJ9.s2npgkPUpgYntBsO6tYJWlP4d-G7Qk6MKD2iEN1SjcA.g9ckC06mGjvstGsU2MlVzG7D9RiXkqrjWGor-uJ2R5E&dib_tag=se&keywords=mastering+nlp+from+foundations+to+llms&qid=1714332505&sprefix=%2Caps%2C220&sr=8-2)，指向一本名为《Mastering NLP: From Foundations to LLMs》的书籍，但遇到了验证码挑战，导致无法直接访问页面内容。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.amazon.com/Mastering-NLP-Foundations-LLMs-Techniques/dp/1804619183/ref=mp_s_a_1_2?crid=3LJY5PNG0V67B&dib=eyJ2IjoiMSJ9.s2npgkPUpgYntBsO6tYJWlP4d-G7Qk6MKD2iEN1SjcA.g9ckC06mGjvstGsU2MlVzG7D9RiXkqrjWGor-uJ2R5E&dib_tag=se&keywords=mastering+nlp+from+foundations+to+llms&qid=1714332505&sprefix=%2Caps%2C220&sr=8-2">未找到标题</a>: 未找到描述</li><li><a href="https://youtu.be/pODI1SWTVeo?si=v4pGsBjR1joZpdnw">如何通过 Panel 为 CrewAI 应用程序创建交互式 Web UI</a>: 在本视频中，我想为您提供一个使用 Panel 框架构建可视化 CrewAI 应用程序的快速教程，其中包括 fe...</li><li><a href="https://www.youtube.com/watch?v=oDGzMF8CiQU">使用 LLaMA3 和 Langchain 的本地 RAG Agent</a>: 我们将了解如何使用 Llama 3 进行 RAG https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_rag_agent_llama3_local.ipynb#pyth...</li><li><a href="https://www.youtube.com/watch?v=au6WQVEgGQo">使用 Langchain 和 Groq 的 Llama 3 网页浏览 Agent</a>: 我们将了解如何使用 Langchain 和 Groq 配合 Llama 3 实现网页浏览 #python #pythonprogramming #llm #ml #ai #aritificialintelligence #la...
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1233347890546606100)** (54 条消息🔥):

- **运行 Llamafile 时出现分段错误 (Segmentation Fault)**：用户报告在 Modal Labs 等各种平台上尝试运行 `llamafile` 时遇到 *segmentation fault*。有提到特定文件生成错误或无法找到，包括 `Phi-3-mini-128k-instruct.F16.llamafile`。

- **htop Bug 误报内存使用情况**：一位成员提供了关于 [htop Bug](https://github.com/htop-dev/htop/issues/1443) 的信息，该 Bug 导致其无法在 Linux 上正确报告共享内存使用情况，这可能会影响用户在模型运行期间对内存占用情况的感知。

- **Llamafile v0.8.1 发布**：公告称 [llamafile v0.8.1](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.1) 现已发布，增加了对 Phi-3 Mini 4k 的支持，解决了之前的 GPU 模块崩溃问题，并为 Ubuntu 用户添加了捆绑的 NVIDIA + AMD 共享对象（shared objects）。鼓励用户反馈这些更改是否生效或问题是否仍然存在。

- **讨论 LLM 行为和输出异常**：成员们讨论了 LLM 的意外行为，包括输出一致性的变化以及包含括号和换行符的异常响应。这些问题出现在通过 `llamafile` 运行的 Llama3 70B 和 Mistral 等模型的不同迭代版本中。

- **Llamafile 技巧和 GPU 使用问题**：用户分享了确保 `llamafile` 能够充分利用系统 RAM 的技巧，并询问了运行 llamafiles 所支持的 GPU。还有关于确定模型是在 GPU 还是 CPU 上运行的问题，以及寻求处理 `llamafile` 无止尽输出的澄清。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.1">Release llamafile v0.8.1 · Mozilla-Ocho/llamafile</a>：引入了对 Phi-3 Mini 4k 的支持；解决了某些系统上导致 GPU 模块崩溃的 Bug；对 Command-R Plus 的支持现已通过正确的 64 位索引验证...</li><li><a href="https://huggingface.co/jartine/Meta-Llama-3-70B-Instruct-llamafile">jartine/Meta-Llama-3-70B-Instruct-llamafile · Hugging Face</a>：未找到描述</li><li><a href="https://vt.tiktok.com/ZSFctaKnm/">TikTok - Make Your Day</a>：未找到描述</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/issues/144">Error: &quot;The server was not compiled for multimodal or the model projector can&#39;t be loaded.&quot; · Issue #144 · Mozilla-Ocho/llamafile</a>：我注意到了标题中提到的浏览器警报弹窗消息。这可能不是一个错误，但对于初次使用的用户来说有点突兀，所以我提了一下。发生了什么...</li><li><a href="https://github.com/htop-dev/htop/issues/1443">htop doesn&#39;t report shared memory usage on Linux · Issue #1443 · htop-dev/htop</a>：在下面的截图中，你会看到我的一个进程正在使用 139GB 内存，但 htop 报告系统使用了 6GB RAM。这是因为 htop 隐藏了 mmap(MAP_SHARED) 内存。这导致了...</li><li><a href="https://github.com/mozilla-ocho/llamafile/?tab=readme-ov-file#supported-oses">GitHub - Mozilla-Ocho/llamafile: Distribute and run LLMs with a single file.</a>：通过单个文件分发和运行 LLM。通过在 GitHub 上创建账号为 Mozilla-Ocho/llamafile 的开发做出贡献。
</li>
</ul>

</div>
  

---



**AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1233883847066648727)** (11 messages🔥): 

- **告别对崩溃的容忍**：频道成员对迎接即将到来的崩溃表达了一种不屑一顾的情绪，暗示了一种幻灭感。

- **关注 AI 伴侣应用**：频道成员重点介绍了两款 **AI 伴侣应用**，**Faraday** 和 **Amica**，认为它们是那些对 AI 陪伴感兴趣的人的值得关注的工具。

- **Faraday，个人推荐**：[**Faraday**](https://faraday.dev/) 在一名成员使用一个月后获得了个人认可，其特点是能够凭借 **llama.cpp** 在 PC 上本地运行。

- **Amica，注重隐私的新秀**：最近发现的应用 [**Amica**](https://heyamica.com/) 承诺其运行方式与 **Faraday** 类似，具有增强的功能并强调 **数据隐私**，支持自托管和云服务。

- **鼓励关注隐私的 AI 关系**：如果成员在与 AI 的互动中重视 **完全的数据隐私**，则鼓励他们探索 **Faraday** 和 **Amica**。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://faraday.dev/">Faraday.dev</a>：与 AI 角色聊天。离线运行。零配置。</li><li><a href="https://heyamica.com/">Amica - 你的朋友</a>：Amica 是一个开源界面，用于与 3D 角色进行交互式通信，支持语音合成和语音识别。
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[events](https://discord.com/channels/1122748573000409160/1131651713204498583/1233613060334288917)** (2 条消息): 

- **Rosebud AI Game Jam 获胜者公布**：Rosebud 测试人员与 AI 助手 Rosie 组队，在 **Rosebud AI Sleep Game Jam** 中展示了他们在游戏设计方面的创意。一款脱颖而出的游戏 **[Bedtime Negotiation](https://play.rosebud.ai/games/dd6e8a7e-6ca1-4cda-8a5c-f4e422f84ba6)** 具有 AI NPC 角色，Twitch 联合创始人 Kevin Lin 作为客座评委加入。获胜者已在 [Twitter](https://twitter.com/Rosebud_AI/status/1784038539769815543) 上公布。

- **新的 Game Jam：教育与 AI**：Rosebud AI 邀请社区参与新的 **Game Jam**，该活动与 Week of AI 合作，主题为**教育与 AI**。参与者需在 Rosebud 的 AI 平台上利用 Phaser JS 创建一款基于浏览器的 2D 游戏，**奖池为 500 美元**，更多活动详情可在 [Twitter](https://twitter.com/Rosebud_AI/status/1785034624256618617) 查看。
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1233839489340936302)** (9 条消息🔥): 

- **AI Town 的成瘾性获得认可**：一位用户链接到一篇 [Twitter 帖子](https://x.com/ivanfioravanti/status/1784248117388353655)，称赞 AI Town 令人上瘾，并激发了创建一个包含开发者、DevOps、DBA、Infra 和产品经理的模拟的想法。
- **发布由 LLM 驱动的 NPC**：一位用户发布了他们的 LLM 驱动的 NPC 模型和推理栈，以解决常见的 NPC 局限性，代码库和模型托管在 [GitHub](https://github.com/GigaxGames/gigax) 和 [Huggingface's Hub](https://huggingface.co/Gigax)，尽管链接的 API 访问页面未找到。
- **征求关于 NPC 的反馈**：该用户强调了他们针对小型 GPU/CPU 的 **NPC 模型低延迟创新**，并计划推出任务生成模型，邀请成员对最近发布的版本提供反馈。
- **深入探讨 NPC 实现挑战**：该用户揭示了一些 **NPC 开发的关键挑战**，包括压缩模型输出的重要性、尽量减少模型调用，以及解决像 GPT-3.5 或 Mistral 这样的通用 Instruct 模型的问题。
- **社区参与 NPC 微调**：随后展开了关于 NPC 角色开发的对话，并**承诺即将发布一篇博客文章**，以深入探讨项目中遇到的挑战和策略。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/ivanfioravanti/status/1784248117388353655">ifioravanti (@ivanfioravanti) 的推文</a>：这个 AI Town 太让人上瘾了！我停不下来看 AI 角色互相聊天 😂 我应该创建一个把开发者、DevOps、DBA、Infra 和产品经理都放在一起的模拟... 🤯</li><li><a href="https://github.com/GigaxGames/gigax">GitHub - GigaxGames/gigax: 在你的机器上运行 LLM 驱动的 NPC</a>：在你的机器上运行 LLM 驱动的 NPC。通过在 GitHub 上创建账户为 GigaxGames/gigax 的开发做出贡献。</li><li><a href="https://tally.so/r/w7d2Rz)">表单 - Tally</a>：使用 Tally 制作，这是创建表单最简单的方法。
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1233380046820081674)** (11 条消息🔥): 

- **讨论 AI Town 中的地图渲染优化**：*[edgarhnd]* 断言，对于较大的地图，将地图存储为数组可能会有问题，并建议将地图渲染设为静态，并将引擎的基本数据存储在数组中可能是一个实用的解决方案。
- **关于地图处理方法的看法**：*[ianmacartney]* 主张将地图作为静态资源而不是传递的参数，以减少读取时的带宽占用，同时承认服务器端仍需要数组进行碰撞检测。
- **回归原始的地图文件读取方法**：*[edgarhnd]* 和 *[.casado]* 似乎都同意，将地图作为文件读取（原始方法）要简单且高效得多。
- **推广 AI Town 安装教程**：*[.casado]* 分享了一个本地安装 AI Town 的 YouTube 教程链接，标题为 "100% Local &quot;AI Town&quot; with Llama 3 AGENTS!!!"，为有兴趣搭建环境的人提供了资源。视频地址：[100% Local "AI Town" with Llama 3 AGENTS!!!](https://www.youtube.com/watch?v=4HBRh1hMoXQ)。

**提到的链接**：<a href="https://www.youtube.com/watch?v=4HBRh1hMoXQ">100% 本地 "AI Town"，使用 Llama 3 AGENTS!!!</a>：🔗 相关链接 🔗 在此处下载 Pinokio - https://pinokio.computer/ 原始 AI Town - https://github.com/a16z-infra/ai-town Fork 后的 AI town - https://github.com/pea...

---

**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1233675968963411969)** (1 条消息):

- **Mixtral 路由系数之谜**：通过对 **Mixtral-8x7B-Instruct-v0.1** 和 **Mixtral-8x22B-Instruct-v0.1** 的对比，发现它们具有不同的 `router_aux_loss_coef` 值，分别为 0.02 和 0.001。这引发了人们的好奇：这些值是反映了实际的训练数值，还是“虚构值（fantasy values）”，并存在较小的 Expert 可能需要更高 `loss_coef` 的可能性。

---

**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1233424155244892161)** (6 条消息):

- **HPC 上的漫长初始化时间**：一名成员报告称，在 HPC 上收集分片时，**DiscoLM_German_7b_v1** 的初始化时间非常缓慢（*2分20秒*），且 4K Token 输入在 GPU 上的推理时间很长（**超过 12 分钟**）；尽管在没有 GPU 的本地机器上初始化很短（**3 秒**）且推理很快（**1.6 分钟**）。
- **GPU 利用率提升推理速度**：在意识到未将模型加载到 GPU 后，该成员修正了问题，将双 Tesla V100 配置下的推理时间缩短至约 **10 秒**，但分片加载时间仍维持在 **2分20秒**。
- **加载时间故障排除无效**：建议使用的 `low_cpu_mem_usage=True` 参数并未改善模型加载时间，表明尽管进行了调整，问题可能依然存在。
- **慢速存储驱动器可能是瓶颈**：另一位参与者建议，高加载时间可能是由于模型存储在慢速存储驱动器上，并建议核实 HF 缓存目录是否设置在快速数据分区上。

---

**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1233329722457591870)** (8 条消息🔥):

- **讨论实际应用**：用户希望看到更多关于 LLM 的*轶事观察*，并表示有兴趣测试像 **lmsys arena** 这样的模型，承认即使是专门的任务也可能非常有益。分享了一条讨论潜在用途的相关推文：[观察讨论](https://twitter.com/csgmaury/status/1783065038309195919)。
- **GPT-3 德语模型下载量激增**：**gguf 模型**在短短两天内获得了 1500 次下载，表现惊人，显示出社区浓厚的兴趣和参与度。
- **对新模型性能的怀疑**：一位用户对新发布模型的性能表示怀疑，因为社区反馈显示其表现不佳，但另一位用户持不同意见，提到 **Phi-3** 模型在德语 RAG Eval 数据集上并未出现过拟合。
- **询问 Llamafied Phi-3 模型 Tokenizer 的更改**：PhilipMay 询问了在 Llamafied **Phi-3** 模型中修改 Tokenizer 的理由，特别是更改了句子结束 Token (EOS Token)。在与模型所有者的讨论中，显然这种修改是为了在使用 *trtllm* 的聊天应用中获得更好的性能 [Tokenizer 更改讨论 7](https://huggingface.co/vonjack/Phi-3-mini-4k-instruct-LLaMAfied/discussions/7) 和 [Tokenizer 更改讨论 6](https://huggingface.co/vonjack/Phi-3-mini-4k-instruct-LLaMAfied/discussions/6)。
- **为实验创建的 Phi-3 MoE 模型**：使用 Llamafied 版本通过 mergekit 和随机初始化的路由开发了一个新的 **Phi-3 MoE** 模型。目前可供实验使用，但在使用前需要进行训练：[Hugging Face 上的 Phi-3 MoE 模型](https://huggingface.co/PhilipMay/Phi-3-MoE-mini-4k-instruct-raw)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/PhilipMay/Phi-3-MoE-mini-4k-instruct-raw">PhilipMay/Phi-3-MoE-mini-4k-instruct-raw · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/vonjack/Phi-3-mini-4k-instruct-LLaMAfied/discussions/7">vonjack/Phi-3-mini-4k-instruct-LLaMAfied · 为什么修改了 tokenizer_config.json 文件中的 eos_token？</a>：未找到描述</li><li><a href="https://huggingface.co/vonjack/Phi-3-mini-4k-instruct-LLaMAfied/discussions/6">vonjack/Phi-3-mini-4k-instruct-LLaMAfied · 为什么修改了 added_tokens.json 文件？</a>：未找到描述</li>
</ul>

</div>

---

**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1233434451896438927)** (7 条消息):

- **高效语言模型的前沿研究**：一篇名为 *[《低成本语言模型：Python 代码生成的综述与性能评估》](https://arxiv.org/abs/2404.11160)* 的新文章讨论了生成 Python 代码且兼容 CPU 的语言模型。该研究引入了一个包含 60 个编程问题的数据集，并采用思维链（Chain-of-Thought）提示词来提升模型性能。

- **关于 HaystackDB 嵌入（Embeddings）的咨询**：一位成员询问 [HaystackDB 仓库](https://github.com/carsonpo/haystackdb) 是否使用了 2bit 嵌入。他们还进一步询问了该仓库上下文中“二进制量化（binary quantized）”一词的含义。

- **通过二进制量化提升效率**：在解释二进制量化嵌入时，另一位成员说明了二进制量化（Binary Quantization, BQ）有助于为相似性搜索创建更小的索引，从而提高数据库的效率。

- **Llama-3 微调困扰**：一位成员咨询是否有人成功微调了 Llama-3，并指出他们的模型存在无法生成句子结束（EOS）标记的问题。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.11160">Low-Cost Language Models: Survey and Performance Evaluation on Python Code Generation</a>：大型语言模型（LLMs）因其解决各种问题并产生高质量结果的能力，已成为许多自然语言处理（NLP）任务的首选方案。具体而言...</li><li><a href="https://github.com/carsonpo/haystackdb">GitHub - carsonpo/haystackdb</a>：通过在 GitHub 上创建账号来为 carsonpo/haystackdb 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1233419174181867551)** (3 条消息): 

- **为企业级 AI 推出 Snowflake Arctic**：分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=nV6eIjnHEH0)，介绍了 **Snowflake Arctic**，这是一个专注于企业级应用的大型语言模型（LLM），旨在挑战企业级 AI 成本效益的极限。
  
- **通过 Langchain 探索 LLaMA3 的 RAG**：链接了一个教程 [视频](https://www.youtube.com/watch?v=oDGzMF8CiQU)，演示了如何结合 **LLaMA3 和 Langchain** 使用本地检索增强生成（RAG）Agent。

- **使用 Langchain 和 Groq 实现 LLaMA3 的网页浏览**：讨论中包含了一个 [视频](https://www.youtube.com/watch?v=au6WQVEgGQo)，关于使用 Langchain 库和 Groq 硬件实现 LLaMA 3 的网页浏览 Agent，重点关注 AI 与网页浏览能力的整合。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=nV6eIjnHEH0">Snowflake Arctic: The Best LLM for Enterprise AI</a>：今天，Snowflake AI 研究团队激动地推出 Snowflake Arctic，这是一个顶级的企业级 LLM，它推向了成本效益的前沿...</li><li><a href="https://www.youtube.com/watch?v=oDGzMF8CiQU">Local RAG agent with LLaMA3 and Langchain</a>：我们将了解如何使用 Llama3 进行 RAG https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_rag_agent_llama3_local.ipynb#pyth...</li><li><a href="https://www.youtube.com/watch?v=au6WQVEgGQo">Llama 3 Web Browsing Agent with Langchain and Groq</a>：我们将了解如何使用 Langchain 和 Groq 实现 Llama 3 的网页浏览 #python #pythonprogramming #llm #ml #ai #aritificialintelligence #la...
</li>
</ul>

</div>
  

---



**LLM Perf Enthusiasts AI ▷ #[jobs](https://discord.com/channels/1168579740391710851/1169107992587812864/1234606317595791490)** (1 条消息):

<ul>
  <li>
    <strong>加入 Gamma 的 AI 革命</strong>：Gamma 被 a16z 评为顶级消费级 AI 应用，目前正在招聘一名 <strong>AI engineer</strong>，负责大规模文本和图像模型的工作。该职位涉及 prompt engineering、evaluations、fine-tuning 以及使用先进 AI 模型进行功能开发。
  </li>
  <li>
    <strong>突破内容创作的界限</strong>：Gamma 利用 generative AI 简化演示文稿和网站的创建，为超过 <strong>1000 万用户</strong>提供轻松的内容创作体验。
  </li>
  <li>
    <strong>由社区驱动的盈利性创新</strong>：Gamma 拥有来自 <strong>Accel 的超过 1000 万美元融资</strong>并已实现盈利，保持着 <strong>16 人的精干团队</strong>，并继续通过口碑实现有机增长。
  </li>
  <li>
    <strong>成为紧密协作团队的一员</strong>：这家总部位于旧金山的公司正寻求扩大其小而强大的团队，招募对挑战 LLMs 极限充满热情的人才，提供每周约 <strong>3 天</strong>的线下协作。
  </li>
  <li>
    <strong>对构建 AI 的未来感兴趣吗？</strong>：渴望探索这一机会的候选人可以通过以下链接了解更多信息并申请：<a href="https://careers.gamma.app/ai-engineer"><strong>https://careers.gamma.app/ai-engineer</strong></a>。
  </li>
</ul>

**提到的链接**：<a href="https://careers.gamma.app/ai-engineer">AI Engineer</a>：AI Engineer 旧金山 点击此处申请

  

---


**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1234583399029805107)** (3 条消息): 

- **泄露版本推测**：一名成员分享了 [来自 @phill__1 的推文](https://x.com/phill__1/status/1784964135920235000)，评论称 **gpt2-chatbot** 因其广泛的领域知识感觉像是 **gpt4.5**。这引发了讨论，认为它可能是 **GPT-4.5** 的泄露版本。
- **社区认可**：对 **gpt2-chatbot** 的质量表达了简单的认可，描述为“它很棒”。

**提到的链接**：<a href="https://x.com/phill__1/status/1784964135920235000">来自 Phil (@phill__1) 的推文</a>：无论 gpt2-chatbot 是什么，它绝对感觉像 gpt4.5。它拥有我从未见过的疯狂领域知识。

  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1234505496761991198)** (1 条消息): 

- **在代码生成中寻求自定义语法**：一名成员询问是否可以传递自定义语法（custom grammar），可能作为模型特定的选项，通过**防止语法错误**并专注于**语义问题**来增强 code-generation。
  

---