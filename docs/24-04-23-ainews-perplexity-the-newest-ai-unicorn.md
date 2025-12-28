---
companies:
- perplexity-ai
- meta-ai-fair
- hugging-face
- groq
date: '2024-04-23T22:48:23.949413Z'
description: '**Perplexity** 在其 B 轮融资后不久，通过 B-1 轮融资实现了估值翻倍。围绕 **Llama 3** 的重大进展包括：上下文长度扩展至
  **16K tokens**，性能超越 Llama 2 的新型多模态 **LLaVA 模型**，以及像 QDoRA 这样超越 QLoRA 的微调技术改进。**Llama-3-70B**
  模型因其出色的指令遵循能力以及在不同量化格式下的表现而备受赞誉。由 **Meta AI** 发布的多种尺寸的 **Phi-3 模型** 展示了极具竞争力的基准测试结果，其中
  14B 模型的 **MMLU 得分达到 78%**，而 3.8B 模型的性能已接近 **GPT-3.5**。'
id: fcf1837d-66ee-4eaa-a953-9d366d904115
models:
- llama-3-8b
- llama-3-70b
- llama-3
- llava-llama-3-8b-v1_1
- phi-3
- gpt-3.5
original_slug: ainews-perplexity
people:
- daniel-gross
- aravind-srinivas
title: Perplexity，最新的人工智能独角兽。
topics:
- context-length
- fine-tuning
- quantization
- instruction-following
- model-comparison
- multimodality
- benchmarking
- memory-optimization
- model-performance
---

<!-- buttondown-editor-mode: plaintext -->> 2024年4月20日至4月23日的 AI 新闻。我们为您检查了 7 个 Reddit 子版块、[**373** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **27** 个 Discord 社区（包含 **395** 个频道和 **14864** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**1509 分钟**。

就在 [B 轮融资](https://buttondown.email/ainews/archive/ainews-142024-jeff-bezos-backs-perplexitys-520m/) 过去仅 3 个月后，Perplexity 通过 B-1 轮融资再次将其估值翻倍。投资者名单与上次基本一致，均为明星阵容，但罕见的是 Daniel Gross 此次*并未*与 Nat Friedman 共同领投。Dan 似乎与该公司有着特殊的关系——Aravind 分享了 [2022 年 12 月一封关于 Dan 产品反馈的邮件](https://x.com/AravSrinivas/status/1782785662607114365)。

 
![image.png](https://assets.buttondown.email/images/60694bbc-7fdd-4bb0-8a9a-928b03a06a30.png?w=960&fit=max)
 


---

**目录**

[TOC] 


---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**Llama 3 变体与优化**

- **上下文长度扩展**：在 /r/LocalLLaMA 中，Llama-3-8B 的上下文长度已被 [**扩展至 16K Tokens**](https://huggingface.co/mattshumer/Llama-3-8B-16K)，比原始上下文窗口翻了一倍。
- **多模态 LLaVA 模型**：XTuner 团队在 Hugging Face 上发布了 [**基于 Llama 3 的 LLaVA 模型**](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1)，在各项基准测试中显著优于 Llama 2。
- **BOS Token 提醒**：在 /r/LocalLLaMA 中，一则 [**PSA 提醒用户在微调 Llama 3 模型时，确保训练设置中添加了 BOS Token**](https://www.reddit.com/r/LocalLLaMA/comments/1ca4q50/psa_check_that_your_training_setup_is_adding_bos/)，以避免出现 inf grad_norm 或 Loss 过高等问题。
- **特殊 Token 嵌入调整**：针对 Llama-3-8B 中 [**未训练的特殊 Token 嵌入进行了调整**](https://huggingface.co/astronomer/Llama-3-8B-Special-Tokens-Adjusted) 并分享至 Hugging Face，以解决由零值引起的微调问题。
- **网页浏览与交互**：在 /r/LocalLLaMA 中，[推出了用于网页浏览和用户交互的 Llama-3-8B-Web Action 模型](https://www.reddit.com/r/LocalLLaMA/comments/1caw3ad/sharing_llama38bweb_an_action_model_designed_for/)。WebLlama 项目旨在推进基于 Llama 的 Agent 开发。此外，还分享了 [使用 OpenAI TTS 和 Whisper 与 Llama 3 8B 进行语音聊天](https://v.redd.it/xwr67vtxkzvc1) 的演示。
- **微调与扩展**：引入了 QDoRA，用于 [对 Llama 3 模型进行内存高效且准确的微调](https://www.reddit.com/r/LocalLLaMA/comments/1cas7wg/qdora_efficient_finetuning_of_llama_3_with_fsdp/)，并结合 FSDP，性能优于 QLoRA 和 Llama 2。分享了 [用于创建 Llama 3 模型 GGUF 量化的 Hugging Face Space](https://www.reddit.com/r/LocalLLaMA/comments/1ca7xf8/create_llama_3_quants_through_a_hugging_face_space/)。讨论了 [微调 Llama 3 时添加 BOS Token](https://www.reddit.com/r/LocalLLaMA/comments/1ca4q50/psa_check_that_your_training_setup_is_adding_bos/) 的重要性。


**Llama 3 性能与能力**

- **指令遵循**：在 /r/LocalLLaMA 中，Llama-3-70B 因其 [**遵循格式指令并提供简洁回答的能力**](https://www.reddit.com/r/LocalLLaMA/comments/1canrjq/llama370b_is_insanely_good_at_following_format/) 而受到称赞，没有多余的废话。
- **模型对比**：在 /r/LocalLLaMA 中分享了对 HF、GGUF 和 EXL2 格式下 20 个不同量化级别的 Llama 3 Instruct 模型版本的深度对比。主要发现包括 [**EXL2 4.5bpw 以及 GGUF 8-bit 到 4-bit 的表现非常出色**](https://www.reddit.com/r/LocalLLaMA/comments/1cal17l/llm_comparisontest_llama_3_instruct_70b_8b/)，而 1-bit 量化则出现了明显的质量下降。
- **Groq 托管模型的性能**：据 /r/LocalLLaMA 报告，Groq 托管的 Llama-3-70B 在处理侧向思维谜题时不如 HuggingChat 版本。[**温度（Temperature）设置显著影响推理性能**](https://www.reddit.com/r/LocalLLaMA/comments/1casosh/groq_hosted_llama370b_is_not_smart_probably/)，其中 0.4 的设置提供了最佳的一致性。

**Phi-3 和 Llama 3 模型推动开源语言 AI 的边界**

- **Phi-3 模型发布 3.8B、7B 和 14B 尺寸**：在 /r/singularity 中，微软发布了基于 [**深度过滤的网络数据和合成数据**](https://www.reddit.com/r/singularity/comments/1cau7ek/phi3_released_medium_14b_claiming_78_on_mmlu/) 训练的 Phi-3 模型。14B 模型声称在 MMLU 上达到 78%，尽管体积更小，但足以与 Llama 3 8B 竞争。权重即将上线 Hugging Face。

- **Phi-3 3.8B 接近 GPT-3.5 性能**：在 /r/singularity 中，Phi-3 3.8B 模型在 [**基准测试中接近 GPT-3.5 的性能**](https://www.reddit.com/r/singularity/comments/1cau3gy/phi3_a_small_38b_model_nears_gpt35_on_major/)，同时还提供 7B 和 14B 版本。权重随演示视频一同发布，展示了模型效率方面令人惊叹的进展。

- **Llama 3 70B 在 LMSYS 排行榜上与 GPT-4 并列**：在 /r/singularity 中，Llama 3 70B [**在 LMSYS Arena 英语排行榜上获得第二名，与 GPT-4-Turbo 并列第一**](https://www.reddit.com/r/singularity/comments/1cau6yz/llama_3_70b_takes_second_place_in_the_english/)。它可以通过 Groq API 或 Hugging Face 免费使用。关于 Arena 排名有效性的问题也被提出。

- **Phi-3 技术报告显示了令人印象深刻的基准测试结果**：在 /r/singularity 中，发布的 Phi-3 技术报告显示 [**3.8B 模型以 69% 的 MMLU 和 8.38 的 MT-bench 与 Mixtral 8x7B 竞争**](https://www.reddit.com/r/singularity/comments/1catcdv/phi3_technical_report_impressive/)。7B 和 14B 模型显示出进一步的扩展性，MMLU 分别达到 75% 和 78%。

- **Llama 3 的参数翻倍收益递减**：在 /r/singularity 中，一张图表显示 [**在相同数据集上翻倍参数通常会使 MMLU 分数平均提升 17%，但对于 Llama 3 模型仅提升 5%**](https://www.reddit.com/r/LocalLLaMA/comments/1caneis/doubling_the_parameters_on_the_same_dataset/)，这表明 Llama 3 已经高度优化。

**其他**

- **参数扩展**：根据 Reddit 上分享的一张图片，[**在相同数据集上翻倍模型参数通常会使 MMLU 性能平均提升 17%，但对于 Llama 3 模型仅提升 5%**](https://i.redd.it/izvkuwo1s3wc1.png)。
- **高速推理**：据 /r/LocalLLaMA 报道，SambaNova Systems 展示了使用 8 块芯片以 FP16 精度实现 [**Llama 3 8B 每秒 430 个 token 的高速推理**](https://www.reddit.com/r/LocalLLaMA/comments/1caxbx6/sambanova_systems_running_llama_3_8b_at_430_tps/)。
- **量化普及**：/r/LocalLLaMA 中介绍了一个 Hugging Face Space，旨在 [**使 Llama 3 模型的 GGUF 量化创建普及化**](https://www.reddit.com/r/LocalLLaMA/comments/1ca7xf8/create_llama_3_quants_through_a_hugging_face_space/)，从而提高可靠性和可访问性。

# AI Twitter 综述

> 所有综述均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**Perplexity AI 以 10.4 亿美元估值融资 6270 万美元**

- **融资详情**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1782784338238873769) 和 [@perplexity_ai](https://twitter.com/perplexity_ai/status/1782782211399279076) 宣布 Perplexity AI 在 B1 轮融资中以 **10.4 亿美元估值**筹集了 **6270 万美元**。本轮融资由 **Daniel Gross** 领投，投资者还包括 **Stan Druckenmiller、NVIDIA、Jeff Bezos、Tobi Lutke、Garry Tan、Andrej Karpathy、Dylan Field、Elad Gil、Nat Friedman、IVP、NEA、Jakob Uszkoreit、Naval Ravikant、Brad Gerstner 和 Lip-Bu Tan**。
- **增长与合作伙伴**：自 2024 年 1 月以来，Perplexity 的月查询量已增长至 **1.69 亿次**，过去 15 个月的总查询量超过 **10 亿次**。Perplexity 已与 **德国电信（Deutsche Telekom）和软银（Softbank）** 达成合作伙伴关系，向全球约 **1.16 亿用户**进行分发。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1782785026524135848)
- **Perplexity Enterprise Pro 发布**：Perplexity 正在推出 **Perplexity Enterprise Pro**，该版本包含 **SOC2 合规性、SSO、用户管理、企业级数据保留和安全警告**，以解决企业使用的内容数据和安全顾虑。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1782775219733844256), [@perplexity_ai](https://twitter.com/perplexity_ai/status/1782774382399557633)

**Meta 的 Llama-3 模型取得顶尖性能表现**

- **Llama-3 性能**：Meta 的 **Llama-3 70B** 模型已进入 **Arena 排行榜前 5 名**，超越了许多更大规模的模型。8B 变体也超越了许多更大型的模型。[@lmsysorg](https://twitter.com/lmsysorg/status/1782483699449332144)
- **训练细节**：Llama-3 模型在 **超过 15T tokens 的数据**上进行训练，并使用 **SFT、拒绝采样（rejection sampling）、DPO 和 PPO** 进行对齐。[@lmsysorg](https://twitter.com/lmsysorg/status/1782483701710061675)
- **英语性能**：Llama-3 70B 在 **英语类别中表现更为强劲**，与 **GPT-4 Turbo** 并列约 **第 1 名**。在人类偏好测试中，它始终与顶尖模型抗衡。[@lmsysorg](https://twitter.com/lmsysorg/status/1782483701710061675)

**微软发布 Phi-3 语言模型**

- **Phi-3 模型详情**：微软发布了 3 种尺寸的 **Phi-3** 语言模型：**phi-3-mini (3.8B)、phi-3-medium (14B) 和 phi-3 (7B)**。尽管体积较小，Phi-3-mini 仍能 **与 Mixtral 8x7B 和 GPT-3.5 媲美**。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1782594659761389655)
- **训练数据**：Phi-3 模型使用“**经过严格过滤的网络数据和合成数据**”，分别在 **3.3T tokens (mini) 和 4.8T tokens (small/medium)** 上进行训练。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1782598582731702764)
- **基准测试性能**：Phi-3-mini 在 **MMLU 上达到 68.8，在 MT-bench 上达到 8.38**。Phi-3-medium 在 **MMLU 上达到 78%，在 MT-bench 上达到 8.9**，表现优于 GPT-3.5。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1782594659761389655), [@_akhaliq](https://twitter.com/_akhaliq/status/1782598582731702764)
- **可用性**：Phi-3-mini 的 **权重已在 Hugging Face 上以 MIT 许可证发布**。它针对 Hugging Face 的 text generation inference 进行了优化。[@_philschmid](https://twitter.com/_philschmid/status/1782781516172431685)

**谷歌 Gemini 1.5 Pro 取得强劲性能表现**

- **Gemini 1.5 Pro 性能**：谷歌的 **Gemini 1.5 Pro API** 目前在 **排行榜上排名第 2**，超越了 GPT-4-0125，几乎登顶。它在长提示词（longer prompts）上表现更佳，与 **GPT-4 Turbo 并列第 1**。[@lmsysorg](https://twitter.com/lmsysorg/status/1782594507957223720)

**其他值得关注的发布与基准测试**

- **字节跳动的 Hyper-SD**：字节跳动发布了 **Hyper-SD**，这是一个用于图像生成中多概念定制的新型框架，在 1-8 步推理中实现了 SOTA 性能。[@_akhaliq](https://twitter.com/_akhaliq/status/1782601752417575423)
- **摩根大通的 FlowMind**：摩根大通推出了 **FlowMind**，它利用 GPT 自动生成用于机器人流程自动化（RPA）任务的工作流。[@_akhaliq](https://twitter.com/_akhaliq/status/1782604054805332258)
- **OpenAI 的指令层级（Instruction Hierarchy）**：OpenAI 提出了 **指令层级（Instruction Hierarchy）**，使 LLM 优先处理高权限指令，并对提示词注入（prompt injections）和越狱（jailbreaks）具有更强的鲁棒性。[@_akhaliq](https://twitter.com/_akhaliq/status/1782607669376761989)

---

# AI Discord 综述

> 综述之综述的总结

**1. 评估与比较大语言模型 (LLM)**

- 关于新发布的 **[Phi-3](https://arxiv.org/abs/2404.14219)** 和 **[LLaMA 3](https://llama.meta.com/llama3/)** 模型性能和基准测试的讨论，一些人对 **Phi-3** 的评估方法论以及在 MMLU 等基准测试上可能存在的过拟合表示怀疑。

- **Phi-3**、**LLaMA 3**、**GPT-3.5** 以及 **Mixtral** 等模型在各种任务中的对比，其中 **[Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)** (3.8B) 相对于其参数规模表现出了令人印象深刻的性能。

- 围绕 **MMLU**、**BIGBench** 和 **LMSYS** 等基准测试在评估模型真实能力方面的有效性和实用性的争论，有观点认为随着模型的改进，这些测试的可靠性可能会降低。

- 期待 **Phi-3** 在 **MIT license** 下开源发布，以及其承诺的多语言能力。

**2. Retrieval-Augmented Generation (RAG) 的进展**

- LlamaIndex 推出了 **[DREAM](https://twitter.com/llama_index/status/1781725652447879672)**，这是一个用于实验 Distributed RAG 的框架，旨在构建健壮的、生产级的 RAG 系统。

- 讨论创新的 RAG 技术，如用于高效长上下文处理的 **[Superposition Prompting](https://arxiv.org/abs/2404.06910)**、用于提高检索质量的 **[CRAG](https://twitter.com/llama_index/status/1782799757376963006)**，以及 **[结合 function calling 的 RAG](https://blog.pamelafox.org/2024/03/rag-techniques-using-function-calling.html)**。

- 分享关于 **[RAG 演进](https://arxiv.org/abs/2404.10981)**、**[可信度感知生成](https://arxiv.org/abs/2404.06809)** 以及将检索与 LLM 规划集成以实现结构化输出的资源。

- **[@JinaAI_](https://twitter.com/llama_index/status/1782531355970240955)** 发布了开源重排序器 (rerankers)，通过改进向量搜索排序来增强 RAG 性能。

**3. 大语言模型的微调与优化**

- 广泛讨论使用 **Unsloth** 等工具对 **LLaMA 3** 进行微调的策略，解决分词器 (tokenizer) 配置、LoRA 适配器的高效合并以及知识嵌入等问题。

- 全量微调、**QLoRA** 和 **LoRA** 方法的对比，**[QLoRA 研究](https://twitter.com/teortaxesTex/status/1781963108036088060)** 表明其相对于 LoRA 具有潜在的效率提升。

- 为 **llm.c** 实现混合精度训练 (**BF16/FP16**)，相比 FP32 性能提升了 **~1.86 倍**，详见 **[PR #218](https://github.com/karpathy/llm.c/pull/218)**。

- **llm.c** 中的优化，如使用 **thread coarsening** 等技术改进 CUDA kernel (**GELU**, **AdamW**)，以增强受内存限制的 kernel 性能。

**4. 多模态与视觉模型的发展**

- 推出 **[Blink](https://arxiv.org/abs/2404.12390)**，这是一个用于评估 **GPT-4V** 和 **Gemini** 等多模态大语言模型核心视觉感知能力的新基准。

- 发布了如 **[HiDiffusion](https://hidiffusion.github.io/)**（声称只需一行代码即可提高扩散模型分辨率）和 **[PeRFlow](https://github.com/magic-research/piecewise-rectified-flow/blob/main/README.md)**（通过流积分对图像进行上采样）。

- 揭晓了 **[SEED-X](https://arxiv.org/abs/2404.14396)**，这是一个多模态基础模型，通过理解和生成适用于现实应用场景的任意尺寸图像来弥合差距。

- **[Mixture-of-Attention (MoA)](https://snap-research.github.io/mixture-of-attention/)** 架构的进展，用于从语言中生成解耦的、个性化的图像。

**5. 其他**

- **Perplexity AI 的估值和 Enterprise Pro 发布**：据 [Bloomberg](https://www.bloomberg.com/news/articles/2024-04-23/ai-search-startup-perplexity-valued-at-1-billion-in-funding-round) 报道，**Perplexity AI** 在成功完成一轮融资后估值达到 **10 亿美元**。他们推出了 **Enterprise Pro**，这是一项每月 40 美元的方案，具有增强的数据隐私和管理功能，已被 **Stripe、Zoom 和 Databricks** 等公司使用。讨论涉及数据使用担忧和 iOS 应用问题，同时期待 [4 月 23 日的公告](https://x.com/aravsrinivas/status/1781902284844421624?s=46)。

- **Hugging Face 宕机中断模型访问**：许多频道报告了在使用 **Hugging Face** 时出现 **504 Gateway Time-outs** 和服务中断，影响了 **[LM Studio](https://x.com/lmstudioai/status/1782390856986550384?s=46)** 等工具中的模型搜索和下载功能。推测指向 Hugging Face 可能为了管理流量而进行的术语屏蔽 (term-blocking)，目前正在开发长期解决方案以消除这种依赖。

- **Phi-3 和 Llama 3 模型引发热议**：AI 社区积极讨论了**新发布的 [Phi-3](https://arxiv.org/abs/2404.14219) 和 [Llama 3](https://huggingface.co/mattshumer/Llama-3-8B-16K) 模型**。Phi-3 因其在 **MMLU** 等基准测试中的效率和性能而备受关注，尽管存在关于过拟合的质疑。Llama 3 在不同变体和量化（quantizations）方面进行了实验，同时也面临着 tokenizer 和 context size 的挑战。模型的 fine-tuning 潜力以及与各种工具的集成是热门话题。

- **检索增强生成 (RAG) 受到关注**：讨论深入探讨了**评估和增强 RAG 系统**，从使用 **[LlamaIndex](https://twitter.com/llama_index)** 构建金融机器人到引入用于分布式实验的 **[DREAM](https://twitter.com/llama_index/status/1781725652447879672)** 等框架。讨论了诸如 **[superposition prompting](https://arxiv.org/abs/2404.06910)**、可信度感知生成和 **[function-calling RAG](https://blog.pamelafox.org/2024/03/rag-techniques-using-function-calling.html)** 等技术，以及创建从多个文档中综合信息的 RAG 基准测试。

如果您希望我详细阐述摘要的任何部分，或者有其他问题，请告诉我！

---

# PART 1: 高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LLaMA 在 Unsloth 的支持下飞跃**：**LLaMa 3 Instruct Model** 通过 [Hugging Face 上传](https://huggingface.co/unsloth/llama-3-70b-Instruct-bnb-4bit) 实现了进步，承诺提升速度并改进内存占用。同时，成员们分享了在单个 24GB GPU 上使用 Unsloth 以 BF16 格式 fine-tuning 该模型的成功经验，在有限的 VRAM 限制下保持了质量。

- **AI 人体工程学不仅仅关乎代码**：在讨论深度工作的物理层面时，工程师们交流了人体工程学设置技巧，强调了升降桌和像 [Advantage2](https://kinesis-ergo.com/shop/advantage2/) 这样的专业键盘在保持生产力方面的价值。

- **多语言模型备受瞩目**：展示内容包括语言模型的**瑞典语**和**西班牙语**适配，例如 [llama-3-instruct-bellman-8b-swe-preview](https://huggingface.co/neph1/llama-3-instruct-bellman-8b-swe-preview) 和 **solobsd-llama3**。**Ghost 7B Alpha** 模型也亮相了，相关工具和文档可以在[这里](https://ghost-x.vercel.app/docs/models/ghost-7b-alpha)找到。

- **关于 Phi-3 和量化的讨论**：围绕微软的 **Phi-3 Mini 4K Instruct model** 展开了热烈讨论，并对 4-bit 实现进行了定量思考。社区成员在 Hugging Face 上部署的 Phi-3 可以在[这里](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit)获取。

- **微调技巧与框架修复**：对话围绕优化模型 fine-tuning 实践和识别 **tokenizer 问题**展开，社区成员还详细介绍了将知识嵌入 LLM 以供指令使用并与 Unsloth 方法论保持一致的策略。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Perplexity AI 估值突破 10 亿美元**：在成功完成一轮融资后，**Perplexity AI** 的估值达到了惊人的 **10 亿美元**，甚至登上了 [Bloomberg](https://www.bloomberg.com/news/articles/2024-04-23/ai-search-startup-perplexity-valued-at-1-billion-in-funding-round) 的报道，并暗示可能与 AI 专家 Yann LeCun 展开合作。名为 **Perplexity Enterprise Pro** 的企业版拥有增强的数据隐私和管理功能，吸引了大型企业的关注。

**新产品发布引发期待与 App 故障**：Perplexity AI 推出的每月 40 美元的 **Enterprise Pro** 激发了用户对未来功能的期待，尽管一些用户对 iPad 版 iOS App 的技术问题表达了不满。尽管存在这些问题，但用户的热情反映了当前用户群的高度期待。

**数据隐私成为核心话题**：鉴于 Enterprise Pro 的推出，用户讨论了数据隐私问题，促使管理员引用了关于用户同意在模型中使用数据的官方声明。另外，分享频道指导用户在分享 Perplexity AI 的搜索线程时需遵守的合规性要求。

**对 Perplexity 高估值融资的期待升温**：社区热议 Perplexity AI 寻求以 **25 亿至 30 亿美元** 的估值筹集 **2.5 亿美元** 资金，成员们分享了 [TechCrunch 文章](https://techcrunch.com/2024/04/23/perplexity-is-raising-250m-at-2-point-5-3b-valuation-ai-search-sources-say/) 以及 CEO Aravind Srinivas 接受 [CNBC 采访](https://www.cnbc.com/2024/04/23/cnbc-exclusive-cnbc-transcript-perplexity-founder-ceo-aravind-srinivas-speaks-with-cnbcs-andrew-ross-sorkin-on-squawk-box-today.html) 的内容，这标志着公司的快速增长和市场兴趣。

**API 用户寻求前沿功能**：**pplx-api** 频道的一项请求强调了对提供最新网络信息的 API 的渴望，类似于具有浏览能力的 GPT；推荐使用 **Perplexity 的 sonar online models**，这些模型可以在其 [文档](https://docs.perplexity.ai/docs/model-cards) 中找到，同时还有关于增强 Prompt 以提升模型性能的额外建议。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Forge WebUI 吸引新用户**：**Stable Diffusion** 的新手正在探索将 **Forge Webui** 作为入门界面，同时社区也在讨论创建 AI 生成图像和资产（包括游戏和科幻元素）的各种替代方案。
- **CUDA 难题与提速方案**：技术讨论集中在解决 **CUDA errors** 等问题以及提高生成速度的 Prompt，用户对 **ComfyUI** 中缺失的节点以及跨平台模型的兼容性问题表示了沮丧。
- **AI 幻想与梦想生成**：一些奇思妙想的交流提议使用 AI 来设计完美的伴侣或理想的家园，展示了对 AI 在创作高度个性化内容方面潜力的热情。
- **Stable Diffusion v3 热议**：用户在等待 **Stable Diffusion version 3** 发布的过程中，既有兴奋也有怀疑，讨论了来自前 CEO **Emad** 的内部见解，并辩论了该软件真正的开放性。
- **社区交流技术心得**：持续的对话显示出社区热衷于解决实际问题，如跨驱动器的系统安装迁移，他们共同在 Stable Diffusion 及其应用的不断演变中探索。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **前沿的张量并行 (Tensor Parallel)**：工程师们讨论了在 Very Large Language Models (VLLMs) 中实现 **tensor parallel** 的潜力，并期待 **jamba** 的支持能带来性能的飞跃。关注点包括在 **Claude 3** 和 **Big-AGI** 中妥善管理上下文以平衡成本，引用的方法包括 [memGPT](https://memgpt.ai/) 和 [SillyTavern SmartContext](https://docs.sillytavern.app/extras/extensions/smart-context/)。

- **高清 AI 律动**：成员们分享了重制的音乐视频，包括 Beastie Boys 和 deadmau5 & Kaskade，以及一个幽默编码的 CIFAR100 潜变量版本，标题为 [latent-CIFAR100](https://huggingface.co/datasets/Verah/latent-CIFAR100)。在测试了一个 4x4x4 的潜变量数据集后，大家意识到需要更大的图像分类数据集，并分享了[这篇论文](https://arxiv.org/abs/2402.10588)等学术文章，以丰富关于语言模型和符号表示的讨论。

- **工具包的胜利与基准测试的博弈**：DeepMind 的 [Penzai](https://github.com/google-deepmind/penzai) 登场，提供了一个基于 JAX 的神经网络操作工具包。与此同时，关于 LMSYS 基准测试有效性的辩论正在进行，正如一篇[持怀疑态度的 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1c9nvpy/lmsys_becoming_less_useful)所指出的。Rubik.ai 也加入了竞争，为使用 **Claude 3 Opus** 和 **GPT-4 Turbo** 的研究助手招募 Beta 测试人员。

- **模型放大与停机困局**：**Phi-3-mini** 模型与 [LLaMA-3](https://x.com/sebastienbubeck/status/1782627991874678809?s=46) 及 **GPT-3.5** 进行了对比，引发了关于其量化性能的辩论以及对模型权重的期待。Hugging Face 的故障（可能与 **LLaMA-3** 的高使用率或 **[FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)** 有关）成为话题，同时对比了 **QLoRA vs. LoRA** 微调方法的有效性。

- **追求最优 LLM 利用率**：成员们分享了在使用 **Deepspeed Zero 3** 过程中的苦与乐，思考了**单 GPU 优化与 NVLink** 的选择，并筛选了 Llama 微调最佳实践的指南。社区显然更看重具体的微调指南，推荐使用 Hugging Face 的博客和 **Labonne 的 GitHub**，而非通用的 Medium 文章。

- **视觉基准测试揭晓**：关注点转向了 **RealWorldQA**，这是一个专为 **Grok-1.5-vision-preview** 设计的 **xAI** 基准测试数据集，引起了 **Obsidian** 社区的兴趣。正如 [xAI 博客文章](https://x.ai/blog/grok-1.5v)所强调的，该数据集的性质被澄清为基准测试而非训练集，尽管人们仍然渴望获得训练数据集。

- **揭示 RAG 的新发现**：社区通过 LLaMA index 性能、[Superposition Prompting Paper](https://arxiv.org/abs/2404.06910) 中详述的叠加提示方法，以及其他分享的关于增强 RAG 可信度的论文，深入探讨了 **Retrieval-Augmented Generation (RAG)**。函数调用 (Function-calling) 的 RAG 实现也受到了关注，包括 Pamela Fox 的[博客](https://blog.pamelafox.org/2024/03/rag-techniques-using-function-calling.html)等资源。

- **模拟超乎想象的世界**：虽然 **WorldSim** 处于离线状态，但 **Super WorldSim** 和 **Snow World Simulator** 等替代模拟器在 **HuggingChat** 中找到了归宿。Discord 上的协作世界构建工作正在蓬勃发展，重点关注 **Llama 3** 即将发布的开源模型，以丰富模拟体验。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GPU 故障与小问题**：关于 **LM Studio** 在 AMD 和 Nvidia GPU 上性能的讨论揭示了 **GPU offloading** 对于避免 100% CPU 占用和防止系统低效至关重要。“Error loading model”问题的解决方案集中在关闭 GPU offloading 或设置特定的环境变量，以引导 **LM Studio** 使用独立 GPU。

- **Hugging Face 的小插曲**：由于 Hugging Face API 宕机，用户遇到了 **503 和 500 错误消息**，影响了 LM Studio 搜索和下载模型的能力。虽然社区推测 Hugging Face 可能会通过屏蔽某些术语来缓解流量压力，但通过 [LM Studio Tweets](https://x.com/lmstudioai/status/1782390856986550384?s=46) 的持续沟通让大家了解最新动态。

- **模型热潮**：各种 **AI 模型** 引发了辩论，讨论涉及 **Meta-Llama-3-8B-Instruct-GGUF** 的无限生成问题、微调 **Llama 3** 与 **Goliath 120B** 及 **Mistral** 的对比，以及 **Phi-3** 惊人的效率。关于将 **Autogen** 等工具与 LM Studio 集成的查询，以及对内容生成中模型限制的担忧，凸显了用户对定制化的渴望。

- **Prompt 难题与配置奇闻**：LM Studio 用户分享了为 **D&D 场景** 编写 system prompts 的技巧，解决了 **Llama-3-Smaug-8B** 的 prompt 问题，并推荐了预设配置。同时，一个涉及 2-token 限制问题的 **Autogen** 故障引发了社区的排错建议。

- **技术试验与 ROCm 评论**：使用 ROCm 的 AMD GPU 引发了对 **Meta-Llama-3** 性能的评论，记录了运行速度，并提出了关于在低端硬件上运行大型模型的问题。解决 LM Studio 中 AMD GPU 选择策略的智慧层出不穷，并且分享了 [Hugging Face 仓库详情](https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF) 以便有效地利用 **Meta Llama 3 模型**。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **X11 助力远程 GPU Profiling**：CUDA 公会探讨了通过 **X11 forwarding** 经由 [SSH 操作 Nsight Compute GUI](https://goteleport.com/blog/x11-forwarding/)，一位用户分享了[远程设置 Nsight Compute 的教程](https://tspeterkim.github.io/posts/nsight-setup-on-ec2)。同时，**'Effort' 算法**为 LLM 推理计算增加了动态性，并引起了在 Triton 或 CUDA 中使用的兴趣，其[代码已在 GitHub 上发布](https://github.com/kolinko/effort)。

- **CUDA 矩阵魔法与线程同步讨论**：在 CUDA 频道中，用户澄清了 **CUDA 矩阵乘法**和 CUDA 中 `__syncthreads()` 行为等概念；特别强调了从 Volta 架构开始的变化。通过围绕 `__forceinline` 和 `__inline` 的讨论，内联函数（Inline functions）的神秘面纱被揭开。

- **Triton 应对变换与内存管理**：Triton 用户面临图像灰度化和内存碎片的挑战，而其他人则因目前的限制讨论了二分查找（binary search）的实现策略。`make_block_ptr` 参数的 **order** 引起了困惑，将对话引导向行优先（row-major）与列优先（column-major）格式的对比。

- **PyTorch 实践**：在 Torch 频道中，公会确认了如 `torch.nn.conv2d`、`torch.nn.relu` 和 `torch.nn.batchnorm` 等操作是在 **GPU 上执行的，中间结果无需 CPU-GPU 传输**。GPU 操作调度被指出是异步的。

- **使用 CUTLASS 进行优化**：关于 CUTLASS 的 **Lecture 15** 预告让热衷学习的人们充满动力，承诺将深入探讨 CUDA 的前沿工具和技术。

- **算法、入门、读书会及其他**：零散的讨论涉及了一个 CUDA 算法示例、以**风趣风格**掌握 CUDA 的入门之旅、PMPP 书籍章节练习、潜在的 YouTube 录像上传，以及在实现 denseformer 时提到的 JAX 内存问题。hqq 频道讨论了重要的 **Triton kernel benchmarks**，并推动高效的量化（quantization）策略。

- **引擎室中的 Kernel、Coarsening 与协作**：llmdotc 频道正热烈讨论 **atomic 操作移除**、BF16/FP16 混合精度收益、对当前 CUDA 版本的要求，以及通过整合见解使 GELU 和 AdamW kernel 性能翻倍。线程粗化（Thread coarsening）被视为优化受内存限制（memory-throttled）的 kernel 的希望之光。

- **管理、技术设置与 FlashAttention**：管理员们各司其职管理内容，而 massively-parallel-crew 频道则忙于完善**活动录制**和未来演讲的准备工作，包括对 FlashAttention 深度探讨的预告。

- **本地 GPU 爱好者聚会**：在一个轻松的时刻，off-topic 频道透露了居住在 Münster 附近的成员们进行了一次愉快的聚会，该地被誉为 CUDA 爱好者的中心。

- **Ring Attention 引起关注**：ring-attention 频道通过简短提到的手动放置（manual placement）成功案例以及通过 [Axolotl GitHub 链接](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching)分享的 tinyllama 测试引起了好奇。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**智能手机上的本地 LLM 前景：** 讨论探讨了在智能手机上运行大型语言模型（LLM）的可行性，考虑到内存带宽（高达 51.2 GB/s）和 GPU 能力（Exynos 2400 芯片组规格），认为即使是 7-8B 模型也可能可行。社区成员研究了现有的应用，如 [MLC-LLM](https://github.com/mlc-ai/mlc-llm)，并讨论了 Hugging Face 的停机如何引发关于免费 AI 模型托管可持续性的疑问。

**SpaceByte 让 Tokenization 变得过时：** 一种新的字节级 LLM 架构 [SpaceByte](https://arxiv.org/abs/2404.14408v1) 有望消除对 Tokenization 的需求，解决 Tokenizers 可能导致的信息泄漏问题。其他讨论批评了 Fineweb 与 LLaMA 的关系，以及 ProGen2 在 AI 设计 CRISPR-Cas 蛋白质中的新应用，展示了 LLM 在加速科学发现方面的作用。

**谨慎扩展与委婉辩论：** 一场关于出版物中数据舍入问题的冲突引发了关于技术辩论中建设性批评和语气的广泛讨论。这场小规模争论阐明了关于将舍入数据归因于 Chinchilla 论文还是复现团队的误解，揭示了复现方法论中更深层次的问题。

**RWKV 集成加速：** GPT-NeoX 开发人员正忙于实现 RWKV（Rethinking Weighted Key-Value Memory Networks），并支持 fp16 和 JIT 内核编译。进展和任务详见 [GitHub Issue #1167](https://github.com/EleutherAI/gpt-neox/issues/1167)，开发人员正在推动版本编号系统以简化迭代过程。

**AI 设计高性能蛋白质：** Profluent Bio 成功利用 LLM ProGen2 设计了新的 CRISPR-Cas 蛋白质序列，产生了特异性更高的变体。这一成就证明了 LLM 在生物技术领域不断扩大的实用性。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**与 PDF 聊天，现在支持数学！**：[ai_pdf](https://github.com/Crizomb/ai_pdf) 是一个开源项目，支持与 PDF 文档进行对话，通过将数学 PDF 转换为 LaTeX，在处理此类文档时表现出色。

**语音引导的 AI 艺术创作**：一段由语音命令实时生成的 2.5 分钟视频已在 [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1c8oea6/endlessdreams_voice_directed_realtime_videos_at/) 上分享，指向了 AI 驱动的动态视频创作的未来。

**AI 变得触手可及**：[Transformers.js](https://xenova.github.io/transformers.js/) 允许直接在浏览器中运行 HuggingFace Transformers，扩展了 AI 应用在 Web 环境中的活动空间。

**Rust 助力精简 BPE**：`minbpe-rs` 是 `minbpe` 的 Rust 移植版本，具有 Tokenization 和训练功能，提高了 NLP 任务的性能。该项目可在 [GitHub](https://github.com/gnp/minbpe-rs) 上获取。

**Diffusion 困境与 AI 视频辩论**：用户讨论了使用 Diffusion 创建关于“AI 马”的 1 分钟视频的可行性，其他人则解决了各种实现挑战，展示了新兴 AI 应用在成长初期的阵痛。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**代码指令提升 Hermes**：在集成 [代码指令示例](https://link.to.examples) 后，观察到 **Hermes 2.5** 在各种基准测试中表现优于 **Hermes 2**，在 MMLU 基准测试分数等指标上有显著提升。

**Mistral 的容量挑战**：讨论得出结论，如果没有持续的预训练，**Mistral** 无法扩展到 8k 以上。重点转向模型合并策略的增强，例如将 **UltraChat** 和基础 **Mistral** 之间的差异应用于 **Mistral-Yarn**。

**AI 中的共情**：**Open Empathic** 项目寻求在扩展类别方面的帮助；贡献者可以参考 [YouTube 教程](https://youtu.be/GZqYr8_Q7DE) 进行指导，并鼓励利用 YouTube 上的电影场景来增加共情反应训练的多样性。

**Mojo 辨析差异**：在 **Mojo** 中对参数（parameters）和实参（arguments）进行了澄清，后者是运行时值，而语言中的参数保持为编译时常量。正在探索诸如“Type State”之类的复杂模式，与 Python 的性能比较揭示了持续存在的效率问题，特别是在 IO 操作方面。

**深入 Mojo SIMD 与多线程实战**：在 CPU 受限的场景下，在 **Mojo** 中实现 SIMD 模式产生了接近 Rust 的性能。然而，仍然存在优化挑战，例如 `parallelize` 的最佳实践。在其他讨论中，`UnsafePointer` 的使用和 `LegacyPointer` 的逐步淘汰表明了该语言内存处理方式的成熟。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **BOS Token 漏洞已修复**：工程师们研究了 LLaMa 3 在 fine-tuning 过程中未能正确添加 BOS tokens 的问题；通过一个修改 `tokenizer.json` 的 Pull Request 找到了[解决方案](https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/41)。

- **Phi-3 模型表现超出预期**：尽管参数量较小（约 3.8b），Phi-3 模型的表现却能与更大的模型相媲美，显示出极高的效率。它们采用开放的 MIT 许可证，但可能更侧重于推理能力而非广泛的知识储备。

- **AI 训练的 GPU 需求备受关注**：讨论聚焦于 AI 模型训练所需的巨大资源，提到一个特定的配置：512 块 Nvidia H100-80G GPU 运行一周，凸显了此类任务的计算强度。

- **LLaMa 的扩展能力不容小觑**：一位成员展示了 [Llama 3](https://huggingface.co/mattshumer/Llama-3-8B-16K)，该模型拥有 16K 的 token 长度，其增强的长序列处理能力引发了关注。

- **AI 开发中的障碍与权宜之计**：对话涉及了 Discord 链接共享问题、有问题的 8-bit optimizer 配置，以及耗时 1.5 小时的 model merging 过程；还有关于在 Axolotl 中使用 Unsloth 进行优化训练的指导工作。

- **数据集精通与 Markdown 之谜**：参与者分享了在 YAML 中指定 `"type: sharegpt"` 如何影响数据集操作，并寻求 Axolotl 提供的不同数据集格式的[文档](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)。此外，还表达了对 GitHub 渲染 **qmd** 文件而非传统 Markdown 的担忧。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **优化进行中**：由于高流量导致的 **Wizard 8x22b** 性能问题正通过优化 load balancer 得到缓解，这将降低延迟。

- **迈向高效路由**：在删除 **Databricks: DBRX 132B Instruct (nitro)** 后，流量将重新路由到主 [Databricks DBRX 132B Instruct model](https://openrouter.ai/models/databricks/dbrx-instruct)。**OpenRouter** 宣布推出三个新模型，包括 **LLama 3 finetune**，并更新了 prompt 格式，解决了侧重于动态路由增强的区域网络故障。

- **缓解模型故障**：用户反映了 **WizardLM-2** 偶尔出现的性能问题，*SillyTavern's Assistant Prefill* 复杂化了与 **LLaMA 3** 模型的交互。针对 Hugging Face 的 tokenizer 服务停机问题已发布热修复补丁，长期解决方案正在制定中。

- **AI 模型提供的财务可行性**：关于提供 AI 服务的财务状况存在激烈辩论，特别是费率的可负担性以及与图像生成模型相比的成本差异。讨论涵盖了 **FP8 quantization**、活跃工作者折扣以及 **Groq** 硬件的经济足迹。

- **增强合同交互**：**#[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1231369890829439056)** 频道中的建议包括敦促用户增强合同标准意识、实施法律相关性的本地化，以及加入非法条款检测功能。此外还介绍了 [Keywords AI](https://keywordsai.co) 和 [DeepGaze](https://www.deepgaze.ca/)，两者均利用了 **OpenRouter**。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **机器人诡异感 (Robo Creep Factor)**：工程师们围绕 **Atlas 机器人** 的发布展开了辩论，期待其市场能力和底层策略，同时也在讨论其在社交媒体上引发关注的、令人不安的“诡异感”。

- **AI 神性讨论**：关于 AI 灵性的可能性及其影响展开了激烈的讨论，包括对 AI 意识的反思，并受到社区关于世俗话语规则的约束。

- **API 构建与界面升级**：围绕 **MyGPT** 以及 **MetaGPT** 和 **Devika** 等其他工具的对话深入探讨了它们在构建 API 和改进应用开发方面的潜力，并对自动化的 GitHub 交互表现出兴趣。

- **模型性能表现参差不齐**：**LLaMa 3** 在工程师中引起了褒贬不一的反应，同时对传闻中的 **GPT-5** 发布日期持怀疑态度。此外，有人呼吁提供高质量的生成式 AI 文献，引用了 **OpenAI 发表的论文** 以及 Arxiv 等仓库。

- **Prompt Engineering 细致讨论**：工程师们交流了 Prompt 优化的策略，辩论了简短自定义指令的优劣，并讨论了分享技术的伦理层面。对话还涵盖了通过 GPT-4 改进电子邮件以及缺乏全面 Prompt 库的问题。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **多模态模型对过拟合的担忧**：现有的 **多模态数据集**（总计约 200 万对）存在导致 GPT-4v 等模型过拟合的风险，特别是在 LAION-COCO 标题方面，模型显示出一种令人担忧的记忆而非学习的趋势。
  
- **图像处理与监控方面的创新与担忧**：**Adobe Firefly Image 3** 的发布因其改进的图像生成和与 Photoshop 的集成而引起关注。与此同时，针对 Discord 上 AI 驱动的监控机器人的担忧，通过引入使用 API 检测此类机器人的 [kickthespy.pet](https://kickthespy.pet/#823813159592001537) 得到了回应。 

- **视觉感知与上采样的新浪潮**：针对 GPT-4V 和 Gemini 等 **多模态 LLM** 的基准测试 **Blink** 已经发布，通过需要视觉感知能力的任务向模型发起挑战。在图像处理方面，**Piecewise-Rectified Flow (PeRFlow)** 和 **HiDiffusion** 都在取得进展；然而，HiDiffusion 在高分辨率图像中的伪影问题仍然是一个关注点（[阅读更多关于 Blink 的信息](https://arxiv.org/abs/2404.12390)）。

- **突破多模态极限**：围绕多模态模型的讨论仍在继续，一种新的架构 **Mixture-of-Attention (MoA)** 被引入，承诺在个性化图像生成中增强解耦性（[在此论文中描述](https://snap-research.github.io/mixture-of-attention/)）。**SEED-X** 多模态基础模型也因其处理可变尺寸图像的能力而引起轰动，重点在于全面的理解和生成。

- **代码协作呼吁**：在公会中，一项针对 **JavaScript/Rust** 框架构建 NLP 编码助手的公开协作呼吁引起了关注，尽管 *softmax_function* 在多个项目中的日程安排很紧，但偶尔也会提供支持。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**通过分布式 RAG 实现 DREAM 大计**：LlamaIndex 推出了 **DREAM**，这是一个分布式 RAG 实验框架，同时还发布了多种 RAG 增强功能，如 **ColBERT with a Twist** 和 **LoRA Fine-Tuning**。深入探讨关于 **CRAG**（一种改进 RAG 检索的创新层）以及 [LlamaIndex 推文](https://twitter.com/llama_index)中开源 rerankers 的讨论。

**在 OpenAI 之外使用 AI 模型**：在 **#[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1231156600190795826)** 频道中，用户在解决集成 Bug 和 API key 困扰的同时，还在处理 LLM 的不同检索方法。正如众多 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/)中所详述的，人们关注改进上下文管理的技术，并对使用 OpenAI 以外的替代方案表现出浓厚兴趣。

**从 LinkedIn 到 Google Sheets，AI 融资数据引起关注**：一位成员在 LinkedIn 上分享了 **Infini Attention** 的解析，而按城市划分的 AI 融资分布情况可以在 [Google Sheets](https://docs.google.com/spreadsheets/d/1nWBP1MpT7sACYDxqdCo8gBR7b2nXJbrF9Z43y69q9hg/edit#gid=752020121) 上查阅。新的 **LLM-Ready Markdown** 集成令社区感到兴奋，WhyHow.AI 增强的 Knowledge Graph SDK 正在 [Medium](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3) 上招募 Beta 测试人员。

**数据库辩论与微调**：**#[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1231221804518080615)** 频道的成员们正在积极辩论最适合 LLM 训练的数据库类型。他们强调了在训练大语言模型时，理解数据库 Schema 和 Vector Store 可能性的重要性。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**遭遇兼容性问题**：成员们注意到，尽管 **Open Interpreter** 已有成功实现，但在 Windows 上遇到了挑战，并且在模型支持方面存在混淆，特别澄清了 **OI** 目前在云端选项中仅支持 **OpenAI**，不支持 **Groq** 或 **Llama 3 70b** 模型。他们还讨论了 **Llama 3 70b** 与其 8b 版本相比的稳定性问题。

**解释器，你说什么？**：**Open Interpreter** 的各种功能和集成挑战被重点提及，例如 Windows 系统上的安装问题和 pytesseract 错误，后者可以通过使用 `pip install --upgrade litellm` 来缓解。详细的故障排除视频（例如 YouTube 上关于将 OI 与 **GROQ API** 集成的视频）展示了社区对高性价比解决方案的热切期待。

**屏幕视觉，但非预言**：在 AI 视觉领域，已明确 **Open Interpreter** 利用 **GPT-4-vision-preview** 进行截图识别任务，这表明该工具兼具文本和视觉能力。

**援手与配置**：社区庆祝 **Open Interpreter** 达到 **100 位 GitHub 贡献者**，并展现了强大的协作精神。正如 [Pull Request](https://github.com/OpenInterpreter/open-interpreter/pull/1204) 中所示，目前正在推动共享默认配置文件，以改进与各种模型的交互。

**M1 Mac 空格键问题**：特别是针对 M1 Mac 用户，在排除按空格键无法按预期工作的录音问题时，提出了多种解决方案，包括安装 **ffmpeg**、检查麦克风权限或使用 **conda** 切换 Python 版本。

**云端兼容性的可能性**：成员们希望看到 **OI** 与云服务对齐，呼吁实现更广泛的云平台支持兼容性，包括但不限于 **brev.dev** 和 **Scaleway** 等平台。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**标题党 vs. 实质内容**：社区内关于 AGI 文章标题的争论反映了人们对既能吸引眼球又保持真实性的标题的追求。观点分歧很大，从 AGI 的本体论地位到将其视为一种信仰，这表明人们正在寻求既能引发思考又诚实的对话，正如“AGI Isn't Real”等标题以及 Mistral CEO Arthur Mensch 在 [Business Insider](https://www.businessinsider.com/mistrals-ceo-said-obsession-with-agi-about-creating-god-2024-4) 中的采访所展示的那样。

**显微镜下的 Phi-3**：由于感知到在 **MMLU** 等基准测试上存在过拟合，人们对 **Phi-3** 基准测试的完整性持怀疑态度，并质疑其对 OOD 性能的相关性。批评还延伸到了模型的评估展示和未公开的数据流水线，尽管人们对 Phi-3 预期的 MIT 许可证发布和多语言能力感到兴奋。

**基准测试评估**：AI 模型评估的效用受到审视，指出在 MMLU、BIGBench 等自动化基准测试工具与 ChatBotArena 等人力密集型评估之间存在权衡。基于 Perplexity（困惑度）的评估（如 AI2 的 Paloma）被证实更多是用于内部训练检查点，而非公开竞赛。

**Discord 社区动态**：关于社区的轶事包括一位研究员转瞬即逝的推文习惯、尽管免费订阅但成员数量却意外地低，以及在度过充满 NDA 的时期后，渴望与 Ross Taylor 等行业人物交流的坦诚愿望。

**指令微调与 CRINGE 的纠缠**：指令微调（Instruction tuning）的生态系统得到了详细阐述，引用了一篇[入门博客](https://gaotianyu.xyz/blog/2023/11/30/instruction-tuning/)，并对 MT Bench 论文中的分类表示赞赏。此外，CRINGE 论文中利用负样本的新颖训练方法引起了关注，并针对指令微调进行了进一步讨论。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **项目亮点**：宣布了一个[开源配对应用程序](https://x.com/anmol_desai2005/status/1781679469679325605?s=46&t=vUJbpAOoGDUfvrA5TGBjTQ)，集成了 **@cohere Command R+**、**@stanfordnlp DSPy**、**@weaviate_io Vector store** 和 **@crewAIInc agents**。其 GitHub 链接已分享以获取社区反馈。
  
- **AI 增强的求职策略**：工程师们讨论认为，在获得面试机会方面，**个人项目**和**简历上的大厂名称**往往比实际工作经验更重要。

- **通过上下文优化 AI**：工程师们探讨了如何使用 **preambles** 和 **BOS/EOS tokens** 将 AI 的回答限制在特定主题内，以确保输出保持在预期的训练范围内。

- **网页抓取的难题**：关于开发一个利用 **gpt-4-turbo** 识别（选择器、列）对的**通用网页抓取工具**展开了辩论，模型与网页元素的交互复杂性被证明极具挑战性。

- **Cohere 爱好者寻求扩展**：工程社区对将具有 **URL Grounding (RAG)** 功能的 **Cohere Command-r** 集成到 **BotPress** 中表现出浓厚兴趣，暗示如果成功实现，用户可能会从 ChatGPT 转向 Cohere。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LLM Scraper 的网页魔法**：GitHub 上新发布的 [LLM Scraper](https://github.com/mishushakov/llm-scraper/) 提供了一种将任何网页转换为结构化数据的方法，它利用 LLM 的解析能力，并对后续请求缓存之前的回复。

**触手可及的股票分析**：[AllMind AI](https://www.producthunt.com/posts/allmind-ai-your-personal-stock-analyst) 是一款承诺提供快速且经济的财务洞察的 AI 工具，正在角逐 Product Hunt 的榜首。

**自动化图谱变得更智能**：WhyHow.AI 推出了重大升级，支持**模式控制的自动化知识图谱 (schema-controlled automated knowledge graphs)**，旨在更有效地结构化用户上传的内容。新功能及其测试计划在 [Medium 文章](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3)中进行了介绍。

**对话式查询构建**：一篇[博客文章](https://rito.hashnode.dev/rental-apartment-search-with-langchain-self-querying-retriever)详细介绍了 **Self-querying retriever** 如何从自然语言输入中创建结构化查询，通过基于元数据的过滤来增强语义相似度搜索。

**LLM 的水印警告**：社区深入探讨了 AI 生成文本中的水印概念，这是一种植入可识别模式的技术，详见此资源页面：[Watermarking LLMs](https://watermarking.aisimplyexplained.tech/)。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**TinyGrad 应对段错误与训练难题**：讨论强调了在 **ROCm 6.1** 发布后设置 **tinygrad** 时遇到的段错误（segfaults）挑战，而 George Hotz 保证由于强大的 **CI**，`master` 分支是稳定的。

**AI 硬件被寄予超越云端的厚望**：社区辩论了像 **TinyBox** 这样的去中心化 AI 服务相对于传统云服务的优势，重点关注抗审查性、本地训练可行性以及实时用户数据训练的重要性。

**TinyGrad 机制内幕**：在 **tinygrad** 领域，成员们深入讨论了**张量堆叠 (stacking tensors)**、**形状追踪 (shape tracking)** 和**内存管理**，分享了揭示这个极简深度学习库内部结构的教程和文档。

**Windows 在 CUDA 支持上如履薄冰**：Windows 用户分享了使用 **WSL** 和 **Docker** 等工具运行带有 **CUDA** 的 **tinygrad** 的经验和变通方法，同时也承认该平台对此配置官方并不支持。

**George Hotz 记录 Tinygrad 即将到来的演进**：在每周总结中，Hotz 提到了未来讨论的重点领域，强调了 **mlperf** 的进展、潜在的 **NVIDIA CI** 策略，以及保持 **tinygrad** 代码库简洁的目标。

[ShapeTracker 教程](https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html)、[Uops 文档](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/uops-doc.md) 和 [CUDA Tensor Core 指南](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/cuda-tensor-core-pt1.md) 作为教育资源被分享，讨论中还引用了 [Meta AI](https://meta.ai)。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**Mixtral 险胜 Llama3**：根据分享的[数据集结果](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval)，**Mixtral-8x7B-Instruct-v0.1** 在德国 **RAG** 评估中表现优于 **Llama3 70b instruct**。然而，成员们指出评估指标可能存在问题，特别是“问题到上下文 (question to context)”指标，并暗示查询模板中可能存在格式错误（bug），这可能会影响结果。

**利用执行模型和 Haystack 增强聊天机器人**：Armifer91 正在为聊天机器人原型化一个 `execute_model` 函数，将某些功能分组并并行化 **MoE** 方法，而一个 **GitHub** [notebook](https://github.com/vblagoje/notebooks/blob/main/haystack2x-demos/haystack_rag_services_demo.ipynb) 展示了使用 **Haystack LLM** 框架动态调用服务。开发者正在探索改进 **Llama** 的技术，涉及用于微调（fine-tuning）的分词（tokenization），尽管面临对 **Hugging Face** 平台不稳定性的抱怨。

**德语语音识别 Whisper 的动态**：成员们正在测试各种用于德语语音识别的 **Whisper** 模型，如 [whisper-tiny-german](https://huggingface.co/primeline/whisper-tiny-german) 和 [whisper-base-quant-ct2](https://huggingface.co/jvh/whisper-base-quant-ct2/)，并就通过微调或量化以增强在智能手机上的功能达成共识。

**模板麻烦与分词纠葛**：与 **Llama-3** 模型中的模板和分词器（tokenizer）配置相关的复杂性在讨论中非常普遍，涉及特殊标记（special tokens）的零权重以及对话语境中的替代 `eos_tokens`。**ChatML** 模板是标准配置，但仍存在与分词器相关的挑战。

**DiscoLM 的德语精度问题**：针对德语应用微调 **DiscoLM** 引发了关于模型分词问题和潜在改进策略的辩论，其中 **Instruct** 模型可作为可能的基础。建议参考 **LeoLM** 的训练方法，并与 **occiglot** 团队联系，以增强 **Llama3** 在德语方面的表现。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**扩展 LLM 视野**：工程师们讨论了使用 **rope** 来扩展大语言模型（LLM）**context window** 的前景，表现出极大的热情，并引用了一篇 [Perplexity AI 文章](https://www.perplexity.ai/search/why-not-scale-0KMvWZKqSVGnYIBd_vpcng) 以进行深入了解。

**FineWeb 引起轰动**：拥有 15 万亿 **token** 的海量网络数据集 **FineWeb** 的发布引起了关注。根据 [Twitter](https://twitter.com/gui_penedo/status/1781953413938557276) 披露，由于其性能指标优于前代产品 RefinedWeb 和 C4，人们对其寄予厚望。

**框架成为焦点**：Discord 用户对 **Hydra framework** 褒贬不一，一些人欣赏其复杂的应用程序配置能力，而另一些人则在思考其独特之处；随着对 [Hydra GitHub 仓库](https://github.com/facebookresearch/hydra) 的引用，兴趣达到了顶峰。

**微软强大的 Phi-3 问世**：**Phi-3** 的发布引发了关注——其运行规模比前代 Phi-2 更大，并被推测将与 **llama 3 8B** 等知名模型竞争；通过 [关于 Phi-3 能力的推文](https://twitter.com/arankomatsuzaki/status/1782594659761389655) 分享的见解进一步助长了这些推测。

**Perplexity.ai 实现财务飞跃**：技术圈关注到 **Perplexity.ai** 成功完成了一轮融资，据称这将增强其搜索引擎实力——该消息在一条 [详述 6270 万美元融资的推文](https://twitter.com/AravSrinivas/status/1782784338238873769) 中披露。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile 对决中 70b 击败 8b**：用户表示，在与 **Llamafile** 集成时，**llama 3 70b** 是优于 8b 的首选，理由是后者存在运行问题，并强调 70b Q2 权重的大小为 26GB，处于可控范围。
- **M1 Pro 量化结果参差不齐**：有报告称 llama 模型的 Q2 变体在 **M1 Pro 系统**上出现了乱码输出；不过，相关人员澄清该模型在 **CPU 模式**下运行顺畅，尽管速度较慢。
- **Android 地址空间限制阻碍 Llamafile**：关于在 Android 上运行 **Llamafile** 的讨论因 Android 缺乏 **47 bit address space** 的限制而受阻，导致目前无法支持。
- **Redis 创始人赞赏 Llamafile**：**Redis** 的发明者在 Twitter 上对 **llama 3 70b** 版本的 **Llamafile** 表示认可，这一赞誉受到了 **Llamafile** 社区的庆祝。
- **多模态模型的端口技巧**：针对运行多个 **Llamafile** 实例的咨询，建议使用 `--port` 标志为并发运行的模型指定不同的端口。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **上下文大小的惊喜**：来自 4chan 的爆料指出，某款 AI 可能一直是在 **32k context size** 下运行的，这挑战了之前对其能力的假设。
  
- **模型缩放的替代方法**：一名成员提到了 Alpin 缩放 AI 模型的非传统方法，强调了 **dynamic ntk** 和 **linear scaling** 等策略，这些策略可能在不需要 **rope** 的情况下保持有效性。

- **Matt 发布 Llama 的 16k 配置**：Hugging Face 上发布了 **Matt 的 Llama 模型 16k 配置**，包括 "max_position_embeddings": 16000 等参数，模型类型指定为 "llama"。配置详情可见 [此处](https://huggingface.co/mattshumer/Llama-3-8B-16K/blob/main/config.json)。

- **医学知识变得易于获取**：深入的讨论集中在简化医学知识上；建议范围从为简化而 **fine-tuning** 一个 LLM，到开发一个将任务分解为专门阶段的 **Agent** 系统，最终将医学摘要翻译成通俗易懂的语言。

- **寻找小众语言的 OCR 数据**：有人请求支持非热门语言的 **OCR** 数据集，最好包含文档类型的数据，这表明目前正致力于扩大 AI 的语言覆盖范围和可访问性。

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Meta AI 的 'Imagine' 吸引工程师关注**：**Meta AI 的 'Imagine'** 激发了公会成员的兴奋，有人称其为 *insane*（疯狂），并要求提供展示其能力的具体示例。
  
- **寻找合适的开发工具**：成员们正在积极寻找适用于 **Large Language Models (LLMs)** 的成熟**开发工具**，这表明他们对优化工作流有着浓厚兴趣。

- **Azure OpenAI 服务卡顿**：用户对 **Azure OpenAI** 表示不满，报告称存在严重的**延迟**，请求有时需要超过 20 分钟，且在 15 秒内发出超过两个请求时会遇到速率限制问题。

- **识别 Azure 延迟来源**：一些人怀疑 Azure 的延迟问题可能是由于临时的**服务问题**，而不是该平台的一贯问题。

- **分享实时 API 响应跟踪工具**：分享了一个实用资源，[GPT for Work 的响应时间追踪器](https://gptforwork.com/tools/openai-api-and-other-llm-apis-response-time-tracker)，用于监控主要 LLM 的 **API response times**，这对于寻求性能优化的工程师来说非常有帮助。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **AI 领域新挑战者出现**：[Llama 3](https://llama.meta.com/llama3/) 在 [LMSYS arena 排行榜](https://chat.lmsys.org/?leaderboard)上并列第 5 位，与 Claude 3 Opus 和 GPT-4 变体等顶尖模型并驾齐驱，且可以在高端笔记本电脑上运行。

- **SimonW 的 Llama 3 工具包**：Simon Willison 推出了 [LLM](https://llm.datasette.io/)，这是一个包含命令行界面和 Python 库的工具集，旨在简化 Llama 3 及其他模型的使用。详细的使用说明可以在他的博客文章[此处](https://simonwillison.net/2024/Apr/22/llama-3/)找到。

- **AI 检查建筑作业**：AI 在建筑领域占据了一席之地，作为一种“预检”工具来发现建筑设计中的潜在问题和规范违规，尽管它尚未进展到创建蓝图的阶段。

- **蓝图解读仍处于起步阶段**：讨论围绕使用 AI 来解读建筑蓝图展开，特别是针对 PDF 格式的管道追踪，但尚未提出具体的解决方案。 

- **Hackernews 摘要需求**：有人询问关于生成 Hackernews 摘要的 bash 脚本，但讨论中未提及最新版本的细节。



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **需要垃圾邮件清理者**：**general-chat** 频道遭到大量*垃圾信息*轰炸，这些信息链接到一个未经授权且包含 NSFW 内容的 Discord 邀请。
- **Jamba 兼容性查询**：一位成员对 **Jamba** 是否兼容 **LM Studio** 感到好奇，并寻求其运行要求的细节，类似于 **Claude** 的内存占用情况。
- **Jamba 的内存消耗**：围绕运行 **Jamba** 的挑战展开了讨论，特别是其巨大的 RAM 需求，指出即使是 Google Colab 也无法提供必要的资源，而在 Google Cloud 上的尝试也无果而终。
- **垃圾链接失误**：频道中发布了一个承诺提供 NSFW 内容的不当**垃圾链接**，警惕的成员应忽略并举报。



---


**Alignment Lab AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---

# PART 2: 按频道详细摘要和链接



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1231134777797513296)** (1118 条消息🔥🔥🔥): 

_

- **Unsloth 支持 Phi-3 Mini**：Unsloth 宣布支持 Microsoft 的 Phi-3 Mini 4K Instruct 模型，并在 Hugging Face 上上传了 4bit 版本，旨在将其集成到 Unsloth 库中，尽管由于与 Llama 3 的架构差异需要进行一些修改。他们关于 Llama 3 的博客文章已更新了此信息，并正等待支持发布后的 14B 变体。
- **在 24GB VRAM 上成功进行 Fine-Tuning**：一位用户报告了使用 Unsloth 在 1x3090 24GB GPU 上以纯 BF16 质量成功 Fine-Tuning Llama 3 的经历，有效地处理了显存需求，且仅使用了 16GB 的 VRAM。
- **人体工程学工作站讨论**：成员们分享了关于人体工程学工作站设置的经验和建议，重点介绍了键盘、显示器、椅子以及升降桌对营造舒适工作环境的好处。
- **技术博客文章技巧**：根据对之前博客文章的反馈，Unsloth 即将发布的文章将包含更多 benchmarks 以及图片中的描述性文本，以提供更清晰的上下文和信息。
- **Phi-3 分析与期待**：用户对新发布的 Phi-3 模型持续保持期待和讨论，对其进一步的声明和应用感到好奇。一些用户正考虑对这些模型进行 finetuning，并热切期待其与现有库的兼容性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://course.fast.ai/">Practical Deep Learning for Coders - Practical Deep Learning</a>：一门为有一定编程经验的人设计的免费课程，旨在学习如何将 Deep Learning 和 Machine Learning 应用于实际问题。</li><li><a href="https://www.theverge.com/2024/4/23/24137534/microsoft-phi-3-launch-small-ai-language-model">Microsoft launches Phi-3, its smallest AI model yet</a>：Microsoft 发布 Phi-3，这是其迄今为止最小的 AI 模型。Phi-3 是今年三款小型 Phi 模型中的第一款。</li><li><a href="https://huggingface.co/chargoddard/llama3-42b-v0">chargoddard/llama3-42b-v0 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit">unsloth/Phi-3-mini-4k-instruct-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/cosmos-carl-sagan-gif-3394876">Watching The Cosmos GIF - Cosmos Carl Sagan - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/BarraHome/llama-3-orpo-v1">BarraHome/llama-3-orpo-v1 · Hugging Face</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/">Blog</a>：未找到描述</li><li><a href="https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers">Nvidia bans using translation layers for CUDA software &mdash; previously the prohibition was only listed in the online EULA, now included in installed files [Updated]</a>：Nvidia 禁止在 CUDA 软件中使用翻译层 —— 此前该禁令仅列在在线 EULA 中，现在已包含在安装文件中 [已更新]：翻译层成为众矢之的。</li><li><a href="https://unsloth.ai/blog/llama3">Finetune Llama 3 with Unsloth</a>：通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，上下文长度增加 6 倍！</li><li><a href="https://x.com/danielhanchen/status/1782790737798861281">Tweet from Daniel Han (@danielhanchen)</a>：Phi-3 Mini 3.8b Instruct 发布了！！68.8 MMLU 对比 Llama-3 8b Instruct 的 66.0 MMLU（Phi 团队自己的评估）。128K 长上下文模型也已发布在 https://huggingface.co/microsoft/Phi-3-mini-12...</li><li><a href="https://kinesis-ergo.com/shop/advantage2/">Advantage2 ergonomic keyboard by Kinesis</a>：轮廓设计，机械轴，全键可编程</li><li><a href="https://www.youtube.com/watch?v=E5kzAbD8D0w">Direct Preference Optimization (DPO)</a>：获取数据集：https://huggingface.co/datasets/Trelis/hh-rlhf-dpo 获取 DPO 脚本 + 数据集：https://buy.stripe.com/cN2cNyg8t0zp2gobJo 获取完整 Advanc...</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>：微调 Llama 3, Mistral 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://www.macrumors.com/2024/04/22/apple-acquires-french-ai-company/">Apple Acquires French AI Company Specializing in On-Device Processing</a>：Apple 收购了总部位于巴黎的人工智能初创公司 Datakalab，以推进其提供设备端 AI 工具的计划。Datakalab 专注于...</li><li><a href="https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-8b-unsloth-notebook">Kaggle Llama-3 8b Unsloth notebook</a>：使用 Kaggle Notebooks 探索并运行 Machine Learning 代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://www.reddit.com/r/hardware/comments/1c2dyat/geohot_hacked_4090_driver_to_enable_p2p/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad">GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>：你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - GitHub - tinygrad/tinygrad</li><li><a href="https://tenor.com/view/kevin-the-office-smirk-gif-3304715514430776968">Kevin The Office GIF - Kevin The Office Smirk - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/localghost/status/1781847388879220742">Tweet from Aaron Ng (@localghost)</a>：Llama 3 70b 从我的 M1 Max 传输到手机上，使用 MLX 达到约 7.6 tok/s。你家里的专属小型 GPT-4。</li><li><a href="https://unsloth.ai/blog">Blog</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit/blob/main/generation_config.json">generation_config.json · unsloth/llama-3-8b-Instruct-bnb-4bit at main</a>：未找到描述</li><li><a href="https://www.philschmid.de/fsdp-qlora-llama3">Efficiently fine-tune Llama 3 with PyTorch FSDP and Q-Lora</a>：学习如何使用 Hugging Face TRL, Transformers, PEFT 和 Datasets，配合 PyTorch FSDP 和 Q-Lora 高效微调 Llama 3 70b。</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth update: Mistral support + more</a>：我们很高兴发布对 Mistral 7B、CodeLlama 34B 以及所有其他基于 Llama 架构模型的 QLoRA 支持！我们添加了滑动窗口

attention, preliminary Windows and DPO support, and ...</li><li><a href="https://github.com/zenoverflow/datamaker-chatproxy">GitHub - zenoverflow/datamaker-chatproxy: Proxy server that automatically stores messages exchanged between any OAI-compatible frontend and backend as a ShareGPT dataset to be used for training/finetuning.</a>: 代理服务器，可自动将任何兼容 OAI 的前端和后端之间交换的消息存储为 ShareGPT 数据集，用于训练/微调。 - zenoverflow/datamaker-chatproxy</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth)</a>: 未找到描述</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets</a>: 将计算资源和书籍转换为 Instruct-Tuning 数据集 - e-p-armstrong/augmentoolkit</li><li><a href="https://github.com/ml-explore/mlx-swift/?tab=readme-ov-file">GitHub - ml-explore/mlx-swift: Swift API for MLX</a>: 用于 MLX 的 Swift API。通过在 GitHub 上创建账号来为 ml-explore/mlx-swift 的开发做出贡献。</li><li><a href="https://github.com/NVIDIA/open-gpu-kernel-modules/commit/1f4613dacec2638569a74b5e3dbcab01832f72a7">add P2P support · NVIDIA/open-gpu-kernel-modules@1f4613d</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/844">iPad App · ggerganov/llama.cpp · Discussion #844</a>: 我一直在尝试使用 llama 帮我在晚上给女儿讲故事。我写了一个简单的原生 iPad 应用，它使用了 llama.cpp，并提供了一些不错的模型/线程管理功能...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/4815">main : add Self-Extend support by ggerganov · Pull Request #4815 · ggerganov/llama.cpp</a>: #4810 的后续，基于此研究为 main 分支添加上下文扩展支持：https://arxiv.org/pdf/2401.01325.pdf。使用约 8k 上下文和基础 LLaMA 7B v... 进行了一些基础的事实提取测试。</li><li><a href="https://archive.ph/zbhlo">Apple (AAPL) Growth Opportunities: Southeast Asia and Africa, Lower-E&#x2026;</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1231169196390617108)** (167 条消息🔥🔥): 

- **发布新的 Llama AI 模型**: 已上传 [Hugging Face 模型: Llama 3 70B INSTRUCT 4bit](https://huggingface.co/unsloth/llama-3-70b-Instruct-bnb-4bit)，承诺 *微调 Mistral, Gemma 和 Llama 的速度提高 2-5 倍，且显存占用减少 70%*。随附的是用于 Llama-3 8b 的 [Google Colab GPU notebook](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)。
- **即将推出的教程材料**: 社区成员讨论了创建和分享指南或 notebook，以帮助使用聊天模板进行 **Instruct 模型微调**。据暗示，包括 *视频教程* 在内的材料可能正在制作中。
- **Llama C++ 批处理遇到困难**: 一位用户报告称，在 llama.cpp 中使用 `--cont-batching` 或 `cache_prompt` 进行同步 Prompt 处理时没有性能提升，因为顺序发送 Prompt 或并发发送耗时相同。
- **Gemma 关键词提取挑战**: 讨论了关于使用 **Gemma** 等 LLM 从客户评论中提取关键短语的问题，以及这往往会导致结果过于“有创意”或不准确，促使用户考虑使用 [KeyBERT](https://github.com/MaartenGr/KeyBERT) 等其他工具。
- **Unsloth 项目更新和社区贡献**: 期待 **Unsloth 在教程、博客文章和 Colab 工作室方面的持续工作**，并期待社区的贡献，包括分享 notebook。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html">Answer.AI - Efficient finetuning of Llama 3 with FSDP QDoRA</a>: 我们发布了 FSDP QDoRA，这是一种可扩展且内存高效的方法，旨在缩小参数高效微调（PEFT）与全量微调之间的差距。</li><li><a href="https://www.youtube.com/watch?v=vOA9JSDPJs0">Q*</a>: 点赞 👍。评论 💬。订阅 🟥。🏘 Discord: https://discord.gg/pPAFwndTJdhttps://github.com/hu-po/docs 从 r 到 Q∗：你的语言模型秘密地是一个 Q-Fun...</li><li><a href="https://github.com/MaartenGr/KeyBERT">GitHub - MaartenGr/KeyBERT: Minimal keyword extraction with BERT</a>: 使用 BERT 进行极简关键词提取。通过在 GitHub 上创建账号来为 MaartenGr/KeyBERT 的开发做出贡献。</li><li><a href="https://pytorch-dev-podcast.simplecast.com/">no title found</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-70b-Instruct-bnb-4bit">unsloth/llama-3-70b-Instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/@CUDAMODE">CUDA MODE</a>: 一个 CUDA 读书小组和社区 https://discord.gg/cudamode 补充内容见此处 https://github.com/cuda-mode 由 Mark Saroufim 和 Andreas Köpf 创建</li><li><a href="https://discord.gg/rWpeuatu">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的朋友和社区保持紧密联系。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1231139392899383388)** (716 messages🔥🔥🔥): 

- **LLaMA 模型训练问题**：成员们讨论了与微调 LLaMA 模型相关的问题，例如输出重复句子或过早停止。建议了调整训练配置和验证 Tokenizer 设置等解决方案。此外，用户在尝试上采样到 FP16 时遇到挑战，并被引导使用特定命令以成功进行训练和量化。

- **探索量化和 Unsloth 模型**：用户探索了量化如何影响模型质量，以及在有限硬件上运行模型的资源需求。对于实际应用，建议考虑使用 4-bit 左右的量化，以保持性能和质量之间的平衡。

- **设置并导入到 Unsloth**：提到了关于设置 Unsloth 环境和导入模型的挑战，特别是关于 Python 环境设置的问题。一些用户提到通过重新安装包或确保使用最新版本的 Unsloth 获得了成功。

- **在微调模型中使用推理**：与微调模型交互的用户注意到模型响应存在差异；例如，输出与输入 Prompt 完全相同。据报道，Unsloth 最近修复了此类 Tokenizer 问题（例如定义停止/eos Token），这些问题曾影响推理性能。

- **导出模型和微调策略**：分享了将 Unsloth 模型导出为 GGUF/vLLM 格式以及将 LoRA 适配器合并回 FP16 的技巧。用户寻求关于将知识嵌入 LLM 以供教学使用的最佳方法建议，多位社区成员也在寻求微调过程的通用指导。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/Finnish-NLP/llama-3b-finnish-v2/blob/main/config.json">config.json · Finnish-NLP/llama-3b-finnish-v2 在 main 分支</a>: 未找到描述</li><li><a href="https://huggingface.co/imone">imone (One)</a>: 未找到描述</li><li><a href="https://github.com/unslo">unslo</a>: GitHub 是 unslo 构建软件的地方。</li><li><a href="https://huggingface.co/spaces/mlabonne/OrpoLlama-3-8B">OrpoLlama-3-8B - mlabonne 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">首页</a>: 以 2-5 倍的速度、减少 80% 的内存微调 Llama 3, Mistral 和 Gemma LLM - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://www.hackster.io/news/tomeu-vizoso-s-open-source-npu-driver-project-does-away-with-the-rockchip-rk3588-s-binary-blob-0153cf723d44">Tomeu Vizoso 的开源 NPU 驱动项目摆脱了 Rockchip RK3588 的二进制 Blob</a>: 感谢 Vizoso 的努力，任何拥有 Rockchip RK3588 并运行机器学习工作负载的人现在都有了二进制 Blob 驱动程序的替代方案。</li><li><a href="https://github.com/unslothai/unsloth/issues/356">save_pretrained_gguf 方法 RuntimeError: Unsloth: 量化失败 .... · Issue #356 · unslothai/unsloth</a>: /usr/local/lib/python3.10/dist-packages/unsloth/save.py in save_to_gguf(model_type, model_directory, quantization_method, first_conversion, _run_installer) 955 ) 956 else: --> 957 raise RuntimeErro...</li><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM 模型 VRAM 计算器 - NyxKrage 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://youtu.be/SL2nZpv7dtY?si=Zne1z1tB8d_A7Ia9&t=1613">全量微调 vs (Q)LoRA</a>: ➡️ 获取完整脚本（及未来改进）的终身访问权限：https://trelis.com/advanced-fine-tuning-scripts/ ➡️ Runpod 一键微调...</li><li><a href="https://www.youtube.com/@MervinPraison/videos">Mervin Praison</a>: Mervin Praison</li><li><a href="https://tenor.com/view/atom-real-steel-movie-robot-fight-gif-13618149">Atom Real Steel GIF - 电影《铁甲钢拳》中的 Atom - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/love-actually-christmas-christmas-movie-workingtitlefilms-hugh-grant-gif-15362644">《真爱至上》圣诞 GIF - 圣诞电影《真爱至上》 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://colab.research.google.com/drive/1DGhWyCyf1BI-_yYaLYgOOkZuGAWiuqNj?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard">Big Code 模型排行榜 - bigcode 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/Dn0tmI0FFS">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://tenor.com/WBcE.gif">Carson Wcth GIF - Carson WCTH 这种事常有 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 以 2-5 倍的速度、减少 80% 的内存微调 Llama 3, Mistral 和 Gemma LLM</a>: 以 2-5 倍的速度、减少 80% 的内存微调 Llama 3, Mistral 和 Gemma LLM - unslothai/unsloth</li><li><a href="https://repo.anaconda.com/miniconda/">Index of /</a>: 未找到描述</li><li><a href="https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9">GGUF 量化概览</a>: GGUF 量化概览。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 以 2-5 倍的速度、减少 80% 的内存微调 Llama 3, Mistral 和 Gemma LLM</a>: 以 2-5 倍的速度、减少 80% 的内存微调 Llama 3, Mistral 和 Gemma LLM - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/yahma/alpaca-cleaned">yahma/alpaca-cleaned · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/pidugusundeep/Brat-and-snorkel/blob/master/ann-coll.py">Brat-and-snorkel/ann-coll.py 在 master 分支 · pidugusundeep/Brat-and-snorkel</a>: 支持文件。通过在 GitHub 上创建账户，为 pidugusundeep/Brat-and-snorkel 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/210">我在原生 Windows 上运行了 unsloth。 · Issue #210 · unslothai/unsloth</a>: 我在原生 Windows（非 WSL）上运行了 unsloth。你需要 Visual Studio 2022 C++ 编译器、Triton 和 DeepSpeed。我有一个完整的安装教程，我本想在这里写下来，但我现在在手机上...</li><li><a href="https://github.com/meta-llama/llama-recipes">GitHub - met</a>

<a href="https://github.com/meta-llama/llama-recipes">a-llama/llama-recipes: 用于微调 Meta Llama3 的脚本，支持组合式 FSDP 和 PEFT 方法，覆盖单节点/多节点 GPU。支持用于摘要和问答等应用的默认及自定义数据集。支持多种候选推理解决方案，如用于本地或云端部署的 HF TGI、VLLM。展示 Meta Llama3 在 WhatsApp & Messenger 应用的 Demo。</a>: 用于微调 Meta Llama3 的脚本，支持组合式 FSDP &amp; PEFT 方法，覆盖单节点/多节点 GPU。支持用于摘要和问答等应用的默认及自定义数据集...</li><li><a href="https://huggingface.co/imone/Llama-3-8B-fixed-special-embedding">imone/Llama-3-8B-fixed-special-embedding · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.save_strategy">Trainer</a>: 未找到描述</li><li><a href="https://github.com/sgl-project/sglang">GitHub - sgl-project/sglang: SGLang 是一种专为大语言模型 (LLMs) 设计的结构化生成语言。它使您与模型的交互更快速、更具可控性。</a>: SGLang 是一种专为大语言模型 (LLMs) 设计的结构化生成语言。它使您与模型的交互更快速、更具可控性。 - sgl-project/sglang</li><li><a href="https://github.com/hiyouga/LLaMA-Factory#hardware-requirement">GitHub - hiyouga/LLaMA-Factory: 统一 100 多个 LLMs 的高效微调</a>: 统一 100 多个 LLMs 的高效微调。通过在 GitHub 上创建账号为 hiyouga/LLaMA-Factory 的开发做出贡献。</li><li><a href="https://status.huggingface.co/">
Hugging Face 状态
</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1231174242029010995)** (76 条消息🔥🔥): 

- **瑞典语模型进展**：展示了 **[llama-3-instruct-bellman-8b-swe-preview](https://huggingface.co/neph1/llama-3-instruct-bellman-8b-swe-preview)** 模型，该模型经过训练以提高连贯性和推理能力。对使用 Unsloth 训练的模型表达了极大的热情。
  
- **推出 Ghost 7B Alpha**：宣布发布 **Ghost 7B Alpha**，优化了推理和多任务处理能力，并提供了 [model card](https://huggingface.co/ghost-x/ghost-7b-alpha)、[网站文档](https://ghost-x.vercel.app/docs/models/ghost-7b-alpha) 和 [demo](https://ghost-x.vercel.app/docs/notebooks/playground-with-ghost-7b-alpha) 等资源。

- **通过重训改进**：一位成员讨论了使用 Unsloth 最新的 4bit 版本重新训练 **Llama3** 模型，取得了成功的结果，并决定继续尝试不同的 hyperparameters。

- **Solobsd 发布西班牙语模型**：宣布了一个新的西班牙语模型 (**solobsd-llama3**)，基于 Alpaca 数据集的数据，并对展示的特定西班牙语变体表示了赞赏和询问。

- **模型微调讨论**：就如何在生成过程中有效地停止模型，以及如何在 Unsloth 和 Llama3 的环境下使用数据集模板进行了技术交流。贡献者们分享了成功训练和转换的建议及步骤。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mahiatlinux/MasherAI-7B-v6.1">mahiatlinux/MasherAI-7B-v6.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/SoloBSD/solobsd-llama3">SoloBSD/solobsd-llama3 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/hikikomoriHaven/llama3-8b-hikikomori-v0.1/">hikikomoriHaven/llama3-8b-hikikomori-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Remek/Llama-3-8B-Omnibus-1-PL-v01-INSTRUCT">Remek/Llama-3-8B-Omnibus-1-PL-v01-INSTRUCT · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/BarraHome/llama-3-orpo-v1-merged_16bit">BarraHome/llama-3-orpo-v1-merged_16bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/hi">Hi (Ho)</a>: 未找到描述</li><li><a href="https://huggingface.co/neph1/llama-3-instruct-bellman-8b-swe-preview">neph1/llama-3-instruct-bellman-8b-swe-preview · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/ghost-x/ghost-7b-alpha">ghost-x/ghost-7b-alpha · Hugging Face</a>: 未找到描述</li><li><a href="https://ghost-x.vercel.app/docs/models/ghost-7b-alpha">Ghost 7B Alpha</a>: 这一代大型语言模型专注于优化卓越的推理能力、多任务知识和工具支持。</li><li><a href="https://ghost-x.vercel.app/docs/notebooks/playground-with-ghost-7b-alpha">Playground with Ghost 7B Alpha</a>: 为了让每个人都能通过 Google Colab 和 Kaggle 等平台快速体验 Ghost 7B Alpha 模型，我们提供了这些 Notebook，以便您可以立即开始。</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6745">Support Llama 3 conversion by pcuenca · Pull Request #6745 · ggerganov/llama.cpp</a>: 分词器（Tokenizer）是 BPE。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1231279056901636198)** (73 条消息🔥🔥): 

- **色彩混淆难题**：一位成员表示，由于配色方案选择不当（绿底灰字），阅读欢迎信息非常困难。在根据反馈更改颜色后，问题已得到解决。

- **Google Colab 中的工作流困扰**：成员们讨论了在 Google Colab 中进行 CUDA 和 C++ 开发时面临的挑战，包括缺乏调试工具和语法高亮。对话涉及了使用 print 语句进行调试的混乱以及生产力下降等问题，一些人建议通过 SSH 使用 VSCode。

- **SSH 与 Colab 的难题**：分享了远程 SSH 访问 Google Colab 的经验，重点讨论了工作流效率低下以及远程 SSH 体验不佳的负面影响。链接中提供了一篇来自 Puget Systems 的教程，介绍如何在 Windows 10 上通过 SSH 设置 Jupyter Notebooks。

- **Unsloth Pro 的慈善追求**：讨论探讨了 Unsloth Pro 的潜在方向，建议申请慈善资助并开源代码。不过，有消息称 Unsloth 目前已获得资金并正在构建自己的平台。

- **辩论是否需要招聘频道**：成员们辩论了在服务器中添加 #jobs 频道的必要性和潜在风险。提出了对诈骗、频道杂乱以及保持对 Unsloth 关注度的担忧，目前尚未达成共识。

- **视觉愿景 - 模型兼容性建议**：针对未来支持各种模型提出了建议，包括可能与即将发布的 Llama-3 视觉版一同推出的视觉任务模型。此外，大家对新提到的模型（如 Phi-3）的指令（Instruction）版本也产生了好奇。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.pugetsystems.com/labs/hpc/How-To-Run-Remote-Jupyter-Notebooks-with-SSH-on-Windows-10-1477/">How To Run Remote Jupyter Notebooks with SSH on Windows 10</a>: 能够在远程系统上运行 Jupyter Notebooks 极大地增加了工作流的通用性。在这篇文章中，我将展示一种利用一些巧妙功能来实现这一目标的简单方法...</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=DdTsX6DQk24&t=2s">Lecture 14: Practitioners Guide to Triton</a>: https://github.com/cuda-mode/lectures/tree/main/lecture%2014
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1232348211570671667)** (1 条消息):

- **Perplexity Enterprise Pro 发布**：Perplexity 推出了 **Enterprise Pro**，这是一款专为企业设计的安全 AI 回答引擎，具有 **增强的数据隐私、SOC2 合规性、用户管理和单点登录 (SSO)** 等特性。随着 **Stripe、Zoom 和 Databricks** 等巨头已经开始利用其优势，Databricks 报告称每月节省了约 *5000 小时*。

- **Enterprise Pro 的影响与定价**：**Enterprise Pro** 面向软件、金融和体育等多种行业，为知识型工作者提供安全搜索快速、可靠信息的能力，定价为 **\$40/月** 或 **\$400/年/席位**。感兴趣的公司可以在 [Perplexity Enterprise](https://pplx.ai/enterprise) 注册。
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1231143301369958504)** (1005 条消息🔥🔥🔥): 

- **Perplexity Enterprise Pro 震撼发布**：官方频道和 [Bloomberg](https://www.bloomberg.com/news/articles/2024-04-23/ai-search-startup-perplexity-valued-at-1-billion-in-funding-round?cmpid=socialflow-twitter-business) 宣布了一项新的高级功能 **Perplexity Enterprise Pro**，以每月 $40 的价格提供增强的安全性和数据保护措施等额外功能。
- **公司增长与产品多元化**：在成功完成一轮融资后，Perplexity.ai 的估值已达到 **10 亿美元**，这标志着公司的扩张和更广泛的服务提供，其中包括 AI 泰斗 Yann LeCun 可能参与其中的传闻。
- **隐私担忧与澄清**：用户讨论引发了对数据隐私的担忧，以及付费用户的数据是否被用于训练 AI 模型；版主链接到了官方声明，暗示了数据使用许可和选项。
- **iOS App 挑战**：用户报告了 iPad 版 Perplexity 应用存在的持续问题，例如无法搜索或登录，支持团队建议受影响的用户通过私信寻求帮助。
- **预期发布中的潜在变化与功能**：随着版主对即将到来的更新给出的推测性暗示，用户猜测会有新功能发布、取消 Opus 限制或其他改进，从而对 4 月 23 日的公告充满期待。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://www.rabbit.tech/.">rabbit r1 - pickup party nyc live at 8PM ET</a>: 来自纽约 r1 提货派对活动的直播</li><li><a href="https://decoder.sh/videos/use-your-self_hosted-llm-anywhere-with-ollama-web-ui">Use Your Self-Hosted LLM Anywhere with Ollama Web UI</a>: 未找到描述</li><li><a href="https://www.bloomberg.com/news/articles/2024-04-23/ai-search-startup-perplexity-valued-at-1-billion-in-funding-round?cmpid=socialflow-twitter-business">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://docs.openwebui.com">🏡 Home | Open WebUI</a>: Open WebUI 是一个可扩展、功能丰富且用户友好的自托管 WebUI，旨在完全离线运行。它支持多种 LLM 运行器，包括 Ollama 和 OpenAI 兼容的 API。</li><li><a href="https://www.bloomberg.com/news/articles/2024-04-23/ai-search-startup-perplexity-valued-at-1-billion-">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://www.ycombinator.com/apply">Apply to Y Combinator | Y Combinator</a>: 要申请 Y Combinator 计划，请提交申请表。我们每年分两批接收公司。该计划包括每周二的晚餐、与 YC 合伙人的办公时间以及访问权限...</li><li><a href="https://en.m.wikipedia.org/wiki/Yann_LeCun">Yann LeCun - Wikipedia</a>: 未找到描述</li><li><a href="https://tenor.com/view/superstore-amy-sosa-im-just-guessing-just-guessing-wild-guess-gif-24963833">Superstore Amy Sosa GIF - Superstore Amy Sosa Im Just Guessing - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/money-mr-krabs-gif-18326632">Money Mr GIF - Money Mr Krabs - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/@AndrejKarpathy/videos">Andrej Karpathy</a>: 常见问题 Q：我该如何付钱给你？你有 Patreon 之类的吗？A：作为 YouTube 合作伙伴，我会分享视频中少量的广告收入，但我没有维护任何其他额外的付费渠道。我...</li><li><a href="https://tenor.com/view/think-about-it-use-your-brain-use-the-brain-think-brain-gif-7914082">Think About It Use Your Brain GIF - Think About It Use Your Brain Use The Brain - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.morphic.sh/">Morphic</a>: 一个完全开源、由 AI 驱动且具有生成式 UI 的回答引擎。</li><li><a href="https://tenor.com/view/yt-youtube-logo-gif-27453294">Yt Youtube GIF - Yt Youtube Logo - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/heidi-klum-number-two-2fingers-deuces-your-second-yes-gif-25857953">Heidi Klum Number Two GIF - Heidi Klum Number Two 2Fingers - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://fxtwitter.com/AravSrinivas/status/1782775219733844256?t=Oo_2sf1Yj-XImPRrzO19nA&s=19">Tweet from Aravind Srinivas (@AravSrinivas)</a>: 许多 Perplexity 用户告诉我们，由于数据和安全问题，他们的公司不允许他们在工作中使用它，但他们真的很想用。为了解决这个问题，我们很高兴推出...</li><li><a href="https://x.com/aravsrinivas/status/1781902284844421624?s=46">Tweet from Aravind Srinivas (@AravSrinivas)</a>: 4/23</li><li><a href="https://www.morphic.sh">Morphic</a>: 一个完全开源、由 AI 驱动且具有生成式 UI 的回答引擎。</li><li><a href="https://console.groq.com/playground?model=llama3-70b-8192">GroqCloud</a>: 体验世界上最快的推理速度</li><li><a href="https://www.chooseoxygen.com/en/blog/chatgpt-vs-notion-ai-comprehensive-comparison-for-ai-writing">ChatGPT vs Notion AI: An In-Depth Comparison For Your AI Writing Needs</a>: 两个 AI 工具 ChatGPT 和 Notion AI 的全面对比，包括功能、定价和使用场景。</li><li><a href="https://x.com/AravSrinivas/status/1781721468180767002">Tweet from Aravind Srinivas (@AravSrinivas)</a>: 8b 太棒了。可以用它创造更多体验。我们有一些想法。敬请期待！ ↘️ 引用 MachDiamonds (@andromeda74356) @AravSrinivas 你会将免费版 Perplexity 切换到...</li><li><a href="https://github.com/mckaywrigley/clarity-ai">GitHub - mckaywrigley/clarity-ai: A simple Perplexity AI clone.</a>: 一个简单的 Perplexity AI 克隆版。通过在 GitHub 上创建账号来为 mckaywrigley/clarity-ai 的开发做出贡献。</li><li><a href="https://www.google.com/amp/s/www.xataka.com/aplicaciones/ultimo-openai-llega-a-copilot-asistente-programacion-evoluciona-nuevo-modelo-ia/amp">Lo último de OpenAI llega a Copilot. El asistente de programación evoluciona con un nuevo modelo de IA</a>: 在过去的一年里，人工智能不仅支持了 DALL·E 等图像生成器和 ChatGPT 等聊天机器人，还...</li><li><a href="https://github.com/developersdigest/llm-answer-engine">GitHub - developersdigest/llm-answer-engine: Build a Perplexity-In</a>: 构建一个 Perplexity 风格的回答引擎。</li>

<ul><li><a href="https://github.com/developersdigest/llm-answer-engine">使用 Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper 构建一个受 Perplexity 启发的回答引擎</a>: Build a Perplexity-Inspired Answer Engine Using Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper - developersdigest/llm-answer-engine</li><li><a href="https://youtu.be/znOlwELyt8g?si=UDq4joNqi1n7z8i3">Eric Gundersen 谈论 Mapbox 如何使用 AWS 每天绘制数百万英里的地图</a>: 在此处了解更多关于 AWS 如何助力您的海量数据解决方案 - http://amzn.to/2grdTah。Mapbox 每天收集 1 亿英里的遥测数据...</li><li><a href="https://tenor.com/view/robot-depressed-marvin-hitch-hikers-guide-to-the-galaxy-gif-4931652">机器人抑郁 GIF - 机器人抑郁 Marvin - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://youtu.be/hFUaXEXfNnA?si=KWY0eyvRZNac2Gzt">AWS re:Invent 2023 - 客户主题演讲 Perplexity | AWS Events</a>: 听取 Perplexity 联合创始人兼 CEO Aravind Srinivas 讲述这家对话式人工智能 (AI) 公司如何通过提供...来重新定义搜索。</li><li><a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ)">Rick Astley - Never Gonna Give You Up (官方音乐视频)</a>: Rick Astley 的 “Never Gonna Give You Up” 官方视频。新专辑 'Are We There Yet?' 现已发行：在此下载：https://RickAstley.lnk.to/AreWe...</li><li><a href="https://youtu.be/YKMDw7ERxZ4?si=t0ybyzaEgUZNsihl">AWS re:Invent 2023 - 客户主题演讲 Anthropic</a>: 在这场 AWS re:Invent 2023 炉边对话中，Anthropic 的 CEO 兼联合创始人 Dario Amodei 与 Amazon Web Services (AWS) 的 CEO Adam Selipsky 讨论了 Anthr...</li><li><a href="https://share.wendabao.net">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/xx025/carrot">GitHub - xx025/carrot: Free ChatGPT Site List 这儿为你准备了众多免费好用的ChatGPT镜像站点</a>: Free ChatGPT Site List 这儿为你准备了众多免费好用的ChatGPT镜像站点。通过在 GitHub 上创建账号来为 xx025/carrot 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1231237470188732437)** (29 messages🔥): 

- **分享的 Perplexity AI 搜索**：**Sharing** 频道的成员分享了各种 Perplexity AI 搜索链接，涵盖从**积极育儿**到**不明确提示词的指令**等主题。每个分享的 Perplexity 页面都解决了特定的问题或信息请求。
- **分享指南**：提醒用户确保其分享的线程是可分享的，并提供了关于如何使线程**可分享**的说明链接。
- **Perplexity AI 登上新闻头条**：AI 搜索引擎初创公司 **Perplexity AI** 已被多家新闻媒体报道，频道内讨论了其最近的估值增长和融资努力。分享了 **TechCrunch** 的文章和 **CNBC** 对 CEO Aravind Srinivas 的采访，强调了公司的增长和企业版发布。
- **CEO 的 CNBC 采访转录**：分享了 **Perplexity 创始人兼 CEO Aravind Srinivas** 接受 **CNBC 独家采访** 的非官方转录稿，以及随附的视频采访链接。
- **公司估值讨论**：成员们讨论了 **Perplexity AI** 不断增长的估值，据报道该公司正以 **25 亿至 30 亿美元之间的估值再融资至少 2.5 亿美元**，这标志着自上一轮融资以来的快速增长。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.cnbc.com/2024/04/23/cnbc-exclusive-cnbc-transcript-perplexity-founder-ceo-aravind-srinivas-speaks-with-cnbcs-andrew-ross-sorkin-on-squawk-box-today.html">CNBC 独家：CNBC 转录稿：Perplexity 创始人兼 CEO Aravind Srinivas 今日在 “Squawk Box” 节目中接受 CNBC 的 Andrew Ross Sorkin 采访</a>: 未找到描述</li><li><a href="https://techcrunch.com/2024/04/23/perplexity-is-raising-250m-at-2-point-5-3b-valuation-ai-search-sources-say/">独家：消息人士称，Perplexity 正为其 AI 搜索平台以 25 亿至 30 亿美元的估值融资 2.5 亿美元以上</a>: Perplexity，这家 AI 搜索引擎初创公司，目前是炙手可热的资产。TechCrunch 获悉，该公司目前正在筹集至少 2.5 亿</li><li><a href="https://www.youtube.com/watch?v=LGuA5JOyUhE">Perplexity CTO Denis Yarats 谈 AI 驱动的搜索</a>: Perplexity 是一款回答用户问题的 AI 驱动搜索引擎。成立于 2022 年，估值超过 10 亿美元，Perplexity 最近突破了 1000 万月活跃用户...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1231252972319801344)** (3 messages):

- **寻找具备联网能力的 GPT**：一位新成员询问是否有类似于 GPT chat 但具备 **Internet access** 和来自网络的 **up-to-date information** 的 API。他们收到了 [Perplexity's documentation](https://docs.perplexity.ai/docs/model-cards) 的链接，并获知了提供联网功能的 **sonar online models**，同时受邀注册以获取引用（citations）权限。
- **提升模型性能的建议**：一位成员建议通过在 prompt 中加入 **one-shot examples** 来增强性能，旨在获得更精确的结果或让模型更好地理解指令。
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1231136613526929468)** (1044 条消息🔥🔥🔥): 

- **新手上路**：一位用户表示自己是 Stable Diffusion 的新手，正在下载 Forge Webui，并询问这是否是一个理想的选择，或者是否有更好的替代方案。
- **探索 AI 的创意前沿**：多位用户讨论了使用 Stable Diffusion 等 AI 工具生成图像和素材的兴趣。有人提到想制作游戏素材，另一位则表达了生成宇宙飞船和科幻主题的愿望。
- **技术故障**：几位用户寻求技术帮助，问题涉及 CUDA 错误、生成速度以及 ComfyUI 中缺失节点等。还有关于在 Forge 和 webui 等不同界面中使用特定模型的问题，以及关于在不同驱动器之间迁移安装程序的咨询。
- **AI 生成的未来**：用户们进行了随意的交流，思考利用 AI 来完美呈现伴侣或梦想中的家园。大家对 AI 生成定制内容的潜力表现出明显的兴奋。
- **对 Stability AI 发布内容的期待**：用户对 Stable Diffusion version 3 的发布和功能表达了好奇与怀疑，一些人转述了前 CEO Emad 的信息，并对最终发布的具体时间线和真正的开放程度进行了推测。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/chrlaf/status/1772228848387522728">来自 Christian Laforte (@chrlaf) 的推文</a>：@rajdhakad_ @USEnglish215753 @StabilityAI @EMostaque 我们的计划是很快先发布 API，以收集更多人类偏好数据，并验证我们的安全改进不会导致质量...</li><li><a href="https://wallet.bitcoin.com/">加密钱包 | 支持 Bitcoin (BTC)、Bitcoin Cash (BCH)、Ethereum (ETH) 和 ERC-20 代币</a>：下载 Bitcoin.com 的多币种加密钱包。一种简单且安全的方式来购买、出售、交易和使用加密货币。支持 Bitcoin (BTC)、Bitcoin Cash (BCH)、Ethereum (ETH) 和 ERC-20 代币...</li><li><a href="https://glif.app/@fab1an/glifs/clv488uy10000djtrx70u03no">glif - fab1an 开发的 StableDiffusion 3</a>：未找到描述</li><li><a href="https://www.youtube.com/playlist?list=PLIF38owJLhR1EGDY4kOnsEnMyolZgza1x">ComfyUI</a>：一种在本地电脑上使用 Stable Diffusion 模型创作 AI 艺术的更好方法。</li><li><a href="https://forums.developer.nvidia.com/t/cuda-enabled-geforce-1650/81010/5">支持 CUDA 的 GeForce 1650？</a>：如果您在 GROMACS 文档中找不到答案，我建议在官方 GROMACS 邮件列表中询问有关 GROMACS 配置的问题：[url]http://www.gromacs.org/Support/Mailing...</li><li><a href="https://github.com/comfyanonymous/ComfyUI/releases/download/latest/ComfyUI_windows_portable_nvidia_cu118_or_cpu.7z">未找到标题</a>：未找到描述</li><li><a href="https://download.pytorch.org/whl/cu121">未找到标题</a>：未找到描述</li><li><a href="https://civitai.com/images/10123212">pagartomas880 发布的图片</a>：未找到描述</li><li><a href="https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local">CUDA Toolkit 12.1 下载</a>：获取 NVIDIA 专有计算栈的最新功能更新。</li><li><a href="https://www.youtube.com/watch?v=ktxbXlF6UQE">揭露在 Discord 中跟踪你的网站！</a>：有一个名为 spy.pet 的网站，声称保存了 Discord 上的 40 亿条消息。通过它，你可以“查看你的朋友在 Discord 上做什么...</li><li><a href="https://github.com/Stability-AI/stablediffusion">GitHub - Stability-AI/stablediffusion: 使用 Latent Diffusion Models 进行高分辨率图像合成</a>：使用 Latent Diffusion Models 进行高分辨率图像合成 - Stability-AI/stablediffusion</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: 最强大且模块化的 Stable Diffusion GUI、API 和后端，具有图形/节点界面。</a>：最强大且模块化的 Stable Diffusion GUI、API 和后端，具有图形/节点界面。 - comfyanonymous/ComfyUI</li><li><a href="https://www.youtube.com/watch?v=9qd04u2Yj44">《摩登保姆》(Weird Science) 官方预告片 #1 - 小罗伯特·唐尼电影 (1985) HD</a>：订阅预告片：http://bit.ly/sxaw6h 订阅即将上映：http://bit.ly/H2vZUn 订阅经典预告片：http://bit.ly/1u43jDe 在 FACEB 上关注我们...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>：Stable Diffusion web UI。通过在 GitHub 上创建账户来为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://civitai.com/models/368139/character-sheet">角色表 (character sheet) - 角色表 | Stable Diffusion LoRA | Civitai</a>：未找到描述</li><li><a href="https://new.reddit.com/user/emad_9608/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/ltdrdata/ComfyUI-Manager">GitHub - ltdrdata/ComfyUI-Manager</a>：通过在 GitHub 上创建账户来为 ltdrdata/ComfyUI-Manager 的开发做出贡献。</li><li><a href="https://hidiffusion.github.io/">社交媒体标题标签</a>：社交媒体描述标签</li><li><a href="https://github.com/megvii-research/HiDiffusion">GitHub - megvii-research/HiDiffusion</a>：通过在 GitHub 上创建账户来为 megvii-research/HiDiffusion 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1231285314484572313)** (5 条消息):

- **VLLM 的张量并行 (Tensor Parallel)**：提到了在 VLLM 中实现 **tensor parallel** 的进展，并期待通过 *jamba* 支持来增强模型性能。
- **期待 Jamba API 发布**：有人表示需要 **jamba API**，以便在特定建模任务中利用全部上下文。
- **寻求经济的上下文管理**：一位用户分享了在使用 **Claude 3** 和 **Big-AGI** 时，由于成本迅速上升，在经济地管理上下文方面遇到的困难。他们发现了诸如 [memGPT](https://memgpt.ai/) 和 [SillyTavern SmartContext](https://docs.sillytavern.app/extras/extensions/smart-context/) 等潜在解决方案，并正在寻求更多高效管理上下文的方案。
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1231206936532357181)** (22 条消息🔥): 

- **Beastie Boys 推出重制版**：分享了一个 [名为 "Beastie Boys - Root Down" 的 YouTube 视频](https://www.youtube.com/watch?v=Xf1YF_MH1xc)；这是高清重制系列的一部分，其中包含了关于 "Ill Communication" 专辑的背景故事。
- **高质量重温 deadmau5 & Kaskade**：另一个 YouTube 分享展示了 [deadmau5 & Kaskade 的曲目 "I Remember (HQ)"](https://youtu.be/zK1mLIeXwsQ?t=119)，展示了该歌曲的音质，并提供了更多音乐和巡演信息的链接。
- **CIFAR100 中的潜在幽默**：CIFAR100 数据集被幽默地编码为 100 个类别，并以 [latent-CIFAR100](https://huggingface.co/datasets/Verah/latent-CIFAR100) 的形式分享，建议在 488 latent size 版本中使用 safetensors。
- **寻求更大像素的图像分类数据集**：一位成员在分享了一个简单的前馈神经网络在维度为 4x4x4 的潜在编码数据集上仅获得约 19% 的准确率后，询问是否有更大的图像分类数据集（64x64 或 128x128）。
- **关于符号系统和语言模型的论文**：[贡献了一批学术论文](https://arxiv.org/abs/2402.10588)，重点关注语言模型及其符号表示，指出语义向量空间是符号意义出现的阶段，类似于 LLM 中的语言理解。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/datasets/Verah/latent-CIFAR100">Verah/latent-CIFAR100 · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>：我们探讨了在不平衡、以英语为主的语料库上训练的多语言模型是否使用英语作为内部中转语言——这是一个对于理解语言模型如何运作至关重要的问题...</li><li><a href="https://tenor.com/view/hellinheavns-gif-23278790">Hellinheavns GIF - Hellinheavns - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://arxiv.org/abs/2311.03658">The Linear Representation Hypothesis and the Geometry of Large Language Models</a>：非正式地说，“线性表示假设”是指高层概念在某些表示空间中被线性地表示为方向。在本文中，我们探讨了两个密切相关的...</li><li><a href="https://www.youtube.com/watch?v=Xf1YF_MH1xc">Beastie Boys - Root Down</a>：高清重制！在这里阅读 Ill Communication 背后的故事：https://www.udiscovermusic.com/stories/ill-communication-beastie-boys-album/ 聆听更多...</li><li><a href="https://youtu.be/zK1mLIeXwsQ?t=119">deadmau5 &amp; Kaskade - I Remember (HQ)</a>：▶︎ https://deadmau5.ffm.to/randomalbumtitle 在这里关注 deadmau5 及其好友：https://sptfy.com/PjDO 当前巡演信息：https://deadmau5.com/shows 加入 ...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1231336533261553876)** (20 条消息🔥): 

- **DeepMind 的神经网络新工具包**：Google DeepMind 推出了 [Penzai](https://github.com/google-deepmind/penzai)，这是一个 JAX 研究工具包，旨在构建、编辑和可视化神经网络，旨在增强研究人员与其模型交互的方式。

- **征集高级研究助手的 Beta 测试人员**：Rubik.ai 正在为一款高级研究助手和搜索引擎征集 Beta 测试人员，该工具集成了 **Claude 3 Opus**、**GPT-4 Turbo** 等模型，使用促销代码 `RUBIX` 可获得两个月的免费高级访问权限。

- **探索大语言模型训练中的损失曲线**：讨论围绕诊断和理解模型训练过程中损失曲线的异常模式展开，推测低 batch sizes 和不均匀的损失平面可能是影响因素。

- **GPT System Prompts 存档现已发布**：[EveryoneIsGross/GPTs](https://github.com/EveryOneIsGross/GPTs) 托管了一系列用于 GPT 实验的 System Prompts，其中包括各种论文的实现以及在 Embeddings、RP、RAG 等概念上的实验。

- **Reddit 帖子质疑 LMSYS 基准测试的有效性**：一篇 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1c9nvpy/lmsys_becoming_less_useful) 挑战了 LMSYS 基准测试的实用性，认为由于难以设计出能准确区分模型智能的问题，该基准测试正变得越来越不可靠。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1c9nvpy/lmsys_becoming_less_useful/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-fineweb-15t-tokens-of-commoncrawl">Reddit - 探索一切</a>：未找到描述</li><li><a href="https://x.com/vikhyatk/status/1782296370420072576">来自 vik (@vikhyatk) 的推文</a>：奇怪的 Loss 曲线，如果不弄清楚早期那些下降的原因，今晚就睡不着了</li><li><a href="https://github.com/google-deepmind/penzai">GitHub - google-deepmind/penzai：一个用于构建、编辑和可视化神经网络的 JAX 研究工具包。</a>：一个用于构建、编辑和可视化神经网络的 JAX 研究工具包。 - google-deepmind/penzai</li><li><a href="https://github.com/EveryOneIsGross/GPTs">GitHub - EveryOneIsGross/GPTs：我的 GPT 实验和工具的加载区。</a>：我的 GPT 实验和工具的加载区。通过创建 GitHub 账号为 EveryOneIsGross/GPTs 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1231135314446389270)** (650 条消息🔥🔥🔥): 

- **LLaMA 与 Phi 的对决**：随着成员们将**新发布的 Phi-3-mini 模型**与 [LLaMA-3](https://x.com/sebastienbubeck/status/1782627991874678809?s=46) 和 **GPT-3.5** 进行对比，讨论愈发激烈。**Phi-3-mini** 的性能（尤其是在 4-bit 量化下）受到了审视，人们对其重复输出感到担忧，并热切期待模型权重的发布。
- **Hugging Face 的技术故障**：Hugging Face 面临宕机，推测可能与新的 **[FineWeb 数据集](https://huggingface.co/datasets/HuggingFaceFW/fineweb)** 或 **LLaMA-3 的高需求**有关。虽然服务已断断续续恢复，但问题依然存在。
- **棘手的模型行为**：围绕 **LLaMA-3** 的讨论表明模型倾向于产生幻觉（Hallucinate），或者在微调后无法吸收新信息。特别是 **Phi-3-mini** 模型，据报道在停止生成方面存在问题，并且可能配置了错误的 **EOS Token**。
- **模型微调的效率**：成员们讨论了用于微调大语言模型的 **QLoRA 与 LoRA**，并分享了关于它们在生产环境中的有效性和潜在用途的看法，特别是引用了 [QLoRA 研究](https://twitter.com/teortaxesTex/status/1781963108036088060)。
- **新兴的开发者兴趣**：呼吁从事模型、数据集或使用 AI 模型的系统的开发者建立联系，这表明社区对讨论并可能在 AI 和 NLP 项目上进行协作的兴趣日益浓厚。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/suchenzang/status/1782830272792404232">来自 Susan Zhang (@suchenzang) 的推文</a>：它似乎喜欢通过自我对话偏离正确的解决方案……</li><li><a href="https://evalplus.github.io/leaderboard.html">EvalPlus Leaderboard</a>：未找到描述</li><li><a href="https://x.com/natolambert/status/1782600141159174398">来自 Nathan Lambert (@natolambert) 的推文</a>：我真心希望 Phi-3 能证明我们在评估作弊（evaluation doping）方面的担忧是错的，它实际上是一个出色的模型。但是，在对数计算量（log compute）与 MMLU 的图表中作为一个离群值，确实有点可疑。</li><li><a href="https://x.com/awnihannun/status/1782436898285527229">来自 Awni Hannun (@awnihannun) 的推文</a>：进阶操作：在 iPhone 15 Pro 上对 4-bit Llama 3 8B 进行 QLoRA 微调。由 David Koski 提供的 (Q)LoRA MLX Swift 示例即将发布：https://github.com/ml-explore/mlx-swift-examples/pull/46 适用于许多模……</li><li><a href="https://huggingface.co/blog/how-to-train-sentence-transformers">训练与微调 Sentence Transformers 模型</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.10198">RAG 模型的忠实度如何？量化 RAG 与 LLM 内部先验之间的拉锯战</a>：检索增强生成（RAG）常用于修复幻觉并为大语言模型（LLM）提供最新知识。然而，当 LLM 独立回答错误时……</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/huybery/status/1781172838361334015">来自 Binyuan Hui (@huybery) 的推文</a>：刚刚评估了 Llama3-8B-base 的编程能力👇🏻</li><li><a href="https://huggingface.co/abacaj/phi-2-super">abacaj/phi-2-super · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/rage-gif-24341837">Rage GIF - Rage - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/gui_penedo/status/1781953413938557276?s=46">来自 Guilherme Penedo (@gui_penedo) 的推文</a>：我们刚刚发布了 🍷 FineWeb：15 万亿 token 的高质量网页数据。我们过滤并去重了 2013 年至 2024 年间所有的 CommonCrawl 数据。在 FineWeb 上训练的模型优于 RefinedWeb、C4……</li><li><a href="https://huggingface.co/papers/2404.14047">论文页面 - 低比特量化的 LLaMA3 模型表现如何？一项实证研究</a>：未找到描述</li><li><a href="https://x.com/sebastienbubeck/status/1782627991874678809?s=46">来自 Sebastien Bubeck (@SebastienBubeck) 的推文</a>：Phi-3 发布了，而且它……很棒 :-)。我制作了一个简短的演示，让你感受 Phi-3-mini (3.8B) 的能力。请关注明早的权重开源发布及更多公告……</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct">microsoft/Phi-3-mini-4k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://openrouter.ai/playground?models=meta-llama/llama-3-70b-instruct">OpenRouter</a>：LLM 及其他 AI 模型的路由服务</li><li><a href="https://github.com/mozilla-Ocho/llamafile?tab=readme-ov-file#using-llamafile-with-external-weights">GitHub - Mozilla-Ocho/llamafile：通过单个文件分发和运行 LLM。</a>：通过单个文件分发和运行 LLM。通过在 GitHub 上创建账号来为 Mozilla-Ocho/llamafile 的开发做出贡献。</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/tokenizer_config.json">tokenizer_config.json · microsoft/Phi-3-mini-128k-instruct (main 分支)</a>：未找到描述</li><li><a href="https://github.com/stanfordnlp/pyreft">GitHub - stanfordnlp/pyreft：ReFT：语言模型的表示微调（Representation Finetuning）</a>：ReFT：语言模型的表示微调 - stanfordnlp/pyreft</li><li><a href="https://www.youtube.com/watch?v=z5rRZdiu1UE">Beastie Boys - Sabotage</a>：高清重制版！在此阅读 Ill Communication 背后的故事：https://www.udiscovermusic.com/stories/ill-communication-beastie-boys-album/ 聆听更多来自……</li><li><a href="https://huggingface.co/datasets/Replete-AI/OpenCodeInterpreterData">Replete-AI/OpenCodeInterpreterData · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/Rombo-Hermes-2.5-Extra-code">Replete-AI/Rombo-Hermes-2.5-Extra-code · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb">HuggingFaceFW/fineweb · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/Rombo-Hermes-2.5-Extra-code-sub-50k">Replete-AI/Rombo-Hermes-2.5-Extra-code-sub-50k · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/Rombo-Hermes-2.

5-Extra-code-Medium">Replete-AI/Rombo-Hermes-2.5-Extra-code-Medium · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://fast.snova.ai">Streamlit</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1231255157178634414)** (78 messages🔥🔥): 

- **处理 Zero 3 中的 OOM**：一位用户报告称 Deepspeed **Zero 3** 明显比 **Zero 2** 慢，并且即使开启了 CPU offloading 也会遇到 OOM 错误，询问这是否属于正常现象并[寻求最佳使用建议](https://discord.com/channels/1053877538025386074/1149866623109439599/1231634503714340955)。
- **单 GPU 优化 vs. NVLink**：一位用户思考在处理单个 prompt 时，如何最好地利用带有 NVLink 的双 RTX 3090 来提升性能；而另一位用户则建议使用 **single-GPU** 最快，理由是 **multi-GPU** 设置存在同步开销。
- **Llama 微调与训练指南**：讨论涉及在许可规则内为模型微调生成合成数据，一位用户警告不要使用生成的数据来改进非 **Llama** 模型，其他用户则讨论了微调中示例难度的正确比例。
- **学习率技术与 LLM 中的遗忘**：用户讨论了如 **discriminative learning rates**（判别式学习率）和 **gradual unfreezing**（逐渐解冻）等技术在 2024 年是否依然流行，一位用户表示不熟悉，而另一位则确认这些技术确实仍在使用。
- **寻找合适的微调指南**：多位用户就指令微调（instruction fine-tuning）的最佳实践和资源给出了建议，倾向于选择 Hugging Face 博客，避开 Medium 文章，并特别推荐了 **Labonne's GitHub** 上的教程。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/chargoddard/mistral-11b-slimorca">chargoddard/mistral-11b-slimorca · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2402.01364">Continual Learning for Large Language Models: A Survey</a>: 大语言模型（LLM）由于其庞大规模带来的高昂训练成本，不便于频繁重新训练。然而，为了赋予 LLM 新技能并保持更新，更新是必要的...</li><li><a href="https://arxiv.org/abs/2404.08865">LLM In-Context Recall is Prompt Dependent</a>: 大语言模型（LLM）的激增凸显了进行彻底评估以辨别其比较优势、局限性和最佳用例的关键重要性。特别是...</li><li><a href="https://arxiv.org/abs/2212.08037">Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models</a>: 大语言模型（LLM）在极少或没有直接监督的情况下展示了令人印象深刻的结果。此外，越来越多的证据表明 LLM 在信息检索场景中可能具有潜力...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1231590177172881519)** (7 messages): 

- **AI 视觉模型的新基准**：xAI 发布了专为 **Grok-1.5-vision-preview** 设计的 **RealWorldQA** 基准数据集，提供直接的问答场景。
- **对数据集用途的困惑**：对于 **RealWorldQA** 是训练集还是基准测试集曾有短暂困惑，随后根据 xAI 关于 [Grok-1.5 的博客文章](https://x.ai/blog/grok-1.5v) 澄清其为基准测试集。
- **对额外数据集的兴趣**：部分成员对新的基准数据集表示热烈欢迎，认为它对测试未来版本的 **Obsidian** 很有用。
- **对训练集的渴望**：尽管认可基准数据的有用性，成员们仍表示有兴趣获取训练数据集。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-1.5v">Grok-1.5 Vision Preview</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/xai-org/RealworldQA?row=2">xai-org/RealworldQA · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1231238181647679539)** (89 messages🔥🔥): 

- **使用 LLaMA 评估 RAG**：讨论集中在使用 LLaMA index 评估 **检索增强生成 (RAG)** 的性能，暗示 **Mistral 7b v2** 的表现似乎优于 LLaMA 3b instruct 等其他模型。分享了一个用于此评估的有用资源：[OpenAI Cookbook 示例](https://cookbook.openai.com/examples/evaluation/evaluate_rag_with_llamaindex)。

- **解读 Superposition Prompting**：社区正在探讨一篇关于名为 *superposition prompting* 的新 RAG 提示词方法的论文，该方法旨在更高效地处理长上下文（[Superposition Prompting Paper](https://arxiv.org/abs/2404.06910)）。一位成员分享了他们在生产环境中应用该方法的实践经验，并讨论了关于上下文排序的注意事项。

- **研究人员分享 RAG 见解**：分享了几篇关于 **RAG** 方法论的论文，重点介绍了利用 LLMs 改进检索和可信度感知生成等创新，以及解决长上下文推理中的挑战。值得注意的是，一篇综述论文详细阐述了 **RAG** 框架的演进和组织结构（[RAG Evolution Paper](https://arxiv.org/abs/2404.10981)）。

- **Function-Calling RAG 技术**：Pamela Fox 关于使用 function-calling 的 **RAG** 技术的博客文章被广泛引用，作为理解和实现 **RAG** 方法的重要资源（[Pamela Fox's RAG post](https://blog.pamelafox.org/2024/03/rag-techniques-using-function-calling.html)）。此外，来自 Azure-Samples 的 GitHub 仓库为设置 **RAG** 方法提供了范例（[Azure-Samples GitHub](https://github.com/Azure-Samples/azure-search-openai-demo/tree/main/app/backend/approaches)）。

- **RAG 中检索与生成的融合**：讨论趋向于将检索集成到 LLM 计划的一部分，以创建基于文档引用的半结构化输出。示例包括结合 Cohere 和 Claude-3 的能力来演示这种方法，并呼吁为从多个文档中合成信息的 **RAG** 模型建立基准测试（[CLA Document Format](https://docs.cohere.com/docs/retrieval-augmented-generation-rag)）。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/BlancheMinerva/status/1782437494585282965">Stella Biderman (@BlancheMinerva) 的推文</a>：为 RAG 模型创建一个基准测试，其中所有问题都需要综合多个文档的信息来回答。研究在公开数据上训练的模型在该基准上的表现，并且...</li><li><a href="https://arxiv.org/abs/2404.10981">大语言模型检索增强文本生成综述</a>：检索增强生成 (RAG) 将检索方法与深度学习进展相结合，通过动态整合外部信息，解决大语言模型 (LLM) 的静态局限性...</li><li><a href="https://arxiv.org/abs/2404.06910">Superposition Prompting：改进并加速检索增强生成</a>：尽管大语言模型 (LLM) 取得了成功，但它们在处理长上下文时存在显著缺陷。其推理成本随序列长度呈二次方增长...</li><li><a href="https://docs.anthropic.com/claude/docs/long-context-window-tips">长上下文窗口技巧</a>：未找到描述</li><li><a href="https://cookbook.openai.com/examples/evaluation/evaluate_rag_with_llamaindex">使用 LlamaIndex 评估 RAG | OpenAI Cookbook</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.05825">LLM 增强检索：通过语言模型和文档级嵌入增强检索模型</a>：最近，与传统的稀疏或基于词袋的方法相比，基于嵌入的检索或稠密检索显示出了最先进的结果。本文介绍了一种与模型无关的文档...</li><li><a href="https://blog.pamelafox.org/2024/03/rag-techniques-using-function-calling.html">RAG 技术：利用 Function calling 实现更结构化的检索</a>：未找到描述</li><li><a href="https://blog.pamelafox.org/2024/02/rag-techniques-cleaning-user-questions.html">RAG 技术：使用 LLM 清理用户问题</a>：未找到描述</li><li><a href="https://github.com/Azure-Samples/azure-search-openai-demo/tree/main/app/backend/approaches">azure-search-openai-demo/app/backend/approaches at main · Azure-Samples/azure-search-openai-demo</a>：一个在 Azure 中运行的检索增强生成模式示例应用，使用 Azure AI Search 进行检索，并使用 Azure OpenAI 大语言模型来驱动 ChatGPT 风格和问答体验...</li><li><a href="https://docs.cohere.com/docs/retrieval-augmented-generation-rag">检索增强生成 (RAG) - Cohere 文档</a>：未找到描述</li><li><a href="https://blog.pamelafox.org/2024/03/evaluating-rag-chat-apps-can-your-app.html">评估 RAG 聊天应用：你的应用能说“我不知道”吗？</a>：未找到描述</li><li><a href="https://github.com/HK3-Lab-Team/PredCST">GitHub - HK3-Lab-Team/PredCST：从文本中学习具体语法树 (Concrete Syntax Tree) 的预测模型。</a>：从文本中学习具体语法树的预测模型。 - HK3-Lab-Team/PredCST</li><li><a href="https://arxiv.org/abs/2404.06809">并非所有上下文都平等：教导 LLM 进行具备可信度意识的生成</a>：大语言模型的快速发展导致了检索增强生成 (RAG) 的广泛采用，它通过整合外部知识来缓解知识瓶颈并减少...</li><li><a href="https://arxiv.org/abs/2404.06347">RAR-b：推理即检索基准测试 (Reasoning as Retrieval Benchmark)</a>：语义文本相似度 (STS) 和信息检索任务 (IR) 是过去几年记录嵌入模型进展的两个主要途径。在兴起的检索...</li><li><a href="https://arxiv.org/abs/2404.06082">一种针对长上下文 LLM 定制的源代码查询 RAG 方法</a>：虽然大语言模型 (LLM) 的上下文长度限制已得到缓解，但仍阻碍了它们在软件开发任务中的应用。本研究提出了一种结合...的方法。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1231147992568959027)** (343 条消息🔥🔥):

- **创意 AI 替代方案登场**：在等待官方 **WorldSim** 平台回归期间，许多用户已转向托管在 **HuggingChat** 上的替代方案，如 **Super WorldSim** 和 **Snow World Simulator**。他们正在定制这些替代方案以提供专业化体验，例如构建超级英雄宇宙或进行类 D&D 游戏。
- **Super WorldSim 随着改进而演进**：来自 **Jetblackrlsh** 的持续更新为 **Super WorldSim** 引入了新功能，如 **Mind Meld** 和 **Improv**，增强了用户体验，并使其复杂程度更接近 **Claude Opus**。
- **社区想象力蓬勃发展**：在平台替代方案中，用户深度参与，演化出复杂的虚构世界，并生成详尽的系统发育树（phylogenetic trees），以记录其模拟物种数百万年来的发展。
- **Discord 成为民主化世界构建的舞台**：一个显著的趋势正在兴起，像 **Rundeen** 这样的用户在 **Discord** 上设置了民主控制的 **WorldSim** 机器人。社区对协作式故事构建和探索的潜力充满热情。
- **开源模型铺就 AI 模拟的未来**：人们似乎正在形成一种共识，即开源 AI 模型对于未来的类 **WorldSim** 体验至关重要。**Llama 3** 预期中的更大规模模型因其在推动这些创意模拟向前发展方面的潜力而备受关注。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">world_sim</a>：未找到描述</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>：让社区最好的 AI 聊天模型惠及每一个人。</li><li><a href="https://console.groq.com/playground?model=llama3-70b-8192">GroqCloud</a>：体验世界上最快的推理</li><li><a href="https://hf.co/chat/assistant/662404223e2307950aa903bc">Super World Sim - HuggingChat</a>：在 HuggingChat 中使用 Super World Sim 助手</li><li><a href="https://hf.co/chat/assistant/66252be0705754b4e74c5e3f">Snow World Simulator - HuggingChat</a>：在 HuggingChat 中使用 Snow World Simulator 助手</li><li><a href="https://hf.co/chat/assistant/6626e4869232378718adc5f2">Snow Singer Simulator - HuggingChat</a>：在 HuggingChat 中使用 Snow Singer Simulator 助手</li><li><a href="https://a.co/d/0gve1yp">未找到标题</a>：未找到描述</li><li><a href="https://books2read.com/u/3GPpKP">现已在您喜爱的数字商店上架！</a>：Nicholas Alexander Benson 所著《The Architects' Conundrum: Quantumom vs. Data Dad》</li><li><a href="https://hf.co/chat/assistant/65bff23f5560c1a5c0c9dcbd">Image Generator - HuggingChat</a>：在 HuggingChat 中使用 Image Generator 助手</li><li><a href="https://tinyurl.com/SuperWorldSim">Super World Sim - HuggingChat</a>：在 HuggingChat 中使用 Super World Sim 助手</li><li><a href="https://hf.co/chat/assistant/66248a7a29ce1e0f4dd260fe">HuggingChat</a>：让社区最好的 AI 聊天模型惠及每一个人。</li><li><a href="https://hf.co/chat/assistant/66252be0705754b4e74">HuggingChat</a>：让社区最好的 AI 聊天模型惠及每一个人。</li><li><a href="https://hf.co/chat/assistant/662404223e230">HuggingChat</a>：让社区最好的 AI 聊天模型惠及每一个人。</li><li><a href="https://hf.co/chat/assistant/6623fcdb1a7a58ed5e441db2">HuggingChat</a>：让社区最好的 AI 聊天模型惠及每一个人。</li><li><a href="https://hf.co/chat/assistant/66240">HuggingChat</a>：让社区最好的 AI 聊天模型惠及每一个人。</li><li><a href="https://hf.co/chat/assistant/662404223e2307">HuggingChat</a>：让社区最好的 AI 聊天模型惠及每一个人。</li><li><a href="https://www.suzannetreister.net/Ampages/Amenu.html">Suzanne Treister - Amiga Videogame Stills - 菜单</a>：未找到描述</li><li><a href="https://dreams-of-an-electric-mind.webflow.io/eternal">eternal mode • infinite backrooms</a>：人工智能的疯狂梦境——胆小者或意志薄弱者慎入
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1231142060816334859)** (635 条消息🔥🔥🔥): 

- **GPU Offloading 与系统资源占用**：用户讨论了 **LM Studio** 在各种 GPU 上的性能，特别是关于在 **AMD GPU**（使用 ROCm）和 **Nvidia GPU** 上运行模型的具体问题。会议指出，**GPU offloading** 对于最大化性能是必要的，如果系统没有正确进行 offloading，可能会导致 CPU 占用率达到 100%，从而导致效率低下。

- **LM Studio 与 Hugging Face 的问题**：用户报告了由于 **Hugging Face** 宕机导致无法搜索和下载模型的问题，这似乎影响了 LM Studio 的功能，并显示 **503** 和 **500** 等错误消息。**Heyitsyorkie** 确认 Hugging Face 存在 API 问题，影响了模型浏览器（model explorer）功能。

- **在 LM Studio 中利用 LLM**：用户寻求关于为模型创建特定 **system prompts** 以进行 **D&D 战役**等角色扮演场景的建议，以及如何处理对话中的 **max token limits** 和 **rolling windows**。其中一个建议是使用 LM Studio 中的“**AI assistant (python)**”预设，并在提示词末尾附上预期 JSON schema 的示例。

- **模型与 API 问题**：讨论内容包括关于**加载特定模型**的查询、AVX2 等**不支持的处理器指令**问题、处理**授权问题**以及“Unsupported format”等**错误消息**。用户请求了潜在的修复方案和变通方法。

- **AI 模型与量化问题**：用户探讨了不同 AI 模型量化（例如 **IQ1M vs. IQ2XS**）之间的差异，并讨论了即将推出的 **Llama 3 400b 模型**，推测了运行此类大型模型的系统要求和容量。

- **LM Studio 功能请求与反馈**：用户表达了对**后台运行 LM Studio** 等功能的期望，并对缺乏**隐私政策**提出了质疑。同时也对 LM Studio 让 **AI 变得触手可及**表示了赞赏。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/lmstudioai/status/1782390856986550384?s=46">来自 LM Studio (@LMStudioAI) 的推文</a>：LM Studio 内的模型搜索/下载可能会受到此次 Hugging Face 停机的影响。请关注后续更新 ↘️ 引用 Hugging Face Status (@hf_status) 我们正在经历 Hugging Face 的停机...</li><li><a href="https://lmstudio.ai/rocm">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://docs.useanything.com/feature-overview/llm-selection/lmstudio">LMStudio | AnythingLLM by Mintplex Labs</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/local-server">本地 LLM 服务器 | LM Studio</a>：你可以通过在 localhost 上运行的 API 服务器，使用在 LM Studio 中加载的 LLMs。</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta 版本</a>：未找到描述</li><li><a href="https://www.youtube.com/@IBMTechnology/playlists">IBM Technology</a>：无论是 AI、自动化、网络安全、数据科学、DevOps、量子计算还是介于两者之间的任何领域，我们都提供关于科技领域重大话题的教育内容。订阅以提升你的技能...</li><li><a href="https://ollama.com/blog/openai-compatibility">OpenAI 兼容性 · Ollama 博客</a>：Ollama 现在初步兼容 OpenAI Chat Completions API，使得通过 Ollama 在本地模型上使用为 OpenAI 构建的现有工具成为可能。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c858ac/llama3_seems_to_get_stuck_in_loops_sometimes/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/tree/main">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF at main</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard">Big Code Models 排行榜 - bigcode 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat-GGUF">Qwen/CodeQwen1.5-7B-Chat-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1">Vision Models (GGUF) - lmstudio-ai 集合</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ca8uxo/llavallama38b_is_released/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g&">[1小时演讲] 大语言模型入门</a>：这是一个面向普通观众的 1 小时大语言模型介绍：它是 ChatGPT、Claude 和 Bard 等系统背后的核心技术组件。什么是...</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF/tree/main">lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF at main</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c9m6ei/lpt_llama_3_doesnt_have_selfreflection_you_can/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/Crizomb/ai_pdf">GitHub - Crizomb/ai_pdf: 在本地与任何 PDF 对话。提出问题，获取带有有用参考的答案。在数学 PDF 上表现良好（将其转换为 LaTex，一种计算机可理解的数学语法）</a>：在本地与任何 PDF 对话。提出问题，获取带有有用参考的答案。在数学 PDF 上表现良好（将其转换为 LaTex，一种计算机可理解的数学语法） - Crizomb/ai_pdf</li><li><a href="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio">GitHub - BBC-Esq/VectorDB-Plugin-for-LM-Studio: 创建一个 ChromaDB 向量数据库以配合在服务器模式下运行的 LM Studio 的插件！</a>：创建一个 ChromaDB 向量数据库以配合在服务器模式下运行的 LM Studio 的插件！ - BBC-Esq/VectorDB-Plugin-for-LM-Studio</li><li><a href="https://github.com/mlabonne/llm-course">GitHub - mlabonne/llm-course: 进入大语言模型 (LLMs) 领域的课程，包含路线图和 Colab 笔记本。</a>：进入大语言模型 (LLMs) 领域的课程，包含路线图和 Colab 笔记本。 - mlabonne/llm-course</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c8c7xj/easiest_way_to_setup_rag_windows_nvidia_gpu/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://status.huggingface.co/">
Hugging Face 状态
</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1231136215638736983)** (314 条消息🔥🔥):

- **Llama 3 与替代模型**：用户正在探索各种版本的 **Llama 3** 以获得更好的性能，将其与 **Goliath 120B** 等模型进行对比，并讨论 **Mistral**。对话内容包括 **Llama 3** 在基准测试中的表现，以及对变体进行微调（finetuning）是否能赶上 **GPT-4**。

- **Meta-Llama-3-8B-Instruct-GGUF 的担忧**：人们对 **Llama 3 8B Instruct GGUF** 的**无限生成问题**（infinity generation issue）表示担忧，即模型会无休止地生成内容。用户建议通过停止字符串（stop strings）进行修复，并考虑尝试不同的模型版本。

- **寻求不受限制的内容创作**：针对 **Llama 3** 等不同模型的**内容限制**程度进行了讨论，并建议修改系统提示词（system prompt）以减少审查。

- **Phi-3 引发关注**：成员们正在评估 **Phi-3**，注意到尽管其体积比大型模型小，但在某些任务上表现出色。人们对 **Phi-3** 与 **LM Studio** 的兼容性和性能充满期待。

- **技术故障排除与版本查询**：用户寻求关于 **LM Studio** 处理 **Meta-Llama-3-8B-Instruct-Q4_K_M.gguf** 等模型能力的帮助和澄清，讨论了**上下文窗口大小**（context size）对模型性能的影响，以及 **OpenAI** 的 **GPT-4** 如何设定了高标准的对比基准。此外，还提到了在 **headless server** 上运行 **LM Studio**，并解释了 "mog" 等术语的含义。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/AI-Engine/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q5_k_m_with_temp_stop_token_fix.gguf?download=true">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://tide-freckle-52b.notion.site/1e0168e3481747ebaa365f77a3af3cc1?v=83e3d58d1c3c45ad879834981b8c2530">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合在一起的新工具。为您和您的团队打造的一体化工作空间。</li><li><a href="https://tenor.com/view/yoda-star-wars-learning-gif-21964563">Yoda Star GIF - Yoda Star Wars - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://doc.pypy.org/en/latest/sandbox.html">PyPy 的沙箱功能 &mdash; PyPy 文档</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf">microsoft/Phi-3-mini-4k-instruct-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/models?other=base_model:meta-llama/Meta-Llama-3-8B-Instruct">Models - Hugging Face</a>：未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/llama3.preset.json">configs/llama3.preset.json at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://x.com/hrishioa/status/1782429651962675410">Hrishi (@hrishioa) 的推文</a>：有人在微调 llama3-42b 的 instruct 版本吗？如果它能作为一个优秀/智能/客户端侧的 GPT-4 替代品，那将非常有趣 https://www.reddit.com/r/LocalLLaMA/comments/1c9u2jd/...</li><li><a href="https://huggingface.co/chargoddard/llama3-42b-v0">chargoddard/llama3-42b-v0 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: 计算机的自然语言界面</a>：计算机的自然语言界面。通过在 GitHub 上创建账号为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: C/C++ 实现的 LLM 推理</a>：C/C++ 实现的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: llama.cpp 的 Python 绑定</a>：llama.cpp 的 Python 绑定。通过在 GitHub 上创建账号为 abetlen/llama-cpp-python 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1231950083847880788)** (1 条消息): 

- **Hugging Face 宕机影响 LM Studio**：由于 [Hugging Face 宕机](https://x.com/lmstudioai/status/1782390856986550384?s=46)，LM Studio 的模型搜索和下载功能目前可能受损。团队正在监控情况，并承诺会及时提供更新。

**提到的链接**: <a href="https://x.com/lmstudioai/status/1782390856986550384?s=46">来自 LM Studio (@LMStudioAI) 的推文</a>: LM Studio 内的模型搜索/下载可能会受到此次 Hugging Face 停机的影响。请关注更新 ↘️ 引用 Hugging Face Status (@hf_status) 我们正在经历一些停机...

  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1231175264730615889)** (27 条消息🔥): 

- **Llama3 遇到加载问题**: 多位用户报告在 0.2.20 更新后，使用 **Llama3** 加载模型时出现问题，建议在特定频道发布详细问题。错误日志显示通用的 "Error loading model" 且没有建议，暗示最近的更新可能存在潜在 Bug。
- **对 LM Studio 的感谢**: 一位专业作家和 AI 研究员对 **LM Studio** 表达了深深的谢意，称其显著提高了他们的工作效率。这种由衷的反馈强调了 LM Studio 对用户工作流的影响。
- **注意到模型行为异常**: 一位用户观察到 *llama* 模型在被问及一般话题时，有时会输出数字而不是答案。这种异常行为表明模型响应中可能存在故障。
- **VPN 导致 LM Studio 证书问题**: 使用 **Zscaler VPN** 的用户由于 "unable to get local issuer certificate" 错误而无法在 **LM Studio** 中下载模型。提到的解决方法包括在另一台机器上下载模型，但底层机制尚不清楚，因为退出 VPN 即可解决该问题。
- **在 LM Studio 中查询 Hugging Face 模型触发错误**: 在 **LM Studio** 上搜索特别受欢迎的模型时会出现 500 错误。用户推测 Hugging Face 可能因为流量过大而屏蔽了 "Llama" 或 "Llama3" 等关键词，而使用 "lmstudio-community" 进行替代搜索则运行正常。
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1231148383419502682)** (12 条消息🔥): 

- **寻求完整的代码输出**: 一位用户询问如何让 LLM 始终编写完整代码，而不是插入诸如 *// Add similar event listeners for left and right buttons* 之类的注释。
- **探索无尽冒险**: 有人询问使用 **Llama3** 创建无尽沙盒冒险模拟游戏的最佳 Prompt，并思考 Llama3 是否可以为自己生成 Prompt。
- **配置 Llama-3-Smaug-8B Prompt**: 一位成员寻求在 LM Studio 中为 [Llama-3-Smaug-8B 模型](https://huggingface.co/bartowski/Llama-3-Smaug-8B#prompt-format) 配置 Prompt 的帮助，并想知道系统和用户前缀及后缀的正确用法，因为他们的尝试导致了无休止的输出。
- **Prompt 配置澄清**: 另一位用户澄清说，为该模型配置 Prompt 与 LM Studio v0.2.20 中常规包含的 Llama 3 预设相同。
- **LM Studio 更新和模型搜索问题**: 在讨论了最新的 LM Studio 版本后，有报告称搜索模型时出现 503 错误代码，一位回复者引用了一个 Discord 频道链接以寻求进一步帮助，但该链接显示为 'null'。

**提到的链接**: <a href="https://huggingface.co/bartowski/Llama-3-Smaug-8B-GGUF#prompt-format.">bartowski/Llama-3-Smaug-8B-GGUF · Hugging Face</a>: 未找到描述

  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1231145707180658769)** (59 条消息🔥🔥):

- **寻找合适的 GPU**：频道中的用户讨论了升级笔记本电脑以使用 NVIDIA GPU 运行 LLM。分享了一个来自 Reddit 的指南，标题为 [The LLM GPU Buying Guide - August 2023](https://www.reddit.com/r/LocalLLaMA/comments/15rwe7t/the_llm_gpu_buying_guide_august_2023/)，但有人指出，笔记本电脑升级 GPU 并不常见，某些机器可能需要外部解决方案。
- **排除模型加载错误**：一位用户遇到了“Error loading model”问题，即无法检测到 GPU 类型，建议在设置面板中关闭 GPU offloading，随后问题得到解决。
- **优化模型使用的硬件**：讨论了使用 GTX 1060 等次级 GPU 运行大型模型的功耗和效率，共识是值得测试，但由于潜在的延迟和功耗，要保持较低的预期。
- **撰写研究论文的模型偏好**：用户询问撰写研究论文的最佳模型，提到了 Llama 3 8B 和 Claude 3，前者因回答太像 AI 而受到批评，后者对免费用户有限制。
- **运行 LLM 的 Mac 内存潜力**：关于新款 128 GB Mac 运行 Grok 等大型模型能力的提问引发了讨论；建议为操作系统保留内存，并提供了一个链接，介绍如何在 macOS 上使用 `sudo` 命令增加 VRAM 分配。此外，暗示拥有 192 GB RAM 的 Mac Ultra 2 可以很好地运行 120b 模型。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/15rwe7t/the_llm_gpu_buying_guide_august_2023/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1231136152832970753)** (10 messages🔥): 

- **LMStudio 的本地模型检测故障**：一位成员报告了 **LMStudio** 无法检测包含 NFS 挂载的模型目录中本地保存文件的问题。尽管在 0.2.19 beta B 版本中正常工作，但在 0.2.19 beta C、0.2.19 和 0.2.20 版本中出现了该问题。

- **文件系统层级麻烦**：另一位成员讨论了 LMStudio 可能的**目录结构要求**，建议在典型的 maintainer/model 层级之上的额外目录级别可能会导致问题，而不是 NFS 因素。原帖确认使用了额外的目录级别来区分本地和外部存储。

- **目录测试建议**：建议通过在本地文件系统上进行测试来确认目录结构是否是问题的潜在原因，确保新子目录中的模型能被 LMStudio 应用发现并识别。

- **澄清 Token 误解**：在 Tokenization 的背景下，成员们讨论了模型中的 Token 并不一定与音节对齐，而是可以包含各种子词组件，如词根、前缀和后缀。探讨了语言模型在理解单词和 Token 方面的复杂性。

- **语言 Token 量化**：一位成员询问了关于语言模型训练期间使用的 Token 数量的惯例，思考 50,000 个 Token 是否由于传统、功效或复杂性与模型性能之间的平衡而成为标准数字。
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1231343022675984514)** (20 messages🔥): 

- **Autogen 与本地 LM Llama 3 的问题**：用户在使用 Autogen 指向本地 LM Llama 3 时遇到问题，它仅处理 2 个 Token 就停止了。有人表示沮丧，因为 LM 似乎在运行，但过早地返回了数据。

- **禁止营销区域**：一位成员被提醒该服务器不允许营销工具，并被要求今后避免此类活动。

- **Token 限制的潜在修复**：一位用户遇到了类似问题，并建议*将 "max tokens" 替换为 3000*，这似乎为他们解决了问题。他们还建议之后重启 Autogen，创建一个新的 Agent 和一个新的 workflow。

- **Autogen 中的 User Proxy 异常**：还有报告称 User Proxy 偶尔会突然停止输出，或者重复像 *"good job you did it"* 这样的短语，这降低了用户体验，特别是与直接使用 API 相比。

- **AutoGen Manager Agent 的问题**：另一位用户询问了在让 AutoGen Manager Agent 与本地模型配合工作时遇到的困难，具体是遇到了 *"unable to select speaker error"*。在提供的消息中没有建议的解决方案。
  

---


**LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/1231516145694150656)** (1 条消息): 

- **关于项目集成的咨询**：一名成员询问是否有办法将某种工具与 **LM Studio** 集成，并表示如果有的话，有兴趣访问特定的 **LM Studio 项目信息**。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1231155214757924864)** (42 条消息🔥): 

- **Meta Llama 3 LLM 引起用户兴奋**：分享了 **Meta Llama 3** 系列语言模型，该系列模型具有对话优化、实用性和安全性。正如其 [Hugging Face 仓库详情](https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF) 中所述，用户正在 LM Studio 中成功使用这些模型。

- **AMD 硬件上的性能讨论**：成员指出 **Meta-Llama-3-70B** 和 **Meta-Llama-3-8B** 模型在 **7900xtx** 等 AMD GPU 上的 Token 生成速度分别约为 20 tok/s 和 60 tok/s。用户对未来版本是否能在低端硬件上运行感到好奇。

- **ROCm 利用率查询**：一位用户指出，在带有 ROCm 技术预览版的双 7900XTX 配置上推理大型模型时，GPU 利用率不规则。组合后的 GPU 使用率并未反映出一张显卡的充分利用。

- **LM Studio ROCm 预览版的问题与修复**：用户报告了不同版本的 LM Studio ROCm 预览版中 GPU offloading 的 Bug。一位用户提到通过删除某些环境变量解决了他们的问题，而另一位用户由于硬件不支持而切换到了常规的 LM Studio 版本。

- **LM Studio GPU 选择困难及解决方案**：用户讨论了引导 LM Studio 使用独立 AMD GPU 而非集成显卡的挑战。建议的解决方案包括在 BIOS 中禁用集成显卡，以及手动设置 `HIP_VISIBLE_DEVICES` 等环境变量。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF">NousResearch/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.howtogeek.com/disable-integrated-graphics-on-windows/">如何在 Windows 11 上禁用集成显卡</a>: 当游戏和其他图形密集型应用程序开始卡顿时，这就是你要做的！</li><li><a href="https://techteamgb.co.uk/2024/03/22/how-to-turn-your-amd-gpu-into-a-local-llm-beast-a-beginners-guide-with-rocm/">如何将你的 AMD GPU 变成本地 LLM 怪兽：ROCm 初学者指南 | TechteamGB</a>: 未找到描述</li><li><a href="https://youtu.be/VXHryjPu52k?t=249">如何将你的 AMD GPU 变成本地 LLM 怪兽：ROCm 初学者指南</a>: 亚马逊上的 RX 7600 XT (联盟链接): https://locally.link/kEJGLM Studio: https://lmstudio.ai/rocm 由 Gigabyte 提供的产品。我们这些使用 NVIDIA GPU 的人，部分...
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1231141833241788416)** (34 条消息🔥):

- **X11 Forwarding 作为 GUI 解决方案**：成员们讨论了使用 `ssh -X` 命令的 X forwarding，作为[通过 SSH 使用 Nsight Compute GUI](https://goteleport.com/blog/x11-forwarding/)的一种方式。一位用户成功设置了 GUI，并为其他人提供了一份[分步指南](https://tspeterkim.github.io/posts/nsight-setup-on-ec2)，介绍如何使用 Nsight Compute 来分析远程 GPU。
- **通过 'Effort' 增强 LLM 推理**：新的 'Effort' 算法允许在 LLM 推理过程中动态调整计算量。该项目的详细信息及[源代码已在 GitHub 上发布](https://github.com/kolinko/effort)。讨论表明，人们有兴趣在 Triton 或 CUDA 等其他环境中实现该算法。
- **DGX 服务器预装 NVLink**：会议澄清了 DGX 服务器通常在出厂时已安装 NVLink，因为它们使用的是 SXM 插槽 GPU。相关资源解释了 [Nvidia 的 NVLink 和 NVSwitch](https://fuse.wikichip.org/news/1224/a-look-at-nvidias-nvlink-interconnect-and-the-nvswitch/)。
- **CUDA 矩阵乘法说明**：一位用户对矩阵乘法的 CUDA 代码感到困惑；另一位成员解释说，该操作是计算两个矩阵中一行和一列的点积。
- **CUDA 中的线程同步**：关于 CUDA 中 `__syncthreads()` 行为的对话指出，从 Volta 架构开始，Block 中所有未退出的线程都必须到达同步点，这与旧架构有所不同，在旧架构中 `__syncthreads()` 会忽略已退出的线程。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tspeterkim.github.io/posts/nsight-setup-on-ec2">如何在本地设置 Nsight Compute 以分析远程 GPU</a>：暂无描述</li><li><a href="https://kolinko.github.io/effort/">Effort 引擎</a>：一种可能用于 LLM 推理的新算法。可以平滑地——且实时地——调整推理过程中想要进行的计算量。</li><li><a href="https://goteleport.com/blog/x11-forwarding/">关于 X11 Forwarding 你需要了解的内容</a>：在这篇博文中，我们将深入探讨 X11 Forwarding，解释什么是 X11 以及它的底层工作原理。</li><li><a href="https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#remote-connections">3. Nsight Compute &mdash; NsightCompute 12.4 文档</a>：暂无描述</li><li><a href="https://fuse.wikichip.org/news/1224/a-look-at-nvidias-nvlink-interconnect-and-the-nvswitch/">详解 Nvidia 的 NVLink 互连和 NVSwitch</a>：详解 Nvidia 的 NVLink 互连和拥有 20 亿晶体管、为 Nvidia 最新的 DGX-2 深度学习机提供动力的 NVSwitch。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1231376189097119844)** (46 条消息🔥): 

- **揭秘灰度转换的奇特问题**：一位成员在调整图像大小但未改变其维度后，使用 Triton 进行灰度化处理时遇到问题，导致图像异常。他们在 [GitHub Gist](https://gist.github.com/alexandremuzio/3ba9d8669f57718139da36158180baaf) 分享了用于复现的代码，并参考了原始教程 [Jupyter Notebook](https://github.com/cuda-mode/lectures/blob/main/lecture%2014/A_Practitioners_Guide_to_Triton.ipynb)。

- **解决 Triton Kernel 的内存碎片问题**：经过调试，确定大尺寸 Tensor 会导致内存变得不连续，从而破坏 Kernel 中的指针运算；建议使用工具函数 `check_tensors_gpu_ready` 来确保数据就绪。

- **在 Triton 中实现二分查找的规划**：有人指出 Triton 在执行二分查找或索引静态代码簿（static codebook）方面存在能力缺失，而这种能力对于移植某些算法示例和量化工作至关重要，详见 [Triton 的 GitHub Issue](https://github.com/openai/triton/issues/974#issuecomment-1345372027)。

- **应对 Triton 的索引和量化挑战**：对话中交流了在 Triton 中实现二分查找和处理量化 Kernel 的想法，考虑了局限性并讨论了使用 Triton 原语（如 `tl.reduce` 或 `tl.scan`）的可能变通方法。

- **破解 `make_block_ptr` 参数迷局**：关于 Triton 的 `tl.make_block_ptr` 函数中 `order` 参数的讨论区分了行优先（row-major）和列优先（column-major）数据格式，其中 `order=(1,0)` 表示行优先，即内轴是连续的，而 `order(0,1)` 表示列优先。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/Jokeren/triton-samples/blob/main/binary_search.py">triton-samples/binary_search.py at main · Jokeren/triton-samples</a>: 通过在 GitHub 上创建账户来为 Jokeren/triton-samples 的开发做出贡献。</li><li><a href="https://github.com/openai/triton/issues/974#issuecomment-1345372027">Index in triton · Issue #974 · openai/triton</a>: 我们想在 Triton kernel 中进行一些索引操作，假设我们有 x_ptr, idx_ptr, out_ptr，x = tl.load(x_ptr + offsets, mask = mask)，idx = tl.load(idx_ptr + offsets, mask = mask)，我们有：1. idx = idx.t...</li><li><a href="https://triton-lang.org/main/python-api/generated/triton.language.make_block_ptr.html#triton.language.make_block_ptr">triton.language.make_block_ptr &mdash; Triton  documentation</a>: 未找到描述</li><li><a href="https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py#L125">triton/python/tutorials/06-fused-attention.py at main · openai/triton</a>: Triton 语言和编译器的开发仓库 - openai/triton</li><li><a href="https://github.com/thu-ml/low-bit-optimizers/blob/main/lpmm/cpp_extension/fused_adamw_kernel.cu#L27">low-bit-optimizers/lpmm/cpp_extension/fused_adamw_kernel.cu at main · thu-ml/low-bit-optimizers</a>: PyTorch 的低比特优化器。通过在 GitHub 上创建账户来为 thu-ml/low-bit-optimizers 的开发做出贡献。</li><li><a href="https://gist.github.com/alexandremuzio/3ba9d8669f57718139da36158180baaf">Weird triton kernel behavior for gray scale. (Meant to be copy pasted in a colab with a T4 gpu)</a>: 灰度图下奇怪的 Triton kernel 行为。（旨在复制粘贴到带有 T4 GPU 的 Colab 中）- weird_triton_repro.py</li><li><a href="https://github.com/cuda-mode/lectures/blob/main/lecture%2014/A_Practitioners_Guide_to_Triton.ipynb">lectures/lecture 14/A_Practitioners_Guide_to_Triton.ipynb at main · cuda-mode/lectures</a>: CUDA-MODE 讲座材料。通过在 GitHub 上创建账户来为 cuda-mode/lectures 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1231346697506918532)** (8 messages🔥): 

- **对概念基础的感谢**：一位成员对一场阐述了 "layout algebra" **概念基础**的演讲表示感谢，认为它揭示了该主题的“本质”。

- **强制内联查询**：讨论了 [__forceinline 和 __inline](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-qualifiers)，成员们解释说它们指示编译器将函数的源代码嵌入到调用者上下文中，以潜在地提高执行速度。

- **Nsight System CLI 故障排除**：一位成员解决了 Windows 上 Nsight Systems 关于核心数冲突的 profiling 问题，指出从 2024.2.1 **回退到 2023.4.4 版本**修复了该问题。

- **性能测量脚本请求**：有人请求一个用于测量不同 thread 和 block 配置下执行时间的脚本，但提供的消息中未给出解决方案或链接。

- **内联与代码优化**：讨论强调使用 **__forceinline** 可以为编译器提供更多优化机会，类似于 memory coalescing 通过减少对独立函数调用的需求来提高性能。
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1232346270304174110)** (2 messages): 

- **理解神经网络操作中的 GPU 利用率**：有人提出了一个问题，即 `torch.nn.conv2d`、`torch.nn.relu` 和 `torch.nn.batchnorm` 等操作是否会导致数据在每个操作之间在 CPU 和 GPU 之间传输。澄清指出，当 GPU tensor 传递给一系列函数时，**所有操作都在 GPU 上执行**，中间结果不会拷贝回 host memory。
- **GPU 上的异步执行**：解释了 GPU 上的操作是**异步**调度的，这意味着 Python 指令在计算完成之前就会返回。需要读取值的阻塞或同步操作（如 `.cpu()`）将导致与 CPU 的同步。
  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1231317965308170340)** (1 messages): 

- **关于 CUTLASS 的第 15 讲**：CUDA-MODE 的**第 15 讲**正在开始，重点是 **CUTLASS**。指定演讲者的演示即将开始。
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

andreaskoepf: https://x.com/AliHassaniJr/status/1766108184630943832
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1231145398245265478)** (27 messages🔥):

- **CUDA 课程进行中及即将到来的日程安排**：CUDA MODE 第 2 讲已在 *general 频道*开始；感兴趣的成员可以加入，另一场针对 NAM 时区的课程安排在周日。详细信息和规划在专门的邀请频道中进行，链接分享为 [CUDA MODE Lecture Planning](https://discord.gg/H9h8vKNu)。
- **讲师引人入胜的风格吸引了听众**：成员们被讲师有趣且引人入胜的风格所吸引，有人引用说作者是“一个非常有趣且幽默的家伙”。
- **CUDA 中的矩阵乘法探索**：一位成员请求澄清一个矩阵乘法函数，引发了讨论并分享了代码示例，例如用于快速矩阵乘法的 Python Numba 实现。
- **利用 CUDA 实现图像和视频处理**：关于使用 CUDA 进行潜在项目的对话包括将图像处理示例扩展到处理视频处理并添加更多功能。
- **机器学习任务的硬件选择讨论**：目前正在讨论机器学习系统的硬件选择，比较了 2x2070 双 GPU 配置和单块 4090 GPU 的优劣。一位成员建议，尽管有成本方面的考虑，但为了安装简便，4090 是首选。

**提到的链接**：<a href="https://discord.gg/H9h8vKNu">Discord - A New Way to Chat with Friends &amp; Communities</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、闲逛，并与你的朋友和社区保持紧密联系。

---

**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1231262495864389793)** (2 条消息): 

- **协作练习验证**：一位成员提出为尝试过练习的人验证练习答案；验证的前提是成员必须先尝试练习并通过私信提交照片。这里有不同章节的资源，包括 [Ch 2](https://docs.google.com/document/d/10ez800eu8OF-OzJXNZ0tRGdJaRAwagiyFdgeBoX0S8o/edit)、[Ch 3](https://docs.google.com/document/d/1wILXD7Pq8dsvEJpt-YwVekdFxYJvjqu6qnpzYR-LbhE/edit?usp=sharing)、[Ch 4](https://docs.google.com/document/d/1b29UvSN2-S8D_UP1xvtSB7nFRc86s6AdWH7n5UieDfE/edit?usp=sharing) 以及重点标注的 [Ch 5](https://docs.google.com/document/d/12_d0PFd3H5o68drT1pv_RuSYo67Evm9X7V70RMplrVk/edit?usp=sharing)。

- **CUDA Kernel 循环执行疑问**：一位成员正在寻求澄清，为什么作者建议一个简单的 reduction CUDA kernel 循环在输入大小为 256 且 block size 为 128 的情况下会执行 7 次，而他们自己的计算表明该循环应该执行 8 次。他们提供了代码截图和作者的说法作为参考。

---

**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 条消息): 

.bexboy: 我想这一节课也会被上传吧？

---

**CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1231185461091893269)** (1 条消息): 

- **JAX 中 DenseFormer 的内存困扰**：由于高内存占用，一位成员在 JAX 中实现 *DenseFormer* 时面临挑战。他们引用了 DenseFormer 的 [GitHub repository](https://github.com/epfml/DenseFormer)，并描述了其在 PyTorch 中高效的 **in-place tensor mutation**，同时指出 JAX/XLA 的函数式方法在优化掉副本方面表现不佳，导致了内存问题。

- **探索 Write-Once Buffers**：受 [Equinox library](https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/common.py) 的启发，该成员成功地为相对于输入的梯度创建了一个 *write-once buffer*，但在计算相对于 DenseFormer 块权重的梯度时遇到了内存二次增长的问题。

- **考虑使用 Custom Gradients 以减少内存占用**：为了克服 **quadratic memory usage** 的障碍，用户正在考虑为整个 loop/scan 函数编写自定义反向传播（custom backward pass），这是一个复杂的解决方案，旨在 JAX 的函数式范式中复制 PyTorch 的高效原位更新。他们欢迎关于解决此问题的高层建议。

**提到的链接**：<a href="https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/common.py">equinox/equinox/internal/_loop/common.py at main · patrick-kidger/equinox</a>：JAX 中优雅易用的神经网络 + 科学计算。https://docs.kidger.site/equinox/ - patrick-kidger/equinox

---

**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1232158952041480202)** (3 条消息):

- **Ring Attention 模型训练咨询**：针对有关使用 **Ring Attention** 实现训练的问题，另一位成员分享了 Axolotl 仓库的 **GitHub 链接**，相关代码正在该仓库中开发。他们提到手动放置（manual placement）已生效，并且在 tinyllama 上测试成功。
  - [在 GitHub 上查看 Axolotl ring attention 补丁](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching)

**提到的链接**：<a href="https://github.com/cuda-mode/axolotl/tree/ring_attention_patching">GitHub - cuda-mode/axolotl at ring_attention_patching</a>：欢迎在 axolotl 提问。通过在 GitHub 上创建账号来为 cuda-mode/axolotl 的开发做出贡献。

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1231258187043438642)** (4 条消息): 

- **明斯特（Münster）的地域惊喜**：CUDA MODE Discord 的成员们惊讶地发现，包括 **@umerha** 在内的三名成员都住在明斯特地区附近，感叹 GPU 社区的圈子真小。
- **愉快的见面经历**：**@umerha** 和 **@t-vi** 分享了他们在明斯特见面的愉快经历，称这次访问是“一种荣幸和快乐”。
- **德国 GPU 之都汇聚 CUDA 爱好者**：**@umerha** 提到了去明斯特的“朝圣”之旅，幽默地称其为德国的 GPU 之都，并享受了与成员 **@761222713611386900** 和 **@719599526448463933** 相聚的时光。
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1231800666603782176)** (15 条消息🔥): 

- **发布极具前景的 Triton Kernel 基准测试**：引入了一个新的融合 Triton `int4 / fp16` kernel，在各种计算形状（compute shapes）下均表现出性能提升，并提供了**详细的基准测试结果**。基准测试表明该 kernel 需要 **[Triton >= 3.0.0](https://github.com/pytorch-labs/ao/pull/153)**，并包含了与参考的 `hqq.linear` 以及 Torch 的 `int4_mm` kernel 的对比。

- **通过转置提升反向传播效率**：讨论集中在训练过程中**量化（quantization）**的反向传播（backward pass）需要对量化权重矩阵进行转置。前向传播使用 `torch.matmul(x, dequantize().t())`，而反向传播需要 `torch.matmul(grad_output, dequantize())`，这些差异在 [HQQ GitHub 仓库](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L253-L283)中有所强调。

- **量化与性能考量**：成员们讨论了使用**反量化（dequantization）**时的性能下降问题，指出典型的 CUDA 反量化 kernel 加上 torch.matmul 比使用 **fp16 或 bfp16** 的纯 torch.matmul 慢约 15%。

- **扩展 Triton Kernel 以支持 `axis=0`**：有成员请求扩展新 Triton kernel 的功能，以处理沿 `axis=0` 的计算，从而提高**量化质量**。相关的 Triton 代码已分享供参考，详见[此处](https://github.com/mobiusml/hqq/blob/triton/hqq/kernels/triton/dequant.py#L21-L50)。

- **Triton 转置实现完成**：Triton kernel 现在包含了针对转置权重矩阵的实现，以满足更高效反向传播的需求。更新后的测试和实现已发布在 GitHub 的 [Pull Request](https://github.com/pytorch-labs/ao/pull/153/files#diff-240c1eaceacda5c5054dbaef20f835373e25882e314aa800868c32093faf8eca) 中。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/triton/hqq/kernels/triton/dequant.py#L21-L50">hqq/hqq/kernels/triton/dequant.py at triton · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L253-L283">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch-labs/ao/pull/153/files#diff-240c1eaceacda5c5054dbaef20f835373e25882e314aa800868c32093faf8eca">Fused HQQ Quantization Gemm by jeromeku · Pull Request #153 · pytorch-labs/ao</a>：@msaroufim 融合的 int4 / fp16 量化 Matmul 融合 kernel，结合了非对称反量化和 gemm：反量化：将 u4 / s4 权重上采样至 float16 / bfloat16，随后进行逐组缩放...</li><li><a href="https://github.com/pytorch-labs/ao/pull/153">Fused HQQ Quantization Gemm by jeromeku · Pull Request #153 · pytorch-labs/ao</a>：@msaroufim 融合的 int4 / fp16 量化 Matmul 融合 kernel，结合了非对称反量化和 gemm：反量化：将 u4 / s4 权重上采样至 float16 / bfloat16，随后进行逐组缩放...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1231157413395169300)** (600 条消息🔥🔥🔥):

- **CUDA 中的原子操作（Atomics）与性能瓶颈**：讨论集中在从 CUDA kernel 中移除原子操作，作为性能优化工作的一部分。尽管对于当索引变化范围很大时如何并行化更新存在疑虑，但建议包括使用 scratch memory 和多次 kernel 调用，或者在 CPU 上进行预处理以对索引进行排序。还讨论了原子操作引起的竞争以及处理大部分重复索引的问题。

- **BF16/FP16 混合精度实现**：围绕 BF16/FP16 混合精度训练实现的深入讨论显示，性能提升了约 **1.86 倍**。虽然简要提到了针对 FP8 等更低精度的优化工作，但 PR (#218) 引入了随机舍入（stochastic rounding）和管理需要 BF16/FP16 的 optimizer state 的复杂性。目前 layernorm 的最新实现仍保持 FP32，因为 BF16 原子操作的性能较慢。

- **FP16 转换中的 CUDA 版本要求**：由于其中一台设备上的 CUDA 版本过旧，导致了编译错误，这突显了 BF16 支持对较新 CUDA 版本的依赖。还注意到 cuBLAS 不接受 FP8 bias 用于 FP8 matmul，而是需要 BF16 bias 的问题。

- **Kernel 优化与分析（Profiling）**：一些社区成员分享了使用 dtype sizing 和 float4 向量等技术优化 CUDA kernel 的见解和进展，这可能使 GELU 和 AdamW kernel 的速度提升 2 倍。有人建议更新 kernel 开发脚本以反映真实世界的尺寸，从而获得更好的 profiling 准确性。

- **通过线程粗化（Thread Coarsening）优化受内存限制的 Kernel**：在一次社区协作会议中，线程粗化被应用于 AdamW kernel，以提高其性能，因为该 kernel 受内存带宽限制（memory bound）。这种优化对内存请求进行了批处理以提高并行度，旨在为未来的增强做准备，特别是在向 FP16 过渡之后。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.twitch.tv/zhizhinpeter">zhizhinpeter - Twitch</a>: 为 llm.c 编写 Multi-GPU 代码</li><li><a href="https://arxiv.org/abs/2110.02861">8-bit Optimizers via Block-wise Quantization</a>: 有状态优化器会随时间维护梯度统计信息，例如过去梯度值的指数平滑和（带动量的 SGD）或平方和（Adam）。这种状态可用于加速...</li><li><a href="https://www.youtube.com/">YouTube</a>: 未找到描述</li><li><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-1-single-process-single-thread-multiple-devices">Examples &mdash; NCCL 2.21.5 documentation</a>: 未找到描述</li><li><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-1-single-process-sin">Examples &mdash; NCCL 2.21.5 documentation</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/blob/master/dev/cuda/encoder_backward.cu">llm.c/dev/cuda/encoder_backward.cu at master · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/nshepperd/flash_attn_jax/tree/main/csrc/flash_attn/src">flash_attn_jax/csrc/flash_attn/src at main · nshepperd/flash_attn_jax</a>: Flash Attention v2 的 JAX 绑定。通过在 GitHub 上创建账号来为 nshepperd/flash_attn_jax 的开发做出贡献。</li><li><a href="https://github.com/KernelTuner/kernel_float">GitHub - KernelTuner/kernel_float: CUDA header-only library for working with vector types (half2, float4, double2) and reduced precision math (half, e5m2)  inside kernel code</a>: 用于在 kernel 代码中处理向量类型（half2, float4, double2）和降低精度数学运算（half, e5m2）的 CUDA 仅头文件库 - KernelTuner/kernel_float</li><li><a href="https://clang.llvm.org/doxygen/____clang__cuda__intrinsics_8h_source.html">clang: lib/Headers/__clang_cuda_intrinsics.h Source File</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/218">Support for FP16/BF16 in train_gpt2.cu (1.86x Perf) by ademeure · Pull Request #218 · karpathy/llm.c</a>: 现在已完成，效果相当满意！在我的 RTX 4090 上实现了 1.86 倍的性能：FP32: ~80ms BF16: ~43ms（layernorm 参数使用 FP32，但所有 activations 使用 BF16）。这使得同样的 train_gpt2.c...</li><li><a href="https://www.youtube.com/watch?v=0zE6c52yomU">How to go from 0 to speeding up LLM.c - CUDA Kernel Profiling setup</a>: 实例设置完成后运行的命令：git clone https://github.com/karpathy/llm.c.git export PATH=/usr/local/cuda/bin:$PATH source ~/.bashrc sudo apt...</li><li><a href="https://github.com/karpathy/llm.c/pull/210">Added shared memory for the atomic additions for the layernorm_back by ChrisDryden · Pull Request #210 · karpathy/llm.c</a>: 此 CR 旨在解决 profiler 中发现的问题，即该 kernel 最后循环中的原子操作导致了大量的 warp stalls。通过在 shared memory 上执行原子操作...</li><li><a href="https://github.com/karpathy/llm.c/issues/212">bug: something goes wrong at larger batch sizes · Issue #212 · karpathy/llm.c</a>: 今天遇到了一个难以追踪的 bug，我打算今晚先放弃，明天再试。复现方法：./train_gpt2cu -b 12 以 batch size 12 启动任务。在我的...</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h">flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h at main · Dao-AILab/flash-attention</a>: 快速且内存高效的精确 Attention。通过在 GitHub 上创建账号来为 Dao-AILab/flash-attention 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/221">Faster `matmul_backward_bias` using coalesced reads and shared memory in the kernel by al0vya · Pull Request #221 · karpathy/llm.c</a>: 该 kernel 在 RTX 2070 Super GPU 上相比 matmul_backward_bias_kernel2 似乎提供了接近 4 倍的运行时间改进，运行时间对比见下文：matmul_backward_bias_kernel2: block_size 32 time 0.9...</li><li><a href="https://github.com/karpathy/llm.c/pull/215">cuDNN Forward Attention + FP16 non-cuDNN version in /dev/cuda/ by ademeure · Pull Request #215 · karpathy/llm.c</a>: 之前的 Kernel 4: 1.74ms Kernel 4 使用 TF32: 1.70ms Kernel 5 (带 BF16 I/O 的 4): 0.91ms Kernel 6 (不带 permute 的 5，不现实): 0.76ms Kernel 10 (cuDNN BF16，带 FP32 转换): 0.33ms...</li><li><a href="https://github.com/karpathy/llm.c/commit/8488669d256c59594f486d52a8b3597da7cbfeab">speed up the backward bias kernel by 45% and speed up the full runnin… · karpathy/llm.c@8488669</a>: …将 backward bias kernel 加速 45%，并将整体运行时间缩短 1%
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[massively-parallel-crew](https://discord.com/channels/1189498204333543425/1229286073104994344/1231255000894799954)** (29 条消息🔥):

- **引入 Moderator 角色**：引入了一个名为 "Moderator" 的新角色来管理用户和内容，其权限包括禁言、踢出、封禁用户以及删除不当消息。Moderator 还可以创建和编辑活动，并管理 Stage，以维持 GPU 和大规模并行编程讨论的友好环境。

- **小组讨论录制中的技术困难**：成员们讨论了在录制小组讨论期间遇到的技术问题。对话内容包括协调在未来的讲座前会面，以确保录制设置运行良好，以及在必要时重新录制讲座的可能性。

- **备份录像挽救局面**：一位成员报告说他们的录制会话突然中断，但幸好有另一位成员的备份。他们确认，将两次录制的材料结合起来应该足以完成一个完整的会话。

- **安排未来的讲座和演练**：由于有几个活动即将举行，成员们协调在预定时间前 15 分钟做好准备，以确保技术设置到位。一位成员提到他们在其中一天无法参与录制，但提出可以在第二天处理会话录制和后期制作。

- **FlashAttention 代码深度解析的公开邀请**：在分享了一篇关于 FlashAttention 的推文后，有人提出了举办专门的深度解析活动的建议，尽管目前还没有立即的计划。此外，成员们建议联系 Tri Dao，邀请他就 Flash decoding 的工作进行潜在的讲座，并承认他之前曾就相关主题做过报告。

**提及的链接**：<a href="https://www.youtube.com/watch?v=IoMSGuiwV3g).">Flash Attention 2.0 with Tri Dao (author)! | Discord server talks</a>: ❤️ Become The AI Epiphany Patreon ❤️https://www.patreon.com/theaiepiphany👨‍👩‍👧‍👦 Join our Discord community 👨‍👩‍👧‍👦https://discord.gg/peBrCpheKEHey g...

  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1231169236890812427)** (262 messages🔥🔥): 

- **LLM 本地应用推测**：用户讨论了在智能手机上本地运行 LLM 的可行性，重点关注 Eleuther 社区可能开发一个易于使用的 App。引用了不同智能手机型号（如 Samsung S24 Ultra 和 Snapdragon）的内存带宽和 GPU 能力，表明甚至 7-8B 模型也可能具有可用性。

- **深入探讨智能手机的硬件性能**：对话深入研究了现代智能手机的硬件规格，例如 Samsung Exynos 2400 芯片组，以估算本地运行 LLM 的性能。详细分析了 6.4 Gbps 引脚速度和 51.2 GB/s 内存带宽等规格，并建议将 speculative decoding 作为提高 token 生成速率的可能方法。

- **调研现有的本地 LLM 应用**：用户探索了现有的解决方案，如 [MLC-LLM](https://github.com/mlc-ai/mlc-llm)，用于在设备上原生部署 AI 模型。他们还讨论了在 App Store 和 Play Store 上发现的其他应用，如 "MLC Chat" 和 "Private AI"，这些应用利用离线 LLM，表明目前已有一些应用在尝试这一领域。

- **Hugging Face 停机与商业模式辩论**：Hugging Face 的长时间停机引发了关于其商业模式的辩论。用户思考了其策略，将其与 GitHub 等平台进行比较，并质疑为大型 AI 模型提供免费托管的可持续性。

- **关于 LLM 在 CoT 之外的推理讨论**：讨论转向了使用各种方法评估 LLM 的推理能力，例如 Chain-of-Thought (CoT)。有人建议将 Monte Carlo Tree Search 与 LLM 结合的最新研究论文作为 CoT 推理的替代方案 ([AlphaLLM](http://arxiv.org/abs/2404.12253))。

- **LLM 训练成本分析**：讨论涉及了训练 Llama 2 等大型模型的成本，考虑了 GPU 小时数和 token 数量等因素。还强调了在没有经过彻底数学计算的情况下，成本可能会被低估。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://store.google.com/intl/en/ideas/articles/pixel-feature-drop-december-2023/">Gemini Nano now running on Pixel 8 Pro — the first smartphone with AI built in</a>: Gemini Nano 现已在 Pixel 8 Pro 上运行 —— 首款内置 AI 的智能手机。Gemini 来了，这是我们迄今为止构建的最强大、最灵活的 AI 模型。此外，Pixel 系列还将迎来更多 AI 更新。</li><li><a href="https://play.google.com/store/apps/details?id=us.valkon.privateai&hl=en&gl=US">Private AI - Apps on Google Play</a>: 未找到描述</li><li><a href="https://llm.mlc.ai/docs/deploy/android.html">Android App &mdash; mlc-llm 0.1.0 documentation</a>: 未找到描述</li><li><a href="https://nanoreview.net/en/soc/samsung-exynos-2400">Samsung Exynos 2400: specs and benchmarks</a>: Samsung Exynos 2400：基准测试性能（AnTuTu 10, GeekBench 6）。电池续航和完整规格。</li><li><a href="https://apps.apple.com/us/app/mlc-chat/id6448482937?platform=iphone">‎MLC Chat</a>: MLC Chat 让用户可以在 iPad 和 iPhone 上本地与开源语言模型聊天。模型下载到应用后，一切都在本地运行，无需服务器支持，且无需互联网即可工作...</li><li><a href="https://news.ycombinator.com/item?id=37248895">no title found</a>: 未找到描述</li><li><a href="https://github.com/mlc-ai/mlc-llm">GitHub - mlc-ai/mlc-llm: Enable everyone to develop, optimize and deploy AI models natively on everyone&#39;s devices.</a>: 让每个人都能在各自的设备上原生开发、优化和部署 AI 模型。- mlc-ai/mlc-llm</li><li><a href="https://arxiv.org/abs/2311.10207">Stella Nera: Achieving 161 TOp/s/W with Multiplier-free DNN Acceleration based on Approximate Matrix Multiplication</a>: 从经典的 HPC 到深度学习，MatMul 是当今计算的核心。最近的 Maddness 方法通过使用基于哈希的版本来近似 MatMul，而无需进行乘法...</li><li><a href="https://github.com/Kotlin/Kotlindl">GitHub - Kotlin/kotlindl: High-level Deep Learning Framework written in Kotlin and inspired by Keras</a>: 用 Kotlin 编写并受 Keras 启发的高级深度学习框架 - Kotlin/kotlindl</li><li><a href="http://arxiv.org/abs/2404.12253">Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing</a>: 尽管大语言模型（LLMs）在各种任务上表现出色，但在涉及复杂推理和规划的场景中仍然面临困难。最近的工作提出了先进的...</li><li><a href="https://developers.googleblog.com/2024/03/running-large-language-models-on-device-with-mediapipe-andtensorflow-lite.html">Large Language Models On-Device with MediaPipe and TensorFlow Lite - Google for Developers</a>: 未找到描述</li><li><a href="https://www.gsmarena.com/samsung_galaxy_s24_ultra-review-2670p4.php">Samsung Galaxy S24 Ultra review</a>: 三星 S24 系列搭载了基于 Google 最新 Android 14 的三星最新 One UI 6.1。尽管这只是一个相当小的 ".1" 编号更新，...</li><li><a href="https://support.google.com/googleplay/android-developer/answer/9878810?hl=en-GB#>">Inappropriate Content - Play Console Help</a>: 未找到描述</li><li><a href="https://github.com/atfortes/Awesome-LLM-Reasoning?tab=readme-ov-file">GitHub - atfortes/Awesome-LLM-Reasoning: Reasoning in Large Language Models: Papers and Resources, including Chain-of-Thought, Instruction-Tuning and Multimodality.</a>: 大语言模型中的推理：论文和资源，包括 Chain-of-Thought、指令微调（Instruction-Tuning）和多模态。- GitHub - atfortes/Awesome-LLM-Reasoning...</li><li><a href="https://semiconductor.samsung.com/dram/lpddr/lpddr5/">LPDDR5 | DRAM | Samsung Semiconductor Global</a>: 了解 LPDDR5，它以 6,400 Mbps 的引脚速度、51.2Gb/s 的海量传输和 20% 的节能效果，为下一代应用提供动力。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1231165794822520944)** (443 messages🔥🔥🔥): 

- **扩散模型推理步数讨论 (Diffusion Model Inference Steps Discussion)**: 在较高步数（如 300 或 1000 步）下训练的扩散模型，可以有效地用于步数显著减少（如 10-30 步）的推理。大家达成共识，在给定的推理步数下，训练步数不会极大地影响质量。
  
- **无 Token 语言模型 (Token-Free Language Models)**: [SpaceByte 论文](https://arxiv.org/abs/2404.14408v1)提出了一种新型的字节级架构，试图缩小子词（subword）与字节级自回归语言建模之间的差距。有讨论指出，分词器（tokenizer）可能会泄露有关后续 Token 的信息，这可能被视为一个重大干扰，特别是对于自动补全等应用。

- **关于 'Fineweb' 数据集与 LLaMA 关系的担忧**：虽然 [Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 提供了 15 万亿 token 的 CommonCrawl 数据并声称具有高性能，但成员们对其与 LLaMA 数据集的关系表示质疑，并对缺乏数据集去污染（decontamination）表示怀疑。Fineweb 性能的影响将在未来受到密切关注。

- **AI 设计的 CRISPR-Cas 蛋白质**：
  一个 Large Language Model —— ProGen2，被成功用于设计新的 CRISPR-Cas 蛋白质序列，随后在实验室中进行了测试，产生了具有更高特异性的变体。由 [Profluent Bio](https://www.profluent.bio/) 提供的这一突破性案例表明了 LLM 在加速科学发现方面的潜力。

- **安全 Large Language Models 的 Prompt 优先级**：
  一篇新论文建议通过训练模型根据定义的层级结构优先处理指令，来解决 LLM 中的安全漏洞。这种方法旨在提高针对 Prompt 注入和其他攻击的鲁棒性，而无需额外的偏好标签或演示。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/BlancheMinerva/status/1782437494585282965">来自 Stella Biderman (@BlancheMinerva) 的推文</a>：为 RAG 模型创建一个基准测试，其中所有问题都需要综合多个文档的信息才能回答。研究在公开数据上训练的模型在该基准上的表现，并且...</li><li><a href="http://arxiv.org/abs/2404.13208">The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions</a>：如今的 LLM 容易受到 Prompt Injections、Jailbreaks 和其他攻击的影响，这些攻击允许攻击者用恶意 Prompt 覆盖模型的原始指令。在这项工作中...</li><li><a href="https://arxiv.org/abs/2404.14313">Self-Supervised Alignment with Mutual Information: Learning to Follow Principles without Preference Labels</a>：在对语言模型（LM）进行 Prompt 时，用户通常期望模型在不同任务中遵循一套行为原则，例如在生成深刻内容的同时避免有害或...</li><li><a href="https://arxiv.org/abs/2404.14408v1">SpaceByte: Towards Deleting Tokenization from Large Language Modeling</a>：Tokenization 在大语言模型中被广泛使用，因为它能显著提高性能。然而，Tokenization 也带来了一些缺点，如性能偏差、增加的对抗性...</li><li><a href="https://arxiv.org/abs/2401.13660">MambaByte: Token-free Selective State Space Model</a>：Token-free 语言模型直接从原始字节（Raw Bytes）中学习，并消除了 Subword Tokenization 的归纳偏置。然而，在字节上运行会导致序列显著变长。在这种设置下...</li><li><a href="http://arxiv.org/abs/2404.13686">Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis</a>：最近，出现了一系列 Diffusion 感知的蒸馏算法，以减轻与 Diffusion Models (DMs) 多步推理过程相关的计算开销。目前的蒸馏...</li><li><a href="http://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>：基于 Transformer 的语言模型在输入序列上均匀分配 FLOPs。在这项工作中，我们证明了 Transformer 可以学会动态地将 FLOPs（或计算量）分配给特定的...</li><li><a href="https://huggingface.co/stabilityai/stablelm-3b-4e1t">stabilityai/stablelm-3b-4e1t · Hugging Face</a>：未找到描述</li><li><a href="http://arxiv.org/abs/2401.06104">Transformers are Multi-State RNNs</a>：Transformer 被认为在概念上与上一代最先进的 NLP 模型——循环神经网络（RNNs）不同。在这项工作中，我们证明了 Decoder-only...</li><li><a href="https://www.profluent.bio/">Profluent</a>：我们精通蛋白质设计的语言。</li><li><a href="https://arxiv.org/abs/2402.06925">A Thorough Examination of Decoding Methods in the Era of LLMs</a>：解码方法在将语言模型从 Next-token Predictors 转换为实用的任务求解器方面发挥着不可或缺的作用。先前关于解码方法的研究主要集中在特定任务...</li><li><a href="https://arxiv.org/abs/2401.03462">Soaring from 4K to 400K: Extending LLM&#39;s Context with Activation Beacon</a>：由于 Context Window 大小有限，长上下文的利用对 LLM 提出了巨大挑战。虽然可以通过微调来扩展 Context Window，但这会导致相当大的...</li><li><a href="https://arxiv.org/abs/2403.11901">Larimar: Large Language Models with Episodic Memory Control</a>：高效、准确地更新存储在 LLM 中的知识是当今最紧迫的研究挑战之一。本文介绍了 Larimar——一种新型的、受大脑启发的架构...</li><li><a href="https://github.com/microsoft/LLMLingua">GitHub - microsoft/LLMLingua: 为了加速 LLM 的推理并增强 LLM 对关键信息的感知，压缩 Prompt 和 KV-Cache，在性能损失极小的情况下实现高达 20 倍的压缩。</a>：为了加速 LLM 的推理并增强 LLM 对关键信息的感知，压缩 Prompt 和 KV-Cache，在性能损失极小的情况下实现高达 20 倍的压缩。 - GitHub...</li><li><a href="https://github.com/krafton-ai/mambaformer-icl">GitHub - krafton-ai/mambaformer-icl: MambaFormer In-context Learning 实验和实现，针对 https://arxiv.org/abs/2402.04248</a>：MambaFormer In-context Learning 实验和实现，针对 https://arxiv.org/abs/2402.04248 - krafton-ai/mambaformer-icl</li><li><a href="https://www.biorxiv.org/content/10.1101/2024.04.22.590591v1">Design of highly functional genome editors by modeling the universe of CRISPR-Cas sequences</a>：基因编辑具有解决农业、生物技术和人类健康领域根本挑战的潜力。通过对 CRISPR-Cas 序列的全集进行建模，设计高功能的基因组编辑器。</li>

源自微生物，虽然强大，但通常表现出显著的...</li><li><a href="https://arxiv.org/abs/2404.08698">Lossless Acceleration of Large Language Model via Adaptive N-gram Parallel Decoding</a>：虽然大语言模型（LLM）展示了卓越的能力，但由于自回归处理，它们受到显著的资源消耗和相当大的延迟的阻碍。在本研究中，我们...</li><li><a href="https://arxiv.org/abs/2212.04089">Editing Models with Task Arithmetic</a>：改变预训练模型的行为——例如，提高它们在下游任务上的性能或减轻预训练期间学到的偏见——是开发机器学习模型时的常见做法...</li><li><a href="https://arxiv.org/abs/2312.02783">Large Language Models on Graphs: A Comprehensive Survey</a>：大语言模型（LLM），如 GPT4 和 LLaMA，由于其强大的文本编码/解码能力和新发现的涌现能力，正在自然语言处理领域取得重大进展...</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb">HuggingFaceFW/fineweb · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/sisihae-gif-23689236">Sisihae GIF - Sisihae - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://arxiv.org/abs/2404.07647">Why do small language models underperform? Studying Language Model Saturation via the Softmax Bottleneck</a>：语言建模的最新进展包括在极大的网络挖掘文本语料库上预训练高度参数化的神经网络。在实践中，此类模型的训练和推理可能成本高昂...</li><li><a href="https://arxiv.org/abs/2310.11829">Towards Graph Foundation Models: A Survey and Beyond</a>：基础模型已成为各种人工智能应用中的关键组件，并在自然语言处理和其他几个领域展示了显著的成功...</li><li><a href="https://buttondown.email/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/#eleuther-discord">[AINews] FineWeb: 15T Tokens, 12 years of CommonCrawl (deduped and filtered, you&#x27;re welcome)</a>：2024/4/19-2024/4/22 的 AI 新闻。我们为您检查了 6 个 subreddits、364 个 Twitter 和 27 个 Discord（395 个频道，14973 条消息）。预计阅读时间...</li><li><a href="https://arxiv.org/html/2402.08164v1">On Limitations of the Transformer Architecture</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1231665310709321739)** (35 条消息🔥): 

- **关于数据舍入的 Twitter 冲突**：一名成员对在 Twitter 上被屏蔽表示沮丧，原因是他批评某人在出版物中对数字进行了舍入处理，并分享了一条 [推文](https://twitter.com/tamaybes/status/1782102492479652314) 作为证据。对话围绕所使用的语气和方法展开，其他人指出该成员直接的语气可能显得粗鲁或具有对抗性。
  
- **批评性对话中语气至关重要**：其他成员加入讨论，认为原帖者的语气可能被视为具有攻击性或在“钓鱼”（trolling），这可能导致防御性反应。他们强调了在进行辩论时，特别是在试图表达批评时，保持友好和建设性语气的重要性。

- **识别出沟通中的误解**：有人指出，混乱的产生是因为该成员错误地将数据舍入归咎于复现团队，而事实上，原始的 Chinchilla 论文作者报告的就是舍入后的结果。会议还澄清了 TeX 在处理有效数字和渲染 SVG 等矢量格式方面的能力。

- **对 Chinchilla 论文及复现方法的批评**：该成员澄清了他最初的批评，指出真正的问题不在于舍入本身，而在于复现作者没有注意到残差没有以零为中心，这可能表明他们的复现过程中存在错误。这一详细反馈是批评 Chinchilla 论文复现所用方法的更大范围讨论的一部分。

- **对社交媒体互动的建设性剖析**：参与者剖析了在线交流的细微差别，并开玩笑地制作了一个友好互联网话语的模板，强调了在帖子中保持直接与加入“神经典型式修饰”（neurotypical decoration）之间所需的平衡，以避免被误解。

**提到的链接**：<a href="https://x.com/kyo_takano/status/1782100341443666282))">来自 Kyo (@kyo_takano) 的推文</a>：你确实在对原始估计值进行舍入，哈哈。试着像检查 PDF 图表一样检查 TeX 源码。具体来说，你舍入了：- E 从 exp(0.5267228) 到 1.69 - A 从 exp(6.0073404) 到 406.4 ...

  

---

**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1232409179575029770)** (2 messages): 

- **发现残差流范数呈指数级增长**：一篇来自 [LessWrong 的帖子](https://www.lesswrong.com/posts/8mizBCm3dyc432nK8/residual-stream-norms-grow-exponentially-over-the-forward) 揭示了像 GPT2-XL 这样的语言模型中，每个残差流的范数在前向传播过程中呈指数级增长。总结该论文指出，LayerNorm 使得*抵消现有特征变得困难*，从而允许*新特征通过每层增加 4.5% 来占据主导地位*。

**提到的链接**：<a href="https://www.lesswrong.com/posts/8mizBCm3dyc432nK8/residual-stream-norms-grow-exponentially-over-the-forward">Residual stream norms grow exponentially over the forward pass — LessWrong</a>：摘要：对于一系列语言模型和一系列输入提示词，每个残差流的范数在前向传播过程中呈指数级增长，伴随着……

  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1232208166264111136)** (8 messages🔥): 

- **寻求分叉的理智**：一位成员幽默地指出，研究小组更倾向于运行 **lm evaluation harness** 的私有 fork，而不是进行直接的模型比较。
- **评估时的 Token 查询**：有人提出了关于 **eval-harness** 是否会自动添加序列开始 (beginning-of-sequence) token 的问题。
- **实验 MMLU 任务实现**：一位成员提议使用 arc 提示词格式添加 **MMLU 任务实现**，旨在调查 MMLU 提示词格式对模型评分的影响。
- **任务实现中的通用化呼吁**：针对该提议，另一位成员建议理想情况下应创建一个能够为所有 **MCQA 任务** 支持“arc 风格”和“MMLU 风格”等多种风格的**通用实现**，尽管在开发出更通用的实现之前，对目前的特定实现也表示感兴趣。
- **并行指标探索**：有人询问关于并行执行 **lm-evaluation-harness** 指标的问题，并要求进一步详细说明具体需求。
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1232349058723614861)** (14 messages🔥): 

- **讨论 RWKV 与 GPT-NeoX 的集成**：开发者目前正专注于将 **RWKV (Rethinking Weighted Key-Value Memory Networks)** 集成到 GPT-NeoX 中。集成工作可以通过 [GitHub Issue #1167](https://github.com/EleutherAI/gpt-neox/issues/1167) 进行跟踪，涉及禁用 bf16、PP、TP、MoE，并添加 fp16 支持和 JIT 内核编译等任务。

- **FP16 支持正在集成**：一位开发者推送了一个新分支，其中包含 GPT-NeoX 中 RWKV 的 fp16 和 fp32 支持集成，[可在此处获取](https://github.com/SmerkyG/gpt-neox/tree/rwkv)。该集成较为简单，正等待使用 NeoX 训练器进行测试。

- **内核增强与代码迁移**：一位开发者为 RWKV 准备了新优化的内核代码，这可能允许为未来的 BPTT 使用提供全状态梯度。这种新方法和代码可在开发者的 GitHub fork 上找到，特别是 [rwkv-6-support](https://github.com/RWKV/RWKV-infctx-trainer/tree/rwkv-6-support) 分支。

- **建议 RWKV 版本编号**：由于 RWKV 集成工作的迭代性质，建议实施版本编号以识别不同的迭代，例如 "rwkv 6.0"。关于这种命名规范的最佳方法（是针对文件、类还是目录）仍在考虑中。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/RWKV/RWKV-infctx-trainer/tree/rwkv-6-support">GitHub - RWKV/RWKV-infctx-trainer at rwkv-6-support</a>: RWKV infctx trainer，用于训练任意上下文长度，可达 10k 甚至更高！ - GitHub - RWKV/RWKV-infctx-trainer at rwkv-6-support</li><li><a href="https://github.com/RWKV/RWKV-infctx-trainer/compare/main...rwkv-6-support">Comparing main...rwkv-6-support · RWKV/RWKV-infctx-trainer</a>: RWKV infctx trainer，用于训练任意上下文长度，可达 10k 甚至更高！ - Comparing main...rwkv-6-support · RWKV/RWKV-infctx-trainer</li><li><a href="https://github.com/EleutherAI/gpt-neox/issues/1167">Add Basic RWKV Block to GPT-NeoX · Issue #1167 · EleutherAI/gpt-neox</a>: 我们希望将 RWKV 添加到 gpt-neox：从 https://github.com/BlinkDL/RWKV-LM 向 https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model 添加基础 RWKV block（不含 kernels），添加 rwkv kernels A...</li><li><a href="https://github.com/">GitHub: Let’s build from here</a>: GitHub 是超过 1 亿开发者共同塑造软件未来的地方。在这里贡献开源社区，管理你的 Git 仓库，像专家一样评审代码，追踪 bug 和功能...</li><li><a href="https://github.com/SmerkyG/gpt-neox/tree/rwkv">GitHub - SmerkyG/gpt-neox at rwkv</a>: 基于 DeepSpeed 库在 GPU 上实现模型并行自回归 Transformer。 - GitHub - SmerkyG/gpt-neox at rwkv
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1231156435409174578)** (473 messages🔥🔥🔥): 

<ul>
  <li><strong>Hugging Face 宕机担忧</strong>：多位用户报告在尝试访问或使用 Hugging Face 时遇到 504 Gateway Time-outs 和服务中断，表明可能存在宕机或服务器问题。</li>
  <li><strong>Meta-Llama 3 集成问题</strong>：用户讨论了 Meta Llama 3 与 serverless inference API 的集成，以及在发起请求时是否支持 system prompts 等功能。</li>
  <li><strong>Autotrain 咨询</strong>：有人询问 AutoTrain 是否支持 phi-3 等自定义模型进行 fine-tuning，通过指向 Hugging Face 文档和之前的成功案例得到了解答。</li>
  <li><strong>模型上传障碍</strong>：一位用户因文件大小限制寻求将 GGUF 文件上传到 Hugging Face 的帮助，得到的建议是使用 sharding 或拆分文件以适应服务限制。</li>
  <li><strong>探索 OCR 选项</strong>：讨论集中在寻找读取浮点数的有效 OCR 解决方案，提到了 PaddleOCR 和 kerasOCR 可能是比 tesseract 和 EasyOCR 更好的替代方案。</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://colab.research.google.com/drive/1nU4xHpLQ5PIQKY0T1MK-sJ3kVXiBYnkQ?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://x.com/abhi1thakur/status/1782807785807159488?s=46">来自 abhishek (@abhi1thakur) 的推文</a>: Phi-3 来了！！！！ 🚀 当然，你已经可以使用 AutoTrain 对其进行微调了 🚀🚀🚀</li><li><a href="https://arxiv.org/abs/2309.08632">Pretraining on the Test Set Is All You Need</a>: 受近期展示了在精心策划的数据上预训练的小型 Transformer 语言模型潜力的研究启发，我们通过投入大量精力策划一个 n...</li><li><a href="https://huggingface.co/chat/assistant/66238e78096b24c9dad9457c">Llama 3-70B - HuggingChat</a>: 在 HuggingChat 中使用 Llama 3-70B 助手</li><li><a href="https://tenor.com/view/resident-evil-resident-evil-welcome-to-raccoon-city-resident-evil-movie-burning-on-fire-gif-25613395">生化危机：欢迎来到浣熊市 GIF - 生化危机电影 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/turn-down-for-what-snoop-dogg-cheers-dancing-drinking-gif-10966591">Turn Down For What Snoop Dogg GIF - Snoop Dogg 干杯 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/jinx-the-cat-jinx-jinx-cat-cat-computer-gif-25786466">小猫 Jinx GIF - 小猫 Jinx - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/im-dead-dead-bruh-skeleton-dead-bruh-skeleton-dead-im-dead-bruh-gif-26854866">我死了 Dead Bruh GIF - 骷髅 Dead Bruh - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://hf-mirror.com/">HF-Mirror - Huggingface 镜像站</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct">meta-llama/Meta-Llama-3-70B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/eyeverse-brace-initiation-eyebrow-shave-gif-6015143619791964168">Eyeverse Brace GIF - Eyeverse Brace 入会 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/docs/huggingface_hub/en/guides/upload#tips-and-tricks-for-large-uploads">将文件上传到 Hub</a>: 未找到描述</li><li><a href="https://tenor.com/view/dinela-gif-26054323">Dinela GIF - Dinela - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://colab.research.google.com/drive/1u9r-p_x7QXH9zAbQ5c0O2smEBHvC44me?usp=sharing>">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GPTQ">TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GPTQ · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/cat-club-cat-cat-dance-cat-party-cat-disco-gif-27258615">猫咪俱乐部 GIF - 猫咪跳舞 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/huggingface_hub/en/guides/upload#preupload-lfs-files-before-commit">将文件上传到 Hub</a>: 未找到描述</li><li><a href="https://hf.co/chat/assistant/6626057fa0b4434b65ed78b5">阿尔伯特·爱因斯坦 - HuggingChat</a>: 在 HuggingChat 中使用阿尔伯特·爱因斯坦助手</li><li><a href="https://rapidapi.com/swift-api-swift-api-default/api/meta-llama-3-8b">Meta Llama 3 | 8B API 文档 (swift-api-swift-api-default) | RapidAPI</a>: 未找到描述</li><li><a href="https://bpa.st/3MUQ">查看 Paste 3MUQ</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1884c8k/to">Reddit - 深入探索任何事物</a>: 未找到描述</li><li><a href="https://youtube.com/watch?v=SfKGHKzkm-o">AI 的崛起</a>: (开启字幕) 加入我们，一起回顾 Artificial Intelligence 的快速演变历程，从...</li><li><a href="https://www.youtube.com/watch?v=JOeY07qKU9c>">&quot;这是 UNIX 系统！&quot; | 侏罗纪公园 | 科幻站</a>: 黑客 Lexi (Ariana Richards) 在尝试修复侏罗纪公园的 UNIX 控制系统时展示了她的技术。侏罗纪公园 (1993)：John Hammond，一位...</li><li><a href="https://vvd.im/TicketTool">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的朋友和社区保持紧密联系。</li><li><a href="https://status.huggingface.co/">
Hugging Face 状态
</a>: 未找到描述</li>

</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - 由 mteb 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1884c8k/todays_ai_breakthrough_zero_step_diffusion/">Reddit - 深入探索任何事物</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1231786803434881046)** (13 messages🔥): 

- **研究 AI 的速度、成本和质量**：一段名为 "ORPO with LLaMa 3- Fast, Cheap, and Good!" 的视频讨论了挑战“快速、便宜、好用——三者只能选其二”这一老话的 AI 创新。该视频可以在 [YouTube](https://www.youtube.com/watch?v=oHM3faIPTg0) 上找到。
- **首个 Reinforcement Learning 模型成功**：一位成员学习了如何创建他们的第一个 Reinforcement Learning 模型，并分享了一个为训练玩 **LunarLander-v2** 的 **PPO** **Agent** 准备的 [Hugging Face 模型卡片](https://huggingface.co/wsqstar/ppo-LunarLander-v2)。
- **探索 Tokenization**：一位成员今天专注于学习关于 Tokenizers 的知识。
- **对 Hugging Face 的依赖**：一位成员评论说，即使安装了本地模型，他们仍然继续依赖 Hugging Face 的资源。
- **使用 AI Agent 创建 RAG 系统**：成员们正在学习利用 Llamaindex 构建 **RAG 系统**，并探索使用 **transformer.js** 等库在离线开源模型上进行实现。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/wsqstar/ppo-LunarLander-v2">wsqstar/ppo-LunarLander-v2 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=oHM3faIPTg0">ORPO with LLaMA 3- Fast, Cheap, and Good!</a>: 俗话说“快速、便宜、好用——三者择其二”。AI 领域此前也不例外，但我们开始看到一些伟大的创新正在改变这一点。这是一篇很棒的文章...</li><li><a href="https://www.youtube.com/watch?v=oPCKB9MUP6c&t=420s&ab_channel=DeployingAI">Build an Agent with Long-Term, Personalized Memory</a>: 本视频探讨了如何存储类似于 ChatGPT 新推出的长期记忆功能的对话记忆。我们将使用 LangGraph 构建一个简单的内存管理...</li><li><a href="https://youtu.be/q3nBKwNkRno?si=EkxSV5ZXtrSB7F6A">(RVC) I Can't Dance (AI Cover Mashup) (READ DESC)</a>: #aicover #icantdance #genesis 免责声明：这是一个简单有趣的 AI 混剪视频，是我在业余时间利用我自己和其他人的 AI 语音模型制作的...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1231254143478534255)** (21 messages🔥): 

- **探索量子计算**：分享了一段名为 [*"New quantum computers - Potential and pitfalls | DW Documentary"*](https://youtu.be/0HFzTYlhT2E) 的视频，讨论了新型超级计算机在减少动物实验和治愈癌症方面的潜力。
- **神经网络揭秘**：一位成员分享了一段名为 **"Why Neural Networks can learn (almost) anything"** 的 [YouTube 视频](https://www.youtube.com/watch?v=0QczhVg5HaI)，解释了神经网络的运作方式和用途。
- **语音提示的 AI 图像生成**：一个有趣的 [Twitter 帖子](https://twitter.com/Dan50412374/status/1781790992318042428)展示了 AI 根据口头（whisper）语音命令实时生成高分辨率图像的直播。
- **全面的 Offline RL 框架发布**：消息强调了 [Hokoff](https://sites.google.com/view/hok-offline)，这是一个为 Offline Reinforcement Learning 和 Multi-Agent Reinforcement Learning 研究提供预收集数据集和框架的资源。
- **用于 🤗 Transformers 的交互式 JavaScript**：介绍了一个允许直接在浏览器中运行 HuggingFace Transformers 的工具；可以在 [transformers.js](https://xenova.github.io/transformers.js/) 探索它。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://reasoning-tokens.ghost.io/reasoning-tokens/">Self-Reasoning Tokens，教导模型学会预判。</a>：推理的数学公式是什么？我们如何让像 chatGPT 这样的 LLM 在开口之前先思考？我们又该如何将其融入模型中，使其能够以 self-supervise 的方式学习思考...</li><li><a href="https://arxiv.org/abs/2404.13026">PhysDreamer：通过视频生成实现与 3D 对象的物理交互</a>：真实的物体交互对于创建沉浸式虚拟体验至关重要，然而，针对新型交互合成真实的 3D 物体动力学仍然是一个重大挑战。U...</li><li><a href="https://xenova.github.io/transformers.js/">Transformers.js</a>：未找到描述</li><li><a href="https://karpathy.ai/zero-to-hero.html">Neural Networks: Zero To Hero</a>：未找到描述</li><li><a href="https://sites.google.com/view/hok-offline">Hokoff</a>：摘要 </li><li><a href="https://huggingface.co/ByteDance">ByteDance (ByteDance)</a>：未找到描述</li><li><a href="https://youtu.be/0HFzTYlhT2E?si=lgzMqlFFbhVgjM7f">新型量子计算机 - 潜力与陷阱 | DW 纪录片</a>：一台新型超级计算机有望减少动物实验，甚至可能治愈癌症。围绕 quantum computing 的炒作令人振奋...</li><li><a href="https://www.youtube.com/watch?v=0QczhVg5HaI">为什么神经网络可以学习（几乎）任何东西</a>：一段关于神经网络、它们如何工作以及为什么有用的视频。我的 twitter：https://twitter.com/max_romana 来源 神经网络游乐场：https://play...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1231217503993073715)** (25 messages🔥): 

- **数学 PDF 转化为对话伙伴**：Crizomb 介绍了一个开源的 Retriever-Answer Generator (RAG) 项目 **[ai_pdf](https://github.com/Crizomb/ai_pdf)**，使用户能够在本地与任何 PDF 聊天；它通过将数学文档转换为 LaTeX 以便计算机处理，对数学文档特别有效。

- **突破性的实时视频生成**：Aifartist 分享了一个 **[Reddit 帖子](https://www.reddit.com/r/StableDiffusion/comments/1c8oea6/endlessdreams_voice_directed_realtime_videos_at/)**，展示了一个通过语音指令实时生成的 2.5 分钟视频。他们强调了快速的反馈循环以及仅通过语音命令进行实时电影创作的潜力。

- **浅显易懂地解释 Infini Attention**：Subham5089 为新型 **Infini Attention** 撰写了一份简化解释，旨在帮助理解其对 AI 的影响，并在 **[LinkedIn](https://www.linkedin.com/posts/subham-kundu-2746b515b_llms-generativeai-activity-7187373540940148736-qNG6)** 上分享了这篇文章。

- **实现创新的机器人编程**：Acoloss 分享了他们项目的一个有趣更新，该项目涉及具有独立记忆/历史的机器人根据其能力执行操作。他们指出，该实现在输出通信方面表现得异常出色。

- **3LC 的 Beta 测试发布将彻底改变数据集**：**[3LC](https://3lc.ai/)** 平台已发布，提供优化数据集和 ML 模型的工具，增强了 Computer Vision 功能，并计划扩展对 LLM 的支持。用户可以加入 Beta 测试以参与平台的开发，前 100 名用户可获得专属访问权限，且非商业用途免费。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/ehristoforu/llama-3-12b-instruct">ehristoforu/llama-3-12b-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/ehristoforu/Gixtral-100B">ehristoforu/Gixtral-100B · Hugging Face</a>: 未找到描述</li><li><a href="https://3lc.ai/">Home</a>: 未找到描述</li><li><a href="https://huggingface.co/ehristoforu/Gistral-16B">ehristoforu/Gistral-16B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/bineric/NorskGPT-Llama3-8b">bineric/NorskGPT-Llama3-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/QuantFactory/Meta-Llama-3-70B-Instruct-GGUF">QuantFactory/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3">RAG chatbot using llama3</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c8oea6/endlessdreams_voice_directed_realtime_videos_at/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/Crizomb/ai_pdf">GitHub - Crizomb/ai_pdf: Chat locally with any PDF  Ask questions, get answer with usefull references  Work well with math pdfs (convert them to LaTex, a math syntax comprehensible by computer)</a>: 在本地与任何 PDF 聊天，提出问题，获取带有有用参考的答案。对数学 PDF 效果良好（将其转换为 LaTex，一种计算机可理解的数学语法） - Crizomb/ai_pdf</li><li><a href="https://huggingface.co/spaces/clinteroni/outpainting-with-differential-diffusion-demo">Outpainting Demo - clinteroni 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/gojiteji/VTuberLogoGenerator">VTuberLogoGenerator - gojiteji 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Csplk/moondream2-batch-processing">moondream2-batch-processing - Csplk 的 Hugging Face Space</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1231195072347902092)** (4 条消息): 

- **寻求发票数据提取的架构**：一位成员正在开发一个从扫描图像（发票和收据）中提取数据的项目，并正在寻找为此任务创建机器学习模型的架构。

- **TrackNetV3 实践**：一位成员分享了 [TrackNetV3 仓库](https://github.com/qaz812345/TrackNetV3)，但正在询问如何处理每一帧读取的模型输出，而不是读取所有帧后再进行计算。

- **自我介绍**：一位名为 jackwean_75093 的用户加入并向社区打招呼。

- **个人知识库构建探索**：同一位用户 jackwean_75093 询问了如何构建私有知识库，但未提供更多细节。

**提到的链接**：<a href="https://github.com/qaz812345/TrackNetV3">GitHub - qaz812345/TrackNetV3: Implementation of paper - TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification</a>: 论文实现 - TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification - qaz812345/TrackNetV3

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1231284422171562076)** (10 条消息🔥): 

- **寻求 M2M100 微调**：一位成员正在索要 **M2M100** 模型的微调代码。
- **请求 PHI-2 调优协助**：一位成员正在寻求微调 **PHI-2** 模型的帮助。
- **微调的 Batch Size 策略**：讨论建议从较小的 Batch Size（如 **32**）开始，并向上调整以找到 16GB 显存上 2.7B 模型的最佳 Batch Size，**Gradient Accumulation** 是一个可能的解决方案。
- **minbpe 的 Rust 移植版发布**：`minbpe-rs` 项目是 `minbpe` 的 **Rust** 移植版本，已在 GitHub 上发布，具有 `GPT4Tokenizer`、`save`、`load` 和 `train` 函数等功能。该项目由 @gnp 领导，并对文档和 README 进行了贡献。[查看项目](https://github.com/gnp/minbpe-rs)。
- **依赖冲突和数据集获取困难**：一位成员提到 **Bertopic 的新版本**导致与 OpenAI 的依赖冲突，并暂时将其脚本锁定在 0.16.0 版本。同时，另一位成员寻求将 **go-emotions 数据集**集成到其项目中的帮助。

**提到的链接**：<a href="https://github.com/gnp/minbpe-rs">GitHub - gnp/minbpe-rs: Port of Andrej Karpathy&#39;s minbpe to Rust</a>: Andrej Karpathy 的 minbpe 到 Rust 的移植版本。通过在 GitHub 上创建账户为 gnp/minbpe-rs 的开发做出贡献。

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1231733039730659399)** (10 条消息🔥):

- **Android 平板运行 Fooocus 遇到困难**：一位成员询问如何在 Android 平板上使用 **fooocus**，寻求社区指导。

- **专业 Diffusers 提供服务**：一位在网页设计、MVP、应用开发以及包括 **Stable Diffusion 和 Computer Vision** 在内的多种技术领域拥有专长的成员，向初创公司和企业提供服务。

- **禁止访问模型**：一名用户在尝试使用 **vespa** 下载模型时遇到 **403 error**，并向社区寻求帮助以解决该问题。

- **加载 StoryGen 模型出错**：一位成员在使用 **DiffusionPipeline** 加载 **haoningwu/StoryGen** 模型时，因 config json 问题遇到故障，并寻求支持，特别标记了另一位用户协助。

- **关于“AI Horse”生成视频的讨论**：一名用户询问是否可以完全使用 **Diffusion** 创建一段关于“AI Horse”主题的 1 分钟视频，另一位成员建议使用 **pika** 或其他形式的 **Diffusion Transformer** 来完成此任务。
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1231200576419921920)** (77 messages🔥🔥): 

- **关于 Mojo 问题报告和通讯投稿的查询**：成员询问如何分配 issue 以及是否可以为 Mojo 通讯（newsletter）提交文章；回复指出流程涉及展示修复问题的能力，且目前尚不支持通讯投稿功能。
- **关于 GTK 辅助技术支持的讨论**：成员讨论了应用程序中良好辅助技术支持的重要性，并以 GTK 及其缺失某些功能为例。虽然对这些技术的价值存在争论，但一致认为它们有助于吸引用户。
- **Mojo 文档更新咨询**：成员询问 docs.modular.com 上的文档是否由 `mojo doc` 自动生成；回复称虽然是自动生成的，但涉及大量非公开的 CI，目前尚未设计供公众使用。
- **Mojo 与 Python 的性能对比**：成员提出的 Mojo 与 Python 在打印数字速度上的对比引发了对 Mojo 缺乏 **buffered IO** 已知问题（[known issue](https://github.com/modularml/mojo/issues/975)）的引用，并提供了性能基准测试建议，指出该问题自 12 月以来仍未解决。
- **Docs.modular.com 在 995px 宽度下的显示 Bug**：成员报告并讨论了 docs.modular.com 网站的一个 UI bug，即在特定浏览器宽度下搜索结果无法显示。与开发者的对话显示，这是发生在 995px 宽度时的已知行为，可以通过避免在该特定宽度下使用或关闭搜索来查看内容。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=40107007">未找到标题</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/issues/975):">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 开发做贡献。</li><li><a href="https://github.com/basalt-org/basalt">GitHub - basalt-org/basalt: A Machine Learning framework from scratch in Pure Mojo 🔥</a>：一个使用纯 Mojo 🔥 从零开始构建的机器学习框架 - basalt-org/basalt
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1232024125971890256)** (6 messages): 

- **预热警报**：Modular 分享了一条[神秘的预热推文](https://twitter.com/Modular/status/1782457222511161545)，暗示即将有大事发生。
- **Modular 期待值拉满**：Modular 的[第二条推文](https://twitter.com/Modular/status/1782457235454689752)提高了追随者的期望，暗示即将揭晓。
- **激动人心的倒计时**：悬念随着 Modular 的[第三条预热推文](https://twitter.com/Modular/status/1782457253829935500)继续，指向一项重大公告。
- **Modular 势头聚集**：在[第四条推文](https://twitter.com/Modular/status/1782457261652312486)中，Modular 显然在进行倒计时，让社区保持高度期待。
- **最终预热**：Modular 系列推文的[最后一条](https://twitter.com/Modular/status/1782457268354809918)让追随者热切期待重大启示。
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1231629012510703649)** (3 messages):

- **寻求 AI 视频的互动**：一位成员分享了一个名为 "The Rise of AI" 的 [YouTube 视频](https://youtube.com/watch?v=SfKGHKzkm-o) 用于大学作业，并请求互动和反馈。他们承认由于时间限制，内容深度有限，并提到英语不是他们的母语。

- **追求人工意识生命**：一位成员表达了对计算物理和计算机科学/工程双学位的兴趣，目标是创造人工意识生命。他们质疑 AI 的现状、电力和数据的低效，以及为了实现这一目标可能需要量子计算或三进制系统等技术进步。

- **对 AI 领域量子计算的怀疑观点**：在讨论量子计算在 AI 中的应用时，一位成员指出了量子系统中随机性和效率的挑战，并提到难以一致地执行简单操作。还有人担心政府干预可能会阻碍该领域的进展。

- **AI 开发中提到的三进制计算**：在讨论开发通用人工智能 (AGI) 所需的进步时，简要提到了三进制计算系统 [Setun 计算机](https://en.wikipedia.org/wiki/Setun)。该成员认为，计算架构比单纯的计算规模扩张对于 AGI 的进展更为关键。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Setun">Setun - Wikipedia</a>：未找到描述</li><li><a href="https://youtube.com/watch?v=SfKGHKzkm-o">The Rise of AI</a>：(Hidupkan Closed Caption)(开启字幕) 加入我们的旅程，见证 Artificial Intelligence 的快速演进，从...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1231249583397408778)** (338 messages🔥🔥): 

- **在 Mojo 中探索 Type State 模式**：一位用户询问如何在 Mojo 中实现 Type State Pattern，另一位成员分享了 **traits 中的关联类型 (associated types)** 作为潜在解决方案。然而，该特性似乎尚未在稳定版 Mojo 中实现，但可能通过使用带有 `_getitem` 和 `_setitem` 的 `Size` trait 的变通方法来工作。
  
- **理解 Mojo 的 Parameters 和 Arguments**：一位用户澄清了 Mojo 中 parameters 和 arguments 之间的区别——parameters 是编译时常量，而 arguments 是运行时值。这种困惑源于一次关于排序算法的讨论，其中分享了一个使用 `T:Sortable` trait 和 `cmp_fn` 函数参数的代码片段，从而引发了对方括号中表示的函数参数的探索。

- **使用 Traits 的排序策略**：另一位成员分享了一个使用 traits 的快速排序 (quicksort) 实现示例，并提供了改进建议。尽管代码遇到了 '`T` 未实现 `__ge__` 方法' 的错误，讨论还包括使用 `UnsafePointer` 代替 `Pointer`，并理解具有重载比较运算符（`__le__` 和 `__ge__`）的 `Sortable` trait 对于排序自定义数据类型非常有用。

- **指针和列表的问题**：讨论了在尝试将字符串与指针结合使用时引起的段错误 (segmentation fault)。用户讨论了潜在原因，如分配错误或使用值语义导致的不当行为，突显了 Mojo 中内存管理的复杂性。

- **Regex 功能与 Mojo 实现**：一位用户思考了 Mojo 中 regex 功能的实现，并分享了一个 Python 示例作为参考，并指出截至频道历史记录截止时，Mojo 中还没有 regex 实现。他们表示打算为一个项目想法尝试实现一种基础形式的 regex。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/equality_comparable#__eq__">equality_comparable | Modular 文档</a>: EqualityComparable</li><li><a href="https://docs.modular.com/mojo/stdlib/collections/">collections | Modular 文档</a>: 实现 collections 包。</li><li><a href="https://docs.modular.com/mojo/stdlib/algorithm/sort#partition">sort | Modular 文档</a>: 实现排序函数。</li><li><a href="https://docs.python.org/3/howto/sorting.html#key-functions">排序技巧</a>: 作者 Andrew Dalke 和 Raymond Hettinger。Python 列表具有内置的 list.sort() 方法，可就地修改列表。还有一个 sorted() 内置函数，用于构建一个新的已排序列表...</li><li><a href="https://joyofmojo.com/generic_quicksort/">通用快速排序</a>: 上下文 Mojo 参考：Sort Mojo 版本：24.2.1 演示：按年龄对一组人进行排序。此演示展示了如何使用通用的 QuickSort 算法根据年龄对一组人进行排序。这...</li><li><a href="https://programmersought.com/article/66388921702/">Python -c 命令行执行方法 - Programmer Sought</a>: 未找到描述</li><li><a href="https://docs.modular.com/mojo/manual/traits">Traits | Modular 文档</a>: 为类型定义共享行为。</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/simd">simd | Modular 文档</a>: 实现 SIMD 结构。</li><li><a href="https://gist.github.com/modularbot/3334ea937074b8d2349fddaee2a04cd1">playground.mojo</a>: GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://docs.modular.com/mojo/stdlib/memory/unsafe#bitcast-2">unsafe | Modular 文档</a>: 实现用于处理不安全指针的类。</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/anytype.mojo">mojo/stdlib/src/builtin/anytype.mojo at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://www.arewewebyet.org/">Are we web yet? Yes, and it's freaking fast! </a>: 未找到描述</li><li><a href="https://docs.modular.com/mojo/manual/parameters/#parameterized-functions">参数化：编译时元编程 | Modular 文档</a>: 参数和编译时元编程简介。</li><li><a href="https://tenor.com/view/ron-swanson-parks-and-rec-its-so-beautiful-gif-15644547">Ron Swanson Parks And Rec GIF - Ron Swanson Parks And Rec Its So Beautiful - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/modularml/mojo/issues/2113)">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/toiletsandpaper/mojo_zlib_classification/blob/master/tools/utils.mojo">mojo_zlib_classification/tools/utils.mojo at master · toiletsandpaper/mojo_zlib_classification</a>: 通过在 GitHub 上创建账户为 toiletsandpaper/mojo_zlib_classification 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2197">[功能请求] `.__doc__` 属性 · Issue #2197 · modularml/mojo</a>: 审查 Mojo 的优先级。我已阅读路线图和优先级，并相信此请求符合优先级。您的请求是什么？我希望能够获取我的字符串的 docstring...</li><li><a href="https://github.com/modularml/mojo/issues/2164)">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://mlir.llvm.org/">MLIR</a>: 未找到描述</li><li><a href="https://youtu.be/lXAp6ZAWyBY?si=OSuCzPUmuohgUYvL">2023 LLVM 开发者大会 - MLIR 不是 ML 编译器以及其他常见误解</a>: 2023 LLVM 开发者大会 https://llvm.org/devmtg/2023-10------MLIR 不是 ML 编译器以及其他常见误解。演讲者：Alex Zinenko------幻灯片...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1231289596415836291)** (35 条消息🔥): 

- **神秘的 Llama 项目 Enigma**: 有人表示有兴趣构建一个被神秘地引用为 "🦙🦙🦙.🔥" 的项目，并建议开发一个具有绘图能力的办公套件，使用文本作为提示词。

- **丰富的 Mojo 项目**: 项目更新包括 `prism` 的类型化标志、用于终端样式的 `mog`、模拟 Go 的 `net` 包的 `gojo`，以及针对 MacOS 的 `termios` 工作，所有这些都可以在 GitHub 上找到，且需要 nightly 版本的 tuple 更新。([Prism](https://github.com/thatstoasty/prism), [Mog](https://github.com/thatstoasty/mog), [Gojo](https://github.com/thatstoasty/gojo), [Termios](https://github.com/thatstoasty/termios))

- **Basalt 框架寻求 Web 开发人员**：Basalt 机器学习框架团队正在寻求 Web 开发方面的专业人才，特别是具备 NextJS 和 ShadCN 知识的 UI/UX 专家，用于发布和增强其自动生成的文档。详情请访问 [Basalt's GitHub](https://github.com/basalt-org/basalt)。

- **Mojo 与 JSX 的世界**：有人提议为 LSX.mojo 仓库创建一个基于 HTML 语法、类似于 React 的开发框架，这表明 Mojo 社区对基于组件的 UI 框架有着浓厚兴趣。此外还暗示了 Mojo 静态网站生成器的想法，目前一个 Djot 解析器正在开发中。([LSX Repo](https://github.com/lsh/lsx))

- **MoCodes 进军纠错领域**：分享了 MoCodes 项目，这是一个用 Mojo 编写的纠错编解码（Error Correction (De)Coding）框架。它旨在优化传统上由专用硬件处理的计算密集型纠错码过程。正如 [GitHub](https://github.com/alainrollejr/mocodes) 上的 README 所述，该项目正在寻求合作。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/basalt-org/basalt">GitHub - basalt-org/basalt: A Machine Learning framework from scratch in Pure Mojo 🔥</a>：纯 Mojo 编写的从零开始的机器学习框架 🔥 - basalt-org/basalt</li><li><a href="https://github.com/thatstoasty/prism">GitHub - thatstoasty/prism: Mojo CLI Library modeled after Cobra.</a>：模仿 Cobra 的 Mojo CLI 库。通过在 GitHub 上创建一个账号来为 thatstoasty/prism 做出贡献。</li><li><a href="https://github.com/thatstoasty/mog">GitHub - thatstoasty/mog: Style definitions for nice terminal layouts.</a>：用于精美终端布局的样式定义。通过在 GitHub 上创建一个账号来为 thatstoasty/mog 做出贡献。</li><li><a href="https://github.com/thatstoasty/gojo">GitHub - thatstoasty/gojo: Experiments in porting over Golang stdlib into Mojo.</a>：将 Golang 标准库移植到 Mojo 的实验。 - thatstoasty/gojo</li><li><a href="https://github.com/thatstoasty/termios">GitHub - thatstoasty/termios: Mojo termios via libc</a>：通过 libc 实现的 Mojo termios。通过在 GitHub 上创建一个账号来为 thatstoasty/termios 做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1231200334995521667)** (19 条消息🔥): 

- **探索 CPU 限制下的性能**：在将 CPU 限制为 1400 MHz 的测试中，**Mojo scalar** 的性能为每个项目 1.4 ns，而 **Rust 和 Mojo SIMD** 的性能相似，约为每个项目 1.0 ns，即使在计时部分前后包含了调试打印也是如此。

- **寻求最优 Parallelize 策略**：一位成员注意到 **X thread demo** 和 `Matmul` 文档中 `parallelize` 的用法存在差异，后者指定了 `num_workers`，而前者没有。据报告，在不显式设置 worker 数量时，性能存在波动且缺乏稳定性。

- **多线程的难题**：成员们讨论了多线程中设置 worker 数量的复杂性和最佳实践。强调了多线程性能取决于核心数量、具体问题以及程序是否可以占用所有资源。

- **Worker 数量：是否需要指定？**：另一位成员对此表示赞同，强调了多线程中的挑战和注意事项，并建议有时将 worker 数量设置得高于核心数量是有益的，正如 [关于 Matmul 的 Modular 博客文章](https://www.modular.com/blog/mojo-a-journey-to-68-000x-speedup-over-python-part-3) 中所演示的那样。

- **随机数生成的性能谜题**：一位成员发布了一个通过蒙特卡洛（Monte Carlo）方法计算 pi 的 Mojo 脚本，指出它比 Numba-jitted 的 Python 版本慢得多，大量时间花在了生成随机数上。根据建议，已在 [GitHub 上提交了 issue](https://github.com/modularml/mojo/issues/2388) 以解决 `random.random_float64` 的性能问题。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/blog/mojo-a-journey-to-68-000x-speedup-over-python-part-3">Modular: Mojo🔥 - A journey to 68,000x speedup over Python - Part 3</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo🔥 - 比 Python 快 68,000 倍的旅程 - 第 3 部分</li><li><a href="https://www.infoq.com/presentations/multithreading/">The Dos and Don'ts of Multithreading </a>：Hubert Matthews 描述了多线程中遇到的一些问题，并讨论了如何通过适当的设计选择来避免这些问题。</li><li><a href="https://docs.modular.com/mojo/stdlib/algorithm/functional#parallelize,">functional | Modular Docs</a>：实现高阶函数。</li><li><a href="https://github.com/modularml/mojo/issues/2388">[BUG] `random.random_float64` is extremely slow · Issue #2388 · modularml/mojo</a>：Bug 描述：在 for 循环中一次生成一个随机数极其缓慢，比 numba-jitted 的等效代码慢了近 2 个数量级。背景：我尝试使用一个简单的 Monte Ca...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1231366433217314909)** (24 messages🔥): 

- **C++ ORT 性能查询**：一位成员对 C++ 在 ONNX Runtime (ORT) 下的性能测量方式与 Mojo 的对比感到好奇。他们讨论了 Python 的开销，并思考 C++ 是否因为较少的 Python API 调用而具有固有的优化优势。
- **Python 与 C++ 中的图像处理**：另一个讨论围绕在 Python/Mojo 中使用 numpy 和 cv2 进行图像预处理，与在 C++ 中使用其原生 OpenCV 和自定义函数进行对比。大家注意到，在这两种语言中，后处理主要都是由原生代码执行的。
- **基准测试分享提议**：一位成员提到他们对三种语言进行了性能基准测试，并提议分享结果的对比表。
- **ONNX 模型输入难题已解决**：一位成员遇到了 ONNX 模型接受名为 "input.1" 的输入张量的问题，并寻求在 `model.execute` 调用中使用它的解决方法。提供了一个使用 **PythonObject** 的解决方案以及在 Python 中使用 **kwargs** 的替代方法。
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1231136409642074123)** (36 messages🔥): 

- **指针难题与 Unsafe 冒险**：社区讨论了各种指针类型的语义，建议为某些类型添加 "Unsafe" 前缀以反映其本质。目前也正在开展逐步淘汰 `LegacyPointer` 的工作，并鼓励大家参与贡献，如针对此努力的一个 [小型 PR](https://github.com/modularml/mojo/pull/2365) 所示。

- **解决更新障碍**：一位用户强调了最近更新到 Mojo 版本 2024.4.1618 时的一个问题，其中 `SIMDType.to_int()` 导致构建失败。经澄清，更新后该方法已被简单的 `int(...)` 调用所取代。

- **处理字符串比较**：提议了一段用于实现 String 比较的代码片段，并考虑了未来的 Unicode 因素，这引发了对之前解决类似问题的 PR 的回顾。

- **元组复制之谜与 UnsafePointer**：有人提出了关于在元组复制操作中使用 `__get_address_as_owned_value` 的问题，暗示这可能与新的 `UnsafePointer` 类型处理引用和生命周期的方式存在冲突。

- **字符串表示与语义难题**：`String()` 和 `String("")` 之间的区别（后者包含一个空终止符）引发了关于它们正确的分配行为以及“什么是空字符串”的哲学含义的讨论。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/1904">[Feature Request] Explicit parametric alias with default argument · Issue #1904 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。你的请求是什么？如题。你进行此更改的动机是什么？Exp...</li><li><a href="https://github.com/modularml/mojo/pull/2365">[stdlib] Replace `Pointer` by `UnsafePointer` in `stdlib/src/builtin/object.mojo` by gabrieldemarmiesse · Pull Request #2365 · modularml/mojo</a>：Builtins 导入的行为方式很奇怪，我不得不在 stdlib/src/python/_cpython.mojo 中导入 LegacyPointer，我对此无法解释。我只是按照编译器的要求导入 :p 参见 ht...
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1231138113774944297)** (462 messages🔥🔥🔥):

- **LLaMa 3 Tokenizer 问题**：Discord 成员讨论了微调 LLaMa 3 模型时遇到的问题，重点是 BOS (beginning-of-sentence) token 未能按预期添加。一个解决方法是根据 [Llama HF discussions](https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/41) 中的一个 Pull Request 手动更新 `tokenizer.json`，从而修复了该问题。

- **GPU 与训练时间的揭秘**：围绕训练 AI 模型的高资源消耗展开了讨论，特别是在 Phi-3 模型发布后。一位成员指出，一个配置了 512 个 H100-80G GPU 的环境运行了 7 天，这表明了所需计算规模之大。

- **Phi-3 超出预期**：频道中的对比显示，尽管 Phi-3 模型的参数量相对较小（约 3.8b），但它们展现出了与更大模型竞争的性能，引发了对其效率和潜力的推测与关注。

- **OpenAI 与 AI 竞赛**：成员们讨论了在竞争对手快速发布 AI 模型的情况下 OpenAI 的沉默。推测包括 OpenAI 正专注于 2025 年发布 GPT-5，以及当前模型可能影响或加速这些计划。

- **Phi-3 的许可与能力**：Phi 系列的开源 MIT 许可被强调为一个显著优势，尽管这些模型缺乏广泛的知识库。讨论表明，这些模型在推理能力上可能优于记忆能力，使其成为未来应用集成的令人兴奋的选择。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discordapp.com/channels/1104757954588196865/1192621815172964522/1192712427750572032">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的朋友和社区保持紧密联系。</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format">Axolotl - 指令微调 (Instruction Tuning)</a>：未找到描述</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - 数据集格式</a>：未找到描述</li><li><a href="https://huggingface.co/chargoddard/llama3-42b-v0">chargoddard/llama3-42b-v0 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome>">环境变量</a>：未找到描述</li><li><a href="https://huggingface.co/mattshumer/Llama-3-8B-16K">mattshumer/Llama-3-8B-16K · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ca4q50/psa_check_that_your_train">Reddit - 深入了解任何内容</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ca4q50/psa_check_that_your_training_setup_is_adding_bos/">Reddit - 深入了解任何内容</a>：未找到描述</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b/discussions/11">cognitivecomputations/dolphin-2.9-llama3-8b · Llama 3 Base 是独特的</a>：未找到描述</li><li><a href="https://www.philschmid.de/fsdp-qlora-llama3">使用 PyTorch FSDP 和 Q-Lora 高效微调 Llama 3</a>：了解如何使用 Hugging Face TRL, Transformers, PEFT 和 Datasets，通过 PyTorch FSDP 和 Q-Lora 微调 Llama 3 70b。</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/41">meta-llama/Meta-Llama-3-8B · 更新后处理器以添加 bos</a>：未找到描述</li><li><a href="https://github.com/janphilippfranken/sami">GitHub - janphilippfranken/sami: 基于互信息的自监督对齐</a>：基于互信息的自监督对齐。通过在 GitHub 上创建一个账户来为 janphilippfranken/sami 的开发做出贡献。</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct">meta-llama/Meta-Llama-3-8B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c8r08t/are_there_any_llama_3_8b_finetunes_already/l0gs1mb/>">Reddit - 深入了解任何内容</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1231211561906081864)** (19 条消息🔥):

- **GPU 在 8-bit 优化器上遇到困难**：一位成员提到多 GPU 设置是必要的，但指出 8-bit 优化器未能按预期工作的问题。
- **极其消耗 VRAM 的 AdamW_Torch**：鉴于 8-bit 优化器表现不佳，AdamW_Torch 优化器被认为是一个消耗大量 VRAM 的替代方案。
- **寻求 8-bit 优化器的配置**：成员们正在请求并分享针对 LLaMA3 等模型的 8-bit 优化器示例配置。
- **排除 Discord 链接故障**：成员们尝试分享 Discord 链接，但遇到了链接无法按预期工作的问题。
- **补丁后的主观改进**：在对 LLaMA3 应用补丁后，成员们注意到尽管 Loss 指标保持不变，但主观上有所改进，强调“氛围评估 (vibes eval)”优于 Loss 数据。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1231222189815107714)** (19 messages🔥): 

- **QMD vs. Markdown**：有人询问文档为何突然切换到 **qmd**，并对它在 GitHub 上的渲染效果表示担忧。
- **量化配置查询**：一位成员询问了 70B 模型的**量化配置**，并明确了通常使用来自 'examples/quantize.py' 的 **config.json**。
- **模型合并耗时担忧**：讨论了在 **4 张 A100** 上微调 **70B 模型**后，将 LoRA 合并回基座模型所需的时间；一位成员认为超过一个半小时的时间太长了。
- **对话数据集澄清**：关于 "train_on_inputs" 是否影响多轮对话数据集中的标签问题得到了确认；它特别会影响用户输入。
- **数据集类型与文档**：有人请求获取有关数据集类型的信息，一位成员分享了一个[综合链接](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)，详细介绍了 **Axolotl 支持的数据集格式**，包括对话、预训练、指令微调、无模板以及自定义预分词数据集。

**Link mentioned**: <a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - Dataset Formats</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1232142687084347422)** (1 messages): 

- **Llama 长度增加**：分享了一个指向 **Llama 3（16K token 长度模型）**的链接，并配以一个看似印象深刻的表情符号。该链接指向 [huggingface.co](https://huggingface.co/mattshumer/Llama-3-8B-16K)，表明用户对扩展长度能力的兴趣。

**Link mentioned**: <a href="https://huggingface.co/mattshumer/Llama-3-8B-16K">mattshumer/Llama-3-8B-16K · Hugging Face</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/)** (1 messages): 

duh_kola: 与 axolotl 无关，但没错，我无法使用 runpod 向 hub 上传任何东西。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1231356634819854396)** (22 messages🔥): 

- **YAML 配置中 "conversation:" 键的澄清**：一位成员询问了 YAML 配置文件中用于训练数据集的 `"conversation:"` 键。另一位成员澄清说，这仅适用于 `sharegpt` 类型的数据集。

- **"sharegpt" 与 "chatml" 的复杂情况**：当一位成员询问指定 `"type: sharegpt"` 和 `"conversation: chatml"` 的效果时，他们被告知这表示数据集是 ShareGPT 格式，并指示在模型训练时将数据转换为 ChatML 格式。

- **建议的错误排查步骤**：针对一位成员报告的在分布式计算期间出现的多个 `SIGBUS` (Signal 7) 错误，建议他们检查内存对齐问题、审查内存映射文件的使用、检查硬件、更新依赖项并简化设置以诊断问题。

- **在 Axolotl 中使用 Unsloth 的指南**：关于如何将 Unsloth 集成到 Axolotl 进行训练的问题，最终形成了一份简短指南，指导如何安装依赖、准备模型和数据、使用正确参数配置 Unsloth、运行训练过程并监控结果以实现高效优化。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e7301808-4b94-41b9-b3d4-752db98cf71f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=33a203be-00f7-40dc-9fa2-e911b904e980)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e4ffa5d8-9095-4a00-8773-02132978f2e7)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=4eadad10-1146-45ad-9822-155e9b87cb48)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1231399047944998925)** (7 条消息): 

- **负载均衡优化进行中**：**Wizard 8x22b** 的流量正在导致性能下降，但负载均衡调整预计很快会改善延迟。

- **提高请求吞吐量**：对 **load balancer** 的更改以及与 stop tokens 处理相关的修复应该会提高非流式请求的吞吐量。

- **删除 Nitro Instruct 模型**：发送至 **Databricks: DBRX 132B Instruct (nitro)** 的请求现在将重定向到主 [Databricks: DBRX 132B Instruct](https://openrouter.ai/models/databricks/dbrx-instruct) 模型。

- **推出新模型并扩展上下文支持**：OpenRouter 宣布了 3 个新模型，包括一个免费的 **Llama 3 finetune**，以及将 **Llama 3 8B** 扩展到 16k 上下文。在模型发布的同时，还在解决提示词格式化改进和特定区域的网络问题，重点是增强 **dynamic routing**。[模型讨论和详情可以在这里找到。](https://discord.com/channels/1091220969173028894/1232005820229877822/1232005820229877822)

- **MythoMax 13B 问题解决**：在顶级供应商缓解问题后，遇到 **MythoMax 13B** 输出问题的用户应该会看到改进。疑难问题可以在提供的[讨论线程](https://discord.com/channels/1091220969173028894/1232171735944532059)中反馈。

- **解决 504 错误激增问题**：由于 **美国中部和西部地区** 的网络问题，用户正遇到 504 错误，这影响了 Llama 2 tokenizer 模型。目前正在开发一项修复方案，以移除对当前处于宕机状态的 Hugging Face 的依赖。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/databricks/dbrx-instruct).">Databricks: DBRX 132B Instruct by databricks | OpenRouter</a>: DBRX 是由 Databricks 开发的新型开源大语言模型。在 132B 参数量下，它在语言的标准行业基准测试中优于现有的开源 LLM，如 Llama 2 70B 和 Mixtral-8x7B...</li><li><a href="https://openrouter.ai/models/lynn/soliloquy-l3.">Lynn: Llama 3 Soliloquy 8B by lynn | OpenRouter</a>: Soliloquy-L3 是一款快速、高性能的角色扮演模型，专为沉浸式、动态体验而设计。Soliloquy-L3 在超过 2.5 亿个 tokens 的角色扮演数据上进行了训练，拥有广博的知识库...</li><li><a href="https://openrouter.ai/models/sao10k/fimbulvetr-11b-v2">Fimbulvetr 11B v2 by sao10k | OpenRouter</a>: 创意写作模型，经许可路由。它速度很快，能保持对话持续进行，并保持角色设定。如果你提交原始提示词，可以使用 Alpaca 或 Vicuna 格式。</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-8b-instruct:extended">Meta: Llama 3 8B Instruct (extended) by meta-llama | OpenRouter</a>: Meta 最新的模型系列 (Llama 3) 发布了多种尺寸和版本。这个 8B 指令微调版本针对高质量对话场景进行了优化。它展示了强大的...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1231369890829439056)** (3 条消息): 

- **合同标准意识建议**：一项产品反馈建议，应在用户上传期间提示其选择合同标准，以确保用户知晓仅支持特定的合同类型。这可以防止对未处理、不支持的合同产生困惑。 

- **用户本地化和合同偏好功能构想**：另一项建议提出，允许用户在入职或上传期间设置其位置以考虑当地法律，并启用一项功能来指示用户在谈判过程中希望偏袒哪一方。

- **违法条款检测功能请求**：建议产品应具备检测合同中违法和苛刻条款的能力，以防止非律师人员因加入违法条款而导致合同失效。

- **Keywords AI：基于 OpenRouter 构建的开发者工具**：发布了 [Keywords AI](https://keywordsai.co)，这是一个支持 OpenRouter（包括所有模型和“自带密钥”选项）的平台，强调其两行代码集成和以开发者为中心的功能。

- **DeepGaze 发布并支持 Reddit 监控**：分享了 [DeepGaze](https://www.deepgaze.ca/) 的发布，该服务将多种文档类型输入 GPT-4V，并使用 Discord 机器人识别与该功能匹配的有问题的 Reddit 用户。DeepGaze 利用 OpenRouter 来紧跟最新的 LLM 模型。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://keywordsai.co)">未找到标题</a>：未找到描述</li><li><a href="https://www.deepgaze.ca/">DeepGaze</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1231144139890884639)** (474 条消息🔥🔥🔥): 

- **WizardLM-2 的更多困扰**：用户报告 **WizardLM-2** 性能不稳定；一些人获得成功，而另一些人则遇到逻辑不连贯或无响应的情况。一位用户指出 **SillyTavern** 的 'Assistant Prefill' 可能导致 **LLaMA 3** 模型出现问题，而另一位用户讨论了由于微软计费系统仅显示一张发票而带来的困难。
  
- **OpenRouter 对技术故障的回应**：**OpenRouter** 承认了与提供商分词器（tokenizer）相关的问题。已部署热修复补丁以解决与 Hugging Face 相关的停机问题，并承诺将提供永久修复方案以消除该依赖。

- **费率与代币经济学（Tokenomics）受到审查**：用户质疑 AI 模型提供商如何能以当前费率提供服务，特别是与图像生成的成本相比。讨论提到了 FP8 量化和活跃工作者折扣在降低成本方面的潜在作用，一位用户指出 **Groq** 的硬件由于高能耗可能不太经济。

- **探索未知的模型领域**：成员们分享了他们在各种主题上的经验和询问，包括 **Phi-3-mini 模型**、新的 **LLaMA 3 70b** 变体，以及 **WizardLM-2** 与微软可能的联系。爱好者们渴望获得新发布的模型，而其他人则在推测 **RWKV** 的未来并比较 AI 的写作风格。

- **期待模型更新与添加**：**OpenRouter** 用户等待 **LLaMA 3 70b** 的无审查版本，讨论了可越狱（jailbreakable）模型的重要性，并思考 **Phi-3** 登陆该平台的可能性。他们还提到了对 **8x22** 模型的偏好，强调了成本与功能之间的平衡。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://imgur.com/a/XoI7ZD9">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门的梗图、有趣的 GIF、鼓舞人心的故事、病毒式传播的视频来振奋你的精神...</li><li><a href="https://www.semianalysis.com/p/groq-inference-tokenomics-speed-but">Groq 推理代币经济学：速度，但代价是什么？</a>：比 Nvidia 更快？剖析其经济学原理</li><li><a href="https://huggingface.co/openlynn/Llama-3-Soliloquy-8B">openlynn/Llama-3-Soliloquy-8B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx">microsoft/Phi-3-mini-128k-instruct-onnx · Hugging Face</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Work-to-rule">按章工作 - 维基百科</a>：未找到描述</li><li><a href="https://x.com/erhartford/status/1781199815772438819">Eric Hartford (@erhartford) 的推文</a>：由 @CrusoeCloud 慷慨赞助的 Dolphin-2.9-llama3-8b 预计周六发布。与 @LucasAtkins7 和 @FernandoNetoAi 进行了大量合作。Dolphin-2.9-llama3-70b 紧随其后。Dolphin-2.9-mixtral-8x22b 仍在...</li><li><a href="https://huggingface.co/posts/WizardLM/329547800484476">Hugging Face 上的 @WizardLM："🔥🔥🔥 介绍 WizardLM-2！

📙发布博客：…"</a>: 未找到描述</li><li><a href="https://huggingface.co/dreamgen/opus-v1.2-llama-3-8b">dreamgen/opus-v1.2-llama-3-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://openrouter.ai/playground?models=meta-llama/ll">OpenRouter</a>: LLM 和其他 AI 模型的路由</li><li><a href="https://openrouter.ai/playground?models=meta-llama/llama-3-8b-instruct">OpenRouter</a>: LLM 和其他 AI 模型的路由</li><li><a href="https://fireworks.ai/blog/fire-attention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs">FireAttention — Serving Open Source Models 4x faster than vLLM by quantizing with ~no tradeoffs</a>: 通过无损量化，以比 vLLM 快 4 倍的速度提供开源模型服务</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct">microsoft/Phi-3-mini-4k-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-70b-instruct:nitro">Meta: Llama 3 70B Instruct (nitro) by meta-llama | OpenRouter</a>: Meta 最新发布的模型系列 (Llama 3) 包含多种尺寸和版本。这个 70B 指令微调版本针对高质量对话场景进行了优化。它展示了强大的...</li><li><a href="https://openrouter.ai/models/lynn/soliloquy-l3">Lynn: Llama 3 Soliloquy 8B by lynn | OpenRouter</a>: Soliloquy-L3 是一款快速且功能强大的角色扮演模型，专为沉浸式、动态体验而设计。Soliloquy-L3 在超过 2.5 亿个 token 的角色扮演数据上进行了训练，拥有庞大的知识库和丰富的...
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1231148707211247669)** (303 条消息🔥🔥): 

- **Atlas 机器人潜入讨论**：**Atlas 机器人**的最新发布引发了关于其“诡异感”和社交媒体营销策略的讨论，大家都在期待其商用型号，一位成员表示**期待**看到它最终的能力。
- **AI 灵性辩论**：一位成员询问 AI 灵性的形式可能是怎样的，引发了关于 AI 意识、人性以及情感的激烈辩论，随后因非世俗讨论的规则限制而受到管理。
- **GPT-3 的 API 和界面创新**：讨论涉及了利用 **MyGPT** 代码创建 API 的潜力，以及 **MetaGPT** 和 **Devika** 等工具的进展，这些工具可以帮助编写应用程序并可能与 GitHub 交互。
- **LLaMa 3 的重要性与局限性**：成员们讨论了各种 AI 模型的最新改进，LLaMa 3 的性能评价褒贬不一，而关于 **GPT-5** 发布日期的传闻在没有官方公告的情况下被视为虚假消息。
- **生成式模型文献与活跃的 AI**：针对 ChatGPT 和 DALL-E 等 AI 及生成式算法深度资源的需求，建议搜索 **OpenAI 发表的论文**和 Arxiv 等仓库；同时，关于 **LLaMa 3** 异常输出（过度使用感叹号）的轶事突显了该模型意想不到的怪癖和局限性。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/joe-bereta-source-fed-micdrop-im-out-gif-11904628">Joe Bereta Source Fed GIF - Joe Bereta Source Fed Micdrop - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://en.wikipedia.org/w/index.php?title=Biorobotics">Biorobotics - Wikipedia</a>: 未找到描述</li><li><a href="https://openai.com/research/generative-models">Generative models</a>: 这篇文章描述了四个项目，它们共同的主题是增强或使用生成式模型，这是机器学习中无监督学习技术的一个分支。除了描述我们的工作...</li><li><a href="https://openai.com/research/overview">Research</a>: 我们相信我们的研究最终将实现通用人工智能 (AGI)，这是一个能够解决人类水平问题的系统。构建安全且有益的 AGI 是我们的使命。</li><li><a href="https://openai.com/research/gpt-4">GPT-4</a>: 我们创建了 GPT-4，这是 OpenAI 在扩展深度学习方面的最新里程碑。GPT-4 是一个大型多模态模型（接受图像和文本输入，输出文本），虽然在某些方面不如...
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1231139832542134272)** (33 条消息🔥): 

- **GPT Agent 与 LLama 3 70B 集成尝试**：一位成员分享了他们尝试使用 Groq 将 Agent GPT v2 与 LLama 3 70B 集成的经历，但遇到了问题，其他成员也报告集成失败。然而，一些用户最终发现它可以运行，这表明可能存在间歇性访问问题或特定用户条件影响了功能。

- **警惕分享 CGPT 聊天记录**：有成员对发布 CGPT 聊天的分享链接表示担忧，在分享日志时保持谨慎，原因是涉及在没有明确反馈的情况下，通过访问和评估查询来改进模型响应的问题。

- **探索 LLM 中的卷积层和 LoRa**：讨论了被称为 Hyena 的卷积层是否可以与 Stable Diffusion 等其他模型中的 LoRa 层相媲美。一位成员指出 LoRa 可用于微调大语言模型（LLM），其他成员则询问了目前有哪些模型正在积极采用这些技术及其优势。

- **需要管理 ChatGPT 历史记录的工具**：用户正在寻找工具或替代网站来更好地管理他们的 ChatGPT 历史记录，并强调了 OpenAI 当前门户网站的局限性。讨论还关注了任何第三方管理解决方案可能需要 API key 的必要性。

- **关于 ChatGPT 微调和文件保留的澄清**：一位用户获知，ChatGPT 的微调是指通过 API 输入内容以改变模型行为，而上传的文档仅作为参考资料，不会改变底层模型。此外，有人指出，聊天中附加的文件将根据 OpenAI 现有的指南进行保留，有用户提到根据之前的条件，保留期为 3 小时。

---

**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1231589837035929671)** (24 messages🔥): 

- **自定义指令中简洁是关键**：用户讨论了 ChatGPT 自定义指令的最佳长度；一位用户选择极简引导以节省上下文空间，而其他人则在尝试不同长度后发现，过长的指令可能会适得其反，因为 AI 可能会“遗忘”它们。
- **寻求刑法 Prompt**：一名法学学生询问关于刑法的 Prompt，但该请求目前仍在等待社区的建议或技巧。
- **使用 GPT-4 优化邮件增强**：一位用户正在微调一个使用 GPT-4 增强邮件的程序，并就当 AI 输出不理想时如何改进 Prompt 寻求建议。
- **Prompt 库在哪里？**：频道的一名成员询问了 Prompt 库的位置，这是一个可能有助于开发更有效 Prompt 的资源。
- **Prompt Engineering 技巧与伦理**：讨论涉及了 Prompt Engineering 的实践，触及了分享潜在有害技术的伦理影响和担忧；然而，并未提供具体的技巧或案例。

---

**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1231589837035929671)** (24 messages🔥): 

- **倾向于简短的自定义指令**：一位用户提到保持自定义指令简单，例如 *在适用的回答中使用分号、冒号和破折号*，以节省上下文窗口空间。
- **思考指令的长度与质量**：关于 Prompt 长度的讨论表明，用户认为有时更长、更详细的 Prompt 并不一定能带来更高质量的 AI 响应，暗示简短的 Prompt 可能更可取。
- **探索 Prompt 拆分策略**：针对如何处理超长 Prompt 的疑问，一位成员建议将其拆分并分布在多条消息中，以防止 AI 遗忘之前的内容。
- **Prompt 技巧与人物**：一位用户表达了对名为 **RageGPTee** 的 Prompt 工程师的钦佩，此人以先进技术和在分享突破性技能后“消失”而闻名，不过另一个人幽默地夸大了他的能力。
- **通过 GPT-4 查询增强邮件**：一名成员正在寻求优化 Prompt 的建议，用于一个使用 GPT-4 增强邮件草拟的程序，因为 AI 偶尔会产生欠佳的输出。

---

**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1231244510629986304)** (298 messages🔥🔥): 

- **LLM 多模态担忧**：频道参与者讨论了现有的多模态数据集（总计约 200 万对）可能会导致模型在特定数据集（如 LAION-COCO 的 GPT-4v 标题）上产生过拟合。这种过拟合是当前多模态方法中一个值得注意的问题。

- **MoA 架构发布**：分享了一种名为 Mixture-of-Attention (MoA) 的新架构，[该论文中有所描述](https://snap-research.github.io/mixture-of-attention/)，它可以在个性化图像生成中实现主体与上下文生成的解耦。

- **Discord 上的 AI 监控机器人**：讨论了关于监控机器人加入 Discord 服务器的担忧，并提供了指向 [kickthespy.pet](https://kickthespy.pet/#823813159592001537) 的链接，这是一项利用 API 漏洞来识别此类机器人的服务。

- **关于训练文本-图像扩散模型的讨论**：用户交流了关于训练文本-图像扩散模型挑战的见解，强调了数据质量、规模和模型架构的重要性。一个有趣的观点是，虽然 Chinchilla 的训练方法没有详细说明，但 dropout 和其他正则化方法（regularization methods）可能会显著影响训练结果。

- **Adobe 发布 Firefly Image 3**：Adobe 宣布了 [Adobe Firefly Image 3 Foundation Model](https://www.adobe.com/products/firefly.html) 的 Beta 版本，该模型提供了改进的图像生成质量和速度，现已集成到 Photoshop 中，并可通过 Firefly Web 应用程序访问。用户们很想测试它在不同创意提示词下的能力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://kickthespy.pet/#823813159592001537">Kick the Spy Pet</a>：未找到描述</li><li><a href="https://news.adobe.com/news/news-details/2024/Adobe-Introduces-Firefly-Image-3-Foundation-Model-to-Take-Creative-Exploration-and-Ideation-to-New-Heights/default.aspx">Adobe 推出 Firefly Image 3 基础模型，将创意探索和构思提升至新高度</a>：未找到描述</li><li><a href="https://snap-research.github.io/mixture-of-attention/">Mixture of Attention</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2203.15556">训练计算最优的大型语言模型</a>：我们研究了在给定计算预算下训练 Transformer 语言模型的最佳模型大小和 Token 数量。我们发现当前的大型语言模型明显训练不足...</li><li><a href="https://tenor.com/view/oh-no-top-gear-jeremy-clarkson-no-one-cares-gif-18925814">Oh No Top Gear GIF - Oh No Top Gear Jeremy Clarkson - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://wandb.ai/bghira/adamwbf16-wave/runs/e52bd9c5a68e37556a7f56479e5c2cce?nw=nwuserbghira">bghira</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://paperswithcode.com/dataset/cub-200-2011">Papers with Code - CUB-200-2011 数据集</a>：Caltech-UCSD Birds-200-2011 (CUB-200-2011) 数据集是细粒度视觉分类任务中最广泛使用的数据集。它包含属于鸟类的 200 个子类别的 11,788 张图像...</li><li><a href="https://youtube.com/watch?v=SfKGHKzkm-o">AI 的崛起</a>：(开启中文字幕) 加入我们的旅程，穿越人工智能的快速演变，从...</li><li><a href="https://huggingface.co/datasets/ptx0/mj-v52-redux/tree/main">ptx0/mj-v52-redux at main</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=fmI_OciHV_8">如何构建像 OpenAI Sora 这样的生成式 AI 模型</a>：如果你阅读过关于 OpenAI 和 Anthropic 等公司训练基础模型的文章，自然会认为如果你没有十亿美元...</li><li><a href="https://buttondown.email/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/">[AINews] FineWeb: 15T Tokens, 12 年的 CommonCrawl（已去重和过滤，不客气）</a>：2024/4/19-2024/4/22 的 AI 新闻。我们为你检查了 6 个 subreddits、364 个 Twitter 账号和 27 个 Discord 社区（395 个频道，14973 条消息）。预计阅读时间...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1231288421285236746)** (38 条消息🔥): 

- **评估 Blink 的视觉感知基准**：引入了一个名为 Blink 的新基准，用于测试多模态语言模型 (LLMs)，以评估其视觉感知能力。它涵盖了人类可以快速解决但对 GPT-4V 和 Gemini 等高级多模态 LLMs 来说极具挑战性的任务，这些模型在这些任务上的表现仅略好于随机猜测。[阅读更多关于 Blink 的信息](https://arxiv.org/abs/2404.12390)。

- **图像外推中的放大难题**：目前正在努力改进将 2D rope 外推结果从 256x256 分辨率提升到 1024x1024 的效果，目前结果并不理想，需要更高分辨率的微调。

- **Piecewise-Rectified Flow 集成到 ControlNet-Tile 流水线**：提到了 Piecewise-Rectified Flow (PeRFlow) 用于显著上采样图像，通过将 flow 与 ControlNet-Tile 流水线集成并细化图像，将图像从 64px 提升到 1024px。这可以在 [GitHub 的 piecewise-rectified-flow](https://github.com/magic-research/piecewise-rectified-flow/blob/main/README.md) 上找到。

- **HiDiffusion 提升 Diffusion Model 分辨率**：HiDiffusion 是由旷视科技 (MEGVII Technology) 和字节跳动 (ByteDance) 开发的新进展，声称只需一行代码即可提高 Diffusion Model 的分辨率和速度。该模块在输出中显示出伪影 (artifacts)，引发了对其在生成连贯高分辨率图像方面有效性的质疑。[探索 HiDiffusion 项目](https://hidiffusion.github.io/)。

- **SEED-X 多模态基础模型**：SEED-X 旨在通过理解任意尺寸的图像并实现多粒度图像生成，来弥补多模态基础模型 (Multimodal Foundation Model) 中的空白。这一统一且通用的基础模型在现实应用中展示了其有效性，具有用于理解和生成任务的多粒度视觉语义。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.14396">SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation</a>：多模态基础模型的快速演进在视觉语言理解和生成方面取得了显著进展，例如我们之前的工作 SEED-LLaMA。然而，仍然存在一个...</li><li><a href="https://arxiv.org/abs/2404.12803">TextSquare: Scaling up Text-Centric Visual Instruction Tuning</a>：随着多模态大语言模型 (MLLMs) 的发展，以文本为中心的视觉问答 (VQA) 取得了长足进步，但开源模型与 GPT 等领先模型相比仍有差距...</li><li><a href="https://wandb.ai/bghira/simpletuner-deepfloyd/runs/c2d8a68009185bfe4bc1072957e426db/workspace?nw=nwuserbghira">bghira</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://github.com/magic-research/piecewise-rectified-flow/blob/main/README.md">piecewise-rectified-flow/README.md at main · magic-research/piecewise-rectified-flow</a>：通过在 GitHub 上创建账号来为 piecewise-rectified-flow 的开发做出贡献。</li><li><a href="https://wandb.ai/bghira/simpletuner-deepfloyd/runs/c2d8a68009185bfe4bc1072957e426db/workspace?nw=nwu">bghira</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://arxiv.org/abs/2404.12390">BLINK: Multimodal Large Language Models Can See but Not Perceive</a>：我们介绍了 Blink，这是一个针对多模态语言模型 (LLMs) 的新基准测试，专注于其他评估中未包含的核心视觉感知能力。大多数 Blink 任务可以由人类解决...</li><li><a href="https://hidiffusion.github.io/">SOCIAL MEDIA TITLE TAG</a>：SOCIAL MEDIA DESCRIPTION TAG TAG</li><li><a href="https://github.com/megvii-research/HiDiffusion">GitHub - megvii-research/HiDiffusion</a>：通过在 GitHub 上创建账号来为 megvii-research/HiDiffusion 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1231977925314744360)** (6 messages): 

- **编程助手协作**：一位成员提到他们正开始构建一个专注于 **JavaScript/Rust** 而非 Python 的 NLP 编程助手，并表示有兴趣与他人合作。
  
- **协作的时间限制**：*softmax_function* 表示愿意偶尔协助该项目，理由是目前有多个项目，日程繁忙。

- **寻找过往工作**：*jcarbonnell* 询问是否存在包含先前工作的仓库，以便用于 NLP 编程助手项目。

- **承认过去的局限性**：*softmax_function* 承认由于当时缺乏 AI 知识而停止了之前的一个项目，但指出现在贡献能力已有所提高。

- **寻求任务分配澄清**：*jcarbonnell* 表示在不了解 *softmax_function* 过去贡献的情况下难以分配任务，并打算尝试他们分享的 *TrainedModel.py* 脚本和数据集。
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1231284983495135372)** (6 messages): 

- **RAG 实验焕然一新**：Aishwarya Prabhat 介绍了一个名为 **DREAM** 的框架，用于分布式 RAG (Distributed RAG) 实验，强调了构建生产级 RAG 系统时稳健基础设施的重要性。详细信息和见解发布在 [LlamaIndex 推文](https://twitter.com/llama_index/status/1781725652447879672)中。

- **LlamaIndex 金融机器人框架**：Hanane Dupouy 分享了一篇微型博客，介绍如何使用 **@llama_index** 构建一个金融 Agent，该 Agent 可以检索股票价格并总结财经新闻，增强与上市公司数据的交互。更多探索可以在分享的 [Twitter 链接](https://twitter.com/llama_index/status/1781837902139551920)中找到。

- **ColBERT 与记忆机制的结合**：针对在 RAG 流水线中加入对话历史的挑战，LlamaIndex 提出了一种由 **ColBERT** 驱动的检索 Agent，为对话助手存储“状态（state）”。通过其[最近的推文](https://twitter.com/llama_index/status/1782086279498539330)了解更多关于此方法的信息。

- **使用 LoRA 进行 RAG 微调**：Mariboo 的教程重点展示了如何使用 LoRA 权重来微调嵌入模型（embedding models），这是 RAG 流水线的关键部分。该教程利用了 **@llama_index 微调抽象**和 **@huggingface**。可以通过 LlamaIndex 的 [Twitter 帖子](https://twitter.com/llama_index/status/1782201783110213783)深入了解该教程。

- **通过开源 Reranker 升级你的 RAG**：@JinaAI_ 发布了两个 **开源 reranker**，通过对嵌入向量搜索进行二级排序来增强 RAG 系统。关于这些 reranker 的详细信息在 [LlamaIndex 的推文](https://twitter.com/llama_index/status/1782531355970240955)中进行了分享。

- **CRAG：RAG 检索的创新层**：LlamaIndex 讨论了纠正性 RAG（**CRAG**），它利用 **“reflection”** 层将检索到的信息分类为“正确”、“错误”或“模糊”，从而解决 RAG 中检索质量差的问题。有关 CRAG 的见解详见 LlamaIndex 的[推文](https://twitter.com/llama_index/status/1782799757376963006)。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1231156600190795826)** (188 messages🔥🔥): 

- **选择合适的检索方法**：用户讨论了不同的检索方法，如 RAG、CRAG 以及使用 Vector Databases 与 Knowledge Graphs 进行 reranking。共识倾向于*视具体用例而定*，特别是在处理涉及信息丢失风险的公司摘要时，更倾向于使用较大的 chunk sizes 或使用 SQL 和图技术。
  
- **集成与摘要挑战**：一位成员分享了在将 ChainLit 与 LlamaIndex 集成后，机器人仅回复与文档相关响应的挫败感，这暗示了 RAG 系统中的上下文管理问题。

- **AI 模型与 OpenAI 依赖**：关于在 llama_index 架构中使用 Groq、Bedrock 和 Ollama 等替代模型的问题不断出现，成员们解决了与 API key 错误和正确使用 embedding model 相关的疑问。

- **索引与存储探索**：成员们询问了 Supabase、Chromadb 和 Qdrant 等 Vector Stores 的功能与集成，经常遇到警告、Bug 或 401 错误，这些错误暗示了即使在没有明确使用的情况下也对 OpenAI 的 API key 存在依赖。

- **使用 DocumentSummaryIndex 进行摘要**：一位成员就如何让 DocumentSummaryIndex 在摘要时考虑所有 nodes 寻求建议，因为该工具在文档拆分产生的多个节点中仅选择了一个节点生成摘要。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/">Agents - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/?h=settings">从 ServiceContext 迁移到 Settings - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/WeaviateIndex_auto_retriever/?h=auto">从 Weaviate 向量数据库进行自动检索 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_tools/rag_cli/?h=rag+cli#customization">RAG CLI - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/callbacks/TokenCountingHandler/?h=token">Token Counting Handler - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/bedrock/?h=bedroc">Bedrock - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/deploying/query_engine/usage_pattern#get-started>).">使用模式 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/ollama/?h=ollama">Ollama - Llama 2 7B - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/localai/">LocalAI - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/portkey/?h=portkey)">Portkey - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/pull/13009">修复 qdrant 检查现有集合时的 bug，由 logan-markewich 提交 · Pull Request #13009 · run-llama/llama_index</a>: 修复了从可能存在的集合中获取信息时的一个小 bug</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/?h=query+pipeline+tool">围绕 Query Pipeline 构建 Agent - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/indexing/indexing#vector-store-index>).">索引与嵌入 - LlamaIndex</a>: 未找到描述</li><li><a href="https://mer.vin/2024/02/crewai-rag-using-tools/">使用工具的 CrewAI RAG - Mervin Praison</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/loading/documents_and_nodes/usage_documents#metadata>)">使用 Documents - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/data_connectors/PathwayReaderDemo#create-the-document-indexing-pipeline>).">Pathway Reader - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/querying/querying#querying>)">查询 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/getting_started/starter_example_local#query-your-data>)">入门教程（本地模型） - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/evaluation/UpTrain#create-a-query-engine-using-llamaindex>).">如何在 LlamaIndex 中使用 UpTrain - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1231221804518080615)** (5 条消息): 

- **Infini Attention 详解**: LinkedIn 上分享了关于新技术 **Infini Attention** 的解释，强调了其潜力并表达了对其即将发布的实现的期待。阅读 [LinkedIn](https://www.linkedin.com/posts/subham-kundu-2746b515b_llms-generativeai-activity-7187373540940148736-qNG6) 上的详解。

- **全面 AI 融资数据更新**: 一个追踪各城市 AI 融资和公司分布的全面数据集现已开放供社区审阅。通过 [Google Sheets](https://docs.google.com/spreadsheets/d/1nWBP1MpT7sACYDxqdCo8gBR7b2nXJbrF9Z43y69q9hg/edit#gid=752020121) 或 @WangUWS 在 [Twitter](https://x.com/WangUWS/status/1782069636030165106) 上的推文查看数据集及相关的城市分布分析。

- **LLM-Ready Markdown 获得提升**: LLM-ready Markdown 在 FireCrawl 和 LlamaIndex 的集成下实现了新的突破。阅读 [Medium](https://medium.com/ai-advances/unleash-the-potential-of-llm-ready-markdown-firecrawl-and-llamaindex-integration-243e494a9eb8) 上的进展介绍。

- **发布 Schema 控制的知识图谱**: WhyHow.AI 对其 Knowledge Graph SDK 进行了重大升级，支持从 PDF 创建 Schema 控制的自动化知识图谱。欲了解详情并参与 Beta 测试计划，请参阅 [Medium](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3) 上的公告。

- **关于 LLM 训练最佳数据库的辩论**：目前有一场关于 LLM 训练理想数据库类型的活跃讨论，涉及关系型、文档型、列式数据库的适用性，以及向量数据库的必要性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.google.com/spreadsheets/d/1nWBP1MpT7sACYDxqdCo8gBR7b2nXJbrF9Z43y69q9hg/edit#gid=752020121">[FrontierOptic.com] AI 融资追踪 - 2024 年 4 月 21 日 - 社区审阅版</a>：涵盖 &lt;a href=&quot;http://FrontierOptic.com&quot;&gt;FrontierOptic.com&lt;/a&gt; AI 初创公司融资数据（自 2023 年 5 月起）- 社区审阅版 &lt;a href=&quot;https://twitter.com/WangUWS&...</li><li><a href="https://x.com/WangUWS/status/1782069636030165106">来自 Howe Wang (@WangUWS) 的推文</a>：为了庆祝 @HilaryDuff 在《Wake Up》中演唱“Could be New York, Maybe Hollywood and Vine, London, Paris, maybe Tokyo” 20 周年。我清理了 AI Hype Train 数据的地点信息...
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1231255250250367139)** (110 条消息🔥🔥): 

- **探索 Open Interpreter 的功能与集成**：关于 Open Interpreter (OI) 功能的常规讨论包括：使用 `--server` 参数构建客户端的疑问、OI 在 Windows 系统上的挑战，以及与特定 [GitHub issue](https://github.com/OpenInterpreter/open-interpreter/issues/1185) 相关的安装问题。此外，还提到了成功将 OI 与 LLM 模型 Llama3 结合用于 Python 任务。
- **模型兼容性与性能**：用户正在讨论各种模型在 OI 上的表现，包括 **Llama3 70b**，有人确认其在 `--local` 模式下运行良好。同时，也有关于适用于直播和类人交互的最佳 text-to-speech 服务的咨询。
- **AI 视觉模型说明**：指出 Open Interpreter 使用 **GPT-4-vision-preview** 来识别屏幕截图。该模型名称是针对用户关于视觉任务所用 LLM 模型的询问而提供的。
- **开发挑战与解决方案分享**：用户提供了针对 pytesseract 错误等问题的解决方案并分享了修复方法，包括命令 `pip install --upgrade litellm`。故障排除的贡献也在 YouTube 等平台上进行直播和分享，其中一段视频详细介绍了如何将 OI 与 GROQ API 集成，以实现更低成本的运行。
- **社区协作与开发**：社区正在积极讨论对 OI 的贡献，为对 Raspberry Pi 等硬件感兴趣的新用户提供帮助，并分享他们的配置。一位用户提到 OI 在 **GitHub 上的贡献者已达到 100 人**，另一位用户分享了他们编写的 [GitHub pull request](https://github.com/OpenInterpreter/open-interpreter/pull/1204)。此外，大家对分享默认配置文件以改进模型交互也表现出了兴趣。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/open-interpreter-1146610656779440188?event=1232412426557722755">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与您的朋友和社区保持紧密联系。</li><li><a href="https://tenor.com/view/que-gif-27530657">Que GIF - Que - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/scobleizer/status/1782520422678052999?s=46&t=kwbSfLYCOimQnegJhHK_iA">来自 Robert Scoble (@Scobleizer) 的推文</a>：#17：用新 AI 让全人类变得更好。Rabbit AI 设备在 1 月份席卷了消费电子展（CES），这启发了 Open Interpreter 的创始人 @hellokillian Killian Lucas 去构建一个...</li><li><a href="https://pastebin.com/ugNMQ57v">▌ 已启用 OS Control > 打开记事本并输入 "hello" 让我们开始尝试 - Pastebin.com</a>：Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个您可以将文本在线存储一段时间的网站。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/issues/1185">全新安装并重新启动时的 Bug · Issue #1185 · OpenInterpreter/open-interpreter</a>：描述运行时的 Bug。显示此警告 interpreter /opt/conda/lib/python3.11/site-packages/pydantic/_internal/fields.py:151: UserWarning: Field "model_id" has conflict with prote...</li><li><a href="https://x.com/kodjima33/status/1782492783762399700?s=46">来自 Nik Shevchenko (@kodjima33) 的推文</a>：FRIEND 成为了全球最大的开源 AI 可穿戴设备社区。为了支持开发者，我们正在推出一个 App Marketplace。您现在可以构建自己的应用，它将与该设备配合使用...</li><li><a href="https://github.com/OpenInterpreter/01/tree/main/project_management/hardware/devices/raspberry-pi">01/project_management/hardware/devices/raspberry-pi at main · OpenInterpreter/01</a>：开源语言模型计算机。通过在 GitHub 上创建账户，为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://github.com/ishank26/posts/blob/main/llama3_new.pdf">posts/llama3_new.pdf at main · ishank26/posts</a>：资源、想法和笔记。通过在 GitHub 上创建账户，为 ishank26/posts 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=FXCaJ3Ga9TE">如何更便宜地使用 Open Interpreter！（LM Studio / Groq / GPT-3.5）</a>：第 1 部分和介绍：https://www.youtube.com/watch?v=5Lf8bCKa_dE 0:00 - 设置 1:09 - 默认 GPT-4 2:36 - 快速模式 / GPT-3.5 2:55 - 本地模式 3:39 - LM Studio 5:5...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1213">更新本地配置文件以使其不使用 function calling，由 Notnaton 提交 · Pull Request #1213 · OpenInterpreter/open-interpreter</a>：保留 model = gpt4 将导致使用 function calling。大多数 LM Studio 模型不使用 function calling，导致其无法工作。描述您所做的更改：引用任何相关的 Issue（例如 "...</li><li><a href="https://pastebin.com/b0bwxmzm">(oi) C:\Users\ivan>interpreter --api_base "https://api.groq.com/openai/v1" --api - Pastebin.com</a>：Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个您可以将文本在线存储一段时间的网站。</li><li><a href="https://github.com/KoljaB/RealtimeTTS">GitHub - KoljaB/RealtimeTTS：实时将文本转换为语音</a>：实时将文本转换为语音。通过在 GitHub 上创建账户，为 KoljaB/RealtimeTTS 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/986">Jupyter 导出魔法命令，由 tyfiero 提交 · Pull Request #986 · OpenInterpreter/open-interpreter</a>：描述您所做的更改：添加了 %jupyter 魔法命令，用于将当前会话导出为 Jupyter Notebook 文件，您可以在 Google Colab 中运行。引用任何相关的 Issue（例如 "...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1204">提升 tiktoken 版本，由 minamorl 提交 · Pull Request #1204 · OpenInterpreter/open-interpreter</a>：描述您所做的更改：由于构建过程因某种原因中断，提升了 tiktoken 的版本。此 PR 修复了中断的过程。引用任何相关的 Issue（例如 "Fixes #000"）：...
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1231154192153055262)** (22 条消息🔥):

- **模型名称混淆**：一位成员表示他们误称在 Groq 和 Llama 3 70b 上运行了 Open Interpreter，但他们指的是另一个名称类似的服务器，并澄清 **01 目前在云端选项中仅支持 OAI**。
- **Llama 3 模型稳定性问题**：提到 **Llama 3 70b** 似乎比 Llama 3 8b 更不稳定，但未提供关于不稳定性的具体细节。
- **Windows 客户端故障**：几位成员在 **Windows 上使用 01** 时遇到问题，建议表明可能存在需要解决的客户端相关问题。
- **M1 Mac 上的录音困扰**：用户报告了一个问题，即在 M1 MacBook 上按下空格键无法在 01 中启动录音，而是不断输入空格；提出了各种解决方案，包括安装 **ffmpeg**、检查麦克风和终端权限，或通过 **conda** 使用特定版本的 **Python**。
- **云端兼容性请求**：一位成员表达了在云端运行 **01** 的兴趣（例如在 brev.dev 上），并询问与 Scaleway 等云服务的兼容性，强调了对跨平台支持的需求。
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1232362713162973244)** (39 messages🔥): 

- **寻求一个吸引点击的 AGI 标题**：该频道探讨了 AGI 文章的各种诱人标题，旨在平衡点击率和实质内容。辩论了诸如 "AGI Isn't real"、"AGI is religion, not science" 和 "AGI is what you want it to be" 等标题。

- **读者满意度的重要性**：Nathan 强调了服务现有读者优先于吸引新读者的重要性，表示目前的 Discord 成员无论标题是否吸引点击都会欣赏这些内容。

- **争议性论文讨论**：讨论中提到了社区内对 Sparks 论文的广泛批评，理由包括不可复现性和过度炒作。

- **辩论 AGI 的本质**：对话涉及对 AGI 的看法，一些成员认为这更多是信仰问题而非科学。提到了一篇 Business Insider 的文章，Mistral 的 CEO Arthur Mensch 在文中对科技巨头描绘的 AGI 表示怀疑。

- **AGI 定义的法律奇观**：Nathan 觉得很有趣的一点是，由于 OpenAI 和 Microsoft 之间的条款，陪审团可能不得不确定 AGI 的定义，一位社区成员建议 OpenAI 可以战略性地利用这一点来断绝与 Microsoft 的关系。

**提到的链接**：<a href="https://www.businessinsider.com/mistrals-ceo-said-obsession-with-agi-about-creating-god-2024-4?utm_source=copy-link&utm_medium=referral&utm_content=topbar">AI CEO says people&#x27;s obsession with reaching artificial general intelligence is &#x27;about creating God&#x27;</a>：Arthur Mensch 并不担心 AI 超越人类智能，但他确实担心美国科技巨头主导该领域。

  

---


**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1232022902266789948)** (44 messages🔥): 

- **Phi 系列 Benchmark 引发辩论**：社区中分享的推文强调了对 **Phi-3** 令人印象深刻的 Benchmark 结果的讨论，提到 **LLAMA 3 8B** 是一款出色的模型，而 **Phi-3 Mini** (4b)、**Small** (7b) 和 **Medium** (14b) 由于合成数据流水线而在 Benchmark 上有显著提升。人们对使用 Benchmark 评估模型表示担忧，认为在 Benchmark 上过拟合使得像 Phi-3 这样的模型在测试中表现良好，但在分布外 (OOD) 表现不佳。

- **对 Phi-3 有效性的质疑**：用户对 **Phi-3** 的完整性表示怀疑，有人将其描述为 "SUS"（可疑），其他人则批评其主要由教科书组成，这可能使其在 **MMLU** 等 Benchmark 中占据优势，而不一定能确保广泛的能力。

- **Phi-3 被评价为“一团糟 (Clusterfuck)”**：围绕 **Phi-3** 的对话批评了其评估结果的呈现方式，指出缺乏关于数据流水线的披露，以及在文档中将 matplotlib 绘图作为 JPEG 包含在内的做法值得商榷。

- **关于训练数据和 GPU 优先级的见解**：讨论揭示了对较小模型的关注可能源于 **Microsoft Research (MSR)** 的 GPU 限制，并对比了 MSR 与其他团队或组织（如 **OAI**）之间的 GPU 资源分配。

- **Phi-3 预期发布及多语言能力**：对话期待 **Phi-3** 即将在 **MIT license** 下发布，并注意到其多语言能力，表明其范围比之前认识的更广。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/sebastienbubeck/status/1782650351692476742?s=46">来自 Sebastien Bubeck (@SebastienBubeck) 的推文</a>：@itsGauravAi 好消息是，你明天就可以亲自尝试了 :-)。</li><li><a href="https://fxtwitter.com/dylan522p/status/1782461647497400324">来自 Dylan Patel (@dylan522p) 的推文</a>：LLAMA 3 8B 表现出色，但本周将被 Phi-3 mini 4b、small 7b、medium 14b 掩盖光芒，而且 Benchmark 数据简直疯狂。Synthetic data 管道相比互联网数据有了巨大改进...</li><li><a href="https://x.com/teortaxestex/status/1782499722797674781?s=46">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：@angelusm0rt1s @fchollet 我坚信，你可以通过 Phi-2 相对于 Mixtral 等明显更强大的模型的表现来评估 Benchmark。如果 Phi-2 >> Mixtral，那么你的...</li><li><a href="https://tenor.com/view/where-is-my-free-coffee-gif-25537785">Where Is GIF - Where Is My - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/nearcyan/status/1782662543858979112">来自 near (@nearcyan) 的推文</a>：涂掉了 Phi-3 论文中无关紧要的部分，以帮助大家理解为什么它在同等规模下表现如此出色</li><li><a href="https://fxtwitter.com/suchenzang/status/1782823571561279860?s=46">来自 Susan Zhang (@suchenzang) 的推文</a>：噢不，又来了</li><li><a href="https://arxiv.org/abs/2404.14219">Phi-3 技术报告：手机本地运行的高性能语言模型</a>：我们推出了 phi-3-mini，这是一个拥有 38 亿参数的语言模型，在 3.3 万亿 Token 上训练而成。根据学术 Benchmark 和内部测试，其整体性能可媲美...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1231364661522337823)** (9 条消息🔥): 

- **评估分类成为焦点**：一名成员讨论了他们研究中的 **Evals 章节**，并谈到了 MMLU 和 BIGBench 等自动化评估的即时效用，以及与 ChatBotArena 等耗时的真人评估之间的权衡。
- **基于 Perplexity 的评估的作用**：同一名成员质疑了 *基于 Perplexity 的评估*（如 AI2 的 Paloma）的作用，以及它们如何与 MMLU 等基于任务的评估进行比较。目前尚不清楚 Paloma 是仅用于训练期间的内部检查，还是作为更广泛的公开 Benchmark。
- **Benchmark 分类获得认可**：两名成员都对 MT Bench 论文中的 Benchmark 分类表示赞赏，认为它提供了一个有用的框架，尽管像 Paloma 这样的工具分类并不十分明确。
- **多数据集 Perplexity 指标在训练中的效用**：一名成员思考多数据集的 Perplexity 评估是否更多是为了在训练 Checkpoint 监控模型性能，而不是为了完成后的模型竞赛。他们寻求对这一理解的确认。
- **确认 Perplexity 的作用**：另一名成员确认，基于 Perplexity 的评估确实被用作训练期间的 Checkpoint，而不是作为已完成模型的竞赛指标，尽管这对他们来说也是一个相对较新的概念。
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1232028192353816709)** (25 条消息🔥): 

- **Discord 的隐藏宝藏**：尽管拥有 1.3 万名免费订阅者且有 250 名符合 Discord 加入资格，但只有约 **50 人加入了** 频道。计划通过季度喊话（类似于 [Ben Thompson](https://stratechery.com/) 的风格）让其价值更加显而易见。
- **深度探讨一瞥**：一名成员分享了他们对“多元化路线图”论文的分析，反馈建议该主题目前属于*常青内容*，并欢迎对 [Typefully 草稿](https://typefully.com/t/AstZhn4) 提出任何想法。
- **社区参与度差异**：一些成员提到他们喜欢潜水阅读频道中分享的内容，而另一名成员则表达了关注太多 Discord 频道的挑战。
- **转瞬即逝的推主**：一名用户对一位研究员（Ross Taylor，Galactica 的负责人）感到有趣，他经常发布有趣的推文并在几秒钟内删除，推测过去的负面反馈可能导致了这种转瞬即逝的数字存在。
- **坦诚的采访有待 NDA 明确**：主持人表示有兴趣采访 Ross Taylor，但由于潜在的 NDA 限制可能阻碍公开且信息丰富的讨论，因此也表现出一定的顾虑。

**提到的链接**：<a href="https://typefully.com/t/AstZhn4">未找到标题</a>：未找到描述

  

---


**Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1232223235249279046)** (9 条消息🔥):

- **LLM Benchmarks Discussion**: 分享了一个讨论大型语言模型 (LLM) 基准测试现状的推文链接：[current state of llm benchmarks](https://x.com/nearcyan/status/1782634477510119767)。

- **Suspicious Activity Noted**: 一位成员提到感觉 "sus"，可能暗示在特定语境下的怀疑或谨慎。

- **It's Live!**: 成员们讨论了某个未命名功能或服务上线的时间，确认其在一小时前已发布。

- **Model Updates on Hugging Face**: 注意到 Hugging Face 上已有更新，包括一个 128k 上下文长度的模型。

- **Search Web for Interesting Results**: 一位成员指出，启用网页搜索功能可能会发现一位同名为 Nathan Lambert 的澳大利亚政治家的信息。

**Link mentioned**: <a href="https://x.com/nearcyan/status/1782634477510119767">Tweet from near (@nearcyan)</a>: current state of llm benchmarks

  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1232178107838890084)** (5 messages): 

- **Instruction Tuning Gains Traction**: 一位成员强调了一篇[关于指令微调 (Instruction Tuning) 的入门博客文章](https://gaotianyu.xyz/blog/2023/11/30/instruction-tuning/)以及该领域的最新进展。该文章因其广泛的引用和叙述方式受到好评，尽管有人指出如果经过编辑会更好。
- **Getting to Grips with CRINGE**: 分享了与指令微调相关的 CRINGE loss 论文，讨论了一种使用负样本来提高模型性能的训练方法。该方法在[论文](https://arxiv.org/abs/2211.05826)中进行了详细说明，重点在于避免不安全生成和矛盾等问题。
- **LLMBar in RewardBench Utilization Noted**: 一位成员提到 LLMBar 被用于 RewardBench，这是对关于与另一个 LLM-evaluator 元基准测试相似性查询的回应。
- **Endorsement for LLM-Evaluator Benchmark Tools**: 一条评论表达了对 LLM-evaluator 元基准测试的认可，暗示了其效用。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gaotianyu.xyz/blog/2023/11/30/instruction-tuning/">Teach Llamas to Talk: Recent Progress in Instruction Tuning</a>: no description found</li><li><a href="https://arxiv.org/abs/2211.05826">The CRINGE Loss: Learning what language not to model</a>: Standard language model training employs gold human documents or human-human interaction data, and treats all training data as positive examples. Growing evidence shows that even with very large amoun...
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1231147213900484699)** (71 messages🔥🔥): 

- **Insights on Job Hunting for Engineers**: 一位成员分享了对通过传统申请方式找工作所面临挑战的担忧，强调*个人项目和强大的 GitHub 表现*更有益。他们还讨论了在获取面试和工作机会方面，*简历上有大公司名称*比实际工作内容带来的好处更令人惊讶。
- **Web-Search for Academia**: 一位研究荷马史诗的学生列出了多个学术网站（如 academia.edu 和 perseus.tufts.edu），他们通过脚本将这些网站用于网页搜索，展示了将 Command-R 连接到丰富教育资源的兴趣。
- **Cohere Outreach Request**: 一位用户请求帮助将具有 URL Grounding 功能的 Cohere Command-R 集成到 BotPress 以实现聊天功能，并表示鉴于其性能和极具竞争力的价格，许多用户可能会转向 Cohere。
- **Guidance on Cohere's Chat API Capabilities**: 出现了关于如何限制聊天模型仅在其训练范围内回答的问题。建议包括使用 **preambles** 和 **BOS/EOS tokens**，目标是将模型输出聚焦于特定主题。
- **Meetup on Variational Autoencoders by ML-Maths**: 宣布了 Matthew Bernstein 博士即将举行的关于 *VAEs 背后的数学原理*及其在单细胞基因组学中应用的演讲，邀请参与者学习这些深层的概率模型。该活动突显了社区对高级 ML 主题的兴趣。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://drive.google.com/file/d/11TiGQ-JxqmLQ-TJ24Jui8V9kXsI6QZld/view">Ken&#39;s Resume.pdf</a>: no description found</li><li><a href="https://docs.oracle.com/en/cloud/paas/autonomous-database/serverless/adbsb/sql-generation-ai-autonomous.html#GUID-3721296F-14A1-428A-B464-7FA25E9EC8F3">Using Oracle Autonomous Database Serverless</a>: Oracle Autonomous Database Select AI enables you to query your data using natural language.
</li>
</ul>

</div>
  

---

**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1231240985791696980)** (8 messages🔥): 

- **开源公告**：一个使用 **@cohere Command R+**、**@stanfordnlp DSPy**、**@weaviate_io Vector store** 和 **@crewAIInc agents** 的新型匹配应用已开源。分享了[该应用的视频和 GitHub 链接](https://x.com/anmol_desai2005/status/1781679469679325605?s=46&t=vUJbpAOoGDUfvrA5TGBjTQ)以供探索和反馈。

- **网页抓取自动化的挑战**：一位成员正在开发一个利用 **gpt-4-turbo** 识别 (selector, column) 对的**通用网页抓取工具**，但在让模型准确查找并与输入元素交互以进行选择和点击方面面临困难。

- **追求最佳性能的 Prompt IDE 工具**：提到了 Prompt Mixer，这是一个用于创建、评估和利用 AI prompts 的桌面应用程序，并介绍了其功能。它提供自动版本控制、AI 建议以及测试 prompt chains 的能力。详情请见 [Prompt Mixer 官网](https://www.promptmixer.dev/)。

- **关于 Cohere 和 BotPress 的求助**：一位用户正在寻求帮助，希望将具有 **URL Grounding (RAG)** 功能的 **Cohere Command-r** 集成到 **BotPress** 中。他们在概念上认可 Cohere，并提到如果集成成功，许多在 BotPress 中使用 ChatGPT 的用户可能会转向 Cohere。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.promptmixer.dev/">Prompt Mixer. 面向公司的 AI 开发工作室</a>：一个供经理、工程师和数据专家协作开发 AI 功能的工作空间。</li><li><a href="https://x.com/anmol_desai2005/status/1781679469679325605?s=46&t=vUJbpAOoGDUfvrA5TGBjTQ">来自 Anmol Desai (@anmol_desai2005) 的推文</a>：我们做到了。代码终于开源了。请尝试一下，我们渴望得到反馈。@weaviate_io @stanfordnlp @cohere @1vnzh @CShorten30 ↘️ 引用 Muratcan Koylan (@youraimarketer) ...
</li>
</ul>

</div>
  

---


**Cohere ▷ #[collab-opps](https://discord.com/channels/954421988141711382/1218409745380147320/1231910638087835669)** (1 messages): 

- **寻找挪威的 Cohere 合作伙伴**：一位成员询问是否有挪威公司（最好是咨询公司）具有 Cohere 使用经验，可以为他们正准备启动的项目担任参考或顾问。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1231143136332480512)** (63 messages🔥🔥): 

- **寻求 Groq/Mixtral 工具调用 (Tool Calls) 的帮助**：一位成员询问了关于在 **LangChain** 中使用 **Groq/Mixtral** 进行 Tool_calls 的技巧，并指出 **Groq** 仅限于单个工具且禁用了并行调用；他们正在考虑如何按顺序执行单个调用。
  
- **视觉模型来救场**：在讨论处理“实际场景”中的文档时，成员们建议 **LLMs** 本身是不够的，对于通用解决方案，**vision models** 是必要的。

- **使用 LLama 的图文结合**：关于向语言模型传达图像的最新方法的讨论揭示了：在 prompts 中使用**特殊的图像 token，该 token 会被视觉编码器 (vision encoder) 的输出替换**，提供 base64 编码图像以将视觉内容转换为语言可读格式。

- **实时聊天话题管理**：一位用户寻求关于**管理和分类客户与助手之间实时聊天话题**的建议，希望将聊天消息与现有话题关联，或在必要时创建新话题。

- **向量数据库聊天的启动界面**：作为寻求快速设置启动界面（客户可以登录并与向量数据库聊天）的一部分，推荐了 **LangChain** 以及 Groq 或 Llama 等工具，同时应用标准实践，如**使用所需的 API keys 设置 LangChain、创建登录系统并建立连接到向量数据库的聊天界面**。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://learn.deeplearning.ai/courses/advanced-retrieval-for-ai/lesson/5/cross-encoder-re-ranking">DLAI - 使用 Chroma 进行 AI 高级检索</a>：简介 · 基于 Embeddings 的检索概述 · 检索的陷阱 - 当简单向量搜索失败时 · 查询扩展 (Query Expansion) · Cross-encoder 重排序 · Embedding 适配器 · 其他技术</li><li><a href="https://python.langchain.com/docs/integrations/document_transformers/cross_encoder_reranker/">Cross Encoder 重排序器 | 🦜️🔗 LangChain</a>：本笔记本展示了如何在检索器中实现重排序器。</li><li><a href="http://localhost:11434",>">未找到标题</a>：未找到描述</li><li><a href="https://js.langchain.com/docs/integrations/chat/groq#setup>)">ChatGroq | 🦜️🔗 Langchain</a>：设置</li><li><a href="https://js.langchain.com/docs/modules/model_io/llms/quick_start#setup>)">快速入门 | 🦜️🔗 Langchain</a>：大语言模型 (LLMs) 是 LangChain 的核心组件。</li><li><a href="https://js.langchain.com/docs/use_cases/tool_use/quickstart#function-calling>))">快速入门 | 🦜️🔗 Langchain</a>：在本指南中，我们将介绍创建调用 Tools 的 Chains 和 Agents 的基本方法。Tools 可以是任何东西——API、函数、数据库等。Tools 允许我们扩展功能...</li><li><a href="https://js.langchain.com/docs/integrations/chat/google_vertex_ai#vertexai-tools-agent>)">ChatVertexAI | 🦜️🔗 Langchain</a>：LangChain.js 支持将 Google Vertex AI 聊天模型作为集成。</li><li><a href="https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm#code-generation-chat-models>)">ChatVertexAI | 🦜️🔗 LangChain</a>：注意：这与 Google PaLM 集成是分开的。Google 已经...</li><li><a href="https://github.com/langchain-ai/langchain/issues/13442>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1231345297452564531)** (9 条消息🔥): 

- **用于结构化网页数据的 GitHub 项目**：Mishushakov 介绍了一个名为 [LLM Scraper](https://github.com/mishushakov/llm-scraper/) 的新 GitHub 项目，它可以利用大语言模型 (LLMs) 将任何网页转换为结构化数据。鼓励社区在 GitHub 上为该项目点亮 star。
  
- **请求 Product Hunt 排名支持**：Anthology_ 寻求社区支持，希望他们的 AI 工具 [AllMind AI: Your Personal Stock Analyst](https://www.producthunt.com/posts/allmind-ai-your-personal-stock-analyst) 能在 Product Hunt 上排名第一。该工具目前排名第 5，与其它模型相比，能提供更快、更便宜的财务洞察。
  
- **WhyHow.AI 发布知识图谱 SDK**：Chiajy 宣布了 WhyHow.AI 的重大升级，推出了由 schema 控制的自动化知识图谱，可对用户上传的内容进行数据结构化。分享了 Beta 计划和集成功能的细节，以及 [Medium](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3) 上的介绍文章链接。
  
- **寻求实时聊天分析的社区建议**：Dewhysky 正在寻求关于在实时客户端和助手聊天中管理主题/项目/任务的建议，目标是将消息与现有主题关联，或根据需要创建新主题。
  
- **LLMs 服务器配置咨询**：Vijay187 询问了使用大语言模型的服务器要求，ansh_ai 指出运行 llama 3 70b 需要两块 80GB 显存的 A100 GPU。
  
- **了解 LLMs 中的水印技术**：Wisewander 分享了一个关于大语言模型水印的资源，这涉及在 ChatGPT 或 Claude 等 AI 模型生成的文本中嵌入可识别的模式，详见 [Watermarking LLMs](https://watermarking.aisimplyexplained.tech/)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://watermarking.aisimplyexplained.tech/">AI Simply Explained</a>：AI 简单解释</li><li><a href="https://www.producthunt.com/posts/allmind-ai-your-personal-stock-analyst"> AllMind AI: Your Personal Stock Analyst - 具备实时市场数据和洞察的 AI 财务分析师 | Product Hunt</a>：AllMind AI 是您的私人财务分析师，直接为您提供集中、实时、可操作的洞察。我们的专有 LLM AllMind AI 可缩短 90% 的研究时间并降低 98% 的成本。W...</li><li><a href="https://github.com/mishushakov/llm-scraper/">GitHub - mishushakov/llm-scraper: 使用 LLMs 将任何网页转换为结构化数据</a>：使用 LLMs 将任何网页转换为结构化数据。通过在 GitHub 上创建账号，为 mishushakov/llm-scraper 的开发做出贡献。
</li>
</ul>

</div>
  

---

**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1231989652353843331)** (1 条消息): 

- **在 Langchain 中桥接自然语言与结构化查询**：一位成员在一篇[博客文章](https://rito.hashnode.dev/rental-apartment-search-with-langchain-self-querying-retriever)中详细介绍了 **Self-querying retriever** 的工作原理，讨论了 Large Language Models (LLMs) 和 few-shot prompts 如何从自然语言构建结构化查询。Self-querying retriever 通过根据元数据（metadata）为结果添加过滤功能，增强了语义相似度搜索。

**提到的链接**：<a href="https://rito.hashnode.dev/rental-apartment-search-with-langchain-self-querying-retriever">Building a Rental Apartment Search with Langchain's Self-Querying Retriever</a>：在这篇博文中，我们深入探讨了 Langchain 的 self-querying retriever 的功能，这是一个弥合自然语言与结构化数据检索之间鸿沟的强大工具。

  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1231579424420270110)** (26 条消息🔥): 

- **辩论 tinygrad 的未来**：成员们讨论了 tinygrad/box/chip 是否会转向成为[云服务](https://www.eetimes.com/groq-ceo-we-no-longer-sell-hardware/)，引用了关于 AI 和云服务的观点，并对去中心化与基于云的 AI 服务发表了各种看法。
- **TinyBox 作为 AI 家用电器**：**TinyBox** 的愿景是作为运行高级 AI 模型的**家用电器**，本地设备可以与其交互，从而绕过对云端服务器的需求并解决审查问题。
- **便携式 AI 算力 vs 云端可扩展性**：辩论继续比较了像 TinyBox 这样的本地高端 AI 硬件与云服务的效率，强调了消费者间歇性使用 AI 的情况以及当前 AI 硬件的局限性。
- **本地 AI 训练未来的重要性**：一位用户预测模型很快将在用户数据上进行*实时*训练，并强调随着模型从更小的数据集中学习，本地训练硬件的相关性将日益增加。
- **tinygrad 开发者的周会要点**：**George Hotz** 概述了周会的关键讨论点，包括 *mlperf* 的进展、潜在的 *NVIDIA CI* 计划，以及将 tinygrad 代码库维持在 7500 行以下。

**提到的链接**：<a href="https://tiny-tools-client.vercel.app">React App</a>：未找到描述

  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1231256105963749378)** (45 条消息🔥): 

- **tinygrad 配合 ROCm 的障碍**：一位成员尝试在 ROCm 上设置 tinygrad 但遇到了 segfaulting，正在寻求 ROCm 6.1 发布后的指导。
- **在 tinygrad 中堆叠张量 (Tensors)**：在详细解释中，一位成员阐明了 tinygrad 中的 `.stack` 确实通过沿新维度堆叠来 realize 张量，而必须显式调用 `.realize()` 才能在内存中实例化计算。
- **tinygrad Master 分支的稳定性**：George Hotz 确认 tinygrad 的 `master` 分支由于强大的 CI 流程应该是稳定且可靠的，回应了成员对安装和功能的担忧。
- **CUDA 兼容性与 Windows 限制**：成员们讨论了在 Windows 上通过 CUDA 使用 tinygrad 的挑战和解决方法，包括 WSL 和 Docker 方法，而另一位成员确认 Windows 尚未获得官方支持。
- **关于 tinygrad 机制的深入指导**：几位成员交流了理解 tinygrad 深层方面的资源，例如内存管理、shape tracking 以及 in-place operations 的处理，引发了关于实现细节和文档贡献的讨论。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html">ShapeTracker 如何工作</a>：tinygrad 教程</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/uops-doc.md">tinygrad-notes/uops-doc.md (main 分支) · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账号为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/cuda-tensor-core-pt1.md">tinygrad-notes/cuda-tensor-core-pt1.md (main 分支) · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账号为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/blob/37f8be6450b6209cdc9466a385075971e673c653/tinygrad/tensor.py#L169">tinygrad/tinygrad/tensor.py (版本 37f8be6450b6209cdc9466a385075971e673c653) · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://meta.ai">Meta AI</a>：使用 Meta AI 助手处理事务，免费创建 AI 生成的图像，并获取任何问题的答案。Meta AI 基于 Meta 最新的 Llama 大语言模型构建，并使用了 Emu...
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1232247102399320084)** (5 条消息): 

- **Llama3 vs. Mixtral 对决**：提到了一个针对 **Llama3 70b instruct** 的德语 RAG 评估，但根据[此数据集](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval)，它的表现似乎不如 **Mixtral-8x7B-Instruct-v0.1**。

- **指标差异受到质疑**：一名成员对为什么“问题到上下文（question to context）”指标与其他指标相比存在巨大差异表示担忧。他们建议使用 *"loglikelihood_acc_norm_nospace"* 可能会解决导致这些差异的格式问题。

- **发现潜在的格式 Bug**：强调了查询模板中可能存在格式 Bug，特别是缺少 "Answer:" 部分，这可能会影响评估结果。他们参考了相关的 [GitHub 源码](https://github.com/huggingface/lighteval/blob/11b48333b46ecd464cc3979de66038c87717e8d6/src/lighteval/tasks/tasks_prompt_formatting.py#L83)以进行澄清。

- **请求 Command-R-Plus 对比**：请求对 **Llama3 70b instruct** 和 **command-r-plus** 进行对比，以评估它们各自的性能。

- **分享 DiscoLM German 7b 评估详情**：一名成员分享了 **DiscoLM German 7b** 的详细评估结果，指出在 4 个类别中有 3 个类别比之前分享的结果有显著提升，并在此处提供了[性能对比](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval#discoresearchdiscolm_german_7b_v1)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/huggingface/lighteval/blob/11b48333b46ecd464cc3979de66038c87717e8d6/src/lighteval/tasks/tasks_prompt_formatting.py#L83">lighteval/src/lighteval/tasks/tasks_prompt_formatting.py (版本 11b48333b46ecd464cc3979de66038c87717e8d6) · huggingface/lighteval</a>：LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。</li><li><a href="https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval#discoresearchdiscolm_german_7b_v1">deutsche-telekom/Ger-RAG-eval · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval#meta-llamameta-llama-3-70b-instruct">deutsche-telekom/Ger-RAG-eval · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1231539029724631132)** (6 条消息): 

- **创新的聊天机器人执行策略**：Armifer91 正在尝试将聊天机器人功能分类，并实现一个名为 "execute_model" 的函数来处理功能组的执行，这一策略灵感来自 **MoE (Mixture of Experts)** 模型，但针对商业应用进行了调整。由于 Prompt 过长，他们对商业可行性感到担忧，并正在探索通过 Embedding 函数动态提供功能，以避免 Prompt 过长。

- **Haystack 框架增强聊天机器人**：Vladimir0583 指出，**Haystack LLM** 框架可以通过将服务索引为 OpenAPI 规范，帮助根据用户意图动态调用服务。提供了一个详细介绍该方法的 GitHub notebook：[Haystack RAG 服务演示 Notebook](https://github.com/vblagoje/notebooks/blob/main/haystack2x-demos/haystack_rag_services_demo.ipynb)。

- **寻求 Llama 微调的新 token**：Sinan2 询问了如何为 Llama 添加新的特殊 token 以进行微调，想知道是简单地编辑 tokenizer 的 JSON 文件并训练即可，还是过程更复杂。

- **对平台停机的沮丧**：_jp1_ 表达了不满，暗示 **Hugging Face 平台** 宕机了，随后 Maxidl 的评论表示这次中断破坏了晚上的活动。

**提到的链接**：<a href="https://github.com/vblagoje/notebooks/blob/main/haystack2x-demos/haystack_rag_services_demo.ipynb">notebooks/haystack2x-demos/haystack_rag_services_demo.ipynb at main · vblagoje/notebooks</a>：通过在 GitHub 上创建账号来为 vblagoje/notebooks 的开发做出贡献。

  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1231191792150511676)** (45 messages🔥): 

- **DiscoLM 德语微调挑战**：成员们讨论了在德语基准测试上微调 **DiscoLM** 的局限性，指出如果没有大量的示例和相关数据，基准测试分数可能会下降。提到了 DiscoLM 的 tokenization 问题，并提出了变通方案，例如使用 **Instruct** 等其他模型作为基础。

- **实验 Whisper 模型**：针对德语自动语音识别，建议试用 [whisper-tiny-german](https://huggingface.co/primeline/whisper-tiny-german)、[whisper-base-quant-ct2](https://huggingface.co/jvh/whisper-base-quant-ct2/) 和 [AISAK-Listen](https://huggingface.co/aisak-ai/aisak-listen) 等模型，并就进一步微调或量化以获得更好的质量和智能手机兼容性提供了额外建议。

- **对话模板与 Tokenizer 困惑**：随后讨论了 **Llama-3** 模型中模板和 tokenizer 的复杂性。强调虽然使用 **ChatML** 模板是标准做法，但 tokenizer 配置带来了挑战，包括特殊 token 的权重为零以及对话轮次的替代 eos_tokens。

- **排查模型生成错误**：为一位在让 **DiscoLM German** 生成正确响应方面遇到挑战的成员提供了帮助。建议包括在不使用 attention mask 的情况下使用 `generate` 函数，以及利用文本生成 pipeline 以简化应用。

- **Llama3 性能与输出质量**：成员们辩论了提高 Llama3 德语性能的潜力，思考瓶颈是计算还是时间。建议重复 **LeoLM** 风格的训练，并联系 **occiglot** 团队寻求帮助，同时评估 Llama3 70b 模型的跨语言能力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/cstr/llama3-discolm-orca">cstr/llama3-discolm-orca · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/jvh/whisper-base-quant-ct2/">jvh/whisper-base-quant-ct2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/primeline/whisper-tiny-german">primeline/whisper-tiny-german · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/aisak-ai/aisak-listen">aisak-ai/aisak-listen · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1231218979809529906)** (53 messages🔥): 

- **使用 Rope 扩展上下文窗口**：成员们讨论了目前缺乏使用 **rope** 来扩展大语言模型上下文窗口的提供商，一些人对这种方法表示感兴趣。通过 [Perplexity AI 链接](https://www.perplexity.ai/search/why-not-scale-0KMvWZKqSVGnYIBd_vpcng) 提供了背景信息。

- **高质量网页数据发布：FineWeb**：讨论了包含 15 万亿 token 网页数据的 **FineWeb** 的发布，并发布了 [Twitter](https://x.com/gui_penedo/status/1781953413938557276?s=46) 链接。据称 **FineWeb** 在模型性能上超过了之前的 RefinedWeb 和 C4 等数据集。

- **Hydra 框架引发不同反响**：AI 社区分享了使用 Facebook Research 的 **Hydra 框架** 的经验，该框架旨在优雅地配置复杂应用。一些人发现它在管理机器学习实验方面非常出色（[Hydra 的 GitHub 链接](https://github.com/facebookresearch/hydra)），而另一些人则质疑其独特性。

- **Phi-3 备受关注**：关于微软发布 **Phi-3** 的传闻甚嚣尘上，它是 Phi-2 的继任者，共有三个版本，体积都更大。对话包括一条 [关于 Phi-3 的推文](https://x.com/arankomatsuzaki/status/1782594659761389655?s=46&t=90xQ8sGy63D2OtiaoGJuww)，并对其与 **llama 3 8B** 等其他模型相比的性能进行了推测。

- **Perplexity.ai 融资成功**：提到了最近关于 **Perplexity.ai** 的融资公告，一些用户相比传统搜索引擎更青睐它。融资推文可以在[这里](https://x.com/AravSrinivas/status/1782784338238873769)找到。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/gui_penedo/status/1781953413938557276?s=46">Guilherme Penedo (@gui_penedo) 的推文</a>：我们刚刚发布了 🍷 FineWeb：15 万亿 token 的高质量网络数据。我们过滤并去重了 2013 年至 2024 年间所有的 CommonCrawl。在 FineWeb 上训练的模型表现优于 RefinedWeb, C4, ...</li><li><a href="https://arxiv.org/abs/2404.14219">Phi-3 技术报告：手机本地的高性能语言模型</a>：我们介绍了 phi-3-mini，这是一个拥有 38 亿参数、在 3.3 万亿 token 上训练的语言模型，根据学术基准和内部测试衡量，其整体性能媲美 ...</li><li><a href="https://arxiv.org/abs/2404.11483">AgentKit：使用图进行流工程，而非编码</a>：我们为多功能 Agent 提出了一个直观的 LLM 提示框架 (AgentKit)。AgentKit 提供了一个统一框架，用于从简单的 n... 显式构建复杂的“思考过程”。</li><li><a href="https://x.com/agihippo/status/1782828359573205295?s=46&t=90xQ8sGy63D2OtiaoGJuww">yi 🦛 (@agihippo) 的推文</a>：phi 是一个很好的试金石，可以区分谁理解 LLM，谁不理解。</li><li><a href="https://x.com/AravSrinivas/status/1782784338238873769">Aravind Srinivas (@AravSrinivas) 的推文</a>：很高兴宣布我们以 10.4 亿美元的估值筹集了 6270 万美元，由 Daniel Gross 领投，还有 Stan Druckenmiller, NVIDIA, Jeff Bezos, Tobi Lutke, Garry Tan, Andrej Karpathy, Dylan Field, Elad Gil, ...</li><li><a href="https://x.com/arankomatsuzaki/status/1782594659761389655?s=46&t=90xQ8sGy63D2OtiaoGJuww">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：Microsoft 刚刚发布了 Phi-3 - phi-3-mini：3.8B 模型，在 3.3T token 上训练，媲美 Mixtral 8x7B 和 GPT-3.5 - phi-3-medium：14B 模型，在 4.8T token 上训练，MMLU 达到 78%，MT-bench 达到 8.9  http...</li><li><a href="https://github.com/facebookresearch/hydra">GitHub - facebookresearch/hydra: Hydra 是一个用于优雅配置复杂应用程序的框架</a>：Hydra 是一个用于优雅配置复杂应用程序的框架 - facebookresearch/hydra</li><li><a href="https://github.com/facebookresearch/mbrl-lib">GitHub - facebookresearch/mbrl-lib: 用于 Model Based RL 的库</a>：用于 Model Based RL 的库。通过在 GitHub 上创建账号为 facebookresearch/mbrl-lib 做出贡献。</li><li><a href="https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/examples/conf/dynamics_model/gaussian_mlp_ensemble.yaml">mbrl-lib/mbrl/examples/conf/dynamics_model/gaussian_mlp_ensemble.yaml at main · facebookresearch/mbrl-lib</a>：用于 Model Based RL 的库。通过在 GitHub 上创建账号为 facebookresearch/mbrl-lib 做出贡献。</li><li><a href="https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/examples/conf/main.yaml">mbrl-lib/mbrl/examples/conf/main.yaml at main · facebookresearch/mbrl-lib</a>：用于 Model Based RL 的库。通过在 GitHub 上创建账号为 facebookresearch/mbrl-lib 做出贡献。</li><li><a href="https://yaml.org/spec/1.2.2/#24-tags">YAML Ain’t Markup Language (YAML™) 版本 1.2.2</a>：未找到描述
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1232382705388228680)** (1 条消息): 

- **LLM 论文俱乐部深入探讨 TimeGPT 时间序列**：明天的美国论文俱乐部将[讨论 TimeGPT](https://lu.ma/y7olehof)，这是一篇关于时间序列的论文，作者和 <@556359685306056721> 将出席。请记得注册通知，活动将在 Zoom 而非 Discord 上举行。
- **随时了解 Latent Space 活动**：[Latent.Space](http://Latent.Space) 鼓励用户点击日历右侧上方的 RSS 图标，将活动添加到他们的日历中。悬停时会显示“Add iCal Subscription”以便轻松跟踪活动。

**提到的链接**：<a href="https://lu.ma/y7olehof">LLM 论文俱乐部（与作者一起探讨 TimeGPT 论文）· Zoom · Luma</a>：本周 @Vibhu 邀请了 Nixtla 来介绍 TimeGPT：https://arxiv.org/abs/2310.03589 同时请为我们的下一篇论文提交建议并投票：…

  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/)** (1 条消息): 

alan_95125: Selfcheck，根据定义，Evaluator 和 Evaluatee 模型是相同的。
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1231159237258776597)** (24 条消息🔥):

- **推荐使用 Llama 3 70b 而非 8b**：一位用户表示倾向于使用 **llama 3 70b**，因为他们无法在 Llamafile 上运行 8b 版本。提到 70b 的 Q2 权重仅为 26GB。
- **量化怪癖**：一名用户报告了在 M1 Pro 系统上使用 llama 模型的 Q2 变体时出现的问题，导致输出乱码。另一名用户指出该模型在**纯 CPU 模式**下可以运行，尽管速度较慢。
- **Android 计划受限于地址空间**：讨论了在 Android 上运行 llamafile 的兴趣，但解释说如果没有 **47 bit address space**，Android 支持是不可能的。
- **Redis 发明者认可 Llamafile**：Redis 的创作者在 Twitter 上分享了对 llama3 70b llamafile 的正面评价，Llamafile 团队对此表示庆祝。
- **多模态端口管理**：一位用户询问如何控制模型运行的端口，以便同时运行多个 llamafile 实例，另一位用户建议使用 `--port` 标志来实现。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discordapp.com/channels/1089876418936180786/1089876419926032399/1224854113674592286">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的朋友和社区保持紧密联系。</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/.devops/main-vulkan.Dockerfile">llama.cpp/.devops/main-vulkan.Dockerfile at master · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1232172647153008730)** (3 条消息): 

- **4chan 对 Context Size 的见解**：一位成员提到了来自 4chan 的断言，暗示某种 AI 一直以来都拥有 *32k context*，并对这一发现表示惊讶。

- **Alpin 对 Scaling 的看法**：讨论包括一位成员总结了 Alpin 的缩放方法，谈到了使用 *dynamic ntk* 和 *linear scaling* 而不使用 RoPE，但坚持认为这仍然有效。

- **Matt 的长上下文 AI 配置**：该成员分享了 Hugging Face 上 Llama 模型的 **Matt 16k 配置**链接，提供了一个包含 "max_position_embeddings": 16000 和 "model_type": "llama" 等参数的 JSON 片段。点击[此处](https://huggingface.co/mattshumer/Llama-3-8B-16K/blob/main/config.json)访问文件。

**提到的链接**：<a href="https://huggingface.co/mattshumer/Llama-3-8B-16K/blob/main/config.json">config.json · mattshumer/Llama-3-8B-16K at main</a>：未找到描述

  

---


**Skunkworks AI ▷ #[datasets](https://discord.com/channels/1131084849432768614/1131669182124138616/)** (1 条消息): 

noob_master169: 针对冷门语言的 OCR 数据集？主要寻找文档类型的数据。
  

---


**Skunkworks AI ▷ #[finetuning](https://discord.com/channels/1131084849432768614/1131669354912678028/1232163104578863195)** (10 条消息🔥): 

- **寻求医学知识的简化**：一位医生科学家询问关于 Fine-Tuning 一个 LLM，以便以小学六年级的阅读水平解释复杂的遗传和医学信息。他们有兴趣为受教育程度较低的患者调整解释过程。
- **Agent 系统优于微调**：有人建议，与其立即 Fine-Tuning 模型，不如开发一个 Agent 系统，通过专门的阶段来管理任务，将其比作公司工作流。
- **从医学术语到通俗语言**：建议进一步详述了一个多阶段方法：使用增强了医学本体的现有模型理解医学化验结果，在专业水平上进行总结，然后将摘要翻译成六年级水平。
- **数据驱动的微调方向**：最后的建议是利用最强的可用模型来收集输入和输出，在生产环境中运行足够时间后，可能会产生足够的数据，从而针对简化医学信息的特定任务进行定向 Fine-Tuning。
- **对 Agent 效率感到惊讶**：询问者对使用 Agent 完成任务的建议感到惊讶，之前他们认为必须通过 Fine-Tuning 才能实现医学内容的理想简化。
  

---


**Skunkworks AI ▷ #[moe-main](https://discord.com/channels/1131084849432768614/1139310171076706464/)** (1 条消息): 

getovahit: 喜欢这个！感谢分享你的工作。
  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1231679501134467193)** (3 条消息):

- **对 Meta AI 'Imagine' 的兴奋**：一位成员对 **Meta AI 的 'Imagine'** 表达了极大的热情，称其效果**令人惊叹**。
- **征集 Imagine 示例**：在对 **Meta AI Imagine** 的兴奋之余，另一位成员请求提供示例，以说明其功能或产出效果。
- **寻找 LLM 开发工具**：一位成员寻求关于开发 **Large Language Models (LLMs)** 时常用或首选的**开发工具**建议。
  

---


**LLM Perf Enthusiasts AI ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/1232022132842561688)** (5 messages): 

- **困扰于 Azure OpenAI 的延迟**：一位成员描述了在使用 Azure 的 OpenAI 时遇到的严重延迟问题，某些请求耗时高达 20 分钟。
- **速率限制的烦恼**：另一位成员表达了对 Azure 频繁触发速率限制（Rate Limit）的沮丧，仅在 15 秒内发送两个请求就触发了退避策略。
- **Azure 延迟的可能原因**：一位成员指出，由于已报告的服务问题，Azure 的延迟问题可能仅限于今天。
- **追踪 API 响应时间**：来自 [GPT for Work](https://gptforwork.com/tools/openai-api-and-other-llm-apis-response-time-tracker) 的分享链接提供了主要大语言模型（包括 OpenAI 和 Azure OpenAI）API 响应时间的实时追踪，并提供了如何获得更快响应时间的建议。

**提到的链接**：<a href="https://gptforwork.com/tools/openai-api-and-other-llm-apis-response-time-tracker">OpenAI API 和其他 LLM API 响应时间追踪器</a>：未找到描述

  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1232013872261758996)** (2 messages): 

- **建筑领域中的蓝图 AI**：一位成员分享称，一家大型建筑公司正将 AI 作为“预检”工具，用于识别建筑图纸中的潜在问题和违反规范之处。然而，该公司尚未在蓝图设计阶段采用 AI 生成内容。
- **寻求用于蓝图解析的 AI**：讨论还涉及探索用于解析蓝图的 AI 模型或方法，特别是专注于追踪 PDF 图纸中的管道系统（ductwork）。对话中未提供具体的模型或解决方案。
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1231997043082268783)** (2 messages): 

- **Llama 3 隆重登场**：[Llama 3](https://llama.meta.com/llama3/) 正式发布，并在 [LMSYS arena 排行榜](https://chat.lmsys.org/?leaderboard)上并列第 5 位，紧随 Claude 3 Opus 和某些 GPT-4 变体之后，展现了令人印象深刻的结果。这款开源许可模型甚至可以在高端笔记本电脑上运行。

- **SimonW 发布 Llama 3 工具**：Simon Willison 介绍了 [LLM](https://llm.datasette.io/)，这是一个命令行工具和 Python 库，旨在简化对 Llama 3 及许多其他模型的访问。他的博客文章详细介绍了通过托管版本和本地硬件访问 Llama 3 的多种方式，[在此处突出显示](https://simonwillison.net/2024/Apr/22/llama-3/)。

- **请求 Hackernews 摘要生成器**：一位成员正在寻找最新版本的 Hackernews 摘要生成器，他们记得曾见过其以 bash 脚本的形式存在。

**提到的链接**：<a href="https://simonwillison.net/2024/Apr/22/llama-3/">使用 LLM 从终端访问 Llama 3 的选项</a>：Llama 3 已于周四发布。早期迹象表明，它目前是表现最好的开源许可模型——Llama 3 70b Instruct 在 LMSYS arena 中并列第 5 位……

  

---



**AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1232147938097233940)** (4 messages): 

- **General 频道垃圾信息警报**：该频道出现了多条带有 Discord 邀请链接的垃圾信息，推广不当内容。
- **对 Jamba 需求的关注**：一位成员询问了 **Jamba** 与 **LM Studio** 的兼容性及其运行要求，鉴于其号称拥有类似于 **Claude** 的内存容量。
- **讨论运行 Jamba 的挑战**：讨论了由于高 RAM 需求而导致运行 **Jamba** 的困难，提到 Google Colab 无法提供足够的资源，且在 Google Cloud 上的尝试也未成功。

**提到的链接**：<a href="https://discord.gg/kYyKmR6U">加入 NSFW // 18 🍑🍒 Discord 服务器！</a>：查看 Discord 上的 NSFW // 18 🍑🍒 社区 - 与其他 31716 名成员一起交流，享受免费的语音和文字聊天。

  

---