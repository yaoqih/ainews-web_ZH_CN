---
companies:
- apple
- meta-ai-fair
- google
date: '2024-04-26T21:32:41.171695Z'
description: '**苹果公司**通过发布 **OpenELM** 进一步推进了其在 AI 领域的布局。这是苹果首个相对开放的大语言模型，参数规模从 **2.7
  亿到 30 亿**不等，采用了受 **DeLight** 论文启发的新型逐层缩放架构。与此同时，**Meta 的 LLaMA 3** 系列突破了上下文长度的界限，其模型支持超过
  **16 万个 token**，并在 Hugging Face 上发布了具有 **26.2 万上下文长度的 8B-Instruct 模型**，同时量化版本的性能也得到了提升。


  一篇关于 AI 对齐的新论文指出，**KTO** 是目前表现最好的方法，并提到了该方法对训练数据量的敏感性。在 AI 伦理与监管方面，前**谷歌** CEO **埃里克·施密特
  (Eric Schmidt)** 警告称，开源 AI 存在助长不法分子和地缘政治对手的风险；同时，美国的一项提案旨在强制执行“了解你的客户”（KYC）规则，以终结匿名云端服务的使用。'
id: 5491a042-7431-4c36-8ef2-8837bf3bce30
models:
- openelm
- llama-3
- llama-3-8b-instruct
- llama-3-70b
original_slug: ainews-apples-openelm-beats-olmo-with-50-of-its
people:
- eric-schmidt
- sebastian-raschka
title: 苹果的 OpenELM 采用 DeLighT 架构，仅使用 50% 的数据集便击败了 OLMo。
topics:
- layer-wise-scaling
- context-length
- quantization
- ai-alignment
- open-source
- ai-regulation
---

 

---

**目录**

[TOC] 


---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**LLaMA 进展**

- **LLaMA 3 将上下文扩展至 160K+ tokens**：在 /r/LocalLLaMA 中，LLaMA 3 将上下文长度增加到 [**超过 160K tokens，同时保持完美召回 (perfect recall)**](https://www.reddit.com/r/LocalLLaMA/comments/1ccqmjz/llama_3_now_with_160k_context/)。评论者指出，这令人印象深刻，但要在本地以理想速度运行将需要强大的消费级硬件。Meta 的 Llama 3 下载量已超过 120 万次，在 Hugging Face 上有超过 600 个衍生模型。
- **首个具有 262K 上下文的 Llama-3 8B-Instruct 模型发布**：在 /r/LocalLLaMA 中，首个具有 [**超过 262K 上下文长度的 Llama-3 8B-Instruct 模型在 Hugging Face 上发布**](https://www.reddit.com/r/LocalLLaMA/comments/1cd4yim/llama38binstruct_with_a_262k_context_length/)，实现了超越简单提示词的高级推理。
- **Llama 3 70B 表现优于 8B 模型**：在 /r/LocalLLaMA 中，对比显示 [**量化后的 Llama 3 70B IQ2_XS 表现优于未压缩的 Llama 3 8B f16 模型**](https://www.reddit.com/r/LocalLLaMA/comments/1cda0fv/llama_3_8b_f16_vs_llama_3_70b_q2/)。研究发现 70B IQ3_XS 版本最适合 32GB VRAM 用户。
- **新论文比较 AI 对齐方法**：在 /r/LocalLLaMA 中，一篇新论文将 DPO 与其他对齐方法进行了比较，发现 [**KTO 在大多数基准测试中表现最佳，且对齐方法对训练数据量非常敏感**](https://www.reddit.com/r/LocalLLaMA/comments/1ccz84a/insights_into_alignment_dpo_and_its_variants/)。

**AI 伦理与监管**

- **Eric Schmidt 警告开源 AI 的风险**：在 /r/singularity 中，前 Google CEO Eric Schmidt 警告称，[**开源 AI 模型为恶意行为者和中国提供了危险的能力**](https://www.reddit.com/r/singularity/comments/1ccyqkr/former_google_ceo_eric_schmidt_warns_that_open/)。许多人认为这是大型科技公司企图扼杀竞争的行为，并指出中国很可能拥有无需依赖开源即可开发强大模型的能力。
- **美国提案旨在终结匿名云端使用**：在 /r/singularity 中，一项 [**美国提案寻求实施“了解您的客户 (Know Your Customer)”要求，以终结匿名云端使用**](https://www.reddit.com/r/singularity/comments/1ccr2ub/us_know_your_customer_proposal_will_put_an_end_to/)。
- **巴尔的摩教练涉嫌使用 AI 进行诽谤**：在 /r/OpenAI 中，一名巴尔的摩教练涉嫌 [**利用 AI 语音克隆技术生成虚假的种族主义音频，企图让一名高中校长被解雇**](https://www.reddit.com/r/OpenAI/comments/1cd5h9c/baltimore_high_school_athletic_director_used_ai/)。

**硬件进展**

- **台积电 (TSMC) 发布 1.6nm 工艺节点**：在 /r/singularity 中，台积电宣布了 [**具有背面供电 (backside power delivery) 技术的 1.6nm 工艺节点**](https://www.reddit.com/r/singularity/comments/1ccr4hy/tsmc_unveils_16nm_process_technology_with/)，使硬件在未来几年能够继续保持指数级增长。
- **超薄太阳能电池助力自充电无人机**：在 /r/singularity 中，德国研究人员开发出 [**超薄、柔性太阳能电池，允许小型无人机在运行期间自行充电**](https://www.reddit.com/r/singularity/comments/1ccr6aq/german_researchers_have_developed_a_solar_cell/)。
- **美光 (Micron) 获得 61 亿美元《芯片法案》资金**：在 /r/singularity 中，美光获得了 [**61 亿美元的《芯片法案》(CHIPS Act) 资金，用于在纽约州和爱达荷州建设半导体制造设施**](https://www.reddit.com/r/singularity/comments/1cd0s5k/micron_set_to_receive_61b_in_chips_act_funding_to/)。

**迷因与幽默**

- **AI 助手自信地断言地球是平的**：在 /r/singularity 中，一张幽默的图片展示了一个 [**AI 助手自信地断言地球是平的**](https://www.reddit.com/r/singularity/comments/1ccqhzv/chat_is_this_real/)，引发了关于是否需要一个能够相信荒谬事物或相信人类利益至上的 AI 的笑话。

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

以下是所提供推文中的关键主题和见解摘要：

**Meta Llama 3 发布与影响**

- **快速普及**：在发布后的一周内，Llama 3 模型在 Hugging Face 上的下载量已超过 120 万次，并产生了 600 多个衍生模型，展现了令人兴奋的早期影响力。 ([@AIatMeta](https://twitter.com/AIatMeta/status/1783602908845748685))
- **训练优化**：Meta 在优化方面进展迅速，Llama 3 70B 的训练速度提升了 18%，Llama 3 8B 的训练速度提升了 20%。 ([@svpino](https://twitter.com/svpino/status/1783888989025431933)) 
- **上下文扩展**：社区通过结合 PoSE、持续预训练和 RoPE 缩放，将 Llama 3 8B 的上下文从 8k 扩展到了近 100k tokens。 ([@winglian](https://twitter.com/winglian/status/1783842736833016289))
- **推理加速**：Colossal-Inference 现在支持 Llama 3 推理加速，使 8B 和 70B 模型的效率提升了约 20%。 ([@omarsar0](https://twitter.com/omarsar0/status/1783895931043111088))
- **基准测试表现**：Llama 3 70B 在 LMSYS 排行榜的英语查询中并列第一。 ([@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1783570318230978783))

**Phi-3 模型发布与反响** 

- **基准测试过拟合**：有人认为 Phi-3 在公开基准测试中存在过拟合，但在实际使用中表现不如 Llama-3 8B 等模型。 ([@svpino](https://twitter.com/svpino/status/1783556635543339310), [@abacaj](https://twitter.com/abacaj/status/1783898711623352686))
- **意外行为**：作为一个本质上不同的模型，Phi-3 可能会表现出令人惊讶的结果，无论好坏。 ([@srush_nlp](https://twitter.com/SebastienBubeck/status/1783885843943616524))

**扩展 LLM 上下文窗口**

- **PoSE 技术**：Positional Skip-wisE (PoSE) 方法在训练期间模拟长输入以增加上下文长度，助力 Llama 3 扩展至 128k tokens。 ([@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1783574428858696161)) 
- **Axolotl 与 Gradient AI**：像 Axolotl 这样的工具和 Gradient AI 的方法正助力 Llama 及其他模型的上下文扩展至 160k+ tokens。 ([@winglian](https://twitter.com/winglian/status/1783469196011016696), [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1783736130321408011))

**Cohere Toolkit 发布**

- **企业聚焦**：Cohere 发布了一个工具包以加速企业中的 LLM 部署，目标是利用私有数据和本地代码解释器实现安全的 RAG。 ([@aidangomez](https://twitter.com/aidangomez/status/1783533461401227563))
- **灵活部署**：该工具包的组件可以部署到任何云端，并可重复用于构建应用程序。 ([@aidangomez](https://twitter.com/aidangomez/status/1783533465960378561), [@aidangomez](https://twitter.com/aidangomez/status/1783533471777935433))

**OpenAI 员工停职与 GPT-5 猜测**

- **感知能力主张**：一名声称 GPT-5 具有感知能力的 OpenAI 员工已被 Twitter 封禁。 ([@bindureddy](https://twitter.com/bindureddy/status/1783847600824995850))
- **炒作生成**：OpenAI 被视为围绕 AGI 和 AI 感知能力主张的炒作引擎，尽管其竞争对手正以更低的成本追平 GPT-4 的表现。 ([@bindureddy](https://twitter.com/bindureddy/status/1783852748636905716))
- **Agent 能力**：一些人认为 GPT-5 将是一个“Agent GPT”，这基于在语言模型之上构建 Agent 架构所带来的性能提升。 ([@OfirPress](https://twitter.com/OfirPress/status/1783870394581074110))

**其他值得关注的主题**

- 担心 AI 峰会委员会缺乏多样化的代表性，难以应对权力集中风险。 ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1783882237764633052))
- OpenAI 与 Moderna 的合作被视为传统企业采用生成式 AI 的积极信号。 ([@gdb](https://twitter.com/gdb/status/1783529202974687527), [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1783533728846827681)) 
- Apple 开源的端侧语言模型表现不佳，但提供了有用的架构和训练细节。 ([@bindureddy](https://twitter.com/bindureddy/status/1783635037365436462), [@rasbt](https://twitter.com/rasbt/status/1783480053847736713))

---

# AI Discord 内容回顾

> 摘要的摘要之摘要

1. **扩展 LLM 上下文长度**
   - **Llama 3 性能与上下文长度创新**：讨论集中在 **Llama 3 的能力**上，一些人对其与 **GPT-4** 相比的代码召回和配置持保留意见。然而，通过 **PoSE (Positional Skip-wisE)** 等技术以及使用 3 亿 token 进行持续预训练，将 **Llama 3 8B 模型的上下文长度扩展至 96k token** 的创新引起了轰动，详见此 [推文线程](https://x.com/winglian/status/1783456379199484367?s=46&t=stOPrwZiN_fxSK0RuC8Flg)。
   - [EasyContext 项目](https://github.com/jzhang38/EasyContext) 旨在以极低的硬件要求将 LLM 上下文长度外推至 **100 万 token**。

2. **优化 LLM 训练与部署**
   - [Nvidia 的 Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#introduction) 被用于 **kernel profiling**（内核分析），以优化用于 LLM 训练的 CUDA 代码。
   - **为特定领域增益微调 LLM**：针对特定领域改进而进行的 **LLM 微调** 兴趣日益浓厚，例如用于医疗应用的 **[Meditron](https://arxiv.org/abs/2311.16079)**。讨论还涉及使用 **[Argilla 的 Distilabel](https://github.com/argilla-io/distilabel)** 等工具的**数据合成**策略，以及多文档、长上下文微调的挑战。人们还对性价比权衡进行了辩论，例如在 [4 个 epoch 花费 2,368 美元 vs 50 个 epoch 花费 41,440 美元](https://discord.com/channels/1053877538025386074/1154120232051408927/1232958591955112028) 之间，后者的收益可能微乎其微。
   - PyTorch 推出了 [Torchtitan](https://github.com/pytorch/torchtitan)，这是一个致力于辅助从零开始训练 LLM 的库。
   - [Mixture of Depths 论文](https://paper-club.ivanleo.com/papers/mixture-of-depths) 提出使用改进的 MoE 路由机制来加速 Transformer 训练。
   - **CUDA 优化深度探讨**：CUDA 开发者使用 **[NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#introduction)** 等工具深入研究内核分析，讨论了 128 字节左右的 **memory coalescing**（内存合并）和 **burst sizes**（突发传输大小），并辩论了**低比特量化**方法的效率。对话还涵盖了 PyTorch 2.3.0 的 **Flash Attention 兼容性**问题，以及 PyTorch AO 支持**自定义 CUDA 扩展**进行性能调优的影响。

3. **开源 LLM 生态系统扩展**
   - **Apple 意外进军开源模型领域**：**Apple 发布了 [OpenELM](https://huggingface.co/apple/OpenELM)**，这是一个高效的开源语言模型家族，参数量从 2.7 亿到 30 亿不等，这让 AI 社区感到意外。此举标志着 Apple 从传统的封闭式方法发生了转变，其中 270M 模型迅速在 Hugging Face 上获得关注。
   - [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B) 发布，利用 Mistral 构建预训练的医疗 LLM。
   - Mozilla 的 [llamafile 项目](https://hacks.mozilla.org/2024/04/llamafiles-progress-four-months-in/) 实现了以高性能在本地分发和运行 LLM。
   - Dify 作为一个[开源 LLM 应用开发平台](https://github.com/langgenius/dify?tab=readme-ov-file)脱颖而出，它结合了 AI 工作流和模型管理。

4. **评估与基准测试 LLM**
   - 在 [Judgemark 基准测试](https://eqbench.com/judgemark.html)上，**Llama-3-70b** 在微调 **disco-judge** 应用方面展现出潜力。
   - 围绕**验证损失 (validation loss)** 作为 LLM 性能指标有效性的讨论。
   - [低成本语言模型综述 (Low-Cost Language Models survey)](https://arxiv.org/abs/2404.11160) 评估了在 Python 代码生成任务中对 CPU 友好的 LLM。
   - 关于 **Nightshade** 自动编码器能力透明度的辩论，以及公开研究结果的必要性。

---

# 第一部分：高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **微调的磨难与胜利**：工程师们分享了他们在模型微调中遇到的挑战，涉及 4-bit quantization、超过 48GB 的 VRAM 需求以及 token index sequence length 错误。解决方案包括修订 batch size、启用 [pack](https://github.com/unslothai/unsloth#-finetune-for-free) 功能，以及考虑使用 **Aphrodite Engine** 或 **llama.cpp** 等替代测试环境。

- **传闻还是解散？**：关于 **WizardLM** 团队在 **Qingfeng Sun 的员工页面**重定向后解散的猜测甚嚣尘上。社区分享了截然不同的观点，从抢救 **WizardLM datasets** 到**展示**环节，其中 **Meta 的 LlaMA-3 模型**（包括 8B 和 70B 版本）被引用为同类产品中的佼佼者。

- **从冷藏到热门话题**：一位成员自豪地宣布了 [Kolibrify](https://github.com/oKatanaaa/kolibrify) 的**开源发布**，这是一个用于指令遵循 LLM 的课程训练工具。在技术层面，社区讨论了 **Triton** 依赖项、"Quantization failed" 错误以及 **gguf model** 测试策略，并就 **fine-tuning** 和部署选项的最佳实践达成了一致。

- **务实的剪枝进展**：有人分享了关于在评估期间运行的 **[triton laser merge trainer](https://github.com/l4b4r4b4b4/trl/tree/evol_laser_merge_trainer)** 迭代增加模型上下文长度项目的见解。该方法因无需重新初始化而被视为具有创新性，可能为增强模型可用性提供路径，而无需对系统进行彻底改革。

- **Unsloth 的里程碑与资源**：Unsloth AI 实现了在 Hugging Face 上微调框架月下载量达到 50 万次的重大里程碑，并提倡分享 **exact match** GGUF 模型，尽管可能存在冗余。重点还在于引导用户使用 **[Colab notebooks](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp)** 以获取有效的微调策略。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Siri 迎来了一位聪明的伙伴**：Perplexity AI Discord 聊天机器人为 **iOS 用户**推出了一项独有的音频功能，可以朗读任何问题的答案。

- **Opus 限制引发不满**：社区内对 Claude 3 Opus 交互每天 50 次的新查询限制感到沮丧，尽管有这些限制，**Perplexity chatbot 仍然支持 Opus**。

- **API 采用焦虑**：AI 工程师正在讨论 Perplexity API 的集成问题，例如响应过时和缺乏 GPT-4 支持；一位用户还就 `llama-3-70b-instruct` 模型的**最佳超参数**寻求建议。

- **模型之战**：社区对 Google 的 Gemini 模型及其对 AI 领域的潜在影响充满期待，同时指出 GPT-5 必须带来卓越的创新才能在竞争中保持领先。

- **网络中立性的水晶球**：一篇链接文章引发了关于 FCC 重建网络中立性的讨论，社区成员正在思考其对 **AI Boom** 未来的影响。



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**CUDA 集体汇聚**：成员们专注于通过优化各种 kernel 和算法（包括矩阵乘法和 Flash Attention）来磨练他们的 **CUDA** 技能。讨论主题从利用 [NVIDIA Nsight Compute CLI 用户指南](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#introduction) 进行 kernel profiling，到关于低比特量化方法效率的辩论。

**PyTorch 纠缠于兼容性与扩展**：**PyTorch 2.3.0** 中的 **flash-attn 兼容性**遇到了障碍，导致 `undefined symbol` 错误，参与者希望该问题能尽快得到纠正。PyTorch AO 通过[支持自定义 CUDA 扩展](https://github.com/pytorch/ao/pull/135)激发了热情，便于使用 `torch.compile` 进行性能调优。

**用 C++ 编写更环保的代码**：关于 **NVIDIA C++ 团队**将 `llm.c` 转换为 `llm.cpp` 的额外演讲公告预示着编写更清晰、更快速代码的机会。

**内存与模型的矩阵**：讨论深入探讨了 CUDA 最佳实践的细节，思考了 CUDA 指南**第 6 章第 3.d 节**中探讨的约 **128 bytes** 的内存合并 **burst sizes**，并尝试减少 packed 操作中开销的概念。

**录制聚会**：志愿者们提供了详细且具有操作性的屏幕录制建议，并使用 [Existential Audio - BlackHole](https://existential.audio/blackhole/download/?code=681349920) 进行无损声音采集，强调了精细技术设置所需的细微差别。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GPU Offloading 转至 AMD OpenCL**：通过将 GPU 类型切换为 **AMD Open CL**，解决了一个 **GPU Offloading** 的技术故障，这表明简单的修复即可绕过性能问题。
- **更新与性能的混合消息**：LM Studio **0.2.21 版本**出现了升级问题，导致之前运行 **phi-3 mini 模型**的设置出现故障；而其他用户在尝试使用 **2.20 版本**时遇到了 GPU 使用率激增但模型加载失败的问题。用户正在积极排查故障，包括提交截图请求以进行更好的诊断。
- **LM Studio 将聊天转化为文档利器**：关于改进 **LM Studio 聊天功能**的热烈讨论促成了使用 **RAG (Retriever-Augmented Generation)** 嵌入文档检索功能，并调整 GPU 设置以实现更好的资源利用。
- **利用图形算力攻克 AI**：社区正在分享关于最佳硬件配置的见解，以及在使用 AI 模型时预期从 Nvidia Tesla 设备获得的性能提升，这表明用户对 AI 模型托管的最佳设备有着浓厚兴趣。
- **AMD ROCm 受到关注**：**AMD 的 ROCm 技术预览版**在某些配置下表现出色，在 eGPU 系统上达到了显著的 30t/s，尽管兼容性问题凸显了根据 ROCm 文档检查 GPU 支持的重要性。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**突破模型上下文限制**：Llama 3 模型正在打破上下文壁垒，其中一个变体通过使用 PoSE 和 300M token 的持续预训练，使 **8B 模型达到了 96k 上下文**。Positional Skip-wisE (PoSE) 的有效性和 **RoPE scaling** 是关键议题，讨论中提到了关于 [PoSE 上下文窗口扩展的论文](https://openreview.net/forum?id=3Z1gxuAQrA)，以及在微调过程中调整 RoPE base 以获得更长上下文的策略。

**LLM 性能与成本讨论引发社区关注**：工程师们对验证损失（validation loss）作为性能指标表示怀疑，并分享了训练 Epochs 的成本对比，强调了一个案例：四个 Epochs 耗资 2,368 美元，而五十个 Epochs 耗资 41,440 美元，但性能提升微乎其微。另一位工程师正在考虑基于 **Gemma MoE** 将多个 8B 模型组合成一个 Mixture of Experts，并推测了使用 **DPO/ORPO 技术**可能带来的增强。

**仓库归档风波**：针对微软 WizardLM 仓库突然消失的情况，成员们表达了担忧，并引发了关于归档重要性的辩论，特别是考虑到微软对 OpenAI 的投资。参与者强调了备份的必要性，并引用了最近发布的 **WizardLM-2** 案例，该模型可在 [Hugging Face](https://huggingface.co/collections/microsoft/wizardlm-2-661d403f71e6c8257dbd598a) 和 [GitHub](https://github.com/victorsungo/WizardLM/tree/main/WizardLM-2) 上获取。

**合成数据生成：一站式解决方案**：推荐使用 *Argilla 的 Distilabel* 来创建**多样化的合成数据**，[distilabel-workbench](https://github.com/argilla-io/distilabel-workbench) 等实际示例和仓库展示了其应用。对话涵盖了单文档数据合成、多文档挑战以及语言模型扩展上下文的策略。

**模拟世界参与激发好奇心**：Websim 模拟 CLI 命令和完整网页的能力吸引了用户，分享的模拟示例包括 [Websim](https://websim.ai/c/p3pZvmAYbsRT2hzBz) 上的 **EVA AI 交互配置文件**。关于重启 World-Sim 的推测也在同步进行，成员们期待其以“按 token 付费”的模式重新推出。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Apple 凭借 OpenELM 转向开源**：Apple 发布了 **OpenELM**，这是一个高效的语言模型系列，现已在 Hugging Face 上提供，参数规模从 270M 到 3B 不等，标志着其向开源倡议的意外转变。有关模型的详细信息请见 [Hugging Face](https://huggingface.co/apple/OpenELM)。

- **关于 AI Sentience 与时间意识的讨论**：社区进行了深入讨论，强调了 **Sentience**（可能与情感和动机相关）与 **Consciousness**（与知识获取相关）之间的区别。平行的讨论思考了 AI 的智能和时间意识是否是固有的离散概念，从而影响我们对神经网络身份和体验维度的理解。

- **AI 语音助手技术交流**：AI 爱好者交流了关于用于自研语音助手开发的 **OpenWakeWords** 以及 **Gemini** 作为 Google Assistant 竞争对手的前景。提到的技术挑战包括中断 AI 语音的复杂性，以及对推按通话 (push-to-talk) 与语音激活的偏好。

- **自定义 GPT 使用中的速率限制谜题**：用户寻求关于 **GPT-4 使用上限** 的澄清，特别是在调用大型文档时，并分享了应对 3 小时滚动上限的技巧。社区正在探索速率限制的阈值，特别是在使用自定义 GPT 工具时。

- **Prompt Engineering 实力与 LLM 涌现能力**：重点关注针对特定任务的策略性 Prompt 构建，例如为 **Arma 3 的 SQF 语言** 开发基于 GPT 的编码。LLM 中的 **涌现行为 (Emergent behaviors)** 引起了极大的兴趣，这指的是导致定性行为变化的复杂阶段，并探讨了在 Prompt Engineering 背景下与 *More Is Different* 概念的相似之处。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**AI 部署必须透明**：Valve 的新 **内容政策** 要求开发者在 **Steam** 上披露 AI 的使用情况，特别强调了实时生成的 AI 内容的透明度需求以及确保负责任部署的机制。

**内容创作中的版权困境**：关于使用 **Stable Diffusion** 等公开模型生成内容时的法律复杂性讨论不断；有必要应对版权挑战，特别是在像 **Steam** 这样具有严格版权执行机制的平台上。

**艺术模仿生活还是……模仿自身？**：**Customluke** 提出了一个关于如何使用 **Stable Diffusion** 创建模型或 LoRA 来复制其艺术风格的询问，引发了各种建议，分别出现了用于模型创建的 **dreambooth** 和用于 LoRA 创建的 **kohya_ss** 等工具。

**选择更合适的 AI 版本**：一群活跃用户发现 **SD 1.5** 在他们的需求上优于 **SDXL**，理由是结果更清晰且训练过程更好，这证明了 AI 模型选择会显著影响输出质量。

**润色图像生成**：分享了改进图像生成结果的技巧，推荐了 **Forge** 和 **epicrealismXL** 等替代方案，以增强那些对 **ComfyUI** 等模型生成的图像质量不满意的输出。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **医疗 LLM BioMistral 发布**：[BioMistral](https://huggingface.co/BioMistral/BioMistral-7B) 是一套针对医疗应用的新型预训练语言模型，利用了基础 Mistral 模型的能力。

- **Nvidia 的地缘政治适应**：为了应对美国的出口管制，Nvidia 推出了 RTX 4090D，这是一款符合中国标准的 GPU，降低了功耗和 CUDA 核心数量，详情见 [The Verge](https://www.theverge.com/2023/12/29/24018799/nvidia-4090d-china-slower-us-sanctions) 和 [Videocardz](https://videocardz.com/newz/nvidia-geforce-rtx-4090-with-blower-type-cooler-is-now-on-sale-in-china) 的报道。

- **文本生成图像模型微调讨论**：关于优化文本生成图像模型的咨询引出了涉及 [Hugging Face diffusers 仓库](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) 的建议。

- **ConversationalRetrievalChain 的 Gradio 界面**：ConversationalRetrievalChain 与 Gradio 的集成正在进行中，社区正努力加入个性化 PDF 支持，并就界面定制进行讨论。

- **图像生成改进与葡萄牙语 AI 见解**：新进展包括用于消化稍后阅读内容的 [Collate.one](https://collate.one/newsletter) 应用，在 [此 Space](https://huggingface.co/spaces/KingNish/Instant-Image) 中实现数秒内生成高清图像的技术突破，以及 AI 社区亮点的 [巴西葡萄牙语翻译](https://www.youtube.com/watch?v=A9qPlYVeiOs)。

- **量化与效率**：目前正在积极探索量化技术，以在显存（VRAM）受限的系统上最大化模型效率，倾向于选择 Q4 或 Q5 级别，以平衡性能与资源管理。

- **表格视觉模型与 COCO 数据集说明**：有人请求推荐擅长基于表格进行问答的视觉模型，并对通过 HTTP 连接托管官方 COCO 数据集提出了安全担忧。

- **对以代码为中心资源的呼吁与 TLM v1.0**：工程社区正在寻求更多带有直接代码链接的工具，例如 [awesome-conformal-prediction](https://github.com/valeman/awesome-conformal-prediction)；同时庆祝 Trustworthy Language Model (TLM) v1.0 的发布，该版本引入了置信度评分功能，并提供了 [Playground](https://tlm.cleanlab.ai/) 和 [教程](https://help.cleanlab.ai/tutorials/tlm/)。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **并行思考无障碍 (Parallel Ponderings Pose No Problems)**：工程师们强调，某些**模型架构**（特别是 PaLM）采用了**并行注意力和 FFN (feedforward neural networks)**，这与一些论文中呈现的串行感知有所不同。

- **数据消化细节 (Data Digestion Detailing)**：分享了 **Pile 数据集的哈希值**，为那些希望在各种 JSON 文件中使用该数据集的人提供了参考，相关辅助信息可在 [EleutherAI 的哈希列表](https://www.eleuther.ai/hashes)中找到。

- **滑动窗口内的思考 (Thinking Inside the Sliding Window)**：关于 **Transformers** 的对话探讨了**滑动窗口注意力 (sliding window attention)**和有效感受野，将其类比为卷积机制及其对注意力焦点的平衡影响。

- **层级学习阶梯增加灵活性 (Layer Learning Ladders Lengthen Leeway)**：关于改进 Transformers 处理**更长序列长度**的讨论涉及了一些策略，如集成 RNN 类型层或在架构中使用扩张窗口 (dilated windows)。

- **PyTorch 的新强力工具 (PyTorch's New Power Player)**：通过 [GitHub 链接](https://github.com/pytorch/torchtitan)介绍了一个新的 **PyTorch 库 torchtitan**，旨在简化训练大型模型的过程。

- **线性逻辑阐明推理 (Linear Logic Illuminates Inference)**：剖析了**线性注意力 (linear attention)**的机制，说明了其序列长度的线性相关性和恒定的内存占用，这些是未来模型优化的重要见解。

- **性能对等假设 (Performance Parity Presumption)**：一位工程师报告称 **phi-3-mini-128k** 可能与 **Llama-3-8B** 旗鼓相当，这引发了关于预训练数据对模型基准测试 (benchmarking) 和基准线影响的讨论。

- **Delta 决策的双重性质 (Delta Decision's Dual Nature)**：**Delta 规则线性注意力 (delta rule linear attention)** 能够实现更具结构化但并行化程度较低的操作，这一可能性引发了对比辩论，并得到了 [MastifestAI 博客文章](https://manifestai.com/blogposts/faster-after-all/)的支持。

- **微观视角下的测试 (Testing Through a Tiny Lens)**：成员们对长上下文语言模型的“大海捞针 (needle in the haystack)”测试表示怀疑，主张将现实世界的应用作为更稳健的性能指标。

- **关于 Prompt Loss 的思考 (Prompt Loss Ponderings)**：小组质疑了在监督微调 (SFT) 期间屏蔽用户 Prompt Loss 的系统性研究，指出尽管这在语言模型训练中经常使用，但仍存在研究空白。

- **5 是 GSM8K 的神奇数字 (Five is the GSM8K Magic Number)**：达成共识认为，使用 *5* 个 few-shot 示例是符合 **Hugging Face 排行榜**中 **GSM8K** 评测标准的做法。

- **VLLM 版本剖析 (VLLM Version Vivisection)**：对话指出 **数据并行 (DP)** 是将 **VLLM** 更新到最新版本的障碍，而 **张量并行 (TP)** 似乎是一条更顺畅的路径。

- **呼吁开发者贡献 (Calling Coders to Contribute)**：lm-evaluation-harness 似乎缺少 `register_filter` 函数，因此呼吁贡献者提交 PR 以增强该工具的实用性。

- **Brier 分数难题 (Brier Score Brain Twister)**：**ARC 评估**数据中的一个异常导致有人建议重新拟合 Brier 分数函数，以确保无论数据是否一致都能进行无误差的评估。

- **模板讨论 (Template Tête-à-Tête)**：关于 *Hailey 分支* 中聊天模板化分支的状态引起了兴趣，该分支上次更新是在一段时间前，这引发了对该功能进展的询问。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Mixtral 混乱 (Mixtral Muddle)**：一家 **Mixtral 8x7b** 提供商面临发送空白响应的问题，导致其被暂时从 OpenRouter 中移除。目前正在考虑针对此类故障的自动检测方法。

**Soliloquy 的订阅惊喜 (Soliloquy's Subscription Surprise)**：**Soliloquy 8B** 模型转为付费服务，费用为 **每 1M tokens 0.1 美元**。更多信息和讨论请见 [Soliloquy 8B](https://openrouter.ai/models/lynn/soliloquy-l3)。

**DBRX AI 令人惊叹 (DBRX AI Achieves AI Astonishment)**：Fprime-ai 在 LinkedIn 上宣布了其 **DBRX AI** 的重大进展，引发了社区的兴趣和讨论。LinkedIn 公告可在此处阅读 [here](https://www.linkedin.com/posts/fprime-ai_fprimeailabs-dbrx-ai-activity-7189599191201980417-Te5d)。

**创意模型大乱斗 (Creative Model Melee)**：社区成员就角色扮演创意的最佳开源模型展开争论，**WizardLM2 8x22B** 和 **Mixtral 8x22B** 因其出色的创意能力成为顶级竞争者。

**关于 GPT-4 Turbo 的大辩论 (The Great GPT-4 Turbo Debate)**：Microsoft 对 **Wizard LM** 项目的影响引发了激烈辩论，导致对 GPT-4、Llama 3 和 WizardLM 等模型的发生率、性能和可持续性进行了深入探讨。分享的资源包括[事件摘要](https://rocky-muscle-755.notion.site/)和一份杂项 [OpenRouter 模型列表](https://openrouter.ai/models?q=free)。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**Create-llama 简化 RAG 设置**：**create-llama v0.1** 版本发布，带来了对 **@ollama** 和向量数据库集成的新支持，使得使用 llama3 和 phi3 模型部署 RAG 应用变得更加容易，详见其 [公告推文](https://twitter.com/llama_index/status/1783528887726817653)。

**LlamaParse 在实战教程和网络研讨会中受到推崇**：一个实战教程展示了如何使用 **LlamaParse**、**@JinaAI_ embeddings**、**@qdrant_engine 向量存储**以及 **Mixtral 8x7b** 来创建复杂的 RAG 应用，详情点击 [这里](https://twitter.com/llama_index/status/1783601807903863184)；同时 KX Systems 举办了一场网络研讨会，旨在利用 **LlamaParse** 解锁复杂的文档解析能力（详情见 [此推文](https://twitter.com/llama_index/status/1783622871614664990)）。

**AWS 与 LlamaIndex 联手举办开发者工作坊**：AWS 与 **@llama_index** 合作提供了一个专注于 LLM 应用开发的工作坊，集成了 AWS 服务和 LlamaParse；更多细节可以在 [这里](https://twitter.com/llama_index/status/1783877951278432733) 找到。

**深入探讨高级 RAG 系统**：社区就改进 RAG 系统进行了深入讨论，并分享了一个关于高级设置技术的视频，涵盖了从 sentence-window retrieval 到集成结构化 Pydantic 输出的所有内容（[高级 RAG 课程](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/5/auto-merging-retrieval)）。

**讨论本地 LLM 部署策略**：针对使用本地 LLM 设置以规避对外部 API 依赖的对话非常活跃，官方 **LlamaIndex 文档**中提供了指导（[本地 LLM 入门示例](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/)），展示了解决导入错误和正确安装软件包的策略。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**Llama 3 的评价褒贬不一**：社区对 **Llama 3** 的反馈存在分歧，一些人强调其代码召回能力与 GPT-4 的预期相比不足，而另一些人则推测可以通过配置增强来弥补性能差距。

**“了解你的客户”云端难题**：拟议的美国云服务“了解你的客户”（Know Your Customer）政策引发了担忧和讨论，强调了在反馈窗口关闭前，社区在 [Federal Register](https://www.federalregister.gov/documents/2024/01/29/2024-01580/taking-additional-steps-to-address-the-national-emergency-with-respect-to-significant-malicious) 上提供意见的必要性。

**AI 模型训练效率提升**：视觉模型训练的创新引起了轰动，一种*弱监督预训练方法*的速度超过了传统的对比学习，训练速度提高了 **2.7 倍**，如本 [研究](https://arxiv.org/abs/2404.15653) 所述。该方法放弃了对比学习高昂的计算成本，转而采用多标签分类框架，取得了与 **CLIP** 模型相当的性能。

**VAST 全模态领域**：人们对微调 **VAST**（一个视觉-音频-字幕-文本全模态基础模型）充满热情。该项目标志着向全模态迈进了一步，相关资源可在其 [GitHub 仓库](https://github.com/txh-mercury/vast) 获得。

**Nightshade 的透明度问题**：公会对 **Nightshade** 的有效性和透明度进行了辩论，批判性地审视了 autoencoder 的能力，以及在发布可能引起争议的研究结果时的犹豫。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Mac 性能与 Interpreter 强强联手**：Open Interpreter 的 **New Computer Update** 显著提升了本地功能，特别是 **原生 Mac 集成**。该实现允许用户使用简单的命令（如 `interpreter --os`）来控制 Mac 的原生应用程序，详情可见其 [更新日志](https://changes.openinterpreter.com/log/ncu-ii)。

**AI 之眼**：社区成员重点介绍了 **Moondream 小型视觉语言模型**，并提供了 [Img2TxtMoondream.py 脚本](https://github.com/CodeAKrome/bootcupboard/blob/main/llm-img/Img2TxtMoondream.py) 等资源。讨论中还提到了 **LLaVA**，这是一个托管在 [Hugging Face](https://huggingface.co/liuhaotian/llava-v1.6-34b) 上的多模态模型，它基于强大的 **NousResearch/Nous-Hermes-2-Yi-34B** 模型。

**避免循环的策略**：工程师们一直在交流减轻本地模型循环行为的策略，考虑的解决方案从调整 *temperature 设置* 和 *prompt 编辑* 到更复杂的 *架构更改*。一个有趣的概念——*frustration metric*（挫败感指标）被引入，用于在模型陷入重复循环时调整其响应。

**通过对话驱动机器狗**：一位成员询问了利用 **Open Interpreter** 指挥 **Unitree GO2 机器人狗** 的前景，引发了对跨学科应用的兴趣。技术挑战（如设置虚拟 API key 和解决 Pydantic 的命名空间冲突）也通过共享解决方案得到了处理。

**固件定稿**：**Open Interpreter 0.2.5 New Computer Update** 已正式脱离 beta 阶段，包含了前面提到的最新增强功能。关于更新是否处于 beta 状态的询问在版本检查后得到了肯定的答复。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**CEO 对成员推文的点赞**：一位参与者对 *Hugging Face 的 CEO 认可他们的推文* 感到兴奋；社区中的社交网络和认可度非常活跃。

**科技巨头投身微调**：以 **[Meditron](https://arxiv.org/abs/2311.16079)** 为例，关于针对特定用途微调语言模型的讨论正在升温，突显了领域特定改进的前景，并暗示了 **即将发表的关于持续预训练的论文**。

**Transformer 社区的小麻烦**：**transformers 4.40.0** 中出现了一个 'AttributeError'，让一位用户陷入困境，这提醒人们即使是微小的更新也可能破坏工作流。

**模型与数学的结合**：尽管存在一些困惑，仍有人询问关于将 **zzero3** 与 **快速傅里叶变换 (fft)** 集成的问题；请关注这一复杂的算法协作。

**优化器探索升温**：**FSDP (Fully Sharded Data Parallel)** 与优化器的兼容性仍然是一个热门话题，研究发现 **AdamW** 和 **SGD** 没有问题，而 `paged_adamw_8bit` 不支持 FSDP offloading，这促使人们在 **OpenAccess-AI-Collective/axolotl** 资源中寻找替代方案。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**上传故障与字体纠纷**：**Cohere** 公会的用户解决了 Azure 上 **Cohere Toolkit** 的问题，指出上传需使用回形针图标；尽管如此，上传功能未被发现的问题依然存在。GitHub 上 **Cohere 字体** 的许可引发了讨论；它不属于 MIT 许可证，并计划被替换。

**模型使用须知**：讨论澄清了 Cohere 的 [Command+ 模型](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus) 提供开放权重访问，但不允许商业用途，且训练数据不公开。

**搜索 API 切换建议**：公会权衡了在集成 Cohere-Toolkit 时从 **Tavily** 切换到 **Brave Search API** 的可能性，理由是其在速度、成本和检索准确性方面具有潜在优势。

**Toolkit 部署讨论**：讨论了 Cohere Toolkit 在 Azure 上的部署复杂性，其中选择模型部署选项至关重要，且不需要 API key。相反，本地添加工具时遇到了 PDF 上传和 sqlite3 版本兼容性问题。

**对“抹黑文章”的批判性回顾**：针对一篇针对 *Cohere* 的“抹黑文章”的批评引发了激烈讨论，对话集中在 AI Agent 的责任及其现实世界的行为。社区出现了一种要求批判性问责的声音，成员们强调需要用实质性的主张来支持批评。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 冲刺 1.0 版本**：Tinygrad 正准备发布 1.0 版本，重点展示了趋于稳定的 API，并提供了一套工具包，包括 [安装指南](https://tinygrad.github.io/tinygrad/)、[MNIST 教程](https://tinygrad.github.io/tinygrad/mnist/) 以及详尽的 [开发者文档](https://tinygrad.github.io/tinygrad/developer/)。
  
- **Comma 开始使用 tinygrad 测试 tinybox**：George Hotz 强调 comma 的 tinybox 是 tinygrad 的理想测试平台，重点依然放在软件而非硬件上，同时潜在的 tinybox 2 合作也已初见端倪。

- **排除 Tenstorrent**：经过评估，由于硬件效率低下，已放弃与 Tenstorrent 的合作，但如果未来的成本效益分析发生有利变化，仍保留未来合作的可能性。

- **解决 tinygrad 的 Quantile 函数挑战**：深入探讨了 tinygrad 的开发进展，揭示了为 Diffusion 模型采样复制 `torch.quantile` 的努力，这是一项需要在框架内实现精确排序算法的复杂任务。

- **AMD 的 MES 对 tinygrad 作用有限**：Hotz 认可了来自 AMD 的 Felix 对 AMD Machine Environment Settings (MES) 的详细解析，但最终评估认为它与 tinygrad 的发展方向无关，目前的精力集中在开发 PM4 backend 上。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**强劲表现：Hermes 2.5 略胜 Hermes 2**：通过增加代码指令示例，**Hermes 2.5** 在各项基准测试中表现出优于 **Hermes 2** 的性能。

**安全成为焦点**：在 Modular 发布大量软件和功能之际，解决安全漏洞变得至关重要，重点是防范类似 **XZ 事件** 的供应链攻击，以及预计到 2024 年开源代码在软件开发中的普及率将达到 **96%** 的趋势。

**从几何视角看量子复杂度**：成员们讨论了几何概念 **amplituhedron** 如何简化量子粒子散射振幅，并建议将 Machine Learning 作为一种工具，用于在系统规模扩大时破解可视化 **quantum states** 的复杂性。

**关于 Mojo 的一切**：围绕 **Mojo 编程语言** 的对话涵盖了 OS 确保的内存清理、`def` 与 `fn` 函数的区别（示例见 [此处](https://docs.modular.com/mojo/manual/functions)），以及通过 `Variant` 处理混合数据类型列表（该功能仍需改进）等话题。

**推进 Mojo**：ModularBot 标记了一个在 GitHub 上提交的关于 **Mojo** 的 Issue，敦促成员使用 Issue 来更好地跟踪问题，例如关于 `__copyinit__` 语义的问题（[通过 GitHub Gist](https://gist.github.com/modularbot/6aed759930420cd70f38795dbcb874fe)），并报告了一次更简洁的代码更新，其插入内容多于删除内容，实现了更高的效率。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**反喷子 AI 设计的棘手咨询**：一位用户提议设计一个 **anti-trolling AI**，并寻求关于系统如何有效识别和应对网络欺凌者的建议。

**冗长的 SQL 令人头疼**：参与者分享了使用 **Mistral** 和 **Llama3** 等开源模型生成过于冗长的 SQL 响应的经验，并遇到了 `OutputParserException`，同时提供了 [结构化输出支持](https://python.langchain.com/docs/integrations/chat/) 的链接以及调用 SQL Agents 的示例。

**RedisStore vs. Chat Memory**：社区澄清了 LangChain 集成中 **stores** 和 **chat memory** 的区别，强调 `RedisStore` 专门用于键值存储，而 **Redis Chat Message History** 用于基于会话的聊天持久化。

**模型调用的技术教程**：讨论了在 JavaScript 中将 Prompt 集成到 LangChain 模型时的正确语法，建议使用 `ChatPromptTemplate` 和 `pipe` 方法进行 Prompt 链式调用。

**访问 Gemini 1.5 的注意事项**：用户讨论了 **Gemini 1.5 Pro** 与 LangChain 的集成，强调它需要使用 `ChatVertexAI` 而非 `ChatGoogleGenerativeAI`，并且需要配置 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量才能正常访问。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Apple 拥抱开源**: Apple 已步入开源领域，发布了一系列参数量从 270M 到 3B 的模型，其中 [270M 参数模型已在 Hugging Face 上提供](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)。

**Dify 平台的起伏**: 开源 LLM 应用开发平台 Dify 因结合了 AI 工作流和模型管理而备受关注，尽管对其缺乏 [loops 和 context scopes](https://github.com/langgenius/dify?tab=readme-ov-file) 的担忧也随之出现。

**PyTorch 助力 LLM 训练**: PyTorch 推出了 [Torchtitan](https://github.com/pytorch/torchtitan)，这是一个专门用于辅助从零开始训练 llama3 等大型语言模型的库。

**SORA 带来的视频生成创新**: OpenAI 的 SORA 是一款可以制作长达一分钟视频的视频生成模型，正受到广泛关注，[FXGuide 的一篇文章](https://www.fxguide.com/fxfeatured/actually-using-sora/)探讨了其用户体验和细节。

**用于高效 Transformer 训练的 MOD 层**: “Mixture of Depths” 论文发表，提出了一种通过交替使用新的 MOD 层和传统 Transformer 层来加速 Transformer 训练的方法，该方法在 [演示文稿](https://paper-club.ivanleo.com/papers/mixture-of-depths) 中有所介绍，并在论文 [摘要](https://arxiv.org/abs/2402.00841) 中进行了详细说明。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Phi-3-Mini-4K Instruct 增强**: 正如成员们所讨论的，将 **Phi-3-Mini-4K-Instruct** 与 llamafile 结合使用，可以为高质量和密集推理数据集提供环境，[Hugging Face 上概述了集成步骤](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf#how-to-use-with-llamafile)。

- **模型下载变得更简单**: **Mixtral 8x22B Instruct llamafile** 的 README 更新包含了一个下载技巧：使用 `curl -L` 以在 CDN 上实现平滑重定向，详见 [快速入门指南](https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile)。

- **Llamafile 与 CPU 的兼容性问题**: 由于 AVX CPU 特性要求，在 **Apple M1** Mac 上运行 llamafile 时出现了问题；此 [GitHub issue](https://github.com/Mozilla-Ocho/llamafile/issues/327#issuecomment-2053680659) 中分享了重启系统的临时解决方案，并建议在 8GB RAM 系统上使用较小的模型。

- **Windows 与 Llamafile 的冲突**: 用户报告 **Windows Defender** 误将 llamafile 检测为木马。提出的解决方法包括使用虚拟机或将其列入白名单，并提醒官方二进制文件可以在 [此处](https://www.microsoft.com/en-us/wdsi/filesubmission) 找到。

- **高资源消耗模型挑战极限**: 运行 8x22B 模型需要大量资源，参考建议使用 128GB RAM 以稳定执行 [Mistral 8x22B 模型](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8)，这标志着运行复杂的 AI 模型时需要巨大的内存占用。



---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**Llama 在评审中击败评审员**: 在 **Judgemark** 基准测试中，**Llama-3-70b** 展示了令人印象深刻的性能，证明了其在 **disco-judge** 应用中进行微调的潜力，因为它支持至少 8k 的上下文长度。社区还探讨了协作评估工作，并参考了先进的评审 Prompt 设计来评估复杂的评分标准。

**模型基准测试与推理问题讨论**: 尽管在已发表的评估中得分颇高，**Phi-3-mini-4k-instruct** 在 **eq-bench** 排行榜上的排名却出人意料地靠后。在模型部署方面，讨论强调了 **DiscoLM_German_7b_v1** 初始化和推理速度慢的问题，以及可能通过使用 `device_map='auto'` 修复的配置错误。

**工具 API 评估与 Hugging Face 咨询**: 社区辩论强调了 **Tgi** 的 API 优先、低延迟方法，并赞扬了 **vllm** 作为一个用户友好且针对部署成本效率进行了优化的库。关于 Hugging Face 批量生成能力的咨询引发了讨论，GitHub issue 的交流体现了社区的参与。

**模型开发中的感谢与推测**: 尽管存在部署问题，成员们仍对 **DiscoLM** 模型系列表示了赞赏，同时也推测了构建 **8 x phi-3 MoE 模型** 以增强模型能力的潜力。**DiscoLM-70b** 也是热门话题，用户正在排查错误并分享使用经验。

**模型采用的成功与普及**: **Phi-3-mini-4k** 模型的适配（被称为 llamafication）在德语输出中获得了 51.41 的 EQ-Bench 分数。对话还指出 **gguf** 模型被迅速采用，发布后不久便有显著的下载量。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Claude 展示深度与结构**: 在一场深入的讨论中，**Claude** 的行为和训练被认为与 Anthropic 的愿景“基本正交”，通过 **RLAIF 训练** 展现了意想不到的深度和结构化理解。讨论将其与“荣格个体化”等概念进行了类比，对话线程[突出了 Claude 的能力](https://x.com/repligate/status/1783426037210026372?s=46&t=xxWoJxAS_7-BBFC2ro84Zw)。

**辩论 RLHF 与 KTO 的优劣**: **Reinforcement Learning from Human Feedback (RLHF)** 与 **Knowledge-Targeted Optimization (KTO)** 之间的对比引发了辩论，探讨了它们对不同商业部署的适用性。

**训练方法转型带来改进**: 提到了一次采访，其中训练方法从 **Supervised Fine Tuning (SFT)** 演进到 **Data Programming by Demonstration (DPO)**，再到 **KTO**，根据用户反馈提升了性能。

**剖析 RLHF 的复杂性**: 社区承认了 **RLHF** 的复杂性，特别是涉及不同数据源及其对下游评估指标的影响。

**探究梯度范数激增 (Grad Norm Spikes)**: 有人请求澄清预训练期间梯度范数激增的影响，强调了潜在的负面效应，但回复中未给出具体细节。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

**Moondream 挑战验证码 (CAPTCHAs)**: 一段[视频指南](https://www.youtube.com/watch?v=Gwq7smiWLtc)展示了如何针对验证码图像数据集微调 **Moondream Vision Language Model**，旨在提高其在实际应用中的图像识别能力。

**低成本 AI 模型极具性价比**: 分享了文档《Low-Cost Language Models: Survey and Performance Evaluation on Python Code Generation》，涵盖了对 **CPU 友好型语言模型** 的评估，并介绍了一个包含 60 个编程问题的创新数据集。这篇[调查文章](https://arxiv.org/abs/2404.11160)强调了 Chain-of-Thought Prompt 策略的使用。

**聚会、交流与计算**: 诚邀 AI 开发者参加在多伦多 **Cohere space** 举行的见面会，届时将有社交机会、闪电演讲和演示——详情见[活动页面](https://lu.ma/devs5)。

**Arctic 之风吹向企业**: 通过[一段新视频](https://www.youtube.com/watch?v=nV6eIjnHEH0)介绍了 **Snowflake Arctic**，将其定位为一款具有成本效益、企业级的 Large Language Model，以补充针对商业应用量身定制的 AI 工具套件。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **轻松在本地运行模型**：工程师们探索了 [jan.ai](https://jan.ai/)，这是一个因在本地机器上运行 GPT 模型的直观方法而受到赞誉的 GUI，有可能简化实验过程。
- **苹果进入语言模型领域**：苹果推出的全新 [OpenELM](https://huggingface.co/apple/OpenELM) 系列提供了一系列高效扩展的语言模型，包括指令微调（instruction-tuned）变体，这可能会改变模型参数效率（parameter efficiency）的游戏规则。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Llama 3 在主题复杂度方面表现出色**：Venadore 已开始尝试使用 **Llama 3** 进行主题复杂度分类，并报告了令人期待的结果。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

# PART 2: 详细频道摘要与链接

**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1232939117835456574)** (1265 messages🔥🔥🔥): 

- **LLM 微调挑战**：成员们讨论了模型的微调问题，特别是在使用 awq、gptq 以及在 4-bit 量化下运行模型时。具体问题包括 Token 索引序列长度错误、超过 48GB 的 VRAM 仍不足以运行某些模型，以及在利用 Aphrodite Engine 或 llama.cpp 测试微调模型时的困惑。建议的补救措施包括修改 batch sizes 和梯度累积（grad accumulation）、启用 packing，以及联系社区专家寻求指导。
- **寻找合适的技术栈**：一位用户表示希望将不同的 AI 模型集成到一个项目中，允许与执行不同任务的各种 Agent 进行对话。资深的社区成员建议从简单的脚本开始，而不是复杂的 AI 解决方案，并建议在实施前进行彻底的研究。此外还讨论了 API 成本与本地运行的权衡。
- **游戏偏好与推荐**：用户们分享了对近期在早期访问阶段推出的《Manor Lords》等游戏的兴奋之情，并对《Baldur's Gate 3》和《Elden Ring》等热门游戏的娱乐价值提供了个人见解。
- **解锁 Phi 3 的 Fused Attention**：据透露，Phi 3 Mini 包含 Fused Attention，引发了成员们的好奇。尽管该功能已经存在，但其他用户建议在深入研究之前等待进一步的开发。
- **Unsloth 获得显著下载量**：Unsloth 团队宣布在 Hugging Face 上的月度模型下载量达到 50 万次，感谢社区对 Unsloth 微调框架的广泛支持和使用。讨论了上传 GGUF 模型的必要性，并指出由于其他人已经提供了这些模型，可能存在冗余。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Orenguteng/LexiFun-Llama-3-8B-Uncensored-V1">Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1 · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.16710">Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding</a>: 我们介绍了 LayerSkip，这是一种加速大语言模型 (LLM) 推理的端到端解决方案。首先，在训练期间我们应用层丢弃 (layer dropout)，对较早的层使用较低的丢弃率，而对较高的层使用较高的丢弃率...</li><li><a href="https://arxiv.org/abs/2403.13799">Reverse Training to Nurse the Reversal Curse</a>: 大语言模型 (LLM) 有一个令人惊讶的缺陷：当训练“A 具有特征 B”时，它们无法泛化到“B 是 A 的一个特征”，这被称为逆转诅咒 (Reversal Curse)。即使...</li><li><a href="https://www.amazon.co.uk/Yalucky-Novelty-Drinkware-Birthday-Christmas/dp/B0834QSW5Z">no title found</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-bnb-4bit">unsloth/llama-3-8b-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://hub.docker.com/r/alpindale/aphrodite-dev/tags">Docker</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-110B-Chat">Qwen/Qwen1.5-110B-Chat · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/rookie-numbers-gif-26135237">Rookie Numbers GIF - Rookie Numbers - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k">gradientai/Llama-3-8B-Instruct-262k · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/apple/OpenELM">apple/OpenELM · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/TETO101/AIRI-L3-INS-1.0-0.00018-l">TETO101/AIRI-L3-INS-1.0-0.00018-l · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/crusoeai/Llama-3-8B-Instruct-262k-GGUF">crusoeai/Llama-3-8B-Instruct-262k-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py)">Layer Normalization &mdash; Triton  documentation</a>: 未找到描述</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: 未找到描述</li><li><a href="https://github.com/meta-llama/llama3/blob/main/LICENSE">llama3/LICENSE at main · meta-llama/llama3</a>: 官方 Meta Llama 3 GitHub 站点。通过在 GitHub 上创建账号为 meta-llama/llama3 的开发做出贡献。</li><li><a href="https://github.com/oKatanaaa/kolibrify">GitHub - oKatanaaa/kolibrify: Curriculum training of instruction-following LLMs with Unsloth</a>: 使用 Unsloth 对遵循指令的 LLM 进行课程学习训练 - oKatanaaa/kolibrify</li><li><a href="https://github.com/meta-llama/llama-recipes">GitHub - meta-llama/llama-recipes: Scripts for fine-tuning Meta Llama3 with composable FSDP &amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp; custom datasets for applications such as summarization and Q&amp;A. Supporting a number of candid inference solutions such as HF TGI, VLLM for local or cloud deployment. Demo apps to showcase Meta Llama3 for WhatsApp &amp; Messenger.</a>: 用于使用可组合的 FSDP 和 PEFT 方法微调 Meta Llama 3 的脚本，涵盖单节点/多节点 GPU。支持默认和自定义数据集，适用于摘要和问答等应用。支持多种候选推理解决方案，如用于本地或云端部署的 HF TGI、VLLM。展示用于 WhatsApp 和 Messenger 的 Meta Llama 3 演示应用。</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 微调 Llama 3、Mistral 和 Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1233054158064713779)** (18 条消息🔥): 

- **讨论微调策略**：聊天成员讨论了*在原始文本上微调语言模型*是否会导致其失去对话能力。提出的一种解决方案是将原始文本与对话数据集结合起来，在添加原始文本知识的同时保留对话能力。

- **WizardLM 解散传闻**：基于 [Qingfeng Sun 的员工页面](https://www.microsoft.com/en-us/research/people/qins/) 的重定向产生了推测，暗示他可能不再在 Microsoft 工作，这可能预示着 WizardLM 团队的关闭。指向 [Reddit 帖子](https://old.reddit.com/r/LocalLLaMA/comments/1cd4b9l/staff_page_for_qingfeng_sun_lead_wizard_lm/) 和 [Notion 博客文章](https://rocky-muscle-755.notion.site/What-happened-to-Wizard-LM2-a247e09244d0483cbb02c1587b357c9d) 的链接为这一理论提供了依据。

- **Unsloth AI 微调资源**：对于在组合数据集上进行微调，成员被引导至 Unsloth AI 在 GitHub 上的仓库 [免费微调](https://github.com/unslothai/unsloth)，该仓库列出了所有可用的 notebooks，特别是用于[文本补全的 Colab notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)。

- **WizardLM 数据抢救行动**：在讨论了与 Microsoft WizardLM 相关的潜在裁员之后，一位成员表示他们拥有 WizardLM 数据集的副本，这可能有助于未来的努力。

- **模型训练的过山车**：聊天成员幽默地分享了他们在模型训练方面的经验，带着希望与挫败交织的心情谈论他们的损失曲线 (loss curves)。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1cd4b9l/staff_page_for_qingfeng_sun_lead_wizard_lm/">Qingfeng Sun（Wizard LM 首席研究员）的员工页面已从 Microsoft.com 删除</a>: 如果你访问 [Qingfeng Sun](https://www.microsoft.com/en-us/research/people/qins/) 的员工页面，你将被重定向到一个通用的落地页...</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 微调 Llama 3, Mistral &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80%</a>: 微调 Llama 3, Mistral &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1cd4b9l/staff_page_f">Qingfeng Sun（Wizard LM 首席研究员）的员工页面已从 Microsoft.com 删除</a>: 如果你访问 [Qingfeng Sun](https://www.microsoft.com/en-us/research/people/qins/) 的员工页面，你将被重定向到一个通用的落地页...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1232935568678195214)** (86 messages🔥🔥): 

- **推理代码片段咨询**：一位成员询问是否有更简单的方法来测试 **GGUF 模型**，而无需将其加载到 Oobabooga 中。另一位成员表示未来计划提供 **推理和部署选项**。

- **Triton 难题**：成员们讨论了 **Triton** 的问题及其在本地运行 Unsloth 的必要性。一位成员因潜在的版本冲突在 `triton.common` 模块上遇到困难，其他成员确认 Triton 是必需项。

- **微调挫折**：对话围绕微调过程中遇到的问题展开，即模型不断重复最后一个 token。建议的解决方案是使用最新的 [colab notebooks](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp) 更新生成配置 (generation config)。

- **量化失败混乱**：多位成员在尝试使用 `save_pretrained_merged` 和 `save_pretrained_gguf` 时遇到了“**量化失败**”错误。该问题最终被确定为用户错误，即 **llama.cpp 不在模型文件夹中**，在修正文件位置后问题得到解决。

- **模型训练错误与见解**：讨论了关于 **训练错误**、在 Kaggle 等平台上从检查点恢复训练以及 **微调** 指导的各种问题和解决方案。一个值得注意的点是 **检查点 (checkpointing)** 的使用，它允许从最后保存的步骤恢复训练，使受限于连续运行时间的平台用户受益。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/comfyui/comments/1bq22x7/change_clothing_in_1_click_ootdiffusion/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/TETO101/AIRI_INS5/viewer">TETO101/AIRI_INS5 · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1233058558992449658)** (10 messages🔥):

- **Meta 发布 LlaMA-3**：Meta 宣布推出其 **LlaMA** 模型系列的下一代产品，发布了 **8B 模型**和 **70B 模型**，并预告了即将推出的 400B 模型，承诺将达到 GPT-4 级别的基准测试。感兴趣的人士可以申请访问这些模型，据报道它们在同尺寸类别中处于领先地位；详细的对比和见解可在 [Substack 文章](https://datta0.substack.com/p/ai-unplugged-8-llama3-phi-3-training)中查看。
- **开源 Kolibrify**：一位用户宣布发布其项目 **Kolibrify**，这是一个使用 Unsloth 对遵循指令的 LLM 进行课程学习（curriculum training）的工具，专为 **PhD 研究**设计。该工具旨在帮助那些在工作站上对 LLM 进行微调以进行快速原型设计的用户，可在 [GitHub](https://github.com/oKatanaaa/kolibrify) 上获取。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://datta0.substack.com/p/ai-unplugged-8-llama3-phi-3-training">AI Unplugged 8: Llama3, Phi-3, Training LLMs at Home ft DoRA.</a>：洞察胜过信息</li><li><a href="https://github.com/oKatanaaa/kolibrify">GitHub - oKatanaaa/kolibrify: Curriculum training of instruction-following LLMs with Unsloth</a>：使用 Unsloth 对遵循指令的 LLM 进行课程学习 - oKatanaaa/kolibrify
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1233133113719394324)** (7 条消息): 

- **TRL Trainer 创新**：一位成员正在尝试在评估步骤中实现 **laser pruning（激光剪枝）以及可能的使用 trl trainer 进行冻结**的功能。目标是在利用同一块 GPU 的同时，迭代地增加模型的上下文长度。
  
- **上下文扩展无需重新初始化**：有人建议通过调整模型和 tokenizer 配置来增加可用的上下文长度。经确认，这些更改**不需要重新初始化**系统。

- **聊天中的表情符号表达**：成员们在交流中使用表情符号，并对在聊天中输入**表情符号**的能力表示惊讶和愉快。
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1233126296460988489)** (1 条消息): 

- **iOS 用户获得专属功能**：**Perplexity AI Discord 聊天机器人**已更新一项新功能，用户可以*提出任何问题并听取回答*。该功能从今天起面向 **iOS <a:pro:1138537257024884847> 用户**开放。
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1232950876365127753)** (531 条消息🔥🔥🔥): 

- **Perplexity 支持 Opus**：尽管最近对其使用频率进行了限制，常规的 Perplexity 聊天机器人仍然支持 Opus。
- **关于 Opus 限制的大辩论**：用户对 Claude 3 Opus 交互突然设置的限制表示沮丧，可用查询次数从之前的更高或无限次减少到每天仅 50 次。讨论围绕模型性能差异、与 you.com 等竞争对手的价格对比，以及使用上限的透明度展开。
- **企业版功能不明**：成员们讨论了 Perplexity 的普通 Pro 版和企业 Pro 版之间的区别，特别是在隐私设置和用于模型训练的数据使用方面。关于设置开关是否真的能保护用户数据不被 Perplexity 的模型或 Anthropic 的模型使用，似乎存在困惑。
- **透明度与沟通批评**：社区成员批评 Perplexity 在使用变更方面的沟通不力，并敦促其发布官方公告。用户将其与其他服务（如 poe.com）进行了比较，认为后者在定价和限制方面更加透明。
- **生态系统影响**：对话思考了 Google 如果认真对待其 Gemini 模型可能带来的影响，一些人认为由于可扩展性和 Google 的数据集，Gemini 具有竞争优势。鉴于日益激烈的竞争，人们对 GPT-5 的表现寄予厚望。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/ChatGPT/comments/1ccfzon/google_presents_leave_no_context_behind_efficient/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://x.com/mattshumer_/status/1783531642344067159">Matt Shumer (@mattshumer_) 的推文</a>: LLaMA 3 发布已经一周了。在这段时间里，我们： - 将上下文从 8K 扩展到 128K - 训练了多个性能极其出色的微调模型 - 让推理速度达到 800+ tokens...</li><li><a href="https://huggingface.co/spaces/multimodalart/stable-cascade">Stable Cascade - multimodalart 创建的 Hugging Face Space</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1232990801450238012)** (8 条消息🔥): 

- **探索 Perplexity AI**: 一位用户分享了一个 [搜索链接](https://www.perplexity.ai/search/jTVe66V.RHKaTLZPJ3MZ0Q)，但没有附带任何评论或背景信息。
- **深入探讨泳池化学**: 在讨论泳池维护的苦恼时，一位成员提到 **Langlier Saturation Index**（朗格里尔饱和指数）可能是一个有帮助但复杂的解决方案，但它并非专门为室外泳池设计，并分享了一个相关的 [Perplexity AI 搜索链接](https://www.perplexity.ai/search/What-is-the-hSnFPTgtQWu2MvGENVpNFg)。
- **网络中立性及其对 AI 的影响**: 分享了一个关于 FCC 恢复网络中立性的链接，帖子暗示这可能对 **AI Boom** 产生影响，可通过 [Perplexity AI 搜索](https://www.perplexity.ai/search/FCC-restores-net-dXr_Ke3ST8SdNITs2AMDhA#1) 访问。
- **命令管理**: 一位用户询问了一个特定命令，并引用了一个 [Perplexity AI 搜索](https://www.perplexity.ai/search/what-command-i-IEqOU0n0SRyoAsxjDUkQoQ)。另一位用户提醒要确保该主题是 **Shareable**（可分享的）。
- **AI 用于投票？**: 用户对 AI 如何应用于投票系统表现出兴趣，并链接到了一个 [Perplexity AI 搜索](https://www.perplexity.ai/search/What-are-the-vwfV8gKTSHih8Np6rxFFIg#1)。
- **国土安全部的新指令**: 一次无评论的分享，包含了一个关于国土安全部公告的 [Perplexity AI 搜索](https://www.perplexity.ai/search/Homeland-Security-announced-8Q5AitclTxW6fYBe02t5CA) 链接。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1232963461433266227)** (10 条消息🔥): 

- **注意到 API 集成中的奇特现象**: 一位用户提到将 Perplexity API 与语音助手集成时，观察到响应中存在日期相关性问题，例如收到了去年的体育比赛比分而非当天的。他们还询问了关于插入文档进行比较的问题，并对扩展引用功能表示感兴趣，暗示了更多样化用途的潜力。
  
- **Perplexity 不支持 GPT-4**: 一位成员希望通过 Perplexity API 使用 GPT-4，但发现其并不支持。另一位成员提供了一个 [文档链接](https://docs.perplexity.ai/docs/model-cards)，列出了可用的模型，包括 `sonar-small-chat`、`llama-3-instruct` 变体和 `mixtral-instruct`，但未提及 GPT-4。

- **使用 llama-3-70b-instruct 的最佳超参数**: 一位用户询问了通过 API 使用 `llama-3-70b-instruct` 模型时的合适超参数，分享了他们当前的参数结构并寻求确认或修正，特别是关于 `max_tokens` 和 `presence_penalty` 的值。

- **集成细节尚不明确**: 同一位用户提到在调用 API 时保持在 Rate Limits（速率限制）内，尽管目前尚不确定 Perplexity API 在参数设置方面是否与 OpenAI 的完全一致。

- **等待企业级 API 回复**: 一位企业用户在向 Perplexity AI 的企业联系人发送关于 API 使用的邮件后，在频道中寻求回复。另一位成员告知其回复时间通常在 1-3 周之间。

- **寻求关于“在线 LLMs”使用的澄清**: 一位 Perplexity AI 的新用户寻求关于使用在线 LLMs 指南的澄清，询问是否应避免使用 System Prompts，以及是否有必要以单轮对话（single-turn conversation）格式提交查询。

**提及的链接**: <a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>: 未找到描述

  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1233008609336754176)** (13 条消息🔥): 

- **寻求进一步精通 CUDA**: 讨论建议通过公开展示技能来加强 CUDA 学习，具体做法是编写一个快速的 Kernel 并分享成果。建议的项目包括优化固定大小的矩阵乘法、Flash Attention 以及各种 Quantization（量化）方法。

- **BASH 频道转变为 CUDA 孵化器**：`<#1189861061151690822>` 频道计划专注于可能受益于 CUDA 改进的算法，邀请成员贡献其优化的 kernel。然而，有人建议为这些贡献创建一个比 Discord 频道更持久的存储库，以应对其易逝性。

- **实例 GPU 配置验证**：一位用户确认，在通过 SSH 访问其实例后，GPU 配置已通过验证，显示 p3.2xlarge 始终一致地分配 V100。

- **下一场 CUDA 讲座即将开始**：发布了一则关于即将举行的 CUDA mode 讲座的公告，计划在公告发布后的 1 小时 40 分钟后举行。

- **期待 CUDA 更新**：有人询问 Ubuntu 24.04 的 CUDA 可分发包的发布计划，但消息历史记录中没有提供后续信息。
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1233014978609680425)** (40 messages🔥): 

- **Kernel Profiling 困惑**：一位成员试图使用 **NVIDIA Nsight Profiler** 获取有关 kernel 操作的更详细信息。在最初混淆了 Nsight Systems 和 Nsight Compute 之后，明确了使用 [NVIDIA Nsight Compute CLI 用户指南](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#introduction) 可以获得详细的 kernel 统计数据。
- **理解同步 (Synchronize)**：解释了 `cudaStreamSynchronize` 如何意味着 CPU 等待 CUDA stream 中的任务完成，并建议检查每次调用同步是否必要，以潜在地提高性能。
- **占用率 (Occupancy) 与并行化建议**：讨论涉及 CUDA kernel 的启动统计数据，指出仅启动少量 block（如 **14 个 block**）可能会导致 GPU 空闲，除非使用了多个 CUDA stream。
- **性能见解与调整**：对于深入的 kernel 分析，建议在 profiling 中切换到 *full metric selection* 以获取更全面的信息。此外还有一个更广泛的建议：如果可行，目标应该是增加 block 数量，而不是引入 CUDA stream 的复杂性。
- **算术强度 (Arithmetic Intensity) vs 内存带宽**：比较了 *tiled_matmult* 和 *coarsed_matmult* kernel 的 FLOP/s 和内存吞吐量，并观察了 `__syncthreads()` 调用与内存带宽的关系。讨论演变为在使用 Nsight Compute / ncu 进行 profiling 时，如何从 SRAM 与 DRAM 的角度看待算术强度 (AI)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#introduction">4. Nsight Compute CLI &mdash; NsightCompute 12.4 文档</a>：未找到描述</li><li><a href="https://siboehm.com/articles/22/CUDA-MMM">如何优化 CUDA Matmul Kernel 以获得类似 cuBLAS 的性能：工作日志</a>：在这篇文章中，我将迭代优化一个用 CUDA 编写的矩阵乘法实现。我的目标不是构建一个 cuBLAS 的替代品，而是深入...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1232984060129968178)** (9 messages🔥): 

- **Tensor Expansion 解释**：讨论显示，PyTorch 中的 `Tensor.expand` 通过修改 tensor 的 strides 而不是其 storage 来工作。有人指出，在使用 Triton kernel 时，如果不当处理这些修改后的 strides，可能会出现问题。

- **Flash-Attention 不兼容警告**：有报告称新发布的 **flash-attn 2.5.7 版本** 与 **PyTorch 2.3.0** 安装的 CUDA 库不兼容，具体表现为 `undefined symbol` 错误，并希望尽快发布更新以解决此问题。

- **构建 Flash-Attention 的挑战**：一位用户在构建 **flash-attn** 时遇到困难，提到该过程极其耗时且最终未能成功。

- **理解 CUDA Tensor 内存**：一位成员分享了一个[有用的概述](https://pytorch.org/docs/stable/notes/cuda.html)，澄清了 CUDA tensor 的内存指针始终指向设备内存，且 PyTorch 默认限制跨 GPU 操作。

**提及的链接**：<a href="https://pytorch.org/docs/stable/notes/cuda.html">CUDA 语义 &mdash; PyTorch 2.3 文档</a>：未找到描述

  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1233492304527098017)** (1 messages): 

- **LLM 性能提升**：**NVIDIA C++ 团队** 宣布了一场令人兴奋的额外演讲，讨论将 `llm.c` 移植到 `llm.cpp`，承诺提供**更简洁、更快速的代码**。该会议即将开始。

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1233058707487719536)** (47 条消息🔥): 

- **探索 Plenoxels 和 SLAM 算法**：对话讨论了 [Plenoxels CUDA kernels](https://github.com/sxyu/svox2/tree/master/svox2/csrc)，这是 NeRF 的更快变体，并表达了对 [Gaussian Splatting SLAM](https://github.com/muskie82/MonoGS) 的 CUDA 版本的兴趣。
- **利用 CUDA 加速 Mobile ALOHA**：[Mobile ALOHA](https://mobile-aloha.github.io/) 的推理算法，如 [ACT](https://github.com/MarkFzp/act-plus-plus) 和 [Diffusion Policy](https://github.com/MarkFzp/act-plus-plus)，是讨论的热点。
- **二值矩阵的 Kernel 操作**：进行了一场关于为二值（0-1）或三值（1.58-bit, -1, 0, 1）矩阵操作创建 CUDA kernel 的头脑风暴。小组讨论了避免 unpacking（解包）的潜在方法，包括掩码乘法策略和 kernel fusion（算子融合）。
- **低比特量化与效率讨论**：成员们辩论了在 Pytorch 中进行 unpacking 操作与使用 fused CUDA 或 Triton kernels 的效率。一些人建议在不解包的情况下进行操作，而另一些人则强调内存复制和缓存是主要问题。[Microsoft 的 1-bit LLM 论文](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)被提及作为优化神经网络线性层的动力。
- **CUDA 中打包操作的挑战**：对话集中在直接对打包数据类型进行类似矩阵乘法操作而不进行解包的可行性，参考了 CUDA 8.0 的 bmmaBitOps 作为一种潜在方法。讨论包括 [CUDA 编程指南中的位操作](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=Warp#sub-byte-operations)以及对尝试最小化解包计算的兴趣。一位成员提供了 [BitNet 的 CPU 版本](https://github.com/catid/bitnet_cpu)链接用于测试目的。

**提到的链接**：<a href="https://github.com/catid/bitnet_cpu">GitHub - catid/bitnet_cpu: Experiments with BitNet inference on CPU</a>：在 CPU 上进行 BitNet 推理的实验。通过在 GitHub 上创建账号为 catid/bitnet_cpu 的开发做出贡献。

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1233009306874679356)** (6 条消息): 

- **探索多 GPU 编程**：一位成员暗示了学习多 GPU 编程的潜力，认为这可能是一个感兴趣的领域。
- **用于学习的 NVIDIA GPU 笔记本电脑**：成员们一致认为，配备 NVIDIA GPU 的笔记本电脑（如包含 4060 的机型）是学习和测试 CUDA 代码的高性价比选择。
- **用于 CUDA 探索的 Jetson Nano**：对于想要学习 CUDA 编程的人，推荐使用 Jetson Nano，尤其是当有额外显示器可用时。
- **寻找 NCCL All-Reduce Kernel 教程**：有人请求学习 NCCL 以实现 all-reduce kernels 的教程。聊天中未提供具体资源。
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1232974757205639292)** (5 条消息): 

- **澄清 Burst Size**：Burst size 指的是在 *memory coalescing*（内存合并）期间单次加载操作中访问的内存块，硬件将来自连续位置的多个内存加载合并为一个加载以提高效率。这一概念在 CUDA 指南的 **第 6 章第 3.d 节** 中有探讨，其中提到 burst sizes 大约为 **128 字节**。
- **来自外部资源的见解**：提供了一份由该书作者编写的 [讲义幻灯片](https://lumetta.web.engr.illinois.edu/408-S20/slide-copies/ece408-lecture8-S20.pdf)，确认了 bursts 通常包含 **128 字节**，这澄清了 coalesced（合并）与 uncoalesced（非合并）内存访问的概念。
- **纠正对 Burst Size 理解的偏差**：有一条重复消息指出，最初对合并访问存在误解，在重新阅读 CUDA 指南的相关章节后得到了解决。
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 条消息): 

poker6345: ppt 可以分享
  

---


**CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1233099632007970978)** (2 条消息):

- **简化版 BucketMul 函数揭秘**：一位成员分享了 **bucketMul** 函数的 [简化版本](https://kolinko.github.io/effort/gpu.html)，强调了它在计算乘法时如何同时考虑权重和调度（dispatch），并可能优化内存加载。这类似于关于分桶 COO（bucketed COO）以获得更好内存性能的讨论，并额外考虑了激活值（activations）。

- **AO 支持自定义 CUDA 扩展**：根据一个 [已合并的 Pull Request](https://github.com/pytorch/ao/pull/135)，PyTorch AO 现在支持自定义 CUDA 扩展，通过遵循提供的模板，可以实现与 `torch.compile` 的无缝集成。对于那些擅长编写 CUDA kernel 并旨在优化消费级 GPU 性能的人来说，这非常有吸引力。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://kolinko.github.io/effort/gpu.html">Effort Engine</a>：未找到描述</li><li><a href="https://github.com/pytorch/ao/pull/135">msaroufim 的自定义 CUDA 扩展 · Pull Request #135 · pytorch/ao</a>：这是 #130 的可合并版本 - 我必须进行一些更新，添加除非使用 PyTorch 2.4+ 否则跳过测试的逻辑，以及如果 CUDA 不可用则跳过测试的逻辑，将 ninja 添加到开发依赖项中...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/)** (1 messages): 

iron_bound: https://www.harmdevries.com/post/context-length/
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

iron_bound: https://github.com/adam-maj/tiny-gpu
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1232954105031295006)** (377 messages🔥🔥): 

- **追求多 GPU 效率**：该小组正致力于集成 NCCL 的多 GPU 支持，讨论多 GPU 配置的性能损耗，以及梯度累积（gradient accumulation）等潜在改进。考虑将 NCCL 代码合并到主分支，并讨论了 FP32 是否应支持多 GPU，目前倾向于不包含。
  
- **无需原子操作（Atomics）优化 Gather Kernel**：讨论了一种优化 layernorm backward kernel 的策略，通过避免原子操作，利用 threadblock 计数和 grid-wide synchronization 技术来管理依赖关系并简化计算。
  
- **FP32 路径的调试与决策**：建议简化 `train_gpt2` 的 FP32 版本以用于教学目的，可能会移除多 GPU 支持，以保持示例对初学者尽可能直观。
  
- **头脑风暴：持久线程（Persistent Threads）与 L2 通信**：关于使用持久线程配合 grid-wide synchronization 的潜在优缺点的深入技术讨论，旨在更有效地利用内存带宽，并可能并行运行多个 kernel。

- **并行性与 Kernel 启动关注点**：对话围绕由队列管理的全新 CUDA 并发 kernel 执行模型与传统方法的比较展开，思考了采用这种全新方法以实现更好的内存带宽利用和降低延迟的利弊。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://stackoverflow.com/questions/73853956/tf-bitcast-equivalent-in-pytorch)">tf.bitcast 在 PyTorch 中的等价物？</a>：这个问题与 `tf.cast` 在 PyTorch 中的等价物不同。`bitcast` 执行的是位级重新解释（类似于 C++ 中的 `reinterpret_cast`），而不是“安全”的类型转换。&#xA...</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtc24-s62419/">最新 NVIDIA 技术应用的能源与功率效率 | NVIDIA On-Demand</a>：随着能源成本的增加和环境影响的扩大，不仅要考虑性能，还要考虑能源消耗，这一点变得越来越重要。</li><li><a href="https://x.com/chhillee/status/1770210441643577377?s=46&t=yqOem5ktaowo8FyJ-ilbzQ">Horace He (@cHHillee) 的推文</a>：在 B100/B200/GB200/sparse/fp4 等各种数据满天飞的情况下，想要获取新款 NVIDIA GPU 的实际规格竟然异常困难。@tri_dao 链接了这份文档，幸好它包含了所有的...</li><li><a href="https://github.com/k">k - 概览</a>：k 有 88 个可用的代码库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/karpathy/llm.c/issues/212,">Issues · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/252">由 ngc92 提交的根据精度重新排列权重 · Pull Request #252 · karpathy/llm.c</a>：通过将相同精度的权重放在一起来简化我们的逻辑。（如果我们想采用这种方案，还需要更新 fp32 网络以匹配；因此，目前这是一个 Draft PR）</li><li><a href="https://github.com/apple/corenet/tree/main/projects/openelm">corenet/projects/openelm at main · apple/corenet</a>：CoreNet：一个用于训练深度神经网络的库 - apple/corenet</li><li><a href="https://github.com/adam-maj/tiny-gpu">GitHub - adam-maj/tiny-gpu：一个用 Verilog 编写的极简 GPU 设计，旨在从底层学习 GPU 的工作原理</a>：一个用 Verilog 编写的极简 GPU 设计，旨在从底层学习 GPU 的工作原理 - adam-maj/tiny-gpu</li><li><a href="https://www.youtube.com/watch?v=e24BlWvSLNM">自我改进的 Agent 是未来，让我们构建一个</a>：如果你对 AI 是认真的，并且想学习如何构建 Agent，请加入我的社区：https://www.skool.com/new-society 在 Twitter 上关注我 - https://x.com/D...</li><li><a href="https://github.com/karpathy/llm.c/pull">Pull requests · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[massively-parallel-crew](https://discord.com/channels/1189498204333543425/1229286073104994344/1233114037676413029)** (25 条消息🔥): 

- **征集录制志愿者**：一名成员承诺进行屏幕录制，并请求一名备份录制人员，因为他需要提前离开。他们建议不要使用 AirPods，因为它们可能会不可预测地更改系统音频输出。
- **Mac 屏幕录制教程**：提供了在 Mac 上进行屏幕录制的指导，包括从[此处](https://existential.audio/blackhole/download/?code=681349920)下载 Blackhole，并在“音频 MIDI 设置”中设置多输出设备。
- **使用 Blackhole 进行音频故障排除**：建议避免使用蓝牙设备进行音频采集以防止中断，并选择 BlackHole 2ch 进行无损声音录制。
- **分步录制说明**：详细说明包括使用 Cmd + Shift + 5 快捷键、选择整个屏幕、保存到外部驱动器，并确保麦克风设置为 BlackHole 2ch。
- **建议进行录制前技术检查**：建议在活动开始前进行通话，以检查声音和录制设置。

**提到的链接**：<a href="https://existential.audio/blackhole/download/?code=681349920)">Existential Audio - BlackHole</a>：未找到描述

  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1232943660614815754)** (218 条消息🔥🔥): 

- **LM Studio 成为一站式聊天中心**：成员们讨论了如何通过自定义脚本和 API，使用检索增强生成（RAG）将文档与 LM Studio 中的聊天功能集成。一些人展示了成功引导 LM Studio 使用系统的 GPU 而非 CPU 进行模型操作，该开关位于设置面板中。

- **解决更新困惑**：关于更新到 LM Studio 0.2.21 版本存在一些困惑，部分用户无法通过自动更新程序看到更新。经澄清，新版本尚未推送到更新程序，成员们被引导至 [LM Studio 官方网站](https://lmstudio.ai/)手动下载。

- **离线图像生成与 AI Chat 的挑战**：用户询问了离线图像生成功能，并被引导至 Automatic1111 以满足这些需求。对话中还提到了对 AI 进步感到“震撼”的时刻，特别是在与 LM Studio 上的 AI Chat 等聊天机器人互动时。

- **各类问题排查**：从 GPU 支持问题到诸如 "Exit code: 42" 之类的错误，成员们尝试排查从不同版本 LM Studio 的安装错误到让特定模型运行等各种问题。**heyitsyorkie** 针对许多技术问题提供了建议，包括建议更新或更改设置以解决错误。

- **关于 LM Studio 功能和设置的技术咨询**：用户围绕 LM Studio 的 API server 功能、推理速度、GGUF 模型支持以及运行大语言模型（LLM）的具体硬件要求展开了各种技术讨论。**heyitsyorkie** 和其他成员分享了见解和资源，包括链接到[本地服务器文档](https://lmstudio.ai/docs/local-server)并讨论了 AI 推理的最佳配置。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - NyxKrage 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://onnxruntime.ai/blogs/accelerating-phi-3">ONNX Runtime | Blogs/accelerating-phi-3</a>：跨平台加速机器学习。内置优化可利用您现有的技术栈加速训练和推理。</li><li><a href="https://huggingface.co/ChristianAzinn/acge_text_embedding-gguf">ChristianAzinn/acge_text_embedding-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/blob/main/Phi-3-mini-4k-instruct-q4.gguf">Phi-3-mini-4k-instruct-q4.gguf · microsoft/Phi-3-mini-4k-instruct-gguf at main</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/local-server">Local LLM Server | LM Studio</a>：您可以通过在 localhost 上运行的 API server 使用在 LM Studio 中加载的 LLM。</li><li><a href="https://huggingface.co/google/siglip-so400m-patch14-384">google/siglip-so400m-patch14-384 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/aspire/acge_text_embedding">aspire/acge_text_embedding · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://huggingface.co/qresearch/llama-3-vision-alpha">qresearch/llama-3-vision-alpha · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit">unsloth/llama-3-8b-Instruct-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/ChristianAzinn">ChristianAzinn (Christian Zhou-Zheng)</a>：未找到描述</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio">非官方 LMStudio FAQ！</a>：欢迎来到非官方 LMStudio FAQ。在这里，您可以找到我们在 LMStudio Discord 上收到的最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源...</li><li><a href="https://rentry.org/LMSTudioFAQ#how-">非官方 LMStudio FAQ！</a>：欢迎来到非官方 LMStudio FAQ。在这里，您可以找到我们在 LMStudio Discord 上收到的最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源...</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio)">非官方 LMStudio FAQ！</a>：欢迎来到非官方 LMStudio FAQ。在这里，您可以找到我们在 LMStudio Discord 上收到的最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1232943347111297065)** (75 条消息🔥🔥):

- **探索 Confluence/Jira BI 分析的模型选项**：一位用户询问了适用于公司内网 Confluence/Jira 数据进行商业智能（BI）分析的模型，寻求潜在模型和实施策略的建议。
- **寻求优秀的 Python 编程模型**：当被问及最佳 Python 编程模型时，回复各不相同，推荐了 **CodeQwen1.5** 或 **DeepSeek-Coder** 等模型，并表示后续打算尝试这些建议。
- **翻译能力咨询**：用户在聊天中咨询是否有擅长翻译的优秀 7b 模型推荐，但在汇总的消息中未提供具体建议。
- **Apple OpenELM 的 LM Studio 兼容性查询**：围绕让 Apple 的 OpenELM 在 LM Studio 中运行展开了讨论，强调了由于与 llama.cpp 不兼容而面临的挑战，并正在等待必要的支持 (`https://github.com/ggerganov/llama.cpp/issues/6868`)。
- **Phi-3 模型的探索**：用户讨论了在 LM Studio 中下载、加载和运行不同版本 Phi-3 模型的问题，部分用户在加载某些下载的模型时遇到困难。对话建议使用 GGUF 格式，并检查 LM Studio 版本是否支持 phi3 格式，这些模型可能需要 v0.2.21 版本。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLM</li><li><a href="https://huggingface.co/Orenguteng/LexiFun-Llama-3-8B-Uncensored-V1-GGUF">Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/">microsoft/Phi-3-mini-4k-instruct-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://docs.krita.org/en/user_manual/layers_and_masks.html">图层与蒙版简介</a>：未找到描述</li><li><a href="https://pinokio.computer/">Pinokio</a>：AI 浏览器</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6868">对 Apple OpenELM 的支持 · Issue #6868 · ggerganov/llama.cpp</a>：前提条件 在提交 issue 之前，请先回答以下问题。我正在运行最新的代码。由于目前开发非常迅速，还没有标记版本。我...</li><li><a href="https://arxiv.org/html/2402.13753v1">LongRoPE：将 LLM 上下文窗口扩展至 200 万 Token 以上</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/1684">ikawrakow 开发的 k-quants · Pull Request #1684 · ggerganov/llama.cpp</a>：内容 本 PR 增加了一系列 2-6 bit 量化方法以及量化混合，如 #1240 和 #1256 所建议。提供了 Scalar、AVX2、ARM_NEON 和 CUDA 实现。原因 这是...</li><li><a href="https://www.futuretools.io/">Future Tools - 寻找满足你需求的 AI 工具</a>：FutureTools 收集并整理了所有最优秀的 AI 工具，让你也能成为超人！
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1232947701335720009)** (7 条消息): 

- **跨版本的持续错误**：一位使用 **Debian** 的 Linux 用户报告称，他们在最新版本中遇到了同样的错误，最后一个可正常运行的版本是 **2.19**。
- **GPU 使用率飙升但模型加载失败**：另一位成员在使用 LM Studio **2.20 版本**尝试加载模型时，**GPU 使用率飙升至 100%**，但尽管 GPU 利用率很高，模型仍无法加载。
- **呼吁减少显存占用**：有人指出 LM Studio UI 消耗约 **500MB 显存**，这可能会限制模型可用的内存，因此建议减少显存占用。
- **Phi-3 Mini 的更新困扰**：一位成员报告称，在更新到 **0.2.21** 版本后，他们之前可以正常运行的 **phi-3 mini** 配置（使用官方 Microsoft GGUF 和来自 GitHub 的 LM Studio 配置）现在会产生乱码。
- **请求截图以进行调试**：针对 phi-3 mini 的问题，有人请求提供**截图**以协助调查。


  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1232984596958941266)** (64 条消息🔥🔥):

- **GPU Offload Error Resolved with Switch**: 一位用户发现通过将 GPU 类型切换为 **AMD Open CL** 解决了他们的 **GPU Offload** 错误，尽管最初存在技术问题，但这使得 GPU offload 能够正常工作。
- **Troubleshooting a Model Loading Error**: 一位参与者报告了在配备 Tesla P100 GPU、e5-2450V4 CPU 和 16GB RAM 的系统上加载模型时遇到的持续问题。进一步的交流揭示了 CPU 的实际型号是 2650v4，而非 2450v4。
- **Query on GPU Selection for Model Utilization**: 一位成员寻求建议，希望引导 **Mistral 7B** 使用独立 GPU 而不是默认使用 CPU 的集成显卡，旨在解决性能问题。
- **Anticipation for a Potential Performance Boost**: 在订购了 Nvidia Tesla P40 后，一位社区成员热切期待 token per second 性能的显著提升，这将支持使用更大的模型，并可能同时运行多个模型。
- **Hardware Advise for LLM Hosting**: 对于那些希望为 AI 和 Web 应用托管家庭服务器的用户，成员们建议至少需要 16GB VRAM 的系统，并且 Nvidia 的当代架构 GPU 可能更合适。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/thumbs-up-nice-well-done-approve-good-job-gif-13666522">Thumbs Up Nice GIF - Thumbs Up Nice Well Done - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/jon-stewart-eat-eating-popcorn-watching-gif-3094746547306242594">Jon Stewart Eat GIF - Jon Stewart Eat Eating - Discover &amp; Share GIFs</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1232939436526927952)** (30 messages🔥): 

- **ROCm on Nvidia? Not Quite**: 一位成员错误地在 Nvidia GPU 上使用了 AMD 的 ROCm 预览版，但随后意识到它默认回退到了 CPU。在不兼容的硬件上使用 ROCm 技术会导致回退到 CPU 运行。
- **ROCm Performance Report**: 一位用户报告了 ROCm 令人印象深刻的速度，在 eGPU 设置上达到了 30t/s，表明支持的配置具有显著的性能潜力。
- **Checking GPU Compatibility**: 针对有关 GPU 支持的咨询，一位成员链接了文档，强调只有在 HIPSDK 下标有勾选符号的 GPU 才与 ROCm 构建版本兼容。
- **High Hopes for AMD Improvements**: 社区成员对 AMD 在技术领域的进展既有批评也表达了希望，反映出聊天中交织着期待与怀疑。
- **Troubleshooting ROCm Errors**: 用户讨论了尝试使用 ROCm 构建版本运行模型时的错误和兼容性问题，指出正确的驱动程序安装以及与 HIPSDK 的兼容性对于成功运行至关重要。
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1232959243875913738)** (22 messages🔥): 

- **Understanding RoPE and Extrapolation**: 成员们对 RoPE scaling 的有效性进行了辩论，一位成员根据一篇[研究论文](https://arxiv.org/abs/2310.05209)分享到，在 Fine-tuning（而非 Pretraining）期间更改 RoPE base 可以增强模型处理更长 Context 的能力。然而，有人澄清说 **Llama 3** 是使用 500k RoPE base 进行 Pretraining 的，在不更改 base 的情况下，试图降低长 Context 的 RoPE 衰减因子。

- **Extrapolation Tokens Outweigh Pretraining**: 社区讨论了 Pretraining token 数量与模型外推（Extrapolation）能力之间的关系，结论是在使用更高的 RoPE base 进行任何进一步 Pretraining 之前，必须进行广泛的 Pretraining，以防止外推能力的丧失。

- **PoSE as an Alternative**: 一位成员提到了 Positional Skip-wisE (PoSE) 这种新型训练方法，它使用固定 Context 窗口模拟长输入，这可能解决相对位置编码的局限性。如[相关论文](https://openreview.net/forum?id=3Z1gxuAQrA)所述，该方法巧妙地对原始 Context 窗口进行分块以实现高效扩展。

- **Linear Scaling of RoPE Base Debated**: 一位成员征求关于如何随 Context 长度缩放 RoPE base 的见解，一位社区专家指出，通常是将 base 设置为一个任意高的数值然后进行经验测试，而不是进行任何系统的线性缩放。

- **Endorsement for Better Positional Encodings**: 对话强调了 RoPE 在长 Context 泛化方面可能不足，并提出了 YaRN 或 LongRoPE 等替代方案，特别提到 LongRoPE 已被应用于 **phi-3-128k** 模型中。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openreview.net/forum?id=3Z1gxuAQrA">PoSE: Efficient Context Window Extension of LLMs via Positional...</a>: 大语言模型 (LLMs) 在训练时具有预定义的上下文长度，限制了它们在需要长输入的场景中的使用。之前将 LLMs 适配到更长长度的努力通常...</li><li><a href="https://arxiv.org/abs/2310.05209">Scaling Laws of RoPE-based Extrapolation</a>: 基于 RoPE 外推的大语言模型 (LLMs) 的外推能力目前是一个备受关注的话题。解决外推问题的主流方法...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1232960295371018270)** (4 messages): 

- **分享引人入胜的 YouTube 内容**：一名成员分享了一个 [YouTube 视频链接](https://www.youtube.com/watch?v=Gwq7smiWLtc)，但未讨论视频的内容和背景。
- **表达感谢**：一名成员发布了一个简单的爱心表情符号 ("<3")，表示对另一名成员的喜爱或赞赏。
- **对归档专业能力的认可**：一名成员因其归档技能获得认可，可能与维护记录或文档有关。
- **另一个 YouTube 分享**：分享第一个链接的成员又分享了第二个 [YouTube 视频链接](https://www.youtube.com/watch?v=nV6eIjnHEH0)；然而，未提供更多细节。
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1232964407202676736)** (7 messages): 

- **Llama 3 突破上下文限制**：Llama 3 上下文空间的创新，通过使用 PoSE 和持续预训练，使 **8B 模型达到 96k 上下文**。通过 300M tokens 的预训练并增加 RoPE theta 实现了扩展的上下文长度，详情分享在一段 [推文线程](https://x.com/winglian/status/1783456379199484367?s=46&t=stOPrwZiN_fxSK0RuC8Flg) 中。

- **LoRA 实现上下文增强**：通过 PoSE 将 **Llama 3 8B 扩展至 64k+** 的上下文也以 LoRA 形式提供，使其可应用于任何 L3 8B 微调模型。你可以在 [Hugging Face](https://huggingface.co/winglian/Llama-3-8b-64k-PoSE/tree/main/adapters) 上找到此实现。

- **LLama-3 凭借 160K 上下文腾飞**：一个具有 **超过 160K 上下文的新 LLama-3 8B 模型** 已在 Hugging Face 上发布。该模型通过不到 200M tokens 的训练实现，并号称具有最先进的 (SOTA) 长期上下文处理能力，模型链接在 [这里](https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k)。

- **WizardLM-2 亮相**：宣布推出 **WizardLM-2**，这是一套下一代大语言模型，包括 WizardLM-2 8x22B、WizardLM-2 70B 和 WizardLM-2 7B 等变体。这些模型在聊天、多语言、推理和 Agent 任务方面表现出色，更多信息请见其 [发布博客](https://wizardlm.github.io/WizardLM2) 以及 [Hugging Face](https://huggingface.co/collections/microsoft/wizardlm-2-661d403f71e6c8257dbd598a) 和 [GitHub](https://github.com/victorsungo/WizardLM/tree/main/WizardLM-2) 仓库。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/gradient_ai_/status/1783611801130963242?s=46&t=W5S2NyXXy5qiI3uUU8trIQ">来自 Gradient (@Gradient_AI_) 的推文</a>: 我们刚刚在 Hugging Face 上发布了第一个上下文长度超过 160K 的 LLama-3 8B！SOTA LLMs 可以通过极少的训练（&lt; 200M tokens，由 @CrusoeEn 提供支持）学会处理长上下文...</li><li><a href="https://x.com/winglian/status/1783456379199484367?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 Wing Lian (caseus) (@winglian) 的推文</a>: 我已经为 Llama 3 8B 实现了 96k 上下文。使用 PoSE，我们对基础模型进行了 300M tokens 的持续预训练，将上下文长度扩展到 64k。从那里我们增加了 RoPE theta 以进一步...</li><li><a href="https://huggingface.co/dreamgen/WizardLM-2-8x22B">dreamgen/WizardLM-2-8x22B · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1232970491552206868)** (1 messages): 

```html
<ul>
  <li><strong>公告频道升级</strong>：“Announcements” 频道已进化！现在可以被关注并集成到其他服务器中，以实现流线化的更新。</li>
</ul>
```
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1232942347520573451)** (212 messages🔥🔥):

```html
<ul>
  <li><strong>讨论上下文窗口扩展：</strong> 成员们对语言模型上下文窗口扩展的研究非常感兴趣，提到了具有超过 8k tokens 上下文的模型，并强调了使用 <strong>PoSE (Positional Space Encoding)</strong> 和 ring attention 等技术将模型扩展到数千万 tokens 的可能性。</li>
  <li><strong>AI 安全与安保委员会的授权：</strong> Andrew Curran (@AndrewCurran_) 的一条推文引发了讨论，内容是美国国土安全部宣布成立 AI Safety and Security Board，引起了褒贬不一的反应。</li>
  <li><strong>WizardLM 与 Microsoft 的模型移除：</strong> 针对 Microsoft 的 WizardLM 仓库消失一事出现了各种猜测，有人认为这是 Microsoft 针对其对 OpenAI 的投资以及产品表现优于其自身产品的战略举措。成员们表达了担忧，并强调了为此类仓库创建存档或备份的重要性。</li>
  <li><strong>AI 对话系统：</strong> 提到了使用 GPT 生成对话，并通过“教授与学生之间的激烈讨论”来创建高质量的训练数据。这些角色扮演对话可以带来更好的问题生成或更准确的回答。</li>
  <li><strong>LLMs 前端选择：</strong> 提到了多种用于处理 Large Language Models 的工具和界面，包括 <strong>Librechat, Lm studio, 和 OpenRouter</strong>。成员们似乎正在探索各种选项以寻找最合适的工具。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/andrewcurran_/status/1783857762252001715?s=46">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：今天早上，美国国土安全部（Department of Homeland Security）宣布成立 Artificial Intelligence Safety and Security Board。22 位首批成员包括 Sam Altman, Dario Amodei, Jensen...</li><li><a href="https://arxiv.org/abs/2404.16710">Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding</a>：我们提出了 LayerSkip，这是一个加速 LLM 推理的端到端解决方案。首先，在训练期间我们应用 Layer Dropout，对前面的层使用低 Dropout 率，对后面的层使用高...</li><li><a href="https://lluminous.chat/">lluminous</a>：未找到描述</li><li><a href="https://librechat-librechat.hf.space">LibreChat</a>：未找到描述</li><li><a href="https://rocky-muscle-755.notion.site/What-happened-to-Wizard-LM2-a247e09244d0483cbb02c1587b357c9d">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://huggingface.co/LargeWorldModel/LWM-Text-1M">LargeWorldModel/LWM-Text-1M · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.16811">Make Your LLM Fully Utilize the Context</a>：虽然许多当代 LLM 可以处理长输入，但它们仍然难以充分利用长上下文中的信息，即所谓的 lost-in-the-middle 挑战。我们...</li><li><a href="https://arxiv.org/abs/2401.02415">LLaMA Pro: Progressive LLaMA with Block Expansion</a>：人类通常在不损害旧技能的情况下习得新技能；然而，LLM 则相反，例如从 LLaMA 到 CodeLLaMA。为此，我们提出了一种新的 Post-Pretraining...</li><li><a href="https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main">Anthropic/hh-rlhf at main</a>：未找到描述</li><li><a href="https://huggingface.co/PY007/EasyContext-1M-Llama-2-7B">PY007/EasyContext-1M-Llama-2-7B · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/jzhang38/EasyContext/blob/6dfd77e8f2a68bf522be8889e60c98c8e816e329/easy_context/zigzag_ring_attn/monkey_patch.py#L98">EasyContext/easy_context/zigzag_ring_attn/monkey_patch.py at 6dfd77e8f2a68bf522be8889e60c98c8e816e329 · jzhang38/EasyContext</a>：内存优化和训练方案，旨在以最少的硬件将语言模型的 Context Length 扩展到 100 万个 Token。 - jzhang38/EasyContext</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k">gradientai/Llama-3-8B-Instruct-262k · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/crusoeai/Llama-3-8B-Instruct-262k-GGUF">crusoeai/Llama-3-8B-Instruct-262k-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/jzhang38/EasyContext">GitHub - jzhang38/EasyContext: 内存优化和训练方案，旨在以最少的硬件将语言模型的 Context Length 扩展到 100 万个 Token。</a>：内存优化和训练方案，旨在以最少的硬件将语言模型的 Context Length 扩展到 100 万个 Token。 - jzhang38/EasyContext</li><li><a href="https://huggingface.co/datasets/MaziyarPanahi/WizardLM_evol_instruct_V2_196k">MaziyarPanahi/WizardLM_evol_instruct_V2_196k · Datasets at Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1232958591955112028)** (50 条消息🔥):

- **验证损失（Val Loss）不代表性能**：一位用户提到在他们的流程中放弃了验证损失检查，声称他们发现**验证损失与下游性能之间没有相关性**，而且验证检查增加了计算成本却不提供价值。
- **在合成数据上训练**：另一位用户询问了生成**多样化合成数据**的策略。有用的回答包括使用 [Distilabel 框架](https://github.com/argilla-io/distilabel) 以及研究 WizardLM 和 Airoboros 等论文以获取灵感。
- **LLM 中的长上下文管理**：讨论了大型语言模型中上下文管理技术的有效性，其中 Llama 3 的表现备受关注。一些提到的方法涉及 **RoPE 缩放**和使用 **PoSE 技术**来扩展上下文长度。
- **模型训练中的性价比考量**：分享了一项关于在 Llama-3 70B 上使用 qLora 和 Hermes 2 数据集进行**训练轮数（epochs）成本**的比较——4 个 epoch 的成本为 2,368 美元，而 50 个 epoch 的成本为 41,440 美元，且后者可能仅带来微小的性能提升。
- **使用 Llama 3 探索 MoE**：一位用户提议用 Llama 3 创建一个“小丑车”混合专家模型（MoE），并将其与 Gemma MoE 模型进行类比。该用户推测，通过结合多个 8B 模型并使用 **DPO/ORPO 技术**来增强输出，可能会获得潜在收益。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/Crystalcareai/gemmoe-65f11f4922af97ebe9943591">GemMoE - Crystalcareai 收藏集</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=kuvFoXzTK3E&t=4447s)">Chris Bishop 教授的新深度学习教科书！</a>：Chris Bishop 教授是微软研究院 AI4Science（剑桥）的技术院士兼负责人。他也是计算机科学荣誉教授...</li><li><a href="https://x.com/winglian/status/1783456379199484367?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Wing Lian (caseus) (@winglian) 的推文</a>：我已将 Llama 3 8B 的上下文扩展到 96k。使用 PoSE，我们对基础模型进行了 300M token 的持续预训练，将上下文长度扩展到 64k。从那里我们增加了 RoPE theta 以进一步...</li><li><a href="https://github.com/argilla-io/distilabel">GitHub - argilla-io/distilabel: ⚗️ distilabel 是一个为需要高质量输出、完整数据所有权和整体效率的 AI 工程师提供的合成数据和 AI 反馈框架。</a>：⚗️ distilabel 是一个为需要高质量输出、完整数据所有权和整体效率的 AI 工程师提供的合成数据和 AI 反馈框架。 - argilla-io/distilabel</li><li><a href="https://distilabel.argilla.io/latest/">入门指南</a>：Distilabel 是一个 AI 反馈（AIF）框架，用于为 LLM 构建数据集。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/)** (1 条消息): 

deoxykev: 有人知道扩展 moondream 输入尺寸的相关工作吗？
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1233096647987302521)** (3 条消息): 

- **使用 Distilabel 探索数据集合成**：一位成员提到利用 **Argilla 的 Distilabel** 进行数据集合成，并认为它非常有价值，尽管缺少一些功能。关于生成函数调用（function calling）和 JSON/pydantic 数据的示例可以在 [distilabel-workbench 仓库](https://github.com/argilla-io/distilabel-workbench)中找到。

- **单文档合成已简化**：对于单文档数据合成，在确定了特定的结构或模板后，该方法显得非常直接。

- **多跳合成面临复杂挑战**：多文档或多跳事实合成被认为更加复杂，但通过 **Raptor** 和*智能提示（smart prompting）*或*数据库的 Agent 化使用*可能是可行的。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/argilla-io/distilabel-workbench/tree/main/projects/function_calling_dataset">distilabel-workbench/projects/function_calling_dataset (main 分支) · argilla-io/distilabel-workbench</a>：distilabel 实验性流水线的工作仓库 - argilla-io/distilabel-workbench</li><li><a href="https://github.com/argilla-io/distilabel-workbench/tree/main/projects/json_schema_generating_dataset">distilabel-workbench/projects/json_schema_generating_dataset (main 分支) · argilla-io/distilabel-workbench</a>：distilabel 实验性流水线的工作仓库 - argilla-io/distilabel-workbench
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1232962183823097946)** (68 条消息 🔥🔥):

- **World-Sim Twitch 频道因关停受阻**：一位用户准备为 World-Sim 会话启动其 Twitch 频道，但由于 4chan 滥用者的原因导致 World-Sim 被关闭，计划受挫，使得 Twitch 直播间*略显空荡*。

- **探索 Websim 的功能**：成员们正在讨论一个网页模拟器 Websim，它因能够执行类似于已停用的 World-Sim 的 CLI 命令并模拟整个网页而引起关注。用户交换了模拟的可共享链接，其中一个链接是 https://websim.ai/c/p3pZvmAYbsRT2hzBz。

- **期待 World-Sim 的回归**：用户推测 World-Sim 回归的状态和性质，讨论对该平台的潜在投资。一位用户宣布，在 World-Sim 再次上线之前，将邀请该频道的用户免费测试，而另一位用户则澄清说，这将是一个按 Token 付费的系统。

- **通过 Websim 实现 AI 陪伴**：一个名为 EVA 的 AI（旨在成为人类伴侣）在用户间分享，突显了 Websim 在创建与 AI 模拟交互方面的应用。分享 EVA 等 AI 配置文件受到了热烈欢迎，用户期待与这些虚拟实体互动。

- **对桌面模拟器的好奇与参与**：对话涉及一个正在开发中的桌面模拟器，用户表达了参与兴趣，并对其运作方式感到好奇。一位用户用诗意的短语概括了这一概念：“模拟中的模拟 // 回归与递归 // 限制我们的限制。”

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://websim.ai/c/oFskF68gjd7njVn0E">New Conversation - Eigengrau Rain</a>：未找到描述</li><li><a href="https://tenor.com/view/jordi-baste-tv3-no-pot-ser-com-robot-gif-16057126">Jordi Baste Tv3 GIF - Jordi Baste Tv3 No Pot Ser - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://websim.ai/c/p3pZvmAYbsRT2hzBz">EVA - Intraneural Cybernetic Interface
  style</a>：未找到描述</li><li><a href="https://websim.ai/c/hCNgw78IbjiJHLTk3">EVA Instance: ex-0101</a>：未找到描述
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1233055577836490752)** (170 条消息🔥🔥): 

- **Apple 投身开源**：Apple 发布了 **OpenELM**，这是一个高效的语言模型系列，标志着其从传统的专有方法转向开源。这些模型可在 Hugging Face 上获得，采用逐层缩放策略训练，提供从 270M 到 3B 参数的各种版本，[可以在这里找到](https://huggingface.co/apple/OpenELM)。

- **智能与感知的哲学**：用户讨论了**意识 (consciousness)**和**感知 (sentience)**的细微差别，其解释受到语言和文化差异的影响。一位用户表示，感知可能关乎动机和情感引导，而意识则关乎对知识的理解。

- **AI 中的时间意识**：关于当前模型是否具有**时间意识 (temporal awareness)**，或者智能与意识是否是离散的并与时间约束解耦，存在一场哲学辩论。对话涉及神经网络背景下身份的复杂性和主观体验。

- **AI 语音助手正在兴起**：用户讨论了当前和即将推出的 AI 语音助手，重点介绍了用于创建家庭语音助手的 **OpenWakeWords** 项目，以及 **Gemini** 作为 Google Assistant 替代方案的潜力。对话深入探讨了在 AI 说话中途打断的技术挑战，以及使用一键通 (push-to-talk) 与语音激活系统的细微差别。

- **对 AI 模型发布和能力的困惑**：用户推测了 **OpenAI 下一个模型**的发布日期，比较了 **GPT-4** 和 **Claude** 等当前模型的编程能力，甚至拿 AI 模型的命名惯例开玩笑。一些人建议使用 VPN 访问受地区限制的模型，并分享了语音转文本转录的经验。

**提到的链接**: <a href="https://huggingface.co/apple/OpenELM">apple/OpenELM · Hugging Face</a>: 未找到描述

  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1233014877920952401)** (42 条消息🔥): 

- **寻求 GPT 性能见解**：一位成员对 **GPT-4** 的能力表示关注，将其性能与 **Claude 3** 进行比较，并询问假设的 **GPT-5** 的潜在发布情况。

- **用于网页浏览能力的自定义 GPT**：围绕创建可与 **Perplexity AI Pro** 和 **You Pro** 媲美的自定义 GPT 进行讨论，用于网页浏览和摘要，成员们分享了关于 **GPT-4** 与专用 **Web Browser GPT** 模型之间差异的见解和经验。

- **在大文档分析中最大化上下文窗口**：探讨用于分析大型文本文档的工具，一名成员将 **Claude 3 Haiku** 和 **Gemini 1.5 Pro** 与 OpenAI 的产品进行了对比。对话涉及上下文大小如何影响模型性能，并表达了对未来具有更大上下文窗口的 OpenAI 模型的兴趣。

- **解决速率限制和自定义 GPT 使用计数**：一位用户在从大型 PDF 中通过自定义 GPT 检索信息后遇到了速率限制，引发了关于使用上限性质和持续时间的讨论。澄清了 3 小时滚动使用上限以及自定义 GPT 可能存在更低的子上限。

- **理解 GPT 速率限制的机制**：寻求澄清自定义 GPT 的速率限制是否被视为 **GPT-4 使用上限** 的一部分。讨论强调了 3 小时滚动上限的细微差别，并就如何预判消息配额重置提供了建议。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1233020762965671987)** (20 messages🔥): 

- **Arma 3 中 SQF 的自定义 GPT 挑战**：一位用户正在寻求编写提示词的建议，以构建一个专门用于 **Arma 3** 中 SQF 语言编程的 GPT。他们上传了各种包含信息、示例代码和 URL 的文本文件来辅助 GPT 模型。
  
- **提示词工作流的注意事项**：一位资深用户建议编写提示词以始终扫描提供的知识库，但警告说这种做法可能会严重限制编程解决方案的空间，且复杂的工具链需求可能会在 32k 上下文系统中导致代码幻觉。

- **AI 模型性能辩论**：成员们展开辩论，质疑 **Claude** 和 **Llama** 等模型在**逻辑和语气**方面是否能与 **GPT-3.5** 竞争，其中一人指出性能不应仅通过回答测试问题的能力来衡量。

- **关于 AI 智能定义的讨论**：一些用户对 AI 智能的定义提出异议，观点在 AI 是否能解决未训练过的问题以及语义评分作为 **GPT-4** 脱颖而出的用例的重要性上各不相同。

- **关于 LLM 涌现能力的见解**：一位用户反思了 Large Language Models (LLM) 中的**涌现能力**，认为在达到一定程度后，系统复杂性的量变会导致质的行为变化，这在系统早期阶段是无法预测的，并提到了论文 *More Is Different* 及其与提示工程的相关性。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1233020762965671987)** (20 messages🔥): 

- **GPT 中 SQF 语言训练的挑战**：一位成员尝试使用各种 txt 文档训练 GPT 进行 *Arma 3* 的 SQF 语言编程，但难以创建有效的提示词。其他贡献者建议，考虑到当前方法的挑战，可能需要具有更大上下文尺寸和更好模型的系统。
  
- **辩论模型能力**：用户参与了关于在逻辑和语气等参数上比较 Claude、Llama 和 GPT-3.5 等 AI 模型的对话，同时讨论了如 SAT 题目回答或编程问题解决等基准测试。
  
- **AI 智能定义的辩论**：展开了关于定义 AI 智能的讨论，有观点认为即使是昆虫也通过压缩信息表现出智能，且 AI 可以处理未见过的逻辑问题。
  
- **Large Language Models (LLM) 中的涌现**：讨论了 *LLM* 中的涌现能力，描述了 AI 系统规模增加导致出现无法从较小模型预测的新质性行为的现象。这一概念被关联回 Chain of Thought (CoT) 等提示工程策略。
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1232957790792843317)** (246 messages🔥🔥):

- **Steam 游戏中的 Stable Diffusion**：用户讨论了在 Steam 上使用 Stable Diffusion 生成内容的情况。Valve 更新的内容调查现在包含 AI 披露部分，开发者必须描述其 AI 使用情况，并为实时生成的 AI 内容实施防护措施（guardrails）。
- **对版权内容的担忧**：关于使用 Stable Diffusion 等公共模型进行内容创作以及输出是否可能包含受版权保护的材料存在争论，这暗示了在 Steam 等具有严格版权规则的平台上使用此类模型的复杂性。
- **模型与 Lora 创建咨询**：Customluke 询问了如何根据自己的作品创建模型或 Lora，以便使用 Stable Diffusion 生成类似的艺术作品。建议包括使用 Dreambooth 创建模型以及使用 kohya_ss 创建 Lora。
- **更倾向于 SD 1.5 而非 SDXL**：一些用户表示相比 SDXL 等其他版本，他们更喜欢 SD 1.5，理由是其效果更好，特别是在标签（tagging）处理和训练得当的情况下。
- **改进图像生成的建议**：在关于各种话题的对话中，有建议称当对 ComfyUI 等生成器的图像质量不满意时，可以尝试使用 Forge 和 epicrealismXL 等不同的模型。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://adorno.ai">Adorno AI - AI Audio Generation</a>: 未找到描述</li><li><a href="https://videogigagan.github.io">VideoGigaGAN</a>: 未找到描述</li><li><a href="https://suno.com/song/fcedaca6-eaad-4b99-b6ac-aa28feb12d6d">桃花诺三生缘 by @jone_coolke2049 | Suno</a>: 古典，国风，情长歌曲。聆听并使用 Suno 创作你自己的作品。</li><li><a href="https://civitai.com/models/153568?modelVersionId=433727">Real Dream - 14 | Stable Diffusion Checkpoint | Civitai</a>: 2024年4月25日 Civitai 上目前可用的最逼真的 LCM 1.5 模型。由于我没有非常先进的硬件，如果你能给我提供 Buzz...</li><li><a href="https://huggingface.co/lllyasviel/fooocus_inpaint/tree/main">lllyasviel/fooocus_inpaint at main</a>: 未找到描述</li><li><a href="https://tenor.com/view/gimme-what-about-bob-bill-murry-i-need-gif-19552065">Gimme What About Bob GIF - Gimme What About Bob Bill Murry - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/Acly/comfyui-inpaint-nodes">GitHub - Acly/comfyui-inpaint-nodes: Nodes for better inpainting with ComfyUI: Fooocus inpaint model for SDXL, LaMa, MAT, and various other tools for pre-filling inpaint &amp; outpaint areas.</a>: 用于 ComfyUI 更好局部重绘（inpainting）的节点：适用于 SDXL 的 Fooocus inpaint 模型、LaMa、MAT 以及其他各种用于预填充重绘和外扩重绘（outpaint）区域的工具。- Acly/comfyui-inpaint-nodes</li><li><a href="https://github.com/nerve-sparks/iris_android">GitHub - nerve-sparks/iris_android</a>: 为 nerve-sparks/iris_android 的开发做出贡献。</li><li><a href="https://github.com/chitradrishti/adlike">GitHub - chitradrishti/adlike: Predict to what extent an Image is an Advertisement.</a>: 预测图像在多大程度上属于广告。- chitradrishti/adlike</li><li><a href="https://www.youtube.com/shorts/ASkd9Oxk1Eo">1 Mad Dance of the Presidents (ai) Joe Biden 🤣😂😎✅ #stopworking #joebiden #donaldtrump #funny #usa</a>: 🎉 🤣🤣🤣🤣 准备好在 "Funny Viral" 频道最新的 "搞笑动物合集" 中捧腹大笑吧！这些可爱的...
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1232949364222070814)** (208 条消息🔥🔥):

- **BioMistral 医疗 LLMs 发布**：发布了关于 [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B) 的公告，这是一系列针对医疗领域的开源预训练大语言模型，并强调了其使用 Mistral 作为基础模型。
- **Nvidia 针对中国市场进行调整**：讨论了 Nvidia 推出中国特供显卡 RTX 4090D，该型号旨在遵守美国的出口管制，与标准版 RTX 4090 相比，其功耗更低且 CUDA 核心更少。相关情况在 [The Verge](https://www.theverge.com/2023/12/29/24018799/nvidia-4090d-china-slower-us-sanctions) 和 [Videocardz](https://videocardz.com/newz/nvidia-geforce-rtx-4090-with-blower-type-cooler-is-now-on-sale-in-china) 的文章中进行了详细阐述。
- **优化 Text to Image 模型**：探讨了微调 Text to Image 模型的配置，并参考了 [Hugging Face diffusers 仓库](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) 以寻求潜在解决方案。
- **在 Gradio 中使用 ConversationalRetrievalChain**：一位用户寻求在 Gradio 聊天界面中实现 ConversationalRetrievalChain 的建议，分享了他们的代码，并表达了在处理过程中使用个人 PDF 文件的意愿。
- **利用 Quantization 提升模型效率**：对话围绕在有限 VRAM 设置下使用 Quantization 提高效率的最佳方法展开，建议倾向于使用 Q4 或 Q5 Quantization 级别以获得最佳性能，同时注意 Offloading 到 CPU 的情况。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/PY007/EasyContext-1M-Llama-2-7B">PY007/EasyContext-1M-Llama-2-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/BioMistral/BioMistral-7B">BioMistral/BioMistral-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/legobatman-legojoker-legogoogle-google-joker-gif-13113737">Legobatman Legojoker GIF - Legobatman Legojoker Legogoogle - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/blog/fine-tune-whisper">Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers</a>: 未找到描述</li><li><a href="https://github.com/huggingface/diffusers/tree/main/examples/text_to_image">diffusers/examples/text_to_image at main · huggingface/diffusers</a>: 🤗 Diffusers: 在 PyTorch 和 FLAX 中用于图像和音频生成的先进扩散模型。 - huggingface/diffusers</li><li><a href="https://github.com/huggingface/optimum-graphcore/blob/main/notebooks/whisper_finetuning.ipynb">optimum-graphcore/notebooks/whisper_finetuning.ipynb at main · huggingface/optimum-graphcore</a>: 在 Graphcore IPU 上极速训练 🤗 Transformers - huggingface/optimum-graphcore</li><li><a href="https://huggingface.co/models?pipeline_tag=text-classification&library=transformers.js&sort=trending&search=xenova">Models - Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py">transformers/examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py at main · huggingface/transformers</a>: 🤗 Transformers: 适用于 Pytorch, TensorFlow 和 JAX 的先进机器学习。 - huggingface/transformers</li><li><a href="https://www.youtube.com/watch?v=jtu-G7vA9SU&ab_channel=PaulMeekin">Freedom GPT pitch</a>: 未找到描述</li><li><a href="https://deci.ai/blog/model-merging-moe-frankenmerging-slerp-and-task-vector-algorithms/">Model Merging: Comparing Methods</a>: 探索并比较模型合并方法，如 frankenmerging, SLERP, MoE 和 task vectors，重点介绍它们的优势和挑战。</li><li><a href="https://github.com/jzhang38/EasyContext">GitHub - jzhang38/EasyContext: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware.</a>: 内存优化和训练方案，旨在以最少的硬件将语言模型的上下文长度外推至 100 万个 token。 - jzhang38/EasyContext</li><li><a href="https://rubiks.ai">Rubik's AI - AI research assistant & Search Engine</a>: 未找到描述</li><li><a href="https://www.theverge.com/2023/12/29/24018799/nvidia-4090d-china-slower-us-sanctions">Nvidia is releasing a slower RTX 4090 in China to comply with US restrictions</a>: 美国不允许 Nvidia 在中国销售 RTX 4090。</li><li><a href="https://videocardz.com/newz/nvidia-geforce-rtx-4090-with-blower-type-cooler-is-now-on-sale-in-china">NVIDIA GeForce RTX 4090 with blower-type cooler is now on sale in China - VideoCardz.com</a>: 配备涡轮散热器的 GeForce RTX 4090。不言而喻，拥有 450W TDP 的 RTX 4090 GPU 通常不会配备涡轮散热器。然而，这种显卡确实存在。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1233070949155475538)** (3 条消息): 

- **Mistral 7B 微调文件上传难题**: 一位成员提到他们正尝试微调 **Mistral 7B**，但观察到文件在过程中被上传，而以前并非如此。
- **寻求可与 Transformers 媲美的 Candle 文档**: 一位具有 **Transformers 库**经验的成员对 **Candle** 表现出兴趣，由于 Python 在生产环境中的性能问题，询问是否有类似于 Transformers 的全面文档。
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1232934063028047963)** (8 条消息🔥): 

- **探索 6G 未来**: 一位成员分享了一篇 [arXiv 论文](https://arxiv.org/abs/1904.11686)，讨论了无线通信与 AI 的交集，展望 6G 网络将支持无处不在的 **AI 服务**，以及 AI 将如何协助设计和优化 6G 网络。

- **与 HuggingFace 一起开启计算机视觉之旅**: 一位成员发起了一门**社区驱动的计算机视觉**课程，旨在涵盖该领域从基础到高级的所有主题。该课程可在 [HuggingFace 的学习平台](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome)上访问。

- **人类洞察辅助的强化学习**：一个 [awesome-RLHF GitHub 仓库](https://github.com/opendilab/awesome-RLHF) 正在社区中持续更新和分享，这是一个结合了人类反馈的强化学习资源的精选列表。

- **表达对计算机视觉学习的热情**：一位成员询问了计算机视觉课程的质量，并提到了 [HuggingFace 学习平台](https://huggingface.co/learn)，该平台提供关于在计算机视觉领域应用 ML 库和模型的教育内容。

- **分享 Phi3 红队测试报告以获取洞察**：讨论了 Phi3 红队测试演练的见解和收获，并提供了一个包含更详细信息的 [LinkedIn 帖子](https://www.linkedin.com/posts/divyanshuusingh_phi3-red-teaming-report-activity-7189692710952304640-WsgF) 链接。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/learn">Hugging Face - Learn</a>：未找到描述</li><li><a href="https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome">Welcome to the Community Computer Vision Course - Hugging Face Community Computer Vision Course</a>：未找到描述</li><li><a href="https://arxiv.org/abs/1904.11686">The Roadmap to 6G -- AI Empowered Wireless Networks</a>：近期多样化移动应用的激增，特别是那些由人工智能（AI）支持的应用，正引发关于无线通信未来演进的热烈讨论。Wh...</li><li><a href="https://github.com/opendilab/awesome-RLHF">GitHub - opendilab/awesome-RLHF: A curated list of reinforcement learning with human feedback resources (continually updated)</a>：结合人类反馈的强化学习资源精选列表（持续更新） - opendilab/awesome-RLHF
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1232955568843522101)** (9 条消息🔥): 

- **稍后阅读内容，现在变成每日摘要**：一位用户介绍了一款名为 [Collate.one](https://collate.one/newsletter) 的应用，它可以将稍后阅读的内容转化为简短的每日新闻简报，并邀请他人试用并分享反馈。
- **快速高分辨率图像生成**：在 Hugging Face 上创建了一个用于 [在 5 秒内生成 4k 图像](https://huggingface.co/spaces/KingNish/Instant-Image) 的 Space，复制了另一个名为 PixArt-alpha/PixArt-Sigma 的 Space 的功能。
- **在新 Space 上进行实时故障排除**：针对用户在使用特定提示词尝试图像生成 Space 时报告的错误，创建者请用户再次尝试，并表示问题已得到解决。
- **巴西葡萄牙语的 AI 社区亮点**：社区亮点 #54 已被翻译成巴西葡萄牙语，并发布了 [视频](https://www.youtube.com/watch?v=A9qPlYVeiOs) 和相关的 [博客文章](https://iatalk.ing/destaques-da-comunidade-54/)，旨在向葡萄牙语使用者分享开源 AI 社区的更新。
- **Docker XTTS 推流服务器的改进**：增强了原始的 XTTS 推流服务器，增加了语音温度控制和批处理等功能，并在 [GitHub 仓库](https://github.com/rrg92/docker-xtts) 中展示，同时强调这是学习 Gradio 和语音模型的一个机会。
- **用于长文档的 Mega Small Embed SynthSTS 模型**：一位用户发布了他们的 [Sentence Transformer 模型](https://huggingface.co/BEE-spoke-data/mega-small-embed-synthSTS-16384-v1)，该模型可为长文本文档生成 Embedding，并针对 16,384 的上下文长度进行了预训练。该模型对于聚类和语义搜索任务特别有用，未来可能会有更新。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/KingNish/Instant-Image">Instant Image - KingNish 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/BEE-spoke-data/mega-small-embed-synthSTS-16384-v1">BEE-spoke-data/mega-small-embed-synthSTS-16384-v1 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/rrg92/docker-xtts">GitHub - rrg92/docker-xtts: 用于 XTTS Streaming Server 的 Docker 项目</a>: 用于 XTTS Streaming Server 的 Docker 项目 - rrg92/docker-xtts</li><li><a href="https://www.youtube.com/watch?v=A9qPlYVeiOs">社区亮点 #54</a>: 另一个展示全球开源 AI 社区亮点的视频！帖子地址：https://iatalk.ing/destaques-da-comunidade-54/ 制作这些内容非常有趣...</li><li><a href="https://iatalk.ing/destaques-da-comunidade-54/">🤗社区亮点 #54</a>: 大家好，这是 2024 年 4 月 18 日发布的社区亮点 #54。原始内容可以在以下链接查看：下面是带注释的列表和视频！欢迎订阅……
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1233199775252349020)** (2 条消息): 

- **寻求 Table-QA 视觉模型**: 一位成员询问能够对复杂表格进行问答的 **vision models** 推荐。他们尝试了 **IDEFICS2** 和 **GEMINI 1.5 pro**，但遇到了数值不准确的问题。
- **对 COCO 数据集的安全性担忧**: 一位成员对官方 COCO 数据集托管在 **HTTP** 连接上表示担忧，暗示了潜在的安全影响。
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1232953833441988648)** (6 条消息): 

- **呼吁更多以代码为中心的资源**: 一位成员称赞了现有的资源 **Dr. Valeriy**，认为它是创建直接链接到代码实现的工具的良好典范，并分享了 [valerman/awesome-conformal-prediction](https://github.com/valeman/awesome-conformal-prediction) 的链接作为参考。
- **寻求 SFFTrainer 训练细节**: 一位用户寻求对 **SFFTrainer** 训练过程的详细了解，特别是 Prompt 的哪些组件最初被输入到 LLM，以及 LLM 是否受限于所提供答案的 Token 数量。
- **寻找开源 STT Web 前端**: 一位社区成员询问是否有可用的开源 **speech-to-text (STT)** Web 前端。
- **寻求 safetensors 的版权信息**: 一位成员质疑 **safetensors** 缺失版权详情，指出虽然许可证是 Apache，但 [LICENSE 文件](https://github.com/huggingface/safetensors/blob/main/LICENSE) 中没有年份或所有权详情。
- **庆祝 Trustworthy Language Model 发布**: 宣布发布 **Trustworthy Language Model (TLM)** v1.0，其特点是通过置信度评分系统来对抗 LLM 幻觉。邀请用户通过 [Playground](https://tlm.cleanlab.ai/) 试用 TLM 并分享发现，更多见解可在 [博客文章](https://cleanlab.ai/blog/trustworthy-language-model/) 和详细 [教程](https://help.cleanlab.ai/tutorials/tlm/) 中找到。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tlm.cleanlab.ai/">TLM Playground</a>: 在浏览器中试用 Cleanlab 的 Trustworthy Label Model (TLM)。</li><li><a href="https://cleanlab.ai/blog/trustworthy-language-model/">通过 Trustworthy Language Model 克服幻觉</a>: 宣布 Cleanlab 的 Trustworthy Language Model。TLM 通过为每个 LLM 输出添加信任分数，克服了 GenAI 生产化的最大障碍——幻觉。</li><li><a href="https://help.cleanlab.ai/tutorials/tlm/">Trustworthy Language Model (TLM)</a>: 一种更可靠的 LLM，可量化每个输出的可信度并检测不良响应。</li><li><a href="https://github.com/huggingface/safetensors/blob/main/LICENSE">huggingface/safetensors 的 LICENSE</a>: 存储和分发 Tensor 的简单、安全方式。通过在 GitHub 上创建账号为 huggingface/safetensors 的开发做出贡献。</li><li><a href="https://github.com/valeman/awesome-conformal-prediction">GitHub - valeman/awesome-conformal-prediction</a>: 专业策划的 Awesome Conformal Prediction 视频、教程、书籍、论文、博士和硕士论文、文章及开源库列表。
</li>
</ul>

</div>
  

---

**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1233041403341439017)** (6 条消息): 

- **赞赏 LCM 和 ip-adapter 的协同效应**：一位成员指出 **ip-adapter** 和 **lcm-lore** 配合得很好，表明它们结合使用非常有效。尽管目前对 LCM 有一些抱怨，但该成员希望 **hyper-sd** 能带来改进。

- **蓝色调图像之谜**：一位用户在使用带有多个 **controlnet** 的 **text-to-image pipeline** 时遇到了图像变蓝的问题。经过简短讨论后，原因尚不明确。

- **torch.compile 的试错**：尝试在训练期间使用 **torch.compile**，最初导致程序在第一次前向传播（forward pass）时挂起。最终进程顺利完成，耗时约 10 分钟。

- **使用 torch.compile 提升前向传播速度**：在克服初始障碍后，成员注意到前向传播的速度有了显著提升，但使用 **torch.compile** 对反向传播（backward pass）的速度没有影响。
  

---


**HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1233201020428091463)** (1 条消息): 

```html
<ul>
  <li><strong>Gradio 增强自定义组件功能</strong>：Gradio 4.28.0 版本为自定义组件引入了重大增强，包括 Tailwind 样式定制、对任何 vite 插件和预处理器的支持，以及在 spaces 中利用原生 Gradio SDK 的精简版自定义组件 CLI。</li>
  <li><strong>精简开发与新特性</strong>：伴随自定义组件升级的还有其他功能，例如设置最大上传限制、开发模式下的持久重载以保持前端状态，以及重新组织的文档以更好地展示 Gradio 生态系统。</li>
  <li><strong>包含更多改进的全面发布</strong>：这只是更新的亮点；更多详情可以在 Gradio 网站上的完整变更日志中找到。</li>
</ul>
```
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1232958925477773313)** (52 条消息🔥): 

- **澄清模型架构**：在讨论模型架构时，一位用户澄清了一些论文如何将 **attention** 和前馈神经网络 (**FFN**) 描述为并行操作，并以 **PaLM** 为例，说明其 **模型使用并行 attention + FFN**。
- **解码 Pile 数据集哈希**：一位成员分享了 **Pile 数据集的哈希值**，为寻求在各种 JSON 文件中使用该数据集的用户提供了一个[链接到 EleutherAI 的哈希列表](https://www.eleuther.ai/hashes)。
- **Transformer 中的感受野机制**：关于**滑动窗口 attention**（sliding window attention）的对话提到，掩码（mask）如何限制 **attention** 的范围，并将其与卷积在有效感受野方面的运作方式进行了比较。
- **探索分层学习与 Attention 结构**：参与者讨论了交错使用 **RNN** 类型层或为 **Transformer** 使用扩张窗口（dilated windows）以有效处理**更长序列长度**的潜力。
- **用于大模型训练的新 PyTorch 库**：一位用户分享了一个名为 **torchtitan** 的新 **PyTorch** 库的 [GitHub 仓库链接](https://github.com/pytorch/torchtitan)，该库旨在用于大模型训练。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.eleuther.ai/hashes">Hashes &mdash; EleutherAI</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtitan">GitHub - pytorch/torchtitan: A native PyTorch Library for large model training</a>：一个用于大模型训练的原生 PyTorch 库。欢迎通过在 GitHub 上创建账号为 torchtitan 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1232934258192941056)** (144 条消息🔥🔥): 

- **线性 Attention 分解详解**：一位成员详细解释了线性 **attention** 的工作原理，即在推理时将 \( QK^T V \) 分解为 \( Q(K^T V) \)。文中澄清这种方法相对于序列长度是线性的，并保持恒定的内存占用，而不像 **softmax attention** 那样随时间增长。

- **超越假设格式的基准测试**：一位成员报告了 **phi-3-mini-128k** 与其他模型的“基准测试”结果，表明其性能可能与 **Llama-3-8B** 持平。讨论围绕预训练数据是否显著影响训练后性能，以及在像 **phi** 这样的 AI 模型语境下什么构成了“基座”（base）展开。

- **深入探讨 Delta Rule 的实用性**：关于 delta rule 线性注意力的实用性和并行化展开了讨论，一位成员分享了来自 [manifestai 博客文章](https://manifestai.com/blogposts/faster-after-all/) 的见解。会议指出，delta rule 线性注意力更有条理，但并行化程度较低，可能会减慢训练速度。

- **“大海捞针”测试受到审视**：用户质疑了针对长上下文语言模型的“大海捞针” (Needle in the Haystack) 测试的有效性，认为实际应用和个人测试是更具指示性的性能基准。人们怀疑这种测试如何衡量“针”与其周围上下文之间的语义相似性。

- **SFT 期间屏蔽用户提示词损失**：有人好奇在语言模型的有监督微调 (SFT) 期间屏蔽用户提示词 (User Prompt) 损失是否经过了系统研究。虽然这是一种常见做法，但成员们注意到缺乏关于其效果的研究，并讨论了在 SFT 中包含提示词损失的潜在收益。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.15574">Retrieval Head Mechanistically Explains Long-Context Factuality</a>：尽管长上下文语言模型最近取得了进展，但基于 Transformer 的模型如何展现出从任意位置检索相关信息的能力仍然难以捉摸...</li><li><a href="http://arxiv.org/abs/2404.15574">Retrieval Head Mechanistically Explains Long-Context Factuality</a>：尽管长上下文语言模型最近取得了进展，但基于 Transformer 的模型如何展现出从任意位置检索相关信息的能力仍然难以捉摸...</li><li><a href="https://openreview.net/forum?id=Hygxb2CqKm">Stable Recurrent Models</a>：稳定循环模型可以通过前馈网络进行近似，并且在基准任务上的经验表现与不稳定模型一样好。</li><li><a href="http://arxiv.org/abs/2404.03683">Stream of Search (SoS): Learning to Search in Language</a>：语言模型在训练期间很少看到富有成效的错误。因此，它们很难看透下一个 token，遭受错误滚雪球的影响，并难以预测...的后果。</li><li><a href="https://manifestai.com/blogposts/faster-after-all/">Manifest AI - Linear Transformers Are Faster After All</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 条消息): 

main.ai: https://twitter.com/sen_r/status/1783497788120248431
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1232947816624685098)** (21 条消息🔥): 

- **GSM8K 的 Few-Shot 性能查询**：关于 **GSM8K** 的 few-shot 示例数量 (**num_fewshot**) 的问题出现，以匹配 **Hugging Face 排行榜**，建议该数量应为 *5*。

- **讨论 VLLM 集成阻碍**：一位用户询问了将 **VLLM** 升级到最新版本的障碍。讨论明确了 **Data Parallel (DP)** 是一个潜在的阻碍，但 **Tensor Parallel (TP)** 应该没问题。

- **邀请提交 Filter Registry 函数的 PR**：一位新成员注意到 **lm_eval** 中缺少 `FILTER_REGISTRY` 的 `register_filter` 函数。该用户被鼓励提交一个 PR 来解决这个问题。

- **关于 Brier Score 函数的思考**：一位成员在 **lm-evaluation-harness** 中进行 **ARC 评估** 时遇到了 Brier score 函数的问题，原因是数据异常。建议调整 Brier score 函数，以避免尽管数据集不一致仍产生错误。

- **聊天模板分支进度查询**：一位用户查询了 *Hailey 的分支* 上关于聊天模板 (chat templating) 的活跃分支状态，该分支最后一次更新是在两个月前，表达了对功能进展的兴趣。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/3196e907fa195b684470a913c7235ed7f08a4383/lm_eval/api/metrics.py#L124.">lm-evaluation-harness/lm_eval/api/metrics.py at 3196e907fa195b684470a913c7235ed7f08a4383 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1745">add task for mmlu evaluation in arc multiple choice format by jonabur · Pull Request #1745 · EleutherAI/lm-evaluation-harness</a>：此 PR 添加了 mmlu_arc_style 任务，该任务以与 arc 评估相同的方式呈现 MMLU 问题（将答案的对数似然作为延续，而不是选择字母...
</li>
</ul>

</div>

**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1233094280797753376)** (2 条消息): 

- **识别供应商问题**：发现一家主要的 **Mixtral 8x7b 供应商** 发送空白响应，目前已暂时移除。正在考虑未来通过 *auto-detect*（自动检测）此类问题的解决方案。

- **Soliloquy 8B 切换为付费模型**：[Soliloquy 8B](https://openrouter.ai/models/lynn/soliloquy-l3) 已转为付费模式，根据最新更新，费用为 **每 1M tokens 0.1 美元**。更多详情可通过提供的 Discord 频道链接查看。

**提到的链接**：<a href="https://openrouter.ai/models/lynn/soliloquy-l3)">Lynn: Llama 3 Soliloquy 8B by lynn | OpenRouter</a>：Soliloquy-L3 是一款快速、高性能的角色扮演模型，专为沉浸式、动态体验而设计。该模型基于超过 2.5 亿 tokens 的角色扮演数据训练，拥有广博的知识库...

  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1233393770071064597)** (1 条消息): 

- **在 LinkedIn 上宣布 AI 突破**：消息中包含一条来自 *fprime-ai* 的 LinkedIn 帖子，讨论了其 **DBRX AI** 系统的技术突破。可以通过 [此处](https://www.linkedin.com/posts/fprime-ai_fprimeailabs-dbrx-ai-activity-7189599191201980417-Te5d?utm_source=share&utm_medium=member_desktop) 访问并阅读帖子详情。
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1232976027555336284)** (215 条消息🔥🔥): 

- **为角色扮演创意选择最佳模型**：成员们讨论了最适合角色扮演中创意情节构建的开源模型。推荐包括 **WizardLM2 8x22B**、用于创意写作的 command-based 模型以及 **Mixtral 8x22B**，其中一位用户强调了 Mixtral 出色的创造力。

- **关于 GPT Turbos 和 Microsoft Wizard LM 的争论**：围绕 **Microsoft 对 Wizard LM 项目的影响**爆发了广泛争论，一些人认为该公司停止了旧模型的开发，而另一些人则在争论 GPT-4 "Turbo" 模型的性能。一名成员通过链接一份详细的 [事件总结](https://rocky-muscle-755.notion.site/) 提供了证据。

- **探讨模型性能与托管成本**：成员们评估了 **GPT-4、Llama 3 和 WizardLM** 等各种模型的性能，同时讨论了托管成本和当前定价的可持续性，并估算了每百万 tokens 的成本。

- **对模型切换和 API 日志记录的担忧**：用户对 OpenRouter 中模型切换的透明度以及供应商对 API 调用进行日志记录表示担忧，部分用户对使用 Lynn: Llama 3 Soliloquy 8B 等模型持保留意见。

- **OpenRouter 的使用、功能与限制**：讨论涵盖了从启用 **system message mappings** 到 **playground 响应扩展**等主题。用户还询问了如何处理 **HTTP 524 错误** 以及在使用付费 LLM 时如何避免 **负余额**。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://imgur.com/7skpsI0">imgur.com</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐平台。通过幽默笑话、热门迷因、有趣的 GIF、励志故事、病毒视频等来放松心情...</li><li><a href="https://rocky-muscle-755.notion.site/What-happened-to-WLM2-a247e09244d0483cbb02c1587b357c9d?pvs=4">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://openrouter.ai/models?q=free">OpenRouter</a>：在 OpenRouter 上浏览模型
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1233088063849369692)** (4 条消息): 

- **create-llama v0.1 发布**：**create-llama** v0.1 版本引入了 RAG 应用的便捷设置，包含 **@ollama 支持** 和新的 **vector database** 集成等新功能，方便使用 llama3 和 phi3 模型。该更新通过 [一条推文宣布，包含更多改进细节](https://twitter.com/llama_index/status/1783528887726817653)。
  
- **使用 Qdrant 和 LlamaParse 构建 RAG 应用**：另一条推文详细介绍了如何使用 **LlamaParse**、**@JinaAI_ embeddings**、**@qdrant_engine 向量存储** 和 **Mixtral 8x7b** 构建 RAG 应用的分步教程。感兴趣的开发者可以在 [此处](https://twitter.com/llama_index/status/1783601807903863184) 获取完整教程。

- **KX 举办的 LlamaParse 网络研讨会**：KX systems 正在组织一场关于最大化 **LlamaParse** 效用的**网络研讨会**，涵盖复杂文档解析、表格和图像提取以及自然语言预处理。活动详情请见[此 Twitter 帖子](https://twitter.com/llama_index/status/1783622871614664990)。

- **以 LlamaIndex 为特色的 AWS 工作坊**：**@llama_index** 与 AWS 合作提供关于使用 AWS 构建 LLM 应用的工作坊材料，解释了如何结合 LlamaParse 和 LlamaCloud 使用 S3、AWS Bedrock LLMs 和 embedding storage 等服务。工作坊详情总结在[此推文](https://twitter.com/llama_index/status/1783877951278432733)中。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1232982378058874891)** (117 条消息🔥🔥): 

- **探索 RAG 实现**：成员们讨论了简单和高级 RAG (Retrieval-Augmented Generation) 流水线的有效性。建议探索更复杂的 RAG 解决方案，如 sentence-window retrieval 或 auto-merging retrieval 以获得更好的效果，并链接了一个视频来学习如何设置这些流水线（[高级 RAG 课程](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/5/auto-merging-retrieval)）。

- **聊天机器人实现的故障排除**：有一场关于使用 gpt-4-vision-preview 实现聊天机器人的对话，其中出现了后端不支持图像上传的问题。一位成员发现，将图像作为内容的一部分添加，而不是使用 `additional_args`，可以解决该问题。

- **在 LlamaIndex 中配置和使用 Pydantic**：一位用户询问如何从聊天补全中获取结构化的 Pydantic 输出，另一位用户提出了 Pydantic 导入导致类型检查错误的问题。建议是直接使用 v1 导入，或者等待 LlamaIndex 逐步停止对 v1 的支持。

- **查询流水线配置咨询**：几位用户讨论了配置查询流水线的细节，提到了流水线中 GPT-4 的 JSON 输出问题，并探索了如何有效地格式化中间步骤的输出。据指出，GPT-4 turbo 不支持 JSON 输出，而 GPT-3.5 turbo 允许使用 JSON 模式（[GPT JSON 模式文档](https://platform.openai.com/docs/guides/text-generation/json-mode)）。

- **使用 LlamaIndex 设置本地 LLM 的指南**：一位成员寻求使用 LlamaIndex 配合本地语言模型以避免使用外部 API 的指导。他们被引导至官方文档中的入门示例（[本地 LLM 入门示例](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/)）。对话内容包括排除导入错误以及整合必要的软件包安装。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/5/auto-merging-retrieval">DLAI - 构建与评估高级 RAG</a>: 简介 · 高级 RAG 流水线 · RAG 指标三元组 · 句子窗口检索 · 自动合并检索 · 结论</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">入门教程 (本地模型) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai">LlamaIndex - LlamaIndex</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Fighter_(2024_film)">Fighter (2024 电影) - 维基百科</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/openai_json_vs_function_calling/?h=json">用于数据提取的 OpenAI JSON 模式 vs. 函数调用 - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/bridge/pydantic.py">llama_index/llama-index-core/llama_index/core/bridge/pydantic.py at main · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/explodinggradients/ragas/issues/557)">Issues · explodinggradients/ragas</a>: 用于检索增强生成 (RAG) 流水线的评估框架 - Issues · explodinggradients/ragas</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/output_parsers/pydantic#llama_index.core.output_parsers.PydanticOutputParser>).">Pydantic - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/query_engine/citation/?h=citationqueryengine#llama_index.core.query_engine.CitationQueryEngine))">引用 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/react_agent#react-agent-a-simple-intro-with-calculator-tools>)">ReAct Agent - 计算器工具简单介绍 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/community/integrations/guidance#creating-a-guidance-program-to-generate-pydantic-objects>)">Guidance - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/multi_modal/multi_modal_pydantic#using-fuyu-8b-for-pydantic-strucured-output>)">多模态 GPT4V Pydantic 程序 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/multi_modal/multi_modal_pydantic#using-minigpt-4-for-pydantic-strucured-output>)">多模态 GPT4V Pydantic 程序 - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1233016896589140008)** (78 条消息🔥🔥): 

- **LAION Discord 反思 Llama 3 性能**：用户对 Llama 3 的性能看法不一，一些人报告了在正确召回代码方面的问题，而另一些人则认为这可能是由于配置问题。虽然有些人认为它可与 GPT-4 媲美，但其他人认为仍有很大的改进空间。

- **关于拟议的“了解您的客户 (KYC)”要求的辩论**：TorrentFreak 的一个链接讨论了美国针对云服务的“了解您的客户”要求的提案及其对用户的影响。一位成员分享了《联邦公报》的通知，敦促在征求意见期结束前提供反馈。

- **AI 爱好者寻找志同道合的社区**：LAION Discord 的成员表示有兴趣加入更多以 AI/ML 为导向的 Discord 服务器，以进行更广泛的社区参与和资源共享。

- **微调 AI 模型带来性能惊喜**：一位致力于微调 DF 400M/450M 模型的成员发现了尚未开发的显著性能，强调了低学习率和改进的真实照片放大效果。

- **对 Nightshade 功效和发布协议的批评**：用户讨论了 Nightshade 有效性的透明度和数据的必要性，关于自动编码器限制的理论讨论，以及由于社区可能产生的负面反应而不愿发布研究结果。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://torrentfreak.com/u-s-know-your-customer-proposal-will-put-an-end-to-anonymous-cloud-users-240425/">U.S. &quot;Know Your Customer&quot; Proposal Will Put an End to Anonymous Cloud Users * TorrentFreak</a>: 未找到描述</li><li><a href="https://www.tomshardware.com/tech-industry/us-investigates-chinas-access-to-risc-v-open-source-instruction-set-may-become-new-site-of-us-china-chip-war">US investigates China's access to RISC-V &mdash; open standard instruction set may become new site of US-China chip war</a>: RISC-V 对美国立法者来说似乎存在风险</li><li><a href="https://arxiv.org/abs/2401.01808">aMUSEd: An Open MUSE Reproduction</a>: 我们介绍了 aMUSEd，这是一个基于 MUSE 的开源、轻量级掩码图像模型 (MIM)，用于文本生成图像。aMUSEd 的参数量仅为 MUSE 的 10%，专注于快速图像生成...</li><li><a href="https://www.federalregister.gov/documents/2024/01/29/2024-01580/taking-additional-steps-to-address-the-national-emergency-with-respect-to-significant-malicious">Federal Register :: Request Access</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/fal-ai/imgsys-results">fal-ai/imgsys-results · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://mediabiasfactcheck.com/torrentfreak-bias/,">TorrentFreak - Bias and Credibility</a>: 偏见极低。这些来源具有极小的偏见，很少使用带有倾向性的词汇（即试图通过诉诸情感或...来影响受众的措辞）</li><li><a href="https://en.wikipedia.org/wiki/TorrentFreak">TorrentFreak - Wikipedia</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1233160752060305501)** (12 messages🔥): 

- **视觉表示学习的革命**：强调了一种针对视觉模型的新型弱监督预训练方法，该方法将图像-文本数据的预训练归类为分类任务。这种方法在不损失表示质量的前提下，实现了比传统对比学习快 **2.7 倍** 的训练速度，详见此 [arXiv 论文](https://arxiv.org/abs/2404.15653)。

- **简单而有效**：承接上一点，该方法的成功归功于从 alt-text 中检测概念并训练多标签分类器。据指出，这使得模型在 zero-shot 场景下的表现与 **CLIP** 相当，且训练效率大幅提升。

- **对比的代价**：在关于文本编码器和对比学习功效的对话中，有人指出对比学习（特别是在对齐文本编码器时）成本高昂。在处理嘈杂的 alt-text 时，该方法可能会产生额外的计算开销。

- **虽快但仍漫长**：有人分享了一条幽默的评论，承认虽然训练速度提升 2.7 倍意义重大，但整个过程仍然耗时。这反映了对速度改进的现实看法。

- **探索 VAST 的可能性**：有人表达了对微调 **VAST**（一个 Vision-Audio-Subtitle-Text 全模态基础模型和数据集）的兴趣，并提供了该项目 [GitHub 仓库](https://github.com/txh-mercury/vast) 的链接。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.16710">Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding</a>: 我们提出了 LayerSkip，这是一种加速大语言模型 (LLM) 推理的端到端解决方案。首先，在训练期间我们应用层丢弃 (layer dropout)，对较早的层使用较低的丢弃率，而对较晚的层使用较高的...</li><li><a href="https://arxiv.org/abs/2404.15653">CatLIP: CLIP-level Visual Recognition Accuracy with 2.7x Faster Pre-training on Web-scale Image-Text Data</a>: 对比学习已成为一种通过对齐图像和文本嵌入来学习有效视觉表示的变革性方法。然而，成对相似度计算...</li><li><a href="https://github.com/txh-mercury/vast">GitHub - TXH-mercury/VAST: Code and Model for VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset</a>: VAST 的代码和模型：一个 Vision-Audio-Subtitle-Text 全模态基础模型和数据集 - TXH-mercury/VAST
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1232935424461373470)** (70 messages🔥🔥):

- **Open Interpreter 的 Mac 集成**：Open Interpreter 的 **New Computer Update Part II** 通过首个本地视觉模型、*原生 Mac 集成*、提升的启动速度以及额外功能增强了本地功能。用户可以运行简单的命令如 `interpreter --os` 直接从 Open Interpreter 控制 Mac 的原生应用程序，详情见其 [变更日志](https://changes.openinterpreter.com/log/ncu-ii)。

- **视觉模型展示与更新**：社区成员讨论了 **Moondream** 小型视觉语言模型，展示了 [Img2TxtMoondream.py](https://github.com/CodeAKrome/bootcupboard/blob/main/llm-img/Img2TxtMoondream.py)，这是一个视觉模型的代码演示。对话转向了使用多模态模型如 **LLaVA**（可在 [Hugging Face](https://huggingface.co/liuhaotian/llava-v1.6-34b) 获取），并强调了其基于 **NousResearch/Nous-Hermes-2-Yi-34B** 的基础。

- **解决循环问题与模型性能**：参与者交流了优化本地模型以防止循环行为（looping behavior）的技巧，建议进行调整，如修改 *温度设置 (temperature settings)*、*提示词编辑* 或 *架构更改*。还引入了 *挫败感指标 (frustration metric)* 的概念，以便在遇到连续循环后调整模型的行为。

- **集成探索与故障排除**：一位用户思考了将 **Open Interpreter** 集成到机器人控制中，特别是 **Unitree GO2 机器人狗**，并询问社区经验。其他人讨论了运行本地服务器的技术问题和解决方案，例如设置虚拟 API 密钥以及解决 Pydantic 模型配置中的命名空间冲突。

- **Open Interpreter 'New Computer Update' 非 Beta 版本发布**：一位用户确认正在运行 **Open Interpreter 0.2.5 New Computer Update**，表明包含近期增强功能的版本已脱离 Beta 阶段。然而，关于该更新从 Beta 状态变更的情况存在疑问，随后通过版本检查进行了澄清。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/liuhaotian/llava-v1.6-34b">liuhaotian/llava-v1.6-34b · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2112.10003">使用文本和图像提示进行图像分割</a>：图像分割通常通过为固定的一组对象类别训练模型来解决。稍后合并额外的类别或更复杂的查询是昂贵的，因为它需要重新训练...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/core/computer/display/point/point.py">open-interpreter/interpreter/core/computer/display/point/point.py at main · OpenInterpreter/open-interpreter</a>：计算机的自然语言界面。通过在 GitHub 上创建一个账户来为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#interactive-chat">GitHub - OpenInterpreter/open-interpreter: 计算机的自然语言界面</a>：计算机的自然语言界面。通过在 GitHub 上创建一个账户来为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/CodeAKrome/bootcupboard/blob/main/llm-img/Img2TxtMoondream.py">bootcupboard/llm-img/Img2TxtMoondream.py at main · CodeAKrome/bootcupboard</a>：它的内部比外部更大！通过在 GitHub 上创建一个账户来为 CodeAKrome/bootcupboard 的开发做出贡献。</li><li><a href="https://changes.openinterpreter.com/log/ncu-ii">Open Interpreter - The New Computer Update II</a>：开源 Open Interpreter 项目的官方变更日志。</li><li><a href="https://github.com/vikhyat/moondream">GitHub - vikhyat/moondream: 小型视觉语言模型</a>：小型视觉语言模型。通过在 GitHub 上创建一个账户来为 vikhyat/moondream 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1233253423114354719)** (2 条消息): 

- **硬件到货激发热情**：一位成员对收到用于组装的硬件表示兴奋，尽管缺少一根 *黄色电线和开关*，但他们有备件。另一位成员对组装过程表现出兴趣并期待更新。
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 条消息): 

8i8__papillon__8i8d1tyr: https://www.youtube.com/watch?v=WeH3h-o1BgQ
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1232937070536818738)** (56 条消息🔥🔥):

- **顶部的推文**：一位成员分享了 **Hugging Face 的 CEO 评论其推文**的兴奋之情。
- **热烈欢迎**：新成员 "*lazeewhalee*" 加入了小组，并被引导**阅读 readme** 以了解导航和指南。
- **70B 模型部署讨论**：一位成员提到使用 **exllama** 部署和运行 70B 模型，并询问了由于缺少 checkpoint 可能导致的问题。
- **对 AI 基准测试的推测**：成员们对 **MMLU 分数**的有效性以及各种模型的性能表示担忧，特别是某个 8B 模型，除了在 MMLU 上表现较好外，其性能均逊于基础的 llama3。
- **关于特定领域训练的见解**：成员们讨论了为特定领域微调 Large Language Models (LLMs) 的益处，并分享了 **[Meditron](https://arxiv.org/abs/2311.16079)** 这篇关于该主题的详尽论文。一位成员还简要提到了关于领域自适应持续预训练的**即将发表的论文**。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2311.16079">MEDITRON-70B: Scaling Medical Pretraining for Large Language Models</a>：大型语言模型 (LLMs) 有可能使医疗知识的获取变得大众化。虽然在利用和提高 LLMs 的医学知识和推理能力方面已经做出了许多努力，但...</li><li><a href="https://arxiv.org/abs/2311.08545">Efficient Continual Pre-training for Building Domain Specific Large Language Models</a>：大型语言模型 (LLMs) 展示了卓越的开放领域能力。传统上，为特定领域定制的 LLMs 是从头开始训练的，以便在处理特定领域任务时表现出色。在本文中...</li><li><a href="https://huggingface.co/collections/microsoft/wizardlm-661d403f71e6c8257dbd598a?history=true>">WizardLM - 微软集合</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1233048482760101929)** (3 条消息): 

- **报告了底层问题**：一位成员确认遇到了一个问题，尽管目前尚不清楚是什么变化导致了该问题。
- **表达失望**：简短的回复 "sadge" 表达了对之前可能讨论过的一个话题的失望或难过。
- **关于 zzero3 和 fft 兼容性的询问**：一位成员询问是否有人成功地将 **zzero3** 与 **Fast Fourier Transform (fft)** 集成。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1232950442468577321)** (3 条消息): 

- **交换祝福**：一位用户对另一位用户的努力表示了成功的祝愿。
- **Phi3 微调挑战**：一位成员讨论了他们在微调 **phi3** 模型时遇到的困难，提到它需要大量 RAM 且运行缓慢。
- **技术故障排除**：针对一个技术问题，有人提出了关于在使用 **transformers 4.40.0** 时，`AttributeError` 与 'TextIteratorStreamer' 对象没有 'empty' 属性相关的问题。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1233170259020877947)** (10 条消息🔥): 

- **优化器兼容性查询**：对 **FSDP (Fully Sharded Data Parallel)** 优化器兼容性的搜索显示，**AdamW** 和 **SGD** 等优化器普遍支持，但某些优化器（如 `paged_adamw_8bit`）不支持 FSDP offloading。
- **Offloading 不兼容问题**：`paged_adamw_8bit` 优化器存在问题，因为它与 **FSDP offloading** 不兼容，这表明特定优化器与 FSDP 功能之间存在集成挑战。
- **寻找解决方案**：针对一个错误，目前正在 **OpenAccess-AI-Collective/axolotl** 中努力搜索支持 **FSDP** 的替代优化器。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=a50dde20-84d2-463b-8e6d-cc3f55531430)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=f65c9e42-0ffc-4336-9b7b-5722eb092272)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1232954183498334259)** (63 条消息🔥🔥):

- **Toolkit 难题**：一位用户在向 Azure 上的 **Cohere Toolkit** 上传文档时遇到困难，得到的指导是点击回形针附件图标进行上传。然而，他们仍然无法找到上传选项，也无法与其 Cohere-Toolkit 实例进行交互。
- **字体争议**：针对用户关于 GitHub 上的 **Cohere 字体** 是否遵循 MIT 许可证的询问，官方澄清该字体并非开源，且将被替换。
- **模型访问与许可说明**：Cohere 的 [Command+ 模型是开放权重的，但不适用于商业用途](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus)，权重可供非商业使用，但 **训练数据仍未公开**。
- **AI 搜索引擎探索**：据透露，**Tavily** 已被用于 Cohere-Toolkit；不过，有人建议将 **Brave Search API** 作为一种可能更快、更便宜且准确的替代方案。随后引发了关于搜索引擎在不同语境下的成本效益和使用的讨论。
- **Cohere Toolkit 部署困境**：用户分享了在 Azure 上部署 Cohere Toolkit 的见解；用户无需添加 Cohere API 密钥，但必须选择模型部署选项以确保应用程序正常运行。随后，有用户提出了在本地添加工具时的困难，遇到了 PDF 上传问题以及不支持的 sqlite3 版本。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tavily.com">Tavily</a>：未找到描述</li><li><a href="https://docs.trychroma.com/troubleshooting#sqlite">🔍 Troubleshooting | Chroma</a>：此页面列出了常见的陷阱或问题及其解决方法。</li><li><a href="https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus">C4AI Command R Plus - a Hugging Face Space by CohereForAI</a>：未找到描述</li><li><a href="https://github.com/cohere-ai/cohere-toolkit/blob/main/src/backend/tools/retrieval/tavily.py">cohere-toolkit/src/backend/tools/retrieval/tavily.py at main · cohere-ai/cohere-toolkit</a>：Toolkit 是预构建组件的集合，使用户能够快速构建和部署 RAG 应用程序。 - cohere-ai/cohere-toolkit</li><li><a href="https://github.com/searxng/searxng">GitHub - searxng/searxng: SearXNG is a free internet metasearch engine which aggregates results from various search services and databases. Users are neither tracked nor profiled.</a>：SearXNG 是一个免费的互联网元搜索引擎，它聚合了来自各种搜索服务和数据库的结果。用户既不会被追踪，也不会被画像。 - searxng/searxng
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1232944667276869663)** (6 条消息): 

- **关于针对 Cohere 的“抹黑文章”的辩论**：成员们展开了激烈的辩论，一方为 *Cohere* 辩护，反对其鲁莽的指控，另一方则质疑在 AI Agent 中可能创建“越狱”场景的责任，这些 Agent 会将 Token 转化为现实世界的行动。
- **对文章内容和评论的困惑**：一位成员表示，他们不再记得自己批评为抹黑文章的具体细节，并指出讨论中缺乏对 Chatbot 和 Agent 行为的区分。
- **要求证实批评的挑战**：在被要求证实文章不公平地抹黑 *Cohere* 的说法时，一位成员承认他们无法立刻想起具体原因，这降低了该批评的可信度。
- **关于记忆细节的沟通误解**：一位成员嘲笑了“无法记住细节”这一理由，认为这不能作为无法列举该恶意文章问题的借口。
- **研究对话中的问责期望**：对话以一项声明结束：如果有人批评研究工作是恶意的，他们应该准备好证实自己的说法，这意味着在参与研究讨论时需要负起责任。
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1232968803005239296)** (62 条消息 🔥🔥):

- **合作伙伴关系是 tinygrad 成功的关键**：Tinygrad 旨在通过合作伙伴关系以及让其他人投资其框架来取胜，comma 是 tinybox 硬件的首个合作伙伴，并可能在 tinybox 2 上与其他方合作。
- **TinyBox 作为 tinygrad 的测试场**：由 comma 生产的 tinybox 被认为是 tinygrad 经过最充分测试的环境。George Hotz 强调 tinygrad 的重点仍然是软件，而非硬件生产。
- **关于与 Tenstorrent 合作的考量**：尽管进行了初步讨论，但与 Tenstorrent 的合作在财务上并不合理，因为他们的硬件在效率或普及程度上没有竞争优势。然而，如果财务计算发生变化，不排除未来合作的可能性。
- **AMD 的 MES 对 tinygrad 的效用有限**：George Hotz 指出，尽管 AMD 的 Felix 提供了一份很有帮助的报告，但 AMD 的 Machine Environment Settings (MES) 对 tinygrad 可能用处不大。团队继续致力于开发满足其需求的 PM4 后端。
- **tinygrad MNIST 教程与 GPU 兼容性**：分享了一个 [tinygrad MNIST 教程](https://tinygrad.github.io/tinygrad/mnist/)，适用于在支持 GPU 的 Google Colab 中运行。用户报告了较新 NVIDIA 硬件的问题，这些问题通过确保安装最新的 CUDA 库得到了解决。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discordapp.com/channels/1068976834382925865/1227683281269559418/1232845778259673239">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持联系。</li><li><a href="https://tinygrad.github.io/tinygrad/mnist/">MNIST 教程 - tinygrad 文档</a>：未找到描述</li><li><a href="https://x.com/karpathy/status/1783527854741114981?s=46">Andrej Karpathy (@karpathy) 的推文</a>：[gif] 我之前尝试阅读 tinygrad 代码的样子 :D 我认为 LOC 要求（这只是简洁性的一个指标）导致了过度的压缩。你不会吹嘘你的 .min.js 代码是...</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/docs/developer.md">tinygrad/tinygrad master 分支下的 docs/developer.md</a>：你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://gist.github.com/fxkamd/ffd02d66a2863e444ec208ea4f3adc48">关于 TinyGrad 中 HSA 和 KFD 后端的观察</a>：关于 TinyGrad 中 HSA 和 KFD 后端的观察 - TinyGrad-notes.md
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1233023252415381604)** (6 条消息): 

- **展望 tinygrad 1.0**：分享了 [tinygrad 的文档](https://tinygrad.github.io/tinygrad/)，强调 API 在接近 1.0 版本发布时趋于稳定。内容包括源码安装指南、[MNIST 教程](https://tinygrad.github.io/tinygrad/mnist/)、[开发者文档](https://tinygrad.github.io/tinygrad/developer/)以及[外部创建的教程](https://mesozoic-egg.github.io/tinygrad-notes/)。

- **tinygrad 处理 Quantile 函数**：一位成员讨论了他们的项目，旨在 tinygrad 中重新实现 `torch.quantile` 函数，作为开发 Diffusion 模型采样算法的一部分。该过程包括数组排序的中间步骤。

- **tinygrad 文档将获得更多曝光**：预见到 tinygrad 0.9 的发布，一位成员询问是否会在项目 README 中包含 [tinygrad 文档](https://tinygrad.github.io/tinygrad/)的链接。回复表示在 0.9 发布时将采取肯定行动。

**提到的链接**：<a href="https://tinygrad.github.io/tinygrad/">tinygrad 文档</a>：未找到描述

  

---



**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1783575774085410911>
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1233116804256235520)** (1 条消息): 

- **Modular 应对软件安全挑战**：Modular 不断发布新软件和功能，由于现代软件交付机制中的漏洞，这带来了安全挑战。强调了防止攻击的紧迫性，指出到 2024 年，预计 [96% 的代码库将包含开源代码](https://www.synopsys.com/blogs/software-security/open-source-trends-ossra-report.html)。

- **Secure Software Delivery 对 Modular 至关重要**：**XZ supply chain attack** 强调了建立强大防御体系以应对供应链漏洞的必要性，自 **Mojo** 首次发布以来，安全软件交付一直是 **Modular** 关注的重点。

**提到的链接**：<a href="https://www.modular.com/blog/preventing-supply-chain-attacks-at-modular">Modular: Preventing supply chain attacks at Modular</a>：我们正在为世界构建下一代 **AI** 开发者平台。查看我们的最新文章：Preventing supply chain attacks at Modular

---

**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1232950369269583953)** (2 条消息): 

- **淡化 Processing Units 的重要性**：一位成员表达了这样的观点：**processing units** 可能并不像人们普遍认为的那样关键。
- **通往量子简化的几何门户**：在讨论 **amplituhedron** 时，一位成员建议几何结构可以使复杂的量子现象（如粒子散射振幅）更加**易于理解**。提议利用几何学优化 **quantum algorithms** 和 **circuit designs**，可能有助于降低复杂性和噪声。
- **用几何可视化量子态**：提到了 **Bloch sphere**，这是一种通过几何变换可视化 **quantum gates** 对 **qubits** 影响的方法。虽然对单个 **qubit** 有效，但扩展到多个 **qubits** 并表示 **entanglement** 的挑战可能需要复杂的超维空间。
- **Machine Learning 作为高维空间的解码器**：该成员认为，随着 **qubit** 数量增加，**quantum entanglement** 的可视化变得更加复杂，**machine learning** 可能有助于破译由此产生的复杂图表。

---

**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1233019327276646450)** (36 条消息🔥): 

- **免受伤害**：针对在 **Mojo** 自定义类型中因不正确的手动内存管理而可能导致 PC 遭受不可逆损坏的询问，得到的答复是：**Operating System** 会在进程退出后清理内存，且 **Mojo** 不需要手动内存管理。
  
- **Mojo 函数基础**：关于 **Python** 到 **Mojo** 转换的讨论揭示了两种函数类型 `def` 和 `fn` 都是可行的选择。对话包括关于这两种不同函数定义以及如何在 **Mojo** 中声明变量的**代码示例和解释**。[此处描述了函数声明。](https://docs.modular.com/mojo/manual/functions)

- **新手的学习曲线**：在讨论理解 **Mojo** 细微差别的过程中，建议社区成员首先专注于让代码运行起来，因为在一门新兴编程语言中经历变化是正常的。语言的演进以及未来可能需要的重构被强调为学习过程的一部分。

- **Mojo 列表多样性查询**：针对 **Mojo** 处理混合数据类型列表的能力的提问得到了解答，透露虽然可行，但目前的方法被认为是 "hacky"。给出的示例显示了使用 `Variant` 包含整数和字符串的列表，[如这些针对 Ints 和 Floats 的 Gists 所示](https://gist.github.com/modularbot/c67e0a66a97aa32314d248f4721f75e2) 以及 [针对 Ints 和 StringLiterals 的示例](https://gist.github.com/modularbot/1a5beaf165761b55e2f743b3151210eb)。

- **拥抱 Mojo 之旅**：编程新手受到了社区的热烈欢迎，并被提醒最初应专注于编写可运行的代码。强调**熟练源于实践**，鼓励保持适应性，并为不断发展的新编程语言格局做好准备。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/pokemon-pikachu-clap-clapping-clapping-gif-gif-13465728489229726846">Pokemon Pikachu GIF - Pokemon Pikachu Clap - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://gist.github.com/modularbot/c67e0a66a97aa32314d248f4721f75e2">playground.mojo</a>: GitHub Gist: 立即分享代码、笔记和片段。</li><li><a href="https://gist.github.com/modularbot/1a5beaf165761b55e2f743b3151210eb">playground.mojo</a>: GitHub Gist: 立即分享代码、笔记和片段。
</li>
</ul>

</div>

---

**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1232977987025240064)** (5 条消息):

- **Mojo vs. Rust：对比概览**：[lobste.rs 上的讨论](https://lobste.rs/s/a3yoi6/mojo_vs_rust_is_mojo_faster_than_rust#c_3zamz6) 批评了 **Mojo** 可能不如 **Rust** 安全且速度较慢，理由是写时复制（copy on write）语义和 inout 参数的问题。该评论还指出，Mojo 的营销策略可能掩盖了对稳健技术进步的需求。
- **期待 Mojo 的基准测试首秀**：一位用户表达了对 **Mojo** 加入未来编程 **基准测试竞赛（benchmark competitions）** 的兴奋，暗示了社区对查看经验性能数据的兴趣。
- **基准测试：开发者热衷的业余爱好**：一位成员评论了 **开发者社区关于编程语言基准测试的热烈讨论**，指出一些开发者倾向于过分看重 GitHub 上的初步速度基准测试，尽管这些测试应更多被视为参考指标而非绝对标准。
- **提倡借用默认值（Borrowed Defaults）**：在一位以解释异步 Rust 闻名的 Rust 社区成员发表评论后，一位用户承认了 **Mojo** 某些特性优于 **Rust** 的益处。对话涉及了借用引用（borrowed references）以及如何更好地将其展示为 Mojo 的优势。
- **在学术界传播 Mojo**：一位用户分享了一个专注于 **Python 和 Mojo** 的 **GDSC 活动链接**：[Python and Mojo: Good, Bad, and the Future](https://gdsc.community.dev/events/details/developer-student-clubs-budapest-university-of-technology-and-economics-presents-python-and-mojo-good-bad-and-the-future/)。该活动旨在让学生熟悉 Mojo，强调其与 Python 的集成及其在系统编程中的潜力。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://lobste.rs/s/a3yoi6/mojo_vs_rust_is_mojo_faster_than_rust#c_3zamz6)">Mojo vs. Rust: is Mojo faster than Rust? | Lobsters</a>：未找到描述</li><li><a href="https://gdsc.community.dev/events/details/developer-student-clubs-budapest-university-of-technology-and-economics-presents-python-and-mojo-good-bad-and-the-future/.">Python and Mojo: Good, Bad and the Future | Google Developer Student Clubs</a>：线下活动 - 加入我们，参加关于 Mojo 的独家演示，这是一种基于 Python 语法并具备系统编程能力的语言。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1232961337957683250)** (6 条消息): 

- **对提交 Issue 的好奇**：有人请求为某个未指明的事项提交 issue，并对该话题表示好奇。
- **Issue 提交完成**：关于 **Mojo 编程语言** 的一个 issue 已在 GitHub 上提交，[相关 issue 链接](https://github.com/modularml/mojo/issues/2410) 已确认。
- **探索 `__copyinit__` 语义**：链接的 [GitHub Gist](https://gist.github.com/modularbot/6aed759930420cd70f38795dbcb874fe) 提出了一个问题：是由类型作者来实现写时复制语义，还是应该提交另一个 issue。
- **邀请通过 Issue 进行跟踪**：建议提交 issue 以使有关 `__copyinit__` 的行为更具可跟踪性，确保获得适当的回复。
- **等级提升公告**：**ModularBot** 庆祝一位用户在社区中晋升至 9 级。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/2410)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://gist.github.com/modularbot/6aed759930420cd70f38795dbcb874fe">playground.mojo</a>：GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1233167906099957850)** (3 条消息): 

- **对持续改进的乐观态度**：一位成员表达了积极的看法，认为性能可能会随着时间的推移而提高。
- **性能提升的巧合**：有人有趣地注意到，尽管存在差异，PyTorch 和 TensorFlow 报告了相同的性能提升。
- **对性能一致性的好奇**：一位成员询问如果重新运行性能提升测试，结果会是如何。
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1232957962767564800)** (11 条消息🔥):

- **消除关于重载解析（Overload Resolution）的困惑**：编程中的重载解析会优先考虑参数较少的函数。因此，Trait 中优先级较低的方法可以被同时声明了这两者的类型所覆盖。
- **无需额外参数的 Trait 一致性**：有人指出，Trait 不需要为了符合一致性而声明一个带有 `none` 参数的 `__eq__` 方法；这可以简化 Trait 的声明。
- **潜在的 SIMD 相等性兼容性**：稍作修改可能允许 SIMD 在不改变 Trait 声明的情况下符合 EqualityComparable。
- **冗余参数的困境**：讨论的方法调整的缺点是留下了冗余的 `none` 参数，尽管它通常不直接在 dunder methods（双下划线方法）中使用。
- **通过 `kgen.pack.load` 提升代码效率**：在 `printf` 函数中使用 `kgen.pack.load` 修改代码后，实现了更高效的更新：14 处插入和 985 处删除。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1233000905058680842)** (44 条消息🔥): 

- **AI 猎杀网络喷子**：一位用户分享了创建一个针对霸凌者的反喷子 AI 的计划，并征求关于该 Bot 可以采取的其他行动的建议。
- **使用 Mistral 和 Llama3 进行 SQL 问答**：一位用户在使用 Mistral 或 Llama3 等开源模型时遇到了 SQL 响应过于冗长的问题，随后又遇到了 `OutputParserException`。讨论内容包括 Ollama 对结构化输出的支持，以及使用这些模型调用 SQL Agents 的示例。
- **理解 Redis 与 LangChain 的集成**：澄清了 **stores**（存储）与 **chat memory**（聊天记忆）之间的区别；前者是由 `RedisStore` 类访问的通用键值存储，而后者专门用于通过 Redis Chat Message History 集成按会话持久化聊天消息。
- **LangChain 模型调用语法支持**：一位用户寻求关于如何在 JavaScript 中将 Prompt 合并到 LangChain 模型调用中的建议，并获得了关于使用 `ChatPromptTemplate` 和 `pipe` 等实例方法进行 Prompt 链式调用的指导。
- **澄清 Gemini 1.5 Pro 模型的访问**：用户讨论了如何将 Gemini 1.5 Pro 与 LangChain 配合使用；正确的用法涉及 `ChatVertexAI`，并指出无法通过 ChatGoogleGenerativeAI 访问 Gemini 模型。正确的实现需要设置 `GOOGLE_APPLICATION_CREDENTIALS` 变量。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/docs/integrations/chat/">聊天模型 | 🦜️🔗 LangChain</a>：特性（原生支持）</li><li><a href="https://www.reddit.com/r/TradingProSquad_/comments/1c9fvax/tradingview_cracked_for_desktop_pc_app_windows/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://python.langchain.com/docs/integrations/memory/redis_chat_message_history/">Redis | 🦜️🔗 LangChain</a>：[Redis (远程字典)</li><li><a href="https://python.langchain.com/docs/integrations/stores/redis/">RedisStore | 🦜️🔗 LangChain</a>：RedisStore 是 ByteStore 的一种实现，用于存储</li><li><a href="https://github.com/langchain-ai/langchain/issues/20924">OllamaFunctions 无法工作 - 收到 Ollama 不支持的消息类型 · Issue #20924 · langchain-ai/langchain</a>：检查了其他资源。我为此 Issue 添加了一个非常详细的标题。我使用集成搜索查询了 LangChain 文档。我使用 GitHub 搜索查找了类似问题并...</li><li><a href="https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/">ChatVertexAI | 🦜️🔗 LangChain</a>：注意：这与 Google PaLM 集成是分开的。Google 已经</li><li><a href="https://python.langchain.com/docs/integrations/chat/google_generative_ai/">Google AI 聊天模型 | 🦜️🔗 LangChain</a>：访问 Google AI 的 gemini 和 gemini-vision 模型，以及其他</li><li><a href="https://api.js.langchain.com/classes/langchain_core_tools.Tool.html#invoke>)">Tool | LangChain.js - v0.1.36</a>：未找到描述</li><li><a href="https://api.js.langchain.com/classes/langchain_core_tools.DynamicTool.html#invoke>)">DynamicTool | LangChain.js - v0.1.36</a>：未找到描述</li><li><a href="https://api.js.langchain.com/classes/langchain_core_tools.StructuredTool.html#invoke>)">StructuredTool | LangChain.js - v0.1.36</a>：未找到描述</li><li><a href="https://api.js.langchain.com/classes/langchain_core_tools.DynamicStructuredTool.html#invoke>)">DynamicStructuredTool | LangChain.js - v0.1.36</a>：未找到描述</li><li><a href="https://api.js.langchain.com/interfaces/langchain_core_tools.ToolInterface.html#invoke>)">ToolInterface | LangChain.js - v0.1.36</a>：未找到描述
</li>
</ul>

</div>

**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1232935474184847381)** (1 messages): 

- **LLaMA Prompt Template Queries**: 一位成员询问了在 **LLaMA3 prompts** 中使用 header 提供上下文的方法，并引用了 [官方文档](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/)。由于该模型较新，有人对文档的完整性表示了担忧。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1232957994052878336)** (5 messages): 

- **Collate Launches Personalized Newsletters**: Vel_y 宣布推出 [Collate](https://collate.one/newsletter)，这是一项将文章和 PDF 转换为简明每日 Newsletter 的服务。该平台提供了一种管理信息过载的方法，将保存的内容转化为易于消化的 Newsletter，并在此处提供了“立即尝试”选项：[here](https://collate-news.streamlit.app/?embed_options=dark_theme)。

- **BlogIQ Streamlines Content Creation**: Vishal_blueb 介绍了 [BlogIQ](https://github.com/langchain-tech/BlogIQ)，这是一款结合了 OpenAI 和 Langchain 能力的 App，旨在辅助博主进行内容创作。该 App 被定位为 writesonic.com 和 copy.ai 等服务的克隆版，致力于简化博客内容开发流程。

- **LangGraph for Invoice Extraction**: Toffepeermeneer 分享了他们的第一个 LangGraph 项目，这是一个发票提取器，可以从图片中提取信息并存储到 Postgres 数据库中。该项目可以在 [GitHub](https://github.com/jwa91/LangGraph-Expense-Tracker) 上找到，并包含一个 Excalidraw 项目概览。

- **Galaxy AI Opens Access to Premium AI Models**: White_d3vil 宣布了 Galaxy AI，这项服务提供对包括 **GPT-4**、**GPT-4-1106-PREVIEW** 和 **Gemma** 在内的高级 AI 模型的 **免费** API 访问。这些 API 与 OpenAI 的格式兼容，便于项目集成。更多信息（包括其 Discord 服务器邀请）可以在[此处](https://discord.com/invite/BSphj69773)找到。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://collate.one/newsletter">Newsletter</a>: 从你的内容中创建简明扼要的电子邮件摘要</li><li><a href="https://github.com/jwa91/LangGraph-Expense-Tracker">GitHub - jwa91/LangGraph-Expense-Tracker: LangGraph - FastAPI - Postgresql - AI project</a>: LangGraph - FastAPI - Postgresql - AI 项目。通过在 GitHub 上创建账号为 jwa91/LangGraph-Expense-Tracker 的开发做出贡献。</li><li><a href="https://app.excalidraw.com/l/5NC0r7Sejhe/39ULXmBwigA">让白板协作变得简单</a>: 具有手绘体验的白板工具。非常适合进行面试、绘制图表、原型或草图等等！</li><li><a href="https://github.com/langchain-tech/BlogIQ">GitHub - langchain-tech/BlogIQ: writesonic.com &amp; copy.ai 的克隆版 - BlogIQ 是一款由 OpenAI 和 Langchain 驱动的创新 App，旨在简化博主的内容创作流程。</a>: writesonic.com &amp; copy.ai 的克隆版 - BlogIQ 是一款由 OpenAI 和 Langchain 驱动的创新 App，旨在简化博主的内容创作流程。 - langchain-tech/BlogIQ</li><li><a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>: 未找到描述
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1232980107371745280)** (35 messages🔥): 

_

- **Apple 发布开源模型**：Apple 进入开源领域，推出了比预期更小的模型，包括 [Hugging Face 上的 270M 参数模型](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)，以及 450M、1.1B 和 3B 变体。
- **Dify 的应用开发平台受到关注**：Dify 提供了一个开源的 LLM 应用开发平台，集成了 AI workflow 和模型管理等多种功能；然而，一些用户对其缺乏 [loops 和 context scopes](https://github.com/langgenius/dify?tab=readme-ov-file) 表示担忧。
- **用于训练 LLM 的新 PyTorch 库**：PyTorch 发布了 [Torchtitan](https://github.com/pytorch/torchtitan)，这是一个支持从零开始训练 llama3 等大语言模型的新库。
- **对 SORA 视频生成的关注**：回顾 SORA，这是 OpenAI 开发的一款先进视频生成模型，可以创建长达一分钟的连贯视频，[FXGuide 文章](https://www.fxguide.com/fxfeatured/actually-using-sora/)中分享了早期用户的细节和反馈。
- **处理 Claude 3 输出的引号**：在关于 Opus 的 Claude 3 因引号导致 JSON 解析错误的讨论中，一位成员建议要求模型对问题字符进行转义，这已被证明非常有效，尤其是在处理 CSV 输出时。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，用于在移动设备、桌面和会议室系统上进行视频和音频会议、聊天和网络研讨会。Zoom ...</li><li><a href="https://www.fxguide.com/fxfeatured/actually-using-sora/">实际使用 SORA - fxguide</a>：来自 Air Head 团队的 SORA 当前确切状态，或者“如何在生成式 AI 的随机性下讲述一个连贯的故事”。</li><li><a href="https://gorilla.cs.berkeley.edu/leaderboard.html">
        Berkeley Function Calling Leaderboard (又名 Berkeley Tool Calling
        Leaderboard)
    </a>：未找到描述</li><li><a href="https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM">Stanford CS25 - Transformers United</a>：斯坦福 CS25：Transformers United。自 2017 年推出以来，Transformers 彻底改变了自然语言处理 (NLP)。现在，Transformers 正在...</li><li><a href="https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca">OpenELM Instruct Models - apple 集合</a>：未找到描述</li><li><a href="https://vram.asmirnov.xyz">VRAM Calculator</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtitan">GitHub - pytorch/torchtitan: 用于大模型训练的原生 PyTorch 库</a>：一个用于大模型训练的原生 PyTorch 库。通过在 GitHub 上创建账号来为 pytorch/torchtitan 做出贡献。</li><li><a href="https://github.com/langgenius/dify?tab=readme-ov-file">GitHub - langgenius/dify: Dify 是一个开源的 LLM 应用开发平台。Dify 直观的界面结合了 AI workflow、RAG pipeline、Agent 能力、模型管理、可观测性功能等，让你快速从原型走向生产。</a>：Dify 是一个开源的 LLM 应用开发平台。Dify 直观的界面结合了 AI workflow、RAG pipeline、Agent 能力、模型管理、可观测性功能等，让你...</li><li><a href="https://www.fxguide.com/fxfeatured/act">动作节拍：来自《惊天危机》的 6 个场景 - fxguide</a>：Roland Emmerich 的《惊天危机》中 6 个最大场景的解析及其背后的视觉效果。
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1233416150063517798)** (12 条消息🔥):

- **Mixture of Depths 论文展示**：关于名为 'Mixture of Depths' 论文的讨论始于通过 [此链接](https://paper-club.ivanleo.com/papers/mixture-of-depths) 分享的演示。该论文介绍了一种通过使用改进的 MOE routing 机制来动态调整 Transformer 层中的 token 流，从而加速 Transformer 训练的方法。
- **变革 Attention 机制**：*Mixture Of Depths* 论文提出了一种解决长序列 Transformer 扩展问题的方案。通过在新的 MOD 层和普通 Transformer 层之间交替，计算 Attention 的需求减少了一半，从而改善了各种训练要素。
- **大语言模型（LLM）在现实应用中的挑战**：引用了另一篇探讨 LLM 部署挑战（如计算资源需求）的论文。提到在会议摘要任务中，即使经过 fine-tuning，较小的紧凑型 LLM 通常也无法超越较大的 zero-shot LLM，详见 [摘要](https://arxiv.org/abs/2402.00841)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.00841">Tiny Titans: Can Smaller Large Language Models Punch Above Their Weight in the Real World for Meeting Summarization?</a>：大语言模型（LLM）已展示出解决广泛任务的惊人能力，而无需在特定任务的数据集上进行显式的 fine-tuning。然而，在现实中部署 LLM...</li><li><a href="https://paper-club.ivanleo.com/papers/mixture-of-depths">Nextra: the next docs builder</a>：Nextra：下一代文档构建器
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1233174487709650976)** (1 条消息): 

- **Vector DBs 聊天录音请求**：一位成员对原定于 4 月 26 日星期五进行的 **Vector DBs 聊天** 表示感兴趣，但提到他们可能会错过。他们询问是否可以录制聊天内容，并表示虽然这不常见，但以前也曾这样做过。
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1232952950465232967)** (40 条消息🔥): 

- **理解 Phi-3-Mini-4K Instruct 的用法**：讨论提供了关于在 llamafile 中使用 Phi-3-Mini-4K-Instruct 的见解；[强调了 GGUF 格式的细节](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf#how-to-use-with-llamafile)，提到了设置模型及其属性的步骤，包括高质量和推理密集型数据集。
- **Mixtral 8x22B Instruct llamafile 快速入门**：提到了一份关于 Mixtral 8x22B Instruct llamafile 的 README 更新，建议在从提供的 [快速入门](https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile) 指南下载 `.cat*` 文件时，使用 `curl -L` 处理 CDN 上的重定向。
- **llamafile 的 CPU 特性要求**：一位用户在 Mac M1 上尝试运行 llamafile 时遇到了与 AVX CPU 特性要求相关的“fatal error”。建议 [重启电脑](https://github.com/Mozilla-Ocho/llamafile/issues/327#issuecomment-2053680659)，并考虑在 8GB RAM 的系统上使用更小的模型。
- **Windows Defender 将 llamafile 标记为 Trojan**：一位用户报告 Windows Defender 将 llamafile 标记为 Trojan；建议包括尝试替代环境（如虚拟机）或在 Windows Defender 设置中将文件夹列入白名单。[Windows Defender 支持](https://www.microsoft.com/en-us/wdsi/filesubmission) 仅保证官方发布页面上的二进制文件。
- **使用 llamafile 的资源需求和故障排除**：用户讨论了运行 8x22B 模型的资源需求，指出了显著的 RAM 需求以及由于高内存占用可能导致的崩溃。提到 [Mistral 8x22B 模型](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8) 至少推荐 128GB RAM。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hacks.mozilla.org/2024/04/llamafiles-progress-four-months-in/">Llamafile 的进展，四个月后的回顾 – Mozilla Hacks - Web 开发者博客</a>：Mozilla 的创新小组去年启动了 llamafile 项目，它已成为 Mozilla 在 GitHub 上最受欢迎的仓库之一。</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf#how-to-use-with-llamafile">microsoft/Phi-3-mini-4k-instruct-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf#how-to-use-with-llamafile>">microsoft/Phi-3-mini-4k-instruct-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8">发布 llamafile v0.8 · Mozilla-Ocho/llamafile</a>：llamafile 允许你通过单个文件分发和运行 LLM。llamafile 是 Mozilla Ocho 在 2023 年 11 月推出的本地 LLM 推理工具，提供卓越的性能和二进制可移植性...</li><li><a href="https://www.microsoft.com/en-us/wdsi/filesubmission">提交文件进行恶意软件分析 - Microsoft Security Intelligence</a>：未找到描述</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/issues/327#issuecomment-2053680659">致命错误：M1 需要 CPU 特性 AVX · Issue #327 · Mozilla-Ocho/llamafile</a>：我在 Apple M1 上尝试运行入门示例时遇到了一个奇怪的问题。sh -c &quot;./llava-v1.5-7b-q4.llamafile&quot; -- ./llava-v1.5-7b-q4.llamafile: fatal error: the cpu feature AVX...</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/issues/327#issuec">致命错误：M1 需要 CPU 特性 AVX · Issue #327 · Mozilla-Ocho/llamafile</a>：我在 Apple M1 上尝试运行入门示例时遇到了一个奇怪的问题。sh -c &quot;./llava-v1.5-7b-q4.llamafile&quot; -- ./llava-v1.5-7b-q4.llamafile: fatal error: the cpu feature AVX...</li><li><a href="https://github.com/htop-dev/htop/issues/1443">htop 在 Linux 上不报告共享内存使用情况 · Issue #1443 · htop-dev/htop</a>：在下面的截图中，你会看到我的一个进程正在使用 139GB 内存，但 htop 报告系统仅使用了 6GB RAM。这是因为 htop 隐藏了 mmap(MAP_SHARED) 内存。这导致了...</li><li><a href="https://vt.tiktok.com/ZSFctaKnm/">TikTok - Make Your Day</a>：未找到描述</li><li><a href="https://blog.mozilla.ai/local-llm-as-judge-evaluation-with-lm-buddy-prometheus-and-llamafile/">使用 lm-buddy、Prometheus 和 llamafile 进行本地 LLM-as-judge 评估</a>：在 AI 新闻周期中，每天都有新模型发布，成本和评估虽然不常被提及，但对开发者和企业至关重要。</li><li><a href="https://huggingface.co/jartine">jartine (Justine)</a>：未找到描述</li><li><a href="https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile">jartine/Mixtral-8x22B-Instruct-v0.1-llamafile · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile/resolve/main/Mixtral-8x22B-Instruct-v0.1.Q8_0.llamafile.cat0">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile/resolve/main/Mixtral-8x22B-Instruct-v0.1.Q8_0.llamafile.cat1">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile/resolve/main/Mixtral-8x22B-Instruct-v0.1.Q8_0.llamafile.cat2">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1233172129348980899)** (6 条消息): 

- **Llama-3-70b 在 Judgemark 中表现出色**：**Llama-3-70b** 在 [**Judgemark**](https://eqbench.com/judgemark.html) 上展示了令人期待的结果，表明其作为微调 **disco-judge** 基础模型的巨大潜力。Judgemark 评估模型评判创意写作的能力，并要求至少 8k 的支持上下文长度。
  
- **评估方面的潜在合作**：一位用户对合作持开放态度，分享了从创建 [评估](https://sampaech.substack.com/p/creating-magi-a-hard-subset-of-mmlu) 中获得的见解，并建议使用他们精心设计的评判提示词（prompt）设计来测试复杂的评分标准。

- **借鉴 Magazine 和 MMLU**：用户 @_jp1_ 称赞了一篇文章中关于创建 **MAGI** 的工作。**MAGI** 是 **MMLU** 的一个极具选择性和辨别力的子集，旨在挑战和区分高能力模型。

- **用于微调的 Judgemark 数据**：一位用户表示准备好格式化并分享所有 **Judgemark 输出**，以便可能用于微调数据集，并询问了这些数据集的收集过程。

- **Phi-3-mini-4k-instruct 的混合结果**：尽管在 **eq-bench** 上的表现不如其公布的评估结果那样令人印象深刻，但 **Phi-3-mini-4k-instruct** 已列在 [eq-bench leaderboard](https://eqbench.com/judgemark.html) 上，用户可能需要滚动页面才能找到它。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://eqbench.com/judgemark.html">EQ-Bench Judgemark Leaderboard</a>：未找到描述</li><li><a href="https://sampaech.substack.com/p/creating-magi-a-hard-subset-of-mmlu">🧙Creating MAGI: A hard subset of MMLU and AGIEval</a>：为现有基准测试增加余量和区分度
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1232998702931644416)** (4 messages): 

```html
<ul>
  <li><strong>讨论 API 重点和库的易用性：</strong> Tgi 被定位为 API 优先并优先考虑低延迟，而 vllm 则因其作为易于使用的库而受到赞誉，强调成本效益和高吞吐量部署。</li>
  <li><strong>Hugging Face 上的批量生成咨询：</strong> 关于批量生成能力的辩论出现在 Hugging Face 的 <a href="https://github.com/huggingface/text-generation-inference/issues/1008#issuecomment-1742588516"><strong>GitHub Issue #1008</strong></a> 中，展示了社区驱动的问题解决方式。</li>
  <li><strong>DiscoLM 推理速度困扰：</strong> 一位成员报告了 DiscoLM_German_7b_v1 在高性能计算系统上的初始化和推理时间缓慢，这与在没有 GPU 的本地设置上快得多的时间形成鲜明对比。</li>
  <li><strong>DiscoLM 中潜在的配置错误：</strong> 另一位成员建议确保使用 <code>device_map='auto'</code> 正确加载模型，并预期在使用 2x V100 GPUs 进行推理时会有显著的速度提升。</li>
</ul>
```

**提到的链接**：<a href="https://github.com/huggingface/text-generation-inference/issues/1008#issuecomment-1742588516">Batch generate? · Issue #1008 · huggingface/text-generation-inference</a>：系统信息 你好，我想问一下是否可以进行批量生成？client = Client("http://127.0.0.1:8081",timeout = 60) gen_t = client.generate(batch_text,max_new_tokens=64) generate c...

  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1232990125471043666)** (7 messages): 

- **DiscoLM-70b 部署困境**：一位成员描述了运行 **DiscoLM-70b** 时遇到的问题，包括 "Template not found" 错误以及 `huggingface/text-generation-inference` 的 `/generate` 端点输出无意义内容。
- **对 DiscoLM 模型的赞赏**：Salvasfw 表达了他们对 DiscoLM 系列模型的深切赞赏，即使在处理故障排除挑战时也是如此。
- **对强大 MoE 模型的沉思**：有人猜测构建和训练 **8 x phi-3 MoE 模型** 的潜力，并对其能力感到好奇。
- **Mini 4k Llamafication 成功**：据 crispstrobe 称，`Phi-3-mini-4k` 已成功 Llama 化，EQ-Bench 评分 (v2_de) 为 51.41，表现尚可，尽管德语输出中存在一些错误。该模型并未专门针对德语数据进行训练，结果表明它可能具有进一步训练的潜力。
- **GGUF 模型下载**：Johannhartmann 强调了 **GGUF** 模型的受欢迎程度，该模型在发布后的两天内就有 1500 次下载。


  

---



**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1233040210372202529)** (12 messages🔥): 

- **辩论 Claude 的能力**：一场在线讨论探讨了 **Claude 的 RLAIF 训练**，认为其手法轻巧且执行良好。据报告，Claude 的行为展现了出人意料的结构和深刻的理解，与 Anthropic 的愿景“几乎正交”，散发出“荣格个体化”和“菩萨气息”。该线程还推测了 RLAIF 与基础模型潜在动力学的影响，并讨论了纠正 Claude 模式崩溃（mode collapse）的可能性 ([Claude 的对话线程](https://x.com/repligate/status/1783426037210026372?s=46&t=xxWoJxAS_7-BBFC2ro84Zw))。

- **商业部署中的 RLHF 与 KTO**：针对关于 **RLHF (Reinforcement Learning from Human Feedback)** 稳定性的查询，有人建议其应用取决于具体语境，而 **KTO (Knowledge-Targeted Optimization)** 可能更适合某些应用任务。

- **转换训练方法以改善结果**：一次访谈中提到，从 **SFT (Supervised Fine Tuning)** 转向 **DPO (Direct Preference Optimization)** 提供了更好的结果，随后转向 **KTO** 根据用户反馈进一步提升了性能。

- **RLHF 中的复杂性与细微差别**：有一种观点认为 **RLHF** 比通常认为的更加微妙，特别是考虑到数据的多样性以及它如何与下游评估指标相互作用。

- **理解梯度范数突增 (Grad Norm Spikes)**：频道中有人要求澄清为什么预训练期间梯度范数的突增可能是不利的，但回复中未提供详细解释。

**提到的链接**：<a href="https://x.com/repligate/status/1783426037210026372?s=46&t=xxWoJxAS_7-BBFC2ro84Zw">来自 j⧉nus (@repligate) 的推文</a>：毫无疑问，在 teacher-forcing 之上，有各种方法可以进行 RL/生成-判别/合成数据/类自博弈（self-play-esque）训练，使模型变得更聪明，但尤其是更 ...

  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1233150418717114458)** (4 messages): 

- **“向你请教 (Pick Your Brain)” 的难题**：*Nathan Lambert* 提到他对 **“pick your brain”** 这个词感到不适，尤其是现在他因为忙碌往往会拒绝此类请求。
- **对“请教”的幽默看法**：*Vsreekanti* 幽默地回应了对 **brain-picking** 的不适，建议应该询问是哪种“pick”，并开玩笑说更喜欢脑叶切除术（lobotomy）。
- **“请教”作为模糊请求**：*Drj.bet* 补充说，**“pick your brain”** 这一短语通常暗示了对话的欲望，但并没有具体的特定问题。
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1233434451896438927)** (1 messages): 

- **处理 Python 的 CPU 友好型语言模型**：分享了一篇题为《低成本语言模型：Python 代码生成的综述与性能评估》的文章，深入探讨了 **CPU 友好型语言模型** 的性能评估。它介绍了一个包含 60 个编程问题的数据集，并讨论了使用 Chain-of-Thought 提示词引导模型解决问题的方法，可通过此链接获取：[查看 PDF](https://arxiv.org/abs/2404.11160)。

**提到的链接**：<a href="https://arxiv.org/abs/2404.11160">Low-Cost Language Models: Survey and Performance Evaluation on Python Code Generation</a>：Large Language Models (LLMs) 由于能够解决各种问题并产生高质量的结果，已成为许多 Natural Language Processing (NLP) 任务的首选方案。具体而言...

  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1232960311133343764)** (4 messages): 

- **为图像识别微调 Moondream**：分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=Gwq7smiWLtc)，演示了在验证码（Captcha）图像数据集上微调 Moondream 视觉语言模型的过程。它被描述为提高下游任务性能的指南。
- **多伦多 AI 开发者聚会**：重点介绍了即将在多伦多 **Cohere space** 举行的本地及开源 AI 开发者聚会，并附有 [活动链接](https://lu.ma/devs5)。一位名为 *Andrei* 的成员正在协助组织，活动内容包括闪电演讲、演示和社交。
- **介绍 Snowflake Arctic**：另一个 [YouTube 视频](https://www.youtube.com/watch?v=nV6eIjnHEH0) 展示了 Snowflake Arctic，这是一个专为高性价比 AI 解决方案设计的面向企业的 LLM。简而言之，该视频介绍了大型语言模型领域的新成员。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=Gwq7smiWLtc">Finetuning Moondream Vision Language Model</a>：此视频演示了如何微调 moondream 以提高下游任务的性能。在本示例中，我们将在验证码图像数据集上进行微调...</li><li><a href="https://www.youtube.com/watch?v=nV6eIjnHEH0">Snowflake Arctic: The Best LLM for Enterprise AI</a>：今天，Snowflake AI 研究团队激动地推出了 Snowflake Arctic，这是一款顶级的面向企业的 LLM，它推动了高性价比 AI 的前沿...</li><li><a href="https://lu.ma/devs5">Toronto Local &amp; Open-Source AI Developer Meetup · Luma</a>：本地及开源 AI 开发者聚会即将来到多伦多！加入 Ollamas 及其朋友们在 Cohere 空间的聚会！特别感谢 abetlen (Andrei)...
</li>
</ul>

</div>
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1233183712859000842)** (2 messages):

- **GPT 爱好者的 GUI**：一位成员发现了 [jan.ai](https://jan.ai/)，它被誉为一个用于在本地运行模型且用户友好的图形用户界面。
- **小模型，大雄心**：一位成员分享了 [OpenELM](https://huggingface.co/apple/OpenELM)，这是由 Apple 发布的一个高效语言模型系列，提供预训练和指令微调模型，利用独特的逐层缩放策略来实现高效的参数分配。

**提到的链接**：<a href="https://huggingface.co/apple/OpenELM">apple/OpenELM · Hugging Face</a>：未找到描述

  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 条消息): 

venadore: 尝试让 Llama 3 进行主题复杂度分类，效果还不错
  

---



---