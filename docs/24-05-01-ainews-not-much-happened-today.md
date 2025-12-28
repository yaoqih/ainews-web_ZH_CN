---
companies:
- anthropic
- openai
- perplexity-ai
- amazon
- apple
- microsoft
- deepmind
date: '2024-05-02T00:47:12.556282Z'
description: '以下是该文本的中文翻译：


  **Anthropic** 在 **OpenAI** 发布约 4 个月后，推出了团队方案（team plan）和 iOS 应用。**Command-R 35B**
  模型在创意写作方面表现出色，超越了 **Goliath-120** 和 **Miqu-120** 等更大的模型。**Llama-3 8B** 模型现在支持 100
  万 token 的上下文窗口，仅需在单台 8xA800 GPU 机器上进行极少量训练，即可提升长上下文理解能力。**TensorRT-LLM** 的基准测试显示，在消费级硬件上，其速度比
  **llama.cpp** 快 30-70%。一项基准测试表明，**GPT2-Chat** 的推理能力可能优于 **GPT-4-Turbo**，尽管这一结果仍存争议。演示项目包括在
  Jetson Orin 上本地运行的自学习 **Llama-3** 语音智能体，以及一个自学习大动作模型（LAM）。**Amazon CodeWhisperer**
  已更名为 **Q Developer**，扩展了其生成式 AI 助手的功能。**苹果（Apple）** 计划在 iOS 18 和 macOS 15 中推出具备
  AI 功能的 Safari 浏览器，并搭载端侧大语言模型（LLM）。科技巨头主导了华盛顿的 AI 游说活动，而美国多家主流报纸则起诉 **OpenAI** 和
  **微软（Microsoft）** 侵犯版权。**DeepMind 的 AlphaZero** 在 9 小时内成为了最强棋手，其“自然化执行微调”（NExT）方法将大模型的代码推理能力提升了
  14-26%。**Stable Diffusion** 被广泛应用于各种图像生成场景。'
id: 97bbb126-3d3f-4d51-bfc9-052241177490
models:
- command-r-35b
- goliath-120
- miqu-120
- llama-3-8b
- tensorrt-llm
- llama-cpp
- gpt2-chat
- gpt-4-turbo
- llama-3
- deepmind-alphazero
original_slug: ainews-to-be-named-2666
people: []
title: 今天没什么事。
topics:
- creative-writing
- context-windows
- benchmarking
- model-performance
- self-learning
- function-calling
- retrieval-augmented-generation
- ai-assistants
- on-device-ai
- ai-lobbying
- copyright-infringement
- code-reasoning
- image-generation
---

  

---

**目录**

[TOC] 


---

# AI Reddit 摘要回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**LLM 模型与框架**

- **Command-R 35B 模型在创意写作方面表现出色**：在 /r/LocalLLaMA 中，Command-R 35B 模型[在创意基准测试中表现优于 Goliath-120 和 Miqu-120 等更大型的模型](https://www.reddit.com/r/LocalLLaMA/comments/1cgv10e/commandr_35b_is_incredible_for_creative_writing/)。正确的 Prompt 引导是释放其潜力的关键。
- **Llama-3 8B 模型上下文窗口扩展**：Llama-3 8B 模型[可以使用 100 万 Token 的上下文窗口](https://www.reddit.com/r/LocalLLaMA/comments/1cgzu2a/llama3_8b_256k_context_exl2_quants/)。[将上下文从 8K 扩展到 80K Token 提升了](https://arxiv.org/abs/2404.19553)长文本理解任务的性能，该过程仅在单台 8xA800 GPU 机器上使用 3500 个 GPT-4 生成的训练样本，耗时 8 小时完成。
- **TensorRT-LLM 在速度上超越 llama.cpp**：根据在消费级笔记本电脑和台式机上的基准测试，[在相同硬件上 TensorRT-LLM 比 llama.cpp 快 30-70%](https://jan.ai/post/benchmarking-nvidia-tensorrt-llm)。
- **基准测试显示 GPT2-Chat 的推理能力优于 GPT 4-Turbo**：在 /r/LocalLLaMA 中，一项包含 80 个 One-shot 任务的新基准测试表明，[**GPT2-Chat 可能比 GPT 4-Turbo 具有更好的推理能力**](https://www.reddit.com/r/LocalLLaMA/comments/1cgp7gi/lmsys_org_constantly_compares_new_gpt2_and_claude/)，尽管其整体智能略低。然而，一些用户对该结果表示怀疑。

**AI Agent 与机器人**

- **自学习 Llama-3 语音 Agent 演示**：一个[具备 Function Calling 和自动 RAG 功能的自学习 Llama-3 语音 Agent 演示](https://www.reddit.com/r/LocalLLaMA/comments/1cgtmuy/selflearning_llama3_voice_agent_with_function/)，在 Jetson Orin 上本地运行。
- **自学习 Large Action Model (LAM) 演示**：一个开源的[自学习 Large Action Model (LAM) 演示](https://www.reddit.com/r/LocalLLaMA/comments/1cgtmuy/selflearning_llama3_voice_agent_with_function/)，无需用户训练。

**AI 助手**

- **Amazon CodeWhisperer 更名为 Q Developer**：[Amazon CodeWhisperer 已更名为 Q Developer](https://www.aboutamazon.com/news/aws/amazon-q-generative-ai-assistant-aws)，扩展了其作为开发者生成式 AI 助手的功能。
- **苹果将推出支持 AI 的 Safari 浏览器**：[苹果计划推出支持 AI 的 Safari 浏览器](https://appleinsider.com/articles/24/04/30/apple-to-unveil-ai-enabled-safari-browser-alongside-new-operating-systems)，并在 iOS 18 和 macOS 15 中配备端侧 LLM。

**AI 伦理与治理**

- **华盛顿的 AI 游说热潮由科技巨头主导**：[科技巨头（Big Tech）正主导着华盛顿的 AI 游说热潮](https://time.com/6972134/ai-lobbying-tech-policy-surge/)，旨在影响 AI 政策。
- **美国主要报社起诉 OpenAI 和 Microsoft 侵犯版权**：[美国多家主要报社已对 OpenAI 和 Microsoft 提起诉讼](https://www.axios.com/2024/04/30/microsoft-openai-lawsuit-copyright-newspapers-alden-global)，指控其侵犯版权。

**AI 研究**

- **DeepMind 的 AlphaZero 在 9 小时内成为最强国际象棋选手**：从零开始，[DeepMind 的 AlphaZero 在短短 9 小时内就成为了最伟大的国际象棋选手](https://twitter.com/tsarnick/status/1785050900647862683)。
- **DeepMind 的 Naturalized Execution Tuning (NExT) 提升了 LLM 代码推理能力**：[DeepMind 的 NExT 提升了 LLM 的代码推理能力](https://www.marktechpost.com/2024/04/26/deepmind-researchers-propose-naturalized-execution-tuning-next-a-self-training-machine-learning-method-that-drastically-improves-the-llms-ability-to-reason-about-code-execution/?amp)，通过让模型检查执行轨迹并提供原理解释，将修复率提高了 14-26%。

**Stable Diffusion 与图像生成**

- **Stable Diffusion 被用于多种应用**：在 /r/StableDiffusion 中，Stable Diffusion 正被用于[生成逼真的自拍、服装选择等](https://www.reddit.com/r/StableDiffusion/comments/1ch5k0m/using_sd_for_other_things_than_nsfw_content/)，而不仅仅是 NSFW 内容。
- **ConsistentID 项目生成高质量肖像**：ConsistentID 项目[生成具有身份一致性和多样性的逼真肖像](https://www.reddit.com/r/StableDiffusion/comments/1cgsw94/consistentid_better_than_ipadapter/)，潜力可能超越 Ipadapter。
- **适用于 SDXL 的 HiDiffusion 生成高质量图像**：在 /r/StableDiffusion 中，[适用于 SDXL 的 HiDiffusion 可生成高质量图像](https://www.reddit.com/r/StableDiffusion/comments/1cgntxz/hidiffusion_for_sdxl_is_something/)，但需要将 CFG 设置为 20 以保持连贯性。

---

# AI Twitter 回顾

> 所有回顾均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和 flow engineering。

**Anthropic 发布 Claude iOS 应用及新功能**

- **Claude iOS 应用发布**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1785701418546180326) 宣布发布 Claude iOS 应用，将他们的 AI 带到移动设备。该应用现已在 App Store 上架。
- **新 Team 方案**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1785685692988940509) 为 Claude 推出了 Team 方案，提供更高的使用额度、用户管理、计费功能，并为处理复杂任务提供 200K context window。
- **即将推出的协作功能**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1785685697275552210) 预告了未来的功能，例如用于验证陈述的可靠来源引用，以及与数据仓库的集成，同时保持安全性和防护。

**AI 专家分享见解**

- **Demis Hassabis 谈 AI 加速科学**：[@demishassabis](https://twitter.com/demishassabis/status/1785646721252336084) 在 @TEDTalks 上发表演讲，讨论 AI 将如何加速科学发现，并帮助应对癌症和气候变化等重大挑战。
- **Yann LeCun 批评当前的 LLM**：[@ylecun](https://twitter.com/ylecun/status/1785290144561373351) 认为 LLM 中的知识积累不能替代真正的理解，并指出了表现出缺乏基本逻辑、常识以及无法承认错误的行为。

**个人经历与反思**

- **Anthropic 员工分享最喜欢的 Claude 帖子**：[@alexalbert__](https://twitter.com/alexalbert__/status/1785369914204938326)，一名 Anthropic 员工，分享了过去两个月公司 Slack 中最幽默的 10 个 Claude 帖子和梗图。
- **应对手部残疾与职业转型**：[@jxnlco](https://twitter.com/jxnlco/status/1785661195149615347) 分享了他在 2020 年因手部残疾而失去编程和工作能力的经历，以及为什么他现在选择做咨询而不是在节奏快速的初创公司工作。
- **带着对 ML 进展的见解离开 Scale AI**：[@russelljkaplan](https://twitter.com/russelljkaplan/status/1785483317397356648) 宣布在工作近 4 年后离开 @scale_AI，回顾了公司的成长以及他对 ML 未来的独特看法。他计划分享更多关于 ML 进展和下一步行动的想法。

**AI 研究与更新**

- **Lmsys.org 为社区提供未发布模型的访问权限**：[@lmsysorg](https://twitter.com/lmsysorg/status/1785394860754866234) 澄清说，他们与模型开发人员合作，为社区提供未发布模型的预览测试访问权限，旨在随着规模扩大以及与开源和商业供应商合作，引入更多模型。
- **关于使用 RLHF+PPO 进行指令遵循的 2020 年论文**：[@rasbt](https://twitter.com/rasbt/status/1785671664920920296) 强调了 Stiennon 等人在 2020 年发表的一篇论文，该论文在 InstructGPT 出现两年前就使用 RLHF+PPO 对 LLM 进行微调以进行指令遵循。
- **Meta 提出用于加速 LLM 的 multi-token prediction**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1785486711646040440) 和 [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1785666587879444645) 分享了 Meta 的一篇论文，该论文关于使用 multi-token prediction 更高效地训练 LM，在保持或提高下游性能的同时，推理速度提升高达 3 倍。

**其他话题**

- **机器学习书籍推荐**：[@svpino](https://twitter.com/svpino/status/1785664640913211439) 分享了他心目中前 3 名的 ML 书籍，涵盖了 ML 工作流、算法以及 Keras、PyTorch 和 Scikit-Learn 等深度学习工具。
- **对 Ilya Sutskever 论点的批评**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1785614874472485096) 质疑了 Sutskever 关于预测目标将成功创建完美先知（oracle）的说法。
- **梗图与幽默**：[@mervenoyann](https://twitter.com/mervenoyann/status/1785688139119353952) 和 [@BorisMPower](https://twitter.com/BorisMPower/status/1785555611943616651) 分享了幽默图片和梗图。

---

# AI Discord 回顾

> 摘要的摘要之摘要

**1. 大型语言模型 (LLM) 进展与基准测试**

- **[LLaMA 3](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)** 正受到关注，[Nous Research 在 LLaMA 3 8B 上的 Hermes 2 Pro](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF) 在 AGIEval 和 GPT4All Suite 等基准测试中表现优于原始模型。关于 LLM **quantizing** 的讨论指出，在质量显著下降前存在 [每权重 5.5 bits 的限制](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)。努力将 **context lengths** 扩展到典型限制之外，例如 [LLaMA 3 的 1M tokens](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k)，尽管实际限制平均为 100-200k。

- **Iterative methods** 如 [Meta 的 Iterative Reasoning Preference Optimization](https://arxiv.org/abs/2404.19733) 提升了 LLaMA-2-70B-Chat 在 GSM8K 和 ARC-Challenge 上的准确率。[Kolmogorov-Arnold Networks (KANs)](https://arxiv.org/abs/2404.19756) 被提议作为比 MLPs 更准确且具可解释性的替代方案。

- ScandEval 德国 NLG 任务上的 [LLaMA vs GPT-4 性能对比](https://scandeval.com/german-nlg/) 引起了兴趣，LLaMA 3 的表现优于 GPT-4。

**2. 高效 LLM 推理的优化与技术**

- 对 **efficient inference** 方法表现出浓厚兴趣，如用于向量矩阵近似的 [effort/bucketMul](http://kolinko.github.io/effort/)，LLM Paper Club 讨论的 [Ring Attention](https://arxiv.org/abs/2310.01889)，以及 llm.c 中的 [CUDA optimizations](https://github.com/karpathy/llm.c)（如 Flash Attention 和 CUDA Graphs）。

- 关于受生物合理性启发的 **binary vector representations** 用于 embeddings 的辩论，参考了 [CLIP](https://arxiv.org/abs/2103.00020)、[Dino](https://arxiv.org/abs/2104.14294) 和 RWKV LLM 方法。

- 提高 **transformer lens** 可解释性的技术，如 [tuned lens method](https://arxiv.org/abs/2303.08112)，以及探索 neural scaling laws 中的 [distributional simplicity bias](https://arxiv.org/abs/2402.04362)。

**3. 开源 AI 工具、库与框架**

- **[LlamaIndex](https://python.langchain.com/docs/use_cases/graph/constructing/)** 在文档知识图谱化方面受到关注，新的 **[LlamaIndex.TS v0.3](https://t.co/mBIrD9uh8c)** 提高了类型安全性和 Agent 支持。讨论了使用 MongoDB Atlas 作为 vector store。

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** 被广泛用于开源 LLM fine-tuning，新增了 LLaMA-3 prompt 策略以及用于编排的 [dstack 集成](https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md)。

- 对 **[llama.cpp](https://github.com/ggerganov/llama.cpp)** 优化的兴趣，包括 [Flash Attention 合并](https://github.com/ggerganov/llama.cpp/pull/5021) 以及支持 LLaMA 3 tokenization 的努力。**[LM Studio](https://discord.com/channels/1110598183144399058/1234988891153629205/)** 期待包含 llama.cpp 更新的 0.2.22 版本发布。

- **[Tinygrad](https://github.com/tinygrad/tinygrad)** 的进展包括将 `Scalar` 重命名为 `ConstType`，探索 const 支持变量，以及 geohot 提交的 [symbolic shape handling](https://github.com/tinygrad/tinygrad/pull/4362)。

**4. 多模态与检索增强 AI 能力**

- 发布了 **multimodal models**，如用于编程的 [Snowflake Arctic 480B](https://openrouter.ai/models/snowflake/snowflake-arctic-instruct) 和 Fireworks 的 [FireLLaVA 13B](https://openrouter.ai/models/fireworks/firellava-13b)（一个基于指令数据训练的开源 LLaVA 模型）。

- 探索使用 [LangChain](https://python.langchain.com/docs/integrations/chat/fireworks/) 结合 Mistral Large 和 LlamaIndex 进行 **Retrieval-Augmented Generation (RAG)**，并提供了关于 [构建高级 RAG 助手](https://youtu.be/ol2QMp64lgo) 和 [复杂度自适应 RAG 策略](https://www.youtube.com/watch?v=QnXdlqEUW80) 的教程。

- 发布了 **multimodal AI assistants**，如用于 Unreal Engine 的 [Neuralgameworks](https://neuralgameworks.com) 和 AI 产品 [Rabbit R1](https://www.youtube.com/watch?v=ddTV12hErTc&ab_channel=MarquesBrownlee)，引发了对 [集成 OpenInterpreter](https://discord.com/channels/1146610656779440188/1194880263122075688/1234781691109703732) 的兴趣。

- **医疗 AI** 的进展，如 [使用 OpenCLIP 的心脏超声研究](https://doi.org/10.1038/s41591-024-02959-y) 和 Google 用于医疗保健的 [Med-Gemini multimodal models](https://youtu.be/xohuoN2WBZs?si=Ku6cztykld6dZLN9)。


---



# PART 1: 高层级 Discord 摘要

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA C++ 优化见解**：开发者分享了 [CUDA C++ Core Libraries](https://twitter.com/marksaroufim/status/1785462414852714954) 的最佳实践，揭示了性能提升，但发现一个原本用于存放幻灯片的 Google Drive 链接是空的。他们进一步讨论了准确的 CUDA kernel profiling 技术，相比 `cudaEventRecord`，更倾向于使用 NVIDIA 的工具如 `nsight compute` 和 `nsight system`，因为它们开销更小且 profiling 功能更强大。

- **Triton 解决 Block Size 和调试问题**：在 Triton 领域，工程师们澄清了 **Triton 的最大 block size** 不受 CUDA 等硬件限制，并建议利用 [Triton 调试讲座](https://www.youtube.com/watch?v=DdTsX6DQk24) 进行调试。频道还提到使用 `triton-nightly` 以受益于最近的解释器 bug 修复。

- **稀疏性算法引发基准测试与学习讨论**：AI 爱好者讨论了一种在 batch size 为 1 时利用激活稀疏性（activation sparsity）的算法，该算法的作者也参与了讨论，并承诺分享新的基准测试以及关于该方法与量化方法相比在速度/质量权衡方面的见解。

- **CUDA 中的 Stride 对齐与 Kernel 优化**：关于 tensor stride 对齐和 kernel 优化策略（如 *matmul_backward_bias*）的讨论主导了 `#llmdotc` 频道。会上辩论了使用 *x128 packing* 策略、实验 CUDA Graphs、cuDNN Flash Attention 优化以及为 master weights 引入 FP32 所带来的性能提升，展示了对更高效 CUDA 编程的追求。

- **AMD 的 ROCm 与 Torch Nightly 讨论**：专注于 AMD ROCm 平台的用户交流了对 torch Nightly（而非 Torch 2.3）的偏好，质疑了 AMD 分支中缺少最新 **version 2.0 of flash attention** 的问题，并分享了为 AMD Flash Attention 添加反向传播（backward pass）的消息，引发了信息丰富的交流，并提供了一个 [AMD HIP 教程资源](https://www.youtube.com/playlist?list=PLB1fSi1mbw6IKbZSPz9a2r2DbnHWnLbF-)。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**GPU 效率引起关注**：A4000 16GB GPU 因其训练效率受到称赞，与 A100 相比，其性价比获得了好评。B200 因其潜力被吹捧，预计在成本相当的情况下，效率将比目前的 H100 高出 25 倍。

**技术探讨**：关于采用 LoRA 还是 QLoRA 的辩论表明，QLoRA 可能会减少 75% 的 VRAM 使用量，但代价可能是损失 1-2% 的模型准确度。建议对训练数据进行 80-10-10 的划分以确保模型的鲁棒性，同时语言模型微调正在取得进展，证明了其在土耳其语翻译中的应用。

**模型训练创新**：用户报告了 `llama.cpp` 的量化问题，导致了 GitHub issue，如 [#3759](https://github.com/ollama/ollama/issues/3759) 和 [#4180](https://github.com/vllm-project/vllm/issues/4180)。微调和训练的工作流程是一个澄清点，提出了 checkpointing 策略和推理提供商（如 Jan 和 GPT4All），可在 [janhq/jan](https://github.com/janhq/jan) 等仓库中获取。

**提议 AI 开发路线图**：简单明了的 AI 项目路线图的倡导者强调了其重要性，同时正在探索小型模型增强对话能力的潜力。此外，检索增强（retrieval augmentation）的概念正受到关注，并参考了 [FlagEmbedding 的 GitHub 仓库](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora) 等实现。

**尺寸与性能**：值得注意的一点是，Phi3 Mini 4k 在 Open LLM 排行榜上的表现优于较大的 128k 版本，这促使人们重新评估模型尺寸的有效性。人们倾向于选择 Phi3 Mini 4k 等模型，因为它们比大型对应模型更高效。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **亮眼的性能优化**：*Flash Attention* 集成到 `llama.cpp` 中，通过将复杂度从 O(N^2) 降低到 O(N) 提升了内存效率，合并后的 PR [Flash ATTENTION support merged into llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5021) 引发了社区的热烈反响。
  
- **模型限制带来的通用性问题**：热烈的讨论揭示了模型在超出设计限制使用时面临的兼容性障碍，例如 *Llama 3* 在旧版本上运行不佳，且在 context 超过 250,000 tokens 时会报错，尽管有人尝试在 36GB VRAM 上实现 1M token 的窗口。

- **充足硬件的必要性**：讨论串一致认为，有效使用 LLM 需要相当大的系统资源，像 Everything 7b q4 这样的模型在仅有 8 GB RAM 的设备上会变得非常缓慢，而更新后的 llama.cpp 分词器错误也暗示了对 RAM 的巨大需求。

- **ROCm 构建障碍**：AMD 用户就 **ROCm 和 OpenCL** 的集成展开了交流，有报告称在 7900xtx 上会出现 **VRAM 容量** 误读的情况（尽管之前使用 RX 6600 时正常），并建议选择 7900XTX 而非 7900 GRE 以确保 LM Studio 的兼容性。

- **追求最新的模型和软件版本**：即将发布的 LM Studio 0.2.22 引起了关注，旨在修复分词器问题并提升模型性能，同时建议使用 `llama.cpp` 的 beta 版本来解决社区反馈的问题。

若要了解技术进展和修复的更新，建议社区关注相关 GitHub 仓库和发布页面，以获取最新的 commit 和构建更新。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **突破 OOD 障碍**：针对位置分布外（OOD）问题提出了一种解决方案，旨在帮助大语言模型泛化到更长的 context，详见[最近发表的论文](https://arxiv.org/pdf/2401.01325)。在 [`llama.cpp` 仓库](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)中可以找到使用 `--grp-attn-n` 和 `--grp-attn-w` 参数的实现示例。

- **Llama-3 的飞跃**：Nous Research 推出了 **Hermes 2 Pro on Llama-3 8B**，重点提升了 Function Calling 和 Structured Output 能力，并在主流基准测试中超越了 Llama-3 8B Instruct。针对效率优化且不牺牲先进性的量化版本也已在 [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF) 上线。

- **LLM 性能与实用性**：讨论指出，**每权重 5.5 bits (bpw)** 的量化是大语言模型性能出现显著损失前的临界点。新的 **Hermes 2 Pro Llama 3** 在获得 Function Calling 等新技能的同时，也“遗忘”了某些特定任务，社区正在探索长 context 长度的优化以及高级分词机制的集成。

- **AI 创新的数据集与工具**：一个新的 **Wikipedia RAG 数据集** 已经发布，与之配套的还有一项关于利用 LLM 合成多语言训练数据的研究，详见[此处](https://huggingface.co/collections/nthakur/swim-ir-dataset-662ddaecfc20896bf14dd9b7)。此外，讨论还涉及了在 Cynde 重构中集成 Pydantic，以及引入 Logfire（一个因简化代码可观测性而受到赞誉的平台），详情见[此处](https://pydantic.dev/logfire)。

- **虚拟仿真进展**：社区见证了商业和音乐**行业模拟器** CompSimulator 和 Snow Singer Simulator 的发布，旨在提供沉浸式的 AI 驱动体验。此外，来自 AGI House SF 的演讲促成了社区聚会计划，并注意到 HF Chat 上的 LLAMA 3 机器人对于相同消息会产生一致的响应。



---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3 发布期待落空**：由于 4 月或 5 月发布的预期未能兑现，对 **Stable Diffusion 3 (SD3)** 发布的怀疑情绪蔓延；人们担心 **Stability AI** 可能会因为关于 SD3 免费且开源的声明而面临批评。
- **本地界面阵容评估**：AI 爱好者正在对比 **ComfyUI**、**AUTO11111**、**Focus** 和 **Forge** 等 Stable Diffusion 本地界面，建议主要取决于用户友好度以及特定的硬件需求（如 NVIDIA 或 AMD GPU 兼容性）。
- **AI 辅助提示词工程**：关于有效图像描述提示词的最佳工具存在持续辩论，提到了 **ChatGPT**、**Gemini**、**Claude 3** 和 **idefics2**；这些工具对于优化提示词以增强图像生成效果具有潜在价值。
- **AI 服务与隐私工具**：讨论显示了投资 **Gemini** 和 **Claude 3** 等 AI 服务的趋势，并结合战略性使用 **VPN** 技术（包括 **DNS over HTTPS**）来绕过地区限制或保持用户匿名。
- **Automatic1111 粉丝的扩展讨论**：出现了关于使用 **Automatic1111 extension** 在图像中嵌入标签的能力，以及在 **ComfyUI** 等自定义界面中是否存在类似于 **clip skip** 和 **stylizer** 功能的疑问。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **聊天控制升级**：OpenAI 为 ChatGPT Free 和 Plus 推出了**更新的数据控制**功能，允许用户在选择不将数据用于训练的同时查看聊天历史。他们还引入了**临时聊天 (Temporary Chat)** 功能，用于不保留聊天历史的一次性会话。
- **GPT-2 在聊天机器人中的复兴**：成员们正在探索 **gpt2-chatbot**，反馈褒贬不一；它在某些场景下表现出色，但也被指出偶尔会失败。尽管有访问问题的报告，但其无限生成的能力引起了人们的兴趣。
- **剖析 AI 情感智能**：关于 AI 发展情感潜力的深入讨论与人类发展进行了类比。重点在于 AI 系统中实现同理心理解或类似情感反应是否可行或可取。
- **DALL-E 免费层级功能辩论**：用户一直在讨论 OpenAI 为免费用户提供的 DALL-E 等服务，在商业可持续性与扩展用户功能之间寻找平衡。
- **利用正向提示词结果**：AI 工程师正在探索高效的 Prompt Engineering，重点关注正向提示 (positive prompting) 和元提示 (meta-prompting)，以实现与 AI 模型更有效的交互，建议使用诸如 *“用 'y' 代替 'x'”* 之类的策略来优化输出质量。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Pages 功能准备 Beta 测试**：**Perplexity AI** 宣布即将推出名为 **Pages** 的功能，旨在对主题进行可分享的深度探索；感兴趣的用户可以获得 Beta 版本的早期访问权限。

**API 引用缺失问题**：工程师们表达了在使用 **Perplexity-online models** 时通过 API 请求访问引用的担忧，并讨论了 Pro UI 与 API 模型结果之间的差异；**[API 文档](https://docs.perplexity.ai/docs/model-cards)** 被明确为获取模型详情的首选资源。

**限制与故障成为焦点**：成员们讨论了 Opus 每天 **50 次的使用限制**、Pro Search 和引用工具中的故障，以及 AI 模型响应缓慢的问题，并针对登录问题提供了关于服务商可能存在的邮件过滤的技术建议。

**通过分享内容进行探索**：用户积极分享了关于各种主题的见解和链接，包括 **Microsoft Research Asia**、**Vimeo API** 和 **Tesla** 的自动驾驶技术；此外，一份分享的 [新闻通讯](https://www.lennysnewsletter.com/p/how-perplexity-builds-product) 提供了了解产品开发见解的窗口。

**Claude 3 政策与模型利用说明**：关于 **Claude 3** 使用政策的咨询引发了关于是适用 Perplexity 还是 **Anthropic** 政策的讨论，同时解释了 Pro UI 中在线模型的使用方式，即要么经过微调，要么采用搜索引擎式的向量数据库 (vector database) 来生成响应。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **利用 Effort/BucketMul 加速推理**：引入了一种名为 **effort/bucketMul** 的新算法，旨在显著加速向量-矩阵近似和大语言模型 (LLM) 推理，有望实现实时计算负载调整，并兼容 Mistral 等模型。更多详情可以在[这里](http://kolinko.github.io/effort/)找到。

- **二进制在嵌入效率上超越超球体**：关于嵌入策略的讨论揭示了二进制向量表示在嵌入效率方面的优势，这得到了生物学合理性和计算节俭性的支持，并与 RWKV LLM 建立了联系，应用这些原理可能有助于加快学习速度。如需深入了解，请阅读关于 [RWKV LLM](https://github.com/BlinkDL/SmallInitEmb) 以及 [CLIP](https://arxiv.org/abs/2103.00020) 和 [Dino](https://arxiv.org/abs/2104.14294) 等开创性嵌入工作的资料。

- **揭秘黑盒并改进基准测试**：围绕 LLM 不透明性的对话指出了其复杂性与人类理解力之间的差距，重点在于通过避免在基准测试集上训练 LLM 来提高基准测试 (benchmark) 比较的公平性。请参阅关于[基准测试数据集](http://arxiv.org/abs/2404.18824)中偏差的讨论。

- **KANs 领先于 MLPs**：新兴研究引入了 **Kolmogorov-Arnold Networks (KANs)**，在准确性和可解释性方面凭借高效的缩放定律 (scaling laws) 超越了多层感知器 (MLPs)。关于 KANs 的关键论文见[此处](http://arxiv.org/abs/2404.19756)。

- **力求透明的 LLM 计算**：一位成员的阐述对序列预测模型中的计算模型进行了理论化，讨论了绑定嵌入 (tied embeddings) 如何影响可解释性，并思考了验证其假设的实验方法。必读内容包括 [Deriving a Model of Computation for Next-Token Prediction](https://docs.google.com/document/d/11w3of15CbfOlWrvQpTjxaJt-UvtOckzr0WQUfTrTnsw/edit?usp=sharing) 以及关于 [tuned lens method](https://arxiv.org/abs/2303.08112) 和 [distributional simplicity bias](https://arxiv.org/abs/2402.04362) 概念的论文。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **CVPR 参赛现金奖励**：HuggingFace 宣布了 [CVPR 竞赛](https://huggingface.co/spaces/BVRA/SnakeCLEF2024)，总奖金池超过 **$120,000**，包括定于 2024 年 6 月 17 日至 21 日举行的 SnakeCLEF、FungiCLEF 和 PlantCLEF 等竞赛。

- **Transformers 和 Gradio 升级**：*Transformers* 库的一次重大更新引入了新模型，其中 [Phi-3 现在可以在浏览器中运行](https://github.com/huggingface/transformers/releases/tag/v4.40.0)。Gradio 也发布了 [v4.28.0](https://www.gradio.app/changelog)，具有自定义组件功能；同时 Datasets 库也发布了并行更新，达到 v2.19.0 版本并实现了与 *Polars* 的兼容。

- **值得尝试的 AI 工具**：分享了新的 AI 工具和方法，包括一篇关于 ["每个人都应该尝试的 5 个有趣的 AI 工具"](https://medium.com/illumination/genai-adventures-5-interesting-ai-tools-everyone-should-try-44ae8f8115af) 的 Medium 文章，以及根据 Hugging Face [文档](https://huggingface.co/docs/diffusers/tutorials/fast_diffusion) 建议的关于在 PyTorch 2 中加速扩散模型的讨论。

- **Med-Gemini：医学 AI 介绍**：一段 [YouTube 视频](https://youtu.be/xohuoN2WBZs?si=Ku6cztykld6dZLN9) 深入介绍了 Google 的 **Med-Gemini**，这是一款专为医学应用设计的多模态 GenAI 模型，旨在促进对该类模型范围和潜力的理解。

- **工作机会与社区见解**：一位拥有丰富经验的软件工程师询问了 Hugging Face 的工作机会，并被引导至现有的 [职位页面](https://apply.workable.com/huggingface/?lng=en)。同时，社区交流还包括关于 Rasa 聊天机器人框架的意图识别问题、PyTorch 与 TensorFlow 之间的学习曲线，以及为 LLM 微调创建指令数据集的讨论。

- **Gradio 状态检查点**：Gradio 的 Share Server 出现问题，影响了在 Colab 上的使用；他们提供了一个 [状态页面](https://status.gradio.app/) 以跟踪修复进度。

- **AI 社区的创新**：社区成员的贡献包括用于无泄漏链接预测的 [PnPR-GCN 技术](https://github.com/Lama-West/PnPR-GCN_ACM_SAC_24) 项目和 **HDR 成像挑战**，阐述了解决方案并参与了关于 AI 进展的更广泛讨论。

- **精益学习方法**：在阅读小组中，注意力转向了图神经网络等主题（[arXiv:2404.14928](https://arxiv.org/abs/2404.14928)），以及在 [arXiv:2402.05863](https://arxiv.org/abs/2402.05863) 分享的 *NegotiationArena* 中提到的将谈判作为评估 LLM 对齐的指标的应用。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RTX 4080：对于小语言模型够用吗？**：工程师们讨论了像 **RTX 4080** 这样的游戏显卡是否适合运行和微调较小的语言模型，指出了 VRAM 的重要性，但也暗示了在小 Batch Size 下微调大于 7B 模型时的局限性。

- **本地 AI 处理重视安全性**：对话强调了本地 PC 在处理 **敏感数据** 和执行强大计算任务方面优于 Google Colab 等云解决方案的优势，后者可能会引发隐私担忧。

- **引入用于 AI 语言管理的 Word Loom**：引入了一种名为 **Word Loom** 的新开放规范，旨在高效管理和交换 AI 语言，目标是实现代码与自然语言的清晰分离以及更好的组合性，详细信息可在 [GitHub](https://gist.github.com/uogbuji/5bd08f74125934fa9e0d37236a8e168e) 上找到。

- **AI 金融天才无需人工协助即可工作**：最近的一条 [推文](https://t.co/6cTNxUBJcr) 强调，一款突破性的金融助手现在拥有在非结构化财务报告中自主 **计算百分比演变、CAGR 和 P/E 比率** 的能力。

- **LlamaIndex 获得新技术能力**：正如 [推文](https://t.co/mBIrD9uh8c) 中宣布的那样，最新发布的 **LlamaIndex.TS 0.3 版本** 带来了重大改进，包括对各种平台的 Agent 支持、Web Streams 增强以及更具弹性的类型系统。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 稳步前行**：Mojo 开发者社区庆祝了 Mojo 发布**一周年**，称赞了 traits、references 和 lifetimes 的加入，这些特性显著增强了标准库。关于增强功能，有人建议通过允许**负数**以及实现标量处理的 fallback 来改进 Mojo，灵感源自 issues 中链接的文章。

**性能大提升**：Mojo 中字符串分配和转换的创新优化将 100M 记录的处理时间从 18.5 秒缩短至 12.5 秒，最新的努力利用多核处理技术将其进一步缩短至 3.8 秒。社区发起了组建 **Team-Mojo** 参加 *One Billion Row Challenge* 的号召，将其视为展示和社区协作的机会。

**语法与语义的协同**：关于语法和语义的讨论强调了 Mojo 语法对齐对用户的重要性，以及 Mojo 中的 `inout` 与 C++ 中的 pass-by-reference 的相似之处及其细微差别。关于 `__source_location()` 函数的问题引发了关于在其输出中包含 `function_name` 以及在 nightly 分支中替换这些功能的讨论。

**探索并发考量**：对话推测了 Mojo 并发模型的潜力，理论上它可能比 golang 风格更接近 actor 模型，重点在于避免沉重的 runtime 引入。拥有 LLVM 后盾的 Mojo 编译器有一个[专门的 YouTube 视频](https://youtu.be/SEwTjZvy8vw)解释其底层原理。

**推文预热引发猜测**：Modular 通过一系列未指明的推文激发了好奇心，预告了有趣的进展但未透露具体细节，激起了人们对公告之外细节的兴趣。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**探索 Axolotl 的模型支持**：在 **#axolotl-phorm-bot** 频道的讨论中，明确了 Axolotl 支持 GaLore 但不支持 phi-3 格式。社区建议查看 [Hugging Face 文档](https://github.com/huggingface/transformers/tree/main/docs/source/en/trainer.md)以获取开启 GaLore 的详细信息。同时，一个[未经测试的 PR](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547) 被强调为那些希望向 Axolotl 添加 command-r 模型的人的参考资源。

**有效 Chat-Tokenization 的策略**：**#general** 频道的成员讨论了 Beginning of Sentence (BOS) token 在 tokenizer 行为中的影响，以及在不同场景中正确指定它的重要性。此外，一项关于 [generalist foundation models 的研究](https://arxiv.org/abs/2311.16452)引发了关于复杂 prompting 策略的有效性以及将学术理论转化为实践的挑战的讨论。

**新模型微调的最佳实践**：**#general-help** 频道非常活跃，社区参与了微调过程，建议初学者使用较小的模型（如 8b 模型）。讨论了 ShareGPT loader 数据集转换的实用技巧，以及关于 fsdp 与 lora 兼容性的查询。

**教程协作引起共鸣**：在 **#community-showcase** 中，分享了一个展示 axolotl 与开源容器编排器 dstack 结合的教程并广受好评，强调了易用性和灵活性。贡献者可前往 [GitHub 查看详细用法](https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md)。

**协作计算资源**：**#axolotl-dev** 频道提供了一项提议，向其他成员提供计算资源，以帮助进行 triage 和故障排除，这对于参与 bug 修复和功能增强的人员特别有用。

## [LAION](https://discord.com/channels/823813159592001537) Discord

**AI 进入 TOS 灰色地带**：围绕用户在**不同意服务条款（TOS）的情况下使用 AI 产品**展开了讨论，突显了用户协议执行中的灰色地带，并引发了关于用户和提供商法律影响的辩论。

**排行榜诚信受到挑战**：人们正在推动建立一个更透明的 **AI 模型排行榜**，强调开放性和可验证性的必要性，同时成员们对 **LMSYS 的 Chatbot Arena** 表示怀疑，担心其缺乏客观性和数据实践不透明。提出了仅纳入**开源模型**并按**开放权重（open weights）**进行过滤的观点，作为改进排行榜的标准。

**渴望效率**：工程讨论围绕多种优化策略展开，从考虑使用 **GANs 进行卓越的模型重建**，到关于 **Natten 的 CUDA 实现**的讨论，以及 [magvit2](https://github.com/lucidrains/magvit2-pytorch) 等项目的开发。

**在 AI 和医学领域取得新突破**：社区关注了一项关于**利用 OpenCLIP 进行心脏超声**的研究，该研究最近发表在 [Nature Medicine](https://doi.org/10.1038/s41591-024-02959-y) 上，尽管该研究目前还存在一些问题。

**变革网络与事实核查**：人们对创新的 **Kolmogorov-Arnold Networks (KANs)** 表现出极大的热情，该网络有望在准确性和可解释性方面超越 MLPs（[关于 KAN 的论文](https://arxiv.org/abs/2404.19756)），此外还介绍了 **VisualFactChecker**，这是一个无需训练的流水线，旨在增强视觉内容描述的忠实度（[关于 VFC 的论文](https://arxiv.org/abs/2404.19752)）。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**去中心化 AI 算力**：Prime Intellect 投入到了**去中心化 AI 训练方法**的探索中，旨在与大型企业使用的庞大 **GPU 集群**竞争。他们的平台致力于利用全球分布的计算资源，详见其详尽的[博客文章](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training)。

**StarCoder 崛起**：Hugging Face 发布了一个名为 **StarCoder2-15B-Instruct-v0.1 的新 LLM**，主要专注于**代码生成**。他们已将模型和流水线开源，邀请社区参与，如其[发布页面](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)所述。

**在消费级硬件上模拟 AI 社会**：据报道，一个涉及 300 个名为 AI Town 的 AI Agents 的实验设置在 MacBook M1 Max 上运行顺畅。这篇有趣的 [推文](https://x.com/cocktailpeanut/status/1785702250599371088?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 揭示了在消费级硬件上进行 AI 模拟的能力和潜力。

**LLM 论文俱乐部：Ring 讨论**：**LLM 论文俱乐部**即将举行的活动将与 StrongCompute 团队就 **Ring Attention** 论文进行协作讨论。对最新研究成果感兴趣的工程师可以通过此 [Zoom 链接](https://lu.ma/oz8e9z3r)加入。

**技术精英视频会议**：已安排了一次 **Zoom 视频会议**，以便进行更直观的互动讨论，可能涉及正在进行的工作或论文俱乐部活动。社区成员可以使用提供的 [Zoom 会议链接](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09)加入。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**尊重是技术最好的朋友**：社区提醒强调了**尊重和建设性互动**的必要性；随着群组的扩大，让每个人都感到被欢迎和重视，对于协作的未来至关重要。

**Open Interpreter 变得精通浏览器**：**Open Interpreter** 工具被确认具备**网页浏览和数据抓取任务**的能力，无需传统的浏览器控制，通过 AI 实现直接的网页交互。

**通过 DIY 扬声器放大器达到理想效果**：为了提升扬声器的音频输出，推荐的一种解决方案是使用**外部放大器**，并重点介绍了 [Amazon](https://www.amazon.com/dp/B01DKAI51M) 上的一款潜在放大器，不过实际应用效果仍有待测试确认。

**R1 的 AI 开箱引发集成讨论**：一段关于 AI 产品 Rabbit R1 的 **MKBHD YouTube 评测**（[在此观看](https://www.youtube.com/watch?v=ddTV12hErTc&ab_channel=MarquesBrownlee)）引发了关于其与 **OpenInterpreter** 集成潜力的讨论，工程师们渴望突破互连 AI 系统的极限。

**为成功连接 OI 建立隧道**：工程师们交流了与 OpenInterpreter 服务器建立稳定连接的诀窍，包括使用 **ngrok** 设置新域的方法以及修改 **tunnel.py** 文件，旨在解决连接中的小问题——更多详情请参阅 [ngrok 域名页面](https://dashboard.ngrok.com/cloud-edge/domains)。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **新 AI 模型登场**：**Snowflake Arctic 480B** 和 **FireLLaVA 13B** 已发布。**Snowflake Arctic 480B** 拥有针对编程优化的混合 Transformer 架构，可在 [Snowflake Arctic 480B](https://openrouter.ai/models/snowflake/snowflake-arctic-instruct) 获取；**FireLLaVA 13B** 是来自 Fireworks 的多模态模型，可在 [FireLLaVA 13B](https://openrouter.ai/models/fireworks/firellava-13b) 获取。价格和开发者规范已更新以反映其增强的能力。

- **OpenRouter 通过高效负载处理变得更智能**：新的**负载均衡**功能旨在更有效地分配提供商的工作负载，并辅以 [Activity 页面](https://openrouter.ai/activity) 上的延迟和提供商性能实时监控工具，提升了系统的整体鲁棒性。

- **为开发者提供精简资源**：**OpenRouter 的文档**现已更新，支持更高效地使用图像和多模态请求、定制化工具调用（tool calls）以及函数调用（function calling）；详情请参阅 [图像请求](https://openrouter.ai/docs#images-_-multimodal-requests) 和 [工具调用](https://openrouter.ai/docs#tool-calls)。

- **AI 服务成本降低**：**OpenRouter** 大幅降价：Mythomax Extended 服务大幅降价 40%，Mixtral 8x7b Instruct 也小幅降价 4%，体现了平台致力于提供负担得起的 AI 服务的承诺。

- **AI 创作带有瑞典风格**：**Skribler** 是一款旨在通过整合不同 AI 模型协助瑞典作者进行各方面写作的工具，其用户群正在增长，且已有用户愿意为其服务付费——请访问 [skribler.se](https://skribler.se) 查看。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**清晰的视觉效果引发关注**：[Hexagen World](https://www.hexagen.world/) 以其**高质量的扩散模型（diffusion model）输出**让成员们感到惊喜，这为交互式 AI 游戏开发提供了充满希望的方向。

**用 AI 重塑复古游戏**：公会讨论了使用生成式 AI（Generative AI）复活像 **Farmville** 这样的复古游戏，WebSim 被视为这些怀旧重启作品的潜在平台。

**间谍游戏遇上生成式城镇**：一个关于 1950 年代主题、带有**共产主义间谍**角色的 **AI 城镇**的有趣概念被提出，引发了在 WebSim 中创建一个沉浸式**猫鼠游戏**的兴趣。

**加入 AI 动画对话**：对 AI 驱动的动画感兴趣的人收到了加入专门 Discord 小组的邀请（通过[社区链接](https://discord.gg/deforum)），为交互式 AI 领域的协作讨论和项目提供了空间。

**开发者讨论凸显兼容性问题**：AI 开发者们解决了本地设置过程中的问题，特别指出了 Windows 系统的问题以及使用正确 **Node 版本**（`nvm use 19`）的重要性。一些人甚至考虑转向 Linux，尤其是考虑到像《群星》（Stellaris）这样的游戏也得到了支持，正如在 [WineHQ](https://appdb.winehq.org/objectManager.php?sClass=application&iId=17537) 上找到的信息所证明的那样。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Command R 表现惊艳**：Cohere 社区对 **CommandR/R+ models** 表示赞赏，强调其出色的性能，在企业级体验方面似乎超越了其他大型语言模型。

**LLM 语法秘籍揭晓**：关于 **LLMs (Large Language Models)** 及其生成语法正确句子能力的讨论揭示了对单词和句子嵌入（embeddings）以及自注意力机制（self-attention mechanism）重要性的见解，并提供了[深入理解的资源](https://docs.cohere.com/docs/the-attention-mechanism)。

**AI 法律专家起航**：一场关于使用 **Cohere's RAG** 构建 AI 法律助手的网络研讨会吸引了社区参与，[YouTube](https://www.youtube.com/watch?v=KfqJsqIFeRY&ab_channel=Cohere) 上提供了录像链接。

**Azure 遇上 OAuth**：明确了在 Azure 上使用 Cohere 工具包设置带有连接器（connectors）的 OAuth 的说明，强调了在保持数据内部化的同时进行 Azure 集成的能力，详见其 [GitHub 页面](https://github.com/cohere-ai/cohere-toolkit/?tab=readme-ov-file#how-to-add-a-connector-to-the-toolkit)。

**多语言精通正在成型**：社区正在积极评估 Command-R 中**多语言支持（multilingual support）**的实现和潜力，特别关注挪威语等语言，并渴望增强基准测试（benchmarks）。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**PDF 表格提取被证明很棘手**：工程师们分享了使用 *unstructure* 库从 PDF 中进行**表格提取（table extraction）**的挑战，指出效果不佳，特别是对于多页表格。目前尚未提供解决方案，表明这是一个值得开发或推荐工具的领域。

**LangChain 与 Llama 3 联手**：有一场关于将 **Llama 3** 与 **LangChain** 集成的对话，引导用户使用 [Fireworks](https://python.langchain.com/docs/integrations/chat/fireworks/) 和相应的 API keys。此外，还提到了在一个项目中重新引入 **Google Drive libraries**，突显了技术依赖的周期性。

**发布、更新与规范介绍**：值得关注的进展包括用于总结 YouTube 内容的 [QuickVid](https://quickvid.vercel.app/) 的发布，LangChain 聊天机器人更新至 **0.1.17**，以及引入 **Word Loom** 作为 AI 语言管理的潜在标准，并在其 [GitHub Gist](https://gist.github.com/uogbuji/5bd08f74125934fa9e0d37236a8e168e) 征求反馈。还提出了关于比较各种用于内容创作的 **LLMs** 详细性能报告有用性的疑问。

**知识图谱愿景与 AI 销售代理**：成员们分享了关于将文档转换为**知识图谱（knowledge graphs）**的工具以及开发 **AI-powered Sales Agents** 的见解。对于前者，提议使用布局解析器（layout parsers）和 Azure Doc AI，同时探索 LangChain 文档中的图构建方法。后者涉及 SalesGPT 逻辑并呼吁建立合作伙伴关系。

**RAG 创新与语言导向教程**：工程师们讨论了各种 RAG 应用，包括为法语社区开发的**高级 RAG 助手（Advanced RAG assistant）**、Llama3 的本地训练，以及一种根据查询复杂度进行响应的**自适应 RAG 技术（Adaptive RAG technique）**。分享了相关的教学视频：[法语 RAG 助手](https://youtu.be/ol2QMp64lgo)、[基于 llama3 的本地 Agentic RAG](https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P) 以及 [LangGraph + Adaptive Rag + LLama3 Python 项目](https://www.youtube.com/watch?v=QnXdlqEUW80)。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Mozilla AI 正在招聘，向 Lm-buddy 招手**：Mozilla AI 目前正在扩大团队，并在其官方 Discord 频道发布了职位机会，同时还发布了 **Lm-buddy**，这是一个旨在提高模型评估效率的新开源工具。

**确认在 M1 MacBook Air 上测试 LLaMA3:8b**：在用户遇到 **LLaMA3:8b** 在 M1 MacBook Air 上运行的问题后，官方回应表示，一旦其他支持问题得到解决，在 M1 上的测试将成为优先级。

**将 Whisper 集成至 Llamafile**：尽管在添加麦克风和扬声器功能方面存在挑战，但已有提议将 **whisper.cpp 模型** 集成到 llamafile 中以增强推理能力。

**性能争论得到澄清**：Justine Tunney 的一篇文章暗示 **np.matmul** 的性能达到 29 gflops，这引发了争议，随后澄清该数据是针对 Ubuntu 上的 Intel 计算机的特定结果，实际性能可能会有所不同。

**同时运行 Llamafile 及路径自定义说明**：频道内的讨论确认，可以同时运行多个加载不同模型的 llamafile，由操作系统管理资源。用户还了解到，使用 `--server --path PUBLIC_PATH` 选项进行的自定义仅限于替换 zip 文件中的 .html 和 .js 文件。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Tinygrad 经历 Tensor 变换**：[tinygrad 项目](https://github.com/tinygrad/tinygrad) 实施了重大更新，通过一个 [将 `Scalar` 重命名为 `ConstType` 的 commit](https://github.com/tinygrad/tinygrad/commit/77589bc7a5430ee470621e43fb1817259d3ce0f5) 推进了代码库的标准化。讨论重点在于通过引入 const 支持变量来优化操作中的常量处理，以及 const Variables 对于与符号维度（symbolic dimensions）相关的操作的重要性。

**反向传播的图可视化引起关注**：对话中表现出对反向操作图表可视化的好奇，重点关注 issue **#3572**。有迹象表明可以使用 dot 文件并设置 `GRAPH=1` 来辅助理解这些操作。

**符号维度成为焦点**：Georgehotz 分享了关于符号形状（symbolic shapes）工作的见解，并提交了一个 [包含符号 arange 跳过测试的 pull request](https://github.com/tinygrad/tinygrad/pull/4362)。这表明 tinygrad 正在持续努力增强其在符号维度方面的能力。

**JIT 构建与均值计算**：关于改进 tinygrad 带有符号变量的 Just-In-Time (JIT) 编译的对话指出，一个稳健的测试将涉及计算可变长度 2D Tensor 的均值。此类增强功能可以优化 JIT 编译器的效率和性能。

**Nvidia Xavier 上的 CUDA 挑战**：技术讨论涉及在 Nvidia Xavier 上运行 EfficientNet 示例时面临的挑战，强调需要确保 `CUDA=1` 以正确执行脚本。成员们还讨论了 tinygrad 中的 Rednode 表示是否可能使符号编译器逻辑复杂化。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Claude 加入 AI 聊天应用领域**：Anthropic 发布了其 Claude 应用，引发了成员对其 **与 OpenAI 解决方案相比的性能** 的好奇。虽然没有提供详细对比，但一位用户下载了该应用并报告了流畅的初步体验，特别赞赏了 Anthropic 的品牌设计。

- **通过反馈提升表现**：在收到尖锐反馈后，一位成员显著提高了工作质量，并得到了同行的表扬。虽然未给出工作改进的具体细节，但这种反应式的生产力提升值得注意。

- **AI 排行榜受到审查**：一篇文章指出 AI 排行榜可能已经过时，强调根据 HumanEval 基准测试，**最准确的代码生成系统** 是 LDB。然而，它 **对 GPT-4 等昂贵模型调用的依赖** 给其效率和成本效益蒙上了阴影。

- **ML Collective 出勤情况**：某位成员确认 **ML Collective** 会议的出勤率稀疏，表示虽然在持续参与，但未讨论会议的具体成果或细节。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **整个社区的垃圾信息警报**：Discord 社区内的多个频道遭到不当内容的入侵，这些内容宣传涉及潜在未成年对象的成人材料，并附带据称提供泄露内容的 Discord 邀请链接。
- **迫切的审核需求**：这些消息违反了社区准则，暗示了非法活动，并无视了技术讨论中应有的专业对话目的。
- **不受欢迎的干扰**：垃圾信息干扰了从 AI 讨论到协作和通用聊天的众多频道，需要管理员引起注意。
- **工程师内容警示**：工程师必须保持警惕，因为垃圾信息包含潜在的安全风险（如网络钓鱼尝试），可能会损害专业和个人数据。
- **行动呼吁**：建议立即采取行动删除内容、封禁发布者，并加强安全措施以防止未来发生类似事件。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Prompt Engineering 推动 LLaMA-3**：**LLaMA-3 instruct prompt 策略**已更新，带来了性能提升，相关更改详见 GitHub [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553)。
- **缓解数据集困扰**：正确使用 `eot_id` 解决了与数据集条目格式化相关的挑战，事实证明这比手动添加 `</s>` 标签更有效。
- **Meta 利用迭代推理**：新的“迭代推理偏好优化 (Iterative Reasoning Preference Optimization)”技术提升了 **LLaMA-2-70B-Chat** 的准确性，在 GSM8K 和 ARC-Challenge 基准测试中得分的提高证明了这一点；论文可以在[这里](https://arxiv.org/abs/2404.19733)阅读。
- **Axolotl 微调成功**：一位用户分享了使用 **Axolotl 微调 LLaMA-3 8b** 的成功经验，并指出模型输出有所增强。
- **开启编程狂欢**：分享了一首励志动漫曲目 "NEVER GIVE UP YOUR WAAAAAAAAAAAAY"，可能旨在为深夜编程环节加油，并附带了 [YouTube 链接](https://youtu.be/tYzMYcUty6s?si=t2utqcq36PHbk9da)和对创作者的 Patreon 支持说明。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**LLaMA 在语言对决中击败 GPT-4**：来自 [scandeval.com](https://scandeval.com/german-nlg/) 的结果表明，**LLaMA 3** 在德语自然语言任务的 ScandEval 基准测试中表现优于 **GPT-4**，引发了关于新 AI 模型能力的讨论。

**本地加速加载胜过迟缓的云端**：一位工程师报告称，一个程序在本地机器上*只需 3 秒即可加载*，这表明在其他地方运行任务时，加载速度较慢是由存储以外的问题造成的。

**Qdora 扩展 LLaMA 的中间路径**：随着 **qdora** 的提及，大型语言模型 (LLM) 扩展方面出现了令人兴奋的进展，这是一种促进 LLaMA 等模型增长的解决方案；该过程在 [Answer.ai 博客文章](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html)中有所概述。

**避免 AI 训练中的遗忘**：社区讨论了在预训练后阶段防止灾难性遗忘 (catastrophic forgetting) 的方法，引用了一篇关于增强 Transformer 块的 [Arxiv 论文](https://arxiv.org/abs/2401.02415)，该论文有助于 LLM 在学习新技能的同时保留旧技能。

**融合 AI 的过去与现在**：社区参与强调了 LLM 中“非遗忘性学习 (Non-forgetful Learning)”的前景，其中扩展技术对于将传统 AI 技能与更新、更先进的能力相结合至关重要。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **设计以用户为中心的数据检索**：一位成员提议为 **Datasette** 开发一个前端功能，允许用户从下拉列表中选择特定国家的数据，目标是改善数据获取的用户体验。
- **关于 URL 与 UI 自定义的辩论**：出现了两种用户体验策略：一种是动态更新 **URL** 以在选择时显示相关数据，另一种是开发一个具有基于用户输入的“可构建”查询的自定义界面。



---

# PART 2: 频道详细摘要与链接



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1235244530857541726)** (4 条消息): 

- **分享 CUDA 最佳实践**：该频道分享了一个关于 [CUDA C++ 核心库最佳实践](https://twitter.com/marksaroufim/status/1785462414852714954)的 Twitter 链接，并通过 Google Drive [链接](https://drive.google.com/drive/folders/1T-t0d_u0Xu8w_-1E5kAwmXNfF72x-HTA)提供了幻灯片，但注意到该文件夹中没有文件。

- **对垃圾信息的快速处理**：一名用户通过提及（@&1189538650011217942）引起了版主的注意，随后另一名成员迅速采取行动，确认删除了一条垃圾帖子。

- **理解 PyTorch 的 autograd.grad**：一名成员提出了关于使用 `torch.autograd.grad` 通过两次连续的梯度计算，来获取函数输出相对于参数的 Hessian 矩阵对角线的问题。

**提到的链接**：<a href="https://drive.google.com/drive/folders/1T-t0d_u0Xu8w_-1E5kAwmXNfF72x-HTA">CCCL - Google Drive</a>：未找到描述

---

**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1234899266837938176)** (13 条消息🔥): 

- **Triton 的 Block Size 谜题**：一名成员询问 **Triton** 中的最大 block size，认为它应该与 CUDA 的限制一致。作为回应，有人解释说 Triton 的 block size 从根本上讲并不受硬件限制，理论上可以非常大，并且与每个 block 启动的线程数没有直接关系。

- **探讨 Triton 调试技术**：一位用户寻求调试 **Triton** kernel 的最佳实践，发现 `TRITON_INTERPRET=1` 和 `device_print` 存在挑战。另一名成员鼓励查看 [Triton 调试讲座](https://www.youtube.com/watch?v=DdTsX6DQk24) 以获取见解，因为它可能会提供有用的策略。

- **需要 Triton 解释器 Bug 修复**：针对调试问题，一名用户提到 `TRITON_INTERPRET=1` 设置导致程序行为异常。建议从源码安装 **Triton** 或使用 `triton-nightly` 以受益于最近的解释器 Bug 修复。

- **对 Triton 发布计划的好奇**：一名成员询问 **Triton** 下一个版本的预期发布日期，因为他们目前正在使用 2.3 版本。得到的回答是目前对于即将发布的版本尚无明确计划。

**提到的链接**：<a href="https://www.youtube.com/watch?v=DdTsX6DQk24">Lecture 14: Practitioners Guide to Triton</a>：https://github.com/cuda-mode/lectures/tree/main/lecture%2014

---

**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1234762936782819398)** (14 条消息🔥): 

- **探索 CUTLASS 与 CuBLAS**：一名成员强调了 [CUTLASS](https://github.com/NVIDIA/cutlass) 的性能，在矩阵乘法基准测试（8192 x 8192 x 8192）中，它以 288 Teraflops 的表现超越了 CuBLAS 的 258 Teraflops。然而，当集成到 Python 中时，CUTLASS 的性能优势消失了，与 CuBLAS 同样为 257 Teraflops。
- **CUDA 中的 Kernel 计时难题**：讨论围绕如何准确分析 CUDA kernel 内的时间跨度展开，因为使用 `cudaEventRecord` 显示出计时不稳定的情况，特别是在具有不同 tile 大小的矩阵乘法 kernel 的共享内存版本中。
- **用于准确分析的 NVIDIA 工具**：建议使用 NVIDIA 的 `nsight compute` 或 `nsight system` 进行更稳健的性能分析，因为它们的设计更为精确，且与使用 `cudaEventRecord` 的自定义分析相比，开销可能更小。
- **理解分析开销**：一名成员询问 `cudaEventRecord` 计时与 `ncu` 的 Duration 字段之间不一致的问题，担心 `ncu` 的报告可能包含了分析开销。回答澄清说 `ncu` 会运行预热 kernel，这可能会导致额外的报告时间，但最终建议以其准确性为准。
- **Nsight Systems 与 NCU 的用途**：澄清了 `nsys` 和 `ncu` 都可以用于分析 CUDA kernel，每种工具都为分析和理解 kernel 性能提供了不同的功能和界面。

**提到的链接**：<a href="https://www.thonking.ai/p/strangely-matrix-multiplications">Strangely, Matrix Multiplications on GPUs Run Faster When Given &quot;Predictable&quot; Data! [short]</a>：伟大的思想讨论每瓦特浮点运算次数。

---

**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1234894006043938926)** (5 条消息): 

- **稀疏性与质量的权衡**：对话围绕一种可能利用 *batch size=1 激活稀疏性* 的算法展开，该算法可能会保留计算量和质量。然而，有人担心这种方法在处理大于 1 的批处理计算时，可能会面临与激活稀疏性类似的限制。
  
- **Effort 创作者加入讨论**：上述算法的创作者加入了聊天，并愿意讨论他们关于该算法性能的研究结果。

- **基准测试启示**：创作者提供了一个更新，显示新的基准测试表明，与量化相比，*effort/bucketMul* 在速度/质量比方面表现较差，随后将发表文章详细介绍这些发现。

- **质量与剪枝同步**：尽管存在速度/质量方面的担忧，作者声称在质量退化方面，他们的方法似乎优于单纯剪枝最小权重，并承诺发布支持性图表。

- **分享直接对比**：分享了一个直接对比，强调了从矩阵中移除最低权重与跳过最不重要计算之间的区别，并提到作者正在持续学习关于 sparsity 的知识。
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1235025584296558632)** (2 条消息): 

- **对 Puzzle 9 中序列长度的困惑**：用户对 Puzzle 9 的术语表示困惑，特别是关于参数 **T** 和 **N0**。*z_i* 的公式也是困惑的焦点，因为用户不确定如何根据提供的信息对其进行解释。
- **注意到可能的描述冲突**：另一位成员承认 Puzzle 9 的题目描述中可能存在冲突信息，并分享了他们的假设，即为了解题，**N0** 等于 **T**。
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1234762607689203752)** (809 条消息🔥🔥🔥): 

- **CUDA 优化讨论升温**：CUDA MODE Discord 社区继续审查和优化各种 kernel 操作。成员们正在尝试对齐 tensor strides 并优化 *matmul_backward_bias* kernel，着眼于未来使用 *x128* packing 进行增强以提升性能。针对 *gradient clipping* 和 *adam optimizer* kernel 提出了多次迭代，考虑了它们对计算效率和内存使用的影响。
- **CUDA Graphs 和 cuDNN Flash Attention 投入使用**：频道贡献者已成功集成了对 cuDNN flash attention 的可选支持，看到了显著的速度提升，尽管相对于目前定制 kernel 的确切性能增益仍处于评估中。CUDA graphs 被提及作为一种优化机制，但需要更多细节来了解它们在社区代码库中的当前使用状态。
- **Comparing PyTorch and llm.c Performance**: 最近的讨论和基准测试表明，*llm.c* 在 GPT-2 模型训练方面的性能与 PyTorch 旗鼓相当甚至有所超越，甚至比 PyTorch 2.3.0 高出多达 32%。然而，由于最近合并的 PR，PyTorch nightly 版本显示出相当大的性能提升，*llm.c* 目前略微落后，token 处理速度慢了约 4%。
- **关于内存效率和操作融合的辩论**：关于将 GELU 等操作与 matmul kernel 融合以节省内存的相对优劣，目前正在进行讨论。虽然这种融合很棘手且可能损害性能，但一些人建议将其融合到前一个 matmul 的 epilogue 中，或者在 backward pass 中重新计算，这可能是一种内存效率较高的折中方案。诸如 prologue vs. epilogue fusion 以及 matmul 在 forward/backward pass 中对输入/输出 tile 的需求等概念是这些辩论的核心。
- **FP32 Master Weights 的潜力**：有人建议默认将 master weights 保留在 FP32 中，以提供更稳定可靠的实现。这一修改将意味着对 optimizer 更新函数和内存分配方案进行某些更改，更新阶段的 lazy initialization 是一种可能的方法。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb">无标题</a>: 未找到描述</li><li><a href="https://pytorch.org/tutorials/recipes/compiling_optimizer.html">(beta) 使用 torch.compile 编译优化器 — PyTorch Tutorials 2.3.0+cu121 文档</a>: 未找到描述</li><li><a href="https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/discard_memory.html">cuda::discard_memory</a>: CUDA C++ Core Libraries</li><li><a href="https://research.colfax-intl.com/adding-fp8-to-flashattention/">使用 FP8 FlashAttention-2 提供 1 PFLOP/s 的性能</a>: 我们最近发布了针对 NVIDIA Hopper™ 架构的 FlashAttention-2 前向传递实现的更新，其中包含多项新的优化和改进，包括……</li><li><a href="https://dev-discuss.pytorch.org/t/performance-comparison-between-torch-compile-and-apex-optimizers/2023">Torch.Compile 与 APEX 优化器之间的性能比较</a>: TL;DR 编译后的 Adam 在所有基准测试中都优于 SOTA 手工优化的 APEX 优化器；在 Torchbench 上提升了 62.99%，在 HuggingFace 上提升了 53.18%，在 TIMM 上提升了 142.75%，在 BlueBerries 上提升了 88.13%。编译后的 AdamW 表现...</li><li><a href="https://stackoverflow.com/questions/28932864/which-compute-capability-is-supported-by-which-cuda-versions">哪些 CUDA 版本支持哪些 Compute Capability？</a>: 以下每个版本支持哪些 Compute Capability：

CUDA 5.5?
CUDA 6.0?
CUDA 6.5?</li><li><a href="https://docs.nvidia.com/deeplearning/cudnn/latest/release-notes.html">发行说明 — NVIDIA cuDNN v9.1.0 文档</a>: 未找到描述</li><li><a href="https://github.com/karpa">karpa - 概览</a>: karpa 有 13 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://godbolt.org/z/hME5EqYrr">Compiler Explorer - CUDA C++ (NVCC 12.2.1)</a>: #include &lt;cuda/barrier&gt; #include &lt;cuda/std/utility&gt; // cuda::std::move #include &lt;cooperative_groups.h&gt; #include &lt;cooperative_groups/reduce.h&gt;  t...</li><li><a href="https://github.com/karpathy/llm.c/pull/313/files">由 ngc92 修复了潜在错误并泛化了 gelu 前向计算 · Pull Request #313 · karpathy/llm.c</a>: 这增加了一个用于从 size_t 安全转换为 ints 的辅助函数（可能也想在 utils.h 中包含它）。该宏随后用于将 size_t 值的 block_size * x128::size 转换回普通的...</li><li><a href="https://github.com/karpathy/llm.c/issues/246">WikiText 103 评估 · Issue #246 · karpathy/llm.c</a>: 我看到一些仓库使用 WikiText-103 作为评估类 GPT 模型的基准数据集，例如：https://github.com/tysam-code/hlb-gpt/tree/main 添加预处理脚本来下载、预处理和分词...</li><li><a href="https://github.com/karpathy/llm.c/pull/325">由 ngc92 为 dev/cuda 提供的混合精度工具 · Pull Request #325 · karpathy/llm.c</a>: 择优挑选（cherry-picked）自 #315</li><li><a href="https://github.com/karpathy/llm.c/pull/314">由 jrhemstad 在 README 中添加 llm.cpp 分支 · Pull Request #314 · karpathy/llm.c</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/326">由 ngc92 提供的将权重保持为 fp32 的选项 · Pull Request #326 · karpathy/llm.c</a>: 增加了一个可选的 fp32 精度权重副本。TODO：缺少 free</li><li><a href="https://github.com/karpathy/llm.c/pull/318">由 karpathy 提供的梯度累积预览 / 开发中（wip） · Pull Request #318 · karpathy/llm.c</a>: 今晚我似乎无法让它工作，有些地方不对劲。Python 部分可以工作。即我们有以下内容。运行默认的 python 脚本可以重现此 PR 之前的旧行为：python ...</li><li><a href="https://github.com/karpathy/llm.c/pull/323">由 karpathy 提供的用于 flash-attention 的 feature/cudnn · Pull Request #323 · karpathy/llm.c</a>: 基于 PR #322 构建。合并 cuDNN 支持的其他细微修复，并随之提供 flash attention</li><li><a href="https://github.com/karpathy/llm.c/pull/273#issuecomment-2087188223">由 PeterZhizhin 添加 NSight Compute 范围，并使用 CUDA 事件进行计时 · Pull Request #273 · karpathy/llm.c</a>: CUDA 事件允许更精确的计时（由 GPU 测量）。nvtxRangePush/nvtxRangePop 为 NSight Systems 添加了简单的堆栈跟踪：示例运行命令：nsys profile mpirun --allow-run-as-roo...</li><li><a href="https://github.com/karpathy/llm.c/pull/227/">由 ngc92 为完全自定义注意力机制提供的第二个 matmul · Pull Request #227 · karpathy/llm.c</a>: 到目前为止，仅在 /dev 文件中，因为对于主脚本，我们还需要修改 backward。出于某种原因，我在这里的基准测试中看到了显著的加速，但在我尝试将其用于...</li><li><a href="https://github.com/karpathy/llm.c/pull/303">由 ChrisDryden 更新 adamw 以使用打包数据类型 · Pull Request #303 · karpathy/llm.c</a>: 在运行时总平均迭代之前

时间：38.547570 ms 运行后：总平均迭代时间：37.901735 ms Kernel 开发文件规范：在当前测试套件中几乎察觉不到：Bef...</li><li><a href="https://github.com/NVIDIA/cudnn-frontend/issues/52#issuecomment-2015335369">cudnn 与 Dao-AILab 之间的 flash attention 实现有什么区别？ · Issue #52 · NVIDIA/cudnn-frontend</a>：这个链接是 flash attention 吗？</li><li><a href="https://github.com/karpathy/llm.c/pull/322">ademeure 提交的 cuDNN Flash Attention 前向与反向 BF16（+35% 性能） · Pull Request #322 · karpathy/llm.c</a>：使用 BF16 且 batch size 为 24 的 RTX 4090：基准测试：232.37ms（约 106K tokens/s）cuDNN：170.77ms（约 144K tokens/s）==&gt; +35% 性能！编译时间：无价 (TM)（约 2.7s 到 48.7s - 这是一个巨大的依赖...）</li><li><a href="https://github.com/karpathy/llm.c/pull/315">ngc92 提交的通过全局范数进行梯度裁剪的初稿 · Pull Request #315 · karpathy/llm.c</a>：一个新的用于计算梯度整体范数的 kernel，以及对 adam kernel 的更新。待办事项：裁剪值在函数调用处硬编码，损坏梯度的错误处理将...</li><li><a href="https://github.com/karpathy/llm.c/pull/262">ngc92 提交的单个 adam kernel 调用处理所有参数 · Pull Request #262 · karpathy/llm.c</a>：通用 Adam kernel 的首次尝试</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L2022">llm.c/train_gpt2.cu 位于 master 分支 · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L2024">llm.c/train_gpt2.cu 位于 master 分支 · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/315/files#diff-49f823d54affd1961dce0e04a078a49ea7bd831326097074aa3db0ea11d0aca4R97-R102">ngc92 提交的通过全局范数进行梯度裁剪的初稿 · Pull Request #315 · karpathy/llm.c</a>：一个新的用于计算梯度整体范数的 kernel，以及对 adam kernel 的更新。待办事项：裁剪值在函数调用处硬编码，损坏梯度的错误处理将...</li><li><a href="https://github.com/karpathy/llm.c/pull">Pull requests · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/pull/120758):">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/pull/121692):">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/273?">PeterZhizhin 提交的添加 NSight Compute 范围，使用 CUDA events 进行计时 · Pull Request #273 · karpathy/llm.c</a>：CUDA events 允许更准确的计时（由 GPU 测量）nvtxRangePush/nvtxRangePop 为 NSight Systems 添加了简单的堆栈跟踪：示例运行命令：nsys profile mpirun --allow-run-as-roo...</li><li><a href="https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/">在 Kepler 上实现更快的并行归约 | NVIDIA 技术博客</a>：并行归约（Parallel reduction）是许多并行算法的常用构建模块。Mark Harris 在 2007 年的一个演示中提供了在 GPU 上实现并行归约的详细策略……</li><li><a href="https://github.com/karpathy/nanoGPT/blob/master/train.py#L307">nanoGPT/train.py 位于 master 分支 · karpathy/nanoGPT</a>：用于训练/微调中型 GPT 的最简单、最快的仓库。 - karpathy/nanoGPT</li><li><a href="https://github.com/pytorch/pytorch/pull/120758">shunting314 提交的 [inductor] 全面填充 · Pull Request #120758 · pytorch/pytorch</a>：来自 ghstack 的堆栈（最早的在底部）：-&gt; #120758 此 PR 添加了在 lowering 期间填充 tensor strides 的功能。目标是确保（如果可能的话）具有不良形状的 tensor 可以具有对齐的 st...</li><li><a href="https://github.com/gevtushenko/llm.c">GitHub - gevtushenko/llm.c: 使用简单、原始的 C/CUDA 进行 LLM 训练</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 gevtushenko/llm.c 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=WiB_3Csfj_Q">奖励课程：CUDA C++ llm.cpp</a>：llm.cpp: https://github.com/gevtushenko/llm.c 幻灯片: https://drive.google.com/drive/folders/1T-t0d_u0Xu8w_-1E5kAwmXNfF72x-HTA?usp=sharing</li><li><a href="https://drive.google.com/drive/folders/1T-t0d_u0Xu8w_-1E5kAwmXNfF72x-HTA">CCCL - Google 云端硬盘</a>：未找到描述</li><li><a href="https://github.com/pytorch/pytorch/pull/99975">Forea

ch kernel codegen in inductor by mlazos · Pull Request #99975 · pytorch/pytorch</a>: 设计文档：在 Inductor 中为 foreach add 的单个重载添加 foreach kernel codegen。覆盖范围将在后续 PR 中扩展到更多算子。示例抄送 @soumith @voznesenskym @penguinwu @anijain2305...</li><li><a href="https://github.com/karpathy/llm.c/pull/306">Packing for Gelu backwards by JaneIllario · Pull Request #306 · karpathy/llm.c</a>: 更新 gelu 反向传播 kernel 以执行 128 位 packing，并创建 gelu 反向传播 cuda 文件。之前的 kernel：block_size 32 | 时间 0.1498 ms | 带宽 503.99 GB/s；block_size 64 | 时间 0.0760...</li><li><a href="https://github.com/karpathy/llm.c/pull/319">convert all float to floatX for layernorm_forward by JaneIllario · Pull Request #319 · karpathy/llm.c</a>: 将所有 kernel 更改为使用 floatX</li><li><a href="https://github.com/karpathy/llm.c/pull/299">Update residual_forward to use packed input by JaneIllario · Pull Request #299 · karpathy/llm.c</a>: 更新 residual_forward 以使用 128 位 packed 输入和 floatX。之前的 Kernel：block_size 32 | 时间 0.1498 ms | 带宽 503.99 GB/s；block_size 64 | 时间 0.0760 ms | 带宽 993.32 GB/s b...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1234763147861299210)** (8 messages🔥): 

- **使用 Torch 2.3 构建时的问题**：一位成员提到在使用 **Torch 2.3** 构建时遇到困难，并表示更倾向于使用 **torch nightly**。
- **AMD 缺少最新的 Flash Attention Kernel**：一位成员询问为什么 AMD 的官方分支还没有移植 **Flash Attention 2.0 版本**，尽管已经有更新的 Flash Attention Kernel 可用。
- **AMD Flash Attention 已添加反向传播**：针对关于 AMD Flash Attention 反向传播（backward pass）的问题，已确认**反向传播确实已实现**，并提供了 [ROCm flash-attention GitHub 仓库链接](https://github.com/ROCm/flash-attention)。
- **Flash Attention 中的 AMD RDNA3 支持**：一位成员询问哪个分支可以让 **RDNA3** 在 ROCm flash-attention 上正常工作，并指出代码中存在 `allowed_archs`。
- **分享 AMD HIP 教程播放列表**：另一位成员认为这些信息很有趣，并分享了一个 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLB1fSi1mbw6IKbZSPz9a2r2DbnHWnLbF-)，内容是 **AMD HIP 教程**，涵盖了在 ROCm 平台上使用 HIP 编程语言。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/playlist?list=PLB1fSi1mbw6IKbZSPz9a2r2DbnHWnLbF-">AMD HIP Tutorial</a>：在这个系列视频中，我们将教授如何使用 HIP 编程语言在 AMD ROCm 平台上为 AMD GPU 编写程序。这套视频是...</li><li><a href="https://github.com/ROCm/flash-attention">GitHub - ROCm/flash-attention: Fast and memory-efficient exact attention</a>：快速且内存高效的精确注意力机制。通过在 GitHub 上创建账号为 ROCm/flash-attention 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1234773459733708810)** (572 messages🔥🔥🔥): 

- **效率与功耗之争**：A4000 16GB GPU 被强调为训练的高效选择，每小时成本显著低于 A100。即将推出的 B200 被视为游戏规则改变者，在相似价格点下，效率可能是 H100 的 25 倍。

- **使用 LoRA 和 QLoRA 进行微调**：讨论澄清了使用 LoRA (16bit) 和 QLoRA (4bit) 在 VRAM 占用和潜在精度损失方面的差异。QLoRA 可节省 75% 的 VRAM，但可能会导致 1-2% 的精度损失，而 LoRA 则没有精度损失。

- **分享训练建议**：推荐的策略是将数据集分为 80% 用于训练，10% 用于超参数调优，并保留 10% 作为隐藏集用于最终模型评估且不再进行调优，以避免污染训练数据。

- **训练土耳其语语言模型**：一位用户正在针对土耳其语的翻译任务微调 Llama 3，使用了超过 43 万个示例。该模型目前的表现像一个翻译机器人，会根据输入语言更改输出语言。

- **在 Unsloth 上进行 ORPO 训练**：分享了一段代码片段，用于在 RTX 4090 GPU 上使用 Unsloth ORPO Trainer 训练 mlabonne/orpo-dpo-mix-40k 数据集，耗时约 5 小时。

- **Unsloth Wiki 更新**：关于微调和训练的贡献已添加到 Unsloth [wiki](https://github.com/unslothai/unsloth/wiki)，并对社区的投入表示感谢。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIk">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2401.10020">Self-Rewarding Language Models</a>: 我们认为，为了实现超人级的 Agent，未来的模型需要超人级的反馈，以提供充足的训练信号。目前的方法通常根据人类偏好训练奖励模型...</li><li><a href="https://tenor.com/view/weird-minion-gif-23757545">Weird Minion GIF - Weird Minion - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cb3q0i/issue_with_with_llama_3_exl2_quant_either_ending/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/19lwcRk_ZQ_ZtX-qzFP3qZBBHZNcMD1hh?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/loading">Load</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k">mlabonne/orpo-dpo-mix-40k · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://gist.github.com/jedt/e45b337e9d9bd0492bf5d3c1d4706c7b">gist:e45b337e9d9bd0492bf5d3c1d4706c7b</a>: GitHub Gist: 即时分享代码、笔记和代码片段。</li><li><a href="https://huggingface.co/NousResearch">NousResearch (NousResearch)</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: 微调 Llama 3, Mistral &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/M-Chimiste/unsloth_finetuning">GitHub - M-Chimiste/unsloth_finetuning</a>: 通过在 GitHub 上创建账号，为 M-Chimiste/unsloth_finetuning 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/400">[FIXED] NotImplementedError: No operator found for `memory_efficient_attention_forward` with inputs · Issue #400 · unslothai/unsloth</a>: 我是尝试使用 Unsloth 的初学者。我运行了免费的 Llama 3 (8B) Notebook，然后遇到了以下错误：在第一步安装过程中我也遇到了以下错误：ERROR: pip's dep...</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: 未找到描述</li><li><a href="https://github.com/huggingface/datasets/issues/6753">Type error when importing datasets on Kaggle · Issue #6753 · huggingface/datasets</a>: 描述 Bug：当尝试运行 import datasets print(datasets.__version__) 时，产生了以下错误 TypeError: expected string or bytes-like object。看起来它找不到 val...</li><li><a href="https://github.com/facebookresearch/xformers#installing-xformers)">GitHub - facebookresearch/xformers: Hackable and optimized Transformers building blocks, supporting a composable construction.</a>: 可定制且经过优化的 Transformer 构建模块，支持组合式构建。 - facebookresearch/xformers</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1234838859401924689)** (6 messages): 

- **AI 模型的大小重要性降低**：有评论指出，**Phi3 Mini 4k** 版本在 Open LLM Leaderboard 上的表现优于其较大的 128k 版本，这表明 Mini 可能是更优的选择。
- **定制化的 Mistral 适配**：有人注意到 **Phi3** 已使用 Mistral 技术进行了修改，但其配置专门用于配合其版本的 Phi 工作。
- **Pi 上的尝试**：一位用户分享了在 Orange Pi Zero 3 上运行 **Phi-3** 的经验，并描述了 Q2 版本的 gemma 2b 性能表现为“稍快”。
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1234773564956086292)** (254 messages🔥🔥):

- **量化与转换问题**：用户报告了使用 `llama.cpp` 进行量化时的问题，例如“*failed to quant q8 gguf messages after a large run*”，手动尝试 GGUF 转换导致了如“*Vocab size mismatch*”之类的错误。引用了与这些问题相关的 [GitHub issues #3759](https://github.com/ollama/ollama/issues/3759) 和 [GitHub issue #4180](https://github.com/vllm-project/vllm/issues/4180)。
- **关于 Few-Shot Learning 和最佳实践的问题**：一位用户询问在训练中是将所有 Few-Shot 示例放在一个用户轮次（user turn）中更好，还是分布在多个轮次中更好。另一位用户 *starsupernova* 建议通过试错来决定，并总体确认两种方法都可行。
- **为以后恢复训练保存微调过程的 Checkpointing**：分享了关于 Checkpointing 的说明，引导用户查看 [Unsloth GitHub Wiki](https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint)，以获取关于如何保存进度并在以后继续训练而不消耗过多存储空间的指导。
- **为微调模型选择推理提供商**：*theyruinedelise* 推荐使用 Jan 或 GPT4All 作为使用 Unsloth 微调的 Llama 3 70B 模型的优秀推理提供商，并附带了 Jan 的 GitHub 仓库链接 ([janhq/jan](https://github.com/janhq/jan))。
- **工作流和教程澄清请求**：多位用户寻求关于训练工作流、保存并推送模型到 Hugging Face 以及如何从 Checkpoints 继续训练的澄清。例如，*starsupernova* 建议将模型和 Tokenizer 都保存到 Hugging Face，并确认在使用 DPO notebook 时设置 `ref_model=None` 是没问题的。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=QmUBVEnvCDJv">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-bnb-4bit">unsloth/llama-3-8b-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF">NousResearch/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>：使用 Unsloth 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/janhq/jan">GitHub - janhq/jan: Jan 是 ChatGPT 的开源替代方案，可 100% 在您的计算机上离线运行。支持多种引擎 (llama.cpp, TensorRT-LLM)</a>：Jan 是 ChatGPT 的开源替代方案，可 100% 在您的计算机上离线运行。支持多种引擎 (llama.cpp, TensorRT-LLM) - janhq/jan</li><li><a href="https://github.com/ollama/ollama/issues/3759">llama3-instruct models not stopping at stop token · Issue #3759 · ollama/ollama</a>：问题是什么？我正在通过兼容 OpenAI 的端点使用 llama3:70b。生成时，我得到了如下输出：请提供上述命令的输出。让我们继续...</li><li><a href="https://huggingface.co/datasets/wikimedia/wikipedia">wikimedia/wikipedia · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/vllm-project/vllm/issues/4180">[Usage]: Llama 3 8B Instruct Inference · Issue #4180 · vllm-project/vllm</a>：您当前的环境：在 2 个 L4 GPU 上使用最新版本的 vLLM。您想如何使用 vllm：我正尝试利用 vLLM 部署 meta-llama/Meta-Llama-3-8B-Instruct 模型并使用 OpenA...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1235093733465395251)** (18 条消息🔥): 

- **多样化数据集且无 VRAM 困扰**：成员们讨论了合并多个数据集是否会增加 VRAM 使用量。共识是合并数据集不会影响 VRAM，而是会增加训练时间。

- **海量数据集的训练挑战**：一位成员思考了在 16GB VRAM 下使用大型数据集微调 Mistral 7B 的可行性。尽管数据集规模巨大，成员们认为虽然可行，但会非常耗时，并建议专注于高质量的合成数据。

- **AI 路线图指南**：有人建议为 AI 项目创建一个简单的路线图。理想情况下，这应该是 README.md 中一个简单的待办事项列表，以明确开发方向和目标。

- **针对聊天机器人的模型增强**：目前正在进行针对更小模型的实验，旨在提高对话能力和准确性。这表明研究重点正转向优化 AI 以实现更好的对话交互。

- **检索增强（Retrieval Augmentation）备受关注**：分享了一个名为 FlagEmbedding 的 GitHub 仓库链接，展示了在检索和检索增强 Long LLMs 方面的工作。对于那些希望通过检索机制改进模型的人来说，这可能很有参考价值。[GitHub 上的 Long_LLM/longllm_qlora](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora)
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/datasets/lmsys/lmsys-chat-1m">lmsys/lmsys-chat-1m · Hugging Face 数据集</a>：未找到描述内容</li><li><a href="https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora">FlagEmbedding/Long_LLM/longllm_qlora at master · FlagOpen/FlagEmbedding</a>：检索和检索增强 LLMs。通过在 GitHub 上创建账号为 FlagOpen/FlagEmbedding 的开发做出贡献。
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1234775457409531926)** (204 条消息🔥🔥): 

- **Flash Attention 已合并至 llama.cpp**：Flash Attention 功能提供了更好的内存效率，并且由于其运行复杂度为 O(N) 而非 O(N^2)，使得上下文更容易适应内存。社区对 llama.cpp 中合并的 PR 表示了极大的热情，链接如下：[FLASH ATTENTION 支持已合并至 llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5021)。
- **加载模型时遇到问题**：用户正在讨论与 LM Studio 中加载模型相关的各种问题，其中一位分享了错误信息，另一位对 VRAM 和物理 RAM 的系统要求表示担忧。
- **关于代理（Proxy）和 LM Studio 的讨论**：在搜索模型时遇到问题的用户可能会发现与公司网络、代理相关的问题，或者在无法路由到 Hugging Face 时需要禁用 IPv6。
- **GPU Offload 说明**：一个重要的建议是在 VRAM 不足时关闭 GPU offload，因为 3GB 的 VRAM 对于 LM Studio 中的某些操作来说是不够的。
- **对 LM Studio Beta 版的期待**：集成了 llama.cpp 新 PR 的 LM Studio 0.2.22 Beta 版发布，吸引了用户进行测试并提供推理质量反馈，同时大家也对 [此更新](https://github.com/ggerganov/llama.cpp/pull/6986) 中看到的 OpenELM 进展充满期待。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/perfecto-chefs-kiss-gif-10500688187407334920">Perfecto Chefs GIF - Perfecto Chefs Kiss - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6986">joshcarp 尝试实现 OpenElm · Pull Request #6986 · ggerganov/llama.cpp</a>：目前在 sgemm.cpp 的第 821 行失败，仍需对 ffn/attention head 信息进行一些解析。目前硬编码了一些内容。修复：#6868。由于需要帮助，将此 PR 作为草案提出...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5021">ggml : 由 ggerganov 添加 Flash Attention · Pull Request #5021 · ggerganov/llama.cpp</a>：参考 #3365。为 ggml 和 llama.cpp 中的 Flash Attention 支持设置所需内容。提议的算子执行：// new res = ggml_flash_attn(ctx, q, k, v, kq_mask, kq_scale);  // fused scale ...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1234772570188939334)** (123 条消息🔥🔥): 

- **探索模型限制**：一位成员询问了下载 1048K 上下文模型却仅使用 20k token 的弊端，并指出更新后的量化选项有限。还有人担心新的 Llama 3 Q8 量化版在 0.2.20 版本（ROCm 预览版）中表现出重复行为。

- **Llama 3 的兼容性问题**：
  参与者讨论了新的 Llama 3 量化版本与旧版本不向后兼容，会出现重复回答的情况。Reddit 上指出：*“[如果你还没有更新到最新的 llama.cpp，这些模型也可以运行，但在你更新工具之前，仍会使用旧的损坏的 Tokenizer。](https://www.reddit.com/r/LocalLLaMA/comments/1cg3e8k/llama_3_8b_instruct_with_fixed_bpe_tokenizer/)”*，这表明为了获得最佳使用效果，更新是必要的。

- **在入门级硬件上的性能缓慢**：用户交流了在 8 GB RAM 的机器上运行无审查模型（如 Everything 7b q4）的可行性。讨论指出模型可以运行，但预期性能会很慢，并建议关闭网页浏览器等额外应用程序以释放资源。

- **图像生成模型的可用性**：在讨论中，明确了 LM Studio 目前不支持直接生成图像。一位成员发布了 AUTOMATIC1111 的 GitHub 仓库链接，这是除了 LM Studio 功能之外，最受欢迎的免费本地图像生成选项之一。

- **寻求增强的人类化 AI 行为**：一位用户寻求关于创建更生动、更像人类的 AI Agent 的建议，并提到了 YouTube 视频中 "Neuro Sama" 的例子。建议包括要求 Llama 3 创建具有特定性格特征的角色提示词，以及探索更多专门研究高级模型行为的频道。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B">vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/dont-know-idk-dunno-no-idea-no-clue-gif-22858277">Dont Know Idk GIF - Dont Know Idk Dunno - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.meta.ai/">Meta AI</a>：使用 Meta AI 助手完成任务，免费创建 AI 生成的图像，并获取任何问题的答案。Meta AI 基于 Meta 最新的 Llama 大语言模型构建，并使用 Emu...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cg3e8k/lla">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://huggingface.co/AI-Engine/BakLLaVA1-MistralLLaVA-7B-GGUF">AI-Engine/BakLLaVA1-MistralLLaVA-7B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/AUTOMATIC1111">AUTOMATIC1111 - Overview</a>：AUTOMATIC1111 在 GitHub 上有 41 个可用的仓库。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ceh5cp/gpt2chatbot_at_lmsys_chatbot_arena/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.youtube.com/shorts/fgG8E6bNwjo">Neuro Challenges Vedal</a>：当 Vedal 向 Neuro 发起挑战时，Neuro 不停地在聊天框刷屏。►Twitch: http://www.twitch.tv/vedal987►Twitter: https://twitter.com/Vedal987#neurosama #vtuber #vedal
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1234792683059281920)** (35 条消息🔥): 

- **模型加载错误报告**：一位用户遇到了错误提示 *"(Exit code: 0). Please check settings and try loading the model again."*，系统环境为 Linux，可用 RAM 为 7.15 GB。
- **Linux 上的各种系统规格**：讨论围绕着观察到异常多的 Linux 用户剩余 RAM 有限；此外，一位拥有 64GB+ RAM 的用户也报告称只有几 KB 的空闲内存。
- **模型生成期间的硬盘杂音**：一位用户注意到，当模型部分卸载到 GPU 并生成 Token 时，电脑发出了 HDD 寻道声或“咔哒声”。他们澄清系统拥有 96GB RAM，且噪音明确来自 HDD，而非电感啸叫或冷却系统。
- **Llama3 模型运行问题**：有关于 Llama3 模型性能的疑问，特别是它是否在向 HDD 缓存而不是保留在 RAM 中。提到的相关模型链接为：[Llama-3-8B-Lexi-Uncensored-GGUF](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF)，运行的上下文大小为 8k Token。
- **LM Studio 与 Ollama 的辩论**：用户分享了对 LM Studio 和 Ollama 的看法，引发了一场关于偏好的辩论，其中一位用户表达了对 LM Studio 的强烈偏好，而另一位用户提醒社区要重视两者，避免负面比较。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF">Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=rJM8rHfsgjk">Hard Drive Sounds</a>：这是我收藏的所有 HDD 硬盘声音的对比。硬盘按从旧到新的时间顺序播放。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1234838825386381312)** (9 条消息🔥): 

- **Llama3 加载困扰**：一位成员在尝试加载具有 100 万 Token 上下文窗口的 **llama3** 时遇到错误，尽管拥有 36GB VRAM 和 128GB RAM。该错误归因于当系统参数设计为 **250,000 上下文大小**时，所需的上下文窗口过大。
  
- **上下文窗口过载**：尝试加载 10 万 Token 的上下文窗口成功达到了系统能力的极限，这表明 100 万 Token 的目标对于资源消耗来说实在太高了。

- **从平方到线性**：一位贡献者提到，上下文问题过去是平方级的，但随着目前的优化，现在“更像是线性的”。

- **配置误读**：一名成员强调，该模型的 **Readme** 指出需要“数百 GB 的 RAM”。这一评论暗示在理解大上下文窗口的硬件需求时可能存在疏忽。

- **模型下载尝试**：该成员提供了一个针对 **Llama-3-8B-Instruct-Gradient-1048k-iMat-GGUF** 的特定目录，这表明其正在尝试下载或引用该模型的特定版本。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1234777495635759104)** (272 messages🔥🔥): 

<ul>
<li><strong>Groq 诱人的 Token 生成速度</strong>：讨论围绕 Groq 为 Llama 3 70B 每秒生成 800 个 Token 的能力展开，并期待即将推出的付费订阅模式。</li>
<li><strong>LLM 硬件指南</strong>：一位成员被告知，他们的 AMD rx 5600m 6GB VRAM 搭配 Ryzen 7 4k 配置对于运行本地模型来说可能偏低，建议他们探索应用首页列出的模型。</li>
<li><strong>模型下载速度</strong>：成员们讨论了在 LM Studio 中从 Hugging Face 下载模型的速度，一人声称约为 10MB/s，另一人则主张对比直接下载与通过 LM Studio 下载的速度。</li>
<li><strong>追求 LLM 的对比准确度</strong>：一位用户寻找能与 ChatGPT 准确度相匹配的 LLM，讨论了最近的 70b Llama 3 和 Wizard 模型，并提到这些性能表现尚属新鲜且未知的领域。</li>
<li><strong>硬件尝试与令人困惑的现象</strong>：围绕 LLM 处理的最佳硬件进行了广泛讨论，重点关注内存速度和 VRAM 等限制因素、SLI/NVLink 功能，以及一个关于两个不同模型在独立情况下生成相同虚构城市名称的轶事，引发了幽默与好奇。</li>
</ul>
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.asrockrack.com/general/productdetail.asp?Model=ROMED8-2T#Specifications">未找到标题</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/#lightbox">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=QK8mJJJvaes">MACKLEMORE &amp; RYAN LEWIS - THRIFT SHOP FEAT. WANZ (OFFICIAL VIDEO)</a>：The Heist 实体豪华版：http://www.macklemoremerch.com The Heist iTunes 数字豪华版：http://itunes.apple.com/WebObjects/MZStore.woa/wa/viewAlb...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1234783013846515752)** (141 messages🔥🔥): 

- **硬件兼容性排障**：一位成员询问软件可以运行但在其硬件配置上遇到 LLM 接受度的问题。另一位参与者建议，i5-4570 搭配 16GB RAM 的硬件对于大多数模型来说可能不足，建议他们只能有效运行 7b Q4 模型。

- **请求新的 llama.cpp Commit**：发布了一个请求，希望获取最新的 llama.cpp Commit 以修复 Tokenizer 问题。回复称很快就会提供。

- **迫切期待 LM Studio 0.2.22**：围绕 **LM Studio 0.2.21** 问题的对话引发了对即将发布的 LM Studio 0.2.22 的期待。讨论表明后续版本可能会解决当前的问题。

- **LM Studio 0.2.22 的发布与快速修复**：宣布发布 **LM Studio 0.2.22 Preview Build 1**，功能包括 UI 优化和更新的 llama.cpp，并分享了 Mac 和修正后的 Windows 安装程序 URL。在对错误的序列号标注产生一些困惑后，提供了一个新的 URL 并确认对 Windows 用户有效。

- **更新后的模型性能讨论**：成员们讨论了 **LM Studio 更新** 后的各种模型性能，重点关注 GGUF 格式问题和近期量化的有效性。一位成员通过“香蕉测试”和苹果数量场景强调了 Llama 3 GGUF 模型在推理方面的差距，并将其与其他格式在逻辑推理任务上的表现进行了对比。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://releases.lmstudio.ai/windows/0.2.22/preview/LM-Studio-0.2.22-Preview-1b-Setup.exe">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/bartowski1182/status/1785764456347103548">来自 bartowski (@bartowski1182) 的推文</a>: 在为 70b instruct 制作 llamacpp 量化版本时遇到了多个问题，我保证很快就会上线 :) 预计明天早上完成</li><li><a href="https://releases.lmstudio.ai/windows/0.2.22/preview/LM-Studio-0.2.22-Preview-1-Setup.exe">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - ggml-org 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://releases.lmstudio.ai/windows/0.2.22/preview/LM-Studio-0.2.22-Preview-1a-Setup.exe">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/doja-cat-star-wars-gif-25078126">Doja Cat GIF - Doja Cat Star - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF">NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF">bartowski/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/qawe-asd-gif-26050335">Qawe Asd GIF - Qawe Asd - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/ojo-huevo-pase-de-huevo-cleanse-clensing-gif-4719953888830735498">Ojo Huevo GIF - Ojo Huevo Pase de huevo - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920).">共同构建更好的软件</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 发现、分叉并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://www.canadacomputers.com/product_info.php?cPath=7_4528_4570&item_id=230804">戴尔宝盒 (黑色) 台式机 i5-4570, 16GB, 512GB SSD, DVD, Win10</a>: 戴尔 RGB 宝盒 OptiPlex SFF (翻新) 家用台式机 Intel Core i5-4570 (最高 3.6GHz), 16GB, 512GB SSD, DVD, Windows 10 Professional (英/法) (黑色)
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1234815876772134932)** (4 条消息): 

- **提出模型加载问题**：一位成员提到在加载模型时遇到问题，并寻求帮助解决。
- **Discord 礼仪提醒**：另一位成员提醒避免在多个频道重复发送问题，建议将此类查询保留在特定频道中。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1234942956536856584)** (40 条消息🔥): 

- **发现 VRAM 读取错误**：一位成员提到 **LM Studio** 错误地读取了其 7900xtx 的 **VRAM 容量**。他们还拥有一台带有集成显卡的 7800x3d，但怀疑这不是导致问题的原因。
- **性能先例引发困惑**：尽管之前曾使用 **RX 6600** 进行 LM Studio GPU offloading，但在更新到 0.2.18 版本后，一位成员遇到了“未检测到支持 ROCm 的设备”的错误。这引发了关于各种 AMD GPU 对 ROCm 和 OpenCL 实现支持的讨论。
- **澄清 HIP SDK 支持误区**：成员们交流了关于**不同 AMD GPU 与 ROCm 和 HIP SDK 兼容性**的信息，指出像 RX 6600 和 6700XT 这样的显卡不受 LM Studio 使用的 HIP SDK 支持。
- **感叹 LM Studio 的 GPU 支持**：虽然一位成员考虑升级到 7900 GRE，但另一位成员建议他们**最好选择 7900XTX**，以确保与 LM Studio 的 ROCm 版本兼容。不同型号在当地的**价格差异**引发了一个幽默的建议：为了购买硬件来一次廉价飞行。
- **寻找 Linux 特定的 ROCm 版本**：对话显示**目前没有针对 Linux 的 ROCm 版本**，这促使人们提到了 Mozilla 在 **llamafile** 上的工作，将其作为解决 AMD 驱动支持相关问题的潜在变通方案。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://rocm.docs.amd.com/en/docs-5.7.1/release/gpu_os_support.html">GPU 和 OS 支持 (Linux) — ROCm 5.7.1 文档主页</a>: 未找到描述</li><li><a href="https://tenor.com/view/doja-cat-star-wars-gif-25078126">Doja Cat GIF - Doja Cat Star - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://future.mozilla.org/news/llamafile-four-months-of-progress-towards-democratizing-ai/">Llamafile：迈向 AI 民主化的四个月进展</a>: 未找到描述</li><li><a href="https://www.ebuyer.com/1597063-sapphire-amd-radeon-rx-7900-xtx-pulse-graphics-card-for-gaming-11322-02-20g">蓝宝石 AMD Radeon RX 7900 XTX PULSE 游戏显卡 - 24GB | Ebuyer.com</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1235008099333439568)** (2 条消息): 

- **CrewAI 与 RAG 的集成**: 一位成员询问如何成功将 **LMStudio** 与 *Retrieval-Augmented Generation* (RAG) 集成，以实现类似于使用 **CrewAI** 进行 **PDFSearch** 或 **WebsiteSearch** 的功能。
- **CrewAI 中的 Embedder 偏好**: 该成员提到可以在 **CrewAI** 中分配像 **huggingface** 这样的 embedder，但表示有兴趣利用 **LMStudio Nomic embed**。
- **模型性能观察**: 他们分享了测试 **Gemma**、**llama3 fp16** 和 **Wizardlm** 模型的经验，发现 **Gemma** 最符合他们的需求。
  

---


**LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/)** (1 条消息): 

yagilb: https://x.com/lmstudioai/status/1785796240656957514
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1235255279474577469)** (25 条消息🔥): 

- **解决上下文扩展中的位置 OOD 问题**: 一位成员强调了一种解决位置分布外 (OOD) 问题的方案，该方案允许模型泛化到更长的上下文。他们分享了一篇 [arXiv 论文](https://arxiv.org/pdf/2401.01325)，该论文提出了这种方法，并认为这是上下文长度扩展领域最被低估 (slept-on) 的论文之一。
- **通过归一化离群值获得更好性能**: 在进一步讨论同一篇论文时，该成员提到模型可以通过归一化离群值在更长的上下文中保持良好性能。这是对早期关于扩展 AI 模型上下文长度讨论的后续。
- **llama.cpp 中的参考实现**: 所讨论概念的一个示例实现可以在 **GitHub** 上的 `llama.cpp` 中找到。它在 server 可执行文件中使用了参数 `--grp-attn-n` 和 `--grp-attn-w`，该成员将其链接到了一个带有配套可视化和说明的 [GitHub 仓库](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)。
- **关于“无限”上下文和 RoPE 的辩论**: 讨论涉及了防止 OOD 问题与扩展上下文能力之间的平衡，一些成员认为 attention truncation 会适得其反。一位成员指出“无限”上下文长度具有误导性，并提到了 [GitHub 上的 ReRoPE 实现](https://github.com/bojone/rerope)，该实现由 RoPE 原作者在 9 个月前发布，暗示可能存在抄袭。
- **无限上下文的神话**: 频道内进行了一次轻松的交流，承认了“无限上下文”模型的不切实际，并调侃了 arXiv 上相关论文数量过多，以及由于 **VRAM** 不足而无法实现此类模型。他们还提到 Google 发布了关于该主题的众多论文之一。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/bojone/rerope">GitHub - bojone/rerope: Rectified Rotary Position Embeddings</a>: Rectified Rotary Position Embeddings。通过在 GitHub 上创建账号为 bojone/rerope 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp/tree/master/examples/server">llama.cpp/examples/server at master · ggerganov/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1234774630288461868)** (25 条消息🔥): 

- **寻找 AI 瑞士军刀 (Swisshutnife)**: 一位成员询问是否有类似于 AI 版 Fiverr 的 **MLOps 悬赏**平台，对此类服务表现出浓厚兴趣。他们得到的建议是，虽然目前还没有专门针对 AI/MLOps 的平台，但在 [Replit](https://replit.com/bounties) 上可以找到通用的编程悬赏。

- **建筑科技职位警报**：分享了一个针对在迈阿密建筑科技公司工作的 Python 和 JavaScript 软件工程师的职位机会。他们在全美范围内有正在进行 beta 测试的项目，并对远程候选人开放。

- **Unreal Engine 上的机器学习**：一位成员宣布推出了**基于 RAG 的 Unreal Engine AI 助手**，旨在改进游戏开发及相关领域的工作流程。他们邀请 Unreal Engine 用户进行尝试并提供反馈，宣传其在加速开发和学习方面的潜力 [点击此处查看](https://neuralgameworks.com)。

- **AI 助手之战**：在基于 RAG 的工具发布后，另一位成员提到了他们开发的**基于 GPT-4 vision 的 Unreal Engine 5 工具**，强调了视觉输入在 UE5 蓝图编辑等特定任务中的优势。

- **算力需求**：一位成员询问了有关**数据生成与评估**的潜在资助或资源，表示需要访问 A100 GPU 等高性能计算资源，以加速其研究并突破当前配置的限制。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://neuralgameworks.com">Neuralgameworks - 你的终极 Unreal Engine AI 助手</a>：未找到描述</li><li><a href="https://replit.com/bounties">Bounties</a>：与顶尖 Replit 创作者合作，将你的创意变为现实。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1234856423482200074)** (9 messages🔥): 

- **AI 泡沫危机**：分享了一个题为“AI 泡沫正在破裂吗？”的 YouTube 视频，讨论了 AI 初创公司生态系统中是否存在泡沫破裂的情况。该视频讲述了涉及三家 AI 初创公司的故事，并使用了 stability/inflection/cohere 分析。[观看 YouTube 视频](https://www.youtube.com/watch?v=p0NxSk7YMrI&ab_channel=Synapse)。

- **数字化记忆**：提到了 **Memary** 的 GitHub 仓库，这是一个旨在为自主 Agent 创建长期记忆的项目，使用 neo4j 以图形方式存储记忆。该项目的新颖方法和潜在性能引起了关注。[探索 Memary 仓库](https://github.com/kingjulio8238/memary)。

- **GPT-2 Chatbot 突然关闭**：来自 @itsandrewgao 的一条推文报告称 gpt2-chatbot 意外下线，引发了人们对这一突然变化的关注。[查看推文](https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw)。

- **AI 理解力挑战**：@VictorTaelin 在 Twitter 上分享了一个极具挑战性的问题，他花了几个小时试图解决但未获成功，并表达了对解决方案的渴望。[查看 Twitter 帖子](https://twitter.com/VictorTaelin/status/1785343416844353697)。

- **AI 高级推理**：一篇 arXiv 论文详细介绍了一种通过使用迭代偏好优化和特殊修改的损失函数来改进 AI 中 **Chain-of-Thought (CoT)** 推理的方法。这种方法显著提高了 **Llama-2-70B-Chat** 在 GSM8K 和 MATH 等各种基准测试中的准确率。[阅读 arXiv 论文](https://arxiv.org/abs/2404.19733)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://verbasizer.com/">Verbasizer</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.19733">Iterative Reasoning Preference Optimization</a>: 迭代推理偏好优化（Iterative Reasoning Preference Optimization）：迭代偏好优化方法最近被证明在通用指令微调任务中表现良好，但通常在推理任务上改进较少 (Yuan et al., 2024, Ch...</li><li><a href="https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw">Andrew Gao (@itsandrewgao) 的推文</a>: gpt2-chatbot 刚刚下线了。我半小时前还在用它！感谢 @shaunralston 的发现 #gpt2 @openai</li><li><a href="https://github.com/kingjulio8238/memary">GitHub - kingjulio8238/memary: Longterm Memory for Autonomous Agents.</a>: 自主 Agent 的长期记忆。通过在 GitHub 上创建账号来为 kingjulio8238/memary 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=p0NxSk7YMrI&ab_channel=Synapse">Is the AI bubble popping?</a>: 三家初创公司的故事描绘了 AI 泡沫可能正在破裂的景象。订阅 Synapse 以获取塑造 AI 领域的深度研究故事...</li><li><a href="https://github.com/KindXiaoming/pykan">GitHub - KindXiaoming/pykan: Kolmogorov Arnold Networks</a>: Kolmogorov Arnold Networks。通过在 GitHub 上创建账号来为 KindXiaoming/pykan 的开发做出贡献。</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI: The Programmable Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected</a>: 可编程的神经符号 AGI，允许你使用基于图的 Prompt Programming 来编程其行为：适用于希望 AI 表现符合预期的人群 - SynaLinks/HybridAGI
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1235338435913453649)** (1 messages): 

- **Hermes 2 基于 Llama-3 推出 Pro 版**: Nous Research 宣布推出 **Hermes 2 Pro on Llama-3 8B**，增强了 Function Calling 和 Structured Output 能力。作为他们首个基于 Llama-3 的模型，它在各种基准测试中超越了前代产品，现已在 [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) 上可用。

- **领跑基准测试**: **Hermes 2 Pro** 在 AGIEval、GPT4All Suite、TruthfulQA 和 BigBench 上展示了优于 Llama-3 8B Instruct 的性能，展示了 AI 评估指标方面的进步。

- **探索量化版本**: 对于那些对轻量级模型感兴趣的用户，**Hermes 2 Pro Llama-3 8B** 的量化版本已发布，在 [HuggingFace GGUF](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF) 上以更高效的体积提供同样的进步。

- **协作成果**: 向 **Hermes 2 Pro** 背后的协作团队致敬，其中包括为最新模型版本的开发和定制做出贡献的特定成员。

- **在 Twitter 上关注动态**: 通过他们的 [Twitter 公告](https://twitter.com/NousResearch/status/1785779313826308096) 关注 Nous Research 关于 Hermes 2 Pro 的最新进展。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF">NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1234808223748587583)** (468 messages🔥🔥🔥): 

- **Llama 和 Hermes 性能讨论**: 成员们讨论了 **Hermes 2 Pro Llama 3** 与之前发布模型之间的性能差异。有人指出 Hermes 2 Pro 可能会遗忘某些任务（如“苹果测试”），但也获得了如 **Function Calling** 等新能力。

- **语言模型量化**: 社区辩论了量化大语言模型（LLM）的有效性。有人指出，在每权重约 **5.5 bits** 左右存在一个极限，超过此限制量化时性能损失会变得显著，而 Q8 量化通常不会导致质量损失。

- **量化训练挑战**: 大家达成共识，**1.58 bit LLM** 可能由于低比特量化的调节特性在训练早期表现良好，但在达到网络容量极限时性能可能会出现分歧。

- **LLMs 中的 Context Length**：**Context length** 的话题也被提及，讨论了其实际限制以及大量的 soft prompt tuning (SPT) 示例是否值得。会议强调，最长的有效样本在文本方面平均约为 **100/200k**。

- **新 LLM 发布与协作努力**：大家对潜在的新 state-of-the-art 模型表现出极大的热情，简要提到了一个 **8B LLM**，并对现有模型的新型 fine-tuning 方法表现出兴趣。这些方面的协作正在进行中，各成员表现出了兴奋感并进行了推测性规划。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://google-research.github.io/seanet/audiopalm/examples/">AudioPaLM</a>: 未找到描述</li><li><a href="https://x.com/hingeloss/">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://x.com/qtnx_/status/1785383089109172705?s=46&t=st">来自 Q (@qtnx_) 的推文</a>: llama-3-vision-alpha 现在可以使用 @huggingface transformers 运行了</li><li><a href="https://x.com/teortaxestex/status/1785682723358622207">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>: 所以即使修复了 token 合并，llama 8b 的量化效果依然不佳。也许是 vocab 的问题，也许只是过度训练，我担心是后者。我（不成熟的）直觉是我们正在精炼 compos...</li><li><a href="https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/">LLM.int8() 与涌现特性 &mdash; Tim Dettmers</a>: 当我参加 NAACL 时，我想做一个小测试。我为我的 LLM.int8() 论文准备了两个推介方案。一个方案是关于我如何使用先进的量化方法来实现无性能损失的转换...</li><li><a href="https://x.com/lmsysorg/status/1785394860754866234?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 lmsys.org (@lmsysorg) 的推文</a>: 感谢社区难以置信的热情！我们真的没预料到这一点。只需澄清几件事：- 根据我们的政策，我们已经与几位模型开发者合作...</li><li><a href="https://huggingface.co/qresearch/llama-3-vision-alpha-hf">qresearch/llama-3-vision-alpha-hf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json">llava_instruct_150k.json · liuhaotian/LLaVA-Instruct-150K at main</a>: 未找到描述</li><li><a href="https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw">来自 Andrew Gao (@itsandrewgao) 的推文</a>: gpt2-chatbot 刚刚下线了。半小时前我还在用它！感谢 @shaunralston 的发现 #gpt2 @openai</li><li><a href="https://tenor.com/view/over9000-dragonball-gif-26144830">Over9000 龙珠 GIF - Over9000 龙珠 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/qtnx_/status/1785383089109172705?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 Q (@qtnx_) 的推文</a>: llama-3-vision-alpha 现在可以使用 @huggingface transformers 运行了</li><li><a href="https://github.com/haotian-liu/LLaVA/blob/main/docs%2FFinetune_Custom_Data.md">LLaVA/docs/Finetune_Custom_Data.md at main · haotian-liu/LLaVA</a>: [NeurIPS'23 Oral] 视觉指令微调 (LLaVA)，旨在实现 GPT-4V 级别的能力及更高水平。- haotian-liu/LLaVA</li><li><a href="https://x.com/sanchitgandhi99/status/1785723896567640356">来自 Sanchit Gandhi (@sanchitgandhi99) 的推文</a>: 上周我们发布了 🤗Diarizers，这是一个用于微调说话人日志（speaker diarization）模型的库 🗣️ 使用免费的 Google Colab，只需 10 分钟即可将多语言性能提升 30%：https://colab.re...</li><li><a href="https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md">DeepSpeed/blogs/deepspeed-ulysses/README.md at master · microsoft/DeepSpeed</a>: DeepSpeed 是一个深度学习优化库，使分布式训练和推理变得简单、高效且有效。- microsoft/DeepSpeed</li><li><a href="https://github.com/cpldcpu/BitNetMCU/blob/main/docs/documentation.md#model-capacity-vs-quantization-scaling">BitNetMCU/docs/documentation.md at main · cpldcpu/BitNetMCU</a>: 在不使用乘法的 CH32V003 RISC-V 微控制器上实现低位宽权重的神经网络 - cpldcpu/BitNetMCU</li><li><a href="https://github.com/tincans-ai/gazelle">GitHub - tincans-ai/gazelle: 语音-语言联合模型 - 直接响应音频！</a>: 语音-语言联合模型 - 直接响应音频！- tincans-ai/gazelle</li><li><a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">“我希望 Llama3 结合我的私有知识发挥 10 倍效能” - 使用 llama3 的本地 Agentic RAG</a>: 高级 RAG 101 - 使用 llama3 构建 Agentic RAG。获取关于 AI 如何重新定义初创公司 GTM 策略的免费 HubSpot 报告：https://clickhubspot.com/4hx🔗 链接- F...</li><li><a href="https://github.com/zhuzilin/ring-flash-attention">GitHub - zhuzilin/ring-flash-attention: 结合 Flash Attention 的 Ring Attention 实现</a>: 结合 Flash Attention 的 Ring Attention 实现 - zhuzilin/ring-flash-attention</li><li><a href="https://youtu.be/ivo-z87x00I?si=w_Jawf7A6mehQnLf">不要忽视 Whisper.cpp</a>: @ggerganov 的 Whisper.cpp 正在将 OpenAI 的 Whisper 推向大众。我们在 “The Changelog” 播客中进行了讨论。🎧 👉 https://changelog.fm/532 订阅以获取更多！...</li><li><a href="https://github.com/jzhang38/EasyContext/blob/main/easy_context/zigzag_ring_attn/monkey_patch.py">EasyContext/easy_context/zigzag_ring_attn/monkey_patch.py at main · jzhang38/EasyContext</a>: 内存优化和训练方案，用于将语言模型的上下文长度外推至 100 万...</li>

ion tokens，仅需极低硬件配置。- jzhang38/EasyContext</li><li><a href="https://x.com/hingeloss/status/1780718391461925049">来自 chris (@hingeloss) 的推文</a>：展示全球最快的 AI 语音聊天——500ms 延迟，本地运行，比其他任何产品快 2 倍。这是如何实现的？👇</li><li><a href="https://demo.tincans.ai/">🦌 Gazelle v0.2</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6936#issuecomment-2088803611">perplexity：更多统计数据，由 JohannesGaessler 添加了文档 · Pull Request #6936 · ggerganov/llama.cpp</a>：我看到一些主观报告称，量化对 LLaMA 3 的损害比对 LLaMA 2 更大。我决定对此进行调查，并为此向 pe... 添加了更多统计数据（和文档）</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama：改进 BPE 预处理 + LLaMA 3 和 Deepseek 支持，由 ggerganov 提交 · Pull Request #6920 · ggerganov/llama.cpp</a>：延续了 @dragnil1 在 #6252 中的工作。此 PR 为 llama.cpp 增加了对 BPE 预分词的支持。摘要：到目前为止，对于所有基于 BPE 的模型，llama.cpp 都应用了默认的预...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1234773238702149632)** (16 messages🔥): 

- **探索百万上下文**：在 lm studio 上加载 1 M 上下文的尝试未成功，并澄清了像 **Phi-3 128k** 这样的模型无法在 ollama 上运行，原因是支持 **Rope Theta** 和 **Ring** 等 attention window 机制存在问题。
   
- **LLaMA Pull Request 前来救场**：用户报告的一个问题已通过 llama.cpp 的新 [pull request](https://github.com/ggerganov/llama.cpp/pull/6920) 得到解决，该 PR 改进了 BPE 预处理并增加了对 LLaMA 3 和 Deepseek 的支持。
   
- **Tokenizer 麻烦与 GGUF**：关于 tokenizer 是否是某个 bug 的根源，以及 GGUF 是否需要重新量化，存在一些困惑，有些人认为问题已解决，而另一些人则不太确定。
   
- **通过逆向工程理解 Grokking**：一份在 [arXiv](https://arxiv.org/abs/2301.05217) 上详细介绍“grokking”现象的研究建议使用机械可解释性（mechanistic interpretability）来逆向工程神经网络的学习行为。
   
- **对 LLM 输出进行排序**：寻求一种对 LLM 输出进行定性排序的方法，建议使用 **argilla distilable** 或奖励模型，尽管在 **distilable** 中执行实际评估的清晰度受到了质疑。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2301.05217">通过机械可解释性衡量 grokking 的进展</a>：神经网络经常表现出涌现行为，即通过扩大参数量、训练数据或训练步数，会产生性质全新的能力。理解这种涌现的一种方法是...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama：改进 BPE 预处理 + LLaMA 3 和 Deepseek 支持，由 ggerganov 提交 · Pull Request #6920 · ggerganov/llama.cpp</a>：延续了 @dragnil1 在 #6252 中的工作。此 PR 为 llama.cpp 增加了对 BPE 预分词的支持。摘要：到目前为止，对于所有基于 BPE 的模型，llama.cpp 都应用了默认的预...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1234865912696537130)** (16 messages🔥): 

- **介绍 Wikipedia RAG 数据集**：分享了 Hugging Face 上 **Wikipedia RAG 数据集**的链接，强调了它与论文《利用 LLM 在多语言密集检索中合成多种语言的训练数据》的相关性。该论文发表于 2023 年 11 月 10 日，可以在[这里](https://huggingface.co/collections/nthakur/swim-ir-dataset-662ddaecfc20896bf14dd9b7)找到。

- **Halal & Kosher 数据集？**：一位成员简要提到了创建标记为 *Halal & Kosher* 的数据集，暗示了在数据集创建中对伦理或文化合规性的考虑。

- **Cynde 集成 Pydantic**：新的 Pydantic 平台正被集成到 Cynde 的重构中，这引起了参与技术开发的成员的兴趣。

- **Logfire 简化代码可观测性**：讨论了 Logfire 平台的引入，它被定位为一个新的可观测性平台，旨在促进函数调用设置中 Pydantic 模型的跟踪。该平台被描述为“直观”且“目前免费”，因其易用性和效率而受到称赞，特别提到了它跟踪嵌套 CV 任务并提供重要数据反馈的能力。有关 Logfire 的更多信息可以在[这里](https://pydantic.dev/logfire)探索。

- **针对特定输出格式的模型微调**：围绕微调 AI 模型以生成特定输出格式展开了讨论，其中一位成员建议简化指令一致性以确保格式正确。**Hermes 2 Pro - Llama-3 8B** 被作为一个例子提及，特别是其在 [Hugging Face 模型页面](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)上的结构化输出部分。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://pydantic.dev/logfire">Pydantic Logfire | 简化的可观测性</a>：Logfire 是一种新型的可观测性平台，建立在与 Pydantic 相同的信念之上——即最强大的工具也可以易于使用。</li><li><a href="https://huggingface.co/collections/nthakur/swim-ir-dataset-662ddaecfc20896bf14dd9b7"> Swan SWIM-IR 数据集 - nthakur 集合</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1234878753155584060)** (24 messages🔥): 

- **虚拟商业和音乐明星模拟器推出**：[CompSimulator](https://hf.co/chat/assistant/662d91081ca01a81e3c21715) 和 [Snow Singer Simulator](https://hf.co/chat/assistant/6626e4869232378718adc5f2) 正式发布，分别在先进 AI 技术的支持下，为用户提供商业和音乐行业的沉浸式体验。
- **架空历史模拟中的 Eldritch 主题**：一位成员描述了一个具有 *Eldritch Nazi* 主题、赛博朋克影响以及 *Reichskommisariat Mittelafrika* 起义内容的架空历史模拟。
- **LLAMA 3 HF Chat 机器人回答的一致性**：有人注意到 HF Chat 上的 **LLAMA 3** 机器人对于发送给它的*相同消息*会生成**相同的回答**。
- **世界模拟演讲与全球社区参与**：分享了一段来自 AGI House SF 演讲的 [YouTube 视频](https://www.youtube.com/watch?v=abWnhmZIL3w)，激发了在洛杉矶举行社区见面会以及连接旧金山和日本的全球活动的计划。
- **Websim 游戏开发更新**：一位用户宣布在 Websim 上创建了一款新游戏，并计划进行一次跨越石器时代到银河时代的更新，虽然发布的链接指向 "null"，但承诺很快会推出更多功能。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://websim.ai/c/mFPjhwkmqAvZROOAU-">未找到标题</a>：未找到描述</li><li><a href="https://hf.co/chat/assistant/6626e4869232378718adc5f2">Snow Singer Simulator - HuggingChat</a>：在 HuggingChat 中使用 Snow Singer Simulator 助手</li><li><a href="https://hf.co/chat/assistant/662d91081ca01a81e3c21715">CompSim - HuggingChat</a>：在 HuggingChat 中使用 CompSim 助手</li><li><a href="https://www.youtube.com/watch?v=abWnhmZIL3w">World Simulation Talks @ AGI House SF</a>：0:00 对话 1:31 Jeremy Nixon 开场 6:08 Nous Research 的 Karan Malhotra 26:22 Websim CEO Rob Hasfield 1:00:08 Midjourney 的 Ivan Vendrov [实时...
</li>
</ul>

</div>
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1234761697533624351)** (497 messages🔥🔥🔥): 

- **对 SD3 发布的怀疑与猜测**：多位用户对 **Stable Diffusion 3 (SD3)** 的发布表示怀疑和担忧，提到曾声称在 4 月发布并期待 5 月发布，结果却对其缺席感到惋惜。讨论充满了怀疑，推测 SD3 可能永远不会正式发布，并猜测 **Stability AI** 可能会因为关于 SD3 免费且开源的误导性陈述而面临抵制。

- **选择适合本地使用的模型**：用户正在积极讨论各种 Stable Diffusion 本地界面的优缺点和教程，包括 **ComfyUI**、**AUTO11111**、**Focus** 和 **Forge**。偏好似乎各不相同，建议根据易用性和用户的具体硬件（如拥有 NVIDIA 还是 AMD GPU）进行选择。

- **使用 AI 进行提示词增强和描述**：个人正在询问图像描述最有效的方法，辩论各种 AI 工具的益处。提到的选项包括使用 **ChatGPT**、**Gemini**，以及采用 **Claude 3** 和 **idefics2** 等模型来分析和改进图像生成的提示词。

- **AI 服务订阅投资与 VPN 使用**：围绕投资 **Gemini** 和 **Claude 3** 等 AI 服务展开了积极的讨论和建议，同时分享了涉及使用 **VPN** 进行区域绕过或维护隐私的实践。用户推荐了各种 VPN，并暗示使用 **DNS over HTTPS** 等功能来增加安全性。

- **在 Automatic 扩展中创建和使用标签**：一位用户询问是否有办法使用 **Automatic1111** 的扩展在输出图像中嵌入标签，随后询问在 **ComfyUI** 等自定义界面中是否存在等效于 **clip skip** 和 **stylizer** 的功能。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://civitai.com/articles/5069">迈向 Pony Diffusion V7 | Civitai</a>：大家好，我很高兴能分享我们即将推出的 V7 的进展更新，以及对 V6 的回顾分析。V6 所获得的认可...</li><li><a href="https://tenor.com/view/yuji-stare-jujutsu-kaisen-blank-shibuya-sukuna-gif-2005904860443811921">虎杖悠仁凝视《咒术回战》GIF - Yuji Stare Jujutsu Kaisen Blank - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/blog/idefics">介绍 IDEFICS：最先进视觉语言模型的开源复现</a>：未找到描述</li><li><a href="https://huggingface.co/blog/idefics2">介绍 Idefics2：为社区提供的强大 8B 视觉语言模型</a>：未找到描述</li><li><a href="https://civitai.com/models/428813">Mythos - v1.0 | Stable Diffusion Checkpoint | Civitai</a>：V1 版本不知为何有 3.55GB 大……我想我成功做了一个稳定的 fp8 剪枝？？我真的不知道它是怎么变成 3.55GB 的……V2 是正常的 6GB 模式……</li><li><a href="https://tenor.com/vD6Ib9MNmkI.gif">Melxts2008 Emoji GIF - Melxts2008 Emoji Smile - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://stability.ai/stable-assistant">Stable Assistant — Stability AI</a>：Stable Assistant 是由 Stability AI 开发的友好聊天机器人，配备了 Stability AI 的文本和图像生成技术，具有 Stable Diffusion 3 和 Stable LM 2 12B。</li><li><a href="https://github.com/hiddenswitch/ComfyUI/blob/master/script_examples/basic_api_example.py">ComfyUI/script_examples/basic_api_example.py (master 分支) · hiddenswitch/ComfyUI</a>：一个带有图形/节点界面的强大且模块化的 Stable Diffusion GUI。- hiddenswitch/ComfyUI</li><li><a href="https://github.com/hiddenswitch/ComfyUI/blob/0862863bc00165b9ba0607595f304f93ca995887/tests/distributed/test_embedded_client.py#L32">ComfyUI/tests/distributed/test_embedded_client.py (特定提交版本) · hiddenswitch/ComfyUI</a>：一个带有图形/节点界面的强大且模块化的 Stable Diffusion GUI。- hiddenswitch/ComfyUI</li><li><a href="https://civitai.com/articles/5069?highlight=301393">迈向 Pony Diffusion V7 | Civitai</a>：大家好，我很高兴能分享我们即将推出的 V7 的进展更新，以及对 V6 的回顾分析。V6 所获得的认可...</li><li><a href="https://civitai.com/articles/4248/what-is-score9-and-how-to-use-it-in-pony-diffusion">什么是 score_9 以及如何在 Pony Diffusion 中使用它 | Civitai</a>：你可能在 Pony Diffusion 的提示词中见过 score_9 或其更长版本 score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up...</li><li><a href="https://github.com/Stability-AI/generative-models/blob/main/model_licenses/LICENSE-SDXL1.0">generative-models/model_licenses/LICENSE-SDXL1.0 (main 分支) · Stability-AI/generative-models</a>：Stability AI 的生成模型。通过在 GitHub 上创建一个账户来为 Stability-AI/generative-models 的开发做出贡献。</li><li><a href="https://github.com/AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin">GitHub - AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin: 一个用户友好的插件，可以轻松地在 Photoshop 中使用 Automatic 或 ComfyUI 作为后端生成 Stable Diffusion 图像。</a>：一个用户友好的插件，可以轻松地在 Photoshop 中使用 Automatic 或 ComfyUI 作为后端生成 Stable Diffusion 图像。- AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1234949294767734865)** (1 条消息): 

- **更多聊天记录控制权**：OpenAI 现在为 ChatGPT 免费版和 Plus 用户**更新了数据控制选项**。即使**选择了退出训练数据**贡献，任何人也可以访问其聊天记录；该更新已在网页端上线，即将登陆移动端。
- **推出临时聊天 (Temporary Chat)**：用户现在拥有一个新的隐私选项——**Temporary Chat** 功能，允许进行不会存储在聊天记录中的一次性对话。
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1234785511420334090)** (375 条消息 🔥🔥):

- **GPT-2 Chatbot 引发好奇**：成员们讨论了 "gpt2-chatbot" 模型，有人表示它在许多情况下表现优于 GPT-4，而另一些人则指出它在某些不明场景下会失败。使用 gpt2-chatbot 似乎可以进行无限生成，但该模型在某些竞技场（arenas）中已变得不可用。

- **AI 与情感**：围绕 AI 和情感的概念展开了深入讨论，成员们思考了 AI 随着时间的推移发展出情感意识的可能性。人们将 AI 的进化与人类的情感发展进行了比较，对于 AI 是否能够或应该努力实现类似于人类的同理心或情感理解，存在不同的看法。

- **免费版的限制**：进行了一场关于免费用户获取 OpenAI 功能（如 DALL-E）的讨论，一些人表达了希望在不订阅的情况下增加功能的愿望。对话体现了对商业现实的认知以及社区对 OpenAI 产品服务的期望。

- **学术界中的 AI 协作**：一位用户询问社区如何在学术写作中有效地与多个 AI 模型（如 ChatGPT 和 Claude）协作。建议包括使用能够在上下文中保留其他 AI 响应的第三方聊天机器人。

- **关于 DALL-E 更新的想法**：讨论涵盖了 DALL-E 的现状以及对未来版本（如 DALL-E 4）的假设。虽然一些用户注意到 DALL-E 3 的改进带来了更好的创作结果，但对话也强调了良好的人机协同仍然至关重要，并辩论了 AI 适应人类认知模式的重要性。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.udio.com/songs/7P8SyrG3cq9C1mpJfaGRMx">Udio | Echoes in the Chaos by Tcald | AI Music Generator - Official Website</a>: 在 Udio 上听 Tcald 的 Echoes in the Chaos。发现、创作并与世界分享音乐。使用最新技术在几秒钟内创作 AI 音乐。</li><li><a href="https://github.com/openai/simple-evals#benchmark-results">GitHub - openai/simple-evals</a>: 通过在 GitHub 上创建账户为 openai/simple-evals 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1234855100279685200)** (10 messages🔥): 

- **GPT-2 在聊天系统中的探索**：一位成员分享了他们在聊天系统集成中实验 **GPT-2** 的经验。他们将讨论的进一步细节引导至特定频道。

- **归档意外与批量删除查询**：一位用户不小心归档了所有聊天记录，并询问批量删除选项，以便处理大量聊天记录，而不是逐个删除。

- **截图分享**：有人询问为什么不能在这个频道发布截图，因为一位成员想分享 GPT 集成的幽默输出。

- **引导至支持图片的频道**：向成员说明截图可以在另一个专门用于此类内容的频道中分享。

- **ChatGPT 字符限制的不一致性**：一位成员指出，ChatGPT 据称误报了其字符限制，允许输入长于所述 4096 个字符的内容。

- **澄清 ChatGPT 的局限性和行为**：一位成员解释说，**ChatGPT 的自我意识有限，因为它没有经过准确了解其功能或版本的训练**。他们区分了具有不同 Token 限制的免费版和 ChatGPT Plus 版本，并提到了当达到上下文限制时 ChatGPT 总结对话的可能性。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1234813834124988448)** (30 messages🔥): 

- **负向提示（Negative Prompting）的挑战**：成员们讨论了*负向提示*的问题，指出提供所需输出的示例比列出禁止内容更有效。一个建议包括将指令重新表述为“使用 *y* 代替 *x*”。

- **地区方言的困扰**：提出了一个使用案例，涉及避免在阿根廷西班牙语方言中具有不同含义的特定词汇。成员们建议测试 AI 对阿根廷西班牙语的理解，并考虑采用解释词汇上下文用法的方法，而不是列出禁止清单。

- **利用正向提示的功效**：强调了正向提示（可能采用“使用 *y* 代替 *x*”的结构）比列出负面示例或禁令更有可能使 GPT 的输出符合要求。

- **元数据提示（Metadata-Prompting）探索**：对于一位业余探索者，讨论了一种使用开放变量和 Markdown 进行强调的简单元提示形式。有人建议这可以增强与 GPT 的交互。

- **与 AI 模型的交互性**：概述了 meta-prompting 在促进交互式、动态和多层提示词方面的潜力。示例包括使用 `{openVariable}` 占位符来引导 AI 的行为，以及构建输出模板以支持交流。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1234813834124988448)** (30 messages🔥): 

- **思考模型提示技术**：成员们讨论了 OpenAI 模型的提示策略，强调使用**正面指令和示例**而非负面指令，以避免不当词汇的使用。他们分享了构建提示词的见解，这些提示词可以驱动 AI 产生更好的结果，而无需列出禁词，并强调了诸如使用 *"instead of 'x', use 'y'"* 之类的措辞来引导 AI 的语言选择。
  
- **知识就是力量**：在讨论从 LinkedIn 数据生成详细的**理想客户画像 (ICP)** 的技术时，一位用户提出了一种策略，包括分析帖子和截图以确定一个人的统计学特征、心理特征和行为。目标是让 AI 作为个人品牌和目标受众专家，作为营销和销售内容策略的一部分。 

- **提示工程入门**：一位作为爱好者的成员请求关于提示工程的建议，希望深入研究与 AI 的交互以获取知识和进行编码。其他参与者提供了建议，如在 meta-prompting 中使用 **open variables**，以及利用 markdown 来结构化和强调提示词的部分内容，以鼓励更复杂的 AI 行为。

- **用于交互体验的 Meta-Prompting**：大家达成共识，认为 **meta-prompting 技术**（用户为 AI 创建动态和交互式提示词）可以显著增强用户完成复杂任务的能力。对话中包含了一个如何构建 meta-prompt 以使 AI 充当专家系统的示例。

- **AI 提示工程之旅**：一位 AI 爱好者获得了关于开始学习提示工程以改善与 OpenAI 模型交互的鼓励和指南。特别讨论了 **open variables** 的作用和在提示词中使用 markdown，以及使用网页搜索功能研究提示工程技术的潜在好处。
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1235010301657743400)** (1 messages): 

- **新功能 'Pages' 的独家早期访问**：一个名为 **Pages** 的新功能即将推出，它提供了一种易于创建、可分享的、针对任何主题的深入探索方式。感兴趣的用户可以通过回复特定的表情符号并前往指定频道加入 beta 测试计划，以获得早期访问权限和提供反馈的机会。
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1234782554809434122)** (241 messages🔥🔥): 

- **API 引用困扰**：一位成员询问在使用 **Perplexity-online** 模型时，如何通过 API 请求获取像 [1] 这样的引用并查看网页 UI 参考资料。另一位成员解释说，由于折扣码的欺诈问题，预期的计划早些时候已暂停。
  
- **Pro Search 和参考功能的缺陷？**：多位用户报告了 Perplexity 上 Pro Search 和参考功能的问题，注意到答案冗余或缺少参考资料；甚至有人声称在升级到高级版后仍面临这些故障。

- **关于 Opus 每日限制的疑问**：围绕 Opus 使用的每日限制展开了讨论，成员们澄清限制为**每天 50 次，每 24 小时重置一次**。一些人对缺乏提高该限制的预计时间表示不满。

- **Perplexity 性能和问题**：用户分享了 AI 模型响应缓慢和登录账户困难的经历。建议仔细检查垃圾邮件文件夹中的登录链接，并推测服务提供商可能会拦截电子邮件。

- **模型差异和功能的明确**：对话涉及了不同模型回答质量的差异，以及 scratchpad 提示词、AI 提示不准确和 context window 大小等功能。一位用户确认 **context window 确实是 32k**。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://chat.reka.ai/">Reka Playground</a>: 探索由 Reka 构建的最新的多模态语言模型</li><li><a href="https://youtu.be/ddTV12hErTc">Rabbit R1: Barely Reviewable</a>: 盒子里的 AI。但是个不同的盒子。在 https://dbrand.com/rabbit 获取 dbrand 皮肤和屏幕保护膜。MKBHD 商品：http://shop.MKBHD.com。我目前正在使用的技术...
</li>
</ul>

</div>
  

---

**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1234786005630976051)** (19 messages🔥): 

- **探索 Perplexity AI**：多位用户分享了 [Perplexity AI 搜索结果](https://www.perplexity.ai/search)，探索的主题涵盖了从 **Microsoft Research Asia** 到 **Vimeo API** 以及关于 **Mac App Store** 的查询。
- **Lenny's Newsletter 关于产品洞察**：一名成员分享了 [Lenny's Newsletter 的链接](https://www.lennysnewsletter.com/p/how-perplexity-builds-product)，内容包括 Duolingo 的增长秘诀以及 AI 将如何影响产品管理，并邀请订阅以获取完整访问权限。
- **Google 最近的裁员**：流传的一条链接涉及 [Google 在其他业务调整中裁员](https://www.perplexity.ai/search/Google-lays-off-ZBS6dB9mSzqqA7OGS0M1sA)的消息。
- **Tesla 全自动驾驶讨论**：汽车技术是一个关注点，分享了关于 [Tesla Full Self-Driving](https://www.perplexity.ai/search/Teslas-full-selfdriving-IJfuMlVMR_ay5YL0F49BlA) 能力的链接。
- **Discord 分享功能提醒**：Perplexity AI 提醒用户确保其线程是可分享的，并提供了直接链接自 Discord 平台的视觉指南。

**提及的链接**：<a href="https://www.lennysnewsletter.com/p/how-perplexity-builds-product?utm_medium=web">How Perplexity builds product</a>：联合创始人兼产品负责人 Johnny Ho 解释了他如何像黏菌（slime mold）一样组织团队，如何使用 AI 来构建他们的 AI 公司，以及更多内容。

---

**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1234782587151450142)** (14 messages🔥): 

- **关于 API 引用（Citations）的困惑**：一名成员询问在使用 **perplexity-online models** 获取网络知识时，是否可以通过 API 请求获取引用，另一名成员提到了之前似乎已解决相关问题的消息。
- **Claude 3 使用政策说明**：用户询问了 Perplexity 提供的 **Claude 3** 的使用政策，特别是关于政治用途，以及在使用其模型时，Perplexity 的使用政策是否优先于 **Anthropic** 的政策。
- **Perplexity Pro 与 API 结果差异**：用户指出 **Perplexity Pro** 界面与使用相同 Prompt 的 **API** 获取的结果之间存在差异，对此，另一名成员澄清说 Perplexity UI 和 API 可能没有使用相同的模型版本。
- **API 文档澄清**：针对模型版本的困惑，一名用户引用了 **[Perplexity API 文档](https://docs.perplexity.ai/docs/model-cards)**，其中列出了 `llama-3-70b-instruct` 等模型及其参数详情，并指导成员如何避免 Prompt Injections。
- **理解 Online 模型**：用户询问 Perplexity Pro UI 使用的是哪种 Online 模型，随后得到的解释是：**online models** 要么经过微调以更有效地使用来源，要么采用类似 RAG 的方法从搜索引擎式的向量数据库（vector database）中综合响应。

**提及的链接**：<a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>：未找到描述

---

**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1234819191492710490)** (28 messages🔥): 

- **用于高效推理的 Effort/bucketMul**：介绍了一种名为 **effort/bucketMul** 的新算法，该算法声称能显著提高向量-矩阵近似和 LLM 推理速度。它被描述为可根据计算负载实时调整，并与 Mistral 等模型兼容。[算法发布地址](http://kolinko.github.io/effort/)。
- **业余 AI 爱好者展示图像补丁（Image Patch）研究**：一位业余 AI 爱好者分享了他们受神经系统启发的高效图像补丁表示研究，可在 [arXiv](https://arxiv.org/abs/2210.13004) 上查看。他们提出了一种通过无监督学习（unsupervised learning）获得的新型二进制向量表示。
- **关于二进制与超球体嵌入（Embeddings）的讨论**：成员们讨论了嵌入的二进制向量表示的优点，将其益处与生物学合理性和计算效率联系起来。一名成员考虑将类似原理应用于 RWKV LLM，以实现潜在的更快学习。[RWKV LLM 方法](https://github.com/BlinkDL/SmallInitEmb)。
- **嵌入策略建议**：针对关于表示方法的讨论，分享了该领域基础论文的链接，包括 CLIP 和 Dino，以便进一步阅读关于嵌入分布的内容。[CLIP 论文](https://arxiv.org/abs/2103.00020)，[Dino 论文](https://arxiv.org/abs/2104.14294)。

- **关于使用 CLIP Embeddings 进行图像分类的查询**：一位成员就使用 CLIP Embeddings 对影星图像进行分类寻求建议，在使用修改后的标签和 Prompt 的情况下，准确率仅为 36%。他们尝试了使用文本描述的余弦相似度，但由于效果没有提升，正在考虑其他替代方案。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2102.11174">Linear Transformers Are Secretly Fast Weight Programmers</a>：我们展示了线性化自注意力机制与 90 年代初的快速权重控制器（fast weight controllers）在形式上的等价性，其中“慢速”神经网络通过梯度下降学习来为“快速”网络编程...</li><li><a href="http://kolinko.github.io/effort/">Effort Engine</a>：一种可能的新型 LLM 推理算法。实时平滑地调整您在推理过程中希望进行的计算量。</li><li><a href="https://arxiv.org/abs/2210.13004">Efficient Representation of Natural Image Patches</a>：利用基于受生物系统启发的极简且现实假设的抽象信息处理模型，我们研究了如何实现早期视觉系统的两个终极目标...</li><li><a href="https://www.reddit.com/user/No_Dragonfruit_5472/comments/1cef7gc/tradingview_premium_pack_crack_2024/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1234762365325545512)** (192 条消息🔥🔥): 

- **揭秘“黑盒”类比**：讨论揭示了关于为什么大语言模型 (LLMs) 常被称为“黑盒”的不同观点。一些参与者指出，相对于我们的理解，LLMs 的内部运作机制非常复杂；而另一些人则认为，这类术语的不精确使用反映了人类鹦鹉学舌般重复简练短语的倾向。

- **在测试集上训练 LLM 会影响公平比较**：分享的一个 [链接](http://arxiv.org/abs/2404.18824) 指出，在基准测试集上训练的 LLMs 会扭曲基准测试的有效性，并导致潜在的不公平比较。

- **LLM 中的思维链 (CoT) 合理性**：针对 LLM 如何解释其推理过程的问题，一些消息指出，LLM 为答案生成的解释并不值得信任，因为它们往往不能反映模型的内部思维过程。

- **Kolmogorov-Arnold Networks (KANs) 优于 MLPs**：重点介绍了一篇 [论文](http://arxiv.org/abs/2404.19756)，该论文引入了 Kolmogorov-Arnold Networks (KANs) 作为多层感知机 (MLPs) 的替代方案，并指出 KANs 具有更好的准确性和可解释性，拥有更快的缩放定律 (scaling laws) 以及直观可视化的潜力。

- **通过迭代偏好优化提升 LLM 推理能力**：分享的研究 ([链接](http://arxiv.org/abs/2404.19733)) 讨论了一种通过优化竞争生成的 CoT 候选之间的偏好来改进 LLM 推理的迭代方法，从而提高了在 GSM8K、MATH 等任务中的准确率。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="http://arxiv.org/abs/2404.18824">Benchmarking Benchmark Leakage in Large Language Models</a>: 在预训练数据使用不断扩大的背景下，基准测试数据集泄露现象日益突出，而不透明的训练过程和通常未披露的包含内容加剧了这一问题...</li><li><a href="https://arxiv.org/abs/2404.19756">KAN: Kolmogorov-Arnold Networks</a>: 受 Kolmogorov-Arnold 表示定理启发，我们提出了 Kolmogorov-Arnold Networks (KANs) 作为 Multi-Layer Perceptrons (MLPs) 的有力替代方案。虽然 MLPs 具有固定的激活函数...</li><li><a href="http://arxiv.org/abs/2404.19733">Iterative Reasoning Preference Optimization</a>: 迭代偏好优化方法最近在通用指令微调任务中表现良好，但在推理任务上通常改进甚微 (Yuan et al., 2024, Ch...</li><li><a href="https://arxiv.org/abs/2404.14662">NExT: Teaching Large Language Models to Reason about Code Execution</a>: 人类开发者的一个基本技能是理解和推理程序执行的能力。例如，程序员可以用自然语言在脑中模拟代码执行来调试...</li><li><a href="http://arxiv.org/abs/2001.04063">ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training</a>: 本文介绍了一种名为 ProphetNet 的新型序列到序列预训练模型，它引入了一种名为未来 n-gram 预测的新型自监督目标以及所提出的 n-stream 自注意力...</li><li><a href="https://videogigagan.github.io/">VideoGigaGAN</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2402.12365">Universal Physics Transformers: A Framework For Efficiently Scaling Neural Operators</a>: 神经算子作为物理代理模型，最近引起了越来越多的关注。随着问题复杂性的不断增加，一个自然的问题出现了：什么是扩展神经算子的有效方式...</li><li><a href="https://arxiv.org/abs/2404.12388">VideoGigaGAN: Towards Detail-rich Video Super-Resolution</a>: 视频超分辨率 (VSR) 方法在升采样视频中表现出了令人印象深刻的时间一致性。然而，随着倍率增加，这些方法往往比图像领域的对应方法产生更模糊的结果...</li><li><a href="http://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>: 在写作和交谈时，人们有时会停下来思考。虽然以推理为中心的工作通常将推理框架化为回答问题或完成 Agent 任务的方法，但推理对于...</li><li><a href="http://arxiv.org/abs/2312.02179">Training Chain-of-Thought via Latent-Variable Inference</a>: 当被指示使用“Chain-of-Thought” (CoT) 提示逐步得出答案时，大型语言模型 (LLMs) 解决问题更加准确且具有可解释性。人们还可以改进...</li><li><a href="https://github.com/lauraaisling/analyse-llms/blob/main/notebooks/Mode_Collapse.ipynb">analyse-llms/notebooks/Mode_Collapse.ipynb at main · lauraaisling/analyse-llms</a>: 通过在 GitHub 上创建一个账户，为 lauraaisling/analyse-llms 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2403.18506">Faster Convergence for Transformer Fine-tuning with Line Search Methods</a>: 最近的研究表明，线搜索方法在各种数据集和架构上大大提高了传统随机梯度下降方法的性能 [1], [2]。在这项工作中，我们建议...</li><li><a href="https://github.com/s-chh/PyTorch-Vision-Transformer-ViT-MNIST-CIFAR10">GitHub - s-chh/PyTorch-Vision-Transformer-ViT-MNIST-CIFAR10: Simplified Pytorch implementation of Vision Transformer (ViT) for small datasets like MNIST, FashionMNIST, SVHN and CIFAR10.</a>: 针对 MNIST, FashionMNIST, SVHN 和 CIFAR10 等小数据集的 Vision Transformer (ViT) 的简化 PyTorch 实现。 - s-chh/PyTorch-Vision-Transformer-ViT-MNIST-CIFAR10</li><li><a href="https://www.biorxiv.org/content/10.1101/2024.04.28.591528v1">Sequential predictive learning is a unifying theory for hippocampal representation and replay</a>: 哺乳动物的海马体包含一个认知地图，代表动物在环境中的位置，并生成离线“回放”以用于回忆、规划和形成长...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1235201401781616690)** (34 messages🔥):

- **Exploring the Computational Model of Sequence-Prediction**：探索 Sequence-Prediction 的计算模型：一位成员对 Sequence-Prediction 模型学习到的计算模型进行了理论化，特别是与 next-token prediction loss 相关的部分，预测了 token 概率中相位转换（phase transitions）的存在，并就其撰写的文章寻求反馈，链接见[此处](https://docs.google.com/document/d/11w3of15CbfOlWrvQpTjxaJt-UvtOckzr0WQUfTrTnsw/edit?usp=sharing)。

- **Connecting Prior Work with Theoretical Predictions**：将先前工作与理论预测相结合：该成员承认了关于 Transformer 和迭代推理（iterative inference）的现有研究的相关性，特别是来自[这篇论文](https://arxiv.org/abs/2303.08112)的 *tuned lens* 方法，并讨论了早期解码的发现如何与其提出的理论相一致。

- **Discussing Model Representations with Tied Embeddings**：讨论具有 Tied Embeddings 的模型表示：随后展开了关于像 Mamba 这样具有 Tied Embeddings 的模型如何影响解释的对话，并推测 Tied Embeddings 实际上可能有利于模型的表示一致性（representational coherence）。

- **Drafting Implementation Plans for Theoretical Predictions**：起草理论预测的实施计划：针对是否考虑通过实施来测试假设，讨论了可能使用 *transformer lens* 和 *gpt-2-small* 进行实验的可能性。

- **Exchanging Interpretability Insights**：交流可解释性见解：成员们就定义和操作模型特征的“原子性”（atomicity）所面临的挑战交换了意见。引用了新兴概念，如 *distributional simplicity bias* 和神经缩放法则（neural scaling laws）的 *Quantization Model*，并链接到了[此处](https://arxiv.org/abs/2402.04362)和[此处](https://arxiv.org/abs/2303.13506)的研究论文。

- **Refining Interpretability Methods with Formal Languages**：使用形式语言优化可解释性方法：有人建议定义一个任意的形式语法（formal grammar），并在该语言的序列上训练网络，以确定语法的规则是否可以被视为“真正的底层特征”，并将 Transformer 对 Dyck languages 的理解作为一个相关的切入点进行研究。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2303.13506">The Quantization Model of Neural Scaling</a>：我们提出了神经缩放法则的 Quantization Model，解释了观察到的 loss 随模型和数据大小呈幂律下降的现象，以及新能力随规模突然出现的现象……</li><li><a href="https://arxiv.org/abs/2402.04362">Neural Networks Learn Statistics of Increasing Complexity</a>：Distributional simplicity bias (DSB) 假设神经网络首先学习数据分布的低阶矩，然后再转向高阶相关性。在这项工作中，我们展示了……</li><li><a href="https://arxiv.org/abs/2303.08112">Eliciting Latent Predictions from Transformers with the Tuned Lens</a>：我们从迭代推理的角度分析 Transformer，试图理解模型预测是如何逐层细化的。为此，我们为冻结的……中的每个 block 训练了一个仿射探针。</li><li><a href="https://docs.google.com/document/d/11w3of15CbfOlWrvQpTjxaJt-UvtOckzr0WQUfTrTnsw/edit?usp=sharing">Deriving a Model of Computation for Next-Token Prediction</a>：未找到描述
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1234762736504672346)** (2 条消息):

- **CVPR 竞赛奖金**：HuggingFace 宣布为 CVPR 活动举办三场不同的竞赛，总奖池超过 **$120,000**。参与者可以在 2024 年 6 月 17 日至 21 日期间加入 [SnakeCLEF](https://huggingface.co/spaces/BVRA/SnakeCLEF2024)、[FungiCLEF](https://huggingface.co/spaces/BVRA/PlantCLEF2024) 和 [PlantCLEF](https://huggingface.co/spaces/BVRA/PlantCLEF2024)。
- **Transformers 库更新**：*Transformers* 库已更新至 [v4.40.0](https://github.com/huggingface/transformers/releases/tag/v4.40.0)，引入了 Phi-3、Llama 3、IDEFICS 2 等模型。此外，Phi-3 已实现可在浏览器内运行，速度达到每秒约 20 个 tokens。
- **Gradio 和 Datasets 库增强**：Gradio 发布了 [v4.28.0](https://www.gradio.app/changelog) 重大更新，重点关注 Custom Components（自定义组件）；同时 Datasets 库已达到 [v2.19.0](https://github.com/huggingface/datasets/releases/tag/2.19.0)，实现了 Polars 兼容性并改进了导出功能。
- **强化你的 Prompt**：HF Blog 通过一篇关于 [Structured Generations](https://huggingface.co/blog/evaluation-structured-outputs)（结构化生成）的文章，重点介绍了增强语言模型输出中 Prompt 一致性的技术。
- **Snowflake 发布令人瞩目的模型**：Snowflake 发布了一个巨大的 408B Dense + Hybrid MoE 模型，拥有 17B 激活参数，具备 SQL 生成、代码编写和指令遵循等广泛能力。这一成就详见此份[公告](https://x.com/reach_vb/status/1783129119435210836)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/fleetwood___/status/1783195985893863578)">Fleetwood (@fleetwood___) 的推文</a>：🚨 Phi-3 在浏览器中运行 🚨 达到约 20 tok/s 🏎️ 仅需 3 行 JS。仍有一些小问题需要解决，即将集成到 Ratchet 0.4.0 中。</li><li><a href="https://x.com/abhi1thakur/status/1785279012232736991)">abhishek (@abhi1thakur) 的推文</a>：我能在 Kaggle 上运行 AutoTrain UI 吗？是的，你可以！！！查看我最新的 Notebook，复制它，填入你的 tokens，即可享受在 Kaggle Notebooks 后端运行的 AutoTrain UI 🚀 Notebook 链接：https://www...</li><li><a href="https://x.com/reach_vb/status/1785039538185703909)!">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：冲吧！！Common Voice 17 - 现已登陆 Hub！🔥 包含 124 种语言的 31,000 小时音频（及转录）。*开启声音 🎶* CV 17 增加了 847 小时的数据，以及 493 小时的...</li><li><a href="https://x.com/BrigitteTousi/status/1783573043815596426):">Brigitte 🤗 (@BrigitteTousi) 的推文</a>：🔊 呼叫所有记者！我们很高兴与 @fdaudens 一起宣布在 @huggingface Hub 上建立一个新社区：Journalists on Hugging Face。📰🤗 https://huggingface.co/JournalistsonHF 1/</li><li><a href="https://x.com/reach_vb/status/1783129119435210836)">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：Snowflake 发布了 408B Dense + Hybrid MoE 🔥 > 17B 激活参数 > 128 个专家 > 在 3.5T tokens 上训练 > 使用 top-2 gating > 完全采用 Apache 2.0 许可（附带数据配方...）</li><li><a href="https://x.com/RisingSayak/status/1785162074844197174)">Sayak Paul (@RisingSayak) 的推文</a>：Diffusers 中的自定义流水线和组件 🎸 想要在 Diffusers 中使用自定义流水线和其他组件（schedulers, unets, text encoders 等）？觉得不够灵活？这个 🧶 线程就是为你准备的...</li><li><a href="https://x.com/lunarflu1/status/1785359306847666431)">lunarflu (@lunarflu1) 的推文</a>：你现在可以在 @huggingface 上 @ 别人了！
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1234770448382824468)** (151 条消息🔥🔥): 

- **Chronos 模型微调咨询**：一名成员寻求关于微调 [Chronos 时间序列预测模型](https://huggingface.co/amazon/chronos-t5-small) 的指导。他们被引导至 GitHub 仓库以获取更多细节。
- **Hugging Face 求职者**：一位拥有 10 年经验的软件工程师寻求在 Hugging Face 的工作机会，并被引导至 Hugging Face 的[职位空缺](https://apply.workable.com/huggingface/?lng=en)页面，包括一个自荐职位 (wild card position)。
- **使用 Rasa 框架开发聊天机器人的困难**：一位新成员在使用 Rasa 框架开发销售相关聊天机器人时遇到了意图识别准确率问题，并正考虑制作自定义 NER 模型。
- **Spaces 新手问题**：成员询问是否能收到 Space 社区线程中新回复的通知，得到的回复是系统默认会发送通知。
- **分享 Kaggle 和 Google Colab 技巧**：几位成员讨论了使用 Kaggle 和 Google Colab 的免费 GPU 进行模型训练，并交流了关于增加 VRAM 设置以及 Kaggle 手机验证以开启互联网访问的建议。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://apply.workable.com/huggingface/?lng=en">Hugging Face</a>：在 Hugging Face，我们致力于为每个人推进和民主化 ML。在此过程中，我们为技术的向好发展做出贡献。</li><li><a href="https://huggingface.co/spaces/Nick088/Stable_Diffusion_Finetuned_Minecraft_Skin_Generator">Stable Diffusion Finetuned Minecraft Skin Generator - Nick088 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/amazon/chronos-t5-small">amazon/chronos-t5-small · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/drax-guardians-of-the-galaxy-odds-bet-chance-gif-8058651">Drax Guardians Of The Galaxy GIF - Drax Guardians Of The Galaxy Odds - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/26">zero-gpu-explorers/README · 邀请申请一直在等待中。获得批准需要多长时间？</a>：未找到描述</li><li><a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">“我希望 Llama3 利用我的私有知识实现 10 倍性能” - 使用 llama3 的本地 Agentic RAG</a>：高级 RAG 101 - 使用 llama3 构建 agentic RAG。获取关于 AI 如何重新定义初创公司 GTM 策略的免费 HubSpot 报告：https://clickhubspot.com/4hx🔗 链接- F...</li><li><a href="https://github.com/johko/computer-vision-course">GitHub - johko/computer-vision-course：该仓库是社区驱动的神经网络计算机视觉课程的大本营。欢迎加入我们的 Hugging Face Discord：hf.co/join/discord</a>：该仓库是社区驱动的神经网络计算机视觉课程的大本营。欢迎加入我们的 Hugging Face Discord：hf.co/join/discord - johko/computer-vision-course</li><li><a href="https://github.com/amazon-science/chronos-forecasting?tab=readme-ov-file">GitHub - amazon-science/chronos-forecasting：Chronos：用于概率时间序列预测的预训练（语言）模型</a>：Chronos：用于概率时间序列预测的预训练（语言）模型 - amazon-science/chronos-forecasting</li><li><a href="https://github.com/huggingface/accelerate/pull/2732">nroggendorff 修复了一些 Sagemaker 配置问题 · Pull Request #2732 · huggingface/accelerate</a>：更新 config_args.py 以适配最新版本的 amazon sagemaker。在这个新版本中，你需要使用 True 或 False 来运行变量操作，例如 --do_eval True，而不是仅仅...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1234782066428608512)** (3 条消息): 

- **寻求微调指导**：一位成员表示有兴趣学习如何生成用于微调大语言模型 (LLM) 的指令数据集。
- **寻求澄清**：另一位成员询问了关于第一位成员提到的生成 LLM 微调指令数据集的具体细节。
- **介绍用于医学的 Med-Gemini**：一位成员分享了一个 [YouTube 视频](https://youtu.be/xohuoN2WBZs?si=Ku6cztykld6dZLN9)，提供了 **Med-Gemini** 的高层级概述，这是 Google 用于医学的多模态 GenAI 模型，旨在向感兴趣的各方介绍并确认该技术。

**提到的链接**：<a href="https://youtu.be/xohuoN2WBZs?si=Ku6cztykld6dZLN9">Med-Gemini：高层级概述</a>：关于 Med-Gemini 的高层级概述，这是 Google 用于医学的多模态 GenAI 模型“家族”（用 Vin Diesel 的声音说）。Med-Gemini 让人们...

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1234797591523229716)** (8 条消息🔥):

- **AI 爱好者的酷工具**：推荐了一篇名为 [“每个人都应该尝试的 5 个有趣的 AI 工具”](https://medium.com/illumination/genai-adventures-5-interesting-ai-tools-everyone-should-try-44ae8f8115af) 的 Medium 文章，列出了该领域中人们可能感兴趣的各种 AI 应用。
- **网页加载的未来**：Medium 上的一篇文章讨论了如何使用 Groq、Langchain 和 Datastax 创建强大的 Webloader RAG 应用程序，[在此阅读更多内容](https://medium.com/ai-advances/building-powerful-webloader-rag-applications-with-groq-langchain-and-datastax-f4816d88bee8)。
- **SQL 简化**：Data Intelligence Alliance 通过其网站 [www.dataialliance.org](https://www.dataialliance.org) 正在开发一个“人员数据库”，允许个人在几乎没有或完全没有 SQL 知识的情况下与数据库进行交互。
- **显微镜图像分割变得简单**：Micro-SAM 的 GitHub 仓库现已上线，该项目旨在简化显微镜图像的分割过程，可以点击[此处](https://github.com/computational-cell-analytics/micro-sam)查看。
- **加速扩散模型**：Hugging Face 文档详细介绍了在不牺牲效果的前提下加速扩散模型的几种技术，并强调了使用 PyTorch 2 如何将文本生成图像（text-to-image）流水线的推理速度提高三倍，特别是在 [Stable Diffusion XL (SDXL)](https://huggingface.co/docs/diffusers/tutorials/fast_diffusion) 上得到了验证。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.dataialliance.org">blog</a>：未找到描述</li><li><a href="https://huggingface.co/docs/diffusers/tutorials/fast_diffusion">加速文本生成图像扩散模型的推理</a>：未找到描述</li><li><a href="https://github.com/computational-cell-analytics/micro-sam">GitHub - computational-cell-analytics/micro-sam: Segment Anything for Microscopy</a>：显微镜领域的 Segment Anything。通过在 GitHub 上创建账号为 computational-cell-analytics/micro-sam 的开发做出贡献。</li><li><a href="https://youtu.be/IDIv92Z6Qvc?si=NlBDh0KtHNq63XvN">ETH Zürich DLSC: Physics-Informed Neural Networks - Applications</a>：↓↓↓ 课程概览如下 ↓↓↓ 苏黎世联邦理工学院（ETH Zürich）2023 年科学计算中的深度学习，第 5 讲：物理信息神经网络 - 应用。讲师：Ben M...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1234859558602670111)** (11 条消息🔥): 

- **无泄漏链接预测方法论**：一个名为 [PnPR-GCN_ACM_SAC_24](https://github.com/Lama-West/PnPR-GCN_ACM_SAC_24) 的 GitHub 仓库解决了传递图（transitive graphs）上 K-fold 交叉验证中的信息泄漏问题。所提出的方法论确保了数据划分时不会发生信息泄漏，从而增强了概念先决条件学习。

- **将调度与 AI 对齐**：
来自 [dstackai](https://twitter.com/dstackai/status/1785315721578459402) 的一条推文介绍了一份关于结合使用 Alignment Handbook 和 dstack 的指南，以促进在云端或本地机器上调度微调（fine-tuning）任务。

- **🤗 Spaces 上的迭代式 SDXL 局部重绘 (Inpainting)**：[inpainting SDXL sketch pad](https://huggingface.co/spaces/tonyassi/inpainting-sdxl-sketch-pad) 允许进行迭代式局部重绘并保留版本历史以恢复之前的图像版本，但目前该 Space 因处于非活动状态而进入休眠。

- **HDR 显示兼容性挑战**：提到图像采用 HDR 编码，建议全屏查看以获得正确的色彩呈现，特别是在 iOS/iPadOS 等设备上，否则图像可能会显得发白。

- **使用 Bloom 进行 55 种语言聊天**：[Bloom Multilingual Chat](https://huggingface.co/spaces/as-cle-bert/bloom-multilingual-chat) 是一个 Hugging Face Space，用户可以通过使用 `deep_translator` Python 库进行查询翻译和回译，与 Bloom 模型进行 55 种语言的对话。

- **批量处理你的 Moon Dreams**：MoonDream2 添加了新的批量处理功能，允许一次处理多张图像。点击[此处](https://huggingface.co/spaces/Csplk/moondream2-batch-processing)查看 MoonDream2 批量处理。

- **FluentlyXL V4 发布**：FluentlyXL V4 模型强调对比度、写实感和准确的解剖结构。你可以在 [Fluently Playground](https://huggingface.co/spaces/fluently/Fluently-Playground) 尝试这个增强模型。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/fluently/Fluently-XL-v4">fluently/Fluently-XL-v4 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/tonyassi/inpainting-sdxl-sketch-pad">Inpainting SDXL Sketch Pad - a Hugging Face Space by tonyassi</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Csplk/moondream2-batch-processing">moondream2-batch-processing - a Hugging Face Space by Csplk</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/as-cle-bert/bloom-multilingual-chat">Bloom Multilingual Chatbot - a Hugging Face Space by as-cle-bert</a>: 未找到描述</li><li><a href="https://github.com/Lama-West/PnPR-GCN_ACM_SAC_24/tree/main">GitHub - Lama-West/PnPR-GCN_ACM_SAC_24</a>: 通过在 GitHub 上创建账户，为 Lama-West/PnPR-GCN_ACM_SAC_24 的开发做出贡献。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1234763524392226856)** (18 messages🔥): 

- **Graph Papers Galore（大量图相关论文）**: 一位成员重点推荐了一篇待读论文，题为“图在表示复杂关系中起着重要作用”，可在 [arXiv:2404.14928](https://arxiv.org/abs/2404.14928) 查阅。他们还提到在考虑其他与图相关的综述，但希望避免过度分散注意力。

- **Distillation Insights on the Horizon（即将到来的蒸馏见解）**: 参与者讨论了 score-based models 中的蒸馏技术，提到 **Laion server** 汇聚了该领域的专家，并推荐了 Segmind 的论文，讨论了 *rectified/instaflow*、*lcm lora* 以及 *piecewise rectified flow*。

- **Reading Group Event Scheduled（读书小组活动已安排）**: 组织了一次读书小组活动，并为参与者提供了建议不同时间的链接，并注明会兼顾每个人的空闲时间。

- **NegotiationArena: A New Playground for LLMs（NegotiationArena：LLM 的新游乐场）**: 大家对一篇关于 Large Language Models (LLMs) 如何使用名为 **NegotiationArena** 的框架相互谈判的论文演示表示赞赏，该论文可在 [arXiv:2402.05863](https://arxiv.org/abs/2402.05863) 找到。

- **Negotiation as an LLM Alignment Metric（谈判作为 LLM 对齐指标）**: 一位成员评论了谈判任务作为一个评估 LLM alignment 潜在指标的独特之处，并认识到该任务不同于常规的下游任务。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.14928">Graph Machine Learning in the Era of Large Language Models (LLMs)</a>: 图在社交网络、知识图谱和分子发现等各个领域的复杂关系表示中起着重要作用。随着深度学习的出现，图神经网络...</li><li><a href="https://arxiv.org/abs/2402.05863">How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis</a>: 谈判是社会互动的基石；人类谈判从汽车价格到如何共享公共资源的一切。随着对使用大语言模型 (LLMs) 的兴趣迅速增长...</li><li><a href="https://discord.gg/hugging-face-879548962464493619?event=1234913780048203856">Join the Hugging Face Discord Server!</a>: 我们正致力于实现优秀机器学习的民主化 🤗 验证以链接您的 Hub 和 Discord 账户！| 77668 名成员</li><li><a href="https://arxiv.org/abs/2312.02783">Large Language Models on Graphs: A Comprehensive Survey</a>: 大语言模型 (LLMs)，如 GPT4 和 LLaMA，由于其强大的文本编码/解码能力和新发现的涌现能力，正在自然语言处理领域取得重大进展...</li><li><a href="https://arxiv.org/abs/2310.11829">Towards Graph Foundation Models: A Survey and Beyond</a>: 基础模型已成为各种人工智能应用中的关键组件，并在自然语言处理和其他几个领域展示了显著的成功。M...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1234761217084493844)** (17 messages🔥):

- **改进 YOLO 模型**：一位成员提到他们正在致力于提高 YOLO 架构的准确性，即使这意味着模型速度会变慢，并意识到修改架构可能非常耗时。
- **寻求 CNN 学习合作**：一位用户正在寻找合作伙伴，共同研究和学习卷积神经网络 (CNNs)。
- **YOLOv5 并行处理技巧**：建议在 YOLOv5 中使用滑动窗口方法进行并行化，并提议研究 YOLO/CNN 之前的图像分割和轮廓算法，暗示图像简化和下采样可以产生有效的结果。
- **PyTorch 与 TensorFlow 的学习曲线**：进行了一场关于学习 PyTorch 还是 TensorFlow 来处理 CNN 的讨论，大家公认 TensorFlow 的学习曲线更陡峭，尽管它提供了更多来自 Google 的 devops 支持，而 PyTorch 则拥有更多的学术支持和社区动力。
- **Kaggle 讨论与计算机视觉工具**：一位用户分享了一个 [Kaggle 讨论](https://www.kaggle.com/discussions/general/498337)链接，介绍了他们旨在辅助训练或微调 CV 模型的工作，并正在寻求反馈。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.opencv.org/3.4/d2/d96/tutorial_py_table_of_contents_imgproc.html">OpenCV: Image Processing in OpenCV</a>: 未找到描述</li><li><a href="https://www.kaggle.com/discussions/general/498337">3LC - Real-Time 3D Visualizer/Debugger/Data Editor for Training/Finetuning your Models - Free! | Kaggle</a>: 3LC - 用于训练/微调模型的实时 3D 可视化器/调试器/数据编辑器 - 免费！| Kaggle。</li><li><a href="https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html">OpenCV: Morphological Transformations</a>: 未找到描述</li><li><a href="https://docs.3lc.ai/3lc/latest/public-notebooks/pytorch-lightning-segformer.html">Training a finetuned SegFormer model with Pytorch Lightning - </a>: 未找到描述</li><li><a href="https://docs.3lc.ai/3lc/latest/public-notebooks/detectron2-balloons.html">Balloons Toy Dataset + Detectron2 + 3LC Tutorial - </a>: 未找到描述</li><li><a href="https://docs.3lc.ai/3lc/latest/user-guide/integrations/yolov5/yolov5.html">Integrating 3LC with YOLOv5 🚀 - </a>: 未找到描述</li><li><a href="https://docs.3lc.ai/3lc/latest/user-guide/integrations/yolov8/yolov8.html">Integrating 3LC with YOLOv8 🚀 - </a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1234885730283556907)** (5 messages): 

- **寻求 NLP 项目指导**：一位新成员正在开发 **chatbot** 项目，在使用 **Rasa framework** 进行意图识别时遇到困难。他们正在考虑创建一个自定义的 **NER model** 来识别与其业务相关的特定术语，并思考是“**制作自己的模型**”、使用 **Spacy**，还是利用来自 **HuggingFace** 的预训练模型来提高机器人的性能。
  
- **咨询 Ollama 模板角色**：另一位成员询问关于在 **Ollama template** 角色中添加“Reviewer”角色的问题，以便评估助手的回答格式，寻求如何通过模板实现这一点。他们参考了 [Transformers chat templating guide](https://huggingface.co/docs/transformers/main/es/chat_templating) 中的现有文档。

- **为大学科技俱乐部开发 Mini Emo 机器人**：一位成员正在为 **Mini bot** 构建 **NLP model**，旨在与口头提示交互、搜索特定信息并提供语音回答，可能部署在 **Raspberry Pi** 上。由于他们是 **NLP** 领域的新手，因此请求协助和指导。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/)** (1 messages): 

sayakpaul: 这可能更适合在 A1111 论坛上提问。
  

---


**HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1234862689357009087)** (1 messages): 

- **Gradio 分享服务器问题警报**：Gradio 目前的分享服务器（Share Server）出现问题，可能会影响 Colab 上的分享和使用。他们正在积极调查并解决该问题，用户可以[在此检查状态](https://status.gradio.app/)。
- **Gradio 状态透明度**：用户可以在其[状态页面](https://status.gradio.app/#)查看 Gradio 在不同时间段（包括过去 24 小时、7 天、30 天和 90 天）的运行正常率统计数据。
- **近期无更新**：截至过去 7 天，没有新的状态更新，但可以在[此处](https://status.gradio.app/#)查看历史事件记录。

**提到的链接**：<a href="https://status.gradio.app/">Gradio Status</a>：未找到描述

**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1234884877413777478)** (4 messages): 

- **财务助手 AI 突破**：一个新的财务助手现在可以在无需人工干预的情况下，针对**非结构化财务报告计算百分比演变、CAGR 和 P/E ratios**。关于构建这一强大工具的简要见解已通过[推文](https://t.co/6cTNxUBJcr)中链接的文章分享。

- **利用 Redis 增强 RAG 应用**：在 **Redisinc、@tchutch94 和 @seldo** 的合作中，了解如何通过语义缓存创建 Agentic Retrieval-Augmented Generation (RAG)。他们在[此资源](https://t.co/oGxFrZLMRn)中讨论了提高质量、效率和降低成本的方法。

- **关于使用 LlamaIndex 部署 AI 的 PulumiCorp 网络研讨会**：一场定于 5 月 8 日举行、由 *_ediri* 和 *@seldo* 主持的网络研讨会将深入探讨如何使用 Pulumi 将 AI 应用程序（重点是 LlamaIndex）部署到 AWS。关于利用基础设施即代码 (Infrastructure as Code) 开发 AI 应用程序的信息已在公告[推文](https://t.co/4IwBhVFEss)中分享。

- **发布最新的 LlamaIndex.TS 更新**：**LlamaIndex.TS 0.3 版本**已发布，增强功能包括对 ReAct、Anthropic、OpenAI 的 Agent 支持，以及通用的 AgentRunner 类、改进的 Web Streams 和更强大的类型系统。这些更新在介绍新版本优势的[推文](https://t.co/mBIrD9uh8c)中得到了强调。

**提及的链接**：<a href="https://t.co/oGxFrZLMRn">未找到标题</a>：未找到描述

---

**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1234764368504164404)** (130 messages🔥🔥): 

- **Max Tokens 与 Embedding 模型**：如果要嵌入的内容超过了 Max Token 限制，模型将只考虑前 max_length 个 Token 并忽略其余部分。如果 Embedding 模型的 Token 限制小于提供的数据，则可能需要进行内容分块 (Chunking)。

- **AzureOpenAI 的本地异步调用**：LlamaIndex 支持使用 `acomplete` 和 `astream_complete` 进行补全，以及使用 `achat` 和 `astream_chat` 进行聊天上下文，从而对 AzureOpenAI 进行异步调用 (Async Calls)。异步允许在不阻塞其他操作的情况下执行 API 调用等任务，从而提高性能。

- **带有源节点的实时摘要**：LlamaIndex 可以生成摘要并指示用于生成摘要的节点。优化此过程涉及改进 Prompt 以及利用 Source Nodes 信息来提高结果的相关性。

- **理解基于 MongoDB Atlas 的 RAG**：有关于在 LlamaIndex 中进行查询而无需重新上传文档并将其转换为节点的问题。回复指出，Embedding 模型对于将查询与索引数据进行比较以检索相关材料至关重要。

- **分析 LlamaIndex 与本地开发的缺点**：Ollama 在本地运行，与 OpenAI 等基于服务器的 API 相比可能较慢，但它为本地开发提供了隐私和成本优势。在 LlamaIndex 中创建和查询索引的过程中，使用 Embedding 模型是不可避免的。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://imgur.com/a/9uLmSxD">Summary and Resources</a>: 在 Imgur 探索互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行模因、娱乐 gif、励志故事、病毒视频等来振奋你的精神...</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/">Starter Tutorial (OpenAI) - LlamaIndex</a>: 未找到描述</li><li><a href="https://www.cloudraft.io/blog/content-moderation-using-llamaindex-and-llm">Content Moderation using AI</a>: 了解如何使用 AI 模型和框架（如 LlamaIndex、moondream 和 Microsoft phi-3）来审核内容。</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/openai#async>).">OpenAI - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/low_level/vector_store/?h=vectorstorequery">Building a (Very Simple) Vector Store from Scratch - LlamaIndex</a>: 未找到描述</li><li><a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">&quot;I want Llama3 to perform 10x with my private knowledge&quot; - Local Agentic RAG w/ llama3</a>: 高级 RAG 101 - 使用 llama3 构建 Agentic RAG。获取关于 AI 如何重新定义初创公司 GTM 策略的免费 HubSpot 报告：https://clickhubspot.com/4hx🔗 链接- F...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/chroma_metadata_filter/?h=metadatafilter">Chroma Vector Store - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/pipeline/query_pipeline_async#query-pipeline-with-asyncparallel-execution>),">Query Pipeline with Async/Parallel Execution - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/pipeline/query_pipeline_async#try-out-queries>).">Query Pipeline with Async/Parallel Execution - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/ingestion/parallel_execution_ingestion_pipeline#in-summary>),">Parallelizing Ingestion Pipeline - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1235227548582150145)** (6 messages): 

- **为 AI 任务选择合适的 GPU**：讨论围绕着像 **RTX 4080** 这样的游戏显卡是否适合运行和微调较小的语言模型展开。一位成员建议，虽然 VRAM 至关重要，但即使有 16GB 或 24GB，也不应指望在小 Batch Size 下微调大于 7B 的模型。

- **隐私考量下的本地计算 vs 云计算**：成员 **tuhe** 澄清说，对本地 PC 的需求源于处理敏感数据以及拥有用于工作的强大计算机的实用性，而不是像 Google Colab 这样可能存在隐私问题的云解决方案。

- **Word Loom 简介**：分享了一个名为 **Word Loom** 的新开放规范，旨在管理和交换 AI 语言，重点是代码与自然语言的分离以及组合性。欢迎对拟议的更新提供反馈，该更新旨在辅助传统的全球化流程，详细信息可在 [GitHub](https://gist.github.com/uogbuji/5bd08f74125934fa9e0d37236a8e168e) 上找到。

**提到的链接**：<a href="https://gist.github.com/uogbuji/5bd08f74125934fa9e0d37236a8e168e">Word Loom proposed update</a>：Word Loom 拟议更新。GitHub Gist：即时分享代码、笔记和片段。

  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1234813018806943758)** (22 messages🔥): 

- **澄清 Subreddit 混淆**：一位成员澄清说，在 [https://www.reddit.com/r/modular_mojo/](https://www.reddit.com/r/modular_mojo/) 有一个 Mojo 的 Subreddit，但 Mojo 社区主要在 GitHub 和 Discord 上交流。

- **并发模型推测**：社区讨论了 Mojo 采用并发模型的潜力，猜测它不会遵循 Golang 风格，而可能倾向于 [actor model](https://github.com/modularml/mojo/pull/1445#issuecomment-1849117416)，另一个观点则强调了不随语言发布庞大 Runtime 的重要性。

- **Mojo 编译器见解**：据分享，Mojo 的编译器是手写的，并重用了 LLVM 的部分内容，更多解释可以在标题为 "2023 LLVM Dev Mtg - Mojo 🔥: A system programming language for heterogenous computing" 的 [YouTube 视频](https://youtu.be/SEwTjZvy8vw)中找到。

- **Playground 中的类型声明错误**：有人提出了一个关于使用 'ui64' 作为类型声明时出现错误消息的问题，困惑于是否支持像 Zig 中那样的自定义位宽整数，并指出 `Int64` 可以工作但 `Int128` 不行。

- **Mojo 一周年回顾**：成员们回顾了 Mojo 发布一周年，强调了 traits、references 和 lifetimes 的加入是重大成就，释放了标准库的巨大潜力。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/engine/reference/cli/input-data-schema#data-types:~:text=ui64%3A%20unsigned%20integer%20with%20bitwidth%2064.">输入数据架构 | Modular 文档</a>：以下 YAML 架构允许您指定所需的输入形状</li><li><a href="https://github.com/modularml/mojo/pull/1445#issuecomment-1849117416)">由 reid-spencer 提出的基于 Mojo 的 Actor 系统提案 · Pull Request #1445 · modularml/mojo</a>：这目前是一个正在进行中的工作。没有代码更改，只是在提案部分写了一个提案。这在 2023 年 6 月的一次对话中得到了 Chris Lattner 的预先批准。我将继续...</li><li><a href="https://youtu.be/SEwTjZvy8vw)">2023 LLVM 开发者大会 - Mojo 🔥：一种用于异构计算的系统编程语言</a>：2023 LLVM 开发者大会 https://llvm.org/devmtg/2023-10------Mojo 🔥：一种用于异构计算的系统编程语言。演讲者：Abdul Dakkak, Chr...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1235010692721938432)** (4 条消息): 

- **Modular 发布神秘推文**：分享了 Modular 最新的 [推文](https://twitter.com/Modular/status/1785447336812101796)，但消息中未说明具体内容。

- **另一个 Modular 更新出现在 Twitter**：通过分享的链接查看来自 [Modular 的最新更新](https://twitter.com/Modular/status/1785447397189161006)。

- **Modular 分享加密信息**：发布了一条来自 [Modular 的新推文](https://twitter.com/Modular/status/1785447412376764507)；此处未描述推文详情。

- **Modular 继续在 Twitter 上预热**：有一条来自 [Modular 的新推文](https://twitter.com/Modular/status/1785720385889243286) 可能令人感兴趣；消息中未包含推文的具体细节。
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1234882195034869781)** (58 条消息🔥🔥): 

- **Julia 的 `@time` 宏深受欢迎**：一位成员称赞了 Julia 的 `@time` 宏能够显示内存分配 (allocations) 的能力，并表示希望在 Mojo 中看到类似功能。
- **'None' 实现之谜**：关于 Mojo 中 `None` 如何实现的搜索引发了困惑和一段指向 GitHub 的讨论。该询问强调了一个关于 `None` 未实现 `__is__` 和 `__isnot__` 方法的错误。
- **对 Mojo 语法的赞赏**：一位用户在评估了各种编程语言后，称赞 Mojo 的语法几乎完美契合了他们理想中的语言语法。
- **讨论 Mojo 中的引用传递**：关于在 struct 中使用 `inout` 和 Mojo 中的 `Reference` 类型的对话澄清了 `inout` 确实像 C++ 一样进行引用传递，但在 Mojo 中有所不同。讨论包括了代码示例，并强调了正在进行的使引用更加优雅的开发工作。
- **Mojo 开发更新与问题**：多条消息涉及了 Mojo 的开源进展、对 Windows 版本的期待，以及确保 Mojo 保持用户友好且易于理解，而不陷入 Rust lifetime 系统的复杂性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/search?q=repo%3Amodularml%2Fmojo+%22None%22&type=code&p=0)">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://youtu.be/kgUXfDpAmGQ?si=VmrPUT7YLBmzMq8I">C++ 作为优化汇编器 - 性能演讲 - Levo DeLellis - CppNorth 2023</a>：https://www.cppnorth.ca​---C++ 作为优化汇编器 - 性能演讲 - Levo DeLellis - CppNorth 2023 您是否厌倦了抽象、模板和协...</li><li><a href="https://rosettacode.org/wiki/99_Bottles_of_Beer/EsoLang">99 Bottles of Beer/EsoLang</a>：未找到描述
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1235000164813639711)** (1 条消息):

- **Mojo 贡献者招募**：一位成员发出了贡献 Mojo 的邀请，建议包括允许负数、为标量处理实现回退机制，以及探索 Issue 链接文章中提到的快速绝对容差（fast absolute tolerances）。目前尚未设定具体计划，为实验性贡献留出了空间。
- **识别 Mojo 缺失的组件**：Mojo 目前缺少 [PMADDUBSW](https://www.felixcloutier.com/x86/pmaddubsw) 指令，该指令对于快速 SIMD `atol`（ASCII 转长整型）至关重要，目前导致需要约 4 个 SIMD 操作的变通方案。此特性是 x86 特有的，在 ARM 架构上不支持。

**提到的链接**：<a href="https://www.felixcloutier.com/x86/pmaddubsw">PMADDUBSW — 乘加打包的有符号和无符号字节</a>：未找到描述

---

**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1235261267250511923)** (3 条消息): 

- **Mojo 语言引发热议**：一段由 Chris Lattner 出镜的新 [YouTube 视频](https://youtu.be/JRcXUuQYR90) 讨论了 **Mojo Lang**，这是一种利用 CPU/GPU 编程技术、有望成为 Python 高性能继任者的语言。
- **对编程语言播客的热爱**：一位成员表达了对该播客的喜爱，分享了他们对编程语言讨论的兴奋之情，并在内部传播了相关内容。

**提到的链接**：<a href="https://youtu.be/JRcXUuQYR90)">Mojo Lang - 未来的高性能 Python？（对话 Chris Lattner）</a>：Mojo 是由 Swift 和 LLVM 创始人推出的最新语言。它尝试吸取 CPU/GPU 级编程的一些最佳技术并进行封装...

---

**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1234776260509569066)** (7 条消息): 

- **呼吁组建 Team-Mojo 参加 1BRC**：一位成员建议组建 **Team-Mojo** 来应对“十亿行挑战赛”（One Billion Row Challenge, 1brc），以此作为技术展示和教程。

- **Mojo 性能优化**：在优化了字符串分配和转换后，一位成员报告称 1 亿条记录的处理时间从 8 秒减少到 1.3 秒，目前的瓶颈在于 hashmap，使总时间从 18.5 秒降至 12.5 秒。该实现仅在 Mojo nightly 版本中运行，可以在 [GitHub](https://github.com/MoSafi2/1brc-mojo/tree/dev) 上找到。

- **对组建 Team-Mojo 的热情**：成员们对组建 Team-Mojo 表现出极大的热情，表示这将是一个非常有趣的项目。

- **参考 Benchmarks Game**：有人建议也考虑 [benchmarks game](https://github.com/modularml/mojo/discussions/843#discussioncomment-7045479)，这是团队之前未完成的一项任务。

- **多核处理更新**：一位成员在更新了支持多核处理的工作后提交了 pull request，指出性能显著提升，现在处理 1 亿条记录仅需 3.8 秒。另一位成员邀请对该更新进行进一步审查，并提到他们打算根据 `atol-simd` 的经验来研究 `atol` 函数。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/discussions/843#discussioncomment-7045479)">Mojo 比 Python 快 68,000 倍之类的博客很棒，但能否与其他语言也进行精彩的对比？ · modularml/mojo · Discussion #843</a>：Mojo 比 Python 快 35,000 倍、快 68,000 倍……这令人印象深刻且酷炫，但对于那些尚未关注 Mojo 的非 Python 用户和反 Python 人士来说……</li><li><a href="https://github.com/MoSafi2/1brc-mojo/tree/dev">GitHub - MoSafi2/1brc-mojo at dev</a>：使用 Mojo 语言实现的十亿行挑战赛 (1brc)。通过创建账号为 MoSafi2/1brc-mojo 的开发做出贡献。
</li>
</ul>

</div>

---

**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1234763287900721172)** (20 条消息🔥):

- **Order Swap Still Buggy**：一位成员提到，尽管修复了初始问题，但更改某些内容的顺序仍然会导致其崩溃。
- **Considering the Future of `bool` in Code**：关于可能将 `bool` 的使用限制为大小 1 的详细观点被表达出来，强调了在编程中保留 `bool` 作为 primitive 的重要性，并理解这种变化的影响。
- **SEMANTICS: Could `simd ternary` mimic `select`?**：一位成员询问 `simd ternary` 是否可以像 `select` 一样工作，另一位成员指出，甚至 `if` 语句的语义在某种程度上也取决于“boolable”的概念。
- **WANTED: Missing `__source_location()` Function**：对话涉及对 `__source_location()` 函数消失的困惑，并建议它可能已被 `__call_location()` 取代。这可以通过 [SourceGraph 搜索链接](https://sourcegraph.com/search?q=context:global+__source_location()&patternType=keyword&sm=0&filters=%5B%5B%22type%22,%22Code%22,%22type:file%22%5D%5D) 看到，该话题得到了进一步讨论，包括具体的代码示例和 GitHub 文档[链接](https://github.com/modularml/mojo/blob/nightly/stdlib/src/testing/testing.mojo)。
- **Function Names in Source Location**：一位成员质疑 `__source_location()` 函数输出中缺少 `function_name`，并暗示其他人也有同样的顾虑。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://sourcegraph.com/search?q=context:global+__source_location()&patternType=keyword&sm=0&filters=%5B%5B%22type%22,%22Code%22,%22type:file%22%5D%5D">context:global __source_… - Sourcegraph</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/testing/testing.mojo">mojo/stdlib/src/testing/testing.mojo at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1234777701626286171)** (23 messages🔥): 

- **Clarifying Tokenizer Behavior**：成员们讨论了在 chat template 中包含 **Beginning of Sentence (BOS)** token 如何影响 encoding，指出 `tokenizer.encode("text")` 会自动添加 BOS，但 `tokenizer.apply_chat_template(chat)` 需要在模板中明确指定。
- **Debating the Value of a Study**：分享了一个[近期研究](https://arxiv.org/abs/2311.16452)的链接，引发了关于其有用性的辩论。一位成员赞扬了其使用 cosine-similarity embeddings 的 prompting 策略，而另一位成员则认为该研究的方法对于 benchmarks 来说过于复杂。
- **The Practical Struggles with Model Tokens**：用户表达了对将新论文投入实践的挫败感，特别是尽管有大量的学术出版物，但弄清楚模型的 tokens 仍然具有挑战性。
- **Discussing User Input Masking Strategies**：出现了一个关于在训练期间 *masking out user inputs* 最佳实践的技术问题：是仅 mask 消息还是也 mask 指令标签，以及如何确保正确学习格式而不是用户的打字风格。
- **Prompting Approaches and Generalist Models**：简要触及了复杂 prompting 策略的相关性，以及在评估 AI 在 benchmarks 上的性能时，仅将技术应用于 generalist models 是否在某种程度上偏离了重点。

**提及的链接**：<a href="https://arxiv.org/abs/2311.16452">Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine</a>：诸如 GPT-4 之类的通用基础模型在广泛的领域和任务中展示了惊人的能力。然而，人们普遍认为它们无法与专业领域的微调模型相媲美……

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1234796962575024128)** (2 messages): 

- **Offer for Compute Help in Triage**：一位成员提议通过提供算力资源来协助 **triage/troubleshooting of bugs/issues**。他们强调这种帮助对项目和他们的心理健康都非常有价值。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1234845235872403546)** (14 messages🔥):

- **Phi3 Finetuning Underway**: 一些成员目前正在进行 **phi3** 的微调。建议其他寻求示例或想进一步探索的成员搜索频道历史记录以获取相关细节。
- **Dataset Format Wrangling for ShareGPT Loader**: 一位寻求微调模型的成员分享了一个按照 OpenAI 格式构建的 JSON 数据集示例，并收到了如何将其转换为 **ShareGPT loader format** 的指导。建议将 `"messages"` 替换为 `"conversations"`，`"role"` 替换为 `"from"`，`"content"` 替换为 `"value"`，`"user"` 替换为 `"human"`，以及 `"assistant"` 替换为 `"gpt"`。
- **Simplified Script for Dataset Conversion**: 为了将数据集适配到所需格式，提供了一个脚本，该脚本可以自动替换键并映射输入 JSON 结构中的角色，以匹配 **ShareGPT** 的预期格式。
- **Choose the Right LLaMA Model for Finetuning**: 在关于微调 LLaMA 模型的讨论中，建议避免微调 **Meta-LLaMA-3-70B-Instruct** 变体，因为它已经过指令微调，使用新格式可能会导致性能下降。同时建议初学者在尝试更复杂的 70b 变体之前，先从 **8b model** 开始。
- **FS-DP Compatibility Query for Lora**: 一位成员在遇到模型加载后训练挂起的问题后，询问了关于将 **fsdp** 与 **lora**（而非 **qlora**）结合使用的问题。建议表明可能只有 **qlora** 与他们的 fsdp 设置兼容。
- **LLaMA Model's Lengthy Output Concerns**: 一位用户报告称，他们的 **LLaMA 3 8b instruct** 模型在经过常规人类对话训练后，会产生冗长的输出和句子。他们思考是否某些 token（如 end-of-text 或标点符号）需要额外的训练，或者更多的数据和 epoch 才是解决此问题的关键。

**Link mentioned**: <a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/conversation.html#sharegpt.load_role)">Axolotl - Conversation</a>: 未找到描述

  

---


**OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/)** (1 messages): 

gbourdin: 添加到我的书签。感谢分享！
  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1234879220686258296)** (2 messages): 

- **Axolotl Meets dstack**: 分享了一个演示如何将 **axolotl** 与开源编排器 **dstack** 结合使用的教程。它允许在任何云端或本地机器池上微调 AI 模型，并可在 [GitHub](https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md) 上获取。
- **Community Approves**: 一位社区成员对分享的教程做出了积极回应，评价其易于使用。

**Link mentioned**: <a href="https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md">dstack/examples/fine-tuning/axolotl/README.md at master · dstackai/dstack</a>: 一个用于在任何云或数据中心运行 AI 工作负载的开源容器编排引擎。https://discord.gg/u8SmfwPpMd - dstackai/dstack

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1234798037612625921)** (51 messages🔥): 

- **Command-r Model Fine-tuning Discussed**: 成员们探讨了微调 command-r 模型的方法，建议使用 `runpod` 模板或手动实现不支持的格式。有人建议参考 [GitHub 上的一个未测试 PR](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547)，以将 **command-r** 模型添加到 Axolotl。

- **Fine-tuning Clarifications Provided**: 确认了如果特定参数（如 **sample packing**）不兼容，它们在过程中会被直接忽略。这导致了关于为什么训练任务耗时异常长的困惑。

- **Axolotl Format Capabilities Queried**: 存在关于 Axolotl 是否支持 **phi-3 format** 和 **GaLore** 的疑问，Phorm 回复称 Axolotl 不支持 phi-3 但支持 GaLore，启用细节可以在 [Hugging Face 文档](https://github.com/huggingface/transformers/tree/main/docs/source/en/trainer.md)中找到。

- **Model Adaptation Features and Functions**: 通过对话暗示，在 Axolotl 中适配模型可能涉及自定义代码调整，熟悉 GitHub 上的项目资源对于启用或配置特定功能（如 GaLore）非常有益。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547">Feat: Add cohere (commandr) by NanoCode012 · Pull Request #1547 · OpenAccess-AI-Collective/axolotl</a>: 描述 动机与背景 如何进行测试？ 未测试！ 截图（如适用） 变更类型 社交账号（可选）</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=83b91c9b-bb5c-4485-894c-0b878d17f7e2)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快速地理解代码。</li><li><a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/README.md#L77L100)">axolotl/README.md at main · OpenAccess-AI-Collective/axolotl</a>: 尽管提出 axolotl 问题。通过在 GitHub 上创建账号，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=1f87fb72-80ec-4321-b37b-d7574206e8af)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快速地理解代码。</li><li><a href="https://github.com/huggingface/transformers/tree/main/docs/source/en/trainer.md#L255L385)">transformers/docs/source/en/trainer.md at main · huggingface/transformers</a>: 🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的前沿机器学习。 - huggingface/transformers</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=dbfe64c8-e886-4d35-98e6-190287b3cd3c)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快速地理解代码。
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1234772710983340033)** (60 messages🔥🔥): 

- **AI 对服务条款的合规性**：一位参与者质疑了个体在未同意条款的情况下使用 AI 产品的情况。这引发了关于用户协议及其执行方式的问题。
- **呼吁建立新的透明 AI 排行榜**：一位用户表达了对更透明的新 AI 模型排行榜的需求。他们主张排行榜应**仅包含可验证的开源模型**，并能够按**开放权重 (open weights)** 过滤结果。
- **对 LMSYS 客观性和数据实践的担忧**：人们对 LMSYS 管理的 **Chatbot Arena** 排行榜的客观性提出了多项担忧；讨论涉及**利益冲突**以及模型评分处理缺乏透明度的问题。
- **关于 AI 模型和数据集的咨询与分享**：用户寻求有关 AI 生成的**国际象棋数据集**的更多信息，并分享了对各种模型性能的看法，例如 **llama3 70b** 在量化为 4-bit 时的能力。
- **技术难题与开发分享**：参与者分享了正在进行的项目链接，如 [magvit2](https://github.com/lucidrains/magvit2-pytorch)，并讨论了优化技术，包括何时使用 GAN 以获得更好的模型重建效果，以及为了提高效率而采用的 **Natten 新的 fused cuda 实现**。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmsys.org/blog/2024-03-01-policy/">LMSYS Chatbot Arena: Live and Community-Driven LLM Evaluation | LMSYS Org</a>: &lt;h2&gt;&lt;a id=&quot;our-mission&quot; class=&quot;anchor&quot; href=&quot;#our-mission&quot; aria-hidden=&quot;true&quot;&gt;&lt;svg aria-hidden=&quot;true&quot; class=&quot;octicon octicon-link&...</li><li><a href="https://xiaoyushi97.github.io/Motion-I2V/">Motion-I2V</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/lmsys/lmsys-chat-1m">lmsys/lmsys-chat-1m · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1234907832730783826)** (25 messages🔥): 

- **心脏超声 AI 研究发表**：一位成员宣布了他们关于心脏超声 **OpenCLIP 微调**研究的发表，尽管承认论文存在一些问题。该研究在经历了 8 个月的修订过程后，现已发表在 [Nature Medicine](https://doi.org/10.1038/s41591-024-02959-y) 上。

- **挑战 StableDiffusion 的可持续性**：讨论涉及一个 GitHub 仓库 [zer0int/CLIP-fine-tune](https://www.reddit.com/r/StableDiffusion/comments/1cgyjvt/github_zer0intclipfinetune_or_sdxl_training_the/)，并关联到对 Reddit 关闭开放 API 访问的担忧，这具有广泛的影响，包括影响应用开发者和盲人用户。

- **Kolmogorov-Arnold 网络优于 MLP**：一篇新论文提出了 **Kolmogorov-Arnold Networks (KANs)**，通过在边上利用可学习的激活函数作为样条 (splines)，在准确性和可解释性方面优于多层感知器 (Multi-Layer Perceptrons)。这一概念引起了成员的共鸣，认为该方法非常有前景（[阅读 arXiv 论文](https://arxiv.org/abs/2404.19756)）。

- **VisualFactChecker 助力增强型字幕生成**：另一篇论文介绍了 **VisualFactChecker (VFC)**，这是一个无需训练的 pipeline，通过引入事实核查显著改进了图像和 3D 物体的字幕生成，可能解决内容幻觉（hallucination）等问题。该研究详细介绍了提高自动字幕生成保真度和细节的方法（[查看 arXiv 论文](https://arxiv.org/abs/2404.19752)）。

- **请求国际象棋数据集生成详情**：为了寻找更好的训练数据，一位成员请求提供用于生成 **LAION stockfish dataset** 的配置详情，以评估其是否足以训练他们的国际象棋机器人，或者是否需要生成额外的数据集。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.19752">Visual Fact Checker: Enabling High-Fidelity Detailed Caption Generation</a>：现有的视觉内容自动字幕生成方法面临细节缺失、内容幻觉和指令遵循能力差等挑战。在这项工作中，我们提出了 VisualFactChecker (VFC)...</li><li><a href="https://arxiv.org/abs/2404.19756">KAN: Kolmogorov-Arnold Networks</a>：受 Kolmogorov-Arnold 表示定理启发，我们提出了 Kolmogorov-Arnold Networks (KANs) 作为 Multi-Layer Perceptrons (MLPs) 的有力替代方案。虽然 MLPs 具有固定的激活函数...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1cgyjvt/github_zer0intclipfinetune_or_sdxl_training_the/">Reddit - 探索一切</a>：未找到描述</li><li><a href="https://doi.org/10.1038/s41591-024-02959-y">Vision–language foundation model for echocardiogram interpretation - Nature Medicine</a>：一个在超过 100 万个超声心动图视频-文本对数据集上训练的视觉-语言基础模型，能够评估各种心脏结构和功能参数...
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1234837931349245973)** (70 条消息🔥🔥): 

- **Prime Intellect 的去中心化 AI 训练**：Prime Intellect 探索了新型去中心化训练方法，以追赶大科技公司对 **GPU clusters** 的扩张。欲了解深入分析，请[阅读其博客文章](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training)，其中讨论了开源 AI 社区面临的挑战以及其平台聚合全球算力资源的目标。

- **AI Agent 还是翻译机？**：一位成员辩论了 AI Agent 的概念，建议将语言模型视为使用共享上下文和记忆的“翻译机”，而无需出于多种原因进行并行化。

- **Starcoder2-Instruct 发布**：Hugging Face 推出了 StarCoder2-15B-Instruct-v0.1，这是一个用于代码生成的自对齐 **Large Language Model (LLM)**。底层的 pipeline 和模型均开源且采用宽松许可，详见其[发布页面](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)。

- **带有世界编辑器的 AI Town**：用户分享了一个[实验性设置](https://x.com/cocktailpeanut/status/1785702250599371088?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)，涉及 300 个 AI Agent 在名为 AI Town 的模拟世界中运行，在 MacBook M1 Max 上运行流畅。

- **Lilian Weng 富有洞见但极具挑战性的博客文章**：一些成员表示对 Lilian Weng 博客文章的深度和复杂性感到压力，特别是 Transformer Family 2.0 这篇文章，质疑是否需要投入全职学习才能掌握所分享的概念。
<div class="linksMentioned">

<strong>提到的链接</strong>：

 封面图](https://images.lumacdn.com/cdn-cgi/image/format=auto,fit=cover,dpr=2,quality=75,width=400,height=400/event-covers/mq/b7a9e5d5-cbd9-4546-a668-972d498d2186)

**提到的链接**：<a href="https://lu.ma/oz8e9z3r">LLM Paper Club (Ring Attention!) · Zoom · Luma</a>：StrongCompute 团队 (@adam_peaston, @fennecs) 今天将讲解 Ring Attention！https://arxiv.org/abs/2310.01889 同时也请为我们的下一篇论文提交建议并投票：…

  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1235305008581312656)** (2 条消息): 

- **分享了 Zoom 会议链接**：为偏好视频通话替代方案的人员提供了 Zoom 会议链接。可以通过 [Zoom Meeting](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09) 访问。

**提到的链接**：<a href="https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可跨移动设备、桌面和会议室系统进行视频和音频会议、聊天和网络研讨会。Zoom ...

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1234866310773735454)** (36 条消息🔥):

- **促进积极的社区互动**：发布了一项提醒，强调随着社区的发展和多样化，保持尊重和建设性的重要性。会议强调，*每个人都有平等分享想法的权利*，并应受到良好对待，以构建更好的未来。
- **活动提醒与回顾查询**：分享了一个社区活动的链接，错过活动的成员询问了回顾内容。提到幻灯片和屏幕录像将会发布，幻灯片将上传至特定频道。
- **Open Interpreter 的 Web 任务能力**：成员们讨论了 Open Interpreter 是否可以执行浏览器任务，如访问网站和抓取数据。澄清了它确实能够执行此类任务，而无需浏览器控制。
- **讨论兼容性与技术问题**：关于 Open Interpreter 的 OS 模式与 Windows 兼容性的问题浮出水面，并提到了持续存在的错误。一位成员确认某些命令在 Windows 上需要修改，并提到 'tesseract' 包是导致问题的原因。
- **分享有用资源**：推荐了一个 YouTube 频道作为获取 Open Interpreter 相关见解和更新的有用资源，并附带了该频道的直接链接。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/SdwpMQaW?event=1232436050165764096">Join the Open Interpreter Discord Server!</a>: 一种使用电脑的新方式 | 8840 members</li><li><a href="https://discord.gg/9rjF24Gz?event=1228030976706220072">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。与你的朋友和社区聊天、聚会并保持紧密联系。</li><li><a href="https://youtube.com/@MikeBirdTech?feature=shared">Mike Bird</a>: AI 工程  
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1234781691109703732)** (31 messages🔥): 

- **探索外部按钮**：成员们讨论了将外部按钮与硬件集成的相关问题，特别是针对 **Atom Echo** 设备。分享了代码修改方案，特别是 **ButtonChecker** 的代码片段，一位实施该方案的成员确认这解决了问题。

- **通过外部硬件放大音频**：一位成员提供了增加连接到硬件的扬声器音量的解决方案，建议使用外部放大器，并提供了一个[潜在放大器的链接](https://www.amazon.com/dp/B01DKAI51M)，但指出他们尚未测试此设置。

- **开箱 AI 创新**：频道中提到了 **MKBHD** 对 AI 产品 Rabbit R1 的 **YouTube 评测**，并附带了[视频链接](https://www.youtube.com/watch?v=ddTV12hErTc&ab_channel=MarquesBrownlee)。关于传统科技评论员在理解和评估非主流 AI 设备方面的有效性展开了辩论。

- **将 R1 连接到 OpenInterpreter**：对话围绕将 **R1 与 OpenInterpreter (OI)** 集成的想法展开，成员们讨论了对此的期待和计划。大家渴望探索这些工具如何协同工作，希望能扩展功能并构建创新的配置。

- **为 OI 定制 ngrok 域名**：一位成员分享了在 **ngrok** 上创建新域名并编辑 01 软件中 **tunnel.py** 文件的具体步骤，以解决服务器连接问题，并提供了 [ngrok 域名页面的直接链接](https://dashboard.ngrok.com/cloud-edge/domains)。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dashboard.ngrok.com/cloud-edge/domains">ngrok - Online in One Line</a>: 无描述</li><li><a href="https://www.youtube.com/watch?v=ddTV12hErTc&ab_channel=MarquesBrownlee">Rabbit R1: Barely Reviewable</a>: 盒子里的 AI。但是一个不同的盒子。在 https://dbrand.com/rabbit 获取 dbrand 皮肤和屏幕保护贴。MKBHD 周边：http://shop.MKBHD.com 我现在使用的科技产品...</li><li><a href="https://www.amazon.com/dp/B01DKAI51M">Amazon.com: HiLetgo Mini 3W+3W DC 5V Audio Amplifier Handy Digital Power Amp Module Board Dual-Channel PAM8403 Stereo Amplifiers with Potentiometer for DIY Portable : Electronics</a>: 无描述
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1235358580249591909)** (2 messages):

```html
<ul>
    <li><strong>Snowflake Arctic 480B 和 FireLLaVA 13B 模型发布</strong>：宣布新模型 <strong>Snowflake Arctic 480B</strong>，采用混合 Transformer 架构，擅长编程，可在 <a href="https://openrouter.ai/models/snowflake/snowflake-arctic-instruct">Snowflake Arctic 480B</a> 获取；以及 <strong>FireLLaVA 13B</strong>，由 Fireworks 开发的开源多模态模型，可在 <a href="https://openrouter.ai/models/fireworks/firellava-13b">FireLLaVA 13B</a> 获取。两者都为开发者提供了新的定价和详细规格。</li>
    <li><strong>改进的负载均衡和详细的提供商统计数据</strong>：OpenRouter 引入了 <strong>load balancing</strong>（负载均衡）来管理提供商的负载激增，现在允许监控延迟和提供商的结束原因（finish reasons），提升了用户性能，可在 <a href="https://openrouter.ai/activity">Activity 页面</a> 查看。</li>
    <li><strong>为开发者精简的文档</strong>：更新了关于图像和多模态请求（multimodal requests）、以及工具调用（tool calls）和函数调用（function calling）的文档，现已在 <a href="https://openrouter.ai/docs#images-_-multimodal-requests">图像请求</a> 和 <a href="https://openrouter.ai/docs#tool-calls">工具调用</a> 页面提供使用指南。</li>
    <li><strong>功能扩展和价格调整</strong>：宣布在 Lepton 模型上支持 <strong>logit_bias</strong> 和 <strong>min_p</strong>，Mythomax Extended 大幅降价 40%，Mixtral 8x7b Instruct 小幅降价 4%。这些变化体现了 OpenRouter 致力于提供高性价比和先进的 AI 能力。</li>
    <li><strong>即将到来的 API 变更和开发者通知</strong>：提醒开发者，非流式补全（non-streaming completions）中的 <code>total_cost</code> 字段即将移除，并且请求中可能要求包含 <code>User-Agent</code> 请求头，以提高服务安全性和效率。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://omnigpt.co/">OmniGPT - 最实惠的 ChatGPT 替代方案</a>：我们以实惠的价格为您提供市场上最好的模型：Claude 3, GPT 4 Turbo, GPT 4, Gemini, Perplexity 等。</li><li><a href="https://syrax.ai/">Syrax AI - 在一个平台上利用多个 AI</a>：通过 Syrax AI，您可以从一个平台访问多个 AI 模型来生成内容、图像等。</li><li><a href="https://openrouter.ai/models/snowflake/snowflake-arctic-instruct">Snowflake: Arctic Instruct by snowflake | OpenRouter</a>：Arctic 是由 Snowflake AI 研究团队从零开始预训练的稠密 MoE 混合 Transformer 架构。Arctic 结合了一个 10B 稠密 Transformer 模型和一个残差 128x3.66B MoE MLP 结果...</li><li><a href="https://openrouter.ai/models/fireworks/firellava-13b">FireLLaVA 13B by fireworks | OpenRouter</a>：首个商业许可的开源 LLaVA 模型。该视觉语言模型完全基于开源 LLM 生成的指令遵循数据进行训练。</li><li><a href="https://openrouter.ai/docs#images-_-multimodal-requests">OpenRouter</a>：构建与模型无关的 AI 应用</li><li><a href="https://openrouter.ai/docs#tool-calls">OpenRouter</a>：构建与模型无关的 AI 应用
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1235131318954623038)** (1 条消息): 

- **Skribler - 瑞典作家的 AI 助手**：几周前发布的 **Skribler** 是一款针对瑞典作家的新工具，通过 OpenRouter 集成了多种模型用于不同的写作任务。可在 [skribler.se](https://skribler.se) 访问，提供诸如为文本段落生成建议、帮助填补写作空白、构思对话以及对创意写作过程的全面支持等功能，介绍视频见 [此处](https://youtu.be/2Q2hb6UqGo4)。
- **积极的反响和用户采用**：**Skribler** 的发布还提到它已经获得了一批付费用户，表明在其目标市场受到了积极认可。

**提到的链接**：<a href="https://skribler.se">Skribler | Skriv med AI</a>：未找到描述

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1234817884748775435)** (64 条消息🔥🔥):

- **OpenRouter 日志查询**：成员们在询问是否可以在 OpenRouter 开启日志记录的情况下查看每个请求的 prompt 和输出。
- **模型 Embedding 能力咨询**：一位成员咨询了 OpenRouter 中支持 embedding 的模型可用性。
- **上下文扩展的好奇**：有一场关于模型上下文窗口扩展的讨论，特别提到了一个上下文长度扩展到超过 100 万的模型，以及关于在 [Hugging Face](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k) 上提供的扩展版 LLama-3 8B 模型性能的讨论。
- **支付问题及解决方案讨论**：用户正在讨论在 OpenRouter 上使用预付信用卡的问题，提到某些卡可能被 Stripe 的欺诈检测拦截，并讨论了潜在的解决方案或支付替代方案。
- **流式取消与模型回退**：有关于 OpenRouter 中流式取消（stream cancellation）可靠性的提问，以及建议使用 AWS 作为 Claude 模型的潜在回退（fallback）方案，类似于 Azure 用于 OpenAI 模型的方式。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/hsiehjackson/RULER">GitHub - hsiehjackson/RULER: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models?</a>：此仓库包含 RULER 的源代码：你的长上下文语言模型的真实上下文大小是多少？ - hsiehjackson/RULER
</li>
</ul>

</div>
  

---



**AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/1235256845418106990)** (28 messages🔥): 

- **清晰的 Diffusion 模型输出**：一位成员提到来自 [Hexagen World](https://www.hexagen.world/) 的 **diffusion 模型输出**非常清晰，标志着高质量的结果。

- **使用生成式 AI 重塑复古游戏**：有建议认为使用生成式 AI (GenAI) 重制像 **Farmville** 这样的早期社交媒体游戏将是一个引人注目的概念，而 WebSim 可能是实现这一目标的最佳平台。

- **嵌入 AI 的怀旧小镇模拟**：一位成员表示有兴趣在 WebSim 中建立一个 1950 年代主题的 AI 小镇，其中一个角色是共产主义间谍，创造一个互动的**猫鼠游戏**。

- **互动动画与 AI 讨论**：对 **AI 动画**感兴趣的参与者被邀请通过提供的 [Discord 邀请链接](https://discord.gg/deforum)加入相关的 Discord 社区。

- **Hexagen World 的发现与分享**：互动 AI 概念 **Hexagen World** 在社区内被分享，该概念通过 [bennyj504 的 Twitter 帖子](https://x.com/bennyj504/status/1785664502903570568)发现，吸引了多位成员的兴趣，并讨论了其功能和潜力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/deforum">加入 Deforum Discord 服务器！</a>：Deforum 是一个开源动画工具，利用 Stable Diffusion 的力量创建 AI 动画。 | 29464 名成员</li><li><a href="https://x.com/bennyj504/status/1785664502903570568">BennyJ504-075⚜😎🤑🔌.yat 🟣 (@bennyj504) 的推文</a>：https://www.hexagen.world/</li><li><a href="https://www.hexagen.world/">集体 AI 生成的游戏世界</a>：一个社交实验，任何人都可以帮助在浏览器中创建一个无限独特的模型。
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1235075486200107029)** (2 messages): 

- **首次体验 Llama3**：一位成员表达了第一次尝试 **Llama3** 的兴奋，表明了新用户对探索该 AI 模型能力的兴趣。
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1234844604638167094)** (33 messages🔥): 

_

- **简单的本地设置成功**：一位成员确认在本地设置系统非常容易实现。
- **Windows 兼容性障碍**：几位成员报告了在 Windows 上运行本地版本的问题，其中一位卡在 *Checking for index or schema changes...*。另一位成员澄清说 **Convex local 不支持 Windows**，但提到 Windows 兼容性的工作正在进行中。
- **分享 Mac 专用的运行命令**：对于在 Mac 上运行的用户，建议使用 `just convex dev` 进行专用同步，并使用 `just convex logs` 获取独立的终端日志输出，从而在不受 `npm run dev` 干扰的情况下平稳运行。
- **正确的 Node 版本至关重要**：一位成员在尝试运行应用时分享了一个与 **node version** 相关的错误。指出需要在与 `npm run dev` 相同的目录下运行 `convex-local-backend`，并确保两个目录中都使用了正确的 node 版本（`nvm use 19`）。
- **切换到 Linux 进行开发**：鉴于上述 Windows 的兼容性问题，一些成员考虑卸载 Windows 并安装 Linux，其中一人询问如何操作以及是否会影响玩游戏 Stellaris。另一位成员提供了一个 [WineHQ 链接](https://appdb.winehq.org/objectManager.php?sClass=application&iId=17537)，表明 Stellaris 有原生的 Mac 和 Linux 版本，暗示兼容性不会是问题。

**提到的链接**：<a href="https://appdb.winehq.org/objectManager.php?sClass=application&iId=17537">WineHQ - Stellaris</a>：未找到描述

  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1234768624204648578)** (35 messages🔥): 

- **语言模型与语法**：指向 LLM University 的链接解释了 LLM 等语言模型如何生成语法正确的句子。它讨论了 word embedding 和 sentence embedding 的概念，以及 self-attention 的关键作用，详细资源可在[此处](https://docs.cohere.com/docs/the-attention-mechanism)找到。
- **Command R 获得好评**：社区成员称赞 Cohere commandR/ R+ 模型，赞扬其高性能并将其与其他 LLM 进行对比，评论认为它们提供了企业级的精致体验。
- **基于 RAG 的 AI 法律助手研讨会**：关于使用 Cohere 的 RAG 构建 AI 法律助手的研讨会录像已分享，可在 [YouTube](https://www.youtube.com/watch?v=KfqJsqIFeRY&ab_channel=Cohere) 上观看。
- **讨论了用于 Connectors 的 Azure 和 OAuth**：对于想知道如何在 Azure 上为 connectors 设置 OAuth 的用户，澄清了可以使用 GitHub 上的 Cohere toolkit，它允许所有内容在 Azure 上运行，确保所有数据保持在内部，不进行外部数据共享。
- **探索 Command-R 的多语言支持**：社区正在 Command-R 上积极测试挪威语等语言，引发了关于语言支持和需要更好基准测试的讨论，尽管某些语言在没有官方支持的情况下似乎运行良好。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/docs/the-attention-mechanism">The Attention Mechanism</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=KfqJsqIFeRY&ab_channel=Cohere">Building a RAG-powered AI legal assistant with Cohere</a>：Cohere 最近发布了 Command R，这是一个高度可扩展的语言模型系列，在高性能和强准确性之间取得了平衡。在本次研讨会中，您将……</li><li><a href="https://github.com/cohere-ai/cohere-toolkit/?tab=readme-ov-file#how-to-add-a-connector-to-the-toolkit">GitHub - cohere-ai/cohere-toolkit: Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications.</a>：Toolkit 是预构建组件的集合，使用户能够快速构建和部署 RAG 应用程序。 - cohere-ai/cohere-toolkit
</li>
</ul>

</div>
  

---


**Cohere ▷ #[collab-opps](https://discord.com/channels/954421988141711382/1218409745380147320/1235223324804775957)** (1 messages): 

提供的单条消息历史记录中没有足够的细节或讨论点来创建摘要。如果提供更多聊天内容，可以按照指南创建摘要。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1234773013615087666)** (24 messages🔥): 

- **寻求 PDF 表格提取帮助**：一位成员询问如何**改进从 PDF 中提取表格**的效果，特别是当表格跨越多个页面时。他们正在使用 *unstructure*，但效果不佳。

- **将 Llama 3 与 LangChain 集成**：一位成员询问如何通过 LangChain 使用 **Llama 3**，并被建议使用 [Fireworks](https://python.langchain.com/docs/integrations/chat/fireworks/) 配合 **Fireworks API Key** 来实现。

- **寻找文档到图谱（Document-to-Graph）转换工具**：成员们讨论了对**自动将文档结构化为知识图谱**工具的需求。建议包括使用像 *unstructured* 或 *Azure Doc AI* 这样的布局解析器，并探索关于构建知识图谱的 [LangChain 文档](https://python.langchain.com/docs/use_cases/graph/constructing/)。

- **探索 AI 销售代理（Sales Agents）**：一位成员正在寻求关于构建 **AI 驱动的 Sales Agents** 的建议，这些 Agent 需要能够处理异议并保持人性化的语气。他们提到正在尝试 SalesGPT 逻辑，并对进一步推进该计划的合作持开放态度。

- **解决 AI Schema 知识局限性**：在一个拥有超过 2000 张表的服务器中，一位成员在 AI 理解所有 Schema 的能力方面面临挑战，这表明了 **AI 在数据库结构知识方面的局限性**。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/docs/use_cases/graph/constructing/">构建知识图谱 | 🦜️🔗 LangChain</a>：本指南将介绍构建知识图谱的基本方法。</li><li><a href="https://python.langchain.com/docs/integrations/chat/fireworks/">ChatFireworks | 🦜️🔗 LangChain</a>：Fireworks 加速产品开发。</li><li><a href="https://fireworks.ai/">Fireworks - 为产品创新而生的生成式 AI！</a>：使用 Fireworks.ai 以极快的速度使用最先进的开源 LLM 和图像模型，或者免费微调并部署您自己的模型！
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1234899074763722844)** (1 条消息): 

- **再次使用 Google Drive 库**：一位成员提到在某些操作中必须使用 Google Drive 库，并指出 **drive key** 应设置为环境变量。据指出，这些库之前被移除后又重新添加到了项目中。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1234773357178916916)** (7 条消息): 

- **用于 YouTube 视频摘要的 QuickVid 发布**：QuickVid 通过提供**极速摘要**和事实核查，引入了一种与 YouTube 内容互动的新方式。前往 [QuickVid](https://quickvid.vercel.app/) 体验这款可以**提升您 YouTube 体验**的工具。

- **高级 Webloader RAG 构建详解**：一位成员分享了一篇关于使用 Groq, LangChain 和 Datastax 构建强大的 **Webloader RAG 应用**的文章。详情可见这篇 [Medium 文章](https://medium.com/ai-advances/building-powerful-webloader-rag-applications-with-groq-langchain-and-datastax-f4816d88bee8)。

- **引入用于 AI 语言管理的 Word Loom 规范**：Word Loom 是一个用于管理 AI 语言的开放规范，旨在通过代码与自然语言分离、可组合性以及对机械比较和 G11N 技术友好等核心原则来改进 Prompt 管理。欢迎对该规范提出反馈，可在 [GitHub Gist](https://gist.github.com/uogbuji/5bd08f74125934fa9e0d37236a8e168e) 上查看。

- **LangChain Chatbot 更新及文档挑战**：LangChain Chatbot 已更新至版本 **0.1.17**，并承认了稳定版发布后过时文档带来的挑战。更新后的 Chatbot 运行示例可在 [LangChain Chatbot](https://langchain-chatbot.streamlit.app) 体验。

- **考虑为内容创作提供 LLM 性能报告**：一位成员正在测试排行榜上的各种 **LLM**，用于剧本创作和文案写作等内容创作场景，并询问详细报告是否对他人有用。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gist.github.com/uogbuji/5bd08f74125934fa9e0d37236a8e168e">Word Loom 提议更新</a>：Word Loom 提议更新。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/carlosplanchon/gpt_pydantic_tools">GitHub - carlosplanchon/gpt_pydantic_tools: 一种使用 Pydantic Schemas 编写 GPT 工具的方法。</a>：一种使用 Pydantic Schemas 编写 GPT 工具的方法。通过在 GitHub 上创建账号为 carlosplanchon/gpt_pydantic_tools 的开发做出贡献。</li><li><a href="https://quickvid.vercel.app/">QuickVid</a>：未找到描述</li><li><a href="https://langchain-chatbot.streamlit.app">无标题</a>：未找到描述
</li>
</ul>

</div>
  

---

**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1234782249166049310)** (3 messages): 

- **Advanced RAG 的巴黎风味**：一段新的教程视频展示了 **LangChain** 与 **Mistral Large** 以及 **Llamaindex** 的集成，旨在为法语社区构建一个 Advanced RAG 助手。内容已在 YouTube 上线，标题为“[Multi-Agent RAG: LangChain et LlamaIndex portés par Mistral Large - Le vent du changement](https://youtu.be/ol2QMp64lgo)”，应用代码已在视频描述中提供。

- **训练本地 Llama3 的新花样**：分享了一段名为“*I want Llama3 to perform 10x with my private knowledge* - Local Agentic RAG w/ llama3”的教学视频，演示了如何利用私有知识训练 **llama3** 以构建 Agentic RAG。视频可以在[这里](https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P)找到。

- **基于复杂度的 RAG 策略选择**：“[LangGraph + Adaptive Rag + LLama3 Python Project: Easy AI/Chat for your Docs](https://www.youtube.com/watch?v=QnXdlqEUW80)”视频介绍了一种 Adaptive RAG 方法，该方法可以根据查询的复杂度调整其策略。这项技术有望优化 AI/Chat 与文档集成的性能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">&quot;I want Llama3 to perform 10x with my private knowledge&quot; - Local Agentic RAG w/ llama3</a>：Advanced RAG 101 - 使用 llama3 构建 Agentic RAG。获取关于 AI 如何重新定义初创公司 GTM 策略的免费 HubSpot 报告：https://clickhubspot.com/4hx🔗 Links- F...</li><li><a href="https://www.youtube.com/watch?v=QnXdlqEUW80">LangGraph + Adaptive Rag + LLama3 Python Project: Easy AI/Chat for your Docs</a>：#langchain #langgraph #rag #python #automation #llm #ai #automation 在这段视频中，我为你准备了一个非常快速的教程，展示如何创建一个完全本地的...</li><li><a href="https://youtu.be/ol2QMp64lgo">Multi-Agent RAG: LangChain et LlamaIndex portés par Mistral Large - Le vent du changement</a>：在这段新视频中，我将 Mistral Large 投入测试，使用 LangChain 和 LlamaIndex 开发一个多 Agent RAG 助手....
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1234890920575631360)** (1 messages): 

```html
<ul>
  <li><strong>加入 Mozilla AI 团队</strong>：Mozilla AI 正在扩大团队并正在招聘。感兴趣的人员可以在其官方 Discord 频道[此处](https://discord.com/channels/1089876418936180786/1230938514955436242/1234870020916510823)查看就业机会。</li>
  <li><strong>介绍 Lm-buddy</strong>：Mozilla AI 发布了一个名为 **Lm-buddy** 的新开源工具，旨在帮助更高效地评估模型。欲了解更多详情和访问权限，请访问其频道中的公告[此处](https://discord.com/channels/1089876418936180786/1230938514955436242/1234589599733518378)。</li>
  <li><strong>本地 LLM 作为数字法官</strong>：有一项关于通过 Prometheus 框架使用 **本地 LLM** 作为法官的讨论。详情可在 Discord 频道查看，访问链接[此处](https://discord.com/channels/1089876418936180786/1234890301143912599/1234890301143912599)。</li>
</ul>
```
  

---


**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1234906250358296607)** (34 messages🔥):

- **M1 MacBook Air 运行 LLaMA3 的问题**：一位成员报告了在 M1 MacBook Air 上运行 **LLaMA3:8b** 的问题，该模型在 ollama 上运行正常，但在 llamafile 上不行。回复称，在解决其他正在进行的后续支持问题后，将优先在 M1 上进行测试。
- **将 Whisper 模型封装进 Llamafile**：有人建议将 **whisper.cpp 模型** 封装进 llamafile 以实现更快的推理，并指出尽管使用 cosmo libc 构建 whisper 很容易，但麦克风和扬声器的集成仍未解决。
- **Justine Tunney 的 GEMM 博客事实核查**：一位用户询问了一篇博客文章 (https://justine.lol/matmul/)，该文章称 **np.matmul** 的性能为 29 gflops，并指出个人经验中的 gflop 性能要高得多；回复澄清了原始测量是在一台安装了 **Ubuntu 的 Intel 电脑**上进行的，并解释了计算 flops 的差异。
- **同时运行多个 Llamafile**：关于同时运行多个加载不同模型的 llamafile 的讨论得到了确认，这是可行的。有人指出操作系统将管理资源分配，并且可能需要额外的工具来进行优化使用。
- **Llamafile 公共路径自定义**：一位成员询问了关于使用 `--server --path PUBLIC_PATH` 选项进行自定义的问题。提到唯一经过测试的自定义方式是替换 zip 中的 .html 和 .js 文件，而不是使用外部目录。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/jartine/Phi-3-mini-4k-instruct-llamafile#prompting">jartine/Phi-3-mini-4k-instruct-llamafile · Hugging Face</a>: no description found</li><li><a href="https://github.com/stanford-futuredata/FrugalGPT">GitHub - stanford-futuredata/FrugalGPT: FrugalGPT: better quality and lower cost for LLM applications</a>: FrugalGPT: better quality and lower cost for LLM applications - stanford-futuredata/FrugalGPT
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1234900403498258542)** (8 messages🔥): 

- **关于反向传播操作图表的疑问**：Shikhar_7985 询问了关于为涉及两个 reduce 操作的反向传播（backward passes）问题 **#3572** 创建图表的方法。Akshatxv 提到有一个可以使用的 dot 文件，而 python273 提示可以设置 `GRAPH=1`。

- **Tinygrad 中的符号形状和跳过的测试**：Georgehotz 提到了他在 Tinygrad 中关于 symbolic shapes 的工作，并分享了一个 [pull request](https://github.com/tinygrad/tinygrad/pull/4362)，其中包含一个针对 symbolic arange 的跳过测试。

- **在 Google 之外寻求 Tinygrad 知识**：Lynn4400 表达了对学习更多 Tinygrad 知识的兴趣，特别是其 kernels，并提到受到了 Lex Fridman 播客的影响。Leikowo 引导他们查看仓库的文档，作为更好地理解 Tinygrad 的良好起点。

**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/pull/4362">tensor variable by geohot · Pull Request #4362 · tinygrad/tinygrad</a>: no description found

  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1234795444773715979)** (13 messages🔥): 

- **Tinygrad 将 Scalar 重命名为 ConstType**：该项目有一个 [commit](https://github.com/tinygrad/tinygrad/commit/77589bc7a5430ee470621e43fb1817259d3ce0f5) 将 `Scalar` 重命名为 `ConstType`，将 `cast_scalar` 重命名为 `as_const`，作为 *pre-req cleanup*，以标准化常量参数类型与 dtype。

- **探索 Const 支持变量**：一位成员建议改进 tinygrad 在操作中对常量的处理，提议使用 const support variables 代替 tensor variables 以简化操作，并在 scheduling phase 断言边界。

- **符号 JIT 和变量均值测试**：在讨论了符号 JIT 增强的需求后，有人指出验证改进的一个好测试是改变 symbolic JIT 变量值，并计算具有可变长度的 2D tensor 的均值。

- **强调让 Const 变量正常工作**：重点在于使 tinygrad 中的 const Variables 能够正常运行，因为它们对于与符号维度和操作相关的操作至关重要。

- **在 Nvidia Xavier 上使用 EfficientNet CUDA**：成员们讨论了在 Nvidia Xavier 上运行 efficientnet 示例的问题，建议检查是否使用了 `CUDA=1` 以确保脚本正确执行。

- **符号逻辑中的技术划分**：关于 tinygrad 代码库中 Rednode 和 OpNode 区别的辩论，质疑 Rednode 是否使符号编译器逻辑复杂化，以及是否应该将其分离出来。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:tinygrad:symbolic-mean-var-pull">比较 tinygrad:master...davidjanoskyrepo:symbolic-mean-var-pull · tinygrad/tinygrad</a>: 你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - 比较 tinygrad:master...davidjanoskyrepo:symbolic-mean-var-pull · tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/compare/86d90511cee2%5E...97a2d44d9840">比较 86d90511cee2^...97a2d44d9840 · tinygrad/tinygrad</a>: 你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - 比较 86d90511cee2^...97a2d44d9840 · tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/commit/77589bc7a5430ee470621e43fb1817259d3ce0f5">将 Scalar 重命名为 ConstType，将 cast_scalar 重命名为 as_const (#3946) · tinygrad/tinygrad@77589bc</a>: 前置清理工作，使 const 参数与 dtype 具有相同的 Python 类型。
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1235293456511799328)** (11 messages🔥): 

- **Anthropic 发布 Claude**: Anthropic 正式发布了 Claude 应用，部分成员已开始下载使用。
- **关于 Claude 质量的疑问**: 成员们好奇 Anthropic 新推出的 Claude 应用与 OpenAI 的产品相比如何，质疑其质量是否过硬。
- **新应用运行顺畅**: 一位成员表示在使用 Claude 应用时未遇到任何问题，并表达了对 Anthropic 品牌设计的喜爱。
- **Anthropic 品牌赢得人心**: 对话反映了对 Anthropic 品牌策略的积极反馈，成员们认可其 Logo 的吸引力。
- **ML Collective 会议持续进行**: 一位成员确认他们仍在参加 ML Collective 会议，尽管不是每周都参加。
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1234876113021440090)** (1 messages): 

- **重新思考 AI 排行榜**: 一篇由 Sayash Kapoor、Benedikt Stroebl 和 Arvind Narayanan 撰写的题为[“AI 排行榜不再有用”](https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful)的分享文章质疑了当前 AI 排行榜的实用性。根据 [HumanEval 基准测试](https://paperswithcode.com/sota/code-generation-on-humaneval)，**LDB** 是目前最准确的公开代码生成系统，但由于需要反复调用 GPT-4 等语言模型，其高昂的成本是一个重大缺陷。

**提及链接**: <a href="https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful">AI 排行榜不再有用。是时候转向帕累托曲线（Pareto curves）了。</a>: 花费 2,000 美元能告诉我们关于评估 AI Agent 的什么。

  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1235253560917233685)** (2 messages): 

- **激励提升成功**: 针对直率的绩效批评，一位成员显著提升了工作质量，引发了其他人的积极且热烈的反应。
  

---



**Alignment Lab AI ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/1234767428035022920)** (1 messages): 

- **不当内容警报**: 该频道收到一条推广 Discord 邀请链接的消息，据称提供涉及未成年人的可疑且可能违法的泄露资料。该消息包含暗示成人内容的表情符号，并艾特了频道中的所有人。

**提及链接**: <a href="https://discord.gg/CYNumE8ABr">加入 e-girl paradise 🍑🍒 // +18 Discord 服务器！</a>: 查看 Discord 上的 e-girl paradise 🍑🍒 // +18 社区 - 与其他 16457 名成员一起聚会，享受免费的语音和文字聊天。

  

---


**Alignment Lab AI ▷ #[programming-help](https://discord.com/channels/1087862276448595968/1087876753462136873/1234767505835425803)** (1 messages): 

- **不当内容警报**: 频道中的一条消息包含免费提供“18+ 青少年女孩和 OnlyFans 泄露内容”的信息，并附带了 Discord 邀请链接。此类内容不适合专注于 AI Alignment 和编程帮助的频道。

**提及链接**: <a href="https://discord.gg/CYNumE8ABr">加入 e-girl paradise 🍑🍒 // +18 Discord 服务器！</a>: 查看 Discord 上的 e-girl paradise 🍑🍒 // +18 社区 - 与其他 16457 名成员一起聚会，享受免费的语音和文字聊天。

  

---


**Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1234767644352315433)** (1 messages): 

- **不当内容警报**: 有人发布了一条消息，提供免费的 **18+ 青少年女孩和 OnlyFans 内容**泄露，并附带 Discord 邀请链接。此类内容违反了社区准则并涉嫌推广非法活动。

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: 查看 Discord 上的 e-girl paradise 🍑🍒 // +18 社区 - 与其他 16457 名成员一起交流，享受免费的语音和文字聊天。

---

**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1234767675062747157)** (1 messages): 

- **不当内容警报**：该频道包含一条推广成人内容的消息，包括 **18+ teen girls** 和 **OnlyFans leaks**。该消息包含表情符号和一个 Discord 邀请链接。

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: 查看 Discord 上的 e-girl paradise 🍑🍒 // +18 社区 - 与其他 16457 名成员一起交流，享受免费的语音和文字聊天。

---

**Alignment Lab AI ▷ #[landmark-dev](https://discord.com/channels/1087862276448595968/1113327574563692654/1234767716267855884)** (1 messages): 

- **不当内容警报**：发布了一条包含成人内容链接和 OnlyFans 泄露资料的消息，看起来是垃圾信息或钓鱼尝试。这包括一个 Discord 频道邀请，据称提供此类内容的免费访问。

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，与你的朋友和社区保持紧密联系。

---

**Alignment Lab AI ▷ #[landmark-evaluation](https://discord.com/channels/1087862276448595968/1118282868595109918/1234767861927645225)** (1 messages): 

- **不当内容警报**：发布了一条包含 NSFW 内容链接的消息，特别是推广 **18+ Teen Girls** 和 **OnlyFans leaks**。发布者分享了一个 Discord 邀请链接并艾特了所有人。

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: 查看 Discord 上的 e-girl paradise 🍑🍒 // +18 社区 - 与其他 16457 名成员一起交流，享受免费的语音和文字聊天。

---

**Alignment Lab AI ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/1234767970668908585)** (1 messages): 

- **不当内容警报**：发布了一条包含潜在露骨内容链接和查看 **onlyfans** 泄露内容邀请的消息，暗示分享针对 18+ 受众的非法内容。该帖子包含表情符号和一个 Discord 邀请链接。

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: 查看 Discord 上的 e-girl paradise 🍑🍒 // +18 社区 - 与其他 16457 名成员一起交流，享受免费的语音和文字聊天。

---

**Alignment Lab AI ▷ #[leaderboard](https://discord.com/channels/1087862276448595968/1135102537817653308/1234768131247964212)** (1 messages): 

- **不当内容警报**：发布了一条包含露骨内容链接的消息，特别提到了一个包含订阅服务 OnlyFans 泄露内容的 Discord 服务器，可能涉及未成年人。该消息包含一个 Discord 邀请链接，并使用了暗示内容为成人性质的表情符号。

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: 查看 Discord 上的 e-girl paradise 🍑🍒 // +18 社区 - 与其他 16457 名成员一起交流，享受免费的语音和文字聊天。

---

**Alignment Lab AI ▷ #[looking-for-workers](https://discord.com/channels/1087862276448595968/1142242166677192774/1234768231554879488)** (1 messages): 

- **不当内容警报**：一条消息包含对以未成年人形象出现的成人内容的不当招揽，包括一个 Discord 邀请链接。该消息因推广不良内容而被标记。

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: 查看 Discord 上的 e-girl paradise 🍑🍒 // +18 社区 - 与其他 16457 名成员一起交流，享受免费的语音和文字聊天。

---

**Alignment Lab AI ▷ #[looking-for-work](https://discord.com/channels/1087862276448595968/1142242683339944027/1234768257148391435)** (1 messages): 

- **不当内容警报**：频道中的一条消息包含提供涉及年轻人的成人内容，以及一个 Discord 邀请链接。此类内容极不恰当，可能违反了多项服务条款以及与分发未成年人露骨内容相关的法律。

**提及链接**：<a href="https://discord.gg/CYNumE8ABr">加入 e-girl paradise 🍑🍒 // +18 Discord 服务器！</a>：查看 Discord 上的 e-girl paradise 🍑🍒 // +18 社区 - 与 16457 名其他成员一起交流，享受免费的语音和文字聊天。

---

**Alignment Lab AI ▷ #[join-in](https://discord.com/channels/1087862276448595968/1143791237669855302/1234768325972856912)** (1 条消息): 

- **不当内容警示**：发布了一条推广**成人内容**的消息，特别是涉及*青少年女性*和 *OnlyFans 泄露*的内容，并附带了 Discord 邀请链接。该帖子似乎旨在将流量引导至另一个可能包含显式内容的 Discord 服务器。

**提及链接**：<a href="https://discord.gg/CYNumE8ABr">加入 e-girl paradise 🍑🍒 // +18 Discord 服务器！</a>：查看 Discord 上的 e-girl paradise 🍑🍒 // +18 社区 - 与 16457 名其他成员一起交流，享受免费的语音和文字聊天。

---

**Alignment Lab AI ▷ #[fasteval-dev](https://discord.com/channels/1087862276448595968/1147528620936548363/1234768398429458506)** (1 条消息): 

无法提供摘要，因为内容不包含与 AI 或 Alignment Lab AI Discord 聊天机器人消息相关的相关主题或讨论点。此外，内容似乎不当，且与通常总结的预期学术或专业讨论不符。

**提及链接**：<a href="https://discord.gg/CYNumE8ABr">加入 e-girl paradise 🍑🍒 // +18 Discord 服务器！</a>：查看 Discord 上的 e-girl paradise 🍑🍒 // +18 社区 - 与 16457 名其他成员一起交流，享受免费的语音和文字聊天。

---

**Alignment Lab AI ▷ #[qa](https://discord.com/channels/1087862276448595968/1147528698669584424/1234768427663495248)** (1 条消息): 

- **不当内容警示**：发布了一条似乎在推广获取成人内容的消息，涉及可能未达法定同意年龄的个人，并附带了一个 Discord 服务器链接。此类内容不仅不当，而且可能违法，应立即举报并删除。

**提及链接**：<a href="https://discord.gg/CYNumE8ABr">加入 e-girl paradise 🍑🍒 // +18 Discord 服务器！</a>：查看 Discord 上的 e-girl paradise 🍑🍒 // +18 社区 - 与 16457 名其他成员一起交流，享受免费的语音和文字聊天。

---

**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1234909827453681764)** (11 条消息🔥): 

- **LLaMA-3 Instruct 提示策略揭晓**：分享了 **LLaMA-3 instruct 提示策略**的更新，声称改进了模型的性能，包括相关的 GitHub [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553)。
  
- **澄清数据集条目混淆**：一位成员详细说明了使用 `eot_id` 解决了之前尝试的方法中遇到的问题，即手动在每个数据集条目末尾添加 `<|eot_id|>`。

- **Meta 的迭代推理优化提升准确率**：传阅了题为“Iterative Reasoning Preference Optimization”的论文，指出 Meta 的进展使得 LLama-2-70B-Chat 在 GSM8K 和 ARC-Challenge 等多个基准测试上的准确率有所提高。论文链接见[此处](https://arxiv.org/abs/2404.19733)。

- **使用 Axolotl 微调 LLaMA-3**：一位用户分享了他们使用 **Axolotl 微调 LLaMA-3 8b** 的经验，导致模型输出包含 `</s>`。

<div class="linksMentioned">

<strong>提及链接</strong>：

<ul>
<li>
<a href="https://x.com/arankomatsuzaki/status/1785489252299485188">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：Meta 展示迭代推理偏好优化（Iterative Reasoning Preference Optimization），提升 Llama-2-70B-Chat 的准确率：- GSM8K 从 55.6% -> 81.6% - MATH 从 12.5% -> 20.8% - ARC-Challenge 从 77.8% -> 86.7% ...</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/conversation.html#sharegpt">Axolotl - 对话</a>：未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553">feat: 为微调添加 LLaMA-3 instruct 提示策略，由 0-hero 提交 · Pull Request #1553 · OpenAccess-AI-Collective/axolotl</a>：描述：此项工作基于并包含了以下 PR 的更改：#1542 #1539。在合并此项之前，需要先合并来自 @TJ-Solergibert 的 Fastchat PR lm-sys/FastChat#3257...
</li>
</ul>

</div>

---

**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1234767923105632326)** (2 条消息):

- **激励人心的节奏，让你充满动力**：分享了一首名为 "NEVER GIVE UP YOUR WAAAAAAAAAAAAY" 的动漫风格励志曲目，其中包含来自经典动漫 *Kill La Kill* 的器乐版本。这段 [YouTube 视频](https://youtu.be/tYzMYcUty6s?si=t2utqcq36PHbk9da) 鼓励观众永不言弃，并附带了 Patreon 的支持链接。
- **算我一个！**：一位成员热情地回应道“我也会在那儿”，表示对之前分享内容的参与或支持。

**提到的链接**：<a href="https://youtu.be/tYzMYcUty6s?si=t2utqcq36PHbk9da">NEVER GIVE UP YOUR WAAAAAAAAAAAAY</a>：NEVA GIVE UP - https://bit.ly/2VrgAcKSong 是来自动漫 Kill La Kill 的 Before my Body is Dry 器乐版本。请考虑向我们的 Patreon 捐赠！https://w...

  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1234775513499963463)** (1 条消息): 

- **本地加载速度快**：一位成员提到，在本地机器上运行程序非常快，*只需 3 秒即可加载*，这表明与提交任务后加载缓慢的情况相比，存储并不是问题所在。
  

---


**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/)** (1 条消息): 

le_mess: Llama 3 在 Scandeval 上似乎击败了 GPT-4
https://scandeval.com/german-nlg/
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1235150707439702057)** (1 条消息): 

- **使用 qdora 探索模型扩展**：一位成员通过提到 **qdora** 引起了大家对 LLM 扩展的兴趣，这是一种针对 LLaMA 等模型的折中方案。他们提供了一个讨论该过程的 [Answer.ai 博客文章](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html) 链接。
- **深入研究 LLaMA Pro 的无遗忘学习**：对话还强调了旨在防止 LLM 出现灾难性遗忘的新预训练后方法，并指向了一篇关于扩展 Transformer 块以在获取新技能的同时保留旧技能的 [Arxiv 论文](https://arxiv.org/abs/2401.02415)。

**提到的链接**：<a href="https://arxiv.org/abs/2401.02415">LLaMA Pro: Progressive LLaMA with Block Expansion</a>：人类通常在不损害旧技能的情况下习得新技能；然而，对于大语言模型（LLMs）来说情况正好相反，例如从 LLaMA 到 CodeLLaMA。为此，我们提出了一种新的预训练后...

  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1234824449552027749)** (2 条消息): 

- **Datasette UX 挑战**：一位成员正在为 Datasette 首页的用户界面寻求创意，用户可以从下拉菜单中选择选项，例如选择一个国家来获取与该选择相关的摘要数据。
- **思考动态 URL 与可定制界面的对比**：针对 Datasette 首页提出了两种 UX 方案；一种涉及在事件发生时动态更新 URL，直接引导用户访问数据；另一种则允许用户通过根据选择更新预设查询（canned queries）来“构建”首页。
  

---



---