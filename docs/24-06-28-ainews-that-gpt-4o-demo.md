---
companies:
- openai
- google-deepmind
- meta-ai-fair
date: '2024-06-29T00:48:47.349723Z'
description: '**Romain Huet** 在 ChatGPT 桌面端演示了 **GPT-4o** 的一个未发布版本，展示了低延迟语音生成、耳语语调调节、相机模式流式传输视频至
  GPT-4o、快速 OCR、用于编程辅助的屏幕共享、剪贴板读取以及基于视觉的代码对话等功能。OpenAI 强调的四个重点投资领域包括：文本智能、效率与成本、模型定制化以及多模态智能体。


  **谷歌 DeepMind** 发布了 9B 和 27B 规模的 **Gemma 2** 模型，分别在 8T 和 13T token 上进行了训练，采用了 SFT（有监督微调）、蒸馏、RLHF（人类反馈强化学习）和模型合并技术，并针对
  TPUv5e 进行了优化，具备强大的性能和安全措施。**Meta AI** 宣布推出基于 Meta Code Llama 构建的 Meta LLM 编译器，具有增强的代码优化和编译器功能。'
id: 521448c3-d0cc-48c6-bb04-66adbc1f25b0
models:
- gpt-4o
- gemma-2
- meta-code-llama
original_slug: ainews-that-openai-demo
people:
- romain-huet
- fchollet
title: 那个 GPT-4o 演示
topics:
- voice-generation
- ocr
- screen-sharing
- vision
- code-understanding
- model-customization
- efficiency
- textual-intelligence
- multimodal-agents
- sft
- distillation
- rlhf
- model-merging
- model-optimization
- safety
---

<!-- buttondown-editor-mode: plaintext -->**Omnimodel is all you need**

> 2024年6月27日至6月28日的 AI 新闻。
我们为您检查了 7 个 subreddit、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**417** 个频道和 **3655** 条消息）。
预计节省阅读时间（按每分钟 200 字计算）：**354 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

[Romain Huet](https://twitter.com/romainhuet) 使用未发布版本的 ChatGPT Desktop 演示 GPT-4o 的视频[昨天引起了轰动](https://x.com/tsarnick/status/1806526891354132604)，这基本上是 GPT-4o 发布后的第二次高规格演示（[我们的报道见此处](https://buttondown.email/ainews/archive/ainews-gpt-4o-the-new-sota-everything-frontier/)），在没有更重大新闻的情况下，这是我们今天的头条首选：

 
![image.png](https://assets.buttondown.email/images/f3179985-63dd-4431-8803-f2f0c1bc62ad.png?w=960&fit=max)
 

演示从[直播的 7:15:50 标记处](https://www.youtube.com/live/vaIiNZoXymg?si=S73xOMIAlTOvAzEc&t=26153)开始，你应该观看完整视频。

演示的功能包括：

- 低延迟语音生成 (voicegen)
- 调节音调至耳语（甚至更轻的耳语）的指令
- 打断功能
- ChatGPT Desktop 上的相机模式 —— 持续向 GPT-4o 传输视频流
-  
![image.png](https://assets.buttondown.email/images/c5cfc90d-8127-4e20-ae47-10486b61be2b.png?w=960&fit=max)
 
- 配合语音理解，它消除了对“发送”或“上传”按钮的需求
- **快速 OCR**：Romain 询问一个随机页码，并展示书中的一页 —— 它几乎瞬间就读出了页面内容！不幸的是，OCR 出现了一点失误 —— 它误读了 "Coca Cola"，但现场演示的条件并不理想。
 
![image.png](https://assets.buttondown.email/images/f8837ed4-8e51-4be2-bcc8-d4df999156a0.png?w=960&fit=max)
 
- **与 ChatGPT 共享屏幕**：与 ChatGPT 交谈以描述他的编程问题，并让它根据视觉上下文进行理解
 
![image.png](https://assets.buttondown.email/images/2bf81e11-7a9c-40d0-aab7-630fc3749825.png?w=960&fit=max)
 
- **读取剪贴板**：复制代码，要求对代码进行“单行概述”（此功能目前已在 ChatGPT Desktop 中提供）
- **与 ChatGPT 讨论代码**：反复讨论代码中的 Tailwind 类名，依赖视觉（而非剪贴板）

![image.png](https://assets.buttondown.email/images/80b51c10-dd96-4e33-ab8a-5c3acec13c23.png?w=960&fit=max)
 

演讲的其余部分讨论了 OpenAI 的 4 个“投资领域”：

- **文本智能**（再次使用了 "GPT Next" 而非 "GPT5"……）
 
![image.png](https://assets.buttondown.email/images/09da9fa4-0123-4f40-b3f8-b927b210508f.png?w=960&fit=max)
 
- **效率/成本**  
![image.png](https://assets.buttondown.email/images/21d2c62d-f81d-4d88-9378-787e644e4667.png?w=960&fit=max)
 
- **模型定制** 
![image.png](https://assets.buttondown.email/images/acbc1d92-631b-4f7e-ae3e-057ee9451558.png?w=960&fit=max)
 
- **多模态 Agent**  
![image.png](https://assets.buttondown.email/images/0637489f-e54e-4d2f-aabf-83eecef1ba65.png?w=960&fit=max)
 ，包括 Sora 和 Voice Engine 的演示，如果你以前没见过，真的应该去看看。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要回顾

> 所有摘要由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**Google DeepMind 发布 Gemma 2**

- **模型尺寸与训练**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1806373224889954449) 宣布发布 9B 和 27B 参数尺寸的 Gemma 2，分别在 13T tokens (27B) 和 8T tokens (9B) 上进行训练。使用了 **SFT, Distillation, RLHF & Model Merging** 技术。在 Google TPUv5e 上完成训练。
- **性能表现**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1806373227305775438) 9B 版本在同尺寸类别的开源模型中提供了**领先的性能**。27B 版本的表现超过了某些尺寸是其两倍以上的模型，并**针对在单个 TPU 主机上高效运行进行了优化**。
- **可用性**：[@fchollet](https://twitter.com/fchollet/status/1806346069653287085) Gemma 2 已在 Kaggle 和 Hugging Face 上线，**使用 Keras 3 编写，并兼容 TensorFlow, JAX 和 PyTorch**。
- **安全性**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1806373232250917334) 遵循了**严格的内部安全流程**，包括过滤预训练数据、严苛的测试和评估，以识别并缓解潜在的偏见和风险。

**Meta LLM Compiler 发布**

- **功能特性**：[@AIatMeta](https://twitter.com/AIatMeta/status/1806361623831171318) 宣布推出 Meta LLM Compiler，该模型基于 Meta Code Llama 构建，具有**额外的代码优化和编译器能力**。它可以模拟编译器，预测代码大小的最优 pass，并进行代码反汇编。
- **可用性**：[@AIatMeta](https://twitter.com/AIatMeta/status/1806361623831171318) LLM Compiler 7B & 13B 模型已在 Hugging Face 上发布，采用**允许研究和商业用途的宽松许可证**。
- **潜力**：[@MParakhin](https://twitter.com/MParakhin/status/1806382680616861961) 认为 LLM 替代编译器可能会带来**近乎完美的优化代码**，扭转数十年来效率下滑的趋势。[@clattner_llvm](https://twitter.com/clattner_llvm/status/1806457173498814708) 提到 Mojo 🔥 是过去 15 年编译器研究、MLIR 以及许多其他经验教训的结晶。

**Perplexity Enterprise Pro 更新**

- **面向学校和非营利机构的降价**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1806408640431288664) 宣布为任何学校、非营利组织、政府机构或非营利团体提供 Perplexity Enterprise Pro 的**价格优惠**。
- **重要性**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1806408644826972571) 这些组织在**解决社会问题和为儿童提供教育**方面发挥着关键作用。Perplexity 希望确保其技术对他们是可触达的。

**LangChain 推出 LangGraph Cloud**

- **功能特性**：[@LangChainAI](https://twitter.com/LangChainAI/status/1806371717084025165) 宣布推出 LangGraph Cloud，这是用于**大规模运行容错 LangGraph Agent** 的基础设施。它能处理大型工作负载，支持调试和快速迭代，并提供集成的追踪与监控。
- **工具特性**：[@hwchase17](https://twitter.com/hwchase17/status/1806376010176483355) LangGraph Studio 是一个**用于测试、调试和分享 LangGraph 应用程序的 IDE**。它基于支持多样化控制流的 LangGraph v0.1 构建。

**其他值得关注的更新与讨论**

- **Gemini 1.5 Pro 更新**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1806345612184482149) 向所有开发者开放了 Gemini 1.5 Pro 的 **200 万 token 上下文窗口**。Gemini API 现已支持**上下文缓存（Context caching）**以降低成本。
- **清醒梦体验**：[@karpathy](https://twitter.com/karpathy/status/1806400213793534010) 分享了一次清醒梦体验，注意到其**极其细致和高分辨率的画面**，并将其比作类似 Sora 的视频+音频生成模型。
- **Anthropic 更新**：[@alexalbert__](https://twitter.com/alexalbert__/status/1806410983931629807) Anthropic 开发者现在可以在 Anthropic Console 的新“使用量和成本”选项卡中查看**按金额、token 数量和 API keys 细分的 API 使用情况**。
- **蒸馏讨论**：[@giffmana](https://twitter.com/giffmana/status/1806402283649036605) 和 [@jeremyphoward](https://twitter.com/jeremyphoward/status/1806446889006666110) 讨论了蒸馏（distillation）的重要性以及在训练小型高性能模型时的**“容量差距诅咒”（curse of the capacity gap）**。

---

# AI Reddit 综述

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 模型与架构**

- **Gemma 2 模型超越 Llama 和 Claude**：在 /r/LocalLLaMA 中，根据一份技术报告，Google 的 Gemma 2 27B 模型在 [**LMSYS 基准测试中击败了 Llama 3 70B**](https://www.reddit.com/r/LocalLLaMA/comments/1dpsal9/gemma_2_lmsys_results_from_technical_report/)。其 9B 变体也 [超越了 Claude 3 Haiku](https://www.reddit.com/r/LocalLLaMA/comments/1dpwlqj/lets_talk_more_about_gemma2_26b_the_smallest/)。
- **针对小型模型的知识蒸馏（Knowledge distillation）**：Gemma 2 9B 是 [**使用 27B 模型作为教师模型进行训练的**](https://www.reddit.com/r/LocalLLaMA/comments/1dpwlqj/lets_talk_more_about_gemma2_26b_the_smallest/)，这种方法可能是中小型模型的未来趋势，甚至可能使用像 Llama 400B 这样更大的模型作为教师模型。
- **无矩阵乘法（MatMul-free）语言建模**：一篇新论文介绍了一种 [**在保持强劲性能的同时，从语言模型中消除矩阵乘法的方法**](https://www.reddit.com/r/LocalLLaMA/comments/1dqdhsh/gemma2_originally_kaggle_posted_fp32_unquantized/)。它在训练时减少了 61% 的内存占用，在推理时减少了 10 倍，配合定制的 FPGA 硬件，处理模型的功耗仅为 13W。
- **Sohu 芯片交付巨大性能**：来自 Etched 的专用 Sohu AI 芯片 [**据称可替代 160 块 H100 GPU**](https://www.reddit.com/r/OpenAI/comments/1dpyba1/sohu_replaces_160_nvidia_gpus_delivers_500000/)，提供 500,000 tokens/sec 的吞吐量。它声称比 Nvidia 的下一代 Blackwell GPU 快 10 倍且更便宜。

**AI 应用与用例**

- **AI 生成的图形小说**：在 /r/StableDiffusion 中，一位作者 [**完全使用 Stable Diffusion 创作了一本图形小说**](https://www.reddit.com/r/StableDiffusion/comments/1dplv1d/i_finally_published_a_graphic_novel_made_100_with/)，使用 SD1.5 iComix 制作角色，ControlNet 保持一致性，并使用 Photoshop 进行排版。这是第一本以阿尔巴尼亚语出版的此类小说。
- **利用 AI 重构网站**：一个 [浏览器扩展使用 OpenAI API 根据提供的提示词重构网站](https://www.reddit.com/r/StableDiffusion/comments/1dplv1d/browser_extension_uses_openai_api_to_redesign_the/)，利用 CSS 变量实现。已在 shadcn.com 和 daisyui.com 上进行了实验。
- **个性化 AI 助手**：/r/LocalLLaMA 上的一篇文章详细介绍了 [**通过 WhatsApp 和 Obsidian 数据扩展个人 Llama 3 8B 模型**](https://www.reddit.com/r/LocalLLaMA/comments/1dpu2oo/personal_local_llm_extended_with_whatsapp/)，以创建一个个性化的 AI 助手。

**迷因与幽默**

- **AI 讨论中的脱节**：一段 [迷因视频嘲讽了 AI 爱好者与普通大众在讨论该技术时的脱节](https://www.reddit.com/r/StableDiffusion/comments/1dplv1d/how_it_feels_to_talk_about_ai_with_normal_people/)。
- **AI 生成的电影**：一段 [幽默视频想象了在不久的将来由 AI 批量生产的公式化电影](https://www.reddit.com/r/StableDiffusion/comments/1dplv1d/your_average_movie_in_2025/)。
- **AI 生成动画的进展**：一段 [威尔·史密斯吃意大利面的迷因视频](https://www.reddit.com/r/StableDiffusion/comments/1dplv1d/new_checkpoint_achieved/) 展示了 AI 生成逼真人物动画能力的提升，目前仅剩细微的面部和手臂瑕疵。

---

# AI Discord 综述

> 摘要的摘要之摘要

**1. 模型性能优化与基准测试**

- **[量化（Quantization）](https://github.com/Vahe1994/AQLM)** 技术如 **AQLM** 和 **QuaRot** 旨在单块 GPU 上运行大型语言模型（**LLMs**）并保持性能。例如：[AQLM 项目](https://github.com/Vahe1994/AQLM) 实现了在 RTX3090 上运行 **Llama-3-70b**。

- 通过 **动态内存压缩（DMC）** 等方法努力 **提升 Transformer 效率**，在 **H100 GPUs** 上可能提高高达 370% 的吞吐量。例如：@p_nawrot 的 [DMC 论文](https://arxiv.org/abs/2403.09636)。

- 关于 **优化 CUDA 操作** 的讨论，例如融合逐元素操作（element-wise operations），使用 **Thrust 库的 `transform`** 来实现接近带宽饱和的性能。例如：[Thrust 文档](https://nvidia.github.io/cccl/thrust/api/groups/group__modifying.html#function-for-each)。

- 在 **AlignBench** 和 **MT-Bench** 等基准测试中进行 **模型性能** 对比，**DeepSeek-V2** 在某些领域超越了 GPT-4。例如：[DeepSeek-V2 发布公告](https://x.com/deepseek_ai/status/1787478986731429933)。

**2. 微调挑战与提示工程策略**

- 在将 **Llama3** 模型转换为 GGUF 格式时，存在 **保留微调数据** 的困难，并讨论了一个 [已确认的 Bug](https://github.com/ggerganov/llama.cpp/issues/7062)。

- **prompt design** 的重要性以及正确模板的使用（包括 end-of-text tokens），对微调（fine-tuning）和评估期间模型性能的影响。示例：[Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47)。

- **prompt engineering** 策略，例如将复杂任务拆分为多个提示词，研究 **logit bias** 以获得更多控制。示例：[OpenAI logit bias guide](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api)。

- 教导 LLM 在不确定时使用 `<RET>` token 进行 **information retrieval**（信息检索），从而提高在低频查询中的表现。示例：[ArXiv paper](https://arxiv.org/abs/2404.19705)。

**3. 开源 AI 发展与协作**

- 发布 **StoryDiffusion**，这是 Sora 的开源替代方案，采用 MIT 许可证，但权重尚未发布。示例：[GitHub repo](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file)。

- 发布 **OpenDevin**，这是一个基于 Cognition 的 Devin 的开源自主 AI 工程师，并举办了 [网络研讨会](https://lu.ma/fp0xr460)，在 GitHub 上引起了越来越多的关注。

- 呼吁在开源 **machine learning paper**（机器学习论文）上进行协作，该论文旨在预测 IPO 成功，托管在 [RicercaMente](https://edopedrocchi.github.io/RicercaMente/Projects/IPO/indexIPO.html)。

- 围绕 **LlamaIndex** 集成的社区努力，包括在更新后遇到的 Supabase Vectorstore 和包导入问题。示例：[llama-hub documentation](https://github.com/run-llama/llama-hub/tree/main#how-to-add-a-loadertoolllama-pack)。

**4. LLM 创新与训练见解**

- **Gemma 2 以高效训练给人留下深刻印象**：Google 的 **[Gemma 2](https://huggingface.co/blog/gemma2)** 模型体积显著更小，且在更少的 token 上进行训练（9B 模型在 8T token 上训练），得益于知识蒸馏（knowledge distillation）和软注意力上限（soft attention capping）等创新，其在基准测试中超越了 Llama3 70B 等竞争对手。
- **Gemma-2 的显存效率提升了 QLoRA 微调**：新的预量化 **[Gemma-2 4-bit 模型](https://huggingface.co/unsloth/gemma-2-27b-bnb-4bit)** 承诺下载速度提高 4 倍并减少 VRAM 碎片，充分利用了 QLoRA 微调的效率改进。
- **MCTSr 提升奥数解题能力**：**[MCT Self-Refine (MCTSr)](https://arxiv.org/abs/2406.07394)** 算法将 LLM 与蒙特卡洛树搜索（Monte Carlo Tree Search）相结合，通过系统地优化解决方案，在解决复杂数学问题方面取得了显著成功。
- **Adam-mini 优化器的内存效率**：**[Adam-mini](https://arxiv.org/abs/2406.16793)** 优化器通过利用简化的参数分区方法，以减少高达 50% 的内存使用量，实现了与 AdamW 相当或更好的性能。

**5. 安全 AI 与伦理考量**

- **Rabbit R1 的安全漏洞在 YouTube 上曝光**：一段名为 **["Rabbit R1 makes catastrophic rookie programming mistake"](https://youtu.be/lkbV8oP-F44)** 的 YouTube 视频揭露了 Rabbit R1 代码库中硬编码的 API 密钥，危及了用户数据安全。
- **AI 使用限制警告与政策合规性**：成员们强调了过度挑战 AI 边界的风险，警告违反 **[OpenAI 的使用政策](https://openai.com/policies/usage-policies)** 可能会导致账号封禁或终止。
- **开源 AI 辩论**：激烈的讨论权衡了开源 AI 模型的利弊，在潜在的滥用风险与访问民主化以及受限 AI 的经济影响之间寻找平衡，同时考虑了其益处和危险。

**6. 实用 AI 集成与社区反馈**

- **高显存需求的 AI 视频生成**：**ExVideo** 的成功实现生成了令人印象深刻的视频结果，尽管需要大量的 VRAM (43GB)，这展示了 AI 能力与硬件限制之间持续的平衡。
- **跨平台模型实现的问题**：在 [LM Studio](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF) 等平台上集成 **Gemma 2** 等模型时，需要手动修复和最新更新以确保最佳性能。
- **RAG 与 API 限制的挑战**：Perplexity 的 **RAG 机制** 因输出不一致而受到批评，同时 **Claude 3 Opus** 等模型也存在局限性，表现出在上下文处理和 API 性能方面的挣扎。

**7. 数据集与基准测试进展**

- **REVEAL 数据集基准测试验证器**：**[REVEAL 数据集](https://reveal-dataset.github.io)** 对 Chain-of-Thought 推理的自动验证器进行了基准测试，强调了在开放域 QA 设置中验证逻辑正确性的难度。
- **用于鲁棒性测试的 XTREME 和 SPPIQA 数据集**：讨论了 **[XTREME](https://huggingface.co/datasets/google/xtreme)** 和 **[SPPIQA](https://huggingface.co/datasets/google/spiqa)** 数据集，分别侧重于评估多语言模型的鲁棒性和多模态问答能力。
- **基于事实的响应生成器的重要性**：通过 [Glaive-RAG-v1](https://huggingface.co/datasets/glaiveai/RAG-v1) 等数据集强调了对提供可靠、基于事实响应的模型的需求，并探讨了用于质量提升的评分指标。

**8. 协作与开发平台**

- **使用 LlamaIndex 构建 Agent 服务**：工程师可以利用 **[LlamaIndex notebook](https://t.co/WYTCaqs6Yb)** 中分享的资源创建向量索引并将其转换为查询引擎，从而增强 AI 服务的部署。
- **Featherless.ai 提供无需 GPU 配置的模型访问**：[Featherless.ai](https://featherless.ai/) 推出了一个平台，提供对 Hugging Face 上 450 多个模型的固定费率访问，满足社区在模型优先级和使用场景方面的需求。
- **LangGraph Cloud 增强 AI 工作流**：LangChainAI 推出的 **[LangGraph Cloud](http://bit.ly/langgraph-cloud-blog-1)** 承诺为 AI Agent 提供强大、可扩展的工作流，并集成了监控和追踪功能以提高可靠性。

---

# PART 1: 高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma-2 变得精简高效**：全新的 **Gemma-2-27B 和 9B 的预量化 4-bit 版本** 现已发布，具有更快的下载速度和更少的 VRAM 碎片，有利于 [QLoRA 微调](https://huggingface.co/unsloth/gemma-2-27b-bnb-4bit)。

- **AI 开发中的操作系统大辩论**：社区内关于在 AI 开发中使用 **Windows vs. Linux** 的优劣展开了积极辩论，涉及 Linux 上的外设兼容性问题，以及尽管 Windows 存在某些限制但用户普遍更倾向于 Linux 的观点。

- **放大镜下的 Hugging Face 评估系统**：社区将 Hugging Face 的评估系统比作“流行度竞赛”，并提出了引入高级付费评估的概念，建议“*只要用户愿意付费，应允许随时进行评估*”。

- **大数据，大麻烦**：关于处理 **2.7TB Reddit 数据集** 的讨论指出，清理这些数据需要巨大的资源，解压后可能会膨胀到“约 15 TB……而得到的数据质量充其量也就一般”。

- **显存极限下的 AI 视频生成**：据报道，使用 **ExVideo 生成视频内容** 效果显著，但它需要高达 43GB 的 VRAM，这凸显了 AI 能力与资源可用性之间不断的权衡。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Gemma 2 表现优于竞争对手**：Google 新发布的 **Gemma 2** 模型已集成到 Transformers 库中，其优势包括尺寸效率——**比 Llama3 小 2.5 倍**——且 **27B 模型**在 **13T tokens** 上进行了强大的训练。诸如**知识蒸馏**和**交替使用局部与全局注意力层**等创新旨在增强推理稳定性并减少内存占用，HuggingFace 的博客文章涵盖了详细的 [Gemma 2 细节](https://huggingface.co/blog/gemma2)。

- **解读 Elixir 的新文件系统**：Elixir 的 FSS 引入了支持 HTTP 的文件系统抽象，并讨论了非扩展性问题。同时，HuggingFace 首个开源的基于图像的检索系统也引起了关注，其中 [宝可梦数据集示例](https://huggingface.co/spaces/not-lain/RAG-on-images) 大受欢迎，此外，像 [CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4) 这样的视觉学习模型项目也备受瞩目。

- **AI 助力多语言应用起飞**：U-C4N 的多语言实时航班跟踪应用程序展示了航空与语言的交汇，而 PixArt 的 900M 新变体则在 Spaces [竞技场](https://huggingface.co/spaces/ptx0/PixArt-900M) 中寻求合作。此外，AI 与音乐叙事在 [Bandcamp](https://vonpsyche.bandcamp.com/album/the-prompt) 等平台上的融合打破了流派界限。

- **准备好，Gradio！**：Gradio 用户需注意，请务必更新到 3.13 以上版本，以避免分享链接失效。遵守此要求将确保能够持续访问 Gradio 资源，操作非常简单，只需运行 `pip install --upgrade gradio`。

- **闪电般的机器学习速度**：一个超高速的 YouTube 教程制作了关于 **10 个关键机器学习算法** 的 1 分钟速成课程，非常适合时间紧迫但渴望知识的人。点击[此处](https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5)查看这节快速课程。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Gemma 2 集成虽有小瑕疵**：最新的 **LM Studio 0.2.26 版本** 增加了对 **Gemma 2** 模型的支持，尽管一些用户报告了集成 bug 和困难。为了解决这些问题，建议手动下载并重新安装配置，并注意某些架构（如 ROCm）仍待支持。

**Gemma-2 令人困惑的能力**：关于 **Gemma-2** 上下文限制的信息差异导致了混乱，有报告称限制为 4k，而另一些则称是 8k。此外，推荐支持叙事模型 [ZeusLabs/L3-Aethora-15B-V2](https://huggingface.co/ZeusLabs/L3-Aethora-15B-V2)，而对于 **Deepseek coder V2 Lite** 等模型，建议用户关注 [GitHub pull requests](https://github.com/ggerganov/llama.cpp/pull/8156) 以获取支持状态的更新。

**Snapdragon 在 LM Studio 中表现出色**：用户称赞了 Snapdragon X Elite 系统与 **LM Studio** 的兼容性，指出其在 CPU/内存任务效率上明显优于 i7 12700K，尽管在特定任务中仍逊色于 4090 GPU。

**针对多 Agent 框架的精细化讨论**：关于模型效能的讨论表明，**0.5B 模型** 或许可以轻松地在多 Agent 框架中充当用户代理；然而，对于此类低端模型处理编码任务的能力，人们仍持怀疑态度。对于硬件爱好者，关于使用双显卡价值的咨询得到了肯定的回答。

**关于 ROCm 兼容性的分歧与 Gemma 2 的首次亮相**：在 **AMD ROCm 技术预览** 频道中，提出了关于 AMD GPU 对 Gemma 2 模型支持的问题，引导用户查看 [GitHub 说明](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#installation-on-windows-0226-) 中描述的针对 **Windows** 新发布的 0.2.26 ROCm “扩展包”。此外，Gemma 2 的发布既伴随着兴奋也伴随着批评，一些用户将其贴上“烂透了”的标签，而另一些用户则急切期待未来更新中承诺的改进。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**AI 使用警告**：一场讨论强调了测试 AI 极限的风险，并给出了明确警告：违反 [OpenAI 的使用政策](https://openai.com/policies/usage-policies) 可能导致账号被暂停或终止。

**开源 AI 辩论**：工程社区就 AI 模型的开源进行了辩论；讨论对比了滥用风险与访问民主化，强调了限制访问的经济影响以及为了公共安全进行监管的必要性。

**RLHF 训练困扰用户**：关于 Reinforcement Learning from Human Feedback (RLHF) 的对话揭示了用户对其偶尔出现的提示以及 OpenAI 处理公开 RLHF 训练的不透明性的困惑。

**AI 集成的成功与挑战**：成员分享的经验包括使用自定义 GPT 执行特定任务（如生成医疗问题）时遇到的问题，以及将 AI 模型和 API 与其他服务集成以增强功能的成功案例。

**Prompt Engineering 见解**：成员们交流了 Prompt Engineering 的技巧，建议保持简单和简洁，并探讨了使用 "logit bias" 进行更深层的 Prompt 控制，还简要触及了随机神经网络的准确定性本质。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **自定义风格数据集引发关注**：参与者讨论了创建具有自定义风格的数据集，指出无意中生成 NSFW 内容的潜在风险。社区强调了监控图像生成的复杂性，以避免平台封禁。
  
- **因 Automatic1111 体验不佳转向 Forge**：由于 **Automatic1111** 的问题，用户正在探索 **Forge** 等替代方案，尽管它也面临自身的内存管理挑战。YouTube 上分享了一个 [Stable Diffusion Webui Forge 简易安装](https://www.youtube.com/watch?v=FKzvHFtc8N0&t=64s) 视频作为安装指南。

- **用户请求恢复 Cascade 频道**：许多成员表达了恢复 **Cascade** 频道的愿望，因为它过去有许多富有成效的讨论，这引发了关于公会方向以及可能将重点转向 **SD3** 的辩论。

- **深入探讨模型训练细节**：关于模型训练的对话涉及了 **LoRa** 训练的细微差别、**3m sde exponential** 等采样器以及 VRAM 限制。还考察了 **ComfyUI**、**Forge** 和 **Stable Swarm** 等工具的有效性和局限性。

- **Discord 社区呼吁透明度**：部分社区成员对删除频道和资源表示不满，促使了关于透明沟通和保留用户生成内容重要性的讨论。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Scarlet AI 进军项目管理**：用于规划复杂项目的 **Scarlet AI** 预览版已推出；尽管尚未正式发布，感兴趣的工程师可以在 [Scarlet AI Preview](https://app.scarletai.co/) 评估其功能。
  
- **Character.AI 拨入语音功能**：**Character.AI** 推出的全新 **Character Calls** 功能允许通过电话进行 AI 交互，适用于面试演练和 RPG 场景。该功能在其 [移动端应用演示](https://share.character.ai/Wv9R/6tdujbbr) 中进行了展示。

- **Meta 通过 LLM 优化编译器代码**：Meta 推出了 **Large Language Model Compiler**，旨在改进编译器优化任务，其 [研究出版物](https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/) 对细节进行了深入探讨。

- **LangGraph Cloud 的基础设施创新**：**LangChainAI** 推出了 **LangGraph Cloud**，承诺为 AI Agent 提供具有弹性、可扩展的工作流，并结合了监控和追踪功能；更多见解请参考其 [发布博客](http://bit.ly/langgraph-cloud-blog-1)。

- **Adept 在与 Amazon 合作之际发生领导层变动**：有消息称 **Adept** 正在调整其战略，多位联合创始人将加入 Amazon 的 AGI 团队；更多信息请参阅 [GeekWire 文章](https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/)。

- **OpenAI Demo 即将火热登场**：公会成员收到通知，OpenAI Demo 即将举行，建议成员立即访问专门的 [OpenAI Demo 频道](https://discord.com/channels/822583790773862470/1197350122112168006)。

- **GPT-4o 准备重塑桌面端编程**：公会讨论了采用 GPT-4o 辅助桌面端编程，并分享了如 `Open-Interpreter` 等可以轻松与本地模型集成的配置。

- **当企鹅更喜欢苹果时**：Linux 用户在流媒体方面的困扰引发了一场半幽默半认真的 Mac 优势对比，并引出了 **Vesktop**——一款提升性能的 Linux 版 Discord 应用，可在 [GitHub](https://github.com/Vencord/Vesktop) 上找到。
  
- **AI 社区的泄露与分享**：有关敏感 GPT 定义在 [GitHub](https://github.com/LouisShark/chatgpt_system_prompt/tree/main/prompts/gpts) 等平台流出的讨论，引发了对隐私问题的关注。成员们还交流了学术资料，介绍了 CoALA 框架和语言 Agent 仓库，可在 [arXiv](https://arxiv.org/abs/2309.02427) 和 [GitHub](https://github.com/ysymyth/awesome-language-agents) 上找到。

- **对同行演讲的赞赏**：成员们对一位同行准备充分的演讲表示赞赏，强调了高质量演示在 AI 领域的重要性。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLM 的指令预训练表现优异**：在大型语料库的预训练中加入 **2 亿条指令-响应对** 提升了性能，使得轻量级的 **Llama3-8B** 能够与 **Llama3-70B** 等重量级模型并驾齐驱。关于高效指令合成器的细节见 [Instruction Pre-Training 论文](https://arxiv.org/abs/2406.14491)，模型已在 [Hugging Face](https://huggingface.co/instruction-pretrain/finance-Llama3-8B) 上可用。

- **MCTSr 与 LLM 融合解决奥数难题**：将 Large Language Models 与 **蒙特卡洛树搜索** (MCTSr) 相结合，在解决奥林匹克数学竞赛问题上取得了显著成功。该技术的内在机制在 [详细研究](https://arxiv.org/abs/2406.07394) 中有深入阐述。

- **数据集大爆发：SPPIQA, XTREME, UNcommonsense**：Nous Research AI 频道讨论了一系列数据集，包括用于推理的 [SPPIQA](https://reveal-dataset.github.io)、用于多语言模型评估的 [XTREME](https://huggingface.co/datasets/google/xtreme)，以及用于探索怪诞维度的 [UNcommonsense](https://huggingface.co/datasets/allenai/UNcommonsense)。

- **Hermes 2 Pro 发布，增强函数调用功能**：**Hermes 2 Pro 70B** 模型正式发布，重点改进了函数调用（Function Call）和结构化 JSON 输出，在评估中分别获得 90% 和 84% 的评分。虽然没有提供学术论文，但可以在 [NousResearch 的 Hugging Face](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B) 探索该模型。

- **辩论 SB 1047 对 AI 的束缚**：成员们激烈辩论了加州的 SB 1047 法案是否会阻碍 AI 的增长。一项反对该法案的 [运动](https://live-2024-stop-sb-1047.pantheonsite.io) 警告称，它可能会抑制 AI 蓬勃发展所必需的冒险精神。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **“诅咒式”复杂度 vs. 实际性能**：**yoco** 架构的 **kv cache** 策略引发了辩论，批评者认为其偏离了标准的 Transformer 实践且过于复杂。讨论还涵盖了模型中 Attention 和 Feed-forward 层的顺序，一些人提出非标准层顺序可以带来效率提升，而另一些人则对其性能收益持怀疑态度。

- **超越传统认知的 Scaling**：围绕 **Scaling laws** 的讨论质疑了 **Challax scaling** 模型的统治地位，一些参与者建议将 “Scaling laws” 视为临时性的，更准确地应称为 “Scaling heuristics”。引用了诸如 [“Parameter Counts in Machine Learning”](https://www.alignmentforum.org/posts/GzoWcYibWYwJva8aL/parameter-counts-in-machine-learning) 等论文来支持关于不同 Scaling 模型有效性的观点。

- **数据隐私困境**：对话探讨了在 **privacy-preserving/federated learning** 背景下的隐私担忧，其中聚合数据暴露了更广泛的攻击空间。讨论了 AI agents 实施安全行为的潜力，考虑了上下文行为识别以及对隐私泄露的主动响应。

- **LLM 评估与创新**：引入了一个新的**推理挑战数据集** **MMLU-SR**，并考虑将其添加到 **lm_eval** 中，通过修改后的问题探测 LLM 的理解能力。分享了该数据集的 [arXiv 论文](https://arxiv.org/abs/2406.15468v1) 链接以及 **MedConceptsQA** 基准测试[添加](https://github.com/EleutherAI/lm-evaluation-harness/pull/2010)的 GitHub PR。

- **GPTNeoX 中的 Instruction Tuning 潜力**：关于 GPTNeoX 中 **Instruction tuning** 的咨询（特别是选择性反向传播损失）引发了讨论，引用了一个正在进行的 [PR](https://github.com/EleutherAI/gpt-neox/pull/1240) 和一个预处理脚本 “[preprocess_data_with_chat_template.py](https://github.com/EleutherAI/gpt-neox/blob/main/tools/datasets/preprocess_data_with_chat_template.py)”，标志着定制化训练工作流的积极开发。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Windows 上的 Triton 困扰**：用户报告了在 Windows 上使用 `torch.compile` 时遇到的 **Triton 安装问题**，导致出现 "RuntimeError: Cannot find a working trilog installation"。这表明 Triton 可能尚未在 Windows 上获得官方支持，因此需要寻找替代安装方法。

- **使用 torch.compile 进行张量修补**：[Lovely Tensors](https://github.com/xl0/lovely-tensors) 的作者遇到了 `torch.compile()` 报错，原因是 `Tensor.__repr__()` 在 FakeTensor 上被调用。社区建议利用 [torch.compiler fine-grain APIs](https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html) 来缓解此类问题。同时，NCCL 的一次更新解决了旧版本中的广播死锁（broadcast deadlock）问题，详情见[此 PR](https://github.com/huggingface/text-generation-inference/pull/2099)。

- **装备 CUDA 知识**：Stephen Jones 带来了 [CUDA 编程的深度概述](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62401/)，涵盖了 *波次量化与单波次内核 (wave quantization & single-wave kernels)*、并行性以及通过分块 (tiling) 优化 L2 cache 性能等技术。

- **CUDA 好奇心与云端查询**：成员们分享了 [Vast.ai](https://vast.ai/) 和 [Runpod.io](https://www.runpod.io/) 等平台，用于在云端 GPU 上探索 CUDA，并建议从 `torch.compile` 开始，逐步过渡到 Triton 或为 Python 编写自定义 CUDA 代码以进行优化。

- **PMPP 出版谜团**：一位成员指出 PMPP（第 4 版）书的纸质版缺失了若干页，引发了关于是否有类似经历的询问。

- **torch/aten Ops 列表与 Bug 狩猎**：torchao 频道出现了对 `FSDP` 等张量子类所需的 `torch/aten ops` 完整列表的需求，讨论了 `__torch_dispatch__` 的递归错误，以及 [Int4Tensor 的重构 PR](https://github.com/pytorch/ao/pull/458)。此外，还有关于 GeForce GTX 1650 缺乏原生 bfloat16 支持的提醒。

- **HuggingFace Hub 的喧嚣**：off-topic 频道热烈讨论了直接在 HuggingFace Hub 上存储模型架构和预处理代码的 *优缺点*。关于模型代码和权重存储的最佳实践存在争论，[Llama 模型](https://github.com/meta-llama/llama)被引用为有效发布策略的案例研究。

- **Gemma 2 成为焦点**：Google 推出的 Gemma 2 模型（拥有 27B 和 9B 参数）在基准测试中表现优于竞争对手，社区对其[开放性表示赞赏](https://x.com/reach_vb/status/1806343018640781675)，并期待更小的 2.6B 变体。讨论还集中在架构选择上，如 [近似 GeGLU 激活函数](https://x.com/danielhanchen/status/1806372357684220308) 以及 ReLU 与 GELU 之争，并辅以[学术研究](https://arxiv.org/pdf/2002.05202v1)支持。FP8 支持方面的硬件挑战导致人们提到了 NVIDIA 库的局限性以及 [Microsoft 在 FP8-LM 上的工作](https://arxiv.org/pdf/2310.18313)。Yuchen 的训练见解表明，在针对 H100 GPU 进行优化时，可能存在平台或数据集特定的问题。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 为公益事业降低成本**：Perplexity 为其 **Enterprise Pro** 产品推出了降价方案，旨在支持学校和非营利组织的社会及教育事业。有关资格和申请的更多信息可以在其 [公告](https://pplx.ai/s3oZX49) 中找到。

- **RAG 挫败感与 API 焦虑**：在 **Perplexity AI** 的讨论中，用户对 **RAG**（**Relevance Aware Generation**）机制不稳定的表现感到沮丧，并要求获取更大的模型，如 **Gemma 2**。此外，用户在使用 **Claude 3 Opus** 时遇到了限制，理由是使用上限多变且具有约束性。

- **安全第一，修复待定**：安全协议已得到处理，引导成员前往 [Trust Center](https://trust.perplexity.ai/) 了解有关 **数据处理** 和 **PII 管理** 的信息。同时，成员建议使用 "#context" 来改进持续性处理，以解决交互中持续存在的上下文保留问题。

- **功能探索与改进**：社区注意力转向探索 **Android 14** 的增强功能，同时提出了 **Minecraft 的机制可能误导儿童** 的问题。针对如何过滤 API 结果以获取近期信息的咨询，社区提供了使用特定日期格式的指导。

- **技术深挖与创新亮点**：分享的内容包括对 **Android 14** 的见解、对 **Linux 性能** 的批评、**Robot Skin** 的创新用途，以及受牡蛎启发的各种可持续建筑。一个值得注意的分享讨论了对 **Minecraft 修复机制** 的批评，认为这可能导致误解。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Character.AI 率先推出双向 AI 语音聊天**：[Character.AI 推出了 Character Calls](https://blog.character.ai/introducing-character-calls/)，实现了与 AI 的语音对话，尽管该体验因 5 秒的延迟和不够流畅的交互而略显逊色。与此同时，行业传闻称亚马逊聘用 Adept 的联合创始人并进行技术授权，使得 Adept 实力削弱，此外还有关于 Adept 拥有毒性工作环境的未经证实的指控。

- **AI Agent 进展落后于炒作曲线**：讨论将 **AI Agent** 的缓慢进展与自动驾驶汽车行业进行了类比，认为炒作超过了实际表现。**AI Agent** 训练数据的质量和来源（包括对合成数据日益增长的关注）被强调为关键挑战。

- **SnailBot News 节目引发讨论**：SnailBot News 最新一期关于 Lina Khan 的节目引发了热议；Natolambert 预告了对 Ross Taylor 和 John Schulman 等知名人物的采访。围绕“请勿在我们的模型输出上进行训练”的数据使用条件的伦理考量也成为了焦点。

- **Scaling 占据 AI 话语权**：怀疑论围绕着“仅靠 Scaling（扩展）就能实现 AGI”的信念，正如 [AI Scaling Myths](https://www.aisnakeoil.com/p/ai-scaling-myths) 中所指出的，同时还讨论了 **LLM** 开发者在高质量数据方面所谓的局限性。Nathan Lambert 敦促对这些观点进行批判性审查，并引用了 [Substack 讨论](https://open.substack.com/pub/aisnakeoil/p/ai-scaling-myths?r=68gy5&utm_campaign=comment-list-share-cta&utm_medium=web&comments=true&commentId=60317135) 和合成数据的最新进展。

- **对 AI 与全球事务的多样化思考**：从 Anthropic CEO 对《最终幻想》的热爱凸显了 AI 领导者人性化的一面，到关于 AI 危机可能比大流行病更复杂的辩论，成员们参与了多样化的对话。一些讨论甚至考虑了智能爆炸如何重塑政治结构，反映了 AI 发展的深远影响。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**LlamaIndex 助力 Agent 服务**：工程师们探索了使用 **LlamaIndex** 构建 **agentic RAG services**，讨论了创建 vector indexes 并将其转换为 query engines 的过程。详细步骤和示例可以在最近分享的 [notebook](https://t.co/WYTCaqs6Yb) 中找到。

**Jina 的 Reranking 革命**：LlamaIndex 社区对 **Jina's newest reranker** 讨论热烈，该工具被誉为他们迄今为止最有效的重排序器。相关细节可见[此处](https://t.co/YsYoVOIirb)。

**Vector Retrievers 中的 Node 权重难题**：AI 从业者正在解决 **LlamaIndex's embedding challenges**，讨论了诸如哪些 node 部分需要进行 embedding，以及模型不匹配导致 vector retrievers 结果不佳等因素。共识倾向于创建简单的测试用例以进行有效的 debugging。

**通过 Edges 进行实体链接**：增强 **entity relationship detection** 引发了辩论，重点在于添加受 embedding 逻辑启发的 edges。人们对可能与 **Neo4j** 合作的技术分享充满期待，预计这将揭示先进的 entity resolution 技术。

**Claude 和 OpenAI Keys 出现问题**：讨论中提到了需要修复 **Claude** 因 Bedrock 的 token 限制导致的空响应问题，以及特定情况下的 *IndexError*；此外还有一个奇怪的环境行为，即代码设置的 **OpenAI keys** 似乎被覆盖了。工程师们还在探索 batch 和 parallel index loading 的优化，旨在加速大文件处理。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Gemma 的多语言实力**：虽然 **Gemma 2** 官方仅支持英语，但用户报告其具有出色的多语言能力，并针对其在韩语中的表现进行了具体咨询。

**模型迁移热潮**：根据[公告](https://openrouter.ai/models/google/gemma-2-9b-it)，包含免费版和标准版的 **Gemma 2.9B** 模型正在席卷而来，同时多款热门模型降价，包括 [Dolphin Mixtral](https://openrouter.ai/models/cognitivecomputations/dolphin-mixtral-8x22b) 降价 10%，[OpenChat](https://openrouter.ai/models/openchat/openchat-8b) 降价 20%。

**OpenRouter 的已知问题**：OpenRouter 严密的审核与 AWS 等平台形成对比；与此同时，用户面临着没有企业支持就无法使用 Opus 的困境，并正在解决 Gemini 模型不规范 API 导致的 **Status 400** 错误。

**Passphrase 难题与 API 分配问题已解决**：工程师们分享了使用 `ssh-add -A` 进行无缝 GitHub 身份验证的心得，并讨论了通过观看 Simon Willison 关于 LLM APIs 的概述来获取启发，资源可在 [YouTube](https://www.youtube.com/watch?v=5zE2sMka620&t=2026s) 及其[博客](https://simonwillison.net/2024/Jun/27/ai-worlds-fair/)上找到。

**AI 偏好调整**：采纳 **daun.ai** 的建议，将默认模型设置为 'auto' 以获得稳定的结果，或者尝试使用 'flavor of the week' 作为备选方案，以确保各项任务的持续生产力。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Gege AI 以新声献唱**：音乐创作工具 **Gege AI** 可以通过少量音频样本模仿任何歌手的声音，引发了关于颠覆音乐产业潜力的幽默评论，以及对 **RIAA** 反应的猜测。
  
- **用户对 Gege AI 和 GPT-4 模型感到沮丧**：用户报告了注册 **Gege AI** 的困难，并伴有关于社交信用的调侃；另一些人则对 **GPT-4 和 4O 模型** 的表现表示失望，认为它们过于刻板，在编程任务上不如 **GPT-3.5** 等早期版本。

- **Adam-mini 优化器减少内存浪费**：根据讨论中提到的一篇最新论文，**Adam-mini** 优化器提供与 **AdamW** 相当或更好的性能，同时通过划分参数并为每个 block 分配单一学习率，可减少 45-50% 的内存消耗。
  
- **Gemma 27B：怀疑与雄心并存**：虽然新的 **Gemma 27B** 模型据报道显示出了一些有前景的性能提升，但成员们因其高置信区间而保持谨慎，质疑其相对于之前版本的整体优势。

- **转向 Claude 以获得更顺畅的体验**：鉴于 **OpenAI's models** 的问题，一些成员选择了 **Claude**，因为它具有卓越的 artifacts 功能以及与 **Hugging Face libraries** 更好的集成，报告称其体验比 **GPT-4** 模型更顺畅。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Bedrock 让工程师感到困惑**：工程师们分享了在将 **csv_agent** 和 **pandas_dataframe_agent** 与 **Bedrock** 集成时遇到的挑战，以及在使用 `ChatPromptTemplate.fromMessages` 配合 **Sonnet 3.5 model** 和 Bedrock 时遇到的错误，这表明可能存在兼容性问题。

- **LangGraph 发布，Human-in-the-Loop 功能遭遇困境**：**LangGraph** 引入了 Human-in-the-loop 功能，特别是 "Interrupt" 和 "Authorize"，但在人工审批后恢复执行时出现了反序列化错误，正如 [LangChain Blog](https://blog.langchain.dev/human-in-the-loop-with-opengpts-and-langgraph) 中所讨论的。

- **JSONL 编辑工具的改进以及使用 Matryoshka Embeddings 的 RAG**：社区成员传阅了一个用于编辑 JSONL 数据集的工具 ([uncensored.com/jsonl](https://uncensored.com/jsonl))，并分享了关于使用 Matryoshka Embeddings [构建 RAG](https://x.com/Prashant_Dixit0/status/1806580075447590974) 以提高检索速度和内存效率的见解，并附带了 [Colab tutorial](https://colab.research.google.com/github/lancedb/vectordb-recipes/blob/main/tutorials/RAG-with_MatryoshkaEmbed-Llamaindex/RAG_with_MatryoshkaEmbedding_and_Llamaindex.ipynb)。

- **Dappier 创造 AI 内容变现机会**：[TechCrunch](https://techcrunch.com/2024/06/26/dappier-is-building-a-marketplace-for-publishers-to-sell-their-content-to-llm-builders/) 报道的 [Dappier platform](https://www.producthunt.com/posts/dappier-2-0) 为创作者提供了一个市场，通过 RAG API 授权内容用于 AI 训练，这为专有数据持有者标志着一条新的收入来源。

- **Testcontainers Python SDK 助力 Ollama**：Testcontainers Python SDK 现在支持 Ollama，增强了通过 Ollama 运行和测试 Large Language Models (LLMs) 的便利性，该功能在 **4.7.0** 版本中可用，并附带了使用示例 ([pull request #618](https://github.com/testcontainers/testcontainers-python/pull/618))。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojolicious 与 Mojo 之间的混淆已解决**：当一名用户请求 **Mojo** 代码示例却收到基于 Perl 的 **Mojolicious** 示例时引起了混淆；但随后澄清了该请求是关于 **Modular** 的 AI 开发语言 **Mojo** 的信息，该语言因其类 Python 的增强能力和类 C 的稳健性而备受推崇。

- **陷入 REPL 网络陷阱**：有报告称 **Mojo REPL** 存在异常，它会静默连接然后毫无预警地关闭，这引发了关于可能开启一个 GitHub issue 以识别并解决这个神秘连接难题的讨论。

- **Nightly 版本笔记：新编译器和 Graph API 切片**：**Modular** 最新的 nightly 版本 '2024.6.2805' 包含一个具有 LSP 行为调整的新编译器，并建议使用 `modular update nightly/mojo`；开发者还需要注意增加了 *"跨维度的整数型字面量切片"*，并建议通过 issues 记录新功能请求以便追溯。

- **SDK 遥测技巧与 MAX 回归**：分享了关于在 **Mojo SDK** 中禁用遥测的指导，并提供了一个有用的 [FAQ link](https://docs.modular.com/mojo/faq#does-the-mojo-sdk-collect-telemetry)；**MAX nightly builds** 再次运行，欢迎试用 *Llama3 GUI Chatbot* 并通过给出的 [Discord link](https://discord.com/channels/1087530497313357884/1256010477637730315/1256010477637730315) 提供反馈。

- **会议标记与社区资料**：社区正在为下一次 **Mojo Community meeting** 做准备，计划在未指定的 *当地时间* 举行，详情可通过 [Zoom](https://modul.ar/community-meeting-zoom) 和 [Google Docs](https://modul.ar/community-meeting-doc) 获取；此外，还向加拿大和美国的节日庆祝者表达了亲切问候。同时，点击 [Modular.com](https://www.modular.com/newsletters/modverse-weekly-38) 即可通过 **Modverse Weekly - Issue 38** 保持信息同步。



---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **社区模型获准加入**：**Torchtune** 团队表达了对社区贡献模型的兴趣，鼓励成员分享自己的实现并增强库的通用性。
- **调试文本补全难题**：**Torchtune** 文本补全中的一个令人困惑的问题被追踪到是由数据集错误插入的句末（EOS）token 引起的，详情见 [GitHub 讨论](https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_text_completion.py#L56)。
- **PreferenceDataset 备受青睐**：对于强化学习应用，**PreferenceDataset** 成为优于文本补全数据集的选择，能更好地对齐“首选”输入-响应对的奖励。
- **预训练机制澄清**：讨论澄清了预训练机制，特别是它涉及用于 token 预测的完整文档，而不是处理碎片化的输入-输出对。
- **EOS Token：加还是不加？**：社区对在 **Torchtune** 的文本补全数据集中引入 **add_eos** 标志进行了讨论并达成一致，解决了近端策略优化（PPO）实现中的一些问题。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**新一代数据科学 IDE 预警**：工程师们讨论了 [Positron](https://github.com/posit-dev/positron)，这是一个前瞻性的数据科学 IDE，已在 [#general](https://discord.com/channels/1238365980128706560/1238365980128706563/1256172869994545183) 频道分享，暗示其对社区的潜在价值。

**摘要生成障碍赛**：出现了一个关于从患者记录中生成结构化摘要的技术咨询，重点在于如何使用 Llama 模型避免幻觉；社区正在征集 Prompt Engineering 和微调策略。

**LLAMA 部署风波**：如 [#🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1255989194275295242) 频道所述，将 **LLAMA** 部署到 Streamlit 时出现了本地环境未见的错误；另一位成员通过调整数据集路径解决了 **Tinyllama** 的 `FileNotFoundError`。

**额度归属问题**：多位成员报告了关于各种应用额度缺失的问题，包括在 [#fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1256121831342215219) 和 [#openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1256146343139737682) 频道的请求，强调需要解决涉及 `kishore-pv-reddy-ddc589` 等标识符和组织 ID `org-NBiOyOKBCHTZBTdXBIyjNRy5` 的问题。

**链接救急与 Predibase 谜题**：在 [#freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1256256276900483132) 频道，一个失效链接被迅速修复；而 [#predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1255978316498735167) 频道关于 Predibase 额度过期的提问目前仍未得到解答。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **安全且开放：Open Interpreter 应对安全挑战**：讨论了 Open Interpreter 的安全措施，例如在执行代码前需要用户确认以及使用 Docker 进行沙箱隔离，强调了社区反馈对项目安全的重要性。

- **速度与技能：代码模型竞技场**：工程师们对比了各种代码模型，认可 **Codestral** 的卓越性能，而 **DeepSeek Coder** 虽然运行速度更快，但有效性约为 70%。**DeepSeek Coder-v2-lite** 以其极速执行和编码效率脱颖而出，潜力可能超越 **Qwen-1.5b**。

- **资源效率咨询：量化版 SMOL 模型**：由于 RAM 限制，有人咨询以量化格式运行 SMOL 多模态模型的问题，突显了 AI 系统在有限资源环境下的适应性挑战。

- **API 密钥泄露：Rabbit R1 的安全疏忽**：一段 [YouTube 视频](https://youtu.be/lkbV8oP-F44) 揭露了一个重大的安全疏忽，发现 Rabbit R1 在其代码库中硬编码了 API 密钥，这对用户数据安全构成了严重威胁。

- **修改 OpenInterpreter 以支持本地运行**：一位 AI 工程师概述了使用非 OpenAI 供应商本地运行 OpenInterpreter 的过程，并在 [GitHub issue 评论](https://github.com/OpenInterpreter/01/issues/272#issuecomment-2119175075) 中详细说明了调整方法。除了订阅费之外，人们还对额外的 API 相关成本表示了担忧。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**tinygrad 获得新的移植福利**：一个支持微调（finetuning）的新 **port** 已完成，标志着 tinygrad 项目的进展。

**FPGA 在人形机器人领域取得胜利**：一个为期 8 个月的项目利用 **FPGA-based systems** 研发出了高能效的**人形机器人**，这被认为比目前因高功耗而消耗电池寿命的 GPU 系统更具成本效益。

**Shapetracker 的零成本 reshape 革命**：tinygrad 中的 *Shapetracker* 允许在不改变底层内存数据的情况下进行张量变形（tensor reshaping），这在 [Shapetracker explanation](https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html) 中有详细说明，成员们讨论了其相对于传统内存步长（memory strides）的优化。

**模型存储的新旧结合**：根据 George Hotz 的说法，在 tinygrad 中，权重由 **safetensors** 处理，计算由 **pickle** 处理，这表明了当前模型存储的方法论。

**对 Shapetracker 血统的好奇**：参与者思考 **Shapetracker** 背后的概念是原创的，还是从现有的深度学习编译器中汲取的灵感，同时赞赏其无需数据复制即可进行优化的能力。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **实习咨询点燃网络**：一名学生将其在 **LLMs and Reinforcement learning** 方面的学术重点与现实应用相结合，向 **Cohere** 员工咨询公司的企业文化和项目。平台上的互动表明，在竞争 AI 领域的实习机会时，展示强大的公开项目组合已达成共识。
- **Cohere 的功能请求**：Cohere 用户对潜在的新功能表现出好奇，促使大家提出可以增强平台产品的建议。
- **AI 博客自动化的愿景**：围绕为博客和社交媒体内容生成设置 **AI-powered automations** 展开了讨论，并将咨询引导至专门的协助渠道。
- **AI Agent 成果发布**：一名成员展示了一个名为 **Data Analyst Agent** 的 AI 项目，该项目使用 **Cohere and Langchain** 构建，并通过 [LinkedIn post](https://www.linkedin.com/posts/eddieotudor_datascience-aitechnology-machinelearning-activity-7212491482287542272-1cSF?utm_source=share&utm_medium=member_ios) 进行了推广。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Gemma2 通过 Pull Request 获得 Sample Packing**：提交了一个 [GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718) 以将 **Gemma2** 与 sample packing 集成。由于需要 Hugging Face 的修复，该 PR 目前处于待定状态，详情见 PR 内部。

- **27b 模型表现平平**：尽管参数量有所增加，但 **27b model** 在基准测试中的表现与 **9b model** 相比并不理想，这表明可能存在扩展或架构效率低下的问题。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Featherless.ai 推出固定费率模型访问**：[Featherless.ai](https://featherless.ai/) 推出了一个平台，以极具竞争力的价格提供对 Hugging Face 上 450 多个模型的访问，基础层级起价为每月 10 美元，无需 GPU 设置或下载。
- **订阅升级**：每月 10 美元的 Feather Basic 计划允许访问最高 15B 的模型，而每月 25 美元的 Feather Premium 计划允许访问最高 72B 的模型，并增加了私密和匿名使用等福利。
- **社区对模型推出的影响**：**Featherless.ai** 正在征求社区对平台模型优先级的意见，强调了目前 AI 角色本地应用以及语言微调（finetuning）和 SQL 模型使用等专门任务的受欢迎程度。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **对 Chatbot Elo 演变的好奇**：用户请求获取超出提供的六周 JSON 数据集之外的更长时间跨度的 **chatbot elo ratings** 数据，表达了对聊天机器人竞技场（chatbot arena）不断演变的竞争格局的兴趣。
- **观察 Elo 竞赛**：从 5 月 19 日开始，注意到领先的聊天机器人在 elo 评分中的“梯队”正在缩短，表明竞争领域非常激烈。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Feature Stores 成为关注焦点**：一场名为“使用 Featureform 和 Databricks 构建企业级 Feature Store”的精彩网络研讨会将于 7 月 23 日上午 8 点（太平洋时间）举行。Simba Khadder 将探讨特征工程的复杂性、Databricks 的利用以及处理大规模数据的路线图，最后设有问答环节。[报名参加以深入了解 Feature Store](https://buff.ly/3zh3B74)。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1255960893062381721)** (549 messages🔥🔥🔥): 

- **Gemma-2 更新：下载速度更快且 VRAM 碎片更少**：已上传 Gemma-2-27B 和 9B 的全新预量化 4-bit 版本，承诺下载速度提升 4 倍，且在进行 QLoRA 微调时 VRAM 碎片减少超过 1GB。[Gemma-2-27B](https://huggingface.co/unsloth/gemma-2-27b-bnb-4bit) 和 [Gemma-2-9B](https://huggingface.co/unsloth/gemma-2-9b-bnb-4bit) 现已在 Huggingface 上线。
- **Windows vs Linux 用于 AI 开发**：成员们讨论了在 AI 开发中使用 Windows 与 Linux 的优缺点。一位用户指出：“Windows 当然没死……但每天都感觉它变得越来越随意，”而另一位用户则表达了对 Linux 上外设兼容性的沮丧。
- **HF 类似 Tiktoker 的评估系统**：几位成员批评了 Hugging Face 的评估系统，将其比作人气竞赛，并建议推出高级付费评估。一位成员表示：*“如果用户愿意为此付费，应该允许随时进行评估。”*
- **大型数据集的挑战**：分享了一个 2.7TB 的 Reddit 数据集，但用户警告说，清理该数据集将耗费大量时间和资源。一位成员估计：*“解压后大约有 15 TB……充其量也就是些平庸的数据。”*
- **使用 ExVideo 进行 AI 视频生成**：多位用户报告了使用 ExVideo 生成视频内容的惊人效果，尽管它需要大量的 VRAM (43GB)。一位成员分享了 [ExVideo Jupyter 的 GitHub 仓库](https://github.com/camenduru/ExVideo-jupyter)链接。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1806410668285030530">来自 Daniel Han (@danielhanchen) 的推文</a>：已将预量化的 4bit bitsandbytes 版本上传至 http://huggingface.co/unsloth。下载速度提升 4 倍，且在进行 QLoRA 微调时减少了超过 1GB 的 VRAM 显存碎片。同时请安装开发版 HF pip...</li><li><a href="https://huggingface.co/datasets/aaronday3/entirety_of_reddit/tree/main">aaronday3/entirety_of_reddit (main 分支)</a>：未找到描述</li><li><a href="https://github.com/beam-cloud/beta9/">GitHub - beam-cloud/beta9：开源的 Serverless GPU 容器运行时。</a>：开源的 Serverless GPU 容器运行时。通过在 GitHub 上创建账号来为 beam-cloud/beta9 的开发做出贡献。</li><li><a href="https://github.com/Alpha-VLLM/Lumina-T2X">GitHub - Alpha-VLLM/Lumina-T2X：Lumina-T2X 是一个统一的 Text to Any Modality Generation 框架</a>：Lumina-T2X 是一个统一的 Text to Any Modality Generation 框架 - Alpha-VLLM/Lumina-T2X</li><li><a href="https://huggingface.co/datasets/aaronday3/entirety_of_reddit">aaronday3/entirety_of_reddit · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://ecnu-cilab.github.io/ExVideoProjectPage/?">ExVideo：通过参数高效的后调优（Parameter-Efficient Post-Tuning）扩展视频扩散模型</a>：未找到描述</li><li><a href="https://ai.meta.com/research/cicero/diplomacy/">未找到标题</a>：未找到描述</li><li><a href="https://github.com/camenduru/ExVideo-jupyter">GitHub - camenduru/ExVideo-jupyter</a>：通过在 GitHub 上创建账号来为 camenduru/ExVideo-jupyter 的开发做出贡献。</li><li><a href="https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10">Subreddit 评论/投稿 2005-06 至 2023-12</a>：未找到描述</li><li><a href="https://x.com/QuanquanGu/status/1805675325998907413">来自 Quanquan Gu (@QuanquanGu) 的推文</a>：我们已经开源了 Self-Play Preference Optimization (SPPO) 的代码和模型！🚀🚀🚀 ⭐ 代码：https://github.com/uclaml/SPPO 🤗模型：https://huggingface.co/collections/UCLA-AGI/sppo-6635f...</li><li><a href="https://foleycrafter.github.io/">FoleyCrafter</a>：未找到描述</li><li><a href="https://github.com/bmaltais/kohya_ss">GitHub - bmaltais/kohya_ss</a>：通过在 GitHub 上创建账号来为 bmaltais/kohya_ss 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth-cli.py">unsloth/unsloth-cli.py (main 分支) · unslothai/unsloth</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html">OpenAI 兼容服务器 — vLLM</a>：未找到描述</li><li><a href="https://ploomber.io/blog/vllm-deploy/">部署 vLLM：分步指南</a>：了解如何部署 vLLM 以高效地提供开源 LLMs 服务</li><li><a href="https://github.com/MC-E/ReVideo">GitHub - MC-E/ReVideo</a>：通过在 GitHub 上创建账号来为 MC-E/ReVideo 的开发做出贡献。</li><li><a href="https://github.com/camenduru/FoleyCrafter-jupyter">GitHub - camenduru/FoleyCrafter-jupyter</a>：通过在 GitHub 上创建账号来为 camenduru/FoleyCrafter-jupyter 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/25Ij9G4haQ">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1255978788466983023)** (16 条消息🔥): 

- **Unsloth 与 Gemma 2 成为技术焦点**：一位用户转发了 Daniel Han 分析 Google 新发布的 Gemma 2 的 [推文](https://x.com/danielhanchen/status/1806372357684220308)，详细介绍了其重要的技术特性，例如 **pre & post layer norms、softcapping attention logits 以及 alternating sliding window/global attention**。Gemma 团队也因提供早期访问权限而受到感谢，尽管 Unsloth 目前尚未支持 Gemma-2 的 finetuning。

- **知识蒸馏引发辩论**：用户们讨论了模型训练中 **Knowledge Distillation (KD)** 的特殊性及其演进。一位用户幽默地提到，“那 2 点 perplexity 的差异 😍”，并观察到了从传统 KD 向“现代”蒸馏方法的转变。

- **推理框架推荐接踵而至**：多位用户为经由 Unsloth 训练的模型寻求并推荐了各种 **inference frameworks**，讨论集中在 multi-GPU 支持和 4-bit loading 等问题上。推荐的框架包括 [vLLM](https://github.com/vllm-project/vllm) 和 [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)，一些用户也指出了现有的 bug 和局限性。

- **Gemma-2-9B finetuning 仍需等待**：社区成员询问了使用 **Unsloth 对 Gemma-2-9B 进行 finetuning** 的可能性，得到的回复澄清目前尚不支持，但正在开发中。

- **Unsloth 与大模型对比**：用户将 9B Gemma-2 等相对较小的模型与更大的模型进行了对比，一些用户对小模型性能的提升表示惊讶。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1806372357684220308">来自 Daniel Han (@danielhanchen) 的推文</a>：刚刚分析了 Google 新发布的 Gemma 2！9B 和 27B 的 base 与 instruct 版本都在这里！1. Pre & Post Layernorms = 像 Grok 一样有 2 倍的 LN。2. 使用了 Grok 的 softcapping！Attn logits 被截断至 (-30, 3...</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: llama.cpp 的 Python 绑定</a>：llama.cpp 的 Python 绑定。通过在 GitHub 上创建账号来为 abetlen/llama-cpp-python 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1255963221349765131)** (113 条消息🔥🔥): 

- **关于瑞典语语言模型训练 LM Head 的讨论**：成员们讨论了为瑞典语训练 lm_head 的价值，其中一位指出，如果模型最初不懂瑞典语，这一点至关重要。另一位强调该过程可以在自己的机器上完成以节省成本，结果将在模型达到 8 个 epochs 后进行测试。

- **推理配置澄清**：一位成员询问了 `FastLanguageModel.for_inference(model)` 与 `model.eval()` 之间的区别。另一位成员解释说，前者用于加载模型，而后者将其切换到评估模式，并指出 Unsloth 的示例 notebooks 使用的是前者方法。

- **语言模型的微调与 VRAM 管理**：成员们讨论了在 RTX 4090 等 GPU 上使用不同 batch sizes 进行微调时的 VRAM 限制。有人分享说，使用 2 的幂次方的 batch sizes 可以避免错误，尽管一些个人经验与之相反。

- **量化模型对 LoRA 的支持**：成员们探讨了在 AWQ 模型上使用 Unsloth adapters 的可行性，并参考了一个支持在量化模型上使用 LoRA 的 GitHub pull request。由于文档和实际案例稀缺，一些人对此表示不确定。

- **持续预训练的问题与解决方案**：一位成员在使用 16GB T4 进行持续预训练后的模型推理时遇到错误。建议包括检查相关的 [GitHub issue](https://github.com/unslothai/unsloth/issues/702#issuecomment-2197477362) 并确保与新版本 PyTorch 没有冲突。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/vllm-project/vllm/issues/5637">[Bug]: 在进程引导阶段 tensor_parallel_size > 1 导致的 RuntimeError · Issue #5637 · vllm-project/vllm</a>: 当前环境 `python collect_env.py` 的输出... 收集环境信息... PyTorch 版本: 2.3.0+cu121 是否为调试构建: False 用于构建 PyTorch 的 CUDA: 12.1 用于构建的 ROCM...</li><li><a href="https://github.com/unslothai/unsloth/issues/702#issuecomment-2197477362">Cache 只有 0 层，尝试访问索引为 0 的层 · Issue #702 · unslothai/unsloth</a>: 我在尝试使用 unsloth 库训练 Phi-3 时遇到 KeyError。错误发生在 model.generate 的生成步骤中。以下是代码和错误详情...</li><li><a href="https://github.com/vllm-project/vllm/pull/5669">[distributed][misc] 为 mp 默认使用 fork，由 youkaichao 提交 · Pull Request #5669 · vllm-project/vllm</a>: 修复了 #5637。在我们创建 cuda context 后，fork 是不安全的。我们应该已经避免在创建 workers 之前初始化 cuda context，所以使用 fork 应该是没问题的，这可以消除必要性...</li><li><a href="https://github.com/vllm-project/vllm/pull/4012">[Core] 支持量化模型上的 LoRA，由 jeejeelee 提交 · Pull Request #4012 · vllm-project/vllm</a>: 基于 #2828 中完成的出色工作。由于 #2828 进展缓慢，我想继续并完成此功能。与 #2828 相比，主要的改进是...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1256051149476200499)** (25 messages🔥): 

- **为 Toki Pona LLM 寻求算力**：一名成员正在为训练 **Toki Pona**（一种高度依赖上下文的语言）的 LLM 寻求算力资源。据称，他们的模型即使只经过一个 epoch 的训练，在资深 Toki Pona 使用者中的评价也优于 ChatGPT-4o。
- **Oracle Cloud 额度提供**：另一名成员提供了几百个即将过期的 Oracle Cloud 额度，并寻求可运行的 **Jupyter notebook**，表示有兴趣使用 Oracle 的 Data Science 平台微调 Toki Pona 模型。
- **讨论 Oracle 平台限制**：讨论了 Oracle 免费试用版的限制，特别是无法启动常规 GPU 实例，因此必须使用 Data Science 平台的 notebook 工作流来进行模型训练和部署。
- **潜在解决方案与建议**：成员们建议适配 **Unsloth colabs** notebook 以在 Oracle 上进行微调，特别是参考韩国语的微调设置。一名成员表示，如果有人能先跑通 notebook，他也愿意尝试 Oracle 平台。
- **Kubeflow 对比**：一名成员将 Oracle 的 notebook 会话功能与典型的 **Jupyter** 设置进行了对比，提到它类似于 **SageMaker** 或 **Kubeflow** 处理机器学习工作流训练和部署的方式。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kubeflow.org/">Kubeflow</a>: Kubeflow makes deployment of ML Workflows on Kubernetes straightforward and automated</li><li><a href="https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-about.htm">Model Deployments</a>: no description found
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1256284481166245908)** (1 messages): 

- **Gemma 2 登陆 Transformers**：Google 发布了 **Gemma 2** 模型，包括 **9B 和 27B** 版本，现已在 Transformers 库中可用。这些模型旨在 **LYMSYS Chat arena** 中脱颖而出，击败了 **Llama3 70B** 和 **Qwen 72B** 等竞争对手。

- **卓越、高效且紧凑**：亮点包括体积比 Llama3 **小 2.5 倍**，且训练使用的 token 数量更少。**27B** 模型在 **13T tokens** 上训练，而 **9B** 模型在 **8T tokens** 上训练。

- **创新的架构增强**：**Gemma 2** 采用了 **knowledge distillation**、**interleaving local and global attention layers**、**soft attention capping** 以及 **WARP model merging** 技术。这些改进旨在提高 **inference** 稳定性，减少内存占用，并修复训练过程中的梯度爆炸问题。

- **无缝集成与易用性**：HuggingFace 宣布 **Gemma 2** 模型现已集成到 **Transformers 库**，并在 **Hub** 上发布。此外还为 **Google Cloud** 和 **Inference Endpoints** 提供了集成，以确保流畅的使用体验。

- **阅读详细信息**：如需深入了解 **Gemma 2** 的架构和技术进步，可查看详尽的 [blog post](https://huggingface.co/blog/gemma2)。鼓励用户查看 **model checkpoints** 和最新的 **Hugging Face Transformers** 版本。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/gemma2">Welcome Gemma 2 - Google’s new open LLM</a>: no description found</li><li><a href="https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315">Gemma 2 Release - a google Collection</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1255961070108151858)** (482 messages🔥🔥🔥):

- **在 Elixir 中探索 FSS**：一位用户提供了 FSS 的概述及[链接](https://hexdocs.pm/fss/0.1.1/FSS.html)，用于 Elixir 中的文件系统抽象。他们讨论了其用例，并指出它支持 HTTP 但似乎不具备可扩展性。
- **Gemma 2 与 GPT-4 参数讨论**：用户讨论了 [Google 公告](https://developers.googleblog.com/en/gemma-family-and-toolkit-expansion-io-2024/)中关于 Gemma 2 的内容。一些对话提到尝试不同的 AI 模型（如 Gemma）及其性能，并对模型带来的挫败感和奇怪行为进行了幽默调侃。
- **新型图像检索系统**：用户宣布使用来自 HF 的开源工具创建了首个基于图像的检索系统。他们分享了自己的兴奋之情以及 [Colab 实现链接](https://colab.research.google.com/drive/1DO5FwwLNDimh6B7T5BX9vNdFPtu1f_Pq?usp=sharing)，并提供了一个用于协作的 [Space](https://huggingface.co/spaces/not-lain/RAG-on-images)。
- **视觉学习模型讨论**：分享了关于视觉学习模型的建议和经验，建议关注 [CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4) 和 [Phi-3-Vision-128K-Instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)。
- **关于 HuggingFace 工具的查询**：用户提出了与特定 HuggingFace 工具和实现相关的问题，包括微调指南和 Gemma 2 等新模型的访问令牌（access tokens）。分享了一个指向 [HuggingFace 训练文档](https://huggingface.co/docs/transformers/en/training)的链接。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://filesystem-spec.readthedocs.io/en/latest/">fsspec: Filesystem interfaces for Python &mdash; fsspec 2024.6.0.post1+g8be9763.d20240613 文档</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/Vipitis/shadermatch">ShaderMatch - Vipitis 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4">THUDM/cogvlm2-llama3-chat-19B-int4 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>：让每个人都能使用社区最优秀的 AI 聊天模型。</li><li><a href="https://huggingface.co/coqui/XTTS-v2">coqui/XTTS-v2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/training">微调预训练模型</a>：未找到描述</li><li><a href="https://tenor.com/view/brain-dog-brian-dog-cooked-wallahi-im-finished-cooked-dog-gif-1849480349705279416">Brain Dog Brian Dog GIF - Brain dog Brian dog Cooked - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://en.wikipedia.org/wiki/Serial_Experiments_Lain">玲音 (Serial Experiments Lain) - 维基百科</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/not-lain/RAG-on-images">Image Retriever - not-lain 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://tenor.com/view/this-dog-detects-twitter-twitter-user-dog-twitter-gif-90166331583470465">This Dog Detects Twitter Twitter User GIF - This dog detects twitter Twitter user Dog - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/monkey-cool-gif-25963936">Monkey Cool GIF - Monkey Cool - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/doubt-gif-13250124">Doubt GIF - Doubt - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/lightning-struck-by-gif-14902359">Lightning Struck GIF - Lightning Struck By - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/dbz-abridged-vegeta-gif-14758870">Dbz Abridged GIF - Dbz Abridged Vegeta - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://lu.ma/fgwhcrsk">Techstars Startup Weekend Tokyo · Luma</a>：Techstars Startup Weekend Tokyo 是一场激动人心且沉浸式的创业世界探索。在充满活力的三天里，你将遇到最优秀的……</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/tree/main/examples">hiyouga/LLaMA-Factory 的 main 分支示例</a>：统一 100 多个 LLM 的高效微调。通过在 GitHub 上创建账户来为 hiyouga/LLaMA-Factory 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/not-lain/RAG-on-images/discussions/2">not-lain/image-retriever · 我真的不会用 git。可能需要更多测试</a>：未找到描述</li><li><a href="https://developers.googleblog.com/en/gemma-family-and-toolkit-expansion-io-2024/">介绍 PaliGemma、Gemma 2 以及升级后的 Responsible AI 工具包</a>：未找到描述</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/tree/main/">GitHub - hiyouga/LLaMA-Factory: 统一 100 多个 LLM 的高效微调</a>：统一 100 多个 LLM 的高效微调。通过在 GitHub 上创建账户来为 hiyouga/LLaMA-Factory 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/82">zero-gpu-explorers/README · ZeroGPU 时长配额问题</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/coai/plantuml_generation">coai/plantuml_generation · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1bpar6s/p40_still_worth_it/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://the-decoder.com/gpt-4-has-a-trillion-parameters/">报告称 GPT-4 拥有超过一万亿个参数</a>：据媒体报道，GPT-4 的规模据称是 GPT-3 的六倍，而 Elon Musk 退出 OpenAI 为 Microsoft 扫清了道路。</li><li><a href="https://huggingface.co/docs/hub/en/api#get-apidatasetsrepoidparquetconfigsplitnparquet">Hub API 端点</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1DO5FwwLNDimh6B7T5BX9vNdFPtu1f_Pq?usp=sharing>">Google Colab</a>：未找到描述</li><li><a href="https://hexdocs.pm/fss/0.1.1/FSS.html">FSS — fss v0.1.1</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/13v3b6q/multiple_cheap_gpus_or_a_single_expensive_one/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/NVIDIA/TensorRT-LLM">GitHub - NVIDIA/TensorRT-LLM: TensorRT-LLM 为用户提供...</a>

提供了一个易于使用的 Python API 来定义大语言模型 (LLMs) 并构建 TensorRT 引擎，这些引擎包含最先进的优化技术，以便在 NVIDIA GPUs 上高效执行推理。TensorRT-LLM 还包含用于创建执行这些 TensorRT 引擎的 Python 和 C++ 运行时的组件。</a>: TensorRT-LLM 为用户提供了一个易于使用的 Python API 来定义大语言模型 (LLMs) 并构建包含最先进优化技术的 TensorRT 引擎，以实现高效推理...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1256127810238677053)** (6 条消息): 

- **1 分钟学习 10 种机器学习算法**: 一位成员分享了一个 [YouTube 视频](https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5)，标题为 "10 Machine Learning Algorithms in 1 Minute"，简要概述了顶尖的机器学习算法。

- **对强化学习项目感兴趣**: 一位成员表达了在短时间内学习 Reinforcement Learning 的兴趣。他们提议启动一个小项目以更好地理解相关概念，并承认目前对它的工作原理只有模糊的认识。

- **咨询 Huggingface 课程更新**: 一位成员正在寻求有关 Huggingface 课程定期更新的信息，并将其与《Natural Language Processing with Transformers (revised edition May 2022)》一书进行对比。他们还询问了 Huggingface 网站上 Diffusion 和社区计算机视觉课程的实时性。

- **改进生物特征步态识别**: 一位成员分享了他们使用基础 2D 视频输入进行生物特征步态识别的进展，在 23 人识别任务中达到了 70% 的测试准确率。他们计划通过获取更多数据集、结合多帧用于 RNN 以及采用 triplet loss 生成 embeddings 来增强模型。

**提到的链接**: <a href="https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5">10 Machine Learning Algorithms in 1 Minute</a>: 大家好！我刚刚制作了一个简短的视频，在短短 1 分钟内涵盖了前 10 名机器学习算法！这是对每一个算法的简要介绍（再次）：Linear Regr...

  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1255975841272299620)** (5 条消息): 

- **关于 Diffusion Models 的启发性博客**：一位成员强烈推荐了 [Lilian Weng 的博客文章](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#citation)，该文章详细解释了 Diffusion Models，并包含了 GAN、VAE、Flow-based models 等各种生成建模技术的更新链接，以及 progressive distillation 和 consistency models 等最新进展。
- **Hermes-2-Pro-Llama-3-70B 发布**：升级后的 [Hermes 2 Pro - Llama-3 70B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B) 现在支持 function calling 功能和 JSON Mode。它在 function calling 评估中达到了 90% 的得分，在结构化 JSON Output 方面达到了 84%。
- **合成多表数据的挑战**：一篇文章讨论了[合成多表 tabular data](https://mltechniques.com/2024/06/15/synthesizing-multi-table-databases-model-evaluation-vendor-comparison/) 的复杂性，包括使用 SDV、Gretel 和 Mostly.ai 等库时遇到的失败和困难，特别是在处理包含日期的列时。
- **一分钟了解顶级 Machine Learning 算法**：一段名为 [“10 Machine Learning Algorithms in 1 Minute”](https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5) 的 YouTube 短视频，承诺快速涵盖核心 Machine Learning 算法。该视频对关键概念进行了快节奏的概述。
- **AI Engineer World's Fair 2024 亮点**：[AI Engineer World’s Fair 2024](https://www.youtube.com/watch?v=5zE2sMka620) YouTube 视频涵盖了主题演讲和 CodeGen Track，Vik 等知名人士出席了会议。该活动展示了 AI engineering 领域的重大进展和演示。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mltechniques.com/2024/06/15/synthesizing-multi-table-databases-model-evaluation-vendor-comparison/">Synthesizing Multi-Table Databases: Model Evaluation &amp; Vendor Comparison - Machine Learning Techniques</a>：与单表相比，合成多表 tabular data 面临着独特的挑战。当数据库包含交易日期或入院日期等日期列时（这在现实中经常发生）……</li><li><a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#citation">What are Diffusion Models?</a>：[2021-09-19 更新：强烈推荐 Yang Song（参考文献中几篇关键论文的作者）关于 score-based generative modeling 的这篇博客]。[2022-08-27 更新：添加了 classifier-free...</li><li><a href="https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5">10 Machine Learning Algorithms in 1 Minute</a>：大家好！我刚刚制作了一个短视频，仅用 1 分钟涵盖了前 10 名 Machine Learning 算法！这里是每个算法的简要介绍：Linear Regr...</li><li><a href="https://github.com/alidenewade/Publications/blob/main/Budget%20Speech%20Essay%20Final.pdf">Publications/Budget Speech Essay Final.pdf at main · alidenewade/Publications</a>：通过在 GitHub 上创建一个账户，为 alidenewade/Publications 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=5zE2sMka620">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen Track</a>：https://twitter.com/aidotengineer</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B">NousResearch/Hermes-2-Pro-Llama-3-70B · Hugging Face</a>：未找到描述。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1255973799854215239)** (8 messages🔥): 

- **Flight Radar 开启多语言实时追踪**：一位成员分享了一个使用 Flask 和 JavaScript 构建的多语言实时航班追踪 Web 应用程序。该程序利用 OpenSky Network API 让用户查看附近的航班、调整搜索半径，并将航班数据下载为 JPG 格式。更多详情请见 [GitHub](https://github.com/U-C4N/Flight-Radar)。

- **PixArt-900M Space 发布**：PixArt 的一个全新 900M 变体版本现已开放实验，包含处于开发中的不同 Batch Size 的 Checkpoint。这是 terminus 研究小组和 fal.ai 的**合作成果**，旨在创建出色的新模型。请在 [Hugging Face Spaces](https://huggingface.co/spaces/ptx0/PixArt-900M) 上查看。

- **使用 Pokémon 数据集的图像检索系统上线**：一个使用 Pokémon 数据集的完全开源图像检索系统已经发布。该成员承诺明天会发布一篇关于此系统的博客文章，但你现在就可以在 [Hugging Face Spaces](https://huggingface.co/spaces/not-lain/image-retriever) 上进行尝试。

- **一分钟了解十大机器学习算法**：分享了一个仅用一分钟涵盖十大机器学习算法的快速 YouTube 视频。[点击此处观看](https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5)。

- **AI 驱动的音乐叙事重新定义流派**：推出了一张融合了 AI 开发与音乐的创新专辑，为机器和人类提供独特的叙事体验。该专辑可在 [Bandcamp](https://vonpsyche.bandcamp.com/album/the-prompt) 和 [SoundCloud](https://soundcloud.com/vonpsyche/sets/the-prompt) 上获取，[YouTube](https://www.youtube.com/watch?v=RH9M7i8ft0E) 上也提供了宣传片。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/not-lain/image-retriever">Image Retriever - 由 not-lain 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5">1 分钟了解 10 个机器学习算法</a>：大家好！我刚刚制作了一个快速视频，在短短 1 分钟内涵盖了前 10 名的机器学习算法！这里是每个算法的简短介绍（再次）：线性回归...</li><li><a href="https://github.com/U-C4N/Flight-Radar">GitHub - U-C4N/Flight-Radar：一个使用 OpenSky Network API 的多语言实时航班追踪 Web 应用程序。</a>：使用 Flask 和 JavaScript 构建，允许用户查看附近的航班，调整搜索半径，并支持六种语言。功能包括地理定位，以及将航班数据下载为 JPG 的能力。</li><li><a href="https://huggingface.co/spaces/ptx0/PixArt-900M">PixArt 900M 1024px 基础模型 - 由 ptx0 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://vonpsyche.bandcamp.com/album/the-prompt">The Prompt，作者 Vonpsyche</a>：包含 12 首曲目的专辑</li><li><a href="https://soundcloud.com/vonpsyche/sets/the-prompt.">THE PROMPT</a>：标题：The Prompt，音乐：Vonpsyche，插图：Iron Goose。剧情：这张专辑带领听众穿越一个反乌托邦世界，在那里一位天才 AI 开发者努力创造一个为人类服务的...</li><li><a href="https://www.youtube.com/watch?v=RH9M7i8ft0E">The Prompt，作者 Vonpsyche</a>：Vonpsyche - The Prompt：沉浸在模糊现实与虚构界限的叙事中。作为对近期事件的离奇反映，'The Pro...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1256308604592590891)** (5 messages): 

- **新活动即将推出**：一位成员宣布：“我稍后会创建一个活动！”，这引起了小组的兴奋，并获得了表示认可和期待的反应。
- **关于 LLM 推理的研究论文**：一位成员分享了一篇有趣的[关于 LLM 推理的研究论文](https://arxiv.org/pdf/2405.16506)。另一位成员对其与 RADIT 相比的性能表现表示好奇，指出两者可能都需要 Finetuning，但对其中包含的 GNN 方法表示赞赏。

**提到的链接**：<a href="https://discord.gg/hugging-face-879548962464493619?event=1256365214895439974">加入 Hugging Face Discord 服务器！</a>：我们正致力于让优秀的机器学习民主化 🤗 验证以链接你的 Hub 和 Discord 账号！| 82343 位成员

  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1255993476395569162)** (13 messages🔥): 

- **寻求用于网页自动化任务的 YOLO**：一位成员询问如何使用 **YOLO**，通过参考图像和完整截图来识别并返回网页上相似元素的坐标。他们正在寻找一种高效的方法或现有的解决方案来满足其自动化需求。

- **探索高效的 SAM 部署**：一位用户就如何高效部署 **Segment Anything Model (SAM)** 寻求建议，并提到了 **MobileSAM** 和 **FastSAM** 等各种高效版本。他们正在寻找最佳实践，以及类似于语言模型中常用的连续批处理（continuous batching）和模型量化（model quantization）的等效技术。

- **Mask Former 微调挑战**：另一位成员报告了在微调用于图像分割的 **Mask Former** 模型时遇到的困难，并质疑该模型（特别是 **facebook/maskformer-swin-large-ads**）是否更倾向于语义分割（semantic segmentation）而非实例分割（instance segmentation）。

- **设计卷积神经网络**：一位用户对如何为特定项目确定合适的卷积层数量、padding、kernel sizes、strides 和 pooling layers 表示困惑。他们发现自己是在随机选择参数，并认为这并非理想的做法。
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1256228265798930482)** (10 messages🔥): 

- **Chatbot Arena 数据集排行榜需要 LLM 脚本**：一位成员将类似 chatbot arena 的数据集翻译成了另一种语言，并希望建立一个排行榜。他们正努力寻找一个脚本，能够使用 **LLM** 代替人工投票来填充 "winner" 字段。
- **需要对聊天机器人进行澄清**：当另一位成员提供帮助时，他们澄清自己指的是 chatbot arena 数据集。这在最初引起了一些混乱，请求被误解为需要一个聊天机器人。
- **Arena 评分中的人类偏好**：Vipitis 提到，arena 通常使用人类偏好来计算 Elo 评分，这表明需要更清晰的指导或替代方法。
- **紧急的 GEC 预测问题**：Shiv_7 表达了对语法错误纠正（GEC）项目的挫败感，其预测列表的 shape 不正确，并紧急请求建议以解决该问题。
  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1256262282804068362)** (1 messages): 

- **旧版本 Gradio 的分享链接即将失效**：从下周三开始，**Gradio** 3.13 及以下版本的分享链接将不再有效。请通过使用命令 `pip install --upgrade gradio` 升级您的 **Gradio** 安装，以保持项目顺利运行。
  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1255964280680091822)** (105 messages🔥🔥): 

```html
<ul>
    <li><strong>Gemma 2 支持现已上线：</strong> LM Studio 0.2.26 版本已添加 Gemma 2 支持。此更新包括 post-norm 和其他功能，但用户报告了一些集成 bug。<a href="https://github.com/ggerganov/llama.cpp/pull/8156">[GitHub PR]</a>。</li>
    <li><strong>更新与集成中持续存在的问题：</strong> 用户在 LM Studio 中遇到 Gemma 2 集成和自动更新困难。建议的修复方法是手动下载并重新安装配置，但 ROCm 等架构仍待支持。</li>
    <li><strong>本地托管模型的辩论：</strong> 本地托管的优势包括隐私、离线访问和个人实验机会。一些人对由于廉价云端解决方案的兴起而导致的其未来相关性表示怀疑。</li>
    <li><strong>LLama 3 模型争议：</strong> 关于 LLama 3 性能的看法不一，有人称其为令人失望的模型，而另一些人则认为它在创意任务中表现出色。性能问题似乎与特定版本有关，讨论集中在最近更新中的停止序列 bug。</li>
    <li><strong>对 Gemma 9B 性能的担忧：</strong> 一些用户报告 Gemma 9B 在 LM Studio 上的表现不如 Phi-3 等同类模型。持续的开发旨在解决这些问题，预计很快会有功能改进。</li>
</ul>
```
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/.">👾 LM Studio - 发现并运行本地 LLM</a>: 查找、下载并实验本地 LLM</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON 配置文件格式和示例配置文件的集合。 - lmstudio-ai/configs</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8197">为 Gemma 2 添加注意力与最终 logit soft-capping by abetlen · Pull Request #8197 · ggerganov/llama.cpp</a>: 此 PR 添加了缺失的注意力层和最终 logit soft-capping。实现参考自 Hugging Face Transformers。注意：注意力 soft-capping 与 Flash Attention 不兼容...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8156">添加对 Gemma2ForCausalLM 的支持 by pculliton · Pull Request #8156 · ggerganov/llama.cpp</a>: 添加了对 Gemma 2 系列模型的推理支持。包括对以下模型的支持：Gemma 2 27B、Gemma 2 9B。更新了 Gemma 架构以包含 post-norm 等功能。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1255978267828031488)** (222 messages🔥🔥): 

```html
- **Gemma-2 对上下文限制引发不满**：关于 **Gemma-2** 仅有 4k 上下文限制的公告引发了失望。一位成员将其描述为 *“就像造了一辆续航只有 80 英里的电动汽车”*，强调了用户对当前模型更高容量的期望。
- **关于 Gemma-2 上下文限制的困惑**：虽然最初的信息显示 **Gemma-2** 的上下文限制为 4k，但其他人将其纠正为 8k，显示出信息存在差异。一位成员指出 *“Gemini 对谷歌自家的产品都搞错了！”*。
- **寻求对故事叙述模型的支持**：推荐支持 [ZeusLabs/L3-Aethora-15B-V2](https://huggingface.co/ZeusLabs/L3-Aethora-15B-V2)，这是一个专为故事叙述和训练期间全上下文使用而设计的模型。建议在模型浏览器中搜索时附加 “GGUF”。
- **Deepseek Coder V2 Lite 和 Gemma 2 状态**：**Gemma 2 9b** 和 **Deepseek coder V2 Lite** 在 LM Studio 中显示为尚未支持，引发了关于何时添加它们的咨询。一位成员确认 **Gemma 2** 最初不被支持，但指出一个 [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/8156) 已经被合并以添加支持。
- **关于 7b~9b 类别中最佳模型的讨论**：辩论了 **Qwen 2 7b**、**Deepseek Coder V2 Lite** 和 **Llama 3** 等各种模型的有效性。一位成员在性能测试后得出结论 *“Deepseek 值得一试”*，但也指出 **Qwen 2 7b** 在未启用 Flash Attention 的情况下存在问题。
```
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/>">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://huggingface.co/ZeusLabs/L3-Aethora-15B-V2">ZeusLabs/L3-Aethora-15B-V2 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/AIatMeta/status/1806361623831171318">来自 AI at Meta (@AIatMeta) 的推文</a>：今天我们宣布推出 Meta LLM Compiler，这是一个基于 Meta Code Llama 构建的模型系列，具有额外的代码优化和编译器功能。这些模型可以模拟编译器，预测最优...</li><li><a href="https://huggingface.co/microsoft/Florence-2-large">microsoft/Florence-2-large · Hugging Face</a>：未找到描述</li><li><a href="https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9">GGUF 量化概览</a>：GGUF 量化概览。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8156">由 pculliton 提交的为 Gemma2ForCausalLM 添加支持 · Pull Request #8156 · ggerganov/llama.cpp</a>：为 Gemma 2 系列模型添加推理支持。包括对以下内容的支持：Gemma 2 27B、Gemma 2 9B。更新了 Gemma 架构以包含 post-norm 等功能。我已阅读贡献...</li><li><a href="https://tenor.com/view/cartoons-tom-and-jerry-ok-mouse-ok-i-got-it-gif-17005831">猫和老鼠 GIF - Cartoons Tom And Jerry Ok - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1256323247704641609)** (1 messages): 

- **LM Studio 0.2.26 发布，支持 Gemma 2**：新的 LM Studio 0.2.26 现在支持谷歌的 Gemma 2 模型，特别是 **9B** 和 **27B** 版本。请在 [lmstudio-community 页面](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF)查看。
- **Windows on ARM64 首次亮相**：得益于与 Qualcomm 的合作，LM Studio 现在可用于 Windows on ARM (Snapdragon X Elite 电脑)。从 [lmstudio.ai](https://lmstudio.ai/snapdragon) 下载 ARM64 版本。
- **报名参加 LM Studio 0.3.0 私测**：LM Studio 的一次重大更新已接近完成，邀请测试人员通过[在此报名](https://forms.gle/K7pTWgTJsdHBmUaWA)来提供帮助。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://lmstudio.ai/snapdragon">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://forms.gle/K7pTWgTJsdHBmUaWA">LM Studio 0.3.0 - 私测报名</a>：感谢您有兴趣帮助测试我们即将发布的版本。LM Studio 0.3.0 充满了新功能，我们希望在向全世界发布之前，得到您的帮助来排除错误...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1255991634538008669)** (13 条消息🔥): 

- **Llama.cpp 报错：不支持的模型架构**：成员们遇到了错误提示 *'error loading model architecture: unknown model architecture: gemma2'*。一位成员指出，该错误是由于 Llama.cpp 尚不支持该架构导致的。
- **Snapdragon X Elite 性能获赞**：一位成员感谢 LM Studio 团队快速支持 Snapdragon X Elite 系统，并指出这些设备具有高性能、低噪音和出色的电池续航。在基准测试中，Snapdragon X Elite 在 CPU/内存任务上的表现优于 i7 12700K，但不及 4090 GPU。
- **LM Studio 中不支持的模型**：成员们讨论了尝试运行 "gemma 2 9b" 模型的情况，并意识到 LM Studio 尚未支持该模型。建议他们使用较旧的模型，或探索 Transformer、MLX 以及量化的 gguf 文件等替代方案。
- **Ubuntu 上的 IPv6 和语法错误**：一位用户通过在 Ubuntu 22.04 上禁用 IPv6 解决了模型加载问题，但在启动时仍遇到 "config-presset file syntax error"（配置文件预设语法错误），目前尚不确定其影响。
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/)** (1 条消息): 

cos2722: 你好。有人能帮我让 GORILL open funcion v2 运行起来吗？我没有任何配置。
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1256058859596812298)** (28 条消息🔥): 

- **LM Studio 支持双显卡**：一位成员询问：*"使用 2 张显卡有价值吗？LM Studio 会同时利用它们吗？"* 另一位成员确认道：**“是的，而且是的。”**

- **4GB 内存运行小型代码生成模型**：讨论了在 4GB 内存上运行代码生成 LLM 的可行性。一个建议是使用 **Qwen 2 0.5B**，但其代码准确性被描述为 *"充其量也就一般"*；而为了获得更好的性能，推荐使用 **Claude 3.5 Sonnet**。

- **基于低端模型的 Multi-Agent 框架**：一位成员计划在 Multi-Agent 框架中使用 **0.5B 模型** 作为用户代理（user proxy），认为它能够轻松胜任该角色。另一位成员对这种低端模型在编码任务中的有效性表示怀疑。

- **Lamini Memory Tuning 可增强 LLM 准确率**：强调了 [Lamini Memory Tuning](https://www.lamini.ai/blog/lamini-memory-tuning) 的潜力。该方法能显著 **“提高事实准确性并减少幻觉”**，并可能使 0.5B 模型在低端机器上更加有效。

- **Intel GPU 性能评价褒贬不一**：有关于 Intel GPU 效能的问题。一位成员指出 **“CPU 更快”**，而另一位补充说 **“Intel GPU 支持正在开发中，但在目前支持的后端上性能低于 CPU。”**

**提到的链接**：<a href="https://www.lamini.ai/blog/lamini-memory-tuning">Introducing Lamini Memory Tuning: 95% LLM Accuracy, 10x Fewer Hallucinations | Lamini - Enterprise LLM Platform</a>：未找到描述。

  

---

### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1256075870263578705)** (33 messages🔥): 

- **“Gemma 2 烂透了”引发质疑**：LM Studio 0.2.26 发布并支持 Gemma 2 后反应不一。一位用户批评道：“Gemma 烂透了……我严重怀疑他们是否做了改进。”另一位用户指出在追问时存在问题，引发了技术排障讨论。
- **“Unexpected end of JSON input”错误的解决方案**：遇到 JSON 输入错误的用户收到建议，重命名有问题的文件并重启应用。他们还被引导至特定的 Discord 频道以获取进一步帮助。
- **更新 llama.cpp commit**：一位用户建议更新到最新的 llama.cpp commit 以获得更好的性能。然而，官方澄清用户需要等待包含该更新的正式版本发布。
- **Gemma 2 加载问题及解决方案**：用户讨论了 Gemma 2 的问题和解决方法，包括重新加载模型。一位用户强调了更新后的模型设置，以及为了获得最佳性能需要使用 LM Studio 0.2.26。
- **Markdown 中的反引号格式化问题**：有报告称生成文本中的代码块格式化存在问题，反引号放置不当，影响了 Markdown 渲染。该问题似乎是瞬态的，且仅针对特定的代码生成。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLM</li><li><a href="https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF">lmstudio-community/gemma-2-9b-it-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/futurama-angry-gif-13063135">Futurama Angry GIF - Futurama Angry - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1256340898820653086)** (4 messages): 

- **Gemma 2 模型支持受到质疑**：一位成员询问 ROCm 预览版是否有支持 Gemma 2 模型的更新，并指出普通的 LM Studio 0.2.6 版本无法像 ROCm 预览版那样检测到 AMD GPU。
- **适用于 Windows 的 ROCm “扩展包”发布**：一位成员宣布了适用于 **Windows** 的 0.2.26 ROCm “扩展包”已可用，由于目前处于开发过渡阶段，提供了高级安装说明。详情请参阅 [GitHub 上的扩展包说明](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#installation-on-windows-0226-)。

**提及的链接**：<a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#installation-on-windows-0226-">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs

  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1256291010150137896)** (1 messages): 

- **Gemma 2 震撼发布**：Gemma 2 现已可在 Windows 和 Mac 上通过 0.2.26 版本下载；Linux 版本即将推出。9B 模型表现出色，而 27B 模型因一些古怪行为正接受审查，并[征求反馈](https://lmstudio.ai/)。
- **轻松获取 Gemma 2 模型**：新模型可以从 Hugging Face 上的 [LM Studio Community](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF) 下载：9B 模型和可能存在古怪行为的 27B 模型已发布，准备好接受测试。

**提及的链接**：<a href="https://lmstudio.ai/">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLM

  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/)** (1 messages): 

mystic9t: 在单个无代码（no-code）环境中集成它们出乎意料地困难。
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1255981680599240866)** (330 条消息🔥🔥): 

- **测试 AI 边界可能会导致封号**：成员们对测试 AI 绕过方案的极限表示担忧，并提醒违反 OpenAI 的使用政策可能导致账号被暂停或终止。有人分享了[使用政策](https://openai.com/policies/usage-policies)的链接，强调了遵守安全防护措施的重要性。
- **开源与闭源 AI 之争升温**：成员们辩论了开源先进 AI 模型的优劣，权衡了潜在滥用的风险与广泛获取带来的收益。一位用户认为，将 AI 限制在富人手中导致的经济失衡可能是有害的，而另一位用户则强调了为了公共安全进行监控的必要性。
- **探索 RLHF 训练经验**：用户对来自人类反馈的强化学习（RLHF）表现出困惑和好奇，讨论了其在 OpenAI 模型中的应用。一些人提到极少见到 RLHF 提示词，而另一些人则思考 OpenAI 如何管理公开的 RLHF 训练。
- **大规模监控引发激烈讨论**：关于 OpenAI 等公司参与监控的深度对话展开，引用了一篇关于瓦解 AI 欺骗性用途的[博客文章](https://openai.com/index/disrupting-deceptive-uses-of-AI-by-covert-influence-operations)。用户辩论了此类监控的伦理和必要性，在隐私与安全的权衡上意见不一。
- **聊天机器人与 API 集成正在开发中**：成员们分享了将 AI 与其他工具和服务集成的经验和项目。一位用户详细介绍了他们在 Discord 中集成 SearxNG 以增强搜索功能的工作，而另一位用户则重点介绍了他们为了实现更好功能而正在实验的各种 AI 模型和 API。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/J0KHiiTtt4w?si=_0wHUumSnvpWTNcw">为什么 Elon Musk 说我们生活在模拟中</a>：你可能喜欢玩《模拟人生》，但 Elon Musk 说你就是那个模拟人。加入 Vox Video Lab 帮助我们制作更多有野心的视频，获取专属福利...</li><li><a href="https://github.com/PierrunoYT/claude-3-artifacts">GitHub - PierrunoYT/claude-3-artifacts</a>：通过在 GitHub 上创建账号为 PierrunoYT/claude-3-artifacts 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1255979888016621568)** (14 条消息🔥): 

- **插件已弃用，GPTs 接管**：一位用户询问如何在单个对话中使用多个插件功能，例如视频摘要工具和图表制作工具。另一位成员澄清说 *"插件现在已弃用，并已被 GPTs 取代，"* 但建议使用 `@mention` 功能，以便在对话中灵活调用多个 GPTs。

- **工作组 API 访问问题**：一位用户询问如何获取供工作组使用的 API。

- **使用自定义 GPT 生成医学问题遇到困难**：一名医学生分享了使用自定义 GPT 创建高难度练习题时遇到的问题。尽管上传了详细的指南和讲义信息，GPT 生成的问题依然水平低下，且来源引用不当。

- **GPTs 丢失，恢复步骤**：有用户报告无法访问其自定义 GPTs 并寻求帮助。另一位成员分享了解决方案，提供了一个 URL [chatgpt.com/gpts/mine](https://chatgpt.com/gpts/mine) 进行重定向，帮助用户在左侧栏恢复对 GPTs 的访问。
  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1255968238370553897)** (25 messages🔥): 

- **Unicode semiotics 谜题**：一位成员讨论了将 **Unicode semiotics** 用于特定任务的情况，指出它们“消耗更多 Token，而非更少”，但占用的字符数更少。尽管发现它对 In-context learning 很有用，但他们无法找到任何相关的解释性论文。
- **API 在 Unshuffle games 中表现挣扎**：另一位成员分享了 API 在解决 Unshuffle games（如将 "edtPto lumAli" 还原为 "Potted Allium"）时遇到的困难。有一个[分享的方法](https://chatgpt.com/share/5c4d7258-cc5c-47f0-8a67-843d7b96c1d8)建议在 API 之外配合使用 Python 以改善结果。
- **Prompt engineering 建议**：一位用户询问了关于从编程转向 PM/业务分析任务的 Prompt engineering 建议。得到的建议是使用简单、清晰、简洁的自然语言 Prompt，并简要提到了 **Logit bias** 概念以实现更深层的 Prompt 控制。
- **准确定性（Quasi-determinism）的困惑**：讨论了 **Stochastic neural networks** 的“准确定性”本质，以澄清这些模型的行为方式。这一解释引起了不同的反应，暗示了理解其复杂性所需的深度。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1255968238370553897)** (25 messages🔥): 

- **Unicode Semiotics 消耗更多 Token**：成员们讨论了使用 Unicode semiotics 来节省 Token 成本或降低延迟。会议澄清了 Unicode semiotics 虽然占用字符较少，但消耗的 Token 更多，目前还没有论文对此进行解释。

- **API 在 Unshuffle games 中的困境**：一位成员分享了让 API 解决 Unshuffle games（如将 "edtPto lumAli" 还原为 "Potted Allium"）的困难。另一位建议使用 Python 生成单词的所有可能重组，然后让 API 挑选正确的，尽管可能会出现类似 "dotted" 的 Hallucinations。

- **关于 Prompt 工程的建议**：针对 Prompt engineering 和 Negative weighting 的新手咨询，建议使用简单平实的语言。**Logit bias** 被提及为一种潜在的高级技术。

- **关于神经网络确定性本质的讨论**：简短的交流澄清了神经网络中的反向函数往往是 Stochastic 和 Quasi-deterministic 的，而非完全确定性的。

- **未解决的符号学论文查询**：一位用户询问是否有关于 Unicode semiotics 的论文，但确认目前不存在此类文档。
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1255971125079576728)** (297 条消息🔥🔥): 

<ul>
    <li><strong>创建自定义风格的数据集引发关注与担忧：</strong> 一位用户分享了他们正在构建一个包含其自定义风格生成图像的大型数据集。讨论涉及了生成 NSFW 内容及被封禁的担忧，突显了监控数据集和图像生成的细微差别。</li>
    <li><strong>Automatic1111 的问题促使用户探索替代方案：</strong> 用户报告了对 Automatic1111 崩溃或损坏的挫败感，导致一些人转向 Forge 等替代方案，尽管它存在内存管理问题。分享了一个 YouTube 指南（[Stable Diffusion Webui Forge Easy Installation](https://www.youtube.com/watch?v=FKzvHFtc8N0&t=64s)）以提供安装帮助。</li>
    <li><strong>Cascade 频道争论仍在继续：</strong> 许多用户表达了希望取消 Cascade 频道归档的愿望，理由是其中包含宝贵的讨论和知识。挫败感显而易见，一些用户怀疑这是为了推动用户参与 SD3 的更广泛举措。</li>
    <li><strong>讨论了模型训练的细节和工具：</strong> 用户讨论了 LoRa 训练的具体细节、3m sde exponential 等采样器以及 VRAM 限制，分享了技巧和经验。强调了不同节点以及 ComfyUI、Forge 和 Stable Swarm 等 UI 工具的使用和局限性。</li>
    <li><strong>Discord 管理和社区不满：</strong> 许多用户对频道删除和归档表示不满，怀疑这代表了重心正从社区驱动的方面转移。呼吁加强沟通并保留社区创建的资源。</li>
    <li><strong>YouTube 幽默活跃了气氛：</strong> 讨论中出现了幽默时刻，包括俏皮地分享了 YouTube 视频 [Was Not Was - Walk The Dinosaur](https://youtu.be/vgiDcJi534Y)，以及关于彩色个人资料图片和怀旧表情符号（如 <:kek:692062611659030548>）的笑话。</li>
</ul>
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://jellybox.com">来自 Jellybox 的推文</a>: 在本地运行 AI 模型。私密且完全离线！Jellybox 以简单易用的软件包为每个人解锁了本地 AI 工具的力量。从聊天到 Agent，再到图像生成，Jelly...</li><li><a href="https://www.youtube.com/watch?v=FKzvHFtc8N0&t=64s">Stable Diffusion Webui Forge 简易安装</a>: Stable Diffusion Webui Forge 简易安装。无需下载安装 Python 或其他任何东西，因为它已全部包含在安装程序中，只需下载...</li><li><a href="https://youtu.be/vgiDcJi534Y">Was Not Was - Walk The Dinosaur</a>: 恐龙被遛了，从而开启了它们种族的终结。</li><li><a href="https://github.com/lkwq007/stablediffusion-infinity">GitHub - lkwq007/stablediffusion-infinity: 在无限画布上使用 Stable Diffusion 进行 Outpainting</a>: 在无限画布上使用 Stable Diffusion 进行 Outpainting - lkwq007/stablediffusion-infinity
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1255966220784767016)** (50 条消息🔥):

```html
- **Scarlet AI 预览版发布**：一位成员介绍了 **Scarlet AI** 的预览版，旨在规划复杂项目和委派任务。可以在 [https://app.scarletai.co/](https://app.scarletai.co/) 进行测试，但目前尚未达到生产级标准。
- **Character.AI 语音功能**：**Character.AI** 推出了 **Character Calls**，允许用户通过电话与 AI 角色互动，适用于练习面试和 RPG 等多种场景。可以在其移动应用上尝试：[https://share.character.ai/Wv9R/6tdujbbr](https://share.character.ai/Wv9R/6tdujbbr)。
- **Meta 用于代码优化的 LLM 编译器**：Meta 推出了 **Large Language Model Compiler**，专为编译器优化任务设计，增强了对中间表示（intermediate representations）和优化技术的理解。更多详情见其 [研究出版物](https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/)。
- **用于可靠 Agent 的 LangGraph Cloud**：**LangChainAI** 推出了 **LangGraph Cloud**，用于构建具有容错能力、可扩展的 Agent 工作流，并集成了追踪和监控功能。加入等待名单并阅读其 [博客文章](http://bit.ly/langgraph-cloud-blog-1) 了解更多。
- **Adept 战略调整及联合创始人加入 Amazon**：**Adept** 宣布了战略更新和领导层变动，多位联合创始人加入了 Amazon 的 AGI 团队。从 [GeekWire 文章](https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/) 中获取更多细节。
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/">Amazon hires founders from well-funded enterprise AI startup Adept to boost tech giant&#8217;s &#8216;AGI&#8217; team</a>: (GeekWire 现场照片 / Kevin Lisota) Amazon 正通过聘请来自 Adept 的高管来加强其 AI 布局，Adept 是一家总部位于旧金山、致力于构建 “Agent” 的初创公司。</li><li><a href="https://x.com/AdeptAILabs/status/1806773469155381705?t=HevOdjCZ31VyPecgHs5KoQ&s=19">来自 Adept (@AdeptAILabs) 的推文</a>: 今天，我们宣布了一些战略更新以及领导层和团队的变动。更多详情请见我们的博客：https://www.adept.ai/blog/adept-update</li><li><a href="https://x.com/noamshazeer/status/1806375332863418564?s=46&t=90xQ8sGy63D2Otiao">来自 Noam Shazeer (@NoamShazeer) 的推文</a>: 为团队正式发布 Character Calls 感到无比自豪！引用 Character.AI (@character_ai) —— AI 聊天变得真实了。介绍 Character Calls，这是我们套件中的最新成员...</li><li><a href="https://x.com/noamshazeer/status/1806375332863418564?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Noam Shazeer (@NoamShazeer) 的推文</a>: 为团队正式发布 Character Calls 感到无比自豪！引用 Character.AI (@character_ai) —— AI 聊天变得真实了。介绍 Character Calls，这是我们套件中的最新成员...</li><li><a href="https://x.com/tiagoefreitas/status/1806428334349504905?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Tiago Freitas (@tiagoefreitas) 的推文</a>: 刚刚发布了新版 http://scarletai.co 的预览版。每个人都是经理！Scarlet 为个人和创始人赋予了 Agency。最终将助力首批独角兽超级个体（solopreneurs）！我们赋能...</li><li><a href="https://x.com/llama_index/status/1806116419995844947?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q">来自 LlamaIndex 🦙 (@llama_index) 的推文</a>: ✨ 刚刚在 @aiDotEngineer 世界博览会的舞台上宣布！✨ 一个用于将多 Agent AI 系统投入生产的全新框架！目前为 Alpha 版本，llama-agents 提供：⭐️ 分布式...</li><li><a href="https://x.com/DavidKPiano/status/1806417216914817514?t=99I0TJJfrKHHDQYeiizv8A&s=19">来自 David K 🎹 (@DavidKPiano) 的推文</a>: 我很喜欢 AI 初创公司如何逐渐（重新）发现用于 Agent 行为和系统的状态机（state machines）和 Actor 模型。仍然不确定为什么需要专门的基础设施；这一切都只是...</li><li><a href="https://x.com/LangChainAI/status/1806371717084025165?t=15TNW0RaIb6EoIJ">来自 LangChain (@LangChainAI) 的推文</a>: 🚀 介绍 LangGraph Cloud 🚀 LangGraph 帮助你构建真正起作用的可靠 Agent。今天，我们发布了 LangGraph Cloud，这是我们用于运行容错 LangGraph Agent 的新基础设施...</li><li><a href="https://x.com/LangChainAI/status/1806371717084025165?t=15TNW0RaIb6EoIJKPq_IjA&s=19">来自 LangChain (@LangChainAI) 的推文</a>: 🚀 介绍 LangGraph Cloud 🚀 LangGraph 帮助你构建真正起作用的可靠 Agent。今天，我们发布了 LangGraph Cloud，这是我们用于运行容错 LangGraph Agent 的新基础设施...</li><li><a href="https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/">Meta Large Language Model Compiler</a>: 编译器优化的基础模型
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1256021421574852659)** (2 条消息): 

- **OpenAI 演示公告**: 一条消息紧急提醒大家关注 OpenAI 的演示，并引导至特定的 [OpenAI Demo 频道](https://discord.com/channels/822583790773862470/1197350122112168006)。该消息虽然缺乏额外背景，但表明这是一个即时事件。

- **OSS GPT Store 综述提醒**: 成员们收到了关于一小时后进行的 OSS GPT Store 综述的提醒。提醒中包含加入特定频道的提示，并建议领取相关角色以接收未来的通知。
  

---

### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1255962568636502156)** (150 条消息🔥🔥): 

- **“GPT-4o 将主导桌面端”**：讨论显示了在桌面端使用 GPT-4o 进行代码辅助的兴奋感，认为它能“帮助你编写代码等”。成员们表示有兴趣为此尝试 Open-Interpreter 及其与本地模型的集成。
- **Linux 与 Mac 的直播问题对比**：成员们在尝试使用 Linux 进行直播时遇到了技术困难，提到了权限和屏幕共享的问题。有人开玩笑说需要一台拥有“如此财富”的 Mac，强调了这种挣扎（“也许不值得为了覆盖这些内容而费劲。桌面应用确实挺酷的”）。
- **直播困扰与修复**：小组遇到了直播问题，主要是视频和音频质量差。通过切换到有线连接以提高稳定性，问题得到了一定程度的缓解。
- **Cursor 高级用户与生产力技巧**：一位成员询问“优质的 Cursor 高级用户内容”以提高生产力。另一位推荐了 YouTube 上的 *“indydevdan”*，介绍了有用的工作流和各种配置工具，以提高使用 Vim 时的编码效率。
- **Vesktop 作为解决方案**：为了解决 Discord 在 Linux 上的性能问题，成员们建议使用 [Vesktop](https://github.com/Vencord/Vesktop)，这是一个旨在增强 Linux 用户性能和支持的自定义 Discord 应用。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=vaIiNZoXymg">AI Engineer World’s Fair 2024 - Keynotes &amp; Multimodality track</a>: https://twitter.com/aidotengineer</li><li><a href="https://www.youtube.com/live/vaIiNZoXymg">AI Engineer World’s Fair 2024 - Keynotes &amp; Multimodality track</a>: https://twitter.com/aidotengineer</li><li><a href="https://github.com/Vencord/Vesktop">GitHub - Vencord/Vesktop: Vesktop is a custom Discord App aiming to give you better performance and improve linux support</a>: Vesktop 是一个自定义 Discord 应用，旨在提供更好的性能并改进 Linux 支持 - Vencord/Vesktop</li><li><a href="https://github.com/dimfeld/dotfiles/blob/master/nvim/.config/nvim/lua/commands/llm.lua">dotfiles/nvim/.config/nvim/lua/commands/llm.lua at master · dimfeld/dotfiles</a>: 通过在 GitHub 上创建账号为 dimfeld/dotfiles 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1256338676573667530)** (34 条消息🔥): 

- **公开 GPTs 提示词泄露的可能性**：一位成员强调，虽然特定的 GPT 定义可能不容易获取，但它们并非真正私密，已被他人在 [GitHub](https://github.com/LouisShark/chatgpt_system_prompt/tree/main/prompts/gpts) 上提取。另一位成员补充道：“最好假设有人能拿到这些，所以不要在里面放秘密。”
- **分享富有见地的研究论文**：一位成员指出了一些研究论文的价值，分享了讨论语言智能体认知架构 (CoALA) 的 [arxiv.org/abs/2309.02427](https://arxiv.org/abs/2309.02427) 链接，以及另一个相关的 [GitHub 仓库](https://github.com/ysymyth/awesome-language-agents) 链接。这些论文提供了一个框架来组织现有的语言 Agent 并规划未来的发展。
- **对精彩演讲和演示的赞扬**：许多成员对准备充分的演示表示赞赏，评论如“精彩的演讲”和“谢谢！”。演讲者因其充分的准备和贡献受到了特别表扬。
- **AI 工程师会议回顾建议**：对于未来的会议，一位成员建议对 AI 工程师会议进行回顾，可能包含一系列闪电演讲。这个想法得到了聊天中其他人的积极反馈。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/ysymyth/awesome-language-agents">GitHub - ysymyth/awesome-language-agents: List of language agents based on paper &quot;Cognitive Architectures for Language Agents&quot;</a>: 基于论文 "Cognitive Architectures for Language Agents" 的语言 Agent 列表 - ysymyth/awesome-language-agents</li><li><a href="https://github.com/LouisShark/chatgpt_system_prompt/tree/main/prompts/gpts">chatgpt_system_prompt/prompts/gpts at main · LouisShark/chatgpt_system_prompt</a>: GPT 系统提示词集合以及各种提示词注入/泄露知识。 - LouisShark/chatgpt_system_prompt</li><li><a href="https://arxiv.org/abs/2309.02427">Cognitive Architectures for Language Agents</a>: 最近的研究通过外部资源（如互联网）或内部控制流（如提示词链）增强了大语言模型 (LLMs)，用于需要落地或推理的任务...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1256114993519263845)** (2 条消息): 

- **Instruction Pre-Training 提升 LM 性能**：一篇新论文提出了 **Instruction Pre-Training**，通过高效的指令合成器生成的 **2 亿条指令-响应对 (instruction-response pairs)** 来增强大型语料库。该方法不仅增强了预训练基座模型，还使 **Llama3-8B** 在持续预训练 (continual pre-training) 中能与 **Llama3-70B** 竞争。[阅读完整论文](https://arxiv.org/abs/2406.14491) 或在 [Hugging Face 上查看模型](https://huggingface.co/instruction-pretrain/finance-Llama3-8B)。
- **MCT Self-Refine 提升数学推理能力**：**MCT Self-Refine (MCTSr) 算法**将 Large Language Models (LLMs) 与 **Monte Carlo Tree Search (MCTS)** 相结合，以提升在复杂数学任务中的表现。大量实验表明，MCTSr 利用系统的探索和启发式自我优化 (heuristic self-refine) 过程，显著提高了解决奥数级别 (Olympiad-level) 问题的成功率。[阅读详细研究](https://arxiv.org/abs/2406.07394)。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.07394">Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B</a>：本文介绍了 MCT Self-Refine (MCTSr) 算法，这是一种将 Large Language Models (LLMs) 与 Monte Carlo Tree Search (MCTS) 创新结合的方法，旨在增强在复杂数学任务中的表现...</li><li><a href="https://arxiv.org/abs/2406.14491">Instruction Pre-Training: Language Models are Supervised Multitask Learners</a>：无监督多任务预训练一直是近期语言模型 (LMs) 成功的关键方法。然而，有监督多任务学习仍具有巨大潜力，随着规模的扩大...</li><li><a href="https://huggingface.co/instruction-pretrain">instruction-pretrain (instruction-pretrain)</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1256081294694023279)** (10 条消息🔥): 

```html
- **公共频道现已开放**：一则帖子宣布频道 <#1105324249721356298> 和 <#1104063238934626386> 已设为公开。

- **REVEAL 基准测试验证器**：一个新数据集 [REVEAL: Reasoning Verification Evaluation](https://reveal-dataset.github.io)，用于在开放域问答设置下基准测试复杂 Chain-of-Thought 推理的自动验证器，突显了它们在验证逻辑正确性方面的困难。该数据集在 [arXiv 论文](https://arxiv.org/abs/2402.00559) 中有详细介绍，包含全面的标签和自由文本理由。

- **XTREME 评估多语言模型**：[XTREME 数据集](https://huggingface.co/datasets/google/xtreme) 评估预训练多语言模型的跨语言泛化能力，涵盖了 40 种语言类型学上多样化的语言。它包含九个需要不同层级语法和语义推理的任务。

- **SPIQA 挑战多模态模型**：[SPIQA 数据集](https://huggingface.co/datasets/google/spiqa) 专为科学论文的多模态问答而设计，包含超过 27 万个专注于图表、表格和文本段落的问题。该数据集旨在评估大型多模态模型理解复杂图表和表格的能力。

- **TACT 测试数值推理**：[TACT](https://huggingface.co/datasets/google/TACT) 被引入用于通过表格评估 LLM 使用复杂指令的推理和计算能力。该数据集显示当代 LLM 表现不佳，总体准确率低于 38%。

- **UNcommonsense 解释怪异情境**：[UNcommonsense](https://huggingface.co/datasets/allenai/UNcommonsense) 专注于解释不寻常和意想不到的情况，其英语语料库包含 2 万个独特语境和 4.1 万个溯因解释，为不寻常的结果提供见解。

- **EmotionalIntelligence-50K 专注于情感**：[EmotionalIntelligence-50K 数据集](https://huggingface.co/datasets/OEvortex/EmotionalIntelligence-50K) 旨在构建和训练能够理解并生成具有情感智能回复的模型，包含 51,751 行关于各种提示词和回复的文本数据。

- **BrightData/IMDb-Media 提供全面的电影数据**：[BrightData/IMDb-Media 数据集](https://huggingface.co/datasets/BrightData/IMDb-Media) 包含超过 24.9 万条记录，涵盖故事片、电视剧等 32 个数据字段，并定期更新评分、评论、演职人员和预算等详尽细节。

- **Opus-WritingPrompts 包含敏感内容**：[Opus-WritingPrompts 数据集](https://huggingface.co/datasets/Gryphe/Opus-WritingPrompts) 包含 3008 篇使用 Reddit 的 Writing Prompts 生成的短篇小说。该数据集包含多样化的内容，包括情色内容，并附有敏感信息免责声明。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://reveal-dataset.github.io">REVEAL</a>: A Chain-of-Thought Is as Strong as Its Weakest Link: A Benchmark for Verifiers of Reasoning Chains (思维链的强度取决于其最薄弱的一环：推理链验证器基准测试)</li><li><a href="https://huggingface.co/datasets/google/TACT">google/TACT · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/google/spiqa">google/spiqa · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/Gryphe/Opus-WritingPrompts">Gryphe/Opus-WritingPrompts · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/allenai/UNcommonsense">allenai/UNcommonsense · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/OEvortex/EmotionalIntelligence-50K">OEvortex/EmotionalIntelligence-50K · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/BrightData/IMDb-Media">BrightData/IMDb-Media · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/google/xtreme">google/xtreme · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/)** (1 条消息): 

deoxykev: 个人而言，我会直接采用经验方法。涉及的变量太多了。
  

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1256328863865241630)** (1 条消息): 

- **讨论长寿研究**：一位成员分享了对潜在反乌托邦社会的担忧，即 *“年长的富人通过牺牲年轻人的寿命来获得永生”*，同时也对旨在提高老年人健康的研究表示赞赏。他们建议应以安全的方式推进此类进展。
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1255992857387470991)** (9 条消息🔥): 

- **RankGPT 昂贵且令人困惑**：一位成员评论说 “RankGPT 很贵”，另一位用户询问 “通过 embedding 进行重排序” 是什么意思，以及为什么它会有 tokens。他们后来弄明白了，但最初的困惑凸显了该工具的复杂性。
- **RAG 数据集应当公开**：在讨论重排序过程时，一位成员指出 RAG 数据集必须是一个公开项目，并建议社区访问可以提高理解和利用率。
- **“Smooth brain” 更倾向于 Hermes 0 shot**：一位用户提到他们更喜欢一篇论文中的方法，该方法显示 Hermes 0 shot 使用 “好或坏” 作为评价是最有效的，尽管承认还有改进空间。他们幽默地坦白说，为了保持 “大脑平滑（不想动脑）”，他们想要避免复杂的解题过程。
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1255974809028792350)** (1 条消息): 

- **Hermes 2 Pro 70B 发布**：Nous Research 发布了 **Hermes 2 Pro 70B**，这是一个纯粹的 Hermes 模型，没有与 Llama-3 Instruct 进行合并。此更新解决了 Function Calling 问题和拒绝回答的情况，但牺牲了一点性能。请在 [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B) 上查看。

- **针对 Function Calling 和 JSON 输出进行了增强**：Hermes 2 Pro 在 *Function Calling* 和 *JSON Structured Outputs* 方面表现出色，在评估中分别获得了 90% 和 84% 的分数。该模型基于更新后的 OpenHermes 2.5 Dataset，并包含一个 **Function Calling and JSON Mode dataset**。

- **多个指标的改进**：新的 Hermes 2 Pro 保持了**出色的通用任务和对话能力**。它在多个领域显示出改进，包括结构化 JSON 输出和 Function Calling，这是*与 Fireworks.AI 合作开发*的。

**提到的链接**：<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B">NousResearch/Hermes-2-Pro-Llama-3-70B · Hugging Face</a>：未找到描述

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1255969935465775228)** (111 条消息🔥🔥): 

- **智能且具备上下文感知能力的 8B 模型令用户惊喜**：成员们讨论了一个 8B 模型令人印象深刻的性能和上下文感知能力，并指出它能够理解对话中的细微差别。一位用户分享道：*“我问了它一个含糊的问题，暗示我们要做什么，它的回答是正确的！”*。
  
- **关于 Hermes 模型的困惑**：关于是否应该优先选择非 Theta 版的 Hermes Pro 模型而非 Theta 版本，存在短暂的困惑。一位成员澄清说，非 Theta 版本可能更适合 function calling，或者在遇到 tokenization 问题时表现更好。

- **对 OpenHermes 2.5 数据集清洗方法的关注**：成员们询问了 OpenHermes 2.5 数据集的清洗过程。遗憾的是，没有分享详细信息。

- **讨论了新工具和基准测试数据集**：讨论了各种新的数据集和基准测试工具，包括 REVEAL 和 UNcommonsense。分享的链接包括 [REVEAL 数据集](https://reveal-dataset.github.io)、[UNcommonsense 数据集](https://huggingface.co/datasets/allenai/UNcommonsense)，以及使用 Self-Play Preference Optimization 的模型，如 [Llama-3-8B-SPPO-Iter3](https://huggingface.co/UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3)。

- **关于 SB 1047 对创新影响的辩论**：成员们辩论了加州 SB 1047 法案对 AI 创新的潜在负面影响。分享了一个讨论该法案潜在意外后果的[链接](https://live-2024-stop-sb-1047.pantheonsite.io)，一位成员表示：*“加州应该鼓励创新，并学会利用 AI 来发挥我们的战略优势。”*

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/datasets/aaronday3/entirety_of_reddit">aaronday3/entirety_of_reddit · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://x.com/natolambert/status/1806437722334417131">Nathan Lambert (@natolambert) 的推文</a>：这是我与 @deanwball 关于 AI 政策的完整 @interconnectsai 访谈。我们几乎对 AI 政策的所有事项进行了国情咨文式的回顾，通常关注点在于开放性。这是一次很棒的访谈！...</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - 一个由 open-llm-leaderboard 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://live-2024-stop-sb-1047.pantheonsite.io/">保护 AI 研究 | 停止 SB 1047</a>：敦促立法机构保护人工智能 (AI) 研究并反对 SB 1047。与其在 AI 萌芽阶段过度监管，我们应该鼓励创新，并学会利用 AI 来发挥我们的战略优势...</li><li><a href="https://suno.com/song/8aaf9c47-41d5-4a04-a05c-f3a49218ccc8">@bozoegg 的 Beneath the Stars | Suno</a>：渐进式电子大气歌曲。在 Suno 上聆听并创作你自己的歌曲。</li><li><a href="https://reveal-dataset.github.io">REVEAL</a>：思维链的强度取决于其最薄弱的环节：推理链验证器的基准测试</li><li><a href="https://tenor.com/view/hug-love-heart-globe-planet-gif-16782181">拥抱爱 GIF - 拥抱爱心 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/interstellarninja/function-calling-eval/tree/main">GitHub - interstellarninja/function-calling-eval: 一个用于评估 LLM 发起的 function calls 的框架</a>：一个用于评估 LLM 发起的 function calls 的框架 - interstellarninja/function-calling-eval</li><li><a href="https://huggingface.co/google/gemma-2-9b">google/gemma-2-9b · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/allenai/UNcommonsense">allenai/UNcommonsense · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3">UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/QuanquanGu/status/1805675325998907413">Quanquan Gu (@QuanquanGu) 的推文</a>：我们已经开源了 Self-Play Preference Optimization (SPPO) 的代码和模型！🚀🚀🚀 ⭐ 代码：https://github.com/uclaml/SPPO 🤗模型：https://huggingface.co/collections/UCLA-AGI/sppo-6635f...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/25Ij9G4haQ">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1255992054740287599)** (2 条消息): 

- **对 Hermes2-Pro-llama-3-70B 的热情**：一位用户表达了对 **Hermes2-Pro-llama-3-70B** 的兴奋。他们询问在哪些场景下该模型会优于 **Hermes-2-Theta**。
- **无上下文分享的链接**：另一位用户分享了一个特定 Discord 消息的链接 [链接](https://discord.com/channels/1053877538025386074/1149866623109439599/1255975965272838275)，暗示其中可能包含相关信息或上下文。
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1256026605780734002)** (85 条消息🔥🔥): 

- **Glaive-RAG-v1 数据集发布**：[Glaive-RAG-v1](https://huggingface.co/datasets/glaiveai/RAG-v1) 包含约 5 万个使用 Glaive 平台构建的 RAG 场景样本，结构包含文档、问题、答案和引用标签。成员们被要求评估其质量以及未来迭代的潜在改进。
- **系统提示词与领域集成**：讨论了将每个领域的 System Prompts 集成到 Hermes RAG 提示词中，涵盖了 "role"（角色）、"style guide"（风格指南）和 "instructions"（指令）部分。成员们正在考虑在提示词中指示上下文和相关性的实际方法。
- **相关性评分机制**：关于是否包含相关性评分（如用于评估响应 Groundedness 的 5 点李克特量表）存在持续辩论。共识倾向于让 LLM 通过引导式 System Prompts 自行评估这些指标。
- **代码与工具分享**：成员们讨论了分享为该项目开发的工具和流水线（pipelines）的效用。例如通过 [Hugging Face Spaces](https://huggingface.co/spaces/not-lain/image-retriever) 分享的图像检索流水线。
- **Grounded 与 Mixed 响应模式**：澄清了在 "Grounded" 模式下，模型应仅使用提供文档中的信息；而在 "Mixed" 模式下，它会将文档信息与模型自身的知识相结合。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/not-lain/image-retriever">Image Retriever - a Hugging Face Space by not-lain</a>: 未找到描述</li><li><a href="https://x.com/bee__computer/status/1806448042406818203">来自 bee (@bee__computer) 的推文</a>: 我们还有一些 alpha 版本可用。在 @aiDotEngineer 获取。私信我们</li><li><a href="https://huggingface.co/datasets/glaiveai/RAG-v1">glaiveai/RAG-v1 · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/explodinggradients/ragas">GitHub - explodinggradients/ragas: 用于检索增强生成 (RAG) 流水线的评估框架</a>: 用于检索增强生成 (RAG) 流水线的评估框架 - explodinggradients/ragas</li><li><a href="https://docs.google.com/spreadsheets/d/1f5fbPxhjGrmPqhbM0exOCX2vAzffRWufhF7QBp24OMw/edit?gid=0#gid=0">RAG 数据合成</a>: 表格 1 领域、课程文件、来源/链接、HF 仓库、大小（行）、状态、负责人、审核人、审核笔记 Websearch Wikipedia Codebase, WIP, Bexboy Academic Papers Books, WIP, EveryoneIsGross Finance...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1255976707345879162)** (5 条消息): 

- **对 world sim 中 Claude 3.5 Sonnet 模型的极高热情**：一位成员对即将加入世界模拟（world simulation）的 **Claude 3.5 Sonnet 模型** 表示兴奋。这表明社区对新 AI 模型集成有着浓厚兴趣。

**提到的链接**：<a href="https://tenor.com/view/lain-lain-iwakura-serial-experiments-lain-wires-wired-gif-1481475804337586659">Lain Lain Iwakura GIF - Lain Lain iwakura Serial experiments lain - 发现并分享 GIF</a>：点击查看 GIF

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1255962506417934358)** (25 条消息🔥): 

- **聚合数据增加隐私风险**：“很快你就会遇到这样一种情况：来自其他用户的聚合数据可以告知或改进本地模型，”这导致了**隐私保护/联邦学习 (Federated Learning) 领域**的挑战。这使得必须应对“来自恶意行为者显著更广泛的攻击空间” ([BSI 链接](https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/Publications/Studies/KI/P464_Provision_use_external_data_trained_models.pdf?__blob=publicationFile&v=7))。

- **AI Agent 安全行为**：讨论了 AI Agent 实施安全行为的情况，包括“通过脚本识别损害隐私的行为”以及“自动降级或销毁可疑数据收集的机制”等活动。然而，一些人认为这在很大程度上可能是启发式的 (heuristic)，缺乏 AI 的泛化能力。

- **新的 MMLU-SR 数据集**：一位用户介绍了 MMLU-SR，这是一个新的**推理挑战数据集**，旨在衡量大语言模型 (LLMs) 的理解能力 ([arXiv 链接](https://arxiv.org/abs/2406.15468v1))。他们发现 LLMs 在修改后的测试问题上表现不佳，这表明其缺乏真正的理解能力。

- **聊天中的挑衅问题**：多名用户举报了一个被封禁的用户“endomorphosis”，他在特定频道使用多个账号进行挑衅。成员们要求将其移除，以获得更积极的社区体验。

- **lm_eval 帮助的频道指南**：寻求 **lm_eval** 协助的新成员被引导至相应的频道 ([lm_eval 频道链接](https://discord.com/channels/729741769192767510/755950983669874798))。该频道被推荐用于处理任务和相关查询。

**提到的链接**：<a href="https://arxiv.org/abs/2406.15468v1">Reasoning or Simply Next Token Prediction? A Benchmark for Stress-Testing Large Language Models</a>：我们提出了 MMLU-SR，这是一个旨在通过挑战 LLMs 在修改术语后的问答任务中的表现，来衡量大语言模型 (LLMs) 真正理解能力的新型数据集...

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1255960857041440989)** (122 条消息🔥🔥): 

- **关于 yoco 和 kv cache 的辩论**：成员们讨论了各种架构中的 **kv cache** 策略，特别是 **yoco**，并对其与替代设计相比的有效性表示怀疑。一位成员称 yoco 的设置“非常诡异（cursed）”，原因是其复杂性以及与标准 Transformer 实践的脱节。

- **层排序的效率**：围绕 **yoco** 和 **mamba** 等模型中 **Attention 和 Feed-forward 层（FFN）的排序**展开了深入讨论。一些人认为将所有 Attention 层放在一端可能更高效，从而降低计算成本，而另一些人则坚持认为交替层可能确保更好的整体性能。

- **模型缩放（Scaling）与性能考量**：参与者辩论了 **层排序的影响** 在小型与大型模型之间的差异，一些人建议小型模型中的问题可能会在大型模型中被平滑掉。一个争论的关键点是，随着模型规模的增长，重新排序层是否会产生可衡量的影响。

- **位置嵌入（Positional Embeddings）的初步探索**：一位成员提出了一个关于在计算 QK 之前将 **位置嵌入（PE）** 预先应用于 Latents 的创新想法，假设这可以更好地处理字符串反转等操作。这引发了成员们的好奇和怀疑，他们质疑这种方法是否会保留或破坏 RoPE 等现有技术的优势。

- **强化学习（Reinforcement Learning）进展**：一位成员分享了一篇 [arXiv 论文](https://arxiv.org/abs/2406.19320)，讨论了 **$\Delta$-IRIS**，这是一种在 RL 中采用基于 Delta 的自动编码的新 Agent，探讨了其在训练时间上相对于传统基于 Attention 方法的效率。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.19320">Efficient World Models with Context-Aware Tokenization</a>: 扩展深度强化学习（RL）方法面临重大挑战。随着生成式建模的发展，基于模型的 RL 成为强有力的竞争者。最近的进展...</li><li><a href="https://arxiv.org/abs/1911.03864">Improving Transformer Models by Reordering their Sublayers</a>: 多层 Transformer 网络由交错的 Self-attention 和 Feedforward 子层组成。以不同的模式排列子层是否能带来更好的性能？我们生成了随机或...</li><li><a href="https://arxiv.org/abs/2403.00801">Self-Retrieval: Building an Information Retrieval System with One Large Language Model</a>: 大语言模型（LLM）的兴起改变了信息检索（IR）系统在人类获取信息方式中的角色。由于孤立的架构和有限的交互...</li><li><a href="https://arxiv.org/abs/2012.15832">Shortformer: Better Language Modeling using Shorter Inputs</a>: 增加输入长度一直是 Transformer 语言建模进步的驱动力。我们确定了较短输入无害的条件，并实现了困惑度（Perplexity）和效率的提升...</li><li><a href="https://colab.research.google.com/drive/1sXpRXz8KWa_OnWveOWvu1ZtgUL3tnPai?usp=sharing">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1256062431613161513)** (45 条消息🔥): 

- **质疑 Chinchilla 作为自然法则的地位**：一位成员质疑为什么人们似乎将 **Chinchilla scaling** 视为不可改变的法则，并认为 power law scaling 不可能是最终的最佳扩展方法。他们观察到关于替代方案的讨论似乎很少，并推测关于这一话题的严肃讨论可能正在私下进行。

- **辩论 Power Law Scaling 的有效性**：一位成员认为 **power law scaling** 可能仍然是一个合理的模型，但承认 **Chinchilla** 模型的关联性与特定的训练方案和数据等条件挂钩。他们还思考了为什么反比例关系不能成为常态，并提到反比例和对数扩展（logarithmic scaling）似乎都是合理的。

- **Scaling Heuristics 与 Laws 的对比**：另一位成员指出，或许应该用 "scaling heuristic"（扩展启发式）来取代 "scaling law"（扩展法则）等术语，以更好地反映其临时性的本质。他们回忆起曾有大量论文声称发现了比 Chinchilla 更好的新“法则”，暗示了对这种定论式语言的怀疑。

- **阅读并引用关键论文**：成员们引用了几篇关键论文来支持他们的论点，包括 ["Parameter Counts in Machine Learning"](https://www.alignmentforum.org/posts/GzoWcYibWYwJva8aL/parameter-counts-in-machine-learning) 和 [Adlam 2021 关于 Scaling Laws 的论文](https://gwern.net/doc/ai/scaling/2021-adlam.pdf)。他们讨论了这些论文如何理解和建模关于大型数据集和参数规模的 scaling laws。

- **实际局限性与未来方向**：对话还涉及了 **data selection methods**（数据选择方法）及其对训练效率影响等实际层面。一位成员强调，更好的数据收集方法（如分层采样）并非万能药，但确实能提高效率，并突显了预测数据对模型性能未来影响的复杂性。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2210.16859">A Solvable Model of Neural Scaling Laws</a>: 拥有海量参数的大型语言模型，在接近互联网规模的 token 数量上进行训练时，已被经验证明遵循神经扩展法则：具体而言，它们的性能表现...</li><li><a href="https://arxiv.org/abs/1905.10843">Asymptotic learning curves of kernel methods: empirical data v.s. Teacher-Student paradigm</a>: 学习一个监督任务需要多少训练数据？通常观察到泛化误差随 $n^{-β}$ 降低，其中 $n$ 是训练样本的数量，$β$ 是一个指数...</li><li><a href="https://en.wikipedia.org/wiki/Empirical_statistical_laws">Empirical statistical laws - Wikipedia</a>: 未找到描述</li><li><a href="https://www.alignmentforum.org/posts/GzoWcYibWYwJva8aL/parameter-counts-in-machine-learning">Parameter counts in Machine Learning — AI Alignment Forum</a>: 简而言之：我们汇编了 1952 年至…之间 n=139 个机器学习系统的开发日期和可训练参数计数的信息。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1256021537606074398)** (15 messages🔥): 

- **将 MMLU-SR 数据集引入 lm_eval**：一位成员介绍了一个名为 **MMLU-SR** 的新数据集，旨在通过符号替换挑战 LLM 的推理能力，并询问如何将其添加到 **lm_eval**。在创建并提交 PR 后，他们迅速收到了评审回复。[arxiv.org/abs/2406.15468v1](https://arxiv.org/abs/2406.15468v1)
  
- **添加 MedConceptsQA 基准测试**：一位成员请求评审他们的 PR，该 PR 添加了旨在进行医学概念问答的 **MedConceptsQA** 基准测试。这个开源基准测试包含各种复杂程度的问题。[github.com/EleutherAI/lm-evaluation-harness/pull/2010](https://github.com/EleutherAI/lm-evaluation-harness/pull/2010)

- **自定义 YAML 配置调试**：一位成员寻求帮助，以使用 harness 运行自定义 YAML 配置进行评估。他们收到了调试建议，并在识别并修复任务名称冲突后成功解决了问题。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/dancing-cat-dance-cat-cat-meme-chinese-cat-gif-6295992666678715767">Dancing Cat Dance GIF - Dancing cat Dance Cat - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2010">由 Ofir408 添加 MedConceptsQA 基准测试 · Pull Request #2010 · EleutherAI/lm-evaluation-harness</a>: 您好，我添加了名为 MedConceptsQA 的新基准测试。MedConceptsQA 是一个专门用于医学概念问答的开源基准测试。该基准测试由各种...的问题组成。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2030">在 `subprocess` 函数调用中使用 `shell=False` 由 pixeeai 提交 · Pull Request #2030 · EleutherAI/lm-evaluation-harness</a>: 此代码修改将 subprocess 模块函数调用中设置为 True 的 shell 关键字参数设置为 False。设置 shell=True 将通过系统 shell 执行提供的命令，而...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1256242980801740830)** (6 messages): 

- **GPTNeoX 中的指令微调**：一位成员询问了在 **GPTNeoX** 中进行 **Instruction Tuning** 的可能性，特别是损失仅针对续写部分（continuations）而非提示词（prompts）进行反向传播的情况。另一位成员建议查看一个[特定的 PR](https://github.com/EleutherAI/gpt-neox/pull/1240)以及相关的预处理脚本——"[preprocess_data_with_chat_template.py](https://github.com/EleutherAI/gpt-neox/blob/main/tools/datasets/preprocess_data_with_chat_template.py)"——并指出该脚本虽仍在评审中，但欢迎提交 Bug 报告。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/pull/1240">SFT 改进（标签修复、不同的打包实现）由 dmahan93 提交 · Pull Request #1240 · EleutherAI/gpt-neox</a>: 添加了不同的打包实现（未打包、打包直到溢出），虽然有点简陋，但对于 SFT 来说应该不是问题；修复标签以使其也具有 valid/test 实现；修复 _get_batch 中的标签掩码...</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/main/tools/datasets/preprocess_data_with_chat_template.py">gpt-neox/tools/datasets/preprocess_data_with_chat_template.py at main · EleutherAI/gpt-neox</a>: 基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自回归 Transformer - EleutherAI/gpt-neox
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1256224175442362448)** (12 messages🔥): 

- **Triton 安装问题困扰 Windows 用户**：一位用户报告了 torch.compile 的持续问题，尽管安装了 Triton，仍遇到 "RuntimeError: Cannot find a working triton installation"。另一位成员澄清说 Triton 可能在 Windows 上没有官方支持，并建议了替代安装方法。
- **使用 Anaconda 调试 Triton**：一位用户提到使用 Anaconda 安装 PyTorch 并遇到了 Triton 相关错误。另一位用户确认在 Windows 上无法通过 `conda install triton` 获取 Triton 包，并要求提供 `conda list` 的输出以进一步排查。
- **寻求 make_block_ptr 的文档**：一位成员询问关于 `triton.language.make_block_ptr` 的详细文档，对现有信息的匮乏表示困惑。
  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1256049551031144458)** (7 messages): 

- **Lovely Tensors 在 torch.compile 2.5.0 中报错**: [Lovely Tensors](https://github.com/xl0/lovely-tensors) 的作者在 `torch.compile()` 中遇到了自定义 `Tensor.__repr__()` 损坏的问题。这是因为他们的 `__repr__` 在 FakeTensor 上被调用，从而导致了错误；一种解决方法是检查该 Tensor 是否为 fake。

- **社区建议使用 torch.compiler API**: 有建议指出可以使用 [torch.compiler_fine_grain_apis](https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html) 来禁用或处理自定义的 `repr` 函数。这种方法可能为面临类似问题的用户提供解决方案。

- **NCCL 的 Broadcast 死锁问题**: 旧版本 NCCL 中的 [broadcast 死锁问题](https://github.com/NVIDIA/nccl/issues/1251) 曾是一个严重问题，但在未随 torch 2.3.1 附带的新版本中已得到修复。用户可以通过执行 `pip install nvidia-nccl-cu12==2.22.3` 来解决，详情参考 [此 TGI PR](https://github.com/huggingface/text-generation-inference/pull/2099)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html">用于细粒度追踪的 TorchDynamo API &mdash; PyTorch 2.3 文档</a>: 未找到描述</li><li><a href="https://github.com/xl0/lovely-tensors">GitHub - xl0/lovely-tensors: 供人类阅读的 Tensors</a>: 供人类阅读的 Tensors。通过在 GitHub 上创建账号来为 xl0/lovely-tensors 的开发做出贡献。</li><li><a href="https://github.com/NVIDIA/nccl/issues/1251">FIFO 队列泄漏 · Issue #1251 · NVIDIA/nccl</a>: 我们遇到了一个问题，8 个进程（每个进程控制节点上的一个 GPU）同时全部锁定。这似乎是确定性的，尽管我们不确定具体是哪个操作导致的...</li><li><a href="https://github.com/huggingface/text-generation-inference/pull/2099">修复 PyTorch 2.3 升级带来的 NCCL 回归问题，由 fxmarty 提交 · Pull Request #2099 · huggingface/text-generation-inference</a>: 正如标题所述，修复了 TGI CUDA 镜像中 NVIDIA/nccl#1251 的问题，该回归是在 #1730 和 #1833 中引入的。我们在 H100 上使用 TP=4 或 TP=8 的 llama 3 70B 模型以及默认 CUDA graph 时遇到了此问题...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1256025541685940284)** (1 messages): 

- **关于编写 CUDA 程序的思考**: Stephen Jones 分享了一个关于[如何思考编写 CUDA 程序](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62401/)的科普环节。主题包括 *wave quantization & single-wave kernels*、并行类型以及通过 tiling 优化 L2 cache 的 block sizes。

**提及的链接**: <a href="https://www.nvidia.com/en-us/on-demand/session/gtc24-s62401/">如何编写 CUDA 程序：Ninja 版 | NVIDIA On-Demand</a>: 加入 CUDA 架构师的深度探讨，了解如何将应用程序映射到大规模并行机器上，涵盖一系列旨在优化性能的不同技术。

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1256091717308776570)** (14 messages🔥): 

- **CUDA 文件类型困惑？别担心！**: 一位成员询问在 Visual Studio 的 CUDA Runtime 项目中，是否应该将所有文件的项类型都设置为 CUDA C/C++。另一位成员建议这取决于个人偏好，只要需要 CUDA 的文件被正确标记即可：“我通常将 CUDA 文件设置为 .cu，非 CUDA 文件设置为 .c/cpp... 这纯属个人偏好。”
  
- **用于 CUDA 实操体验的云端 GPU**: 一位初学者询问在没有本地 GPU 的情况下如何在云端 GPU 上使用 CUDA Toolkit，并寻求高性价比的云厂商。建议包括 [Vast.ai](https://vast.ai/) 和 [Runpod.io](https://www.runpod.io/)，并提到 Lightning AI 提供每月 22 小时的免费 L4 使用额度。

- **Python 到 CUDA 的优化流程**: 对于优化 PyTorch 代码，推荐的流程是：先使用 `torch.compile`，考虑自定义 Triton，最后在必要时编写自定义 CUDA 代码。关键建议包括检查 GPU 瓶颈，并使用像 `F.spda()` 这样高效的 attention 实现以确保最大化利用率。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://vast.ai/">租用 GPU | Vast.ai</a>: 通过最佳的云端 GPU 租赁服务，将您的云计算成本降低 3-5 倍。Vast.ai 简单的搜索界面允许公平比较所有供应商的 GPU 租赁价格。</li><li><a href="https://www.runpod.io/">RunPod - 为 AI 构建的云</a>: 在一个云端开发、训练和扩展 AI 模型。通过 GPU Cloud 开启按需 GPU，通过 Serverless 扩展 ML 推理。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1256232766786240613)** (1 messages): 

- **PMPP 书籍缺页**：一位成员报告说他们最近购买的 PMPP（第 4 版）**书籍缺失了几页**——具体是 148, 149, 150, 290, 291, 292, 447 和 448。他们询问是否有人遇到同样的问题。
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1255968160587317412)** (16 messages🔥): 

```html
- **自定义静态分析工具讨论**：一位用户提到想在项目上运行自定义静态分析工具。这引起了小组内的兴奋和赞同。
- **需要 torch/aten ops 需求列表**：一位成员建议维护一个针对不同 tensor subclass 使用场景（如 `FSDP`）所需的 `torch/aten ops` 列表或表格。例如，为了交换线性权重，实现 `F.linear` 和 `aten.detach.default` 是必要的。
- **__torch_dispatch__ 的递归错误**：一位用户在 `__torch_dispatch__` 中打印参数时遇到了递归错误，引发了关于可能原因和解决方案的讨论。这包括检查 `__repr__()` 中的特殊函数以及使用调试器进行检查。
- **Int4Tensor 重构 PR**：创建了一个 [PR](https://github.com/pytorch/ao/pull/458) 用于重构 `Int4Tensor` 并进行一些代码清理，该工作将在周末完成。 
- **NVIDIA GeForce GTX 1650 警告**：一位用户对 NVIDIA GeForce GTX 1650 不原生支持 bfloat16 编译的警告表示担忧。澄清指出，这可能会导致性能影响，例如多次 kernel 启动，这与 quant API 中 bfloat 的使用有关。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/blob/26d633b7213c80371985ba88e6db4a2f796a2e50/torch/_inductor/compile_fx.py#L1647C5-L1647C31),">pytorch/torch/_inductor/compile_fx.py at 26d633b7213c80371985ba88e6db4a2f796a2e50 · pytorch/pytorch</a>: Python 中的 Tensor 和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/ao/pull/458">[WIP] Int4Tensor refactor to implements pattern by melvinebenezer · Pull Request #458 · pytorch/ao</a>: 重构 UInt4Tensor 以具有类似于 nf4tensor 和 UInt2Tensor 的 implements 模式。待办事项：为 UInt4Tensor 和 PerChannelSymmetricWeight 创建 implements；测试用例；将 uint4i 移动到 uint4.py</li><li><a href="https://github.com/vayuda/ao/blob/intx/torchao/prototype/intx/intx.py#L365C4-L368C13">ao/torchao/prototype/intx/intx.py at intx · vayuda/ao</a>: 用于量化和稀疏化的原生 PyTorch 库 - vayuda/ao</li><li><a href="https://github.com/pytorch/ao/issues/391">[RFC] Tensor Subclass based Quantization API · Issue #391 · pytorch/ao</a>: 状态：草案 更新日期：06/17/2024 目标：在本文档中，我们将讨论面向建模用户和开发者的基于 Tensor subclass 的量化 API。建模用户 API 指的是...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1256151743448875019)** (9 条消息🔥): 

- **HuggingFace Hub 上的模型发布优先考虑存储便利性**：一位用户注意到一种日益增长的趋势，即模型架构和预处理代码直接存储在 HuggingFace Hub 上，而不是添加到 transformers 仓库中。他们推测这是否源于**代码许可问题 (code licensing issues)**，并分享了[此处](https://huggingface.co/microsoft/Florence-2-base-ft/tree/main)和[此处](https://huggingface.co/Qwen/Qwen-Audio/tree/main)的示例。

- **HuggingFace Hub 策略的优缺点**：另一位用户指出了这种策略的好处，例如作者无需官方 HF 团队发布即可发布新模型，但也提到了缺点，比如除非启用 `trust_remote_code`，否则无法在 HF 平台上将这些模型用于某些功能。

- **关于最佳代码和模型存储方案的辩论**：讨论强调了关于存储模型代码和权重的最佳实践的不同意见。一位用户建议在 GitHub 上发布代码并在 **HuggingFace Hub** 上存储模型权重可能是理想的，尽管其他人指出了潜在的兼容性问题以及使用 HF Hub 的便利因素。

- **以 Llama 发布为例**：讨论提到了 **Llama 模型**的发布策略，其中包括在 GitHub 上维护独立于 transformers 库的推理代码。这种方法的一个示例仓库是 [GitHub 上的 meta-llama](https://github.com/meta-llama/llama)。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/meta-llama/llama">GitHub - meta-llama/llama: Inference code for Llama models</a>: Llama 模型的推理代码。通过在 GitHub 上创建账户为 meta-llama/llama 的开发做出贡献。</li><li><a href="https://huggingface.co/microsoft/Florence-2-base-ft/tree/main">microsoft/Florence-2-base-ft at main</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen-Audio/tree/main">Qwen/Qwen-Audio at main</a>: 未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct/tree/main">deepseek-ai/DeepSeek-Coder-V2-Instruct at main</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1255975840458477700)** (68 messages🔥🔥): 

- **Google Gemma 2 在基准测试中表现出色**：Google 新推出的 Gemma 2 模型（27B 和 9B）在 LYMSYS Chat arena 中超越了 Llama3 70B 和 Qwen 72B。[发布详情](https://x.com/reach_vb/status/1806343018640781675)赞赏了实验的公开分享，并计划很快发布 2.6B 模型。
- **Danielhanchen 分析 Gemma 2 架构**：Daniele Hanchen 强调了 Gemma 2 模型的核心要素，包括 pre 和 post Layernorms 以及 [approx GeGLU 激活函数](https://x.com/danielhanchen/status/1806372357684220308)。该模型使用 sliding window 和 global attention 层来实现高效处理。
- **ReLU 与 GELU 之争**：讨论了在激活函数方面 ReLU 是否优于 GELU，引用了一篇 [arxiv 论文](https://arxiv.org/pdf/2002.05202v1)、测试结果以及硬件优势。辩论内容包括了为 ReLU 设置准确超参数的重要性。
- **FP8 在硬件和库方面的挑战**：探讨了当前硬件和库对 FP8 支持的挑战，重点关注 NVIDIA 的 NCCL 和 cuDNN 的局限性。随后对替代方案和潜在的解决方法进行了详细讨论，包括引用 [Microsoft 的 FP8-LM 论文](https://arxiv.org/pdf/2310.18313)。
- **训练见解与优化**：Yuchen 使用 H100 GPU 进行的训练显示，更高的学习率带来了不错的结果，这表明所面临的问题可能与特定平台或数据集有关。随后讨论了各种优化器及其具体实现细节，指出了训练过程的复杂性和敏感性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/reach_vb/status/1806343018640781675">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：冲啊！Google 刚刚发布了 Gemma 2 27B & 9B 🔥 > 在 LYMSYS Chat arena 中超越了 Llama3 70B/ Qwen 72B/ Command R+，且 9B 是目前 15B 以下最强的模型。 > 比 Llama 2.5 倍小...</li><li><a href="https://x.com/danielhanchen/status/1806372357684220308">来自 Daniel Han (@danielhanchen) 的推文</a>：刚刚分析了 Google 新发布的 Gemma 2！9B & 27B 的 Base 和 Instruct 版本都上线了！ 1. Pre & Post Layernorms = 像 Grok 一样多出 2 倍的 LN。 2. 使用了 Grok 的 softcapping！Attn logits 被截断至 (-30, 3...</li><li><a href="https://x.com/Yuchenj_UW/status/1806713556047716603">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：使用 @karpathy 的 llm.c，仅用 1/5 的训练 token 就超越了 GPT-3 (1.5B) 🌠 之前我训练了 GPT-2 Small (124M) 模型。最近我使用...训练了 GPT-2 XL (1.5B)，即官方的 GPT-2。</li><li><a href="https://github.com/clu0/unet.cu">GitHub - clu0/unet.cu: 纯 CUDA 实现的 UNet 扩散模型</a>：纯 CUDA 实现的 UNet 扩散模型。欢迎通过创建账号为 clu0/unet.cu 的开发做出贡献。</li><li><a href="https://github.com/Azure/MS-AMP">GitHub - Azure/MS-AMP: Microsoft 自动混合精度库</a>：Microsoft 自动混合精度库。欢迎通过创建账号为 Azure/MS-AMP 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1256052708700454932)** (1 messages): 

- **为公益组织降低 Perplexity Enterprise Pro 的价格**：Perplexity 现在向学校、非营利组织、政府机构和非营利团体提供 Perplexity Enterprise Pro 的优惠价格。该倡议旨在支持那些在社会和教育发展中发挥重要作用但面临预算限制的组织。[了解更多](https://pplx.ai/s3oZX49)。
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1255963603161583636)** (94 条消息🔥🔥): 

- **探讨 Perplexity 的 RAG 性能**：成员们讨论了 Perplexity 的 Relevance Aware Generation (RAG) 机制有时如何导致输出质量下降，特别是在尝试以不一致的方式整合文件时。有人指出，**writing mode 旨在避免 RAG**，但实际结果仍经常出现幻觉 (hallucinations)。
  
- **Claude 3 Opus 使用限制令用户沮丧**：**Claude 3 Opus** 的每日使用限制一直是用户持续沮丧的根源，该限制从 **5 次波动到 600 次，现在上限为每天 50 次对话**。一位用户将这种限制变化描述为“坐过山车”。
  
- **解答安全和数据问题**：一名成员询问了 Perplexity 企业解决方案的**安全措施**和 **PII 处理**。回复将他们引导至 [Trust Center](https://trust.perplexity.ai/) 并提供了一个用于进一步咨询的电子邮件。
  
- **Perplexity 的间歇性上下文问题**：用户注意到 Perplexity 在长时间交互中往往会丢失上下文。一位用户建议在修复方案实施前，使用 **“#context” 等关键词**来提高连贯性。
  
- **VPN 和访问问题**：一些成员报告了 Perplexity 在 **Cloudflare WARP** 等 **VPN** 上无法工作的问题，导致连接和登录故障。有人建议切换到 DNS-only 模式作为变通方案。

**Link mentioned**: <a href="https://trust.perplexity.ai/.">Trust Center</a>: 展示我们的安全态势，以在网络上建立信任。

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1255978834453467186)** (8 条消息🔥): 

- **探索 Android 14 见解**：分享了一个指向 [Perplexity AI 页面](https://www.perplexity.ai/page/Android-14-boosts-WxN8GGdgRQSKPPn7DftxtQ)的链接，详细介绍了 Android 14 引入的功能和增强。该页面可能讨论了操作系统更新的细节及其对用户体验的影响。

- **关于 Perplexity AI 上 RDP 的问题**：提供的链接深入探讨了 [Microsoft Remote Desktop Protocol (RDP)](https://www.perplexity.ai/page/Why-Microsoft-STILL-ZHlLAmi2Riyx3S0.Li.kwg)。它讨论了 Microsoft 生态系统中 RDP 使用的持续考量和潜在改进。

- **CriticGPT、活体机器人皮肤和可持续创新**：分享了一个 YouTube 视频，标题显示正在讨论 [CriticGPT、活体机器人皮肤和受牡蛎启发的混凝土](https://www.youtube.com/embed/Kqw56PUMa6M)。该视频似乎涵盖了受自然方案启发的尖端技术和可持续材料。

- **Linux 性能探索**：一个指向 [Perplexity AI 搜索](https://www.perplexity.ai/search/why-Linux-are-E54_z89aSlKnHo6nVWPaIg)的链接深入探讨了 Linux 性能和采用问题背后的原因。该页面可能探讨了 Linux 用户面临的常见挑战和解决方案。

- **误导性的 Minecraft 机制**：分享了一篇名为《[Minecraft 修复机制误导儿童](https://www.perplexity.ai/page/Minecraft-Repair-Mechanics-NdRggXKXRXyGY8LgKsp1dQ)》的文章。它提出了对儿童在玩游戏时可能产生的机械知识误解的担忧。

**Link mentioned**: <a href="https://www.youtube.com/embed/Kqw56PUMa6M">YouTube</a>: 未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1256005005878497360)** (13 条消息🔥): 

- **社区希望支持 Gemma 2 模型**：“我知道这不言而喻，但在可用模型中加入 Gemma 2 会很酷。”
- **关于支持更大模型和 Pro Search 的问题**：一位成员询问：“Perplexity API 何时开始支持 Pro Search 以及像 GPT-4 和 Sonnet 这样更大的模型？”另一位成员回答说，GPT-4 和 Sonnet 可以通过各自的供应商使用，而 API 的 Pro Search 目前没有计划。
- **澄清 Perplexity 的附加价值**：一位成员指出：“Perplexity 的核心意义在于它会增加更多的在线搜索和更好的 Prompt 来与 GPT-4 或 Sonnet 对话，从而提供更好的体验。”目前的模型如 `llama-3-sonar-large-32k-online` 可以使用特定参数，这些参数可以在 [Perplexity model cards](https://docs.perplexity.ai/docs/model-cards) 文档中找到。
- **对当前 API 性能的批评**：一位成员表达了不满，称：“我尝试过它们，但它们在理解力上不如 GPT-4 或 Sonnet 3.5”，并指出当前 API 中存在“大量幻觉”。
- **询问如何过滤结果**：另一位成员询问如何将结果限制在过去 30 天内的新信息，得到的建议是尝试使用 `after:2024-05-28`。

**提到的链接**：<a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1255961250786181241)** (40 条消息🔥): 

- **Character.AI 推出 Character Calls**：Character.AI 推出了 [Character Calls](https://blog.character.ai/introducing-character-calls/)，这是一项与 AI 角色进行双向语音对话的功能，在其 App 上免费提供。然而，用户反馈强调了诸如 5 秒延迟和机械化声音等问题，影响了对话的流畅性。
- **Amazon 收购 Adept 的人才和技术**：讨论集中在 [Amazon 聘请 Adept 的联合创始人并授权其技术](https://x.com/anissagardizy8/status/1806812006009442671?s=46)，使 Adept 仅剩约 20 名员工。有传言称 Adept 内部存在毒性文化，导致最初创立该公司的 Transformer 论文作者们离职。
- **对 AI Agent 进展的怀疑**：人们将 AI Agent 的炒作与自动驾驶汽车进行了比较，暗示 Agent “总是指日可待，但从未能可靠运行”。对话指出，尽管有大量的资金和人才投入，但与视频生成等其他 AI 进展相比，实用型 AI Agent 的开发速度较慢。
- **AI Agent 训练数据的挑战**：参与者讨论了开发 AI Agent 的瓶颈很可能是训练数据的收集和质量。重点正在转向生成合成数据并获取带注释的反事实示例，以提高 Agent 的可靠性和性能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.adept.ai/blog/adept-update">An Update to Adept</a>：宣布我们战略和公司的一些更新。</li><li><a href="https://blog.character.ai/introducing-character-calls/">Introducing Character Calls</a>：0:00 / 0:07 1× 呼叫 Character.AI 社区！我们很高兴推出一项令人振奋的新功能，它将重新定义您的 Character.AI 体验：Character Calls！...</li><li><a href="https://x.com/giffmana/status/1806411302190915603?s=46">Lucas Beyer (bl16) (@giffmana) 的推文</a>：@fouriergalois @character_ai 刚试过，不幸的是它没有可比性，本来会非常令人印象深刻！它一点也不流畅。我说话结束后有 5 秒的延迟。我无法打断...</li><li><a href="https://x.com/anissagardizy8/status/1806812006009442671?s=46">Anissa Gardizy (@anissagardizy8) 的推文</a>：据该初创公司的发帖和 Amazon 高管的内部邮件显示，Amazon 已聘请人工智能初创公司 Adept 的联合创始人并授权了其部分技术。Adept 仅剩...
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1255991140621226085)** (7 messages): 

- **Anthropic CEO 怀旧地讨论《最终幻想》(Final Fantasy)**：一位成员分享了 [Anthropic CEO Dario Amodei 的 YouTube 视频](https://youtu.be/xm6jNMSFT7g?si=BnYoL-E1QXGTw23P&t=3880)，他在视频中讨论了自己和姐姐从小到大一直都在玩《最终幻想》。该成员觉得这段轶事非常亲切。

- **AI 危机与大流行病的辩论引发热议**：Natolambert 认为 Dwarkesh 将 AI 危机比作比大流行病更难处理的观点是武断的。Dwarkesh 还发表了颇具争议的言论，称“我们以前做过疫苗”，暗示 COVID 疫苗是常态。

- **政治动荡引发极端期望**：Natolambert 表示，如果 Trump 当选总统，他希望发生一场不稳定的智能爆炸（intelligence explosion），从而使政府变得毫无意义。这种情绪表明了对 AI 发展驱动重大变革的渴望。
  
- **欧洲成员对辩论感到沮丧**：Xeophon 表达了对这场辩论的负面感受，并提到了他的欧洲视角。这暗示了关于 AI 和政治问题的辩论具有更全球化的影响。

**提到的链接**：<a href="https://youtu.be/xm6jNMSFT7g?si=BnYoL-E1QXGTw23P&t=3880">Dario Amodei - CEO of Anthropic | Podcast | In Good Company | Norges Bank Investment Management</a>：Anthropic CEO Dario Amodei：Claude、新模型、AI safety 以及经济影响。下一代 AI 模型会有多大、多强？Anthropic 的 CEO...

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1255980323674656918)** (5 messages): 

- **Memes：《谍影重重》(Bourne Supremacy) 引用**：一位成员分享了一个名为 "[The Bourne Supremacy (9/9) Movie CLIP - Final Call to Pamela (2004) HD](https://youtu.be/I3znSbbu9IU?si=EbbsoUgHAFS1wuMY&t=65)" 的 YouTube 视频。该视频是电影《谍影重重 2》的片段。

**提到的链接**：<a href="https://youtu.be/I3znSbbu9IU?si=EbbsoUgHAFS1wuMY&t=65">The Bourne Supremacy (9/9) Movie CLIP - Final Call to Pamela (2004) HD</a>：《谍影重重》电影片段：http://j.mp/1uvIXs9 购买电影：http://amzn.to/tor8Hh 不要错过最热门的新预告片：http://bit.ly/1u2y6pr 片段描述...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1256212211114967060)** (3 messages): 

- **揭秘 AI Scaling Myths**：文章 [AI Scaling Myths](https://www.aisnakeoil.com/p/ai-scaling-myths) 挑战了 Scaling 的可预测性，认为仅靠 Scaling 几乎不可能实现 AGI。文章指出 LLM 开发者已接近高质量数据的极限，并强调尽管 [scaling laws](https://arxiv.org/abs/2001.08361) 显示了可预测性，但模型大小仍面临下行压力。

- **关于 AGI 定义和合成数据的讨论**：Nathan Lambert 批评该文章没有定义 AGI 且忽略了合成数据（synthetic data），并推荐了[这项关于重写预训练数据的研究](https://arxiv.org/abs/2401.16380v1)。Lambert 还提到，关于行业停止开发大模型的说法是短期现象且与资本支出（capital expenditures）有关，并鼓励在 [Substack](https://open.substack.com/pub/aisnakeoil/p/ai-scaling-myths?r=68gy5&utm_campaign=comment-list-share-cta&utm_medium=web&comments=true&commentId=60317135) 上进行进一步讨论。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://open.substack.com/pub/aisnakeoil/p/ai-scaling-myths?r=68gy5&utm_campaign=comment-list-share-cta&utm_medium=web&comments=true&commentId=60317135">Nathan Lambert 评论 AI Snake Oil</a>：我是粉丝，但我觉得这篇文章陷入了与 AGI 信徒相同的几个陷阱，只不过是从另一个角度：1. 在没有定义的情况下很容易做到这一点。你没有定义 AGI，也没有评论...</li><li><a href="https://www.aisnakeoil.com/p/ai-scaling-myths">AI scaling myths</a>：Scaling 终会枯竭。问题是什么时候。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1255994147794325604)** (19 messages🔥): 

```html
<ul>
    <li><strong>SnailBot News 剧集讨论</strong>：成员们对最新一期的 SnailBot News 剧集表示兴奋，该剧集讨论了关于 Lina Khan (FTC 主席) 在 Hard Fork 上的内容 [TikTok 链接](https://www.tiktok.com/@hardfork/video/7301774206440656171?lang=en)。Natolambert 提到计划在未来进行更多采访，包括 Paperswithcode/Galactica 的 Ross Taylor 和 John Schulman。</li>
    <li><strong>模型输出训练限制</strong>：一位用户强调了数据提供商要求的“请勿在我们的模型输出上进行训练”条款这一有趣点。Natolambert 证实，如果数据提供商不要求，某些模型会取消这一限制，并引用了 DBRX 团队的例子。</li>
    <li><strong>讨论潜在的受访者</strong>：Natolambert 透露了未来剧集的潜在嘉宾，包括 Amanda Askell，一位成员对她以往表现中展现的见解表示热切期待。Xeophon 提到了 Ross Taylor 虽低调但意义重大的见解，引起了小组的兴趣。</li>
    <li><strong>实验室中的昵称与影响力</strong>：420gunna 幽默地提到了昵称“DBRex”，Natolambert 对此表示认领。随后是对 Natolambert 在实验室内部影响力的轻松调侃。</li>
    <li><strong>部署前测试与影响 AI 实验室</strong>：对话涉及了部署前测试问题，以及对 AI 实验室与政府人物影响力的对比。一位成员认为，相比影响政府人物，影响 AI 实验室的想法不太现实。</li>
</ul>
```

**提到的链接**：<a href="https://www.tiktok.com/@hardfork/video/7301774206440656171?lang=en">TikTok - Make Your Day</a>：未找到描述

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1256283272808366131)** (2 messages): 

- **使用 llama-agents 构建 Agentic RAG 服务**：一个 Notebook 演示了如何创建 Vector Indexes，将其转化为 Query Engines，并在将这些工具作为服务启动前提供给 Agents。详细步骤请查看此 [Notebook](https://t.co/WYTCaqs6Yb)。
- **Jina 发布了迄今为止最好的 Reranker**：LlamaIndex 用户对新的 Jina Reranker 充满热情，称其为迄今为止最好的版本。更多详情可以在[这里](https://t.co/YsYoVOIirb)找到。

**提到的链接**：<a href="https://t.co/WYTCaqs6Yb">llama-agents/examples/agentic_rag_toolservice.ipynb at main · run-llama/llama-agents</a>：通过在 GitHub 上创建账号来为 run-llama/llama-agents 的开发做出贡献。

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1255982061328535603)** (68 messages🔥🔥): 

- **Embedding 节点权重及 Vector Retrievers 的问题**：多位成员讨论了 **LlamaIndex** 的 Embedding 问题，重点关注节点的哪些部分被嵌入，以及 **Vector Retrievers** 产生较差结果的问题（可能是由于 Embedding 模型不匹配）。一位成员建议：“开始调试的最有效方法是创建一个简单的测试用例，重现某些非预期结果。”
- **实体关系链接**：成员们辩论是否根据 Embedding 条件添加边，以更好地捕获传统方式无法检测到的实体关系。他们提到 **Neo4J 和 LlamaIndex 合作的一篇关于实体解析的文章**可能会有帮助。
- **Claude 的空响应**：关于通过 **Bedrock** 处理 **Claude** 响应的技术讨论，提到如果 **max tokens** 设置过低会导致空响应。一个边缘情况导致了 *IndexError*，促使一位成员分享了临时修复方案，并承诺会清理并分享验证用的 Notebook。
- **对新发布的兴奋感**：成员们对新的 **Gemma2 模型**和最新的 **Agents 框架公告**表示热切期待。分享了 [Hugging Face 上的 Gemma2 模型链接](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF)，成员们正在排查集成问题。
- **OpenAI Key 环境变量的挑战**：一位用户报告了非预期行为，即尽管在代码中设置了 **OpenAI keys**，系统仍会从环境变量中寻找。此外，还出现了关于**批量和并行加载索引**以更快处理大文件的优化咨询。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llamahub.ai/l/tools/llama-index-tools-openapi">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/gemma-2-9b-it-GGUF">bartowski/gemma-2-9b-it-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1256295002452135956)** (1 条消息): 

- **Gemma 2.9B 发布新模型**：新款 [google/gemma-2-9b-it](https://openrouter.ai/models/google/gemma-2-9b-it) 的*免费版和标准版*现已上线。OpenRouter 宣布了这一 2023-2024 年度的更新。

- **宣布降价**：多款热门模型已降价。显著降幅包括 [cognitivecomputations/dolphin-mixtral-8x22b](https://openrouter.ai/models/cognitivecomputations/dolphin-mixtral-8x22b) 降价 10%，[openchat/openchat-8b](https://openrouter.ai/models/openchat/openchat-8b) 降价 20%，以及 [meta-llama/llama-3-70b-instruct](https://openrouter.ai/models/meta-llama/llama-3-70b-instruct) 降价 3.5%。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemma-2-9b-it>)">Google: Gemma 2 9B by google</a>: Google 的 Gemma 2 9B 是一款先进的开源语言模型，为其尺寸级别的效率和性能树立了新标准。它专为各种任务而设计，助力开发者...</li><li><a href="https://openrouter.ai/models/cognitivecomputations/dolphin-mixtral-8x22b>)">Dolphin 2.9.2 Mixtral 8x22B 🐬 by cognitivecomputations</a>: Dolphin 2.9 专为指令遵循、对话和编程而设计。该模型是 [Mixtral 8x22B Instruct](/models/mistralai/mixtral-8x22b-instruct) 的微调版本。它具有 64k 上下文...</li><li><a href="https://openrouter.ai/models/openchat/openchat-8b>)">OpenChat 3.6 8B by openchat</a>: OpenChat 8B 是一个开源语言模型库，使用 “C-RLFT (Conditioned Reinforcement Learning Fine-Tuning)” 进行微调——这是一种受离线强化学习启发的策略。它...</li><li><a href="https://openrouter.ai/models/gryphe/mythomax-l2-13b>)">MythoMax 13B by gryphe</a>: Llama 2 13B 性能最高且最受欢迎的微调版本之一，具有丰富的描述和角色扮演能力。#merge</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-70b-instruct>)">Meta: Llama 3 70B (Base) by meta-llama</a>: Meta 最新的模型系列 (Llama 3) 推出了多种尺寸和版本。这是基础的 70B 预训练版本。与领先的封闭模型相比，它展示了强大的性能...</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-72b-instruct>)">Qwen 2 72B Instruct by qwen</a>: Qwen2 72B 是一款基于 Transformer 的模型，在语言理解、多语言能力、编程、数学和推理方面表现出色。它具有 SwiGLU 激活、attention QKV bias 和 gro...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1256045942189330462)** (57 messages🔥🔥): 

- **OpenRouter 审查严格程度讨论**：成员们将 OpenRouter 的自我审查与 AWS 和 Anthropic 进行了比较，认为其审查更为严格。一位用户提到：“两者在没有 prefill 的情况下都会拒绝，但在有基础 prefill 的情况下会开始编写。”

- **Opus 可用性问题**：一位用户指出，目前在没有企业支持的情况下无法启用 Opus。他们链接了一篇讨论此限制的 [Reddit 帖子](https://www.reddit.com/r/aws/comments/1cy1hce/claude_opus_shows_as_unavailable_in_us_west_2/l68rawl/)。

- **GitHub 身份验证故障排除**：成员们分享了在不重复输入密码短语的情况下进行 GitHub push 的解决方案，推荐使用 `ssh-add -A` 等工具并将命令添加到 `~/.bash_profile`。一篇 [SuperUser 帖子](https://superuser.com/a/1158050)中链接了详细指南。

- **API 差异与问题**：讨论揭示了 API 的差异，特别是 Gemini 模型产生 “Status 400” 错误的问题。强调了 Google APIs 不遵循标准格式，需要针对 tool roles 进行特定调整。

- **评估 LLM APIs**：一位成员建议观看 Simon Willison 的演讲以了解 LLM APIs 的概况，并分享了 [YouTube 链接](https://www.youtube.com/watch?v=5zE2sMka620&t=2026s)及其 [博客文章](https://simonwillison.net/2024/Jun/27/ai-worlds-fair/)的链接。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/docs/provider-routing#custom-routing">Provider Routing | OpenRouter</a>：跨多个提供商路由请求</li><li><a href="https://www.reddit.com/r/aws/comments/1cy1hce/claude_opus_shows_as_unavailable_in_us_west_2/l68rawl/>">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=5zE2sMka620&t=2026s">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen Track</a>：https://twitter.com/aidotengineer</li><li><a href="https://forum.cursor.com/">Cursor Community Forum</a>：讨论 Cursor 的地方（错误、反馈、想法等）</li><li><a href="https://superuser.com/a/1158050">macOS keeps asking my ssh passphrase since I updated to Sierra</a>：它以前能记住密码短语，但现在每次都问我。我读到需要用这个命令重新生成公钥，我已经做了：ssh-keygen -y...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[일반](https://discord.com/channels/1091220969173028894/1246338143226167349/)** (1 messages): 

voidnewbie: Gemma 2 虽然名义上只支持英语，但似乎拥有出色的多语言能力。有人试过韩语吗？
  

---


### **OpenRouter (Alex Atallah) ▷ #[tips](https://discord.com/channels/1091220969173028894/1256159780628861001/1256160868203499551)** (1 messages): 

- **明智地设置默认模型**：daun.ai 建议将默认模型设置为 'auto'，以便在大多数任务中获得可靠的输出。或者，使用 'flavor of the week' 以获得更多意外惊喜的结果，如果没有选择特定模型且请求失败，它将作为备选模型。
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1255964886056308849)** (36 messages🔥): 

- **Gege AI 威胁音乐产业**：一位成员分享了一个关于 **Gege AI** 的 [Reddit 链接](https://www.reddit.com/r/singularity/comments/1dpxocg/gege_ai_a_new_music_maker_ai_that_allows_you_to/)，这是一款只需少量样本即可克隆任何歌手声音的 AI。他们幽默地评论道：“音乐产业安息吧”，并建议 **RIAA** 应该起诉中国。

- **Gege AI 注册挑战**：用户讨论了在注册 **Gege AI** 时遇到的问题。有人开玩笑说这与“社会信用分不足”有关。

- **Gemma 27B 模型令人印象深刻但也引发质疑**：一位成员声称 **Gemma 27B** 表现良好，但其他人对其真实能力表示怀疑。他们指出，尽管置信区间很高，其表现似乎仍优于其前代产品。

- **对 GPT-4 和 4O 模型的抱怨**：多位用户提到了 **GPT-4 和 4O 模型** 的问题，指出它们往往过于字面地理解 prompts，且在编程方面不如 **GPT-3.5** 有效。一位用户在与 **Gemini 1.5 Pro** 比较时表示：“免费替代方案占据上风”。

- **切换到 Claude 以获得更好体验**：由于更好的 artifacts 功能以及与 **Hugging Face** 库的兼容性，一些用户已从 OpenAI 的模型切换到 **Claude**。他们报告了比 **GPT-4** 模型更好的体验。

**提及的链接**：<a href="https://www.reddit.com/r/singularity/comments/1dpxocg/gege_ai_a_new_music_maker_ai_that_allows_you_to/">Reddit - Dive into anything</a>：未找到描述

  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1256017421429510295)** (2 条消息): 

- **Adam-mini 优化器在不牺牲性能的前提下大幅降低内存占用**：一种名为 **Adam-mini** 的新型优化器可以在内存占用减少 45% 到 50% 的情况下，实现与 **AdamW** 相当或更好的性能。论文指出，Adam 中的大部分学习率资源（特别是 $1/\\sqrt{v}$）可以通过将参数划分为块并为每个块分配单一的、经过优化的学习率来移除，最终在某些情况下表现优于 Adam。

- **为权重张量设置单一学习率以消除冗余**：这种为每个权重张量使用一个预搜索学习率的创新方法显示出比 Adam 显著的性能提升。“每个权重张量一个预搜索的学习率显著优于 Adam”，强调了精细的资源分配和优化如何提高效率。

**提到的链接**：<a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>：我们提出了 Adam-mini，这是一种优化器，它在内存占用减少 45% 到 50% 的情况下，实现了与 AdamW 持平或更好的性能。Adam-mini 通过削减学习率资源来减少内存……

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1256002857438810223)** (26 条消息🔥): 

- **Bedrock 上的 CSV 和 Pandas DataFrame Agent 问题**：一名成员在 **Bedrock** 上构建和运行 **csv_agent** 或 **pandas_dataframe_agent** 时遇到问题。他们向社区寻求故障排除帮助。

- **Sonnet 3.5 与 Bedrock 的错误**：另一名成员在使用 `ChatPromptTemplate.fromMessages` 将 **Sonnet 3.5 模型**与 **Bedrock** 集成时遇到困难。他们分享了一个示例格式，并提到尽管尝试调整消息格式，但仍收到错误。

- **LangGraph 和 Human-in-the-Loop 发布**：[LangChain Blog](https://blog.langchain.dev/human-in-the-loop-with-opengpts-and-langgraph) 宣布发布 **LangGraph**，其特色是通过 "Interrupt"（中断）和 "Authorize"（授权）功能实现 Human-in-the-loop（人工干预）能力。讨论强调了在人工审批步骤后尝试恢复执行时出现的反序列化错误。

- **关于 CSV 文件处理的讨论**：一位用户讨论了使用 **LangChain** 的 **CSV Loader**，并表示难以有效处理多个 CSV 文件。他们分享了一个[文档链接](https://python.langchain.com/v0.2/docs/how_to/document_loader_csv/)并寻求社区关于更好方法的建议。

- **Python 版 Human-in-the-Loop 示例**：分享了一个详细的示例和[在 Python 中实现 Human-in-the-loop 的指南](https://python.langchain.com/v0.2/docs/how_to/tools_human/#adding-human-approval)链接。这包括在工具调用步骤中请求人工审批以及处理工具调用接受或拒绝的机制。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.langchain.dev/human-in-the-loop-with-opengpts-and-langgraph/?">Human-in-the-loop with OpenGPTs and LangGraph</a>：TLDR；今天我们在 OpenGPTs 中推出了两个“Human-in-the-loop”功能：Interrupt 和 Authorize，均由 LangGraph 提供支持。我们最近推出了 LangGraph，这是一个帮助开发者构建……的库。</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/tools_human/#adding-human-approval>).">How to add a human-in-the-loop for tools | 🦜️🔗 LangChain</a>：有些工具我们不信任模型能独立执行。在这种情况下，我们可以在调用工具之前要求人工审批。</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/document_loader_csv/">How to load CSVs | 🦜️🔗 LangChain</a>：逗号分隔值 (CSV) 文件是一种使用逗号分隔值的定界文本文件。文件的每一行都是一条数据记录。每条记录由一个或多个字段组成，由逗号分隔……
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1255977451998416998)** (8 条消息🔥): 

- **针对 LangChain 的无代码 Chrome Extension**：一名成员分享了一个名为 [YouTube 视频](https://www.youtube.com/watch?v=-OKC7CY2bbQ) *"使用 Visual LangChain 的无代码 Chrome Extension 聊天机器人"*。该视频演示了如何设计一个具有交互式聊天功能的 LangChain RAG 应用。

- **Dappier 推出 AI 内容市场**：一个名为 [Dappier](https://www.producthunt.com/posts/dappier-2-0) 的新平台旨在将用于 AI 使用和训练的专有内容变现。[TechCrunch 的文章](https://techcrunch.com/2024/06/26/dappier-is-building-a-marketplace-for-publishers-to-sell-their-content-to-llm-builders/) 对其进行了报道，该平台允许创作者通过 RAG API 授权数据模型。

- **使用 Cohere 和 LangChain 的 Data Analyst Agent**：一名成员利用 Cohere 和 LangChain 构建了一个 [Data Analyst Agent](https://www.linkedin.com/posts/eddieotudor_datascience-aitechnology-machinelearning-activity-7212491482287542272-1cSF)，并在 LinkedIn 上分享了该项目。

- **Testcontainers 新增 Ollama 支持**：[testcontainers-python](https://github.com/testcontainers/testcontainers-python/pull/618) 的一个新 PR 已被接受，增加了对 Ollama 模块的支持。鼓励用户尝试在 4.7.0 版本中发布的功能。

- **JSONL 数据集编辑工具**：分享了一个用于编辑 JSONL 格式的微调和聊天数据集的免费工具：[uncensored.com/jsonl](https://uncensored.com/jsonl)。创作者强调了手动编辑 JSONL 数据集的繁琐。

- **使用 Matryoshka Embeddings 构建 RAG**：一名成员分享了关于使用 Matryoshka Embeddings 和 Llama Index [构建 RAG](https://x.com/Prashant_Dixit0/status/1806580075447590974) 的细节。其优势包括提高检索速度和减少内存占用，并提供了一个 [Colab 教程](https://colab.research.google.com/github/lancedb/vectordb-recipes/blob/main/tutorials/RAG-with_MatryoshkaEmbed-Llamaindex/RAG_with_MatryoshkaEmbedding_and_Llamaindex.ipynb)。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=-OKC7CY2bbQ">No Code Chrome Extension Chat Bot Using Visual LangChain</a>：在此演示中，我展示了 Visual Agents 的一个令人兴奋的新功能，你可以设计你的 LangChain RAG 应用，包括一个交互式聊天功能...</li><li><a href="https://x.com/Prashant_Dixit0/status/1806580075447590974">Prashant Dixit (@Prashant_Dixit0) 的推文</a>：使用 @llama_index 构建 Matryoshka RAG。这些嵌入模型产生一系列嵌入维度（768, 512, 256, 128 和 64）。🌟 优势：✅ 提升检索速度性能 ✅ 减少内存...</li><li><a href="https://uncensored.com/jsonl">Chat Uncensored AI | 获评 2024 年最佳无限制 AI</a>：新内容：现在支持无限制图像！最新且最先进的无限制 AI (2024)。无需登录，100% 私密，极速体验。全球超过 10,000 名用户信赖。</li><li><a href="https://www.producthunt.com/posts/dappier-2-0"> Dappier 2.0 - 对抗 AI 数据抓取并公平地获取内容报酬 | Product Hunt</a>：Dappier 是全球首个 AI 内容和数据权利在线市场。当你的授权内容被全球 AI 公司访问时，获得公平的报酬。</li><li><a href="https://github.com/testcontainers/testcontainers-python/pull/618">fix: Add support for ollama module by bricefotzo · Pull Request #618 · testcontainers/testcontainers-python</a>：添加了一个新的 OllamaContainer 类，其中包含一些处理 Ollama 容器的方法。_check_and_add_gpu_capabilities 方法检查主机是否有 GPU 并添加必要的权能...</li><li><a href="https://github.com/testcontainers/testcontainers-python/issues/617#issuecomment-2194351846">New Container: OllamaContainer · Issue #617 · testcontainers/testcontainers-python</a>：添加对 OllamaContainer 的支持，以简化通过 Ollama 运行和测试 LLM。你想要什么样的新容器？我想请求支持一个新容器：OllamaCo...</li><li><a href="https://techcrunch.com/2024/06/26/dappier-is-building-a-marketplace-for-publishers-to-sell-their-content-to-llm-builders/">Dappier 正在为出版商构建一个向 LLM 构建者出售内容的市场 | TechCrunch</a>：Dappier 是一家初创公司，正在构建一个市场，出版商和数据所有者可以在其中向 LLM 构建者出售其内容的访问权限。
</li>
</ul>

</div>

---

### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1256296697475371121)** (1 messages): 

- **Testcontainers Python SDK 增加 Ollama 支持**：一位用户宣布，他们为 Testcontainers Python SDK 增加 Ollama 支持的 [pull request](https://github.com/testcontainers/testcontainers-python/pull/618) 已被接受并在 4.7.0 版本中发布。他们还提供了一个 [示例](https://github.com/testcontainers/testcontainers-python/issues/617#issuecomment-2194351846) 以帮助其他人快速上手。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/testcontainers/testcontainers-python/pull/618">fix: Add support for ollama module by bricefotzo · Pull Request #618 · testcontainers/testcontainers-python</a>: 添加了一个新的 OllamaContainer 类，包含几个处理 Ollama 容器的方法。_check_and_add_gpu_capabilities 方法会检查主机是否具有 GPU，并向其添加必要的能力...</li><li><a href="https://github.com/testcontainers/testcontainers-python/issues/617#issuecomment-2194351846">New Container: OllamaContainer · Issue #617 · testcontainers/testcontainers-python</a>: 添加对 OllamaContainer 的支持，以简化通过 Ollama 运行和测试 LLM 的过程。你想要什么样的新容器？我想请求支持一个新容器：OllamaCo...
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1255984799752126574)** (2 messages): 

- **下一次 Mojo 社区会议已安排**：下一次 Mojo 社区会议将于 [当地时间] 举行。详情请通过 [Zoom](https://modul.ar/community-meeting-zoom) 参加会议，并在 [Google Docs](https://modul.ar/community-meeting-doc) 上查看议程。 
- **节日祝福**：祝加拿大用户以及美国 7 月 4 日当周休假的用户节日快乐！
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://modul.ar/community-meeting-zoom.">加入我们的 Cloud HD 视频会议</a>: Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可跨移动设备、桌面和会议室系统进行视频和音频会议、聊天和网络研讨会。Zoom ...</li><li><a href="https://modul.ar/community-meeting-doc.">[公开] Mojo 社区会议</a>: Mojo 社区会议文档链接：https://modul.ar/community-meeting-doc 这是一个公开文档；欢迎所有人查看和评论/建议。所有会议参与者必须遵守...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1806718451089817703>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1256053477873025024)** (11 messages🔥): 

- **关于 Mojolicious 和 Mojo 的混淆**：一位用户请求一个 Mojo 代码示例，当 ModularBot 提供了一个基于 Perl 的 Mojolicious 示例时引发了混淆。另一位成员澄清说，该咨询专门针对 Mojo，即 Modular 在 2022 年创建的 AI 开发语言。 
- **Mojo 的澄清**：在进一步追问后，ModularBot 承认了错误并讨论了 Mojo 的能力，将其比作 *“一位冒险进入未知领域的骑士”*，其增强能力类似于 Python 但像 C 一样健壮。

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1255961095089291336)** (12 messages🔥): 

- **Mojo SDK 和遥测 (Telemetry) 问题**：成员们讨论了 Mojo SDK 的遥测数据收集，其中一位指出可以将其禁用，并分享了 [FAQ 链接](https://docs.modular.com/mojo/faq#does-the-mojo-sdk-collect-telemetry) 以获取更多信息。另一位对其提供的有用信息表示感谢。
- **Mojo REPL 中的连接问题**：一位成员观察到运行 REPL 会开启一个连接但不显示网络流量，随后该连接会意外关闭。他们确认了该问题并建议提交 GitHub issue 以进行进一步调查。
- **关于 Mojo 包列表的讨论**：一位成员运行了列出 Mojo 包的命令，显示了 mojo 24.4.0 版本的包详情。这引发了关于其配置和设置的讨论。 
- **Mojo 语言设计选择**：一位成员注意到 Io 模块缺乏功能，需要通过 Python 接口来读取 stdin。他们思考这是刻意为之，还是 Modular 会接受扩展该功能的贡献。 



**提到的链接**: <a href="https://docs.modular.com/mojo/faq#does-the-mojo-sdk-collect-telemetry">Mojo🔥 FAQ | Modular Docs</a>: 关于 Mojo 预期问题的解答。

  

---

### **Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - 第 38 期
https://www.modular.com/newsletters/modverse-weekly-38
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1255983829693497394)** (4 messages): 

- **更新日志中遗漏的 Graph API 改进**：最新版本包含了 *“整数面值切片以及跨所有维度的切片”*，并带有一些语义限制。团队强调了提交 issue 以进行跟踪以及处理外部对新特性关注的重要性。
- **unsqueeze 操作的临时解决方案**：对于需要 *“unsqueeze”* 的用户，建议使用 `ops.unsqueeze(x, axis=-1)` 作为权宜之计。
- **MAX nightly 版本重新上线**：MAX nightly 版本已恢复正常，鼓励用户通过提供的 [Discord 链接](https://discord.com/channels/1087530497313357884/1256010477637730315/1256010477637730315) 演示 *Llama3 GUI Chatbot* 并分享反馈。
- **发布了新的 Mojo nightly 编译器**：发布了新的 nightly 编译器版本 `2024.6.2805`，显著更新包括 LSP 行为的变化。用户需使用 `modular update nightly/mojo` 进行更新，并可以查看 [原始差异 (raw diff)](https://github.com/modularml/mojo/compare/ddaa1d0a0979998a96084e3d7bd335fbcda3e8cb...439d86d608d3b6c12cead112eb651752ba1ad40d) 和当前的 [更新日志 (changelog)](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1255987610577403954)** (30 messages🔥): 

- **欢迎社区对模型做出贡献**：*“这已在我们的计划中，但如果有人想立即使用该模型，我们也欢迎社区贡献该模型。”*
- **文本补全中与 EOS token 相关的异常行为**：一位用户发现续写中出现了 *“超级奇怪的行为”*，这是由于 *“数据集在编码时添加了 eos token”* 导致的，详见此 [GitHub 链接](https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_text_completion.py#L56)。
- **PPO 实现推荐使用 PreferenceDataset**：在讨论数据集配置时，一位用户推荐在 RL 中使用 **PreferenceDataset**，其中 *“奖励模型观察整个 输入+响应 的‘偏好度’”*。这与用于单一文本体持续预训练的 **text completion dataset** 形成对比。
- **澄清了关于预训练示例的困惑**：讨论澄清了预训练的输入和输出，强调预训练涉及整个文档，模型在其中预测 token 并惩罚错误预测，而不是处理分段的输入-输出对。
- **为文本补全数据集添加 EOS token 选项被认为是合理的**：用户讨论了在文本补全数据集中添加 **add_eos** 选项是否有意义，结论是这是一个实用的想法，并帮助解决了一个 PPO 实现问题。

**提到的链接**：<a href="https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_text_completion.py#L56).">torchtune/torchtune/datasets/_text_completion.py at main · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。欢迎通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1256172869994545183)** (6 messages): 

- **查看 evals 工作坊**：一位成员分享了一个关于 evals 工作坊讨论的链接，暗示了其重要性。他们还分享了 Positron 的 [GitHub 链接](https://github.com/posit-dev/positron)，这是一个下一代数据科学 IDE。

- **寻找 JSONL 数据编辑器**：两位成员表示对一种能在同一界面中直接遍历和编辑 JSONL 文件示例的工具感兴趣。其中一人提到尝试过 Lilac，它几乎满足需求但缺乏直接编辑功能。

- **总结患者记录**：一位成员正在寻找工具或论文，以便从 JSON 格式的患者记录中生成结构化摘要，并指出需要与文本到文本摘要不同的方法。他们正在测试 Llama 模型以避免幻觉 (hallucinations)，并寻求关于提示工程 (prompt engineering) 和微调 (fine-tuning) 技术的建议。

**提到的链接**：<a href="https://github.com/posit-dev/positron">GitHub - posit-dev/positron: Positron, a next-generation data science IDE</a>：Positron，一个下一代数据科学 IDE。欢迎通过在 GitHub 上创建账号来为 posit-dev/positron 的开发做出贡献。

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1255989194275295242)** (4 messages): 

- **LLAMA 在 Streamlit 上报错**：一名成员在将 **LLAMA** 部署到 Streamlit 用于 RAG 应用后寻求错误帮助。他们提到该问题在本地未出现，仅在部署环境中产生。
- **账户额度缺失**：一名成员请求协助解决尚未收到额度的问题，并提供了用户名和电子邮件以便后续跟进。
- **Tinyllama 自定义数据集路径错误已解决**：最初，一名成员在通过自定义数据集微调 **Tinyllama** 时遇到了 `FileNotFoundError`。他们随后通过正确设置 `path: my_test.jsonl` 且不包含 `data/` 目录解决了该问题。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1256256276900483132)** (2 messages): 

- **链接失效问题已解决**：一位用户提到在课程期间分享的一个链接已失效并请求更新。该问题很快得到另一位用户的确认并修复。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1256121831342215219)** (2 messages): 

- **Kishore 请求额度协助**：Kishore 报告未收到额度并寻求帮助，提供了他的标识符 `kishore-pv-reddy-ddc589`。

- **Christopher 为 fireworks 寻求额度**：Christopher 同样请求了 fireworks 的额度，并附上了他的标识符 `christopher-438388`。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1255978316498735167)** (2 messages): 

- **你有新消息！**：一位用户在 Discord 频道通过 DM 提醒另一名成员查看消息。他们使用了祈求表情 🙏 来强调紧迫性。
- **Predibase 额度过期咨询**：一名成员询问 **Predibase 额度是否会在 7 月 4 日过期**。在可见的消息记录中没有回复。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1256146343139737682)** (1 messages): 

- **用户等待 OpenAI 额度**：一位用户发帖称尚未收到他们的 OpenAI 额度。他们提供了组织 ID **org-NBiOyOKBCHTZBTdXBIyjNRy5** 以及相关的电子邮件地址（**karthikv2k@gmail.com** 和 **karthik@v2k.ai**）。
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1256009698508013698)** (14 messages🔥): 

- **Open Interpreter 通过公开讨论优先考虑安全性**：一名成员对 Open Interpreter 的安全风险表示担忧，引发了关于现有安全措施的详细回应，例如代码执行前的用户确认以及使用 Docker 进行沙箱隔离。对话强调了透明度和社区参与对确保项目安全的重要性。
- **代码模型性能对比**：成员们讨论了各种代码模型的性能，指出 **Codestral** 表现最佳，而 **DeepSeek Coder** 速度明显更快，但性能约为前者的 70%。
- **DeepSeek Coder-v2-lite 因速度和代码能力受赞赏**：一位成员表达了对 **DeepSeek Coder-v2-lite** 的偏好，因其快速的性能和编码效率，并认为它可能优于 **Qwen-1.5b**。
- **量化模型支持咨询**：有人咨询由于 RAM 限制，是否能以量化形式运行用于图像理解的 SMOL 多模态模型，强调了在资源受限环境中对效率的需求。
- **YouTube 视频揭露 Rabbit R1 安全漏洞**：分享了一段题为 ["Rabbit R1 犯了灾难性的新手编程错误"](https://youtu.be/lkbV8oP-F44) 的 YouTube 视频，揭示了 Rabbit R1 的代码库包含硬编码的 API keys，危及用户数据安全。

**提到的链接**：<a href="https://youtu.be/lkbV8oP-F44">Rabbit R1 犯了灾难性的新手编程错误</a>：一组越狱者最近发现 Rabbit R1 代码库包含硬编码的 API keys —— 这让他们可以轻松访问其 AI 技术中的用户数据……

  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1256255566645563393)** (3 messages): 

- **在本地运行带有修改的 OpenInterpreter**：一位成员解释说，在本地使用非 OpenAI 提供商运行 OpenInterpreter 需要进行一些更改。他们在 [GitHub issue 评论](https://github.com/OpenInterpreter/01/issues/272#issuecomment-2119175075)中详细说明了必要的差异。

- **API 默认使用 OpenAI**：据指出，系统默认可能使用 OpenAI 的 API，可能是 GPT-4 Turbo。然而，由于有一段时间没有审查，具体细节尚未确认。

- **对额外 API 费用的担忧**：另一位成员对使用 API 时产生的额外费用表示担忧，这些费用与订阅费用是分开的。

**提到的链接**：<a href="https://github.com/OpenInterpreter/01/issues/272#issuecomment-2119175075">Litellm/01 is unable to connect to non-openAI providers. · Issue #272 · OpenInterpreter/01</a>：导致问题的原因：运行 01 并指定任何非 OpenAI 服务器主机和 API Key。预期结果：能够连接到 Groq, Anthropic, OpenRouter 等其他服务，因为它们似乎可以与 b... 配合使用。

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1256088170605314089)** (7 messages): 

- **完成的移植支持微调 (finetuning)**：一位成员宣布完成了一个移植版本，并提到它现在也支持微调。这标志着他们项目的进展和潜在的新功能。
- **用于节能机器人的基于 FPGA 系统**：另一位成员详细介绍了他们为期 8 个月的项目，重点是节能的人形机器人。他们强调了使用基于 FPGA 的系统来实现大 DRAM 空间和不错的推理速度的成本效益和逻辑方法。
- **人形机器人在 GenAI 上的电池消耗**：同一位成员指出，人形机器人目前在运行 GenAI 时消耗大量电池电量，考虑到每个机器人使用 3-4 个基于 GPU 的 SOM，这是低效的。他们暗示目前的方案是不可持续的。
- **JSON/YAML 在 tinygrad 中的效用**：一位用户建议让 tinygrad 能够从 JSON/YAML 文件中读取模型，认为这可以简化配置。另一位成员回应说，模型已经以 dict 形式保存和加载。
- **当前模型存储机制**：George Hotz 澄清说，在 tinygrad 中，**safetensors** 用于权重，**pickle** 用于计算。这突出了该项目当前的模型存储方法。
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1255977369727140013)** (4 messages): 

- **Shapetracker 实现零成本 Tensor 重塑 (reshaping)**：成员们讨论了 [博客文章](https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html) 中解释的 Shapetracker 功能。一位成员指出：“如果你必须重塑一个巨大的 Tensor，内存中的底层数据不需要改变，只需要改变你访问它的方式。”
- **关于掩码求解 (mask solving) 的问题**：在 Shapetracker 的背景下，一位成员询问了掩码解决了什么问题的澄清。他们将此查询与理解内存表示中的 shape 和 strides 联系起来。
- **对 Shapetracker 起源的好奇**：一位成员对 Shapetracker 背后的逻辑是从头开始发明的还是受到其他深度学习编译器的启发表示好奇。他们对其复杂程度感到惊讶：“大多数框架使用 strides 进行优化，但 Shapetracker 允许任意的 movement ops，且完全不需要副本。”

**提到的链接**：<a href="https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html">How ShapeTracker works</a>：tinygrad 教程

  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1255987719281311886)** (7 messages): 

- **Cohere 的实习机会引起关注**：一位正在研究 **LLM 和强化学习 (Reinforcement Learning)** 的学生寻求对 Cohere 工作和文化的见解，并请求现任员工私信。另一位成员指出，在主要的 AI 公司获得实习机会非常困难，强调了拥有实质性公开作品集的必要性。

- **Cohere 的愿望清单**：一位成员询问是否有人们希望添加到 **Cohere** 的功能。

- **用于博客的 AI 自动化**：一位成员寻求帮助，以设置 **AI 驱动的自动化**来创建博客并发布到社交平台。他们被引导到另一个频道寻求帮助。

- **展示使用 Cohere 和 Langchain 构建的 AI**：一位成员分享了他们的 [LinkedIn 帖子](https://www.linkedin.com/posts/eddieotudor_datascience-aitechnology-machinelearning-activity-7212491482287542272-1cSF?utm_source=share&utm_medium=member_ios)，内容是关于使用 **Cohere 和 Langchain** 创建 **Data Analyst Agent**。

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1255974663784239184)** (4 messages): 

- **支持带有 Sample Packing 的 Gemma2**：一位成员分享了一个支持 Gemma2 及其 Sample Packing 的 [GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718)。他们正在等待 PR 中链接的上游 Hugging Face 修复。
- **27b 模型在 Benchmarks 中表现平平**：一位用户提到，与 9b 模型相比，27b 模型在 Benchmarks 中的表现出人意料地平庸，暗示了较大模型存在性能问题。

**提到的链接**：<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718">support for gemma2 w sample packing by winglian · Pull Request #1718 · OpenAccess-AI-Collective/axolotl</a>：描述、动机与背景、测试方式、屏幕截图（如适用）、变更类型、社交账号（可选）。

  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1256088661678886912)** (3 messages): 

- **Featherless.ai 推出模型访问平台**：最近，**Featherless.ai** 推出了一个新平台，只需每月 10 美元起的固定订阅费用，即可访问 Hugging Face 上的 450 多个模型。该[平台](https://featherless.ai/)的特点包括无需 GPU 设置/下载、兼容 OpenAI 的 API 访问，以及每周新增模型和具有竞争力的价格分级。
- **订阅计划详情**：**Featherless.ai** 提供两个订阅层级：Feather Basic 每月 10 美元，支持最高 15B 的模型；Feather Premium 每月 25 美元，支持最高 72B 的模型。两个计划都提供无限的个人使用权，Feather Premium 将权益扩展到更大的模型尺寸，并提供私密、安全且匿名的使用体验。
- **征求模型优先级的反馈**：社区正在征求关于应优先将哪些模型添加到 **Featherless** 平台的反馈。该平台的早期采用者主要将其用于 AI 角色本地应用以及更具体的用途，如语言 finetuning 和 SQL 模型。

**提到的链接**：<a href="https://featherless.ai/"> Featherless - Serverless LLM</a>：Featherless - 最新的 LLM 模型，Serverless 架构，随取随用。

  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1256304289484378204)** (3 messages): 

- **请求历史 Elo 数据**：一位用户询问是否有“Chatbot Arena Elo 分数的历史数据集或视图”。他们注意到可用的 JSON 数据仅涵盖约六周的时间。
- **梯队追赶**：提到的时间线从 5 月 19 日开始，观察到“第一梯队（the pack）”正在顶部追赶。
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1256019429360140381)** (1 messages): 

- **企业级 Feature Store 网络研讨会**：即将举行一场名为“使用 Featureform 和 Databricks 构建企业级 Feature Store”的网络研讨会，时间为太平洋时间 7 月 23 日星期二上午 8 点。会议将由 Simba Khadder 主持，他将讨论简化 Feature Engineering、利用 Databricks 以及管理大规模数据的最佳实践，随后是问答环节。[在此注册](https://buff.ly/3zh3B74)。

**提到的链接**：<a href="https://buff.ly/3zh3B74">Building an Enterprise-Scale Feature Store with Featureform and Databricks</a>：加入我们与 Featureform 创始人的 1 小时网络研讨会，了解如何通过使用 Featureform 和 Databricks 为您的数据赋能！

  

---



---



---



---



---



{% else %}


> 完整的频道细分内容已在邮件中截断。
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}