---
companies:
- reka-ai
- cohere
- google
- rewind
- apple
- mistral-ai
- microsoft
- paypal
date: '2024-04-15T22:42:55.173152Z'
description: '在 4 月 12 日至 15 日期间，AI 领域发生了以下动态：


  **Reka Core** 发布了一款新型 GPT-4 级多模态基础模型，并附带了一份被描述为“全 Shazeer (full Shazeer)”的详细技术报告。**Cohere
  Compass** 推出了一款基础嵌入模型，用于对电子邮件和发票等这类多维度的企业数据进行索引和搜索。开源模型 **IDEFICS 2-8B** 继续对谷歌 Flamingo
  多模态模型进行复现。


  **Rewind** 转型为名为 Limitless 的多平台应用，试图摆脱“间谍软件”的负面印象。Reddit 上的讨论热点包括：**Apple MLX**
  在 M2 Ultra GPU 上的表现优于 **Ollama** 和 **Mistral Instruct**；适用于大语言模型（LLM）和 Stable Diffusion
  的 GPU 选择；以及微软研究院 Chris Bishop 对人工智能与人类的对比。


  此外，PayPal 前首席执行官 Dan Schulman 预测，**GPT-5** 将使工作范畴大幅缩减 80%。**Mistral** 首席执行官 Arthur
  Mensch 则批评了对通用人工智能（AGI）的痴迷，将其比作“创造上帝”。'
id: b138e365-95ff-4f55-8811-5a76a5e17422
models:
- gpt-4
- idefics-2-8b
- mistral-instruct
- apple-mlx
- gpt-5
original_slug: ainews-multi-modal-multi-aspect-multi-form-factor
people:
- arthur-mensch
- dan-schulman
- chris-bishop
title: '**多模态、多维度、多形态 AI**'
topics:
- multimodality
- foundation-models
- embedding-models
- gpu-performance
- model-comparison
- enterprise-data
- open-source
- performance-optimization
- job-impact
- agi-criticism
- technical-report
---

 


---

**目录**

[TOC] 


---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。评论抓取功能现已上线，但仍有很大改进空间！

**AI 模型与性能**

- **Apple MLX 性能**：在 /r/LocalLLaMA 中，Apple MLX (0.10.0) 在 [**配备 76 核 GPU 的 M2 Ultra 上达到了 105.5 tokens/s**](https://www.reddit.com/r/LocalLLaMA/comments/1c3uzu6/apple_mlx_on_m2_ultra_76gpu_105_tokenss_with/)，在使用 Mistral Instruct 4bit 时超越了 Ollama/Llama.cpp 的 95.1 tokens/s。
- **Ollama 性能对比**：在 /r/LocalLLaMA 中，使用 Mistral Instruct 0.2 q4_0 的 Ollama 性能显示，[**M2 Ultra 76GPU 以 95.1 t/s 领先**](https://www.reddit.com/r/LocalLLaMA/comments/1c3v3q6/ollama_performance_on_m2_ultra_m3_max_windows/)，随后是 Windows Nvidia 3090 (89.6 t/s)、WSL2 NVidia 3090 (86.1 t/s) 和 M3 Max 40GPU (67.5 t/s)。Apple MLX 在 M2 Ultra 上达到 103.2 t/s，在 M3 Max 上达到 76.2 t/s。
- **M3 Max vs M1 Max Prompt 速度**：在 /r/LocalLLaMA 中，当处理长上下文时，使用 Command-R 和 Dolphin-Mixtral 8x7B 模型，[**M3 Max 64GB 的 Prompt 速度比 M1 Max 64GB 快一倍以上**](https://www.reddit.com/r/LocalLLaMA/comments/1c3t538/comparing_m1_max_64gb_vs_m3_max_64gb_prompt_speed/)。
- **LLM 的 GPU 考量**：在 /r/LocalLLaMA 中，一位用户正在[寻求关于围绕 RTX 4090 组装机器还是购买 MAC 的建议](https://www.reddit.com/r/LocalLLaMA/comments/1c3qfg7/rtx_4090_vs_mac/)，以便运行 Command R 和 Mixtral 模型，并考虑未来的可升级性。
- **用于 Stable Diffusion 的 GPU**：在 /r/StableDiffusion 中，一项关于 [3060 12GB vs 4060 16GB 用于 Stable Diffusion](https://www.reddit.com/r/StableDiffusion/comments/1c3nk9g/3060_12gb_vs_4060_16gb_for_sdxl_or_wait_for_50xx/) 的对比建议选择在经济承受范围内尽可能大的 VRAM。4060ti 16GB 运行 SDXL 20 步需要 18.8 秒。


**LLM 与 AI 发展**

- **AI 与人类的对比**：在 /r/singularity 中，Microsoft Research 的 Chris Bishop 将只会机械重复信息的 AI 模型比作“随机鹦鹉”，并指出[**做同样事情的人类却被授予了大学学位**](https://www.reddit.com/r/singularity/comments/1c3o1o2/microsoft_researchs_chris_bishop_when_ai_models/)。评论区讨论了学位的有效性，以及学位是否代表了不仅仅是信息的机械重复。
- **对就业的影响**：前 PayPal CEO Dan Schulman 预测，由于 AI 的发展，[**GPT-5 将带来一个“恐慌时刻”，80% 的工作岗位将在职责范围上缩减 80%**](https://twitter.com/woloski/status/1778783006389416050)。
- **对 AGI 的痴迷**：Mistral 的 CEO Arthur Mensch 认为[对实现 AGI 的痴迷本质上是关于“创造上帝”](https://www.businessinsider.com/mistrals-ceo-said-obsession-with-agi-about-creating-god-2024-4)。Gary Marcus 也在一条推文中[敦促不要创造有意识的 AI](https://www.reddit.com/r/singularity/comments/1c3vbkz/people_have_happily_worked_so_hard_to_build_stuff/)。
- **为未来而建设**：Sam Altman 发推文称，[人们正努力开发技术，以便后代能够继续推进](https://www.reddit.com/r/singularity/comments/1c3vbkz/people_have_happily_worked_so_hard_to_build_stuff/)，而不期望能见到这些成果的受益者。
- **直面意识问题**：/r/singularity 的一个帖子认为，[实现 AGI 将迫使人类直面自己对意识产生机制的缺乏理解](https://www.reddit.com/r/singularity/comments/1c3yvs1/agi_will_cause_humans_to_confront_that_they_do/)，并预测了围绕 AI 伦理和权利的辩论与紧张局势。

**行业与职业**

- **识别“虚假”的 ML 职位**：在 /r/MachineLearning 中，一位博士生在入职了一个不涉及实际 ML 工作的工作岗位后，[寻求关于识别“虚假” ML 职位的建议](https://www.reddit.com/r/MachineLearning/comments/1c3z8ug/d_advice_for_spotting_fake_ml_roles/)，并指出由于潜在的不诚实行为，在面试中提问可能并不奏效。
- **传统 NLP 任务的相关性**：另一位博士生[质疑在 LLM 时代，传统的 NLP 任务（如文本分类、NER 和 RE）是否依然重要](https://www.reddit.com/r/MachineLearning/comments/1c4a7sa/d_are_traditional_nlp_tasks_such_as_text/)，并对自己的研究前景感到担忧。
- **LLM 的实际用途**：/r/MachineLearning 的一个帖子[征求除文本生成之外，能提供良好 ROI 的 LLM 实际工业应用案例](https://www.reddit.com/r/MachineLearning/comments/1c4cr32/d_in_industry_nlp_are_there_any_actualpractical/)，并指出像语义搜索这样的任务可以由其他模型很好地处理。

**工具与资源**

- **llama.cpp 支持 DBRX**：[llama.cpp 现在支持 DBRX](https://github.com/ggerganov/llama.cpp/pull/6515)，这是一种用于 LLM 的二进制格式。
- **更快的结构化生成**：在 /r/LocalLLaMA 中，一种新的 [LLM 结构化生成方法被声称比 llama.cpp 的方法快得多](https://www.reddit.com/r/LocalLLaMA/comments/1c3oa8f/faster_than_llamacpps_grammar_structured/)，其运行时间与语法复杂度或模型/词表大小无关。作者计划很快将其开源。
- **Python 数据分类工具**：作者[开源了用于自动化数据分类和整理的 Python 工具集](https://github.com/nazpins/naztech-automated-data-sorting-tools)，旨在高效处理杂乱的文件和海量数据。
- **简单的离散扩散实现**：/r/MachineLearning 分享了一个[开源的、仅用 400 行代码实现的简单 PyTorch 离散扩散（discrete diffusion）版本](https://www.reddit.com/r/MachineLearning/comments/1c3pvx5/p_extremely_short_and_simple_implementation_of/)。

**硬件与性能**

- **M1 Max vs M3 Max**：在 /r/LocalLLaMA 中，一项[关于 64GB RAM 的 M1 Max 与 M3 Max 提示词速度对比](https://www.reddit.com/r/LocalLLaMA/comments/1c3t538/comparing_m1_max_64gb_vs_m3_max_64gb_prompt_speed/)显示，M3 Max 的速度是前者的两倍以上，尤其是在长上下文（long contexts）情况下。
- **RTX 4090 vs Mac**：一篇帖子征求[关于组装 RTX 4090 PC 或购买 Mac 运行 LLM 的建议](https://www.reddit.com/r/LocalLLaMA/comments/1c3qfg7/rtx_4090_vs_mac/)，发帖者认为 PC 更便宜且更具可升级性。
- **M2 Ultra 上的 MLX 性能**：Apple 的 MLX 库[在拥有 76 核 GPU 的 M2 Ultra 上达到了 105.5 tokens/s](https://www.reddit.com/r/LocalLLaMA/comments/1c3uzu6/apple_mlx_on_m2_ultra_76gpu_105_tokenss_with/)，在运行 Mistral Instruct 4-bit 模型时超过了 llama.cpp 的 95.1 tokens/s。
- **Ollama 性能对比**：一项[在各种硬件上运行 Ollama 库的性能对比](https://www.reddit.com/r/LocalLLaMA/comments/1c3v3q6/ollama_performance_on_m2_ultra_m3_max_windows/)显示，运行 Mistral Instruct 模型时，M2 Ultra 以 95.1 t/s 领先，随后是 Windows Nvidia 3090、WSL2 Nvidia 3090 和 M3 Max。

**模因与幽默 (Memes and Humor)**

- 几个模因和幽默帖子获得了高赞，包括 ["The Anti-AI Manifesto"](https://www.reddit.com/r/singularity/comments/1c3o1o2/microsoft_researchs_chris_bishop_when_ai_models/)、["Maybe maybe maybe"](https://v.redd.it/i2dxn7cu5guc1)、["the singularity is being driven by an outside force"](https://i.redd.it/fkp6wr5qwiuc1.png)、["Ai women are women"](https://v.redd.it/dppfbk4uzfuc1) 以及 ["Real reason AI (Alien Intelligence) can't do hands? 👽"](https://i.redd.it/frcwgrkurjuc1.jpeg)。

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行 clustering 和 flow engineering。

**AI 模型与架构**

- **新模型发布**：[@RekaAILabs](https://twitter.com/RekaAILabs/status/1779894622334189592) 发布了 Reka Core，这是他们“迄今为止最强大且能力最强的 multimodal 语言模型”，可与 GPT-4/Opus 级别的模型竞争。[@WizardLM_AI](https://twitter.com/WizardLM_AI/status/1779899325868589372) 发布了 WizardLM-2，这是一个模型家族，包括 8x22B、70B 和 7B 变体，可与领先的闭源 LLMs 竞争。
- **架构与训练**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1779684686618628323) 指出 **transformers 是一种架构**，通常与 diffusion 结合使用。[@ylecun](https://twitter.com/ylecun/status/1779845304788955292) 表示 AI 最终将超越人类智能，但仅靠目前的 auto-regressive LLMs 是无法实现的。
- **优化与扩展**：[@karpathy](https://twitter.com/karpathy/status/1779272336186978707) 使用 C 语言优化了一个 LLM 以匹配 PyTorch 的性能，目前达到 26.2ms/iteration，并使用了 fp32 模式下的 cuBLAS 等技巧。[@_lewtun](https://twitter.com/_lewtun/status/1779804085677404583) 认为强大的 Mixtral-8x22B fine-tunes 版本将缩小与闭源模型之间的差距。

**AI 能力与基准测试**

- **Multimodal 能力**：[@RekaAILabs](https://twitter.com/RekaAILabs/status/1779894626083864873) 展示了 Reka Core 的视频理解能力，在 multimodal chat 方面超越了 Claude 3 Opus。[@DrJimFan](https://twitter.com/DrJimFan/status/1779558822543229221) 推测 Tesla FSD v13 可能会使用 language tokens 来对复杂的自动驾驶场景进行推理。
- **编程与数学**：[@OfirPress](https://twitter.com/OfirPress/status/1779195498328429045) 指出开源编程 Agent SWE-agent 在发布 10 天后已拥有 1.5k 用户。[@WizardLM_AI](https://twitter.com/WizardLM_AI/status/1779899333678387318) 使用了一个完全由 AI 驱动的 synthetic 训练系统来提升 WizardLM-2。
- **基准测试与排行榜**：[@svpino](https://twitter.com/svpino/status/1779185295541575710) 注意到 Claude 3 在 GPT-4 更新前的 17 秒内曾是 human eval 排行榜上的最佳模型。[@bindureddy](https://twitter.com/bindureddy/status/1779186163464708314) 分析了 GPT-4 的编程、数学能力和 knowledge cutoff 是其性能表现的关键原因。

**开源与 AI 民主化**

- **开源模型与数据**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1779891910871490856) 宣布了 EleutherAI 的 Pile-T5 模型，该模型使用了来自 Pile 的 2T tokens 和 Llama tokenizer。[@_philschmid](https://twitter.com/_philschmid/status/1779922877589889400) 推出了 Idefics2，这是一个参数量低于 10B 的开源 VLM，具有强大的 OCR、文档理解和视觉推理能力。
- **可访问性与成本**：[@maximelabonne](https://twitter.com/maximelabonne/status/1779801605702836454) 指出开源模型现在落后顶尖闭源模型的时间是 6-10 个月，而不是数年。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1779805019841142791) 预测这一差距将在年底前完全消除，因为对于大多数用途而言，开源模型更快、更便宜且更安全。
- **算力与工具**：[@WizardLM_AI](https://twitter.com/WizardLM_AI/status/1779899329844760771) 正在 Hugging Face 上分享 8x22B 和 7B 的 WizardLM-2 权重。[@aidangomez](https://twitter.com/aidangomez/status/1779882113573044625) 宣布了用于多维度数据搜索的 Compass embedding model 测试版。

**行业与生态系统**

- **公司扩张**：[@gdb](https://twitter.com/gdb/status/1779762694473551924) 和 [@hardmaru](https://twitter.com/hardmaru/status/1779783633961935218) 指出 OpenAI 向日本扩张是 AI 存在感的重要体现。[@adcock_brett](https://twitter.com/adcock_brett/status/1779541107577151916) 分享了加拿大在 AI 能力和基础设施方面的 24 亿美元投资。
- **新兴应用**：[@svpino](https://twitter.com/svpino/status/1779843933276672195) 利用 Langflow 的可视化界面和 Langchain，在无需代码的情况下构建了一个完整的 RAG 应用。[@llama_index](https://twitter.com/llama_index/status/1779542320133947622) 展示了如何使用 LLMs 和 knowledge graphs 来加速生物材料的发现。
- **伦理考量**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1779171248083177500) 对在 Facebook 期间发生的“政治迫害”中没有给予 @PalmerLuckey 更多支持表示遗憾。[@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1779721105407873411) 称赞一段视频是迄今为止对 AI 存在性风险（existential risk）最好的报道。

---

# AI Discord 摘要

> 摘要之摘要的总结

1. **大语言模型 (LLMs) 的进展**：各平台和机构对 LLMs 的新发布和新功能表现出极大的兴奋和讨论。关键案例包括：

- **[Pile-T5](https://blog.eleuther.ai/pile-t5/)** 来自 EleutherAI，这是一个在 2 万亿 token 上训练的 T5 模型变体，在 SuperGLUE 和 MMLU 等基准测试中表现出更强的性能。包括模型权重和脚本在内的所有资源均已在 [GitHub 上开源](https://github.com/EleutherAI/improved-t5)。

- **[WizardLM-2](https://wizardlm.github.io/WizardLM2/)** 系列发布，包含 8x22B、70B 和 7B 等模型尺寸，引发了在 OpenRouter 上部署的热潮，其中 WizardLM-2 8x22B 的表现可与 GPT-4 媲美。

- **[Reka Core](https://publications.reka.ai/reka-core-tech-report.pdf)**，来自 Reka AI 的前沿级**多模态语言模型**（multimodal language model），其技术报告分享了训练、架构和评估的细节。

2. **LLM 训练与推理的优化与技术**：广泛的讨论围绕着优化 LLM 开发的各个方面，包括：

- 通过 [Ring Attention](https://coconut-mode.com/posts/ring-attention/) 等方法进行**高效的上下文处理**，使模型能够通过使用多个设备扩展到几乎无限的上下文窗口。

- 用于减少内存占用的**模型压缩**技术，如 **LoRA**、**QLoRA** 和 **16-bit quantization**，以及来自 [Lightning AI](https://lightning.ai/pages/community/lora-insights/) 和社区实验的见解。

- **硬件加速**策略，例如使用 [tinygrad 的驱动补丁](https://github.com/tinygrad/open-gpu-kernel-modules)在 **NVIDIA 4090 GPU 上启用 P2P 支持**，从而获得显著的性能提升。

- [LLM.c](https://github.com/karpathy/llm.c) 和 [torchao](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py) 等框架中的**内核优化**，探索用于矩阵运算的高效张量布局、padding 和 swizzling。

3. **开源倡议与社区协作**：AI 社区展现了对开源开发和知识共享的坚定承诺，具体体现在：

- [Pile-T5](https://github.com/EleutherAI/improved-t5)、[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) 和 [Mixtral](https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/io.mojo#L362)（Mojo 版本）等重大项目的**开源**，促进了透明度和协作努力。

- [CUDA MODE 讲座](https://github.com/cuda-mode/lectures)等**教育资源**，以及招募志愿者录制和分享内容，推动了知识传播。

- 社区项目如 [llm.mojo](https://github.com/dorjeduck/llm.mojo)（llm.c 的 Mojo 移植版）、[Perplexica](https://github.com/ItzCrazyKns/Perplexica/)（Perplexity AI 克隆版）以及用于文档检索的 [LlamaIndex 集成](https://medium.com/ai-advances/enhancing-document-retrieval-with-memory-a-tutorial-for-llamaindex-with-colbert-based-agent-1c3c47461122)，展示了草根创新。

4. **LLM 开发的数据集与数据策略**：讨论强调了数据质量、策展（curation）以及训练数据的战略方法的重要性，包括：

- **合成数据生成**技术，如 [StableLM](https://arxiv.org/abs/2402.17834)（ReStruct 数据集）和 [MiniCPM](https://arxiv.org/abs/2404.06395)（混合 OpenOrca 和 EvolInstruct）中使用的技术，重点关注用于通过定制合成数据对齐 LLM 的 [CodecLM](https://arxiv.org/pdf/2404.05875.pdf)。

- **数据过滤策略**和[数据策展缩放定律](https://x.com/pratyushmaini/status/1778577153107570770)的发展，正如 CVPR 2024 论文中所述，强调策展不能脱离算力（compute-agnostic）。

- **多语言和多模态数据集**，呼吁使用符合版权许可的欧盟文本和多模态数据来训练大型开放多模态模型，反映了对多样化数据源日益增长的需求。[Source 1](https://blog.eleuther.ai/pile-t5/) | [Source 2](https://wizardlm.github.io/WizardLM2/) | [Source 3](https://publications.reka.ai/reka-core-tech-report.pdf) | [Source 4](https://coconut-mode.com/posts/ring-attention/)

5. **其他**

- **Stable Diffusion 3 引发关注与辩论**：AI 社区正热切期待 **[Stable Diffusion 3 (SD3)](https://www.youtube.com/watch?v=mQSKoAEaIJA)** 的发布，讨论其在质量和效率方面的潜在改进。对话围绕使用 **[SD Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)** 等工具在性能较低的 GPU 上优化性能，并探索结合 **[ControlNet](https://github.com/lllyasviel/ControlNet)**、**Lora** 和 **outpainting** 技术的 AI 驱动创意工作流。SD3 中严厉的提示词审查引发了对潜在质量下降的担忧，正如 [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1c3ro5y/stability_employee_preview_of_stable_diffusion_3/) 上所讨论的那样。

- **Perplexity AI 的路线图与模型对比**：**Perplexity AI** 的 6 月路线图展示了诸如强制 JSON grammar、新的 Databricks 模型、Model Info 接口、状态页面以及多语言支持等新功能，可在其 [功能路线图页面](https://docs.perplexity.ai/docs/feature-roadmap) 查看。讨论中比较了 **Claude Opus**、**GPT-4** 和 **RAG** 在各种任务中的 context window 和性能。据这篇 [文章](https://analyticsindiamag.com/meta-releases-ai-on-whatsapp-looks-like-perplexity-ai/) 报道，Meta 在 WhatsApp 上发布了一个类似于 Perplexity AI 的 AI 界面，引发了人们对 AI 在即时通讯平台中日益融合的兴趣。

- **Tinygrad 在 NVIDIA GPU 上实现 P2P**：**Tinygrad** 通过修改 NVIDIA 驱动，成功在 **NVIDIA 4090 和 4070 TI Super GPU** 上启用了 peer-to-peer (P2P) 支持，实现了 14.7 GB/s 的 AllReduce 性能。这一突破已在 [Twitter](https://twitter.com/__tinygrad__/status/1778677126092509611) 上分享，代码可在 [GitHub](https://github.com/tinygrad/open-gpu-kernel-modules) 获取，这对于降低运行 LLM 的成本具有重要意义。**CUDA community** 正在通过 [One Billion Row Challenge](https://1brc.dev/) 挑战性能极限，并探索低精度计算的优化策略。

- **Eleuther AI 的 Pile-T5 模型与研究洞察**：**EleutherAI** 推出了 **[Pile-T5](https://blog.eleuther.ai/pile-t5/)**，这是一种在来自 Pile 的 2 万亿 token 上训练的 T5 模型变体，在 *SuperGLUE* 和 *MMLU* 等基准测试中表现出更优的性能。该模型在代码相关任务中表现出色，其权重和训练脚本已在 [GitHub 上开源](https://github.com/EleutherAI/improved-t5)。研究讨论深入探讨了 **MoE** 与 dense transformer 模型的对比、tokenization 在 LLM 中的作用，以及使用 Google 的 **[Patchscopes](http://research.google/blog/patchscopes-a-unifying-framework-for-inspecting-hidden-representations-of-language-models/)** 等框架对 hidden representations 的可解释性。

---

# PART 1: High level Discord summaries

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Stable Diffusion 3 引发热议**：AI 爱好者们对 [Stable Diffusion 3 (SD3)](https://www.youtube.com/watch?v=mQSKoAEaIJA) 充满期待，讨论其提升效率的潜力，并辩论 SD Forge 在优化低性能 GPU 表现方面的优点。

**Pixart-Sigma 挑战 VRAM 极限**：在 [ComfyUI](https://github.com/city96/ComfyUI_ExtraModels) 中应用带有 T5 conditioning 的 Pixart-Sigma 引发了关于 VRAM 占用的讨论，参与者指出，即使在 8-bit 压缩模式下，T5 也能将 VRAM 占用维持在 10.3GB 以下。

**内容创作 AI 工具备受关注**：关于 ControlNet、Lora 等 AI 工具以及 outpainting 等扩展技术的交流，暗示了在创意工作流中对一致色彩生成和背景延伸的需求。

**辩论 AI 的 CPU 与 GPU 效率**：社区成员交流了 GPU 显存优化的排错技巧，并为在性能较低的机器上运行 Stable Diffusion 的用户推荐了 `--lowvram` 等功能，强调了 CPU 和 GPU 处理速度之间的显著差异。

**艺术家寻求技术驱动的合作**：AI 与艺术工具融合的趋势仍在继续，一位数字艺术家正在寻求关于一款结合 AI 功能的绘画应用的建议和教程帮助，项目详情见 [GitHub](https://github.com/QuintessentialForms/ParrotLUX)。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **文本与逻辑中的 AI 模型表现**：辩论强调 **Claude Opus** 在文本创作方面表现出色，而 **GPT-4** 在逻辑推理方面更胜一筹。从业者应根据应用需求探索最适合的模型。

- **上下文依赖于模型选择**：**Claude Opus** 因能精准执行顺序指令而受到赞誉，而讨论表明 **GPT-4** 在多次追问后可能会出现失误；**RAG** 在文件上传的大上下文检索方面具有优势。

- **Perplexity 路线图首次展示多语言支持**：Perplexity AI 的 6 月路线图预告了强制 JSON grammar、新的 Databricks 模型、Model Info endpoint、状态页、多语言支持以及 N>1 sampling，可在其 [特性路线图页面](https://docs.perplexity.ai/docs/feature-roadmap) 查看。

- **API 细节引发对话**：用户深入探讨了 Perplexity API 的细微差别——从模型中的默认 temperature 设置、引用功能、社区辅助获取响应中的 URL，到提高 rate limits 的流程。

- **Meta 模仿 Perplexity？**：有传言称 Meta 为 WhatsApp 推出的 AI 界面与 Perplexity 相似，这表明 AI 正在加速集成到即时通讯平台中，一篇对比两者的文章进一步强调了这一点（[Meta Releases AI on WhatsApp, Looks Like Perplexity AI](https://analyticsindiamag.com/meta-releases-ai-on-whatsapp-looks-like-perplexity-ai/)）。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Geohot 为 4090 破解并恢复了 P2P 支持**："geohot" 巧妙地在 NVIDIA 4090 上实现了 peer-to-peer 支持，从而增强了多 GPU 设置；详情可在 [GitHub](https://github.com/tinygrad/open-gpu-kernel-modules) 上找到。

**Unsloth 获得多 GPU 动力**：对 Unsloth AI 中多 GPU 支持的兴趣激增，Llama-Factory 被认为是值得调研的集成路径。

**来自 Hugging Face 的鼓励**：Unsloth AI 获得了 Hugging Face CEO Clement Delangue 的关注，他在某平台上关注了该项目，暗示了潜在的合作意向。

**AI 博士生活的语言迷宫**：一位博士生概述了他们在为母语开发指令遵循 LLM 时面临的挑战，强调了该项目的复杂性远不止简单的翻译和微调。

**百万美元的数学 AI**：社区关注了一个 1000 万美元的 Kaggle AI 奖项，旨在创建一个能够精通国际数学奥林匹克竞赛（IMO）的 LLM，并讨论了 [AMS](https://www.ams.org/profession/prizes-awards/ams-supported/beal-prize-rules) 为 Beal Conjecture 的证明或反例提供的 100 万美元悬赏。

**资源丰富的 VRAM 实践与战略微调**：AI 工程师们聚集在一起讨论如何高效利用 VRAM 来训练像 Mistral 这样强大的 LLM，分享了诸如 Unsloth 的 "减少 30% VRAM" 更新以及从较短示例开始微调的细致方法。

**语言数据集的文化征服**：交流了扩大低资源语言数据集的策略，包括使用来自 HuggingFace 等平台的翻译数据。

**开创性地使用 Podman 进行 AI 部署**：创新者展示了在 Podman 容器中部署 Unsloth AI，简化了本地和云端实现，如本演示所示：[Podman Build - For cost-conscious AI](https://youtu.be/3iEhFKIDXp0)。

**Ghost 7B Alpha 提升基准测试**：Ghost 7B Alpha 因其优于其他模型的推理能力而受到称赞，标志着在不扩展 tokenizer 的情况下微调的实力，正如爱好者们所讨论的那样。

**模型压缩的思维碰撞**：讨论了 adapter 的融合，特别是 QLoRA 和用于 vLLM 或 GGUF 的 16bit 保存技术，思考了简单合并与反量化策略的复杂性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Pile-T5 的力量**：EleutherAI 推出了 **Pile-T5**，这是一款增强型的 T5 模型变体，通过使用来自 [Pile](https://blog.eleuther.ai/pile-t5/) 的 2 万亿个 token 进行训练。它在 *SuperGLUE* 和 *MMLU* 基准测试中展现了显著的性能提升，并在代码相关任务中表现出色，相关资源包括权重和脚本已在 [GitHub 上开源](https://github.com/EleutherAI/improved-t5)。

**熵数据过滤**：**CVPR 2024 论文** 揭示了在解析数据过滤中熵的重要性方面取得的显著进展。一项实证研究揭示了捕捉数据策展（data curation）如何与熵根本相关的缩放法则（scaling laws），丰富了社区对异构且有限的网页数据及其实际影响的理解。在此处探索该研究：[here](https://arxiv.org/abs/2404.07177)。

**透视 Transformer 黑盒**：Google 的 Patchscopes 框架旨在通过生成自然语言解释，使 LLM 的隐藏表示更具可解释性。同样，一篇介绍 Transformer 因果操纵（causal manipulations）工具包的论文展示了在训练期间定位关键模型子电路（subcircuits）的价值，这可能为避免常见的训练障碍提供路径。有关 JAX 工具包的详细信息可以在 [Stephanie Chan (@scychan_brains) 的这条推文](https://x.com/scychan_brains/status/1779357918737158364?s=46)中找到，关于 Patchscopes 框架的信息请见[此处](http://research.google/blog/patchscopes-a-unifying-framework-for-inspecting-hidden-representations-of-language-models/)。

**MoE 与稠密 Transformer 之争**：社区讨论探讨了 MoE 与稠密 Transformer 模型的容量和优势。关键见解揭示了 MoE 在没有 VRAM 限制时的相对优势，并质疑了稠密模型在相当的参数预算下是否具有同等的性能。人们对驱动模型行为（除指标之外）的基础属性表现出明显的关注。

**NeoX 细节揭晓**：**GPT-NeoX 项目**内部的问题涉及了一些复杂细节，例如为了 **GPU 效率**而采用的超大嵌入矩阵（embedding matrices），以及可能由于非标准激活函数导致的特殊**权重衰减（weight decay）**行为。关于**旋转嵌入（rotary embeddings）**的一项评论指出，其在 NeoX 中的应用与其他模型相比仅为部分应用。目前正在制定[企业级 CLA](https://discord.com/channels/729741769192767510/730090096287547444/1228274522034012200) 以促进对该项目的贡献。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Google 的 Infini-attention 论文备受关注**：一位成员对 Google 的 "Infini-attention" 论文表示了兴趣，该论文可在 [arXiv](https://arxiv.org/pdf/2404.07143.pdf) 上获取，提出了一种处理语言模型更长序列的新方法。
  
- **社区深入研究向量搜索**：在向量搜索速度的对比中，FAISS 的 IVFPQ 以 0.36ms 位居榜首，其次是 FFVec 的 0.44ms；与此同时，FAISS Flat 和 HNSW 的耗时较长，分别为 10.58ms 和 1.81ms。

- **AI 基准测试受到审查**：成员之间的讨论挑战了当前 AI 基准测试的有效性，提出需要更难作弊、针对特定领域任务的评估，例如一个经过审查的[税务基准测试 (tax benchmark)](https://www.vals.ai/taxeval)。

- **期待 WorldSim 的更新**：对即将发布的 **WorldSim** 更新的热情正在高涨，成员们分享了受其启发的实验，同时为下周三预告的新功能做准备。

- **显微镜下的 LLM**：技术交流包括对一篇 [arXiv 论文](https://arxiv.org/abs/2403.13313)中详述的安全导向医疗 LLM 的咨询，在考虑预训练必要性的同时对希腊语 LLM 进行微调的挑战，以及在检索增强生成 (RAG) 查询中进行行内引用 (in-line citations) 的方法。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Mixtral 模型混淆**：社区报告称 `Mixtral 8x22B:free` 已停用；用户应迁移到 [Mixtral 8x22B 标准模型](https://openrouter.ai/models/mistralai/mixtral-8x22b)。实验性模型 **Zephyr 141B-A35B** 和 **Fireworks: Mixtral-8x22B Instruct (preview)** 已开放测试；后者是 Mixtral 8x22B 的微调版本。

**Token 交易故障**：一位用户在购买 Token 时遇到的问题被认为与 OpenRouter 无关；建议其直接联系 **Syrax** 寻求解决。

**展示 Rubik's 研究助手**：有兴趣测试新型 **Rubiks.ai** 研究助手的用户可以加入 Beta 测试，并获得 2 个月的免费高级会员。该工具支持访问 **Claude 3 Opus** 和 **GPT-4 Turbo** 等模型；测试者应使用代码 `RUBIX` 并提供反馈。[探索 Rubik's AI](https://rubiks.ai/)。

**动态路由研讨**：关于通过动态路由到最高每秒交易数（t/s）端点来提升 **Mixtral 8x7B Instruct (nitro)** 速度的讨论非常热烈。对于 **Zephyr 141b** 和 **Firextral-8x22B** 等模型的性能，大家也持有不同意见。

**WizardLM-2 系列激起热潮**：新公布的 **WizardLM-2** 系列包含 8x22B、70B 和 7B 等尺寸，引起了社区的极大兴趣。关注点特别集中在 **WizardLM-2 8x22B** 在 OpenRouter 上的预期表现。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**模型运行出故障**：用户报告了 **LM Studio** 在不同版本和操作系统中存在的模型加载问题，包括 `"Error loading model."` 错误信息，以及在降级版本和关闭 GPU offload 后问题依然存在。一位用户在对模型性能持续感到沮丧后，寻求彻底删除并重新安装 LM Studio 的方法。

**关注硬件**：围绕有效运行 AI 模型的**硬件需求**展开了大量讨论，强调了若要获得媲美 GPT-4 的体验需要高端设备，并指出了 **Threadripper Pro** CPU 利用不足的问题。相比之下，**ROCm** 的支持受到质疑，特别提到了 **Radeon 6600** 在 LM Studio 中不被支持。

**量化困境与创新推理**：**Quantization**（量化）是一个热门话题，用户注意到了性能变化，并争论输出质量的权衡是否值得。同时，一篇题为 "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention" 的论文被广泛传阅，阐述了 Transformer 有效处理超长输入的尖端方法。

**探索新模型领域**：用户分享了对 **Command R+** 和 **Mixtral 8x22b** 等一系列新模型的体验和建议，重点关注不同的配置、性能，甚至介绍了一个基于命令的工具 [missionsquad.ai](https://missionsquad.ai/)。值得注意的是，**Command R+** 在分析长文档方面的表现被赞誉超过了同类模型。

**Beta 版困扰与 Docker 发行**：虽然一些用户在 **MACOS 0.2.19** 中苦苦挣扎，但另一些用户则发起了倡议，例如在 GitHub 上创建并维护 **Autogen Studio** 的 Docker 镜像。与此同时，由于 Linux Beta 版本缺少 ROCm 支持，社区表达了失望情绪，凸显了技术用户遇到的特定障碍。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 自我意识：事实还是虚构？** 围绕 AI 的进步对意识的影响以及伦理使用，展开了激烈的辩论。伦理考量被重点强调，但对于 AI 系统是否具有自我意识尚未达成共识。

- **Token 处理技巧**：讨论了应对 GPT 的 Token 限制的策略，用户建议通过总结内容和使用后续提示词（follow-up prompts）来在约束范围内有效地管理上下文。

- **GPT-4 Turbo 的视觉能力**：澄清了 'gpt-4-turbo' 集成了视觉能力，虽然由于内存限制仍有 4k Token 的限制，但分析图像需要使用脚本。

- **精准提示词技巧**：用户分享了 GPT 模型不一致性的经验，指出特定版本的行为和微调方面会影响响应的准确性，并建议采用 meta-prompting 模板以获得更好的结果。

- **征集 Prompt 挑战黑客**：呼吁对探索 Prompt Hacking 感兴趣的工程师组队参加比赛，建议在与 LLM 的创意交互中进行协作尝试。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**与 Scarlet AI 跨越边界的协作**：全新的 **Scarlet** 平台通过**协作式 AI Agent** 增强了任务管理，这些 Agent 提供自动化的 Context 并支持工具编排（tool orchestration），目前正计划与 Zapier NLA 集成。文中提到了一个正在开发中的 Demo，[Scarlet 官网](https://scarletai.co) 提供了其服务的详细信息。

**关于 Perplexity 消失传闻的讨论**：Perplexity 的 "online" 模型从 LMSYS Arena 中移除，引发了关于该模型效能及其在工程工具中集成潜力的辩论。这一事件凸显了人们对 AI 模型集成效率的共同关注。

**AI 治理 YouTube 的混乱内容**：工程师们探讨了为 AI 应用**转录并总结 YouTube 内容**的策略，辩论了包括 Descript、youtube-dl 以及带有 diarization 功能的 Whisper 在内的各种工具的优劣。讨论反映了在简化 AI 模型训练内容处理方面的持续努力。

**Limitless 展望个性化 AI 的未来**：Rewind 更名为 **Limitless**，并推出了一款具备**个性化 AI 能力**的可穿戴设备，引发了围绕本地处理和数据隐私的讨论。这体现了人们对新型 AI 驱动的可穿戴技术在安全影响方面的高度关注。

**在语义搜索中探索向量空间**：关于语义搜索复杂性的深入讨论，包括向量大小、内存使用和检索性能，最终促成了一个旨在深入研究 Embedding 模型的黑客松提案。这些对话反映了对优化 AI 驱动的搜索应用中速度、效率和性能的密切关注。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**赢得你的 Mojo 勋章**：通过向 `modularml/mojo` 仓库贡献代码或创建酷炫的开源项目来参与 **Mojo** 社区，从而获得久负盛名的 `Mojician` 角色。拥有已合并 PR 的成员可以私信 Jack，将其 GitHub 贡献与 Discord 身份绑定。

**Mojo 对 Python 的愿景**：社区正热切期待 Mojo 扩展对 Python 库的全方位支持，目标是包含**带有 C 扩展的 Python 包**。同时，增强 Mojo `Reference` 易用性的工作正在进行中，Chris Lattner 暗示了一项可能简化可变性管理（mutability management）的提案。

**代码生成与错误处理的前沿**：GPU 代码生成策略以及可能借鉴 Swift 实现的 "extensions" 功能（用于卓越的错误处理）引发了成员间的技术辩论，预示了 Mojo 开发的未来方向。

**社区代码协作激增**：llm.mojo 项目活动频繁，**向量化和并行化**等性能提升技术可能会受益于集体智慧，包括在单个仓库中保持 C 和 Mojo 代码库的同步。

**Mojo 更新的 Nightly 新闻**：Mojo 团队将在即将发布的版本中修复标准库中包命名的差异。此外，将 `StaticTuple` 更新为 `Array` 并支持 `AnyType` 的想法引起了关注，社区还发起了对标准库 Unicode 支持贡献的号召，并讨论了正确的项赋值（item assignment）语法。

**Modular 更新的 Twitter 动态**：关注 [Modular 的 Twitter](https://twitter.com/Modular) 以获取最新的推文动态，包括更新和公告。该机构最近发布的六条推文揭示了他们正在进行的各项计划。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**P2P 获得速度提升**：**Tinygrad** 增强了对 **NVIDIA 4090 GPUs** 的 P2P 支持，在修改 NVIDIA 驱动后，AllReduce 性能达到 14.7 GB/s，详情见[此处](https://x.com/__tinygrad__/status/1778676746378002712)；而 **PyTorch** 正在处理 namespace 构建的复杂性，nightly build 显示其性能慢于 `torch.matmul`。

**海量行排序挑战等待 CUDA 竞争者**：tspeterkim_89106 发起了 CUDA 实现的 [One Billion Row Challenge](https://1brc.dev/)，在 **V100** 上运行时间为 16.8 秒，令人印象深刻，并邀请他人挑战 [4090 GPU 上的 6 秒纪录](https://tspeterkim.github.io/posts/cuda-1brc)。

**性能优化的新领域**：CUDA 讨论围绕在独立 CUDA streams 中运行独立 matmuls 的优点、利用 `stable_fast` 优于 `torch.compile` 的方案，以及 `stable_fast` 挑战 **int8 quant tensorrt** 速度所展示的高效低精度计算追求。

**记录并分享 CUDA 专业知识**：**CUDA MODE** 正在招募志愿者通过 YouTube 录制和分享内容，讲课材料也维护在 [GitHub](https://github.com/cuda-mode/lectures) 上，并强调可能会转向直播以应对不断增长的成员规模。

**HQQ 和 LLM.c：迈向高效**：gpt-fast 上的 **HQQ 实现** 更新通过 **torchao int4 kernel** 支持提升了 token generation 速度，而 **LLM.c** 正面临 CUDA softmax 集成挑战，探索 online softmax algorithm 效率，并兼顾峰值性能与教学清晰度的双重目标。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**连接器困惑已消除**：一名成员解决了 **Cohere's connector API** 的问题，了解到连接器 URL 必须以 `/search` 结尾以避免错误。

**精细微调**：**Cohere's base models** 可通过其仪表板进行微调，如[仪表板链接](https://dashboard.cohere.com/fine-tuning/)所示，选项已扩展至 Amazon Bedrock 和 [AWS](https://aws.amazon.com/marketplace/seller-profile?id=87af0c85-6cf9-4ed8-bee0-b40ce65167e0)。

**新的 Cohere 工具即将问世**：分享了关于新 **Cohere** 功能的更新，特别是名为 **Coral** 的聊天机器人界面，以及即将发布的用于连接器实现的 **AWS toolkits**。

**模型性能讨论**：围绕 **Cohere models**（如 **command-r-plus**）的对话涉及了它们在不同硬件上的性能，包括 Nvidia 最新的显卡和 TPUs。

**AI 新手的学习途径**：寻求教育资源的社区新成员被引导至 **LLM University** 等免费课程，并提供了 [Cohere 教育文档](https://docs.cohere.com/docs/llmu)的链接。

**Command-R 震撼核心**：**Command-R** 因成为 Cohere 新集成的核心模块而获得赞誉，突显了其重要性。

**量化与 AI 融合**：发布了 **Quant Based Finance** 的 Beta 测试邀请，吸引对 AI 驱动的财务分析感兴趣的人士，链接见[此处](https://quantfino.com/join-beta)。

**Rubiks.AI 推出**：分享了 **Rubiks.AI** 的 Beta 测试邀请，这是一个新的高级研究助手和搜索引擎，提供对 **Claude 3 Opus** 和 **GPT-4 Turbo** 等模型的早期访问，网址为 [Rubiks.AI](https://rubiks.ai/)。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**多模型技术诀窍汇总**：工程师们交流了在单个 GPU 上部署**多个 A1111 模型**的技巧，重点讨论了并行运行模型时的资源分配。在 [NLP](https://huggingface.co/docs/transformers/model_doc/bart) 频道中，讨论探索了轻量级 Embedding 选项，如 **all-MiniLM-L6-v2**，并建议将 **paraphrase-multilingual-MiniLM-L12-v2** 用于学术目的。

**认知碰撞：AI 模型跨越现实**：[cool-finds 频道](https://github.com/QUVA-Lab/e2cnn)分享了 PyTorch 与 Blender 集成以实现实时数据流水线的链接，而一篇 *Medium* 文章介绍了 **LlamaIndex** 的文档检索增强功能。用于 Zero-Shot 目标检测的 **Grounding DINO** 模型及其在 Transformers 中的应用在 [computer-vision](https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino) 频道引起关注。

**社区驱动的 AI 时间线与心智模型**：旨在通过关键论文绘制数据科学演进图谱的项目 **RicercaMente** 在 [cool-finds](https://github.com/EdoPedrocchi/RicercaMente) 和 [NLP](https://github.com/EdoPedrocchi/RicercaMente) 频道受到推崇，并邀请社区协作。同时，一种名为 Counterfactual Inception 的方法被提出用于解决 AI 回答中的幻觉问题，详见 [arXiv 论文](https://arxiv.org/abs/2403.13513)及相关的 [GitHub 项目](https://github.com/ivy-lvlm/counterfactual-inception)。

**训练的磨难与挑战**：在 [computer-vision](https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino) 频道中，一位用户在 U-Net 训练 11 个 epoch 遇到平台期后，考虑将 **Deep Image Prior** 用于图像清洗任务。在 [diffusion-discussions](https://github.com/huggingface/diffusers/issues/7672) 中，讨论了多模态 Embedding，并澄清了一个关于 Gradio 聊天机器人在图像生成时 Token 限制警告过载的问题。

**跨界交流：活动与教育**：[today-im-learning](https://www.eventbrite.ca/e/llm-reading-group-march-5-19-april-2-16-30-may-14-28-tickets-851921368747?aff=oddtdtcreator) 和 [reading-group](https://www.eventbrite.ca/e/llm-reading-group-march-5-19-april-2-16-30-may-14-28-tickets-851921368747?aff=oddtdtcreator) 频道宣传了即将举行的 LLM Reading Group 课程，重点关注开创性研究，包括 OpenAI 的 CLIP 模型和 Zero-Shot Classification。此外，还为 NLP 初学者推荐了资源，包括入门指南和 Transformer 概览视频。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**诈骗提示：LAION 不存在 NFT**：社区反复发布关于诈骗的警告，涉及一个冒充 **LAION** 声称提供 NFT 的虚假 Twitter 账号。社区已确认 **LAION** 仅专注于**免费、开源的 AI 资源**，不从事任何销售活动。

**何时引导 Diffusion**：一篇 [arXiv 论文](https://arxiv.org/abs/2404.07724)介绍了最新发现，即 Diffusion 模型在特定噪声水平下应用引导（Guidance）时图像效果最佳，强调应重点关注生成的中间阶段，同时避开早期和晚期阶段。

**从 AI 音频创新到伦理讨论**：讨论范围广泛，从引入新的 AI 模型（如用于声音生成的 Hugging Face **Parler TTS**），到关于正确使用“伦理数据集”的辩论，以及此类政治术语在 AI 研究中的影响。

**Stable Diffusion 3 的困境**：社区见解指出，**Stable Diffusion 3** 由于严格的提示词审查（Prompt Censorship）面临质量下降的风险。人们期待进一步的优化来解决这一潜在问题，相关讨论特别分享在 [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1c3ro5y/stability_employee_preview_of_stable_diffusion_3/) 上。

**Diffusion 模型故障排除**：一位成员分享了一个 [GitHub 仓库](https://github.com/K3dA2/Diffusion-Model)，他在训练 Diffusion 模型时遇到了问题：尽管尝试调整了正则化和学习率，但在推理过程中模型仍输出随机噪声和纯黑图像。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**警惕垃圾信息**：多个频道报告了包含成人内容链接的垃圾信息，虚假宣传 **TEEN/ONLYFANS PORN**，存在潜在的网络钓鱼风险。建议成员避免点击可疑链接或内容。

**RAG 操作需要精确性**：用户在对法律合同进行检索增强生成（RAG）操作时遇到了文档切分问题，章节内容被错误地链接到前一章节，从而损害了检索准确性。

**LangChain 实现并行化**：利用 LangChain 的 `RunnableParallel` 类可以并行执行任务，提高 LangGraph 操作的效率——对于那些追求性能优化的开发者来说，这是一种值得考虑的方法。

**新兴 AI 工具与技术**：分享了各种资源、教程和项目，包括 [Meeting Reporter](https://meeting-reporter.streamlit.app/)、[RAG 指南](https://luisciber.substack.com/p/how-to-build-a-rag-system-locally) 以及个性化推荐系统，旨在为 AI 专业人士提供前沿知识和实用解决方案。

**观看与学习**：推荐了一系列 YouTube 教程，重点介绍如何使用 Vertex AI Agent Builder 实现聊天 Agent，并将其与 Slack 等通信平台集成，这对那些对 AI 驱动应用开发感兴趣的人非常有价值。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**OpenAccess AI 走向开源**：NVIDIA Linux GPU 的 P2P 支持通过 [GitHub 上的 open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules) 得到了提升，为增强 GPU 功能提供了工具。

**Fireworks AI 凭借 Instruct MoE 引起关注**：Fireworks AI 的 Mixtral 8x22b Instruct OH 取得了令人期待的结果，预览见[此处](https://fireworks.ai/models/fireworks/mixtral-8x22b-instruct-preview)。尽管在 DeepSpeed zero 3 上遇到了小问题，但通过从 DeepSpeed 的 [main 分支](https://github.com/microsoft/DeepSpeed/pull/5008)拉取更新已得到解决。

**DeepSpeed 的贡献说明**：虽然它不直接加速模型训练，但 *deepspeed zero 3* 在训练大型模型方面表现出色。通过集成来自 DeepSpeed 官方 GitHub 仓库的更新，已成功解决了 MoE 模型的兼容性问题。

**与 AI 的和谐关系**：AI 音乐创作的进展备受关注，一首由 AI 创作的曲目可在 [udio.com](https://www.udio.com/songs/eY7xtug1dV6hbfCDhyHJua) 试听。

**高级模型训练工具**：讨论围绕模型合并、LISA 和 DeepSpeed 的使用及其对模型性能的影响展开。[该工具](https://github.com/MeNicefellow/Mixtral-Model-Expert-Extractor/blob/main/main.py)被引用用于提取 Mixtral 模型专家（experts），同时还讨论了硬件先决条件。

**动态权重解冻**：针对 GPU 受限的用户，出现了关于**动态权重解冻（dynamic weight unfreezing）**策略的讨论，以及 **Mixture-of-Depths** 的非官方 GitHub 实现，可在此处访问：[GitHub 链接](https://github.com/astramind-ai/Mixture-of-depths)。

**RTX 4090 GPU 与 P2P**：在 RTX 4090 上通过 **tinygrad** 成功启用 P2P 显存访问，引发了关于消除 P2P 使用障碍的讨论，社区对这一成果表现出极大的热情。

**对抗模型重复性**：针对模型产生重复输出的持续缺陷，成员们开始探索多样化的数据集和微调方法。**Mergekit** 成为模型手术的首选工具，其配置灵感源自 [WestLake-10.7B-v2](https://huggingface.co/froggeric/WestLake-10.7B-v2/blob/main/mergekit_config.yml)。

**微调与 Prompt 手术**：深入探讨了 Axolotl 的微调细节，在 [Axolotl GitHub 仓库](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples)的指导下解决了 IndexError 和 Prompt 格式问题，并成功调整了配置。

**Mistral V2 表现出众**：Mistral v2 instruct 在第一轮（first-epoch）训练中就取得了异常出色的结果，在包括元数据提取在内的多种任务中表现优异，凭借新的**自动化能力**超越了 **qwen** 等模型。

**DeepSpeed Docker 深度探索**：通过 DeepSpeed 进行分布式训练需要自定义 Docker 构建，并简化 SSH 密钥以实现节点间的免密通信。使用正确的环境变量启动容器至关重要，详情见此 [Phorm AI 代码搜索](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=d905c33d-1397-4ef3-82ad-a16aadf1eb1f)链接。

**DeepSpeed 与 🤗 Accelerate 的协作**：DeepSpeed 与 🤗 Accelerate 无缝集成，且不会覆盖自定义的学习率调度器（learning rate schedulers），`push_to_hub` 方法进一步简化了 Hugging Face 模型中心仓库的创建。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**恶意软件误报与命令行谜题**：工程师们提出了关于 **Avast 抗病毒软件** 检测以及 OpenInterpreter 中 `ngrok/ngrok/ngrok` 命令行引起困惑的问题；建议更新文档以消除用户的疑虑。

**Tiktoken 获得构建进度的 PR**：一个旨在通过更新 OpenInterpreter 的 tiktoken 版本来**解决构建错误**的 GitHub 拉取请求（PR）表明改进正在进行中；在此查看更改 [here](https://github.com/OpenInterpreter/open-interpreter/pull/1204)。

**持久化难题：Assistants API 的出现**：讨论了集成 **Assistants API** 进行数据持久化的问题，社区成员创建了 Python 助手模块以实现更好的会话管理；正在寻求关于节点操作实现的建议。

**Ubuntu 上的 OS Mode 历险记**：Ubuntu 上的 **Open Interpreter OS Mode** 引发了故障排除讨论，重点在于降级到 **Python 3.10** 以实现平台兼容性并配置辅助功能设置。

**自行定制：O1 的用户驱动创新**：O1 社区通过**个人修改和增强**（如改进电池和定制外壳）展示了他们的创造力；一个基于 Open Interpreter 文档训练的自定义 GPT 模型正在为 ChatGPT Plus 用户提供帮助。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**LlamaIndex 迁移 PandasQueryEngine**：最新的 **LlamaIndex 更新 (v0.10.29)** 将 **PandasQueryEngine** 重新定位到 `llama-index-experimental`，这需要调整导入路径，并提供了错误消息以指导过渡。

**AI 应用生成器引起关注**：**LlamaIndex** 与 T-Systems 及 Marcus Schiesser 合作推出了 **create-tsi**，这是一个用于生成符合 GDPR 规范的 AI 应用程序的命令行工具，通过一条推广 [推文](https://twitter.com/llama_index/status/1778812761893650551) 引起了社区的兴趣。

**重新定义数据库和检索策略**：社区交流深入探讨了用于相似性搜索的理想向量数据库，对比了 Qdrant、pg vector 和 Cassandra，并讨论了利用混合搜索进行多模态数据检索，参考了 [LlamaIndex 向量存储指南](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/)。

**增强 AI 推理的教程和技术**：分享的文章和[教程](https://medium.com/ai-advances/enhancing-document-retrieval-with-memory-a-tutorial-for-llamaindex-with-colbert-based-agent-1c3c47461122)展示了在基于 Colbert 的 Agent 中通过记忆强化文档检索的方法，以及集成小型知识图谱以提升 **RAG 系统**，正如 [WhyHow.AI](https://medium.com/enterprise-rag/the-role-of-small-vs-big-knowledge-graphs-995d9449c208) 所强调的那样。

**社区表彰与技术支持**：社区对推进 AI 推理的文章表示赞赏，同时 **LlamaIndex** 用户解决了技术难题并鼓励积极为文档做贡献，强化了平台对知识共享和支持的承诺。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **堆叠 4090 削减了 TinyBox 的开支**：关于 **RTX 4090 驱动补丁** 登上 Hacker News 头条的热情高涨，这标志着其在 GPU 效率方面的突破。通过堆叠，**RTX 4090** 在运行大语言模型 (LLMs) 时能大幅降低成本。计算数据显示，与红色阵营 (Team Red) Tinybox 相比，成本降低了 **36.04%**；与绿色阵营 (Team Green) Tinybox 相比，成本更是惊人地降低了 **61.624%**。

- **Tinygrad 开发持续升温**：**George Hotz** 与 Tinygrad 社区正致力于改进 **tinygrad 文档**，特别是针对开发者体验和错误信息的清晰度。社区达成共识，将代码行数限制提高到 *7500* 行以支持新的 backends，并解决了不稳定的 mnist 测试以及 scheduler 中的非确定性问题。

- **Tinygrad 中的算子融合与图切分**：对 **Tinygrad** 内核生成的深入剖析揭示了融合过程中的图切分 (graph splits) 可能源于广播 (broadcasting) 需求等原因，详见 `create_schedule_with_vars` 函数。用户了解到 **Tinygrad** 能够运行 Stable Diffusion 和 Llama 等重量级模型，尽管它并不直接用于这些模型的训练。

- **Colab 与贡献故事**：AI 工程师们交流了在 **Google Colab** 上设置和利用 **Tinygrad** 的技巧，推荐的安装命令为 `pip install git+https://github.com/tinygrad/tinygrad`。讨论中提到了张量填充 (tensor padding) 和 Transformer 模型带来的挑战，而使用 `None` 进行填充被认为是一个实用的解决方案。

- **鼓励贡献与文档深度探索**：为了帮助新手，社区分发了 **Tinygrad 文档** 链接和个人学习笔记，揭开了引擎运行机制的神秘面纱。关于在没有 CUDA 支持的情况下进行贡献的问题受到了欢迎，强调了在参与贡献之前深入了解 Tinygrad 生态系统的重要性。

进一步探索和理解的相关链接包括：
- [Google Colab 设置](https://colab.research.google.com/drive/1vy97ByB5mLielmE8TzlnnaSZtMUykm-A?usp=sharing),
- [关于 ScheduleItem 的 Tinygrad 笔记](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/scheduleitem.md),
- [Tinygrad 文档](https://github.com/tinygrad/tinygrad/tree/master/docs),
- [Tinygrad CodeGen 笔记](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/codegen.md),
- [Tinygrad GitHub 仓库](https://github.com/tinygrad/tinygrad).

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Hugging Face Collections 简化了产出物组织**：[Hugging Face collections](https://huggingface.co/collections/natolambert/2024-interconnects-artifacts-6619a19e944c1e47024e9988) 已被引入，用于汇总关于开源模型和数据集博客文章中的产出物。这些集合便于再次访问，并配有 API，详见 [Hugging Face 文档](https://huggingface.co/docs/huggingface_hub/en/package_reference/collections)。

**“增量”更新之争与开源数据倡议**：社区成员在 Claude 2 到 Claude 3 转变的重要性上存在分歧，并呼吁关注可能被忽视的开源数据倡议的价值。与此同时，AI 发布公告密集出现，Pile-T5 和 WizardLM 2 位居前列。

**机器学习讨论综述**：对话涉及了 ACL 修订版上传的义务、“大模型”的优势，以及 critic vs reward models 之间的区别——其中 Nato 的 RewardBench 项目是关注焦点。来自 @andre_t_martins 的一条 [推文](https://twitter.com/andre_t_martins/status/1779263540785725737) 澄清了 ACL 修订版上传的流程。

**启发性的论文与研究**：重点介绍的论文包括 "CodecLM: Aligning Language Models with Tailored Synthetic Data" 和 "LLaMA: Open and Efficient Foundation Language Models"（Hugging Face 标识符为 [2302.13971](https://huggingface.co/papers/2302.13971)）。合成数据的作用以及向更强大的模型学习是讨论的核心收获。

**图表获得认可，建议对机器人保持耐心**：在时事通讯领域，图表因其清晰度赢得了赞誉，并承诺未来将进行增强并集成到 Python 库中。有人提到一个实验性机器人可能更需要耐心，而非过早的人为干预。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**Haiku 的速度瓶颈**：工程师们讨论了 **Haiku** 较慢的总响应时间（与其吞吐量形成对比），并建议将 **Bedrock** 作为潜在替代方案，尽管那里也存在速度方面的担忧。

**Claude 的 RP 限制**：由于 **Claude** 拒绝参与角色扮演（RP）提示词（如扮演女仆战士或发送虚构的垃圾邮件），即使在使用各种提示词技术后，社区仍表示担忧。

**越狱枢纽**：在讨论中，分享了 *@elder_plinius* 的一条推文，内容关于 **Claude 3** 的通用越狱方法，以启用更激进的内容，从而绕过像 **Gemini 1.5 Pro** 中那样严格的内容过滤器。社区似乎正在评估其影响；推文链接见[此处](https://x.com/elder_plinius/status/1773455789056745782?s=46)。

**代码能力主张**：某未命名工具的新版本因其增强的代码编写能力和速度而受到赞誉，一位成员正考虑重新激活其 **ChatGPT Plus** 订阅以进一步测试这些升级。

**Claude 的上下文优势**：尽管其他工具在改进，**Claude** 在长上下文窗口（context window）代码任务中仍保持其独特优势，这暗示了 ChatGPT 在处理上下文窗口大小方面的局限性。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**Mixtral 的精通还是神话？**：社区讨论涉及 **Mixtral MoE models** 训练和微调（finetuning）效率的不确定性，一些人怀疑其性能背后有未公开的技术。人们对周末尝试使用 en-de instruct 数据进行微调表现出兴趣，**Envoid 的 "fish" 模型**被提及为 **Mixtral 中 RP/ERP 应用**的一个奇特案例，尽管由于硬件限制尚未测试。

**Llama-Tokenizer 调整教程技巧**：优化自定义 **Llama-tokenizer** 以降低硬件占用的努力促成了资源分享，例如来自 [Hugging Face](https://github.com/huggingface/transformers/blob/fe2d20d275d3591e2619a1adb0fa6ae272605208/src/transformers/convert_slow_tokenizer.py#L534) 的 `convert_slow_tokenizer.py` 和来自 [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp/blob/master/convert.py) 带有 `--vocab-only` 选项的 `convert.py` 脚本。此外，社区还在征集用于训练大型开放多模态模型的**无版权欧盟文本和多模态数据**。

**翻译器的模板倾向**：**occiglot/7b-de-en-instruct** 模型在评估中展现了模板敏感性，由于模板正确性（如 Hugging Face 的[正确模板](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval#occiglotocciglot-7b-de-en-instruct)用法所示），其在德国 RAG 任务上的表现有所波动。

**来自 StableLM 和 MiniCPM 的训练启示**：预训练（pretraining）方法的见解被重点强调，参考了 StableLM 使用的受 [ExpressAI GitHub repo](https://github.com/ExpressAI/reStructured-Pretraining) 启发的 **ReStruct dataset**，以及 MiniCPM 在冷却阶段偏好混合 OpenOrca 和 EvolInstruct 等数据的做法，详见其研究的[第 5 章](https://arxiv.org/abs/2404.06395)中的详细说明。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Burnai 引起关注**：社区中的 Rust 爱好者指出 [Burnai project](https://burn.dev/) 在优化推理（inference）方面的潜力尚未被充分利用，并对 Mozilla 尽管是 Rust 的创造者却缺乏参与提出了疑问。

**Llamafile 获得 Mcaffee 的信任**：*llamafile 0.7 binary* 已成功进入 Mcaffee 的白名单，这标志着该项目安全声誉的一次胜利。

**新合作者活跃讨论**：一位新参与者加入了进来，渴望投入到协作和知识共享中，预示着新视角的出现。

**对 Vulkan 和 tinyblas 的好奇心达到顶峰**：人们对预期的 v0.8 版本中潜在的 Vulkan 兼容性以及将 *tinyblas* 合并到上游以用于 ROCm 应用的好处产生了浓厚兴趣，表明了对性能增强的集中关注。

**模型打包寻求帮助**：对将自定义模型打包进 llamafile 的指导需求引发了社区交流，促成了诸如[关于容器发布的 GitHub Pull Request](https://github.com/Mozilla-Ocho/llamafile/pull/59#issuecomment-1840814790) 等贡献。

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **垃圾邮件清理机器人建议**：有人建议在服务器中集成 **wick bot**，以自动遏制垃圾邮件问题。
- **寻求通过 DM 进行调试**：一位用户在寻求代码问题帮助时，表示希望通过 DM（私信）进行直接协作，并以此请求来验证自己是真人，以消除可能的机器人嫌疑。
- **重新考虑 Discord 邀请**：有建议提出应禁止 Discord 邀请，以防止服务器内出现复杂情况，但目前尚未达成共识或做出决定。
- **项目活跃度检查**：一位用户询问了项目的状态，仅通过一句“项目还活着吗？”来质疑其活跃度，未提供更多上下文。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Code-Jamba-v0.1 展示多语言精通**：[Code-Jamba-v0.1](https://huggingface.co/ajibawa-2023/Code-Jamba-v0.1) 模型在 **2 x H100 GPU** 上训练了 **162 小时**，利用 [Code-290k-ShareGPT](https://huggingface.co/datasets/ajibawa-2023/Code-290k-ShareGPT) 和 [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) 等数据集，在 **Python, Java 和 Rust** 代码编写方面表现出色。
  
- **征集 Code-Jamba-v0.1 评估数据**：成员们强调了对 **Code-Jamba-v0.1** 进行性能基准测试的需求，并思考了 AI21 Labs 数据集在增强其他 Large Language Models 方面的效用。

- **关注 Jamba API 公告**：一位成员关于即将推出的 **Jamba API** 的提问暗示了其与 Fireworks AI 集成的可能性，目前正等待与其领导层的进一步讨论。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **AI 的密码说服游戏**：[PasswordGPT](https://passwordgpt.io/) 是一款引人入胜的游戏，挑战玩家说服 AI 泄露密码，在模拟的社会工程中测试自然语言生成的极限。
- **标注艺术与原始 LLM 分析的对比**：社区在偏好上存在分歧：是应该在建模前对数据集进行细致标注以获得更好的理解，还是依赖 Large Language Models (LLMs) 的原始预测能力。
- **LLM 时代之前的历史数据破译**：大家认可了一项共同努力，即一个团队使用 LLM 之前的 Transformer 模型从历史文档中细致地提取记录，展示了跨时代的 AI 应用。
- **通过开放提示词增强用户参与度**：共识倾向于支持开放提示词的 AI 演示，这可以通过揭示底层 Prompt 来引导更深层次的用户参与，并提高“智胜” AI 的能力。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **扩展 AI 前沿**：宣布了一场名为“**解锁将 Gen AI 应用扩展到生产环境的秘密**”的即将举行的见面会，探讨扩展 Gen AI 应用面临的重大挑战。来自 [Portkey](https://portkey.ai/)、[Noetica AI](https://www.noetica.ai/) 和 [LastMile AI](https://lastmileai.dev/) 的专家小组成员将分享见解，注册链接请点击[此处](https://lu.ma/llms-in-prod-nyc)。

---

# PART 2: 频道详细摘要与链接

**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1228284147433209898)** (1129 条消息🔥🔥🔥): 

- **Stable Diffusion 讨论集中在 SD3 和效率上**：对话重点围绕即将推出的 Stable Diffusion 3 (SD3) 的影响和能力，成员们热切期待其发布和功能。提供的[示例视频](https://www.youtube.com/watch?v=mQSKoAEaIJA)展示了可感知的质量提升，同时讨论还深入探讨了 SD Forge 对低性能 GPU 的有效性以及 SD3 的发布时间表。

- **Pixart-Sigma, ComfyUI 和 T5 调节**：用户正在 [ComfyUI 环境](https://github.com/city96/ComfyUI_ExtraModels)中探索带有 T5 调节（Conditioning）的 Pixart-Sigma，讨论了 VRAM 需求和性能，并提到在 8-bit 压缩模式下，T5 不会超过 10.3GB VRAM。该模型的提示词遵循度、生成速度以及如 'res_momentumized' 之类的采样器是大家关注的话题，并分享了优化生成质量的见解。

- **ControlNet, Lora 和 Outpainting 查询**：社区正在就各种 AI 工具交换建议，包括 ControlNet、Lora 和外扩填充（Outpainting）功能，以满足特定项目需求（如填充物体背后的背景）。有人询问了在使用 ControlNet 时保持生成的图像颜色一致的可能性。

- **CPU 与 GPU 性能关注点**：成员们讨论了 Stable Diffusion (SD) 在 CPU 上运行时的性能较慢的问题，图像生成时间比使用 GPU 显著更长。用户分享了经验和建议，包括在 SD 中使用 `--lowvram` 标志来减少 GPU 显存占用。

- **社区项目与 AI 功能**：一位艺术家宣布正在开发一款集成 AI 功能的绘画应用，并寻求社区的反馈和教程制作支持。该应用旨在将基本的数字艺术工具与独特的 AI 驱动功能相结合，项目概览可在[此处](https://github.com/QuintessentialForms/ParrotLUX)查看。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://rom1504.github.io/clip-retrieval/?back=https%3A%2F%2Fknn.laion.ai&index=laion5B-H-14&useMclip=false)">Clip front</a>: 未找到描述</li><li><a href="https://tenor.com/view/why-not-both-why-not-take-both-gif-11478682">Why Not Both Take Both GIF - Why Not Both Why Not Take Both - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/nervous-courage-courage-the-cowardly-dog-tense-anxious-gif-17225992103088597327">Nervous Courage GIF - Nervous Courage Courage the cowardly dog - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://civitai.com/posts/2163684">Perturbed-Attention-Guidance Test | Civitai</a>: rMada 的帖子。标签为 . PAG (Perturbed-Attention Guidance): https://ku-cvlab.github.io/Perturbed-Attention-Guidance/ PROM...</li><li><a href="https://www.tiktok.com/@edwinskeletrix/video/7355974950945164586">TikTok - Make Your Day</a>: 未找到描述</li><li><a href="https://subpac.com/">SUBPAC - 体验声音的新方式：Feel it.™</a>: SUBPAC 通过让身体沉浸在低频、高保真的物理声音中，让你感受到低音，外部保持安静。非常适合音乐、游戏和 VR。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1b37h5z/supir_super_resolution_tutorial_to_run_it_locally/?rdt=36399">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://stability.ai/">Stability AI</a>: 通过生成式 AI 激发人类潜力。为每个人、每个地方提供各种模态的开源模型。</li><li><a href="https://tenor.com/bEz63.gif">Server Is For Js Javascript GIF - Server Is For Js Javascript Js - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=PqREA6-bC3w">SUPIR: 新的 SOTA 开源图像放大和增强模型，优于 Magnific & Topaz AI 教程</a>: 使用 V8 版本，现在也可以在 12 GB GPU 上配合 Juggernaut-XL-v9 基础模型运行。在本教程视频中，我介绍了 SUPIR (Scaling-UP Image Restoration)，一种最先进的...</li><li><a href="https://leonardo.ai/">Home v2</a>: 使用我们的 AI 图像生成器改变您的项目。以无与伦比的速度和风格生成高质量的 AI 图像，提升您的创意愿景</li><li><a href="https://youtu.be/PqREA6-bC3w?t=1564">SUPIR: 新的 SOTA 开源图像放大和增强模型，优于 Magnific & Topaz AI 教程</a>: 使用 V8 版本，现在也可以在 12 GB GPU 上配合 Juggernaut-XL-v9 基础模型运行。在本教程视频中，我介绍了 SUPIR (Scaling-UP Image Restoration)，一种最先进的...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c2je28/comment/kzc0ltv/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/city96/ComfyUI_ExtraModels/issues/20">是否可以使用 Flan-T5 · Issue #20 · city96/ComfyUI_ExtraModels</a>: 是否可以在 Pixart-Sigma 中使用仅编码器版本的 Flan-T5？这个：https://huggingface.co/Kijai/flan-t5-xl-encoder-only-bf16/tree/main</li><li><a href="https://github.com/QuintessentialForms/ParrotLUX">GitHub - QuintessentialForms/ParrotLUX: 开源 AI 的数位板绘画应用</a>: 开源 AI 的数位板绘画应用。通过在 GitHub 上创建账户来为 QuintessentialForms/ParrotLUX 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c2je28/i_got_access_to_sd3_on_stable_assistant_platform/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/KU-CVLAB/Perturbed-Attention-Guidance">GitHub - KU-CVLAB/Perturbed-Attention-Guidance: "Perturbed-Attention Guidance" 的官方实现</a>: "Perturbed-Attention Guidance" 的官方实现 - KU-CVLAB/Perturbed-Attention-Guidance</li><li><a href="https://youtu.be/rZC65Q3F_Mk">曲速引擎：新模拟</a>: 从 Brilliant 的科学课程中了解更多！前 30 天免费，使用我们的链接可享受年度高级订阅 8 折优惠 ➜ https://brilliant....</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c35onn/remember_when_emad_said_sd3_looks_better_than/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/PKU-YuanGroup/MagicTime">GitHub - PKU-YuanGroup/MagicTime: MagicTime: 作为变形模拟器的延时摄影视频生成模型</a>: MagicTime: 作为变形模拟器的延时摄影视频生成模型 - PKU-YuanGroup/MagicTime</li><li><a href="https://www.youtube.com/watch?v=vxRY_9bv8sc">Dungeons & Dancemusic</a>: 🧙🏻‍♂️🧙🏻‍♂️🧙🏻‍♂️欢迎，年轻的节奏创作者！舞池巫师 Racso 将教你强力舞池节奏的六个基本属性。Str...</li><li><a href="https://github.com/city96/ComfyUI_ExtraModels">GitHub - city96/ComfyUI_ExtraModels: 支持各种图像模型。目前支持：DiT, PixArt, T5 和一些自定义 VAE</a>: 支持...</li>

各种图像模型。目前支持：DiT, PixArt, T5 以及一些自定义 VAEs - city96/ComfyUI_ExtraModels</li><li><a href="https://www.reddit.com/r/StableDiffusion/s/dJpuPy7qOV">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>：通过在 GitHub 上创建账户，为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://civitai.com/models/311874/socks-segmentation-adetailer-sockssegsyolov8">Socks Segmentation - ADetailer - (socks_seg_s_yolov8) - v1.0 | Stable Diffusion Other | Civitai</a>：一个用于分割图像中袜子的 Yolov8 检测模型。该模型可用作 ADetailer 模型（用于 Automatic1111 / Stable Diffusion），或者 ...</li><li><a href="https://huggingface.co/datasets/MohamedRashad/midjourney-detailed-prompts">MohamedRashad/midjourney-detailed-prompts · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1228264914037244025)** (879 条消息🔥🔥🔥): 

- **Context Window 和 RAG 讨论**：用户讨论了 Context Window 的工作原理，指出 **Claude Opus** 能成功遵循指令，而 GPT-4 在多次追问后可能会丢失上下文。据称 RAG 仅检索文档的相关部分，从而为文件上传提供更大的上下文。

- **模型切换与 Perplexity 功能**：一些用户正在考虑是继续使用 **Perplexity Pro** 还是 **You.com**，权衡各平台之间的优缺点，如 RAG 功能、长上下文处理和代码解释功能。

- **代码生成问题**：用户报告了模型在提示时不输出完整代码片段的问题——提到 **Claude Opus** 在生成完整代码方面比 **GPT-4** 更可靠，后者往往倾向于描述性而非可执行的代码。

- **AI 模型评估**：用户讨论了各种 AI 模型在不同任务中的有效性，认为 **Claude Opus** 更擅长散文写作，而 **GPT-4** 更擅长逻辑推理。他们强调通过反复试验来寻找最适合特定需求的模型。

- **其他查询与关注点**：用户询问了有关账户问题和意外扣费的辅助响应，提到了模型的技术问题（如幻觉和错误的搜索结果），提出了对新 AI 服务的隐私担忧，并对 **Reka** 等新模型的可能性以及视频输入功能（涉及成本和上下文限制）表示了兴趣。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.ai/blog/grok-1.5v">Grok-1.5 Vision Preview</a>：未找到描述</li><li><a href="https://retrododo.com/first-official-emulator-igba-now-on-apple-app-store-to-download/">首款官方模拟器 "iGBA" 现已在 Apple App Store 上架下载</a>：一款新的 Game Boy 模拟器正式登陆 Apple App Store，让复古游戏玩家能够畅玩过去喜爱的游戏。</li><li><a href="https://www.forbes.com.mx/como-si-wikipedia-y-chatgpt-tuvieran-un-hijo-asi-es-la-startup-buzzy-ai-madre-de-perplexity-el-buscador-que-acabaria-con-google/">‘Como si Wikipedia y ChatGPT tuvieran un hijo’：这就是初创公司 Buzzy AI，它是 Perplexity 的“母亲”，这款搜索引擎可能会终结 Google</a>：Perplexity 是一款由人工智能驱动的搜索引擎，得到了杰夫·贝佐斯等技术名人的支持，并拥有像执行董事...这样的亿万富翁。</li><li><a href="https://www.lavanguardia.com/andro4all/tecnologia/openai-planea-lanzar-un-buscador-para-competir-con-google">OpenAI 计划推出搜索引擎与 Google 竞争</a>：由 Sam Altman 领导的公司据称正在开发一款搜索引擎，利用 AI 进行比传统搜索更精确的搜索。</li><li><a href="https://docs.perplexity.ai/docs/rate-limits">Rate Limits</a>：未找到描述</li><li><a href="https://tenor.com/view/take-my-money-fry-futurama-meme-gif-16499696">Take My Money Fry GIF - Take My Money Fry Futurama - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://monica.im/home">Monica - 您的 ChatGPT AI 助手 Chrome 扩展程序</a>：未找到描述</li><li><a href="https://vm.tiktok.com/ZGeuW63qm/">TikTok - Make Your Day</a>：未找到描述</li><li><a href="https://www.cognosys.ai/">Cognosys</a>：认识这款能简化任务并加速工作流程的 AI。更快速、更智能地完成工作，无需烦恼</li><li><a href="https://vm.tiktok.com/ZGeuWJMes/">TikTok - Make Your Day</a>：未找到描述
</li>
</ul>

</div>
  

---

**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1228310073898438698)** (23 messages🔥): 

- **美国政策探讨**：成员们正在研究美国政策，有人分享了一个 [Perplexity 搜索链接](https://www.perplexity.ai/search/US-is-considering-lJ9faQytRx.6RItBXyKFSQ)，引导至相关查询。
- **探讨诚实的价值**：分享了一个 [Perplexity AI 搜索](https://www.perplexity.ai/search/Why-honesty-is-I6x.NhtaQ5K.BycdYIXwrA) 链接，建议就诚实的重要性展开讨论。
- **了解 Durable Functions**：参与者通过在 Perplexity AI 上分享的 [搜索查询](https://www.perplexity.ai/search/what-is-durable-XjhaEk7uSGi7.iVc01E1Nw) 深入了解 Durable Functions 的世界。
- **Meta 为 WhatsApp AI 模仿 Perplexity**：一篇文章指出，[Meta 在 WhatsApp 上的 AI](https://analyticsindiamag.com/meta-releases-ai-on-whatsapp-looks-like-perplexity-ai/) 与 Perplexity AI 进行了对比，旨在洞察 AI 在即时通讯平台中日益扩大的应用。
- **Scratchpad-Thinking 与 AutoExpert 结合**：讨论了将 *scratchpad-think* 与 *autoexpert* 功能结合的可能性，并由一个 [Perplexity 搜索](https://www.perplexity.ai/search/for-a-liter-ucusD8pxSDqUo0B0x_3y0w) 突出显示。

**提到的链接**：<a href="https://analyticsindiamag.com/meta-releases-ai-on-whatsapp-looks-like-perplexity-ai/">Meta Releases AI on WhatsApp, Looks Like Perplexity AI</a>：Meta 已在印度及非洲部分地区的 WhatsApp、Instagram 和 Messenger 上悄然发布了其 AI 驱动的聊天机器人。

---

**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1228423998497361920)** (26 messages🔥): 

- **Perplexity API 路线图公布**：Perplexity AI 的路线图（6 天前更新）概述了计划在 6 月发布的功能，例如强制 JSON grammar、N>1 sampling、新的 Databricks 模型、Model Info endpoint、状态页以及多语言支持。文档可在其 [功能路线图页面](https://docs.perplexity.ai/docs/feature-roadmap) 查看。

- **默认 Temperature 设置探讨**：在调用 Perplexity API 且未指定 temperature 时，“online”模型的默认值为 0.2，对于其他模型来说，这个值可能出乎意料地低。建议用户在 API 请求中明确指定 temperature 以避免歧义。

- **寻求 Citations 功能的明确说明**：多位用户询问了 Perplexity API 中将如何以及何时实现 citations（引用），并参与了讨论，分享了 Perplexity 文档中关于功能申请流程的链接。

- **API 问题响应中的循环引用**：一位寻求帮助的用户在 API 响应中未收到新闻报道的 URL，被引导至 Discord 频道的一条帖子获取信息。这体现了社区成员尝试互相帮助的情况。

- **API Rate Limit 提升请求**：一位用户询问如何提高 API 的 rate limit，并被引导填写 Perplexity 网站上的表格，该过程似乎涉及说明提高容量的业务需求。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai/docs/feature-roadmap">Feature Roadmap</a>：未找到描述</li><li><a href="https://docs.perplexity.ai/discuss/65c4824891a9930010e8e9ee">Get Sources and Related Information when using API</a>：未找到描述</li><li><a href="https://docs.perplexity.ai/discuss/66027e76bb5a05005bd5001e">Data from perplexity web and perplexity api</a>：未找到描述</li><li><a href="https://docs.perplexity.ai/discuss/65fef9fdc6b367004540612b">API gives very low quality answers?</a>：未找到描述</li><li><a href="https://perplexity.typeform.com/to/j50rnNiB>">Discover Typeform, where forms = fun</a>：在几分钟内无需代码即可创建精美、互动的表单。免费开始使用。
</li>
</ul>

</div>

---

**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1228257322661580811)** (431 messages🔥🔥🔥):

```html
<ul>
  <li><strong>Unsloth Multi-GPU 更新查询</strong>：有关于 Unsloth AI 的 Multi-GPU 支持更新的咨询。提到了查看集成了 Unsloth 的 Llama-Factory 的建议。</li>
  <li><strong>Geohot 为 4090 增加 P2P 支持</strong>：分享了一个重大更新，“geohot” 已将 P2P 功能破解回 NVIDIA 4090，并附带了相关的 <a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub 链接</a>。</li>
  <li><strong>即将举行的 Unsloth AI 演示和问答活动提醒</strong>：发布了由 Analytics Vidhya 主办的 Unsloth AI 现场演示及问答环节的公告。感兴趣的人员可以通过发布的 <a href="https://us06web.zoom.us/webinar/register/WN_-uq-XlPzTt65z23oj45leQ">Zoom 链接</a>加入。</li>
  <li><strong>Mistral 模型融合策略讨论</strong>：讨论了将 MOE 专家合并到单个模型中的实用性，并对输出质量表示怀疑。一些人考虑将针对特定任务微调 Mistral 并移除较少使用的专家作为一种潜在的压缩方法。</li>
  <li><strong>Hugging Face CEO 在 X 平台关注 Unsloth</strong>：Hugging Face 的联合创始人兼 CEO Clement Delangue 现在在某平台关注了 Unsloth，这引发了对两个 AI 社区未来合作的期待。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>：语言模型的物理学：第 3.3 部分，知识容量缩放定律。缩放定律描述了语言模型规模与其能力之间的关系。与以往通过损失或基准测试评估模型能力的研究不同，我们估计了 n...</li><li><a href="https://us06web.zoom.us/webinar/register/WN_-uq-XlPzTt65z23oj45leQ">Video Conferencing, Web Conferencing, Webinars, Screen Sharing</a>：Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，可跨移动端、桌面端和会议室系统进行视频和音频会议、聊天和网络研讨会。Zoom ...</li><li><a href="https://arxiv.org/abs/2310.08659">LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models</a>：量化是服务大语言模型 (LLM) 不可或缺的技术，最近已应用于 LoRA 微调。在这项工作中，我们关注量化和 L...</li><li><a href="https://huggingface.co/Vezora/Mistral-22B-v0.1">Vezora/Mistral-22B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/AI-Sweden-Models/">AI-Sweden-Models (AI Sweden Model Hub)</a>：未找到描述</li><li><a href="https://huggingface.co/collections/LumiOpen/viking-660fa4c659d8544c00f77d9b">Viking - a LumiOpen Collection</a>：未找到描述</li><li><a href="https://lightning.ai/pages/community/lora-insights/">Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments - Lightning AI</a>：LoRA 是用于训练自定义 LLM 的最广泛使用的参数高效微调技术之一。从使用 QLoRA 节省内存到选择最佳 LoRA 设置，本文提供了实用的...</li><li><a href="https://developer.nvidia.com/nccl">NVIDIA Collective Communications Library (NCCL)</a>：未找到描述</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: NVIDIA Linux open GPU with P2P support</a>：支持 P2P 的 NVIDIA Linux 开源 GPU。通过在 GitHub 上创建账号为 tinygrad/open-gpu-kernel-modules 的开发做出贡献。</li><li><a href="https://github.com/astramind-ai/Mixture-of-depths">GitHub - astramind-ai/Mixture-of-depths: Unofficial implementation for the paper &quot;Mixture-of-Depths: Dynamically allocating compute in transformer-based language models&quot;</a>：论文 "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models" 的非官方实现 - astramind-ai/Mixture-of-depths</li><li><a href="https://github.com/OptimalScale/LMFlow/issues/726">[BUG] LISA: same loss regardless of lisa_activated_layers · Issue #726 · OptimalScale/LMFlow</a>：描述该 Bug：我认为目前的 LISA 实现可能存在问题。无论激活多少层，训练损失都没有区别。没有使用 LMFlow 而是使用了 HF ...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1228628722035331083)** (38 条消息🔥): 

- **探索 AI 领域的博士生活**：一位 AI 领域的一年级博士生讨论了攻读博士学位的压力与乐趣，包括产出结果的压力和自我怀疑。在一条消息中，他们提到自己的课题是在母语中创建一个指令遵循 LLM，并指出这是一个复杂的项目，不仅仅是翻译和微调。

- **AI 竞赛中的潜在“金矿”**：对话转向了 [Kaggle AI 数学奥林匹克竞赛](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize)，该竞赛为能够获得国际数学奥林匹克竞赛金牌的 LLM 提供 1000 万美元的奖金。一位博士生还提到了让他们的后辈来研究这个问题的想法。

- **关于比尔奖（Beal Prize）的讨论**：聊天中包含了一个指向 [美国数学学会（American Mathematical Society）](https://www.ams.org/profession/prizes-awards/ams-supported/beal-prize-rules) 的链接，详细介绍了比尔奖的规则，提到了证明或反驳比尔猜想（Beal Conjecture）的条件，并指出奖金总额为 1,000,000 美元。

- **大学优先事项的辩论**：一名成员指出，在研究课题方面，大学更注重发表论文和产生“影响力”，而不是竞赛奖金，尽管他们在讨论竞赛中巨额的货币奖励。

- **AI 与语言的本质**：有一些关于思维和语言作为自引用系统的哲学思考。有人认为思想是自我生成的，而语言只能通过更多的语言来定义。

- **Instagram 上展示的 AI 技术**：一段简短的插曲展示了对一个 [分享的 Instagram 视频](https://www.instagram.com/reel/C5hP-5lAG6h/?igshid=34e80b5a5eda) 的参与，该视频因其内容和特别整洁的代码而受到称赞，尽管并非所有人都能完全理解它。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize">AI Mathematical Olympiad - Progress Prize 1 | Kaggle</a>：未找到描述</li><li><a href="https://www.instagram.com/reel/C5hP-5lAG6h/?igsh=MzRlODBiNWFlZA=="> Julia | Instagram 上的数学与编程：&quot;当你是一名研究生时 #gradschool #phd #university #stem #phdstudent #machinelearning #ml #ai&quot;</a>：6,197 个赞，139 条评论 - juliaprogramming.jl，2024 年 4 月 8 日：&quot;当你是一名研究生时 #gradschool #phd #university #stem #phdstudent #machinelearning #ml #ai&quot;</li><li><a href="https://www.ams.org/profession/prizes-awards/ams-supported/beal-prize-rules">比尔奖规则与程序</a>：推进研究，建立联系。</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1c2pzqn/jumped_on_the_trend_and_asked_chatgpt_to_make_a/">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1228279453969289256)** (313 条消息🔥🔥): 

- **高效的 VRAM 使用和微调策略**：参与者讨论了高效微调 **Mistral** 等大语言模型的方法，并对训练长样本时的 **vRAM 占用** 表示担忧。建议使用来自 **Unsloth** 的减少 30% VRAM 占用的更新，利用 **梯度累积步数（accumulation steps）** 来抵消大 **batch size** 的影响，并考虑先在短样本上微调再转向长样本的策略。

- **为新语言选择基座模型**：成员们寻求关于在**低资源语言**上进行微调的建议，用户分享了在 **Gemma** 等模型上的经验，并讨论了混合数据集和持续预训练（continuing pretraining）等策略。

- **保持工作流不中断的技巧与窍门**：用户讨论了 **unsloth 安装** 的故障排除问题，并分享了 **Kaggle** 和 **starsupernova** 的安装说明，以克服诸如 **CUDA not linked** 错误，以及与 torch 版本、**flash-attn** 和 **xformers** 的依赖问题。

- **用于生产环境的 Adapter 合并**：有一段关于合并 **QLoRA** 的 adapter 的正确方法的对话，以及是使用**朴素合并（naive merging）**还是在合并前进行**反量化（dequantization）**等更复杂的过程。**Starsupernova** 提供了关于将模型保存为 **16bit** 以便合并到 vLLM 或 GGUF 的 wiki 链接。

- **对话格式转换与自定义训练**：一位用户分享了一个脚本，用于将纯文本对话数据集转换为 ShareGPT 格式，以便准备模拟个人聊天风格，同时建议在转换后使用 **Unsloth** 的 notebook 进行自定义训练。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://huggingface.co/ura-hcmut/ura-llama-7b">ura-hcmut/ura-llama-7b · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/ghost-x/ghost-7b-v0.9.0">ghost-x/ghost-7b-v0.9.0 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only">Supervised Fine-tuning Trainer</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">Home</a>：提速 2-5 倍，节省 80% 显存的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>：提速 2-5 倍，节省 80% 显存的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.co">GitHub: Let’s build from here</a>：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专业人士一样审查代码，跟踪错误和功能...</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>：提速 2-5 倍，节省 80% 显存的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/akameswa/CodeGenerationMoE/blob/main/code/finetune.ipynb">CodeGenerationMoE/code/finetune.ipynb at main · akameswa/CodeGenerationMoE</a>：用于代码生成的 Mixture of Expert 模型。通过在 GitHub 上创建账户来为 akameswa/CodeGenerationMoE 的开发做出贡献。</li><li><a href="https://download.pytorch.org/whl/cu118">no title found</a>：未找到描述</li><li><a href="https://github.com/huggingface/datasets/issues/6753">Type error when importing datasets on Kaggle · Issue #6753 · huggingface/datasets</a>：描述该 Bug。当尝试运行 import datasets print(datasets.__version__) 时，会产生以下错误 TypeError: expected string or bytes-like object。看起来它找不到 val...</li><li><a href="https://github.com/facebookresearch/xformers#installing-xformers)">GitHub - facebookresearch/xformers: Hackable and optimized Transformers building blocks, supporting a composable construction.</a>：可定制且经过优化的 Transformers 构建模块，支持组合式构建。 - facebookresearch/xformers</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning</a>：提速 2-5 倍，节省 80% 显存的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth)</a>：未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1228316423558926376)** (54 条消息🔥): 

- **低资源语言增强技术**：成员们讨论了增强低资源语言数据集的方法，提到了使用翻译数据和 HuggingFace 上的可用资源。
- **EEVE-Korean 模型数据效率主张**：一位成员引用了一篇关于 [EEVE-Korean-v1.0](https://arxiv.org/abs/2402.14714) 的论文，强调了一种基于以英文为中心的大语言模型 (LLM) 的高效词汇扩展方法，据称仅需 20 亿个 token 即可显著提升非英语语言能力。
- **打包在 Podman 容器中的 Unsloth AI**：分享了一个视频，演示了容器化生成式 AI 应用程序的 podman 构建，包括 Unsloth AI，能够部署在本地或云服务商：[Podman Build - 针对成本敏感型 AI](https://youtu.be/3iEhFKIDXp0)。
- **通过 Ghost 7B Alpha 增强 LLM**：讨论强调了 Ghost 7B Alpha 在推理能力上优于 ChatGPT 等模型的优势。进一步的讨论集中在该模型的微调方法上，其中包括困难的推理问题，暗示了一种类似于 orca 的方法，且没有扩展 tokenizer 词汇表。
- **初步基准测试的有趣结果**：一位成员分享了他们的初步基准测试结果，引发了其他参与者的积极反馈和肯定，表明性能前景广阔。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.14714">Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models</a>：该报告介绍了 \texttt{EEVE-Korean-v1.0}，这是一个针对韩语适配的大语言模型，在英语和韩语文本理解方面表现出卓越的能力。基于最近的高...</li><li><a href="https://tenor.com/view/sloth-crawling-slow-gif-9915689">Sloth Crawling GIF - Sloth Crawling Slow - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/ghostx_ai">Tweet from undefined</a>：未找到描述</li><li><a href="https://huggingface.co/ghost-x">ghost-x (Ghost X)</a>：未找到描述</li><li><a href="https://youtu.be/3iEhFKIDXp0?si=HnDtELfgo4Q3cBOH">Podman Build- For cost-conscious AI</a>：关于 @podman 如何构建容器化生成式 AI 应用的快速演示，这些应用可以部署在本地或你首选的云服务商。这种选择是...
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1229450843569520811)** (1 messages): 

- **Pile-T5 亮相**：EleutherAI 发布了 **Pile-T5**，这是一款 T5 模型变体，在来自 [*Pile*](https://blog.eleuther.ai/pile-t5/) 的 2 万亿 token 上使用 **LLAMA tokenizer** 进行了训练，在 *SuperGLUE* 和 *MMLU* 等基准测试中表现出显著的性能提升。该模型在代码相关任务中也表现更好，并为 HF 和 T5x 版本提供了中间检查点（intermediate checkpoints）。
  
- **模型细节与贡献**：**Pile-T5** 经过了 200 万步的精心训练，并结合了*原始 span corruption 方法*以改进微调效果。正如公告中所确认的，该项目是 EleutherAI 成员共同努力的结果。

- **面向社区开源**：所有与 **Pile-T5** 相关的资源，包括模型权重和训练脚本，均已在 [GitHub 上开源](https://github.com/EleutherAI/improved-t5)。这包括中间检查点，允许社区进行实验并在此基础上开展工作。

- **Twitter 上的致谢与链接**：**Pile-T5** 的发布和开源已在 [Twitter](https://x.com/arankomatsuzaki/status/1779891910871490856) 上公布，邀请社区阅读博客文章、下载权重并为仓库做出贡献。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://blog.eleuther.ai/pile-t5/">Pile-T5</a>：在 Pile 上训练的 T5</li><li><a href="https://github.com/EleutherAI/improved-t5">GitHub - EleutherAI/improved-t5: Experiments for efforts to train a new and improved t5</a>：训练全新且改进的 T5 的实验工作 - EleutherAI/improved-t5</li><li><a href="https://x.com/arankomatsuzaki/status/1779891910871490856">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>：🚀 介绍 Pile-T5！🔗 我们 (EleutherAI) 非常激动地开源了我们最新的 T5 模型，该模型使用 Llama tokenizer 在来自 Pile 的 2T token 上训练。✨ 具有中间检查点和...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1228268846008504444)** (123 messages🔥🔥): 

- **评估集（Evals）汇编的协作**：一位参与者请求在编译和解决评估列表方面提供协助和协作，并将感兴趣的成员引导至一份共享的 [Google 文档](https://docs.google.com/document/d/1qt7GjbrFToxSIUKC9nWccvHZN9LnO8r6myMAFUX5SVQ/edit?usp=sharing)。

- **NeurIPS 高中赛道征集队友**：一位成员正在寻找潜在队友参加 [NeurIPS 的新高中赛道](https://neurips.cc/Conferences/2024/CallforHighSchoolProjects)，并邀请感兴趣的人士与其联系。

- **模型生成技术讨论**：一场关于技术细节的讨论展开了，即为什么对于 Phi-2 等模型，逐个生成 token 与使用 Hugging Face 的 `generate()` 一次性生成所有 token 会产生略有不同的结果。多位社区成员提出了假设，并提供了相关 GitHub [issues](https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535) 和代码 [片段](https://huggingface.co/microsoft/phi-2/blob/main/modeling_phi.py) 的链接。

- **AI 音乐生成展示**：一位成员讨论了他们在 AI 生成音乐方面的进展，并提出可以根据要求生成任何独奏钢琴曲的续写，并暗示在 audio-models 频道即将发布一篇重大帖子。

- **AGI 路径的社区项目提案与讨论**：关于一名成员提出的基于 STRIPS 调度器创建“友好 AGI”的方案，展开了广泛的辩论。尽管存在一些质疑，提案者还是提供了其[概念论文链接](https://arxiv.org/pdf/2304.11477)并详细阐述了该想法，理由是迫切需要 AI 对齐（Alignment）和安全性的替代方案。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://us05web.zoom.us/j/87189451768?pwd=6PlOsbZ2AbaaIhwvaa6kjyvn0NelT2.1">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。Zoom ...</li><li><a href="https://docs.google.com/document/d/1qt7GjbrFToxSIUKC9nWccvHZN9LnO8r6myMAFUX5SVQ/edit?usp=sharing">[你希望我们/你处理/探索/解决的评估列表]</a>：未找到描述</li><li><a href="https://www.latent.space/p/transformers-math#details>">训练 LLMs 的数学 —— 与 Eleuther AI 的 Quentin Anthony 交流</a>：现在收听 | 拆解爆火的 Transformers Math 101 文章，以及针对基于 Transformers 架构的高性能分布式训练（或者说“我如何学会停止空谈并开始...”）</li><li><a href="https://github.com/elicit/machine-learning-list">GitHub - elicit/machine-learning-list</a>：通过创建 GitHub 账号为 elicit/machine-learning-list 的开发做出贡献。</li><li><a href="https://youtu.be/e1UgzSTicuY?si=0vS7ImdlVKC2QJJd">为什么我立即离开我的公司 (Stability AI) 与 Emad Mostaque | 第 93 集</a>：在本集中，Peter 和 Emad 讨论了 Emad 辞去 StabilityAI CEO 职务、他下一步进入去中心化 AI 领域的计划，以及为什么现在如此紧迫...</li><li><a href="https://github.com/huggingface/transformers/issues/25420#issuecomment-177531753">Llama（原始）模型中 KV Caching 的潜在 Bug · Issue #25420 · huggingface/transformers</a>：系统信息 transformers==4.31.0 huggingface_hub 版本：0.15.1 平台：Linux-5.15.0-78-generic-x86_64-with-glibc2.35 Python 版本：3.10.12 运行于 iPython ?: 否 运行于 notebook ?: 否 R...</li><li><a href="https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535">Llama（原始）模型中 KV Caching 的潜在 Bug · Issue #25420 · huggingface/transformers</a>：系统信息 transformers==4.31.0 huggingface_hub 版本：0.15.1 平台：Linux-5.15.0-78-generic-x86_64-with-glibc2.35 Python 版本：3.10.12 运行于 iPython ?: 否 运行于 notebook ?: 否 R...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1228290369658294293)** (534 条消息🔥🔥🔥): 

- **探索 MoE 与 Dense 模型容量**：关于混合专家模型 (MoE) 与稠密 (Dense) Transformer 模型的相对性能和优势正在进行辩论，重点在于在不受 VRAM 限制时 MoE 是否绝对更好，以及在相同的参数预算下 Dense 模型是否实际上表现更好。

- **Token Unigram 模型异常**：一篇新讨论的论文提出，对于在没有分词（Tokenization）的高阶马尔可夫过程上训练的 Transformer，它们无法学习到正确的分布，而是默认退化为 Unigram 模型。

- **使用 CLIP 进行 Deep Dreaming**：有人对将 Deep Dream 技术应用于 CLIP 模型产生了好奇，并建议 CLIP 的因果语言模型 (Causal LM) 结构可能会为图像的“梦境化”提供有意义的损失信号。

- **重新审视 DenseNet 的实用性**：最近的一篇论文让 DenseNets 重新焕发生机，认为过时的训练方法和设计元素之前可能掩盖了它们的潜力，现在理论上已经超越了 Swin Transformer 和 ConvNeXt 等现代架构。

- **语言模型的研究空白**：参与者讨论了专门比较 Dense 和 MoE Transformer 学习内容的平衡研究的缺乏，强调了在性能指标之外理解模型差异方面的空白。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://arxiv.org/abs/2404.07965">Rho-1: Not All Tokens Are What You Need</a>：以往的语言模型预训练方法对所有训练 Token 统一应用下一个 Token 预测损失。挑战这一常规，我们认为“语料库中的并非所有 Token 都是平等的...”</li><li><a href="https://storm.genie.stanford.edu/">Streamlit</a>：未找到描述</li><li><a href="https://arxiv.org/abs/1701.06538">Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer</a>：神经网络吸收信息的能力受其参数数量的限制。条件计算（Conditional computation），即网络的一部分针对每个样本按需激活，已被提出...</li><li><a href="http://arxiv.org/abs/2404.08636">Probing the 3D Awareness of Visual Foundation Models</a>：大规模预训练的最新进展产生了具有强大能力的视觉基础模型。最近的模型不仅可以将其训练任务泛化到任意图像，它们的内部...</li><li><a href="https://arxiv.org/abs/2404.07177">Scaling Laws for Data Filtering -- Data Curation cannot be Compute Agnostic</a>：视觉语言模型 (VLMs) 在精心策划的网络数据集上经过了数千个 GPU 小时的训练。近年来，随着多项研究开发出相关策略，数据策展（Data curation）变得日益重要...</li><li><a href="https://colab.research.google.com/drive/10rSQ_D80M4jn0-A1yv_2fOMilXdhYaVo">Google Colaboratory</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.07647">Why do small language models underperform? Studying Language Model Saturation via the Softmax Bottleneck</a>：语言建模的最新进展在于在极大规模的网络挖掘文本语料库上预训练高度参数化的神经网络。此类模型的训练和推理在实践中可能成本高昂...</li><li><a href="https://arxiv.org/abs/2403.19588">DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs</a>：本文复兴了密集连接卷积网络 (DenseNets)，并揭示了其在主流 ResNet 风格架构之上被低估的有效性。我们认为 DenseNets 的潜力曾被忽视...</li><li><a href="https://arxiv.org/abs/2309.02591">Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning</a>：我们介绍了 CM3Leon（发音为 "Chameleon"），这是一种检索增强的、基于 Token 的、仅解码器（decoder-only）的多模态语言模型，能够生成和填充文本及图像。CM3Leon 使用了...</li><li><a href="https://lilianweng.github.io/posts/2024-04-12-diffusion-video/">Diffusion Models for Video Generation</a>：扩散模型在过去几年中在图像合成方面展示了强大的结果。现在研究界已经开始致力于一项更艰巨的任务——将其用于视频生成。这项任务本身...</li><li><a href="https://arxiv.org/abs/2403.01643">You Need to Pay Better Attention</a>：我们引入了三种新的注意力机制，它们在效率和学习能力方面优于标准的多头注意力（multi-head attention），从而提高了性能和更广泛的部署能力...</li><li><a href="https://arxiv.org/abs/2404.08335">Toward a Theory of Tokenization in LLMs</a>：虽然已有大量研究试图绕过语言建模中的分词（tokenization）过程（Clark et al., 2022; Xue et al., 2022），但目前的共识是，它是必要的初始步骤...</li><li><a href="http://arxiv.org/abs/2302.09057">Consistent Diffusion Models: Mitigating Sampling Drift by Learning to be Consistent</a>：不完美的得分匹配（score-matching）会导致扩散模型的训练分布与采样分布之间出现偏差。由于生成过程的递归性质，前几步的错误会产生...</li><li><a href="https://en.wikipedia.org/wiki/Predictive_coding">Predictive coding - Wikipedia</a>：未找到描述</li><li><a href="https://fixupx.com/Birchlabs/status/1642680652377075712">Tweet from Birchlabs (@Birchlabs)</a>：扩散模型难以生成训练分布之外尺寸的图像，在 #stablediffusion 中通过从平均值外推增加 self-attn softmax 分母，在无需重新训练的情况下解决了此问题...</li><li><a href="https://fixupx.com/Birchlabs/status/1660438858675224576">Tweet from Birchlabs (@Birchlabs)</a>：我将 #stablediffusion 变成了一个声音转图像模型，这里它描绘了 fire.wav。这基本上是 CLIP 引导，只不过使用了 Meta 新的 ImageBind 模型。它接受文本、图像、音频、视频、深度...</li><li><a href="https://arxiv.org/abs/2402.14800">Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models</a>：大语言模型 (LLMs) 发展中的一个关键进步是混合专家 (MoE) LLMs 的出现。与传统的 LLMs 相比，MoE LLMs 可以通过...实现更高的性能。</li><li><a href="https://proceedings.mlr.press/v139/wies21a.html">Which transform</a>

<li><a href="https://arxiv.org/abs/2109.10220">哪种架构适合我的数据？Self-attention 中的词汇瓶颈</a>：在自然语言处理领域取得成功后，Transformer 架构正成为许多领域的行业标准。在新型模态上部署它们的一个障碍是...</li><li><a href="https://smallandroidphone.com/">我想要一部小屏 Android 手机！</a>：你是否觉得所有的 Android 手机都太大了？我同意！我们需要共同努力，说服厂商再次制造小屏 Android 手机。</li><li><a href="https://discord.gg/AJGjXd5q">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://arxiv.org/abs/1412.0035">通过反转来理解深度图像表示</a>：从 SIFT 和视觉词袋（Bag of Visual Words）到卷积神经网络（CNNs），图像表示几乎是任何图像理解系统的关键组成部分。然而，我们对它们的理解...</li><li><a href="https://arxiv.org/abs/1812.02903">应用联邦学习：改进 Google 键盘查询建议</a>：Federated learning 是一种分布式的机器学习形式，其训练数据和模型训练都是去中心化的。在本文中，我们在一个商业化的、全球规模的系统中使用 Federated learning...</li><li><a href="https://fixupx.com/fly51fly/status/1779872116458020991">来自 fly51fly (@fly51fly) 的推文</a>：[CL] 迈向 LLMs 中的 Tokenization 理论 N Rajaraman, J Jiao, K Ramchandran [加州大学伯克利分校] (2024) https://arxiv.org/abs/2404.08335 - 在某些简单的高阶马尔可夫数据上训练的 Transformers...</li><li><a href="https://tenor.com/view/bait-thats-bait-tom-hardy-mad-max-gif-5055384">Bait Thats Bait GIF - Bait Thats Bait Tom Hardy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://arxiv.org/abs/2310.05209">基于 RoPE 外推的 Scaling Laws</a>：基于 Rotary Position Embedding 的大语言模型（LLMs）的外推能力是目前备受关注的话题。解决外推问题的主流方法是...</li><li><a href="https://github.com/Birch-san/diffusers-play/blob/27f359c121fc5d82f305927c4f5e8a3ad963440e/src/helpers/attention/logit_scaling_attn.py#L80>">diffusers-play/src/helpers/attention/logit_scaling_attn.py 在 27f359c121fc5d82f305927c4f5e8a3ad963440e · Birch-san/diffusers-play</a>：用于探索 k-diffusion 和 diffusers 的仓库，并可在其中测试对上述软件包的更改。- Birch-san/diffusers-play</li><li><a href="https://github.com/ClashLuke/TrueGrad/blob/main/truegrad/optim.py#L378>)">TrueGrad/truegrad/optim.py 在 main · ClashLuke/TrueGrad</a>：TrueGrad 优化器的 PyTorch 接口。通过在 GitHub 上创建账号来为 ClashLuke/TrueGrad 的开发做出贡献。</li><li><a href="https://github.com/Birch-san/diffusers-play/blob/27f359c121fc5d82f305927c4f5e8a3ad963440e/src/helpers/attention/wacky_softmax_attn.py#L233>">diffusers-play/src/helpers/attention/wacky_softmax_attn.py 在 27f359c121fc5d82f305927c4f5e8a3ad963440e · Birch-san/diffusers-play</a>：用于探索 k-diffusion 和 diffusers 的仓库，并可在其中测试对上述软件包的更改。- Birch-san/diffusers-play</li><li><a href="https://arxiv.org/abs/2404.05405">语言模型的物理学：第 3.3 部分，知识容量 Scaling Laws</a>：Scaling laws 描述了语言模型规模与其能力之间的关系。与以往通过 loss 或基准测试评估模型能力的研究不同，我们估计了...</li><li><a href="https://arxiv.org/abs/2401.16380">改写网络：一种计算和数据高效的语言建模方案</a>：大语言模型是在海量的网络抓取数据上训练的，这些数据通常是无结构的、有噪声的且表述不清。目前的 Scaling laws 表明，从这类数据中学习需要大量的...</li><li><a href="https://www.tensorflow.org/api_docs/python/tf/image/total_variation">未找到标题</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2309.16620">残差网络中的深度超参数迁移：动力学与 Scaling Limit</a>：深度学习中超参数调优的成本随着模型规模的增加而上升，这促使从业者寻找使用较小网络作为代理的新调优方法。其中一个提议是使用 $μ$P...</li><li><a href="https://www.youtube.com/watch?v=xK4tyMqthYk">Mufan Li - 作为深度随机过程的无限深度神经网络</a>：摘要：神经网络研究的最新进展主要集中在无限宽度架构上，然而建模中固有的复杂性...</li><li><a href="https://www.unihertz.com/collections/jelly-series">Jelly 系列</a>：立即从 Unihertz 购买 Jelly 系列智能手机及其配件！这些运行 Android 系统的掌上智能手机可以...</li>

满足您的日常需求，同时为您的口袋或手提包节省更多空间！T...</li><li><a href="https://arxiv.org/html/2403.12963v1">FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis</a>：未找到描述</li><li><a href="https://pure.nsu.ru/portal/ru/">Обзор исследований</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1228262144349311096)** (10 messages🔥): 

- **首个数据过滤 Scaling Laws 揭晓**：@pratyushmaini 的一条推文宣布开发了首个用于数据策展（Data Curation）的 Scaling Laws，强调它*不能*与计算无关。该研究发表于 CVPR 2024，由 @goyalsachin007, @zacharylipton, @AdtRaghunathan 和 @zicokolter 共同撰写，详见其论文 [此处](https://arxiv.org/abs/2404.07177)。

- **数据过滤方法中的隐式熵**：一位成员认为，这篇关于异构且有限的网络数据 Scaling Laws 的论文在没有明确提及熵方法的情况下实现了实证研究，这令人印象深刻，暗示该概念是研究的基础。

- **讨论新研究中熵的隐秘性质**：有一段关于研究中熵与编码方案/模型之间内在联系的对话，建议在进一步评论之前期待更深入的分析。

- **寻找熵在效用定义中的作用**：一位成员观察到，之前讨论的论文可能隐含地将熵重新定义为“效用（utility）”，尽管这导致了一些非常规的概念化方式。这表明实证方法可能掩盖了对熵的基础性依赖。

**提及的链接**：<a href="https://x.com/pratyushmaini/status/1778577153107570770">来自 Pratyush Maini (@pratyushmaini) 的推文</a>：1/ 🥁Scaling Laws for Data Filtering 🥁 TLDR: Data Curation *cannot* be compute agnostic! 在我们的 #CVPR2024 论文中，我们开发了首个针对异构且有限的网络数据的 Scaling Laws。w/@goyalsach...

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1229062395541065809)** (6 messages): 

- **Transformer In-Context Learning 的创新**：一篇新论文介绍了一个 JAX 工具包，该工具包有助于在 Transformer 模型训练期间进行**因果干预（causal manipulations）**。该工具包的“钳制（clamping）”功能揭示了**对 In-Context Learning 至关重要的特定子电路**和感应头（induction heads），同时还强调了独特的学习动态，即某些组件的钳制会改变整体训练行为，并有助于避免鞍点和相变。[阅读更多研究内容](https://x.com/scychan_brains/status/1779357918737158364?s=46)。

- **用语言解释机器学习机制**：Google 的 **Patchscopes** 框架旨在通过利用 LLM 的语言能力使其隐藏表示更易于理解，从而统一解释大语言模型（LLMs）的方法。该倡议可以增强模型的透明度，特别是通过创建模型内部运作的**自然语言解释**来理解易出错的情况。[了解 Patchscopes](http://research.google/blog/patchscopes-a-unifying-framework-for-inspecting-hidden-representations-of-language-models/)。

- **光遗传学（Optogenetics）启发 AI 研究**：AI 研究背景下提到的“optogenetic”一词表示受到一种生物技术的启发，该技术利用光控制细胞活动，提供对神经元的精确控制，从而阐明决策过程中的路径。维基百科链接提供了关于 **Optogenetics** 的更详细见解。[探索光遗传学技术](https://en.m.wikipedia.org/wiki/Optogenetics)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/scychan_brains/status/1779357918737158364?s=46">来自 Stephanie Chan (@scychan_brains) 的推文</a>：我们的新论文深入探讨了 Transformer In-Context Learning (ICL) 的电路和训练动态 🥳 关键亮点包括 1️⃣ 一个新的开源 JAX 工具包，可通过以下方式实现因果干预...</li><li><a href="https://en.m.wikipedia.org/wiki/Optogenetics">光遗传学 - 维基百科</a>：未找到描述</li><li><a href="http://research.google/blog/patchscopes-a-unifying-framework-for-inspecting-hidden-representations-of-language-models/">Patchscopes：一个用于检查语言模型隐藏表示的统一框架</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1229121918381330612)** (1 messages):

- **为 GPQA 寻求最佳 num_fewshot**：一位用户正在针对 GPQA 对 **Mistral 7B** 进行测试，寻求关于理想 `num_fewshot` 设置的建议。Mistral 7B 在温度为 0 和 1 时表现不佳，成功率低于 10%，但尚未找到既定的基准 `num_fewshot`。

- **在 EQ Bench 中独立运行子任务**：同一位用户询问如何在 **EQ Bench** 中运行单个子任务（特别是 **creative writing**），而不是运行所有任务。他们参考了 [EQ Bench GitHub 页面](https://github.com/EQ-bench/EQ-Bench)上的说明，但需要关于隔离特定子任务的指导。

**提及的链接**：<a href="https://github.com/EQ-bench/EQ-Bench">GitHub - EQ-bench/EQ-Bench: A benchmark for emotional intelligence in large language models</a>：大型语言模型情感智能基准测试 - EQ-bench/EQ-Bench

---

**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1228274522034012200)** (23 条消息🔥): 

- **为 GPT-NeoX 寻求企业级 CLA**：一位用户请求为贡献 **GPT-NeoX** 项目提供**企业级 CLA**，特别是针对具有融合算子 (fused kernels) 和 fp8 等增强功能的 **TE integration**。Stellaathena 作出回应，表示在提供具体要求后可以编写**定制 CLA**。
  
- **调查 NeoX Embeddings 异常**：对 **NeoX embeddings** 的调查显示，在词表大小之外，权重衰减 (weight decay) 并没有导致数值趋于零，这与其他模型不同。讨论围绕可能的解释展开，建议这可能是由于权重衰减的实现方式或独特的初始化方法所致。

- **关于 GPU 效率技巧的澄清**：Stellaathena 澄清说，在 **NeoX** 中，嵌入矩阵被特意放大以提高 **GPU efficiency**，但实际词表大小之外的数值只是占位符，不应进行分析。

- **关于 NeoX 的 Rotary Embedding 见解**：分享了一项观察结果，即 **NeoX** 仅将其 25% 的嵌入应用于 **rotary embeddings**，这使其与 **Pythia** 等模型有所不同。

- **权重衰减实现细节**：讨论表明，**weight decay** 可能只影响已激活（接收到梯度）的权重，这意味着 **NeoX** 模型中未使用的哑标记 (dummy tokens) 不会受到影响，这可能解释了它们的非零值。

---

**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1228523999412355124)** (1 条消息): 

- **Infini-attention 引起关注**：一位成员强调了 Google 最近发表的题为 "Infini-attention" 的论文，并对其内容表示关注。该论文可通过 [Infini-attention Paper](https://arxiv.org/pdf/2404.07143.pdf) 访问。

---

**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1228262914981498920)** (21 条消息🔥): 

- **GitHub 上的狭义相对论**：分享了一个名为 "special_relativity_greg_egan.md" 的 GitHub Gist，其中包含与狭义相对论相关的代码、笔记和片段。可在 [fullstack6209's GitHub](https://gist.github.com/fullstackwebdev/34ccaf0fb79677890c8f93a795f8472a) 查看。

- **向量搜索速度对比**：给出了向量搜索查询的性能对比：在 10 万个向量的测试中，FFVec 达到 0.44ms，FAISS Flat 为 10.58ms，FAISS HNSW 为 1.81ms，FAISS IVFPQ 为 0.36ms。

- **创办开源研究实验室**：有人请求指导如何创建一个类似于 Carper AI 或 Nous 的开源研究实验室。建议从一个简单的 GitHub 仓库和 Discord 服务器开始启动该计划。

- **将 YouTube 视频转换为博客文章**：介绍了一个名为 "youtube2blog" 的新工具，能够使用 Mixtral 8x7b 和 Deepgram 将 YouTube 视频转换为博客文章。在 GitHub 上查找：[S4mpl3r/youtube2blog](https://github.com/S4mpl3r/youtube2blog)。

- **构建二进制量化友好型嵌入**：讨论了一个用于创建二进制量化友好型嵌入 (binary-quantization friendly embeddings) 的工具，并提供了一个相关博客文章链接，解释了此类嵌入在大规模语义搜索应用中的重要性。分享了 GitHub 仓库 [carsonpo/ffvec](https://github.com/carsonpo/ffvec)，以及旨在生成这些嵌入的 HuggingFace 模型 [carsonpoole/binary-embeddings](https://huggingface.co/carsonpoole/binary-embeddings)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://txt.cohere.com/int8-binary-embeddings/">Cohere int8 &amp; binary Embeddings - 将您的向量数据库扩展到大型数据集</a>：Cohere Embed 现在原生支持 int8 和 binary embeddings，以降低内存成本。</li><li><a href="https://huggingface.co/carsonpoole/binary-embeddings">carsonpoole/binary-embeddings · Hugging Face</a>：未找到描述</li><li><a href="https://gist.github.com/fullstackwebdev/34ccaf0fb79677890c8f93a795f8472a">special_relativity_greg_egan.md</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/S4mpl3r/youtube2blog">GitHub - S4mpl3r/youtube2blog：使用 Mixtral 8x7b 和 Deepgram 将任何 Youtube 视频转换为精美的博客文章。</a>：使用 Mixtral 8x7b 和 Deepgram 将任何 Youtube 视频转换为精美的博客文章。- S4mpl3r/youtube2blog</li><li><a href="https://github.com/carsonpo/ffvec">GitHub - carsonpo/ffvec</a>：通过在 GitHub 上创建账户来为 carsonpo/ffvec 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1228390362536739007)** (29 条消息🔥): 

- **三足机器人椅子摇摆进入视野**：一名成员分享了一个[机器人项目链接](https://shin0805.github.io/chair-type-tripedal-robot/)，涉及一把会走路的三脚椅子，随附的视频为对话增添了魅力。该作品将在 RoboSoft2024 上展示，并引发了成员们的幽默讨论，提到了它的“可爱”以及对虐待椅子的轻松调侃。

- **重新思考 AI 基准测试**：成员们达成明确共识，认为当前的 AI 基准测试存在缺陷或容易被操纵，引发了关于评估 AI 模型的新方法（不可作弊）以及特定领域任务评估效用的讨论。

- **以税务基准测试为例的潜在作弊风险**：一位成员指出了一个[税务基准测试](https://www.vals.ai/taxeval)作为当前的评估方法，但很快遭到批评，认为如果基于类测试题，可能很容易被操纵。

- **衡量 AI 连贯性的递归方法**：提出了一个评估“通用智能”的想法，即让模型作为第三方旁观者加入叙事，然后让模型对此进行评论，形成一个潜在的无限循环，以观察连贯性能维持多久。

- **AGI 秘诀辩论与工具的实用性**：分享了一个来自 InfraNodus 的视频，关于使用[知识图谱来提示 LLM](https://www.youtube.com/watch?v=uwIyYZ3En1A)，被认为是 AGI 开发的潜在重要资源，但在其实际可用性方面收到了褒贬不一的评价。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.vals.ai/taxeval">Vals.ai: LegalBench</a>：使用开源数据集 LegalBench，我们对 13 个流行的开源和闭源模型进行了基准测试，以查看它们在法律任务上的表现。随着新模型的发展，该基准测试将保持更新。</li><li><a href="https://x.com/shin0805__/status/1777992583396131246?s=46">来自 Shintaro Inoue / 井上信多郎 (@shin0805__) 的推文</a>：发布了再现《铃芽之旅》中出现的三脚椅子的机器人设计，以及通过强化学习生成步态的论文！下周将在美国举行的 RoboSoft2024 上发表！网站 - https://shin0805.github.io/chair-type-tripedal-robot/ #すずめの戸締まり</li><li><a href="https://github.com/nus-apr/auto-code-rover">GitHub - nus-apr/auto-code-rover：一个具备项目结构感知能力的自主软件工程师，旨在实现自主程序改进。在完整的 SWE-bench 中解决了 15.95% 的任务。</a>：一个具备项目结构感知能力的自主软件工程师，旨在实现自主程序改进。在完整的 SWE-bench 中解决了 15.95% 的任务 - nus-apr/auto-code-rover</li><li><a href="https://www.youtube.com/watch?v=uwIyYZ3En1A">如何使用 InfraNodus 通过知识图谱提示 LLM</a>：在本视频中，我将向您展示如何使用 https://infranodus.com 通过知识图谱提示 LLM。我将以 GPT-4 为例，但您也可以使用...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1228268978095263877)** (382 条消息🔥🔥):

- **超级对齐（Superalignment）目标与现实检查**：对话围绕在 2027 年实现超级对齐展开，并推测这一时间表是否暗示 AGI 可能在此之前出现。成员们对其可行性持怀疑态度，并讨论了该问题的结论不确定性。
- **关于 GPT-4 表现优于 Command-R Plus 的争论**：讨论围绕 **LMSys Arena** 展开，据称在过滤排名中的**排除项（exclusions）**后，Command-R Plus 的表现优于所有版本的 GPT-4，这引发了关于评估方法和性能指标的辩论。
- **对误用 Nous Research 名称的担忧**：在关于一个名为 OpenML 的无关代币的讨论中，澄清了 Nous Research 与该代币没有任何关联，并且他们已要求从任何相关材料中删除 Nous Research 的名称。
- **个人项目与实验讨论**：成员们分享了各种个人 AI 项目经验，例如在单台笔记本电脑上用三天时间对语言模型进行指令微调（instruction tuning），以及创建 RAG 数据库。讨论重点在于障碍、成果和最佳实践。
- **ChatGPT Plus 的易用性问题**：讨论了 ChatGPT Plus 每 3 小时 40 条消息限制的实用性，不同用户表达了他们如何管理或规避这一限制，一些人对这些约束表示沮丧。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.marktechpost.com/2024/04/13/google-ai-introduces-codeclm-a-machine-learning-framework-fo">未找到标题</a>：未找到描述</li><li><a href="https://mihaiii.github.io/semantic-autocomplete/">SemanticAutocomplete 演示</a>：未找到描述</li><li><a href="https://www.marktechpost.com/2024/04/13/google-ai-introduces-codeclm-a-machine-learning-framework-for-generating-high-quality-synthetic-data-for-llm-alignment/?amp">未找到标题</a>：未找到描述</li><li><a href="https://x.com/osanseviero/status/1778816205727424884?s=46">来自 Omar Sanseviero (@osanseviero) 的推文</a>：欢迎 Zephyr 141B 加入 Hugging Chat🔥 🎉一个 Mixtral-8x22B 微调版本 ⚡️使用 TGI 实现超快生成 🤗完全开源（从数据到 UI） https://huggingface.co/chat/models/HuggingFaceH4/zeph...</li><li><a href="https://huggingface.co/Vezora/Mistral-22B-v0.1">Vezora/Mistral-22B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/karpathy/status/1647278857601564672">来自 Andrej Karpathy (@karpathy) 的推文</a>：@dsmilkov 没有关注但听起来很有趣。“训练一个带有样本权重的线性模型来进行类别平衡”...？</li><li><a href="https://www.ora.io/app/imo/olm">ORA</a>：未找到描述</li><li><a href="https://poe.com/Mixtral8x22b-Inst-FW">Mixtral8x22b-Inst-FW - Poe</a>：来自 Mistral AI 的 Mixtral 8x22B Mixture-of-Experts 基础模型，由 Fireworks.AI 进行微调。https://fireworks.ai/models/fireworks/mixtral-8x22b-instruct-preview</li><li><a href="https://huggingface.co/chat/models/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1/">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 - HuggingChat</a>：在 HuggingChat 中使用 HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1</li><li><a href="https://gist.github.com/Hellisotherpeople/45c619ee22aac6865ca4bb328eb58faf#file-prompt_weighting_llm-py">你可能不知道如何进行提示工程（Prompt Engineering），让我来教你。</a>：你可能不知道如何进行提示工程（Prompt Engineering），让我来教你。 - blog.md</li><li><a href="https://gist.github.com/Hellisotherpeople/45c619ee22aac6865ca4bb328eb58faf#file-prompt_weighting_llm">你可能不知道如何进行提示工程（Prompt Engineering），让我来教你。</a>：你可能不知道如何进行提示工程（Prompt Engineering），让我来教你。 - blog.md</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/mixtral-8x22b-qlora-fsdp.yml">axolotl/examples/mistral/mixtral-8x22b-qlora-fsdp.yml at main · OpenAccess-AI-Collective/axolotl</a>：尽管向 axolotl 提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1228304489748889663)** (68 条消息🔥🔥): 

- **寻找专注于健康的 LLM**：成员们讨论了医疗保健和医学领域的模型（LLM）现状。其中一个重点资源是 arXiv 上的一篇论文，作者包括 [Subhabrata Mukherjee](https://arxiv.org/abs/2403.13313)，该论文探讨了医学主题。

- **希腊语 LLMs 引发讨论**：讨论围绕为希腊语微调 LLMs 展开，成员们争论了对低资源语言进行预训练的必要性。参考资料包括 [Meltemi 希腊语模型](https://huggingface.co/ilsp/Meltemi-7B-v1)，并建议加入 [Unsloth Discord](https://discord.com/invite/WSaMctCn) 等社区进行协作模型改进。

- **探索 RAG 中的引用**：成员们询问了检索增强生成 (RAG) 中最先进的引用方法，建议指向 **cmd r+** 等模型以执行行级引用。

- **Capybara 数据集和 Amplify-Instruct 方法**：对 Capybara 数据集和 "Amplify-Instruct" 技术的查询引导一位成员表示，虽然技术报告尚未发布，但感兴趣的各方可以阅读 Capybara 数据集的数据集卡片以获取部分细节。

- **多模态 LLM Reka 的比较**：围绕多模态 LLM Reka Core 的能力展开了讨论，并将其与其他领先模型进行了比较。分享了一个关于 Reka 性能亮点的链接，但对于封闭模型与开源模型的比较（无论基准测试如何）存在怀疑态度。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1nT4T7pCr_dWfgqAFyTNmhQTwoUOULpo1?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2403.13313">Polaris: A Safety-focused LLM Constellation Architecture for Healthcare</a>：我们开发了 Polaris，这是首个专注于安全的 LLM 星座架构，用于实时患者-AI 医疗对话。与以往专注于问答等任务的医疗 LLM 工作不同，我们的工作...</li><li><a href="https://www.reka.ai/news/reka-core-our-frontier-class-multimodal-language-model">Reka Core: Our Frontier Class Multimodal Language Model &mdash; Reka AI</a>：发布 Reka Core，我们的前沿级多模态语言模型！</li><li><a href="https://huggingface.co/ilsp/Meltemi-7B-v1">ilsp/Meltemi-7B-v1 · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c4mgda/inference_issue_using_llamacpp_and_openhermes/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2304.08177">Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca</a>：大型语言模型 (LLMs)，如 ChatGPT 和 GPT-4，极大地改变了自然语言处理研究，并展示了迈向通用人工智能 (AGI) 的巨大进步。</li><li><a href="https://huggingface.co/ghost-x/ghost-7b-v0.9.1">ghost-x/ghost-7b-v0.9.1 · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/18s7iw1/capybara_dataset_is">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://showcase.reka.ai/">Reka Core showcase</a>：展示 Reka Core 与其他主要模型响应的定性示例</li><li><a href="https://github.com/erikbern/ann-benchmarks">GitHub - erikbern/ann-benchmarks: Benchmarks of approximate nearest neighbor libraries in Python</a>：Python 中近似最近邻库的基准测试 - erikbern/ann-benchmarks
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1229117966076088430)** (5 条消息): 

- **澄清频道目的**：针对有关频道目的的查询，分享了 **RAG/Long Context Reasoning Dataset** 文档的链接。提供的链接指向 Google Docs 登录页面 ([RAG 数据集文档](https://docs.google.com/document/d/1o8asa0hD0qK5mKkdY5riUeGm-bxKzL02--1r3MgPgdM/edit))。

- **关于长上下文限制的查询**：一位成员询问了长上下文能力的范围，涉及数百万个文档或像 Wikipedia 这样的整个数据集，但消息中未提供关于限制的明确答案。

**提及的链接**：<a href="https://docs.google.com/document/d/1o8asa0hD0qK5mKkdY5riUeGm-bxKzL02--1r3MgPgdM/edit">RAG/Long Context Reasoning Dataset</a>：未找到描述

  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1228296399242924104)** (66 条消息 🔥🔥): 

- **WorldSim 回归倒计时**：成员们正热切期待 **WorldSim** 的回归，目标定于下周三。这种兴奋显而易见，讨论围绕模拟提供的丰富体验和可能性展开。
  
- **关于 AI 和游戏的对话**：关于 LLMs 如何彻底改变游戏的讨论非常热烈，想象着具有可适应的 AI 生成内容、复杂的魔法系统以及类似于 *Noita* 和 *Dwarf Fortress* 等作品的全可破坏环境的游戏。

- **来自 WorldSim 的创意火花**：用户们在反思 **WorldSim** 如何影响了他们的思维，一些人表示它的缺失激发了他们去尝试自己的 AI 模型模拟，而另一些人则幻想它为游戏玩法和叙事带来的魔法般的不可预测性。

- **预期管理**：虽然社区里充满了期待，但也有提醒要保持理性的声音。**WorldSim** 下一个版本的发布计划据说是“下周三之前”，但与任何开发工作一样，时间表可能会发生变动。

- **暗示未来增强功能**：**WorldSim** 背后的团队暗示了下一个版本中的新功能。这种预热持续激发着社区对未来内容的猜测和热情。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://play.aidungeon.com/scenario/9D9o0X3tA8Vb/world-sim">AI Dungeon</a>：未找到描述</li><li><a href="https://tenor.com/view/interstellar-cost-little-maneuver-51years-51-gif-24426899">Interstellar Cost GIF - Interstellar Cost Little Maneuver - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/lbl-laidbackllamas-laid-back-llamas-wen-llama-llama-gif-181137193945110141">Lbl Laidbackllamas GIF - Lbl Laidbackllamas Laid-back-llamas - Discover &amp; Share GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1228292797765783595)** (1 messages): 

- **Mixtral 免费模型已禁用**：`Mixtral 8x22B:free` 的免费变体已不再可用；建议用户切换到标准的 [Mixtral 8x22B](https://openrouter.ai/models/mistralai/mixtral-8x22b)。
- **供测试的新实验性模型**：两个新的实验性模型已上线供测试：[Zephyr 141B-A35B](https://openrouter.ai/models/huggingfaceh4/zephyr-orpo-141b-a35b) 和 [Fireworks: Mixtral-8x22B Instruct (preview)](https://openrouter.ai/models/fireworks/mixtral-8x22b-instruct-preview)。这些是 Mixtral 8x22B 的 instruct fine-tunes 版本。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/mistralai/mixtral-8x22b>)">Mixtral 8x22B by mistralai | OpenRouter</a>：Mixtral 8x22B 是来自 Mistral AI 的大规模语言模型。它由 8 个专家组成，每个专家有 220 亿参数，每个 token 每次使用 2 个专家。它通过 [X](https://twitter...) 发布。</li><li><a href="https://openrouter.ai/models/huggingfaceh4/zephyr-orpo-141b-a35b>)">Zephyr 141B-A35B by huggingfaceh4 | OpenRouter</a>：Zephyr 141B-A35B 是一个混合专家模型 (MoE)，总参数为 141B，激活参数为 35B。在公开可用的合成数据集混合物上进行了微调。它是...的指令微调版本。</li><li><a href="https://openrouter.ai/models/fireworks/mixtral-8x22b-instruct-preview>)">Fireworks Mixtral 8x22B Instruct OH by fireworks | OpenRouter</a>：来自 Mistral 的最新混合专家模型的第一个指令微调版本：[Mixtral 8x22B](/models/mistralai/mixtral-8x22b)。该模型在来自 [OpenHermes](https://hu...) 的约 1 万条条目上进行了微调。
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1228417709243502695)** (9 messages🔥): 

- **代币购买故障排除**：有报告称在购买代币时遇到问题；然而，经澄清这与 OpenRouter 无关。建议用户直接联系 **Syrax**。
- **退出聊天模式以解决问题**：一位用户被告知他们仍处于 **/chat 模式**，需要再次发出 **/chat 命令**来切换进入或退出，这应该能解决他们的问题。
- **进入展示区**：为了让应用出现在 **app-showcase** 板块，用户被建议确保该应用获得**足够的使用量**。
- **邀请 Beta 测试高级研究助手**：发布了一个名为 **Rubiks.ai** 的新型高级研究助手和搜索引擎的公告，正在招募 **beta 测试人员**，并提供 2 个月的免费高级会员优惠，包括著名的模型如 **Claude 3 Opus, GPT-4 Turbo, Mistral Large, Mixtral-8x22B** 等。感兴趣的各方应发送私信以提供反馈，并可以使用促销代码 `RUBIX`。[点击此处查看](https://rubiks.ai/)

**提到的链接**：<a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>：未找到描述

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1228258855226703935)** (480 messages🔥🔥🔥):

- **Mixtral 性能问题与动态路由**：用户注意到 **Mixtral 8x7B Instruct (nitro)** 与 **Base** 版本相比速度非常慢。一位用户通过重新路由解决了该问题以提高速度，并讨论了未来默认动态路由到每秒事务数 (t/s) 最高的端点的计划。
  
- **Zephyr 和 Firextral 模型讨论**：关于各种模型性能的讨论，包括 **Zephyr 141b** 被贴上表现平平的标签，以及 **Firextral-8x22B** 评价褒贬不一（一位用户觉得出奇地好，而其他人认为很糟糕）。用户还分享了对 **Gemini Pro 1.5** 等文本生成模型的不同体验，并辩论了 **GPT-4** 等模型的规模和尺度优势。

- **OpenRouter (OR) 工具/函数调用 (Function Calls) 说明**：关于某些模型是否支持 OpenRouter 上的函数调用/工具使用存在一些困惑。一位用户纠正说，OR 上的非 OpenAI 模型可以处理函数调用，但没有专门的输出字段。另一位用户澄清了 OR 中非 OpenAI 模型（特别是 OSS 模型）如何处理函数调用。

- **关于微调和 LLM 快速演进的讨论**：对话反映了大型语言模型 (LLM) 的快速演进，以及社区对新官方微调版本的期待，包括潜在的 Mixtral-8x22B instruct 模型。此外，还认可了 WizardLM 系列的历史价值及其对社区的影响。

- **新 WizardLM-2 系列发布**：**WizardLM-2** 系列的发布（包括 8x22B、70B 和 7B 尺寸的模型）引起了社区的兴奋。用户热切期待这些新模型在 OpenRouter 上的部署，其中 **WizardLM-2 8x22B** 被认为可以与 **GPT-4** 等其他领先模型相媲美。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://api.together.xyz',">未找到标题</a>：未找到描述</li><li><a href="https://wizardlm.github.io/WizardLM2/">WizardLM 2</a>：社交媒体描述标签</li><li><a href="https://huggingface.co/fireworks-ai/mixtral-8x22b-instruct-oh">fireworks-ai/mixtral-8x22b-instruct-oh · Hugging Face</a>：未找到描述</li><li><a href="https://docs.together.ai/reference/chat-completions">Chat Completions</a>：未找到描述</li><li><a href="https://docs.together.ai/docs/inference-models">Inference Models</a>：未找到描述</li><li><a href="https://huggingface.co/TheBloke/Falcon-180B-Chat-GPTQ">TheBloke/Falcon-180B-Chat-GPTQ · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/18ljvxb/llm_prompt_format_comparisontest_mixtral_8x7b/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://docs.together.ai/docs/function-calling">Function calling</a>：未找到描述</li><li><a href="https://deepinfra.com/docs/advanced/function_calling">在 Deep Infra 端点使用 Function Calling | ML 模型 | Deep Infra</a>：查找有关使用 Deep Infra 端点进行函数调用的信息、集成等！
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1228275170033143878)** (237 条消息🔥🔥): 

- **错误故障与解决**：用户在尝试启动模型 `lmstudio-community • codegemma` 时遇到了错误消息 `(Exit code: 139)?. Please check settings and try loading the model again.`。根据建议，关闭 GPU offload 解决了该问题。
- **LM Studio 模型推荐**：对于初学者，**heyitsyorkie** 建议从应用首页选择一个 7B 模型，并确认 **LM Studio** 不支持图像生成；它是一个纯文本工具。
- **Mixtral 与集成想法**：用户讨论了集成 **LM Studio** 和 **Stable Diffusion** 的想法，即由 LLM 编写图像提示词（prompts），然后由 Stable Diffusion 生成图像。**heyitsyorkie** 提到了 LLM 与 ComfyUI 之间用于提示词生成的连接。
- **高效无限上下文 Transformer**：分享了一篇题为《Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention》的论文，概述了一种使用 **Infini-attention** 扩展 Transformer 以处理无限长输入的新方法，基准测试显示了其在大序列任务上的有效性。
- **新模型的兴奋与性能讨论**：用户分享了对 **Command R+** 和 **Mixtral 8x22b** 等新模型的体验，讨论了最佳设置配置以及在不同硬件（包括 **M1, M3 Max 和 4090 GPU**）上的性能表现。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.amazon.ca/CORSAIR-Vengeance-PC4-25600-Desktop-Memory/dp/B07Y4ZZ7LQ/ref=sr_1_25?dib=eyJ2IjoiMSJ9.of-9wP0cpMNIBO5VEC3rUnPgXDDfZNeXir57w2hl0vwoj5lxbP31-GTPisdwSW4GGP71ZrL7P1KQvFl5x9FG3CLv7Svz7H9CUlfO9leCj6whiTcEZ8FGWfp27kvV-nYPyyqa0VZl3okpzcaYIsV-V-94yf6uEsR-gbwLfDfPVXJqVJAtB9ltY-88GiDXVlVOWQg4v1Nb53Wn3-Fjd0UerxqkU8PWd9LZSFHxWUC219k1JtsxKpuYEIy2ZtrP8r3c2kcePbcVt2QXqqjfXZEgaNIbbY4cOLkPEcgtTokbfTA.JZSm-ZSuVy4_KNhvKEpt7R7F7hB6_5TlCawp0EPeybs&dib_tag=se&hvadid=671271279297&hvdev=c&hvlocphy=9000686&hvnetw=g&hvqmt=b&hvrand=7421562605769050133&hvtargid=kwd-488359302712&hydadcr=8362_13664276&keywords=ddr4%2Bram&qid=1712950458&sr=8-25&th=1">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/ldjconfirmed/status/1778974720647458885">来自 LDJ (@ldjconfirmed) 的推文</a>: Zephyr-ORPO-141B 是我见过的第一个能始终正确解释 JEPA 缩写含义的模型。我甚至尝试了 Claude-3-Opus，它也失败了，甚至最新的 GPT-4-turbo...</li><li><a href="https://huggingface.co/pmysl/c4ai-command-r-plus-GGUF">pmysl/c4ai-command-r-plus-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://missionsquad.ai/">missionsquad-web</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1atvxu2/current_state_of_training_on_amd_radeon_7900_xtx/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://huggingface.co/search/full-text?q=Command+R%2B>">全文搜索 - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF/tree/main">dranger003/c4ai-command-r-plus-iMat.GGUF 在 main 分支</a>: 未找到描述</li><li><a href="https://arxiv.org/html/2404.07143v1">Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention</a>: 未找到描述</li><li><a href="https://www.amazon.ca/CORSAIR-Vengeance-PC4-25600-Desktop-Memory/dp/B07Y4ZZ7LQ/ref=sr_1_25?dib=eyJ2I">未找到标题</a>: 未找到描述</li><li><a href="https://www.amazon.co.uk/2666MHz-Desktop-PC4-21300-Unbuffered-Computer-Black/dp/B0CD7PJ4DV/?_encoding=UTF8&pd_rd_w=Fof0w&content-id=amzn1.sym.386c33bb-9a6d-4a4d-9a06-3bb24cb22d5d%3Aamzn1.symc.cdb151ed-d8fe-485d-b383-800c8b0e3fd3&pf_rd_p=386c33bb-9a6d-4a4d-9a06-3bb24cb22d5d&pf_rd_r=CCWDNV1ZK7D84Q2MAPPJ&pd_rd_wg=Et1Qj&pd_rd_r=7c25eec1-65d6-4330-a90d-005da4542c5f&ref_=pd_gw_ci_mcx_mr_hp_atf_m">未找到标题</a>: 未找到描述</li><li><a href="https://www.newegg.ca/kingston-64mb/p/N82E16820242771?item=N82E16820242771&source=region&nm_mc=knc-googleadwordsca-pc&cm_mmc=knc-googleadwordsca-pc-_-pla-_-memory+%28server+memory%29-_-N82E16820242771&utm_source=google&utm_medium=paid+shopping&utm_campaign=knc-googleadwordsca-pc-_-pla-_-memory+%28server+memory%29-_-N82E16820242771&id0=Google&id1=12409815000&id2=119872896404&id3=&id4=&id5=pla-2281720576259&id6=&id7=9000686&id8=&id9=g&id10=c&id11=&id12=CjwKCAjwt-OwBhBnEiwAgwzrUr4leQFCH4rFOGg64jUHpprq-H_yAxFNPhLghRNlWhft-so2HWUnaxoCoa0QAvD_BwE&id13=&id14=Y&id15=&id16=500489819465&id17=&id18=&id19=&id20=&id21=pla&id22=102503995&id23=online&id24=N82E16820242771&id25=CA&id26=2281720576259&id27=Y&id28=&id29=&id30=1484129855650391095&id31=en&id32=&id33=&id34=&gad_source=1&gclid=CjwKCAjwt-OwBhBnEiwAgwzrUr4leQFCH4rFOGg64jUHpprq-H_yAxFNPhLghRNlWhft-so2HWUnaxoCoa0QAvD_BwE">Kingston 64GB DDR5 4800 ECC Registered DIMM 内存 - Newegg.com</a>: 购买 Kingston 64GB DDR5 4800 ECC Registered DIMM 内存，享受快速发货和顶级客户服务。一旦了解，就选 Newegg！</li><li><a href="https://www.newegg.ca/kingston-64mb/p/N82E16820242771?item=N82E16820242771&sou">Kingston 64GB DDR5 4800 ECC Registered DIMM 内存 - Newegg.com</a>: 购买 Kingston 64GB DDR5 4800 ECC Registered DIMM 内存，享受快速发货和顶级客户服务。一旦了解，就选 Newegg！</li><li><a href="https://www.youtube.com/watch?v=a75TC-w2aQ4">新版 Mixtral 8x22b 测试 - Mistral 的新旗舰 MoE 开源模型</a>: Mistral AI 刚刚发布了 Mixtral 8x22，这是一个在基准测试中名列前茅的大型 MoE 开源模型。让我们来测试一下！订阅我的时事通讯以获取定期 AI 更新 👇...</li><li><a href="https://github.com/rjmacarthy/twinny">GitHub - rjmacarthy/twinny: 最务实的本地托管（或 API 托管）AI 代码补全 Visual Studio Code 插件，类似于 GitHub Copilot 但 100% 免费！</a>: 最务实的本地托管（或 API 托管）AI 代码补全 Visual Studio Code 插件，类似于 GitHub Copilot 但 100% 免费！ - rjmacarthy/twinny
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1228306022905680003)** (87 条消息🔥🔥):

- **Command R+ 因其敏锐的分析能力而受到青睐**：成员们观察到 **Command R+** 在处理 22000 token 的文档时提供了敏锐的分析，其能力足以媲美甚至超越 **ChatGPT 4** 等同类模型。
- **寻找大型 LLM**：用户讨论了在特定硬件配置上进行文本生成的最佳模型，指出对于支持 AVX2 指令、125B+ 能力的服务器，推荐使用 **CMDR+**、**Goliath 120B** 和 **MiquLiz 120B** 等模型，并建议在 NVIDIA 设备上通过 `nvidia-smi nvlink --status` 检查 NVLink 状态。
- **海量 GGUF 存档上线**：一位用户宣布了一个包含 320 个模型的新 GGUF 存档，包括 Mini MOEs、Super MOEs 以及各种高质量量化模型，并附带了其在 [Hugging Face](https://huggingface.co/DavidAU) 上的集合链接。
- **量化与质量的权衡**：围绕 **Command R Plus** 等量化模型性能的讨论表明，极度量化（iQ3XS, iQ1S）可能会带来微小的性能提升，但可能会严重损害输出质量。
- **探索专注于编程的 LLM**：成员们正在寻找并推荐针对 Python 和 Java 等语言的编程相关任务量身定制的各种大型语言模型，指出 **WaveCoder** 系列和 **MythoMax** 是编程辅助的不错选择。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/microsoft/wavecoder-ultra-6.7b">microsoft/wavecoder-ultra-6.7b · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.cohere.com/docs/command-r-plus#multilingual-capabilities">Command R+</a>: 未找到描述</li><li><a href="https://huggingface.co/TheBloke/MythoMax-L2-13B-GGUF">TheBloke/MythoMax-L2-13B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/DavidAU">DavidAU (David Belton)</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://huggingface.co/DavidAU?sort_models=created#models">DavidAU (David Belton)</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1228548452783357963)** (30 条消息🔥): 

- **模型加载受阻**：一位用户报告了将 [模型加载](https://releases.lmstudio.ai/mac/arm64/0.2.18/latest/LM-Studio-0.2.18-arm64.dmg) 到内存时的问题。他们收到了“Error loading model”的错误，并详细列出了显示资源充足的系统配置。
- **降级版本解决问题**：该用户确认在降级到 **LM-Studio 0.2.18** 版本后，模型成功加载，这表明最新更新可能存在潜在问题。
- **推测底层问题**：一位成员建议，底层的 llama.cpp 可能存在新的开销，导致在 **GPU offload 设为 100%** 时出现模型加载困难。
- **对 UI 杂乱的投诉**：某未具名产品的 UI 因过于杂乱且元素相互遮挡而受到批评；提到了截图但未提供。
- **显示器尺寸并非罪魁祸首**：针对 UI 问题，有人指出长宽比可能比显示器尺寸起着更重要的作用，并注意到一位拥有 32 英寸显示器的用户仍然面临布局问题。
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1229462981604020266)** (11 条消息🔥): 

- **LM Studio 模型加载故障**：一位用户在尝试在 LM Studio 中加载模型时遇到错误。错误消息提示未知错误，并建议尝试不同的模型和/或配置。

- **寻求恢复默认设置**：该用户表示希望恢复到默认设置，而不给系统的 GPU 或 RAM 带来压力。由于性能担忧，他们寻求关于在不开启 GPU offload 的情况下运行模型的指导。

- **持续的模型加载错误**：尽管关闭了 GPU offload，用户在对默认设置进行调整后加载模型时仍然面临错误。他们报告称，即使是以前在 32 GB RAM 上可以运行的 12 GB 模型现在也无法加载。

- **请求彻底清除数据**：用户请求协助从其 PC 中完全删除所有 LM Studio 数据，以便进行全新的重新安装并恢复默认设置。此请求是在持续出现错误和模型加载问题之后提出的。


  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1228312813571932260)** (53 条消息🔥):

- **GitHub 上的 Open GPU Kernel Modules**：分享了一个指向 **tinygrad/open-gpu-kernel-modules** 的链接，这是一个在 GitHub 上提供的支持 P2P 的 NVIDIA Linux 开源 GPU 项目。点击[此处](https://github.com/tinygrad/open-gpu-kernel-modules)查看详情和贡献。

- **理解 CPU 与 GPU 推理**：讨论澄清了 CPU 推理比 GPU 推理慢的原因，很大程度上是因为其缺乏并行化能力；一位成员指出，更大的 RAM 允许加载更大的模型，但不一定能提高 CPU 推理的速度。

- **AI 深度学习的硬件需求**：一位成员表示，要获得类似于 GPT-4 的体验，需要对硬件进行大量投资，可能在六位数的范围内，即使是像搭载 **M1/2/3** 芯片和 64GB 以上 RAM 的高端笔记本电脑也可能无法提供这种水平的性能。

- **LM Studio 的 GPU 推理问题**：几位成员讨论了包括在 **RTX 3090 TI** 上运行 **ggml-dbrx-instruct-16x12b-iq1_s.gguf** 模型在内的问题，并指出 DBRX 模型目前在 LM Studio 中仍不受支持，从而导致下载和兼容性问题。

- **Windows 中的技术故障排除**：成员们参与排除了一项 CPU 问题，即特定核心被 Windows 大量占用，建议包括检查恶意软件到减少后台进程，但尚未提供最终解决方案。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/hardware/comments/1c2dyat/geohot_hacked_4090_driver_to_enable_p2p/?share_id=d7ey-6LE-pki-7UluvgXH">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: 支持 P2P 的 NVIDIA Linux 开源 GPU</a>：支持 P2P 的 NVIDIA Linux 开源 GPU。通过在 GitHub 上创建账号来为 tinygrad/open-gpu-kernel-modules 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1228519498894803084)** (6 条消息): 

- **MACOS 0.2.19 中的模型文件夹谜团**：一位成员报告了 **MACOS 0.2.19**（预览版 C 和正式版）中的异常行为，系统无法识别其本地模型文件夹中的任何模型，无论该文件夹是标准目录还是符号链接的 NFS 挂载。回滚到 0.2.19_previewA 可解决此问题。

- **Linux Beta 版被排除在 ROCm 支持之外？**：一位成员询问为什么 Linux Beta 版中缺少 **ROCm** 支持，而 Windows Beta 版却有。

- **关于不同“Beta”版本的困惑已澄清**：对不同 Beta 版本存在短暂困惑，后经澄清为误解——Linux 和 ROCm 尚未得到官方支持，这使其实际上处于“三重 Beta”状态。

- **对 Linux ROCm Beta 缺失表示遗憾**：当确认 Linux 上缺乏对 ROCm 的支持时，该成员表达了失望。
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1228897380414259321)** (4 条消息): 

- **Autogen Studio Docker 镜像现已可用**：一位成员创建了 **Autogen Studio** 的 Docker 镜像，并使用 [renovate](https://github.com/lludlow/autogen-studio) 保持更新。该镜像已在 GitHub 上提供，社区可以参与贡献。
- **Threadripper Pro 利用率不足之谜**：一位成员的 **Threadripper Pro** 出现了利用率不足的情况，CPU 使用率被限制在 50-56% 左右，原因尚不明确。
- **带有模型和提示词 GUI 的 Missionsquad.ai**：另一位成员介绍了 [missionsquad.ai](https://missionsquad.ai/)，这是一个带有 GUI 的工具，允许用户直接在用户界面内定义模型和提示词。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://missionsquad.ai/">missionsquad-web</a>：未找到描述</li><li><a href="https://github.com/lludlow/autogen-studio">GitHub - lludlow/autogen-studio: Autogen studio docker</a>：Autogen studio docker。通过在 GitHub 上创建账号来为 lludlow/autogen-studio 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1228472444613033995)** (38 条消息🔥):

- **LM Studio 未利用 GPU**：尽管安装了 **ROCm** 驱动并选择了 GPU，一位成员表达了对 Windows 11 上的 LM Studio 0.2.19.0 未使用 **8GB Radeon 6600** 的担忧，这与能跑满 GPU 利用率的 GPT4ALL 不同。
- **Radeon 6600 不受支持**：一位成员简短回复称，对于 LM Studio 中的 **ROCm** 而言，**Radeon 6600** 属于*不受支持的显卡*。
- **Llama.cpp 在 Windows 上的构建困扰**：围绕在 Windows 上使用 IC Compiler 构建 **llama.cpp** 的挑战展开了讨论，随后出现了关于使用 **WSL**、**AMD 版本的 BLAS** 以及遵循共享代码片段中提供的特定*构建指令*的建议。
- **在备选驱动器上安装 WSL**：一位成员确认了将 **WSL** 安装在 C 盘以外驱动器的可能性，指出它确实可以放置在 D 盘。
- **ROCm LM Studio 的卡顿报告**：一位成员报告在 ROCm LM Studio 中遇到延迟，即使在未加载模型且清理了大量文本聊天后也是如此。文中给出了详细的系统规格，随后讨论了如何监控该问题并识别潜在原因，如耗时操作或应用程序内的特定用户行为。

---

**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1228300620755107860)** (397 messages🔥🔥): 

- **理解 AI 模型与意识**：用户们就 AI 意识和感知能力的属性及发展展开了长篇辩论，特别是考虑到 AI 的进步。人们对负责任地使用 AI 以及具有潜在感知能力的系统的伦理影响表示担忧。

- **使用 ChatGPT 的实用编程帮助**：一位成员就为什么 ChatGPT 提供指南而不是按要求直接修改代码寻求建议。尽管尝试调整了请求的措辞，且其他成员进行了澄清，但对话中仍未解决根本问题。

- **Prompt Hacking 竞赛咨询**：一位用户分享了 [Blade Hack](https://bladehack.tech/) 竞赛的链接，寻求队友以及关于 LLM 的 Prompt Hacking 建议。对话显示出人们对这类活动所需的伦理方面和技能既好奇又担忧。

- **LLM 背景与神经生物学进展**：分享了关于神经生物学研究的历史和发展、它与 AI 设计的关系以及数据解释瓶颈的详细解释和见解。一位用户讨论了受此类研究影响的个人“心智模型 (Mind Model)”，提供了对其中复杂性的见解。

- **Gemini 1.5 与 AI 自我意识的问题**：一位成员对 Gemini 1.5 表示失望，因为它无法识别自己分析视频的能力，这反映了 AI 系统无法意识到自身功能的更广泛问题——据指出，这一问题与 Bing Chat 类似。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/John_C._Lilly#%22Solid_State_Intelligence%22>">John C. Lilly - 维基百科</a>：未找到描述</li><li><a href="https://bladehack.tech/">Blade Hack</a>：Blade Hack 是一项全球性活动，专注于 Prompt Hacking，涉及以创造性方式探索大型语言模型 (LLM) 并与之交互。Blade 将面向初学者和个人...</li><li><a href="https://direct.mit.edu/books/oa-monograph/3653/Language-in-Our-BrainThe-Origins-of-a-Uniquely)">Language in Our Brain: The Origins of a Uniquely Human Capacity</a>：对语言的神经生物学基础进行了全面阐述，认为物种特有的大脑差异可能是人类能力的核心。
</li>
</ul>

</div>

---

**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1228323876832940214)** (36 messages🔥): 

- **解释 GPT 的 API 交互**：一位用户询问如何使用 GPT 处理大型文档，并被告知编辑文档需要托管文件并编写供 GPT 交互的 API，因为它无法直接与文档交互。

- **澄清 GPT-4 模型能力**：对于为什么新版 Turbo 版本的 GPT-4 仍然有 4k Token 限制存在困惑，对此解释称 Token 限制是内存限制，也是对处理能力的权衡。

- **将 Vision 集成到 GPT**：一位成员寻求关于如何使用 GPT-4 模型分析图像的建议，并被引导至 API 文档，该文档指出 'gpt-4-turbo' 具备 Vision 能力，需要通过编程语言脚本来调用。

- **理解 GPT-4 中的字数限制**：在关于改进 GPT-4 字数统计的讨论中，有人提到 GPT 模型本质上无法生成具有特定字数的输出，应使用定性语言来引导所需的输出长度。

- **分享 Token 管理策略**：为了减轻 Token 限制对 GPT 输出的影响，用户分享了一些策略，例如总结对话以及在后续交互中使用新的 Prompt，以便在不超出限制的情况下保持上下文。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1228571377150201888)** (11 messages🔥): 

- **GPT-4 不同版本之间的有趣差异**：一些成员观察到，在版权遵守方面，**`gpt-4-turbo-2024-04-09`** 与 `gpt-4-0125-preview` 及 `gpt-4-vision-preview` 的响应存在不一致，这暗示了可能存在的微调差异或版本特定行为。

- **GPT 的波动暗示暂时性的不稳定**：一位用户分享了 GPT 模型最初无法从学术论文中检索内容，但随后在无需干预的情况下自行恢复的经历，暗示潜在的“实时微调（live fine-tuning）”可能导致了瞬时问题。

- **对 Custom GPT 与默认 GPT-4 模型的见解**：一位成员思考 Custom GPTs 与默认 ChatGPT4 是否存在本质区别，以及这是否需要不同的 Prompt 风格或优化技术来获得与 Custom GPTs 的最佳交互。

- **Meta-prompting 模板提供替代工作流**：一位用户建议在创建 Custom GPT 模型时使用 Meta-prompting 模板，认为这比配置面板（configure pane）更具优势。

- **征集 Prompt Hacking 队友**：一项公告征集参与者组队参加某项 Prompt Hacking 竞赛，为社区内的协作创造了机会。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1228571377150201888)** (11 messages🔥): 

- **对模型版权响应的困惑**：用户观察到不同版本的 GPT-4 在处理版权问题时存在不一致：在 Prompt 相同的情况下，**`gpt-4-turbo-2024-04-09`** 提供了答案，而 **`gpt-4-0125-preview`** 则引用了版权法。
- **模型在访问学术论文方面的不一致性**：有报告称模型在访问学术论文时会不一致地声称版权限制，但这一问题似乎在最近几周已得到解决。
- **Custom GPT Prompt 优化策略**：一位用户分享了改进 Custom GPT Prompt 的策略：在 Custom GPT 执行环境中优化 Prompt，或在默认 ChatGPT 会话中单独优化，并指出后者通常能产生更好的上下文处理和推理效果。
- **Meta-Prompting 中的模型行为**：针对 Prompt 优化的讨论，**darthgustav.** 建议在创建 GPT 的创建者面板（creator pane）中使用 Meta-prompting 模板，认为这为该过程提供了更好的工作流。
- **邀请加入 Prompt Hacking 竞赛**：**wisewander** 表达了组队参加 Prompt Hacking 竞赛的兴趣，并正在寻找队友。
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1228276727659237436)** (228 messages🔥🔥): 

- **Scarlet AI 加强协作**：Scarlet 提供用于复杂任务管理的**协作式 AI Agent**，具有实时协作、基于数据的自动上下文获取以及 Agent/工具编排功能。一位参与者分享了 [Scarlet 官网](https://scarletai.co)链接，并提到一个正在开发中的 Demo 以及即将与 Zapier NLA 的集成。

- **Perplexity 从 LMSYS 神秘消失**：一位成员对 Perplexity “在线”模型从 LMSYS Arena 中移除表示担忧，引发了关于 Perplexity 有效性的讨论，另一位成员则希望将其集成到自己的工具中。

- **利用 YouTube 内容为 AI 服务**：讨论了**转录和总结 YouTube 内容**的最佳方式，探索了 Descript、youtube-dl 以及带有角色分离（diarization）功能的 Whisper 等工具，用于 AI 模型应用。

- **Limitless 旨在通过先进 AI 重新定义可穿戴设备**：Rewind 正在更名为 **Limitless**，承诺通过可穿戴吊坠、App 和“机密云（confidential cloud）”提供**个性化 AI** 能力。针对这一新产品，讨论中涉及了对本地处理和端到端加密的担忧。

- **探索 Generative AI 中的搜索潜力**：讨论了 Generative AI 对搜索工具的影响，成员们提到了 Kagi、Phind、Perplexity，以及该领域的新入局者如用于数据搜索的 **Cohere's Compass**，并分享了如何管理 Vectorstores 以实现一致且具有成本效益的 Embedding 技巧。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<a href="https://scarletai.co">Scarlet</a>: 未找到描述</li><li><a href="https://x.com/lmsysorg/status/1778555678174663100?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 lmsys.org (@lmsysorg) 的推文</a>: 🔥令人兴奋的消息 —— GPT-4-Turbo 刚刚再次夺回了 Arena 排行榜的第一名！哇！我们收集了来自不同领域的 8,000 多张用户投票，并观察到其强大的代码编写和推理能力...</li><li><a href="https://x.com/dsiroker/status/1779857843895599383?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Dan Siroker (@dsiroker) 的推文</a>: 介绍 Limitless：一款由你所见、所言、所闻驱动的个性化 AI。它包含 Web 应用、Mac 应用、Windows 应用以及一款可穿戴设备。0:06 揭晓 0:48 为什么选择 Limitless？ 1:39 演示 3:05 Pendant 4:2...</li><li><a href="https://www.datasette.cloud/blog/2024/datasette-extract/">使用 Datasette 和 GPT-4 Turbo 从非结构化文本和图像中提取数据 - Datasette Cloud</a>: 具有明确定义的列和行的干净数据是一件美好的事情 —— 它是任何数据分析或可视化项目的理想起点。遗憾的是，世界上有趣的原始数据中只有极少部分...</li><li><a href="https://x.com/aidangomez/status/1779882113573044625?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Aidan Gomez (@aidangomez) 的推文</a>: 很高兴宣布 Compass Beta 版，这是一个由新型 Embedding 模型 Compass 驱动的、功能非常强大的多维度数据搜索系统。我们正在寻求帮助来对该模型的能力进行压力测试...</li><li><a href="https://x.com/harrystebbings/status/1779910559753802010?s=">来自 Harry Stebbings (@HarryStebbings) 的推文</a>: 10 年前，我在卧室里开始了 20VC，没有资金也没有人脉。今天，我们发布了与 @sama、@bradlightcap 以及历史上增长最快的公司 @OpenAI 合作的 20VC。互联网的力量...</li><li><a href="https://llm-price.com/">LLM 定价 - 比较大语言模型成本和价格</a>: 未找到描述</li><li><a href="https://x.com/hrishioa">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://blog.robertelder.org/causes-of-bit-flips-in-computer-memory/#chip-orientation">计算机内存中位翻转（Bit Flips）的原因是什么？</a>: 未找到描述</li><li><a href="https://www.bloomberg.com/news/articles/2024-04-11/elon-musk-s-xai-seeks-up-to-4-billion-to-compete-with-openai">Bloomberg - 你是机器人吗？</a>: 未找到描述</li><li><a href="https://x.com/xuefz/status/1778474016854192216">来自 Fuzhao Xue (@XueFz) 的推文</a>: @mejia_petit @fouriergalois 哦当然，没关系。这项工作太旧了哈哈。它被会议拒绝了很多次，我放弃再次提交了，呵呵。但老实说，我确实认为这是一个重要的...</li><li><a href="https://x.com/RekaAILabs/status/1779894626083864873">来自 Reka (@RekaAILabs) 的推文</a>: 随 Core 一起，我们发布了一份技术报告，详细介绍了 Reka 模型的训练、架构、数据和评估。https://publications.reka.ai/reka-core-tech-report.pdf</li><li><a href="https://www.descript.com/blog/article/all-new-descript-backed-by-openai-startup-fund">它来了：由 OpenAI Startup Fund 支持的全新 Descript</a>: 我们正在发布全新版本的 Descript，并宣布 OpenAI Startup Fund 将领投我们 5000 万美元的 C 轮融资。</li><li><a href="https://x.com/imranchaudhri/status/1778445220050333711?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Imran Chaudhri (@imranchaudhri) 的推文</a>: 我们花了一些时间才走到今天。既然我们已经到了这里，我们迫不及待地想向大家展示 @Humane 未来几年的样子 —— 令人兴奋的时刻！</li><li><a href="https://barryzhang.substack.com/p/making-peace-with-llm-non-determinism">与 LLM 的非确定性达成和解</a>: 深入研究稀疏 MoE 和 GPU 周期，才意识到非确定性并不新鲜，新鲜的是语言。</li><li><a href="https://x.com/pronounced_kyle/status/1779899769982464492?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Christian Keil (@pronounced_kyle) 的推文</a>: @TouristShaun 是的，这时机选得相当狠，老实说我挺喜欢 Dan 这么做的哈哈</li><li><a href="https://x.com/winglian/status/1779968341332860940?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Wing Lian (caseus) (@winglian) 的推文</a>: 好了，在等待 axolotl ai 域名转移期间，我们先用点老办法。在这里注册以获取私测版的访问权限。https://docs.google.com/forms/d/e/1FAIpQLSd0uWGZOwviIZPoOPOAaFDv3edcCXEIG...</li><li><a href="https://x.com/rekaailabs/status/1779894622334189592?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Reka (@RekaAILabs) 的推文</a>: 认识一下 Reka Core，这是我们迄今为止最优秀、最强大的多模态语言模型。🔮 过去几个月我们一直忙于训练这个模型，很高兴终于发布了！💪 Core 拥有许多能力，而且...</li><li><a href="https://x.com/yitayml/status/1779895037335343521?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Yi Tay (@YiTayML) 的推文</a>: 这是一段疯狂的旅程。我们只有 20 个人，在过去的时间里消耗了数千张 H100

几个月了，我们很高兴终于能与世界分享这个！💪 我们在创办 Rek 时的一个目标是...</li><li><a href="https://scalingknowledge.substack.com/p/rag">检索增强生成研究：2017-2024</a>：RAG 文献综述，包括：REPLUG, Fusion-in-Decoder, KNN-LM, RETRO, FLARE, Knowledge Graph Fusion-in-Decoder, SILO, WebGPT, Toolformer, Self-RAG, GRIT 等</li><li><a href="https://x.com/lilianweng/status/1779914184874160170?s=46&t=90xQ8sGy63D2OtiaoGJuww">Lilian Weng (@lilianweng) 的推文</a>：🎨 花了一些时间重构了 2021 年关于 Diffusion Model 的文章，并增加了新内容：https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ ⬇️ ⬇️ ⬇️ 🎬 接着是另一篇关于 Diffusion Video 的短文...</li><li><a href="https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems">浮点运算 - 维基百科</a>：未找到描述</li><li><a href="https://github.com/MahmoudAshraf97/whisper-diarization">GitHub - MahmoudAshraf97/whisper-diarization：基于 OpenAI Whisper 的带说话人日志（Speaker Diarization）的自动语音识别</a>：基于 OpenAI Whisper 的带说话人日志的自动语音识别 - MahmoudAshraf97/whisper-diarization</li><li><a href="https://x.com/xai/status/1778963570098855947?s=46&t=90xQ8sGy63D2OtiaoGJuww">xAI (@xai) 的推文</a>：👀 https://x.ai/blog/grok-1.5v</li><li><a href="https://www.youtube.com/watch?v=nsXyVo30bWk">LLM 办公时间，老人们的抱怨第三部分。</a>：未找到描述</li><li><a href="https://www.reddit.com/r/singularity/comments/1bxq84h/openai_transcribed_over_a_million_hours_of/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/openai/whisper/discussions/1515">与 Descript 相比性能极慢 · openai/whisper · Discussion #1515</a>：了解到 Descript 和 Whisper 针对的是不同的人群，我想知道为什么转录同一段音频的速度差异如此之大。（30 分钟的 wav 文件...</li><li><a href="https://x.com/collincornwell/status/1779662507705036819?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Collin (@CollinCornwell) 的推文</a>：智能手机非常不可思议，我每天都在用，很高兴我们的第一代产品甚至有机会挑战一个价值 4571.8 亿美元的行业。如果我们能占领其中的 1%，我们就赢了 @Humane，如果我们继续...</li><li><a href="https://www.youtube.com/watch/nsXyVo30bWk```">LLM 办公时间，老人们的抱怨第三部分。</a>：未找到描述</li><li><a href="https://github.com/hrishioa/lumentis">GitHub - hrishioa/lumentis：AI 驱动的一键式从转录文本和文本生成综合文档。</a>：AI 驱动的一键式从转录文本和文本生成综合文档。 - hrishioa/lumentis</li><li><a href="https://github.com/dimfeld/sbbp/blob/b9ed2bb73537e327eee17ea8038c7156ebe4726a/api/src/jobs/download.rs#L71">dimfeld/sbbp 中的 sbbp/api/src/jobs/download.rs</a>：本该是一篇博客文章。通过在 GitHub 上创建账号来为 dimfeld/sbbp 的开发做出贡献。</li><li><a href="https://youtu.be/TitZV6k8zfA?si=x2CCuE2uMK6sklvK">我评测过的最差产品... 目前为止</a>：Humane AI pin 很... 糟糕。几乎没有人应该买它。至少现在是。MKBHD 周边：http://shop.MKBHD.com 我目前使用的技术产品：https://www.amazon.com/shop/MKBHDIn...</li><li><a href="https://github.com/VikParuchuri/surya">GitHub - VikParuchuri/surya：支持 90 多种语言的 OCR、布局分析和行检测</a>：支持 90 多种语言的 OCR、布局分析和行检测 - VikParuchuri/surya</li><li><a href="https://www.llamaindex.ai/blog/introducing-llamacloud-and-llamaparse-af8cedf9006b">介绍 LlamaCloud 和 LlamaParse — LlamaIndex，LLM 应用的数据框架</a>：LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型（LLM）。</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse：为优化 RAG 解析文件</a>：为优化 RAG 解析文件。通过在 GitHub 上创建账号来为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://x.com/harrystebbings/status/1779910559753802010?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Harry Stebbings (@HarryStebbings) 的推文</a>：10 年前，我在卧室里创办了 20VC，没有钱也没有人脉。今天，我们发布了与 @sama、@bradlightcap 以及历史上增长最快的公司 @OpenAI 合作的 20VC。互联网的力量...</li><li><a href="https://youtu.be/G8T1O81W96Y?si=OHJXeiI69YSOfG57">Sam Altman &amp; Brad Lightcap：哪些公司会被 OpenAI 碾压？ | E1140</a>：Sam Altman 是 OpenAI 的 CEO，该公司的使命是确保通用人工智能（AGI）造福全人类。OpenAI 是最快... 的公司之一。
</li>

**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1228434569834004540)** (143 messages🔥🔥): 

- **语义搜索与内存使用 (Semantic Search and Memory Usage)**：成员们讨论了语义搜索模型中向量大小（vector size）之间的权衡，特别是针对内存使用的讨论。有人提出疑问，为什么尽管维度更小，内存使用量却无法显著降低，速度与内存之间的非线性关系被形容为“奇怪的非线性”。

- **重排序与检索性能 (Re-ranking and Retrieval Performance)**：讨论涉及了检索模型中的重排序策略，以及对检索性能额外衡量指标的需求。分享了一个关于重排序阶段（re-ranking pass）的简短讨论链接，质疑了内存节省和速度与检索任务中实际性能之间的平衡。

- **量化与模型大小 (Quantization and Model Size)**：针对量化对 Embedding 模型的影响、降低向量维度的益处以及提升延迟（latency）的潜力进行了深入交流。提到了一篇关于 PostgreSQL 配合 pgvector 及其权衡的博客文章，增加了讨论的技术深度。

- **多模态嵌入的潜力 (Potential of Multi-Modal Embeddings)**：对话中提到了多模态嵌入（multi-modal embeddings）的概念及其特性，以及应用于文本嵌入的优化是否可以类比应用到图像嵌入等其他类型中。

- **黑客松规划与协作项目 (Hackathon Planning and Collaborative Projects)**：参与者组织了一场专注于 Embedding 的黑客松活动，并号召社区协作参与。大家积极为黑客松的运行（包括时间安排和结构）出谋划策。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://about.xethub.com/blog/you-dont-need-a-vector-database">XetHub | You Don't Need a Vector Database </a>: 你不需要仅仅为了托管用于 RAG（检索增强生成）的文档嵌入而使用向量数据库。在这篇文章中，我将探索使用经典信息检索实现的更简单方案...</li><li><a href="https://getyarn.io/yarn-clip/6a6e70a4-866c-4f88-8e7d-2454c4d084d4">YARN |  |  | Video clips by quotes | 6a6e70a4 | 紗</a>: Yarn 是通过台词搜索视频片段的最佳工具。找到你想分享的电视剧、电影或音乐视频中的确切时刻。轻松前进或后退以获取...</li><li><a href="https://jkatz05.com/post/postgres/pgvector-scalar-binary-quantization/">Scalar and binary quantization for pgvector vector search and storage |
Jonathan Katz
</a>: 未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024 主题、日期、主持人、资源，GenAI 的 UI/UX 模式，1/26/2024，nuvic...</li><li><a href="https://interpreter-weekend.devpost.com">the Arena Online presents: Friday Night Firefight, weekend warrior edition</a>: 社区新推出的 SDK，专为你提供的早期访问权限。我相信为能够执行代码的 LLM 构思酷炫的用途并不难。
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1228551167009751124)** (26 messages🔥): 

- **揭秘获得 Mojician 角色的途径**: 要获得梦寐以求的 `Mojician` 角色，必须 *使用 Mojo 构建一个酷炫的开源项目*，或者向 `modularml/mojo` 提交高质量的 PR。Jack 提到，由于 GitHub 用户名很难与 Discord 身份匹配，已合并 PR 的成员应私信他以获取角色。

- **编程讨论应在别处进行**: 当成员们深入讨论编程语言的学习难度等技术话题时，被提醒将聊天转移到专门的频道，特别是 <#1104620458168553563>。

- **MacOS Intel 用户寻求避免使用虚拟机**: 一位成员分享了在 MacOS Intel 硬件上原生运行项目的偏好，以避免使用虚拟机 (VM)，强调了开发工具中跨平台兼容性的重要性。

- **开源中的匿名顾虑**: 尽管准备好提交多个 PR，但一位成员对向 `modularml/mojo` 贡献代码表示犹豫，因为这要求公开真实姓名，凸显了对公共贡献中个人匿名性的担忧。

- **Kapa AI 集成协助**: 为了增强他们的私有服务器，一位成员寻求添加 kapa.ai 机器人的帮助，其他成员提供了指导并链接了相关资源，如 [Kapa AI 官方网站](https://www.kapa.ai/) 及其 [安装指南](https://docs.kapa.ai/installation-discord)。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.kapa.ai/use-cases/community-engagement">kapa.ai - 面向开发者产品的 ChatGPT</a>: kapa.ai 让面向开发者的公司能够轻松地为其社区构建基于 LLM 的支持和入职机器人。OpenAI、Airbyte 和 NextJS 的团队使用 kapa 来提升他们的开发者体验...</li><li><a href="https://docs.kapa.ai/installation-discord">Discord Bot | kapa.ai 文档</a>: Kapa 可以作为机器人安装在您的 Discord 服务器上。该机器人允许用户使用自然语言询问有关您产品的问题，这改善了开发者体验，因为开发者可以快速找到答案...</li><li><a href="https://www.kapa.ai/">kapa.ai - 技术问题的即时 AI 解答</a>: kapa.ai 让面向开发者的公司能够轻松地为其社区构建基于 LLM 的支持和入职机器人。OpenAI、Airbyte 和 NextJS 的团队使用 kapa 来提升他们的开发者体验...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1228478459857076284)** (6 条消息): 

- **Modular 的最新动态**: Modular 分享了关于近期进展和公告的推文。以下是他们最新推文的链接：[Tweet 1](https://twitter.com/Modular/status/1778918965110354023), [Tweet 2](https://twitter.com/Modular/status/1779913837216719118), [Tweet 3](https://twitter.com/Modular/status/1779913865914134561), [Tweet 4](https://twitter.com/Modular/status/1779913874957086978), [Tweet 5](https://twitter.com/Modular/status/1779913908649914783), [Tweet 6](https://twitter.com/Modular/status/1779913912009597323)。
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/)** (1 条消息): 

docphaedrus: https://www.youtube.com/watch?v=1cQbu2zXTKk

Podman Pull - 关于基础终端命令
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1228283258186498089)** (218 条消息🔥🔥): 

- **探索高级语言特性**: 成员们讨论了在 Mojo 中添加条件性 trait 一致性（conditional trait conformance）的可能性，并将潜在实现与 Rust 和 Swift 中的特性进行了比较。关于是否使用 if 块以及语法应如何处理 struct 内的条件函数声明，存在不同意见。
- **期待 Mojo 标准库扩展**: 对话表明，一些基础模块如 threading、async I/O 和 crypto 在 Mojo 中尚不可用。他们交流了变通方案的想法，并期待更多 stdlib 能够开源。
- **改进 Mojo Reference 语法**: Chris Lattner 提到正在努力使 Mojo 的 `Reference` 更易于使用，并暗示了[未来的提案](https://github.com/modularml/mojo/blob/main/proposals/inferred-parameters.md)，该提案可能会简化可变性（mutability）处理。
- **推动 Python 兼容性**: 讨论揭示了一个愿景，即 Mojo 将演变为完全支持 Python 库，其中包含 C 扩展的 Python 包被列入长期目标。
- **讨论中的 MLIR 和未来扩展**: 用户询问了 Mojo 的 GPU 代码生成方法，以及是否会实现类似 Swift 的 “extensions” 功能，以管理多态错误处理和静态分析。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/enumerations/">文档</a>：未找到描述</li><li><a href="https://xkcd.com/927/">标准</a>：未找到描述</li><li><a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/generics#Extensions-with-a-Generic-Where-Clause">文档</a>：未找到描述</li><li><a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols#Conditionally-Conforming-to-a-Protocol">文档</a>：未找到描述</li><li><a href="https://docs.modular.com/mojo/manual/functions#overloaded-functions>),">函数 | Modular 文档</a>：Mojo `fn` 和 `def` 函数简介。</li><li><a href="https://www.modular.com/blog/what-is-loop-unrolling-how-you-can-speed-up-mojo">Modular：什么是循环展开？如何使用 @unroll 加速 Mojo🔥 代码</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：什么是循环展开？如何使用 @unroll 加速 Mojo🔥 代码</li><li><a href="https://docs.modular.com/mojo/manual/functions#variadic-arguments>">函数 | Modular 文档</a>：Mojo `fn` 和 `def` 函数简介。</li><li><a href="https://docs.modular.com/mojo/lib">Mojo🔥 模块 | Modular 文档</a>：Mojo 标准库中所有模块的列表。</li><li><a href="https://github.com/modularml/mojo/blob/main/proposals/inferred-parameters.md">mojo/proposals/inferred-parameters.md (main 分支) · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/tairov/llama2.mojo/blob/master/llama2.mojo">llama2.mojo/llama2.mojo (master 分支) · tairov/llama2.mojo</a>：仅用一个纯 🔥 文件实现 Llama 2 推理。通过在 GitHub 上创建账号来为 tairov/llama2.mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/main/proposals/lifetimes-and-provenance.md">mojo/proposals/lifetimes-and-provenance.md (main 分支) · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/Moosems/TkLineNums/blob/main/tklinenums/tklinenums.py">TkLineNums/tklinenums/tklinenums.py (main 分支) · Moosems/TkLineNums</a>：一个用于 tkinter 的简单行号组件。通过在 GitHub 上创建账号来为 Moosems/TkLineNums 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/43)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/1245)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/abf2d97e88a2cc97e3029a578d2824895a88d0c1/stdlib/src/builtin/io.mojo#L362">mojo/stdlib/src/builtin/io.mojo (abf2d97e88a2cc97e3029a578d2824895a88d0c1) · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2308">条件 Trait 一致性 · Issue #2308 · modularml/mojo</a>：审查 Mojo 的优先级。我已经阅读了路线图和优先级，并认为此请求符合优先级。你的请求是什么？我认为如果能有条件的...</li><li><a href="https://youtu.be/pdJQ8iVTwj8?si=ML7lZfXAel9zEgj0&t=5763">Chris Lattner：编程与 AI 的未来 | Lex Fridman 播客 #381</a>：Chris Lattner 是一位传奇的软件和硬件工程师，曾在 Apple、Tesla、Google、SiFive 和 Modular AI 领导项目，包括 S... 的开发。</li><li><a href="https://github.com/modularml/mojo/issues/43#issuecomment-1542270428).">[功能请求] 实现联合类型（而非名义枚举）· Issue #43 · modularml/mojo</a>：摘要。我建议考虑在 Mojo 中添加 TypeScript 风格的联合类型，而不是 Rust 风格的名义枚举。鉴于 Mojo 计划扩展 Python，这种方法似乎会很有效...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1228586629413732423)** (23 条消息🔥):

- **Terminal UI 的 Pythonic 灵感**：寻找 **TUI 库灵感**的成员提到可以参考 [Textualize](https://github.com/Textualize)，该项目以创建 Rich 和 Textual 而闻名，尽管有些人认为其代码流难以解析。
- **llm.mojo 的发布引发社区关注**：[llm.mojo](https://github.com/dorjeduck/llm.mojo) 发布，这是 Andrej Karpathy 的 llm.c 向 **Mojo** 的移植版本。它通过 **向量化和并行化** 调整提升了性能，并欢迎社区反馈。
- **建议对 llm.mojo 进行协作贡献**：社区成员建议通过保持 C 实现与上游同步来改进 `llm.mojo`，这将允许在同一个仓库中同时提供 C 版本和 Mojo 移植版。
- **新人寻求 Llama 实现基准测试框架的帮助**：一位新人请求协助在他们的 Codespace 中设置 [lamatune 框架](https://github.com/tairov/lamatune)，并被鼓励在仓库中提交 Issue 以获取支持。
- **探索 Mojo 代码的 gRPC 支持**：一位社区成员提到他们从功能性的 Mojo 代码中获得了理想的结果，并询问是否有人正在研究 **gRPC 支持**，以便将其与现有的 C++ 代码连接以进行产品增强。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/dorjeduck/llm.mojo">GitHub - dorjeduck/llm.mojo: port of Andrjey Karpathy&#39;s llm.c to Mojo</a>：Andrjey Karpathy 的 llm.c 到 Mojo 的移植版。通过在 GitHub 上创建账户为 dorjeduck/llm.mojo 的开发做出贡献。</li><li><a href="https://github.com/tairov/lamatune">GitHub - tairov/lamatune: LLama implementations benchmarking framework</a>：LLama 实现基准测试框架。通过在 GitHub 上创建账户为 tairov/lamatune 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1228387825938989188)** (4 messages): 

- **使用 Mojo 进行行主序与列主序性能分析**：基于 [Modular 博客文章](https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy) 发起的讨论，探讨了计算机内存中矩阵的行主序和列主序的区别。提供了一个关联的 [GitHub 上的 Jupyter notebook](https://github.com/modularml/devrel-extras/blob/main/blogs/mojo-row-major-column-major/row_col_mojo.ipynb) 用于实际探索。

- **Mojo 矩阵错误解决方法**：在运行博客文章 notebook 中的代码遇到错误后，有人建议由于存在 bug，可以在 *执行每个单元格后重启 Jupyter 内核*，这被证明是一个有效的解决方案。
  
- **对博客见解的赞赏**：一位成员对博客文章中详尽的分析表示钦佩，称赞该贡献是出色的工作。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/blog">Modular: Blog</a>：在 Modular，我们相信优秀的文化是创建伟大公司的关键。我们工作的三个支柱是：打造用户喜爱的产品、赋能于人、以及成为一支不可思议的团队。</li><li><a href="https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy">Modular: Row-major vs. column-major matrices: a performance analysis in Mojo and NumPy</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：行主序 vs. 列主序矩阵：Mojo 和 NumPy 中的性能分析。</li><li><a href="https://github.com/modularml/devrel-extras/blob/main/blogs/mojo-row-major-column-major/row_col_mojo.ipynb">devrel-extras/blogs/mojo-row-major-column-major/row_col_mojo.ipynb at main · modularml/devrel-extras</a>：包含开发者关系博客文章、视频和研讨会的辅助材料 - modularml/devrel-extras
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1228753951520849972)** (3 messages): 

- **MojoAPI 与 CPython 的兼容性**：一位新成员询问 MojoAPI 是否可以与普通的 CPython 一起使用以缩短推理时间。另一位用户回答说，可以在 Mojo 中使用 `Python.module_import` 来导入可用的 Python 库，并且可以利用 Mojo 重写 Python 代码中缓慢的部分。

- **使用 MAX 基准测试进行高效推理**：一位成员询问除了使用提供的基准测试选项（如 `--mlperf-scenario` 和 `--num-threads`）之外，如何减少模型推理时间。消息中未提供进一步的回复或建议。
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1228475768468672585)** (55 messages🔥🔥):

- **Mojo Nightly 关于标准库目录结构的更新**：Mojo 团队提交了一个 PR，解决了由于新开源的标准库中目录结构与包命名不一致导致工具无法识别包名的问题。该修复预计将在下一个 `nightly/mojo` 版本中发布。

- **拟议的 Modular ML `StaticTuple` 增强功能**：目前正在讨论将 `StaticTuple` 更新为接受 `AnyType` 而非 `AnyRegType`，这将是一个实质性的进展。一名成员表示有兴趣参与此项工作，并提出了将 `StaticTuple` 重命名为 `Array` 的想法，同时强调在 PR 中应将重命名与功能重构分开。

- **Mojo 字符串的 Unicode 支持进展**：一名成员正在为 Mojo 标准库贡献 Unicode 支持，重点是最小化内存占用，并征求关于是通过 PR 还是正式提案进行的反馈。后续讨论中提到了为向量化编写的标准库方法。

- **关于 MLIR 和 Mojo Dialects 的澄清**：内部讨论澄清了 `pop` 是一个内部 MLIR dialect，代表 "parametric operators"（参数化算子），与 Mojo 参数系统相关。'pop.array' 是 `StaticTuple` 的基础，计划在下一个 nightly 版本中重写 tuple 以支持 `AnyType`。

- **修正 Mojo 中的项赋值（Item Assignment）**：一名成员指出并验证了一个问题：当 `__getitem__` 返回 `Reference` 时，诸如 `a[i] = value` 的项赋值解糖（desugars）不正确。它应该利用 `__setitem__` 或 `__refitem__`，如果定义正确，则无需其他两个方法。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/1871),">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 开发做出贡献。</li><li><a href="https://github.com/mzaks/mojo-unicode/blob/main/to_lower.mojo">mojo-unicode/to_lower.mojo at main · mzaks/mojo-unicode</a>：通过在 GitHub 上创建账户为 mzaks/mojo-unicode 开发做出贡献。</li><li><a href="https://mlir.llvm.org/docs/Dialects/IndexOps/">'index' Dialect - MLIR</a>：未找到描述</li><li><a href="https://mlir.llvm.org/docs/LangRef/">MLIR Language Reference - MLIR</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/pull/2294">[mojo-stdlib] Create `Array` type by lsh · Pull Request #2294 · modularml/mojo</a>：此 PR 创建了一个接受任何 CollectionElement 而不仅仅是 AnyRegType 的 Array 类型。另请参阅 Discord 上的此讨论。
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1228334768601694320)** (29 messages🔥): 

- **4090 已添加 P2P 支持**：**tinygrad** 的一项新更新使得在 NVIDIA 4090 GPU 上使用 P2P 支持成为可能，通过修改 NVIDIA 驱动程序，在 tinybox green 上实现了令人印象深刻的 14.7 GB/s AllReduce 性能。技术细节在[此处](https://x.com/__tinygrad__/status/1778676746378002712)发布的帖子中讨论。
- **CUDA 挑战“十亿行挑战”（One Billion Row Challenge）**：tspeterkim_89106 分享了一篇关于他们在 CUDA 中实现 [One Billion Row Challenge](https://1brc.dev/) 的博客文章，该实现在 V100 上运行耗时 16.8 秒，并公开邀请 CUDA 社区改进该[解决方案](https://github.com/tspeterkim/cuda-1brc/blob/main/fast.cu)。社区反馈提出了可能的改进方案，以及在 [4090 GPU 上运行更快的 6 秒版本](https://tspeterkim.github.io/posts/cuda-1brc)。
- **酷材料委员会（Council for Cool Materials）**：一名成员建议创建一个专门用于分享有价值资料的频道，作为回应，#suggestions 被重新调整用途以促进此类交流。
- **探索 CUDA 中的并行性**：在一次技术交流中，zippika 分享了关于优化 CUDA 性能的见解，讨论了 warp 级原语（warp level primitives）的潜在用途、CPU 速度对 kernel 执行的影响，以及预分配缓冲区和利用 `cudaMemcpyAsync` 提高效率等策略。利用 CUDA kernel 的异步操作来隐藏部分拆分时间被强调为一种潜在的优化技术。
- **应对垃圾机器人围攻**：一名成员报告了频道中涌入的大量垃圾机器人（spambots），通过标记 moderator 角色向管理员发出警报。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tspeterkim.github.io/posts/cuda-1brc">The One Billion Row Challenge in CUDA: from 17m to 17s</a>：未找到描述</li><li><a href="https://x.com/__tinygrad__/status/1778676746378002712">来自 tiny corp (@__tinygrad__) 的推文</a>：我们通过修改 NVIDIA 驱动程序为 4090 添加了 P2P 支持。适用于 tinygrad 和 nccl (又名 torch)。在 tinybox green 上实现 14.7 GB/s AllReduce！
</li>
</ul>

</div>

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1228274705975476267)** (77 条消息🔥🔥): 

- **PyTorch 版本兼容性与性能问题**：用户讨论了由于 `error: namespace "at::cuda" has no member` 导致构建特定 Kernel 失败的问题，该问题通过使用 torch-nightly 构建版本（而非 PyTorch 2.2）得到了解决。基准测试显示，在 4090 上，自定义 Kernel 的速度比 `torch.matmul` **慢 1.20 倍**。[提供了基准测试代码和结果](https://gist.github.com/mobicham/7fb59e825fed0831fccf44752cb21214)。

- **探索潜在的优化路径**：有建议提出在独立的 CUDA Stream 中运行独立的 QKV matmuls，并考虑到 PyTorch 的 `torch.compile` 可能会在后台优化性能，从而影响对比基准测试的结果。

- **融合操作算子（Fused Operation Kernels）的探索**：用户对添加更多融合操作算子表现出兴趣，例如使用更低位宽的算子，并参考了 `cutlass int4xint4` Kernel，同时建议解决 matmul 实现中的复杂性。

- **Stable Fast 与全图编译（Full-Graph Compilation）讨论**：用户讨论了利用 `stable_fast` 进行编译的实用性，在某些情况下它似乎比开启 `fullgraph=True` 的 `torch.compile` 更快，但在用于逐 token 解码（token-by-token decoding）时，使用 `fullgraph=True` 可能会报错。用户还提出了关于[模型编译问题](https://gist.github.com/mobicham/0e51c9f572721a76a5ac1e06fea533e9#file-stable_fast_llama_example-py-L14)的担忧，以及 `enable_cuda_graph=False` 可能带来的好处。

- **低精度计算中极具前景的结果**：据报告，通过使用 `stable_fast` 和其他优化手段，性能已接近 int8 量化的 TensorRT 优化速度，展示了在 Stable Diffusion 等模型中进行高效低精度计算的潜力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gist.github.com/mobicham/0e51c9f572721a76a5ac1e06fea533e9#file-stable_fast_llama_example-py-L14">stable_fast_llama_example.py</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://huggingface.co/papers/2404.07839">论文页面 - RecurrentGemma: Moving Past Transformers for Efficient Open Language Models</a>：未找到描述</li><li><a href="https://github.com/chengzeyi/stable-fast/blob/main/src/sfast/jit/passes/__init__.py">stable-fast/src/sfast/jit/passes/__init__.py</a>：NVIDIA GPU 上针对 HuggingFace Diffusers 的最佳推理性能优化框架。</li><li><a href="https://gist.github.com/mobicham/7fb59e825fed0831fccf44752cb21214">hqq_hgemm_benchmark.py</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/spcl/QuaRot/blob/main/quarot/kernels/gemm.cu#L32">QuaRot/quarot/kernels/gemm.cu</a>：QuaRot 的代码，实现大语言模型的端到端 4-bit 推理。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1228267331457912842)** (15 条消息🔥): 

- **NVIDIA GPU 上启用 P2P**：**Tinygrad** 通过破解开源 GPU 内核模块，成功在 NVIDIA 4090 和 4070 TI Super GPU 上启用了 P2P 支持。正如在 [Twitter](https://twitter.com/__tinygrad__/status/1778677126092509611) 上分享的那样，这些修改使所有 all reduce 操作实现了 **58% 的提速**，代码可在 [GitHub](https://github.com/tinygrad/open-gpu-kernel-modules) 上获取。

- **收集 Peer-To-Peer 性能数据**：用户正在整合 P2P 性能数据，并讨论已知“反特性”（anti-features）的潜在问题。可以通过 [此处](https://github.com/cuda-mode/p2p-perf) 向包含基准测试的仓库提交贡献。

- **分享 P2P 性能测试结果**：一位用户向 [p2p-perf 仓库](https://github.com/cuda-mode/p2p-perf/pull/1) 提交了他们的 **RTX 4070 TI Super 双卡构建基准测试**，用于测量 Peer-To-Peer 性能。

- **高效的三值权重（Ternary Weight）表示**：目前使用 fp16 表示三值权重的 **BitNet 实现** 被认为效率低下，正在考虑使用自定义 2-bit 张量或使用 int8 打包等替代方案。参考 [这段代码](https://github.com/mobiusml/hqq/blob/master/hqq/core/bitpack.py#L43) 提供了一种位打包（bit-packing）的解决方案。

- **PyTorch 示例中的分布式 CI**：PyTorch/examples 仓库现在包含分布式 CI，使贡献者更容易测试其代码。对于有兴趣为分布式示例做贡献的人，可以在 [此处](https://github.com/pytorch/examples/pull/1243/files) 找到相关的 Pull Request，在 [此处](https://github.com/pytorch/examples/tree/main/distributed) 找到贡献目录。

<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/bitpack.py#L43">hqq/hqq/core/bitpack.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/cuda-mode/p2p-perf/pull/1">Add rtx-4070-ti-super-2x benchmarks by morgangiraud · Pull Request #1 · cuda-mode/p2p-perf</a>: 暂无描述</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: NVIDIA Linux open GPU with P2P support</a>: 支持 P2P 的 NVIDIA Linux 开源 GPU 内核模块。通过在 GitHub 上创建账号来为 tinygrad/open-gpu-kernel-modules 做出贡献。</li><li><a href="https://github.com/cuda-mode/p2p-perf">GitHub - cuda-mode/p2p-perf: measuring peer-to-peer (p2p) transfer on different cuda devices</a>: 在不同 CUDA 设备上测量点对点 (P2P) 传输 - cuda-mode/p2p-perf</li><li><a href="https://github.com/pytorch/examples/pull/1243/files">Update TP examples to align with tutorials by wanchaol · Pull Request #1243 · pytorch/examples</a>: 如题</li><li><a href="https://github.com/pytorch/examples/tree/main/distributed">examples/distributed at main · pytorch/examples</a>: 一系列围绕 PyTorch 在视觉、文本、强化学习等领域的示例 - pytorch/examples
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1228778979440197672)** (1 条消息): 

- **CUDA 变得简单**: 一位成员推荐了 [PyTorch Dev 播客集](https://open.spotify.com/episode/3dhD1irFc2gDNkLwUveKum?si=6DPutTGlSqaU0oflNbBW7Q)，作为学习 CUDA 概念的良好起点。它涵盖了 GPU 基础知识、CUDA 编程模型、异步执行挑战以及使用 **PyTorch** 的 CUDA kernel 设计原则。
- **CUDA 学习资源与工具**: 对于有兴趣深入研究 CUDA 的人，该播客集推荐了 **关于 CUDA 语义的 PyTorch 文档** (*https://pytorch.org/docs/stable/notes/cuda.html*) 以及《Programming Massively Parallel Processors》一书。提到的调试工具包括设置环境变量 `CUDA_LAUNCH_BLOCKING=1` 和使用 `cuda-memcheck` (*https://docs.nvidia.com/cuda/cuda-memcheck/index.html*)。

**提到的链接**: <a href="https://open.spotify.com/episode/3dhD1irFc2gDNkLwUveKum?si=6DPutTGlSqaU0oflNbBW7Q)">Just enough CUDA to be dangerous</a>: 在 Spotify 上收听来自 PyTorch Developer Podcast 的这一集。曾经想学习 CUDA 但不知道从哪里开始？在这 16 分钟的节目中，我尝试塞进尽可能多的 CUDA 知识...

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1228315933882454026)** (5 条消息): 

- **并行计算免费在线课程**: [阿尔托大学 (Aalto University)](https://ppc-exercises.cs.aalto.fi/courses) 提供“Programming Parallel Computers”课程的公开版本，涵盖 OpenMP、向量化和 GPU 编程的材料。课程为自主进度，提供自动化的代码测试和基准测试，但不提供学分或人工评分。

- **利用 NVIDIA GPU 进行并行数学运算**: torch/TensorFlow 可以利用 CUDA C/C++ 函数为 NVIDIA GPU 编译机器代码，利用 GPU 的多核进行并行处理，相比 Python 中顺序执行的 CPU 处理具有显著优势。

- **PMPP 讲座从排除初始后勤信息开始**: 第一节 PMPP 讲座从第 31 分钟开始，以跳过介绍性的后勤内容，并建议稍后自行查看这些细节。

- **调整会议时间的投票**: 提醒第一节 PMPP 会议的参与者在 invite 频道中的投票中进行投票，以帮助为未来的会议寻找更好的时间段。

- **请求 CPU vs GPU 架构视频**: 一位成员向另一位成员索要解释 CPU 和 GPU 架构差异的视频链接；然而，在提供的消息历史记录中尚未提供该链接。

**提到的链接**: <a href="https://ppc-exercises.cs.aalto.fi/courses">Courses</a>: 暂无描述

  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1228743555821080656)** (9 条消息🔥): 

- **3D 线程配置的好奇**: 一位成员思考了在 CUDA 中将线程排列为 3D 之后再将其扁平化的必要性。聊天中的建议是查看第 3.2 节以获得更好的理解，暗示后续章节将对该主题进行更深入的阐述。

- **CUDA Block 维度对比**：针对使用 CUDA 处理向量时两组不同配置参数的最优选择展开了讨论。一位成员建议 *两种配置都可以*，并提到 **thread coarsening**（线程粗化）作为一种替代方案，即在 warp 内部使用 for 循环。
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1229133117634248846)** (6 messages): 

- **招募 YouTube 录制志愿者**：正在招募志愿者协助 CUDA MODE Discord 频道的每周录制任务。志愿者负责录制、剪辑并上传内容到 YouTube，将获得特殊角色和直接署名，以及来自社区的感谢。
- **提供录制任务协助**：一名成员迅速响应并志愿参与每周录制工作，表示该任务是可以胜任的。
- **询问近期讲座录制情况**：一名成员询问特定周末的讲座是否已录制。
- **确认讲座录制**：确认该周末的讲座确实已录制并已发布。
- **努力提升讲座录制质量**：另一名成员表示他们正在重新录制以提高素材质量，并计划在次日上传。
  

---


**CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1229371863176843265)** (2 messages): 

- **在 torchao 中优化 Tensor 布局**：一名成员提出了 **torchao** 优化 Tensor 布局的可能性，即通过存储权重的方式使其已经为矩阵乘法中的 swizzles 对齐。他们提到，虽然 element-wise 操作对数据布局不敏感，但矩阵乘法的优化可能涉及 swizzling 和 padding，以便从 cublas 等快速矩阵乘法库中获益。
- **torch.compile 已处理部分优化**：针对 Tensor 布局优化的建议，另一名成员指出 padding 和布局优化可能已经由 **torch.compile** 处理，并提供了 PyTorch 仓库中相关部分的链接。[Padding 优化](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L420) 和 [布局优化](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L261) 在配置文件中被提及。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L420">pytorch/torch/_inductor/config.py at main · pytorch/pytorch</a>：Python 中具有强 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L261">pytorch/torch/_inductor/config.py at main · pytorch/pytorch</a>：Python 中具有强 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1228270666684563568)** (3 messages): 

- **Google 涉足 RNN 和 Transformers**：一名成员引用了 Google 在整合 RNN 与 Transformers 方面的工作，暗示了机器学习领域的一项进展。
- **发布 Ring Attention 深度解析**：一篇关于 **Ring Attention** 的解析文章已由 [Kilian Haefeli, Simon Zirui Guo, 和 Bonnie Li 发布](https://coconut-mode.com/posts/ring-attention/)，详细介绍了它从 naive attention 到 block-wise parallel attention 的演进。这种方法通过使用多个设备，可能使模型扩展到**近乎无限的上下文窗口**。
- **提高上下文长度标准**：LLM 的上下文长度飞速增长，从 **GPT 3.5 的 16k tokens** 到 **Gemini 1.5 Pro 的 100 万 tokens**，由于能够利用更多信息，开启了令人兴奋的新用例。
- **赞赏清晰的视觉效果**：一名成员称赞了 Ring Attention 解析文章中的**动画**，强调了良好的视觉辅助在理解复杂概念中的价值。


**提到的链接**：<a href="https://coconut-mode.com/posts/ring-attention/">Ring Attention Explained | Coconut Mode</a>：为语言模型提供近乎无限的上下文窗口。

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1228672477618507786)** (2 messages): 

- **表达感谢**：一名成员对 KJ 为社区做出的贡献表示感谢，认可其对每个参与者的积极影响。

- **改变游戏规则的事物**：一名成员分享了 @yacineMTB 的 [一条推文链接](https://x.com/yacinemtb/status/1778867085608644875)，并强调 **“它改变了一切！！！！！！”**，但未提供关于内容或背景的进一步细节。

**提到的链接**: <a href="https://x.com/yacinemtb/status/1778867085608644875">kache (dingboard.com) (@yacineMTB) 的推文</a>: 它改变了一切!!!!!!

---

**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1228370522341900298)** (12 条消息🔥): 

- **CUDA MODE: HQQ 实现更新**：一位贡献者更新了他们在 [GitHub](https://github.com/pytorch-labs/gpt-fast/commit/551af74b04ee1e761736fbccfe98d37137d04176) 上的分支，将 HQQ W_q 权重转换为 gpt-fast 权重。此次更新使他们在启用 `--compile` 选项的情况下，实现了 5.375 的困惑度 (ppl) 和 200 tokens/s 的 token 生成速度。

- **HQQ 优化说明**：在测试 HQQ 时，确认应始终在 quant_config 中开启 `meta['optimize']`，因为这会带来性能提升。

- **HQQ vs. HQQ+**：HQQ 是一种无需校准的训练后量化 (post-training quantization) 技术，而 HQQ+ 则包括带有可训练低秩适配器 (low-rank adapters) 以及 scale/zero 参数的 HQQ。HQQ+ 可用于低至 1 bit 的位宽，并有潜力达到或超过全精度模型的表现。

- **低比特推理的挑战**：尽管前景广阔，但低比特推理仍面临挑战，例如为低比特操作寻找高效的 matmul kernels，以及解决 1 bit 或 2 bit 所需的小 group-sizes 带来的 VRAM 限制。

- **Torchao Int4 Kernel 带来的性能飞跃**：对 Transformer 模型的生成流水线进行了完全重写，现在支持 torchao int4 kernel，性能显著提升，从 FP16 的 59 tokens/sec 提高到了 152 tokens/sec。

**提到的链接**: <a href="https://github.com/pytorch-labs/gpt-fast/commit/551af74b04ee1e761736fbccfe98d37137d04176">HQQ 4 bit llama 2 7b · pytorch-labs/gpt-fast@551af74</a>: export MODEL_REPO=meta-llama/Llama-2-7b-hf scripts/prepare.sh $MODEL_REPO python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4-hqq --groupsize 64 python generate.py --...

---

**CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1228481314676805753)** (3 条消息): 

- **对可视化过程的困惑**：一位成员表示 **GIF** 中描述的最后一步令人困惑，表明可能需要进一步澄清或额外的指导。
- **SVGs 作为代码表示的媒介**：该成员目前正在将输出写入 **SVG** 以进行可视化，展示了其项目中动手实践的方法。
- **寻求可视化控制结构的方法**：目前正致力于设计一种方法来有效可视化 **if 语句** 和 **for 循环** 等编程结构，这表明其重点在于通过视觉辅助增强代码的可理解性。

---

**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1228280026500436008)** (135 条消息🔥🔥): 

- **CUDA Softmax 集成挑战**：由于因果掩码 (causal mask) 导致 exp(INF-INF)，softmax 实现面临 NaN 问题；将 -INF 切换为 -FLT_MAX 解决了该问题。讨论还围绕在 reduce 调用中集成任意函数的可读性和复杂性展开，最终倾向于为了清晰起见使用独立的 reductions。

- **Cooperative Groups 带来的编译器成本**：成员们注意到，由于编译器无法获知确切的 threadblock 大小，使用超过 32 个线程的 cooperative groups 效率较低。这通常会导致性能欠佳和难以预测的行为，即使使用 godbolt 等详细工具也难以缓解。

- **讨论 Online Softmax 以优化性能**：对 online softmax 算法进行了剖析，重点关注顺序 reduction 和并行 reduction 方法之间的权衡。建议包括需要使用各种 "C" 值进行更全面的基准测试，以测试性能改进。

- **研究 cuBLAS 和 CUTLASS 以实现最佳 Kernel 使用**：对话探讨了使用 cuBLAS、cuBLASLt 和 CUTLASS 来优化 kernel 操作。虽然 cuBLAS 的简单集成在不增加复杂性的情况下提升了性能，但 CUTLASS 将涉及集成大型头文件和大量的编译时工作，可能会使代码库复杂化。

- **兼顾巅峰性能与教育价值**：成员们希望 LLM.c 既能通过使用 CUTLASS 和 cuDNN 等工具实现高效，又能通过从简单到复杂的各种手写 kernels 具有教育意义。最终目标是使用易于理解和维护的代码来挑战 PyTorch 的性能，甚至可能重写以确保矩阵具有性能友好的形状。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/cudaCompressibleMemory/compMalloc.cpp">cuda-samples/Samples/3_CUDA_Features/cudaCompressibleMemory/compMalloc.cpp at master · NVIDIA/cuda-samples</a>: CUDA 开发者示例，展示了 CUDA Toolkit 中的各项功能 - NVIDIA/cuda-samples</li><li><a href="https://github.com/karpathy/llm.c/pull/98">use cublaslt and optionally tf32, which fuses bias by karpathy · Pull Request #98 · karpathy/llm.c</a>: cc @ademeure，目前这只是 matmul 的 cuBLASLt 版本，包含一系列细微改动。例如，我尝试移除了全局变量和设置，按照惯例我更倾向于尽可能地...</li><li><a href="https://github.com/karpathy/llm.c/pull/79/files#diff-a00ef278da39f24a9d5cb4306c15626b921d437013fb5aa60ac2d8df6b5a5508R362)">Include the online softmax CPU code and a fully parallelized GPU kernal by lancerts · Pull Request #79 · karpathy/llm.c</a>: 包含 online softmax CPU 代码（源自论文 Online normalizer calculation for softmax）。其原生移植到 GPU kernel 5（用于教学对比）。包含完全并行的 kernel ...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[recording-crew](https://discord.com/channels/1189498204333543425/1229286073104994344/1229286494187946014)** (12 messages🔥): 

- **扩大 CUDA MODE 管理规模**：由于当前管理员在管理工作组的新进展方面应接不暇，该频道正在寻求更多管理员。
- **分享讲座录制流程**：列出了 CUDA MODE 详细的 7 步讲座录制流程，涉及 OBS streamlabs 和 Da Vinci resolve 等工具，以及活动封面图协调和会议音检。最终视频上传至 [CUDA MODE 的 YouTube](https://www.youtube.com/@CUDAMODE) 频道，资料也会在 [其 GitHub 讲座仓库](https://github.com/cuda-mode/lectures) 中更新。
- **流程改进公开征集**：公开邀请他人提出新方法或承担录制流程的部分工作，以促进社区平稳增长。
- **潜在的直播转型**：建议直接在 YouTube 上直播讲座，以应对不断增长的成员规模，并利用 YouTube 的编辑器进行剪辑；同时提醒不要使用 AirPods 以防止录制事故。
- **协作完成即将到来的录制**：两名成员正在协调处理即将到来的 ADA-CUDA 讲座录制，初步试运行定于本周的一次演讲。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/live/24EWxyEMp1w?si=bfXVZPK4hJi_g49I)">HF Reading Group: Mobile ALOHA</a>: 我们邀请了 Mobile ALOHA 的一位作者在 Hugging Face Discord 的 #reading-group 频道进行讨论。遗憾的是，其他人的声音没有被录制下来...</li><li><a href="https://github.com/cuda-mode/lectures">GitHub - cuda-mode/lectures: Material for cuda-mode lectures</a>: cuda-mode 讲座资料。通过在 GitHub 上创建账号为 cuda-mode/lectures 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1228278760445579354)** (308 messages🔥🔥): 

- **解决了 Connector API 的误解**：一名成员在使用 **Cohere 的 connector API** 时遇到问题；由于 connector URL 格式不正确导致报错。小组讨论得出，有效的 connector URL 必须以 `/search` 结尾才能正常工作。
- **关于 Cohere 微调可用性的澄清**：另一名成员询问了 **Cohere 基础模型** 微调的可用性，讨论指出可以通过 [Cohere 控制面板](https://dashboard.cohere.com/fine-tuning/) 访问。Sandra 确认可以在 **Amazon Bedrock** 上进行微调，且 **command-r** 模型也已在 [AWS](https://aws.amazon.com/marketplace/seller-profile?id=87af0c85-6cf9-4ed8-bee0-b40ce65167e0) 上线。
- **聊天机器人界面和 Grounded AI 进展**：Sandra 分享了关于 Cohere 功能的更新，如聊天机器人界面演示环境 **Coral**，并暗示即将发布用于在 AWS 和私有设置上启用 connector 的 **toolkits**。
- **探讨 Cohere 模型及可能的硬件限制**：交流了 AI 模型的性能表现，特别是讨论了 **command-r-plus** 模型在包括 NVIDIA 4090 在内的各种硬件上的运行时间，以及 **TPU** 的可用性和性能。
- **为新人提供学习资源和支持**：针对新社区成员关于学习材料的咨询，提到了 **LLM University** 等免费课程，并引导至 [Cohere 文档](https://docs.cohere.com/docs/llmu) 以进一步学习。
<div class="linksMentioned">

_

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.ai/blog/grok-1.5v">Grok-1.5 Vision Preview</a>：未找到描述</li><li><a href="https://discord.gg/Y4msga6k?event=1208132674762575994">加入 Cohere 社区 Discord 服务器！</a>：Cohere 社区服务器。欢迎来聊聊 Cohere API、LLMs、Generative AI 以及相关的一切。 | 15435 名成员</li><li><a href="https://docs.cohere.com/reference/chat">Chat API 参考 - Cohere 文档</a>：未找到描述</li><li><a href="https://sites.google.com/cohere.com/c4ai-community/community-programs/computer-vision?authuser=0)?">社区 - 计算机视觉</a>：频道：#computer-vision 共同负责人：Benedict - Discord 上的 @Harkhymadhe，Twitter 上的 @Arkhymadhe 物流：发生时间：每月第二个星期二上午 8 点 PT 欢迎添加论文/文章...</li><li><a href="https://docs.cohere.com/docs/connectors">Connectors 简介 - Cohere 文档</a>：未找到描述</li><li><a href="https://dashboard.cohere.com/fine-tuning/?">登录 | Cohere</a>：Cohere 通过一个易于使用的 API 提供对先进 Large Language Models 和 NLP 工具的访问。免费开始使用。</li><li><a href="https://en.wikipedia.org/wiki/Special:Search?search=glucose",">glucose&quot;, - 搜索结果 - 维基百科</a>：未找到描述</li><li><a href="https://aws.amazon.com/marketplace/seller-profile?id=87af0c85-6cf9-4ed8-bee0-b40ce65167e0">AWS Marketplace: Cohere</a>：未找到描述</li><li><a href="https://coral.cohere.com/?s=t">登录 | Cohere</a>：Cohere 通过一个易于使用的 API 提供对先进 Large Language Models 和 NLP 工具的访问。免费开始使用。</li><li><a href="https://dashboard.cohere.com/playground/chat">登录 | Cohere</a>：Cohere 通过一个易于使用的 API 提供对先进 Large Language Models 和 NLP 工具的访问。免费开始使用。
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1229333844424134676)** (2 条消息): 

- **Command-R 核心模块备受赞誉**：一位成员庆祝将 **Command-R** 作为核心模块之一引入，并赞扬了 Cohere 的努力。
- **加入量化金融领域**：分享了关于 Quant Based Finance 的推广及 Beta 测试邀请链接。感兴趣的人士请访问 [Quant Based Finance](https://quantfino.com/join-beta)。
- **Rubiks.AI 招募 Beta 测试人员**：宣布推出一款先进的研究助手和搜索引擎。Beta 测试人员将获得为期 2 个月的免费高级访问权限，可使用包括 **Claude 3 Opus、GPT-4 Turbo、Mistral Large** 等在内的一系列模型，访问地址为 [Rubiks.AI](https://rubiks.ai/)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://rubiks.ai/">Rubik's AI - AI 研究助手与搜索引擎</a>：未找到描述</li><li><a href="https://quantfino.com/join-beta">Quantfino - 强大的 AI 驱动金融之家</a>：Quantfino 是由 LLM 驱动和 Langchain 辅助的金融分析平台。
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1228267768537812992)** (172 条消息🔥🔥): 

- **在本地运行多个模型**：一位用户询问是否可以在单台机器上同时运行多个模型，另一位成员确认只要资源充足就是可行的。他们提到曾在一块 GPU 上成功运行了 6 个 A1111 模型实例。

- **关于模型优化的讨论**：关于 LLM.c 优化的对话，成员们讨论了其设计初衷并非为了追求极致速度。一位社区成员特别解释说，其重点是展示 GPT-2 可以在不依赖库的情况下用 C 语言实现，而不是为了速度。

- **Hugging Face Spaces 连接问题**：多位用户报告了访问 Hugging Face Spaces 的问题，页面加载为空白或提示无效响应错误。成员们建议检查路由器和 Wi-Fi 设置，一些用户通过禁用拦截网站的安全防护功能成功解决了问题。

- **聊天机器人建模问题**：用户咨询了创建和运行聊天模型的各个方面，包括数据集格式化和优化建议。一位用户详细说明了他们希望在聊天机器人中使用 langchain 和 neuralhermes-2-pro 以维持对话记忆。

- **工作与项目咨询**：讨论包括一位用户寻求关于招聘模型实现和训练人员的建议，而另一位用户分享了他们日益增长的自由职业工作量，并正在寻找团队成员来处理不断增加的项目。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://us05web.zoom.us/j/87189451768?pwd=6PlOsbZ2AbaaIhwvaa6kjyvn0NelT2.1">加入我们的 Cloud HD Video Meeting</a>：Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，适用于移动端、桌面端和会议室系统的视频和音频会议、聊天及网络研讨会。Zoom ...</li><li><a href="https://www.missiononesmile.org/generation-generated.html">Mission One Smile</a>：未找到描述</li><li><a href="https://rubiks.ai/">Rubik's AI - AI 研究助手与搜索引擎</a>：未找到描述</li><li><a href="https://github.com/leejet/stable-diffusion.cpp">GitHub - leejet/stable-diffusion.cpp: 纯 C/C++ 实现的 Stable Diffusion</a>：纯 C/C++ 实现的 Stable Diffusion。通过在 GitHub 上创建账户，为 leejet/stable-diffusion.cpp 的开发做出贡献。</li><li><a href="https://status.huggingface.co/">
Hugging Face 状态
</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1228457025319473182)** (6 条消息): 

```html
<ul>
  <li><strong>解码 CLIP 的概念：</strong>由 <a href="https://www.linkedin.com/in/matthewbrems/">Matthew Brems</a> 撰写的一篇博客，旨在简化对 <strong>OpenAI 的 CLIP 模型</strong>的理解，涵盖了它是什么、其工作原理及其重要性。这个来自 OpenAI 的多模态模型通过同时在图像和文本上进行训练，为计算机视觉提供了一种新颖的方法。<a href="https://www.kdnuggets.com/2021/03/beginners-guide-clip-model.html">CLIP 模型初学者指南</a>。</li>
  <li><strong>应用 Zero-Shot 分类：</strong>在学习了使用 <strong>bart-large-mnli 模型</strong>进行 Zero-Shot 分类后，创建了一个演示，展示了其在分类来自 <a href="https://thebasemesh.com/">thebasemesh.com</a> 的 3D 资产方面的应用，该网站是一个为创意项目提供基础网格（base meshes）的免费资源。在 <a href="https://www.youtube.com/watch?v=jJFvOPyEzTY">YouTube</a> 上观看探索过程。</li>
  <li><strong>使用 Podman 实现高性价比 AI：</strong>分享了一个演示视频，展示了如何使用 <strong>Podman</strong> 构建容器化的生成式 AI 应用程序，这为本地和云端部署提供了一种具有成本效益的替代方案。该技术承诺在部署选项方面提供控制权和选择权。<a href="https://youtu.be/3iEhFKIDXp0?si=nQJt-PBJUB960gpU">YouTube</a> 上的教程。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/safetensors/index">Safetensors</a>：未找到描述</li><li><a href="https://www.kdnuggets.com/2021/03/beginners-guide-clip-model.html">CLIP 模型初学者指南 - KDnuggets</a>：CLIP 是计算机视觉和自然语言处理之间的桥梁。我在这里通过一篇通俗易懂且有趣的文章为你解析 CLIP！在这篇博文中，我将介绍 CLIP 是什么，CLIP 如何工作...</li><li><a href="https://blog.roboflow.com/how-to-use-openai-clip/">如何尝试 CLIP：OpenAI 的 Zero-Shot 图像分类器</a>：本周早些时候，OpenAI 在计算机视觉领域投下了一枚炸弹。</li><li><a href="https://youtu.be/3iEhFKIDXp0?si=nQJt-PBJUB960gpU">Podman Build - 适用于注重成本的 AI</a>：关于 @podman 如何构建可部署到本地或你首选云提供商的容器化生成式 AI 应用程序的快速演示。这种选择是 ...</li><li><a href="https://thebasemesh.com/">THE BASE MESH</a>：一个包含 1000 多个基础网格的公共库。所有资产均为真实世界比例且已展开（unwrapped）！100% 免费。CC0 许可。</li><li><a href="https://www.youtube.com/watch?v=jJFvOPyEzTY">机器学习：使用 bart-large-mlni 的 Zero-Shot 分类演示</a>：#machinelearning #zeroshotclassification #naturallanguageinference #transferlearning Zero-Shot Classification https://huggingface.co/tasks/zero-shot-classific...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1228284633310040144)** (10 条消息🔥): 

- **将 AI 与文档检索融合**：Medium 上的一篇文章介绍了集成 **Colbert-based Agent** 的 **LlamaIndex**，以增强带记忆的文档检索。教程可以在[这里](https://medium.com/ai-advances/enhancing-document-retrieval-with-memory-a-tutorial-for-llamaindex-with-colbert-based-agent-1c3c47461122)找到。
  
- **PyTorch 遇上 Blender**：分享了用于将 PyTorch 与 Blender 集成的 GitHub 仓库：[btorch](https://github.com/j20232/btorch) 包含用于 Blender 的 PyTorch 工具，而 [pytorch-blender](https://github.com/cheind/pytorch-blender.git) 提供将 Blender 无缝、实时集成到 PyTorch 数据流水线的功能。

- **通过群等变性提升 CNN**：GitHub 上的 [e2cnn 库](https://github.com/QUVA-Lab/e2cnn) 通过提供 E(2)-等变 CNN 的 PyTorch 实现来增强卷积神经网络（CNNs），这对于需要旋转不变性的任务非常有益。

- **面向 Text-to-Music AI 创新者的数据集**：可在 [Papers with Code](https://paperswithcode.com/dataset/magnatagatune) 上获取的 **MagnaTagATune 数据集** 提供了一套完整的音乐片段，并附带丰富的人工生成标签，为任何有兴趣创建开源 Text-to-Music 服务的人提供了一个潜在的起点。

- **凸显 AI 理解视频的能力**：一段名为 "Large Language Models Are Zero Shot Reasoners" 的 YouTube 视频展示了 Edge 浏览器生成视频高光的功能；[一位用户的体验感悟](https://youtu.be/fTOIzQAHNtc) 被分享出来，展示了该功能的有效性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=GuhzATF13TOmwUMU">Neural networks</a>：学习神经网络和反向传播（backpropagation）的基础知识，这是现代世界最重要的算法之一。</li><li><a href="https://paperswithcode.com/dataset/magnatagatune">Papers with Code - MagnaTagATune Dataset</a>：MagnaTagATune 数据集包含 25,863 个音乐片段。每个片段是长达 29 秒的摘录，属于 5223 首歌曲、445 张专辑和 230 位艺术家之一。这些片段涵盖了广泛的流派，如 Cl...</li><li><a href="https://g.co/gemini/share/e8962ab90c1c">‎Gemini - Cours Data Science, IA et GenAI</a>：由 Gemini 创建</li><li><a href="https://github.com/j20232/btorch">GitHub - j20232/btorch: btorch 包含用于 Blender 的简单 PyTorch 实用工具。</a>：btorch 包含用于 Blender 的简单 PyTorch 实用工具。 - j20232/btorch</li><li><a href="https://github.com/cheind/pytorch-blender.git">GitHub - cheind/pytorch-blender: :sweat_drops: 将 Blender 无缝、分布式、实时地集成到 PyTorch 数据流水线中</a>：:sweat_drops: 将 Blender 无缝、分布式、实时地集成到 PyTorch 数据流水线中 - cheind/pytorch-blender</li><li><a href="https://github.com/QUVA-Lab/e2cnn">GitHub - QUVA-Lab/e2cnn: 用于 PyTorch 的 E(2)-等变 CNN 库</a>：用于 PyTorch 的 E(2)-等变 CNN 库。通过在 GitHub 上创建账户为 QUVA-Lab/e2cnn 的开发做出贡献。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1228563022000423002)** (17 条消息🔥): 

- **生成逼真 AI 皮肤的技巧**：一位成员分享了在 AI 作品中生成更逼真皮肤纹理的技巧，建议在 negative prompt 中使用 "perfect skin"，并在 positive prompt 中添加特定特征如 "(moles:0.8)"，以实现带有痣的自然外观。
- **通过 Hugging Face 构建企业级 AI 枢纽**：一位用户宣布他们开发了一个 **Enterprise AI Hub**，该枢纽集成了各种企业应用程序，并能够进行基于 prompt 的智能交互，展示了 Hugging Face 和 OpenAI 在企业解决方案中的多功能性。该枢纽使用 Java、Spring Boot 构建，并连接了宠物店、书店和缺陷跟踪系统等系统。[*EnterpriseAIHub* on Spaces](https://huggingface.co/spaces/VishalMysore/EnterpriseAIHub)
- **通过反事实初始（Counterfactual Inception）减轻幻觉**：一位成员介绍了一篇名为 "What if...?: Counterfactual Inception to Mitigate Hallucination Effects in Large Multimodal Models" 的论文，概述了一种在无需额外指令微调（instruction tuning）的情况下提高 AI 响应准确性、减少幻觉的新方法。项目代码已在 GitHub 上发布。[论文](https://arxiv.org/abs/2403.13513) | [GitHub 仓库](https://github.com/ivy-lvlm/counterfactual-inception)
- **RicercaMente：绘制数据科学演进图谱**：分享了一个名为 RicercaMente 的项目，旨在通过重要的科学论文追溯数据科学的历史，邀请他人参与贡献或在 GitHub 上提供支持。据称参与该项目不需要编程，仅需约 10 分钟时间。[GitHub 仓库](https://github.com/EdoPedrocchi/RicercaMente)
- **机器人足球胜利与温度间隙填充**：一位成员讨论了他们参与的两个项目：一个项目助力赢得了类人机器人足球世界杯 RoboCup，并发布了相关进展和源代码；另一个项目旨在利用部分卷积（partial convolutions）来填充地球监测中缺失的温度数据。[RoboCup 论文](https://arxiv.org/pdf/2302.02956.pdf) | [温度间隙填充研究](https://www.mdpi.com/1424-8220/24/5/1604)
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://kunalmishra.info/">Text Generation</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/VishalMysore/EnterpriseAIHub">EnterpriseAIHub - VishalMysore 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/Merkoba/Meltdown">GitHub - Merkoba/Meltdown: llama.cpp 和 ChatGPT 的界面</a>：llama.cpp 和 ChatGPT 的界面。通过在 GitHub 上创建账户来为 Merkoba/Meltdown 的开发做出贡献。</li><li><a href="https://github.com/EdoPedrocchi/RicercaMente">GitHub - EdoPedrocchi/RicercaMente: 旨在通过多年来发表的科学研究追溯数据科学历史的开源项目</a>：旨在通过多年来发表的科学研究追溯数据科学历史的开源项目 - EdoPedrocchi/RicercaMente</li><li><a href="https://arxiv.org/abs/2403.13513">What if...?: 通过反事实引导减轻 Large Multimodal Models 中的幻觉效应</a>：本文提出了一种增强 Large Multimodal Models (LMMs) 处理幻觉效应可靠性的方法，幻觉效应是指模型生成错误或无关的响应。在无需额外...</li><li><a href="https://github.com/ivy-lvlm/counterfactual-inception">GitHub - IVY-LVLM/Counterfactual-Inception</a>：通过在 GitHub 上创建账户来为 IVY-LVLM/Counterfactual-Inception 的开发做出贡献。</li><li><a href="https://www.mdpi.com/1424-8220/24/5/1604">基于部分卷积的遥感地表温度数据深度插值</a>：地表温度 (LST) 是各种任务的重要资源。这些数据大多免费，结合了高空间和时间分辨率，以及在...范围内可靠的数据收集。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1228457751479451648)** (7 条消息): 

- **Cohere 值得赞赏的开源贡献**：*Cohere For AI* 以其对开源 AI 和开放科学的奉献精神脱颖而出，正如即将举行的 LLM Reading Group 会议所强调的那样。本次会议将深入探讨 **Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning**，计划于 4 月 16 日举行，感兴趣的人士可以通过此 [Eventbrite 链接](https://www.eventbrite.ca/e/llm-reading-group-march-5-19-april-2-16-30-may-14-28-tickets-851921368747?aff=oddtdtcreator) 预约。

- **日历便利性问题**：多位参与者请求获取 LLM Reading Group 会议的 Google Calendar 链接，他们被引导在预约后于 Eventbrite 上查找日历订阅选项。

- **解决 CUDA 兼容性**：一位成员寻求关于 CUDA 11.8 版本应使用哪个安装程序的指导，得到的建议是选择标记为 **11.x** 的版本，这表示兼容所有 CUDA 11 版本，包括 11.7、11.8 和 11.9。

- **Human Feedback Foundation 的关键作用**：**Human Feedback Foundation** 旨在将公众输入整合到 AI 模型中，成为医疗保健和治理等关键领域的必要组成部分。他们力求在为 AI 开发人员建立全面的人类反馈数据库方面发挥中立方的作用。

- **LLM Reading Group 春季会议如火如荼**：LLM Reading Group 的提醒强调了未来的会议主题，包括 **LLM 谈判**、**LLM 生成文本检测**以及**来自人类反馈的强化学习**。春季会议和论文的完整列表可以在季度时间表中找到，该系列活动每两周举行一次，直到 **5 月底**。

**提到的链接**：<a href="https://www.eventbrite.ca/e/llm-reading-group-march-5-19-april-2-16-30-may-14-28-tickets-851921368747?aff=oddtdtcreator">LLM Reading Group (3月5日, 19日; 4月2日, 16日, 30日; 5月14日; 28日)</a>：快来见见 LLM/NLP 研究领域一些开创性论文的作者，听他们讲述自己的工作。

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1228408018811752478)** (7 条消息): 

- **U-Net 模型在训练期间陷入停滞**：一位正在开发用于清理具有透印（bleed-through）效果的扫描图像的 **U-Net** 模型的成员，正面临模型在 11 个 epoch 后停止学习的困扰。该成员计划探索 **Deep Image Prior** 作为替代方案，希望能获得更好的结果。

- **Grounding DINO 加入 Transformers 库**：用于 zero-shot object detection 的新模型 **Grounding DINO** 现已包含在 Transformers 库中。官方文档和演示 Notebook 分别可以在 [Transformers Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino) 和 [Inference with Grounding DINO Notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/Inference_with_Grounding_DINO_for_zero_shot_object_detection.ipynb) 找到。

- **邀请加入 RicercaMente 项目**：一位成员宣布了 **RicercaMente**，这是一个旨在通过重要研究论文记录 Data Science 演进的项目。他们鼓励大家参与并提供了 GitHub 链接：[RicercaMente GitHub](https://github.com/EdoPedrocchi/RicercaMente)。

- **Stable Diffusion Gradio 聊天机器人发布**：一位成员分享了一个新的基于 Stable Diffusion 的 **Gradio 聊天机器人（用于图像生成）**，邀请他人尝试该工具，并在 GitHub 仓库 [Awesome Tiny SD GitHub](https://github.com/AstraBert/awesome-tiny-sd) 以及 [Hugging Face Spaces](https://huggingface.co/spaces/as-cle-bert/awesome-tiny-sd) 的在线实例上提供 Star 支持。

- **咨询针对特定任务微调 Vision Models**：一位成员寻求关于微调 Vision Model 以进行图像 captioning 和识别低分辨率图像中实体的建议，强调了对较小图像尺寸进行优化的需求。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/EdoPedrocchi/RicercaMente">GitHub - EdoPedrocchi/RicercaMente: 旨在通过多年来发表的科学研究追溯 Data Science 历史的开源项目</a>: 旨在通过多年来发表的科学研究追溯 Data Science 历史的开源项目 - EdoPedrocchi/RicercaMente</li><li><a href="https://github.com/AstraBert/awesome-tiny-sd">GitHub - AstraBert/awesome-tiny-sd: 基于 segmind/small-sd 的小型 Stable Diffusion 聊天机器人，根据文本提示生成图像。</a>: 基于 segmind/small-sd 的小型 Stable Diffusion 聊天机器人，根据文本提示生成图像。 - AstraBert/awesome-tiny-sd</li><li><a href="https://huggingface.co/spaces/as-cle-bert/awesome-tiny-sd">Awesome Tiny Sd - as-cle-bert 提供的 Hugging Face Space</a>: 暂无描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino">Grounding DINO</a>: 暂无描述</li><li><a href="https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/Inference_with_Grounding_DINO_for_zero_shot_object_detection.ipynb">Transformers-Tutorials/Grounding DINO/Inference_with_Grounding_DINO_for_zero_shot_object_detection.ipynb at master · NielsRogge/Transformers-Tutorials</a>: 该仓库包含我使用 HuggingFace 的 Transformers 库制作的演示。 - NielsRogge/Transformers-Tutorials
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1228384523499212901)** (14 条消息🔥): 

- **探索轻量级 Embedding 模型**：成员们讨论了适合项目的 Embedding 模型，建议包括使用 **all-MiniLM-L6-v2**，因为它在语义性能和轻量级架构之间取得了平衡。对于多语言需求，推荐使用 **paraphrase-multilingual-MiniLM-L12-v2**，特别是在论文写作等学术场景中。
  
- **组装巨型模型的碎片**：有人提出了如何处理像 **mixtral-8x22B** 这样被拆分为多个 GGUF 文件的超大模型的问题。讨论中没有提供关于加载或合并这些文件的具体操作方法的直接回答。

- **RicercaMente 项目亮相**：社区受邀为 **RicercaMente** 贡献力量，这是一个记录 Data Science 历史重大科学论文的 GitHub 项目。呼吁包括请求 Star、贡献和社交分享。[在 GitHub 上查看该项目](https://github.com/EdoPedrocchi/RicercaMente)。

- **寻求自定义 RAG LLM 教程**：一位成员寻求关于创建和微调用于问答的 **RAG LLM** 的教程或 YouTube 视频，强调要针对特定用例进行定制，但未提供具体资源。

- **寻求 NLP 入门指导**：NLP 初学者询问了路线图和学习资源，收到的建议从 Jurafsky 的著作《Speech and Language Processing》到入门材料如 [HuggingFace 学习资源](https://huggingface.co/learn) 和 [SpaCy 教程](https://spacy.io/usage)。3blue1brown 最近关于 Transformer 的视频也被提及为对初学者非常有帮助的资源。

**提到的链接**：<a href="https://github.com/EdoPedrocchi/RicercaMente">GitHub - EdoPedrocchi/RicercaMente: Open source project that aims to trace the history of data science through scientific research published over the years</a>：一个开源项目，旨在通过多年来发表的科学研究来追踪数据科学的历史 - EdoPedrocchi/RicercaMente

---

**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1228393964818665503)** (6 条消息):

- **深入探讨多模态搜索机制**：一位用户询问了 Web App 演示中多模态 Embedding 的细节，对演示如何理解图像和文本元素（特别是识别带有拼写错误的候选产品名称）表示困惑。他们引用了一个展示多模态 Vertex Embedding 的 [Google demo](https://ai-demos.dev/)，以更好地理解底层技术。

- **简短的回复仍具启发性**：针对关于多模态 Embedding 的查询，另一位用户简单地回复了 "CLIP retrieval"，暗示了针对前述在图像搜索中整合文本信息问题的潜在机制或探索方向。

- **介绍 ControlNet++**：一位用户分享了关于 ControlNet++ 论文的链接，这是对之前用于文本到图像扩散模型的 ControlNet 的改进，旨在通过像素级循环一致性（pixel-level cycle consistency）实现更好的控制。分享的 [arXiv 论文](https://arxiv.org/abs/2404.07987)介绍了它如何提高与条件图像控制的对齐度。

- **澄清语言模型的格局**：一位新成员对不同类型的语言模型表示困惑，特别是区分 LLM、Embedding 模型、openAIEmbadding 和 gpt_3.5_turbo 模型。

- **Stable Diffusion Cascade 模型查询**：一位用户在对 Stable Diffusion Cascade 模型使用长 Prompt 时遇到错误并寻求帮助；回复澄清该消息是关于 Token 限制的警告而非错误。该问题在 [GitHub issue](https://github.com/huggingface/diffusers/issues/7672) 中得到了进一步讨论。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ai-demos.dev/">AI Demos</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.07987">ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback</a>：为了增强文本到图像扩散模型的可控性，现有的努力（如 ControlNet）引入了基于图像的条件控制。在本文中，我们揭示了现有方法仍然……</li><li><a href="https://github.com/huggingface/diffusers/issues/7672">error in using stable cascade with long prompt · Issue #7672 · huggingface/diffusers</a>：你好，当我在 Stable Cascade 模型中使用长 Prompt 时，收到以下错误。Token 索引序列长度大于该模型指定的序列长度（165 > 77）。运行此脚本……
</li>
</ul>

</div>

---

**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1228357505889992835)** (147 条消息 🔥🔥):

- **诈骗警报：LAION 与 NFT 或代币无关**：成员们被警告防范一起诈骗，一个据称代表 LAION 的 Twitter 账号正在推广 NFT。该 [Twitter 诈骗](https://twitter.com/OraProtocol/status/1778107440992756126)也引起了讨论，用户寻求澄清 LAION 与加密货币之间的关联，LAION 成员的官方声明断然否认了任何联系。

- **模型训练的技术讨论**：对不同的训练方法进行了深入的技术讨论，包括 Loss 权重、信号噪声和理想的时间步数（timesteps）。特别关注了 *min-snr-gamma* 概念，并分享了关于在 diffusers 中的实现及其对模型性能的实际影响的关键见解。

- **新的 AI 音频模型考量**：对话转向用于声音生成的 AI 模型，涉及音乐 AI 的开源，以及由 Hugging Face 开发的 TTS 推理和训练库 [Parler TTS](https://github.com/huggingface/parler-tts)。

- **关于“伦理数据集”一词的辩论**：围绕“伦理数据集（ethical datasets）”一词出现了哲学辩论，批评其作为政治化语言的使用，并讨论了其在一篇关于 Adobe AI 训练方法的 [Bloomberg 文章](https://finance.yahoo.com/news/adobe-ethical-firefly-ai-trained-123004288.html)中的含义。

- **AI 模型输出与扩散技术**：用户探索了像素艺术扩散模型的工作原理，包括其分辨率能力，而其他人则讨论了 AI 模型输出以及分享潜在冒犯性材料的伦理问题。提到了在归一化输入后成功应用保持幅度的学习层（magnitude-preserving learned layers）。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>（此处链接已在正文中提及或为重复引用）</li>
</ul>

</div>

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.ora.io/app/imo/olm">ORA</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/declare-lab/tango">Tango - a Hugging Face Space by declare-lab</a>：未找到描述</li><li><a href="https://mirror.xyz/orablog.eth/X3DYXDHnjkpB-DOz88DZO5RdfZPxxRi5j53bxttNgsk>">World’s First Initial Model Offering (IMO) for OpenLM</a>：OpenLM 是一个高性能语言建模 (LM) 仓库，旨在促进对中型 LM 的研究。</li><li><a href="https://github.com/huggingface/diffusers/issues/5654>">Issues · huggingface/diffusers</a>：🤗 Diffusers：用于 PyTorch 和 FLAX 中图像和音频生成的尖端扩散模型。- Issues · huggingface/diffusers</li><li><a href="https://github.com/huggingface/parler-tts">GitHub - huggingface/parler-tts: Inference and training library for high-quality TTS models.</a>：高质量 TTS 模型的推理和训练库。- huggingface/parler-tts</li><li><a href="https://finance.yahoo.com/news/adobe-ethical-firefly-ai-trained-123004288.html">Adobe&#x2019;s &#x2018;Ethical&#x2019; Firefly AI Was Trained on Midjourney Images</a>：(Bloomberg) -- 当 Adobe Inc. 去年发布其 Firefly 图像生成软件时，该公司表示该人工智能模型主要是在其拥有数亿数据的数据库 Adobe Stock 上训练的...</li><li><a href="https://youtu.be/e1UgzSTicuY?si=0vS7ImdlVKC2QJJd">Why I&#39;m Leaving My Company Immediately (Stability AI) w/ Emad Mostaque | EP #93</a>：在本集中，Peter 和 Emad 讨论了 Emad 辞去 Stability AI CEO 职务、他迈向去中心化 AI 的下一步，以及为什么这件事如此紧迫...
</li>
</ul>

</div>
  

---


**LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/1228398882598162553)** (1 条消息): 

- **警惕虚假的 Twitter NFT 声明**：一份公告澄清，一个虚假的 Twitter 账号谎称 **Laion** 正在发布 NFT，并强调 **Laion 不销售任何东西**。Laion 仍然是 **开源社区** 的一部分，致力于提供 **免费的 AI 资源**。
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1228342334303768667)** (46 条消息🔥): 

- **扩散模型中的引导优化**：一项在 [arXiv 论文](https://arxiv.org/abs/2404.07724) 中概述的新研究显示，通过将引导限制在特定的噪声水平范围内，图像生成扩散模型可以达到最佳性能。研究指出，引导在链条的早期阶段是有害的，在末期基本是不必要的，仅在中期有益，应用有限的引导间隔可以显著改善图像质量。

- **自定义回调释放 Diffusers 的潜力**：[Hugging Face diffusers 库文档](https://huggingface.co/docs/diffusers/en/using-diffusers/callback) 详细说明了 `callback_on_step_end` 的用法，用于在去噪循环期间进行动态调整，例如更改 Prompt 嵌入或引导缩放（guidance scale）。该功能被强调为动态分类器自由引导（dynamic classifier-free guidance）的一个有用应用，其中 CFG 在特定的推理步骤后被禁用，以最小的计算成本增强性能。

- **数据累积防止模型崩溃**：一篇 [arXiv 论文](https://arxiv.org/pdf/2404.01413.pdf) 提供的证据反驳了当模型在合成生成的输出上训练时必然发生模型崩溃的观点，认为随着时间的推移累积数据（而不是替换数据）可以减轻崩溃风险。

- **Stable Diffusion 3 面临审查和质量挑战**：[Reddit](https://www.reddit.com/r/StableDiffusion/comments/1c3ro5y/stability_employee_preview_of_stable_diffusion_3/) 上的用户讨论了 Stable Diffusion 3 的严格审查及其由于严格的 Prompt 控制而导致的潜在质量下降，并希望通过大量的微调来增强其能力。

- **缩减版 CLIP 模型的训练策略**：最近发布的一篇 [arXiv 论文](https://arxiv.org/abs/2404.08197) 建议，高质量的数据和数据增强可以使 CLIP 模型在减少训练数据集的情况下表现得同样出色，证明了数据质量和战略性训练比单纯的数据集规模更重要。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/RekaAILabs/status/1779894626083864873">来自 Reka (@RekaAILabs) 的推文</a>：随着 Core 的发布，我们还发布了一份技术报告，详细介绍了 Reka 模型的训练、架构、数据和评估。https://publications.reka.ai/reka-core-tech-report.pdf</li><li><a href="https://arxiv.org/abs/2404.08197">Scaling (Down) CLIP: A Comprehensive Analysis of Data, Architecture, and Training Strategies</a>：本文研究了对比语言-图像预训练 (CLIP) 在缩减至有限计算预算时的性能。我们从数据、架构三个维度探讨了 CLIP...</li><li><a href="https://arxiv.org/abs/2404.07724">Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models</a>：Guidance 是从图像生成 Diffusion Models 中提取最佳性能的关键技术。传统上，在整个采样链中应用恒定的 Guidance 权重...</li><li><a href="https://huggingface.co/docs/diffusers/en/using-diffusers/callback">Pipeline callbacks</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2309.16620">Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit</a>：深度学习中超参数调优的成本随着模型规模的增加而上升，促使从业者寻找使用小型网络作为代理的新调优方法。其中一项提议使用了 $μ$P p...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c3ro5y/stability_employee_preview_of_stable_diffusion_3/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/huggingface/diffusers/issues/7657>">Issues · huggingface/diffusers</a>：🤗 Diffusers：用于 PyTorch 和 FLAX 图像和音频生成的 SOTA Diffusion Models。- Issues · huggingface/diffusers
</li>
</ul>

</div>
  

---


**LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1228701880272818267)** (1 条消息): 

- **Diffusion Model 的困扰：从噪声到纯黑**：一位正在研究 Diffusion Model 的成员遇到了一个问题，模型在训练进入平台期后，推理过程中仅产生噪声——最初是随机噪声，后来演变为纯黑图像。他们认为问题可能是由于误差累积或模式崩溃（mode collapse）导致的，在尝试了正则化和调整学习率但未获成功后，分享了噪声示例以及模型代码的 [GitHub repository](https://github.com/K3dA2/Diffusion-Model)。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1228272760237719595)** (142 条消息🔥🔥): 

- **法律合同的 RAG 问题**：一位用户在对法律合同进行检索增强生成 (RAG) 操作时遇到了文档切分困难。具体表现为，某个章节的内容被错误地归并到前一个章节，影响了检索准确性。
- **LangChain 中的并行执行**：有关于在 LangGraph 中并行运行节点的咨询。已确认在 LangChain 中使用 `RunnableParallel` 类可以实现任务的并行执行。
- **Azure OpenAI 与 LangChain**：一位 Azure OpenAI 新用户正在寻求关于在 Azure 上使用 LangChain 的优势见解，特别是询问是否存在成本效益，以及在 Azure 上与个人文档聊天时，LangChain 除了 RAG 之外还有哪些用途。
- **推荐系统咨询**：一位成员正在寻求资源，以帮助理解和实现使用存储在数据库中的用户行为数据的个性化推荐系统。
- **关于 LLM 保护的文章**：分享了一篇题为“Safeguarding AI: Strategies and Solutions for LLM Protection”的新文章，讨论了 LLM 安全的挑战、Prompt 攻击、解决方案和工具。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/pornox">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://flashcardfy.lol/">Flashcardfy - 带有个性化反馈的 AI 闪卡生成器</a>: 通过提供个性化反馈的 AI 生成闪卡，让学习更快速、更智能。</li><li><a href="https://devanshus-organization.gitbook.io/llm-security">守护 AI：LLM 保护的策略与解决方案 | LLM Security</a>: 在这份全面指南中探索 LLM 的安全挑战和解决方案。我们涵盖了潜在风险、控制机制以及用于更安全 LLM 应用的最新工具。</li><li><a href="https://docs.anaconda.com/free/miniconda/index.html">Miniconda &#8212; Anaconda 文档</a>: 未找到描述</li><li><a href="https://js.langchain.com/docs/modules/agents/how_to/custom_agent#bind-tools-to-llm>))">自定义 Agent | 🦜️🔗 Langchain</a>: 本 Notebook 介绍了如何创建你自己的自定义 Agent。</li><li><a href="https://js.langchain.com/docs/use_cases/tool_use/agents#create-agent>))">Agents | 🦜️🔗 Langchain</a>: 当我们知道针对任何用户输入所需的特定工具使用顺序时，Chain 非常有用。但对于某些用例，工具的使用次数取决于输入。</li><li><a href="https://js.langchain.com/docs/use_cases/tool_use/quickstart#create-a-tool>))">快速入门 | 🦜️🔗 Langchain</a>: 在本指南中，我们将介绍创建调用 Tool 的 Chain 和 Agent 的基本方法。Tool 可以是任何东西——API、函数、数据库等。Tool 允许我们扩展功能...</li><li><a href="https://github.com/langchain-ai/langchain/issues/6254>).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/docs/integrations/chat/anthropic#beta-tool-calling>)).">ChatAnthropic | 🦜️🔗 LangChain</a>: 本 Notebook 涵盖了如何开始使用 Anthropic 聊天模型。</li><li><a href="https://js.langchain.com/docs/modules/agents/how_to/custom_agent#bind-tools-to-llm>)).">自定义 Agent | 🦜️🔗 Langchain</a>: 本 Notebook 介绍了如何创建你自己的自定义 Agent。</li><li><a href="https://github.com/langchain-ai/langchain/issues/17352>):">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://bladehack.tech/">Blade Hack</a>: Blade Hack 是一个专注于 Prompt Hacking 的全球性活动，涉及以创意方式探索大型语言模型 (LLM) 并与之交互。Blade 将面向初学者和个人...</li><li><a href="https://python.langchain.com/docs/expression_language/why#invoke>)">LCEL 的优势 | 🦜️🔗 LangChain</a>: 我们建议阅读 LCEL [入门]</li><li><a href="https://github.com/langchain-ai/langchain/issues/10223>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/2435>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13635>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/docs/integrations/providers/vectara/vectara_chat#conversationalretrievalchain-with_map_reduce>).">使用 Vectara 对文档进行聊天 | 🦜️🔗 LangChain</a>: 设置}</li><li><a href="https://github.com/langchain-ai/langchain/issues/5328>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa#using-in-a-chain>)).">使用本地模型 | 🦜️🔗 LangChain</a>: 诸如此类项目的流行度</li><li><a href="https://python.langchain.com/docs/integrations/document_loaders/docusaurus#filtering-sitemap-urls>)).">Docusaurus | 🦜️🔗 LangChain</a>: Docusaurus 是一个静态网站生成器，它
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1228287178115649546)** (3 条消息):

- **不当内容警报**：发布了一条带有可疑链接的消息，声称提供成人内容，并使用了表情符号和 @everyone 标签；这似乎是垃圾信息或网络钓鱼。（出于安全原因，未包含直接链接）。
- **寻求 Langfuse Callbacks 指导**：一位成员询问了如何在 **langserve** 中使用 **langfuse callback handler 进行追踪**，旨在设计一个能够记录问题、session ID、user ID 等信息的 API。他们正在请求任何可用的资源或示例来协助其实现。

**提及的链接**：<a href="https://discord.gg/pornox">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。

  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1228274187991253002)** (4 messages): 

- **垃圾信息警报**：似乎发布了一条包含成人内容 Discord 服务器链接的垃圾消息。该消息承诺提供 **TEEN/ONLYFANS PORN**，并包含表情符号和 Discord 邀请链接。

- **Python LangChain 技术问题**：一位成员在尝试使用 `langchain serve` 运行 LangChain 文档中的模板时遇到了 `ModuleNotFoundError`。尽管检查了目录和文件设置，他们仍怀疑 `pyproject.toml` 文件未能正确读取本地包。

**提及的链接**：<a href="https://discord.gg/pornox">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。

  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1228272795054506027)** (17 messages🔥): 

- **AI 驱动的 Streamlit 新闻应用**：新应用 [Meeting Reporter](https://meeting-reporter.streamlit.app/) 结合了 Streamlit 和 LangGraph，以及多个 Agent 和 Human-in-the-loop，以创建更可靠的新闻报道。开源代码可在 [GitHub](https://github.com/tevslin/meeting-reporter) 上获取，项目的详细说明可以在 [博客文章](https://blog.tomevslin.com/2024/04/human-in-the-loop-artificial-intelligence.html) 中找到。

- **本地构建 RAG 指南**：分享了一个关于使用 LangChain、Ollama 和 PostgreSQL 构建本地 RAG 系统的简单教程，这可能对希望创建此类设置的人有所帮助。详细说明见 [Substack](https://luisciber.substack.com/p/how-to-build-a-rag-system-locally) 上的博客文章。

- **GitHub 上的 LLM 项目集合**：对于那些有兴趣探索大语言模型 (LLMs) 的人，分享了一个新的 GitHub 仓库 [LLM-Projects](https://github.com/luiscib3r/LLM-Projects)，其中包含多个旨在学习和实验的项目。

- **Perplexica，一个开源 AI 搜索引擎**：介绍了 Perplexica，它是 Perplexity AI 的本地运行克隆版，提供了一个替代的 AI 驱动搜索引擎，可以 100% 利用本地资源运行，包括使用 Ollama 替代 OpenAI 的可能性。更新和修复后的版本可在 [GitHub](https://github.com/ItzCrazyKns/Perplexica/) 上访问。

- **YouTube 教程探索 Vertex AI Agent Builder**：分享了一系列 YouTube 教程，展示了使用 Vertex AI Agent Builder 开发和实现 Agent 的过程，并提供了设置路由链 (route chains) 以及与 Slack 等平台集成的指南。视频链接如下：[Vertex AI Agent Builder 概述](https://youtu.be/9WytPcE64GQ)，[实现 Router Chain](https://youtu.be/mBmfcBjb3RA)，以及 [与 Slack 集成](https://youtu.be/bY3o5RzgWVw)。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/pornox">Discord - 与朋友和社区交流的新方式</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与您的朋友和社区保持紧密联系。</li><li><a href="https://youtu.be/mBmfcBjb3RA">使用 Vertex AI Agent builder 实现 router chain</a>：在此视频中，我展示了如何使用 Vertex AI Agent builder 构建 route chain。</li><li><a href="https://play.google.com/store/apps/details?id=com.projecthit.gptai&referrer=ph-aai">GPT AI - Chat GPT-4 &amp; Vision - Google Play 上的应用</a>：未找到描述</li><li><a href="https://github.com/ItzCrazyKns/Perplexica/">GitHub - ItzCrazyKns/Perplexica: Perplexica 是一款 AI 驱动的搜索引擎。它是 Perplexity AI 的开源替代方案</a>：Perplexica 是一款 AI 驱动的搜索引擎。它是 Perplexity AI 的开源替代方案 - ItzCrazyKns/Perplexica</li><li><a href="https://youtu.be/9WytPcE64GQ">Vertex AI Agent Builder</a>：在此视频中，我展示了如何使用 Vertex AI Agent builder</li><li><a href="https://youtu.be/bY3o5RzgWVw">Vertex AI Agent Builder 与 Slack 的集成</a>：此视频展示了如何将 Vertex AI Agent Builder 与 Slack 集成</li><li><a href="https://github.com/ItzCrazyKns/Perplexica">GitHub - ItzCrazyKns/Perplexica: Perplexica 是一款 AI 驱动的搜索引擎。它是 Perplexity AI 的开源替代方案</a>：Perplexica 是一款 AI 驱动的搜索引擎。它是 Perplexity AI 的开源替代方案 - ItzCrazyKns/Perplexica</li><li><a href="https://luisciber.substack.com/p/how-to-build-a-rag-system-locally">如何在本地构建 RAG 系统</a>：使用 Ollama 和带有 PgVector 的 PostgreSQL 在本地构建 RAG 系统</li><li><a href="https://github.com/VinciGit00/Scrapegraph-ai">GitHub - VinciGit00/Scrapegraph-ai: 基于 AI 的 Python 爬虫</a>：基于 AI 的 Python 爬虫。通过在 GitHub 上创建账号来为 VinciGit00/Scrapegraph-ai 的开发做出贡献。</li><li><a href="https://github.com/ossirytk/llama-cpp-chat-memory">GitHub - ossirytk/llama-cpp-chat-memory: 具有 Chroma 向量存储记忆的本地角色 AI 聊天机器人，以及一些用于为 Chroma 处理文档的脚本</a>：具有 Chroma 向量存储记忆的本地角色 AI 聊天机器人，以及一些用于为 Chroma 处理文档的脚本 - ossirytk/llama-cpp-chat-memory</li><li><a href="https://bladehack.tech/">Blade Hack</a>：Blade Hack 是一项专注于 prompt hacking 的全球性活动，涉及以创意方式探索和与 Large Language Models (LLMs) 互动。Blade 将面向初学者和个人...</li><li><a href="https://github.com/luiscib3r/LLM-Projects">GitHub - luiscib3r/LLM-Projects: 专注于探索和构建 Large Language Models (LLMs) 应用的项目集合。</a>：专注于探索和构建 Large Language Models (LLMs) 应用的项目集合。 - luiscib3r/LLM-Projects</li><li><a href="https://meeting-reporter.streamlit.app/">未找到标题</a>：未找到描述</li><li><a href="https://github.com/tevslin/meeting-reporter">GitHub - tevslin/meeting-reporter: 人机 AI 协作，根据会议记录或转录生成会议新闻报道</a>：人机 AI 协作，根据会议记录或转录生成会议新闻报道 - tevslin/meeting-reporter
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1228273306487230464)** (5 条消息): 

- **垃圾信息警报**：频道中发布了一条带有 Discord 链接的推广成人内容的垃圾信息。
- **分块知识**：分享了一个关于[使用 LangChain 进行文档分块](https://youtu.be/tMwdl9hFPns)的视频链接，该视频提供了通过正确拆分文档以在提问期间保留其内容，从而创建**更好的 RAG 应用**的见解。
- **Prompt Hacking 黑客松组队**：发布了一个为即将到来的 [Blade Hack prompt hacking 竞赛](https://bladehack.tech/)组建团队的公告。该活动挑战参与者使用 **prompt engineering 来操纵 AI 模型**以产生绕过限制的响应，奖金为 9000 美元及周边奖励。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/pornox">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。</li><li><a href="https://youtu.be/tMwdl9hFPns">但是，Chunking 是如何完成的？使用 LangChain 的分块基础</a>：为了创建更好的 RAG 应用，你需要了解如何对文档进行拆分或分块（chunk），以便在提问时保留内容。在本视频中，我...</li><li><a href="https://bladehack.tech/">Blade Hack</a>：Blade Hack 是一项专注于 prompt hacking 的全球性活动，涉及以创造性的方式探索 LLM 并与之交互。Blade 将面向初学者和个人...
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1228274134933438516)** (64 条消息🔥🔥): 

- **GitHub 上的开源 GPU 内核模块**：一个支持 P2P 的 NVIDIA Linux GPU 开源项目已在 GitHub 上发布，可以在 [tinygrad/open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules) 找到。
- **Fireworks AI 发布新款 Instruct MoE 模型**：Fireworks AI 推出了 Fireworks Mixtral 8x22b Instruct OH，这是使用 OpenHermes 数据集微调的 8x22b MoE 模型版本，预览版可在其 [Playground](https://fireworks.ai/models/fireworks/mixtral-8x22b-instruct-preview) 获取。文中提到了 deepspeed zero 3 的一个 bug，并分享了涉及从 DeepSpeed 主分支更新的解决方案。
- **关于 DeepSpeed 支持多 GPU 训练的讨论**：指出 *deepspeed zero 3* 有助于训练更大的模型，而不是加快训练过程。讨论了在 MoE 模型中使用 DeepSpeed zero 3 问题的成功解决方法，涉及从官方 DeepSpeed GitHub 仓库进行更新。
- **分享 AI 生成音乐的链接**：一位成员认可了 AI 生成音乐的进展，分享了一个使用 AI 创建的歌曲链接：[udio.com](https://www.udio.com/songs/eY7xtug1dV6hbfCDhyHJua)。
- **训练和调优模型的技术与工具**：讨论了关于模型合并、使用 LISA 和 DeepSpeed 等工具的技术对话，以及它们对训练和模型性能的影响。提供了一个用于提取 Mixtral 模型专家的 GitHub 链接（有一定硬件要求）：[Mixtral-Model-Expert-Extractor](https://github.com/MeNicefellow/Mixtral-Model-Expert-Extractor/blob/main/main.py)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.udio.com/songs/eY7xtug1dV6hbfCDhyHJua">Udio | 《沙丘》百老汇音乐剧，BobbyB 创作的曲目、原声带</a>：制作你的音乐</li><li><a href="https://huggingface.co/fireworks-ai/mixtral-8x22b-instruct-oh">fireworks-ai/mixtral-8x22b-instruct-oh · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/_lewtun/status/1778697910713979124?s=46">Lewis Tunstall (@_lewtun) 的推文</a>：我们在 Zephyr 141B 和全量训练中遇到了完全相同的问题 - 该问题与 DeepSpeed 中的这个 PR 有关：https://github.com/microsoft/DeepSpeed/pull/5008 如果你正在使用 HF trainer,...</li><li><a href="https://huggingface.co/blog/damjan-k/rslora">Rank-Stabilized LoRA: Unlocking the Potential of LoRA Fine-Tuning</a>：未找到描述</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: NVIDIA Linux open GPU with P2P support</a>：支持 P2P 的 NVIDIA Linux 开源 GPU 内核模块。通过在 GitHub 上创建账号来为 tinygrad/open-gpu-kernel-modules 的开发做出贡献。</li><li><a href="https://github.com/MeNicefellow/Mixtral-Model-Expert-Extractor/blob/main/main.py">Mixtral-Model-Expert-Extractor/main.py at main · MeNicefellow/Mixtral-Model-Expert-Extractor</a>：通过在 GitHub 上创建账号来为 MeNicefellow/Mixtral-Model-Expert-Extractor 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1477/files?short_path=3520786#diff-35207863e6e0da8dfa2d1311bf863b60c52a067c5e65253c24543edda5da00d0">shahdivax 编写的多节点分布式微调指南 · Pull Request #1477 · OpenAccess-AI-Collective/axolotl</a>：标题：多节点设置分布式微调指南。描述：此 PR 引入了使用 Axolotl 和 Accelerate 设置分布式微调环境的全面指南。该...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1192">winglian 提交的 Mixtral 修复 20240124 · Pull Request #1192 · OpenAccess-AI-Collective/axolotl</a>：描述：针对 Mixtral 与 Zero3 的各种修复，需要 DeepSpeed 0.13.1</li><li><a href="https://smicro.eu/nvidia-gpu-baseboard-8-h200-liquid-cool-935-24287-0040-000-2">NVIDIA GPU Baseboard 8 H200 Liquid Cool - 935-24287-0040-000</a>：图形引擎：H200
 总线：NVIDIA NVLink
 显存大小：1128 GB
 显存类型：HBM3e
 理论性能：536 TFLOP</li><li><a href="https://www.ebay.com/sch/i.html?_from=R40&_nkw=T480&_sacat=0&LH_BIN=1&rt=nc&_udhi=250&mkcid=1&mkrid=711-53200-19255-0&siteid=0&campid=5338557920&customid=&toolid=20012&mkevt=1">T480 待售 | eBay</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1228314262913548330)** (19 messages🔥): 

- **探索训练期间的权重解冻**：一名成员思考将**动态权重解冻**作为 GPU 受限用户的策略，考虑在每一步解冻权重的随机子集以优化训练过程。
- **自定义解冻功能**：讨论了动态权重解冻的*自定义实现*可行性，因为当前系统仅支持从开始就冻结某些权重。
- **分享动态计算分配模型**：分享了一个与 **Mixture-of-Depths** 相关的非官方 GitHub 实现，该模型涉及 Transformer 中计算的动态分配：[GitHub 上的 Mixture-of-depths](https://github.com/astramind-ai/Mixture-of-depths)。
- **RTX 4090 上的 Peer-to-Peer 内存访问成功**：一名成员成功在 **RTX 4090 GPU** 上通过 **tinygrad** 启用了 P2P 内存访问，并寻求关于绕过 Axolotl 的 P2P 禁用检查的建议，随后引发了关于检查 Accelerate 代码的讨论。
- **RTX 4090 P2P 兼容性和社区兴趣**：进一步澄清表示没有明确禁用 P2P 的代码，社区对成功利用 RTX 4090 GPU 进行 P2P 访问表示了兴趣。

**提到的链接**：<a href="https://github.com/astramind-ai/Mixture-of-depths">GitHub - astramind-ai/Mixture-of-depths: Unofficial implementation for the paper &quot;Mixture-of-Depths: Dynamically allocating compute in transformer-based language models&quot;</a>：论文 "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models" 的非官方实现 - astramind-ai/Mixture-of-depths

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1228552074715987988)** (39 messages🔥): 

- **卡在重复输出上**：一名成员遇到了模型产生重复输出的问题，类似于 [Izzy's Blog](https://www.izzy.co/blogs/robo-boys) 分享的数据集。尽管尝试了不同的数据集和微调方法，模型输出仍局限于 Discord 用户名或空白文本。

- **Mergekit 用于模型手术 (Model Surgery)**：成员们讨论了使用 **mergekit** 从大语言模型中删除层，并分享了来自 [Hugging Face 上的 WestLake-10.7B-v2](https://huggingface.co/froggeric/WestLake-10.7B-v2/blob/main/mergekit_config.yml) 的配置示例。

- **Axolotl 微调故障排除**：一位正在进行 **Mixtral-8x22** 微调的用户报告了在评估步骤中出现 'IndexError'。其他人建议使用 Axolotl 的 preprocess 命令进行调试，或参考 [Axolotl GitHub 仓库](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples) 中的示例逐步调整 YAML 配置。

- **Prompt 格式化挑战**：讨论了在微调过程中让 **Axolotl** 使用特定 Prompt 格式的困难。一位用户最终通过调整配置（包括使用 `chat_template` 设置）获得了成功。

- **数据集混合与训练 Loss 关注**：成员们就训练过程中混合补全（completion）和指令（instruction）数据交换了意见，有人建议分别对它们进行训练可能更有效。他们还讨论了纯补全模型的合适 Loss 数值以及 Batch Size 的潜在影响。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/LR-AI-Labs/vbd-llama2-7B-50b-chat">LR-AI-Labs/vbd-llama2-7B-50b-chat · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/froggeric/WestLake-10.7B-v2/blob/main/mergekit_config.yml">mergekit_config.yml · froggeric/WestLake-10.7B-v2 at main</a>：未找到描述</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1/discussions/4">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 · Prompt format</a>：未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples">axolotl/examples at main · OpenAccess-AI-Collective/axolotl</a>：欢迎提出 Axolotl 问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://www.irina-lab.ai/blog/continued-pretraining-blog">CERC-AAI Lab - Continued Pretraining Blog</a>：基础模型的持续学习：CL-FoMo 系列 9.6B 和 410M LLM </li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/config.qmd#L103-L112">axolotl/docs/config.qmd at main · OpenAccess-AI-Collective/axolotl</a>：欢迎提出 Axolotl 问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/)** (1 条消息): 

b.nodnarb: 感谢分享，<@915779530122207333> ！
  

---


**OpenAccess AI Collective (axolotl) ▷ #[docs](https://discord.com/channels/1104757954588196865/1167137552470392842/1228441757285355672)** (9 条消息🔥): 

- **Mistral V2 Instruct 表现出色**：一位成员报告在仅经过一个 Epoch 后，Mistral v2 instruct 就取得了**良好的效果**，优于 LoRA 和 QLoRA 等其他方法。
- **在多样化任务上有效**：他们强调了在专门针对元数据提取任务的 **NER、分类和摘要训练**中取得的成功。
- **超越以往方法**：与另一个名为 **qwen** 的模型实验相比，性能显著提升。
- **元数据生成的自动化**：明确了由大语言模型（**LLM**）负责**生成**（而非仅仅是提取）文档中的元数据。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1228317412193996822)** (5 条消息):

- **Docker Meets DeepSpeed for Distributed Training**：使用 DeepSpeed 进行分布式训练的 Docker 方案：使用 DeepSpeed 进行多节点微调需要一个包含所有依赖项的定制化 Docker 镜像，以及一个用于配置设置的详细 `ds_config.json` 文件。步骤包括构建 Docker 镜像、设置跨节点的 SSH 访问，以及运行带有适当环境变量的容器以进行多节点通信。
- **SSH Keys Unlock Multi-Node Communication**：SSH 密钥解锁多节点通信：为了成功进行多节点训练，必须配置 SSH 密钥以允许节点之间的免密访问。DeepSpeed 的设置依赖于通过 SSH 进行的无缝节点通信，以有效地执行分布式计算。
- **Running Training Containers Across Nodes**：在节点间运行训练容器：用户必须在每个节点上启动 Docker 容器，并仔细设置定义主节点地址、端口、world size 和每个节点 rank 的环境变量。正确的设置将实现在 Docker 环境中使用 DeepSpeed 在不同机器上部署微调过程。
- **Training Initialization with DeepSpeed**：使用 DeepSpeed 初始化训练：训练脚本应集成 DeepSpeed 初始化并接受特定参数（`--deepspeed` 和 `ds_config.json`），以充分发挥分布式训练功能的潜力。确保正确使用这些参数对于微调操作在多节点环境中运行至关重要。

**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=d905c33d-1397-4ef3-82ad-a16aadf1eb1f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1228427843663302746)** (25 messages🔥): 

- **Understanding DeepSpeed Scheduler Configuration**：理解 DeepSpeed 调度器配置：澄清了在将 **DeepSpeed** 与 🤗 Accelerate 结合使用时，如果你使用 Accelerate 的方法正确准备了自定义学习率调度器，它将被保留，因为 DeepSpeed 不会强制替换它，除非在配置文件中另有说明。
  
- **Optimizing Micro-Batch Sizes**：优化 Micro-Batch 大小：关于 `micro_batch_size` 与 `gradient_accumulation_steps` 理想比例的讨论强调了有效利用 GPU 显存所需的平衡，建议最大化 `micro_batch_size` 并相应调整 `gradient_accumulation_steps` 以达到所需的有效 Batch Size。

- **Multi-Node Training with Axolotl**：使用 Axolotl 进行多节点训练：在 **Axolotl** 中配置多节点训练涉及使用指定网络详细信息（如 `main_process_ip` 和 `main_process_port`）的配置文件，并使用 `accelerate` CLI 在机器之间启动训练。

- **Opening Ports for Distributed Training**：为分布式训练开放端口：为了确保 `main_process_port` 在 TCP 中开放且可被分布式训练设置中的其他机器访问，说明包括识别主节点机器、使用 `iptables` 或 Windows Defender Firewall 设置等系统命令来开放端口，并验证网络可访问性。

- **Ease of Hugging Face Model Hub Repository Creation**：简化 Hugging Face Model Hub 仓库创建：解释了 `push_to_hub` 方法，如果指定的 Model ID 尚未对应现有仓库，该方法会自动在 Hugging Face Model Hub 上创建一个新仓库，从而为用户简化流程。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e8e1f994-4d79-4e03-b88a-ebc3dc1d0933)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=98d8d920-0524-4543-b179-9bc5f7810e7f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=43a2ad80-c4db-40e0-8bbf-e85b2a2dc2a1)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://github.com/huggingface/transformers/tree/main/src/transformers/utils/hub.py#L657L690))">transformers/src/transformers/utils/hub.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=23860f45-87ae-4250-ade9-727c7de241f7)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c53633a8-bb1b-475d-a241-0931651d36c8)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>

**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1228269684944666685)** (79 messages🔥🔥): 

- **恶意软件警告与命令行困惑**：有用户对 Avast 发出的潜在恶意软件警告以及命令行 `ngrok/ngrok/ngrok` 引起的困惑表示担忧。有人提到在文档中澄清这一点以安抚用户可能会很有帮助。
- **Open Interpreter 针对构建错误的 PR 提案**：提交了一个新的 Pull Request 来解决构建过程中的错误，具体是提升了 tiktoken 的版本。PR 链接在[这里](https://github.com/OpenInterpreter/open-interpreter/pull/1204)。
- **使用 Assistants API 实现数据持久化**：关于在 Open Interpreter 中使用 Assistants API 进行数据持久化的讨论，涉及成功创建 Python Assistant 模块以改进会话管理。有用户请求关于在节点操作中实现此功能的具体建议。
- **Ubuntu 上 Open Interpreter 的 OS Mode 故障排除**：用户分享了在 Ubuntu 上为 Open Interpreter 设置 **OS Mode** 相关的问题和解决方案。包括启用屏幕监控、辅助功能以及降级到 Python 3.10 以保证兼容性的建议。
- **Airchat 作为 OI 用户语音交流平台**：几位成员分享了他们的 [Airchat](https://www.joinairchat.com/) 账号（例如 `mike.bird`），并讨论了获取邀请以加入实时语音对话的可能性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/MaG00_GpHXw?si=I3MC3jHZ6lfLBhXL">Polaris 机器人手机</a>：http://www.plasticpals.com - Flower Robotics 受 KDDI 委托生产了这款可与手机配合使用的机器人，旨在通过其...改善您的生活。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1204">minamorl 提升 tiktoken 版本 · Pull Request #1204 · OpenInterpreter/open-interpreter</a>：描述所做的更改：由于构建过程因某些原因损坏，提升了 tiktoken 的版本。此 PR 修复了损坏的过程。引用任何相关问题（例如 "Fixes #000"）：...
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1228381814129688638)** (72 messages🔥🔥): 

- **初创公司发货预期延迟**：新产品的预订正在进行中，但一位成员提醒要保持耐心，指出初创公司通常需要比*初始估计更长的时间*，特别是从 GitHub 更新来看，该产品仍处于*预生产阶段*。
- **热点连接问题**：用户在尝试通过 iPhone 热点连接 O1 时遇到问题，设备有时无法识别网络或在连接过程中崩溃。即使使用示例代码，问题依然存在，引发了关于特定库是否可能导致崩溃的推测。
- **Windows 上的环境变量难题**：Windows 用户在将 `OPENAI_API_KEY` 设置为环境变量时遇到困难，建议尝试不同的终端（如使用 Command Prompt 而非 PowerShell），并确保在设置后启动新会话。这里有一个分享的链接可以帮助[测试 Key](https://chat.openai.com/share/c625f9c8-102e-4669-9c87-5d874b57a241)，但持续存在的问题引发了关于替代方案的进一步讨论。
- **Windows 上 WebSocket 的挫败感**：尽管有旨在解决该问题的更新，但一些用户在 Windows 上仍然遇到 WebSocket 无法打开的问题，而 Mac 上则没有此问题。建议检查端口是否已被占用，并分享错误截图以便进行故障排除。
- **O1 的开发与 DIY 尝试**：社区成员正在修改和改进他们的 O1 设计，从增加更好的电池到方便使用的手带。一位成员为外壳提供免费的 3D 打印服务，并提到 ChatGPT Plus 用户可以使用根据 Open Interpreter 文档训练的自定义 GPT 提供协助。

**提到的链接**：<a href="https://github.com/rbrisita/01/tree/linux">GitHub - rbrisita/01 at linux</a>：开源语言模型计算机。通过在 GitHub 上创建账户为 rbrisita/01 的开发做出贡献。

  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

aime_bln: https://api.aime.info
  

---



**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1228370900940750989)** (1 messages): 

- **LlamaIndex 更新影响 PandasQueryEngine**：LlamaIndex 的新版本 0.10.29 将把 **PandasQueryEngine** 迁移到 `llama-index-experimental` 包中。用户必须相应地调整其导入（import）方式，系统将为使用旧模块的用户提供错误消息引导。

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1228371970798915716)** (8 messages🔥): 

- **新的 AI 应用创建工具包发布**：与 @tsystemscom、@MarcusSchiesser 合作开发，并受 @llama_index 的 create-llama 工具包启发，**create-tsi** 工具包允许用户通过命令行界面生成符合 GDPR 规范的 AI 应用程序。该公告包含了 [推文](https://twitter.com/llama_index/status/1778812761893650551) 链接及相关图片。

- **Chain of Abstraction 技术揭晓**：LLM（如 @OpenAI、@MistralAI、@AnthropicAI）已通过一种名为 Chain of Abstraction 的技术得到改进，以辅助多步查询规划；该创新的详细信息可以在提供的 [推文](https://twitter.com/llama_index/status/1778845258119524640) 中找到。

- **使用 AWS Bedrock 构建全栈 RAG 指南**：现已提供一份参考指南，解释如何使用 AWS Bedrock、@llama_index 和 @streamlit 界面构建全栈 RAG 应用程序；更多信息见其 [推文](https://twitter.com/llama_index/status/1779185836690653435)。

- **LLM 应用中的数据管理挑战**：正如其 [公告](https://twitter.com/llama_index/status/1779235219469742112) 所强调的，LLM 应用程序中实时数据的有效数据管理是 @llama_index 开源版和 LlamaCloud 的核心功能。

- **用于加速生物材料发现的知识图谱**：@ProfBuehlerMIT 的一篇论文展示了如何从 1000 多篇关于生物材料的科学论文中构建广泛的知识图谱，以加速生物材料的发现；来源可以在其 [推文](https://twitter.com/llama_index/status/1779542320133947622) 中探索。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1228294385087680564)** (113 messages🔥🔥): 

- **探究向量数据库偏好**：成员们讨论了用于强相似度搜索的向量数据库偏好，比较了包括 Qdrant、pgvector 和 Cassandra 在内的不同选项。分享了文档链接以评估混合搜索（hybrid search）和元数据过滤（metadata filtering）等功能支持（[LlamaIndex 向量存储指南](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/)）。

- **混合搜索机制探讨**：对话扩展到关于多模态嵌入（multimodal embeddings）实现的辩论，以及如何检索和排序多模态数据——特别是涉及图像和文本数据。引用了关于融合算法的文章，并提到了 Weaviate 等工具的混合搜索能力。

- **LlamaIndex 技术故障排除**：用户遇到了各种技术挑战，例如在 Vectara Managed Index 中处理 `MockLLM` 错误，以及在定义 `MarkdownElementNodeParser` 时遇到 `ValidationError`。社区支持提供了指导和建议，强调了更新到最新版本的重要性。

- **查阅文档与贡献**：一位用户发现了文档中关于 `GithubRepositoryReader` 的错误，社区成员鼓励在 GitHub 上提交 Pull Request 来纠正文档。强调了积极参与 LlamaIndex 项目社区贡献的重要性。

- **参与 LlamaIndex 的业务方面**：关于与用户客户签署商业伙伴协议 (BAA) 的讨论，促使分享了 LlamaIndex 团队的联系信息，并保证该请求将被转达以获得快速响应。聊天机器人强调了社区支持的重要性以及 LlamaIndex 团队的可触达性。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ai-demos.dev/">AI Demos</a>: 未找到描述</li><li><a href="https://www.llamaindex.ai/contact">联系我们 — LlamaIndex，LLM 应用的数据框架</a>: 如果您对 LlamaIndex 有任何疑问，请联系我们，我们将尽快安排通话。</li><li><a href="https://llamahub.ai/l/llama-packs/llama-index-packs-fuzzy-citation?from=">未找到标题</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/data_connectors/GithubRepositoryReaderDemo/">Github Repo Reader - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/">Vector Stores - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings/?h=settings">Settings - LlamaIndex</a>: 未找到描述</li><li><a href="https://weaviate.io/blog/hybrid-search-fusion-algorithms">解锁混合搜索的力量 - 深入探讨 Weaviate 的融合算法 | Weaviate - Vector Database</a>: 混合搜索的工作原理，以及 Weaviate 融合算法的底层机制。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/qdrant_hybrid/">Qdrant Hybrid Search - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/07cb4c07409b0e0584e2dee70db0d3169f877b8e/llama-index-integrations/indices/llama-index-indices-managed-vectara/llama_index/indices/managed/vectara/retriever.py#L233">llama_index/llama-index-integrations/indices/llama-index-indices-managed-vectara/llama_index/indices/managed/vectara/retriever.py at 07cb4c07409b0e0584e2dee70db0d3169f877b8e · run-llama/llama_index</a>: LlamaIndex 是适用于您的 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_memory/">Query Pipeline Chat Engine - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_parse/blob/main/">GitHub - run-llama/llama_parse: 为优化 RAG 解析文件</a>: 为优化 RAG 解析文件。通过在 GitHub 上创建账户为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb">llama_parse/examples/demo_advanced.ipynb at main · run-llama/llama_parse</a>: 为优化 RAG 解析文件。通过在 GitHub 上创建账户为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>: 未找到描述</li><li><a href="https://coursera.org/projects/langchain-chat-with-your-data-project">LangChain Chat with Your Data</a>: 在 2 小时内完成此指导项目。LangChain: Chat With Your Data 深入探讨了两个主要主题：(1) Retrieval Augmented Generation (RAG)，一种...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_memory/?h=query+pipeline">Query Pipeline Chat Engine - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/?h=query+pipeline">围绕 Query Pipeline 构建 Agent - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1228354294034595940)** (5 条消息): 

- **通过 LlamaIndex 增强文档检索**：分享了一个新的[增强文档检索教程](https://medium.com/ai-advances/enhancing-document-retrieval-with-memory-a-tutorial-for-llamaindex-with-colbert-based-agent-1c3c47461122)，展示了如何使用 **LlamaIndex** 将 memory 整合到基于 Colbert 的 Agent 中。
  
- **利用小型知识图谱提升 RAG 系统**：分享了一篇关于使用小型知识图谱提高 RAG 系统准确性和解释能力的讨论和文章。[WhyHow.AI](https://medium.com/enterprise-rag/the-role-of-small-vs-big-knowledge-graphs-995d9449c208) 正在开发工具，以促进构建这些图谱，从而增强 **retrieval-augmented generation (RAG)** 的性能。

- **集成 LlamaIndex 以实现高级推理**：一位成员强调了一篇关于将 **LlamaIndex** 与 chain of abstraction 策略集成以解锁更高效推理的出版物，并附上了相关[文章](https://ai.gopubby.com/unlocking-efficient-reasoning-integrating-llamaindex-with-chain-of-abstraction-1b1844ba66e6)的链接。

- **对高效推理见解的赞赏**：反馈表达了对最近分享的关于 AI 系统推理进展文章的赞赏。
  

---

**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1228306974064640132)** (31 messages🔥): 

- **RTX 4090 驱动补丁登上 Hacker News**：有成员提到 **4090 驱动补丁** 登上了 Hacker News 首页，暗示其对技术社区产生了重大影响。
- **堆叠 4090 的成本效益**：关于堆叠 **RTX 4090s** 的讨论表明，运行大语言模型 (LLMs) 的成本将显著降低。一位成员提供了与 *TinyBox* 的详细成本对比，显示其价格比红队 (Team Red) Tinybox 降低了 **36.04%**，比绿队 (Team Green) Tinybox 降低了 **61.624%**。
- **Tinygrad 寻求更好的文档**：提出了关于改进 **tinygrad 文档** 的查询，成员们确认其正在开发中。征集了关于如何处理 `fetch_mnist(tensors=False)` 实例的建议，以便在 **tinygrad examples** 中进行方便且连贯的更新。
- **开发者体验是 tinygrad 的首要任务**：**George Hotz** 强调了在改进 tinygrad 文档和开发者体验方面的进展，并指出该工具需要更好的错误理解能力。
- **增加行数限制与不稳定的 MNIST 测试**：一致同意增加代码行数限制以容纳新的后端，**George Hotz** 提议将其提升至 *7500* 行。此外，还提出了关于 *flaky mnist test*（不稳定的 mnist 测试）和调度器 (scheduler) 中非确定性的问题，表明项目正专注于健壮性和可靠性。

**提及的链接**：<a href="https://github.com/tinygrad/tinygrad/actions/runs/8694852621/job/23844626455">hotfix: bump line count to 7500 for NV backend · tinygrad/tinygrad@e14a9bc</a>：你喜欢 pytorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - hotfix: bump line count to 7500 for NV backend · tinygrad/tinygrad@e14a9bc

---

**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1228260143167438858)** (72 messages🔥🔥): 

- **Kernel 生成中的融合过程**：围绕在融合 (fusion) 过程中计算图被拆分为多个 Kernel 的条件进行了深入讨论。明确了每个 **ScheduleItem** 就是一个 Kernel，而计算图拆分的原因（例如导致独立 Kernel 的广播需求）可以在源码的 `create_schedule_with_vars` 函数中找到。

- **使用 Tinygrad 运行模型**：澄清了 **Tinygrad 运行模型**（如 Stable Diffusion 和 Llama）的能力。解释说虽然 Tinygrad 可以执行这些模型，但它不一定用于训练它们——这展示了 **Tinygrad** 在对接不同框架训练的模型时的灵活性。

- **Tinygrad 在 Colab 上的安装与使用案例**：成员们分享了在 Google Colab 中安装和使用 Tinygrad 的经验和建议，包括使用 `pip install git+https://github.com/tinygrad/tinygrad` 以及解决导入错误。讨论了在各种环境和不同方法论下使用 **Tinygrad** 的好处。

- **Tensor 填充与 Transformer 错误**：一位用户提出了 Tensor 填充 (padding) 和 Transformer 模型的问题，引发了对错误消息的探讨，以及在 Tinygrad 模型架构中调整填充的可能性。通过反复试验，确认使用 `None` 进行填充是一个可行的解决方案。

- **文档与贡献**：分享了 Tinygrad 文档和个人学习笔记的链接，以帮助新手理解 Tinygrad 的内部工作原理。回答了关于在没有 CUDA 支持的情况下如何为 Tinygrad 做出贡献的问题，强调了在贡献之前熟悉文档和代码库的价值。

<ul>
<li>
<a href="https://colab.research.google.com/drive/1vy97ByB5mLielmE8TzlnnaSZtMUykm-A?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/scheduleitem.md">tinygrad-notes/scheduleitem.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账号来为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/tree/master/docs">tinygrad/docs at master · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/codegen.md">tinygrad-notes/codegen.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账号来为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad">GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>：你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</li><li><a href="https://github.com/pranjaldatta/tinygrad">GitHub - pranjaldatta/tinygrad: A simple, basic autodiff engine written to learn the inner workings of autograd!</a>：一个为了学习 autograd 内部工作原理而编写的简单、基础的 autodiff 引擎！- pranjaldatta/tinygrad
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1228861251342307408)** (5 messages): 

- **整理 Artifacts**：一个新的博客功能引入了 [Hugging Face collections](https://huggingface.co/collections/natolambert/2024-interconnects-artifacts-6619a19e944c1e47024e9988)，用于整理博客文章底部链接的所有关于“开放模型和数据集”的 artifacts。
- **便捷访问**：创建这些 collections 是为了方便查找和重新访问链接的 artifacts。
- **团队扩张计划**：作者构想在雇佣助手后建立一个 "interconnects" Hugging Face 组织。
- **Collections API 可用**：提到 Hugging Face collections 拥有 API，现在就可以使用，[Hugging Face 文档](https://huggingface.co/docs/huggingface_hub/en/package_reference/collections)中对此进行了演示。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/huggingface_hub/en/package_reference/collections">管理 collections</a>：未找到描述</li><li><a href="https://huggingface.co/collections/natolambert/2024-interconnects-artifacts-6619a19e944c1e47024e9988">2024 Interconnects Artifacts - natolambert 的 Collection</a>：未找到描述
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1228381066197336145)** (11 messages🔥): 

- **对 Hard Fork 的沮丧**：对本周的 **hard fork** 表达了沮丧，认为这由于一些不反映 AI 现状的废话而引发了负面情绪。
- **对 Claude 增量更新的怀疑**：从 **Claude 2 到 Claude 3** 的过渡受到质疑；“INCREMENTAL?”（增量更新？）一词暗示了对其重要性的怀疑。
- **指出开放数据倡议**：提醒人们开放数据倡议的存在，暗示它们在当前的讨论中可能被忽视了，正如这句话所言：*"no one is detailing open data? AHEM"*（没人详细说明开放数据吗？咳咳）。
- **记者对技术的看法**：一位成员评论道，**知名科技记者**似乎对技术持有一种敌对态度，这令人费解。
- **欢迎对批评的互动**：在 **Twitter** 上受到批评后，有一条关于互动的积极记录，因为一位科技记者在成员向其发送推文后开始关注该成员。
- **AI 发布潮**：今天宣布了一系列发布，包括来自 EleutherAI 的 **Pile-T5**、**WizardLM 2** 以及来自 Yi Tay 的 **Reka Core**，并提供了相应来源的[链接](https://blog.eleuther.ai/pile-t5/)。**Wizard LM 2** 被强调为特别有趣。
- **开源版权改造**：一条评论赞扬了将 **Dolma** 设为 **ODC-BY** 的决定，并对大量的开源公告感叹道：*"God can open source stop with the announcements today 😭"*（天哪，开源界今天能不能停止发布公告 😭）。
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1228565155957178431)** (13 messages🔥):

- **关于 ACL commitment 修订的困惑**：一位成员询问是否有必要在评审后上传修订后的 PDF，Nato 表示不确定，并暗示可能在某个时间点需要这样做。
- **合成数据讨论升温**：一位成员对 Nato 关于 [合成数据和 CAI](https://www.interconnects.ai/p/llm-synthetic-data) 的文章表示热赞，引发了关于 2023 年底以来合成数据元策略（meta）的对话。Nato 提到目前正在取得进展，并暗示将发布一个基于修订版的数据集。
- **ORPO 方法——到底是怎么回事？**：ORPO 方法受到了质疑，Nato 认为拥有“大模型”是一个好策略，**暗示了模型规模（model size）的重要性**优于特定技术。
- **AI 分析中的 Critic 模型与 Reward 模型**：成员们讨论了像 **Prometheus** 这样的 “critic” 模型与 Nato 的 RewardBench 项目中展示的 “reward” 模型之间的区别。对话深入探讨了它们的功能，例如提供二元偏好或标量分数，而 critics 则提供修订建议或批判。
- **是否上传 PDF**：分享了 @andre_t_martins 的一条推文链接，表明 **ACL commitment 可能毕竟不需要上传修订后的 PDF**。[点击此处查看推文](https://twitter.com/andre_t_martins/status/1779263540785725737)。
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1229465023680741446)** (4 messages): 

- **Newsletter 中的图表受到好评**：一位成员表示非常喜欢今天 Newsletter 中包含的图表，赞赏其清晰度和有效性。
- **承诺包含图表**：频道所有者确认，在未来使用这些信息丰富的图表将成为一个固定特征。
- **图表增强功能即将推出**：讨论了未来的改进，计划精炼图表并将其整合到一个 Python 库中。
- **探索开源模型的基础**：所有者表示，这些精美的图表将在撰写关于新开源模型的文章时，作为一种有价值的工具。
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1228425858675507262)** (1 messages): 

- **CodecLM 研究论文讨论**：一位成员分享了 Google 在 4 月 8 日发表的题为“CodecLM: Aligning Language Models with Tailored Synthetic Data”的[研究论文链接](https://arxiv.org/pdf/2404.05875.pdf)。他们提到这似乎是一种*向更强模型学习*的方法，但未提供进一步分析。
  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/1228412327486029916)** (1 messages): 

- **分享 LLaMA 论文**：一位成员在 Hugging Face Collections 上发布了 2023 年 2 月 27 日发表的论文“LLaMA: Open and Efficient Foundation Language Models”的链接。该论文的标识符为 [2302.13971](https://huggingface.co/papers/2302.13971)。

**提到的链接**：<a href="https://huggingface.co/collections/natolambert/aligning-open-language-models-66197653411171cc9ec8e425">[lecture artifacts] aligning open language models - a natolambert Collection</a>：未找到描述

  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1229469203468259510)** (2 messages): 

- **耐心可能是一种美德**：一位成员正在实验一个 bot，并提到可能需要等待更长时间，而不是手动干预。等待的具体背景或原因未指明。
  

---



**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1228312970837364767)** (25 messages🔥): 

- **对 Haiku 的迟疑**：一些成员对 **Haiku 的速度**表示担忧，提到了总响应时间而非吞吐量的问题。有人建议 **Bedrock** 可能会提供更好的响应时间，尽管一位成员指出它也很慢。

- **Claude 拒绝角色扮演**：由于 **Claude** 不配合角色扮演场景（包括扮演女仆战士或发送虚构的垃圾邮件），成员们感到很沮丧。尽管尝试了指令提示（instruction prompting）以及 one-shot 或 few-shot 技术，问题依然存在。

- **越狱 Claude 3？**：成员们正在讨论为角色扮演游戏等前卫内容**越狱 Claude 3** 的必要性。一位成员引用了 *@elder_plinius* 的一条推文，报告了 Claude 3 的通用越狱方法，这可以绕过像 **Gemini 1.5 Pro** 等模型中更严格的内容过滤。该推文可以在[此处](https://x.com/elder_plinius/status/1773455789056745782?s=46)找到。

**提及的链接**：<a href="https://x.com/elder_plinius/status/1773455789056745782?s=46">来自 Pliny the Prompter 🐉 (@elder_plinius) 的推文</a>：越狱警报！刚刚发现了 Claude 3 的通用越狱方法。恶意软件、硬毒品配方、炸弹制作，应有尽有。在高温度设置下的 Haiku 似乎是最佳组合，但在我有限的...

---

**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1228403053653397544)** (4 条消息): 

- **代码能力升级**：一名成员指出，新版本“**在代码方面确实更强**”且“**速度也更快**”，表明其在编程任务中的性能有所提升。
- **考虑重新激活以进行测试**：鉴于感知到的改进，一名成员正考虑重新激活其 **ChatGPT Plus** 订阅，以测试这些新功能。
- **Claude 在长代码任务中仍保持优势**：Claude 仍被认为在“长上下文窗口代码任务”中具有价值，这表明 ChatGPT 在上下文窗口大小方面可能仍存在局限。

---

**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1228272255696371734)** (8 条消息🔥): 

- **MoE 之谜仍在继续**：一名成员表示，感觉 Transformers 的 MoE 训练/微调/RLHF 过程中缺少了一些关键要素。但未详细说明缺失的具体元素。
- **记起 Fish 模型合并**：在 [Reddit 讨论](https://huggingface.co/Envoid)中，Envoid 开发的 "fish" 模型被推荐为 Mixtral 中有效的 RP/ERP 合并模型。尽管评论者因硬件限制尚未测试，但作者对 Mixtral 合并技术的贡献被认为可能很有帮助。
- **关于 Mixtral “独门秘籍”的推测**：参与者推测 Mixtral 是否拥有其论文中未公开的有效训练“秘密”，暗示要实现更好的 MoE 性能可能存在隐藏要素。
- **周末尝试 Mixtral 微调**：一名成员打算在周末实验 Mixtral，重点是使用英德指令数据进行全量微调（full finetuning），并提到 Zephyr 的配方代码似乎运行良好。
- **寻找更优的 Mixtral 秘方**：有人询问是否能以超越官方 Mixtral Instruct 的方式微调 Mixtral 模型，特别是关于路由优化（routing optimization）相关的“独门秘籍”是否已被社区成员发现。

---

**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1229399893198639215)** (4 条消息): 

- **自定义 Llama-Tokenizer 训练困境**：一名成员正在探索如何训练自定义 **Llama-tokenizer** 以优化部署在小型硬件上的体积，目标是最小化嵌入层（embedding layer）的大小。他们指出在创建所需的 *tokenizer.model* 文件时遇到困难，并寻求帮助以解决该问题，从而使其能成功用于 **llama.cpp**。
- **Token 生成策略**：另一名参与者建议尝试使用 GitHub 上的一个脚本，该脚本可能有助于自定义 **Llama-tokenizer** 的转换过程。他们指向了 [Hugging Face Transformers 仓库](https://github.com/huggingface/transformers/blob/fe2d20d275d3591e2619a1adb0fa6ae272605208/src/transformers/convert_slow_tokenizer.py#L534)中的 `convert_slow_tokenizer.py` 脚本。
- **使用 llama.cpp 脚本轻松转换**：有人建议使用 [llama.cpp GitHub 仓库](https://github.com/ggerganov/llama.cpp/blob/master/convert.py)中的 `convert.py` 脚本并带上 `--vocab-only` 选项，暗示这可能是解决 Tokenizer 模型文件创建问题的方案。
- **为开放多模态模型倡议征集数据收集者**：发出了寻求协助的呼吁，旨在寻找具有版权许可或免费的欧盟文本及多模态数据，用于训练一个**大型开放多模态模型**。请拥有相关数据的感兴趣方通过私信联系。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/ggerganov/llama.cpp/blob/master/convert.py">llama.cpp/convert.py at master · ggerganov/llama.cpp</a>：C/C++ 环境下的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/huggingface/transformers/blob/fe2d20d275d3591e2619a1adb0fa6ae272605208/src/transformers/convert_slow_tokenizer.py#L534">transformers/src/transformers/convert_slow_tokenizer.py at fe2d20d275d3591e2619a1adb0fa6ae272605208 · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的先进机器学习库。 - huggingface/transformers
</li>
</ul>

</div>

---

**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1228317593643913266)** (2 条消息): 

- **Occiglot 模型显示出模板敏感性**：对 **occiglot/occiglot-7b-de-en-instruct** 的评估表明，与[错误模板](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval#occiglotocciglot-7b-de-en-instruct-1)相比，使用 [Hugging Face 上的正确模板](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval#occiglotocciglot-7b-de-en-instruct)在德语 RAG 任务上的表现有所提升。该问题可能源于 Data Processing Operation 中的某个步骤，而非系统性错误。

- **分享 StableLM 和 MiniCPM 预训练见解**：Quicksort 讨论了 StableLM 在其[技术论文](https://arxiv.org/abs/2402.17834)中提到的报告，其中包括使用 **ReStruct 数据集**进行预训练，其灵感来自[此 GitHub 仓库](https://github.com/ExpressAI/reStructured-Pretraining)的方法。此外，MiniCPM 的研究（详见其[第 5 章](https://arxiv.org/abs/2404.06395)的消融实验）表明，在预训练的 cooldown 阶段加入 OpenOrca 和 EvolInstruct 等数据会带来收益。

**提到的链接**：<a href="https://arxiv.org/abs/2402.17834">Stable LM 2 1.6B Technical Report</a>：我们推出了 StableLM 2 1.6B，这是我们新一代语言模型系列中的首款模型。在本技术报告中，我们详细介绍了导致 Base 和 Instruct 模型产生的数据和训练过程...

  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1228288816482353204)** (9 条消息🔥): 

- **Burnai 项目引发关注**：一名成员赞扬了 [Burnai 项目](https://burn.dev/)极具前景的能力，强调了其通过 Rust 代码在各种平台上优化推理的潜力。他们表示好奇，尽管 Rust 是 Mozilla 的产物，但为什么 Mozilla 还没有对 Burnai 进行调研。

- **Llamafile 进入 Mcaffee 白名单**：社区庆祝 *llamafile 0.7 二进制文件*被 Mcaffee 列入白名单，认为这是项目迈出的积极一步。

- **欢迎新合作者**：一位新参与者的自我介绍表达了对社区内深度合作和交流的热情。

- **关于 Vulkan 和 tinyblas 的疑问**：有成员询问了即将发布的 v0.8 版本是否可能支持 Vulkan，以及 *tinyblas* 的上游化（upstreaming）情况，特别是对其在 ROCm 上的应用感兴趣。

- **寻求打包自定义模型的指导**：社区对如何将自定义模型打包进 llamafile 的指南表现出明显兴趣，这促使成员们寻找并分享相关资源，例如关于[容器发布的 GitHub Pull Request](https://github.com/Mozilla-Ocho/llamafile/pull/59#issuecomment-1840814790)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://burn.dev/">Burn</a>：未找到描述</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/pull/59#issuecomment-1840814790">dzlab 通过 GitHub Actions 将容器发布到 Docker Hub · Pull Request #59 · Mozilla-Ocho/llamafile</a>：在发布时使用 GitHub Actions 构建并将容器发布到 Docker Hub #29。为此，需要设置仓库机密（secrets）：DOCKER_HUB_USERNAME 和 DOCKER_HUB_ACCESS_TOKEN</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6414">jart 提升 CPU Prompt 评估速度 · Pull Request #6414 · ggerganov/llama.cpp</a>：此更改将 llamafile 的 CPU 矩阵乘法内核上游化，从而提高了图像和 Prompt 的评估速度。首先，Q4_0 和 Q8_0 权重在 CPU 上的运行速度应提高约 40%。最大的...
</li>
</ul>

</div>
  

---



**Alignment Lab AI ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/1228638326945218632)** (2 条消息): 

- **垃圾信息解决方案建议**：一名成员建议邀请 **wick bot** 到聊天频道以自动删除垃圾信息。他们认为这可以防止发生类似于被标记消息所指示的事件。
  

---


**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1228342844377403512)** (4 条消息): 

- **请求私信帮助**：一名成员表示需要代码方面的帮助，并通过标记特定用户请求私信（DM）。
- **对 Discord 邀请链接的担忧**：另一名成员对在服务器各处发布 Discord 邀请链接表示担忧，并建议应彻底禁止此类行为以避免相关问题。 
- **人类验证呼吁**：那位需要代码帮助的成员再次重申了请求，并提到这是为了证明自己不是机器人。
  

---

**Alignment Lab AI ▷ #[join-in](https://discord.com/channels/1087862276448595968/1143791237669855302/1229226519327543377)** (1 messages): 

没有合适的消可以总结。

**提到的链接**: <a href="https://discord.gg/VWffHaCpED">Discord - 与朋友和社区交流的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、闲逛，与你的朋友和社区保持紧密联系。

  

---


**Alignment Lab AI ▷ #[oo2](https://discord.com/channels/1087862276448595968/1176548760814375022/)** (1 messages): 

aslawliet: 这个项目还活跃吗？
  

---



**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1228632341036011622)** (6 messages): 

- **Code-Jamba-v0.1 挑战多语言编程**：分享了一个名为 [Code-Jamba-v0.1](https://huggingface.co/ajibawa-2023/Code-Jamba-v0.1) 的微调模型，该模型以在包括 **Python, Java, 和 Rust** 在内的多种编程语言中生成代码的高效性而闻名。它在 **2 x H100 GPU** 上训练了 **162 小时**，使用了 [Code-290k-ShareGPT](https://huggingface.co/datasets/ajibawa-2023/Code-290k-ShareGPT) 和 [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) 数据集。

- **关于模型基准测试和数据集共享的疑问**：一位成员对 **Code-Jamba-v0.1** 缺乏模型评估数据表示好奇，并询问 AI21 Labs 是否会分享其数据集，强调了其在 256K 上下文下微调 LLM 的潜力。

- **Jamba API 期待**：有人询问了 **Jamba** API 的可用性，并提到在与 CEO 沟通后，Fireworks AI 可能会集成 Jamba。

**提到的链接**: <a href="https://huggingface.co/ajibawa-2023/Code-Jamba-v0.1">ajibawa-2023/Code-Jamba-v0.1 · Hugging Face</a>：未找到描述

  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1229235258344345630)** (5 messages): 

- **密码猜测游戏介绍**：成员们分享了 [PasswordGPT](https://passwordgpt.io/) 的链接，这是一个目标是说服 AI 泄露密码的游戏。
- **关于数据标注的辩论**：一位成员表示倾向于在建模前对数据进行标注，以了解数据集内容，并想知道其他人是否遵循类似做法，还是更多地依赖没有广泛标注的 LLM。
- **使用前 LLM 模型提取历史文档记录**：一位成员提到他所在团队负责策划数据，并使用早于 LLM 的 Transformer 模型从历史文档中提取记录。
- **用户与 AI Demo 的互动**：相比于封闭提示词的猜测游戏，人们更倾向于开放提示词的 AI Demo，并强调看到提示词可以帮助用户更有效地参与并击败 AI。

**提到的链接**: <a href="https://passwordgpt.io/">PasswordGPT</a>：未找到描述

  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1229410199375188048)** (1 messages): 

- **将 AI 应用扩展到生产环境**：发布了一个名为“**解锁将 Gen AI 应用扩展到生产环境的秘密**”的见面会公告，强调了将 Gen AI 应用扩展到全规模生产的挑战。该活动的嘉宾包括 [Portkey](https://portkey.ai/)、[Noetica AI](https://www.noetica.ai/) 和 [LastMile AI](https://lastmileai.dev/) 的联合创始人，可通过[此链接](https://lu.ma/llms-in-prod-nyc)注册。感兴趣的人员需要获得主办方的批准才能参加。

**提到的链接**: <a href="https://lu.ma/llms-in-prod-nyc">LLMs in Prod w/ Portkey, Flybridge VC, Noetica, LastMile · Luma</a>：解锁将 Gen AI 应用扩展到生产环境的秘密。虽然原型化一个 Gen AI 应用很容易，但将其投入全规模生产却很难。我们汇集了从业者 &....