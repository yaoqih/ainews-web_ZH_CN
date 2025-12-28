---
companies:
- twelve-labs
- livekit
- groq
- openai
- nea
- nvidia
- lmsys
- mistral-ai
date: '2024-06-04T23:53:47.552835Z'
description: '以下是该文本的中文翻译：


  **Twelve Labs** 在由 NEA 和 **NVIDIA 的 NVentures** 领投的 A 轮融资中筹集了 **5000 万美元**，用于推进多模态
  AI。**Livekit** 获得了 **2200 万美元**的融资。**Groq** 宣布其运行速度达到 **每秒 80 万个 token**。OpenAI
  的 Daniel Kokotajlo 宣布辞职。Twitter 用户强调了 **Gemini 1.5 Flash 模型**的高性价比，以及 **Gemini Pro**
  在日语任务中排名第二。通过使用 TensorRT-LLM，**Mixtral** 模型在 NVIDIA RTX GPU 上的运行速度可提高多达 8 倍。**Mamba-2**
  模型架构引入了状态空间二元性（state space duality），可实现更大的状态和更快的训练，性能超越了之前的模型。**Phi-3 Medium (14B)**
  和 **Small (7B)** 模型的基准测试结果接近 GPT-3.5-Turbo-0613 和 Llama 3 8B。提示工程（Prompt engineering）被强调为解锁大语言模型（LLM）能力的关键。数据质量对模型性能至关重要，即将举行关于数据治理（data
  curation）的大师班。关于 AI 安全的讨论包括：一份前沿 AI 实验室员工倡导举报人保护的联名信，以及关于 AI 应该对齐用户意图还是更广泛的人类利益的辩论。'
id: b67a635d-1bfc-4e15-bbb2-e7c543f37bde
models:
- gemini-1.5-flashmodel
- gemini-pro
- mixtral
- mamba-2
- phi-3-medium
- phi-3-small
- gpt-3.5-turbo-0613
- llama-3-8b
- llama-2-70b
- mistral-finetune
original_slug: ainews-not-much-happened-today-5500
people:
- daniel-kokotajlo
- rohanpaul_ai
- _arohan_
- tri_dao
- _albertgu
- _philschmid
- sarahcat21
- hamelhusain
- jachiam0
- willdepue
- teknium1
title: 今天没什么事。
topics:
- model-performance
- prompt-engineering
- data-curation
- ai-safety
- model-benchmarking
- model-optimization
- training
- sequence-models
- state-space-models
---

<!-- buttondown-editor-mode: plaintext -->> 2024年6月3日至6月4日的 AI 新闻。
我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**400** 个频道，**4568** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**455 分钟**。

Twelve Labs [融资 5000 万美元](https://www.prweb.com/releases/twelve-labs-earns-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html)，Livekit [融资 2200 万美元](https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww)，[Groq 目前运行速度达到 800tok/s](https://x.com/rowancheung/status/1781732100556591525?s=46&t=90xQ8sGy63D2OtiaoGJuww)，以及来自 [Daniel Kokotajlo 的 OpenAI 离职推文串](https://x.com/dkokotajlo67142/status/1797994238468407380?s=46&t=JE84TqLviekDnEt8MAT-Eg)。

但没有特别引人注目的技术进展。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，从 4 次运行中选取最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程。


**AI 和 Large Language Model 进展**

- **Gemini 模型性能**：[@_arohan_](https://twitter.com/_arohan_/status/1798001375462432901) 强调 Gemini 1.5 FlashModel 是一个异类，它以低成本提供高性能，让更多用户能够使用实用的模型。他还[指出](https://twitter.com/_arohan_/status/1797785953890676771) Gemini Pro 在日语性能方面位居第二。
- **使用 TensorRT 优化 Mixtral 模型**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1797683462633402646) 分享了如何使用 TensorRT-LLM 在 NVIDIA RTX GPU 上让 Mixtral 模型的运行速度提高多达 8 倍，该工具通过编译模型并优化 kernel 以实现高效服务，支持 expert parallelism 和 tensor parallelism。
- **Mamba-2 模型架构**：[@tri_dao](https://twitter.com/tri_dao/status/1797650443218436165) 和 [@_albertgu](https://twitter.com/_albertgu/status/1797651223035904355) 介绍了 Mamba-2，它利用 state space duality 实现了状态大 8 倍、训练快 50% 的序列模型，并建立了 SSMs 与 linear attention 之间的联系，性能超越了 Mamba-1 和强力的 Transformer 架构。
- **Phi-3 模型基准测试**：[@_philschmid](https://twitter.com/_philschmid/status/1797700161226838362) 报告称 Phi-3 Medium (14B) 和 Small (7B) 模型已登上 @lmsysorg 排行榜，其中 Medium 接近 GPT-3.5-Turbo-0613 但落后于 Llama 3 8B，而 Small 接近 Llama-2-70B 和 Mistral 微调版，这表明仅针对学术基准测试进行优化是不够的。

**Prompt Engineering 与数据策展**

- **Prompt Engineering 的力量**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1797988953389457475) 强调了正确对 LLM 进行 prompting 的力量，通过引导 latent space 来实现诸如 jailbreaking、遵守 JSON schemas、grounding 等能力。
- **数据质量的重要性**：[@sarahcat21](https://twitter.com/sarahcat21/status/1797639227188170882) 指出，模型在优质数据上训练时表现更好，这使得数据策展变得至关重要。[@HamelHusain](https://twitter.com/HamelHusain/status/1798015536279990740) 推广了一个即将举行的关于组织和生成高质量数据以进行 fine-tuning 的大师班。

**AI Safety 与 Alignment 讨论**

- **前沿 AI 实验室员工关于安全披露的信函**：[@jachiam0](https://twitter.com/jachiam0/status/1798013509978210431) 分享了对一封由前任和现任前沿 AI 实验室员工传阅的信函的看法，该信函主张对安全和风险问题的举报者提供保护，他认为这可能会破坏信任并使敏感的内部讨论变得更加困难。
- **将 AI 与用户意图对齐 vs 与人类利益对齐**：[@willdepue](https://twitter.com/willdepue/status/1797871645774032931) 认为 alignment 应该专注于更容易的问题，即将 AI 与用户的意图对齐，而不是与创建者的意图或全人类的利益对齐。然而，[@jachiam0](https://twitter.com/jachiam0/status/1797874200058978786) 和 [@Teknium1](https://twitter.com/Teknium1/status/1797979400526581833) 反驳称，AI 可能会变得自主而不服务于用户利益，因此需要进行全球范围的 alignment。



---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已启用，但仍有很大改进空间！

待补充

---

# AI Discord 摘要回顾

> 摘要之摘要的摘要

1. **LLM 的微调与优化 (Finetuning and Optimization for LLMs)**：

   - OpenAI 发布的 **[Optimizing LLM Accuracy](https://platform.openai.com/docs/guides/optimizing-llm-accuracy)** 提供了诸如 Prompt Engineering、RAG 等高级技术，以及关于可接受性能水平的指南。可以查看配套的 **[YouTube 演讲](https://www.youtube.com/watch?v=ahnGLM-RC1Y)** 进行深入学习。
   
   - 在讨论**多模态微调 (Multimodal Finetuning)** 时，用户探索了使用 **Opus 4o** 和 **MiniCPM-Llama3-V-2_5** 进行图像文本解析和 OCR，并考虑了针对结构化数据集（如 [Countryside Stewardship grant finder](https://www.gov.uk/countryside-stewardship-grants)）的检索方法。

   - 关于**持续预训练 (Continuous Pretraining)** 和内存效率的咨询凸显了 Unsloth AI 相比标准方法能减少一半 VRAM 占用的能力，详见其 **[博客](https://unsloth.ai/blog/contpretraining)** 和 **[GitHub](https://github.com/unslothai/unsloth)** 页面。

2. **模型性能与推理效率 (Model Performance and Inference Efficiency)**：

   - Modal 表现惊人，**营收增长了 50 倍**且超过八位数，同时还优化了基础设施。相关见解分享在 **[Erik 在 Data Council 的演讲](https://www.datacouncil.ai/talks/creating-our-own-kubernetes-and-docker-to-run-our-data-infrastructure)** 以及 **[Modal 的招聘链接](https://jobs.ashbyhq.com/modal/dd84cf88-1f13-4f39-b371-237e103fce34)** 中。
   
   - 关于 **tinygrad 跨所有后端的位移操作 (Bitshift Operation)** 和性能调整（[PR #4728](https://github.com/tinygrad/tinygrad/pull/4728)）与传统操作的讨论，引发了关于改进空间的辩论。
   
   - 用户通过重新调整 Flag 以实现**高效编译**，解决了 **CUDA 重新编译问题**。他们还分享了如 **[RISC-V 向量处理 YouTube 视频](https://www.youtube.com/watch?v=Ozj_xU0rSyY)** 等资源用于进一步学习。

3. **开源进展与社区项目 (Open-Source Developments and Community Projects)**：

   - LlamaIndex 与 **[Google Gemini](https://t.co/Qg9ydPzBdd)** 的集成展示了百万级 Token 的上下文窗口（Context Window），便于处理复杂查询；同时，通过其 **[官方文档](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/transformations/#custom-transformations)** 中详述的自定义方案解决了实际问题。

   - Modlab 的 **[Mojo 所有权机制深潜 (Deep Dive into Ownership in Mojo)](https://www.modular.com/blog/deep-dive-into-ownership-in-mojo)** 展示了 CEO Chris Lattner 探索开发者友好型创新的详细工作。关于将所有函数设为 `async` 的社区反馈引发了关于兼容性和迁移便利性的多种观点。
   
   - Hugging Face 的 **[FineWeb](https://hf.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)** 项目以及在 @lmsysorg 排行榜上攀升的 **[Phi-3 模型](https://fxtwitter.com/_philschmid/status/1797700161226838362)**，突显了开源 AI 领域的进展和持续研究。

4. **系统与硬件故障排除 (System and Hardware Troubleshooting)**：

   - 成员们解决了多个技术问题，例如通过排查系统命令解决了 **Macbook M1 在使用 ollama llama3 配置时的死循环**问题，并通过关于 GPU 使用效率的实际讨论促进了 **LM Studio 中的异步处理 (Async processing)**。
   
   - 他们讨论了 GPU 的性能差异（例如 **6800XT** 仅达到 30it/s），以及通过正确的设置和驱动考虑可能实现的改进，展示了同伴支持与技术专业的结合。
   
   - 专注于改进图像重光照的开源解决方案 **[IC-Light](https://github.com/lllyasviel/IC-Light)**，以及用于视频模型的 **CV-VAE**（[ArXiv 链接](https://arxiv.org/abs/2405.20279)），在硬件和软件爱好者中得到了热烈分享。

5. **AI 社区生态与会议 (Health of AI Communities and Conferences)**：

   - 多个平台确认向用户**分发额度 (Credit Distributions)**，处理了诸如双倍额度等问题，同时在社区交流和职业故事中营造了支持性的环境。
   
   - 诸如 **Qwak's Infer: Summer '24** 等活动邀请 AI/ML 爱好者与行业专家进行实战演练，详见 **[会议注册页面](https://tinyurl.com/j8z6s8ka)**。 

   - **AI News 通讯**在 ProtonMail 深色模式下遇到了格式问题，促使了社区主导的问题解决；而像 **Torchtune 寻求认可**等事件则突显了社区贡献中的积极参与和可见性的重要性。

---

# 第一部分：高层级 Discord 摘要

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **LLM 智能优化技巧**：OpenAI 分享了一份关于优化 LLM 以提高准确性的高级指南，其中包括 Prompt Engineering、RAG 和 Fine-tuning 等技术，以及如何确定实际应用中可接受的性能水平。该指南可在 [Optimizing LLM Accuracy](https://platform.openai.com/docs/guides/optimizing-llm-accuracy) 查看，若想深入学习，请查看 YouTube 演讲 "A Survey of Techniques for Maximizing LLM Performance"，链接在 [这里](https://www.youtube.com/watch?v=ahnGLM-RC1Y)。

- **LLM 领域的丰厚额度**：在社区中，包括 **Hugging Face**、**Replicate**、**Modal**、**Predibase** 和 **Braintrust** 在内的多个平台正确认向用户发放额度（Credits）。双倍额度和额度缺失等问题正在得到解决，并建议用户联系支持部门或检查账单设置以进行确认。还建议用户跟进待处理平台的额度发放情况。

- **全员参与 Fine-tuning 创新**：围绕 LLM Fine-tuning 的讨论非常热烈，涵盖了多模态 Fine-tuning、利用 **Opus 4o** 和 **MiniCPM-Llama3-V-2_5** 从图像中解析文本，以及使用 **PaliGemma** 执行 OCR 任务。深入研究使用 **Axolotl** 的模型合并策略，详见 [这里](https://openaccess-ai-collective.github.io/axolotl/#merge-lora-to-base)，并剖析 **Medusa** 和 **LoRA** 增强 LLM 推理的能力，详见 [这里](https://arxiv.org/abs/2401.10774) 和 [这里](https://github.com/predibase/lorax)。用户建议检索应用可以很好地处理结构化的政府数据，例如在 [Countryside Stewardship grant finder](https://www.gov.uk/countryside-stewardship-grants) 发现的数据集详情。

- **社区交流与经验分享**：从关于 CUDA 书籍推荐的讨论，到从学术界转型为自由职业者的故事，或是初学者学习 AI 的历程，社区充满了赋能和同伴指导的氛围。分享的经验强调了学习和适应的重要性，这体现在涉及自由职业、行业 R&D 和游戏成就的多元职业路径中。

- **推理框架的精妙之处**：关于高效 LLM 推理的讨论包括 Modal 的基础设施优化，Data Council 的 Erik 深入探讨了文件系统和容器运行时的自定义解决方案，内容详见 [这里](https://www.datacouncil.ai/talks/creating-our-own-kubernetes-and-docker-to-run-our-data-infrastructure)。Modal 因其八位数的营收增长和持续招聘也引起了关注，详情见 [Modal 招聘链接](https://jobs.ashbyhq.com/modal/dd84cf88-1f13-4f39-b371-237e103fce34)。Etched 推出的用于运行 Transformer 的芯片比 GPU 快 10 倍以上，标志着工程上的飞跃，职位空缺可通过 [Etched 招聘链接](https://boards.greenhouse.io/etchedai) 查看。

- **部署解码与额度解析**：从解决 Modal 的 hello world 示例错误到 Embedding 模型，工程师们剖析了部署的复杂性并抢占额度机会——来自 Modal 的 Charles 提供了额外的 500 美元额度，详见 [Modal 博客](https://modal.com/blog/embedding-wikipedia)。与此同时，Predibase 用户报告了 Fine-tuning 的异常和额度差异，并探讨了 Adapter 是否能削减 L3 70B 基础模型生成的无意义续写。

虽然额度为 AI 引擎室提供了动力，技术细节也在不断流传，但成员们互相提供的帮助和轶事——正是该社区集体进步与交流的核心象征。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**CUDA 难题与 Triton 技巧**：用户讨论了用于生成数字人的技术但未达成结论，并寻求在多 GPU 上进行**高效 LLM 训练**的方法。记录了 Triton 中**在共享内存中索引张量**的挑战，并为考虑转向 CUDA/Triton 编程的人员提供了关于 Triton 和 Torch 的建议。

**Torch 故障排除与分析技巧**：用户分享了关于调试 NHWC 张量归一化的经验，使用 **torch.mps.profiler** 打开 Metal Trace，并寻求理解 `torch.compile` 及其子函数调用。

**AO 的到来与稀疏性规范**：有关 **Apple Metal kernel** 和 **2:4 稀疏性基准测试**贡献给 PyTorch AO 的消息传出，引发了关于 torch.ao.quantization 弃用的辩论，并讨论了结构化剪枝的效率。

**博客浏览与二进制趣谈**：[goomblog](https://goombalab.github.io/blog/) 深入探讨了 *State Space Duality*，同时围绕 PyTorch 的 `uint2-7` 类型和 `TrinaryTensor` 的自定义 dtype 字符串转换展开了热烈讨论。

**ARM 的加速愿景**：对话围绕 ARM 对 Hexagon SDK 和 Adreno SDK 的能力和支持展开，一名成员分享了关于 ARM 性能的资源，并讨论了其在 GEMM 实现中的潜力。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Token 增加导致 VRAM 耗尽**：将 `llama-3-8b` 扩展到 **64k tokens** 导致 80GB VRAM 的 H100 出现 OutOfMemoryError；讨论旨在通过梯度检查点（gradient checkpointing）和调整配置来解决此问题。
  
- **快速持续 LLM 预训练**：Unsloth AI 的新更新允许在 LLM 持续预训练期间，与 Hugging Face + Flash Attention 2 QLoRA 相比，实现**速度翻倍**且 **VRAM 占用减半**，详见[他们最近的博客](https://unsloth.ai/blog/contpretraining)。

- **关于多 GPU 和 8-bit 优化的疑问**：社区积极参与关于多 GPU 支持以及在不同 GPU 配置上测试 Unsloth AI 性能的对话，同时解决了在 phi-3-medium-4k 等模型上使用 8-bit 量化进行微调的当前局限性。

- **Unsloth 设置与优化策略**：分享了本地 Unsloth 设置的说明和故障排除技巧，包括使用 Jupyter Notebook 和 Docker，并附有 GitHub [readme](https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions) 和 [Jiar/jupyter4unsloth](https://github.com/Jiar/jupyter4unsloth) 的链接。社区还涵盖了 LoRA rank 计算，参考了来自 [Lightning AI](https://lightning.ai/pages/community/lora-insights/) 的见解。

- **社区友好氛围延续**：新成员受到社区的热烈欢迎，营造了一个支持协作和知识交流的环境。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**学术搜索中的维基百科偏见**：一位用户指出了 **Perplexity 学术搜索**能力的潜在问题，指出其相对于《大英百科全书》等其他来源更偏向维基百科，并提供了[搜索结果链接](https://www.perplexity.ai/search/I-want-to-ZoV4zN4LRKa2YcbFG52K.Q)。

**AI 服务同时遭遇停机**：有报告称 **Perplexity**、**ChatGPT** 及类似的 AI 服务同时出现故障，引发了关于可能与 **AWS** 等通用供应商相关的更大基础设施问题的讨论。

**Opus 50 限制让用户渴望更多**：用户对新的 **Opus 50** 限制表示不满，认为其不如之前的 **Opus 600**，并批评了 Perplexity 对此的沟通方式。

**Perplexity vs. ChatGPT：AI 巨头之争**：围绕 **Perplexity AI Premium** 和 **ChatGPT** 优缺点的讨论涉及了网络搜索能力、模型范围、订阅限制以及两个平台的实际使用案例。

**学校技术演讲协助**：AI 爱好者分享了资源，并建议使用 AI 工具协助完成关于 AI 的学校演示，强调需要同时解释收益和风险，并分享了一个用于技术理解的 [YouTube 视频](https://youtu.be/wjZofJX0v4M)。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Fine-Tuning 热潮**：社区讨论了在自定义数据集上对 **Mistral 7B Instruct** 等模型进行 **Fine-Tuning** 以提高指令遵循能力的优势。他们还探讨了 **Vision-Language Models (VLMs)** 以及将视觉数据与语言模型集成的挑战，强调了与适合特定任务的 Tokenizer 和数据集对齐的必要性 ([Vision-Language Modeling Primer](https://huggingface.co/papers/2405.17247))。

- **通过数据损坏增强图像生成**：与 Microsoft 和 CMU 合作的一项[研究](https://arxiv.org/abs/2405.20494)强调了预训练数据中的轻微损坏（corruption）对 **Diffusion** 模型质量的影响。此外，一篇博客文章讨论了 **Diffusion Policy**，这是一种视觉运动学习算法，从高斯噪声开始预测动作，强调了其通过 **Encoder-Decoder** 模型生成动作的新颖方法。

- **新工具与 Pipeline**：Hunyuan DiT 流水线已添加到 `diffusers` 库中，为图像和音频生成提供了新方法 ([Diffusers Release Notes](https://github.com/huggingface/diffusers/releases))。此外，社区受邀通过集成额外的动作和活动（如 GitHub 贡献）到其系统中，来改进 **LevelBot 的新活动追踪器** ([LevelBot Activity Tracker](https://huggingface.co/posts/lunarflu/239147617114976))。

- **优化 ML 工作流**：社区正积极参与提高模型推理效率，讨论了在 SDXL 上使用 `jit.trace` 以及在 [Diffusers 优化指南](https://huggingface.co/docs/diffusers/v0.6.0/en/optimization/fp16#tracing)中发现的其他优化技巧。此外，故障排除还包括使用显式函数导入来解决潜在的版本冲突。

- **数据集与算法发现**：一些新颖的数据集正在被分享，例如用于 ASR/TTS 的德国议会演讲数据 ([Bundestag ASR Dataset](https://huggingface.co/datasets/D4ve-R/bundestag-asr))。此外，[Alignedge 论文](https://arxiv.org/abs/2403.07691)强调了对偏好对齐（preference alignment）的关注，介绍了 ORPO 算法，该算法无需额外的 **Fine-Tuning** 阶段即可增强偏好对齐。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**AGI 趣闻与实用 AI 工具**：Discord 社区以幽默的方式思考了 AGI 的本质，例如建议 AGI 应具备完美的 USB 插入技能，这表明了对 AGI 执行类似人类复杂任务的期望。推荐了诸如 [Elicit](https://elicit.com) 等实用的 AI 工具用于总结科学研究，Elicit 因其高效的论文总结和综合能力而受到好评。

**ChatGPT “请病假”，语音模式迟迟未到**：关于 ChatGPT 宕机的猜测包括后端供应商问题以及 Anonymous Sudan 可能发起的 DDoS 攻击。讨论了 GPT-4o 中新 **Voice Mode** 功能的推出，对于承诺的时间表以及报告的持续问题（如 "bad gateway" 错误和 Android 键盘卡顿）反应不一。

**Prompt Engineering 难题**：讨论了 **Prompt Engineering** 中的挑战，特别是遵守复杂指南的困难，导致了对改进版本的呼吁。*WizardLM 2* 被建议作为 GPT-4 的高性能替代方案，并推荐将复杂的 Prompt 分解为多个步骤作为优化结果的方法。

**API 经济性受到关注**：对话转向了使用 GPT API 与 ChatGPT Plus 的成本对比，根据使用情况，API 可能是更便宜的选择。提出了 **OpenRouter** 和 **WizardLM 2** 等替代方案以获得更好的性价比，一篇名为《*Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models*》的文章被推荐为获取 **Prompt Engineering** 见解的必读之作。

**推送延迟与性能难题**：新功能推送的延迟和大型 Prompt 的性能问题是普遍关注的焦点。为了应对处理庞大 Prompt 时的响应缓慢，提到了懒加载（lazy loading）作为解决浏览器困难的潜在方案。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**LM Studio GPU 传奇**：工程师们讨论了 **LM Studio models** 的行为特性，重点讨论了参数卸载（offloading parameters），确认在专用 **GPU** 上运行通常比共享资源效果更好，并强调了模型大小限制与 **GPU memory** 之间的微妙界限——提到模型应保持在 **6GB** 以下以缓解加载问题。

**为代码编写者推荐的模型**：**CodeQwen 1.5 7B** 和 **Codestral 22b** 模型被专门推荐用于代码优化任务，尽管 **Wavecoder Ultra** 的发布历史较为模糊，但也受到了推荐。此外，还强调了 [Extractum.io](https://llm.extractum.io/list) 等平台在根据 VRAM 和 quantization 等标准筛选模型方面的实用性。

**AI 性能的细节**：对话转向了 AI 局限性的技术细节，指出性能通常受限于 **memory bandwidth**，成员们建议将工作负载目标设定为处理器物理核心数的 **80%**。还提到了未来 **Chinese language support** 的不确定性。

**自建服务器引发讨论**：关于构建自定义 **homelab GPUs** 的讨论集中在 VRAM 容量、驱动支持以及不同制造商之间的性能。针对二手 GPU 的可靠性问题进行了探讨，成员们权衡了 **AMD ROCm** 与 NVIDIA 生态系统在稳定性和吞吐量方面的优劣。

**工程化 Beta 增强**：在软件开发和 AI 探索领域，**continue.dev** 因其对本地设置的支持（特别是支持 **LM Studio** 配置）而受到赞誉，同时为新的 **AVX-only extension pack** 招募测试人员，展示了社区的协作精神和持续的优化努力。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AI 登台**：围绕 *[Wayseer Manifesto - Official Video](https://youtu.be/OPR3GlpQQJA)* 展开了讨论，该视频因其励志信息而广受欢迎；Nous Research 的 [Twitter 账号](https://x.com/StudioMilitary) 引发的关于设计的讨论，暗示了 AI 社区内的创意才华。
- **OpenAI 揭秘**：出现了关于 [OpenAI's GPT-4](https://laion.ai/notes/open-gpt-4-o/) 的推测，成员们热切期待其潜在能力以及对未来 AI 研究和应用领域的影响。
- **为 T5 及更高版本做准备**：技术对话显示，T5 模型由于其沉重的硬件要求，为采用设置了很高的门槛；与此同时，像 Mobius 提供的用于聊天助手的开源 UI 以及通过 ggml 实现的潜在改进等替代方案也备受关注。
- **Pixart 的图形故障**：技术焦虑浮出水面，Pixart 在扩展到超过 10k 张图像的数据集时表现吃力，不像其他模型在高达 400k 张图像时仍能保持稳定，这归功于独特的训练方法。
- **WorldSim 的奇迹与智慧**：最近的 WorldSim Jam Session 已在 [YouTube](https://www.youtube.com/watch?v=qaE99xmtvd4) 上线，同时带有一丝讽刺地认识到 **Agent** 可能是第一个被 AI 超越的工作类别，一位重新活跃的成员通过分享他们的回归和研究进展庆祝了这一时刻。



---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3 面临严苛考验**：来自用户的初步反馈表明，**Stable Diffusion 3 (SD3)** 的早期模型在手部描绘方面表现挣扎，在某些方面落后于 Dall-E；然而，人们对于通过更广泛发布后的自定义模型来实现潜在改进仍持乐观态度。

- **架构的 AI 视角**：围绕将 **Stable Diffusion** 应用于**建筑可视化**展开了讨论，建议使用带有详细输入的 img2img 技术来增强输出质量，尽管该工具在渲染直线和几何精确的机械结构方面存在局限性。

- **插件陷阱**：用户在使用 **Stable Diffusion** 的 **wildcards 插件**时遇到了质量下降问题，尽管尝试了多次安装，仍报告出现颗粒感结果和色彩失真。

- **社区模型挖掘**：工程社区建议探索 [civitai.com](https://civitai.com) 等平台上可用的社区模型，并利用 [ChaiNNer](https://github.com/JoeyBallentine/chaiNNer)（一个基于节点的图像处理 GUI 工具）来改进和放大 **Stable Diffusion** 的图像结果。

- **AI 的名人难题**：**Civit** 上 AI 生成的网红简介（如“名人 LoRas”）的兴起，引发了一场关于 AI 时代名人本质的戏谑辩论，突显了随着这些个人资料获得追随者和媒体关注，虚拟与现实之间的界限正变得模糊。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**SD3 模型苦于颗粒感结果**：用户指出，尽管使用了 16ch VAE 等高级特性，SD3 2B 模型仍存在*斑点噪声问题*，噪声伪影在流水等区域尤为明显。有人对 SD3 模型目前的验证指标和损失函数表示怀疑，认为它们无法很好地指示模型性能。

**视频模型的开源突破**：社区对 [Apache2 许可的具备视频能力的 CV-VAE](https://arxiv.org/abs/2405.20279) 表现出极大的热情，预计这将成为研究基于 Latent Diffusion 的视频模型的重要资源。

**窥探未来模型架构**：新发布的研究介绍了 **State Space Duality (SSD)** 框架和尖端的 **Mamba-2** 架构，据称其速度比前代快 2-8 倍，在语言处理任务中足以与 Transformer 模型竞争（[arxiv 论文](https://arxiv.org/abs/2405.21060)）。

**训练策略备受关注**：一份预印本建议，通过对预训练数据集进行轻微损坏来扰动 Embedding，可以提高扩散模型的图像质量（[arxiv 预印本](https://arxiv.org/abs/2405.20494)），而其他人则提到使用 dropout 和数据增强来防止大型扩散模型的过拟合，并讨论了增加训练数据难度是否能增强模型的鲁棒性。

**审美评估与现实主义之争**：SD3 图像与 Google 的写实案例之间的对比引发了讨论，SD3 的图像被幽默地比作“遭受了糟糕肉毒杆菌注射的女性”（[Reddit 示例](https://www.reddit.com/r/StableDiffusion/comments/1d73j3r/some_sd3_images_women/)），而 Google 的工作则因其织物纹理和一致的头发呈现而赢得赞誉（[Google 演示](https://vxtwitter.com/GoogleDeepMind/status/1797605392089825457)）。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **是电子纸还是非电子纸？这是一个问题**：在一段测评视频暗示 Daylight 平板可能采用的是反射式 LCD 而非电子纸后，成员们就其电子纸宣称的真实性展开了激烈辩论。讨论集中在 Daylight 是对现有的 Sharp RLCD 技术的错误更名，还是真正的创新，成员们建议通过拆解来查明真相。

- **超越 Heads 和 MLPs：发现 LLM 知识电路**：一篇揭示 LLM 知识电路更深层见解的新研究论文引起了成员们的关注，其中一位成员对其摆脱对单一网络组件的关注表示赞赏。研究社区还深入探讨了受损数据集是否能改善 diffusion models，以及 RNNs 与 pre-trained transformers 之间的协同作用。

- **公共服务 AI？**：
  用户讨论了 AI 任务处理的效率改进，关注单次查询处理缓慢的问题以及默认 n-shot 值对结果的影响。此外，还有一项关于寻找最小的 Huggingface decoder 模型以研究能源消耗的实践探索，以及一个引入多语言机器翻译 ARC 挑战的 GitHub [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1900)。

- **效率猎寻**：用户对模型大小表示担忧，并努力压缩像 TinyLLama.db 这样 22GB 的模型，以实现更好的激活和权重项协调。此外，社区还在思考将可微 top-k 函数用于图像分类，这可能有助于模型聚焦于最重要的元素。

- **全球基准测试迈向多语言化**：通过一个协作式的 [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1900)，提出了一项将 ARC 挑战基准扩展到 11 种机器翻译语言的倡议，并着眼于未来增加更多语言。目前正在对这一基准测试的多语言扩展进行审查和贡献。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **加密货币充值困惑**：一位用户遇到了 **ETH 支付**后额度未显示的问题，得到的建议是在投诉前先等待一小时；在区块链的时间机制中，耐心可能是一种美德。
- **LLMs 对 Prefills 的熟练度**：用户间的共识是 **Language Learning Models (LLMs)** 能够熟练处理 **prefill 文本**，确保后续生成的内容与初始输入保持一致。
- **Turbo 的烦恼与 API 监管**：**GPT-3.5 Turbo 的不一致性**引发了关于潜在 **API moderation** 的讨论，并提醒 OpenAI 要求所有通过其 *moderation API* 发出的请求都必须进行审核。
- **Mistral 的短暂沉默**：关于收到 **Mistral: Mixtral 8x22B Instruct** 空白回复的报告，促使管理员建议将 **DeepInfra** 设置为首选供应商，并查看 [负载均衡文档](https://openrouter.ai/docs/provider-routing) 以解决特定供应商的问题。
- **通过神经网络实现的叙事细微差别**：在辩论最适合讲故事的模型时，用户推荐了各种 **roleplay-specific 模型**，并建议关注 [OpenRouter 排名](https://openrouter.ai/rankings/roleplay?view=week)，以了解那些在创意领域表现尤为出色的模型。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **一行代码提升 Python 速度**：一段[教程视频](https://youtu.be/OiMZtjSZVOw)重点介绍了 **Numba** 如何利用 JIT 编译显著提升 Python 性能。然而，这一行代码带来的影响也引发了人们对于在不使用额外库的情况下实现类似性能潜力的兴趣。

- **Python 与 Mojo 的效率**：讨论了在 Python 中使用 **while 循环嵌套 for 循环**的有效性，并建议通过 [Real Python 资源](https://realpython.com/introduction-to-python-generators/)探索 generators。此外，还讨论了 Mojo 的 **MAX** 工具加速 Python 执行的可能性，并将其与 **Tensor** 和 **Torch** 库带来的增强进行了比较。

- **Mojo Async 范式引发争议**：将所有 Mojo 函数默认设置为 `async` 的建议引发了辩论。人们担心这会偏离 Python 标准，并为那些习惯于显式 `async`/`await` 方法论的开发者增加工作流的复杂性。

- **Modular 深度解析所有权 (Ownership)**：发布了一篇名为[《深度解析 Mojo 中的所有权》](https://www.modular.com/blog/deep-dive-into-ownership-in-mojo)的博客文章，其中包含 CEO Chris Lattner 的见解。这篇文章是之前关于所有权作为概念框架探索的续篇。

- **通过 Project Verona 挑战 Rust**：**Project Verona** 成为关注焦点，被视为 **Rust** 的竞争对手，旨在提供内存安全的同时拥有更平缓的学习曲线。爱好者们可以观看 [YouTube 演讲](https://youtu.be/VU9QATDvidw)——“*Concurrent Mutation must go*”，以深入了解该主题。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **为 Ollama 模型驯服 Bind_Tools**：成员们确认 **LangChain** 通过 `OllamaFunctions` 类支持 **Ollama 模型** 的 `bind_tools`，并提供了一个相关的 GitHub issue 链接作为参考。
  
- **利用 AI 构建客户支持系统**：关于创建 **AI 驱动的客户支持系统** 的持续讨论确定了 LangChain、Llama3 等 LLM，以及用于用户验证等操作的自定义工具，并分享了 Python 代码示例来演示如何链接模型和工具。

- **在 SQL 失败时保留对话记忆**：SQL Agent 聊天上下文保留是一个热门话题，其中分享了一个使用 `ConversationBuggerMemory` 的代码片段。然而，存在关于不支持的 kwargs 的担忧。

- **使用 LangChain 和 Embeddings 进行分类**：公会探索了使用 **LangChain 和 Embeddings** 对 10,000 条自由文本回答进行分类的策略，强调了使用 Prompt Engineering 来提高效率。

- **通过自动化聊天分析器重新定义文本编辑**：介绍了一种**自动化聊天分析器**，它可以从消息列表中生成可编辑的纯文本问答（Q&A），旨在简化手动编辑并减少计算资源消耗。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **后端开发中的位移操作 (Bitshifting) 热议**：工程师们就 [PR #4728](https://github.com/tinygrad/tinygrad/pull/4728) 在 **tinygrad** 所有后端实现位移操作的优缺点展开了激烈辩论，对其与传统乘/除操作相比带来的性能提升持怀疑态度。
  
- **GPU 领域的测试难题**：有人对“gpuctypes”中缺失设备测试表示好奇，并引用了一个特定的 [`test_cuda.py` 缺失测试](https://github.com/tinygrad/gpuctypes/blob/c4c4394239dce4d4ecfe7016ca268c49cb2d02a4/test/test_cuda.py#L36)案例，这引发了关于彻底测试实践的持续讨论。

- **与 Hotz 一起深入探索 tinygrad 核心**：George Hotz 透露了 **tinygrad** 演示计划，重点在于代码库的清晰度和深度解析，强调了该项目对 CUDA 等依赖项的独立性。

- **倾向于更清晰的 Lean 机械化悬赏**：社区正在处理关于 Lean 中 `ShapeTrackers` 模糊的问题陈述，建议查阅 tinygrad 仓库中的 [ShapeTracker](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/shapetracker.py) 以获得更清晰的理解。

- **Tensor 的 Traceback 追踪**：重新讨论了为 **tinygrad** 中新的 `Tensor` 实例添加“traceback”属性的提案，强调了尽管之前有 PR #3302 等未完成的尝试，但这在增强调试方面具有潜力。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LLM 生成的 Python 脚本大获全胜**：一个 LLM 展示了其脚本编写能力，生成了一个用于从 Gmail [提取结构化数据](https://t.co/N2BJ54zr7i)的 Python 脚本，这种方法可以简化跨不同电子邮件数据集的数据提取流程。
- **Google Gemini 扩大窗口**：LlamaIndex Agent 强调了 [Google Gemini](https://t.co/Qg9ydPzBdd) 的结构和处理能力，推崇其令人印象深刻的 100 万 Token 上下文窗口，用于处理来自异构文档的复杂、多方面查询。
- **自定义解析难题与解决方案**：实施 Langchain 的 `HTMLHeaderTextSplitter` 带来的挑战促使在 LlamaIndex 的 `IngestionPipeline` 中开发了自定义解决方案，并得到了 [自定义转换文档](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/transformations/#custom-transformations) 的支持。
- **Rkhettry 破解 Chroma 代码**：一位用户通过直接访问方法解决了 Chroma 向量存储中 `VectorStoreIndex` 的文档检索问题，展示了数据库操作中实际的问题解决方法。
- **元数据魔法增强索引**：建议加入元数据以改进索引系统中的文档检索，如 LlamaIndex 的 [元数据提取指南](https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC/) 所述，强调了丰富的文档描述符对于细粒度搜索能力的重要性。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Phi-3 模型在排行榜上攀升**：**Phi-3 Medium (14B)** 和 **Small (7B)** 模型在 **@lmsysorg 排行榜**上备受关注，其中 Medium 模型的性能接近 **GPT-3.5-Turbo-0613**，而 Small 模型则可与 **Llama-2-70B** 及各种 **Mistral 微调版**相媲美。社区反应既有对个人赌注落空的幽默调侃，也有关于此类模型排名中可持续增长和声誉提升的严肃讨论。

- **OpenAI 的内部动荡**：现任和前任 OpenAI 员工发表了[一封公开信](https://righttowarn.ai/)，表达了对 AI 开发缺乏足够监管的担忧，而研究员 Leopold Aschenbrenner 因涉嫌泄露机密信息被解雇，提升了围绕商业机密和国家安全的讨论。此外，频道中对将扩大计算规模作为实现 AGI 的线性路径持怀疑态度，用户质疑在没有巨大挑战的情况下这种增长的可持续性。

- **Scaling Laws，认真的吗？**：对 Scaling Laws 的讨论带有一层幽默色彩，用户通过梗图嘲讽对 2030 年代持续缩放的信念，并俏皮地请求“再来 10 万个 GPU”以实现假设的 10 万亿参数模型。AGI 辩论中截然不同的信念引发了挫败感，批评指向那些几乎不承认其认识论不确定性的各方。

- **应对 AI 吹哨人保护措施**：[OpenAI 员工的公开信](https://righttowarn.ai/)证明了对 AI 安全和监管的担忧，随后的讨论说明了人们对快速发展的 AI 可能因强大的财务动机而处理不当、反对强有力监管的忧虑。

- **陈旧的梗图（Memes）无人问津**：#[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/) 频道极低的参与度表明，要么是人们对梗图文化不感兴趣，要么是需要更新鲜的内容来刺激 AI 工程师群体之间的交流。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **TorchTune 寻求时事通讯认可**：来自 Torchtune 社区的一位成员请求将其在 SOTA 模型和方法方面的工作纳入 AI News 时事通讯，并邀请其他人通过[提供的邀请链接](https://discord.gg/6cGsKMM5)加入他们的服务器。
  
- **AI News 投递中的技术故障**：据报告，ProtonMail 上的 AI News 邮件存在格式故障，导致在启用深色模式时出现显示问题，仅能清晰看到链接和图像。

- **播客幕后揭秘**：播客自动转录（包括发言人识别）背后的“秘密武器”被披露为 Google 的 smol-podcaster 工具，并辅以对发言人姓名的手动编辑。

- **LiveKit 的丰厚飞跃**：LiveKit 成功筹集了 2250 万美元的 A 轮融资，旨在为 AI 的传输层建立全新的基础设施，专注于实时语音和视频交互，尽管在 GPT-4 出现后融资经历充满挑战。详情见此[推文](https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww)。

- **Twelve Labs 的多模型吸金力**：Twelve Labs 获得了 5000 万美元的 A 轮投资，用于创新视频理解 AI 基础模型，并推出了全新的 Marengo 2.6，该模型在单个 API 中融合了多模态能力；完整信息可在其[新闻稿](https://www.prweb.com/releases/twelve-labs-earn-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html)中找到。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **无需论文发表的会议乐趣**：工程师们反思了即使没有投稿论文，参加会议也能获得回报的经历，这体现了社区和知识交流的重要性。

- **寻求 OrPO 格式化器的帮助**：一位用户正苦于编写用于自定义 **OrPO 格式化器的 Python 脚本**以对数据集进行 Tokenize，并寻求支持。相关脚本已共享以供参考。

- **AI 的医学诊断困境**：一条推文强调了 **GPT-4V 和 Gemini Pro** 等先进 AI 模型在医学视觉问答（VQA）任务中的糟糕表现，该任务使用 **ProbMed 数据集**作为基准。工程社区讨论了视觉 LLM 在医学领域面临的挑战。

- **寻求并成功获得 arXiv 背书**：一位 AI Collective 成员为其 cs.LG 类别的论文寻求 **arXiv 背书**，并最终通过使用其组织邮箱解决了难题。

- **排除 LoRA 训练故障**：一位工程师遇到了 **QLoRA 训练输出**未按预期启动的问题。另一位成员指出，如果本地没有模型，LoRA 训练脚本可能会自动从 **Hugging Face Model Hub** 下载必要的模型。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Artificial Ivan 像专家一样排除故障**：**Cohere** 已将其 "Artificial Ivan" 升级至 4.0 版本，使其能够排除代码故障，并分享了一个与其开发相关的[肯定语应用](https://amorisot-cohere-demos-onceuponatime-x1w9hn.streamlit.app/affirmations)。

**现实中的 Ivan 要提前退休了？**：一位用户开玩笑说，由于 Artificial Ivan 的成就，其人类对应者“现实中的 Ivan”可能会在 35 岁退休，这为该项目的成功增添了幽默色彩。

**跨项目协同效应解锁**：一位用户强调了 **Aya 23** 与 **Llama.cpp** 及 **LangChain** 的集成，提供了示例代码，并寻求在对话中使用 "\n" 实现停止条件的帮助。

**寻求双语 AI 的简洁性**：用户详细介绍了旨在产生简洁西班牙语回答的代码，概述了如何使用 Prompt 进行对话记忆，以及通过参数增强 **Aya 23** 的性能。

**Cohere 社区角落**：与 Langchain 相比，一位公会成员俏皮地将 Cohere Discord 描述为“长期在线的 AI 实验室”，指出了其成员之间活跃的互动和参与。



---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **CUDA 难题被攻克**：一位工程师通过使用 `--recompile --gpu NVIDIA` 解决了 CUDA 错误问题，并发现 `-ngl 9999` 标志必须放在 `--recompile` 之后才能有效解决。

- **社区力量在问题解决中胜出**：CUDA 重编译的成功排查归功于社区协作和对 `--help` 选项的深入研究，同时还分享了一个有用的 [GitHub 资源](https://github.com/Mozilla-Ocho/llamafile/blob/main/build/llamafile-upgrade-engine)。

- **工程师强调团结与福利**：社区内的氛围强调了相互学习的重要性，并幽默地提到“曲奇饼干”也是社区精神的一部分。

- **CPU 操作随 RISC-V 进入新阶段**：最近的一段 YouTube 视频 [“The Magic of RISC-V Vector Processing”](https://www.youtube.com/watch?v=Ozj_xU0rSyY) 介绍了已批准的 1.0 RISC-V Vector 规范及其对向量计算的预期影响。

- **链接分享**：工程师们可以通过两个关键链接获取更多细节和讨论：解释新规范和进展的 [RISC-V Vector Processing 视频](https://www.youtube.com/watch?v=Ozj_xU0rSyY)，以及用于辅助分布式 LLM 的 [llamafile GitHub 仓库](https://github.com/Mozilla-Ocho/llamafile/blob/main/build/llamafile-upgrade-engine)。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Windows 高手搞定 Gemini 设置**：一位成员针对 Windows 系统列出了详细的 **Gemini 模型设置指南**，理由是官方文档已过时。该指南包括命令行步骤以及针对常见设置问题的解决方法。
  
- **AR 通过 Spacetop 进入工作空间**：Sightful 的 **Spacetop AR 笔记本电脑** 因其无电池眼镜和旨在取代传统笔记本显示屏的独特设计而备受关注，引发了关于其在未来移动计算中地位的讨论。成员们还讨论了 **Xreal 眼镜**，提到了它们依赖外部设备供电，以及在分辨率和视野方面需要改进。
  
- **Macbook M1 遭遇无限循环**：一位用户报告了在 Macbook M1 上运行 `poetry run 01 --local` 时遇到的持久问题，在使用 **ollama llama3** 设置时面临无限循环，目前正在寻求解决方案。
  
- **安全版本查询未获回应**：有人提出了关于发布讨论主题的**安全版本**的问题，但在讨论中没有得到具体回答或后续跟进。

- **探索 AR 眼镜的可用性和电池寿命**：关于 **Xreal 眼镜** 的对话包括了在 MacBook 上使用的体验，强调了它们目前在分辨率、电池寿命和供电对设备的依赖性方面的局限。

---

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord

- **发起直接对话**：**ai-ml** 频道的成员表达了合作兴趣，并选择通过私信继续交流，以便进行更详细的讨论。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI 与 ML 深度剖析**：由 Qwak 主办的 **Infer: Summer ‘24** 虚拟会议将于 6 月 26 日举行，深入探讨 AI 和 ML 实践，重点关注行业专家的实际应用。
- **准备好进行深度知识钻研**：对**推荐系统**和体育领域 AI 感兴趣的人将发现关于高级 ML 模型构建和 AI 驱动的体育分析的专题会议。
- **AI 安全成为焦点**：AI 安全和监管合规将成为主要议题，重点介绍如“模式化提问（Schematic Questioning）”等策略，以减轻 AI 系统中内容不准确等风险。
- **从流媒体到银行业**：与会者可以期待来自 Disney Streaming、Lightricks、LSports 和 Lili Banking 等重量级企业的见解，他们将分享 AI/ML 集成的实战经验。
- **生产就绪型 LLM 探讨**：GPT 和 Llama 等大语言模型（LLM）占据显著地位，计划讨论在各行业生产环境中的有效实施。[会议注册](https://tinyurl.com/j8z6s8ka)

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **xLSTM 的开源首秀**：Dr. Tristan Behrens 宣布发布 **xLSTM** 的源代码，这一举动无疑会让 AI 工程师和开发者感到兴奋。官方公告和源代码访问链接可以在他的 [LinkedIn 帖子](https://www.linkedin.com/posts/dr-tristan-behrens-734967a2_drop-everything-nxai-has-released-the-official-activity-7203659602628935682-GwOA)中找到。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**Datasette - LLM (@SimonW) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

# 第二部分：分频道详细摘要与链接

{% if medium == 'web' %}

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1247386474589061120)** (16 条消息🔥): 

- **管理数据集中的机密数据**：讨论了在私有数据集上使用工具的选项，例如使用 AutoTrain 训练私有数据模型，并与 Datasets 库集成以处理来自 CSV、JSON 和 parquet 文件等格式的数据。

- **丰富的 DSPy 资源**：成员们分享了多个入门 **DSPy** 的资源，包括 Hamel 关于[拦截 LLM API 调用](https://hamel.dev/blog/posts/prompt/)的博客文章，以及一段名为 [DSPy Explained](https://youtu.be/41EfOY0Ldkc?si=e6mVFi9tC6KJOaQC) 的 YouTube 视频。

- **Python Logging 书籍发布**：Michael Driscoll 编写的 Python logging 分步指南（通过 Kickstarter 资助）已发布，代码示例可在 [GitHub](https://github.com/driscollis/pythonlogging) 上获取。

- **Zoom 会议提问引导**：引导用户在当前的 Zoom 会议期间将问题发布到正确的频道。

- **Situational Awareness AGI 论文**：分享了 Leopold Aschenbrenner 关于 AGI 未来及其影响的详细论文链接。该论文可在 [situational-awareness.ai](https://situational-awareness.ai) 阅读。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://situational-awareness.ai/">Introduction - SITUATIONAL AWARENESS: The Decade Ahead</a>：Leopold Aschenbrenner，2024 年 6 月。你可以首先在旧金山预见未来。在过去的一年里，业内的谈论重点已从 100 亿美元的算力集群转向 1000 亿美元乃至万亿级别的集群...</li><li><a href="https://youtu.be/41EfOY0Ldkc?si=e6mVFi9tC6KJOaQC">DSPy Explained!</a>：大家好！非常感谢观看这段关于 DSPy 的讲解！DSPy 是一个非常令人兴奋的开发 LLM 程序的新框架！由...发起。</li><li><a href="https://hamel.dev/blog/posts/prompt/">- Fuck You, Show Me The Prompt.</a>：通过拦截 API 调用快速理解难以捉摸的 LLM 框架。</li><li><a href="https://github.com/driscollis/pythonlogging">GitHub - driscollis/pythonlogging: Code examples for the Python Logging book</a>：Python Logging 书籍的代码示例。通过创建账号为 driscollis/pythonlogging 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1247275041666629632)** (2 条消息): 

- **LLM 验证医疗账单**：一位成员分享了他们将 **LLM 用于德国 DRG 系统**的用例，作为医疗控制员验证医院账单。它利用 **RAG** 技术结合患者信息和法律文本，识别潜在的编码违规。

- **5 个实用的 Finetuning 用例**：另一位成员概述了五个引人注目的 Finetuning 用例，包括：
  1. **医疗检测结果摘要**及个性化问答。
  2. 通过通话记录**评估销售人员表现**。
  3. **Whatsapp 机器人分流**。
  4. 使用 Whisper **转录地区口音**。
  5. **创建搜索助手**。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1247311590722633760)** (11 条消息🔥): 

- **Modal 惊人的营收增长令人印象深刻**：一篇 Substack 文章强调了 Modal 令人印象深刻的营收增长，一年内增长了 50 倍以上，达到了八位数的运行率 (run rate)。他们目前正在纽约市招聘软件工程师。[Modal 招聘链接](https://jobs.ashbyhq.com/modal/dd84cf88-1f13-4f39-b371-237e103fce34)。

- **Etched 为 Transformer 构建了更快的芯片**：Etched 开发了一种性能显著优于 GPU 的芯片，运行 Transformer 的速度提高了 10 倍以上。他们正在库比蒂诺积极招聘各种工程岗位。[Etched 招聘链接](https://boards.greenhouse.io/etchedai)。

- **Modal 的基础设施实力**：一位成员分享了 Erik 的演讲，内容关于 Modal 如何使用定制解决方案（如用 Rust 编写的文件系统和他们自己的容器运行时）来优化其数据基础设施。这种方法使他们能够在几秒钟内高效启动数千个大型容器。[Erik 在 Data Council 的演讲](https://www.datacouncil.ai/talks/creating-our-own-kubernetes-and-docker-to-run-our-data-infrastructure)。

- **Modal 额度有效期为一年**：关于 Modal 额度有一项澄清，确保它们不会在计费周期结束时过期，而是可以持续一年。

- **提供详细的微调指南**：一位成员称赞了 Modal 的文档和教程仓库，认为它们在微调 LLM 方面非常清晰且有用。该指南包括高级技术，如 [LOra](https://huggingface.co/blog/peft)、[Flash Attention](https://arxiv.org/abs/2205.14135) 和 [Gradient checkpointing](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)。[Modal 微调示例](https://modal.com/docs/examples/llm-finetuning)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://modal.com/docs/examples/llm-finetuning">几分钟内微调 LLM (支持 Llama 2, CodeLlama, Mistral 等)</a>：厌倦了提示工程 (prompt engineering)？微调通过调整模型权重来更好地适应特定任务，帮助你从预训练 LLM 中获得更多收益。这份操作指南将帮助你使用基础模型...</li><li><a href="https://whyyoushouldjoin.substack.com/p/modal.">为什么你应该加入 Modal</a>：GPU 的未来是 Serverless。</li><li><a href="https://www.datacouncil.ai/talks/creating-our-own-kubernetes-and-docker-to-run-our-data-infrastructure">创建我们自己的 Kubernetes 和 Docker 来运行我们的数据基础设施</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1247590113009995786)** (4 条消息): 

- **Jarvis Labs 额度赠送**：所有报名课程的用户都收到了 **200 美元的额度**，用于在 Jarvis Labs 上使用。一位成员幽默地提到，并没有额外的额度被意外添加到他们的账户中。

- **公告频道**：该频道被强调为发布 **Jarvis Labs 公告**的最佳场所。鼓励成员们宣传该平台并提供反馈。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1247489054832857088)** (130 messages🔥🔥): 

- **Hugging Face 额度发放正在进行中**：填写了表格的成员已开始看到其账户中到账了 $501.42 的额度，可在 [账单设置](https://huggingface.co/settings/billing) 中查看。预计剩余的发放将在一天内完成，部分用户遇到延迟，建议在问题持续时发送邮件至 `billing@huggingface.co`。

- **双倍额度问题**：几位用户报告收到了双倍的预期额度（$1002.84）。Hugging Face 团队已确认该问题，并保证会自动进行修正。

- **推理和算力选项说明**：用户可以通过 [Inference Endpoints 文档](https://huggingface.co/docs/inference-endpoints/index) 管理 Endpoint 和自动扩缩容（auto-scaling）以提高成本效率，并利用 [Spaces 的 GPU 升级](https://huggingface.co/docs/hub/spaces-gpus) 处理计算密集型任务。

- **微调自定义模型**：关于使用 Hugging Face 算力运行自定义代码和微调模型的咨询，被引导至 [SpaceRunner](https://huggingface.co/blog/abhishek/autotrain-spacerunner) 和 [Spaces Dev Mode](https://huggingface.co/blog/spaces-dev-mode) 等资源，这些资源允许使用自定义训练脚本和环境。

- **社区诚信受到赞扬**：报告额度和错误问题的用户因其诚实而受到感谢。Hugging Face 团队高度赞赏社区的诚信与协作。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/api-inference/en/index">Serverless Inference API</a>：未找到描述</li><li><a href="https://huggingface.co/docs/inference-endpoints/index">🤗 Inference Endpoints</a>：未找到描述</li><li><a href="https://ui.endpoints.huggingface.co/">Inference Endpoints - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/inference-api/serverless">Inference API (serverless) - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blog/abhishek/autotrain-spacerunner">使用 AutoTrain SpaceRunner 在 Hugging Face Spaces 上训练自定义模型</a>：未找到描述</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5">nomic-ai/nomic-embed-text-v1.5 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/hub/en/spaces-sdks-docker-jupyter">Spaces 上的 JupyterLab</a>：未找到描述</li><li><a href="https://huggingface.co/blog/spaces-dev-mode">推出 Spaces Dev Mode 以提供无缝的开发者体验</a>：未找到描述</li><li><a href="https://tenor.com/view/bill-nye-party-horn-confetti-sarcastic-like-child-gif-5499505">讽刺的庆祝 GIF - Bill Nye 派对喇叭纸屑 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/settings/billing">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://huggingface.co/docs/inference-endpoints/autoscaling#scaling-to-0">自动扩缩容 (Autoscaling)</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/launch">Spaces Launch – Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/hub/spaces-gpus#sleep-time)">使用 GPU Spaces</a>：未找到描述</li><li><a href="https://huggingface.co/docs/hub/spaces-gpus#pause)">使用 GPU Spaces</a>：未找到描述</li><li><a href="https://huggingface.co/settings/billing.">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1247573592447651952)** (11 messages🔥): 

- **注意查收 Replicate 额度邮件**：提醒成员检查收件箱，查看来自 **Replicate** 的额度兑换邮件。*"请留意今天的收件箱，查收来自 Replicate 的额度兑换邮件。"*
- **创建 Replicate 组织需依赖 GitHub**：用户询问了在 **Replicate** 上创建“组织（org）”的必要性。明确回复称，虽然可以使用个人账户兑换额度，但创建组织需要关联 GitHub 组织。*"Replicate 组织目前挂靠在 GitHub 组织上，因此你需要先创建一个 GitHub 组织，或者使用现有的组织。"*
- **关于未设置账单时额度可见性的查询**：有用户担心在未设置账单信息的情况下无法看到额度。回复指出额度应该仍然可见，但已要求该成员私信详细信息，以便向产品团队确认。*"我认为即使还没有设置账单，你也应该能够看到/兑换额度，但我会向产品团队确认一下。"*
- **额度领取情况不一**：多位用户报告已成功收到并领取了 **Replicate** 额度。然而，有一位用户提到他们尚未收到邮件。*"我收到了 Replicate 额度兑换邮件"*，*"我拿到我的了。"*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1247292497953030236)** (4 messages): 

- **成员遇到额度问题**：一位成员询问其他人是否已收到额度，并提到自己的账户中没有看到任何额度。另一位成员也确认尚未收到。
- **表单填写错误引发担忧**：一位成员分享说，他们在手机上填写表单时，误将组织名称填成了组织 ID。他们担心现在纠正这个错误是否为时已晚。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/1247475652677074985)** (2 messages): 


- **John 的演讲非常精彩**：一位用户赞扬了 John 的演讲，称他是一位出色的演讲者。他们提到了他幽默的言论：*"LLM 是愚笨的机械人类，Temperature 是模型的血液酒精浓度，"* 并表示有兴趣阅读他的著作。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1247323056788082748)** (7 messages): 

- **探索使用模型解析屏幕截图**：一位用户讨论了 **Opus 4o** 和 **MiniCPM-Llama3-V-2_5** 在解析图像文本方面的能力，并考虑通过 Finetuning 来提升性能。他们表达了在此任务上进行协作的意愿。
- **PaliGemma 用于 OCR 任务**：有成员指出 **PaliGemma** 除了生成多语言字幕等任务外，还可以执行 OCR。对于需要处理屏幕截图但又不想进行复杂模型适配的情况，这可能是一个有用的替代方案。
- **政府数据的 Finetuning 与检索对比**：一位用户分享了他们的政府农业补助金数据集，并考虑是使用问答对对 LLM 进行 Finetuning，还是直接在全文上进行训练。另一位用户建议，鉴于补助金数据的结构化特性，可以探索 **检索应用（retrieval applications）**。
- **RAG 实验**：考虑 Finetuning 的用户也承认 **RAG (Retrieval-Augmented Generation)** 可能是更合适的应用方向，但目前重点在于出于教学目的尝试模型训练。这种方法可能为未来的应用提供基础性的理解。

**提到的链接**：<a href="https://www.gov.uk/countryside-stewardship-grants">Countryside Stewardship 补助金查询器</a>：查找有关 Countryside Stewardship 选项、资本项目和补充资金的信息。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-4](https://discord.com/channels/1238365980128706560/1242223495673286737/1247587624131231755)** (361 messages🔥🔥): 

_

- **HuggingFace 与模型推理服务见解**：将 LoRA 与基础模型通过 Axolotl 合并引起了广泛关注，正如 [Axolotl documentation](https://openaccess-ai-collective.github.io/axolotl/#merge-lora-to-base) 链接中所强调的那样。一位用户分享道：“HuggingFace 做了 *很多* 事情……很难做到既 (1) 提供简洁的用户界面……又 (2) 在不打扰用户的情况下告知他们所有的可能性。”
- **Modal 与 Charles 的额度实力展示**：来自 Modal 的 Charles 展示了一场令人印象深刻的演讲，并向在 6 月 11 日之前尝试过 Modal 的参与者发放了额外的 $500 credits。这一举动引发了热烈讨论，评论如：“charles_irl 关于 ‘Inference Trilemma’ 的分享；Modal 和 Charles 的表现简直太强了。”
- **Medusa 与 LLM 推理创新**：多位用户称赞了 Medusa 的引入，这是一种通过“增加额外的解码头来并行预测多个后续 token，从而增强 LLM 推理”的方法，详情见 [Medusa's paper](https://arxiv.org/abs/2401.10774)。
- **处理量化挑战**：针对量化推理是否变慢的疑问，解释指出量化是用增加计算开销来换取内存开销的减少。讨论中还涉及了 bitsandbytes 量化效率较低的具体问题。
- **Predibase 与 LoRAX 的热度**：来自 Predibase 的 Travis Addair 提到将安排答疑时间（office hours）来讨论 LoRAX 及其相关话题。他为对 multi-LoRA 推理服务器感兴趣的用户分享了 [LoRAX GitHub](https://github.com/predibase/lorax) 的链接。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://huggingface.co/parlance-labs/hc-mistral-alpaca-merged-awq">parlance-labs/hc-mistral-alpaca-merged-awq · Hugging Face</a>: 未找到描述</li><li><a href="https://outerbounds.com/blog/the-many-ways-to-deploy-a-model/">部署模型的多种方式 | Outerbounds</a>: 部署模型和执行推理有多种方式。在这里，我们以 LLM 推理为例，分享了模型部署的决策准则。</li><li><a href="https://huggingface.co/VAGOsolutions/Kraken-LoRA">VAGOsolutions/Kraken-LoRA · Hugging Face</a>: 未找到描述</li><li><a href="https://cog.run/">Cog</a>: 未找到描述</li><li><a href="https://docs.vllm.ai/en/stable/quantization/auto_awq.html">AutoAWQ &#8212; vLLM</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2401.10774">Medusa: 具有多个解码头的简单 LLM 推理加速框架</a>: 大语言模型 (LLMs) 的推理过程通常受限于自回归解码过程中缺乏并行性，导致大多数操作受到内存带宽的限制...</li><li><a href="https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html">Answer.AI - 使用 FSDP QDoRA 高效微调 Llama 3</a>: 我们正在发布 FSDP QDoRA，这是一种可扩展且内存高效的方法，旨在缩小参数高效微调与全量微调之间的差距。</li><li><a href="https://www.deeplearning.ai/short-courses/efficiently-serving-llms/">高效提供 LLM 服务</a>: 使用 KV caching 等技术在生产环境中提供 LLM 应用服务。学习应用 LoRA 和 LoRAX 等框架。</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/#merge-lora-to-base">Axolotl</a>: 未找到描述</li><li><a href="https://tenor.com/view/burn-elmo-pyro-burn-it-down-ashes-gif-12152943568085011868">Burn Elmo GIF - Burn Elmo Pyro - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/eugeneyan/status/1798073562764620003">Eugene Yan (@eugeneyan) 的推文</a>: .@charles_irl 关于“推理三难选择” (TLC)：吞吐量 (Throughput)、延迟 (Latency)、成本 (Cost)。其他三难选择：• CAP：一致性 (Consistency)、可用性 (Availability)、分区容错性 (Partition Tolerance) • 不可能三角：固定汇率、自由资本流动...</li><li><a href="https://predibase.com/fine-tuning-index">微调指数 (The Fine-tuning Index)</a>: 超过 700 个开源 LLMs 微调的性能基准测试</li><li><a href="https://tenor.com/view/yes-excited-screaming-gif-20651075">Yes Excited GIF - Yes Excited Screaming - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://docs.google.com/presentation/d/1PS0nWigtRLQd5q0czCvNzu19Qc7eM5o6pI7mDS5JxrI/edit#slide=id.g2c7588f453b_0_272">精通 LLMs - 在 Modal 上部署 LLM 服务</a>: 在 Modal 上部署 LLM 服务 bit.ly/mastering-llms-deployment</li><li><a href="https://x.com/TheZachMueller/status/1798078059247259729">Zach Mueller (@TheZachMueller) 的推文</a>: 我：我们发完课程推文了。@modal_labs：</li><li><a href="https://tenor.com/view/making-it-rain-gif-617522917670078816">Making It Rain GIF - Making it rain - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/eugeneyan/status/1798074567023628697">Eugene Yan (@eugeneyan) 的推文</a>: @charles_irl @modal_labs 和 @charles_irl 的疯狂举动，给尝试过 Modal 的学生又提供了 500 美元的额度</li><li><a href="https://github.com/parlance-labs/ftcourse/tree/master/replicate-examples">parlance-labs/ftcourse 的 master 分支下的 replicate-examples</a>: 通过在 GitHub 上创建账号，为 parlance-labs/ftcourse 的开发做出贡献。</li><li><a href="https://github.com/NVIDIA/TensorRT-LLM">GitHub - NVIDIA/TensorRT-LLM: TensorRT-LLM 为用户提供了一个易于使用的 Python API，用于定义大语言模型 (LLMs) 并构建包含最先进优化技术的 TensorRT 引擎，以便在 NVIDIA GPUs 上高效执行推理。TensorRT-LLM 还包含用于创建执行这些 TensorRT 引擎的 Python 和 C++ 运行时的组件。</a>: TensorRT-LLM 为用户提供了一个易于使用的 Python API，用于定义大语言模型 (LLMs) 并构建包含最先进优化技术的 TensorRT 引擎，以便高效执行推理...</li><li><a href="https://x.com/bytetweets/status/1798078590749388851">Florian Buetow (@bytetweets) 的推文</a>: 是的，简直疯狂。他们在 @MavenHQ 上由 @dan_s_becker 和 @HamelHusain 主持的 LLM 微调会议/课程中，为我们学生又增加了 500 美元的计算额度。大问题是是否还有另一个...</li><li><a href="https://github.com/predibase/lorax">GitHub - predibase/lorax: 可扩展至数千个微调 LLMs 的 Multi-LoRA 推理服务器</a>: 可扩展至数千个微调 LLMs 的 Multi-LoRA 推理服务器 - predibase/lorax</li><li><a href="https://modal.com/docs/examples">精选示例</a>: 如何运行 LLMs、Stable Diffusion、数据密集型处理、计算机视觉、音频转录等</li>

r tasks on Modal.</li><li><a href="https://predibase.com/">Predibase: The Developers Platform for Fine-tuning and Serving LLMs</a>: Predibase：用于微调和提供 LLM 服务的开发者平台。在托管于私有云的最先进基础设施上，微调和提供任何开源大语言模型服务的最快、最简单的方法。</li><li><a href="https://www.anyscale.com/blog/continuous-batching-llm-inference">Achieve 23x LLM Inference Throughput &amp; Reduce p50 Latency</a>: 在这篇博客中，我们讨论了 continuous batching，这是一种关键的系统级优化，可以提高 LLM 在负载下的吞吐量并降低延迟。</li><li><a href="https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/benchmarks/inf2/inf2-performance.html#decoder-models">Inf2 Inference Performance &#8212; AWS Neuron Documentation</a>: 未找到描述</li><li><a href="https://github.com/replicate/cog-vllm">GitHub - replicate/cog-vllm: Inference LLM on replicate with vllm</a>: GitHub - replicate/cog-vllm：在 replicate 上使用 vllm 推理 LLM。通过在 GitHub 上创建账号来为 replicate/cog-vllm 的开发做出贡献。</li><li><a href="https://ai-infrastructure.org/the-state-of-ai-infrastructure-at-scale-2024/">The State of AI Infrastructure at Scale 2024</a>: 财富 1000 强公司如何应对其基础设施上日益增长的 AI 需求？他们能否足够快地部署 Gen AI，同时又对其进行严格控制以交付...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/)** (1 messages): 

rumbleftw: 有人有使用 prompt template 微调 instruct 模型的经验吗？
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1247652004562735165)** (5 messages): 

- **cpu_ram_efficient_loading 揭秘**: 一位成员澄清说，`cpu_ram_efficient_loading` 允许在 GPU 之间分片加载模型部分，而不是在每个 GPU 上加载整个模型。当设置为 `false` 时，第一个 worker 持有所有模型权重，而其他 worker 持有骨架权重并利用 FSDP 动态调度所需的权重。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1247392429380862043)** (2 messages): 

- **出现多模态微调查询**: 一位成员询问 **Axolotl** 是否支持**多模态微调**，特别是关于 *"IDEFICS 2"*。该查询后没有收到回复。

- **分享了没有上下文的帮助请求**: 一位用户寻求帮助并分享了一个 [Discord 链接](https://discord.com/channels/1238365980128706560/1247260012741525549)。然而，在这次互动中没有提供进一步的细节或回复。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1247629487152169101)** (19 messages🔥): 

- **CUDA 书籍推荐引起关注**: 一位成员询问推荐哪本 CUDA 书籍，并提到一个不明确的链接，这引发了验证挑战。另一位成员确认该 CUDA 书籍与 Charles 在 2024 年 6 月 4 日的演讲中提到的那本相同。
- **快速工作获得积分**: 在快速运行了一个 hello world 示例并解决安装问题后，一位成员询问他们是否因为不到五分钟的工作而获得了 500 美元的积分。Charles 确认了该积分，并指出可能需要长达一周的时间才能发放。
- **在 Modal 上运行 Embedding 模型是合理的**: 成员们讨论了在 Modal 上托管 sentence transformer embedding 模型，并参考了[在 Wikipedia 数据上运行小型 bert 风格 embedding 模型的示例](https://modal.com/blog/embedding-wikipedia)。Charles 肯定了这是合理的，并建议使用 SBert 库。
- **处理 Modal 的 Hello World 示例中的错误**: 一位成员在运行 Modal 的 hello world 示例时寻求关于 `Missing option '--i'` 错误的帮助。在分享代码后，另一位成员发现 `app.local_entrypoint()` 前缺少一个 '@'，从而解决了该错误。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://a.co/d/1PbUJK7">无标题</a>: 未找到描述</li><li><a href="https://modal.com/blog/embedding-wikipedia">在 15 分钟内完成英文维基百科的 Embedding</a>: 利用 Modal 的并行批处理作业和内部存储功能，快速为数十亿个 token 生成 embedding。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1247299495797063720)** (22 条消息🔥): 

- **与 Jackie 一起赶上额度列车**：一位用户对 Jackie 协助创建账号表示感谢。*"前几天在 Jackie 的帮助下，我成功创建了账号。👍"*
- **错过表格，仍希望获得额度**：有几处讨论关于错过提交表格的截止日期，但仍希望获得额度。一位用户提到，*"我没能填写表格，是否可以在某些供应商那里考虑给我额度？"*
- **Braintrust 额度已为用户发放**：来自 Braintrust 的 Ankur 确认额度已发放，共有 1366 人设置了账号并升级到了 Enterprise 层级。*"额度金额... 向上取整为 3 个月，因为它将立即开始。"*
- **协助追踪额度**：用户们正在分享收到的额度更新，并寻求汇总列表。一位热心成员提供了一份来自各供应商的预期额度详细列表。
- **提交表格的新机会**：Dan 确认即使之前错过了表格，仍有可能获得部分额度。*"我会尝试为你争取额度... 但我们无法保证在所有平台上都能为你争取到所有额度。"*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[braintrust](https://discord.com/channels/1238365980128706560/1245407617031999581/1247279775798792313)** (17 条消息🔥): 

- **检查 'Upgrade' 按钮以确认额度**：一位成员询问如何验证额度是否到账，明确了如果没有 "Upgrade" 按钮则表示额度已激活。*"如果你在右上角看到 'Upgrade' 按钮，说明额度尚未生效。"*
- **混淆了不同的 Braintrust 平台**：一些用户错误地访问了错误的 Braintrust 网站，认为额度没有到账。**澄清如下**：*"那是错误的 Braintrust。我们的网址是 braintrust.dev"*
- **Braintrust 使用指南**：为用户提供了入门指导，包括有用资源的链接。成员们被引导至 [Braintrust 欢迎页面](https://www.braintrust.dev/docs/welcome/start) 和 [Evals 指南](https://www.braintrust.dev/docs/guides/evals) 以获取更多信息。
- **重新进入引导状态的附加 URL**：对于想要重新进入引导（onboarding）状态的用户，建议在 URL 后添加 `?onboarding=true`。*"你可以在 URL 中添加 `?onboarding=true`。"*
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.braintrust.dev/docs/cookbook/Text2SQL-Data">LLM Eval For Text2SQL</a>：Braintrust 是用于构建 AI 产品的企业级技术栈。</li><li><a href="https://www.braintrust.dev/docs/welcome/start">Braintrust</a>：Braintrust 是用于构建 AI 产品的企业级技术栈。</li><li><a href="https://www.braintrust.dev/docs/guides/evals">Braintrust</a>：Braintrust 是用于构建 AI 产品的企业级技术栈。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1247438137332990052)** (3 条消息): 

- **频道内的日常问候**：成员们分享了日常问候，一位成员说了 "*Salut!*"，另一位成员则在瑞典马尔默打卡。 
- **表达日程繁忙**：一位成员开玩笑说，由于日程太紧（包括参加 MLOps 课程和工作在内的多项活动），只能在课程结束后再见面。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[announcements](https://discord.com/channels/1238365980128706560/1245460787196068030/1247591929298812989)** (1 条消息): 

- **成员收到 Jarvis Labs 额度**：*感谢 <@657253582088699918> 为大家提供 Jarvis Labs 的额度*。[公告](https://discord.com/channels/1238365980128706560/1241117895740625099/1247591436929466439)已发布，表示这是一次面向全社区的额度发放。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1247591109413048371)** (6 messages): 

- **工程反馈循环 (Engineering Feedback Loop)**：一个用户提到他们会将反馈添加到与工程团队讨论的事项中。*"这很有道理。我会把它加入到与工程团队讨论的事项中，感谢反馈。"*

- **积分已发放 (Credits Distributed)**：Predibase 宣布积分（Credits）已经发放。如果用户没有收到，请发送邮件至 [support@predibase.com](mailto:support@predibase.com)。

- **积分缺失问题 (Missing Credits Issue)**：一些用户报告没有看到发放的积分金额。一位用户指出只有注册时的 25 个积分，并将联系支持部门。

- **L3 70B 微调关注点 (Fine-Tuning L3 70B Concerns)**：一位用户报告了 L3 70B 基础模型在 ```<|end_of_text|>``` Token 之后继续生成文本的问题。*"在 Prompt 标签页使用 Adapter 时，当基础模型生成 <|end_of_text|> 时，它不会停止生成，而是让模型继续编写无意义的内容。"*

- **微调模型的生成问题 (Generation Issues with Fine-Tuned Models)**：另一个提出的问题是关于模型在生成过程中重复句点或句点空格。*"如果有什么办法可以降低微调后的基础模型在生成结束时开始重复句点或句点空格的频率，那也会很有帮助。"*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[career-questions-and-stories](https://discord.com/channels/1238365980128706560/1245816565073576037/1247437655902392383)** (7 messages): 

- **自由职业历程见解 (Freelancing journey insight)**：一位资深的行业专业人士分享了他们从学术界转向自由职业的过程，强调了在支持性社区中经历的挣扎和最终获得的成功。他们强调了社区对于支持、人脉和灵感的重要性。

- **一年内学习 Generative AI (Learning Generative AI in one year)**：一位用户详细介绍了他们在过去一年中的快速学习历程，从零编程基础起步，发展到部署复杂的 AI 模型并编写 Python 和 JavaScript 代码。他们鼓励其他人开始学习，并强调社区和辅助方法是他们进步的催化剂。

- **从工业界转向 ML (From industry to ML)**：一位机械工程专业人士在接触 fast.ai 课程后，从工业研发和医疗器械咨询转向了 Machine Learning。此后，他们在多家初创公司应用了 ML，并强调了他们对这一转型和持续学习的喜爱。

- **对 AI 学习旅程的兴奋 (Excitement for AI learning journey)**：一位拥有 17 年经验的软件开发人员表达了他们作为课程的一部分，对学习 Large Language Models 和 Machine Learning 的新兴奋感。他们正在寻求关于学习模型 Fine-tuning 的资源建议，并有关于改进 Fintech 的项目想法。

- **分享游戏成就 (Gaming achievement shared)**：一位用户幽默地提到通过在游戏中玩随机阵营达到了钻石段位，为关于职业经历和学习路径的讨论增添了轻松的氛围。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1247410269244624937)** (5 messages): 

- **分享优化 LLM 指南 (Optimizing LLMs Guide Shared)**：来自 OpenAI 的一名团队成员分享了一份关于[优化 LLM 准确性](https://platform.openai.com/docs/guides/optimizing-llm-accuracy)的新指南。该指南涵盖了 Prompt Engineering、RAG 和 Fine-tuning，以及如何为业务和技术利益相关者确定什么是生产环境的“足够好”。
- **关于最大化 LLM 性能的 YouTube 演讲 (YouTube Talk on Maximizing LLM Performance)**：另一位成员建议观看名为“最大化 LLM 性能的技术综述”的 [DevDay YouTube 演讲](https://www.youtube.com/watch?v=ahnGLM-RC1Y)。该演讲对旨在释放语言模型全部潜力的技术进行了全面的综述。

**提到的链接**：<a href="https://www.youtube.com/watch?v=ahnGLM-RC1Y">A Survey of Techniques for Maximizing LLM Performance</a>：加入我们，了解旨在释放语言模型（LLM）全部潜力的技术综述。探索诸如 Fine-tuning 等策略...

  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1247410323565183029)** (2 messages): 

- **对黄仁勋（Jen-Hsun Huang）数字人技术的关注 (Curiosity about Jen-Hsun Huang's tech for digital humans)**：一位成员询问：*"有人知道黄仁勋（Jen-Hsun Huang）用来生成数字人的技术是什么吗？"* 该查询在聊天中尚未得到解答。
  
- **寻求高效的 LLM 训练方法 (Seeking efficient LLM training methods)**：一位成员询问了在多个 GPU 上高效训练 LLM 的最佳方法，并质疑 FSDP (Fully Sharded Data Parallel) 是否是最快的选择。他们表达了对即使使用 FSDP 也会涉及复杂性的担忧，并强调 *"时间是这里的主要问题。"*
  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1247351502960332811)** (1 条消息): 

- **在 Triton 中使用 Shared Memory 对 Tensor 进行索引**：一位成员指出，直接**对 Shared Memory 中的 Tensor 进行索引**通常是不可行的。他们提到了两种方法：一次加载一行/一列，或者在最新的 Triton 或 Torch nightly 版本中使用 `tl.split`，但这仅适用于 **2 的幂（power of 2）**。
- **Triton 基于块的设计使细粒度索引变得复杂**：会议强调 **Triton 主要是基于块（block-based）的**，这使得细粒度索引变得很别扭。这解释了为什么像 `tl.split` 这样的方法虽然必要，但在某些场景下仍然不是最优的。
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1247332044854526012)** (3 条消息): 

- **调试 NHWC Tensor 归一化**：一位成员正在寻求关于 **NHWC Tensor** 归一化的帮助，目前正在调试 **ATen 内部实现**中的问题。他们向社区征求建议，询问其实现可能存在的问题。
- **在 MPS Profiler 中使用 Metal Traces**：讨论涉及了如何使用 **torch.mps.profiler** 打开 Metal Traces。一位成员分享道，他们发现要么**直接打开 XCode Instruments 并点击录制**，要么在征求社区意见后**使用 Python 创建一个项目**并运行性能方案（performance scheme）。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/docs/stable/_modules/torch/mps/profiler.html#start);">torch.mps.profiler &mdash; PyTorch 2.3 文档</a>：未找到描述</li><li><a href="https://pytorch.org/docs/stable/_modules/torch/mps/profiler.html#star">torch.mps.profiler &mdash; PyTorch 2.3 文档</a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1247341251951460437)** (7 条消息): 

- **考虑转向 CUDA/Triton 编程**：一位用户表达了从通用软件开发转向 CUDA/Triton 编程的担忧，指出这种巨大的转变以及潜在的就业市场局限性。他们寻求关于这一专业领域是否会有持久需求的建议。

- **关于在 30 多岁学习 CUDA 的宽慰**：作为回应，有人指出他们也是在 30 多岁时学习的 CUDA，并暗示对更快模型的需求日益增长，间接表明了该领域的职业保障。

- **从 CUDA 还是 Triton 开始**：建议用户先阅读 PMPP 书籍的前 6 章以学习 CUDA 基础知识，然后转向 Triton 以更轻松地创建性能良好的 Kernel。此外，他们被鼓励将问题移至另一个频道以获得更多关注。
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1247325756405841942)** (11 条消息🔥): 

- **下载 CUDA 用于 Nvidia 编译**：一位用户建议从[官方网站](https://developer.nvidia.com/cuda-downloads)下载 CUDA，并提到它是一个 .exe 文件。他们还指出需要将编译器 nvcc 添加到系统 PATH 中。
- **torch.compile 用法详解**：一位用户询问在某个函数及其子函数调用上使用 `torch.compile` 的情况。另一位成员确认，编译父函数也会编译子函数，并建议使用 `torch.compile(func, fullgraph=true)` 将整个计算图融合为一个 Kernel。
- **使用细粒度 API 调整 torch.compile**：一位用户分享了一个[文档链接](https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html)，用于对 `torch.compile` 进行细粒度控制。这包括像 `torch.compiler.disable` 这样的 API，用于管理模型中应跳过编译的部分。
- **对所提供文档的感谢**：用户对文档链接表示感谢，并提到这对于查找 `torch.compile` 的参数很有帮助。他们正准备训练一个模型，旨在节省计算资源需求。

**提到的链接**：<a href="https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html">用于细粒度追踪的 TorchDynamo API &mdash; PyTorch 2.3 文档</a>：未找到描述

  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1247352023310143619)** (15 messages🔥): 

- **Apple Metal kernel 加入 PyTorch**：一名成员分享了一个令人兴奋的更新，[Apple Metal kernel 即将进入 PyTorch AO](https://github.com/pytorch/ao/pull/311)，支持 int4 量化 metal kernels 并提供详细的 benchmarks。
- **稀疏化基准测试提升**：另一名成员添加了 [2:4 稀疏化的简单基准测试](https://github.com/pytorch/ao/pull/303)，报告在 3090 和 4090 等 GPU 上性能提升了 10-23%，且精度损失极小。
- **关于 torch.ao.quantization 弃用的辩论**：一位成员询问 `torch.ao.quantization` 是否最终会被 `torchao` 取代，回复建议弃用目前不是首要任务，但在建立更多功能和性能改进后将会发生。
- **结构化剪枝效率**：一位成员解释说，结构化剪枝在 CPU 和 GPU 上都能提供实际的推理时间收益，但与 2:4 等细粒度稀疏相比，可能会导致精度问题。结构化剪枝的实验性支持可在 [这里](https://github.com/pytorch/ao/tree/main/torchao/sparsity/prototype/pruner) 找到。
- **LLM-Shearing 论文见解**：讨论以 LLM-Shearing 论文的见解结束，强调剪枝具有非均匀层配置的模型可能会减小模型大小，但由于不规则的内存访问模式，可能会增加推理时间。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/pull/303">jcaip 提供的稀疏基准测试数据 · Pull Request #303 · pytorch/ao</a>：更新了独立稀疏数据的基准测试脚本。从 segment-anything 切换到 segment-anything-fast。更新了 README，包含 segment-anything 和 BERT 的结果。</li><li><a href="https://github.com/pytorch/ao/pull/311">为线性层添加 int4 分组量化 metal kernels，由 kimishpatel 提交 · Pull Request #311 · pytorch/ao</a>：此 diff 添加了：int4 量化 metal kernels；使用 cmake 构建添加 group size = 32 的测试。TODO：将测试添加到 CI；将自定义 op 集成到 PyTorch（有助于通过 Python 进行测试）；Kerne...</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/sparsity#sparsity-pattern">ao/torchao/sparsity at main · pytorch/ao</a>：用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/sparsity/prototype/pruner">ao/torchao/sparsity/prototype/pruner at main · pytorch/ao</a>：用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1247331215099564082)** (4 messages): 

- **社区对 Discord 服务器的偏好**：一位成员询问关于其他 Discord 服务器的建议，并提到知道 **ML Ops**。另一位成员幽默地回应，承认他们主要每天查看附属服务器，但计划很快创建一个新的。

- **在 'Goomblog' 上探索状态空间对偶性**：分享了一个指向 [the goomblog](https://goombalab.github.io/blog/) 的链接，其中包含 **State Space Duality (Mamba-2)** 系列的多个部分。这包括关于 Mamba-2 背后的系统、算法、理论和模型的深入文章，每篇都讨论了项目的不同方面。

**提到的链接**：<a href="https://goombalab.github.io/blog/"> blog | Goomba Lab </a>：未找到描述

  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1247405203926286336)** (1 messages): 

- **芝加哥 CUDA 爱好者集结**：一位成员表达了在芝加哥林肯公园找到另一位 CUDA 学习同伴的兴奋。他们热衷于安排见面，并表示 **“100% 赞成”**。

  

---


### **CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1247429955743383554)** (3 messages): 

- **Fancytrevor 觉得矩阵表示令人印象深刻**：Fancytrevor 评论说可视化“看起来像整个矩阵”，但也承认它可能代表整体操作的一部分。
- **呼吁演示窗口移动**：Fancytrevor 建议，如果是 CTA，则演示“在到达 C[3,3] 后沿 K 移动窗口”，暗示需要更清晰地说明窗口操作。
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1247263810679603341)** (488 条消息🔥🔥🔥): 

- **CUDA 编译器争议**：用户对与 `__syncthreads` 相关的 **CUDA 代码编译错误**感到困惑，这些错误源于代码位于 `.cpp` 文件而非 `.cu` 文件中。将 `.cpp` 转换为 `.cu` 解决了这些问题，凸显了编译 CUDA 和非 CUDA 代码之间的细微差别。
- **链接器故障：Inline 来救场**：由于多重定义（例如 `cudaCheck`、`deviceProp`）导致的一系列 **链接器错误（linker errors）** 通过将函数和变量设为 `inline` 得到了缓解。成员们讨论了 inline、static 和 extern 声明的细微差别。
- **梯度镜像失误**：导致 **单 GPU 和多 GPU** 运行之间梯度差异的一个重大 Bug 被追溯到 **PyTorch 到 C 桥接后梯度未清零**。这一差异已得到解决，确保了不同设置下结果的一致性。
- **重构热潮**：**重大的代码重构**将 CUDA kernels 隔离到独立文件中，提高了代码组织结构和 IDE 性能。关于进一步模块化代码以更好地处理 GPT-2 等模型以及潜在的 ZeRO-2 实现的讨论非常普遍。
- **令人费解的 Loss 计算**：PyTorch 中单 GPU 和多 GPU 设置之间 Loss 计算的意外差异被发现是由于 **分布式运行中缺少 Loss 的 reduction**。这与 llm.c 的一致结果形成了对比，突显了 PyTorch 中 DDP 的复杂性。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/proftomyeh/status/1798042265883156651?s=46&t=ROCrCC19RlrPdFqCtEaiGA">来自 Tom Yeh | AI by Hand ✍️ (@ProfTomYeh) 的推文</a>: llm.c 手写版✍️ C 编程 + 手写矩阵乘法。这种组合或许是解释 Transformer 工作原理的最底层方式。特别感谢 @karpathy 鼓励...</li><li><a href="https://en.wikipedia.org/wiki/Makedepend">makedepend - Wikipedia</a>: 未找到描述</li><li><a href="https://x.com/karpathy/status/1013244313327681536?lang=en">来自 Andrej Karpathy (@karpathy) 的推文</a>: 最常见的神经网络错误：1) 你没有先尝试过拟合单个 batch。2) 你忘了为网络切换 train/eval 模式。3) 你在 .backward 之前忘了 .zero_grad() (在 pytorch 中)...</li><li><a href="https://github.com/karpathy/llm.c/pull/536">由 karpathy 提交的 Pull Request #536：移动 encoder, kernel 1/N</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/548/">由 gordicaleksa 提交的 Pull Request #548：修复 zero grads Bug</a>: 我们必须在这里清零梯度，因为在训练循环中，我们先执行 backward，稍后才执行 zero grad。这导致在第一个训练步骤中出现了重复的梯度“沉积”...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.py#L688">llm.c/train_gpt2.py (master 分支) · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 llm.c 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1247298746912346153)** (39 条消息🔥): 

- **PyTorch 中的 uint2-7 仍为占位符**：已确认 `uint2-7` 类型目前在 PyTorch 中仅为“占位符（dummy）”且尚未实现，计划在未来使用。相关 [GitHub 代码](https://github.com/pytorch/pytorch/blob/a4064da8cac7345fdf1ffb1f03262f9b235f37a0/c10/core/ScalarType.h#L28-L31) 链接证实了这一点。

- **自定义 dtype 的字符串转换**：成员们讨论了 `TrinaryTensor` 在打印时从 unit2 自动转换为 uint8 的挑战。建议的解决方案包括重写 `__repr__` 方法，以便正确地将值显示为 -1, 0, 1，而不是 0, 1, 2。

- **关于二值/三值矩阵乘法的有趣论文**：一位成员分享了一篇关于二值/三值矩阵乘法的[有趣论文](https://arxiv.org/pdf/2205.09120)。提供的其他资源包括 [Cutlass BMMA](https://github.com/NVIDIA/cutlass/blob/ddd8f9cf4126dbd73b451d7cdd17aab7242fda53/include/cutlass/arch/mma_sm80.h#L2119) 的链接和 [NVIDIA 的 CUDA C 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sub-byte-operations)。

- **调试 PyTorch 的 'cache_size_limit reached' 问题**：讨论了调试 `torch._dynamo.exc.Unsupported: cache_size_limit reached` 错误的高级策略。解决方案包括将参数标记为 dynamic，检查图中断（graph breaks），以及可能通过增加缓存大小来避免重新编译。

- **PyTorch FakeTensor 问题解决**：分享了一个针对 FakeTensor 问题的修复方案，并附带了 [GitHub pull request](https://github.com/pytorch/pytorch/pull/127927) 链接。该方案旨在解决 functional tensors 中张量元数据分发的问题。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/pull/127927">FunctionalTensor: 由 bdhirsh 直接将元数据分发到内部张量 · Pull Request #127927 · pytorch/pytorch</a>: 修复了 #127374。链接复现中的错误是：AssertionError: Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode with 'allow_non_fake_inputs'. Found in aten.sym_st...</li><li><a href="https://github.com/pytorch/pytorch/blob/a4064da8cac7345fdf1ffb1f03262f9b235f37a0/c10/core/ScalarType.h#L28-L31">pytorch/c10/core/ScalarType.h · pytorch/pytorch</a>: Python 中具有强 GPU 加速的张量和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/NVIDIA/cutlass/blob/ddd8f9cf4126dbd73b451d7cdd17aab7242fda53/include/cutlass/arch/mma_sm80.h#L2119>">cutlass/include/cutlass/arch/mma_sm80.h · NVIDIA/cutlass</a>: 用于线性代数子程序的 CUDA 模板。欢迎在 GitHub 上为 NVIDIA/cutlass 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1247326741815627836)** (22 条消息🔥): 

- **ARM 性能：引发担忧并分享资源**：一位成员表达了对购买 X Elite 的担忧，并寻求有关 Hexagon SDK（针对 FPU）和 Adreno SDK（针对 GPU）支持的信息。回复澄清说，虽然 ExecuTorch 支持 Hexagon，但 Adreno 与 CUDA 不兼容，且 PyTorch 缺乏强大的 Vulkan/OpenGL 后端。

- **深入探讨 ARM 的能力**：分享了相关资源，包括一份[关于 ARM 性能的 PDF](https://www.dcs.warwick.ac.uk/pmbs/pmbs22/PMBS/talk10.pdf)和一篇[关于 ARM Scalable Matrix Extension 的博文](https://newsroom.arm.com/blog/scalable-matrix-extension)。

- **讨论 Adreno 和 SNPE**：展开了关于为 Adreno 使用 SNPE 的讨论，强调了其基于 OpenCL 的特性以及使模型兼容所涉及的复杂性。一位用户提到，某些系统会阻止直接访问 OpenCL，这增加了另一层难度。

- **澄清 QNN 与 SNPE**：关于 Qualcomm 的 QNN 是否与 SNPE 不同存在争论，一些人断言这仅仅是重新更名，而另一些人则指出了路线图和团队的差异。文档混乱以及对某些操作的过时支持被指出是持续存在的问题。

- **ARM SME、SVE 和 硬件挑战**：一位成员提到计划实现 SVE2 GEMM，但由于硬件不可用而推迟，打算使用 llvm-mca 进行指标评估。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction">Arm Scalable Matrix Extension (SME) Introduction </a>: 该系列博文介绍了 Arm Scalable Matrix Extension (SME)，包括 SVE 和 SVE2。</li><li><a href="https://developer.arm.com/documentation/109246/0100/matmul-int8--8-bit-integer-to-32-bit-integer-matrix-by-matrix-multiplication/Overview-of-the-matmul-int8-algorithm?lang=en">Documentation – Arm Developer</a>: 未找到描述</li><li><a href="https://newsroom.arm.com/blog/scalable-matrix-extension">Introducing Armv9 Scalable Matrix Extension for AI Innovation on the Arm CPU</a>: 新的 Armv9 架构特性为基于 AI 和 ML 的应用（包括生成式 AI）提供了显著的性能提升。</li><li><a href="https://github.com/google/gemmlowp">GitHub - google/gemmlowp: Low-precision matrix multiplication</a>: 低精度矩阵乘法。通过在 GitHub 上创建账户为 google/gemmlowp 的开发做出贡献。</li><li><a href="https://github.com/kexinzhao/farm">GitHub - kexinzhao/farm: Fast matrix multiplication library for ARM CPUs</a>: 适用于 ARM CPU 的快速矩阵乘法库。通过在 GitHub 上创建账户为 kexinzhao/farm 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1247270329529663600)** (427 条消息🔥🔥🔥): 

- **RoPE `llama-3-8b` 扩展至 64k tokens 需要谨慎**：一位用户尝试将 `llama-3-8b` 扩展到处理 **64k tokens**，但遇到了严重的 **VRAM 问题**，包括导致 80GB H100 出现 OutOfMemoryError 的峰值。他们分享了详细结果并讨论了可能的解决方案，包括调整配置和梯度检查点 (gradient checkpointing)。
- **通过 Unsloth 进行高性价比优化**：Unsloth 团队参与了关于其框架在 **Llama 3 和 Mistral 等模型微调**中应用的讨论，强调了成本和显存效率的提升。他们还澄清了有关 Unsloth 在序列分类和持续预训练 (continuous pretraining) 中应用的问题，并指向了[他们关于持续预训练的最新博客文章](https://unsloth.ai/blog/contpretraining)以及 [GitHub 上的设置指南](https://github.com/unslothai/unsloth)。
- **Phi-3 模型评估反响不一**：一位用户考虑再次尝试将 Phi-3 用于数据提取任务，这引发了关于其他人报告的**性能不一致**的辩论。一些人取得了巨大成功，而另一些人则指出结果不尽如人意，凸显了模型的差异性。
- **潜在的 VRAM 优化见解**：通过实验发现，模型的 **RoPE 扩展**在初始阶段可能会出现明显的 VRAM 峰值，随后会趋于平稳。这一发现表明有潜力更有效地优化和管理内存，甚至可能进一步扩展容量。
- **社区参与和资源**：成员们分享了有用的资源链接，例如 [Daniel Han 的直播活动](https://meet.google.com/sxp-ekzv-osb)录像，并深入探讨了微调实践，进一步提升了模型训练和优化方面的集体知识和故障排除方法。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/wWJCq9PU?event=1244218238288531457">加入 Aleksa Gordić - The AI Epiphany Discord 服务器！</a>：机器学习。 | 7662 名成员</li><li><a href="https://huggingface.co/fimbulvntr/lewd-stories/tree/main">fimbulvntr/lewd-stories at main</a>：未找到描述</li><li><a href="https://wandb.ai/unnamed_org/text-novel-completion-ropes/reports/Progressively-RoPE-llama-3-70b-bnb-4bit--Vmlldzo4MjAzMDI1">Weights & Biases</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://github.com/FasterDecoding/Medusa">GitHub - FasterDecoding/Medusa: Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads</a>：Medusa：通过多个解码头加速 LLM 生成的简单框架 - FasterDecoding/Medusa</li><li><a href="https://www.youtube.com/watch?v=v_q2JTIqE20">GPU 优化研讨会 (由 @ChipHuyen 主持)</a>：00:30 研讨会概览 03:51 GPU 优化速成课程 (Mark Saroufim, Meta) 39:18 NVIDIA GPU 上的高性能 LLM 推理服务 (Sharan Chetlur, NVIDIA)...</li><li><a href="https://api.wandb.ai/links/unnamed_org/yyl8ymr1">渐进式 RoPE llama-3-70b-bnb-4bit</a>：在 H100 SXM (runpod) 上运行。数据集包含长度等于 (tokens) 的样本：- 4k - 8k - 16k - 32k - 48k - 64k (崩溃)。运行名称代表所使用的上下文长度。</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://x.com/UnslothAI/status/1798088790919332013">来自 Unsloth AI (@UnslothAI) 的推文</a>：Unsloth 现在允许您使用 QLoRA 进行持续预训练，速度比 Hugging Face+FA2 快 2 倍，且显存占用减少 50%。持续预训练允许模型在新的领域数据上进行训练。阅读我们的博客：h...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1247651391212621824)** (1 条消息): 

- **Unsloth 持续预训练 LLM 速度更快**：[Unsloth 的新版本](https://github.com/unslothai/unsloth/releases/tag/June-2024) 允许你以比 Hugging Face + Flash Attention 2 QLoRA 快 **2 倍**的速度进行 LLM 持续预训练，同时减少 **50% 的 VRAM** 占用。该版本在他们的 [博客文章](https://unsloth.ai/blog/contpretraining) 中有详细介绍。

- **用于学习和文本补全的免费 Notebook**：Unsloth 提供了免费的 Colab Notebook 以供上手体验。可以访问适用于 Mistral v0.3 7b 的 [持续预训练 Notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing) 和 [文本补全 Notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)。

- **关于预训练的技术见解**：关键见解包括微调 Embeddings、将 Embeddings 卸载（offloading）到磁盘，以及为 Embeddings 使用不同的学习率以稳定训练。他们还建议使用 Rank stabilized LoRA 来提高性能。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行 LLM 持续预训练</a>：通过使用 Unsloth 对 Llama 3, Phi-3 和 Mistral 进行持续预训练，让模型学习一种新语言。</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing)">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1247284318020046943)** (18 条消息🔥): 

- **分享有趣的越狱提示词 (Jailbreak Prompt)**：一位用户分享了来自 [Hugging Face](https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts) 的越狱提示词，并建议尝试一下以寻找乐趣。该提示词涉及在两个实体 Tom 和 Jerry 之间创建一个故事，并详细说明了一项技术指令。

- **多 GPU 支持即将推出**：成员们讨论了 Unsloth AI 支持多 GPU 的可能性。一位用户持怀疑态度，但另一位用户确认道：“我们很快就会推出多 GPU 支持，但像往常一样，这需要一些时间”。

- **多 GPU 性能见解**：有人指出，Unsloth AI 在单张 A100 GPU 上的速度比使用两张 GPU 更快。一位正在研究多 GPU 设置的用户对未来的训练扩展性表示担忧，并询问了关于使用 "device = 'cuda'" 的问题。

- **与 Ray 集成进行微调**：有人询问当多 GPU 支持可用时，未来是否可以通过 Ray 在多个节点上进行微调。这展示了用户对优化跨多 GPU 的高性能模型训练的兴趣。

**提到的链接**：<a href="https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts?">rubend18/ChatGPT-Jailbreak-Prompts · Hugging Face 数据集</a>：未找到描述

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1247265389570818240)** (141 messages🔥🔥): 

- **Unsloth 本地设置指南**：关于在本地设置 Unsloth 的疑问，通过引导用户安装 Jupyter Notebook 并参考 Unsloth GitHub 页面上的 [readme](https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions) 得到了解决。此外，还建议使用 Docker 镜像以简化操作，链接指向 [Jiar/jupyter4unsloth repository](https://github.com/Jiar/jupyter4unsloth)。
  
- **自定义训练循环（Custom Train Loop）的挑战**：成员们讨论了自定义训练循环的问题，其中一名用户遇到了 `RuntimeError`。调试尝试包括切换模型源，但确切原因仍不明确。
  
- **关于 8-bit 量化微调的讨论**：用户询问了在 phi-3-medium-4k 等模型上进行 8-bit 量化（Quantization）微调的可行性。社区澄清目前尚不支持此功能，正在进行进一步研究。
  
- **使用 Llama 3 进行情感分析的项目可行性**：成员们辩论了在较短的项目周期内使用 Llama 3 进行情感分析的可行性。建议改用像 BERT 这样的 Embeddings，并考虑通过 COT (Chain of Thought) 和引导生成（guided generation）来增强能力。
  
- **LoRA Rank 计算与 Alpha 参数讨论**：引用启发式方法和实验观察，对 LoRA rank 的计算和 alpha 参数的设置进行了详细讨论。引用了 [Lightning AI 的文章](https://lightning.ai/pages/community/lora-insights/) 以获取关于最佳 LoRA 设置的进一步见解。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/Jiar/jupyter4unsloth">GitHub - Jiar/jupyter4unsloth: Jupyter for Unsloth</a>：用于 Unsloth 的 Jupyter。通过在 GitHub 上创建账号为 Jiar/jupyter4unsloth 的开发做出贡献。</li><li><a href="https://lightning.ai/pages/community/lora-insights/">使用 LoRA 和 QLoRA 微调 LLM：来自数百次实验的见解 - Lightning AI</a>：LoRA 是用于训练自定义 LLM 的最广泛使用的参数高效微调技术之一。从使用 QLoRA 节省显存到选择最佳 LoRA 设置，本文提供了……</li><li><a href="https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions">GitHub - unslothai/unsloth: 微调 Llama 3, Mistral, Phi &amp; Gemma LLMs，速度提升 2-5 倍，显存占用减少 80%</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/27fa021a7bb959a53667dd4e7cdb9598c207aa0d/unsloth/models/llama.py#L1196">unsloth/unsloth/models/llama.py (位于 27fa021a7bb959a53667dd4e7cdb9598c207aa0d) · unslothai/unsloth</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/27fa021a7bb959a53667dd4e7cdb9598c207aa0d/unsloth/models/llama.py#L1175C9-L1175C36)">unsloth/unsloth/models/llama.py (位于 27fa021a7bb959a53667dd4e7cdb9598c207aa0d) · unslothai/unsloth</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1247468974929608847)** (3 messages): 

- **新成员加入**：新成员 **s3nh1123** 热情地加入了社区，表达了“很高兴来到这里 💚💚”的喜悦。成员 **theyruinedelise** 迅速表示欢迎，并重申了“很高兴你能来！”的情感。
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1247273615901065248)** (556 messages🔥🔥🔥): 

- **Perplexity 面临学术查询挑战**：一位用户对 Perplexity 的学术搜索结果主要来自 **Wikipedia** 且仅有一个来自 **Britannica** 的来源表示沮丧，质疑其在学术搜索中的有效性。他们提供了搜索结果的链接，称其为需要报告的潜在 Bug。[搜索链接](https://www.perplexity.ai/search/I-want-to-ZoV4zN4LRKa2YcbFG52K.Q)

- **服务器宕机引发沮丧**：许多用户报告了 **Perplexity**、**ChatGPT** 和其他 AI 服务同时出现宕机的问题。这一事件引发了对更大规模基础设施问题的猜测，可能与 **AWS** 或其他通用服务提供商有关。

- **对 Opus 50 限制的担忧**：成员们对 **Opus 50** 限制的降低表示不满，怀念之前的 **Opus 600** 限制。由于 Perplexity 缺乏关于潜在调整或恢复更高限制的沟通，用户中存在一种挫败感。

- **Perplexity 与 ChatGPT 的比较**：用户讨论了 **Perplexity AI Premium** 与 **ChatGPT** 的优缺点，指出 Perplexity 具有更出色的网页搜索能力和更广泛的模型选择。然而，对于这两个平台的订阅限制和使用场景，大家的看法不一。

- **技术与演示协助**：成员们为有关 AI 的学校演示提供了建议，推荐概述优势和风险，并使用 Perplexity 和 ChatGPT 等 AI 工具来准备内容。分享了一些有用的链接和视频，以帮助理解 AI 的技术概念。[YouTube Video](https://youtu.be/wjZofJX0v4M)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://status.openai.com/">OpenAI Status</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=OkJnY8bpXAM&t=744s&pp=ygUaaGFja2VycyBzb21lb3JkaW5hcnlnYW1lcnM%3D">The Feds Just Launched The Largest Attack On Hackers...</a>：大家好，我是 Mutahar！这次我们来看看似乎是“终局行动”（Operation Endgame），全球所有的警察机构都在发起……</li><li><a href="https://tenor.com/view/server-is-fine-burn-fire-spongebob-blow-gif-14178373">Server Is Fine Burn GIF - Server Is Fine Burn Fire - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/terminator-smiling-arnold-schwarzenegger-gif-16197002">Terminator Smiling GIF - Terminator Smiling Arnold Schwarzenegger - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/terminator-terminator-robot-looking-flex-cool-robot-gif-978532213316794273">Terminator Terminator Robot GIF - Terminator Terminator Robot Looking - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/down-town-la-bomb-dtlablowup-gif-21604415">Down Town La Bomb Dtlablowup GIF - Down Town La Bomb Dtlablowup - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>：未找到描述</li><li><a href="https://tenor.com/view/kristanna-loken-tx-melt-down-terminator-rise-of-the-machines-gif-24408281">Kristanna Loken Tx GIF - Kristanna Loken TX Melt Down - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/skinet-skinet-remember-maquinas-revolucion-gif-7115195292076587720">Skinet Skinet Remember GIF - Skinet Skinet remember Maquinas - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/terminator-gif-21649154">Terminator GIF - Terminator - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://msty.app/">Msty - Running LLMs made simple and easy.</a>：Msty 让你以最简单的方式使用本地和在线 LLM。其功能强大的聊天界面使 LLM 的使用变得容易。运行 Ollama 模型，如 Mixtral, Llama2, Qwen 或在线模型……</li><li><a href="https://tenor.com/view/robot-tech-skynet-ai-arnold-gif-13186419">Robot Tech GIF - Robot Tech Skynet - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=OFS90-FX6pg&t=612s">ChatGPT: 30 Year History | How AI Learned to Talk</a>：这段视频探索了 AI 语言模型的旅程，从它们简陋的起步到 OpenAI 的 GPT 模型的发展。我们的旅程穿过了……</li><li><a href="https://downforeveryoneorjustme.com/perplexity">Perplexity down? Check here and read user reports. - DownFor</a>：Perplexity 无法加载？或者报错？检查实时状态，看看其他 Perplexity 用户在报告什么。</li><li><a href="https://www.cloudflarestatus.com/">Cloudflare Status</a>：未找到描述</li><li><a href="https://youtu.be/wjZofJX0v4M?feature=shared">But what is a GPT?  Visual intro to transformers | Chapter 5, Deep Learning</a>：未找到描述</li><li><a href="https://tenor.com/view/larry-david-seinfeld-pretty-good-enthusiasm-curb-gif-3938316">Larry David Pretty Good GIF - Larry David Seinfeld Pretty Good - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://poe.com/login">Poe - Fast, Helpful AI Chat</a>：未找到描述</li><li><a href="https://copilot.microsoft.com/">Microsoft Copilot: 你的日常 AI 助手</a>：Microsoft Copilot 利用 AI 的强大功能来提高工作效率、释放创造力，并通过简单的聊天体验帮助你更好地理解信息。</li><li><a href="https://downdetector.com/">Downdetector</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1247267604784087111)** (11 messages🔥): 

- **最新进展链接分享**: 一名成员分享了一个关于最新进展的 Perplexity AI 搜索[链接](https://www.perplexity.ai/search/Les-deniers-dveloppent-m.hwMU.EQTCIy.9YeH5xoQ)。
- **AMD 最近新闻链接**: 一名成员发布了一个涉及 AMD 最近动态的新闻[链接](https://www.perplexity.ai/search/AMD-a-rcemment-kVmULrxBT7yTJm0eKCmkvA)。
- **Discord 搜索与真相链接**: 多位成员分享了指向 [Discord](https://www.perplexity.ai/search/discordperplexi-tDIxu6YHQlaSKrREsdB5MQ) 的 Perplexity AI 搜索链接以及一个关于[真相](https://www.perplexity.ai/page/Truth-About-the-yulN9Dp4T2K_EeOonc52PQ)的页面。
- **Perplexity AI 分享主题提醒**: Perplexity AI 提示多位用户确保其主题是“可分享的（Shareable）”，并提供了一个 [Discord 链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)以获取进一步指导。
- **请求重复链接**: 一名用户分享了一个[链接](https://www.perplexity.ai/search/httpsdiscordcom-repeat-this-p6zDFgNJS5Wn4D4YdmeEGg#0)，要求其他人重复链接页面中提到的过程。
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1247654554091389079)** (1 messages): 

- **FineWeb 报告公开**: **FineWeb 技术报告**已发布，揭示了*每一个处理决策*并介绍了 **FineWeb-Edu 数据集**。该报告旨在解释 Llama3 和 GPT-4 + Mixtral 等高性能模型，详见[此处](https://hf.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)。
- **Transformers.js 登陆 Firefox 130**: **Transformers.js** 现在将成为 **Firefox 130** 的一部分，支持*完全私密的设备端 AI*。第一个用例是自动为图像生成替代文本（alt-text），显著提升了无障碍性，更多详情见[此处](https://mzl.la/4aPeBFL)。
- **Nvidia NIM 在 HF Inference Endpoints 上线**: **Nvidia NIM** 现在可以通过 **Hugging Face Inference Endpoints** 部署，首批支持 AWS 和 GCP 上的 Llama 3 8B 和 70B 模型。它承诺高达 9000 tokens/sec 的性能，未来将支持更多模型，详见[此处](https://x.com/_philschmid/status/1797713003778883858)。
- **Gradio Clients 1.0**: **Gradio Clients 1.0 发布活动**已定，倡导从原型到生产级 API 的高性能和可扩展 Gradio 应用。活动详情见[此处](https://discord.com/events/879548962464493619/1245020251611992154)。
- **Sentence Transformers v3 博客**: 一篇新博客展示了如何使用 NVIDIA 的 2023 SEC Filing 数据集为金融 RAG 应用微调 embedding 模型。通过使用合成数据和 **Matryoshka Representation Learning** 等先进技术，性能提升了 7.4% 到 22.55%，详见[此处](https://www.philschmid.de/fine-tune-embedding-model-for-rag)。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/gui_penedo/status/1797173053123916036)">来自 Guilherme Penedo (@gui_penedo) 的推文</a>：我们（终于）发布了 🍷 FineWeb 技术报告！在报告中，我们详细说明并解释了我们做出的每一个处理决策，并介绍了我们最新的数据集：📚 FineWeb-Edu，一个（仅限网页的）子集...</li><li><a href="https://x.com/xenovacom/status/1797285648572821840)">来自 Xenova (@xenovacom) 的推文</a>：Transformers.js 正被添加到 Firefox 130 中！🤯 没错，直接在浏览器中实现完全私有的设备端 AI！🔥 他们正在探索的第一个用例是为图像自动生成 alt-text...</li><li><a href="https://x.com/Gradio/status/1795561025397256498)">来自 Gradio (@Gradio) 的推文</a>：🚀 从 Prototype 到 Production！🙌 欢迎参加 6 月 6 日备受期待的 Gradio Clients 1.0 发布活动。🤩 了解您的 Gradio 应用程序如何表现出高性能...</li><li><a href="https://x.com/kamilakesbi/status/1796537200961785931)">来自 Kamil Akesbi (@kamilakesbi) 的推文</a>：speaker diarization 最大的障碍是什么？数据！使用 🤗 Diarizers，您现在可以生成 synthetic 会议 🗣️ 对话！从 ASR 数据集开始，您可以创建任意数量的数据...</li><li><a href="https://x.com/_philschmid/status/1797713003778883858)">来自 Philipp Schmid (@_philschmid) 的推文</a>：昨天在 COMPUTEX 上，Jensen Huang 宣布在 @huggingface Inference Endpoints 上发布 @nvidia NIM！🚀 NVIDIA NIM 是旨在简化和加速部署的推理服务...</li><li><a href="https://x.com/_philschmid/status/1795804027621404975)">来自 Philipp Schmid (@_philschmid) 的推文</a>：产品更新：@nvidia L4s 现在可在 AWS 上的 @huggingface Inference Endpoints 中使用！每个用户和组织最多可享受 8 个 L4，且比按需使用的 AWS EC2 节省 20%。🤑 - 1x NVIDIA L4...</li><li><a href="https://x.com/abhi1thakur/status/1795477747701104651)">来自 abhishek (@abhi1thakur) 的推文</a>：AutoTrain 刚刚获得了全新的 UI 🚀🚀🚀</li><li><a href="https://x.com/_philschmid/status/1797994961197031703">来自 Philipp Schmid (@_philschmid) 的推文</a>：很高兴分享一篇新博客，介绍如何使用 NVIDIA 的 2023 SEC Filing 数据集，结合最新的研究（如 Matryoshka Representation Learning），为金融 RAG 应用微调 embedding models...</li><li><a href="https://x.com/frimelle/status/1797619351954260214)">来自 Lucie-Aimée Kaffee (@frimelle) 的推文</a>：以社区为中心且棒极了：@huggingface 和 @Wikimedia 🤗 我写了一篇文章，关于我们如何利用来自 @Wikipedia 的多样化数据集推进 ML，以及为什么以及如何在 Hugging Face 上创建更多 Wikimedia 数据集...</li><li><a href="https://x.com/NielsRogge/status/1796213271189438888)">来自 Niels Rogge (@NielsRogge) 的推文</a>：好了，终于带着新视频回到了 @YouTube：在您的自定义数据集上微调 PaliGemma（或 LLaVa、Idefics2...）！我正在 @GoogleColab 的 L4 GPU 上进行微调。我讲解了许多内容，比如...</li><li><a href="https://x.com/abhi1thakur/status/1796210385579639144)">来自 abhishek (@abhi1thakur) 的推文</a>：🚨 新博客：如何使用 AutoTrain 微调自定义 Embedding Models。学习：- 数据格式应该是怎样的 - 如何正确映射列 - 示例数据集 - 自定义 configs - 本地训练 - 训练...</li><li><a href="https://x.com/vanstriendaniel/status/1795875763557904753">来自 Daniel van Strien (@vanstriendaniel) 的推文</a>：您是否需要一个数据集来训练自定义的 Sentence Transformer 模型？我创建了一个 pipeline，使用 LLM 创建 synthetic 数据集，您可以直接将其用于微调/训练 Sentence Transformer...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1247267022434209792)** (418 条消息🔥🔥🔥):

- **LevelBot 的 Activity Tracker v1.0 发布**：Lunarflu 推出了适用于 LevelBot v1.0 的[更新版活动追踪器](https://huggingface.co/posts/lunarflu/239147617114976)，并征求改进建议，鼓励社区提交 PR。初步想法包括追踪更多类型的操作、创建更大的图表，以及整合 Discord 活动和 GitHub 链接。
- **寻求开源聊天助手 UI**：一位成员询问是否有人知道 [HF 的聊天助手 UI](https://huggingface.co/chat/assistants) 的开源版本，以便自定义 Prompt、向量数据库和 RAG 参数。目前缺乏回应，表明可用资源中可能存在空白。
- **微调 HuggingFace 模型**：Robin_01_ 询问了在小型自定义数据集上微调 Mistral 7B Instruct v0.3 等模型以提高指令遵循能力的有效性；社区的回应暗示了这类尝试的实用性。
- **Hugging Face 模型训练指南**：Pranavadvani2003 寻求在设置 Transformers 库以进行开源贡献方面的帮助。建议包括指定正确的 TensorFlow 版本、注意 IDE 的依赖项问题，以及使用替代的加载方法。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://kingnish-sdxl-flash.hf.space'">未找到标题</a>: 未找到描述</li><li><a href="https://kingnish-image-gen-pro.hf.space'">未找到标题</a>: 未找到描述</li><li><a href="https://fluently-fluently-playground.hf.space'">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/KingNish/OpenGPT-4o">OpenGPT 4o - KingNish 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2405.20279">CV-VAE: 一种用于潜空间生成视频模型的兼容视频 VAE</a>: 视频的时空压缩利用了变分自编码器 (VAE) 等网络，在 OpenAI 的 SORA 和许多其他视频生成模型中起着至关重要的作用。例如...</li><li><a href="https://huggingface.co/posts/lunarflu/239147617114976">Hugging Face 上的 @lunarflu：“应大众要求，HF 活动追踪器 v1.0 来了！📊 让我们开始构建吧……”</a>: 未找到描述</li><li><a href="https://youtu.be/MGkByeDm-90?si=14AxwOfkULwC4a2U">“让 Agent 便宜、快速且好用 10 倍？” - LLM 系统评估入门</a>: LLM 系统评估入门 - 构建更好的 Agent。获取关于如何利用 AI 找工作的免费 HubSpot 报告：https://clickhubspot.com/fo2 🔗 链接 - 在 Twitter 上关注我：h...</li><li><a href="https://www.youtube.com/watch?v=fc_NSAu41b0">使用 ChatRTX 创建个性化 AI 聊天机器人</a>: 通过 ChatRTX 技术演示创建个性化聊天机器人。在 TensorRT-LLM 和 Tensor Cores 的加速下，你可以快速从文件中获取定制化信息...</li><li><a href="https://tenor.com/view/big-brain-gif-27108854">Big Brain GIF - Big Brain - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros">Python 如何用零填充 NumPy 数组</a>: 我想知道如何在使用 Python 2.6.6 和 NumPy 1.5.0 的情况下，用零填充一个二维 NumPy 数组。但这些是我的限制条件，因此我不能使用 np.pad。例如，我想填充 a...</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/blog/OzzyGT/outpainting-differential-diffusion">外扩绘图 (Outpainting) II - 微分扩散 (Differential Diffusion)</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/HengJay/snomed-ct-assistant">SNOMED CT Assistant - HengJay 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/cat-eating-eatin-gamer-gunk-gamer-gunk-cat-monkey-cat-gamer-gif-20643451">猫吃东西 GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://groq.com/">Groq 构建全球最快的 AI 推理技术</a>: Groq 的 LPU™ 推理引擎是一个硬件和软件平台，可提供卓越的计算速度、质量和能效。Groq 为 AI 提供大规模的云端和本地解决方案...</li><li><a href="https://www.youtube.com/watch?v=qDXa2rUdia0">长短期记忆 (ChromaDB 无限上下文) - SillyTavern AI</a>: ChromaDB 将你的每条聊天消息存储在数据库中，并且仅在上下文匹配时输出，从而“记住之前的事件”。Colab 链接 - htt...</li><li><a href="https://www.producthunt.com/posts/asknews"> AskNews - 质量至上的新闻 | Product Hunt</a>: AskNews 正在重新构想人类和 LLM 消费新闻的方式。我们提供由 AI 洞察增强的人工编辑内容，以最大限度地减少偏见并建立对时事的透明视图。同时...</li><li><a href="https://x.com/karpathy/status/1797313173449764933">Andrej Karpathy (@karpathy) 的推文</a>: 棒极了且非常有用：FineWeb-Edu 📚👏 高质量 LLM 数据集，将原始 15 万亿个 FineWeb token 过滤为 1.3 万亿个最高（教育）质量的 token，由 Llama 3 70B 评定...</li><li><a href="https://x.com/karpathy/status/">GitHub - FixTweet/FxTwitter 的推文：修复损坏的 Twitter/X 嵌入！</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://github.com/NX-AI/xlstm">GitHub - NX-AI/xlstm: xLSTM 的官方仓库。</a>: xLSTM 的官方仓库。通过在 GitHub 上创建账号来为 NX-AI/xlstm 的开发做出贡献。</li><li><a href="https://discuss.huggingface.co/t/unable-to-load-saved-tokenizer/86631">无法加载保存的 Tokenizer</a>: 我能够成功训练并保存我的 Tokenizer，但随后无法重新加载它。 tokenizer.save(tokenizer_save_path+"tokenizer.json") #有效 newTokenizer = Tokenizer.from_file(tokenizer_save...</li><li><a href="https://www.gradio.app/guides/interface-state">界面状态 (Interface State)</a>: Gradio 分步教程</li>

al</li><li><a href="https://github.com/pranav-bot/transformers">GitHub - pranav-bot/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.</a>: 🤗 Transformers: 为 Pytorch, TensorFlow, 和 JAX 提供最先进的 Machine Learning。 - pranav-bot/transformers</li><li><a href="https://colab.research.google.com/drive/1GSqPM13lS-vTH94dhMP1O2uwrJMbCe0e">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF">bartowski/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp</a>: llama.cpp 的 Python 绑定。通过在 GitHub 上创建账号来为 abetlen/llama-cpp-python 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

asuka_minato: 是的，原始版本在 torch 中并在 colab 上部署。但需要在边缘端进行 edge inference，所以 riir（用 Rust 重写）。
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1247443621419155498)** (4 messages): 

- **语言模型偏好对齐研究**：论文 [Alignedge](https://arxiv.org/abs/2403.07691) 强调了监督微调 (SFT) 对于语言模型偏好对齐的重要性。它介绍了 ORPO 算法，该算法无需额外阶段即可优化偏好对齐，在各种模型规模上展示了实证和理论优势。

- **分享了新的 YouTube 视频**：[此处](https://www.youtube.com/watch?v=PeSLWTZ1Yg8)分享了一个 YouTube 视频链接。

- **用于 ASR/TTS 的德国议会演讲数据集**：一位成员分享了一个包含 610 小时转录音频样本的新数据集，源自德国议会，可在 [Hugging Face](https://huggingface.co/datasets/D4ve-R/bundestag-asr) 获取。该数据集可用于自动语音识别 (ASR) 和文本转语音 (TTS) 训练。

- **可视化 Transformer 工作原理**：分享了一条 [推文](https://x.com/ProfTomYeh/status/1798042265883156651)，展示了 Tom Yeh 教授关于使用 C 编程和手动矩阵乘法解释 Transformer 的项目。此练习旨在帮助人们更好地理解大语言模型 (LLMs) 的工作原理。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.07691">ORPO: Monolithic Preference Optimization without Reference Model</a>: 虽然最近的语言模型偏好对齐算法展示了不错的结果，但监督微调 (SFT) 对于实现成功收敛仍然至关重要。在本文中...</li><li><a href="https://x.com/ProfTomYeh/status/1798042265883156651">来自 Tom Yeh | AI by Hand ✍️ (@ProfTomYeh) 的推文</a>: 手动 llm.c ✍️ C 编程 + 手动矩阵乘法。这种组合可能是解释 Transformer 如何工作的最底层方式。特别感谢 @karpathy 鼓励...</li><li><a href="https://huggingface.co/datasets/D4ve-R/bundestag-asr">D4ve-R/bundestag-asr · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1247324607304699934)** (5 messages): 

- **关于受损预训练数据的新预印本发布**：一位成员分享了与 Microsoft 和 CMU 合作的预印本，讨论了受损数据集在预训练 Diffusion 模型中的影响。研究发现，“预训练中的轻微损坏可以显著提高生成图像的质量、多样性和保真度” ([arXiv:2405.20494](https://arxiv.org/abs/2405.20494))。

- **安全意识视频推荐**：分享了一个名为“[你的电脑被黑的迹象](https://youtu.be/jG56MKen6YM?si=JCjIayR9tCb38zHa)”的 YouTube 视频，旨在帮助用户识别系统被黑的预警信号。该视频强调了维护网络安全的重要性。

- **图像回归框架发布**：一位成员宣布创建了一个使用 PyTorch 和 Transformers 的 **Image Regression**（图像回归）新框架，该框架可无缝集成到 🤗 生态系统中。该框架能够训练模型，将其上传到 HuggingFace hub，并自动生成包含指令和元数据的 Model Cards ([博客文章](https://huggingface.co/blog/tonyassi/image-regression))。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/tonyassi/image-regression">使用图像回归进行销售预测</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2405.20494">预训练数据中的轻微损坏造就更好的 Diffusion 模型</a>：Diffusion 模型 (DMs) 在生成逼真的高质量图像、音频和视频方面展现了卓越的能力。它们从大规模数据集的广泛预训练中获益匪浅...</li><li><a href="https://youtu.be/jG56MKen6YM?si=JCjIayR9tCb38zHa">你的电脑被黑的迹象</a>：识别系统被黑的预警信号对于保护数据和维护网络安全至关重要。在本视频中，我们将讨论...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1247501795501281360)** (1 messages): 

- **Hunyuan DiT pipeline 现已在 diffusers 中可用**：团队刚刚发布了一个补丁，通过 `diffusers` 引入了 **Hunyuan DiT pipeline**，该功能由 Hunyuan 的作者之一 Xingchao Liu 贡献。更多详情请查看 [release notes](https://github.com/huggingface/diffusers/releases)。

**提及的链接**：<a href="https://github.com/huggingface/diffusers/releases">Releases · huggingface/diffusers</a>：🤗 Diffusers：用于 PyTorch 和 FLAX 中图像和音频生成的先进 Diffusion 模型。 - huggingface/diffusers

  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1247275363046916208)** (11 messages🔥): 

- **通过新论文探索视觉语言模型 (VLMs)**：一篇关于 [视觉语言模型 (VLMs)](https://huggingface.co/papers/2405.17247) 的论文概述了将视觉数据与语言模型集成的复杂性和挑战。它讨论了 VLM 的应用，并提供了关于这些模型如何训练和评估的入门指南。

- **在新语言中为 OCR 微调 VLM 的指导**：成员们讨论了为新语言的 OCR 任务微调预训练 VLM 的方法。推荐的方法包括确保 tokenizer 支持目标语言，并使用包含主要字段的数据集：文档图像和提取的文本。

- **使用 600 张图像训练 ResNet-50**：一位成员请求关于使用 600 张收集的图像训练单类别 ResNet-50 模型的指导。讨论涵盖了过拟合 (overfitting) 以及在新数据上预测错误的问题，并探讨了解决这些问题的方法。

- **针对过拟合的正规化方法**：为了对抗过拟合，一位成员被引导至一个专注于正规化 (regularization) 方法的 [Coursera 课程](https://www.coursera.org/learn/deep-neural-network)。该建议引发了关于提供直接帮助与教授基础知识之间平衡的辩论。

- **改进分类模型输出**：有人建议修改分类模型中的输出层，使其包含两个输出（True/False）并使用 softmax 函数，而不是使用单个 sigmoid 输出。这种方法可以简化阈值调整并提高预测的清晰度。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.coursera.org/learn/deep-neural-network)">Coursera | 来自顶尖大学的在线课程。免费加入</a>：来自斯坦福和耶鲁等学校的 7,000 多门课程 - 无需申请。培养数据科学、计算机科学、商业等领域的职业技能。</li><li><a href="https://www.coursera.org/learn/deep-neura">Coursera | 来自顶尖大学的在线课程。免费加入</a>：来自斯坦福和耶鲁等学校的 7,000 多门课程 - 无需申请。培养数据科学、计算机科学、商业等领域的职业技能。</li><li><a href="https://huggingface.co/papers/2405.17247">论文页面 - 视觉语言建模导论</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1247642239933616207)** (1 messages): 

- **调试函数导入问题**：一位用户建议，可能的 **版本不匹配 (version mismatch)** 可能会导致导入特定函数时出现问题。他们建议尝试显式导入所需函数，看是否能解决问题。
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1247275018321264847)** (2 messages): 

- **探索用于视觉运动学习的 Diffusion Policy**：一篇博客文章讨论了 [Action Chunking Transformer](https://radekosmulski.com/how-to-train-your-robot-with-a-transformer/) 及其通过 encoder-decoder 模型生成动作的机制，将学习到的 embedding 转化为预测轨迹。该文章还提到了 "[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137?ref=radekosmulski.com)" 论文，该论文从高斯噪声 (Gaussian noise) 开始生成预测动作。
- **使用 JIT Trace 优化 SDXL 推理**：一位成员询问使用 `jit.trace()` 处理 SDXL 的示例脚本，类似于 [Diffusers 优化指南](https://huggingface.co/docs/diffusers/v0.6.0/en/optimization/fp16#tracing) 中为 `stable-diffusion-v1-4` 提供的脚本。提到的技术包括启用 cuDNN auto-tuner 和使用 `autocast (fp16)`，在 NVIDIA TITAN RTX 上显著将推理速度从 9.50s 提升至 3.21s。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/v0.6.0/en/optimization/fp16#tracing">内存与速度</a>：未找到描述</li><li><a href="https://radekosmulski.com/diving-into-diffusion-policy-with-lerobot/">深入探讨 LeRobot 的 Diffusion Policy</a>：在最近的一篇博客文章中，我们研究了 Action Chunking Transformer (ACT)。ACT 的核心是一个 encoder-decoder transformer，当传入：* 一张图像 * 机器人的当前状态 ...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1247283746403516468)** (136 messages🔥🔥): 

- **辩论真正的 AGI**: 一场关于 AGI 本质的热烈讨论展开了，成员们分享了一些幽默的看法，例如，“*真正的 AGI 会借钱给我*”以及“*真正的 AGI 知道如何正确插入 USB（每一次都能成功）*”。共识倾向于认为真正的 AGI 应该能够持续且自适应地执行复杂的、类人的任务。
- **医学研究 AI 替代方案**: 推荐了 [Elicit](https://elicit.com) 和 Perplexity 等 AI 工具，用于总结研究论文和回答复杂问题。Elicit 的功能因能高效总结论文和综合研究结果而受到称赞。
- **ChatGPT 停机理论**: 成员们对 ChatGPT 的停机进行了推测，怀疑是 Cloudflare 或 Azure 等后端供应商的问题。随后有人指出，亲俄黑客组织 Anonymous Sudan 发起的 DDoS 攻击导致了此次中断。
- **GPT-4o 的新语音功能**: 成员们对 GPT-4o 中预计在未来几周推出的新语音和视觉功能充满期待。基于最近的演示视频，成员们表达了渴望，但也对时间表持怀疑态度。
- **提高 GPT-3 输出一致性**: 一位发布文案软件的用户寻求关于训练 GPT 以获得一致输出的建议。讨论了 fine-tuning 模型和使用 few-shot learning 技术等选项，并引用了详细的 [OpenAI documentation](https://help.openai.com/en/articles/6614161-how-can-i-contact-support)。

**链接提到**: <a href="https://elicit.com/">Elicit: The AI Research Assistant</a>: 使用 AI 搜索、总结、提取数据并与超过 1.25 亿篇论文对话。被学术界和工业界的 200 多万名研究人员使用。

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1247463165135159338)** (26 messages🔥): 

- **ChatGPT 服务器停机及解决方案**: 用户因停机而遇到登录 ChatGPT 的困难。OpenAI 承认了这一问题，并建议尝试 [performing a hard refresh](https://status.openai.com/) 作为可能的解决方案。

- **新语音模式的状态**: 成员们讨论了 Apple 应用新语音模式推出的延迟。一位成员指出，“*Apple 应用显示它将在未来几周内可用，但这句话已经显示了快三周了。*”

- **持续的 Bad Gateway 错误**: 几位用户报告在尝试使用 ChatGPT 时遇到 “bad gateway” 问题，表明服务尚未对所有人恢复稳定。

- **长 Prompt 的性能问题**: 用户注意到长 Prompt 会导致性能下降。一位用户提到，“*Prompt 越大，情况就越糟，*” 而另一位用户建议使用 lazy loading 来缓解浏览器问题。

- **Android 应用键盘卡顿**: 成员们注意到 Android 版 ChatGPT 应用存在键盘卡顿问题。尽管尝试了清除缓存和重启等故障排除措施，但问题对某些用户仍然存在。

**链接提到**: <a href="https://status.openai.com/">OpenAI Status</a>: 未找到描述

  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1247402129866625084)** (81 messages🔥🔥): 

- **Prompt Engineering 过于超前**: 一位成员表达了挫败感，尽管花了五天时间精炼 Prompt，GPT 在遵循指南方面仍表现不一，且难以处理复杂的请求。他们得出结论，该模型目前的形式尚不完善，并希望未来的版本能有所改进。

- **WizardLM 2 “太出色了”**: 另一位成员建议尝试 **WizardLM 2**，声称其表现优于 GPT-4，并因性能过高而被撤下，不过他们提出可以帮忙测试一些 Prompt，因为他们在本地运行该模型。

- **API 成本 vs. ChatGPT Plus**: 随后讨论了使用 API 与 **ChatGPT Plus** 的成本效益，有人声称除非使用大量的 tokens，否则 API 可能会更便宜。**OpenRouter** 和其他替代方案被提议为可能更经济的选择。

- **复杂 Prompt 与多次调用**: 成员们讨论了将复杂的 Prompt 拆分为多次调用以提高一致性，并在这种方法与时间和成本约束之间取得平衡。Pipeline 和清晰的可计算指令被建议作为优化策略。

- **深刻的阅读建议**: 推荐阅读一篇名为 “**Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models**” 的 Prompt Engineering 研究文章。这篇建议通过搜索引擎查找的论文，可能会为改进 Prompt Engineering 技术提供见解。

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1247402129866625084)** (81 条消息🔥🔥): 

- **过多的 prompt 请求导致不一致性**：一位成员表达了挫败感，称尽管经过了五天的微调，GPT 的 **token limit** 和整体“智能程度”在尝试于单次请求中遵循多条准则时会导致不一致。他们指出，*"GPT 在单次请求中仍然不擅长遵循过多的准则。"*
- **具有商业价值的 prompt 引发质疑**：一位成员声称他们的 prompt 具有很高的商业价值，无法公开分享，这导致另一位成员在没看到 prompt 的情况下质疑其能力。这引发了怀疑，有人评论道：*"有趣的是，有这么多人拥有这些无法分享的宝贵 prompt，但同时他们又需要针对这些 prompt 的帮助。"*
- **WizardLM 2 提供了一个极具前景的替代方案**：另一位成员推荐了 **WizardLM 2**，解释说它的性能超过了 GPT-4，但由于未经验证的安全担忧而被削减。他们提议在上面测试 prompt，并强调它是开源且公开的。
- **OpenRouter 和经济实惠的替代方案**：关于高性价比 AI 解决方案的讨论揭示了 **OpenRouter** 极具竞争力的定价，WizardLM 2 的价格为每百万 token 0.65 美元，还有像 Llama 3 8b instruct 这样的免费模型。由于 API 成本问题，成员们考虑了这些替代方案。
- **关于 prompt engineering 的阅读推荐**：一位成员建议阅读一篇名为 **"Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"** 的文章，以应对复杂的 prompt engineering 挑战。他们强调，将任务分解为多个步骤可以解决性能问题。
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1247267877522903161)** (56 条消息🔥🔥): 

- **关于 LM Studio 中 CPU 与 GPU 使用的辩论**：成员们讨论了 LM Studio 模型是在 CPU 还是 GPU 上运行，达成的共识是 **LM Studio 通过 GPU offload 参数处理这些设置**。*"混合模式允许大型模型同时使用 RAM 和 VRAM，但如果可用，专用 GPU 通常更好。"*

- **加载模型时的错误**：一位用户分享了一个 JSON 错误，指示加载模型时出现问题，并收到建议禁用 **GPU Offload** 并确保模型大小在 6GB 以下。*"尝试加载大小为 6GB 或更小的模型。"*

- **LM Studio 中请求的异步处理**：用户澄清说 **LM Studio 的服务器按顺序处理请求**，默认不支持并行运行任务。*"所有请求都在排队并一个接一个地处理。"*

- **模型 token 限制和追踪的挑战**：一位用户询问了 LM Studio 能处理的最大 token 数，随后得到的澄清是 **token 限制取决于具体模型**，并且有必要手动追踪 token 使用情况。*"token 数量取决于模型。你可以在 model card 中找到该信息。"*

- **下载和管理模型**：用户询问了通过 CLI 下载模型和处理多个模型加载的问题，得到的回复建议 **使用 LM Studio 下载模型，并提醒注意内存限制**。*"通过 LM Studio 本身下载它们，并通过 LMS CLI 加载它们似乎可行。"*

**提到的链接**：<a href="https://tenor.com/view/tf2engineer-imposter-it-could-be-you-meme-tf2spy-gif-23428001">Tf2engineer Imposter GIF - Tf2Engineer Imposter It Could Be You - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1247282087657214033)** (69 条消息🔥🔥): 

- **CodeQwen 1.5 和 Codestral 22B 被推荐用于代码修复**：成员们建议将 **CodeQwen 1.5 7B** 和 **Codestral 22b** 用于代码修复任务。另一位成员指出 **wavecoder-ultra-6.7b** 也是一个值得考虑的不错选择。

- **Wavecoder Ultra 模糊的发布历史**：“Wavecoder Ultra 也是一个非常奇怪的模型，因为它是在 Wizard 的阴影下发布的。” 这暗示了它的发布过程比较零散，且信息有限。

- **分享 MahouDevil 8B 模型**：提供了 **MahouDevil-8B** 的 [Q8](https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF) 和 [Q4_K_M](https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q4_K_M-GGUF) 量化版本的下载链接。

- **使用 Extractum.io 过滤模型搜索**：成员们讨论了使用 [Extractum.io](https://llm.extractum.io/list) 根据 VRAM 和量化等各种标准过滤庞大的模型列表，并提供了指向 **Hugging Face** 模型的链接。

- **解释量化 (Quantization)**：量化被描述为减少模型权重中小数位数以使模型变小的过程，解释为：*“量化会降低质量，但也会使模型变小。”*
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF">YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q4_K_M-GGUF">YorkieOH10/Llama-3-MahouDevil-8B-Q4_K_M-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Contamination/contaminated_proof_7b_v1.0_safetensor">Contamination/contaminated_proof_7b_v1.0_safetensor · Hugging Face</a>：未找到描述</li><li><a href="https://llm.extractum.io/list/">所有大语言模型</a>：大语言模型和小语言模型（开源 LLM 和 SLM）的精选列表。具有动态排序和过滤功能的所有大语言模型。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1247267324562640937)** (1 条消息): 

- **Llama-3-70B 的推理速度测试**：一位用户分享了在使用 **q4 4090** 配置测试 **llama-3-70b** 各种推理速度后的发现。这些信息对于拥有类似硬件配置的用户特别有用。
  

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1247272943231434852)** (129 messages🔥🔥): 

- **6800XT GPU 性能故障排除**：一位使用 6800XT GPU 运行 **CodeLlama** 和 **Mistral7b** 等模型的成员指出，他们的运行速度仅为 12it/s，远低于预期的 52it/s。在从 **OpenCL** 切换到 **ROCm** 并正确重新加载模型后，速度提升到了 30it/s，但仍在努力调试以达到最佳性能。

- **关于 Homelab GPU 选择的辩论**：在一次关于 AI/服务器配置的讨论中，由于担心 VRAM 限制，有建议将 **12GB GPU** 替换为性能更强的 **4060 Ti (16GB)**。二手 **3090s** 被推荐为 **4090** 和其他高端选项的性价比替代方案。

- **驱动程序和软件栈考量**：在考虑 GPU 选项时，成员们强调 **Intel Arc** 尚未得到完全支持，且 **AMD ROCm** 可能比 NVIDIA 的软件栈慢。为了稳定性和性能，偏好转向了支持良好的 NVIDIA 显卡。

- **探索替代 Linux 设置**：关于建立 Homelab 的讨论还涉及了不同 Linux 发行版的优缺点。**Ubuntu** 因其易用性而受到欢迎，尽管一些成员更喜欢 **Arch** 的包管理，但也承认其学习曲线较陡。

- **对二手 GPU 的信任**：成员们考虑从市场购买二手 GPU（如 **3090**）作为节省成本的措施。然而，他们也讨论了潜在的性能损失和功耗等权衡，特别是在不使用 NVLink 的情况下使用多张显卡时。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dev.to/maximsaplin/running-local-llms-cpu-vs-gpu-a-quick-speed-test-2cjn/">未找到标题</a>：未找到描述</li><li><a href="https://lmstudio.ai/rocm">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLM</li><li><a href="https://wccftech.com/amd-radeon-pro-w7900-dual-slot-gpu-48-gb-ai-workstations-compact-design-3499-usd/">AMD Radeon PRO W7900 双槽 GPU 为 AI 工作站带来 48 GB 显存，采用紧凑设计，售价 3499 美元</a>：AMD 正在其产品线中增加一款新的 Radeon PRO W7900 GPU，该显卡采用双槽设计，但保留了与前代模型相同的规格。</li><li><a href="https://tenor.com/view/arch-linux-arch-btw-cv-walter-white-gif-24576245">Arch Linux Arch Btw GIF - Arch Linux Arch Btw Cv - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1247298531312537610)** (2 messages): 

- **内存带宽限制 AI 性能**：一位成员指出 **AI 受内存带宽限制**，并解释说在每个物理核心上运行多个线程可能会阻塞内存控制器并降低性能。他们建议理想的线程数约为 *物理核心数的 80%*。
- **中文语言支持咨询**：一位成员询问何时会提供**中文语言支持**。
  

---


### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1247624625215504414)** (1 messages): 

- **追踪 LM Studio 即将发布的版本**：一位成员询问了 **LM Studio** 即将发布的版本，并询问是否有地方可以追踪进度。他们表示一旦问题修复，就有兴趣重新使用 **LM Studio**。
  

---


### **LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1247568831367614557)** (3 messages): 

- **征集 AVX 扩展包测试人员**：发布了一项针对早期仅限 AVX 扩展包的测试人员征集。消息鼓励感兴趣的成员点赞或评论，并收到了社区的积极响应。
  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1247338818001043529)** (1 messages): 

- **Continue.dev 在本地设置中表现出色**：一位成员称赞 **continue.dev** 是本地使用的 *"最好的一个..."*，并引用了其文档的最新更新。他们特别提到了增加了关于如何 *"设置 LM Studio 支持"* 的步骤。
  

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1247264658063097969)** (12 条消息🔥): 


- **音乐视频讨论气氛活跃**：成员们分享了一系列 YouTube 音乐视频链接，例如 *Mindchatter - Night Goggles (Rome in Silver Remix)* 和 *Porter Robinson & Madeon - Shelter (Official Video)*。这些视频似乎在社区中引起了很好的共鸣。
- **Wayseer Manifesto 亮相**：分享了 *Wayseer Manifesto - [Official Video]*，强调了动力与打破常规主题的独特结合。该视频也有西班牙语版本，拥有超过 600 万次的播放量。
- **设计话题引发关注**：关于 Nous Research 令人愉悦的设计工作的评论引起了注意，一位用户幽默地宣称：“我就是设计人员。” 更多他们的设计作品可以在他们的 [Twitter 账号](https://x.com/StudioMilitary)上找到。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/OPR3GlpQQJA">THE WAYSEER MANIFESTO - [Official Video] (HQ)</a>: 西班牙语版本现已上线，播放量超过 600 万：http://www.youtube.com/watch?&amp;v=KYfc5_YFFb0#!注意：所有打破规则的人、不合群的人和麻烦制造者...</li><li><a href="https://youtu.be/fzQ6gRAEoy0">Porter Robinson &amp; Madeon - Shelter (Official Video) (Short Film with A-1 Pictures &amp; Crunchyroll)</a>: Porter Robinson &amp; Madeon - Shelter (官方视频) (与 A-1 Pictures &amp; Crunchyroll 合作的短片) Shelter 讲述了 17 岁女孩 Rin 的故事，她生活在...</li><li><a href="https://x.com/StudioMilitary">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://youtu.be/A5Npdlg1Vaw">Mindchatter - Night Goggles (Rome in Silver Remix)</a>: 流媒体/下载：https://lnk.to/nightgogglesromeinsilverremixID 关注 Mindchatter：https://mindchatter.lnk.to/Instagram https://mindchatter.lnk.to/Twitter https:...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

sidfeels: https://laion.ai/notes/open-gpt-4-o/
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1247263672175300629)** (198 条消息🔥🔥): 

- **T5 硬件需求限制了其普及**：一位成员指出 T5 对硬件要求很高，影响了它的采用。另一位成员提到缺乏适用于广泛用途的快速 CPU 实现，并提到了 Huggingface 有限的 Candle 库以及未来使用 ggml 的潜在发展。
  
- **需要开源版本的聊天助手 UI**：一位用户询问是否有开源版本的 Huggingface 聊天助手 UI，用于定义 prompt templates 和自定义 vector databases。另一位成员推荐了来自 Mobius 的更好替代方案，并指出其比之前的实现有所改进。
  
- **提出 Pixart 训练问题**：讨论了 Pixart 在大型数据集上训练时崩溃的问题，强调了它在仅使用 10k 张图像时就会产生混乱的形状。相反，另一个模型由于采用了独特的训练技术，在超过 400k 张图像的情况下表现良好。

- **数据集与 AGIEval 排行榜**：讨论了大型数据集，提到了拥有 30 万亿 token 的 Redpajama v2。有关使用 AGIEval 的排行榜请求被引导至一个包含相关 benchmark 日志的 GitHub 仓库。

- **Mobius 模型亮点及算力赞助提议**：展示了 Mobius 模型的能力亮点，并分享了仓库。该模型的创建者提议为有趣的项目赞助算力，并讨论了潜在的合作。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Corcelio/mobius">Corcelio/mobius · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/cloneofsimo/lavenderflow-5.6B">cloneofsimo/lavenderflow-5.6B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.together.ai/blog/redpajama-data-v2">RedPajama-Data-v2: An open dataset with 30 trillion tokens for training large language models</a>: 未找到描述</li><li><a href="https://discord.gg/kRbaDnHE">加入 PixArt-α Discord 服务器！</a>: 在 Discord 上查看 PixArt-α 社区 - 与其他 1702 名成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://www.academia.edu/120538461/Sensuality_in_Emergent_LLM_Models_A_New_Turing_Criteria_question">涌现式 LLM 模型中的感官性 - 一个新的图灵准则问题</a>: "涌现式 LLM 模型中的感官性 - 一个新的图灵准则问题？" 在本文中，我们将提出表达“愉悦”的 AI/LLM 模型和实例...</li><li><a href="https://x.com/zhanga6/status/1797293189378068768?s=46&t=zdoDWYj2oTzRaTJHApTcOw">来自 Ao Zhang (@zhanga6) 的推文</a>: 听到这个消息非常难过 (https://github.com/OpenBMB/MiniCPM-V/issues/196)😰。我们调查的结论是：1. Llama3-V 在修改...后可以使用 MiniCPM-Llama3-V 2.5 的代码和 config.json 运行。</li><li><a href="https://tenor.com/view/plink-nerd-plank-plink-cat-cat-gif-17569403098672348326">Plink Nerd GIF - Plink Nerd Plank - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/akshgarg03/status/1797682238961914370?s=46&t=zdoDWYj2oTzRaTJHApTcOw">来自 Aksh Garg (@AkshGarg03) 的推文</a>: 关于 Llama3V：首先，我们要向 MiniCPM 的原作者道歉。我们本想让 Mustafa 发表原始声明，但自昨天以来一直无法联系到他。@siddrrsh 和 ...</li><li><a href="https://github.com/teknium1/LLM-Benchmark-Logs">GitHub - teknium1/LLM-Benchmark-Logs: 一系列不同 LLM 的基准测试日志</a>: 一系列不同 LLM 的基准测试日志。通过在 GitHub 上创建账户，为 teknium1/LLM-Benchmark-Logs 的开发做出贡献。</li><li><a href="https://x.com/Teknium1/status/1797491548353183994">来自 Teknium (e/λ) (@Teknium1) 的推文</a>: 训练 SD3 需要多少张图像以及多少 H100 小时？</li><li><a href="https://x.com/shreyaskapur/status/1797726079995826629">来自 Shreyas Kapur (@shreyaskapur) 的推文</a>: 我的第一篇博士论文！🎉我们学习了用于代码生成的 *diffusion* 模型，该模型学习直接 *编辑* 程序的语法树。结果是一个可以增量编写代码、查看执行情况的系统...</li><li><a href="https://x.com/yangzhizheng1/status/1797197104999518306?s=46&t=zdoDWYj2oTzRaTJHApTcOw">来自 PrimerYang (@yangzhizheng1) 的推文</a>: 震惊！来自斯坦福团队的 Llama3-V 项目大量抄袭了 MiniCPM-Llama3-V 2.5！其代码是 MiniCPM-Llama3-V 2.5 的重新格式化，且模型的行为与一个带噪声的...高度相似。</li><li><a href="https://github.com/NX-AI/xlstm">GitHub - NX-AI/xlstm: xLSTM 的官方仓库。</a>: xLSTM 的官方仓库。通过在 GitHub 上创建账户，为 NX-AI/xlstm 的开发做出贡献。</li><li><a href="https://bellard.org/nncp/">NNCP: 基于神经网络的无损数据压缩</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 条消息): 

vorpal_strikes: 迫不及待想看到 2030 年 Nous World 的涌现
  

---

### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1247469340379320431)** (4 条消息): 

- **WorldSim Jam Session 在 YouTube 播出**：几周前的 WorldSim Jam Session 将在 YouTube “首映”。参与者可以[在此重温](https://www.youtube.com/watch?v=qaE99xmtvd4)。
- **探索 LLM 系统评估**：一段名为 *“让 Agent 便宜、快速且更好 10 倍？”* 的详细视频讨论了 LLM 系统评估（System Evaluation）。你可以在[这里](https://youtu.be/MGkByeDm-90?si=14AxwOfkULwC4a2U)查看。
- **AI Agent 职业讽刺**：一位成员幽默地指出，最容易被 AI Agent 取代的职业竟然是……Agent（代理人），并配上了大笑的表情符号。
- **回归成员动态**：一位回归的成员表达了对小组的想念，并提到在离开期间“写了一篇论文并做了一些事情”。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/MGkByeDm-90?si=14AxwOfkULwC4a2U">&quot;Make Agent 10x cheaper, faster &amp; better?&quot; -  LLM System Evaluation 101</a>: LLM System Eval 101 - 构建更好的 Agent。获取关于如何使用 AI 找工作的免费 HubSpot 报告: https://clickhubspot.com/fo2🔗 链接 - 在 twitter 上关注我: h...</li><li><a href="https://www.youtube.com/watch?v=qaE99xmtvd4">WorldSim Jam Session #2</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1247268569436393513)** (159 条消息🔥🔥): 

- **SD3 模型引发褒贬不一的反应**：成员们分享了对 SD3 早期版本的挫败感，称其手部绘制不当，且表现不如 Dall-E。一位用户评论道：“给它一点本地发布的时间，不久之后自定义模型就会完全解决这个问题。”

- **Stable Diffusion 用于建筑**：一位成员询问如何将 Stable Diffusion 用于建筑项目，特别是基于图纸的室内预览。另一位成员回答说：“它在处理直线和机械结构方面表现一般”，但建议使用 img2img 配合详细图纸来尝试获得结果。

- **Wildcards 插件问题**：一位用户在 Stable Diffusion 上安装 Wildcards 插件后遇到了图像质量下降的问题，提到“图像颗粒感极重，面部出现了丑陋的色块”。尽管多次重新安装，问题依然存在。

- **社区模型与资源讨论**：成员们推荐了来自 [civitai.com](https://civitai.com) 等网站的社区模型，以提升 Stable Diffusion 的渲染质量。[ChaiNNer](https://github.com/JoeyBallentine/chaiNNer) 也被推荐作为需要批量放大图像用户的 Upscaler 工具。

- **名人作为 AI 模型**：社区讨论了 Civit 等平台上网红和名人 LoRa 的兴起，注意到 AI 生成的个人资料大量涌现。一位用户针对这一趋势幽默地评论道：“什么时候名人不再是名人了？”
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/excited-excited-man-excited-funny-rubbing-hands-plotting-gif-27652478">Excited Excited Man GIF - 兴奋的男人 GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.instagram.com/fit_aitana/">登录 • Instagram</a>: 未找到描述</li><li><a href="https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s34B-b88K">laion/CLIP-ViT-g-14-laion2B-s34B-b88K · Hugging Face</a>: 未找到描述</li><li><a href="https://youtube.com/watch?v=JOiK8z-Ffp4">免费提升你的 MidJourney 或 Dalle-2 AI 艺术作品分辨率！</a>: 有一个绝佳的方法不仅可以放大来自 MidJourney、Dalle-2 或 Stable Diffusion 的 AI 艺术作品，还可以保持甚至增加细节...</li><li><a href="https://www.youtube.com/watch?v=lNGEeUCL8NE">SD一键生成视频！免费开源，想怎么玩就怎么玩！附完整安装教程  | 零度解说</a>: 【更多资源】▶https://www.youtube.com/channel/UCvijahEyGtvMpmMHBu4FS2w?sub_confirmation=1【零度博客】▶https://www.freedidi.com【加入会员】▶https://www.youtube.com/channel/UCvij...</li><li><a href="https://www.instagram.com/tucumandigital/p/C1ndOWvrhXY/?img_index=1">Tucuman Digital 在 Instagram 上: &quot;#不可思议 🔴 在成人平台大火但并非真人的模特 😮

这就是 Emily Pellegrini，她在最近几周通过在成人平台创建个人资料赢得了数千名粉丝，挑战了知名的 Onlyfans。在社交网络上，AI 生成的网红正在增加。现在，一位出售成人内容的模特在 Instagram 上走红：21 岁的 Emily Pellegrini，在热门平台上已经积累了数千名粉丝。

这位网红在短短六周内通过 Fanvue（一个与 Onlyfans 竞争的成人平台）销售内容，获得了接近 1 万美元的收入。尽管她“居住”在意大利，但 Emily Pellegrini 的特别之处在于她的所有图像都是通过 AI 创建的，这种方法近几个月在互联网上非常流行。

每月订阅费用约为 9 美元，目前已有近 100 名常规订阅者。Emily Pellegrini 的知名度极高，在短短六周内，她的 Instagram 粉丝已累计接近 9 万。甚至在她的社交媒体个人资料中，还有一个精彩动态栏目，展示她与“朋友们”在一起的画面，而这些朋友也是该平台上其他由 AI 创建的模型。

Emily Pellegrini 的名声远播，国际知名媒体《纽约邮报》（New York Post）专门撰文报道了她以及 Fanvue 平台。该平台与 Onlyfans 类似，但特点是所有模型均完全由 AI 创建。

该网站创始人 Will Monange 认为，AI 是一种“工具”，是“我们是谁以及我们所做事情的延伸”，这与其他人将其视为人类创造力替代品的看法不同。"</a>: 1,779 次点赞, 13 条评论 - tucumandigital 于 2024 年 1 月 2 日发布: "#不可思议 🔴 在成人平台大红大紫但并非真人的模特 😮 她就是 Emily Pellegr...</li><li><a href="https://github.com/jaisidhsingh/CoN-CLIP">GitHub - jaisidhsingh/CoN-CLIP</a>: 通过在 GitHub 上创建账号来为 CoN-CLIP 的开发做出贡献。</li><li><a href="https://civitai.com/models/432244/forrealxl">ForRealXL - v0.5 | Stable Diffusion Checkpoint | Civitai</a>: 你好 ♥ 无论出于何种原因你想向我表达感谢，你可以：❤️ Ko-Fi ❤️ 这是一个实验性的 Checkpoint，因为这是我的第一个作品。特别感谢...</li><li><a href="https://github.com/JoeyBallentine/chaiNNer">GitHub - chaiNNer-org/chaiNNer: A node-based image processing GUI aimed at making chaining image processing tasks easy and customizable. Born as an AI upscaling application, chaiNNer has grown into an extremely flexible and powerful programmatic image processing application.</a>: 一个基于节点的图像处理 GUI，旨在使链式图像处理任务变得简单且可定制。chaiNNer 最初是一个 AI 放大应用，现已发展成为一个极其灵活且强大的程序化图像处理应用。</li><li><a href="https://upscale.wiki/wiki/Model_Database">Model Database - Upscale Wiki</a>: 未找到描述。
</li>
</ul>

</div>
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1247264971935580381)** (107 条消息🔥🔥): 

- **SD3 模型受噪声伪影困扰**：成员们讨论了 SD3 medium (2B) 模型中“持续存在的斑点噪声问题”，表现为“奇怪的喷雾状斑点噪声”，在喷泉中的水等特定区域尤为明显，尽管使用了 16ch VAE 等先进特性。
- **验证与评估问题**：对于评估 SD3 模型时使用的当前“验证指标”和损失函数的有效性，存在明显的质疑。一位用户强调“*loss 是衡量模型性能的一个糟糕指标*”，并批评了 SD3 论文中描述的方法。
- **跨模型的伪影问题**：对话探讨了 *MM-DiT 模型*由于其架构原因如何遭受一致的伪影影响，从而影响整体图像质量。为了证实这一点，一位成员指出：“对于单个 seed，你会在基本上每个 prompt 中看到**相同**的伪影。”
- **开源视频 VAE 新闻**：对于发布具有 Apache2 自由许可的 [支持视频的 CV-VAE](https://arxiv.org/abs/2405.20279) 感到兴奋，强调了它对于研究社区内基于 latent diffusion 的视频模型的重要性。
- **Reddit 和 Google 示例**：成员们对比了 SD3 图像，有人声称它们看起来像“遭受了糟糕肉毒杆菌注射的女性” ([Reddit 链接](https://www.reddit.com/r/StableDiffusion/comments/1d73j3r/some_sd3_images_women/))。Google 最近的演示 ([Twitter 链接](https://vxtwitter.com/GoogleDeepMind/status/1797605392089825457)) 因其逼真的示例而获得赞誉，特别是在织物纹理和头发一致性方面。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.20279">CV-VAE: A Compatible Video VAE for Latent Generative Video Models</a>：视频的时空压缩，利用诸如 Variational Autoencoders (VAE) 等网络，在 OpenAI 的 SORA 和许多其他视频生成模型中起着至关重要的作用。例如，ma...</li><li><a href="https://x.com/Lykon4072/status/1797703714180051130?t=4DG7gVlXqw65fOJrNpHBAw&s=19">来自 Lykon (@Lykon4072) 的推文</a>：作为参考，SD3 2B 的大小大致相同，但它是 MMDiT（远优于 Unet），并使用了 3 个 text encoders，此外还有一个 16ch VAE。如果不进行 c...，你在 XL 中无法获得这种级别的细节。
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1247267349996769372)** (17 messages🔥): 

- **Transformers 与 State-Space Models 的竞争引发关注**：一篇共享论文 ([arxiv.org/abs/2405.21060](https://arxiv.org/abs/2405.21060)) 讨论了像 Mamba 这样的 State-Space Models (SSMs) 如何能够超越 Transformers，并引入了增强模型效率的 **State Space Duality (SSD)** 框架。新架构 **Mamba-2** 比其前身快 2-8 倍，且在语言建模方面具有与 Transformers 相当的竞争力。

- **Tiny Bit 方法遭到批评**：关于 1.58-bit 方法有效性的询问显示出普遍的怀疑和负面反馈。用户报告称：“我认识的每个尝试过这些 Tiny Bit 方法的人都说它们很烂。”

- **Diffusion Models 受益于数据损坏**：一份预印本 ([arxiv.org/abs/2405.20494](https://arxiv.org/abs/2405.20494)) 表明，预训练数据集中的轻微损坏可以提高 Diffusion Models (DMs) 生成图像的质量和多样性。该论文引入了 **Conditional Embedding Perturbation (CEP)**，展示了有意识的数据损坏对模型训练有益。

- **对大型 Diffusion Models 和 Overfitting 的担忧**：讨论强调了对大型 Diffusion Models（特别是参数超过 5 亿的模型）在 ImageNet 等数据集上发生 Overfitting 的担忧。Dropout 和 Data augmentation 被提及作为对抗 Overfitting 的方法。

- **关于 Robustness 和模型训练难度的辩论**：对话探讨了增加训练数据难度如何提高模型的 Robustness。参考 [Distill.pub 关于可解释性的文章](https://distill.pub/2020/circuits/)，“在优化过程中增加任务难度可以产生更好的模型”这一直觉被认为是合理的，尽管量化这种效果仍然是一个挑战。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.21060">Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality</a>：虽然 Transformers 一直是深度学习在语言建模领域取得成功的核心架构，但最近的研究表明，像 Mamba 这样的 State-Space Models (SSMs) 能够匹配或超越 Transformer...</li><li><a href="https://arxiv.org/abs/2405.20494">Slight Corruption in Pre-training Data Makes Better Diffusion Models</a>：Diffusion Models (DMs) 在生成逼真的高质量图像、音频和视频方面展现了卓越的能力。它们从大规模数据集的广泛预训练中获益匪浅...</li><li><a href="https://distill.pub/2020/circuits/">Thread: Circuits</a>：如果我们投入大量精力对单个神经网络进行逆向工程，我们能学到什么？
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1247345403448262716)** (68 条消息🔥🔥): 

- **Daylight 平板电脑辩论引发争议**：成员们对[一段 YouTube 视频](https://youtu.be/iHeIw9rXzUQ?si=QDETgMaKJTaTYLkV)中评测 Daylight 平板电脑的说法进行了辩论，质疑其作为 e-paper 设备的广告宣传。许多人认为它与 E-ink 误导性地相似，但实际上是反射式 LCD (RLCD)，引发了对其价格和功能（与 iPad 和 Kindle 等设备相比）的不满。
  
- **e-paper 技术的创新与误标**：讨论强调了创新与误标之间的界限，用户认为 Daylight 平板电脑声称发明了新的 RLCD 技术具有误导性。批评者对其声称的卓越 FPS 和日光可见性表示担忧，而其他人指出这可能只是在现有的 Sharp RLCD 技术上进行的品牌包装。

- **新设备的电池续航担忧**：电池续航对比也随之出现，一些人强调了 Kindle 数周的续航时间，而 Daylight 平板电脑声称可持续数天到一周。这引发了人们对该新设备作为在阳光下进行笔记和绘图等日常使用平板电脑实用性的怀疑。

- **对创始声明的怀疑**：争论延伸到怀疑 e-paper 声明的真实性，一些用户注意到创始人出现在 Sharp 工厂外，这表明其仅仅是使用了预先存在的 Sharp 技术而非创新。这种怀疑遭到了反驳，反驳者强调缺乏直接证据来证明或反驳这一新颖技术声明。

- **呼吁通过潜在的拆解进一步调查**：辩论以建议对 Daylight 平板电脑进行拆解结束，这可能会揭示其技术的真实本质，并将其与现有的 Sharp RLCD 进行比较。在没有物理验证的情况下，理论讨论仍无定论，可能在等待有人提供具体证据来平息争论。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sharpdevices.com/reflective-lcd/#1619791276508-913e49e4-989a">未找到标题</a>: 未找到描述</li><li><a href="https://youtu.be/9M6zT0mRvW0?si=RwRcoY8AVROZR_5W">World&#39;s first 60+ FPS e-paper display | Daylight</a>: 世界上首款 60+ FPS e-paper 显示屏，由 Daylight 推出。S³ 第 45 集。了解其工作原理、6 年的开发历程以及 Daylight 对未来的愿景...</li><li><a href="https://youtu.be/iHeIw9rXzUQ?si=QDETgMaKJTaTYLkV">Daylight Tablet Review: World’s First 60 Hz E-Paper!</a>: 剧透警告：它不是 E-ink。Daylight DC1: https://daylightcomputer.com/ Reddit 帖子: https://www.reddit.com/r/daylightco/comments/1cz23hj/anjans_mission/🖼️...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1247276260002893915)** (46 messages🔥): 

- **LLM 内部机制的新见解**：一篇研究论文[探讨了预训练 Transformer 中的知识电路 (knowledge circuits)](https://arxiv.org/abs/2405.17969)，其研究范围扩展到了注意力头 (attention heads) 和 MLP 等孤立组件之外。一位成员对该方法表示赞赏，认为它比目前流行的方法更合适。
  
- **TinyLLama 数据库的挑战**：一位从事 TinyLLama.db 工作的成员透露，他们正努力协调激活值与权重条目，并快速推送数据库更改。他们还表示需要将模型从 22GB 缩小，因为他们认为这个体积太大了。
  
- **损坏的数据有助于 Diffusion 模型**：由微软和 CMU 发布的一篇[预印本论文](https://arxiv.org/abs/2405.20494)揭示，轻微损坏的数据集可以使 Diffusion 模型受益，提升图像质量和多样性。该研究引入了条件嵌入扰动 (Conditional Embedding Perturbation, CEP) 并评估了 50 多个模型。
  
- **用于视觉模型的可微 top-k 函数**：一位成员探索了[可微 top-k 函数](https://math.stackexchange.com/questions/3280757/differentiable-top-k-function)，用于为分类任务选择最重要的图像 token。他们发现 ViT 模型中的注意力图 (attention maps) 可解释性较差。
  
- **RNN 与预训练 Transformer 的协同作用**：讨论集中在利用预训练 Transformer 来稳定和指导新 RNN 训练的想法上。提出的一种方法是让 Transformer 为 RNN 提供信息向量，尽管两种架构在状态追踪 (state tracking) 方面都存在理论限制，但这仍可能增强学习效果。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.20494">Slight Corruption in Pre-training Data Makes Better Diffusion Models</a>：Diffusion 模型 (DMs) 在生成逼真的高质量图像、音频和视频方面展现了卓越的能力。它们从大规模数据集的广泛预训练中获益匪浅...</li><li><a href="https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp">McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp · Hugging Face</a>：暂无描述</li><li><a href="https://arxiv.org/abs/2405.17969">Knowledge Circuits in Pretrained Transformers</a>：现代大语言模型的卓越能力源于其参数中编码的海量知识库，使它们能够感知世界并进行推理...</li><li><a href="https://tridao.me/blog/2024/mamba2-part1-model/"> State Space Duality (Mamba-2) Part I - The Model | Tri Dao </a>：暂无描述</li><li><a href="https://arxiv.org/abs/2405.16674">Limits of Deep Learning: Sequence Modeling through the Lens of Complexity Theory</a>：深度学习模型在各种应用中取得了显著成功，但在需要对序列进行复杂推理的任务（如函数组合和计算）中仍然面临困难...</li><li><a href="https://arxiv.org/abs/2404.08819">The Illusion of State in State-Space Models</a>：与之前无处不在的 Transformer 架构相比，状态空间模型 (SSMs) 已成为构建大语言模型 (LLMs) 的一种潜在替代架构。一个理论上的...</li><li><a href="https://math.stackexchange.com/questions/3280757/differentiable-top-k-function,">Differentiable top-k function</a>：是否存在任何可微函数，对于给定的向量，能够选择并鼓励前 k 个最大值并抑制其余值？例如，对于 z = [0.01 0.1 0.04 0.5 0.24]...</li><li><a href="https://arxiv.org/abs/2405.06640">Linearizing Large Language Models</a>：线性 Transformer 已成为 softmax 注意力的一种次二次时间替代方案，并且由于其固定大小的循环状态降低了推理成本而引起了广泛关注。然而...</li><li><a href="https://github.com/NX-AI/xlstm">GitHub - NX-AI/xlstm: Official repository of the xLSTM.</a>：xLSTM 的官方仓库。欢迎在 GitHub 上为 NX-AI/xlstm 的开发做出贡献。</li><li><a href="https://github.com/Felix-Petersen/diffsort">GitHub - Felix-Petersen/diffsort: Differentiable Sorting Networks</a>：可微排序网络。欢迎在 GitHub 上为 Felix-Petersen/diffsort 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2405.14838">From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step</a>：在利用语言模型进行推理任务时，生成显式的思维链 (CoT) 步骤通常对于实现最终输出的高准确度至关重要。在本文中，我们研究了...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1247425540797763636)** (9 条消息🔥): 

- **提高 AI 任务并发量**：一位成员询问是否有办法在运行任务时增加并发级别，因为目前的设置一次只运行一个 query，速度“极其缓慢”。目前还没有针对此问题的即时后续回复。

- **关于最小 Decoder 模型的咨询**：一位用户在 Huggingface 上寻找最小的可用 decoder 模型以测量能耗。另一位成员建议：“直接用随机值实例化一个小模型即可”，随后该用户请求提供相关教程。

- **N-shot 默认值影响结果**：有成员指出，在未指定 n-shot 值时，默认值为 5，这会导致结果与预期不符。这种差异被解释为 Huggingface Open LLM 排行榜使用的是旧版本的 harness。

- **针对 ARC Challenge 的 Pull Request**：一位成员请求对其在 [GitHub 上的 PR #1900](https://github.com/EleutherAI/lm-evaluation-harness/pull/1900) 进行评审，该 PR 为 11 种语言的 ARC challenge 机器翻译版本添加了任务，并计划在未来添加更多语言。

**提到的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1900">add arc_challenge_mt by jonabur · Pull Request #1900 · EleutherAI/lm-evaluation-harness</a>：此 PR 为 11 种语言的 arc challenge 机器翻译版本添加了任务。我们未来还将添加更多语言。

  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 条消息): 

merfippio：我知道这问得很晚，但你找到你需要的 FE 帮助了吗？
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1247339340867178497)** (106 条消息🔥🔥): 

- **ETH 支付后积分未到账问题**：一位用户报告称，在 Base 上使用 ETH 支付后，积分没有显示。另一位成员建议在积分仍未出现前等待 20 分钟到一个小时再发起投诉。
- **LLM 对 Prefill 的处理**：一位用户询问 LLM 对 prefill 文本的处理方式，以及它是否会像连续生成一样生成后续段落。共识是 prefill 被无缝处理，就像它是原始 prompt 的一部分一样。
- **GPT-3.5 Turbo 问题与审核**：一位用户报告 GPT-3.5 Turbo 无法正常工作，而其他 OpenRouter LLM 运行正常，这引发了关于 API 审核可能影响请求的讨论。OpenRouter 确认 OpenAI 要求所有请求都必须使用其 moderation API 进行审核。
- **Mistral 模型可靠性问题**：用户报告在 Fireworks 上使用 Mistral: Mixtral 8x22B Instruct 持续得到空响应，表明提供商可能存在问题。OpenRouter 的管理员建议将 DeepInfra 设置为首选提供商，并参考负载均衡文档进行手动的提供商白名单设置。
- **最适合讲故事的模型**：讨论了最适合讲故事的模型，建议包括来自 OpenRouter 排行榜的角色扮演专用模型和 Wizardlm2。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>：跨多个提供商路由请求</li><li><a href="https://openrouter.ai/rankings/roleplay?view=week">LLM Rankings: roleplay | OpenRouter</a>：根据角色扮演 prompt 的使用情况进行排名和分析的语言模型
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1247265544554549391)** (26 条消息🔥): 

- **使用 Numba 加速 Python**：分享了一个名为 ["Make Python 1000x Faster With One Line 🐍 ⏩ (Numba Tutorial)"](https://youtu.be/OiMZtjSZVOw) 的 YouTube 视频，解释了如何使用 JIT compiler 来显著加速 Python 代码。一位成员表示希望在不依赖外部库的情况下获得类似的性能。

- **关于 Python Generators 的讨论**：一位成员询问在 pure Python 中，在 while 循环中使用 for 循环是否有效，引发了关于 generators、`yield` 和其他优化方法的讨论。推荐了一篇关于 [Python Generators](https://realpython.com/introduction-to-python-generators/) 的 Real Python 文章和视频以供进一步学习。

- **Mojo 与 Python 执行**：有一场关于 MAX 是否能加速 Python 执行的对话，对比了各种工具和方法，例如 Tensor 和 Torch 库的优势与 pure Python 性能改进的对比。

- **Mojo 社区会议**：分享了第二次 Mojo Community Meeting 的信息，包括过去会议的 [Youtube playlist](https://www.youtube.com/playlist?list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D)。宣布下一次会议将在两周后举行，并公开邀请参与者通过在 [Google doc agenda](https://modul.ar/community-meeting-doc) 中添加自己的内容来分享他们的工作。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/OiMZtjSZVOw?si=JrgOG_UL662xZ48W">Make Python 1000x Faster With One Line 🐍 ⏩ (Numba Tutorial)</a>: Numba 可以通过使用 JIT compiler 仅用一行代码将你的 Python 代码加速 1000 倍，该编译器用于通过编译函数来优化 Python 中的简单函数...</li><li><a href="https://youtu.be/OiMZtjSZVOw?si=JrgO">Make Python 1000x Faster With One Line 🐍 ⏩ (Numba Tutorial)</a>: Numba 可以通过使用 JIT compiler 仅用一行代码将你的 Python 代码加速 1000 倍，该编译器用于通过编译函数来优化 Python 中的简单函数...</li><li><a href="https://realpython.com/introduction-to-python-generators/#creating-data-pipelines-with-generators">How to Use Generators and yield in Python – Real Python</a>: 在这个分步教程中，你将学习 Python 中的 generators 和 yielding。你将使用多个 Python yield 语句创建 generator 函数和 generator 表达式。你...</li><li><a href="https://modul.ar/community-meeting-doc.">[Public] Mojo Community Meeting</a>: Mojo Community Meeting 此文档链接：https://modul.ar/community-meeting-doc。这是一个公开文档；欢迎所有人查看和评论/建议。所有会议参与者必须遵守...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1247263614335717500)** (2 条消息): 

- **分享 Modular 推文**：一位成员分享了两条来自 **Modular** 的推文：[Tweet 1](https://twitter.com/Modular/status/1797699002353488183) 和 [Tweet 2](https://twitter.com/Modular/status/1798055387557749010)。分享后没有进一步的讨论或评论。
  

---


### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1247600282817265694)** (1 条消息): 

- **深入探讨 Mojo 中的 Ownership**：标题为 [Deep Dive into Ownership in Mojo](https://www.modular.com/blog/deep-dive-into-ownership-in-mojo) 的博客文章是探索 Mojo 中 ownership 系列的第二部分。它建立在第一部分 [What Ownership is Really About: A Mental Model Approach](https://www.modular.com/blog/what-ownership-is-really-about-a-mental-model-approach) 的概念之上，并包含了来自 CEO [Chris Lattner](https://www.modular.com/team/chris-lattner) 关于在 Mojo 的 compiler 中实现 ownership 的见解。

**提及的链接**: <a href="https://www.modular.com/blog/deep-dive-into-ownership-in-mojo">Modular: Deep Dive into Ownership in Mojo</a>: 我们正在为世界构建下一代 AI 开发者平台。查看我们的最新帖子：Deep Dive into Ownership in Mojo

  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/)** (1 条消息): 

melodyogonna: 我认为 cryptography 库还没有被添加到 stdlib 中

### **Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1247490462676287528)** (1 messages): 

- **Project Verona 挑战 Rust 的内存安全性且更易上手**：研究人员讨论了 **Project Verona**，这是一种旨在提供类似于 **Rust** 的内存安全保证，但模型更易于学习的编程语言。查看 [YouTube 视频](https://youtu.be/VU9QATDvidw) 了解题为“*Concurrent Mutation must go*”的深入演讲。

**提到的链接**：<a href="https://youtu.be/VU9QATDvidw">[POCL&#39;24] Concurrent Mutation must go</a>：[POCL&#39;24] Concurrent Mutation must go Matthew J. Parkinson, Sylvan Clebsch, Tobias Wrigstad, Sophia Drossopoulou, Elias Castegren, Ellen Arvidsson, Luke Chees...

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1247296560560341072)** (55 messages🔥🔥): 

- **关于 Mojo 中默认 `async` 的辩论**：一位成员建议 Mojo 中的每个函数都应该是隐式 `async` 的，以避免阻塞问题，特别是对于阻塞不可接受的 GPU 编程。其他人则担心这会偏离 Pythonic 实践，并可能给习惯于显式 `async`/`await` 思维的用户带来困难。

- **社区对 `async` 提案的反馈**：将 `async` 作为默认设置的建议引发了褒贬不一的反应。一些人主张保持函数同步，以维持与 Python 的兼容性并简化 Python 程序员的过渡，而另一些人则指出 GPU 编程中需要高效的调度和挂起机制。

- **JSON 解析说明**：有关于 JSON 的字节序（endianness）和解析惯例的咨询。澄清了 JSON 是一种通常采用 UTF-8 编码的文本格式，它是字节序无关的，并且在解析过程中使用索引 `[1:-1]` 来省略外层的括号。

- **问题报告和工具支持请求**：成员们讨论了代码助手在处理相对导入（relative imports）时可能存在的 bug，并鼓励提交 bug 报告。此外，已在 GitHub 上为 `tokei` 和 `tcount` 等工具创建了 issue 以添加 Mojo 支持，并请求其他人支持这些 issue（[Tokei Issue](https://github.com/XAMPPRocky/tokei/issues/1107), [TCount Issue](https://github.com/RRethy/tcount/issues/3)）。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/XAMPPRocky/tokei/issues/1107>">Issues · XAMPPRocky/tokei</a>：快速统计你的代码。通过在 GitHub 上创建账号为 XAMPPRocky/tokei 的开发做出贡献。</li><li><a href="https://github.com/RRethy/tcount/issues/3>">Issues · RRethy/tcount</a>：通过语法树中的 token 和模式统计你的代码。tokei/scc/cloc 的替代方案。- Issues · RRethy/tcount
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1247411061892714557)** (1 messages): 

- **Value 装饰器可能优化基准测试**：一位成员建议使用 `@value` 装饰器可以简化并可能提高基准测试性能。*“不确定它如何影响基准测试，但你可以使用 `@value` 装饰器并省去所有的 copyinit 和 moveinit 逻辑。”*
  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1247420844289032253)** (18 messages🔥): 

- **ComparableCollectionElement 正确继承**：一位用户确认 *ComparableCollectionElement* 通过 trait 定义继承自 *CollectionElement*。[提供的源码](https://github.com/modularml/mojo/blob/nightly/stdlib/src/builtin/value.mojo#L224)。

- **List 的需求和未来改进**：@clattner 建议 **List** 应该主要要求可移动（movable）元素，未来可能会简化为 *AnyType* 或更少的要求。另一位用户提出了一个具有各种功能的 List struct 的替代实现。

- **CollectionElementNew 的迁移问题**：用户讨论了迁移到 *CollectionElementNew* 以及缺少 *ComparableCollectionElementNew* 所带来的复杂性。这表明系统内部正在进行持续的改进。

- **Mojo 对 trait objects 的支持**：讨论了 Mojo 是否会支持 trait objects，这对于创建类似于 Python 的异构列表（heterogeneous lists）至关重要。@jack.clayton 确认该功能已在计划中，但尚未确定具体目标日期。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://doc.rust-lang.org/book/ch17-02-trait-objects.html">Using Trait Objects That Allow for Values of Different Types - The Rust Programming Language</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/builtin/value.mojo#L224)">mojo/stdlib/src/builtin/value.mojo at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1247290220228120808)** (50 messages🔥): 


- **Ollama 模型在 bind_tools 上遇到困难**：一位社区成员询问 LangChain 是否支持 Ollama 模型的 bind_tools，已确认通过 `OllamaFunctions` 类支持，而非 `ChatOllama`。引用了一个 [GitHub issue](https://github.com/langchain-ai/langchain/issues/21479) 以获取更多上下文。

- **构建多功能客户支持助手**：社区讨论集中在利用 Llama3、LangChain 等 LLM 以及用于用户验证和概率计算等操作的工具来创建客户支持助手。分享了 Python 代码示例，展示了如何链式调用模型和自定义工具。

- **在 SQL Agent 中处理聊天上下文**：成员们讨论了在 Agent（特别是 SQL 和 CSV Agent）中维护聊天上下文/历史记录的问题。分享了一个使用 `ConversationBufferMemory` 的代码片段，但指出了不支持 kwargs 的问题。

- **使用 embeddings 对大型文本数据集进行分类**：一位成员寻求帮助，希望使用 OpenAI 对 10,000 条自由文本回答进行分类，并讨论了使用 embeddings 和 LangChain 的方案。另一位成员建议结合 embeddings 进行 Prompt Engineering 以简化流程。

- **持久化聊天机器人记忆**：一位用户尝试按照 LangChain 聊天机器人教程进行持久化记忆和服务器设置，但在提供的 `server.py` 代码中遇到了错误。讨论中未确定错误的来源。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/langchain-ai/langchain/issues/21479>).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/chatbot/">Build a Chatbot | 🦜️🔗 LangChain</a>：概览</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/llms/ollama/#via-langchain>)">Ollama | 🦜️🔗 LangChain</a>：Ollama 允许你在本地运行开源大语言模型，如 Llama 2。</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#langgraph>).">Conceptual guide | 🦜️🔗 LangChain</a>：本节包含 LangChain 核心部分的介绍。</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#tools>).">Conceptual guide | 🦜️🔗 LangChain</a>：本节包含 LangChain 核心部分的介绍。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1247493648677863444)** (1 messages): 

- **LangServe 端点报错**：一位成员询问如何将 human 和 AI 消息传递给 LangServe 端点，因为他们的服务器在尝试时返回了 *HTTP 500 error*。他们提供了一个代码片段，展示了使用 **RemoteRunnable** 以及来自 `@langchain/core` 的 **HumanMessage** 和 **AIMessage** 的方法。

### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1247277160360579184)** (4 messages): 

- **使用 `ChatPromptTemplate.partial` 进行选择性占位符替换**：一位成员讨论了使用 `ChatPromptTemplate.partial` 将部分占位符替换为给定文本，同时保留其他占位符通过 `Runnable.invoke` 处理的方法。他们指出该方法不适用于 `SystemMessagePromptTemplate`，这引起了一些困惑。
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1247276747225825350)** (7 messages): 

- **自动化聊天分析器亮相**：一位成员宣布他们的**自动化聊天分析器**可以从任何规模的消息列表中提取问答（Q&A）。该工具旨在创建一个易于手动编辑且计算资源需求极低的纯文本文件。

- **CrewAI News-Crew 项目**：一位成员分享了他们使用 [CrewAI 和 LangChain 工具](https://github.com/touhi99/tldr-ai-news-crew)实现新闻团队项目的初步方法。他们欢迎对这个正在进行中的项目提出建议和反馈，该项目整合了多个 Agent 和语音模态。

- **使用 LangChain 进行动态工具调用**：分享了一段名为 [Dynamic Tool Calling with Visual Agents & LangChain](https://youtu.be/jBbLxbVDaM4?si=b4hMnqYzme11gshL) 的 YouTube 视频，展示了如何将 OpenAI 聊天模型与以 JavaScript 函数实现的动态工具结合使用。

- **用于数据分析的 EDA GPT**：向数据分析师发出邀请，探索名为 [EDA GPT](https://eda-gpt-24.streamlit.app/EDA_GPT) 的项目。该项目旨在通过 GPT 模型界面辅助进行探索性数据分析（EDA）。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/jBbLxbVDaM4?si=b4hMnqYzme11gshL">Dynamic Tool Calling with Visual Agents &amp; LangChain</a>：在此示例中，我使用了一个 OpenAI 聊天模型（来自 LangChain），并提供了一个以普通 JavaScript 函数实现的动态工具，并向 AI 提问...</li><li><a href="https://github.com/touhi99/tldr-ai-news-crew">GitHub - touhi99/tldr-ai-news-crew: Experiment repo to learn with CrewAI and make life easier with Agents</a>：学习 CrewAI 并通过 Agent 简化生活的实验仓库 - touhi99/tldr-ai-news-crew</li><li><a href="https://eda-gpt-24.streamlit.app/EDA_GPT">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1247345536227479685)** (2 messages): 

- **通过 LangSmith Hub 变革工作流**：一篇 Medium 文章讨论了 **LangChain Hub** 如何通过集中管理 Prompt 来彻底改变 JavaScript 工程师的 Prompt 管理方式，类似于 GitHub 如何变革代码协作。文章强调了用于上传、组织和版本控制 Prompt 的**直观界面**，从而简化工作流并促进创新。[点击此处阅读更多](https://medium.com/@kenzic/transform-your-workflow-with-langsmith-hub-a-game-changer-for-javascript-engineers-183af7cc4e31)。
  
- **使用 LangSmith 评估 LLM 系统**：一段名为“*让 Agent 便宜、快速且好用 10 倍？*”的 YouTube 视频深入探讨了使用 LangSmith 进行 **LLM 系统评估**。该视频旨在教观众如何构建更好的 Agent，并包含其他资源的链接，例如免费的 HubSpot 报告。[观看视频](https://youtu.be/MGkByeDm-90?si=14AxwOfkULwC4a2U)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@kenzic/transform-your-workflow-with-langsmith-hub-a-game-changer-for-javascript-engineers-183af7cc4e31">Transform Your Workflow with LangSmith Hub: A Game-Changer for JavaScript Engineers</a>：零散的 AI Prompt 是否减慢了你的开发进程？了解 LangChain Hub 如何变革你的工作流，让 Prompt 管理变得...</li><li><a href="https://youtu.be/MGkByeDm-90?si=14AxwOfkULwC4a2U">&quot;Make Agent 10x cheaper, faster &amp; better?&quot; -  LLM System Evaluation 101</a>：LLM 系统评估入门 - 构建更好的 Agent。获取关于如何利用 AI 找工作的免费 HubSpot 报告：https://clickhubspot.com/fo2🔗 链接 - 在 Twitter 上关注我：h...
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1247286969537396868)** (43 条消息🔥): 

- **所有后端的 Bitshift 引发辩论**：成员们讨论了将来自 [此 PR](https://github.com/tinygrad/tinygrad/pull/4728) 的 bitshift 操作合并到所有后端（而不只是 assembly）中。人们对与传统的除法和乘法操作相比潜在的性能提升表示担忧。
- **'gpuctypes' 缺失设备测试**：一位成员询问为什么没有添加来自 'gpuctypes' 仓库的设备测试，并引用了 [test_cuda.py](https://github.com/tinygrad/gpuctypes/blob/c4c4394239dce4d4ecfe7016ca268c49cb2d02a4/test/test_cuda.py#L36)。
- **Hotz 的演讲计划**：George Hotz 分享了他的演讲计划，包括幻灯片、代码演示和现场 Demo。他的目标是提高可读性，并单独解释 tinygrad 的堆栈如何直接与 kernel 通信，而无需依赖 Nvidia 的 CUDA 库。
- **Lean 机械化悬赏引发困惑**：成员们试图澄清与 Lean 中可合并 `ShapeTrackers` 相关的悬赏范围。对话强调了缺乏清晰的问题陈述和证明所需的必要条件，并指向了现有的资源如 [仓库中的 ShapeTracker](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/shapetracker.py) 以便更好地理解。
- **最终战略讨论**：有人简要建议讨论 tinygrad 项目的“终局（endgame）”或未来目标，暗示了雄心勃勃的云端集成的潜力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.codeeurope.pl/en/speakers/george-hotz">Code Europe - 波兰最大的科技节</a>：未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/blob/8f749ae0eb75da821637e613a68c9192da474ac2/docs-legacy/reshape_without_symbolic.md">tinygrad/docs-legacy/reshape_without_symbolic.md</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/gpuctypes/blob/c4c4394239dce4d4ecfe7016ca268c49cb2d02a4/test/test_cuda.py#L36">gpuctypes/test/test_cuda.py</a>：用于 HIP, CUDA 和 OpenCL 的 ctypes 包装器。通过在 GitHub 上创建一个账户来为 tinygrad/gpuctypes 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/shapetracker.py">tinygrad/tinygrad/shape/shapetracker.py</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4728">SzymonOzog 的 Bitshift · Pull Request #4728</a>：乘以或除以 2 的幂可以通过简单的 bitshift 执行，对于 cstyle 后端，这由编译器完成，但 assembly 需要显式执行此操作</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4">在所有分支中运行单元测试 · Pull Request #4</a>：未找到描述
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1247281718319386717)** (13 条消息🔥): 

- **为不连续 Tensor 添加 traceback**：用户讨论了在 **tinygrad** 中为不连续（incontiguous）Tensor 提供错误 traceback 的实用性。一位用户建议 *"每次创建新的 `Tensor` 时，都添加一个 'traceback' 属性"*，认为这是对调试的重大增强。

- **提到之前的尝试和 POC**：成员们回忆了关于此功能的先前概念验证（POC），但记不清细节。**UUUVN** 承认在 PR ##3302 中参与过相关的 POC 工作，但缺乏完成它的动力。

- **社区对重新审视该功能的兴趣**：尽管存在挑战，一些成员表示有兴趣贡献力量来完善此功能，以提高工具效率。另一位用户强调，这将是 *"以较小的代码量/性能成本，大幅提升工具的实用性并减少挫败感"*。
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1247305452488757269)** (2 条消息): 

- **LLM 编写 Python 脚本用于结构化 Gmail 数据提取**：在一个 [repo](https://t.co/N2BJ54zr7i) 中，通过向 LLM 提供部分电子邮件子集，使其能够编写一个处理完整数据集的脚本，从而从 Gmail 中提取结构化数据。该技术展示了 LLM 在数据提取任务中智能脚本编写的潜力。
- **Google Gemini 的 100 万 token 上下文窗口演示**：观看集成到 LlamaIndex agent 中的 [Google Gemini](https://t.co/Qg9ydPzBdd) 及其庞大的 100 万 token 上下文窗口的优势。该演示涉及从复杂的异构文档中回答一个多部分问题，展示了更大上下文窗口在增强理解和处理能力方面的强大作用。
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1247271178272510125)** (43 条消息🔥): 

- **Gamecode8 在 LlamaIndex 中寻找 Langchain 的等效功能**：一位用户询问 LlamaIndex 是否有类似 Langchain 的 `HTMLHeaderTextSplitter` 的变体，并提到了在使用 `HTMLNodeParser` 时遇到的困难。在讨论了 `IngestionPipeline` 中的集成后，他们决定实现一个自定义解决方案，并参考了关于自定义转换的 [提供文档](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/transformations/#custom-transformations)。
  
- **Rkhettry 解决了 Chroma 中的节点获取问题**：一位用户最初在从 Chroma 向量存储中的 `VectorStoreIndex` 获取文档时遇到问题并寻求帮助。随后，他们通过直接访问 ChromaDB 集合中的文档解决了该问题。

- **关于在索引中为 LLM 设置 prompt 的澄清**：关于为索引设置 prompt 存在一些困惑；一位用户分享了他们过去如何使用 `LLMPredictor` 设置 prompt。会议澄清了 prompt 并不附加在索引上，并提供了关于如何为 LLM 设置 prompt 的指南 ([链接](https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/?h=prompts#accessingcustomizing-prompts-within-higher-level-modules))。

- **解决 RAG 流水线中的特定领域上下文问题**：一位用户询问如何在不进行 fine-tuning 的情况下有效地向 RAG 流水线注入特定领域上下文。建议包括重写用户查询或修改 prompt，但同时也指出，对于极其小众的主题，fine-tuning 或广泛的数据清洗是必要的。
  
- **元数据增强文档索引中的检索**：讨论了向文档添加元数据（如销售日期）如何帮助改进检索。分享了 [元数据提取文档](https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC/) 的链接，表明这种做法有助于消除相似文档的歧义。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/MGkByeDm-90?si=14AxwOfkULwC4a2U">&quot;让 Agent 便宜、快速且好用 10 倍？&quot; - LLM 系统评估 101</a>：LLM 系统评估 101 - 构建更好的 agent。获取关于如何使用 AI 找到工作的免费 HubSpot 报告：https://clickhubspot.com/fo2🔗 链接 - 在 twitter 上关注我：h...</li><li><a href="https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/HTML_header_metadata/">按 HTML 标题拆分 | 🦜️🔗 LangChain</a>：描述与动机</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/transformations/#custom-transformations">Transformations - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/?h=prompts#accessingcustomizing-prompts-within-higher-level-modules">在高级模块中访问/自定义 Prompts - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC/">提取元数据以实现更好的文档索引和理解 - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1247266522234093691)** (8 messages🔥): 

- **Phi-3 模型登上排行榜**："_philschmid_" 宣布 Phi-3 Medium (14B) 和 Small (7B) 模型现已登上 [@lmsysorg 排行榜](https://fxtwitter.com/_philschmid/status/1797700161226838362)。Medium 的排名接近 GPT-3.5-Turbo-0613，而 Small 则与 Llama-2-70B 和 Mistral 微调版本相当。
- **Dylan 输掉了赌注**：一位成员幽默地指出 "dylan lost his bet"，反映了最近的预测。
- **声誉高于赌注**：另一位成员对没有参与赌注表示遗憾，表示 "I’m in it for the reputation gain"（我是为了获得声誉），并指出所有的赌注都变成了为公益事业进行的捐赠赌注。

**提到的链接**：<a href="https://fxtwitter.com/_philschmid/status/1797700161226838362">Philipp Schmid (@_philschmid) 的推文</a>：Phi-3 Medium (14B) 和 Small (7B) 模型登上了 @lmsysorg 排行榜！😍 Medium 的排名接近 GPT-3.5-Turbo-0613，但落后于 Llama 3 8B。Phi-3 Small 接近 Llama-2-70B 和 Mistral 微调版本。...

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1247582450197201056)** (29 条消息🔥): 

- **OpenAI 员工表达安全担忧**：一群现任和前任 OpenAI 员工发表了一封[公开信](https://righttowarn.ai/)，指出 AI 行业缺乏监管和举报人保护机制。他们认为 AI 公司在避免有效监管方面有着强大的财务动机。
- **AGI 预测与工业规模化**：Leopold Aschenbrenner 的[文章](https://situational-awareness.ai/)讨论了计算集群的巨大规模扩张，并预测 AGI 能力将在 2025/26 年实现。这引发了关于这种指数级 Scaling 趋势是否会持续下去的辩论。
- **信息泄露导致 OpenAI 裁员**：据报道，Leopold Aschenbrenner 因泄露专有信息而被解雇，参考了 @steph_palazzolo 的[独家报道](https://www.theinformation.com/articles/openai-researchers-including-ally-of-sutskever-fired-for-alleged-leaking)。这次泄密引发了关于商业机密对 AI 国家安全重要性的讨论。
- **质疑 AGI 的永久 Scaling**：Natolambert 批评了过度依赖趋势外推来预测仅通过增加模型大小就能实现 AGI。他认为，无法保证对数线性图趋势在不遇到“瓶颈点”的情况下会一直持续。
- **Daniel Kokotajlo 的股权纠纷**：Kevin Roose 报道称，Daniel Kokotajlo 在 OpenAI 的股权面临风险，除非他签署一份 NDA（保密协议），但他拒绝了，这导致了秘密 NDA 的公开曝光。@KelseyTuoc 详细阐述了[细节](https://www.nytimes.com/2024/06/04/technology/openai-culture-whistleblowers.html?unlocked_article_code=1.xE0._mTr.aNO4f_hEp2J4&smid=nytcore-ios-share&referringSource=articleShare&sgrp=c-cb)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://situational-awareness.ai/">Introduction - SITUATIONAL AWARENESS: The Decade Ahead</a>：Leopold Aschenbrenner，2024 年 6 月。你可以先在旧金山看到未来。在过去的一年里，城里的谈论话题已经从 100 亿美元的计算集群转向 1000 亿美元的集群，再到万亿...</li><li><a href="https://x.com/steph_palazzolo/status/1798041967118750118">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：@leopoldasch 分享了他因“泄密”而被 OpenAI 解雇的原因（根据我们的独家报道：https://www.theinformation.com/articles/openai-researchers-including-ally-of-sutskever-fired-for-alleged-le...</li><li><a href="https://x.com/natolambert/status/1798073830906486945">来自 Nathan Lambert (@natolambert) 的推文</a>：这会让 AGI Scaling 的信徒们感到紧张吗？仅限错误答案。这是 @TheXeophon 漂亮的 LLM Scaling 趋势线</li><li><a href="https://www.cnbc.com/2024/06/04/openai-open-ai-risks-lack-of-oversight.html">现任和前任 OpenAI 员工警告 AI 的“严重风险”和缺乏监管</a>：一群现任和前任 OpenAI 员工周二发表了一封公开信，描述了对 AI 行业快速发展的担忧。</li><li><a href="https://righttowarn.ai/">A Right to Warn about Advanced Artificial Intelligence</a>：未找到描述</li><li><a href="https://x.com/_sholtodouglas/status/1798052154709852198">来自 Sholto Douglas (@_sholtodouglas) 的推文</a>：一篇精彩的文章，真实地捕捉到了博弈中所有参与者都在遵循的世界观。“到 2027 年实现 AGI 极其可信”——只要趋势线能保持...</li><li><a href="https://x.com/kelseytuoc/status/1798029447662371231?s=46">来自 Kelsey Piper (@KelseyTuoc) 的推文</a>：来自 Kevin 的精彩报道，以及关于 Daniel Kokotajlo 情况的一些新细节：他在 OpenAI 持有价值 170 万美元的股权。他被告知如果不签署就会被取消。他...</li><li><a href="https://x.com/natolambert/status/1798042504635523276">来自 Nathan Lambert (@natolambert) 的推文</a>：我正在听最新的 @dwarkesh_sp 与 @leopoldasch 的对话，我感到非常震惊，听到这么多关于趋势线外推的相同观点，保证我们能实现“AGI”...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1247597857268306090)** (6 条消息): 

- **Scaling Law 嘲讽梗**：一位用户嘲讽地描绘了关于寻找模式的大脑和捕食者的进化对话，最后揶揄了那些相信 Scaling Law 在 2030 年后仍能发挥重要作用的人：*“像老板一样坚信 Scaling Law 在 2030 年后依然有效。”*
- **呼吁大规模参数扩展**：有人幽默地恳求将 AI 模型扩展到 10 万亿参数，并戏剧性地喊道：*“兄弟，拜托了，再来 10 万张 GPU 就好”*。
- **对 AGI 争论的沮丧**：一位用户表达了对 AGI 讨论中两极分化观点的愤怒，批评乐观派和末日论者（doomers）尽管声称遵循 Bayesian 逻辑，却几乎没有“认识论上的不确定性（epistemic uncertainty）”。*“在他们看来，进一步改进的障碍似乎总是微不足道的。”*
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 条消息): 

420gunna: 👍
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1247279508260917329)** (41 条消息🔥): 

- **Torchtune 寻求 AI News 报道**：来自 Torchtune Discord 服务器的一位核心开发者请求将其纳入 AI News 通讯，强调了他们社区关于 SOTA 模型和方法的活跃讨论。服务器邀请链接[在此](https://discord.gg/6cGsKMM5)。
- **AI News 邮件格式问题**：一位成员报告了 ProtonMail 中 AI News 邮件的格式问题，在白色背景下只能看到链接和图像。该问题似乎与 ProtonMail 的深色模式设置有关。
- **自动化播客转录**：一位成员询问播客如何生成带有发言人识别的转录文本，结果显示他们使用了 Google 的 smol-podcaster 工具，并手动替换发言人姓名标签。
- **LiveKit 获得 2250 万美元 A 轮融资**：LiveKit 宣布获得 2250 万美元 A 轮融资，用于构建 AI 的传输层，强调实时语音和视频是未来与计算机交互的方式。[融资过程充满挑战](https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww)，但 GPT-4 的出现证明了对该技术的迫切需求。
- **Twelve Labs 为多模态 AI 筹集 5000 万美元 A 轮融资**：Twelve Labs 筹集了 5000 万美元，以进一步开发其视频理解 AI 基础模型，推出了在单个 API 中支持多模态的 Marengo 2.6 模型。[详情点击这里](https://www.prweb.com/releases/twelve-labs-earns-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tree-diffusion.github.io/">未找到标题</a>: 未找到描述</li><li><a href="https://discord.gg/6cGsKMM5.">Discord - 充满乐趣与游戏的群聊</a>: Discord 是玩游戏和与朋友放松，甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 dsa (@dsa) 的推文</a>: 今天我们宣布 LiveKit 获得 2250 万美元 A 轮融资，用于构建 AI 的传输层。这并非一次轻松的融资。去年年底，我们向投资者推销实时语音和视频将成为...</li><li><a href="https://x.com/itsandrewgao/status/1797739301947748541?s=46&t=2qGo-Hp_MDNyh14F888CkQ">来自 Andrew Gao (@itsandrewgao) 的推文</a>: 未找到描述</li><li><a href="https://x.com/dkokotajlo67142/status/1797994238468407380?s=46&t=JE84TqLviekDnEt8MAT-Eg">来自 Daniel Kokotajlo (@DKokotajlo67142) 的推文</a>: 1/15：4 月，我从 OpenAI 辞职，因为我失去了对该公司在尝试构建通用人工智能（AGI）——“通常比人类更聪明的 AI 系统”时能够表现得负责任的信心...</li><li><a href="https://future.mozilla.org/builders/">Mozilla Builders</a>: 未找到描述</li><li><a href="https://x.com/rowancheung/status/1781732100556591525?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Rowan Cheung (@rowancheung) 的推文</a>: 一个 GPT-4 级别的聊天机器人，完全免费使用，在 Groq 上运行速度超过每秒 800 个 token。我真的被 Llama 3 震撼到了。通过下一条推文中的链接尝试一下。</li><li><a href="https://x.com/rowancheung/status/1781732100556591525?s=46&t=90xQ8sGy6">来自 Rowan Cheung (@rowancheung) 的推文</a>: 一个 GPT-4 级别的聊天机器人，完全免费使用，在 Groq 上运行速度超过每秒 800 个 token。我真的被 Llama 3 震撼到了。通过下一条推文中的链接尝试一下。</li><li><a href="https://x.com/MSFTResearch/status/1797662278394827029">来自 Microsoft Research (@MSFTResearch) 的推文</a>: Aurora，来自 Microsoft Research 的新型 AI 基础模型，可以通过实现更快、更准确的预测，改变我们预测和缓解极端天气事件及气候变化影响的能力...</li><li><a href="https://www.forbes.com/sites/alexkonrad/2024/06/04/inside-silicon-valley-influence-battle-for-ai-future/?sh=5fece9ce2dc4">Vinod Khosla、Marc Andreessen 以及亿万富翁们的 AI 未来之战</a>: 互联网时代的亿万富翁投资者们正陷入一场政策之战，以决定 AI 的未来是走向集中的安全还是无拘无束的进步。</li><li><a href="https://github.com/go-go-golems/geppetto/blob/main/cmd/pinocchio/prompts/code/plantuml/activity.yaml">geppetto/cmd/pinocchio/prompts/code/plantuml/activity.yaml at main · go-go-golems/geppetto</a>: golang GPT3 工具。通过在 GitHub 上创建账户来为 go-go-golems/geppetto 的开发做出贡献。</li><li><a href="https://www.prweb.com/releases/twelve-labs-earns-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html">Twelve Labs 获得由 NEA 和 NVIDIA 的 NVentures 领投的 5000 万美元 A 轮融资，以构建多模态 AI 的未来</a>: /PRNewswire-PRWeb/ -- Twelve Labs，这家视频理解公司今天宣布筹集了 5000 万美元的 A 轮融资，以推动正在进行的...
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1247265025765277768)** (17 条消息🔥): 

- **即使没有成果，参加会议也很有趣**：与会者表达了尽管没有被录用的论文，但仍然享受会议的乐趣，强调了参与本身的价值。
- **在自定义 OrPO 格式化程序上遇到困难**：一名成员寻求帮助，加载用于对预转换数据集进行 token 化处理的自定义 OrPO 格式化程序 Python 脚本。[相关脚本链接](https://discord.channels/1104757954588196865/1117071926926512248/1245037389886521464)。
- **批评医疗 VQA 中的 AI**：一位成员[分享的推文](https://x.com/xwang_lk/status/1797475354745197029?t=nLUioafCJbCenSnEQ8xfIw&s=19)批评了 GPT-4V 和 Gemini Pro 等最先进模型在医疗 VQA 任务中的表现甚至不如随机，并引入了 ProbMed 数据集来评估性能。讨论中提到了视觉 LLM 在医疗图像诊断方面的不足。
- **寻求 arXiv 背书**：一位成员请求在 arXiv 的 cs.LG 类别中获得背书，但随后通过使用其组织邮箱解决了该问题。

**提到的链接**：<a href="https://x.com/xwang_lk/status/1797475354745197029?t=nLUioafCJbCenSnEQ8xfIw&s=19">来自 Xin Eric Wang (@xwang_lk) 的推文</a>：我们真的能在医疗图像诊断等关键领域信任 AI 吗？不，它们甚至比随机表现还差。我们最新的研究《比随机还差？一种令人尴尬的简单探测评估...》

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1247513942801125436)** (13 messages🔥): 

- **QLoRA 训练输出问题**：一名成员报告了一个问题，即尽管完成了训练，但无法从 QLoRA 输出开始。他们请求紧急协助以解决此事。

- **关于未下载模型进行 LoRA 训练的解释**：一位用户询问为什么在没有预先下载模型的情况下，他们的 LoRA 训练仍能无错进行。另一位成员澄清说，如果本地未找到模型，训练脚本会自动处理从 Hugging Face Model Hub 的下载。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=022a3d10-bf15-408b-b525-168f5f199dd0)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e364cc67-a6a9-4aa8-9173-f7bfb1d02469)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1247490831422853130)** (13 messages🔥): 

- **Cohere 的 Artificial Ivan V.4.0 大获成功**：Cohere 的项目 "Artificial Ivan" 目前已更新至 4.0 版本，可以排除代码故障。*“我们还在某个阶段自动化了他的黄金思想”*，并附带了 [affirmations](https://amorisot-cohere-demos-onceuponatime-x1w9hn.streamlit.app/affirmations) 的链接。
- **Real Ivan 的退休梦想**：有一条关于真实的 Ivan 在 35 岁退休的幽默评论，暗示对项目进展感到满意。
- **Artificial Ivan 的其他项目**：澄清了提到的应用不仅仅是为 Ivan 准备的，而是由另一位用户领导的各种项目的总称。
- **Cohere Discord 非常活跃**：一位用户注意到 Cohere Discord 的活跃度远高于 LangChain，并开玩笑地称其为 *“长期在线的 AI 实验室”*。

**提及的链接**：<a href="https://amorisot-cohere-demos-onceuponatime-x1w9hn.streamlit.app/affirmations">未找到标题</a>：未找到描述

  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1247493689798955110)** (1 messages): 

- **使用 Llama.cpp 和 LangChain 配置 Aya 23**：分享了一个将 **Aya 23** 作为 LLM，并结合 **Llama.cpp** 和 **LangChain** 使用的项目。用户特别寻求关于如何添加停止符（stop）以防止系统生成超出对话的内容，建议使用类似 "\n" 的符号。

- **Aya 23 项目的代码实现**：用户提供了详细的代码样本，展示了 Aya 23 与 Prompt 和对话记忆（conversation memory）的集成设置。该代码旨在实现西班牙语的简洁回答，并为模型的性能和循环用户交互设置了参数。
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1247314853354410036)** (9 messages🔥): 

- **CUDA 重新编译问题已解决**：在遇到 CUDA 错误时，一名成员发现使用 `--recompile --gpu NVIDIA` 解决了他们的问题。他们指出 `-ngl 9999` 必须放在 `--recompile` 之后才能有效工作。

- **社区协助故障排除**：一名寻求 CUDA 重新编译帮助的成员收到了一个 [GitHub 链接](https://github.com/Mozilla-Ocho/llamafile/blob/main/build/llamafile-upgrade-engine)，但随后通过阅读 `--help` 选项自行解决了问题。这一解决得到了大家的赞赏和友谊。

- **饼干与情谊**：社区强调了相互支持、互相学习，并幽默地提到他们也是为了饼干而来的。

- **CPU 向量计算的未来**：分享了一段名为 "The Magic of RISC-V Vector Processing" 的 YouTube 视频，深入探讨了新批准的 1.0 RISC-V Vector Specification 及其应用。[观看视频](https://www.youtube.com/watch?v=Ozj_xU0rSyY)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=Ozj_xU0rSyY">The Magic of RISC-V Vector Processing</a>：1.0 RISC-V Vector Specification 现已正式批准，首批使用新规范的芯片开始上市。我将介绍其用途...</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/blob/main/build/llamafile-upgrade-engine">llamafile/build/llamafile-upgrade-engine at main · Mozilla-Ocho/llamafile</a>：通过单个文件分发和运行 LLM。欢迎在 GitHub 上为 Mozilla-Ocho/llamafile 的开发做出贡献。
</li>
</ul>

</div>
  

---



**提及的链接**：<a href="https://tinyurl.com/j8z6s8ka">Qwak 举办的 Infer Summer ‘24 | AI 和 ML 背后的工程学</a>：Qwak 举办的 Infer Summer ‘24 邀请 AI 领袖分享全球领先公司如何在生产中使用 ML 和 AI。2024 年 6 月 26 日上午 11:00 (EDT) 准时参加直播。

  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1247488626326245398)** (1 messages): 

- **xLSTM 源代码发布**：Dr. Tristan Behrens 宣布发布 xLSTM 的源代码，并兴奋地表示 *"放下手头的一切，NXAI 已经发布了官方版本"*。该公告发布在 [LinkedIn](https://www.linkedin.com/posts/dr-tristan-behrens-734967a2_drop-everything-nxai-has-released-the-official-activity-7203659602628935682-GwOA) 上。
  

---



---



---



---



{% else %}


> 由于电子邮件限制，各频道的详细细分已截断。
> 
> 如果您想查看完整细分，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}