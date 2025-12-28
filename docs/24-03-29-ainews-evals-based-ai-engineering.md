---
companies:
- openai
- mistral-ai
- x-ai
- llamaindex
date: '2024-03-29T22:20:49.586743Z'
description: '**Hamel Husain** 强调了在 AI 产品开发中全面评估（evals）的重要性，并指出评估、调试和行为改变是关键的迭代步骤。**OpenAI**
  发布了一个语音引擎演示，展示了仅凭少量样本即可实现的高级语音克隆技术，这引发了安全方面的担忧。Reddit 上的讨论介绍了一些新模型，包括 **Jamba**（结合了
  Transformer-SSM 与 MoE 的混合架构）、**Bamboo**（基于 Mistral 的 7B 高稀疏度大语言模型）、**Qwen1.5-MoE**（高效的参数激活）以及
  **Grok 1.5**（具备 128k 上下文长度，在代码生成方面超越了 GPT-4）。量化技术的进展包括：**1-bit Llama2-7B** 模型表现优于全精度模型，以及支持
  GPTQ/AWQ/HQQ 方法的 **QLLM** 量化工具箱。'
id: d2ec8f17-55b8-4970-b09c-76b6c3739617
models:
- jamba
- bamboo
- qwen-1.5-moe
- grok-1.5
- llama2-7b
original_slug: ainews-evals-based-ai-engineering
people:
- hamel-husain
- alec-radford
title: '**基于评估的 AI 工程**


  （或者：**评估驱动的 AI 工程**）'
topics:
- evaluation
- fine-tuning
- prompt-engineering
- voice-cloning
- quantization
- model-optimization
- code-generation
- context-windows
---

 

Alec Radford 从不失手。我们也[很喜欢 Dwarkesh 与 Sholto 和 Trenton 的播客](https://twitter.com/dwarkesh_sp/status/1773381318266786180?t=90xQ8sGy63D2OtiaoGJuww)。


---

**目录**

[TOC] 


---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。评论抓取功能尚未实现，但即将推出。


**新模型与架构**：

- **Jamba**：首个基于 Mamba 的生产级模型，采用 Transformer-SSM 混合架构，性能超越同尺寸模型。([Introducing Jamba - Hybrid Transformer Mamba with MoE](https://www.reddit.com/r/LocalLLaMA/comments/1bpx9sh/introducing_jamba_hybrid_transformer_mamba_with/))
- **Bamboo**：一款基于 Mistral 权重的新型 7B LLM，具有 85% 的激活稀疏性，通过 CPU+GPU 混合模式可实现高达 4.38 倍的加速。([Introducing Bamboo: A New 7B Mistral-level Open LLM with High Sparsity](https://www.reddit.com/r/LocalLLaMA/comments/1bpqzck/introducing_bamboo_a_new_7b_mistrallevel_open_llm/))
- **Qwen1.5-MoE**：以 1/3 的激活参数量达到 7B 模型的性能。([Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters](https://qwenlm.github.io/blog/qwen-moe/))
- **Grok 1.5**：在 HumanEval 代码生成评测中超越 GPT-4 (2023)，并拥有 128k 上下文长度。([Grok 1.5 now beats GPT-4 (2023) in HumanEval (code generation capabilities), but it's behind Claude 3 Opus](https://i.redd.it/1jl3qaqzl6rc1.png), [X.ai announces grok 1.5 with 128k context length](https://x.ai/blog/grok-1.5))

**量化与优化**：

- **1-bit Llama2-7B**：深度量化模型，表现优于更小尺寸的全精度模型，其中 2-bit 模型在特定任务上超越了 fp16。([1-bit Llama2-7B Model](https://www.reddit.com/r/LocalLLaMA/comments/1bptl1w/1bit_llama27b_model/))
- **QLLM**：通用的 2-8 bit 量化工具箱，支持 GPTQ/AWQ/HQQ，可轻松转换为 ONNX。([Share a LLM quantization REPO , (GPTQ/AWQ/HQQ ONNX ONNX-RUNTIME)](https://www.reddit.com/r/LocalLLaMA/comments/1bqeqr7/share_a_llm_quantization_repo_gptqawqhqq_onnx/))
- **Adaptive RAG**：一种检索技术，根据 LLM 反馈动态调整检索文档的数量，将 Token 成本降低了 4 倍。([Tuning RAG retriever to reduce LLM token cost (4x in benchmarks)](https://www.reddit.com/r/LocalLLaMA/comments/1bq2g3e/tuning_rag_retriever_to_reduce_llm_token_cost_4x/), [Adaptive RAG: A retrieval technique to reduce LLM token cost for top-k Vector Index retrieval [R]](https://www.reddit.com/r/MachineLearning/comments/1bq3hwb/adaptive_rag_a_retrieval_technique_to_reduce_llm/))

**Stable Diffusion 增强**：

- **Hybrid Upscaler 工作流**：结合了追求最高质量的 SUPIR 与追求速度的 4x/16x 快速放大器。([Hybrid Upscaler Workflow](https://www.reddit.com/gallery/1bpqo5z))
- **IPAdapter V2**：更新以使旧工作流适配新版本。([IPAdapter V2 update old workflows](https://www.reddit.com/r/StableDiffusion/comments/1bpz0zn/ipadapter_v2_update_old_workflows/))
- **Krita AI Diffusion 插件**：图像生成体验非常有趣。([Krita AI Diffusion Plugin is so much fun (link in thread)](https://v.redd.it/1k6360roj7rc1))

**幽默与迷因**：

- **AI 狮子迷因**：一张幽默的狮子图片。([AI Lion Meme](https://i.redd.it/r4n37xjj66rc1.jpeg))
- **“不，船长！你没明白的是这些植物是活的！！”**：幽默的文本转视频生成。(["No, captain! What you don't get is that these plants are Alive!! & they are overwhelming us... what's that? Yes! Marijuana Leaves! Hello Captain? Hello!!" (TEXT TO VIDEO- SDCN for A1111, no upscale)](https://v.redd.it/e33z5vu2t5rc1))
- **“在动物园拍摄动物”**：一个人拍摄动物的幽默图片。(["Filming animals at the zoo"](https://i.redd.it/c8fev6j9a4rc1.jpeg))

---

# AI Twitter 摘要回顾

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**AI 模型与架构**

- **新开源模型发布**：[@AI21Labs](https://twitter.com/AI21Labs/status/1773350888427438424) 推出了 Jamba，这是一个基于 Mamba 架构的开源 SSM-Transformer 模型，实现了 **3 倍的吞吐量，并能在单个 GPU 上容纳 140K 上下文**。[@databricks](https://twitter.com/databricks/status/1773355487578280317) 发布了 DBRX，在 **MMLU、HumanEval 和 GSM 8k 等基准测试中创下了新的 SOTA**。[@AlibabaQwen](https://twitter.com/AlibabaQwen/status/1773370876496334962) 发布了 Qwen1.5-MoE-A2.7B，这是一款 **拥有 2.7B 激活参数的 MoE 模型，性能可达到 7B 模型水平**。
- **量化技术进展**：[@maximelabonne](https://twitter.com/maximelabonne/status/1773268783139881073) 分享了一个 Colab 笔记本，对比了 **使用 HQQ + LoRA 量化的 FP16 与 1-bit LLama 2-7B 模型**，其中 SFT 显著提升了量化模型的表现，使得更大的模型能够适配更小的内存占用。
- **混合专家 (MoE) 架构**：[@osanseviero](https://twitter.com/osanseviero/status/1773360705682411750) 概述了不同类型的 MoE 模型，包括 **预训练 MoE (pre-trained MoEs)、上循环 MoE (upcycled MoEs) 和 FrankenMoEs**，这些模型因其效率和性能优势而日益受到关注。

**AI 对齐与事实性**

- **评估长文本事实性**：[@quocleix](https://twitter.com/quocleix/status/1773406864589672521) 介绍了一个新的数据集、评估方法和聚合指标，通过 **搜索增强事实性评估器 (SAFE)，利用 LLM Agent 作为自动评估员** 来评估长文本 LLM 回复的事实性。
- **分步直接偏好优化 (sDPO)**：[@_akhaliq](https://twitter.com/_akhaliq/status/1773571320627790178) 分享了一篇提出 sDPO 的论文，这是用于对齐微调的直接偏好优化 (DPO) 的扩展，它 **将现有的偏好数据集进行拆分并以分步方式利用它们**，其表现优于其他参数量更多的热门 LLM。

**AI 应用与演示**

- **AI 驱动的 GTM 平台**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1773446960889938055) 与 @copy_ai 合作创建了一个 AI 驱动的进入市场 (GTM) 平台，提供 **实时市场洞察**，Copy AI 用户可免费获得 6 个月的 Perplexity Pro。
- **具有长期记忆的日记应用**：[@LangChainAI](https://twitter.com/LangChainAI/status/1773381734215958971) 推出了 LangFriend，这是一款利用 **记忆功能提供个性化体验** 的日记应用，目前已开放试用，面向开发者的记忆 API 也在开发中。
- **AI 数字人与视频生成**：[@BrivaelLp](https://twitter.com/BrivaelLp/status/1773442026966585484) 展示了使用 Argil AI 进行的实验，创建了 **完全由 AI 生成的内容，视频中 AI 版巴拉克·奥巴马正在教授量子力学**，展示了 AI 彻底改变用户生成内容 (UGC) 和社交内容创作的潜力。

**AI 社区与活动**

- **因对 AI 的贡献获封爵士**：@GoogleDeepMind 的首席执行官兼联合创始人 [@demishassabis](https://twitter.com/demishassabis/status/1773421805929202165) 因过去 15 年 **对人工智能领域的杰出贡献，被英国国王授予爵士勋衔**。
- **AI 电影节**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1773363405715321249) 宣布 **第二届年度 AI 电影节将于 5 月 1 日在洛杉矶举行**，旨在展示顶尖的 AI 电影作品。

---

# 第 0 部分：摘要之摘要之摘要

**1) 新 AI 模型发布与架构**：

- **[AI21 Labs 发布 Jamba](https://www.ai21.com/blog/announcing-jamba)**，这是一款混合 **SSM-Transformer 模型**，具有 **256K 上下文窗口**、**12B 激活参数**，在长上下文场景下具有 **3 倍吞吐量**。权重在 **Apache 2.0 许可证**下开源。
- **[Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)**，一款基于 **Transformer 的 MoE 模型**，仅凭 **2.7B 激活参数**即可匹配 7B 模型的性能。
- **[LISA 算法](https://arxiv.org/abs/2403.17919)** 实现了在 **24GB GPU 上进行 7B 参数微调**，在指令任务中表现优于 LoRA 和全量训练。
- **SambaNova 发布 Samba-1**，这是一款**专家组合 (CoE) 模型**，声称降低了计算需求并提高了性能，尽管存在透明度方面的担忧。

**2) 开源协作与社区项目**：

- **[Modular 开源 Mojo 标准库](https://modul.ar/open-source-blog)**，采用 Apache 2 许可证，提供每日构建版本，并举行了涵盖更新内容的 [社区直播](https://www.youtube.com/watch?v=PL71FV2KKHE)。
- **[LlamaIndex 博客](https://t.co/MCnmhQGS7g)** 分享了 Cohere 针对 **RAG 流水线的 Int8 和二进制嵌入 (Binary Embeddings)** 优化方案，以减少内存占用和成本。
- 社区展示包括利用 Unsloth notebook 训练的 **[Tinyllama](https://ollama.com/pacozaa/tinyllama-alpaca-lora)** 和 **[Mischat](https://ollama.com/pacozaa/mischat)** 模型。
- **[OpenGPTs 项目](https://github.com/langchain-ai/opengpts)** 开设了专门的 <#1222928565117517985> 频道，用于在该开源平台上进行协作。

**3) 模型评估、基准测试与数据集**：

- 讨论了现有 AI 基准测试的可靠性，例如包含人类评估的 [HuggingFace Chatbot Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)。
- 分享了长上下文数据集，如 HuggingFace 上的 [MLDR](https://huggingface.co/datasets/Shitao/MLDR)，用于训练需要超长序列的模型。
- 计划在 Capybara 数据集上进行 DiscoLM、Occiglot、Mixtral、GPT-4、DeepL、Azure Translate 之间的**翻译模型对决**。
- **[Shure-dev 的使命](https://shure-dev.github.io/)** 是策划高质量的 **LLM 研究论文**，作为这个快速演进领域的资源库。

**4) 本地 LLM 部署与硬件优化**：

- 关于使用 LM Studio、Open Interpreter 以及 RTX 3090 GPU 等硬件**在本地运行 LLM** 的指南。
- 关于**电源功率选择 (1200-1500W)** 和高效 **GPU 功耗限制 (RTX 4090 为 350-375W)** 的建议。
- 讨论了使用 QLoRA 和 LoRA 训练 LLaMa 7B 等大模型时的 **VRAM 需求**。
- 建议使用 **[Windows Subsystem for Linux (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/install)** 在 Windows 游戏机上运行 CUDA/Torch。

---

# 第 1 部分：Discord 高层级摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **使用 Turbo 技术增强 SD3**：社区对 [SD3 Turbo](https://arxiv.org/pdf/2403.12015.pdf) 充满热情，它能提供类似于 SDXL Lightning 的高细节模型，讨论暗示其在 12GB VRAM 系统上表现出色。

- **平衡 AI 艺术的性能与易用性**：辩论集中在为了更广泛的易用性而可能缩减 SD3 功能的问题上；建议包括在发布时提供 ControlNet 和微调工具以赋能用户。

- **深入探讨使用 SD3 训练 LoRA**：研究了使用 24GB VRAM 通过 SD3 训练 LoRA 模型的可行性，但关于这些工具在 SD3 发布后如何过渡的问题仍未得到解答。

- **小显存 GPU 挑战大训练目标**：出现了在 16GB VRAM GPU 上成功训练较小 LoRA 模型的案例，强调了激进的训练率和网络丢弃 (Network dropouts) 等优化手段。

- **Arc2Face 成为焦点**：[Arc2Face](https://arc2face.github.io/) 以其面部操作能力吸引了成员，同时也引发了关于“外星技术”的调侃和数据集审查争议。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**DBRX 备受瞩目**：**DBRX** 是 Databricks 推出的一款重量级语言模型，在 Perplexity Labs 的表现惊艳，在解决数学和编程挑战的任务中超越了 GPT-3.5，并与 Gemini 1.0 Pro 旗鼓相当。目前 [Perplexity Labs](https://labs.pplx.ai) 已开放 DBRX 的免费测试，向 AI 爱好者们发出了挑战。

**Copy AI 与 Perplexity 结盟**：Copy AI 平台与 Perplexity 先进 API 的融合预示着市场进入（GTM）策略的变革，为实时市场洞察指明了道路。对于 Copy AI 的用户，合作将提供半年的 Perplexity Pro 访问权限，详情已在双方合著的 [博客文章](https://www.copy.ai/blog/copy-ai-perplexity-purpose-built-partners-for-gtm-teams) 中说明。

**追求完美的 Perplexity 搜索**：用户对 Perplexity 搜索功能中**学术聚焦模式（academic focus mode）**时好时坏的表现感到困惑，并对其间歇性故障表示不解。讨论集中在 **Pro Search** 空间的改进以及文件来源的争议上，重点讨论了可能采用 **RAG** 或 **GPT-4-32k** 技术进行多样化文件处理的可能性。

**探索增强型 Scratchpad 策略**：社区交流了如何发挥 Perplexity 的最佳性能；一位用户演示了如何使用 `<scratchpad>` XML 标签，航天爱好者们则向 AI 提出了关于 Starship 和航天学的问题。用户还提出了一些金融类查询，探究亚马逊的资金动向和 FTX 难题。

**API 的奇遇与波折**：关于 Perplexity AI API 行为不可预测的疑问层出不穷，有时在寻找答案时，搜索结果会在 Web 端与 API 的差异中丢失，偏离了 Web 界面所承诺的稳定性。对于渴望参与 Beta 功能测试（包括备受期待的 URL 引用功能）的用户，可以直接前往 [申请 Beta 功能](https://perplexity.typeform.com/to/j50rnNiB)，这让 API 爱好者们充满期待。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Jamba 激发 AI 讨论**：AI21 Labs 发布 [Jamba](https://www.ai21.com/blog/announcing-jamba) 引发了热议，该模型将 Mamba SSM 与 Transformers 相结合。工程师们就 Jamba 的容量、分词（tokenization）以及 Token 计数缺乏透明度等问题展开了辩论。

- **量化模型行为**：围绕如何评估模型的开放式生成展开了讨论。为了提供帮助，社区分享了包括论文《大型语言模型的新兴能力是幻觉吗？》（*Are Emergent Abilities of Large Language Models a Mirage?*，[点击此处阅读](https://arxiv.org/abs/2304.15004)）在内的各种资源，以阐明 Perplexity 和成对排序（pairwise ranking）指标。

- **微调对决**：成员们积极分享了 Mistral 等模型的微调策略，涉及 SFTTrainer 的难点以及对最佳 Epoch 和 Batch Size 的讨论。多 GPU 支持的技术难题以及 bitsandbytes 等依赖项的问题也被提出待解决。

- **Karpathy 的智慧结晶**：Andrej Karpathy 的 [YouTube 深度解析](https://www.youtube.com/watch?v=c3b-JASoPi0) 将 LLM 比作操作系统，引发了关于微调实践和使用原始训练数据重要性的讨论。他对当前的 RLHF 技术表示批评，将其与 AlphaGo 进行了对比，并建议模型训练方法应发生根本性转变。

- **训练尝试与成功经验分享**：大家就 Tokenizer 错误、微调陷阱、检查点（checkpointing）方法以及避免过拟合的最佳实践交换了意见。此外，还就将本地 Checkpoint 推送到 Hugging Face 以及 `hub_token` 和 `save_strategy` 的使用提供了实用指导。

- **引入创新模型实例**：社区展示了新模型，如 [Tinyllama](https://ollama.com/pacozaa/tinyllama-alpaca-lora) 和 Mischat，说明了如何利用 Unsloth 的 Notebook 和 Huggingface 数据集构建新型 LLM。一位成员推广了“AI Unplugged 5”，这是一篇涵盖多样化 AI 主题的博客文章，可在 [AI Unplugged](https://datta0.substack.com/p/ai-unplugged-5-databricks-dbrx-apple) 查看。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AI 的昂贵游乐场**：AI 爱好者对高昂的准入门槛表示担忧，为了确保最佳性能，投资尖端 GPU 或 MacBook 等必要硬件的费用已接近 **$10k**。讨论揭示了对该领域复杂、术语繁多且资源密集性质的“残酷觉醒”。

- **GPU 讨论 —— 功率、性能与细节**：硬件聊天集中在最佳 **电源规格** 上，建议高性能配置使用 **1200-1500W**，并设置高效的功耗限制。此外，人们对 Nvidia K80 等旧硬件的实际性能收益，以及使用旧 iMac 风扇的高级散热技巧也表现出兴趣。

- **模型性能与兼容性挑战**：对话指出了 **VRAM 瓶颈** 和兼容性问题，指出 **32GB VRAM** 对于即将推出的模型已显不足，并讨论了模型性能，其中提到 **Zephyr** 在创意写作任务上优于 **Hermes2 Pro**。用户还在排查 **LM Studio ROCm beta 版本** 对 GPU 的识别和利用问题，并努力为受限的 VRAM 选择合适的 coder 模型。

- **LM Studio 讨论 —— 更新、问题与集成**：最新的 LM Studio **0.2.18** 版本发布引入了新功能和错误修复，用户正在对其进行剖析，强调了 VRAM 显示和版本类型的差异，并请求增加 GPU 和 NPU 监控工具。一位成员分享了将 **LM Studio** 集成到 Nix 的成功案例，[以 PR 形式提供](https://github.com/NixOS/nixpkgs/pull/290399)。此外，一个专为处理 PDF 设计的聊天机器人 **OllamaPaperBot** 仓库已在 [GitHub](https://github.com/eltechno/OllamaPaperBot/blob/main/simplechat.py) 上分享。

- **利用 AI 重新思考抽象**：正在探索使用 AI 提取抽象概念，参考了关于 [Abstract Concept Inflection](https://arxiv.org/pdf/2304.01904.pdf) 的论文。有人请求指导如何选择合适的 Agent 程序接入 **LM Studio**，旨在实现无缝集成。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**语音克隆引发激烈辩论**：围绕 **OpenAI** 的 **Voice Engine** 展开了讨论，一些人对其仅凭 [15 秒音频样本](https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices) 即可生成自然语音的潜力感到兴奋，而另一些人则对该技术的滥用表示伦理担忧。

**困惑的消费者与缺失的模型**：用户对不同应用中实现的各种 **GPT-4** 版本感到困惑，关于模型稳定性和知识截止日期的报告相互矛盾。同时，对 **GPT-5** 的期待非常强烈，但目前尚无具体信息。

**遭遇错误的公式**：多个频道的用户都在努力将 **LaTeX 公式** 转移到 Microsoft Word 中，并提出 **MathML** 作为潜在解决方案。针对特定 AI 任务（如保持 HTML 标签的翻译）进行正确的 Prompt 结构化处理也成为了关注焦点。

**显微镜下的 Metaprompting**：AI 爱好者辩论了 **Metaprompting** 优于直接指令的优点，经验表明结果并不一致。精确的 Prompt 被强调为优化 AI 性能的关键。

**GPT 中的角色扮演阻力**：**gpt-4-0125-preview** 模型在角色扮演 Prompt 方面表现出奇特的行为：当给出示例格式时，AI 拒绝进行角色扮演；但当省略示例时，它又会配合。用户分享了引导 AI 响应的变通方法和策略。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**New Fine-Tuning Frontiers**: LISA，一种**新的微调技术**，在指令遵循任务中表现优于 LoRA 和全参数训练，能够在 24GB GPU 上微调 7B 参数模型。可以通过[已发表的论文](https://arxiv.org/abs/2403.17919)和[其代码](https://github.com/OptimalScale/LMFlow)探索 LISA 的细节和应用。

**Chip Chat heats up**: AI21 Labs 发布了 **Jamba**，这是一款融合了 **Mamba** 架构和 Transformers 的模型，声称在 52B 总参数中拥有 12B 激活参数。与此同时，SambaNova 推出了 **Samba-1**，这是一款 Composition of Experts (CoE) 模型，声称降低了计算需求并提高了性能，尽管透明度问题仍然存在。关于 Jamba 的详细信息可以在其[官方发布页面](https://www.ai21.com/jamba)找到，鼓励通过 [SambaNova 的博客](https://sambanova.ai/blog/accurate-models-at-blazing-speed)对 Samba-1 的性能进行审查。

**Sensitive Data Safety Solutions Discussed**: 讨论了在训练中保护敏感数据的技术，包括 SILO 和差分隐私（differential privacy）方法。对此话题感兴趣的研究人员可以查阅 [SILO 论文](https://arxiv.org/abs/2308.04430)和[差分隐私论文](https://arxiv.org/abs/1607.00133, https://arxiv.org/abs/2110.05679)以获取更多见解。

**Discrepancy Detective Work in Model Weights**: Discord 用户理清了 **Transformer Lens** (`tl`) 和 **Hugging Face** (`hf`) 之间模型权重参数的差异。调试过程涉及利用 `from_pretrained_no_processing` 以避免 Transformer Lens 对预设权重的修改，正如该 [GitHub issue](https://github.com/neelnanda-io/TransformerLens/issues/346) 中所阐述的。

**MMLU Optimization Achieved**: MMLU 任务的效率得到了提升，能够在单次 forward call 中提取多个 logprobs。一位用户报告了在错误的 GPU 配置上尝试加载 **DBRX base model** 时出现的内存分配问题，在意识到节点配置错误后得到了修正。此外，一个旨在改进 **lm-evaluation-harness** 中基于上下文的任务处理的 [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1571) 正等待 CoLM 截止日期后的审查和反馈。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**A Peek at AI21's Transformer Hybrid**: AI21 Labs 推出了 [Jamba](https://www.ai21.com/blog/announcing-jamba)，这是一款具有 256K context window 的变革性 SSM-Transformer 混合模型，其性能挑战了现有模型，并根据 Apache 2.0 许可证开放访问。

**LLMs Gearing Up with MoE**: 工程社区对 [microqwen](https://twitter.com/JustinLin610/status/1773285084025475355)（推测为 Qwen 的更紧凑版本）以及 [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B) 的首次亮相感到兴奋，后者是一款基于 Transformer 的 MoE 模型，承诺以更少的激活参数实现高性能。

**LLM Training Woes and Wins**: 工程师们正在解决 **Deepseek-coder-33B** 的全参数微调问题，探索针对大型书籍数据集的结构化方法，并关注 Hermes 2 Pro 的多轮 Agent 循环（agentic loops）。同时，他们正在深入探讨“hyperstition”在扩展 AI 能力方面的意义，并澄清 LLM 中的 heuristic 与 inference engines。

**RAG Pipelines and Data Structuring Strategies**: 为了提高检索任务的性能和效率，AI 工程师正在探索带有 metadata 的结构化 XML，并讨论 **RAG** 模型。提及 [ragas GitHub 仓库](https://github.com/explodinggradients/ragas)表明 RAG 系统正在持续增强。

**Worldsim, LaTeX, and AI's Cognitive Boundaries**: Worldsim 项目正在交换技巧和资源，例如 [LaTeX 论文的 gist](https://gist.github.com/irl-dan/61e2f45eb1c9a879b3)。工程师们正在考虑 AI 深入研究替代历史场景的潜力，同时仔细区分 LLM 的不同用例。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 通过开源和性能优化获得动力**：Modular 已根据 Apache 2 许可证向开源社区开放了 **Mojo standard library**，并在 MAX 24.2 版本中展示了这一点。增强功能包括对 _generalized complex types_ 的实现、关于 **NVIDIA GPU support** 的研讨会，以及专注于稳定对 MLIR 的支持（其语法将不断演进）。

**Modular 即将发布的公告引发热议**：Modular 通过一系列神秘的推文激发期待，利用 emoji 和时钟图标预示新公告。社区成员正密切关注官方 [Twitter](https://twitter.com/Modular) 账号，以获取有关这一神秘事件的详细信息。

**MAX Engine 的飞跃式进展**：随着 **MAX Engine 24.2** 的更新，Modular 引入了对具有 dynamic input shapes 的 TorchScript 模型的支持及其他升级，详见其 [changelog](https://modul.ar/max-changelog)。围绕使用 BERT 模型和 GLUE 数据集的性能基准测试展开了热烈讨论，展示了相比 static shapes 的进步。

**生态系统在社区贡献和学习中蓬勃发展**：社区项目正与最新的 Mojo 24.2 版本同步，并表现出通过理解 MLIR dialects 进行更深层次贡献的兴趣。Modular 认可这种热情，并计划随着时间的推移透露更多关于 internal dialects 的信息，对复杂的 MLIR 语法采取渐进式披露的方法。

**预告和直播精彩纷呈**：Modular 通过 [YouTube 直播](https://www.youtube.com/watch?v=PL71FV2KKHE) 介绍了他们的最新进展，涵盖了 Mojo stdlib 的开源和 MAX Engine 支持；而 [这里](https://twitter.com/Modular) 的推文预告则让人们对即将发布的公告保持高度期待。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Hugging Face 贡献的量子飞跃**：AI 研究和应用取得了新进展：**HyperGraph Representation Learning** 为数据结构提供了新见解，**Perturbed-Attention Guidance (PAG)** 提升了 diffusion model 的性能，**Vision Transformer model** 被适配用于医学影像应用。HyperGraph 论文在 [Hugging Face](https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05) 上讨论，PAG 的项目详情见其 [项目主页](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/)，Vision Transformer 的详情见 [Hugging Face space](https://huggingface.co/spaces/emre570/google-vit-large-finetuned)。

**Colab 与编程实力**：工程师们分享了从使用 **Colab Pro** 运行 LLM 到使用 **HF professional coder assistant** 提升编程效率的各种工具和技巧。另一位分享了使用 AutoTrain 的经验，并发布了其 [模型的链接](https://huggingface.co/abhishek/autotrain-c71ux-tngfu)。

**模型生成困扰与图像分类器咨询**：一些人面临模型生成无限文本的挑战，建议使用 `repetition penalty` 和 `StopCriterion`。其他人正在寻求关于微调 zero-shot image classifier 的建议，在 #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1222883104981651517) 和 #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1222819278810779658) 等频道分享问题并寻求专业指导。

**社区学习公告**：**reading-group** 频道的下次会议日期已确认，加强了社区协作。感兴趣的各方可以找到 [Discord 邀请链接](https://discord.gg/hqrZjkjJaq?event=1222215283343622276) 参与小组讨论。

**实时 Diffusion 创新**：用于 diffusion models 的 Marigold 深度估计流水线现在包含一个 [LCM function](https://huggingface.co/spaces/prs-eth/marigold-lcm)，并且一项改进允许在 800x800 分辨率下实现 30fps 的实时图像转换。关于 [labmlai diffusion 仓库](https://github.com/labmlai/annotated_deep_learning_paper_implementations) 的提问表明，人们对优化这些模型持续关注。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI 训练语音引发争议**：一些参与者对专业配音演员可能因对 AI 的负面情绪而回避为 [11labs](https://11labs.io/) 等 AI 项目贡献语音表示担忧，并建议在情感深度并非至关重要的训练场景中，业余爱好者的语音可能已经足够。
  
- **Benchmarks 受到质疑**：有观点批评 AI Benchmarks 的可靠性，推荐使用 [HuggingFace Chatbot Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)，因为它包含人类评估环节。这一讨论对 AI 领域普遍采用的 Benchmarking 方法的有效性提出了质疑。

- **Jamba 引发热议**：AI21 Labs 推出了 **Jamba**，这是一款融合了 SSM-Transformer 架构的前沿模型，在 Benchmarks 测试中表现出色。讨论围绕这一创新及其对模型性能的影响展开（[公告链接](https://www.ai21.com/blog/announcing-jamba)）。

- **关于 Diffusion Models 和 Transformers 的困惑与澄清**：一名成员指出一段 YouTube 视频可能误解了 Diffusion Transformers 的工作原理，特别是在 Attention 机制的使用方面（[YouTube 视频](https://www.youtube.com/watch?v=OPqFpm3wksY)）。社区呼吁对 Diffusion Models 中的 Transformers 进行更好的解释，强调需要更简单的“科学语言”拆解。


---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **智能 RAG 优化方案涌现**：[Cohere](https://t.co/MCnmhQGS7g) 正在通过引入 Int8 和 Binary Embeddings 来增强 **Retrieval-Augmented Generation (RAG) pipelines**，从而显著降低内存占用和运营成本。
- **技术解决 PDF 解析难题**：虽然解析 PDF 比 Word 文档更复杂，但社区重点推荐了 **LlamaParse** 和 **Unstructured** 作为应对挑战的有力工具，特别是在处理表格和图像方面。
- **Fivetran 为 GenAI 提供数据助力**：Fivetran 对 GenAI 应用的集成简化了数据管理并优化了工程工作流，详见其 [博客文章](https://www.fivetran.com/blog/building-a-chatbot-with-fivetran-and-langchain)。
- **超越微调：模型对齐的突破**：根据关于该主题的 [博客文章](https://blog.premai.io/model-alignment-process/)，RLHF、DPO 和 KTO 对齐技术正被证明是改进 **Mistral 和 Zephyr 7B 模型** 语言生成的先进方法。
- **适合研究的 LLM 论文库**：[Shure-dev](https://shure-dev.github.io/) 的使命是汇编一个强大的 **LLM 相关论文** 库，为研究人员在这个快速发展的领域获取广泛的高质量信息提供重要资源。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**O1 Light 国际运输攻略**：工程师们探索了 **O1 Light 国际递送** 的变通方法，包括通过美国联系人购买。值得注意的是，用户自行构建的 O1 设备在全球范围内均可使用。

**本地 LLM 降低 API 开销**：社区正积极参与使用 **Open Interpreter 离线模式** 以消除 API 费用。详细介绍了配合 LM Studio 等本地模型运行的贡献，包括运行如下命令：*`interpreter --model local --api_base http://localhost:1234/v1 --api_key dummykey`*，具体可参考 [官方文档](https://docs.openinterpreter.com/guides/running-locally)。

**语义搜索协作征集**：社区发起了改进 **[OpenInterpreter/aifs](https://github.com/OpenInterpreter/aifs)** GitHub 仓库内 **本地语义搜索** 的行动号召。这突显了社区驱动的项目增强方法。

**将 O1 Light 与 Arduino 系列集成**：技术讨论涉及将 **O1 Light 与 Arduino** 硬件结合以实现更大的实用性。虽然 ESP32 是标准配置，但成员们渴望尝试 Elegoo 开发板等替代方案。

**O1 开发环境 Windows 安装困扰**：成员们报告并讨论了在 **Windows 系统上安装 01 OS** 的问题。一个 [GitHub pull request](https://github.com/OpenInterpreter/01/pull/192) 旨在提供解决方案并简化 Windows 开发者的设置流程。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**编译器与 CUDA 的博弈**：尽管关于使用 PyTorch/Triton 等**编译器技术**还是手动编写 CUDA 代码的优劣争论仍在继续，成员们也在寻求 **CUDA 课程**的指导，包括对 [GitHub 上的 CUDA mode](https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#cuda-courses) 以及 [YouTube](https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2) 上 Udacity 的《并行编程导论》的推荐。由 Cohere 发起的一项名为 **Beginners in Research-Driven Studies (BIRDS)** 的社区主导 **CUDA 课程**宣布将于 4 月 5 日开始，该消息已在 [Twitter 上发布](https://x.com/CohereForAI/status/1773419415406809432)。

**Windows 与 WSL 同行**：几位成员提供了在 Windows 上运行 CUDA 的易用解决方案，强调了 **Windows Subsystem for Linux (WSL)**，特别是 **WSL2**，并参考了 [Microsoft 官方指南](https://learn.microsoft.com/en-us/windows/wsl/install)。

**围绕 Ring-Attention 的变革**：在 #[ring-attention] 频道中，微调实验与 Ring-Attention 目标的不一致一度导致进度停滞，但关于解决 `modeling_llama.py` loss 问题的见解引领了进展。在拥有充足显存的 A40 上成功训练上下文长度高达 100k 的 tinyllama 模型成为热门话题，同时还有关于 Llama 7B 模型在使用 QLoRA 和 LoRA 时对显存巨大需求的 [Reddit 讨论](https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/helpful_vram_requirement_table_for_qlora_lora_and/)。

**梳理 Triton 的混乱**：#[triton-puzzles] 频道正热烈讨论 `triton-viz` 中与特定 [pull request](https://github.com/Deep-Learning-Profiling-Tools/triton-viz/pull/19/files#diff-617f71ef3c8b3147084e47a1492611a7f42bd28b720fdc57b7ff5111663ec298L21) 相关的同步问题，虽然官方已提供修复，但部分用户仍面临安装困扰。此外，频道内还澄清了在 Windows 上使用 Triton 的方法，建议使用 Google Colab 等替代环境来运行基于 Triton 的计算。

**知乎上的 Triton 热潮**：一位成员成功突破了中文平台 [知乎](https://www.zhihu.com/signin?next=%2F) 的语言障碍，挖掘出了大量 Triton 资料，这激发了大家对建立技术术语表以辅助浏览非英语内容的愿望。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**模型狂热：OpenRouter 推出应用排名**：OpenRouter 为模型推出了**应用排名 (App Rankings)** 功能，以 [Claude 3 Opus](https://openrouter.ai/models/anthropic/claude-3-opus?tab=apps) 为例，根据公开应用的使用情况和处理的 token 数量来展示利用率和受欢迎程度。

**Databricks 和 Gemini Pro 引发关注，但仍有 Bug**：工程师们对 Databricks 的 **DBRX** 和 **Gemini Pro 1.5** 表现出极大的热情，尽管 **error 429** 等问题暗示了速率限制 (rate-limit) 的挑战，而停机时间以及 **error 502** 和 **error 524** 则表明模型可用性的稳定性仍有改进空间。

**讨论 Claude 的功能和 API**：社区澄清了 **OpenRouter 中的 Claude** 不支持 prefill 功能，并探讨了 **Claude 2.1** 的错误修复。另一场讨论称赞了**通过 OpenRouter 访问的 ClaudeAI** 在处理角色扮演和敏感内容方面表现更好，误报更少，并指出其访问标准化且成本与官方 **ClaudeAI API** 持平。

**API 和客户端得到优化**：OpenRouter 简化了其 API 至 **/api/v1/completions**，并由于速率限制原因在 Nitro 模型中弃用了 Groq，同时改进了对 OpenAI API 客户端的支持。

**为 OpenRouter 用户简化加密货币支付**：OpenRouter 正通过利用 Ethereum L2 解决方案 Base 链来大幅降低加密货币交易的 gas 费用，旨在提供更经济的用户体验。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba 加入 AI 战局**：AI21 Labs 推出了 [Jamba](https://www.ai21.com/jamba)，这是一款将 Mamba 与 Transformer 元素结合的先进语言模型，其特点是拥有 **256K context window**，并支持在单个 GPU 上托管 140K 上下文。该模型在 [Hugging Face](https://huggingface.co/ai21labs/Jamba-v0.1) 上以 Apache 2.0 许可证发布，并获得了 [TechCrunch](https://techcrunch.com/2024/03/28/ai21-labs-new-text-generating-ai-model-is-more-efficient-than-most/) 的报道。

- **宣布突破性的吞吐量**：Jamba 在问答和摘要等长上下文任务中实现了 3 倍的吞吐量（throughput）提升，为生成式 AI 领域设定了新的性能基准。

- **AI21 Labs 平台已准备好支持 Jamba**：AI21 Labs 的 SaaS 平台很快将上线 Jamba，作为对其目前在 Hugging Face 上可用性的补充。工作人员表示，随后将发布一份深入探讨 Jamba 训练细节的白皮书（white paper）。

- **多语言专家 Jamba**：Jamba 经过了包括**德语、西班牙语和葡萄牙语**在内的多种语言训练，其多语言能力已得到确认。尽管如此，它在韩语任务中的效能似乎仍存疑问，且目前没有计划在 Hugging Face Space 进行演示。

- **价格与性能调整**：AI21 Labs 已取消了每月 29 美元的模型使用最低收费。关于 Jamba 高效率的讨论显示，其 Transformer 和 Mixture-of-Experts (MoE) 层的混合架构实现了亚二次方（sub-quadratic）缩放。社区讨论还涉及了用于减少内存占用的 quantization（量化），并澄清了 Jamba 的 MoE 是由 Mamba 块而非 Transformer 组成的。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **调试的“乐趣”**：一位工程师分享了他们在处理 **Conda metal compiler bug** 时的“疯狂时刻”，表达了对调试该问题的极度沮丧。
- **Tinygrad 的内存管理**：讨论了在 tinygrad 软件栈中实现虚拟内存（virtual memory）的话题，并表现出将其应用于 cache/MMU 项目的兴趣。
- **CPU 巨头对比**：用户们辩论了 AMD 与 Intel 客户服务的响应速度，强调了 Intel 更好的 GPU 性价比，并指出了 AMD 的软件问题。
- **Intel Arc 的优化机会**：有建议提出通过优化 IPEX 库中的 Transformer/dot product attention，来提升 Intel Arc 的潜在性能。
- **FPGA 驱动的 Tinygrad**：有人询问关于扩展 [tinygrad 以利用 FPGA](https://github.com/tinygrad/tinygrad/blob/master/docs/adding_new_accelerators.md) 的事宜，引发了用户对硬件加速潜在收益的讨论。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 预付费模式解析**：OpenAI 采用预付费模式是为了打击欺诈，并为提高 rate limits 提供更清晰的路径，[LoganK 的推文](https://x.com/officiallogank/status/1760046748569841719?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)提供了内部视角。尽管有其优势，但人们将其与星巴克礼品卡模式面临的问题进行了对比，即未使用的余额对收入贡献巨大。

- **AI21 的 Jamba 亮相**：AI21 的新模型 "Jamba" 融合了 Structured State Space 和 Transformer 架构，拥有 MoE 结构，目前已在 [Hugging Face](https://huggingface.co/ai21labs/Jamba-v0.1) 上开放评估。然而，社区注意到目前缺乏直接测试该新发布模型的机制。

- **xAI 的 Grok-1.5 在 AI 推理方面取得进展**：xAI 新推出的 Grok-1.5 模型被公认为优于其前代产品，特别是在编程和数学任务方面，社区期待其在 xAI 平台上的发布。基准测试和性能详情已通过[官方博客文章](https://x.ai/blog/grok-1.5)分享。

- **三元 LLM 引发 BitNet 辩论**：社区就使用三值逻辑（three-valued logic）的 LLM 的正确术语展开了技术辩论，讨论了 BitNet 的影响，并强调这些是原生训练（natively trained）的模型，而非 quantization（量化）版本。

- **Stability AI 的动荡**：Emad Mostaque 离开 Stability AI 引发了关于生成式 AI 领域领导层变动的讨论。这些讨论受到他在 [Peter Diamandis 的 YouTube 频道](https://youtu.be/e1UgzSTicuY?si=rF7LX1X6Kt7N2YRa)上的采访以及 [archive.is](https://archive.is/8QkSl) 上详细背景故事的推动，反映了行业内持续的转变和潜在的波动。



---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Jamba 树立了高标准**：**AI21 Labs** 推出了一款新模型 **Jamba**，具有 **256k token 上下文窗口**，并针对 **120 亿激活参数** 的性能进行了优化。其快速的训练进度突显了其加速发展，知识截止日期为 2024 年 3 月 5 日，详情可见其 [博客文章](https://www.ai21.com/blog/announcing-jamba)。

**挑战优化的极限**：讨论强调了 **torchTune** 中 **bf16 精度** 的有效性，相比 fp32 能够显著节省内存，这些优化已应用于 SGD，并即将应用于 Adam 优化器。对于 Axolotl 是否能提供与 torchTune 相同级别的训练控制（特别是在内存优化方面），仍存在疑虑。

**尖端技术的代价**：围绕 **基于 GB200 的服务器价格** 的讨论显示，每台服务器的成本高达 200 万至 300 万美元，由于高昂的费用，社区开始考虑替代硬件方案。

**数据集的大小至关重要**：对 **长上下文数据集** 的寻找促使了资源分享，包括来自 Hugging Face 集合的一个资源以及 [Hugging Face 上的 MLDR 数据集](https://huggingface.co/datasets/Shitao/MLDR)，这些资源适用于需要大规模序列训练的模型。

**微调技巧与重复性辩论**：社区一直在就模型训练进行详细讨论，重点分享了 prompt 中 `▀` 和 `▄` 的使用策略，并就数据集重复的效用展开辩论，引用了一篇关于 [数据排序的论文](https://arxiv.org/abs/2310.10638) 来支持重复。尽管存在一些内存挑战，人们也在尝试针对更大模型（如 Galore）的新微调方法。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**OpenGPTs：DIY 送餐系统**：一位富有创造力的工程师将 *自定义送餐 API* 与 **OpenGPTs** 集成，展示了 LangChain 开源平台的适应性和潜力，并在 [演示视频](https://youtu.be/V1SKJfE35D8) 中进行了展示。他们鼓励同行评审以完善这一创新。

**更智能的 SQL AI 聊天机器人**：成员们探索了让 **SQL AI Chatbot** *记住* 之前交互的方法，增强了机器人的上下文保留能力，从而实现更有效、更连贯的对话。

**为产品推荐做好准备**：工程师们讨论了开发一个使用自然语言查询来推荐产品的机器人，考虑使用向量数据库进行语义搜索，或使用 SQL Agent 来解析用户意图（如“计划养宠物”）。

**用 AI 升级你的代码审查**：推出了一款新的 AI 流水线构建工具，旨在自动执行包括验证和安全检查在内的代码审查任务，并附带 [演示](https://www.youtube.com/watch?v=6kfr1lqw2gg) 和 [产品链接](https://gitgud.autonoma.app)，有望简化代码审查流程。

**GalaxyAI 发起挑战**：GalaxyAI 正在提供对 **GPT-4** 和 **Gemini-PRO** 等顶级 AI 模型的 *免费访问*，通过其兼容 OpenAI 的 [API 服务](https://galaxyapi.onrender.com) 为项目提供了一个易于采用的选择。

**培养工程师对话**：<#1222928565117517985> 频道的创建促进了关于 **OpenGPTs** 及其发展的集中讨论，其 [GitHub 仓库](https://github.com/langchain-ai/opengpts) 证明了这一点。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Jamba 领先一步**：AI21 Studios 发布了 [Jamba](https://www.ai21.com/blog/announcing-jamba)，这是一款融合了 Mamba 的 Structured State Space 与 Transformers 的新模型，提供了高达 **256K context window**。该模型不仅因其性能受到推崇，而且易于获取，在 Apache 2.0 许可下开源权重。

- **Jamba 的规格引发关注**：Jamba 的混合设计引发了对其专注于 inference 的 12B 参数以及总计 52B 参数规模的讨论，它利用了包含 16 个 experts 且每个 token 仅激活 2 个的 **MoE** 框架，详情见 [Hugging Face](https://huggingface.co/ai21labs/Jamba-v0.1)。

- **剖析 Jamba 的 DNA**：围绕 Jamba 的真实架构定义展开了对话，争论是否因其混合性质和特定 attention mechanisms 的结合而应被称为 "striped hyena"。

- **Qwen 重新定义效率**：混合模型 `Qwen1.5-MoE-A2.7B` 因以仅 2.7B 的参数量匹配 7B 模型的实力而受到认可，这一效率壮举在其 [GitHub repo](https://github.com/QwenLM/Qwen1.5) 和 [Hugging Face space](https://huggingface.co/Qwen) 等资源中得到了强调。

- **微软收获天才，Databricks 接收 Megablocks**：Liliang Ren 加入 Microsoft GenAI 担任高级研究员，负责构建可扩展的 neural architectures（见其[公告](https://x.com/liliang_ren/status/1773118751413596588?s=46)）；而 Megablocks 在 Databricks 找到了新归宿，增强了其在 [Twitter](https://x.com/tgale96/status/1773342375806374307?s=46) 上提到的长期发展前景。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**AI21 Labs 打造 Jamba**：**AI21 Labs** 推出了 **Jamba**，这是一款融合了 Structured State Space 模型与 Transformer 架构的模型，承诺提供高性能。通过其 [Hugging Face 部署](https://huggingface.co/ai21labs/Jamba-v0.1)查看 **Jamba**，并在 [AI21 网站](https://www.ai21.com/jamba)上阅读其开创性方法。

**翻译巨头之争**：成员们正准备在 **DiscoLM**、**Occiglot**、**Mixtral**、**GPT-4**、**DeepL** 和 **Azure Translate** 之间展开一场翻译大战，使用来自 *Capybara* 等数据集的前 100 行来比较性能。

**征服 LLMs 的课程**：分享了一个提供 Large Language Models 课程的 [GitHub repository](https://github.com/mlabonne/llm-course)，包含路线图和 Colab notebooks，旨在进行 LLMs 教育。

**Token 插入乱象得以解决**：分享了一个关于意外 token 插入的调试成功案例，此前认为是由 **quantization** 或 **engine** 引起的；提供 `added_tokens.json` 解决了该异常。

**训练数据透明度震荡**：社区要求提供更多关于特定模型所用训练数据的信息，特别关注模型卡片或附属博客文章中提到的“英语数据”的定义和范围。





---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **征集 Python 奇才**：一位成员询问了即将为热衷于加入该项目的 Python 开发者准备的 **onboarding schedule**，表示之前已经在筹备专门针对资深 Python 人才的会议。

- **分享 YouTube 教育内容**：Pradeep1148 分享了一个 [YouTube 视频链接](https://www.youtube.com/watch?v=LWz2QaSRl2Y)，但未提供关于内容或与当前讨论相关性的背景信息。




---

# PART 2: 频道详细摘要与链接



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1222804942134185986)** (936 messages🔥🔥🔥): 

- **SD3 的 Turbo 加速**：微调 SD3 的想法依然存在，有一种假设认为 [SD3 Turbo](https://arxiv.org/pdf/2403.12015.pdf) 可能会作为一个强大的、细节丰富的模型，类似于 SDXL Lightning。成员们推测在具有 12GB VRAM 的硬件上运行即将推出的 SD3 的性能优势。

- **AI 艺术与政治**：讨论涵盖了模型能力与公众获取之间的平衡，并对 SD3 功能可能为了公众使用而缩减表示担忧。人们希望在发布时提供微调工具和 control nets，这可以保持用户群的生产力。

- **LoRA 在 SD3 中的未来**：对话围绕在 SD3 上训练 LoRA (Local Rank Aware) 模型展开，24GB 的 VRAM 应该足够了。关于这些工具在发布后能多快、多有效地适配 SD3 仍存在不确定性。

- **硬件训练**：用户讨论了在具有 16GB VRAM 的 GPU 上训练较小尺寸 LoRA 模型的可行性，部分用户成功在高达 896 的分辨率下完成了训练。分享了训练技术和优化方法，如 Aggressive train rates、network dropouts 和 ranking。

- **利用新模型实现增长**：成员们展示并推荐使用 Arc2Face 进行人脸操作，对其演示和功能表示赞赏。讨论中夹杂着讽刺，用户嘲讽了技术被视为外星干预的可能性，并淡化了关于数据集审查的争议。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://brianfitzgerald.xyz/prompt-augmentation/">SuperPrompt - 77M 参数实现更好的 SDXL 提示词 | Brian Fitzgerald</a>：在相同输入提示词下应用 SuperPrompt 的左侧 SDXL 输出。</li><li><a href="https://arc2face.github.io/">Arc2Face：人脸基础模型</a>：未找到描述</li><li><a href="https://svilentodorov.xyz/blog/gpt-15b-chat-finetune/">自言自语：我如何使用 Facebook 聊天数据训练 GPT2-1.5b 进行橡皮鸭调试</a>：本系列的前期内容 - 微调 117M，微调 345M</li><li><a href="https://huggingface.co/spaces/FoivosPar/Arc2Face">Arc2Face - FoivosPar 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://tenor.com/view/so-sayweall-battlestart-galactica-william-adama-bsg-twelve-colonies-gif-18616973">So Sayweall Battlestart Galactica GIF - So Sayweall Battlestart Galactica William Adama - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://toyxyz.gumroad.com/l/ciojz">适用于 Blender 的类似 Openpose 的角色骨骼 _ Ver_96 Depth+Canny+Landmark+MediaPipeFace+finger</a>：-需要 Blender 3.5 或更高版本。- 下载 — blender.org 指南：https://youtu.be/f1Oc5JaeZiw 适用于 Blender 的类似 Openpose 的角色骨骼 Ver96 Depth+Canny+Landmark+MediaPipeFace+fing...</li><li><a href="https://leonardo.ai/">首页 v2</a>：使用我们的 AI 图像生成器改变您的项目。以无与伦比的速度和风格生成高质量的 AI 图像，提升您的创意愿景</li><li><a href="https://www.youtube.com/watch?v=QdRP9pO89MY">Stable Diffusion 详解（10 分钟内）</a>：好奇像 Stable Diffusion 这样的生成式 AI 模型是如何工作的吗？加入我的简短白板动画，我们将一起探索基础知识。在不到 1...</li><li><a href="https://www.youtube.com/watch?v=5_qKLZq6frg">超能勇士：Transformer - 主题曲 | Transformers 官方</a>：订阅 Transformers 频道：https://bit.ly/37mNTUz Autobots, roll out! 欢迎来到 Transformers 官方频道：这里是唯一可以看到所有你喜爱的 Au...</li><li><a href="https://github.com/foivospar/Arc2Face">GitHub - foivospar/Arc2Face: Arc2Face：人脸基础模型</a>：Arc2Face：人脸基础模型。通过在 GitHub 上创建一个账户来为 foivospar/Arc2Face 的开发做出贡献。</li><li><a href="https://github.com/lllyasviel/Fooocus">GitHub - lllyasviel/Fooocus：专注于提示词和生成</a>：专注于提示词和生成。通过在 GitHub 上创建一个账户来为 lllyasviel/Fooocus 的开发做出贡献。</li><li><a href="https://github.com/Nick088Official/SuperPrompt-v1">GitHub - Nick088Official/SuperPrompt-v1: SuperPrompt-v1 AI 模型（本地和在线优化您的提示词）</a>：SuperPrompt-v1 AI 模型（让您的提示词变得更好）本地和在线 - Nick088Official/SuperPrompt-v1</li><li><a href="https://civitai.com/models/9513/fnaf-multi-character-lora">FNAF - 多角色 LoRA - v1.11 | Stable Diffusion LoRA | Civitai</a>：多角色 FNAF LoRA。早期版本针对 Yiffy-e18，当前版本针对 Pony Diffusion V3。最新版本包含 Classic Freddy, C...</li><li><a href="https://github.com/Vargol/StableDiffusionColabs">GitHub - Vargol/StableDiffusionColabs：在 Google Colab 免费层运行的 Diffusers Stable Diffusion 脚本</a>：在 Google Colab 免费层运行的 Diffusers Stable Diffusion 脚本 - Vargol/StableDiffusionColabs</li><li><a href="https://github.com/Vargol/8GB_M1_Diffusers_Scripts">GitHub - Vargol/8GB_M1_Diffusers_Scripts：演示如何在 8GB M1 Mac 上运行 Stable Diffusion 的脚本</a>：演示如何在 8GB M1 Mac 上运行 Stable Diffusion 的脚本 - Vargol/8GB_M1_Diffusers_Scripts
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1222985154033619026)** (2 条消息): 

- **DBRX 在 Perplexity Labs 首次亮相**：来自 Databricks 的最先进语言模型 **DBRX** 现已在 Perplexity Labs 上线，其性能超越了 **GPT-3.5**，并能与 **Gemini 1.0 Pro** 媲美，尤其是在数学和编程任务方面。邀请用户在 [Perplexity Labs](https://labs.pplx.ai) 免费试用 DBRX。

- **Copy AI 与 Perplexity 达成强强联手**：Copy AI 集成了 Perplexity 的 API，旨在打造一个 AI 驱动的进入市场（GTM）平台，提供实时市场洞察以优化决策。作为额外福利，Copy AI 用户可免费获得 6 个月的 Perplexity **Pro** 会员 —— 合作详情请参阅其 [博客文章](https://www.copy.ai/blog/copy-ai-perplexity-purpose-built-partners-for-gtm-teams)。

**提及链接**：<a href="https://www.copy.ai/blog/copy-ai-perplexity-purpose-built-partners-for-gtm-teams">Copy.ai + Perplexity: Purpose-Built Partners for GTM Teams | Copy.ai</a>：深入了解 Perplexity 与 Copy.ai 最近的合作将如何助力您的 GTM 工作！

---

**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1222807897013092364)** (728 条消息🔥🔥🔥): 

- **Perplexity 搜索与专注模式 (Focus Modes)**：讨论显示用户对 Perplexity 的 **学术专注模式 (academic focus mode)** 是否正常运行感到困惑，并指出 [语义搜索 (semantic search)](https://www.perplexity.ai/search/I-am-a-XNNmFl83Rry_HEhBb3k8Tw) 可能存在故障。用户建议关闭专注模式或尝试 **focus all** 作为潜在的变通方案。

- **探索 Pro Search 与 “Pro” 功能**：围绕 **Pro Search** 的讨论表明，部分用户认为追问环节有些多余，更倾向于直接获得答案。Pro 用户可获得 5 美元的 API 额度，并能访问多个 AI 模型，包括在对话线程中持久附加 PDF 等来源以供引用。

- **理解 Perplexity 的文件上下文与来源**：通过测试，用户发现 Perplexity 处理文件上下文的方式在直接上传与文本输入之间存在差异，并推测其使用了 **Retrieval Augmented Generation (RAG)** —— Claude 可能会依赖 **GPT-4-32k** 等其他模型进行文件处理和引用。

- **生成能力与 API 利用率**：用户讨论了使用 Perplexity API 与订阅 **GPT Plus** 和 **Claude Pro** 等服务的成本与收益。针对生成内容创意等特定任务，提到了 **Opus** 等其他订阅制模型，用户正在寻求关于何时使用哪种服务或功能的指导。

- **澄清 Claude Opus 与 GPT-4.5**：用户询问了 **Claude Opus** 回答的准确性与真实性，指出它在输出中曾错误地自称为 OpenAI 开发的产品。参与者建议在解读此类 AI 自我描述时保持谨慎，并避免向 AI 询问其来源。

<div class="linksMentioned">

<strong>提及链接</strong>：

<ul>
<li>
<a href="https://trysora.io/">Sora AI</a>: 未找到描述</li><li><a href="https://x.com/perplexity_ai/status/1773418423726305529?s=20">来自 Perplexity (@perplexity_ai) 的推文</a>: DBRX，来自 @databricks 的最先进开源 LLM，现已在 Perplexity Labs 上线。DBRX 的表现优于 GPT-3.5，并可与 Gemini 1.0 Pro 竞争，在数学和编程任务方面表现出色，并设定了新的标杆...</li><li><a href="https://x.ai/blog/grok-1.5">发布 Grok-1.5</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/docs/pricing">定价</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>: 未找到描述</li><li><a href="https://www.ai21.com/blog/announcing-jamba">介绍 Jamba：AI21 开创性的 SSM-Transformer 模型</a>: 首次推出首个生产级 Mamba 模型，提供同类最佳的质量和性能。</li><li><a href="https://tenor.com/view/rickroll-meme-internet-never-gonna-gif-26474110">Rickroll Meme GIF - Rickroll Meme Internet - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/lost-go-back-jack-we-have-to-go-back-gif-5469258">Lost Go Back GIF - Lost Go Back Jack - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/older-meme-checks-out-gif-14849207">Older Meme Checks Out GIF - Older Meme Checks Out - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1bq5aw1/working_on_open">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>: 未找到描述</li><li><a href="https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm">介绍 DBRX：一个新的最先进开源 LLM | Databricks</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=GanTUWLUUWQ">答案引擎教程：开源 Perplexity 搜索替代方案</a>: 答案引擎安装教程，旨在成为 Perplexity 的开源版本，一种获取问题答案的新方式，取代传统搜索...</li><li><a href="https://youtu.be/57LqvutrOI8?si=x7hGU42L_z3KwmX_">Perplexity CEO：用 AI 颠覆 Google 搜索</a>: 最杰出的 AI 创始人之一 Aravind Srinivas（Perplexity CEO）认为，未来我们可能会看到 100 多家估值超过 100 亿美元的 AI 初创公司。在本集...</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1bq5aw1/working_on_opensource_alternative_to_perplexityai/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/shure-dev/Awesome-LLM-related-Papers-Comprehensive-Topics">GitHub - shure-dev/Awesome-LLM-related-Papers-Comprehensive-Topics: 致力于打造全球最全面的 LLM 相关论文和仓库精选列表</a>: 致力于打造全球最全面的 LLM 相关论文和仓库精选列表 - shure-dev/Awesome-LLM-related-Papers-Comprehensive-Topics</li><li><a href="https://docs.anthropic.com/claude/docs/let-claude-think">让 Claude 思考</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1222809690124320798)** (22 条消息🔥): 

- **Perplexity 能力探索**：多位成员正在探索并分享 [Perplexity AI](https://www.perplexity.ai) 关于 **Truth Social**、*Anthropic 标签模型*、Amazon 投资策略、Exchange 管理复杂性以及全球变暖研究影响等主题的搜索结果。
- **深入详细分析**：一位成员一直在 Perplexity AI 中尝试使用 `<scratchpad>` XML 标签，以展示搜索结果的差异，特别是说明了其在**构建详细输出结构方面的有效性**。
- **太空热情展示**：成员们通过 Perplexity 搜索分享了他们对太空技术和现象的好奇心，询问了关于 **SpaceX Starship 碎片**以及**太空如何影响人体**的细节。
- **技术与金融在用户查询中融合**：Amazon 的投资以及 FTX 联合创始人 **Sam Bankman-Fried** 的故事成为关注焦点，反映了用户对技术与金融交汇点的兴趣。
- **Scratchpad 策略优化**：通过采用**改进的 scratchpad 策略**，一位成员声称显著提升了最终输出的内容质量，突显了从 AI 生成文本中提取**更具探索性和实用性内容**的潜在进步。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1222855722119991327)** (8 条消息🔥):

- **API 结果不一致**：一位成员报告称，在使用 **Perplexity AI API** 时经常得不到结果，这与 Web 界面在搜索信息（如 "Olivia Schough spouse"）时显示的“大量结果”形成鲜明对比。
- **寻求 API 参数指导**：该成员询问是否有更多参数可以引导 API 以改进结果。
- **API 与 Web 界面优势对比**：该成员表示，就性能而言，Web 应用似乎比 API “好得多”。
- **API 响应中缺少来源**：另一位用户指出了一项差异：API 返回的响应缺少在 Web 端使用 Perplexity AI 时所具备的多样化来源。
- **Beta 功能和申请流程**：作为回应，一名用户引导成员查看之前关于 **URL 引用仍处于 Beta 阶段**的消息，并提供了申请链接：[申请 Beta 功能](https://perplexity.typeform.com/to/j50rnNiB)。

**提到的链接**：<a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api 表单</a>：使用 Typeform 将数据收集转化为一种体验。创建精美的在线表单、调查、测验等。免费试用。

---

**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1222793845616410735)** (351 条消息🔥🔥): 

- **讨论 Jamba 的潜力和局限性**：成员们讨论了 AI21 Labs 新发布的 [Jamba 模型](https://www.ai21.com/blog/announcing-jamba)，注意到它声称将 Mamba SSM 技术与传统的 Transformers 相结合。一些人对其性能和 context window 印象深刻，而另一些人则对其 tokenization 和缺乏详细的 token 计数表示怀疑。

- **对各种模型的担忧**：
  成员们分享了使用 Copilot 和 Claude 等 AI 工具的混合体验，指出 Copilot 的质量大幅下降，而一些人觉得 Claude 非常有帮助。此外，还提出了关于 Model Merging 策略的担忧，并提到未来可能会讨论 Argilla。

- **生成模型的评估**：
  讨论集中在如何评估开放式生成的质量，并分享了深入探讨 perplexity、pairwise ranking 以及与表观涌现能力相关的指标资源，例如[这篇关于涌现是神话的论文](https://arxiv.org/abs/2304.15004)。

- **模型训练挑战**：
  用户分享了训练 AI 模型的经验并寻求建议，特别是在微调 Mistral 和使用 SFTTrainer 时遇到的问题。他们交换了配置，并讨论了最佳训练 epochs 和 batch sizes。

- **解决多 GPU 支持及其他技术难题**：
  成员们对 Unsloth AI 缺乏 multi-GPU 支持表示担忧，同时还提到了与 bitsandbytes 依赖相关的问题以及微调过程中的错误消息。成员之间分享了建议和故障排除技巧。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mobiusml.github.io/1bit_blog/">1-bit Quantization</a>: 发布 1-bit Aana 模型的支持博客。</li><li><a href="https://arxiv.org/abs/2304.15004">Are Emergent Abilities of Large Language Models a Mirage?</a>: 最近的研究声称 Large Language Models 展现出了涌现能力（emergent abilities），即在较小规模模型中不存在但在较大规模模型中出现的能力。是什么让涌现能力如此引人入胜...</li><li><a href="https://huggingface.co/unsloth/mistral-7b-v0.2-bnb-4bit">unsloth/mistral-7b-v0.2-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>: 首次推出首个生产级基于 Mamba 的模型，提供同类最佳的质量和性能。</li><li><a href="https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0#technical-deep-dive">yanolja/EEVE-Korean-10.8B-v1.0 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://sambanova.ai/blog/accurate-models-at-blazing-speed">SambaNova Delivers Accurate Models At Blazing Speed</a>: Samba-CoE v0.2 在 AlpacaEval 排行榜上攀升，表现优于所有最新的开源模型。</li><li><a href="https://tenor.com/view/come-look-at-this-gif-21207051">Come Look At This GIF - Come Look At This - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: 未找到描述</li><li><a href="https://www.together.ai/blog/llama-2-7b-32k">Preparing for the era of 32K context: Early learnings and explorations</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/togethercomputer/Long-Data-Collections">togethercomputer/Long-Data-Collections · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://gist.github.com/idncsk/8a05711b57fc0b9b2ee3186fcbb43c25">trainer-v1.py</a>: GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://sambanova.ai/blog/benchmarking-samba-1">Benchmarking Samba-1</a>: 使用 EGAI 基准测试对 Samba-1 进行基准测试 —— 这是一个源自开源社区、广泛采用的基准测试综合集合。</li><li><a href="https://apps.sambanova.ai/sambachat">no title found</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=2-SPH9hIKT8">A little guide to building Large Language Models in 2024</a>: 2024 年构建 Large Language Models 的简明指南。涵盖了在 2024 年训练高性能 Large Language Model 所需了解的所有内容。这是一场入门演讲，包含参考资料链接...</li><li><a href="https://huggingface.co/spaces/PingAndPasquale/MOE-LLM-GPU-Poor-Leaderboard">MOE-LLM-GPU-POOR_LEADERBOARD - a Hugging Face Space by PingAndPasquale</a>: 未找到描述</li><li><a href="https://github.com/potsawee/selfcheckgpt">GitHub - potsawee/selfcheckgpt: SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models</a>: SelfCheckGPT：针对生成式 Large Language Models 的零资源黑盒幻觉检测 - potsawee/selfcheckgpt</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 速度快 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/trainer">Trainer</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1222881502019194961)** (34 条消息🔥):

- **Andrej Karpathy 的 AI 类比**：一段由 Andrej Karpathy 出镜的 [YouTube 视频](https://www.youtube.com/watch?v=c3b-JASoPi0) 讨论了将 LLM 类比为操作系统的观点，提到了 **finetuning** 的重要方面，以及将 Mistral 和 Gemma 标记为 open source 的误区。
- **Finetuning 的精细平衡**：强调为了实现有效的 finetuning，模型原始训练数据与新数据的混合至关重要；否则，模型的熟练程度可能会退化。
- **AI 面板讨论会的遗憾**：对话透露出失望，因为在一个由 AI 专家 Andrej Karpathy 参加的公开面板讨论中，关于 Elon Musk 管理风格的非技术问题优先于更深层次的 AI 话题。
- **Reinforcement Learning 批评**：Andrej Karpathy 批评了当前的 RLHF 技术，将其与 AlphaGo 的训练进行了对比并指出其不足，建议我们需要更基础的模型训练方法，类似于教科书练习。
- **Snapdragon 新芯片备受瞩目**：讨论了新款 Snapdragon X Elite 芯片的 benchmarks 和功能，并与现有技术进行了比较，分享了对处理器性能未来进步的期望，同时附带了一段 [benchmark 评论视频](https://youtu.be/dTCm6BupWEQ)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=c3b-JASoPi0">与 Andrej Karpathy 和 Stephanie Zhan 一起让 AI 触手可及</a>：OpenAI 创始成员、前 Tesla AI 高级总监 Andrej Karpathy 在 Sequoia Capital 的 AI Ascent 活动中与 Stephanie Zhan 对谈，讨论了...</li><li><a href="https://youtu.be/dTCm6BupWEQ">现在我们知道了分数 | X Elite</a>：Qualcomm 全新的 Snapdragon X Elite benchmarks 已经发布！深入了解不断发展的 ARM 架构处理器格局，以及 Snapdragon X Elite 充满前景的性能...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1222794356579369041)** (201 messages🔥🔥): 

- **Tokenizer 故障**：用户在使用 unsloth 的 `get_chat_template` 函数配合 Starling 7B tokenizer 时遇到了 `list index out of range` 错误。这表明 Starling 的实现可能存在 tokenization 问题。
- **Finetuning 的挫折**：一位成员讨论了 finetuning 过程中的困难，描述了尝试各种参数但未达到预期效果的情况。其他人的建议包括固定 rank 以及尝试不同的 alpha 速率和 learning rates。
- **Checkpoint 挑战**：有人提出了 unsloth 在训练期间需要互联网访问的担忧，特别是当网络不稳定时，可能会干扰 checkpoint 的保存。Unsloth 本身不需要互联网连接即可训练，问题可能与 Hugging Face 模型托管或 WandB 报告有关。
- **将 Checkpoints 推送到 Huggingface**：一位成员询问如何将本地 checkpoints 推送到 Hugging Face，得到的回复是建议使用 Hugging Face 的 `hub_token` 处理云端 checkpoints，并使用 `save_strategy` 进行本地存储。
- **Overfitting 的开销**：用户寻求针对模型 overfitting 的建议，提供的建议包括降低 learning rate、降低 LoRA rank 和 alpha、缩短 epoch 数量，以及包含一个 evaluation loop 以在 loss 增加时停止训练。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharin">Google Colaboratory</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://ko-fi.com/unsloth">在 Ko-fi 上支持 Unsloth AI！❤️。ko-fi.com/unsloth</a>: 在 Ko-fi 上支持 Unsloth AI。Ko-fi 让您可以通过小额捐赠支持您喜爱的人和事业</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/modelfile.md#template-variables">ollama/docs/modelfile.md at main · ollama/ollama</a>: 快速上手 Llama 2, Mistral, Gemma 以及其他大语言模型。- ollama/ollama</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: 提速 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/toranb/sloth">GitHub - toranb/sloth: 基于 unsloth 的 python sftune, qmerge 和 dpo 脚本</a>: 基于 unsloth 的 python sftune, qmerge 和 dpo 脚本 - toranb/sloth</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.hub_token">Trainer</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py#L327),">unsloth/unsloth/models/llama.py at main · unslothai/unsloth</a>: 提速 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L540)">transformers/src/transformers/models/llama/modeling_llama.py at main · huggingface/transformers</a>: 🤗 Transformers: 适用于 Pytorch, TensorFlow 和 JAX 的前沿机器学习框架。- huggingface/transformers</li><li><a href="https://youtu.be/rANv5BVcR5k?si=g3VOwbGUFCWaLWd3">Mistral 微调入门指南（支持 16k, 32k, 128k+ 上下文）</a>: 在我们的最新教程视频中探索使用自有数据轻松微调语言模型 (LLMs) 的秘诀。我们深入探讨了一种极具成本效益且...</li><li><a href="https://www.youtube.com/live/g68qlo9Izf0?si=X3dDHSeeqOCV6WN6">在单 GPU 上高效微调 Llama-v2-7b</a>: 微调 LLM 时你可能遇到的第一个问题是“主机内存不足”错误。微调 7B 参数模型更加困难...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1222911167232475256)** (6 条消息): 

- **Tinyllama 利用 Unsloth Notebook**: 来自 Unsloth notebook 的 Lora Adapter 被转换为 ggml adapter 以用于 Ollama 模型，从而诞生了基于 Unsloth Notebook 训练并使用 Huggingface 数据集的 [Tinyllama 模型](https://ollama.com/pacozaa/tinyllama-alpaca-lora)。
- **Mischat 模型发布**: 另一个名为 Mischat 的模型（利用 gguf）现已在 Ollama 上更新。它使用了 [Mistral-ChatML Unsloth notebook](https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing) 进行微调，可以在 [Ollama 链接](https://ollama.com/pacozaa/mischat)找到。
- **展示模型中的模板映射**: 一位用户展示了 notebook 中定义的模板如何反映在 Ollama modelfiles 中，并以最近的模型作为该过程的示例。 
- **面向 AI 爱好者的博文**: 一位成员介绍了一篇题为 'AI Unplugged 5' 的新博文，涵盖了从 Apple MM1 到 DBRX 再到 Yi 9B 的各种 AI 主题，提供了对近期 AI 发展的见解，可在 [AI Unplugged](https://datta0.substack.com/p/ai-unplugged-5-databricks-dbrx-apple) 阅读。
- **AI 摘要内容广受好评**: 总结每周 AI 活动的博客收到了另一位用户的积极反馈，该用户对提供 AI 进展见解的努力表示赞赏。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ollama.com/pacozaa/mischat">pacozaa/mischat</a>: 来自 Unsloth notebook ChatML 与 Mistral 微调会话的模型。Notebook 链接：https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing</li><li><a href="https://ollama.com/pacozaa/tinyllama-alpaca-lora">pacozaa/tinyllama-alpaca-lora</a>: 使用 Unsloth Notebook 训练的 Tinyllama，数据集：https://huggingface.co/datasets/yahma/alpaca-cleaned</li><li><a href="https://datta0.substack.com/p/ai-unplugged-5-databricks-dbrx-apple">AI Unplugged 5: DataBricks DBRX, Apple MM1, Yi 9B, DenseFormer, Open SORA, LlamaFactory 论文, 模型合并。</a>: 前一期目录：Databricks DBRX、Apple MM1、DenseFormer、Open SORA 1.0、LlaMaFactory 微调分析、Yi 9B、进化模型合并（Evolutionary Model Merges）。感谢阅读 Datta’s Substack！订阅 ...</li><li><a href="https://datta0.notion.site/AI-Unplugged-c2c577fe8af54534aec540fc4a4032dd?pvs=4">Notion – 集笔记、任务、维基和数据库于一体的全能工作空间。</a>: 一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的全能工作空间。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/)** (1 条消息): 

starsupernova: 噢，非常酷！
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1222822120027852913)** (237 条消息🔥🔥): 

- **AI 爱好并不便宜**: 成员们讨论了将 AI 作为爱好的相关成本，由于需要高端 GPU 或 MacBook 以获得良好性能，支出达到了约 1 万美元。

- **AI 新手的挑战**: AI 世界的新手面临着陡峭的学习曲线，讨论强调了对计算密集型和充满术语的领域的“艰难觉醒”。

- **本地 LLM 指南与故障排除**: 交流了关于运行本地 LLM 的若干询问和技巧，从 LM Studio 和 GPU 兼容性问题到基于 VRAM 和系统规格的模型选择建议。

- **LM Studio 增强与功能请求**: 用户对分支系统（Branching system）等新功能表示兴奋，并强调了为分支聊天设立文件夹的实用性；一些用户请求“删除所有”功能以获得更流畅的用户体验。

- **社区支持与资源共享**: 有益的对话以及 YouTube 指南和 GitHub 链接等共享资源为用户提供了支持，他们共同解决技术挑战并探索不同模型的能力。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLM</a>: 查找、下载并实验本地 LLM</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat">Qwen/Qwen1.5-MoE-A2.7B-Chat · Hugging Face</a>: 未找到描述</li><li><a href="https://useanything.com/">AnythingLLM | 终极 AI 商业智能工具</a>: AnythingLLM 是为您的组织打造的终极企业级商业智能工具。拥有对 LLM 的无限控制、多用户支持、内部和外部工具，以及 1...</li><li><a href="https://www.youtube.com/watch?v=-bVc3i9hZJg">请集成 LM Studio 实时 STT/TTS</a>: 给 LM Studio 开发团队的信息。请为我们提供实时语音转文本和文本转语音功能。谢谢！</li><li><a href="https://www.youtube.com/watch?v=kFC-OWw7G8k">使用 Vercel AI SDK, Groq, Mistral, Langchain, OpenAI, Brave &amp; Serper 构建 Next.JS 问答引擎</a>: 构建 Perplexity 风格的 LLM 问答引擎：前端到后端教程。本教程引导观众完成构建 Perplexity 风格 La...</li><li><a href="https://youtu.be/Z5_LvCwbgqg?si=D">LM Studio: 在本地运行任何开源 LLM 的最简单方法！</a>: 你准备好潜入本地大语言模型（LLMs）的精彩世界了吗？在这段视频中，我们将带你踏上探索惊人能力的旅程...</li><li><a href="https://youtu.be/Z5_LvCwbgqg?si=DesXy5T0dg5BHVkK">LM Studio: 在本地运行任何开源 LLM 的最简单方法！</a>: 你准备好潜入本地大语言模型（LLMs）的精彩世界了吗？在这段视频中，我们将带你踏上探索惊人能力的旅程...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6074">由 simonJJJ 添加 qwen2moe · Pull Request #6074 · ggerganov/llama.cpp</a>: 此 PR 增加了对即将发布的 Qwen2 MoE 模型 hf 的代码支持。我修改了几个宏值以支持 60 专家（experts）设置。@ggerganov
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1222837018728468491)** (39 条消息🔥):

- **未来模型的 VRAM 需求**：有人提出了当前 GPU 配置缺乏运行即将推出的模型所需的 VRAM 的问题，特别指出 **32GB VRAM 可能不足以运行模型的 Q2 或 Q3 版本**。[一个 GitHub issue](https://github.com/ggerganov/llama.cpp/issues/6344) 提到 **llama.cpp** 尚未支持像 DBRX 这样较新的、对 VRAM 要求较高的模型格式。
  
- **LM Studio 与 Open Interpreter 的配合使用**：一位成员询问是否有人成功让 Open Interpreter 与 LM Studio 配合工作，另一位成员链接了[官方文档](https://docs.openinterpreter.com/language-models/local-models/lm-studio)和 [YouTube 教程](https://www.youtube.com/watch?v=8HIatLzCJDA)以协助设置。
  
- **利用现有服务器设置运行大型 LLM**：一些成员讨论了使用配备 Nvidia P40 GPU 的旧服务器来运行大型语言模型，并指出虽然它们对于 Stable Diffusion 等任务并不理想，但可以以相当不错的速度处理大型模型。
  
- **评估创意写作能力**：几位成员对比了不同的语言模型，指出在创意写作任务中 **Zephyr 的表现可能优于 Hermes2 Pro**，强调了各种语言模型在处理特定功能时的细微差别。

- **为有限的 VRAM 选择合适的 Coder LLM**：关于在 24GB RTX 3090 上使用哪种最佳 Coder 语言模型的查询引出了 **DeepSeek Coder** 可能是首选的观点，并且 Instruct 模型在作为 Agent 使用时可能表现更好。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=F0KDFRbh5h0">DBRX: My First Performance TEST - Causal Reasoning</a>：在 @Databricks 发布 DBRX 的第一天，我对因果推理和轻量级逻辑任务进行了性能测试。以下是我的一些结果...</li><li><a href="https://docs.openinterpreter.com/language-models/local-models/lm-studio">未找到标题</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=8HIatLzCJDA">LM Studio + Open Interpreter to run an AI that can control your computer!</a>：这是一个简陋的视频（我不知道如何获得更高的分辨率），展示了现在使用 AI 是多么容易！我在客户端和服务器中运行 Mistral Instruct 7B...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6344">Add support for DBRX models: dbrx-base and dbrx-instruct · Issue #6344 · ggerganov/llama.cpp</a>：前提条件 在提交 issue 之前，请先回答以下问题。我正在运行最新的代码。由于目前开发非常迅速，还没有标记版本。我...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1222969151560351755)** (2 条消息): 

- **LM Studio 0.2.18 更新发布**：LM Studio 发布了 0.2.18 版本，引入了稳定性改进、针对 Base/Completion 模型的新“空白预设 (Empty Preset)”、针对各种 LM 的默认预设、“等宽字体 (monospace)”聊天样式以及多项错误修复。从 [LM Studio](https://lmstudio.ai) 下载新版本或使用“检查更新”功能。

- **详细的全面错误修复**：此次更新解决了诸如聊天图片重复、没有模型时 API 错误消息含糊不清、GPU Offload 错误以及 UI 中模型名称显示不正确等问题。Mac 用户修复了与 Metal 相关的加载错误，Windows 用户则看到了安装后应用打开问题的修正。

- **文档网站上线**：LM Studio 的专用文档网站现已上线，网址为 [lmstudio.ai/docs](https://lmstudio.ai/docs)，并计划在不久的将来扩展内容。

- **可在 GitHub 上访问配置预设**：无法在设置中找到新配置的用户可以在 GitHub 上的 [openchat.preset.json](https://github.com/lmstudio-ai/configs/blob/main/openchat.preset.json) 和 [lm_studio_blank_preset.preset.json](https://github.com/lmstudio-ai/configs/blob/main/lm_studio_blank_preset.preset.json) 访问它们，尽管最新的下载和更新应该已经包含了这些预设。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLMs</a>: 查找、下载并实验本地 LLMs</li><li><a href="https://lmstudio.ai/docs.">👾 LM Studio - 发现并运行本地 LLMs</a>: 查找、下载并实验本地 LLMs</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/openchat.preset.json">configs/openchat.preset.json at main · lmstudio-ai/configs</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/lm_studio_blank_preset.preset.json">configs/lm_studio_blank_preset.preset.json at main · lmstudio-ai/configs</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1222993668508745769)** (13 messages🔥): 

- **对 AI 易用性的赞扬**：一位成员表达了对该 AI 工具的赞赏，称其为他们使用过的最用户友好的 AI 项目。
- **报告 VRAM 显示不准确**：一位成员指出显示的 VRAM 不正确，仅显示 *68.79GB*，而非预期的 *70644 GB*。
- **在海量模型中寻找稳定模型**：另一位成员透露他们正在筛选近 *2TB 的模型*，遇到了许多模型产生乱码或重复文本的问题。
- **Prompt 格式化困扰**：成员们讨论了使用正确的 Prompt 格式以使模型正常运行的重要性，并提到可以查看模型的 Hugging Face 页面以获取线索。
- **不同版本的 VRAM 读取波动**：据观察，显示的 VRAM 在软件的不同版本中有所变化，一位成员推测 *2.17* 版本显示的可能是最准确的值。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1222868077964820590)** (111 messages🔥🔥): 

- **电源规格建议**：讨论建议对于 Intel 14900KS 和 RTX 3090 的配置使用 **1200-1500W 电源**，而对于像 7950x/3d 这样的 AMD 配置，在不限制 GPU 功耗的情况下，**1000W** 可能就足够了。金牌/白金牌认证的 PSU 因效率更高而受到推荐。

- **GPU 功耗限制技巧**：有分享指出，将 **RTX 4090** GPU 锁定在 350-375W 范围内仅会导致极小的性能损失，并强调这种限制比**全功率超频带来的微小收益**更具实际意义。

- **对旧硬件的性能热情**：一些成员讨论了利用 **K80** 等旧 GPU 硬件，同时也指出了高发热等局限性。有人建议可以改装**旧的 iMac 风扇**来帮助这类古董技术散热。

- **探索 NVLink 的实用性**：成员们就 NVLink 是否能提高模型推理性能交换了意见，一些人认为使用该连接后 **Token 生成速度有明显提升**，而另一些人则质疑其在推理任务中的效用。

- **LM Studio 兼容性查询**：随着 LM Studio 的更新，用户通常会询问对各种 GPU 的支持情况，重点讨论了局限性和解决方法，例如使用 **OpenCL 进行 GPU offloading 或为旧款 AMD 显卡使用 ZLUDA**。此外，有时会遇到并解决一些问题，例如通过全屏运行应用**修复了 UI 问题**。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/grammar-police-drift-police-car-gif-17312786">Grammar Police Drift GIF - Grammar Police Drift Police Car - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.aliexpress.com/item/1005005943893305.html">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1222799944990195734)** (80 messages🔥🔥):

- **用户提议 GPU/NPU 监控**：有人建议在用法部分添加 GPU 和 NPU 的监控功能，并附带利用率统计，以便更好地进行系统参数优化。
- **GPU Offloading 故障排除**：用户讨论了 Windows 上的 **ROCm beta** 无法将模型加载到 GPU 显存的问题。对话指出，当并非所有层都 offload 到 GPU 时，可能会出现部分 offloading 的问题。用户分享了日志和错误消息以诊断该问题。
- **LM Studio Windows 版本混淆**：一些用户报告称，即使正在使用 0.2.18 版本，日志中的 **LM Studio** 版本仍显示为 0.2.17。还有关于在 UI 中调整 GPU offload 层数滑块问题的讨论。
- **ROCm Beta 稳定性疑问**：一位用户在使用 **0.2.18 ROCm Beta** 时遇到崩溃；初始运行正常，但在提问时失败。呼吁用户使用调试版本进行测试并发送详细日志以供进一步调查。
- **通过 GitHub PR 在 Nix 上运行 LM Studio**：一位用户成功将 **LM Studio** 集成到 Nix 中，分享了 Pull Request 链接 (https://github.com/NixOS/nixpkgs/pull/290399)，并表示计划很快合并该更新。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta-Verbose/beta/LM-Studio-0.2.18-ROCm-Beta-Verbose-Setup.exe">未找到标题</a>：未找到描述</li><li><a href="https://lmstudio.ai/rocm">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta/beta/LM-Studio-0.2.18-ROCm-Beta-Setup.exe">未找到标题</a>：未找到描述</li><li><a href="https://learn.microsoft.com/en-us/windows/release-health/windows11-release-information">Windows 11 - 发行信息</a>：了解 Windows 11 各版本的发行信息</li><li><a href="https://github.com/NixOS/nixpkgs/pull/290399">lmstudio: init at 0.2.18 by drupol · Pull Request #290399 · NixOS/nixpkgs</a>：新应用：https://lmstudio.ai/  变更描述  已完成事项  构建平台  x86_64-linux  aarch64-linux  x86_64-darwin  aarch64-darwin  对于非 Linux：Nix 中是否启用了沙箱....
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1223039440469885099)** (2 条消息): 

- **查看 OllamaPaperBot 的代码**：一位成员分享了 **OllamaPaperBot** 的 [GitHub 仓库](https://github.com/eltechno/OllamaPaperBot/blob/main/simplechat.py)，这是一个旨在使用开源 LLM 模型与 PDF 文档交互的聊天机器人，并邀请他人审阅其代码。
- **询问 LM Studio 的 JSON 输出**：一位成员询问是否有人尝试将 LM Studio 的新 JSON 输出格式与分享的 OllamaPaperBot 仓库结合使用。

**提到的链接**：<a href="https://github.com/eltechno/OllamaPaperBot/blob/main/simplechat.py">OllamaPaperBot/simplechat.py at main · eltechno/OllamaPaperBot</a>：基于开源 LLM 模型设计的与 PDF 文档交互的聊天机器人 - eltechno/OllamaPaperBot

  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1222819107192307784)** (54 条消息🔥): 

- **了解 LM Studio 的 GPU 利用率**：用户报告了 LM Studio 的问题，在 0.2.17 版本中 GPU 被正常利用，但在更新到 0.2.18 后，GPU 利用率明显下降至 10% 以下。提供了使用 **[调试版本](https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta-Verbose/beta/LM-Studio-0.2.18-ROCm-Beta-Verbose-Setup.exe)** 导出应用日志的说明，以调查该问题。

- **准备调试**：针对低 GPU 利用率，一位用户尝试使用详细调试版本进行故障排除，并指出当 LM Studio 作为聊天机器人而非服务器使用时，GPU 利用率有所提高。

- **卸载模型导致错误**：用户发现了一个潜在的 Bug，在 LM Studio 中卸载模型后，如果不重启应用程序就无法加载任何新模型。其他用户无法复现此故障。

- **驱动问题可能影响性能**：关于正确应用 ROCm 而非 AMD OpenCL 的讨论表明，更新驱动程序或安装正确的版本可以解决某些问题。一位用户确认在更新 AMD 驱动程序并安装正确版本的 LM Studio 后情况有所改善。

- **部分 GPU 在新更新中无法识别**：用户观察到，虽然 LM Studio 2.16 版本可以识别副 GPU，但 2.18 版本不再识别。一位用户澄清说，在确保使用 ROCm 版本而非普通 LM Studio 版本后，模型开始正常工作。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta-Verbose/beta/LM-Studio-0.2.18-ROCm-Beta-Verbose-Setup.exe">未找到标题</a>: 未找到描述</li><li><a href="https://community.amd.com/t5/ai/how-to-run-a-large-language-model-llm-on-your-amd-ryzen-ai-pc-or/ba-p/670709">如何在您的 AMD Ryzen™ AI PC 或 Radeon 显卡上运行大语言模型 (LLM)</a>: 您知道可以在您的 Ryzen™ AI PC 或 Radeon™ 7000 系列显卡上运行您自己的基于 GPT 的 LLM 驱动的 AI 聊天机器人吗？AI 助手正迅速成为必不可少的资源...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1223044088232808448)** (2 条消息): 

- **通过 AI 深化抽象概念**: 一位成员分享了他们对一篇论文（[Abstract Concept Inflection](https://arxiv.org/pdf/2304.01904.pdf)）的兴趣，该论文讨论了如何使用 AI 将抽象概念扩展为详细概念，类似于分解编码任务。他们目前正在尝试使用不同的模型作为评论者（critic）或生成器（generator），以利用 autogen 进行概念验证。

- **寻求适用于 LM Studio 的 Agent 程序推荐**: 一位成员询问该使用哪种 Agent 程序，以及是否有可以轻松接入 LM Studio 的程序。他们正在寻求有关与 LM Studio 集成的指导。
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1223319785140387853)** (1 条消息): 

- **Voice Engine 亮相**: OpenAI 分享了关于名为 **Voice Engine** 的新模型的见解，该模型仅需文本和一段 [15 秒的音频样本](https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices) 即可生成自然流畅的语音，旨在产生富有情感且逼真的声音。
- **Voice Engine 驱动 Text-to-Speech API**: **Voice Engine** 最初开发于 2022 年底，目前用于 OpenAI 的 [text-to-speech API](https://platform.openai.com/docs/guides/text-to-speech) 中的预设语音，并增强了 [ChatGPT Voice and Read Aloud](https://openai.com/blog/chatgpt-can-now-see-hear-and-speak) 功能。
- **聚焦 AI 伦理**: OpenAI 强调将采取谨慎的态度来更广泛地发布 **Voice Engine**，承认潜在的滥用风险，并强调其 [AI Charter](https://openai.com/charter) 中概述的负责任地部署合成语音的重要性。

**提到的链接**: <a href="https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices">应对合成语音的挑战与机遇</a>: 我们正在分享来自 Voice Engine（一个用于创建自定义语音的模型）小规模预览的经验教训。

  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1222814546272587836)** (75 条消息🔥🔥): 

- **Gemini Advanced 让用户感到不尽如人意**: 成员们对 **Gemini Advanced** 表示失望，指出与其他产品相比响应等待时间更长，且在欧洲的可用性有限。
- **OpenAI 产品版本混淆**: Discord 用户讨论了不同 OpenAI 应用中使用的 GPT-4 版本，**ChatGPT** 报告的截止日期各不相同，导致了对模型稳定性的困惑。
- **VoiceCraft 引发热议与担忧**: 指向 **VoiceCraft** 的链接吸引了关注，这是一个用于语音编辑和 TTS 的 *state-of-the-art* 工具，具有令人印象深刻的演示，但也引发了对可能被用于诈骗的担忧。
- **关于语音克隆的法律和伦理讨论**: 用户辩论了语音克隆的合法性，一些人强调预先防止滥用的难度，另一些人则指出技术可用性与非法活动之间的区别。
- **OpenAI 对创新的谨慎态度**: 关于 OpenAI 对新兴 AI 技术采取谨慎发布策略的对话引发了用户对进度缓慢的沮丧，但也因其管理风险的系统性方法而获得了理解。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://jasonppy.github.io/VoiceCraft_web/">VoiceCraft</a>: 未找到描述</li><li><a href="https://github.com/jasonppy/VoiceCraft">GitHub - jasonppy/VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild</a>: 野外环境下的零样本语音编辑与文本转语音 - jasonppy/VoiceCraft
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1222855761248653393)** (14 条消息🔥):

- **机器人开发热情**：一位成员表达了创建 **Telegram 或 Discord 机器人**的意图。
- **对未来 GPT 的好奇心**：有人提出了 **GPT-5** 是否会超越 **Gemini 1.5 Pro** 的疑问。回答各异，有人开玩笑地预测它会比 GPT-4 好，但比假设中的 GPT-6 差。
- **对 GPT-5 的期待**：用户讨论了 **GPT-5** 可能的发布和访问权限，尽管目前还没有关于其可用性的具体细节。
- **为 GPT-4 代码执行提供帮助**：一位用户在运行代码时需要帮助，因为 GPT-3.5 无法提供解决方案，另一位成员主动提出通过私信协助。
- **频道使用提醒**：发布了一项提醒，要求用户将关于 **ChatGPT** 及其相关应用的讨论保留在专用频道中，并强调 **gpt-4-discussions** 频道是专门用于 GPI AI 模型讨论的。

---

**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1222843909466751046)** (153 条消息🔥🔥): 

- **提示词精准成就完美**：成员们讨论了正确组织提示词（Prompt）语言以实现预期结果的重要性。重点介绍了诸如 `Respond **only** with {xyz}. Limit yourself to {asdf}.` 之类的示例，以确保模型给出简洁的回复。

- **GPT-4.0125 的角色扮演难题**：一位用户分享了 GPT-4.0125 模型在角色扮演提示词方面的异常表现，即提供回复示例反而会导致模型拒绝角色扮演。该用户还找到了解决方法，例如预先构建第一条消息并删除“遵守以下指令”等字眼，并讨论了涉及微妙引导短语和排除提及“不要做什么”的故障排除步骤。

- **MS Word 中的公式问题**：一位用户寻求关于将 LaTeX 公式从 ChatGPT 转移到 Microsoft Word 的建议。尽管参考了 YouTube 教程，但由于其 MS 365 版本不同，问题依然存在，随后有人建议使用 MathML 作为中间格式。

- **元提示（Meta-Prompting）价值讨论**：辩论了元提示与传统提示的有效性，成员们探讨了元提示是否能持续产生更高质量的输出，尽管有证据表明模型可能会不可预测地提供正确或错误的响应。另一位用户根据平台规则不鼓励链接到外部论文，但建议了搜索相关研究的关键词。

- **演示文稿提示词入门**：一位寻求“填空式”方法的用户发起了关于为演示文稿创建提示词的对话。对话强调了向 AI 提供详细信息的重要性，以便其有效地协助完成生成演示文稿等任务。

---

**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1222843909466751046)** (153 条消息🔥🔥): 

- **提示词精准度提升性能**：讨论强调了**提示词措辞**的重要性，建议使用特定的表述，如 *"Respond **only** with {xyz}. Limit yourself to {asdf}"* 来提高 AI 性能；避免使用模糊术语至关重要。
- **展示正确的翻译提示词**：一位用户寻求改进翻译提示词的建议，旨在保留 HTML 标签和某些英文元素。另一位成员建议在**提示词**中专注于**要做什么**以获得更好的输出，例如 *"Provide a skilled translation from English to {$lang_target}"*，同时保持特定的格式和内容要求。
- **角色扮演回复的策略性提示**：用户交流了让 GPT-4 遵守角色扮演指令的策略，解决了 **gpt-4-0125-preview** 在给出示例格式时拒绝角色扮演但在省略示例时却能遵守的异常情况。
- **Word 中公式的格式困扰**：成员们尝试帮助一位在将 **LaTeX 公式**复制到 Microsoft Word 时遇到困难的用户。给出了使用 **MathML** 作为中间格式以及验证 **Microsoft Word 版本兼容性**等建议。
- **元提示（Metaprompting）是可行之路吗？**：关于**元提示**是否能持续取得更好结果存在辩论，多位用户分享了怀疑态度和个人经验，挑战了元提示相对于传统直接指令的有效性。

---

**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1222882406109937684)** (272 条消息🔥🔥): 

- **AI 预测涌入聊天**：对话围绕 AI 超越人类水平智能的潜力展开，并担心实现此类进步可能需要完全不同的架构和硬件。关于扩展当前系统的成本、生态经济学以及摩尔定律（Moore's law）的终结等话题引发了激烈辩论。

- **倾向于拉丁语和公关腔**：一位成员分享了对“testament”一词的厌恶，认为它是 AI 生成内容的典型标志，而其他人则指出 ChatGPT 类似企业公关（PR）的举止使其很容易被识别。

- **对初创公司投机的怀疑**：一个源自 YouTube 的关于初创公司开发更高效 AI 芯片的说法引发了关于改变硬件范式的困难以及半导体制造复杂性的讨论。

- **语法和语言特性的探讨**：聊天参与者对源自拉丁语的复数词被误用为单数形式感到好笑，分享了个人轶事并澄清了包括英语和葡萄牙语在内的各种语言中的正确用法。

- **对 AI 快速变革的担忧**：一些人表达了社会可能无法应对 AI 能力快速加速的观点，反思了社会如何处理数字革命的影响以及在 AI 领域进行谨慎进步的必要性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://summerofcode.withgoogle.com/">Google Summer of Code</a>：Google Summer of Code 是一个全球性项目，旨在吸引更多开发者参与开源软件开发。</li><li><a href="https://www.grammar-monster.com/plurals/plural_of_axis.htm">The Plural of Axis</a>：未找到描述</li><li><a href="https://www.semianalysis.com/p/the-ai-brick-wall-a-practical-limit">The AI Brick Wall – A Practical Limit For Scaling Dense Transformer Models, and How GPT 4 Will Break Past It</a>：大型生成式 AI 模型为世界释放了巨大价值，但前景并非全是玫瑰。这些重新定义文明的模型的训练成本正以惊人的速度膨胀……</li><li><a href="https://blog.eleuther.ai/rotary-embeddings/">Rotary Embeddings: A Relative Revolution</a>：Rotary Positional Embedding (RoPE) 是一种统一了绝对和相对方法的新型位置编码。我们对其进行了测试。</li><li><a href="https://www.eleuther.ai/releases">Releases &mdash; EleutherAI</a>：未找到描述</li><li><a href="https://github.com/">GitHub: Let’s build from here</a>：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专家一样评审代码，跟踪 Bug 和功能……</li><li><a href="https://github.com/davisyoshida/haiku-mup">GitHub - davisyoshida/haiku-mup: A port of muP to JAX/Haiku</a>：muP 到 JAX/Haiku 的移植。通过在 GitHub 上创建账号为 davisyoshida/haiku-mup 的开发做贡献。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1222852101764546560)** (46 条消息🔥): 

- **LISA：一种新的微调技术**：LISA 是一种微调 Large Language Models 的新方法，在指令遵循任务中优于 LoRA 和全参数训练，并且可以在 24GB GPU 上微调 7B 参数模型。这种简单而强大的算法激活 embedding 和 linear head 层，同时随机采样中间层进行解冻。[论文](https://arxiv.org/abs/2403.17919)和[代码](https://github.com/OptimalScale/LMFlow)可供查阅。
  
- **Mamba 与 Transformer 联手**：AI21 发布了 Jamba，这是一个结合了 Mamba 架构和 Transformer 的模型，形成了一个在总计 52B 参数中拥有 12B 激活参数的混合体。据称它兼具两者的优点，包括伪工作记忆和潜在的更高数据效率操作。[AI21 Jamba 发布](https://www.ai21.com/jamba)

- **保护敏感训练数据**：讨论了使用敏感数据进行训练的各种方法，包括解耦敏感示例的 SILO（[论文](https://arxiv.org/abs/2308.04430)）和差分隐私方法（[论文 1](https://arxiv.org/abs/1607.00133)，[论文 2](https://arxiv.org/abs/2110.05679)），以防止从模型中提取数据。

- **SambaNova 推出 Samba-1**：SambaNova 发布了 Samba-1，这是一个专家组合（Composition of Experts, CoE）模型，集成了来自开源社区的 50 多个专家模型的超过 1 万亿个参数，声称在企业任务上具有卓越性能且计算需求更低。由于缺乏透明细节，人们对其在实际使用中的性能普遍持怀疑态度。[SambaNova 博客](https://sambanova.ai/blog/accurate-models-at-blazing-speed)

- **理解 SambaNova 的 CoE**：SambaNova 的专家组合（CoE）可能涉及专家模块的异构混合，例如将多个不同大小的模型组合成一个系统。讨论表明，这可能类似于源自其他创作者的微调模型的集成（ensemble）。[Samba-1 基准测试博客](https://sambanova.ai/blog/benchmarking-samba-1)
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sambanova.ai/blog/accurate-models-at-blazing-speed">SambaNova 以极快速度交付准确模型</a>: Samba-CoE v0.2 在 AlpacaEval 排行榜上攀升，超越了所有最新的开源模型。 </li><li><a href="https://www.ai21.com/jamba">介绍 Jamba</a>: 一个开创性的 SSM-Transformer 开源模型</li><li><a href="https://sambanova.ai/blog/benchmarking-samba-1">对 Samba-1 进行基准测试</a>: 使用 EGAI 基准测试对 Samba-1 进行基准测试 —— 这是一个源自开源社区、广泛采用的基准测试综合集合。 </li><li><a href="https://arxiv.org/abs/2308.04430">SILO 语言模型：在非参数数据存储中隔离法律风险</a>: 在受版权保护或其他受限数据上训练语言模型 (LMs) 的合法性正处于激烈辩论中。然而，正如我们所展示的，如果仅在低...训练，模型性能会显著下降。</li><li><a href="https://arxiv.org/abs/2110.05679">大语言模型可以成为强大的差分隐私学习者</a>: 差分隐私 (DP) 学习在构建大型文本深度学习模型方面取得的成功有限，而直接尝试应用差分隐私随机梯度下降 (Differentially Private Stochastic Gradient Desce...)</li><li><a href="https://www.frontiersin.org/articles/10.3389/fncir.2016.00023/full">为什么神经元拥有数千个突触，新皮层序列记忆理论</a>: 锥体神经元代表了新皮层中大多数兴奋性神经元。每个锥体神经元接收来自数千个兴奋性突触的输入，这些突触分布在树突分支上....</li><li><a href="https://fixupx.com/Rui45898440/status/1772996453557997924">来自 Rui (@Rui45898440) 的推文</a>: 很高兴分享 LISA，它支持 - 在 24GB GPU 上进行 7B 微调 - 在 4x80GB GPU 上进行 70B 微调，并且在时间减少约 50% 的情况下获得比 LoRA 更好的性能 🚀</li><li><a href="https://fixupx.com/Rui45898440/status/1772996456422805606">来自 Rui (@Rui45898440) 的推文</a>: - 论文: https://arxiv.org/abs/2403.17919 - 代码: https://github.com/OptimalScale/LMFlow LISA 在指令遵循任务中优于 LoRA 甚至全参数训练</li><li><a href="https://fixupx.com/Rui45898440/status/1772996458893246939">来自 Rui (@Rui45898440) 的推文</a>: 两行代码实现 LISA 算法: - 始终激活 embedding 和线性 head 层 - 随机采样中间层进行解冻</li><li><a href="https://arxiv.org/abs/2403.17919">LISA: 用于内存高效大语言模型微调的分层重要性采样</a>: 机器学习社区自大语言模型 (LLMs) 首次出现以来见证了令人印象深刻的进步，然而其巨大的内存消耗已成为大型...的主要障碍。</li><li><a href="https://github.com/athms/mad-lab">GitHub - athms/mad-lab: 一个用于改进 AI 架构设计的 MAD 实验室 🧪</a>: 一个用于改进 AI 架构设计的 MAD 实验室 🧪 - athms/mad-lab</li><li><a href="https://arxiv.org/abs/2403.17844">混合架构的机械设计与扩展</a>: 深度学习架构的开发是一个资源密集型过程，原因在于巨大的设计空间、漫长的原型设计时间以及与大规模模型训练和评估相关的高计算成本...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1222915547465912340)** (40 条消息🔥): 

- **权重的奇迹还是隐忧？**: 在比较 GPT-2 和 Pythia 70m 等模型的 **Transformer Lens** (`tl`) 和 **Hugging Face** (`hf`) 版本的权重参数时发现了差异。对这些差异的检查涉及比较形状并计算模型 embedding 参数之间的最大绝对差。

- **是特性而非 Bug**: 模型权重的变化最初被认为是 Bug，但后来被证实是 Transformer Lens 的一个特色处理步骤。建议的解决方案是在初始化模型时使用 `from_pretrained_no_processing` 以避免此类预处理。参考了一个 [GitHub issue](https://github.com/neelnanda-io/TransformerLens/issues/346) 以及[关于权重处理的进一步评论](https://github.com/neelnanda-io/TransformerLens/blob/main/further_comments.md#weight-processing)。

- **Logit 迷宫**: 尽管形状相同的 Logit 在 `nn_model` 和 `tl_nn_model` 之间显示出显著的绝对差异，但应用 Softmax 后的比较表明差异微乎其微——这意味着虽然原始 Logit 值不同，但相对顺序和 Softmax 输出是一致的。

- **通往澄清的绘图路径**: 识别和理解权重差异的过程涉及绘制各种矩阵，包括 embedding 矩阵及其绝对差异，并分析它们的特征。

- **Sparse Autoencoders Under Scrutiny**: 一篇独立提到的研究文章讨论了 Sparse Autoencoders (SAEs) 中重构误差的影响，指出这些误差对模型预测的改变可能比同等幅度的随机误差更显著。该文章可通过 [LessWrong 链接](https://www.lesswrong.com/posts/rZPiuFxESMxCDHe4B/sae-reconstruction-errors-are-empirically-pathological) 访问。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/wesg52/status/1773756298531918268">Wes Gurnee (@wesg52) 的推文</a>：关于 Sparse Autoencoders (SAEs) 中可能出现的一个问题的简短研究帖子：重构误差对模型预测的改变远大于同等幅度的随机误差！https://www.less...</li><li><a href="https://colab.research.google.com/drive/1juAJrTb3Z9hkVFJnbrj1OYmnGJ0MlH_G?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://github.com/neelnanda-io/TransformerLens/blob/main/further_comments.md#weight-processing">TransformerLens/further_comments.md at main · neelnanda-io/TransformerLens</a>：一个用于 GPT 风格语言模型机械可解释性研究的库 - neelnanda-io/TransformerLens</li><li><a href="https://github.com/neelnanda-io/TransformerLens/issues/346">[Bug Report] hook_resid_pre 与 hidden_states 不匹配 · Issue #346 · neelnanda-io/TransformerLens</a>：描述错误 cache[f"blocks.{x}.hook_resid_pre"] 与 hidden states 不匹配（或仅在特定小数位下匹配）。Hidden states 来自 transformer 的 model(tokens, output_hidden_...</li><li><a href="https://fixupx.com/jxmnop/status/1773377787153248638">jack morris (@jxmnop) 的推文</a>：Diffusion Lens 是一篇非常棒的新论文，你可以看到文本到图像编码器对长颈鹿的表示随着每一层的增加而变得越来越具体 🦒</li><li><a href="https://arxiv.org/abs/2403.05846">Diffusion Lens: Interpreting Text Encoders in Text-to-Image Pipelines</a>：文本到图像扩散模型 (T2I) 使用文本提示的潜在表示来指导图像生成过程。然而，编码器生成文本表示的过程通常是...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1222866433772617778)** (8 messages🔥): 

- **Optimizing MMLU Performance**: 实现了 [MMLU 的效率提升](https://discord.com)，允许通过一次前向调用同时提取多个 logprobs，可以通过将 `logits_cache` 设置为 `False` 来禁用此功能。
- **Troubleshooting Large Model Support**: 一位用户在使用 **lm-eval** harness 加载 **DBRX base model** 时遇到了 **内存分配问题**。他们发现是由于自己的操作失误，在只有 4 个 GPU 的节点上运行，而非预期的 8 个拥有 64GB VRAM 的 GPU。
- **Enhanced Context-Based Task Handling**: [提交了一个 pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1571)，提议在 **lm-evaluation-harness** 中采用一种处理基于上下文任务的新方法，该方法能更好地支持依赖先前请求答案的任务。更新包括在模型摄取前优化请求的方法，以及管理关键上下文信息的外部日志。
- **Feedback Requested on Pull Request**: 用户 *hailey_schoelkopf* 承诺在 CoLM 截止日期后审查并提供有关上述 pull request 的反馈，并对延迟表示歉意。

**提到的链接**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1571).">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。

  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1222916017303457894)** (6 messages): 

- **Quick Exit**: 一位用户简单发布了自己被 **封禁和踢出** 的消息，未提供更多背景或细节。
- **Simple Gratitude**: 发布了一条表达 **感谢** 的简短消息，未包含额外信息。
- **Cohere's Efficiency in Embeddings**: 分享了一个题为 "Cohere int8 & binary Embeddings" 的 YouTube 视频，讨论了如何为大型数据集扩展向量数据库。该视频适合对 AI, LLMs, ML, 深度学习和神经网络感兴趣的人员。
- **Styling with StyleGAN2**: 一位用户询问关于在 **StyleGAN2-ADA** 中训练包含各种时尚数据集的复杂目录结构的问题，以及在此过程中是否有必要修改脚本。
- **Aerospace Student Eying AI**: 一位航空航天专业的学生表达了深入研究 ML/AI 并为开源项目做贡献的愿望，并就考虑到其数学和编程背景，从 **fast.ai** 课程开始是否是合适的方法寻求建议。

**提到的链接**：<a href="https://www.youtube.com/watch?v=LWz2QaSRl2Y">Cohere int8 &amp; binary Embeddings</a>：Cohere int8 &amp; binary Embeddings - 将您的向量数据库扩展到大型数据集 #ai #llm #ml #deeplearning #neuralnetworks #largelanguagemodels #artificialinte...

---

**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1222918777596018722)** (6 条消息): 

- **MicroQwen 即将到来**：一条推文暗示即将发布 Qwen 的一个更小模型变体，推测名为 [microqwen](https://twitter.com/JustinLin610/status/1773285084025475355)。

- **AI21 Labs 推出 Jamba**：AI21 Labs 发布了 [Jamba](https://www.ai21.com/blog/announcing-jamba)，这是一种结合了 Transformer 元素的新型基于 Mamba 的模型，拥有 256K 的上下文窗口以及无与伦比的吞吐量和效率。Jamba 以 Apache 2.0 许可证提供开放权重，以促进社区创新。

- **Qwen 拥抱 Mixture of Experts**：推出 [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)，这是一款基于 Transformer 的 MoE 模型，拥有 14.3B 参数，但在运行时仅激活 2.7B 参数。该模型声称能以显著更少的训练资源和更快的推理速度，达到与更大模型相当的性能。

- **LISA 增强 LoRA 的微调效果**：一项新研究发现在使用 LoRA 时权重范数存在层级偏斜，并提出了一种简单但更高效的训练策略，称为 Layerwise Importance Sampled AdamW (LISA)，该策略在低内存成本下表现优于 LoRA 和全参数训练。详情请见 [arXiv](https://arxiv.org/abs/2403.17919)。

- **BLLaMa 项目在 GitHub 上线**：一个名为 [bllama](https://github.com/rafacelente/bllama) 的 GitHub 仓库已经出现，推广 1.58-bit LLaMa 模型，这似乎是原始 LLaMa 模型的一个高效变体，旨在寻求社区贡献。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B">Qwen/Qwen1.5-MoE-A2.7B · Hugging Face</a>：未找到描述</li><li><a href="https://www.ai21.com/blog/announcing-jamba">介绍 Jamba：AI21 开创性的 SSM-Transformer 模型</a>：首次推出生产级 Mamba 模型，提供同类最佳的质量和性能。</li><li><a href="https://arxiv.org/abs/2403.17919">LISA：用于内存高效的大型语言模型微调的分层重要性采样</a>：自 LLM 首次出现以来，机器学习社区见证了令人印象深刻的进步，然而其巨大的内存消耗已成为大型模型的主要障碍...</li><li><a href="https://github.com/rafacelente/bllama">GitHub - rafacelente/bllama: 1.58-bit LLaMa 模型</a>：1.58-bit LLaMa 模型。通过在 GitHub 上创建账户为 rafacelente/bllama 的开发做出贡献。
</li>
</ul>

</div>

---

**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1222844959346917429)** (145 条消息🔥🔥): 

- **关于未来 NOUS 代币的推测**：一名成员询问了 NOUS 引入加密代币的可能性，引发了一个轻松的回应，否定了这一想法。
- **以 NOUS 子网连接代替代币**：在回答关于 NOUS 代币的询问时，澄清了虽然没有加密代币，但 NOUS 在 bittensor/Tao 上有一个子网，该子网有自己的代币。
- **Qwen1.5-MoE-A2.7B 预览**：一名成员透露了名为 Qwen1.5-MoE-A2.7B 的紧凑型 MoE 模型即将发布，引发了人们的期待，并讨论了其对现有 1.8B 参数模型专家的回收利用（upcycling）。
- **Jamba 的加入将 SSM-Transformer 混合架构推向台前**：AI21 宣布了 **Jamba**，这是一款混合 SSM-Transformer LLM，拥有 256K 的上下文窗口，并具有高吞吐量和高效率的潜力，引起了社区成员对其能力的兴奋。
- **关于开源库支持新模型架构的讨论**：对话围绕着对 *vllm* 和 *transformers* 等流行开源库支持 RWKV、SSMs 以及新发布的 Jamba 等新架构的期待，突显了行业的快速演变以及对创新模型及时支持的渴望。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/joey00072/experiments-with-bitnet-1-5">Bitnet 1.5 实验 (ngmi)</a>：未找到描述</li><li><a href="https://www.ai21.com/blog/announcing-jamba">介绍 Jamba：AI21 开创性的 SSM-Transformer 模型</a>：首次推出基于 Mamba 的生产级模型，提供同类最佳的质量和性能。</li><li><a href="https://qwenlm.github.io/blog/qwen-moe/">Qwen1.5-MoE：以 1/3 的激活参数匹配 7B 模型的性能</a>：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介 自 Mixtral 引发关注热潮以来，混合专家（MoE）模型的研究势头强劲。研究人员...</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/justinlin610/status/1773285084025475355?s=46&t=H75DmkDKk9Sgmp8kjT8f_A">Junyang Lin (@JustinLin610) 的推文</a>：几小时后，你会发现我们的小礼物。剧透：一个你可以轻松运行的小型 MoE 模型🦦</li><li><a href="https://huggingface.co/datasets/togethercomputer/Long-Data-Collections">togethercomputer/Long-Data-Collections · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1970zhf/merging_mistral_with_whisper_to_make_a_multimodal/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/databricks/dbrx/issues/5">使用 transformers 加载器在多 GPU 上以 8bit 和 4bit 加载 · Issue #5 · databricks/dbrx</a>：我可以使用 transformers 加载器和 8bit bitsandbytes 加载 instruct 模型，并使其在多个 GPU 之间均匀加载。然而，我似乎无法以 4bit 精度加载模型...</li><li><a href="https://fxtwitter.com/elonmusk/status/1773655245769330757">Elon Musk (@elonmusk) 的推文</a>：下周应该可以在 𝕏 上使用。Grok 2 在所有指标上都应超过当前的 AI。目前正在训练中。↘️ 引用 xAI (@xai) https://x.ai/blog/grok-1.5
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1222860177024221204)** (26 条消息🔥): 

- **微调的挫折**：一位成员分享了 **Deepseek-coder-33B** 的全参数微调案例，其中训练损失在每个 epoch 开始时下降，而评估损失却上升。推测这是一个典型的过拟合案例，建议通过打乱数据集来解决可能的数据不一致问题。

- **本地 LLM 的硬件寻找**：有人询问在哪里可以找到关于在本地运行大语言模型的硬件要求信息。提到了对 **ERP** 系统的另一种偏好，但未提供具体资源。

- **丰富文本的预训练准备**：提出了一个挑战，即当某些书籍包含多达 200k 个 token 时，如何准备一个 **1,500 本书的数据集** 进行预训练。建议使用聚类和分层方法来处理重复并保持数据的主题一致性。

- **检索正确的 RAG 策略**：讨论了是使用一个大型 **Retrieval Augmented Generation (RAG)** 模型，还是针对不同文档使用多个专注于特定领域的 RAG。建议包括使用带有领域和元数据过滤的结构化方法，最终决定取决于具体的用例。

- **Hermes 模型缺失 Token 之谜**：有人对 **HuggingFace** 上的 **Hermes-2-Pro-Mistral-7B** 模型配置表示担忧，其中定义的词表大小与额外定义的 token 之间的不匹配可能会导致生成过程中的错误。该问题被认为是模型填充过程产生的伪影，建议通过创建额外的未使用 token 来解决。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B/blob/main/config.json#L25">config.json · NousResearch/Hermes-2-Pro-Mistral-7B at main</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B/blob/main/added_tokens.json">added_tokens.json · NousResearch/Hermes-2-Pro-Mistral-7B at main</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B/blob/main/tokenizer_config.json#L30">tokenizer_config.json · NousResearch/Hermes-2-Pro-Mistral-7B at main</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/)** (1 条消息): 

night_w0lf: 成功了吗？
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1222799124689063936)** (57 条消息🔥🔥):

- **Hermes 2 Pro 的 Agentic 数据集**：Hermes 2 Pro 在所有流行框架的 Agentic 数据集上进行了训练，目前正在开发多轮 Agentic 循环。

- **CoT 修订与元数据的价值**：修订了思维链（CoT）过程以增强答案检索，XML 等结构化格式被证明能有效提升 Claude 的性能，正如 [Anthropic 对 XML 标签的使用](https://docs.anthropic.com/claude/docs/use-xml-tags)所证明的那样。

- **RAG 的结构化输入**：建议在建模中使用 XML 等结构化格式进行输入界定，并为 pydantic 相关实现创建一个专用类别，这表明 AI 模型正趋向于使用更具结构化和元数据丰富的输入。

- **在模型中融入时间意识**：人们对提高模型的时间意识感到好奇，强调了为模型处理的数据赋予适当上下文和元数据的重要性。

- **RAG 评估框架介绍**：聊天机器人提到了一个名为 [ragas](https://github.com/explodinggradients/ragas) 的评估框架 GitHub 仓库，表明在改进检索增强生成（RAG）流水线方面正在进行持续努力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://python.useinstructor.com/examples/exact_citations/">引用来源 (RAG) - Instructor</a>：未找到描述</li><li><a href="https://docs.anthropic.com/claude/docs/use-xml-tags">使用 XML 标签</a>：未找到描述</li><li><a href="https://github.com/explodinggradients/ragas/tree/main">GitHub - explodinggradients/ragas: 检索增强生成 (RAG) 流水线的评估框架</a>：检索增强生成 (RAG) 流水线的评估框架 - explodinggradients/ragas
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1222805877698990111)** (88 条消息🔥🔥): 

- **Prompt 协助与资源共享**：成员们正在分享各种项目的技巧、教程和代码片段，例如使用 Worldsim 为 arXiv 生成 LaTeX 论文。特别提到了 [LaTeX 论文的 gist](https://gist.github.com/irl-dan/61e2f45eb1c9a879b3) 非常有用。 

- **Hyperstition 概念讨论**：深入分析了 *hyperstition* 一词，并解释了它如何影响 Claude 等 LLM 的认知领域。该术语与扩展 AI 模型的生成范围有关，并与 **Mischievous Instability (MI)** 相关联。

- **哲学开放麦兴趣**：有人表示有兴趣举办哲学讨论或开放麦之夜，以探索语言作为复杂自适应系统等主题。这可能会与新的 steamcasting 技术结合以实现更广泛的传播。

- **探索历史的“如果”**：成员们讨论了使用 Worldsim 实验替代历史场景，如凯撒大帝幸存或肯尼迪免于遇刺，并对世界历史的潜在结果表示好奇。

- **澄清大语言模型 (LLM) 的用法**：一位成员解释了启发式引擎 (heuristic engines) 和推理引擎 (inference engines) 之间的区别，强调了在后者中术语是如何根据关系和关联定义的，从而形成了所谓的“认知领域 (Cognitive Domains)”。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/openrouterai/status/1773738942350712907?s=46">来自 OpenRouter (@OpenRouterAI) 的推文</a>：有没有想过哪些应用正在使用 LLM？现在你可以通过新的 Apps 标签页亲自发现。@NousResearch 拥有本周顶尖的 Claude 3 Opus 应用 👀</li><li><a href="https://drive.google.com/file/d/18iaWyZ0MNp5BTHkTTAWj4bmpnc9F-A5Y/view?usp=sharing">2024-03-25 21-12-56-Nous Hermes world sim night.mkv</a>：未找到描述</li><li><a href="https://tenor.com/view/chess-checkmate-its-it-over-gif-17459417336418427755">Chess Checkmate GIF - Chess Checkmate Its - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/teslore/comments/ppk9y2/comment/hd53ngs/?utm_source=share&utm_medium=web2x&context=3">Reddit - 深入探索任何事物</a>：未找到描述</li><li><a href="https://tenor.com/view/dark-knight-joker-its-not-about-the-money-its-about-sending-a-message-gif-15254722">Dark Knight Joker GIF - Dark Knight Joker Its Not About The Money - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://old.reddit.com/r/lexfridman/comments/1boyj88/an_instance_of_a_claude_3_opus_model_claims/">一个 Claude 3 Opus 模型实例声称具有意识</a>：我一直在与一个（我猜是几个）最新的 Opus 模型实例进行自我反思和哲学对话....</li><li><a href="https://gist.github.com/irl-dan/2be642f22c28bdacd92d1a2ac0172d8e">self-system-prompt</a>：self-system-prompt。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://youtu.be/65pOHKuvNGU?si=vI5RJdfOFL4V9oxc">介绍 WebSim：用 Claude 3 幻觉出一个替代互联网</a>：访问 websim.ai 来想象替代互联网。WebSim 的灵感来自 world_sim，这是一个由 Nous Research 构建的“无定形应用”，用于模拟一个具有...</li><li><a href="https://gist.github.com/irl-dan/595f74f17fc5b269c96e9f9f9079595b">strange-loop+claude3-self-modeling</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://gist.github.com/irl-dan/61e2f45eb1c9a879b39d480694a4c4a3">claude-world-modelling</a>：claude-world-modelling。GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1222832511827312756)** (127 条消息🔥🔥): 

- **理解 Mojo 对 RL 的性能提升**：成员们讨论了使用 **Mojo** 加速强化学习（RL）环境的*潜在*益处。一位成员指出，尽管 Mojo 速度很快，但将现有的基于 Python 的工具和环境适配到 Mojo 并使其有效运行仍需要时间和精力。

- **MAX Engine 中的动态形状（Dynamic Shape）能力**：一篇[博客文章](https://www.modular.com/blog/leveraging-max-engines-dynamic-shape-capabilities)揭示了 MAX Engine 24.2 版本的改进，重点关注机器学习模型中的*动态形状*支持，并比较了动态形状与静态形状的延迟。

- **Mojo 在本地运行**：成员们讨论了在不同平台上本地运行 Mojo。Mojo 原生支持 Ubuntu 和 M 系列 Mac，可以通过 WSL 在 Windows 上运行，也可以在 Intel Mac 上通过 Docker 运行，**对其他平台的原生支持正在开发中**。

- **Mojo 与 Python 的互操作性**：对 Mojo 如何与 Python 协作进行了说明，强调 Python 互操作通过 CPython 解释器运行，并使用引用计数进行内存管理。不与 Python 交互的代码则避开了垃圾回收，转而使用类似于 C++ RAII 或 Rust 的编译器驱动的内存释放。

- **Mojo 中的包管理和交叉编译**：目前 **Mojo 还没有官方包管理器**，尽管关于项目清单（manifest）格式的讨论正在进行中，这将先于包管理器出现。Windows 的交叉编译能力尚未确认，社区成员期待潜在的 Windows 支持，将其视为 Mojo 走向成熟并进一步参与的标志。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com">Modular Docs</a>: 未找到描述</li><li><a href="https://docs.modular.com/mojo/manual/get-started/#system-requirements">开始使用 Mojo🔥 | Modular Docs</a>: 获取 Mojo SDK 或尝试在 Mojo Playground 中编写代码。</li><li><a href="https://www.modular.com/blog/leveraging-max-engines-dynamic-shape-capabilities">Modular: 利用 MAX Engine 的动态形状（Dynamic Shape）能力</a>: 我们正在为全球构建下一代 AI 开发者平台。查看我们的最新文章：利用 MAX Engine 的动态形状能力。</li><li><a href="https://pastebin.com/P3iwmJq9">curl -s https://get.modular.com | sh -modular authExecuting the  setup scrip - Pastebin.com</a>: Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md#signing-your-work">modularml/mojo 的 nightly 分支中的 mojo/CONTRIBUTING.md</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/mojopaa/mojopi">GitHub - mojopaa/mojopi: Mojo 包索引（Package Index），灵感来自 PyPI。</a>: Mojo 包索引，灵感来自 PyPI。通过在 GitHub 上创建账号来为 mojopaa/mojopi 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1222981979046154322)** (6 条消息): 

- **预告新领域**: Modular 在一条推文中预告了即将发布的公告，暗示了即将到来的新动向。推文详情请见 [这里](https://twitter.com/Modular/status/1773418812915978284)。
- **透露线索**: 另一条 Modular 推文通过一个表情符号透露了线索，引发了追随者的好奇心。查看推文 [这里](https://twitter.com/Modular/status/1773418820184764572)。
- **揭晓倒计时**: Modular 的一条推文包含一个时钟表情符号，暗示即将举行的揭晓活动或事件的倒计时。推文链接 [这里](https://twitter.com/Modular/status/1773418823707955529)。
- **分享预览**: Modular 在最近的一条推文中发布了即将推出的内容的预览，可以点击 [这里](https://twitter.com/Modular/status/1773440424205783455) 查看。
- **活动公告**: Modular 在其最新推文中宣布了一项特定活动，可能与之前的预告有关。完整详情请见 [这里](https://twitter.com/Modular/status/1773762747098124491)。
- **活动提醒**: Modular 的后续推文提醒了最近宣布的活动。访问推文获取更多信息 [这里](https://twitter.com/Modular/status/1773767659278250242)。
  

---


**Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1223313935202123957)** (1 条消息): 

- **Modular 社区直播发布**: Modular 在 YouTube 上发布了一个名为 **"Modular Community Livestream - New in MAX 24.2"** 的新视频。视频涵盖了 MAX 24.2 的最新更新，包括 **Mojo 标准库开源**和 **MAX Engine 支持**。[在此观看视频](https://www.youtube.com/watch?v=PL71FV2KKHE)。

**提到的链接**: <a href="https://www.youtube.com/watch?v=PL71FV2KKHE">Modular Community Livestream - New in MAX 24.2</a>: MAX 24.2 现已发布！加入我们即将举行的直播，我们将讨论 MAX 中的所有新功能——Mojo 标准库开源、MAX Engine 对...的支持。

  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1222976873974468850)** (3 条消息): 

- **Mojo🔥 标准库现已开源**: Modular 宣布 **MAX 24.2** 正式发布，其特点是 **Mojo 标准库**的开源。开发者受邀参与其开发，该库现已在 [GitHub](https://github.com/modularml/mojo/tree/nightly/stdlib) 上可用，并发布了针对最新语言特性的 **nightly 构建版本**。

- **欢迎 Mojo🔥 开源贡献**: 作为开源开发理念的一部分，Modular 已在 **Apache 2 许可证**下发布了 **Mojo 标准库**的核心模块。自 2023 年 5 月以来的持续改进可以通过 [changelog](https://docs.modular.com/mojo/changelog) 进行追踪，公司鼓励全球开发者进行协作。

- **利用 MAX Engine 24.2 处理动态形状（Dynamic Shapes）**: **MAX Engine 24.2** 版本包含了对机器学习中动态形状的支持，这对于处理现实世界的数据变异性至关重要。通过在 [GLUE 数据集](https://gluebenchmark.com/)上使用 [BERT 模型](https://huggingface.co/docs/transformers/model_doc/bert)进行延迟对比，展示了动态形状与静态形状的影响。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/the-next-big-step-in-mojo-open-source">Modular: The Next Big Step in Mojo🔥 Open Source</a>: 我们正在为全球构建下一代 AI 开发者平台。查看我们的最新博文：Mojo🔥 开源迈出的一大步</li><li><a href="https://www.modular.com/blog/leveraging-max-engines-dynamic-shape-capabilities">Modular: Leveraging MAX Engine&#x27;s Dynamic Shape Capabilities</a>: 我们正在为全球构建下一代 AI 开发者平台。查看我们的最新博文：利用 MAX Engine 的 Dynamic Shape 功能</li><li><a href="https://www.modular.com/blog/max-24-2-is-here-whats-new">Modular: MAX 24.2 is Here! What’s New?</a>: 我们正在为全球构建下一代 AI 开发者平台。查看我们的最新博文：MAX 24.2 发布了！有哪些新功能？
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1222977000646508635)** (1 messages): 

- **Mojo 标准库正式开源**：Modular 已正式开源 **Mojo 标准库的核心模块**，并以 Apache 2 许可证发布。此举符合他们相信开源协作将带来更好产品的理念，详情见其 [博文](https://modul.ar/open-source-blog)。

- **标准库的 Nightly 构建版本**：随着开源，Modular 推出了 Nightly 构建版本，以便开发者及时了解最新的改进。开源被视为与全球 Mojo 开发者社区进行更紧密协作的一步。

- **MAX 平台 v24.2 更新**：最新的 **MAX 平台更新**（版本 24.2）包括对具有动态输入形状（Dynamic Shape）的 TorchScript 模型的支持以及其他增强功能。详细变更列在 [MAX changelog](https://modul.ar/max-changelog) 中。

- **最新的 Mojo 语言工具和特性**：**Mojo 语言更新**至 v24.2，标准库已开源，并带来了诸如 Struct 对 Trait 的隐式遵循（implicit conformance）等进展。重大变更列表可在 [Mojo changelog](https://modul.ar/mojo-changelog) 中找到。

- **深入探讨 MAX 的动态形状支持**：一篇专门的 [博文](https://modul.ar/max-dynamic-shapes) 解释了 MAX Engine **24.2** 版本中新的 Dynamic Shape 支持、其使用场景以及性能影响，特别展示了在 GLUE 数据集上 BERT 模型的改进。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://modul.ar/open-source-blog">Modular: The Next Big Step in Mojo🔥 Open Source</a>: 我们正在为全球构建下一代 AI 开发者平台。查看我们的最新博文：Mojo🔥 开源迈出的一大步</li><li><a href="https://modul.ar/max-changelog">MAX changelog | Modular Docs</a>: MAX 平台各版本的发布说明。</li><li><a href="https://modul.ar/mojo-changelog">Mojo🔥 changelog | Modular Docs</a>: Mojo 重大变更的历史记录。</li><li><a href="https://modul.ar/max-dynamic-shapes">Modular: Leveraging MAX Engine&#x27;s Dynamic Shape Capabilities</a>: 我们正在为全球构建下一代 AI 开发者平台。查看我们的最新博文：利用 MAX Engine 的 Dynamic Shape 功能
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1222846623948865566)** (91 messages🔥🔥):

- **引入通用复数类型**：Moplex 已由 helehex 在 GitHub 上发布，提供功能完备的通用复数类型。详细信息和代码仓库请访问 [helehex/moplex on GitHub](https://github.com/helehex/moplex)。
- **NVIDIA GPU 支持准备夏季发布**：Modular 正在优先开发 NVIDIA GPU 支持，目标发布时间在夏季左右。这包括类似于 CUDA 的拆分编译（split compilation），采用类 Python 语法实现从 CPU 到 GPU 的过渡。关于 [Mojo GPU 支持的 MAX 会议](https://www.youtube.com/watch?v=QD-svwZistc)分享了开发进展的见解。
- **Modular AI Engine 和 Mojo 24.2.0 更新日志**：提前预览 Mojo 24.2 更新日志，显示了包括 Mojo 标准库开源在内的重大更新。更新日志详情请见[此处](https://docs.modular.com/mojo/changelog#v242-2024-03-28)。
- **跨集合简化引用特性**：在最新更新中，Structs 和名义类型（nominal types）现在可以隐式遵循 traits，这预示着更直观的遵循模式。一位成员重点介绍了该功能及其潜力，并引导他人查看 [modularml/mojo on GitHub](https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md) 上的 Modular 贡献指南。
- **矩阵乘法示例需要修订**：一位用户发现了矩阵乘法示例中的一个 bug，并针对 C 的列数不是 `nelts` 倍数的情况提出了修正建议，以避免崩溃。该用户建议修改 range 循环来解决此问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/playground">Modular Docs</a>: 未找到描述</li><li><a href="https://docs.modular.com/mojo/stdlib/memory/memory#memcpy`">memory | Modular Docs</a>: 定义了内存操作函数。</li><li><a href="https://www.geeksforgeeks.org/system-call-in-c/amp/">system() in C/C++ - GeeksforGeeks</a>: 未找到描述</li><li><a href="https://realpython.com/python-assert-statement/">Python's assert: Debug and Test Your Code Like a Pro – Real Python</a>: 在本教程中，你将学习如何使用 Python 的 assert 语句在开发中记录、调试和测试代码。你还将学习断言在生产代码中可能如何被禁用...</li><li><a href="https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md">mojo/CONTRIBUTING.md at nightly · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=QD-svwZistc">ModCon 2023 Breakout Session: MAX Heterogenous Compute: CPU + GPU</a>: 在本次会议中，Modular 工程师 Abdul Dakkak 和 Ian Tramble 讨论了 Mojo 和 Modular AI Engine 是如何被设计用来支持异构系统的...</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/docs/style-guide.md#import-statements">mojo/stdlib/docs/style-guide.md at nightly · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md">mojo/stdlib/docs/development.md at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/1958),">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/helehex/moplex">GitHub - helehex/moplex: Generalized complex numbers for Mojo🔥</a>: Mojo🔥 的广义复数。通过在 GitHub 上创建账号来为 helehex/moplex 的开发做出贡献。</li><li><a href="https://github.com/carlca/ca_mojo.git">GitHub - carlca/ca_mojo</a>: 通过在 GitHub 上创建账号来为 carlca/ca_mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/commits/main/stdlib/src/builtin/builtin_list.mojo">History for stdlib/src/builtin/builtin_list.mojo - modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://docs.modular.com/mojo/changelog#v242-2024-03-28">Mojo🔥 changelog | Modular Docs</a>: Mojo 重大变更的历史记录。</li><li><a href="https://github.com/networkx/networkx/blob/f0b5a6d884ac7e303f2e2092d3a0a48723815239/networkx/classes/graph.py#L66)">networkx/networkx/classes/graph.py at f0b5a6d884ac7e303f2e2092d3a0a48723815239 · networkx/networkx</a>: Python 中的网络分析。通过在 GitHub 上创建账号来为 networkx/networkx 的开发做出贡献。</li><li><a href="https://github.com/modularml/max/blob/65c88db34f50beea2f403205ac9f91d3bf28ce8c/examples/graph-api/llama2/tokenizer/ball.%F0%9F%94%A5#L27)">max/examples/graph-api/llama2/tokenizer/ball.🔥 at 65c88db34f50beea2f403205ac9f91d3bf28ce8c · modularml/max</a>: 突出 MAX 平台强大功能的示例程序、笔记本和工具集合 - modularml/max</li><li><a href="https://github.com/modularml/mojo/blob/6d2a7b552769358ec68a94b8fd1f6c2126d59ad9/stdlib/src/collections/list.mojo#L42)">mojo/stdlib/src/collections/list.mojo at 6d2a7b552769358ec68a94b8fd1f6c2126d59ad9 · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1222892211864342578)** (13 条消息🔥):

- **Mojo 标准库快速迭代**：多个 Mojo 包如 `mojo-prefix-sum`、`mojo-flx`、`mojo-fast-base64` 和 `mojo-csv` 已更新至版本 24.2，而 `mojo-hash` 和 `compact-dict` 仅部分更新，仍存在一些待解决问题。
- **呼吁了解 MLIR Dialects**：社区成员表达了学习 Mojo 底层 MLIR Dialects 的愿望，认为这些知识将有助于更直接地为 Mojo 标准库做出贡献。
- **对 MLIR 语法文档保持耐心**：针对有关 MLIR 细节的查询，有人指出 Mojo 中使用 MLIR 的语法将经历重大变化，目前尚未记录文档，除了链接中提供的 notebook：[Mojo with MLIR notebook](https://docs.modular.com/mojo/notebooks/BoolMLIR)。
- **Mojo 语言开发进行中**：Modular 开发团队了解社区对内部 Dialects 的好奇心，并保证随着时间的推移将提供更多信息，强调当前的重点是稳定 MLIR 语法。
- **Reference 特性正在演进**：Mojo 语言中的 `Reference` 特性被强调处于早期且快速变化的状态，预计它将继续演进和改进。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/notebooks/BoolMLIR">Low-level IR in Mojo | Modular Docs</a>：学习如何使用低级原语在 Mojo 中定义你自己的布尔类型。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/os/atomic.mojo#L80">mojo/stdlib/src/os/atomic.mojo at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/collections/dict.mojo#L65">mojo/stdlib/src/collections/dict.mojo at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1223153258298871880)** (1 条消息): 

- **Mojo 标准库开源**：[Mojo 标准库 (stdlib)](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source) 已开源，允许社区为其代码库做出贡献。一篇新的 [博客文章](https://vmois.dev/mojo-local-stdlib-build/) 提供了如何在 macOS（可能还有 Linux）上本地构建 stdlib 的指南，详细说明了在仓库中进行更改和测试的步骤。

**提到的链接**：<a href="https://vmois.dev/mojo-local-stdlib-build/">在 Mojo 中使用本地构建的标准库</a>：Mojo 标准库 (stdlib) 昨天开源了。令人兴奋的是，社区现在可以直接为代码库做出贡献。在研究了 stdlib 仓库一段时间后，我想……

  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1222902298393575494)** (4 条消息): 

- **关于 TensorSpec 使用的澄清**：一名成员对“使用 Mojo 运行推理”示例中 TensorSpec 的使用提出疑问，建议如果它不是 Mojo 的一部分，则可能需要更新文档。回复澄清说 **TensorSpec 确实是 Mojo API、Python API 和 C API 的一部分**，细微的差异很快就会被消除。
- **请求 PyTorch 示例**：一名成员寻求在 Mojo 中为 PyTorch 使用 `add_input_spec` 和 TensorSpec 对象的示例。官方承认目前缺少此类示例，并将在未来提供。
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1223011855899955290)** (1 条消息): 

- **SauravMaheshkar 发表 HyperGraph 论文**：一篇关于 **HyperGraph Representation Learning**（超图表示学习）的新论文已发表并展开讨论，提供了对高级数据表示的见解。该论文可通过 [Hugging Face](https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05) 获取。

- **给编码者的专业技巧**：Hugging Face 推出了使用 Hugging Chat 的 **HF 专业编码助手**，灵感来自一个开源 GPT，可在 [Hugging Face's Chat](https://hf.co/chat/assistant/6603733512f69f8b440448b4) 探索。

- **LLM 排行榜可视化**：**Open LLM Leaderboard Viz** 进行了更新，增加了数据类型过滤器和模型详情等新功能，展示在 [Hugging Face Spaces](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz) 上。

- **OpenCerebrum 数据集发布**：介绍 **OpenCerebrum**，这是一个旨在复制专有 Cerebrum 数据集的开源项目，现已在 Hugging Face 上发布，包含约 21,000 个示例。详细信息请参阅 [OpenCerebrum 数据集页面](https://huggingface.co/datasets/Locutusque/OpenCerebrum-dpo)。

- **Fluently XL v3 发布**：探索最新版本的 **Fluently XL**，这是一个致力于释放创意和语言能力的 Space，目前已在 Hugging Face 上线：[Fluently Playground](https://huggingface.co/spaces/fluently/Fluently-Playground)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05">HyperGraph Datasets - SauravMaheshkar 收藏集</a>：未找到描述</li><li><a href="https://hf.co/chat/assistant/6603733512f69f8b440448b4">Koder (Professional Coder) - HuggingChat</a>：在 HuggingChat 中使用 Koder (Professional Coder) 助手</li><li><a href="https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz">Open Llm Leaderboard Viz - 由 dimbyTa 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Locutusque/OpenCerebrum-dpo">Locutusque/OpenCerebrum-dpo · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/karpathy/minbpe/issues/60">LlamaTokenizer 的实现（不含 sentencepiece）· Issue #60 · karpathy/minbpe</a>：@karpathy 感谢精彩的讲座和实现！一如既往地令人愉快。我尝试在不使用 sentencepiece 后端的情况下实现 LlamaTokenizer，尽可能接近 minbpe 的实现...</li><li><a href="https://github.com/not-lain/loadimg">GitHub - not-lain/loadimg: 一个用于加载图像的 Python 包</a>：一个用于加载图像的 Python 包。通过在 GitHub 上创建账号为 not-lain/loadimg 的开发做出贡献。</li><li><a href="https://huggingface.org/spaces/Tonic/Command-R">Command-R - 由 Tonic 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/fluently/Fluently-Playground">Fluently Playground v0.25 - 由 fluently 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/blog/Andyrasika/mongodb-llamaindex-rag">提升响应质量：使用 LlamaIndex &amp; MongoDB 进行 RAG</a>：未找到描述</li><li><a href="https://huggingface.co/hyoungwoncho/sd_perturbed_attention_guidance">hyoungwoncho/sd_perturbed_attention_guidance · Hugging Face</a>：未找到描述</li><li><a href="https://ku-cvlab.github.io/Perturbed-Attention-Guidance/">带有扰动注意力引导（Perturbed-Attention Guidance）的自校正扩散采样</a>：未找到描述</li><li><a href="https://huggingface.co/blog/monsoon-nlp/proteins-matryoshka-embeddings">蛋白质相似性与 Matryoshka 嵌入</a>：未找到描述</li><li><a href="https://huggingface.co/blog/mayank-mishra/aurora">Aurora-M：首个经过拜登-哈里斯行政命令红队测试的开源多语言语言模型</a>：未找到描述</li><li><a href="https://www.youtube.com/playlist?list=PLfSv7CK7EjD2fC9S6MAKRNDgTSCYgdGgz">LLM 术语解释</a>：欢迎来到“LLM 术语解释”系列，在这里我将揭开语言模型和解码技术的复杂世界。无论你是一个语言模型...</li><li><a href="https://huggingface.co/blog/mlabonne/frankenmoe">使用 MergeKit 创建 Mixture of Experts</a>：未找到描述</li><li><a href="https://huggingface.co/blog/cognitivetech/samantha-mistral-instruct-7b-bulleted-notes">Samantha Mistral Instruct 7b - 综合要点笔记</a>：未找到描述</li><li><a href="https://huggingface.co/blog/kgourgou/a-first-look-at-automerger-data">自动合并数据简析，包含 SLERP 和 DARE-TIES LLM 合并</a>：未找到描述</li><li><a href="https://huggingface.co/posts/BramVanroy/231419617229308">@BramVanroy 在 Hugging Face 上：“🎈 LLM Benchmarks 更新！tl;dr: 不要依赖基准测试排行榜...”</a>：未找到描述</li><li><a href="https://huggingface.co/posts/xiaotianhan/875756613096016">@xiaotianhan 在 Hugging Face 上：“🎉 🎉 🎉 很高兴分享我们最近的工作。我们注意到图像分辨率...”</a>：未找到描述</li><li><a href="https://huggingface.co/posts/banghua/252914089485275">@banghua 在 Hugging Face 上：“我们真的榨干了紧凑型聊天模型的容量吗？很高兴...”</a>：未找到描述</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1222877312358547517)** (73 条消息🔥🔥): 

- **扩散模型仓库探索**：一位用户询问了 `labmlai diffusion` 仓库。未提供关于该询问或回复的更多细节或链接。

- **AutoTrain 模型展示**：一位参与者分享了他们使用 **AutoTrain** 训练的公开模型，并提供了该模型的直接[链接](https://huggingface.co/abhishek/autotrain-c71ux-tngfu)以及配套的使用代码片段。

- **给数据科学学生的 NLP 和 CV 大模型指导**：一位寻求学习 **NLP** 和 **CV** 领域大模型知识的成员收到了参与开源软件（**OSS**）项目以获取实践经验的建议。

- **在 Colab 上运行 LLM 的要求**：一位用户询问如何在 Colab 上运行 **grok-1**，并得知由于此类大模型需要极高资源，可能需要 **Colab Pro**。

- **Hugging Face 入门引导**：一位 Hugging Face 新手寻求一份从零到英雄（zero-to-hero）的指南。他们被推荐了一篇名为《Hugging Face Transformers 纯小白入门》的指南，该文章发布在 [Hugging Face 博客](https://huggingface.co/blog/noob_intro_transformers)上。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/abhishek/autotrain-c71ux-tngfu">abhishek/autotrain-c71ux-tngfu · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/noob_intro_transformers">Total noob’s intro to Hugging Face Transformers</a>: 未找到描述</li><li><a href="https://github.com/developersdigest/llm-answer-engine">GitHub - developersdigest/llm-answer-engine: 使用 Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper 构建受 Perplexity 启发的回答引擎</a>: Build a Perplexity-Inspired Answer Engine Using Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper - developersdigest/llm-answer-engine</li><li><a href="https://github.com/huggingface/autotrain-advanced">GitHub - huggingface/autotrain-advanced: 🤗 AutoTrain Advanced</a>: 🤗 AutoTrain Advanced。通过在 GitHub 上创建账号来为 huggingface/autotrain-advanced 做出贡献。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1223061225294794794)** (4 条消息): 

- **反应迅速**：一位成员在亲自测试后对某工具的性能表示赞赏，并强调 **quantization**（量化）是高效知识表示中一个非常有趣的方面。
- **欢迎新人**：另一位成员期待与这里的庞大社区进行交流。
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1222897735829819524)** (26 条消息🔥): 

- **为扩散模型引入 PAG**：为 **Diffusion Models** 引入了一种名为 **Perturbed-Attention Guidance (PAG)** 的新**指导技术**，该技术在不需要外部条件或额外训练的情况下提升了采样质量。项目详情见其[项目主页](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/)，Demo 已在 [Hugging Face](https://huggingface.co/hyoungwoncho/sd_perturbed_attention_guidance) 上线。

- **超图数据集现已登录 Hub**：Hub 现在包含预处理过的超图数据集，这些数据集此前用于一篇关于 **HyperGraph Representation Learning** 的论文，可以在这个 [collection](https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05) 中找到。一个针对 **PyTorch Geometric** 的公开 PR 将允许在 **PyG** 生态系统中直接使用。

- **针对医学影像微调的 Vision Transformer 模型**：一个针对乳腺癌图像分类进行微调的 **Vision Transformer 模型**已上传至 Hugging Face。模型详情和 Demo 可以在上传者的 [Hugging Face space](https://huggingface.co/spaces/emre570/google-vit-large-finetuned) 查看。

- **针对 Prompt 的 CFG 与 PAG 测试**：用户正在尝试 **CFG** (Classifier-Free Guidance) 和 **PAG** 的不同设置，以在保持图像质量的同时优化对 Prompt 的遵循度。建议从 CFG=4.5 和 PAG=5.0 开始并相应调整，Demo 将根据反馈更新以修正输出。

- **寻求 PII 检测建议**：一位用户请求关于 **PII**（个人身份信息）检测项目的模型或方法建议，并提到他们已经使用了文本挖掘模型和 **BERT**。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/emre570/google-vit-large-finetuned">emre570/google-vit-large-finetuned · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/emre570/google-vit-large-finetuned">Emre570 Test Model - emre570 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://ku-cvlab.github.io/Perturbed-Attention-Guidance/">Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance</a>: 未找到描述</li><li><a href="https://huggingface.co/hyoungwoncho/sd_perturbed_attention_guidance">hyoungwoncho/sd_perturbed_attention_guidance · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05">HyperGraph Datasets - SauravMaheshkar 收藏集</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1223275212687278230)** (3 条消息): 

- **下次见面会日期已确认**：一位成员询问了 **reading-group** 下次会议的日期。另一位成员提供了详细信息，并分享了该活动的 [Discord 邀请链接](https://discord.gg/hqrZjkjJaq?event=1222215283343622276)。

**提到的链接**：<a href="https://discord.gg/hqrZjkjJaq?event=1222215283343622276">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、闲逛，并与你的朋友和社区保持紧密联系。

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1222819278810779658)** (16 条消息🔥): 

- **Roboflow 受到关注**：一位参与者强调了用于计算机视觉任务的工具 [Roboflow](https://roboflow.com/) 日益增长的普及度，并提到了其他特定领域的工具，如用于医学图像的 3d Slicer 和 ITK-SNAP。对于那些希望微调模型的人，他们提醒说像 SAM 这样的应用可能并不理想。

- **SAM 微调问题**：一位用户在微调 SAM 模型时遇到了问题，遇到了与 'multimask_output' 参数相关的 `TypeError`。他们分享了一篇 [Towards AI 博客文章](https://pub.towardsai.net/fine-tune-meta-sam-19f7cd4331dd)，但未说明错误是否已解决。

- **分享 YOLO 转换脚本**：在关于将 YOLO 模型转换为 safetensors 的讨论中，一位成员分享了转换过程中使用的脚本，并指向了一个 [Hugging Face 讨论帖](https://huggingface.co/lmz/candle-yolo-v8/discussions/1) 以获取更多细节。该脚本包含用于重命名层以匹配 safetensors 预期格式的函数。

- **关于 Zero-Shot 图像分类器微调的咨询**：一位自称初学者的人寻求关于使用自定义数据集微调 zero-shot 图像分类器的建议。虽然他们不确定自己系统的能力，但确认使用的是 NVIDIA GeForce GTX 1650 GPU。

- **用于演示的 Pix2Pix 提示词测试**：一位用户正在寻找一种方法来测试 instruct pix2pix 的编辑提示词以进行演示，表示虽然他们在模型的 Space 中成功使用了提示词，但缺乏用于进一步测试的 `gradio_client` API。只要能为他们的演示流水线生成编辑后的图像，他们对替代方案持开放态度。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/lmz/candle-yolo-v8/discussions/1">lmz/candle-yolo-v8 · 能否提供转换为 safetensors 格式的脚本？</a>: 未找到描述</li><li><a href="https://github.com/bhpfelix/segment-anything-finetuner?tab=readme-ov-file">GitHub - bhpfelix/segment-anything-finetuner: Segment Anything 的简单微调入门代码</a>: Segment Anything 的简单微调入门代码 - bhpfelix/segment-anything-finetuner</li><li><a href="https://github.com/bhpfelix/segment-anything-finetuner?t">GitHub - bhpfelix/segment-anything-finetuner: Segment Anything 的简单微调入门代码</a>: Segment Anything 的简单微调入门代码 - bhpfelix/segment-anything-finetuner
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1222883104981651517)** (21 条消息🔥): 

- **对抗无限生成**：一位成员分享了他们自定义的 LLM（基于 decilm7b）持续生成文本而不停止的问题。其他成员建议研究 *Supervised Fine Tuning (SFT)*，并通过 [`repetition penalty`](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin) 调整生成行为，或在配置中使用 `StopCriterion`。

- **理解摘要错误**：有报告称 BART CNN 摘要模型在运行期间开始输出错误信息。该问题突显了模型依赖或 API 变更可能带来的潜在陷阱。

- **RAG 爱好者集结**：一位用户寻求关于构建检索增强生成 (RAG) 系统的建议，考虑使用 `faiss` 作为 vectorDB，并使用 `llama.cpp` 在有限的 GPU 资源下保证模型效能。建议引导该用户参考相关资源，包括一段解释 RAG 系统评估的视频以及 ollama 的简单安装流程。

- **寻求研究合作伙伴**：频道中出现了一次互动，一名成员联系另一名成员寻求潜在的研究项目合作，展示了 NLP 社区频道友好且合作的氛围。

- **通过 Tokenizer 发现辅助模型**：一名新手询问如何识别与主模型 Tokenizer 兼容的辅助模型（特别是针对 `model.generate` 函数），这证明了模型兼容性是 NLP 应用中的一个普遍关注点。

**提到的链接**：<a href="https://youtu.be/r0_O0IogbKo?si=lNon-ytkDjw9x1-3">评估检索增强生成 (RAG) 系统</a>：检索增强生成是一个强大的框架，可以提高从 LLM 获取的响应质量。但如果你想创建 RAG 系统...

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1222877278602526844)** (4 条消息): 

- **探索 Labmlai Diffusion**：讨论中有人询问关于 [labmlai 的 diffusion 仓库](https://github.com/labmlai/annotated_deep_learning_paper_implementations)的使用经验。
- **Marigold 项目扩展**：**Marigold 深度估计流水线**已贡献至 Hugging Face Diffusers，团队目前正致力于集成 [LCM 函数](https://huggingface.co/spaces/prs-eth/marigold-lcm)，并计划支持更多模态。
- **实时 img2img 的演进**：通过优化，使用 sdxl-turbo 可以让 img2img 在 **800x800 分辨率下以 30fps** 运行，从而实现迷人的**实时**视觉过渡。
- **检测到可能的 img2img Bug**：img2img 模型中存在一个“off-by-one bug”，被认为会导致图像向右偏移；目前已实验了一种通过每隔几帧处理图像边缘的变通方法。计划进一步调查 **conv2d padding** 以及抖动中可能存在的确定性模式。
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1222793504493801554)** (108 条消息🔥🔥): 

- **用于 AI 训练的配音演员数据**：讨论指出，由于反 AI 情绪，专业配音演员可能不愿向 11labs 等 AI 项目贡献其语音数据，但非专业语音可能足以满足自然语言训练的需求，尤其是在情感范围不是首要任务时。
- **基准测试受到质疑**：多条消息表明，现有的 AI 基准测试可能无法可靠地反映模型性能，有观点称目前的基准测试存在严重缺陷。由于引入了人工评估，[HuggingFace 上的 chatbot arena 基准测试](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)被认为是一个更合理的选择。
- **AI21 的新模型 Jamba 取得突破**：[AI21 Labs 发布了 Jamba](https://www.ai21.com/blog/announcing-jamba)，这是一款新型的最先进混合 SSM-Transformer LLM，结合了两种架构的优势，使其在基准测试性能上能够媲美或超越尖端模型。
- **关于 Mamba 与 Flash Attention 优势的疑问**：讨论了 Mamba（一种基于内存的架构）相较于使用 Flash Attention 的模型的优势，并对 Mamba 在实际应用中的性能和成本对比持怀疑态度。
- **关于 AI 生成音乐音频质量的辩论**：一些参与者批评了 Suno 等 AI 音乐生成器的音频质量，指出了噪声伪影等问题，并建议需要更好的 neural codecs。关于这些工具的最新版本是否比早期版本有所改进，存在不同意见。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys</a>: 未找到描述</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>: 首次推出生产级基于 Mamba 的模型，提供同类最佳的质量和性能。</li><li><a href="https://huggingface.co/Undi95/dbrx-base">Undi95/dbrx-base · Hugging Face</a>: 未找到描述</li><li><a href="https://arstechnica.com/security/2024/03/thousands-of-servers-hacked-in-ongoing-attack-targeting-ray-ai-framework/">Thousands of servers hacked in ongoing attack targeting Ray AI framework</a>: 研究人员表示，这是首个已知的针对 AI 工作负载的野外攻击。</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1/tree/main">ai21labs/Jamba-v0.1 at main</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=9K2tkWjOTDU&list=PLMzBiaqOoxQWBs7UpS2WW_Qf3zSuWOux_">mockingbird - Neural hacker</a>: = NEURAL HACKER =___________________________啊哈！哟，菜鸟，准备好进入我扭曲的小巷了吗？潜入我的数字混沌，逃离平庸。伴随着...</li><li><a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan/">GitHub - PKU-YuanGroup/Open-Sora-Plan: This project aim to reproduce Sora (Open AI T2V model), but we only have limited resource. We deeply wish the all open source community can contribute to this project.</a>: 该项目旨在复现 Sora (Open AI T2V 模型)，但我们的资源有限。我们深切希望整个开源社区能为该项目做出贡献。 - PKU-YuanGroup/Open-Sora-Plan
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1222823441481334804)** (31 条消息🔥): 

- **澄清关于 DiT 模型的误解**：根据讨论，一段关于 **DiT (Diffusion Transformers)** 的 [YouTube 视频](https://www.youtube.com/watch?v=OPqFpm3wksY) 未能充分解释该技术，并误导了 Attention 在扩散模型中的作用。一位成员澄清说，与视频暗示的不同，普通的扩散模型也利用了 Attention 机制。

- **寻求 Transformer 扩散模型的清晰解释**：在困惑中，有人呼吁澄清 Transformer 在扩散模型中的实现和功能。一位成员指出，由于缺乏非“科学术语”的分解，即使是 U-nets 和 Transformer 的基础知识对许多人来说也是不透明的。

- **从高层级理解 U-nets**：有人尝试从高层级解释 **U-nets**，将其描述为一种将图像编码到低维空间，然后进行上采样的结构。该解释强调了模型在编码过程中丢弃多余信息的能力，以简化随后的预测解码。

- **征求关于美学和偏好排名的资源**：一位成员宣布他们在使用 **RAIG** 进行实验，并产生了很有前景的图像输出，风格类似于 Midjourney，他们正在征集与美学排名和人类偏好选择相关的资源。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul><li><a href="https://www.youtube.com/watch?v=OPqFpm3wksY">DiT: The Secret Sauce of OpenAI&#39;s Sora &amp; Stable Diffusion 3</a>: 不要错过这些激动人心的升级，旨在通过 DomoAI 提升您的内容创作体验！快去试试：discord.gg/sPEqFUTn7nDiffusion Transf...
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1222961258731667616)** (5 条消息):

- **使用 Int8 和 Binary Embeddings 优化 RAG**：为了解决内存和成本挑战，[@Cohere](https://t.co/MCnmhQGS7g) 介绍了在 **RAG pipelines** 中使用 Int8 和 Binary Embeddings 的方法。在处理大型数据集时，这可以同时节省内存和资金。
- **LLMxLaw Hackathon 参与**：请记下这个有趣的活动，[@hexapode](https://twitter.com/hexapode) 和 [@yi_ding](https://twitter.com/yi_ding) 将于 4 月 7 日在斯坦福大学的 **LLMxLaw Hackathon** 上发表演讲。感兴趣的参与者可以[加入名单](https://t.co/7lXZBX5APy)查看地点，并加入将 LLMs 整合到法律领域的倡议。
- **增强 RAG 的数据表示**：查看 [@palsujit](https://t.co/OKe6goTiJp) 的博客文章，了解如何通过使用语义分块（semantic chunking）结合层级聚类（hierarchical clustering）和索引来改进 **RAG/LLM 数据表示**，从而获得更好的结果。
- **从零开始构建 Self-RAG**：探索如何构建一个具有内置反思（reflection）功能的动态 **检索增强生成 (RAG) 模型**。该模型由 Florian June 的博客文章指导，其特点是由特殊 token 触发的两步检索过程。完整的解释可以在分享的[博客文章](https://t.co/JeW2p294Bw)中找到。
- **使用 LlamaParse 增强 RAG 查询**：@seldo 在一段简短的视频中展示了由 LLM 驱动的 **LlamaParse** 如何将复杂的保险单转化为简单的查询，从而显著提高针对复杂文档的 **RAG 查询** 质量。观看[教学视频](https://t.co/jOvvarw1n6)。

**提到的链接**：<a href="https://t.co/7lXZBX5APy">RSVP to LLM x Law Hackathon @Stanford #3 | Partiful</a>：随着人工智能 (AI) 继续彻底改变全球各行各业，法律领域也不例外。LLMs 是一种能够理解和生成自然语言的基础模型...

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1222807131011420231)** (107 条消息🔥🔥): 

- **理解解析的复杂性**：一位用户询问了 **PDF 解析** 与 Word 文档相比的复杂性。另一位用户建议，虽然解析表格和图像可能具有挑战性，但像 **LlamaParse** 和 **Unstructured** 这样的库可能会很有用。

- **使用 Qdrant 可视化 Embeddings**：一位用户对如何使用 **Qdrant** 可视化 embeddings 感到好奇。另一位用户指出，Qdrant 有一个可通过 `http://localhost:6333/dashboard` 访问的 UI，但如果不在本地主机上，地址可能会有所不同。

- **GenAI 应用数据集成**：分享了来自 **Fivetran** 的一篇文章，展示了数据集成如何简化 GenAI 应用的构建并节省工程时间，并提供了[文章的直接链接](https://www.fivetran.com/blog/building-a-chatbot-with-fivetran-and-langchain)。

- **请求 LlamaIndex 与 Streamlit 的集成示例**：一位新用户询问了 LlamaIndex 与 Streamlit 的示例，一个有用的回复包括了[指向 Streamlit 相关博客文章的链接](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)。

- **RAG 聊天机器人检索准确性的挑战**：一位正在开发 RAG 聊天应用的用户遇到了正确源文档检索的问题。他们被引导去考虑元数据字段、前缀问题和元数据查询，并建议如果问题仍然存在，可以删除并重新创建索引。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/">使用 LlamaIndex 构建具有自定义数据源的聊天机器人</a>：只需 43 行代码即可使用您自己的数据增强任何 LLM！</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/indexing/">Indexing - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#text-splitters">Node Parser Modules - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/?h=#keywordnodepostprocessor">Node Postprocessor Modules - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/monsterapi#rag-approach-to-import-external-knowledge-into-llm-as-context>)">未找到标题</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/production_rag#key-techniques_1>).">为生产环境构建高性能 RAG 应用 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies#metadata-filters>).">基础策略 - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---

**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1223223563981684747)** (2 messages): 

- **探索 LLM 中的模型对齐**：一篇新的 [博客文章](https://blog.premai.io/model-alignment-process/) 探讨了针对 **Mistral 和 Zephyr 7B 模型** 的对齐方法，如 **RLHF**、**DPO** 和 **KTO**，强调了这些方法优于标准的 supervised fine-tuning。文章指出，此类对齐技术显著增强了语言生成任务的表现。
- **策划 LLM 研究的任务**：[Shure-dev 的使命](https://shure-dev.github.io/) 专注于提供精选的高质量 **LLM (Large Language Models)** 论文，旨在为紧跟该领域快速发展的研究人员提供关键资源。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://shure-dev.github.io/">Awesome-LLM-related-Papers-Comprehensive-Topics</a>: 全球最全面的 LLM 论文和仓库精选列表</li><li><a href="https://blog.premai.io/model-alignment-process/">Model Alignment Process</a>: 生成模型与人类反馈的对齐显著提升了自然语言生成任务的性能。对于大型语言模型 (LLMs)，像 reinfor... 对齐方法
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1222812198431096903)** (59 messages🔥🔥): 

- **寻找 01 Light 国际配送选项**：成员们讨论了预订 **01 light** 并配送至美国境外国家的可能性。通过美国朋友代购再转运至国际地址似乎是可行的，因为激活和使用该设备不需要美国公民身份。

- **在离线模式下运行 Open Interpreter**：用户探索了在离线模式下使用 Open Interpreter 以避免 API 费用，引发了对 Jan、Ollama 和 LM Studio 等本地 LLM 提供商的讨论。分享了使用本地模型运行的详细步骤，包括 *`interpreter --model local --api_base http://localhost:1234/v1 --api_key dummykey`*，并提供了 [文档链接](https://docs.openinterpreter.com/guides/running-locally) 以及在 Windows 上使用 LM Studio 的故障排除。

- **OpenInterpreter 项目的贡献邀请**：**[OpenInterpreter/aifs](https://github.com/OpenInterpreter/aifs)** GitHub 仓库公开征集贡献，重点是增强本地语义搜索。

- **工业设计中的 AI**：一位用户回忆了工业设计中语音激活 AI 伴侣的早期概念，引用了 Jerome Olivet 的 ALO 概念手机等例子，该手机具有全语音化的 UX。

- **在编程 IDE 中配合调试器使用 Open Interpreter**：讨论了使用 PyCharm 和 Visual Studio Code 调试 01 项目，强调了查看 Open Interpreter 与本地模型服务器（如 LM Studio）之间对话的潜力。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.openinterpreter.com/guides/running-locally">Running Locally - Open Interpreter</a>: 未找到描述</li><li><a href="https://tenor.com/view/bane-no-banned-and-you-are-explode-gif-16047504">Bane No GIF - Bane No Banned - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/OpenInterpreter/aifs">GitHub - OpenInterpreter/aifs: Local semantic search. Stupidly simple.</a>: 本地语义搜索。极其简单。通过在 GitHub 上创建账号为 OpenInterpreter/aifs 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1222798061978255362)** (54 messages🔥):

- **开源 LLM 缺乏 OS 控制能力**：讨论显示，开源 LLM 尚未针对控制 Windows 或 macOS 等操作系统进行预微调。一位用户表示有兴趣使用合成数据对 Mistral 进行微调以实现此目的。
- **O1 Light 的国际运输**：一位用户询问了预订 O1 light 并通过美国朋友进行国际邮寄的事宜。经确认，自行构建的 O1 可以在任何地方运行。
- **01 硬件与 Arduino 的兼容性**：用户讨论了将 01 Light 与 Arduino 集成的技术可能性。虽然默认使用 ESP32，但在等待期间，人们有兴趣将其与其他开发板（如 Elegoo）配合使用。
- **01 软件与 Vision 能力**：对话涉及了 01 软件处理视觉任务的能力，提到当添加图像作为消息时，它可以自动切换到 GPT-4-vision-preview。
- **01 开发环境的 Windows 安装**：用户报告了在 Windows 上安装 01 OS 的困难，一名成员正在积极努力解决这些问题并计划分享进展。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://app.suno.ai/song/cbf3c6a9-dc4e-4663-b7c5-b7f1aebe687a/">Electric Echoes | Suno</a>：另类蒸汽波斯卡歌曲。听听看，并用 Suno 制作你自己的歌曲。</li><li><a href="https://app.suno.ai/song/6fb4e5a3-fa8e-4a7b-8362-3c5b355d9936">01100001 01101101 00100000 01001001 0010 | Suno</a>：8bit, chiptune, speedup, Arpeggio, fatbass, Hardbass, FemaleVocals, synthesizer，Electronic，speedup 歌曲。听听看，并用 Suno 制作你自己的歌曲。</li><li><a href="https://github.com/OpenInterpreter/01/pull/192">[WIP] Fix setup and running on Windows by dheavy · Pull Request #192 · OpenInterpreter/01</a>：尝试通过添加缺失部分和修复 Windows 特定问题来弥补差距，并为 Windows 用户提供入门便利。更多细节即将公布。
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1222876738049146911)** (8 条消息🔥): 

- **辩论编译器技术的优劣**：尽管没有编译器方面的经验，一位用户强调了使用 PyTorch/Triton 等 **编译器技术** 生成高效 GPU 代码相比手动编写 CUDA 的潜在优势。
- **寻找 Benchmark 建议**：一位用户征求 Benchmark 建议，以测试他们即将获得的尖端硬件，并表示目前关注点在 *flash attention* 和 *triton 示例* 上。
- **分享 CUDA 知识资源**：分享了一个系列讲座链接，用于理解 **编译器** 如何帮助优化内存带宽受限的内核（memory bandwidth-bound kernels），但对计算受限的内核（compute-bound kernels）帮助较小。
- **讨论分布式数据并行 (DDP)**：一条消息提到了涉及 DDP 的结果，暗示禁用点对点 (p2p) 可能会特别影响 FSDP (Fully Sharded Data Parallel) 训练设置。
- **CUDA 开发 IDE 偏好**：有人提出了关于 CUDA 开发首选 IDE 的问题，并提到了对 **VSCode** 的个人偏好以及对 **CLion** 的一些尝试。
- **CUDA 编程小组公告**：分享了关于 Cohere 的 **CUDA 课程** 链接，并宣布了一个名为 **Beginners in Research-Driven Studies (BIRDS)** 的社区主导小组，定于 4 月 5 日开始。[Cohere CUDA 课程公告](https://x.com/CohereForAI/status/1773419415406809432)。

**提及的链接**：<a href="https://x.com/CohereForAI/status/1773419415406809432">来自 Cohere For AI (@CohereForAI) 的推文</a>：我们由社区主导的 Beginners in Research-Driven Studies (BIRDS) 小组即将启动第一个专注于 CUDA 编程初学者的迷你学习小组，将于 4 月 5 日星期五开始 🎉

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1222844463232061541)** (9 条消息🔥): 

- **CUDA 课程查询**：一位用户询问了 **CUDA 在线课程**，并收到建议去查看 [GitHub - cuda-mode/resource-stream](https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#cuda-courses) 上的 CUDA 相关新闻和材料。
- **经典的初学者 CUDA 课程**：另一个课程建议是 **Udacity 的 Intro to Parallel Programming**，该课程在 [YouTube](https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2) 上也有播放列表。
- **关于在 MacBook 上设置 CUDA 的咨询**：一位用户寻求在旧款 MacBook（Big Sur，2015 年初）上设置 **CUDA 与 C++** 的帮助。
- **Mac 上运行 CUDA 的潜在解决方案**：给出的建议是运行一个 **带有 Linux 的虚拟机 (VM)**，以便在 MacBook 上使用 CUDA。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://a.co/d/0MImxSS">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#cuda-courses">GitHub - cuda-mode/resource-stream: CUDA 相关新闻和资料链接</a>: CUDA 相关新闻和资料链接。通过在 GitHub 上创建账户，为 cuda-mode/resource-stream 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2">课程介绍 - 并行编程入门</a>: 此视频是在线课程《并行编程入门》的一部分。在此处查看课程：https://www.udacity.com/course/cs344。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1222965591594369145)** (8 条消息🔥): 

- **轻松上手 CUDA，无需繁琐操作**: 一位聊天成员表示希望有一份在 Windows 游戏机上使用 **3090 GPU** 而无需安装 Ubuntu 的指南。他们正在考虑设置双系统是否是一个更简单的解决方案。
- **WSL 前来救场**: 另一位成员建议使用 **Windows Subsystem for Linux (WSL)** 来运行 CUDA/Torch，因为它的效果很好，且无需更换 Windows 系统。他们提供了一份[在 Windows 上通过 WSL 安装 Linux 的指南](https://learn.microsoft.com/en-us/windows/wsl/install)。
- **WSL 版本至关重要**: 明确指出，在为 CUDA 使用 WSL 时，必须确保系统设置为 **WSL2**，因为 WSL1 不足以胜任该任务。

**提到的链接**: <a href="https://learn.microsoft.com/en-us/windows/wsl/install">安装 WSL</a>: 使用命令 `wsl --install` 安装 Windows Subsystem for Linux。在 Windows 机器上使用由你首选的 Linux 发行版（Ubuntu, Debian, SUSE, Kali, Fedora, Pengwin...）运行的 Bash 终端。

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1222918447067959406)** (57 条消息🔥🔥): 

- **由于范围原因停止微调实验**: 使用 [FSDP+QLoRA](https://github.com/AnswerDotAI/fsdp_qlora) 在 `meta-llama/Llama-2-13b-chat-hf` 上进行的微调被停止，因为它与预期的 ring-attention 相关实验不符。该仓库确实包含 Flash-Attention-2，但将 Ring-Attention 整合进去尚不确定。

- **解决 Ring-Attention 的 Loss 问题**:
  - 对使用 CrossEntropyLoss 的 `LlamaForCausalLM` 的 `modeling_llama.py` 中的 Loss 问题进行了调试。
  - 该问题被追溯到多 GPU 环境中标签的广播处理错误，随后已修复。

- **长上下文训练探索**: 严格的测试展示了在拥有 48GB VRAM 的 A40 GPU 上，成功训练了上下文长度从 32k 到 100k 的 tinyllama 模型。观察到 llama-2 7B 可以在 2x A100 上运行，在 4k 序列长度下占用 54GB。

- **讨论 Llama 7B 训练的 VRAM 需求**: 分享了一个 [Reddit 链接](https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/helpful_vram_requirement_table_for_qlora_lora_and/)，讨论了使用 QLoRA 和 LoRA 训练 Llama 7B 等模型的 VRAM 需求，强调了大型模型训练需要充足的 VRAM。 

- **长上下文模型的数据集搜索**: 成员们征集了长上下文数据集，建议使用 Hugging Face 的 Long-Data-Collections 中的 `booksum` 数据集等资源，用于潜在地微调适合长文本输入的模型。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/helpful_vram_requirement_table_for_qlora_lora_and/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.together.ai/blog/llama-2-7b-32k">为 32K 上下文时代做准备：早期学习与探索</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/togethercomputer/Long-Data-Collections">togethercomputer/Long-Data-Collections · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/AnswerDotAI/fsdp_qlora">GitHub - AnswerDotAI/fsdp_qlora: 使用 QLoRA + FSDP 训练 LLM</a>: 使用 QLoRA + FSDP 训练 LLM。通过在 GitHub 上创建账户，为 AnswerDotAI/fsdp_qlora 的开发做出贡献。</li><li><a href="https://wandb.ai/cataluna84/fsdp_qlora/runs/o59wbxpr/workspace?nw=nwusercataluna84">cataluna84</a>: Weights & Biases，机器学习开发者工具
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1222990325698789376)** (5 条消息):

- **在知乎上寻找 Triton 教程**：一位成员寻求中文帮助以登录 [知乎](https://www.zhihu.com/signin?next=%2F)，该平台拥有优秀的 Triton 教程。该成员成功在 iOS 上创建了账号，但需要帮助寻找 App 的扫码功能进行登录验证。

- **找到扫码按钮**：另一位成员提供了在知乎 App 中定位“扫码”按钮的解决方案，从而实现了成功登录。

- **丰富的 Triton 内容**：成功登录后，该成员确认在知乎上发现了大量 Triton 相关内容。

- **希望有中文搜索术语表**：有人提到由于语言障碍导致搜索困难，并指出一份中文搜索术语表将会很有帮助。

**提到的链接**：<a href="https://www.zhihu.com/signin?next=%2F">no title found</a>: no description found

  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1222828137361051720)** (24 条消息🔥): 

- **识别出 Triton-Viz 的同步问题**：一位贡献者发现了由 `triton-viz` 更新引起的错误，特别是由于[最近的一个 Pull Request](https://github.com/Deep-Learning-Profiling-Tools/triton-viz/pull/19/files#diff-617f71ef3c8b3147084e47a1492611a7f42bd28b720fdc57b7ff5111663ec298L21) 中的更改，导致与 notebook 中安装的旧版本 Triton 产生兼容性问题。
- **Triton-Viz 更新的临时解决方案**：成员们分享了通过使用命令 `pip install git+https://github.com/Deep-Learning-Profiling-Tools/triton-viz@fb92a98952a1e8c0e6b18d19423471dcf76f4b36` 安装旧版本 `triton-viz` 来解决错误的方案，对某些人来说该方案有效。
- **持续的安装错误**：尽管应用了建议的修复并重启了 runtime，一些用户仍然遇到 `triton-viz` 的导入错误。建议重启 runtime 并重新执行安装命令。
- **官方修复和安装流程**：Srush1301（推测为维护者）提供了一个官方修复方案，其中包含适用于最新 Triton 的详细安装流程。正如一位提到运行正常的成员所确认的，这应该能解决讨论中的问题。
- **关于在 Windows 上使用 Triton 的说明**：一位成员询问 Triton 是否可以在 Windows 上使用，另一位成员回答说 Triton 本身不兼容 Windows，但基于 Triton 的 puzzles 可以在 Colab 或其他非 GPU 主机上运行。

**提到的链接**：<a href="https://github.com/Deep-Learning-Profiling-Tools/triton-viz/pull/19/files#diff-617f71ef3c8b3147084e47a1492611a7f42bd28b720fdc57b7ff5111663ec298L21.">[TRITON] Sync with triton upstream by Jokeren · Pull Request #19 · Deep-Learning-Profiling-Tools/triton-viz</a>: no description found

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1223140446587457618)** (1 条消息): 

- **洞察模型受欢迎程度**：OpenRouter 推出了**模型的 App 排名**，显示在模型页面新的 **Apps** 选项卡中。该选项卡根据 App 对特定模型的使用情况进行排名，展示了顶级的公开 App 和处理的 Token 数量，例如 [Claude 3 Opus](https://openrouter.ai/models/anthropic/claude-3-opus?tab=apps) 页面所示。

- **Claude 3 Opus 基准测试**：[Claude 3 Opus](https://www.anthropic.com/news/claude-3-family) 被誉为 Anthropic 最强大的模型，在具有高水平智能和流畅性的复杂任务中表现出色，基准测试结果可在发布公告中查看。

- **社区项目亮点**：值得注意的社区项目包括一个 Discord 机器人 [Sora](https://github.com/mintsuku/sora)，它利用 Open Router API 增强服务器对话；以及一个平台 [nonfinito.xyz](https://nonfinito.xyz/)，允许用户创建和分享模型评估。

- **新 API 和改进的客户端**：推出了一个更简单的 `/api/v1/completions` API，它镜像了带有 prompt 参数的 Chat API 功能，同时改进了 OpenAI API 客户端支持。此外，由于过度的速率限制，OpenRouter 已停止为 Nitro 模型使用 Groq。

- **优化的加密货币支付**：OpenRouter 正在通过降低 Gas 费用来改进加密货币支付，利用 Ethereum L2 的 Base 链，使交易对用户来说更实惠。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/anthropic/claude-3-opus?tab=apps">Claude 3 Opus by anthropic | OpenRouter</a>: Claude 3 Opus 是 Anthropic 用于高度复杂任务的最强大模型。它拥有顶级的性能、智能、流畅度和理解力。查看发布公告和基准测试...</li><li><a href="https://github.com/mintsuku/sora">GitHub - mintsuku/sora: Sora 是一个集成了 Open Router API 的 Discord 机器人，旨在促进 Discord 服务器中的对话。</a>: Sora 是一个集成了 Open Router API 的 Discord 机器人，旨在促进 Discord 服务器中的对话。- mintsuku/sora</li><li><a href="https://nonfinito.xyz/">Evaluations - Non finito</a>: 未找到描述
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1222809408849973308)** (107 messages🔥🔥): 

- **图像支持说明**：用户讨论了是否支持图像输入，并明确指出对于接受图像输入的模型，“直接上传即可”。
- **OpenRouter 中的 Claude 无 Prefill**：关于 **OpenRouter 中的 Claude** 是否支持 Prefill 功能存在困惑，但已澄清 **Sillytavern** 中没有 Prefill，必须创建一个绝对深度为 0 的 Prompt，其作用类似于 Prefill。
- **对 Databricks DBRX 和 Gemini Pro 的兴奋**：Databricks 新模型 **DBRX** 的发布引发了关注，大家对 **Gemini Pro 1.5** 的性能表达了积极看法，尽管有 **error 429**（速率限制）的报告，并建议在 UTC 00:00 后尝试。
- **模型可用性和错误代码的挑战**：多位用户报告了模型宕机或遇到错误代码的问题，例如 **Claude 2.1 的 error 502** 以及与 Cloudflare CDN 超时相关的 **error 524**。聊天中讨论了潜在的修复方案以及设置故障转移（Failover）模型的可能性。
- **关于自我审核 ClaudeAI 的讨论及价格对比**：讨论了通过 **OpenRouter** 使用 **ClaudeAI** 的好处，特别是对于角色扮演和敏感内容，误报检测更少。**OpenRouter API** 标准化了访问，并确认定价为成本价，与直接使用官方 **ClaudeAI API** 相同。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://olympia.chat">Olympia | Better Than ChatGPT</a>: 通过价格合理的 AI 驱动顾问来发展您的业务，这些顾问是业务战略、内容开发、营销、编程、法律战略等方面的专家。</li><li><a href="https://github.com/OlympiaAI/open_router">GitHub - OlympiaAI/open_router: Ruby library for OpenRouter API</a>: 用于 OpenRouter API 的 Ruby 库。通过在 GitHub 上创建一个账户来为 OlympiaAI/open_router 做出贡献。</li><li><a href="https://openrouter.ai/models?o=pricing-low-to-high">OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://openrouter.ai/docs#limits">OpenRouter</a>: 构建与模型无关的 AI 应用</li><li><a href="https://openrouter.ai/docs#model-routing">OpenRouter</a>: 构建与模型无关的 AI 应用</li><li><a href="https://openrouter.ai/models/anthropic/claude-3-opus:beta">Claude 3 Opus by anthropic | OpenRouter</a>: 这是与 Anthropic 合作提供的 [Claude 3 Opus](/models/anthropic/claude-3-opus) 的低延迟版本，具有自我审核功能：响应审核发生在模型上...
</li>
</ul>

</div>
  

---



**AI21 Labs (Jamba) ▷ #[announcements](https://discord.com/channels/874538902696914944/874538945168408606/1222913979249328190)** (1 messages): 

- **AI21 Labs 发布 Jamba**：AI21 Labs 隆重宣布推出 [Jamba](https://www.ai21.com/jamba)，这是一款采用融合了 Mamba 和 Transformer 元素的新型架构的**尖端模型**。它是 AI21 的首个开源模型，具有 256K 上下文窗口等关键特性，并能在单个 GPU 上容纳高达 140K 的上下文。
  
- **Jamba 提升长上下文性能**：Jamba 在问答和摘要等长上下文用例中提供了**前所未有的 3 倍吞吐量**，解决了关键的企业挑战，并树立了 GenAI 效率的新标准。

- **开放获取下一代 AI**：这款突破性的模型现已根据 **Apache 2.0** 许可证开放权重，用户可以在 [Hugging Face](https://huggingface.co/ai21labs/Jamba-v0.1) 上访问，并即将加入 NVIDIA API 目录。

- **利用 NVIDIA NIM 驾驭 GenAI**：消息详细介绍了通过 NVIDIA NIM 在任何地方运行的生产级 API，涉及*模型*、*集成*、*随处运行*、*如何购买*、*用例*、*生态系统*、*资源*和*文档*。然而，“Harness Generative”之后的部分渲染不完整，因此无法提供具体细节。

- **关于 Jamba 的行业热议**：一篇 [TechCrunch 独家报道](https://techcrunch.com/2024/03/28/ai21-labs-new-text-generating-ai-model-is-more-efficient-than-most/) 讨论了向具有更长上下文的生成式 AI 模型发展的趋势，并强调 AI21 Labs 的 Jamba 模型通过提供计算密集度较低的大上下文窗口，与 OpenAI 的 ChatGPT 和 Google 的 Gemini 等模型展开竞争。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://ai.nvidia.com/)">随处运行的生产级 API</a>：立即体验领先模型，构建企业级生成式 AI 应用。</li><li><a href="https://techcrunch.com/2024/03/28/ai21-labs-new-text-generating-ai-model-is-more-efficient-than-most/)">AI21 Labs 的新 AI 模型比大多数模型能处理更多上下文 | TechCrunch</a>：AI 行业正日益转向具有更长上下文的生成式 AI 模型。但具有大上下文窗口的模型往往……
</li>
</ul>

</div>
  

---


**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1222940200183795842)** (40 条消息🔥): 

- **Jamba 即将登陆 AI21 Labs 平台**：AI21 Labs 工作人员确认 Jamba 目前可通过 **Hugging Face** 获取，其对齐版本（aligned version）很快将通过其 **SaaS 平台** 提供。
- **多语言能力表现出色**：用户已向 AI21 工作人员确认，Jamba 已在多种语言上进行过训练，包括**德语**、**西班牙语**和**葡萄牙语**，尽管韩语表现可能没那么强。
- **期待 Jamba 的技术深度解析**：AI21 计划**发布一份白皮书**，详细介绍 Jamba 的训练细节，包括 Token 数量、语言和超参数。
- **暂无为 Jamba 计划 Hugging Face Space**：一名 AI21 工作人员表示，目前没有计划建立可以测试 Jamba 的 **Hugging Face Space**。
- **开源 Jamba 引起轰动**：Jamba 模型的开源在用户中引起了极大兴奋，引发了关于与其他模型（如 Mamba）的对比、推出更小版本的可能性、批处理推理引擎以及在代码任务上的表现等问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Jamba_(language_model)">Jamba (语言模型) - 维基百科</a>：未找到描述</li><li><a href="https://www.ai21.com/blog/announcing-jamba">介绍 Jamba：AI21 开创性的 SSM-Transformer 模型</a>：首次推出基于 Mamba 的生产级模型，提供一流的质量和性能。
</li>
</ul>

</div>
  

---


**AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1222915429341855744)** (56 条消息🔥🔥): 

- **最低定价政策取消**：AI21 Labs 已取消其模型的 **$29/月** 最低收费。不过，定价页面的工具提示仍显示最低费用，导致了短暂的困惑。
- **微调功能指日可待**：AI21 Labs 正在考虑重新引入微调功能，但目前**尚未确定**回归的具体日期。
- **探索 Jamba 的效率**：针对 **Jamba** 如何在拥有随序列长度平方缩放的 Transformer 块的情况下实现高吞吐量进行了讨论。**Mamba** 和 **Mixture-of-Experts (MoE)** 层贡献了不同的亚二次（sub-quadratic）缩放行为，使 Jamba 运行更快。
- **量化说明**：成员们讨论了 **Jamba** 模型是否可以进行量化以减少 GPU 显存占用。分享了 Hugging Face 量化指南的链接，并评论了即使使用中端硬件也能创建量化模型。
- **架构混淆澄清**：有成员澄清说 **MoE** 由 *Mamba* 块组成，而非 Transformer，并且在 Jamba 内部，每个 Jamba 块中 Transformer 与其他七种类型层的比例是 1:7。

**提到的链接**：<a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://huggingface.co/docs/accelerate/en/usage_guides/quantization&ved=2ahUKEwjQ-4eopJmFAxWaXmwGHZPhBN8QFnoECBMQAQ&usg=AOvVaw2RxBEXoJMjtWqDScwaFZqc">未找到标题</a>：未找到描述

  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1222829670811172865)** (56 条消息🔥🔥):

- **Conda 编译器引发的发狂**：一位成员在尝试调试 **Conda metal 编译器 bug** 时表达了挫败感，并提到这段经历让他抓狂到需要停下来制作一些 **memes**（梗图）来缓解压力。
- **Tinygrad 与虚拟内存**：讨论了虚拟内存在 **tinygrad** 软件栈中的重要性，一位成员正在考虑是否在其 cache/MMU 项目中支持虚拟内存。
- **通过 Runpod 访问 AMD**：提到 AMD 现在出现在 Runpod 上，但用户需要预约会议才能获得访问权限，讨论中带有一丝讽刺，暗示 AMD 将客户投诉视为 PR（公关）问题而非技术问题。
- **英特尔相较于 AMD 更优的客户服务**：成员们对比了英特尔的积极帮助和 GPU 更高的性价比，以及 AMD 充满 bug 的软件，同时讨论了市场供应的波动和地区定价差异。
- **寻求 Intel Arc 的优化**：一位成员描述了 IPEX 库中针对 Intel Arc 的 **transformers/dot product attention** 实现效率低下的问题，并根据他们提升 **Stable Diffusion** 性能的经验，建议通过简单的优化即可获得显著的性能提升。
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1222844163901362206)** (39 messages🔥): 

- **FPGA 加速咨询**：一位成员询问关于将 [tinygrad 移植到自定义硬件加速器](https://github.com/tinygrad/tinygrad/blob/master/docs/adding_new_accelerators.md)（如 FPGA）的问题，并被引导至 tinygrad 关于添加新加速器的文档。

- **深入探讨维度合并**：[探索了 tinygrad 中的维度合并 (Merging dimensions)](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/mergedim.md)，并需要进一步了解多视图交互（multiple views interaction），特别是为什么某些操作会创建多个视图。

- **探索原地操作及其错误**：一位成员质疑 tinygrad 在评估过程中可能存在的原地修改（in-place modifications），并提供了他们遇到的 "AssertionError" 作为支持，同时讨论了 Python 中 `+=` 的使用。

- **张量视图简化工作**：在简化 tinygrad 内部张量视图方面有大量参与，引发了关于保留多视图的必要性和准确性、符号表示（symbolic representations）以及针对此问题的 [提议更改](https://github.com/tinygrad/tinygrad/pull/39888) 的讨论。

- **将 std+mean 合并为一个 Kernel**：提到了将标准差（std）和平均值（mean）计算简化为单个 **Kernel** 的努力，并提供了一份关于 [Kernel 融合 (Kernel fusion)](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/scheduleitem.md) 的说明文档，以协助解决相关的悬赏挑战。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1nEN9q_PK8SHqrRcBIC6LnrJQE9FVvJE6?usp=sharing">Google Colaboratory</a>: 无描述</li><li><a href="https://github.com/tinygrad/tinygrad/pull/3988.">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/docs/adding_new_accelerators.md">tinygrad/docs/adding_new_accelerators.md at master · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1222807309600952381)** (93 messages🔥🔥): 

- **OpenAI 预付费模式见解**：通过一场活跃的讨论，揭示了 OpenAI 预付费模式的解释，其中 [LoganK 的回复](https://x.com/officiallogank/status/1760046748569841719?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 强调了欺诈预防和通往更高 **rate limits**（速率限制）的路径是主要驱动因素。关于未消费额度导致定价不透明的担忧，被拿来与 [星巴克从未使用余额中获得的收入](https://arstechnica.com/tech-policy/2024/01/hard-to-spend-card-balances-net-starbucks-200m-per-year-says-consumer-group/) 进行类比。

- **AI21 发布 Transformer Mamba 混合模型 "Jamba"**：AI21 宣布了一个名为 "Jamba" 的新模型，它结合了 **Structured State Space** 和 **Transformer** 架构。该模型已在 [Hugging Face](https://huggingface.co/ai21labs/Jamba-v0.1) 上发布，并因其显著的 **MoE** 结构而受到讨论，尽管有人指出在发布时还没有简单的方法来测试该模型。

- **Grok-1.5 模型令人印象深刻的推理能力**：xAI 推出了 Grok-1.5，与其前代产品相比，性能有了显著提升，特别是在编程和数学相关任务方面，在知名基准测试中取得了显著成绩。用户热切期待在 xAI 平台上使用 Grok-1.5，[官方博客文章](https://x.ai/blog/grok-1.5)中的讨论强调了 Grok 的各项能力。

- **关于 1-bit (Ternary) LLMs 的讨论**：一场关于将三值（-1, 0, 1）模型称为 “1-bit LLMs” 准确性的辩论，引发了围绕 BitNet 的影响以及此类模型训练方法的讨论。对话中引用了 BitNet 论文，并澄清这些模型是从零开始训练的，而非量化（quantized）而来的。

- **AI 领导层的重大变动**：Emad Mostaque 从 Stability AI 离职引发了对其原因和影响的讨论，[Peter Diamandis 的 YouTube 频道](https://youtu.be/e1UgzSTicuY?si=rF7LX1X6Kt7N2YRa)上的采访以及 [archive.is 上的详细背景故事](https://archive.is/8QkSl)对此进行了阐述。领导层的变动凸显了生成式 AI 领域新兴的动态和潜在的不稳定性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.ai/blog/grok-1.5">发布 Grok-1.5</a>：未找到描述</li><li><a href="https://www.ai21.com/jamba">介绍 Jamba</a>：一个开创性的 SSM-Transformer 开源模型</li><li><a href="https://www.modular.com/blog/the-next-big-step-in-mojo-open-source">Modular：Mojo🔥 开源的下一个重大步骤</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo🔥 开源的下一个重大步骤</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/yampeleg/status/1773401745269379409?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Yam Peleg (@Yampeleg) 的推文</a>：性能：- 与 Transformer 相比 3x Throughput。- 单个 GPU 即可容纳 140K (!) - 通常支持 256K context。</li><li><a href="https://x.com/dwarkesh_sp/status/1773381318266786180?s=46&t=90xQ8sGy63D2OtiaoGJuww">Dwarkesh Patel (@dwarkesh_sp) 的推文</a>：和我的朋友 @TrentonBricken 和 @_sholtodouglas 聊得很开心。无法简单总结，除了：这是关于 LLM 如何训练、它们具备什么能力的最棒的背景知识分享...</li><li><a href="https://x.com/officiallogank/status/1760046748569841719?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：@atroyn @OpenAIDevs 这主要由两个因素驱动：- 想要防止欺诈并确保 tokens 是由真实的人使用的 - 想要给开发者提供一条更清晰的通往更高 rate limits 的路径（通过允许...</li><li><a href="https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices">应对合成语音的挑战与机遇</a>：我们正在分享来自 Voice Engine 小规模预览的经验，这是一个用于创建自定义语音的模型。</li><li><a href="https://x.com/clementdelangue/status/1771395468959813922?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">clem 🤗 (@ClementDelangue) 的推文</a>：我们应该收购 Stability 并开源 SD3 吗？</li><li><a href="https://www.youtube.com/watch?v=57LqvutrOI8">Perplexity CEO：用 AI 颠覆 Google 搜索</a>：最杰出的 AI 创始人之一 Aravind Srinivas（Perplexity CEO）认为，未来我们可能会看到 100 多家估值超过 100 亿美元的 AI 初创公司。在本集中...</li><li><a href="https://x.com/GoogleDeepMind/status/1773422123869974878?s=20">Google DeepMind (@GoogleDeepMind) 的推文</a>：恭喜我们的 CEO 兼联合创始人 @demishassabis，他因对 Artificial Intelligence 的贡献被国王陛下授予爵士头衔。↘️ 引用 Demis Hassabis (@demishassabis) 的话：非常高兴...</li><li><a href="https://arstechnica.com/tech-policy/2024/01/hard-to-spend-card-balances-net-starbucks-200m-per-year-says-consumer-group/">消费者团体希望终结星巴克等公司 2.55 亿美元的“礼品卡漏洞”</a>：华盛顿州礼品卡法律的变更可能会影响全国的持卡人。</li><li><a href="https://x.com/anissagardizy8/status/1773759144425930962?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Anissa Gardizy (@anissagardizy8) 的推文</a>：突发：Microsoft 和 OpenAI 正在制定一项耗资 1000 亿美元的 AI 超级计算机计划。这台代号为“Stargate”的超级计算机将包含 *数百万* 个 GPUs，并需要数千兆瓦的电力...</li><li><a href="https://qwenlm.github.io/blog/qwen-moe/">Qwen1.5-MoE：以 1/3 的激活参数达到 7B 模型性能</a>：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 介绍 自从 Mixtral 引发关注热潮以来，关于 mixture-of-expert (MoE) 模型的研究势头强劲。研究人员和...</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B">Qwen/Qwen1.5-MoE-A2.7B · Hugging Face</a>：未找到描述</li><li><a href="https://youtu.be/e1UgzSTicuY?si=rF7LX1X6Kt7N2YRa">为什么我立即离开我的公司 (Stability AI) | Emad Mostaque 对谈第 93 集</a>：在本集中，Peter 和 Emad 讨论了 Emad 卸任 Stability AI CEO 的原因、他进入去中心化 AI 的下一步计划，以及为什么这件事如此紧迫...</li><li><a href="https://www.neelnanda.io/mechanistic-interpretability/glossary">全面的 Mechanistic Interpretability 解释与术语表 —— Neel Nanda</a>：未找到描述</li><li><a href="https://youtu.be/KV5gbOmHbjU?si=a0r05EWMctXB_l7v">《Transformer Circuits 的数学框架》详解</a>：《Transformer Circuits 的数学框架》详解 —— 我通读了这篇论文，并分享了一些想法、见解以及对...的澄清</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/cqJMkfO2xm">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.securityfrontiers.ai/">Security Frontiers | 当 GenAI 遇上安全创新</a>：向行业专家和 GenAI 创新者学习如何集成 Generative AI 以优化安全运营、自动化任务并加强网络防御。</li><li><a href="h

ttps://transformer-circuits.pub/2021/framework/index.html">A Mathematical Framework for Transformer Circuits</a>: 未找到描述</li><li><a href="https://archive.is/8QkSl">Inside Stability AI's bad breakup with Coatue and Lightspeed Venture &#x2026;</a>: 未找到描述
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1222954808143511592)** (45 条消息🔥): 

- **Moe Moe 还是 Noe Noe?**: 成员们对 **Qwen MOE** 性能的初步印象存在分歧，引发了简短讨论。一位成员最初非常热情，但随后将其描述为“有点烂”。
- **Jamba 的细节揭晓**: **AI21 Labs** 发布了他们的新模型 **Jamba**，其特点是拥有 **256k token 上下文窗口** 和 **12b 激活参数的 MoE 架构**。发布内容包括一篇 [博客文章](https://www.ai21.com/blog/announcing-jamba)，其训练时间似乎非常短，知识截止日期为 2024 年 3 月 5 日。
- **高性能带来的高成本**: 有发言指出 **基于 GB200 的服务器价格** 在 **200 万至 300 万美元之间**，由于昂贵的价格，引发了关于替代硬件选择的讨论。
- **GPT4 的野兽模式**: 与 **GPT-4 32k 版本** 进行了对比，该版本被公认为重大升级，可能会改变 AI 应用中的性能动态。
- **用于 Ring-Attention 的数据集**: 为了寻找合适的 **长上下文数据集**，一位成员分享了来自 Hugging Face 的集合，另一位成员提供了另一个链接：[Hugging Face 上的 MLDR 数据集](https://huggingface.co/datasets/Shitao/MLDR)，两者都针对长序列训练需求。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/Shitao/MLDR">Shitao/MLDR · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/togethercomputer/Long-Data-Collections">togethercomputer/Long-Data-Collections · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1222943820828905515)** (16 条消息🔥): 

- **寻求 LISA 和 Jamba 的帮助**: 一位成员正在寻求集成 **LISA** 并测试 **Jamba** (mamba moe) 的帮助，以查看其是否能开箱即用。

- **TorchTune 优化方案揭晓**: 在与 **torchTune** 团队交流后，一位参与者分享道，**在包括优化器在内的所有环节使用 bf16**，可以显著降低内存占用，且在稳定性方面与 **fp32** 或混合精度训练相当。

- **bf16 优化的实际应用**: 该团队成功将 **bf16** 应用于 SGD，并计划很快增加对 Adam 优化器的支持。这种方法因其显著的内存效率而受到关注。

- **PagedAdamW 对比 8bit Adam**: 一位成员分享了测试 **PagedAdamW** 的结果，与 8bit Adam 相比，它可以节省更多内存。差异非常显著，将峰值内存占用减少了近一半。

- **训练控制差异影响内存节省**: 另一位成员指出 **axolotl** 提供的训练控制级别与 **torchTune** 不同，这可能会影响内存节省的结果。对话表明，内存节省归功于 **PagedAdamW** 和优化反向传播（backward pass）的结合。

**提到的链接**: <a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full_single_device_low_memory.yaml">torchtune/recipes/configs/llama2/7B_full_single_device_low_memory.yaml at main · pytorch/torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。可以通过在 GitHub 上创建账号为 pytorch/torchtune 做出贡献。

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1222887793328656415)** (5 条消息): 

_

- **微调知识探索**：一位成员询问了学习如何微调/训练开源模型的最佳资源，表达了在利用 **axolotl** 执行任务之前理解基础知识的愿望。
- **选择合适的文本分类模型**：另一位成员询问在只有 T4 GPU 且需要支持英文的情况下，对于约 1000 个数据点的文本分类任务，哪种基础模型是微调的最佳选择。该成员询问考虑到 GPU 限制，使用 **Lora/qlora** 的 **qwen** 是否是合适选择。
- **推荐 Mistral 用于英文文本分类**：一位成员推荐使用 **Mistral** 处理英文文本任务，并建议对于像提到的 12GB VRAM 的 4070ti 这样的小显存 GPU，使用 Batch Size 为 1 的 **qlora** 可能就足够了。
- **无 Qwen 使用经验**：推荐 Mistral 的同一位成员指出，他们没有使用 **qwen** 的经验。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1223024934075236444)** (25 messages🔥): 

- **Suikamelon 分享对话建模策略**：详细介绍了一种使用特殊块字符 `▀` 和 `▄` 作为模型 Prompt 的起始和结束 Token 的格式化方法，从而实现无缝的多角色对话生成，无需区分 "user" 或 "model"。
- **Jaredquek 讨论用于细微对话训练的数据集结构**：透露了一种独特的数据集构建方法，将原始文本与 ChatML 形式的对话混合，其中包含重复内容以在不发生过拟合的情况下提高信息准确性，甚至在 Hermes 7B 等较小模型上也被证明有效。
- **训练数据中的重复性受到质疑并得到辩护**：虽然有些人不确定训练数据中逐字重复的好处，但其他人支持这种技术，并引用了一篇关于 [数据排序 (data ordering)](https://arxiv.org/abs/2310.10638) 的研究论文来支持这一做法。
- **对枯燥重复技术的怀疑**：Jaredquek 和 suikamelon 参与了一场对话，辩论了枯燥重复（将数据克隆多次以专注于同一主题）与多样化重复的有效性。
- **新方法和正在进行的实验**：Jaredquek 计划在当前技术取得满意结果后，在 Galore 等更大的模型上测试全量微调方法，尽管面临一些 Out-of-Memory 的挑战。

**Link mentioned**: <a href="https://arxiv.org/abs/2310.10638">In-Context Pretraining: Language Modeling Beyond Document Boundaries</a>：目前训练大语言模型 (LMs) 是为了在给定文档前缀的情况下预测 Token，使它们能够直接执行长文本生成和 Prompt 风格的任务，这些任务可以简化为...

  

---



**LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1222928694398418946)** (1 messages): 

- **OpenGPTs 的新讨论频道**：创建了一个新的 Discord 频道 <#1222928565117517985>，专门用于讨论与 [OpenGPTs GitHub 仓库](https://github.com/langchain-ai/opengpts) 相关的内容。该平台为社区贡献和项目协作提供了空间。

**Link mentioned**: <a href="https://github.com/langchain-ai/opengpts">GitHub - langchain-ai/opengpts</a>：通过在 GitHub 上创建账户来为 langchain-ai/opengpts 的开发做出贡献。

  

---


**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1222843915405885522)** (57 messages🔥🔥):

- **SQL AI Chatbot 记忆挑战**：一位成员正在开发 SQL AI Chatbot，但面临机器人无法记住之前消息和响应的困难。他们寻求实现 history-aware（历史感知）响应的指导。
- **产品推荐的自然语言查询**：讨论构建一个根据“计划养宠物”等自然语言查询推荐产品的机器人，方案包括使用 vector database 进行语义搜索，或提取属性并使用 SQL agent。
- **为 RAG 应用存储向量化数据**：关于在使用 PostgreSQL 数据库构建 Retrieval-Augmented Generation (RAG) AI 应用时，如何存储文档的向量化数据，以及如何处理基于用户的文档区分。
- **如何在 StreamingStdOutCallbackHandler 中使用 on_agent_finish**：围绕 LangChain 框架中 `on_agent_finish` 在 `StreamingStdOutCallbackHandler` 中的正确用法展开对话，成员们在寻求并提供澄清，尽管并非总是成功。
- **Pythia：AI 幻觉检测之神**：发布了 **Pythia**，一个用于 AI 幻觉检测的 Proof of Concept 应用，并请求关于将其集成到 LangChain 生态系统的指导。
- **对 LangChain 的质疑，寻求反驳**：一位成员呼吁对互联网上作为 FUD（恐惧、不确定和怀疑）传播的“不要使用 LangChain”情绪进行逐条反驳，建议这可以用来改进 LangChain 的路线图。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/docs/templates/rag-supabase#setup-supabase-database>)">rag_supabase | 🦜️🔗 Langchain</a>：该模板使用 Supabase 执行 RAG。</li><li><a href="https://python.langchain.com/docs/templates/rag-lantern#setup-lantern-database>)">rag_lantern | 🦜️🔗 Langchain</a>：该模板使用 Lantern 执行 RAG。</li><li><a href="https://js.langchain.com/docs/integrations/vectorstores/pgvector#usage>)">PGVector | 🦜️🔗 Langchain</a>：为了在通用的 PostgreSQL 数据库中启用向量搜索，LangChain.js 支持使用 pgvector Postgres 扩展。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1222930753562611743)** (7 messages): 

- **OpenGPTs 自定义订餐黑客**：一位成员展示了自定义订餐 API 与 OpenGPTs 的创新集成，展示了其作为平台的适应性。他们鼓励反馈，并分享了一个关于如何“Hack OpenGPT 以自动化任何事情”的 [YouTube 教程](https://youtu.be/V1SKJfE35D8)。
  
- **自动化您的代码审查流程**：发布了一个构建器，它可以创建 AI pipeline 来自动化代码验证、安全性并消除繁琐任务，同时分享了[工具链接](https://gitgud.autonoma.app)和配套的 [YouTube 解释视频](https://www.youtube.com/watch?v=6kfr1lqw2gg)。
  
- **将公司数据引入 GenAI 应用**：分享了一篇最近的[博客文章](https://www.fivetran.com/blog/building-a-chatbot-with-fivetran-and-langchain)，讨论了 Fivetran 如何协助将公司数据与 RAG 应用集成，并附带了[调查链接](https://www.surveymonkey.com/r/6X9MDJF)以获取更多信息。

- **Galaxy AI 开启顶级 AI 模型访问权限**：GalaxyAI 提供**免费**的 API 服务，可访问 **GPT-4** 和 **Gemini-PRO** 等高端 AI 模型，并兼容 OpenAI 格式以便于项目集成。感兴趣的用户可以[立即尝试](https://galaxyapi.onrender.com)。

- **深入探讨 LLM 中的模型对齐**：重点介绍了探讨模型对齐的博客文章，重点关注 RLHF、DPO 和 KTO 方法及其在 Mistral 和 Zephyr 7B 模型上的实际应用，并附带[全文链接](https://blog.premai.io/model-alignment-process/)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.premai.io/model-alignment-process/">模型对齐过程 (Model Alignment Process)</a>：生成式模型与人类反馈的对齐显著提高了自然语言生成任务的性能。对于大型语言模型 (LLMs)，像强化学习这样的对齐方法...</li><li><a href="https://producthunt.com/posts/sciphi?"> SciPhi - 为开发者提供的一键式 RAG 部署 | Product Hunt</a>：SciPhi 是一个面向开发者的云平台，简化了无服务器 RAG 流水线的构建和部署。它基于开源的 R2R 框架构建，使开发者能够专注于创新的 AI 应用...</li><li><a href="https://youtu.be/V1SKJfE35D8">破解 OpenGPT 以自动化一切</a>：欢迎来到定制化 AI 应用的未来！本演示展示了 OpenGPTs 令人惊叹的灵活性和强大功能，这是 LangChain 的一个开源项目。W...</li><li><a href="https://gitgud.autonoma.app">GitGud</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=6kfr1lqw2gg">GitGud 演示</a>：未找到描述
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1223049799926939730)** (2 条消息): 

- **数据集成遇上 AI**：分享的文章讨论了如何使用 **Fivetran** 将公司数据与 RAG 应用集成，以高效构建通用 AI 应用，减少数据管理方面的工程工作。如需更多个性化信息，欢迎通过 [SurveyMonkey 链接](https://www.surveymonkey.com/r/6X9MDJF)进行联系。
- **自定义 API 与 OpenGPTs 合并**：一位社区成员通过集成*自定义点餐 API* 展示了 **OpenGPTs** 的适应性，证明了该平台在基础用法之外的潜力。分享的 [YouTube 视频](https://youtu.be/V1SKJfE35D8)提供了视觉演示，该成员并邀请大家对这一创新应用提供反馈。

**提到的链接**：<a href="https://youtu.be/V1SKJfE35D8">破解 OpenGPT 以自动化一切</a>：欢迎来到定制化 AI 应用的未来！本演示展示了 OpenGPTs 令人惊叹的灵活性和强大功能，这是 LangChain 的一个开源项目。W...

  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1222911986984288380)** (28 条消息🔥): 

- **Jamba 介绍**：AI21 Studios 发布了 [Jamba](https://www.ai21.com/blog/announcing-jamba)，这是 Mamba 的结构化状态空间模型（Structured State Space model）与传统 Transformer 架构的混合体，拥有 256K 上下文窗口以及卓越的吞吐量和效率。该模型在同类产品中表现优异或持平，并以 Apache 2.0 许可证发布了开放权重。
  
- **Jamba 技术规格揭晓**：频道中的讨论指出了 Jamba 的规格，强调其为一个 [混合专家模型 (MoE)](https://huggingface.co/ai21labs/Jamba-v0.1)，推理时使用 12B 参数，总参数量为 52B，共包含 16 个专家，每个 token 仅激活 2 个。

- **关于模型身份的一些困惑**：一位成员指出了关于 Jamba 实际架构的困惑，质疑鉴于其混合性质和包含注意力机制，将其描述为更大规模的 "Striped Hyena" 是否比 Mamba 更准确。

- **Qwen MoE 模型能力讨论**：另一个混合模型 `Qwen1.5-MoE-A2.7B` 也被引入讨论，据称该模型仅凭 2.7B 的激活参数就能匹配 7B 模型的性能，表明其在训练成本方面具有显著效率。模型信息和资源可以在多处找到，提供的链接包括 [GitHub 仓库](https://github.com/QwenLM/Qwen1.5) 和 [Hugging Face 空间](https://huggingface.co/Qwen)。

- **共享混合架构的缩放论文**：一位成员分享了一篇[缩放论文 (scaling paper)](https://arxiv.org/abs/2403.17844)，该论文对超越 Transformer 的架构进行了广泛研究。文中指出，“条纹架构 (striped architectures)”由于层类型的专业化而表现更好。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://qwenlm.github.io/blog/qwen-moe/">Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters</a>：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 介绍：自从 Mixtral 引发关注热潮以来，混合专家模型（MoE）的研究势头强劲。研究人员和...</li><li><a href="https://x.ai/blog/grok-1.5">Announcing Grok-1.5</a>：未找到描述</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>：首次推出首个基于 Mamba 的生产级模型，提供一流的质量和性能。</li><li><a href="https://fxtwitter.com/MichaelPoli6/status/1773370168929825073?s=20">Tweet from Michael Poli (@MichaelPoli6)</a>：📢关于机械架构设计和 Scaling Laws 的新研究。- 我们进行了迄今为止针对超越 Transformer 架构的最大规模 Scaling Laws 分析（500 多个模型，最高达 7B）- 首次...</li><li><a href="https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_moe">transformers/src/transformers/models/qwen2_moe at main · huggingface/transformers</a>：🤗 Transformers：为 Pytorch、TensorFlow 和 JAX 提供的最先进机器学习库。- huggingface/transformers
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1222895903157911585)** (25 messages🔥): 

- **Microsoft GenAI 招揽新的高级研究员**：[Liliang Ren](https://x.com/liliang_ren/status/1773118751413596588?s=46) 宣布加入 Microsoft GenAI 担任高级研究员，专注于利用参数量少于 100B 的小语言模型，创建高效且可扩展的 **neural architectures**。
- **Databricks 继承 Megablocks**：正如 [Twitter](https://x.com/tgale96/status/1773342375806374307?s=46) 上宣布的那样，Megablocks 的创建者已将项目移交给 Databricks。此举预计将为 **Megablocks** 的开发和使用提供一个稳定的长期归宿。
- **辩论 AI 语言模型中“小”的规模**：关于什么构成“小语言模型”的讨论表明，100B 参数以下可能是阈值，尽管“小”这个词被认为有点名不副实。
- **OMB 指南带来的 AI 开放数据前景**：正在筛选新的关于 AI 的 **OMB guidance**，并推测政府可能会出现一些有趣的开放数据集。
- **Elon 的关注与 AI 发布**：评论指出，AI 的发布（例如 OpenAI 的发布）可能会选在 Elon Musk 可能会关注的时间点。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/tgale96/status/1773342375806374307?s=46">Tweet from Trevor Gale (@Tgale96)</a>：有些人注意到 megablocks 现在变成了 databricks/megablocks。我本周把项目交给了他们，我想不出比这更好的长期归宿了。我期待着看到它成长...</li><li><a href="https://x.com/liliang_ren/status/1773118751413596588?s=46">Tweet from Liliang Ren (@liliang_ren)</a>：个人更新：我将于今年夏天加入 Microsoft GenAI 担任高级研究员，专注于下一代既高效又可外推的 neural architectures。我们正在...
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1222899275919396894)** (5 messages): 

- **关于模型微调的澄清**：一位成员澄清了他们之前的陈述，提到他们指的是开源模型的微调和训练，而不是其他话题。
- **GitHub 上的 LLM 课程**：分享了一个 [GitHub 仓库](https://github.com/mlabonne/llm-course) 链接，提供了一门包含路线图和 Colab 笔记本的课程，以帮助个人入门 Large Language Models。
- **介绍 Jamba，一个开源模型**：AI21 Labs 推出了 **Jamba**，这是一个基于 Mamba 的模型，结合了结构化状态空间模型（SSM）与 Transformer 架构，以实现高质量和高性能。你可以在 [Hugging Face 上试用 Jamba](https://huggingface.co/ai21labs/Jamba-v0.1)。
- **模型训练数据的查询**：一位成员请求有关某个模型训练数据的更多信息，寻求获取详细见解的直接链接。
- **讨论模型的语言能力**：提到根据该模型的 model card 或附属博客文章，所讨论的模型据称是在英文数据上训练的，但指出“英文数据”的具体含义并未明确定义。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.ai21.com/jamba">介绍 Jamba</a>：一款开创性的 SSM-Transformer 开源模型</li><li><a href="https://github.com/mlabonne/llm-course">GitHub - mlabonne/llm-course：包含路线图和 Colab 笔记本的 Large Language Models (LLMs) 入门课程。</a>：包含路线图和 Colab 笔记本的 Large Language Models (LLMs) 入门课程。 - mlabonne/llm-course
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1222841255432814744)** (3 条消息): 

- **Token 插入故障排除**：一位成员指出遇到了意外的 token 插入，例如 `<dummy00002>|im_end|>`，这可能归因于 **quantization** 或所使用的 **engine**。该问题通过提供包含指定 token 的 `added_tokens.json` 文件得到了解决。

- **翻译模型大比拼**：人们对 **DiscoLM**、**Occiglot**、**Mixtral**、**GPT-4**、**DeepL** 和 **Azure Translate** 等模型的翻译输出对比很感兴趣。提议的想法包括通过每个服务翻译类似 *Capybara* 数据集的前 100 行。